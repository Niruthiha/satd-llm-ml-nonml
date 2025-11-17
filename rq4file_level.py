#!/usr/bin/env python3
"""

rq4file_level.py —  File-level SATD survival (paper-faithful) with robust git handling

- Introduction (per FILE):
    time_days = first SATD date in the file - file creation date
    event_observed = 1 if file ever gets SATD, else 0 (censored at repo last commit)

- Removal (per FILE, first SATD only):
    time_days = removal date - first SATD date
    event_observed = 1 if removed, else 0 (censored at repo last commit)

- Robustness:
    * Skips invalid/corrupted repos (no .git)
    * Tries to unshallow/fetch full history to avoid git 128
    * Continues past traversal errors

Requires:
    pip install pydriller lifelines gitpython pandas matplotlib
"""

from __future__ import annotations
import os
import re
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from pydriller import Repository
import git
from git.exc import InvalidGitRepositoryError, NoSuchPathError
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
CLONED_ROOT = "/root/satd_detection/cloned_repos"
SATD_OUTPUT_DIR = "/root/satd_detection/RQ1/outputs"
OUT_DIR = "/root/satd_detection/file_level_survival"

REPO_CONFIGS = {
    "ml": {
        "urls_file": "/root/satd_detection/ml_159.txt",
        "predictions": f"{SATD_OUTPUT_DIR}/ml_predictions.tsv",
    },
    "nonml": {
        "urls_file": "/root/satd_detection/nonml_159.txt",
        "predictions": f"{SATD_OUTPUT_DIR}/nonml_predictions.tsv",
    },
     "llm": {
         "urls_file": "/root/satd_detection/llm_159.txt",
         "predictions": f"{SATD_OUTPUT_DIR}/llm_predictions.tsv",
     },
}

ONLY_PYTHON = True
SKIP_MERGES = True
DENY_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}

os.makedirs(OUT_DIR, exist_ok=True)

# ===================== HELPERS =====================

def norm_text(x) -> str:
    """Normalize classifier & diff lines (safe for NaN)."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x)
    s = re.sub(r"\s+", " ", s.strip())
    return s

def is_comment_line(line: str) -> bool:
    """Cross-language-ish comment detection (conservative)."""
    if line is None:
        return False
    s = str(line)
    if re.match(r"^\s*#", s):
        return True
    if '"""' in s or "'''" in s:
        return True
    if re.match(r"^\s*//", s) or re.match(r"^\s*/\*", s) or re.search(r"\*/\s*$", s):
        return True
    return False

def is_python(path_before: Optional[str], path_after: Optional[str]) -> bool:
    p = (path_after or path_before or "").lower()
    return p.endswith(".py")

def should_skip(p: Path) -> bool:
    return any(part in DENY_DIRS for part in p.parts)

def guess_repo_path(url: str, cloned_root: str) -> Path:
    base = Path(cloned_root)
    u = url.strip().rstrip("/").rstrip(".git")
    try:
        owner, name = u.split("github.com/")[1].split("/")[:2]
    except Exception as e:
        raise ValueError(f"Unexpected URL format: {url}") from e

    candidates = [
        base / f"{owner}__{name}",
        base / f"{owner}_{name}",
        base / f"{owner}-{name}",
        base / owner / name,
        base / name,
        base / name.lower(),
        base / name.replace('-', '_'),
        base / name.replace('_', '-'),
    ]
    for c in candidates:
        if c.exists() and (c / ".git").exists():
            return c.resolve()

    for item in base.iterdir():
        if item.is_dir() and name.lower() in item.name.lower() and (item / ".git").exists():
            return item.resolve()

    raise FileNotFoundError(f"Could not find local clone for {url} under {cloned_root}")

def is_valid_git_repo(repo_path: Path) -> bool:
    try:
        if (repo_path / ".git").exists():
            return True
        _ = git.Repo(str(repo_path))
        return True
    except (InvalidGitRepositoryError, NoSuchPathError, Exception):
        return False

def ensure_full_history(repo_path: Path):
    """If repo is shallow/missing objects, fetch full history to avoid git 128."""
    if not is_valid_git_repo(repo_path):
        return
    try:
        subprocess.run(
            ["git", "-C", str(repo_path), "fetch", "--all", "--tags", "--prune", "--unshallow"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "fetch", "--all", "--tags", "--prune"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass

def repo_last_commit_dt(repo_path: Path) -> Optional[pd.Timestamp]:
    try:
        gp = git.Repo(str(repo_path))
        return pd.to_datetime(gp.head.commit.committed_datetime, utc=True)
    except (InvalidGitRepositoryError, NoSuchPathError) as e:
        print(f"[WARN] not a valid git repo: {repo_path} ({e})", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[WARN] failed to read last commit for {repo_path}: {e}", file=sys.stderr)
        return None

def file_creation_dt(repo_path: Path, file_rel: str) -> Optional[pd.Timestamp]:
    """
    First time file appeared in history (true birth).
    Uses `git log --diff-filter=A --follow --format=%ct -- <file>`.
    """
    try:
        cmd = ["git", "-C", str(repo_path), "log", "--diff-filter=A", "--follow",
               "--format=%ct", "--", file_rel]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if not out:
            return None
        ts = min(int(line) for line in out.splitlines() if line.strip())
        return pd.to_datetime(pd.Timestamp.utcfromtimestamp(ts), utc=True)
    except subprocess.CalledProcessError:
        return None
    except Exception:
        return None

def load_satd_predictions(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    text_col = None
    for cand in ["text", "comment_text", "comment", "content"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        raise KeyError(f"Couldn't find a text column in {tsv_path}. "
                       f"Expected one of: text, comment_text, comment, content. "
                       f"Got: {list(df.columns)}")
    if "label" not in df.columns:
        raise KeyError(f"'label' column not found in {tsv_path}")

    df["text_norm"] = df[text_col].map(norm_text)
    df = df[df["text_norm"] != ""].copy()
    df["isSATD"] = df["label"].str.contains("SATD", case=False, na=False)
    df = df.drop_duplicates(subset=["text_norm"], keep="first")
    return df[["text_norm", "isSATD", "label"]].reset_index(drop=True)

# ===================== DATA STRUCTS =====================

@dataclass
class FileTimeline:
    created_at: Optional[pd.Timestamp] = None
    first_satd_at: Optional[pd.Timestamp] = None
    first_satd_text: Optional[str] = None
    first_satd_removed_at: Optional[pd.Timestamp] = None

@dataclass
class RepoTimeline:
    files: Dict[str, FileTimeline] = field(default_factory=lambda: defaultdict(FileTimeline))
    repo_last_commit: Optional[pd.Timestamp] = None

# ===================== CORE SCAN =====================

def build_repo_timelines(repo_path: Path, satd_lookup: Dict[str, bool]) -> RepoTimeline:
    """
    Build per-file timelines:
      - file creation date (true birth)
      - first SATD introduction date (match added comment lines to isSATD via normalized text)
      - first removal date for that same text
      - repo last commit date
    """
    ensure_full_history(repo_path)
    rt = RepoTimeline()
    rt.repo_last_commit = repo_last_commit_dt(repo_path)
    if rt.repo_last_commit is None:
        return rt  # caller will skip

    # <-- FIX 1: Use order='reverse' to process from oldest to newest
    kwargs = {"only_no_merge": SKIP_MERGES, "order": "reverse"}
    try:
        for commit in Repository(str(repo_path), **kwargs).traverse_commits():
            cdate = pd.to_datetime(commit.committer_date, utc=True)
            for mf in commit.modified_files:

                # <-- FIX 2: Handle file renames
                if mf.new_path and mf.old_path and mf.new_path != mf.old_path:
                    if mf.old_path in rt.files:
                        # Move the history (FileTimeline object) to the new path
                        rt.files[mf.new_path] = rt.files.pop(mf.old_path)
                
                fpath = mf.new_path or mf.old_path
                if not fpath:
                    continue
                if ONLY_PYTHON and not is_python(mf.old_path, mf.new_path):
                    continue
                if should_skip(Path(fpath)):
                    continue

                ft = rt.files[fpath]

                # <-- FIX 3: Robust file creation date (remove 'else cdate' fallback)
                if ft.created_at is None:
                    ft.created_at = file_creation_dt(repo_path, fpath)
                    # If birth is still None, we leave it None.
                    # The 'else cdate' (which would be the *first* commit) 
                    # is still less accurate than the git log command.

                # Parse diff safely
                try:
                    dp = mf.diff_parsed or {}
                    added = dp.get("added", [])
                    deleted = dp.get("deleted", [])
                except Exception:
                    added, deleted = [], []

                # INTRODUCTION: first added SATD comment in this file
                # (This logic is now correct because we iterate oldest-to-newest)
                if ft.first_satd_at is None:
                    for _, text in added:
                        if is_comment_line(text) and satd_lookup.get(norm_text(text), False):
                            ft.first_satd_at = cdate
                            ft.first_satd_text = norm_text(text)
                            break

                # REMOVAL: detect deletion of that exact first SATD text
                if ft.first_satd_at is not None and ft.first_satd_removed_at is None and ft.first_satd_text:
                    for _, text in deleted:
                        if is_comment_line(text) and norm_text(text) == ft.first_satd_text:
                            if cdate >= ft.first_satd_at:
                                ft.first_satd_removed_at = cdate
                                break
    except Exception as e:
        print(f"[WARN] failed to traverse {repo_path.name}: {e}", file=sys.stderr)

    return rt

# ===================== COHORT BUILD =====================

def build_cohort(cohort: str, urls_file: str, predictions_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]

    # predictions → lookup
    satd_df = load_satd_predictions(predictions_file)
    satd_lookup = {row["text_norm"]: bool(row["isSATD"]) for _, row in satd_df.iterrows()}

    intro_rows: List[Dict] = []
    removal_rows: List[Dict] = []

    for url in urls:
        try:
            repo_path = guess_repo_path(url, CLONED_ROOT)
        except Exception as e:
            print(f"[WARN] {cohort}: cannot locate repo for {url}: {e}", file=sys.stderr)
            continue

        print(f"[INFO] {cohort}: scanning {repo_path.name}...")

        if not is_valid_git_repo(repo_path):
            print(f"[WARN] {cohort}: {repo_path} is not a valid git repo; skipping.", file=sys.stderr)
            continue

        rt = build_repo_timelines(repo_path, satd_lookup)
        if rt.repo_last_commit is None:
            print(f"[WARN] {cohort}: {repo_path.name} has no last commit; skipping.", file=sys.stderr)
            continue

        for fpath, ft in rt.files.items():
            if ft.created_at is None:
                # Skip file if we don't know its creation date (Fix 3)
                continue

            # ---- Introduction row (one per file)
            if ft.first_satd_at is not None:
                t = (ft.first_satd_at - ft.created_at).days
                intro_rows.append({
                    "cohort": cohort,
                    "repo": repo_path.name,
                    "file": fpath,
                    "time_days": max(int(t), 0),
                    "event_observed": 1,  # file did get SATD
                })
            else:
                t = (rt.repo_last_commit - ft.created_at).days
                intro_rows.append({
                    "cohort": cohort,
                    "repo": repo_path.name,
                    "file": fpath,
                    "time_days": max(int(t), 0),
                    "event_observed": 0,  # censored (no SATD)
                })

            # ---- Removal row (only for files that got SATD; first SATD only)
            if ft.first_satd_at is not None:
                if ft.first_satd_removed_at is not None:
                    t_rem = (ft.first_satd_removed_at - ft.first_satd_at).days
                    removal_rows.append({
                        "cohort": cohort,
                        "repo": repo_path.name,
                        "file": fpath,
                        "time_days": max(int(t_rem), 0),
                        "event_observed": 1,  # removed
                    })
                else:
                    t_rem = (rt.repo_last_commit - ft.first_satd_at).days
                    removal_rows.append({
                        "cohort": cohort,
                        "repo": repo_path.name,
                        "file": fpath,
                        "time_days": max(int(t_rem), 0),
                        "event_observed": 0,  # censored (still present)
                    })

    intro_df = pd.DataFrame(intro_rows)
    removal_df = pd.DataFrame(removal_rows)

    intro_path = f"{OUT_DIR}/intro_{cohort}_files.csv"
    removal_path = f"{OUT_DIR}/removal_{cohort}_files.csv"
    intro_df.to_csv(intro_path, index=False)
    removal_df.to_csv(removal_path, index=False)

    print(f"[OK] {cohort}: {intro_path} ({len(intro_df):,} files)")
    print(f"[OK] {cohort}: {removal_path} ({len(removal_df):,} SATD instances)")

    return intro_df, removal_df

# ===================== PLOTS & SUMMARY =====================

def plot_km(intro_by_cohort: Dict[str, pd.DataFrame],
            removal_by_cohort: Dict[str, pd.DataFrame]) -> None:
    kmf = KaplanMeierFitter()

    # Introduction
    fig, ax = plt.subplots(figsize=(11, 7))
    for cohort, df in intro_by_cohort.items():
        if df.empty:
            continue
        kmf.fit(durations=df["time_days"], event_observed=df["event_observed"], label=cohort.upper())
        kmf.plot(ci_show=True, ax=ax)
        print(f"[STATS] {cohort.upper()} Intro: files={len(df):,}, "
              f"events={int(df['event_observed'].sum()):,}, "
              f"median={kmf.median_survival_time_} days")
    ax.set_xlabel("Days from FILE creation")
    ax.set_ylabel("Probability file is SATD-free")
    ax.set_title("SATD Introduction (per file)")
    ax.axvline(365, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/km_intro.png", dpi=300)
    plt.close()

    # Removal
    fig, ax = plt.subplots(figsize=(11, 7))
    for cohort, df in removal_by_cohort.items():
        if df.empty:
            continue
        kmf.fit(durations=df["time_days"], event_observed=df["event_observed"], label=cohort.upper())
        kmf.plot(ci_show=True, ax=ax)
        print(f"[STATS] {cohort.upper()} Removal: satd_files={len(df):,}, "
              f"removed={int(df['event_observed'].sum()):,}, "
              f"median={kmf.median_survival_time_} days")
    ax.set_xlabel("Days from FIRST SATD in file")
    ax.set_ylabel("Probability SATD still present")
    ax.set_title("SATD Removal (per file, first SATD only)")
    ax.axvline(365, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/km_removal.png", dpi=300)
    plt.close()

# ===================== MAIN =====================

def main():
    intro_by = {}
    removal_by = {}

    for cohort, cfg in REPO_CONFIGS.items():
        intro_df, removal_df = build_cohort(
            cohort=cohort,
            urls_file=cfg["urls_file"],
            predictions_file=cfg["predictions"]
        )
        intro_by[cohort] = intro_df
        removal_by[cohort] = removal_df

    plot_km(intro_by, removal_by)

if __name__ == "__main__":
    main()
