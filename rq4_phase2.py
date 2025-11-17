#!/usr/bin/env python3
"""
rq4_phase2.py (ML Cohort Only)

This script performs the "Genealogy Extraction" phase for ML cohort only.
It traverses the entire commit history of ML repositories specified in the
config and outputs a CSV file containing metadata for *every*
Python file modification in every commit.

FIXES:
- Added safe attribute access with getattr() for missing PyDriller attributes
- Made DMM metrics optional (can be disabled via config)
- Added timeout protection for expensive operations
- Better error handling and logging
- MODIFIED: Only processes ML cohort, others commented out
"""

import os
import re
import sys
import subprocess
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import signal
from contextlib import contextmanager

from pydriller import Repository
import git
from git.exc import InvalidGitRepositoryError, NoSuchPathError

# ===================== CONFIG =====================
HOME = "/root/"
CLONED_ROOT = Path(f"{HOME}/satd_detection/cloned_repos").resolve()
TXT_FILE_DIR = Path(f"{HOME}/satd_detection").resolve()
OUT_DIR = Path(f"{HOME}/satd_detection/satd_work_repl/outputs").resolve()

# This script will run ONLY ML cohort
REPO_CONFIGS = {
"ml": {
         "urls_file": TXT_FILE_DIR / "ml_159.txt",
    },

}

# Output file - ML specific
OUTPUT_CSV = OUT_DIR / "genealogy_ml_modifications.csv"

ONLY_PYTHON = True
SKIP_MERGES = True
DENY_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}

# NEW: Options to handle problematic metrics
SKIP_DMM_METRICS = True  # Set to True to skip expensive DMM calculations
DMM_TIMEOUT_SECONDS = 5  # Timeout for DMM metric calculation per commit

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== TIMEOUT HANDLER =====================

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to enforce time limit on operations."""
    def signal_handler(signum, frame):
        raise TimeoutException("Operation timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# ===================== HELPERS =====================

def normalize_repo_url(u: str) -> str:
    u = u.strip()
    if not u:
        return ""
    if u.endswith(".git"):
        u = u[:-4]
    if u.startswith("git@github.com:"):
        rest = u.split("git@github.com:", 1)[1]
        u = f"https://github.com/{rest}"
    u = u.rstrip("/")
    if "github.com" in u:
        parts = u.split("github.com", 1)[1].strip("/")
        parts = parts.split("/")
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            return f"https://github.com/{owner.lower()}/{repo.lower()}"
    return u.lower()

def get_origin_url(repo_dir: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "config", "--get", "remote.origin.url"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        raw = proc.stdout.strip()
        return normalize_repo_url(raw) if raw else None
    except subprocess.CalledProcessError:
        return None

def guess_repo_path(url: str, cloned_root: Path) -> Path:
    """Find local path for cloned repository based on URL."""
    repo_name_from_url = url.strip().rstrip("/").split("/")[-1].rstrip(".git")
    
    # Try direct name match first (most common)
    direct_path = cloned_root / repo_name_from_url
    if direct_path.exists() and (direct_path / ".git").exists():
        if get_origin_url(direct_path) == normalize_repo_url(url):
             return direct_path.resolve()

    # Fallback: search for partial matches (slower)
    for item in cloned_root.iterdir():
        if item.is_dir() and repo_name_from_url.lower() in item.name.lower():
            if (item / ".git").exists():
                if get_origin_url(item) == normalize_repo_url(url):
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
        is_shallow = (repo_path / ".git" / "shallow").exists()
        if is_shallow:
            print(f"   [Git] Unshallowing {repo_path.name}...")
            subprocess.run(
                ["git", "-C", str(repo_path), "fetch", "--unshallow"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300
            )
        else:
             print(f"   [Git] Fetching {repo_path.name}...")
             subprocess.run(
                ["git", "-C", str(repo_path), "fetch", "--all", "--tags", "--prune"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300
            )
    except Exception as e:
        print(f"   [WARN] Failed to fetch full history for {repo_path.name}: {e}", file=sys.stderr)
        pass

def is_python(path: Optional[str]) -> bool:
    return (path or "").lower().endswith(".py")

def should_skip(p: Path) -> bool:
    return any(part in DENY_DIRS for part in p.parts)

def safe_get_dmm_metric(commit_obj, metric_name: str, timeout_sec: int = 5):
    """
    Safely get DMM metrics with timeout protection.
    Returns None if the metric can't be computed or times out.
    """
    if SKIP_DMM_METRICS:
        return None
    
    try:
        with time_limit(timeout_sec):
            return getattr(commit_obj, metric_name, None)
    except TimeoutException:
        return None
    except Exception:
        return None

# ===================== CORE SCAN =====================

def get_project_modifications(repo_path: Path, repo_url: str, cohort: str) -> List[Dict]:
    """
    Replicates the getProjectDiff function from SATD_genealogy.ipynb.
    Mines all modifications for a single repo.
    
    FIXED: Uses safe attribute access and optional DMM metrics.
    """
    repo_name = repo_path.name
    mod_rows = []
    
    try:
        ensure_full_history(repo_path)
    except Exception as e:
        print(f"   [ERROR] Failed during history fetch for {repo_name}: {e}", file=sys.stderr)
        return []

    kwargs = {"only_no_merge": SKIP_MERGES}
    max_c_no = 0
    
    # First pass to get commit count
    try:
        max_c_no = len(list(Repository(str(repo_path), **kwargs).traverse_commits()))
    except Exception as e:
        print(f"   [ERROR] Could not count commits for {repo_name}: {e}", file=sys.stderr)
        return []
    
    if max_c_no == 0:
        print(f"   [WARN] No commits found for {repo_name}, skipping.", file=sys.stderr)
        return []

    # Second pass to get modification data
    try:
        for c_no, c in enumerate(Repository(str(repo_path), **kwargs).traverse_commits()):
            for m in c.modified_files: 
                fpath = m.new_path or m.old_path
                if not fpath or (ONLY_PYTHON and not is_python(fpath)) or should_skip(Path(fpath)):
                    continue
                
                # Use safe attribute access for potentially missing attributes
                mod_rows.append({
                    "project": repo_name,
                    "cohort": cohort,
                    "repo_url": repo_url,
                    "c_no": c_no,
                    "c_hash": c.hash,
                    "c_msg": c.msg,
                    "c_author_email": c.author.email,
                    "c_committer_email": c.committer.email,
                    "c_committer_date": pd.to_datetime(c.committer_date, utc=True),
                    "c_branches": str(c.branches),
                    "c_in_main_branch": c.in_main_branch,
                    "c_merge": c.merge,
                    "c_deletions": c.deletions,
                    "c_insertions": c.insertions,
                    "c_lines": c.lines,
                    "c_files": c.files,
                    # DMM metrics with timeout protection
                    "c_dmm_unit_size": safe_get_dmm_metric(c, "dmm_unit_size", DMM_TIMEOUT_SECONDS),
                    "c_dmm_unit_complexity": safe_get_dmm_metric(c, "dmm_unit_complexity", DMM_TIMEOUT_SECONDS),
                    "c_dmm_unit_interfacing": safe_get_dmm_metric(c, "dmm_unit_interfacing", DMM_TIMEOUT_SECONDS),
                    # Modification metrics with safe defaults
                    "m_diff": getattr(m, "diff", None),
                    "m_added": getattr(m, "added_lines", getattr(m, "added", None)),  # Try both attribute names
                    "m_removed": getattr(m, "deleted_lines", getattr(m, "removed", None)),  # Try both attribute names
                    "m_filename": m.filename,
                    "m_change_type": m.change_type.name if hasattr(m.change_type, "name") else str(m.change_type),
                    "m_changed_methods": str(list(getattr(m, "changed_methods", []))),
                    "m_nloc": getattr(m, "nloc", None),
                    "m_complexity": getattr(m, "complexity", None),
                    "m_token_count": getattr(m, "token_count", None),
                    "m_old_path": m.old_path,
                    "m_new_path": m.new_path,
                    "c_freq": (c_no + 1) / max_c_no if max_c_no > 0 else 0
                })
    except KeyboardInterrupt:
        print(f"\n   [INTERRUPT] User cancelled processing of {repo_name}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"   [ERROR] Failed to traverse {repo_name}: {e}", file=sys.stderr)
    
    return mod_rows

# ===================== MAIN =====================

# ===================== MAIN =====================

def main():
    print("Starting Phase 2: Genealogy Extraction (ML COHORT ONLY)")
    print(f"Cloned Repos Location: {CLONED_ROOT}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"DMM Metrics: {'DISABLED' if SKIP_DMM_METRICS else 'ENABLED'}")
    
    # --- MODIFIED: Define GZ path early and check for existing file ---
    OUTPUT_CSV_GZ = OUTPUT_CSV.with_suffix(".csv.gz")
    processed_repos = set()
    
    if OUTPUT_CSV_GZ.exists():
        print(f"\n[RESUME MODE] Found existing output file: {OUTPUT_CSV_GZ}")
        try:
            # Load only the 'repo_url' column to save memory
            existing_df = pd.read_csv(OUTPUT_CSV_GZ, usecols=['repo_url'])
            processed_repos.update(existing_df['repo_url'].unique())
            print(f"   Found {len(processed_repos)} repos already processed. They will be skipped.")
        except Exception as e:
            print(f"   [WARN] Could not read existing file: {e}. Will run from scratch.")
            processed_repos.clear()
    
    print("="*60)

    all_mod_rows = []
    
    for cohort, cfg in REPO_CONFIGS.items():
        print(f"\nProcessing Cohort: {cohort.upper()}")
        urls_file = Path(cfg["urls_file"])
        
        if not urls_file.exists():
            print(f"   [WARN] URL file not found, skipping: {urls_file}", file=sys.stderr)
            continue
            
        with open(urls_file, "r", encoding="utf-8") as f:
            urls = [u.strip() for u in f if u.strip() and not u.strip().startswith("#")]
        
        url_allowlist = {normalize_repo_url(u) for u in urls if normalize_repo_url(u)}
        print(f"   Found {len(url_allowlist)} unique URLs for this cohort.")

        repo_paths = []
        for url in url_allowlist:
            try:
                repo_path = guess_repo_path(url, CLONED_ROOT)
                if is_valid_git_repo(repo_path):
                    repo_paths.append((repo_path, url))
                else:
                    print(f"   [WARN] Invalid git repo, skipping: {repo_path.name}", file=sys.stderr)
            except FileNotFoundError:
                print(f"   [WARN] Repo not found locally, skipping: {url}", file=sys.stderr)
        
        # --- MODIFIED: Filter out repos that are already processed ---
        original_count = len(repo_paths)
        if processed_repos:
            repo_paths = [(path, url) for path, url in repo_paths if url not in processed_repos]
            skipped_count = original_count - len(repo_paths)
            print(f"   Skipping {skipped_count} repos already in output file.")
        
        print(f"   Found {len(repo_paths)} remaining repos to process.")
        
        # Use tqdm for a progress bar
        for repo_path, repo_url in tqdm(repo_paths, desc=f"Scanning {cohort}"):
            try:
                repo_mods = get_project_modifications(repo_path, repo_url, cohort)
                all_mod_rows.extend(repo_mods)
            except KeyboardInterrupt:
                print("\n\n[USER CANCELLED] Saving partial results...")
                break # Break from tqdm loop
            except Exception as e:
                print(f"   [FATAL] Unhandled error in {repo_path.name}: {e}", file=sys.stderr)
        
        # --- ADDED: If user cancelled, break from the cohort loop too ---
        if not all_mod_rows and not processed_repos:
            # This handles case where user cancels on the *first* repo
            print("   No new modifications recorded.")
            continue


    print("\n" + "="*60)
    if not all_mod_rows:
        print("No *new* modifications found. Exiting.")
        if processed_repos:
             print(f"Total results (including previous) are in: {OUTPUT_CSV_GZ}")
        return

    print("Combining *new* modification data...")
    master_df = pd.DataFrame(all_mod_rows)
    
    print(f"Total *new* modifications mined: {len(master_df):,}")
    print(f"Total *new* projects processed: {master_df['project'].nunique():,}")
    
    # --- MODIFIED: Append to existing file or create new one ---
    file_exists = OUTPUT_CSV_GZ.exists()
    print(f"Saving to {OUTPUT_CSV_GZ} (mode: {'append' if file_exists else 'write'})...")
    
    master_df.to_csv(
        OUTPUT_CSV_GZ, 
        index=False, 
        compression="gzip",
        mode='a', # 'a' for append
        header=not file_exists # Write header only if file is new
    )
    
    print("\nPhase 2 (Genealogy Extraction) complete for ML cohort.")
    print(f"Output file is: {OUTPUT_CSV_GZ}")
    print("You can now run Phase 3 (SATD Labeling) on this file.")
    print("\nTo process other cohorts later, uncomment the respective")
    print("entries in REPO_CONFIGS and update OUTPUT_CSV filename.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Script terminated by user.")
        sys.exit(1)
