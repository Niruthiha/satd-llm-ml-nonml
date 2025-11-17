#!/usr/bin/env python3
"""
SATD Detection Script (nonML Cohort)
=================================
This script processes the nonML GitHub repositories to detect SATD.
"""
import csv
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

SHEBANG_RE = re.compile(r"^#!")
SKIP_DIRS = ("site-packages", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", "__pycache__")

# JAVA PATH - Hardcoded for your system
JAVA_PATH = "/usr/bin/java"

# ---------------------------
# URL normalization & selection
# ---------------------------
def normalize_repo_url(u: str) -> str:
    u = u.strip()
    if not u: return ""
    if u.endswith(".git"): u = u[:-4]
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

def get_repo_name_from_url(url: str) -> str:
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-1]
    return None

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

def clone_repository(repo_url: str, target_dir: Path, timeout: int = 120) -> tuple[bool, str]:
    repo_name = get_repo_name_from_url(repo_url)
    if not repo_name:
        print(f"   ERROR: Could not extract repo name from {repo_url}")
        return False, "failed"
    repo_path = target_dir / repo_name
    if repo_path.exists() and (repo_path / ".git").exists():
        origin = get_origin_url(repo_path)
        if origin and normalize_repo_url(origin) == normalize_repo_url(repo_url):
            return True, "exists"
        else:
            print(f"   WARNING: Directory {repo_name} exists but has different origin")
            return False, "failed"
    
    print(f"   Cloning: {repo_name} from {repo_url}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            capture_output=True, text=True, check=True, timeout=timeout
        )
        print(f"   ✓ Successfully cloned: {repo_name}")
        return True, "cloned"
    except Exception as e:
        print(f"   ✗ Failed to clone {repo_name}: {e}")
        return False, "failed"

def load_url_allowlist(txt_path: Path) -> set[str]:
    with open(txt_path, "r", encoding="utf-8") as fh:
        urls = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    return {normalize_repo_url(u) for u in urls if normalize_repo_url(u)}

def find_matched_repos(repos_root: Path, allowlist: set[str]) -> list[tuple[str, Path]]:
    matched = []
    for repo_url in allowlist:
        repo_name = get_repo_name_from_url(repo_url)
        if not repo_name:
            continue
        repo_path = repos_root / repo_name
        if repo_path.exists() and (repo_path / ".git").exists():
            origin = get_origin_url(repo_path)
            if origin and normalize_repo_url(origin) == normalize_repo_url(repo_url):
                matched.append((repo_url, repo_path))
    return matched

# ---------------------------
# Comment extraction
# ---------------------------
def extract_all_python_comments(repo_path: Path, repo_id: str):
    for py in repo_path.rglob("*.py"):
        sp = str(py)
        if any(sd in sp for sd in SKIP_DIRS):
            continue
        try:
            with open(py, "r", encoding="utf-8", errors="ignore") as f:
                for ln, line in enumerate(f, 1):
                    if "#" not in line:
                        continue
                    if ln == 1 and SHEBANG_RE.match(line):
                        continue
                    try:
                        comment = line.split("#", 1)[1].strip()
                    except Exception:
                        continue
                    if not comment:
                        continue
                    if re.fullmatch(r"https?://\S+", comment):
                        continue
                    yield (str(uuid.uuid4()), repo_id, str(py.relative_to(repo_path)), ln, comment)
        except Exception:
            continue

# ---------------------------
# REPL runner (interactive JAR)
# ---------------------------
def classify_with_repl(jar_path: Path, comments_iter):
    proc = subprocess.Popen(
        [JAVA_PATH, 
         "--add-opens", "java.base/java.lang=ALL-UNNAMED",
         "--add-opens", "java.base/java.util=ALL-UNNAMED",
         "--add-opens", "java.base/java.io=ALL-UNNAMED",
         "-jar", str(jar_path), "test"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, encoding='utf-8', errors='ignore'
    )
    def read_label():
        while True:
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                stderr_output = proc.stderr.read()
                raise RuntimeError(f"JAR exited unexpectedly. Stderr: {stderr_output}")
            s = line.strip()
            if s.startswith(">"): s = s[1:].strip()
            if not s or s == ">": continue
            s_lower = s.lower()
            if "satd" in s_lower:
                return "Not SATD" if "not" in s_lower else "SATD"
    try:
        for row in comments_iter:
            (_, _, _, _, text) = row
            if proc.poll() is not None:
                stderr_output = proc.stderr.read()
                raise RuntimeError(f"JAR process terminated unexpectedly. Stderr: {stderr_output}")
            proc.stdin.write(text + "\n")
            proc.stdin.flush()
            label = read_label()
            yield (row, label)
    finally:
        try:
            if proc.poll() is None:
                proc.stdin.write("/exit\n")
                proc.stdin.flush()
            proc.communicate(timeout=2)
        except Exception:
            proc.kill()

# ---------------------------
# Prevalence
# ---------------------------
def prevalence_from_predictions(tsv_path: Path, label_col: str = "label") -> dict:
    tot = satd = 0
    with open(tsv_path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            tot += 1
            if str(row[label_col]).strip().upper() == "SATD":
                satd += 1
    pct = 100.0 * satd / tot if tot else 0.0
    return {"satd": satd, "total": tot, "prevalence_pct": pct}

# ---------------------------
# Main
# ---------------------------
def main():
    # --- nonML COHORT CONFIG ---
    repo_list_path = Path("/root/satd_detection/nonml_159.txt").resolve()
    repos_root = Path("/root/satd_detection/cloned_repos").resolve()
    jar_path = Path("/root/satd_detection/satd_detector.jar").resolve()
    workdir = Path("./RQ1").resolve()
    cohort = "nonml"
    # --------------------------
    
    auto_clone = True
    clone_timeout = 120
    
    outputs_dir = workdir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== CONFIGURATION (nonML Cohort) ===")
    print(f"Repo list:  {repo_list_path}")
    print(f"Repos root: {repos_root}")
    print(f"JAR path:   {jar_path}")
    print(f"Java path:  {JAVA_PATH}")
    print(f"Cohort:     {cohort}")
    print(f"Auto-clone: {auto_clone}\n")

    # --- Validations ---
    if not Path(JAVA_PATH).exists():
        print(f"ERROR: Java not found at {JAVA_PATH}", file=sys.stderr)
        sys.exit(1)
    if not repo_list_path.exists():
        print(f"ERROR: Repo list file not found: {repo_list_path}", file=sys.stderr)
        sys.exit(1)
    if not jar_path.exists():
        print(f"ERROR: JAR file not found: {jar_path}", file=sys.stderr)
        sys.exit(1)
    repos_root.mkdir(parents=True, exist_ok=True)

    # --- Cloning ---
    allowlist = load_url_allowlist(repo_list_path)
    print(f"Loaded {len(allowlist)} allowed repositories.")
    if auto_clone:
        print("\nChecking and cloning missing repositories...")
        newly_cloned = already_exists = failed_count = 0
        for i, repo_url in enumerate(allowlist, 1):
            success, status = clone_repository(repo_url, repos_root, timeout=clone_timeout)
            if status == "cloned": newly_cloned += 1
            elif status == "exists": already_exists += 1
            else: failed_count += 1
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(allowlist)} repositories processed...")
        print(f"\nRepository summary: {already_exists} existed, {newly_cloned} cloned, {failed_count} failed.")
    
    # --- Find Repos ---
    matched = find_matched_repos(repos_root, allowlist)
    print(f"Found {len(matched)} repositories from your list ready for analysis")
    if not matched:
        print("ERROR: No repositories found to analyze.", file=sys.stderr)
        sys.exit(1)
    
    # --- Run Detection ---
    preds_tsv = outputs_dir / f"{cohort}_predictions.tsv"
    with open(preds_tsv, "w", encoding="utf-8", newline="") as out_f:
        w = csv.writer(out_f, delimiter="\t")
        w.writerow(["id", "repo", "path", "line", "text", "label"])
        
        def iter_all_comments():
            for origin, repo_path in matched:
                repo_id = origin
                print(f"Extracting comments from {repo_path.name}")
                yield from extract_all_python_comments(repo_path, repo_id)

        comment_count = 0
        satd_count = 0
        for (row, label) in classify_with_repl(jar_path, iter_all_comments()):
            w.writerow([*row, label])
            comment_count += 1
            if label == "SATD":
                satd_count += 1
            if comment_count % 100 == 0:
                print(f"   Processed {comment_count} comments ({satd_count} SATD so far)...")
    
    # --- Summarize ---
    stats = prevalence_from_predictions(preds_tsv, label_col="label")
    summary = {
        "cohort": cohort,
        "n_repos_processed": len(matched),
        "stats": stats,
        "predictions_tsv": str(preds_tsv),
    }
    summary_path = outputs_dir / f"{cohort}_prevalence_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PREVALENCE SUMMARY (nonML) ===")
    print(f"Repos processed: {len(matched)}")
    print(f"Total comments:  {stats['total']}")
    print(f"SATD comments:   {stats['satd']}")
    print(f"Prevalence:      {stats['prevalence_pct']:.3f}%")
    print(f"\nFiles:\n   Predictions TSV: {preds_tsv}\n   Summary JSON:    {summary_path}")

if __name__ == "__main__":
    main()