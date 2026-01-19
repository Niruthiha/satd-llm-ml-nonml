#!/usr/bin/env python3
"""
rq4_phase3.py (ML Cohort Only)

This script performs the "SATD Labeling" phase (like SATD_RQ1_RQ2.ipynb).
It consumes the large CSV from Phase 2 (genealogy_ml_modifications.csv.gz),
parses the 'm_diff' column to extract added/removed comments, and runs
the Java SATD detector on them.

It outputs a new, labeled CSV ready for the final survival analysis (Phase 4).
"""

import os
import re
import sys
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import uuid
import csv

# Enable progress bars for pandas `apply`
tqdm.pandas(desc="Parsing comments")

# ===================== CONFIG =====================
JAVA_PATH = "/usr/bin/java"
JAR_PATH = Path("/root/satd_detection/satd_detector.jar").resolve()
IN_DIR = Path("/root/satd_detection/satd_work_repl/outputs").resolve()
OUT_DIR = IN_DIR # Save outputs to the same directory

# Input file from Phase 2
INPUT_CSV_GZ = IN_DIR / "genealogy_ml_modifications.csv.gz"

# Output file for Phase 4
OUTPUT_LABELED_CSV_GZ = OUT_DIR / "genealogy_ml_labeled.csv.gz"

# --- THIS IS THE FIX ---
# Define ALL columns to keep from Phase 2
ALL_FEATURE_COLUMNS = [
    "project", "cohort", "repo_url", "c_no", "c_hash", "c_msg",
    "c_author_email", "c_committer_email", "c_committer_date",
    "c_branches", "c_in_main_branch", "c_merge",
    "c_deletions", "c_insertions", "c_lines", "c_files",
    "c_dmm_unit_size", "c_dmm_unit_complexity", "c_dmm_unit_interfacing",
    "m_filename", "m_change_type", "m_changed_methods",
    "m_nloc", "m_complexity", "m_token_count",
    "m_old_path", "m_new_path", "c_freq"
]

# Define columns for the new labeled file
OUTPUT_COLUMNS = [
    "comment_id", "comment_text", "comment_type", "label",
] + ALL_FEATURE_COLUMNS
# ---------------------

# ===================== COMMENT PARSING HELPERS =====================
def CommentParser(data: str) -> List[str]:
    comments = []
    mul = False
    if not isinstance(data, str): return comments
    for eachLine in data.split('\n'):
        eachLine = eachLine.lstrip()        
        if mul:
            comments.append(eachLine)
            if eachLine.endswith("'''") or eachLine.endswith('"""'):
                mul = False
            continue
        if eachLine:
            if eachLine.startswith('#'):
                comments.append(eachLine)
            elif eachLine.startswith("'''") or eachLine.startswith('"""'):
                mul = True
                comments.append(eachLine)
                if (eachLine.count('"""') == 2 and eachLine.endswith('"""')) or \
                   (eachLine.count("'''") == 2 and eachLine.endswith("'''")):
                    mul = False
    return comments

def getAdd(str_: str) -> str:
    add = []
    if isinstance(str_, str):
        for eachLine in str_.split('\n'):
            if eachLine.startswith('+'):
                add.append(eachLine[1:])
    return '\n'.join(add)

def getSub(str_: str) -> str:
    add = []
    if isinstance(str_, str):    
        for eachLine in str_.split('\n'):
            if eachLine.startswith('-'):
                add.append(eachLine[1:])
    return '\n'.join(add)

def normalize(comment: str) -> str:
    if not isinstance(comment, str): return ""
    comment = comment.replace(',', ' ')
    comment = re.sub(r'\W+',' ', comment)
    comment = re.sub(r'\s+', ' ', comment.strip())
    return comment

# ===================== REPL RUNNER =====================
def classify_with_repl(jar_path: Path, comments_iter):
    print("   [Java] Starting SATD detector JAR...")
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
        count = 0
        satd_count = 0
        for comment_id, comment_text in comments_iter:
            if proc.poll() is not None:
                stderr_output = proc.stderr.read()
                raise RuntimeError(f"JAR process terminated unexpectedly. Stderr: {stderr_output}")
            
            proc.stdin.write(comment_text + "\n")
            proc.stdin.flush()
            label = read_label()
            yield (comment_id, label)
            count += 1
            if label == "SATD":
                satd_count += 1
            if count % 1000 == 0:
                print(f"   [Java] Processed {count:,} comments ({satd_count:,} SATD)...")
    finally:
        print("   [Java] Shutting down JAR...")
        try:
            if proc.poll() is None:
                proc.stdin.write("/exit\n")
                proc.stdin.flush()
            proc.communicate(timeout=5)
        except Exception:
            proc.kill()

# ===================== MAIN =====================
def main():
    print("Starting Phase 3 (Corrected): SATD Labeling with Features (NONML COHORT)")
    print(f"Input file: {INPUT_CSV_GZ}")
    print(f"Output file: {OUTPUT_LABELED_CSV_GZ}")
    print("="*60)

    print(f"[1/4] Loading {INPUT_CSV_GZ}...")
    try:
        df = pd.read_csv(INPUT_CSV_GZ, compression="gzip")
    except Exception as e:
        print(f"   [ERROR] Failed to read file: {e}", file=sys.stderr)
        print("   Did Phase 2 complete successfully?", file=sys.stderr)
        sys.exit(1)
    
    print(f"   Loaded {len(df):,} modifications.")
    
    # --- Check that all feature columns exist ---
    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            print(f"   [WARN] Missing expected column: {col}. Will fill with None.", file=sys.stderr)
            df[col] = None
    
    # Also check for m_diff
    if 'm_diff' not in df.columns:
         print(f"   [ERROR] Critical column 'm_diff' not found. Cannot parse comments.", file=sys.stderr)
         sys.exit(1)

    print("[2/4] Parsing diffs to find added/removed comments...")
    df['additions'] = df['m_diff'].progress_apply(getAdd)
    df['subtractions'] = df['m_diff'].progress_apply(getSub)
    df['add_comments'] = df['additions'].progress_apply(CommentParser)
    df['sub_comments'] = df['subtractions'].progress_apply(CommentParser)
    print("   Parsing complete.")

    print("[3/4] Exploding data to 1 row per comment...")
    
    # Process added comments
    df_add = df[df['add_comments'].str.len() > 0].copy()
    df_add = df_add.explode('add_comments')
    df_add = df_add.rename(columns={"add_comments": "comment_text"})
    df_add["comment_type"] = "added"
    
    # Process removed comments
    df_sub = df[df['sub_comments'].str.len() > 0].copy()
    df_sub = df_sub.explode('sub_comments')
    df_sub = df_sub.rename(columns={"sub_comments": "comment_text"})
    df_sub["comment_type"] = "removed"
    
    # Combine
    all_comments_df = pd.concat([
        df_add[ALL_FEATURE_COLUMNS + ["comment_text", "comment_type"]],
        df_sub[ALL_FEATURE_COLUMNS + ["comment_text", "comment_type"]]
    ], ignore_index=True)
    
    all_comments_df["comment_text"] = all_comments_df["comment_text"].progress_apply(normalize)
    all_comments_df = all_comments_df[all_comments_df["comment_text"].str.len() > 2]
    all_comments_df["comment_id"] = [str(uuid.uuid4()) for _ in range(len(all_comments_df))]
    
    print(f"   Found {len(all_comments_df):,} total comments (added and removed).")

    print(f"[4/4] Running SATD detector on {len(all_comments_df):,} comments...")
    
    def comments_generator():
        for row in all_comments_df.itertuples():
            yield row.comment_id, row.comment_text
            
    try:
        with open(OUTPUT_LABELED_CSV_GZ, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(OUTPUT_COLUMNS) # Write the full header
            
            comment_map = all_comments_df.set_index('comment_id').to_dict('index')
            satd_count = 0
            total_count = 0
            
            for comment_id, label in classify_with_repl(JAR_PATH, comments_generator()):
                if label == "SATD":
                    satd_count += 1
                total_count += 1
                
                row_data = comment_map[comment_id]
                
                # Build the full row to write
                row_to_write = [
                    comment_id,
                    row_data['comment_text'],
                    row_data['comment_type'],
                    label,
                ] + [row_data.get(col) for col in ALL_FEATURE_COLUMNS]
                
                writer.writerow(row_to_write)
                
        print("\n" + "="*60)
        print("Phase 3 (Corrected) complete.")
        print(f"Total comments processed: {total_count:,}")
        print(f"Total SATD detected:     {satd_count:,}")
        print(f"Output file is: {OUTPUT_LABELED_CSV_GZ}")
        print("You can now run 'rq5.py' (Phase 5a) on this new file.")
    except Exception as e:
        print(f"\n   [FATAL] An error occurred during detection: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if not INPUT_CSV_GZ.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV_GZ}")
        print("Please run Phase 2 (genealogy extraction) first.")
        sys.exit(1)
    if not JAR_PATH.exists():
        print(f"ERROR: SATD Detector JAR not found: {JAR_PATH}")
        sys.exit(1)
    if not Path(JAVA_PATH).exists():
        print(f"ERROR: Java executable not found: {JAVA_PATH}")
        sys.exit(1)
    try:
        import tqdm
    except ImportError:
        print("ERROR: tqdm not found. Please run: pip install tqdm")
        sys.exit(1)
    main()
