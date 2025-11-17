#!/usr/bin/env python3
"""
phase5a_rq5_dataprep.py (CORRECTED for MEMORY)

This script prepares the final, combined dataset for the RQ5 analysis.

FIX:
- Added 'COLUMNS_TO_LOAD' list and 'usecols' to pd.read_csv
  to prevent Segmentation Fault (out of memory) errors on large
  files like the LLM cohort.
"""

import os
import re
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

tqdm.pandas(desc="Processing")

# ===================== CONFIG =====================
IN_DIR = Path("/root/satd_detection/satd_work_repl/outputs").resolve()
OUT_DIR = IN_DIR

# Process ONE cohort at a time to save memory
COHORT_FILES = {
    # "ML": IN_DIR / "genealogy_ml_labeled.csv.gz",
    # "Non-ML": IN_DIR / "genealogy_nonml_labeled.csv.gz",
     "LLM": IN_DIR / "genealogy_llm_labeled.csv.gz"
}

# Output file for the R script - LLM SPECIFIC
OUTPUT_DATASET = OUT_DIR / "rq5_R_input_llm.csv.gz"

# --- MEMORY FIX: Define exactly which columns to load ---
# These are all the columns needed for the rest of this script
COLUMNS_TO_LOAD = [
    'label', 'c_committer_date', 'project', 'm_filename', 'comment_text',
    'comment_type', 'c_deletions', 'c_insertions', 'c_lines', 'c_files',
    'm_change_type', 'm_complexity', 'm_token_count', 'm_nloc'
]
# ----------------------------------------------------

# ===================== HELPERS =====================
def load_labeled_csv(path: Path, cohort_name: str) -> pd.DataFrame:
    print(f"   Loading {path.name}...")
    try:
        # --- MEMORY FIX: Use 'usecols' to load only what we need ---
        df = pd.read_csv(path, compression=None, usecols=COLUMNS_TO_LOAD)
    except FileNotFoundError:
        print(f"   [WARN] File not found, skipping: {path.name}", file=sys.stderr)
        # Try the /root/ path for LLM as a fallback
        if cohort_name == "LLM":
            path_alt = Path("/root/satd_detection/satd_work_repl/outputs/genealogy_llm_labeled.csv.gz")
            if path_alt.exists():
                print(f"   Found LLM data at {path_alt}")
                df = pd.read_csv(path_alt, compression=None, usecols=COLUMNS_TO_LOAD)
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        # Handle if a column is missing (e.g., from an older run)
        if "usecols" in str(e):
             print(f"   [WARN] Columns mismatch. Trying to load full file... (this may fail)")
             df = pd.read_csv(path, compression=None)
        else:
            print(f"   [ERROR] Failed to read file {path.name}: {e}", file=sys.stderr)
            return pd.DataFrame()
            
    df['c_committer_date'] = pd.to_datetime(df['c_committer_date'], utc=True)
    if 'label' in df.columns:
        df['isSATD'] = (df['label'].str.strip() == "SATD")
    else:
        print(f"   [ERROR] 'label' column not found in {path.name}", file=sys.stderr)
        return pd.DataFrame()
        
    df['cohort'] = cohort_name
    return df

# ===================== MAIN =====================
def main():
    print("Starting Phase 5a: Creating R Dataset (Memory Optimized)")
    print(f"Output Dataset: {OUTPUT_DATASET}")
    print("=" * 60)

    # --- 1. Load and Combine Data ---
    print("[1/4] Loading and combining all cohort data...")
    all_dfs = []
    for cohort_name, file_path in COHORT_FILES.items():
        cohort_df = load_labeled_csv(file_path, cohort_name)
        if not cohort_df.empty:
            all_dfs.append(cohort_df)
    
    if not all_dfs:
        print("   [ERROR] No data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    df = pd.concat(all_dfs, ignore_index=True)
    all_dfs = None # Free memory
    print(f"   Loaded a total of {len(df):,} labeled comments.")

    # --- 2. Calculate Survival Data (for all SATD) ---
    print("[2/4] Calculating survival time for all SATD instances...")
    
    satd_df = df[df['isSATD'] == True].copy()
    df = None # Free memory
    
    satd_df['comment_key'] = satd_df['project'] + '|' + satd_df['m_filename'] + '|' + satd_df['comment_text']

    print(f"   Found {len(satd_df):,} total SATD comments ({satd_df['comment_key'].unique().size:,} unique instances).")
    
    adds = satd_df[satd_df['comment_type'] == 'added']
    rems = satd_df[satd_df['comment_type'] == 'removed']

    # Get introduction data (first_date and all commit features)
    adds = adds.sort_values('c_committer_date', ascending=True)
    first_adds = adds.drop_duplicates(subset=['comment_key'], keep='first').copy()
    
    removal_dates = rems.groupby('comment_key')['c_committer_date'].min().to_frame('removal_date')
    
    # We only need project and last commit date from the original SATD df
    last_commit_dates = satd_df.groupby('project')['c_committer_date'].max().to_frame('last_commit_date')
    satd_df = None # Free memory

    # Join survival data back to the introduction commit features
    survival_data = first_adds.join(removal_dates, on='comment_key', how='left')
    survival_data = survival_data.join(last_commit_dates, on='project')
    
    # Filter out git noise
    survival_data = survival_data[survival_data['removal_date'].isna() | (survival_data['removal_date'] > survival_data['c_committer_date'])]

    removed_mask = survival_data['removal_date'].notna()
    survival_data['event_observed'] = 0
    survival_data.loc[removed_mask, 'event_observed'] = 1
    
    # c_committer_date is the introduction date (since we filtered for first_adds)
    survival_data.loc[removed_mask, 'survival_days'] = \
        (survival_data[removed_mask]['removal_date'] - survival_data[removed_mask]['c_committer_date']).dt.days
    survival_data.loc[~removed_mask, 'survival_days'] = \
        (survival_data[~removed_mask]['last_commit_date'] - survival_data[~removed_mask]['c_committer_date']).dt.days

    survival_data['survival_days'] = survival_data['survival_days'].apply(lambda x: max(x, 0))
    survival_data = survival_data.dropna(subset=['survival_days'])
    print(f"   Calculated survival for {len(survival_data):,} SATD instances.")

    # --- 3. Define Target Variable (y) ---
    print("[3/4] Defining target variable (Quick vs. Long-lasting)...")
    
    q25 = survival_data.groupby('cohort')['survival_days'].quantile(0.25).to_dict()
    q75 = survival_data.groupby('cohort')['survival_days'].quantile(0.75).to_dict()

    def classify_duration(row):
        q25_val = q25.get(row['cohort'])
        q75_val = q75.get(row['cohort'])
        if q25_val is None or q75_val is None:
            return None
        if row['survival_days'] <= q25_val:
            return 'quick_removal'
        if row['survival_days'] >= q75_val:
            return 'long_lasting'
        return None

    survival_data['rq5_target'] = survival_data.apply(classify_duration, axis=1)
    
    rq5_dataset = survival_data.dropna(subset=['rq5_target']).copy()
    survival_data = None # Free memory
    
    print(f"   Created final balanced dataset of {len(rq5_dataset):,} instances.")

    # --- 4. Rename & Save Final Dataset ---
    print(f"[4/4] Renaming columns and saving final dataset to {OUTPUT_DATASET}...")
    
    # Map from the Bhatia et al. R script names (Table 5) to our PyDriller/Phase2 names
    rename_map = {
        "LD_[diff]": "c_deletions",
        "LA_[diff]": "c_insertions",
        "LM_[diff]": "c_lines",
        "MF_[diff]": "c_files",
        "MD_[diff]": "c_files", 
        "LA_[mod]": "c_insertions", 
        "LD_[mod]": "c_deletions",
        "CT_[mod]": "m_change_type",
        "CM_[mod]": "c_files", 
        "Complexity": "m_complexity",
        "Tokens": "m_token_count",
        "LOC": "m_nloc",
        # History metrics are not available in our Phase 2 data, so we omit them
        # "LCT_[hist]": "c_files", 
        # "PT_[hist]": "c_files", 
        # "FC_[hist]": "c_files", 
        # "Had_SATD_[hist]": "c_files",
    }
    
    # Create the new dataframe
    final_df = pd.DataFrame()
    final_df['rq5_target'] = rq5_dataset['rq5_target']

    for r_name, py_name in rename_map.items():
        if py_name in rq5_dataset.columns:
            final_df[r_name] = rq5_dataset[py_name]
        else:
            print(f"   [WARN] Feature '{py_name}' not found. Skipping column '{r_name}'.")
            
    # Fill any NaNs with 0
    final_df = final_df.fillna(0)

    final_df.to_csv(OUTPUT_DATASET, index=False, compression="gzip")

    print("\n" + "="*60)
    print("Phase 5a (Data Prep) complete.")
    print(f"Feature dataset saved to: {OUTPUT_DATASET}")
    print("You can now run 'RQ5-Corr_Redun.R'.")

if __name__ == "__main__":
    main()