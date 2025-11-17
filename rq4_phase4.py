#!/usr/bin/env python3
"""
phase4_survival_analysis_ml.py (LLM Cohort Only)

This script performs the "Survival Analysis" phase (like RQ4.ipynb).
It consumes the labeled CSV from Phase 3 (genealogy_llm_labeled.csv.gz)
and produces the final survival analysis plots and summary statistics
for the LLM cohort.
"""

import os
import re
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

# Ensure lifelines and matplotlib are installed
try:
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    HAVE_LIFELINES = True
except ImportError:
    print("ERROR: Missing dependencies. Please run:", file=sys.stderr)
    print("pip install lifelines matplotlib", file=sys.stderr)
    sys.exit(1)

# Enable progress bars for pandas `apply`
tqdm.pandas(desc="Calculating Survival")

# ===================== CONFIG =====================
IN_DIR = Path("/root/satd_detection/satd_work_repl/outputs").resolve()
OUT_DIR = IN_DIR  # Save outputs to the same directory

# Input file from Phase 3
INPUT_LABELED_CSV_GZ = IN_DIR / "genealogy_llm_labeled.csv.gz"

# Output files
OUTPUT_INTRO_PLOT = OUT_DIR / "genealogy_llm_intro_survival.png"
OUTPUT_REMOVAL_PLOT = OUT_DIR / "genealogy_llm_removal_survival.png"
OUTPUT_SUMMARY = OUT_DIR / "genealogy_llm_summary.csv"

# ===================== HELPERS =====================

def km_median(durations: pd.Series, events: pd.Series) -> float | None:
    if durations.empty or events.empty:
        return np.inf
    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events)
    return float(kmf.median_survival_time_)

def plot_km(data: pd.DataFrame, time_col: str, event_col: str, title: str, xlabel: str, ylabel: str, outfile: Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data[time_col], event_observed=data[event_col], label="LLM Cohort")
    kmf.plot(ci_show=True, ax=ax)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axvline(365, linestyle="--", alpha=0.4, color="gray")
    ax.text(375, 0.1, "1 Year", rotation=90, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✓ Wrote plot: {outfile.name}")

# ===================== MAIN =====================

def main():
    print("Starting Phase 4: Survival Analysis (LLM COHORT ONLY)")
    print(f"Input file: {INPUT_LABELED_CSV_GZ}")
    print("=" * 60)

    # --- 1. Load Data from Phase 3 ---
    print(f"[1/4] Loading {INPUT_LABELED_CSV_GZ}...")
    try:
        df = pd.read_csv(INPUT_LABELED_CSV_GZ, compression=None)
        df['c_committer_date'] = pd.to_datetime(df['c_committer_date'], utc=True)
    except Exception as e:
        print(f"   [ERROR] Failed to read file: {e}", file=sys.stderr)
        print("   Did Phase 3 complete successfully?", file=sys.stderr)
        sys.exit(1)
    
    print(f"   Loaded {len(df):,} labeled comments.")

    # --- THIS IS THE FIX ---
    # Create the 'isSATD' column from the 'label' column
    if 'label' in df.columns and 'isSATD' not in df.columns:
        print("   [INFO] Creating 'isSATD' column from 'label' column...")
        # Use an exact match, not 'contains'
        df['isSATD'] = (df['label'].str.strip() == "SATD")
    elif 'isSATD' not in df.columns:
         print("   [ERROR] Neither 'label' nor 'isSATD' column found.", file=sys.stderr)
         sys.exit(1)
    # ---------------------

    # --- 2. Introduction Analysis (RQ4a: Time from Project Start) ---
    print("[2/4] Running Introduction Analysis (Time from Project Start)...")
    
    print("   Calculating project start dates...")
    project_starts = df.groupby('project')['c_committer_date'].min().reset_index()
    project_starts = project_starts.rename(columns={'c_committer_date': 'project_start_date'})
    
    df = df.merge(project_starts, on='project', how='left')
    
    df['time_elapsed_days'] = (df['c_committer_date'] - df['project_start_date']).dt.days
    df['time_elapsed_days'] = df['time_elapsed_days'].apply(lambda x: max(x, 0))

    satd_adds = df[(df['isSATD'] == True) & (df['comment_type'] == 'added')].copy()
    nonsatd_adds = df[(df['isSATD'] == False) & (df['comment_type'] == 'added')].copy()

    print(f"   Found {len(satd_adds):,} SATD-added comments.")
    print(f"   Found {len(nonsatd_adds):,} Non-SATD-added comments.")

    if satd_adds.empty:
        print("   [WARN] No SATD 'added' comments found. Skipping Introduction analysis.")
        intro_median = None
    else:
        # Sample non-SATD to match SATD count
        if len(nonsatd_adds) >= len(satd_adds):
            nonsatd_sample = nonsatd_adds.sample(n=len(satd_adds), random_state=42)
            print(f"   Sampling {len(nonsatd_sample):,} Non-SATD comments for comparison.")
        else:
            print(f"   [WARN] Fewer Non-SATD comments ({len(nonsatd_adds)}) than SATD comments ({len(satd_adds)}). Using all.")
            nonsatd_sample = nonsatd_adds
            
        intro_df = pd.concat([
            satd_adds[['time_elapsed_days', 'isSATD']].rename(columns={'isSATD': 'event_observed'}),
            nonsatd_sample[['time_elapsed_days', 'isSATD']].rename(columns={'isSATD': 'event_observed'})
        ], ignore_index=True)
        
        plot_km(
            intro_df, 'time_elapsed_days', 'event_observed',
            'SATD Introduction (Genealogy Method, LLM Cohort)',
            'Days from Project Start',
            'Probability of Being SATD-Free',
            OUTPUT_INTRO_PLOT
        )
        
        # Calculate median time TO INTRODUCTION for SATD comments
        intro_median = satd_adds['time_elapsed_days'].median()
        print(f"   Median days to introduction (Project Start -> SATD): {intro_median:.1f}")

    # --- 3. Removal Analysis (RQ4b: Time from Introduction to Removal) ---
    print("[3/4] Running Removal Analysis (Time to Removal)...")
    
    satd_df = df[df['isSATD'] == True].copy()
    satd_df['comment_key'] = satd_df['project'] + '|' + satd_df['m_filename'] + '|' + satd_df['comment_text']

    print(f"   Found {len(satd_df):,} total SATD comments ({len(satd_df['comment_key'].unique()):,} unique instances).")
    
    adds = satd_df[satd_df['comment_type'] == 'added']
    rems = satd_df[satd_df['comment_type'] == 'removed']

    first_dates = adds.groupby('comment_key')['c_committer_date'].min().to_frame('first_date')
    removal_dates = rems.groupby('comment_key')['c_committer_date'].min().to_frame('removal_date')

    last_commit_dates = df.groupby('project')['c_committer_date'].max().to_frame('last_commit_date')

    removal_df = first_dates.join(removal_dates, how='left')
    
    key_to_project = satd_df[['comment_key', 'project']].drop_duplicates().set_index('comment_key')
    removal_df = removal_df.join(key_to_project)
    removal_df = removal_df.join(last_commit_dates, on='project')
    
    # Remove any instances that were removed *before* or *at* introduction (git noise)
    removal_df = removal_df[removal_df['removal_date'].isna() | (removal_df['removal_date'] > removal_df['first_date'])]

    removed_mask = removal_df['removal_date'].notna()
    
    removal_df['event_observed'] = 0 # 0 = Censored (still exists)
    removal_df.loc[removed_mask, 'event_observed'] = 1 # 1 = Event (was removed)
    
    # For removed items, time = removal_date - first_date
    removal_df.loc[removed_mask, 'survival_days'] = \
        (removal_df[removed_mask]['removal_date'] - removal_df[removed_mask]['first_date']).dt.days
        
    # For censored items, time = last_commit_date - first_date
    removal_df.loc[~removed_mask, 'survival_days'] = \
        (removal_df[~removed_mask]['last_commit_date'] - removal_df[~removed_mask]['first_date']).dt.days

    removal_df['survival_days'] = removal_df['survival_days'].apply(lambda x: max(x, 0))
    removal_df = removal_df.dropna(subset=['survival_days'])

    plot_km(
        removal_df, 'survival_days', 'event_observed',
        'SATD Removal (Genealogy Method, LLM Cohort)',
        'Days from SATD Introduction',
        'Probability SATD Still Present',
        OUTPUT_REMOVAL_PLOT
    )

    removal_median = km_median(removal_df['survival_days'], removal_df['event_observed'])
    removal_rate = removal_df['event_observed'].mean() * 100
    print(f"   Median survival time (Introduction -> Removal): {removal_median:.1f} days")
    print(f"   Total removal rate: {removal_rate:.1f}%")
    
    # --- 4. Save Summary ---
    print("[4/4] Saving summary file...")
    
    summary = {
        "cohort": "llm",
        "total_comments_processed": len(df),
        "total_satd_comments": len(satd_df),
        "total_satd_instances": len(removal_df),
        "satd_instances_added": len(satd_adds),
        "satd_instances_removed": int(removal_df['event_observed'].sum()),
        "intro_median_days_from_project_start": intro_median,
        "removal_median_survival_days": removal_median,
        "removal_rate_pct": removal_rate
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    
    print("\n" + "="*60)
    print("Phase 4 (Survival Analysis) complete.")
    print(f"Summary file saved: {OUTPUT_SUMMARY}")
    print(f"Plots saved to: {OUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    if not INPUT_LABELED_CSV_GZ.exists():
        print(f"ERROR: Input file not found: {INPUT_LABELED_CSV_GZ}")
        print("Please run Phase 3 (SATD Labeling) first.")
        sys.exit(1)
        
    main()