#!/usr/bin/env python3
"""
phase5_combined_analysis.py

This script combines the labeled data from all three cohorts (ML, NonML, LLM)
and generates a single 1x2 grid of plots for direct comparison.

- Plot 1: Introduction Survival (Project Start -> SATD Add)
- Plot 2: Removal Survival (SATD Add -> SATD Removal)

This is the final analysis script for the "Genealogy" (Project-Level) method.
"""

import os
import re
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

try:
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    HAVE_LIFELINES = True
except ImportError:
    print("ERROR: Missing dependencies. Please run:", file=sys.stderr)
    print("pip install lifelines matplotlib", file=sys.stderr)
    sys.exit(1)

tqdm.pandas(desc="Processing")

# ===================== CONFIG =====================
# Use the correct directory where your files are located
IN_DIR = Path("/root/satd_detection/satd_work_repl/outputs").resolve()
OUT_DIR = IN_DIR  # Save plots to the same directory

# Create output directory if it doesn't exist
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files from Phase 3
COHORT_FILES = {
    "ML": IN_DIR / "genealogy_ml_labeled.csv.gz",
    "Non-ML": IN_DIR / "genealogy_nonml_labeled.csv.gz",
    "LLM": IN_DIR / "genealogy_llm_labeled.csv.gz"
}

# Output files
OUTPUT_COMBINED_PLOT = OUT_DIR / "genealogy_combined_survival_plots.png"
OUTPUT_COMBINED_SUMMARY = OUT_DIR / "genealogy_combined_summary.csv"

# ===================== HELPERS =====================

def load_labeled_csv(path: Path, cohort_name: str) -> pd.DataFrame:
    """Loads a labeled CSV, adding cohort and isSATD columns."""
    print(f"   Loading {path.name}...")
    try:
        # Try with gzip compression first, then without
        try:
            df = pd.read_csv(path, compression='gzip')
        except:
            df = pd.read_csv(path, compression=None)
    except FileNotFoundError:
        print(f"   [WARN] File not found, skipping: {path.name}", file=sys.stderr)
        return pd.DataFrame()
    except Exception as e:
        print(f"   [ERROR] Failed to read file {path.name}: {e}", file=sys.stderr)
        return pd.DataFrame()
        
    df['c_committer_date'] = pd.to_datetime(df['c_committer_date'], utc=True)
    
    # Fix: Create 'isSATD' from 'label' (exact match)
    if 'label' in df.columns:
        df['isSATD'] = (df['label'].str.strip() == "SATD")
    else:
        print(f"   [ERROR] 'label' column not found in {path.name}", file=sys.stderr)
        return pd.DataFrame()
        
    df['cohort'] = cohort_name
    return df

def km_median(durations: pd.Series, events: pd.Series) -> float | None:
    if durations.empty or events.empty: return np.inf
    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events)
    return float(kmf.median_survival_time_)

# ===================== MAIN =====================

def main():
    print("Starting Phase 5: Combined Survival Analysis")
    print(f"Input Directory: {IN_DIR}")
    print(f"Output Plot: {OUTPUT_COMBINED_PLOT}")
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
    print(f"   Loaded a total of {len(df):,} labeled comments from {len(all_dfs)} cohorts.")

    # --- 2. Introduction Analysis (RQ4a: Time from Project Start) ---
    print("[2/4] Running Introduction Analysis (Time from Project Start)...")
    
    project_starts = df.groupby('project')['c_committer_date'].min().reset_index()
    project_starts = project_starts.rename(columns={'c_committer_date': 'project_start_date'})
    df = df.merge(project_starts, on='project', how='left')
    
    df['time_elapsed_days'] = (df['c_committer_date'] - df['project_start_date']).dt.days
    df['time_elapsed_days'] = df['time_elapsed_days'].apply(lambda x: max(x, 0))

    satd_adds = df[(df['isSATD'] == True) & (df['comment_type'] == 'added')].copy()
    nonsatd_adds = df[(df['isSATD'] == False) & (df['comment_type'] == 'added')].copy()

    # Sample non-SATD *within each cohort* to match SATD count
    intro_dfs = []
    for cohort in df['cohort'].unique():
        satd_sample = satd_adds[satd_adds['cohort'] == cohort]
        nonsatd_sample = nonsatd_adds[nonsatd_adds['cohort'] == cohort]
        
        if satd_sample.empty:
            continue
        
        if len(nonsatd_sample) >= len(satd_sample):
            nonsatd_sample = nonsatd_sample.sample(n=len(satd_sample), random_state=42)
        
        cohort_intro_df = pd.concat([
            satd_sample[['time_elapsed_days', 'isSATD', 'cohort']].rename(columns={'isSATD': 'event_observed'}),
            nonsatd_sample[['time_elapsed_days', 'isSATD', 'cohort']].rename(columns={'isSATD': 'event_observed'})
        ], ignore_index=True)
        intro_dfs.append(cohort_intro_df)
    
    intro_df = pd.concat(intro_dfs, ignore_index=True)

    # --- 3. Removal Analysis (RQ4b: Time from Introduction to Removal) ---
    print("[3/4] Running Removal Analysis (Time to Removal)...")
    
    satd_df = df[df['isSATD'] == True].copy()
    satd_df['comment_key'] = satd_df['project'] + '|' + satd_df['m_filename'] + '|' + satd_df['comment_text']

    adds = satd_df[satd_df['comment_type'] == 'added']
    rems = satd_df[satd_df['comment_type'] == 'removed']

    first_dates = adds.groupby('comment_key')['c_committer_date'].min().to_frame('first_date')
    removal_dates = rems.groupby('comment_key')['c_committer_date'].min().to_frame('removal_date')
    last_commit_dates = df.groupby('project')['c_committer_date'].max().to_frame('last_commit_date')

    removal_df = first_dates.join(removal_dates, how='left')
    
    # Get cohort and project for joining
    key_to_meta = satd_df[['comment_key', 'project', 'cohort']].drop_duplicates().set_index('comment_key')
    removal_df = removal_df.join(key_to_meta)
    removal_df = removal_df.join(last_commit_dates, on='project')
    
    removal_df = removal_df[removal_df['removal_date'].isna() | (removal_df['removal_date'] > removal_df['first_date'])]

    removed_mask = removal_df['removal_date'].notna()
    removal_df['event_observed'] = 0
    removal_df.loc[removed_mask, 'event_observed'] = 1
    
    removal_df.loc[removed_mask, 'survival_days'] = \
        (removal_df[removed_mask]['removal_date'] - removal_df[removed_mask]['first_date']).dt.days
    removal_df.loc[~removed_mask, 'survival_days'] = \
        (removal_df[~removed_mask]['last_commit_date'] - removal_df[~removed_mask]['first_date']).dt.days

    removal_df['survival_days'] = removal_df['survival_days'].apply(lambda x: max(x, 0))
    removal_df = removal_df.dropna(subset=['survival_days'])

    # --- 4. Plot Combined Grid ---
    print("[4/4] Generating combined plots and summary...")
    
    fig, (ax_intro, ax_removal) = plt.subplots(1, 2, figsize=(20, 8)) # 1 row, 2 columns
    
    summary_data = []

    # --- Plot 1: Introduction ---
    for cohort in sorted(intro_df['cohort'].unique()):
        cohort_data = intro_df[intro_df['cohort'] == cohort]
        kmf_intro = KaplanMeierFitter()
        kmf_intro.fit(cohort_data['time_elapsed_days'], cohort_data['event_observed'], label=cohort)
        kmf_intro.plot(ci_show=True, ax=ax_intro)
        
        # Get stats
        satd_adds_cohort = satd_adds[satd_adds['cohort'] == cohort]
        intro_median = satd_adds_cohort['time_elapsed_days'].median()
        summary_data.append({
            "cohort": cohort,
            "analysis": "Introduction",
            "median_days": intro_median,
            "event_rate_pct": None, # Event rate is 50% by design of sampling
            "n_subjects": len(satd_adds_cohort)
        })

    ax_intro.set_title('SATD Introduction (Genealogy Method)', fontsize=16)
    ax_intro.set_xlabel('Days from Project Start', fontsize=12)
    ax_intro.set_ylabel('Probability of Being SATD-Free', fontsize=12)
    ax_intro.axvline(365, linestyle="--", alpha=0.4, color="gray")
    ax_intro.legend()

    # --- Plot 2: Removal ---
    for cohort in sorted(removal_df['cohort'].unique()):
        cohort_data = removal_df[removal_df['cohort'] == cohort]
        kmf_removal = KaplanMeierFitter()
        kmf_removal.fit(cohort_data['survival_days'], cohort_data['event_observed'], label=cohort)
        kmf_removal.plot(ci_show=True, ax=ax_removal)
        
        # Get stats
        removal_median = km_median(cohort_data['survival_days'], cohort_data['event_observed'])
        removal_rate = cohort_data['event_observed'].mean() * 100
        summary_data.append({
            "cohort": cohort,
            "analysis": "Removal",
            "median_days": removal_median,
            "event_rate_pct": removal_rate,
            "n_subjects": len(cohort_data)
        })

    ax_removal.set_title('SATD Removal (Genealogy Method)', fontsize=16)
    ax_removal.set_xlabel('Days from SATD Introduction', fontsize=12)
    ax_removal.set_ylabel('Probability SATD Still Present', fontsize=12)
    ax_removal.axvline(365, linestyle="--", alpha=0.4, color="gray")
    ax_removal.legend()

    # --- Save plot and summary ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Genealogy-Level SATD Survival Analysis (All Cohorts)', fontsize=20)
    
    # Ensure the output directory exists before saving
    OUTPUT_COMBINED_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_COMBINED_PLOT, dpi=300)
    plt.close()  # Close the figure to free memory
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.replace(np.inf, "inf")
    summary_df.to_csv(OUTPUT_COMBINED_SUMMARY, index=False)

    print("\n" + "="*60)
    print("Phase 5 (Combined Analysis) complete.")
    print(f"Combined Plot saved: {OUTPUT_COMBINED_PLOT}")
    print(f"Combined Summary saved: {OUTPUT_COMBINED_SUMMARY}")
    print("="*60)
    print("\nFinal Summary Statistics:")
    
    # Try to use markdown format if available, otherwise use regular print
    try:
        print(summary_df.to_markdown(index=False, floatfmt=".1f"))
    except:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    # Check that all files exist before starting
    all_found = True
    for cohort, path in COHORT_FILES.items():
        if not path.exists():
            print(f"ERROR: Input file not found for cohort '{cohort}': {path}")
            all_found = False
            
    if not all_found:
        print("Please ensure all Phase 3 scripts have run successfully.")
        sys.exit(1)
        
    main()