#!/usr/bin/env python3
"""
rq5model_and_feature.py

This script replaces the R script AND the model building notebook for RQ5.
It reads the final dataset from Phase 5a and:
1.  Calculates and plots a Spearman correlation heatmap.
2.  Trains a Random Forest classifier to predict "long-lasting" vs. "quick removal".
3.  Prints the most important features, directly answering RQ5.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Ensure all libraries are installed:
# pip install pandas scikit-learn matplotlib seaborn
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
except ImportError:
    print("ERROR: Missing dependencies. Please run:", file=sys.stderr)
    print("pip install pandas scikit-learn matplotlib seaborn", file=sys.stderr)
    sys.exit(1)

# ===================== CONFIG =====================
IN_DIR = Path("/root/satd_detection/satd_work_repl/outputs").resolve()
OUT_DIR = IN_DIR

# Input file from Phase 5a (the one you just made)
INPUT_DATASET = IN_DIR / "rq5_R_input_nonml.csv.gz"

# Output files
OUTPUT_CORR_PLOT = OUT_DIR / "rq5_correlation_heatmap_nonml.png"
OUTPUT_IMPORTANCE_PLOT = OUT_DIR / "rq5_feature_importance_nonml.png"

# ===================== MAIN =====================
def main():
    print("Starting Phase 5b: RQ5 Model Building (nonml Cohort)")
    print(f"Input Dataset: {INPUT_DATASET}")
    print("=" * 60)

    # --- 1. Load Data from Phase 5a ---
    print(f"[1/4] Loading {INPUT_DATASET}...")
    try:
        df = pd.read_csv(INPUT_DATASET, compression="gzip")
    except Exception as e:
        print(f"   [ERROR] Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"   Loaded {len(df):,} instances for classification.")

    # --- 2. Correlation Analysis (Replaces R script) ---
    print("[2/4] Calculating Spearman correlation matrix...")
    
    # Select only numeric features for correlation
    numeric_features = df.select_dtypes(include=np.number)
    
    if numeric_features.empty:
        print("   [ERROR] No numeric features found to correlate.", file=sys.stderr)
    else:
        corr_matrix = numeric_features.corr(method='spearman')
        
        # Plot heatmap
        try:
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Spearman Correlation Heatmap (nonml Cohort Features)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(OUTPUT_CORR_PLOT, dpi=300)
            print(f"   ✓ Saved correlation heatmap to {OUTPUT_CORR_PLOT}")
        except Exception as e:
            print(f"   [WARN] Plotting heatmap failed: {e}. Skipping plot.")
        
        # Print high correlations
        print("\n   Highly Correlated Pairs (|r| > 0.7):")
        for col in corr_matrix.columns:
            for idx in corr_matrix.index:
                if (col != idx) and (abs(corr_matrix.loc[idx, col]) > 0.7):
                    if col > idx: # Avoid printing duplicates
                        print(f"     - {idx} <-> {col}: {corr_matrix.loc[idx, col]:.3f}")

    # --- 3. Prepare Data for Model ---
    print("\n[3/4] Preparing data for Random Forest model...")
    
    # Convert categorical columns to numeric
    le = LabelEncoder()
    # 'CT_mod' (Change Type) is categorical
    if 'CT_mod' in df.columns:
        df['CT_mod'] = le.fit_transform(df['CT_mod'])
        
    # 'rq5_target' is our target 'y'
    df['rq5_target'] = le.fit_transform(df['rq5_target'])
    # 0 = long_lasting, 1 = quick_removal (based on alphabetical order)
    target_names = le.classes_
    print(f"   Target mapping: {list(enumerate(target_names))}")

    # Define X (features) and y (target)
    y = df['rq5_target']
    X = df.drop(columns=['rq5_target'])
    
    # Handle any remaining non-numeric columns (e.g., if proxies failed)
    X = X.select_dtypes(include=np.number).fillna(0)
    
    print(f"   Using {len(X.columns)} features to predict target.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- 4. Train Model and Get Feature Importance (RQ5) ---
    print("[4/4] Training model and extracting feature importance...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Print classification report
    print("\n   Model Performance (on test set):")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # --- THIS IS THE ANSWER TO RQ5 ---
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print("\n" + "="*60)
    print("          RQ5: Feature Importance Results (nonml Cohort)")
    print("  (Which features best predict if SATD will be 'long-lasting')")
    print("="*60)
    print(feature_importance_df.to_markdown(index=False, floatfmt=".4f"))
    
    # Plot feature importance
    try:
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('RQ5: Feature Importance for Predicting Long-Lasting SATD (nonml)')
        plt.tight_layout()
        plt.savefig(OUTPUT_IMPORTANCE_PLOT, dpi=300)
        print(f"\n   ✓ Saved feature importance plot to {OUTPUT_IMPORTANCE_PLOT}")
    except Exception as e:
        print(f"   [WARN] Plotting importance failed: {e}. Skipping plot.")

    print("\nPhase 5b (Model Building) complete.")

if __name__ == "__main__":
    if not INPUT_DATASET.exists():
        print(f"ERROR: Input file not found: {INPUT_DATASET}")
        print("Please run 'phase5a_create_R_dataset.py' first.")
        sys.exit(1)
        
    main()
