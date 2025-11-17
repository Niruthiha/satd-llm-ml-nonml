
# SATD Genealogy Analysis Pipeline

This README explains the 3-phase "Genealogy-Level" pipeline and includes the specific instructions you requested about how to re-run it for the **nonml** and **llm** cohorts.

---

# SATD Analysis Replication Pipeline

This project contains two distinct Python pipelines for replicating and extending the quantitative SATD survival analyses from **Bhatia et al. (2024)**.

---

## 1. Genealogy-Level Analysis (Project-Start)

- Measures SATD lifecycle **from the project's start date**.  
- Replicates methodology from the original `SATD_genealogy.ipynb` and `RQ4.ipynb` notebooks.  
- Implemented as a **3‑phase pipeline**:
  - `rq4_phase2.py`
  - `rq4_phase3.py`
  - `rq4_phase4.py`  
- Must be run **separately for each cohort** (ML, Non-ML, LLM).

---

## 2. File-Level Analysis (File-Creation)

- Measures SATD lifecycle **from the file’s creation date**.
- Replicates methodology from **Table 4** of Bhatia et al. (2024).
- Implemented as a **single script**: `rq4file_level.py`
- Processes **all cohorts in one run**.

---

# Prerequisites

## Environment
- Python 3.9+
- Java 8+ (`/usr/bin/java` must be set in `rq4_phase3.py`)
- Git

## Python Libraries
Install:
```bash
pip install pandas pydriller gitpython tqdm lifelines matplotlib
```

## Required Files & Directories

Place the following:

### Repositories  
`/root/satd_detection/cloned_repos/`  
Contains all cloned repositories.

### Cohort Lists  
Located at `/root/satd_detection/`:
- `ml_159.txt`
- `nonml_159.txt`
- `llm_159.txt`

### SATD Detector  
[`/root/satd_detection/satd_detector.jar`](https://dl.acm.org/doi/10.1145/3183440.3183478)

### RQ1 Predictions  
`/root/satd_detection/RQ1/outputs/`:
- `ml_predictions.tsv`
- `nonml_predictions.tsv`
- `llm_predictions.tsv`

### Pipeline Output Directories
- Genealogy-Level outputs → `/root/satd_detection/satd_work_repl/outputs/`
- File-Level outputs → `/root/satd_detection/file_level_survival/`

---

# Analysis 1: Genealogy-Level Pipeline (Project-Start)

This is a **3-phase pipeline** configured by default for ML.

---

## Phase 2: Genealogy Extraction
- Scans full git history of each repository in a cohort.
- Extracts metadata for all `.py` file modifications.

**Script:** `rq4_phase2.py`  
**Input:** `ml_159.txt`  
**Output:** `genealogy_ml_modifications.csv.gz`

---

## Phase 3: SATD Labeling
- Extracts added/removed comments.
- Runs the Java SATD detector.

**Script:** `rq4_phase3.py`  
**Input:** `genealogy_ml_modifications.csv.gz`  
**Output:** `genealogy_ml_labeled.csv.gz`

---

## Phase 4: Survival Analysis
- Computes survival curves and median times.

**Script:** `rq4_phase4.py`  
**Input:** `genealogy_ml_labeled.csv.gz`  
**Output:**  
- `genealogy_ml_summary.csv`  
- Survival plots

---

# How to Replicate for Non-ML and LLM Cohorts

Edit **all 3 scripts** for each cohort.

---

## Example: Running the *nonml* Cohort

### 1. Edit `rq4_phase2.py`
```python
REPO_CONFIGS = {
    "nonml": {
        "urls_file": TXT_FILE_DIR / "nonml_159.txt",
    },
}
OUTPUT_CSV = OUT_DIR / "genealogy_nonml_modifications.csv"
```

Run:
```bash
python rq4_phase2.py
```

---

### 2. Edit `rq4_phase3.py`
```python
INPUT_CSV_GZ = IN_DIR / "genealogy_nonml_modifications.csv.gz"
OUTPUT_LABELED_CSV_GZ = OUT_DIR / "genealogy_nonml_labeled.csv.gz"
```

Run:
```bash
python rq4_phase3.py
```

---

### 3. Edit `rq4_phase4.py`
Update inputs, outputs, labels, and cohort string:

```python
INPUT_LABELED_CSV_GZ = IN_DIR / "genealogy_nonml_labeled.csv.gz"

OUTPUT_INTRO_PLOT = OUT_DIR / "genealogy_nonml_intro_survival.png"
OUTPUT_REMOVAL_PLOT = OUT_DIR / "genealogy_nonml_removal_survival.png"
OUTPUT_SUMMARY = OUT_DIR / "genealogy_nonml_summary.csv"

print("Starting Phase 4: Survival Analysis (NON-ML COHORT ONLY)")

kmf.fit(..., label="Non-ML Cohort")

summary = { "cohort": "nonml", ... }
```

Run:
```bash
python rq4_phase4.py
```

Repeat this for the **llm** cohort, replacing `nonml` → `llm`.

---

# Analysis 2: File-Level Pipeline (File-Creation)

This pipeline requires **no configuration changes**.

---

## Script: `rq4file_level.py`
This script:

- Processes **ML**, **Non-ML**, and **LLM** automatically  
- Computes:
  - Time from file creation to first SATD
  - Time from first SATD to removal (if applicable)
- Saves outputs to `/root/satd_detection/file_level_survival/`

### Run:
```bash
python rq4file_level.py
```

### Output Includes:
- Per-cohort CSVs:  
  - `intro_ml_files.csv`  
  - `removal_llm_files.csv`  
  - etc.
- Final comparison plots:  
  - `km_intro.png`  
  - `km_removal.png`
- Console summary with `[STATS]` for all cohorts.

---

# End of Document

