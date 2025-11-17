# SATD Genealogy Analysis Pipeline (RQ4 - Project-Level)

This README explains the 3-phase "Genealogy-Level" pipeline and includes the specific instructions you requested about how to re-run it for the **nonml** and **llm** cohorts.

---

## Overview

This directory contains the Python scripts to replicate the "Genealogy-Level" (or "Project-Level") analysis from the Bhatia et al. (2024) paper. This pipeline analyzes SATD from the perspective of the project's start date, as described in their **SATD_genealogy.ipynb** and **RQ4.ipynb** notebooks.

The analysis answers two main questions:

1. **Introduction:** How long from a project's start until a typical SATD comment is introduced?  
2. **Removal:** How long does a typical SATD comment survive from its introduction until it's removed?

This pipeline is run separately for each cohort (**ML**, **Non-ML**, **LLM**).  
The scripts are pre-configured for **ML** only; instructions for running the others appear later.

---

## Prerequisites

### Environment
- Python 3.9+
- Java (must be accessible at `/usr/bin/java`)

### Repositories
- All cloned repositories must live here:  
  `/root/satd_detection/cloned_repos/`

### Configuration Files
Place these in `/root/satd_detection/`:
- `ml_159.txt`
- `nonml_159.txt`
- `llm_159.txt`

### SATD Detector
- `satd_detector.jar` must be located at:  
  `/root/satd_detection/satd_detector.jar`

### Python Libraries
Install:
```bash
pip install pandas pydriller gitpython tqdm lifelines matplotlib
```

---

## Pipeline Workflow (ML Cohort Example)

This analysis has **3 phases** and they must be executed **in order**.

---

### **Phase 1: Genealogy Extraction**

**Script:** `rq4_phase2.py`  
**Action:** Mines git history of repos listed in `ml_159.txt`.  
**Output:** `genealogy_ml_modifications.csv.gz`

This step scans *all* commits and extracts `.py` file modification metadata.

---

### **Phase 2: SATD Labeling**

**Script:** `rq4_phase3.py`  
**Input:** `genealogy_ml_modifications.csv.gz`  
**Action:** Extracts added/removed comments and runs the Java SATD detector.  
**Output:** `genealogy_ml_labeled.csv.gz`

This step labels ~1.2M comments as *SATD* or *Not SATD*.

---

### **Phase 3: Survival Analysis**

**Script:** `rq4_phase4.py`  
**Input:** `genealogy_ml_labeled.csv.gz`  
**Action:** Computes survival curves for SATD introduction and removal.  
**Output:**
- `genealogy_ml_summary.csv`
- `genealogy_ml_intro_survival.png`
- `genealogy_ml_removal_survival.png`

---

## How to Run for **Non-ML** and **LLM** Cohorts

The scripts are hardcoded for ML.  
To run the pipeline for `nonml` or `llm`, you must **edit each of the 3 scripts**.

---

## Example: Running the **nonml** Cohort

---

### **1. Edit `rq4_phase2.py`**

Change repository config and output filename:

```python
# === In rq4_phase2.py ===
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

### **2. Edit `rq4_phase3.py`**

Change input and output file names:

```python
# === In rq4_phase3.py ===
INPUT_CSV_GZ = IN_DIR / "genealogy_nonml_modifications.csv.gz"
OUTPUT_LABELED_CSV_GZ = OUT_DIR / "genealogy_nonml_labeled.csv.gz"
```

Run:
```bash
python rq4_phase3.py
```

---

### **3. Edit `rq4_phase4.py`**

Update labeled input file, outputs, plot labels, and cohort metadata:

```python
# === In rq4_phase4.py ===
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

---

## Running for the **LLM** Cohort

Repeat all three steps above, replacing:
- `nonml` → `llm`
- `nonml_159.txt` → `llm_159.txt`
- All filenames and labels accordingly.

---

## End of Document

