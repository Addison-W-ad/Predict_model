# Predict model
# Xenium + Clinical Survival Prediction

This repository trains survival models by integrating **10x Genomics Xenium** spatial transcriptomics data with **clinical covariates** (e.g., age, BMI, HIV status, stage, sex). We provide CoxPH and a neural network (Cox partial likelihood) with metabolism-pathway attention.

## Goals
- Preprocess Xenium outputs and derive sample-/patient-level features.
- Combine with curated clinical variables.
- Train/evaluate survival models (CoxPH, NN) and report C-index.

## Data Requirements
- **Xenium outputs** (typical files):
  - `cells.csv` (per-cell features, cell IDs, coordinates, cell types if available)
  - `cell_feature_matrix.h5` or equivalent
  -  could also include cell location information
- **Clinical table** (CSV): one row per patient/sample with columns like:
  - `patient_id`, `age`, `BMI`, `HIV_status`, `sex`, `stage`, `treatment`, …
  - `time` (follow-up time), `event` (1=event, 0=censored)

> Ensure that `patient_id` links Xenium samples to the clinical table (e.g., slide → patient mapping).

## Feature Engineering (examples)
- **Spatial/omics:** gene set scores, pathway activities (metabolism), cell-type proportions, neighborhood metrics.
- **Clinical:** numeric standardized (z-score); categorical one-hot or ordinal as appropriate.

## Model Overview
- **CoxPH (lifelines)** for hazard ratios and baseline-agnostic inference.
- **Neural Network (PyTorch)** with:
  - Metabolism attention → 64-d embedding
  - Clinical encoder → 32-d embedding
  - Concatenation → MLP → risk score
  - Trained via **Cox partial likelihood** (stable `logcumsumexp` risk-set loss)

