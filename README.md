# Criteo CTR Prediction (Logistic Regression and LightGBM)

This repo trains Logistic Regression models and LightGBM CTR model on the Criteo Display Ads dataset with:
- hashed categorical features
- numeric log transforms + missing indicators
- evaluation: AUC, logloss, PR-AUC, Brier, ECE
- SHAP analysis

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

## Notebooks
see 
- notebooks/01_ctr_baseline.ipynb.ipynb
- notebooks/02_ctr_validation_tune_improve.ipynb
- notebooks/03_ctr_lightgbm.ipynb

## Data
Download the Criteo Display Ads dataset and place train.txt under:
- data/criteo/train.txt