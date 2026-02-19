# Criteo CTR Prediction (LightGBM)

This repo trains a LightGBM CTR model on the Criteo Display Ads dataset with:
- hashed categorical features
- numeric log transforms + missing indicators
- evaluation: AUC, logloss, PR-AUC, Brier, ECE
- SHAP analysis

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

