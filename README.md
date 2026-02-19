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
```

## Notebooks
See the model training and evaluation result in notebooks:
- notebooks/01_ctr_baseline.ipynb.ipynb
- notebooks/02_ctr_validation_tune_improve.ipynb
- notebooks/03_ctr_lightgbm.ipynb

## Data
Download the Criteo [Display Ads dataset](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset?select=dac) 
and place train.txt under:
- data/criteo/train.txt

## Key results

### LR train on 5M rows, Train/Test split: 0.8/0.2 

- split seed = 42

Best metric on test dataset:

| Metric   | Value  |
|----------|--------|
| AUC      | 0.7696 |
| LogLoss  | 0.4726 |
| Brier    | 0.1536 |
| Mean p   | 0.2511 |

This is achieved by

- Feature engineering
  - Feature hashing
  - Frequecy encoding

- Model Architecture
  - SGDClassifier optimization method.
  - Bucketized calibration layer by platt scaling.
    - Isotonic regression and single calibration layer gives similar performance.  

### LightGBM train on 5M rows, Train/Eval/Test split: 0.8/0.1/0.1 
split seed = 42, 11

The best offline metrics on test set we get is:
| Metric         | Value  |
|----------------|--------|
| auc            | 0.7964 |
| logloss        | 0.4498 |
| pr_auc         | 0.5867 |
| brier          | 0.1456 |
| ctr_mean_true  | 0.2513 |
| ctr_mean_pred  | 0.2500 |
| ECE            | 0.0042 |

{'k': 0.01, 'n_top': 5000, 'ctr_top': 0.8986, 'ctr_all': 0.25125, 'lift': 3.5765174129353094} 

{'k': 0.05, 'n_top': 25000, 'ctr_top': 0.78296, 'ctr_all': 0.25125, 'lift': 3.1162587064676495}

This is achieved by:

- Feature engineering
  - Features added
    - hashing of categorical features
    - log1p on numerical features
    - category feature freq 
    - missing features
  - Features excluded
    - Feature crossings
    - Log frequencies

  
- LightGBM model params:
```python
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.03,
        "num_leaves": 255,
        "max_depth": 12,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10,
        "verbosity": -1,
        "seed": 42,
    }
```


