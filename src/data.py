# CTR Baseline Notebook
# Logistic Regression on small Criteo subset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

def test_func():
    print(2);

def load_data(data_path, data_size, train_eval_random_state=20):
    print(train_eval_random_state)
    # 0. add header
    # -----------------------------
    num_features = [f"I{i}" for i in range(1, 14)]
    cat_features = [f"C{i}" for i in range(1, 27)]
    columns = ["label"] + num_features + cat_features
    
    # -----------------------------
    # 1. Load small subset of dataset
    # -----------------------------
    # Read first nrows to start
    data = pd.read_csv(data_path, sep='\t', header=None, names=columns, nrows=data_size)
    
    
    # -----------------------------
    # 2. Separate target and features
    # -----------------------------
    y = data['label']
    X = data.drop(['label'], axis=1)
    
    # -----------------------------
    # 3. Train/Test Split
    # -----------------------------
    X_tmp_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # -----------------------------
    # 4. Train/Eval Split
    # -----------------------------
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_tmp_raw, y_train, test_size=0.1, random_state=train_eval_random_state
    )
    
    return num_features, cat_features, columns, X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test

# Fill missing values.
def fill_missing_values(X_train, X_val, X_test, num_features, cat_features):
    # Fill missing numeric features with 0
    for col in num_features:
        X_train[col] = X_train[col].fillna(0)
        X_val[col] = X_val[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)
    # Fill missing categorical features with a placeholder
    for col in cat_features:
        X_train[col] = X_train[col].fillna("missing")
        X_val[col] = X_val[col].fillna("missing")
        X_test[col] = X_test[col].fillna("missing")
