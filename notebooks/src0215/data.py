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

def load_data(data_path, data_size):
    # 0. add header
    # -----------------------------
    num_features = [f"I{i}" for i in range(1, 14)]
    cat_features = [f"C{i}" for i in range(1, 27)]
    columns = ["label"] + num_features + cat_features
    
    # -----------------------------
    # 1. Load small subset of dataset
    # -----------------------------
    
    # Read first 100k rows to start
    data = pd.read_csv(data_path, sep='\t', header=None, names=columns, nrows=data_size)
    
    
    # -----------------------------
    # 2. Separate target and features
    # -----------------------------
    y = data['label']
    X = data.drop(['label'], axis=1)
    
    # -----------------------------
    # 3. Train/Test Split
    # -----------------------------
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Fill missing values.
    # Fill missing numeric features with 0
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()
    for col in num_features:
        X_train[col] = X_train[col].fillna(0)
        X_val[col] = X_val[col].fillna(0)
    
    # Fill missing categorical features with a placeholder
    for col in cat_features:
        X_train[col] = X_train[col].fillna("missing")
        X_val[col] = X_val[col].fillna("missing")
    
    return num_features, cat_features, columns, X_train, X_val, X_train_raw, X_val_raw, y_train, y_val

