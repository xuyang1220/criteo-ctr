import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import SGDClassifier

def process_features(X_train, X_val, num_features, cat_features):
    # -----------------------------
    # 1. Preprocess categorical features
    # -----------------------------
    # Baseline LabelEncoder
    # cat_cols = X.select_dtypes(include='object').columns
    # for col in cat_cols:
    #     X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # 1. Add column name and do feature hashing.
    def row_to_features(row):
        return [f"{col}={row[col]}" for col in cat_features]
    
    train_strings = X_train.apply(row_to_features, axis=1)
    val_strings = X_val.apply(row_to_features, axis=1)

    # Init hasher
    hasher = FeatureHasher(n_features=2**20, input_type='string')
    X_train_hash = hasher.transform(train_strings)
    X_val_hash = hasher.transform(val_strings)
    
    # 2. Frequency encoding
    def frequency_encode_column(train_col, val_col, alpha=10):
        freq_map = train_col.value_counts(normalize=True)
        smoothed = freq_map / (freq_map + alpha)
    
        train_encoded = train_col.map(smoothed)
        val_encoded = val_col.map(smoothed).fillna(0)
    
        return train_encoded, val_encoded, smoothed
    
    freq_maps = {}
    
    X_train_freq = X_train.copy()
    X_val_freq = X_val.copy()
    
    for col in cat_features:
        X_train_freq[col], X_val_freq[col], freq_map = frequency_encode_column(
            X_train_freq[col], X_val_freq[col]
        )
        freq_maps[col] = freq_map
    
    # 3. We combine:
    #    Frequency encoded categorical
    #    Scaled numerical
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_features])
    X_val_num = scaler.transform(X_val[num_features])
    
    # X_train_dense = csr_matrix(X_train_num)
    # X_val_dense = csr_matrix(X_val_num)
    X_train_dense = csr_matrix(
        np.hstack([X_train_freq[cat_features].values, X_train_num])
    )
    
    X_val_dense = csr_matrix(
        np.hstack([X_val_freq[cat_features].values, X_val_num])
    )
    
    # 4. Combine Sparse + Dense
    X_train_final = hstack([X_train_hash, X_train_dense])
    X_val_final = hstack([X_val_hash, X_val_dense])
    # X_train_final = hstack([X_train_dense])
    # X_val_final = hstack([X_val_dense])

    return X_train_final, X_val_final
