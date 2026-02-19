import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import SGDClassifier
import hashlib

HASH_BITS = 20              # 2^20 ≈ 1M bins
HASH_SIZE = 1 << HASH_BITS

import numpy as np
import pandas as pd

def add_num_log_and_missing(train_df, val_df, test_df, num_cols):
    """
    For each numeric column:
      - add <col>__miss indicator (1 if original value was NaN)
      - fill NaN with 0
      - add <col>__log = log1p(filled_value)
    Returns updated dfs and the new numeric feature column list.
    """
    new_cols = []
    for c in num_cols:
        # missing flags based on original NaNs
        for df in (train_df, val_df, test_df):
            df[c + "__miss"] = df[c].isna().astype(np.int8)

        # fill + log1p
        for df in (train_df, val_df, test_df):
            filled = df[c].fillna(0).astype(np.float32)
            df[c + "__log"] = np.log1p(filled)

        new_cols.extend([c + "__log", c + "__miss"])
    return train_df, val_df, test_df, new_cols

def add_cat_frequency(train_df, val_df, test_df, cat_cols):
    """
    For each categorical column:
      - compute value_counts on TRAIN only
      - map counts to train/val/test as <col>__freq
    Returns updated dfs and the new frequency column list.
    """
    freq_cols = []
    for c in cat_cols:
        vc = train_df[c].value_counts(dropna=False)
        fname = c + "__freq"

        train_df[fname] = train_df[c].map(vc).fillna(0).astype(np.float32)
        val_df[fname]   = val_df[c].map(vc).fillna(0).astype(np.float32)
        test_df[fname]  = test_df[c].map(vc).fillna(0).astype(np.float32)

        freq_cols.append(fname)
    return train_df, val_df, test_df, freq_cols

def add_log_freq(X_train, X_val, X_test, cat_cols, suffix="__freq"):
    #log_cols = []
    for c in cat_cols:
        freq_col = c + suffix
        #log_col = c + "__logfreq"
        for X in (X_train, X_val, X_test):
            # log1p to handle zeros safely
            X[freq_col] = np.log1p(X[freq_col].astype(np.float32))
        #log_cols.append(log_col)
    return X_train, X_val, X_test, cat_cols

def hash_str(x, num_bins):
    return int(hashlib.md5(x.encode("utf-8")).hexdigest(), 16) % num_bins

def convert_cat_to_hash(X, cat_cols):
    for c in cat_cols:
        X[c] = X[c].astype(str).apply(
            lambda x: hash_str(x, HASH_SIZE)
        ).astype("int32")

def row_to_features(row):
    return [f"{col}={row[col]}" for col in cat_features]

def frequency_encode_column(train_col, val_col, alpha=10):
    freq_map = train_col.value_counts(normalize=True)
    smoothed = freq_map / (freq_map + alpha)

    train_encoded = train_col.map(smoothed)
    val_encoded = val_col.map(smoothed).fillna(0)

    return train_encoded, val_encoded, smoothed

def add_quantile_bin(train, val, test, col, n_bins=64):
    # Compute edges on TRAIN only
    x = train[col].astype(np.float32)
    # If you used log features, consider binning col+"__log" instead of raw
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    # Guard: if too few unique edges, reduce bins
    if len(edges) < 3:
        # fallback: treat as 1 bin
        edges = np.array([x.min(), x.max()], dtype=np.float32)

    def apply_bin(df):
        # digitize returns 1..len(edges)-1; shift to 0-based
        return (np.digitize(df[col].astype(np.float32), edges[1:-1], right=False)).astype(np.int32)

    bcol = col + "_bin"
    train[bcol] = apply_bin(train)
    val[bcol]   = apply_bin(val)
    test[bcol]  = apply_bin(test)
    return train, val, test, bcol, edges

def stable_hash_to_int(s, num_bins):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % num_bins

def add_hashed_cross(X_train, X_val, X_test, col_a, col_b, new_name, num_bins=1<<20):
    def cross_hash(df):
        # IMPORTANT: cast to str so it works for ints and strings
        return df[col_a].astype(str).str.cat(df[col_b].astype(str), sep="|").map(
            lambda z: stable_hash_to_int(new_name + "|" + z, num_bins)
        ).astype(np.int32)

    X_train[new_name] = cross_hash(X_train)
    X_val[new_name]   = cross_hash(X_val)
    X_test[new_name]  = cross_hash(X_test)
    return X_train, X_val, X_test, new_name

def process_features(X_train, X_val, num_features, cat_features):
    # -----------------------------
    # 1. Preprocess categorical features
    # -----------------------------
    # Baseline LabelEncoder
    # cat_cols = X.select_dtypes(include='object').columns
    # for col in cat_cols:
    #     X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # 1. Add column name and do feature hashing.
    
    train_strings = X_train.apply(row_to_features, axis=1)
    val_strings = X_val.apply(row_to_features, axis=1)

    # Init hasher
    hasher = FeatureHasher(n_features=2**20, input_type='string')
    X_train_hash = hasher.transform(train_strings)
    X_val_hash = hasher.transform(val_strings)
    
    # 2. Frequency encoding
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
