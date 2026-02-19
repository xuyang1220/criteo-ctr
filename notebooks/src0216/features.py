from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------- Rare handling ----------
def fit_rare_categories(X_train: pd.DataFrame, cat_features: List[str], min_count: int = 10) -> Dict[str, set]:
    rare_dict: Dict[str, set] = {}
    for col in cat_features:
        vc = X_train[col].value_counts(dropna=False)
        rare_dict[col] = set(vc[vc < min_count].index.astype(str))
    return rare_dict


def apply_rare_categories(X: pd.DataFrame, cat_features: List[str], rare_dict: Dict[str, set]) -> pd.DataFrame:
    X2 = X.copy()
    for col in cat_features:
        X2[col] = X2[col].astype(str).apply(lambda v: "__RARE__" if v in rare_dict[col] else v)
    return X2


# ---------- Frequency encoding ----------
def frequency_encode(train_col: pd.Series, val_col: pd.Series, normalize: bool = True) -> Tuple[pd.Series, pd.Series]:
    freq = train_col.value_counts(normalize=normalize, dropna=False)
    train_encoded = train_col.map(freq).astype(float)
    val_encoded = val_col.map(freq).fillna(0.0).astype(float)
    return train_encoded, val_encoded


# ---------- Hashing helpers ----------
def row_to_features(row: pd.Series, cat_features: List[str]) -> List[str]:
    return [f"{col}={row[col]}" for col in cat_features]


@dataclass
class FoldTransform:
    hasher: FeatureHasher
    scaler: StandardScaler
    rare_dict: Dict[str, set]


def fit_transform_fold(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str],
    min_count: int = 10,
    hash_dim: int = 2**18,
    rare_handling: bool = True,
    freq_encoding: bool = True,
    hashing: bool = True,
    freq_normalize: bool = True,
) -> Tuple["scipy.sparse.csr_matrix", "scipy.sparse.csr_matrix", FoldTransform]:
    """
    Fits rare_dict + scaler on train, applies to train/val.
    Hashing is stateless but we keep the hasher object for consistency.
    Returns combined sparse matrices.
    """

    # ---- Rare handling (fit on train only)
    X_train_r = X_train
    X_val_r = X_val
    rare_dict = fit_rare_categories(X_train, cat_features, min_count=min_count)
    if rare_handling:
        X_train_r = apply_rare_categories(X_train, cat_features, rare_dict)
        X_val_r = apply_rare_categories(X_val, cat_features, rare_dict)

    # ---- Frequency encoding (fit on train only)
    X_train_freq = X_train_r.copy()
    X_val_freq = X_val_r.copy()
    if freq_encoding:
        for col in cat_features:
            X_train_freq[col], X_val_freq[col] = frequency_encode(
                X_train_r[col].astype(str), X_val_r[col].astype(str), normalize=freq_normalize
            )
    # Baseline LabelEncoder
    else:
        for col in cat_features:
            X_train_freq[col] = LabelEncoder().fit_transform(X_train_freq[col].astype(str))
            X_val_freq[col] = LabelEncoder().fit_transform(X_val_freq[col].astype(str))

    # ---- Hashing (stateless)
    hasher = FeatureHasher(n_features=hash_dim, input_type="string")
    if hashing:
        train_strings = X_train_r.apply(lambda r: row_to_features(r, cat_features), axis=1)
        val_strings = X_val_r.apply(lambda r: row_to_features(r, cat_features), axis=1)
    
        X_train_hash = hasher.transform(train_strings)
        X_val_hash = hasher.transform(val_strings)

    # ---- Numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_r[num_features].astype(float)) if num_features else np.zeros((len(X_train_r), 0))
    X_val_num = scaler.transform(X_val_r[num_features].astype(float)) if num_features else np.zeros((len(X_val_r), 0))

    # ---- Dense block (freq + num) -> sparse
    X_train_dense = csr_matrix(np.hstack([X_train_freq[cat_features].values, X_train_num]))
    X_val_dense = csr_matrix(np.hstack([X_val_freq[cat_features].values, X_val_num]))

    # ---- Combine
    X_train_final = X_train_dense.tocsr()
    X_val_final = X_val_dense.tocsr()
    if hashing:
        X_train_final = hstack([X_train_hash, X_train_dense]).tocsr()
        X_val_final = hstack([X_val_hash, X_val_dense]).tocsr()

    bundle = FoldTransform(hasher=hasher, scaler=scaler, rare_dict=rare_dict)
    return X_train_final, X_val_final, bundle
