from __future__ import annotations

from typing import Iterator, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_stratified_kfold(n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def iter_folds(
    X: pd.DataFrame,
    y: pd.Series,
    skf: StratifiedKFold,
) -> Iterator[Tuple[int, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Yields: fold_id, X_train, X_val, y_train, y_val
    """
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        yield fold, X_train, X_val, y_train, y_val
