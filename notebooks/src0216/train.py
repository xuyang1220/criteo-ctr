from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

import src.data as data
import src.features as features 
import src.evaluate as evaluate 


def run_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features,
    num_features,
    n_splits: int = 5,
    min_count: int = 10,
    hash_dim: int = 2**18,
    alpha: float = 1e-4,
    max_iter: int = 10,
    random_state: int = 42,
    rare_handling: bool = True,
    freq_encoding: bool = True,
    hashing: bool = True,
    freq_normalize: bool = True,
    optimizer_type = 'SGD'
):
    skf = data.make_stratified_kfold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auc_scores = []
    logloss_scores = []

    for fold, X_train, X_val, y_train, y_val in data.iter_folds(X, y, skf):
        #print(f"\n===== Fold {fold} =====")

        X_train_final, X_val_final, _bundle = features.fit_transform_fold(
            X_train=X_train,
            X_val=X_val,
            cat_features=cat_features,
            num_features=num_features,
            min_count=min_count,
            hash_dim=hash_dim,
            rare_handling = rare_handling,
            freq_encoding = freq_encoding,
            hashing = hashing,
            freq_normalize = freq_normalize,
        )
        
        if optimizer_type == 'Logistic':
            model = LogisticRegression(max_iter=1, warm_start=True, solver='saga')

        else:
            model = SGDClassifier(
                loss="log_loss",
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
            )
    
        model.fit(X_train_final, y_train)

        y_val_pred = model.predict_proba(X_val_final)[:, 1]
        m = evaluate.compute_metrics(y_val, y_val_pred)

        auc_scores.append(m.auc)
        logloss_scores.append(m.logloss)

        #print(f"AUC: {m.auc:.4f}, LogLoss: {m.logloss:.4f}")

    #print("\n===== CV Summary =====")
    #print(f"AUC    : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    #print(f"LogLoss: {np.mean(logloss_scores):.4f} ± {np.std(logloss_scores):.4f}")

    return auc_scores, logloss_scores, model

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features,
    num_features,
    n_splits: int = 5,
    min_count: int = 10,
    hash_dim: int = 2**18,
    alpha: float = 1e-4,
    max_iter: int = 10,
    random_state: int = 42,
    rare_handling: bool = True,
    freq_encoding: bool = True,
    hashing: bool = True,
    freq_normalize: bool = True,
    optimizer_type = 'SGD'
):
    skf = data.make_stratified_kfold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, X_train, X_val, y_train, y_val in data.iter_folds(X, y, skf):
        #print(f"\n===== Fold {fold} =====")

        X_train_final, X_val_final, _bundle = features.fit_transform_fold(
            X_train=X_train,
            X_val=X_val,
            cat_features=cat_features,
            num_features=num_features,
            min_count=min_count,
            hash_dim=hash_dim,
            rare_handling = rare_handling,
            freq_encoding = freq_encoding,
            hashing = hashing,
            freq_normalize = freq_normalize,
        )
        
        if optimizer_type == 'Logistic':
            model = LogisticRegression(max_iter=1, warm_start=True, solver='saga')

        else:
            model = SGDClassifier(
                loss="log_loss",
                alpha=alpha,
                max_iter=max_iter,
                random_state=random_state,
            )
    
        model.fit(X_train_final, y_train)

        y_val_pred = model.predict_proba(X_val_final)[:, 1]
        m = evaluate.compute_metrics(y_val, y_val_pred)

        auc_scores.append(m.auc)
        logloss_scores.append(m.logloss)

        #print(f"AUC: {m.auc:.4f}, LogLoss: {m.logloss:.4f}")

    #print("\n===== CV Summary =====")
    #print(f"AUC    : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    #print(f"LogLoss: {np.mean(logloss_scores):.4f} ± {np.std(logloss_scores):.4f}")

    return auc_scores, logloss_scores, model

# Example usage (edit to match your dataset variables)
if __name__ == "__main__":
    # You should replace this with your own df loading
    raise SystemExit(
        "Import run_kfold_cv() in your notebook/script and call it with X, y, cat_features, num_features."
    )
