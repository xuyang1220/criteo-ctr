from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


@dataclass
class Metrics:
    auc: float
    logloss: float


def compute_metrics(y_true, y_pred_proba) -> Metrics:
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    return Metrics(
        auc=float(roc_auc_score(y_true, y_pred_proba)),
        logloss=float(log_loss(y_true, y_pred_proba)),
    )
