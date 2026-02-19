import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    average_precision_score,
    brier_score_loss,
)

def safe_clip(p, eps=1e-15):
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, eps, 1 - eps)

def metrics_basic(y_true, p):
    """Core CTR metrics: ranking + proper scoring."""
    p = safe_clip(p)
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "logloss": float(log_loss(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),  # a.k.a. AP
        "brier": float(brier_score_loss(y_true, p)),          # calibration-ish
        "ctr_mean_true": float(np.mean(y_true)),
        "ctr_mean_pred": float(np.mean(p)),
    }

def expected_calibration_error(y_true, p, n_bins=15):
    """
    ECE with equal-width bins on probability.
    Returns (ece, bin_table) where bin_table is useful for debugging.
    """
    p = safe_clip(p)
    y = np.asarray(y_true)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins[1:-1], right=False)  # 0..n_bins-1

    ece = 0.0
    table = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        frac = float(np.mean(mask))
        acc = float(np.mean(y[mask]))        # empirical CTR in bin
        conf = float(np.mean(p[mask]))       # mean predicted prob in bin
        ece += frac * abs(acc - conf)
        table.append({
            "bin": b,
            "count": int(np.sum(mask)),
            "frac": frac,
            "p_mean": conf,
            "y_mean": acc,
            "abs_gap": abs(acc - conf),
        })
    return float(ece), table

def topk_lift(y_true, p, k=0.01):
    """
    Lift at top-k fraction (e.g., k=0.01 for top 1%).
    lift = (CTR in top-k) / (overall CTR)
    """
    p = np.asarray(p)
    y = np.asarray(y_true)
    n = len(y)
    m = max(1, int(np.floor(k * n)))
    idx = np.argpartition(-p, m-1)[:m]
    ctr_top = float(np.mean(y[idx]))
    ctr_all = float(np.mean(y))
    lift = ctr_top / (ctr_all + 1e-15)
    return {"k": k, "n_top": m, "ctr_top": ctr_top, "ctr_all": ctr_all, "lift": float(lift)}

def psi(p_ref, p_new, n_bins=10):
    """
    Population Stability Index between two score distributions.
    Use to compare predictions across runs/seeds/data slices.
    Rule of thumb: PSI < 0.1 stable, 0.1-0.25 shift, >0.25 big shift.
    """
    r = safe_clip(p_ref)
    n = safe_clip(p_new)

    # Use ref quantile bins so buckets are comparable
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(r, qs)
    edges[0], edges[-1] = 0.0, 1.0

    def bucket_frac(x):
        counts, _ = np.histogram(x, bins=edges)
        frac = counts / (np.sum(counts) + 1e-15)
        return np.clip(frac, 1e-8, 1.0)  # avoid log(0)

    r_f = bucket_frac(r)
    n_f = bucket_frac(n)
    return float(np.sum((n_f - r_f) * np.log(n_f / r_f)))
def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(), values.std(ddof=1)  # sample std

def print_mean_std(name, values, fmt="{:.4f}"):
    mean, std = mean_std(values)
    print(f"{name:<8}: {fmt.format(mean)} ± {fmt.format(std)}")