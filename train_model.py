import numpy as np
import pandas as pd
import lightgbm as lgb
import src.data as data 
import src.features as features
import src.metrics as metrics 

#Reload all the modules to make sure the changes to the dependency files are always live.
# importlib.reload(data)
# importlib.reload(features)

from sklearn.metrics import roc_auc_score, log_loss

# --------------------------
# 1) Load Criteo train.txt (no header)
# label + 13 numerical (I1..I13) + 26 categorical (C1..C26)
# --------------------------

nrows =8000000           # <-- set None for full file (can be huge)
seed = 42

(num_features, 
    cat_features, 
    columns, 
    X_train_raw, 
    X_val_raw, 
    X_test_raw, 
    y_train, 
    y_val, 
    y_test) = data.load_data(
    data_path = "../criteo-deepfm-ctr/data/criteo/train.txt", data_size = nrows, train_eval_random_state = seed)

# print(f"X_train_raw shape: {X_train_raw.shape}")
# print(X_train_raw.dtypes.value_counts())
# print("GB:", X_train_raw.memory_usage(deep=True).sum() / 1024**3)

# --------------------------
# 2) Label-encode categoricals (fit on TRAIN only)
#    Unseen categories in val -> -1
# --------------------------
# encoders = {}
# for c in cat_features:
#     uniq = X_train[c].unique()
#     encoders[c] = {v: i for i, v in enumerate(uniq)}
#     X_train[c] = X_train[c].map(encoders[c]).astype(np.int32)
#     X_val[c] = X_val[c].map(encoders[c]).fillna(-1).astype(np.int32)


# 3) Add numeric engineered features
X_train_raw, X_val_raw, X_test_raw, num_eng_cols = features.add_num_log_and_missing(
    X_train_raw, X_val_raw, X_test_raw, num_features
)

# 4.1) Add categorical frequency features (uses TRAIN distribution only)
X_train_raw, X_val_raw, X_test_raw, freq_cols = features.add_cat_frequency(
    X_train_raw, X_val_raw, X_test_raw, cat_features
)

# 4.2) Add categorical log frequency features (uses TRAIN distribution only)
X_train_raw, X_val_raw, X_test_raw, freq_cols = features.add_log_freq(
    X_train_raw, X_val_raw, X_test_raw, cat_features
)

# 5) Fill missing values.
data.fill_missing_values(X_train_raw, X_val_raw, X_test_raw, num_features, cat_features)
X_train = X_train_raw
X_val = X_val_raw
X_test = X_test_raw

# 6) Convert cat_features to hashes
features.convert_cat_to_hash(X_train, cat_features)
features.convert_cat_to_hash(X_val, cat_features)

# 7) Feature crossing on already hashed features
X_train, X_val, X_test, I11_bin_col, I11_edges = features.add_quantile_bin(X_train, X_val, X_test, "I11", n_bins=64)
HASH_BINS = 1 << 20  # 1,048,576

# C10 × C17
X_train, X_val, X_test, cross1 = features.add_hashed_cross(
    X_train, X_val, X_test,
    "C10", "C17",
    new_name="C10xC17",
    num_bins=HASH_BINS
)

# C10 × I11_bin
X_train, X_val, X_test, cross2 = features.add_hashed_cross(
    X_train, X_val, X_test,
    "C10", I11_bin_col,
    new_name="C10xI11bin",
    num_bins=HASH_BINS
)

# --------------------------
# 8) LightGBM datasets
# --------------------------
new_cat_cols = [cross1, cross2, I11_bin_col]     # treat bins as categorical
new_num_cols = freq_cols                      # numeric

feature_cols = list(X_train.columns)  # or explicitly build: num_eng_cols + cat_cols + freq_cols + logfreq_cols + [I11_bin_col, cross1, cross2]
all_cat_features = cat_features + new_cat_cols

# downcast float64 -> float32
float64_cols = X_train.select_dtypes(include=["float64"]).columns
X_train[float64_cols] = X_train[float64_cols].astype(np.float32)
X_val[float64_cols] = X_val[float64_cols].astype(np.float32)

# downcast int64 -> int32 (or int16/int8 if safe)
int64_cols = X_train.select_dtypes(include=["int64"]).columns
X_train[int64_cols] = X_train[int64_cols].astype(np.int32)
X_val[int64_cols] = X_val[int64_cols].astype(np.int32)

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=all_cat_features, free_raw_data=True)
dval = lgb.Dataset(X_val, label=y_val, categorical_feature=all_cat_features, reference=dtrain, free_raw_data=True)

print(f"type(X_train): {type(X_train)}")      
print(X_train.shape)        
print(X_train.dtypes)
print(list(X_train.columns)[:5])
from scipy import sparse
print(sparse.issparse(X_train))   
bank = X_train.sample(n=500_000, random_state=7)  # or X_val
bank.to_parquet("artifacts/feature_bank.parquet", index=False)     

del X_train, y_train
import gc; gc.collect()

# --------------------------
# 9) Minimal params for CTR
# --------------------------
params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.03,
    "max_bin": 127,          # 255 -> 127 saves memory
    "num_leaves": 63,        # reduce complexity
    "min_data_in_leaf": 50,  # reduces tree size + histogram usage
    "num_threads": 4,        # lowers per-thread buffers
    "max_depth": 12,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10,
    "verbosity": -1,
    "seed": 42,
}

# --------------------------
# 10) Train with early stopping
# --------------------------

# model = lgb.train(
#     params,
#     dtrain,
#     num_boost_round=2000,
#     valid_sets=[dval],
#     valid_names=["val"],
#     callbacks=[
#         lgb.early_stopping(stopping_rounds=50),
#         lgb.log_evaluation(period=50),
#     ],
# )

# --------------------------
# 11) Evaluate
# --------------------------
# p_val = model.predict(X_val, num_iteration=model.best_iteration)

# print(metrics.metrics_basic(y_val, p_val))

# ece, ece_table = metrics.expected_calibration_error(y_val, p_val, n_bins=15)
# print("ECE:", ece)

# print(metrics.topk_lift(y_val, p_val, k=0.01))   # top 1%
# print(metrics.topk_lift(y_val, p_val, k=0.05))   # top 5

# booster = model.booster_ if hasattr(model, "booster_") else model
# print(type(booster))
# print("num_features:", booster.num_feature())
# print("feature_name sample:", booster.feature_name()[:5])

# # Save the model to disk
# model.save_model("models/lgb_ctr_model_8M.txt")