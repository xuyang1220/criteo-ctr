# CTR Baseline Notebook
# Logistic Regression on small Criteo subset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import numpy as np

# 0. add header
# -----------------------------
num_features = [f"I{i}" for i in range(1, 14)]
cat_features = [f"C{i}" for i in range(1, 27)]
columns = ["label"] + num_features + cat_features

# -----------------------------
# 1. Load small subset of dataset
# -----------------------------
# Replace with your local path to Criteo dataset
data_path = "../data/criteo/train.txt"

# Read first 100k rows to start
df = pd.read_csv(data_path, sep='\t', header=None, names=columns, nrows=100*1000)

# Fill missing numeric features with 0
for col in num_features:
    df[col] = data[col].fillna(0)


# Fill missing categorical features with a placeholder
for col in cat_features:
    df[col] = data[col].fillna("missing")

# -----------------------------
# 2. Separate target and features
# -----------------------------
y = data['label']
X = data.drop(['label'], axis=1)

print("X: ", X.head(10))
print("y: ", y.head(10))
print("data: ", data.head(10))


# -----------------------------
# 3. Preprocess categorical features
# -----------------------------
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# -----------------------------
# 4. Preprocess numerical features
# -----------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[num_cols] = StandardScaler().fit_transform(X[num_cols])

# print("X: ", X)
# print("y: ", y)


# -----------------------------
# 5. Separate
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 6. Train with warm_start
# ----------------------------
# n_epochs = 100
# train_auc_list = []
# val_auc_list = []

# clf = LogisticRegression(
#     max_iter=1,
#     warm_start=True,
#     solver="saga",
#     n_jobs=-1
# )

# train_auc_list = []
# val_auc_list = []

# train_loss_list = []
# val_loss_list = []

# for epoch in range(n_epochs):
#     clf.fit(X_train, y_train)

#     y_train_pred = clf.predict_proba(X_train)[:, 1]
#     y_val_pred = clf.predict_proba(X_val)[:, 1]

#     # AUC
#     train_auc = roc_auc_score(y_train, y_train_pred)
#     val_auc = roc_auc_score(y_val, y_val_pred)

#     # Log Loss
#     train_loss = log_loss(y_train, y_train_pred)
#     val_loss = log_loss(y_val, y_val_pred)

#     train_auc_list.append(train_auc)
#     val_auc_list.append(val_auc)

#     train_loss_list.append(train_loss)
#     val_loss_list.append(val_loss)

#     print(
#         f"Epoch {epoch+1:02d} | "
#         f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
#         f"Train LogLoss: {train_loss:.4f} | Val LogLoss: {val_loss:.4f}"
#     )

# plt.figure(figsize=(8,5))
# plt.plot(range(1, n_epochs+1), train_loss_list, label="Train LogLoss")
# plt.plot(range(1, n_epochs+1), val_loss_list, label="Validation LogLoss")
# plt.xlabel("Epoch")
# plt.ylabel("Log Loss")
# plt.title("Criteo Logistic Regression - LogLoss per Epoch")
# plt.legend()
# plt.grid(True)
# plt.show()