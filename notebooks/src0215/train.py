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

def train_model(X_train, y_train):
    # -----------------------------
    # Logistic Regression Baseline
    # -----------------------------
    # clf = LogisticRegression(max_iter=1, warm_start=True, solver='saga')  # saga supports warm_start & large datasets
    # clf.fit(X_train_final, y_train)
    # y_pred = clf.predict_proba(X_val_final)[:, 1]
    
    # SGDClassifier
    model = SGDClassifier(
        loss="log_loss", 
        max_iter=100, 
        #class_weight="balanced",
        alpha=0.0001
    )
    model.fit(X_train, y_train)
    return model