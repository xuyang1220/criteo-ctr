import pandas as pd
import numpy as np

# -----------------------------
# Evaluate AUC
# -----------------------------
def model_predict(model, X_val_final):
    return model.predict_proba(X_val_final)[:, 1]