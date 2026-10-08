"""Polynomial (ridge-regularised) regression for both problems. Writes <ROLLNO>_pred_var<k>.csv and metrics.json."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA, PRED, REPORT = ROOT/'data', ROOT/'predictions', ROOT/'report'

import json
import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score

ROLL = "BT2024171"
CONFIG = {1: dict(degree=5, alpha=3.1623), 2: dict(degree=12, alpha=0.1)}  # chosen by CV (see cv_explore.py / cv_refine.py)

metrics = {}
for v, c in CONFIG.items():
    tr = pd.read_csv(DATA/f"{ROLL}_train_var{v}.csv")
    te = pd.read_csv(DATA/f"{ROLL}_test_var{v}.csv")
    X, y = tr.drop(columns="y").values, tr["y"].values
    poly = PolynomialFeatures(c["degree"], include_bias=False)
    P, Pt = poly.fit_transform(X), poly.transform(te.values)
    model = Ridge(alpha=c["alpha"])
    cv_pred = cross_val_predict(model, P, y, cv=KFold(5, shuffle=True, random_state=0))
    model.fit(P, y)
    metrics[v] = dict(**c, n_terms=P.shape[1],
                      train_mse=mean_squared_error(y, model.predict(P)), train_r2=r2_score(y, model.predict(P)),
                      cv_mse=mean_squared_error(y, cv_pred), cv_r2=r2_score(y, cv_pred))
    pd.DataFrame({"y": model.predict(Pt)}).to_csv(PRED/f"{ROLL}_pred_var{v}.csv", index=False)
    print(v, metrics[v])
json.dump(metrics, open(REPORT/"metrics.json", "w"), indent=1)
