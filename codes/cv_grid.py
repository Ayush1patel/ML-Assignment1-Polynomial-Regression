"""5-fold CV MSE for every (degree, alpha) pair; saved to report/cv_grid_var<k>.csv (used for the report figures)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA, REPORT = ROOT/'data', ROOT/'report'
import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

MAXDEG = {1: 10, 2: 20}
ALPHAS = [1e-6, 1e-4, 1e-2, 1e-1, 1, 3.1623, 10, 100]
kf = KFold(5, shuffle=True, random_state=0)
for v, maxd in MAXDEG.items():
    d = pd.read_csv(DATA/f'BT2024171_train_var{v}.csv'); X = d.drop(columns='y').values; y = d.y.values
    rows = []
    for deg in range(1, maxd+1):
        P = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
        for a in ALPHAS:
            mse = np.mean([np.mean((Ridge(alpha=a).fit(P[tr], y[tr]).predict(P[va]) - y[va])**2) for tr, va in kf.split(P)])
            rows.append((deg, a, mse))
        print(v, deg, flush=True)
    pd.DataFrame(rows, columns=['degree', 'alpha', 'cv_mse']).to_csv(REPORT/f'cv_grid_var{v}.csv', index=False)
