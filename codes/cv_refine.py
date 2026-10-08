from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA, PRED, REPORT = ROOT/'data', ROOT/'predictions', ROOT/'report'
import numpy as np, pandas as pd, sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
v=int(sys.argv[1]); degs=range(int(sys.argv[2]),int(sys.argv[3])+1)
d=pd.read_csv(DATA/f'BT2024171_train_var{v}.csv'); X=d.drop(columns='y').values; y=d.y.values
alphas=np.logspace(-4,2,13)
rkf=RepeatedKFold(n_splits=5,n_repeats=3,random_state=1)
for deg in degs:
    P=PolynomialFeatures(deg,include_bias=False).fit_transform(X)
    res=[]
    for a in alphas:
        res.append(np.mean([np.mean((Ridge(alpha=a).fit(P[tr],y[tr]).predict(P[va])-y[va])**2) for tr,va in rkf.split(P)]))
    i=int(np.argmin(res)); print(deg,'alpha %.4g'%alphas[i],'cvMSE %.4f'%res[i],flush=True)
