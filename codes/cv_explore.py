from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA, PRED, REPORT = ROOT/'data', ROOT/'predictions', ROOT/'report'
import numpy as np, pandas as pd, sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
v=int(sys.argv[1]); maxd=int(sys.argv[2])
d=pd.read_csv(DATA/f'BT2024171_train_var{v}.csv'); X=d.drop(columns='y').values; y=d.y.values
alphas=[1e-8,1e-6,1e-4,1e-2,1,100]
kf=KFold(5,shuffle=True,random_state=0)
for deg in range(1,maxd+1):
    P=PolynomialFeatures(deg,include_bias=False).fit_transform(X)
    res={}
    for a in alphas:
        mse=[]
        for tr,va in kf.split(P):
            m=Ridge(alpha=a).fit(P[tr],y[tr]); mse.append(np.mean((m.predict(P[va])-y[va])**2))
        res[a]=np.mean(mse)
    b=min(res,key=res.get); print(deg,P.shape[1],'best alpha',b,'cvMSE %.5f'%res[b],{k:round(x,4) for k,x in res.items()},flush=True)
