from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA, PRED, REPORT = ROOT/'data', ROOT/'predictions', ROOT/'report'
import json, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

m = json.load(open(REPORT/"metrics.json"))
# best 5-fold CV MSE per degree over alpha in {1e-8,1e-6,1e-4,1e-2,1,100} (output of cv_explore.py)
cv1 = {1:9.0546,2:2.6500,3:0.9101,4:0.5847,5:0.4287,6:0.4720,7:0.5512,8:0.6572,9:0.7531,10:0.7584}
cv2 = {1:33.962,2:21.237,3:10.041,4:3.297,5:1.230,6:0.551,7:0.357,8:0.257,9:0.260,10:0.244,11:0.246,12:0.250,
       13:0.256,14:0.251,15:0.253,16:0.251,17:0.257,18:0.259,19:0.267,20:0.271}

def page(pdf, title, body, fig_fn=None):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.95, title, fontsize=16, weight="bold", va="top")
    fig.text(0.08, 0.91, body, fontsize=8.5, va="top", family="monospace", linespacing=1.45)
    if fig_fn: fig_fn(fig)
    pdf.savefig(fig); plt.close(fig)

def w(s): return "\n".join(textwrap.fill(p, 92) if p and not p.startswith(("  ", "$")) else p for p in s.split("\n"))

intro = w(f"""Roll number: BT2024171.   Models: polynomial regression (ridge-regularised), built with scikit-learn.
GitHub repository (all training and inference code): https://github.com/Ayush1patel/ML-Assignment1-Polynomial-Regression

1. Approach
For each problem the features are expanded to all monomials of total degree <= d (PolynomialFeatures, no constant column; the intercept is fitted by Ridge). The model is y = w^T phi(x) + b. Parameters minimise
  sum_i (y_i - w^T phi(x_i) - b)^2 + alpha ||w||^2 ,
i.e. ordinary least-squares polynomial regression plus an L2 (ridge) penalty alpha (the intercept is not penalised). The only inputs are the polynomial terms of x1..xk, so this is purely polynomial regression.

2. Why a penalty was needed
With 1000 training samples, a degree-d polynomial has C(k+d, d) - 1 terms: 8007 for var1 at d=10 and 1770 for var2 at d=20, far more than the samples. Even at moderate degree the design matrix is badly conditioned (inputs lie in [-1, 1], so high powers are nearly collinear). Unpenalised least squares gave huge CV errors at high degree (e.g. var2, d=16, alpha=1e-8: CV MSE 255). The penalty alpha is therefore chosen together with the degree.

3. Degree / alpha selection
For every degree d in 1..10 (var1) and 1..20 (var2), alpha was chosen from a grid by 5-fold cross-validation on the training set (shuffled, fixed seed), and the CV MSE of the best alpha was recorded. The promising degrees were then re-checked with a finer alpha grid (1e-4 .. 1e2, 13 values) and 3x repeated 5-fold CV. The test files are never used for fitting or selection. Rule: take the lowest degree whose CV MSE is within noise of the minimum (simpler model, safer on the test inputs, see section 5).""")

res = w(f"""4. Results
{'':<20}{'var1 (6 features)':<20}var2 (3 features)
  {'chosen degree':<18}{m['1']['degree']:<20}{m['2']['degree']}
  {'ridge alpha':<18}{m['1']['alpha']:<20.3f}{m['2']['alpha']:.3f}
  {'number of terms':<18}{m['1']['n_terms']:<20}{m['2']['n_terms']}
  {'train MSE':<18}{m['1']['train_mse']:<20.4f}{m['2']['train_mse']:.4f}
  {'train R^2':<18}{m['1']['train_r2']:<20.4f}{m['2']['train_r2']:.4f}
  {'5-fold CV MSE':<18}{m['1']['cv_mse']:<20.4f}{m['2']['cv_mse']:.4f}
  {'5-fold CV R^2':<18}{m['1']['cv_r2']:<20.4f}{m['2']['cv_r2']:.4f}

var1: CV MSE falls from 9.05 (d=1) to 0.43 at d=5, then rises again (0.47 at d=6, 0.76 at d=10) as the model overfits (underfitting below 5, overfitting above). d=5 and d=6 are tied in the repeated CV (0.433 vs 0.436); d=5 was chosen as the simpler model. The task statement says the true degree is at most 10; the CV minimum at 5 is consistent with that.

var2: CV MSE drops steeply to about 0.25 by d=8 and then stays flat (0.23-0.27) up to d=20; the true degree is at most 20. Repeated CV with finer alpha gave 0.2375 (d=10), 0.2290 (d=12), 0.2295 (d=14); these are statistically equivalent, so the lowest, d=12, was used. Higher degrees give no gain, and the flat curve suggests the remaining error (about 0.23) is mostly irreducible noise.

5. Other notes
- Distribution shift: the train inputs have std about 0.71 (var1) / 0.68 (var2), but the test inputs are more spread out (std about 0.82 for var1) with more points on the clipped boundary at +-1. The model therefore partly extrapolates, where high-degree polynomials can blow up; this is a further reason for ridge and for the lowest adequate degree. Predicted test ranges (var1 -10.5..15.1, var2 -29.5..37.6) are close to the training target ranges (-10.6..14.8, -28.3..40.0), so no explosion was seen. CV on the training data cannot measure this shift, so the true test error may be somewhat higher than the CV figures.
- Data had no missing values; the files were used as given (no extra preprocessing, no feature scaling).
- Metrics: MSE = mean (y - yhat)^2 ; R^2 = 1 - sum (y - yhat)^2 / sum (y - ybar)^2.

6. Files (in the repository)
codes/train_predict.py (final training + prediction), codes/cv_explore.py and codes/cv_refine.py (degree/alpha selection), codes/make_report.py (this report), predictions/BT2024171_pred_var1.csv and predictions/BT2024171_pred_var2.csv (one column y, 1000 rows, same order as the test files).""")

def curves(fig):
    for i, (cv, t, sel) in enumerate([(cv1, "var1", 5), (cv2, "var2", 12)]):
        ax = fig.add_axes([0.1 + 0.46*i, 0.08, 0.38, 0.2])
        ax.semilogy(list(cv), list(cv.values()), "o-", ms=4)
        ax.axvline(sel, color="r", ls="--", lw=1, label=f"chosen d={sel}")
        ax.set_xlabel("degree"); ax.set_ylabel("5-fold CV MSE"); ax.set_title(t); ax.legend(); ax.grid(alpha=.3)

with PdfPages(REPORT/"BT2024171_report.pdf") as pdf:
    page(pdf, "Polynomial Regression: Assignment Report", intro)
    page(pdf, "Results and Discussion", res, curves)
