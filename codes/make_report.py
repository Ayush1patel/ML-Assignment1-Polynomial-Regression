"""Builds report/BT2024171_report.pdf (4 pages). Needs report/metrics.json (train_predict.py) and report/cv_grid_var*.csv (cv_grid.py)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA, PRED, REPORT = ROOT/'data', ROOT/'predictions', ROOT/'report'
import json, textwrap
from math import comb
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LogNorm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

ROLL = "BT2024171"
REPO = "https://github.com/Ayush1patel/ML-Assignment1-Polynomial-Regression"
NAVY, BLUE, ORANGE, GREY, LIGHT = "black", "#222222", "#666666", "#555555", "#eeeeee"  # black and white theme
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

m = json.load(open(REPORT/"metrics.json"))
tr = {v: pd.read_csv(DATA/f"{ROLL}_train_var{v}.csv") for v in (1, 2)}
te = {v: pd.read_csv(DATA/f"{ROLL}_test_var{v}.csv") for v in (1, 2)}
pr = {v: pd.read_csv(PRED/f"{ROLL}_pred_var{v}.csv")["y"].values for v in (1, 2)}
grid = {v: pd.read_csv(REPORT/f"cv_grid_var{v}.csv") for v in (1, 2)}
CFG = {v: m[str(v)] for v in (1, 2)}
COL = {1: BLUE, 2: ORANGE}

# out-of-fold predictions of the final models (same CV split as train_predict.py)
oof = {}
for v in (1, 2):
    X, y = tr[v].drop(columns="y").values, tr[v]["y"].values
    P = PolynomialFeatures(CFG[v]["degree"], include_bias=False).fit_transform(X)
    oof[v] = cross_val_predict(Ridge(alpha=CFG[v]["alpha"]), P, y, cv=KFold(5, shuffle=True, random_state=0))


def new_page(title, sub):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.972, title, color="black", fontsize=16, weight="bold", va="center")
    fig.text(0.06, 0.949, sub, color=GREY, fontsize=8.5, va="center")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.937, 0.937], color="black", lw=1.0))
    fig.text(0.5, 0.012, f"{ROLL}  |  Polynomial Regression  |  {REPO}", fontsize=6.5, color=GREY, ha="center")
    return fig


def head(fig, x, y, s):
    fig.text(x, y, s, fontsize=11.5, weight="bold", color=NAVY, va="top")
    fig.add_artist(plt.Line2D([x, x + 0.88], [y - 0.0165, y - 0.0165], color="black", lw=0.6))


def para(fig, x, y, s, width=104, size=8.6, color="#222222", ls=1.5):
    t = "\n".join(textwrap.fill(p, width) if p else "" for p in s.split("\n"))
    fig.text(x, y, t, fontsize=size, va="top", color=color, linespacing=ls)


def card(fig, x, y, w, h, big, small, color=BLUE):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01", transform=fig.transFigure,
                                  fc=LIGHT, ec="#888888", lw=0.8))
    fig.text(x + w/2, y + h*0.62, big, ha="center", va="center", fontsize=15, weight="bold", color=color)
    fig.text(x + w/2, y + h*0.22, small, ha="center", va="center", fontsize=7.2, color=GREY)


def table(fig, x, y, colw, rows, rh=0.0215, size=8):
    for i, r in enumerate(rows):
        yy = y - i*rh
        if i == 0:
            fig.add_artist(plt.Rectangle((x, yy - rh*0.75), sum(colw), rh, transform=fig.transFigure, color="#d9d9d9"))
        cx = x
        for j, c in enumerate(r):
            fig.text(cx + 0.008, yy - rh*0.25, str(c), fontsize=size, va="center",
                     color="black", weight="bold" if (i == 0 or j == 0) else "normal")
            cx += colw[j]


def ax_at(fig, l, b, w, h):
    ax = fig.add_axes([l, b, w, h]); ax.tick_params(labelsize=7); ax.grid(alpha=.25)
    return ax


def nterms(k, d):
    return comb(k + d, d) - 1


def clipped(df):
    return (df.abs() >= 1).values.mean()


pages = []

# ------------------------------------------------------------------ page 1
f = new_page("Polynomial Regression: Assignment 1",
             f"Roll no. {ROLL}   |   Phase 1: turbine power score (var1)   |   Phase 2: thermal reservoir map (var2)")
card(f, 0.06, 0.835, 0.205, 0.07, f"d = {CFG[1]['degree']}", "var1 chosen degree")
card(f, 0.285, 0.835, 0.205, 0.07, f"R² = {CFG[1]['cv_r2']:.3f}", f"var1 5-fold CV  (MSE {CFG[1]['cv_mse']:.3f})")
card(f, 0.51, 0.835, 0.205, 0.07, f"d = {CFG[2]['degree']}", "var2 chosen degree", ORANGE)
card(f, 0.735, 0.835, 0.205, 0.07, f"R² = {CFG[2]['cv_r2']:.3f}", f"var2 5-fold CV  (MSE {CFG[2]['cv_mse']:.3f})", ORANGE)

head(f, 0.06, 0.80, "1. Problem and data")
para(f, 0.06, 0.775, "The task is to predict a continuous target y from the inputs using only polynomial regression, and to choose "
     "the polynomial degree carefully so the model neither underfits nor overfits. Two personalised datasets are used "
     "(the files matching roll number BT2024171). A polynomial of degree d contains every term whose powers of the "
     "features add up to at most d.")
rows = [["", "var1 (turbine)", "var2 (reservoir)"],
        ["Features", "6 (x1..x6)", "3 (x1..x3)"],
        ["Max degree allowed", "10", "20"],
        ["Train / test rows", f"{len(tr[1])} / {len(te[1])}", f"{len(tr[2])} / {len(te[2])}"],
        ["Missing values", "0", "0"],
        ["Target y range (train)", f"{tr[1].y.min():.1f} to {tr[1].y.max():.1f}", f"{tr[2].y.min():.1f} to {tr[2].y.max():.1f}"],
        ["Target mean / std (train)", f"{tr[1].y.mean():.2f} / {tr[1].y.std():.2f}", f"{tr[2].y.mean():.2f} / {tr[2].y.std():.2f}"],
        ["Terms at max degree", f"{nterms(6, 10)}", f"{nterms(3, 20)}"]]
table(f, 0.06, 0.700, [0.30, 0.29, 0.29], rows)

head(f, 0.06, 0.515, "2. Method")
para(f, 0.06, 0.490, "Each input vector x is expanded into all monomials of total degree at most d, phi(x) = [x1, x2, ..., x1^2, "
     "x1*x2, ..., xk^d], with k = 6 (var1) or k = 3 (var2). The model is  y_hat = b + w . phi(x).  The number of terms is "
     "C(k+d, d) - 1, which grows very fast: 461 terms for var1 at d = 5 but 8007 at d = 10; 454 for var2 at d = 12 but 1770 "
     "at d = 20. Training has only 1000 rows, so high degrees have more parameters than data points, and because the inputs "
     "lie in [-1, 1] the high powers are nearly collinear.")
para(f, 0.06, 0.385, "To keep the fit stable we minimise the least-squares error plus an L2 (ridge) penalty on the weights "
     "(the intercept b is not penalised):")
f.text(0.5, 0.335, r"$\min_{b,\,w}\;\sum_{i=1}^{n}\left(y_i-b-w^{\top}\phi(x_i)\right)^2\;+\;\alpha\,\|w\|_2^2$",
       ha="center", va="center", fontsize=12.5, color=NAVY)
para(f, 0.06, 0.305, "With alpha -> 0 this is ordinary polynomial least squares; larger alpha shrinks the coefficients and "
     "reduces variance. The model is still a polynomial in the inputs. Nothing else (no other model family, no feature "
     "scaling, no extra preprocessing) was used, because the data are already clean.")

head(f, 0.06, 0.22, "3. Model selection protocol")
para(f, 0.06, 0.195, "For every degree d (1..10 for var1, 1..20 for var2) and every alpha in {1e-6, 1e-4, 1e-2, 0.1, 1, 3.16, "
     "10, 100} we measured the mean squared error of 5-fold cross-validation (shuffled, fixed seed) on the training set "
     "only. The (d, alpha) pair with the lowest CV MSE was taken; when several degrees were statistically tied, the lowest "
     "was preferred (simpler model, safer to extrapolate). A finer alpha grid with 3x repeated CV was used to confirm the "
     "close calls. The test files were never used to fit or tune anything.")
pages.append(f)

# ------------------------------------------------------------------ page 2
f = new_page("Choosing the degree and the penalty", "5-fold cross-validated MSE on the training data")
head(f, 0.06, 0.915, "4. Degree selection curves (best alpha at each degree)")
for i, v in enumerate((1, 2)):
    g = grid[v]
    best = g.groupby("degree").cv_mse.min()
    un = g[g.alpha == 1e-6].set_index("degree").cv_mse
    ax = ax_at(f, 0.09 + 0.47*i, 0.715, 0.38, 0.14)
    ax.semilogy(best.index, best.values, "o-", ms=3.5, color=COL[v], label="best alpha (ridge)")
    ax.semilogy(un.index, un.values, "s--", ms=3, color="#aaaaaa", lw=1, label="alpha = 1e-6 (almost unpenalised)")
    ax.axvline(CFG[v]["degree"], color="black", ls=":", lw=1.3)
    ax.set_ylim(top=min(2000, un.max()*1.5))
    ax.set_title(f"var{v}: chosen d = {CFG[v]['degree']}", fontsize=9, color=NAVY, weight="bold")
    ax.set_xlabel("polynomial degree", fontsize=7.5); ax.set_ylabel("CV MSE (log scale)", fontsize=7.5)
    if i == 0:
        ax.legend(fontsize=6.3, loc="upper right")

head(f, 0.06, 0.685, "5. Joint search over degree and alpha (CV MSE, log colour scale)")
for i, v in enumerate((1, 2)):
    g = grid[v]
    pv = g.pivot(index="alpha", columns="degree", values="cv_mse")
    ax = f.add_axes([0.09 + 0.47*i, 0.49, 0.33, 0.155]); ax.tick_params(labelsize=6.5)
    im = ax.imshow(pv.values, aspect="auto", origin="lower", cmap="gray_r", norm=LogNorm(vmin=0.2, vmax=100))
    step = 1 if v == 1 else 3
    ax.set_xticks(range(0, len(pv.columns), step)); ax.set_xticklabels(pv.columns[::step])
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels([f"{a:.3g}" for a in pv.index])
    bi = list(pv.index).index(min(pv.index, key=lambda a: abs(a - CFG[v]["alpha"])))
    bj = list(pv.columns).index(CFG[v]["degree"])
    ax.plot(bj, bi, "*", color="white", ms=11, mec="black")
    ax.set_title(f"var{v}  (star = chosen)", fontsize=9, color=NAVY, weight="bold")
    ax.set_xlabel("degree", fontsize=7.5); ax.set_ylabel("alpha", fontsize=7.5)
    cb = f.colorbar(im, ax=ax, pad=0.02, fraction=0.05); cb.ax.tick_params(labelsize=6)

head(f, 0.06, 0.445, "6. What the curves show")
b1 = grid[1].groupby("degree").cv_mse.min(); b2 = grid[2].groupby("degree").cv_mse.min()
u1 = grid[1][(grid[1].degree == 10) & (grid[1].alpha == 1e-6)].cv_mse.iloc[0]
u2 = grid[2][(grid[2].degree == 16) & (grid[2].alpha == 1e-6)].cv_mse.iloc[0]
para(f, 0.06, 0.42,
     f"var1 (up to degree 10). CV MSE falls from {b1[1]:.2f} at d = 1 to {b1[5]:.3f} at d = 5: below 5 the polynomial is too "
     f"simple (underfitting; the error at d = 3 is still {b1[3]:.2f}). After d = 5 the error rises again ({b1[6]:.3f} at d = 6, "
     f"{b1[10]:.3f} at d = 10) because the 8000-term model starts to fit noise (overfitting). d = 5 and d = 6 are tied in "
     "repeated CV (0.433 vs 0.436), so d = 5 was chosen. The best alpha is moderate (about 3) and grows with degree "
     "(about 10 for d >= 7): the richer the model, the stronger the penalty it needs.\n\n"
     f"var2 (up to degree 20). The error drops steeply until d = 8 ({b2[8]:.3f}) and then stays on a plateau of about 0.23 to "
     "0.27 up to d = 20. Repeated CV with a finer alpha grid gave 0.2375 (d = 10), 0.2290 (d = 12) and 0.2295 (d = 14): "
     "statistically equivalent, so the lowest, d = 12, was selected. The curve is flat rather than U-shaped only because of "
     f"the ridge penalty; the grey dashed line shows what happens without it (CV MSE {u2:.0f} at d = 16 for var2 and {u1:.1f} "
     "at d = 10 for var1).\n\n"
     "Why not simply the highest degree? Once the plateau is reached, a higher degree is not rewarded; it only adds "
     "parameters and extrapolates worse (see Section 10). The lowest adequate degree is the more defensible choice.")
pages.append(f)

# ------------------------------------------------------------------ page 3
f = new_page("Results", "Cross-validated performance of the final models and fit quality")
head(f, 0.06, 0.915, "7. Final models")
rows = [["Metric", "var1", "var2"],
        ["Polynomial degree", CFG[1]["degree"], CFG[2]["degree"]],
        ["Ridge alpha", f"{CFG[1]['alpha']:.2f}", f"{CFG[2]['alpha']:.2f}"],
        ["Number of terms (excl. intercept)", CFG[1]["n_terms"], CFG[2]["n_terms"]],
        ["Train MSE / R²", f"{CFG[1]['train_mse']:.4f} / {CFG[1]['train_r2']:.4f}", f"{CFG[2]['train_mse']:.4f} / {CFG[2]['train_r2']:.4f}"],
        ["5-fold CV MSE / R²", f"{CFG[1]['cv_mse']:.4f} / {CFG[1]['cv_r2']:.4f}", f"{CFG[2]['cv_mse']:.4f} / {CFG[2]['cv_r2']:.4f}"]]
table(f, 0.06, 0.885, [0.36, 0.26, 0.26], rows)
para(f, 0.06, 0.755, "MSE = mean((y - y_hat)^2) and R² = 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2). Train metrics come from "
     "fitting on all 1000 rows; CV metrics use out-of-fold predictions, so they estimate performance on unseen points from "
     f"the same distribution. The gap between train and CV MSE ({CFG[1]['train_mse']:.2f} vs {CFG[1]['cv_mse']:.2f} and "
     f"{CFG[2]['train_mse']:.2f} vs {CFG[2]['cv_mse']:.2f}) is the expected optimism of in-sample fitting and is small, so the "
     "regularised models are not badly overfit.")
head(f, 0.06, 0.68, "8. Out-of-fold predictions vs. actual values")
for i, v in enumerate((1, 2)):
    y = tr[v].y.values
    ax = ax_at(f, 0.09 + 0.47*i, 0.49, 0.37, 0.14)
    ax.scatter(y, oof[v], s=5, alpha=.45, color=COL[v], lw=0)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color="black", lw=1, label="ideal: y_hat = y")
    ax.set_title(f"var{v}: CV R² = {CFG[v]['cv_r2']:.4f}", fontsize=9, color=NAVY, weight="bold")
    ax.set_xlabel("actual y", fontsize=7.5); ax.set_ylabel("predicted y (out-of-fold)", fontsize=7.5)
    if i == 0:
        ax.legend(fontsize=6.5, loc="upper left")
head(f, 0.06, 0.45, "9. Residuals and test-set predictions")
for i, v in enumerate((1, 2)):
    res = tr[v].y.values - oof[v]
    ax = ax_at(f, 0.09 + 0.47*i, 0.255, 0.37, 0.15)
    ax.hist(res, bins=40, color=COL[v], alpha=.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_title(f"var{v} residuals: mean {res.mean():.3f}, std {res.std():.3f}", fontsize=8.5, color=NAVY, weight="bold")
    ax.set_xlabel("y - y_hat (out-of-fold)", fontsize=7.5); ax.set_ylabel("count", fontsize=7.5)
para(f, 0.06, 0.205,
     "The scatter plots follow the diagonal over the whole target range, and the residuals are centred near zero with no "
     "obvious structure, which suggests the polynomial family captures the underlying surface and the remaining error behaves "
     f"like noise. The residual std ({(tr[1].y.values - oof[1]).std():.2f} for var1, {(tr[2].y.values - oof[2]).std():.2f} for var2) "
     "is the error level to expect.\n\n"
     "Test predictions (1000 rows each) were produced by refitting each model on all 1000 training rows. Their ranges are "
     f"var1: {pr[1].min():.1f} to {pr[1].max():.1f} (train y: {tr[1].y.min():.1f} to {tr[1].y.max():.1f}) and "
     f"var2: {pr[2].min():.1f} to {pr[2].max():.1f} (train y: {tr[2].y.min():.1f} to {tr[2].y.max():.1f}). They are in the "
     "same range as the training targets; no wild extrapolation values appear.")
pages.append(f)

# ------------------------------------------------------------------ page 4
f = new_page("Discussion, limitations and reproducibility",
             "Distribution shift between train and test inputs, and how the code is organised")
head(f, 0.06, 0.915, "10. Train vs. test input distribution")
for i, v in enumerate((1, 2)):
    cols = list(te[v].columns)
    ax = ax_at(f, 0.09 + 0.47*i, 0.725, 0.37, 0.15)
    xs = np.arange(len(cols)); w = 0.38
    ax.bar(xs - w/2, [tr[v][c].std() for c in cols], w, color=COL[v], label="train std")
    ax.bar(xs + w/2, [te[v][c].std() for c in cols], w, color="#aaaaaa", label="test std")
    ax.set_xticks(xs); ax.set_xticklabels(cols, fontsize=7); ax.set_ylim(0, 1.2)
    ax.set_title(f"var{v}: values clipped at ±1: train {clipped(tr[v].drop(columns='y')):.0%}, test {clipped(te[v]):.0%}",
                 fontsize=7.8, color=NAVY, weight="bold")
    ax.set_ylabel("std of feature", fontsize=7.5); ax.legend(fontsize=6.5, loc="upper right", ncol=2)
para(f, 0.06, 0.685,
     "All features lie in [-1, 1] and a sizeable share of values sits exactly on the boundary (the data are clipped). For var1 "
     "the test features are clearly more spread out than the training features (std about 0.82 against 0.71) and have more "
     f"boundary values ({clipped(te[1]):.0%} against {clipped(tr[1].drop(columns='y')):.0%}), so many test points sit nearer "
     "the edges and corners of the input cube, where there are fewer training samples. For var2 the shift is small. This "
     "matters because polynomials extrapolate badly: a high-degree term such as x^10 grows steeply near the edge, so tiny "
     "coefficient errors become large prediction errors. Cross-validation on the training data cannot measure this, so the "
     "test error is likely somewhat higher than the CV numbers above, particularly for var1.")
head(f, 0.06, 0.545, "11. Why these choices")
para(f, 0.06, 0.52,
     "- Degree by cross-validation, not by training error. Training error keeps decreasing with degree; the CV error shows "
     "where the model stops generalising.\n"
     "- Ridge penalty. Needed because the polynomial has far more terms than samples and strongly correlated columns; it "
     "makes the fit stable and keeps predictions bounded between sample points.\n"
     "- Lowest adequate degree. Among tied degrees the smallest was picked, which lowers variance and reduces the risk when "
     "extrapolating towards the borders where the test inputs lie.\n"
     "- No test-set tuning. Test inputs were used only to produce predictions.\n"
     "- Limitations: one selection protocol (5-fold CV) was used, so the choice among near-tied degrees has some noise; a "
     "neighbouring degree could perform equally well. The same penalty is applied to all terms (no per-degree weighting), a "
     "simple choice that worked well here.")
head(f, 0.06, 0.345, "12. Reproducibility and deliverables")
para(f, 0.06, 0.32, f"GitHub repository (all training and inference code): {REPO}", width=150, color="black")
rows = [["Path in repository", "Content"],
        ["codes/cv_explore.py, cv_refine.py", "degree / alpha search by CV (coarse and fine)"],
        ["codes/cv_grid.py", "full CV grid used for the figures in this report"],
        ["codes/train_predict.py", "final fit on all training rows + writes the predictions"],
        ["codes/make_report.py", "builds this PDF"],
        ["predictions/BT2024171_pred_var1.csv", "1000 predicted y values for var1 test (column y)"],
        ["predictions/BT2024171_pred_var2.csv", "1000 predicted y values for var2 test (column y)"],
        ["data/, report/, docs/", "provided datasets, this report + metrics, assignment text"]]
table(f, 0.06, 0.29, [0.37, 0.51], rows, rh=0.0225, size=7.6)
para(f, 0.06, 0.105, "Run:  pip install -r requirements.txt  then  python codes/train_predict.py  (all scripts take their paths "
     "from the repository root, so they can be run from any directory). Fixed random seeds are used for cross-validation, so "
     "the numbers in this report reproduce exactly. The prediction files follow the sample submission format: a single column "
     "named y, one row per test row, in the same order as the test file.")
pages.append(f)

with PdfPages(REPORT/f"{ROLL}_report.pdf") as pdf:
    for p in pages:
        pdf.savefig(p)
png = REPORT/"_png"
if png.exists():  # optional: preview images when the folder exists
    for i, p in enumerate(pages):
        p.savefig(png/f"p{i+1}.png", dpi=75)
