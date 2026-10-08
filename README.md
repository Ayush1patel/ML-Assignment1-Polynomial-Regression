# Assignment 1: Polynomial Regression (BT2024171)

Polynomial regression for two personalised datasets:

- **var1** (turbine Net Power Score): 6 input features, polynomial degree up to 10
- **var2** (thermal anomaly score): 3 input features, polynomial degree up to 20

Features are expanded to all monomials of total degree at most `d`, and a linear model is fitted on those terms with an L2 (ridge) penalty:

    minimise  sum_i ( y_i - b - w . phi(x_i) )^2  +  alpha * ||w||^2

The degree `d` and the penalty `alpha` are chosen by 5-fold cross-validation on the training data only. The test files are used only to produce predictions.

## Results

| Problem | Degree | Ridge alpha | Terms | Train MSE | 5-fold CV MSE | 5-fold CV R² |
|---|---|---|---|---|---|---|
| var1 | 5 | 3.16 | 461 | 0.187 | 0.416 | 0.958 |
| var2 | 12 | 0.1 | 454 | 0.149 | 0.235 | 0.995 |

Why a penalty: with 1000 training rows, a degree-10 polynomial in 6 variables has 8007 terms and a degree-20 polynomial in 3 variables has 1770, and the high powers of inputs in [-1, 1] are nearly collinear. Without the penalty the cross-validation error explodes at high degree. When several degrees were tied, the lowest one was chosen, because the test inputs are more spread out than the training inputs and so need some extrapolation.

The full write-up (method, figures, discussion) is in `report/BT2024171_report.pdf`.

## Repository layout

```
data/          provided train/test CSVs and sample_submission.csv
codes/         all code
  cv_explore.py     coarse CV search over degree and alpha
  cv_refine.py      finer alpha grid with repeated CV for the close calls
  cv_grid.py        full CV grid saved to report/ (used for report figures)
  train_predict.py  final fit on all training rows, writes predictions and metrics
  make_report.py    builds the PDF report
predictions/   BT2024171_pred_var1.csv, BT2024171_pred_var2.csv
report/        BT2024171_report.pdf, metrics.json, CV grid CSVs
docs/          assignment statement
requirements.txt
```

## How to run

```
pip install -r requirements.txt

# 1. model selection (optional, results are already in report/)
python codes/cv_explore.py 1 10          # var1, degrees 1..10
python codes/cv_explore.py 2 20          # var2, degrees 1..20
python codes/cv_refine.py 1 4 8          # var, min degree, max degree
python codes/cv_grid.py                  # full degree x alpha grid

# 2. final models and predictions
python codes/train_predict.py            # writes predictions/ and report/metrics.json

# 3. report
python codes/make_report.py              # writes report/BT2024171_report.pdf
```

All scripts locate the data through the repository root, so they can be run from any directory. Cross-validation uses fixed seeds, so the numbers reproduce exactly.

## Prediction files

`predictions/BT2024171_pred_var1.csv` and `predictions/BT2024171_pred_var2.csv` follow the sample submission format: one column named `y`, 1000 rows, in the same order as the corresponding test file.
