# Assignment 1: Polynomial Regression (BT2024171)

Ridge-regularised polynomial regression for two problems (var1: 6 features, var2: 3 features). Degree and ridge penalty are chosen by cross-validation on the training data.

| Problem | Degree | Ridge alpha | 5-fold CV MSE |
|---|---|---|---|
| var1 | 5 | 3.16 | 0.416 |
| var2 | 12 | 0.1 | 0.235 |

## Layout
- `data/` train/test CSVs and sample submission
- `codes/` all code
- `predictions/` `BT2024171_pred_var1.csv`, `BT2024171_pred_var2.csv`
- `report/` PDF report and metrics
- `docs/` assignment statement

## Run
```
pip install -r requirements.txt
python codes/cv_explore.py <var> <max_degree>     # degree/alpha search (e.g. 1 10, 2 20)
python codes/cv_refine.py <var> <min_deg> <max_deg>
python codes/cv_grid.py                           # full CV grid (degree x alpha) for the report figures
python codes/train_predict.py                     # final fit + predictions + metrics
python codes/make_report.py                       # builds the PDF report
```
