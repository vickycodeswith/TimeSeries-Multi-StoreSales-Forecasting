# Favorita sales forecasting: a portfolio case study

## The problem

Corporación Favorita sells thousands of products across Ecuadorian grocery
stores. A useful forecast must estimate demand for many related store–product
series while handling promotions, holidays, store differences, and gaps in the
historical record.

The project evaluates forecasts with RMSLE. This emphasizes relative error and
prevents high-volume products from completely dominating the score.

## What I built

The repository has two layers:

1. A small, testable Python workflow for loading data, generating lag and
   calendar features, evaluating a chronological holdout, and writing artifacts.
2. The original Darts/LightGBM experiment, preserved as a legacy research record.

The clean workflow can be run without opening a notebook:

```bash
python -m pip install -e ".[dev,visualization]"
python -m favorita_forecasting --data-dir data --output-dir artifacts --plots
```

It writes `artifacts/metrics.json`, `artifacts/validation_predictions.csv`,
`artifacts/forecast.png`, and `artifacts/error_by_store.png`. The default model
is a weekly seasonal-naive baseline. The optional linear model uses calendar,
lag, and rolling features:

```bash
python -m pip install -e ".[dev,modeling]"
python -m favorita_forecasting --model linear-lag
```

## Modeling progression

| Experiment | RMSLE | Evidence |
|---|---:|---|
| Linear Regression | 0.36712 | Recorded in the legacy notebook |
| LightGBM ensemble | 0.33725 | Recorded in the legacy notebook |
| Tuned LightGBM ensemble | 0.33667 | Recorded in the legacy notebook |
| Kaggle submission | 0.38202 | Historical public leaderboard result |

These values are not reproduced by the lightweight workflow yet. They are kept
with their provenance so the project remains honest and easy to audit.

## Data decisions

- Lag and rolling features are shifted so the current target cannot enter its
  own features.
- Validation uses the final chronological dates, never a random split.
- The original notebook uses one 16-day holdout (`folds=1`); it should not be
  described as rolling-origin cross-validation until multiple folds are run.

## What to inspect

- [The CLI entry point](../src/favorita_forecasting/train.py)
- [The data boundary](../src/favorita_forecasting/data.py)
- [The forecasting models](../src/favorita_forecasting/models.py)
- [The evaluation contract](../src/favorita_forecasting/evaluation.py)
- [The feature leakage tests](../tests/test_features.py)

The plots are generated from the same validation artifact used for the metric;
no synthetic forecast chart is included in the repository.

## Limitations and next step

The full competition data and Kaggle credentials are not committed, so the
legacy Darts pipeline is not executed in CI. The next engineering step is to
extract its holiday, transaction, oil-price, and store-metadata joins into the
same tested workflow, then compare the lightweight baselines with the original
LightGBM configuration on identical validation dates.
