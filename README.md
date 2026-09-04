# Favorita sales forecasting

Forecasting daily product-family demand across Corporación Favorita’s grocery
stores in Ecuador.

This project started as a Kaggle competition experiment and is being renewed
as a small, reproducible forecasting case study. It focuses on a practical
retail question:

> How can a retailer forecast demand for thousands of store–product series
> while accounting for promotions, holidays, store characteristics, oil prices,
> and gaps in the source data?

## Why this project is interesting

- The dataset contains roughly 3 million historical sales records across 54
  stores and 33 product families.
- The target is evaluated with RMSLE, which rewards accurate relative errors and
  makes over-predicting low-volume products especially visible.
- The original experiment compared a linear baseline with LightGBM models and
  an ensemble of different lag windows.
- The workflow includes missing-date handling, holiday normalization, calendar
  features, static store covariates, lagged demand, and time-aware validation.

The original notebook reports a historical public leaderboard result of 0.38202
(25th place out of approximately 650 teams, as of 2023-08-30). That result is
preserved as historical context; it has not yet been re-run in this environment.

## Results from the legacy notebook

The following values are recorded in the preserved notebook or its historical
competition record. They are evidence of the original experiment, not results
reproduced by the renewed package.

| Experiment | RMSLE | Provenance |
|---|---:|---|
| Linear Regression validation | 0.36712 | Notebook-recorded |
| LightGBM ensemble validation | 0.33725 | Notebook-recorded |
| Tuned LightGBM ensemble validation | 0.33667 | Notebook-recorded |
| Kaggle submission | 0.38202 | Historical; rank 25 of approximately 650 |

The validation scores and Kaggle leaderboard score use different evaluation
contexts and should not be compared as if they were the same measurement.

## Reproducibility and limitations

- The original experiment requires Kaggle credentials and the competition data;
  the raw files are intentionally not committed.
- The new `src/favorita_forecasting` package is tested, but it does not yet
  reproduce the notebook's Darts training, LightGBM configuration, or
  submission pipeline. The notebook remains the historical source of the
  results above.
- The notebook configuration sets `folds=1` with a `forecast_horizon` of 16.
  Its reported validation is therefore a single chronological 16-day holdout,
  not rolling-origin cross-validation.
- The notebook's early documentation describes `test.csv` as covering 15 days,
  while its recorded data inspection shows 16 unique dates (2017-08-16 through
  2017-08-31). This horizon discrepancy is preserved rather than silently
  resolved.
- The plotting module documents the seam for forecast and error visualizations,
  but no plots are committed because the competition data is unavailable here.
  A data-backed run should pass actual and forecast values to those functions
  and save the resulting figures as artifacts.

## Project structure

```text
.
├── docs/case-study.md                          # portfolio-ready project story
├── notebooks/legacy/                           # original exploratory notebook
├── src/favorita_forecasting/                  # reusable, tested utilities
│   ├── data.py                                  # loading and schema validation
│   ├── evaluation.py                            # holdout evaluation and artifacts
│   ├── features.py                             # calendar, lag, rolling features
│   ├── metrics.py                              # RMSLE
│   ├── models.py                                # seasonal-naive and linear models
│   ├── splitting.py                            # chronological validation split
│   ├── train.py                                 # one-command experiment
│   └── visualization.py                         # forecast and error plots
├── tests/                                      # fast behavior tests
├── pyproject.toml                              # runtime and development setup
└── data/                                       # downloaded Kaggle files (ignored)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

The [case study](docs/case-study.md) is the primary project showcase. The
reusable package can be imported directly:

```python
from favorita_forecasting import (
    add_calendar_features,
    add_lag_features,
    chronological_split,
    evaluate_forecast,
    rmsle,
    seasonal_naive_forecast,
)
```

Run the lightweight workflow without opening a notebook:

```bash
python -m favorita_forecasting --data-dir data --output-dir artifacts
```

Optional plotting support is available with `matplotlib`:

```bash
python -m pip install -e ".[visualization]"
```

## Reproduce the original experiment

The raw competition data is not committed to this repository. Download it from
the [Kaggle Store Sales - Time Series Forecasting competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting),
and place the downloaded `store-sales-time-series-forecasting.zip` in the
repository root. Then install the notebook dependencies and open the notebook
from the repository root:

```bash
python -m pip install -e ".[notebook]"
jupyter notebook notebooks/legacy/store-sales-time-series-forecasting.ipynb
```

The notebook downloads the archive with the Kaggle CLI, extracts files into
`data/`, prepares the time series, validates several LightGBM configurations,
and writes `submission.csv`. Kaggle credentials are required for the download
and submission commands.

## Renewal roadmap

The repository now has a tested, code-first workflow around the original
notebook. The next portfolio-facing slices are:

1. Extract the legacy holiday, transaction, oil-price, and store-metadata joins.
2. Run the legacy pipeline against available Kaggle data and replace notebook-only
   claims with an auditable, data-backed benchmark.
3. Add a lightweight demo so visitors can select a store/family pair and inspect
   a forecast without downloading the full competition dataset.

## License

MIT. See [LICENSE](LICENSE).
