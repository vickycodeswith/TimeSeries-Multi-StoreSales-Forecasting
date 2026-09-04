"""Evaluation and artifact helpers for the showcase workflow."""

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .metrics import rmsle
from .models import seasonal_naive_forecast
from .splitting import chronological_split

ForecastFunction = Callable[[pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]


def evaluate_forecast(
    history: pd.DataFrame,
    validation_days: int = 16,
    forecast_function: ForecastFunction | None = None,
) -> tuple[float, pd.DataFrame]:
    """Evaluate a forecast function on the final chronological holdout."""

    train, validation = chronological_split(history, validation_days)
    dates = pd.DatetimeIndex(pd.to_datetime(validation["date"]).unique()).sort_values()
    forecaster = forecast_function or seasonal_naive_forecast
    predictions = forecaster(train, dates).rename(columns={"sales": "forecast"})
    comparison = validation.rename(columns={"sales": "actual"}).merge(
        predictions,
        on=["date", "store_nbr", "family"],
        how="left",
    )
    if comparison["forecast"].isna().any():
        raise ValueError("Forecast did not produce every validation row")
    return rmsle(comparison["actual"], comparison["forecast"]), comparison


def write_run_artifacts(
    output_dir: str | Path,
    model_name: str,
    score: float,
    comparison: pd.DataFrame,
    validation_days: int,
) -> None:
    """Write machine-readable predictions and metadata for a run."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(directory / "validation_predictions.csv", index=False)
    metadata = {
        "model": model_name,
        "metric": "RMSLE",
        "score": score,
        "validation_days": validation_days,
        "validation_start": comparison["date"].min().date().isoformat(),
        "validation_end": comparison["date"].max().date().isoformat(),
    }
    (directory / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_plot_artifacts(output_dir: str | Path, comparison: pd.DataFrame) -> None:
    """Generate forecast and store-error plots from a completed run."""

    from .visualization import plot_error_breakdown, plot_forecast

    directory = Path(output_dir)
    forecast_axis = plot_forecast(comparison)
    forecast_axis.figure.savefig(directory / "forecast.png", dpi=150, bbox_inches="tight")
    forecast_axis.figure.clf()

    error_axis = plot_error_breakdown(comparison, "store_nbr")
    error_axis.figure.savefig(directory / "error_by_store.png", dpi=150, bbox_inches="tight")
    error_axis.figure.clf()
