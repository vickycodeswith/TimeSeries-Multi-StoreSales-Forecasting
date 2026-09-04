"""Command-line entry point for the lightweight reproducible workflow."""

import argparse

from .data import load_train_csv
from .evaluation import evaluate_forecast, write_plot_artifacts, write_run_artifacts
from .models import fit_linear_lag_model, seasonal_naive_forecast
from .splitting import chronological_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Favorita validation experiment")
    parser.add_argument("--data-dir", default="data", help="Directory containing train.csv")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for run outputs")
    parser.add_argument("--validation-days", type=int, default=16)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write forecast.png and error_by_store.png (requires the visualization extra)",
    )
    parser.add_argument(
        "--model",
        choices=("seasonal-naive", "linear-lag"),
        default="seasonal-naive",
    )
    args = parser.parse_args()

    history = load_train_csv(f"{args.data_dir}/train.csv")
    if args.model == "linear-lag":
        training_history, _ = chronological_split(history, args.validation_days)
        fitted = fit_linear_lag_model(training_history)
        forecast_function = fitted.forecast
    else:
        forecast_function = seasonal_naive_forecast

    score, comparison = evaluate_forecast(
        history,
        validation_days=args.validation_days,
        forecast_function=forecast_function,
    )
    write_run_artifacts(
        args.output_dir,
        model_name=args.model,
        score=score,
        comparison=comparison,
        validation_days=args.validation_days,
    )
    if args.plots:
        write_plot_artifacts(args.output_dir, comparison)
    print(f"{args.model} RMSLE: {score:.5f}")
