"""Small, data-backed plotting seams for forecast inspection."""

from collections.abc import Iterable

import pandas as pd

from .metrics import rmsle


def plot_forecast(
    frame: pd.DataFrame,
    date_col: str = "date",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    *,
    ax=None,
):
    """Plot actual and forecast values and return the Matplotlib axes.

    ``frame`` is intentionally supplied by the caller so this function cannot
    imply a result without a data-backed forecast run.
    """

    _require_columns(frame, [date_col, actual_col, forecast_col])
    plt = _matplotlib_pyplot()
    axis = ax if ax is not None else plt.subplots()[1]
    dates = pd.to_datetime(frame[date_col])
    axis.plot(dates, frame[actual_col], label="Actual")
    axis.plot(dates, frame[forecast_col], label="Forecast")
    axis.set_xlabel("Date")
    axis.set_ylabel("Sales")
    axis.set_title("Actual versus forecast")
    axis.legend()
    axis.figure.autofmt_xdate()
    return axis


def rmsle_by_group(
    frame: pd.DataFrame,
    group_col: str,
    actual_col: str = "actual",
    forecast_col: str = "forecast",
) -> pd.Series:
    """Return one RMSLE value per group for error-breakdown plots."""

    _require_columns(frame, [group_col, actual_col, forecast_col])
    values = {
        group: rmsle(group_frame[actual_col], group_frame[forecast_col])
        for group, group_frame in frame.groupby(group_col, sort=False, dropna=False)
    }
    return pd.Series(values, name="rmsle")


def plot_error_breakdown(
    frame: pd.DataFrame,
    group_col: str,
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    *,
    ax=None,
):
    """Plot RMSLE by a store, product family, or other grouping column."""

    errors = rmsle_by_group(frame, group_col, actual_col, forecast_col)
    plt = _matplotlib_pyplot()
    axis = ax if ax is not None else plt.subplots()[1]
    errors.plot.bar(ax=axis)
    axis.set_xlabel(group_col)
    axis.set_ylabel("RMSLE")
    axis.set_title(f"Forecast error by {group_col}")
    return axis


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting requires matplotlib; install the optional 'visualization' dependency"
        ) from error
    return plt
