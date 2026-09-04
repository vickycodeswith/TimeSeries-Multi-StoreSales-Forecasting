"""Leakage-safe feature builders for tabular sales forecasting."""

from collections.abc import Iterable

import pandas as pd


def add_calendar_features(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar features while preserving the input rows and columns."""

    _require_columns(frame, [date_col])
    result = frame.copy()
    dates = pd.to_datetime(result[date_col])

    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["day"] = dates.dt.day
    result["day_of_week"] = dates.dt.dayofweek
    result["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    result["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype("int8")
    return result


def add_lag_features(
    frame: pd.DataFrame,
    target_col: str,
    group_cols: Iterable[str],
    lags: Iterable[int] = (1, 7, 14),
    rolling_windows: Iterable[int] = (7, 28),
    date_col: str = "date",
) -> pd.DataFrame:
    """Add historical lags and rolling means without using the current target.

    The returned frame has the same row order as ``frame``.  Rolling features
    are shifted before aggregation, so a row can never use its own target as an
    input feature.
    """

    groups = list(group_cols)
    lag_values = _positive_ints(lags, "lags")
    window_values = _positive_ints(rolling_windows, "rolling_windows")
    if not groups:
        raise ValueError("group_cols must contain at least one column")
    _require_columns(frame, [date_col, target_col, *groups])

    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result["__original_order"] = range(len(result))
    result = result.sort_values([*groups, date_col, "__original_order"])
    grouped_target = result.groupby(groups, sort=False, dropna=False)[target_col]

    for lag in lag_values:
        result[f"{target_col}_lag_{lag}"] = grouped_target.shift(lag)

    for window in window_values:
        result[f"{target_col}_rolling_mean_{window}"] = grouped_target.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )

    return result.sort_values("__original_order").drop(columns="__original_order")


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _positive_ints(values: Iterable[int], name: str) -> list[int]:
    result = list(values)
    if not result or any(not isinstance(value, int) or value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result
