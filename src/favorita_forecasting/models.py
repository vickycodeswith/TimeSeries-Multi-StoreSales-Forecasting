"""Small forecasting models used by the reproducible showcase workflow."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import add_calendar_features, add_lag_features


def seasonal_naive_forecast(
    history: pd.DataFrame,
    forecast_dates: Iterable[pd.Timestamp],
    group_cols: Iterable[str] = ("store_nbr", "family"),
    target_col: str = "sales",
    date_col: str = "date",
    season_length: int = 7,
) -> pd.DataFrame:
    """Repeat each group's most recent weekly pattern."""

    groups = list(group_cols)
    dates = pd.DatetimeIndex(pd.to_datetime(list(forecast_dates))).sort_values()
    if len(dates) == 0:
        raise ValueError("forecast_dates must not be empty")
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    missing = sorted({date_col, target_col, *groups}.difference(history.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    history = history.copy()
    history[date_col] = pd.to_datetime(history[date_col])
    predictions: list[dict[str, object]] = []

    for group_values, group_frame in history.groupby(groups, sort=False, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        ordered = group_frame.sort_values(date_col)
        recent = ordered[target_col].tail(season_length).to_numpy(dtype=float)
        if len(recent) < season_length:
            raise ValueError("Each group needs at least season_length observations")

        for offset, date in enumerate(dates):
            row = dict(zip(groups, group_values, strict=True))
            row[date_col] = date
            row[target_col] = max(0.0, float(recent[offset % season_length]))
            predictions.append(row)

    return pd.DataFrame(predictions).sort_values([date_col, *groups], ignore_index=True)


@dataclass
class LinearLagModel:
    """A compact global linear model over calendar and lag features."""

    estimator: object
    feature_columns: list[str]
    target_col: str = "sales"
    group_cols: tuple[str, ...] = ("store_nbr", "family")
    date_col: str = "date"
    lags: tuple[int, ...] = (1, 7, 14)
    rolling_windows: tuple[int, ...] = (7, 28)

    def forecast(
        self,
        history: pd.DataFrame,
        forecast_dates: Iterable[pd.Timestamp],
    ) -> pd.DataFrame:
        """Generate recursive forecasts, adding each prediction to history."""

        dates = pd.DatetimeIndex(pd.to_datetime(list(forecast_dates))).sort_values()
        working = history.copy()
        predictions: list[pd.DataFrame] = []

        for date in dates:
            groups = working[list(self.group_cols)].drop_duplicates()
            future = groups.assign(**{self.date_col: date, self.target_col: np.nan})
            combined = pd.concat([working, future], ignore_index=True)
            featured = _feature_frame(
                combined,
                self.target_col,
                self.group_cols,
                self.date_col,
                self.lags,
                self.rolling_windows,
            )
            current = featured.loc[featured[self.date_col].eq(date)].copy()
            current[self.target_col] = np.maximum(
                0.0,
                self.estimator.predict(current[self.feature_columns]),
            )
            predictions.append(current[list(self.group_cols) + [self.date_col, self.target_col]])
            working = pd.concat([working, current], ignore_index=True)

        return pd.concat(predictions, ignore_index=True)


def fit_linear_lag_model(
    history: pd.DataFrame,
    target_col: str = "sales",
    group_cols: Iterable[str] = ("store_nbr", "family"),
    date_col: str = "date",
    lags: Iterable[int] = (1, 7, 14),
    rolling_windows: Iterable[int] = (7, 28),
) -> LinearLagModel:
    """Fit a global linear regression over historical lag features."""

    try:
        from sklearn.linear_model import LinearRegression
    except ImportError as error:
        raise ImportError(
            "Linear modeling requires scikit-learn; install the 'modeling' extra"
        ) from error

    groups = tuple(group_cols)
    lag_values = tuple(lags)
    window_values = tuple(rolling_windows)
    featured = _feature_frame(
        history, target_col, groups, date_col, lag_values, window_values
    ).dropna()
    feature_columns = _numeric_feature_columns(
        featured, target_col, date_col, groups, lag_values, window_values
    )
    estimator = LinearRegression().fit(featured[feature_columns], featured[target_col])
    return LinearLagModel(
        estimator=estimator,
        feature_columns=feature_columns,
        target_col=target_col,
        group_cols=groups,
        date_col=date_col,
        lags=lag_values,
        rolling_windows=window_values,
    )


def _feature_frame(
    frame: pd.DataFrame,
    target_col: str,
    group_cols: Iterable[str],
    date_col: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    featured = add_calendar_features(frame, date_col)
    return add_lag_features(
        featured,
        target_col=target_col,
        group_cols=group_cols,
        date_col=date_col,
        lags=lags,
        rolling_windows=rolling_windows,
    )


def _numeric_feature_columns(
    frame: pd.DataFrame,
    target_col: str,
    date_col: str,
    group_cols: Iterable[str],
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> list[str]:
    excluded = {target_col, date_col, *group_cols}
    engineered = {
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "is_weekend",
        *(f"{target_col}_lag_{lag}" for lag in lags),
        *(f"{target_col}_rolling_mean_{window}" for window in rolling_windows),
    }
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded and column in engineered
    ]
