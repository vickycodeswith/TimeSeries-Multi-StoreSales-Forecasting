"""Time-aware train/validation splits."""

import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
    validation_days: int,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame at the last ``validation_days`` calendar days.

    Unlike a random split, this preserves the forecasting direction and makes
    validation representative of predicting dates after the training period.
    """

    if date_col not in frame.columns:
        raise ValueError(f"Missing required column: {date_col}")
    if not isinstance(validation_days, int) or validation_days <= 0:
        raise ValueError("validation_days must be a positive integer")

    dates = pd.to_datetime(frame[date_col])
    unique_dates = dates.drop_duplicates().sort_values()
    if len(unique_dates) <= validation_days:
        raise ValueError("validation_days must be smaller than the number of unique dates")

    cutoff = unique_dates.iloc[-validation_days]
    train = frame.loc[dates < cutoff].copy()
    validation = frame.loc[dates >= cutoff].copy()
    return train, validation
