import pandas as pd

from favorita_forecasting import add_calendar_features, add_lag_features, chronological_split


def test_calendar_features_capture_weekend_and_iso_week() -> None:
    frame = pd.DataFrame({"date": ["2024-01-01", "2024-01-06"]})

    result = add_calendar_features(frame)

    assert result[["year", "month", "day_of_week", "is_weekend"]].to_dict("records") == [
        {"year": 2024, "month": 1, "day_of_week": 0, "is_weekend": 0},
        {"year": 2024, "month": 1, "day_of_week": 5, "is_weekend": 1},
    ]
    assert result["week_of_year"].tolist() == [1, 1]


def test_lag_and_rolling_features_do_not_use_the_current_target() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "store_nbr": [1, 1, 1],
            "family": ["A", "A", "A"],
            "sales": [10.0, 20.0, 30.0],
        }
    )

    result = add_lag_features(
        frame,
        "sales",
        ["store_nbr", "family"],
        lags=[1],
        rolling_windows=[2],
    )

    assert result["sales_lag_1"].isna().tolist() == [True, False, False]
    assert result["sales_lag_1"].tolist()[1:] == [10.0, 20.0]
    assert result["sales_rolling_mean_2"].isna().tolist() == [True, False, False]
    assert result["sales_rolling_mean_2"].tolist()[1:] == [10.0, 15.0]


def test_chronological_split_uses_the_last_calendar_days() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "sales": range(5)})

    train, validation = chronological_split(frame, validation_days=2)

    assert train["date"].dt.day.tolist() == [1, 2, 3]
    assert validation["date"].dt.day.tolist() == [4, 5]
