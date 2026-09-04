import pandas as pd

from favorita_forecasting import add_calendar_features, add_lag_features, chronological_split


def test_public_feature_and_split_path_handles_a_tiny_store_family_panel() -> None:
    dates = pd.date_range("2024-01-01", periods=8)
    frame = pd.DataFrame(
        [
            {"date": date, "store_nbr": store, "family": family, "sales": float(index + store)}
            for index, date in enumerate(dates)
            for store, family in [(1, "A"), (2, "B")]
        ]
    ).sample(frac=1, random_state=7)

    featured = add_calendar_features(frame)
    featured = add_lag_features(
        featured,
        target_col="sales",
        group_cols=["store_nbr", "family"],
        lags=[1],
        rolling_windows=[2],
    )
    train, validation = chronological_split(featured, validation_days=2)

    assert {"year", "day_of_week", "sales_lag_1", "sales_rolling_mean_2"} <= set(
        featured.columns
    )
    assert train["date"].max() < validation["date"].min()
    assert len(train) == 12
    assert len(validation) == 4
    assert validation["sales_lag_1"].notna().all()
