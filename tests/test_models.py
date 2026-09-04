import pandas as pd

from favorita_forecasting import seasonal_naive_forecast


def test_seasonal_naive_repeats_each_group_weekly_pattern() -> None:
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=7).tolist() * 2,
            "store_nbr": [1] * 7 + [2] * 7,
            "family": ["A"] * 14,
            "sales": list(range(1, 8)) + list(range(11, 18)),
        }
    )

    result = seasonal_naive_forecast(
        history,
        pd.date_range("2024-01-08", periods=3),
    )

    assert result[result.store_nbr.eq(1)]["sales"].tolist() == [1.0, 2.0, 3.0]
    assert result[result.store_nbr.eq(2)]["sales"].tolist() == [11.0, 12.0, 13.0]
