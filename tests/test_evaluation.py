import pandas as pd

from favorita_forecasting import evaluate_forecast


def test_evaluate_forecast_returns_predictions_and_rmsle() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "store_nbr": [1] * 10,
            "family": ["A"] * 10,
            "sales": [1.0] * 10,
        }
    )

    score, comparison = evaluate_forecast(frame, validation_days=3)

    assert score == 0.0
    assert len(comparison) == 3
    assert {"actual", "forecast"} <= set(comparison.columns)
