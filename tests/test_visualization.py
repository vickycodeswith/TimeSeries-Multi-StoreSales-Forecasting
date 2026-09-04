import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from favorita_forecasting import (  # noqa: E402
    plot_error_breakdown,
    plot_forecast,
    rmsle_by_group,
)


@pytest.fixture
def forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "store_nbr": [1, 1, 2, 2],
            "actual": [10.0, 12.0, 8.0, 9.0],
            "forecast": [11.0, 11.0, 7.0, 10.0],
        }
    )


def test_plot_forecast_uses_supplied_data(forecast_frame: pd.DataFrame) -> None:
    axis = plot_forecast(forecast_frame)

    assert len(axis.lines) == 2
    assert [line.get_label() for line in axis.lines] == ["Actual", "Forecast"]
    plt.close(axis.figure)


def test_error_breakdown_and_plot_are_grouped_by_store(
    forecast_frame: pd.DataFrame,
) -> None:
    errors = rmsle_by_group(forecast_frame, "store_nbr")
    axis = plot_error_breakdown(forecast_frame, "store_nbr")

    assert errors.index.tolist() == [1, 2]
    assert errors.name == "rmsle"
    assert len(axis.patches) == 2
    plt.close(axis.figure)
