import pytest

from favorita_forecasting import rmsle


def test_rmsle_matches_the_competition_definition() -> None:
    assert rmsle([0, 1, 10], [0, 2, 8]) == pytest.approx(0.261196, abs=1e-6)


@pytest.mark.parametrize(
    ("actual", "forecast", "message"),
    [
        ([1], [1, 2], "same shape"),
        ([], [], "not be empty"),
        ([1], [-1], "non-negative"),
        ([float("nan")], [1], "finite"),
    ],
)
def test_rmsle_rejects_invalid_inputs(actual, forecast, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rmsle(actual, forecast)
