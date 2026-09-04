import pandas as pd
import pytest

from favorita_forecasting import data


def test_load_train_csv_sorts_and_validates_schema(tmp_path) -> None:
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01"],
            "store_nbr": [1, 1],
            "family": ["A", "A"],
            "sales": [2.0, 1.0],
            "onpromotion": [0, 1],
        }
    ).to_csv(path, index=False)

    result = data.load_train_csv(path)

    assert result["date"].dt.day.tolist() == [1, 2]


def test_load_train_csv_rejects_negative_sales(tmp_path) -> None:
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "store_nbr": [1],
            "family": ["A"],
            "sales": [-1.0],
            "onpromotion": [0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="sales"):
        data.load_train_csv(path)
