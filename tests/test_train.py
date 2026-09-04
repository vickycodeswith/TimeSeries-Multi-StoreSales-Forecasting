import json
import sys

import pandas as pd

from favorita_forecasting import train


def test_cli_writes_reproducible_run_artifacts(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "store_nbr": [1] * 10,
            "family": ["A"] * 10,
            "sales": [1.0] * 10,
            "onpromotion": [0] * 10,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "favorita-forecasting",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--validation-days",
            "3",
        ],
    )

    train.main()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["model"] == "seasonal-naive"
    assert metrics["validation_days"] == 3
    assert (output_dir / "validation_predictions.csv").exists()


def test_cli_supports_linear_lag_model(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=35),
            "store_nbr": [1] * 35,
            "family": ["A"] * 35,
            "sales": [float(index % 7 + 1) for index in range(35)],
            "onpromotion": [0] * 35,
        }
    ).to_csv(data_dir / "train.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "favorita-forecasting",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--validation-days",
            "3",
            "--model",
            "linear-lag",
        ],
    )

    train.main()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["model"] == "linear-lag"
