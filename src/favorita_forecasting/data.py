"""Loading and validating the competition's core sales tables."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TRAIN_COLUMNS = ("date", "store_nbr", "family", "sales", "onpromotion")
TEST_COLUMNS = ("id", "date", "store_nbr", "family", "onpromotion")


@dataclass(frozen=True)
class CompetitionData:
    """The tables needed by the lightweight, reproducible workflow."""

    train: pd.DataFrame
    test: pd.DataFrame | None = None


def load_train_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a Favorita ``train.csv`` file."""

    frame = pd.read_csv(path, parse_dates=["date"])
    _validate_columns(frame, TRAIN_COLUMNS, "train")
    _validate_non_negative(frame, ["sales", "onpromotion"])
    return frame.sort_values(["date", "store_nbr", "family"], ignore_index=True)


def load_test_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a Favorita ``test.csv`` file."""

    frame = pd.read_csv(path, parse_dates=["date"])
    _validate_columns(frame, TEST_COLUMNS, "test")
    _validate_non_negative(frame, ["onpromotion"])
    return frame.sort_values(["date", "store_nbr", "family"], ignore_index=True)


def load_competition_data(data_dir: str | Path, include_test: bool = False) -> CompetitionData:
    """Load the core competition tables from a data directory."""

    directory = Path(data_dir)
    train = load_train_csv(directory / "train.csv")
    test = load_test_csv(directory / "test.csv") if include_test else None
    return CompetitionData(train=train, test=test)


def _validate_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name}.csv is missing required columns: {', '.join(missing)}")


def _validate_non_negative(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if (frame[column] < 0).any():
            raise ValueError(f"{column} must contain non-negative values")
