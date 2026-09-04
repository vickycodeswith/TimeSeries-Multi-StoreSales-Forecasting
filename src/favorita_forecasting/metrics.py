"""Evaluation metrics used by the Favorita competition."""

import numpy as np
from numpy.typing import ArrayLike


def rmsle(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return root mean squared logarithmic error.

    RMSLE is only defined for non-negative, finite observations.  Failing fast
    here prevents invalid forecasts from being hidden by an apparently valid
    aggregate score.
    """

    actual = np.asarray(y_true, dtype=float)
    forecast = np.asarray(y_pred, dtype=float)

    if actual.shape != forecast.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if actual.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("y_true and y_pred must contain only finite values")
    if (actual < 0).any() or (forecast < 0).any():
        raise ValueError("RMSLE requires non-negative values")

    log_error = np.log1p(forecast) - np.log1p(actual)
    return float(np.sqrt(np.mean(log_error**2)))
