"""Uniform forecaster interface and the two baseline models.

Every model of the project implements :class:`Forecaster`:

* ``fit(history)`` takes a panel slice (monthly ``PeriodIndex``) that ends at the forecast
  origin, i.e. the last month whose value is known. Models that only need the target read the
  ``deposits`` column.
* ``predict(horizon, exog=None)`` returns a DataFrame indexed by the next ``horizon`` months
  with the columns ``mean``, ``lower`` and ``upper``. ``exog`` is reserved for future driver
  paths (for example rate scenarios) and is ignored by the baselines.

All models share one nominal band level, ``level`` (default 0.8, a central 80 % interval).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import NormalDist

import numpy as np
import pandas as pd

TARGET = "deposits"
DEFAULT_LEVEL = 0.8
DEFAULT_DRIFT_WINDOW = 36
RESULT_COLUMNS = ("mean", "lower", "upper")


def _z_score(level: float) -> float:
    """Normal quantile for a central interval with nominal coverage ``level``."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    return NormalDist().inv_cdf(0.5 + level / 2)


class Forecaster(ABC):
    """Common ``fit``/``predict`` interface for all models."""

    def __init__(self, level: float = DEFAULT_LEVEL) -> None:
        self.z = _z_score(level)
        self.level = level
        self.origin_: pd.Period | None = None

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> Forecaster:
        """Fit on a panel slice that ends at the forecast origin; returns ``self``."""

    @abstractmethod
    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
        """Forecast the next ``horizon`` months as a DataFrame ``mean``/``lower``/``upper``."""

    def _read_target(self, history: pd.DataFrame) -> pd.Series:
        """Validate ``history``, remember the origin and return the target series."""
        index = history.index
        if not isinstance(index, pd.PeriodIndex) or index.freqstr != "M":
            raise ValueError("history must have a monthly PeriodIndex")
        if TARGET not in history:
            raise ValueError(f"history must contain the column {TARGET!r}")
        y = history[TARGET].astype(float)
        if len(y) < 3 or y.isna().any():
            raise ValueError("history needs at least 3 observations and no missing values")
        self.origin_ = index[-1]
        return y

    def _frame(self, horizon: int, mean: np.ndarray, half_width: np.ndarray) -> pd.DataFrame:
        if self.origin_ is None:
            raise RuntimeError("call fit() before predict()")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        index = pd.period_range(self.origin_ + 1, periods=horizon, freq="M")
        return pd.DataFrame(
            {"mean": mean, "lower": mean - half_width, "upper": mean + half_width},
            index=index,
        )


class _RandomWalkBand(Forecaster):
    """Shared logic: last value plus ``h * drift``, band ``z * sigma * sqrt(h)``."""

    def __init__(self, level: float = DEFAULT_LEVEL) -> None:
        super().__init__(level)
        self.last_: float | None = None
        self.drift_: float = 0.0
        self.sigma_: float | None = None

    def _estimation_changes(self, changes: pd.Series) -> pd.Series:
        return changes

    def _drift(self, changes: pd.Series) -> float:
        return 0.0

    def fit(self, history: pd.DataFrame) -> _RandomWalkBand:
        y = self._read_target(history)
        changes = self._estimation_changes(y.diff().dropna())
        self.last_ = float(y.iloc[-1])
        self.drift_ = self._drift(changes)
        self.sigma_ = float(changes.std(ddof=1))
        return self

    def _path(self, horizon: int) -> pd.DataFrame:
        if self.last_ is None or self.sigma_ is None:
            raise RuntimeError("call fit() before predict()")
        steps = np.arange(1, horizon + 1)
        return self._frame(
            horizon, self.last_ + steps * self.drift_, self.z * self.sigma_ * np.sqrt(steps)
        )

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
        return self._path(horizon)


class NaiveForecaster(_RandomWalkBand):
    """Next value equals the last value (random walk without drift).

    The band is ``z * sigma * sqrt(h)`` with ``sigma`` the standard deviation of the monthly
    changes over the full history.
    """


class DriftForecaster(_RandomWalkBand):
    """Last value plus the average monthly change, with optional planned outflows.

    ``mean = last + h * drift - cumulative planned outflows``, where ``drift`` is the average
    monthly change over the trailing ``window`` months (``None`` for the full history) and the
    band is ``+/- z * sigma * sqrt(h)`` with ``sigma`` the standard deviation of the monthly
    changes in the same window. There is deliberately no seasonality.

    ``planned_outflows`` is a schedule of known future outflows: a series indexed by month
    (``PeriodIndex``) with amounts in the unit of the target (EUR bn, positive numbers). An
    outflow reduces the mean from its month onwards, and the band is shifted by the same
    amount. Entries on or before the forecast origin or beyond the horizon have no effect.
    """

    def __init__(
        self,
        window: int | None = DEFAULT_DRIFT_WINDOW,
        planned_outflows: pd.Series | None = None,
        level: float = DEFAULT_LEVEL,
    ) -> None:
        super().__init__(level)
        if window is not None and window < 2:
            raise ValueError("window must be at least 2 monthly changes, or None")
        if planned_outflows is not None:
            index = planned_outflows.index
            if not isinstance(index, pd.PeriodIndex) or index.freqstr != "M":
                raise ValueError("planned_outflows must be indexed by a monthly PeriodIndex")
            if planned_outflows.isna().any():
                raise ValueError("planned_outflows must not contain missing values")
        self.window = window
        self.planned_outflows = planned_outflows

    def _estimation_changes(self, changes: pd.Series) -> pd.Series:
        return changes if self.window is None else changes.iloc[-self.window :]

    def _drift(self, changes: pd.Series) -> float:
        return float(changes.mean())

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
        result = self._path(horizon)
        if self.planned_outflows is not None:
            per_month = self.planned_outflows.groupby(level=0).sum().reindex(result.index)
            cumulative = per_month.fillna(0.0).cumsum().to_numpy()
            result = result.sub(cumulative, axis=0)
        return result
