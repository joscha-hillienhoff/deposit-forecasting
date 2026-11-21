"""Transparent rate-driven regression of monthly deposit growth on the rate spread.

The model is an OLS regression (statsmodels) of the monthly log change of deposits on

* the spread between time-deposit and overnight rate, lagged by one month,
* monthly seasonal dummies (January is the reference, absorbed by the constant),
* the lagged log change of deposits.

``predict`` needs the spread path of the forecast months in ``exog`` and iterates the
regression forward month by month. See ``docs/methodology.md`` for the design decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from deposit_forecasting.models import DEFAULT_LEVEL, Forecaster

SPREAD = "spread"
MONTH_COLUMNS = [f"month_{m:02d}" for m in range(2, 13)]
FEATURES = ["const", "spread_lag1", "dlog_lag1", *MONTH_COLUMNS]


def _month_dummies(months: pd.PeriodIndex) -> np.ndarray:
    """Dummy matrix for February to December; January is the reference month."""
    return (months.month.to_numpy()[:, None] == np.arange(2, 13)).astype(float)


class RateRegressionForecaster(Forecaster):
    """OLS of the monthly log change of deposits on the lagged spread, seasonality and AR(1).

    The mean path is ``last * exp(cumulative predicted log changes)``. The band is
    ``z * sigma * sqrt(h)`` on the log scale, with ``sigma`` the residual standard deviation,
    back-transformed with ``exp``. This treats the monthly forecast errors as independent, so
    the h-step variance is ``h * sigma^2``; the band is therefore only approximate and ignores
    parameter uncertainty. Because of the back-transformation the ``mean`` is the median path.
    """

    def __init__(self, level: float = DEFAULT_LEVEL) -> None:
        super().__init__(level)
        self.result_ = None
        self.sigma_: float | None = None
        self.last_level_: float | None = None
        self.last_change_: float | None = None
        self.last_spread_: float | None = None

    def fit(self, history: pd.DataFrame) -> RateRegressionForecaster:
        y = self._read_target(history)
        if SPREAD not in history or history[SPREAD].isna().any():
            raise ValueError(f"history must contain the column {SPREAD!r} without missing values")
        spread = history[SPREAD].astype(float)
        change = np.log(y).diff()
        design = pd.DataFrame(
            {
                "spread_lag1": spread.shift(1),
                "dlog_lag1": change.shift(1),
                **dict(zip(MONTH_COLUMNS, _month_dummies(y.index).T, strict=True)),
            },
            index=y.index,
        )
        sample = pd.concat([change.rename("y"), design], axis=1).dropna()
        if len(sample) < len(FEATURES) + 2:
            raise ValueError("history is too short for the regression")
        exog = sm.add_constant(sample[FEATURES[1:]], has_constant="add")
        self.result_ = sm.OLS(sample["y"], exog).fit()
        self.sigma_ = float(np.sqrt(self.result_.scale))
        self.last_level_ = float(y.iloc[-1])
        self.last_change_ = float(change.iloc[-1])
        self.last_spread_ = float(spread.iloc[-1])
        return self

    def coefficients(self) -> pd.DataFrame:
        """Estimates, standard errors and t-values of the fitted regression."""
        if self.result_ is None:
            raise RuntimeError("call fit() before coefficients()")
        return pd.DataFrame(
            {
                "estimate": self.result_.params,
                "std_error": self.result_.bse,
                "t_value": self.result_.tvalues,
            }
        )

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.result_ is None or self.sigma_ is None:
            raise RuntimeError("call fit() before predict()")
        if exog is None or SPREAD not in exog:
            raise ValueError(f"predict needs exog with the {SPREAD!r} path of the forecast months")
        frame = self._frame(horizon, np.zeros(horizon), np.zeros(horizon))
        months = frame.index
        path = exog[SPREAD].astype(float).reindex(months)
        if path.isna().any():
            raise ValueError(f"exog {SPREAD!r} must cover every forecast month without gaps")
        # The spread of month t - 1 drives month t, so only the first h - 1 values enter.
        lagged = np.concatenate([[self.last_spread_], path.to_numpy()[:-1]])
        dummies = _month_dummies(months)
        params = self.result_.params
        beta = params[FEATURES].to_numpy()
        changes = np.empty(horizon)
        previous = self.last_change_
        for h in range(horizon):
            x = np.concatenate([[1.0, lagged[h], previous], dummies[h]])
            previous = changes[h] = float(x @ beta)
        steps = np.arange(1, horizon + 1)
        log_mean = np.log(self.last_level_) + np.cumsum(changes)
        half_width = self.z * self.sigma_ * np.sqrt(steps)
        return pd.DataFrame(
            {
                "mean": np.exp(log_mean),
                "lower": np.exp(log_mean - half_width),
                "upper": np.exp(log_mean + half_width),
            },
            index=months,
        )
