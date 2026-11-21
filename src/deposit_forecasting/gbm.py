"""Gradient-boosting forecaster (LightGBM) with the direct multi-horizon strategy.

For every horizon ``h`` three models are fitted on the cumulative log change of deposits from
the origin to ``origin + h``: the lower quantile, the median and the upper quantile of the
shared band level. The median is the point forecast. See ``docs/methodology.md``.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from deposit_forecasting.models import DEFAULT_LEVEL, TARGET, Forecaster

MAX_HORIZON = 12
LAGS = (1, 2, 3, 6, 12)
MIN_HISTORY = 12 + MAX_HORIZON + 12  # lags and rolling change, plus a few training rows
DRIVER_COLUMNS = ("spread", "deposit_facility_rate")
# Fixed in advance, conservative for roughly 100-280 training rows; never tuned on the
# evaluation period.
DEFAULT_PARAMS: dict[str, float | int | bool | str] = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 4,
    "max_depth": 2,
    "min_child_samples": 15,
    "reg_lambda": 5.0,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def origin_features(history: pd.DataFrame) -> pd.DataFrame:
    """Features known at each origin (row), computed only from that row and earlier rows.

    Columns: ``lag_1`` ... the monthly log change of deposits ``k - 1`` months before the
    origin month (``lag_1`` is the latest change), ``roll_12`` (log change over the last 12
    months), ``spread`` and ``deposit_facility_rate`` at the origin, ``spread_chg_3`` and
    ``dfr_chg_6`` (their changes over the last 3 and 6 months).
    """
    log_y = np.log(history[TARGET].astype(float))
    change = log_y.diff()
    features = pd.DataFrame(index=history.index)
    for k in LAGS:
        features[f"lag_{k}"] = change.shift(k - 1)
    features["roll_12"] = log_y.diff(12)
    spread = history["spread"].astype(float)
    dfr = history["deposit_facility_rate"].astype(float)
    features["spread"] = spread
    features["deposit_facility_rate"] = dfr
    features["spread_chg_3"] = spread.diff(3)
    features["dfr_chg_6"] = dfr.diff(6)
    return features


def horizon_features(
    origin_feats: pd.DataFrame, spread_target: pd.Series, horizon: int
) -> pd.DataFrame:
    """Add the horizon-specific features: calendar month of the target and its spread.

    ``spread_target`` is indexed by the target month of each row (the origin row ``t`` uses
    the value at ``t + horizon``).
    """
    out = origin_feats.copy()
    out["target_month"] = (origin_feats.index + horizon).month
    out["spread_target"] = spread_target.reindex(origin_feats.index + horizon).to_numpy()
    return out


class GradientBoostingForecaster(Forecaster):
    """Direct multi-horizon LightGBM with quantile bands.

    ``predict`` needs the spread of the forecast months in ``exog`` (ex-post conditional, as in
    the backtest). At most ``max_horizon`` months can be forecast.
    """

    def __init__(
        self,
        level: float = DEFAULT_LEVEL,
        seed: int = 0,
        max_horizon: int = MAX_HORIZON,
        params: dict | None = None,
    ) -> None:
        super().__init__(level)
        self.seed = seed
        self.max_horizon = max_horizon
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.models_: dict[int, dict[str, lgb.LGBMRegressor]] = {}
        self.train_X_: dict[int, pd.DataFrame] = {}
        self.last_: float | None = None
        self.origin_features_: pd.Series | None = None

    def _quantiles(self) -> dict[str, float]:
        tail = (1 - self.level) / 2
        return {"lower": tail, "mean": 0.5, "upper": 1 - tail}

    def fit(self, history: pd.DataFrame) -> GradientBoostingForecaster:
        y = self._read_target(history)
        missing = [c for c in DRIVER_COLUMNS if c not in history]
        if missing:
            raise ValueError(f"history must contain the columns {missing}")
        if (y <= 0).any():
            raise ValueError("deposits must be positive")
        if len(y) < MIN_HISTORY:
            raise ValueError(f"history needs at least {MIN_HISTORY} observations")
        feats = origin_features(history)
        log_y = np.log(y)
        spread = history["spread"].astype(float)
        self.models_, self.train_X_ = {}, {}
        for h in range(1, self.max_horizon + 1):
            X = horizon_features(feats, spread, h)
            target = log_y.shift(-h) - log_y
            rows = X.notna().all(axis=1) & target.notna()
            X, target = X[rows], target[rows]
            self.train_X_[h] = X
            self.models_[h] = {
                name: lgb.LGBMRegressor(
                    objective="quantile", alpha=alpha, random_state=self.seed, **self.params
                ).fit(X, target)
                for name, alpha in self._quantiles().items()
            }
        self.last_ = float(y.iloc[-1])
        self.origin_features_ = feats.iloc[-1]
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.last_ is None or self.origin_features_ is None:
            raise RuntimeError("call fit() before predict()")
        if not 1 <= horizon <= self.max_horizon:
            raise ValueError(f"horizon must be between 1 and {self.max_horizon}")
        if exog is None or "spread" not in exog:
            raise ValueError("exog with the column 'spread' for the forecast months is required")
        index = pd.period_range(self.origin_ + 1, periods=horizon, freq="M")
        spread_target = exog["spread"].reindex(index)
        if spread_target.isna().any():
            raise ValueError("exog must cover every forecast month with a spread value")
        columns = list(self.train_X_[1].columns)
        values = np.empty((horizon, 3))
        for h in range(1, horizon + 1):
            row = self.origin_features_.copy()
            row["target_month"] = index[h - 1].month
            row["spread_target"] = spread_target.iloc[h - 1]
            X = pd.DataFrame([row[columns]], columns=columns)
            values[h - 1] = [self.models_[h][n].predict(X)[0] for n in ("lower", "mean", "upper")]
        # Separately fitted quantile models can cross; sorting restores lower <= mean <= upper.
        lower, mean, upper = (self.last_ * np.exp(np.sort(values, axis=1))).T
        return pd.DataFrame({"mean": mean, "lower": lower, "upper": upper}, index=index)
