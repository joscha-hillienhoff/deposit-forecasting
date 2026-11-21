"""Classical time-series forecasters: exponential smoothing (ETS) and SARIMA.

Both models work on log deposits, so seasonal and trend effects are proportional to the level.
Intervals are model-based at the shared level and are back-transformed with ``exp``; quantiles
survive a monotone transform, so ``lower``/``upper`` keep their nominal level, whereas the
back-transformed ``mean`` is the conditional median, not the conditional mean. A fit that fails
raises; there is no silent fallback. See ``docs/methodology.md``.
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from deposit_forecasting.models import DEFAULT_LEVEL, TARGET, Forecaster

SEASONAL_PERIOD = 12
# SARIMA is fitted on 100 * log deposits (log points x 100) so the innovation variance is not
# ~1e-5, which makes the optimiser fail to converge at some origins. Affects scale only.
SARIMA_LOG_SCALE = 100.0
SELECTION_END = pd.Period("2010-12", freq="M")

# Orders were selected once by AIC with ``select_sarima_orders`` (``uv run python -m
# deposit_forecasting.classical``) on the data up to SELECTION_END only (2003-01 to 2010-12),
# and are held fixed for every backtest origin. d = D = 1 is fixed in advance (see the grid).
SARIMA_ORDER = (1, 1, 2)
SARIMA_SEASONAL_ORDER = (0, 1, 1, SEASONAL_PERIOD)

_GRID_P = (0, 1, 2)
_GRID_Q = (0, 1, 2)
_GRID_SEASONAL_P = (0, 1)
_GRID_SEASONAL_Q = (0, 1)


class _FitFailedError(RuntimeError):
    """A model could not be fitted; raised instead of returning a fallback forecast."""


def _log_target(forecaster: Forecaster, history: pd.DataFrame) -> pd.Series:
    y = forecaster._read_target(history)
    if (y <= 0).any():
        raise ValueError("deposits must be strictly positive for the log transform")
    # A plain integer index avoids statsmodels' frequency inference on a PeriodIndex.
    return pd.Series(np.log(y.to_numpy()))


def _require_converged(result, label: str) -> None:
    converged = result.mle_retvals.get("converged", False)
    if not converged or not np.isfinite(result.llf) or not np.isfinite(result.params).all():
        raise _FitFailedError(f"{label} fit failed or did not converge")


def _back_transform(forecaster: Forecaster, horizon: int, log_frame: pd.DataFrame):
    out = forecaster._frame(horizon, np.zeros(horizon), np.zeros(horizon))
    out["mean"] = np.exp(log_frame["mean"].to_numpy())
    out["lower"] = np.exp(log_frame["lower"].to_numpy())
    out["upper"] = np.exp(log_frame["upper"].to_numpy())
    return out


class EtsForecaster(Forecaster):
    """Additive-error ETS(A,Ad,A) with period 12 on log deposits.

    Specification fixed in advance: additive error, damped additive trend, additive monthly
    seasonality. Parameters and initial states are estimated by maximum likelihood at each
    origin; the specification itself is never searched.
    """

    def __init__(self, level: float = DEFAULT_LEVEL) -> None:
        super().__init__(level)
        self.result_ = None

    def fit(self, history: pd.DataFrame) -> EtsForecaster:
        log_y = _log_target(self, history)
        if len(log_y) < 2 * SEASONAL_PERIOD:
            raise ValueError("ETS needs at least two full seasonal cycles")
        model = ETSModel(
            log_y,
            error="add",
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = model.fit(disp=False, maxiter=2000)
        _require_converged(result, "ETS")
        self.result_ = result
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.result_ is None:
            raise RuntimeError("call fit() before predict()")
        n = self.result_.nobs
        pred = self.result_.get_prediction(start=n, end=n + horizon - 1)
        frame = pred.summary_frame(alpha=1 - self.level)
        log_frame = frame.rename(columns={"pi_lower": "lower", "pi_upper": "upper"})
        return _back_transform(self, horizon, log_frame)


class SarimaForecaster(Forecaster):
    """SARIMA with fixed orders on log deposits (see ``SARIMA_ORDER``)."""

    def __init__(
        self,
        order: tuple[int, int, int] = SARIMA_ORDER,
        seasonal_order: tuple[int, int, int, int] = SARIMA_SEASONAL_ORDER,
        level: float = DEFAULT_LEVEL,
    ) -> None:
        super().__init__(level)
        self.order = order
        self.seasonal_order = seasonal_order
        self.result_ = None

    def fit(self, history: pd.DataFrame) -> SarimaForecaster:
        log_y = SARIMA_LOG_SCALE * _log_target(self, history)
        self.result_ = _fit_sarimax(log_y, self.order, self.seasonal_order)
        return self

    def predict(self, horizon: int, exog: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.result_ is None:
            raise RuntimeError("call fit() before predict()")
        forecast = self.result_.get_forecast(horizon)
        interval = forecast.conf_int(alpha=1 - self.level)
        log_frame = pd.DataFrame(
            {
                "mean": np.asarray(forecast.predicted_mean),
                "lower": interval.iloc[:, 0].to_numpy(),
                "upper": interval.iloc[:, 1].to_numpy(),
            }
        )
        log_frame = log_frame / SARIMA_LOG_SCALE
        return _back_transform(self, horizon, log_frame)


def _fit_sarimax(log_y: pd.Series, order, seasonal_order):
    model = SARIMAX(
        log_y,
        order=order,
        seasonal_order=seasonal_order,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = model.fit(disp=False, maxiter=200)
    _require_converged(result, f"SARIMA{order}x{seasonal_order}")
    return result


def select_sarima_orders(history: pd.DataFrame) -> tuple[tuple, tuple, pd.DataFrame]:
    """Pick SARIMA orders by AIC on ``history`` over a small grid with d = D = 1.

    Differencing is fixed because AIC is not comparable across differencing orders; a trend and
    seasonal pattern in deposits make d = D = 1 the standard choice for a monthly stock.
    Returns the best order, best seasonal order and the full table of AIC values.
    """
    log_y = np.log(history[TARGET].astype(float).to_numpy())
    log_y = pd.Series(SARIMA_LOG_SCALE * log_y)
    rows = []
    for p, q, sp, sq in itertools.product(_GRID_P, _GRID_Q, _GRID_SEASONAL_P, _GRID_SEASONAL_Q):
        order, seasonal = (p, 1, q), (sp, 1, sq, SEASONAL_PERIOD)
        try:
            aic = _fit_sarimax(log_y, order, seasonal).aic
        except _FitFailedError:
            continue
        rows.append({"order": order, "seasonal_order": seasonal, "aic": aic})
    if not rows:
        raise _FitFailedError("no SARIMA candidate in the grid could be fitted")
    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    best = table.iloc[0]
    return best["order"], best["seasonal_order"], table


def main(argv: list[str] | None = None) -> None:
    """Reproduce the recorded SARIMA orders from the data up to ``SELECTION_END``."""
    from deposit_forecasting.data import load_panel

    argparse.ArgumentParser(description=main.__doc__).parse_args(argv)
    panel = load_panel().loc[:SELECTION_END]
    order, seasonal, table = select_sarima_orders(panel)
    print(table.head(10).to_string())
    print(f"selected: order={order} seasonal_order={seasonal}")


if __name__ == "__main__":
    main()
