import numpy as np
import pandas as pd
import pytest

from deposit_forecasting import classical
from deposit_forecasting.classical import (
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER,
    EtsForecaster,
    SarimaForecaster,
)


def seasonal_history(n=72, start="2018-01") -> pd.DataFrame:
    rng = np.random.default_rng(3)
    t = np.arange(n)
    season = 0.03 * np.sin(2 * np.pi * t / 12)
    log_y = np.log(100) + 0.004 * t + season + rng.normal(0, 0.003, n)
    return pd.DataFrame(
        {"deposits": np.exp(log_y)}, index=pd.period_range(start, periods=n, freq="M")
    )


MODELS = [EtsForecaster, SarimaForecaster]


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return seasonal_history()


@pytest.mark.parametrize("cls", MODELS)
def test_interface_contract(cls, history):
    out = cls().fit(history).predict(12)
    assert list(out.columns) == ["mean", "lower", "upper"]
    assert isinstance(out.index, pd.PeriodIndex)
    assert out.index[0] == history.index[-1] + 1 and len(out) == 12
    assert (out.lower <= out["mean"]).all() and (out["mean"] <= out.upper).all()
    assert (out.lower > 0).all()
    width = out.upper - out.lower
    assert width.iloc[-1] > width.iloc[0]


@pytest.mark.parametrize("cls", MODELS)
def test_deterministic(cls, history):
    a = cls().fit(history).predict(6)
    b = cls().fit(history).predict(6)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("cls", MODELS)
def test_seasonal_pattern_is_reproduced(cls, history):
    out = cls().fit(history).predict(12)
    t = np.arange(len(history), len(history) + 12)
    expected = 0.03 * np.sin(2 * np.pi * t / 12)
    log_mean = np.log(out["mean"].to_numpy())
    detrended = log_mean - 0.004 * t
    assert np.corrcoef(detrended, expected)[0, 1] > 0.9


@pytest.mark.parametrize("cls", MODELS)
def test_band_level_and_errors(cls, history):
    narrow = cls(level=0.5).fit(history).predict(3)
    wide = cls(level=0.95).fit(history).predict(3)
    assert ((wide.upper - wide.lower) > (narrow.upper - narrow.lower)).all()
    with pytest.raises(RuntimeError):
        cls().predict(3)
    with pytest.raises(ValueError):
        cls().fit(history.reset_index(drop=True))
    with pytest.raises(ValueError):
        cls().fit(history.assign(deposits=-1.0))


def test_ets_needs_two_seasons():
    with pytest.raises(ValueError):
        EtsForecaster().fit(seasonal_history(n=20))


def test_sarima_orders_are_the_recorded_constants():
    model = SarimaForecaster()
    assert (model.order, model.seasonal_order) == ((1, 1, 2), (0, 1, 1, 12))
    assert model.order == SARIMA_ORDER and model.seasonal_order == SARIMA_SEASONAL_ORDER
    fitted = model.fit(seasonal_history())
    assert fitted.result_.model.order == SARIMA_ORDER
    assert fitted.result_.model.seasonal_order == SARIMA_SEASONAL_ORDER


def test_failed_fit_raises(monkeypatch):
    class NotConverged:
        llf = 1.0
        params = np.array([1.0])
        mle_retvals = {"converged": False}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, **kwargs):
            return NotConverged()

    monkeypatch.setattr(classical, "SARIMAX", FakeModel)
    with pytest.raises(RuntimeError, match="did not converge"):
        SarimaForecaster().fit(seasonal_history())
