import numpy as np
import pandas as pd
import pytest

from deposit_forecasting.models import DriftForecaster, NaiveForecaster


def make_history(values, start="2020-01") -> pd.DataFrame:
    index = pd.period_range(start, periods=len(values), freq="M")
    return pd.DataFrame({"deposits": values}, index=index)


@pytest.fixture
def history() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return make_history(100 + np.cumsum(rng.normal(0.5, 1.0, 60)))


@pytest.mark.parametrize("model", [NaiveForecaster(), DriftForecaster(), DriftForecaster(None)])
def test_interface_contract(model, history):
    out = model.fit(history).predict(12)
    assert list(out.columns) == ["mean", "lower", "upper"]
    assert isinstance(out.index, pd.PeriodIndex)
    assert out.index[0] == history.index[-1] + 1
    assert len(out) == 12
    assert (out.lower <= out["mean"]).all() and (out["mean"] <= out.upper).all()
    width = out.upper - out.lower
    assert width.is_monotonic_increasing and width.iloc[-1] > width.iloc[0]


def test_predict_before_fit_and_bad_input(history):
    with pytest.raises(RuntimeError):
        NaiveForecaster().predict(3)
    with pytest.raises(ValueError):
        NaiveForecaster().fit(history.reset_index(drop=True))
    with pytest.raises(ValueError):
        NaiveForecaster().fit(history).predict(0)
    with pytest.raises(ValueError):
        NaiveForecaster(level=1.0)


def test_naive_equals_last_value(history):
    out = NaiveForecaster().fit(history).predict(6)
    assert (out["mean"] == history["deposits"].iloc[-1]).all()


def test_band_level_is_configurable(history):
    narrow = NaiveForecaster(level=0.5).fit(history).predict(3)
    wide = NaiveForecaster(level=0.95).fit(history).predict(3)
    assert ((wide.upper - wide.lower) > (narrow.upper - narrow.lower)).all()


def test_naive_band_arithmetic():
    out = NaiveForecaster(level=0.8).fit(make_history([0.0, 1.0, 0.0, 1.0])).predict(4)
    sigma = np.std([1.0, -1.0, 1.0], ddof=1)
    z = 1.2815515655446004  # 90th percentile of the standard normal
    assert out["upper"].iloc[3] - out["mean"].iloc[3] == pytest.approx(z * sigma * 2)


def test_drift_arithmetic_on_toy_series():
    out = DriftForecaster(window=None).fit(make_history([10.0, 12.0, 14.0, 16.0])).predict(3)
    assert out["mean"].tolist() == pytest.approx([18.0, 20.0, 22.0])
    assert out["upper"].tolist() == pytest.approx(out["lower"].tolist())  # sigma = 0


def test_drift_window_parameter():
    hist = make_history([0.0, 10.0, 20.0, 21.0, 22.0, 23.0])  # changes 10, 10, 1, 1, 1
    last3 = DriftForecaster(window=3).fit(hist).predict(1)
    full = DriftForecaster(window=None).fit(hist).predict(1)
    assert last3["mean"].iloc[0] == pytest.approx(24.0)
    assert full["mean"].iloc[0] == pytest.approx(23.0 + 23.0 / 5)
    with pytest.raises(ValueError):
        DriftForecaster(window=1)


def test_planned_outflows_are_subtracted_cumulatively():
    hist = make_history([10.0, 12.0, 14.0, 16.0], start="2020-01")  # origin 2020-04
    schedule = pd.Series(
        [1.0, 0.5, 9.0, 9.0],
        index=pd.PeriodIndex(["2020-06", "2020-07", "2020-04", "2021-12"], freq="M"),
    )  # 2020-04 is the origin and 2021-12 is beyond the horizon: both ignored
    base = DriftForecaster(window=None).fit(hist).predict(4)
    planned = DriftForecaster(window=None, planned_outflows=schedule).fit(hist).predict(4)
    expected = [0.0, 1.0, 1.5, 1.5]
    assert (base - planned)["mean"].tolist() == pytest.approx(expected)
    assert (base - planned)["upper"].tolist() == pytest.approx(expected)
    assert (base - planned)["lower"].tolist() == pytest.approx(expected)
