import numpy as np
import pandas as pd
import pytest

from deposit_forecasting import backtest
from deposit_forecasting.backtest import RATE_TURN, run_backtest, score
from deposit_forecasting.models import Forecaster, NaiveForecaster


def make_panel(months: int = 30, start: str = "2020-01") -> pd.DataFrame:
    index = pd.period_range(start, periods=months, freq="M")
    return pd.DataFrame(
        {
            "deposits": 100.0 + np.arange(months, dtype=float),
            "overnight_rate": 0.1,
            "spread": 0.5,
        },
        index=index,
    )


class Probe(Forecaster):
    """Records what it sees; forecasts the last value."""

    instances: list["Probe"] = []

    def __init__(self) -> None:
        super().__init__()
        self.history: pd.DataFrame | None = None
        self.exog: pd.DataFrame | None = None
        Probe.instances.append(self)

    def fit(self, history):
        self.history = history
        self.origin_ = history.index[-1]
        return self

    def predict(self, horizon, exog=None):
        self.exog = exog
        last = float(self.history["deposits"].iloc[-1])
        return self._frame(horizon, np.full(horizon, last), np.full(horizon, 1.0))


def test_no_lookahead_and_exog_are_realised_drivers():
    Probe.instances = []
    panel = make_panel()
    run_backtest(panel, {"probe": Probe}, first_origin="2020-06")
    for probe in Probe.instances:
        origin = probe.history.index[-1]
        assert probe.history.equals(panel.loc[:origin])
        assert list(probe.exog.columns) == ["overnight_rate", "spread"]
        assert probe.exog.index[0] == origin + 1
        assert len(probe.exog) <= 12


def test_fresh_instance_per_origin():
    Probe.instances = []
    run_backtest(make_panel(), {"probe": Probe}, first_origin="2020-06")
    assert len(Probe.instances) == len({id(p) for p in Probe.instances}) == 24


def test_origin_and_horizon_counts_with_truncated_end():
    panel = make_panel(30)
    table = run_backtest(panel, {"naive": NaiveForecaster}, first_origin="2020-06")
    # origins 2020-06 .. 2022-05 (the last month has nothing after it)
    assert table["origin"].nunique() == 24
    assert table["origin"].min() == pd.Period("2020-06", freq="M")
    assert table["origin"].max() == pd.Period("2022-05", freq="M")
    per_origin = table.groupby("origin")["horizon"].max()
    assert (per_origin.iloc[:-12] == 12).all()
    assert list(per_origin.iloc[-11:]) == list(range(11, 0, -1))
    assert (table["target"] <= panel.index[-1]).all()
    assert (table["target"] == table["origin"] + table["horizon"]).all()
    assert list(table.columns) == backtest.FORECAST_COLUMNS


def test_unknown_first_origin_rejected():
    with pytest.raises(ValueError):
        run_backtest(make_panel(), {"naive": NaiveForecaster}, first_origin="2019-01")


def toy_forecasts() -> pd.DataFrame:
    month = pd.Period("2022-03", freq="M")
    return pd.DataFrame(
        {
            "model": "m",
            "origin": month,
            "horizon": [1, 1, 1, 1],
            "target": pd.period_range("2021-11", periods=4, freq="M"),
            "actual": [100.0, 200.0, 50.0, 100.0],
            "mean": [110.0, 190.0, 50.0, 80.0],
            "lower": [105.0, 195.0, 40.0, 105.0],
            "upper": [115.0, 205.0, 60.0, 125.0],
        }
    )


def test_metric_arithmetic():
    result = score(toy_forecasts(), segments={"all": None}, level=0.8).iloc[0]
    assert result["n"] == 4
    assert result["mae"] == pytest.approx((10 + 10 + 0 + 20) / 4)
    assert result["mape"] == pytest.approx(100 * (0.1 + 0.05 + 0 + 0.2) / 4)
    # inside the band: row 1 no (100 < 105), rows 2 and 3 yes, row 4 no (100 < 105)
    assert result["coverage"] == pytest.approx(0.5)
    assert result["nominal"] == 0.8


def test_segment_filters_on_target_month_inclusive():
    result = score(toy_forecasts(), segments={"turn": RATE_TURN})
    # targets 2021-11, 2021-12, 2022-01, 2022-02: only the last two are in the window
    assert result["n"].iloc[0] == 2
    assert result["segment"].iloc[0] == "turn"
    custom = pd.Period("2021-12", freq="M"), pd.Period("2021-12", freq="M")
    assert score(toy_forecasts(), segments={"x": custom})["n"].iloc[0] == 1


def test_score_columns_stable():
    result = score(toy_forecasts())
    assert list(result.columns) == backtest.SCORE_COLUMNS
    assert set(result["segment"]) == {"full", "rate_turn"}


def test_cli_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(backtest, "load_panel", lambda: make_panel(60, "2016-01"))
    backtest.main(["--first-origin", "2018-01", "--out", str(tmp_path / "out")])
    printed = capsys.readouterr().out
    assert all(name in printed for name in backtest.DEFAULT_MODELS)
    forecasts = pd.read_csv(tmp_path / "out" / "forecasts.csv")
    scores = pd.read_csv(tmp_path / "out" / "scores.csv")
    assert list(forecasts.columns) == backtest.FORECAST_COLUMNS
    assert list(scores.columns) == backtest.SCORE_COLUMNS
