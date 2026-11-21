"""Offline tests: everything runs from the committed cache in data/raw/."""

import pandas as pd
import pytest
import requests

from deposit_forecasting import data


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return data.load_panel()


def test_period_and_rows(panel):
    assert panel.index[0] == pd.Period("2003-01", freq="M")
    assert panel.index[-1] == pd.Period("2026-09", freq="M")
    assert len(panel) == 285
    assert panel.index.is_monotonic_increasing
    assert panel.index.is_unique
    assert not panel.index.to_series().diff().dropna().ne(pd.offsets.MonthEnd(1)).any()


def test_columns_and_no_missing(panel):
    assert list(panel.columns) == [
        "deposits",
        "overnight_rate",
        "time_deposit_rate",
        "spread",
        "deposit_facility_rate",
    ]
    assert not panel.isna().any().any()


def test_spread_arithmetic(panel):
    expected = panel["time_deposit_rate"] - panel["overnight_rate"]
    pd.testing.assert_series_equal(panel["spread"], expected, check_names=False)
    assert panel.loc["2026-09", "spread"].item() == pytest.approx(2.39 - 0.52)


def test_deposit_reference_values(panel):
    assert panel.loc["2026-09", "deposits"].item() == pytest.approx(1939.581)  # EUR bn
    assert panel.loc["2003-01", "deposits"].item() == pytest.approx(369, abs=1)


@pytest.mark.parametrize(
    ("month", "level"), [("2023-12", 4.00), ("2022-06", -0.50), ("2026-09", 2.50)]
)
def test_deposit_facility_rate_end_of_month(panel, month, level):
    assert panel.loc[month, "deposit_facility_rate"].item() == pytest.approx(level)


def test_deposit_facility_rate_uses_level_in_force_at_month_end(tmp_path):
    csv = "TIME_PERIOD,OBS_VALUE\n2023-09-20,4.00\n2023-10-04,4.00\n2023-10-31,3.50\n"
    raw = csv.encode()
    months = pd.period_range("2023-09", "2023-11", freq="M")
    series = data.SERIES_BY_COLUMN["deposit_facility_rate"]
    result = data._month_end_policy_rate(series, raw, months)
    assert result.tolist() == [4.00, 3.50, 3.50]


def test_last_observation_status_is_visible(panel):
    assert panel.attrs["last_deposit_status"] == "P"


def test_cache_first_makes_no_network_call(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("network used although the cache exists")

    monkeypatch.setattr(requests, "get", fail)
    assert len(data.load_panel()) == 285


def _fake_response(monkeypatch, fail_on=None):
    calls = []

    class Response:
        content = b"TIME_PERIOD,OBS_VALUE\n2003-01,1\n"

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        calls.append(url)
        if fail_on is not None and len(calls) >= fail_on:
            raise requests.ConnectionError("down")
        return Response()

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    return calls, Response.content


def test_refresh_downloads_and_overwrites_cache(monkeypatch, tmp_path):
    calls, content = _fake_response(monkeypatch)
    raws = data.fetch_all(refresh=True, raw_dir=tmp_path)
    assert calls == [s.url for s in data.SERIES]
    assert set(raws.values()) == {content}
    assert all(s.cache_path(tmp_path).read_bytes() == content for s in data.SERIES)
    # A second default call is served from the new cache.
    data.fetch_all(raw_dir=tmp_path)
    assert len(calls) == len(data.SERIES)


def test_failed_refresh_leaves_cache_untouched(monkeypatch, tmp_path):
    old = b"TIME_PERIOD,OBS_VALUE\n2003-01,9\n"
    for s in data.SERIES:
        s.cache_path(tmp_path).write_bytes(old)
    _fake_response(monkeypatch, fail_on=3)
    with pytest.raises(RuntimeError):
        data.fetch_all(refresh=True, raw_dir=tmp_path)
    assert all(s.cache_path(tmp_path).read_bytes() == old for s in data.SERIES)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        s.cache_path(tmp_path).name for s in data.SERIES
    )


def test_status_refers_to_last_panel_month(tmp_path):
    for s in data.SERIES:
        s.cache_path(tmp_path).write_bytes(s.cache_path().read_bytes())
    path = data.SERIES_BY_COLUMN["deposits"].cache_path(tmp_path)
    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    row = lines[-1].split(",")
    row[header.index("TIME_PERIOD")] = "2099-01"
    row[header.index("OBS_STATUS")] = "X"
    path.write_text("\n".join([*lines, ",".join(row)]) + "\n")
    panel = data.load_panel(raw_dir=tmp_path)
    assert panel.index[-1] < pd.Period("2099-01", freq="M")
    assert panel.attrs["last_deposit_status"] == "P"


def test_download_retries_then_fails(monkeypatch):
    attempts = []

    def flaky(url, timeout):
        attempts.append(url)
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", flaky)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError):
        data.download(data.SERIES[0], retries=3)
    assert len(attempts) == 3


def test_registry_has_four_series_with_cached_files():
    assert [s.column for s in data.SERIES] == [
        "deposits",
        "overnight_rate",
        "time_deposit_rate",
        "deposit_facility_rate",
    ]
    assert all(s.cache_path().exists() for s in data.SERIES)
