"""Download, cache and assemble the ECB series into a monthly panel.

Panel convention: the index is a monthly ``pandas.PeriodIndex`` (freq ``"M"``). A period
stands for the whole calendar month; stock and rate values refer to the end of that month
(deposits are end-of-month outstanding amounts, the deposit facility rate is the level in
force on the last day of the month).
"""

from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

API_BASE = "https://data-api.ecb.europa.eu/service/data"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SAMPLE_START = pd.Period("2003-01", freq="M")  # the target series starts here
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 4
DEFAULT_BACKOFF = 2.0


@dataclass(frozen=True)
class Series:
    """One ECB series: where it lives and what it means."""

    column: str
    dataset: str
    key: str  # key without the dataset prefix
    description: str
    unit: str

    @property
    def full_key(self) -> str:
        return f"{self.dataset}.{self.key}"

    @property
    def url(self) -> str:
        return f"{API_BASE}/{self.dataset}/{self.key}?format=csvdata"

    def cache_path(self, raw_dir: Path | None = None) -> Path:
        return (raw_dir or RAW_DIR) / f"{self.column}.csv"


SERIES: tuple[Series, ...] = (
    Series(
        "deposits",
        "BSI",
        "M.DE.N.A.L21.A.1.U2.2250.Z01.E",
        "Overnight deposits of euro area households and NPISH at German MFIs excl. central "
        "bank, outstanding amount at end of month, not seasonally adjusted",
        "EUR million",
    ),
    Series(
        "overnight_rate",
        "MIR",
        "M.DE.B.L21.A.R.A.2250.EUR.N",
        "Bank rate on household overnight deposits, new business, Germany",
        "% p.a.",
    ),
    Series(
        "time_deposit_rate",
        "MIR",
        "M.DE.B.L22.F.R.A.2250.EUR.N",
        "Bank rate on household deposits with agreed maturity up to 1 year, new business, Germany",
        "% p.a.",
    ),
    Series(
        "deposit_facility_rate",
        "FM",
        "B.U2.EUR.4F.KR.DFR.LEV",
        "ECB deposit facility rate, level (change dates only)",
        "% p.a.",
    ),
)
SERIES_BY_COLUMN = {s.column: s for s in SERIES}


def download(
    series: Series,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> bytes:
    """Fetch the raw CSV from the ECB API, retrying with exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(series.url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * 2**attempt)
    raise RuntimeError(f"Download of {series.full_key} failed after {retries} attempts") from (
        last_error
    )


def fetch_all(*, refresh: bool = False, raw_dir: Path | None = None) -> dict[str, bytes]:
    """Return the raw CSV bytes of every series, keyed by column name.

    The cache is read unless a refresh is requested or a file is missing. All needed
    downloads finish before any file is written, and each file is replaced atomically, so
    a failed download or an interruption never leaves a mix of vintages in the cache. The
    API response is stored unchanged, so the cache is a faithful copy of the source.
    """
    raws: dict[str, bytes] = {}
    fetched: dict[Path, bytes] = {}
    for series in SERIES:
        path = series.cache_path(raw_dir)
        if not refresh and path.exists():
            raws[series.column] = path.read_bytes()
        else:
            raws[series.column] = fetched[path] = download(series)
    for path, content in fetched.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)
    return raws


def _parse(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), dtype={"OBS_STATUS": "string"})


def _monthly(series: Series, raw: bytes) -> pd.Series:
    frame = _parse(raw)
    values = pd.Series(
        frame["OBS_VALUE"].to_numpy(dtype=float),
        index=pd.PeriodIndex(frame["TIME_PERIOD"], freq="M"),
        name=series.column,
    )
    if series.column == "deposits":
        values = values / 1000.0  # EUR million -> EUR billion
    return values


def _month_end_policy_rate(series: Series, raw: bytes, months: pd.PeriodIndex) -> pd.Series:
    """Level of the rate in force on the last day of each month.

    The API lists only the dates on which the rate changed; the level is carried forward
    from the latest change on or before the month end.
    """
    frame = _parse(raw)
    changes = pd.Series(
        frame["OBS_VALUE"].to_numpy(dtype=float), index=pd.to_datetime(frame["TIME_PERIOD"])
    ).sort_index()
    month_ends = months.to_timestamp(how="end").normalize()
    levels = [changes.asof(day) for day in month_ends]
    return pd.Series(levels, index=months, name=series.column)


def load_panel(*, refresh: bool = False, raw_dir: Path | None = None) -> pd.DataFrame:
    """Build the monthly panel from the cached raw series.

    Columns: ``deposits`` (EUR bn), ``overnight_rate``, ``time_deposit_rate``, ``spread``
    (time deposit rate minus overnight rate, percentage points) and ``deposit_facility_rate``
    (end-of-month level). The sample runs from 2003-01 to the last month in which the target
    and both MIR rates exist. ``panel.attrs["last_deposit_status"]`` holds the observation
    status of the last panel month.
    """
    raws = fetch_all(refresh=refresh, raw_dir=raw_dir)
    deposits = _monthly(SERIES_BY_COLUMN["deposits"], raws["deposits"])
    overnight = _monthly(SERIES_BY_COLUMN["overnight_rate"], raws["overnight_rate"])
    time_dep = _monthly(SERIES_BY_COLUMN["time_deposit_rate"], raws["time_deposit_rate"])
    core = pd.concat([deposits, overnight, time_dep], axis=1).dropna()
    core = core.loc[core.index >= SAMPLE_START]
    panel = core.assign(spread=core["time_deposit_rate"] - core["overnight_rate"])
    panel["deposit_facility_rate"] = _month_end_policy_rate(
        SERIES_BY_COLUMN["deposit_facility_rate"], raws["deposit_facility_rate"], panel.index
    )
    panel = panel[
        ["deposits", "overnight_rate", "time_deposit_rate", "spread", "deposit_facility_rate"]
    ]
    panel.index.name = "month"
    status_frame = _parse(raws["deposits"])
    statuses = pd.Series(
        status_frame["OBS_STATUS"].to_numpy(),
        index=pd.PeriodIndex(status_frame["TIME_PERIOD"], freq="M"),
    )
    panel.attrs["last_deposit_status"] = str(statuses.loc[panel.index[-1]])
    return panel


def summary(panel: pd.DataFrame) -> str:
    missing = int(panel.isna().sum().sum())
    return "\n".join(
        [
            f"period: {panel.index[0]} .. {panel.index[-1]}",
            f"rows: {len(panel)}",
            f"missing values: {missing}",
            f"last deposits observation status: {panel.attrs['last_deposit_status']}"
            " (A = final, P = provisional)",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="deposit-forecasting-data",
        description="Load the ECB series (from cache by default) and print a short summary.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="download again and overwrite data/raw/"
    )
    args = parser.parse_args(argv)
    print(summary(load_panel(refresh=args.refresh)))


if __name__ == "__main__":
    main()
