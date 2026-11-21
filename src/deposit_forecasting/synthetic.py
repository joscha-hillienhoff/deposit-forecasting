"""SYNTHETIC deposit book of one fictional bank, for the planned-outflow logic only.

Public data exists only in aggregate, so announced departures of large clients cannot be shown
on the ECB series. This module generates a small, fully SYNTHETIC book instead: a retail
component (linear trend plus noise) and a few large clients ("Synthetic Client 01", ...) that
leave at announced dates. Nothing here is real data; amounts are in EUR bn like the ECB panel.
The ECB backtest runs on the real aggregate data without outflows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deposit_forecasting.models import DriftForecaster

LABEL = "SYNTHETIC DATA (fictional bank, not real deposits)"
N_MONTHS = 108
HORIZON = 12  # departures fall in the last HORIZON months; the demo origin sits just before


@dataclass(frozen=True)
class SyntheticBook:
    """A synthetic deposit book and its announced outflow schedule."""

    label: str
    panel: pd.DataFrame  # columns: deposits (total), retail, large_clients; EUR bn
    planned_outflows: pd.Series  # departure month -> balance that leaves, EUR bn
    departures: pd.DataFrame  # one row per "Synthetic Client NN"


def generate_synthetic_book(seed: int = 0, n_clients: int = 5) -> SyntheticBook:
    """Deterministic synthetic book; the same ``seed`` always gives the same data."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("2015-01", periods=N_MONTHS, freq="M")
    t = np.arange(N_MONTHS)
    retail = 5.0 + 0.02 * t + rng.normal(0.0, 0.03, N_MONTHS)

    names = [f"Synthetic Client {i:02d}" for i in range(1, n_clients + 1)]
    balances = np.round(rng.uniform(0.2, 0.6, n_clients), 3)
    leave_pos = np.sort(rng.choice(np.arange(N_MONTHS - HORIZON, N_MONTHS), n_clients, False))
    clients = pd.DataFrame(
        {
            name: np.where(t < pos, bal, 0.0)
            for name, bal, pos in zip(names, balances, leave_pos, strict=True)
        },
        index=index,
    )
    departures = pd.DataFrame(
        {"balance": balances, "departure_month": index[leave_pos]}, index=names
    )
    panel = pd.DataFrame(
        {"retail": retail, "large_clients": clients.sum(axis=1)}, index=index
    ).assign(deposits=lambda d: d.retail + d.large_clients)[["deposits", "retail", "large_clients"]]
    outflows = pd.Series(balances, index=index[leave_pos], name="planned_outflow").sort_index()
    return SyntheticBook(LABEL, panel, outflows, departures)


def demonstrate(book: SyntheticBook | None = None) -> pd.DataFrame:
    """Drift forecast with and without planned outflows against the realised synthetic path."""
    book = book or generate_synthetic_book()
    history = book.panel.iloc[:-HORIZON]
    plain = DriftForecaster().fit(history).predict(HORIZON)
    planned = DriftForecaster(planned_outflows=book.planned_outflows).fit(history).predict(HORIZON)
    out = pd.DataFrame(
        {
            "realised": book.panel["deposits"].iloc[-HORIZON:],
            "drift": plain["mean"],
            "drift_with_outflows": planned["mean"],
        }
    )
    out.attrs["label"] = book.label
    return out


def main() -> None:
    result = demonstrate()
    print(result.attrs["label"])
    print(result.round(3).to_string())
    for col in ("drift", "drift_with_outflows"):
        mae = (result[col] - result["realised"]).abs().mean()
        print(f"{col}: mean absolute error {mae:.3f} EUR bn (synthetic)")


if __name__ == "__main__":
    main()
