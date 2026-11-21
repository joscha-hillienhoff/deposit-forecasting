"""Interest-rate scenarios for household overnight deposits.

A scenario is a parallel shift of the ECB deposit facility rate from the last observed month,
held for the whole horizon. The shift is passed through to the overnight and the time-deposit
rate with an estimated partial-adjustment regression, the resulting spread path drives
:class:`~deposit_forecasting.regression.RateRegressionForecaster`. The baseline keeps all rates
at their last observed values. See ``docs/methodology.md`` for the design decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from deposit_forecasting.data import load_panel
from deposit_forecasting.models import DEFAULT_LEVEL, TARGET
from deposit_forecasting.regression import RateRegressionForecaster

POLICY_RATE = "deposit_facility_rate"
BANK_RATES = ("overnight_rate", "time_deposit_rate")
HORIZON = 24
SHOCKS_BP = {"baseline": 0, "+100bp": 100, "-100bp": -100}
SCENARIO_COLUMNS = ["scenario", "month", "mean", "lower", "upper", "spread"]
DEFAULT_FIGURE = Path(__file__).resolve().parents[2] / "docs" / "figures" / "rate_scenarios.png"
HISTORY_MONTHS = 60


def estimate_pass_through(panel: pd.DataFrame) -> pd.DataFrame:
    """Partial-adjustment regression of each bank rate on the deposit facility rate.

    ``rate_t = c + phi * rate_{t-1} + b * policy_t`` by OLS. The immediate pass-through is
    ``b``, the long-run pass-through ``b / (1 - phi)``. Returns one row per bank rate.
    """
    rows = {}
    for name in BANK_RATES:
        data = pd.DataFrame(
            {"lag": panel[name].shift(1), "policy": panel[POLICY_RATE], "y": panel[name]}
        ).dropna()
        params = sm.OLS(data["y"], sm.add_constant(data[["lag", "policy"]])).fit().params
        rows[name] = {
            "phi": params["lag"],
            "impact": params["policy"],
            "long_run": params["policy"] / (1 - params["lag"]),
        }
    return pd.DataFrame(rows).T


def rate_paths(
    panel: pd.DataFrame, shock_bp: float, horizon: int = HORIZON, pass_through=None
) -> pd.DataFrame:
    """Bank-rate and spread paths for a parallel policy-rate shift of ``shock_bp`` basis points.

    Each bank rate is its last observed value plus the dynamic response to the shift implied by
    the pass-through regression: ``delta_h = phi * delta_{h-1} + b * shift`` with ``delta_0 = 0``.
    """
    if pass_through is None:
        pass_through = estimate_pass_through(panel)
    index = pd.period_range(panel.index[-1] + 1, periods=horizon, freq="M")
    shift = shock_bp / 100
    paths = pd.DataFrame(index=index)
    for name in BANK_RATES:
        phi, impact = pass_through.loc[name, ["phi", "impact"]]
        delta, previous = np.empty(horizon), 0.0
        for h in range(horizon):
            previous = delta[h] = phi * previous + impact * shift
        paths[name] = panel[name].iloc[-1] + delta
    paths["spread"] = paths["time_deposit_rate"] - paths["overnight_rate"]
    return paths


def run_scenarios(
    panel: pd.DataFrame, *, horizon: int = HORIZON, level: float = DEFAULT_LEVEL
) -> pd.DataFrame:
    """Forecast deposits for the baseline and the +/-100 bp shocks.

    Returns a tidy table with the columns ``SCENARIO_COLUMNS``; ``spread`` is the driver path
    in percentage points.
    """
    model = RateRegressionForecaster(level=level).fit(panel)
    pass_through = estimate_pass_through(panel)
    tables = []
    for scenario, shock in SHOCKS_BP.items():
        paths = rate_paths(panel, shock, horizon, pass_through)
        forecast = model.predict(horizon, exog=paths[["spread"]])
        table = forecast.assign(scenario=scenario, spread=paths["spread"]).rename_axis("month")
        tables.append(table.reset_index())
    return pd.concat(tables, ignore_index=True)[SCENARIO_COLUMNS]


def plot_scenarios(
    panel: pd.DataFrame, scenarios: pd.DataFrame, path: Path, *, level: float = DEFAULT_LEVEL
) -> None:
    """Save the chart: recent history plus the three scenario paths with their bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#7f7f7f", "+100bp": "#d95f02", "-100bp": "#1f77b4"}
    recent = panel[TARGET].iloc[-HISTORY_MONTHS:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(recent.index.to_timestamp(), recent, color="black", label="Observed")
    for name, part in scenarios.groupby("scenario", sort=False):
        x = part["month"].dt.to_timestamp()
        ax.plot(
            [recent.index[-1].to_timestamp(), *x],
            [recent.iloc[-1], *part["mean"]],
            color=colors[name],
            label=name,
        )
        ax.fill_between(x, part["lower"], part["upper"], color=colors[name], alpha=0.12)
    ax.set_ylabel("Household overnight deposits, Germany (EUR bn)")
    ax.set_xlabel("Month")
    ax.set_title(
        "Rate scenarios: parallel shift of the ECB deposit facility rate\n"
        f"(shaded: nominal {level:.0%} bands)"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, "Source: ECB Data Portal; own calculations", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="deposit-forecasting-scenarios",
        description="Forecast deposits under +/-100 bp rate shocks and save the scenario chart.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_FIGURE, help="figure path (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    panel = load_panel()
    scenarios = run_scenarios(panel)
    print(estimate_pass_through(panel).round(3).to_string())
    last = scenarios.groupby("scenario", sort=False).tail(1)
    print(last.to_string(index=False, float_format="{:.2f}".format))
    plot_scenarios(panel, scenarios, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
