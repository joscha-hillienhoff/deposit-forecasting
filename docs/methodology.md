# Methodology

Each step adds a short section: what was decided, why, and which alternative was rejected.

## Data

**Target: overnight deposits of households.** Decision: forecast the end-of-month stock of
overnight deposits of households and non-profit institutions serving households (NPISH) at
German monetary financial institutions (excluding the central bank). Reason: the item includes
current-account balances, not only savings accounts, so it is the large, rate-sensitive and
operationally relevant part of household liquidity. Rejected: total deposits, which mix in
fixed-term products that behave differently.

**Not seasonally adjusted.** Decision: use the unadjusted series. Reason: seasonality (for
example the November effect around year-end and salary dates) is part of what a practical
forecast must reproduce, and it is modelled explicitly in later steps. Rejected: the
seasonally adjusted series, because the adjustment is itself a model, is revised over time and
would hide a pattern we want to compare models on.

**Sample start 2003-01.** Decision: the panel starts in 2003-01. Reason: the target series
starts there, and all regressors are available from then. Rejected: starting earlier with
missing target values, which would require imputation.

**Deposit facility rate as end-of-month level.** Decision: use the level in force on the last
day of the month, carried forward from the change dates. Reason: it matches the end-of-month
timing of the deposit stock and needs no assumption about intra-month timing. Rejected: the
monthly mean, which blends two levels in months with a change and no longer lines up with the
stock date.

**Spread.** Decision: define `spread = time_deposit_rate - overnight_rate` in percentage
points. Reason: it is the opportunity cost of holding money at call; when time deposits pay
clearly more, households have an incentive to shift out of overnight deposits, so it is the
main economic driver we expect. Rejected: using the two rates only as separate levels, which
hides that the decision depends on their difference.

**Provisional last observation.** Decision: keep the observation status of the last panel
month (`panel.attrs["last_deposit_status"]`, currently `P`); the panel ends at the last month
where the target and both bank rates exist, so this is the status of a month that is in the
panel. Reason: the latest value can be revised, so evaluation and the README must not present
it as final. Rejected: silently dropping it, which would discard the most recent information.

**Units and index.** Decision: deposits are converted from EUR million to EUR billion; the
index is a monthly `PeriodIndex` meaning the calendar month, with values referring to its end.
Reason: readable magnitudes, and no ambiguity about the day within the month. Rejected:
month-end timestamps, which suggest a precise day that the data does not carry.

**Cached and committed raw data.** Decision: store the unchanged API CSVs in `data/raw/`,
read from the cache by default and download only on an explicit `--refresh`, which first
downloads every series and then replaces each file atomically, so a failed refresh never leaves
a mix of vintages; commit the files.
Reason: results and tests are reproducible and run offline (including in CI), and later data
revisions cannot silently change reported numbers. Rejected: downloading on every run, which
makes results depend on the day they are produced and on network availability.

## Model interface and baselines

**One `fit`/`predict` interface.** Decision: every model implements `fit(history)` on a panel
slice ending at the forecast origin and `predict(horizon, exog=None)`, returning a DataFrame
indexed by the next `horizon` months with columns `mean`, `lower`, `upper`. Reason: the
backtest can treat all models alike, and `exog` already reserves room for driver paths (rates,
scenarios) of later models. Rejected: model-specific signatures, which would force special
cases into every evaluation.

**One nominal band level.** Decision: all models share the level `level`, default 0.8 (central
80 % interval), configurable. Reason: bands are only comparable if they claim the same
coverage; 80 % keeps bands informative at the small sample sizes of a rolling backtest.
Rejected: 95 %, whose wide bands are hard to check empirically with few forecasts. The level is
nominal; observed coverage is evaluated later, not assumed.

**Naive baseline.** Decision: mean = last value. Reason: the standard minimum benchmark that
any model must beat. Rejected: a historical mean, which ignores that the level is persistent.

**Drift baseline and its window.** Decision: mean = last value + h * average monthly change over
the trailing 36 months (`window`, or `None` for the full history). Reason: an auditable
expectation model with one parameter; a trailing window follows the recent trend, whereas the
full history averages over very different rate regimes. Rejected: a full-history default.
There is deliberately no seasonality: the baseline stays transparent and later models are
judged against it.

**Band of width z * sigma * sqrt(h).** Decision: sigma is the standard deviation of monthly
changes (in the drift window for the drift model, full history for the naive model). Reason:
for a random walk the h-step error variance is h * sigma^2. This assumes uncorrelated changes
with constant variance; autocorrelation or seasonality makes the band only approximate.
Rejected: a constant-width band, which ignores that uncertainty grows with the horizon.

**Planned outflows.** Decision: an optional schedule of known future outflows (month, EUR bn)
is subtracted cumulatively from the mean from its month onwards, and the band is shifted by the
same amount. Reason: announced departures are information the trend cannot contain, and a
transparent subtraction stays auditable. Rejected: folding them into the drift, which would
smear a known one-off event over all months. The schedule is an input: if the estimation
window already contains similar departures, the drift partly reflects them too.

**Synthetic data for outflows.** Decision: planned outflows are demonstrated on a small,
labelled synthetic deposit book of a fictional bank (retail trend plus noise, a few large
clients that leave on announced dates), generated deterministically from a seed. Reason: public
data exists only in aggregate, so client-level departures are not observable. Rejected: using
the ECB aggregate, where no such schedule exists. The ECB backtest uses the real data without
outflows; the synthetic book only illustrates the logic and proves nothing about real accuracy.
