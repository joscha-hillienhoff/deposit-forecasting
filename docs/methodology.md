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
nominal; observed coverage is evaluated by the backtest, not assumed.

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

## Backtesting

**Rolling origin instead of a single split.** Decision: forecast from many origins, one per
month, and average the errors per horizon. Reason: a single train/test split gives one
error path whose value depends on the split date (in this series, on whether it falls before
or after the rate turn) and leaves too few forecasts for stable coverage estimates. Rejected:
a single hold-out split.

**Expanding window.** Decision: each origin fits on all history up to and including the
origin, with a fresh model instance and no state carried over. Reason: the history is short
(under 25 years of monthly data), so discarding early observations costs information, and
models that want a shorter memory (such as the drift window) say so themselves. Rejected: a
sliding window, which would silently give every model the same short memory.

**No look-ahead.** Decision: the model receives only the panel rows up to the origin. Reason:
the evaluation is only meaningful if the information set is that of a real forecast at that
date. Caveat: the panel is the latest data vintage, so revisions to past values are not
reproduced.

**First origin 2010-12.** Decision: the first origin is configurable and defaults to 2010-12.
Reason: that leaves eight years (96 months) of history, enough for every planned model,
including seasonal ones and a 36-month drift window. Rejected: starting at the minimum any
model could run on, which would score the most data-starved fits.

**Horizons 1 to 12 months.** Decision: every origin forecasts up to 12 months, and a horizon is
scored only if its target month is realised, so the last origins contribute fewer horizons.
Reason: one year covers the planning range and one full seasonal cycle. Rejected: padding the
end with partial scores or dropping the last 11 origins, which would waste the latest data.
Near the end the models are asked only for the realised months, which for any model whose
forecast for month h does not depend on the requested horizon is the same as truncating a
12-month forecast.

**Ex-post conditional evaluation of drivers.** Decision: the realised driver columns (all
panel columns except `deposits`) of the forecast months are passed as `exog`. Reason: driver
models are then judged on how well they translate a rate path into deposits, not on how well
the rate path was guessed, and models that ignore `exog` are unaffected. This is not a real-time forecast: a user would have
to supply scenario paths instead. Rejected: forecasting the drivers too, which mixes two questions and
needs a second model.

**Segment by target month.** Decision: scores are reported for the full period and for the
2022-2023 rate turn, selected by the month being forecast (2022-01 to 2023-12); the windows
are a parameter. Reason: the turn is the stress test for a driver-based model, and a forecast
belongs to the regime of the month it predicts, however early it was made. Rejected: selecting
by origin, which would mix forecasts for the turn with forecasts for the calm period after it.
Within the turn the number of forecasts is small (24 per horizon), so differences between
models there are indicative.

**MAE as the main metric.** Decision: the headline metric is MAE in EUR bn. Reason: it is in
the unit of the target, easy to explain, and weights every euro of error equally. Rejected:
RMSE, whose weight on single large errors makes the ranking depend on a few months.

**MAPE as a secondary metric.** Decision: report MAPE (%) next to MAE. Reason: it lets a reader
judge the error relative to the level. It misleads when the series is near zero or changes
sign (the denominator explodes), for flows instead of stocks (the same absolute error looks
huge in a quiet month), when over- and under-forecasts are treated unequally (the same
absolute error is a larger percentage when the actual is small, so MAPE favours forecasts that
are too high), and when comparing series of different scale or level. Deposits are a large,
strictly positive stock, so none of these dominates here, but the trend makes the percentage
error shrink mechanically as the level rises, so MAPE is never the only metric. Rejected:
sMAPE, which fixes one asymmetry but has its own.

**Coverage next to the nominal level.** Decision: the share of actuals inside
`[lower, upper]` is shown next to the nominal level (default 80 %). Reason: a band is only
useful if its observed coverage is close to its claim; a much lower value means the band is
too narrow, a much higher one that it is uninformative. Coverage is a mean over overlapping
forecasts from adjacent origins, which are strongly dependent, so small gaps to the nominal
level are not significant. Rejected: interval scores, which combine width and coverage in one
number that is harder to read.

## Gradient boosting

**Target: cumulative log change, not the level.** Decision: the model predicts
`log(deposits[t+h]) - log(deposits[t])` and the forecast is the last value times the
exponential of the prediction. Reason: trees predict piecewise-constant values and cannot
extrapolate a trending level beyond the range seen in training; a change is roughly stationary
and a log change makes the effect proportional to the level. Rejected: fitting the level
directly, which would flatten out below the latest values.

**Direct strategy.** Decision: one model per horizon `h` (1 to 12), each trained on the
cumulative change from the origin to `origin + h`. Reason: errors do not compound through
recursive feedback of the model's own forecasts, and every horizon gets its own band.
Rejected: a recursive one-step model, which needs forecasts of all features in between and
accumulates their errors. The price is that the horizons are fitted independently and the
longest ones have the fewest training rows.

**Features.** Decision: twelve named features, all known at the origin except the last. Lags
of the monthly log change (`lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`), the rolling 12-month
change (`roll_12`), the spread and the deposit facility rate at the origin (`spread`,
`deposit_facility_rate`) and their recent changes (`spread_chg_3`, `dfr_chg_6`), the calendar
month of the target (`target_month`) and the spread of the target month (`spread_target`,
taken from `exog`, so ex-post conditional exactly like the backtest of the other driver
models). Reason: momentum, seasonality and the rate drivers named in the data section, with
few enough features for 100 to 280 training rows. Rejected: a long list of lags and
transformations, which invites overfitting and makes the SHAP picture unreadable. Training
rows start 12 months into the history because the longest lag needs them.

**Quantile band.** Decision: per horizon, three LightGBM models with the quantile objective at
the lower tail, 0.5 and the upper tail of the shared level (10 %, 50 %, 90 % for the default
80 %). The median model is the point forecast ("mean" in the interface), and the three values
are sorted per forecast month so that `lower <= mean <= upper` always holds. Reason: the band
comes from the model's own conditional distribution and needs no Gaussian assumption.
Rejected: a symmetric band from in-sample residuals, which understates uncertainty because
the in-sample fit is better than the out-of-sample one. The median is the natural point
forecast of the quantile objective, but it differs from a conditional mean if the
distribution is skewed.

**Fixed, conservative hyperparameters.** Decision: 100 rounds, learning rate 0.05, trees with
at most 4 leaves and depth 2, at least 15 rows per leaf, L2 regularisation 5, no row or
feature subsampling, fixed seed, single-threaded deterministic fits. They were set before the
first backtest from general guidance for small tabular data and were never changed afterwards;
nothing is tuned on the evaluation period (the backtest was run once with them). Reason:
with a few hundred rows the variance of a flexible model dominates, and a tuning loop over
overlapping rolling-origin forecasts would leak the evaluation into the model. Rejected:
per-origin or global hyperparameter search, which would need a nested validation scheme
that the short history cannot afford.

**SHAP for transparency.** Decision: `deposit-forecasting-explain` fits the model on the
full history and explains the median model of the 1-month and 12-month horizons with
`TreeExplainer`; it writes summary plots to `docs/figures/` and prints the mean absolute SHAP
value per feature (in log-change units). Reason: SHAP values are exact and additive for trees
and show which drivers move a forecast and in which direction. They describe the fitted
model's use of the features, not causal effects, and they are computed on the training rows.
Rejected: built-in split-count importances, which show no direction and overstate
high-cardinality features.

**Why a simple model can beat a complex one here.** About 280 monthly observations are few for
a flexible learner: its variance is large relative to the signal, the 12-month models have
even fewer rows, and consecutive monthly rows overlap heavily, so the effective sample is
smaller than it looks. The series contains a handful of regimes (low rates for a decade, the
rate turn of 2022-2023), so the model has seen few examples of a regime change and has to
extrapolate when one arrives; trees cannot extrapolate beyond the feature range they saw, and
a plain drift or a linear rate regression can. A simple model has few parameters to
estimate and degrades gracefully; a complex one only wins if the extra flexibility captures
structure that the sample can pin down. Rejected: assuming that the more complex model is the
better one. The backtest decides, per horizon and segment.
