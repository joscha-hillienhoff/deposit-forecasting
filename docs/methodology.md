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

## Rate regression and scenarios

**No look-ahead in the specification.** Decision: the form of the model (target, regressors, lag
of one month, seasonal dummies, one autoregressive term) was fixed in advance from economic
reasoning and is not searched or tuned; the only estimated quantities are OLS coefficients, which
the backtest refits at every origin on data up to that origin. Reason: any choice made while
looking at the evaluation period would leak into the scores. Rejected: selecting lags or
regressors by backtest error.

**Log change instead of level target.** Decision: regress the monthly log change of deposits.
Reason: the level is trending and strongly persistent, so a level regression mostly fits the
trend; the change is close to stationary, and log changes are growth rates, which are comparable
between the low and the high deposit levels in the sample. Rejected: a regression in levels,
which risks spurious fits between trending series.

**Spread as the driver, lagged by one month.** Decision: the regressor is the spread between
time-deposit and overnight rate of the previous month-end. Reason: the deposit stock is a
month-end value, so flows during month t respond to the incentive visible at the end of month
t-1; this also avoids simultaneity between the rates and the stock of the same month, and it
means the first forecast month does not depend on a rate path. Rejected: the contemporaneous
spread, and the two rates as separate regressors, which hides that households react to the
difference.

**Seasonal dummies.** Decision: eleven monthly dummies (January is the reference). Reason: the
series is not seasonally adjusted and has a strong year-end pattern (for example November and
December), which the baselines deliberately lack; dummies keep it visible as coefficients.
Rejected: Fourier terms, which are less direct to read, and seasonally adjusted data.

**Lagged change.** Decision: one lag of the dependent variable. Reason: monthly growth is
persistent, and without it the residuals are autocorrelated. Rejected: more lags, which add
coefficients that are hard to interpret for a model meant to be explained.

**Plain OLS with reported coefficients.** Decision: OLS (statsmodels) and a `coefficients()`
helper that returns estimates, standard errors and t-values. Reason: every effect is one number
that can be explained. Rejected: regularised or non-linear fits, which trade away that
transparency. The standard errors assume independent errors and are indicative.

**Residual band.** Decision: the band is `z * sigma * sqrt(h)` on the log scale, with `sigma` the
residual standard deviation, back-transformed with `exp`, at the shared level. Reason: it reuses
the random-walk logic of the baselines and keeps bands comparable. It assumes independent
monthly errors and ignores parameter uncertainty, and the mean is the median path after the
back-transformation. Rejected: simulating the autoregressive error propagation, which adds
machinery for a small gain. The spread path is taken as given, so the band does not include
rate uncertainty.

**Ex-post conditional backtest.** Decision: in the backtest the model receives the realised
spread of the forecast months, as for every driver model. Reason: it measures how well rates
are translated into deposits, not how well rates are guessed (see Backtesting). Because of the
one-month lag only the first h-1 spread values enter an h-month forecast.

**Shock definition.** Decision: a scenario is a parallel shift of the ECB deposit facility rate
by +100 or -100 bp from the last observed month, held constant for the horizon; the baseline
holds all rates at their last observed values. Reason: it is the simplest definition that
isolates the effect of the policy rate. Rejected: gradual or announced rate paths, which need
assumptions on timing that the data cannot support.

**Pass-through estimation.** Decision: each bank rate is regressed on the deposit facility rate
with one lag of itself (`rate_t = c + phi * rate_{t-1} + b * policy_t`), estimated on the full
history; the scenario adds the implied dynamic response to the last observed rate. Reason: bank
rates adjust gradually, and the estimate shows the known asymmetry: time-deposit rates follow the
policy rate far more than overnight rates, so a rate rise widens the spread and a cut narrows it.
Rejected: a static level regression, which gives only the long-run effect and applies it
instantly, and an assumed one-to-one pass-through.

**Scenario horizon.** Decision: 24 months from the last observed month. Reason: long enough
for the slow overnight-rate adjustment and the deposit response to unfold, short enough to keep
the linear relationship plausible. Rejected: the 12-month backtest horizon, which would stop
before the effects mature, and longer horizons.

**Limits.** The model is linear (no floor at zero rates, no saturation), assumes the estimated
relations are stable (no structural break, although the sample contains the negative-rate era
and the 2022 turn), and uses German aggregate data without customer segments, so differences
between household types are invisible. The scenarios show conditional model responses, not
forecasts of what the ECB will do.
