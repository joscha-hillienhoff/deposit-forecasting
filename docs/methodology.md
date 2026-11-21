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
huge in a quiet month), when over- and under-forecasts are treated unequally (an
under-forecast can cost at most 100 % while an over-forecast is unbounded, so MAPE favours
forecasts that are too low), and when comparing series of different scale or level. Deposits are a large,
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

## Exponential smoothing and SARIMA

Both models are in `classical.py`, use the shared interface and are scored by the same
backtest. Neither uses `exog`.

**Log transform.** Decision: model log deposits and back-transform forecasts with `exp`.
Reason: deposits grew several-fold over the sample, so seasonal swings and trend are
proportional to the level, and a log scale makes them roughly constant; it also keeps the
band positive. Rejected: modelling the level, which would need a multiplicative form or let
the seasonal amplitude stay fixed while the level multiplies.

**Damped additive trend (ETS).** Decision: ETS(A,Ad,A), additive error, damped additive trend,
additive seasonality, fixed in advance and never searched per origin; parameters and initial
states are re-estimated at each origin. Reason: on the log scale an additive trend is a
constant growth rate, and damping stops a growth rate estimated in one regime from being
extrapolated for a year; the specification is the same for every origin, so no choice can
leak information from the evaluation period. Rejected: an undamped trend, which extrapolates
the latest growth rate at full strength, and automatic selection among ETS variants per
origin, which makes the model change over time and is hard to audit.

**Additive seasonality on the log scale.** Decision: monthly seasonality (period 12) is
additive in logs, i.e. multiplicative in levels. Reason: it follows from the log transform
and lets the seasonal effect scale with the level. Rejected: a multiplicative seasonal term
on the log scale, which has no interpretation there; and seasonal dummies, which would be a
separate model rather than part of the smoothing.

**Orders chosen before the first origin (SARIMA).** Decision: orders were selected once by
AIC over a small grid (p, q in 0..2, seasonal P, Q in 0..1, with d = D = 1 fixed in advance)
using only the data up to 2010-12, the first backtest origin, and are then held fixed for every
origin. Result: SARIMA(1,1,2)(0,1,1)12, stored as constants and reproducible with
`uv run python -m deposit_forecasting.classical`. Reason: choosing orders on the evaluation
period would be look-ahead; AIC is not comparable across differencing orders, hence fixed
d and D, and a trending seasonal stock calls for one regular and one seasonal difference.
Rejected: re-selecting orders at every origin, which costs a grid search per origin and
lets the model structure drift, and selecting on the full sample, which is look-ahead. The
parameters are still re-estimated at each origin. The model is fitted on 100 times log
deposits, which only rescales the series: with the raw log the tiny innovation variance made
the optimiser fail to converge at some origins.

**Model-based intervals at the shared level.** Decision: intervals come from the fitted
model (the ETS state-space forecast distribution, SARIMA `get_forecast`) at the shared level
and are back-transformed with `exp`. Reason: the bands then widen with the horizon as the
model implies and need no extra calibration step. Both assume Gaussian innovations with
constant variance and ignore parameter uncertainty, so observed coverage can differ from
the nominal level; the backtest reports it. Rejected: bootstrap or empirical-error bands,
which are costlier and add a second source of choices.

**Back-transformed mean is a median.** Decision: `mean` is `exp` of the log-scale forecast
and is therefore the conditional median, not the conditional mean, which is larger by about
half the log variance. Reason: quantiles survive a monotone transform, so the band keeps
its nominal level and stays consistent with the point forecast; at the horizons and
variances here the difference is small. Rejected: a bias-corrected mean, which relies more
strongly on the normality assumption and would sit off-centre in the band.

**Failures raise.** Decision: a fit that does not converge or yields non-finite values raises
an error; convergence warnings themselves are silenced so the CLI output stays clean. Reason:
a silent fallback would put another model's forecasts under this model's name. Rejected:
falling back to a simpler model or the previous origin's parameters.

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

## Questions and answers

**Why rolling backtesting instead of a simple train/test split?** A single split yields one
error path whose value depends on where the split falls. Here that matters: a split dated after
2023 never tests the rate turn, while a split dated early in 2022 makes the turn most of the
test set, and the ranking of the
models differs between the two (over all months SARIMA leads, in the 2022-2023 segment the
naive forecast and ETS do). A rolling origin gives 189 one-month and 178 twelve-month forecasts
under the information set of each date, so errors, horizon profiles and band coverage can be
averaged, and the turn can be examined as a separate segment. The price is that the forecasts
of adjacent origins overlap and are strongly dependent, so the effective sample is smaller than
the counts suggest, and that every model is refitted at every origin.

**When is MAPE misleading?** MAPE divides by the actual, so it explodes when actuals are near
zero or change sign, penalises over-forecasts more than under-forecasts (an under-forecast
costs at most 100 %, an over-forecast is unbounded, so it favours forecasts that are too low), is
unusable for flows that can be zero, and cannot compare series of different scale. For a
strictly positive stock of EUR 0.7 to 1.9 trillion none of these bites hard, which is why it
stays a secondary metric. There is a milder trap in this project: the level grows about
2.7-fold from the first backtest origin to the end of the sample, so the same EUR error is a
smaller percentage late in the sample and MAPE flatters recent years. MAE in EUR bn stays the headline
because it states the error in the unit that a liquidity planner cares about. For a single
client or a net-flow series, where the actual can be close to zero, MAPE would be
meaningless.

**Why does a simple model often beat a complex one?** With about 280 monthly observations,
heavily overlapping rows and few regimes, a flexible model has to estimate many parameters
from little independent information, so its variance rivals the signal. In this backtest the
naive forecast is the most accurate model at 12 months in the 2022-2023 segment (MAE 63 against
97 for LightGBM and 129 for SARIMA), and ETS with a damped trend, a deliberately restrained
model, beats the more flexible SARIMA there at every horizon. The complex model does win where
its flexibility captures structure the sample can pin down: LightGBM has the lowest full-period
error at 11 and 12 months. A simple model also degrades gracefully, is cheap to monitor, and
sets the bar any more complex model has to clear.

**How does the 2022-2023 rate turn affect the models?** Before 2022 deposits grew steadily
while rates were very low. When the ECB raised rates, growth stopped and the stock fell, so
trend-following models kept extrapolating growth. From a 2021-12 origin the drift and the rate
regression forecast continued growth, to 1946 and 1990 EUR bn at 12 months, while deposits grew
only about 2.6 % to 1868 EUR bn in 2022, peaked in late 2022 and fell in 2023. Within the
segment the drift and the rate regression have a 12-month MAE of 131 and 159 EUR bn against 63
for the naive forecast, and their 80 % bands contain none of the realised values at 12 months.
The rate regression was estimated on a sample dominated by low rates, so the response to a
large spread change is an extrapolation of a linear, stable relationship. ETS limits the
damage because its damped trend flattens, and the naive forecast is hard to beat at 12 months.
The practical lessons are that bands fitted in calm periods are too narrow in a regime change
(at 12 months the observed coverage of the drift, rate regression, LightGBM, naive and SARIMA bands
is 0 to 46 % against 80 %; only the ETS bands hold, at 79 %) and that a model's explanatory power in normal times does not
guarantee it at a structural break.

**What would a bank additionally consider?** The ECB aggregate hides what a bank sees in its
own book. First, individual large deals and planned outflows: a few large clients can move the
total, and a known departure is information that no statistical model can extract from the
history. The synthetic demonstration (a fictional book) shows the mechanism: subtracting the
announced outflows reduces the drift forecast's MAE from 1.09 to 0.05 EUR bn on that
synthetic data, which proves nothing about real accuracy but shows the logic. Second, customer
segments: retail and corporate customers, and rate-sensitive and sticky customers, react very
differently to the same rate path, so a bank models segments separately and an aggregate
elasticity such as the one estimated here is an average over very different behaviour. Third,
the stable core versus the volatile part of deposits: the share of balances that stays through
rate and market stress (the core, which behavioural models assign a long maturity) is
distinguished from the share that leaves quickly (the volatile part, which is treated as
short-dated). Behavioural maturity assumptions feed liquidity risk management (the
liquidity coverage ratio and stress outflow assumptions, funding and buffer planning) and
interest-rate risk management (the repricing profile and duration of non-maturing deposits,
and thereby the sensitivity of net interest income and economic value to rate changes).
Forecast scenarios such as the +/-100 bp paths here are an input to these
assessments, not a substitute for them.
