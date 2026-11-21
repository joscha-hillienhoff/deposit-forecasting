# deposit-forecasting

Comparing forecasting models for household overnight deposits on public ECB data, judged on
**accuracy**, **transparency** and **practicality**.

All data is public and comes from the ECB Data Portal. The only exception is a small fictional
deposit book in `synthetic.py`, always labelled synthetic, that demonstrates planned outflows.

## Data

Series are downloaded from the keyless ECB Data Portal API
(`https://data-api.ecb.europa.eu/service/data/<DATASET>/<KEY>?format=csvdata`).

| Column | Series key | Meaning | Unit |
|---|---|---|---|
| `deposits` | [`BSI.M.DE.N.A.L21.A.1.U2.2250.Z01.E`](https://data.ecb.europa.eu/data/datasets/BSI/BSI.M.DE.N.A.L21.A.1.U2.2250.Z01.E) | Overnight deposits of households and NPISH at German MFIs, end of month, not seasonally adjusted | EUR bn (API: EUR m) |
| `overnight_rate` | [`MIR.M.DE.B.L21.A.R.A.2250.EUR.N`](https://data.ecb.europa.eu/data/datasets/MIR/MIR.M.DE.B.L21.A.R.A.2250.EUR.N) | Bank rate on household overnight deposits, new business | % p.a. |
| `time_deposit_rate` | [`MIR.M.DE.B.L22.F.R.A.2250.EUR.N`](https://data.ecb.europa.eu/data/datasets/MIR/MIR.M.DE.B.L22.F.R.A.2250.EUR.N) | Bank rate on household deposits with agreed maturity up to 1 year, new business | % p.a. |
| `deposit_facility_rate` | [`FM.B.U2.EUR.4F.KR.DFR.LEV`](https://data.ecb.europa.eu/data/datasets/FM/FM.B.U2.EUR.4F.KR.DFR.LEV) | ECB deposit facility rate, level (published on change dates) | % p.a. |

`load_panel()` builds a monthly panel (`PeriodIndex`, 2003-01 to the last month with target and
both bank rates) with these columns plus `spread` (time deposit rate minus overnight rate, in
percentage points). The latest deposits value is provisional; its status is available as
`panel.attrs["last_deposit_status"]`. Design decisions are in
[`docs/methodology.md`](docs/methodology.md).

## Reproduce

```bash
uv sync                                  # install dependencies
uv run deposit-forecasting-data          # load the panel from the committed cache, print a summary
uv run deposit-forecasting-data --refresh  # download again and overwrite data/raw/
uv run deposit-forecasting-backtest      # rolling-origin backtest of the baselines (--out DIR writes CSVs)
uv run pytest                            # offline tests
```

## Status

- [x] Data
- [x] Baselines
- [x] Rolling-origin backtest
- [x] Exponential smoothing and SARIMA
- [ ] Rate regression and scenarios
- [ ] Gradient boosting
- [ ] Results and recommendation

## Results

To be added once models and the backtest exist.

## Data source

Data: European Central Bank, ECB Data Portal (<https://data.ecb.europa.eu>). Reuse follows the
ECB's terms for statistical data.
