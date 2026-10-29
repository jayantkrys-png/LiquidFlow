# Methodology

This note describes how LiquidFlow builds its forecasts, how they were tested and what the results do and do not show. Every setting below is fixed in [`protocol.json`](../protocol.json). The analysis reads it at start-up and records its SHA-256 in `artifacts/evidence.json` and `artifacts/fit_manifest.json`.

## 1. Problem and target

The brief asks for a forecast of liquid-fund outflows and a cash buffer that beats a simple rule: hold last year's outflow for the same month.

- **Target:** AMFI category-level monthly net flow (sales minus redemptions) in ₹ crore, for the Liquid and Overnight categories. Negative is a net outflow.
- **Horizons:** 1 to 4 months ahead of the latest AMFI month. AMFI publishes each month's figures early in the following month, so with September 2026 known, horizon 1 is October 2026 (a nowcast) and horizon 4 is January 2027.
- **Scored horizons:** 1 to 3 for model selection and test scores. Horizon 4 is reported but not used to rank models. The cash backtest uses horizon 1 only, so overlapping forecasts of the same month are not charged twice.

## 2. Data

| Source | Coverage used | Notes |
| --- | --- | --- |
| AMFI monthly reports | Jul 2019 to Sep 2026, 87 months per category | 102 reports downloaded. The 15 oldest use a layout that combines liquid and money-market funds, so they are excluded rather than spliced. For two months (Dec 2021, Jan 2022) AMFI's revised report is used. |
| RBI Handbook, Table 174 | Apr 2020 to Jun 2026 | Weighted average call-money rate. |
| RBI Handbook, Table 182 | Apr 2018 to Jun 2026 | Nifty 50 monthly average, used as a monthly percentage change. |
| RBI Bulletin, liquidity operations | Jul 2019 to Jul 2026 | Net RBI liquidity operations, used as a direction signal for banking liquidity. Months with fewer than 20 dated observations are not treated as complete. |
| Income Tax Department | Advance-tax schedule | 15 June, 15 September, 15 December and 15 March. |

Each AMFI row is checked on parsing: funds mobilised minus redemptions must equal the reported net flow to within ₹1 crore, and AUM must be positive. All 174 category-months match exactly. Every raw file is stored unmodified with its SHA-256 in `data/processed/amfi_manifest.json` and `data/processed/macro_manifest.json`, and `make audit` checks them all.

## 3. Features

All features are computed from information available at the forecast origin. Flow features are divided by the assets under management (AUM) known at the origin, so the two categories and different years sit on the same scale.

| Group | Features |
| --- | --- |
| Calendar (target month) | First and second annual harmonics (sine and cosine); flags for advance-tax quarter-end months (March, June, September, December), March and April; each also interacted with an Overnight indicator |
| Flow history | Net flow as a share of AUM at lags 0 to 5; 3-month rolling mean; 6-month rolling standard deviation; same month last year as a share of that month's opening AUM; 3-month AUM growth; the other category's latest flow rate |
| Rates | Call-money rate, its 3-month change, months since last observation |
| Equity | Nifty 50 monthly change, 3-month average change, months since last observation |
| Liquidity | Net RBI operations (₹ lakh crore), 3-month change, months since last observation |
| Other | Category indicator, horizon number |

RBI and market inputs are lagged two months from the forecast origin to reflect publication delays. Missing values are never filled with zero. The latest earlier observation is carried forward and its age is kept as a feature. The RBI Handbook supplies latest-vintage values, so the history is not an exact replay of what was known at each date.

## 4. Candidate models

| Name | Description |
| --- | --- |
| `seasonal_naive` | The same month last year, in ₹ crore. |
| `seasonal_scaled` | The same month last year as a share of that month's opening AUM, times current AUM. |
| `sarimax` | AR(1) with a constant and the calendar features as exogenous inputs, fitted separately per category on monthly flow divided by previous-month AUM. Forecasts are multiplied by origin AUM. |
| `ridge_core` | Ridge regression (alpha 10) on calendar and flow-history features, pooled across categories and horizons. Median imputation with missing-value indicators, then standardisation. |
| `ridge_full` | As `ridge_core`, plus the rates, equity and liquidity groups. |
| `catboost_core` | CatBoost, 260 iterations, depth 3, learning rate 0.035, L2 8, MAE loss, on the `ridge_core` features. |
| `catboost_full` | The same CatBoost on the `ridge_full` features. |

Regressors predict flow as a share of origin AUM (direct multi-horizon: one row per origin, category and horizon). Every model is refitted at every monthly origin from June 2021 to September 2026 on an expanding window, using only rows whose target month is already known at that origin. The loop checks this and stops if any future label enters training. SARIMAX converged in all 128 fits with no warnings.

Four extra Ridge fits each drop one input group (calendar, rates, equity, liquidity). They are used only to measure how much each group contributes.

## 5. Ranges and point forecasts

Ranges come from each model's own recent errors, not from distributional assumptions.

1. For each model, category and horizon, take the last 24 forecast errors whose target month had been published by the forecast origin.
2. Divide each error by the AUM at its origin.
3. Take the empirical 2.5th, 5th, 10th, 50th, 90th, 95th and 97.5th percentiles, scale them by the current origin AUM and add them to the raw forecast.

The median of the recent errors is used as a bias correction, so the published central forecast is the raw forecast plus that median. This applies to every candidate, including seasonal naive. An uncorrected seasonal naive is also reported in `evidence.json` for reference.

The 80% range runs from the 10th to the 90th percentile (`q100` to `q900` in the CSVs), the 95% range from the 2.5th to the 97.5th (`q025` to `q975`).

## 6. Evaluation design

| Period | Target months | Use |
| --- | --- | --- |
| Warm-up | up to Sep 2023 | Builds the error history for the ranges |
| Selection | Oct 2023 to Sep 2024 | Choose one model, by MAE divided by origin AUM, averaged over both categories and horizons 1 to 3 |
| Test | Oct 2024 to Sep 2026 | Score the chosen model once, against all others |

Selection uses only selection-year scores, and test scores play no part in it. Random train/test splits, same-month market inputs, future AUM values and errors from months not yet published were all excluded by construction.

Metrics: MAE in ₹ crore, MAE as a share of origin AUM, RMSE, MAPE (unstable when flows are near zero, so not used for selection), MASE against the in-sample seasonal difference, average pinball loss over the seven quantiles, and 80% and 95% coverage and width.

The 144 test forecasts (2 categories, 3 horizons, 24 months) are not independent. A circular block bootstrap (3,000 replicates, blocks of three consecutive target months, all categories and horizons for a month kept together) gives an approximate interval for the difference between SARIMAX and seasonal naive.

## 7. Cash buffer and cost

**Buffer:** the 5th percentile of the horizon-1 net flow forecast, turned into an outflow and capped at origin AUM:

```
buffer = min(max(0, -q050), origin_aum)
```

**Baseline (the brief's rule):** last year's outflow for the same month, `min(max(0, -flow_same_month_last_year), origin_aum)`.

**Cost**, with `g` = 0.5% a year yield given up on cash, `d` = days in the month, `outflow = max(0, -actual)`:

```
idle      = max(buffer - outflow, 0)
shortfall = max(outflow - buffer, 0)
cost      = g * d / 365 * idle + penalty * shortfall
```

The brief's "3x penalty" can be read two ways, and both are reported:

- **Literal:** penalty = 3, meaning ₹3 per rupee short. This is the headline reading.
- **Yield-gap:** penalty = 3 × g × d / 365, three times the monthly cost of holding cash.

Under the literal reading a shortfall costs about 7,300 times as much as idle cash, so the cost-minimising buffer sits at the 99.99th percentile. Under the yield-gap reading it sits at the 75th. The 95% buffer is a fixed rule and was not tuned to either. A sensitivity grid over g in {0.25%, 0.5%, 1%} and penalty in {1, 3, 5} is in `artifacts/cost_sensitivity.csv`.

## 8. Results

### Selection year (Oct 2023 to Sep 2024)

MAE as a share of origin AUM, horizons 1 to 3, 72 forecasts each:

| Model | Error |
| --- | ---: |
| **SARIMAX** | **12.09%** |
| CatBoost full | 13.43% |
| Ridge full | 13.57% |
| Ridge core | 13.90% |
| Seasonal scaled | 14.06% |
| CatBoost core | 14.49% |
| Seasonal naive | 15.59% |

SARIMAX was selected.

### Test years (Oct 2024 to Sep 2026), horizons 1 to 3, 144 forecasts

| Model | MAE (% of AUM) | MAE (₹ cr) | MAPE | 80% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seasonal naive | 12.42 | 24,875 | 285% | 84.0% | 91.7% |
| Seasonal scaled | 12.35 | 24,919 | 299% | 81.3% | 90.3% |
| **SARIMAX (selected)** | 13.32 | 27,310 | 168% | 69.4% | 85.4% |
| Ridge core | 13.29 | 27,331 | 135% | 72.2% | 90.3% |
| Ridge full | 13.24 | 27,742 | 159% | 75.0% | 90.3% |
| CatBoost core | 13.84 | 29,573 | 147% | 74.3% | 93.1% |
| CatBoost full | 13.29 | 28,166 | 151% | 76.4% | 92.4% |

SARIMAX did not beat seasonal naive on rupee error. The bootstrap difference is +0.89 percentage points of AUM (worse), with an approximate 95% interval of −1.29 to +3.19, so the two cannot be separated on this sample. SARIMAX cut MAPE by 41%. MAPE gives most weight to months with small net flows, so it and rupee error can rank models differently.

### Coverage of the selected model by category

| Category | Horizon | 80% range | 95% range |
| --- | ---: | ---: | ---: |
| Liquid | 1 | 66.7% | 87.5% |
| Liquid | 2 | 70.8% | 95.8% |
| Liquid | 3 | 75.0% | 95.8% |
| Overnight | 1 | 70.8% | 75.0% |
| Overnight | 2 | 66.7% | 79.2% |
| Overnight | 3 | 66.7% | 79.2% |

The liquid-fund 95% range is close to target. The overnight range is too narrow and should be read as a lower bound.

### Input groups (Ridge, selection year)

Removing a group and refitting changes the error (percentage points of AUM; positive means the group helped):

| Group removed | Change |
| --- | ---: |
| Calendar | +1.14 |
| Rates | +0.45 |
| Equity | −0.05 |
| Liquidity | −0.08 |

### Cash backtest, horizon 1, 48 category-months

| Rule | Months short | Average buffer (% of AUM) | Literal cost (₹ cr) | Yield-gap cost (₹ cr) |
| --- | ---: | ---: | ---: | ---: |
| Last year's same-month outflow | 18 | 5.9% | 895,707 | 461 |
| LiquidFlow buffer (SARIMAX 5th percentile) | 4 | 18.6% | 111,863 | 487 |
| Seasonal-naive 95% band | 2 | 22.6% | 51,375 | 741 |
| Hold all assets in cash (reference) | 0 | 100% | 5,955 | 5,955 |

Under the literal reading, sizing cash to the bottom of a forecast range cuts the cost by 87.5% against last year's rule. The seasonal-naive band does better because its ranges were wider over the test years. Under the yield-gap reading, shortfalls are cheap and the smallest buffer wins, so last year's rule is cheapest. Holding all assets in cash is cheapest of all under the literal reading, which shows how heavily that reading punishes any shortfall. The ranking depends on how the penalty is read, which is why both are shown.

## 9. Limits

- 87 months per category, so tail percentiles rest on few observations.
- Older AMFI layouts that combine liquid and money-market funds are excluded, not spliced.
- AMFI revisions and latest-vintage RBI tables mean the history is not an exact original-release replay.
- Macro inputs use a fixed two-month publication lag.
- Advance-tax months are also quarter ends, and monthly data cannot separate the two effects.
- Coverage is measured on 24 test months and can drift with changes in regime.
- Monthly net flow hides gross and intra-month redemptions. A single fund also needs its own daily redemption history, investor concentration and maturity profile.
- Category forecasts are not a regulatory liquidity test for any scheme.

## 10. References

- AMFI, Monthly data archive. https://www.amfiindia.com/research-information/amfi-monthly
- Reserve Bank of India, Handbook of Statistics on Indian Economy, Tables 174 and 182.
- Reserve Bank of India, RBI Bulletin, current statistics on liquidity operations.
- Income Tax Department, advance tax due dates under section 211 of the Income-tax Act, 1961.
- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., OTexts, 2021 (seasonal naive benchmark, empirical prediction intervals).
- Gneiting, T. and Raftery, A. E. "Strictly proper scoring rules, prediction, and estimation." *Journal of the American Statistical Association* 102 (2007): 359 to 378.
- Seabold, S. and Perktold, J. "statsmodels: Econometric and statistical modeling with Python." *Proceedings of the 9th Python in Science Conference*, 2010.
- Prokhorenkova, L. et al. "CatBoost: unbiased boosting with categorical features." *NeurIPS*, 2018.
- Pedregosa, F. et al. "Scikit-learn: Machine learning in Python." *JMLR* 12 (2011): 2825 to 2830.
