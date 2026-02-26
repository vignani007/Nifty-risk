# Market Risk Internal Model — NIFTY50 Equity Portfolio

**A production-style quantitative market risk framework built in Python, covering portfolio construction, multi-method VaR/ES, GARCH volatility modelling, backtesting, Basel III classification, stress testing, and initial margin proxy.**

---

## Report

A full technical report documenting every methodology, formula, computed result, backtest analysis, and stress scenario is available here:

📘 [Market Risk Internal Model — Full PDF Report](docs/NIFTY50_MarketRisk_Report_Final.pdf)

---

## Overview

This repository implements a modular internal market risk framework for a constrained minimum-variance equity portfolio drawn from the NIFTY50 universe. The framework demonstrates a production-style risk model architecture applicable to market risk analyst and quantitative risk roles.

**Data:** Daily log returns, NIFTY50 constituents, January 2016 – December 2023  
**Universe:** 50 Indian large-cap equities  
**Portfolio:** Long-only constrained minimum-variance (Ledoit–Wolf shrinkage covariance, 5% weight cap)  
**Backtesting window:** 1,510 trading days at 99% confidence

---

## Key Results

| Metric | Value | Model / Basis |
|---|---|---|
| 1-Day VaR (99%) — lowest | 1.44% | EWMA (latest regime) |
| 1-Day VaR (99%) — highest | 3.26% | Cornish–Fisher (fat-tail adjusted) |
| Backtesting exceptions — best | 14 of 1,510 | Historical Simulation (p = 0.77 ✅) |
| Backtesting exceptions — worst | 32 of 1,510 | EWMA (p = 0.0001 ❌) |
| Basel traffic-light (250 days) | Green — all models | 0–4 exceptions, multiplier = 3.0 |
| Worst single-day loss | −5.47% | 23 March 2020 (COVID-19) |
| Worst 10-day loss | −18.35% | Week of 23 March 2020 |
| IM proxy range (MPOR = 10d) | 4.58% – 10.31% | EWMA to Cornish–Fisher |
| GARCH persistence (α + β) | ≈ 0.980 | Half-life ≈ 34 trading days |

---

## Framework Components

### 1. Portfolio Construction

- **Covariance estimator:** Ledoit–Wolf shrinkage — regularises the sample covariance matrix to produce a well-conditioned, positive-definite estimator; critical for stable minimum-variance weights in a 50-stock universe
- **Optimisation:** Constrained minimum-variance (no expected return inputs required)
- **Constraints:** Long-only (w ≥ 0), maximum weight cap of 5% per stock
- **Result:** 25 stocks receive non-zero allocations; 13 stocks are at the 5% cap; the remaining 25 are excluded by the optimiser as variance-increasing

---

### 2. VaR Methodologies (1-Day, 99%)

| Model | VaR | Type |
|---|---|---|
| Historical Simulation | 2.34% | Non-parametric |
| Gaussian Parametric | 1.91% | Parametric |
| Cornish–Fisher | 3.26% | Semi-parametric (skew + kurtosis correction) |
| EWMA (λ = 0.94) | 1.44% | Dynamic parametric |
| Filtered Historical Simulation | 1.78% | Hybrid |
| Monte Carlo — Normal | 1.52% | Simulation |
| Monte Carlo — Student-t (df = 6) | 2.08% | Fat-tail simulation |
| GARCH-N (latest) | 1.72% | Conditional volatility |
| GARCH-t (latest) | 1.98% | Conditional volatility + fat tails |

The 226 bps spread across models quantifies **model risk** — the uncertainty arising from methodology choice alone.

---

### 3. Expected Shortfall (ES)

ES (Conditional VaR / CVaR) is the expected loss given that the loss exceeds VaR. It is a coherent risk measure and the primary regulatory metric under FRTB (Basel IV, 97.5% confidence). Both 97.5% and 99% ES are computed. During the COVID-19 period (Feb–Mar 2020), the 1-day 99% Historical ES reached **5.47%**.

---

### 4. GARCH(1,1) Volatility Modelling

Two GARCH(1,1) variants are fitted using maximum likelihood via the `arch` Python package:

| Parameter | GARCH-Normal | GARCH-Student-t |
|---|---|---|
| α (ARCH — sensitivity to new shocks) | ≈ 0.080 | ≈ 0.075 |
| β (GARCH — persistence of past vol) | ≈ 0.900 | ≈ 0.905 |
| α + β (total persistence) | ≈ 0.980 | ≈ 0.980 |
| Degrees of freedom ν | — | ≈ 11.76 |

α + β ≈ 0.98 implies a volatility half-life of ~34 trading days. Ljung–Box tests confirm no residual ARCH effects in squared standardised residuals.

---

### 5. Backtesting — Kupiec POF Test

1,510 observations, 99% confidence. Expected exceptions: **15.1**.

| Model | Exceptions | p-value | Result |
|---|---|---|---|
| Historical Simulation | 14 | 0.77 | ✅ Pass |
| Gaussian Parametric | 23 | 0.057 | ⚠️ Borderline |
| EWMA (λ = 0.94) | 32 | 0.0001 | ❌ Fail |
| GARCH-Normal | 18 | 0.47 | ✅ Pass |
| GARCH-Student-t | 13 | 0.58 | ✅ Pass |

EWMA fails decisively — its conditional-Normal structure underestimates tail risk, particularly during volatility clustering events. GARCH-t achieves the best coverage.

---

### 6. Basel III Traffic-Light Classification (Last 250 Days)

| Model | Exceptions | Zone | Plus Factor | Multiplier |
|---|---|---|---|---|
| GARCH-N | 0 | 🟢 Green | 0.00 | 3.00 |
| GARCH-t | 0 | 🟢 Green | 0.00 | 3.00 |
| HS (rolling 250) | 2 | 🟢 Green | 0.00 | 3.00 |
| EWMA (λ = 0.94) | 4 | 🟢 Green | 0.00 | 3.00 |

All models are in the Green supervisory zone. During the COVID-19 crisis, multiple models would have breached into Yellow, materially increasing capital requirements.

---

### 7. 10-Day Horizon Risk

Two approaches are compared:

1. **Square-root-of-time scaling** — VaR₁₀ ≈ VaR₁ × √10 (assumes i.i.d. daily returns)
2. **Direct 10-day Historical VaR** — uses overlapping 10-day empirical return windows

| Period | Direct Breaches | SQRT Breaches |
|---|---|---|
| Full sample (1,510 obs) | 23 | 12 |
| COVID window (Feb–Apr 2020) | 13 of 40 obs | 11 of 40 obs |

The SQRT assumption breaks down severely under stress. Volatility clustering during COVID-19 produced consecutive large losses that scaled non-linearly — the worst 10-day loss (−18.35%) was approximately **10× the typical 1-day VaR**.

---

### 8. Stress Testing

Six stress categories are implemented:

| Scenario | 1-Day Loss |
|---|---|
| −3σ parametric shock | 2.06% |
| Correlation stress ×1.3 | 2.30% |
| Correlation stress ×1.8 | 2.65% |
| Volatility shock ×1.5 (3σ event) | 3.09% |
| −5σ parametric shock | 3.43% |
| Volatility shock ×2.0 (3σ event) | 4.12% |
| Worst historical day (23 Mar 2020) | 5.47% |
| Worst historical 10-day window | 18.35% |

COVID period (Feb–Mar 2020) HS VaR was **5.46%** vs **1.92%** in the 2021–22 regime — a 2.8× difference, demonstrating the importance of stressed calibration windows under FRTB.

---

### 9. Initial Margin Proxy (MPOR = 10 Days)

IM proxy = VaR₁d × √10

| Model | VaR 1-Day | IM Proxy (10d) |
|---|---|---|
| EWMA (latest) | 1.45% | 4.58% |
| MC Normal | 1.52% | 4.81% |
| GARCH-N | 1.72% | 5.42% |
| Gaussian | 1.91% | 6.02% |
| GARCH-t | 1.98% | 6.26% |
| MC Student-t | 2.08% | 6.59% |
| Historical Simulation | 2.34% | 7.40% |
| Cornish–Fisher | 3.26% | 10.31% |

The 2.25× spread between EWMA and Cornish–Fisher illustrates significant model risk in margin calibration.

---

## Repository Structure

```
src/          Core model modules (var_models, es_models, garch_model, backtesting, stress, margin...)
tests/        Executable test runners
outputs/      Generated CSVs, tables, and charts
docs/         Technical documentation and PDF report
```

---

## Tech Stack

- **Python 3** — NumPy, SciPy, Pandas
- **arch** — GARCH model fitting
- **scikit-learn** — Ledoit–Wolf covariance estimation
- **matplotlib** — charts and visualisations
- **yfinance** — NIFTY50 price data

---

## Disclaimer

This repository is a prototype internal model built for research and portfolio demonstration purposes. It is not a regulatory-approved model and omits components required under FRTB IMA including P&L attribution testing, NMRF classification, liquidity horizon bucketing, and stressed ES window identification.
