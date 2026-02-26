\# Market Risk Internal Model Prototype  

\### NIFTY50 Equity Portfolio (Constrained Minimum-Variance)



---



\## Overview



This repository implements a modular market risk framework for a liquid Indian equity portfolio derived from the NIFTY50 universe.



The objective is to demonstrate a production-style internal risk model architecture including:



\- Portfolio construction (Ledoit–Wolf shrinkage + constrained minimum-variance)

\- Multiple Value-at-Risk (VaR) methodologies

\- Expected Shortfall (ES)

\- Monte Carlo simulation (Normal and Student-t)

\- Volatility modeling (GARCH(1,1))

\- Backtesting (Kupiec test)

\- Basel traffic-light classification

\- Initial Margin proxy (MPOR scaling)

\- Stress testing (historical replay and scenario shocks)

\- 10-day horizon risk analysis



All models are implemented in a modular Python package under `src/`.



---



\## Portfolio Construction



Universe: NIFTY50 equities  

Data window: 2016–2023  

Estimation window: 504 trading days  



Method:



\- Sample returns: daily log returns  

\- Covariance: Ledoit–Wolf shrinkage  

\- Optimization: Long-only minimum-variance  

\- Maximum weight cap: 5%  



This avoids concentration risk and stabilizes covariance estimation.



---



\## VaR Methodologies (1-Day, 99%)



Implemented models:



\- Historical Simulation (HS)

\- Gaussian Parametric

\- Cornish–Fisher expansion

\- EWMA volatility model

\- Filtered Historical Simulation (FHS)

\- Monte Carlo (Normal)

\- Monte Carlo (Student-t)

\- GARCH(1,1) (Normal \& Student-t)



Both VaR and ES are computed where appropriate.



---



\## Backtesting



\- Horizon: 1-day VaR

\- Confidence: 99%

\- Observations: 1510

\- Kupiec Proportion-of-Failures test

\- Exception clustering summary

\- Basel traffic-light (last 250 days)



Results show:



\- EWMA underestimates tail risk

\- Gaussian borderline optimistic

\- GARCH and Historical models stable

\- All models currently in green supervisory zone



---



\## 10-Day Horizon Risk



Two approaches implemented:



1\. Direct 10-day historical VaR

2\. Square-root-of-time scaled 1-day VaR



Comparison during COVID shock highlights breakdown of iid scaling assumptions.



---



\## Stress Testing



Stress framework includes:



\- Historical worst-day replay

\- Historical worst 10-day replay

\- COVID shock window analysis

\- Regime stress (2021–2022)

\- Parametric sigma shocks (-3σ, -5σ)

\- Volatility multipliers (1.5×, 2.0×)

\- Correlation stress (off-diagonal scaling)



---



\## Initial Margin Proxy



IM proxy computed via:



IM ≈ VaR₁d × √MPOR



Used to compare model sensitivity under 10-day margin period of risk.



---



\## Repository Structure





src/ Core model modules

tests/ Executable runners

outputs/ Generated results \& tables

docs/ Technical documentation





---



\## Disclaimer



This repository represents a prototype internal model for research and demonstration purposes.  

It is not a regulatory-approved model and omits liquidity, modellability, and capital aggregation components required under FRTB IMA.



---





