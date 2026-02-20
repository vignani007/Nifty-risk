\# Technical Note  

\## Market Risk Internal Model – NIFTY50 Portfolio



---



\## 1. Model Objective



The objective is to construct and validate a modular internal market risk framework for an optimized equity portfolio derived from the NIFTY50 universe.



The framework evaluates tail risk using multiple VaR methodologies, performs statistical validation, and compares short-horizon and multi-day risk behavior under stress.



---



\## 2. Portfolio Construction



\- Data: 2016–2023 daily log returns  

\- Estimation window: 504 trading days  

\- Covariance estimator: Ledoit–Wolf shrinkage  

\- Optimization: Long-only minimum-variance  

\- Weight cap: 5%



Shrinkage mitigates sample instability and improves conditioning of the covariance matrix.



---



\## 3. 1-Day VaR Comparison (99%)



| Model | VaR |

|-------|------|

| Historical Simulation | 2.64% |

| Gaussian Parametric | 1.97% |

| Cornish–Fisher | 3.54% |

| EWMA (latest) | 1.44% |

| FHS | 1.78% |

| MC Normal | 1.52% |

| MC Student-t (df=6) | 2.08% |

| GARCH-N (latest) | 1.72% |

| GARCH-t (latest) | 1.98% |



Observations:



\- Historical VaR captures heavy tails.

\- Gaussian underestimates extreme risk.

\- Student-t Monte Carlo increases tail thickness.

\- Cornish–Fisher sensitive to higher moments.

\- GARCH captures volatility clustering.



---



\## 4. Backtesting (1510 Observations)



Expected exceptions at 99% ≈ 15.1



| Model | Exceptions | p-value |

|--------|------------|---------|

| HS | 14 | 0.77 |

| Gaussian | 23 | 0.057 |

| EWMA | 32 | 0.0001 |

| GARCH-N | 18 | 0.47 |

| GARCH-t | 13 | 0.58 |



EWMA underestimates tail risk.  

GARCH and Historical methods demonstrate statistical adequacy.



---



\## 5. Basel Traffic-Light (250 days)



All tested models fall within the green supervisory zone (multiplier = 3.0).



---



\## 6. GARCH Diagnostics



Estimated parameters (Normal):



α + β ≈ 0.98



Indicates strong volatility persistence.



Standardized residuals:



\- Mean ≈ 0  

\- Std ≈ 1  

\- Excess kurtosis ≈ 1.12  



Ljung–Box tests indicate no residual autocorrelation or ARCH effects.



Student-t variant estimates ν ≈ 11.76, indicating moderate fat tails.



---



\## 7. 10-Day Horizon Risk



Direct 10-day historical VaR breaches:



\- Full sample: 1.84%

\- COVID window: 32.5%



Square-root scaling breaches:



\- Full sample: 0.95%

\- COVID window: 27.5%



Scaling assumption deteriorates under stress due to volatility clustering and correlation breakdown.



---



\## 8. Stress Testing



Worst historical 10-day loss: −18.35% (COVID shock)



Scenario stresses show:



\- 2× volatility shock exceeds Gaussian VaR

\- Correlation stress materially increases loss estimates



Stress results confirm model sensitivity to tail events.



---



\## 9. Initial Margin Proxy



IM\_proxy = VaR₁d × √10



Model ranking under MPOR=10d:



\- EWMA lowest

\- Cornish–Fisher highest

\- Historical and Student-t mid-to-high range



Demonstrates model sensitivity to distributional assumptions.



---



\## 10. Limitations



\- No liquidity adjustment

\- No risk factor mapping (full IMA)

\- No P\&L attribution testing

\- No NMRF classification

\- No stressed VaR window



Framework focuses on statistical risk modeling layer.



---



