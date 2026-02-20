import os
import numpy as np
import pandas as pd

from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.var_models import portfolio_returns

from src.stress import (
    hs_var_es,
    worst_days,
    worst_horizon,
    period_slice,
    portfolio_sigma_from_cov,
    shock_loss_sigma,
    corr_from_cov,
    cov_from_corr,
    stress_correlations,
    stress_vols,
)

# --------------------------
# Portfolio build (same as main project)
# --------------------------
prices = download_price_data(NIFTY50_TICKERS, "2016-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")

rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w_series = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

rp = portfolio_returns(rets, w_series)

alpha = 0.99

# --------------------------
# Baseline VaR/ES context (historical)
# --------------------------
var99, es99 = hs_var_es(rp, alpha=alpha)

print("\n=== Baseline HS VaR/ES (1D, 99%) ===")
print("VaR:", var99, "ES:", es99)

# --------------------------
# Historical replay stresses
# --------------------------
wd = worst_days(rp, k=10)
w10 = worst_horizon(rp, horizon_days=10, k=5)

print("\n=== Worst daily losses (historical replay) ===")
print(wd)

print("\n=== Worst 10-day cumulative losses (historical replay) ===")
print(w10)

# Period stress replay (named windows)
periods = {
    "COVID shock": ("2020-02-01", "2020-03-31"),
    "2021-2022 regime": ("2021-01-01", "2022-12-31"),
}

period_rows = []
for name, (s, e) in periods.items():
    r = period_slice(rp, s, e)
    if len(r) < 30:
        continue

    var_p, es_p = hs_var_es(r, alpha=alpha)
    worst_1d = float(r.min())
    worst_10d = float(r.rolling(10).sum().min())

    period_rows.append({
        "period": name,
        "start": s,
        "end": e,
        "obs": int(len(r)),
        "HS_VaR_1D_99": var_p,
        "HS_ES_1D_99": es_p,
        "worst_1D_return": worst_1d,
        "worst_1D_loss": -worst_1d,
        "worst_10D_return": worst_10d,
        "worst_10D_loss": -worst_10d,
    })

period_df = pd.DataFrame(period_rows)
print("\n=== Period stress replay summary ===")
print(period_df)

# --------------------------
# Parametric scenario stresses (sigma / vol / corr)
# --------------------------
w = w_series.reindex(rets_est.columns).fillna(0.0).values
port_sigma = portfolio_sigma_from_cov(Sigma, w)

scen_rows = []

# sigma shocks
for n_sig in [3, 5]:
    scen_rows.append({
        "scenario": f"Shock: -{n_sig}σ day",
        "loss_1D": shock_loss_sigma(port_sigma, n_sigma=float(n_sig)),
        "reference": "Parametric sigma shock"
    })

# vol shocks: keep Corr, increase vols
Corr, vol = corr_from_cov(Sigma)
for mult in [1.5, 2.0]:
    vol_s = stress_vols(vol, vol_mult=mult)
    Sigma_s = cov_from_corr(vol_s, Corr)
    sig_s = portfolio_sigma_from_cov(Sigma_s, w)
    scen_rows.append({
        "scenario": f"Vol shock: vols x{mult}",
        "loss_1D": shock_loss_sigma(sig_s, n_sigma=3.0),
        "reference": "3σ under shocked vols"
    })

# correlation stress: increase off-diagonals
for fac in [1.3, 1.8]:
    Corr_s = stress_correlations(Corr, factor=fac, cap=0.99)
    Sigma_s = cov_from_corr(vol, Corr_s)
    sig_s = portfolio_sigma_from_cov(Sigma_s, w)
    scen_rows.append({
        "scenario": f"Corr stress: off-diag x{fac} (cap 0.99)",
        "loss_1D": shock_loss_sigma(sig_s, n_sigma=3.0),
        "reference": "3σ under corr stress"
    })

scen_df = pd.DataFrame(scen_rows).sort_values("loss_1D")
print("\n=== Scenario stress results (loss_1D) ===")
print(scen_df)

# --------------------------
# Save outputs
# --------------------------
os.makedirs("outputs", exist_ok=True)
wd.to_csv("outputs/stress_worst_days.csv")
w10.to_csv("outputs/stress_worst_10d.csv")
period_df.to_csv("outputs/stress_period_replay.csv", index=False)
scen_df.to_csv("outputs/stress_scenarios.csv", index=False)

print("\nSaved outputs:")
print(" - outputs/stress_worst_days.csv")
print(" - outputs/stress_worst_10d.csv")
print(" - outputs/stress_period_replay.csv")
print(" - outputs/stress_scenarios.csv")