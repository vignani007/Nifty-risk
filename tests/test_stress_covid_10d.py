import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.var_models import portfolio_returns
from src.backtesting import rolling_historical_var  # 1D HS VaR series (windowed)
from src.horizon_var import rolling_hs_var_horizon, horizon_log_return, scale_var_sqrt_time


def make_level_from_log_returns(log_ret: pd.Series, start: float = 100.0) -> pd.Series:
    lr = log_ret.dropna()
    lvl = start * np.exp(lr.cumsum())
    lvl.name = "ClosingPrice"
    return lvl


def build_breach_table(
    log_ret_1d: pd.Series,
    var_10d: pd.Series,
    horizon_days: int = 10,
    start_level: float = 100.0,
) -> pd.DataFrame:
    lvl = make_level_from_log_returns(log_ret_1d, start=start_level)
    ret_h = horizon_log_return(log_ret_1d, horizon_days=horizon_days)

    df = pd.concat(
        [
            lvl,
            log_ret_1d.rename("LogReturn"),
            var_10d.rename("VaR_10D"),
            ret_h.rename("Ret_10D"),
        ],
        axis=1,
    ).dropna()

    df["Breach"] = (-df["Ret_10D"] > df["VaR_10D"])
    return df


def breach_summary(df: pd.DataFrame) -> dict:
    n = int(df.shape[0])
    x = int(df["Breach"].sum()) if n > 0 else 0
    pct = 100.0 * x / n if n > 0 else float("nan")
    return {"obs": n, "breaches": x, "breach_pct": pct}


def plot_breaches(df: pd.DataFrame, title: str, outpath: str):
    plt.figure()
    plt.plot(df.index, df["Ret_10D"], label="Ret_10D")
    plt.plot(df.index, -df["VaR_10D"], label="-VaR_10D threshold")
    idx = df.index[df["Breach"]]
    plt.scatter(idx, df.loc[idx, "Ret_10D"], marker="x", label="Breaches")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)


# --------------------------
# Settings
# --------------------------
alpha = 0.99
horizon_days = 10
window = 250
weight_cap = 0.05

start_date = "2016-01-01"
end_date = "2023-12-31"

covid_start, covid_end = "2020-02-01", "2020-03-31"

# --------------------------
# Build portfolio returns (daily log returns)
# --------------------------
prices = download_price_data(NIFTY50_TICKERS, start_date, end_date)
prices = prices.dropna(axis=1, how="all")

rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=weight_cap)

rp = portfolio_returns(rets, w)  # daily portfolio log return

# --------------------------
# Two 10D VaR definitions
# --------------------------
# 1) 1D HS VaR forecast series (windowed) -> scaled to 10D
var_1d_hs = rolling_historical_var(rp, alpha=alpha, window=window).shift(1)
var_10d_scaled = scale_var_sqrt_time(var_1d_hs, horizon_days=horizon_days)

# 2) Direct 10D HS VaR forecast series on 10D realized returns
var_10d_direct = rolling_hs_var_horizon(
    rp,
    alpha=alpha,
    horizon_days=horizon_days,
    window=window,
)

# --------------------------
# Build breach packs
# --------------------------
df_direct = build_breach_table(rp, var_10d_direct, horizon_days=horizon_days)
df_scaled = build_breach_table(rp, var_10d_scaled, horizon_days=horizon_days)

print("\n=== 10D DIRECT HS VaR breach summary (full sample) ===")
print(breach_summary(df_direct))

print("\n=== 10D SCALED (sqrt-time) VaR breach summary (full sample) ===")
print(breach_summary(df_scaled))

# COVID window subsets
df_covid_direct = df_direct.loc[covid_start:covid_end].dropna()
df_covid_scaled = df_scaled.loc[covid_start:covid_end].dropna()

print("\n=== COVID window: DIRECT 10D HS VaR breach summary ===")
print(breach_summary(df_covid_direct))

print("\n=== COVID window: SCALED 10D VaR breach summary ===")
print(breach_summary(df_covid_scaled))

# --------------------------
# Save outputs
# --------------------------
os.makedirs("outputs", exist_ok=True)

# Full tables
df_direct.to_csv("outputs/breach_table_portfolio_direct10d_full.csv")
df_scaled.to_csv("outputs/breach_table_portfolio_scaled10d_full.csv")

# Breaches only
df_direct[df_direct["Breach"]].to_csv("outputs/breaches_portfolio_direct10d.csv")
df_scaled[df_scaled["Breach"]].to_csv("outputs/breaches_portfolio_scaled10d.csv")

# COVID tables
df_covid_direct.to_csv("outputs/breach_table_portfolio_direct10d_covid.csv")
df_covid_scaled.to_csv("outputs/breach_table_portfolio_scaled10d_covid.csv")

print("\nSaved:")
print(" - outputs/breach_table_portfolio_direct10d_full.csv")
print(" - outputs/breach_table_portfolio_scaled10d_full.csv")
print(" - outputs/breaches_portfolio_direct10d.csv")
print(" - outputs/breaches_portfolio_scaled10d.csv")
print(" - outputs/breach_table_portfolio_direct10d_covid.csv")
print(" - outputs/breach_table_portfolio_scaled10d_covid.csv")

# --------------------------
# Plots (COVID window)
# --------------------------
plot_breaches(
    df_covid_direct,
    "COVID window: Ret_10D vs DIRECT HS VaR_10D (Breaches marked)",
    "outputs/plot_breaches_covid_direct10d.png",
)
plot_breaches(
    df_covid_scaled,
    "COVID window: Ret_10D vs SCALED VaR_10D (Breaches marked)",
    "outputs/plot_breaches_covid_scaled10d.png",
)

print("\nSaved plots:")
print(" - outputs/plot_breaches_covid_direct10d.png")
print(" - outputs/plot_breaches_covid_scaled10d.png")