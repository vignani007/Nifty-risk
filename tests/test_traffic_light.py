import os
import pandas as pd

from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights

from src.var_models import (
    portfolio_returns,
    var_ewma_parametric,
)
from src.backtesting import rolling_historical_var, compute_exceptions

from src.garch_model import (
    fit_garch11_normal,
    fit_garch11_t,
    garch_var_series,
    garch_var_series_t,
)

from src.traffic_light import basel_traffic_light


def last_250(series: pd.Series) -> pd.Series:
    return series.dropna().tail(250)


# --------------------------
# Build portfolio returns
# --------------------------
tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2016-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")
rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

rp = portfolio_returns(rets, w)

alpha = 0.99

# --------------------------
# Build VaR series (daily forecasts)
# NOTE: Traffic light uses last 250 obs
# --------------------------
# HS rolling VaR over 250 window (already forecast-style)
var_hs = rolling_historical_var(rp, alpha=alpha, window=250)

# EWMA (shift to avoid look-ahead)
var_ewma = var_ewma_parametric(rp, alpha=alpha, lam=0.94).shift(1)

# GARCH-N / GARCH-t (shift to forecast)
res_n = fit_garch11_normal(rp, mean="Zero")
res_t = fit_garch11_t(rp, mean="Zero")

var_gn = garch_var_series(res_n, alpha=alpha).shift(1)
var_gt = garch_var_series_t(res_t, alpha=alpha).shift(1)

models = {
    "HS(rolling250)": var_hs,
    "EWMA(lam=0.94)": var_ewma,
    "GARCH-N": var_gn,
    "GARCH-t": var_gt,
}

rows = []

for name, var_series in models.items():
    r250 = last_250(rp)
    v250 = last_250(var_series)

    # align dates strictly
    aligned = pd.concat([r250, v250], axis=1).dropna()
    ret = aligned.iloc[:, 0]
    var = aligned.iloc[:, 1]

    exc = compute_exceptions(ret, var)
    n_exc = int(exc.sum())

    tl = basel_traffic_light(n_exc)

    rows.append({
        "model": name,
        "obs": int(len(exc)),
        "exceptions": tl.exceptions,
        "zone": tl.zone,
        "plus_factor": tl.plus_factor,
        "multiplier_m": tl.multiplier_m,
    })

df = pd.DataFrame(rows).set_index("model").sort_values(["zone", "exceptions"])

print("\n=== Basel Traffic Light (last 250 obs, 99% VaR) ===")
print(df)

os.makedirs("outputs", exist_ok=True)
outpath = "outputs/traffic_light_250d_alpha99.csv"
df.to_csv(outpath)
print("\nSaved:", outpath)
