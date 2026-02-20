from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights

from src.var_models import (
    portfolio_returns,
    var_historical,
    var_parametric_gaussian,
    var_cornish_fisher,
    var_ewma_parametric,
)

from src.monte_carlo import mc_var_es_normal, mc_var_es_student_t
from src.garch_model import (
    fit_garch11_normal,
    fit_garch11_t,
    garch_var_series,
    garch_var_series_t,
)

from src.margin import im_proxy_table

import os


# --------------------------
# 1) Data + portfolio
# --------------------------
tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2016-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")

rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

# Fix weights from last 2y
rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

rp = portfolio_returns(rets, w)

alpha = 0.99
mpor_days = 10


# --------------------------
# 2) 1-day VaR point estimates
# --------------------------
var_hs = var_historical(rp, alpha=alpha)
var_g  = var_parametric_gaussian(rp, alpha=alpha)
var_cf = var_cornish_fisher(rp, alpha=alpha)

var_ewma_series = var_ewma_parametric(rp, alpha=alpha, lam=0.94)
var_ewma = float(var_ewma_series.dropna().iloc[-1])

# MC (use estimation window)
mc_v_n, _ = mc_var_es_normal(rets_est, w, alpha=alpha, n_sims=50_000, seed=42)
mc_v_t, _ = mc_var_es_student_t(rets_est, w, df=6, alpha=alpha, n_sims=50_000, seed=42)

# GARCH snapshots
res_gn = fit_garch11_normal(rp, mean="Zero")
res_gt = fit_garch11_t(rp, mean="Zero")

var_gn_series = garch_var_series(res_gn, alpha=alpha)
var_gt_series = garch_var_series_t(res_gt, alpha=alpha)

var_gn = float(var_gn_series.dropna().iloc[-1])
var_gt = float(var_gt_series.dropna().iloc[-1])

# NOTE: FHS not included here yet (we’ll add once we locate your FHS function)
var_dict = {
    "HS": var_hs,
    "Gaussian": var_g,
    "Cornish-Fisher": var_cf,
    "EWMA(latest)": var_ewma,
    "MC Normal (2y)": float(mc_v_n),
    "MC t df=6 (2y)": float(mc_v_t),
    "GARCH-N(latest)": var_gn,
    "GARCH-t(latest)": var_gt,
}

df = im_proxy_table(var_dict, mpor_days=mpor_days)

print("\n=== IM Proxy Table (MPOR=10d, alpha=99%) ===")
print(df)

os.makedirs("outputs", exist_ok=True)
outpath = "outputs/im_proxy_table_10d_alpha99.csv"
df.to_csv(outpath)
print("\nSaved:", outpath)
