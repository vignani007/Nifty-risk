from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.var_models import portfolio_returns, var_ewma_parametric
from src.backtesting import rolling_historical_var, compute_exceptions, kupiec_pof_test, exception_clustering_summary
from src.diagnostics import ljung_box, basic_moments

from src.garch_model import (
    fit_garch11_normal,
    fit_garch11_t,
    garch_standardized_residuals,
    garch_var_series,
    garch_var_series_t,
)

tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2016-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")
rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

# weights fixed from last 2y
rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

rp = portfolio_returns(rets, w)

alpha = 0.99
window = 250

var_hs = rolling_historical_var(rp, alpha=alpha, window=window)
var_ewma = var_ewma_parametric(rp, alpha=alpha, lam=0.94).shift(1)

# Fit both models
res_n = fit_garch11_normal(rp, mean="Zero")
res_t = fit_garch11_t(rp, mean="Zero")

# Diagnostics
z_n = garch_standardized_residuals(res_n)
z_t = garch_standardized_residuals(res_t)

print("\n=== GARCH-N params ===")
print(res_n.params)
print("alpha+beta:", float(res_n.params.get("alpha[1]",0)+res_n.params.get("beta[1]",0)))
print("Std resid moments:", basic_moments(z_n))
print("LB(resid):", ljung_box(z_n, 10))
print("LB(resid^2):", ljung_box(z_n**2, 10))

print("\n=== GARCH-t params ===")
print(res_t.params)
print("alpha+beta:", float(res_t.params.get("alpha[1]",0)+res_t.params.get("beta[1]",0)))
print("nu:", float(res_t.params.get("nu", float("nan"))))
print("Std resid moments:", basic_moments(z_t))
print("LB(resid):", ljung_box(z_t, 10))
print("LB(resid^2):", ljung_box(z_t**2, 10))

# VaR series (shift for forecast)
var_garch_n = garch_var_series(res_n, alpha=alpha).shift(1)
var_garch_t = garch_var_series_t(res_t, alpha=alpha).shift(1)

for name, var_series in [("HS", var_hs), ("EWMA", var_ewma), ("GARCH-N", var_garch_n), ("GARCH-t", var_garch_t)]:
    exc = compute_exceptions(rp, var_series)
    bt = kupiec_pof_test(exc, alpha=alpha)
    cl = exception_clustering_summary(exc)
    print(f"\n=== Backtest: {name} ===")
    print("Obs:", bt["n"], "Exceptions:", bt["x"], "Expected:", round(bt["n"]*(1-alpha), 2))
    print("Kupiec LR:", round(bt["LR_pof"], 4), "p-value:", round(bt["p_value"], 4))
    print("Clustering:", cl)
