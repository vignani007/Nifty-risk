from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.var_models import portfolio_returns
from src.var_models import var_ewma_parametric
from src.backtesting import (
    rolling_historical_var, rolling_gaussian_var,
    compute_exceptions, kupiec_pof_test, exception_clustering_summary
)

tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2016-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")

rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)
rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

rp = portfolio_returns(rets, w)

alpha = 0.99
window = 250

# Rolling VaR forecasts
var_hs = rolling_historical_var(rp, alpha=alpha, window=window)
var_g = rolling_gaussian_var(rp, alpha=alpha, window=window)
var_ewma = var_ewma_parametric(rp, alpha=alpha, lam=0.94)

# Align EWMA for backtest: ensure it uses info up to t-1 (shift by 1)
var_ewma = var_ewma.shift(1)
var_ewma.name = "VaR_EWMA_shifted"

for name, var_series in [("HS", var_hs), ("Gaussian", var_g), ("EWMA", var_ewma)]:
    exc = compute_exceptions(rp, var_series)
    res = kupiec_pof_test(exc, alpha=alpha)
    cl = exception_clustering_summary(exc)

    print(f"\n=== Backtest: {name} ===")
    print("Obs:", res["n"], "Exceptions:", res["x"], "Expected:", round(res["n"]*(1-alpha), 2))
    print("Kupiec LR:", round(res["LR_pof"], 4), "p-value:", round(res["p_value"], 4))
    print("Clustering:", cl)
