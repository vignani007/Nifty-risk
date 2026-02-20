from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.var_models import portfolio_returns, var_ewma_parametric
from src.backtesting import (
    rolling_historical_var,
    compute_exceptions,
    kupiec_pof_test,
    exception_clustering_summary,
)

from src.garch_model import (
    fit_garch11_normal,
    garch_standardized_residuals,
    garch_var_series,
)

from src.diagnostics import ljung_box, basic_moments


# --------------------------
# 1) Data + portfolio
# --------------------------
tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2016-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")

rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

# Fix weights from last 2y for a stable "book definition"
rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

rp = portfolio_returns(rets, w)


alpha = 0.99

# Choose a rolling backtest window for HS benchmark
# (250 is standard-ish 1y; 504 is consistency with your estimation horizon)
window = 250


# --------------------------
# 2) Benchmark VaR series
# --------------------------
var_hs = rolling_historical_var(rp, alpha=alpha, window=window)
var_ewma = var_ewma_parametric(rp, alpha=alpha, lam=0.94).shift(1)  # look-ahead safe


# --------------------------
# 3) Fit GARCH (portfolio-level)
# --------------------------
res = fit_garch11_normal(rp, mean="Zero")
z = garch_standardized_residuals(res)

print("\n=== GARCH(1,1) Normal fit params ===")
print(res.params)

alpha1 = float(res.params.get("alpha[1]", 0.0))
beta1  = float(res.params.get("beta[1]", 0.0))
print("alpha+beta:", alpha1 + beta1)


print("\n=== Standardized residual diagnostics ===")
print("Moments:", basic_moments(z))
print("LB(resid, 10):", ljung_box(z, lags=10))
print("LB(resid^2, 10):", ljung_box(z**2, lags=10))


# --------------------------
# 4) GARCH VaR series (shift for forecast)
# --------------------------
var_garch = garch_var_series(res, alpha=alpha).shift(1)


# --------------------------
# 5) Backtest comparison
# --------------------------
for name, var_series in [("HS", var_hs), ("EWMA", var_ewma), ("GARCH-N", var_garch)]:
    exc = compute_exceptions(rp, var_series)
    res_bt = kupiec_pof_test(exc, alpha=alpha)
    cl = exception_clustering_summary(exc)

    print(f"\n=== Backtest: {name} ===")
    print("Obs:", res_bt["n"], "Exceptions:", res_bt["x"], "Expected:", round(res_bt["n"] * (1 - alpha), 2))
    print("Kupiec LR:", round(res_bt["LR_pof"], 4), "p-value:", round(res_bt["p_value"], 4))
    print("Clustering:", cl)
