from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.var_models import portfolio_returns
from src.backtesting import kupiec_pof_test, exception_clustering_summary
from src.mc_backtest import rolling_mc_var, backtest_exceptions

tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2016-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")

rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

# Portfolio weights fixed from last 2y (consistent with your book definition)
rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

rp = portfolio_returns(rets, w)

alpha = 0.99
window = 504
n_sims = 5_000  # keep manageable; increase later if needed

# Rolling MC VaR series (shift by 1 is already handled because sample uses t-window:t-1)
var_mc_n = rolling_mc_var(rets, w, alpha=alpha, window=window, n_sims=n_sims, dist="normal", seed=42)
var_mc_t = rolling_mc_var(rets, w, alpha=alpha, window=window, n_sims=n_sims, dist="t", df=6.0, seed=42)

for name, var_series in [("MC_Normal", var_mc_n), ("MC_t_df6", var_mc_t)]:
    exc = backtest_exceptions(rp, var_series)
    res = kupiec_pof_test(exc, alpha=alpha)
    cl = exception_clustering_summary(exc)

    print(f"\n=== Backtest: {name} ===")
    print("Obs:", res["n"], "Exceptions:", res["x"], "Expected:", round(res["n"]*(1-alpha), 2))
    print("Kupiec LR:", round(res["LR_pof"], 4), "p-value:", round(res["p_value"], 4))
    print("Clustering:", cl)
