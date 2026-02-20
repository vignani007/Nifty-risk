from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.monte_carlo import mc_var_es_student_t

tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2016-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")
rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

alpha = 0.99

for df in [5, 6, 8, 10, 15, 30]:
    v, e = mc_var_es_student_t(rets_est, w, df=df, alpha=alpha, n_sims=50_000, seed=42)
    print(f"df={df:>2}  VaR={v:.5f}  ES={e:.5f}")
