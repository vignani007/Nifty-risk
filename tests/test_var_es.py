from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.var_models import portfolio_returns, var_historical, var_parametric_gaussian
from src.es_models import es_historical, es_parametric_gaussian

# 1) Data
tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2019-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")

rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

# 2) Weights (from last 2y window)
rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

# 3) Portfolio returns
rp = portfolio_returns(rets, w)

# 4) Risk metrics
alpha = 0.99
print("Portfolio length:", len(rp))
print("HS VaR(99%):", var_historical(rp, alpha))
print("Gaussian VaR(99%):", var_parametric_gaussian(rp, alpha))
print("HS ES(99%):", es_historical(rp, alpha))
print("Gaussian ES(99%):", es_parametric_gaussian(rp, alpha))
