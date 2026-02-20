from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.var_models import portfolio_returns, var_historical, var_parametric_gaussian, var_cornish_fisher
from src.es_models import es_historical, es_parametric_gaussian
from src.var_models import var_ewma_parametric
from src.fhs import fhs_var_es

tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2019-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")

rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)
rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)
rp = portfolio_returns(rets, w)

alpha = 0.99

print("HS VaR:", var_historical(rp, alpha))
print("Gaussian VaR:", var_parametric_gaussian(rp, alpha))
print("CF VaR:", var_cornish_fisher(rp, alpha))

# EWMA time series: print last value
ewma_var_ts = var_ewma_parametric(rp, alpha=alpha, lam=0.94)
print("EWMA VaR (latest):", float(ewma_var_ts.iloc[-1]))

fhs_var, fhs_es = fhs_var_es(rp, alpha=alpha, lam=0.94)
print("FHS VaR (today):", fhs_var)
print("FHS ES (today):", fhs_es)

print("HS ES:", es_historical(rp, alpha))
print("Gaussian ES:", es_parametric_gaussian(rp, alpha))
