from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.var_models import portfolio_returns, var_historical
from src.es_models import es_historical
from src.monte_carlo import mc_var_es_normal, mc_var_es_student_t

tickers = NIFTY50_TICKERS
prices = download_price_data(tickers, "2016-01-01", "2023-12-31")
prices = prices.dropna(axis=1, how="all")
rets = clean_returns(compute_log_returns(prices), max_nan_frac=0.05)

# weights from last 2y
rets_est = rets.tail(504)
Sigma = ledoit_wolf_covariance(rets_est)
w = min_variance_weights(Sigma, tickers=list(rets_est.columns), weight_cap=0.05)

# portfolio series (for HS reference)
rp = portfolio_returns(rets, w)

alpha = 0.99

# Use same estimation window for MC parameters
varN, esN = mc_var_es_normal(rets_est, w, alpha=alpha, n_sims=50_000, seed=42)
varT, esT = mc_var_es_student_t(rets_est, w, df=6.0, alpha=alpha, n_sims=50_000, seed=42)

print("HS VaR/ES (full sample):", var_historical(rp, alpha), es_historical(rp, alpha))
print("MC Normal VaR/ES (est window):", varN, esN)
print("MC Student-t VaR/ES (df=6):", varT, esT)
