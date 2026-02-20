import numpy as np
import pandas as pd
from scipy.stats import norm


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Compute portfolio return series: rp_t = sum_i w_i r_{i,t}
    returns: (T x N) DataFrame
    weights: Series indexed by ticker (N,)
    """
    w = weights.reindex(returns.columns).fillna(0.0)
    rp = returns @ w
    rp.name = "portfolio_return"
    return rp


def var_historical(port_ret: pd.Series, alpha: float = 0.99) -> float:
    """
    Historical Simulation VaR (positive number).
    VaR = -quantile_{1-alpha}(portfolio returns)
    """
    q = port_ret.quantile(1 - alpha)
    return float(-q)


def var_parametric_gaussian(port_ret: pd.Series, alpha: float = 0.99) -> float:
    """
    Gaussian parametric VaR using mean and std of portfolio returns (positive number).
    """
    mu = port_ret.mean()
    sigma = port_ret.std(ddof=1)
    z = norm.ppf(1 - alpha)  # negative
    var = -(mu + z * sigma)
    return float(var)


def var_cornish_fisher(port_ret: pd.Series, alpha: float = 0.99) -> float:
    """
    Cornish–Fisher VaR (positive number).
    Adjusts the Normal quantile using skewness and excess kurtosis.

    z_cf = z + (1/6)(z^2-1)S + (1/24)(z^3-3z)K - (1/36)(2z^3-5z)S^2
    where S = skewness, K = excess kurtosis
    """
    mu = port_ret.mean()
    sigma = port_ret.std(ddof=1)

    x = port_ret - mu
    m2 = (x**2).mean()
    m3 = (x**3).mean()
    m4 = (x**4).mean()

    S = m3 / (m2 ** 1.5)
    K_excess = (m4 / (m2 ** 2)) - 3.0

    z = norm.ppf(1 - alpha)  # negative
    z_cf = (
        z
        + (1/6) * (z**2 - 1) * S
        + (1/24) * (z**3 - 3*z) * K_excess
        - (1/36) * (2*z**3 - 5*z) * (S**2)
    )

    var = -(mu + z_cf * sigma)
    return float(var)


def var_ewma_parametric(port_ret: pd.Series, alpha: float = 0.99, lam: float = 0.94) -> pd.Series:
    """
    Time series of EWMA-Parametric VaR (positive numbers).
    Uses EWMA sigma_t and assumes conditional Normal.
    Returns VaR_t aligned to return dates.
    """
    from scipy.stats import norm
    from src.volatility import ewma_sigma

    mu = port_ret.mean()  # keep simple; could also use rolling mean
    sig = ewma_sigma(port_ret, lam=lam)
    z = norm.ppf(1 - alpha)  # negative

    var_ts = -(mu + z * sig)
    var_ts.name = f"VaR_EWMA_{int(alpha*100)}"
    return var_ts

