import pandas as pd
from scipy.stats import norm


def es_historical(port_ret: pd.Series, alpha: float = 0.99) -> float:
    """
    Historical ES (positive number).
    ES = -E[ r | r <= q_{1-alpha} ]
    """
    q = port_ret.quantile(1 - alpha)
    tail = port_ret[port_ret <= q]
    return float(-tail.mean())


def es_parametric_gaussian(port_ret: pd.Series, alpha: float = 0.99) -> float:
    """
    Gaussian ES (positive number).
    ES = -(mu - sigma * phi(z)/ (1-alpha)) where z = Phi^{-1}(1-alpha)
    """
    mu = port_ret.mean()
    sigma = port_ret.std(ddof=1)
    z = norm.ppf(1 - alpha)          # negative
    phi = norm.pdf(z)
    es = -(mu - sigma * (phi / (1 - alpha)))
    return float(es)
