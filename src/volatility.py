import numpy as np
import pandas as pd


def ewma_variance(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """
    EWMA variance forecast series.
    sigma_t^2 = lam*sigma_{t-1}^2 + (1-lam)*r_{t-1}^2
    """
    r = returns.dropna()
    var = pd.Series(index=r.index, dtype=float)

    # initialize with sample variance
    var.iloc[0] = r.var(ddof=1)

    for i in range(1, len(r)):
        var.iloc[i] = lam * var.iloc[i-1] + (1 - lam) * (r.iloc[i-1] ** 2)

    var.name = "ewma_var"
    return var


def ewma_sigma(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    v = ewma_variance(returns, lam=lam)
    return np.sqrt(v).rename("ewma_sigma")
