import numpy as np
import pandas as pd
from typing import Literal


def rolling_mc_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.99,
    window: int = 504,
    n_sims: int = 20_000,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    seed: int = 42,
) -> pd.Series:
    """
    Rolling 1-day-ahead MC VaR forecast series (look-ahead safe).
    For each date t (starting after `window` observations), estimate mu and Sigma
    using returns up to t-1, simulate n_sims scenarios, compute VaR_t.

    returns: (T x N) asset return matrix
    weights: Series indexed by ticker
    """
    rng = np.random.default_rng(seed)

    cols = list(returns.columns)
    r = returns[cols].dropna(how="any").copy()
    w = weights.reindex(cols).fillna(0.0).values

    var = pd.Series(index=r.index, dtype=float)

    for i in range(window, len(r)):
        # estimation sample up to t-1
        sample = r.iloc[i - window:i]
        mu = sample.mean().values
        Sigma = sample.cov().values

        if dist == "normal":
            sim = rng.multivariate_normal(mean=mu, cov=Sigma, size=n_sims)
        elif dist == "t":
            # elliptical t: mu + z * sqrt(df/u)
            z = rng.multivariate_normal(mean=np.zeros(len(cols)), cov=Sigma, size=n_sims)
            u = rng.chisquare(df, size=n_sims)
            scale = np.sqrt(df / u).reshape(-1, 1)
            sim = mu + z * scale
        else:
            raise ValueError("dist must be 'normal' or 't'")

        rp = sim @ w
        q = np.quantile(rp, 1 - alpha)
        var.iloc[i] = -q  # positive VaR

    var.name = f"VaR_MC_{dist}_roll_{window}"
    return var


def backtest_exceptions(port_ret: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    Exception indicator: 1 if r_t < -VaR_t (loss exceeds VaR).
    """
    r, v = port_ret.align(var_series, join="inner")
    exc = (r < -v).astype(int)
    exc.name = "exception"
    return exc
