import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame (date index, tickers as columns).
    """
    prices = prices.sort_index()
    rets = np.log(prices / prices.shift(1))
    return rets.dropna(how="all")


def clean_returns(
    returns: pd.DataFrame,
    max_nan_frac: float = 0.02,
    winsorize: bool = True,
    winsor_q: float = 0.005
) -> pd.DataFrame:
    """
    Clean return matrix:
      - drop tickers with too many NaNs
      - fill remaining NaNs (ffill then bfill)
      - optional winsorization to reduce outlier impact
    """
    # Drop columns with too many missing values
    nan_frac = returns.isna().mean(axis=0)
    keep_cols = nan_frac[nan_frac <= max_nan_frac].index
    r = returns[keep_cols].copy()

    # Fill remaining missing values
    r = r.ffill().bfill()

    # Optional winsorization (clip extreme tails)
    if winsorize:
        lo = r.quantile(winsor_q)
        hi = r.quantile(1 - winsor_q)
        r = r.clip(lower=lo, upper=hi, axis=1)

    return r
