import numpy as np
import pandas as pd


def horizon_log_return(log_ret_1d: pd.Series, horizon_days: int = 10) -> pd.Series:
    r = log_ret_1d.dropna()
    out = r.rolling(horizon_days).sum()
    out.name = f"Ret_{horizon_days}D"
    return out


def rolling_hs_var_horizon(
    log_ret_1d: pd.Series,
    alpha: float = 0.99,
    horizon_days: int = 10,
    window: int = 250,
) -> pd.Series:
    ret_h = horizon_log_return(log_ret_1d, horizon_days=horizon_days)
    q = ret_h.rolling(window).quantile(1 - alpha)
    var_h = -q
    var_h.name = f"VaR_HS_{horizon_days}D_{int(alpha*100)}"
    return var_h.shift(1)


def scale_var_sqrt_time(var_1d: pd.Series, horizon_days: int = 10) -> pd.Series:
    v = var_1d.dropna()
    out = v * np.sqrt(horizon_days)
    out.name = f"VaR_sqrt_{horizon_days}D"
    return out