import numpy as np
import pandas as pd


def sqrt_time_scale(x: float, days: int) -> float:
    """
    Square-root-of-time scaling.
    Assumes iid / weak dependence of daily P&L increments.
    """
    return float(x) * np.sqrt(days)


def im_proxy_from_var(var_1d: float, mpor_days: int = 10) -> float:
    """
    Simple Initial Margin proxy from 1-day VaR using MPOR scaling.

    IM_proxy ≈ VaR_1d * sqrt(MPOR_days)

    Notes:
    - This is a proxy, not CCP/SIMM.
    - Assumes iid scaling; ignores liquidity & gap risk.
    """
    return sqrt_time_scale(var_1d, mpor_days)


def im_proxy_table(
    var_dict: dict,
    mpor_days: int = 10,
) -> pd.DataFrame:
    """
    Build a table of IM proxies from a dict of {model_name: VaR_1d}.
    """
    rows = []
    for name, v in var_dict.items():
        rows.append({
            "model": name,
            "VaR_1d": float(v),
            f"IM_proxy_{mpor_days}d": im_proxy_from_var(float(v), mpor_days=mpor_days)
        })

    col = f"IM_proxy_{mpor_days}d"
    df = pd.DataFrame(rows).set_index("model").sort_values(col)
    return df
