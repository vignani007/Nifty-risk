import numpy as np
import pandas as pd


def fhs_var_es(
    port_ret: pd.Series,
    alpha: float = 0.99,
    lam: float = 0.94
) -> tuple[float, float]:
    """
    Filtered Historical Simulation (EWMA filter):
    1) compute EWMA sigma_t
    2) standardize returns z_t = r_t / sigma_t
    3) take historical quantile of z
    4) VaR_today = - q_z * sigma_today
    ES_today = - mean(z | z <= q_z) * sigma_today
    """
    from src.volatility import ewma_sigma

    sig = ewma_sigma(port_ret, lam=lam)
    aligned = port_ret.align(sig, join="inner")[0]
    sig = sig.loc[aligned.index]

    z = aligned / sig
    qz = z.quantile(1 - alpha)

    var_today = -qz * sig.iloc[-1]

    tail = z[z <= qz]
    es_today = -(tail.mean()) * sig.iloc[-1]

    return float(var_today), float(es_today)
