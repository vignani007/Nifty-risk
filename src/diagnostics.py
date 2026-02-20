import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


def ljung_box(series: pd.Series, lags: int = 10) -> dict:
    """
    Ljung-Box test for autocorrelation up to a chosen lag.
    Returns a small dict (easy to print in reports).
    """
    s = series.dropna()
    out = acorr_ljungbox(s, lags=[lags], return_df=True)

    return {
        "lags": int(lags),
        "lb_stat": float(out["lb_stat"].iloc[0]),
        "p_value": float(out["lb_pvalue"].iloc[0]),
    }


def excess_kurtosis(series: pd.Series) -> float:
    """
    Excess kurtosis = kurtosis - 3. Positive => fat tails vs Normal.
    (No scipy dependency; stable and transparent.)
    """
    x = series.dropna().values
    if len(x) < 5:
        return float("nan")

    m = x.mean()
    v = ((x - m) ** 2).mean()
    if v <= 0:
        return float("nan")

    k = ((x - m) ** 4).mean() / (v ** 2)
    return float(k - 3.0)


def basic_moments(series: pd.Series) -> dict:
    """
    Mean, std, skew (population), excess kurtosis.
    Useful for standardized residual diagnostics.
    """
    x = series.dropna().values
    if len(x) < 5:
        return {"mean": float("nan"), "std": float("nan"), "skew": float("nan"), "excess_kurt": float("nan")}

    m = x.mean()
    v = ((x - m) ** 2).mean()
    s = np.sqrt(v)

    if s == 0:
        return {"mean": float(m), "std": 0.0, "skew": float("nan"), "excess_kurt": float("nan")}

    skew = (((x - m) ** 3).mean()) / (s ** 3)
    ex_kurt = excess_kurtosis(pd.Series(x))

    return {"mean": float(m), "std": float(s), "skew": float(skew), "excess_kurt": float(ex_kurt)}
