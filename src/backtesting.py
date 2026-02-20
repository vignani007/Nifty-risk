import numpy as np
import pandas as pd
from scipy.stats import chi2


def compute_exceptions(port_ret: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    Exception indicator: 1 if loss > VaR.
    We treat loss = -return. Exception if -r_t > VaR_t  <=> r_t < -VaR_t
    """
    r, v = port_ret.align(var_series, join="inner")
    exc = (r < -v).astype(int)
    exc.name = "exception"
    return exc


def kupiec_pof_test(exceptions: pd.Series, alpha: float) -> dict:
    """
    Kupiec Proportion-of-Failures (POF) test.
    H0: exception probability = (1-alpha)
    Returns LR statistic and p-value.
    """
    exc = exceptions.dropna().astype(int)
    n = len(exc)
    x = int(exc.sum())

    p = 1 - alpha  # expected exception probability

    # Avoid log(0)
    if x == 0 or x == n:
        # Degenerate case: still return something interpretable
        return {
            "n": n, "x": x, "alpha": alpha,
            "LR_pof": np.inf, "p_value": 0.0
        }

    phat = x / n
    LR = -2 * (
        (n - x) * np.log((1 - p) / (1 - phat)) +
        x * np.log(p / phat)
    )
    pval = 1 - chi2.cdf(LR, df=1)

    return {"n": n, "x": x, "alpha": alpha, "LR_pof": float(LR), "p_value": float(pval)}


def exception_clustering_summary(exceptions: pd.Series) -> dict:
    """
    Simple clustering diagnostics (not full Christoffersen):
    - number of exception runs
    - longest run length
    - average gap between exceptions
    """
    exc = exceptions.dropna().astype(int).values
    idx = np.where(exc == 1)[0]

    if len(idx) == 0:
        return {"num_exceptions": 0, "num_runs": 0, "longest_run": 0, "avg_gap": None}

    # runs of consecutive 1s
    longest_run = 1
    num_runs = 0
    i = 0
    while i < len(exc):
        if exc[i] == 1:
            num_runs += 1
            run_len = 1
            i += 1
            while i < len(exc) and exc[i] == 1:
                run_len += 1
                i += 1
            longest_run = max(longest_run, run_len)
        else:
            i += 1

    gaps = np.diff(idx)
    avg_gap = float(gaps.mean()) if len(gaps) > 0 else None

    return {
        "num_exceptions": int(len(idx)),
        "num_runs": int(num_runs),
        "longest_run": int(longest_run),
        "avg_gap": avg_gap
    }


def rolling_historical_var(port_ret: pd.Series, alpha: float = 0.99, window: int = 250) -> pd.Series:
    """
    Rolling Historical VaR forecast series.
    VaR_t computed using returns up to t-1 (look-ahead safe).
    """
    r = port_ret.dropna()
    var = pd.Series(index=r.index, dtype=float)

    for i in range(window, len(r)):
        sample = r.iloc[i - window:i]  # up to t-1
        var.iloc[i] = -sample.quantile(1 - alpha)

    var.name = f"VaR_HS_roll_{window}"
    return var


def rolling_gaussian_var(port_ret: pd.Series, alpha: float = 0.99, window: int = 250) -> pd.Series:
    """
    Rolling Gaussian VaR series using rolling mean/std up to t-1.
    """
    from scipy.stats import norm

    r = port_ret.dropna()
    z = norm.ppf(1 - alpha)

    mu = r.rolling(window).mean().shift(1)
    sig = r.rolling(window).std(ddof=1).shift(1)

    var = -(mu + z * sig)
    var.name = f"VaR_Gauss_roll_{window}"
    return var
