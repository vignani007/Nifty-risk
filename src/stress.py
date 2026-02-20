import numpy as np
import pandas as pd


def hs_var_es(port_ret: pd.Series, alpha: float = 0.99) -> tuple[float, float]:
    """
    Historical (empirical) VaR and ES for a return series.
    Returns positive numbers (loss magnitudes).
    """
    r = port_ret.dropna()
    q = r.quantile(1 - alpha)  # negative
    var = -float(q)
    tail = r[r <= q]
    es = -float(tail.mean()) if len(tail) > 0 else float("nan")
    return var, es


def worst_days(port_ret: pd.Series, k: int = 10) -> pd.DataFrame:
    """
    k worst daily returns and associated 1-day loss.
    """
    r = port_ret.dropna().sort_values()  # most negative first
    out = pd.DataFrame({"LogReturn": r.head(k)})
    out["Loss_1D"] = -out["LogReturn"]
    return out


def rolling_horizon_log_return(port_ret: pd.Series, horizon_days: int = 10) -> pd.Series:
    """
    Rolling multi-day log return (sum of daily log returns).
    """
    r = port_ret.dropna()
    ret_h = r.rolling(horizon_days).sum()
    ret_h.name = f"Ret_{horizon_days}D"
    return ret_h


def worst_horizon(port_ret: pd.Series, horizon_days: int = 10, k: int = 5) -> pd.DataFrame:
    """
    Worst k rolling horizon returns (e.g., worst 10-day cumulative return).
    """
    ret_h = rolling_horizon_log_return(port_ret, horizon_days=horizon_days).dropna().sort_values()
    out = pd.DataFrame({ret_h.name: ret_h.head(k)})
    out[f"Loss_{horizon_days}D"] = -out[ret_h.name]
    return out


def period_slice(series: pd.Series, start: str, end: str) -> pd.Series:
    """
    Slice a time series by date strings.
    """
    s = series.dropna()
    return s.loc[start:end]


def portfolio_sigma_from_cov(Sigma: np.ndarray, w: np.ndarray) -> float:
    """
    Portfolio standard deviation from covariance matrix and weights.
    """
    return float(np.sqrt(w.T @ Sigma @ w))


def shock_loss_sigma(port_sigma: float, n_sigma: float = 3.0) -> float:
    """
    Single-day parametric shock loss = n_sigma * sigma (positive).
    """
    return float(n_sigma * port_sigma)


def corr_from_cov(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert covariance to (correlation, vol vector).
    """
    vol = np.sqrt(np.diag(Sigma))
    vol_safe = np.where(vol == 0, 1e-12, vol)
    Corr = Sigma / np.outer(vol_safe, vol_safe)
    np.fill_diagonal(Corr, 1.0)
    return Corr, vol


def cov_from_corr(vol: np.ndarray, Corr: np.ndarray) -> np.ndarray:
    """
    Build covariance from vol and correlation.
    """
    return np.outer(vol, vol) * Corr


def stress_correlations(Corr: np.ndarray, factor: float = 1.3, cap: float = 0.99) -> np.ndarray:
    """
    Stress correlations by scaling off-diagonals and capping absolute value.
    """
    C = Corr.copy()
    n = C.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = np.sign(C[i, j]) * min(abs(C[i, j]) * factor, cap)
    np.fill_diagonal(C, 1.0)
    return C


def stress_vols(vol: np.ndarray, vol_mult: float = 1.5) -> np.ndarray:
    """
    Stress vols by a multiplier (e.g., 1.5x or 2.0x).
    """
    return vol * float(vol_mult)