import numpy as np
import pandas as pd
from scipy.optimize import minimize


def min_variance_weights(
    cov: np.ndarray,
    tickers: list[str],
    weight_cap: float = 0.05
) -> pd.Series:
    """
    Min-variance long-only portfolio with weight cap.
    Constraints:
      - sum(w)=1
      - 0 <= w_i <= weight_cap
    """

    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError("cov must be square (N x N).")
    if len(tickers) != n:
        raise ValueError("tickers length must match cov dimension.")

    def objective(w):
        return float(w.T @ cov @ w)

    # Sum to 1 constraint
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Long-only + cap
    bounds = [(0.0, weight_cap) for _ in range(n)]

    # Start from equal weights (but capped)
    w0 = np.ones(n) / n
    if np.any(w0 > weight_cap):
        w0 = np.minimum(w0, weight_cap)
        w0 = w0 / w0.sum()

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 10_000, "ftol": 1e-12}
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    # Numerical cleanup
    w[w < 0] = 0.0
    w = w / w.sum()

    return pd.Series(w, index=tickers, name="weight")
