import numpy as np
import pandas as pd


def _portfolio_from_paths(sim_rets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # sim_rets: (M x N), weights: (N,)
    return sim_rets @ weights


def mc_var_es_normal(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.99,
    n_sims: int = 50_000,
    seed: int = 42
) -> tuple[float, float]:
    """
    Monte Carlo VaR/ES assuming multivariate Normal returns.
    returns: (T x N) asset return matrix
    weights: Series indexed by ticker
    """
    rng = np.random.default_rng(seed)

    cols = list(returns.columns)
    w = weights.reindex(cols).fillna(0.0).values

    mu = returns.mean().values
    Sigma = returns.cov().values

    sim = rng.multivariate_normal(mean=mu, cov=Sigma, size=n_sims)
    rp = _portfolio_from_paths(sim, w)

    q = np.quantile(rp, 1 - alpha)
    var = -q
    es = -rp[rp <= q].mean()

    return float(var), float(es)


def mc_var_es_student_t(
    returns: pd.DataFrame,
    weights: pd.Series,
    df: float = 6.0,
    alpha: float = 0.99,
    n_sims: int = 50_000,
    seed: int = 42
) -> tuple[float, float]:
    """
    Monte Carlo VaR/ES using an elliptical Student-t construction with df degrees of freedom.
    Keeps correlation via Sigma and introduces fat tails via chi-square scaling.
    """
    rng = np.random.default_rng(seed)

    cols = list(returns.columns)
    w = weights.reindex(cols).fillna(0.0).values

    mu = returns.mean().values
    Sigma = returns.cov().values

    # Step 1: draw z ~ N(0, Sigma)
    z = rng.multivariate_normal(mean=np.zeros(len(cols)), cov=Sigma, size=n_sims)

    # Step 2: draw u ~ chi2(df)
    u = rng.chisquare(df, size=n_sims)

    # Step 3: scale to get t-like heavy tails
    scale = np.sqrt(df / u).reshape(-1, 1)
    sim = mu + z * scale

    rp = _portfolio_from_paths(sim, w)

    q = np.quantile(rp, 1 - alpha)
    var = -q
    es = -rp[rp <= q].mean()

    return float(var), float(es)
