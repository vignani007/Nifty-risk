import numpy as np
import pandas as pd
from arch import arch_model


# ============================================================
# 1. Generic GARCH(1,1) Fit Function
# ============================================================

def fit_garch11(port_ret: pd.Series, mean: str = "Zero", dist: str = "normal") -> object:
    """
    Fit GARCH(1,1) to portfolio returns.

    dist: "normal" or "t"
    """

    r = port_ret.dropna()
    r_pct = 100.0 * r  # arch prefers percentage scale

    am = arch_model(
        r_pct,
        mean=mean,
        vol="GARCH",
        p=1,
        q=1,
        dist=dist
    )

    res = am.fit(disp="off")
    return res


# ============================================================
# 2. Convenience Wrappers
# ============================================================

def fit_garch11_normal(port_ret: pd.Series, mean: str = "Zero") -> object:
    return fit_garch11(port_ret, mean=mean, dist="normal")


def fit_garch11_t(port_ret: pd.Series, mean: str = "Zero") -> object:
    return fit_garch11(port_ret, mean=mean, dist="t")


# ============================================================
# 3. Index Compatibility Helper
# ============================================================

def _get_arch_index(res) -> pd.Index:
    if hasattr(res.model, "_x_index"):
        return res.model._x_index
    if hasattr(res.model, "_index"):
        return res.model._index
    return pd.RangeIndex(start=0, stop=len(res.conditional_volatility), step=1)


# ============================================================
# 4. Conditional Volatility
# ============================================================

def garch_conditional_sigma(res) -> pd.Series:
    idx = _get_arch_index(res)
    sigma_pct = pd.Series(res.conditional_volatility, index=idx)
    sigma = sigma_pct / 100.0
    sigma.name = "garch_sigma"
    return sigma


# ============================================================
# 5. Standardized Residuals
# ============================================================

def garch_standardized_residuals(res) -> pd.Series:
    idx = _get_arch_index(res)
    eps = pd.Series(res.resid, index=idx) / 100.0
    sigma = garch_conditional_sigma(res)
    z = eps / sigma
    z.name = "std_resid"
    return z


# ============================================================
# 6. GARCH-Normal VaR
# ============================================================

def garch_var_series(res, alpha: float = 0.99) -> pd.Series:
    from scipy.stats import norm

    sigma = garch_conditional_sigma(res)
    z_quantile = norm.ppf(1 - alpha)

    mean_cls = res.model.__class__.__name__.lower()

    if "zero" in mean_cls:
        mu = pd.Series(0.0, index=sigma.index)
    else:
        mu_const = float(res.params.get("mu", 0.0)) / 100.0
        mu = pd.Series(mu_const, index=sigma.index)

    var = -(mu + z_quantile * sigma)
    var.name = f"VaR_GARCHN_{int(alpha * 100)}"
    return var


# ============================================================
# 7. GARCH-t VaR
# ============================================================

def garch_var_series_t(res, alpha: float = 0.99) -> pd.Series:
    from scipy.stats import t as student_t

    sigma = garch_conditional_sigma(res)

    mean_cls = res.model.__class__.__name__.lower()

    if "zero" in mean_cls:
        mu = pd.Series(0.0, index=sigma.index)
    else:
        mu_const = float(res.params.get("mu", 0.0)) / 100.0
        mu = pd.Series(mu_const, index=sigma.index)

    nu = float(res.params.get("nu", np.nan))
    q = student_t.ppf(1 - alpha, df=nu)

    var = -(mu + q * sigma)
    var.name = f"VaR_GARCHt_{int(alpha * 100)}"
    return var
