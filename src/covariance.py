import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Sample covariance matrix of returns.
    returns: DataFrame (T x N)
    """
    return returns.cov().values


def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit–Wolf shrinkage covariance matrix of returns.
    returns: DataFrame (T x N)
    """
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_
