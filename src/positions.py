import pandas as pd
from pathlib import Path


def save_positions(weights: pd.Series, out_path: str) -> None:
    """
    Save portfolio weights to CSV (ticker, weight).
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df = weights.sort_values(ascending=False).reset_index()
    df.columns = ["ticker", "weight"]
    df.to_csv(p, index=False)


def load_positions(path: str) -> pd.Series:
    """
    Load portfolio weights from CSV into a Series indexed by ticker.
    """
    df = pd.read_csv(path)
    return pd.Series(df["weight"].values, index=df["ticker"].values, name="weight")
