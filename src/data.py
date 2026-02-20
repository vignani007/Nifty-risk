import yfinance as yf
import pandas as pd
from typing import List


def download_price_data(tickers: List[str],
                        start: str,
                        end: str) -> pd.DataFrame:

    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(axis=1, how="all")
    prices = prices.dropna()

    return prices
