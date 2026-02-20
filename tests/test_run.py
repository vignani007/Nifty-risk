from src.config import NIFTY50_TICKERS
from src.data import download_price_data
from src.returns import compute_log_returns, clean_returns
from src.covariance import ledoit_wolf_covariance
from src.portfolio import min_variance_weights
from src.positions import save_positions


def main():
    print("Starting test_run...")

    tickers = NIFTY50_TICKERS
    prices = download_price_data(
        tickers=tickers,
        start="2019-01-01",
        end="2023-12-31"
    )

    prices = prices.dropna(axis=1, how="all")
    print("Downloaded prices shape:", prices.shape)

    rets = compute_log_returns(prices)
    rets = clean_returns(rets, max_nan_frac=0.05)

    print("Clean returns shape:", rets.shape)

    rets_est = rets.tail(504)
    Sigma = ledoit_wolf_covariance(rets_est)

    weights = min_variance_weights(
        cov=Sigma,
        tickers=list(rets_est.columns),
        weight_cap=0.05
    )

    print("\n=== Portfolio weights summary ===")
    print("Num assets used:", len(weights))
    print("Sum weights:", float(weights.sum()))
    print("Max weight:", float(weights.max()))
    print("\nTop 10 weights:")
    print(weights.sort_values(ascending=False).head(10))

    save_positions(weights, "outputs/positions_minvar_lw_cap5.csv")
    print("\nSaved positions to outputs/positions_minvar_lw_cap5.csv")


if __name__ == "__main__":
    main()
