"""
generate_data.py
Generates realistic synthetic stock market data for demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_stock_data(ticker: str, days: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Simulate realistic OHLCV stock data using Geometric Brownian Motion.
    """
    np.random.seed(seed)

    start_prices = {"AAPL": 150, "GOOGL": 2800, "TSLA": 200, "AMZN": 3200, "MSFT": 300}
    start_price = start_prices.get(ticker, 100)

    # GBM parameters
    mu = 0.0003       # daily drift
    sigma = 0.018     # daily volatility

    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(days)]
    dates = [d for d in dates if d.weekday() < 5][:days]  # weekdays only

    prices = [start_price]
    for _ in range(len(dates) - 1):
        shock = np.random.normal(mu, sigma)
        prices.append(round(prices[-1] * (1 + shock), 2))

    df = pd.DataFrame()
    df["Date"] = dates[:len(prices)]
    df["Close"] = prices

    # Derive OHLV realistically
    df["Open"]   = df["Close"].shift(1).fillna(df["Close"].iloc[0])
    df["High"]   = df[["Open", "Close"]].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.008, len(df))))
    df["Low"]    = df[["Open", "Close"]].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.008, len(df))))
    df["Volume"] = np.random.randint(20_000_000, 120_000_000, size=len(df))
    df["Ticker"] = ticker

    df = df.round(2)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    for ticker in ["AAPL", "GOOGL", "TSLA"]:
        df = generate_stock_data(ticker, days=500)
        df.to_csv(f"data/{ticker}.csv", index=False)
        print(f"[✓] Generated data/{ticker}.csv  ({len(df)} rows)")
