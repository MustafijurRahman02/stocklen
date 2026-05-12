"""
fetch_data.py
Fetches real OHLCV stock data using yfinance.
Falls back to synthetic GBM data if download fails.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from generate_data import generate_stock_data


def fetch_real_stock_data(ticker: str, period: str = "2y") -> tuple[pd.DataFrame, bool]:
    """
    Download real stock data from Yahoo Finance.
    Returns (dataframe, is_real) — is_real=False means fallback to synthetic.
    """
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

        if raw.empty or len(raw) < 60:
            raise ValueError(f"Not enough data returned for {ticker}")

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date", "Datetime": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Ticker"] = ticker
        df = df.dropna().reset_index(drop=True)

        print(f"  [Data] ✓ Real data fetched for {ticker} — {len(df)} rows")
        return df, True

    except Exception as e:
        print(f"  [Data] ⚠ Could not fetch real data for {ticker}: {e}")
        print(f"  [Data] → Falling back to synthetic data")
        df = generate_stock_data(ticker, days=500)
        df["Date"] = pd.to_datetime(df["Date"])
        return df, False
