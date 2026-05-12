"""
preprocessing.py
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np


def load_stock_data(filepath: str) -> pd.DataFrame:
    """Load CSV and parse date column."""
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and fill/drop missing values."""
    before = len(df)
    df = df.drop_duplicates(subset="Date")
    df = df.dropna(subset=["Close", "Open", "High", "Low"])

    # Forward-fill any remaining NaNs in numeric cols
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].ffill()

    after = len(df)
    print(f"  [Preprocessing] Rows before: {before}  |  After cleaning: {after}  |  Removed: {before - after}")
    return df.reset_index(drop=True)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute widely-used technical indicators:
      - SMA  (20-day & 50-day Simple Moving Average)
      - EMA  (12-day & 26-day Exponential Moving Average)
      - MACD & Signal line
      - RSI  (14-day Relative Strength Index)
      - Bollinger Bands (20-day, 2 std)
      - Daily Return & Cumulative Return
      - Volatility (20-day rolling std of daily return)
    """
    df = df.copy()

    # ---------- Moving averages ----------
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # ---------- MACD ----------
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ---------- RSI ----------
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ---------- Bollinger Bands ----------
    df["BB_Mid"]   = df["Close"].rolling(window=20).mean()
    bb_std         = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

    # ---------- Returns & Volatility ----------
    df["Daily_Return"]      = df["Close"].pct_change() * 100
    df["Cumulative_Return"] = (1 + df["Close"].pct_change()).cumprod() - 1
    df["Volatility_20"]     = df["Daily_Return"].rolling(window=20).std()

    return df


def get_summary_statistics(df: pd.DataFrame) -> dict:
    """Return a dict of key summary statistics for the stock."""
    latest = df["Close"].iloc[-1]
    start  = df["Close"].iloc[0]
    stats = {
        "Latest Close":      round(latest, 2),
        "Start Close":       round(start, 2),
        "Total Return (%)":  round((latest / start - 1) * 100, 2),
        "Max Price":         round(df["Close"].max(), 2),
        "Min Price":         round(df["Close"].min(), 2),
        "Avg Daily Return":  round(df["Daily_Return"].mean(), 4),
        "Volatility (std)":  round(df["Daily_Return"].std(), 4),
        "Sharpe Ratio":      round(df["Daily_Return"].mean() / df["Daily_Return"].std() * (252 ** 0.5), 3),
        "Total Volume":      int(df["Volume"].sum()),
    }
    return stats
