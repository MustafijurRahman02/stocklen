"""
ml_model.py
Applies Linear Regression to predict future stock closing prices.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix using lag features and technical indicators.
    Features: lag-1..lag-5 of Close, SMA_20, SMA_50, RSI, MACD, Volume.
    """
    feat = df[["Date", "Close", "Volume", "SMA_20", "SMA_50", "RSI", "MACD"]].copy()

    for lag in range(1, 6):
        feat[f"Lag_{lag}"] = feat["Close"].shift(lag)

    feat.dropna(inplace=True)
    feat.reset_index(drop=True, inplace=True)
    return feat


def train_linear_regression(df: pd.DataFrame):
    """
    Train a Linear Regression model.
    Returns: model, scaler, test_df (with Actual & Predicted columns), metrics dict.
    """
    feat = prepare_features(df)

    feature_cols = ["Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_5",
                    "SMA_20", "SMA_50", "RSI", "MACD", "Volume"]

    X = feat[feature_cols].values
    y = feat["Close"].values
    dates = feat["Date"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    metrics = {
        "RMSE":  round(rmse, 4),
        "MAE":   round(mae,  4),
        "R²":    round(r2,   4),
        "MSE":   round(mse,  4),
    }

    test_df = pd.DataFrame({
        "Date":      dates_test,
        "Actual":    y_test,
        "Predicted": y_pred,
    })

    print(f"  [ML] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return model, scaler, test_df, metrics


def predict_future(model, scaler, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    Iteratively predict `days` future closing prices using the trained model.
    """
    feat = prepare_features(df)
    feature_cols = ["Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_5",
                    "SMA_20", "SMA_50", "RSI", "MACD", "Volume"]

    last_row = feat[feature_cols].iloc[-1].values.copy()
    last_close = feat["Close"].iloc[-5:].values.tolist()  # last 5 prices

    # Use last known values for static features
    static_sma20 = feat["SMA_20"].iloc[-1]
    static_sma50 = feat["SMA_50"].iloc[-1]
    static_rsi   = feat["RSI"].iloc[-1]
    static_macd  = feat["MACD"].iloc[-1]
    static_vol   = feat["Volume"].iloc[-1]

    last_date  = pd.to_datetime(feat["Date"].iloc[-1])
    future_prices = []
    future_dates  = []

    for i in range(days):
        # Build feature row
        row = np.array([
            last_close[-1], last_close[-2], last_close[-3], last_close[-4], last_close[-5],
            static_sma20, static_sma50, static_rsi, static_macd, static_vol
        ]).reshape(1, -1)

        row_scaled = scaler.transform(row)
        pred = model.predict(row_scaled)[0]

        # Advance date (skip weekends)
        next_date = last_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)

        future_prices.append(round(pred, 2))
        future_dates.append(next_date)
        last_close.append(pred)
        last_date = next_date

        # Gently drift static indicators toward new price
        static_sma20 = static_sma20 * 0.95 + pred * 0.05
        static_sma50 = static_sma50 * 0.98 + pred * 0.02

    return pd.DataFrame({"Date": future_dates, "Predicted_Price": future_prices})
