"""
app.py — Flask web server for Stock Market Analysis System
"""

import os
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
import pandas as pd

from generate_data     import generate_stock_data
from preprocessing     import clean_data, add_technical_indicators, get_summary_statistics
from ml_model          import train_linear_regression, predict_future

app = Flask(__name__)

SUPPORTED_TICKERS = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT"]


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def run_full_analysis(ticker: str):
    """Generate data, preprocess, run ML, return charts + stats as base64."""
    from visualization import (plot_price_and_ma, plot_volume, plot_rsi,
                                plot_macd, plot_return_distribution,
                                plot_ml_predictions, plot_dashboard)

    # Monkey-patch visualization functions to return base64 instead of saving files
    import visualization as viz

    charts = {}

    # 1. Generate / load data
    df = generate_stock_data(ticker, days=500)
    df["Date"] = pd.to_datetime(df["Date"])
    df = clean_data(df)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Stats
    stats = get_summary_statistics(df)

    # 3. ML
    model, scaler, test_df, metrics = train_linear_regression(df)
    future_df = predict_future(model, scaler, df, days=30)

    # 4. Build charts inline (return base64)
    DARK_BG = "#0d1117"; CARD_BG = "#161b22"
    GREEN = "#3fb950"; RED = "#f85149"; BLUE = "#58a6ff"
    ORANGE = "#e3b341"; PURPLE = "#bc8cff"; TEAL = "#76e3ea"
    GRAY = "#8b949e"; WHITE = "#e6edf3"

    def dark(fig, axes=None):
        fig.patch.set_facecolor(DARK_BG)
        if axes is None: return
        for ax in (axes if hasattr(axes, "__iter__") else [axes]):
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors=GRAY, labelsize=8)
            ax.xaxis.label.set_color(GRAY); ax.yaxis.label.set_color(GRAY)
            ax.title.set_color(WHITE)
            for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    # Price & MA
    fig, ax = plt.subplots(figsize=(13, 4)); dark(fig, ax)
    ax.plot(df["Date"], df["Close"],  color=BLUE,   lw=1.4, label="Close")
    ax.plot(df["Date"], df["SMA_20"], color=ORANGE, lw=1.2, ls="--", label="SMA 20")
    ax.plot(df["Date"], df["SMA_50"], color=PURPLE, lw=1.2, ls="--", label="SMA 50")
    ax.fill_between(df["Date"], df["BB_Upper"], df["BB_Lower"], color=BLUE, alpha=0.06)
    ax.plot(df["Date"], df["BB_Upper"], color=TEAL, lw=0.6, ls=":")
    ax.plot(df["Date"], df["BB_Lower"], color=TEAL, lw=0.6, ls=":")
    ax.set_title(f"{ticker} — Price & Moving Averages", fontsize=12)
    ax.set_ylabel("Price (USD)", color=GRAY)
    ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    ax.grid(color="#21262d", lw=0.5)
    fig.tight_layout()
    charts["price_ma"] = fig_to_base64(fig)

    # Volume
    fig, ax = plt.subplots(figsize=(13, 3)); dark(fig, ax)
    vcols = [GREEN if r >= 0 else RED for r in df["Daily_Return"].fillna(0)]
    ax.bar(df["Date"], df["Volume"] / 1e6, color=vcols, width=0.8, alpha=0.85)
    ax.set_title(f"{ticker} — Volume (Millions)", fontsize=12)
    ax.set_ylabel("Volume (M)", color=GRAY)
    ax.grid(axis="y", color="#21262d", lw=0.5)
    fig.tight_layout()
    charts["volume"] = fig_to_base64(fig)

    # RSI
    fig, ax = plt.subplots(figsize=(13, 3)); dark(fig, ax)
    ax.plot(df["Date"], df["RSI"], color=ORANGE, lw=1.2)
    ax.axhline(70, color=RED,   lw=0.8, ls="--", label="Overbought (70)")
    ax.axhline(30, color=GREEN, lw=0.8, ls="--", label="Oversold (30)")
    ax.fill_between(df["Date"], df["RSI"], 70, where=df["RSI"] >= 70, color=RED,   alpha=0.15)
    ax.fill_between(df["Date"], df["RSI"], 30, where=df["RSI"] <= 30, color=GREEN, alpha=0.15)
    ax.set_ylim(0, 100); ax.set_title(f"{ticker} — RSI (14-day)", fontsize=12)
    ax.set_ylabel("RSI", color=GRAY)
    ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    ax.grid(color="#21262d", lw=0.5); fig.tight_layout()
    charts["rsi"] = fig_to_base64(fig)

    # MACD
    fig, ax = plt.subplots(figsize=(13, 3)); dark(fig, ax)
    ax.plot(df["Date"], df["MACD"],        color=BLUE,   lw=1.2, label="MACD")
    ax.plot(df["Date"], df["MACD_Signal"], color=ORANGE, lw=1.2, label="Signal")
    hc = [GREEN if v >= 0 else RED for v in df["MACD_Hist"].fillna(0)]
    ax.bar(df["Date"], df["MACD_Hist"], color=hc, width=0.7, alpha=0.6)
    ax.axhline(0, color=GRAY, lw=0.6, ls="--")
    ax.set_title(f"{ticker} — MACD", fontsize=12); ax.set_ylabel("MACD", color=GRAY)
    ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    ax.grid(color="#21262d", lw=0.5); fig.tight_layout()
    charts["macd"] = fig_to_base64(fig)

    # Returns
    ret = df["Daily_Return"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4)); dark(fig, axes)
    axes[0].hist(ret, bins=50, color=BLUE, edgecolor=DARK_BG, alpha=0.85)
    axes[0].axvline(ret.mean(), color=ORANGE, lw=1.5, ls="--", label=f"Mean {ret.mean():.3f}%")
    axes[0].set_title("Daily Return Distribution", fontsize=11)
    axes[0].set_xlabel("Return (%)", color=GRAY); axes[0].set_ylabel("Frequency", color=GRAY)
    axes[0].legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    axes[0].grid(color="#21262d", lw=0.5)
    axes[1].plot(df["Date"], df["Cumulative_Return"] * 100, color=GREEN, lw=1.4)
    axes[1].axhline(0, color=GRAY, lw=0.6, ls="--")
    axes[1].set_title("Cumulative Return (%)", fontsize=11)
    axes[1].set_xlabel("Date", color=GRAY); axes[1].set_ylabel("Return (%)", color=GRAY)
    axes[1].grid(color="#21262d", lw=0.5)
    fig.suptitle(f"{ticker} — Returns Analysis", color=WHITE, fontsize=13)
    fig.tight_layout()
    charts["returns"] = fig_to_base64(fig)

    # ML predictions + forecast
    fig, axes = plt.subplots(2, 1, figsize=(13, 7)); dark(fig, axes)
    axes[0].plot(test_df["Date"], test_df["Actual"],    color=BLUE,   lw=1.4, label="Actual")
    axes[0].plot(test_df["Date"], test_df["Predicted"], color=ORANGE, lw=1.4, ls="--", label="Predicted")
    axes[0].fill_between(test_df["Date"], test_df["Actual"], test_df["Predicted"], alpha=0.12, color=ORANGE)
    axes[0].set_title(f"{ticker} — ML Test: Actual vs Predicted", fontsize=12)
    axes[0].set_ylabel("Price (USD)", color=GRAY)
    axes[0].text(0.01, 0.96, f"RMSE={metrics['RMSE']}  MAE={metrics['MAE']}  R²={metrics['R²']}",
                 transform=axes[0].transAxes, color=TEAL, fontsize=8, va="top")
    axes[0].legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    axes[0].grid(color="#21262d", lw=0.5)

    last_actual = test_df["Actual"].iloc[-30:]
    last_dates  = test_df["Date"].iloc[-30:]
    axes[1].plot(last_dates, last_actual, color=BLUE, lw=1.4, label="Recent Actual")
    axes[1].plot(future_df["Date"], future_df["Predicted_Price"], color=GREEN, lw=1.6, ls="--", label="30-day Forecast")
    axes[1].axvline(future_df["Date"].iloc[0], color=GRAY, lw=0.8, ls=":")
    band = future_df["Predicted_Price"] * 0.02
    axes[1].fill_between(future_df["Date"],
                         future_df["Predicted_Price"] - band,
                         future_df["Predicted_Price"] + band,
                         alpha=0.15, color=GREEN, label="±2% band")
    axes[1].set_title(f"{ticker} — 30-Day Price Forecast", fontsize=12)
    axes[1].set_ylabel("Price (USD)", color=GRAY)
    axes[1].legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    axes[1].grid(color="#21262d", lw=0.5)
    fig.tight_layout(h_pad=2)
    charts["ml"] = fig_to_base64(fig)

    return charts, stats, metrics, future_df


@app.route("/")
def index():
    return render_template("index.html", tickers=SUPPORTED_TICKERS)


@app.route("/analyse", methods=["POST"])
def analyse():
    data   = request.get_json()
    ticker = data.get("ticker", "AAPL").upper()
    if ticker not in SUPPORTED_TICKERS:
        return jsonify({"error": f"Unsupported ticker: {ticker}"}), 400

    try:
        charts, stats, metrics, future_df = run_full_analysis(ticker)
        forecast_snippet = [
            {"date": str(r["Date"])[:10], "price": r["Predicted_Price"]}
            for _, r in future_df.iterrows()
        ]
        return jsonify({
            "charts":   charts,
            "stats":    stats,
            "metrics":  metrics,
            "forecast": forecast_snippet,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
