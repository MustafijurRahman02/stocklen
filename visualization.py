"""
visualization.py
Produces all charts for the stock analysis project.
"""

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os

# ── Global style ─────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
GREEN     = "#3fb950"
RED       = "#f85149"
BLUE      = "#58a6ff"
ORANGE    = "#e3b341"
PURPLE    = "#bc8cff"
TEAL      = "#76e3ea"
GRAY      = "#8b949e"
WHITE     = "#e6edf3"

def _apply_dark_style(fig, axes=None):
    fig.patch.set_facecolor(DARK_BG)
    if axes is None:
        return
    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=GRAY, labelsize=8)
        ax.xaxis.label.set_color(GRAY)
        ax.yaxis.label.set_color(GRAY)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")


# ── 1. Price & Moving Averages ─────────────────────────────────────────────
def plot_price_and_ma(df: pd.DataFrame, ticker: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    _apply_dark_style(fig, ax)

    ax.plot(df["Date"], df["Close"],  color=BLUE,   lw=1.4, label="Close")
    ax.plot(df["Date"], df["SMA_20"], color=ORANGE, lw=1.2, ls="--", label="SMA 20")
    ax.plot(df["Date"], df["SMA_50"], color=PURPLE, lw=1.2, ls="--", label="SMA 50")

    ax.fill_between(df["Date"], df["BB_Upper"], df["BB_Lower"],
                    color=BLUE, alpha=0.06, label="Bollinger Bands")
    ax.plot(df["Date"], df["BB_Upper"], color=TEAL, lw=0.6, ls=":")
    ax.plot(df["Date"], df["BB_Lower"], color=TEAL, lw=0.6, ls=":")

    ax.set_title(f"{ticker}  —  Price & Moving Averages", fontsize=13, pad=10)
    ax.set_ylabel("Price (USD)")
    ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    ax.grid(color="#21262d", lw=0.5)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{ticker}_price_ma.png")
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [Chart] Saved {path}")
    return path


# ── 2. Volume Bar Chart ────────────────────────────────────────────────────
def plot_volume(df: pd.DataFrame, ticker: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(14, 3))
    _apply_dark_style(fig, ax)

    colors = [GREEN if r >= 0 else RED for r in df["Daily_Return"].fillna(0)]
    ax.bar(df["Date"], df["Volume"] / 1e6, color=colors, width=0.8, alpha=0.85)
    ax.set_title(f"{ticker}  —  Trading Volume (Millions)", fontsize=12, pad=8)
    ax.set_ylabel("Volume (M)")
    ax.grid(axis="y", color="#21262d", lw=0.5)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{ticker}_volume.png")
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [Chart] Saved {path}")
    return path


# ── 3. RSI ──────────────────────────────────────────────────────────────────
def plot_rsi(df: pd.DataFrame, ticker: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(14, 3))
    _apply_dark_style(fig, ax)

    ax.plot(df["Date"], df["RSI"], color=ORANGE, lw=1.2)
    ax.axhline(70, color=RED,   lw=0.8, ls="--", label="Overbought (70)")
    ax.axhline(30, color=GREEN, lw=0.8, ls="--", label="Oversold (30)")
    ax.fill_between(df["Date"], df["RSI"], 70, where=df["RSI"] >= 70,
                    color=RED,   alpha=0.15)
    ax.fill_between(df["Date"], df["RSI"], 30, where=df["RSI"] <= 30,
                    color=GREEN, alpha=0.15)
    ax.set_ylim(0, 100)
    ax.set_title(f"{ticker}  —  RSI (14-day)", fontsize=12, pad=8)
    ax.set_ylabel("RSI")
    ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    ax.grid(color="#21262d", lw=0.5)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{ticker}_rsi.png")
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [Chart] Saved {path}")
    return path


# ── 4. MACD ─────────────────────────────────────────────────────────────────
def plot_macd(df: pd.DataFrame, ticker: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(14, 3))
    _apply_dark_style(fig, ax)

    ax.plot(df["Date"], df["MACD"],        color=BLUE,   lw=1.2, label="MACD")
    ax.plot(df["Date"], df["MACD_Signal"], color=ORANGE, lw=1.2, label="Signal")
    hist_colors = [GREEN if v >= 0 else RED for v in df["MACD_Hist"].fillna(0)]
    ax.bar(df["Date"], df["MACD_Hist"], color=hist_colors, width=0.7, alpha=0.6, label="Histogram")
    ax.axhline(0, color=GRAY, lw=0.6, ls="--")
    ax.set_title(f"{ticker}  —  MACD", fontsize=12, pad=8)
    ax.set_ylabel("MACD")
    ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    ax.grid(color="#21262d", lw=0.5)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{ticker}_macd.png")
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [Chart] Saved {path}")
    return path


# ── 5. Daily Returns Distribution ───────────────────────────────────────────
def plot_return_distribution(df: pd.DataFrame, ticker: str, out_dir: str):
    ret = df["Daily_Return"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    _apply_dark_style(fig, axes)

    # Histogram
    axes[0].hist(ret, bins=50, color=BLUE, edgecolor=DARK_BG, alpha=0.85)
    axes[0].axvline(ret.mean(), color=ORANGE, lw=1.5, ls="--", label=f"Mean {ret.mean():.3f}%")
    axes[0].set_title("Daily Return Distribution", fontsize=11)
    axes[0].set_xlabel("Return (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    axes[0].grid(color="#21262d", lw=0.5)

    # Cumulative return
    axes[1].plot(df["Date"], df["Cumulative_Return"] * 100, color=GREEN, lw=1.4)
    axes[1].axhline(0, color=GRAY, lw=0.6, ls="--")
    axes[1].set_title("Cumulative Return (%)", fontsize=11)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Return (%)")
    axes[1].grid(color="#21262d", lw=0.5)

    fig.suptitle(f"{ticker}  —  Returns Analysis", color=WHITE, fontsize=13, y=1.01)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{ticker}_returns.png")
    fig.savefig(path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart] Saved {path}")
    return path


# ── 6. ML Actual vs Predicted + Future Forecast ────────────────────────────
def plot_ml_predictions(test_df: pd.DataFrame, future_df: pd.DataFrame,
                        ticker: str, metrics: dict, out_dir: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    _apply_dark_style(fig, axes)

    # Panel 1: Test predictions
    ax = axes[0]
    ax.plot(test_df["Date"], test_df["Actual"],    color=BLUE,   lw=1.4, label="Actual")
    ax.plot(test_df["Date"], test_df["Predicted"], color=ORANGE, lw=1.4, ls="--", label="Predicted")
    ax.fill_between(test_df["Date"],
                    test_df["Actual"], test_df["Predicted"],
                    alpha=0.12, color=ORANGE)
    ax.set_title(f"{ticker}  —  ML Test Set: Actual vs Predicted", fontsize=12, pad=8)
    ax.set_ylabel("Price (USD)")
    info = f"RMSE={metrics['RMSE']}   MAE={metrics['MAE']}   R²={metrics['R²']}"
    ax.text(0.01, 0.96, info, transform=ax.transAxes,
            color=TEAL, fontsize=8, va="top")
    ax.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    ax.grid(color="#21262d", lw=0.5)

    # Panel 2: Future forecast
    ax2 = axes[1]
    last_actual = test_df["Actual"].iloc[-30:]
    last_dates  = test_df["Date"].iloc[-30:]
    ax2.plot(last_dates,           last_actual,              color=BLUE,   lw=1.4, label="Recent Actual")
    ax2.plot(future_df["Date"],    future_df["Predicted_Price"], color=GREEN, lw=1.6, ls="--", label="30-day Forecast")
    ax2.axvline(future_df["Date"].iloc[0], color=GRAY, lw=0.8, ls=":")

    # Confidence band (±2%)
    band = future_df["Predicted_Price"] * 0.02
    ax2.fill_between(future_df["Date"],
                     future_df["Predicted_Price"] - band,
                     future_df["Predicted_Price"] + band,
                     alpha=0.15, color=GREEN, label="±2% band")

    ax2.set_title(f"{ticker}  —  30-Day Price Forecast", fontsize=12, pad=8)
    ax2.set_ylabel("Price (USD)")
    ax2.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=8)
    ax2.grid(color="#21262d", lw=0.5)

    fig.tight_layout(h_pad=2.5)
    path = os.path.join(out_dir, f"{ticker}_ml_predictions.png")
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [Chart] Saved {path}")
    return path


# ── 7. Dashboard Summary (single composite figure) ──────────────────────────
def plot_dashboard(df: pd.DataFrame, test_df: pd.DataFrame,
                   future_df: pd.DataFrame, stats: dict,
                   ticker: str, metrics: dict, out_dir: str):
    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    ax_price  = fig.add_subplot(gs[0, :2])
    ax_vol    = fig.add_subplot(gs[1, :2])
    ax_rsi    = fig.add_subplot(gs[2, 0])
    ax_macd   = fig.add_subplot(gs[2, 1])
    ax_ml     = fig.add_subplot(gs[0, 2])
    ax_future = fig.add_subplot(gs[1, 2])
    ax_stats  = fig.add_subplot(gs[2, 2])

    all_axes = [ax_price, ax_vol, ax_rsi, ax_macd, ax_ml, ax_future, ax_stats]
    _apply_dark_style(fig, all_axes)

    # ── Price + MAs
    ax_price.plot(df["Date"], df["Close"],  color=BLUE,   lw=1.2, label="Close")
    ax_price.plot(df["Date"], df["SMA_20"], color=ORANGE, lw=0.9, ls="--", label="SMA20")
    ax_price.plot(df["Date"], df["SMA_50"], color=PURPLE, lw=0.9, ls="--", label="SMA50")
    ax_price.fill_between(df["Date"], df["BB_Upper"], df["BB_Lower"], color=BLUE, alpha=0.05)
    ax_price.set_title("Price + Bollinger Bands", fontsize=9)
    ax_price.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=6, ncol=3)
    ax_price.grid(color="#21262d", lw=0.4)

    # ── Volume
    vcols = [GREEN if r >= 0 else RED for r in df["Daily_Return"].fillna(0)]
    ax_vol.bar(df["Date"], df["Volume"] / 1e6, color=vcols, width=0.8, alpha=0.8)
    ax_vol.set_title("Volume (M)", fontsize=9)
    ax_vol.grid(axis="y", color="#21262d", lw=0.4)

    # ── RSI
    ax_rsi.plot(df["Date"], df["RSI"], color=ORANGE, lw=1.0)
    ax_rsi.axhline(70, color=RED,   lw=0.7, ls="--")
    ax_rsi.axhline(30, color=GREEN, lw=0.7, ls="--")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title("RSI (14)", fontsize=9)
    ax_rsi.grid(color="#21262d", lw=0.4)

    # ── MACD
    ax_macd.plot(df["Date"], df["MACD"],        color=BLUE,   lw=0.9)
    ax_macd.plot(df["Date"], df["MACD_Signal"], color=ORANGE, lw=0.9)
    hc = [GREEN if v >= 0 else RED for v in df["MACD_Hist"].fillna(0)]
    ax_macd.bar(df["Date"], df["MACD_Hist"], color=hc, width=0.7, alpha=0.5)
    ax_macd.set_title("MACD", fontsize=9)
    ax_macd.grid(color="#21262d", lw=0.4)

    # ── ML actual vs predicted
    ax_ml.plot(test_df["Date"], test_df["Actual"],    color=BLUE,   lw=1.0, label="Actual")
    ax_ml.plot(test_df["Date"], test_df["Predicted"], color=ORANGE, lw=1.0, ls="--", label="Pred")
    ax_ml.set_title(f"ML Test  R²={metrics['R²']}", fontsize=9)
    ax_ml.legend(facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=6)
    ax_ml.grid(color="#21262d", lw=0.4)

    # ── Future
    ax_future.plot(future_df["Date"], future_df["Predicted_Price"], color=GREEN, lw=1.2)
    band = future_df["Predicted_Price"] * 0.02
    ax_future.fill_between(future_df["Date"],
                           future_df["Predicted_Price"] - band,
                           future_df["Predicted_Price"] + band,
                           alpha=0.2, color=GREEN)
    ax_future.set_title("30-Day Forecast", fontsize=9)
    ax_future.grid(color="#21262d", lw=0.4)

    # ── Stats text box
    ax_stats.axis("off")
    lines = [f"{'SUMMARY':^26}"] + ["─" * 26] + \
            [f"{k:<18} {v:>7}" for k, v in stats.items()]
    ax_stats.text(0.05, 0.95, "\n".join(lines), transform=ax_stats.transAxes,
                  color=WHITE, fontsize=7.5, va="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_BG, edgecolor="#30363d"))

    # Title
    fig.text(0.5, 0.98, f"{'█ ' * 3}  {ticker}  STOCK ANALYSIS DASHBOARD  {'█ ' * 3}",
             ha="center", color=BLUE, fontsize=15, fontweight="bold")

    path = os.path.join(out_dir, f"{ticker}_dashboard.png")
    fig.savefig(path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart] Saved {path}")
    return path
