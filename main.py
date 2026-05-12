"""
main.py
─────────────────────────────────────────────────────────────────
Stock Market Analysis System
  • Data generation / loading
  • Preprocessing & feature engineering
  • Statistical analysis
  • Technical indicator charting
  • Linear Regression prediction
  • 30-day price forecast
  • Dashboard summary
─────────────────────────────────────────────────────────────────
Usage:
    python main.py                   # analyse all 3 default tickers
    python main.py --tickers AAPL    # single ticker
    python main.py --tickers AAPL TSLA GOOGL
"""

import os
import sys
import argparse
import pandas as pd

# ── Project modules ──────────────────────────────────────────────────────────
from generate_data   import generate_stock_data
from preprocessing   import load_stock_data, clean_data, add_technical_indicators, get_summary_statistics
from ml_model        import train_linear_regression, predict_future
from visualization   import (plot_price_and_ma, plot_volume, plot_rsi, plot_macd,
                              plot_return_distribution, plot_ml_predictions, plot_dashboard)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "output"
TICKERS    = ["AAPL", "GOOGL", "TSLA"]


def banner(msg: str):
    line = "═" * 60
    print(f"\n{line}\n  {msg}\n{line}")


def run_analysis(ticker: str):
    banner(f"Analysing  {ticker}")

    # ── 1. Data ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        print(f"  [Data] Generating synthetic data for {ticker} …")
        df_raw = generate_stock_data(ticker, days=500)
        os.makedirs(DATA_DIR, exist_ok=True)
        df_raw.to_csv(csv_path, index=False)
    else:
        print(f"  [Data] Loading {csv_path}")

    df = load_stock_data(csv_path)

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    print("\n[Step 2] Preprocessing …")
    df = clean_data(df)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Final shape: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── 3. Analysis ──────────────────────────────────────────────────────────
    print("\n[Step 3] Statistical Analysis …")
    stats = get_summary_statistics(df)
    print("  ┌─────────────────────────────────┐")
    for k, v in stats.items():
        print(f"  │  {k:<24} {str(v):>7}  │")
    print("  └─────────────────────────────────┘")

    # ── 4. Machine Learning ──────────────────────────────────────────────────
    print("\n[Step 4] Machine Learning …")
    model, scaler, test_df, metrics = train_linear_regression(df)
    future_df = predict_future(model, scaler, df, days=30)
    print(f"  [Forecast] First predicted price: ${future_df['Predicted_Price'].iloc[0]}")
    print(f"  [Forecast] Last  predicted price: ${future_df['Predicted_Price'].iloc[-1]}")

    # ── 5. Visualisation ─────────────────────────────────────────────────────
    print("\n[Step 5] Generating charts …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_price_and_ma(df, ticker, OUTPUT_DIR)
    plot_volume(df, ticker, OUTPUT_DIR)
    plot_rsi(df, ticker, OUTPUT_DIR)
    plot_macd(df, ticker, OUTPUT_DIR)
    plot_return_distribution(df, ticker, OUTPUT_DIR)
    plot_ml_predictions(test_df, future_df, ticker, metrics, OUTPUT_DIR)
    plot_dashboard(df, test_df, future_df, stats, ticker, metrics, OUTPUT_DIR)

    print(f"\n  ✓ All charts saved to ./{OUTPUT_DIR}/")
    return stats, metrics


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stock Market Analysis System")
    parser.add_argument("--tickers", nargs="+", default=TICKERS,
                        help="Space-separated list of tickers (default: AAPL GOOGL TSLA)")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║         STOCK MARKET ANALYSIS SYSTEM  ·  Python DS          ║
╚══════════════════════════════════════════════════════════════╝
 Technologies: Pandas · NumPy · Matplotlib · Scikit-Learn
 Methods    : Data Cleaning · Technical Indicators · Linear Regression
""")

    all_results = {}
    for ticker in args.tickers:
        stats, metrics = run_analysis(ticker.upper())
        all_results[ticker] = {"stats": stats, "metrics": metrics}

    # ── Final comparison table ────────────────────────────────────────────
    banner("COMPARISON SUMMARY")
    header = f"{'Metric':<28}" + "".join(f"{t:>12}" for t in all_results)
    print(header)
    print("─" * len(header))

    compare_keys = ["Total Return (%)", "Sharpe Ratio", "Volatility (std)"]
    for k in compare_keys:
        row = f"{k:<28}"
        for t in all_results:
            val = all_results[t]["stats"].get(k, "N/A")
            row += f"{str(val):>12}"
        print(row)

    print("\nML Metrics")
    print("─" * len(header))
    for k in ["RMSE", "MAE", "R²"]:
        row = f"{k:<28}"
        for t in all_results:
            val = all_results[t]["metrics"].get(k, "N/A")
            row += f"{str(val):>12}"
        print(row)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✓  Analysis complete!  Charts saved to ./{OUTPUT_DIR}/       
║                                                              ║
║  NOTE: Predictions use Linear Regression on lag features.   ║
║  These are NOT financial advice. Markets are complex.        ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
