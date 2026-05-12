# 📈 StockLens — Stock Market Analysis System

> Real-time stock analysis with technical indicators, ML predictions & 30-day price forecasting.

🔗 **Live Demo:** [https://stocklen.onrender.com](https://stocklen.onrender.com)

---

## 🖼️ Overview

StockLens is a full-stack web application that fetches real stock market data from Yahoo Finance and performs deep technical and machine learning analysis — all visualized in a sleek dark-themed dashboard.

---

## ✨ Features

- 📡 **Real Market Data** — Pulls 2 years of OHLCV data via `yfinance`
- 📊 **6 Interactive Charts** — Price/MA, Volume, RSI, MACD, Returns, ML Predictions
- 🤖 **ML Price Prediction** — Linear Regression with lag features & technical indicators
- 🔮 **30-Day Forecast** — Future price prediction with ±2% confidence band
- 📉 **Technical Indicators** — SMA, EMA, MACD, RSI, Bollinger Bands, Volatility
- 📋 **Stats Dashboard** — Sharpe Ratio, Total Return, Volatility, and more
- 🔁 **Smart Fallback** — Uses synthetic GBM data if Yahoo Finance is unavailable

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Gunicorn |
| Data | yfinance, Pandas, NumPy |
| ML | Scikit-learn (Linear Regression, MinMaxScaler) |
| Visualization | Matplotlib |
| Frontend | HTML, CSS, Vanilla JS |
| Hosting | Render (free tier) |

---

## 📁 Project Structure

```
stock_app/
├── app.py                  # Flask web server & chart rendering
├── fetch_data.py           # Real data fetching via yfinance (+ fallback)
├── generate_data.py        # Synthetic GBM data generator (fallback)
├── preprocessing.py        # Data cleaning & technical indicators
├── ml_model.py             # Linear Regression training & forecasting
├── visualization.py        # Matplotlib chart functions (CLI use)
├── main.py                 # Standalone CLI analysis script
├── requirements.txt        # Python dependencies
├── Procfile                # Render/Heroku start command
├── render.yaml             # Render deployment config
└── templates/
    └── index.html          # Frontend UI
```

---

## 📦 Supported Tickers

`AAPL` · `GOOGL` · `TSLA` · `AMZN` · `MSFT` · `NVDA` · `META` · `NFLX`

---

## 🚀 Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/MustafijurRahman02/stocklens.git
cd stocklens
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## ☁️ Deploy to Render (Free)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → **New + → Web Service**
3. Connect your GitHub repo
4. Render auto-detects settings from `render.yaml`
5. Click **Deploy** — your app goes live in ~3 minutes

---

## 🧠 How It Works

### Data Pipeline
```
Yahoo Finance (yfinance)
        ↓
2 years of real OHLCV data
        ↓
preprocessing.py → Clean + Technical Indicators
        ↓
ml_model.py → Linear Regression (lag features)
        ↓
Flask → Base64 charts → Browser
```

### Technical Indicators Computed
| Indicator | Description |
|---|---|
| SMA 20 / 50 | Simple Moving Average |
| EMA 12 / 26 | Exponential Moving Average |
| MACD | Momentum oscillator |
| RSI (14) | Relative Strength Index |
| Bollinger Bands | Volatility bands (20-day, 2σ) |
| Daily Return | % change per day |
| Cumulative Return | Total return over period |
| Volatility | 20-day rolling std of returns |

### ML Model
- **Algorithm:** Linear Regression
- **Features:** Lag-1 to Lag-5 of Close price + SMA20, SMA50, RSI, MACD, Volume
- **Scaler:** MinMaxScaler
- **Split:** 80% train / 20% test
- **Metrics:** RMSE, MAE, R²

---

## 📊 Sample Results (AAPL)

| Metric | Value |
|---|---|
| R² Score | ~0.97 |
| RMSE | ~1.49 |
| MAE | ~1.26 |

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**.
> Predictions are based on Linear Regression and historical patterns.
> **This is NOT financial advice.** Do not use this for real investment decisions.

---

## 👤 Author

**Mustafijur Rahman**
GitHub: [@MustafijurRahman02](https://github.com/MustafijurRahman02)

---

## 📄 License

MIT License — free to use, modify, and distribute.
