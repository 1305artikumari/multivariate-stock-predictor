# Multivariate Stock Price Prediction Using Machine Learning and Deep Learning

An end-to-end AI/ML project that predicts the **next trading day's closing price** of a stock
(RELIANCE.NS by default) using multiple market and technical features rather than just the
closing price.

## What it does

1. Loads and cleans historical daily OHLCV data (local `reliance_stock_data.csv`).
2. Performs exploratory data analysis (EDA) with charts.
3. Engineers multivariate features: returns, lags, moving averages, RSI, MACD,
   Bollinger Bands, ATR, volume ratios, and calendar features.
4. Creates the target `Close(t+1)` (next-day close).
5. Splits data chronologically (70/15/15) - no shuffling.
6. Trains and compares models: Naive Baseline, Linear Regression, Random Forest,
   XGBoost, and (optionally) an LSTM.
7. Evaluates with MAE, RMSE, R2, MAPE, and directional accuracy.
8. Saves the best model, scaler, feature list, and metadata.
9. Serves an interactive Streamlit dashboard.

## Project structure

```
multivariate-stock-predictor/
├── data/
│   ├── raw/reliance_stock_data.csv
│   └── processed/features.csv
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── outputs/
├── app.py
├── run.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

On a corporate network with SSL interception, add:

```bash
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

## Run the full pipeline

```bash
python run.py
```

This generates features, trains all models, evaluates them, writes plots to `outputs/`,
and saves artifacts to `models/`.

## Launch the dashboard

```bash
streamlit run app.py
```

## Predict the next day from the command line

```bash
python -m src.predict
```

## Disclaimer

This project is for **educational purposes only**. Stock markets are highly uncertain and
influenced by news, economic events, and investor behaviour. A low prediction error does
**not** guarantee profitable trading. This is not financial advice.
