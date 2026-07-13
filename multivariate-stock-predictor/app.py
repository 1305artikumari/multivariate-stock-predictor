"""Streamlit dashboard for the multivariate stock price predictor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import load_clean
from src.features import build_features
from src.predict import predict_next_day

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

st.set_page_config(page_title="Multivariate Stock Price Prediction", layout="wide")


@st.cache_data
def get_data():
    data, report = load_clean()
    feats = build_features(data)
    return data, report, feats


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


st.title("Multivariate Stock Price Prediction")
st.caption("Machine Learning and Deep Learning - Next-Day Closing Price Forecast")

metadata = load_json(MODELS_DIR / "model_metadata.json")
metrics = load_json(MODELS_DIR / "metrics.json")

if metadata is None:
    st.warning("No trained model found. Run `python run.py` first to train the models.")
    st.stop()

data, report, feats = get_data()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Stock Ticker", value="RELIANCE.NS", disabled=True)
    st.caption("This build uses the locally saved RELIANCE.NS dataset.")
    st.markdown("---")
    st.subheader("Best Model")
    st.success(metadata["best_model"])
    st.write(f"Features used: {metadata['n_features']}")
    st.write(f"Data range: {metadata['date_range'][0]} to {metadata['date_range'][1]}")

# Top metrics
latest_close = float(data["Close"].iloc[-1])
pred = predict_next_day()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close", f"{latest_close:,.2f}", help=f"As of {pred['as_of_date']}")
c2.metric(
    "Predicted Next Close",
    f"{pred['predicted_next_close']:,.2f}",
    delta=f"{pred['expected_change_pct']:+.2f}%",
)
c3.metric("Direction", pred["direction"])
c4.metric("Model", pred["model"])

st.markdown("---")

# Historical price chart
st.subheader("Historical Closing Price")
price_df = data[["Close"]].copy()
price_df["MA20"] = data["Close"].rolling(20).mean()
price_df["MA50"] = data["Close"].rolling(50).mean()
st.line_chart(price_df)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Trading Volume")
    st.bar_chart(data[["Volume"]])

with col_b:
    st.subheader("Latest Technical Indicators")
    last = feats.iloc[-1]
    indicator_cols = ["RSI14", "MACD", "MACDSignal", "MA20", "MA50", "ATR14", "VolumeRatio", "Volatility20"]
    ind = {k: round(float(last[k]), 4) for k in indicator_cols if k in feats.columns}
    st.table(pd.DataFrame(ind.items(), columns=["Indicator", "Value"]))

st.markdown("---")

# Model comparison
st.subheader("Model Evaluation and Comparison")
if metrics is not None:
    comp = pd.DataFrame(metrics["models"]).T
    ordering = ["MAE", "RMSE", "R2", "MAPE", "DirectionAccuracy"]
    comp = comp[[c for c in ordering if c in comp.columns]].sort_values("RMSE")
    st.dataframe(comp.style.format("{:.4f}"), use_container_width=True)

# Plots
st.subheader("Actual vs Predicted (Test Set)")
avp = OUTPUTS_DIR / "actual_vs_predicted.png"
if avp.exists():
    st.image(str(avp), use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    comp_img = OUTPUTS_DIR / "model_comparison_rmse.png"
    if comp_img.exists():
        st.image(str(comp_img), caption="Model RMSE comparison", use_container_width=True)
with col_d:
    fi_img = OUTPUTS_DIR / "feature_importance.png"
    if fi_img.exists():
        st.image(str(fi_img), caption="Feature importance", use_container_width=True)

st.markdown("---")
st.info(
    "Disclaimer: This dashboard is for educational purposes only. Stock markets are "
    "highly uncertain. A low prediction error does not guarantee profitable trading. "
    "This is not financial advice."
)
