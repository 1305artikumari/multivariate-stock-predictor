"""MarketPulse - colorful, interactive Streamlit dashboard for stock prediction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data_loader import load_clean
from src.features import build_features
from src.predict import predict_next_day

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- palette (aurora)
GREEN = "#2dd4bf"   # teal-green for gains
RED = "#fb7185"     # rose for losses
CYAN = "#a78bfa"    # violet accent (primary highlight)
BLUE = "#6366f1"    # indigo
PURPLE = "#c084fc"  # light violet
AMBER = "#fbbf24"   # gold
PANEL_BG = "#14122b"
PANEL_BORDER = "#2a2750"
TRANSPARENT = "rgba(0,0,0,0)"
PLOT_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------- styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(1200px 700px at 15% -10%, #241a44 0%, #140f2e 45%, #0b0a1a 100%);
    }}
    .block-container {{ padding-top: 1.4rem; }}
    section[data-testid="stSidebar"] {{ background: #0d0b1e; border-right: 1px solid #2a2750; }}

    .hero {{
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 20px;
        background: linear-gradient(120deg, rgba(99,102,241,0.16), rgba(139,92,246,0.14), rgba(236,72,153,0.10));
        border: 1px solid #34305e;
        box-shadow: 0 10px 30px rgba(99,102,241,0.18);
    }}
    .brand {{ display: flex; align-items: center; gap: 14px; }}
    .brand-logo {{
        font-size: 2.2rem; line-height: 1;
        width: 56px; height: 56px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, {BLUE}, {CYAN});
    }}
    .main-title {{
        font-size: 2.4rem; font-weight: 900; letter-spacing: -0.5px;
        color: #f8fafc; margin: 0;
    }}
    .main-title span {{ color: {CYAN}; }}
    .subtitle {{ color: #8b97a7; font-size: 0.98rem; margin-top: 4px; }}
    .pill {{
        display: inline-block; margin-top: 12px; margin-right: 8px;
        padding: 5px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 600;
        color: #ddd6fe; background: #1c1940; border: 1px solid #34305e;
    }}

    .panel {{
        border-radius: 12px; padding: 16px 18px;
        background: #14122b; border: 1px solid #2a2750;
        height: 100%;
    }}
    .panel .label {{ color: #8b97a7; font-size: 0.82rem; font-weight: 600; margin: 0; }}
    .panel .value {{ font-size: 2.0rem; font-weight: 800; color: #f8fafc; margin-top: 6px; line-height: 1.1; }}
    .panel .sub {{ font-size: 0.82rem; margin-top: 4px; }}
    .accent-bar {{ height: 4px; width: 46px; border-radius: 4px; margin-bottom: 12px; }}
    .up {{ color: {GREEN}; }}
    .down {{ color: {RED}; }}

    .footer {{
        text-align: center; color: #5b6675; font-size: 0.84rem;
        padding: 16px 0 4px 0; margin-top: 10px;
        border-top: 1px solid #2a2750;
    }}
    .footer b {{ color: {CYAN}; }}

    /* ---------- Sidebar polish ---------- */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #f8fafc; font-weight: 800;
    }}
    section[data-testid="stSidebar"] .stHeading {{
        border-left: 4px solid {CYAN}; padding-left: 10px;
    }}
    section[data-testid="stSidebar"] hr {{ border-color: #2a2750; }}
    /* multiselect tags -> gradient */
    span[data-baseweb="tag"] {{
        background: linear-gradient(135deg, {BLUE}, {CYAN}) !important;
        border: none !important; color: #041016 !important; font-weight: 700 !important;
    }}
    span[data-baseweb="tag"] span {{ color: #041016 !important; }}
    /* text inputs / selects */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background: #14122b !important; border-color: #34305e !important;
    }}

    /* ---------- Tabs: bold, colorful, highlighted ---------- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; border-bottom: none; }}
    .stTabs [data-baseweb="tab"] {{
        background: #14122b; border: 1px solid #2a2750; border-radius: 12px;
        padding: 10px 20px; color: #8b97a7; font-weight: 700; transition: all 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"] p {{ font-size: 0.95rem; font-weight: 700; margin: 0; }}
    .stTabs [data-baseweb="tab"]:hover {{ color: #e7e5f4; border-color: #4b3f7a; }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {BLUE}, {CYAN});
        border-color: transparent;
        box-shadow: 0 8px 22px rgba(34,211,238,0.30);
        transform: translateY(-1px);
    }}
    .stTabs [aria-selected="true"] p {{ color: #ffffff !important; }}
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{ background: transparent; display: none; }}

    /* ---------- Metric cards (indicator snapshot) ---------- */
    [data-testid="stMetric"] {{
        background: #14122b; border: 1px solid #2a2750; border-radius: 12px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricValue"] {{ color: #f8fafc; font-weight: 800; }}
    [data-testid="stMetricLabel"] p {{ color: {CYAN}; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    data, report = load_clean()
    feats = build_features(data)
    return data, report, feats


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def styled_layout(fig, height=420, title=None):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        title=title,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font=dict(color="#cbd5e1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2a2750", zerolinecolor="#2a2750")
    fig.update_yaxes(gridcolor="#2a2750", zerolinecolor="#2a2750")
    return fig


def panel(label, value, sub_html="", accent=CYAN):
    return (
        f'<div class="panel"><div class="accent-bar" style="background:{accent}"></div>'
        f'<p class="label">{label}</p>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub_html}</div></div>'
    )


# ---------------------------------------------------------------- header
st.markdown(
    """
    <div class="hero">
      <div class="brand">
        <div class="brand-logo">📈</div>
        <div>
          <p class="main-title">Stock Price <span>Predictor</span></p>
          <p class="subtitle">Intelligent Stock Price Forecasting · Next-Day Closing Price · ML &amp; Deep Learning</p>
        </div>
      </div>
      <span class="pill">🇮🇳 RELIANCE.NS</span>
      <span class="pill">🤖 XGBoost · Random Forest · LSTM</span>
      <span class="pill">📊 45 Engineered Features</span>
      <span class="pill">⏱️ Time-Series Validated</span>
    </div>
    """,
    unsafe_allow_html=True,
)

metadata = load_json(MODELS_DIR / "model_metadata.json")
metrics = load_json(MODELS_DIR / "metrics.json")

if metadata is None:
    st.warning("No trained model found. Run `python run.py` first to train the models.")
    st.stop()

data, report, feats = get_data()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    ticker = st.text_input("Stock Ticker", value="RELIANCE.NS", disabled=True)
    st.caption("This build uses the locally saved RELIANCE.NS dataset.")

    st.markdown("---")
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    default_start = data.index[max(0, len(data) - 365)].date()
    date_range = st.slider(
        "📅 Date range",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM-DD",
    )

    chart_style = st.radio("📊 Price chart style", ["Candlestick", "Line"], horizontal=True)
    show_ma = st.multiselect("Moving averages", ["MA20", "MA50"], default=["MA20", "MA50"])

    st.markdown("---")
    st.subheader("🏆 Best Model")
    st.success(metadata["best_model"])
    st.write(f"Features used: **{metadata['n_features']}**")
    st.caption(f"Data: {metadata['date_range'][0]} → {metadata['date_range'][1]}")

# Filter data by selected date range
start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
mask = (data.index >= start) & (data.index <= end)
d = data.loc[mask]

# ---------------------------------------------------------------- metric panels
latest_close = float(data["Close"].iloc[-1])
pred = predict_next_day()
up = pred["direction"] == "UP"
dir_color = GREEN if up else RED

# Best-model test accuracy (R2) for the summary card
best_r2 = None
if metrics is not None:
    best_stats = metrics.get("models", {}).get(metadata["best_model"], {})
    best_r2 = best_stats.get("R2")

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    panel("LATEST CLOSE", f"₹{latest_close:,.2f}",
          f'<span style="color:#8b97a7">as of {pred["as_of_date"]}</span>', BLUE),
    unsafe_allow_html=True,
)
c2.markdown(
    panel("PREDICTED NEXT CLOSE", f"₹{pred['predicted_next_close']:,.2f}",
          f'<span class="{"up" if up else "down"}">{pred["expected_change_pct"]:+.2f}% expected</span>', dir_color),
    unsafe_allow_html=True,
)
acc_value = f"{best_r2 * 100:.1f}%" if best_r2 is not None else "N/A"
c3.markdown(
    panel("MODEL ACCURACY (R²)", acc_value,
          '<span style="color:#8b97a7">on unseen test data</span>', GREEN),
    unsafe_allow_html=True,
)
c4.markdown(
    panel("MODEL USED", pred["model"],
          '<span style="color:#8b97a7">best by test RMSE</span>', PURPLE),
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs
tab_price, tab_tech, tab_perf, tab_pred = st.tabs(
    ["📈 Price & Volume", "🔬 Technical Indicators", "🧠 Model Performance", "🎯 Prediction"]
)

with tab_price:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.72, 0.28], subplot_titles=("Price", "Volume"),
    )
    if chart_style == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                name="Price", increasing_line_color=GREEN, decreasing_line_color=RED,
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=d.index, y=d["Close"], name="Close", line=dict(color=CYAN, width=2)),
            row=1, col=1,
        )

    ma_colors = {"MA20": AMBER, "MA50": PURPLE}
    for ma in show_ma:
        series = d["Close"].rolling(int(ma[2:])).mean()
        fig.add_trace(
            go.Scatter(x=d.index, y=series, name=ma, line=dict(color=ma_colors[ma], width=1.5)),
            row=1, col=1,
        )

    vol_colors = [GREEN if c >= o else RED for c, o in zip(d["Close"], d["Open"])]
    fig.add_trace(
        go.Bar(x=d.index, y=d["Volume"], name="Volume", marker_color=vol_colors, showlegend=False),
        row=2, col=1,
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(styled_layout(fig, height=560), width="stretch")

with tab_tech:
    f = feats.loc[(feats.index >= start) & (feats.index <= end)]

    col1, col2 = st.columns(2)
    with col1:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=f.index, y=f["RSI14"], name="RSI(14)", line=dict(color=CYAN, width=1.6)))
        rsi_fig.add_hrect(y0=70, y1=100, fillcolor=RED, opacity=0.12, line_width=0)
        rsi_fig.add_hrect(y0=0, y1=30, fillcolor=GREEN, opacity=0.12, line_width=0)
        rsi_fig.add_hline(y=70, line=dict(color=RED, dash="dash"))
        rsi_fig.add_hline(y=30, line=dict(color=GREEN, dash="dash"))
        st.plotly_chart(styled_layout(rsi_fig, height=340, title="RSI (Overbought / Oversold)"), width="stretch")

    with col2:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Bar(x=f.index, y=f["MACDHist"], name="Histogram",
                                  marker_color=[GREEN if v >= 0 else RED for v in f["MACDHist"]]))
        macd_fig.add_trace(go.Scatter(x=f.index, y=f["MACD"], name="MACD", line=dict(color=CYAN, width=1.6)))
        macd_fig.add_trace(go.Scatter(x=f.index, y=f["MACDSignal"], name="Signal", line=dict(color=AMBER, width=1.6)))
        st.plotly_chart(styled_layout(macd_fig, height=340, title="MACD"), width="stretch")

    bb_fig = go.Figure()
    bb_fig.add_trace(go.Scatter(x=f.index, y=f["BollingerUpper"], name="Upper Band",
                                line=dict(color=PURPLE, width=1, dash="dot")))
    bb_fig.add_trace(go.Scatter(x=f.index, y=f["BollingerLower"], name="Lower Band",
                                line=dict(color=PURPLE, width=1, dash="dot"), fill="tonexty",
                                fillcolor="rgba(124,92,252,0.12)"))
    bb_fig.add_trace(go.Scatter(x=f.index, y=f["Close"], name="Close", line=dict(color=CYAN, width=1.8)))
    st.plotly_chart(styled_layout(bb_fig, height=380, title="Bollinger Bands"), width="stretch")

    st.subheader("Latest Indicator Snapshot")
    last = feats.iloc[-1]
    snap_cols = ["RSI14", "MACD", "MACDSignal", "MA20", "MA50", "ATR14", "VolumeRatio", "Volatility20"]
    cells = st.columns(4)
    for i, name in enumerate([c for c in snap_cols if c in feats.columns]):
        cells[i % 4].metric(name, f"{float(last[name]):.3f}")

with tab_perf:
    if metrics is not None:
        comp = pd.DataFrame(metrics["models"]).T
        ordering = ["MAE", "RMSE", "R2", "MAPE", "DirectionAccuracy"]
        comp = comp[[c for c in ordering if c in comp.columns]].sort_values("RMSE")

        colL, colR = st.columns([1, 1])
        with colL:
            rmse_fig = go.Figure()
            colors = [GREEN if m == metadata["best_model"] else BLUE for m in comp.index]
            rmse_fig.add_trace(go.Bar(x=comp.index, y=comp["RMSE"], marker_color=colors,
                                      text=comp["RMSE"].round(2), textposition="outside"))
            st.plotly_chart(styled_layout(rmse_fig, height=380, title="Test RMSE (lower is better)"), width="stretch")
        with colR:
            if "DirectionAccuracy" in comp.columns:
                da = comp["DirectionAccuracy"].dropna()
                da_fig = go.Figure()
                da_fig.add_trace(go.Bar(x=da.index, y=da, marker_color=CYAN,
                                        text=da.round(1), textposition="outside"))
                da_fig.add_hline(y=50, line=dict(color=AMBER, dash="dash"), annotation_text="coin-flip 50%")
                st.plotly_chart(styled_layout(da_fig, height=380, title="Directional Accuracy (%)"), width="stretch")

        st.subheader("Full Metrics Table")
        st.dataframe(
            comp.style.format("{:.4f}").background_gradient(cmap="RdYlGn_r", subset=["RMSE", "MAE"]),
            width="stretch",
        )

    fi_img = OUTPUTS_DIR / "feature_importance.png"
    if fi_img.exists():
        st.subheader("Feature Importance")
        st.image(str(fi_img), width="stretch")

with tab_pred:
    colG, colT = st.columns([1, 1])
    with colG:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred["predicted_next_close"],
            delta={"reference": latest_close, "increasing": {"color": GREEN}, "decreasing": {"color": RED}},
            title={"text": "Predicted Next-Day Close (₹)"},
            gauge={
                "axis": {"range": [latest_close * 0.9, latest_close * 1.1]},
                "bar": {"color": dir_color},
                "steps": [
                    {"range": [latest_close * 0.9, latest_close], "color": "rgba(234,57,67,0.22)"},
                    {"range": [latest_close, latest_close * 1.1], "color": "rgba(22,199,132,0.22)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": latest_close},
            },
        ))
        st.plotly_chart(styled_layout(gauge, height=380), width="stretch")

    with colT:
        st.markdown("### Prediction Summary")
        st.write(f"**Model:** {pred['model']}")
        st.write(f"**As of:** {pred['as_of_date']}")
        st.write(f"**Latest close:** ₹{latest_close:,.2f}")
        st.write(f"**Predicted next close:** ₹{pred['predicted_next_close']:,.2f}")
        st.write(f"**Expected change:** {pred['expected_change_pct']:+.2f}%")
        if up:
            st.success("Model expects the price to move UP ▲")
        else:
            st.error("Model expects the price to move DOWN ▼")

    avp = OUTPUTS_DIR / "actual_vs_predicted.png"
    if avp.exists():
        st.subheader("Actual vs Predicted (Test Set)")
        st.image(str(avp), width="stretch")

st.info(
    "Disclaimer: This dashboard is for educational purposes only. Stock markets are "
    "highly uncertain. A low prediction error does not guarantee profitable trading. "
    "This is not financial advice."
)

st.markdown(
    """
    <div class="footer">
      <b>Stock Price Predictor</b> — built with Python, Scikit-learn, XGBoost, TensorFlow &amp; Streamlit<br>
      Multivariate Stock Price Prediction · Educational Data Science Project
    </div>
    """,
    unsafe_allow_html=True,
)
