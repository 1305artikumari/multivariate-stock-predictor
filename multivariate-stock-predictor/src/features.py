"""Multivariate feature engineering.

Every feature for a given trading day uses only information available on or
before that day (no look-ahead). The target is the NEXT day's closing price,
created by shifting Close backward by one row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"

TARGET_COL = "Target"


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical, lag, rolling, and calendar features + target."""
    out = df.copy()

    # Price-derived returns and ranges
    out["Return"] = out["Close"].pct_change()
    out["LogReturn"] = np.log(out["Close"] / out["Close"].shift(1))
    out["HighLowRange"] = (out["High"] - out["Low"]) / out["Close"]
    out["OpenCloseChange"] = (out["Close"] - out["Open"]) / out["Open"]

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        out[f"CloseLag{lag}"] = out["Close"].shift(lag)
        out[f"ReturnLag{lag}"] = out["Return"].shift(lag)
        out[f"VolumeLag{lag}"] = out["Volume"].shift(lag)

    # Moving averages and rolling stats
    for window in [5, 10, 20, 50]:
        out[f"MA{window}"] = out["Close"].rolling(window).mean()
        out[f"RollingStd{window}"] = out["Close"].rolling(window).std()
    out["Volatility20"] = out["Return"].rolling(20).std()

    # RSI
    out["RSI14"] = _rsi(out["Close"], 14)

    # MACD (12/26 EMA, 9 signal)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACDSignal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACDHist"] = out["MACD"] - out["MACDSignal"]

    # Bollinger Bands (20-day, 2 std)
    mid = out["Close"].rolling(20).mean()
    std20 = out["Close"].rolling(20).std()
    out["BollingerUpper"] = mid + 2 * std20
    out["BollingerLower"] = mid - 2 * std20
    out["BollingerWidth"] = (out["BollingerUpper"] - out["BollingerLower"]) / mid

    # ATR
    out["ATR14"] = _atr(out, 14)

    # Volume features
    out["VolumeMA20"] = out["Volume"].rolling(20).mean()
    out["VolumeRatio"] = out["Volume"] / out["VolumeMA20"]

    # Calendar features
    out["DayOfWeek"] = out.index.dayofweek
    out["Month"] = out.index.month

    # Target = next trading day's close
    out[TARGET_COL] = out["Close"].shift(-1)

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything except the target."""
    return [c for c in df.columns if c != TARGET_COL]


def make_dataset(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Build features, drop rows with NaNs (from lags/rolling/target), save."""
    feats = build_features(df)
    feats = feats.dropna()
    if save:
        PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(PROCESSED_CSV)
    return feats


if __name__ == "__main__":
    try:
        from data_loader import load_clean
    except ModuleNotFoundError:
        from src.data_loader import load_clean

    data, _ = load_clean()
    dataset = make_dataset(data)
    print("Feature matrix shape:", dataset.shape)
    print("Feature columns:", feature_columns(dataset))
    print("\nHead:\n", dataset.head())
