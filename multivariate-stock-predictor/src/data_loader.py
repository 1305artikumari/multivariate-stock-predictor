"""Load, clean, and validate historical OHLCV stock data.

The local CSV is produced by yfinance and carries a 3-row header:

    Price,Close,High,Low,Open,Volume
    Ticker,RELIANCE.NS,...
    Date,,,,,
    2015-01-01,<close>,<high>,<low>,<open>,<volume>

This module normalises that into a clean, chronologically sorted DataFrame
indexed by trading Date with numeric OHLCV columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "reliance_stock_data.csv"

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_raw(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Read the yfinance multi-header CSV into a clean OHLCV frame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    # Row 0 = feature names, rows 1-2 = ticker/date metadata, data starts at row 3.
    df = pd.read_csv(
        csv_path,
        skiprows=3,
        header=None,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


def validate(df: pd.DataFrame) -> dict:
    """Run data-quality checks and return a report dictionary."""
    report: dict = {}
    report["rows"] = int(len(df))
    report["start_date"] = str(df.index.min().date()) if len(df) else None
    report["end_date"] = str(df.index.max().date()) if len(df) else None
    report["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}
    report["duplicate_dates"] = int(df.index.duplicated().sum())

    price_cols = ["Open", "High", "Low", "Close"]
    report["non_positive_prices"] = {
        c: int((df[c] <= 0).sum()) for c in price_cols
    }
    report["negative_volume"] = int((df["Volume"] < 0).sum())

    daily_change = df["Close"].pct_change().abs()
    report["max_daily_change_pct"] = float(daily_change.max() * 100) if len(df) else None
    report["extreme_moves_gt_20pct"] = int((daily_change > 0.20).sum())
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate dates and any rows with missing OHLCV, keep order."""
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=OHLCV_COLUMNS)
    return df.sort_index()


def load_clean(csv_path: str | Path = DEFAULT_CSV) -> tuple[pd.DataFrame, dict]:
    """Convenience: load, validate, then clean. Returns (df, report)."""
    raw = load_raw(csv_path)
    report = validate(raw)
    cleaned = clean(raw)
    report["rows_after_clean"] = int(len(cleaned))
    return cleaned, report


if __name__ == "__main__":
    data, rep = load_clean()
    print("Data loaded:", rep["rows"], "rows")
    print("Date range:", rep["start_date"], "->", rep["end_date"])
    print("\nHead:\n", data.head())
    print("\nTail:\n", data.tail())
    print("\nValidation report:")
    for k, v in rep.items():
        print(f"  {k}: {v}")
