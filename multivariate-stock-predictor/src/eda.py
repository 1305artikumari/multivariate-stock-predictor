"""Exploratory data analysis: generate charts and summary statistics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def run_eda(df: pd.DataFrame, outputs_dir: Path = OUTPUTS_DIR) -> dict:
    """Create EDA plots and return summary statistics."""
    outputs_dir.mkdir(parents=True, exist_ok=True)

    returns = df["Close"].pct_change()
    ma20 = df["Close"].rolling(20).mean()
    ma50 = df["Close"].rolling(50).mean()
    volatility = returns.rolling(20).std()

    # Closing price with moving averages
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Close", linewidth=1)
    ax.plot(df.index, ma20, label="MA 20", linewidth=1)
    ax.plot(df.index, ma50, label="MA 50", linewidth=1)
    ax.set_title("Closing Price with 20/50-Day Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs_dir / "eda_price_trend.png", dpi=120)
    plt.close(fig)

    # Trading volume
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df.index, df["Volume"], width=1.0)
    ax.set_title("Trading Volume Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    fig.tight_layout()
    fig.savefig(outputs_dir / "eda_volume.png", dpi=120)
    plt.close(fig)

    # Daily return + distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(df.index, returns, linewidth=0.7)
    axes[0].set_title("Daily Percentage Return")
    axes[0].set_xlabel("Date")
    axes[1].hist(returns.dropna(), bins=80)
    axes[1].set_title("Return Distribution")
    fig.tight_layout()
    fig.savefig(outputs_dir / "eda_returns.png", dpi=120)
    plt.close(fig)

    # Rolling volatility
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, volatility, color="crimson", linewidth=1)
    ax.set_title("20-Day Rolling Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Std of daily returns")
    fig.tight_layout()
    fig.savefig(outputs_dir / "eda_volatility.png", dpi=120)
    plt.close(fig)

    # Correlation matrix
    corr = df[["Open", "High", "Low", "Close", "Volume"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("OHLCV Correlation Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outputs_dir / "eda_correlation.png", dpi=120)
    plt.close(fig)

    summary = df[["Open", "High", "Low", "Close", "Volume"]].describe().to_dict()
    return summary


if __name__ == "__main__":
    try:
        from data_loader import load_clean
    except ModuleNotFoundError:
        from src.data_loader import load_clean

    data, _ = load_clean()
    stats = run_eda(data)
    print("EDA plots written to", OUTPUTS_DIR)
