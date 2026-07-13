"""Regression metrics and comparison / prediction plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_close: np.ndarray | None = None,
) -> dict:
    """MAE, RMSE, R2, MAPE and (optionally) directional accuracy.

    Directional accuracy compares the predicted vs actual movement relative to
    the current-day close (`prev_close`), i.e. whether next-day up/down is right.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    mask = y_true != 0
    mape = float(np.mean(np.abs(error[mask] / y_true[mask])) * 100) if mask.any() else float("nan")

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    if prev_close is not None:
        prev_close = np.asarray(prev_close, dtype=float)
        actual_dir = np.sign(y_true - prev_close)
        pred_dir = np.sign(y_pred - prev_close)
        metrics["DirectionAccuracy"] = float(np.mean(actual_dir == pred_dir) * 100)

    return metrics


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from {model_name: metrics}."""
    table = pd.DataFrame(results).T
    ordering = ["MAE", "RMSE", "R2", "MAPE", "DirectionAccuracy"]
    cols = [c for c in ordering if c in table.columns]
    return table[cols].sort_values("RMSE")


def plot_comparison(table: pd.DataFrame, outputs_dir: Path = OUTPUTS_DIR) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    table["RMSE"].plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Model Comparison - Test RMSE (lower is better)")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(outputs_dir / "model_comparison_rmse.png", dpi=120)
    plt.close(fig)


def plot_predictions(
    dates,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, y_true, label="Actual", linewidth=1.2)
    ax.plot(dates, y_pred, label="Predicted", linewidth=1.2, alpha=0.8)
    ax.set_title(f"Actual vs Predicted Close - {model_name} (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs_dir / "actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, residuals, color="darkorange", linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Prediction Residuals - {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted - Actual")
    fig.tight_layout()
    fig.savefig(outputs_dir / "residuals.png", dpi=120)
    plt.close(fig)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    model_name: str,
    outputs_dir: Path = OUTPUTS_DIR,
    top_n: int = 20,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(names)), vals, color="seagreen")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(outputs_dir / "feature_importance.png", dpi=120)
    plt.close(fig)
