"""Load saved artifacts and predict the next trading day's closing price."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

try:
    from .data_loader import load_clean
    from .features import build_features, feature_columns
except ImportError:
    from data_loader import load_clean
    from features import build_features, feature_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _load_metadata() -> dict:
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python run.py` first to train."
        )
    return json.loads(meta_path.read_text())


def predict_next_day() -> dict:
    """Return the next-day close prediction using the saved best model."""
    meta = _load_metadata()
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feat_cols = json.loads((MODELS_DIR / "feature_columns.json").read_text())

    data, _ = load_clean()
    feats = build_features(data)

    # Rows usable as model input keep all features (target may be NaN on last row).
    usable = feats[feature_columns(feats)].dropna()
    last_date = usable.index[-1]
    latest_close = float(data.loc[last_date, "Close"])

    # Models predict the next-day return; reconstruct the price from it.
    if meta.get("is_lstm"):
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model

        model = load_model(MODELS_DIR / "model.keras")
        lookback = int(meta["lstm_lookback"])
        window = usable[feat_cols].to_numpy(dtype=float)[-lookback:]
        window_scaled = scaler.transform(window)
        seq = window_scaled.reshape(1, lookback, len(feat_cols))
        pred_ret = float(model.predict(seq, verbose=0).ravel()[0])
    else:
        model = joblib.load(MODELS_DIR / "best_model.pkl")
        x = usable[feat_cols].to_numpy(dtype=float)[-1:]
        x_scaled = scaler.transform(x)
        pred_ret = float(model.predict(x_scaled)[0])

    if meta.get("prediction_mode") == "return_reconstructed":
        pred = latest_close * (1.0 + pred_ret)
    else:
        pred = pred_ret

    direction = "UP" if pred > latest_close else "DOWN"
    return {
        "model": meta["best_model"],
        "as_of_date": str(last_date.date()),
        "latest_close": latest_close,
        "predicted_next_close": pred,
        "expected_change": pred - latest_close,
        "expected_change_pct": (pred - latest_close) / latest_close * 100,
        "direction": direction,
    }


if __name__ == "__main__":
    result = predict_next_day()
    print("Model:", result["model"])
    print("As of:", result["as_of_date"], "| Latest close:", round(result["latest_close"], 2))
    print(
        "Predicted next-day close:",
        round(result["predicted_next_close"], 2),
        f"({result['direction']}, {result['expected_change_pct']:+.2f}%)",
    )
