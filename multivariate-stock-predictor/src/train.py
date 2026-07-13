"""Train, compare, and persist models for next-day close prediction.

Models: Naive Baseline, Linear Regression, Random Forest, XGBoost, and an
optional LSTM (only if TensorFlow is importable). Data is split strictly
chronologically (70/15/15) with the scaler fit on the training split only.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from .features import TARGET_COL, feature_columns, make_dataset
    from . import evaluate as ev
except ImportError:  # allow running as a script
    from features import TARGET_COL, feature_columns, make_dataset
    import evaluate as ev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

LSTM_LOOKBACK = 30
RANDOM_STATE = 42


def chronological_split(n: int, train_frac=0.70, val_frac=0.15):
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return train_end, val_end


def _tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def _train_lstm(X_train, y_train, X_val, y_val, X_test, n_features):
    """Build/train an LSTM on rolling sequences (targets are next-day returns).

    Returns (model, predicted_returns) aligned to the test rows.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(RANDOM_STATE)

    def to_sequences(X, y=None):
        seqs, targets = [], []
        for i in range(LSTM_LOOKBACK, len(X)):
            seqs.append(X[i - LSTM_LOOKBACK : i])
            if y is not None:
                targets.append(y[i])
        seqs = np.array(seqs)
        return (seqs, np.array(targets)) if y is not None else seqs

    Xtr_seq, ytr_seq = to_sequences(X_train, y_train)
    Xval_seq, yval_seq = to_sequences(X_val, y_val)

    model = models.Sequential(
        [
            layers.Input(shape=(LSTM_LOOKBACK, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="huber")
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    model.fit(
        Xtr_seq,
        ytr_seq,
        validation_data=(Xval_seq, yval_seq),
        epochs=60,
        batch_size=32,
        verbose=0,
        callbacks=[early],
    )

    # For test predictions we need the LOOKBACK rows preceding the test window,
    # so prepend the tail of validation to the test features.
    combined = np.vstack([X_val[-LSTM_LOOKBACK:], X_test])
    Xtest_seq = to_sequences(combined)
    preds = model.predict(Xtest_seq, verbose=0).ravel()
    return model, preds


def train_all(save: bool = True) -> dict:
    try:
        from .data_loader import load_clean
    except ImportError:
        from data_loader import load_clean

    data, report = load_clean()
    dataset = make_dataset(data)

    feat_cols = feature_columns(dataset)
    X = dataset[feat_cols].to_numpy(dtype=float)
    y = dataset[TARGET_COL].to_numpy(dtype=float)
    prev_close = dataset["Close"].to_numpy(dtype=float)
    dates = dataset.index

    # Learning target is the next-day return: models predict the return and we
    # reconstruct the price (close_t * (1 + ret)). This keeps the target
    # stationary so tree/linear/LSTM models can extrapolate on a trending series.
    y_ret = y / prev_close - 1.0

    n = len(dataset)
    train_end, val_end = chronological_split(n)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    yr_train, yr_val = y_ret[:train_end], y_ret[train_end:val_end]
    y_test = y[val_end:]
    prev_test = prev_close[val_end:]
    dates_test = dates[val_end:]

    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xval_s = scaler.transform(X_val)
    Xtest_s = scaler.transform(X_test)

    def to_price(ret_pred: np.ndarray) -> np.ndarray:
        return prev_test * (1.0 + np.asarray(ret_pred, dtype=float))

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    test_preds: dict[str, np.ndarray] = {}

    # 1. Naive baseline: next-day close = today's close (predicted return = 0)
    baseline_pred = prev_test
    results["Naive Baseline"] = ev.compute_metrics(y_test, baseline_pred, prev_test)
    test_preds["Naive Baseline"] = baseline_pred

    # 2. Linear Regression
    lr = LinearRegression().fit(Xtr_s, yr_train)
    lr_pred = to_price(lr.predict(Xtest_s))
    results["Linear Regression"] = ev.compute_metrics(y_test, lr_pred, prev_test)
    fitted["Linear Regression"] = lr
    test_preds["Linear Regression"] = lr_pred

    # 3. Random Forest
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(Xtr_s, yr_train)
    rf_pred = to_price(rf.predict(Xtest_s))
    results["Random Forest"] = ev.compute_metrics(y_test, rf_pred, prev_test)
    fitted["Random Forest"] = rf
    test_preds["Random Forest"] = rf_pred

    # 4. XGBoost (with validation-based early stopping)
    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        early_stopping_rounds=40,
        eval_metric="rmse",
    )
    xgb.fit(Xtr_s, yr_train, eval_set=[(Xval_s, yr_val)], verbose=False)
    xgb_pred = to_price(xgb.predict(Xtest_s))
    results["XGBoost"] = ev.compute_metrics(y_test, xgb_pred, prev_test)
    fitted["XGBoost"] = xgb
    test_preds["XGBoost"] = xgb_pred

    # 5. Optional LSTM
    lstm_model = None
    if _tensorflow_available():
        try:
            lstm_model, lstm_ret = _train_lstm(
                Xtr_s, yr_train, Xval_s, yr_val, Xtest_s, len(feat_cols)
            )
            lstm_pred = to_price(lstm_ret)
            results["LSTM"] = ev.compute_metrics(y_test, lstm_pred, prev_test)
            test_preds["LSTM"] = lstm_pred
        except Exception as exc:  # keep pipeline alive if LSTM fails
            print(f"[LSTM skipped due to error: {exc}]")
    else:
        print("[LSTM skipped: TensorFlow not available]")

    table = ev.comparison_table(results)

    # Best model among trained ML models (exclude naive for persistence)
    ml_models = [m for m in table.index if m != "Naive Baseline"]
    best_name = table.loc[ml_models, "RMSE"].idxmin()

    # Plots
    ev.plot_comparison(table)
    ev.plot_predictions(dates_test, y_test, test_preds[best_name], best_name)
    if best_name in ("Random Forest", "XGBoost"):
        ev.plot_feature_importance(
            feat_cols, np.asarray(fitted[best_name].feature_importances_), best_name
        )

    if save:
        _save_artifacts(
            best_name=best_name,
            fitted=fitted,
            lstm_model=lstm_model,
            scaler=scaler,
            feat_cols=feat_cols,
            results=results,
            table=table,
            report=report,
            n_sizes=(train_end, val_end - train_end, n - val_end),
            date_range=(str(dates.min().date()), str(dates.max().date())),
        )

    return {
        "table": table,
        "best_name": best_name,
        "results": results,
        "dates_test": dates_test,
        "y_test": y_test,
        "best_pred": test_preds[best_name],
        "feature_columns": feat_cols,
    }


def _save_artifacts(
    best_name,
    fitted,
    lstm_model,
    scaler,
    feat_cols,
    results,
    table,
    report,
    n_sizes,
    date_range,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(feat_cols, indent=2)
    )

    is_lstm = best_name == "LSTM"
    if is_lstm and lstm_model is not None:
        lstm_model.save(MODELS_DIR / "model.keras")
        best_file = "model.keras"
    else:
        joblib.dump(fitted[best_name], MODELS_DIR / "best_model.pkl")
        best_file = "best_model.pkl"

    metadata = {
        "best_model": best_name,
        "best_model_file": best_file,
        "is_lstm": is_lstm,
        "lstm_lookback": LSTM_LOOKBACK if is_lstm else None,
        "n_features": len(feat_cols),
        "target": "next_day_close",
        "prediction_mode": "return_reconstructed",
        "train_size": n_sizes[0],
        "val_size": n_sizes[1],
        "test_size": n_sizes[2],
        "date_range": date_range,
        "data_report": report,
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    metrics_out = {
        "models": results,
        "comparison_sorted_by_rmse": table.reset_index().rename(columns={"index": "Model"}).to_dict(orient="records"),
        "best_model": best_name,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str))


if __name__ == "__main__":
    out = train_all()
    print("\n=== Model Comparison (sorted by RMSE) ===")
    print(out["table"].round(4).to_string())
    print("\nBest model:", out["best_name"])
