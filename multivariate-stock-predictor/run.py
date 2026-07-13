"""End-to-end pipeline entry point.

Runs: load -> clean/validate -> EDA -> feature engineering -> train & compare
models -> evaluate -> save artifacts -> report next-day prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_clean
from src.eda import run_eda
from src.train import train_all
from src.predict import predict_next_day


def main() -> None:
    print("=" * 70)
    print(" MULTIVARIATE STOCK PRICE PREDICTION - FULL PIPELINE")
    print("=" * 70)

    print("\n[1/5] Loading and validating data ...")
    data, report = load_clean()
    print(f"  Rows: {report['rows']} | Range: {report['start_date']} -> {report['end_date']}")
    print(f"  Duplicate dates: {report['duplicate_dates']} | "
          f"Extreme moves (>20%): {report['extreme_moves_gt_20pct']}")

    print("\n[2/5] Running exploratory data analysis ...")
    run_eda(data)
    print("  EDA charts saved to outputs/")

    print("\n[3/5] Engineering features and training models ...")
    out = train_all(save=True)

    print("\n[4/5] Model comparison (sorted by test RMSE):")
    print(out["table"].round(4).to_string())
    print(f"\n  Best model: {out['best_name']}")

    print("\n[5/5] Next trading day prediction:")
    pred = predict_next_day()
    print(f"  Model used     : {pred['model']}")
    print(f"  As of date     : {pred['as_of_date']}")
    print(f"  Latest close   : {pred['latest_close']:.2f}")
    print(f"  Predicted close: {pred['predicted_next_close']:.2f} "
          f"({pred['direction']}, {pred['expected_change_pct']:+.2f}%)")

    print("\nArtifacts saved to models/ ; plots saved to outputs/.")
    print("Launch the dashboard with:  streamlit run app.py")
    print("\nDisclaimer: Educational use only. Not financial advice.")


if __name__ == "__main__":
    main()
