from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import mean_absolute_error

from scintilla.hpge_archive import load_activity_symbols, load_split_archive
from scintilla.model import RidgeQuantifier


def thresholded_relative_accuracy(
    expected: np.ndarray,
    predicted: np.ndarray,
    *,
    minimum_activity_bq: float = 1.0,
    relative_error_limit: float = 0.15,
) -> float:
    mask = expected >= minimum_activity_bq
    relative_errors = np.abs(predicted[mask] - expected[mask]) / expected[mask]
    return float(np.mean(relative_errors <= relative_error_limit))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ridge on the public 41-isotope HPGe benchmark.")
    parser.add_argument("archive", help="Path to hpge-soil-gamma-41.npz")
    parser.add_argument("activity_csv", help="Path to the upstream Activity.csv label file")
    parser.add_argument("--output", default="artifacts/hpge-ridge.joblib")
    args = parser.parse_args()

    dataset = load_split_archive(args.archive)
    symbols = load_activity_symbols(args.activity_csv)
    if len(symbols) != dataset.y_train.shape[1]:
        raise ValueError(f"Loaded {len(symbols)} isotope names for {dataset.y_train.shape[1]} outputs.")

    model = RidgeQuantifier(symbols).fit(dataset.x_train, dataset.y_train)
    predictions = model.predict(dataset.x_test)
    model.save(args.output)

    mae = mean_absolute_error(dataset.y_test, predictions)
    within_limit = thresholded_relative_accuracy(dataset.y_test, predictions)
    print(f"Saved 41-output Ridge model to {args.output}")
    print(f"Test MAE: {mae:.3f} Bq")
    print(f"Test values within +/-15% relative error above 1 Bq: {within_limit:.1%}")


if __name__ == "__main__":
    main()

