from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scintilla.hpge_archive import load_activity_symbols, load_split_archive

DEFAULT_INDICES = (0, 596, 840)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export held-out HPGe spectra as dashboard-ready CSV files.")
    parser.add_argument("archive", help="Path to hpge-soil-gamma-41.npz")
    parser.add_argument("activity_csv", help="Path to the upstream Activity.csv label file")
    parser.add_argument("--output-dir", default="outputs/hpge-upload-examples")
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    args = parser.parse_args()

    dataset = load_split_archive(args.archive)
    symbols = load_activity_symbols(args.activity_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_rows = []
    for index in args.indices:
        if index < 0 or index >= len(dataset.x_test):
            raise ValueError(f"Held-out index {index} is outside the test split.")
        spectrum_path = output_dir / f"heldout-{index:04d}-spectrum.csv"
        with spectrum_path.open("w", newline="", encoding="utf-8") as spectrum_file:
            writer = csv.writer(spectrum_file)
            writer.writerow(["counts"])
            writer.writerows((value,) for value in dataset.x_test[index])
        truth_rows.append({"heldout_index": index, **dict(zip(symbols, dataset.y_test[index], strict=True))})
        print(f"Exported {spectrum_path}")

    truth_path = output_dir / "heldout-ground-truth.csv"
    with truth_path.open("w", newline="", encoding="utf-8") as truth_file:
        writer = csv.DictWriter(truth_file, fieldnames=truth_rows[0].keys())
        writer.writeheader()
        writer.writerows(truth_rows)
    print(f"Exported {truth_path}")


if __name__ == "__main__":
    main()

