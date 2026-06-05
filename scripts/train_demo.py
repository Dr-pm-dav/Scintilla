from __future__ import annotations

import argparse

from scintilla.model import RidgeQuantifier
from scintilla.spectra import SpectrumGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Scintilla demo Ridge quantifier.")
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--output", default="artifacts/demo-ridge.joblib")
    args = parser.parse_args()

    generator = SpectrumGenerator(seed=41)
    spectra, activities = generator.make_dataset(samples=args.samples)
    model = RidgeQuantifier().fit(spectra, activities)
    model.save(args.output)
    print(f"Saved demo model to {args.output}")


if __name__ == "__main__":
    main()

