from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .isotopes import DEMO_ISOTOPES

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IsotopeEstimate:
    symbol: str
    activity: float
    present: bool
    confidence: float


class RidgeQuantifier:
    """Transparent baseline for multi-isotope activity estimation."""

    def __init__(self, isotope_symbols: Sequence[str] | None = None, alpha: float = 10.0) -> None:
        self.isotope_symbols = tuple(isotope_symbols or [item.symbol for item in DEMO_ISOTOPES])
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, positive=True)),
            ]
        )

    def fit(self, spectra: FloatArray, activities: FloatArray) -> "RidgeQuantifier":
        self.pipeline.fit(spectra, activities)
        return self

    def predict(self, spectra: FloatArray) -> FloatArray:
        predictions = self.pipeline.predict(np.atleast_2d(spectra))
        return np.clip(np.asarray(predictions, dtype=np.float64), 0.0, None)

    def estimates(self, spectrum: FloatArray, threshold_bq: float = 4.0) -> list[IsotopeEstimate]:
        activities = self.predict(spectrum)[0]
        return [
            IsotopeEstimate(
                symbol=symbol,
                activity=float(activity),
                present=bool(activity >= threshold_bq),
                # This is a UI ranking score, not a calibrated probability.
                confidence=float(1.0 - np.exp(-activity / 35.0)),
            )
            for symbol, activity in sorted(
                zip(self.isotope_symbols, activities, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"symbols": self.isotope_symbols, "pipeline": self.pipeline}, target)

    @classmethod
    def load(cls, path: str | Path) -> "RidgeQuantifier":
        payload = joblib.load(path)
        model = cls(payload["symbols"])
        model.pipeline = payload["pipeline"]
        return model

