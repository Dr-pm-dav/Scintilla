from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .isotopes import DEMO_ISOTOPES, IsotopeProfile

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Spectrum:
    energy_kev: FloatArray
    counts: FloatArray
    activities: FloatArray


class SpectrumGenerator:
    """Generate HPGe-like teaching spectra with controlled imperfections."""

    def __init__(
        self,
        isotopes: Sequence[IsotopeProfile] = DEMO_ISOTOPES,
        channels: int = 2048,
        max_energy_kev: float = 3000.0,
        seed: int = 13,
    ) -> None:
        self.isotopes = tuple(isotopes)
        self.channels = channels
        self.max_energy_kev = max_energy_kev
        self.energy_kev = np.linspace(0.0, max_energy_kev, channels, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self._templates = np.vstack([self._template(isotope) for isotope in self.isotopes])

    def _template(self, isotope: IsotopeProfile) -> FloatArray:
        spectrum = np.zeros_like(self.energy_kev)
        for line_energy, relative_intensity in isotope.gamma_lines_kev:
            # HPGe energy resolution broadens modestly with energy.
            sigma = 1.25 + 0.0012 * line_energy
            photopeak = np.exp(-0.5 * ((self.energy_kev - line_energy) / sigma) ** 2)
            compton_edge = line_energy * 0.82
            continuum = np.where(
                self.energy_kev < compton_edge,
                np.exp(-self.energy_kev / max(180.0, line_energy * 0.55)),
                0.0,
            )
            spectrum += relative_intensity * (photopeak + 0.035 * continuum)
        total = spectrum.sum()
        return spectrum / total if total else spectrum

    def sample_activities(
        self,
        max_isotopes: int = 3,
        activity_range: tuple[float, float] = (12.0, 180.0),
    ) -> FloatArray:
        activities = np.zeros(len(self.isotopes), dtype=np.float64)
        mixture_size = int(self.rng.integers(1, max_isotopes + 1))
        selected = self.rng.choice(len(self.isotopes), size=mixture_size, replace=False)
        activities[selected] = self.rng.uniform(*activity_range, size=mixture_size)
        return activities

    def generate(
        self,
        activities: Sequence[float] | Mapping[str, float] | None = None,
        *,
        gain_shift: float | None = None,
        poisson_noise: bool = True,
        background_level: float | None = None,
    ) -> Spectrum:
        vector = self._activity_vector(activities)
        gain_shift = float(self.rng.normal(1.0, 0.0015) if gain_shift is None else gain_shift)
        background_level = float(
            self.rng.uniform(1.5, 8.0) if background_level is None else background_level
        )
        shifted_energy = self.energy_kev / gain_shift
        signal = vector @ self._templates
        signal = np.interp(shifted_energy, self.energy_kev, signal, left=0.0, right=0.0)
        background = background_level * (
            0.45 + np.exp(-self.energy_kev / 720.0) + 0.08 * np.sin(self.energy_kev / 95.0) ** 2
        )
        expected_counts = np.clip(signal * 3200.0 + background, 0.0, None)
        counts = self.rng.poisson(expected_counts) if poisson_noise else expected_counts
        return Spectrum(self.energy_kev.copy(), counts.astype(np.float64), vector)

    def make_dataset(self, samples: int = 1200, max_isotopes: int = 3) -> tuple[FloatArray, FloatArray]:
        spectra = np.empty((samples, self.channels), dtype=np.float64)
        labels = np.empty((samples, len(self.isotopes)), dtype=np.float64)
        for row in range(samples):
            spectrum = self.generate(self.sample_activities(max_isotopes=max_isotopes))
            spectra[row] = spectrum.counts
            labels[row] = spectrum.activities
        return spectra, labels

    def _activity_vector(self, activities: Sequence[float] | Mapping[str, float] | None) -> FloatArray:
        if activities is None:
            return self.sample_activities()
        if isinstance(activities, Mapping):
            return np.array([float(activities.get(item.symbol, 0.0)) for item in self.isotopes])
        vector = np.asarray(activities, dtype=np.float64)
        if vector.shape != (len(self.isotopes),):
            raise ValueError(f"Expected {len(self.isotopes)} activities, received shape {vector.shape}")
        return vector

