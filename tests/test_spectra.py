import numpy as np

from scintilla.spectra import SpectrumGenerator


def test_generate_returns_expected_shapes() -> None:
    generator = SpectrumGenerator(channels=256, seed=2)
    spectrum = generator.generate({"Cs-137": 80.0}, gain_shift=1.0)

    assert spectrum.energy_kev.shape == (256,)
    assert spectrum.counts.shape == (256,)
    assert spectrum.activities.shape == (6,)
    assert np.all(spectrum.counts >= 0)


def test_dataset_contains_mixtures() -> None:
    generator = SpectrumGenerator(channels=128, seed=3)
    spectra, activities = generator.make_dataset(samples=40, max_isotopes=3)

    assert spectra.shape == (40, 128)
    assert activities.shape == (40, 6)
    assert np.any((activities > 0).sum(axis=1) > 1)

