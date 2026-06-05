from scintilla.model import RidgeQuantifier
from scintilla.spectra import SpectrumGenerator


def test_ridge_quantifier_recovers_dominant_demo_source() -> None:
    generator = SpectrumGenerator(channels=256, seed=7)
    spectra, activities = generator.make_dataset(samples=450)
    model = RidgeQuantifier().fit(spectra, activities)
    sample = generator.generate({"Co-60": 140.0}, gain_shift=1.0, background_level=3.0)

    estimates = model.estimates(sample.counts)

    assert estimates[0].symbol == "Co-60"
    assert estimates[0].present

