from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IsotopeProfile:
    symbol: str
    name: str
    half_life: str
    category: str
    gamma_lines_kev: tuple[tuple[float, float], ...]
    summary: str


# A compact demo library. Gamma-line intensities are normalized teaching
# approximations for synthetic spectra, not a substitute for evaluated data.
DEMO_ISOTOPES: tuple[IsotopeProfile, ...] = (
    IsotopeProfile(
        "Cs-137",
        "Cesium-137",
        "30.05 years",
        "fission product",
        ((661.7, 1.0),),
        "A common calibration and environmental-monitoring isotope.",
    ),
    IsotopeProfile(
        "Co-60",
        "Cobalt-60",
        "5.27 years",
        "industrial and medical source",
        ((1173.2, 1.0), (1332.5, 1.0)),
        "Used in industrial radiography and some irradiation applications.",
    ),
    IsotopeProfile(
        "K-40",
        "Potassium-40",
        "1.25 billion years",
        "naturally occurring",
        ((1460.8, 1.0),),
        "A naturally occurring isotope found in soil and living tissue.",
    ),
    IsotopeProfile(
        "I-131",
        "Iodine-131",
        "8.06 days",
        "medical and fission product",
        ((284.3, 0.16), (364.5, 1.0), (637.0, 0.10)),
        "Used in thyroid diagnosis and treatment; internal exposure is important.",
    ),
    IsotopeProfile(
        "Am-241",
        "Americium-241",
        "432.2 years",
        "industrial source",
        ((59.5, 1.0),),
        "Used in some industrial gauges and ionization smoke detectors.",
    ),
    IsotopeProfile(
        "Na-22",
        "Sodium-22",
        "2.60 years",
        "laboratory source",
        ((511.0, 1.0), (1274.5, 1.0)),
        "A positron-emitting isotope with an annihilation peak near 511 keV.",
    ),
)


def isotope_by_symbol(symbol: str) -> IsotopeProfile:
    for isotope in DEMO_ISOTOPES:
        if isotope.symbol == symbol:
            return isotope
    raise KeyError(f"Unknown demo isotope: {symbol}")

