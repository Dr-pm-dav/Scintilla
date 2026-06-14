from __future__ import annotations

from dataclasses import dataclass

from .isotopes import isotope_by_symbol


@dataclass(frozen=True)
class SafetySource:
    title: str
    url: str
    tags: tuple[str, ...]
    note: str


SOURCES: tuple[SafetySource, ...] = (
    SafetySource(
        "CDC: How to Measure Radiation and Radioactivity",
        "https://www.cdc.gov/radiation-health/about/how-to-measure-radiation.html",
        ("measurement", "interpretation", "professional"),
        "Detector readings require suitable equipment and trained interpretation.",
    ),
    SafetySource(
        "CDC: About Radiation Emergencies",
        "https://www.cdc.gov/radiation-emergencies/about/index.html",
        ("emergency", "exposure", "contamination"),
        "For an actual radiation emergency, follow instructions from public officials.",
    ),
    SafetySource(
        "CDC: Iodine-131",
        "https://www.cdc.gov/radiation-emergencies/hcp/isotopes/iodine-131.html",
        ("I-131", "thyroid", "internal exposure"),
        "CDC describes I-131 properties, uses, and internal-exposure considerations.",
    ),
    SafetySource(
        "NRC: Radiation Protection",
        "https://www.nrc.gov/about-nrc/radiation",
        ("protection", "regulation", "exposure"),
        "NRC provides public radiation-protection context for civilian uses.",
    ),
    SafetySource(
        "IAEA: Radioisotopes",
        "https://www.iaea.org/topics/nuclear-science/isotopes/radioisotopes",
        ("radioisotope", "industry", "medicine"),
        "IAEA summarizes radioisotope properties and common applications.",
    ),
)


def safety_brief(symbols: list[str]) -> tuple[str, list[SafetySource]]:
    isotopes = []
    unknown_symbols = []
    for symbol in symbols:
        try:
            isotopes.append(isotope_by_symbol(symbol))
        except KeyError:
            unknown_symbols.append(symbol)
    names = ", ".join(symbols) or "no isotope"
    notes = " ".join(f"{item.symbol}: {item.summary}" for item in isotopes)
    if unknown_symbols:
        notes += (
            " The public benchmark also includes decay-chain labels without curated "
            "isotope-specific summaries in this first build."
        )
    selected = [source for source in SOURCES if any(symbol in source.tags for symbol in symbols)]
    selected.extend(source for source in SOURCES if source not in selected)
    brief = (
        f"The model flagged {names}. {notes} "
        "Treat this as an educational screening result, not a field determination. "
        "A calibrated instrument and a radiation-safety professional are required "
        "for real-world interpretation."
    )
    return brief, selected[:4]
