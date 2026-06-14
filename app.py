from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from scintilla.isotopes import DEMO_ISOTOPES
from scintilla.model import RidgeQuantifier
from scintilla.safety import safety_brief
from scintilla.spectra import SpectrumGenerator
from scintilla.uploads import parse_spectrum_csv

PROJECT_ROOT = Path(__file__).resolve().parent
HPGE_MODEL_PATH = PROJECT_ROOT / "artifacts" / "hpge-ridge.joblib"
HPGE_EXAMPLE_PATH = PROJECT_ROOT / "outputs" / "hpge-upload-examples" / "heldout-0000-spectrum.csv"
HPGE_CHANNELS = 8192
HPGE_TOP_RESULTS = 8

st.set_page_config(page_title="Scintilla", page_icon="*", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 75% 0%, #102719 0, #07110d 40%, #040807 100%); }
    h1, h2, h3 { color: #d8ffe9; }
    [data-testid="stMetric"] {
        background: rgba(16, 56, 39, 0.42);
        border: 1px solid rgba(108, 255, 170, 0.22);
        border-radius: 14px;
        padding: 0.75rem;
    }
    .brief {
        background: rgba(7, 22, 16, 0.78);
        border-left: 3px solid #6cffaa;
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def demo_system() -> tuple[SpectrumGenerator, RidgeQuantifier]:
    generator = SpectrumGenerator(seed=29)
    spectra, activities = generator.make_dataset(samples=1500)
    return generator, RidgeQuantifier().fit(spectra, activities)


@st.cache_resource
def hpge_model() -> RidgeQuantifier | None:
    return RidgeQuantifier.load(HPGE_MODEL_PATH) if HPGE_MODEL_PATH.exists() else None


def spectrum_figure(
    horizontal_axis: np.ndarray,
    counts: np.ndarray,
    *,
    axis_title: str = "Energy (keV)",
    hover_suffix: str = "keV",
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=horizontal_axis,
            y=counts,
            mode="lines",
            line={"color": "#6cffaa", "width": 1.3},
            fill="tozeroy",
            fillcolor="rgba(76, 255, 155, 0.12)",
            hovertemplate=f"%{{x:.1f}} {hover_suffix}<br>%{{y:.0f}} counts<extra></extra>",
        )
    )
    figure.update_layout(
        height=420,
        margin={"l": 15, "r": 15, "t": 18, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#b8d9c6"},
        xaxis={"title": axis_title, "gridcolor": "rgba(160,255,190,0.08)"},
        yaxis={"title": "Counts", "gridcolor": "rgba(160,255,190,0.08)"},
    )
    return figure


generator, demo_model = demo_system()
real_model = hpge_model()

st.title("* SCINTILLA")
st.caption("Full-spectrum isotope fingerprinting | synthetic teaching mode + 41-isotope HPGe benchmark")
st.warning(
    "Research demo only. Results are synthetic or simulated and educational, not suitable for emergency "
    "response, medical decisions, or regulatory compliance."
)

with st.sidebar:
    st.header("Spectrum source")
    mode = st.radio("Input", ["Synthetic mixture", "Demo CSV upload", "HPGe benchmark CSV upload"])
    activities: dict[str, float] = {}
    if mode == "Synthetic mixture":
        st.caption("Compose a source and watch the complete spectrum respond.")
        for isotope in DEMO_ISOTOPES:
            activities[isotope.symbol] = st.slider(isotope.symbol, 0, 180, 0, 5)
        if not any(activities.values()):
            activities = {"Cs-137": 85.0, "Co-60": 55.0}
            st.info("Showing a Cs-137 + Co-60 starter mixture.")
        spectrum = generator.generate(activities, gain_shift=1.0)
        counts = spectrum.counts
        inference_model = demo_model
        axis = spectrum.energy_kev
        axis_title = "Energy (keV)"
        hover_suffix = "keV"
        result_unit = "a.u."
        result_limit = len(DEMO_ISOTOPES)
        threshold_bq = 4.0
    elif mode == "Demo CSV upload":
        upload = st.file_uploader("Spectrum CSV", type=["csv"])
        st.caption(f"Expected: one `counts` column with {generator.channels} rows.")
        if upload is None:
            st.stop()
        try:
            counts = parse_spectrum_csv(upload, generator.channels)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        inference_model = demo_model
        axis = generator.energy_kev
        axis_title = "Energy (keV)"
        hover_suffix = "keV"
        result_unit = "a.u."
        result_limit = len(DEMO_ISOTOPES)
        threshold_bq = 4.0
    else:
        st.caption("Use one raw MCNP spectrum from the public HPGe benchmark.")
        if real_model is None:
            st.error("Train `artifacts/hpge-ridge.joblib` before using real benchmark inference.")
            st.stop()
        if HPGE_EXAMPLE_PATH.exists():
            st.download_button(
                "Download held-out example CSV",
                data=HPGE_EXAMPLE_PATH.read_bytes(),
                file_name=HPGE_EXAMPLE_PATH.name,
                mime="text/csv",
                help="Download this public benchmark sample, then upload it below to exercise the real model.",
            )
        upload = st.file_uploader("HPGe benchmark spectrum CSV", type=["csv"])
        st.caption(f"Expected: one row or one `counts` column with {HPGE_CHANNELS:,} raw channels.")
        if upload is None:
            st.stop()
        try:
            counts = parse_spectrum_csv(upload, HPGE_CHANNELS)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        inference_model = real_model
        axis = np.arange(HPGE_CHANNELS, dtype=np.float64)
        axis_title = "Raw MCNP channel"
        hover_suffix = "channel"
        result_unit = "Bq"
        result_limit = HPGE_TOP_RESULTS
        threshold_bq = 1.0

estimates = inference_model.estimates(counts, threshold_bq=threshold_bq)
present = [estimate for estimate in estimates if estimate.present]
ranked_present = present[:result_limit]

left, right = st.columns([1.85, 1])
with left:
    st.subheader("Detector response")
    st.plotly_chart(
        spectrum_figure(axis, counts, axis_title=axis_title, hover_suffix=hover_suffix),
        width="stretch",
    )
    st.caption("Full-spectrum inference uses peaks, continuum shape, and background together.")

with right:
    st.subheader("Fingerprint")
    if not present:
        st.info("No isotope exceeded the activity threshold.")
    for estimate in ranked_present:
        isotope = next((item for item in DEMO_ISOTOPES if item.symbol == estimate.symbol), None)
        help_text = (
            f"{isotope.name} | {isotope.half_life} | ranking score {estimate.confidence:.0%}"
            if isotope
            else "Public benchmark label | ranking score is not a calibrated probability"
        )
        st.metric(estimate.symbol, f"{estimate.activity:.1f} {result_unit}", help=help_text)
    if len(present) > result_limit:
        st.caption(f"Showing the top {result_limit} of {len(present)} labels above {threshold_bq:g} Bq.")

if mode == "HPGe benchmark CSV upload":
    with st.expander("All 41 activity estimates"):
        st.dataframe(
            {
                "isotope": [estimate.symbol for estimate in estimates],
                "estimated_activity_bq": [round(estimate.activity, 3) for estimate in estimates],
                "above_threshold": [estimate.present for estimate in estimates],
            },
            hide_index=True,
            width="stretch",
        )

st.subheader("Educational safety brief")
brief, sources = safety_brief([estimate.symbol for estimate in ranked_present])
st.markdown(f'<div class="brief">{brief}</div>', unsafe_allow_html=True)
for source in sources:
    st.markdown(f"- [{source.title}]({source.url}) - {source.note}")

with st.expander("What this first build does"):
    st.markdown(
        """
        - Generates HPGe-like spectra with photopeaks, a simple continuum, background, Poisson noise, and gain shift.
        - Estimates non-negative activities with a transparent Ridge baseline.
        - Includes an optional multitask CNN module for the next experiment.
        - Retrieves a short, cited educational brief from a curated public-source index.
        - Loads the trained 41-isotope HPGe Ridge artifact for raw 8,192-channel benchmark uploads.
        """
    )
