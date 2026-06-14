# Scintilla

Scintilla fingerprints radioactive isotopes from gamma-ray spectra, estimates
their activities in mixtures, and generates a source-grounded educational
radiation-safety brief.

The initial vertical slice includes:

- a synthetic HPGe-like spectrum generator for a runnable demo;
- a Ridge regression baseline for non-negative activity estimates;
- an optional multitask 1D CNN for isotope presence and activity estimation;
- a loader for the public `hpge-soil-gamma-41` NumPy archive;
- a Streamlit dashboard with an animated, glow-style spectrum plot; and
- a deliberately bounded safety retriever backed by public CDC, NRC, and IAEA
  pages.

This is research software, not a calibrated radiation instrument. Its output is
educational and must not be used for emergency response, medical decisions, or
regulatory compliance.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The bundled models in `artifacts/` were trained with scikit-learn 1.9.0, and
`requirements.txt` pins that version so they load without a version-mismatch
warning; training your own model removes the constraint.

The app trains a small demo model in memory on first launch. To create a saved
demo model:

```powershell
python -m scripts.train_demo --output artifacts/demo-ridge.joblib
```

After training the real benchmark model, choose `HPGe benchmark CSV upload` in
the sidebar. Upload one raw 8,192-channel spectrum as either a single CSV row or
a one-column CSV with a `counts` header. Scintilla shows the top eight activity
estimates in Bq and exposes the full ranked 41-label table.

Export upload-ready held-out examples and their ground-truth activity values:

```powershell
python -m scripts.export_hpge_examples `
  data\hpge-soil-gamma-41\hpge-soil-gamma-41.npz `
  data\hpge-soil-gamma-41\data\Activity.csv
```

The HPGe upload mode shows a download button for the first exported example.

## Public Benchmark

The 2025 HPGe soil benchmark contains 6,000 MCNP-simulated spectra and activity
labels for 41 radionuclides:

- Dataset: <https://github.com/haipn91/hpge-soil-gamma-41>
- Paper: <https://doi.org/10.21203/rs.3.rs-6532814/v1>

The repository includes `Activity.csv`, the MCNP input deck, and example
notebooks. The larger `.npz` archive is hosted externally due to GitHub's file
size limit. Clone the repository, download `hpge-soil-gamma-41.npz` from its
documented Google Drive link, then inspect the archive:

```powershell
python -m scripts.inspect_hpge_archive path\to\archive.npz
```

Train the published 41-output Ridge baseline:

```powershell
python -m scripts.train_hpge_ridge `
  path\to\hpge-soil-gamma-41.npz `
  data\hpge-soil-gamma-41\data\Activity.csv `
  --output artifacts\hpge-ridge.joblib
```

The loader validates the upstream `x_train`, `y_train`, `x_test`, and `y_test`
contract before training.

## Roadmap

1. Reproduce published Ridge and XGBoost baselines on all 41 outputs.
2. Train and calibrate the multitask CNN under Poisson noise and gain shifts.
3. Add detector-aware augmentation and pairwise overlap diagnostics.
4. Replace the lexical safety retriever with a cited vector-store RAG pipeline.
5. Validate transfer behavior on real spectra and a portable NaI(Tl) dataset.
