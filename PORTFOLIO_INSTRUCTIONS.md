# Portfolio Run Instructions: Scintilla

Scintilla is the isotope-fingerprinting portfolio demo. Use it to show
full-spectrum gamma-ray inference, benchmark HPGe uploads, transparent activity
estimates, and a bounded educational radiation-safety brief.

## Recommended Demo Path

1. Start from this folder:

   ```powershell
   cd "I:\Schools & Learnings\Projects\Projects\flagship-idea-isotope-fingerprinting-from-what"
   ```

2. Create and install the local environment. This project requires Python 3.11
   or newer.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```

3. Start the Streamlit app.

   ```powershell
   .\.venv\Scripts\python -m streamlit run app.py
   ```

4. Open the local Streamlit URL printed in the terminal, usually:

   ```text
   http://localhost:8501
   ```

## Portfolio Demo Flow

1. Begin in `Synthetic mixture` mode.
2. Use the sliders to create a simple Cs-137 plus Co-60 mixture.
3. Point out the full-spectrum plot, the ranked isotope fingerprint, and the
   educational safety brief with public-source citations.
4. Switch to `HPGe benchmark CSV upload` only after confirming that
   `artifacts/hpge-ridge.joblib` exists.
5. Download the held-out example from the app, upload it back, and show the
   top activity estimates plus the expandable 41-label table.

## Optional Benchmark Preparation

The repository already contains the benchmark archive and trained Ridge artifact
in the expected locations. To regenerate the benchmark model:

```powershell
.\.venv\Scripts\python -m scripts.train_hpge_ridge `
  data\hpge-soil-gamma-41\hpge-soil-gamma-41.npz `
  data\hpge-soil-gamma-41\data\Activity.csv `
  --output artifacts\hpge-ridge.joblib
```

Export upload-ready held-out examples:

```powershell
.\.venv\Scripts\python -m scripts.export_hpge_examples `
  data\hpge-soil-gamma-41\hpge-soil-gamma-41.npz `
  data\hpge-soil-gamma-41\data\Activity.csv
```

## Verification Before Presenting

Run the tests:

```powershell
.\.venv\Scripts\python -m pytest
```

Quickly confirm the app imports:

```powershell
.\.venv\Scripts\python -m pytest tests\test_app.py
```

## Smooth Portfolio Checklist

- Keep `artifacts/hpge-ridge.joblib` in place before presenting HPGe upload
  mode.
- Keep at least one CSV in `outputs/hpge-upload-examples/` for a fast upload
  demo.
- Lead with the disclaimer: this is educational research software, not a
  calibrated radiation instrument.
- Use this framing: "Scintilla treats the whole spectrum as the signal, then
  converts it into ranked activity estimates and source-grounded safety context."

## Troubleshooting

- If Streamlit is missing, reinstall dependencies with
  `.\.venv\Scripts\python -m pip install -r requirements.txt`.
- If HPGe upload mode stops with a missing model message, run the benchmark
  training command above.
- If a CSV upload fails, use a one-column file with a `counts` header or a
  single-row 8,192-channel HPGe spectrum.
- If the app starts on a different port, use the URL printed by Streamlit.

