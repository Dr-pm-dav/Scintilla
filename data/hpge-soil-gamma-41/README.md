# HPGe Soil Gamma Benchmark (external data)

Scintilla's real-benchmark mode uses the public **MCNP-simulated HPGe soil
gamma-ray spectra** dataset: 6,000 simulated spectra of 8,192 channels each with
activity labels for 41 soil radionuclides. The data is **not bundled in this
repository** because of its size. Download it from the sources below and place
the files in this folder.

## Source and license

- Upstream repository: https://github.com/haipn91/hpge-soil-gamma-41
- Paper: https://doi.org/10.21203/rs.3.rs-6532814/v1
- Contact: haipn91@ioit.ai.vn
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

This is third-party data. Cite the authors above if you use it.

## What to download

The same links are listed in [`data/link_download.txt`](./data/link_download.txt).

- `hpge-soil-gamma-41.npz` (pre-split arrays: `x_train` 4800x8192, `y_train`
  4800x41, `x_test` 1200x8192, `y_test` 1200x41) from the Google Drive link in
  the upstream repository, saved here as `hpge-soil-gamma-41.npz`.
- `Activity.csv` (41-column activity labels) from the upstream repository, saved
  here as `data/Activity.csv` (the path the training and export scripts expect).

With both files in place, the `train_hpge_ridge` and `export_hpge_examples`
scripts in the project README will run. Scintilla also ships a pre-trained model
at `artifacts/hpge-ridge.joblib`, so the Streamlit app and the HPGe upload mode
work without downloading anything.
