from __future__ import annotations

import io
from typing import BinaryIO

import numpy as np
import pandas as pd


def parse_spectrum_csv(uploaded_file: BinaryIO | io.BytesIO, expected_channels: int) -> np.ndarray:
    """Read a single spectrum from a row vector or one-column CSV."""
    payload = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    attempts = (
        pd.read_csv(io.BytesIO(payload)),
        pd.read_csv(io.BytesIO(payload), header=None),
    )
    for frame in attempts:
        counts = _extract_counts(frame, expected_channels)
        if counts is not None:
            return counts
    raise ValueError(
        f"Expected one spectrum with {expected_channels} numeric channels. "
        "Use a single `counts` column or one row of channel values."
    )


def _extract_counts(frame: pd.DataFrame, expected_channels: int) -> np.ndarray | None:
    if "counts" in frame.columns:
        candidate = frame["counts"].to_numpy()
    elif frame.shape == (expected_channels, 1):
        candidate = frame.iloc[:, 0].to_numpy()
    elif frame.shape == (1, expected_channels):
        candidate = frame.iloc[0].to_numpy()
    else:
        return None
    try:
        return np.asarray(candidate, dtype=np.float64)
    except ValueError:
        return None
