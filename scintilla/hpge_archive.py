from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HpgeArchive:
    spectra: FloatArray
    activities: FloatArray
    spectrum_key: str
    activity_key: str


@dataclass(frozen=True)
class HpgeSplitArchive:
    x_train: FloatArray
    y_train: FloatArray
    x_test: FloatArray
    y_test: FloatArray


def inspect_archive(path: str | Path) -> dict[str, tuple[int, ...]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key].shape for key in archive.files}


def load_archive(
    path: str | Path,
    spectrum_key: str | None = None,
    activity_key: str | None = None,
) -> HpgeArchive:
    """Load the public archive while tolerating descriptive upstream key names."""
    with np.load(Path(path), allow_pickle=False) as archive:
        shapes = {key: archive[key].shape for key in archive.files}
        spectrum_key = spectrum_key or _infer_matrix_key(shapes, expected_columns=8192)
        activity_key = activity_key or _infer_matrix_key(shapes, expected_columns=41)
        spectra = np.asarray(archive[spectrum_key], dtype=np.float64)
        activities = np.asarray(archive[activity_key], dtype=np.float64)

    if spectra.ndim != 2 or activities.ndim != 2:
        raise ValueError("Spectra and activities must both be two-dimensional matrices.")
    if spectra.shape[0] != activities.shape[0]:
        raise ValueError("Spectra and activity matrices must contain the same number of rows.")
    return HpgeArchive(spectra, activities, spectrum_key, activity_key)


def load_split_archive(path: str | Path) -> HpgeSplitArchive:
    """Load the concrete train/test contract published by hpge-soil-gamma-41."""
    required_keys = ("x_train", "y_train", "x_test", "y_test")
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = [key for key in required_keys if key not in archive.files]
        if missing:
            raise ValueError(f"Missing required archive arrays: {missing}")
        arrays = {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in required_keys
        }

    _validate_pair(arrays["x_train"], arrays["y_train"], "training")
    _validate_pair(arrays["x_test"], arrays["y_test"], "testing")
    if arrays["x_train"].shape[1] != arrays["x_test"].shape[1]:
        raise ValueError("Training and testing spectra must use the same channel count.")
    if arrays["y_train"].shape[1] != arrays["y_test"].shape[1]:
        raise ValueError("Training and testing labels must use the same isotope count.")
    return HpgeSplitArchive(**arrays)


def load_activity_symbols(path: str | Path) -> tuple[str, ...]:
    with Path(path).open(newline="", encoding="utf-8-sig") as activity_file:
        header = next(csv.reader(activity_file))
    return tuple(symbol.strip() for symbol in header)


def _validate_pair(spectra: FloatArray, activities: FloatArray, split_name: str) -> None:
    if spectra.ndim != 2 or activities.ndim != 2:
        raise ValueError(f"The {split_name} spectra and activities must both be matrices.")
    if spectra.shape[0] != activities.shape[0]:
        raise ValueError(f"The {split_name} spectra and activities must contain the same rows.")


def _infer_matrix_key(shapes: dict[str, tuple[int, ...]], expected_columns: int) -> str:
    candidates = [key for key, shape in shapes.items() if len(shape) == 2 and shape[1] == expected_columns]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not infer a unique matrix with {expected_columns} columns. "
            f"Available arrays: {shapes}"
        )
    return candidates[0]
