from io import BytesIO

import numpy as np

from scintilla.uploads import parse_spectrum_csv


def test_parse_spectrum_csv_accepts_counts_column() -> None:
    spectrum = parse_spectrum_csv(BytesIO(b"counts\n1\n2\n3\n"), expected_channels=3)

    np.testing.assert_array_equal(spectrum, [1.0, 2.0, 3.0])


def test_parse_spectrum_csv_accepts_headerless_row() -> None:
    spectrum = parse_spectrum_csv(BytesIO(b"1,2,3\n"), expected_channels=3)

    np.testing.assert_array_equal(spectrum, [1.0, 2.0, 3.0])


def test_parse_spectrum_csv_accepts_file_handle(tmp_path) -> None:
    spectrum_path = tmp_path / "spectrum.csv"
    spectrum_path.write_text("counts\n1\n2\n3\n", encoding="utf-8")

    with spectrum_path.open("rb") as spectrum_file:
        spectrum = parse_spectrum_csv(spectrum_file, expected_channels=3)

    np.testing.assert_array_equal(spectrum, [1.0, 2.0, 3.0])
