import numpy as np

from scintilla.hpge_archive import load_activity_symbols, load_split_archive


def test_load_split_archive_validates_public_contract(tmp_path) -> None:
    archive_path = tmp_path / "hpge.npz"
    np.savez(
        archive_path,
        x_train=np.zeros((8, 16)),
        y_train=np.zeros((8, 3)),
        x_test=np.zeros((2, 16)),
        y_test=np.zeros((2, 3)),
    )

    archive = load_split_archive(archive_path)

    assert archive.x_train.shape == (8, 16)
    assert archive.y_test.shape == (2, 3)


def test_load_activity_symbols_strips_upstream_spacing(tmp_path) -> None:
    activity_path = tmp_path / "Activity.csv"
    activity_path.write_text("208Tl, 212BiPo,7Be\n1,2,3\n", encoding="utf-8")

    assert load_activity_symbols(activity_path) == ("208Tl", "212BiPo", "7Be")

