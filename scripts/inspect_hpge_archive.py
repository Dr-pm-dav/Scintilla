from __future__ import annotations

import argparse

from scintilla.hpge_archive import inspect_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect arrays in the public HPGe NumPy archive.")
    parser.add_argument("archive")
    args = parser.parse_args()
    for key, shape in inspect_archive(args.archive).items():
        print(f"{key}: {shape}")


if __name__ == "__main__":
    main()

