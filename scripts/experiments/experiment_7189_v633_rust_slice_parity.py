#!/usr/bin/env python3
"""Run the V633 Rust fixed-cardinality slice parity experiment."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7189_v633_rust_slice_parity import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
