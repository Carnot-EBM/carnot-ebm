#!/usr/bin/env python3
"""REQ-REPORT-8207: run private CLI and replay without ambient import paths."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.restricted_action_methods_8207 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
