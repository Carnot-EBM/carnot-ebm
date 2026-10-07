#!/usr/bin/env python3
"""REQ-REPORT-8219: expose the frozen method through a direct script CLI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.utility_patch_methods_8219 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
