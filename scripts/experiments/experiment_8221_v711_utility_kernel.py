#!/usr/bin/env python3
"""REQ-REPORT-8221: direct CLI binds frozen utility mechanics to real work."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.utility_kernel_qualification_8221 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
