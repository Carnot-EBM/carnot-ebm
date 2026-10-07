#!/usr/bin/env python3
"""REQ-REPORT-8223: resolve imports for real external-directory seal children."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.utility_seal_execution_8223 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
