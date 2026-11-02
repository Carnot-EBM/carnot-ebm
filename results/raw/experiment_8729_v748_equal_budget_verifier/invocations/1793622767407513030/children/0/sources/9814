#!/usr/bin/env python3
"""REQ-REPORT-8277: direct script entrypoint for native load-only qualification."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.verify.lease_backend_8277 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
