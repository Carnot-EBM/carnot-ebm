"""REQ-REPORT-8391: permit fresh direct entry without an ambient import path."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.native_evidence_8391 import main

if __name__ == "__main__":
    raise SystemExit(main())
