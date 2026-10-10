"""REQ-REPORT-8350: standalone seal-gated audit and fresh-process replay."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.static_benefit_audit_8350 import main

if __name__ == "__main__":
    raise SystemExit(main())
