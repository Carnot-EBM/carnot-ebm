"""REQ-VERIFY-8375: direct script execution keeps current model activity at zero."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.external_evidence_runner_8375 import main

if __name__ == "__main__":
    raise SystemExit(main())
