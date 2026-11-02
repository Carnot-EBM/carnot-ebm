"""REQ-REPORT-8382: use one entry point for runtime evidence and cold replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.runtime_evidence_execution_8382 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
