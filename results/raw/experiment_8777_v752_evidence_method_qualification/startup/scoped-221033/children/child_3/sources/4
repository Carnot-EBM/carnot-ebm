"""REQ-REPORT-8777: qualify evidence method execution without loading a model."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.evidence_qualification_runner_8777 import main

if __name__ == "__main__":
    raise SystemExit(main())
