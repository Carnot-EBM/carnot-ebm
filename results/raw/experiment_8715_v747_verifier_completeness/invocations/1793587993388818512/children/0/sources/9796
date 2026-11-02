#!/usr/bin/env python3
"""REQ-REPORT-8291: direct CPU entrypoint also supports private cold replay."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting.dependency_admission_execution_8291 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
