#!/usr/bin/env python3
"""REQ-REPORT-8239: resolve imports for private external-directory audit children."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.margin_decision_audit_execution_8239 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
