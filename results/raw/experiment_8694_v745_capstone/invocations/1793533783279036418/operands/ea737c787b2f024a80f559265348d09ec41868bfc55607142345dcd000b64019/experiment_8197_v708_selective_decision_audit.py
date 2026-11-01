"""REQ-REPORT-8197: explicit checkout imports work from any caller directory.

The module reduces cached evidence and never loads a language model.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.selective_decision_audit_8197 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
