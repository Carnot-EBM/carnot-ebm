"""REQ-REPORT-8185: use explicit checkout paths from any caller directory.

The audit reads cached evidence and never loads a language model.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.sentence_decision_audit_8185 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
