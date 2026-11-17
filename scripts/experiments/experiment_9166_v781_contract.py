"""REQ-REPORT-9166-V781: bind fourteen tasks and freeze diagnostics without model work."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v781_contract import main

if __name__ == "__main__":
    raise SystemExit(main())
