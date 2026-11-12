"""REQ-REPORT-9031-V771: run with explicit imports outside the checkout."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.graph_corpus_runner_9031 import main

if __name__ == "__main__":
    raise SystemExit(main())
