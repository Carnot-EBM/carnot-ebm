"""REQ-REPORT-9058-V773: collect the fixed prospective shard with private response custody."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.fresh_corpus_runner_9058 import main

if __name__ == "__main__":
    raise SystemExit(main())
