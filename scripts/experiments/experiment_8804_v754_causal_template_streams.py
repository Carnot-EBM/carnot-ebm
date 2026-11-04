"""REQ-REPORT-8804: direct CPU energy-training CLI, without language-model work."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.causal_template_streams_8804 import main

if __name__ == "__main__":
    raise SystemExit(main())
