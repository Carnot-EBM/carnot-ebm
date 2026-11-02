"""REQ-REPORT-8492: thin standalone entry preserves direct CLI and private replay paths."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.publication_cost_runner_8492 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
