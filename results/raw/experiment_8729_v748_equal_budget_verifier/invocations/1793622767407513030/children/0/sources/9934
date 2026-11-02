"""REQ-REPORT-8487: thin entry keeps reusable CPU fitting and private validation together."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v730_source_fit_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
