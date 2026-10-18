"""REQ-REPORT-8486: thin dated entry uses reusable qualification and bounded validation."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v730_source_control_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
