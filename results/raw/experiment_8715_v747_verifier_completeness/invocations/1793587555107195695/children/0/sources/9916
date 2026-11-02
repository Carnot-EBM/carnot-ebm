"""REQ-REPORT-8494: thin standalone entry preserves direct CLI and cold replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.append_service_cost_runner_8494 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
