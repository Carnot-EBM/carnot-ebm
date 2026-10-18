"""REQ-REPORT-8493: thin standalone entry preserves direct CLI and cold replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.append_store_runner_8493 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
