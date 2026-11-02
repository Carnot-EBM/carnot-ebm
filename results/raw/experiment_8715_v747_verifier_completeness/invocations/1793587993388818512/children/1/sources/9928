"""REQ-REPORT-8519: a direct entry loads only the current qualification reader."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.append_import_runner_8519 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
