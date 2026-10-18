"""REQ-REPORT-8458: expose the dated source control runner from any working directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting.v728_source_control_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
