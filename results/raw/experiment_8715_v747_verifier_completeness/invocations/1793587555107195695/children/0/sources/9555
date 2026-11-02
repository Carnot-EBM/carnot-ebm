"""REQ-REPORT-7958: run or cold-reduce the bounded whole-response measurement."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7958_v690_qwen_response_risk import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
