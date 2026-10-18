"""REQ-REPORT-8457: expose the no-model runner from inside or outside the checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting.v728_contract_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
