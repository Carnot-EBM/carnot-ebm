"""REQ-REPORT-8443: expose the dated no-model runner outside the checkout too."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting.v727_contract_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
