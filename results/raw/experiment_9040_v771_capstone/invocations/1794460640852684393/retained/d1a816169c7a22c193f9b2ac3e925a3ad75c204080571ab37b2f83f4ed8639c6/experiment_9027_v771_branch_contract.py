"""REQ-REPORT-9027-V771: publish checked administrative custody with zero model loads."""

import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import v755_branch_runner as runner
from carnot.reporting import v771_branch_contract as contract

if __name__ == "__main__":
    with (
        patch.object(runner, "c", contract),
        patch.object(runner, "run_check", contract.run_check),
        patch.object(runner, "validate", contract.validate),
        patch.object(runner, "terminal", contract.terminal),
    ):
        raise SystemExit(runner.main(["--date", "20261112", *sys.argv[1:]]))
