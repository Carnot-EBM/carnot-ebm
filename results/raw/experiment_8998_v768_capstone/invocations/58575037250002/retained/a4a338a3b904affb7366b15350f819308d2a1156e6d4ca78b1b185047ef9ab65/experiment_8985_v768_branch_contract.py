"""REQ-REPORT-8985: run bounded owned checks before publishing exact administrative bytes."""

from pathlib import Path
import sys
from unittest.mock import patch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import v755_branch_runner as runner
from carnot.reporting import v768_branch_contract as contract

if __name__ == "__main__":
    with patch.object(runner, "c", contract), patch.object(runner, "run_check", contract.run_check):
        raise SystemExit(runner.main(["--date", "20261111", *sys.argv[1:]]))
