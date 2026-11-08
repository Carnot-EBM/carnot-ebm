"""REQ-REPORT-8891: bounded private validation and atomic publication without activation."""

from pathlib import Path
import sys
from unittest.mock import patch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import v755_branch_runner as runner
from carnot.reporting import v761_branch_contract as contract

if __name__ == "__main__":
    with patch.object(runner, "c", contract):
        raise SystemExit(runner.main(["--date", "20261107", *sys.argv[1:]]))
