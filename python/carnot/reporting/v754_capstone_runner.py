"""REQ-VERIFY-8813: reuse bounded validation and checked publication without model work."""

from pathlib import Path
import sys
from typing import Any

from carnot.reporting import v753_capstone_runner as previous
from carnot.reporting import v754_capstone as c
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
_runtime = namespace(previous, "v754_runtime")
_runtime.c = c
_runtime._validator.c = c


def terminal(candidate: Path, raw: Path) -> Json:
    """An outside-checkout process verifies the same candidate inspected by unchanged scanners."""
    result: Json = _runtime._validator.terminal(candidate, raw)
    receipt = _runtime.child(
        "outside_checkout_replay",
        [
            "env",
            "-u",
            "PYTHONPATH",
            "-C",
            "/var/tmp",
            sys.executable,
            str(c.ROOT / c.CLI),
            "--cold-replay",
            str(candidate),
        ],
        raw / "terminal_checks",
        deadline=120,
    )
    result["receipts"].append(receipt)
    result["passed"] = result["passed"] and receipt["passed"]
    c.atomic_json(raw / "terminal_validation.json", result)
    return result


_runtime.terminal = terminal
main = _runtime.main
