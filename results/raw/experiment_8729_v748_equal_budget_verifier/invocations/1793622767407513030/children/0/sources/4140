"""REQ-VERIFY-8286: reuse qualified execution without altering historical modules.

Private function namespaces retain the original validation and publication
statements while binding this invocation's identity, paths and reader.
"""

from collections.abc import Callable
from pathlib import Path
from types import FunctionType
from typing import Any, cast

from carnot.reporting import arc_outcome_execution_8272 as baseline
from carnot.reporting import arc_outcome_frontier_8286 as reader
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.v686_contract_validation import coverage_complete
from carnot.reporting.v709_execution import progress as original_progress

qualified = baseline.qualified
ROOT = reader.ROOT
CLI = "scripts/experiments/experiment_8286_v715_arc_outcome_frontier.py"
OUTPUT = ROOT / "results/experiment_8286_v715_arc_outcome_frontier.json"
TEST = "tests/python/test_arc_outcome_frontier_8286.py"
OWNED = [
    "python/carnot/reporting/arc_outcome_frontier_8286.py",
    "python/carnot/reporting/arc_outcome_execution_8286.py",
    CLI,
]
Json = dict[str, Any]


def bind(name: str) -> Callable[..., Any]:
    """Reuse original bytecode with private globals so historical imports remain intact."""
    original = getattr(qualified, name)
    return cast(Callable[..., Any], FunctionType(original.__code__, vars(qualified) | globals()))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Phase counts identify this invocation even inside reused execution code."""
    original_progress(phase.replace("exp8257", "exp8286"), completed, pending)


def commands(private: Path) -> list[Json]:
    """Keep the qualified checks while restricting coverage to current added statements."""
    plan = baseline.commands(private)
    replacements = dict(zip(baseline.OWNED, OWNED, strict=True)) | {baseline.TEST: TEST}
    for spec in plan:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        if spec["name"] == "affected_consumers":
            spec["argv"].insert(-1, baseline.TEST)
    config = private / "coverage.ini"
    text = config.read_text()
    for old, new in replacements.items():
        text = text.replace(old, new)
    config.write_text(text)
    return plan


def preconditions(private: Path) -> Json:
    """Check current paths and tools before measuring any outcomes."""
    available = baseline.preconditions(private)
    for label in [*OWNED, TEST, str(reader.FRONTIER)]:
        path = ROOT / label
        digest = sha256_file(path) if path.is_file() else None
        available["checks"].append(
            dict(
                reader.authority.operand(path, "is_file", True, path.is_file(), digest),
                required=True,
                passed=path.is_file(),
            )
        )
        if digest:
            available["hashes"][str(path)] = digest
    available["failures"] = [row for row in available["checks"] if not row["passed"]]
    return available


def normalize_artifact_for_template_write(value: Json) -> Json:
    """Change invocation labels before terminal checks see the candidate bytes."""
    value = dict(
        value,
        experiment_id=8286,
        experiment=8286,
        run_date="20261008",
        milestone="2026.10.715",
        schema="arc-outcome-frontier-v715",
        methodology="Authenticate Exp8272 bytes and its stored frontier through the qualified receipt authority; only later environment outcomes support observational arm ordering.",
    )
    value["field_principles"]["receipt_frontier"] = (
        "Actual Exp8272 hash, clock and receipt/event IDs bound extraction of changed sources."
    )
    return qualified.normalize_artifact_for_template_write(value)


def replay(value: Json) -> list[str]:
    """Validate current identity before translating labels for the inherited replay checks."""
    identity = dict(
        experiment_id=8286,
        experiment=8286,
        task_id=reader.TASK_ID,
        run_date="20261008",
        milestone="2026.10.715",
        schema="arc-outcome-frontier-v715",
    )
    errors = [
        "execution_claim:" + key for key, expected in identity.items() if value.get(key) != expected
    ]
    translated = dict(value, experiment_id=8257, run_date="20261007")
    return errors + cast(list[str], bind("replay")(translated))


def terminal(candidate: Path, raw: Path) -> Json:
    """Unchanged validators and a fresh current CLI process inspect identical candidate bytes."""
    return cast(Json, bind("terminal")(candidate, raw))


def execute(
    locator: Path, frontier: Path, output: Path, private: Path, *, fixture: bool = False
) -> int:
    """The qualified parent owns checks, durable receipts and atomic publication."""
    return int(bind("execute")(locator, frontier, output, private, fixture=fixture))
