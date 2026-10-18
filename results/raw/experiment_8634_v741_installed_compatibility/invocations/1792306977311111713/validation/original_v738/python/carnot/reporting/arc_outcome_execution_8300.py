"""REQ-VERIFY-8300: isolate existing consumers without changing their assertions.

The qualified executor retains exact evidence and atomic terminal publication.
Separate child processes change execution conditions; only actual passes qualify readiness.
"""

from collections.abc import Callable
import json
from pathlib import Path
import re
from types import FunctionType
from typing import Any, cast

from carnot.reporting import arc_outcome_execution_8286 as baseline
from carnot.reporting import arc_outcome_frontier_8300 as reader
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v686_contract_validation import coverage_complete
from carnot.reporting.v709_execution import child as original_child
from carnot.reporting.v709_execution import progress as original_progress

qualified = baseline.qualified
ROOT = reader.ROOT
CLI = "scripts/experiments/experiment_8300_v716_arc_outcome_frontier.py"
OUTPUT = ROOT / "results/experiment_8300_v716_arc_outcome_frontier.json"
TEST = "tests/python/test_arc_outcome_frontier_8300.py"
OWNED = [
    "python/carnot/reporting/arc_outcome_frontier_8300.py",
    "python/carnot/reporting/arc_outcome_execution_8300.py",
    CLI,
]
CONSUMERS = [
    "tests/python/test_arc_supervisor_frontier_8189.py",
    "tests/python/test_arc_supervisor_frontier_8202.py",
    "tests/python/test_arc_supervisor_refinement.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_arc_outcome_delta_8229.py",
    "tests/python/test_arc_outcome_frontier_8257.py",
    "tests/python/test_arc_outcome_frontier_8272.py",
]
Json = dict[str, Any]


def bind(name: str) -> Callable[..., Any]:
    """Reuse checked execution statements with private bindings to this invocation."""
    original = getattr(qualified, name)
    return cast(Callable[..., Any], FunctionType(original.__code__, vars(qualified) | globals()))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the audit remains visible while bounded children run."""
    original_progress(phase.replace("exp8257", "exp8300"), completed, pending)


def commands(private: Path) -> list[Json]:
    """Freeze the same seven consumers and measure only newly owned statements."""
    plan = baseline.commands(private)
    replacements = dict(zip(baseline.OWNED, OWNED, strict=True)) | {baseline.TEST: TEST}
    split: list[Json] = []
    for spec in plan:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        if spec["name"] != "affected_consumers":
            split.append(spec)
            continue
        common = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
        ]
        split.append(
            dict(
                spec,
                name="consumer_collection",
                deadline_s=60,
                argv=[*common, "--collect-only", "-q", *CONSUMERS],
            )
        )
        for index, path in enumerate(CONSUMERS):
            split.append(
                dict(
                    spec,
                    name=f"consumer_{index}",
                    deadline_s=240,
                    argv=[*common, "-vv", path, "--basetemp=" + str(private / f"consumer_{index}")],
                )
            )
    config = private / "coverage.ini"
    content = config.read_text()
    for old, new in replacements.items():
        content = content.replace(old, new)
    config.write_text(content)
    return split


def child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Bind each split run to collected identities; a silent pass proves no assertion."""
    row = original_child(name, argv, logs, **kwargs)
    if name.startswith("consumer_") and name != "consumer_collection":
        path = CONSUMERS[int(name.split("_")[1])]
        collected = (logs / "consumer_collection.stdout").read_text().splitlines()
        expected = sorted(p.strip() for p in collected if p.startswith(path + "::"))
        observed = re.findall(
            r"^(tests/python/\S+::\S+) (PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)",
            Path(row["stdout_path"]).read_text(),
            re.MULTILINE,
        )
        identities = sorted(p for p, _ in observed)
        passed = sorted(p for p, status in observed if status == "PASSED")
        row.update(
            collected_test_ids=expected,
            completed_test_ids=identities,
            passed_test_ids=passed,
            collection_equivalent=bool(expected) and expected == identities,
        )
        row["passed"] = row["passed"] and row["collection_equivalent"] and passed == expected
        atomic_json(logs / (name + ".receipt.json"), row)
    return row


def preconditions(private: Path) -> Json:
    """Hash the consumer files and required tools before the qualified reader measures."""
    available = baseline.preconditions(private)
    for label in [
        *OWNED,
        TEST,
        *CONSUMERS,
        str(reader.FRONTIER),
        "results/experiment_8286_v715_arc_outcome_frontier.json",
    ]:
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
    available["failures"] = [r for r in available["checks"] if not r["passed"]]
    return available


def consumer_summary(receipts: list[Json]) -> Json:
    """Keep collection equivalence and elapsed consumer allowance independently auditable."""
    runs = [
        r
        for r in receipts
        if r["name"].startswith("consumer_") and r["name"] != "consumer_collection"
    ]
    return dict(
        files=CONSUMERS,
        per_file=runs,
        collected_test_ids=sorted(p for r in runs for p in r["collected_test_ids"]),
        passed_test_ids=sorted(p for r in runs for p in r["passed_test_ids"]),
        collection_equivalent=len(runs) == 7 and all(r["collection_equivalent"] for r in runs),
        elapsed_s=sum(r["duration_s"] for r in receipts if r["name"].startswith("consumer_")),
        per_file_deadline_s=240,
        total_allowance_s=1800,
    )


def normalize_artifact_for_template_write(value: Json) -> Json:
    """Name the actual invocation and retain the historical disqualification as provenance."""
    value = dict(
        value,
        experiment_id=8300,
        experiment=8300,
        run_date="20261008",
        milestone="2026.10.716",
        schema="arc-outcome-frontier-v716",
        consumer_validation=consumer_summary(value["validation_receipts"]),
        execution_condition="Each of the same seven affected-consumer files runs in a separate unbuffered process with a 240-second deadline; isolation alone proves no test passed.",
        historical_frontier_authority="exp8272-arc-outcome-frontier",
        historical_disqualified_artifact=str(baseline.OUTPUT),
        runtime_allowances_s=dict(
            implementation_and_measurement=1800, consumers=1800, closeout=900, total=4500
        ),
        methodology="Authenticate Exp8272 and its exact receipt authority; inspect only later environment outcomes. Exp8286 remains disqualified. Split bounded consumers qualify execution separately from observational support.",
    )
    value["field_principles"].update(
        receipt_frontier="Exact qualified Exp8272 hash, clock and event frontier bound later bytes.",
        consumer_validation="Collected and passed identities bind process isolation to unchanged assertions.",
        execution_condition="Changed execution conditions require actual exits and identity equivalence.",
        historical_frontier_authority="Disqualified Exp8286 cannot advance the qualified frontier.",
        historical_disqualified_artifact="Preserve the failed primary and its log bytes as history.",
        runtime_allowances_s="Frozen bounded commands prevent expanding known failing work beyond the task cap.",
    )
    return qualified.normalize_artifact_for_template_write(value)


def replay(value: Json) -> list[str]:
    """Recompute primitive reductions and reject current identity or consumer summary tampering."""
    identity = dict(
        experiment_id=8300,
        experiment=8300,
        task_id=reader.TASK_ID,
        run_date="20261008",
        milestone="2026.10.716",
        schema="arc-outcome-frontier-v716",
    )
    errors = [
        "execution_claim:" + k for k, expected in identity.items() if value.get(k) != expected
    ]
    if value.get("consumer_validation") != consumer_summary(value["validation_receipts"]):
        errors.append("consumer_summary_drift")
    translated = dict(value, experiment_id=8257, run_date="20261007")
    return errors + cast(list[str], bind("replay")(translated))


def terminal(candidate: Path, raw: Path) -> Json:
    """Unchanged validators and a fresh current CLI check identical private candidate bytes."""
    return cast(Json, bind("terminal")(candidate, raw))


def execute(
    locator: Path, frontier: Path, output: Path, private: Path, *, fixture: bool = False
) -> int:
    """Publish only after checked children exit; the parent retains every failure receipt."""
    return int(bind("execute")(locator, frontier, output, private, fixture=fixture))
