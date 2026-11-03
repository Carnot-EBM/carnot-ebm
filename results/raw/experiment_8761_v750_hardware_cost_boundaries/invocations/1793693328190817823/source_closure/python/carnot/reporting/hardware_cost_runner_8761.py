"""REQ-VERIFY-8761: reuse sealed publication with complete scoped validation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from carnot.reporting import hardware_cost_boundaries_8761 as e8761
from carnot.reporting import hardware_operation_runner_8721 as original
from carnot.reporting import hardware_operation_runner_8707 as original_checks
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
r = namespace(original, "cost_runner_8761")
r.retained = namespace(original_checks, "cost_checks_8761")
r.e, r.i = e8761.e, e8761.source.i
r.BASE_VALIDATE, r.BASE_REPLAY = r.retained.validate, r.retained.replay
r.BASE_TERMINAL, r.BASE_RUN = r.retained.terminal, r.retained.run
BASE_MEASURE = r.measure
BASE_VALIDATE = r.validate
HEALTH = (
    e8761.e.ROOT
    / "results/raw"
    / e8761.e.NAME
    / "development/health/full_python_suite.receipt.json"
)
CONSUMERS = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    "tests/python/test_primary_publication_7928.py",
]


def measure(raw: Path, date: str, root: Path, private: bool) -> Json:
    """Collect and hash exact test nodes before reading the measured cost operands."""
    receipt = child(
        "freeze_nodes",
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            "--collect-only",
            *e8761.e.TESTS,
            *CONSUMERS,
        ],
        raw / "freeze",
        deadline=120,
    )
    nodes = [
        line
        for line in Path(receipt["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    atomic_json(
        raw / "frozen_nodes.json", dict(nodes=nodes, sha256=canonical_hash(nodes), receipt=receipt)
    )
    work: Json = BASE_MEASURE(raw, date, root, private)
    work["frozen_test_nodes"] = nodes
    work["source_refs"] += [r.i.reference(raw / "frozen_nodes.json")]
    work["repository_health_reference"] = r.i.reference(HEALTH)
    work["source_refs"].append(work["repository_health_reference"])
    if not receipt["passed"] or not nodes:
        work["resource_failures"].append(
            dict(
                upstream=e8761.e.TASK,
                **r.i.reference(raw / "frozen_nodes.json"),
                check="test_collection",
                field="passed",
                operator="==",
                expected=True,
                observed=False,
                passed=False,
                operand_state="failed_owned_validation",
            )
        )
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=r.i.reference(raw / "measurement.json"), code_refs=work["code_refs"]),
    )
    return work


def validate(raw: Path) -> list[Json]:
    """Require the entire scoped union; repository health never enters acceptance."""
    receipts: list[Json] = BASE_VALIDATE(raw)
    frozen = r.i.load(r.i.reference(raw / "frozen_nodes.json"))["nodes"]
    actual = r.i.load(r.i.reference(raw / "measurement.json"))["frozen_test_nodes"]
    import json

    collected = json.loads((raw / "validation/collected_nodes.json").read_bytes())
    receipts.append(
        dict(
            name="frozen_test_node_union",
            passed=frozen == actual == collected and bool(frozen),
            references=[
                r.i.reference(raw / "frozen_nodes.json"),
                r.i.reference(raw / "validation/collected_nodes.json"),
            ],
        )
    )
    return receipts


r.measure, r.validate = measure, validate


def main(argv: list[str] | None = None) -> int:
    """Set the current run date while retaining shared guards and calendar checks."""
    return int(r.main(["--date", "20261103", *(sys.argv[1:] if argv is None else argv)]))
