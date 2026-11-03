"""REQ-VERIFY-8774: reuse sealed publication and real subprocess coverage custody."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from carnot.reporting import hardware_operation_decision_8774 as report
from carnot.reporting import hardware_operation_runner_8721 as original
from carnot.reporting import hardware_operation_runner_8707 as checks
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
r = namespace(original, "operation_runner_8774")
r.retained = namespace(checks, "operation_checks_8774")
r.e, r.i = report.e, report.source.i
r.BASE_VALIDATE, r.BASE_REPLAY = r.retained.validate, r.retained.replay
r.BASE_TERMINAL, r.BASE_RUN = r.retained.terminal, r.retained.run
BASE_MEASURE, BASE_VALIDATE = r.measure, r.validate
CONSUMERS = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    "tests/python/test_primary_publication_7928.py",
]
HEALTH = (
    report.e.ROOT
    / "results/raw"
    / report.e.NAME
    / "development/health/full_python_suite.receipt.json"
)


def measure(raw: Path, date: str, root: Path, private: bool) -> Json:
    """Freeze method and nodes before the shared reader authenticates source operands."""
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
            *report.e.TESTS,
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
    note = report.e.ROOT / "docs/research-notes/v751-hardware-operation-decision.md"
    frozen = dict(
        method=report.METHOD,
        nodes=nodes,
        sources=[r.i.reference(report.e.ROOT / p) for p in report.e.OWNED],
        note=r.i.reference(note),
        inputs=[
            r.i.reference(root / p)
            for p in [
                report.source.BOARD,
                "results/experiment_8770_v751_optimized_cost_attribution.json",
                "results/experiment_8772_v751_ordinary_durable_cost.json",
                "results/experiment_8766_v751_causal_brier_stream.json",
                "results/experiment_8757_v750_feature_reuse_cost.json",
            ]
        ],
    )
    atomic_json(raw / "method.json", frozen)
    work: Json = BASE_MEASURE(raw, date, root, private)
    for ref in frozen["inputs"] + frozen["sources"]:
        r.i.load_bytes(ref)
    work["source_refs"] += frozen["inputs"] + frozen["sources"]
    work["source_refs"] += [r.i.reference(raw / p) for p in ["frozen_nodes.json", "method.json"]]
    work["frozen_test_nodes"] = nodes
    work["repository_health_reference"] = r.i.reference(HEALTH)
    work["source_refs"].append(work["repository_health_reference"])
    if not receipt["passed"] or not nodes:
        work["resource_failures"].append(
            dict(
                upstream=report.e.TASK,
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
    """Owned coverage and exact node custody are independent of repository health."""
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
    """Supply today's default without changing calendar rejection or private guards."""
    return int(r.main(["--date", "20261103", *(sys.argv[1:] if argv is None else argv)]))
