"""REQ-VERIFY-8787: retain existing bounded validation and sealed publication."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from carnot.reporting import hardware_operation_boundary_8787 as report
from carnot.reporting import hardware_operation_runner_8774 as prior
from carnot.reporting import hardware_operation_runner_8721 as original
from carnot.reporting import hardware_operation_runner_8707 as checks
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
r = namespace(original, "operation_runner_8787")
r.retained = namespace(checks, "operation_checks_8787")
r.e, r.i = report.e, report.source.i
r.BASE_VALIDATE, r.BASE_REPLAY = r.retained.validate, r.retained.replay
r.BASE_TERMINAL, r.BASE_RUN = r.retained.terminal, r.retained.run
BASE_MEASURE, BASE_VALIDATE = r.measure, r.validate
CONSUMERS = prior.CONSUMERS
HEALTH = (
    report.e.ROOT
    / "results/raw"
    / report.e.NAME
    / "development/health/full_python_suite.receipt.json"
)


def measure(raw: Path, date: str, root: Path, private: bool) -> Json:
    """Freeze node inventories and cost inputs before independently admitting any branch."""
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
    frozen = dict(
        method=report.METHOD,
        nodes=nodes,
        resource_projection=dict(
            model_bytes=0,
            gpu_bytes=0,
            minimum_disk_bytes=2**30,
            maximum_reader_s=3600,
            validation_reserve_s=900,
        ),
        sources=[r.i.reference(report.e.ROOT / p) for p in report.e.OWNED],
        note=r.i.reference(
            report.e.ROOT / "docs/research-notes/v752-hardware-operation-boundary.md"
        ),
        inputs=[
            r.i.reference(root / p)
            for p in [
                report.source.BOARD,
                "results/experiment_8780_v752_causal_template_learning.json",
                "results/experiment_8784_v752_durable_cost_components.json",
                "results/experiment_8785_v752_memory_benefit_audit.json",
                "results/experiment_8757_v750_feature_reuse_cost.json",
                "results/experiment_8707_v746_hardware_operation_boundary.json",
            ]
        ],
    )
    atomic_json(raw / "method.json", frozen)
    work: Json = BASE_MEASURE(raw, date, root, private)
    for ref in frozen["inputs"] + frozen["sources"]:
        r.i.load_bytes(ref)
    work["source_refs"] += frozen["inputs"] + frozen["sources"] + [frozen["note"]]
    work["source_refs"] += [r.i.reference(raw / p) for p in ["frozen_nodes.json", "method.json"]]
    work["frozen_test_nodes"] = nodes
    work["repository_health_reference"] = r.i.reference(
        raw / "repository_health.not_run" if private else HEALTH
    )
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
    """Reuse complete private controls and require the frozen node union to match."""
    receipts: list[Json] = BASE_VALIDATE(raw)
    import json

    frozen = r.i.load(r.i.reference(raw / "frozen_nodes.json"))["nodes"]
    actual = json.loads((raw / "validation/collected_nodes.json").read_bytes())
    receipts.append(
        dict(
            name="frozen_test_node_union",
            passed=bool(frozen) and frozen == actual,
            references=[
                r.i.reference(raw / "frozen_nodes.json"),
                r.i.reference(raw / "validation/collected_nodes.json"),
            ],
        )
    )
    return receipts


r.measure, r.validate = measure, validate


def main(argv: list[str] | None = None) -> int:
    """The common CLI enforces private roots, calendar dates and heartbeat polling."""
    return int(r.main(["--date", "20261104", *(sys.argv[1:] if argv is None else argv)]))
