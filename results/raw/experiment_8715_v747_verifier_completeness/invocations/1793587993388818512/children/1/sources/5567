"""REQ-REPORT-8601: original operands preserve the difference between slots and science."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v737_capstone_evidence as previous
from carnot.reporting import v738_contract_methods as methods

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8601_v738_capstone", "exp8601-capstone", "2026.10.738"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v738_capstone_8601.py"
TESTS = [TEST]
OWNED = [f"python/carnot/reporting/v738_capstone{s}.py" for s in ["_evidence", "_report", ""]] + [
    CLI
]
MODEL_SPECS: list[Json] = []
__all__ = ["methods", "old"]
RUN_DATE = ""
_OUTCOME = previous.outcome
reference, read, checksum, atomic_json, gate, freeze, child, old = (
    methods.reference,
    methods.read,
    methods.checksum,
    methods.atomic_json,
    methods.gate,
    previous.freeze,
    previous.child,
    previous.old,
)
PINS = dict(
    zip(
        [8591, 8592, 8593, 8594, 8596, 8597, 8598, 8599, 8600],
        [
            "d0fc7e60a2054ecebac3b21eb6cb452a4671b8b1a24b5ce20881cbb4147ad837",
            "fc73c667216efe5dd26a0662667ab39b1b7a54901365dd0229b45c26b895dc69",
            "d19461a13336e5a9563013d49a9b14dfd78aebec8e48064e751c3f7bef44a4eb",
            "d545688cf1e01fdad7774e008e1d530ce77e227d16a272c3da14ba7013d965f0",
            "5dd00385cb4f330b900f842af1a43b580f06c5948a658c36ce3b01d35b46ac6c",
            "c3d315ded484dcb219d2102ce4de810b8bcd6e8d42a581f4f5eb794e07440ea1",
            "3d6d82f6a0d404515df46be4193a0aab6ce63dd6f8784b282c3928e26dae02fb",
            "3864ae15ee590649ef66b6bdd70920717be4226d569fa80981f039371ff90db0",
            "9a3e51adf232e530cbb4ad885b43d216b05dbe5cbb6210fc064b3223b16b2270",
        ],
        strict=True,
    )
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush each boundary so a parent can supervise CPU work without model activity."""
    print(
        f"[exp8601] phase={phase} completed={min(completed, 11)} pending={min(pending, 11)}",
        flush=True,
    )


def outcome(task: Json, primary: Json, raw: Path) -> Json:
    """Keep original readers and their rejected producer flags rather than repairing history."""
    with patch.dict(previous.PINS, PINS):
        value = dict(_OUTCOME(task, primary, raw))
    if value["producer_executed"]:
        original = read(primary)
        gates = original["acceptance_gates"]
        value["acceptance_gates"] = gates
        value["scientific_metrics"] = {
            k: v
            for k, v in original.items()
            if k
            in {
                "primary_contrast",
                "lifecycle_contrast",
                "throughput_intervals",
                "paired_session_rows",
                "nfr01_score",
                "resident_native_benefit_score",
                "refinement_eligible_score",
                "evaluation_eligible_score",
            }
        }
        if value["verdict_class"] in {"positive", "circular_positive"} and not all(gates.values()):
            value["eligible"] = False
            value["checks"].append(
                gate(Path(primary["path"]), "positive_acceptance_gates", True, gates, task["id"])
            )
    return value


def slots(root: Path, raw: Path) -> tuple[list[Json], list[Json]]:
    """Adapt only the task-map key; preserve the original resolver and missing cascade path."""
    protocol = methods.protocol()
    with (
        patch.multiple(previous, outcome=outcome, progress=progress),
        patch.object(
            previous.methods,
            "protocol",
            return_value=dict(protocol, task_map_v737=protocol["task_map_v738"]),
        ),
    ):
        rows, refs = previous.slots(root, raw)
        return list(rows), list(refs)


def sample_worker(output: Path) -> int:
    """Recompute exposed arithmetic without rerunning learning, benchmarking or model generation."""
    previous.sample_worker(output)
    diagnostic = read(reference(output))
    producer = read(methods.protocol()["original_producers"]["8582"])
    learning = read(producer["prediction_rows_reference"])["rows"][:2]
    diagnostic["learning_losses"] = [
        dict(
            unit_id=r["unit_id"],
            recomputed_brier=(r["probability"] - r["evaluator_target"]) ** 2,
            matches_original=(r["probability"] - r["evaluator_target"]) ** 2 == r["brier"],
            qualified=False,
        )
        for r in learning
    ]
    retention_ref = producer["retention_rows_reference"]
    assert methods.sha256_file(Path(retention_ref["path"])) == retention_ref["sha256"]
    retention = json.loads(Path(retention_ref["path"]).read_bytes())[:2]
    diagnostic["retention_sample"] = [
        dict(
            unit_id=r["unit_id"],
            recomputed_brier=(r["probability"] - r["target"]) ** 2,
            contrast=(r["probability"] - r["target"]) ** 2 - r["baseline_brier"],
            matches_original=(r["probability"] - r["target"]) ** 2 == r["brier"],
            qualified=False,
        )
        for r in retention
    ]
    resident = read(reference(ROOT / "results/experiment_8596_v738_resident_complete_cost.json"))
    row = resident["paired_session_rows"][0]
    primitive = read(row["primitive_reference"])
    total = sum((s["end_ns"] - s["start_ns"]) / 1e9 for s in primitive["sessions"])
    diagnostic["qualified_resident_sample"] = dict(
        source=reference(ROOT / "results/experiment_8596_v738_resident_complete_cost.json"),
        unit_id=row["unit_id"],
        recomputed_wall_s=total,
        matches_original=total == row["total_session_wall_s"],
        qualified=True,
    )
    descriptor = read(
        reference(ROOT / "results/experiment_8593_v738_resident_provenance_qualification.json")
    )
    diagnostic["numerical_head"] = read(descriptor["head_descriptor_reference"])
    atomic_json(output, diagnostic)
    progress("samples_complete", 4, 0)
    return 0


def samples(raw: Path) -> tuple[Json, Json]:
    """Use a new bounded interpreter; failed children cannot supply scientific operands."""
    with patch.multiple(previous, CLI=CLI, child=child):
        diagnostic, receipt = previous.samples(raw)
        return dict(diagnostic), dict(receipt)


def measure(root: Path, raw: Path) -> Json:
    """Bind original authority, current resources and immutable evidence to this invocation."""
    with patch.multiple(methods, OWNED=OWNED, TESTS=TESTS):
        work = dict(methods.measure(root, raw, RUN_DATE))
    for failure in work["precondition_failures"]:
        failure.update(upstream=TASK, upstream_id=TASK)
    work["slots"], refs = slots(ROOT, raw)
    work["prior_owned_attempts"] = [
        reference(p) for p in sorted(raw.parent.glob("*/aborted_attempt.json"))
    ]
    work["source_refs"] += work["prior_owned_attempts"]
    work["source_refs"] += refs + [
        reference(ROOT / p)
        for p in [
            "ops/exclusion_manifest.yaml",
            "scripts/experiment_template.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "docs/research-notes/v738-capstone.md",
        ]
    ]
    work["preconditions"].update(optional_cuda_required=False, optional_boards_required=False)
    progress("diagnostic_before", 10, 1)
    work["samples"], receipt = samples(raw / "samples")
    work["branch_receipts"] = [receipt]
    work["end_ns"] = time.monotonic_ns()
    work["duration_s"] = (work["end_ns"] - work["start_ns"]) / 1e9
    if work["duration_s"] >= 3600:
        raise TimeoutError("measurement_deadline")
    work["phase_spans"].append(dict(phase="capstone_aggregation", duration_s=work["duration_s"]))
    atomic_json(raw / "measurement.json", work)
    progress("diagnostic_after", 11, 0)
    return work
