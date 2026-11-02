"""REQ-REPORT-8679: authenticate costs independently and keep device claims conditional."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import hardware_operation_evidence_8665 as old
from carnot.reporting import hardware_boundary_costs_8679 as costs
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import reader_receipt, validate_primary

Json = dict[str, Any]
ROOT, reference, checked, checksum = old.ROOT, old.reference, old.checked, old.checksum
NAME, TASK = (
    "experiment_8679_v744_hardware_operation_boundary",
    "exp8679-hardware-operation-boundary",
)
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_operation_boundary_8679.py"
OWNED = [
    f"python/carnot/reporting/{p}_8679.py"
    for p in ["hardware_operation_boundary", "hardware_boundary_costs", "hardware_boundary_runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
PRODUCERS = [
    (8671, "v744", "causal_brier_stream", "learning_cost_ready_score", "delayed_learning"),
    (
        8659,
        "v743",
        "complete_source_comparison",
        "source_comparison_ready_score",
        "source_extraction",
    ),
    (8675, "v744", "installed_batch_cost", "batch_cost_ready_score", "installed_inference"),
]
SOURCE_PIN = "sha256:55b263c77701f25e828cdc2522e90286f23521c10b386eb45011ab1b7fc71e6f"
ORIGINAL_AUTHENTICATE = old.authenticate
HISTORICAL_INPUTS = {
    "results/experiment_8665_v743_hardware_operation_evidence.json": "sha256:436a721cdb3430fcc0ff79ee18bd747158356e9b320f93961f6605fad7a5548f",
    "results/experiment_8661_v743_installed_complete_cost.json": "sha256:6670350979ad4fdfcaac26996a1a8705d061d2f4d3cf9dc494621c69cafae4d8",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts because a supervisor must distinguish slow work from silence."""
    print(f"[exp8679] phase={phase} completed={completed} pending={pending}", flush=True)


def admit(root: Path, producer: tuple[int, str, str, str, str]) -> Json:
    """Each original producer qualifies alone; an absent sibling cannot veto valid costs."""
    number, version, suffix, field, boundary = producer
    path = root / f"results/experiment_{number}_{version}_{suffix}.json"
    result: Json = dict(
        experiment_id=number,
        boundary=boundary,
        qualified=False,
        cells=[],
        refs=[],
        checks=[],
        acceptance_shape=None,
        missing_reason=None,
    )
    observed, expected, state = None, 1, "missing_file"
    try:
        value = json.loads(path.read_bytes())
        result["refs"].append(reference(path))
        validate_primary(value, path)
        observed = value.get(field)
        state = (
            "missing_field"
            if field not in value
            else "observed_zero"
            if type(observed) is int and observed == 0
            else "observed"
        )
        if type(observed) is not int or observed != 1:
            raise ValueError(field)
        field, expected, observed = "qualified_producer_checks", True, False
        shape = costs.acceptance(value["acceptance_gates"])
        result["acceptance_shape"] = shape
        findings = costs.flags(value.get("adversarial_findings", []))
        if (
            value["required_checks_passed"] is not True
            or value.get("flagged_adversarial") is not False
            or value["task_id"] != f"exp{number}-{suffix.replace('_', '-')}"
            or value["verdict_class"] not in {"null", "positive", "circular_positive"}
            or not shape["qualified"]
            or any(f["severity"] != "info" for f in findings)
        ):
            raise ValueError(field)
        field, state = "authenticated_terminal_producer", "changed_bytes"
        if number == 8659 and root.resolve() == ROOT:
            checked(dict(path=str(path), sha256=SOURCE_PIN))
        result["refs"].append(old.receipt(path))
        terminal = reader_receipt(value["task_id"], path.parent, field=producer[3])
        if not terminal["passed"] or terminal["gate_path"] != str(path.absolute()):
            raise ValueError("terminal_reader_identity")
        result["terminal_reader"] = terminal
        field, state, observed = "operation_cost_operands", "changed_bytes", None
        if number == 8659 and "primitive_reference" in value:
            result["cells"] = costs.source_cells(value, result["refs"])
        else:
            closure = value.get("operation_cost_closure", value.get("complete_operation_spans"))
            if not isinstance(closure, list) or not closure:
                state = (
                    "missing_field" if "operation_cost_closure" not in value else "observed_empty"
                )
                raise ValueError("missing_complete_operation_cost_closure_or_endpoint_references")
            for item in closure:
                ref = item.get("primitive_reference", item)
                primitive = checked(ref)
                if not primitive.resolve().is_relative_to(
                    (path.parent / "raw" / path.stem).resolve()
                ):
                    raise ValueError("foreign_primitive")
                cell = costs.reduce_spans(json.loads(primitive.read_bytes()), boundary)
                result["cells"].append(costs.conditional(dict(cell, primitive_reference=ref)))
                result["refs"].append(ref)
        cells = result["cells"]
        if len({(c["cohort"], c["arm"]) for c in cells}) != len(cells):
            raise ValueError("duplicate_cohort_arm")
        result.update(qualified=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(cells=[], missing_reason=str(error))
        result["checks"].append(
            dict(
                old.previous.failure(path, f"exp{number}", field, expected, observed),
                passed=False,
                observed_state=state,
                reason=str(error),
            )
        )
    return result


def authenticate(root: Path) -> Json:
    """Original pinned board authority remains separate from new science producer status."""
    board = ORIGINAL_AUTHENTICATE(root)["board"]
    for relative, digest in HISTORICAL_INPUTS.items():
        path = root / relative
        try:
            checked(dict(path=str(path), sha256=digest))
            board["refs"] += [reference(path), old.receipt(path)]
        except (OSError, ValueError, KeyError, TypeError) as error:
            board["checks"].append(
                dict(
                    old.previous.failure(
                        path,
                        path.stem,
                        "original_sha256",
                        digest,
                        reference(path)["sha256"] if path.is_file() else None,
                    ),
                    observed_state="changed_bytes" if path.is_file() else "missing_file",
                    reason=str(error),
                )
            )
            board["qualified"] = False
    return dict(board=board, branches=[admit(root, p) for p in PRODUCERS])


def measure(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Freeze original operands before any validation and seal outside candidate-owned work."""
    progress("preconditions_and_independent_admission", 0, 3)
    with (
        patch.object(old, "authenticate", authenticate),
        patch.object(old, "OWNED", OWNED),
        patch.object(old, "TEST", TEST),
    ):
        work = old.measure(root, raw, private, run_date)
    work["code_config_hashes"] += [
        reference(ROOT / p)
        for p in [
            "docs/research-notes/v744-hardware-operation-boundary.md",
            "openspec/change-proposals/v743-methods-protocol.json",
            "python/carnot/reporting/hardware_operation_evidence_8665.py",
            "python/carnot/reporting/hardware_evidence_validation_8665.py",
            "python/carnot/reporting/hardware_evidence_runner_8651.py",
        ]
    ]
    work["preconditions_checked"]["measurement_budget_s"] = 3600
    filesystem = subprocess.run(
        ["findmnt", "-T", str(private), "-n", "-o", "FSTYPE,SOURCE"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    if filesystem.split()[0] in {"tmpfs", "ramfs"}:
        raise ValueError("disk_backed_scratch_required")
    work["preconditions_checked"]["scratch_filesystem"] = filesystem
    work["acceptance_shape_regression"] = costs.reproduce_shapes(raw)
    atomic_json(raw / "measurement.json", work)
    atomic_json(raw / "measurement.seal.json", reference(raw / "measurement.json"))
    (raw / "measurement.json").chmod(0o444)
    (raw / "measurement.seal.json").chmod(0o444)
    progress("immutable_measurement_complete", 3, 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """A qualified map does not establish physical readiness or learning acceleration."""
    value = old.build(work, receipts, raw, output)
    value.update(
        experiment_id=8679,
        task_id=TASK,
        milestone="2026.11.744",
        random_seed=7448679,
        honest_verdict=f"complete_{value['verdict_class']}_v744_hardware_operation_boundary",
        inference_substrate="aggregation_from_upstream_artifacts",
        device_commands_executed=[],
    )
    branches = work["evidence"]["branches"]
    value["acceptance_shape_rows"] = [
        dict(experiment_id=b["experiment_id"], **b["acceptance_shape"])
        for b in branches
        if b["acceptance_shape"] is not None
    ]
    value["acceptance_shape_rows"] += work["acceptance_shape_regression"]
    value["raw_shard_hashes"] += [
        r["primitive_reference"] for r in work["acceptance_shape_regression"]
    ]
    cells = value["complete_boundary_rows"]
    value["conditional_acceleration_bounds"] = [
        dict(
            c,
            interpretation="Conditional 1/(1-f) only; batch transport is not device arithmetic acceleration.",
        )
        for c in cells
    ]
    value["conditional_ceiling_rows"] = value["conditional_acceleration_bounds"]
    value["missing_cost_rows"] += [
        dict(
            producer=c["producer"],
            cohort=c["cohort"],
            arm=c["arm"],
            missing_reason=c["missing_reason"],
            required_costs=c["required_costs"],
        )
        for c in cells
        if c["missing_reason"]
    ]
    value["hardware_100x_requirements"].update(
        target_met=False,
        required_total_learning_ratio=100,
        necessary_unaccelerated_fraction_below=0.01,
        required_costs=["transfer", "sparse_coordinate_updates", "durable_store", "feedback_drain"],
        learning_execution="CPU now; potential fixed-point FPGA only after range/error and typed-decision parity",
        required_changes=[
            "Measure complete causal learning, retention and durable resume costs before claiming benefit.",
            "Implement and authenticate a compatible spline/update operation; current KV260 quadratic Ising is incompatible.",
            "Bound fixed-point error, preserve calibrated typed decisions, measure memory/transfer/store and full-system clocks.",
            "Reduce the complete unaccelerated fraction below 1 percent and demonstrate at least 100x measured learning cost parity.",
        ],
        literature_numbers_are_local_evidence=False,
        transport_gain_is_arithmetic_acceleration=False,
    )
    value["preserved_quarantines"]["Exp8664"] = dict(
        flagged_adversarial=True,
        verdict_class="disqualified",
        reason="Repeated observation callbacks violated original observer checks; later clean scans cannot erase the stamp.",
    )
    value["preserved_historical_outcomes"].append(
        "V743 ended fourteen scheduling slots. Exp8654 passed39 core nodes then changed sources; readiness stayed zero. The25-node remainder and learning never ran. Exp8656/8659 qualified full-source processing and recovered55 of190 responses in128 exposed groups. Exp8660 compatibility qualified. Exp8661 packaging noninferiority passed, complete10x native/Python NFR-01 failed. Exp8663 reset repair does not qualify flagged Exp8664."
    )
    value["future_hardware_commands"] = dict(
        device="KV260",
        tools=["SSH", "scp", "xmutil"],
        prerequisite="An authenticated compatible workload and kernel; no commands executed now.",
    )
    value["acquisition_priority_delta"] = dict(
        changed=False, reason="No new evidence-backed acquisition prerequisite."
    )
    value["methodology_note"] = (
        "Read-only source, learning and batching cost aggregation. No model load, numerical evaluation, benchmark, device probe or firmware operation. Original source clocks exclude imports. Unavailable compatible implementations and inner arithmetic endpoints keep conditional bounds null."
    )
    value["measurement_seal_path"] = str(raw / "measurement.seal.json")
    value["field_principles"] = {
        k: "Invocation-bound immutable evidence; readiness, benefit and physical execution remain separate. Missing operands are null."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """An independent measurement seal defeats rehashed candidate/work substitutions."""
    try:
        value = json.loads(path.read_bytes())
        work_path = checked(value["work_reference"])
        raw = work_path.parent
        sealed_ref = json.loads((raw / "measurement.seal.json").read_bytes())
        if sealed_ref["path"] != str((raw / "measurement.json").absolute()):
            return False
        measured = json.loads(checked(sealed_ref).read_bytes())
        work = json.loads(work_path.read_bytes())
        if any(
            work[k] != measured[k]
            for k in [
                "evidence",
                "snapshots",
                "root",
                "run_date",
                "code_config_hashes",
                "preconditions_checked",
                "acceptance_shape_regression",
            ]
        ):
            return False
        fresh = authenticate(Path(work["root"]))
        if fresh["board"] != work["evidence"]["board"] or any(
            observed != original
            for observed, original in zip(
                fresh["branches"], work["evidence"]["branches"], strict=True
            )
            if original["refs"] or original["qualified"]
        ):
            return False
        for snapshot in work["snapshots"]:
            if snapshot["sha256"] != snapshot["original_reference"]["sha256"]:
                return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            checked(ref)
        return bool(
            build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
