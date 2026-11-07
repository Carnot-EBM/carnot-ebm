"""REQ-REPORT-8825: map source operations without inventing measured fractions."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8825_v755_hardware_operation_map"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/hardware_operation_map_8825.py",
    "python/carnot/reporting/hardware_operation_runner_8825.py",
    CLI,
]
TEST = "tests/python/test_hardware_operation_map_8825.py"
SOURCES = [
    "research-hardware-wishlist.md",
    "ops/hardware-bringup-prep.md",
    "hardware/kv260/discrete_sb_256.v",
    "research-references.md",
    "python/carnot/memory/causal_memory_8802.py",
    "python/carnot/models/gibbs/__init__.py",
    "python/carnot/reporting/learning_signal_diagnosis_8815.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "scripts/experiment_template.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "pyproject.toml",
    "docs/research-notes/v755-hardware-operation-map.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/hardware-operation-boundary/spec.md",
    *OWNED,
    TEST,
]
HISTORY = {
    "results/experiment_8230_v711_kv260_workload_boundary.json": "sha256:dc98bfa0f84e104175aa8fbb7f77c7e6450ecdb0b8f26f5c4c95f262f4f1f75b",
    "results/experiment_8259_v713_polarfire_dispatch_qualification.json": "sha256:9933e509b6d05112843986d8271455df2194dc74f4000a2e68abbf52e0d04589",
    "results/experiment_8302_v716_gatemate_physical_delta.json": "sha256:fb847291bb18a922ed944e08f38b98d486a6dbc1dce29cf322155a7f61ee6f12",
    "results/experiment_8808_v754_archived_observer_costs.json": "sha256:b02bc692bf349161c46c76d78752eba701c6d376abaacdd2f9113918a0a2904b",
    "openspec/change-proposals/research-roadmap-v754-preserved-20261104.md": "sha256:021962592d30e304c97c7d45b99ffef9997837e4a21b395893c3d76316a56f17",
}
BRANCHES = {
    "learning": (
        "results/experiment_8820_v755_causal_objective_streams.json",
        "stream_rows_ready_score",
    ),
    "cost": ("results/experiment_8822_v755_parent_window_cost.json", "whole_cost_ready_score"),
}
OPERATIONS = {
    "categorical_energy": "float32; three 9-input candidates, 4 hidden SiLU units; 45 parameters",
    "dense_forward": "float32 dense 9x4 and 4x1 multiplies; 120 scalar multiplies per three candidates",
    "three_label_normalization": "stable logsumexp/softmax of three logits; exact enumeration",
    "dense_backpropagation": "float32 gradients and optimizer state; no established spline-local FPGA implementation",
    "template_joins": "exact symbolic joins; variable bindings and template store; content-dependent counts",
    "durable_bytes": "serialized parameters, templates, acknowledgement and fsync/restart state; bytes unmeasured",
    "state_transfer": "input, parameters, output and gradient buffers; wire precision/layout unmeasured",
    "sampler_calls": "zero owned MCMC calls; categorical enumeration supplies no sampler-eligible work",
    "quadratic_ising": "historical quadratic fabric only; independent from categorical learning",
    "cpu_dispatch": "historical authenticated Linux CPU dispatch; categorical workload and timing unmeasured",
}
BOARDS = {
    "KV260": dict(
        boundary="quadratic fabric k_max<=5; SSH-only future access",
        next_evidence="compatible kernel, precision parity and complete SSH/host/transfer/durable timing",
    ),
    "PolarFire": dict(
        boundary="authenticated Linux CPU dispatch; fabric unqualified",
        next_evidence="categorical CPU dispatch parity and complete costs; separate fabric qualification",
    ),
    "GateMate": dict(
        boundary="unchanged 0xffffffff JTAG blocker",
        next_evidence="dated cable/port/power change, then IDCODE0x20000001 and ordered n16 smoke",
        retry_allowed=False,
    ),
}


def progress(phase: str) -> None:
    """Flush boundaries because quiet source audits must still show their progress."""
    print(f"[exp8825] phase={phase} completed=0 pending=0", flush=True)


def snapshot(raw: Path) -> tuple[Json, Json]:
    """Keep exact source bytes so a later document append does not change history."""
    refs, evidence = {}, {}
    for name in [*SOURCES, *HISTORY]:
        path = raw / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        digest = sha256_file(path)
        if name in HISTORY and digest != HISTORY[name]:
            raise ValueError("historical_authority_drift:" + name)
        refs[name] = dict(path=str(path.absolute()), sha256=digest)
    for branch, (name, _) in BRANCHES.items():
        path = ROOT / name
        ref: Json = dict(path=str(path), sha256=None, value=None)
        if path.is_file():
            retained = raw / "inputs" / name
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, retained)
            ref.update(
                path=str(retained.absolute()),
                sha256=sha256_file(retained),
                value=json.loads(retained.read_bytes()),
            )
        evidence[branch] = ref
    return evidence, refs


def reduce(evidence: Json) -> Json:
    """Admit each branch independently; static arithmetic never becomes a time share."""
    gates, budgets, targets = [], [], []
    for branch, (_, field) in BRANCHES.items():
        ref = evidence.get(branch, {})
        value = ref.get("value")
        reason = "missing_file"
        if value is not None:
            reason = "missing_field" if field not in value else "observed_zero"
            if value.get(field) == 1:
                valid = (
                    value.get("required_checks_passed") is True
                    and value.get("flagged_adversarial") is False
                    and str(value.get("honest_verdict", "")).startswith("complete_")
                    and value.get("verdict_class") in {"positive", "circular_positive", "null"}
                )
                reason = "admitted" if valid else "rejected_producer"
                if valid and branch == "cost":
                    rows = value.get("complete_operation_rows", [])
                    valid = (
                        isinstance(rows, list)
                        and bool(rows)
                        and all(
                            isinstance(r, dict)
                            and isinstance(r.get("unit_id"), str)
                            and bool(r["unit_id"])
                            and isinstance(r.get("total_s"), (int, float))
                            and not isinstance(r["total_s"], bool)
                            and math.isfinite(r["total_s"])
                            and r["total_s"] > 0
                            and r.get("denominator") == 1
                            for r in rows
                        )
                    )
                    reason = "admitted" if valid else "missing_field"
                    if valid:
                        budgets = [
                            dict(**r, source_path=ref["path"], source_sha256=ref["sha256"])
                            for r in rows
                        ]
                        targets = [
                            dict(
                                unit_id=r["unit_id"],
                                complete_operation_s=r["total_s"],
                                target_latency_s=r["total_s"] / 100,
                                claim="future100x requirement; no speedup prediction",
                            )
                            for r in rows
                        ]
        if reason != "admitted":
            gates.append(
                dict(
                    check="producer_admission",
                    upstream=branch,
                    artifact_field=field,
                    expected=1,
                    observed=None if value is None else value.get(field),
                    path=ref.get("path", str(ROOT / BRANCHES[branch][0])),
                    sha256=ref.get("sha256"),
                    missing_reason=reason,
                )
            )
    rows = []
    for board in BOARDS:
        for operation, requirement in OPERATIONS.items():
            status = (
                "supported"
                if (operation == "quadratic_ising" and board == "KV260")
                or (operation == "cpu_dispatch" and board == "PolarFire")
                else "unsupported"
                if operation == "dense_backpropagation" or (board == "GateMate")
                else "unmeasured"
            )
            rows.append(
                dict(
                    unit_id=board + "/" + operation,
                    cluster_id=board,
                    arm=operation,
                    seed=0,
                    metric=int(status == "supported"),
                    denominator=1,
                    status="completed",
                    missing_reason=None,
                    operation_status=status,
                    requirements=dict(precision_state_transfer=requirement),
                    boundary=BOARDS[board]["boundary"],
                    measured_time_s=None,
                )
            )
    for name, field, observed, expected, reason in [
        (
            "results/experiment_8808_v754_archived_observer_costs.json",
            "component_cost_ready_score",
            0,
            1,
            "observed_zero",
        ),
        (
            "results/experiment_8302_v716_gatemate_physical_delta.json",
            "dated_physical_change_receipt",
            None,
            "dated physical change before IDCODE/n16 smoke",
            "missing_field",
        ),
    ]:
        gates.append(
            dict(
                check="historical_boundary",
                upstream="archived_profiler" if observed == 0 else "GateMate",
                artifact_field=field,
                expected=expected,
                observed=observed,
                path=str(ROOT / name),
                sha256=HISTORY.get(name),
                missing_reason=reason,
            )
        )
    return dict(
        board_operation_rows=rows,
        rows=rows,
        supported_operation_counts={
            b: sum(r["operation_status"] == "supported" for r in rows if r["cluster_id"] == b)
            for b in BOARDS
        },
        operation_inventory=dict(
            dense_forward=dict(
                scalar_multiplies=120,
                formula="3*(9*4+4)",
                scope="static per decision; excludes backprop compiler and activation arithmetic",
            ),
            categorical_energy=dict(
                candidates=3, hidden_silu=12, parameter_scalars=45, parameter_float32_bytes=180
            ),
            three_label_normalization=dict(
                candidates=3,
                exponentials=3,
                denominator_adds=2,
                probability_divisions=3,
                logsumexp_logarithms=1,
                scope="conceptual stable normalization; fused implementation counts unmeasured",
            ),
            template_joins=dict(count=None, scope="content dependent; no learner executed"),
            durable_bytes=dict(
                count=None, scope="180 parameter bytes exclude JSON/templates/WAL/fsync"
            ),
            sampler_calls=dict(
                count=0, scope="owned audit calls only; upstream absence is not zero"
            ),
            dense_backpropagation=dict(
                count=None, scope="AD compiler operations and optimizer state unmeasured"
            ),
        ),
        gate_check_summary=gates,
        measured_complete_cost_refs=budgets,
        target_latency_requirements=targets,
        unknown_costs=[
            "complete learner costs when absent",
            "transfer",
            "startup",
            "durability",
            "backpropagation",
            "template joins",
            "compatible measured fraction",
        ],
        physical_evidence_state=BOARDS,
        sampling_call_count=0,
        rejected_profiler_evidence=dict(
            component_cost_ready_score=0,
            source_path="results/experiment_8808_v754_archived_observer_costs.json",
            source_sha256=HISTORY.get("results/experiment_8808_v754_archived_observer_costs.json"),
            use="historical rejection only; supplies no compatible fraction",
        ),
    )


def build(evidence: Json, refs: Json, validation: Json, start: int, end: int, date: str) -> Json:
    """Administrative mapping readiness records valid audit evidence, never benefit."""
    reduced = reduce(evidence)
    valid = validation["passed"]
    verdict = (
        "disqualified" if not valid else "blocked" if reduced["gate_check_summary"] else "null"
    )
    receipt = build_current_work_receipt(
        run_id=str(start),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_verifier",
        inference_substrate_details=dict(
            operation="static source/design audit; no model or device execution"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(
                phase="operation_map_invocation_including_validation",
                start_s=0,
                end_s=(end - start) / 1e9,
            )
        ],
    )
    value = dict(
        **reduced,
        experiment_id=8825,
        task_id="exp8825-hardware-operation-map",
        milestone="2026.11.755",
        run_date=date,
        honest_verdict="complete_" + verdict + "_hardware_operation_map",
        verdict_class=verdict,
        hardware_map_ready_score=int(valid),
        **receipt,
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=False,
        random_seed=0,
        reproducibility_checksum=canonical_hash(
            dict(evidence=evidence, sources=refs, operations=OPERATIONS)
        ),
        source_artifact_hashes=refs,
        code_config_hashes={p: refs[p] for p in OWNED},
        raw_shard_hashes=refs,
        preconditions_checked=[
            dict(check="audit_only_no_probe_no_flash", passed=True),
            dict(check="immutable_historical_authorities", passed=True),
        ],
        intended_count=len(reduced["rows"]),
        completed_count=len(reduced["rows"]),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        acceptance_gates=dict(
            three_boards_audited=True, owned_validation=valid, physical_acceleration=False
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        validation_state=validation,
        owned_statement_coverage=validation["coverage"],
        exposure_scope="private source/design audit; no natural text or hidden game evidence",
        independent_generalization_score=0,
        cited_upstream_artifacts=evidence,
        flagged_adversarial=False,
        methodology_note="Static source counts are distinct from whole-operation parent timings. No MCMC, hardware probe, flash, Amdahl curve or speedup claim. CPU counters are immediate; GPU batch training is a feasible future path. hardware100x remains an aspiration requiring a measured compatible fraction.",
    )
    value["field_principles"] = {
        k: "All three board boundaries stay separate. Source counts are not timing; unknown is not zero. Readiness is evidence validity only."
        for k in value
    }
    value["field_principles"]["field_principles"] = "Every extra field has a stated evidence limit."
    return value


def replay(path: Path) -> Json:
    """Recompute source-bound reductions so consistently rehashed row changes still fail."""
    value = json.loads(path.read_bytes())
    for name, ref in value["source_artifact_hashes"].items():
        if sha256_file(Path(ref["path"])) != ref["sha256"] or (
            name in HISTORY and ref["sha256"] != HISTORY[name]
        ):
            raise ValueError("source_hash_drift:" + name)
    evidence = value["cited_upstream_artifacts"]
    for ref in evidence.values():
        if ref["value"] is not None and (
            sha256_file(Path(ref["path"])) != ref["sha256"]
            or json.loads(Path(ref["path"]).read_bytes()) != ref["value"]
        ):
            raise ValueError("producer_hash_drift")
    if validate_current_work_receipt(value):
        raise ValueError("current_receipt_drift")
    expected = build(
        evidence,
        value["source_artifact_hashes"],
        value["validation_state"],
        value["started_monotonic_ns"],
        value["ended_monotonic_ns"],
        value["run_date"],
    )
    expected["current_owner_pid"] = value["current_owner_pid"]
    if expected != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, checked_sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Keep private fixture execution separate from the validated public result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    progress("preconditions")
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps(dict(passed=False, reason=str(exc))), flush=True)
            return 1
    from carnot.reporting.hardware_operation_runner_8825 import publish, validate

    start = time.monotonic_ns()
    output = (args.private_output or ROOT / "results" / (NAME + ".json")).absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(start)
    raw.mkdir(parents=True, mode=0o700)
    progress("snapshot_before")
    evidence, refs = snapshot(raw)
    progress("snapshot_after")
    validation = (
        dict(
            passed=False,
            receipts=[],
            coverage=dict(percent=0, reason="private fixture has no validation claim"),
        )
        if args.private_output
        else validate(raw)
    )
    progress("reduce_before")
    value = build(evidence, refs, validation, start, time.monotonic_ns(), args.date)
    progress("reduce_after")
    publish(output, value, raw, private=bool(args.private_output))
    progress("complete")
    return 0
