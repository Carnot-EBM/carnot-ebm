"""REQ-REPORT-8732: keep complete throughput distinct from arithmetic and learning."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.reporting import resident_sparse_study_8731 as upstream
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import resident_update_cost_8732 as numerical

__all__ = ["upstream", "numerical", "atomic_json"]

Json = dict[str, Any]
ROOT = upstream.e.ROOT
NAME, TASK = "experiment_8732_v748_resident_update_cost", "exp8732-resident-update-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/verify/resident_update_cost_8732.py",
    "python/carnot/reporting/resident_update_cost_8732.py",
    "python/carnot/reporting/resident_update_runner_8732.py",
    CLI,
]
TESTS = ["tests/python/test_resident_update_cost_8732.py"]
ARMS = ("python", "json_native", "resident_native")
MODEL_SPECS: list[Json] = []
PRODUCER = "results/experiment_8731_v748_resident_sparse_prototype.json"
PIN = "sha256:e2c6271989cee2e644644311f2068fd0c8f66bf050d2a9b48a3828325f1f5e89"
reference, verify, read = upstream.e.reference, upstream.e.verify, upstream.e.read
progress = numerical.resident.progress


def inputs() -> Json:
    """Authenticate this branch directly, without a global learning or CUDA gate."""
    admitted: Json = upstream.inputs()
    operand = dict(path=str(ROOT / PRODUCER), sha256=PIN)
    admitted["refs"].append(operand)
    try:
        value = read(operand)
        for field, expected in [
            ("experiment_id", 8731),
            ("resident_update_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            row = dict(
                upstream.e.previous.gate(operand, field, expected),
                upstream="exp8731-resident-sparse-prototype",
            )
            if field not in value or value[field] != expected:
                row.update(
                    passed=False,
                    observed=value.get(field),
                    operand_state="missing_field" if field not in value else "observed",
                )
            if not row["passed"]:
                admitted["failures"].append(row)
        admitted["refs"] += upstream.e.previous.terminal(operand, value)
        work, checks = read(value["primitive_reference"]), read(value["validation_reference"])
        if checks["receipts"] != value["validation_receipts"] or not all(
            r["passed"] for r in checks["receipts"]
        ):
            raise ValueError("qualified_resident_checks")
        for ref in value["code_config_hashes"] + [value["loaded_extension_reference"]]:
            operand = ref
            verify(ref)
            admitted["refs"].append(ref)
        if work["private_control"] or not value["loaded_extension_reference"]["actual_loaded"]:
            raise ValueError("qualified_resident_identity")
        admitted["extension"] = value["loaded_extension_reference"]
        admitted["refs"] += [value["primitive_reference"], value["validation_reference"]]
        protocol_ref = dict(
            path=str(ROOT / upstream.e.previous.PROTOCOL), sha256=upstream.e.previous.PROTOCOL_PIN
        )
        protocol = read(protocol_ref)
        feedback = protocol["learning_registration"]["source_inputs"]
        durable_producer = next(
            r
            for r in feedback
            if r["path"].endswith("experiment_8433_v726_block_checkpoint_qualification.json")
        )
        durable = read(durable_producer)
        admitted["refs"] += [protocol_ref, durable_producer] + upstream.e.previous.terminal(
            durable_producer, durable
        )
        for ref in durable["code_config_hashes"]:
            operand = ref
            verify(ref)
            admitted["refs"].append(ref)
        if durable["block_store_ready_score"] != 1:
            raise ValueError("qualified_durable_protocol")
        for field in ["required_checks_passed", "flagged_adversarial"]:
            if durable[field] != (field == "required_checks_passed"):
                raise ValueError("qualified_durable_protocol")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        admitted["failures"].append(
            dict(
                upstream="exp8731-resident-sparse-prototype",
                **operand,
                check="qualified_resident_source_binary_and_durability",
                field="qualified_operands",
                operator="==",
                expected="authenticated",
                observed=None if isinstance(error, FileNotFoundError) else str(error),
                operand_state="missing_file"
                if isinstance(error, FileNotFoundError)
                else "invalid_or_changed_operand",
                passed=False,
            )
        )
    return admitted


def roster(private: bool = False) -> list[Json]:
    """Freeze order before timing; repeats share one constructed workload."""
    rng = random.Random(7488732)
    rows = []
    for repeat in range(1 if private else 10):
        arms = list(ARMS)
        rng.shuffle(arms)
        for order, arm in enumerate(arms):
            rows.append(
                dict(
                    unit_id=f"{repeat}:{arm}",
                    repeat=repeat,
                    arm=arm,
                    order=order,
                    updates=4 if private else 4096,
                    restart_at=2 if private else 2048,
                    boundary_updates=2 if private else 64,
                )
            )
    return rows


def intervals(rows: list[Json]) -> Json:
    """Resample matched process repeats; individual updates cannot inflate confidence."""
    by = {(r["repeat"], r["arm"]): r["complete_ns"] for r in rows}
    repeats = sorted({r["repeat"] for r in rows})
    result: Json = dict(draws=10000, seed=7488732, pairing="process_repeat")
    rng = np.random.default_rng(7488732)
    draws = rng.integers(0, len(repeats), size=(10000, len(repeats)))
    for name, baseline, target in [
        ("resident_over_json", "json_native", "resident_native"),
        ("resident_over_python", "python", "resident_native"),
        ("json_over_python", "python", "json_native"),
    ]:
        a = np.array([by[(r, baseline)] for r in repeats], dtype=float)
        b = np.array([by[(r, target)] for r in repeats], dtype=float)
        ratios = a[draws].sum(axis=1) / b[draws].sum(axis=1)
        result[name] = dict(
            ratio=float(a.sum() / b.sum()),
            lower95=float(np.quantile(ratios, 0.025)),
            upper95=float(np.quantile(ratios, 0.975)),
        )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Passing evidence releases cost readiness; benefit still needs its own interval."""
    processes = work["process_rows"]
    updates: list[Json] = []
    controls: list[Json] = []
    for process in processes:
        for ref in process["summaries"]:
            summary = read(ref)
            rows = read(summary["rows_reference"])["rows"]
            target = controls if summary["control"] else updates
            target.extend(dict(r, arm=process["arm"], repeat=process["repeat"]) for r in rows)
    bad = any(not r["passed"] for r in receipts) or any(not r["passed"] for r in processes)
    complete = len(processes) == len(work["roster"])
    qualified = complete and not bad and not work["failures"]
    paired = intervals(processes) if qualified else {}
    latency: Json = {}
    for arm in ARMS:
        rows = [r["latency_ns"] for r in updates if r["arm"] == arm]
        timing = [p for p in processes if p["arm"] == arm]
        if rows:
            overhead = (sum(p["complete_ns"] for p in timing) - sum(rows)) / len(rows)
            latency[arm] = dict(
                per_feedback_p50_ns=float(np.quantile(rows, 0.5)),
                per_feedback_p95_ns=float(np.quantile(rows, 0.95)),
                complete_p50_ns=float(np.quantile(rows, 0.5) + overhead),
                complete_p95_ns=float(np.quantile(rows, 0.95) + overhead),
                shared_lifecycle_overhead_per_update_ns=overhead,
            )
    benefit = bool(
        qualified
        and paired["resident_over_json"]["lower95"] > 1.05
        and latency["resident_native"]["complete_p95_ns"]
        <= latency["json_native"]["complete_p95_ns"]
        and latency["resident_native"]["per_feedback_p95_ns"]
        <= latency["json_native"]["per_feedback_p95_ns"]
    )
    ready = int(qualified and not work["private_control"])
    ten = int(bool(ready and paired["resident_over_python"]["lower95"] >= 10))
    verdict = (
        "disqualified"
        if bad
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if benefit and ready
        else "null"
    )
    rows = [
        dict(
            r,
            status="completed" if i < len(processes) else "censored",
            censored=i >= len(processes),
            excluded=False,
            passed=processes[i]["passed"] if i < len(processes) else False,
            complete_ns=processes[i]["complete_ns"] if i < len(processes) else None,
        )
        for i, r in enumerate(work["roster"])
    ]
    value = dict(
        experiment_id=8732,
        task_id=TASK,
        milestone="2026.11.748",
        run_date=work["run_date"],
        invocation_id=Path(work["raw"]).name,
        honest_verdict="complete_"
        + verdict
        + (
            "_durability_budget"
            if work.get("durability_probe") and work["failures"]
            else "_external_operand"
            if verdict == "blocked"
            else "_resident_update_cost"
        ),
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        rows=rows,
        intended_count=len(rows),
        completed_count=len(processes),
        failed_count=sum(not p["passed"] for p in processes),
        censored_count=len(rows) - len(processes),
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="constructed_repeated_original_state_update_cost_only",
        independent_generalization_score=0,
        physical_fpga_execution=False,
        update_cost_ready_score=ready,
        resident_speedup_score=int(benefit and bool(ready)),
        complete_10x_score=ten,
        branch_gate_values=dict(
            update_cost_ready_score=ready,
            resident_speedup_score=int(benefit and bool(ready)),
            complete_10x_score=ten,
        ),
        process_rows=processes,
        update_rows=updates,
        boundary_control_rows=controls,
        complete_cost_rows=[
            dict(
                unit_id=p["unit_id"],
                complete_ns=p["complete_ns"],
                inclusive_components=[
                    "launch",
                    "imports",
                    "extension_load",
                    "input_preparation",
                    "host_scaling",
                    "nonlinear_transform",
                    "sparse_arithmetic",
                    "dense_renormalization",
                    "copies",
                    "checkpoint_serialization",
                    "fsync",
                    "acknowledgement",
                    "restart",
                    "shutdown",
                ],
                boundary_control_included=True,
                restart_included=True,
            )
            for p in processes
        ],
        paired_intervals=paired,
        update_latency=latency,
        kernel_only_timing=None,
        arithmetic_boundary_timing=[
            dict(
                unit_id=p["unit_id"],
                duration_ns=p["kernel_ns"],
                scope="step_call_includes_host_quantization_and_binding_transport_not_pure_kernel",
            )
            for p in processes
        ],
        kernel_only_limit="Qualified binary exposes no internal kernel timer. Caller timings include binding and host work; no kernel-only speedup inferred.",
        durability_boundary=work["durability_boundary"],
        durability_probe=work.get("durability_probe", {}),
        required_checks_passed=not bad,
        flagged_adversarial=bad,
        validation_receipts=receipts,
        acceptance_gates=dict(
            owned_validation=not bad,
            authenticated_inputs=not work["failures"],
            complete_rows=complete,
            exact_decisions_and_state=all(p["passed"] for p in processes),
            resident_benefit=benefit,
            complete_10x=bool(ten),
        ),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7488732,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"],
        cited_upstream_artifacts=work["upstream_refs"],
        task_authority=work["contract"],
        primitive_reference=reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=reference(Path(work["raw"]) / "validation.json"),
        publication_output=str(output),
        private_control=work["private_control"],
        sample_size_budget=dict(
            processes=30,
            primary_updates=122880,
            boundary_control_updates=1920,
            paired_repeats=10,
            independent_learning_cases=0,
        ),
        methodology_note="Ten paired constructed lifecycle repeats. Original4096 updates plus original64 dense-boundary controls; real restart at2048. Prediction seal and feedback commit use the unchanged durable Store. Complete cost includes both child processes and shutdown. No natural accuracy or Exp8727 future-learning measurement.",
        preserved_historical_limits=[
            "Exp8709 collection failed; skipped learning slots are scheduling evidence only.",
            "Exp8712 constructed collisions do not prove natural extraction.",
            "Exp8715 120 versus 130 detections used unequal probe counts.",
            "Exp8717 batching 1.138x; same-batch native/Python 1.005x, below 10x.",
            "Exp8719 exact signed32/24 parity establishes no speed.",
            "No causal CUDA repair, new ARC outcomes or physical acceleration established.",
        ],
    )
    value["field_principles"] = {
        k: "Bind complete constructed costs to matched durable work and sealed evidence; readiness, benefit and the 10x target are separate."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(
        dict(code=work["code_refs"], sources=work["source_refs"], roster=work["roster"])
    )
    return value
