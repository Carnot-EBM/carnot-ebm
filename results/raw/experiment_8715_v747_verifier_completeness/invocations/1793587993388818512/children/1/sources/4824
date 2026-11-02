"""REQ-REPORT-8029: retain board history without transferring scientific claims."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.experiment_8007_v694_conditioning_diagnosis import copy_evidence
from carnot.experiment_8021_v695_typed_decision_test import action
from carnot.reporting import hardware_update_8016 as history
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.verify import causal_online_8025 as causal
from carnot.verify.fixedpoint_sparse_8003 import Fixed

Json = dict[str, Any]
ROOT = history.ROOT
SOURCES = {
    8025: ("experiment_8025_v695_causal_online_updates", "learning_measurement_ready_score"),
    8027: ("experiment_8027_v695_native_update_cost", "native_update_ready_score"),
    8023: ("experiment_8023_v695_likelihood_calibration", "likelihood_calibration_ready_score"),
    8024: ("experiment_8024_v695_likelihood_decision_test", "likelihood_decision_ready_score"),
}
CONFIG = dict(
    storage_bits=24,
    accumulator_bits=32,
    fractional_bits=12,
    scale=4096,
    rounding="nearest_ties_to_even",
    overflow="saturate",
    update_budget=8192,
    drift_max=0.001,
    action_disagreements_max=0,
    overflow_max=0,
    seed=69529,
    model_load_budget=0,
    device_execution_budget=0,
    geometry_venue="CPU float64 frozen fitted geometry",
    sigmoid_venue="CPU float64 readout of fixed accumulator",
    trajectory_scope="all original exposed-development arms and seeds",
)


def load(root: Path, raw: Path) -> Json:
    """Reuse board authentication, then qualify each current source independently."""
    prior = history.authenticate(root, raw / "custody")
    refs = prior["cited_upstream_artifacts"]
    checks: list[Json] = []
    producers: Json = {}
    for eid, (stem, field) in SOURCES.items():
        path = root / "results" / (stem + ".json")
        value = json.loads(path.read_bytes()) if path.is_file() else {}
        digest = sha256_file(path) if path.is_file() else None
        gates = []
        for key, expected in (
            ("experiment_id", eid),
            ("run_date", "20261002"),
            (field, 1),
            ("flagged_adversarial", False),
            ("verdict_class", ["null", "positive"]),
        ):
            observed = value.get(key, "MISSING_CONTRACT_FIELD")
            passed = (
                observed in expected
                if isinstance(expected, list)
                else (type(observed) is type(expected) and observed == expected)
            )
            gates.append(
                dict(
                    upstream_id=f"exp{eid}",
                    artifact_path=str(path),
                    artifact_hash=digest,
                    artifact_field=key,
                    expected=expected,
                    observed=observed,
                    passed=passed,
                    branch="numeric"
                    if eid == 8025
                    else "transaction"
                    if eid == 8027
                    else "teacher_forced_scoring",
                )
            )
        receipts = value.get("validation_receipts", [])
        required = [
            r
            for r in receipts
            if isinstance(r, dict) and r.get("required", r.get("scope") != "repository_health")
        ]
        gates.append(
            dict(
                gates[0],
                artifact_field="validation_receipts.required_checks_passed",
                expected=True,
                observed=bool(required) and all(r.get("passed") is True for r in required),
                passed=bool(required) and all(r.get("passed") is True for r in required),
            )
        )
        checks.extend(gates)
        if path.is_file():
            refs.append(copy_evidence(reference(path), raw))
            for receipt in value.get("validation_receipts", []):
                if (
                    isinstance(receipt, dict)
                    and receipt.get("log_path")
                    and not receipt.get("passed")
                ):
                    refs.append(
                        copy_evidence(
                            dict(path=receipt["log_path"], sha256=receipt["log_sha256"]), raw
                        )
                    )
        producers[eid] = value if all(g["passed"] for g in gates) else None
    trajectory = None
    if producers[8025] is not None:
        producer = producers[8025]
        ref = next(r for r in producer["raw_shard_hashes"] if Path(r["path"]).name == "inputs.json")
        refs.append(copy_evidence(ref, raw))
        trajectory = json.loads(checked(refs[-1]).read_bytes())
        trajectory["updates"] = producer["update_rows"]
        for ref in producer["checkpoint_references"]:
            refs.append(copy_evidence(ref, raw))
    scoring = None
    if producers[8023] is not None:
        scoring = [
            dict(
                duration_s=(r["ended_monotonic_ns"] - r["started_monotonic_ns"]) / 1e9,
                operation="teacher_forced_prefill_scoring",
                id=r["id"],
            )
            for r in producers[8023]["rows"]
        ]
    boards = prior["boards"]
    for board in boards:
        board["custody_status"] = "valid" if board["custody_valid"] else "blocked"
        board["custody_verdict_class"] = "null" if board["custody_valid"] else "blocked"
        board["last_actual_execution_date_field"] = (
            "run_date" if board.get("last_actual_execution_date") else None
        )
        board["last_execution_date_reason"] = board.get("execution_date_absence_reason")
    diagnostic = root / "results/experiment_8024_likelihood_decision_test.json"
    if diagnostic.is_file():
        refs.append(copy_evidence(reference(diagnostic), raw))
    historical = []
    original = root / "results/experiment_8016_v694_hardware_update_boundary.json"
    if original.is_file():
        refs.append(copy_evidence(reference(original), raw))
        original_value = json.loads(checked(refs[-1]).read_bytes())
        historical.append(
            dict(
                experiment_id=8016,
                honest_verdict=original_value["honest_verdict"],
                verdict_class=original_value["verdict_class"],
                reference=refs[-1],
            )
        )
    return dict(
        boards=boards,
        checks=checks,
        references=refs,
        fixture=False,
        trajectory=trajectory,
        timings=producers[8027]["timing_rows"] if producers[8027] is not None else None,
        scoring=scoring,
        historical_failed_operands=prior["historical_failed_operands"] + prior["checks"],
        preserved_historical_outcomes=historical,
    )


def controls() -> Json:
    """Threshold stress demonstrates quantization loss without natural sample credit."""
    rows = []
    for threshold in (0.1, 0.5):
        for offset in (-1e-5, 0.0, 1e-5):
            original = threshold + offset
            fixed = Fixed(24).encode(original) / 4096
            rows.append(
                dict(
                    threshold=threshold,
                    offset=offset,
                    float_probability=original,
                    fixed_probability=fixed,
                    float_action=action(original),
                    fixed_action=action(fixed),
                    action_disagreement=action(original) != action(fixed),
                    independent=0,
                    fixture=True,
                    scope="synthetic_threshold_stress",
                )
            )
    return dict(
        working=any(r["action_disagreement"] for r in rows),
        rows=rows,
        scope="synthetic_stress_only_no_natural_qualification",
        independent=0,
    )


def fixed_step(head: Json, vector: list[float], y: int) -> tuple[Json, Json]:
    """Integer products, lazy decay and calibration expose each lost conversion."""
    q, acc = Fixed(24), Fixed(32)
    theta = [q.encode(v) for v in head["parameters"]]
    weights = [q.encode(v) for v in vector]
    decay = q.encode(head["decay_scale"])
    a, b = [q.encode(v) for v in head["calibration"]]

    def probability(values: list[int], multiplier: int) -> float:
        total = 0
        for weight, coefficient in zip(weights, values, strict=True):
            total = acc.add(total, acc.mul(weight, q.mul(coefficient, multiplier)))
        return float(expit(acc.add(a, acc.mul(b, total)) / 4096))

    before = probability(theta, decay)
    next_decay = q.mul(
        decay, q.encode(1 - 2 * causal.CONFIG["l2"] * causal.CONFIG["learning_rate"])
    )
    inverse = q.encode(4096 / next_decay)
    residual = q.mul(q.encode(before - y), b)
    for i, weight in enumerate(weights):
        if weight:
            gradient = q.mul(residual, weight)
            delta = q.mul(q.mul(q.encode(causal.CONFIG["learning_rate"]), gradient), inverse)
            theta[i] = q.add(theta[i], -delta)
    after = probability(theta, next_decay)
    state = dict(head, parameters=[v / 4096 for v in theta], decay_scale=next_decay / 4096)
    return state, dict(
        before=before,
        after=after,
        scale=4096,
        saturation_count=q.saturations + acc.saturations,
        storage_saturations=q.saturations,
        accumulator_saturations=acc.saturations,
    )


def numeric(trajectory: Json, fixture: bool) -> list[Json]:
    """Replay complete arm histories and retain restart states before qualification."""
    updates = trajectory["updates"]
    if not updates or len(updates) > CONFIG["update_budget"]:
        raise ValueError("trajectory_budget")
    sources = {r["family_id"]: r for r in trajectory["sources"]}
    states: Json = {}
    rows = []
    started = time.monotonic()
    for index, update in enumerate(updates):
        key = f"{update['arm']}/{update['seed']}"
        if key not in states:
            states[key] = (deepcopy(trajectory["head"]), deepcopy(trajectory["head"]))
        floating, fixed = states[key]
        source = sources[update["family_id"]]
        vector = causal.design(floating, source)
        before = causal.probability(floating, vector)
        restored = json.loads(json.dumps(fixed))
        fixed, metric = fixed_step(fixed, vector.tolist(), update["y"])
        restarted, restart_metric = fixed_step(restored, vector.tolist(), update["y"])
        causal.update(floating, vector, update["y"])
        after = causal.probability(floating, vector)
        float_actions = [action(before), action(after)]
        fixed_actions = [action(metric["before"]), action(metric["after"])]
        agrees = restarted == fixed and restart_metric == metric
        drift = max(abs(before - metric["before"]), abs(after - metric["after"]))
        checkpoint = dict(float_head=deepcopy(floating), fixed_head=deepcopy(fixed))
        rows.append(
            dict(
                id=f"{key}/{update['update_index']}",
                arm=update["arm"],
                seed=update["seed"],
                source_cluster_id=source["source_cluster_id"],
                family_id=update["family_id"],
                condition="signed24_q12_acc32",
                metric="max_probability_drift",
                numerator=drift,
                denominator=1,
                status="completed",
                exclusion_reason=None,
                failure_reason=None,
                censor_reason=None,
                independent=0,
                fixture=fixture,
                float_probability_before=before,
                float_probability_after=after,
                float_actions=float_actions,
                fixed_actions=fixed_actions,
                action_disagreements=sum(
                    a != b for a, b in zip(float_actions, fixed_actions, strict=True)
                ),
                restart_agrees=agrees,
                restart_decisions=[
                    action(restart_metric["before"]),
                    action(restart_metric["after"]),
                ],
                checkpoint=checkpoint,
                **metric,
            )
        )
        states[key] = (floating, fixed)
        if index % 256 == 0 or index == len(updates) - 1:
            print(
                f"[exp8029] numeric elapsed_s={time.monotonic() - started:.3f} "
                f"completed={index + 1} pending={len(updates) - index - 1}",
                flush=True,
            )
    return rows


def costs(plan: Json) -> Json:
    """Amdahl fractions use complete matching transactions without imported speedups."""
    bounds, sampling = [], []
    for row in plan["timings"] or []:
        total, arithmetic = row["transaction_ns"], row["arithmetic_ns"]
        if not 0 <= arithmetic <= total or total <= 0:
            raise ValueError("cost_partition")
        serial = (total - arithmetic) / total
        bounds.append(
            dict(
                arm=row["arm"],
                batch=row["batch"],
                repetition=row["repetition"],
                transaction_ns=total,
                sparse_arithmetic_ns=arithmetic,
                serial_share=serial,
                speedup_bound=total / (total - arithmetic + arithmetic / 100),
                ffi_ns=row["binding_ns"],
                storage_ns=row["serialization_ns"] + row["persistence_fsync_ns"],
                feature_construction_ns=row["feature_construction_ns"],
                scope="current CPU transaction only; model scoring and device transfer excluded",
            )
        )
        if row.get("operation") == "ising_sampling" and 0 < row.get("sampling_ns", 0) < total:
            fraction = row["sampling_ns"] / total
            sampling.append(
                dict(serial_share=1 - fraction, speedup_bound=1 / (1 - fraction + fraction / 100))
            )
    return dict(
        sparse_arithmetic_100x=bounds or None,
        hypothetical_sampling=sampling or None,
        teacher_forced_scoring_spans=plan["scoring"],
        combined_service_bound=None,
        current_device_compatible_fraction=0,
    )


def reduce(plan: Json) -> Json:
    """Custody is a valid null even when scientific or workload branches cannot qualify."""
    boards = deepcopy(plan["boards"])
    numeric_rows = numeric(plan["trajectory"], plan["fixture"]) if plan["trajectory"] else []
    gates = dict(
        max_probability_drift=max((r["numerator"] for r in numeric_rows), default=None),
        action_disagreements=sum(r["action_disagreements"] for r in numeric_rows),
        overflow=sum(r["saturation_count"] for r in numeric_rows),
        restart_agrees=all(r["restart_agrees"] for r in numeric_rows) if numeric_rows else None,
    )
    numeric_pass = (
        bool(numeric_rows)
        and gates["max_probability_drift"] <= 0.001
        and (gates["action_disagreements"] == gates["overflow"] == 0 and gates["restart_agrees"])
    )
    custody = len(boards) == 3 and all(b["custody_valid"] for b in boards)
    rows = [
        dict(
            id=b["board"],
            arm="board_custody",
            metric="authenticated_receipt",
            numerator=int(b["custody_valid"]),
            denominator=1,
            seed=None,
            status="completed" if b["custody_valid"] else "blocked",
            independent=0,
            exclusion_reason=None,
            failure_reason=None,
            censor_reason=None,
        )
        for b in boards
    ]
    rows += [{k: v for k, v in r.items() if k != "checkpoint"} for r in numeric_rows]
    missing = [
        "device_transfer",
        "device_kernel",
        "power",
        "complete_model_service_join",
        "prefill_scoring_partition",
        "orchestration_partition",
    ]
    if plan["timings"] is None:
        missing += ["qualified_transaction", "sparse_arithmetic", "FFI", "storage"]
    if plan["scoring"] is None:
        missing += ["qualified_current_teacher_forced_scoring"]
    return dict(
        honest_verdict="complete_null_independent_hardware_custody"
        if custody
        else "complete_blocked_hardware_custody",
        verdict_class="null" if custody else "blocked",
        hardware_custody_ready_score=int(custody),
        quantized_update_ready_score=int(numeric_pass),
        numeric_branch_status="qualified"
        if numeric_pass
        else "disqualified"
        if numeric_rows
        else "blocked",
        transaction_branch_status="qualified" if plan["timings"] else "blocked",
        scoring_branch_status="qualified" if plan["scoring"] else "blocked",
        board_rows=boards,
        cumulative_error_rows=numeric_rows,
        rows=rows,
        acceleration_bounds=costs(plan),
        missing_cost_components=missing,
        acceptance_gate_results=dict(
            custody=custody, numeric=gates, numeric_pass=numeric_pass, device_benefit=False
        ),
        sample_size_budget=dict(
            intended=len(rows),
            eligible=len(rows),
            started=len(rows),
            completed=sum(r["status"] == "completed" for r in rows),
            excluded=0,
            failed=sum(r["status"] == "blocked" for r in rows),
            censored=0,
            independent=0
            if plan["fixture"]
            else len({r["source_cluster_id"] for r in numeric_rows}),
        ),
        current_device_execution_count=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=plan["fixture"],
        genuine_headroom=False,
        positive_control_results=controls(),
        gate_check_summary=[r for r in plan["checks"] if not r["passed"]],
        preconditions_checked=plan["checks"],
        historical_failed_operands=plan["historical_failed_operands"],
        preserved_historical_outcomes=plan.get("preserved_historical_outcomes", []),
        cited_upstream_artifacts=plan["references"],
        unqualified_substrates=["NPU", "TSU"],
        workload_placement_rows=[
            dict(
                operation=op,
                current_venue=venue,
                executable_device_kernel=False,
                device_execution_count=0,
            )
            for op, venue in (
                ("model_prefill_teacher_forced_scoring", "original model producer"),
                ("spline_geometry_calibrated_gradient", "CPU"),
                ("sparse_coefficient_updates", "CPU"),
                ("FFI_orchestration", "CPU"),
                ("durable_storage", "host filesystem"),
                ("Ising_boundary_clamp_sampling", "no matching current operation"),
            )
        ],
        paper_operation_matches=[
            dict(
                url="https://arxiv.org/abs/2602.15985v2",
                match="Graph decomposition, boundary clamps and data-dependent gather/scatter match Ising orchestration concepts; spline gradients and LLM prefill require other kernels.",
                local_device_evidence=False,
            ),
            dict(
                url="https://extropic.ai/writing/z1t",
                match="Sparse probabilistic statistics on TSUs and digital neural stages form hybrid operations; the existing Ising overlay does not implement Z1T, calibrated spline updates or Qwen scoring.",
                local_device_evidence=False,
            ),
        ],
        purchase_recommendation="No purchase: no qualified useful measured compatible device bottleneck and executable kernel",
    )
