"""REQ-REPORT-8042: conservative recurrence bounds keep CPU fallback accountable."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.experiment_8007_v694_conditioning_diagnosis import copy_evidence
from carnot.experiment_8021_v695_typed_decision_test import action
from carnot.reporting import hardware_update_8016 as history
from carnot.reporting import hardware_workload_8029 as old
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.verify import causal_online_8025 as causal
from carnot.verify import windowed_online_8038 as window
from carnot.verify.fixedpoint_sparse_8003 import Fixed

Json = dict[str, Any]
ROOT = old.ROOT
SOURCES = {
    8038: ("experiment_8038_v696_windowed_online_learning", "learning_trajectory_ready_score"),
    8040: ("experiment_8040_v696_native_transaction_cost", "native_transaction_ready_score"),
}
CONFIG: Json = dict(
    old.CONFIG,
    seed=8042,
    numerical_budget_s=600,
    thresholds=[0.1, 0.5],
    rounding_cell_radius=1 / 8192,
    float_roundoff_allowance=2**-40,
    sigmoid_bounded_logit=32,
    bound_method="outward interval recurrence; Q12 rounding cells; actual fixed center loss",
    fallback="threshold overlap or saturation; existing float64 CPU shadow",
    acceptance="all readouts contained; zero final disagreements/unhandled overflow; exact restart",
    width_tuning=False,
    retention_labels_opened=False,
)


def outward(lo: float, hi: float) -> list[float]:
    """A fixed relative allowance covers host arithmetic and dot-product ordering.

    The allowance is frozen above 2*n*epsilon for the 110-term CPU dot product.
    It is independent of evaluation errors, labels and observed decisions.
    """
    pad = CONFIG["float_roundoff_allowance"] * (1 + max(abs(lo), abs(hi)))
    return [math.nextafter(lo - pad, -math.inf), math.nextafter(hi + pad, math.inf)]


def add(left: list[float], right: list[float]) -> list[float]:
    """Outward sums retain both endpoint arithmetic errors."""
    return outward(left[0] + right[0], left[1] + right[1])


def multiply(left: list[float], right: list[float]) -> list[float]:
    """All sign combinations matter when calibration or coefficients are negative."""
    values = [a * b for a in left for b in right]
    return outward(min(values), max(values))


def divide(left: list[float], right: list[float]) -> list[float]:
    """Lazy decay must stay positive for a bounded inverse recurrence."""
    if right[0] <= 0:
        raise ValueError("decay_interval")
    return multiply(left, outward(1 / right[1], 1 / right[0]))


def sigmoid(z: list[float]) -> list[float]:
    """Monotonic sigmoid maps a logit enclosure to a probability enclosure.

    Outside the frozen finite logit domain use the complete probability range.
    Within it, the frozen absolute allowance also covers the CPU sigmoid readout.
    """
    if z[0] < -32 or z[1] > 32:
        return [0.0, 1.0]
    lo, hi = outward(float(expit(z[0])), float(expit(z[1])))
    return [max(0.0, lo), min(1.0, hi)]


def initialize(head: Json) -> Json:
    """Rounding cells enclose inputs; saturated cells include exact initial loss."""
    q = Fixed(24)

    def cell(value: float) -> list[float]:
        center = q.encode(value) / 4096
        radius = max(CONFIG["rounding_cell_radius"], abs(value - center))
        return outward(center - radius, center + radius)

    return dict(
        parameters=[cell(v) for v in head["parameters"]],
        decay=cell(head["decay_scale"]),
        calibration=[cell(v) for v in head["calibration"]],
    )


def readout(state: Json, x: list[float]) -> tuple[list[float], list[float]]:
    """The same CPU geometry enters both paths; no fitted error determines width."""
    total = [0.0, 0.0]
    for coefficient, weight in zip(state["parameters"], x, strict=True):
        total = add(total, multiply(multiply(coefficient, state["decay"]), [weight, weight]))
    z = add(state["calibration"][0], multiply(state["calibration"][1], total))
    return z, sigmoid(z)


def advance(state: Json, x: list[float], y: int) -> Json:
    """Propagate BCE, affine calibration, sparse gradients and global L2 decay.

    Labels are released recurrence operands. They never select an interval width.
    Quantized centers may drift; this enclosure retains all prior update uncertainty.
    """
    changed = deepcopy(state)
    probability = readout(state, x)[1]
    decay = 1 - 2 * causal.CONFIG["l2"] * causal.CONFIG["learning_rate"]
    changed["decay"] = multiply(state["decay"], [decay, decay])
    residual = multiply(add(probability, [-y, -y]), state["calibration"][1])
    for i, weight in enumerate(x):
        if weight:
            gradient = multiply(residual, [weight, weight])
            delta = divide(multiply(gradient, [0.01, 0.01]), changed["decay"])
            changed["parameters"][i] = add(state["parameters"][i], [-delta[1], -delta[0]])
    return changed


def operand(eid: Any, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Absent contract operands remain explicit, distinct from measured zeros."""
    return dict(
        upstream_id=f"exp{eid}",
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        check_name=field,
        expected=expected,
        observed=observed,
        passed=type(expected) is type(observed) and expected == observed,
    )


def fixed_read(head: Json, x: list[float]) -> tuple[float, int, float]:
    """Use original Q12 products and signed32 sums without spending an update."""
    q, acc = Fixed(24), Fixed(32)
    scale = q.encode(head["decay_scale"])
    total = 0
    for v, w in zip(head["parameters"], x, strict=True):
        total = acc.add(total, acc.mul(q.encode(w), q.mul(q.encode(v), scale)))
    a, b = [q.encode(v) for v in head["calibration"]]
    z = acc.add(a, acc.mul(b, total))
    return float(expit(z / 4096)), q.saturations + acc.saturations, z / 4096


def numeric(trajectory: Json, fixture: bool, *, budget_s: float | None = None) -> Json:
    """Replay original event order, keeping measured costs outside deterministic rows."""
    events = trajectory["events"]
    count = sum(r["kind"] == "gradient" for r in events)
    if not events or count > CONFIG["update_budget"]:
        raise ValueError("trajectory_budget")
    states: Json = {}
    rows, costs = [], []
    started = time.monotonic()
    for index, event in enumerate(events):
        if time.monotonic() - started > (
            CONFIG["numerical_budget_s"] if budget_s is None else budget_s
        ):
            raise TimeoutError("numerical_budget_s")
        key = f"{event['arm']}/{event['seed']}"
        if key not in states:
            print(
                f"[exp8042] small_head_load_before elapsed_s={time.monotonic() - started:.3f} completed={index} pending={len(events) - index}",
                flush=True,
            )
            head = deepcopy(trajectory["head"])
            states[key] = dict(floating=head, fixed=deepcopy(head), interval=initialize(head))
            print(
                f"[exp8042] small_head_load_after elapsed_s={time.monotonic() - started:.3f} completed={index} pending={len(events) - index}",
                flush=True,
            )
        state = states[key]
        floating, fixed, interval = state["floating"], state["fixed"], state["interval"]
        x = event["x"]
        restart = json.loads(json.dumps(state))
        overflow = 0
        for stage in ("before", "after") if event["kind"] == "gradient" else ("issued",):
            began = time.perf_counter_ns()
            if stage == "after":
                fixed, info = old.fixed_step(fixed, x, event["y"])
                overflow += info["saturation_count"]
                interval = advance(interval, x, event["y"])
            quantized, saturated, fixed_logit = fixed_read(fixed, x)
            overflow += saturated
            z, bounds = readout(interval, x)
            z = [min(z[0], fixed_logit), max(z[1], fixed_logit)]
            bounds = sigmoid(z)
            quantized_ns = time.perf_counter_ns() - began
            began = time.perf_counter_ns()
            if stage == "after":
                causal.update(floating, np.asarray(x), event["y"])
            probability = causal.probability(floating, np.asarray(x))
            shadow_ns = time.perf_counter_ns() - began
            fallback = overflow > 0 or any(
                bounds[0] <= t <= bounds[1] for t in CONFIG["thresholds"]
            )
            began = time.perf_counter_ns()
            transfer = json.dumps(floating, sort_keys=True).encode() if fallback else b""
            restored = json.loads(transfer) if transfer else floating
            chosen = causal.probability(restored, np.asarray(x)) if fallback else quantized
            fallback_ns = time.perf_counter_ns() - began
            if stage == "after":
                rq, _ = old.fixed_step(restart["fixed"], x, event["y"])
                ri = advance(restart["interval"], x, event["y"])
                causal.update(restart["floating"], np.asarray(x), event["y"])
                agrees = rq == fixed and ri == interval and restart["floating"] == floating
            else:
                agrees = restart == state and action(
                    causal.probability(restart["floating"], np.asarray(x))
                ) == action(probability)
            expected = event.get("expected_head_hash") if stage != "before" else None
            if expected and canonical_hash(floating) != expected:
                raise ValueError("trajectory_state_drift")
            row = dict(
                id=f"{key}/{index}/{stage}",
                arm=event["arm"],
                seed=event["seed"],
                family_id=event["family_id"],
                source_cluster_id=event["source_cluster_id"],
                condition="signed24_q12_acc32_with_float64_fallback",
                stage=stage,
                metric="interval_containment",
                numerator=int(bounds[0] <= probability <= bounds[1]),
                denominator=1,
                status="completed",
                exclusion_reason=None,
                failure_reason=None,
                censor_reason=None,
                independent=0,
                fixture=fixture,
                logit_interval=z,
                probability_interval=bounds,
                contained=bounds[0] <= probability <= bounds[1],
                float_probability=probability,
                fixed_probability=quantized,
                quantization_error=abs(probability - quantized),
                used_float64=fallback,
                decision=action(chosen),
                reference_decision=action(probability),
                post_fallback_action_disagreement=action(chosen) != action(probability),
                handled_overflow=overflow if fallback else 0,
                unhandled_overflow=overflow if not fallback else 0,
                restart_agrees=agrees,
                transfer_bytes=len(transfer),
                coefficient_error_bound=max(
                    max(abs(v - a), abs(v - b))
                    for v, (a, b) in zip(fixed["parameters"], interval["parameters"], strict=True)
                ),
                state_hash=canonical_hash(dict(floating=floating, fixed=fixed, interval=interval)),
            )
            rows.append(row)
            costs.append(
                dict(
                    id=row["id"],
                    arm=row["arm"],
                    event_kind=event["kind"],
                    quantized_and_bound_ns=quantized_ns,
                    unconditional_shadow_ns=shadow_ns,
                    fallback_arithmetic_transfer_ns=fallback_ns,
                    transfer_bytes=len(transfer),
                )
            )
        states[key] = dict(floating=floating, fixed=fixed, interval=interval)
        if index % 256 == 0 or index == len(events) - 1:
            print(
                f"[exp8042] numeric elapsed_s={time.monotonic() - started:.3f} completed={index + 1} pending={len(events) - index - 1}",
                flush=True,
            )
    return dict(rows=rows, checkpoints=states, cpu_cost_rows=costs)


def load(root: Path, raw: Path) -> Json:
    """Authenticate original custody and current byte-bound journals independently."""
    prior = history.authenticate(root, raw / "custody")
    refs, checks, producers = prior["cited_upstream_artifacts"], [], {}
    for eid, (stem, ready) in SOURCES.items():
        path = root / "results" / (stem + ".json")
        value = json.loads(path.read_bytes()) if path.is_file() else {}
        gates = [
            operand(eid, path, k, expected, value.get(k, "MISSING_CONTRACT_FIELD"))
            for k, expected in (
                ("experiment_id", eid),
                ("run_date", "20261002"),
                (ready, 1),
                ("flagged_adversarial", False),
                ("verdict_class", "null"),
            )
        ]
        if path.is_file():
            refs.append(copy_evidence(reference(path), raw))
        if all(g["passed"] for g in gates):
            try:
                terminal = Path(value["terminal_validation_sidecar_path"])
                binding = json.loads(terminal.read_bytes())["publication"]
                sidecar = Path(binding["sidecar_path"])
                report = json.loads(sidecar.read_bytes())
                for label, obj in (("binding", binding), ("sidecar", report)):
                    gates += [
                        operand(
                            eid,
                            path,
                            label + ".primary_sha256",
                            sha256_file(path),
                            obj["primary_sha256"],
                        ),
                        operand(eid, path, label + ".primary_path", str(path), obj["primary_path"]),
                    ]
                gates.append(operand(eid, path, "report.passed", True, report["report"]["passed"]))
                required = [
                    r
                    for r in value["validation_receipts"]
                    if r.get("required", r.get("scope") != "repository_health")
                ]
                gates.append(
                    operand(
                        eid,
                        path,
                        "required_exits",
                        True,
                        bool(required)
                        and all(
                            r["passed"] is True
                            and type(r["exit_code"]) is int
                            and r["exit_code"] == 0
                            for r in required
                        ),
                    )
                )
                for r in required:
                    refs.append(
                        copy_evidence(dict(path=r["log_path"], sha256=r["log_sha256"]), raw)
                    )
                refs.extend(copy_evidence(reference(p), raw) for p in (terminal, sidecar))
            except (KeyError, ValueError, OSError) as error:
                gates.append(
                    operand(
                        eid, path, "terminal_contract", "complete byte-bound evidence", str(error)
                    )
                )
        checks.extend(gates)
        producers[eid] = value if all(g["passed"] for g in gates) else None
    trajectory = None
    if producers[8038] is not None:
        value = producers[8038]
        directory = Path(value["trajectory_directory"])
        refs.append(copy_evidence(value["trajectory_seal"], raw))
        seal = json.loads(checked(refs[-1]).read_bytes())
        for ref in seal["references"]:
            refs.append(copy_evidence(ref, raw))
        measured = window.reduce(directory)
        inputs = json.loads((directory / "inputs.json").read_bytes())
        sources = {r["family_id"]: r for r in inputs["sources"]}
        events = []
        excluded = []
        for kind, records in (
            ("issue", measured["issued_prediction_rows"]),
            ("gradient", measured["gradient_rows"]),
        ):
            for row in records:
                if kind == "issue" and not row["eligibility"]:
                    excluded.append(
                        dict(row, exclusion_reason=row["exclusion_reason"] or "public_unavailable")
                    )
                    continue
                events.append(
                    dict(
                        kind=kind,
                        arm=row["arm"],
                        seed=row["seed"],
                        family_id=row["family_id"],
                        source_cluster_id=sources[row["family_id"]]["source_cluster_id"],
                        x=causal.design(inputs["head"], sources[row["family_id"]]).tolist(),
                        y=row.get("y"),
                        update_index=row.get("update_index", -1),
                        durable_commit_id=row["durable_commit_id"],
                        expected_head_hash=row["head_hash"]
                        if kind == "issue"
                        else row["after_head_hash"],
                    )
                )
        events.sort(key=lambda r: (r["durable_commit_id"], r["update_index"]))
        trajectory = dict(head=inputs["head"], events=events, excluded_rows=excluded)
    transactions = None
    if producers[8040] is not None:
        transactions = producers[8040]["transaction_rows"]
        for row in transactions:
            refs.append(copy_evidence(row["checkpoint"], raw))
    historical = []
    path = root / "results/experiment_8029_v695_hardware_workload_boundary.json"
    if path.is_file():
        refs.append(copy_evidence(reference(path), raw))
        original = json.loads(checked(refs[-1]).read_bytes())
        historical.append(
            dict(
                reference=refs[-1],
                honest_verdict=original["honest_verdict"],
                acceptance_gate_results=original["acceptance_gate_results"],
            )
        )
    boards = prior["boards"]
    for board in boards:
        board.update(
            custody_status="valid" if board["custody_valid"] else "blocked",
            custody_checked_date="20261002",
            current_hardware_execution=False,
        )
    checks = [
        dict(
            r,
            path=r.get("path", r.get("artifact_path")),
            sha256=r.get("sha256", r.get("artifact_hash")),
            check_name=r.get("check_name", r["artifact_field"]),
        )
        for r in prior["checks"] + checks
    ]
    return dict(
        boards=boards,
        checks=checks,
        references=refs,
        fixture=False,
        trajectory=trajectory,
        transactions=transactions,
        historical_failed_operands=prior["historical_failed_operands"],
        historical_outcomes=historical,
    )


def costs(transactions: list[Json] | None, cpu: list[Json]) -> Json:
    """Imported complete transactions bound only matching hypothetical arithmetic.

    Current shadow, interval, fallback and host-copy costs stay serial. This is
    additive CPU accounting, not an end-to-end device or complete model service.
    """
    bounds = []
    for row in transactions or []:
        if row["excluded"]:
            continue
        total, arithmetic = row["transaction_ns"], row["arithmetic_ns"]
        if not 0 <= arithmetic <= total or total <= 0:
            raise ValueError("cost_partition")
        matching = [
            r for r in cpu if r["arm"] == row["condition"] and r["event_kind"] == "gradient"
        ]
        serial = sum(
            r["quantized_and_bound_ns"]
            + r["unconditional_shadow_ns"]
            + r["fallback_arithmetic_transfer_ns"]
            for r in matching
        )
        overhead = serial / len(matching) * 2 * row["denominator"] if matching else 0.0
        augmented = total + overhead
        bounds.append(
            dict(
                arm=row["arm"],
                condition=row["condition"],
                repetition=row["repetition"],
                imported_transaction_ns=total,
                compatible_arithmetic_ns=arithmetic,
                fallback_transfer_serial_ns=overhead,
                serial_share=(augmented - arithmetic) / augmented,
                speedup_bound=augmented / (augmented - arithmetic + arithmetic / 100),
                scope="hypothetical additive transaction bound; all CPU fallback/shadow/bound/copy cost serial",
            )
        )
    return dict(
        hypothetical_compatible_arithmetic_100x=bounds or None,
        current_device_compatible_fraction=0,
        complete_service_bound=None,
        state_transfer_scope="measured CPU JSON encode/decode; device transfer unmeasured",
    )


def controls() -> Json:
    """Adversarial controls cannot increase the independent development sample."""
    results = []
    for threshold, size in ((0.1, 2), (0.5, 2), (0.5, 128), (1.0, 2)):
        coefficient = 1e8 if threshold == 1 else math.log(threshold / (1 - threshold))
        trajectory = dict(
            head=dict(parameters=[coefficient, 0.1], decay_scale=1.0, calibration=[0.0, 1.0]),
            events=[
                dict(
                    kind="gradient",
                    arm="control",
                    seed=8042,
                    family_id="synthetic",
                    source_cluster_id="synthetic",
                    x=[1.0, 0.0],
                    y=i % 2,
                    update_index=i,
                )
                for i in range(size)
            ],
        )
        measured = numeric(trajectory, True, budget_s=600)
        results.append(
            dict(
                threshold=threshold,
                updates=size,
                independent=0,
                containment=all(r["contained"] for r in measured["rows"]),
                parity=all(not r["post_fallback_action_disagreement"] for r in measured["rows"]),
                fallback_count=sum(r["used_float64"] for r in measured["rows"]),
                overflow=sum(r["handled_overflow"] for r in measured["rows"]),
                restart=all(r["restart_agrees"] for r in measured["rows"]),
            )
        )
    return dict(
        working=all(r["containment"] and r["parity"] and r["restart"] for r in results),
        rows=results,
        independent=0,
        scope="synthetic threshold saturation long-update controls",
    )


def reduce(plan: Json) -> Json:
    """Custody, numerical safety and current hardware benefit have independent gates."""
    checks = deepcopy(plan["checks"])
    measured: Json = dict(rows=[], checkpoints={}, cpu_cost_rows=[])
    blocked_operand = "MISSING_QUALIFIED_TRACE"
    if plan["trajectory"] is not None:
        try:
            measured = numeric(plan["trajectory"], plan["fixture"])
        except TimeoutError:
            blocked_operand = "EXCEEDED_NUMERICAL_BUDGET"
    numeric_rows = measured["rows"]
    if not numeric_rows:
        checks.append(
            operand(
                8042, Path("numeric_branch"), "qualified_trace_within_600s", True, blocked_operand
            )
        )
    gates = dict(
        containment_failures=sum(not r["contained"] for r in numeric_rows),
        post_fallback_action_disagreements=sum(
            r["post_fallback_action_disagreement"] for r in numeric_rows
        ),
        handled_overflow=sum(r["handled_overflow"] for r in numeric_rows),
        unhandled_overflow=sum(r["unhandled_overflow"] for r in numeric_rows),
        restart_agrees=all(r["restart_agrees"] for r in numeric_rows) if numeric_rows else None,
    )
    passed = (
        bool(numeric_rows)
        and gates["restart_agrees"]
        and gates["containment_failures"]
        == gates["post_fallback_action_disagreements"]
        == gates["unhandled_overflow"]
        == 0
    )
    custody = len(plan["boards"]) == 3 and all(b["custody_valid"] for b in plan["boards"])
    rows = [
        dict(
            id=b["board"],
            arm="board_custody",
            seed=None,
            metric="authenticated_receipt",
            numerator=int(b["custody_valid"]),
            denominator=1,
            independent=0,
            status="completed" if b["custody_valid"] else "blocked",
            exclusion_reason=None,
            failure_reason=None if b["custody_valid"] else "receipt_authentication",
            censor_reason=None,
        )
        for b in plan["boards"]
    ] + numeric_rows
    cpu = plan.get("cpu_cost_rows", measured["cpu_cost_rows"])
    excluded = plan["trajectory"].get("excluded_rows", []) if plan["trajectory"] else []
    sample = dict(
        intended=len(rows) + len(excluded),
        eligible=len(rows),
        started=len(rows),
        completed=sum(r["status"] == "completed" for r in rows),
        excluded=len(excluded),
        failed=sum(r["status"] == "blocked" for r in rows),
        censored=0,
        independent=0 if plan["fixture"] else len({r["source_cluster_id"] for r in numeric_rows}),
        seeds_are_independent=False,
        independent_datasets=0 if plan["fixture"] else int(bool(numeric_rows)),
    )
    result = dict(
        honest_verdict="complete_circular_positive_fallback_controls"
        if custody and plan["fixture"]
        else "complete_null_bounded_fallback_custody"
        if custody
        else "complete_blocked_hardware_custody",
        verdict_class="circular_positive"
        if custody and plan["fixture"]
        else "null"
        if custody
        else "blocked",
        hardware_custody_ready_score=int(custody),
        fallback_parity_ready_score=int(bool(passed)),
        numeric_branch_status="qualified"
        if passed
        else "disqualified"
        if numeric_rows
        else "blocked",
        board_rows=plan["boards"],
        rows=rows,
        excluded_rows=excluded,
        interval_containment_rows=numeric_rows,
        fallback_rows=[
            dict(
                id=r["id"],
                used_float64=True,
                decision=r["decision"],
                transfer_bytes=r["transfer_bytes"],
                reason="overflow" if r["handled_overflow"] else "typed_threshold_overlap",
            )
            for r in numeric_rows
            if r["used_float64"]
        ],
        quantization_error_rows=[
            dict(
                id=r["id"],
                probability_error=r["quantization_error"],
                coefficient_error_bound=r["coefficient_error_bound"],
                fixture=r["fixture"],
            )
            for r in numeric_rows
        ],
        fallback_fraction=sum(r["used_float64"] for r in numeric_rows) / len(numeric_rows)
        if numeric_rows
        else None,
        fallback_fraction_denominator=len(numeric_rows),
        cpu_emulation_cost_rows=cpu,
        final_checkpoints=measured["checkpoints"],
        acceleration_bounds=costs(plan["transactions"], cpu),
        missing_cost_components=[
            "device_kernel",
            "device_transfer",
            "power",
            "complete_model_service_join",
        ]
        + ([] if plan["transactions"] is not None else ["qualified_current_transaction"]),
        current_device_execution_count=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=plan["fixture"],
        genuine_headroom=False,
        positive_control_results=controls(),
        gate_check_summary=checks,
        historical_failed_operands=plan["historical_failed_operands"],
        preserved_historical_outcomes=plan["historical_outcomes"],
        cited_upstream_artifacts=plan["references"],
        acceptance_gate_results=dict(
            custody=custody, numeric=gates, numeric_pass=bool(passed), device_benefit=False
        ),
        sample_size_budget=sample,
        purchase_recommendation="No acquisition: no useful measured compatible device bottleneck or executable gradient kernel",
        existing_ising_fabric_gradient_compatible=False,
        external_vendor_numbers_are_local=False,
    )
    result.update(
        {
            k + "_count": sample[k]
            for k in (
                "intended",
                "eligible",
                "completed",
                "excluded",
                "failed",
                "censored",
                "independent",
            )
        }
    )
    return result
