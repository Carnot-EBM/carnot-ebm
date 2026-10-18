"""REQ-VERIFY-8230: cached utility arithmetic does not execute the Ising fabric.

The host reference tests rounding and action safety. Hardware support, source
availability and whole-request benefit each retain their own evidence boundary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.request_trace_inventory_8200 import copy_bytes, operand
from carnot.verify import utility_kernel_8221 as kernel
from carnot.verify import utility_patch_methods_8219 as frozen

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8230_v711_kv260_workload_boundary"
CLI = "scripts/experiments/" + NAME + ".py"
CONFIG: Json = dict(
    seed=7118230,
    format="signed Q16.16",
    scale=65536,
    rounding="nearest ties to even",
    clip_integer=[1, 65535],
    thresholds=[0.1, 1 / 6, 0.5],
    approximation_deployed=False,
)
PRODUCERS = {
    "historical": (8216, "v709_hardware_workload_obligations", "hardware_boundary_ready_score"),
    "kernel": (8221, "v711_utility_kernel", "static_kernel_ready_score"),
    "learning": (8225, "v711_delayed_utility_learning", "utility_trajectory_ready_score"),
    "service": (8228, "v711_concurrent_service", "concurrent_service_ready_score"),
}
PINS = {
    8216: "sha256:5b31ca2cca90b76d47f7c1b9575264e352a3990ba303992c8083f97c0fff823f",
    8221: "sha256:5ff6f09312a81ce95b752d38dec94d8a1c57bbf894cde785ebe7494ca7c639e5",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries distinguish finite host work from a silent child."""
    print(f"[exp8230] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(path: Path, raw: Path, data: Json) -> Path:
    """Save exact historical bytes so later producers cannot change this reduction."""
    ref = copy_bytes(path, raw)
    data["references"].append(
        dict(path=ref["frozen_path"], sha256=ref["sha256"], original_path=str(path))
    )
    return Path(ref["frozen_path"])


def load(root: Path, raw: Path) -> Json:
    """Check each branch independently; missing learning cannot erase the board."""
    data: Json = dict(
        checks=[], references=[], cited=[], branches={}, board={}, requests=[], cases=[]
    )
    for index, (name, (eid, suffix, score)) in enumerate(PRODUCERS.items()):
        progress("authenticate_before_" + name, index, 4 - index)
        path = root / "results" / f"experiment_{eid}_{suffix}.json"
        begin = len(data["checks"])
        value: Json = {}
        data["checks"].append(operand("exists", path, True, True if path.is_file() else None))
        try:
            value = json.loads(freeze(path, raw, data).read_bytes())
            for field, expected, observed in [
                ("object_schema", True, isinstance(value, dict)),
                ("experiment_id", eid, value.get("experiment_id")),
                (
                    "task_id",
                    f"exp{eid}-" + suffix.split("_", 1)[1].replace("_", "-"),
                    value.get("task_id"),
                ),
                (score, 1, value.get(score)),
                ("required_checks_passed", True, value.get("required_checks_passed")),
                ("flagged_adversarial", False, value.get("flagged_adversarial")),
                ("fixture_mode", False, bool(value.get("fixture_mode") or value.get("fixture"))),
                (
                    "verdict_class_qualified",
                    True,
                    value.get("verdict_class") in {"null", "positive", "circular_positive"},
                ),
                ("sha256", PINS.get(eid, sha256_file(path)), sha256_file(path)),
            ]:
                data["checks"].append(operand(field, path, expected, observed))
            terminal = Path(value["terminal_validation_sidecar_path"])
            receipt = json.loads(freeze(terminal, raw, data).read_bytes())["publication"]
            sidecar = Path(receipt["sidecar_path"])
            report = read_bound_sidecar(path, sidecar)
            freeze(sidecar, raw, data)
            data["checks"].append(
                operand(
                    "publication.primary_path", path, str(path.absolute()), report["primary_path"]
                )
            )
            data["checks"].append(
                operand("terminal_report_passed", path, True, report["report"]["passed"])
            )
            valid = all(c["passed"] for c in data["checks"][begin:])
            if valid and name == "historical":
                board = next(b for b in value["board_rows"] if b["board"] == "KV260")
                if board.get("source_transcript"):
                    freeze(
                        checked(
                            dict(
                                path=board["source_transcript"],
                                sha256=board["source_transcript_sha256"],
                            )
                        ),
                        raw,
                        data,
                    )
                data["board"] = board
            if valid and name == "kernel":
                work = json.loads(
                    freeze(checked(value["measurement_reference"]), raw, data).read_bytes()
                )
                restart = next(
                    r
                    for r in value["raw_shard_hashes"]
                    if Path(r["path"]).name == "restart-input.json"
                )
                rows = json.loads(freeze(checked(restart), raw, data).read_bytes())["rows"]
                model = work["static"]["model"]
                natural = [
                    dict(r, p=r["baseline_p"]) for r in work["protocol"]["public_fit_membership"]
                ]
                data["cases"] = [
                    dict(scope="fixture", arm="frozen_static", model=model, rows=rows),
                    dict(
                        scope="natural_public_fixture_model",
                        arm="frozen_static",
                        model=model,
                        rows=natural,
                    ),
                ]
            if valid and name == "service":
                data["requests"] = value.get("whole_request_spans", [])
        except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as error:
            data["checks"].append(
                operand(
                    "authenticated_schema_and_primitives",
                    path,
                    "byte-bound valid schema",
                    str(error),
                )
            )
        checks = data["checks"][begin:]
        data["branches"][name] = dict(
            ready=all(c["passed"] for c in checks),
            verdict=value.get("honest_verdict") if isinstance(value, dict) else None,
            fields={
                k: value.get(k)
                for k in [
                    score,
                    "verdict_class",
                    "required_checks_passed",
                    "flagged_adversarial",
                    "owned_failure",
                    "fixture_mode",
                ]
            }
            if isinstance(value, dict)
            else {},
            failed_operands=[c for c in checks if not c["passed"]],
        )
        data["cited"].append(
            dict(
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else None,
                imported_fields=[
                    score,
                    "verdict_class",
                    "terminal_validation_sidecar_path",
                    "board_rows",
                    "measurement_reference",
                    "raw_shard_hashes",
                    "whole_request_spans",
                ],
            )
        )
        progress("authenticate_after_" + name, index + 1, 3 - index)
    return data


def qpredict(model: Json, row: Json) -> tuple[float | None, bool]:
    """Integer adds clip in saved order; unsupported mixtures stay exact on CPU."""
    if row["p"] is None:
        return None, False
    p = kernel.clip(row["p"])
    if model["kind"] == "input":
        return min(65535, max(1, round(p * 65536))) / 65536, False
    if model["kind"] == "mixture":
        return kernel.predict(model, row), True
    if model["kind"] != "patch":
        raise ValueError("model_kind")
    base, unsupported = qpredict(model["base"], row)
    assert base is not None
    q = round(base * 65536)
    baseline = row["baseline_p"]
    qb = round(baseline * 65536) if baseline is not None else None
    for op in model["patches"]:
        group, interval = op["group"], op["group"]["interval"]
        member = (
            interval is None
            or qb is not None
            and round(interval[0] * 65536) <= qb
            and (qb < round(interval[1] * 65536) or qb == 65536 == round(interval[1] * 65536))
        )
        member = member and (not group["reject_only"] or row["baseline_action"] == "reject")
        unsupported |= bool(member) != frozen.member(row, group)
        q = min(65535, max(1, q + round(op["delta"] * 65536) * bool(member)))
    return q / 65536, unsupported


def precision(data: Json) -> list[Json]:
    """Shadow comparisons retain raw errors; CPU fallback preserves final actions."""
    measured = []
    for case in data["cases"]:
        progress("benchmark_before_" + case["scope"], 0, len(case["rows"]))
        for index, row in enumerate(case["rows"]):
            p = kernel.predict(case["model"], row)
            q, unsupported = qpredict(case["model"], row)
            result = dict(
                unit_id=row["unit_id"],
                source_cluster_id=row["source_cluster_id"],
                arm=case["arm"],
                scope=case["scope"],
                condition=case["scope"],
                metric="raw_action_difference",
                denominator=1,
                status="excluded",
                numerator=None,
                exclusion_reason="missing_probability",
            )
            if p is not None and q is not None:
                reference_action = kernel.rule.action(p, row["baseline_action"])
                raw_action = kernel.rule.action(q, row["baseline_action"])
                radius = abs(p - q) + 1 / 65536
                fallback = (
                    unsupported
                    or min(abs(q - t) for t in CONFIG["thresholds"]) <= radius
                    or raw_action != reference_action
                )
                final_action = reference_action if fallback else raw_action
                result.update(
                    status="completed",
                    exclusion_reason=None,
                    numerator=int(raw_action != reference_action),
                    probability_fp64=p,
                    probability_q16_16=q,
                    absolute_error=abs(p - q),
                    shadow_error_radius=radius,
                    fallback=fallback,
                    unsupported_or_lookup_difference=unsupported,
                    fallback_policy="shadow FP64 error plus one LSB; no deployed approximation",
                    reference_action=reference_action,
                    raw_action=raw_action,
                    final_action=final_action,
                    baseline_action=row["baseline_action"],
                    final_mismatch=int(final_action != reference_action),
                )
            measured.append(result)
            if (index + 1) % 32 == 0 or index + 1 == len(case["rows"]):
                progress("precision_" + case["scope"], index + 1, len(case["rows"]) - index - 1)
        progress("benchmark_after_" + case["scope"], len(case["rows"]), 0)
    return measured


def bounds(requests: list[Json]) -> list[Json]:
    """Amdahl uses the same request's full partition; absent clocks stay absent."""
    result = []
    for r in requests or [dict(unit_id="missing_service_spans")]:
        fields = [
            "total_ns",
            "lookup_add_ns",
            "acquisition_ns",
            "transfer_ns",
            "readout_ns",
            "durable_host_ns",
            "unsupported_cpu_ns",
        ]
        valid = all(
            isinstance(r.get(k), (int, float))
            and not isinstance(r.get(k), bool)
            and math.isfinite(r[k])
            and r[k] >= 0
            for k in fields
        )
        valid = (
            valid
            and r["total_ns"] > 0
            and sum(r[k] for k in fields[1:]) == r["total_ns"]
            and r["lookup_add_ns"] < r["total_ns"]
        )
        b = dict(
            unit_id=r["unit_id"],
            status="available" if valid else "unavailable",
            observed_spans=r,
            exact_field="whole_request_spans.disjoint_component_ns",
            ideal_upper_bound=None,
            practical_estimate=None,
            eligible_fraction=None,
            existing_fabric_eligible_ns=0,
            existing_fabric_upper_bound=1.0 if valid else None,
            scope="hypothetical lookup/add elimination; existing Ising utility support is zero",
        )
        if valid:
            f = r["lookup_add_ns"] / r["total_ns"]
            b.update(eligible_fraction=f, ideal_upper_bound=1 / (1 - f))
            device = ["device_lookup_add_ns", "board_transfer_ns", "board_readout_ns"]
            if all(
                isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) and r[k] >= 0
                for k in device
            ):
                b["practical_estimate"] = r["total_ns"] / (
                    r["total_ns"] - r["lookup_add_ns"] + sum(r[k] for k in device)
                )
        result.append(b)
    return result


def reduce(data: Json, measured: list[Json]) -> Json:
    """Readiness describes checked boundaries while missing science stays blocked."""
    operations = [
        dict(operation=name, existing_fabric_supported=False, execution="CPU", reason=reason)
        for name, reason in [
            (
                "interval_lookup",
                "Frozen baseline bins and permission predicates are absent from the quadratic fabric.",
            ),
            (
                "ordered_clipped_adds",
                "Sequential probability clipping is not quadratic spin energy evaluation.",
            ),
            ("global_calibration", "Scale/intercept and sigmoid require CPU arithmetic."),
            ("log_normalization", "Logarithms are unsupported by the existing dispatch."),
            ("probability_mixture", "Saved final-probability interpolation remains CPU."),
            (
                "durable_host_state",
                "Versioned state, journals, fsync and restart remain host obligations.",
            ),
        ]
    ]
    complete = [r for r in measured if r["status"] == "completed"]
    parity = all(
        r["final_mismatch"] == 0
        and r["numerator"] == int(r["raw_action"] != r["reference_action"])
        and (r["final_action"] != "accept" or r["baseline_action"] == "accept")
        for r in complete
    )
    board = data["board"]
    ready = int(
        parity and bool(complete) and board.get("custody_valid") is True and board.get("k_max") == 5
    )
    limits = bounds(data["requests"])
    blocked = (
        any(not b["ready"] for b in data["branches"].values())
        or any(b["status"] == "unavailable" for b in limits)
        or not ready
    )
    kind = (
        "disqualified"
        if not parity
        else "blocked"
        if blocked
        else "circular_positive"
        if data.get("fixture")
        else "null"
    )
    missing = [
        dict(
            unit_id=name + "_branch",
            source_cluster_id=name,
            arm=name,
            condition="upstream_branch",
            status="excluded",
            metric="branch_available",
            numerator=None,
            denominator=1,
            exclusion_reason=b.get("failed_operands") or "branch_unavailable",
        )
        for name, b in data["branches"].items()
        if not b["ready"]
    ]
    rows = measured + missing
    summaries = []
    for scope in sorted({r["scope"] for r in measured}):
        selected = [r for r in measured if r["scope"] == scope]
        available = [r for r in selected if r["status"] == "completed"]
        summaries.append(
            dict(
                scope=scope,
                intended=len(selected),
                available=len(available),
                missing=len(selected) - len(available),
                action_difference_numerator=sum(r["numerator"] for r in available),
                action_difference_denominator=len(available),
                fallback_numerator=sum(r["fallback"] for r in available),
                fallback_denominator=len(available),
                fallback_rate=sum(r["fallback"] for r in available) / len(available)
                if available
                else None,
            )
        )
    return dict(
        honest_verdict="complete_"
        + kind
        + (
            "_service_spans_or_external_operand"
            if blocked and parity
            else "_kv260_workload_boundary"
        ),
        verdict_class=kind,
        kv260_boundary_ready_score=ready,
        branch_readiness=data["branches"],
        operation_mapping=operations,
        precision_rows=measured,
        precision_summaries=summaries,
        whole_request_bounds=limits,
        kv260_obligation=dict(
            historical=board,
            k_max=5,
            current_reachability="not_probed",
            current_board_execution=False,
            supported_operation="historical quadratic Ising fabric only",
            eligible_utility_operations=[],
            board_experiment_justified=False,
            future_access_command=["ssh", "kria"],
            reopen_evidence="Authenticated useful k<=5 quadratic workload and same-request eligible spans; compatible existing dispatch; measured transfer/readout and durable-host costs; action/fallback parity. Current lookup/add evidence supplies no compatible dispatch.",
            TSU_access="unqualified",
            NPU_access="unqualified",
        ),
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=0,
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        independent_count=0,
        verifier_is_oracle=True,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        acceptance_gates=dict(
            final_action_parity=parity,
            supported_utility_dispatch=False,
            measured_service_acceleration=False,
        ),
        approximation_deployed=False,
        current_device_execution_count=0,
    )
