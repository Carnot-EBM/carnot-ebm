"""REQ-VERIFY-8244: CPU coefficient sensitivity does not qualify a fabric kernel.

The existing CPU scorer supplies bases, probabilities and restricted decisions.
Only coefficient storage changes; all nonlinear and durable work stays on host.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.request_trace_inventory_8200 import copy_bytes, operand
from carnot.verify import margin_energy_training_8237 as cpu

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8244_v712_kv260_decision_boundary"
CLI = "scripts/experiments/" + NAME + ".py"
CONFIG: Json = dict(
    seed=7128244,
    bits=[8, 16],
    thresholds=[0.1, 1 / 6, 0.5],
    arithmetic="float64 CPU; signed coefficient storage only",
    rounding="nearest ties to even",
    approximation_deployed=False,
)
MODEL_SPECS: list[Json] = []
PRODUCERS = dict(
    historical=(8230, "v711_kv260_workload_boundary", "kv260_boundary_ready_score"),
    heads=(8237, "v712_margin_energy_training", "margin_fit_ready_score"),
    service=(8242, "v712_independent_concurrent_service", "concurrent_service_ready_score"),
)
PINS = {
    8230: "sha256:dc98bfa0f84e104175aa8fbb7f77c7e6450ecdb0b8f26f5c4c95f262f4f1f75b",
    8237: "sha256:0e4b9317c60d1184315bfca2cce58f6ab4e07f2cf7425563a6cb27d3b6c93e86",
    8242: "sha256:4a0d189e5f18656b9ab2817c4717ac1adf592097be2b2120145139f2133f2062",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so a supervisor can distinguish finite work from a stalled child."""
    print(f"[exp8244] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(path: Path, raw: Path, data: Json) -> Path:
    """Read copied bytes so a concurrent producer cannot change this invocation."""
    ref = copy_bytes(path, raw)
    data["references"].append(
        dict(path=ref["frozen_path"], sha256=ref["sha256"], original_path=str(path))
    )
    return Path(ref["frozen_path"])


def validate_heads(heads: list[Json]) -> None:
    """Malformed coefficients or temperatures cannot become a numerical measurement."""
    if (
        not heads
        or len({h["arm"] for h in heads}) != len(heads)
        or any(
            h["basis"] not in {"energy", "additive", "logistic"}
            or len(h["weights"]) != 17
            or not np.isfinite(h["weights"]).all()
            or not math.isfinite(h["temperature"])
            or h["temperature"] <= 0
            for h in heads
        )
    ):
        raise ValueError("head_schema")


def import_reference(ref: Json, field: str, raw: Path, data: Json) -> Json:
    """Name the exact missing or changed operand before attempting to read it."""
    path = Path(ref["path"])
    data["checks"].append(
        operand(field, path, ref["sha256"], sha256_file(path) if path.is_file() else None)
    )
    value: Json = json.loads(freeze(checked(ref), raw, data).read_bytes())
    return value


def load(root: Path, raw: Path) -> Json:
    """Check branches separately so unavailable science never deletes board custody."""
    data: Json = dict(
        checks=[],
        references=[],
        cited=[],
        branches={},
        board={},
        heads=[],
        samples=[],
        requests=[],
        cold_costs=[],
    )
    for index, (branch, (eid, suffix, score)) in enumerate(PRODUCERS.items()):
        progress("authenticate_before_" + branch, index, 3 - index)
        path = root / "results" / f"experiment_{eid}_{suffix}.json"
        begin = len(data["checks"])
        data["checks"].append(operand("exists", path, True, path.is_file()))
        try:
            value = json.loads(freeze(path, raw, data).read_bytes())
            validate_primary(value, path)
            for field, expected in [
                ("sha256", PINS[eid]),
                (score, 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
                ("fixture_mode", False),
            ]:
                actual = sha256_file(path) if field == "sha256" else value.get(field, False)
                data["checks"].append(operand(field, path, expected, actual))
            side = (
                path.parent
                / "raw"
                / path.stem
                / "validators"
                / (sha256_file(path).split(":")[1] + ".json")
            )
            report = read_bound_sidecar(path, side)
            freeze(side, raw, data)
            data["checks"].append(
                operand("terminal_report_passed", path, True, report["report"]["passed"])
            )
            data["checks"].append(
                operand("terminal_primary_path", path, str(path.absolute()), report["primary_path"])
            )
            if not all(c["passed"] for c in data["checks"][begin:]):
                raise ValueError("upstream_gate")
            if branch == "historical":
                board = deepcopy(value["kv260_obligation"]["historical"])
                for field, hashfield in [
                    ("source_path", "source_hash"),
                    ("source_transcript", "source_transcript_sha256"),
                ]:
                    if board.get(field):
                        source = Path(board[field])
                        source = source if source.is_absolute() else root / source
                        freeze(checked(dict(path=str(source), sha256=board[hashfield])), raw, data)
                if board.get("custody_valid") is not True or board.get("k_max") != 5:
                    raise ValueError("historical_custody")
                data["board"] = board
            if branch == "heads":
                ref = dict(path=value["trained_heads_path"], sha256=value["trained_heads_sha256"])
                payload = import_reference(ref, "trained_heads_sha256", raw, data)
                work = import_reference(value["work_reference"], "work_reference.sha256", raw, data)
                if (
                    payload["schema"] != "carnot.v712.margin-heads.v1"
                    or payload["heads"] != work["fitted"]["heads"]
                ):
                    raise ValueError("head_work_binding")
                validate_heads(payload["heads"])
                data["heads"] = payload["heads"]
                data["samples"] = [r for r in work["public_rows"] if r["role"] != "reserved"]
            if branch == "service":
                ref = next(
                    r for r in value["raw_shard_hashes"] if r["path"] == value["request_rows_path"]
                )
                rows = import_reference(ref, "request_rows_sha256", raw, data)["rows"]
                if rows != value["rows"]:
                    raise ValueError("request_row_binding")
                data["requests"], data["cold_costs"] = rows, value["cold_costs"]
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            data["checks"].append(
                operand(
                    "authenticated_schema_and_primitives",
                    path,
                    "byte-bound valid schema",
                    str(error),
                )
            )
        data["branches"][branch] = all(c["passed"] for c in data["checks"][begin:])
        data["cited"].append(
            dict(
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else None,
                imported_fields=[
                    score,
                    "kv260_obligation",
                    "trained_heads_path",
                    "work_reference",
                    "rows",
                    "cold_costs",
                ],
            )
        )
        progress("authenticate_after_" + branch, index + 1, 2 - index)
    return data


def precision(data: Json) -> list[Json]:
    """Reuse the CPU scorer on rounded copies; exact fallback protects decision ties.

    The same sigmoid slope bound used by the existing CPU precision evaluator
    bounds coefficient residuals. Bases and temperature remain float64, so this
    does not test integer exponentials, normalization or a device implementation.
    """
    measured = []
    for head in data["heads"]:
        progress("benchmark_before_" + head["arm"], 0, len(data["samples"]))
        for index, source in enumerate(data["samples"]):
            query = {k: source[k] for k in ["x", "p0", "baseline_action"]}
            reference = cpu.score(head, query)
            for bits in CONFIG["bits"]:
                row = dict(
                    unit_id=source["unit_id"],
                    source_cluster_id=source["source_cluster_id"],
                    arm=head["arm"],
                    bits=bits,
                    condition=source["role"],
                    metric="raw_action_changed",
                    numerator=None,
                    denominator=1,
                    status="excluded",
                    exclusion_reason="missing_public_features_or_probability",
                )
                if reference["p"] is not None:
                    weights = np.asarray(head["weights"], dtype=float)
                    step = max(float(np.max(abs(weights))), 1e-30) / (2 ** (bits - 1) - 1)
                    integers = np.rint(weights / step).astype(np.int64)
                    rounded = deepcopy(head)
                    rounded["weights"] = (integers * step).tolist()
                    approximate = cpu.score(rounded, query)
                    phi = cpu.rule.design(
                        head["basis"], np.asarray([source["x"]]), head["geometry"]
                    )[0]
                    radius = (
                        float(np.sum(abs(phi) * abs(weights - integers * step)))
                        / (4 * head["temperature"])
                        + 1e-12
                    )
                    changed = int(approximate["action"] != reference["action"])
                    fallback = (
                        changed
                        or min(abs(approximate["p"] - t) for t in CONFIG["thresholds"]) <= radius
                    )
                    final = reference["action"] if fallback else approximate["action"]
                    y = source["y"]
                    false_delta = (
                        (
                            int(approximate["action"] == "accept" and y == 1)
                            - int(reference["action"] == "accept" and y == 1)
                        )
                        if y in (0, 1)
                        else None
                    )
                    row.update(
                        status="completed",
                        exclusion_reason=None,
                        numerator=changed,
                        probability_reference=reference["p"],
                        probability_quantized=approximate["p"],
                        probability_error=abs(reference["p"] - approximate["p"]),
                        probability_error_bound=radius,
                        raw_action_changed=changed,
                        final_action_changed=int(final != reference["action"]),
                        baseline_action=source["baseline_action"],
                        reference_action=reference["action"],
                        raw_action=approximate["action"],
                        final_action=final,
                        fallback=bool(fallback),
                        false_accept_delta=false_delta,
                        false_accept_denominator=int(y in (0, 1)),
                        coefficient_scale=step,
                        coefficient_integers=integers.tolist(),
                        coefficient_storage_bytes=bits * len(weights) // 8,
                        software_bound_only=True,
                        fixedpoint_arithmetic_executed=False,
                    )
                measured.append(row)
            if (index + 1) % 32 == 0 or index + 1 == len(data["samples"]):
                progress("precision_" + head["arm"], index + 1, len(data["samples"]) - index - 1)
        progress("benchmark_after_" + head["arm"], len(data["samples"]), 0)
    return measured


def costs(data: Json) -> tuple[list[Json], Json]:
    """Remove scoring optimistically while charging issue-to-durability and cold costs.

    Summed request work is distinct from concurrent wall throughput. Failed
    requests still consume workload costs, but never qualify a completion gain.
    """
    rows = []
    fields = [
        "acquisition_s",
        "scoring_s",
        "service_s",
        "queue_s",
        "issue_fsync_s",
        "terminal_fsync_s",
    ]
    for request in data["requests"]:
        clocks = request.get("clocks", {})
        valid = all(type(clocks.get(k)) is int for k in ["issue", "durability"]) and all(
            type(request.get(k)) in {int, float} and math.isfinite(request[k]) and request[k] >= 0
            for k in fields
        )
        total = (clocks["durability"] - clocks["issue"]) / 1e9 if valid else None
        valid = bool(
            valid
            and total is not None
            and total > 0
            and request["scoring_s"] < total
            and request["scoring_s"] <= request["service_s"]
            and request["acquisition_s"] + request["service_s"] <= total + 1e-9
        )
        complete = valid and request["status"] == "completed"
        rows.append(
            dict(
                unit_id=request["request_id"],
                source_cluster_id=request["source_cluster_id"],
                arm=request["arm"],
                condition=request["workload"],
                metric="whole_request_gain_bound",
                status="completed" if complete else "excluded",
                numerator=total if complete else None,
                denominator=total - request["scoring_s"] if complete else None,
                exclusion_reason=None if complete else "unqualified_clocks_or_failed_request",
                upstream_request_status=request["status"],
                clocks_qualified=valid,
                total_s=total,
                removable_scoring_s=request.get("scoring_s"),
                retained_s=total - request["scoring_s"] if valid else None,
                maximum_gain=total / (total - request["scoring_s"]) if complete else None,
                observed_spans={k: request.get(k) for k in fields},
                clocks=clocks,
            )
        )
    complete_rows = [r for r in rows if r["status"] == "completed"]
    cold = data["cold_costs"]
    cold_valid = bool(
        rows
        and cold
        and {r["condition"] for r in rows} == {c["workload"] for c in cold}
        and all(r["clocks_qualified"] for r in rows)
        and all(
            type(c.get(k)) in {int, float} and math.isfinite(c[k]) and c[k] >= 0
            for c in cold
            for k in ["startup_s", "shutdown_s"]
        )
    )
    total = (
        sum(r["total_s"] for r in rows) + sum(c["startup_s"] + c["shutdown_s"] for c in cold)
        if cold_valid
        else None
    )
    scoring = sum(r["removable_scoring_s"] for r in rows) if cold_valid else None
    return rows, dict(
        status="available" if complete_rows else "unavailable",
        maximum_request_gain=max((r["maximum_gain"] for r in complete_rows), default=None),
        qualified_request_count=len(complete_rows),
        intended_request_count=len(rows),
        cold_inclusive_work_sum_s=total,
        removable_scoring_s=scoring,
        cold_inclusive_work_sum_gain=total / (total - scoring) if cold_valid else None,
        cold_costs=cold,
        existing_fabric_gain=1.0 if complete_rows else None,
        practical_device_gain=None,
        software_bound_only=True,
        scope="optimistic complete scoring elimination; summed request work plus cold costs, not concurrent throughput",
    )


def reduce(data: Json, measured: list[Json]) -> Json:
    """Readiness certifies this boundary; exposed-source arithmetic earns no science win."""
    complete = [r for r in measured if r["status"] == "completed"]
    safe = all(
        r["final_action_changed"] == 0
        and r["probability_error"] <= r["probability_error_bound"]
        and (r["final_action"] != "accept" or r["baseline_action"] == "accept")
        for r in complete
    )
    service, bound = costs(data)
    board = data["board"]
    ready = int(safe and board.get("custody_valid") is True and board.get("k_max") == 5)
    checks = list(data["checks"])
    if not complete:
        checks.append(
            operand(
                "qualified_precision_rows",
                ROOT / "results/experiment_8237_v712_margin_energy_training.json",
                1,
                0,
                ">=",
            )
        )
    if not any(r["status"] == "completed" for r in service):
        checks.append(
            operand(
                "rows.clocks.issue_to_durability",
                ROOT / "results/experiment_8242_v712_independent_concurrent_service.json",
                "qualified complete current request spans",
                None,
            )
        )
    blocked = (
        not all(data["branches"].values())
        or not complete
        or bound["status"] == "unavailable"
        or not ready
    )
    kind = (
        "disqualified"
        if not safe
        else "blocked"
        if blocked
        else "circular_positive"
        if data.get("fixture")
        else "null"
    )
    failures = [c for c in checks if not c["passed"]]
    missing = [
        dict(
            unit_id=branch + "_obligation",
            source_cluster_id=branch,
            arm=branch,
            condition="upstream_availability",
            metric="available",
            numerator=None,
            denominator=1,
            status="excluded",
            exclusion_reason="unqualified_upstream",
        )
        for branch, valid in sorted(data["branches"].items())
        if not valid
    ]
    rows = measured + service + missing
    summaries = []
    groups = sorted({(r["arm"], r["bits"], r["source_cluster_id"]) for r in measured})
    for arm, bits, source in groups:
        selected = [
            r
            for r in measured
            if (r["arm"], r["bits"], r["source_cluster_id"]) == (arm, bits, source)
        ]
        available = [r for r in selected if r["status"] == "completed"]
        deltas = [r["false_accept_delta"] for r in available if r["false_accept_delta"] is not None]
        summaries.append(
            dict(
                arm=arm,
                bits=bits,
                source_cluster_id=source,
                intended_count=len(selected),
                completed_count=len(available),
                excluded_count=len(selected) - len(available),
                probability_error_numerator=sum(r["probability_error"] > 0 for r in available),
                probability_error_denominator=len(available),
                maximum_probability_error=max(
                    (r["probability_error"] for r in available), default=None
                ),
                changed_action_numerator=sum(r["raw_action_changed"] for r in available),
                changed_action_denominator=len(available),
                false_accept_increases=sum(d > 0 for d in deltas),
                false_accept_decreases=sum(d < 0 for d in deltas),
                false_accept_delta=sum(deltas) if deltas else None,
                false_accept_denominator=len(deltas),
                fallback_count=sum(r["fallback"] for r in available),
            )
        )
    operations = [
        dict(
            operation=name,
            existing_fabric_supported=False,
            execution="CPU",
            missing_device_evidence=why,
        )
        for name, why in [
            (
                "coefficient_multiply_accumulate",
                "Authenticated head dispatch and scaling; not quadratic spin evaluation",
            ),
            ("gaussian_exponential", "16-center distance/exponential operator and parity"),
            ("tanh_additive_basis", "Nonlinear basis implementation and precision bounds"),
            (
                "feature_standardization",
                "Host feature extraction, mean/scale and transfer accounting",
            ),
            (
                "sigmoid_energy_normalization",
                "Exponentials, log energies, normalization and temperature parity",
            ),
            (
                "permission_threshold_fallback",
                "Original acceptance mask, ties and exact CPU fallback costs",
            ),
            ("durable_host_persistence", "Version binding, journal, fsync, restart and readout"),
        ]
    ]
    return dict(
        honest_verdict="complete_"
        + kind
        + "_"
        + (
            failures[0]["artifact_field"]
            if kind == "blocked" and failures
            else "kv260_decision_boundary"
        ),
        verdict_class=kind,
        kv260_boundary_ready_score=ready,
        branch_readiness=data["branches"],
        precision_rows=measured,
        precision_source_summaries=summaries,
        operation_compatibility=operations,
        measured_service_spans=service,
        whole_request_bound=bound,
        gate_check_summary=checks,
        kv260_obligation=dict(
            historical=board,
            k_max=5,
            current_reachability="not_probed",
            current_board_execution=False,
            supported_operation="historical quadratic Ising fabric only",
            eligible_head_operations=[],
            future_access_command=["ssh", "kria"],
            reopen_evidence="Useful k<=5 quadratic workload with authenticated dispatch, precision parity and transfer-inclusive complete-request clocks",
        ),
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        failed_count=0,
        censored_count=0,
        independent_count=0,
        original_public_source_count=len({r["source_cluster_id"] for r in complete}),
        verifier_is_oracle=bool(data.get("fixture")),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        current_device_execution_count=0,
        acceptance_gates=dict(
            guarded_cpu_parity=safe,
            boundary_execution=bool(ready),
            head_bytes_qualified=data["branches"]["heads"],
            service_bytes_qualified=data["branches"]["service"],
        ),
        trained_head_specs=[
            dict(
                arm=h["arm"],
                basis=h["basis"],
                coefficient_count=len(h["weights"]),
                head_sha256=canonical_hash(h),
                current_fit=False,
                generator=False,
                source_experiment_id=8237,
            )
            for h in data["heads"]
        ],
    )
