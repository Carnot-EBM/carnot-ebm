"""REQ-REPORT-8016: retain custody without substituting old inference science."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.reporting import hardware_sparse_8003 as old
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.verify import fixedpoint_sparse_8003 as f
from carnot.verify import sparse_energy_7996 as s
from carnot.verify.typed_development_7997 import action

Json = dict[str, Any]
ROOT = old.ROOT
UPSTREAM = {
    8003: "experiment_8003_v693_hardware_sparse_boundary.json",
    8012: "experiment_8012_v694_budgeted_online_updates.json",
    8015: "experiment_8015_v694_native_update_parity.json",
}
PRIOR_HASH = "sha256:385d7f7ac2c12253f0c6a101039e4f554e18e47cd4ec2896df591d36edf27322"
CONFIG = dict(
    storage_bits=24,
    accumulator_bits=32,
    fractional_bits=12,
    scale=4096,
    probability_drift_max=0.001,
    natural_action_disagreements_max=0,
    overflow_max=0,
    rounding="nearest_ties_to_even",
    overflow="saturate",
    update_budget=4096,
    seed=69416,
    model_load_budget=0,
    device_execution_budget=0,
)


def fixture() -> Json:
    """Controls use a short ordered stream and add no independent observations."""
    prior = old.fixture()
    return dict(
        boards=prior["boards"],
        checks=[],
        cited_upstream_artifacts=[],
        fixture=True,
        trajectory=dict(
            head=prior["head"],
            fit=prior["data"]["fit"],
            updates=[
                dict(
                    id=r["family_id"],
                    source_cluster_id=r["source_cluster_id"],
                    x=[r["q"], *r["features"]],
                    y=r["y"],
                    learning_rate=0.01,
                )
                for r in prior["data"]["fit"]
            ],
        ),
        native_timing=None,
        historical_failed_operands=[],
    )


def authenticate(root: Path, raw: Path) -> Json:
    """Each board owns its bytes; optional trajectory absence cannot erase it."""
    checks: list[Json] = []
    refs: list[Json] = []
    path = old.seal(
        root,
        dict(path=str(root / "results" / UPSTREAM[8003]), sha256=PRIOR_HASH),
        raw,
        8003,
        checks,
        refs,
    )
    prior = json.loads(path.read_bytes()) if path else {}
    boards = deepcopy(prior.get("board_rows", old.fixture()["boards"]))
    for board in boards:
        before = len(checks)
        source = (
            old.seal(
                root,
                dict(path=board["source_path"], sha256=board["source_hash"]),
                raw,
                8003,
                checks,
                refs,
            )
            if path
            else None
        )
        receipt = json.loads(source.read_bytes()) if source else {}
        for label, digest in (
            (
                receipt.get("kv260_terminal_transcript_path"),
                receipt.get("kv260_terminal_transcript_sha256"),
            ),
            (
                receipt.get("raw_dispatch_transcript_path"),
                next(
                    (
                        r.get("latest_receipt_hash")
                        for r in receipt.get("board_rows", [])
                        if r.get("board") == "PolarFire"
                    ),
                    None,
                ),
            ),
        ):
            if label and digest:
                old.seal(
                    root,
                    dict(
                        path=label,
                        sha256=digest if digest.startswith("sha256:") else "sha256:" + digest,
                    ),
                    raw,
                    8003,
                    checks,
                    refs,
                )
        board.update(
            custody_valid=bool(source) and all(r["passed"] for r in checks[before:]),
            current_hardware_execution=False,
            compatible_update_kernel=False,
            last_actual_execution_date=receipt.get("run_date")
            if receipt.get("hardware_operations_issued")
            else None,
            last_actual_execution_hash=sha256_file(source)
            if source and receipt.get("hardware_operations_issued")
            else None,
            execution_date_absence_reason="Original execution timestamp absent or this receipt is read-only history",
            terminal_criterion=board.get(
                "terminal_criterion",
                "Authenticated compatible workload and complete service timing",
            ),
            next_missing_prerequisite=board.get(
                "next_missing_prerequisite",
                board.get("blocker") or "Measured compatible update kernel",
            ),
        )
        if receipt.get("kv260_terminal_transcript_sha256"):
            board["last_actual_execution_hash"] = "sha256:" + receipt[
                "kv260_terminal_transcript_sha256"
            ].removeprefix("sha256:")
        if receipt.get("hardware_operations_issued"):
            board["execution_date_absence_reason"] = None
    producers = {}
    for eid, field in ((8012, "learning_measurement_ready_score"), (8015, "parity_ready_score")):
        source = root / "results" / UPSTREAM[eid]
        exists = source.is_file()
        checks.append(
            old.operand(
                eid,
                source,
                "artifact.exists",
                True,
                exists,
                sha256_file(source) if exists else None,
            )
        )
        if exists:
            sealed = old.seal(
                root, dict(path=str(source), sha256=sha256_file(source)), raw, eid, checks, refs
            )
            value = json.loads(sealed.read_bytes())  # type: ignore[union-attr]
            for key, expected in (
                ("experiment_id", eid),
                (field, 1),
                ("flagged_adversarial", False),
            ):
                if key not in value:
                    raise ValueError("upstream_contract:" + key)
                checks.append(
                    old.operand(eid, source, key, expected, value[key], sha256_file(source))
                )
            producers[eid] = value
        elif eid == 8012:
            diagnostic = root / "results" / "experiment_8012_budgeted_online_updates.json"
            if diagnostic.is_file():
                sealed = old.seal(
                    root,
                    dict(path=str(diagnostic), sha256=sha256_file(diagnostic)),
                    raw,
                    eid,
                    checks,
                    refs,
                )
                skipped = json.loads(sealed.read_bytes())  # type: ignore[union-attr]
                for gate in skipped["gates_evaluated"]:
                    checks.append(
                        dict(
                            upstream_id=gate["upstream"],
                            artifact_path=gate["artifact_path"],
                            artifact_hash=gate["artifact_sha256"],
                            artifact_field=gate["artifact_field"],
                            op=gate["op"],
                            expected=gate["expected"],
                            observed=gate["actual"],
                            passed=gate["passed"],
                            scope="historical_conductor_block_diagnostic_not_trajectory",
                        )
                    )
    qualified = all(r["passed"] for r in checks if r["upstream_id"] in {"exp8012", "exp8015"})
    trajectory = None
    if qualified:
        ref = producers[8012]["checkpoints"]["trajectory"]
        sealed = old.seal(root, ref, raw, 8012, checks, refs)
        trajectory = json.loads(sealed.read_bytes()) if sealed else None
    return dict(
        boards=boards,
        checks=checks,
        cited_upstream_artifacts=refs,
        fixture=False,
        trajectory=trajectory,
        native_timing=producers.get(8015, {}).get("native_timing") if qualified else None,
        historical_failed_operands=prior.get("historical_failed_operands", []),
    )


def step(head: Json, row: Json, table: Json) -> tuple[Json, Json]:
    """24-bit stored values and a 32-bit sum expose loss at each conversion."""
    q, acc = f.Fixed(24), f.Fixed(32)
    scaled, _ = s.scale(np.asarray([row["x"]]), head["scaler"])
    ids, weights = [108], [4096]
    for feature, value in enumerate(scaled[0]):
        slot = f.nearest(table["basis"], q.encode(float(value)) / 4096)
        for index, weight in enumerate(table["basis_values"][slot]):
            if weight > 0:
                ids.append(feature * 12 + index)
                weights.append(q.encode(weight))
    theta = [q.encode(v) for v in head["parameters"]]
    decay, inv_t = q.encode(head["decay_scale"]), q.encode(1 / head["temperature"])
    offset = q.encode(
        table["logit_values"][f.nearest(table["q"], float(np.clip(row["x"][0], 1e-4, 1 - 1e-4)))]
    )

    def probability(coefficients: list[int], multiplier: int) -> float:
        z = offset
        for i, weight in zip(ids, weights, strict=True):
            z = acc.add(z, q.mul(weight, q.mul(coefficients[i], multiplier)))
        return float(expit(acc.mul(z, inv_t) / 4096))

    before = probability(theta, decay)
    residual = q.mul(q.encode(before - row["y"]), inv_t)
    next_decay = q.mul(decay, q.encode(1 - 0.002 * row["learning_rate"]))
    inverse, rate = q.encode(4096 / next_decay), q.encode(row["learning_rate"])
    for i, weight in zip(ids, weights, strict=True):
        theta[i] = q.add(theta[i], -q.mul(q.mul(rate, q.mul(weight, residual)), inverse))
    after = probability(theta, next_decay)
    changed = dict(head, parameters=[v / 4096 for v in theta], decay_scale=next_decay / 4096)
    return changed, dict(
        before=before,
        after=after,
        saturation_count=q.saturations + acc.saturations,
        storage_saturations=q.saturations,
        accumulator_saturations=acc.saturations,
        coefficient_reads=len(ids),
        coefficient_writes=len(ids),
        scale=4096,
    )


def reduce(plan: Json) -> Json:
    """Numerical validity and retained board history are separate terminal gates."""
    boards = deepcopy(plan["boards"])
    numeric: list[Json] = []
    trajectory = plan["trajectory"]
    restart_agrees = True
    started = time.monotonic()
    if trajectory:
        updates = trajectory["updates"]
        if not updates or len(updates) > CONFIG["update_budget"]:
            raise ValueError("trajectory_budget")
        table = f.grids(trajectory["head"], trajectory["fit"])
        float_head, fixed_head = deepcopy(trajectory["head"]), deepcopy(trajectory["head"])
        restarted = deepcopy(fixed_head)
        for index, row in enumerate(updates):
            before_hash = canonical_hash(fixed_head)
            x = np.asarray(row["x"], dtype=float)
            before = float(s.predict(float_head, x[None, :])[0])
            float_head, _ = s.update(float_head, x, row["y"], row["learning_rate"])
            after = float(s.predict(float_head, x[None, :])[0])
            fixed_head, metric = step(fixed_head, row, table)
            restarted, restart_metric = step(json.loads(json.dumps(restarted)), row, table)
            agrees = restarted == fixed_head and restart_metric == metric
            restart_agrees &= agrees
            drift = max(abs(before - metric["before"]), abs(after - metric["after"]))
            disagreements = int(action(before) != action(metric["before"])) + int(
                action(after) != action(metric["after"])
            )
            numeric.append(
                dict(
                    id=row["id"],
                    source_cluster_id=row["source_cluster_id"],
                    arm="signed24_q12_acc32",
                    seed=69416,
                    metric="max_probability_drift",
                    numerator=drift,
                    denominator=1,
                    status="completed",
                    exclusion_reason=None,
                    censor_reason=None,
                    fixture=plan["fixture"],
                    step=index,
                    float_probability_before=before,
                    float_probability_after=after,
                    fixed_probability_before=metric["before"],
                    fixed_probability_after=metric["after"],
                    float_actions=[action(before), action(after)],
                    fixed_actions=[action(metric["before"]), action(metric["after"])],
                    action_disagreements=disagreements,
                    state_before=before_hash,
                    state_after=canonical_hash(fixed_head),
                    checkpoint=dict(float_head=float_head, fixed_head=fixed_head),
                    restart_agrees=agrees,
                    **metric,
                )
            )
            if index % 32 == 0 or index == len(updates) - 1:
                print(
                    f"[exp8016] updates elapsed_s={time.monotonic() - started:.3f} completed={index + 1} pending={len(updates) - index - 1}",
                    flush=True,
                )
    gates = dict(
        max_probability_drift=max((r["numerator"] for r in numeric), default=None),
        action_disagreements=sum(r["action_disagreements"] for r in numeric),
        overflow=sum(r["saturation_count"] for r in numeric),
        restart_agrees=restart_agrees if numeric else None,
    )
    numeric_pass = (
        bool(numeric)
        and gates["max_probability_drift"] <= 0.001
        and gates["action_disagreements"] == gates["overflow"] == 0
        and restart_agrees
    )
    custody = all(b["custody_valid"] for b in boards)
    ready = custody and bool(trajectory)
    rows = [
        dict(
            id=b["board"],
            arm="board_custody",
            metric="authenticated_receipt",
            numerator=int(b["custody_valid"]),
            denominator=1,
            seed=None,
            status="completed" if b["custody_valid"] else "blocked",
            exclusion_reason=None,
            censor_reason=None,
            independent=0,
        )
        for b in boards
    ] + numeric
    if not trajectory:
        rows.append(
            dict(
                id="exp8012-update-trajectory",
                arm="natural_quantized_update",
                metric="max_probability_drift",
                numerator=None,
                denominator=0,
                seed=69416,
                status="blocked",
                exclusion_reason="No qualified exact update trajectory and native parity",
                censor_reason=None,
                independent=0,
            )
        )
    timing = plan["native_timing"] or {}
    components = {
        key: timing.get(key)
        for key in (
            "kernel_s",
            "host_s",
            "ffi_s",
            "storage_s",
            "historical_model_load_s",
            "historical_generation_s",
            "transfer_s",
        )
    }
    measured = all(components[k] is not None for k in ("kernel_s", "host_s", "ffi_s", "storage_s"))
    total = (
        sum(components[k] for k in ("kernel_s", "host_s", "ffi_s", "storage_s"))
        if measured
        else None
    )
    placements = [
        dict(
            operation=op,
            current_venue="CPU/Rust host",
            hypothetical_venue=venue,
            deployed_compatible_kernel=False,
            current_device_execution_count=0,
        )
        for op, venue in (
            ("coefficient_reads_writes", "FPGA BRAM gather/scatter; TSU compatibility unqualified"),
            ("spline_basis_logit_sigmoid", "FPGA LUT/arithmetic; Ising overlay incompatible"),
            ("pending_label_storage", "host keyed durable queue"),
            ("durable_commits", "host serialization/fsync"),
            ("typed_actions_and_FFI", "host CPU"),
        )
    ]
    return dict(
        honest_verdict="complete_circular_positive_update_fixture"
        if ready and plan["fixture"]
        else "complete_null_update_boundary_no_device_execution"
        if ready
        else "complete_blocked_no_qualified_update_trajectory_or_board_custody",
        verdict_class="circular_positive"
        if ready and plan["fixture"]
        else "null"
        if ready
        else "blocked",
        hardware_evidence_ready_score=int(ready),
        board_custody_ready_score=int(custody),
        quantized_update_ready_score=int(numeric_pass),
        board_rows=boards,
        cumulative_error_rows=numeric,
        overflow_rows=[
            dict(
                id=r["id"],
                saturation_count=r["saturation_count"],
                storage_saturations=r["storage_saturations"],
                accumulator_saturations=r["accumulator_saturations"],
            )
            for r in numeric
        ],
        workload_placement_rows=placements,
        acceleration_bounds=dict(
            whole_service_ideal=1.0 if total else None,
            current_device_compatible_fraction=0,
            hypothetical_kernel_ideal=total / (total - components["kernel_s"])
            if total and total > components["kernel_s"]
            else None,
            components=components,
            scope="No measured device gain; historical acquisition spans remain distinct",
        ),
        missing_cost_components=[k for k, v in components.items() if v is None],
        current_device_execution_count=0,
        rows=rows,
        sample_size_budget=dict(
            intended=len(rows),
            eligible=len(boards) + len(numeric),
            started=len(boards) + len(numeric),
            completed=sum(b["custody_valid"] for b in boards) + len(numeric),
            excluded=0,
            failed=sum(not b["custody_valid"] for b in boards),
            censored=int(not trajectory),
            independent=0 if plan["fixture"] else len({r["source_cluster_id"] for r in numeric}),
        ),
        acceptance_gate_results=dict(
            custody=custody,
            qualified_trajectory=bool(trajectory),
            numeric=gates,
            numeric_pass=numeric_pass,
            device_benefit=False,
        ),
        positive_control_results=dict(
            working=numeric_pass,
            scope="circular_cpu_protocol_control" if plan["fixture"] else "not_run",
        ),
        genuine_headroom=False,
        verifier_is_oracle=plan["fixture"],
        gate_check_summary=[r for r in plan["checks"] if not r["passed"]],
        preconditions_checked=plan["checks"],
        cited_upstream_artifacts=plan["cited_upstream_artifacts"],
        historical_failed_operands=plan["historical_failed_operands"],
        trained_head_specs=[dict(parameter_count=109, pretrained=False, current_fitting=False)]
        if trajectory
        else [],
        unqualified_substrates=["NPU", "TSU"],
        purchase_recommendation="No purchase without a measured compatible bottleneck",
        paper_operation_matches=[
            dict(
                url="https://arxiv.org/abs/2602.02056v4",
                match="Local spline coefficient updates motivate arithmetic testing; delayed labels and durable host commits remain outside the kernel",
                local_device_evidence=False,
            ),
            dict(
                url="https://www.nature.com/articles/s41467-026-75119-0",
                match="Tiled coordinate storage for sparse Ising matrix-vector operations does not implement spline nonlinearities or learning commits",
                local_device_evidence=False,
            ),
        ],
    )


def replay(value: Json) -> Json:
    """Fresh reduction verifies raw checkpoints, receipts and all derived fields."""
    for ref in (
        value.get("raw_shard_hashes", [])
        + value.get("code_config_hashes", [])
        + value["replay_inputs"]["cited_upstream_artifacts"]
    ):
        path = Path(ref["path"])
        if not path.is_file() or sha256_file(path) != ref["sha256"]:
            raise ValueError("receipt_drift:" + str(path))
    for receipt in value.get("validation_receipts", []):
        if (
            receipt.get("log_path")
            and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
        ):
            raise ValueError("receipt_drift:validation")
    fresh = reduce(value["replay_inputs"])
    for ref in value.get("raw_shard_hashes", []):
        path = Path(ref["path"])
        expected_raw = {
            "replay_inputs.json": value["replay_inputs"],
            "primitive_rows.json": dict(rows=fresh["rows"]),
            "update_checkpoints.json": [r["checkpoint"] for r in fresh["cumulative_error_rows"]],
        }
        if path.name in expected_raw and json.loads(path.read_bytes()) != expected_raw[path.name]:
            raise ValueError("raw_reduction_drift:" + path.name)
    for key, expected in fresh.items():
        if value["verdict_class"] == "disqualified" and key in {
            "verdict_class",
            "honest_verdict",
            "hardware_evidence_ready_score",
        }:
            continue
        if value.get(key) != expected:
            raise ValueError("reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(fresh["rows"]))
