"""REQ-REPORT-7990: preserve evidence without enlarging executable board scope.

A fitted polynomial or spline head is not the qualified Ising sampler. Exact
source bytes support placement estimates, while current device work stays zero.
"""

from copy import deepcopy
from datetime import UTC, datetime
import math
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import experiment_7977_v691_hardware_evidence as history
from carnot.reporting import service_cost_7989 as service
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7862_v682_hardware_evidence import _check

ROOT = history.ROOT
PRIOR = "results/experiment_7977_v691_hardware_evidence.json"
SERVICE = "results/experiment_7989_v692_service_cost.json"
PRIOR_HASH = "sha256:15147d4b31b7a1c525eda300e7333942346e05e74efd1f28af28ffe4a5c45d7e"
SERVICE_HASH = "sha256:bdcfb3a03ca93d36785b15c51b8ca94ae71bc449e818a8f12c0776b045387d3f"
TERMINAL_HASHES = {
    7977: "sha256:43993417cc30156e82295a9407593e2bc8a07b70e49d3d69e19defe1de9c1c8d",
    7989: "sha256:c9684d7d9b5bfbfe915b7829e1e2847a5706fe3d3481d17a508cca162cbd58ec",
}
Json = dict[str, Any]


def input_evidence(
    root: Path, label: str, pin: str, eid: int, checks: list[Json], sources: Json
) -> Json:
    """Authenticate producer identity and its closed receipt chain before import."""
    value = history.authenticate(root, label, pin, checks, sources)
    digest = sources[label]["sha256"]
    role = "historical_board_custody" if eid == 7977 else "measured_fitted_head_service"
    sources[label].update(role=role, producer_invocation_date=value.get("run_date"))
    receipts = value.get("validation_receipts", {})
    required_passed = (
        receipts.get("required_checks_passed")
        if isinstance(receipts, dict)
        else bool(receipts) and all(r.get("passed") for r in receipts if r.get("required"))
    )
    for field, expected in (
        ("experiment_id", eid),
        ("task_id", f"exp{eid}-" + ("hardware-evidence" if eid == 7977 else "service-cost")),
        ("milestone", "2026.10.691" if eid == 7977 else "2026.10.692"),
        ("run_date", "20261001"),
        ("flagged_adversarial", False),
        ("hardware_evidence_ready_score" if eid == 7977 else "service_measurement_ready_score", 1),
        ("validation_receipts.required_checks_passed", True),
    ):
        actual = required_passed if "." in field else value.get(field)
        checks.append(_check(label, digest, field, expected, actual))
    checks.append(
        _check(
            label,
            digest,
            "verdict_class.eligible",
            True,
            value.get("verdict_class") in {"null", "positive", "circular_positive"},
        )
    )
    if digest != pin:
        return {}
    terminal_label = value["terminal_validation_sidecar_path"]
    terminal = history.authenticate(root, terminal_label, TERMINAL_HASHES[eid], checks, sources)
    key = "candidate_sha256" if eid == 7977 else "primary_sha256"
    checks.append(
        _check(terminal_label, sources[terminal_label]["sha256"], key, digest, terminal.get(key))
    )
    if eid == 7989:
        terminal = history.authenticate(
            root,
            terminal.get("sidecar_path", "missing-validator.json"),
            "sha256:13e17f1d95816895eada84b755dcb621d74ada163e4ac7a6820abbd1f316a67b",
            checks,
            sources,
        ).get("report", {})
    checks.append(_check(terminal_label, digest, "terminal.passed", True, terminal.get("passed")))
    refs = value["source_artifact_hashes"]
    refs = list(refs.values()) if isinstance(refs, dict) else list(refs)
    refs += value.get("code_config_hashes", []) + value.get("raw_shard_hashes", [])
    if value.get("input_checkpoint"):
        refs.append(value["input_checkpoint"])
    refs += [{"path": p, "sha256": h} for p, h in value.get("terminal_receipt_hashes", {}).items()]
    refs += [
        {"path": r["log_path"], "sha256": r["log_sha256"]}
        for r in terminal.get("reports", terminal.get("receipts", []))
    ]
    for index, ref in enumerate(refs):
        if not ref.get("sha256"):
            continue
        path = history.bound_path(root, ref["path"])
        actual = sha256_file(path) if path.is_file() else None
        checks.append(_check(ref["path"], actual, "sha256", ref["sha256"], actual))
        sources[ref["path"]] = dict(ref, sha256=actual, expected_sha256=ref["sha256"])
        if index % 100 == 0:
            print(f"[exp7990] authenticated_source={eid} receipt={index}/{len(refs)}", flush=True)
    return value


def bounds(fraction: float, total_s: float, transfer_s: float | None) -> Json:
    """Transfer and all host work stay in the denominator of conditional estimates."""
    if (
        not all(math.isfinite(x) for x in (fraction, total_s))
        or not 0 <= fraction <= 1
        or total_s <= 0
        or transfer_s is not None
        and (not math.isfinite(transfer_s) or transfer_s < 0)
    ):
        raise ValueError("invalid_bound_operand")
    return dict(
        compatible_fraction=fraction,
        ideal_amdahl_bound=1 / (1 - fraction) if fraction < 1 else None,
        ideal_limit="unbounded" if fraction == 1 else "finite",
        modeled_100x_bound=1 / ((1 - fraction) + fraction / 100 + (transfer_s or 0) / total_s),
        transfer_time_s=transfer_s,
        total_time_s=total_s,
        comparison_class="optimistic_estimate" if transfer_s is None else "modeled_estimate",
        measured_speedup=None,
    )


def read_evidence(root: Path, run_date: str) -> Json:
    """SCENARIO-REPORT-7990-CUSTODY: a complete null still needs all required bytes."""
    started = time.monotonic()
    utc_start = datetime.now(UTC).isoformat()
    if run_date != "20261001":
        raise ValueError("run_date_mismatch")
    print("[exp7990] phase=custody before_source_authentication", flush=True)
    checks: list[Json] = []
    sources: Json = {}
    prior = input_evidence(root, PRIOR, PRIOR_HASH, 7977, checks, sources)
    custody_ready = all(r["passed"] for r in checks)
    result = deepcopy(prior) if prior else history.read_evidence(root, run_date)
    current = input_evidence(root, SERVICE, SERVICE_HASH, 7989, checks, sources)
    manifest = root / "ops/exclusion_manifest.yaml"
    exclusions = yaml.safe_load(manifest.read_text()) if manifest.is_file() else {}
    retired = {
        r.get("experiment_id")
        for key in ("retired", "retired_experiments")
        for r in (exclusions or {}).get(key, [])
    }
    for eid, label in ((7977, PRIOR), (7989, SERVICE)):
        checks.append(_check(label, sources[label]["sha256"], "retired", False, eid in retired))
    if current and all(r["passed"] for r in checks):
        try:
            replay_value = deepcopy(current)
            references = replay_value["source_artifact_hashes"] + replay_value["code_config_hashes"]
            references.append(replay_value["input_checkpoint"])
            for ref in references:
                ref["path"] = str(history.bound_path(root, ref["path"]))
            service.replay(replay_value)
            replay = "passed"
        except (OSError, ValueError, KeyError, TypeError) as error:
            replay = str(error)
        checks.append(
            _check(
                SERVICE, sources[SERVICE]["sha256"], "primitive_service_replay", "passed", replay
            )
        )
    ready = all(r["passed"] for r in checks)
    print(
        f"[exp7990] phase=mapping custody_valid={custody_ready} service_valid={ready}", flush=True
    )
    boards = deepcopy(result["board_rows"])
    for board in boards:
        board.update(
            custody_valid=custody_ready,
            evidence_age_days=(
                datetime.strptime(run_date, "%Y%m%d")
                - datetime.strptime(board["receipt_date"], "%Y%m%d")
            ).days
            if board["receipt_date"]
            else None,
            execution_class=board.get("processor_class"),
            compatible_workload=[],
            exact_blocker=board["blocker"] or "no_compatible_measured_fabric_kernel",
            required_external_change=board.get("next_operator_or_device_change"),
        )
    placements, estimates = [], []
    if ready:
        for summary in current["service_summary"]:
            selected = [
                r
                for r in current["rows"]
                if (r["arm"], r["storage"]) == (summary["arm"], summary["storage"])
            ]
            total = sum(r["wall_ns"] for r in selected) / 1e9
            estimates.append(
                dict(arm=summary["arm"], storage=summary["storage"], **bounds(0, total, None))
            )
            for phase in service.PHASES:
                elapsed = sum(r["exclusive_phase_spans"][phase] for r in selected) / 1e9
                for board in boards:
                    placements.append(
                        dict(
                            arm=summary["arm"],
                            storage=summary["storage"],
                            operation=phase,
                            board=board["board"],
                            source_path=board["source_path"],
                            source_hash=board["source_hash"],
                            authenticated_capability=board["scope"],
                            duration_s=elapsed,
                            total_time_s=total,
                            host_fraction=elapsed / total,
                            compatible_fraction=0,
                            compatible=False,
                            disposition="measured_host_cpu_no_fabric_match",
                            hardware_execution_measured=False,
                            service_hash=sources[SERVICE]["sha256"],
                        )
                    )
    for key in (
        "raw_rows_path",
        "raw_rows_sha256",
        "validation_evidence_files",
        "validation_command_manifest_sha256",
    ):
        result.pop(key, None)
    result.update(
        experiment_id=7990,
        task_id="exp7990-hardware-evidence",
        milestone="2026.10.692",
        run_date=run_date,
        execution_date=run_date,
        current_execution_date=run_date,
        started_at=utc_start,
        finished_at=datetime.now(UTC).isoformat(),
        honest_verdict="complete_null_no_compatible_measured_board_kernel"
        if ready
        else "complete_blocked_required_custody_or_service",
        verdict_class="null" if ready else "blocked",
        hardware_evidence_ready_score=int(ready),
        required_custody_valid=custody_ready,
        flagged_adversarial=False,
        board_rows=boards,
        rows=deepcopy(boards),
        rows_checksum=canonical_hash(boards),
        preconditions_checked=checks,
        gate_check_summary=[r for r in checks if not r["passed"]],
        source_artifact_hashes=sources,
        source_receipt_hashes={name: r["sha256"] for name, r in sources.items()},
        workload_attachment_available=ready,
        workload_attachment_operands=checks,
        workload_branch_readiness=current.get("branch_readiness", {}),
        workload_placement_rows=placements,
        workload_feasibility_rows=placements,
        operation_map=placements,
        operation_map_validity="authenticated_no_fabric_match" if ready else "blocked_mapping",
        measured_host_costs={
            k: deepcopy(current.get(k))
            for k in (
                "service_summary",
                "complete_service_cost",
                "acquisition_setup",
                "durable_costs",
                "sample_size_budget",
            )
        }
        if ready
        else {},
        compatible_fraction=0 if ready else None,
        ideal_amdahl_bound=1 if ready else None,
        modeled_100x_bound=1 if ready else None,
        modeled_acceleration_bounds=estimates,
        transfer_assumptions=dict(
            time_s=None,
            assumed_time_s=0,
            label="optimistic_estimate",
            reason="No offload transfer measured; zero compatible work requires no transfer",
        ),
        measured_offload_ceiling_available=False,
        measured_offload_ceiling=dict(
            comparison_class="optimistic_estimate", hardware_speedup=None
        ),
        spline_accounting=dict(
            coefficient_touches={
                r["storage"]: r["coefficient_touches"]
                for r in current.get("service_summary", [])
                if r["arm"] == "spline"
            }
            if ready
            else {},
            scope="Recorded fitted-head parameter touches; not a hardware memory transaction measurement",
            online_update_touches=None,
            feasibility_boundary="Future fixed-point basis LUT and coefficient storage require bounded quantization error, frozen-decision parity, sparse update byte counts, new fabric implementation, transfer, final readout and persistence timing; existing KV260 quadratic Ising overlay cannot execute this head",
        ),
        trained_head_specs=deepcopy(current.get("trained_head_specs", [])) if ready else [],
        current_device_execution_count=0,
        scheduled_device_operations=[],
        hardware_speedup_claimed=False,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        methodology="Read-only authentication and reduction of measured service rows. Include public processing, normalization, scoring, typed final readout, serialization and file/directory fsync. Historical acquisition remains separate. Unknown durable learning and vendor headline exclusions cannot define complete Carnot service cost.",
        code_config_hashes=[
            dict(path=str(path), sha256=sha256_file(path))
            for path in (
                Path(__file__),
                ROOT / "python/carnot/reporting/validation_7990.py",
                ROOT / "scripts/experiments/experiment_7990_v692_hardware_evidence.py",
            )
        ],
        raw_shard_hashes=deepcopy(current.get("raw_shard_hashes", [])),
        random_seed=0,
        validation_receipts=dict(checks=[], required_checks_passed=False),
        validation_command_manifest_path=None,
        coverage_statement_counts={},
        observed_child_commands=[],
        terminal_validation_sidecar_path=None,
        primary_resolution_receipt=dict(path=None),
        wishlist_decision=dict(
            priorities_changed=False,
            purchase_scheduled=False,
            reason="No authenticated compatible bottleneck justifies acquisition",
        ),
        cited_upstream_artifacts=[
            dict(experiment_id=eid, **sources[label], fields_imported=fields)
            for eid, label, fields in (
                (7977, PRIOR, ["board_rows", "historical_required_failures"]),
                (
                    7989,
                    SERVICE,
                    [
                        "service_rows",
                        "service_summary",
                        "trained_head_specs",
                        "complete_service_cost",
                    ],
                ),
            )
        ],
    )
    result["acceptance_gate_results"].update(
        validity=ready,
        readiness=int(ready),
        efficiency="conditional_estimate_only",
        decision_benefit="not_measured",
    )
    result["claim_scope"].update(
        workload="exposed_development" if ready else "blocked_mapping",
        acceleration="no measured device execution",
    )
    result["reproducibility_checksum"] = canonical_hash(
        dict(
            sources=sources,
            boards=boards,
            mapping=placements,
            code=result["code_config_hashes"],
            date=run_date,
        )
    )
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"] = {"evidence_read_s": result["duration_s"]}
    result["field_principles"] = {
        k: "Bind evidence to exact source bytes and current identity; preserve unknown cost and distinguish estimates from execution."
        for k in result
    }
    print("[exp7990] phase=mapping after_source_reduction", flush=True)
    return result


def cold_reduce(root: Path, candidate: Json) -> Json:
    """SCENARIO-VERIFY-7990-REPLAY: reconstruct claims rather than trust summaries."""
    fresh = read_evidence(root, candidate["run_date"])
    variable = {
        "duration_s",
        "phase_spans",
        "started_at",
        "finished_at",
        "field_principles",
        "reproducibility_checksum",
        "validation_receipts",
        "validation_command_manifest_path",
        "observed_child_commands",
        "coverage_statement_counts",
        "repository_health",
        "primary_resolution_receipt",
        "terminal_validation_sidecar_path",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "hardware_evidence_ready_score",
        "acceptance_gate_results",
        "gate_check_summary",
        "scratch_root_receipt",
    }
    for field in fresh.keys() - variable:
        if candidate.get(field) != fresh[field]:
            raise ValueError(f"claims_changed:{field}")
    gates = candidate.get("gate_check_summary")
    if not isinstance(gates, list) or fresh["gate_check_summary"] != [
        r for r in gates if r.get("upstream_id") != "owned_validation"
    ]:
        raise ValueError("claims_changed:gate_operands")
    disqualified = candidate.get("verdict_class") == "disqualified"
    if candidate.get("hardware_evidence_ready_score") != (
        0 if disqualified else fresh["hardware_evidence_ready_score"]
    ):
        raise ValueError("claims_changed:readiness")
    if disqualified and not any(
        r.get("classification") == "required" and not r.get("passed")
        for r in candidate.get("validation_receipts", {}).get("checks", [])
    ):
        raise ValueError("claims_changed:unsubstantiated_disqualification")
    return dict(rows_checksum=fresh["rows_checksum"], row_count=len(fresh["rows"]))
