"""REQ-REPORT-7977: host costs cannot enlarge historical hardware scope.

Frozen producer bytes preserve custody. Current service timings support
conditional placement estimates, without device execution or purchase claims.
"""

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

from coverage import CoverageData
import yaml

from carnot.reporting import experiment_7964_v690_hardware_evidence as history
from carnot.reporting import service_cost_7976 as service
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.experiment_7862_v682_hardware_evidence import _check
from carnot.reporting.experiment_7876_v683_hardware_evidence import _load

ROOT = history.ROOT
PRIOR = "results/experiment_7964_v690_hardware_evidence.json"
SERVICE = "results/experiment_7976_v691_service_cost.json"
PRIOR_HASH = "sha256:6a8a6fd5f4455aa050ad851d1219f3ab896987b085a8865ad645d3fb1a62d68d"
TERMINAL_HASH = "sha256:29e687810bb4398e3f57a2ff5460eff0aa234e9ddeb0e126bc6a85f3eb11bb5d"
SERVICE_HASH = "sha256:48444e30e0aeeb796ff7a5a1cad04bcc3f6c9eaf8d8c61505aa3ef6efee44190"
SERVICE_TERMINAL_HASH = "sha256:675ea52cc345b1f1edba75833a2525ed4b933325757b052b29e7a84cae5a39db"
BOARDS = history.BOARDS


def bound_path(root: Path, label: str) -> Path:
    """Private copies keep historical absolute labels without editing originals."""
    path = Path(label)
    return root / path.relative_to(ROOT) if path.is_relative_to(ROOT) else root / path


def authenticate(
    root: Path, label: str, pin: str, checks: list[dict[str, Any]], sources: dict[str, Any]
) -> dict[str, Any]:
    """Compare exact bytes before trusting any fields imported from them."""
    value, digest = _load(root, str(bound_path(root, label)))
    checks.append(_check(label, digest, "sha256", pin, digest))
    sources[label] = {"path": label, "sha256": digest, "expected_sha256": pin}
    return value or {}


def custody_checks(
    root: Path, prior: dict[str, Any], checks: list[dict[str, Any]], sources: dict[str, Any]
) -> None:
    """Each historical receipt retains its own date and execution scope."""
    digest = sources[PRIOR]["sha256"]
    for field, expected in (
        ("experiment_id", 7964),
        ("task_id", "exp7964-hardware-evidence"),
        ("milestone", "2026.09.690"),
        ("run_date", "20261001"),
        ("hardware_evidence_ready_score", 1),
        ("flagged_adversarial", False),
        ("verdict_class", "null"),
    ):
        checks.append(_check(PRIOR, digest, field, expected, prior.get(field)))
    terminal_label = str(prior.get("terminal_validation_sidecar_path", "missing-terminal.json"))
    terminal = authenticate(root, terminal_label, TERMINAL_HASH, checks, sources)
    checks.append(
        _check(
            terminal_label,
            sources[terminal_label]["sha256"],
            "candidate_sha256",
            digest,
            terminal.get("candidate_sha256"),
        )
    )
    references = [*prior.get("source_artifact_hashes", {}).values()]
    references.extend(
        {"path": r["log_path"], "sha256": r["log_sha256"]} for r in terminal.get("reports", [])
    )
    references.extend(
        {"path": name, "sha256": pin}
        for name, pin in prior.get("terminal_receipt_hashes", {}).items()
    )
    references.extend(
        {"path": r["source_path"], "sha256": r["source_hash"]} for r in prior.get("board_rows", [])
    )
    for ref in references:
        if ref.get("sha256"):
            path = bound_path(root, ref["path"])
            actual = sha256_file(path) if path.is_file() else None
            checks.append(_check(ref["path"], actual, "sha256", ref["sha256"], actual))
            sources[ref["path"]] = {**ref, "sha256": actual}


def service_checks(
    root: Path, sources: dict[str, Any], retired: set[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A shortened stub cannot replace the declared measured service primary."""
    checks: list[dict[str, Any]] = []
    source = authenticate(root, SERVICE, SERVICE_HASH, checks, sources)
    digest = sources[SERVICE]["sha256"]
    for field, expected, observed in (
        ("experiment_id", 7976, source.get("experiment_id")),
        ("task_id", "exp7976-service-cost", source.get("task_id")),
        ("milestone", "2026.10.691", source.get("milestone")),
        ("run_date", "20261001", source.get("run_date")),
        ("service_measurement_ready_score", 1, source.get("service_measurement_ready_score")),
        ("flagged_adversarial", False, source.get("flagged_adversarial")),
        (
            "verdict_class.eligible",
            True,
            source.get("verdict_class") in {"positive", "circular_positive", "null"},
        ),
        ("retired", False, 7976 in retired),
    ):
        checks.append(_check(SERVICE, digest, field, expected, observed))
    if not all(r["passed"] for r in checks):
        return source, checks
    label = source["terminal_validation_sidecar_path"]
    terminal = authenticate(root, label, SERVICE_TERMINAL_HASH, checks, sources)
    checks.append(
        _check(
            label,
            sources[label]["sha256"],
            "primary_sha256",
            digest,
            terminal.get("primary_sha256"),
        )
    )
    report = authenticate(
        root,
        terminal.get("sidecar_path", "missing-validator.json"),
        "sha256:2d169d6e89b081aecd0078571c1cb082e2f1859d2fda4094a6076111be905f96",
        checks,
        sources,
    )
    checks.append(
        _check(label, digest, "report.passed", True, report.get("report", {}).get("passed"))
    )
    for ref in [
        *source["source_artifact_hashes"],
        *source["code_config_hashes"],
        source["input_checkpoint"],
        *[
            {"path": r["log_path"], "sha256": r["log_sha256"]}
            for r in report.get("report", {}).get("receipts", [])
        ],
    ]:
        path = bound_path(root, ref["path"])
        actual = sha256_file(path) if path.is_file() else None
        checks.append(_check(ref["path"], actual, "sha256", ref["sha256"], actual))
        sources[ref["path"]] = {**ref, "sha256": actual}
    try:
        service.replay(source)
        replay = "passed"
    except (OSError, ValueError, KeyError, TypeError) as error:
        replay = str(error)
    checks.append(_check(SERVICE, digest, "primitive_span_checkpoint_replay", "passed", replay))
    return source, checks


def workload_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Timing modes remain separate while repeats remain dependent observations."""
    rows = []
    for branch, ready in source["branch_readiness"].items():
        if not ready or branch == "durable_learning":
            continue
        for summary in source["rows"]:
            selected = [r for r in source["service_rows"] if r["mode"] == summary["mode"]]
            total = sum(r["wall_ns"] for r in selected) / 1e9
            for phase in service.PHASES:
                duration = sum(r["exclusive_phase_spans"][phase] for r in selected) / 1e9
                rows.append(
                    dict(
                        branch=branch,
                        mode=summary["mode"],
                        operation=phase,
                        duration_s=duration,
                        start_s=0.0,
                        end_s=duration,
                        complete_cached_cpu_s=total,
                        kernel_share=0.0,
                        host_fraction=duration / total,
                        transfer_bytes=source["transfer_bytes"],
                        state_bytes=None,
                        representation="nonlinear_gibbs"
                        if phase == "head_evaluation"
                        else "host_service",
                        topology="small_dense_head" if phase == "head_evaluation" else None,
                    )
                )
    return rows


def placements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CPU timing suggests portability, without establishing another substrate."""
    return [
        dict(
            **row,
            target=target,
            compatible_operations=[row["operation"]]
            if target in {"CPU", "Rust"}
            or target == "GPU"
            and row["operation"]
            in {"feature_projection", "head_evaluation", "energy_normalization"}
            else [],
            disposition="measured_host_cpu"
            if target == "CPU"
            else "candidate_port_no_measurement"
            if target in {"Rust", "GPU"}
            else "unqualified_nonlinear_or_host_operation",
            hardware_execution_measured=False,
            speedup=None,
            coefficient_memory_traffic="Sparse-spline reference needs coefficient reads and writes; this overlay has no spline training implementation.",
            vendor_tsu_projection="Vendor projection only; no local TSU measurement or access qualification.",
        )
        for row in rows
        for target in ("CPU", "Rust", "GPU", "LUT/FPGA", "TSU")
    ]


def read_evidence(root: Path, run_date: str) -> dict[str, Any]:
    """SCENARIO-REPORT-7977-CUSTODY: optional workload cannot supply board scope."""
    started = time.monotonic()
    utc_start = datetime.now(UTC).isoformat()
    if run_date != "20261001":
        raise ValueError("run_date_mismatch")
    checks: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    prior = authenticate(root, PRIOR, PRIOR_HASH, checks, sources)
    result = (
        deepcopy(prior)
        if sources[PRIOR]["sha256"] == PRIOR_HASH
        else history.read_evidence(root, run_date)
    )
    custody_checks(root, prior if sources[PRIOR]["sha256"] == PRIOR_HASH else {}, checks, sources)
    manifest = root / "ops/exclusion_manifest.yaml"
    exclusions = yaml.safe_load(manifest.read_text()) if manifest.is_file() else {}
    retired = {
        r.get("experiment_id")
        for key in ("retired", "retired_experiments")
        for r in (exclusions or {}).get(key, [])
    }
    checks.append(_check(PRIOR, sources[PRIOR]["sha256"], "retired", False, 7964 in retired))
    ready = all(r["passed"] for r in checks)
    boards = deepcopy(result["board_rows"])
    for board in boards:
        if not ready:
            board.update(
                custody_valid=False,
                status="blocked_custody",
                scope=None,
                eligible=0,
                excluded=True,
                blocker="required_custody_invalid",
            )
    source, operands = service_checks(root, sources, retired)
    attached = all(r["passed"] for r in operands)
    measured = workload_rows(source) if attached else []
    mapped = history.operation_map(
        boards,
        measured if attached else [{"operation": op} for op in service.PHASES],
        sources[SERVICE]["sha256"] if attached else None,
    )
    for field in (
        "raw_rows_path",
        "raw_rows_sha256",
        "validation_evidence_files",
        "validation_command_manifest_sha256",
    ):
        result.pop(field, None)
    result.update(
        experiment_id=7977,
        task_id="exp7977-hardware-evidence",
        milestone="2026.10.691",
        run_date=run_date,
        execution_date=run_date,
        current_execution_date=run_date,
        started_at=utc_start,
        finished_at=datetime.now(UTC).isoformat(),
        duration_scope="Monotonic work until candidate freeze; final validator spans remain in the hash-bound sidecar.",
        honest_verdict="complete_null_historical_board_scope_current_host_mapping"
        if ready
        else "complete_blocked_board_source_custody",
        verdict_class="null" if ready else "blocked",
        flagged_adversarial=False,
        hardware_evidence_ready_score=int(ready),
        required_custody_valid=ready,
        board_rows=boards,
        rows=deepcopy(boards),
        rows_checksum=canonical_hash(boards),
        typed_blockers=[
            dict(
                board=r["board"],
                type="required_custody_blocked"
                if not ready
                else "physical_jtag_blocked"
                if r["board"] == "GateMate"
                else "current_device_execution_unmeasured",
                blocker=r["blocker"],
                placement=None,
            )
            for r in boards
        ],
        typed_blocker_rows=[]
        if ready
        else [
            dict(board=r["board"], type="required_custody_blocked", placement=None) for r in boards
        ],
        preconditions_checked=checks,
        gate_check_summary=[r for r in checks if not r["passed"]],
        source_artifact_hashes=sources,
        source_receipt_hashes={name: r["sha256"] for name, r in sources.items()},
        workload_attachment_available=attached,
        workload_attachment_operands=operands,
        workload_branch_readiness=deepcopy(source.get("branch_readiness", {})) if attached else {},
        workload_placement_rows=placements(measured) if ready else [],
        workload_feasibility_rows=mapped if attached else [],
        operation_map=mapped,
        operation_map_validity="qualified_historical_custody"
        if ready
        else "blocked_unknown_placement",
        measured_offload_ceiling_available=attached and ready,
        measured_offload_ceiling=dict(
            status="estimate_from_measured_host_shares" if attached and ready else "unmeasured",
            comparison_class="estimate",
            hardware_speedup=None,
        ),
        measured_host_costs={
            key: deepcopy(source[key])
            for key in (
                "rows",
                "transfer_bytes",
                "durable_costs",
                "complete_service_cost",
                "cached_incremental_cost",
                "exclusive_phase_spans",
            )
        }
        if attached
        else {},
        modeled_acceleration_bounds={
            key: deepcopy(source[key])
            for key in (
                "compatible_fraction",
                "transfer_fraction",
                "ideal_amdahl_bound",
                "modeled_100x_bound",
            )
        }
        if attached
        else {},
        wishlist_decision=dict(
            priorities_changed=False,
            purchase_scheduled=False,
            reason="Current CPU timings do not establish complete-service device benefit.",
        ),
        current_device_execution_count=0,
        scheduled_device_operations=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        target_model=None,
        trained_head_specs=[],
        validation_receipts={"checks": [], "required_checks_passed": False},
        validation_command_manifest_path=None,
        observed_child_commands=[],
        coverage_statement_counts={},
        primary_resolution_receipt={"path": None, "binding": "exact final bytes in sidecar"},
        terminal_validation_sidecar_path=None,
        scratch_root_receipt={"path": None, "scope": "read-only aggregation; no scratch allocated"},
        retire_if_same_verdict=dict(
            prior_experiment_id=7964,
            identical_failure=True,
            reason="Unchanged board verdicts retire; current host mapping adds no physical evidence.",
        ),
    )
    result["claim_scope"].update(workload="exposed_development" if attached else "unmeasured")
    result["acceptance_gate_results"].update(
        validity=ready,
        readiness=int(ready),
        calibration="not_measured",
        decision_benefit="not_measured",
        retention="historical_board_obligations_preserved",
        efficiency="conditional_host_estimate" if attached else "unknown",
    )
    result["sample_size_budget"].update(eligible=2 if ready else 0, excluded=1 if ready else 3)
    result["cited_upstream_artifacts"] = [
        dict(experiment_id=eid, path=name, sha256=sources[name]["sha256"], fields_imported=fields)
        for eid, name, fields in (
            (7964, PRIOR, ["board_rows", "historical_required_failures", "repository_health"]),
            (7976, SERVICE, ["service_rows", "rows", "branch_readiness", "complete_service_cost"]),
        )
    ]
    result["reproducibility_checksum"] = canonical_hash(
        dict(
            sources=sources,
            rows=boards,
            mapping=mapped,
            code=sha256_file(Path(__file__)),
            date=run_date,
            seed=0,
        )
    )
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"] = {"evidence_read_s": result["duration_s"]}
    result["field_principles"].update(
        {
            name: f"Bind {name.replace('_', ' ')} to authenticated evidence within its declared scope."
            for name in result
            if name not in result["field_principles"]
        }
    )
    result["field_principles"].update(
        workload_placement_rows="Each measured branch and mode retains operation costs; other substrate placements are candidates only.",
        measured_host_costs="Preserve exact CPU spans, state byte traffic, transfer bytes and upstream acquisition costs; unknown durable costs stay unknown.",
        modeled_acceleration_bounds="Zero compatible quadratic work yields unit speedup bounds; kernel and ideal bounds are conditional estimates.",
        source_receipt_hashes="Each historical and service receipt is authenticated by exact byte hash before import.",
        typed_blockers="Keep distinct board custody, physical JTAG and unmeasured execution obligations without retrying unchanged devices.",
        wishlist_decision="CPU timings cannot establish measured full-service accelerator benefit or justify a purchase.",
        workload_branch_readiness="One qualified branch cannot supply missing peer or learning evidence.",
        started_at="Current producer UTC start is separate from historical invocation identity.",
        finished_at="UTC candidate freeze is separate from the producer's honest monotonic elapsed work.",
        execution_date="Current execution records 20261001; upstream tasks retain their frozen original dates.",
        scratch_root_receipt="Mutable validation fixtures stay in a unique external TemporaryDirectory; closed evidence is archived below raw.",
    )
    return result


def cold_reduce(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-REPORT-7977-VALIDATION: recompute claims from primitive evidence."""
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
        "resolved_imports",
        "repository_health",
        "primary_resolution_receipt",
        "terminal_validation_sidecar_path",
        "scratch_root_receipt",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "hardware_evidence_ready_score",
        "acceptance_gate_results",
        "gate_check_summary",
    }
    for field in fresh.keys() - variable:
        if fresh[field] != candidate.get(field):
            raise ValueError(f"claims_changed:{field}")
    gates = candidate.get("gate_check_summary")
    if (
        not isinstance(gates, list)
        or any(not isinstance(r, dict) for r in gates)
        or fresh["gate_check_summary"]
        != [r for r in gates if r.get("upstream_id") != "owned_validation"]
    ):
        raise ValueError("claims_changed:gate_operands")
    disqualified = candidate.get("verdict_class") == "disqualified"
    expected = 0 if disqualified else fresh["hardware_evidence_ready_score"]
    if candidate.get("hardware_evidence_ready_score") != expected:
        raise ValueError("claims_changed:readiness")
    if disqualified and not any(
        not r.get("passed")
        for r in candidate.get("validation_receipts", {}).get("checks", [])
        if r.get("classification") == "required"
    ):
        raise ValueError("claims_changed:unsubstantiated_disqualification")
    return {"rows_checksum": fresh["rows_checksum"], "row_count": len(fresh["rows"])}


def check_coverage_shards(paths: list[Path]) -> None:
    """Empty or unrelated shards cannot support owned statement coverage."""
    from carnot.reporting.validation_7977 import MEASURED

    if not paths:
        raise ValueError("empty_coverage_shards")
    owned = {str((ROOT / name).resolve()) for name in MEASURED}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"empty_coverage_shard:{path}")
        data = CoverageData(basename=str(path))
        data.read()
        if not any(data.lines(name) for name in data.measured_files() if name in owned):
            raise ValueError(f"empty_coverage_shard:{path}")


def qualify(root: Path, run_date: str, output: Path, raw_root: Path) -> int:
    """Reuse bounded validation before publishing the one checked primary."""
    from carnot.reporting.publication_7977 import qualify as publish

    return publish(root, run_date, output, raw_root)
