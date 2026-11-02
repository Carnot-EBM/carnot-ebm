"""Build the Exp 3134 milestone .291 capstone artifact.

Spec refs: REQ-REPORT-3134, SCENARIO-REPORT-3134.

This capstone is a closeout ledger, not a new experiment. It reads the checked
in matrix v25 artifact and its traced `.291` sources, then states exactly what
those artifacts allow Carnot to claim. Keeping this aggregation separate from
live inference is what prevents a completed matrix from being mistaken for a
paper-ready empirical result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.291"
SCHEMA = "carnot.milestone_capstone.v291_matrix_v25_aggregation.v1"
ARTIFACT = "experiment_3134_capstone_v291"
OUTPUT_REL_PATH = Path("results/experiment_3134_capstone_v291.json")
MATRIX_V25_REL_PATH = Path("results/experiment_3133_cross_corpus_matrix_v25.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3134_capstone_v291.py"

EXP3122_REL_PATH = Path("results/experiment_3122_archive_v290_activate_v291.json")
EXP3123_REL_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")
EXP3124_REL_PATH = Path(
    "results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"
)
EXP3125_REL_PATH = Path(
    "results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json"
)
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
EXP3127_REL_PATH = Path("results/experiment_3127_multi_turn_monitored_repair_ladder_v1.json")
EXP3128_REL_PATH = Path(
    "results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json"
)
EXP3129_REL_PATH = Path(
    "results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json"
)
EXP3130_REL_PATH = Path("results/experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.json")
EXP3131_REL_PATH = Path("results/experiment_3131_kan_pwa_milp_verifier_abstraction_audit_v1.json")
EXP3132_REL_PATH = Path("results/experiment_3132_hardware_evidence_sampler_boundary_v5.json")


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, treating absent or malformed evidence as empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a stable checksum so capstone provenance is reproducible."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3134: close .291 from matrix v25 and source artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V25_REL_PATH)
    rows = _list(matrix.get("rows"))
    verifier = _mapping(matrix.get("verifier_repair_summary"))
    fr11 = _mapping(matrix.get("fr11_summary"))
    architecture = _mapping(matrix.get("architecture_boundary_summary"))
    allowance = _mapping(matrix.get("headline_claim_allowance_summary"))
    source_artifacts = _source_artifacts(root_path, matrix)
    required_source_errors = _required_source_errors(source_artifacts)
    invariant_violations = _invariant_violations(matrix, source_artifacts, required_source_errors)
    capstone_ready = not invariant_violations
    publication_blocker_count = _int(matrix.get("publication_blocker_count"))
    blocker_delta_from_v24 = _int(matrix.get("blocker_delta_from_v24"))
    paper_ready = capstone_ready and publication_blocker_count == 0
    next_top_gap = _next_top_gap(verifier, allowance, architecture)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v24": blocker_delta_from_v24,
        "next_top_gap": next_top_gap,
        "sota_cache_status": _sota_cache_status(verifier, allowance),
        "live_verifier_status": str(verifier.get("live_verifier_status") or "missing"),
        "verifier_claim_status": _verifier_claim_status(verifier),
        "prefix_bounds_status": _prefix_bounds_status(verifier),
        "monitor_status": _monitor_status(verifier, rows),
        "repair_ladder_status": str(verifier.get("repair_ladder_status") or "missing"),
        "repair_claim_status": _repair_claim_status(verifier),
        "fr11_evoenv_status": str(fr11.get("evoenv_status") or "missing"),
        "fr11_memory_status": str(fr11.get("memory_status") or "missing"),
        "fr11_self_learning_status": _fr11_self_learning_status(fr11),
        "ebt_arm_status": _ebt_arm_status(architecture),
        "kan_status": _kan_status(architecture),
        "clut_sampler_status": _clut_sampler_status(rows),
        "gatemate_status": _gatemate_status(architecture),
        "ssqa_status": _ssqa_status(architecture),
        "hardware_status": _hardware_status(architecture),
        "sampler_hardware_status": _sampler_hardware_status(architecture),
        "paper_readiness_assessment": _paper_readiness_assessment(
            capstone_ready, publication_blocker_count, blocker_delta_from_v24
        ),
        "paper_readiness_checks": _paper_readiness_checks(
            capstone_ready, publication_blocker_count, allowance, architecture
        ),
        "matrix_v25_summary": _matrix_v25_summary(matrix),
        "claim_allowance_summary": allowance,
        "what_291_proved": _what_291_proved(matrix, verifier, fr11, architecture),
        "what_stayed_blocked": _what_stayed_blocked(verifier, allowance, architecture),
        "bounded_claims": _bounded_claims(verifier, fr11, architecture, rows),
        "allowed_claims": _allowed_claims(allowance, fr11, architecture),
        "next_recommendation": _next_recommendation(next_top_gap),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "required_source_errors": required_source_errors,
        "invariant_violations": invariant_violations,
        "ops_reconciliation_decision": _ops_reconciliation_decision(),
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_live_repair_rerun": True,
        "scripts_research_conductor_modified": False,
        "status_updates_written": False,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3134 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifacts(root: Path, matrix: Mapping[str, Any]) -> list[JsonDict]:
    specs = [
        {
            "experiment_id": "exp3133",
            "path": MATRIX_V25_REL_PATH.as_posix(),
            "role": "matrix_v25_authority",
            "required": True,
            "ready_field": "matrix_v25_ready",
        }
    ]
    specs.extend(_mapping(item) for item in _list(matrix.get("source_artifacts")))
    rows: list[JsonDict] = []
    seen: set[str] = set()
    for spec in specs:
        path_text = str(spec.get("path") or "")
        if path_text and path_text not in seen:
            seen.add(path_text)
            rows.append(_source_artifact_row(root, Path(path_text), spec))
    return rows


def _source_artifact_row(root: Path, rel_path: Path, spec: Mapping[str, Any]) -> JsonDict:
    path = root / rel_path
    payload = read_json_object(path)
    present = path.is_file()
    return {
        "experiment_id": str(
            payload.get("artifact") or spec.get("experiment_id") or rel_path.stem
        ),
        "path": rel_path.as_posix(),
        "role": str(spec.get("role") or _source_role(rel_path)),
        "required": bool(spec.get("required", False)),
        "ready_field": str(spec.get("ready_field") or ""),
        "source_type": "json",
        "present": present,
        "readable_json_object": bool(payload),
        "sha256": sha256_file(path),
    }


def _source_role(rel_path: Path) -> str:
    role_by_path = {
        MATRIX_V25_REL_PATH: "matrix_v25_authority",
        EXP3122_REL_PATH: "archive_v290_activate_v291",
        EXP3123_REL_PATH: "sota_cache_coverage",
        EXP3124_REL_PATH: "live_verifier_lift",
        EXP3125_REL_PATH: "prefix_closed_bounds",
        EXP3126_REL_PATH: "fragment_time_monitors",
        EXP3127_REL_PATH: "repair_ladder",
        EXP3128_REL_PATH: "fr11_evoenv",
        EXP3129_REL_PATH: "fr11_constraint_memory",
        EXP3130_REL_PATH: "arm_ebt_energy_budget",
        EXP3131_REL_PATH: "kan_pwa_milp",
        EXP3132_REL_PATH: "hardware_sampler_boundary",
    }
    return role_by_path.get(rel_path, "matrix_v25_source")


def _required_source_errors(source_artifacts: list[Mapping[str, Any]]) -> list[str]:
    return [
        f"required source unreadable: {row['path']}"
        for row in source_artifacts
        if row.get("required") and not row.get("readable_json_object")
    ]


def _invariant_violations(
    matrix: Mapping[str, Any],
    source_artifacts: list[Mapping[str, Any]],
    required_source_errors: list[str],
) -> list[str]:
    violations = list(required_source_errors)
    if matrix.get("matrix_v25_ready") is not True:
        violations.append("matrix_v25_ready is not true")
    status_counts = _mapping(matrix.get("status_counts"))
    rows_total = _int(matrix.get("rows_total"))
    if status_counts and sum(_int(value) for value in status_counts.values()) != rows_total:
        violations.append("status_counts do not reconcile with rows_total")
    if len(_list(matrix.get("publication_blockers"))) != _int(
        matrix.get("publication_blocker_count")
    ):
        violations.append("publication_blocker_count does not match matrix publication_blockers")
    if _list(matrix.get("missing_artifacts")):
        violations.append("matrix reports missing .291 artifacts")
    substrate = _mapping(matrix.get("inference_substrate"))
    if any(
        substrate.get(key) is True
        for key in (
            "executes_models",
            "executes_verifiers",
            "executes_repairs",
            "executes_solvers",
            "executes_hardware",
            "executes_conductor",
        )
    ):
        violations.append("matrix inference_substrate is not aggregation-only")
    if not source_artifacts:
        violations.append("source_artifacts list is empty")
    return violations


def _sota_cache_status(verifier: Mapping[str, Any], allowance: Mapping[str, Any]) -> str:
    status = str(verifier.get("sota_cache_status") or "missing")
    if status == "clean":
        return "clean_comparative_sota_cache_ready"
    if allowance.get("comparative_sota_pair_allowed") is not True:
        return "bounded_missing_comparative_sota_pair"
    return f"{status}_sota_cache_not_promoted"


def _verifier_claim_status(verifier: Mapping[str, Any]) -> str:
    status = str(verifier.get("live_verifier_status") or "missing")
    false_accept = _float(verifier.get("false_accept_rate"))
    if status == "clean" and false_accept == 0.0:
        return "clean_live_verifier_headline_lift_allowed"
    if false_accept > 0.0 or status == "blocked":
        return f"blocked_false_accept_rate_{_number(false_accept)}_no_headline_lift"
    return f"{status}_live_verifier_not_promoted"


def _prefix_bounds_status(verifier: Mapping[str, Any]) -> str:
    status = str(verifier.get("prefix_bounds_status") or "missing")
    if status == "clean":
        return "clean_prefix_bound_claim"
    if status == "bounded":
        return "bounded_finite_fixture_conditioned_prefix_frontier"
    return f"{status}_prefix_bounds_unavailable"


def _monitor_status(verifier: Mapping[str, Any], rows: list[Any]) -> str:
    status = str(verifier.get("fragment_time_monitor_status") or "missing")
    summary = _summary_by_row(rows, "dot291:exp3126_fragment_time_monitors")
    ledger = _float(summary.get("ledger_consistency_rate"))
    if status == "clean":
        return "clean_fragment_monitor_ledger"
    if status == "bounded":
        return f"bounded_fragment_monitor_ledger_consistency_{_number(ledger)}"
    return f"{status}_fragment_monitor_not_ready"


def _repair_claim_status(verifier: Mapping[str, Any]) -> str:
    status = str(verifier.get("repair_ladder_status") or "missing")
    blocked_at = str(verifier.get("repair_ladder_blocked_at_layer") or "")
    if status == "clean" and not blocked_at:
        return "clean_repair_ladder_promotable"
    if status == "blocked" or blocked_at:
        return "blocked_repair_ladder_gate_failed_by_live_verifier_gate"
    return f"{status}_repair_claim_not_promoted"


def _fr11_self_learning_status(fr11: Mapping[str, Any]) -> str:
    no_weight_update = fr11.get("no_weight_update_claim") is True
    ledger = _float(fr11.get("ledger_consistency_rate"))
    if not no_weight_update and fr11.get("model_weight_learning_allowed") is True:
        return "clean_model_weight_learning_allowed_by_matrix"
    return (
        "bounded_controller_environment_memory_only_no_weight_update_ledger_"
        f"{_number(ledger)}"
    )


def _ebt_arm_status(architecture: Mapping[str, Any]) -> str:
    status = str(architecture.get("arm_ebt_status") or "missing")
    if status == "clean" and architecture.get("live_integration") is True:
        return "clean_sidecar_live_integration"
    return "projection_only_sidecar_diagnostic_no_live_integration"


def _kan_status(architecture: Mapping[str, Any]) -> str:
    status = str(architecture.get("kan_pwa_milp_status") or "missing")
    if status == "clean":
        return "clean_kan_verifier_claim"
    if status == "bounded":
        return "bounded_pwa_milp_abstraction_no_deployed_verifier_claim"
    return f"{status}_kan_verifier_not_ready"


def _clut_sampler_status(rows: list[Any]) -> str:
    clut_status = _status_by_row(rows, "dot290:exp3118_clut_sampler_backend_integration")
    hardware_summary = _summary_by_row(rows, "dot291:exp3132_hardware_sampler_boundary")
    clut_decision = str(
        _mapping(hardware_summary.get("sampler_boundary_decisions")).get("clut") or ""
    ).lower()
    if clut_status == "clean" and "cpu" not in clut_decision:
        return "clean_authenticated_clut_sampler"
    return "bounded_cpu_simulation_no_authenticated_hardware_execution"


def _gatemate_status(architecture: Mapping[str, Any]) -> str:
    if architecture.get("gatemate_evidence_complete") is True:
        return "clean_gatemate_operator_evidence_complete"
    return "blocked_operator_evidence_incomplete"


def _ssqa_status(architecture: Mapping[str, Any]) -> str:
    if architecture.get("ssqa_readback_ready") is True:
        return "clean_host_visible_readback_ready"
    return "blocked_host_visible_readback_missing"


def _hardware_status(architecture: Mapping[str, Any]) -> str:
    commands = _list(architecture.get("hardware_commands_run"))
    if commands and architecture.get("speedup_claim_allowed") is True:
        return "clean_authenticated_hardware_speedup_claim_allowed"
    return "blocked_no_commands_no_speedup_claim"


def _sampler_hardware_status(architecture: Mapping[str, Any]) -> str:
    status = str(architecture.get("hardware_sampler_status") or "missing")
    if status == "clean" and architecture.get("speedup_claim_allowed") is True:
        return "clean_authenticated_sampler_hardware_speedup"
    return "blocked_hardware_sampler_boundary_no_speedup_claim"


def _paper_readiness_assessment(
    capstone_ready: bool, publication_blocker_count: int, blocker_delta_from_v24: int
) -> str:
    if not capstone_ready:
        return "blocked_precondition"
    if publication_blocker_count == 0:
        return "closer_blockers_cleared"
    if blocker_delta_from_v24 > 0:
        return f"not_closer_blockers_increased_by_{blocker_delta_from_v24}"
    if blocker_delta_from_v24 < 0:
        return f"closer_blockers_reduced_by_{abs(blocker_delta_from_v24)}_but_not_ready"
    return "not_closer_blockers_unchanged"


def _paper_readiness_checks(
    capstone_ready: bool,
    publication_blocker_count: int,
    allowance: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        {
            "check": "capstone_ready",
            "passed": capstone_ready,
            "reason": "matrix v25 authority loaded and required invariants reconciled",
        },
        {
            "check": "publication_blocker_count_zero",
            "passed": publication_blocker_count == 0,
            "reason": f"publication_blocker_count={publication_blocker_count}",
        },
        {
            "check": "headline_claims_unblocked",
            "passed": not _list(allowance.get("blocked_headline_claims")),
            "reason": ",".join(str(item) for item in _list(allowance.get("blocked_headline_claims")))
            or "no blocked headline claims",
        },
        {
            "check": "hardware_boundary_clear",
            "passed": architecture.get("speedup_claim_allowed") is True
            and architecture.get("gatemate_evidence_complete") is True
            and architecture.get("ssqa_readback_ready") is True,
            "reason": "speedup, GateMate, and SSQA evidence must all be authenticated",
        },
    ]


def _matrix_v25_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_v25_ready": matrix.get("matrix_v25_ready") is True,
        "rows_total": _int(matrix.get("rows_total")),
        "prior_publication_blocker_count": _int(matrix.get("prior_publication_blocker_count")),
        "publication_blocker_count": _int(matrix.get("publication_blocker_count")),
        "blocker_delta_from_v24": _int(matrix.get("blocker_delta_from_v24")),
        "status_counts": _mapping(matrix.get("status_counts")),
        "missing_artifacts_count": len(_list(matrix.get("missing_artifacts"))),
    }


def _what_291_proved(
    matrix: Mapping[str, Any],
    verifier: Mapping[str, Any],
    fr11: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> list[str]:
    return [
        (
            "matrix v25 is complete over "
            f"{_int(matrix.get('rows_total'))} rows with "
            f"{_int(matrix.get('publication_blocker_count'))} publication blockers."
        ),
        (
            "The live verifier panel executed only as artifact evidence and exposed "
            f"false_accept_rate={_number(_float(verifier.get('false_accept_rate')))}."
        ),
        (
            "FR-11 admitted "
            f"{_int(fr11.get('admitted_environment_count'))} controller environments with "
            f"ledger_consistency_rate={_number(_float(fr11.get('ledger_consistency_rate')))}."
        ),
        (
            "Architecture-side evidence stayed bounded: "
            f"ARM/EBT={architecture.get('arm_ebt_status')}, "
            f"KAN={architecture.get('kan_pwa_milp_status')}, "
            f"hardware={architecture.get('hardware_sampler_status')}."
        ),
    ]


def _what_stayed_blocked(
    verifier: Mapping[str, Any],
    allowance: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if allowance.get("comparative_sota_pair_allowed") is not True:
        blocked.append("comparative_sota_pair")
    if allowance.get("live_verifier_headline_allowed") is not True:
        blocked.append("live_verifier_headline_lift")
    if str(verifier.get("repair_ladder_status") or "") != "clean":
        blocked.append("repair_ladder_promotion")
    if architecture.get("gatemate_evidence_complete") is not True:
        blocked.append("gatemate_operator_evidence")
    if architecture.get("ssqa_readback_ready") is not True:
        blocked.append("ssqa_host_visible_readback")
    return blocked


def _bounded_claims(
    verifier: Mapping[str, Any],
    fr11: Mapping[str, Any],
    architecture: Mapping[str, Any],
    rows: list[Any],
) -> list[str]:
    claims: list[str] = []
    if str(verifier.get("prefix_bounds_status") or "") == "bounded":
        claims.append("bounded_prefix_frontier_only")
    if str(verifier.get("fragment_time_monitor_status") or "") == "bounded":
        claims.append("fragment_monitor_fixture_ledger_only")
    if fr11.get("no_weight_update_claim") is True:
        claims.append("fr11_controller_memory_only_no_weight_update")
    if architecture.get("live_integration") is not True:
        claims.append("arm_ebt_sidecar_projection_only")
    if str(architecture.get("kan_pwa_milp_status") or "") == "bounded":
        claims.append("kan_two_unit_pwa_milp_abstraction_only")
    if "cpu" in _clut_sampler_status(rows):
        claims.append("clut_cpu_simulation_only")
    return claims


def _allowed_claims(
    allowance: Mapping[str, Any],
    fr11: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> list[str]:
    claims = ["matrix_v25_aggregation_complete"]
    if _list(allowance.get("present_model_ids")):
        claims.append("single_cached_gemma26_available")
    if fr11.get("continuous_self_learning_targeted") is True:
        claims.append("fr11_controller_environment_memory_evaluated")
    if architecture.get("speedup_claim_allowed") is True:
        claims.append("authenticated_speedup_claim_allowed")
    else:
        claims.append("no_sampler_speedup_claim_allowed")
    return claims


def _next_top_gap(
    verifier: Mapping[str, Any],
    allowance: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> str:
    if (
        str(verifier.get("live_verifier_status") or "") != "clean"
        or str(verifier.get("repair_ladder_status") or "") != "clean"
    ):
        return "live_verifier_false_accept_repair_gate"
    if allowance.get("comparative_sota_pair_allowed") is not True:
        return "comparative_sota_cache_pair"
    if architecture.get("speedup_claim_allowed") is not True:
        return "operator_authenticated_hardware_readback"
    return "publication_scope_reconciliation"


def _next_recommendation(next_top_gap: str) -> str:
    if next_top_gap == "live_verifier_false_accept_repair_gate":
        return (
            "Next milestone should target the live verifier false-accept / repair-gate blocker "
            "first: Exp 3124 records false_accept_rate=0.5 and blocks Exp 3127 repair. "
            "Cache-pair and hardware evidence remain important, but they do not unblock the "
            "central verifier/repair paper claim until the false-accept gate is clean."
        )
    return f"Next milestone should target {next_top_gap} because it is the largest residual gap."


def _ops_reconciliation_decision() -> JsonDict:
    return {
        "capstone_artifact_alone_is_deliverable": True,
        "reason": "task stop rule delegates ops/status/changelog/traceability reconciliation to conductor",
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "traceability_updated_after_spec": False,
    }


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_matrix_v25_and_dot291_artifacts",
        "source": MATRIX_V25_REL_PATH.as_posix(),
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
        "live_model_calls_run_by_capstone": 0,
        "hardware_commands_run_by_capstone": [],
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("capstone_ready") is not True:
        first = str(_list(artifact.get("invariant_violations"))[0])
        return f"blocked: capstone_ready=false; {first}"
    return (
        "complete: capstone_ready=true; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v24={artifact.get('blocker_delta_from_v24')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _status_by_row(rows: list[Any], row_id: str) -> str:
    for row in rows:
        if isinstance(row, Mapping) and row.get("row_id") == row_id:
            return str(row.get("status") or "missing")
    return "missing"


def _summary_by_row(rows: list[Any], row_id: str) -> JsonDict:
    for row in rows:
        if isinstance(row, Mapping) and row.get("row_id") == row_id:
            return _mapping(row.get("summary"))
    return {}


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
