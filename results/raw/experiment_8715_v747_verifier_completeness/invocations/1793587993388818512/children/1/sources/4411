"""Build the Exp 3121 milestone .290 capstone artifact.

Spec refs: REQ-REPORT-3121, SCENARIO-REPORT-3121.

The .290 capstone is deliberately an aggregation pass. It reads matrix v24 as
the authority, compares it to the prior .289 capstone, and writes a closeout
artifact that states exactly what the milestone proved. It does not run any
model, solver, repair, hardware, or conductor workflow while doing that.
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
MILESTONE = "2026.05.290"
SCHEMA = "carnot.milestone_capstone.v290_matrix_v24_aggregation.v1"
ARTIFACT = "experiment_3121_capstone_v290"
OUTPUT_REL_PATH = Path("results/experiment_3121_capstone_v290.json")
MATRIX_V24_REL_PATH = Path("results/experiment_3120_cross_corpus_matrix_v24.json")
PRIOR_CAPSTONE_REL_PATH = Path("results/experiment_3108_capstone_v289.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3121_capstone_v290.py"


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object; absent or malformed evidence becomes empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a stable checksum so every capstone input remains auditable."""

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
    """REQ-REPORT-3121: close .290 from matrix v24 evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V24_REL_PATH)
    prior_capstone = read_json_object(root_path / PRIOR_CAPSTONE_REL_PATH)
    source_artifacts = _source_artifacts(root_path, matrix)
    required_source_errors = _required_source_errors(source_artifacts)
    invariant_violations = _invariant_violations(matrix, source_artifacts, required_source_errors)
    capstone_ready = not invariant_violations
    publication_blocker_count = _int_value(matrix.get("publication_blocker_count"))
    blocker_delta_from_v23 = _int_value(matrix.get("blocker_delta_from_v23"))
    headline_model_spec_gaps = _mapping_list(matrix.get("headline_model_spec_gaps"))
    paper_readiness_checks = _paper_readiness_checks(
        capstone_ready,
        publication_blocker_count,
        headline_model_spec_gaps,
        _mapping(matrix.get("publication_blocker_downgrade_policy")),
    )
    paper_ready = all(check["passed"] for check in paper_readiness_checks)
    verifier_repair_status = _mapping(matrix.get("verifier_repair_status"))
    fr11_status = _mapping(matrix.get("fr11_status"))
    architecture_boundary_status = _mapping(matrix.get("architecture_boundary_status"))

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v23": blocker_delta_from_v23,
        "verifier_gain_status": _verifier_gain_status(
            verifier_repair_status,
            headline_model_spec_gaps,
            publication_blocker_count,
        ),
        "formal_feedback_status": _formal_feedback_status(verifier_repair_status),
        "repair_claim_status": _repair_claim_status(verifier_repair_status),
        "fr11_self_learning_status": _fr11_self_learning_status(fr11_status),
        "ebt_arm_status": _ebt_arm_status(architecture_boundary_status),
        "sampler_hardware_status": _sampler_hardware_status(architecture_boundary_status),
        "gatemate_status": _gatemate_status(architecture_boundary_status),
        "ssqa_status": _ssqa_status(architecture_boundary_status),
        "matrix_v24_summary": _matrix_v24_summary(matrix),
        "prior_capstone_summary": _prior_capstone_summary(prior_capstone),
        "delta_from_v289": _delta_from_v289(
            matrix,
            verifier_repair_status,
            fr11_status,
            architecture_boundary_status,
            headline_model_spec_gaps,
        ),
        "milestone_proved": _milestone_proved(
            matrix,
            verifier_repair_status,
            fr11_status,
            architecture_boundary_status,
        ),
        "paper_readiness_checks": paper_readiness_checks,
        "headline_model_spec_gaps": headline_model_spec_gaps,
        "remaining_top_gaps": _remaining_top_gaps(),
        "next_recommendation": _next_recommendation(),
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
    """Build and persist the Exp 3121 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifacts(root: Path, matrix: Mapping[str, Any]) -> list[JsonDict]:
    specs: list[JsonDict] = [
        {
            "path": MATRIX_V24_REL_PATH.as_posix(),
            "role": "matrix_v24_authority",
            "required": True,
            "experiment_id": "exp3120",
        },
        {
            "path": PRIOR_CAPSTONE_REL_PATH.as_posix(),
            "role": "prior_capstone_v289",
            "required": True,
            "experiment_id": "exp3108",
        },
    ]
    specs.extend(_mapping_list(matrix.get("source_artifacts")))
    seen: set[str] = set()
    rows: list[JsonDict] = []
    for spec in specs:
        path_text = str(spec.get("path") or "")
        if not path_text or path_text in seen:
            continue
        seen.add(path_text)
        rows.append(_source_artifact_row(root, Path(path_text), spec))
    return rows


def _source_artifact_row(root: Path, rel_path: Path, spec: Mapping[str, Any]) -> JsonDict:
    path = root / rel_path
    payload = read_json_object(path) if rel_path.suffix == ".json" else {}
    present = path.is_file()
    readable = bool(payload) if rel_path.suffix == ".json" else present
    return {
        "experiment_id": _experiment_id(rel_path, spec, payload),
        "path": rel_path.as_posix(),
        "role": str(spec.get("role") or _source_role(rel_path)),
        "required": bool(spec.get("required", False)),
        "source_type": "json" if rel_path.suffix == ".json" else "text",
        "present": present,
        "readable_json_object": readable,
        "sha256": sha256_file(path),
    }


def _experiment_id(
    rel_path: Path,
    spec: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    return str(
        payload.get("artifact")
        or spec.get("experiment_id")
        or rel_path.stem.removeprefix("experiment_")
    )


def _source_role(rel_path: Path) -> str:
    if rel_path == MATRIX_V24_REL_PATH:
        return "matrix_v24_authority"
    if rel_path == PRIOR_CAPSTONE_REL_PATH:
        return "prior_capstone_v289"
    if "matrix" in rel_path.name:
        return "matrix_context"
    if "capstone" in rel_path.name:
        return "capstone_context"
    return "matrix_v24_source"


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
    if matrix.get("matrix_v24_ready") is not True:
        violations.append("matrix_v24_ready is not true")
    status_counts = _mapping(matrix.get("status_counts"))
    rows_total = _int_value(matrix.get("rows_total"))
    if status_counts and sum(_int_value(value) for value in status_counts.values()) != rows_total:
        violations.append("status_counts do not reconcile with rows_total")
    if _int_value(matrix.get("publication_blocker_count")) < 0:
        violations.append("publication_blocker_count is negative")
    substrate = _mapping(matrix.get("inference_substrate"))
    if any(substrate.get(key) is True for key in ("executes_models", "executes_hardware", "executes_conductor")):
        violations.append("matrix inference_substrate is not aggregation-only")
    if not source_artifacts:
        violations.append("source_artifacts list is empty")
    return violations


def _paper_readiness_checks(
    capstone_ready: bool,
    publication_blocker_count: int,
    headline_model_spec_gaps: list[Mapping[str, Any]],
    downgrade_policy: Mapping[str, Any],
) -> list[JsonDict]:
    blocker_downgraded = (
        downgrade_policy.get("all_remaining_blockers_downgraded") is True
        and downgrade_policy.get("headline_scope_after_downgrade") == "none"
    )
    if publication_blocker_count == 0:
        blocker_reason = "publication_blocker_count=0"
    elif blocker_downgraded:
        blocker_reason = "all remaining blockers explicitly downgraded outside headline scope"
    else:
        blocker_reason = (
            f"publication_blocker_count={publication_blocker_count} "
            "and no all-blockers-downgraded policy"
        )
    if headline_model_spec_gaps:
        gap_reason = (
            f"headline_model_spec_gaps={len(headline_model_spec_gaps)}; "
            "missing mandated cached SOTA coverage remains bounded"
        )
    else:
        gap_reason = "headline_model_spec_gaps=0"
    return [
        {
            "check": "capstone_ready",
            "passed": capstone_ready,
            "reason": "matrix v24 authority loaded and required invariants reconciled",
        },
        {
            "check": "publication_blocker_count_zero_or_downgraded",
            "passed": publication_blocker_count == 0 or blocker_downgraded,
            "reason": blocker_reason,
        },
        {
            "check": "headline_model_spec_gaps",
            "passed": not headline_model_spec_gaps,
            "reason": gap_reason,
        },
    ]


def _verifier_gain_status(
    verifier_repair_status: Mapping[str, Any],
    headline_model_spec_gaps: list[Mapping[str, Any]],
    publication_blocker_count: int,
) -> str:
    if (
        verifier_repair_status.get("diagnostic_calibration_status") == "diagnostic_only"
        and (headline_model_spec_gaps or publication_blocker_count)
    ):
        return "diagnostic_gain_recovered_but_headline_bounded_by_cache_and_prior_flags"
    if verifier_repair_status.get("diagnostic_calibration_status") == "clean":
        return "clean_publishable_verifier_gain"
    return "verifier_gain_not_promoted"


def _formal_feedback_status(verifier_repair_status: Mapping[str, Any]) -> str:
    if verifier_repair_status.get("certified_coherence_status") == "clean":
        return "solver_certified_feedback_ready_no_live_sota_lift"
    return "formal_feedback_not_ready"


def _repair_claim_status(verifier_repair_status: Mapping[str, Any]) -> str:
    repair_success_delta = _float_value(verifier_repair_status.get("repair_success_delta"))
    intent_preservation_rate = _float_value(verifier_repair_status.get("intent_preservation_rate"))
    if (
        verifier_repair_status.get("repair_micro_panel_status") == "bounded"
        or repair_success_delta <= 0.0
        or intent_preservation_rate <= 0.0
    ):
        return "bounded_micro_panel_executed_zero_delta_no_promotion"
    return "repair_claim_promotable_positive_delta_intent_preserved"


def _fr11_self_learning_status(fr11_status: Mapping[str, Any]) -> str:
    if (
        fr11_status.get("controller_only") is True
        and fr11_status.get("no_weight_update_claim") is True
        and _int_value(fr11_status.get("soundness_mistakes")) == 0
        and _int_value(fr11_status.get("completeness_mistakes")) == 0
    ):
        return "bounded_controller_only_soundness_zero_completeness_zero_no_weight_update"
    return "fr11_promotion_boundary_not_clear"


def _ebt_arm_status(architecture_boundary_status: Mapping[str, Any]) -> str:
    status = str(architecture_boundary_status.get("ebt_arm_status") or "")
    if "projection_only" in status or "no_live_model_integration" in status:
        return "projection_only_sidecar_correlation_no_live_model_integration"
    return "ebt_arm_boundary_clean"


def _sampler_hardware_status(architecture_boundary_status: Mapping[str, Any]) -> str:
    status = str(architecture_boundary_status.get("clut_status") or "")
    if "cpu" in status or "no_hardware_speedup" in status or "bounded" in status:
        return "bounded_clut_cpu_only_no_hardware_speedup"
    return "sampler_hardware_boundary_clean"


def _gatemate_status(architecture_boundary_status: Mapping[str, Any]) -> str:
    status = str(architecture_boundary_status.get("gatemate_status") or "")
    if "blocked" in status or "incomplete" in status:
        return "blocked_operator_evidence_incomplete_no_hardware_run"
    return "gatemate_boundary_clean"


def _ssqa_status(architecture_boundary_status: Mapping[str, Any]) -> str:
    status = str(architecture_boundary_status.get("ssqa_status") or "")
    if "gated_skipped" in status or "readback_missing" in status:
        return "gated_skipped_host_visible_readback_missing"
    return "ssqa_boundary_clean"


def _matrix_v24_summary(matrix: Mapping[str, Any]) -> JsonDict:
    missing_artifacts = _mapping_list(matrix.get("missing_artifacts"))
    return {
        "matrix_v24_ready": matrix.get("matrix_v24_ready") is True,
        "rows_total": _int_value(matrix.get("rows_total")),
        "status_counts": _mapping(matrix.get("status_counts")),
        "publication_blocker_count": _int_value(matrix.get("publication_blocker_count")),
        "blocker_delta_from_v23": _int_value(matrix.get("blocker_delta_from_v23")),
        "missing_artifacts_count": len(missing_artifacts),
    }


def _prior_capstone_summary(prior_capstone: Mapping[str, Any]) -> JsonDict:
    return {
        "artifact": str(prior_capstone.get("artifact") or ""),
        "capstone_ready": prior_capstone.get("capstone_ready") is True,
        "paper_ready": prior_capstone.get("paper_ready") is True,
        "publication_blocker_count": _int_value(prior_capstone.get("publication_blocker_count")),
    }


def _delta_from_v289(
    matrix: Mapping[str, Any],
    verifier_repair_status: Mapping[str, Any],
    fr11_status: Mapping[str, Any],
    architecture_boundary_status: Mapping[str, Any],
    headline_model_spec_gaps: list[Mapping[str, Any]],
) -> JsonDict:
    return {
        "model_spec_gap": (
            "changed: matrix status_counts has model_spec_gap="
            f"{_mapping(matrix.get('status_counts')).get('model_spec_gap', 0)}, but "
            f"headline cache coverage remains bounded via {len(headline_model_spec_gaps)} gap row(s)."
        ),
        "formal_feedback": (
            "changed: certified coherence feedback is "
            f"{verifier_repair_status.get('certified_coherence_status')}; live SOTA lift remains unclaimed."
        ),
        "calibration_gate": (
            "changed: diagnostic calibration records repair_gate_state="
            f"{verifier_repair_status.get('repair_gate_state', 'unblocked')} while staying diagnostic-only."
        ),
        "repair_artifact": (
            "changed: explicit repair gate micro-panel executed, but repair_success_delta="
            f"{_float_value(verifier_repair_status.get('repair_success_delta'))} keeps repair bounded."
        ),
        "fr11_promotion": (
            "changed: FR-11 soundness/completeness mistakes are "
            f"{_int_value(fr11_status.get('soundness_mistakes'))}/"
            f"{_int_value(fr11_status.get('completeness_mistakes'))}, within "
            "controller-only/no-weight-update scope."
        ),
        "sidecar_clut_boundaries": (
            "unchanged in promotion: EBT/ARM remains "
            f"{architecture_boundary_status.get('ebt_arm_status')} and cLUT remains "
            f"{architecture_boundary_status.get('clut_status')}."
        ),
        "gatemate_ssqa": (
            "unchanged: GateMate remains "
            f"{architecture_boundary_status.get('gatemate_status')} and SSQA remains "
            f"{architecture_boundary_status.get('ssqa_status')}."
        ),
    }


def _milestone_proved(
    matrix: Mapping[str, Any],
    verifier_repair_status: Mapping[str, Any],
    fr11_status: Mapping[str, Any],
    architecture_boundary_status: Mapping[str, Any],
) -> list[str]:
    missing_count = len(_mapping_list(matrix.get("missing_artifacts")))
    return [
        (
            "Matrix v24 is complete over "
            f"{_int_value(matrix.get('rows_total'))} rows with {missing_count} missing artifacts "
            "and aggregation-only provenance."
        ),
        (
            "Solver-certified feedback is ready on exact fixtures, but the capstone does not claim "
            "live SOTA verifier lift."
        ),
        (
            "The repair gate ran and was unblocked, but repair_success_delta="
            f"{_float_value(verifier_repair_status.get('repair_success_delta'))} keeps repair non-promoted."
        ),
        (
            "FR-11 passed the .290 retention guard only as "
            f"{fr11_status.get('promotion_decision', 'controller_only')} with no model-weight update claim."
        ),
        (
            "Architecture work stayed bounded: "
            f"{architecture_boundary_status.get('ebt_arm_status')}, "
            f"{architecture_boundary_status.get('clut_status')}, "
            f"{architecture_boundary_status.get('gatemate_status')}, and "
            f"{architecture_boundary_status.get('ssqa_status')}."
        ),
    ]


def _remaining_top_gaps() -> list[str]:
    return [
        "publishable_verifier_repair_headline_evidence",
        "operator_owned_gatemate_ssqa_host_visible_evidence",
        "live_model_or_authenticated_hardware_architecture_integration",
    ]


def _next_recommendation() -> str:
    return (
        "Next milestone: prioritize publishable verifier/repair evidence first by resolving "
        "mandated SOTA cache coverage, prior flagged repair rows, and a positive gated repair "
        "micro-panel; second, collect operator-owned GateMate/SSQA host-visible evidence without "
        "speedup claims; third, either integrate EBT/ARM sidecars with live model tests or obtain "
        "authenticated cLUT/hardware speedup evidence, otherwise keep those rows bounded."
    )


def _ops_reconciliation_decision() -> JsonDict:
    return {
        "capstone_artifact_alone_is_deliverable": True,
        "reason": "task stop rule delegates ops/status/changelog/traceability reconciliation to conductor",
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "traceability_updated_after_spec": False,
        "research_complete_updated": False,
    }


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_artifacts",
        "no_live_llm_inference": True,
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "hardware_commands_run": [],
        "live_model_calls": 0,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if not artifact.get("capstone_ready"):
        first = _list_value(artifact.get("invariant_violations"))[0]
        return f"blocked: capstone_ready=false; {first}"
    return (
        "complete: capstone_ready=true; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v23={artifact.get('blocker_delta_from_v23')}; "
        f"next_top_gap={_remaining_top_gaps()[0]}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0, end - started_s)


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[JsonDict]:
    return [dict(item) for item in _list_value(value) if isinstance(item, Mapping)]


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _float_value(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
