"""Build the Exp 3162 milestone .293 capstone artifact.

Spec refs: REQ-REPORT-3162, SCENARIO-REPORT-3162.

This module is a closeout aggregator, not a fresh experiment runner. It reads
checked-in matrix and source artifacts, then states the publication boundary in
plain fields so downstream planning can avoid turning blocked live evidence,
skipped repair, controller-only FR-11 memory, sidecar energy scores, KAN monitor
records, or hardware-adjacent notes into headline claims.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.293"
SCHEMA = "carnot.milestone_capstone.v293_matrix_v27_aggregation.v1"
ARTIFACT = "experiment_3162_capstone_v293"
OUTPUT_REL_PATH = Path("results/experiment_3162_capstone_v293.json")
MATRIX_V27_REL_PATH = Path("results/experiment_3161_cross_corpus_matrix_v27.json")
CAPSTONE_V292_REL_PATH = Path("results/experiment_3148_capstone_v292.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3162_capstone_v293.py"


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and return an empty mapping for unusable evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a stable file checksum for capstone source traceability."""

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
    """REQ-REPORT-3162: close .293 from matrix v27 without new execution."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V27_REL_PATH)
    capstone_v292 = read_json_object(root_path / CAPSTONE_V292_REL_PATH)
    recovery = _mapping(matrix.get("false_accept_recovery_summary"))
    repair = _mapping(matrix.get("repair_summary"))
    fr11 = _mapping(matrix.get("fr11_summary"))
    architecture = _mapping(matrix.get("architecture_boundary_summary"))
    paper_implications = _mapping(matrix.get("paper_readiness_implications"))
    source_artifacts = _source_artifacts(root_path, matrix)
    invariant_violations = _invariant_violations(matrix, capstone_v292)
    capstone_ready = not invariant_violations
    publication_blocker_count = _int(matrix.get("publication_blocker_count"))
    blocker_delta_from_v26 = _int(matrix.get("blocker_delta_from_v26"))
    verifier_clean = _verifier_clean(recovery)
    repair_clean = _repair_clean(repair)
    fr11_promotion_allowed = _fr11_promotion_allowed(fr11)
    architecture_clean = _architecture_headline_safe(architecture)
    paper_ready = (
        capstone_ready
        and publication_blocker_count == 0
        and verifier_clean
        and repair_clean
        and fr11_promotion_allowed
        and architecture_clean
    )
    next_top_gap = _next_top_gap(
        publication_blocker_count,
        verifier_clean,
        repair_clean,
        fr11_promotion_allowed,
        architecture_clean,
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_v293_ready": capstone_ready,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v26": blocker_delta_from_v26,
        "next_top_gap": next_top_gap,
        "verifier_evidence_status": _verifier_evidence_status(recovery),
        "repair_gate_status": _repair_gate_status(repair),
        "repair_ladder_status": _repair_ladder_status(repair),
        "repair_was_executed": repair.get("live_repair_executed") is True,
        "repair_skip_was_correct": _repair_skip_was_correct(repair),
        "fr11_self_learning_status": _fr11_self_learning_status(fr11),
        "fr11_improved_from_v292": _fr11_improved_from_v292(capstone_v292, fr11),
        "fr11_promotion_allowed": fr11_promotion_allowed,
        "ebt_arm_status": _ebt_arm_status(architecture),
        "kan_status": _kan_status(architecture),
        "sampler_hardware_status": _sampler_hardware_status(architecture),
        "hardware_energy_kan_headline_safe": architecture_clean,
        "paper_readiness_checks": _paper_readiness_checks(
            capstone_ready,
            publication_blocker_count,
            verifier_clean,
            repair_clean,
            fr11_promotion_allowed,
            architecture_clean,
        ),
        "matrix_v27_summary": _matrix_v27_summary(matrix),
        "what_293_proved": _what_293_proved(matrix, recovery, repair, fr11, architecture),
        "next_milestone_recommendations": _next_milestone_recommendations(next_top_gap),
        "headline_safe_claims": _headline_safe_claims(recovery, fr11, architecture),
        "headline_forbidden_claims": _headline_forbidden_claims(paper_implications),
        "missing_artifacts": _list(matrix.get("missing_artifacts")),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "required_source_errors": _list(matrix.get("required_source_errors")),
        "invariant_violations": invariant_violations,
        "ops_reconciliation_decision": _ops_reconciliation_decision(),
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_repair_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
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
    """Build and persist the Exp 3162 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifacts(root: Path, matrix: Mapping[str, Any]) -> list[JsonDict]:
    specs: list[Mapping[str, Any]] = [
        {
            "experiment_id": "exp3161",
            "path": MATRIX_V27_REL_PATH,
            "loaded_path": MATRIX_V27_REL_PATH,
            "role": "matrix_v27_authority",
            "required": True,
            "ready_field": "matrix_v27_ready",
            "source_type": "json",
        }
    ]
    specs.extend(
        item for item in _list(matrix.get("source_artifacts")) if isinstance(item, Mapping)
    )
    return [_source_artifact_row(root, spec) for spec in specs]


def _source_artifact_row(root: Path, spec: Mapping[str, Any]) -> JsonDict:
    rel_path = Path(str(spec.get("path") or ""))
    loaded_path = Path(str(spec.get("loaded_path") or rel_path.as_posix()))
    payload = read_json_object(root / loaded_path)
    return {
        "experiment_id": str(spec.get("experiment_id") or payload.get("artifact") or rel_path.stem),
        "path": rel_path.as_posix(),
        "loaded_path": loaded_path.as_posix(),
        "role": str(spec.get("role") or "source"),
        "required": spec.get("required") is True,
        "ready_field": str(spec.get("ready_field") or ""),
        "present": (root / rel_path).is_file() or (root / loaded_path).is_file(),
        "primary_present": (root / rel_path).is_file(),
        "alias_loaded": loaded_path != rel_path,
        "readable_json_object": bool(payload),
        "sha256": sha256_file(root / loaded_path),
        "source_type": str(spec.get("source_type") or "json"),
    }


def _invariant_violations(matrix: Mapping[str, Any], capstone_v292: Mapping[str, Any]) -> list[str]:
    status_counts = _mapping(matrix.get("status_counts"))
    rows_total = _int(matrix.get("rows_total"))
    checks = (
        (not matrix, "matrix_v27 authority is missing or malformed"),
        (
            bool(matrix) and matrix.get("matrix_v27_ready") is not True,
            "matrix_v27_ready is not true",
        ),
        (not capstone_v292, "capstone_v292 authority is missing or malformed"),
        (
            bool(capstone_v292) and capstone_v292.get("capstone_ready") is not True,
            "capstone_v292 capstone_ready is not true",
        ),
        (
            bool(status_counts)
            and sum(_int(value) for value in status_counts.values()) != rows_total,
            "status_counts do not reconcile with rows_total",
        ),
        (
            len(_list(matrix.get("publication_blockers")))
            != _int(matrix.get("publication_blocker_count")),
            "publication_blocker_count does not match publication_blockers",
        ),
        (
            bool(_list(matrix.get("required_source_errors"))),
            "matrix_v27 reports required source errors",
        ),
        (
            bool(_list(matrix.get("invariant_violations"))),
            "matrix_v27 reports invariant violations",
        ),
        (
            _substrate_runs_execution(matrix),
            "matrix_v27 inference_substrate is not aggregation-only",
        ),
    )
    return [message for failed, message in checks if failed]


def _substrate_runs_execution(matrix: Mapping[str, Any]) -> bool:
    substrate = _mapping(matrix.get("inference_substrate"))
    return any(
        substrate.get(key) is True
        for key in (
            "executes_models",
            "executes_verifiers",
            "executes_repairs",
            "executes_solvers",
            "executes_hardware",
            "executes_conductor",
        )
    )


def _verifier_clean(recovery: Mapping[str, Any]) -> bool:
    return (
        recovery.get("live_verifier_evidence_trusted") is True
        and recovery.get("preflight_passed") is True
        and recovery.get("clean_live_rerun_status") == "clean"
    )


def _repair_clean(repair: Mapping[str, Any]) -> bool:
    return (
        repair.get("repair_gate_status") == "clean"
        and repair.get("repair_gate_state") == "unblocked"
        and repair.get("repair_ladder_status") == "clean"
        and repair.get("live_repair_executed") is True
        and repair.get("repair_claim_allowed") is True
    )


def _fr11_promotion_allowed(fr11: Mapping[str, Any]) -> bool:
    return (
        _float(fr11.get("ledger_consistency_rate")) == 1.0
        and fr11.get("model_weight_learning_allowed") is True
        and fr11.get("no_weight_update_claim") is not True
    )


def _architecture_headline_safe(architecture: Mapping[str, Any]) -> bool:
    return (
        architecture.get("live_integration_claim_allowed") is True
        and architecture.get("deployed_verifier_claim_allowed") is True
        and architecture.get("authenticated_speedup_claim_allowed") is True
        and bool(_list(architecture.get("hardware_commands_run")))
    )


def _verifier_evidence_status(recovery: Mapping[str, Any]) -> str:
    if _verifier_clean(recovery):
        return "clean_live_verifier_evidence_trusted"
    if (
        recovery.get("known_false_accept_recovery_preserved") is True
        and recovery.get("live_verifier_evidence_trusted") is not True
    ):
        return (
            "corrigendum_preserved_exact_replay_but_did_not_unblock_repair_live_evidence_untrusted"
        )
    return f"{recovery.get('recovery_claim_status') or 'missing'}_live_verifier_not_headline"


def _repair_gate_status(repair: Mapping[str, Any]) -> str:
    if (
        repair.get("repair_gate_status") == "clean"
        and repair.get("repair_gate_state") == "unblocked"
    ):
        return "clean_repair_gate_unblocked"
    if repair.get("repair_gate_status") == "blocked":
        return "blocked_pending_clean_rerun_gate_failed"
    return f"{repair.get('repair_gate_status') or 'missing'}_repair_gate_not_unblocked"


def _repair_ladder_status(repair: Mapping[str, Any]) -> str:
    if repair.get("repair_ladder_status") == "clean" and repair.get("live_repair_executed") is True:
        return "clean_repair_ladder_executed"
    if _repair_skip_was_correct(repair):
        return "correctly_skipped_gate_blocked_no_live_repair_attempts"
    return f"{repair.get('repair_ladder_status') or 'missing'}_repair_ladder_not_promoted"


def _repair_skip_was_correct(repair: Mapping[str, Any]) -> bool:
    return (
        repair.get("live_repair_executed") is not True
        and repair.get("repair_attempt_count") in (None, 0)
        and repair.get("repair_gate_state") == "blocked"
    )


def _fr11_self_learning_status(fr11: Mapping[str, Any]) -> str:
    if _fr11_promotion_allowed(fr11):
        return "clean_fr11_promotion_allowed"
    ledger = _number(_float(fr11.get("ledger_consistency_rate")))
    if fr11.get("no_weight_update_claim") is True:
        return f"improved_to_{ledger}_but_promotion_blocked_controller_memory_only_no_weight_update"
    return f"bounded_fr11_ledger_{ledger}_promotion_not_allowed"


def _fr11_improved_from_v292(capstone_v292: Mapping[str, Any], fr11: Mapping[str, Any]) -> bool:
    current = _float(fr11.get("ledger_consistency_rate")) or 0.0
    prior_numbers = [
        float(value)
        for value in re.findall(
            r"\d+\.\d+", str(capstone_v292.get("fr11_self_learning_status") or "")
        )
    ]
    prior_best = max(prior_numbers) if prior_numbers else 0.0
    return current > prior_best


def _ebt_arm_status(architecture: Mapping[str, Any]) -> str:
    if architecture.get("live_integration_claim_allowed") is True:
        return "clean_energy_sidecar_live_integration"
    status = str(architecture.get("energy_sidecar_status") or "missing")
    auc = _number(_float(architecture.get("scalar_energy_auc")))
    rows = _int(architecture.get("exact_labeled_row_count"))
    blockers = _int(architecture.get("energy_residual_blocker_count"))
    return f"{status}_scalar_auc_{auc}_exact_rows_{rows}_no_live_integration_blockers_{blockers}"


def _kan_status(architecture: Mapping[str, Any]) -> str:
    if architecture.get("deployed_verifier_claim_allowed") is True:
        return "clean_deployed_kan_verifier_claim"
    status = str(architecture.get("kan_status") or "missing")
    records = _int(architecture.get("monitor_record_count"))
    new_records = _int(architecture.get("new_monitor_record_count"))
    blockers = _int(architecture.get("kan_residual_blocker_count"))
    return f"{status}_monitor_records_{records}_new_{new_records}_no_deployed_verifier_blockers_{blockers}"


def _sampler_hardware_status(architecture: Mapping[str, Any]) -> str:
    if architecture.get("authenticated_speedup_claim_allowed") is True and bool(
        _list(architecture.get("hardware_commands_run"))
    ):
        return "clean_authenticated_sampler_hardware_speedup"
    missing = _int(architecture.get("missing_operator_evidence_count"))
    return (
        f"blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_{missing}"
    )


def _next_top_gap(
    publication_blocker_count: int,
    verifier_clean: bool,
    repair_clean: bool,
    fr11_promotion_allowed: bool,
    architecture_clean: bool,
) -> str:
    if publication_blocker_count > 0 and (not verifier_clean or not repair_clean):
        return "clean_live_verifier_corrigendum_repair_gate"
    if not fr11_promotion_allowed:
        return "fr11_ledger_consistency_promotion_gate"
    if not architecture_clean:
        return "architecture_boundary_headline_evidence"
    return "publication_scope_reconciliation"


def _paper_readiness_checks(
    capstone_ready: bool,
    publication_blocker_count: int,
    verifier_clean: bool,
    repair_clean: bool,
    fr11_promotion_allowed: bool,
    architecture_clean: bool,
) -> list[JsonDict]:
    return [
        {"check": "capstone_ready", "passed": capstone_ready},
        {"check": "publication_blocker_count_zero", "passed": publication_blocker_count == 0},
        {"check": "clean_verifier_evidence", "passed": verifier_clean},
        {"check": "repair_status_promotable", "passed": repair_clean},
        {"check": "fr11_promotion_allowed", "passed": fr11_promotion_allowed},
        {"check": "architecture_boundary_headline_safe", "passed": architecture_clean},
    ]


def _matrix_v27_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_v27_ready": matrix.get("matrix_v27_ready") is True,
        "rows_total": _int(matrix.get("rows_total")),
        "prior_publication_blocker_count": _int(matrix.get("prior_publication_blocker_count")),
        "publication_blocker_count": _int(matrix.get("publication_blocker_count")),
        "blocker_delta_from_v26": _int(matrix.get("blocker_delta_from_v26")),
        "status_counts": _mapping(matrix.get("status_counts")),
        "missing_artifact_count": len(_list(matrix.get("missing_artifacts"))),
    }


def _what_293_proved(
    matrix: Mapping[str, Any],
    recovery: Mapping[str, Any],
    repair: Mapping[str, Any],
    fr11: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> list[str]:
    return [
        (
            "corrigendum preserved exact known-false-accept replay, but live verifier "
            f"evidence_trusted={str(recovery.get('live_verifier_evidence_trusted')).lower()} "
            f"and clean_live_rerun_status={recovery.get('clean_live_rerun_status')}."
        ),
        (
            "repair was not executed: repair_gate_state="
            f"{repair.get('repair_gate_state')}, repair_ladder_status="
            f"{repair.get('repair_ladder_status')}, repair_attempt_count="
            f"{_int(repair.get('repair_attempt_count'))}."
        ),
        (
            "FR-11 improved to ledger_consistency_rate="
            f"{_number(_float(fr11.get('ledger_consistency_rate')))}, but promotion remains "
            f"{fr11.get('promotion_recommendation')}."
        ),
        (
            "matrix v27 is complete over "
            f"{_int(matrix.get('rows_total'))} rows with "
            f"{_int(matrix.get('publication_blocker_count'))} publication blockers."
        ),
        (
            "Architecture claims stay bounded: energy_sidecar="
            f"{architecture.get('energy_sidecar_status')}, KAN={architecture.get('kan_status')}, "
            f"hardware={architecture.get('hardware_status')}."
        ),
    ]


def _next_milestone_recommendations(next_top_gap: str) -> list[str]:
    if next_top_gap == "clean_live_verifier_corrigendum_repair_gate":
        return [
            "Run a clean live verifier rerun with complete duration, seed, checksum, transcript, and model-load evidence.",
            "Open the repair gate only after the clean live verifier rerun is trusted and the false-accept regression rows stay blocked.",
            "Execute the multi-turn repair ladder only after the gate opens; otherwise keep repair correctly skipped.",
        ]
    return [f"Target {next_top_gap} before adding new headline claim surfaces."]


def _headline_safe_claims(
    recovery: Mapping[str, Any], fr11: Mapping[str, Any], architecture: Mapping[str, Any]
) -> list[str]:
    claims = ["matrix_v27_aggregation_complete"]
    if recovery.get("known_false_accept_recovery_preserved") is True:
        claims.append("exact_replay_known_false_accept_rows_preserved")
    if _float(fr11.get("ledger_consistency_rate")):
        claims.append("fr11_controller_memory_ledger_replay_bounded")
    if _int(architecture.get("monitor_record_count")):
        claims.append("kan_exact_monitor_records_bounded")
    return claims


def _headline_forbidden_claims(paper_implications: Mapping[str, Any]) -> list[str]:
    return _text_list(paper_implications.get("blocked_headline_claims"))


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
        "kind": "aggregation_from_checked_in_matrix_v27_capstone_v292_and_dot293_artifacts",
        "source": MATRIX_V27_REL_PATH.as_posix(),
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
    if artifact.get("capstone_v293_ready") is not True:
        first = str(_list(artifact.get("invariant_violations"))[0])
        return f"blocked: capstone_v293_ready=false; {first}"
    return (
        "complete: capstone_v293_ready=true; "
        f"capstone_ready={str(artifact.get('capstone_ready')).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v26={artifact.get('blocker_delta_from_v26')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text_list(value: Any) -> list[str]:
    return [str(item) for item in _list(value)]


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number(value: float | None) -> str:
    if value is None:
        return "missing"
    if value == int(value):
        return f"{value:.1f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
