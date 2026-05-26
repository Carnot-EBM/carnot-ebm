"""Build the Exp 3148 milestone .292 capstone artifact.

Spec refs: REQ-REPORT-3148, SCENARIO-REPORT-3148.

This is an aggregation closeout, not a fresh experiment. The module reads the
checked-in matrix v26 artifact and its traced source files, then states what
those files support without rerunning models, verifiers, repair, solvers, or
hardware. That boundary is important because .292 includes useful
false-accept recovery evidence, but some of the evidence is adversarially
flagged and the repair ladder never executed.
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
MILESTONE = "2026.05.292"
SCHEMA = "carnot.milestone_capstone.v292_matrix_v26_aggregation.v1"
ARTIFACT = "experiment_3148_capstone_v292"
OUTPUT_REL_PATH = Path("results/experiment_3148_capstone_v292.json")
MATRIX_V26_REL_PATH = Path("results/experiment_3147_cross_corpus_matrix_v26.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3148_capstone_v292.py"

EXP3135_REL_PATH = Path("results/experiment_3135_archive_v291_activate_v292.json")
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path(
    "results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json"
)
EXP3139_REL_PATH = Path("results/experiment_3139_live_sota_verifier_rerun_v7.json")
EXP3140_REL_PATH = Path("results/experiment_3140_repair_gate_unlock_decision_v1.json")
EXP3141_REL_PATH = Path("results/experiment_3141_multi_turn_repair_ladder_v2.json")
EXP3142_REL_PATH = Path("results/experiment_3142_fr11_vera_evoenv_hardening_v2.json")
EXP3143_REL_PATH = Path(
    "results/experiment_3143_fr11_experience_driven_verifier_memory_v1.json"
)
EXP3144_REL_PATH = Path(
    "results/experiment_3144_ebt_arm_false_accept_calibration_boundary_v3.json"
)
EXP3145_REL_PATH = Path("results/experiment_3145_kan_proof_carrying_monitor_boundary_v2.json")
EXP3146_REL_PATH = Path("results/experiment_3146_hardware_sampler_evidence_boundary_v6.json")

SOURCE_SPECS: tuple[JsonDict, ...] = (
    {
        "experiment_id": "exp3135",
        "path": EXP3135_REL_PATH,
        "role": "archive_v291_activate_v292",
        "required": False,
        "ready_field": "archive_v291_activate_v292_ready",
    },
    {
        "experiment_id": "exp3136",
        "path": EXP3136_REL_PATH,
        "role": "false_accept_autopsy",
        "required": False,
        "ready_field": "false_accept_autopsy_v1_ready",
    },
    {
        "experiment_id": "exp3137",
        "path": EXP3137_REL_PATH,
        "role": "accept_abstain_contract",
        "required": False,
        "ready_field": "acceptance_contract_v1_ready",
    },
    {
        "experiment_id": "exp3138",
        "path": EXP3138_REL_PATH,
        "role": "canonical_grounding",
        "required": False,
        "ready_field": "canonical_grounding_pilot_v1_ready",
    },
    {
        "experiment_id": "exp3139",
        "path": EXP3139_REL_PATH,
        "role": "live_verifier_rerun",
        "required": False,
        "ready_field": "live_verifier_rerun_v7_ready",
    },
    {
        "experiment_id": "exp3140",
        "path": EXP3140_REL_PATH,
        "role": "repair_gate",
        "required": False,
        "ready_field": "repair_gate_decision_v1_ready",
    },
    {
        "experiment_id": "exp3141",
        "path": EXP3141_REL_PATH,
        "role": "repair_ladder",
        "required": False,
        "ready_field": "multi_turn_repair_ladder_v2_ready",
    },
    {
        "experiment_id": "exp3142",
        "path": EXP3142_REL_PATH,
        "role": "fr11_vera_evoenv",
        "required": False,
        "ready_field": "fr11_vera_evoenv_v2_ready",
    },
    {
        "experiment_id": "exp3143",
        "path": EXP3143_REL_PATH,
        "role": "fr11_experience_memory",
        "required": False,
        "ready_field": "fr11_experience_verifier_memory_v1_ready",
    },
    {
        "experiment_id": "exp3144",
        "path": EXP3144_REL_PATH,
        "role": "ebt_arm_calibration",
        "required": False,
        "ready_field": "ebt_arm_false_accept_calibration_v3_ready",
    },
    {
        "experiment_id": "exp3145",
        "path": EXP3145_REL_PATH,
        "role": "kan_monitor_boundary",
        "required": False,
        "ready_field": "kan_proof_carrying_monitor_v2_ready",
    },
    {
        "experiment_id": "exp3146",
        "path": EXP3146_REL_PATH,
        "role": "hardware_boundary",
        "required": False,
        "ready_field": "hardware_sampler_evidence_boundary_v6_ready",
    },
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed on missing or malformed evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a reproducible checksum for source-artifact traceability."""

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
    """REQ-REPORT-3148: close .292 from matrix v26 and source artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V26_REL_PATH)
    allowance = _mapping(matrix.get("headline_claim_allowance_summary"))
    recovery = _mapping(matrix.get("false_accept_recovery_summary"))
    repair = _mapping(matrix.get("repair_gate_summary"))
    fr11 = _mapping(matrix.get("fr11_summary"))
    architecture = _mapping(matrix.get("architecture_boundary_summary"))
    source_artifacts = _source_artifacts(root_path, matrix)
    invariant_violations = _invariant_violations(matrix, source_artifacts)
    capstone_ready = not invariant_violations
    publication_blocker_count = _int(matrix.get("publication_blocker_count"))
    blocker_delta_from_v25 = _int(matrix.get("blocker_delta_from_v25"))
    blocked_headline_claims = _text_list(allowance.get("blocked_headline_claims"))
    paper_ready = (
        capstone_ready
        and publication_blocker_count == 0
        and not blocked_headline_claims
    )
    next_top_gap = _next_top_gap(allowance, recovery, repair, fr11, architecture)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v25": blocker_delta_from_v25,
        "next_top_gap": next_top_gap,
        "false_accept_recovery_status": _false_accept_recovery_status(recovery),
        "live_verifier_status": str(recovery.get("live_verifier_status") or "missing"),
        "verifier_claim_status": _verifier_claim_status(recovery, allowance),
        "repair_gate_status": _repair_gate_status(repair),
        "repair_ladder_status": _repair_ladder_status(repair),
        "repair_claim_status": _repair_claim_status(repair),
        "fr11_vera_evoenv_status": _fr11_vera_evoenv_status(fr11),
        "fr11_experience_memory_status": _fr11_experience_memory_status(fr11),
        "fr11_self_learning_status": _fr11_self_learning_status(fr11),
        "ebt_arm_status": _ebt_arm_status(architecture),
        "kan_status": _kan_status(architecture),
        "sampler_hardware_status": _sampler_hardware_status(architecture),
        "local_sota_cache_status": _local_sota_cache_status(allowance),
        "paper_readiness_assessment": _paper_readiness_assessment(
            capstone_ready, paper_ready, blocker_delta_from_v25
        ),
        "paper_readiness_checks": _paper_readiness_checks(
            capstone_ready, publication_blocker_count, blocked_headline_claims
        ),
        "matrix_v26_summary": _matrix_v26_summary(matrix),
        "what_292_proved": _what_292_proved(matrix, recovery, repair, fr11, architecture),
        "what_stayed_blocked": _what_stayed_blocked(allowance, recovery, repair, architecture),
        "bounded_claims": _bounded_claims(allowance, recovery, repair, fr11, architecture),
        "allowed_claims": _allowed_claims(allowance),
        "forbidden_claims": _forbidden_claims(allowance),
        "next_recommendation": _next_recommendation(next_top_gap),
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
    """Build and persist the Exp 3148 deliverable JSON."""

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
            "experiment_id": "exp3147",
            "path": MATRIX_V26_REL_PATH,
            "role": "matrix_v26_authority",
            "required": True,
            "ready_field": "matrix_v26_ready",
            "source_type": "json",
        }
    ]
    matrix_sources = [
        item for item in _list(matrix.get("source_artifacts")) if isinstance(item, Mapping)
    ]
    specs.extend(matrix_sources)
    return [_source_artifact_row(root, spec) for spec in specs]


def _source_artifact_row(root: Path, spec: Mapping[str, Any]) -> JsonDict:
    rel_path = Path(str(spec.get("path") or ""))
    path = root / rel_path
    payload = read_json_object(path)
    return {
        "experiment_id": str(spec.get("experiment_id") or payload.get("artifact") or rel_path.stem),
        "path": rel_path.as_posix(),
        "role": str(spec.get("role") or "source"),
        "required": spec.get("required") is True,
        "ready_field": str(spec.get("ready_field") or ""),
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "sha256": sha256_file(path),
        "source_type": str(spec.get("source_type") or "json"),
    }


def _invariant_violations(
    matrix: Mapping[str, Any],
    source_artifacts: list[Mapping[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if not matrix:
        violations.append("matrix_v26 authority is missing or malformed")
    if matrix and matrix.get("matrix_v26_ready") is not True:
        violations.append("matrix_v26_ready is not true")
    status_counts = _mapping(matrix.get("status_counts"))
    rows_total = _int(matrix.get("rows_total"))
    if status_counts and sum(_int(value) for value in status_counts.values()) != rows_total:
        violations.append("status_counts do not reconcile with rows_total")
    if len(_list(matrix.get("publication_blockers"))) != _int(
        matrix.get("publication_blocker_count")
    ):
        violations.append("publication_blocker_count does not match publication_blockers")
    if _list(matrix.get("required_source_errors")):
        violations.append("matrix_v26 reports required source errors")
    if _list(matrix.get("invariant_violations")):
        violations.append("matrix_v26 reports invariant violations")
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
        violations.append("matrix_v26 inference_substrate is not aggregation-only")
    if not source_artifacts:
        violations.append("source_artifacts list is empty")
    return violations


def _false_accept_recovery_status(recovery: Mapping[str, Any]) -> str:
    claim = str(recovery.get("recovery_claim_status") or "missing")
    false_accept = _number(_float(recovery.get("rerun_false_accept_rate")))
    if claim == "exact_safe_recovery_ready":
        return "clean_exact_safe_recovery_ready"
    if claim == "blocked_by_adversarial_corrigendum":
        return f"blocked_by_adversarial_corrigendum_false_accept_{false_accept}_known_rows_blocked"
    return f"{claim}_false_accept_{false_accept}"


def _verifier_claim_status(recovery: Mapping[str, Any], allowance: Mapping[str, Any]) -> str:
    status = str(recovery.get("live_verifier_status") or "missing")
    false_accept = _number(_float(recovery.get("rerun_false_accept_rate")))
    gain = _number(_float(recovery.get("rerun_verifier_gain_delta")))
    if status == "clean" and allowance.get("live_verifier_headline_allowed") is True:
        return f"clean_live_verifier_false_accept_{false_accept}_headline_allowed"
    if status == "flagged":
        return f"flagged_live_verifier_false_accept_{false_accept}_gain_{gain}_no_headline"
    return f"{status}_live_verifier_false_accept_{false_accept}_not_promoted"


def _repair_gate_status(repair: Mapping[str, Any]) -> str:
    status = str(repair.get("repair_gate_status") or "missing")
    state = str(repair.get("repair_gate_state") or "missing")
    if status == "clean" and state == "unblocked":
        return "clean_repair_gate_unblocked"
    blockers = _int(repair.get("repair_blocker_count"))
    disqualifiers = _int(repair.get("headline_disqualifier_count"))
    return f"{status}_repair_gate_state_{state}_blockers_{blockers}_disqualifiers_{disqualifiers}"


def _repair_ladder_status(repair: Mapping[str, Any]) -> str:
    status = str(repair.get("repair_ladder_status") or "missing")
    if status == "clean" and repair.get("repair_ladder_present") is True:
        return "clean_present"
    if status == "gated_skipped" and repair.get("repair_ladder_present") is not True:
        return "gated_skipped_missing_artifact"
    return status


def _repair_claim_status(repair: Mapping[str, Any]) -> str:
    ladder = str(repair.get("repair_ladder_status") or "missing")
    if repair.get("headline_repair_claim_allowed") is True and ladder == "clean":
        return "clean_repair_ladder_promotable"
    if not _list(repair.get("selected_repair_rows")):
        return f"blocked_repair_ladder_{ladder}_no_selected_rows"
    return f"{ladder}_repair_ladder_not_promoted"


def _fr11_vera_evoenv_status(fr11: Mapping[str, Any]) -> str:
    status = str(fr11.get("vera_evoenv_status") or "missing")
    ledger = _number(_float(fr11.get("vera_ledger_consistency_rate")))
    suffix = "model_weight_learning" if fr11.get("model_weight_learning_allowed") is True else "controller_only"
    return f"{status}_ledger_{ledger}_{suffix}"


def _fr11_experience_memory_status(fr11: Mapping[str, Any]) -> str:
    status = str(fr11.get("experience_memory_status") or "missing")
    ledger = _number(_float(fr11.get("experience_ledger_consistency_rate")))
    suffix = "model_weight_learning" if fr11.get("model_weight_learning_allowed") is True else "controller_only"
    return f"{status}_ledger_{ledger}_{suffix}"


def _fr11_self_learning_status(fr11: Mapping[str, Any]) -> str:
    if (
        fr11.get("model_weight_learning_allowed") is True
        and fr11.get("no_weight_update_claim") is not True
    ):
        return "clean_model_weight_learning_allowed"
    vera = _number(_float(fr11.get("vera_ledger_consistency_rate")))
    experience = _number(_float(fr11.get("experience_ledger_consistency_rate")))
    return f"bounded_controller_memory_only_no_weight_update_vera_{vera}_experience_{experience}"


def _ebt_arm_status(architecture: Mapping[str, Any]) -> str:
    status = str(architecture.get("ebt_arm_status") or "missing")
    if status == "clean" and architecture.get("live_integration") is True:
        return "clean_ebt_arm_live_integration"
    blockers = _int(architecture.get("integration_blocker_count"))
    return f"{status}_no_live_integration_blockers_{blockers}"


def _kan_status(architecture: Mapping[str, Any]) -> str:
    status = str(architecture.get("kan_monitor_status") or "missing")
    if architecture.get("deployed_kan_verifier_claim") is True:
        return "clean_deployed_kan_verifier_claim"
    records = _int(architecture.get("kan_attached_monitor_record_count"))
    return f"{status}_monitor_records_{records}_no_deployed_verifier"


def _sampler_hardware_status(architecture: Mapping[str, Any]) -> str:
    commands = _list(architecture.get("hardware_commands_run"))
    if architecture.get("speedup_claim_allowed") is True and commands:
        return "clean_authenticated_sampler_hardware_speedup"
    missing = _int(architecture.get("missing_operator_evidence_count"))
    return f"blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_{missing}"


def _local_sota_cache_status(allowance: Mapping[str, Any]) -> str:
    missing = _text_list(allowance.get("missing_model_ids"))
    if allowance.get("comparative_sota_pair_allowed") is True:
        return "clean_comparative_sota_cache_ready"
    if _text_list(allowance.get("present_model_ids")):
        return f"bounded_single_cached_model_comparative_pair_missing_{len(missing)}"
    return "blocked_no_local_sota_cache"


def _paper_readiness_assessment(
    capstone_ready: bool, paper_ready: bool, blocker_delta_from_v25: int
) -> str:
    if not capstone_ready:
        return "blocked_precondition"
    if paper_ready:
        return "paper_ready_blockers_cleared"
    if blocker_delta_from_v25 > 0:
        return f"not_closer_publication_blockers_increased_by_{blocker_delta_from_v25}"
    if blocker_delta_from_v25 < 0:
        return f"closer_blockers_reduced_by_{abs(blocker_delta_from_v25)}_but_not_ready"
    return "not_ready_publication_blockers_unchanged"


def _paper_readiness_checks(
    capstone_ready: bool, publication_blocker_count: int, blocked_headline_claims: list[str]
) -> list[JsonDict]:
    return [
        {
            "check": "capstone_ready",
            "passed": capstone_ready,
            "reason": "matrix v26 authority loaded and invariant checks passed",
        },
        {
            "check": "publication_blocker_count_zero",
            "passed": publication_blocker_count == 0,
            "reason": f"publication_blocker_count={publication_blocker_count}",
        },
        {
            "check": "blocked_headline_claims_empty",
            "passed": not blocked_headline_claims,
            "reason": ",".join(blocked_headline_claims) or "no blocked headline claims",
        },
    ]


def _matrix_v26_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_v26_ready": matrix.get("matrix_v26_ready") is True,
        "rows_total": _int(matrix.get("rows_total")),
        "prior_publication_blocker_count": _int(matrix.get("prior_publication_blocker_count")),
        "publication_blocker_count": _int(matrix.get("publication_blocker_count")),
        "blocker_delta_from_v25": _int(matrix.get("blocker_delta_from_v25")),
        "status_counts": _mapping(matrix.get("status_counts")),
        "missing_artifact_count": len(_list(matrix.get("missing_artifacts"))),
    }


def _what_292_proved(
    matrix: Mapping[str, Any],
    recovery: Mapping[str, Any],
    repair: Mapping[str, Any],
    fr11: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> list[str]:
    return [
        (
            "matrix v26 is complete over "
            f"{_int(matrix.get('rows_total'))} rows with "
            f"{_int(matrix.get('publication_blocker_count'))} publication blockers."
        ),
        (
            "Exact accept/abstain and canonical grounding replay blocked "
            f"{_int(recovery.get('canonical_false_accept_rows_blocked'))} known false-accept rows."
        ),
        (
            "The live verifier rerun source records false_accept_rate="
            f"{_number(_float(recovery.get('rerun_false_accept_rate')))} but remains "
            f"{recovery.get('live_verifier_status')}."
        ),
        (
            "Repair gate state is "
            f"{repair.get('repair_gate_state')} with repair_ladder_status="
            f"{repair.get('repair_ladder_status')}."
        ),
        (
            "FR-11 stayed controller-only: VeRA ledger="
            f"{_number(_float(fr11.get('vera_ledger_consistency_rate')))}, experience ledger="
            f"{_number(_float(fr11.get('experience_ledger_consistency_rate')))}."
        ),
        (
            "Architecture rows remain bounded: EBT/ARM="
            f"{architecture.get('ebt_arm_status')}, KAN={architecture.get('kan_monitor_status')}, "
            f"hardware={architecture.get('hardware_boundary_status')}."
        ),
    ]


def _what_stayed_blocked(
    allowance: Mapping[str, Any],
    recovery: Mapping[str, Any],
    repair: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> list[str]:
    blocked = _text_list(allowance.get("blocked_headline_claims"))
    if recovery.get("recovery_claim_status") == "blocked_by_adversarial_corrigendum":
        blocked.append("false_accept_recovery_corrigendum")
    if repair.get("repair_ladder_present") is not True:
        blocked.append("repair_ladder_execution_missing")
    if architecture.get("speedup_claim_allowed") is not True:
        blocked.append("authenticated_sampler_hardware_speedup")
    return _dedupe(blocked)


def _bounded_claims(
    allowance: Mapping[str, Any],
    recovery: Mapping[str, Any],
    repair: Mapping[str, Any],
    fr11: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> list[str]:
    claims: list[str] = []
    if allowance.get("comparative_sota_pair_allowed") is not True:
        claims.append("single_cached_model_not_comparative_sota_pair")
    if recovery.get("live_verifier_status") != "clean":
        claims.append("live_verifier_rerun_not_headline_until_flags_clear")
    if repair.get("headline_repair_claim_allowed") is not True:
        claims.append("repair_promotion_blocked_until_ladder_executes")
    if fr11.get("no_weight_update_claim") is True:
        claims.append("fr11_controller_memory_only_no_model_weight_learning")
    if architecture.get("live_integration") is not True:
        claims.append("ebt_arm_sidecar_calibration_no_live_integration")
    if architecture.get("deployed_kan_verifier_claim") is not True:
        claims.append("kan_replay_records_no_deployed_verifier")
    if architecture.get("speedup_claim_allowed") is not True:
        claims.append("sampler_hardware_no_authenticated_speedup")
    return claims


def _allowed_claims(allowance: Mapping[str, Any]) -> list[str]:
    claims = ["matrix_v26_aggregation_complete"]
    if allowance.get("exact_safe_contract_claim_allowed") is True:
        claims.append("exact_safe_accept_abstain_contract_replay")
    if allowance.get("canonical_grounding_claim_allowed") is True:
        claims.append("canonical_grounding_blocks_known_false_accept_rows")
    if allowance.get("sota_cache_headline_allowed") is True:
        claims.append("single_cached_gemma26_available")
    return claims


def _forbidden_claims(allowance: Mapping[str, Any]) -> list[str]:
    forbidden: list[str] = []
    if allowance.get("live_verifier_headline_allowed") is not True:
        forbidden.append("live_verifier_headline_lift")
    if allowance.get("repair_headline_claim_allowed") is not True:
        forbidden.append("repair_promotion")
    if allowance.get("fr11_model_weight_learning_allowed") is not True:
        forbidden.append("fr11_model_weight_learning")
    if allowance.get("ebt_arm_live_integration_allowed") is not True:
        forbidden.append("ebt_arm_live_integration")
    if allowance.get("kan_deployed_verifier_allowed") is not True:
        forbidden.append("kan_deployed_verifier")
    if allowance.get("hardware_speedup_claim_allowed") is not True:
        forbidden.append("hardware_speedup")
    return forbidden


def _next_top_gap(
    allowance: Mapping[str, Any],
    recovery: Mapping[str, Any],
    repair: Mapping[str, Any],
    fr11: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> str:
    if (
        recovery.get("recovery_claim_status") != "exact_safe_recovery_ready"
        or recovery.get("live_verifier_status") != "clean"
        or repair.get("repair_gate_status") != "clean"
        or repair.get("repair_ladder_status") != "clean"
    ):
        return "false_accept_recovery_corrigendum_repair_gate"
    if allowance.get("comparative_sota_pair_allowed") is not True:
        return "comparative_sota_cache_pair"
    if fr11.get("model_weight_learning_allowed") is not True:
        return "fr11_model_weight_learning"
    if architecture.get("speedup_claim_allowed") is not True:
        return "authenticated_sampler_hardware"
    return "publication_scope_reconciliation"


def _next_recommendation(next_top_gap: str) -> str:
    if next_top_gap == "false_accept_recovery_corrigendum_repair_gate":
        return (
            "Next milestone should clear the false-accept recovery corrigendum and live-verifier "
            "authenticity blockers, then execute the repair ladder only after the repair gate opens. "
            "That path is higher leverage than adding more architecture sidecars because matrix v26 "
            "still blocks live verifier headline and repair promotion claims."
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
        "kind": "aggregation_from_checked_in_matrix_v26_and_dot292_artifacts",
        "source": MATRIX_V26_REL_PATH.as_posix(),
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
        f"blocker_delta_from_v25={artifact.get('blocker_delta_from_v25')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text_list(value: Any) -> list[str]:
    return [str(item) for item in _list(value)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


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
