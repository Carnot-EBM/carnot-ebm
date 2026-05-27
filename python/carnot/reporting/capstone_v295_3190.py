"""Build the Exp 3190 milestone .295 capstone artifact.

Spec refs: REQ-REPORT-3190, SCENARIO-REPORT-3190.

This closeout reads matrix v29 and the checked-in `.295` source artifacts,
then records what the milestone proved and what it did not prove. It is
intentionally aggregation-only: a capstone should make the evidence ledger
auditable, not create new live-model, repair, solver, or hardware evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.295"
SCHEMA = "carnot.milestone_capstone.v295_matrix_v29_aggregation.v1"
ARTIFACT = "experiment_3190_capstone_v295"
OUTPUT_REL_PATH = Path("results/experiment_3190_capstone_v295.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3190_capstone_v295.py"

MATRIX_V29_REL_PATH = Path("results/experiment_3189_cross_corpus_matrix_v29.json")
CAPSTONE_V294_REL_PATH = Path("results/experiment_3176_capstone_v294.json")
EXP3178_REL_PATH = Path("results/experiment_3178_receipt_backed_authenticity_contract_v3.json")
EXP3179_REL_PATH = Path("results/experiment_3179_local_sota_receipt_smoke_v3.json")
EXP3180_REL_PATH = Path("results/experiment_3180_controlled_invariance_executor_v2.json")
EXP3181_REL_PATH = Path("results/experiment_3181_clean_live_sota_verifier_rerun_v10.json")
EXP3182_REL_PATH = Path("results/experiment_3182_distributional_ebm_exact_row_sidecar_v1.json")
EXP3183_REL_PATH = Path("results/experiment_3183_counterexample_certificate_expansion_v3.json")
EXP3184_REL_PATH = Path("results/experiment_3184_repair_gate_decision_v4.json")
EXP3185_REL_PATH = Path("results/experiment_3185_multi_turn_repair_ladder_v5.json")
EXP3186_REL_PATH = Path("results/experiment_3186_fr11_controller_memory_promotion_pack_v1.json")
EXP3187_REL_PATH = Path("results/experiment_3187_fr11_cross_environment_drift_replay_v1.json")
EXP3188_REL_PATH = Path("results/experiment_3188_thrml_factor_graph_api_boundary_v1.json")


@dataclass(frozen=True)
class SourceSpec:
    """A checked-in source JSON that the capstone must account for explicitly."""

    experiment_id: str
    path: Path
    role: str
    ready_field: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp3189", MATRIX_V29_REL_PATH, "matrix_v29_authority", "cross_corpus_matrix_v29_ready"
    ),
    SourceSpec("exp3176", CAPSTONE_V294_REL_PATH, "capstone_v294_authority", "capstone_v294_ready"),
    SourceSpec(
        "exp3178",
        EXP3178_REL_PATH,
        "receipt_backed_authenticity_contract",
        "receipt_backed_authenticity_contract_v3_ready",
    ),
    SourceSpec(
        "exp3179", EXP3179_REL_PATH, "local_sota_receipt_smoke", "local_sota_receipt_smoke_v3_ready"
    ),
    SourceSpec(
        "exp3180",
        EXP3180_REL_PATH,
        "controlled_invariance_executor",
        "controlled_invariance_executor_v2_ready",
    ),
    SourceSpec(
        "exp3181",
        EXP3181_REL_PATH,
        "clean_live_sota_verifier_rerun",
        "clean_live_sota_verifier_rerun_v10_ready",
    ),
    SourceSpec(
        "exp3182",
        EXP3182_REL_PATH,
        "distributional_ebm_exact_row_sidecar",
        "distributional_ebm_exact_row_sidecar_v1_ready",
    ),
    SourceSpec(
        "exp3183",
        EXP3183_REL_PATH,
        "counterexample_certificate_expansion",
        "counterexample_certificate_expansion_v3_ready",
    ),
    SourceSpec("exp3184", EXP3184_REL_PATH, "repair_gate_v4", "repair_gate_decision_v4_ready"),
    SourceSpec(
        "exp3185", EXP3185_REL_PATH, "repair_ladder_v5", "multi_turn_repair_ladder_v5_ready"
    ),
    SourceSpec(
        "exp3186",
        EXP3186_REL_PATH,
        "fr11_controller_memory_promotion_pack",
        "fr11_controller_memory_promotion_pack_v1_ready",
    ),
    SourceSpec(
        "exp3187",
        EXP3187_REL_PATH,
        "fr11_cross_environment_drift_replay",
        "fr11_cross_environment_drift_replay_v1_ready",
    ),
    SourceSpec(
        "exp3188", EXP3188_REL_PATH, "thrml_boundary", "thrml_factor_graph_api_boundary_v1_ready"
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence for absent or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Checksum a source artifact so downstream reviewers can reproduce inputs."""

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
    """REQ-REPORT-3190: close .295 from matrix v29 without fresh execution."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = {
        spec.experiment_id: read_json_object(root_path / spec.path) for spec in SOURCE_SPECS
    }
    matrix = payloads["exp3189"]
    capstone_v294 = payloads["exp3176"]
    receipt = payloads["exp3179"]
    invariance = payloads["exp3180"]
    clean_rerun = payloads["exp3181"]
    sidecar = payloads["exp3182"]
    repair_gate = payloads["exp3184"]
    repair_ladder = payloads["exp3185"]
    fr11_promotion = payloads["exp3186"]
    fr11_drift = payloads["exp3187"]
    thrml = payloads["exp3188"]

    publication_blocker_count = _int(matrix.get("publication_blocker_count"))
    blocker_delta_from_v28 = _int(matrix.get("blocker_delta_from_v28"))
    missing_artifact_count = len(_list(matrix.get("missing_artifacts")))
    local_sota_status = _local_sota_receipt_status(receipt)
    controlled_invariance_status = _controlled_invariance_status(invariance)
    repair_gate_status = _repair_gate_status(repair_gate)
    repair_ladder_status = _repair_ladder_status(repair_ladder)
    fr11_replay_passed = _fr11_promotion_drift_replay_passed(fr11_promotion, fr11_drift)
    thrml_boundary_status = _thrml_boundary_status(thrml)
    verifier_status = str(matrix.get("verifier_status") or "missing_verifier_status")
    sidecar_status = str(matrix.get("sidecar_status") or "missing_sidecar_status")
    hardware_sampler_status = str(matrix.get("hardware_status") or "missing_hardware_status")
    narrowing_preserved = _paper_v6_narrowing_preserved(matrix)
    matrix_paper_ready = matrix.get("paper_ready") is True
    headline_gates_clean = (
        local_sota_status == "passed_receipt_clean_rerun_allowed"
        and verifier_status == "clean_live_verifier_ready"
        and repair_gate_status == "clean_repair_gate_unblocked"
        and repair_ladder_status == "clean_repair_ladder_materialized"
        and _sidecar_deployed(sidecar, sidecar_status)
        and _hardware_claim_clean(hardware_sampler_status)
    )
    paper_ready = matrix_paper_ready and publication_blocker_count == 0 and headline_gates_clean
    invariant_violations = _invariant_violations(
        payloads,
        matrix,
        capstone_v294,
        publication_blocker_count,
        blocker_delta_from_v28,
        missing_artifact_count,
        narrowing_preserved,
    )
    capstone_ready = not invariant_violations
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_v295_ready": capstone_ready,
        "capstone_ready": capstone_ready,
        "matrix_authority": MATRIX_V29_REL_PATH.as_posix(),
        "paper_ready": capstone_ready and paper_ready,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v28": blocker_delta_from_v28,
        "missing_artifact_count": missing_artifact_count,
        "local_sota_receipt_status": local_sota_status,
        "controlled_invariance_status": controlled_invariance_status,
        "verifier_status": verifier_status,
        "clean_verifier_status": _clean_verifier_status(clean_rerun, verifier_status),
        "repair_gate_status": repair_gate_status,
        "repair_ladder_status": repair_ladder_status,
        "fr11_self_learning_status": str(matrix.get("fr11_status") or "missing_fr11_status"),
        "fr11_promotion_drift_replay_passed": fr11_replay_passed,
        "sidecar_status": sidecar_status,
        "distributional_sidecar_deployed": _sidecar_deployed(sidecar, sidecar_status),
        "hardware_sampler_status": hardware_sampler_status,
        "thrml_boundary_status": thrml_boundary_status,
        "ops_docs_updated": False,
        "ops_reconciliation_decision": _ops_reconciliation_decision(),
        "next_top_gap": str(
            matrix.get("next_top_gap") or _next_top_gap(local_sota_status, repair_gate_status)
        ),
        "paper_v6_narrowing_preserved": narrowing_preserved,
        "phase_outcome_summary": _phase_outcome_summary(
            local_sota_status,
            controlled_invariance_status,
            verifier_status,
            repair_gate_status,
            repair_ladder_status,
            sidecar,
            sidecar_status,
            fr11_replay_passed,
            thrml_boundary_status,
        ),
        "matrix_summary": _matrix_summary(matrix, capstone_v294),
        "source_artifacts": _source_artifacts(root_path, payloads),
        "source_checksums": {
            row["path"]: row.get("sha256") for row in _source_artifacts(root_path, payloads)
        },
        "invariant_violations": invariant_violations,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_repair_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "scripts_research_conductor_modified": False,
        "research_roadmap_modified": False,
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
    """Build and persist the Exp 3190 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _local_sota_receipt_status(receipt: Mapping[str, Any]) -> str:
    if not receipt:
        return "missing_local_sota_receipt"
    if receipt.get("local_sota_receipt_smoke_v3_ready") is not True:
        return "blocked_local_sota_receipt_not_ready"
    if receipt.get("clean_rerun_allowed") is True and receipt.get("headline_claim_allowed") is True:
        return "passed_receipt_clean_rerun_allowed"
    if str(receipt.get("substrate_classification") or "") == "cpu_fallback_receipt_only":
        return "cpu_fallback_receipt_only_non_headline_clean_rerun_blocked"
    return "blocked_local_sota_receipt_not_headline_eligible"


def _controlled_invariance_status(invariance: Mapping[str, Any]) -> str:
    if not invariance:
        return "missing_controlled_invariance"
    if invariance.get("controlled_invariance_executor_v2_ready") is not True:
        return "blocked_controlled_invariance_not_ready"
    if (
        invariance.get("controlled_invariance_passed") is not True
        or _int(invariance.get("exact_row_count")) <= 0
    ):
        return "blocked_controlled_invariance_not_passed"
    if invariance.get("flagged_adversarial") is True:
        return "passed_controlled_invariance_exact_authority_receipts_flagged"
    return "passed_controlled_invariance_exact_authority_receipts_clean"


def _clean_verifier_status(clean_rerun: Mapping[str, Any], verifier_status: str) -> str:
    if not clean_rerun:
        return "missing_clean_verifier_rerun"
    if clean_rerun.get("clean_live_sota_verifier_rerun_v10_ready") is not True:
        return "blocked_clean_verifier_rerun_not_ready"
    if (
        verifier_status == "clean_live_verifier_ready"
        and clean_rerun.get("headline_claim_allowed") is True
        and _int(clean_rerun.get("live_call_count")) > 0
    ):
        return "clean_live_verifier_ready"
    if clean_rerun.get("gated_skip") is True:
        return "gated_skip_receipt_precondition_no_clean_live_verifier"
    return "blocked_clean_verifier_not_headline_eligible"


def _repair_gate_status(repair_gate: Mapping[str, Any]) -> str:
    if not repair_gate:
        return "missing_repair_gate_decision"
    if repair_gate.get("repair_gate_decision_v4_ready") is not True:
        return "blocked_repair_gate_decision_not_ready"
    state = str(repair_gate.get("repair_gate_state") or "missing_state")
    if state == "unblocked":
        return "clean_repair_gate_unblocked"
    return state


def _repair_ladder_status(repair_ladder: Mapping[str, Any]) -> str:
    if not repair_ladder:
        return "missing_repair_ladder"
    if repair_ladder.get("multi_turn_repair_ladder_v5_ready") is not True:
        return "blocked_repair_ladder_not_ready"
    if (
        repair_ladder.get("headline_claim_allowed") is True
        and _int(repair_ladder.get("repair_attempt_count")) > 0
    ):
        return "clean_repair_ladder_materialized"
    if (
        repair_ladder.get("gated_skip") is True
        and _int(repair_ladder.get("repair_attempt_count")) == 0
    ):
        return "materialized_gated_skip_repair_gate_blocked_no_live_repair_attempts"
    return "blocked_repair_ladder_not_promotable"


def _fr11_promotion_drift_replay_passed(
    promotion: Mapping[str, Any], drift: Mapping[str, Any]
) -> bool:
    return (
        promotion.get("fr11_controller_memory_promotion_pack_v1_ready") is True
        and promotion.get("promotion_allowed") is True
        and _no_weight_update_claimed(promotion)
        and drift.get("fr11_cross_environment_drift_replay_v1_ready") is True
        and drift.get("promotion_allowed") is True
        and drift.get("rollback_triggered") is not True
        and _int(drift.get("negative_control_regression_count")) == 0
        and _no_weight_update_claimed(drift)
    )


def _no_weight_update_claimed(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("no_model_weight_update_claimed") is True
        or payload.get("model_weight_update_claimed") is False
    )


def _sidecar_deployed(sidecar: Mapping[str, Any], sidecar_status: str) -> bool:
    return (
        sidecar.get("distributional_ebm_exact_row_sidecar_v1_ready") is True
        and sidecar.get("deployed_verifier_claim_allowed") is True
        and not sidecar_status.startswith("diagnostic_only")
    )


def _hardware_claim_clean(hardware_sampler_status: str) -> bool:
    return hardware_sampler_status.startswith("clean_")


def _thrml_boundary_status(thrml: Mapping[str, Any]) -> str:
    if not thrml:
        return "missing_thrml_boundary"
    if thrml.get("thrml_factor_graph_api_boundary_v1_ready") is not True:
        return "blocked_thrml_boundary_not_ready"
    if thrml.get("local_api_smoke_passed") is not True:
        return "blocked_thrml_boundary_local_api_smoke_missing"
    if (
        thrml.get("hardware_speedup_claim_allowed") is True
        and thrml.get("kona_or_tsu_execution_claimed") is not True
    ):
        return "local_api_smoke_with_bounded_speedup_permission"
    if (
        thrml.get("hardware_speedup_claim_allowed") is not True
        and thrml.get("kona_or_tsu_execution_claimed") is not True
    ):
        return "local_api_smoke_only_no_speedup_no_tsu_kona_execution"
    return "blocked_thrml_boundary_overclaimed_execution"


def _paper_v6_narrowing_preserved(matrix: Mapping[str, Any]) -> bool:
    narrowing = _mapping(matrix.get("paper_v6_narrowing"))
    forbidden_claims = (
        "kv260_speedup_claimed",
        "tsu_or_kona_execution_claimed",
        "deployed_verifier_sidecar_claimed",
        "model_weight_self_learning_claimed",
        "paper_ready_streak_claimed",
    )
    return matrix.get("paper_v6_narrowing_preserved") is True and not any(
        narrowing.get(field) is True for field in forbidden_claims
    )


def _phase_receipt_outcome(local_sota_status: str) -> str:
    if local_sota_status == "passed_receipt_clean_rerun_allowed":
        return "passed"
    if local_sota_status == "cpu_fallback_receipt_only_non_headline_clean_rerun_blocked":
        return "cpu_only_non_headline_evidence"
    return "blocked"


def _phase_outcome_summary(
    local_sota_status: str,
    controlled_invariance_status: str,
    verifier_status: str,
    repair_gate_status: str,
    repair_ladder_status: str,
    sidecar: Mapping[str, Any],
    sidecar_status: str,
    fr11_replay_passed: bool,
    thrml_boundary_status: str,
) -> JsonDict:
    return {
        "receipt_backed_local_sota_path": _phase_receipt_outcome(local_sota_status),
        "controlled_invariance_passed": controlled_invariance_status.startswith("passed_"),
        "clean_verifier_unblocked": verifier_status == "clean_live_verifier_ready",
        "repair_gate_unblocked": repair_gate_status == "clean_repair_gate_unblocked",
        "repair_ladder_executed": repair_ladder_status == "clean_repair_ladder_materialized",
        "distributional_sidecar_deployed": _sidecar_deployed(sidecar, sidecar_status),
        "fr11_controller_memory_promoted_without_weight_update": fr11_replay_passed,
        "thrml_boundary_local_api_only": thrml_boundary_status
        == "local_api_smoke_only_no_speedup_no_tsu_kona_execution",
    }


def _matrix_summary(matrix: Mapping[str, Any], capstone_v294: Mapping[str, Any]) -> JsonDict:
    return {
        "prior_matrix_version": str(matrix.get("prior_matrix_version") or "missing"),
        "prior_publication_blocker_count": _int(matrix.get("prior_publication_blocker_count")),
        "v29_publication_blocker_count": _int(matrix.get("publication_blocker_count")),
        "capstone_v294_publication_blocker_count": _int(
            capstone_v294.get("publication_blocker_count")
        ),
        "blocker_delta_from_v28": _int(matrix.get("blocker_delta_from_v28")),
        "status_counts": _mapping(matrix.get("status_counts")),
    }


def _source_artifacts(root: Path, payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        payload = payloads.get(spec.experiment_id) or {}
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "role": spec.role,
                "ready_field": spec.ready_field,
                "present": (root / spec.path).is_file(),
                "readable_json_object": bool(payload),
                "ready": payload.get(spec.ready_field) is True,
                "sha256": sha256_file(root / spec.path),
                "source_type": "json",
            }
        )
    return rows


def _invariant_violations(
    payloads: Mapping[str, Mapping[str, Any]],
    matrix: Mapping[str, Any],
    capstone_v294: Mapping[str, Any],
    publication_blocker_count: int,
    blocker_delta_from_v28: int,
    missing_artifact_count: int,
    narrowing_preserved: bool,
) -> list[str]:
    checks: list[tuple[bool, str]] = [
        (not matrix, "matrix_v29 authority is missing or malformed"),
        (
            bool(matrix) and matrix.get("cross_corpus_matrix_v29_ready") is not True,
            "matrix_v29 authority is not ready",
        ),
        (not capstone_v294, "capstone_v294 authority is missing or malformed"),
        (
            bool(capstone_v294) and capstone_v294.get("capstone_v294_ready") is not True,
            "capstone_v294 authority is not ready",
        ),
        (
            bool(matrix)
            and publication_blocker_count
            != _int(matrix.get("prior_publication_blocker_count")) + blocker_delta_from_v28,
            "matrix_v29 blocker delta does not reconcile",
        ),
        (
            bool(matrix)
            and missing_artifact_count
            != _int(
                _mapping(matrix.get("missing_artifact_comparison")).get(
                    "v29_missing_artifact_count"
                )
            ),
            "matrix_v29 missing artifact count does not reconcile",
        ),
        (
            bool(matrix) and bool(_list(matrix.get("required_source_errors"))),
            "matrix_v29 reports required source errors",
        ),
        (
            bool(matrix) and bool(_list(matrix.get("invariant_violations"))),
            "matrix_v29 reports invariant violations",
        ),
        (
            bool(matrix) and _substrate_runs_execution(_mapping(matrix.get("inference_substrate"))),
            "matrix_v29 inference_substrate is not aggregation-only",
        ),
        (
            bool(matrix) and matrix.get("paper_ready") is True and publication_blocker_count > 0,
            "matrix_v29 paper_ready cannot coexist with publication blockers",
        ),
        (bool(matrix) and not narrowing_preserved, "paper-v6 narrowing is not preserved"),
    ]
    for spec in SOURCE_SPECS:
        payload = payloads.get(spec.experiment_id) or {}
        checks.append((not payload, f"{spec.role} source is missing or malformed"))
        checks.append(
            (
                bool(payload) and payload.get(spec.ready_field) is not True,
                f"{spec.role} source is not ready",
            )
        )
    return [message for failed, message in checks if failed]


def _substrate_runs_execution(substrate: Mapping[str, Any]) -> bool:
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


def _inference_substrate() -> JsonDict:
    return {
        "kind": "capstone_aggregation_from_checked_in_matrix_v29_and_dot295_artifacts",
        "source": "matrix_v29_capstone_v294_and_dot295_phase_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def _ops_reconciliation_decision() -> JsonDict:
    return {
        "delegated_to_conductor": True,
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "traceability_updated_after_spec": False,
        "reason": "task stop rule delegates ops/status/changelog/traceability reconciliation to conductor",
    }


def _next_top_gap(local_sota_status: str, repair_gate_status: str) -> str:
    if local_sota_status != "passed_receipt_clean_rerun_allowed":
        return "full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
    if repair_gate_status != "clean_repair_gate_unblocked":
        return "repair_gate_unblock"
    return "publication_scope_reconciliation"


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("capstone_v295_ready") is not True:
        first = str(_list(artifact.get("invariant_violations"))[0])
        return f"blocked: capstone_v295_ready=false; {first}"
    return (
        "complete: capstone_v295_ready=true; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v28={artifact.get('blocker_delta_from_v28')}; "
        f"missing_artifact_count={artifact.get('missing_artifact_count')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
