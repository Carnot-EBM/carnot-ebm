"""Build the Exp 3176 milestone .294 capstone artifact.

Spec refs: REQ-REPORT-3176, SCENARIO-REPORT-3176.

This is a closeout aggregator. It reads checked-in matrix and phase artifacts,
then states what `.294` did and did not clear. The capstone deliberately avoids
model calls, verifier scoring, repairs, solvers, hardware commands, conductor
execution, or roadmap mutation because the closeout job is evidence accounting,
not fresh research execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.294"
SCHEMA = "carnot.milestone_capstone.v294_matrix_v28_aggregation.v1"
ARTIFACT = "experiment_3176_capstone_v294"
OUTPUT_REL_PATH = Path("results/experiment_3176_capstone_v294.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3176_capstone_v294.py"

MATRIX_V28_REL_PATH = Path("results/experiment_3175_cross_corpus_matrix_v28.json")
MATRIX_V27_REL_PATH = Path("results/experiment_3161_cross_corpus_matrix_v27.json")
CAPSTONE_V293_REL_PATH = Path("results/experiment_3162_capstone_v293.json")
EXP3164_REL_PATH = Path("results/experiment_3164_duration_corrected_authenticity_contract_v2.json")
EXP3167_REL_PATH = Path("results/experiment_3167_clean_live_sota_verifier_rerun_v9.json")
EXP3168_REL_PATH = Path("results/experiment_3168_repair_gate_decision_v3.json")
EXP3169_REL_PATH = Path("results/experiment_3169_repair_ladder_materializer_v4.json")
EXP3172_REL_PATH = Path("results/experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.json")
EXP3173_REL_PATH = Path("results/experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.json")
EXP3174_REL_PATH = Path("results/experiment_3174_hardware_tooling_boundary_v8.json")

SOURCE_SPECS: tuple[tuple[str, Path, str, str], ...] = (
    ("exp3175", MATRIX_V28_REL_PATH, "matrix_v28_authority", "matrix_v28_ready"),
    ("exp3161", MATRIX_V27_REL_PATH, "matrix_v27_authority", "matrix_v27_ready"),
    ("exp3162", CAPSTONE_V293_REL_PATH, "capstone_v293_authority", "capstone_ready"),
    (
        "exp3164",
        EXP3164_REL_PATH,
        "duration_corrected_authenticity_contract",
        "duration_corrected_authenticity_contract_v2_ready",
    ),
    (
        "exp3167",
        EXP3167_REL_PATH,
        "clean_live_verifier_rerun",
        "clean_live_verifier_rerun_v9_ready",
    ),
    ("exp3168", EXP3168_REL_PATH, "repair_gate_v3", "repair_gate_decision_v3_ready"),
    ("exp3169", EXP3169_REL_PATH, "repair_ladder_v4", "repair_ladder_materializer_v4_ready"),
    (
        "exp3172",
        EXP3172_REL_PATH,
        "fr11_nonforgetting_self_learning",
        "fr11_nonforgetting_self_learning_pilot_v2_ready",
    ),
    (
        "exp3173",
        EXP3173_REL_PATH,
        "ebcn_kan_bounded_diagnostics",
        "ebcn_kan_bounded_diagnostic_expansion_v2_ready",
    ),
    (
        "exp3174",
        EXP3174_REL_PATH,
        "hardware_tooling_boundary",
        "hardware_tooling_boundary_v8_ready",
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence for missing or invalid files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a source checksum so capstone consumers can trace exact inputs."""

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
    """REQ-REPORT-3176: close .294 from matrix v28 without new execution."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = {
        experiment_id: read_json_object(root_path / rel_path)
        for experiment_id, rel_path, _, _ in SOURCE_SPECS
    }
    matrix_v28 = payloads["exp3175"]
    matrix_v27 = payloads["exp3161"]
    capstone_v293 = payloads["exp3162"]
    contract = payloads["exp3164"]
    verifier = payloads["exp3167"]
    repair_gate = payloads["exp3168"]
    repair_ladder = payloads["exp3169"]
    fr11 = payloads["exp3172"]
    sidecar = payloads["exp3173"]
    hardware = payloads["exp3174"]

    duration_status = _duration_corrected_authenticity_status(contract)
    clean_verifier = _clean_live_verifier_evidence_exists(matrix_v28, verifier)
    repair_gate_status = _repair_gate_status(repair_gate)
    repair_ladder_status = _repair_ladder_status(repair_ladder)
    fr11_consistency = _fr11_promotion_grade_consistency(fr11)
    sidecar_clean = _sidecar_headline_clean(sidecar)
    hardware_clean = _hardware_headline_clean(hardware)
    publication_blocker_count = _int(matrix_v28.get("publication_blocker_count"))
    blocker_delta_from_v27 = _int(matrix_v28.get("blocker_delta_from_v27"))
    missing_artifact_count = len(_list(matrix_v28.get("missing_artifacts")))
    derived_paper_ready = (
        publication_blocker_count == 0
        and clean_verifier
        and repair_gate_status == "clean_repair_gate_unblocked"
        and repair_ladder_status == "clean_repair_ladder_materialized"
        and sidecar_clean
        and hardware_clean
    )
    invariant_violations = _invariant_violations(
        matrix_v28,
        matrix_v27,
        capstone_v293,
        derived_paper_ready,
        missing_artifact_count,
    )
    capstone_ready = not invariant_violations
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_v294_ready": capstone_ready,
        "capstone_ready": capstone_ready,
        "paper_ready": capstone_ready and derived_paper_ready,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v27": blocker_delta_from_v27,
        "missing_artifact_count": missing_artifact_count,
        "duration_corrected_authenticity_status": duration_status,
        "clean_live_verifier_evidence_exists": clean_verifier,
        "verifier_status": str(matrix_v28.get("verifier_status") or "missing_verifier_status"),
        "repair_gate_status": repair_gate_status,
        "repair_ladder_status": repair_ladder_status,
        "fr11_self_learning_status": str(matrix_v28.get("fr11_status") or "missing_fr11_status"),
        "fr11_promotion_grade_consistency": fr11_consistency,
        "fr11_model_weight_update_claimed": fr11.get("model_weight_update_claimed") is True,
        "ebcn_kan_status": str(matrix_v28.get("sidecar_status") or "missing_ebcn_kan_status"),
        "hardware_sampler_status": str(
            matrix_v28.get("hardware_status") or "missing_hardware_status"
        ),
        "hardware_claims_blocked": not hardware_clean,
        "next_top_gap": _next_top_gap(
            duration_status,
            clean_verifier,
            repair_gate_status,
            repair_ladder_status,
            sidecar_clean,
            hardware_clean,
            publication_blocker_count,
        ),
        "matrix_comparison": _matrix_comparison(matrix_v28, matrix_v27, capstone_v293),
        "phase_outcome_summary": _phase_outcome_summary(
            duration_status,
            clean_verifier,
            repair_gate_status,
            repair_ladder_status,
            fr11_consistency,
            sidecar_clean,
            hardware_clean,
        ),
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
        "ops_reconciliation_decision": _ops_reconciliation_decision(),
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
    """Build and persist the Exp 3176 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _duration_corrected_authenticity_status(contract: Mapping[str, Any]) -> str:
    if not contract:
        return "missing_duration_contract"
    if contract.get("duration_corrected_authenticity_contract_v2_ready") is not True:
        return "blocked_duration_contract_not_ready"
    observed = _mapping(contract.get("observed_source_assessment"))
    if (
        contract.get("old_fixed_duration_rule_retired_as_hard_gate") is True
        and observed.get("passed") is True
    ):
        return "passed_duration_corrected_contract_old_fixed_floor_retired"
    return "blocked_duration_contract_measurement_violations"


def _clean_live_verifier_evidence_exists(
    matrix_v28: Mapping[str, Any], verifier: Mapping[str, Any]
) -> bool:
    return (
        matrix_v28.get("verifier_status") == "clean_live_verifier_ready"
        and verifier.get("clean_live_verifier_rerun_v9_ready") is True
        and verifier.get("controlled_invariance_passed") is True
        and verifier.get("false_accept_gate_passed") is True
        and verifier.get("headline_claim_allowed") is True
        and _int(verifier.get("live_call_count")) > 0
    )


def _repair_gate_status(repair_gate: Mapping[str, Any]) -> str:
    if not repair_gate:
        return "missing_repair_gate_decision"
    if repair_gate.get("repair_gate_decision_v3_ready") is not True:
        return "blocked_repair_gate_decision_not_ready"
    state = str(repair_gate.get("repair_gate_state") or "missing_state")
    if state == "unblocked":
        return "clean_repair_gate_unblocked"
    return state


def _repair_ladder_status(repair_ladder: Mapping[str, Any]) -> str:
    if not repair_ladder:
        return "missing_repair_ladder_materializer"
    if repair_ladder.get("repair_ladder_materializer_v4_ready") is not True:
        return "blocked_repair_ladder_materializer_not_ready"
    if (
        repair_ladder.get("headline_repair_claim_allowed") is True
        and _int(repair_ladder.get("repair_attempt_count")) > 0
    ):
        return "clean_repair_ladder_materialized"
    if (
        repair_ladder.get("gated_skip") is True
        and _int(repair_ladder.get("repair_attempt_count")) == 0
    ):
        return "materialized_gated_skip_repair_gate_blocked_no_live_repair_attempts"
    return "blocked_repair_ladder_not_promotable"


def _fr11_promotion_grade_consistency(fr11: Mapping[str, Any]) -> bool:
    return (
        fr11.get("fr11_nonforgetting_self_learning_pilot_v2_ready") is True
        and _float(fr11.get("after_ledger_consistency_rate")) == 1.0
        and _float(fr11.get("heldout_consistency_rate")) == 1.0
        and fr11.get("nonforgetting_passed") is True
        and fr11.get("controller_memory_update_applied") is True
        and fr11.get("promotion_allowed") is True
        and fr11.get("model_weight_update_claimed") is not True
    )


def _sidecar_headline_clean(sidecar: Mapping[str, Any]) -> bool:
    return (
        sidecar.get("ebcn_kan_bounded_diagnostic_expansion_v2_ready") is True
        and sidecar.get("live_integration_claim_allowed") is True
        and sidecar.get("deployed_verifier_claim_allowed") is True
    )


def _hardware_headline_clean(hardware: Mapping[str, Any]) -> bool:
    return (
        hardware.get("hardware_tooling_boundary_v8_ready") is True
        and hardware.get("authenticated_speedup_claim_allowed") is True
        and hardware.get("speedup_claim_made") is True
        and bool(_list(hardware.get("hardware_commands_run")))
    )


def _next_top_gap(
    duration_status: str,
    clean_verifier: bool,
    repair_gate_status: str,
    repair_ladder_status: str,
    sidecar_clean: bool,
    hardware_clean: bool,
    publication_blocker_count: int,
) -> str:
    if not duration_status.startswith("passed_"):
        return "duration_corrected_authenticity_contract_repair"
    if (
        not clean_verifier
        or repair_gate_status != "clean_repair_gate_unblocked"
        or repair_ladder_status != "clean_repair_ladder_materialized"
    ):
        return "clean_live_verifier_adversarial_flag_clearance_repair_gate_unblock"
    if not sidecar_clean:
        return "ebcn_kan_live_integration_boundary"
    if not hardware_clean:
        return "hardware_sampler_authenticated_speedup_evidence"
    if publication_blocker_count > 0:
        return "publication_blocker_retirement"
    return "publication_scope_reconciliation"


def _matrix_comparison(
    matrix_v28: Mapping[str, Any],
    matrix_v27: Mapping[str, Any],
    capstone_v293: Mapping[str, Any],
) -> JsonDict:
    comparison = _mapping(matrix_v28.get("missing_artifact_comparison"))
    return {
        "v27_publication_blocker_count": _int(matrix_v27.get("publication_blocker_count")),
        "v28_publication_blocker_count": _int(matrix_v28.get("publication_blocker_count")),
        "capstone_v293_publication_blocker_count": _int(
            capstone_v293.get("publication_blocker_count")
        ),
        "blocker_delta_from_v27": _int(matrix_v28.get("blocker_delta_from_v27")),
        "v27_missing_artifact_count": _int(comparison.get("v27_missing_artifact_count")),
        "v28_missing_artifact_count": _int(comparison.get("v28_missing_artifact_count")),
        "missing_artifact_delta_from_v27": _int(comparison.get("missing_artifact_delta_from_v27")),
    }


def _phase_outcome_summary(
    duration_status: str,
    clean_verifier: bool,
    repair_gate_status: str,
    repair_ladder_status: str,
    fr11_consistency: bool,
    sidecar_clean: bool,
    hardware_clean: bool,
) -> JsonDict:
    return {
        "duration_corrected_authenticity_contract_passed": duration_status.startswith("passed_"),
        "clean_live_verifier_evidence_exists": clean_verifier,
        "repair_unblocked": repair_gate_status == "clean_repair_gate_unblocked",
        "repair_materialized_as_gated_skip": repair_ladder_status.startswith(
            "materialized_gated_skip"
        ),
        "fr11_reached_promotion_grade_controller_memory_consistency": fr11_consistency,
        "ebcn_kan_remain_bounded_diagnostics": not sidecar_clean,
        "hardware_claims_remain_blocked": not hardware_clean,
    }


def _source_artifacts(root: Path, payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for experiment_id, rel_path, role, ready_field in SOURCE_SPECS:
        payload = payloads.get(experiment_id) or {}
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": rel_path.as_posix(),
                "role": role,
                "ready_field": ready_field,
                "present": (root / rel_path).is_file(),
                "readable_json_object": bool(payload),
                "ready": payload.get(ready_field) is True,
                "sha256": sha256_file(root / rel_path),
                "source_type": "json",
            }
        )
    return rows


def _invariant_violations(
    matrix_v28: Mapping[str, Any],
    matrix_v27: Mapping[str, Any],
    capstone_v293: Mapping[str, Any],
    derived_paper_ready: bool,
    missing_artifact_count: int,
) -> list[str]:
    checks = (
        (not matrix_v28, "matrix_v28 authority is missing or malformed"),
        (
            bool(matrix_v28) and matrix_v28.get("matrix_v28_ready") is not True,
            "matrix_v28 authority is not ready",
        ),
        (not matrix_v27, "matrix_v27 authority is missing or malformed"),
        (
            bool(matrix_v27) and matrix_v27.get("matrix_v27_ready") is not True,
            "matrix_v27 authority is not ready",
        ),
        (not capstone_v293, "capstone_v293 authority is missing or malformed"),
        (
            bool(capstone_v293) and capstone_v293.get("capstone_ready") is not True,
            "capstone_v293 authority is not ready",
        ),
        (
            bool(matrix_v28)
            and _int(matrix_v28.get("publication_blocker_count"))
            != _int(matrix_v28.get("prior_publication_blocker_count"))
            + _int(matrix_v28.get("blocker_delta_from_v27")),
            "matrix_v28 blocker delta does not reconcile",
        ),
        (
            bool(matrix_v28)
            and missing_artifact_count
            != _int(
                _mapping(matrix_v28.get("missing_artifact_comparison")).get(
                    "v28_missing_artifact_count"
                )
            ),
            "matrix_v28 missing artifact count does not reconcile",
        ),
        (
            bool(matrix_v28) and bool(_list(matrix_v28.get("required_source_errors"))),
            "matrix_v28 reports required source errors",
        ),
        (
            bool(matrix_v28) and bool(_list(matrix_v28.get("invariant_violations"))),
            "matrix_v28 reports invariant violations",
        ),
        (
            bool(matrix_v28)
            and _substrate_runs_execution(_mapping(matrix_v28.get("inference_substrate"))),
            "matrix_v28 inference_substrate is not aggregation-only",
        ),
        (
            bool(matrix_v28) and matrix_v28.get("paper_ready") is not derived_paper_ready,
            "matrix_v28 paper_ready disagrees with derived capstone paper_ready",
        ),
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
        "kind": "capstone_aggregation_from_checked_in_matrix_v28_and_phase_artifacts",
        "source": "matrix_v28_matrix_v27_capstone_v293_and_dot294_phase_artifacts",
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
        "capstone_artifact_alone_is_deliverable": True,
        "reason": "task stop rule delegates ops/status/changelog/traceability reconciliation to conductor",
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "traceability_updated_after_spec": False,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("capstone_v294_ready") is not True:
        first = str(_list(artifact.get("invariant_violations"))[0])
        return f"blocked: capstone_v294_ready=false; {first}"
    return (
        "complete: capstone_v294_ready=true; "
        f"capstone_ready={str(artifact.get('capstone_ready')).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v27={artifact.get('blocker_delta_from_v27')}; "
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


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
