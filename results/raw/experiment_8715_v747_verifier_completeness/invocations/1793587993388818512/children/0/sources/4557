"""Build the Exp 2998 cross-corpus matrix v15 artifact.

Spec refs: REQ-REPORT-2998, SCENARIO-REPORT-2998.

This module is an aggregation layer. It reads matrix v14, the .280 capstone,
and the checked-in .281 artifacts, then emits claim-boundary rows. It does not
rerun model inference, verifier scoring, solver execution, synthesis, board
flashing, readback, or hardware smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
MILESTONE = "2026.05.281"
SCHEMA = "carnot.cross_corpus_matrix.v15_281_claim_boundary.v1"
ARTIFACT = "experiment_2998_cross_corpus_matrix_v15"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2998_cross_corpus_matrix_v15.json")

MATRIX_V14_REL_PATH = Path("results/experiment_2986_cross_corpus_matrix_v14.json")
CAPSTONE_V280_REL_PATH = Path("results/experiment_2987_capstone_v280.json")
EXP2988_REL_PATH = Path("results/experiment_2988_archive_v280_activate_v281.json")
EXP2989_REL_PATH = Path("results/experiment_2989_sota_gguf_cache_provenance_preflight_v1.json")
EXP2990_REL_PATH = Path("results/experiment_2990_verifier_backed_hard_code_stress_manifest_v1.json")
EXP2991_REL_PATH = Path("results/experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1.json")
EXP2992_REL_PATH = Path("results/experiment_2992_sota_solver_formalization_provenance_reproduction_v1.json")
EXP2993_REL_PATH = Path("results/experiment_2993_aquaforte_beaver_substrate_corrigendum_v1.json")
EXP2994_REL_PATH = Path("results/experiment_2994_prompt_validator_dialogue_schema_v1.json")
EXP2995_REL_PATH = Path("results/experiment_2995_fr11_verifier_grounded_trace_memory_v2.json")
EXP2996_REL_PATH = Path("results/experiment_2996_gatemate_host_visible_readback_smoke_v1.json")
EXP2997_REL_PATH = Path("results/experiment_2997_ssqa_dual_bram_rtl_pnr_resource_report_v1.json")

STATUSES = (
    "clean",
    "flagged",
    "blocked",
    "gated-skipped",
    "pilot-only",
    "projection-only",
    "missing",
)

HARDWARE_FORBIDDEN_FIELDS = (
    "sampler_claim_allowed",
    "sampler_claim_made",
    "speedup_claim_allowed",
    "speedup_claim_made",
    "thermodynamic_claim_allowed",
    "thermodynamic_claim_made",
    "boltzmann_thermalization_claim_made",
    "same_basis_cpu_fpga_speedup_claim_made",
    "extropic_execution_claim_made",
    "npu_acceleration_claim_made",
    "photonic_execution_claim_made",
    "hardware_sovereignty_claim_made",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp2986", MATRIX_V14_REL_PATH, required=True),
    SourceSpec("exp2987", CAPSTONE_V280_REL_PATH, required=True),
    SourceSpec("exp2988", EXP2988_REL_PATH),
    SourceSpec("exp2989", EXP2989_REL_PATH),
    SourceSpec("exp2990", EXP2990_REL_PATH),
    SourceSpec("exp2991", EXP2991_REL_PATH),
    SourceSpec("exp2992", EXP2992_REL_PATH),
    SourceSpec("exp2993", EXP2993_REL_PATH),
    SourceSpec("exp2994", EXP2994_REL_PATH),
    SourceSpec("exp2995", EXP2995_REL_PATH),
    SourceSpec("exp2996", EXP2996_REL_PATH),
    SourceSpec("exp2997", EXP2997_REL_PATH),
)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON artifact, returning `{}` when it is absent or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA256 digest for an artifact path, or `None` when absent."""

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
) -> dict[str, Any]:
    """REQ-REPORT-2998: build matrix v15 from upstream artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    source_artifacts = _source_artifacts_read(root_path, payloads)
    rows = [*_prior_v14_rows(payloads.get("exp2986", {})), *_v15_rows(payloads)]
    counts = _status_counts(rows)
    violations = _claim_boundary_violations(rows)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_v15_ready": False,
        "honest_verdict": "blocked_required_upstream_missing",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts_read": source_artifacts,
        "source_checksums": _source_checksums(source_artifacts),
        "rows": rows,
        "row_count": len(rows),
        "n_clean": counts["clean"],
        "n_flagged": counts["flagged"],
        "n_blocked": counts["blocked"],
        "n_gated_skipped": counts["gated-skipped"],
        "n_pilot_only": counts["pilot-only"],
        "n_projection_only": counts["projection-only"],
        "n_missing": counts["missing"],
        "claim_rows": _claim_rows(rows),
        "hardware_claim_boundary": _hardware_claim_boundary(rows),
        "self_learning_claim_boundary": _self_learning_claim_boundary(rows),
        "paper_v6_claim_boundary": _paper_v6_claim_boundary(rows),
        "prd_openspec_claim_boundary": _prd_openspec_claim_boundary(rows),
        "claim_boundary_violations": violations,
        "unresolved_blockers": _unresolved_blockers(rows),
        "next_milestone_recommendations": _next_milestone_recommendations(rows),
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "duration_s": duration_s,
    }

    required_errors = _required_source_errors(payloads)
    if required_errors:
        artifact["required_upstream_errors"] = required_errors
        return artifact

    artifact["matrix_v15_ready"] = True
    artifact["honest_verdict"] = _complete_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2998 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    return {spec.experiment_id: read_json_object(root / spec.path) for spec in SOURCE_SPECS}


def _source_artifacts_read(
    root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "present": path.is_file(),
                "required": spec.required,
                "readable_json_object": bool(payloads.get(spec.experiment_id)),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _source_checksums(source_artifacts: list[dict[str, Any]]) -> dict[str, str | None]:
    return {str(row["path"]): row["sha256"] for row in source_artifacts}


def _required_source_errors(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        if spec.required and not payloads.get(spec.experiment_id):
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "path": spec.path.as_posix(),
                    "reason": "missing_or_malformed_artifact",
                }
            )
    return errors


def _prior_v14_rows(matrix_v14: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in matrix_v14.get("rows") if isinstance(matrix_v14.get("rows"), list) else []:
        if not isinstance(source, Mapping):
            continue
        status = str(source.get("status") or "missing")
        if status not in STATUSES:
            status = "missing"
        source_row_id = str(source.get("row_id") or "unknown")
        rows.append(
            _row(
                row_id=f"carry_forward_v14:{source_row_id}",
                source_experiment_id=str(source.get("source_experiment_id") or "exp2986"),
                status=status,
                claim_class="prior_v14_carry_forward",
                evidence_type="matrix_v14_row",
                inference_substrate=INFERENCE_SUBSTRATE,
                claim_boundary="Matrix v14 row carried forward without metric recomputation or claim promotion.",
                source_honest_verdict=str(source.get("source_honest_verdict") or matrix_v14.get("honest_verdict") or ""),
                summary={
                    "source_matrix": "v14",
                    "source_row_id": source_row_id,
                    "source_status": status,
                    "source_claim_class": str(source.get("claim_class") or ""),
                },
            )
        )
    return rows


def _v15_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _exp2988_row(payloads.get("exp2988", {})),
        _exp2989_row(payloads.get("exp2989", {})),
        _exp2990_row(payloads.get("exp2990", {})),
        _exp2991_row(payloads.get("exp2991", {})),
        _exp2992_row(payloads.get("exp2992", {})),
        _exp2993_row(payloads.get("exp2993", {})),
        _exp2994_row(payloads.get("exp2994", {})),
        _exp2995_row(payloads.get("exp2995", {})),
        _exp2996_row(payloads.get("exp2996", {})),
        _exp2997_row(payloads.get("exp2997", {})),
    ]


def _exp2988_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = payload.get("archive_ready") is True
    status = _status_with_guards("projection-only" if ready else "blocked", payload, [])
    return _artifact_row(
        "exp2988_archive_activation",
        "exp2988",
        status,
        "archive_activation",
        "aggregation_only_archive_state",
        "Archive/activation bookkeeping is not new research evidence.",
        payload,
        {
            "archive_ready": bool(payload.get("archive_ready")),
            "activated_milestone": str(payload.get("activated_milestone") or ""),
            "status_updates_written": bool(payload.get("status_updates_written")),
        },
        [],
    )


def _exp2989_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    available = payload.get("sota_models_available") if isinstance(payload.get("sota_models_available"), list) else []
    transcripts = _string_list(payload.get("live_transcript_paths"))
    ready = payload.get("sota_headline_ready") is True and bool(available) and bool(transcripts)
    status = _status_with_guards("clean" if ready else "blocked", payload, [])
    return _artifact_row(
        "exp2989_sota_cache",
        "exp2989",
        status,
        "sota_cache_provenance",
        "live_llm_inference",
        "SOTA cache row proves only the mandated local model transcripts actually produced.",
        payload,
        {
            "sota_headline_ready": bool(payload.get("sota_headline_ready")),
            "n_available_sota_models": len(available),
            "n_live_transcripts": len(transcripts),
            "legacy_smoke_only_used": bool(payload.get("legacy_smoke_only_used")),
        },
        [],
    )


def _exp2990_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = (
        payload.get("hard_code_stress_set_ready") is True
        and payload.get("all_items_have_tests") is True
        and payload.get("all_reference_solutions_pass") is True
        and payload.get("all_baseline_candidates_fail") is True
        and _coerce_int(payload.get("n_items")) is not None
    )
    status = _status_with_guards("clean" if ready else "blocked", payload, [])
    return _artifact_row(
        "exp2990_hard_code_manifest",
        "exp2990",
        status,
        "hard_code_manifest",
        "deterministic_executable_manifest_generation",
        "Hard-code manifest readiness is executable verifier evidence, not repair improvement evidence.",
        payload,
        {
            "hard_code_stress_set_ready": bool(payload.get("hard_code_stress_set_ready")),
            "n_items": _coerce_int(payload.get("n_items")),
            "manifest_path": str(payload.get("manifest_path") or ""),
            "flaky_items": payload.get("flaky_items") if isinstance(payload.get("flaky_items"), list) else [],
            "rejected_item_ids": payload.get("rejected_item_ids") if isinstance(payload.get("rejected_item_ids"), list) else [],
        },
        [],
    )


def _exp2991_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("repair_rerun_clean") is True
        and _coerce_int(payload.get("n_tasks")) is not None
        and (_coerce_int(payload.get("n_tasks")) or 0) >= 20
        and (_coerce_float(payload.get("pass_at_1_delta")) or 0.0) > 0.0
        and (_coerce_float(payload.get("pass_at_k_delta")) or 0.0) > 0.0
        and (_coerce_float(payload.get("verifier_false_accept_delta")) or 0.0) <= 0.0
    )
    status = _status_with_guards("clean" if clean else "flagged", payload, [])
    return _artifact_row(
        "exp2991_intent_preserving_repair",
        "exp2991",
        status,
        "repair_eval",
        "live_llm_inference",
        "Repair cannot be promoted while promotion gates or artifact flags fail.",
        payload,
        {
            "repair_rerun_clean": bool(payload.get("repair_rerun_clean")),
            "headline_result": bool(payload.get("headline_result")),
            "n_tasks": _coerce_int(payload.get("n_tasks")),
            "pass_at_1_delta": _coerce_float(payload.get("pass_at_1_delta")),
            "pass_at_k_delta": _coerce_float(payload.get("pass_at_k_delta")),
            "verifier_false_accept_delta": _coerce_float(payload.get("verifier_false_accept_delta")),
            "headline_models_used": _string_list(payload.get("headline_models_used")),
        },
        [],
    )


def _exp2992_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("solver_provenance_reproduced") is True
        and payload.get("formalization_clean") is True
        and _coerce_float(payload.get("parseability")) == 1.0
        and _coerce_float(payload.get("solver_verified_accuracy")) == 1.0
        and _coerce_float(payload.get("z3_execution_rate")) == 1.0
        and _coerce_float(payload.get("tautology_rate")) == 0.0
        and payload.get("model_checksums_recorded") is True
        and payload.get("prompt_hashes_recorded") is True
        and payload.get("z3_transcript_hashes_recorded") is True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, [])
    return _artifact_row(
        "exp2992_solver_provenance",
        "exp2992",
        status,
        "solver_provenance",
        "live_llm_inference_plus_z3_provenance",
        "Solver row is a provenance reproduction claim, not broad verifier generalization.",
        payload,
        {
            "solver_provenance_reproduced": bool(payload.get("solver_provenance_reproduced")),
            "formalization_clean": bool(payload.get("formalization_clean")),
            "n_items": _coerce_int(payload.get("n_items")),
            "parseability": _coerce_float(payload.get("parseability")),
            "solver_verified_accuracy": _coerce_float(payload.get("solver_verified_accuracy")),
            "z3_execution_rate": _coerce_float(payload.get("z3_execution_rate")),
            "tautology_rate": _coerce_float(payload.get("tautology_rate")),
            "models_used": _string_list(payload.get("models_used")),
        },
        [],
    )


def _exp2993_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("substrate_corrigendum_complete") is True
        and payload.get("live_llm_retry_measured") is True
        and payload.get("enumerator_only_fallback_measured") is True
        and payload.get("substrate_labels_corrected") is True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, [])
    conditions = _mapping(payload.get("verifier_results_by_condition"))
    return _artifact_row(
        "exp2993_aquaforte_beaver_substrate",
        "exp2993",
        status,
        "aquaforte_beaver_substrate",
        "live_llm_plus_exact_verifier_and_enumerator_fallback",
        "Live LLM retry and enumerator-only fallback stay separate substrate claims.",
        payload,
        {
            "substrate_corrigendum_complete": bool(payload.get("substrate_corrigendum_complete")),
            "live_llm_retry_measured": bool(payload.get("live_llm_retry_measured")),
            "enumerator_only_fallback_measured": bool(payload.get("enumerator_only_fallback_measured")),
            "substrate_labels_corrected": bool(payload.get("substrate_labels_corrected")),
            "live_llm_retry": _mapping(conditions.get("live_llm_retry")),
            "enumerator_only_fallback": _mapping(conditions.get("enumerator_only_fallback")),
        },
        [],
    )


def _exp2994_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("prompt_validator_protocol_ready") is True
        and payload.get("exact_verifier_authority_preserved") is True
        and payload.get("static_transition_representation_designed") is True
        and payload.get("live_llm_judge_used") is not True
        and payload.get("no_speed_claim_made") is True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, [])
    return _artifact_row(
        "exp2994_prompt_validator_protocol",
        "exp2994",
        status,
        "prompt_validator_protocol",
        "deterministic_prompt_validator_harness",
        "Prompt validators are protocol/exact-check evidence, not live LLM judge evidence.",
        payload,
        {
            "prompt_validator_protocol_ready": bool(payload.get("prompt_validator_protocol_ready")),
            "exact_verifier_authority_preserved": bool(payload.get("exact_verifier_authority_preserved")),
            "static_transition_representation_designed": bool(payload.get("static_transition_representation_designed")),
            "llm_inference_run": bool(payload.get("llm_inference_run")),
            "live_llm_judge_used": bool(payload.get("live_llm_judge_used")),
            "n_validator_tree_fixtures": _coerce_int(payload.get("n_validator_tree_fixtures")),
        },
        [],
    )


def _exp2995_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("continuous_self_learning_task") is True
        and payload.get("trace_memory_ready") is True
        and payload.get("independent_self_learning_boundary_preserved") is True
        and payload.get("no_identical_metric_flag") is True
        and payload.get("forgetting_guard_passed") is True
        and payload.get("leakage_flag") is not True
        and payload.get("controls_improve_equally") is not True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, [])
    return _artifact_row(
        "exp2995_fr11_trace_memory",
        "exp2995",
        status,
        "fr11_self_learning",
        "artifact_replay_from_solver_and_validator_traces",
        "FR-11 evidence is verifier-grounded trace memory, not broad autonomous self-improvement.",
        payload,
        {
            "continuous_self_learning_task": bool(payload.get("continuous_self_learning_task")),
            "trace_memory_ready": bool(payload.get("trace_memory_ready")),
            "independent_self_learning_boundary_preserved": bool(
                payload.get("independent_self_learning_boundary_preserved")
            ),
            "no_identical_metric_flag": bool(payload.get("no_identical_metric_flag")),
            "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
            "leakage_flag": bool(payload.get("leakage_flag")),
            "controls_improve_equally": bool(payload.get("controls_improve_equally")),
            "n_trace_memories": _coerce_int(payload.get("n_trace_memories")),
            "heldout_metric_deltas": _mapping(payload.get("heldout_metric_deltas")),
            "negative_control_deltas": _mapping(payload.get("negative_control_deltas")),
        },
        [],
    )


def _exp2996_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _hardware_claim_violations("exp2996_gatemate_readback_smoke", payload)
    readback_or_smoke = bool(payload.get("readback_hash")) or payload.get("smoke_vector_passed") is True
    clean = payload.get("hardware_smoke_boundary_recorded") is True and readback_or_smoke
    status = _status_with_guards("clean" if clean else "blocked", payload, violations)
    return _artifact_row(
        "exp2996_gatemate_readback_smoke",
        "exp2996",
        status,
        "hardware_readback_smoke",
        "physical_gatemate_board",
        "GateMate remains blocked until host-visible readback or smoke output exists.",
        payload,
        {
            "hardware_smoke_boundary_recorded": bool(payload.get("hardware_smoke_boundary_recorded")),
            "board_detected": bool(payload.get("board_detected")),
            "flash_attempted": bool(payload.get("flash_attempted")),
            "flash_succeeded": bool(payload.get("flash_succeeded")),
            "readback_attempted": bool(payload.get("readback_attempted")),
            "readback_hash_present": bool(payload.get("readback_hash")),
            "smoke_vector_attempted": bool(payload.get("smoke_vector_attempted")),
            "smoke_vector_passed": bool(payload.get("smoke_vector_passed")),
            "missing_interface": str(payload.get("missing_interface") or ""),
        },
        violations,
    )


def _exp2997_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _hardware_claim_violations("exp2997_ssqa_dual_bram_rtl_pnr", payload)
    ready = payload.get("ssqa_rtl_pnr_report_ready") is True
    projection = payload.get("projection_only") is True
    status = _status_with_guards("clean" if ready else "projection-only" if projection else "blocked", payload, violations)
    return _artifact_row(
        "exp2997_ssqa_dual_bram_rtl_pnr",
        "exp2997",
        status,
        "hardware_ssqa_rtl_pnr",
        "rtl_pnr_resource_report",
        "SSQA evidence requires RTL/PnR/resource evidence and still cannot claim sampler speedup.",
        payload,
        {
            "ssqa_rtl_pnr_report_ready": bool(payload.get("ssqa_rtl_pnr_report_ready")),
            "preconditions_checked": bool(payload.get("preconditions_checked")),
            "rtl_path": str(payload.get("rtl_path") or ""),
            "pnr_report_path": str(payload.get("pnr_report_path") or ""),
            "resource_report_path": str(payload.get("resource_report_path") or ""),
            "readback_boundary_used": bool(payload.get("readback_boundary_used")),
            "projection_only": bool(payload.get("projection_only")),
        },
        violations,
    )


def _artifact_row(
    row_id: str,
    source_experiment_id: str,
    status: str,
    claim_class: str,
    evidence_type: str,
    claim_boundary: str,
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    claim_boundary_violations: list[dict[str, Any]],
) -> dict[str, Any]:
    if not payload:
        status = "missing"
    return _row(
        row_id=row_id,
        source_experiment_id=source_experiment_id,
        status=status,
        claim_class=claim_class,
        evidence_type=evidence_type,
        inference_substrate=str(payload.get("inference_substrate") or evidence_type),
        claim_boundary=claim_boundary,
        source_honest_verdict=str(payload.get("honest_verdict", "")),
        summary=summary,
        claim_boundary_violations=claim_boundary_violations,
        upstream_flags=_flag_kinds(payload),
        model_boundary=_model_boundary(payload),
        hardware_boundary=_hardware_boundary(payload),
        self_learning_boundary=_self_learning_boundary(payload),
    )


def _row(
    *,
    row_id: str,
    source_experiment_id: str,
    status: str,
    claim_class: str,
    evidence_type: str,
    inference_substrate: str,
    claim_boundary: str,
    source_honest_verdict: str,
    summary: Mapping[str, Any],
    claim_boundary_violations: list[dict[str, Any]] | None = None,
    upstream_flags: list[str] | None = None,
    model_boundary: Mapping[str, Any] | None = None,
    hardware_boundary: Mapping[str, Any] | None = None,
    self_learning_boundary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    violations = claim_boundary_violations or []
    return {
        "row_id": row_id,
        "source_experiment_id": source_experiment_id,
        "milestone": MILESTONE,
        "status": status,
        "claim_class": claim_class,
        "evidence_type": evidence_type,
        "inference_substrate": inference_substrate,
        "headline_eligible": status == "clean",
        "paper_claim_eligible": status == "clean",
        "claim_boundary": claim_boundary,
        "claim_boundary_guard_passed": not violations,
        "claim_boundary_violations": violations,
        "source_honest_verdict": source_honest_verdict,
        "upstream_flags": upstream_flags or [],
        "model_boundary": dict(model_boundary or {"status": "not_applicable"}),
        "hardware_boundary": dict(hardware_boundary or {"status": "not_applicable"}),
        "self_learning_boundary": dict(self_learning_boundary or {"status": "not_applicable"}),
        "summary": dict(summary),
    }


def _status_with_guards(
    base_status: str,
    payload: Mapping[str, Any],
    violations: list[dict[str, Any]],
) -> str:
    if _gated_verdict(payload.get("honest_verdict")):
        return "gated-skipped"
    if violations or _has_flags(payload):
        return "flagged"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    return base_status


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith("blocked")


def _gated_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("gate_blocked", "gated", "skip")
    )


def _has_flags(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True or bool(
        payload.get("corrigendum_pending") if isinstance(payload.get("corrigendum_pending"), list) else []
    )


def _flag_kinds(payload: Mapping[str, Any]) -> list[str]:
    kinds: list[str] = []
    if payload.get("flagged_adversarial") is True:
        kinds.append("flagged_adversarial=true")
    flags = payload.get("corrigendum_pending")
    if isinstance(flags, list):
        for item in flags:
            if isinstance(item, Mapping):
                kinds.append(f"{item.get('kind', 'unknown')}:{item.get('severity', 'unknown')}")
    return _unique_strings(kinds)


def _model_boundary(payload: Mapping[str, Any]) -> dict[str, Any]:
    headline = _headline_model_ids(payload)
    used = _models_used(payload)
    if not headline:
        return {"status": "not_applicable"}
    if not set(headline).intersection(used):
        return {"status": "non_compliant_missing_mandated_model", "headline_model_ids": headline, "models_used": used}
    if _has_flags(payload):
        return {"status": "flagged_mandated_model_evidence", "headline_model_ids": headline, "models_used": used}
    return {"status": "compliant", "headline_model_ids": headline, "models_used": used}


def _headline_model_ids(payload: Mapping[str, Any]) -> list[str]:
    model_specs = _mapping(payload.get("model_specs"))
    return (
        _string_list(payload.get("headline_model_ids"))
        or _string_list(payload.get("headline_models"))
        or _string_list(payload.get("mandatory_headline_model_ids"))
        or _string_list(model_specs.get("headline_models"))
    )


def _models_used(payload: Mapping[str, Any]) -> list[str]:
    available = payload.get("sota_models_available")
    available_ids = [str(item.get("hf_id")) for item in available if isinstance(item, Mapping)] if isinstance(available, list) else []
    return _unique_strings(
        [
            *_string_list(payload.get("models_used")),
            *_string_list(payload.get("headline_models_used")),
            *available_ids,
        ]
    )


def _hardware_boundary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"status": "not_applicable"}
    if payload.get("projection_only") is True:
        return {"status": "projection_only"}
    violations = [field for field in HARDWARE_FORBIDDEN_FIELDS if payload.get(field) is True]
    if violations:
        return {"status": "claim_boundary_violation", "fields": violations}
    if payload.get("inference_substrate") == "physical_gatemate_board":
        if bool(payload.get("readback_hash")) or payload.get("smoke_vector_passed") is True:
            return {"status": "clean"}
        return {"status": "blocked"}
    return {"status": "not_applicable"}


def _self_learning_boundary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("continuous_self_learning_task") is not True:
        return {"status": "not_applicable"}
    clean = (
        payload.get("independent_self_learning_boundary_preserved") is True
        and payload.get("no_identical_metric_flag") is True
        and payload.get("forgetting_guard_passed") is True
        and payload.get("leakage_flag") is not True
        and payload.get("controls_improve_equally") is not True
    )
    return {
        "status": "clean" if clean else "blocked",
        "boundary": "verifier-grounded trace memory only; no broad autonomous self-improvement or model-weight update claim",
        "continuous_self_learning_task": True,
        "trace_memory_ready": bool(payload.get("trace_memory_ready")),
        "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
        "leakage_flag": bool(payload.get("leakage_flag")),
    }


def _hardware_claim_violations(row_id: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = [field for field in HARDWARE_FORBIDDEN_FIELDS if payload.get(field) is True]
    return [{"row_id": row_id, "violation": "unsupported_hardware_claim", "fields": fields}] if fields else []


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {status: sum(1 for row in rows if row.get("status") == status) for status in STATUSES}


def _claim_boundary_violations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        row_violations = row.get("claim_boundary_violations")
        if isinstance(row_violations, list):
            violations.extend(item for item in row_violations if isinstance(item, dict))
    return violations


def _claim_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {str(row.get("row_id")): row for row in rows}
    return {
        "sota_cache": dict(by_id.get("exp2989_sota_cache", {})),
        "hard_code_manifest": dict(by_id.get("exp2990_hard_code_manifest", {})),
        "repair": dict(by_id.get("exp2991_intent_preserving_repair", {})),
        "solver_provenance": dict(by_id.get("exp2992_solver_provenance", {})),
        "aquaforte_beaver_substrate": dict(by_id.get("exp2993_aquaforte_beaver_substrate", {})),
        "prompt_validator_protocol": dict(by_id.get("exp2994_prompt_validator_protocol", {})),
        "fr11_self_learning": dict(by_id.get("exp2995_fr11_trace_memory", {})),
        "gatemate": dict(by_id.get("exp2996_gatemate_readback_smoke", {})),
        "ssqa": dict(by_id.get("exp2997_ssqa_dual_bram_rtl_pnr", {})),
    }


def _hardware_claim_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claim_rows = _claim_rows(rows)
    violations = _claim_boundary_violations(
        [row for row in rows if str(row.get("claim_class", "")).startswith("hardware")]
    )
    return {
        "forbidden_claims_absent": not violations,
        "paper_v6_boundary": "No sampler speedup, Boltzmann thermalization, same-basis CPU-vs-FPGA speedup, Extropic, NPU, photonic, or hardware-sovereignty claim is made.",
        "gatemate": {
            "status": claim_rows["gatemate"].get("status", "missing"),
            "summary": claim_rows["gatemate"].get("summary", {}),
            "hardware_boundary": claim_rows["gatemate"].get("hardware_boundary", {}),
        },
        "ssqa": {
            "status": claim_rows["ssqa"].get("status", "missing"),
            "summary": claim_rows["ssqa"].get("summary", {}),
            "hardware_boundary": claim_rows["ssqa"].get("hardware_boundary", {}),
        },
        "violations": violations,
    }


def _self_learning_claim_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fr11 = _claim_rows(rows)["fr11_self_learning"]
    boundary = _mapping(fr11.get("self_learning_boundary"))
    return {
        "status": fr11.get("status", "missing"),
        "boundary": boundary.get("boundary", "FR-11 row missing"),
        "summary": fr11.get("summary", {}),
        "source_honest_verdict": fr11.get("source_honest_verdict", ""),
    }


def _paper_v6_claim_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hardware = _hardware_claim_boundary(rows)
    unsafe = [row["row_id"] for row in rows if row.get("status") in {"flagged", "blocked", "gated-skipped", "missing"} and row.get("headline_eligible")]
    return {
        "forbidden_claims_absent": hardware["forbidden_claims_absent"] and not unsafe,
        "anchored_claims_preserved": [
            "distribution-bound verifier calibration only",
            "hardware correctness-first sparse fast-path plus CPU fallback only",
            "self-learning is narrow verified-memory growth only",
        ],
        "not_promoted_statuses": ["flagged", "blocked", "gated-skipped", "pilot-only", "projection-only", "missing"],
        "unsafe_headline_rows": unsafe,
    }


def _prd_openspec_claim_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claim_rows = _claim_rows(rows)
    return {
        "prd_refs": ["FR-09", "FR-10", "FR-11", "FR-12"],
        "openspec_refs": ["REQ-REPORT-2998", "SCENARIO-REPORT-2998"],
        "repair_status": claim_rows["repair"].get("status", "missing"),
        "solver_status": claim_rows["solver_provenance"].get("status", "missing"),
        "fr11_status": claim_rows["fr11_self_learning"].get("status", "missing"),
        "hardware_status": claim_rows["gatemate"].get("status", "missing"),
    }


def _unresolved_blockers(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    for row in rows:
        status = str(row.get("status"))
        if status in {"flagged", "blocked", "gated-skipped", "missing"}:
            blockers.append(
                {
                    "row_id": str(row.get("row_id")),
                    "status": status,
                    "claim_class": str(row.get("claim_class")),
                    "reason": str(row.get("source_honest_verdict") or row.get("claim_boundary")),
                }
            )
    return blockers


def _next_milestone_recommendations(rows: list[dict[str, Any]]) -> list[str]:
    claim_rows = _claim_rows(rows)
    recommendations = [
        "Repair: resolve Exp 2991 tautology/methodology flags before promoting the positive hard-set delta.",
        "AquaForte/BEAVER: rerun live retry with methodology and duration provenance, keeping enumerator fallback labeled separately.",
        "GateMate: add host-visible readback or smoke-vector transport before any sampler-facing hardware row.",
        "SSQA: produce or explicitly gate-skip the dual-BRAM RTL/PnR/resource report.",
    ]
    if claim_rows["solver_provenance"].get("status") == "clean":
        recommendations.append("Solver: carry Exp 2992 provenance reproduction forward as a narrow Z3-backed row.")
    if claim_rows["fr11_self_learning"].get("status") == "clean":
        recommendations.append("FR-11: preserve independent metrics, negative controls, and forgetting guards as mandatory boundaries.")
    return recommendations


def _complete_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v15_ready=true; "
        f"clean={artifact['n_clean']}; "
        f"flagged={artifact['n_flagged']}; "
        f"blocked={artifact['n_blocked']}; "
        f"gated_skipped={artifact['n_gated_skipped']}; "
        f"pilot_only={artifact['n_pilot_only']}; "
        f"projection_only={artifact['n_projection_only']}; "
        f"missing={artifact['n_missing']}"
    )


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
