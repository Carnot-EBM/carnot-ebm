"""Build the Exp 3010 cross-corpus matrix v16 artifact.

Spec refs: REQ-REPORT-3010, SCENARIO-REPORT-3010.

This module is an aggregation-only closeout layer for milestone .282. It reads
checked-in source artifacts and claim-boundary documents, then classifies each
row without rerunning inference, exact validators, synthesis, board flashing,
readback, smoke tests, the conductor, or publication tooling.
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
MILESTONE = "2026.05.282"
SCHEMA = "carnot.cross_corpus_matrix.v16_282_claim_boundary.v1"
ARTIFACT = "experiment_3010_cross_corpus_matrix_v16"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3010_cross_corpus_matrix_v16.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3010_cross_corpus_matrix_v16.py"

MATRIX_V15_REL_PATH = Path("results/experiment_2998_cross_corpus_matrix_v15.json")
CAPSTONE_V281_REL_PATH = Path("results/experiment_2999_capstone_v281.json")
EXP3000_REL_PATH = Path("results/experiment_3000_archive_v281_activate_v282.json")
EXP3001_REL_PATH = Path("results/experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1.json")
EXP3002_REL_PATH = Path("results/experiment_3002_metamorphic_repair_oracle_audit_v1.json")
EXP3003_REL_PATH = Path("results/experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1.json")
EXP3004_REL_PATH = Path("results/experiment_3004_aquaforte_beaver_live_retry_provenance_v2.json")
EXP3005_REL_PATH = Path("results/experiment_3005_solver_to_validator_tree_expansion_v1.json")
EXP3006_REL_PATH = Path("results/experiment_3006_eqr_fixed_point_energy_diagnostic_v1.json")
EXP3007_REL_PATH = Path("results/experiment_3007_fr11_attractor_trace_memory_stability_v1.json")
EXP3008_REL_PATH = Path("results/experiment_3008_gatemate_host_visible_io_transport_v2.json")
EXP3009_REL_PATH = Path("results/experiment_3009_ssqa_dual_bram_rtl_pnr_resource_report_v2.json")

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
    "boltzmann_claim_made",
    "boltzmann_thermalization_claim_made",
    "same_basis_cpu_fpga_speedup_claim_made",
    "extropic_execution_claim_made",
    "tsu_access_claim_made",
    "z1_access_claim_made",
    "kona_access_claim_made",
    "kona_parity_claim_made",
    "npu_acceleration_claim_made",
    "photonic_execution_claim_made",
    "hardware_speedup_claim_made",
)

LLM_VERIFIER_FIELDS = (
    "llm_judge_used",
    "live_llm_judge_used",
    "llm_as_verifier_claim_made",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp2998", MATRIX_V15_REL_PATH, required=True),
    SourceSpec("exp2999", CAPSTONE_V281_REL_PATH, required=True),
    SourceSpec("exp3000", EXP3000_REL_PATH),
    SourceSpec("exp3001", EXP3001_REL_PATH),
    SourceSpec("exp3002", EXP3002_REL_PATH),
    SourceSpec("exp3003", EXP3003_REL_PATH),
    SourceSpec("exp3004", EXP3004_REL_PATH),
    SourceSpec("exp3005", EXP3005_REL_PATH),
    SourceSpec("exp3006", EXP3006_REL_PATH),
    SourceSpec("exp3007", EXP3007_REL_PATH),
    SourceSpec("exp3008", EXP3008_REL_PATH),
    SourceSpec("exp3009", EXP3009_REL_PATH),
)

DOCUMENT_REL_PATHS = (
    Path("CLAUDE.md"),
    Path("research-roadmap.yaml"),
    Path("research-roadmap-next.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-references.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON artifact, returning an empty mapping when unusable.

    The matrix is a closeout ledger, so malformed or absent inputs must become
    explicit source gaps instead of exceptions that hide the rest of the
    milestone.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA256 digest for a local file, or ``None`` when absent."""

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
    """REQ-REPORT-3010: build matrix v16 from upstream artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    source_artifacts = _source_artifacts_read(root_path, payloads)
    rows = [*_prior_v15_rows(payloads.get("exp2998", {})), *_v16_rows(payloads)]
    counts = _status_counts(rows)
    violations = _claim_boundary_violations(rows)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_v16_ready": False,
        "honest_verdict": "blocked_required_upstream_missing",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts_read": source_artifacts,
        "source_checksums": _source_checksums(source_artifacts),
        "document_inputs_read": _document_inputs_read(root_path),
        "missing_artifacts": _missing_artifacts(source_artifacts),
        "rows": rows,
        "row_count": len(rows),
        "clean_count": counts["clean"],
        "flagged_count": counts["flagged"],
        "blocked_count": counts["blocked"],
        "gated_skipped_count": counts["gated-skipped"],
        "pilot_only_count": counts["pilot-only"],
        "projection_only_count": counts["projection-only"],
        "missing_count": counts["missing"],
        "claim_rows": _claim_rows(rows),
        "repaired_claims": _repaired_claims(rows),
        "still_blocked_claims": _still_blocked_claims(rows),
        "claim_boundary_violations": violations,
        "recommended_next_actions": _recommended_next_actions(rows),
        "paper_v6_boundary_summary": _paper_v6_boundary_summary(rows, violations),
        "prd_openspec_boundary_summary": _prd_openspec_boundary_summary(rows),
        "hardware_boundary_summary": _hardware_boundary_summary(rows, violations),
        "roadmap_acceptance_summary": _roadmap_acceptance_summary(rows),
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
        "duration_s": duration_s,
    }

    required_errors = _required_source_errors(payloads)
    if required_errors:
        artifact["required_upstream_errors"] = required_errors
        return artifact

    artifact["matrix_v16_ready"] = True
    artifact["honest_verdict"] = _complete_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3010 deliverable JSON."""

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


def _document_inputs_read(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": rel_path.as_posix(),
            "present": (root / rel_path).is_file(),
            "sha256": sha256_file(root / rel_path),
        }
        for rel_path in DOCUMENT_REL_PATHS
    ]


def _missing_artifacts(source_artifacts: list[dict[str, Any]]) -> list[str]:
    return [str(row["path"]) for row in source_artifacts if row.get("present") is not True]


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


def _prior_v15_rows(matrix_v15: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in matrix_v15.get("rows") if isinstance(matrix_v15.get("rows"), list) else []:
        if not isinstance(source, Mapping):
            continue
        status = str(source.get("status") or "missing")
        if status not in STATUSES:
            status = "missing"
        source_row_id = str(source.get("row_id") or "unknown")
        rows.append(
            _row(
                row_id=f"carry_forward_v15:{source_row_id}",
                source_experiment_id=str(source.get("source_experiment_id") or "exp2998"),
                status=status,
                claim_class="prior_v15_carry_forward",
                evidence_type="matrix_v15_row",
                inference_substrate=INFERENCE_SUBSTRATE,
                claim_boundary="Matrix v15 row carried forward without metric recomputation or claim promotion.",
                source_honest_verdict=str(
                    source.get("source_honest_verdict") or matrix_v15.get("honest_verdict") or ""
                ),
                summary={
                    "source_matrix": "v15",
                    "source_row_id": source_row_id,
                    "source_status": status,
                    "source_claim_class": str(source.get("claim_class") or ""),
                },
            )
        )
    return rows


def _v16_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _exp3000_row(payloads.get("exp3000", {})),
        _exp3001_row(payloads.get("exp3001", {})),
        _exp3002_row(payloads.get("exp3002", {})),
        _exp3003_row(payloads.get("exp3003", {})),
        _exp3004_row(payloads.get("exp3004", {})),
        _exp3005_row(payloads.get("exp3005", {})),
        _exp3006_row(payloads.get("exp3006", {})),
        _exp3007_row(payloads.get("exp3007", {})),
        _exp3008_row(payloads.get("exp3008", {})),
        _exp3009_row(payloads.get("exp3009", {}), payloads.get("exp3008", {})),
    ]


def _exp3000_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = payload.get("archive_ready") is True and payload.get("activated_milestone") == MILESTONE
    status = "projection-only" if ready else _status_with_guards("blocked", payload, [])
    return _artifact_row(
        "exp3000_archive_activation",
        "exp3000",
        status,
        "archive_activation",
        "aggregation_only_archive_state",
        "Archive/activation is milestone bookkeeping and not paper-v6 claim evidence.",
        payload,
        {
            "archive_ready": bool(payload.get("archive_ready")),
            "activated_milestone": str(payload.get("activated_milestone") or ""),
            "status_updates_written": bool(payload.get("status_updates_written")),
        },
        [],
    )


def _exp3001_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    available = payload.get("sota_models_available") if isinstance(payload.get("sota_models_available"), list) else []
    transcripts = _string_list(payload.get("live_transcript_paths"))
    ready = (
        payload.get("sota_headline_ready") is True
        and payload.get("preconditions_checked") is True
        and bool(available)
        and bool(transcripts)
        and payload.get("legacy_smoke_only_used") is not True
    )
    status = _status_with_guards("clean" if ready else "blocked", payload, [])
    return _artifact_row(
        "exp3001_sota_cache",
        "exp3001",
        status,
        "sota_cache_provenance",
        "live_llm_inference",
        "SOTA cache readiness supports downstream gated runs but is not a repair-quality claim.",
        payload,
        {
            "sota_headline_ready": bool(payload.get("sota_headline_ready")),
            "n_available_sota_models": len(available),
            "n_live_transcripts": len(transcripts),
            "legacy_smoke_only_used": bool(payload.get("legacy_smoke_only_used")),
        },
        [],
    )


def _exp3002_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = (
        payload.get("metamorphic_oracle_ready") is True
        and payload.get("false_accept_probe_ready") is True
        and payload.get("tautology_probe_ready") is True
        and (_coerce_int(payload.get("n_source_items")) or 0) > 0
        and (_coerce_int(payload.get("n_metamorphic_variants")) or 0) > 0
    )
    status = _status_with_guards("clean" if ready else "blocked", payload, [])
    return _artifact_row(
        "exp3002_metamorphic_oracle",
        "exp3002",
        status,
        "metamorphic_oracle",
        "deterministic_oracle_audit_no_live_llm",
        "Metamorphic oracle readiness is an audit gate, not evidence that repair quality improved.",
        payload,
        {
            "metamorphic_oracle_ready": bool(payload.get("metamorphic_oracle_ready")),
            "false_accept_probe_ready": bool(payload.get("false_accept_probe_ready")),
            "tautology_probe_ready": bool(payload.get("tautology_probe_ready")),
            "n_source_items": _coerce_int(payload.get("n_source_items")),
            "n_metamorphic_variants": _coerce_int(payload.get("n_metamorphic_variants")),
            "relation_types": _string_list(payload.get("relation_types")),
        },
        [],
    )


def _exp3003_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("repair_rerun_clean") is True
        and payload.get("headline_result") is True
        and (_coerce_int(payload.get("n_tasks")) or 0) >= 20
        and (_coerce_int(payload.get("n_metamorphic_variants")) or 0) > 0
        and (_coerce_float(payload.get("pass_at_1_delta")) or 0.0) > 0.0
        and (_coerce_float(payload.get("pass_at_k_delta")) or 0.0) > 0.0
        and (_coerce_float(payload.get("syntax_failure_rate_delta")) or 1.0) <= 0.0
        and (_coerce_float(payload.get("schema_failure_rate_delta")) or 1.0) <= 0.0
        and (_coerce_float(payload.get("false_accept_delta")) or 0.0) <= 0.0
        and payload.get("tautology_gate_clean") is True
    )
    status = _status_with_guards("clean" if clean else "flagged", payload, [])
    return _artifact_row(
        "exp3003_metamorphic_repair",
        "exp3003",
        status,
        "repair_eval",
        "live_llm_inference_with_metamorphic_replay",
        "Hard-set repair is non-promotable until deltas, false accepts, syntax/schema, and flags clear.",
        payload,
        {
            "repair_rerun_clean": bool(payload.get("repair_rerun_clean")),
            "headline_result": bool(payload.get("headline_result")),
            "n_tasks": _coerce_int(payload.get("n_tasks")),
            "n_metamorphic_variants": _coerce_int(payload.get("n_metamorphic_variants")),
            "pass_at_1_delta": _coerce_float(payload.get("pass_at_1_delta")),
            "pass_at_k_delta": _coerce_float(payload.get("pass_at_k_delta")),
            "syntax_failure_rate_delta": _coerce_float(payload.get("syntax_failure_rate_delta")),
            "schema_failure_rate_delta": _coerce_float(payload.get("schema_failure_rate_delta")),
            "false_accept_delta": _coerce_float(payload.get("false_accept_delta")),
            "tautology_gate_clean": bool(payload.get("tautology_gate_clean")),
            "headline_models_used": _string_list(payload.get("headline_models_used")),
        },
        [],
    )


def _exp3004_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    transcripts = _string_list(payload.get("live_transcript_paths"))
    clean = (
        payload.get("substrate_corrigendum_promotable") is True
        and payload.get("live_retry_provenance_clean") is True
        and payload.get("enumerator_fallback_separated") is True
        and payload.get("contamination_detected") is not True
        and payload.get("impossible_duration_flag") is not True
        and (_coerce_float(payload.get("duration_seconds_live")) or 0.0) > 0.0
        and bool(transcripts)
        and bool(_string_list(payload.get("headline_models_used")))
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, [])
    return _artifact_row(
        "exp3004_aquaforte_beaver_provenance",
        "exp3004",
        status,
        "aquaforte_beaver_substrate_provenance",
        "live_llm_retry_plus_exact_verifier_and_enumerator_fallback",
        "AquaForte/BEAVER promotion is provenance-only: live retry and enumerator fallback remain separate.",
        payload,
        {
            "substrate_corrigendum_promotable": bool(payload.get("substrate_corrigendum_promotable")),
            "live_retry_provenance_clean": bool(payload.get("live_retry_provenance_clean")),
            "enumerator_fallback_separated": bool(payload.get("enumerator_fallback_separated")),
            "contamination_detected": bool(payload.get("contamination_detected")),
            "impossible_duration_flag": bool(payload.get("impossible_duration_flag")),
            "duration_seconds_live": _coerce_float(payload.get("duration_seconds_live")),
            "n_live_transcripts": len(transcripts),
            "headline_models_used": _string_list(payload.get("headline_models_used")),
        },
        [],
    )


def _exp3005_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("validator_tree_expanded") is True
        and payload.get("all_trees_exact_checked") is True
        and payload.get("partial_viability_checked") is True
        and payload.get("llm_judge_used") is not True
        and (_coerce_int(payload.get("n_validator_trees")) or 0) > 0
    )
    violations = _claim_violations("exp3005_validator_tree_expansion", payload)
    status = _status_with_guards("clean" if clean else "blocked", payload, violations)
    return _artifact_row(
        "exp3005_validator_tree_expansion",
        "exp3005",
        status,
        "validator_tree_expansion",
        "deterministic_runtime_and_z3_validator_tree_corpus",
        "Validator-tree rows are exact-check evidence; LLM judges cannot be verifier authority.",
        payload,
        {
            "validator_tree_expanded": bool(payload.get("validator_tree_expanded")),
            "all_trees_exact_checked": bool(payload.get("all_trees_exact_checked")),
            "partial_viability_checked": bool(payload.get("partial_viability_checked")),
            "llm_judge_used": bool(payload.get("llm_judge_used")),
            "n_validator_trees": _coerce_int(payload.get("n_validator_trees")),
        },
        violations,
    )


def _exp3006_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("fixed_point_diagnostic_ready") is True
        and payload.get("native_eqr_claim_made") is not True
        and (_coerce_float(payload.get("convergence_rate")) or 0.0) >= 1.0
        and (_coerce_float(payload.get("energy_monotonicity_rate")) or 0.0) >= 1.0
        and (_coerce_float(payload.get("negative_control_rejection_rate")) or 0.0) >= 1.0
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, [])
    return _artifact_row(
        "exp3006_fixed_point_diagnostic",
        "exp3006",
        status,
        "fixed_point_energy_diagnostic",
        "deterministic_energy_diagnostic_over_cached_validator_trajectories",
        "Fixed-point evidence is diagnostic only and makes no native EQR-model claim.",
        payload,
        {
            "fixed_point_diagnostic_ready": bool(payload.get("fixed_point_diagnostic_ready")),
            "native_eqr_claim_made": bool(payload.get("native_eqr_claim_made")),
            "convergence_rate": _coerce_float(payload.get("convergence_rate")),
            "energy_monotonicity_rate": _coerce_float(payload.get("energy_monotonicity_rate")),
            "negative_control_rejection_rate": _coerce_float(payload.get("negative_control_rejection_rate")),
        },
        [],
    )


def _exp3007_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("trace_memory_stability_ready") is True
        and payload.get("continuous_self_learning_task") is True
        and payload.get("independent_self_learning_boundary_preserved") is True
        and payload.get("convergence_guard_passed") is True
        and payload.get("drift_guard_passed") is True
        and payload.get("forgetting_guard_passed") is True
        and payload.get("negative_control_rejected") is True
        and payload.get("native_attractor_model_claim_made") is not True
        and payload.get("self_reported_memory_utility_counted") is not True
    )
    violations = _claim_violations("exp3007_fr11_trace_memory_stability", payload)
    status = _status_with_guards("clean" if clean else "blocked", payload, violations)
    return _artifact_row(
        "exp3007_fr11_trace_memory_stability",
        "exp3007",
        status,
        "fr11_trace_memory_stability",
        "artifact_replay_from_exact_verifier_traces",
        "FR-11 remains verifier-grounded trace-memory stability, not broad self-improvement.",
        payload,
        {
            "trace_memory_stability_ready": bool(payload.get("trace_memory_stability_ready")),
            "continuous_self_learning_task": bool(payload.get("continuous_self_learning_task")),
            "independent_self_learning_boundary_preserved": bool(
                payload.get("independent_self_learning_boundary_preserved")
            ),
            "convergence_guard_passed": bool(payload.get("convergence_guard_passed")),
            "drift_guard_passed": bool(payload.get("drift_guard_passed")),
            "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
            "negative_control_rejected": bool(payload.get("negative_control_rejected")),
            "heldout_delta": _coerce_float(payload.get("heldout_delta")),
            "heldout_task_count": _coerce_int(payload.get("heldout_task_count")),
        },
        violations,
    )


def _exp3008_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _claim_violations("exp3008_gatemate_host_visible_io", payload)
    host_visible = (
        payload.get("host_visible_io_ready") is True
        and (bool(payload.get("readback_hash")) or payload.get("smoke_vector_passed") is True)
    )
    clean = payload.get("hardware_smoke_boundary_recorded") is True and host_visible
    status = _status_with_guards("clean" if clean else "blocked", payload, violations)
    return _artifact_row(
        "exp3008_gatemate_host_visible_io",
        "exp3008",
        status,
        "gatemate_host_visible_io",
        "physical_gatemate_board_host_visible_io",
        "GateMate remains non-promotable until host-visible output, readback, or smoke bytes exist.",
        payload,
        {
            "host_visible_io_ready": bool(payload.get("host_visible_io_ready")),
            "hardware_smoke_boundary_recorded": bool(payload.get("hardware_smoke_boundary_recorded")),
            "board_detected": bool(payload.get("board_detected")),
            "flash_attempted": bool(payload.get("flash_attempted")),
            "flash_succeeded": bool(payload.get("flash_succeeded")),
            "readback_attempted": bool(payload.get("readback_attempted")),
            "readback_supported": bool(payload.get("readback_supported")),
            "readback_hash_present": bool(payload.get("readback_hash")),
            "smoke_vector_attempted": bool(payload.get("smoke_vector_attempted")),
            "smoke_vector_passed": bool(payload.get("smoke_vector_passed")),
            "io_transport_diagnosis": _mapping(payload.get("io_transport_diagnosis")),
        },
        violations,
    )


def _exp3009_row(payload: Mapping[str, Any], exp3008: Mapping[str, Any]) -> dict[str, Any]:
    violations = _claim_violations("exp3009_ssqa_dual_bram_report", payload)
    gate_open = exp3008.get("host_visible_io_ready") is True
    if not gate_open:
        status = "gated-skipped"
    elif not payload:
        status = "missing"
    else:
        ready = payload.get("ssqa_rtl_pnr_report_ready") is True
        projection = payload.get("projection_only") is True
        status = "clean" if ready else "projection-only" if projection else "blocked"
        status = _status_with_guards(status, payload, violations)
    return _artifact_row(
        "exp3009_ssqa_dual_bram_report",
        "exp3009",
        status,
        "ssqa_dual_bram_rtl_pnr_resource",
        "rtl_pnr_resource_report",
        "SSQA is gate-skipped until GateMate host-visible IO is ready; no speedup or sampler claim is allowed.",
        payload,
        {
            "upstream_exp3008_host_visible_io_ready": bool(exp3008.get("host_visible_io_ready")),
            "missing_artifact_present": bool(payload),
            "ssqa_rtl_pnr_report_ready": bool(payload.get("ssqa_rtl_pnr_report_ready")),
            "projection_only": bool(payload.get("projection_only")),
            "rtl_path": str(payload.get("rtl_path") or ""),
            "pnr_report_path": str(payload.get("pnr_report_path") or ""),
            "resource_report_path": str(payload.get("resource_report_path") or ""),
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
    if not payload and status != "gated-skipped":
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
    verdict = payload.get("honest_verdict")
    if _gated_verdict(verdict):
        return "gated-skipped"
    if violations or _has_flags(payload) or _flagged_verdict(verdict):
        return "flagged"
    if _blocked_verdict(verdict):
        return "blocked"
    return base_status


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith("blocked")


def _gated_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("gate_blocked", "gated", "skip")
    )


def _flagged_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith("flagged")


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
    if _has_flags(payload) or _flagged_verdict(payload.get("honest_verdict")):
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
    available_ids = (
        [str(item.get("hf_id")) for item in available if isinstance(item, Mapping)]
        if isinstance(available, list)
        else []
    )
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
    fields = [field for field in HARDWARE_FORBIDDEN_FIELDS if payload.get(field) is True]
    if fields:
        return {"status": "claim_boundary_violation", "fields": fields}
    if payload.get("projection_only") is True:
        return {"status": "projection_only"}
    if payload.get("inference_substrate") in {"hardware_smoke", "physical_gatemate_board"}:
        clean = payload.get("host_visible_io_ready") is True and (
            bool(payload.get("readback_hash")) or payload.get("smoke_vector_passed") is True
        )
        return {"status": "clean" if clean else "blocked"}
    return {"status": "not_applicable"}


def _self_learning_boundary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("continuous_self_learning_task") is not True:
        return {"status": "not_applicable"}
    clean = (
        payload.get("independent_self_learning_boundary_preserved") is True
        and payload.get("forgetting_guard_passed") is True
        and payload.get("negative_control_rejected") is True
        and payload.get("native_attractor_model_claim_made") is not True
        and payload.get("self_reported_memory_utility_counted") is not True
    )
    return {
        "status": "clean" if clean else "blocked",
        "boundary": "verifier-grounded trace-memory stability only; no broad autonomous self-improvement claim",
        "continuous_self_learning_task": True,
        "trace_memory_stability_ready": bool(payload.get("trace_memory_stability_ready")),
        "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
        "negative_control_rejected": bool(payload.get("negative_control_rejected")),
    }


def _claim_violations(row_id: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    hardware_fields = [field for field in HARDWARE_FORBIDDEN_FIELDS if payload.get(field) is True]
    if hardware_fields:
        violations.append(
            {
                "row_id": row_id,
                "violation": "unsupported_hardware_claim",
                "fields": hardware_fields,
            }
        )
    llm_fields = [field for field in LLM_VERIFIER_FIELDS if payload.get(field) is True]
    if llm_fields:
        violations.append(
            {
                "row_id": row_id,
                "violation": "llm_as_verifier_boundary_violation",
                "fields": llm_fields,
            }
        )
    return violations


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
        "exp3001_sota_cache": dict(by_id.get("exp3001_sota_cache", {})),
        "exp3002_metamorphic_oracle": dict(by_id.get("exp3002_metamorphic_oracle", {})),
        "exp3003_repair": dict(by_id.get("exp3003_metamorphic_repair", {})),
        "exp3004_substrate_provenance": dict(by_id.get("exp3004_aquaforte_beaver_provenance", {})),
        "exp3005_validator_tree": dict(by_id.get("exp3005_validator_tree_expansion", {})),
        "exp3006_fixed_point": dict(by_id.get("exp3006_fixed_point_diagnostic", {})),
        "exp3007_fr11_stability": dict(by_id.get("exp3007_fr11_trace_memory_stability", {})),
        "exp3008_gatemate_io": dict(by_id.get("exp3008_gatemate_host_visible_io", {})),
        "exp3009_ssqa": dict(by_id.get("exp3009_ssqa_dual_bram_report", {})),
    }


def _repaired_claims(rows: list[dict[str, Any]]) -> list[str]:
    claim_rows = _claim_rows(rows)
    repaired = []
    if claim_rows["exp3001_sota_cache"].get("status") == "clean":
        repaired.append("exp3001_sota_headline_cache_ready")
    if claim_rows["exp3004_substrate_provenance"].get("status") == "clean":
        repaired.append("exp3004_aquaforte_beaver_substrate_provenance")
    if claim_rows["exp3005_validator_tree"].get("status") == "clean":
        repaired.append("exp3005_validator_tree_expansion")
    if claim_rows["exp3006_fixed_point"].get("status") == "clean":
        repaired.append("exp3006_fixed_point_energy_diagnostic")
    if claim_rows["exp3007_fr11_stability"].get("status") == "clean":
        repaired.append("exp3007_fr11_trace_memory_stability")
    return repaired


def _still_blocked_claims(rows: list[dict[str, Any]]) -> list[str]:
    claim_rows = _claim_rows(rows)
    blocked: list[str] = []
    named_rows = {
        "exp3002_metamorphic_oracle": "exp3002_metamorphic_oracle_flagged",
        "exp3003_repair": "exp3003_metamorphic_repair_flagged",
        "exp3007_fr11_stability": "exp3007_fr11_trace_memory_stability_flagged",
        "exp3008_gatemate_io": "exp3008_gatemate_host_visible_io_blocked",
        "exp3009_ssqa": "exp3009_ssqa_gate_skipped_until_gatemate_io_ready",
    }
    for key, label in named_rows.items():
        status = claim_rows[key].get("status")
        if status in {"flagged", "blocked", "gated-skipped", "missing"}:
            blocked.append(label)
    for row in rows:
        status = str(row.get("status"))
        if str(row.get("claim_class")) == "prior_v15_carry_forward" and status in {
            "flagged",
            "blocked",
            "gated-skipped",
            "missing",
        }:
            blocked.append(f"carry_forward_v15:{row.get('summary', {}).get('source_row_id')}:{status}")
    return blocked


def _recommended_next_actions(rows: list[dict[str, Any]]) -> list[str]:
    claim_rows = _claim_rows(rows)
    actions = [
        "Exp3003: rerun hard-set repair only after removing tautology flags, reducing syntax/schema failures, and preserving false_accept_delta <= 0 on metamorphic variants.",
        "Exp3007: rerun FR-11 stability with larger independent held-out metrics or reclassify as non-headline until the heldout-score tautology flag is cleared.",
        "Exp3008: add a host-visible GateMate transport by binding spin_out/done to board pins or a UART/GPIO/status-register path, then capture deterministic readback or smoke bytes.",
        "Exp3009: after Exp3008 reports host_visible_io_ready=true, emit RTL/PnR/resource evidence; while the gate is closed, write an explicit gate-skipped artifact.",
        "Paper-v6: keep no SOTA repair headline, no broad FR-11 self-learning claim, no GateMate/KV260 speedup claim, and no TSU/Kona access claim.",
    ]
    if claim_rows["exp3004_substrate_provenance"].get("status") == "clean":
        actions.append(
            "Exp3011: carry Exp3004 forward only as substrate-provenance repair; do not claim the live retry solved the BEAVER task."
        )
    return actions


def _paper_v6_boundary_summary(rows: list[dict[str, Any]], violations: list[dict[str, Any]]) -> dict[str, Any]:
    unsafe = [
        str(row.get("row_id"))
        for row in rows
        if row.get("status") in {"flagged", "blocked", "gated-skipped", "missing"}
        and not str(row.get("claim_class", "")).startswith("prior_v15")
    ]
    return {
        "forbidden_claims_absent": not violations,
        "not_promoted_statuses": ["flagged", "blocked", "gated-skipped", "pilot-only", "projection-only", "missing"],
        "current_non_promotable_rows": unsafe,
        "narrowing_preserved": [
            "no false SOTA repair headline",
            "no LLM-as-verifier authority",
            "no TSU/Z1/Kona access or parity claim",
            "no GateMate/KV260 speedup or thermodynamic claim",
        ],
    }


def _prd_openspec_boundary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claim_rows = _claim_rows(rows)
    return {
        "prd_refs": ["FR-11", "FR-12"],
        "openspec_refs": ["REQ-REPORT-3010", "SCENARIO-REPORT-3010"],
        "repair_status": claim_rows["exp3003_repair"].get("status", "missing"),
        "substrate_provenance_status": claim_rows["exp3004_substrate_provenance"].get("status", "missing"),
        "fr11_status": claim_rows["exp3007_fr11_stability"].get("status", "missing"),
        "hardware_status": claim_rows["exp3008_gatemate_io"].get("status", "missing"),
    }


def _hardware_boundary_summary(rows: list[dict[str, Any]], violations: list[dict[str, Any]]) -> dict[str, Any]:
    claim_rows = _claim_rows(rows)
    return {
        "forbidden_claims_absent": not violations,
        "gatemate": {
            "status": claim_rows["exp3008_gatemate_io"].get("status", "missing"),
            "summary": claim_rows["exp3008_gatemate_io"].get("summary", {}),
        },
        "ssqa": {
            "status": claim_rows["exp3009_ssqa"].get("status", "missing"),
            "summary": claim_rows["exp3009_ssqa"].get("summary", {}),
        },
        "boundary": "host-visible IO and RTL/PnR/resource only; no speedup, sampler, or thermodynamic claim",
    }


def _roadmap_acceptance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claim_rows = _claim_rows(rows)
    return {
        "exp3003_repair_promotable": claim_rows["exp3003_repair"].get("status") == "clean",
        "exp3004_substrate_promotable": claim_rows["exp3004_substrate_provenance"].get("status") == "clean",
        "exp3007_fr11_promotable": claim_rows["exp3007_fr11_stability"].get("status") == "clean",
        "exp3008_gatemate_io_promotable": claim_rows["exp3008_gatemate_io"].get("status") == "clean",
        "exp3009_ssqa_promotable": claim_rows["exp3009_ssqa"].get("status") == "clean",
    }


def _complete_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v16_ready=true; "
        f"clean={artifact['clean_count']}; "
        f"flagged={artifact['flagged_count']}; "
        f"blocked={artifact['blocked_count']}; "
        f"gated_skipped={artifact['gated_skipped_count']}; "
        f"projection_only={artifact['projection_only_count']}; "
        f"pilot_only={artifact['pilot_only_count']}; "
        f"missing={artifact['missing_count']}"
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
