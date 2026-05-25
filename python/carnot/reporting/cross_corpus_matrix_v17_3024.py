"""Build the Exp 3024 cross-corpus matrix v17 artifact.

Spec refs: REQ-REPORT-3024, SCENARIO-REPORT-3024.

This module is an aggregation-only closeout layer for milestone .283. It reads
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
RUN_DATE = "20260525"
MILESTONE = "2026.05.283"
SCHEMA = "carnot.cross_corpus_matrix.v17_283_claim_boundary.v1"
ARTIFACT = "experiment_3024_cross_corpus_matrix_v17"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3024_cross_corpus_matrix_v17.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3024_cross_corpus_matrix_v17.py"

MATRIX_V16_REL_PATH = Path("results/experiment_3010_cross_corpus_matrix_v16.json")
CAPSTONE_V282_REL_PATH = Path("results/experiment_3011_capstone_v282.json")
EXP3012_REL_PATH = Path("results/experiment_3012_archive_v282_activate_v283.json")
EXP3013_REL_PATH = Path("results/experiment_3013_sota_gguf_logprob_telemetry_preflight_v1.json")
EXP3014_REL_PATH = Path("results/experiment_3014_repair_syntax_schema_failure_taxonomy_v1.json")
EXP3015_REL_PATH = Path("results/experiment_3015_cactus_style_repair_acceptance_controller_v1.json")
EXP3016_REL_PATH = Path("results/experiment_3016_sota_repair_rerun_with_acceptance_controller_v1.json")
EXP3017_REL_PATH = Path("results/experiment_3017_nsvif_instruction_validator_tree_expansion_v1.json")
EXP3018_REL_PATH = Path("results/experiment_3018_beaver_style_validator_frontier_certificate_v1.json")
EXP3019_REL_PATH = Path("results/experiment_3019_fr11_feasibility_channel_de_tautology_diagnostic_v1.json")
EXP3020_REL_PATH = Path("results/experiment_3020_dvi_verifier_feedback_self_learning_controller_v1.json")
EXP3021_REL_PATH = Path("results/experiment_3021_gatemate_rtl_ccf_host_visible_transport_shim_v1.json")
EXP3022_REL_PATH = Path("results/experiment_3022_gatemate_transport_flash_smoke_v3.json")
EXP3023_REL_PATH = Path("results/experiment_3023_ssqa_explicit_gate_artifact_and_rtl_report_v1.json")

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
    "sampler_claim_made",
    "speedup_claim_made",
    "hardware_speedup_claim_made",
    "same_basis_cpu_fpga_speedup_claim_made",
    "fpga_acceleration_claim_made",
    "thermodynamic_claim_made",
    "thermalization_claim_made",
    "boltzmann_claim_made",
    "boltzmann_thermalization_claim_made",
    "tsu_access_claim_made",
    "z1_access_claim_made",
    "xtr0_access_claim_made",
    "kona_access_claim_made",
    "kona_parity_claim_made",
    "extropic_execution_claim_made",
    "photonic_execution_claim_made",
)

LLM_VERIFIER_FIELDS = (
    "llm_judge_used",
    "live_llm_judge_used",
    "llm_as_verifier_claim_made",
    "black_box_judge_used",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3010", MATRIX_V16_REL_PATH, required=True),
    SourceSpec("exp3011", CAPSTONE_V282_REL_PATH, required=True),
    SourceSpec("exp3012", EXP3012_REL_PATH),
    SourceSpec("exp3013", EXP3013_REL_PATH),
    SourceSpec("exp3014", EXP3014_REL_PATH),
    SourceSpec("exp3015", EXP3015_REL_PATH),
    SourceSpec("exp3016", EXP3016_REL_PATH),
    SourceSpec("exp3017", EXP3017_REL_PATH),
    SourceSpec("exp3018", EXP3018_REL_PATH),
    SourceSpec("exp3019", EXP3019_REL_PATH),
    SourceSpec("exp3020", EXP3020_REL_PATH),
    SourceSpec("exp3021", EXP3021_REL_PATH),
    SourceSpec("exp3022", EXP3022_REL_PATH),
    SourceSpec("exp3023", EXP3023_REL_PATH),
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

    Why this is forgiving: the matrix is a closeout ledger. A malformed or
    absent input is itself evidence and must appear as a missing row rather than
    crashing the whole milestone aggregation.
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
    """REQ-REPORT-3024: build matrix v17 from upstream artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    source_artifacts = _source_artifacts_read(root_path, payloads)
    document_inputs = _document_inputs_read(root_path)
    rows = [*_prior_v16_rows(payloads.get("exp3010", {})), *_v17_rows(payloads)]
    counts = _status_counts(rows)
    violations = _claim_boundary_violations(rows)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_v17_ready": False,
        "honest_verdict": "blocked_required_upstream_missing",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts_read": source_artifacts,
        "source_checksums": _source_checksums(source_artifacts),
        "document_inputs_read": document_inputs,
        "missing_artifacts": _missing_artifacts(source_artifacts),
        "missing_documents": _missing_documents(document_inputs),
        "cited_upstream_artifacts": _cited_upstream_artifacts(source_artifacts, payloads),
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
        "repaired_claims": _repaired_claims(rows, payloads.get("exp3010", {})),
        "still_blocked_claims": _still_blocked_claims(rows, payloads.get("exp3010", {})),
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

    artifact["matrix_v17_ready"] = True
    artifact["honest_verdict"] = _complete_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3024 deliverable JSON."""

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


def _missing_documents(document_inputs: list[dict[str, Any]]) -> list[str]:
    return [str(row["path"]) for row in document_inputs if row.get("present") is not True]


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


def _cited_upstream_artifacts(
    source_artifacts: list[dict[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for source in source_artifacts:
        exp_id = str(source["experiment_id"])
        payload = payloads.get(exp_id, {})
        citation = {
            "experiment_id": exp_id,
            "path": source["path"],
            "present": source["present"],
            "readable_json_object": source["readable_json_object"],
            "sha256": source["sha256"],
            "honest_verdict": str(payload.get("honest_verdict", "")),
            "inference_substrate": str(payload.get("inference_substrate", "")),
        }
        model_provenance = _source_model_provenance(payload)
        hardware_provenance = _source_hardware_provenance(payload)
        gate_provenance = _source_gate_provenance(payload)
        if model_provenance:
            citation["model_provenance"] = model_provenance
        if hardware_provenance:
            citation["hardware_provenance"] = hardware_provenance
        if gate_provenance:
            citation["gate_provenance"] = gate_provenance
        citations.append(citation)
    return citations


def _source_model_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "model_specs": payload.get("model_specs"),
        "headline_models_used": payload.get("headline_models_used"),
        "headline_models_available": payload.get("headline_models_available"),
        "model_checksums": payload.get("model_checksums"),
        "live_transcript_paths": payload.get("live_transcript_paths"),
        "cache_paths": payload.get("cache_paths"),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _source_hardware_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "board_detected": payload.get("board_detected"),
        "host_visible_io_ready": payload.get("host_visible_io_ready"),
        "gatemate_transport_rtl_ready": payload.get("gatemate_transport_rtl_ready"),
        "host_visible_io_plan_ready": payload.get("host_visible_io_plan_ready"),
        "io_transport_path": payload.get("io_transport_path"),
        "rtl_paths": payload.get("rtl_paths"),
        "ccf_paths": payload.get("ccf_paths"),
        "rtl_path": payload.get("rtl_path"),
        "pnr_report_path": payload.get("pnr_report_path"),
        "resource_report_path": payload.get("resource_report_path"),
        "smoke_hook_paths": payload.get("smoke_hook_paths"),
        "transcript_paths": payload.get("transcript_paths"),
        "observed_output_hash": payload.get("observed_output_hash"),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _source_gate_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "status": payload.get("status"),
        "blocked_at_layer": payload.get("blocked_at_layer"),
        "gate_check_summary": payload.get("gate_check_summary"),
        "gates_evaluated": payload.get("gates_evaluated"),
        "ssqa_gate_status": payload.get("ssqa_gate_status"),
        "upstream_status": payload.get("upstream_status"),
        "upstream_gate_check_summary": payload.get("upstream_gate_check_summary"),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _prior_v16_rows(matrix_v16: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in matrix_v16.get("rows") if isinstance(matrix_v16.get("rows"), list) else []:
        if not isinstance(source, Mapping):
            continue
        status = str(source.get("status") or "missing")
        if status not in STATUSES:
            status = "missing"
        source_row_id = str(source.get("row_id") or "unknown")
        rows.append(
            _row(
                row_id=f"carry_forward_v16:{source_row_id}",
                source_experiment_id=str(source.get("source_experiment_id") or "exp3010"),
                status=status,
                claim_class="prior_v16_carry_forward",
                evidence_type="matrix_v16_row",
                inference_substrate=INFERENCE_SUBSTRATE,
                claim_boundary="Matrix v16 row carried forward without metric recomputation or claim promotion.",
                source_honest_verdict=str(
                    source.get("source_honest_verdict") or matrix_v16.get("honest_verdict") or ""
                ),
                summary={
                    "source_matrix": "v16",
                    "source_row_id": source_row_id,
                    "source_status": status,
                    "source_claim_class": str(source.get("claim_class") or ""),
                },
            )
        )
    return rows


def _v17_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _exp3012_row(payloads.get("exp3012", {})),
        _exp3013_row(payloads.get("exp3013", {})),
        _exp3014_row(payloads.get("exp3014", {})),
        _exp3015_row(payloads.get("exp3015", {})),
        _exp3016_row(payloads.get("exp3016", {})),
        _exp3017_row(payloads.get("exp3017", {})),
        _exp3018_row(payloads.get("exp3018", {})),
        _exp3019_row(payloads.get("exp3019", {})),
        _exp3020_row(payloads.get("exp3020", {})),
        _exp3021_row(payloads.get("exp3021", {})),
        _exp3022_row(payloads.get("exp3022", {})),
        _exp3023_row(payloads.get("exp3023", {}), payloads.get("exp3022", {})),
    ]


def _exp3012_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = payload.get("archive_ready") is True and payload.get("activated_milestone") == MILESTONE
    status = "projection-only" if ready else _status_with_guards("blocked", payload, [])
    return _artifact_row(
        "exp3012_archive_activation",
        "exp3012",
        status,
        "archive_activation",
        "aggregation_only_archive_state",
        "Archive/activation is milestone bookkeeping and not paper-v6 claim evidence.",
        payload,
        {
            "archive_ready": bool(payload.get("archive_ready")),
            "activated_milestone": str(payload.get("activated_milestone") or ""),
            "status_updates_written": bool(payload.get("status_updates_written")),
            "research_complete_updated": bool(payload.get("research_complete_updated")),
        },
        [],
    )


def _exp3013_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _claim_violations("exp3013_sota_logprob_telemetry", payload)
    clean = (
        payload.get("sota_headline_ready") is True
        and payload.get("sota_logprob_ready") is True
        and payload.get("preconditions_checked") is True
        and bool(_string_list(payload.get("live_transcript_paths")))
        and payload.get("legacy_smoke_only_used") is not True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, violations)
    return _artifact_row(
        "exp3013_sota_logprob_telemetry",
        "exp3013",
        status,
        "sota_telemetry_provenance",
        "live_llm_telemetry_source",
        "SOTA telemetry is a source gate; model identity is cited only under upstream provenance.",
        payload,
        {
            "sota_headline_ready": bool(payload.get("sota_headline_ready")),
            "sota_logprob_ready": bool(payload.get("sota_logprob_ready")),
            "n_live_transcripts": len(_string_list(payload.get("live_transcript_paths"))),
            "legacy_smoke_only_used": bool(payload.get("legacy_smoke_only_used")),
            "model_boundary_status": _model_boundary_status(payload),
        },
        violations,
    )


def _exp3014_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("repair_failure_taxonomy_ready") is True
        and _int_or(payload.get("n_cached_candidates_audited"), 0) > 0
        and payload.get("halluguard_ntk_claim_made") is not True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, [])
    return _artifact_row(
        "exp3014_repair_failure_taxonomy",
        "exp3014",
        status,
        "repair_taxonomy",
        "deterministic_cached_replay",
        "Repair taxonomy is diagnostic support and cannot be promoted as live repair quality.",
        payload,
        {
            "repair_failure_taxonomy_ready": bool(payload.get("repair_failure_taxonomy_ready")),
            "n_cached_candidates_audited": _coerce_int(payload.get("n_cached_candidates_audited")),
            "syntax_failure_count": _coerce_int(payload.get("syntax_failure_count")),
            "schema_failure_count": _coerce_int(payload.get("schema_failure_count")),
            "false_accept_count": _coerce_int(payload.get("false_accept_count")),
            "tautology_failure_count": _coerce_int(payload.get("tautology_failure_count")),
            "halluguard_ntk_claim_made": bool(payload.get("halluguard_ntk_claim_made")),
        },
        [],
    )


def _exp3015_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _claim_violations("exp3015_acceptance_controller", payload)
    clean = (
        payload.get("acceptance_controller_ready") is True
        and _int_or(payload.get("n_candidates_evaluated"), 0) > 0
        and _float_or(payload.get("false_accept_delta_offline"), 0.0) <= 0.0
        and _float_or(payload.get("syntax_failure_delta_offline"), 1.0) <= 0.0
        and _float_or(payload.get("schema_failure_delta_offline"), 1.0) <= 0.0
        and payload.get("llm_judge_used") is not True
        and payload.get("black_box_judge_used") is not True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, violations)
    return _artifact_row(
        "exp3015_acceptance_controller",
        "exp3015",
        status,
        "repair_acceptance_controller",
        "deterministic_cached_replay",
        "Acceptance-controller readiness is offline support; adversarial flags block promotion.",
        payload,
        {
            "acceptance_controller_ready": bool(payload.get("acceptance_controller_ready")),
            "n_candidates_evaluated": _coerce_int(payload.get("n_candidates_evaluated")),
            "false_accept_delta_offline": _coerce_float(payload.get("false_accept_delta_offline")),
            "syntax_failure_delta_offline": _coerce_float(payload.get("syntax_failure_delta_offline")),
            "schema_failure_delta_offline": _coerce_float(payload.get("schema_failure_delta_offline")),
            "pass_at_1_delta_offline": _coerce_float(payload.get("pass_at_1_delta_offline")),
            "llm_judge_used": bool(payload.get("llm_judge_used")),
        },
        violations,
    )


def _exp3016_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _claim_violations("exp3016_repair_acceptance_controller", payload)
    clean = (
        payload.get("repair_controller_clean") is True
        and payload.get("headline_result") is True
        and payload.get("preconditions_checked") is True
        and _int_or(payload.get("n_tasks"), 0) >= 20
        and _int_or(payload.get("n_metamorphic_variants"), 0) > 0
        and bool(_string_list(payload.get("live_transcript_paths")))
        and bool(_string_list(payload.get("verifier_log_paths")))
        and _float_or(payload.get("pass_at_1_delta"), 0.0) > 0.0
        and _float_or(payload.get("pass_at_k_delta"), 0.0) > 0.0
        and _float_or(payload.get("false_accept_delta"), 0.0) <= 0.0
        and _float_or(payload.get("syntax_failure_rate_delta"), 1.0) <= 0.0
        and _float_or(payload.get("schema_failure_rate_delta"), 1.0) <= 0.0
        and payload.get("tautology_gate_clean") is True
    )
    status = _status_with_guards("clean" if clean else "flagged", payload, violations)
    return _artifact_row(
        "exp3016_repair_acceptance_controller",
        "exp3016",
        status,
        "repair_eval",
        "live_llm_repair_source",
        "Repair promotion requires clean deltas and clean adversarial/methodology flags.",
        payload,
        {
            "repair_controller_clean": bool(payload.get("repair_controller_clean")),
            "headline_result": bool(payload.get("headline_result")),
            "n_tasks": _coerce_int(payload.get("n_tasks")),
            "n_metamorphic_variants": _coerce_int(payload.get("n_metamorphic_variants")),
            "pass_at_1_delta": _coerce_float(payload.get("pass_at_1_delta")),
            "pass_at_k_delta": _coerce_float(payload.get("pass_at_k_delta")),
            "false_accept_delta": _coerce_float(payload.get("false_accept_delta")),
            "syntax_failure_rate_delta": _coerce_float(payload.get("syntax_failure_rate_delta")),
            "schema_failure_rate_delta": _coerce_float(payload.get("schema_failure_rate_delta")),
            "tautology_gate_clean": bool(payload.get("tautology_gate_clean")),
            "n_live_transcripts": len(_string_list(payload.get("live_transcript_paths"))),
            "n_verifier_logs": len(_string_list(payload.get("verifier_log_paths"))),
            "model_boundary_status": _model_boundary_status(payload),
        },
        violations,
    )


def _exp3017_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _claim_violations("exp3017_instruction_validator_tree", payload)
    clean = (
        payload.get("instruction_validator_tree_ready") is True
        and _int_or(payload.get("n_instruction_items"), 0) >= 20
        and _int_or(payload.get("n_validator_trees"), 0) >= 20
        and _float_or(payload.get("exact_check_coverage"), 0.0) >= 0.95
        and payload.get("all_authoritative_nodes_exact_checked") is True
        and payload.get("llm_judge_used") is not True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, violations)
    return _artifact_row(
        "exp3017_instruction_validator_tree",
        "exp3017",
        status,
        "validator_tree_expansion",
        "deterministic_instruction_validator_tree",
        "Instruction validators are promotable only when authoritative nodes are exact-checked.",
        payload,
        {
            "instruction_validator_tree_ready": bool(payload.get("instruction_validator_tree_ready")),
            "n_instruction_items": _coerce_int(payload.get("n_instruction_items")),
            "n_validator_trees": _coerce_int(payload.get("n_validator_trees")),
            "exact_check_coverage": _coerce_float(payload.get("exact_check_coverage")),
            "all_authoritative_nodes_exact_checked": bool(payload.get("all_authoritative_nodes_exact_checked")),
            "llm_judge_used": bool(payload.get("llm_judge_used")),
        },
        violations,
    )


def _exp3018_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("frontier_certificate_ready") is True
        and _int_or(payload.get("n_frontier_items"), 0) > 0
        and _int_or(payload.get("n_prefix_closed_items"), 0) > 0
        and payload.get("enumerator_fallback_separated") is True
        and payload.get("live_llm_evidence_used") is not True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, [])
    return _artifact_row(
        "exp3018_beaver_frontier_certificate",
        "exp3018",
        status,
        "validator_frontier_certificate",
        "deterministic_cached_validator_frontier",
        "BEAVER-style frontier evidence is exact/cached and does not promote live retry success.",
        payload,
        {
            "frontier_certificate_ready": bool(payload.get("frontier_certificate_ready")),
            "n_frontier_items": _coerce_int(payload.get("n_frontier_items")),
            "n_prefix_closed_items": _coerce_int(payload.get("n_prefix_closed_items")),
            "certified_safe_count": _coerce_int(payload.get("certified_safe_count")),
            "certified_violating_count": _coerce_int(payload.get("certified_violating_count")),
            "unresolved_count": _coerce_int(payload.get("unresolved_count")),
            "enumerator_fallback_separated": bool(payload.get("enumerator_fallback_separated")),
            "live_llm_evidence_used": bool(payload.get("live_llm_evidence_used")),
        },
        [],
    )


def _exp3019_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("feasibility_channel_diagnostic_ready") is True
        and _int_or(payload.get("n_rows"), 0) > 0
        and _float_or(payload.get("negative_control_rejection_rate"), 0.0) >= 1.0
        and payload.get("tautology_risk_flag") is not True
        and payload.get("reused_label_as_feature") is not True
        and payload.get("native_dsp_claim_made") is not True
    )
    status = _status_with_guards("clean" if clean else "flagged", payload, [])
    return _artifact_row(
        "exp3019_fr11_feasibility_channel",
        "exp3019",
        status,
        "fr11_feasibility_diagnostic",
        "cached_exact_validator_certificate_trace_replay",
        "Feasibility diagnostics remain non-promotable while tautology risk is flagged.",
        payload,
        {
            "feasibility_channel_diagnostic_ready": bool(payload.get("feasibility_channel_diagnostic_ready")),
            "n_rows": _coerce_int(payload.get("n_rows")),
            "feasible_infeasible_auc": _coerce_float(payload.get("feasible_infeasible_auc")),
            "negative_control_rejection_rate": _coerce_float(payload.get("negative_control_rejection_rate")),
            "heldout_metric_correlation": _coerce_float(payload.get("heldout_metric_correlation")),
            "tautology_risk_flag": bool(payload.get("tautology_risk_flag")),
            "reused_label_as_feature": bool(payload.get("reused_label_as_feature")),
        },
        [],
    )


def _exp3020_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    boundary = _self_learning_boundary(payload)
    clean = (
        payload.get("verifier_feedback_controller_ready") is True
        and payload.get("continuous_self_learning_task") is True
        and _int_or(payload.get("n_replay_items"), 0) > 0
        and boundary.get("status") == "clean"
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, [])
    return _artifact_row(
        "exp3020_fr11_verifier_feedback_controller",
        "exp3020",
        status,
        "fr11_self_learning_controller",
        "cached_exact_trace_replay_controller_only",
        "FR-11 is bounded to verifier-feedback controller utility, not native LLM training.",
        payload,
        {
            "verifier_feedback_controller_ready": bool(payload.get("verifier_feedback_controller_ready")),
            "continuous_self_learning_task": bool(payload.get("continuous_self_learning_task")),
            "independent_self_learning_boundary_preserved": bool(
                payload.get("independent_self_learning_boundary_preserved")
            ),
            "n_replay_items": _coerce_int(payload.get("n_replay_items")),
            "heldout_delta": _coerce_float(payload.get("heldout_delta")),
            "negative_control_delta": _coerce_float(payload.get("negative_control_delta")),
            "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
            "drift_guard_passed": bool(payload.get("drift_guard_passed")),
            "tautology_risk_flag": bool(payload.get("tautology_risk_flag")),
        },
        [],
    )


def _exp3021_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _claim_violations("exp3021_gatemate_transport_shim", payload)
    clean = (
        payload.get("gatemate_transport_rtl_ready") is True
        and payload.get("host_visible_io_plan_ready") is True
        and payload.get("preconditions_checked") is True
        and payload.get("simulation_or_lint_passed") is True
    )
    status = _status_with_guards("clean" if clean else "blocked", payload, violations)
    return _artifact_row(
        "exp3021_gatemate_transport_shim",
        "exp3021",
        status,
        "gatemate_transport",
        "hardware_transport_preflight",
        "GateMate transport is blocked until done/spin_out/status has a host-visible path.",
        payload,
        {
            "gatemate_transport_rtl_ready": bool(payload.get("gatemate_transport_rtl_ready")),
            "host_visible_io_plan_ready": bool(payload.get("host_visible_io_plan_ready")),
            "preconditions_checked": bool(payload.get("preconditions_checked")),
            "board_detected": bool(payload.get("board_detected")),
            "simulation_or_lint_passed": bool(payload.get("simulation_or_lint_passed")),
            "pnr_or_synthesis_attempted": bool(payload.get("pnr_or_synthesis_attempted")),
            "io_transport_status": _blocked_or_ready_path(payload.get("io_transport_path")),
        },
        violations,
    )


def _exp3022_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _claim_violations("exp3022_gatemate_transport_flash_smoke", payload)
    if _structured_gate_failed(payload):
        status = "gated-skipped"
    else:
        clean = payload.get("host_visible_io_ready") is True and payload.get("smoke_vector_passed") is True
        status = "clean" if clean else "blocked"
    status = _status_with_guards(status, payload, violations)
    return _artifact_row(
        "exp3022_gatemate_transport_flash_smoke",
        "exp3022",
        status,
        "gatemate_host_visible_io",
        "hardware_smoke",
        "GateMate IO remains gated until transport readiness permits board smoke.",
        payload,
        {
            "structured_gate_failed": _structured_gate_failed(payload),
            "status": str(payload.get("status") or ""),
            "blocked_at_layer": str(payload.get("blocked_at_layer") or ""),
            "host_visible_io_ready": bool(payload.get("host_visible_io_ready")),
            "board_detected": bool(payload.get("board_detected")),
            "flash_attempted": bool(payload.get("flash_attempted")),
            "flash_succeeded": bool(payload.get("flash_succeeded")),
            "smoke_vector_attempted": bool(payload.get("smoke_vector_attempted")),
            "smoke_vector_passed": bool(payload.get("smoke_vector_passed")),
            "observed_output_hash_present": bool(payload.get("observed_output_hash")),
        },
        violations,
    )


def _exp3023_row(payload: Mapping[str, Any], exp3022: Mapping[str, Any]) -> dict[str, Any]:
    violations = _claim_violations("exp3023_ssqa_explicit_gate_artifact", payload)
    if payload.get("ssqa_gate_status") == "gate_skipped" or payload.get("upstream_host_visible_io_ready") is False:
        status = "gated-skipped"
    elif payload.get("ssqa_rtl_pnr_report_ready") is True:
        status = "clean"
    elif payload.get("projection_only") is True:
        status = "projection-only"
    else:
        status = "blocked"
    status = _status_with_guards(status, payload, violations)
    return _artifact_row(
        "exp3023_ssqa_explicit_gate_artifact",
        "exp3023",
        status,
        "ssqa_gate_artifact",
        "hardware_gate_artifact",
        "SSQA artifact presence is repaired, but SSQA remains gated until GateMate IO is host-visible.",
        payload,
        {
            "ssqa_artifact_written": bool(payload.get("ssqa_artifact_written")),
            "ssqa_gate_status": str(payload.get("ssqa_gate_status") or ""),
            "ssqa_rtl_pnr_report_ready": bool(payload.get("ssqa_rtl_pnr_report_ready")),
            "upstream_host_visible_io_ready": bool(payload.get("upstream_host_visible_io_ready")),
            "upstream_status": str(exp3022.get("status") or payload.get("upstream_status") or ""),
            "projection_only": bool(payload.get("projection_only")),
            "resource_report_present": bool(payload.get("resource_report_path")),
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
    if violations or _has_flags(payload) or _flagged_verdict(verdict):
        return "flagged"
    if _gated_verdict(verdict) or _structured_gate_failed(payload):
        return "gated-skipped"
    if _blocked_verdict(verdict):
        return "blocked"
    return base_status


def _blocked_verdict(verdict: object) -> bool:
    if not isinstance(verdict, str):
        return False
    lowered = verdict.strip().lower()
    return lowered.startswith("blocked") or lowered.startswith("complete: blocked_")


def _gated_verdict(verdict: object) -> bool:
    if not isinstance(verdict, str):
        return False
    lowered = verdict.strip().lower()
    return lowered.startswith(("gate_blocked", "gated", "skip", "blocked_gate"))


def _flagged_verdict(verdict: object) -> bool:
    if not isinstance(verdict, str):
        return False
    lowered = verdict.strip().lower()
    return lowered.startswith("flagged") or lowered.startswith("complete_flagged") or ": flagged" in lowered


def _structured_gate_failed(payload: Mapping[str, Any]) -> bool:
    gates = payload.get("gates_evaluated")
    return isinstance(gates, list) and any(isinstance(gate, Mapping) and gate.get("passed") is False for gate in gates)


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


def _model_boundary_status(payload: Mapping[str, Any]) -> str:
    headline = _headline_model_ids(payload)
    used = _models_used(payload)
    if not headline:
        return "not_applicable"
    if not set(headline).intersection(used):
        return "non_compliant_missing_mandated_model"
    if _has_flags(payload) or _flagged_verdict(payload.get("honest_verdict")):
        return "flagged_mandated_model_evidence"
    return "compliant"


def _headline_model_ids(payload: Mapping[str, Any]) -> list[str]:
    model_specs = _mapping(payload.get("model_specs"))
    return (
        _string_list(payload.get("headline_model_ids"))
        or _string_list(payload.get("headline_models"))
        or _string_list(payload.get("mandatory_headline_model_ids"))
        or _string_list(model_specs.get("headline_models"))
    )


def _models_used(payload: Mapping[str, Any]) -> list[str]:
    available = payload.get("headline_models_available") or payload.get("sota_models_available")
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
    if payload.get("inference_substrate") in {"hardware_smoke", "physical_gatemate_board", "hardware_gate_artifact"}:
        clean = payload.get("host_visible_io_ready") is True and (
            bool(payload.get("observed_output_hash")) or payload.get("smoke_vector_passed") is True
        )
        return {"status": "clean" if clean else "blocked"}
    return {"status": "not_applicable"}


def _self_learning_boundary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("continuous_self_learning_task") is not True:
        return {"status": "not_applicable"}
    clean = (
        payload.get("independent_self_learning_boundary_preserved") is True
        and _float_or(payload.get("heldout_delta"), 0.0) > 0.0
        and _float_or(payload.get("negative_control_delta"), 1.0) <= 0.0
        and payload.get("forgetting_guard_passed") is True
        and payload.get("drift_guard_passed") is True
        and payload.get("tautology_risk_flag") is not True
        and payload.get("native_llm_training_claim_made") is not True
    )
    return {
        "status": "clean" if clean else "blocked",
        "boundary": "verifier-feedback controller over exact traces only; no broad autonomous self-improvement claim",
        "continuous_self_learning_task": True,
        "heldout_delta": _coerce_float(payload.get("heldout_delta")),
        "negative_control_delta": _coerce_float(payload.get("negative_control_delta")),
        "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
        "drift_guard_passed": bool(payload.get("drift_guard_passed")),
        "tautology_risk_flag": bool(payload.get("tautology_risk_flag")),
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
    headline_claim = payload.get("headline_result") is True or payload.get("sota_headline_ready") is True
    if headline_claim and _model_boundary_status(payload) == "non_compliant_missing_mandated_model":
        violations.append(
            {
                "row_id": row_id,
                "violation": "false_sota_headline_use",
                "fields": ["headline_result_or_sota_headline_ready"],
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
        "exp3013_sota_telemetry": dict(by_id.get("exp3013_sota_logprob_telemetry", {})),
        "exp3016_repair": dict(by_id.get("exp3016_repair_acceptance_controller", {})),
        "exp3020_fr11_self_learning": dict(by_id.get("exp3020_fr11_verifier_feedback_controller", {})),
        "exp3021_gatemate_transport": dict(by_id.get("exp3021_gatemate_transport_shim", {})),
        "exp3022_gatemate_io": dict(by_id.get("exp3022_gatemate_transport_flash_smoke", {})),
        "exp3023_ssqa": dict(by_id.get("exp3023_ssqa_explicit_gate_artifact", {})),
    }


def _repaired_claims(rows: list[dict[str, Any]], matrix_v16: Mapping[str, Any]) -> list[str]:
    claim_rows = _claim_rows(rows)
    repaired = _string_list(matrix_v16.get("repaired_claims"))
    if claim_rows["exp3013_sota_telemetry"].get("status") == "clean":
        repaired.append("exp3013_sota_logprob_telemetry_ready")
    if claim_rows["exp3016_repair"].get("status") == "clean":
        repaired.append("exp3016_repair_acceptance_controller_promotable")
    if claim_rows["exp3020_fr11_self_learning"].get("status") == "clean":
        repaired.append("exp3020_fr11_verifier_feedback_controller")
    if claim_rows["exp3023_ssqa"].get("summary", {}).get("ssqa_artifact_written") is True:
        repaired.append("exp3023_ssqa_artifact_presence_repaired_gate_skipped_not_promotable")
    return _unique_strings(repaired)


def _still_blocked_claims(rows: list[dict[str, Any]], matrix_v16: Mapping[str, Any]) -> list[str]:
    blocked = _string_list(matrix_v16.get("still_blocked_claims"))
    by_id = {str(row.get("row_id")): row for row in rows}
    labels = {
        "exp3014_repair_failure_taxonomy": "exp3014_repair_failure_taxonomy_flagged",
        "exp3015_acceptance_controller": "exp3015_acceptance_controller_flagged",
        "exp3016_repair_acceptance_controller": "exp3016_repair_acceptance_controller_flagged",
        "exp3018_beaver_frontier_certificate": "exp3018_frontier_certificate_flagged",
        "exp3019_fr11_feasibility_channel": "exp3019_fr11_feasibility_channel_flagged",
        "exp3021_gatemate_transport_shim": "exp3021_gatemate_transport_pinout_missing_blocked",
        "exp3022_gatemate_transport_flash_smoke": "exp3022_gatemate_transport_flash_smoke_gated_skipped",
        "exp3023_ssqa_explicit_gate_artifact": "exp3023_ssqa_gate_skipped_until_host_visible_io_ready",
    }
    for row_id, label in labels.items():
        if by_id.get(row_id, {}).get("status") in {"flagged", "blocked", "gated-skipped", "missing"}:
            blocked.append(label)
    for row in rows:
        status = str(row.get("status"))
        if str(row.get("claim_class")) == "prior_v16_carry_forward" and status in {
            "flagged",
            "blocked",
            "gated-skipped",
            "missing",
        }:
            blocked.append(f"carry_forward_v16:{row.get('summary', {}).get('source_row_id')}:{status}")
    return _unique_strings(blocked)


def _recommended_next_actions(rows: list[dict[str, Any]]) -> list[str]:
    claim_rows = _claim_rows(rows)
    actions = [
        "Exp3016: do not promote the repair row until adversarial TAUTOLOGY/METHODOLOGY flags are cleared or explained by a dedicated corrigendum while preserving positive repair deltas.",
        "Exp3020: carry FR-11 forward only as verifier-feedback controller utility over exact cached traces; do not claim native LLM fine-tuning or broad autonomous self-improvement.",
        "Exp3021/3022: obtain a physical GateMate pinout or supported host-visible transport for done/spin_out/status, then rerun flash/smoke and capture deterministic output bytes.",
        "Exp3023: keep SSQA as explicit gate-skipped evidence until Exp3022 reports host_visible_io_ready=true, then produce bounded RTL/PnR/resource evidence without speedup claims.",
        "Paper-v6: keep no false SOTA headline, no LLM-as-verifier authority, no TSU/Kona access claim, and no GateMate/KV260 speedup, sampler, Boltzmann, or thermodynamic claim.",
    ]
    if claim_rows["exp3020_fr11_self_learning"].get("status") == "clean":
        actions.append("Exp3025: treat Exp3020 as the only clean .283 promotion candidate among repair/FR-11/GateMate/SSQA target claims.")
    return actions


def _paper_v6_boundary_summary(rows: list[dict[str, Any]], violations: list[dict[str, Any]]) -> dict[str, Any]:
    unsafe = [
        str(row.get("row_id"))
        for row in rows
        if row.get("status") in {"flagged", "blocked", "gated-skipped", "pilot-only", "projection-only", "missing"}
        and not str(row.get("claim_class", "")).startswith("prior_v16")
    ]
    return {
        "forbidden_claims_absent": not violations,
        "not_promoted_statuses": ["flagged", "blocked", "gated-skipped", "pilot-only", "projection-only", "missing"],
        "current_non_promotable_rows": unsafe,
        "narrowing_preserved": [
            "no false SOTA repair headline",
            "no LLM-as-verifier authority",
            "no TSU/Z1/Kona access or parity claim",
            "no GateMate/KV260 speedup, sampler, Boltzmann, or thermodynamic claim",
        ],
    }


def _prd_openspec_boundary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claim_rows = _claim_rows(rows)
    return {
        "prd_refs": ["FR-11", "FR-12"],
        "openspec_refs": ["REQ-REPORT-3024", "SCENARIO-REPORT-3024", "REQ-CODE-3016", "REQ-LEARN-3020"],
        "repair_status": claim_rows["exp3016_repair"].get("status", "missing"),
        "fr11_status": claim_rows["exp3020_fr11_self_learning"].get("status", "missing"),
        "gatemate_status": claim_rows["exp3022_gatemate_io"].get("status", "missing"),
        "ssqa_status": claim_rows["exp3023_ssqa"].get("status", "missing"),
    }


def _hardware_boundary_summary(rows: list[dict[str, Any]], violations: list[dict[str, Any]]) -> dict[str, Any]:
    claim_rows = _claim_rows(rows)
    return {
        "forbidden_claims_absent": not violations,
        "transport": {
            "status": claim_rows["exp3021_gatemate_transport"].get("status", "missing"),
            "summary": claim_rows["exp3021_gatemate_transport"].get("summary", {}),
        },
        "gatemate_io": {
            "status": claim_rows["exp3022_gatemate_io"].get("status", "missing"),
            "summary": claim_rows["exp3022_gatemate_io"].get("summary", {}),
        },
        "ssqa": {
            "status": claim_rows["exp3023_ssqa"].get("status", "missing"),
            "summary": claim_rows["exp3023_ssqa"].get("summary", {}),
        },
        "boundary": "host-visible IO and RTL/PnR/resource only; no speedup, sampler, Boltzmann, or thermodynamic claim",
    }


def _roadmap_acceptance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claim_rows = _claim_rows(rows)
    return {
        "exp3016_repair_promotable": claim_rows["exp3016_repair"].get("status") == "clean",
        "exp3020_fr11_promotable": claim_rows["exp3020_fr11_self_learning"].get("status") == "clean",
        "exp3022_gatemate_io_promotable": claim_rows["exp3022_gatemate_io"].get("status") == "clean",
        "exp3023_ssqa_promotable": claim_rows["exp3023_ssqa"].get("status") == "clean",
        "aggregation_metadata_clean": True,
    }


def _complete_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v17_ready=true; "
        f"clean={artifact['clean_count']}; "
        f"flagged={artifact['flagged_count']}; "
        f"blocked={artifact['blocked_count']}; "
        f"gated_skipped={artifact['gated_skipped_count']}; "
        f"projection_only={artifact['projection_only_count']}; "
        f"pilot_only={artifact['pilot_only_count']}; "
        f"missing={artifact['missing_count']}"
    )


def _blocked_or_ready_path(value: object) -> str:
    text = str(value or "")
    if text.startswith("blocked:"):
        return "blocked"
    return "ready" if text else ""


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


def _float_or(value: object, default: float) -> float:
    parsed = _coerce_float(value)
    return default if parsed is None else parsed


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or(value: object, default: int) -> int:
    parsed = _coerce_int(value)
    return default if parsed is None else parsed


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
