"""Build the Exp 3025 milestone .283 terminal capstone artifact.

Spec refs: REQ-REPORT-3025, SCENARIO-REPORT-3025.

This module is a go/no-go closeout ledger, not a new experiment. It reads
matrix v17 and checked-in upstream artifacts, then decides which .283 claims
are promotable while preserving the paper-v6, PRD, roadmap, and hardware claim
boundaries recorded upstream.
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
SCHEMA = "carnot.milestone_capstone.v283_claim_repair_terminal_go_no_go.v1"
ARTIFACT = "experiment_3025_capstone_v283"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3025_capstone_v283.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3025_capstone_v283.py"

MATRIX_V17_REL_PATH = Path("results/experiment_3024_cross_corpus_matrix_v17.json")
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

NEXT_MILESTONE_RECOMMENDATION = (
    "2026.05.284: repair-corrigendum-and-gatemate-output-contract - clear Exp3016 "
    "TAUTOLOGY/METHODOLOGY flags before promoting repair, preserve Exp3020 only as "
    "bounded FR-11 verifier-feedback controller evidence, obtain GateMate host-visible "
    "output before rerunning SSQA, and keep paper_ready=false until repair, hardware IO, "
    "SSQA, and carry-forward matrix blockers are clean."
)

STATUS_BUCKETS = (
    "clean",
    "flagged",
    "blocked",
    "gated-skipped",
    "pilot-only",
    "projection-only",
    "missing",
)

TASK_ORDER = (
    "exp3012",
    "exp3013",
    "exp3014",
    "exp3015",
    "exp3016",
    "exp3017",
    "exp3018",
    "exp3019",
    "exp3020",
    "exp3021",
    "exp3022",
    "exp3023",
    "exp3024",
)

TASK_ROW_IDS = {
    "exp3012": "exp3012_archive_activation",
    "exp3013": "exp3013_sota_logprob_telemetry",
    "exp3014": "exp3014_repair_failure_taxonomy",
    "exp3015": "exp3015_acceptance_controller",
    "exp3016": "exp3016_repair_acceptance_controller",
    "exp3017": "exp3017_instruction_validator_tree",
    "exp3018": "exp3018_beaver_frontier_certificate",
    "exp3019": "exp3019_fr11_feasibility_channel",
    "exp3020": "exp3020_fr11_verifier_feedback_controller",
    "exp3021": "exp3021_gatemate_transport_shim",
    "exp3022": "exp3022_gatemate_transport_flash_smoke",
    "exp3023": "exp3023_ssqa_explicit_gate_artifact",
    "exp3024": "exp3024_cross_corpus_matrix_v17",
}

EXP_SOURCE_PATHS = {
    "exp3012": EXP3012_REL_PATH,
    "exp3013": EXP3013_REL_PATH,
    "exp3014": EXP3014_REL_PATH,
    "exp3015": EXP3015_REL_PATH,
    "exp3016": EXP3016_REL_PATH,
    "exp3017": EXP3017_REL_PATH,
    "exp3018": EXP3018_REL_PATH,
    "exp3019": EXP3019_REL_PATH,
    "exp3020": EXP3020_REL_PATH,
    "exp3021": EXP3021_REL_PATH,
    "exp3022": EXP3022_REL_PATH,
    "exp3023": EXP3023_REL_PATH,
    "exp3024": MATRIX_V17_REL_PATH,
}

DOCUMENT_REL_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-roadmap.yaml"),
    Path("research-roadmap-next.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-complete.yaml"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
)


@dataclass(frozen=True)
class SourceSpec:
    """A local upstream artifact the capstone reads and checksums."""

    experiment_id: str
    path: Path
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3024", MATRIX_V17_REL_PATH, required=True),
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


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object, returning empty evidence when the file is unusable."""

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
    """REQ-REPORT-3025: synthesize the .283 go/no-go from matrix v17."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    matrix = payloads["exp3024"]
    source_artifacts = _source_artifacts_read(root_path, payloads)
    required_errors = _required_source_errors(payloads)
    rows = _matrix_rows(matrix)
    by_id = _row_by_id(rows)
    task_rows = [_task_row_from_matrix(exp_id, matrix, by_id) for exp_id in TASK_ORDER]
    task_counts = _counts(task_rows)
    matrix_flagged = _matrix_wide_status_rows(matrix, "flagged")
    matrix_blocked = _matrix_wide_status_rows(matrix, "blocked")
    matrix_gated = _matrix_wide_status_rows(matrix, "gated-skipped")
    matrix_missing = _matrix_wide_status_rows(matrix, "missing")
    decisions = _claim_promotion_decisions(matrix, by_id)
    gate_checks = _publication_gate_checks(
        matrix,
        decisions,
        required_errors,
        matrix_flagged,
        matrix_blocked,
        matrix_gated,
        matrix_missing,
    )
    paper_ready_blockers = _paper_ready_blockers(
        matrix,
        decisions,
        required_errors,
        gate_checks,
        [*matrix_flagged, *matrix_blocked, *matrix_gated, *matrix_missing],
    )
    capstone_ready = not required_errors and matrix.get("matrix_v17_ready") is True
    paper_ready = capstone_ready and not paper_ready_blockers
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "n_tasks_evaluated": len(task_rows),
        "task_rows": task_rows,
        "task_classification_counts": task_counts,
        "clean_task_rows": _status_rows(task_rows, "clean"),
        "flagged_task_rows": _status_rows(task_rows, "flagged"),
        "blocked_task_rows": _status_rows(task_rows, "blocked"),
        "gated_skipped_task_rows": _status_rows(task_rows, "gated-skipped"),
        "missing_task_rows": _status_rows(task_rows, "missing"),
        "pilot_only_task_rows": _status_rows(task_rows, "pilot-only"),
        "projection_only_task_rows": _status_rows(task_rows, "projection-only"),
        "clean_rows": _matrix_wide_status_rows(matrix, "clean"),
        "flagged_rows": matrix_flagged,
        "blocked_rows": matrix_blocked,
        "gated_skipped_rows": matrix_gated,
        "missing_rows": matrix_missing,
        "pilot_only_rows": _matrix_wide_status_rows(matrix, "pilot-only"),
        "projection_only_rows": _matrix_wide_status_rows(matrix, "projection-only"),
        "repaired_rows": _repaired_rows(decisions),
        "claim_promotion_decisions": decisions,
        "repaired_282_blockers": _repaired_282_blockers(decisions),
        "unrepaired_282_blockers": _unrepaired_282_blockers(decisions),
        "publication_gate_checks": gate_checks,
        "paper_ready_blockers": paper_ready_blockers,
        "publication_action_allowed": False,
        "external_publication_triggered": False,
        "next_milestone_recommendation": NEXT_MILESTONE_RECOMMENDATION,
        "matrix_v17_ready": bool(matrix.get("matrix_v17_ready")),
        "matrix_v17_honest_verdict": str(matrix.get("honest_verdict") or ""),
        "matrix_row_count": len(rows),
        "matrix_status_counts": _counts(rows),
        "matrix_repaired_claims": _string_list(matrix.get("repaired_claims")),
        "matrix_still_blocked_claims": _string_list(matrix.get("still_blocked_claims")),
        "matrix_recommended_next_actions": _string_list(matrix.get("recommended_next_actions")),
        "claim_boundary_violations": _dict_list(matrix.get("claim_boundary_violations")),
        "missing_artifacts": _string_list(matrix.get("missing_artifacts")),
        "missing_documents": _string_list(matrix.get("missing_documents")),
        "cited_upstream_artifacts": _cited_upstream_artifacts(source_artifacts, payloads, matrix),
        "source_artifacts_read": source_artifacts,
        "source_checksums": _source_checksums(source_artifacts),
        "document_inputs_read": _document_inputs_read(root_path),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
        "duration_s": duration_s,
        "honest_verdict": "blocked_required_upstream_missing"
        if required_errors
        else _honest_verdict(
            paper_ready,
            _repaired_rows(decisions),
            matrix_flagged,
            matrix_blocked,
            matrix_gated,
            matrix_missing,
        ),
    }
    if required_errors:
        artifact["required_upstream_errors"] = required_errors
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3025 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def main(root: Path | str = REPO_ROOT) -> int:
    """Write the capstone artifact and return process-style success."""

    output = write_artifact(root)
    artifact = read_json_object(output)
    return 0 if artifact.get("capstone_ready") is True else 1


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


def _matrix_rows(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = matrix.get("rows")
    return [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _row_by_id(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["row_id"]): dict(row)
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("row_id"), str)
    }


def _task_row_from_matrix(
    exp_id: str,
    matrix: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row_id = TASK_ROW_IDS.get(exp_id)
    if row_id is None:
        return _synthetic_task_row(exp_id, f"{exp_id}_unknown", "missing", "")
    if exp_id == "exp3024":
        status = "clean" if matrix.get("matrix_v17_ready") is True else "missing"
        return _synthetic_task_row(
            exp_id,
            row_id,
            status,
            str(matrix.get("honest_verdict") or ""),
            {"matrix_v17_ready": bool(matrix.get("matrix_v17_ready"))},
        )
    source = by_id.get(row_id)
    if not source:
        return _synthetic_task_row(exp_id, row_id, "missing", "")
    return {
        "experiment_id": exp_id,
        "row_id": row_id,
        "status": _normalized_status(source.get("status")),
        "claim_class": str(source.get("claim_class") or ""),
        "evidence_type": str(source.get("evidence_type") or ""),
        "inference_substrate": str(source.get("inference_substrate") or ""),
        "source_honest_verdict": str(source.get("source_honest_verdict") or ""),
        "upstream_flags": _string_list(source.get("upstream_flags")),
        "summary": dict(source.get("summary")) if isinstance(source.get("summary"), Mapping) else {},
        "claim_boundary": str(source.get("claim_boundary") or ""),
    }


def _synthetic_task_row(
    exp_id: str,
    row_id: str,
    status: str,
    verdict: str,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "row_id": row_id,
        "status": _normalized_status(status),
        "claim_class": "matrix_synthesis" if exp_id == "exp3024" else "",
        "evidence_type": "aggregation_only_matrix" if exp_id == "exp3024" else "",
        "inference_substrate": INFERENCE_SUBSTRATE if exp_id == "exp3024" else "",
        "source_honest_verdict": verdict,
        "upstream_flags": [],
        "summary": dict(summary or {}),
        "claim_boundary": "matrix v17 authority row" if exp_id == "exp3024" else "",
    }


def _normalized_status(value: object) -> str:
    status = str(value or "missing")
    return status if status in STATUS_BUCKETS else "missing"


def _status_rows(rows: list[Mapping[str, Any]], status: str) -> list[str]:
    return [
        str(row["row_id"])
        for row in rows
        if isinstance(row.get("row_id"), str) and _normalized_status(row.get("status")) == status
    ]


def _matrix_wide_status_rows(matrix: Mapping[str, Any], status: str) -> list[str]:
    return _status_rows(_matrix_rows(matrix), status)


def _counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {status: len(_status_rows(rows, status)) for status in STATUS_BUCKETS}


def _claim_row(
    matrix: Mapping[str, Any],
    claim_key: str,
    row_id: str,
    by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    claim_rows = matrix.get("claim_rows")
    if isinstance(claim_rows, Mapping) and isinstance(claim_rows.get(claim_key), Mapping):
        return dict(claim_rows[claim_key])
    return dict((by_id or {}).get(row_id, {}))


def _claim_promotion_decisions(
    matrix: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    repair = _decision(
        "repair",
        _claim_row(matrix, "exp3016_repair", "exp3016_repair_acceptance_controller", by_id),
        "Exp 3016 repair is promotable only when matrix v17 is clean and adversarial/methodology flags are absent.",
    )
    fr11 = _decision(
        "fr11_self_learning",
        _claim_row(matrix, "exp3020_fr11_self_learning", "exp3020_fr11_verifier_feedback_controller", by_id),
        "FR-11 promotion is bounded to verifier-feedback controller utility over exact cached traces only.",
    )
    gatemate = _decision(
        "gatemate_io",
        _claim_row(matrix, "exp3022_gatemate_io", "exp3022_gatemate_transport_flash_smoke", by_id),
        "GateMate IO is promotable only with deterministic host-visible output and no sampler or speedup claim.",
    )
    ssqa = _decision(
        "ssqa",
        _claim_row(matrix, "exp3023_ssqa", "exp3023_ssqa_explicit_gate_artifact", by_id),
        "SSQA is promotable only after GateMate host-visible IO opens and bounded RTL/PnR/resource evidence exists.",
    )
    aggregation = {
        "row_id": "exp3024_cross_corpus_matrix_v17",
        "status": "clean" if matrix.get("matrix_v17_ready") is True else "missing",
        "promotable": _aggregation_metadata_clean(matrix),
        "repaired_282_blocker": _aggregation_metadata_clean(matrix),
        "claim_boundary": (
            "Aggregation metadata is clean only when top-level substrate is aggregation-only and "
            "model/hardware provenance remains nested under cited upstream artifacts."
        ),
        "source_honest_verdict": str(matrix.get("honest_verdict") or ""),
        "summary": {
            "matrix_v17_ready": bool(matrix.get("matrix_v17_ready")),
            "inference_substrate": str(matrix.get("inference_substrate") or ""),
        },
        "upstream_flags": [],
    }
    repair["repaired_282_blocker"] = repair["promotable"]
    fr11["promotable"] = fr11["promotable"] and _self_learning_boundary_clean(fr11)
    fr11["repaired_282_blocker"] = fr11["promotable"]
    gatemate["promotable"] = gatemate["promotable"] and _gatemate_host_visible(gatemate)
    gatemate["repaired_282_blocker"] = gatemate["promotable"]
    ssqa["promotable"] = ssqa["promotable"] and _ssqa_rtl_ready(ssqa)
    ssqa["repaired_282_blocker"] = _ssqa_artifact_written(ssqa)
    return {
        "repair": repair,
        "fr11_self_learning": fr11,
        "gatemate_io": gatemate,
        "ssqa": ssqa,
        "aggregation_metadata": aggregation,
    }


def _decision(key: str, row: Mapping[str, Any], boundary: str) -> dict[str, Any]:
    status = _normalized_status(row.get("status"))
    return {
        "row_id": str(row.get("row_id") or _default_decision_row_id(key)),
        "status": status,
        "promotable": status == "clean",
        "repaired_282_blocker": False,
        "claim_boundary": boundary,
        "source_honest_verdict": str(row.get("source_honest_verdict") or ""),
        "summary": dict(row.get("summary")) if isinstance(row.get("summary"), Mapping) else {},
        "upstream_flags": _string_list(row.get("upstream_flags")),
    }


def _default_decision_row_id(key: str) -> str:
    return {
        "repair": "exp3016_repair_acceptance_controller",
        "fr11_self_learning": "exp3020_fr11_verifier_feedback_controller",
        "gatemate_io": "exp3022_gatemate_transport_flash_smoke",
        "ssqa": "exp3023_ssqa_explicit_gate_artifact",
    }.get(key, f"{key}_missing")


def _self_learning_boundary_clean(decision: Mapping[str, Any]) -> bool:
    summary = _mapping(decision.get("summary"))
    return (
        summary.get("continuous_self_learning_task") is True
        and summary.get("independent_self_learning_boundary_preserved") is True
        and _float_or(summary.get("heldout_delta"), 0.0) > 0.0
        and _float_or(summary.get("negative_control_delta"), 1.0) <= 0.0
        and summary.get("forgetting_guard_passed") is True
        and summary.get("drift_guard_passed") is True
        and summary.get("tautology_risk_flag") is not True
    )


def _gatemate_host_visible(decision: Mapping[str, Any]) -> bool:
    summary = _mapping(decision.get("summary"))
    return (
        summary.get("host_visible_io_ready") is True
        and (summary.get("smoke_vector_passed") is True or summary.get("observed_output_hash_present") is True)
    )


def _ssqa_rtl_ready(decision: Mapping[str, Any]) -> bool:
    summary = _mapping(decision.get("summary"))
    return summary.get("ssqa_rtl_pnr_report_ready") is True and summary.get("upstream_host_visible_io_ready") is True


def _ssqa_artifact_written(decision: Mapping[str, Any]) -> bool:
    return _mapping(decision.get("summary")).get("ssqa_artifact_written") is True


def _aggregation_metadata_clean(matrix: Mapping[str, Any]) -> bool:
    forbidden_top_level = {
        "model_specs",
        "target_model",
        "cuda",
        "gpu",
        "gguf",
        "headline_models_used",
        "model_checksums",
    }
    return (
        matrix.get("matrix_v17_ready") is True
        and matrix.get("inference_substrate") == INFERENCE_SUBSTRATE
        and not forbidden_top_level.intersection(matrix.keys())
    )


def _publication_gate_checks(
    matrix: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
    required_errors: list[dict[str, Any]],
    flagged_rows: list[str],
    blocked_rows: list[str],
    gated_rows: list[str],
    missing_rows: list[str],
) -> dict[str, bool]:
    every_promotion_gate_clean = all(
        decisions[key].get("promotable") is True for key in ("repair", "fr11_self_learning", "gatemate_io", "ssqa")
    )
    boundary_clean = not _boundary_failed(matrix)
    no_false_sota = not _has_violation(matrix, "false_sota_headline_use")
    no_live_substrate_ambiguity = matrix.get("inference_substrate") == INFERENCE_SUBSTRATE
    no_aggregation_false_positive = decisions["aggregation_metadata"].get("promotable") is True
    durable = (
        not required_errors
        and every_promotion_gate_clean
        and not flagged_rows
        and not blocked_rows
        and not gated_rows
        and not missing_rows
    )
    return {
        "durable_verifier_evidence_for_every_claimed_result": durable,
        "no_false_sota_substitution": no_false_sota,
        "no_live_substrate_ambiguity": no_live_substrate_ambiguity,
        "no_aggregation_live_inference_false_positive": no_aggregation_false_positive,
        "no_hardware_claim_boundary_breach": boundary_clean,
        "every_promotion_gate_clean": every_promotion_gate_clean,
    }


def _has_violation(matrix: Mapping[str, Any], violation: str) -> bool:
    return any(item.get("violation") == violation for item in _dict_list(matrix.get("claim_boundary_violations")))


def _boundary_failed(matrix: Mapping[str, Any]) -> bool:
    if matrix.get("claim_boundary_violations"):
        return True
    paper = matrix.get("paper_v6_boundary_summary")
    hardware = matrix.get("hardware_boundary_summary")
    return not (
        isinstance(paper, Mapping)
        and paper.get("forbidden_claims_absent") is True
        and isinstance(hardware, Mapping)
        and hardware.get("forbidden_claims_absent") is True
    )


def _paper_ready_blockers(
    matrix: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
    required_errors: list[dict[str, Any]],
    gate_checks: Mapping[str, bool],
    non_promotable_rows: list[str],
) -> list[str]:
    blockers: list[str] = []
    if required_errors:
        blockers.append("required upstream matrix or prior capstone is missing or malformed")
    if matrix.get("matrix_v17_ready") is not True:
        blockers.append("matrix_v17_ready is not true")
    if _boundary_failed(matrix):
        blockers.append("matrix_v17 claim_boundary_violations is non-empty")
    labels = {
        "repair": ("repair row", "exp3016_repair_acceptance_controller"),
        "fr11_self_learning": ("FR-11 row", "exp3020_fr11_verifier_feedback_controller"),
        "gatemate_io": ("GateMate IO row", "exp3022_gatemate_transport_flash_smoke"),
        "ssqa": ("SSQA row", "exp3023_ssqa_explicit_gate_artifact"),
    }
    for key, (label, row_id) in labels.items():
        status = _normalized_status(decisions.get(key, {}).get("status"))
        if status != "clean":
            blockers.append(f"{label} {row_id} is {status}")
    gate_messages = {
        "durable_verifier_evidence_for_every_claimed_result": (
            "durable verifier evidence is not clean for every claimed result"
        ),
        "no_false_sota_substitution": "false SOTA substitution risk is not cleared",
        "no_live_substrate_ambiguity": "live/substrate ambiguity is not cleared",
        "no_aggregation_live_inference_false_positive": (
            "aggregation-live-inference false-positive risk is not cleared"
        ),
        "no_hardware_claim_boundary_breach": "hardware claim boundary breach risk is not cleared",
        "every_promotion_gate_clean": "not every promotion gate is clean",
    }
    for key, message in gate_messages.items():
        if gate_checks.get(key) is not True:
            blockers.append(message)
    if non_promotable_rows:
        counts = {
            "flagged": len(_matrix_wide_status_rows(matrix, "flagged")),
            "blocked": len(_matrix_wide_status_rows(matrix, "blocked")),
            "gated_skipped": len(_matrix_wide_status_rows(matrix, "gated-skipped")),
            "missing": len(_matrix_wide_status_rows(matrix, "missing")),
        }
        blockers.append(
            "matrix contains non-clean rows: "
            f"flagged={counts['flagged']}, blocked={counts['blocked']}, "
            f"gated_skipped={counts['gated_skipped']}, missing={counts['missing']}"
        )
    return _unique_strings(blockers)


def _repaired_rows(decisions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows: list[str] = []
    if decisions["repair"].get("promotable") is True:
        rows.append("exp3016_repair_acceptance_controller")
    if decisions["fr11_self_learning"].get("repaired_282_blocker") is True:
        rows.append("exp3020_fr11_verifier_feedback_controller")
    if decisions["gatemate_io"].get("promotable") is True:
        rows.append("exp3022_gatemate_transport_flash_smoke")
    if decisions["ssqa"].get("repaired_282_blocker") is True:
        rows.append("exp3023_ssqa_explicit_gate_artifact_presence_only")
    if decisions["aggregation_metadata"].get("repaired_282_blocker") is True:
        rows.append("exp3024_cross_corpus_matrix_v17")
    return rows


def _repaired_282_blockers(decisions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    repaired: list[str] = []
    if decisions["repair"].get("repaired_282_blocker") is True:
        repaired.append("exp3003_repair_methodology_repaired_by_exp3016")
    if decisions["fr11_self_learning"].get("repaired_282_blocker") is True:
        repaired.append("exp3007_fr11_stability_repaired_by_exp3020_bounded_controller")
    if decisions["gatemate_io"].get("repaired_282_blocker") is True:
        repaired.append("exp3008_hardware_io_repaired_by_exp3022")
    if decisions["ssqa"].get("repaired_282_blocker") is True:
        repaired.append("exp3009_ssqa_missing_artifact_repaired_by_exp3023_artifact_presence_only")
    if decisions["aggregation_metadata"].get("repaired_282_blocker") is True:
        repaired.append("exp3011_aggregation_false_positive_risk_repaired_by_exp3024_nested_provenance")
    return repaired


def _unrepaired_282_blockers(decisions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    unrepaired: list[str] = []
    if decisions["repair"].get("promotable") is not True:
        unrepaired.append(
            f"exp3003_repair_methodology_still_{_status_token(decisions['repair'].get('status'))}_by_exp3016"
        )
    if decisions["fr11_self_learning"].get("promotable") is not True:
        unrepaired.append(
            "exp3007_fr11_stability_still_"
            f"{_status_token(decisions['fr11_self_learning'].get('status'))}_by_exp3020"
        )
    if decisions["gatemate_io"].get("promotable") is not True:
        unrepaired.append(
            "exp3008_hardware_io_still_"
            f"{_status_token(decisions['gatemate_io'].get('status'))}_by_exp3022"
        )
    if decisions["ssqa"].get("promotable") is not True:
        unrepaired.append(f"exp3009_ssqa_promotion_still_{_status_token(decisions['ssqa'].get('status'))}_by_exp3023")
    if decisions["aggregation_metadata"].get("promotable") is not True:
        unrepaired.append(
            "exp3011_aggregation_false_positive_risk_still_"
            f"{_status_token(decisions['aggregation_metadata'].get('status'))}_by_exp3024"
        )
    return unrepaired


def _cited_upstream_artifacts(
    source_artifacts: list[dict[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matrix_source = next((row for row in source_artifacts if row["experiment_id"] == "exp3024"), None)
    citations: list[dict[str, Any]] = []
    if matrix_source:
        citations.append(
            {
                "experiment_id": "exp3024",
                "path": matrix_source["path"],
                "present": matrix_source["present"],
                "readable_json_object": matrix_source["readable_json_object"],
                "sha256": matrix_source["sha256"],
                "honest_verdict": str(matrix.get("honest_verdict", "")),
                "inference_substrate": str(matrix.get("inference_substrate", "")),
            }
        )
    matrix_citations = matrix.get("cited_upstream_artifacts")
    if isinstance(matrix_citations, list):
        citations.extend(dict(item) for item in matrix_citations if isinstance(item, Mapping))
    else:
        citations.extend(_fallback_citations(source_artifacts, payloads))
    return citations


def _fallback_citations(
    source_artifacts: list[dict[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for source in source_artifacts:
        if source["experiment_id"] == "exp3024":
            continue
        payload = payloads.get(str(source["experiment_id"]), {})
        citation = {
            "experiment_id": source["experiment_id"],
            "path": source["path"],
            "present": source["present"],
            "readable_json_object": source["readable_json_object"],
            "sha256": source["sha256"],
            "honest_verdict": str(payload.get("honest_verdict", "")),
            "inference_substrate": str(payload.get("inference_substrate", "")),
        }
        model = _source_model_provenance(payload)
        hardware = _source_hardware_provenance(payload)
        if model:
            citation["model_provenance"] = model
        if hardware:
            citation["hardware_provenance"] = hardware
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
        "transcript_paths": payload.get("transcript_paths"),
        "observed_output_hash": payload.get("observed_output_hash"),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _honest_verdict(
    paper_ready: bool,
    repaired_rows: list[str],
    flagged_rows: list[str],
    blocked_rows: list[str],
    gated_rows: list[str],
    missing_rows: list[str],
) -> str:
    return (
        f"complete: capstone_ready=true; paper_ready={str(paper_ready).lower()}; "
        f"repaired={len(repaired_rows)}; flagged={len(flagged_rows)}; "
        f"blocked={len(blocked_rows)}; gated_skipped={len(gated_rows)}; "
        f"missing={len(missing_rows)}; next=2026.05.284 repair-corrigendum-and-gatemate-output-contract"
    )


def _status_token(value: object) -> str:
    return _normalized_status(value).replace("-", "_")


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _dict_list(value: object) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


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


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
