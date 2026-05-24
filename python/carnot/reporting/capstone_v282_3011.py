"""Build the Exp 3011 milestone .282 terminal capstone artifact.

Spec refs: REQ-REPORT-3011, SCENARIO-REPORT-3011.

This module is a closeout ledger, not an experiment runner. It reads matrix
v16 and the local .282 artifacts to make a go/no-go decision about claim
repair while preserving the paper-v6 and hardware boundaries already recorded
in the upstream evidence.
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
SCHEMA = "carnot.milestone_capstone.v282_claim_repair_go_no_go.v1"
ARTIFACT = "experiment_3011_capstone_v282"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3011_capstone_v282.json")

MATRIX_V16_REL_PATH = Path("results/experiment_3010_cross_corpus_matrix_v16.json")
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

NEXT_MILESTONE_RECOMMENDATION = (
    "2026.05.283: claim-repair-v2 - fix Exp3003 repair methodology, clear Exp3007 "
    "FR-11 stability flags, implement GateMate host-visible IO before SSQA, and keep "
    "paper_ready=false until every promotion gate is clean."
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
    "exp3000",
    "exp3001",
    "exp3002",
    "exp3003",
    "exp3004",
    "exp3005",
    "exp3006",
    "exp3007",
    "exp3008",
    "exp3009",
    "exp3010",
)

TASK_ROW_IDS = {
    "exp3000": "exp3000_archive_activation",
    "exp3001": "exp3001_sota_cache",
    "exp3002": "exp3002_metamorphic_oracle",
    "exp3003": "exp3003_metamorphic_repair",
    "exp3004": "exp3004_aquaforte_beaver_provenance",
    "exp3005": "exp3005_validator_tree_expansion",
    "exp3006": "exp3006_fixed_point_diagnostic",
    "exp3007": "exp3007_fr11_trace_memory_stability",
    "exp3008": "exp3008_gatemate_host_visible_io",
    "exp3009": "exp3009_ssqa_dual_bram_report",
    "exp3010": "exp3010_cross_corpus_matrix_v16",
}

EXP_SOURCE_PATHS = {
    "exp3000": EXP3000_REL_PATH,
    "exp3001": EXP3001_REL_PATH,
    "exp3002": EXP3002_REL_PATH,
    "exp3003": EXP3003_REL_PATH,
    "exp3004": EXP3004_REL_PATH,
    "exp3005": EXP3005_REL_PATH,
    "exp3006": EXP3006_REL_PATH,
    "exp3007": EXP3007_REL_PATH,
    "exp3008": EXP3008_REL_PATH,
    "exp3009": EXP3009_REL_PATH,
    "exp3010": MATRIX_V16_REL_PATH,
}

REPAIR_LABEL_TO_ROW_ID = {
    "exp3001_sota_headline_cache_ready": "exp3001_sota_cache",
    "exp3003_metamorphic_repair": "exp3003_metamorphic_repair",
    "exp3004_aquaforte_beaver_substrate_provenance": "exp3004_aquaforte_beaver_provenance",
    "exp3005_validator_tree_expansion": "exp3005_validator_tree_expansion",
    "exp3006_fixed_point_energy_diagnostic": "exp3006_fixed_point_diagnostic",
    "exp3007_fr11_trace_memory_stability": "exp3007_fr11_trace_memory_stability",
    "exp3008_gatemate_host_visible_io": "exp3008_gatemate_host_visible_io",
    "exp3009_ssqa_dual_bram_report": "exp3009_ssqa_dual_bram_report",
}


@dataclass(frozen=True)
class SourceSpec:
    """A local source artifact the capstone reads and checksums."""

    experiment_id: str
    path: Path
    required: bool = False


SOURCE_SPECS = (
    SourceSpec("exp3010", MATRIX_V16_REL_PATH, required=True),
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


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object, returning an empty mapping when evidence is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return a source-file checksum, or ``None`` when the file is absent."""

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
    """REQ-REPORT-3011: synthesize the .282 go/no-go from matrix v16."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    matrix = payloads["exp3010"]
    source_artifacts = _source_artifacts_read(root_path, payloads)
    required_errors = _required_source_errors(payloads)
    matrix_rows = _matrix_rows(matrix)
    by_id = _row_by_id(matrix_rows)
    task_rows = [_task_row_from_matrix(exp_id, matrix, by_id) for exp_id in TASK_ORDER]
    task_counts = _counts(task_rows)
    matrix_flagged = _matrix_wide_status_rows(matrix, "flagged")
    matrix_blocked = _matrix_wide_status_rows(matrix, "blocked")
    matrix_gated = _matrix_wide_status_rows(matrix, "gated-skipped")
    matrix_missing = _matrix_wide_status_rows(matrix, "missing")
    repaired_rows = _repaired_rows(matrix, by_id)
    decisions = _claim_repair_decisions(by_id)
    capstone_ready = not required_errors and matrix.get("matrix_v16_ready") is True
    paper_ready_blockers = _paper_ready_blockers(
        matrix,
        decisions,
        required_errors,
        [*matrix_flagged, *matrix_blocked, *matrix_gated, *matrix_missing],
        _string_list(matrix.get("missing_artifacts")),
    )
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
        "repaired_rows": repaired_rows,
        "flagged_rows": matrix_flagged,
        "blocked_rows": matrix_blocked,
        "gated_skipped_rows": matrix_gated,
        "missing_rows": matrix_missing,
        "missing_artifacts": _string_list(matrix.get("missing_artifacts")),
        "claim_repair_decisions": decisions,
        "repaired_281_blockers": _repaired_281_blockers(decisions),
        "unrepaired_281_blockers": _unrepaired_281_blockers(decisions),
        "paper_ready_blockers": paper_ready_blockers,
        "publication_action_allowed": False,
        "external_publication_triggered": False,
        "next_milestone_recommendation": NEXT_MILESTONE_RECOMMENDATION,
        "matrix_v16_ready": bool(matrix.get("matrix_v16_ready")),
        "matrix_v16_honest_verdict": str(matrix.get("honest_verdict") or ""),
        "matrix_row_count": len(matrix_rows),
        "matrix_status_counts": _counts(matrix_rows),
        "matrix_repaired_claims": _string_list(matrix.get("repaired_claims")),
        "matrix_still_blocked_claims": _string_list(matrix.get("still_blocked_claims")),
        "matrix_recommended_next_actions": _string_list(matrix.get("recommended_next_actions")),
        "source_artifacts_read": source_artifacts,
        "source_checksums": _source_checksums(source_artifacts),
        "required_upstream_errors": required_errors,
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
        else _honest_verdict(paper_ready, repaired_rows, matrix_flagged, matrix_blocked, matrix_gated, matrix_missing),
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3011 deliverable JSON."""

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
    if exp_id == "exp3010":
        status = "clean" if matrix.get("matrix_v16_ready") is True else "missing"
        return _synthetic_task_row(
            exp_id,
            row_id,
            status,
            str(matrix.get("honest_verdict") or ""),
            {"matrix_v16_ready": bool(matrix.get("matrix_v16_ready"))},
        )
    source = by_id.get(row_id)
    if not source:
        return _synthetic_task_row(exp_id, row_id, "missing", "")
    return {
        "experiment_id": exp_id,
        "row_id": row_id,
        "status": _normalized_status(source.get("status")),
        "claim_class": str(source.get("claim_class") or ""),
        "source_honest_verdict": str(source.get("source_honest_verdict") or ""),
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
        "claim_class": "matrix_synthesis" if exp_id == "exp3010" else "",
        "source_honest_verdict": verdict,
        "summary": dict(summary or {}),
        "claim_boundary": "matrix v16 authority row" if exp_id == "exp3010" else "",
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


def _repaired_rows(matrix: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    repaired: list[str] = []
    for label in _string_list(matrix.get("repaired_claims")):
        row_id = REPAIR_LABEL_TO_ROW_ID.get(label)
        if row_id and by_id.get(row_id, {}).get("status") == "clean" and row_id not in repaired:
            repaired.append(row_id)
    return repaired


def _claim_repair_decisions(by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    repair = _decision(
        "exp3003_metamorphic_repair",
        by_id,
        "Exp 2991 repair-methodology blocker is repaired only by a clean metamorphic repair row.",
    )
    substrate = _decision(
        "exp3004_aquaforte_beaver_provenance",
        by_id,
        "Substrate provenance is promotable only as live retry provenance; not a BEAVER-task solution.",
    )
    fr11 = _decision(
        "exp3007_fr11_trace_memory_stability",
        by_id,
        "FR-11 is bounded to verifier-grounded trace-memory stability, not broad self-improvement.",
    )
    gatemate = _decision(
        "exp3008_gatemate_host_visible_io",
        by_id,
        "GateMate is bounded to host-visible IO evidence with no sampler or speedup claim.",
    )
    ssqa = _decision(
        "exp3009_ssqa_dual_bram_report",
        by_id,
        "SSQA is bounded to RTL/PnR/resource evidence and remains gated on GateMate IO.",
    )
    return {
        "repair": repair,
        "substrate_provenance": {**substrate, "repaired_281_blocker": substrate["promotable"]},
        "fr11_stability": fr11,
        "gatemate_io": gatemate,
        "ssqa": ssqa,
    }


def _decision(
    row_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
    boundary: str,
) -> dict[str, Any]:
    row = by_id.get(row_id, {})
    status = _normalized_status(row.get("status"))
    return {
        "row_id": row_id,
        "status": status,
        "promotable": status == "clean",
        "repaired_281_blocker": False,
        "claim_boundary": boundary,
        "source_honest_verdict": str(row.get("source_honest_verdict") or ""),
        "summary": dict(row.get("summary")) if isinstance(row.get("summary"), Mapping) else {},
    }


def _repaired_281_blockers(decisions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    repaired: list[str] = []
    if decisions["repair"].get("promotable") is True:
        repaired.append("exp2991_methodology_repaired_by_exp3003")
    if decisions["substrate_provenance"].get("promotable") is True:
        repaired.append("exp2993_provenance_repaired_by_exp3004_substrate_provenance")
    if decisions["fr11_stability"].get("promotable") is True:
        repaired.append("fr11_stability_carry_forward_repaired_by_exp3007")
    if decisions["gatemate_io"].get("promotable") is True:
        repaired.append("exp2996_hardware_repaired_by_exp3008")
    if decisions["ssqa"].get("promotable") is True:
        repaired.append("exp2997_ssqa_repaired_by_exp3009")
    return repaired


def _unrepaired_281_blockers(decisions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    unrepaired: list[str] = []
    if decisions["repair"].get("promotable") is not True:
        unrepaired.append(
            f"exp2991_methodology_still_{_status_token(decisions['repair'].get('status'))}_by_exp3003"
        )
    if decisions["fr11_stability"].get("promotable") is not True:
        unrepaired.append(
            "fr11_stability_carry_forward_still_"
            f"{_status_token(decisions['fr11_stability'].get('status'))}_by_exp3007"
        )
    if decisions["gatemate_io"].get("promotable") is not True:
        unrepaired.append(
            f"exp2996_hardware_still_{_status_token(decisions['gatemate_io'].get('status'))}_by_exp3008"
        )
    if decisions["ssqa"].get("promotable") is not True:
        unrepaired.append(f"exp2997_ssqa_still_{_status_token(decisions['ssqa'].get('status'))}_by_exp3009")
    if decisions["substrate_provenance"].get("promotable") is not True:
        unrepaired.append(
            "exp2993_provenance_still_"
            f"{_status_token(decisions['substrate_provenance'].get('status'))}_by_exp3004"
        )
    return unrepaired


def _status_token(value: object) -> str:
    return _normalized_status(value).replace("-", "_")


def _paper_ready_blockers(
    matrix: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
    required_errors: list[dict[str, Any]],
    non_promotable_rows: list[str],
    missing_artifacts: list[str],
) -> list[str]:
    blockers: list[str] = []
    if required_errors:
        blockers.append("required upstream matrix or prior capstone is missing or malformed")
    if matrix.get("matrix_v16_ready") is not True:
        blockers.append("matrix_v16_ready is not true")
    if _boundary_failed(matrix):
        blockers.append("matrix_v16 claim_boundary_violations is non-empty")
    labels = {
        "repair": "repair/metamorphic row",
        "substrate_provenance": "substrate-provenance row",
        "fr11_stability": "FR-11 stability row",
        "gatemate_io": "GateMate IO row",
        "ssqa": "SSQA row",
    }
    for key, label in labels.items():
        status = _normalized_status(decisions.get(key, {}).get("status"))
        if status != "clean":
            blockers.append(f"{label} is {status}")
    if non_promotable_rows:
        blockers.append("non-clean matrix rows remain: " + ", ".join(non_promotable_rows))
    if missing_artifacts:
        blockers.append("missing source artifacts remain: " + ", ".join(missing_artifacts))
    return blockers


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
        f"missing={len(missing_rows)}; next=2026.05.283 claim-repair-v2"
    )


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
