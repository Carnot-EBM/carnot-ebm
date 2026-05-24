"""Build the Exp 2999 milestone .281 capstone artifact.

Spec refs: REQ-REPORT-2999, SCENARIO-REPORT-2999.

This module is a closeout synthesizer. It reads the local .281 matrix, the
prior .280 capstone, and the available .281 result artifacts, then writes a
single terminal judgment. It deliberately does not rerun models, verifiers,
solvers, synthesis, board flashing, smoke tests, the conductor, or publication
tooling because the capstone's job is to report what the milestone proved,
not to create new evidence.
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
SCHEMA = "carnot.milestone_capstone.v281_aggregation.v1"
ARTIFACT = "experiment_2999_capstone_v281"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2999_capstone_v281.json")

MATRIX_V15_REL_PATH = Path("results/experiment_2998_cross_corpus_matrix_v15.json")
CAPSTONE_V280_REL_PATH = Path("results/experiment_2987_capstone_v280.json")
EXP2988_REL_PATH = Path("results/experiment_2988_archive_v280_activate_v281.json")
EXP2989_REL_PATH = Path("results/experiment_2989_sota_gguf_cache_provenance_preflight_v1.json")
EXP2990_REL_PATH = Path("results/experiment_2990_verifier_backed_hard_code_stress_manifest_v1.json")
EXP2991_REL_PATH = Path("results/experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1.json")
EXP2992_REL_PATH = Path(
    "results/experiment_2992_sota_solver_formalization_provenance_reproduction_v1.json"
)
EXP2993_REL_PATH = Path("results/experiment_2993_aquaforte_beaver_substrate_corrigendum_v1.json")
EXP2994_REL_PATH = Path("results/experiment_2994_prompt_validator_dialogue_schema_v1.json")
EXP2995_REL_PATH = Path("results/experiment_2995_fr11_verifier_grounded_trace_memory_v2.json")
EXP2996_REL_PATH = Path("results/experiment_2996_gatemate_host_visible_readback_smoke_v1.json")
EXP2997_REL_PATH = Path("results/experiment_2997_ssqa_dual_bram_rtl_pnr_resource_report_v1.json")

STATUS_BUCKETS = (
    "clean",
    "flagged",
    "blocked",
    "missing",
    "gated-skipped",
    "pilot-only",
    "projection-only",
)
REQUIRED_CLAIM_KEYS = (
    "sota_cache",
    "hard_code_manifest",
    "repair",
    "solver_provenance",
    "aquaforte_beaver_substrate",
    "prompt_validator_protocol",
    "fr11_self_learning",
    "gatemate",
    "ssqa",
)
PROXIMITY_CLAIM_KEYS = (
    "sota_cache",
    "hard_code_manifest",
    "solver_provenance",
    "prompt_validator_protocol",
    "fr11_self_learning",
)


@dataclass(frozen=True)
class SourceSpec:
    """One local artifact the capstone should inspect and checksum."""

    experiment_id: str
    path: Path
    required: bool = False


SOURCE_SPECS = (
    SourceSpec("exp2998", MATRIX_V15_REL_PATH, required=True),
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
    """Read a JSON object, failing closed when the file is absent or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA256 digest for a present source artifact."""

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
    """REQ-REPORT-2999: synthesize the terminal .281 capstone from local evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    matrix = payloads["exp2998"]
    capstone_v280 = payloads["exp2987"]
    rows = _matrix_rows(matrix)
    source_artifacts = _source_artifacts_read(root_path)
    required_errors = _required_source_errors(payloads)
    buckets = {status: _row_ids_by_status(rows, status) for status in STATUS_BUCKETS}
    proof_summary = _milestone_proof_summary(matrix)
    blockers = _paper_ready_blockers(matrix, rows, required_errors)
    capstone_ready = not required_errors and matrix.get("matrix_v15_ready") is True
    paper_ready = capstone_ready and not blockers
    paper_closer = capstone_ready and any(
        proof_summary[key]["status"] == "clean" for key in PROXIMITY_CLAIM_KEYS
    )
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "paper_v6_closer_to_readiness": paper_closer,
        "clean_artifacts": buckets["clean"],
        "flagged_artifacts": buckets["flagged"],
        "blocked_artifacts": buckets["blocked"],
        "missing_artifacts": buckets["missing"],
        "gated_skipped_artifacts": buckets["gated-skipped"],
        "pilot_only_artifacts": buckets["pilot-only"],
        "projection_only_artifacts": buckets["projection-only"],
        "artifact_classification_counts": {status: len(buckets[status]) for status in STATUS_BUCKETS},
        "matrix_v15_ready": bool(matrix.get("matrix_v15_ready")),
        "matrix_v15_honest_verdict": str(matrix.get("honest_verdict") or ""),
        "capstone_v280_honest_verdict": str(capstone_v280.get("honest_verdict") or ""),
        "milestone_proof_summary": proof_summary,
        "paper_ready_blockers": blockers,
        "gaps_closed": _gaps_closed(proof_summary),
        "gaps_remaining": _gaps_remaining(proof_summary, rows, blockers),
        "next_milestone_recommendations": _next_milestone_recommendations(matrix, proof_summary),
        "external_publication_triggered": False,
        "honest_verdict": "blocked_required_upstream_missing"
        if required_errors
        else _honest_verdict(paper_ready, buckets),
        "required_upstream_errors": required_errors,
        "source_artifacts_read": source_artifacts,
        "source_checksums": _source_checksums(source_artifacts),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_external_publication_action": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration_s,
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2999 deliverable JSON."""

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


def _source_artifacts_read(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "present": path.is_file(),
                "required": spec.required,
                "readable_json_object": bool(read_json_object(path)),
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


def _matrix_rows(matrix: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_rows = matrix.get("rows")
    if not isinstance(raw_rows, list):
        return []
    return [row for row in raw_rows if isinstance(row, Mapping)]


def _row_ids_by_status(rows: list[Mapping[str, Any]], status: str) -> list[str]:
    return [
        str(row["row_id"])
        for row in rows
        if row.get("status") == status and isinstance(row.get("row_id"), str)
    ]


def _claim_statuses(matrix: Mapping[str, Any]) -> dict[str, str]:
    claim_rows = matrix.get("claim_rows")
    if not isinstance(claim_rows, Mapping):
        return {}
    return {
        str(key): str(value.get("status"))
        for key, value in claim_rows.items()
        if isinstance(value, Mapping) and value.get("status") is not None
    }


def _milestone_proof_summary(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    claim_rows = matrix.get("claim_rows") if isinstance(matrix.get("claim_rows"), Mapping) else {}
    summary: dict[str, dict[str, Any]] = {}
    for key in REQUIRED_CLAIM_KEYS:
        row = claim_rows.get(key) if isinstance(claim_rows, Mapping) else None
        if isinstance(row, Mapping):
            summary[key] = {
                "row_id": str(row.get("row_id") or ""),
                "source_experiment_id": str(row.get("source_experiment_id") or ""),
                "status": str(row.get("status") or "missing"),
                "source_honest_verdict": str(row.get("source_honest_verdict") or ""),
                "headline_eligible": bool(row.get("headline_eligible")),
                "paper_claim_eligible": bool(row.get("paper_claim_eligible")),
                "summary": row.get("summary") if isinstance(row.get("summary"), Mapping) else {},
            }
        else:
            summary[key] = {
                "row_id": "",
                "source_experiment_id": "",
                "status": "missing",
                "source_honest_verdict": "",
                "headline_eligible": False,
                "paper_claim_eligible": False,
                "summary": {},
            }
    return summary


def _paper_ready_blockers(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    required_errors: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if required_errors:
        blockers.append("required upstream matrix or prior capstone is missing or malformed")
    if matrix.get("matrix_v15_ready") is not True:
        blockers.append("matrix_v15_ready is not true")
    for key in REQUIRED_CLAIM_KEYS:
        status = _claim_statuses(matrix).get(key, "missing")
        if status != "clean":
            blockers.append(f"{key} claim row status is {status}")
    for status in ("flagged", "blocked", "missing", "gated-skipped"):
        row_ids = _row_ids_by_status(rows, status)
        if row_ids:
            blockers.append(f"{status} rows remain: {', '.join(row_ids)}")
    if matrix.get("unresolved_blockers"):
        blockers.append("matrix_v15 unresolved_blockers is non-empty")
    if matrix.get("claim_boundary_violations"):
        blockers.append("matrix_v15 claim_boundary_violations is non-empty")
    paper_boundary = matrix.get("paper_v6_claim_boundary")
    if isinstance(paper_boundary, Mapping) and paper_boundary.get("forbidden_claims_absent") is not True:
        blockers.append("paper-v6 forbidden-claim boundary failed")
    hardware_boundary = matrix.get("hardware_claim_boundary")
    if isinstance(hardware_boundary, Mapping) and hardware_boundary.get("forbidden_claims_absent") is not True:
        blockers.append("hardware forbidden-claim boundary failed")
    return blockers


def _gaps_closed(proof_summary: Mapping[str, Mapping[str, Any]]) -> list[str]:
    closed: list[str] = []
    if proof_summary["sota_cache"]["status"] == "clean":
        closed.append("SOTA cache gap closed narrowly: at least one mandated GGUF transcript is local.")
    if proof_summary["hard_code_manifest"]["status"] == "clean":
        closed.append("Hard repair precondition closed: verifier-backed hard-code manifest is ready.")
    if proof_summary["solver_provenance"]["status"] == "clean":
        closed.append("Solver provenance gap closed: Z3-backed formalization reproduction is clean.")
    if proof_summary["prompt_validator_protocol"]["status"] == "clean":
        closed.append("Prompt-validator schema gap closed: deterministic exact-check protocol is ready.")
    if proof_summary["fr11_self_learning"]["status"] == "clean":
        closed.append("FR-11 carry-forward gap closed: verifier-grounded trace memory passed its boundary.")
    return closed


def _gaps_remaining(
    proof_summary: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
    blockers: list[str],
) -> list[str]:
    remaining: list[str] = []
    if proof_summary["repair"]["status"] != "clean":
        remaining.append("Repair remains non-promotable until Exp 2991 artifact flags and gates are clean.")
    if proof_summary["aquaforte_beaver_substrate"]["status"] != "clean":
        remaining.append("AquaForte/BEAVER remains flagged until live retry methodology is clean.")
    if proof_summary["gatemate"]["status"] != "clean":
        remaining.append("GateMate remains blocked until host-visible readback or smoke output exists.")
    if proof_summary["ssqa"]["status"] != "clean":
        remaining.append("SSQA remains missing until RTL/PnR/resource evidence is written.")
    carry_forward_open = [
        str(row.get("row_id"))
        for row in rows
        if str(row.get("row_id", "")).startswith("carry_forward_v14:")
        and row.get("status") in {"flagged", "blocked", "missing", "gated-skipped"}
    ]
    if carry_forward_open:
        remaining.append(f"Prior matrix carry-forward blockers remain: {len(carry_forward_open)} rows.")
    if blockers:
        remaining.append("Paper-v6 remains blocked by local evidence gates; no publication action is allowed.")
    return remaining


def _next_milestone_recommendations(
    matrix: Mapping[str, Any],
    proof_summary: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    recommendations = [
        "Milestone .282 should be a claim-repair milestone, not an external-publication milestone.",
        "Repair: rerun or correct Exp 2991 with non-tautological pass@1/pass@k evidence, random seed, methodology, and false-accept provenance.",
        "AquaForte/BEAVER: rerun live retry with model specs, checksums, and durable duration provenance while keeping enumerator fallback separate.",
        "GateMate: implement host-visible output transport before another smoke/readback promotion attempt.",
        "SSQA: produce the missing dual-BRAM RTL/PnR/resource artifact or explicitly gate-skip it.",
        "Solver and FR-11: carry forward Exp 2992 and Exp 2995 only as narrow Z3-provenance and verifier-grounded trace-memory claims.",
    ]
    if proof_summary["sota_cache"]["status"] == "clean":
        recommendations.append("SOTA cache: reuse the working mandated GGUF path but do not broaden beyond recorded local models.")
    for item in matrix.get("next_milestone_recommendations", []):
        if isinstance(item, str) and item not in recommendations:
            recommendations.append(item)
    return recommendations


def _honest_verdict(paper_ready: bool, buckets: Mapping[str, list[str]]) -> str:
    return (
        f"complete: capstone_ready=true; paper_ready={str(paper_ready).lower()}; "
        f"clean={len(buckets['clean'])}; flagged={len(buckets['flagged'])}; "
        f"blocked={len(buckets['blocked'])}; missing={len(buckets['missing'])}; "
        f"gated_skipped={len(buckets['gated-skipped'])}"
    )
