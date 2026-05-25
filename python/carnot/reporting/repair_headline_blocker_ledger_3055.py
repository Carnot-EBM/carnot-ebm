"""Build the Exp 3055 repair headline retirement and blocker ledger.

Spec refs: REQ-REPORT-3055, SCENARIO-REPORT-3055.

This module is a methodology ledger, not a repair rerun. It reads the .285
matrix/capstone chain, records which repair headline wording is retired, keeps
bounded repair evidence out of headline phrasing, and names the evidence gates
that must exist before any future repair rerun can be considered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.285"
SCHEMA = "carnot.repair_headline_retirement.blocker_ledger.v1"
ARTIFACT = "experiment_3055_repair_headline_retirement_and_blocker_ledger_v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3055_repair_headline_retirement_and_blocker_ledger_v1.json"
)
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3055_repair_headline_blocker_ledger.py"

EXP3041_REL_PATH = Path("results/experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1.json")
EXP3042_REL_PATH = Path("results/experiment_3042_repair_promotion_reconciliation_v3.json")
MATRIX_V19_REL_PATH = Path("results/experiment_3052_cross_corpus_matrix_v19.json")
CAPSTONE_V285_REL_PATH = Path("results/experiment_3053_capstone_v285.json")
MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
MANIFEST_ENTRY_ID = "repair_headline_wording_retired_v285"

SOURCE_SPECS = (
    ("exp3041", EXP3041_REL_PATH, "flag_hygiene_authority", True, "json"),
    ("exp3042", EXP3042_REL_PATH, "repair_reconciliation_authority", True, "json"),
    ("exp3052", MATRIX_V19_REL_PATH, "matrix_v19_authority", True, "json"),
    ("exp3053", CAPSTONE_V285_REL_PATH, "capstone_v285_authority", True, "json"),
    ("manifest", MANIFEST_REL_PATH, "exclusion_manifest", True, "text"),
)
BLOCKER_LIST_KEYS = (
    "true_blocker_rows",
    "missing_metadata_rows",
    "unresolved_bound_rows",
    "remaining_blockers",
)
REPAIR_TOKENS = ("repair", "exp3016", "exp3028", "exp3029", "sota")
INFERENCE_SUBSTRATE = {
    "kind": "aggregation_from_upstream_artifacts",
    "source": "checked_in_artifacts",
    "executes_models": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "no_live_llm_inference": True,
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while treating missing or malformed files as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 digest for an existing source file."""

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
    """REQ-REPORT-3055: build the repair headline retirement ledger."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_payloads(root_path)
    source_artifacts = _source_artifacts(root_path)
    source_errors = _source_errors(source_artifacts)
    matrix = payloads["exp3052"]
    capstone = payloads["exp3053"]
    retired_claims = _retired_repair_claims(payloads)
    bounded_claims = _bounded_repair_claims(payloads)
    blockers = _repair_blockers(payloads)
    manifest_updates = _manifest_updates(root_path, retired_claims)
    consumability_errors = _matrix_v20_consumability_errors(
        matrix,
        capstone,
        retired_claims,
        bounded_claims,
        blockers,
    )
    blocked_reasons = _blocked_reasons(
        source_errors=source_errors,
        manifest_updates=manifest_updates,
        retired_claims=retired_claims,
        bounded_claims=bounded_claims,
        consumability_errors=consumability_errors,
        matrix=matrix,
        capstone=capstone,
    )
    ready = not blocked_reasons

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "repair_headline_retirement_ready": ready,
        "repair_claim_status": str(
            matrix.get("repair_claim_status") or capstone.get("repair_claim_status") or ""
        ),
        "retired_repair_claims": retired_claims,
        "still_bounded_repair_claims": bounded_claims,
        "extracted_repair_blockers": blockers,
        "rerun_prerequisites": _rerun_prerequisites(),
        "manifest_updates": manifest_updates,
        "matrix_v20_consumability_errors": consumability_errors,
        "source_artifacts": source_artifacts,
        "source_checksums": {str(row["path"]): row.get("sha256") for row in source_artifacts},
        "missing_source_artifacts": [
            str(row["path"]) for row in source_artifacts if row.get("present") is not True
        ],
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_historical_artifact_rewrite": True,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "blocked_reasons": blocked_reasons,
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
    """Build and persist the Exp 3055 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_payloads(root: Path) -> dict[str, JsonDict]:
    return {
        "exp3041": read_json_object(root / EXP3041_REL_PATH),
        "exp3042": read_json_object(root / EXP3042_REL_PATH),
        "exp3052": read_json_object(root / MATRIX_V19_REL_PATH),
        "exp3053": read_json_object(root / CAPSTONE_V285_REL_PATH),
    }


def _source_artifacts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for experiment_id, rel_path, role, required, source_type in SOURCE_SPECS:
        path = root / rel_path
        payload = read_json_object(path) if source_type == "json" else {}
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": rel_path.as_posix(),
                "role": role,
                "required": required,
                "present": path.is_file(),
                "readable": path.is_file(),
                "readable_json_object": bool(payload) if source_type == "json" else None,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _source_errors(source_artifacts: Iterable[Mapping[str, Any]]) -> list[JsonDict]:
    errors: list[JsonDict] = []
    for row in source_artifacts:
        if row.get("required") is not True:
            continue
        if row.get("present") is not True:
            errors.append({"path": str(row.get("path")), "reason": "missing_required_source"})
        elif row.get("readable_json_object") is False:
            errors.append({"path": str(row.get("path")), "reason": "malformed_required_json"})
    return errors


def _retired_repair_claims(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    candidates: list[Mapping[str, Any]] = []
    candidates.extend(_rows_with_status(payloads["exp3052"], "rows", {"retired"}))
    candidates.extend(_rows_with_status(payloads["exp3053"], "retired_claims", {"retired"}))
    candidates.extend(_retired_from_blockers(payloads["exp3041"]))
    candidates.extend(_retired_from_blockers(payloads["exp3042"]))

    rows: dict[str, JsonDict] = {}
    for candidate in candidates:
        if not _is_repair_related(candidate):
            continue
        row = _normalize_retired_claim(candidate)
        rows.setdefault(str(row["claim_id"]), row)
    return list(rows.values())


def _bounded_repair_claims(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    candidates: list[Mapping[str, Any]] = []
    candidates.extend(_rows_with_status(payloads["exp3052"], "rows", {"bounded"}))
    candidates.extend(_rows_with_status(payloads["exp3053"], "bounded_claims", {"bounded"}))
    candidates.extend(_rows_with_status(payloads["exp3042"], "bounded_claims", {"bounded"}))
    candidates.extend(_rows_with_classification(payloads["exp3041"], "unresolved_bound_rows"))
    candidates.extend(
        _rows_with_classification(payloads["exp3042"], "remaining_blockers", {"unresolved_bound"})
    )

    rows: dict[str, JsonDict] = {}
    for candidate in candidates:
        if not _is_repair_related(candidate):
            continue
        row = _normalize_bounded_claim(candidate)
        rows.setdefault(str(row["row_id"]), row)
    return list(rows.values())


def _repair_blockers(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    candidates: list[Mapping[str, Any]] = []
    for key in BLOCKER_LIST_KEYS:
        candidates.extend(_rows_with_classification(payloads["exp3041"], key))
        candidates.extend(_rows_with_classification(payloads["exp3042"], key))
    candidates.extend(_rows_with_status(payloads["exp3052"], "rows", {"flagged", "blocked"}))
    candidates.extend(
        _rows_with_status(payloads["exp3053"], "blocked_claims", {"flagged", "blocked", "missing"})
    )

    rows: dict[str, JsonDict] = {}
    for candidate in candidates:
        if not _is_repair_related(candidate):
            continue
        row_id = _row_id(candidate)
        rows.setdefault(
            row_id,
            {
                "row_id": row_id,
                "classification": str(
                    candidate.get("classification") or candidate.get("status") or "repair_blocker"
                ),
                "status": str(
                    candidate.get("status") or candidate.get("classification") or "blocking"
                ),
                "blocking": candidate.get("blocking") is not False,
                "source_artifact": _source_artifact_from_row(candidate),
                "source_field": str(candidate.get("source_field") or ""),
                "rationale": str(candidate.get("rationale") or ""),
                "evidence": _evidence(candidate),
                "matrix_v20_consumable": _has_consumable_fields(candidate),
            },
        )
    return list(rows.values())


def _rows_with_status(
    payload: Mapping[str, Any],
    key: str,
    statuses: set[str],
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for row in _as_list(payload.get(key)):
        mapping = _as_mapping(row)
        if str(mapping.get("status") or "").lower() in statuses:
            rows.append(mapping)
    return rows


def _rows_with_classification(
    payload: Mapping[str, Any],
    key: str,
    classifications: set[str] | None = None,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for row in _as_list(payload.get(key)):
        mapping = _as_mapping(row)
        classification = str(mapping.get("classification") or "")
        if classifications is None or classification in classifications:
            rows.append(mapping)
    return rows


def _retired_from_blockers(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in ("true_blocker_rows", "remaining_blockers"):
        for row in _as_list(payload.get(key)):
            mapping = _as_mapping(row)
            evidence = _as_mapping(mapping.get("evidence"))
            source_field = str(mapping.get("source_field") or "")
            if (
                evidence.get("classification") == "retired"
                or "retired_or_blocked_claims" in source_field
            ):
                rows.append(mapping)
    return rows


def _normalize_retired_claim(row: Mapping[str, Any]) -> JsonDict:
    evidence = _evidence_mapping(row)
    claim_id = str(evidence.get("claim_id") or _claim_id_from_row_id(_row_id(row)))
    source_artifact = _source_artifact_from_row(row)
    source_field = str(row.get("source_field") or "")
    return {
        "claim_id": claim_id,
        "row_id": _row_id(row),
        "status": "retired_for_headline_use",
        "unsupported_headline_wording": str(
            evidence.get("proposed_repair_claim") or evidence.get("claim_id") or claim_id
        ),
        "allowed_wording": str(evidence.get("allowed_wording") or "Use bounded wording only."),
        "blockers": _as_list(evidence.get("blockers")),
        "required_support_fields": _as_list(evidence.get("required_support_fields")),
        "observed_support_fields": _as_mapping(evidence.get("observed_support_fields")),
        "source_artifact": source_artifact,
        "source_field": source_field,
        "matrix_v20_consumable": bool(
            claim_id and _row_id(row) and source_artifact and source_field
        ),
    }


def _normalize_bounded_claim(row: Mapping[str, Any]) -> JsonDict:
    evidence = _evidence_mapping(row)
    summary = _as_mapping(row.get("summary"))
    source_artifact = _source_artifact_from_row(row)
    source_field = str(row.get("source_field") or "")
    row_id = _row_id(row)
    return {
        "claim_id": str(evidence.get("claim_id") or _claim_id_from_row_id(row_id)),
        "row_id": row_id,
        "status": "bounded_evidence_not_headline",
        "repair_claim_status": str(
            summary.get("repair_claim_status")
            or evidence.get("classification")
            or row.get("classification")
            or "bounded"
        ),
        "repair_promotion_candidate": bool(summary.get("repair_promotion_candidate") is True),
        "allowed_wording": str(
            evidence.get("allowed_wording")
            or "Bounded repair evidence only; do not promote to headline wording."
        ),
        "blockers": _as_list(evidence.get("blockers")),
        "source_artifact": source_artifact,
        "source_field": source_field,
        "matrix_v20_consumable": bool(row_id and source_artifact and source_field),
    }


def _evidence_mapping(row: Mapping[str, Any]) -> JsonDict:
    evidence = _as_mapping(row.get("evidence"))
    if evidence:
        return evidence
    summary = _as_mapping(row.get("summary"))
    nested = _as_mapping(summary.get("evidence"))
    return nested if nested else summary


def _evidence(row: Mapping[str, Any]) -> Any:
    if "evidence" in row:
        return row.get("evidence")
    if "summary" in row:
        return row.get("summary")
    return {}


def _source_artifact_from_row(row: Mapping[str, Any]) -> str:
    if row.get("source_artifact"):
        return str(row["source_artifact"])
    nested = _as_mapping(row.get("evidence"))
    if nested.get("source_artifact_path"):
        return str(nested["source_artifact_path"])
    return ""


def _row_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("row_id") or _as_mapping(row.get("evidence")).get("row_id") or "repair:unknown"
    )


def _claim_id_from_row_id(row_id: str) -> str:
    return row_id.split(":", 1)[1] if ":" in row_id else row_id


def _is_repair_related(row: Mapping[str, Any]) -> bool:
    haystack = json.dumps(row, sort_keys=True, default=str).lower()
    return any(token in haystack for token in REPAIR_TOKENS)


def _has_consumable_fields(row: Mapping[str, Any]) -> bool:
    return bool(
        _row_id(row) and _source_artifact_from_row(row) and str(row.get("source_field") or "")
    )


def _manifest_updates(root: Path, retired_claims: list[Mapping[str, Any]]) -> list[JsonDict]:
    if not retired_claims:
        return []
    manifest_text = ""
    try:
        manifest_text = (root / MANIFEST_REL_PATH).read_text(encoding="utf-8")
    except OSError:
        manifest_text = ""
    return [
        {
            "id": MANIFEST_ENTRY_ID,
            "path": MANIFEST_REL_PATH.as_posix(),
            "applied": MANIFEST_ENTRY_ID in manifest_text,
            "reason": (
                "CLAUDE.md failed-rerun and exclusion-manifest discipline requires "
                "retired headline scope to be traceable."
            ),
            "retired_by_artifact": OUTPUT_REL_PATH.as_posix(),
        }
    ]


def _matrix_v20_consumability_errors(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    retired_claims: list[Mapping[str, Any]],
    bounded_claims: list[Mapping[str, Any]],
    blockers: list[Mapping[str, Any]],
) -> list[JsonDict]:
    errors: list[JsonDict] = []
    for key, rows in (
        ("matrix.rows", _as_list(matrix.get("rows"))),
        ("capstone.retired_claims", _as_list(capstone.get("retired_claims"))),
        ("capstone.bounded_claims", _as_list(capstone.get("bounded_claims"))),
    ):
        for row in rows:
            mapping = _as_mapping(row)
            if _is_repair_related(mapping) and not _has_consumable_fields(mapping):
                errors.append(
                    {
                        "source": key,
                        "row_id": _row_id(mapping),
                        "reason": "missing_row_source_fields",
                    }
                )
    for group_name, rows in (
        ("retired_repair_claims", retired_claims),
        ("still_bounded_repair_claims", bounded_claims),
        ("extracted_repair_blockers", blockers),
    ):
        for row in rows:
            if row.get("matrix_v20_consumable") is not True:
                errors.append(
                    {
                        "source": group_name,
                        "row_id": str(row.get("row_id") or row.get("claim_id")),
                        "reason": "normalized_decision_missing_consumable_fields",
                    }
                )
    return errors


def _blocked_reasons(
    *,
    source_errors: list[Mapping[str, Any]],
    manifest_updates: list[Mapping[str, Any]],
    retired_claims: list[Mapping[str, Any]],
    bounded_claims: list[Mapping[str, Any]],
    consumability_errors: list[Mapping[str, Any]],
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if source_errors:
        reasons.append("required source artifacts missing or malformed")
    if matrix.get("matrix_v19_ready") is not True:
        reasons.append("matrix v19 is not ready")
    if capstone.get("capstone_ready") is not True:
        reasons.append("capstone v285 is not ready")
    if not retired_claims:
        reasons.append("no retired repair headline claims found")
    if not bounded_claims:
        reasons.append("no bounded repair claims found")
    if any(row.get("applied") is not True for row in manifest_updates):
        reasons.append("manifest update missing")
    if consumability_errors:
        reasons.append("matrix v20 cannot consume every repair decision")
    return reasons


def _rerun_prerequisites() -> list[JsonDict]:
    return [
        {
            "gate": "deterministic_fingerprint",
            "required": True,
            "exact_evidence_needed": "Stable transcript/content fingerprint for every before/after repair case.",
            "checker_authority": "checked-in fingerprint artifact with SHA-256 per transcript",
        },
        {
            "gate": "seed",
            "required": True,
            "exact_evidence_needed": "Top-level random_seed plus per-tool seed policy used by the repair run.",
            "checker_authority": "artifact metadata and reproducibility checksum",
        },
        {
            "gate": "duration_sanity",
            "required": True,
            "exact_evidence_needed": "Wall-clock duration consistent with the declared inference substrate.",
            "checker_authority": "scripts/adversarial_verify.py duration and methodology checks",
        },
        {
            "gate": "de_tautology_metrics",
            "required": True,
            "exact_evidence_needed": "Independently derived pass@1, pass@k, false-accept, syntax, and schema deltas.",
            "checker_authority": "non-self-grading metric derivation with no bit-identical distinct deltas",
        },
        {
            "gate": "verifier_gain",
            "required": True,
            "exact_evidence_needed": "Repair improves verifier outcome without increasing false accepts or intent drift.",
            "checker_authority": "independent verifier/checker comparison against source transcripts",
        },
        {
            "gate": "exact_checker_authority",
            "required": True,
            "exact_evidence_needed": "Named checker implementation, version or checksum, and source field for every pass/fail.",
            "checker_authority": "exact checker code and artifact field citations",
        },
    ]


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("repair_headline_retirement_ready") is True:
        return (
            "complete: repair_headline_retirement_ready=true; "
            f"retired={len(_as_list(artifact.get('retired_repair_claims')))}; "
            f"bounded={len(_as_list(artifact.get('still_bounded_repair_claims')))}; "
            f"rerun_prerequisites={len(_as_list(artifact.get('rerun_prerequisites')))}"
        )
    reasons = _as_list(artifact.get("blocked_reasons"))
    if "manifest update missing" in reasons:
        return "blocked_manifest_update_missing: repair headline retirement manifest entry absent"
    if "matrix v20 cannot consume every repair decision" in reasons:
        return "blocked_matrix_v20_not_consumable: repair decisions lack source fields"
    return "blocked_precondition: repair headline retirement ledger incomplete"


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}
