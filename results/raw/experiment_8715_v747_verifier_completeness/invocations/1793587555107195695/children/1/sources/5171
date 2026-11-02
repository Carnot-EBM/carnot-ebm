"""Build the Exp 3056 repair de-tautology protocol artifact.

Spec refs: REQ-REPORT-3056, SCENARIO-REPORT-3056.

This module writes a protocol, not a repair result. It reads checked-in
artifacts that already exposed the Exp 3028-style methodology blockers, turns
those blockers into explicit future-run checks, and declares the JSON fields
Exp 3059 must emit before matrix v20 can consider a local SOTA repair row.
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
MILESTONE = "2026.05.286"
SCHEMA = "carnot.repair_de_tautology.protocol.v1"
ARTIFACT = "experiment_3056_repair_de_tautology_protocol_v1"
OUTPUT_REL_PATH = Path("results/experiment_3056_repair_de_tautology_protocol_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3056_repair_de_tautology_protocol_v1.py"

EXP3028_REL_PATH = Path("results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json")
EXP3042_REL_PATH = Path("results/experiment_3042_repair_promotion_reconciliation_v3.json")
EXP3043_REL_PATH = Path("results/experiment_3043_verified_speculation_transcript_fingerprint_v1.json")
EXP3055_REL_PATH = Path("results/experiment_3055_repair_headline_retirement_and_blocker_ledger_v1.json")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")

INFERENCE_SUBSTRATE = {
    "kind": "aggregation_from_upstream_artifacts",
    "source": "checked_in_artifacts",
    "executes_models": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "no_live_llm_inference": True,
}

SOURCE_SPECS = (
    ("exp3028", EXP3028_REL_PATH, "blocked_repair_evidence", True, "json"),
    ("exp3042", EXP3042_REL_PATH, "repair_reconciliation_authority", True, "json"),
    ("exp3043", EXP3043_REL_PATH, "fingerprint_linkage_reference", True, "json"),
    ("exp3055", EXP3055_REL_PATH, "repair_blocker_ledger", True, "json"),
    ("research_references", RESEARCH_REFERENCES_REL_PATH, "aprad_reference_boundary", True, "text"),
)
REQUIRED_BLOCKER_CATEGORIES = (
    "tautology",
    "implausible_perfect_results",
    "duration",
    "missing_seed",
    "unresolved_methodology",
)
KNOWN_OBSERVED_FIELDS = (
    "pass_at_1_delta",
    "pass_at_k_delta",
    "false_accept_delta",
    "schema_failure_rate_delta",
    "syntax_failure_rate_delta",
    "duration_s",
    "random_seed",
    "model_specs",
    "reproducibility_checksum",
)
EXP3059_REQUIRED_LIVE_RUN_FIELDS = (
    "schema",
    "artifact",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "wall_clock_start_utc",
    "wall_clock_end_utc",
    "random_seed",
    "seed_log",
    "model_specs",
    "models_used",
    "decode_config",
    "inference_substrate",
    "n_tasks",
    "candidate_count",
    "baseline_metrics",
    "repair_metrics",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "pass_at_1_derivation",
    "pass_at_k_derivation",
    "per_task_results",
    "non_vacuous_outcome_summary",
    "accepted_candidate_count",
    "rejected_candidate_count",
    "false_accept_delta",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "intent_drift_count",
    "intent_preservation_checks",
    "checker_authority",
    "transcript_fingerprints",
    "raw_transcript_paths",
    "fingerprint_linkage",
    "reproducibility_checksum",
    "duration_sanity_check",
    "blocked_prior_fields_checked",
    "promotion_disqualifiers",
    "adversarial_verify_flags",
    "tests_run",
    "honest_verdict",
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while treating missing, malformed, or array JSON as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a source file digest when the protocol can cite the file."""

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
    """REQ-REPORT-3056: build the machine-checkable repair rerun protocol."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_payloads(root_path)
    source_artifacts = _source_artifacts(root_path)
    source_errors = _source_errors(source_artifacts)
    research_text = _read_text(root_path / RESEARCH_REFERENCES_REL_PATH)
    blocked_prior_fields = _blocked_prior_fields(payloads)
    aprad_reference_available = "AprAD" in research_text and "implementation" in research_text
    blocked_reasons = _blocked_reasons(
        source_errors=source_errors,
        blocked_prior_fields=blocked_prior_fields,
        aprad_reference_available=aprad_reference_available,
    )
    ready = not blocked_reasons
    duration_s = _duration(start, now_s)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "repair_de_tautology_protocol_ready": ready,
        "blocked_prior_fields": blocked_prior_fields,
        "acceptance_checks": _acceptance_checks(),
        "required_live_run_fields": list(EXP3059_REQUIRED_LIVE_RUN_FIELDS),
        "exp3059_matrix_v20_required_fields": list(EXP3059_REQUIRED_LIVE_RUN_FIELDS),
        "intent_preservation_checks": _intent_preservation_checks(aprad_reference_available),
        "duration_sanity_rule": _duration_sanity_rule(),
        "fingerprint_requirements": _fingerprint_requirements(),
        "promotion_disqualifiers": _promotion_disqualifiers(),
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "missing_source_artifacts": [
            row["path"] for row in source_artifacts if row.get("present") is not True
        ],
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_historical_artifact_rewrite": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "blocked_reasons": blocked_reasons,
        "duration_s": duration_s,
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
    """Build and persist the Exp 3056 protocol deliverable."""

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
        "exp3028": read_json_object(root / EXP3028_REL_PATH),
        "exp3042": read_json_object(root / EXP3042_REL_PATH),
        "exp3043": read_json_object(root / EXP3043_REL_PATH),
        "exp3055": read_json_object(root / EXP3055_REL_PATH),
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


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
                "readable_text": bool(_read_text(path)) if source_type == "text" else None,
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


def _blocked_prior_fields(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp3028 = payloads.get("exp3028", {})
    exp3042 = payloads.get("exp3042", {})
    corrigendum = _as_list(exp3028.get("corrigendum_pending"))
    blocked: JsonDict = {}

    tautology_rows = _rows_by_kind(corrigendum, "TAUTOLOGY")
    if tautology_rows:
        index, row = tautology_rows[0]
        blocked["tautology"] = _prior_blocker_row(
            source_artifact=EXP3028_REL_PATH.as_posix(),
            source_field=f"corrigendum_pending[{index}]",
            kind="TAUTOLOGY",
            severity=str(row.get("severity") or ""),
            detail=str(row.get("detail") or ""),
            observed_fields=_observed_fields_from_detail(str(row.get("detail") or "")),
            acceptance_check="pass_at_1_vs_pass_at_k_independent_derivation",
        )

    implausible_rows = _rows_by_kind(corrigendum, "IMPLAUSIBLE_PERFECT")
    if implausible_rows:
        observed_fields = _unique(
            field
            for _, row in implausible_rows
            for field in _observed_fields_from_detail(str(row.get("detail") or ""))
        )
        blocked["implausible_perfect_results"] = {
            "source_artifact": EXP3028_REL_PATH.as_posix(),
            "source_field": _range_source_field("corrigendum_pending", implausible_rows),
            "kind": "IMPLAUSIBLE_PERFECT",
            "severity": "info",
            "details": [str(row.get("detail") or "") for _, row in implausible_rows],
            "observed_fields": observed_fields,
            "acceptance_check": "per_case_non_vacuous_delta_evidence",
        }

    duration_rows = _rows_by_kind(corrigendum, "DURATION_TOO_SHORT")
    if duration_rows:
        index, row = duration_rows[0]
        blocked["duration"] = _prior_blocker_row(
            source_artifact=EXP3028_REL_PATH.as_posix(),
            source_field=f"corrigendum_pending[{index}]",
            kind="DURATION_TOO_SHORT",
            severity=str(row.get("severity") or ""),
            detail=str(row.get("detail") or ""),
            observed_fields=_observed_fields_from_detail(str(row.get("detail") or "")),
            acceptance_check="wall_clock_sanity_for_compute_bound_markers",
        )

    missing_seed_rows = _rows_by_kind(corrigendum, "METHODOLOGY_MISSING")
    if missing_seed_rows:
        index, row = missing_seed_rows[0]
        blocked["missing_seed"] = _prior_blocker_row(
            source_artifact=EXP3028_REL_PATH.as_posix(),
            source_field=f"corrigendum_pending[{index}]",
            kind="METHODOLOGY_MISSING",
            severity=str(row.get("severity") or ""),
            detail=str(row.get("detail") or ""),
            observed_fields=_observed_fields_from_detail(str(row.get("detail") or "")),
            acceptance_check="top_level_and_per_transcript_seed_logging",
        )

    unresolved_rows = _unresolved_methodology_rows(exp3042)
    if unresolved_rows:
        blocked["unresolved_methodology"] = {
            "source_artifact": EXP3042_REL_PATH.as_posix(),
            "source_field": "remaining_blockers",
            "kind": "UNRESOLVED_METHODOLOGY",
            "observed_fields": ["remaining_blockers", "repair_claim_status"],
            "acceptance_check": "all_prior_blockers_checked_before_matrix_v20_promotion",
            "rows": unresolved_rows,
        }
    return blocked


def _prior_blocker_row(
    *,
    source_artifact: str,
    source_field: str,
    kind: str,
    severity: str,
    detail: str,
    observed_fields: list[str],
    acceptance_check: str,
) -> JsonDict:
    return {
        "source_artifact": source_artifact,
        "source_field": source_field,
        "kind": kind,
        "severity": severity,
        "detail": detail,
        "observed_fields": observed_fields,
        "acceptance_check": acceptance_check,
    }


def _rows_by_kind(rows: Iterable[Any], kind: str) -> list[tuple[int, JsonDict]]:
    matches: list[tuple[int, JsonDict]] = []
    for index, row in enumerate(rows):
        mapping = _as_mapping(row)
        if mapping.get("kind") == kind:
            matches.append((index, mapping))
    return matches


def _range_source_field(prefix: str, rows: list[tuple[int, JsonDict]]) -> str:
    indexes = [index for index, _ in rows]
    if not indexes:
        return prefix
    if indexes == list(range(indexes[0], indexes[-1] + 1)):
        return f"{prefix}[{indexes[0]}:{indexes[-1] + 1}]"
    return ",".join(f"{prefix}[{index}]" for index in indexes)


def _observed_fields_from_detail(detail: str) -> list[str]:
    return [field for field in KNOWN_OBSERVED_FIELDS if field in detail]


def _unresolved_methodology_rows(exp3042: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in _as_list(exp3042.get("remaining_blockers")):
        mapping = _as_mapping(row)
        classification = str(mapping.get("classification") or mapping.get("status") or "")
        if classification not in {"missing_metadata", "unresolved_bound"}:
            continue
        rows.append(
            {
                "row_id": str(mapping.get("row_id") or ""),
                "classification": classification,
                "source_artifact": str(mapping.get("source_artifact") or EXP3042_REL_PATH.as_posix()),
                "source_field": str(mapping.get("source_field") or ""),
                "rationale": str(mapping.get("rationale") or ""),
                "evidence": mapping.get("evidence"),
            }
        )
    return rows


def _acceptance_checks() -> list[JsonDict]:
    return [
        {
            "id": "pass_at_1_vs_pass_at_k",
            "required": True,
            "check": (
                "Derive pass@1 from the first candidate and pass@k from the full candidate "
                "set. If k==1, declare that equality is expected; otherwise bit-identical "
                "pass@1/pass@k deltas require an explicit per-task explanation."
            ),
        },
        {
            "id": "non_vacuous_task_outcomes",
            "required": True,
            "check": (
                "Emit per-task baseline, repaired, accepted, rejected, and checker outcomes; "
                "the run cannot promote if every side-effect delta is exact zero without "
                "case-level support."
            ),
        },
        {
            "id": "seed_and_logging",
            "required": True,
            "check": "Record top-level random_seed, seed_log, raw transcript paths, and tests_run.",
        },
        {
            "id": "model_specs_identity",
            "required": True,
            "check": "Record model name, hf_id, path or cache id, checksum/hash, GPU assignment, and decode config.",
        },
        {
            "id": "checker_authority",
            "required": True,
            "check": "Name the deterministic checker, code path/checksum, source field, and pass/fail authority for every accepted repair.",
        },
    ]


def _intent_preservation_checks(aprad_reference_available: bool) -> list[JsonDict]:
    return [
        {
            "id": "draft_intent_anchor",
            "required": True,
            "requirement": (
                "Record original task prompt, draft intent summary/hash, failing trace, and "
                "expected behavior before applying repair gates."
            ),
        },
        {
            "id": "aprad_inspired_distribution_distortion_guard",
            "required": True,
            "requirement": (
                "AprAD-inspired: preserve draft intent while satisfying hard verifier gates; "
                "reject repairs that satisfy tests by changing the task, dropping constraints, "
                "or replacing the requested behavior."
            ),
            "reference_available": aprad_reference_available,
            "claims_aprad_implementation": False,
        },
        {
            "id": "intent_drift_zero_for_promotion",
            "required": True,
            "requirement": (
                "Set intent_drift_count and per-case intent_drift; matrix v20 promotion "
                "requires intent_drift_count == 0."
            ),
        },
    ]


def _duration_sanity_rule() -> JsonDict:
    return {
        "applies_to": "live local SOTA repair runs",
        "minimum_live_compute_duration_s": 60.0,
        "applies_when_any_marker_present": ["GGUF", "CUDA", "live model", "model_specs"],
        "requires_monotonic_wall_clock": True,
        "failure_action": "disqualify_matrix_v20_promotion",
        "source_blocker": "exp3028.corrigendum_pending[DURATION_TOO_SHORT]",
    }


def _fingerprint_requirements() -> list[JsonDict]:
    fields = (
        "transcript_fingerprints[].prompt_hash",
        "transcript_fingerprints[].raw_output_hash",
        "transcript_fingerprints[].normalized_output_hash",
        "transcript_fingerprints[].model_hash_or_cache_path",
        "transcript_fingerprints[].seed",
        "transcript_fingerprints[].run_index",
        "raw_transcript_paths[]",
        "reproducibility_checksum",
    )
    return [
        {
            "field": field,
            "required": True,
            "source_reference": EXP3043_REL_PATH.as_posix(),
        }
        for field in fields
    ]


def _promotion_disqualifiers() -> list[JsonDict]:
    rows = (
        ("prior_tautology_not_cleared", "TAUTOLOGY remains in blocked_prior_fields_checked."),
        (
            "pass_at_1_pass_at_k_delta_bit_identical_without_k1_declaration",
            "pass@1 and pass@k deltas match exactly while k>1 or candidate ordering is missing.",
        ),
        (
            "implausible_perfect_delta_without_per_case_evidence",
            "Exact-zero false-accept, syntax, or schema deltas lack per-case checker support.",
        ),
        ("duration_too_short_for_live_compute", "Live compute markers appear but duration_s is below the sanity rule."),
        ("missing_random_seed_or_seed_log", "Top-level seed or per-transcript seed log is missing."),
        ("missing_model_specs_identity", "model_specs lacks model identity, path/cache id, checksum/hash, or GPU assignment."),
        ("missing_transcript_fingerprints", "Raw transcript paths or transcript fingerprints are missing."),
        (
            "checker_authority_missing_or_self_graded",
            "Accepted repair was graded by the generator or lacks exact deterministic checker authority.",
        ),
        ("intent_drift_detected", "Any accepted repair changes task intent or drops required constraints."),
        (
            "false_accept_or_syntax_schema_regression",
            "False accepts, syntax failures, or schema failures increase versus baseline.",
        ),
        (
            "legacy_smoke_or_non_headline_model_used",
            "Evidence uses smoke-only or non-headline models for a headline local SOTA claim.",
        ),
        (
            "unresolved_methodology_blocker_present",
            "Exp 3042/3055 methodology, bounded, or retirement blockers remain attached.",
        ),
    )
    return [{"id": row_id, "required_clearance": requirement} for row_id, requirement in rows]


def _blocked_reasons(
    *,
    source_errors: list[Mapping[str, Any]],
    blocked_prior_fields: Mapping[str, Any],
    aprad_reference_available: bool,
) -> list[str]:
    reasons: list[str] = []
    if source_errors:
        reasons.append("required source artifacts missing or malformed")
    missing = [
        category
        for category in REQUIRED_BLOCKER_CATEGORIES
        if category not in blocked_prior_fields
    ]
    if missing:
        reasons.append(f"missing blocked_prior_fields categories: {', '.join(missing)}")
    if not aprad_reference_available:
        reasons.append("AprAD reference unavailable")
    return reasons


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("repair_de_tautology_protocol_ready") is True:
        return (
            "complete: repair_de_tautology_protocol_ready=true; "
            f"blocked_prior_fields={len(_as_mapping(artifact.get('blocked_prior_fields')))}; "
            f"required_live_run_fields={len(_as_list(artifact.get('required_live_run_fields')))}"
        )
    return (
        "blocked_precondition: repair de-tautology protocol incomplete; "
        f"reasons={len(_as_list(artifact.get('blocked_reasons')))}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}
