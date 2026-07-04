"""Exp 5223: GAP-4 flagged-pool authenticity audit.

Spec refs: REQ-REPORT-5223, SCENARIO-REPORT-5223-QUARANTINE,
SCENARIO-REPORT-5223-PREFLIGHT.

This audit does not score GAP-4. It records why the flagged v477 pool is not
headline evidence and exposes a reusable preflight guard for the next canonical
pool builder and validation run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5223_gap4_flagged_pool_authenticity_audit_v478"
EXPERIMENT_ID = 5223
SCHEMA = "carnot.gap4_flagged_pool_authenticity_audit_5223.v1"
RESULT_RELATIVE_PATH = "results/experiment_5223_gap4_flagged_pool_authenticity_audit_v478.json"
CANONICAL_SCHEMA_RELATIVE_PATH = "python/carnot/schemas/gap4_candidate_record_v1.json"
EXP5211_RELATIVE_PATH = "results/experiment_5211_gap4_sota_local_candidate_expansion_v477.json"
EXP5211_CHECKPOINT_RELATIVE_PATH = (
    "results/experiment_5211_gap4_sota_local_candidate_expansion_v477.checkpoint.json"
)
EXP5212_RELATIVE_PATH = "results/experiment_5212_gap4_scale_validation_gated_v477.json"
QUARANTINED_ARTIFACTS = (
    EXP5211_RELATIVE_PATH,
    EXP5211_CHECKPOINT_RELATIVE_PATH,
    EXP5212_RELATIVE_PATH,
)
INFERENCE_SUBSTRATE = "artifact_provenance_audit"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
LIVE_GGUF_DURATION_FLOOR_S = 60.0
SPEC_REFS = [
    "REQ-REPORT-5223",
    "SCENARIO-REPORT-5223-QUARANTINE",
    "SCENARIO-REPORT-5223-PREFLIGHT",
]

CANONICAL_CANDIDATE_REQUIRED_FIELDS = (
    "candidate_id",
    "source_task_id",
    "model_id",
    "model_path_or_digest",
    "prompt_digest",
    "random_seed",
    "generation_started_at",
    "generation_duration_s",
    "decoding_protocol",
    "pass_at_1_fields",
    "pass_at_2_fields",
    "validation_inputs_digest",
    "provenance_kind",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "gap4_pool_repairable": {
        "principle": (
            "BARE top-level boolean. True only if existing rows can be repaired "
            "without inventing model/protocol provenance."
        )
    },
    "validated_pool_n": {
        "principle": "BARE top-level integer. Count of rows with scored protocol validation.",
    },
    "protocol_fields_complete": {
        "principle": (
            "BARE top-level boolean. True only when every accepted candidate has "
            "complete pass@1 and pass@2 protocol fields."
        )
    },
    "quarantined_artifacts": {
        "principle": "List of flagged v477 GAP-4 artifact paths excluded from headlines.",
    },
    "canonical_schema_path": {
        "principle": "Path to the canonical GAP-4 candidate-record JSON schema.",
    },
    "guard_tests_added": {
        "principle": "True only when this task added tests that exercise the preflight guard.",
    },
    "tests_run": {
        "principle": "Commands actually run for this artifact, with pass/fail status.",
    },
    "inference_substrate": {
        "principle": "Must be artifact_provenance_audit.",
    },
    "honest_verdict": {
        "principle": (
            "Must start with complete:/complete_/success:/success_ and state whether "
            "the old GAP-4 pool is repairable or must be regenerated."
        )
    },
}

REQUIRED_FIELDS = (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "gap4_pool_repairable",
    "validated_pool_n",
    "protocol_fields_complete",
    "quarantined_artifacts",
    "canonical_schema_path",
    "guard_tests_added",
    "tests_run",
    "inference_substrate",
    "honest_verdict",
    "preflight_passed",
    "preflight_reasons",
    "artifact_findings",
    "canonical_candidate_required_fields",
    "field_principles",
    "duration_s",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class Gap4PreflightResult:
    """Machine-readable outcome for a GAP-4 pool validation preflight."""

    passed: bool
    reasons: list[str]
    validated_pool_n: int
    protocol_fields_complete: bool
    gap4_pool_repairable: bool
    checked_candidate_n: int


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value and "principle" in value:
        return value.get("value")
    return value


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _wrapped_int(value: Any) -> int | None:
    value = _unwrap(value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _wrapped_number(value: Any) -> float | None:
    value = _unwrap(value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _candidate_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    rows = artifact.get("candidate_rows")
    if not isinstance(rows, list):
        rows = artifact.get("events")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _scored_rows(artifact: Mapping[str, Any] | None) -> list[JsonDict]:
    if artifact is None:
        return []
    rows = artifact.get("scored_rows")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _row_model_id(row: Mapping[str, Any]) -> Any:
    return row.get("model_id") or row.get("model_hf_id") or row.get("hf_id")


def _row_model_path(row: Mapping[str, Any]) -> Any:
    return row.get("model_path_or_digest") or row.get("model_path") or row.get("model_digest")


def _row_has_model_provenance(row: Mapping[str, Any]) -> bool:
    return _nonempty_string(_row_model_id(row)) and _nonempty_string(_row_model_path(row))


def _row_has_seed(row: Mapping[str, Any]) -> bool:
    seed = row.get("random_seed")
    return isinstance(seed, int) and not isinstance(seed, bool)


def _row_has_pass_at_1(row: Mapping[str, Any]) -> bool:
    fields = row.get("pass_at_1_fields")
    if isinstance(fields, Mapping):
        return (
            isinstance(fields.get("vote_top1"), bool)
            and isinstance(fields.get("gated_top1"), bool)
            and _nonempty_string(fields.get("scoring_protocol"))
        )
    return isinstance(row.get("vote_top1"), bool) and isinstance(row.get("gated_top1"), bool)


def _row_has_pass_at_2(row: Mapping[str, Any]) -> bool:
    fields = row.get("pass_at_2_fields")
    if isinstance(fields, Mapping):
        return (
            isinstance(fields.get("vote_top2"), bool)
            and isinstance(fields.get("gated_top2"), bool)
            and _nonempty_string(fields.get("scoring_protocol"))
        )
    return isinstance(row.get("vote_top2"), bool) and isinstance(row.get("gated_top2"), bool)


def _row_has_generation_provenance(row: Mapping[str, Any]) -> bool:
    return (
        _nonempty_string(row.get("prompt_digest"))
        and _nonempty_string(row.get("generation_started_at"))
        and _wrapped_number(row.get("generation_duration_s")) is not None
        and isinstance(row.get("decoding_protocol"), Mapping)
        and bool(row.get("decoding_protocol"))
    )


def canonical_candidate_record_errors(row: Mapping[str, Any]) -> list[str]:
    """Return canonical GAP-4 candidate-record schema errors for one row."""

    errors: list[str] = []
    for field in CANONICAL_CANDIDATE_REQUIRED_FIELDS:
        if field in {
            "model_id",
            "model_path_or_digest",
            "random_seed",
            "generation_duration_s",
            "decoding_protocol",
            "pass_at_1_fields",
            "pass_at_2_fields",
        }:
            continue
        if field not in row or row.get(field) is None:
            errors.append(f"missing_{field}")

    if not _nonempty_string(row.get("model_id")) or not _nonempty_string(
        row.get("model_path_or_digest")
    ):
        errors.append("missing_model_provenance")
    if not _row_has_seed(row):
        errors.append("missing_random_seed")
    if _wrapped_number(row.get("generation_duration_s")) is None:
        errors.append("missing_generation_duration_s")
    if not isinstance(row.get("decoding_protocol"), Mapping) or not row.get("decoding_protocol"):
        errors.append("missing_decoding_protocol")
    if not _row_has_pass_at_1(row):
        errors.append("missing_pass_at_1_fields")
    if not _row_has_pass_at_2(row):
        errors.append("missing_pass_at_2_fields")
    return sorted(dict.fromkeys(errors))


def _models_used_missing(pool_artifact: Mapping[str, Any]) -> bool:
    models_used = pool_artifact.get("models_used")
    return not (
        isinstance(models_used, list)
        and any(isinstance(item, str) and item.strip() for item in models_used)
    )


def _validation_n_scored(validation_artifact: Mapping[str, Any] | None) -> int:
    if validation_artifact is None:
        return 0
    wrapped = _wrapped_int(validation_artifact.get("n_scored"))
    if wrapped is not None:
        return wrapped
    return len(_scored_rows(validation_artifact))


def _live_prompted_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("live_prompted") is True)


def _demo_lookup_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if str(row.get("repair_strategy") or "") == "demo_lookup_same_shape")


def _tautology_shaped_pool_ready(pool_artifact: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    n = _wrapped_int(pool_artifact.get("candidate_pool_n"))
    accepted = _wrapped_int(pool_artifact.get("accepted_rows"))
    repairs = _wrapped_int(pool_artifact.get("repair_attempts"))
    usable = pool_artifact.get("gap4_expansion_usable") is True
    return bool(
        usable
        and n is not None
        and n >= 120
        and accepted == n
        and repairs == n
        and rows
        and _live_prompted_count(rows) <= 1
        and _demo_lookup_count(rows) == len(rows)
    )


def _generation_duration_too_short(pool_artifact: Mapping[str, Any]) -> bool:
    duration = _wrapped_number(pool_artifact.get("duration_s"))
    substrate = str(_unwrap(pool_artifact.get("inference_substrate")) or "")
    if duration is None:
        return False
    return "live_llm_generation" in substrate and duration < LIVE_GGUF_DURATION_FLOOR_S


def preflight_gap4_validation(
    *,
    pool_artifact: Mapping[str, Any],
    validation_artifact: Mapping[str, Any] | None = None,
) -> Gap4PreflightResult:
    """Reject GAP-4 validation inputs that lack canonical provenance."""

    rows = _candidate_rows(pool_artifact)
    reasons: list[str] = []

    if _models_used_missing(pool_artifact):
        reasons.append("missing_models_used")
    if rows and any(not _row_has_model_provenance(row) for row in rows):
        reasons.append("missing_model_provenance")
    if rows and any(not _row_has_seed(row) for row in rows):
        reasons.append("missing_random_seed")
    if rows and any(not _row_has_generation_provenance(row) for row in rows):
        reasons.append("missing_generation_provenance")
    if rows and any(not _row_has_pass_at_1(row) for row in rows):
        reasons.append("missing_protocol_pass1_fields")
    if rows and any(not _row_has_pass_at_2(row) for row in rows):
        reasons.append("missing_protocol_pass2_fields")

    validated_pool_n = _validation_n_scored(validation_artifact)
    if validation_artifact is not None and validated_pool_n == 0:
        reasons.append("validation_n_scored_zero")
    if _generation_duration_too_short(pool_artifact):
        reasons.append("generation_duration_too_short")
    if _tautology_shaped_pool_ready(pool_artifact, rows):
        reasons.append("tautology_shaped_pool_ready")

    protocol_fields_complete = bool(rows) and all(
        _row_has_pass_at_1(row) and _row_has_pass_at_2(row) for row in rows
    )
    unrepairable = {
        "missing_random_seed",
        "missing_generation_provenance",
        "missing_protocol_pass1_fields",
        "missing_protocol_pass2_fields",
        "validation_n_scored_zero",
    }
    unique_reasons = sorted(dict.fromkeys(reasons))
    gap4_pool_repairable = bool(rows) and not any(reason in unrepairable for reason in unique_reasons)
    return Gap4PreflightResult(
        passed=not unique_reasons,
        reasons=unique_reasons,
        validated_pool_n=validated_pool_n,
        protocol_fields_complete=protocol_fields_complete,
        gap4_pool_repairable=gap4_pool_repairable,
        checked_candidate_n=len(rows),
    )


def require_gap4_pool_preflight(result: Gap4PreflightResult) -> None:
    """Raise if a GAP-4 validation task tries to proceed with a failed preflight."""

    if not result.passed:
        raise ValueError("GAP-4 pool preflight failed: " + ", ".join(result.reasons))


def _artifact_flags(payload: Mapping[str, Any]) -> list[JsonDict]:
    flags = payload.get("corrigendum_pending")
    if not isinstance(flags, list):
        return []
    return [dict(flag) for flag in flags if isinstance(flag, Mapping)]


def _exp5211_findings(payload: Mapping[str, Any], path: str) -> JsonDict:
    rows = _candidate_rows(payload)
    missing_seed = sum(1 for row in rows if not _row_has_seed(row))
    missing_pass1 = sum(1 for row in rows if not _row_has_pass_at_1(row))
    missing_pass2 = sum(1 for row in rows if not _row_has_pass_at_2(row))
    missing_generation = sum(1 for row in rows if not _row_has_generation_provenance(row))
    live_prompted = _live_prompted_count(rows)
    demo_lookup = _demo_lookup_count(rows)
    flags = _artifact_flags(payload)
    blockers = []
    if _models_used_missing(payload):
        blockers.append("models_used is empty despite a pool-ready artifact")
    if missing_seed:
        blockers.append(f"{missing_seed} candidate rows are missing random_seed")
    if missing_generation:
        blockers.append(f"{missing_generation} candidate rows are missing canonical generation fields")
    if missing_pass1:
        blockers.append(f"{missing_pass1} candidate rows are missing pass@1 protocol fields")
    if missing_pass2:
        blockers.append(f"{missing_pass2} candidate rows are missing pass@2 protocol fields")
    if _generation_duration_too_short(payload):
        blockers.append(
            f"duration_s={payload.get('duration_s')} is below the "
            f"{LIVE_GGUF_DURATION_FLOOR_S:.0f}s live-GGUF floor"
        )
    if _tautology_shaped_pool_ready(payload, rows):
        blockers.append(
            f"pool-ready counts collapse to candidate_pool_n={payload.get('candidate_pool_n')}, "
            f"accepted_rows={payload.get('accepted_rows')}, repair_attempts={payload.get('repair_attempts')} "
            f"with live_prompted_rows={live_prompted} and demo_lookup_rows={demo_lookup}"
        )
    if flags:
        blockers.append("adversarial_verify recorded flags on the artifact")
    return {
        "path": path,
        "exists": bool(payload),
        "headline_eligible": False,
        "candidate_pool_n": int(payload.get("candidate_pool_n") or 0),
        "models_used": payload.get("models_used") if isinstance(payload.get("models_used"), list) else [],
        "candidate_rows": len(rows),
        "missing_random_seed_rows": missing_seed,
        "missing_generation_provenance_rows": missing_generation,
        "missing_pass_at_1_rows": missing_pass1,
        "missing_pass_at_2_rows": missing_pass2,
        "live_prompted_rows": live_prompted,
        "demo_lookup_rows": demo_lookup,
        "adversarial_flags": flags,
        "headline_blockers": blockers,
    }


def _exp5212_findings(payload: Mapping[str, Any], path: str) -> JsonDict:
    n_scored = _validation_n_scored(payload)
    excluded = _wrapped_int(payload.get("excluded_rows")) or 0
    blockers = []
    if n_scored == 0:
        blockers.append("n_scored=0; no protocol-scored rows exist")
    if payload.get("failure_mode"):
        blockers.append(f"failure_mode={payload.get('failure_mode')}")
    if _artifact_flags(payload):
        blockers.append("adversarial_verify recorded flags on the artifact")
    return {
        "path": path,
        "exists": bool(payload),
        "headline_eligible": False,
        "validated_pool_n": n_scored,
        "excluded_rows": excluded,
        "failure_mode": payload.get("failure_mode"),
        "exclusion_summary": dict(payload.get("exclusion_summary") or {}),
        "adversarial_flags": _artifact_flags(payload),
        "headline_blockers": blockers,
    }


def _checkpoint_findings(payload: Mapping[str, Any], path: str) -> JsonDict:
    rows = _candidate_rows(payload)
    return {
        "path": path,
        "exists": bool(payload),
        "headline_eligible": False,
        "candidate_rows": len(rows),
        "missing_random_seed_rows": sum(1 for row in rows if not _row_has_seed(row)),
        "missing_pass_at_1_rows": sum(1 for row in rows if not _row_has_pass_at_1(row)),
        "missing_pass_at_2_rows": sum(1 for row in rows if not _row_has_pass_at_2(row)),
        "headline_blockers": [
            "checkpoint preserves the same noncanonical rows; it is audit evidence, not repair provenance"
        ],
    }


def _verdict(result: Gap4PreflightResult) -> str:
    if result.gap4_pool_repairable:
        return "success: old GAP-4 pool repairable under canonical audit; no headline evidence emitted"
    return (
        "complete: old GAP-4 pool must be regenerated or rebuilt by a canonical-pool "
        "builder; exp5211/exp5212 are quarantined from headline evidence"
    )


def build_audit_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str],
    guard_tests_added: bool,
    duration_s: float,
) -> JsonDict:
    root_path = Path(root)
    exp5211 = _read_json(root_path / EXP5211_RELATIVE_PATH)
    checkpoint = _read_json(root_path / EXP5211_CHECKPOINT_RELATIVE_PATH)
    exp5212 = _read_json(root_path / EXP5212_RELATIVE_PATH)
    pool = dict(exp5211) if isinstance(exp5211, Mapping) else {}
    validation = dict(exp5212) if isinstance(exp5212, Mapping) else {}
    checkpoint_payload = dict(checkpoint) if isinstance(checkpoint, Mapping) else {}
    preflight = preflight_gap4_validation(pool_artifact=pool, validation_artifact=validation)
    findings = {
        EXP5211_RELATIVE_PATH: _exp5211_findings(pool, EXP5211_RELATIVE_PATH),
        EXP5211_CHECKPOINT_RELATIVE_PATH: _checkpoint_findings(
            checkpoint_payload, EXP5211_CHECKPOINT_RELATIVE_PATH
        ),
        EXP5212_RELATIVE_PATH: _exp5212_findings(validation, EXP5212_RELATIVE_PATH),
    }
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "gap4_pool_repairable": preflight.gap4_pool_repairable,
        "validated_pool_n": preflight.validated_pool_n,
        "protocol_fields_complete": preflight.protocol_fields_complete,
        "quarantined_artifacts": list(QUARANTINED_ARTIFACTS),
        "canonical_schema_path": CANONICAL_SCHEMA_RELATIVE_PATH,
        "guard_tests_added": bool(guard_tests_added),
        "tests_run": list(tests_run),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _verdict(preflight),
        "preflight_passed": preflight.passed,
        "preflight_reasons": list(preflight.reasons),
        "artifact_findings": findings,
        "canonical_candidate_required_fields": list(CANONICAL_CANDIDATE_REQUIRED_FIELDS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not isinstance(artifact.get("gap4_pool_repairable"), bool):
        errors.append("gap4_pool_repairable_bare_bool")
    n = artifact.get("validated_pool_n")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        errors.append("validated_pool_n_bare_int")
    protocol_complete = artifact.get("protocol_fields_complete")
    if not isinstance(protocol_complete, bool):
        errors.append("protocol_fields_complete_bare_bool")
    if protocol_complete is True and (not isinstance(n, int) or n <= 0):
        errors.append("protocol_fields_complete")
    if artifact.get("quarantined_artifacts") != list(QUARANTINED_ARTIFACTS):
        errors.append("quarantined_artifacts")
    if artifact.get("canonical_schema_path") != CANONICAL_SCHEMA_RELATIVE_PATH:
        errors.append("canonical_schema_path")
    if not isinstance(artifact.get("guard_tests_added"), bool):
        errors.append("guard_tests_added_bare_bool")
    if not isinstance(artifact.get("tests_run"), list):
        errors.append("tests_run")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(dict.fromkeys(errors))


def write_artifact(root: Path | str, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    _write_json_atomic(path, artifact)
    return path


def run(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str] = (),
    guard_tests_added: bool = True,
    now: Any = time.time,
) -> JsonDict:
    started = float(now())
    artifact = build_audit_artifact(
        root=root,
        tests_run=tests_run,
        guard_tests_added=guard_tests_added,
        duration_s=0.0,
    )
    artifact["duration_s"] = max(0.0, round(float(now()) - started, 6))
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(artifact["honest_verdict"])
    print(f"gap4_pool_repairable={artifact['gap4_pool_repairable']}")
    print(f"validated_pool_n={artifact['validated_pool_n']}")
    print(f"protocol_fields_complete={artifact['protocol_fields_complete']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
