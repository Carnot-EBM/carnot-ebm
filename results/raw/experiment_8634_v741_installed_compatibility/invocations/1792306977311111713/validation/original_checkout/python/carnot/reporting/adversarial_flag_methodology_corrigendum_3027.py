"""Build the Exp 3027 adversarial-flag methodology corrigendum.

Spec refs: REQ-REPORT-3027, SCENARIO-REPORT-3027.

This module is an audit ledger, not a rerun harness. It reads the .283 repair,
validator, matrix, and capstone artifacts and decides what a later live repair
task must do. The MARCH-style rule is simple: aggregation rows can point to
problems, but the diagnosis must cite independent source fields whenever those
fields exist.
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
MILESTONE = "2026.05.284"
SCHEMA = "carnot.adversarial_flag_methodology_corrigendum.v1"
ARTIFACT = "experiment_3027_adversarial_flag_methodology_corrigendum"
OUTPUT_REL_PATH = Path("results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json")

EXP3013_REL_PATH = Path("results/experiment_3013_sota_gguf_logprob_telemetry_preflight_v1.json")
EXP3014_REL_PATH = Path("results/experiment_3014_repair_syntax_schema_failure_taxonomy_v1.json")
EXP3015_REL_PATH = Path("results/experiment_3015_cactus_style_repair_acceptance_controller_v1.json")
EXP3016_REL_PATH = Path(
    "results/experiment_3016_sota_repair_rerun_with_acceptance_controller_v1.json"
)
EXP3018_REL_PATH = Path(
    "results/experiment_3018_beaver_style_validator_frontier_certificate_v1.json"
)
MATRIX_V17_REL_PATH = Path("results/experiment_3024_cross_corpus_matrix_v17.json")
CAPSTONE_V283_REL_PATH = Path("results/experiment_3025_capstone_v283.json")

CLASSIFICATIONS = (
    "true_methodology_blocker",
    "aggregation_false_positive",
    "missing_metadata",
    "unresolved_bound",
    "hardware_blocked",
    "clean_but_not_headline",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    required: bool = True


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3013", EXP3013_REL_PATH),
    SourceSpec("exp3014", EXP3014_REL_PATH),
    SourceSpec("exp3015", EXP3015_REL_PATH),
    SourceSpec("exp3016", EXP3016_REL_PATH),
    SourceSpec("exp3018", EXP3018_REL_PATH),
    SourceSpec("exp3024", MATRIX_V17_REL_PATH),
    SourceSpec("exp3025", CAPSTONE_V283_REL_PATH),
)

SOURCE_PATH_BY_EXPERIMENT = {spec.experiment_id: spec.path for spec in SOURCE_SPECS}


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, treating bad inputs as absent audit evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive filesystem guard
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
    """REQ-REPORT-3027: build the corrigendum from checked-in artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    source_artifacts = _source_artifacts(root_path, payloads)
    required_errors = _required_source_errors(payloads)
    matrix = payloads.get("exp3024", {})
    capstone = payloads.get("exp3025", {})
    repair_decision = _repair_rerun_decision(root_path, payloads.get("exp3016", {}))
    rows = _review_rows(matrix)
    flagged_rows_reviewed = _flagged_row_count(matrix, capstone, rows)
    row_classifications = [_classification_for_row(row, payloads) for row in rows]
    by_class = _rows_by_class(row_classifications)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    ready = (
        not required_errors and bool(rows) and flagged_rows_reviewed == _count_rows(rows, "flagged")
    )
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "methodology_corrigendum_ready": ready,
        "sota_headline_ready": bool(payloads.get("exp3013", {}).get("sota_headline_ready")),
        "repair_rerun_required": bool(repair_decision["repair_rerun_required"]),
        "flagged_rows_reviewed": flagged_rows_reviewed,
        "row_classifications": row_classifications,
        "true_methodology_blockers": by_class["true_methodology_blocker"],
        "aggregation_false_positive_rows": by_class["aggregation_false_positive"],
        "missing_metadata_rows": by_class["missing_metadata"],
        "unresolved_bound_rows": by_class["unresolved_bound"],
        "hardware_blocked_rows": by_class["hardware_blocked"],
        "clean_but_not_headline_rows": by_class["clean_but_not_headline"],
        "repair_rerun_decision": repair_decision,
        "source_artifacts": source_artifacts,
        "inference_substrate": {
            "kind": "aggregation_from_upstream_artifacts",
            "no_live_llm_inference": True,
            "no_historical_artifact_rewrite": True,
            "no_top_level_live_model_metadata": True,
            "source_model_metadata_location": "source_artifacts[].collected_fields",
        },
        "march_audit_principle": "source_row_does_not_grade_itself",
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "status_updates_written": False,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(ready, repair_decision, flagged_rows_reviewed, by_class),
    }
    if required_errors:
        artifact["required_source_errors"] = required_errors
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3027 deliverable JSON."""

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


def _source_artifacts(
    root: Path, payloads: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        _source_artifact(root, spec, payloads.get(spec.experiment_id, {})) for spec in SOURCE_SPECS
    ]


def _source_artifact(root: Path, spec: SourceSpec, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = root / spec.path
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "present": path.is_file(),
        "required": spec.required,
        "readable_json_object": bool(payload),
        "sha256": sha256_file(path),
        "collected_fields": _collected_fields(payload),
    }


def _collected_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "duration_s": payload.get("duration_s"),
        "model_specs": payload.get("model_specs"),
        "inference_substrate": payload.get("inference_substrate"),
        "live_transcript_paths": payload.get("live_transcript_paths"),
        "transcript_paths": payload.get("transcript_paths"),
        "transcript_sha256": payload.get("transcript_sha256"),
        "transcript_sha256s": payload.get("transcript_sha256s"),
        "random_seed": payload.get("random_seed"),
        "model_checksums": payload.get("model_checksums"),
        "reproducibility_checksum": payload.get("reproducibility_checksum"),
        "flagged_adversarial": payload.get("flagged_adversarial"),
        "corrigendum_pending": payload.get("corrigendum_pending"),
        "sota_headline_ready": payload.get("sota_headline_ready"),
        "paper_ready": payload.get("paper_ready"),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _required_source_errors(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "experiment_id": spec.experiment_id,
            "path": spec.path.as_posix(),
            "reason": "missing_or_malformed_artifact",
        }
        for spec in SOURCE_SPECS
        if spec.required and not payloads.get(spec.experiment_id)
    ]


def _review_rows(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = matrix.get("rows")
    if not isinstance(raw_rows, list):
        return []
    rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    return [
        row
        for row in rows
        if row.get("status") in {"flagged", "blocked", "gated-skipped", "missing"}
        or _is_clean_nonheadline_row(row)
    ]


def _is_clean_nonheadline_row(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "clean":
        return False
    claim_class = str(row.get("claim_class") or "")
    substrate = str(row.get("inference_substrate") or "")
    return claim_class in {
        "fr11_self_learning_controller",
        "ssqa_gate_artifact",
    } or substrate.startswith("cached_")


def _flagged_row_count(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> int:
    capstone_flagged = capstone.get("flagged_rows")
    if isinstance(capstone_flagged, list):
        return len(capstone_flagged)
    matrix_count = matrix.get("flagged_count")
    if isinstance(matrix_count, int):
        return matrix_count
    return _count_rows(rows, "flagged")


def _count_rows(rows: list[dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if row.get("status") == status)


def _classification_for_row(
    row: Mapping[str, Any], payloads: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    row_id = str(row.get("row_id") or "unknown")
    source_exp = str(row.get("source_experiment_id") or "")
    source_payload = payloads.get(source_exp, {})
    source_path = SOURCE_PATH_BY_EXPERIMENT.get(source_exp, MATRIX_V17_REL_PATH)
    classification, supporting_fields, rationale = _classification_reason(row, source_payload)
    return {
        "row_id": row_id,
        "source_experiment_id": source_exp,
        "matrix_status": str(row.get("status") or ""),
        "classification": classification,
        "source_artifact_path": source_path.as_posix(),
        "supporting_fields": supporting_fields,
        "rationale": rationale,
        "march_audit_principle": "source_row_does_not_grade_itself",
    }


def _classification_reason(
    row: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], str]:
    row_id = str(row.get("row_id") or "")
    status = str(row.get("status") or "")
    claim_class = str(row.get("claim_class") or "")
    substrate = str(row.get("inference_substrate") or "")
    summary = _mapping(row.get("summary"))
    flags = _string_list(row.get("upstream_flags"))

    if status in {"blocked", "gated-skipped"} and _hardware_like(row_id, claim_class, substrate):
        return (
            "hardware_blocked",
            [_field("rows[].status", status), _field("rows[].claim_class", claim_class)],
            "The row is a hardware or SSQA gate, not live repair evidence.",
        )
    if status == "missing":
        return (
            "missing_metadata",
            [_field("rows[].status", status), _field("rows[].summary", summary)],
            "The matrix reports absent source evidence.",
        )
    if row_id.startswith("carry_forward_") and status == "flagged":
        return (
            "aggregation_false_positive",
            [_field("rows[].summary.source_status", summary.get("source_status"))],
            "The current matrix carried forward an older flag; it is not new .283 live evidence.",
        )
    if row_id == "exp3016_repair_acceptance_controller":
        missing = _missing_repair_metadata(source_payload)
        if missing:
            return (
                "missing_metadata",
                [_field(name, source_payload.get(name)) for name in missing],
                "The live repair row is positive but cannot be reconstructed cleanly while required metadata is absent.",
            )
    if row_id == "exp3018_beaver_frontier_certificate" and (
        _int_or(source_payload.get("unresolved_count"), 0) > 0
        or _mapping(source_payload.get("probability_bound_policy")).get(
            "exact_probability_computed"
        )
        is False
    ):
        return (
            "unresolved_bound",
            [
                _field("unresolved_count", source_payload.get("unresolved_count")),
                _field("probability_bound_policy", source_payload.get("probability_bound_policy")),
            ],
            "The validator frontier is useful, but unresolved probability bounds remain explicit.",
        )
    if row_id == "exp3019_fr11_feasibility_channel" or summary.get("tautology_risk_flag") is True:
        return (
            "true_methodology_blocker",
            [_field("rows[].summary.tautology_risk_flag", summary.get("tautology_risk_flag"))],
            "The row reports a tautology risk and must not grade itself.",
        )
    if row_id == "exp3015_acceptance_controller":
        return (
            "clean_but_not_headline",
            [
                _field("inference_substrate", source_payload.get("inference_substrate")),
                _field("llm_judge_used", source_payload.get("llm_judge_used")),
            ],
            "The offline controller can inform gates, but it is not live headline repair evidence.",
        )
    if _deterministic_false_positive(substrate, source_payload, flags):
        return (
            "aggregation_false_positive",
            [
                _field(
                    "inference_substrate", source_payload.get("inference_substrate") or substrate
                ),
                _field("live_llm_inference_run", source_payload.get("live_llm_inference_run")),
            ],
            "The adversarial duration/live-metadata flag is explained by deterministic cached replay.",
        )
    if _is_clean_nonheadline_row(row):
        return (
            "clean_but_not_headline",
            [_field("rows[].status", status), _field("rows[].inference_substrate", substrate)],
            "The evidence is bounded utility or cached replay, not a headline claim.",
        )
    return (
        "true_methodology_blocker",
        [_field("rows[].upstream_flags", flags), _field("rows[].status", status)],
        "No independent source field resolves the adversarial flag.",
    )


def _missing_repair_metadata(payload: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if not _string_list(payload.get("live_transcript_paths")):
        missing.append("live_transcript_paths")
    if not _mapping(payload.get("model_specs")):
        missing.append("model_specs")
    if "random_seed" not in payload:
        missing.append("random_seed")
    if not _mapping(payload.get("model_checksums")):
        missing.append("model_checksums")
    if not _transcript_hash_evidence(payload):
        missing.append("transcript_sha256s")
    return missing


def _repair_rerun_decision(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "live_transcripts": _presence_check(
            "live_transcript_paths",
            _string_list(payload.get("live_transcript_paths")),
            transcript_file_presence=_transcript_file_presence(
                root, _string_list(payload.get("live_transcript_paths"))
            ),
        ),
        "model_specs": _presence_check("model_specs", _mapping(payload.get("model_specs"))),
        "random_seed": _presence_check("random_seed", payload.get("random_seed")),
        "model_checksums": _presence_check(
            "model_checksums", _mapping(payload.get("model_checksums"))
        ),
        "transcript_hashes": _presence_check(
            "transcript_sha256s", _transcript_hash_evidence(payload)
        ),
    }
    missing = [name for name, check in checks.items() if check["status"] != "present"]
    required = bool(missing)
    return {
        "repair_rerun_required": required,
        "decision": "live_rerun_required" if required else "reconstruct_from_existing_transcripts",
        "reason": (
            "Exp 3016 is missing required reconstruction metadata: " + ", ".join(missing)
            if required
            else "Exp 3016 has transcript paths, model specs, seed, checksum, and transcript-hash evidence."
        ),
        "metadata_checks": checks,
    }


def _presence_check(source_field: str, value: object, **extra: object) -> dict[str, Any]:
    present = value not in (None, [], {})
    check = {
        "source_field": source_field,
        "status": "present" if present else "missing",
        "value": value,
    }
    check.update(extra)
    return check


def _transcript_file_presence(root: Path, paths: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "present": (root / path).is_file(),
            "sha256": sha256_file(root / path),
        }
        for path in paths
    ]


def _transcript_hash_evidence(payload: Mapping[str, Any]) -> object:
    return (
        payload.get("transcript_sha256s")
        or payload.get("live_transcript_sha256s")
        or payload.get("transcript_sha256")
    )


def _hardware_like(row_id: str, claim_class: str, substrate: str) -> bool:
    text = " ".join([row_id, claim_class, substrate]).lower()
    return any(token in text for token in ("gatemate", "hardware", "ssqa", "flash", "smoke"))


def _deterministic_false_positive(
    substrate: str, source_payload: Mapping[str, Any], flags: list[str]
) -> bool:
    source_substrate = str(source_payload.get("inference_substrate") or substrate)
    deterministic = "deterministic" in source_substrate or "cached" in source_substrate
    flagged_duration = any(
        "DURATION_TOO_SHORT" in flag or "METHODOLOGY_MISSING" in flag for flag in flags
    )
    no_live = (
        source_payload.get("live_llm_inference_run") is False or "no_live_llm" in source_substrate
    )
    return deterministic and (flagged_duration or no_live)


def _rows_by_class(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        classification: [row for row in rows if row["classification"] == classification]
        for classification in CLASSIFICATIONS
    }


def _honest_verdict(
    ready: bool,
    repair_decision: Mapping[str, Any],
    flagged_rows_reviewed: int,
    by_class: Mapping[str, list[dict[str, Any]]],
) -> str:
    if not ready:
        return "blocked_required_corrigendum_sources_missing"
    rerun = str(repair_decision["decision"])
    return (
        "complete: methodology_corrigendum_ready=true; "
        f"repair_decision={rerun}; "
        f"flagged_rows_reviewed={flagged_rows_reviewed}; "
        f"missing_metadata={len(by_class['missing_metadata'])}; "
        f"unresolved_bound={len(by_class['unresolved_bound'])}"
    )


def _field(field: str, value: object) -> dict[str, Any]:
    return {"field": field, "value": value}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _int_or(value: object, fallback: int) -> int:
    return value if isinstance(value, int) else fallback
