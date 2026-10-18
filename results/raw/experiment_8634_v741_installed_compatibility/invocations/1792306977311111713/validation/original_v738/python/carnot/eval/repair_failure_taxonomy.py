"""Exp 3014 deterministic repair-failure taxonomy for cached Exp 3003 rows.

Spec: REQ-CODE-3014, SCENARIO-CODE-3014.

This module diagnoses why the Exp 3003 repair rerun stayed flagged. It reads
only checked-in artifacts, cached transcripts, verifier logs, and patch files,
then replays parser/schema checks plus original and metamorphic validators. No
live model call is made here; the output is a controller-facing failure table
and a minimal gate plan for later live reruns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import ast
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as metamorphic
from carnot.eval.gated_sota_intent_preserving_repair_hard_set import (
    parse_repair_output,
    syntax_diagnostics,
)


JsonDict = dict[str, Any]
ClockFunc = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
ARTIFACT_NAME = "experiment_3014_repair_syntax_schema_failure_taxonomy_v1"
ARTIFACT_FILENAME = f"{ARTIFACT_NAME}.json"
SCHEMA = "carnot.repair_failure_taxonomy.v1"
EXP3002_FILENAME = "experiment_3002_metamorphic_repair_oracle_audit_v1.json"
EXP3003_FILENAME = "experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1.json"
RAW_REL_DIR = Path("results/raw") / ARTIFACT_NAME
TAXONOMY_TABLE_REL_PATH = RAW_REL_DIR / "taxonomy_table.jsonl"
INFERENCE_SUBSTRATE = "deterministic_cached_replay_no_live_llm"
FAILURE_ROOT_CAUSES: tuple[str, ...] = (
    "prompt-format pressure",
    "parser/schema mismatch",
    "invalid patch shape",
    "oracle ambiguity",
    "intent drift",
    "false accept",
    "tautology",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "repair_failure_taxonomy_ready",
    "taxonomy_table_path",
    "n_cached_candidates_audited",
    "syntax_failure_count",
    "schema_failure_count",
    "false_accept_count",
    "tautology_failure_count",
    "intent_drift_count",
    "recommended_acceptance_rules",
    "halluguard_ntk_claim_made",
    "honest_verdict",
)

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_PROMPT_FORMAT_MARKERS = (
    "<|channel>",
    "<channel|>",
    "```json",
    "```python",
    "**analyze",
    "**task:**",
    "draft the intent",
    "develop the patch",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths for the cached Exp 3014 taxonomy replay."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    taxonomy_table_path: Path | None = None
    hard_manifest_path: Path | None = None
    exp3002_artifact_path: Path | None = None
    exp3003_artifact_path: Path | None = None
    metamorphic_manifest_path: Path | None = None
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: ClockFunc = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_taxonomy_table_path(self) -> Path:
        return self.taxonomy_table_path or self.repo_root / TAXONOMY_TABLE_REL_PATH

    def resolved_hard_manifest_path(self) -> Path:
        return self.hard_manifest_path or self.repo_root / hard.DEFAULT_MANIFEST_REL_PATH

    def resolved_exp3002_artifact_path(self) -> Path:
        return self.exp3002_artifact_path or self.repo_root / "results" / EXP3002_FILENAME

    def resolved_exp3003_artifact_path(self) -> Path:
        return self.exp3003_artifact_path or self.repo_root / "results" / EXP3003_FILENAME


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the Exp 3014 artifact from cached candidates and validators only."""

    config = config or ExperimentConfig()
    started = config.start_time()
    exp3003_path = config.resolved_exp3003_artifact_path()
    exp3003 = _read_json_if_present(exp3003_path)
    candidates = [dict(row) for row in exp3003.get("candidate_evaluations") or []]
    cached_candidates = [
        row
        for row in candidates
        if _resolve_repo_path(config.repo_root, row.get("candidate_patch_path")).is_file()
    ]
    if not cached_candidates:
        return _blocked_artifact(config, started)

    exp3002_path = config.resolved_exp3002_artifact_path()
    exp3002 = _read_json_if_present(exp3002_path)
    hard_items = _load_hard_items(config)
    variants = _load_metamorphic_variants(config, exp3002)
    candidate_rows = [
        _audit_candidate(config, row, hard_items, variants, index)
        for index, row in enumerate(cached_candidates)
    ]
    validator_rows = _validator_taxonomy_rows(exp3002, variants)
    table_rows = [*candidate_rows, *validator_rows]
    table_path = config.resolved_taxonomy_table_path()
    _write_jsonl(table_path, table_rows)

    counts = _failure_counts(table_rows)
    ready = bool(candidate_rows and table_path.is_file())
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "repair_failure_taxonomy_ready": ready,
        "taxonomy_table_path": str(_relative_or_absolute(config.repo_root, table_path)),
        "n_cached_candidates_audited": len(candidate_rows),
        "syntax_failure_count": counts["syntax"],
        "schema_failure_count": counts["schema"],
        "false_accept_count": counts["false_accept"],
        "tautology_failure_count": counts["tautology"],
        "intent_drift_count": counts["intent_drift"],
        "recommended_acceptance_rules": _recommended_acceptance_rules(),
        "halluguard_ntk_claim_made": False,
        "honest_verdict": (
            "complete: repair failure taxonomy ready for exp3015/exp3016 gates"
            if ready
            else "blocked: repair failure taxonomy table was not written"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_llm_inference_run": False,
        "failure_root_causes": list(FAILURE_ROOT_CAUSES),
        "failure_root_cause_counts": _root_cause_counts(table_rows),
        "oracle_failure_count": counts["oracle_ambiguity"],
        "candidate_patch_paths": [
            str(
                _relative_or_absolute(
                    config.repo_root,
                    _resolve_repo_path(config.repo_root, row.get("candidate_patch_path")),
                )
            )
            for row in cached_candidates
        ],
        "verifier_log_paths": [
            str(
                _relative_or_absolute(
                    config.repo_root,
                    _resolve_repo_path(config.repo_root, row.get("verifier_log_path")),
                )
            )
            for row in cached_candidates
            if row.get("verifier_log_path")
        ],
        "source_artifacts": _source_artifacts(config, exp3002, exp3003),
        "taxonomy_table_sha256": _sha256_file(table_path),
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 3014 terminal JSON artifact."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    path = config.artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _blocked_artifact(config: ExperimentConfig, started: float) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "repair_failure_taxonomy_ready": False,
        "taxonomy_table_path": "",
        "n_cached_candidates_audited": 0,
        "syntax_failure_count": 0,
        "schema_failure_count": 0,
        "false_accept_count": 0,
        "tautology_failure_count": 0,
        "intent_drift_count": 0,
        "recommended_acceptance_rules": _recommended_acceptance_rules(),
        "halluguard_ntk_claim_made": False,
        "honest_verdict": "blocked: exp3003 cached candidates unavailable",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_llm_inference_run": False,
        "failure_root_causes": list(FAILURE_ROOT_CAUSES),
        "failure_root_cause_counts": {},
        "oracle_failure_count": 0,
        "candidate_patch_paths": [],
        "verifier_log_paths": [],
        "source_artifacts": _source_artifacts(config, {}, {}),
        "taxonomy_table_sha256": None,
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _audit_candidate(
    config: ExperimentConfig,
    candidate: Mapping[str, Any],
    hard_items: Mapping[str, Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    index: int,
) -> JsonDict:
    item_id = str(candidate.get("item_id") or "")
    item = hard_items.get(item_id, {})
    patch_path = _resolve_repo_path(config.repo_root, candidate.get("candidate_patch_path"))
    transcript = _read_json_if_present(
        _resolve_repo_path(config.repo_root, candidate.get("transcript_path"))
    )
    source_transcript = _source_transcript(config, candidate, transcript)
    verifier_log = _read_json_if_present(
        _resolve_repo_path(config.repo_root, candidate.get("verifier_log_path"))
    )
    patch_code = patch_path.read_text(encoding="utf-8")
    raw_response = str(
        source_transcript.get("raw_response") or transcript.get("raw_response") or ""
    )
    parsed = parse_repair_output(raw_response) if raw_response else None
    schema_valid = (
        bool(parsed.schema_valid)
        if parsed is not None
        else bool(candidate.get("schema_valid", True))
    )
    syntax_success, syntax_errors = syntax_diagnostics(patch_code)
    original, variant_outcomes = _replay_validators(item, variants, patch_code)
    variants_all_pass = bool(variant_outcomes) and all(row.passed for row in variant_outcomes)
    false_accept = bool(original.passed and not variants_all_pass)
    entry_point = str(item.get("entry_point") or candidate.get("entry_point") or "")
    entry_present = (
        _entry_point_present(patch_code, entry_point) if syntax_success and entry_point else False
    )
    draft_intent = str(
        (parsed.draft_intent if parsed is not None else "") or candidate.get("draft_intent") or ""
    )
    intent_drift = bool(
        syntax_success
        and schema_valid
        and not false_accept
        and (
            not original.passed
            or not entry_present
            or _token_overlap(draft_intent, str(item.get("expected_behavior") or "")) < 0.15
        )
    )
    failure_modes = _candidate_failure_modes(
        schema_valid=schema_valid,
        syntax_success=syntax_success,
        false_accept=false_accept,
        intent_drift=intent_drift,
    )
    primary = _primary_candidate_root_cause(
        schema_valid=schema_valid,
        syntax_success=syntax_success,
        false_accept=false_accept,
        intent_drift=intent_drift,
        prompt_pressure=_looks_like_prompt_format_pressure(raw_response + "\n" + patch_code),
        entry_present=entry_present,
    )
    return {
        "row_type": "candidate",
        "source_index": int(candidate.get("source_index", index)),
        "item_id": item_id,
        "model_hf_id": str(candidate.get("model_hf_id") or ""),
        "candidate_patch_path": str(_relative_or_absolute(config.repo_root, patch_path)),
        "transcript_path": str(
            _relative_or_absolute(
                config.repo_root,
                _resolve_repo_path(config.repo_root, candidate.get("transcript_path")),
            )
        ),
        "verifier_log_path": str(
            _relative_or_absolute(
                config.repo_root,
                _resolve_repo_path(config.repo_root, candidate.get("verifier_log_path")),
            )
        ),
        "candidate_sha256": _sha256_text(patch_code),
        "schema_valid": schema_valid,
        "schema_errors": list(parsed.schema_errors) if parsed is not None else [],
        "syntax_success": syntax_success,
        "syntax_errors": syntax_errors,
        "entry_point_present": entry_present,
        "original_passed": bool(original.passed),
        "metamorphic_passed_all": variants_all_pass,
        "metamorphic_variant_count": len(variant_outcomes),
        "metamorphic_pass_count": sum(1 for row in variant_outcomes if row.passed),
        "false_accept": false_accept,
        "intent_drift": intent_drift,
        "draft_intent_overlap": _token_overlap(
            draft_intent, str(item.get("expected_behavior") or "")
        ),
        "failure_mode": failure_modes[0] if failure_modes else "passed",
        "failure_modes": failure_modes,
        "primary_root_cause": primary,
        "verifier_log_present": bool(verifier_log),
        "deterministic_replay_only": True,
    }


def _replay_validators(
    item: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    patch_code: str,
) -> tuple[hard.VerificationOutcome, list[hard.VerificationOutcome]]:
    candidate_item = {**dict(item), "repair_candidate": patch_code}
    original = hard.run_candidate_tests(candidate_item, "repair_candidate")
    outcomes: list[hard.VerificationOutcome] = []
    for variant in variants:
        if str(variant.get("source_item_id") or "") != str(item.get("item_id") or ""):
            continue
        adapted = metamorphic._adapt_candidate(
            patch_code,
            str(variant.get("source_entry_point") or item.get("entry_point") or ""),
            str(variant.get("entry_point") or ""),
        )
        outcomes.append(
            hard.run_candidate_tests(
                {**dict(variant), "repair_candidate": adapted}, "repair_candidate"
            )
        )
    return original, outcomes


def _candidate_failure_modes(
    *,
    schema_valid: bool,
    syntax_success: bool,
    false_accept: bool,
    intent_drift: bool,
) -> list[str]:
    modes: list[str] = []
    if not syntax_success:
        modes.append("syntax")
    if not schema_valid:
        modes.append("schema")
    if false_accept:
        modes.append("false_accept")
    if intent_drift:
        modes.append("intent_drift")
    return modes


def _primary_candidate_root_cause(
    *,
    schema_valid: bool,
    syntax_success: bool,
    false_accept: bool,
    intent_drift: bool,
    prompt_pressure: bool,
    entry_present: bool,
) -> str:
    if prompt_pressure and (not schema_valid or not syntax_success):
        return "prompt-format pressure"
    if false_accept:
        return "false accept"
    if not schema_valid:
        return "parser/schema mismatch"
    if not syntax_success or not entry_present:
        return "invalid patch shape"
    if intent_drift:
        return "intent drift"
    return "passed"


def _validator_taxonomy_rows(
    exp3002: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, rejected in enumerate(exp3002.get("rejected_variants") or []):
        reason = str(rejected.get("reason") or "")
        if reason == "tautological_oracle_rejected":
            rows.append(_validator_row(index, rejected, "tautology", "tautology"))
        elif reason == "reference_failed_semantics_changed":
            rows.append(_validator_row(index, rejected, "oracle_ambiguity", "oracle ambiguity"))
    reference_failures = [
        variant
        for variant in variants
        if variant.get("reference_verification", {}).get("passed") is False
    ]
    for offset, variant in enumerate(reference_failures, start=len(rows)):
        rows.append(_validator_row(offset, variant, "oracle_ambiguity", "oracle ambiguity"))
    return rows


def _validator_row(
    index: int, payload: Mapping[str, Any], failure_mode: str, root_cause: str
) -> JsonDict:
    return {
        "row_type": "validator",
        "source_index": index,
        "item_id": str(payload.get("item_id") or payload.get("source_item_id") or ""),
        "failure_mode": failure_mode,
        "failure_modes": [failure_mode],
        "primary_root_cause": root_cause,
        "reason": str(payload.get("reason") or ""),
        "relation_type": str(payload.get("relation_type") or ""),
        "deterministic_replay_only": True,
    }


def _failure_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    candidate_rows = [row for row in rows if row.get("row_type") == "candidate"]
    validator_rows = [row for row in rows if row.get("row_type") == "validator"]
    return {
        "syntax": sum(1 for row in candidate_rows if "syntax" in row.get("failure_modes", [])),
        "schema": sum(1 for row in candidate_rows if "schema" in row.get("failure_modes", [])),
        "false_accept": sum(
            1 for row in candidate_rows if "false_accept" in row.get("failure_modes", [])
        ),
        "intent_drift": sum(
            1 for row in candidate_rows if "intent_drift" in row.get("failure_modes", [])
        ),
        "tautology": sum(1 for row in validator_rows if row.get("failure_mode") == "tautology"),
        "oracle_ambiguity": sum(
            1 for row in validator_rows if row.get("failure_mode") == "oracle_ambiguity"
        ),
    }


def _root_cause_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    out = {cause: 0 for cause in FAILURE_ROOT_CAUSES}
    for row in rows:
        cause = str(row.get("primary_root_cause") or "")
        if cause in out:
            out[cause] += 1
    return out


def _recommended_acceptance_rules() -> list[str]:
    return [
        "Reject candidates unless schema/parser replay extracts a non-empty JSON draft_intent and final_patch.",
        "Reject candidates unless final_patch parses as Python and defines the expected public entry point.",
        "Reject candidates that pass original tests but fail any accepted Exp 3002 metamorphic variant.",
        "Reject candidates whose draft_intent has weak lexical overlap with the hard-set expected behavior.",
        "Reject promotion when Exp 3002 tautology probes are absent or accepted as passing validators.",
        "Route oracle-ambiguity rows to fixture repair before another live repair rerun.",
    ]


def _load_hard_items(config: ExperimentConfig) -> dict[str, Mapping[str, Any]]:
    path = config.resolved_hard_manifest_path()
    items = hard.load_manifest(path) if path.is_file() else hard.default_items()
    return {str(item.get("item_id") or ""): item for item in items}


def _load_metamorphic_variants(
    config: ExperimentConfig, exp3002: Mapping[str, Any]
) -> list[JsonDict]:
    path = _metamorphic_manifest_path(config, exp3002)
    return _read_jsonl(path) if path.is_file() else []


def _metamorphic_manifest_path(config: ExperimentConfig, exp3002: Mapping[str, Any]) -> Path:
    if config.metamorphic_manifest_path is not None:
        return config.metamorphic_manifest_path
    rel = exp3002.get("metamorphic_manifest_path") or metamorphic.METAMORPHIC_MANIFEST_REL_PATH
    return _resolve_repo_path(config.repo_root, rel)


def _source_transcript(
    config: ExperimentConfig,
    candidate: Mapping[str, Any],
    transcript: Mapping[str, Any],
) -> JsonDict:
    path_value = transcript.get("source_transcript_path") or candidate.get("live_transcript_path")
    return _read_json_if_present(_resolve_repo_path(config.repo_root, path_value))


def _source_artifacts(
    config: ExperimentConfig,
    exp3002: Mapping[str, Any],
    exp3003: Mapping[str, Any],
) -> list[JsonDict]:
    paths = [
        config.resolved_exp3002_artifact_path(),
        _metamorphic_manifest_path(config, exp3002),
        config.resolved_exp3003_artifact_path(),
        config.resolved_hard_manifest_path(),
    ]
    return [
        {
            "path": str(_relative_or_absolute(config.repo_root, path)),
            "present": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
            "artifact_loaded": bool(exp3002)
            if path.name == EXP3002_FILENAME
            else bool(exp3003)
            if path.name == EXP3003_FILENAME
            else None,
        }
        for path in paths
    ]


def _looks_like_prompt_format_pressure(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _PROMPT_FORMAT_MARKERS)


def _entry_point_present(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in _TOKEN_RE.findall(left)}
    right_tokens = {token.lower() for token in _TOKEN_RE.findall(right)}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def _read_json_if_present(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _resolve_repo_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _relative_or_absolute(root: Path, path: Path) -> Path:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return path.resolve(strict=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)


__all__ = [
    "ARTIFACT_FILENAME",
    "EXP3002_FILENAME",
    "EXP3003_FILENAME",
    "ExperimentConfig",
    "build_artifact",
    "write_artifact",
]
