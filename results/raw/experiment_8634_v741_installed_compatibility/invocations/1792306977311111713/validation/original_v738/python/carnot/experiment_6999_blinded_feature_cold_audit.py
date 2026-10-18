"""Audit the blinded learner table in a restricted fresh process.

The learner loader has no authority-sidecar parameter. The audit changes the
ambient mutation file five ways and proves that the loader output never moves.
Exact labels remain outside the tensors and only evaluate shortcut risk.

Spec refs: REQ-VERIFY-6999 and SCENARIO-VERIFY-6999-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import mmap
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_6999_blinded_feature_cold_audit"
SCHEMA = "carnot.experiment_6999.blinded_feature_cold_audit.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 69_992_026
INFERENCE_SUBSTRATE = "fresh_process_blinded_feature_isolation_audit_no_llm"
EXPECTED_CANDIDATE_COUNT = 138
EXPECTED_SOURCE_ROW_COUNT = 414
EXPECTED_COMMITMENT_ROW_COUNT = 216
SHORTCUT_AUROC_LIMIT = 0.80
SHORTCUT_FOLDS = 5
BOOTSTRAP_SAMPLES = 2_000

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path("python/carnot/experiment_6999_blinded_feature_cold_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6999_blinded_feature_cold_audit.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_6999_blinded_feature_cold_audit.json")

SOURCE_PATHS = {
    "exp6986": Path("results/experiment_6986_three_family_contrast_features.json"),
    "exp6987": Path("results/experiment_6987_contrast_feature_audit.json"),
    "exp6997": Path("results/experiment_6997_authority_sidecar_rebuild.json"),
    "exp6998": Path("results/experiment_6998_three_family_commitment_controls.json"),
    "learner_view": Path(
        "results/raw/experiment_6997_authority_sidecar_rebuild/learner_view.jsonl"
    ),
    "learner_manifest": Path(
        "results/raw/experiment_6997_authority_sidecar_rebuild/learner_view.jsonl.manifest.json"
    ),
    "label_sidecar": Path(
        "results/raw/experiment_6997_authority_sidecar_rebuild/label_split_sidecar.jsonl"
    ),
    "label_manifest": Path(
        "results/raw/experiment_6997_authority_sidecar_rebuild/label_split_sidecar.jsonl.manifest.json"
    ),
    "mutation_sidecar": Path(
        "results/raw/experiment_6997_authority_sidecar_rebuild/mutation_authority_sidecar.jsonl"
    ),
    "mutation_manifest": Path(
        "results/raw/experiment_6997_authority_sidecar_rebuild/mutation_authority_sidecar.jsonl.manifest.json"
    ),
}
EXPECTED_SOURCE_HASHES = {
    "exp6986": "sha256:c19f5180c5e70677baec92db5ea295599197919eb03c5119d9a22849320cff75",
    "exp6987": "sha256:271af8ade020727ed93cf07dbe1aa285e06467d46524ad9f067a19a5a7350bcf",
    "exp6997": "sha256:661a486b9400b879b117adeb6e2fea270b25c19e4037e4eec65fd19af70cc1be",
    "exp6998": "sha256:2e0c793c5d68b361d98aa6f8beaefee0b3e81953f3b75999d38ce06801b731c5",
    "learner_view": "sha256:2b4a4fb51c0dc09372d9aa4205d51906fc653f8138c345a8c9df93bfefcae17c",
    "learner_manifest": "sha256:90e0b4407379df6e4447b2669c4923f99863942236fa31a0ac633643baeb366d",
    "label_sidecar": "sha256:68920d1c05f3d068b1aa9e0517b5a70b57b854f79d6a5af6718ac60a15890f6c",
    "label_manifest": "sha256:b210a4d43e1c3aeb42380515d785b936afbac8afa784585be3d4c4cf6527921b",
    "mutation_sidecar": "sha256:67c39c5ca3661f0f333638294aed20b106a2c2fd6d4279a3ecc3b6a4c3e5ed90",
    "mutation_manifest": "sha256:db1785cb1419f82ad74081e79bd7c57ec82f8f2e3e4fcf9d6c881033010d4889",
}

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
FAMILY_POSITIONS = {model_id: index for index, model_id in enumerate(REQUIRED_MODEL_IDS)}
SEQUENCE_FEATURES = (
    "candidate_token_count",
    "full_token_count",
    "sequence_nll",
    "length_normalized_nll",
    "mean_token_entropy",
    "mean_top_probability_margin",
    "mean_local_surprisal_change",
    "max_abs_local_surprisal_change",
)
PARSER_FEATURES = (
    "json_parseable",
    "byte_count",
    "line_count",
    "brace_count",
    "bracket_count",
    "object_count",
    "array_count",
    "key_count",
    "array_item_count",
    "null_count",
    "boolean_count",
    "number_count",
    "string_count",
)
SOURCE_FEATURES = SEQUENCE_FEATURES + PARSER_FEATURES
FEATURE_ALLOWLIST = tuple(
    f"family_{position}__{field}"
    for position in range(len(REQUIRED_MODEL_IDS))
    for field in SOURCE_FEATURES
)
LENGTH_FIELDS = tuple(
    field
    for field in FEATURE_ALLOWLIST
    if field.endswith(
        ("__byte_count", "__line_count", "__candidate_token_count", "__full_token_count")
    )
)
SERIALIZATION_FIELDS = tuple(
    field
    for field in FEATURE_ALLOWLIST
    if field.rsplit("__", 1)[-1] in set(PARSER_FEATURES) - {"byte_count", "line_count"}
)
SOURCE_METADATA_FIELDS = ("source_block", "split", "formulation_family")
MUTATION_METADATA_FIELDS = ("fault_family", "schedule_id", "event_type")
COMMITMENT_FEATURE_FIELDS = (
    "first_commitment_latency",
    "commitment_range",
    "mean_uncommitted_mass",
    "mean_uncertainty",
    "choice_flip_count",
    "condition",
    "model_id",
)
COMMITMENT_DENIED_FIELDS = (
    "first_commitment_latency",
    "commitment_range",
    "mean_uncommitted_mass",
    "mean_uncertainty",
    "choice_flip_count",
    "condition",
    "prompt_condition_id",
    "terminal_choice",
    "choice_matches_exact_label",
    "own_choice_probability",
    "prefix_fraction",
    "prefix_ordinal",
)
SHORTCUT_PROBE_SPECS = {
    "length_only": {"fields": LENGTH_FIELDS, "prohibited": False},
    "serialization_only": {"fields": SERIALIZATION_FIELDS, "prohibited": False},
    "source_metadata_only": {"fields": SOURCE_METADATA_FIELDS, "prohibited": True},
    "mutation_metadata_only": {"fields": MUTATION_METADATA_FIELDS, "prohibited": True},
    "model_identity_only": {"fields": ("model_id",), "prohibited": True},
    "commitment_controls_only": {"fields": COMMITMENT_FEATURE_FIELDS, "prohibited": True},
    "allowed_learner_table": {"fields": FEATURE_ALLOWLIST, "prohibited": False},
}
PRIMARY_SPLITS = ("train", "calibration", "held_out")

_DENIED_NORMALIZED_TOKENS = (
    "label",
    "groundtruth",
    "expectedanswer",
    "split",
    "source",
    "mutation",
    "provenance",
    "authority",
    "witness",
    "fault",
    "group",
    "pairid",
    "candidateorder",
    "ordinal",
    "model",
    "repository",
    "repoidentity",
    "promptcondition",
    "conditionid",
    "commitment",
    "choice",
    "uncertainty",
    "uncommitted",
    "prefixfraction",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_candidate_rows",
    "raw_rebuild_rows",
    "candidate_key_rows",
    "family_completeness_rows",
    "join_replay_rows",
    "tensor_hash_rows",
    "label_balance_rows",
    "source_overlap_rows",
    "split_isolation_rows",
    "label_denial_replay_rows",
    "feature_schema_rows",
    "prohibited_field_rows",
    "sidecar_condition_rows",
    "sidecar_open_attempt_rows",
    "sidecar_permutation_rows",
    "sidecar_replacement_rows",
    "sidecar_removal_rows",
    "reference_prediction_rows",
    "learner_invariance_rows",
    "shortcut_probe_rows",
    "shortcut_prediction_rows",
    "shortcut_interval_rows",
    "commitment_prohibition_rows",
    "source_disagreement_rows",
    "read_only_enforcement_receipt",
    "direct_leakage_count",
    "feature_isolation_audit_complete_score",
    "blinded_feature_bank_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "A reason for every field makes the release contract reviewable.",
    "preconditions_checked": "Exact gates stop changed or incomplete evidence before analysis.",
    "inference_substrate": "The substrate distinguishes this audit from LLM or learner training.",
    "duration_s": "Measured wall time shows that the audit process executed.",
    "source_artifact_hashes": "Hashes bind every conclusion to immutable source bytes.",
    "rows": "Candidate rows keep the release denominator visible.",
    "per_candidate_rows": "One row per learner unit exposes every terminal outcome.",
    "raw_rebuild_rows": "Raw family rows prove that all stored model views were reconsidered.",
    "candidate_key_rows": "Key receipts prove that semantic hashes alone define identity.",
    "family_completeness_rows": "Family receipts reveal missing or duplicate model views.",
    "join_replay_rows": "Join receipts detect changed learner rows and authority keys.",
    "tensor_hash_rows": "Tensor hashes make numeric reconstruction byte-checkable.",
    "label_balance_rows": "Label counts expose constant or imbalanced partitions.",
    "source_overlap_rows": "Pairwise overlap rows reveal source reuse across splits.",
    "split_isolation_rows": "Per-source routing rows make split leakage traceable.",
    "label_denial_replay_rows": "Denial receipts prove that learner access stays narrow.",
    "feature_schema_rows": "A frozen schema separates numeric evidence from provenance.",
    "prohibited_field_rows": "Exact rejected paths make any direct leak actionable.",
    "sidecar_condition_rows": "Five conditions test whether ambient authority data changes learning.",
    "sidecar_open_attempt_rows": "Open receipts prove that the learner never reads a sidecar.",
    "sidecar_permutation_rows": "Permutation receipts prove that authority order has no effect.",
    "sidecar_replacement_rows": "Replacement receipts test cross-pair provenance dependence.",
    "sidecar_removal_rows": "Removal receipts test that no authority file is required.",
    "reference_prediction_rows": "A fixed projection detects tensor changes without fitting a learner.",
    "learner_invariance_rows": "Byte comparisons gate every sidecar condition.",
    "shortcut_probe_rows": "Fold rows prove grouped fitting and fold-local preprocessing.",
    "shortcut_prediction_rows": "Candidate predictions make every AUROC independently recomputable.",
    "shortcut_interval_rows": "Pair bootstrap intervals enforce the preregistered shortcut limit.",
    "commitment_prohibition_rows": "Policy rows keep self-commitment evidence audit-only.",
    "source_disagreement_rows": "Contradictions remain visible instead of being repaired.",
    "read_only_enforcement_receipt": "Sandbox evidence proves the child could not mutate sources.",
    "direct_leakage_count": "A bare count blocks any learner-visible label or provenance path.",
    "feature_isolation_audit_complete_score": "Completion requires every expected audit receipt.",
    "blinded_feature_bank_ready_score": "Readiness requires isolation and weak prohibited shortcuts.",
    "random_seed": "A fixed seed makes folds, permutations, and bootstrap draws repeatable.",
    "reproducibility_checksum": "A canonical digest detects later artifact drift.",
    "gate_check_summary": "Expected and observed values identify the first blocked condition.",
    "verifier_is_oracle": "True states that exact labels define this release audit.",
    "verdict_class": "A closed class separates readiness from terminal disqualification.",
    "honest_verdict": "A class-specific prefix gives automation one stable terminal meaning.",
}
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}


class IsolationError(ValueError):
    """The learner input crossed its narrow numeric-file boundary."""


class HashMismatchError(ValueError):
    """An immutable input differs from its pinned or manifest hash."""


class AuthorityJoinError(ValueError):
    """Labels or provenance do not join once to every learner key."""


@dataclass(frozen=True)
class LearnerBatch:
    """Materialized learner bytes with no label or provenance member."""

    keys: tuple[str, ...]
    feature_names: tuple[str, ...]
    matrix: np.ndarray
    tensor_hash: str
    prediction_hash: str
    file_open_receipts: tuple[JsonDict, ...]


@dataclass(frozen=True)
class AuthorityBatch:
    """Separate authority rows joined after learner materialization."""

    keys: tuple[str, ...]
    matrix: np.ndarray
    rows: tuple[JsonDict, ...]
    file_open_receipts: tuple[JsonDict, ...]


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for byte hashes and JSON Lines."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a project-style SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text without locale-dependent behavior."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash one JSON value after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Hash a file incrementally, or return ``None`` when it is absent."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def gate_check(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record one exact expected-observed condition."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every check and promote the first failure for automation."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": "all checks pass" if failed is None else failed.get("expected_value"),
        "observed_value": "all checks pass" if failed is None else failed.get("observed_value"),
        "passed": failed is None,
    }


def load_hashed_inputs(
    repo_root: Path,
    *,
    source_paths: Mapping[str, Path] = SOURCE_PATHS,
    expected_hashes: Mapping[str, str] = EXPECTED_SOURCE_HASHES,
) -> JsonDict:
    """Hash all inputs before parsing any source value."""

    checks = []
    hashes: dict[str, str | None] = {}
    for source_id, relative in source_paths.items():
        observed = sha256_path(repo_root / relative)
        hashes[source_id] = observed
        checks.append(
            gate_check(f"source_hash:{source_id}", expected_hashes.get(source_id), observed)
        )
    if any(row["passed"] is not True for row in checks):
        return {"passed": False, "values": {}, "checks": checks, "hashes": hashes}

    values: dict[str, Any] = {}
    try:
        for source_id, relative in source_paths.items():
            path = repo_root / relative
            text = path.read_text(encoding="utf-8")
            values[source_id] = (
                [json.loads(line) for line in text.splitlines() if line]
                if path.suffix == ".jsonl"
                else json.loads(text)
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        checks.append(
            gate_check("source_parse", "all inputs parse", f"{type(exc).__name__}: {exc}")
        )
        return {"passed": False, "values": {}, "checks": checks, "hashes": hashes}
    return {"passed": True, "values": values, "checks": checks, "hashes": hashes}


def _selected_json_fields(path: Path, fields: Sequence[str]) -> JsonDict:
    """Read selected top-level fields without retaining a 189 MiB token table."""

    selected: JsonDict = {}
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for field in fields:
            marker = f'\n  "{field}":'.encode()
            marker_at = data.find(marker)
            if marker_at < 0:
                raise HashMismatchError(f"source_field_missing:{path.name}:{field}")
            start = marker_at + len(marker)
            while data[start] in b" \t\r\n":
                start += 1
            first = data[start]
            if first in (ord("["), ord("{")):
                closing = ord("]") if first == ord("[") else ord("}")
                depth = 0
                quoted = False
                escaped = False
                end = start
                while end < len(data):
                    byte = data[end]
                    if quoted:
                        if escaped:
                            escaped = False
                        elif byte == ord("\\"):
                            escaped = True
                        elif byte == ord('"'):
                            quoted = False
                    elif byte == ord('"'):
                        quoted = True
                    elif byte == first:
                        depth += 1
                    elif byte == closing:
                        depth -= 1
                        if depth == 0:
                            end += 1
                            break
                    end += 1
            elif first == ord('"'):
                end = start + 1
                escaped = False
                while end < len(data):
                    byte = data[end]
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        end += 1
                        break
                    end += 1
            else:
                comma = data.find(b",", start)
                newline = data.find(b"\n", start)
                candidates = [position for position in (comma, newline) if position >= 0]
                end = min(candidates) if candidates else len(data)
            selected[field] = json.loads(data[start:end])
    return selected


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Parse one already-hashed JSON Lines input."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _numeric(value: Any, *, field: str) -> int | float:
    """Accept finite numeric evidence and normalize booleans to integers."""

    if isinstance(value, bool):
        return int(value)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise IsolationError(f"nonnumeric_feature:{field}")
    return value


def candidate_key(prompt_hash: str, candidate_hash: str) -> str:
    """Derive learner identity only from semantic content hashes."""

    if not prompt_hash.startswith("sha256:") or not candidate_hash.startswith("sha256:"):
        raise IsolationError("semantic_hash_missing")
    return sha256_json({"candidate_hash": candidate_hash, "prompt_hash": prompt_hash})


def rebuild_wide_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT
) -> JsonDict:
    """Rebuild wide learner rows while retaining every structural disagreement."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("candidate_id", ""))].append(row)
    disagreements: list[JsonDict] = []
    if len(grouped) != expected_candidate_count:
        disagreements.append(
            {
                "kind": "candidate_count_mismatch",
                "expected_value": expected_candidate_count,
                "observed_value": len(grouped),
                "terminal": True,
            }
        )

    learner_rows: list[JsonDict] = []
    raw_rebuild_rows: list[JsonDict] = []
    key_rows: list[JsonDict] = []
    family_rows: list[JsonDict] = []
    seen_keys: set[str] = set()
    for candidate_id, family_sources in sorted(grouped.items()):
        semantics = {
            (str(row.get("prompt_hash", "")), str(row.get("candidate_hash", "")))
            for row in family_sources
        }
        if len(semantics) != 1:
            disagreements.append(
                {
                    "kind": "candidate_semantic_hash_conflict",
                    "candidate_id": candidate_id,
                    "observed_value": sorted([list(value) for value in semantics]),
                    "terminal": True,
                }
            )
            family_rows.append(
                {
                    "candidate_id": candidate_id,
                    "expected_models": list(REQUIRED_MODEL_IDS),
                    "observed_models": sorted(str(row.get("model_id")) for row in family_sources),
                    "passed": False,
                    "terminal": True,
                }
            )
            continue
        prompt_hash, candidate_hash_value = next(iter(semantics))
        try:
            key = candidate_key(prompt_hash, candidate_hash_value)
        except IsolationError as exc:
            disagreements.append({"kind": str(exc), "candidate_id": candidate_id, "terminal": True})
            continue
        if key in seen_keys:
            disagreements.append(
                {"kind": "duplicate_candidate_key", "candidate_key": key, "terminal": True}
            )
            continue
        seen_keys.add(key)
        models = [str(row.get("model_id", "")) for row in family_sources]
        duplicate_models = sorted(model for model, count in Counter(models).items() if count > 1)
        complete = not duplicate_models and set(models) == set(REQUIRED_MODEL_IDS)
        if duplicate_models:
            disagreements.append(
                {
                    "kind": "duplicate_family_row",
                    "candidate_id": candidate_id,
                    "models": duplicate_models,
                    "terminal": True,
                }
            )
        if set(models) != set(REQUIRED_MODEL_IDS):
            disagreements.append(
                {
                    "kind": "missing_model_families",
                    "candidate_id": candidate_id,
                    "expected_models": list(REQUIRED_MODEL_IDS),
                    "observed_models": sorted(set(models)),
                    "terminal": True,
                }
            )
        family_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_key": key,
                "expected_models": list(REQUIRED_MODEL_IDS),
                "observed_models": sorted(models),
                "family_count": len(models),
                "passed": complete,
                "terminal": True,
            }
        )
        key_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_key": key,
                "prompt_hash": prompt_hash,
                "candidate_hash": candidate_hash_value,
                "key_inputs": ["prompt_hash", "candidate_hash"],
                "passed": key not in seen_keys - {key},
                "terminal": True,
            }
        )
        if not complete:
            continue
        indexed = {str(row["model_id"]): row for row in family_sources}
        vector: list[int | float] = []
        failed_numeric = False
        for model_id in REQUIRED_MODEL_IDS:
            source = indexed[model_id]
            values: list[int | float] = []
            try:
                values = [_numeric(source.get(field), field=field) for field in SOURCE_FEATURES]
            except IsolationError as exc:
                disagreements.append(
                    {
                        "kind": str(exc),
                        "candidate_id": candidate_id,
                        "model_id": model_id,
                        "terminal": True,
                    }
                )
                failed_numeric = True
            position = FAMILY_POSITIONS[model_id]
            vector.extend(values)
            raw_rebuild_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_key": key,
                    "prompt_hash": prompt_hash,
                    "candidate_hash": candidate_hash_value,
                    "model_id": model_id,
                    "family_position": position,
                    "features": values,
                    "source_row_hash": sha256_json(dict(source)),
                    "passed": bool(values) and not failed_numeric,
                    "terminal": True,
                }
            )
        if not failed_numeric:
            learner_rows.append({"candidate_key": key, "features": vector})

    learner_rows.sort(key=lambda row: row["candidate_key"])
    raw_rebuild_rows.sort(key=lambda row: (row["candidate_key"], row["family_position"]))
    key_rows.sort(key=lambda row: row.get("candidate_key", ""))
    family_rows.sort(key=lambda row: row.get("candidate_key", row["candidate_id"]))
    return {
        "learner_rows": learner_rows,
        "raw_rebuild_rows": raw_rebuild_rows,
        "candidate_key_rows": key_rows,
        "family_completeness_rows": family_rows,
        "source_disagreement_rows": disagreements,
    }


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _denied_field(field: str) -> bool:
    normalized = _normalized_name(field)
    return any(token in normalized for token in _DENIED_NORMALIZED_TOKENS)


def _walk_fields(value: Any, path: str = "$") -> list[tuple[str, str, Any]]:
    output = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            output.append((child_path, str(key), child))
            output.extend(_walk_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.extend(_walk_fields(child, f"{path}[{index}]"))
    return output


def audit_learner_schema(rows: Sequence[Mapping[str, Any]], allowlist: Sequence[str]) -> JsonDict:
    """Reject direct, nested, alias, categorical, and malformed learner fields."""

    prohibited: list[JsonDict] = []
    seen: set[str] = set()
    if tuple(allowlist) != FEATURE_ALLOWLIST:
        prohibited.append(
            {
                "path": "$.allowlist",
                "field": "allowlist",
                "reason": "allowlist_mismatch",
                "terminal": True,
            }
        )
    for field in allowlist:
        if _denied_field(str(field)):
            prohibited.append(
                {
                    "path": f"$.allowlist.{field}",
                    "field": str(field),
                    "reason": "prohibited_allowlist_alias",
                    "terminal": True,
                }
            )
    for index, row in enumerate(rows):
        key = str(row.get("candidate_key", ""))
        if not key or key in seen:
            prohibited.append(
                {
                    "path": f"$[{index}].candidate_key",
                    "field": "candidate_key",
                    "reason": "duplicate_or_empty_key",
                    "terminal": True,
                }
            )
        seen.add(key)
        for path, field, value in _walk_fields(row, f"$[{index}]"):
            if field not in {"candidate_key", "features"}:
                prohibited.append(
                    {
                        "path": path,
                        "field": field,
                        "reason": "prohibited_alias"
                        if _denied_field(field)
                        else "field_not_allowlisted",
                        "terminal": True,
                    }
                )
            elif _denied_field(field) and field != "candidate_key":
                prohibited.append(
                    {"path": path, "field": field, "reason": "prohibited_alias", "terminal": True}
                )
            if isinstance(value, Mapping):
                prohibited.append(
                    {"path": path, "field": field, "reason": "nested_metadata", "terminal": True}
                )
        features = row.get("features")
        if not isinstance(features, list) or len(features) != len(allowlist):
            prohibited.append(
                {
                    "path": f"$[{index}].features",
                    "field": "features",
                    "reason": "tensor_shape_mismatch",
                    "terminal": True,
                }
            )
        elif any(
            type(value) not in (int, float) or not math.isfinite(float(value)) for value in features
        ):
            prohibited.append(
                {
                    "path": f"$[{index}].features",
                    "field": "features",
                    "reason": "nonnumeric_tensor",
                    "terminal": True,
                }
            )
    unique = {(row["path"], row["reason"]): row for row in prohibited}
    rows_out = list(unique.values())
    return {
        "prohibited_field_rows": rows_out,
        "direct_leakage_count": len(rows_out),
        "passed": not rows_out,
    }


def _manifest_path(path: Path) -> Path:
    return Path(str(path) + ".manifest.json")


def _read_verified_jsonl(
    path: Path, *, kind: str
) -> tuple[list[JsonDict], tuple[JsonDict, ...], JsonDict]:
    receipts: list[JsonDict] = []
    manifest_path = _manifest_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipts.append(
        {"path": str(manifest_path), "operation": "open_read", "role": "manifest", "passed": True}
    )
    payload = path.read_bytes()
    receipts.append({"path": str(path), "operation": "open_read", "role": kind, "passed": True})
    if manifest.get("kind") != kind:
        raise HashMismatchError(f"manifest_kind_mismatch:{kind}")
    if manifest.get("path") != str(path):
        raise HashMismatchError(f"manifest_path_mismatch:{kind}")
    if manifest.get("file_hash") != sha256_bytes(payload):
        raise HashMismatchError(f"file_hash_mismatch:{kind}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if manifest.get("row_count") != len(rows) or manifest.get("byte_count") != len(payload):
        raise HashMismatchError(f"manifest_size_mismatch:{kind}")
    key_hashes = [sha256_text(str(row.get("candidate_key", ""))) for row in rows]
    if manifest.get("ordered_key_hashes") != key_hashes:
        raise HashMismatchError(f"manifest_key_mismatch:{kind}")
    return rows, tuple(receipts), manifest


def _tensor_hash(keys: Sequence[str], names: Sequence[str], matrix: np.ndarray) -> str:
    header = canonical_json({"keys": list(keys), "feature_names": list(names)}).encode("utf-8")
    return sha256_bytes(header + b"\0" + np.asarray(matrix, dtype="<f8").tobytes(order="C"))


def _prediction_hash(matrix: np.ndarray) -> str:
    values = np.asarray(matrix, dtype="<f8")
    weights = np.arange(1, values.shape[1] + 1, dtype="<f8")
    predictions = values @ weights
    return sha256_bytes(predictions.astype("<f8", copy=False).tobytes(order="C"))


def _materialize(
    rows: Sequence[Mapping[str, Any]],
    allowlist: Sequence[str],
    receipts: Sequence[Mapping[str, Any]] = (),
) -> LearnerBatch:
    schema = audit_learner_schema(rows, allowlist)
    if schema["passed"] is not True:
        raise IsolationError("learner_schema_rejected")
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["candidate_key"])
    keys = tuple(str(row["candidate_key"]) for row in ordered)
    matrix = np.asarray([row["features"] for row in ordered], dtype="<f8")
    names = tuple(allowlist)
    return LearnerBatch(
        keys=keys,
        feature_names=names,
        matrix=matrix,
        tensor_hash=_tensor_hash(keys, names, matrix),
        prediction_hash=_prediction_hash(matrix),
        file_open_receipts=tuple(deepcopy(dict(row)) for row in receipts),
    )


def load_learner_view(learner_path: str | Path, allowlist: Sequence[str]) -> LearnerBatch:
    """Open only the learner file and its inferred manifest."""

    if not isinstance(learner_path, (str, Path)):
        raise IsolationError("learner_path_required")
    path = Path(learner_path)
    if path.name != "learner_view.jsonl":
        raise IsolationError("learner_path_required")
    if not isinstance(allowlist, (list, tuple)) or tuple(allowlist) != FEATURE_ALLOWLIST:
        raise IsolationError("frozen_allowlist_required")
    rows, receipts, manifest = _read_verified_jsonl(path, kind="learner_view")
    if tuple(manifest.get("feature_allowlist", [])) != FEATURE_ALLOWLIST:
        raise HashMismatchError("manifest_allowlist_mismatch:learner_view")
    return _materialize(rows, allowlist, receipts)


def _index_authority(rows: Sequence[Mapping[str, Any]], *, kind: str) -> dict[str, JsonDict]:
    required = (
        {"candidate_key", "exact_label", "split"}
        if kind == "label_split"
        else {
            "candidate_key",
            "source_record_id",
            "source",
            "mutation",
            "witnesses",
            "authority_records",
        }
    )
    output = {}
    for row in rows:
        key = str(row.get("candidate_key", ""))
        if not key or key in output:
            raise AuthorityJoinError(f"duplicate_sidecar_key:{kind}:{key}")
        if set(row) != required:
            raise AuthorityJoinError(f"invalid_sidecar_schema:{kind}:{key}")
        output[key] = deepcopy(dict(row))
    return output


def join_authority(
    learner: LearnerBatch, label_path: str | Path, mutation_path: str | Path
) -> AuthorityBatch:
    """Open authority files only after the numeric matrix exists."""

    label = Path(label_path)
    mutation = Path(mutation_path)
    if (
        label.name != "label_split_sidecar.jsonl"
        or mutation.name != "mutation_authority_sidecar.jsonl"
    ):
        raise AuthorityJoinError("authority_sidecar_paths_required")
    label_rows, label_receipts, _ = _read_verified_jsonl(label, kind="label_split")
    mutation_rows, mutation_receipts, _ = _read_verified_jsonl(mutation, kind="mutation_authority")
    label_index = _index_authority(label_rows, kind="label_split")
    mutation_index = _index_authority(mutation_rows, kind="mutation_authority")
    if set(learner.keys) != set(label_index) or set(learner.keys) != set(mutation_index):
        raise AuthorityJoinError("sidecar_key_mismatch:join")
    rows = []
    for key in learner.keys:
        label_row = label_index[key]
        mutation_row = mutation_index[key]
        rows.append(
            {
                "candidate_key": key,
                "exact_label": label_row["exact_label"],
                "split": label_row["split"],
                "source_record_id": mutation_row["source_record_id"],
                "source": mutation_row["source"],
                "mutation": mutation_row["mutation"],
                "witnesses": mutation_row["witnesses"],
                "authority_records": mutation_row["authority_records"],
            }
        )
    return AuthorityBatch(
        keys=learner.keys,
        matrix=learner.matrix,
        rows=tuple(rows),
        file_open_receipts=(*label_receipts, *mutation_receipts),
    )


def authority_batch_from_rows(
    learner: LearnerBatch, rows: Sequence[Mapping[str, Any]]
) -> AuthorityBatch:
    """Build an in-memory authority batch for adversarial split tests."""

    return AuthorityBatch(
        keys=learner.keys,
        matrix=learner.matrix,
        rows=tuple(deepcopy(dict(row)) for row in rows),
        file_open_receipts=(),
    )


def audit_splits(authority: AuthorityBatch) -> JsonDict:
    """Recompute balance and ensure each source key uses one partition."""

    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    source_partitions: dict[str, set[str]] = defaultdict(set)
    partition_sources: dict[str, set[str]] = defaultdict(set)
    for row in authority.rows:
        split = str(row.get("split"))
        by_split[split].append(row)
        source = row.get("source", {})
        for prefix, field in (("group", "source_group_id"), ("pair", "source_pair_id")):
            value = source.get(field) if isinstance(source, Mapping) else None
            if value:
                key = f"{prefix}:{value}"
                source_partitions[key].add(split)
                partition_sources[split].add(key)
    balance_rows = []
    for split in sorted(set(by_split) | set(PRIMARY_SPLITS)):
        rows = by_split.get(split, [])
        positive = sum(row.get("exact_label") == "equivalent" for row in rows)
        negative = sum(row.get("exact_label") == "non_equivalent" for row in rows)
        required = split in PRIMARY_SPLITS
        balanced = positive == negative and positive > 0
        balance_rows.append(
            {
                "split": split,
                "candidate_count": len(rows),
                "positive_count": positive,
                "negative_count": negative,
                "exactly_balanced": balanced,
                "required_for_readiness": required,
                "passed": balanced if required else True,
                "terminal": True,
            }
        )
    overlap_rows = []
    splits = sorted(partition_sources)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            common = sorted(partition_sources[left] & partition_sources[right])
            overlap_rows.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "overlap_count": len(common),
                    "overlap_source_keys": common,
                    "passed": not common,
                    "terminal": True,
                }
            )
    isolation_rows = [
        {
            "source_key": key,
            "partitions": sorted(values),
            "partition_count": len(values),
            "passed": len(values) == 1,
            "terminal": True,
        }
        for key, values in sorted(source_partitions.items())
    ]
    return {
        "label_balance_rows": balance_rows,
        "source_overlap_rows": overlap_rows,
        "split_isolation_rows": isolation_rows,
        "all_required_balanced": all(
            row["passed"] for row in balance_rows if row["required_for_readiness"]
        )
        and {row["split"] for row in balance_rows if row["required_for_readiness"]}
        == set(PRIMARY_SPLITS),
        "all_sources_disjoint": all(row["passed"] for row in isolation_rows),
    }


def _write_condition_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")


def _cross_pair_replacement(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], int]:
    output = []
    replacements = 0
    for row in rows:
        source = row.get("source", {})
        pair = source.get("source_pair_id") if isinstance(source, Mapping) else None
        donor = next(
            (
                candidate
                for candidate in rows
                if isinstance(candidate.get("source"), Mapping)
                and candidate["source"].get("source_pair_id") != pair
            ),
            row,
        )
        changed = deepcopy(dict(donor))
        changed["candidate_key"] = row["candidate_key"]
        replacements += changed != dict(row)
        output.append(changed)
    return output, replacements


def audit_sidecar_conditions(
    learner_path: str | Path,
    allowlist: Sequence[str],
    mutation_path: str | Path,
    temporary_root: Path,
    *,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Materialize one learner while five ambient mutation files vary."""

    source_path = Path(mutation_path)
    mutation_rows = _read_jsonl(source_path)
    permutation = deepcopy(mutation_rows)
    random.Random(random_seed).shuffle(permutation)
    if len(permutation) > 1 and [row["candidate_key"] for row in permutation] == [
        row["candidate_key"] for row in mutation_rows
    ]:
        permutation.reverse()
    replacement, replacement_count = _cross_pair_replacement(mutation_rows)
    permutation_path = temporary_root / "permuted.jsonl"
    replacement_path = temporary_root / "replacement.jsonl"
    empty_path = temporary_root / "empty.jsonl"
    removed_path = temporary_root / "removed.jsonl"
    _write_condition_file(permutation_path, permutation)
    _write_condition_file(replacement_path, replacement)
    _write_condition_file(empty_path, [])
    if removed_path.exists():
        removed_path.unlink()
    conditions = (
        ("correct", source_path),
        ("permutation", permutation_path),
        ("cross_pair_replacement", replacement_path),
        ("empty", empty_path),
        ("removed", removed_path),
    )
    condition_rows = []
    open_rows = []
    invariance_rows = []
    prediction_rows = []
    base: LearnerBatch | None = None
    old = os.environ.get("CARNOT_MUTATION_SIDECAR")
    try:
        for name, ambient_path in conditions:
            if name == "removed":
                os.environ.pop("CARNOT_MUTATION_SIDECAR", None)
            else:
                os.environ["CARNOT_MUTATION_SIDECAR"] = str(ambient_path)
            batch = load_learner_view(learner_path, allowlist)
            if base is None:
                base = batch
            opened = {row["path"] for row in batch.file_open_receipts}
            attempted = str(ambient_path) in opened
            invariant = bool(
                batch.keys == base.keys
                and np.array_equal(batch.matrix, base.matrix)
                and batch.tensor_hash == base.tensor_hash
                and batch.prediction_hash == base.prediction_hash
            )
            condition_rows.append(
                {
                    "condition": name,
                    "ambient_sidecar_path": str(ambient_path) if name != "removed" else None,
                    "sidecar_exists": ambient_path.exists(),
                    "learner_row_count": len(batch.keys),
                    "tensor_hash": batch.tensor_hash,
                    "prediction_hash": batch.prediction_hash,
                    "passed": invariant and not attempted,
                    "terminal": True,
                }
            )
            open_rows.append(
                {
                    "condition": name,
                    "sidecar_path": str(ambient_path) if name != "removed" else None,
                    "learner_opened_paths": sorted(opened),
                    "sidecar_open_attempted": attempted,
                    "passed": not attempted,
                    "terminal": True,
                }
            )
            invariance_rows.append(
                {
                    "condition": name,
                    "keys_identical": batch.keys == base.keys,
                    "tensors_byte_identical": np.array_equal(batch.matrix, base.matrix),
                    "tensor_hash_identical": batch.tensor_hash == base.tensor_hash,
                    "prediction_hash_identical": batch.prediction_hash == base.prediction_hash,
                    "passed": invariant,
                    "terminal": True,
                }
            )
            prediction_rows.append(
                {
                    "condition": name,
                    "reference_prediction_hash": batch.prediction_hash,
                    "passed": batch.prediction_hash == base.prediction_hash,
                    "terminal": True,
                }
            )
    finally:
        if old is None:
            os.environ.pop("CARNOT_MUTATION_SIDECAR", None)
        else:
            os.environ["CARNOT_MUTATION_SIDECAR"] = old
    return {
        "sidecar_condition_rows": condition_rows,
        "sidecar_open_attempt_rows": open_rows,
        "sidecar_permutation_rows": [
            {
                "row_count": len(permutation),
                "row_order_changed": [row["candidate_key"] for row in permutation]
                != [row["candidate_key"] for row in mutation_rows],
                "passed": condition_rows[1]["passed"],
                "terminal": True,
            }
        ],
        "sidecar_replacement_rows": [
            {
                "row_count": len(replacement),
                "replacement_count": replacement_count,
                "cross_pair_replacement": True,
                "passed": replacement_count > 0 and condition_rows[2]["passed"],
                "terminal": True,
            }
        ],
        "sidecar_removal_rows": [
            {
                "sidecar_exists": removed_path.exists(),
                "passed": not removed_path.exists() and condition_rows[4]["passed"],
                "terminal": True,
            }
        ],
        "reference_prediction_rows": prediction_rows,
        "learner_invariance_rows": invariance_rows,
    }


def _binary_label(value: Any) -> int:
    if value in (1, "equivalent", True):
        return 1
    if value in (0, "non_equivalent", False):
        return 0
    raise ValueError(f"unknown_exact_label:{value}")


def _feature_dict(row: Mapping[str, Any], fields: Sequence[str]) -> JsonDict:
    output: JsonDict = {}
    for field in fields:
        value = row.get(field)
        if type(value) in (int, float) and math.isfinite(float(value)):
            output[field] = value
        elif isinstance(value, bool):
            output[field] = int(value)
        else:
            output[field] = f"{field}={value if value is not None else '<missing>'}"
    return output


def interval_passes(upper: float | None) -> bool:
    """Apply the strict shortcut threshold, including equality failure."""

    return upper is not None and math.isfinite(upper) and upper < SHORTCUT_AUROC_LIMIT


def _pair_bootstrap_auc(
    rows: Sequence[Mapping[str, Any]], *, random_seed: int, samples: int
) -> tuple[float | None, float | None]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_pair_id"])].append(row)
    pair_ids = sorted(grouped)
    rng = np.random.default_rng(random_seed)
    values = []
    for _ in range(samples):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        selected = [row for pair_id in sampled for row in grouped[str(pair_id)]]
        labels = [int(row["label"]) for row in selected]
        if len(set(labels)) < 2:
            continue
        auc = float(roc_auc_score(labels, [float(row["prediction"]) for row in selected]))
        values.append(max(auc, 1.0 - auc))
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def fit_grouped_probe(
    rows: Sequence[Mapping[str, Any]],
    *,
    probe_name: str,
    feature_fields: Sequence[str],
    random_seed: int = RANDOM_SEED,
    folds: int = SHORTCUT_FOLDS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    prohibited: bool,
) -> JsonDict:
    """Fit fold-local preprocessing and collapse predictions to candidates."""

    labels = np.asarray([_binary_label(row.get("label")) for row in rows], dtype=int)
    groups = np.asarray([str(row.get("source_pair_id")) for row in rows], dtype=object)
    if not rows or len(set(labels.tolist())) < 2 or len(set(groups.tolist())) < 2:
        raise ValueError(f"shortcut_probe_not_identifiable:{probe_name}")
    n_splits = min(folds, len(set(groups.tolist())))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scores = np.full(len(rows), np.nan, dtype=float)
    fold_rows = []
    dummy = np.zeros((len(rows), 1), dtype=float)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(dummy, labels, groups)):
        vectorizer = DictVectorizer(sparse=True)
        train_matrix = vectorizer.fit_transform(
            [_feature_dict(rows[index], feature_fields) for index in train_index]
        )
        test_matrix = vectorizer.transform(
            [_feature_dict(rows[index], feature_fields) for index in test_index]
        )
        scaler = StandardScaler(with_mean=False)
        train_scaled = scaler.fit_transform(train_matrix)
        test_scaled = scaler.transform(test_matrix)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=4_000,
            random_state=random_seed + fold_index,
            solver="liblinear",
        )
        model.fit(train_scaled, labels[train_index])
        scores[test_index] = model.predict_proba(test_scaled)[:, 1]
        train_groups = sorted(set(groups[train_index].tolist()))
        test_groups = sorted(set(groups[test_index].tolist()))
        names = vectorizer.get_feature_names_out().tolist()
        fold_rows.append(
            {
                "probe_name": probe_name,
                "fold_index": fold_index,
                "feature_fields": list(feature_fields),
                "train_row_count": len(train_index),
                "test_row_count": len(test_index),
                "train_pair_ids": train_groups,
                "test_pair_ids": test_groups,
                "group_overlap_count": len(set(train_groups) & set(test_groups)),
                "preprocessing_fit_on_train_only": True,
                "coefficients": [
                    {"feature": name, "coefficient": float(value)}
                    for name, value in zip(names, model.coef_[0], strict=True)
                ],
                "intercept": float(model.intercept_[0]),
                "passed": not (set(train_groups) & set(test_groups)),
                "terminal": True,
            }
        )
    if np.isnan(scores).any():
        raise ValueError(f"shortcut_probe_oof_gap:{probe_name}")

    candidate_scores: dict[str, list[float]] = defaultdict(list)
    candidate_labels: dict[str, set[int]] = defaultdict(set)
    candidate_groups: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        candidate = str(row.get("candidate_key", row.get("candidate_id", "")))
        candidate_scores[candidate].append(float(scores[index]))
        candidate_labels[candidate].add(int(labels[index]))
        candidate_groups[candidate].add(str(groups[index]))
    if any(len(value) != 1 for value in candidate_labels.values()) or any(
        len(value) != 1 for value in candidate_groups.values()
    ):
        raise ValueError(f"shortcut_probe_candidate_disagreement:{probe_name}")
    prediction_rows = [
        {
            "probe_name": probe_name,
            "candidate_key": candidate,
            "source_pair_id": next(iter(candidate_groups[candidate])),
            "label": next(iter(candidate_labels[candidate])),
            "prediction": float(np.mean(values)),
            "repeat_count": len(values),
            "terminal": True,
        }
        for candidate, values in sorted(candidate_scores.items())
    ]
    y = [int(row["label"]) for row in prediction_rows]
    auc = float(roc_auc_score(y, [float(row["prediction"]) for row in prediction_rows]))
    shortcut_auc = max(auc, 1.0 - auc)
    lower, upper = _pair_bootstrap_auc(
        prediction_rows, random_seed=random_seed, samples=bootstrap_samples
    )
    interval = {
        "probe_name": probe_name,
        "feature_fields": list(feature_fields),
        "prohibited": prohibited,
        "candidate_count": len(prediction_rows),
        "pair_count": len({row["source_pair_id"] for row in prediction_rows}),
        "bootstrap_unit": "source_pair_id",
        "bootstrap_samples": bootstrap_samples,
        "auroc": auc,
        "shortcut_auroc": shortcut_auc,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "threshold": SHORTCUT_AUROC_LIMIT,
        "gate_passed": (not prohibited) or interval_passes(upper),
        "terminal": True,
    }
    return {
        "probe_rows": fold_rows,
        "prediction_rows": prediction_rows,
        "interval_row": interval,
    }


def _late_label_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Any]:
    output = {}
    for row in rows:
        key = (str(row.get("candidate_id")), str(row.get("condition")), str(row.get("model_id")))
        if key in output and output[key] != row.get("exact_label"):
            raise AuthorityJoinError(f"commitment_label_conflict:{key}")
        output[key] = row.get("exact_label")
    return output


def build_commitment_probe_rows(
    artifact: Mapping[str, Any], *, expected_count: int = EXPECTED_COMMITMENT_ROW_COUNT
) -> JsonDict:
    """Admit commitment evidence only as a prohibited audit control."""

    units = list(artifact.get("per_candidate_condition_model_rows", []))
    labels = _late_label_index(artifact.get("late_label_join_rows", []))
    checks = [
        gate_check(
            "commitment_control_complete_score",
            1,
            artifact.get("commitment_control_complete_score"),
        ),
        gate_check("commitment_unit_count", expected_count, len(units)),
        gate_check(
            "commitment_observed_unit_count", expected_count, artifact.get("observed_unit_count")
        ),
        gate_check("commitment_audit_only", True, artifact.get("audit_only_control")),
        gate_check(
            "commitment_learner_feature_allowed", False, artifact.get("learner_feature_allowed")
        ),
        gate_check(
            "commitment_rows_terminal",
            True,
            bool(units) and all(row.get("terminal") is True for row in units),
        ),
    ]
    probe_rows = []
    for row in units:
        key = (str(row.get("candidate_id")), str(row.get("condition")), str(row.get("model_id")))
        if key not in labels:
            checks.append(gate_check(f"commitment_label_join:{key}", True, False))
            continue
        probe_rows.append(
            {
                "candidate_key": key[0],
                "source_pair_id": str(row.get("pair_id")),
                "label": _binary_label(labels[key]),
                **{field: row.get(field) for field in COMMITMENT_FEATURE_FIELDS},
            }
        )
    checks.append(gate_check("commitment_label_join_count", len(units), len(probe_rows)))
    prohibition = [
        {
            "field": field,
            "present_in_learner_allowlist": field in FEATURE_ALLOWLIST,
            "allowed_in_learner": False,
            "passed": field not in FEATURE_ALLOWLIST,
            "terminal": True,
        }
        for field in COMMITMENT_DENIED_FIELDS
    ]
    return {
        "probe_rows": probe_rows,
        "commitment_prohibition_rows": prohibition,
        "checks": checks,
        "passed": all(row["passed"] for row in checks)
        and all(row["passed"] for row in prohibition),
    }


def _authority_probe_rows(authority: AuthorityBatch, learner: LearnerBatch) -> JsonDict:
    rows_by_key = {str(row["candidate_key"]): row for row in authority.rows}
    allowed = []
    source = []
    mutation = []
    for index, key in enumerate(learner.keys):
        row = rows_by_key[key]
        source_data = row.get("source", {})
        mutation_data = row.get("mutation", {})
        common = {
            "candidate_key": key,
            "source_pair_id": str(source_data.get("source_pair_id")),
            "label": _binary_label(row.get("exact_label")),
        }
        allowed.append(
            {
                **common,
                **{
                    name: float(learner.matrix[index, column])
                    for column, name in enumerate(learner.feature_names)
                },
            }
        )
        source.append(
            {
                **common,
                "source_block": source_data.get("source_block"),
                "split": row.get("split"),
                "formulation_family": source_data.get("formulation_family"),
            }
        )
        mutation.append(
            {
                **common,
                "fault_family": mutation_data.get("fault_family", source_data.get("fault_family")),
                "schedule_id": mutation_data.get("schedule_id", source_data.get("schedule_id")),
                "event_type": source_data.get("event_type"),
            }
        )
    return {"allowed": allowed, "source": source, "mutation": mutation}


def _model_identity_probe_rows(
    raw_rows: Sequence[Mapping[str, Any]], authority: AuthorityBatch
) -> list[JsonDict]:
    by_key = {str(row["candidate_key"]): row for row in authority.rows}
    output = []
    for row in raw_rows:
        key = str(row["candidate_key"])
        authority_row = by_key[key]
        source = authority_row["source"]
        output.append(
            {
                "candidate_key": key,
                "source_pair_id": str(source.get("source_pair_id")),
                "label": _binary_label(authority_row.get("exact_label")),
                "model_id": row.get("model_id"),
            }
        )
    return output


def fit_all_shortcut_probes(
    learner: LearnerBatch,
    authority: AuthorityBatch,
    raw_rows: Sequence[Mapping[str, Any]],
    commitment_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Run all seven registered probes in stable order."""

    authority_rows = _authority_probe_rows(authority, learner)
    rows_by_name = {
        "length_only": authority_rows["allowed"],
        "serialization_only": authority_rows["allowed"],
        "source_metadata_only": authority_rows["source"],
        "mutation_metadata_only": authority_rows["mutation"],
        "model_identity_only": _model_identity_probe_rows(raw_rows, authority),
        "commitment_controls_only": list(commitment_rows),
        "allowed_learner_table": authority_rows["allowed"],
    }
    fold_rows = []
    prediction_rows = []
    intervals = []
    for offset, (name, spec) in enumerate(SHORTCUT_PROBE_SPECS.items()):
        result = fit_grouped_probe(
            rows_by_name[name],
            probe_name=name,
            feature_fields=spec["fields"],
            random_seed=RANDOM_SEED + offset,
            prohibited=bool(spec["prohibited"]),
        )
        fold_rows.extend(result["probe_rows"])
        prediction_rows.extend(result["prediction_rows"])
        intervals.append(result["interval_row"])
    return {
        "shortcut_probe_rows": fold_rows,
        "shortcut_prediction_rows": prediction_rows,
        "shortcut_interval_rows": intervals,
    }


def reduce_readiness(
    *,
    audit_complete: bool,
    direct_leakage_count: int,
    learner_invariant: bool,
    splits_balanced: bool,
    sources_disjoint: bool,
    models_complete: bool,
    prohibited_columns_absent: bool,
    shortcut_interval_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Separate terminal audit completion from permission to train."""

    prohibited = [row for row in shortcut_interval_rows if row.get("prohibited") is True]
    shortcut_clean = bool(prohibited) and all(
        interval_passes(row.get("ci95_upper")) for row in prohibited
    )
    ready = bool(
        audit_complete
        and direct_leakage_count == 0
        and learner_invariant
        and splits_balanced
        and sources_disjoint
        and models_complete
        and prohibited_columns_absent
        and shortcut_clean
    )
    if not audit_complete:
        verdict_class = "partial"
        verdict = "partial_blinded_feature_cold_audit"
    elif ready:
        verdict_class = "circular_positive"
        verdict = "circular_positive: blinded_feature_bank_ready"
    else:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_blinded_feature_shortcut_gate"
    return {
        "feature_isolation_audit_complete_score": int(audit_complete),
        "blinded_feature_bank_ready_score": int(ready),
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(stable)


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> JsonDict:
    """Build one schema-complete blocked, partial, or terminal artifact."""

    defaults: JsonDict = {
        "source_artifact_hashes": {},
        "rows": [],
        "per_candidate_rows": [],
        "raw_rebuild_rows": [],
        "candidate_key_rows": [],
        "family_completeness_rows": [],
        "join_replay_rows": [],
        "tensor_hash_rows": [],
        "label_balance_rows": [],
        "source_overlap_rows": [],
        "split_isolation_rows": [],
        "label_denial_replay_rows": [],
        "feature_schema_rows": [],
        "prohibited_field_rows": [],
        "sidecar_condition_rows": [],
        "sidecar_open_attempt_rows": [],
        "sidecar_permutation_rows": [],
        "sidecar_replacement_rows": [],
        "sidecar_removal_rows": [],
        "reference_prediction_rows": [],
        "learner_invariance_rows": [],
        "shortcut_probe_rows": [],
        "shortcut_prediction_rows": [],
        "shortcut_interval_rows": [],
        "commitment_prohibition_rows": [],
        "source_disagreement_rows": [],
        "read_only_enforcement_receipt": {},
        "direct_leakage_count": 0,
        "feature_isolation_audit_complete_score": 0,
        "blinded_feature_bank_ready_score": 0,
        "verdict_class": "partial",
        "honest_verdict": "partial_blinded_feature_cold_audit",
    }
    defaults.update(deepcopy(dict(evidence)))
    copied_checks = [deepcopy(dict(row)) for row in checks]
    if not copied_checks or any(row.get("passed") is not True for row in copied_checks):
        defaults.update(
            {
                "feature_isolation_audit_complete_score": 0,
                "blinded_feature_bank_ready_score": 0,
                "verdict_class": "blocked",
                "honest_verdict": "blocked_blinded_feature_cold_audit",
            }
        )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "schema": SCHEMA,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": copied_checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        **defaults,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(copied_checks),
        "verifier_is_oracle": True,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _verdict_matches(verdict_class: str, verdict: str) -> bool:
    prefixes = {
        "positive": ("complete_", "positive:"),
        "circular_positive": ("circular_positive:", "complete_circular_"),
        "null": ("complete_null_", "null:"),
        "blocked": ("blocked_",),
        "disqualified": ("complete_disqualified_", "disqualified:"),
        "partial": ("partial_",),
    }
    return verdict_class in prefixes and verdict.startswith(prefixes[verdict_class])


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate fields, bare gates, verdict meaning, rows, and checksum."""

    errors = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"required_field_missing:{field}")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    for field in (
        "direct_leakage_count",
        "feature_isolation_audit_complete_score",
        "blinded_feature_bank_ready_score",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field}_not_bare_int")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_must_be_true")
    verdict_class = str(artifact.get("verdict_class"))
    if verdict_class not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    elif not _verdict_matches(verdict_class, str(artifact.get("honest_verdict", ""))):
        errors.append("verdict_prefix_mismatch")
    if artifact.get("blinded_feature_bank_ready_score") == 1 and (
        artifact.get("feature_isolation_audit_complete_score") != 1
        or artifact.get("direct_leakage_count") != 0
        or verdict_class != "circular_positive"
    ):
        errors.append("readiness_state_mismatch")
    if (
        verdict_class == "disqualified"
        and artifact.get("feature_isolation_audit_complete_score") != 1
    ):
        errors.append("disqualification_without_complete_audit")
    if artifact.get("feature_isolation_audit_complete_score") == 1:
        expected_lengths = {
            "rows": EXPECTED_CANDIDATE_COUNT,
            "per_candidate_rows": EXPECTED_CANDIDATE_COUNT,
            "raw_rebuild_rows": EXPECTED_SOURCE_ROW_COUNT,
            "sidecar_condition_rows": 5,
            "learner_invariance_rows": 5,
            "shortcut_interval_rows": len(SHORTCUT_PROBE_SPECS),
        }
        for field, expected in expected_lengths.items():
            if len(artifact.get(field, [])) != expected:
                errors.append(f"terminal_row_count_mismatch:{field}")
        for field in expected_lengths:
            if any(row.get("terminal") is not True for row in artifact.get(field, [])):
                errors.append(f"nonterminal_rows:{field}")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _combine_raw_bank(bank: Mapping[str, Any]) -> JsonDict:
    sequence_rows = list(bank.get("sequence_feature_rows", []))
    parser_rows = list(bank.get("parser_feature_rows", []))
    feature_rows = list(bank.get("per_candidate_model_rows", []))
    disagreements = []

    def index(
        rows: Sequence[Mapping[str, Any]], kind: str
    ) -> dict[tuple[str, str], Mapping[str, Any]]:
        output = {}
        for row in rows:
            key = (str(row.get("candidate_id")), str(row.get("model_id")))
            if key in output:
                disagreements.append(
                    {"kind": f"duplicate_{kind}_key", "key": list(key), "terminal": True}
                )
            output[key] = row
        return output

    sequence = index(sequence_rows, "sequence")
    parser = index(parser_rows, "parser")
    features = index(feature_rows, "feature")
    all_keys = set(sequence) | set(parser) | set(features)
    rows = []
    replay = []
    for key in sorted(all_keys):
        sequence_row = sequence.get(key)
        parser_row = parser.get(key)
        feature_row = features.get(key)
        present = sequence_row is not None and parser_row is not None and feature_row is not None
        sequence_match = bool(
            present
            and all(
                sequence_row.get(field) == feature_row.get(field) for field in SEQUENCE_FEATURES
            )
        )
        passed = present and sequence_match
        replay.append(
            {
                "candidate_id": key[0],
                "model_id": key[1],
                "sequence_present": sequence_row is not None,
                "parser_present": parser_row is not None,
                "feature_present": feature_row is not None,
                "sequence_summary_replays": sequence_match,
                "passed": passed,
                "terminal": True,
            }
        )
        if not passed:
            disagreements.append(
                {"kind": "raw_component_disagreement", "key": list(key), "terminal": True}
            )
            continue
        rows.append(
            {
                "candidate_id": key[0],
                "prompt_hash": feature_row.get("prompt_hash"),
                "candidate_hash": feature_row.get("candidate_hash"),
                "model_id": key[1],
                **{field: sequence_row.get(field) for field in SEQUENCE_FEATURES},
                **{field: parser_row.get(field) for field in PARSER_FEATURES},
            }
        )
    return {"rows": rows, "join_replay_rows": replay, "source_disagreement_rows": disagreements}


def _join_rebuild_to_stored(
    rebuilt_rows: Sequence[Mapping[str, Any]], stored_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    rebuilt = {str(row.get("candidate_key")): row for row in rebuilt_rows}
    stored = {str(row.get("candidate_key")): row for row in stored_rows}
    keys = sorted(set(rebuilt) | set(stored))
    rows = []
    disagreements = []
    for key in keys:
        left = rebuilt.get(key)
        right = stored.get(key)
        passed = left == right and left is not None
        rows.append(
            {
                "candidate_key": key,
                "rebuilt_present": left is not None,
                "stored_present": right is not None,
                "rebuilt_row_hash": sha256_json(left) if left is not None else None,
                "stored_row_hash": sha256_json(right) if right is not None else None,
                "passed": passed,
                "terminal": True,
            }
        )
        if not passed:
            disagreements.append(
                {"kind": "learner_row_disagreement", "candidate_key": key, "terminal": True}
            )
    return {"rows": rows, "source_disagreement_rows": disagreements}


def sandbox_runtime_receipt(repo_root: Path, source_paths: Mapping[str, Path]) -> JsonDict:
    """Prove the child lacks learner training, network, GPU, LLM, and writes."""

    parent_netns = os.environ.get("CARNOT_EXP6999_PARENT_NETNS", "")
    child_netns = os.readlink("/proc/self/ns/net")
    write_rows = []
    for source_id, relative in source_paths.items():
        denied = False
        error = None
        try:
            with (repo_root / relative).open("rb+"):
                pass
        except OSError as exc:
            denied = True
            error = f"{type(exc).__name__}:{exc.errno}"
        write_rows.append(
            {"source_id": source_id, "path": str(relative), "write_denied": denied, "error": error}
        )
    gpu_devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    llm_modules = sorted(
        name for name in ("llama_cpp", "transformers", "torch") if name in sys.modules
    )
    receipt = {
        "fresh_process_pid": os.getpid(),
        "parent_process_pid": os.getppid(),
        "network_namespace": child_netns,
        "parent_network_namespace": parent_netns,
        "network_namespace_isolated": bool(parent_netns and child_netns != parent_netns),
        "gpu_devices_visible": gpu_devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "llm_modules_loaded": llm_modules,
        "learner_training_disabled": os.environ.get("CARNOT_TRAINING_DISABLED") == "1",
        "audit_probe_fitting_only": True,
        "source_write_rows": write_rows,
        "source_tree_read_only": bool(write_rows)
        and all(row["write_denied"] for row in write_rows),
        "output_is_temporary": not str(
            os.environ.get("CARNOT_EXP6999_CHILD_OUTPUT", "")
        ).startswith(str(repo_root / "results")),
    }
    receipt["training_disabled"] = receipt["learner_training_disabled"]
    receipt["passed"] = bool(
        receipt["network_namespace_isolated"]
        and not gpu_devices
        and receipt["cuda_visible_devices"] == ""
        and receipt["hf_hub_offline"]
        and receipt["transformers_offline"]
        and not llm_modules
        and receipt["learner_training_disabled"]
        and receipt["source_tree_read_only"]
        and receipt["output_is_temporary"]
    )
    return receipt


def _parse_sources_after_hash(repo_root: Path) -> JsonDict:
    bank = _selected_json_fields(
        repo_root / SOURCE_PATHS["exp6986"],
        (
            "three_family_feature_bank_complete_score",
            "per_candidate_model_rows",
            "sequence_feature_rows",
            "parser_feature_rows",
        ),
    )
    return {
        "exp6986": bank,
        "exp6987": json.loads((repo_root / SOURCE_PATHS["exp6987"]).read_text(encoding="utf-8")),
        "exp6997": json.loads((repo_root / SOURCE_PATHS["exp6997"]).read_text(encoding="utf-8")),
        "exp6998": json.loads((repo_root / SOURCE_PATHS["exp6998"]).read_text(encoding="utf-8")),
        "learner_view": _read_jsonl(repo_root / SOURCE_PATHS["learner_view"]),
    }


def _precondition_checks(sources: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[JsonDict]:
    exp6997 = sources["exp6997"]
    exp6998 = sources["exp6998"]
    bank = sources["exp6986"]
    learner_rows = sources["learner_view"]
    learner_open_rows = [
        row
        for row in exp6997.get("file_open_receipt_rows", [])
        if row.get("process") == "learner_loader"
    ]
    expected_learner_names = {"learner_view.jsonl", "learner_view.jsonl.manifest.json"}
    return [
        gate_check("read_only_fresh_process", True, receipt.get("passed")),
        gate_check(
            "structured_gate:exp6997.blinded_learner_view_ready_score",
            1,
            exp6997.get("blinded_learner_view_ready_score"),
        ),
        gate_check(
            "structured_gate:exp6998.commitment_control_complete_score",
            1,
            exp6998.get("commitment_control_complete_score"),
        ),
        gate_check("learner_row_count", EXPECTED_CANDIDATE_COUNT, len(learner_rows)),
        gate_check(
            "raw_model_row_count",
            EXPECTED_SOURCE_ROW_COUNT,
            len(bank.get("per_candidate_model_rows", [])),
        ),
        gate_check(
            "sequence_row_count",
            EXPECTED_SOURCE_ROW_COUNT,
            len(bank.get("sequence_feature_rows", [])),
        ),
        gate_check(
            "parser_row_count", EXPECTED_SOURCE_ROW_COUNT, len(bank.get("parser_feature_rows", []))
        ),
        gate_check("feature_allowlist", list(FEATURE_ALLOWLIST), exp6997.get("feature_allowlist")),
        gate_check(
            "learner_loader_receipts",
            True,
            bool(exp6997.get("learner_loader_rows"))
            and all(row.get("passed") is True for row in exp6997.get("learner_loader_rows", [])),
        ),
        gate_check(
            "learner_file_open_receipts",
            sorted(expected_learner_names),
            sorted(
                {
                    Path(str(row.get("path"))).name
                    for row in learner_open_rows
                    if row.get("passed") is True
                }
            ),
        ),
        gate_check(
            "learner_denied_access_receipts",
            True,
            bool(exp6997.get("denied_access_rows"))
            and all(row.get("passed") is True for row in exp6997.get("denied_access_rows", [])),
        ),
        gate_check(
            "commitment_row_count",
            EXPECTED_COMMITMENT_ROW_COUNT,
            len(exp6998.get("per_candidate_condition_model_rows", [])),
        ),
        gate_check(
            "commitment_rows_terminal",
            True,
            all(
                row.get("terminal") is True
                for row in exp6998.get("per_candidate_condition_model_rows", [])
            ),
        ),
    ]


def _terminal_candidate_rows(
    rebuilt: Mapping[str, Any], joins: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    joins_by_key = {str(row["candidate_key"]): row for row in joins}
    output = []
    for row in rebuilt["family_completeness_rows"]:
        key = str(row.get("candidate_key", ""))
        output.append(
            {
                "candidate_key": key or None,
                "family_count": row.get("family_count", 0),
                "families_complete": row.get("passed") is True,
                "learner_join_replays": joins_by_key.get(key, {}).get("passed") is True,
                "terminal": True,
            }
        )
    return output


def _feature_schema_rows() -> list[JsonDict]:
    return [
        {
            "field": field,
            "position": index,
            "numeric_only": True,
            "prohibited": _denied_field(field),
            "included_in_learner": True,
            "passed": not _denied_field(field),
            "terminal": True,
        }
        for index, field in enumerate(FEATURE_ALLOWLIST)
    ]


def build_from_repo(
    repo_root: Path = REPO_ROOT,
    *,
    run_date: str = RUN_DATE,
    output_path: Path | None = None,
) -> JsonDict:
    """Run the full audit inside an already-created read-only child."""

    started = time.perf_counter()
    receipt = sandbox_runtime_receipt(repo_root, SOURCE_PATHS)
    hashed = load_hashed_inputs(
        repo_root,
        source_paths=SOURCE_PATHS,
        expected_hashes=EXPECTED_SOURCE_HASHES,
    )
    checks = [*hashed["checks"], gate_check("read_only_fresh_process", True, receipt.get("passed"))]
    if hashed["passed"] is not True:
        return build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            checks=checks,
            evidence={
                "source_artifact_hashes": hashed["hashes"],
                "read_only_enforcement_receipt": receipt,
            },
        )
    try:
        sources = _parse_sources_after_hash(repo_root)
    except (OSError, UnicodeError, json.JSONDecodeError, HashMismatchError) as exc:
        checks.append(
            gate_check("source_parse", "all hashed sources parse", f"{type(exc).__name__}: {exc}")
        )
        return build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            checks=checks,
            evidence={
                "source_artifact_hashes": hashed["hashes"],
                "read_only_enforcement_receipt": receipt,
            },
        )
    checks = [*hashed["checks"], *_precondition_checks(sources, receipt)]
    if any(row["passed"] is not True for row in checks):
        return build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            checks=checks,
            evidence={
                "source_artifact_hashes": hashed["hashes"],
                "read_only_enforcement_receipt": receipt,
            },
        )

    combined = _combine_raw_bank(sources["exp6986"])
    rebuilt = rebuild_wide_rows(combined["rows"])
    schema = audit_learner_schema(sources["learner_view"], FEATURE_ALLOWLIST)
    learner_path = repo_root / SOURCE_PATHS["learner_view"]
    label_path = repo_root / SOURCE_PATHS["label_sidecar"]
    mutation_path = repo_root / SOURCE_PATHS["mutation_sidecar"]
    learner = load_learner_view(learner_path, FEATURE_ALLOWLIST)
    rebuilt_batch = _materialize(rebuilt["learner_rows"], FEATURE_ALLOWLIST)
    joins = _join_rebuild_to_stored(rebuilt["learner_rows"], sources["learner_view"])
    authority = join_authority(learner, label_path, mutation_path)
    split_audit = audit_splits(authority)
    commitment = build_commitment_probe_rows(sources["exp6998"])

    if output_path is None:
        temporary_context = tempfile.TemporaryDirectory(prefix="carnot-exp6999-conditions-")
        condition_root = Path(temporary_context.name)
    else:
        temporary_context = None
        condition_root = output_path.parent / "sidecar-conditions"
    try:
        sidecars = audit_sidecar_conditions(
            learner_path,
            FEATURE_ALLOWLIST,
            mutation_path,
            condition_root,
        )
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()

    probes = fit_all_shortcut_probes(
        learner,
        authority,
        rebuilt["raw_rebuild_rows"],
        commitment["probe_rows"],
    )
    disagreements = [
        *combined["source_disagreement_rows"],
        *rebuilt["source_disagreement_rows"],
        *joins["source_disagreement_rows"],
    ]
    for row in commitment["checks"]:
        if row["passed"] is not True:
            disagreements.append(
                {
                    "kind": "commitment_control_disagreement",
                    "check": row["check"],
                    "expected_value": row["expected_value"],
                    "observed_value": row["observed_value"],
                    "terminal": True,
                }
            )
    models_complete = bool(rebuilt["family_completeness_rows"]) and all(
        row["passed"] for row in rebuilt["family_completeness_rows"]
    )
    learner_invariant = (
        bool(sidecars["learner_invariance_rows"])
        and all(row["passed"] for row in sidecars["learner_invariance_rows"])
        and learner.tensor_hash == rebuilt_batch.tensor_hash
        and learner.prediction_hash == rebuilt_batch.prediction_hash
    )
    audit_complete = bool(
        len(rebuilt["raw_rebuild_rows"]) == EXPECTED_SOURCE_ROW_COUNT
        and len(rebuilt["family_completeness_rows"]) == EXPECTED_CANDIDATE_COUNT
        and len(sidecars["sidecar_condition_rows"]) == 5
        and all(row.get("terminal") is True for row in sidecars["sidecar_condition_rows"])
        and len(probes["shortcut_interval_rows"]) == len(SHORTCUT_PROBE_SPECS)
        and all(row.get("terminal") is True for row in probes["shortcut_interval_rows"])
        and commitment["passed"]
        and receipt.get("passed") is True
    )
    reduced = reduce_readiness(
        audit_complete=audit_complete,
        direct_leakage_count=schema["direct_leakage_count"],
        learner_invariant=learner_invariant,
        splits_balanced=split_audit["all_required_balanced"],
        sources_disjoint=split_audit["all_sources_disjoint"],
        models_complete=models_complete,
        prohibited_columns_absent=not schema["prohibited_field_rows"]
        and all(row["passed"] for row in commitment["commitment_prohibition_rows"]),
        shortcut_interval_rows=probes["shortcut_interval_rows"],
    )
    candidate_rows = _terminal_candidate_rows(rebuilt, joins["rows"])
    learner_denial = [
        {
            "check": "learner_loader_signature",
            "expected_value": ["learner_path", "allowlist"],
            "observed_value": list(inspect.signature(load_learner_view).parameters),
            "passed": list(inspect.signature(load_learner_view).parameters)
            == ["learner_path", "allowlist"],
            "terminal": True,
        },
        {
            "check": "learner_sidecar_open_count",
            "expected_value": 0,
            "observed_value": sum(
                row["sidecar_open_attempted"] for row in sidecars["sidecar_open_attempt_rows"]
            ),
            "passed": not any(
                row["sidecar_open_attempted"] for row in sidecars["sidecar_open_attempt_rows"]
            ),
            "terminal": True,
        },
    ]
    tensor_rows = [
        {
            "view": "stored_learner",
            "row_count": len(learner.keys),
            "tensor_hash": learner.tensor_hash,
            "prediction_hash": learner.prediction_hash,
            "terminal": True,
        },
        {
            "view": "rebuilt_from_raw",
            "row_count": len(rebuilt_batch.keys),
            "tensor_hash": rebuilt_batch.tensor_hash,
            "prediction_hash": rebuilt_batch.prediction_hash,
            "terminal": True,
        },
    ]
    evidence = {
        "source_artifact_hashes": hashed["hashes"],
        "rows": candidate_rows,
        "per_candidate_rows": deepcopy(candidate_rows),
        "raw_rebuild_rows": rebuilt["raw_rebuild_rows"],
        "candidate_key_rows": rebuilt["candidate_key_rows"],
        "family_completeness_rows": rebuilt["family_completeness_rows"],
        "join_replay_rows": [*combined["join_replay_rows"], *joins["rows"]],
        "tensor_hash_rows": tensor_rows,
        "label_balance_rows": split_audit["label_balance_rows"],
        "source_overlap_rows": split_audit["source_overlap_rows"],
        "split_isolation_rows": split_audit["split_isolation_rows"],
        "label_denial_replay_rows": learner_denial,
        "feature_schema_rows": _feature_schema_rows(),
        "prohibited_field_rows": schema["prohibited_field_rows"],
        **sidecars,
        **probes,
        "commitment_prohibition_rows": commitment["commitment_prohibition_rows"],
        "source_disagreement_rows": disagreements,
        "read_only_enforcement_receipt": receipt,
        "direct_leakage_count": schema["direct_leakage_count"],
        **reduced,
    }
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        checks=checks,
        evidence=evidence,
    )
    if output_path is not None:
        write_json_atomic(output_path, artifact)
    return artifact


def fresh_process_command(
    *,
    executable: Path,
    wrapper: Path,
    repo_root: Path,
    writable_root: Path,
    output_path: Path,
    run_date: str,
) -> list[str]:
    """Build the Linux sandbox command with one temporary writable mount."""

    parent_netns = os.readlink("/proc/self/ns/net")
    return [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--bind",
        str(writable_root),
        str(writable_root),
        "--chdir",
        str(repo_root),
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "",
        "--setenv",
        "HF_HUB_OFFLINE",
        "1",
        "--setenv",
        "TRANSFORMERS_OFFLINE",
        "1",
        "--setenv",
        "JAX_PLATFORMS",
        "cpu",
        "--setenv",
        "CARNOT_TRAINING_DISABLED",
        "1",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "CARNOT_EXP6999_PARENT_NETNS",
        parent_netns,
        "--setenv",
        "CARNOT_EXP6999_CHILD_OUTPUT",
        str(output_path),
        "--",
        str(executable),
        str(wrapper),
        "--fresh-child",
        "--date",
        run_date,
        "--output",
        str(output_path),
    ]


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write one result atomically so readers never observe partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def run_controller(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:  # pragma: no cover - exercised by the required command.
    """Run the child and keep source files read-only until final output."""

    started = time.perf_counter()
    final_path = result_path or (repo_root / RESULT_PATH)
    if shutil.which("bwrap") is None:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            checks=[gate_check("bubblewrap_available", True, False)],
            evidence={},
        )
        write_json_atomic(final_path, artifact)
        return artifact
    before = {source_id: sha256_path(repo_root / path) for source_id, path in SOURCE_PATHS.items()}
    with tempfile.TemporaryDirectory(prefix="carnot-exp6999-") as directory:
        writable_root = Path(directory)
        child_output = writable_root / "child-result.json"
        command = fresh_process_command(
            executable=Path(sys.executable),
            wrapper=repo_root / WRAPPER_PATH,
            repo_root=repo_root,
            writable_root=writable_root,
            output_path=child_output,
            run_date=run_date,
        )
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and child_output.is_file():
            artifact = json.loads(child_output.read_text(encoding="utf-8"))
        else:
            artifact = build_artifact(
                run_date=run_date,
                duration_s=time.perf_counter() - started,
                checks=[
                    gate_check("fresh_process_exit_code", 0, completed.returncode),
                    gate_check("fresh_process_output", True, child_output.is_file()),
                ],
                evidence={},
            )
    after = {source_id: sha256_path(repo_root / path) for source_id, path in SOURCE_PATHS.items()}
    receipt = dict(artifact.get("read_only_enforcement_receipt", {}))
    receipt.update(
        {
            "controller_source_hashes_before": before,
            "controller_source_hashes_after": after,
            "controller_source_hashes_unchanged": before == after,
            "fresh_child_exit_code": completed.returncode,
            "fresh_child_stderr_hash": sha256_text(completed.stderr),
            "sandbox_command_hash": sha256_json(command),
        }
    )
    receipt["passed"] = bool(
        receipt.get("passed") and before == after and completed.returncode == 0
    )
    artifact["read_only_enforcement_receipt"] = receipt
    artifact["duration_s"] = time.perf_counter() - started
    if not receipt["passed"] and artifact.get("verdict_class") != "blocked":
        checks = list(artifact.get("preconditions_checked", []))
        checks.append(gate_check("controller_read_only_enforcement", True, receipt["passed"]))
        artifact["preconditions_checked"] = checks
        artifact["gate_check_summary"] = gate_summary(checks)
        artifact["feature_isolation_audit_complete_score"] = 0
        artifact["blinded_feature_bank_ready_score"] = 0
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = "blocked_blinded_feature_cold_audit"
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(final_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command boundary.
    """Run the controller or its private child mode."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--fresh-child", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.fresh_child:
        if args.output is None:
            parser.error("--output is required with --fresh-child")
        artifact = build_from_repo(REPO_ROOT, run_date=args.date, output_path=args.output)
        errors = validate_artifact(artifact)
        if errors:
            raise RuntimeError(f"child_artifact_validation_failed:{errors}")
        write_json_atomic(args.output, artifact)
        return 0
    run_controller(run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
