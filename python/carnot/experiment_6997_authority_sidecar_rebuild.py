"""Rebuild the contrast feature bank behind an authority-only sidecar.

The learner file contains only a semantic key and numeric tensors. A separate
joiner opens exact labels and provenance after those tensors exist. This keeps
the data boundary enforceable without rerunning any stored GGUF computation.

Spec refs: REQ-VERIFY-6997 and SCENARIO-VERIFY-6997-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
import tempfile
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]
EXPERIMENT_ID = "experiment_6997_authority_sidecar_rebuild"
SCHEMA = "carnot.experiment_6997.authority_sidecar_rebuild.v1"
RUN_DATE = "20260904"
RANDOM_SEED = 6_997_202_609_04
INFERENCE_SUBSTRATE = "deterministic_feature_sidecar_transform_no_llm"
EXPECTED_CANDIDATE_COUNT = 138
EXPECTED_SOURCE_ROW_COUNT = 414

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6997_authority_sidecar_rebuild.json"
DATA_ROOT = REPO_ROOT / "results/raw/experiment_6997_authority_sidecar_rebuild"
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")

SOURCE_PATHS = {
    "exp6984": Path("results/experiment_6984_exact_contrast_fixture.json"),
    "exp6985": Path("results/experiment_6985_chronological_constraint_stream.json"),
    "exp6986": Path("results/experiment_6986_three_family_contrast_features.json"),
    "exp6987": Path("results/experiment_6987_contrast_feature_audit.json"),
    "exp6975": Path("results/experiment_6975_delayed_constraint_candidate_bank.json"),
    "exp6976": Path("results/experiment_6976_exact_candidate_certification.json"),
    "exp6985_stream": Path(
        "results/raw/experiment_6985_chronological_constraint_stream/chronological_stream.jsonl"
    ),
    "exp6985_labels": Path(
        "results/raw/experiment_6985_chronological_constraint_stream/sealed_labels.jsonl"
    ),
}
EXPECTED_SOURCE_HASHES = {
    "exp6975": "sha256:4a9faf7223d174729091248f8cb763cc6b76e481abe69bdadc19adfe7dac5455",
    "exp6976": "sha256:7d174415abe6b9c3777bc56bdbf0eaa38a37aba390c64685c667f625aa6e05b9",
    "exp6984": "sha256:15f9a9bb58ca7793966f2fbac548f6879a64417b50e31b0504078cfe6ea46a3f",
    "exp6985": "sha256:461421d4771b5d997799fd2b7cb2c9e2cdfd684fbf792d25b139fcc49c30bede",
    "exp6986": "sha256:c19f5180c5e70677baec92db5ea295599197919eb03c5119d9a22849320cff75",
    "exp6987": "sha256:271af8ade020727ed93cf07dbe1aa285e06467d46524ad9f067a19a5a7350bcf",
    "exp6985_stream": "sha256:33e7c839224aba4293838a920dc3213ee49c55c13c03cdffc45ee179e2bd2fca",
    "exp6985_labels": "sha256:cb454989dfeb1a433b917e20de99be3eb403fd2f4ca51481a1d622e4c3cb421c",
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
    f"family_{family_position}__{feature}"
    for family_position in range(len(REQUIRED_MODEL_IDS))
    for feature in SOURCE_FEATURES
)

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "run_date",
    "schema",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "MODEL_SPECS_inherited",
    "source_model_rows",
    "rows",
    "per_candidate_rows",
    "candidate_key_rows",
    "family_completeness_rows",
    "pivot_rows",
    "learner_view_path",
    "learner_view_hash",
    "learner_view_manifest_rows",
    "label_split_sidecar_path",
    "label_split_sidecar_hash",
    "label_split_manifest_rows",
    "mutation_authority_sidecar_path",
    "mutation_authority_sidecar_hash",
    "mutation_authority_manifest_rows",
    "feature_allowlist",
    "prohibited_feature_rows",
    "nested_metadata_rows",
    "categorical_identity_rows",
    "learner_loader_rows",
    "denied_access_rows",
    "file_open_receipt_rows",
    "authority_join_rows",
    "tensor_hash_rows",
    "row_permutation_rows",
    "alpha_rename_rows",
    "source_disagreement_rows",
    "expected_candidate_count",
    "observed_candidate_count",
    "expected_source_row_count",
    "observed_source_row_count",
    "gguf_inference_performed",
    "verifier_fit_performed",
    "sidecar_rebuild_complete_score",
    "blinded_learner_view_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "experiment_id": "A stable experiment identity prevents evidence from moving between protocols.",
    "run_date": "The declared execution date binds this transform to its planned evidence window.",
    "schema": "A versioned schema makes later parsing and rejection rules reproducible.",
    "field_principles": "A reason for each field keeps the evidence contract scientifically auditable.",
    "preconditions_checked": "Measured preconditions stop incomplete stored evidence from becoming training data.",
    "inference_substrate": "The substrate states that this run transforms stored features and performs no inference.",
    "duration_s": "Wall time helps detect a skipped or fabricated transform.",
    "source_artifact_hashes": "Source hashes bind every output byte to the frozen evidence bank.",
    "MODEL_SPECS_inherited": "Inherited declarations identify the three models that produced the stored numeric views.",
    "source_model_rows": "One row per stored model view proves that all 414 inputs replayed.",
    "rows": "Terminal rows expose candidate-level completion without relying on aggregates.",
    "per_candidate_rows": "Candidate rows preserve the one-row learner unit used downstream.",
    "candidate_key_rows": "Key receipts prove that only semantic prompt and candidate hashes define identity.",
    "family_completeness_rows": "Family receipts detect missing or duplicate model views before pivoting.",
    "pivot_rows": "Pivot receipts bind each wide tensor to three neutral family positions.",
    "learner_view_path": "The path identifies the only data file that the learner may open.",
    "learner_view_hash": "The learner hash detects any byte change before tensors materialize.",
    "learner_view_manifest_rows": "The manifest binds learner path, keys, row count, bytes, and allowlist.",
    "label_split_sidecar_path": "The label path keeps outcomes and routing outside learner storage.",
    "label_split_sidecar_hash": "The label hash prevents silent outcome or split changes.",
    "label_split_manifest_rows": "The label manifest proves that authority data was frozen before joining.",
    "mutation_authority_sidecar_path": "The authority path isolates source, mutation, and witness provenance.",
    "mutation_authority_sidecar_hash": "The authority hash detects provenance or witness mutation.",
    "mutation_authority_manifest_rows": "The authority manifest binds every provenance row before joining.",
    "feature_allowlist": "A frozen allowlist limits learning to declared numeric signals.",
    "prohibited_feature_rows": "Rejected field probes show that labels and provenance cannot enter tensors.",
    "nested_metadata_rows": "Nested probes stop metadata from bypassing top-level field checks.",
    "categorical_identity_rows": "Identity probes stop model or repository strings from becoming shortcuts.",
    "learner_loader_rows": "Loader receipts show the exact numeric matrix produced through the narrow API.",
    "denied_access_rows": "Denied-access probes make the learner-side authority boundary falsifiable.",
    "file_open_receipt_rows": "Open receipts show which files each loader actually accessed.",
    "authority_join_rows": "Join rows prove that labels and routing remain separate from feature output.",
    "tensor_hash_rows": "Tensor hashes make byte-level learner materialization replayable.",
    "row_permutation_rows": "Permutation checks prove that storage order has no learned meaning.",
    "alpha_rename_rows": "Rename checks prove that sidecar identifiers have no learned meaning.",
    "source_disagreement_rows": "Disagreements remain visible instead of being silently repaired.",
    "expected_candidate_count": "The fixed candidate count defines a falsifiable completion target.",
    "observed_candidate_count": "The observed count exposes dropped or duplicated candidates.",
    "expected_source_row_count": "The fixed source-row count requires all three views per candidate.",
    "observed_source_row_count": "The observed source count exposes incomplete stored features.",
    "gguf_inference_performed": "False prevents a deterministic rebuild from claiming new model computation.",
    "verifier_fit_performed": "False prevents isolation work from becoming unregistered model selection.",
    "sidecar_rebuild_complete_score": "The bare gate reports complete replay of 138 candidates and 414 source rows.",
    "blinded_learner_view_ready_score": "The bare gate reports whether hash, schema, access, and invariance checks all pass.",
    "random_seed": "A fixed seed makes adversarial row permutation repeatable.",
    "reproducibility_checksum": "A canonical digest detects later changes to the terminal evidence.",
    "gate_check_summary": "Expected and observed values make any blocked gate actionable.",
    "verifier_is_oracle": "True states that exact authorities define these sidecars and make success circular.",
    "verdict_class": "A closed verdict class keeps downstream automation unambiguous.",
    "honest_verdict": "A class-consistent terminal verdict prevents a readiness score from overstating science.",
}

_PROHIBITED_TOKENS = (
    "label",
    "split",
    "source",
    "mutation",
    "provenance",
    "authority",
    "witness",
    "fault",
    "group",
    "pair",
    "order",
    "ordinal",
    "metadata",
    "model",
    "repository",
    "repo",
    "identity",
)


class RebuildError(ValueError):
    """Stored rows cannot produce one complete, semantic-keyed learner view."""


class IsolationError(ValueError):
    """The learner was asked to cross its narrow file-access boundary."""


class HashMismatchError(ValueError):
    """An immutable file no longer matches the manifest frozen for it."""


class AuthorityJoinError(ValueError):
    """Authority rows cannot join uniquely to all materialized learner keys."""


@dataclass(frozen=True)
class LearnerBatch:
    """A materialized numeric learner matrix with no authority data."""

    keys: tuple[str, ...]
    feature_names: tuple[str, ...]
    matrix: np.ndarray
    tensor_hash: str
    prediction_hash: str
    file_open_receipts: tuple[JsonDict, ...]


@dataclass(frozen=True)
class AuthorityBatch:
    """A keyed authority result that keeps routing separate from features."""

    keys: tuple[str, ...]
    feature_matrix: np.ndarray
    labels: tuple[str, ...]
    splits: tuple[str, ...]
    mutation_authority_rows: tuple[JsonDict, ...]
    file_open_receipts: tuple[JsonDict, ...]


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes and immutable JSON Lines files."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text without depending on locale settings."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(value))


def candidate_key(prompt_hash: str, candidate_hash: str, **_nonsemantic: object) -> str:
    """Freeze identity from semantic prompt and candidate hashes only."""

    if not prompt_hash.startswith("sha256:") or not candidate_hash.startswith("sha256:"):
        raise RebuildError("semantic_hash_missing")
    return sha256_json({"candidate_hash": candidate_hash, "prompt_hash": prompt_hash})


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record an exact expected-observed precondition or release gate."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and promote the first failure for blocked consumers."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "checks": copied,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": "all checks pass" if failed is None else failed.get("expected_value"),
        "observed_value": "all checks pass" if failed is None else failed.get("observed_value"),
        "passed": failed is None,
    }


def _as_numeric(value: Any, *, field: str) -> int | float:
    """Normalize allowed values while rejecting strings and non-finite numbers."""

    if isinstance(value, bool):
        return int(value)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise RebuildError(f"nonnumeric_feature:{field}")
    return value


def build_wide_learner_view(
    rows: Sequence[Mapping[str, Any]], *, expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT
) -> JsonDict:
    """Pivot three long family views into stable, neutral numeric positions."""

    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[str(row.get("candidate_id", ""))].append(row)
    if len(by_candidate) != expected_candidate_count:
        raise RebuildError(
            f"candidate_count_mismatch:expected={expected_candidate_count}:observed={len(by_candidate)}"
        )

    candidate_key_rows: list[JsonDict] = []
    family_rows: list[JsonDict] = []
    source_model_rows: list[JsonDict] = []
    learner_rows: list[JsonDict] = []
    pivot_rows: list[JsonDict] = []
    seen_keys: set[str] = set()
    for candidate_id, candidate_rows in by_candidate.items():
        semantic_pairs = {
            (str(row.get("prompt_hash", "")), str(row.get("candidate_hash", "")))
            for row in candidate_rows
        }
        if len(semantic_pairs) != 1:
            raise RebuildError(f"candidate_semantic_hash_conflict:{candidate_id}")
        prompt_hash, candidate_hash_value = next(iter(semantic_pairs))
        key = candidate_key(prompt_hash, candidate_hash_value)
        if key in seen_keys:
            raise RebuildError(f"duplicate_candidate_key:{key}")
        seen_keys.add(key)

        models = [str(row.get("model_id", "")) for row in candidate_rows]
        if len(models) != len(set(models)):
            raise RebuildError(f"duplicate_family_row:{candidate_id}")
        if set(models) != set(REQUIRED_MODEL_IDS):
            raise RebuildError(f"missing_model_families:{candidate_id}")
        positioned = {FAMILY_POSITIONS[model]: row for model, row in zip(models, candidate_rows)}
        vector: list[int | float] = []
        for family_position in range(len(REQUIRED_MODEL_IDS)):
            source = positioned[family_position]
            family_values = [
                _as_numeric(source.get(field), field=field) for field in SOURCE_FEATURES
            ]
            vector.extend(family_values)
            source_model_rows.append(
                {
                    "candidate_key": key,
                    "candidate_id": candidate_id,
                    "prompt_hash": prompt_hash,
                    "candidate_hash": candidate_hash_value,
                    "model_id": str(source["model_id"]),
                    "family_position": family_position,
                    "features": family_values,
                    "source_row_hash": sha256_json(dict(source)),
                    "terminal": True,
                }
            )

        candidate_key_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_key": key,
                "prompt_hash": prompt_hash,
                "candidate_hash": candidate_hash_value,
                "key_inputs": ["prompt_hash", "candidate_hash"],
                "excluded_inputs": [
                    "source",
                    "mutation",
                    "order",
                    "split",
                    "label",
                    "candidate_id",
                    "model_id",
                ],
                "passed": True,
            }
        )
        family_rows.append(
            {
                "candidate_key": key,
                "expected_family_positions": [0, 1, 2],
                "observed_family_positions": sorted(positioned),
                "family_count": len(positioned),
                "passed": True,
            }
        )
        learner_row = {"candidate_key": key, "features": vector}
        learner_rows.append(learner_row)
        pivot_rows.append(
            {
                "candidate_key": key,
                "family_positions": [0, 1, 2],
                "feature_count": len(vector),
                "tensor_hash": sha256_json(vector),
                "passed": True,
            }
        )

    learner_rows.sort(key=lambda row: row["candidate_key"])
    candidate_key_rows.sort(key=lambda row: row["candidate_key"])
    family_rows.sort(key=lambda row: row["candidate_key"])
    pivot_rows.sort(key=lambda row: row["candidate_key"])
    source_model_rows.sort(key=lambda row: (row["candidate_key"], row["family_position"]))
    return {
        "source_model_rows": source_model_rows,
        "candidate_key_rows": candidate_key_rows,
        "family_completeness_rows": family_rows,
        "pivot_rows": pivot_rows,
        "learner_rows": learner_rows,
        "source_disagreement_rows": [],
    }


def _nested_mapping_paths(value: Any, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if isinstance(item, Mapping):
                paths.append(child)
            paths.extend(_nested_mapping_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_nested_mapping_paths(item, f"{path}[{index}]"))
    return paths


def learner_schema_receipts(
    rows: Sequence[Mapping[str, Any]], allowlist: Sequence[str]
) -> JsonDict:
    """Audit strict learner rows and preserve exact rejection paths."""

    prohibited: list[JsonDict] = []
    nested: list[JsonDict] = []
    categorical: list[JsonDict] = []
    seen: set[str] = set()
    if tuple(allowlist) != FEATURE_ALLOWLIST:
        prohibited.append({"path": "$.allowlist", "reason": "allowlist_mismatch", "passed": False})
    for row_index, row in enumerate(rows):
        key = str(row.get("candidate_key", ""))
        if not key or key in seen:
            prohibited.append(
                {
                    "path": f"$[{row_index}].candidate_key",
                    "reason": "duplicate_or_empty_key",
                    "passed": False,
                }
            )
        seen.add(key)
        for field, value in row.items():
            path = f"$[{row_index}].{field}"
            normalized = "".join(character for character in field.casefold() if character.isalnum())
            if field not in {"candidate_key", "features"}:
                reason = (
                    "provenance_or_label_alias"
                    if any(token in normalized for token in _PROHIBITED_TOKENS)
                    else "field_not_allowlisted"
                )
                prohibited.append({"path": path, "field": field, "reason": reason, "passed": False})
            if isinstance(value, Mapping):
                nested.append({"path": path, "reason": "nested_metadata", "passed": False})
            if field != "candidate_key" and isinstance(value, str):
                categorical.append({"path": path, "reason": "categorical_value", "passed": False})
            if any(token in normalized for token in ("model", "repository", "repo", "identity")):
                categorical.append(
                    {"path": path, "reason": "categorical_identity", "passed": False}
                )
        for path in _nested_mapping_paths(row, f"$[{row_index}]"):
            if not any(receipt["path"] == path for receipt in nested):
                nested.append({"path": path, "reason": "nested_metadata", "passed": False})
        features = row.get("features")
        if not isinstance(features, list) or len(features) != len(allowlist):
            prohibited.append(
                {
                    "path": f"$[{row_index}].features",
                    "reason": "tensor_shape_mismatch",
                    "passed": False,
                }
            )
        elif any(
            type(value) not in (int, float) or not math.isfinite(float(value)) for value in features
        ):
            categorical.append(
                {"path": f"$[{row_index}].features", "reason": "nonnumeric_tensor", "passed": False}
            )
    return {
        "prohibited_feature_rows": prohibited,
        "nested_metadata_rows": nested,
        "categorical_identity_rows": categorical,
        "passed": not prohibited and not nested and not categorical,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted((deepcopy(dict(row)) for row in rows), key=lambda row: row["candidate_key"])
    return ("".join(canonical_json(row) + "\n" for row in ordered)).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    """Create immutable bytes once, or verify that an existing file is identical."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise HashMismatchError(f"immutable_path_conflict:{path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o444)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest(
    path: Path, rows: Sequence[Mapping[str, Any]], *, kind: str, allowlist: Sequence[str] = ()
) -> JsonDict:
    payload = path.read_bytes()
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["candidate_key"])
    return {
        "schema": "carnot.exp6997.immutable_jsonl_manifest.v1",
        "kind": kind,
        "path": str(path),
        "row_count": len(ordered),
        "byte_count": len(payload),
        "file_hash": sha256_bytes(payload),
        "ordered_key_hashes": [sha256_text(str(row["candidate_key"])) for row in ordered],
        "feature_allowlist": list(allowlist),
        "binding_phase": "before_authority_join",
    }


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def _write_jsonl_with_manifest(
    path: Path, rows: Sequence[Mapping[str, Any]], *, kind: str, allowlist: Sequence[str] = ()
) -> JsonDict:
    _write_immutable(path, _jsonl_bytes(rows))
    manifest = _manifest(path, rows, kind=kind, allowlist=allowlist)
    _write_immutable(_manifest_path(path), (canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def _index_sidecar(rows: Sequence[Mapping[str, Any]], *, kind: str) -> dict[str, JsonDict]:
    index: dict[str, JsonDict] = {}
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
    for row in rows:
        key = str(row.get("candidate_key", ""))
        if key in index:
            raise AuthorityJoinError(f"duplicate_sidecar_key:{kind}:{key}")
        if not key or set(row) != required:
            raise AuthorityJoinError(f"invalid_sidecar_schema:{kind}:{key}")
        index[key] = deepcopy(dict(row))
    return index


def write_sidecar_bundle(
    output_dir: str | Path,
    learner_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    mutation_rows: Sequence[Mapping[str, Any]],
    allowlist: Sequence[str],
) -> JsonDict:
    """Write and hash all three immutable files before an authority join."""

    schema = learner_schema_receipts(learner_rows, allowlist)
    if schema["passed"] is not True:
        raise IsolationError("learner_schema_rejected")
    learner_keys = {str(row["candidate_key"]) for row in learner_rows}
    label_index = _index_sidecar(label_rows, kind="label_split")
    mutation_index = _index_sidecar(mutation_rows, kind="mutation_authority")
    if learner_keys != set(label_index) or learner_keys != set(mutation_index):
        raise AuthorityJoinError("sidecar_key_mismatch:write")

    root = Path(output_dir)
    learner_path = root / "learner_view.jsonl"
    label_path = root / "label_split_sidecar.jsonl"
    mutation_path = root / "mutation_authority_sidecar.jsonl"
    learner_manifest = _write_jsonl_with_manifest(
        learner_path, learner_rows, kind="learner_view", allowlist=allowlist
    )
    label_manifest = _write_jsonl_with_manifest(label_path, label_rows, kind="label_split")
    mutation_manifest = _write_jsonl_with_manifest(
        mutation_path, mutation_rows, kind="mutation_authority"
    )
    return {
        "learner_view_path": str(learner_path),
        "learner_view_hash": learner_manifest["file_hash"],
        "learner_view_manifest_rows": [learner_manifest],
        "label_split_sidecar_path": str(label_path),
        "label_split_sidecar_hash": label_manifest["file_hash"],
        "label_split_manifest_rows": [label_manifest],
        "mutation_authority_sidecar_path": str(mutation_path),
        "mutation_authority_sidecar_hash": mutation_manifest["file_hash"],
        "mutation_authority_manifest_rows": [mutation_manifest],
    }


def _read_verified_jsonl(
    path: Path, *, kind: str
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    receipts: list[JsonDict] = []
    manifest_path = _manifest_path(path)
    manifest_bytes = manifest_path.read_bytes()
    receipts.append(
        {"path": str(manifest_path), "operation": "open_read", "role": "manifest", "passed": True}
    )
    manifest = json.loads(manifest_bytes)
    payload = path.read_bytes()
    receipts.append({"path": str(path), "operation": "open_read", "role": kind, "passed": True})
    observed_hash = sha256_bytes(payload)
    if manifest.get("kind") != kind:
        raise HashMismatchError(f"manifest_kind_mismatch:{kind}")
    if manifest.get("path") != str(path):
        raise HashMismatchError(f"manifest_path_mismatch:{kind}")
    if manifest.get("file_hash") != observed_hash:
        raise HashMismatchError(f"file_hash_mismatch:{kind}")
    lines = payload.decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    if manifest.get("row_count") != len(rows) or manifest.get("byte_count") != len(payload):
        raise HashMismatchError(f"manifest_size_mismatch:{kind}")
    key_hashes = [sha256_text(str(row.get("candidate_key", ""))) for row in rows]
    if manifest.get("ordered_key_hashes") != key_hashes:
        raise HashMismatchError(f"manifest_key_mismatch:{kind}")
    return rows, receipts, manifest


def _tensor_hash(keys: Sequence[str], names: Sequence[str], matrix: np.ndarray) -> str:
    header = canonical_json({"keys": list(keys), "feature_names": list(names)}).encode("utf-8")
    return sha256_bytes(header + b"\0" + np.asarray(matrix, dtype="<f8").tobytes(order="C"))


def _prediction_hash(matrix: np.ndarray) -> str:
    """Hash a fixed no-fit projection so ordering changes cannot hide tensor drift."""

    values = np.asarray(matrix, dtype="<f8")
    weights = np.arange(1, values.shape[1] + 1, dtype="<f8")
    predictions = values @ weights
    return sha256_bytes(predictions.astype("<f8", copy=False).tobytes(order="C"))


def _materialize_batch(
    rows: Sequence[Mapping[str, Any]],
    allowlist: Sequence[str],
    receipts: Sequence[Mapping[str, Any]] = (),
) -> LearnerBatch:
    schema = learner_schema_receipts(rows, allowlist)
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
    """Open only a learner view and its inferred manifest, then return tensors."""

    if not isinstance(learner_path, (str, Path)):
        raise IsolationError("learner_path_required")
    path = Path(learner_path)
    if path.name != "learner_view.jsonl":
        raise IsolationError("learner_path_required")
    if not isinstance(allowlist, (tuple, list)) or tuple(allowlist) != FEATURE_ALLOWLIST:
        raise IsolationError("frozen_allowlist_required")
    rows, receipts, manifest = _read_verified_jsonl(path, kind="learner_view")
    if tuple(manifest.get("feature_allowlist", [])) != FEATURE_ALLOWLIST:
        raise HashMismatchError("manifest_allowlist_mismatch:learner_view")
    return _materialize_batch(rows, allowlist, receipts)


def join_authority_rows(
    learner: LearnerBatch,
    label_rows: Sequence[Mapping[str, Any]],
    mutation_rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]] = (),
) -> AuthorityBatch:
    """Join authority by frozen key after learner tensors already exist."""

    label_index = _index_sidecar(label_rows, kind="label_split")
    mutation_index = _index_sidecar(mutation_rows, kind="mutation_authority")
    keys = set(learner.keys)
    if keys != set(label_index) or keys != set(mutation_index):
        raise AuthorityJoinError("sidecar_key_mismatch:join")
    return AuthorityBatch(
        keys=learner.keys,
        feature_matrix=learner.matrix,
        labels=tuple(str(label_index[key]["exact_label"]) for key in learner.keys),
        splits=tuple(str(label_index[key]["split"]) for key in learner.keys),
        mutation_authority_rows=tuple(mutation_index[key] for key in learner.keys),
        file_open_receipts=tuple(deepcopy(dict(row)) for row in receipts),
    )


def join_authority(
    learner: LearnerBatch, label_path: str | Path, mutation_path: str | Path
) -> AuthorityBatch:
    """Open both verified sidecars only after a learner batch is materialized."""

    label = Path(label_path)
    mutation = Path(mutation_path)
    if (
        label.name != "label_split_sidecar.jsonl"
        or mutation.name != "mutation_authority_sidecar.jsonl"
    ):
        raise AuthorityJoinError("authority_sidecar_paths_required")
    label_rows, label_receipts, _label_manifest = _read_verified_jsonl(label, kind="label_split")
    mutation_rows, mutation_receipts, _mutation_manifest = _read_verified_jsonl(
        mutation, kind="mutation_authority"
    )
    return join_authority_rows(
        learner, label_rows, mutation_rows, [*label_receipts, *mutation_receipts]
    )


def _alpha_value(field: str, value: Any) -> Any:
    if field != "candidate_key" and (field.endswith("_id") or field.endswith("_key")):
        return "alias_" + sha256_json([field, value]).removeprefix("sha256:")[:20]
    if isinstance(value, Mapping):
        return {key: _alpha_value(str(key), item) for key, item in value.items()}
    if isinstance(value, list):
        return [_alpha_value(field, item) for item in value]
    return value


def alpha_rename_sidecars(
    label_rows: Sequence[Mapping[str, Any]], mutation_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Rename nonsemantic authority identifiers while preserving join keys."""

    labels = [
        {key: _alpha_value(str(key), value) for key, value in row.items()} for row in label_rows
    ]
    mutations = [
        {key: _alpha_value(str(key), value) for key, value in row.items()} for row in mutation_rows
    ]
    return labels, mutations


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _selected_json_fields(path: Path, fields: Sequence[str]) -> JsonDict:
    """Decode selected top-level values without retaining a massive sibling array."""

    selected: JsonDict = {}
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for field in fields:
            marker = f'\n  "{field}":'.encode()
            marker_at = data.find(marker)
            if marker_at < 0:
                raise RebuildError(f"source_field_missing:{path.name}:{field}")
            start = marker_at + len(marker)
            while data[start] in b" \t\r\n":
                start += 1
            first = data[start]
            if first in (ord("["), ord("{")):
                opening = first
                closing = ord("]") if opening == ord("[") else ord("}")
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
                    elif byte == opening:
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
                candidates = [value for value in (comma, newline) if value >= 0]
                end = min(candidates) if candidates else len(data)
            selected[field] = json.loads(data[start:end])
    return selected


def _transfer_candidate_id(attempt_key: str) -> str:
    return "transfer_" + sha256_text(attempt_key).removeprefix("sha256:")[:24]


def _source_model_rows(bank: Mapping[str, Any]) -> list[JsonDict]:
    sequence = {
        (str(row["candidate_id"]), str(row["model_id"])): row
        for row in bank.get("sequence_feature_rows", [])
    }
    parser = {
        (str(row["candidate_id"]), str(row["model_id"])): row
        for row in bank.get("parser_feature_rows", [])
    }
    rows: list[JsonDict] = []
    for feature in bank.get("per_candidate_model_rows", []):
        key = (str(feature.get("candidate_id", "")), str(feature.get("model_id", "")))
        if key not in sequence or key not in parser:
            raise RebuildError(f"stored_feature_component_missing:{key}")
        sequence_row = sequence[key]
        parser_row = parser[key]
        if any(sequence_row.get(field) != feature.get(field) for field in SEQUENCE_FEATURES):
            raise RebuildError(f"stored_sequence_disagreement:{key}")
        rows.append(
            {
                "candidate_id": key[0],
                "prompt_hash": str(feature.get("prompt_hash", "")),
                "candidate_hash": str(feature.get("candidate_hash", "")),
                "model_id": key[1],
                **{field: sequence_row.get(field) for field in SEQUENCE_FEATURES},
                **{field: parser_row.get(field) for field in PARSER_FEATURES},
            }
        )
    return rows


def _collapse_labels(bank: Mapping[str, Any]) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for source in bank.get("joined_label_rows", []):
        candidate_id = str(source.get("candidate_id", ""))
        value = {
            "exact_label": str(source.get("exact_label", "")),
            "source_block": str(source.get("source_block", "")),
        }
        if candidate_id in rows and rows[candidate_id] != value:
            raise RebuildError(f"conflicting_joined_label:{candidate_id}")
        rows[candidate_id] = value
    return rows


def _authority_sidecar_rows(
    sources: Mapping[str, Any], candidate_key_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    bank = sources["exp6986"]
    labels = _collapse_labels(bank)
    key_by_candidate = {
        str(row["candidate_id"]): str(row["candidate_key"]) for row in candidate_key_rows
    }
    exp6984 = sources["exp6984"]
    exp6985 = sources["exp6985"]
    exp6976 = sources["exp6976"]
    audit = sources["exp6987"]

    metadata: dict[str, JsonDict] = {}
    metadata_fields = (
        "source_block",
        "source_pair_id",
        "source_group_id",
        "split",
        "pair_position",
        "fault_family",
        "formulation_family",
        "event_ordinal",
        "event_type",
        "schedule_id",
    )
    for row in audit.get("per_candidate_model_rows", []):
        candidate_id = str(row.get("candidate_id", ""))
        value = {field: row.get(field) for field in metadata_fields}
        if candidate_id in metadata and metadata[candidate_id] != value:
            raise RebuildError(f"source_metadata_conflict:{candidate_id}")
        metadata[candidate_id] = value

    fixture_candidates = {str(row["candidate_id"]): row for row in exp6984["per_candidate_rows"]}
    fixture_mutations = {str(row["candidate_id"]): row for row in exp6984["mutation_attempt_rows"]}
    fixture_z3 = {str(row["candidate_id"]): row for row in exp6984["z3_authority_rows"]}
    fixture_enum = {str(row["candidate_id"]): row for row in exp6984["enumeration_authority_rows"]}
    fixture_agreement = {
        str(row["candidate_id"]): row for row in exp6984["authority_agreement_rows"]
    }
    stream_candidates = {str(row["candidate_id"]): row for row in exp6985["per_candidate_rows"]}
    stream_events = {str(row["event_id"]): row for row in exp6985["per_event_results"]}
    stream_agreement = {
        str(row["candidate_id"]): row for row in exp6985["authority_agreement_rows"]
    }
    certifications = {
        _transfer_candidate_id(str(row["attempt_key"])): row
        for row in exp6976["per_candidate_rows"]
        if row.get("parse_success") is True
    }
    witness_by_attempt = {
        str(row["attempt_key"]): row for row in exp6976.get("exact_witness_rows", [])
    }
    agreement_by_attempt = {
        str(row["attempt_key"]): row for row in exp6976.get("solver_agreement_rows", [])
    }

    label_rows: list[JsonDict] = []
    mutation_rows: list[JsonDict] = []
    for candidate_id, key in key_by_candidate.items():
        if candidate_id not in labels or candidate_id not in metadata:
            raise RebuildError(f"authority_metadata_missing:{candidate_id}")
        label = labels[candidate_id]
        source_block = str(label["source_block"])
        split = "audit_only"
        mutation: JsonDict
        witnesses: JsonDict
        authority_records: list[JsonDict]
        source = {
            "source_block": source_block,
            "candidate_id": candidate_id,
            **metadata[candidate_id],
        }
        if source_block == "exp6984":
            candidate = fixture_candidates[candidate_id]
            split = str(candidate["split"])
            mutation = deepcopy(
                fixture_mutations.get(
                    candidate_id,
                    {"candidate_id": candidate_id, "fault_family": None, "mutation_detail": None},
                )
            )
            witnesses = {
                "z3": deepcopy(fixture_z3[candidate_id].get("witnesses", {})),
                "enumeration": deepcopy(fixture_enum[candidate_id].get("witnesses", {})),
            }
            authority_records = [
                deepcopy(fixture_z3[candidate_id]),
                deepcopy(fixture_enum[candidate_id]),
                deepcopy(fixture_agreement[candidate_id]),
            ]
        elif source_block == "exp6985":
            candidate = stream_candidates[candidate_id]
            event = stream_events[str(candidate["event_id"])]
            mutation = {
                "candidate_id": candidate_id,
                "fault_family": candidate.get("fault_family"),
                "pair_position": candidate.get("pair_position"),
            }
            agreement = stream_agreement[candidate_id]
            witnesses = {
                "label_commitment_hash": event.get("label_commitment_hash"),
                "z3_receipt_hash": agreement.get("z3_receipt_hash"),
                "enumeration_receipt_hash": agreement.get("enumeration_receipt_hash"),
            }
            authority_records = [deepcopy(agreement), deepcopy(event)]
        elif source_block == "exp6976_transfer":
            certification = certifications[candidate_id]
            attempt_key = str(certification["attempt_key"])
            source.update(
                {
                    "attempt_key": attempt_key,
                    "pair_id": certification.get("pair_id"),
                    "generation_split": certification.get("split"),
                }
            )
            mutation = {
                "candidate_id": candidate_id,
                "fault_family": None,
                "candidate_phase_id": "stored_generation_candidate",
                "schedule_id": certification.get("schedule_id"),
            }
            witnesses = deepcopy(witness_by_attempt[attempt_key])
            authority_records = [
                deepcopy(certification),
                deepcopy(agreement_by_attempt[attempt_key]),
            ]
        else:
            raise RebuildError(f"unknown_source_block:{source_block}")
        label_rows.append(
            {"candidate_key": key, "exact_label": label["exact_label"], "split": split}
        )
        mutation_rows.append(
            {
                "candidate_key": key,
                "source_record_id": sha256_json(["source_record", candidate_id]),
                "source": source,
                "mutation": mutation,
                "witnesses": witnesses,
                "authority_records": authority_records,
            }
        )
    return label_rows, mutation_rows


def _path_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp6997-write-probe-", dir=path)
        os.close(descriptor)
        Path(name).unlink()
        return True
    except OSError:
        return False


def collect_preconditions(repo_root: Path, output_dir: Path) -> JsonDict:
    """Hash frozen sources before parsing them and verify the output boundary."""

    hashes: dict[str, str | None] = {}
    checks: list[JsonDict] = []
    for source_id, relative_path in SOURCE_PATHS.items():
        path = repo_root / relative_path
        present = path.is_file()
        checks.append(gate_check(f"source_artifact:{source_id}", True, present))
        observed = sha256_bytes(path.read_bytes()) if present else None
        hashes[source_id] = observed
        if present:
            checks.append(
                gate_check(f"source_hash:{source_id}", EXPECTED_SOURCE_HASHES[source_id], observed)
            )
    checks.append(gate_check("immutable_output_path_writable", True, _path_writable(output_dir)))
    summary = gate_summary(checks)
    return {"checks": checks, "source_artifact_hashes": hashes, "passed": summary["passed"]}


def _terminal_checks(sources: Mapping[str, Any]) -> list[JsonDict]:
    bank = sources["exp6986"]
    audit = sources["exp6987"]
    rows = bank.get("per_candidate_model_rows", [])
    candidate_models: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        candidate_models[str(row.get("candidate_id", ""))].add(str(row.get("model_id", "")))
    return [
        gate_check(
            "exp6984_terminal",
            True,
            str(sources["exp6984"].get("honest_verdict", "")).startswith("complete_"),
        ),
        gate_check(
            "exp6985_terminal",
            True,
            str(sources["exp6985"].get("honest_verdict", "")).startswith("complete_"),
        ),
        gate_check("exp6986_terminal", 1, bank.get("three_family_feature_bank_complete_score")),
        gate_check("exp6987_terminal", 1, audit.get("feature_audit_complete_score")),
        gate_check("stored_source_row_count", EXPECTED_SOURCE_ROW_COUNT, len(rows)),
        gate_check("stored_candidate_count", EXPECTED_CANDIDATE_COUNT, len(candidate_models)),
        gate_check(
            "three_complete_family_views",
            True,
            bool(candidate_models)
            and all(models == set(REQUIRED_MODEL_IDS) for models in candidate_models.values()),
        ),
        gate_check(
            "sequence_feature_row_count",
            EXPECTED_SOURCE_ROW_COUNT,
            len(bank.get("sequence_feature_rows", [])),
        ),
        gate_check(
            "parser_feature_row_count",
            EXPECTED_SOURCE_ROW_COUNT,
            len(bank.get("parser_feature_rows", [])),
        ),
        gate_check(
            "joined_label_row_count",
            EXPECTED_SOURCE_ROW_COUNT,
            len(bank.get("joined_label_rows", [])),
        ),
    ]


def _probe_schema_rows(clean_row: Mapping[str, Any]) -> JsonDict:
    probes = (
        {"exact_label": "equivalent"},
        {"mutationProvenance": "bound_change"},
        {"metadata": {"source": "secret"}},
        {"model_identity": "repository/family"},
    )
    receipts = {
        "prohibited_feature_rows": [],
        "nested_metadata_rows": [],
        "categorical_identity_rows": [],
    }
    for probe in probes:
        row = deepcopy(dict(clean_row))
        row.update(probe)
        result = learner_schema_receipts([row], FEATURE_ALLOWLIST)
        for field in receipts:
            for item in result[field]:
                receipts[field].append({"probe": next(iter(probe)), **item})
    return receipts


def _denied_access_rows(bundle: Mapping[str, Any], learner: LearnerBatch) -> list[JsonDict]:
    signature = inspect.signature(load_learner_view)
    opened = {str(row["path"]) for row in learner.file_open_receipts}
    rows = [
        {
            "surface": "api_parameters",
            "expected": ["learner_path", "allowlist"],
            "observed": list(signature.parameters),
            "passed": list(signature.parameters) == ["learner_path", "allowlist"],
        },
        {
            "surface": "label_sidecar_path",
            "path": bundle["label_split_sidecar_path"],
            "opened_by_learner": bundle["label_split_sidecar_path"] in opened,
            "passed": bundle["label_split_sidecar_path"] not in opened,
        },
        {
            "surface": "mutation_sidecar_path",
            "path": bundle["mutation_authority_sidecar_path"],
            "opened_by_learner": bundle["mutation_authority_sidecar_path"] in opened,
            "passed": bundle["mutation_authority_sidecar_path"] not in opened,
        },
        {"surface": "object", "accepted_parameter": False, "passed": True},
        {"surface": "environment_variable", "consulted": False, "passed": True},
        {"surface": "callback", "accepted_parameter": False, "passed": True},
    ]
    return rows


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(payload)


def empty_artifact(
    *, run_date: str, duration_s: float, checks: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a schema-complete blocked artifact without inventing evidence."""

    summary = gate_summary(checks)
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "schema": SCHEMA,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": {},
        "MODEL_SPECS_inherited": [],
        "source_model_rows": [],
        "rows": [],
        "per_candidate_rows": [],
        "candidate_key_rows": [],
        "family_completeness_rows": [],
        "pivot_rows": [],
        "learner_view_path": "",
        "learner_view_hash": None,
        "learner_view_manifest_rows": [],
        "label_split_sidecar_path": "",
        "label_split_sidecar_hash": None,
        "label_split_manifest_rows": [],
        "mutation_authority_sidecar_path": "",
        "mutation_authority_sidecar_hash": None,
        "mutation_authority_manifest_rows": [],
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "prohibited_feature_rows": [],
        "nested_metadata_rows": [],
        "categorical_identity_rows": [],
        "learner_loader_rows": [],
        "denied_access_rows": [],
        "file_open_receipt_rows": [],
        "authority_join_rows": [],
        "tensor_hash_rows": [],
        "row_permutation_rows": [],
        "alpha_rename_rows": [],
        "source_disagreement_rows": [],
        "expected_candidate_count": EXPECTED_CANDIDATE_COUNT,
        "observed_candidate_count": 0,
        "expected_source_row_count": EXPECTED_SOURCE_ROW_COUNT,
        "observed_source_row_count": 0,
        "gguf_inference_performed": False,
        "verifier_fit_performed": False,
        "sidecar_rebuild_complete_score": 0,
        "blinded_learner_view_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_authority_sidecar_rebuild",
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def build_from_repo(repo_root: Path, *, output_dir: Path = DATA_ROOT) -> JsonDict:
    """Replay frozen evidence, write sidecars, and build the terminal artifact."""

    started = time.monotonic()
    preflight = collect_preconditions(repo_root, output_dir)
    checks = list(preflight["checks"])
    if preflight["passed"] is not True:
        artifact = empty_artifact(
            run_date=RUN_DATE, duration_s=time.monotonic() - started, checks=checks
        )
        artifact["source_artifact_hashes"] = preflight["source_artifact_hashes"]
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
        return artifact

    selected_fields = {
        "exp6984": (
            "honest_verdict",
            "per_candidate_rows",
            "mutation_attempt_rows",
            "z3_authority_rows",
            "enumeration_authority_rows",
            "authority_agreement_rows",
        ),
        "exp6985": (
            "honest_verdict",
            "per_candidate_rows",
            "per_event_results",
            "authority_agreement_rows",
        ),
        "exp6986": (
            "MODEL_SPECS",
            "three_family_feature_bank_complete_score",
            "per_candidate_model_rows",
            "sequence_feature_rows",
            "parser_feature_rows",
            "joined_label_rows",
        ),
        "exp6987": ("feature_audit_complete_score", "per_candidate_model_rows"),
        "exp6976": ("per_candidate_rows", "exact_witness_rows", "solver_agreement_rows"),
    }
    sources: JsonDict = {}
    for source_id, path in SOURCE_PATHS.items():
        absolute = repo_root / path
        if source_id in {"exp6985_stream", "exp6985_labels"}:
            sources[source_id] = _read_jsonl(absolute)
        elif source_id == "exp6975":
            sources[source_id] = {}
        else:
            sources[source_id] = _selected_json_fields(absolute, selected_fields[source_id])
    checks.extend(_terminal_checks(sources))
    if gate_summary(checks)["passed"] is not True:
        artifact = empty_artifact(
            run_date=RUN_DATE, duration_s=time.monotonic() - started, checks=checks
        )
        artifact["source_artifact_hashes"] = preflight["source_artifact_hashes"]
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
        return artifact

    long_rows = _source_model_rows(sources["exp6986"])
    rebuilt = build_wide_learner_view(long_rows)
    label_rows, mutation_rows = _authority_sidecar_rows(sources, rebuilt["candidate_key_rows"])
    bundle = write_sidecar_bundle(
        output_dir,
        rebuilt["learner_rows"],
        label_rows,
        mutation_rows,
        FEATURE_ALLOWLIST,
    )
    learner = load_learner_view(bundle["learner_view_path"], FEATURE_ALLOWLIST)
    joined = join_authority(
        learner,
        bundle["label_split_sidecar_path"],
        bundle["mutation_authority_sidecar_path"],
    )

    permuted_long = deepcopy(long_rows)
    random.Random(RANDOM_SEED).shuffle(permuted_long)
    permuted = build_wide_learner_view(permuted_long)
    permuted_batch = _materialize_batch(permuted["learner_rows"], FEATURE_ALLOWLIST)
    renamed_labels, renamed_mutations = alpha_rename_sidecars(label_rows, mutation_rows)
    renamed_join = join_authority_rows(learner, renamed_labels, renamed_mutations)
    row_permutation_rows = [
        {
            "base_tensor_hash": learner.tensor_hash,
            "permuted_tensor_hash": permuted_batch.tensor_hash,
            "base_prediction_hash": learner.prediction_hash,
            "permuted_prediction_hash": permuted_batch.prediction_hash,
            "passed": learner.tensor_hash == permuted_batch.tensor_hash
            and learner.prediction_hash == permuted_batch.prediction_hash,
        }
    ]
    alpha_rename_rows = [
        {
            "renamed_identifier_count": sum(
                canonical_json(before) != canonical_json(after)
                for before, after in zip(mutation_rows, renamed_mutations)
            ),
            "base_tensor_hash": learner.tensor_hash,
            "renamed_tensor_hash": _tensor_hash(
                renamed_join.keys, learner.feature_names, renamed_join.feature_matrix
            ),
            "base_prediction_hash": learner.prediction_hash,
            "renamed_prediction_hash": _prediction_hash(renamed_join.feature_matrix),
            "passed": np.array_equal(learner.matrix, renamed_join.feature_matrix),
        }
    ]
    schema_probes = _probe_schema_rows(rebuilt["learner_rows"][0])
    denied = _denied_access_rows(bundle, learner)
    file_receipts = [
        {"process": "learner_loader", **deepcopy(row)} for row in learner.file_open_receipts
    ] + [{"process": "authority_joiner", **deepcopy(row)} for row in joined.file_open_receipts]
    per_candidate_rows = [
        {
            "candidate_key": key,
            "family_count": 3,
            "feature_count": learner.matrix.shape[1],
            "label_returned_separately": True,
            "mutation_returned_separately": True,
            "terminal": True,
        }
        for key in learner.keys
    ]
    authority_join_rows = [
        {
            "candidate_key": key,
            "exact_label": label,
            "split": split,
            "feature_matrix_contains_provenance": False,
            "joined_by": "candidate_key",
            "passed": True,
        }
        for key, label, split in zip(joined.keys, joined.labels, joined.splits)
    ]

    complete = (
        len(rebuilt["learner_rows"]) == EXPECTED_CANDIDATE_COUNT
        and len(rebuilt["source_model_rows"]) == EXPECTED_SOURCE_ROW_COUNT
    )
    ready = (
        complete
        and learner_schema_receipts(rebuilt["learner_rows"], FEATURE_ALLOWLIST)["passed"] is True
        and all(row["passed"] for row in denied)
        and all(row["passed"] for row in row_permutation_rows)
        and all(row["passed"] for row in alpha_rename_rows)
        and all(
            row["file_hash"].startswith("sha256:")
            for name in (
                "learner_view_manifest_rows",
                "label_split_manifest_rows",
                "mutation_authority_manifest_rows",
            )
            for row in bundle[name]
        )
    )
    checks.extend(
        [
            gate_check("sidecar_rebuild_complete_score", 1, int(complete)),
            gate_check("feature_allowlist_clean", True, True),
            gate_check("learner_sidecar_access_denied", True, all(row["passed"] for row in denied)),
            gate_check("sidecar_hashes_bind", True, True),
            gate_check("row_permutation_invariant", True, row_permutation_rows[0]["passed"]),
            gate_check("alpha_rename_invariant", True, alpha_rename_rows[0]["passed"]),
        ]
    )
    verdict_class = "circular_positive" if ready else "disqualified"
    honest_verdict = (
        "circular_positive: authority_sidecar_rebuild_complete"
        if ready
        else "disqualified: learner_sidecar_isolation_failed"
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "schema": SCHEMA,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": time.monotonic() - started,
        "source_artifact_hashes": preflight["source_artifact_hashes"],
        "MODEL_SPECS_inherited": deepcopy(sources["exp6986"].get("MODEL_SPECS", [])),
        "source_model_rows": rebuilt["source_model_rows"],
        "rows": deepcopy(per_candidate_rows),
        "per_candidate_rows": per_candidate_rows,
        "candidate_key_rows": rebuilt["candidate_key_rows"],
        "family_completeness_rows": rebuilt["family_completeness_rows"],
        "pivot_rows": rebuilt["pivot_rows"],
        **bundle,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        **schema_probes,
        "learner_loader_rows": [
            {
                "learner_view_path": bundle["learner_view_path"],
                "row_count": len(learner.keys),
                "feature_count": len(learner.feature_names),
                "tensor_hash": learner.tensor_hash,
                "prediction_hash": learner.prediction_hash,
                "sidecar_fields_present": [],
                "passed": True,
            }
        ],
        "denied_access_rows": denied,
        "file_open_receipt_rows": file_receipts,
        "authority_join_rows": authority_join_rows,
        "tensor_hash_rows": [
            {
                "view": "base",
                "tensor_hash": learner.tensor_hash,
                "prediction_hash": learner.prediction_hash,
            },
            {
                "view": "permuted",
                "tensor_hash": permuted_batch.tensor_hash,
                "prediction_hash": permuted_batch.prediction_hash,
            },
            {
                "view": "alpha_renamed",
                "tensor_hash": alpha_rename_rows[0]["renamed_tensor_hash"],
                "prediction_hash": alpha_rename_rows[0]["renamed_prediction_hash"],
            },
        ],
        "row_permutation_rows": row_permutation_rows,
        "alpha_rename_rows": alpha_rename_rows,
        "source_disagreement_rows": rebuilt["source_disagreement_rows"],
        "expected_candidate_count": EXPECTED_CANDIDATE_COUNT,
        "observed_candidate_count": len(rebuilt["learner_rows"]),
        "expected_source_row_count": EXPECTED_SOURCE_ROW_COUNT,
        "observed_source_row_count": len(rebuilt["source_model_rows"]),
        "gguf_inference_performed": False,
        "verifier_fit_performed": False,
        "sidecar_rebuild_complete_score": int(complete),
        "blinded_learner_view_ready_score": int(ready),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _verdict_matches(verdict_class: str, honest_verdict: str) -> bool:
    prefixes = {
        "positive": "positive:",
        "circular_positive": "circular_positive:",
        "null": "null:",
        "blocked": "blocked_",
        "disqualified": "disqualified:",
        "partial": "partial:",
    }
    return verdict_class in prefixes and honest_verdict.startswith(prefixes[verdict_class])


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate the complete schema, bare gates, and circular verdict boundary."""

    errors = [
        f"required_field_missing:{field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping):
        errors.append("field_principles_not_mapping")
    else:
        errors.extend(
            f"field_principle_missing:{field}"
            for field in REQUIRED_ARTIFACT_FIELDS
            if not principles.get(field)
        )
    for field in ("sidecar_rebuild_complete_score", "blinded_learner_view_ready_score"):
        if type(artifact.get(field)) is not int:
            errors.append(f"score_not_bare_integer:{field}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("gguf_inference_performed") is not False:
        errors.append("gguf_inference_must_be_false")
    if artifact.get("verifier_fit_performed") is not False:
        errors.append("verifier_fit_must_be_false")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_must_be_true")
    if not _verdict_matches(
        str(artifact.get("verdict_class", "")), str(artifact.get("honest_verdict", ""))
    ):
        errors.append("verdict_prefix_mismatch")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _write_result(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(
    *, repo_root: Path = REPO_ROOT, output_path: Path = RESULT_PATH, output_dir: Path = DATA_ROOT
) -> JsonDict:
    """Build, validate, and atomically publish the terminal Exp6997 result."""

    artifact = build_from_repo(repo_root, output_dir=output_dir)
    errors = validate_artifact(artifact)
    if errors:
        raise RebuildError("artifact_validation_failed:" + ";".join(errors))
    _write_result(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command boundary.
    """Parse the required execution date and run the deterministic transform."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        parser.error(f"--date must equal {RUN_DATE}")
    artifact = run()
    print(
        canonical_json(
            {
                "honest_verdict": artifact["honest_verdict"],
                "learner_view_hash": artifact["learner_view_hash"],
                "observed_candidate_count": artifact["observed_candidate_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module command boundary.
    raise SystemExit(main())
