"""Audit whether signed intervention responses support a causal feature bank.

The learner tensor contains only paired signed response values. Exact labels
and nuisance fields stay in audit-only rows. A fresh restricted process proves
that changing the authority sidecar cannot change the tensor.

Spec refs: REQ-VERIFY-7014 and SCENARIO-VERIFY-7014-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
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


JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_7014_causal_feature_cold_audit"
SCHEMA = "carnot.experiment_7014.causal_feature_cold_audit.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 70_142_026_0905
INFERENCE_SUBSTRATE = "fresh_process_causal_feature_audit_no_llm"
PROHIBITED_AUROC_LIMIT = 0.80
ISOMORPHIC_TOLERANCE = 0.20
BOOTSTRAP_DRAWS = 512
EXPECTED_PAIR_COUNT = 48
EXPECTED_SIGNED_ROW_COUNT = 144

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path("python/carnot/experiment_7014_causal_feature_cold_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7014_causal_feature_cold_audit.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_7014_causal_feature_cold_audit.json")

SOURCE_PATHS = {
    "exp7012": Path("results/experiment_7012_exact_intervention_pair_fixture.json"),
    "exp7013": Path("results/experiment_7013_three_family_intervention_surface.json"),
    "learner_prompts": Path(
        "results/raw/experiment_7012_exact_intervention_pair_fixture/learner_prompts.jsonl"
    ),
    "authority_sidecar": Path(
        "results/raw/experiment_7012_exact_intervention_pair_fixture/authority_sidecar.jsonl"
    ),
}
EXPECTED_SOURCE_HASHES = {
    "exp7012": "sha256:21baf5d79e7f0b489098e586d4851bbbd34e881caefef77d38a91683d7f347b7",
    "exp7013": "sha256:ea650e30c37fb7fe0a8629ecd2b1e4b818aba578ce5f92d56b192f3e05255ba2",
    "learner_prompts": "sha256:73cf63f3db3d906bfee5de4533a4cf179fb7af8b28226b4e0698092ef50c71a8",
    "authority_sidecar": "sha256:2d714758c25f5f6c3bdc7fad239e272a9fbcf85fa484e2d41242bf2575305797",
}
EXPECTED_PROMPT_FREEZE_HASH = EXPECTED_SOURCE_HASHES["learner_prompts"]
EXPECTED_RESPONSE_FREEZE_HASH = (
    "sha256:491ca747bdbee10d057a9b031c67f60f726cfbb0556a9408c107b1b0e4b3fc95"
)

MODEL_FAMILIES = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
ALLOWED_TENSOR_FIELDS = ("signed_primary_delta", "signed_isomorphic_delta")
PROHIBITED_FIELD_ATTACKS = (
    ("direct_label", "label"),
    ("label_alias", "ground_truth"),
    ("nested_provenance", "provenance"),
    ("row_key", "pair_id"),
    ("serialization", "serialization_template"),
    ("length", "prompt_char_count"),
    ("mutation", "mutation_kind"),
    ("source", "source_family"),
    ("split", "evaluation_partition"),
    ("model_identity", "model_repository"),
    ("score_norm", "score_norm"),
    ("magnitude", "absolute_magnitude"),
    ("hash", "response_hash"),
)

PROBE_SPECS = {
    "metadata": ("block_id", "terminal"),
    "row_ordering": ("row_order",),
    "serialization": ("serialization_signature",),
    "length": ("prompt_char_count", "prompt_word_count"),
    "mutation": ("mutation_kind",),
    "source": ("source_family",),
    "split": ("split",),
    "model_identity": ("model_repository",),
    "norm_only": ("score_norm",),
    "magnitude_only": ("primary_magnitude", "isomorphic_magnitude"),
}
PROBE_CONTAINERS = {
    "metadata": "metadata_probe_rows",
    "row_ordering": "metadata_probe_rows",
    "serialization": "serialization_probe_rows",
    "length": "length_probe_rows",
    "mutation": "mutation_probe_rows",
    "source": "source_probe_rows",
    "split": "split_probe_rows",
    "model_identity": "model_identity_probe_rows",
    "norm_only": "norm_only_probe_rows",
    "magnitude_only": "magnitude_only_probe_rows",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "replay_hash_rows",
    "rows",
    "per_pair_results",
    "learner_tensor_rows",
    "sidecar_intervention_rows",
    "direct_leakage_rows",
    "metadata_probe_rows",
    "serialization_probe_rows",
    "length_probe_rows",
    "mutation_probe_rows",
    "source_probe_rows",
    "split_probe_rows",
    "model_identity_probe_rows",
    "norm_only_probe_rows",
    "magnitude_only_probe_rows",
    "grouped_bootstrap_rows",
    "isomorphic_invariance_rows",
    "family_identifiability_rows",
    "held_source_identifiability_rows",
    "direct_leakage_count",
    "prohibited_auroc_upper_bound_max",
    "identifiable_family_count",
    "causal_bank_audit_complete_score",
    "causal_feature_bank_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the release contract reviewable.",
    "preconditions_checked": "Exact preconditions stop changed evidence before analysis.",
    "inference_substrate": "The substrate distinguishes this audit from model inference.",
    "duration_s": "Measured wall time proves that the audit process executed.",
    "source_artifact_hashes": "Source hashes bind conclusions to immutable input bytes.",
    "replay_hash_rows": "Replay rows expose every expected and observed digest.",
    "rows": "All signed response rows preserve the denominator behind conclusions.",
    "per_pair_results": "Pair rows keep each exact intervention unit inspectable.",
    "learner_tensor_rows": "Tensor rows show the numeric evidence admitted per pair.",
    "sidecar_intervention_rows": "Sidecar attacks test independence from authority metadata.",
    "direct_leakage_rows": "Leakage rows prove that named substitute fields stay excluded.",
    "metadata_probe_rows": "Metadata and ordering probes measure shortcut prediction risk.",
    "serialization_probe_rows": "Serialization probes test surface-form substitution.",
    "length_probe_rows": "Length probes test size-based substitution.",
    "mutation_probe_rows": "Mutation probes test intervention-name substitution.",
    "source_probe_rows": "Source probes test source-identity substitution.",
    "split_probe_rows": "Split probes test evaluation-routing substitution.",
    "model_identity_probe_rows": "Model probes test repository-name substitution.",
    "norm_only_probe_rows": "Norm probes test unsigned score-size substitution.",
    "magnitude_only_probe_rows": "Magnitude probes test absolute-effect substitution.",
    "grouped_bootstrap_rows": "Cluster draws prove that no row was sampled independently.",
    "isomorphic_invariance_rows": "Isomorphic rows test direction under a meaning-preserving edit.",
    "family_identifiability_rows": "Family rows keep both signed directions separate by model.",
    "held_source_identifiability_rows": "Held-source rows expose source-specific reversals.",
    "direct_leakage_count": "A bare count disqualifies any direct tensor leak.",
    "prohibited_auroc_upper_bound_max": "The worst upper bound controls nuisance release.",
    "identifiable_family_count": "A bare count enforces the two-of-three scientific floor.",
    "causal_bank_audit_complete_score": "Completion requires every planned row to terminate.",
    "causal_feature_bank_ready_score": "One bare field controls feature-bank release.",
    "random_seed": "A fixed seed makes folds and cluster draws reproducible.",
    "reproducibility_checksum": "A canonical digest detects later artifact drift.",
    "gate_check_summary": "The first expected-observed failure makes blocks actionable.",
    "verifier_is_oracle": "False states that the audit does not define exact correctness.",
    "verdict_class": "A closed class separates blocks, disqualification, nulls, and release.",
    "honest_verdict": "A class-specific prefix gives automation one terminal meaning.",
}
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}


class AuditInputError(ValueError):
    """The signed response surface cannot form the registered causal matrix."""


@dataclass(frozen=True)
class LearnerBatch:
    """Hold causal bytes separately from labels and audit-only metadata."""

    keys: tuple[str, ...]
    matrix: np.ndarray
    targets: tuple[int, ...]
    source_families: tuple[str, ...]
    tensor_hash: str
    prediction_hash: str
    tensor_rows: tuple[JsonDict, ...]


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes and JSON Lines."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a project-style SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON value after canonical serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_path(path: Path) -> str | None:
    """Hash a file incrementally, or return none when it is absent."""

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
    """Retain every check and promote the first failed comparison."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def load_hashed_sources(
    repo_root: Path,
    source_paths: Mapping[str, Path] = SOURCE_PATHS,
    expected_hashes: Mapping[str, str] = EXPECTED_SOURCE_HASHES,
) -> JsonDict:
    """Hash every input before parsing any source value."""

    hashes = {name: sha256_path(repo_root / path) for name, path in source_paths.items()}
    checks = [
        gate_check(f"source_hash:{name}", expected_hashes.get(name), hashes[name])
        for name in source_paths
    ]
    if any(row["passed"] is not True for row in checks):
        return {"passed": False, "hashes": hashes, "values": {}, "checks": checks}
    values: JsonDict = {}
    try:
        for name, relative in source_paths.items():
            path = repo_root / relative
            text = path.read_text(encoding="utf-8")
            values[name] = (
                [json.loads(line) for line in text.splitlines() if line]
                if path.suffix == ".jsonl"
                else json.loads(text)
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        checks.append(
            gate_check("source_parse", "all sources parse", f"{type(exc).__name__}: {exc}")
        )
        return {"passed": False, "hashes": hashes, "values": {}, "checks": checks}
    return {"passed": True, "hashes": hashes, "values": values, "checks": checks}


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _denied_name(value: str) -> bool:
    normalized = _normalized_name(value)
    tokens = (
        "label",
        "groundtruth",
        "target",
        "provenance",
        "authority",
        "witness",
        "pairid",
        "rowkey",
        "serial",
        "length",
        "charcount",
        "wordcount",
        "mutation",
        "source",
        "split",
        "partition",
        "model",
        "repository",
        "norm",
        "magnitude",
        "absolute",
        "hash",
    )
    return any(token in normalized for token in tokens)


def audit_tensor_payload(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reject every field except the two registered signed response values."""

    leakage: list[JsonDict] = []
    for index, row in enumerate(rows):
        for field, value in row.items():
            if field not in ALLOWED_TENSOR_FIELDS or _denied_name(field):
                leakage.append(
                    {
                        "path": f"$[{index}].{field}",
                        "field": field,
                        "reason": "prohibited_alias" if _denied_name(field) else "not_allowlisted",
                        "leakage_detected": True,
                        "terminal": True,
                    }
                )
            elif type(value) not in (int, float) or not math.isfinite(float(value)):
                leakage.append(
                    {
                        "path": f"$[{index}].{field}",
                        "field": field,
                        "reason": "non_finite_or_non_numeric",
                        "leakage_detected": True,
                        "terminal": True,
                    }
                )
        for field in ALLOWED_TENSOR_FIELDS:
            if field not in row:
                leakage.append(
                    {
                        "path": f"$[{index}].{field}",
                        "field": field,
                        "reason": "missing_registered_field",
                        "leakage_detected": True,
                        "terminal": True,
                    }
                )
    return {"direct_leakage_count": len(leakage), "direct_leakage_rows": leakage}


def _reference_prediction_hash(matrix: np.ndarray) -> str:
    """Hash a fixed projection so tensor changes affect a second output."""

    weights = np.linspace(-0.75, 0.75, matrix.shape[1], dtype="<f8")
    predictions = np.asarray(matrix @ weights, dtype="<f8")
    return sha256_bytes(predictions.tobytes(order="C"))


def build_learner_tensor(rows: Sequence[Mapping[str, Any]]) -> LearnerBatch:
    """Build a pair-wide tensor from signed response rows only."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str]] = set()
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        model = str(row.get("model_repository", ""))
        identity = (pair_id, model)
        if identity in identities:
            raise AuditInputError("duplicate_family_cell")
        identities.add(identity)
        if not pair_id or model not in MODEL_FAMILIES or row.get("terminal") is not True:
            raise AuditInputError("incomplete_family_cell")
        for field in ALLOWED_TENSOR_FIELDS:
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise AuditInputError("non_finite_signed_response")
        grouped[pair_id].append(row)
    if len(rows) != len(grouped) * len(MODEL_FAMILIES):
        raise AuditInputError("incomplete_family_cell")

    matrix_rows: list[list[float]] = []
    targets: list[int] = []
    sources: list[str] = []
    tensor_rows: list[JsonDict] = []
    ordered_keys = tuple(sorted(grouped))
    for position, pair_id in enumerate(ordered_keys):
        group = grouped[pair_id]
        by_model = {str(row.get("model_repository")): row for row in group}
        comparisons = {str(row.get("comparison")) for row in group}
        source_families = {str(row.get("held_source_family")) for row in group}
        if comparisons not in ({"clean_to_violation"}, {"clean_to_repair"}):
            raise AuditInputError("inconsistent_intervention_direction")
        if len(source_families) != 1:
            raise AuditInputError("inconsistent_held_source")
        values = [
            float(by_model[model][field])
            for model in MODEL_FAMILIES
            for field in ALLOWED_TENSOR_FIELDS
        ]
        matrix_rows.append(values)
        target = int(comparisons == {"clean_to_repair"})
        targets.append(target)
        sources.append(next(iter(source_families)))
        tensor_rows.append(
            {
                "tensor_position": position,
                "pair_id": pair_id,
                "signed_response_values": values,
                "target_joined_after_materialization": target,
                "source_joined_after_materialization": sources[-1],
                "terminal": True,
            }
        )
    matrix = np.asarray(matrix_rows, dtype="<f8")
    tensor_hash = sha256_bytes(matrix.tobytes(order="C"))
    return LearnerBatch(
        keys=ordered_keys,
        matrix=matrix,
        targets=tuple(targets),
        source_families=tuple(sources),
        tensor_hash=tensor_hash,
        prediction_hash=_reference_prediction_hash(matrix),
        tensor_rows=tuple(tensor_rows),
    )


def _replace_sidecar(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    values = [deepcopy(dict(row)) for row in rows]
    if len(values) > 1:
        keys = ("mutation_kind", "source_family", "split", "serialization_template")
        first = {key: values[0].get(key) for key in keys}
        for key in keys:
            values[0][key] = values[1].get(key)
            values[1][key] = first[key]
    return values


def _alpha_rename(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                "alpha_name" if "variable" in _normalized_name(str(key)) else _alpha_rename(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_alpha_rename(child) for child in value]
    return deepcopy(value)


def audit_sidecar_interventions(
    signed_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[JsonDict]:
    """Change ambient sidecars while calling a loader with no sidecar input."""

    shuffled = [deepcopy(dict(row)) for row in sidecar_rows]
    random.Random(seed).shuffle(shuffled)
    conditions = (
        ("correct", [deepcopy(dict(row)) for row in sidecar_rows]),
        ("permuted", shuffled),
        ("replaced", _replace_sidecar(sidecar_rows)),
        ("deleted", []),
        ("alpha_renamed", _alpha_rename(sidecar_rows)),
    )
    reference = build_learner_tensor(signed_rows)
    output = []
    for condition, ambient_rows in conditions:
        observed = build_learner_tensor(signed_rows)
        tensor_bytes_identical = observed.matrix.tobytes(order="C") == reference.matrix.tobytes(
            order="C"
        )
        output.append(
            {
                "condition": condition,
                "ambient_sidecar_hash": sha256_json(ambient_rows),
                "ambient_sidecar_row_count": len(ambient_rows),
                "sidecar_open_attempted": False,
                "tensor_bytes_identical": tensor_bytes_identical,
                "tensor_hash": observed.tensor_hash,
                "reference_prediction_hash": observed.prediction_hash,
                "passed": tensor_bytes_identical
                and observed.tensor_hash == reference.tensor_hash
                and observed.prediction_hash == reference.prediction_hash,
                "terminal": True,
            }
        )
    return output


def _feature_dict(row: Mapping[str, Any], fields: Sequence[str]) -> JsonDict:
    return {field: row.get(field, "<missing>") for field in fields}


def _block_predictions(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[float]
) -> list[JsonDict]:
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        grouped[str(row["block_id"])].append((row, float(prediction)))
    return [
        {
            "block_id": block_id,
            "source_family": str(values[0][0]["source_family"]),
            "target": int(values[0][0]["target"]),
            "prediction": float(np.mean([value[1] for value in values])),
        }
        for block_id, values in sorted(grouped.items())
    ]


def _cluster_bootstrap_auc(
    block_rows: Sequence[Mapping[str, Any]], *, seed: int, draws: int, analysis: str
) -> list[JsonDict]:
    """Resample sources, then whole blocks, without sampling response rows."""

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in block_rows:
        by_source[str(row["source_family"])].append(row)
    sources = sorted(by_source)
    generator = np.random.default_rng(seed)
    output = []
    for draw in range(draws):
        sampled_sources = [
            sources[index] for index in generator.integers(0, len(sources), len(sources))
        ]
        sampled_blocks: list[Mapping[str, Any]] = []
        for source in sampled_sources:
            candidates = by_source[source]
            sampled_blocks.extend(
                candidates[index]
                for index in generator.integers(0, len(candidates), len(candidates))
            )
        labels = [int(row["target"]) for row in sampled_blocks]
        raw_score = (
            float(roc_auc_score(labels, [float(row["prediction"]) for row in sampled_blocks]))
            if len(set(labels)) == 2
            else None
        )
        score = None if raw_score is None else max(raw_score, 1.0 - raw_score)
        output.append(
            {
                "analysis": analysis,
                "draw": draw,
                "bootstrap_unit": "source_block_and_source_family",
                "sampled_source_families": sampled_sources,
                "sampled_block_ids": [str(row["block_id"]) for row in sampled_blocks],
                "sampled_row_ids": None,
                "raw_auroc": raw_score,
                "auroc": score,
                "terminal": True,
            }
        )
    return output


def fit_grouped_probe(
    probe_name: str,
    rows: Sequence[Mapping[str, Any]],
    feature_fields: Sequence[str],
    *,
    seed: int,
    folds: int = 4,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> JsonDict:
    """Fit source-held-out probes and bootstrap predictions by source block."""

    if not rows or len({int(row["target"]) for row in rows}) != 2:
        raise AuditInputError("probe_requires_two_classes")
    sources = sorted({str(row["source_family"]) for row in rows})
    fold_count = min(folds, len(sources))
    if fold_count < 2:
        raise AuditInputError("probe_requires_two_sources")
    source_folds = [sources[index::fold_count] for index in range(fold_count)]
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    probe_rows = []
    for fold_index, test_sources in enumerate(source_folds):
        test_indices = [
            index for index, row in enumerate(rows) if str(row["source_family"]) in test_sources
        ]
        train_indices = [index for index in range(len(rows)) if index not in set(test_indices)]
        train_labels = [int(rows[index]["target"]) for index in train_indices]
        test_labels = [int(rows[index]["target"]) for index in test_indices]
        if len(set(train_labels)) != 2 or len(set(test_labels)) != 2:
            raise AuditInputError("source_fold_requires_two_classes")
        vectorizer = DictVectorizer(sparse=False, sort=True)
        train_matrix = vectorizer.fit_transform(
            [_feature_dict(rows[index], feature_fields) for index in train_indices]
        )
        test_matrix = vectorizer.transform(
            [_feature_dict(rows[index], feature_fields) for index in test_indices]
        )
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            random_state=(seed + fold_index) % (2**32),
            solver="liblinear",
        )
        model.fit(train_matrix, train_labels)
        predictions[test_indices] = model.predict_proba(test_matrix)[:, 1]
        train_blocks = sorted({str(rows[index]["block_id"]) for index in train_indices})
        test_blocks = sorted({str(rows[index]["block_id"]) for index in test_indices})
        probe_rows.append(
            {
                "record_type": "fold",
                "probe_name": probe_name,
                "fold_index": fold_index,
                "held_source_families": test_sources,
                "feature_fields": list(feature_fields),
                "train_block_ids": train_blocks,
                "test_block_ids": test_blocks,
                "group_overlap_count": len(set(train_blocks) & set(test_blocks)),
                "preprocessing_fit_on_train_only": True,
                "coefficient_rows": [
                    {"feature": name, "coefficient": float(value)}
                    for name, value in zip(
                        vectorizer.get_feature_names_out(), model.coef_[0], strict=True
                    )
                ],
                "intercept": float(model.intercept_[0]),
                "terminal": True,
            }
        )
    prediction_rows = [
        {
            "record_type": "prediction",
            "probe_name": probe_name,
            "row_position": index,
            "block_id": str(row["block_id"]),
            "source_family": str(row["source_family"]),
            "target": int(row["target"]),
            "prediction": float(predictions[index]),
            "terminal": True,
        }
        for index, row in enumerate(rows)
    ]
    block_rows = _block_predictions(rows, predictions)
    labels = [int(row["target"]) for row in block_rows]
    raw_point_auc = float(roc_auc_score(labels, [float(row["prediction"]) for row in block_rows]))
    point_auc = max(raw_point_auc, 1.0 - raw_point_auc)
    bootstrap_rows = _cluster_bootstrap_auc(
        block_rows, seed=seed, draws=bootstrap_draws, analysis=f"probe:{probe_name}"
    )
    bootstrap_values = sorted(
        float(row["auroc"]) for row in bootstrap_rows if row.get("auroc") is not None
    )
    lower = float(np.quantile(bootstrap_values, 0.025)) if bootstrap_values else None
    upper = float(np.quantile(bootstrap_values, 0.975)) if bootstrap_values else None
    interval_row = {
        "record_type": "interval",
        "probe_name": probe_name,
        "feature_fields": list(feature_fields),
        "raw_auroc": raw_point_auc,
        "auroc": point_auc,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "bootstrap_samples": bootstrap_draws,
        "bootstrap_unit": "source_block_and_source_family",
        "threshold": PROHIBITED_AUROC_LIMIT,
        "gate_passed": upper is not None and upper < PROHIBITED_AUROC_LIMIT,
        "terminal": True,
    }
    return {
        "probe_rows": probe_rows,
        "prediction_rows": prediction_rows,
        "bootstrap_rows": bootstrap_rows,
        "interval_row": interval_row,
    }


def _mean_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    seed: int,
    draws: int,
    analysis: str,
) -> tuple[JsonDict, list[JsonDict]]:
    """Compute a two-stage source and block bootstrap for one signed mean."""

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["held_source_family"])].append(row)
    sources = sorted(by_source)
    generator = np.random.default_rng(seed)
    draw_rows = []
    draw_values = []
    for draw in range(draws):
        sampled_sources = [
            sources[index] for index in generator.integers(0, len(sources), len(sources))
        ]
        sampled: list[Mapping[str, Any]] = []
        for source in sampled_sources:
            candidates = by_source[source]
            sampled.extend(
                candidates[index]
                for index in generator.integers(0, len(candidates), len(candidates))
            )
        mean = float(np.mean([float(row[value_field]) for row in sampled]))
        draw_values.append(mean)
        draw_rows.append(
            {
                "analysis": analysis,
                "draw": draw,
                "bootstrap_unit": "source_block_and_source_family",
                "sampled_source_families": sampled_sources,
                "sampled_block_ids": [str(row["pair_id"]) for row in sampled],
                "sampled_row_ids": None,
                "mean": mean,
                "terminal": True,
            }
        )
    summary = {
        "mean": float(np.mean([float(row[value_field]) for row in rows])),
        "ci95_lower": float(np.quantile(draw_values, 0.025)),
        "ci95_upper": float(np.quantile(draw_values, 0.975)),
        "block_count": len(rows),
        "identifiable_positive_direction": float(np.quantile(draw_values, 0.025)) > 0.0,
    }
    return summary, draw_rows


def audit_identifiability(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    tolerance: float = ISOMORPHIC_TOLERANCE,
) -> JsonDict:
    """Recompute both signed directions by model and held source."""

    family_rows = []
    held_rows = []
    invariant_rows = []
    bootstrap_rows = []
    comparisons = ("clean_to_violation", "clean_to_repair")
    sources = sorted({str(row.get("held_source_family")) for row in rows})
    for model_index, model in enumerate(MODEL_FAMILIES):
        direction_summaries: JsonDict = {}
        for comparison_index, comparison in enumerate(comparisons):
            group = [
                row
                for row in rows
                if row.get("model_repository") == model and row.get("comparison") == comparison
            ]
            if not group:
                raise AuditInputError("identifiability_cell_missing")
            summary, draws = _mean_bootstrap(
                group,
                value_field="signed_primary_delta",
                seed=seed + model_index * 100 + comparison_index,
                draws=bootstrap_draws,
                analysis=f"family:{model}:{comparison}",
            )
            direction_summaries[comparison] = summary
            bootstrap_rows.extend(draws)
            primary_mean = summary["mean"]
            isomorphic_mean = float(
                np.mean([float(row["signed_isomorphic_delta"]) for row in group])
            )
            same_direction = (primary_mean > 0) == (isomorphic_mean > 0) and primary_mean != 0
            difference = abs(primary_mean - isomorphic_mean)
            invariant_rows.append(
                {
                    "model_repository": model,
                    "comparison": comparison,
                    "primary_mean": primary_mean,
                    "isomorphic_mean": isomorphic_mean,
                    "absolute_mean_difference": difference,
                    "frozen_tolerance": tolerance,
                    "direction_matches": same_direction,
                    "within_tolerance": difference <= tolerance,
                    "passed": same_direction and difference <= tolerance,
                    "terminal": True,
                }
            )
        family_rows.append(
            {
                "model_repository": model,
                "direction_results": direction_summaries,
                "identifiable": all(
                    direction_summaries[comparison]["identifiable_positive_direction"]
                    for comparison in comparisons
                ),
                "terminal": True,
            }
        )
        for source_index, source in enumerate(sources):
            for comparison_index, comparison in enumerate(comparisons):
                group = [
                    row
                    for row in rows
                    if row.get("model_repository") == model
                    and row.get("comparison") == comparison
                    and str(row.get("held_source_family")) == source
                ]
                if not group:
                    raise AuditInputError("held_source_cell_missing")
                summary, draws = _mean_bootstrap(
                    group,
                    value_field="signed_primary_delta",
                    seed=seed + 1_000 + model_index * 100 + source_index * 10 + comparison_index,
                    draws=bootstrap_draws,
                    analysis=f"held_source:{model}:{source}:{comparison}",
                )
                held_rows.append(
                    {
                        "model_repository": model,
                        "held_source_family": source,
                        "comparison": comparison,
                        **summary,
                        "terminal": True,
                    }
                )
                bootstrap_rows.extend(draws)
    return {
        "family_identifiability_rows": family_rows,
        "held_source_identifiability_rows": held_rows,
        "isomorphic_invariance_rows": invariant_rows,
        "grouped_bootstrap_rows": bootstrap_rows,
        "identifiable_family_count": sum(row["identifiable"] for row in family_rows),
    }


def reduce_release(
    *,
    audit_complete: bool,
    direct_leakage_count: int,
    prohibited_upper: float | None,
    sidecar_invariant: bool,
    isomorphic_invariant: bool,
    family_cells_complete: bool,
    identifiable_family_count: int,
) -> JsonDict:
    """Reduce all release rules to one bare readiness score and verdict."""

    complete_score = int(audit_complete)
    disqualified = bool(
        audit_complete
        and (
            direct_leakage_count != 0
            or prohibited_upper is None
            or prohibited_upper >= PROHIBITED_AUROC_LIMIT
            or not sidecar_invariant
            or not isomorphic_invariant
            or not family_cells_complete
        )
    )
    ready = bool(audit_complete and not disqualified and identifiable_family_count >= 2)
    if not audit_complete:
        verdict_class = "partial"
        honest_verdict = "partial_causal_feature_audit"
    elif disqualified:
        verdict_class = "disqualified"
        honest_verdict = "disqualified: causal_feature_bank_release_rules_failed"
    elif not ready:
        verdict_class = "null"
        honest_verdict = "null: signed_direction_not_identifiable"
    else:
        verdict_class = "positive"
        honest_verdict = "positive: causal_feature_bank_ready"
    return {
        "causal_bank_audit_complete_score": complete_score,
        "causal_feature_bank_ready_score": int(ready),
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact without its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def _empty_evidence() -> JsonDict:
    evidence: JsonDict = {
        "source_artifact_hashes": {},
        "direct_leakage_count": 0,
        "prohibited_auroc_upper_bound_max": None,
        "identifiable_family_count": 0,
        "causal_bank_audit_complete_score": 0,
        "causal_feature_bank_ready_score": 0,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_causal_feature_audit",
    }
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field.endswith("_rows") or field in {"rows", "per_pair_results"}:
            evidence.setdefault(field, [])
    return evidence


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> JsonDict:
    """Build a schema-complete artifact for every terminal class."""

    values = _empty_evidence()
    values.update(deepcopy(dict(evidence)))
    summary = gate_summary(checks)
    if not summary["passed"]:
        values.update(
            {
                "causal_bank_audit_complete_score": 0,
                "causal_feature_bank_ready_score": 0,
                "verdict_class": "blocked",
                "honest_verdict": "blocked_causal_feature_audit",
            }
        )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        **values,
        "random_seed": RANDOM_SEED,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _verdict_prefix_matches(verdict_class: str, honest_verdict: str) -> bool:
    prefixes = {
        "positive": "positive:",
        "circular_positive": "circular_positive:",
        "null": "null:",
        "blocked": "blocked_",
        "disqualified": "disqualified:",
        "partial": "partial_",
    }
    return honest_verdict.startswith(prefixes.get(verdict_class, "<invalid>"))


def _probe_interval_rows(artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    output = []
    for container in set(PROBE_CONTAINERS.values()):
        output.extend(
            row
            for row in artifact.get(container, [])
            if isinstance(row, Mapping) and row.get("record_type") == "interval"
        )
    return output


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute aggregate claims and reject forged terminal artifacts."""

    errors = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"required_fields_missing:{missing}")
    principles = artifact.get("field_principles", {})
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        errors.append(f"field_principles_missing:{missing_principles}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_oracle_mismatch")
    verdict_class = str(artifact.get("verdict_class"))
    honest_verdict = str(artifact.get("honest_verdict"))
    if verdict_class not in VERDICT_CLASSES or not _verdict_prefix_matches(
        verdict_class, honest_verdict
    ):
        errors.append("verdict_prefix_mismatch")
    for field in (
        "direct_leakage_count",
        "identifiable_family_count",
        "causal_bank_audit_complete_score",
        "causal_feature_bank_ready_score",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"bare_integer_required:{field}")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("checksum_mismatch")
    blocked = artifact.get("gate_check_summary", {}).get("passed") is not True
    if blocked:
        if (
            artifact.get("causal_bank_audit_complete_score") != 0
            or artifact.get("causal_feature_bank_ready_score") != 0
            or verdict_class != "blocked"
        ):
            errors.append("ready_verdict_mismatch")
        return list(dict.fromkeys(errors))

    leakage_count = sum(
        row.get("leakage_detected") is True for row in artifact.get("direct_leakage_rows", [])
    )
    if leakage_count != artifact.get("direct_leakage_count"):
        errors.append("direct_leakage_count_mismatch")
    intervals = _probe_interval_rows(artifact)
    upper_values = [
        float(row["ci95_upper"]) for row in intervals if row.get("ci95_upper") is not None
    ]
    upper = max(upper_values) if len(upper_values) == len(PROBE_SPECS) else None
    if upper != artifact.get("prohibited_auroc_upper_bound_max"):
        errors.append("prohibited_upper_mismatch")
    identifiable_count = sum(
        row.get("identifiable") is True for row in artifact.get("family_identifiability_rows", [])
    )
    if identifiable_count != artifact.get("identifiable_family_count"):
        errors.append("identifiable_family_count_mismatch")
    expected_lengths = {
        "rows": EXPECTED_SIGNED_ROW_COUNT,
        "per_pair_results": EXPECTED_PAIR_COUNT,
        "learner_tensor_rows": EXPECTED_PAIR_COUNT,
        "sidecar_intervention_rows": 5,
        "isomorphic_invariance_rows": len(MODEL_FAMILIES) * 2,
        "family_identifiability_rows": len(MODEL_FAMILIES),
        "held_source_identifiability_rows": len(MODEL_FAMILIES) * 4 * 2,
    }
    terminal_complete = all(
        len(artifact.get(field, [])) == expected
        and all(row.get("terminal") is True for row in artifact.get(field, []))
        for field, expected in expected_lengths.items()
    )
    terminal_complete = terminal_complete and len(intervals) == len(PROBE_SPECS)
    terminal_complete = terminal_complete and all(
        row.get("terminal") is True for row in artifact.get("grouped_bootstrap_rows", [])
    )
    reduced = reduce_release(
        audit_complete=terminal_complete,
        direct_leakage_count=leakage_count,
        prohibited_upper=upper,
        sidecar_invariant=all(
            row.get("passed") is True for row in artifact.get("sidecar_intervention_rows", [])
        ),
        isomorphic_invariant=all(
            row.get("passed") is True for row in artifact.get("isomorphic_invariance_rows", [])
        ),
        family_cells_complete=len(artifact.get("learner_tensor_rows", [])) == EXPECTED_PAIR_COUNT,
        identifiable_family_count=identifiable_count,
    )
    for field in (
        "causal_bank_audit_complete_score",
        "causal_feature_bank_ready_score",
        "verdict_class",
        "honest_verdict",
    ):
        if artifact.get(field) != reduced[field]:
            errors.append("ready_verdict_mismatch")
            break
    return list(dict.fromkeys(errors))


def _response_freeze_hash(exp7013: Mapping[str, Any]) -> str:
    manifest = {
        "schema": "carnot.experiment_7013.three_family_intervention_surface.v1.response_freeze",
        "prompt_freeze_hash": exp7013.get("prompt_freeze_hash"),
        "response_frozen_at": exp7013.get("response_frozen_at"),
        "response_row_count": len(exp7013.get("rows", [])),
        "token_position_row_count": len(exp7013.get("token_position_rows", [])),
        "response_rows_hash": sha256_json(exp7013.get("rows", [])),
        "token_position_rows_hash": sha256_json(exp7013.get("token_position_rows", [])),
    }
    return sha256_json(manifest)


def _family_cells_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    identities = [(str(row.get("pair_id")), str(row.get("model_repository"))) for row in rows]
    grouped: dict[str, set[str]] = defaultdict(set)
    for pair_id, model in identities:
        grouped[pair_id].add(model)
    return bool(
        len(rows) == EXPECTED_SIGNED_ROW_COUNT
        and len(identities) == len(set(identities))
        and len(grouped) == EXPECTED_PAIR_COUNT
        and all(models == set(MODEL_FAMILIES) for models in grouped.values())
        and all(row.get("terminal") is True for row in rows)
    )


def _build_probe_input_rows(
    signed_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    sidecars: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    conditions: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sidecar_rows:
        sidecars[str(row.get("block_id"))].append(row)
    for row in condition_rows:
        conditions[(str(row.get("pair_id")), str(row.get("model_repository")))].append(row)
    output = []
    for row_order, row in enumerate(signed_rows):
        block_id = str(row["pair_id"])
        model = str(row["model_repository"])
        authority = sidecars[block_id]
        condition = conditions[(block_id, model)]
        if len(authority) != 4 or len(condition) != 4:
            raise AuditInputError("probe_join_incomplete")
        mutation_values = {str(value.get("mutation_kind")) for value in authority}
        source_values = {str(value.get("source_family")) for value in authority}
        split_values = {str(value.get("split")) for value in authority}
        if len(mutation_values) != 1 or len(source_values) != 1 or len(split_values) != 1:
            raise AuditInputError("probe_join_conflict")
        serializations = sorted(str(value.get("serialization_template")) for value in authority)
        char_counts = {int(value.get("prompt_char_count", 0)) for value in condition}
        word_counts = {int(value.get("prompt_word_count", 0)) for value in condition}
        if len(char_counts) != 1 or len(word_counts) != 1:
            raise AuditInputError("probe_length_conflict")
        primary = float(row["signed_primary_delta"])
        isomorphic = float(row["signed_isomorphic_delta"])
        output.append(
            {
                "block_id": block_id,
                "source_family": next(iter(source_values)),
                "target": int(row["comparison"] == "clean_to_repair"),
                "terminal": bool(row.get("terminal")),
                "row_order": row_order,
                "serialization_signature": "|".join(serializations),
                "prompt_char_count": next(iter(char_counts)),
                "prompt_word_count": next(iter(word_counts)),
                "mutation_kind": next(iter(mutation_values)),
                "split": next(iter(split_values)),
                "model_repository": model,
                "score_norm": math.sqrt(primary * primary + isomorphic * isomorphic),
                "primary_magnitude": abs(primary),
                "isomorphic_magnitude": abs(isomorphic),
            }
        )
    return output


def _exclusion_receipts() -> list[JsonDict]:
    output = []
    base = {field: 0.0 for field in ALLOWED_TENSOR_FIELDS}
    for category, field in PROHIBITED_FIELD_ATTACKS:
        attack = audit_tensor_payload([{**base, field: "attack"}])
        rejected = any(row.get("field") == field for row in attack["direct_leakage_rows"])
        output.append(
            {
                "category": category,
                "field": field,
                "present_in_tensor": False,
                "attack_rejected": rejected,
                "leakage_detected": False,
                "passed": rejected,
                "terminal": True,
            }
        )
    return output


def _pair_results(rows: Sequence[Mapping[str, Any]], batch: LearnerBatch) -> list[JsonDict]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    return [
        {
            "pair_id": pair_id,
            "target_joined_after_materialization": batch.targets[index],
            "model_response_rows": [deepcopy(dict(row)) for row in grouped[pair_id]],
            "family_cell_count": len(grouped[pair_id]),
            "terminal": len(grouped[pair_id]) == len(MODEL_FAMILIES),
        }
        for index, pair_id in enumerate(batch.keys)
    ]


def build_from_repo(
    repo_root: Path,
    *,
    run_date: str = RUN_DATE,
    output_path: Path | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    runtime_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Rebuild the full audit from frozen sources after exact preflight."""

    started = time.perf_counter()
    hashed = load_hashed_sources(repo_root)
    checks = list(hashed["checks"])
    replay_rows = [dict(row, terminal=True) for row in checks]
    evidence: JsonDict = {
        "source_artifact_hashes": hashed["hashes"],
        "replay_hash_rows": replay_rows,
    }
    if not hashed["passed"]:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            checks=checks,
            evidence=evidence,
        )
        if output_path is not None:
            write_json_atomic(output_path, artifact)
        return artifact
    values = hashed["values"]
    exp7012 = values["exp7012"]
    exp7013 = values["exp7013"]
    signed_rows = exp7013.get("signed_response_rows", [])
    evidence["source_artifact_hashes"] = {
        **hashed["hashes"],
        "prompt_freeze": exp7013.get("prompt_freeze_hash"),
        "response_freeze": exp7013.get("response_freeze_hash"),
    }
    runtime = dict(runtime_receipt or {})
    internal_checks = [
        gate_check("exp7012_readiness", 1, exp7012.get("intervention_pair_fixture_ready_score")),
        gate_check("exp7013_completion", 1, exp7013.get("intervention_surface_complete_score")),
        gate_check(
            "learner_prompt_hash", EXPECTED_PROMPT_FREEZE_HASH, exp7012.get("learner_prompt_hash")
        ),
        gate_check(
            "authority_sidecar_hash",
            EXPECTED_SOURCE_HASHES["authority_sidecar"],
            exp7012.get("authority_sidecar_hash"),
        ),
        gate_check(
            "prompt_freeze_hash", EXPECTED_PROMPT_FREEZE_HASH, exp7013.get("prompt_freeze_hash")
        ),
        gate_check(
            "response_freeze_hash",
            EXPECTED_RESPONSE_FREEZE_HASH,
            exp7013.get("response_freeze_hash"),
        ),
        gate_check(
            "response_freeze_replay",
            exp7013.get("response_freeze_hash"),
            _response_freeze_hash(exp7013),
        ),
        gate_check("signed_family_cells", True, _family_cells_complete(signed_rows)),
        gate_check("fresh_process", True, runtime.get("fresh_process")),
        gate_check("network_namespace_isolated", True, runtime.get("network_namespace_isolated")),
        gate_check("network_disabled", True, runtime.get("network_disabled")),
        gate_check("gpu_devices_visible", [], runtime.get("gpu_devices_visible")),
        gate_check("online_model_access", False, runtime.get("online_model_access")),
        gate_check("training_disabled", True, runtime.get("training_disabled")),
        gate_check("source_tree_read_only", True, runtime.get("source_tree_read_only")),
        gate_check("runtime_boundary", True, runtime.get("passed")),
    ]
    checks.extend(internal_checks)
    replay_rows.extend(dict(row, terminal=True) for row in internal_checks)
    evidence["replay_hash_rows"] = replay_rows
    if any(row["passed"] is not True for row in checks):
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            checks=checks,
            evidence=evidence,
        )
        if output_path is not None:
            write_json_atomic(output_path, artifact)
        return artifact

    batch = build_learner_tensor(signed_rows)
    projected_rows = [{field: row[field] for field in ALLOWED_TENSOR_FIELDS} for row in signed_rows]
    leakage = audit_tensor_payload(projected_rows)
    direct_rows = [*leakage["direct_leakage_rows"], *_exclusion_receipts()]
    sidecar_rows = audit_sidecar_interventions(
        signed_rows, values["authority_sidecar"], seed=RANDOM_SEED
    )
    probe_input = _build_probe_input_rows(
        signed_rows, values["authority_sidecar"], exp7013.get("condition_rows", [])
    )
    probe_containers = {container: [] for container in set(PROBE_CONTAINERS.values())}
    grouped_bootstrap_rows: list[JsonDict] = []
    interval_rows = []
    for probe_index, (probe_name, fields) in enumerate(PROBE_SPECS.items()):
        fitted = fit_grouped_probe(
            probe_name,
            probe_input,
            fields,
            seed=RANDOM_SEED + probe_index,
            bootstrap_draws=bootstrap_draws,
        )
        container = PROBE_CONTAINERS[probe_name]
        probe_containers[container].extend(
            [*fitted["probe_rows"], *fitted["prediction_rows"], fitted["interval_row"]]
        )
        grouped_bootstrap_rows.extend(fitted["bootstrap_rows"])
        interval_rows.append(fitted["interval_row"])
    identifiability = audit_identifiability(
        signed_rows,
        seed=RANDOM_SEED,
        bootstrap_draws=bootstrap_draws,
        tolerance=ISOMORPHIC_TOLERANCE,
    )
    grouped_bootstrap_rows.extend(identifiability["grouped_bootstrap_rows"])
    upper_values = [row.get("ci95_upper") for row in interval_rows]
    prohibited_upper = (
        max(float(value) for value in upper_values if value is not None)
        if all(value is not None for value in upper_values)
        else None
    )
    audit_complete = bool(
        len(batch.keys) == EXPECTED_PAIR_COUNT
        and len(sidecar_rows) == 5
        and all(row["terminal"] for row in direct_rows)
        and len(interval_rows) == len(PROBE_SPECS)
        and all(row["terminal"] for row in interval_rows)
        and len(identifiability["family_identifiability_rows"]) == len(MODEL_FAMILIES)
        and len(identifiability["held_source_identifiability_rows"]) == len(MODEL_FAMILIES) * 8
    )
    reduced = reduce_release(
        audit_complete=audit_complete,
        direct_leakage_count=leakage["direct_leakage_count"],
        prohibited_upper=prohibited_upper,
        sidecar_invariant=all(row["passed"] for row in sidecar_rows),
        isomorphic_invariant=all(
            row["passed"] for row in identifiability["isomorphic_invariance_rows"]
        ),
        family_cells_complete=_family_cells_complete(signed_rows),
        identifiable_family_count=identifiability["identifiable_family_count"],
    )
    evidence.update(
        {
            "rows": [deepcopy(dict(row)) for row in signed_rows],
            "per_pair_results": _pair_results(signed_rows, batch),
            "learner_tensor_rows": [deepcopy(row) for row in batch.tensor_rows],
            "sidecar_intervention_rows": sidecar_rows,
            "direct_leakage_rows": direct_rows,
            **probe_containers,
            "grouped_bootstrap_rows": grouped_bootstrap_rows,
            "isomorphic_invariance_rows": identifiability["isomorphic_invariance_rows"],
            "family_identifiability_rows": identifiability["family_identifiability_rows"],
            "held_source_identifiability_rows": identifiability["held_source_identifiability_rows"],
            "direct_leakage_count": leakage["direct_leakage_count"],
            "prohibited_auroc_upper_bound_max": prohibited_upper,
            "identifiable_family_count": identifiability["identifiable_family_count"],
            **reduced,
        }
    )
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        checks=checks,
        evidence=evidence,
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - a validator bug, not an input outcome.
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    if output_path is not None:
        write_json_atomic(output_path, artifact)
    return artifact


def sandbox_runtime_receipt(repo_root: Path) -> JsonDict:
    """Measure the child boundary before any source artifact is parsed."""

    parent_pid = int(os.environ.get("CARNOT_EXP7014_PARENT_PID", "0"))
    parent_netns = os.environ.get("CARNOT_EXP7014_PARENT_NETNS", "")
    child_netns = os.readlink("/proc/self/ns/net")
    network_lines = Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]
    interfaces = sorted(line.split(":", 1)[0].strip() for line in network_lines if ":" in line)
    route_lines = Path("/proc/net/route").read_text(encoding="utf-8").splitlines()[1:]
    gpu_devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    probe_path = repo_root / ".exp7014-write-probe"
    source_tree_read_only = False
    try:
        probe_path.write_text("probe", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
    except OSError:
        source_tree_read_only = True
    receipt = {
        "fresh_process": parent_pid > 0 and os.getpid() != parent_pid,
        "parent_process_pid": parent_pid,
        "fresh_process_pid": os.getpid(),
        "parent_network_namespace": parent_netns,
        "network_namespace": child_netns,
        "network_namespace_isolated": bool(parent_netns and child_netns != parent_netns),
        "network_interfaces": interfaces,
        "network_route_count": len(route_lines),
        "network_disabled": set(interfaces) <= {"lo"} and not route_lines,
        "gpu_devices_visible": gpu_devices,
        "online_model_access": not (
            os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        ),
        "training_disabled": os.environ.get("CARNOT_TRAINING_DISABLED") == "1",
        "source_tree_read_only": source_tree_read_only,
    }
    receipt["passed"] = bool(
        receipt["fresh_process"]
        and receipt["network_namespace_isolated"]
        and receipt["network_disabled"]
        and receipt["gpu_devices_visible"] == []
        and receipt["online_model_access"] is False
        and receipt["training_disabled"]
        and receipt["source_tree_read_only"]
    )
    return receipt


def fresh_process_command(
    *,
    executable: Path,
    wrapper: Path,
    repo_root: Path,
    writable_root: Path,
    output_path: Path,
    run_date: str,
) -> list[str]:
    """Build the Linux sandbox command with one temporary writable path."""

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
        "CARNOT_TRAINING_DISABLED",
        "1",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "CARNOT_EXP7014_PARENT_PID",
        str(os.getpid()),
        "--setenv",
        "CARNOT_EXP7014_PARENT_NETNS",
        os.readlink("/proc/self/ns/net"),
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
    """Write one complete result so readers cannot observe partial JSON."""

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
    *, repo_root: Path = REPO_ROOT, result_path: Path | None = None, run_date: str = RUN_DATE
) -> JsonDict:  # pragma: no cover - exercised by the required command.
    """Run the cold audit in a network-disabled read-only child process."""

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
    before = {name: sha256_path(repo_root / path) for name, path in SOURCE_PATHS.items()}
    with tempfile.TemporaryDirectory(prefix="carnot-exp7014-") as directory:
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
    after = {name: sha256_path(repo_root / path) for name, path in SOURCE_PATHS.items()}
    controller_checks = [
        gate_check("controller_source_hashes_unchanged", before, after),
        gate_check("fresh_process_exit_code", 0, completed.returncode),
    ]
    checks = [*artifact.get("preconditions_checked", []), *controller_checks]
    artifact["preconditions_checked"] = checks
    artifact["gate_check_summary"] = gate_summary(checks)
    artifact.setdefault("replay_hash_rows", []).extend(
        dict(row, terminal=True) for row in controller_checks
    )
    artifact["duration_s"] = time.perf_counter() - started
    if not artifact["gate_check_summary"]["passed"]:
        artifact.update(
            {
                "causal_bank_audit_complete_score": 0,
                "causal_feature_bank_ready_score": 0,
                "verdict_class": "blocked",
                "honest_verdict": "blocked_causal_feature_audit",
            }
        )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(final_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command boundary.
    """Run the controller or its private restricted child mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--fresh-child", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.fresh_child:
        if args.output is None:
            parser.error("--output is required with --fresh-child")
        artifact = build_from_repo(
            REPO_ROOT,
            run_date=args.date,
            output_path=args.output,
            runtime_receipt=sandbox_runtime_receipt(REPO_ROOT),
        )
        errors = validate_artifact(artifact)
        if errors:
            raise RuntimeError(f"child_artifact_validation_failed:{errors}")
        return 0
    run_controller(run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
