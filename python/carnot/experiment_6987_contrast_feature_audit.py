"""Audit the frozen contrast bank without importing its producer.

The controller uses a Linux sandbox for the child audit. The child hashes all
inputs before it parses claims, rebuilds joins from per-row evidence, and fits
only preregistered shortcut probes. A clean audit can still disqualify the bank.

Spec refs: REQ-VERIFY-6987 and SCENARIO-VERIFY-6987-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
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


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260904"
EXPERIMENT_ID = 6987
SCHEMA = "carnot.exp6987.contrast_feature_audit.v1"
INFERENCE_SUBSTRATE = "fresh_process_contrast_leakage_audit_no_llm"
RANDOM_SEED = 69_872_026
EXPECTED_CANDIDATE_ROWS = 138
EXPECTED_FEATURE_ROWS = 414
SHORTCUT_AUROC_LIMIT = 0.80
SHORTCUT_FOLDS = 5
BOOTSTRAP_SAMPLES = 2_000

MODULE_PATH = Path("python/carnot/experiment_6987_contrast_feature_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6987_contrast_feature_audit.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_6987_contrast_feature_audit.json")

SOURCE_PATHS = {
    "exp6975": Path("results/experiment_6975_delayed_constraint_candidate_bank.json"),
    "exp6976": Path("results/experiment_6976_exact_candidate_certification.json"),
    "exp6984": Path("results/experiment_6984_exact_contrast_fixture.json"),
    "exp6985": Path("results/experiment_6985_chronological_constraint_stream.json"),
    "exp6985_stream": Path(
        "results/raw/experiment_6985_chronological_constraint_stream/chronological_stream.jsonl"
    ),
    "exp6985_labels": Path(
        "results/raw/experiment_6985_chronological_constraint_stream/sealed_labels.jsonl"
    ),
    "exp6986": Path("results/experiment_6986_three_family_contrast_features.json"),
}
EXPECTED_SOURCE_HASHES = {
    "exp6975": "sha256:4a9faf7223d174729091248f8cb763cc6b76e481abe69bdadc19adfe7dac5455",
    "exp6976": "sha256:7d174415abe6b9c3777bc56bdbf0eaa38a37aba390c64685c667f625aa6e05b9",
    "exp6984": "sha256:15f9a9bb58ca7793966f2fbac548f6879a64417b50e31b0504078cfe6ea46a3f",
    "exp6985": "sha256:461421d4771b5d997799fd2b7cb2c9e2cdfd684fbf792d25b139fcc49c30bede",
    "exp6985_stream": "sha256:33e7c839224aba4293838a920dc3213ee49c55c13c03cdffc45ee179e2bd2fca",
    "exp6985_labels": "sha256:cb454989dfeb1a433b917e20de99be3eb403fd2f4ca51481a1d622e4c3cb421c",
    "exp6986": "sha256:c19f5180c5e70677baec92db5ea295599197919eb03c5119d9a22849320cff75",
}
EXPECTED_CONTENT_HASHES = {
    "raw_token_rows": "sha256:4ddcc4de18872271e457350045285f85b4d28c7e935fa073e4d86290653be9a7",
    "joined_label_rows": "sha256:32c062aa7bfd3ee189f6ff99be66357f3a1662f433ab0388390984865fcd595f",
    "per_candidate_model_rows": (
        "sha256:3a57cb75f2c2e802a9905e20d02d52591b55aeddc726d3c13eb8a1d00e397c3a"
    ),
    "parser_feature_rows": (
        "sha256:108e7520cdb7f5d5be671decb7bfc6237336de6d0ed0cfe0afe77305726da27a"
    ),
    "scoring_manifest_rows": (
        "sha256:2fbde55b3d6495612f31fa315b2a7a2822c23b9aaaaa11075f112f058fa93cb0"
    ),
}
EXPECTED_SCORING_MANIFEST_HASH = (
    "sha256:1ab5b765a5a1c379fba1edc544c98765a4777779489be2c830e4c99ac16068c1"
)

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_INPUT_FIELDS = frozenset({"candidate_id", "candidate_text", "prompt_text"})
PRIMARY_SPLITS = ("train", "calibration", "held_out")

PROHIBITED_DIRECT_FIELDS = frozenset(
    {
        "authority_result",
        "authority_results",
        "certified_relation",
        "enumeration_result",
        "exact_label",
        "exact_labels",
        "exact_semantic_success",
        "expected_label",
        "fault_family",
        "future_window",
        "future_windows",
        "held_future_window",
        "held_future_window_rows",
        "mutation_type",
        "mutation_types",
        "recurrence_link",
        "recurrence_links",
        "source_group",
        "source_group_id",
        "split",
        "z3_result",
    }
)
PROHIBITED_FUTURE_FIELDS = frozenset(
    {
        "future_window",
        "future_windows",
        "held_future_window",
        "held_future_window_rows",
        "recurrence_link",
        "recurrence_links",
    }
)
PROHIBITED_TRAINING_FIELDS = frozenset(
    set(PROHIBITED_DIRECT_FIELDS)
    | {
        "candidate_hash",
        "candidate_id",
        "candidate_order",
        "contrast_group_id",
        "event_id",
        "event_ordinal",
        "full_token_hash",
        "manifest_ordinal",
        "model_id",
        "pair_id",
        "pair_position",
        "prompt_hash",
        "prompt_token_hash",
        "raw_token_hash",
        "serialization_hash",
        "source_block",
        "source_pair_id",
    }
)
TRAINING_FEATURE_FIELDS = (
    "array_count",
    "array_item_count",
    "boolean_count",
    "brace_count",
    "bracket_count",
    "byte_count",
    "candidate_token_count",
    "full_token_count",
    "json_parseable",
    "key_count",
    "length_normalized_nll",
    "line_count",
    "max_abs_local_surprisal_change",
    "mean_local_surprisal_change",
    "mean_token_entropy",
    "mean_top_probability_margin",
    "null_count",
    "number_count",
    "object_count",
    "sequence_nll",
    "string_count",
)

SHORTCUT_PROBE_SPECS = {
    "length_only": ("byte_count", "line_count", "candidate_token_count", "full_token_count"),
    "serialization_only": (
        "json_parseable",
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
    ),
    "source_metadata_only": ("source_block", "split", "formulation_family"),
    "mutation_metadata_only": ("fault_family", "schedule_id", "event_type"),
    "candidate_order_only": ("manifest_ordinal", "pair_position"),
    "identifier_style_only": (
        "candidate_id_prefix",
        "candidate_id_length",
        "candidate_id_digest_fraction",
    ),
    "model_identity_only": ("model_id",),
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_candidate_model_rows",
    "join_replay_rows",
    "row_count_rows",
    "family_coverage_rows",
    "label_balance_rows",
    "source_overlap_rows",
    "split_isolation_rows",
    "label_denial_replay_rows",
    "feature_schema_rows",
    "prohibited_field_rows",
    "shortcut_probe_rows",
    "shortcut_interval_rows",
    "source_disagreement_rows",
    "read_only_enforcement_receipt",
    "direct_leakage_count",
    "feature_audit_complete_score",
    "contrast_feature_bank_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "A reason for every field makes the audit contract reviewable.",
    "preconditions_checked": "Pinned checks prevent changed evidence from entering the audit.",
    "inference_substrate": "The substrate separates an offline audit from model scoring.",
    "duration_s": "Wall time shows that the audit process executed.",
    "source_artifact_hashes": "Byte hashes bind every conclusion to frozen source evidence.",
    "rows": "Primary replay rows keep the full denominator visible.",
    "per_candidate_model_rows": "Joined rows expose every candidate and model view.",
    "join_replay_rows": "Join receipts detect missing, duplicate, or altered evidence.",
    "row_count_rows": "Count receipts prevent aggregate claims from hiding omissions.",
    "family_coverage_rows": "Coverage rows require all three models for every candidate.",
    "label_balance_rows": "Split counts expose constant-label and imbalance shortcuts.",
    "source_overlap_rows": "Overlap rows reveal source reuse across partitions.",
    "split_isolation_rows": "Per-source rows make partition leakage traceable.",
    "label_denial_replay_rows": "Payload receipts prove oracle fields stayed outside scoring.",
    "feature_schema_rows": "A frozen schema separates trainable signals from bookkeeping.",
    "prohibited_field_rows": "Exact paths make direct leakage actionable.",
    "shortcut_probe_rows": "Fold receipts prove grouped evaluation without pair crossover.",
    "shortcut_interval_rows": "Pair bootstrap bounds provenance-only discrimination.",
    "source_disagreement_rows": "Every row-level contradiction remains visible.",
    "read_only_enforcement_receipt": "Sandbox receipts prove the audit could not alter sources.",
    "direct_leakage_count": "A bare count gates any direct oracle or provenance exposure.",
    "feature_audit_complete_score": "Completion requires every audit receipt, even on failure.",
    "contrast_feature_bank_ready_score": "Readiness requires integrity and no strong shortcut.",
    "random_seed": "One seed fixes folds and pair bootstrap draws.",
    "reproducibility_checksum": "A timing-free digest detects audit evidence drift.",
    "gate_check_summary": "Expected and observed values expose the first failed gate.",
    "verifier_is_oracle": "True states that exact labels judge leakage and balance.",
    "verdict_class": "A closed class separates readiness from disqualification.",
    "honest_verdict": "A class-specific prefix gives automation a stable terminal state.",
}
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}


def canonical_json(value: Any) -> str:
    """Return stable JSON text for small receipts and tests."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a project-style SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text without platform-dependent encoding choices."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash canonical JSON incrementally so the raw bank needs no second copy."""

    digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash a file as bytes, or return None when the source is unavailable."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def gate_check(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record a fail-closed check with its exact comparison values."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve all checks and promote the first failure for automation."""

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
    *,
    source_paths: Mapping[str, Path] = SOURCE_PATHS,
    expected_hashes: Mapping[str, str] = EXPECTED_SOURCE_HASHES,
) -> JsonDict:
    """Hash every source before parsing any JSON or aggregate claim."""

    hash_rows: list[JsonDict] = []
    observed: dict[str, str | None] = {}
    for source_id, relative_path in source_paths.items():
        value = sha256_path(repo_root / relative_path)
        observed[source_id] = value
        hash_rows.append(
            gate_check(f"source_hash:{source_id}", expected_hashes.get(source_id), value)
            | {"terminal": True}
        )
    if any(row["passed"] is not True for row in hash_rows):
        return {"passed": False, "sources": {}, "hash_rows": hash_rows, "hashes": observed}

    sources: dict[str, Any] = {}
    try:
        for source_id, relative_path in source_paths.items():
            text = (repo_root / relative_path).read_text(encoding="utf-8")
            if relative_path.suffix == ".jsonl":
                sources[source_id] = [json.loads(line) for line in text.splitlines() if line]
            else:
                sources[source_id] = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        hash_rows.append(
            gate_check("source_json_parse", "all sources parse", f"{type(exc).__name__}: {exc}")
            | {"terminal": True}
        )
        return {"passed": False, "sources": {}, "hash_rows": hash_rows, "hashes": observed}
    return {"passed": True, "sources": sources, "hash_rows": hash_rows, "hashes": observed}


def audit_bank_preconditions(
    bank: Mapping[str, Any],
    *,
    expected_content_hashes: Mapping[str, str] = EXPECTED_CONTENT_HASHES,
    expected_feature_rows: int = EXPECTED_FEATURE_ROWS,
    expected_candidate_rows: int = EXPECTED_CANDIDATE_ROWS,
) -> JsonDict:
    """Check bare readiness, row counts, content hashes, and model coverage."""

    features = list(bank.get("per_candidate_model_rows", []))
    manifest = list(bank.get("scoring_manifest_rows", []))
    joined = list(bank.get("joined_label_rows", []))
    checks = [
        gate_check(
            "three_family_feature_bank_complete_score",
            1,
            bank.get("three_family_feature_bank_complete_score"),
        ),
        gate_check("feature_row_count", expected_feature_rows, len(features)),
        gate_check("candidate_row_count", expected_candidate_rows, len(manifest)),
        gate_check("joined_label_row_count", expected_feature_rows, len(joined)),
    ]
    for field, expected in expected_content_hashes.items():
        checks.append(gate_check(f"{field}_hash", expected, sha256_json(bank.get(field, []))))
    checks.append(
        gate_check(
            "scoring_manifest_hash",
            EXPECTED_SCORING_MANIFEST_HASH if expected_feature_rows == EXPECTED_FEATURE_ROWS else None,
            bank.get("scoring_manifest_hash")
            if expected_feature_rows == EXPECTED_FEATURE_ROWS
            else None,
        )
    )
    keys = [(row.get("candidate_id"), row.get("model_id")) for row in features]
    candidate_models: dict[str, set[str]] = defaultdict(set)
    for candidate_id, model_id in keys:
        candidate_models[str(candidate_id)].add(str(model_id))
    checks.extend(
        (
            gate_check("unique_feature_keys", len(keys), len(set(keys))),
            gate_check(
                "all_candidate_model_families",
                True,
                bool(candidate_models)
                and all(models == set(REQUIRED_MODEL_IDS) for models in candidate_models.values()),
            ),
        )
    )
    failed = [str(row["check"]) for row in checks if row["passed"] is not True]
    return {"passed": not failed, "checks": checks, "failed_checks": failed}


def _transfer_candidate_id(attempt_key: str) -> str:
    return "transfer_" + sha256_text(attempt_key).removeprefix("sha256:")[:24]


def _normalize_split(value: Any) -> str:
    text = str(value or "")
    return "held_out" if text == "heldout" else text


def build_metadata_catalog(sources: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Derive labels and provenance from source rows instead of Exp6986 joins."""

    catalog: dict[str, JsonDict] = {}
    fixture = sources["exp6984"]
    fixture_pairs = {
        str(row["contrast_group_id"]): row for row in fixture.get("per_pair_results", [])
    }
    fixture_labels = {
        str(row["candidate_id"]): row.get("certified_relation")
        for row in fixture.get("authority_agreement_rows", [])
    }
    fixture_faults = {
        str(row["candidate_id"]): row.get("fault_family")
        for row in fixture.get("mutation_attempt_rows", [])
    }
    for row in fixture.get("per_candidate_rows", []):
        candidate_id = str(row["candidate_id"])
        pair = fixture_pairs[str(row["contrast_group_id"])]
        catalog[candidate_id] = {
            "source_block": "exp6984",
            "source_pair_id": str(pair["source_pair_id"]),
            "source_group_id": str(pair["source_group_id"]),
            "split": _normalize_split(row.get("split")),
            "pair_position": row.get("pair_position"),
            "fault_family": fixture_faults.get(candidate_id),
            "formulation_family": row.get("formulation_family"),
            "event_ordinal": None,
            "event_type": None,
            "schedule_id": None,
            "exact_label": fixture_labels.get(candidate_id),
            "audit_only": False,
        }

    stream = sources["exp6985"]
    stream_events = {
        str(row["event_id"]): row for row in stream.get("per_event_results", [])
    }
    stream_labels: dict[str, Any] = {}
    for event in sources["exp6985_labels"]:
        for candidate_id, label in zip(
            event.get("candidate_ids", []), event.get("exact_labels", []), strict=True
        ):
            stream_labels[str(candidate_id)] = label
    for row in stream.get("per_candidate_rows", []):
        candidate_id = str(row["candidate_id"])
        event = stream_events[str(row["event_id"])]
        catalog[candidate_id] = {
            "source_block": "exp6985",
            "source_pair_id": str(event["source_pair_id"]),
            "source_group_id": str(event["source_group_id"]),
            "split": "chronological_audit",
            "pair_position": row.get("pair_position"),
            "fault_family": row.get("fault_family"),
            "formulation_family": next(
                (
                    raw.get("formulation_family")
                    for raw in sources["exp6985_stream"]
                    if raw.get("event_id") == row.get("event_id")
                ),
                None,
            ),
            "event_ordinal": row.get("event_ordinal"),
            "event_type": event.get("event_type"),
            "schedule_id": None,
            "exact_label": stream_labels.get(candidate_id),
            "audit_only": True,
        }

    certification = sources["exp6976"]
    for row in certification.get("per_candidate_rows", []):
        if row.get("parse_success") is not True:
            continue
        candidate_id = _transfer_candidate_id(str(row["attempt_key"]))
        split = _normalize_split(row.get("split"))
        catalog[candidate_id] = {
            "source_block": "exp6976_transfer",
            "source_pair_id": str(row.get("pair_id")),
            "source_group_id": str(row.get("pair_id")),
            "split": f"transfer_{split}",
            "pair_position": None,
            "fault_family": None,
            "formulation_family": row.get("formulation_family"),
            "event_ordinal": None,
            "event_type": None,
            "schedule_id": row.get("schedule_id"),
            "exact_label": row.get("expected_label"),
            "audit_only": True,
        }
    return catalog


def _duplicates(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> set[tuple[Any, ...]]:
    keys = [tuple(row.get(field) for field in fields) for row in rows]
    return {key for key, count in Counter(keys).items() if count > 1}


def _json_structure_counts(value: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if isinstance(value, Mapping):
        counts["object_count"] += 1
        counts["key_count"] += len(value)
        for child in value.values():
            counts.update(_json_structure_counts(child))
    elif isinstance(value, list):
        counts["array_count"] += 1
        counts["array_item_count"] += len(value)
        for child in value:
            counts.update(_json_structure_counts(child))
    elif value is None:
        counts["null_count"] += 1
    elif isinstance(value, bool):
        counts["boolean_count"] += 1
    elif isinstance(value, (int, float)):
        counts["number_count"] += 1
    elif isinstance(value, str):
        counts["string_count"] += 1
    return counts


def parser_features(candidate_text: str) -> JsonDict:
    """Recompute structural features without calling the Exp6986 producer."""

    count_fields = (
        "object_count",
        "array_count",
        "key_count",
        "array_item_count",
        "null_count",
        "boolean_count",
        "number_count",
        "string_count",
    )
    try:
        parsed = json.loads(candidate_text)
        counts = _json_structure_counts(parsed)
        parseable = True
    except json.JSONDecodeError:
        counts = Counter()
        parseable = False
    return {
        "json_parseable": parseable,
        "byte_count": len(candidate_text.encode("utf-8")),
        "line_count": candidate_text.count("\n") + 1,
        "brace_count": candidate_text.count("{") + candidate_text.count("}"),
        "bracket_count": candidate_text.count("[") + candidate_text.count("]"),
        **{field: int(counts[field]) for field in count_fields},
    }


def _same_number(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(
        float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
    )


def _raw_replay(feature: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    metrics: dict[str, float] = {}
    if raw_rows:
        surprisals = [float(row["surprisal"]) for row in raw_rows]
        entropies = [float(row["token_entropy"]) for row in raw_rows]
        margins = [float(row["top_probability_margin"]) for row in raw_rows]
        changes = [float(row["local_surprisal_change"]) for row in raw_rows]
        metrics = {
            "sequence_nll": sum(surprisals),
            "length_normalized_nll": sum(surprisals) / len(surprisals),
            "mean_token_entropy": sum(entropies) / len(entropies),
            "mean_top_probability_margin": sum(margins) / len(margins),
            "mean_local_surprisal_change": sum(changes) / len(changes),
            "max_abs_local_surprisal_change": max(abs(value) for value in changes),
        }
    metric_matches = bool(metrics) and all(
        _same_number(feature.get(field), value) for field, value in metrics.items()
    )
    token_hash_matches = True
    if feature.get("candidate_token_hash") is not None:
        token_hash_matches = feature.get("candidate_token_hash") == sha256_json(
            [int(row["selected_token_id"]) for row in raw_rows]
        )
    return {
        "raw_token_count_replays": len(raw_rows) == feature.get("raw_token_count"),
        "raw_token_hash_replays": sha256_json(raw_rows) == feature.get("raw_token_hash"),
        "candidate_token_hash_replays": token_hash_matches,
        "sequence_metrics_replay": metric_matches,
    }


def rebuild_candidate_model_joins(
    bank: Mapping[str, Any], metadata_catalog: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    """Rebuild every candidate-model-label join and retain each disagreement."""

    manifest = list(bank.get("scoring_manifest_rows", []))
    features = list(bank.get("per_candidate_model_rows", []))
    labels = list(bank.get("joined_label_rows", []))
    parsers = list(bank.get("parser_feature_rows", []))
    raw_rows = list(bank.get("raw_token_rows", []))
    disagreements: list[JsonDict] = []

    for kind, rows in (("feature", features), ("label", labels), ("parser", parsers)):
        for key in sorted(_duplicates(rows, ("candidate_id", "model_id")), key=str):
            disagreements.append(
                {"kind": f"duplicate_{kind}_key", "key": list(key), "terminal": True}
            )
    manifest_duplicates = _duplicates(manifest, ("candidate_id",))
    for key in sorted(manifest_duplicates, key=str):
        disagreements.append({"kind": "duplicate_manifest_key", "key": list(key), "terminal": True})

    manifest_by_id = {str(row.get("candidate_id")): row for row in manifest}
    feature_by_key = {
        (str(row.get("candidate_id")), str(row.get("model_id"))): row for row in features
    }
    label_by_key = {
        (str(row.get("candidate_id")), str(row.get("model_id"))): row for row in labels
    }
    parser_by_key = {
        (str(row.get("candidate_id")), str(row.get("model_id"))): row for row in parsers
    }
    raw_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        raw_by_key[(str(row.get("candidate_id")), str(row.get("model_id")))].append(row)

    expected_keys = {
        (candidate_id, model_id)
        for candidate_id in manifest_by_id
        for model_id in REQUIRED_MODEL_IDS
    }
    for kind, indexed in (("feature", feature_by_key), ("label", label_by_key), ("parser", parser_by_key)):
        for key in sorted(expected_keys - set(indexed)):
            disagreements.append({"kind": f"missing_{kind}_key", "key": list(key), "terminal": True})
        for key in sorted(set(indexed) - expected_keys):
            disagreements.append({"kind": f"extra_{kind}_key", "key": list(key), "terminal": True})

    joined_rows: list[JsonDict] = []
    replay_rows: list[JsonDict] = []
    for candidate_id, model_id in sorted(
        expected_keys, key=lambda key: (int(manifest_by_id[key[0]].get("ordinal", 0)), key[1])
    ):
        manifest_row = manifest_by_id[candidate_id]
        feature = feature_by_key.get((candidate_id, model_id))
        label = label_by_key.get((candidate_id, model_id))
        parser = parser_by_key.get((candidate_id, model_id))
        metadata = metadata_catalog.get(candidate_id)
        if feature is None or label is None or parser is None or metadata is None:
            replay_rows.append(
                {
                    "candidate_id": candidate_id,
                    "model_id": model_id,
                    "passed": False,
                    "terminal": True,
                }
            )
            continue

        raw_replay = _raw_replay(feature, raw_by_key.get((candidate_id, model_id), []))
        parser_recomputed = parser_features(str(manifest_row.get("candidate_text", "")))
        parser_replays = all(parser.get(field) == value for field, value in parser_recomputed.items())
        expected_label = metadata.get("exact_label")
        checks = {
            "candidate_hash_replays": (
                feature.get("candidate_hash") == manifest_row.get("candidate_hash")
                == label.get("candidate_hash")
                == sha256_text(str(manifest_row.get("candidate_text", "")))
            ),
            "source_label_replays": expected_label is None or label.get("exact_label") == expected_label,
            "terminal_replays": feature.get("terminal") is True,
            "parser_features_replay": parser_replays,
            **raw_replay,
        }
        passed = all(checks.values())
        replay_rows.append(
            {
                "candidate_id": candidate_id,
                "model_id": model_id,
                **checks,
                "passed": passed,
                "terminal": True,
            }
        )
        if not passed:
            disagreements.append(
                {
                    "kind": "row_replay_failed",
                    "key": [candidate_id, model_id],
                    "failed_checks": [name for name, value in checks.items() if not value],
                    "terminal": True,
                }
            )
        combined = {
            **deepcopy(dict(feature)),
            **{key: value for key, value in parser.items() if key not in {"candidate_id", "model_id"}},
            **deepcopy(dict(metadata)),
            "candidate_id": candidate_id,
            "candidate_hash": feature.get("candidate_hash"),
            "model_id": model_id,
            "manifest_ordinal": manifest_row.get("ordinal"),
            "exact_label": label.get("exact_label"),
        }
        prefix = candidate_id.split("_", 1)[0].split(":", 1)[0]
        combined["candidate_id_prefix"] = prefix
        combined["candidate_id_length"] = len(candidate_id)
        combined["candidate_id_digest_fraction"] = int(
            hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8], 16
        ) / float(0xFFFFFFFF)
        joined_rows.append(combined)

    family_rows = []
    for candidate_id in manifest_by_id:
        models = sorted(
            str(row.get("model_id"))
            for row in features
            if str(row.get("candidate_id")) == candidate_id
        )
        family_rows.append(
            {
                "candidate_id": candidate_id,
                "expected_models": list(REQUIRED_MODEL_IDS),
                "observed_models": models,
                "complete": models == sorted(REQUIRED_MODEL_IDS),
                "terminal": True,
            }
        )
    row_count_rows = [
        {"table": "scoring_manifest_rows", "expected": len(manifest_by_id), "observed": len(manifest)},
        {"table": "per_candidate_model_rows", "expected": len(expected_keys), "observed": len(features)},
        {"table": "joined_label_rows", "expected": len(expected_keys), "observed": len(labels)},
        {
            "table": "raw_token_rows",
            "expected": sum(int(row.get("raw_token_count", 0) or 0) for row in features),
            "observed": len(raw_rows),
        },
    ]
    for row in row_count_rows:
        row["passed"] = row["expected"] == row["observed"]
        row["terminal"] = True
    return {
        "per_candidate_model_rows": joined_rows,
        "join_replay_rows": replay_rows,
        "row_count_rows": row_count_rows,
        "family_coverage_rows": family_rows,
        "source_disagreement_rows": disagreements,
    }


def audit_label_balance(
    rows: Sequence[Mapping[str, Any]], *, required_splits: Sequence[str] = PRIMARY_SPLITS
) -> list[JsonDict]:
    """Count labels once per candidate and mark only primary splits as release gates."""

    candidates: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidates.setdefault(str(row.get("candidate_id")), row)
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates.values():
        by_split[str(row.get("split"))].append(row)
    output = []
    for split in sorted(set(by_split) | set(required_splits)):
        split_rows = by_split.get(split, [])
        positive = sum(row.get("exact_label") == "equivalent" for row in split_rows)
        negative = sum(row.get("exact_label") == "non_equivalent" for row in split_rows)
        required = split in required_splits
        output.append(
            {
                "split": split,
                "candidate_count": len(split_rows),
                "pair_count": len({row.get("source_pair_id") for row in split_rows}),
                "positive_count": positive,
                "negative_count": negative,
                "nonconstant": positive > 0 and negative > 0,
                "exactly_balanced": positive == negative and positive > 0,
                "required_for_readiness": required,
                "passed": (positive == negative and positive > 0) if required else True,
                "terminal": True,
            }
        )
    return output


def audit_source_isolation(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compare source-group and pair identities across every declared partition."""

    candidates: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidates.setdefault(str(row.get("candidate_id")), row)
    partition_keys: dict[str, set[str]] = defaultdict(set)
    key_partitions: dict[str, set[str]] = defaultdict(set)
    for row in candidates.values():
        partition = str(row.get("split"))
        keys = {
            f"group:{row.get('source_group_id')}" if row.get("source_group_id") else "",
            f"pair:{row.get('source_pair_id')}" if row.get("source_pair_id") else "",
        } - {""}
        partition_keys[partition].update(keys)
        for key in keys:
            key_partitions[key].add(partition)

    overlaps = []
    partitions = sorted(partition_keys)
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            common = sorted(partition_keys[left] & partition_keys[right])
            overlaps.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "overlap_count": len(common),
                    "overlap_source_keys": common,
                    "passed": not common,
                    "terminal": True,
                }
            )
    isolation = [
        {
            "source_key": key,
            "partitions": sorted(partitions_for_key),
            "partition_count": len(partitions_for_key),
            "passed": len(partitions_for_key) == 1,
            "terminal": True,
        }
        for key, partitions_for_key in sorted(key_partitions.items())
    ]
    return {
        "source_overlap_rows": overlaps,
        "split_isolation_rows": isolation,
        "all_disjoint": all(row["passed"] for row in overlaps),
    }


def _prohibited_paths(value: Any, prohibited: set[str] | frozenset[str], path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).casefold() in prohibited:
                paths.append(child_path)
            paths.extend(_prohibited_paths(child, prohibited, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_prohibited_paths(child, prohibited, f"{path}[{index}]"))
    return paths


def audit_future_isolation(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reject later event references and future-only keys in current features."""

    output = []
    for row in rows:
        current = int(row.get("event_ordinal", 0) or 0)
        visible = [int(value) for value in row.get("visible_event_ordinals", [])]
        later = sorted(value for value in visible if value >= current)
        paths = _prohibited_paths(row.get("feature_payload", {}), PROHIBITED_FUTURE_FIELDS)
        hash_replays = row.get("manifest_hash_replays", True) is True
        passed = not later and not paths and hash_replays
        output.append(
            {
                "candidate_id": row.get("candidate_id"),
                "event_ordinal": current,
                "visible_event_ordinals": visible,
                "later_event_references": later,
                "prohibited_paths": paths,
                "manifest_hash_replays": hash_replays,
                "passed": passed,
                "terminal": True,
            }
        )
    return {"rows": output, "passed": bool(output) and all(row["passed"] for row in output)}


def build_future_view_rows(
    repo_root: Path, sources: Mapping[str, Any], joined_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Hash each visibility file before rebuilding its chronological view."""

    features_by_candidate: dict[str, JsonDict] = defaultdict(dict)
    for row in joined_rows:
        candidate_id = str(row.get("candidate_id"))
        if not features_by_candidate[candidate_id]:
            features_by_candidate[candidate_id] = {
                field: row.get(field) for field in TRAINING_FEATURE_FIELDS if field in row
            }
    event_candidates = {
        int(row["event_ordinal"]): list(row.get("candidate_ids", []))
        for row in sources["exp6985"].get("per_event_results", [])
    }
    output = []
    base = repo_root / "results/raw/experiment_6985_chronological_constraint_stream"
    for receipt in sources["exp6985"].get("visibility_manifest_rows", []):
        ordinal = int(receipt["event_ordinal"])
        path = base / str(receipt["manifest_path"])
        observed = sha256_path(path)
        expected = receipt.get("manifest_sha256")
        view: Mapping[str, Any] = {}
        if observed == expected:
            view = json.loads(path.read_text(encoding="utf-8"))
        for candidate_id in event_candidates.get(ordinal, []):
            output.append(
                {
                    "candidate_id": candidate_id,
                    "event_ordinal": ordinal,
                    "visible_event_ordinals": list(view.get("visible_event_ordinals", [])),
                    "feature_payload": features_by_candidate.get(str(candidate_id), {}),
                    "manifest_hash_replays": observed == expected,
                }
            )
    return output


def rebuild_model_input_blocks(manifest_rows: Sequence[Mapping[str, Any]]) -> list[list[JsonDict]]:
    """Strip controller metadata exactly as the frozen scorer receipts declare."""

    blocks = []
    for source_block in ("exp6984", "exp6985", "exp6976_transfer"):
        blocks.append(
            [
                {field: row[field] for field in MODEL_INPUT_FIELDS}
                for row in manifest_rows
                if row.get("source_block") == source_block
            ]
        )
    return blocks


def audit_label_denial(
    bank: Mapping[str, Any], training_feature_fields: Sequence[str]
) -> JsonDict:
    """Replay worker payload receipts and scan nested model text for direct fields."""

    blocks = rebuild_model_input_blocks(bank.get("scoring_manifest_rows", []))
    prohibited_rows: list[JsonDict] = []
    for path in _prohibited_paths(blocks, PROHIBITED_DIRECT_FIELDS):
        prohibited_rows.append({"category": "model_input_field", "path": path, "terminal": True})
    for block_index, block in enumerate(blocks):
        for row_index, row in enumerate(block):
            try:
                parsed = json.loads(str(row.get("candidate_text", "")))
            except json.JSONDecodeError:
                continue
            for path in _prohibited_paths(parsed, PROHIBITED_DIRECT_FIELDS):
                prohibited_rows.append(
                    {
                        "category": "nested_candidate_text",
                        "path": f"$[{block_index}][{row_index}].candidate_text_json{path[1:]}",
                        "terminal": True,
                    }
                )
    for field in training_feature_fields:
        if field.casefold() in PROHIBITED_TRAINING_FIELDS:
            prohibited_rows.append(
                {
                    "category": "training_schema",
                    "path": f"training_schema.{field}",
                    "terminal": True,
                }
            )

    payload_hash = sha256_json(blocks)
    replay_rows = []
    for receipt in bank.get("process_isolation_rows", []):
        expected_fields = sorted(MODEL_INPUT_FIELDS)
        passed = bool(
            receipt.get("passed") is True
            and receipt.get("scorer_input_field_names") == expected_fields
            and receipt.get("input_payload_hash") == payload_hash
            and receipt.get("denied_fields_visible") == []
        )
        replay_rows.append(
            {
                "check": "model_process_payload",
                "model_id": receipt.get("model_id"),
                "expected_input_fields": expected_fields,
                "observed_input_fields": receipt.get("scorer_input_field_names"),
                "expected_payload_hash": payload_hash,
                "observed_payload_hash": receipt.get("input_payload_hash"),
                "passed": passed,
                "terminal": True,
            }
        )
    stored_receipts = {
        str(row.get("mutation_field")): row for row in bank.get("label_denial_rows", [])
    }
    path_text = "\n".join(str(row["path"]) for row in prohibited_rows)
    for field in sorted(PROHIBITED_DIRECT_FIELDS):
        stored = stored_receipts.get(field)
        count = sum(field in line for line in path_text.splitlines())
        replay_rows.append(
            {
                "check": "prohibited_field_absent",
                "field": field,
                "observed_path_count": count,
                "stored_mutation_receipt_present": stored is not None,
                "stored_mutation_receipt_passed": stored.get("passed") if stored else None,
                "passed": count == 0 and (stored is None or stored.get("passed") is True),
                "terminal": True,
            }
        )
    return {
        "label_denial_replay_rows": replay_rows,
        "prohibited_field_rows": prohibited_rows,
        "direct_leakage_count": len(prohibited_rows),
        "passed": bool(replay_rows) and all(row["passed"] for row in replay_rows),
    }


def build_feature_schema_rows(
    sequence_fields: set[str], parser_fields: set[str]
) -> list[JsonDict]:
    """Freeze the training allowlist and explain every excluded bank field."""

    output = []
    for field in sorted(sequence_fields | parser_fields):
        included = field in TRAINING_FEATURE_FIELDS
        output.append(
            {
                "field": field,
                "source_schema": (
                    "sequence" if field in sequence_fields else "parser"
                ),
                "included_in_training_schema": included,
                "prohibited": field in PROHIBITED_TRAINING_FIELDS,
                "reason": (
                    "preregistered_label_blind_feature"
                    if included
                    else "bookkeeping_identifier_provenance_or_receipt"
                ),
                "terminal": True,
            }
        )
    return output


def _binary_label(value: Any) -> int:
    if value == "equivalent":
        return 1
    if value == "non_equivalent":
        return 0
    raise ValueError(f"unknown_exact_label:{value}")


def _feature_dict(row: Mapping[str, Any], fields: Sequence[str]) -> JsonDict:
    output: JsonDict = {}
    for field in fields:
        value = row.get(field)
        if isinstance(value, (bool, int, float)) and not (
            isinstance(value, float) and not math.isfinite(value)
        ):
            output[field] = value
        else:
            output[field] = f"{field}={value if value is not None else '<missing>'}"
    return output


def _pair_bootstrap_auc(
    candidate_rows: Sequence[Mapping[str, Any]], *, random_seed: int, samples: int
) -> tuple[float, float, list[JsonDict]]:
    pair_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        pair_rows[str(row["source_pair_id"])].append(row)
    pair_ids = sorted(pair_rows)
    rng = np.random.default_rng(random_seed)
    values = []
    for _ in range(samples):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        selected = [row for pair_id in sampled for row in pair_rows[str(pair_id)]]
        labels = [int(row["label"]) for row in selected]
        if len(set(labels)) < 2:
            continue
        auc = float(roc_auc_score(labels, [float(row["score"]) for row in selected]))
        values.append(max(auc, 1.0 - auc))
    summaries = [
        {
            "source_pair_id": pair_id,
            "candidate_count": len(group),
            "positive_count": sum(int(row["label"]) for row in group),
            "negative_count": sum(1 - int(row["label"]) for row in group),
            "mean_oof_score": float(np.mean([float(row["score"]) for row in group])),
        }
        for pair_id, group in sorted(pair_rows.items())
    ]
    if not values:
        return math.nan, math.nan, summaries
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), summaries


def fit_shortcut_probe(
    rows: Sequence[Mapping[str, Any]],
    *,
    probe_name: str,
    feature_fields: Sequence[str],
    random_seed: int = RANDOM_SEED,
    folds: int = SHORTCUT_FOLDS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> JsonDict:
    """Fit one fold-local probe and bootstrap its out-of-fold candidate scores by pair."""

    if not rows:
        return {
            "probe_rows": [],
            "interval_row": {
                "probe_name": probe_name,
                "feature_fields": list(feature_fields),
                "shortcut_auroc": None,
                "ci95_lower": None,
                "ci95_upper": None,
                "gate_passed": False,
                "terminal": True,
            },
        }
    labels = np.asarray([_binary_label(row.get("exact_label")) for row in rows], dtype=int)
    groups = np.asarray([str(row.get("source_pair_id")) for row in rows], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2 or len(set(labels.tolist())) < 2:
        raise ValueError(f"shortcut_probe_not_identifiable:{probe_name}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scores = np.full(len(rows), np.nan, dtype=float)
    probe_rows = []
    dummy = np.zeros((len(rows), 1), dtype=float)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(dummy, labels, groups)):
        vectorizer = DictVectorizer(sparse=True)
        train_features = [_feature_dict(rows[index], feature_fields) for index in train_index]
        test_features = [_feature_dict(rows[index], feature_fields) for index in test_index]
        train_matrix = vectorizer.fit_transform(train_features).toarray()
        test_matrix = vectorizer.transform(test_features).toarray()
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=random_seed + fold_index,
            solver="liblinear",
        )
        model.fit(train_matrix, labels[train_index])
        scores[test_index] = model.predict_proba(test_matrix)[:, 1]
        train_groups = set(groups[train_index].tolist())
        test_groups = set(groups[test_index].tolist())
        fold_auc = (
            float(roc_auc_score(labels[test_index], scores[test_index]))
            if len(set(labels[test_index].tolist())) == 2
            else None
        )
        probe_rows.append(
            {
                "probe_name": probe_name,
                "fold_index": fold_index,
                "feature_fields": list(feature_fields),
                "train_row_count": len(train_index),
                "test_row_count": len(test_index),
                "train_pair_count": len(train_groups),
                "test_pair_count": len(test_groups),
                "group_overlap_count": len(train_groups & test_groups),
                "fold_auroc": fold_auc,
                "preprocessing_fit_on_train_only": True,
                "terminal": True,
            }
        )
    if np.isnan(scores).any():
        raise ValueError(f"shortcut_probe_oof_gap:{probe_name}")

    candidate_scores: dict[str, list[float]] = defaultdict(list)
    candidate_labels: dict[str, set[int]] = defaultdict(set)
    candidate_pairs: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        candidate_id = str(row.get("candidate_id"))
        candidate_scores[candidate_id].append(float(scores[index]))
        candidate_labels[candidate_id].add(int(labels[index]))
        candidate_pairs[candidate_id].add(str(groups[index]))
    if any(len(values) != 1 for values in candidate_labels.values()) or any(
        len(values) != 1 for values in candidate_pairs.values()
    ):
        raise ValueError(f"shortcut_probe_candidate_disagreement:{probe_name}")
    candidates = [
        {
            "candidate_id": candidate_id,
            "source_pair_id": next(iter(candidate_pairs[candidate_id])),
            "label": next(iter(candidate_labels[candidate_id])),
            "score": float(np.mean(values)),
        }
        for candidate_id, values in sorted(candidate_scores.items())
    ]
    candidate_y = [int(row["label"]) for row in candidates]
    auc = float(roc_auc_score(candidate_y, [float(row["score"]) for row in candidates]))
    shortcut_auc = max(auc, 1.0 - auc)
    lower, upper, pair_rows = _pair_bootstrap_auc(
        candidates, random_seed=random_seed, samples=bootstrap_samples
    )
    valid_interval = math.isfinite(lower) and math.isfinite(upper)
    interval = {
        "probe_name": probe_name,
        "feature_fields": list(feature_fields),
        "candidate_count": len(candidates),
        "pair_count": len(pair_rows),
        "bootstrap_unit": "source_pair_id",
        "bootstrap_samples": bootstrap_samples,
        "auroc": auc,
        "shortcut_auroc": shortcut_auc,
        "ci95_lower": lower if valid_interval else None,
        "ci95_upper": upper if valid_interval else None,
        "threshold": SHORTCUT_AUROC_LIMIT,
        "gate_passed": valid_interval and upper < SHORTCUT_AUROC_LIMIT,
        "pair_rows": pair_rows,
        "terminal": True,
    }
    return {"probe_rows": probe_rows, "interval_row": interval}


def fit_all_shortcut_probes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Run every preregistered shortcut probe in a stable order."""

    probe_rows = []
    interval_rows = []
    for offset, (name, fields) in enumerate(SHORTCUT_PROBE_SPECS.items()):
        result = fit_shortcut_probe(
            rows,
            probe_name=name,
            feature_fields=fields,
            random_seed=RANDOM_SEED + offset,
        )
        probe_rows.extend(result["probe_rows"])
        interval_rows.append(result["interval_row"])
    return {"shortcut_probe_rows": probe_rows, "shortcut_interval_rows": interval_rows}


def reduce_readiness(
    *,
    audit_complete: bool,
    direct_leakage_count: int,
    models_complete: bool,
    splits_balanced: bool,
    sources_disjoint: bool,
    future_isolated: bool,
    shortcut_interval_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Separate terminal audit completion from permission to train."""

    audit_score = int(audit_complete)
    shortcut_clean = bool(shortcut_interval_rows) and all(
        row.get("gate_passed") is True for row in shortcut_interval_rows
    )
    ready = bool(
        audit_complete
        and direct_leakage_count == 0
        and models_complete
        and splits_balanced
        and sources_disjoint
        and future_isolated
        and shortcut_clean
    )
    if not audit_complete:
        verdict_class = "partial"
        verdict = "partial_contrast_feature_audit"
    elif ready:
        verdict_class = "circular_positive"
        verdict = "complete_circular_contrast_feature_bank_ready"
    else:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_contrast_feature_bank_shortcut_gate"
    return {
        "feature_audit_complete_score": audit_score,
        "contrast_feature_bank_ready_score": int(ready),
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> JsonDict:
    """Build one complete blocked, partial, disqualified, or ready artifact."""

    checks = [deepcopy(dict(row)) for row in preconditions_checked]
    preconditions_pass = bool(checks) and all(row.get("passed") is True for row in checks)
    defaults: JsonDict = {
        "source_artifact_hashes": {},
        "rows": [],
        "per_candidate_model_rows": [],
        "join_replay_rows": [],
        "row_count_rows": [],
        "family_coverage_rows": [],
        "label_balance_rows": [],
        "source_overlap_rows": [],
        "split_isolation_rows": [],
        "label_denial_replay_rows": [],
        "feature_schema_rows": [],
        "prohibited_field_rows": [],
        "shortcut_probe_rows": [],
        "shortcut_interval_rows": [],
        "source_disagreement_rows": [],
        "read_only_enforcement_receipt": {},
        "direct_leakage_count": 0,
        "feature_audit_complete_score": 0,
        "contrast_feature_bank_ready_score": 0,
        "verdict_class": "partial",
        "honest_verdict": "partial_contrast_feature_audit",
    }
    defaults.update(deepcopy(dict(evidence)))
    if not preconditions_pass:
        defaults.update(
            {
                "feature_audit_complete_score": 0,
                "contrast_feature_bank_ready_score": 0,
                "verdict_class": "blocked",
                "honest_verdict": "blocked_contrast_feature_audit",
            }
        )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        **defaults,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, bare gates, verdict meaning, and checksum."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        errors.append(f"required_fields_missing:{sorted(missing)}")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    for field in (
        "direct_leakage_count",
        "feature_audit_complete_score",
        "contrast_feature_bank_ready_score",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field}_not_bare_int")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict_class not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    prefixes = {
        "positive": "complete_",
        "circular_positive": "complete_circular_",
        "null": "complete_null_",
        "blocked": "blocked_",
        "disqualified": "complete_disqualified_",
        "partial": "partial_",
    }
    if verdict_class in prefixes and not verdict.startswith(prefixes[verdict_class]):
        errors.append("verdict_prefix_mismatch")
    if artifact.get("contrast_feature_bank_ready_score") == 1 and (
        artifact.get("feature_audit_complete_score") != 1
        or verdict_class != "circular_positive"
    ):
        errors.append("readiness_state_mismatch")
    if verdict_class == "disqualified" and artifact.get("feature_audit_complete_score") != 1:
        errors.append("disqualification_without_complete_audit")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def sandbox_runtime_receipt(repo_root: Path, source_paths: Mapping[str, Path]) -> JsonDict:
    """Prove the child lacks network, GPU, LLM, and source write capabilities."""

    parent_netns = os.environ.get("CARNOT_EXP6987_PARENT_NETNS", "")
    child_netns = os.readlink("/proc/self/ns/net")
    write_rows = []
    for source_id, relative in source_paths.items():
        path = repo_root / relative
        denied = False
        error = None
        try:
            with path.open("rb+"):
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
        "source_write_rows": write_rows,
        "source_tree_read_only": bool(write_rows) and all(row["write_denied"] for row in write_rows),
        "output_is_temporary": not str(os.environ.get("CARNOT_EXP6987_CHILD_OUTPUT", "")).startswith(
            str(repo_root / "results")
        ),
    }
    receipt["passed"] = bool(
        receipt["network_namespace_isolated"]
        and not gpu_devices
        and receipt["cuda_visible_devices"] == ""
        and receipt["hf_hub_offline"]
        and receipt["transformers_offline"]
        and not llm_modules
        and receipt["source_tree_read_only"]
        and receipt["output_is_temporary"]
    )
    return receipt


def _source_claim_disagreements(
    sources: Mapping[str, Any], joined_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Compare stored source-count claims with the rebuilt candidate roster."""

    counts = Counter()
    seen = set()
    for row in joined_rows:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in seen:
            counts[str(row.get("source_block"))] += 1
            seen.add(candidate_id)
    output = []
    for stored in sources["exp6986"].get("candidate_source_count_rows", []):
        source = str(stored.get("source_block"))
        if counts[source] != stored.get("observed"):
            output.append(
                {
                    "kind": "source_count_disagreement",
                    "source_block": source,
                    "expected_value": stored.get("observed"),
                    "observed_value": counts[source],
                    "terminal": True,
                }
            )
    return output


def build_from_repo(
    repo_root: Path = REPO_ROOT, *, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:
    """Execute the complete audit inside the already-created sandbox child."""

    started = time.perf_counter()
    receipt = sandbox_runtime_receipt(repo_root, SOURCE_PATHS)
    loaded = load_hashed_sources(repo_root)
    preconditions = [
        *loaded["hash_rows"],
        gate_check("read_only_sandbox", True, receipt.get("passed")),
    ]
    if loaded["passed"] is not True:
        return build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions_checked=preconditions,
            evidence={
                "source_artifact_hashes": loaded["hashes"],
                "read_only_enforcement_receipt": receipt,
            },
        )

    sources = loaded["sources"]
    bank = sources["exp6986"]
    bank_checks = audit_bank_preconditions(bank)
    preconditions.extend(bank_checks["checks"])
    preconditions.extend(
        (
            gate_check(
                "exp6986_embedded_exp6984_hash",
                EXPECTED_SOURCE_HASHES["exp6984"],
                bank.get("source_artifact_hashes", {}).get("exp6984"),
            ),
            gate_check(
                "exp6986_embedded_exp6985_hash",
                EXPECTED_SOURCE_HASHES["exp6985"],
                bank.get("source_artifact_hashes", {}).get("exp6985"),
            ),
        )
    )
    if any(row["passed"] is not True for row in preconditions):
        return build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions_checked=preconditions,
            evidence={
                "source_artifact_hashes": loaded["hashes"],
                "read_only_enforcement_receipt": receipt,
            },
        )

    metadata = build_metadata_catalog(sources)
    rebuilt = rebuild_candidate_model_joins(bank, metadata)
    joined_rows = rebuilt["per_candidate_model_rows"]
    balance_rows = audit_label_balance(joined_rows)
    source_isolation = audit_source_isolation(joined_rows)
    future_rows = build_future_view_rows(repo_root, sources, joined_rows)
    future = audit_future_isolation(future_rows)
    denial = audit_label_denial(bank, TRAINING_FEATURE_FIELDS)
    sequence_fields = {
        str(field) for row in bank.get("per_candidate_model_rows", []) for field in row
    }
    parser_fields = {str(field) for row in bank.get("parser_feature_rows", []) for field in row}
    schema_rows = build_feature_schema_rows(sequence_fields, parser_fields)
    probes = fit_all_shortcut_probes(joined_rows)
    disagreements = [
        *rebuilt["source_disagreement_rows"],
        *_source_claim_disagreements(sources, joined_rows),
    ]

    models_complete = bool(rebuilt["family_coverage_rows"]) and all(
        row["complete"] for row in rebuilt["family_coverage_rows"]
    )
    splits_balanced = all(
        row["passed"] for row in balance_rows if row["required_for_readiness"]
    ) and {row["split"] for row in balance_rows if row["required_for_readiness"]} == set(
        PRIMARY_SPLITS
    )
    audit_complete = bool(
        not disagreements
        and all(row["passed"] for row in rebuilt["join_replay_rows"])
        and all(row["passed"] for row in rebuilt["row_count_rows"])
        and models_complete
        and denial["passed"]
        and future["passed"]
        and all(row.get("ci95_upper") is not None for row in probes["shortcut_interval_rows"])
        and receipt["passed"]
    )
    reduced = reduce_readiness(
        audit_complete=audit_complete,
        direct_leakage_count=denial["direct_leakage_count"],
        models_complete=models_complete,
        splits_balanced=splits_balanced,
        sources_disjoint=source_isolation["all_disjoint"],
        future_isolated=future["passed"],
        shortcut_interval_rows=probes["shortcut_interval_rows"],
    )
    evidence = {
        "source_artifact_hashes": loaded["hashes"],
        "rows": rebuilt["join_replay_rows"],
        "per_candidate_model_rows": joined_rows,
        "join_replay_rows": rebuilt["join_replay_rows"],
        "row_count_rows": rebuilt["row_count_rows"],
        "family_coverage_rows": rebuilt["family_coverage_rows"],
        "label_balance_rows": balance_rows,
        "source_overlap_rows": source_isolation["source_overlap_rows"],
        "split_isolation_rows": source_isolation["split_isolation_rows"],
        "label_denial_replay_rows": denial["label_denial_replay_rows"],
        "feature_schema_rows": schema_rows,
        "prohibited_field_rows": denial["prohibited_field_rows"],
        "shortcut_probe_rows": probes["shortcut_probe_rows"],
        "shortcut_interval_rows": probes["shortcut_interval_rows"],
        "source_disagreement_rows": disagreements,
        "read_only_enforcement_receipt": receipt,
        "direct_leakage_count": denial["direct_leakage_count"],
        **reduced,
    }
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        preconditions_checked=preconditions,
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
    """Build the Linux sandbox command with one narrow writable output mount."""

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
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "CARNOT_EXP6987_PARENT_NETNS",
        parent_netns,
        "--setenv",
        "CARNOT_EXP6987_CHILD_OUTPUT",
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
    """Run the audit child and keep the repository read-only until final output."""

    started = time.perf_counter()
    final_path = result_path or (repo_root / RESULT_PATH)
    if shutil.which("bwrap") is None:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions_checked=[gate_check("bubblewrap_available", True, False)],
            evidence={},
        )
        write_json_atomic(final_path, artifact)
        return artifact

    before = {source_id: sha256_path(repo_root / path) for source_id, path in SOURCE_PATHS.items()}
    with tempfile.TemporaryDirectory(prefix="carnot-exp6987-") as directory:
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
                preconditions_checked=[
                    gate_check("fresh_process_exit_code", 0, completed.returncode),
                    gate_check("fresh_process_output", True, child_output.is_file()),
                ],
                evidence={},
            )
    after = {source_id: sha256_path(repo_root / path) for source_id, path in SOURCE_PATHS.items()}
    unchanged = before == after
    receipt = dict(artifact.get("read_only_enforcement_receipt", {}))
    receipt.update(
        {
            "controller_source_hashes_before": before,
            "controller_source_hashes_after": after,
            "controller_source_hashes_unchanged": unchanged,
            "fresh_child_exit_code": completed.returncode,
            "fresh_child_stderr_hash": sha256_text(completed.stderr),
            "sandbox_command_hash": sha256_json(command),
        }
    )
    receipt["passed"] = bool(receipt.get("passed") and unchanged and completed.returncode == 0)
    artifact["read_only_enforcement_receipt"] = receipt
    artifact["duration_s"] = time.perf_counter() - started
    if not receipt["passed"] and artifact.get("verdict_class") != "blocked":
        checks = list(artifact.get("preconditions_checked", []))
        checks.append(gate_check("controller_read_only_enforcement", True, receipt["passed"]))
        artifact["preconditions_checked"] = checks
        artifact["gate_check_summary"] = gate_summary(checks)
        artifact["feature_audit_complete_score"] = 0
        artifact["contrast_feature_bank_ready_score"] = 0
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = "blocked_contrast_feature_audit"
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(final_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command boundary.
    """Run the controller or its private fresh-child mode."""

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
