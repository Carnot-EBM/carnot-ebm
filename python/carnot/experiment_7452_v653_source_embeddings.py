"""Capture final-layer source-conditioned embeddings without generation.

The experiment uses llama.cpp per-token embedding output. It stores the final
token vector from one fixed final layer for two complete text views. Labels are
not read until token eligibility is sealed. This evidence does not reproduce a
mid-layer NF4 study because the runtime exposes only the final-layer surface.

Spec refs: REQ-VERIFY-7452 and SCENARIO-VERIFY-7452-*.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from typing import Any, Protocol

from carnot import experiment_7449_v653_source_protocol as source_protocol
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.sota_models import cached_current_model
from carnot.reporting.experiment_7303_validation_scope import CommandSpec


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
PHASE = 2
EXPERIMENT_ID = "exp7452-v653-source-embeddings"
TASK_ID = "experiment_7452_v653_source_embeddings"
SCHEMA = "carnot.exp7452.v653.source_embeddings.v1"
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
RESULT_PATH = Path("results/experiment_7452_v653_source_embeddings.json")
RAW_DIR = Path("results/raw/experiment_7452_v653_source_embeddings")
MODULE_PATH = Path("python/carnot/experiment_7452_v653_source_embeddings.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7452_v653_source_embeddings.py")
TEST_PATH = Path("tests/python/test_experiment_7452_v653_source_embeddings.py")
SHARED_TEST_PATH = Path("tests/python/test_experiment_7449_v653_source_protocol.py")
LIFECYCLE_PATH = Path("results/experiment_7448_v653_capture_lifecycle.json")
PROTOCOL_PATH = Path("results/experiment_7449_v653_source_protocol.json")
PROTOCOL_RAW_DIR = Path("results/raw/experiment_7449_v653_source_protocol")

MODEL_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [{"hf_id": MODEL_HF_ID, "quantization": "Q4_K_M"}]
INFERENCE_SUBSTRATE_CLASS = "model_load_no_generation"
EXECUTION_VENUE = "host"
TOKEN_CEILING = 2_048
PROJECTION_DIMENSIONS = 32
PROJECTION_SEED = 6_530_032
STREAM_SEED = 6_530_052
RESAMPLING_SEED = 6_530_010
FIT_SEEDS = (65_301, 65_302, 65_303, 65_304, 65_305)
ARMS = ("response", "source_response")
MINIMUM_GROUPS = deepcopy(source_protocol.MINIMUM_GROUPS)
GROUPS_PER_SHARD = 16
MAX_EVALUATION_FORWARDS = 800
MAX_DEVELOPMENT_FORWARDS = 8
FORWARD_BUDGET_S = 1_800.0
CALL_CEILING_S = 30.0
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))

AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(), SHARED_TEST_PATH.as_posix()),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Use a versioned top-level schema and exact experiment, milestone, and terminal status.",
    "run_date": "Use 20260920 and retain actual UTC, monotonic duration, boot identity, and segment identity.",
    "preconditions_checked": "Name actual paths, resources, identities, and observed upstream gate values before model work.",
    "MODEL_SPECS": "Name only unsloth/Qwen3.8-27B-GGUF for this planned current LLM task.",
    "model_invoked": "Separate an attempted current model load from archived or scripted model-shaped events.",
    "invocation_counts": "Balance attempted, completed, failed, cancelled, and in-flight loads, forwards, and generations.",
    "inference_substrate": "Name the actual native final-layer surface and keep model and device facts in typed details.",
    "inference_substrate_class": "Use model_load_no_generation; load and embedding work has a two-second classification floor without padding.",
    "execution_venue": "Use host while CUDA identity remains separate from the compute class.",
    "duration_s": "Measure real current work and separate load, forward, numeric, and validation time without padding.",
    "phase_spans": "Bind phase timings, progress events, checkpoints, and monotonic clock segments.",
    "random_seed": "Freeze projection, stream, fit, and resampling seeds even though no selector is trained here.",
    "reproducibility_checksum": "Bind code, protocol, immutable inputs, model, row shards, and exact validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes, original classes, and original flags without rehabilitation.",
    "rows": "Keep every arm and group, including failed, censored, excluded, and unstarted cells.",
    "sample_size_budget": "Separate planned, attempted, completed, failed, censored, excluded, and unstarted units under fixed limits.",
    "acceptance_gate_results": "Name each validity or benefit category, operator, expected value, observed value, and principle.",
    "gate_check_summary": "Name the first exact failure and keep missing, null, and zero as different observations.",
    "verifier_is_oracle": "False because human source-support labels, not this representation, supply scoring authority.",
    "honest_verdict": "Use complete_ for completed findings and a specific blocked_ reason for unchanged external absence.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Preserve critical findings because flagged or disqualified science cannot supply readiness.",
    "validation_receipts": "Record exact scoped commands, environments, exits, durations, and log hashes for all terminal readers.",
    "field_principles": "Explain field intent here while gate fields remain bare scalars.",
    "promotion_score": "Always zero because this milestone permits no rollout, generator change, or publication.",
    "embedding_capture_ready_score": "One requires authenticated final vectors, complete paired cells, and frozen role minima.",
    "representation_identity": "Bind exact final layer, last-token pooling, dimension, model, runner, tokenizer, offload, and lease.",
    "feature_shards": "Keep raw numerical vectors and the fixed projection reproducible without another model load.",
    "eligibility_rows": "Count token exclusions, failed calls, and budget censoring before any labels are used.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)


class CaptureInvalid(ValueError):
    """Reject evidence that cannot identify or replay the measured vector."""


class Tokenizer(Protocol):
    """Expose only the output-free tokenizer operation needed for eligibility."""

    def tokenize(self, text: str) -> list[int]:
        """Return the exact embedded-token IDs for one complete text view."""


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so independent readers get the same identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without loading a large GGUF into host memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON object only after its bytes reach storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for missing or malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _gate(
    check: str,
    category: str,
    upstream: str,
    path: str,
    field: str,
    op: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep one scalar comparison and its scientific meaning explicit."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def collect_upstream_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Authenticate exact upstream bytes and preserve their original dispositions."""

    definitions = (
        (
            "exp7448-capture-lifecycle",
            LIFECYCLE_PATH,
            "capture_lifecycle_ready_score",
        ),
        ("exp7449-source-protocol", PROTOCOL_PATH, "source_protocol_ready_score"),
    )
    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for upstream, relative, readiness_field in definitions:
        path = root / relative
        artifact = _load_object(path)
        present = path.is_file() and bool(artifact)
        observed_present: Any = True if present else "missing"
        checks.append(
            _gate(
                f"{upstream}.artifact_readable",
                "validity",
                upstream,
                relative.as_posix(),
                "artifact",
                "==",
                True,
                observed_present,
                present,
                "Dependent model work requires the exact readable upstream artifact.",
            )
        )
        if present:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_verdict_class": artifact.get("verdict_class"),
                "original_flagged_adversarial": artifact.get("flagged_adversarial"),
            }
        observations = (
            (
                readiness_field,
                "==",
                1,
                artifact.get(readiness_field) if present else "missing",
                artifact.get(readiness_field) == 1 if present else False,
            ),
            (
                "verdict_class",
                "in",
                ["null", "positive"],
                artifact.get("verdict_class") if present else "missing",
                artifact.get("verdict_class") in {"null", "positive"} if present else False,
            ),
            (
                "flagged_adversarial",
                "==",
                False,
                artifact.get("flagged_adversarial") if present else "missing",
                artifact.get("flagged_adversarial") is False if present else False,
            ),
        )
        for field, op, expected, observed, passed in observations:
            checks.append(
                _gate(
                    f"{upstream}.{field}",
                    "validity",
                    upstream,
                    relative.as_posix(),
                    field,
                    op,
                    expected,
                    observed,
                    passed,
                    "Readiness requires the original upstream value without reinterpretation.",
                )
            )
    return checks, hashes


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the first failed field without converting zero into absence."""

    failed = next((row for row in gates if row.get("passed") is not True), None)
    if failed is None:
        return {"passed": True, "failed_check": None}
    return {
        "passed": False,
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "path": failed.get("path"),
        "field": failed.get("field"),
        "op": failed.get("op"),
        "expected": deepcopy(failed.get("expected")),
        "observed": deepcopy(failed.get("observed")),
    }


def _representation_text(predictor: Mapping[str, Any], arm: str) -> str:
    """Create one complete neutral view without adding a classification request."""

    return source_protocol.representation_bytes(predictor, arm).decode("utf-8")


def build_eligibility_rows(
    predictors: Sequence[Mapping[str, Any]],
    tokenizer: Tokenizer,
    *,
    token_ceiling: int = TOKEN_CEILING,
) -> list[JsonDict]:
    """Count tokens in sealed hash order without reading evaluator labels."""

    pending: list[JsonDict] = []
    for predictor in predictors:
        source_hash = source_protocol.normalized_source_hash(str(predictor["source_text"]))
        response_hash = canonical_hash(str(predictor["response_text"]))
        for arm in ARMS:
            seal_hash = canonical_hash(
                {
                    "row_key": predictor["row_key"],
                    "group_id": predictor["group_id"],
                    "source_hash": source_hash,
                    "response_hash": response_hash,
                    "arm": arm,
                }
            )
            text = _representation_text(predictor, arm)
            tokens = tokenizer.tokenize(text)
            eligible = 0 < len(tokens) <= token_ceiling
            pending.append(
                {
                    "unit_id": f"{predictor['row_key']}:{arm}",
                    "row_key": predictor["row_key"],
                    "group_id": predictor["group_id"],
                    "source_hash": source_hash,
                    "response_hash": response_hash,
                    "corpus": predictor["corpus"],
                    "role": predictor["role"],
                    "arm": arm,
                    "seal_hash": seal_hash,
                    "token_count": len(tokens),
                    "token_ceiling": token_ceiling,
                    "eligible": eligible,
                    "truncated": False,
                    "status": "unstarted" if eligible else "excluded_token_ceiling",
                    "attempted": False,
                    "completed": False,
                    "failed": False,
                    "censored": False,
                    "unstarted": eligible,
                }
            )
    ordered = sorted(pending, key=lambda row: str(row["seal_hash"]))
    for index, row in enumerate(ordered):
        row["seal_order"] = index
    return ordered


def bind_response_identities(
    eligibility: Sequence[Mapping[str, Any]],
    evaluators: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join response identities by row key and verify group fields, never position."""

    by_key = {str(row.get("row_key")): row for row in evaluators}
    expected = {str(row.get("row_key")) for row in eligibility}
    if expected != set(by_key):
        raise CaptureInvalid("evaluator identity set mismatch")
    output: list[JsonDict] = []
    for cell in eligibility:
        evaluator = by_key[str(cell["row_key"])]
        if evaluator.get("group_id") != cell.get("group_id") or evaluator.get("corpus") != cell.get(
            "corpus"
        ):
            raise CaptureInvalid("evaluator identity mismatch")
        output.append({**deepcopy(dict(cell)), "response_id": evaluator.get("response_id")})
    return output


def _vector_bytes(vector: Sequence[float]) -> bytes:
    """Encode native values as little-endian float32 for stable compact shards."""

    if not vector:
        raise CaptureInvalid("empty vector")
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise CaptureInvalid("nonfinite vector")
    return struct.pack(f"<{len(values)}f", *values)


def _projection(vector: Sequence[float]) -> list[float]:
    """Apply one fixed signed CountSketch without fitting on any labels."""

    output = [0.0] * PROJECTION_DIMENSIONS
    scale = math.sqrt(max(1, len(vector)))
    for index, value in enumerate(vector):
        mixed = (index * 0x9E3779B185EBCA87 + PROJECTION_SEED) & ((1 << 64) - 1)
        bucket = mixed % PROJECTION_DIMENSIONS
        sign = -1.0 if (mixed >> 63) else 1.0
        output[bucket] += sign * float(value) / scale
    return [round(value, 9) for value in output]


def _identity_fields(row: Mapping[str, Any]) -> JsonDict:
    """Select every identifier that prevents positional or family-based joins."""

    fields = ("row_key", "group_id", "source_hash", "response_id", "role", "corpus", "arm")
    return {field: row.get(field) for field in fields}


def vector_row(
    identity: Mapping[str, Any], vector: Sequence[float], *, token_count: int
) -> JsonDict:
    """Reduce one final-token vector into replayable raw and projected evidence."""

    payload = _vector_bytes(vector)
    values = list(struct.unpack(f"<{len(payload) // 4}f", payload))
    identities = _identity_fields(identity)
    return {
        **identities,
        "unit_id": f"{identity.get('row_key')}:{identity.get('arm')}",
        "identity_hash": canonical_hash(identities),
        "status": "completed",
        "eligible": True,
        "attempted": True,
        "completed": True,
        "failed": False,
        "censored": False,
        "unstarted": False,
        "token_count": int(token_count),
        "dimension": len(values),
        "finite": True,
        "norm": math.sqrt(sum(value * value for value in values)),
        "vector_hash": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "vector_encoding": "base64_little_endian_float32",
        "vector_b64": base64.b64encode(payload).decode("ascii"),
        "projected_features": _projection(values),
        "projection_dimensions": PROJECTION_DIMENSIONS,
        "projection_seed": PROJECTION_SEED,
    }


def decode_vector(row: Mapping[str, Any]) -> list[float]:
    """Decode and authenticate one raw numerical vector from a feature shard."""

    try:
        payload = base64.b64decode(str(row["vector_b64"]), validate=True)
    except Exception as exc:
        raise CaptureInvalid("vector encoding invalid") from exc
    dimension = row.get("dimension")
    if not isinstance(dimension, int) or dimension <= 0 or len(payload) != dimension * 4:
        raise CaptureInvalid("vector dimension mismatch")
    if "sha256:" + hashlib.sha256(payload).hexdigest() != row.get("vector_hash"):
        raise CaptureInvalid("vector hash mismatch")
    values = list(struct.unpack(f"<{dimension}f", payload))
    if not all(math.isfinite(value) for value in values):
        raise CaptureInvalid("vector nonfinite")
    if not math.isclose(
        math.sqrt(sum(value * value for value in values)),
        float(row.get("norm", -1.0)),
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise CaptureInvalid("vector norm mismatch")
    if canonical_hash(_identity_fields(row)) != row.get("identity_hash"):
        raise CaptureInvalid("vector identity mismatch")
    if _projection(values) != row.get("projected_features"):
        raise CaptureInvalid("vector projection mismatch")
    return values


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize raw rows with stable keys and one object per line."""

    return b"".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Commit one shard only after all numerical bytes reach storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_feature_shards(
    raw_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    groups_per_shard: int = GROUPS_PER_SHARD,
) -> list[JsonDict]:
    """Write raw vectors in durable shards with at most sixteen source groups."""

    completed = [dict(row) for row in rows if row.get("completed") is True]
    groups = sorted({str(row["group_id"]) for row in completed})
    shards: list[JsonDict] = []
    feature_dir = raw_dir / "features"
    for shard_index, offset in enumerate(range(0, len(groups), groups_per_shard)):
        selected = set(groups[offset : offset + groups_per_shard])
        shard_rows = [row for row in completed if str(row["group_id"]) in selected]
        payload = _jsonl_bytes(shard_rows)
        path = feature_dir / f"vectors-{shard_index:03d}.jsonl"
        _atomic_bytes(path, payload)
        if len(payload) >= 20 * 1024 * 1024:
            raise CaptureInvalid("feature shard exceeds 20 MiB")
        shards.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "rows": len(shard_rows),
                "groups": len(selected),
                "group_ids": sorted(selected),
            }
        )
    return shards


def reload_feature_shards(raw_dir: Path, shards: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Rehash every shard and validate each numerical row without a model load."""

    rows: list[JsonDict] = []
    for shard in shards:
        path = raw_dir / str(shard["path"])
        payload = path.read_bytes()
        if "sha256:" + hashlib.sha256(payload).hexdigest() != shard.get("sha256"):
            raise CaptureInvalid("feature shard hash mismatch")
        parsed = [json.loads(line) for line in payload.splitlines()]
        if len(parsed) != shard.get("rows"):
            raise CaptureInvalid("feature shard row count mismatch")
        for row in parsed:
            decode_vector(row)
            rows.append(row)
    return rows


def run_integrity_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reload authentic rows and prove identity and dimension mutations fail."""

    authentic = bool(rows)
    if authentic:
        for row in rows[: min(4, len(rows))]:
            decode_vector(row)
    permuted_rejected = False
    dimension_rejected = False
    if rows:
        permuted = deepcopy(dict(rows[0]))
        permuted["response_id"] = str(permuted.get("response_id")) + "-permuted"
        try:
            decode_vector(permuted)
        except CaptureInvalid:
            permuted_rejected = True
        fabricated = deepcopy(dict(rows[0]))
        fabricated["dimension"] = int(fabricated["dimension"]) + 1
        try:
            decode_vector(fabricated)
        except CaptureInvalid:
            dimension_rejected = True
    passed = authentic and permuted_rejected and dimension_rejected
    return {
        "authentic_subset_reloaded": authentic,
        "permuted_join_rejected": permuted_rejected,
        "fabricated_dimension_rejected": dimension_rejected,
        "passed": passed,
    }


def _coverage(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count every terminal cell state and complete paired source group."""

    statuses = Counter(str(row.get("status")) for row in rows)
    paired: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("completed") is True:
            paired[(str(row.get("role")), str(row.get("group_id")))].add(str(row.get("arm")))
    complete_groups = Counter(role for (role, _group), arms in paired.items() if arms == set(ARMS))
    return {
        "planned_cells": len(rows),
        "eligible_cells": sum(row.get("eligible") is True for row in rows),
        "attempted_cells": sum(row.get("attempted") is True for row in rows),
        "completed_cells": sum(row.get("completed") is True for row in rows),
        "failed_cells": sum(row.get("failed") is True for row in rows),
        "censored_cells": sum(row.get("censored") is True for row in rows),
        "unstarted_cells": sum(row.get("unstarted") is True for row in rows),
        "excluded_cells": sum(str(row.get("status", "")).startswith("excluded_") for row in rows),
        "status_counts": dict(sorted(statuses.items())),
        "eligible_complete_groups": {role: complete_groups.get(role, 0) for role in MINIMUM_GROUPS},
    }


def reduce_capture(
    rows: Sequence[Mapping[str, Any]],
    *,
    controls_passed: bool,
    validation_passed: bool,
) -> JsonDict:
    """Reduce readiness from raw cell states without reading predictive labels."""

    coverage = _coverage(rows)
    complete_groups = coverage["eligible_complete_groups"]
    minima_met = all(complete_groups[role] >= count for role, count in MINIMUM_GROUPS.items())
    eligible_complete = coverage["eligible_cells"] == coverage["completed_cells"]
    flagged = not controls_passed
    ready = int(minima_met and eligible_complete and controls_passed and validation_passed)
    unfinished = coverage["unstarted_cells"] > 0 or coverage["censored_cells"] > 0
    if flagged or not validation_passed:
        verdict_class = "disqualified"
        honest = "complete_disqualified_source_embedding_evidence"
    elif unfinished:
        verdict_class = "partial"
        honest = "partial_retryable_source_embedding_capture"
    elif ready:
        verdict_class = "null"
        honest = "complete_null_source_embeddings_captured"
    else:
        verdict_class = "null"
        honest = "complete_null_source_embedding_capture_incomplete"
    return {
        "embedding_capture_ready_score": ready,
        "promotion_score": 0,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
        "flagged_adversarial": flagged,
        "eligible_complete_groups": complete_groups,
        "minimum_groups": deepcopy(MINIMUM_GROUPS),
        "coverage_diagnostics": coverage,
        "controls_passed": controls_passed,
        "validation_passed": validation_passed,
    }


def _zero_invocation_counts() -> JsonDict:
    """Return balanced counters for all prohibited and permitted operations."""

    return {
        operation: {
            state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
        }
        for operation in ("model_loads", "forward_calls", "generation_calls")
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the complete terminal record without recursively hashing itself."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return canonical_hash(value)


def _public_row(row: Mapping[str, Any]) -> JsonDict:
    """Keep vector metadata in the artifact while raw values remain in shards."""

    return {key: deepcopy(value) for key, value in row.items() if key != "vector_b64"}


def build_artifact_for_test(
    *, rows: Sequence[Mapping[str, Any]], reduction: Mapping[str, Any]
) -> JsonDict:
    """Build a small schema-complete artifact for cold-validator tests."""

    counts = _zero_invocation_counts()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "complete" if reduction.get("verdict_class") != "partial" else "partial",
        "run_date": RUN_DATE,
        "started_at_utc": "2026-09-20T00:00:00Z",
        "ended_at_utc": "2026-09-20T00:00:01Z",
        "clock_identity": {"boot_id": "fixture", "segment_id": "fixture"},
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": False,
        "invocation_counts": counts,
        "current_invocation_events": [],
        "inference_substrate": "fixture_final_layer_last_token_embeddings",
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate_details": {"generation_permitted": False},
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 1.0,
        "duration_components_s": {"load": 0.0, "forward": 0.0, "numeric": 0.0, "validation": 0.0},
        "phase_spans": [],
        "random_seed": {
            "projection": PROJECTION_SEED,
            "stream": STREAM_SEED,
            "fit": list(FIT_SEEDS),
            "resampling": RESAMPLING_SEED,
        },
        "source_artifact_hashes": {},
        "rows": [_public_row(row) for row in rows],
        "sample_size_budget": {**_coverage(rows), "max_evaluation_forwards": 800},
        "acceptance_gate_results": [],
        "gate_check_summary": {"passed": True, "failed_check": None},
        "verifier_is_oracle": False,
        "honest_verdict": reduction["honest_verdict"],
        "verdict_class": reduction["verdict_class"],
        "flagged_adversarial": reduction["flagged_adversarial"],
        "validation_receipts": [],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "embedding_capture_ready_score": reduction["embedding_capture_ready_score"],
        "representation_identity": {
            "layer": "final_model_layer",
            "pooling": "final_token_from_per_token_pooling_none",
            "dimension": rows[0].get("dimension") if rows else None,
            "model_hf_id": MODEL_HF_ID,
            "runner": "llama_cpp_python",
            "tokenizer": "embedded_gguf",
        },
        "feature_shards": [],
        "eligibility_rows": reduction["coverage_diagnostics"],
        "integrity_controls": {"passed": True},
        "required_validation_passed": True,
        "final_layer_scope_statement": "Final-layer last-token evidence does not replicate a cited mid-layer NF4 study.",
        "small_ebm_training": {"performed": False, "reason": "selector training is out of scope"},
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "numbered_e2e_applicable": False,
    }
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, counters, rows, principles, and reproducibility."""

    errors: list[str] = []
    errors.extend(
        f"missing_required_field:{field}" for field in REQUIRED_FIELDS if field not in value
    )
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("run_identity_invalid")
    specs = value.get("MODEL_SPECS")
    if not isinstance(specs, list) or not specs or specs[0].get("hf_id") != MODEL_HF_ID:
        errors.append("model_specs_invalid")
    counts = value.get("invocation_counts") or {}
    generation = counts.get("generation_calls") or {}
    if any(
        generation.get(state) != 0
        for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    ):
        errors.append("generation_counts_invalid")
    if value.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class_invalid")
    if value.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_invalid")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("verifier_is_oracle") is not False:
        errors.append("verifier_authority_invalid")
    if set(value.get("field_principles") or {}) != REQUIRED_FIELDS:
        errors.append("field_principles_invalid")
    if (
        value.get("final_layer_scope_statement")
        != "Final-layer last-token evidence does not replicate a cited mid-layer NF4 study."
    ):
        errors.append("surface_scope_invalid")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def _date_argument(value: str) -> str:
    """Reject a run date that does not match the frozen milestone contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public entrypoint and cold-validator modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


@dataclass
class NativeFinalLayerBackend:  # pragma: no cover - live native model boundary.
    """Expose tokenizer and final-token vectors without a generation method."""

    model_path: str
    gpu_index: int = 0
    _llm: Any = None

    def load(self) -> JsonDict:
        """Load the mandated GGUF with native per-token embeddings enabled."""

        import llama_cpp
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1,
            main_gpu=self.gpu_index,
            seed=STREAM_SEED,
            n_ctx=TOKEN_CEILING,
            n_batch=TOKEN_CEILING,
            n_ubatch=512,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE,
            logits_all=False,
            verbose=False,
        )
        if self._llm.pooling_type() != llama_cpp.LLAMA_POOLING_TYPE_NONE:
            raise CaptureInvalid("native per-token embedding surface unavailable")
        system_info = llama_cpp.llama_print_system_info()
        if isinstance(system_info, bytes):
            system_info = system_info.decode("utf-8", errors="replace")
        return {
            "runner": "llama_cpp_python",
            "runner_version": getattr(llama_cpp, "__version__", "unknown"),
            "runner_module": str(Path(llama_cpp.__file__).resolve()),
            "native_system_info": str(system_info),
            "pooling_type": "LLAMA_POOLING_TYPE_NONE",
            "embedding_mode": True,
            "generation_enabled": False,
            "logits_used_as_hidden_state": False,
            "n_gpu_layers": -1,
            "main_gpu": self.gpu_index,
        }

    def tokenize(self, text: str) -> list[int]:
        """Use the tokenizer embedded in the loaded GGUF."""

        if self._llm is None:
            raise CaptureInvalid("model not loaded")
        return list(self._llm.tokenize(text.encode("utf-8"), add_bos=True, special=False))

    def embed_final_token(self, text: str) -> list[float]:
        """Return only the final token from native final-layer per-token output."""

        if self._llm is None:
            raise CaptureInvalid("model not loaded")
        vectors = self._llm.embed(text, normalize=False, truncate=False)
        if not vectors or not isinstance(vectors[-1], list):
            raise CaptureInvalid("native per-token embedding surface unavailable")
        vector = [float(value) for value in vectors[-1]]
        _vector_bytes(vector)
        return vector

    def close(self) -> None:
        """Release native model memory before lease validation."""

        self._llm = None
        gc.collect()


def _utc_now() -> str:  # pragma: no cover - runtime clock.
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7452] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _phase_span(
    name: str, phase_started: float, run_started: float, **details: Any
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": name,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        **details,
    }


def _read_protocol_view(root: Path, kind: str) -> list[JsonDict]:  # pragma: no cover - runtime I/O.
    """Read one authenticated protocol view so labels stay unopened until needed."""

    raw = root / PROTOCOL_RAW_DIR
    manifest = _load_object(raw / "corpus_manifest.json")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_hash(unhashed):
        raise CaptureInvalid("protocol manifest hash mismatch")
    shard = next((row for row in manifest.get("shards", []) if row.get("kind") == kind), None)
    if not isinstance(shard, Mapping):
        raise CaptureInvalid(f"protocol {kind} shard missing")
    path = raw / str(shard["path"])
    payload = path.read_bytes()
    if "sha256:" + hashlib.sha256(payload).hexdigest() != shard.get("sha256"):
        raise CaptureInvalid(f"protocol {kind} shard hash mismatch")
    rows = [json.loads(line) for line in payload.splitlines()]
    if len(rows) != shard.get("rows"):
        raise CaptureInvalid(f"protocol {kind} row count mismatch")
    return rows


def _gpu_snapshot() -> JsonDict:  # pragma: no cover - host NVIDIA boundary.
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    rows = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 5:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "uuid": parts[1],
                        "name": parts[2],
                        "memory_used_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                    }
                )
    return {"command": command, "exit_code": completed.returncode, "gpus": rows}


def _hash_with_progress(path: Path, started: float) -> str:  # pragma: no cover - large model I/O.
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    last = time.monotonic()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
            done += len(chunk)
            if time.monotonic() - last >= 60:
                _progress(
                    started, "preconditions", "model_hash_outstanding", bytes=done, total=total
                )
                last = time.monotonic()
    return "sha256:" + digest.hexdigest()


def _development_probes(
    backend: NativeFinalLayerBackend, started: float
) -> JsonDict:  # pragma: no cover
    probes = (
        "A short neutral representation probe.",
        "Numbers 2, 3, and 5 form a deterministic input.",
        "SOURCE\nA compact source.\n\nRESPONSE\nA compact response.",
        "Unicode remains input evidence: café.",
    )
    first: list[JsonDict] = []
    second: list[JsonDict] = []
    for pass_index, target in enumerate((first, second), start=1):
        for index, text in enumerate(probes):
            call_started = time.monotonic()
            vector = backend.embed_final_token(text)
            elapsed = time.monotonic() - call_started
            payload = _vector_bytes(vector)
            target.append(
                {
                    "probe": index,
                    "pass": pass_index,
                    "token_count": len(backend.tokenize(text)),
                    "dimension": len(vector),
                    "finite": all(math.isfinite(value) for value in vector),
                    "vector_hash": "sha256:" + hashlib.sha256(payload).hexdigest(),
                    "duration_s": elapsed,
                    "within_call_ceiling": elapsed <= CALL_CEILING_S,
                }
            )
            _progress(
                started,
                "development",
                "forward_complete",
                unit=(pass_index - 1) * 4 + index + 1,
                total=MAX_DEVELOPMENT_FORWARDS,
                duration_s=round(elapsed, 3),
            )
    dimensions = {row["dimension"] for row in [*first, *second]}
    return {
        "rows": [*first, *second],
        "dimension": next(iter(dimensions)) if len(dimensions) == 1 else None,
        "finite_values": all(row["finite"] for row in [*first, *second]),
        "deterministic_replay": [row["vector_hash"] for row in first]
        == [row["vector_hash"] for row in second],
        "pooling": "final_token_from_per_token_pooling_none",
        "passed": len(dimensions) == 1
        and all(row["finite"] and row["within_call_ceiling"] for row in [*first, *second])
        and [row["vector_hash"] for row in first] == [row["vector_hash"] for row in second],
    }


def _run_forwards(
    predictors: Sequence[Mapping[str, Any]],
    eligibility: Sequence[Mapping[str, Any]],
    backend: NativeFinalLayerBackend,
    started: float,
) -> tuple[list[JsonDict], float]:  # pragma: no cover - live native model boundary.
    by_key = {str(row["row_key"]): row for row in predictors}
    cells = [deepcopy(dict(row)) for row in eligibility]
    by_group: dict[str, list[JsonDict]] = defaultdict(list)
    for row in cells:
        by_group[str(row["group_id"])].append(row)
    forward_started = time.monotonic()
    attempted = 0
    completed_groups = 0
    last_heartbeat = time.monotonic()
    output: list[JsonDict] = []
    for group_id in sorted(
        by_group, key=lambda group: min(int(row["seal_order"]) for row in by_group[group])
    ):
        group = by_group[group_id]
        pair_eligible = len(group) == 2 and all(row.get("eligible") is True for row in group)
        for cell in sorted(group, key=lambda row: ARMS.index(str(row["arm"]))):
            if not pair_eligible:
                if cell.get("eligible") is True:
                    cell["status"] = "excluded_paired_group"
                    cell["eligible"] = False
                    cell["unstarted"] = False
                output.append(cell)
                continue
            if (
                attempted >= MAX_EVALUATION_FORWARDS
                or time.monotonic() - forward_started >= FORWARD_BUDGET_S
            ):
                cell["status"] = "censored_forward_budget"
                cell["censored"] = True
                cell["unstarted"] = False
                output.append(cell)
                continue
            predictor = by_key[str(cell["row_key"])]
            text = _representation_text(predictor, str(cell["arm"]))
            call_started = time.monotonic()
            attempted += 1
            try:
                vector = backend.embed_final_token(text)
                call_elapsed = time.monotonic() - call_started
                if call_elapsed > CALL_CEILING_S:
                    raise TimeoutError("forward_call_ceiling_exceeded")
                row = vector_row(cell, vector, token_count=int(cell["token_count"]))
                row["forward_duration_s"] = call_elapsed
                row["seal_order"] = cell["seal_order"]
                output.append(row)
            except Exception as exc:
                cell.update(
                    {
                        "status": f"failed:{type(exc).__name__}:{exc}",
                        "attempted": True,
                        "completed": False,
                        "failed": True,
                        "censored": False,
                        "unstarted": False,
                        "forward_duration_s": time.monotonic() - call_started,
                    }
                )
                output.append(cell)
        completed_groups += 1
        if completed_groups % GROUPS_PER_SHARD == 0 or time.monotonic() - last_heartbeat >= 60:
            _progress(
                started,
                "capture",
                "groups_complete",
                completed_groups=completed_groups,
                total_groups=len(by_group),
                attempted_forwards=attempted,
                forward_elapsed_s=round(time.monotonic() - forward_started, 3),
            )
            last_heartbeat = time.monotonic()
    return sorted(
        output, key=lambda row: int(row.get("seal_order", 0))
    ), time.monotonic() - forward_started


def _terminal_commands(
    candidate: Path,
) -> list[PlannedCommand]:  # pragma: no cover - subprocess plan.
    python = ".venv/bin/python"
    relative = candidate.relative_to(REPO_ROOT).as_posix()
    commands = (
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate-artifact",
                relative,
            ),
            "candidate_artifact",
        ),
        CommandSpec(
            "independent_cold_reducer",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                relative,
            ),
            "raw_rows_and_shards",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", relative),
            "candidate_artifact",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", relative),
            "candidate_artifact",
        ),
    )
    return [PlannedCommand(command, "terminal_reader", True) for command in commands]


def _runtime_artifact(  # pragma: no cover - runtime assembly.
    *,
    started: float,
    started_at: str,
    spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    model_hash: str | None,
    model_invoked: bool,
    counts: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
    representation: Mapping[str, Any],
    controls: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_passed: bool,
    durations: Mapping[str, Any],
    blocked_reason: str | None = None,
) -> JsonDict:
    reduction = (
        reduce_capture(
            rows,
            controls_passed=controls.get("passed") is True,
            validation_passed=validation_passed,
        )
        if not blocked_reason
        else {
            "embedding_capture_ready_score": 0,
            "promotion_score": 0,
            "verdict_class": "null" if model_invoked else "blocked",
            "honest_verdict": (
                f"complete_null_{blocked_reason}" if model_invoked else blocked_reason
            ),
            "flagged_adversarial": False,
            "coverage_diagnostics": _coverage(rows),
        }
    )
    gates = [deepcopy(dict(row)) for row in preconditions]
    if blocked_reason:
        last_event = events[-1] if events else {}
        gates.append(
            _gate(
                blocked_reason,
                "validity",
                MODEL_HF_ID,
                str(model_spec.get("model_path") or "missing"),
                "native_final_layer_embedding_surface",
                "==",
                "available",
                last_event.get("error", "unavailable"),
                False,
                "The mandated native final-layer surface must load before vector controls apply.",
            )
        )
    gates.extend(
        (
            _gate(
                "integrity_controls",
                "validity",
                EXPERIMENT_ID,
                "feature_shards",
                "passed",
                "==",
                True if not blocked_reason else "not_applicable_surface_unavailable",
                controls.get("passed")
                if not blocked_reason
                else "not_applicable_surface_unavailable",
                controls.get("passed") is True if not blocked_reason else True,
                "Raw vector identity and dimension mutations must fail closed.",
            ),
            _gate(
                "required_validation",
                "validity",
                EXPERIMENT_ID,
                "validation_receipts",
                "passed",
                "==",
                True,
                validation_passed,
                validation_passed,
                "All scoped and terminal readers must pass.",
            ),
            _gate(
                "promotion_forbidden",
                "benefit",
                EXPERIMENT_ID,
                RESULT_PATH.as_posix(),
                "promotion_score",
                "==",
                0,
                0,
                True,
                "This capture cannot authorize rollout.",
            ),
        )
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": (
            "complete_null"
            if blocked_reason and model_invoked
            else (
                "blocked"
                if blocked_reason
                else ("complete" if reduction["verdict_class"] != "partial" else "partial")
            )
        ),
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "clock_identity": {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(),
            "segment_id": canonical_hash({"pid": os.getpid(), "started_at_utc": started_at}),
            "started_monotonic_ns": int(started * 1_000_000_000),
            "ended_monotonic_ns": time.monotonic_ns(),
        },
        "preconditions_checked": list(gates[: len(preconditions)]),
        "MODEL_SPECS": [
            {
                **deepcopy(dict(MODEL_SPECS[0])),
                **deepcopy(dict(model_spec)),
                "model_sha256": model_hash,
            }
        ],
        "model_invoked": model_invoked,
        "invocation_counts": deepcopy(dict(counts)),
        "current_invocation_events": deepcopy(list(events)),
        "inference_substrate": "blocked_no_run"
        if not model_invoked
        else "native_gguf_final_layer_last_token_embeddings_no_generation",
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate_details": {
            "model": deepcopy(dict(model_spec)),
            "surface": deepcopy(dict(representation)),
            "generation_permitted": False,
            "call_ceiling_s": CALL_CEILING_S,
            "forward_budget_s": FORWARD_BUDGET_S,
        },
        "execution_venue": EXECUTION_VENUE,
        "duration_s": time.monotonic() - started,
        "duration_components_s": deepcopy(dict(durations)),
        "phase_spans": deepcopy(list(spans)),
        "random_seed": {
            "projection": PROJECTION_SEED,
            "stream": STREAM_SEED,
            "fit": list(FIT_SEEDS),
            "resampling": RESAMPLING_SEED,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [_public_row(row) for row in rows],
        "sample_size_budget": {
            **_coverage(rows),
            "max_evaluation_forwards": MAX_EVALUATION_FORWARDS,
            "max_development_forwards": MAX_DEVELOPMENT_FORWARDS,
            "forward_stop_s": FORWARD_BUDGET_S,
            "call_ceiling_s": CALL_CEILING_S,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": reduction["honest_verdict"],
        "verdict_class": reduction["verdict_class"],
        "flagged_adversarial": reduction["flagged_adversarial"],
        "validation_receipts": deepcopy(list(validation_receipts)),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "embedding_capture_ready_score": reduction["embedding_capture_ready_score"],
        "representation_identity": deepcopy(dict(representation)),
        "feature_shards": deepcopy(list(shards)),
        "eligibility_rows": reduction["coverage_diagnostics"],
        "integrity_controls": deepcopy(dict(controls)),
        "required_validation_passed": validation_passed,
        "final_layer_scope_statement": "Final-layer last-token evidence does not replicate a cited mid-layer NF4 study.",
        "small_ebm_training": {
            "performed": False,
            "reason": "selector training and held-out labels are out of scope",
        },
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "numbered_e2e_applicable": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _run_validation(
    root: Path, raw_dir: Path, started: float
) -> tuple[list[JsonDict], bool, float]:  # pragma: no cover
    private = Path(tempfile.mkdtemp(prefix="exp7452-validation-", dir="/tmp"))
    commands = build_command_plan(root, AFFECTED_MANIFEST, private)
    errors = validate_command_plan(root, AFFECTED_MANIFEST, commands)
    _progress(started, "validation", "before_affected_subprocesses", plan_errors=len(errors))
    phase_started = time.monotonic()
    receipts = (
        []
        if errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    )
    reduced = reduce_affected_receipts(root, AFFECTED_MANIFEST, receipts)
    _progress(started, "validation", "after_affected_subprocesses", passed=reduced["passed"])
    return receipts, bool(reduced["passed"] and not errors), time.monotonic() - phase_started


def run_experiment(
    root: Path, run_date: str
) -> JsonDict:  # pragma: no cover - orchestration and live I/O.
    """Run the bounded capture and publish only after all terminal readers pass."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date must be {RUN_DATE}")
    started = time.monotonic()
    started_at = _utc_now()
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []
    events: list[JsonDict] = []
    counts = _zero_invocation_counts()
    durations = {"load": 0.0, "forward": 0.0, "numeric": 0.0, "validation": 0.0}

    phase_started = time.monotonic()
    _progress(started, "preconditions", "start")
    preconditions, source_hashes = collect_upstream_preconditions(root)
    if not all(row["passed"] for row in preconditions):
        spans.append(_phase_span("preconditions", phase_started, started))
        artifact = _runtime_artifact(
            started=started,
            started_at=started_at,
            spans=spans,
            preconditions=preconditions,
            source_hashes=source_hashes,
            model_spec={},
            model_hash=None,
            model_invoked=False,
            counts=counts,
            events=events,
            rows=[],
            shards=[],
            representation={},
            controls={},
            validation_receipts=[],
            validation_passed=False,
            durations=durations,
            blocked_reason="blocked_external_prerequisite",
        )
        atomic_json(root / RESULT_PATH, artifact)
        return artifact
    if os.environ.get("CARNOT_FORCE_LIVE") != "1":
        raise RuntimeError("CARNOT_FORCE_LIVE=1 is required")
    resolved = cached_current_model()
    model_path = Path(str((resolved or {}).get("model_path", "")))
    if not resolved or resolved.get("hf_id") != MODEL_HF_ID or not model_path.is_file():
        blocked = _gate(
            "current_model_cached",
            "validity",
            "cached_current_model",
            str(model_path),
            "hf_id",
            "==",
            MODEL_HF_ID,
            (resolved or {}).get("hf_id", "missing"),
            False,
            "The headline GGUF must already exist locally; no substitute is permitted.",
        )
        preconditions.append(blocked)
        spans.append(_phase_span("preconditions", phase_started, started))
        artifact = _runtime_artifact(
            started=started,
            started_at=started_at,
            spans=spans,
            preconditions=preconditions,
            source_hashes=source_hashes,
            model_spec=resolved or {},
            model_hash=None,
            model_invoked=False,
            counts=counts,
            events=events,
            rows=[],
            shards=[],
            representation={},
            controls={},
            validation_receipts=[],
            validation_passed=False,
            durations=durations,
            blocked_reason="blocked_no_run_current_model_cache_miss",
        )
        atomic_json(root / RESULT_PATH, artifact)
        return artifact
    _progress(started, "preconditions", "before_model_hash", bytes=model_path.stat().st_size)
    model_hash = _hash_with_progress(model_path, started)
    _progress(started, "preconditions", "after_model_hash", sha256=model_hash)
    source_hashes["model_file"] = {
        "path": str(model_path),
        "sha256": model_hash,
        "original_flagged_adversarial": None,
    }
    gpu_before = _gpu_snapshot()
    gpu = next(
        (row for row in gpu_before["gpus"] if row["index"] == int(resolved.get("gpu", 0))), None
    )
    if gpu is None:
        raise RuntimeError("configured GPU is unavailable")
    lease = lease_api.GpuLease.acquire(
        runtime_dir=LEASE_RUNTIME_DIR,
        task_id=TASK_ID,
        device_uuid=str(gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(gpu["memory_used_mb"]),
        ttl_s=3_600.0,
    )
    lease.transition("admitted")
    lease.transition("loading")
    spans.append(_phase_span("preconditions", phase_started, started, lease_id=lease.lease_id))
    _progress(started, "preconditions", "complete", lease_id=lease.lease_id)

    backend = NativeFinalLayerBackend(str(model_path), int(resolved.get("gpu", 0)))
    rows: list[JsonDict] = []
    shards: list[JsonDict] = []
    representation: JsonDict = {"lease": lease.owner_receipt()}
    controls: JsonDict = {}
    blocked_reason: str | None = None
    try:
        phase_started = time.monotonic()
        load_started = time.monotonic()
        counts["model_loads"]["attempted"] += 1
        events.append(
            {
                "operation": "model_load",
                "status": "attempted",
                "started_at_utc": _utc_now(),
                "monotonic_ns": time.monotonic_ns(),
            }
        )
        _progress(started, "model_load", "before_model_load", model=MODEL_HF_ID)
        try:
            loader = backend.load()
            durations["load"] = time.monotonic() - load_started
            counts["model_loads"]["completed"] += 1
            events[-1].update(
                {"status": "completed", "duration_s": durations["load"], "ended_at_utc": _utc_now()}
            )
            gpu_resident = _gpu_snapshot()
            resident = next(
                row for row in gpu_resident["gpus"] if row["index"] == int(resolved.get("gpu", 0))
            )
            lease.transition("resident", vram_mb=int(resident["memory_used_mb"]))
            lease.transition("inferencing")
            _progress(
                started, "model_load", "after_model_load", duration_s=round(durations["load"], 3)
            )
            representation.update(loader)
        except Exception as exc:
            durations["load"] = time.monotonic() - load_started
            counts["model_loads"]["failed"] += 1
            events[-1].update(
                {
                    "status": "failed",
                    "duration_s": durations["load"],
                    "error": f"{type(exc).__name__}:{exc}",
                    "ended_at_utc": _utc_now(),
                }
            )
            blocked_reason = "blocked_embedding_surface_unavailable"
            protocol_artifact = _load_object(root / PROTOCOL_PATH)
            rows = [
                deepcopy(dict(row))
                for row in protocol_artifact.get("rows", [])
                if isinstance(row, Mapping)
            ]
            _progress(
                started, "model_load", "after_model_load", status="failed", error=type(exc).__name__
            )
        spans.append(_phase_span("model_load", phase_started, started))

        if blocked_reason is None:
            phase_started = time.monotonic()
            _progress(started, "development", "start", planned_forwards=MAX_DEVELOPMENT_FORWARDS)
            development = _development_probes(backend, started)
            counts["forward_calls"]["attempted"] += MAX_DEVELOPMENT_FORWARDS
            counts["forward_calls"]["completed"] += MAX_DEVELOPMENT_FORWARDS
            durations["forward"] += sum(float(row["duration_s"]) for row in development["rows"])
            representation.update(
                {
                    "layer": "final_model_layer",
                    "pooling": "final_token_from_per_token_pooling_none",
                    "dimension": development["dimension"],
                    "model_hf_id": MODEL_HF_ID,
                    "model_sha256": model_hash,
                    "tokenizer": "embedded_gguf",
                    "development": development,
                    "cuda_offload": {
                        "requested_n_gpu_layers": -1,
                        "gpu_before": gpu_before,
                        "gpu_resident": _gpu_snapshot(),
                    },
                }
            )
            if not development["passed"]:
                blocked_reason = "blocked_embedding_surface_unavailable"
            spans.append(
                _phase_span(
                    "development", phase_started, started, completed=MAX_DEVELOPMENT_FORWARDS
                )
            )
            _progress(started, "development", "complete", passed=development["passed"])

        if blocked_reason is None:
            phase_started = time.monotonic()
            _progress(started, "eligibility", "start")
            predictors = _read_protocol_view(root, "predictor")
            eligibility = build_eligibility_rows(predictors, backend)
            sealed_eligibility_hash = canonical_hash(eligibility)
            _progress(
                started,
                "eligibility",
                "labels_still_unread",
                cells=len(eligibility),
                seal=sealed_eligibility_hash,
            )
            evaluators = _read_protocol_view(root, "evaluator")
            eligibility = bind_response_identities(eligibility, evaluators)
            spans.append(
                _phase_span(
                    "eligibility",
                    phase_started,
                    started,
                    cells=len(eligibility),
                    seal=sealed_eligibility_hash,
                )
            )
            _progress(
                started,
                "eligibility",
                "complete",
                eligible=sum(row["eligible"] for row in eligibility),
            )

            phase_started = time.monotonic()
            _progress(started, "capture", "before_forward_loop", cells=len(eligibility))
            numeric_started = time.monotonic()
            rows, forward_elapsed = _run_forwards(predictors, eligibility, backend, started)
            durations["forward"] += forward_elapsed
            evaluation_attempted = sum(row.get("attempted") is True for row in rows)
            evaluation_completed = sum(row.get("completed") is True for row in rows)
            evaluation_failed = sum(row.get("failed") is True for row in rows)
            counts["forward_calls"]["attempted"] += evaluation_attempted
            counts["forward_calls"]["completed"] += evaluation_completed
            counts["forward_calls"]["failed"] += evaluation_failed
            counts["forward_calls"]["cancelled"] += sum(row.get("censored") is True for row in rows)
            shards = write_feature_shards(raw_dir, rows)
            durations["numeric"] = time.monotonic() - numeric_started - forward_elapsed
            reloaded = reload_feature_shards(raw_dir, shards)
            controls = run_integrity_controls(reloaded)
            representation["eligibility_seal_sha256"] = sealed_eligibility_hash
            spans.append(
                _phase_span(
                    "capture",
                    phase_started,
                    started,
                    completed=evaluation_completed,
                    shards=len(shards),
                )
            )
            _progress(
                started,
                "capture",
                "after_forward_loop",
                completed=evaluation_completed,
                failed=evaluation_failed,
                shards=len(shards),
            )
    finally:
        phase_started = time.monotonic()
        _progress(started, "model_unload", "before_model_unload")
        backend.close()
        gc.collect()
        gpu_after = _gpu_snapshot()
        after = next(
            (row for row in gpu_after["gpus"] if row["index"] == int(resolved.get("gpu", 0))), gpu
        )
        try:
            if lease.document.get("phase") == "inferencing":
                lease.transition("unloading")
                lease.transition(
                    "validating",
                    vram_mb=int(after["memory_used_mb"]),
                    exit_code=0,
                    unload_observed=True,
                )
                lease.transition("terminal_complete")
                representation["lease_release"] = lease.release()
            else:
                lease.transition("terminal_blocked")
                representation["lease_release"] = lease.release()
        except Exception as exc:
            representation["lease_release_error"] = f"{type(exc).__name__}:{exc}"
            lease.close()
        spans.append(_phase_span("model_unload", phase_started, started))
        _progress(started, "model_unload", "after_model_unload")

    phase_started = time.monotonic()
    affected_receipts, affected_passed, validation_elapsed = _run_validation(root, raw_dir, started)
    durations["validation"] += validation_elapsed
    spans.append(_phase_span("affected_validation", phase_started, started))

    candidate = _runtime_artifact(
        started=started,
        started_at=started_at,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        model_spec=resolved,
        model_hash=model_hash,
        model_invoked=True,
        counts=counts,
        events=events,
        rows=rows,
        shards=shards,
        representation=representation,
        controls=controls,
        validation_receipts=affected_receipts,
        validation_passed=affected_passed,
        durations=durations,
        blocked_reason=blocked_reason,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    phase_started = time.monotonic()
    _progress(started, "terminal_validation", "before_terminal_subprocesses")
    terminal_receipts = run_categorized_commands(
        root, _terminal_commands(candidate_path), log_dir=raw_dir / "validation/terminal"
    )
    terminal_passed = all(row["passed"] for row in terminal_receipts)
    critical = any("CRITICAL" in str(row.get("output_tail", "")) for row in terminal_receipts)
    terminal_passed = terminal_passed and not critical
    durations["validation"] += time.monotonic() - phase_started
    spans.append(_phase_span("terminal_validation", phase_started, started))
    _progress(
        started,
        "terminal_validation",
        "after_terminal_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    final = _runtime_artifact(
        started=started,
        started_at=started_at,
        spans=spans,
        preconditions=preconditions,
        source_hashes=source_hashes,
        model_spec=resolved,
        model_hash=model_hash,
        model_invoked=True,
        counts=counts,
        events=events,
        rows=rows,
        shards=shards,
        representation=representation,
        controls=controls,
        validation_receipts=[*affected_receipts, *terminal_receipts],
        validation_passed=affected_passed and terminal_passed,
        durations=durations,
        blocked_reason=blocked_reason,
    )
    final_errors = validate_artifact(final)
    if final_errors:
        raise CaptureInvalid("final artifact invalid:" + ",".join(final_errors))
    _progress(started, "publish", "before_atomic_publish", path=RESULT_PATH.as_posix())
    atomic_json(root / RESULT_PATH, final)
    _progress(started, "publish", "after_atomic_publish", path=RESULT_PATH.as_posix())
    return final


def _independent_candidate_reduce(
    root: Path, candidate: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover
    blocked = candidate.get("verdict_class") == "blocked"
    surface_unavailable = (candidate.get("gate_check_summary") or {}).get(
        "failed_check"
    ) == "blocked_embedding_surface_unavailable"
    shards = candidate.get("feature_shards") or []
    loaded = reload_feature_shards(root / RAW_DIR, shards) if shards else []
    controls = run_integrity_controls(loaded) if loaded else {"passed": False}
    rows = candidate.get("rows") or []
    reduction = reduce_capture(
        rows,
        controls_passed=controls.get("passed") is True,
        validation_passed=candidate.get("required_validation_passed") is True,
    )
    return {
        "passed": not validate_artifact(candidate)
        and (
            blocked
            or surface_unavailable
            or (
                controls.get("passed") is True
                and reduction["embedding_capture_ready_score"]
                == candidate.get("embedding_capture_ready_score")
            )
        ),
        "embedding_capture_ready_score": reduction["embedding_capture_ready_score"],
        "verdict_class": reduction["verdict_class"],
        "integrity_controls": controls,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    args = parse_args(argv)
    if args.validate_artifact:
        value = _load_object(REPO_ROOT / args.validate_artifact)
        errors = validate_artifact(value)
        print(json.dumps({"passed": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce:
        value = _load_object(REPO_ROOT / args.independent_reduce)
        reduced = _independent_candidate_reduce(REPO_ROOT, value)
        print(json.dumps(reduced, sort_keys=True), flush=True)
        return int(reduced["passed"] is not True)
    run_experiment(REPO_ROOT, args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
