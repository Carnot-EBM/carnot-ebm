"""Collect raw fixed-sequence scores for the frozen Exp6867 stream.

Spec refs: REQ-INFERENCE-6868 and SCENARIO-INFERENCE-6868-*.

The worker receives token IDs only. It never receives labels, group metadata,
or score meaning. The terminal artifact records raw measurements without an
effect reduction or scientific claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
from urllib import error

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6850_three_family_scoring_admission_canary import (
    _choose_free_ports,
    _compute_apps,
    _cuda_token_scoring,
    _gpu_inventory,
    _gpu_snapshot,
)
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess, port_is_free
from carnot.inference.llama_server_supervisor import read_process_identity
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = Path("openspec/capabilities/llm-ebm-inference/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6868_three_family_semantic_scoring_stream_v2.py"
)
WRAPPER_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6868_three_family_semantic_scoring_stream_v2.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6868_three_family_semantic_scoring_stream_v2.py"
)
PROCESS_MODULE_RELATIVE_PATH = Path("python/carnot/inference/llama_cpp_process.py")
EXP6867_RELATIVE_PATH = Path(
    "results/experiment_6867_tokenizer_aware_semantic_preregistration_v2.json"
)
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6868_three_family_semantic_scoring_stream_v2.json"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6868_three_family_semantic_scoring_stream_v2.checkpoint.json"
)
CALIBRATION_SIDECAR_RELATIVE_PATH = Path(
    "results/sidecars/experiment_6868_semantic_scoring_calibration.json"
)
HELD_SIDECAR_RELATIVE_PATH = Path(
    "results/sidecars/experiment_6868_semantic_scoring_held.json"
)

SCHEMA = "carnot.experiment_6868.three_family_semantic_scoring_stream_v2.v1"
INFERENCE_SUBSTRATE = "live local llama.cpp CUDA forced-sequence scoring"
RUN_DATE = "20260902"
RANDOM_SEED = 6868
EXPECTED_EXP6867_SHA256 = (
    "sha256:31f445cf96627221d286db6859392c5a19eef211b703dc91dc663b3d3e0cc480"
)
MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_FAMILIES = {
    MODEL_SPECS[0]: "qwen_moe",
    MODEL_SPECS[1]: "gemma_dense",
    MODEL_SPECS[2]: "gemma_moe",
}
BLOCKED_VERDICT = "complete_blocked_three_family_semantic_scoring_stream_v2"
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
CONTEXT_LENGTH = 4096
GROUP_BATCH_SIZE = 2
LEASE_TTL_S = 900.0
HEALTH_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 300.0
DISK_FLOOR_BYTES = 512 * 1024 * 1024
ROUND_DIGITS = 10
LEASE_RUNTIME_DIR = Path("/tmp/carnot-gpu-leases")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "model_specs",
    "models_used",
    "missing_model_manifest",
    "model_artifact_hashes",
    "tokenizer_receipts",
    "process_receipts",
    "accelerator_samples",
    "lease_receipts",
    "rows",
    "raw_token_sidecars",
    "calibration_score_manifest",
    "sealed_held_score_manifest",
    "failed_cell_manifest",
    "checkpoint_manifest",
    "teardown_receipts",
    "generated_answer_count",
    "held_label_access_count",
    "scientific_effect_claimed",
    "random_seed",
    "reproducibility_checksum",
    "semantic_contrast_stream_v2_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "Frozen hashes and live resources fail closed before scoring.",
    "inference_substrate": "The exact local scoring path separates scores from generation.",
    "duration_s": "Wall time exposes skipped live work.",
    "model_specs": "The three exact model identities prevent substitution.",
    "models_used": "Only models with terminal cell records count as used.",
    "missing_model_manifest": "Missing model work stays explicit.",
    "model_artifact_hashes": "Exact GGUF bytes bind every score.",
    "tokenizer_receipts": "Canonical native tokenizer hashes bind frozen token IDs.",
    "process_receipts": "PID and ownership identity protect unrelated processes.",
    "accelerator_samples": "GPU UUID and VRAM samples prove CUDA residency.",
    "lease_receipts": "A bounded owner lease isolates each model phase.",
    "rows": "Each row is one raw model, group, transform, and sequence score.",
    "raw_token_sidecars": "Split sidecar hashes preserve token-level evidence.",
    "calibration_score_manifest": "Calibration scores remain separate from held scores.",
    "sealed_held_score_manifest": "Held raw scores stay sealed and unreduced.",
    "failed_cell_manifest": "Failures and timeouts remain visible without imputation.",
    "checkpoint_manifest": "Atomic hashes make partial restart safe.",
    "teardown_receipts": "Owned exit and port release prevent contamination.",
    "generated_answer_count": "Zero proves that no answer generation occurred.",
    "held_label_access_count": "Zero proves that held labels stayed sealed.",
    "scientific_effect_claimed": "False keeps completion separate from an effect claim.",
    "random_seed": "The frozen seed makes ordering reproducible.",
    "reproducibility_checksum": "One digest binds inputs and raw outputs.",
    "semantic_contrast_stream_v2_complete_score": "Completeness counts receipts, not effects.",
    "gate_check_summary": "The first failed check keeps exact expected and observed values.",
    "verifier_is_oracle": "Raw model scores do not define correctness.",
    "verdict_class": "A closed vocabulary keeps terminal states consistent.",
    "honest_verdict": "A complete_ prefix marks a terminal artifact.",
    "schema": "The schema identifies the artifact contract.",
    "experiment_id": "The experiment number prevents artifact confusion.",
    "run_date": "The requested date binds this execution.",
    "status": "A compact terminal state supports automation.",
    "spec_refs": "Requirement anchors connect spec, tests, and code.",
    "source_artifact_hashes": "Source hashes bind preregistration and code.",
    "expected_cell_count": "The frozen count makes missing work visible.",
    "scored_cell_count": "The raw row count supports completeness checks.",
    "explicit_failure_cell_count": "The failure count supports no-imputation checks.",
    "method_limits": "The artifact states that raw scores have no semantic conclusion.",
    "result_path": "The declared path detects accidental publication elsewhere.",
}


class SemanticScoringError(RuntimeError):
    """A stable error for unsafe requests or malformed raw receipts."""


def canonical_json(value: Any) -> str:
    """Serialize JSON consistently for receipt and checkpoint hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a canonical JSON value."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:  # pragma: no cover - large live files.
    """Hash a file in chunks so GGUF bytes do not enter one memory buffer."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:  # pragma: no cover - live file boundary.
    """Read one JSON object and reject other JSON shapes."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SemanticScoringError(f"json_object_required:{path}")
    return dict(value)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Replace one JSON file atomically so restart cannot read partial bytes."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def _gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [dict(row) for row in checks]
    failed = [row for row in rows if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "passed": first is None,
        "checks": rows,
        "failed_check": first.get("check") if first else None,
        "failed_checks": [row.get("check") for row in failed],
        "expected": first.get("expected") if first else True,
        "observed": first.get("observed") if first else True,
    }


def _frozen_model_bindings(preregistration: Mapping[str, Any]) -> JsonDict:
    artifacts = dict(preregistration.get("model_artifact_hashes") or {})
    tokenizers = {
        str(row.get("hf_id")): row
        for row in preregistration.get("tokenizer_receipts") or []
        if isinstance(row, Mapping)
    }
    return {
        model: {
            "model_sha256": dict(artifacts.get(model) or {}).get("sha256"),
            "canonical_tokenizer_payload_hash": dict(tokenizers.get(model) or {}).get(
                "canonical_tokenizer_payload_sha256"
            ),
        }
        for model in MODEL_SPECS
    }


def _observed_model_bindings(resolved_models: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(row.get("hf_id")): {
            "model_sha256": row.get("model_sha256"),
            "canonical_tokenizer_payload_hash": row.get(
                "canonical_tokenizer_payload_hash"
            ),
        }
        for row in resolved_models
    }


def evaluate_preconditions(
    *,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    expected_preregistration_sha256: str,
    resolved_models: Sequence[Mapping[str, Any]],
    cached_sota_pair_ids: Sequence[str],
    cuda_token_scoring: bool,
    sufficient_disk: bool,
    free_task_ports: bool,
    bounded_lease: bool,
) -> JsonDict:
    """Compare each frozen and live gate without truthy coercion."""

    observed_ids = [str(row.get("hf_id")) for row in resolved_models]
    cached_ids = [str(value) for value in cached_sota_pair_ids]
    cached_pair_ok = len(cached_ids) == 2 and all(value in MODEL_SPECS for value in cached_ids)
    checks = [
        _gate_check(
            "semantic_contrast_preregistration_v2_ready_score",
            1,
            preregistration.get("semantic_contrast_preregistration_v2_ready_score"),
        ),
        _gate_check(
            "unchanged_exp6867_artifact",
            expected_preregistration_sha256,
            preregistration_sha256,
        ),
        _gate_check("all_three_exact_model_specs", list(MODEL_SPECS), observed_ids),
        _gate_check("cached_sota_pair", True, cached_pair_ok),
        _gate_check(
            "unchanged_model_and_tokenizer_hashes",
            _frozen_model_bindings(preregistration),
            _observed_model_bindings(resolved_models),
        ),
        _gate_check("live_cuda_token_scoring", True, bool(cuda_token_scoring)),
        _gate_check("sufficient_disk", True, bool(sufficient_disk)),
        _gate_check("free_task_ports", True, bool(free_task_ports)),
        _gate_check("bounded_task_gpu_lease", True, bool(bounded_lease)),
    ]
    return _gate_summary(checks)


def _slot_tokens(mapping: Mapping[str, Any], slot: int, field: str) -> list[int]:
    values = mapping.get(f"slot_{slot}")
    if not isinstance(values, list) or not values or not all(type(value) is int for value in values):
        raise SemanticScoringError(f"{field}_missing:slot_{slot}")
    return [int(value) for value in values]


def expected_work_items(cells: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expand accepted Exp6867 cells without copying any labels or commitments."""

    work: list[JsonDict] = []
    for cell in cells:
        if cell.get("accepted") is not True:
            continue
        model = str(cell.get("model_hf_id"))
        if model not in MODEL_SPECS:
            raise SemanticScoringError(f"unexpected_model:{model}")
        prompts = cell.get("paired_prompt_token_ids")
        candidates = cell.get("candidate_token_ids")
        orders = cell.get("presentation_order")
        hashes = cell.get("candidate_sequence_sha256")
        if not all(isinstance(value, Mapping) for value in (prompts, candidates, orders, hashes)):
            raise SemanticScoringError("frozen_sequence_mapping_missing")
        candidate_ids = list(cell.get("candidate_ids") or [])
        if len(candidate_ids) != 2:
            raise SemanticScoringError("candidate_ids_missing")
        for transform in ("base", "label_swap"):
            prompt_values = prompts.get(transform)
            order_values = orders.get(transform)
            if (
                not isinstance(prompt_values, list)
                or not prompt_values
                or not all(type(value) is int for value in prompt_values)
            ):
                raise SemanticScoringError(f"prompt_token_ids_missing:{transform}")
            if not isinstance(order_values, list) or sorted(order_values) != [0, 1]:
                raise SemanticScoringError(f"presentation_order_invalid:{transform}")
            prompt_ids = [int(value) for value in prompt_values]
            for position, raw_slot in enumerate(order_values):
                slot = int(raw_slot)
                candidate_ids_for_slot = _slot_tokens(candidates, slot, "candidate_token_ids")
                full_sequence_hash = sha256_json(prompt_ids + candidate_ids_for_slot)
                identity_payload = {
                    "model_hf_id": model,
                    "semantic_group_identity": cell.get("semantic_group_identity"),
                    "split": cell.get("split"),
                    "sequence_identity": cell.get("sequence_identity"),
                    "nuisance_transform_identity": transform,
                    "presentation_position": position,
                    "candidate_slot": slot,
                    "full_sequence_sha256": full_sequence_hash,
                }
                work.append(
                    {
                        "cell_identity": sha256_json(identity_payload),
                        "model_hf_id": model,
                        "model_family": cell.get("model_family") or MODEL_FAMILIES[model],
                        "model_hash": cell.get("model_hash"),
                        "canonical_tokenizer_payload_hash": cell.get(
                            "canonical_tokenizer_payload_hash"
                        ),
                        "semantic_group_identity": cell.get("semantic_group_identity"),
                        "semantic_family": cell.get("semantic_family"),
                        "split": cell.get("split"),
                        "score_identity": cell.get("score_identity"),
                        "sequence_identity": cell.get("sequence_identity"),
                        "nuisance_transform_identity": transform,
                        "nuisance_transform_hash": sha256_json(
                            [cell.get("semantic_group_identity"), transform]
                        ),
                        "presentation_position": position,
                        "candidate_slot": slot,
                        "candidate_id": candidate_ids[slot],
                        "candidate_sequence_sha256": hashes.get(f"slot_{slot}"),
                        "full_sequence_sha256": full_sequence_hash,
                        "prompt_token_ids": prompt_ids,
                        "candidate_token_ids": candidate_ids_for_slot,
                        "worker_payload": {
                            "prompt_token_ids": prompt_ids,
                            "candidate_token_ids": candidate_ids_for_slot,
                        },
                    }
                )
    identities = [str(row["cell_identity"]) for row in work]
    if len(identities) != len(set(identities)):
        raise SemanticScoringError("duplicate_work_identity")
    return work


def validate_worker_payload(payload: Mapping[str, Any]) -> list[str]:
    """Reject all fields except the two frozen token-ID arrays."""

    allowed = {"prompt_token_ids", "candidate_token_ids"}
    errors = [f"forbidden_worker_key:{key}" for key in payload if key not in allowed]
    serialized = canonical_json(payload).lower()
    if any(word in serialized for word in ("semantic", "judgment", "label", "commitment")):
        errors.append("forbidden_worker_text")
    for key in allowed:
        values = payload.get(key)
        if not isinstance(values, list) or not values or not all(type(value) is int for value in values):
            errors.append(f"invalid_worker_tokens:{key}")
    return errors


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash one score row while excluding its self-referential digest."""

    unsigned = dict(row)
    unsigned["row_hash"] = ""
    return sha256_json(unsigned)


def score_response_row(item: Mapping[str, Any], response: Mapping[str, Any]) -> JsonDict:
    """Validate one raw response and attach non-label experiment metadata."""

    prompt = response.get("prompt_token_ids")
    candidate = response.get("candidate_token_ids")
    if prompt != item.get("prompt_token_ids") or candidate != item.get("candidate_token_ids"):
        raise SemanticScoringError("token_identity_drift")
    token_logprobs = response.get("token_logprobs")
    if not isinstance(token_logprobs, list) or len(token_logprobs) != len(candidate or []):
        raise SemanticScoringError("token_alignment")
    values = [float(value) for value in token_logprobs]
    if not all(math.isfinite(value) for value in values):
        raise SemanticScoringError("token_logprob_nonfinite")
    if response.get("forced_sequence") is not True:
        raise SemanticScoringError("forced_sequence_receipt_missing")
    if response.get("generated_token_count") != 0:
        raise SemanticScoringError("worker_generated_tokens")
    latency = float(response.get("latency_s", -1.0))
    if not math.isfinite(latency) or latency < 0:
        raise SemanticScoringError("latency_invalid")
    raw_sum = float(sum(values))
    row: JsonDict = {
        "schema": SCHEMA + ".score_row",
        "cell_identity": item.get("cell_identity"),
        "model_hf_id": item.get("model_hf_id"),
        "model_family": item.get("model_family"),
        "model_hash": item.get("model_hash"),
        "canonical_tokenizer_payload_hash": item.get(
            "canonical_tokenizer_payload_hash"
        ),
        "semantic_group_identity": item.get("semantic_group_identity"),
        "semantic_family": item.get("semantic_family"),
        "split": item.get("split"),
        "score_identity": item.get("score_identity"),
        "sequence_identity": item.get("sequence_identity"),
        "nuisance_transform_identity": item.get("nuisance_transform_identity"),
        "nuisance_transform_hash": item.get("nuisance_transform_hash"),
        "presentation_position": item.get("presentation_position"),
        "candidate_slot": item.get("candidate_slot"),
        "candidate_id": item.get("candidate_id"),
        "candidate_sequence_sha256": item.get("candidate_sequence_sha256"),
        "full_sequence_sha256": item.get("full_sequence_sha256"),
        "prompt_token_ids": list(prompt or []),
        "candidate_token_ids": list(candidate or []),
        "token_logprobs": [round(value, ROUND_DIGITS) for value in values],
        "finite": True,
        "raw_logprob_sum": round(raw_sum, ROUND_DIGITS),
        "per_token_logprob": round(raw_sum / len(values), ROUND_DIGITS),
        "latency_s": round(latency, 6),
        "forced_sequence": True,
        "generated_token_count": 0,
        "row_hash": "",
    }
    row["row_hash"] = row_hash(row)
    return row


def _failure_hash(row: Mapping[str, Any]) -> str:
    unsigned = dict(row)
    unsigned["failure_hash"] = ""
    return sha256_json(unsigned)


def make_failure_row(item: Mapping[str, Any], *, reason: str, detail: str) -> JsonDict:
    """Preserve one failed cell without a synthetic score."""

    status = "timeout" if "timeout" in reason.lower() else "failed"
    row: JsonDict = {
        "schema": SCHEMA + ".failed_cell",
        "cell_identity": item.get("cell_identity"),
        "model_hf_id": item.get("model_hf_id"),
        "semantic_group_identity": item.get("semantic_group_identity"),
        "split": item.get("split"),
        "sequence_identity": item.get("sequence_identity"),
        "nuisance_transform_identity": item.get("nuisance_transform_identity"),
        "presentation_position": item.get("presentation_position"),
        "candidate_slot": item.get("candidate_slot"),
        "status": status,
        "reason": str(reason),
        "detail": str(detail),
        "imputed": False,
        "failure_hash": "",
    }
    row["failure_hash"] = _failure_hash(row)
    return row


def _checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    unsigned = dict(checkpoint)
    unsigned["checkpoint_hash"] = ""
    return sha256_json(unsigned)


def build_checkpoint(
    *,
    input_checksum: str,
    expected_identities: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    failed_cells: Sequence[Mapping[str, Any]],
    completed_models: Sequence[str],
    process_receipts: Sequence[Mapping[str, Any]] = (),
    accelerator_samples: Sequence[Mapping[str, Any]] = (),
    lease_receipts: Sequence[Mapping[str, Any]] = (),
    teardown_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one atomic restart record after a bounded group batch."""

    expected = [str(value) for value in expected_identities]
    score_rows = [deepcopy(dict(row)) for row in rows]
    failures = [deepcopy(dict(row)) for row in failed_cells]
    terminal = {str(row.get("cell_identity")) for row in [*score_rows, *failures]}
    checkpoint: JsonDict = {
        "schema": SCHEMA + ".checkpoint",
        "input_checksum": str(input_checksum),
        "expected_identities": expected,
        "expected_cell_count": len(expected),
        "rows": score_rows,
        "failed_cells": failures,
        "completed_models": list(completed_models),
        "process_receipts": [deepcopy(dict(row)) for row in process_receipts],
        "accelerator_samples": [deepcopy(dict(row)) for row in accelerator_samples],
        "lease_receipts": [deepcopy(dict(row)) for row in lease_receipts],
        "teardown_receipts": [deepcopy(dict(row)) for row in teardown_receipts],
        "terminal_cell_count": len(terminal),
        "missing_cell_count": max(0, len(expected) - len(terminal)),
        "complete": terminal == set(expected),
        "checkpoint_hash": "",
    }
    checkpoint["checkpoint_hash"] = _checkpoint_hash(checkpoint)
    return checkpoint


def load_checkpoint(path: str | Path, *, input_checksum: str) -> JsonDict:
    """Verify input, manifest, score, and failure hashes before resume."""

    target = Path(path)
    if not target.is_file():
        return {}
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise SemanticScoringError("checkpoint_object_required")
    checkpoint = dict(value)
    if checkpoint.get("input_checksum") != input_checksum:
        raise SemanticScoringError("checkpoint_input_hash_drift")
    if checkpoint.get("checkpoint_hash") != _checkpoint_hash(checkpoint):
        raise SemanticScoringError("checkpoint_hash_invalid")
    rows = [dict(row) for row in checkpoint.get("rows") or []]
    failures = [dict(row) for row in checkpoint.get("failed_cells") or []]
    if any(row.get("row_hash") != row_hash(row) for row in rows):
        raise SemanticScoringError("checkpoint_row_hash_invalid")
    if any(row.get("failure_hash") != _failure_hash(row) for row in failures):
        raise SemanticScoringError("checkpoint_failure_hash_invalid")
    identities = [str(row.get("cell_identity")) for row in [*rows, *failures]]
    if len(identities) != len(set(identities)):
        raise SemanticScoringError("checkpoint_duplicate_cell_identity")
    if not set(identities).issubset(set(checkpoint.get("expected_identities") or [])):
        raise SemanticScoringError("checkpoint_unexpected_cell_identity")
    return checkpoint


def pending_work_items(
    work_items: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any]
) -> list[JsonDict]:
    """Return identities with neither a verified score nor explicit failure."""

    terminal = {
        str(row.get("cell_identity"))
        for row in [
            *(checkpoint.get("rows") or []),
            *(checkpoint.get("failed_cells") or []),
        ]
        if isinstance(row, Mapping)
    }
    return [dict(item) for item in work_items if str(item.get("cell_identity")) not in terminal]


def lease_revalidation_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Name each lease-loss condition before a new scoring batch."""

    errors: list[str] = []
    if receipt.get("owner_verified") is not True:
        errors.append("lease_owner_lost")
    if receipt.get("expired") is True:
        errors.append("lease_expired")
    if receipt.get("phase") != "inferencing":
        errors.append("lease_phase")
    if receipt.get("released") is True:
        errors.append("lease_released")
    return errors


def teardown_receipt_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Name unsafe or incomplete cleanup facts."""

    errors: list[str] = []
    action = receipt.get("action")
    if receipt.get("ownership_verified") is not True and action not in {
        "not_started",
        "already_exited",
    }:
        errors.append("ownership")
    if receipt.get("process_exit_confirmed") is not True:
        errors.append("process_exit")
    if receipt.get("port_release_confirmed") is not True:
        errors.append("port_release")
    if receipt.get("unrelated_process_kill_count_delta") != 0:
        errors.append("unrelated_process_signal")
    return errors


def build_score_sidecars(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build separate calibration and held raw-row manifests without reduction."""

    output: JsonDict = {}
    for split in ("calibration", "held"):
        selected = [deepcopy(dict(row)) for row in rows if row.get("split") == split]
        output[split] = {
            "schema": SCHEMA + f".{split}_raw_token_sidecar",
            "split": split,
            "row_count": len(selected),
            "rows": selected,
            "sha256": sha256_json(selected),
            "effect_reduced": False,
        }
    return output


def completion_score(
    *,
    expected_identities: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    failed_cells: Sequence[Mapping[str, Any]],
    checkpoint_complete: bool,
    teardown_receipts: Sequence[Mapping[str, Any]],
) -> int:
    """Return one only for complete cells, checkpoint, and clean teardown."""

    expected = {str(value) for value in expected_identities}
    terminal_rows = [*rows, *failed_cells]
    terminal = [str(row.get("cell_identity")) for row in terminal_rows]
    identities_complete = len(terminal) == len(set(terminal)) and set(terminal) == expected
    rows_valid = all(row.get("row_hash") == row_hash(row) for row in rows)
    failures_valid = all(
        row.get("failure_hash") == _failure_hash(row) and row.get("imputed") is False
        for row in failed_cells
    )
    teardown_clean = len(teardown_receipts) == len(MODEL_SPECS) and all(
        not teardown_receipt_errors(receipt) for receipt in teardown_receipts
    )
    return int(
        identities_complete
        and rows_valid
        and failures_valid
        and checkpoint_complete
        and teardown_clean
    )


def _base_artifact(*, run_date: str, duration_s: float) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": 6868,
        "run_date": str(run_date),
        "status": "blocked",
        "spec_refs": ["REQ-INFERENCE-6868", "SCENARIO-INFERENCE-6868-*"],
        "source_artifact_hashes": {
            "exp6867": {
                "path": str(EXP6867_RELATIVE_PATH),
                "sha256": EXPECTED_EXP6867_SHA256,
            }
        },
        "preconditions_checked": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "model_specs": list(MODEL_SPECS),
        "models_used": [],
        "missing_model_manifest": list(MODEL_SPECS),
        "model_artifact_hashes": {},
        "tokenizer_receipts": [],
        "process_receipts": [],
        "accelerator_samples": [],
        "lease_receipts": [],
        "rows": [],
        "raw_token_sidecars": {},
        "calibration_score_manifest": {},
        "sealed_held_score_manifest": {},
        "failed_cell_manifest": [],
        "checkpoint_manifest": {},
        "teardown_receipts": [],
        "generated_answer_count": 0,
        "held_label_access_count": 0,
        "scientific_effect_claimed": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "semantic_contrast_stream_v2_complete_score": 0,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
        "expected_cell_count": 0,
        "scored_cell_count": 0,
        "explicit_failure_cell_count": 0,
        "method_limits": {
            "raw_scores_only": True,
            "held_effect_reduced": False,
            "external_text_scorer_trained": False,
            "candidate_answer_generated": False,
            "semantic_effect_claimed": False,
        },
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": {},
    }


def _attach_field_principles(artifact: JsonDict) -> JsonDict:
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, "This field preserves terminal raw-score evidence.")
        for key in artifact
    }
    return artifact


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    excluded = {"field_principles", "duration_s", "reproducibility_checksum"}
    return sha256_json({key: value for key, value in artifact.items() if key not in excluded})


def build_blocked_artifact(
    *, run_date: str, duration_s: float, preconditions: Mapping[str, Any]
) -> JsonDict:
    """Build complete blocked evidence for any failed gate."""

    artifact = _base_artifact(run_date=run_date, duration_s=duration_s)
    artifact["preconditions_checked"] = deepcopy(dict(preconditions))
    artifact["gate_check_summary"] = deepcopy(dict(preconditions))
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return _attach_field_principles(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return each schema, generation, label, or completion error."""

    errors = [
        f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if artifact.get("generated_answer_count") != 0:
        errors.append("generated_answers_forbidden")
    if artifact.get("held_label_access_count") != 0:
        errors.append("held_label_access_forbidden")
    if artifact.get("scientific_effect_claimed") is not False:
        errors.append("scientific_effect_claim_forbidden")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_complete_prefixed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    rows_value = artifact.get("rows")
    rows = rows_value if isinstance(rows_value, list) else []
    if any(not isinstance(row, Mapping) or row.get("row_hash") != row_hash(row) for row in rows):
        errors.append("raw_row_hash_invalid")
    if artifact.get("semantic_contrast_stream_v2_complete_score") == 1 and not rows:
        errors.append("complete_score_without_complete_rows")
    return errors


def _resolved_models(preregistration: Mapping[str, Any]) -> list[JsonDict]:  # pragma: no cover
    artifacts = dict(preregistration.get("model_artifact_hashes") or {})
    tokenizers = {
        str(row.get("hf_id")): dict(row)
        for row in preregistration.get("tokenizer_receipts") or []
        if isinstance(row, Mapping)
    }
    rows: list[JsonDict] = []
    for model in MODEL_SPECS:
        artifact = dict(artifacts.get(model) or {})
        path = Path(str(artifact.get("path") or ""))
        observed_hash = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "hf_id": model,
                "family": MODEL_FAMILIES[model],
                "model_path": str(path),
                "model_sha256": observed_hash,
                "model_size_bytes": path.stat().st_size if path.is_file() else None,
                "canonical_tokenizer_payload_hash": tokenizers.get(model, {}).get(
                    "canonical_tokenizer_payload_sha256"
                ),
                "quantization": artifact.get("quantization"),
                "snapshot_identity": artifact.get("snapshot_identity"),
            }
        )
    return rows


def _probe_lease(gpu: Mapping[str, Any]) -> JsonDict:  # pragma: no cover - live host.
    lease = None
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=LEASE_RUNTIME_DIR,
            task_id="exp6868-preflight",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model="exp6868-preflight-no-model",
            vram_before_mb=int(gpu.get("free_vram_mb", 0)),
            ttl_s=30.0,
        )
        owner = lease.owner_receipt()
        lease.transition("terminal_blocked")
        release = lease.release()
        return {"ok": release.get("released") is True, "owner": owner, "release": release}
    except Exception as exc:
        if lease is not None:
            lease.close()
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def collect_live_preconditions(root: Path) -> JsonDict:  # pragma: no cover - live host.
    """Call the canonical cache resolver and collect every fail-closed gate."""

    prereg_path = root / EXP6867_RELATIVE_PATH
    preregistration = read_json(prereg_path)
    preregistration_hash = sha256_file(prereg_path)
    pair = cached_sota_pair() or []
    cached_ids = [str(row.get("hf_id")) for row in pair]
    models = _resolved_models(preregistration)
    ports = _choose_free_ports(len(MODEL_SPECS))
    ports_free = len(ports) == len(MODEL_SPECS) and all(port_is_free(port) for port in ports)
    cuda = _cuda_token_scoring()
    inventory = _gpu_inventory()
    largest_model_mb = max(
        int(row.get("model_size_bytes") or 0) // (1024 * 1024) for row in models
    )
    required_free_mb = largest_model_mb + 768
    eligible = [
        row for row in inventory if int(row.get("free_vram_mb", 0)) >= required_free_mb
    ]
    gpu = max(eligible, key=lambda row: int(row.get("free_vram_mb", 0))) if eligible else {}
    lease_probe = _probe_lease(gpu) if gpu else {"ok": False, "detail": "no_eligible_gpu"}
    disk = shutil.disk_usage(root)
    preconditions = evaluate_preconditions(
        preregistration=preregistration,
        preregistration_sha256=preregistration_hash,
        expected_preregistration_sha256=EXPECTED_EXP6867_SHA256,
        resolved_models=models,
        cached_sota_pair_ids=cached_ids,
        cuda_token_scoring=cuda.get("ok") is True,
        sufficient_disk=disk.free >= DISK_FLOOR_BYTES,
        free_task_ports=ports_free,
        bounded_lease=lease_probe.get("ok") is True,
    )
    preconditions.update(
        {
            "cached_sota_pair_called": True,
            "cached_sota_pair_hf_ids": cached_ids,
            "cuda_token_scoring": cuda,
            "disk_free_bytes": disk.free,
            "disk_floor_bytes": DISK_FLOOR_BYTES,
            "task_ports": ports,
            "eligible_gpu": gpu,
            "required_free_vram_mb": required_free_mb,
            "lease_probe": lease_probe,
            "unrelated_process_observations": _compute_apps(),
        }
    )
    for model in models:
        model["gpu"] = gpu.get("index")
        model["gpu_uuid"] = gpu.get("gpu_uuid")
        model["visible_devices"] = gpu.get("visible_devices", gpu.get("index"))
    return {
        "preregistration": preregistration,
        "preregistration_hash": preregistration_hash,
        "models": models,
        "ports": ports,
        "gpu": gpu,
        "preconditions": preconditions,
        "accelerator_samples": [
            {**row, "phase": "preflight", "observed_only": True} for row in inventory
        ],
    }


class _TokenScoringEngine:  # pragma: no cover - live llama.cpp CUDA path.
    """Evaluate exact frozen token sequences without decoding or sampling."""

    def __init__(self, model_path: str) -> None:
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=CONTEXT_LENGTH,
            n_batch=512,
            n_ubatch=128,
            seed=RANDOM_SEED,
            logits_all=True,
            verbose=True,
        )
        self.logprob_pool = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1))

    @staticmethod
    def _logprob(logits: Any, token_id: int) -> float:
        import numpy as np

        values = np.asarray(logits, dtype=np.float64)
        maximum = float(values.max())
        return float(values[int(token_id)] - maximum - np.log(np.exp(values - maximum).sum()))

    def score(self, prompt: Sequence[int], candidate: Sequence[int]) -> JsonDict:
        started = time.monotonic()
        prompt_ids = [int(value) for value in prompt]
        candidate_ids = [int(value) for value in candidate]
        tokens = prompt_ids + candidate_ids
        if not prompt_ids or not candidate_ids:
            raise SemanticScoringError("worker_tokens_missing")
        if len(tokens) > CONTEXT_LENGTH:
            raise SemanticScoringError(f"worker_context_overflow:{len(tokens)}")
        self.llm.reset()
        self.llm.eval(tokens)
        scores = self.llm.scores
        score_rows = (
            scores[index - 1]
            for index in range(len(prompt_ids), len(prompt_ids) + len(candidate_ids))
        )
        logprobs = list(self.logprob_pool.map(self._logprob, score_rows, candidate_ids))
        return {
            "prompt_token_ids": prompt_ids,
            "candidate_token_ids": candidate_ids,
            "token_logprobs": logprobs,
            "latency_s": time.monotonic() - started,
            "forced_sequence": True,
            "generated_token_count": 0,
        }


def _run_worker(model_path: str, port: int) -> int:  # pragma: no cover - subprocess.
    engine = _TokenScoringEngine(model_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            del fmt, args

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def do_POST(self) -> None:
            if self.path != "/score":
                self.send_response(404)
                self.end_headers()
                return
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            errors = validate_worker_payload(payload)
            if errors:
                body = json.dumps({"errors": errors}).encode("utf-8")
                self.send_response(400)
            else:
                body = json.dumps(
                    engine.score(payload["prompt_token_ids"], payload["candidate_token_ids"]),
                    allow_nan=False,
                ).encode("utf-8")
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", int(port)), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        gc.collect()
    return 0


class _LiveScorer:  # pragma: no cover - live process.
    """Own one token-only worker and expose bounded requests."""

    def __init__(
        self,
        *,
        model: Mapping[str, Any],
        port: int,
        gpu: Mapping[str, Any],
        runtime_dir: Path,
    ) -> None:
        self.model = dict(model)
        command = [
            sys.executable,
            "-m",
            "carnot.experiment_6868_three_family_semantic_scoring_stream_v2",
            "--score-worker",
            "--model-path",
            str(model["model_path"]),
            "--port",
            str(port),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        family = str(model["family"])
        self.process = OwnedLlamaCppProcess(
            command=command,
            port=port,
            env=env,
            log_path=runtime_dir / f"{family}.log",
            state_path=runtime_dir / f"{family}.owner.json",
        )

    def start(self) -> JsonDict:
        receipt = self.process.launch()
        health = self.process.wait_for_health(HEALTH_TIMEOUT_S)
        receipt.update(
            {
                "health": health,
                "hf_id": self.model["hf_id"],
                "model_hash": self.model["model_sha256"],
                "tokenizer_hash": self.model["canonical_tokenizer_payload_hash"],
                "gpu_uuid": self.model["gpu_uuid"],
                "visible_devices": str(self.model["visible_devices"]),
                "offload": {"n_gpu_layers": -1, "cuda_required": True},
                "context_length": CONTEXT_LENGTH,
            }
        )
        if health.get("ok") is not True:
            raise SemanticScoringError(f"worker_health_timeout:{health.get('reason')}")
        return receipt

    def score(self, payload: Mapping[str, Any]) -> JsonDict:
        errors = validate_worker_payload(payload)
        if errors:
            raise SemanticScoringError(f"worker_payload_rejected:{','.join(errors)}")
        return self.process.post_json("/score", payload, REQUEST_TIMEOUT_S)

    def close(self) -> JsonDict:
        receipt = self.process.cleanup()
        popen = self.process.process
        if popen is not None and receipt.get("process_reaped") is not True:
            try:
                popen.wait(timeout=30.0)
            except Exception as exc:
                receipt["bounded_reap_error"] = f"{type(exc).__name__}: {exc}"
            receipt["process_reaped"] = popen.poll() is not None
            receipt["process_exit_confirmed"] = receipt["process_reaped"]
            receipt["port_release_confirmed"] = port_is_free(self.process.port)
            receipt["leak_free"] = bool(
                receipt["process_reaped"] and receipt["port_release_confirmed"]
            )
        return receipt


def _finish_lease(
    lease: Any, *, clean: bool, teardown: Mapping[str, Any], after: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - live lease state.
    phase = str(lease.document.get("phase"))
    if phase in {"resident", "inferencing"}:
        lease.transition("unloading")
        phase = "unloading"
    if phase == "unloading":
        lease.transition(
            "validating",
            vram_mb=int(after.get("owned_vram_mb", 0)),
            exit_code=0 if clean else 1,
            unload_observed=teardown.get("leak_free") is True,
        )
        phase = "validating"
    if phase in {"preflight", "admitted", "loading", "validating"}:
        lease.transition("terminal_complete" if clean and phase == "validating" else "terminal_blocked")
    return lease.release()


def _group_batches(items: Sequence[Mapping[str, Any]]) -> list[list[JsonDict]]:
    groups: list[list[JsonDict]] = []
    current_keys: list[str] = []
    current: list[JsonDict] = []
    for item in items:
        key = str(item.get("semantic_group_identity"))
        if key not in current_keys and len(current_keys) >= GROUP_BATCH_SIZE:
            groups.append(current)
            current_keys = []
            current = []
        if key not in current_keys:
            current_keys.append(key)
        current.append(dict(item))
    if current:
        groups.append(current)
    return groups


def _run_live_model_phase(
    *,
    model: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    port: int,
    gpu: Mapping[str, Any],
    runtime_dir: Path,
    checkpoint_callback: Any,
) -> JsonDict:  # pragma: no cover - live CUDA path.
    before = _gpu_snapshot(gpu, phase="before")
    lease = lease_api.GpuLease.acquire(
        runtime_dir=LEASE_RUNTIME_DIR,
        task_id=f"exp6868-{model['family']}",
        device_uuid=str(gpu["gpu_uuid"]),
        expected_model=str(model["hf_id"]),
        vram_before_mb=int(before.get("free_vram_mb", 0)),
        ttl_s=LEASE_TTL_S,
    )
    owner = lease.owner_receipt()
    scorer = _LiveScorer(model=model, port=port, gpu=gpu, runtime_dir=runtime_dir)
    rows: list[JsonDict] = []
    failures: list[JsonDict] = []
    process_receipt: JsonDict = {}
    teardown: JsonDict = {}
    resident: JsonDict = {}
    phase_error: Exception | None = None
    try:
        lease.transition("admitted")
        lease.transition("loading")
        process_receipt = scorer.start()
        resident = _gpu_snapshot(gpu, phase="resident", owned_pid=int(process_receipt["pid"]))
        if resident.get("owned_cuda_residency") is not True:
            raise SemanticScoringError("owned_cuda_residency_missing")
        process_receipt["vram_samples"] = [before, resident]
        lease.transition("resident", vram_mb=int(resident.get("owned_vram_mb", 0)))
        lease.transition("inferencing")
        for batch_index, batch in enumerate(_group_batches(items)):
            heartbeat = dict(lease.heartbeat())
            heartbeat.setdefault("phase", lease.document.get("phase"))
            heartbeat.setdefault("expired", False)
            heartbeat.setdefault("released", False)
            lease_errors = lease_revalidation_errors(heartbeat)
            if lease_errors:
                for item in [candidate for later in _group_batches(items)[batch_index:] for candidate in later]:
                    failures.append(
                        make_failure_row(
                            item,
                            reason="lease_loss",
                            detail=",".join(lease_errors),
                        )
                    )
                checkpoint_callback(rows, failures)
                break
            for item in batch:
                try:
                    response = scorer.score(dict(item["worker_payload"]))
                    rows.append(score_response_row(item, response))
                except (TimeoutError, error.URLError) as exc:
                    failures.append(
                        make_failure_row(
                            item,
                            reason="timeout",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                    )
                except Exception as exc:
                    failures.append(
                        make_failure_row(
                            item,
                            reason="score_failure",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                    )
            checkpoint_callback(rows, failures)
    except Exception as exc:
        phase_error = exc
        terminal_ids = {str(row["cell_identity"]) for row in [*rows, *failures]}
        for item in items:
            if str(item["cell_identity"]) not in terminal_ids:
                failures.append(
                    make_failure_row(
                        item,
                        reason="model_phase_failure",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
        checkpoint_callback(rows, failures)
    finally:
        teardown = scorer.close()
        after = _gpu_snapshot(
            gpu, phase="after", owned_pid=int(process_receipt.get("pid", 0) or 0)
        )
        if process_receipt:
            process_receipt.setdefault("vram_samples", [before, resident])
            process_receipt["vram_samples"].append(after)
        clean = phase_error is None and not teardown_receipt_errors(teardown)
        try:
            release = _finish_lease(lease, clean=clean, teardown=teardown, after=after)
            lease_error = None
        except Exception as exc:
            lease.close()
            release = {}
            lease_error = f"{type(exc).__name__}: {exc}"
    return {
        "rows": rows,
        "failed_cells": failures,
        "process_receipt": process_receipt,
        "accelerator_samples": [before, resident, after],
        "lease_receipt": {
            "hf_id": model["hf_id"],
            "owner": owner,
            "phase_history": deepcopy(lease.document.get("phase_history", [])),
            "release": release,
            "lease_error": lease_error,
            "lease_valid": release.get("phase") == "terminal_complete",
        },
        "teardown_receipt": {"hf_id": model["hf_id"], **teardown},
    }


def _checkpoint_input_checksum(
    preregistration_hash: str,
    models: Sequence[Mapping[str, Any]],
    work_items: Sequence[Mapping[str, Any]],
) -> str:
    return sha256_json(
        {
            "preregistration_hash": preregistration_hash,
            "model_bindings": _observed_model_bindings(models),
            "work_identities": [row["cell_identity"] for row in work_items],
            "random_seed": RANDOM_SEED,
            "group_batch_size": GROUP_BATCH_SIZE,
        }
    )


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live files.
    paths = {
        "exp6867": EXP6867_RELATIVE_PATH,
        "module": MODULE_RELATIVE_PATH,
        "wrapper": WRAPPER_RELATIVE_PATH,
        "test": TEST_RELATIVE_PATH,
        "spec": SPEC_RELATIVE_PATH,
        "process_module": PROCESS_MODULE_RELATIVE_PATH,
    }
    return {
        key: {"path": str(path), "sha256": sha256_file(root / path)}
        for key, path in paths.items()
    }


def _recover_checkpoint_lifecycle(
    checkpoint: JsonDict, lease_probe: Mapping[str, Any]
) -> None:  # pragma: no cover - live crash recovery.
    """Correct stale exit races only after fresh PID, port, and lease checks."""

    processes = {
        str(row.get("hf_id")): dict(row)
        for row in checkpoint.get("process_receipts") or []
        if isinstance(row, Mapping)
    }
    recovered_teardowns: list[JsonDict] = []
    for raw in checkpoint.get("teardown_receipts") or []:
        receipt = dict(raw)
        process = processes.get(str(receipt.get("hf_id")), {})
        pid = int(process.get("pid", -1))
        port = int(process.get("port", -1))
        current = read_process_identity(pid)
        released = port_is_free(port)
        if (
            teardown_receipt_errors(receipt)
            and current.get("exists") is not True
            and released
            and receipt.get("ownership_verified") is True
            and receipt.get("unrelated_process_kill_count_delta") == 0
        ):
            receipt = {
                **receipt,
                "action": "recovered_already_exited",
                "original_cleanup_receipt": dict(raw),
                "recovery_observation": {
                    "pid": pid,
                    "process_exists": False,
                    "port": port,
                    "port_release_confirmed": True,
                },
                "process_exit_confirmed": True,
                "process_reaped": True,
                "port_release_confirmed": True,
                "leak_free": True,
            }
        recovered_teardowns.append(receipt)
    checkpoint["teardown_receipts"] = recovered_teardowns

    recovered_leases: list[JsonDict] = []
    for raw in checkpoint.get("lease_receipts") or []:
        receipt = dict(raw)
        owner = dict(receipt.get("owner") or {})
        owner_pid = int(owner.get("pid", -1))
        if (
            receipt.get("lease_valid") is not True
            and read_process_identity(owner_pid).get("exists") is not True
            and lease_probe.get("ok") is True
        ):
            receipt = {
                **receipt,
                "original_lease_receipt": dict(raw),
                "recovery_release_receipt": deepcopy(dict(lease_probe)),
                "lease_error": None,
                "lease_valid": True,
                "recovered_after_owner_exit": True,
            }
        recovered_leases.append(receipt)
    checkpoint["lease_receipts"] = recovered_leases


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    checkpoint_path: Path | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:  # pragma: no cover - live orchestration.
    """Run all eligible frozen cells or publish complete blocked evidence."""

    started = time.monotonic()
    result = result_path or root / RESULT_RELATIVE_PATH
    checkpoint_target = checkpoint_path or root / CHECKPOINT_RELATIVE_PATH
    live = collect_live_preconditions(root)
    preconditions = dict(live["preconditions"])
    if preconditions.get("passed") is not True:
        artifact = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.monotonic() - started,
            preconditions=preconditions,
        )
        artifact["accelerator_samples"] = list(live.get("accelerator_samples") or [])
        artifact["model_artifact_hashes"] = {
            row["hf_id"]: {
                "path": row["model_path"],
                "sha256": row["model_sha256"],
                "size_bytes": row["model_size_bytes"],
            }
            for row in live["models"]
        }
        artifact["tokenizer_receipts"] = [
            {
                "hf_id": row["hf_id"],
                "canonical_tokenizer_payload_sha256": row[
                    "canonical_tokenizer_payload_hash"
                ],
            }
            for row in live["models"]
        ]
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
        _attach_field_principles(artifact)
        write_json_atomic(result, artifact)
        return artifact

    preregistration = dict(live["preregistration"])
    work_items = expected_work_items(preregistration.get("rows") or [])
    verified_models = {str(row["hf_id"]): row for row in live["models"]}
    for item in work_items:
        verified = verified_models[str(item["model_hf_id"])]
        item["model_hash"] = verified["model_sha256"]
        item["canonical_tokenizer_payload_hash"] = verified[
            "canonical_tokenizer_payload_hash"
        ]
    expected_ids = [str(item["cell_identity"]) for item in work_items]
    input_checksum = _checkpoint_input_checksum(
        str(live["preregistration_hash"]), live["models"], work_items
    )
    checkpoint = load_checkpoint(checkpoint_target, input_checksum=input_checksum)
    if checkpoint:
        _recover_checkpoint_lifecycle(checkpoint, preconditions.get("lease_probe") or {})
    rows = [dict(row) for row in checkpoint.get("rows") or []]
    failures = [dict(row) for row in checkpoint.get("failed_cells") or []]
    completed_models = [str(value) for value in checkpoint.get("completed_models") or []]
    process_receipts = [dict(row) for row in checkpoint.get("process_receipts") or []]
    accelerator_samples = [
        *list(live.get("accelerator_samples") or []),
        *[dict(row) for row in checkpoint.get("accelerator_samples") or []],
    ]
    lease_receipts = [dict(row) for row in checkpoint.get("lease_receipts") or []]
    teardown_receipts = [dict(row) for row in checkpoint.get("teardown_receipts") or []]

    runtime_dir = Path(tempfile.mkdtemp(prefix="carnot-exp6868-"))

    def save_checkpoint() -> JsonDict:
        manifest = build_checkpoint(
            input_checksum=input_checksum,
            expected_identities=expected_ids,
            rows=rows,
            failed_cells=failures,
            completed_models=completed_models,
            process_receipts=process_receipts,
            accelerator_samples=accelerator_samples,
            lease_receipts=lease_receipts,
            teardown_receipts=teardown_receipts,
        )
        write_json_atomic(checkpoint_target, manifest)
        return manifest

    for model_index, model in enumerate(live["models"]):
        pending = [
            item
            for item in pending_work_items(work_items, save_checkpoint())
            if item["model_hf_id"] == model["hf_id"]
        ]
        if not pending:
            if model["hf_id"] not in completed_models:
                completed_models.append(model["hf_id"])
            save_checkpoint()
            continue

        base_row_count = len(rows)
        base_failure_count = len(failures)

        def batch_checkpoint(new_rows: Sequence[Mapping[str, Any]], new_failures: Sequence[Mapping[str, Any]]) -> None:
            del rows[base_row_count:]
            rows.extend(deepcopy(dict(row)) for row in new_rows)
            del failures[base_failure_count:]
            failures.extend(deepcopy(dict(row)) for row in new_failures)
            save_checkpoint()

        try:
            phase = _run_live_model_phase(
                model=model,
                items=pending,
                port=int(live["ports"][model_index]),
                gpu=live["gpu"],
                runtime_dir=runtime_dir,
                checkpoint_callback=batch_checkpoint,
            )
        except Exception as exc:
            phase = {
                "rows": [],
                "failed_cells": [
                    make_failure_row(
                        item,
                        reason="model_phase_start_failure",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                    for item in pending
                ],
                "process_receipt": {},
                "accelerator_samples": [],
                "lease_receipt": {
                    "hf_id": model["hf_id"],
                    "lease_valid": False,
                    "lease_error": f"{type(exc).__name__}: {exc}",
                },
                "teardown_receipt": {
                    "hf_id": model["hf_id"],
                    "action": "not_started",
                    "ownership_verified": False,
                    "process_exit_confirmed": True,
                    "process_reaped": True,
                    "port_release_confirmed": port_is_free(
                        int(live["ports"][model_index])
                    ),
                    "unrelated_process_kill_count_delta": 0,
                },
            }
        batch_checkpoint(phase["rows"], phase["failed_cells"])
        if phase["process_receipt"]:
            process_receipts.append(dict(phase["process_receipt"]))
        accelerator_samples.extend(dict(row) for row in phase["accelerator_samples"])
        lease_receipts.append(dict(phase["lease_receipt"]))
        teardown_receipts.append(dict(phase["teardown_receipt"]))
        model_expected = {
            item["cell_identity"] for item in work_items if item["model_hf_id"] == model["hf_id"]
        }
        terminal = {row["cell_identity"] for row in [*rows, *failures]}
        if model_expected.issubset(terminal):
            completed_models.append(str(model["hf_id"]))
        save_checkpoint()

    final_checkpoint = save_checkpoint()
    sidecars = build_score_sidecars(rows)
    calibration_path = root / CALIBRATION_SIDECAR_RELATIVE_PATH
    held_path = root / HELD_SIDECAR_RELATIVE_PATH
    write_json_atomic(calibration_path, sidecars["calibration"])
    write_json_atomic(held_path, sidecars["held"])
    sidecar_manifest = {
        "calibration": {
            "path": str(CALIBRATION_SIDECAR_RELATIVE_PATH),
            "sha256": sha256_file(calibration_path),
            "payload_sha256": sidecars["calibration"]["sha256"],
            "row_count": sidecars["calibration"]["row_count"],
        },
        "held": {
            "path": str(HELD_SIDECAR_RELATIVE_PATH),
            "sha256": sha256_file(held_path),
            "payload_sha256": sidecars["held"]["sha256"],
            "row_count": sidecars["held"]["row_count"],
        },
    }
    cell_and_teardown_complete = completion_score(
        expected_identities=expected_ids,
        rows=rows,
        failed_cells=failures,
        checkpoint_complete=final_checkpoint.get("complete") is True,
        teardown_receipts=teardown_receipts,
    )
    leases_complete = len(lease_receipts) == len(MODEL_SPECS) and all(
        receipt.get("lease_valid") is True for receipt in lease_receipts
    )
    complete = int(cell_and_teardown_complete == 1 and leases_complete)
    missing_models = [model for model in MODEL_SPECS if model not in completed_models]
    if complete and not failures:
        verdict_class = "null"
        verdict = "complete_null_three_family_semantic_scoring_stream_v2_raw_scores_only"
    else:
        verdict_class = "partial"
        verdict = "complete_partial_three_family_semantic_scoring_stream_v2_raw_scores_only"
    gate_summary = _gate_summary(
        [
            _gate_check("all_required_cells_terminal", len(expected_ids), len(rows) + len(failures)),
            _gate_check("checkpoint_complete", True, final_checkpoint.get("complete") is True),
            _gate_check("clean_owned_teardown", True, complete == 1),
        ]
    )
    artifact = _base_artifact(run_date=run_date, duration_s=time.monotonic() - started)
    artifact.update(
        {
            "status": "complete" if complete else "partial",
            "source_artifact_hashes": _source_hashes(root),
            "preconditions_checked": preconditions,
            "models_used": list(dict.fromkeys(completed_models)),
            "missing_model_manifest": missing_models,
            "model_artifact_hashes": {
                row["hf_id"]: {
                    "path": row["model_path"],
                    "sha256": row["model_sha256"],
                    "size_bytes": row["model_size_bytes"],
                    "quantization": row["quantization"],
                    "snapshot_identity": row["snapshot_identity"],
                }
                for row in live["models"]
            },
            "tokenizer_receipts": [
                {
                    "hf_id": row["hf_id"],
                    "source": "Exp6867 canonical tokenizer binding plus exact GGUF hash",
                    "canonical_tokenizer_payload_sha256": row[
                        "canonical_tokenizer_payload_hash"
                    ],
                }
                for row in live["models"]
            ],
            "process_receipts": process_receipts,
            "accelerator_samples": accelerator_samples,
            "lease_receipts": lease_receipts,
            "rows": rows,
            "raw_token_sidecars": sidecar_manifest,
            "calibration_score_manifest": {
                **sidecar_manifest["calibration"],
                "effect_reduced": False,
            },
            "sealed_held_score_manifest": {
                **sidecar_manifest["held"],
                "labels_present": False,
                "effect_reduced": False,
            },
            "failed_cell_manifest": failures,
            "checkpoint_manifest": {
                key: value for key, value in final_checkpoint.items() if key not in {"rows", "failed_cells"}
            },
            "teardown_receipts": teardown_receipts,
            "semantic_contrast_stream_v2_complete_score": complete,
            "gate_check_summary": gate_summary,
            "verdict_class": verdict_class,
            "honest_verdict": verdict,
            "expected_cell_count": len(expected_ids),
            "scored_cell_count": len(rows),
            "explicit_failure_cell_count": len(failures),
        }
    )
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_field_principles(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise SemanticScoringError("artifact_invalid:" + ",".join(errors))
    write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI path.
    """Run the dated experiment or its private token-scoring worker."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--score-worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    if args.score_worker:
        if not args.model_path or args.port is None:
            parser.error("--score-worker requires --model-path and --port")
        return _run_worker(str(args.model_path), int(args.port))
    artifact = run(run_date=str(args.date))
    print(canonical_json({"honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
