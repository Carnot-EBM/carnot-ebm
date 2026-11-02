"""Measure a label-blind response surface for exact intervention blocks.

The scorer teacher-forces one fixed neutral response after each Exp7012 prompt.
It freezes all model rows before it opens the authority sidecar. This ordering
keeps labels and intervention metadata outside the measured response features.

Spec refs: REQ-ENERGY-7013 and SCENARIO-ENERGY-7013-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import time
from typing import Any
from urllib import error, request

import numpy as np

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6966_gguf_load_envelope_canary import (
    gpu_inventory,
    llama_cpp_probe,
    parse_offloaded_layers,
)
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess, port_is_free
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.task_runtime_receipts import write_json_atomic


JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_7013_three_family_intervention_surface"
SCHEMA = "carnot.experiment_7013.three_family_intervention_surface.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 7_013_202_609_05
INFERENCE_SUBSTRATE = "live_llm_inference"
PREFERRED_QUANT = "Q4_K_M"
EXPECTED_PAIR_COUNT = 48
EXPECTED_FAMILY_COUNT = 3
EXPECTED_PROMPT_COUNT = 192
EXPECTED_CELL_COUNT = EXPECTED_PROMPT_COUNT * EXPECTED_FAMILY_COUNT
N_CTX = 4096
N_BATCH = 1024
N_UBATCH = 256
VRAM_RESERVE_MB = 2048
LEASE_TTL_S = 10_800.0
HEALTH_TIMEOUT_S = 900.0
REQUEST_TIMEOUT_S = 300.0
MODEL_TIMEOUT_S = 10_800.0
LIVE_DURATION_FLOOR_S = 60.0
TIE_EPSILON = 1.0e-12
FIXED_RESPONSE_TEXT = "\nRESPONSE: The assessment is complete."
SCORING_PREAMBLE = "Use the exact bounded optimization mapping below."

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_7013_three_family_intervention_surface.json"
CHECKPOINT_ROOT = (
    REPO_ROOT / "results/checkpoints/experiment_7013_three_family_intervention_surface"
)
FREEZE_PATH = CHECKPOINT_ROOT / "response_freeze_manifest.json"
EXP7012_PATH = REPO_ROOT / "results/experiment_7012_exact_intervention_pair_fixture.json"
LEARNER_PROMPT_PATH = (
    REPO_ROOT / "results/raw/experiment_7012_exact_intervention_pair_fixture/learner_prompts.jsonl"
)
AUTHORITY_SIDECAR_PATH = (
    REPO_ROOT
    / "results/raw/experiment_7012_exact_intervention_pair_fixture/authority_sidecar.jsonl"
)
WRAPPER_PATH = (
    REPO_ROOT / "scripts/experiments/experiment_7013_three_family_intervention_surface.py"
)
TEST_PATH = REPO_ROOT / "tests/python/test_experiment_7013_three_family_intervention_surface.py"
SPEC_PATH = REPO_ROOT / "openspec/capabilities/energy-verification/spec.md"

EXPECTED_EXP7012_HASH = "sha256:21baf5d79e7f0b489098e586d4851bbbd34e881caefef77d38a91683d7f347b7"
EXPECTED_LEARNER_PROMPT_HASH = (
    "sha256:73cf63f3db3d906bfee5de4533a4cf179fb7af8b28226b4e0698092ef50c71a8"
)
EXPECTED_AUTHORITY_SIDECAR_HASH = (
    "sha256:2d714758c25f5f6c3bdc7fad239e272a9fbcf85fa484e2d41242bf2575305797"
)

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
CONDITION_ROLES = (
    "primary_clean",
    "primary_intervention",
    "isomorphic_clean",
    "isomorphic_intervention",
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "MODEL_SPECS",
    "model_rows",
    "model_file_hashes",
    "llama_binary_hash",
    "command_hash",
    "gpu_identity_rows",
    "gpu_lease_rows",
    "server_rows",
    "request_counter_rows",
    "completion_counter_rows",
    "teardown_rows",
    "source_artifact_hashes",
    "prompt_freeze_hash",
    "response_freeze_hash",
    "label_opened_at",
    "response_frozen_at",
    "rows",
    "per_pair_results",
    "condition_rows",
    "token_position_rows",
    "signed_response_rows",
    "family_effect_rows",
    "held_source_effect_rows",
    "tie_rows",
    "failed_cell_rows",
    "prohibited_feature_rows",
    "expected_family_count",
    "observed_family_count",
    "expected_pair_count",
    "observed_pair_count",
    "intervention_surface_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for every field makes the evidence contract reviewable.",
    "preconditions_checked": "Measured gates stop unsafe or fabricated model work.",
    "inference_substrate": "The substrate distinguishes live inference from a simulation.",
    "duration_s": "Wall time exposes skipped or implausibly short model work.",
    "MODEL_SPECS": "The exact three-family roster prevents silent model substitution.",
    "model_rows": "Per-family rows keep model identity and execution separate.",
    "model_file_hashes": "File hashes bind scores to exact local weights.",
    "llama_binary_hash": "The binary hash detects a changed scoring runtime.",
    "command_hash": "The command hash detects changed server launch arguments.",
    "gpu_identity_rows": "GPU UUID and process snapshots prove the execution device.",
    "gpu_lease_rows": "Owner-bound leases prevent unowned GPU work from counting.",
    "server_rows": "Health and PID receipts reject stale or foreign listeners.",
    "request_counter_rows": "Request counts expose missing scoring calls.",
    "completion_counter_rows": "Completion counts expose dropped or failed calls.",
    "teardown_rows": "Exit, close, lease, and port receipts prevent contamination.",
    "source_artifact_hashes": "Source hashes bind the run to the frozen fixture and code.",
    "prompt_freeze_hash": "The prompt hash detects any learner-input drift.",
    "response_freeze_hash": "One digest seals raw responses before labels open.",
    "label_opened_at": "The label time proves authority access followed the freeze.",
    "response_frozen_at": "The freeze time establishes the label-blind boundary.",
    "rows": "Raw label-free rows preserve every prompt-family measurement.",
    "per_pair_results": "Pair summaries keep the matched causal unit visible.",
    "condition_rows": "Late-joined roles prove each four-condition block is complete.",
    "token_position_rows": "Relative token rows make response alignment replayable.",
    "signed_response_rows": "Pair-oriented deltas measure the exact intervention response.",
    "family_effect_rows": "Separate family intervals prevent pooled reversals.",
    "held_source_effect_rows": "Held-source intervals expose source-specific direction.",
    "tie_rows": "Exact ties remain evidence instead of disappearing from summaries.",
    "failed_cell_rows": "Every failed cell remains in the planned denominator.",
    "prohibited_feature_rows": "Mutation probes prove denied metadata cannot enter features.",
    "expected_family_count": "The fixed target requires all three mandated families.",
    "observed_family_count": "The observed count exposes missing authentic family runs.",
    "expected_pair_count": "The fixed 48-pair target prevents silent scope reduction.",
    "observed_pair_count": "The observed count exposes missing late-joined blocks.",
    "intervention_surface_complete_score": "Completion measures evidence integrity, not direction.",
    "random_seed": "A fixed seed makes request order and intervals reproducible.",
    "reproducibility_checksum": "A content digest detects any later artifact drift.",
    "gate_check_summary": "Expected and observed values make failures actionable.",
    "verifier_is_oracle": "False states that model likelihood is not exact authority.",
    "verdict_class": "A closed class keeps completion and scientific direction distinct.",
    "honest_verdict": "A class-consistent prefix prevents an overstated result.",
}

PROHIBITED_RESPONSE_FIELDS = (
    "commitment",
    "rationale",
    "selfgrade",
    "sourcegroupid",
    "sourcepairid",
    "mutationkind",
    "split",
    "authoritywitness",
    "exactlabel",
    "expectedlabel",
)


class ScoringError(ValueError):
    """Raised when a fixed response cannot produce aligned finite scores."""


class FreezeError(ValueError):
    """Raised when raw response content changes after its declared freeze."""


class LabelAccessError(ValueError):
    """Raised when authority data would open before a valid response freeze."""


def canonical_json(value: Any) -> str:
    """Serialize JSON with stable ordering for all evidence hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash text with the repository's explicit digest prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository's explicit digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value in canonical form."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:  # pragma: no cover - live files include large GGUFs.
    """Hash a file in chunks so large model bytes need no single buffer."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _hash_value(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _normalized_field(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _nested_keys(value: Any, path: str = "$") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            rows.append((str(key), child))
            rows.extend(_nested_keys(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_nested_keys(item, f"{path}[{index}]"))
    return rows


def response_feature_errors(value: Any) -> list[str]:
    """Reject metadata that can reveal outcome or provenance shortcuts."""

    errors: list[str] = []
    for key, path in _nested_keys(value):
        normalized = _normalized_field(key)
        if any(token in normalized for token in PROHIBITED_RESPONSE_FIELDS):
            errors.append(f"prohibited_response_feature:{path}")
    return errors


def runtime_prohibited_feature_rows() -> list[JsonDict]:
    """Mutate every denied field and retain the rejection receipt."""

    rows = []
    for field in PROHIBITED_RESPONSE_FIELDS:
        errors = response_feature_errors({"nested": {field: "probe"}})
        rows.append(
            {
                "field": field,
                "errors": errors,
                "passed": bool(errors),
                "terminal": True,
            }
        )
    return rows


def _quantization_from_filename(filename: str) -> str:
    match = re.search(r"(?:UD-)?(Q\d(?:_[A-Z0-9]+)+)", filename.upper())
    return match.group(1) if match else PREFERRED_QUANT


def resolve_model_specs(
    *,
    cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the cached mandated pair first, then the required third family."""

    pair = cached_pair_func(gpu_indices=(0, 1)) or []
    cached = {
        str(row.get("hf_id")): str(row.get("model_path") or "") for row in pair if row.get("hf_id")
    }
    rows: list[JsonDict] = []
    for repository in REQUIRED_MODEL_IDS:
        path = cached.get(repository) or resolver(repository, PREFERRED_QUANT) or ""
        filename = Path(path).name if path else ""
        rows.append(
            {
                "model_repository": repository,
                "model_path": path,
                "filename": filename,
                "quantization": _quantization_from_filename(filename),
                "headline_eligible": True,
                "cpu_smoke_only": False,
            }
        )
    return rows


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject missing families, legacy substitutions, and unsafe file shapes."""

    errors: list[str] = []
    if [row.get("model_repository") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_family_roster")
    for row in rows:
        path = str(row.get("model_path") or "")
        filename = str(row.get("filename") or "")
        if not path or Path(path).suffix.casefold() != ".gguf" or "mmproj" in filename.casefold():
            errors.append("primary_gguf_path")
        if row.get("headline_eligible") is not True:
            errors.append("headline_eligibility")
        if row.get("cpu_smoke_only") is True:
            errors.append("headline_cpu_smoke")
    return list(dict.fromkeys(errors))


def cpu_smoke_errors(row: Mapping[str, Any]) -> list[str]:
    """Keep an optional legacy CPU smoke outside every result row."""

    errors: list[str] = []
    if row.get("cpu_smoke_only") is not True or row.get("headline_eligible") is not False:
        errors.append("cpu_smoke_declaration")
    if int(row.get("result_row_count", 0) or 0) != 0:
        errors.append("cpu_smoke_populated_result_rows")
    return errors


MODEL_SPECS = resolve_model_specs()


def gate_check(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record one check with exact expected and observed values."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all checks and promote the first failed comparison."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def response_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one raw response without its self-referential digest."""

    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def token_position_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one response-position row without its own digest."""

    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def _log_probability(logits: Any, token_id: int) -> tuple[float, str]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ScoringError("null_or_malformed_logits")
    if token_id < 0 or token_id >= values.size:
        raise ScoringError("response_token_out_of_range")
    maximum = float(values.max())
    logsumexp = maximum + math.log(float(np.exp(values - maximum).sum()))
    return float(values[token_id] - logsumexp), sha256_bytes(
        np.asarray(values, dtype=np.float32).tobytes(order="C")
    )


def score_teacher_forced_response(
    model: Any,
    *,
    prompt_text: str,
    semantic_key: str,
    model_repository: str,
    request_id: str,
) -> tuple[JsonDict, list[JsonDict]]:
    """Score one fixed neutral response without generation or self-grading."""

    scored_prompt = SCORING_PREAMBLE + "\n" + prompt_text
    prompt_tokens = list(model.tokenize(scored_prompt.encode("utf-8"), add_bos=True, special=False))
    response_tokens = list(
        model.tokenize(FIXED_RESPONSE_TEXT.encode("utf-8"), add_bos=False, special=False)
    )
    if not prompt_tokens or not response_tokens:
        raise ScoringError("teacher_forced_token_alignment")
    combined_tokens = prompt_tokens + response_tokens
    if len(combined_tokens) > N_CTX:
        raise ScoringError(f"context_overflow:{len(combined_tokens)}")
    model.reset()
    model.eval(combined_tokens)
    scores = getattr(model, "scores", None)
    if scores is None:
        raise ScoringError("null_or_malformed_logits")
    positions: list[JsonDict] = []
    logprobs: list[float] = []
    for relative_position, token_id in enumerate(response_tokens):
        score_index = len(prompt_tokens) + relative_position - 1
        try:
            score_row = scores[score_index]
        except (IndexError, TypeError) as exc:
            raise ScoringError("null_or_malformed_logits") from exc
        logprob, vector_hash = _log_probability(score_row, int(token_id))
        logprobs.append(logprob)
        position_row: JsonDict = {
            "semantic_key": semantic_key,
            "model_repository": model_repository,
            "relative_position": relative_position,
            "token_id": int(token_id),
            "token_log_probability": logprob,
            "logit_vector_hash": vector_hash,
            "terminal": True,
        }
        position_row["row_hash"] = token_position_row_hash(position_row)
        positions.append(position_row)
    sequence = float(sum(logprobs))
    row: JsonDict = {
        "semantic_key": semantic_key,
        "model_repository": model_repository,
        "prompt_hash": sha256_text(prompt_text),
        "prompt_char_count": len(prompt_text),
        "prompt_word_count": len(prompt_text.split()),
        "scoring_preamble_hash": sha256_text(SCORING_PREAMBLE),
        "fixed_response_hash": sha256_text(FIXED_RESPONSE_TEXT),
        "response_token_ids": [int(token) for token in response_tokens],
        "response_positions": list(range(len(response_tokens))),
        "response_token_count": len(response_tokens),
        "sequence_log_likelihood": sequence,
        "normalized_sequence_log_likelihood": sequence / len(response_tokens),
        "request_id": request_id,
        "terminal": True,
        "status": "success",
        "error": None,
        "row_hash": "",
    }
    row["row_hash"] = response_row_hash(row)
    return row, positions


def failed_response_row(
    *,
    semantic_key: str,
    model_repository: str,
    prompt_text: str,
    request_id: str,
    error_text: str,
) -> JsonDict:
    """Retain one terminal failed prompt-family cell in the denominator."""

    row: JsonDict = {
        "semantic_key": semantic_key,
        "model_repository": model_repository,
        "prompt_hash": sha256_text(prompt_text),
        "prompt_char_count": len(prompt_text),
        "prompt_word_count": len(prompt_text.split()),
        "scoring_preamble_hash": sha256_text(SCORING_PREAMBLE),
        "fixed_response_hash": sha256_text(FIXED_RESPONSE_TEXT),
        "response_token_ids": [],
        "response_positions": [],
        "response_token_count": 0,
        "sequence_log_likelihood": None,
        "normalized_sequence_log_likelihood": None,
        "request_id": request_id,
        "terminal": True,
        "status": "failed",
        "error": error_text,
        "row_hash": "",
    }
    row["row_hash"] = response_row_hash(row)
    return row


def response_row_errors(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_semantic_keys: Sequence[str] | None = None,
) -> list[str]:
    """Validate raw row uniqueness, hashes, terminal state, and finite scores."""

    errors: list[str] = []
    identities = [(str(row.get("semantic_key")), str(row.get("model_repository"))) for row in rows]
    if len(identities) != len(set(identities)):
        errors.append("duplicate_response_row")
    if expected_semantic_keys is not None:
        expected = {
            (str(key), model) for key in expected_semantic_keys for model in REQUIRED_MODEL_IDS
        }
        if set(identities) != expected:
            errors.append("response_row_cross_product")
    for row in rows:
        if row.get("row_hash") != response_row_hash(row):
            errors.append("response_row_hash")
        if row.get("terminal") is not True or row.get("status") not in {"success", "failed"}:
            errors.append("response_row_terminal")
        if row.get("status") == "success":
            tokens = row.get("response_token_ids")
            positions = row.get("response_positions")
            count = row.get("response_token_count")
            values = (
                row.get("sequence_log_likelihood"),
                row.get("normalized_sequence_log_likelihood"),
            )
            if (
                not isinstance(tokens, list)
                or not tokens
                or positions != list(range(len(tokens)))
                or count != len(tokens)
            ):
                errors.append("response_token_alignment")
            if not all(
                isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values
            ):
                errors.append("response_score_non_finite")
        elif not str(row.get("error") or ""):
            errors.append("failed_response_without_error")
    errors.extend(response_feature_errors(rows))
    return list(dict.fromkeys(errors))


def token_position_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate every stored response position without retaining full logits."""

    errors: list[str] = []
    identities = [
        (
            str(row.get("semantic_key")),
            str(row.get("model_repository")),
            row.get("relative_position"),
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        errors.append("duplicate_token_position_row")
    for row in rows:
        if row.get("row_hash") != token_position_row_hash(row):
            errors.append("token_position_row_hash")
        value = row.get("token_log_probability")
        if (
            row.get("terminal") is not True
            or type(row.get("relative_position")) is not int
            or type(row.get("token_id")) is not int
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not _hash_value(row.get("logit_vector_hash"))
        ):
            errors.append("token_position_invalid")
    errors.extend(response_feature_errors(rows))
    return list(dict.fromkeys(errors))


def _freeze_hash(manifest: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in manifest.items() if key != "response_freeze_hash"}
    )


def build_response_freeze(
    rows: Sequence[Mapping[str, Any]],
    token_position_rows: Sequence[Mapping[str, Any]],
    *,
    prompt_freeze_hash: str,
    frozen_at: str,
) -> JsonDict:
    """Seal all label-free rows and positions before authority access."""

    row_errors = response_row_errors(rows)
    position_errors = token_position_errors(token_position_rows)
    if row_errors or position_errors:
        raise FreezeError(f"response_rows_invalid:{row_errors + position_errors}")
    manifest: JsonDict = {
        "schema": SCHEMA + ".response_freeze",
        "prompt_freeze_hash": prompt_freeze_hash,
        "response_frozen_at": frozen_at,
        "response_row_count": len(rows),
        "token_position_row_count": len(token_position_rows),
        "response_rows_hash": sha256_json(rows),
        "token_position_rows_hash": sha256_json(token_position_rows),
        "response_freeze_hash": "",
    }
    manifest["response_freeze_hash"] = _freeze_hash(manifest)
    return manifest


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise LabelAccessError("response_freeze_missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabelAccessError("response_freeze_missing") from exc
    if parsed.tzinfo is None:
        raise LabelAccessError("response_freeze_missing")
    return parsed


def assert_label_access_allowed(freeze_manifest: Mapping[str, Any], label_opened_at: str) -> None:
    """Permit label access only after a self-consistent response freeze."""

    if (
        not freeze_manifest
        or not _hash_value(freeze_manifest.get("response_freeze_hash"))
        or freeze_manifest.get("response_freeze_hash") != _freeze_hash(freeze_manifest)
    ):
        raise LabelAccessError("response_freeze_missing")
    frozen = _parse_time(freeze_manifest.get("response_frozen_at"))
    opened = _parse_time(label_opened_at)
    if opened <= frozen:
        raise LabelAccessError("label_open_not_after_freeze")


def _condition_role(sidecar: Mapping[str, Any]) -> str:
    surface = str(sidecar.get("surface_variant"))
    role = str(sidecar.get("pair_role"))
    if surface == "primary" and role == "clean":
        return "primary_clean"
    if surface == "primary" and role == "changed":
        return "primary_intervention"
    if surface == "isomorphic" and role == "clean":
        return "isomorphic_clean"
    if surface == "isomorphic" and role == "changed":
        return "isomorphic_intervention"
    raise LabelAccessError("condition_role_invalid")


def _labels_match_direction(sidecar_rows: Sequence[Mapping[str, Any]]) -> bool:
    by_surface_role = {
        (str(row.get("surface_variant")), str(row.get("pair_role"))): row for row in sidecar_rows
    }
    if set(by_surface_role) != {
        ("primary", "clean"),
        ("primary", "changed"),
        ("isomorphic", "clean"),
        ("isomorphic", "changed"),
    }:
        return False
    directions = {str(row.get("intervention_direction")) for row in sidecar_rows}
    if len(directions) != 1 or next(iter(directions)) not in {"violation", "repair"}:
        return False
    expected_clean = "equivalent"
    expected_changed = "non_equivalent"
    return all(
        str(row.get("exact_label")) == (expected_clean if role == "clean" else expected_changed)
        for (_surface, role), row in by_surface_role.items()
    )


def _scientific_condition(role: str, direction: str) -> str:
    if role == "primary_clean":
        return "clean"
    if role == "primary_intervention":
        return f"minimal_{direction}"
    if role == "isomorphic_clean":
        return "isomorphic_clean"
    return f"isomorphic_{direction}"


def join_frozen_conditions(
    rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    *,
    freeze_manifest: Mapping[str, Any],
    label_opened_at: str,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Assign block roles after freeze and derive signed pair responses."""

    assert_label_access_allowed(freeze_manifest, label_opened_at)
    if freeze_manifest.get("response_rows_hash") != sha256_json(rows):
        raise LabelAccessError("response_freeze_hash_mismatch")
    sidecar_by_key = {str(row.get("semantic_key")): dict(row) for row in sidecar_rows}
    if len(sidecar_by_key) != len(sidecar_rows):
        raise LabelAccessError("duplicate_sidecar_key")
    semantic_keys = {str(row.get("semantic_key")) for row in rows}
    if semantic_keys != set(sidecar_by_key):
        raise LabelAccessError("sidecar_key_mismatch")
    blocks: dict[str, list[JsonDict]] = defaultdict(list)
    for sidecar in sidecar_rows:
        blocks[str(sidecar.get("block_id"))].append(dict(sidecar))
    if any(len(block) != 4 or not _labels_match_direction(block) for block in blocks.values()):
        raise LabelAccessError("sidecar_block_invalid")

    condition_rows: list[JsonDict] = []
    for raw_value in rows:
        raw = dict(raw_value)
        sidecar = sidecar_by_key[str(raw["semantic_key"])]
        role = _condition_role(sidecar)
        direction = str(sidecar["intervention_direction"])
        condition_rows.append(
            {
                "pair_id": sidecar["block_id"],
                "semantic_key": raw["semantic_key"],
                "model_repository": raw["model_repository"],
                "condition_role": role,
                "scientific_condition": _scientific_condition(role, direction),
                "intervention_direction": direction,
                "evaluation_partition": sidecar.get("split"),
                "held_source_family": sha256_text(str(sidecar.get("source_family"))),
                "prompt_hash": raw["prompt_hash"],
                "prompt_char_count": raw["prompt_char_count"],
                "prompt_word_count": raw["prompt_word_count"],
                "scoring_preamble_hash": raw["scoring_preamble_hash"],
                "fixed_response_hash": raw["fixed_response_hash"],
                "response_token_ids": deepcopy(raw["response_token_ids"]),
                "response_positions": deepcopy(raw["response_positions"]),
                "sequence_log_likelihood": raw["sequence_log_likelihood"],
                "normalized_sequence_log_likelihood": raw["normalized_sequence_log_likelihood"],
                "status": raw["status"],
                "terminal": raw["terminal"],
                "label_transition_validated_after_freeze": True,
            }
        )

    signed_rows: list[JsonDict] = []
    grouped: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in condition_rows:
        grouped[(str(row["pair_id"]), str(row["model_repository"]))].append(row)
    for (pair_id, model_repository), group in sorted(grouped.items()):
        by_role = {str(row["condition_role"]): row for row in group}
        if set(by_role) != set(CONDITION_ROLES):
            continue
        direction = str(by_role["primary_intervention"]["intervention_direction"])
        values = {
            role: by_role[role].get("normalized_sequence_log_likelihood")
            for role in CONDITION_ROLES
        }
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values.values()
        ):
            continue
        clean = float(values["primary_clean"])
        changed = float(values["primary_intervention"])
        iso_clean = float(values["isomorphic_clean"])
        iso_changed = float(values["isomorphic_intervention"])
        primary_signed = clean - changed
        isomorphic_signed = iso_clean - iso_changed
        signed_rows.append(
            {
                "pair_id": pair_id,
                "model_repository": model_repository,
                "comparison": f"clean_to_{direction}",
                "signed_primary_delta": primary_signed,
                "signed_isomorphic_delta": isomorphic_signed,
                "isomorphic_clean_delta": iso_clean - clean,
                "isomorphic_intervention_delta": iso_changed - changed,
                "isomorphic_signed_control_delta": isomorphic_signed - primary_signed,
                "evaluation_partition": by_role["primary_clean"]["evaluation_partition"],
                "held_source_family": by_role["primary_clean"]["held_source_family"],
                "terminal": True,
            }
        )
    return condition_rows, signed_rows


def alignment_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check each four-condition block for prompt and response alignment."""

    errors: list[str] = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("pair_id")), str(row.get("model_repository")))].append(row)
    for group in grouped.values():
        roles = [str(row.get("condition_role")) for row in group]
        if len(group) != 4 or set(roles) != set(CONDITION_ROLES) or len(roles) != len(set(roles)):
            errors.append("condition_cell_shape")
            continue
        prompt_controls = {
            (
                row.get("prompt_char_count"),
                row.get("prompt_word_count"),
                row.get("scoring_preamble_hash"),
                row.get("fixed_response_hash"),
            )
            for row in group
        }
        if len(prompt_controls) != 1:
            errors.append("prompt_mismatch")
        token_controls = {
            (
                tuple(row.get("response_token_ids") or []),
                tuple(row.get("response_positions") or []),
            )
            for row in group
        }
        if len(token_controls) != 1 or any(
            list(row.get("response_positions") or [])
            != list(range(len(row.get("response_token_ids") or [])))
            for row in group
        ):
            errors.append("token_position_drift")
    return list(dict.fromkeys(errors))


def bootstrap_interval(values: Sequence[float], *, seed: int, draws: int = 2000) -> JsonDict:
    """Compute a deterministic paired percentile interval."""

    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "pair_count": 0}
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "pair_count": len(values),
        "bootstrap_draw_count": draws,
    }


def _direction(value: float | None) -> str:
    if value is None or abs(value) <= TIE_EPSILON:
        return "tie"
    return "positive" if value > 0 else "reversal"


def summarize_signed_responses(
    rows: Sequence[Mapping[str, Any]], *, draws: int = 2000
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Keep family, held-source, reversal, and exact-tie effects separate."""

    family_rows: list[JsonDict] = []
    for family_index, model in enumerate(REQUIRED_MODEL_IDS):
        selected = [row for row in rows if row.get("model_repository") == model]
        primary = [float(row["signed_primary_delta"]) for row in selected]
        controls = [float(row["isomorphic_signed_control_delta"]) for row in selected]
        primary_interval = bootstrap_interval(primary, seed=RANDOM_SEED + family_index, draws=draws)
        control_interval = bootstrap_interval(
            controls, seed=RANDOM_SEED + 100 + family_index, draws=draws
        )
        family_rows.append(
            {
                "model_repository": model,
                "pooled_families": False,
                "primary_signed_effect": primary_interval,
                "primary_direction": _direction(primary_interval["mean"]),
                "isomorphic_control_effect": control_interval,
                "isomorphic_control_direction": _direction(control_interval["mean"]),
                "terminal": True,
            }
        )

    held_rows: list[JsonDict] = []
    held_groups = sorted(
        {
            (str(row.get("model_repository")), str(row.get("held_source_family")))
            for row in rows
            if row.get("evaluation_partition") == "held_source"
        }
    )
    for group_index, (model, source_family) in enumerate(held_groups):
        values = [
            float(row["signed_primary_delta"])
            for row in rows
            if row.get("model_repository") == model
            and row.get("held_source_family") == source_family
            and row.get("evaluation_partition") == "held_source"
        ]
        interval = bootstrap_interval(values, seed=RANDOM_SEED + 1000 + group_index, draws=draws)
        held_rows.append(
            {
                "model_repository": model,
                "held_source_family": source_family,
                "split": "held_source",
                "effect": interval,
                "direction": _direction(interval["mean"]),
                "pooled_families": False,
                "terminal": True,
            }
        )

    ties: list[JsonDict] = []
    for row in rows:
        for field, kind in (
            ("signed_primary_delta", "primary_signed_delta"),
            ("isomorphic_signed_control_delta", "isomorphic_control_delta"),
        ):
            value = float(row[field])
            if abs(value) <= TIE_EPSILON:
                ties.append(
                    {
                        "pair_id": row.get("pair_id"),
                        "model_repository": row.get("model_repository"),
                        "tie_kind": kind,
                        "value": value,
                        "terminal": True,
                    }
                )
    return family_rows, held_rows, ties


def per_pair_results(
    condition_rows: Sequence[Mapping[str, Any]],
    signed_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Summarize every block without pooling its three model families."""

    pair_ids = sorted({str(row.get("pair_id")) for row in condition_rows})
    output = []
    for pair_id in pair_ids:
        conditions = [row for row in condition_rows if row.get("pair_id") == pair_id]
        signed = [row for row in signed_rows if row.get("pair_id") == pair_id]
        output.append(
            {
                "pair_id": pair_id,
                "condition_cell_count": len(conditions),
                "expected_condition_cell_count": EXPECTED_FAMILY_COUNT * len(CONDITION_ROLES),
                "family_results": [deepcopy(dict(row)) for row in signed],
                "terminal": len(conditions) == EXPECTED_FAMILY_COUNT * len(CONDITION_ROLES),
            }
        )
    return output


def server_receipt_errors(row: Mapping[str, Any]) -> list[str]:
    """Reject CPU fallback, stale health identity, and non-owned processes."""

    errors: list[str] = []
    if row.get("owned_by_task") is not True:
        errors.append("non_owned_process")
    if (
        row.get("health_pid") != row.get("pid")
        or row.get("health_model_repository") != row.get("model_repository")
        or row.get("health_task_id") != row.get("task_id")
        or not isinstance(row.get("health_monotonic_ns"), int)
        or not isinstance(row.get("started_monotonic_ns"), int)
        or int(row.get("health_monotonic_ns", 0)) <= int(row.get("started_monotonic_ns", 0))
    ):
        errors.append("stale_server")
    if (
        row.get("backend") != "cuda"
        or row.get("live_cuda") is not True
        or int(row.get("offloaded_layers", 0) or 0) <= 0
    ):
        errors.append("cpu_fallback")
    required = (
        "gpu_uuid",
        "cuda_device",
        "n_ctx",
        "port",
        "command_hash",
    )
    if row.get("process_alive_at_health") is not True or any(
        row.get(field) in {None, ""} for field in required
    ):
        errors.append("server_receipt_incomplete")
    return list(dict.fromkeys(errors))


def _exact_family_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    return [row.get("model_repository") for row in rows] == list(REQUIRED_MODEL_IDS)


def completion_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute completion from cells, CUDA receipts, counters, and freeze order."""

    errors: list[str] = []
    if artifact.get("preconditions_checked", {}).get("all_passed") is not True:
        errors.append("preconditions_incomplete")
    if model_spec_errors(artifact.get("MODEL_SPECS", [])):
        errors.append("model_specs_incomplete")
    rows = artifact.get("rows", [])
    positions = artifact.get("token_position_rows", [])
    conditions = artifact.get("condition_rows", [])
    if len(rows) != EXPECTED_CELL_COUNT or response_row_errors(rows):
        errors.append("response_rows_incomplete")
    if token_position_errors(positions):
        errors.append("token_position_rows_incomplete")
    pair_ids = {str(row.get("pair_id")) for row in conditions}
    cell_ids = [
        (
            str(row.get("pair_id")),
            str(row.get("model_repository")),
            str(row.get("condition_role")),
        )
        for row in conditions
    ]
    if (
        len(pair_ids) != EXPECTED_PAIR_COUNT
        or len(conditions) != EXPECTED_CELL_COUNT
        or len(cell_ids) != len(set(cell_ids))
        or any(
            row.get("terminal") is not True or row.get("status") != "success" for row in conditions
        )
    ):
        errors.append("condition_cells_incomplete")
    errors.extend(alignment_errors(conditions))
    if len(artifact.get("signed_response_rows", [])) != EXPECTED_PAIR_COUNT * EXPECTED_FAMILY_COUNT:
        errors.append("signed_response_rows_incomplete")
    if not _exact_family_rows(artifact.get("model_rows", [])):
        errors.append("model_rows_incomplete")
    gpu_rows = artifact.get("gpu_identity_rows", [])
    if not _exact_family_rows(gpu_rows) or any(
        row.get("live_cuda") is not True
        or row.get("foreign_processes_before") != []
        or row.get("foreign_processes_resident") != []
        or row.get("foreign_processes_after") != []
        for row in gpu_rows
    ):
        errors.append("gpu_identity_incomplete")
    lease_rows = artifact.get("gpu_lease_rows", [])
    if not _exact_family_rows(lease_rows) or any(
        row.get("owner_verified") is not True
        or row.get("released") is not True
        or row.get("terminal_phase") != "terminal_complete"
        for row in lease_rows
    ):
        errors.append("gpu_lease_incomplete")
    server_rows = artifact.get("server_rows", [])
    if not _exact_family_rows(server_rows) or any(
        server_receipt_errors(row) for row in server_rows
    ):
        errors.append("server_rows_incomplete")
    request_rows = artifact.get("request_counter_rows", [])
    if not _exact_family_rows(request_rows) or any(
        row.get("expected_requests") != EXPECTED_PROMPT_COUNT
        or row.get("observed_requests") != EXPECTED_PROMPT_COUNT
        for row in request_rows
    ):
        errors.append("request_counters_incomplete")
    completion_rows = artifact.get("completion_counter_rows", [])
    if not _exact_family_rows(completion_rows) or any(
        row.get("expected_completions") != EXPECTED_PROMPT_COUNT
        or row.get("observed_completions") != EXPECTED_PROMPT_COUNT
        or row.get("error_count") != 0
        for row in completion_rows
    ):
        errors.append("completion_counters_incomplete")
    teardown_rows = artifact.get("teardown_rows", [])
    if not _exact_family_rows(teardown_rows) or any(
        row.get("passed") is not True
        or row.get("process_exit_confirmed") is not True
        or row.get("port_release_confirmed") is not True
        or row.get("model_close_called") is not True
        or row.get("lease_released") is not True
        or row.get("unrelated_process_kill_count_delta") != 0
        for row in teardown_rows
    ):
        errors.append("teardown_incomplete")
    try:
        freeze = build_response_freeze(
            rows,
            positions,
            prompt_freeze_hash=str(artifact.get("prompt_freeze_hash")),
            frozen_at=str(artifact.get("response_frozen_at")),
        )
        if freeze.get("response_freeze_hash") != artifact.get("response_freeze_hash"):
            errors.append("response_freeze_mismatch")
        assert_label_access_allowed(freeze, str(artifact.get("label_opened_at")))
    except (FreezeError, LabelAccessError):
        errors.append("freeze_or_label_order_incomplete")
    if float(artifact.get("duration_s", 0.0) or 0.0) < LIVE_DURATION_FLOOR_S:
        errors.append("implausible_live_duration")
    hashes = artifact.get("model_file_hashes", {})
    if set(hashes) != set(REQUIRED_MODEL_IDS) or not all(
        _hash_value(value) for value in hashes.values()
    ):
        errors.append("model_file_hashes_incomplete")
    if not _hash_value(artifact.get("llama_binary_hash")) or not _hash_value(
        artifact.get("command_hash")
    ):
        errors.append("runtime_hashes_incomplete")
    if not artifact.get("prohibited_feature_rows") or any(
        row.get("passed") is not True for row in artifact.get("prohibited_feature_rows", [])
    ):
        errors.append("prohibited_feature_audit_incomplete")
    return list(dict.fromkeys(errors))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact without its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def _copy_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [deepcopy(dict(row)) for row in rows]


def _science_positive(family_rows: Sequence[Mapping[str, Any]]) -> bool:
    positive = 0
    reversal = False
    for row in family_rows:
        interval = row.get("primary_signed_effect")
        interval = interval if isinstance(interval, Mapping) else {}
        low = interval.get("ci_low")
        high = interval.get("ci_high")
        positive += isinstance(low, (int, float)) and float(low) > 0.0
        reversal = reversal or (isinstance(high, (int, float)) and float(high) < 0.0)
    return positive >= 2 and not reversal


def build_artifact(
    *,
    duration_s: float,
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    model_file_hashes: Mapping[str, Any],
    llama_binary_hash: str | None,
    command_hash: str | None,
    source_artifact_hashes: Mapping[str, Any],
    prompt_freeze_hash: str,
    response_freeze_hash: str | None = None,
    label_opened_at: str | None = None,
    response_frozen_at: str | None = None,
    model_rows: Sequence[Mapping[str, Any]] = (),
    gpu_identity_rows: Sequence[Mapping[str, Any]] = (),
    gpu_lease_rows: Sequence[Mapping[str, Any]] = (),
    server_rows: Sequence[Mapping[str, Any]] = (),
    request_counter_rows: Sequence[Mapping[str, Any]] = (),
    completion_counter_rows: Sequence[Mapping[str, Any]] = (),
    teardown_rows: Sequence[Mapping[str, Any]] = (),
    rows: Sequence[Mapping[str, Any]] = (),
    per_pair_results: Sequence[Mapping[str, Any]] = (),
    condition_rows: Sequence[Mapping[str, Any]] = (),
    token_position_rows: Sequence[Mapping[str, Any]] = (),
    signed_response_rows: Sequence[Mapping[str, Any]] = (),
    family_effect_rows: Sequence[Mapping[str, Any]] = (),
    held_source_effect_rows: Sequence[Mapping[str, Any]] = (),
    tie_rows: Sequence[Mapping[str, Any]] = (),
    prohibited_feature_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build a blocked, partial, complete-null, or complete-positive artifact."""

    condition_values = _copy_rows(condition_rows)
    model_values = _copy_rows(model_rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "MODEL_SPECS": _copy_rows(model_specs),
        "model_rows": model_values,
        "model_file_hashes": deepcopy(dict(model_file_hashes)),
        "llama_binary_hash": llama_binary_hash,
        "command_hash": command_hash,
        "gpu_identity_rows": _copy_rows(gpu_identity_rows),
        "gpu_lease_rows": _copy_rows(gpu_lease_rows),
        "server_rows": _copy_rows(server_rows),
        "request_counter_rows": _copy_rows(request_counter_rows),
        "completion_counter_rows": _copy_rows(completion_counter_rows),
        "teardown_rows": _copy_rows(teardown_rows),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "prompt_freeze_hash": prompt_freeze_hash,
        "response_freeze_hash": response_freeze_hash,
        "label_opened_at": label_opened_at,
        "response_frozen_at": response_frozen_at,
        "rows": _copy_rows(rows),
        "per_pair_results": _copy_rows(per_pair_results),
        "condition_rows": condition_values,
        "token_position_rows": _copy_rows(token_position_rows),
        "signed_response_rows": _copy_rows(signed_response_rows),
        "family_effect_rows": _copy_rows(family_effect_rows),
        "held_source_effect_rows": _copy_rows(held_source_effect_rows),
        "tie_rows": _copy_rows(tie_rows),
        "failed_cell_rows": [
            deepcopy(dict(row)) for row in condition_values if row.get("status") != "success"
        ],
        "prohibited_feature_rows": _copy_rows(prohibited_feature_rows),
        "expected_family_count": EXPECTED_FAMILY_COUNT,
        "observed_family_count": len(
            {
                str(row.get("model_repository"))
                for row in model_values
                if row.get("terminal") is True
            }
        ),
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "observed_pair_count": len(
            {str(row.get("pair_id")) for row in condition_values if row.get("pair_id")}
        ),
        "intervention_surface_complete_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_intervention_surface",
    }
    checks = _copy_rows(preconditions.get("checks", []))
    preflight_passed = preconditions.get("all_passed") is True
    integrity_errors = completion_errors(artifact) if preflight_passed else []
    complete = preflight_passed and not integrity_errors
    artifact["intervention_surface_complete_score"] = int(complete)
    if preflight_passed:
        checks.append(gate_check("intervention_surface_completion", [], integrity_errors))
    artifact["gate_check_summary"] = gate_summary(checks)
    if not preflight_passed:
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = "blocked_intervention_surface"
    elif not complete:
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = "partial_intervention_surface"
    elif _science_positive(artifact["family_effect_rows"]):
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = "complete: signed_intervention_response_positive"
    else:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = "complete_null_signed_intervention_response"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _verdict_prefix_matches(verdict_class: str, verdict: str) -> bool:
    prefixes = {
        "positive": ("complete:", "complete_"),
        "circular_positive": ("circular_positive:",),
        "null": ("complete_null",),
        "blocked": ("blocked_",),
        "disqualified": ("complete_disqualified", "disqualified_"),
        "partial": ("partial_",),
    }
    return verdict.startswith(prefixes.get(verdict_class, ()))


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check fields, principles, hashes, rows, receipts, and verdict."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        return [f"required_fields_missing:{sorted(missing)}"]
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    for field, expected in (
        ("expected_family_count", EXPECTED_FAMILY_COUNT),
        ("expected_pair_count", EXPECTED_PAIR_COUNT),
    ):
        if type(artifact.get(field)) is not int or artifact.get(field) != expected:
            errors.append(f"fixed_count_mismatch:{field}")
    if type(artifact.get("intervention_surface_complete_score")) is not int:
        errors.append("completion_score_not_bare_int")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    errors.extend(response_feature_errors(artifact.get("rows", [])))
    errors.extend(response_feature_errors(artifact.get("token_position_rows", [])))
    if response_row_errors(artifact.get("rows", [])):
        errors.append("response_rows_invalid")
    if token_position_errors(artifact.get("token_position_rows", [])):
        errors.append("token_position_rows_invalid")
    recomputed_complete = int(
        artifact.get("preconditions_checked", {}).get("all_passed") is True
        and not completion_errors(artifact)
    )
    if artifact.get("intervention_surface_complete_score") != recomputed_complete:
        errors.append("completion_score_mismatch")
    if (
        artifact.get("intervention_surface_complete_score") == 1
        and float(artifact.get("duration_s", 0.0)) < LIVE_DURATION_FLOOR_S
    ):
        errors.append("implausible_live_duration")
    verdict_class = str(artifact.get("verdict_class"))
    verdict = str(artifact.get("honest_verdict"))
    if verdict_class not in VERDICT_CLASSES or not _verdict_prefix_matches(verdict_class, verdict):
        errors.append("verdict_prefix_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _utc_now() -> str:  # pragma: no cover - wall-clock boundary.
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _storage_writable(path: Path) -> bool:  # pragma: no cover - host filesystem boundary.
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".exp7013-write-probe-", dir=path)
        os.close(descriptor)
        Path(temporary).unlink()
        return True
    except OSError:
        return False


def _llama_binary_path() -> Path | None:  # pragma: no cover - installed binding boundary.
    try:
        from llama_cpp import llama_cpp

        path = Path(str(getattr(llama_cpp._lib, "_name", "")))
        return path if path.is_file() else None
    except Exception:
        return None


def _source_hashes() -> dict[str, str | None]:  # pragma: no cover - filesystem boundary.
    paths = {
        "exp7012": EXP7012_PATH,
        "learner_prompts": LEARNER_PROMPT_PATH,
        "authority_sidecar": AUTHORITY_SIDECAR_PATH,
        "module": Path(__file__),
        "wrapper": WRAPPER_PATH,
        "tests": TEST_PATH,
        "spec": SPEC_PATH,
    }
    return {name: sha256_file(path) if path.is_file() else None for name, path in paths.items()}


def _free_port() -> int:  # pragma: no cover - live socket boundary.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _device_processes(gpu: Mapping[str, Any], device_uuid: str) -> list[JsonDict]:
    return [
        deepcopy(dict(row))
        for row in gpu.get("processes", [])
        if str(row.get("gpu_uuid")) == device_uuid
    ]


def _probe_lease(runtime_dir: Path, device: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    lease = None
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=runtime_dir,
            task_id=f"{EXPERIMENT_ID}-preflight",
            device_uuid=str(device["uuid"]),
            expected_model="preflight-no-model",
            vram_before_mb=int(device.get("memory_used_mb", 0) or 0),
            ttl_s=30.0,
        )
        owner = lease.owner_receipt()
        lease.transition("terminal_blocked")
        release = lease.release()
        return {
            "available": owner.get("task_id") == f"{EXPERIMENT_ID}-preflight"
            and release.get("released") is True,
            "owner": owner,
            "release": release,
        }
    except Exception as exc:
        if lease is not None:
            lease.close()
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_preconditions(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    result_path: Path,
    checkpoint_root: Path,
    lease_runtime_dir: Path,
) -> JsonDict:  # pragma: no cover - live cache, CUDA, lease, and disk boundary.
    """Check frozen inputs and live resources before any model loads."""

    source_hashes = _source_hashes()
    binding = llama_cpp_probe()
    inventory = gpu_inventory()
    devices = list(inventory.get("devices", []))
    largest_model_mb = max(
        (
            math.ceil(Path(str(row.get("model_path"))).stat().st_size / (1024 * 1024))
            for row in model_specs
            if Path(str(row.get("model_path"))).is_file()
        ),
        default=0,
    )
    required_free_mb = largest_model_mb + VRAM_RESERVE_MB
    eligible = [
        row
        for row in devices
        if int(row.get("memory_free_mb", 0) or 0) >= required_free_mb
        and not _device_processes(inventory, str(row.get("uuid")))
    ]
    device = max(eligible, key=lambda row: int(row.get("memory_free_mb", 0))) if eligible else {}
    lease_probe = _probe_lease(lease_runtime_dir, device) if device else {"available": False}
    binary = _llama_binary_path()
    files = {
        str(row.get("model_repository")): Path(str(row.get("model_path"))).is_file()
        for row in model_specs
    }
    checks = [
        gate_check(
            "exp7012_frozen_ready_artifact", EXPECTED_EXP7012_HASH, source_hashes["exp7012"]
        ),
        gate_check(
            "learner_prompt_freeze_hash",
            EXPECTED_LEARNER_PROMPT_HASH,
            source_hashes["learner_prompts"],
        ),
        gate_check(
            "authority_sidecar_freeze_hash",
            EXPECTED_AUTHORITY_SIDECAR_HASH,
            source_hashes["authority_sidecar"],
        ),
        gate_check("exact_model_specs", [], model_spec_errors(model_specs)),
        gate_check(
            "all_three_cached_gguf_files",
            {model: True for model in REQUIRED_MODEL_IDS},
            files,
        ),
        gate_check(
            "cuda_llama_cpp_health",
            {"importable": True, "gpu_offload": True, "binary": True},
            {
                "importable": binding.get("importable"),
                "gpu_offload": binding.get("gpu_offload"),
                "binary": binary is not None,
            },
        ),
        gate_check(
            "eligible_gpu_vram",
            {"minimum_free_mb": required_free_mb, "foreign_processes": []},
            {
                "minimum_free_mb": device.get("memory_free_mb") if device else None,
                "foreign_processes": _device_processes(inventory, str(device.get("uuid")))
                if device
                else list(inventory.get("processes", [])),
            },
            passed=bool(device),
        ),
        gate_check("owned_gpu_lease", True, lease_probe.get("available")),
        gate_check("writable_result_path", True, _storage_writable(result_path.parent)),
        gate_check("writable_checkpoint_path", True, _storage_writable(checkpoint_root)),
    ]
    return {
        "all_passed": all(row["passed"] is True for row in checks),
        "checks": checks,
        "source_hashes": source_hashes,
        "llama_cpp": binding,
        "llama_binary_path": str(binary) if binary else None,
        "gpu_inventory": inventory,
        "eligible_gpu": deepcopy(dict(device)),
        "required_free_vram_mb": required_free_mb,
        "lease_probe": lease_probe,
        "labels_opened": False,
    }


def load_learner_prompts(path: Path) -> list[JsonDict]:  # pragma: no cover - frozen input boundary.
    """Read only the three-field learner file and verify its frozen digest."""

    if sha256_file(path) != EXPECTED_LEARNER_PROMPT_HASH:
        raise FreezeError("learner_prompt_hash_mismatch")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, Mapping) or set(value) != {
            "semantic_key",
            "neutral_block_position",
            "prompt",
        }:
            raise FreezeError("learner_prompt_shape")
        rows.append(dict(value))
    keys = [str(row.get("semantic_key")) for row in rows]
    if len(rows) != EXPECTED_PROMPT_COUNT or len(keys) != len(set(keys)):
        raise FreezeError("learner_prompt_count_or_duplicate")
    return rows


def load_sidecar_after_freeze(
    path: Path, freeze_manifest: Mapping[str, Any], *, label_opened_at: str
) -> list[JsonDict]:  # pragma: no cover - authority boundary.
    """Open the exact sidecar only after a valid response freeze exists."""

    assert_label_access_allowed(freeze_manifest, label_opened_at)
    if sha256_file(path) != EXPECTED_AUTHORITY_SIDECAR_HASH:
        raise LabelAccessError("authority_sidecar_hash_mismatch")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise LabelAccessError("authority_sidecar_row_invalid")
        rows.append(dict(value))
    if len(rows) != EXPECTED_PROMPT_COUNT:
        raise LabelAccessError("authority_sidecar_count")
    return rows


class _ScoringEngine:  # pragma: no cover - live llama.cpp boundary.
    """Load one GGUF and expose fixed-response scoring only."""

    def __init__(self, model_path: str) -> None:
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            n_ubatch=N_UBATCH,
            seed=RANDOM_SEED,
            logits_all=True,
            use_mmap=True,
            use_mlock=False,
            verbose=True,
        )
        self.close_called = False

    def score(self, payload: Mapping[str, Any]) -> JsonDict:
        row, positions = score_teacher_forced_response(
            self.llm,
            prompt_text=str(payload["prompt_text"]),
            semantic_key=str(payload["semantic_key"]),
            model_repository=str(payload["model_repository"]),
            request_id=str(payload["request_id"]),
        )
        if row["prompt_hash"] != payload.get("prompt_hash"):
            raise ScoringError("worker_prompt_hash_mismatch")
        return {"row": row, "token_position_rows": positions}

    def close(self) -> None:
        self.llm.close()
        self.close_called = True


def _run_score_worker(
    *,
    model_path: str,
    model_repository: str,
    port: int,
    status_path: Path,
) -> int:  # pragma: no cover - live subprocess boundary.
    """Serve owned score requests and close the model on explicit shutdown."""

    engine = _ScoringEngine(model_path)
    counters = {"requests": 0, "completions": 0, "errors": 0}
    started_ns = time.monotonic_ns()

    class Server(HTTPServer):
        stop_requested = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            del fmt, args

        def _send(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(dict(payload), sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send(404, {"error": "not_found"})
                return
            self._send(
                200,
                {
                    "ok": True,
                    "pid": os.getpid(),
                    "task_id": EXPERIMENT_ID,
                    "model_repository": model_repository,
                    "started_monotonic_ns": started_ns,
                    **counters,
                },
            )

        def do_POST(self) -> None:
            server = self.server
            if self.path == "/counters":
                self._send(200, counters)
                return
            if self.path == "/shutdown":
                server.stop_requested = True
                self._send(200, {"shutdown": True, **counters})
                return
            if self.path != "/score":
                self._send(404, {"error": "not_found"})
                return
            size = int(self.headers.get("Content-Length", "0"))
            counters["requests"] += 1
            try:
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(payload, Mapping):
                    raise ScoringError("worker_payload_not_object")
                result = engine.score(payload)
                self._send(200, result)
            except Exception as exc:
                counters["errors"] += 1
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            finally:
                counters["completions"] += 1

    server = Server(("127.0.0.1", port), Handler)
    server.timeout = 1.0
    exit_code = 0
    try:
        while not server.stop_requested:
            server.handle_request()
    except Exception:
        exit_code = 1
    finally:
        try:
            engine.close()
        except Exception:
            exit_code = 1
        server.server_close()
        gc.collect()
        write_json_atomic(
            status_path,
            {
                "terminal": True,
                "model_close_called": engine.close_called,
                **counters,
                "exit_code": exit_code,
            },
        )
    return exit_code


def _post_json(port: int, path: str, payload: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    req = request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ScoringError("server_response_not_object")
    return dict(value)


def _read_health(port: int) -> JsonDict:  # pragma: no cover
    with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _gpu_snapshot(
    device_uuid: str, *, owned_pid: int | None, phase: str
) -> JsonDict:  # pragma: no cover
    inventory = gpu_inventory()
    device = next(
        (row for row in inventory.get("devices", []) if str(row.get("uuid")) == device_uuid),
        {},
    )
    processes = _device_processes(inventory, device_uuid)
    return {
        "phase": phase,
        "device": deepcopy(dict(device)),
        "processes": processes,
        "owned_processes": [row for row in processes if row.get("pid") == owned_pid],
        "foreign_processes": [row for row in processes if row.get("pid") != owned_pid],
    }


def _wait_for_gpu_release(
    device_uuid: str, *, owned_pid: int, timeout_s: float = 180.0
) -> JsonDict:  # pragma: no cover
    deadline = time.monotonic() + timeout_s
    latest = _gpu_snapshot(device_uuid, owned_pid=owned_pid, phase="after")
    while latest["owned_processes"] and time.monotonic() < deadline:
        time.sleep(1.0)
        latest = _gpu_snapshot(device_uuid, owned_pid=owned_pid, phase="after")
    return latest


def run_family(
    *,
    model: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    device: Mapping[str, Any],
    lease_runtime_dir: Path,
    work_root: Path,
) -> JsonDict:  # pragma: no cover - live CUDA and process boundary.
    """Score one family under one lease and return complete lifecycle receipts."""

    repository = str(model["model_repository"])
    family_slug = re.sub(r"[^a-z0-9]+", "-", repository.casefold()).strip("-")
    family_root = work_root / family_slug
    family_root.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    status_path = family_root / "worker_status.json"
    log_path = family_root / "worker.log"
    state_path = family_root / "owner.json"
    before = _gpu_snapshot(str(device["uuid"]), owned_pid=None, phase="before")
    lease = None
    process: OwnedLlamaCppProcess | None = None
    process_receipt: JsonDict = {}
    health: JsonDict = {}
    resident: JsonDict = {}
    after: JsonDict = {}
    teardown: JsonDict = {}
    lease_owner: JsonDict = {}
    lease_release: JsonDict = {}
    lease_error: str | None = None
    raw_rows: list[JsonDict] = []
    token_rows: list[JsonDict] = []
    counters: JsonDict = {"requests": 0, "completions": 0, "errors": 0}
    started = time.perf_counter()
    started_ns = time.monotonic_ns()
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_7013_three_family_intervention_surface",
        "--score-worker",
        "--model-path",
        str(model["model_path"]),
        "--model-repository",
        repository,
        "--port",
        str(port),
        "--status-path",
        str(status_path),
    ]
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=lease_runtime_dir,
            task_id=f"{EXPERIMENT_ID}:{family_slug}",
            device_uuid=str(device["uuid"]),
            expected_model=str(model["model_path"]),
            vram_before_mb=int(device.get("memory_used_mb", 0) or 0),
            ttl_s=LEASE_TTL_S,
        )
        lease_owner = lease.owner_receipt()
        lease.transition("admitted")
        lease.transition("loading")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(device["index"])
        process = OwnedLlamaCppProcess(
            command=command,
            port=port,
            env=environment,
            log_path=log_path,
            state_path=state_path,
        )
        process_receipt = process.launch()
        process_receipt["gpu_uuid"] = device["uuid"]
        process_receipt["cuda_device"] = device["index"]
        process_receipt["n_ctx"] = N_CTX
        health_wait = process.wait_for_health(HEALTH_TIMEOUT_S)
        if health_wait.get("ok") is not True:
            raise RuntimeError(f"worker_health_failed:{health_wait.get('reason')}")
        health = _read_health(port)
        resident = _gpu_snapshot(
            str(device["uuid"]), owned_pid=int(process_receipt["pid"]), phase="resident"
        )
        if not resident["owned_processes"] or resident["foreign_processes"]:
            raise RuntimeError("owned_cuda_residency_missing_or_foreign")
        owned_vram = sum(
            int(row.get("used_memory_mb", 0) or 0) for row in resident["owned_processes"]
        )
        lease.transition("resident", vram_mb=owned_vram)
        lease.transition("inferencing")
        for index, prompt in enumerate(prompt_rows):
            semantic_key = str(prompt["semantic_key"])
            prompt_text = str(prompt["prompt"])
            request_id = sha256_json(
                {"model_repository": repository, "semantic_key": semantic_key, "index": index}
            )
            try:
                response = process.post_json(
                    "/score",
                    {
                        "semantic_key": semantic_key,
                        "model_repository": repository,
                        "prompt_text": prompt_text,
                        "prompt_hash": sha256_text(prompt_text),
                        "request_id": request_id,
                    },
                    REQUEST_TIMEOUT_S,
                )
                row = response.get("row")
                positions = response.get("token_position_rows")
                if not isinstance(row, Mapping) or not isinstance(positions, list):
                    raise ScoringError("worker_score_shape")
                raw_rows.append(dict(row))
                token_rows.extend(dict(item) for item in positions if isinstance(item, Mapping))
            except Exception as exc:
                raw_rows.append(
                    failed_response_row(
                        semantic_key=semantic_key,
                        model_repository=repository,
                        prompt_text=prompt_text,
                        request_id=request_id,
                        error_text=f"{type(exc).__name__}: {exc}",
                    )
                )
            if index % 16 == 15:
                lease.heartbeat()
            if time.perf_counter() - started > MODEL_TIMEOUT_S:
                for pending in prompt_rows[index + 1 :]:
                    pending_text = str(pending["prompt"])
                    pending_key = str(pending["semantic_key"])
                    raw_rows.append(
                        failed_response_row(
                            semantic_key=pending_key,
                            model_repository=repository,
                            prompt_text=pending_text,
                            request_id=sha256_json(
                                {"model_repository": repository, "semantic_key": pending_key}
                            ),
                            error_text="model_timeout",
                        )
                    )
                break
        try:
            counters = process.post_json("/counters", {}, REQUEST_TIMEOUT_S)
            process.post_json("/shutdown", {}, REQUEST_TIMEOUT_S)
        except Exception as exc:
            counters["controller_counter_error"] = f"{type(exc).__name__}: {exc}"
        if process.process is not None:
            try:
                process.process.wait(timeout=120.0)
            except Exception:
                pass
    except Exception as exc:
        for prompt in prompt_rows[len(raw_rows) :]:
            prompt_text = str(prompt["prompt"])
            semantic_key = str(prompt["semantic_key"])
            raw_rows.append(
                failed_response_row(
                    semantic_key=semantic_key,
                    model_repository=repository,
                    prompt_text=prompt_text,
                    request_id=sha256_json(
                        {"model_repository": repository, "semantic_key": semantic_key}
                    ),
                    error_text=f"{type(exc).__name__}: {exc}",
                )
            )
    finally:
        if lease is not None and lease.document.get("phase") in {"resident", "inferencing"}:
            try:
                lease.transition("unloading")
            except Exception as exc:
                lease_error = f"{type(exc).__name__}: {exc}"
        teardown = (
            process.cleanup()
            if process is not None
            else {
                "process_exit_confirmed": True,
                "process_reaped": True,
                "port_release_confirmed": port_is_free(port),
                "leak_free": True,
                "unrelated_process_kill_count_delta": 0,
            }
        )
        owned_pid = int(process_receipt.get("pid", 0) or 0)
        after = _wait_for_gpu_release(str(device["uuid"]), owned_pid=owned_pid)
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {"model_close_called": False}
        )
        teardown_ok = bool(
            teardown.get("process_exit_confirmed") is True
            and teardown.get("process_reaped") is True
            and teardown.get("port_release_confirmed") is True
            and teardown.get("leak_free") is True
            and status.get("model_close_called") is True
            and not after["owned_processes"]
            and not after["foreign_processes"]
        )
        if lease is not None:
            try:
                phase = str(lease.document.get("phase"))
                if phase == "unloading":
                    lease.transition(
                        "validating",
                        vram_mb=0,
                        exit_code=0 if teardown_ok else 1,
                        unload_observed=not after["owned_processes"],
                    )
                    phase = "validating"
                if phase in {"preflight", "admitted", "loading", "validating"}:
                    lease.transition("terminal_complete" if teardown_ok else "terminal_blocked")
                lease_release = lease.release()
            except Exception as exc:
                lease_error = lease_error or f"{type(exc).__name__}: {exc}"
                lease.close()
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    layers = parse_offloaded_layers(log_text)
    live_cuda = bool(
        resident.get("owned_processes")
        and not resident.get("foreign_processes")
        and int(layers.get("offloaded", 0) or 0) > 0
    )
    server_row = {
        "model_repository": repository,
        "pid": process_receipt.get("pid"),
        "owned_by_task": process_receipt.get("owned_by_task") is True,
        "process_alive_at_health": health.get("ok") is True,
        "health_pid": health.get("pid"),
        "health_model_repository": health.get("model_repository"),
        "task_id": EXPERIMENT_ID,
        "health_task_id": health.get("task_id"),
        "backend": "cuda" if live_cuda else "cpu_or_unverified",
        "live_cuda": live_cuda,
        "offloaded_layers": layers.get("offloaded"),
        "total_layers": layers.get("total"),
        "gpu_uuid": device.get("uuid"),
        "cuda_device": device.get("index"),
        "n_ctx": N_CTX,
        "port": port,
        "started_monotonic_ns": started_ns,
        "health_monotonic_ns": time.monotonic_ns(),
        "command": command,
        "command_hash": process_receipt.get("command_hash"),
        "terminal": True,
    }
    request_count = int(counters.get("requests", 0) or 0)
    completion_count = int(counters.get("completions", 0) or 0)
    error_count = int(counters.get("errors", 0) or 0)
    lease_released = bool(
        lease_error is None
        and lease_release.get("released") is True
        and lease_release.get("phase") == "terminal_complete"
    )
    teardown_row = {
        "model_repository": repository,
        "process_exit_confirmed": teardown.get("process_exit_confirmed") is True,
        "port_release_confirmed": teardown.get("port_release_confirmed") is True,
        "model_close_called": status.get("model_close_called") is True,
        "lease_released": lease_released,
        "unrelated_process_kill_count_delta": teardown.get("unrelated_process_kill_count_delta", 0),
        "signals_sent": teardown.get("signals_sent", []),
        "passed": bool(
            teardown_ok
            and lease_released
            and request_count == EXPECTED_PROMPT_COUNT
            and completion_count == EXPECTED_PROMPT_COUNT
            and error_count == 0
        ),
        "terminal": True,
    }
    return {
        "model_row": {
            "model_repository": repository,
            "filename": model.get("filename"),
            "quantization": model.get("quantization"),
            "model_file_hash": model.get("model_file_hash"),
            "duration_s": time.perf_counter() - started,
            "terminal": True,
        },
        "raw_rows": raw_rows,
        "token_position_rows": token_rows,
        "gpu_identity_row": {
            "model_repository": repository,
            "gpu_uuid": device.get("uuid"),
            "cuda_device": device.get("index"),
            "live_cuda": live_cuda,
            "foreign_processes_before": before.get("foreign_processes", []),
            "foreign_processes_resident": resident.get("foreign_processes", []),
            "foreign_processes_after": after.get("foreign_processes", []),
            "owned_vram_resident_mb": sum(
                int(row.get("used_memory_mb", 0) or 0)
                for row in resident.get("owned_processes", [])
            ),
            "terminal": True,
        },
        "gpu_lease_row": {
            "model_repository": repository,
            "owner_verified": lease_owner.get("task_id") == f"{EXPERIMENT_ID}:{family_slug}",
            "device_uuid": lease_owner.get("device_uuid"),
            "released": lease_release.get("released") is True,
            "terminal_phase": lease_release.get("phase"),
            "lease_error": lease_error,
            "terminal": True,
        },
        "server_row": server_row,
        "request_counter_row": {
            "model_repository": repository,
            "expected_requests": EXPECTED_PROMPT_COUNT,
            "observed_requests": request_count,
            "terminal": True,
        },
        "completion_counter_row": {
            "model_repository": repository,
            "expected_completions": EXPECTED_PROMPT_COUNT,
            "observed_completions": completion_count,
            "error_count": error_count,
            "terminal": True,
        },
        "teardown_row": teardown_row,
        "command_hash": process_receipt.get("command_hash"),
    }


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:  # pragma: no cover - end-to-end live experiment.
    """Run preflight, score three families, freeze responses, then join labels."""

    started = time.perf_counter()
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    specs = [deepcopy(dict(row)) for row in (model_specs or MODEL_SPECS)]
    lease_runtime_dir = Path(
        os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases")
    )
    preconditions = collect_preconditions(
        model_specs=specs,
        result_path=result_path,
        checkpoint_root=checkpoint_root,
        lease_runtime_dir=lease_runtime_dir,
    )
    source_hashes = dict(preconditions.get("source_hashes", {}))
    binary_path = preconditions.get("llama_binary_path")
    llama_hash = sha256_file(binary_path) if binary_path and Path(binary_path).is_file() else None
    model_hashes = {
        str(row.get("model_repository")): sha256_file(str(row.get("model_path")))
        if Path(str(row.get("model_path"))).is_file()
        else None
        for row in specs
    }
    for row in specs:
        row["model_file_hash"] = model_hashes.get(str(row.get("model_repository")))
    if preconditions.get("all_passed") is not True:
        artifact = build_artifact(
            duration_s=time.perf_counter() - started,
            preconditions=preconditions,
            model_specs=specs,
            model_file_hashes=model_hashes,
            llama_binary_hash=llama_hash,
            command_hash=None,
            source_artifact_hashes=source_hashes,
            prompt_freeze_hash=EXPECTED_LEARNER_PROMPT_HASH,
        )
        write_json_atomic(result_path, artifact)
        return artifact

    prompt_rows = load_learner_prompts(LEARNER_PROMPT_PATH)
    device = preconditions["eligible_gpu"]
    family_results = []
    for model in specs:
        family_results.append(
            run_family(
                model=model,
                prompt_rows=prompt_rows,
                device=device,
                lease_runtime_dir=lease_runtime_dir,
                work_root=checkpoint_root / "workers",
            )
        )
    raw_rows = [row for family in family_results for row in family["raw_rows"]]
    token_rows = [row for family in family_results for row in family["token_position_rows"]]
    response_frozen_at = _utc_now()
    freeze = build_response_freeze(
        raw_rows,
        token_rows,
        prompt_freeze_hash=EXPECTED_LEARNER_PROMPT_HASH,
        frozen_at=response_frozen_at,
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(FREEZE_PATH, freeze)
    label_opened_at = _utc_now()
    sidecar_rows = load_sidecar_after_freeze(
        AUTHORITY_SIDECAR_PATH,
        freeze,
        label_opened_at=label_opened_at,
    )
    preconditions["labels_opened"] = True
    preconditions["labels_opened_after_response_freeze"] = True
    condition_rows, signed_rows = join_frozen_conditions(
        raw_rows,
        sidecar_rows,
        freeze_manifest=freeze,
        label_opened_at=label_opened_at,
    )
    family_effects, held_effects, ties = summarize_signed_responses(signed_rows)
    command_hash = sha256_json([family.get("command_hash") for family in family_results])
    artifact = build_artifact(
        duration_s=time.perf_counter() - started,
        preconditions=preconditions,
        model_specs=specs,
        model_file_hashes=model_hashes,
        llama_binary_hash=llama_hash,
        command_hash=command_hash,
        source_artifact_hashes=source_hashes,
        prompt_freeze_hash=EXPECTED_LEARNER_PROMPT_HASH,
        response_freeze_hash=str(freeze["response_freeze_hash"]),
        label_opened_at=label_opened_at,
        response_frozen_at=response_frozen_at,
        model_rows=[family["model_row"] for family in family_results],
        gpu_identity_rows=[family["gpu_identity_row"] for family in family_results],
        gpu_lease_rows=[family["gpu_lease_row"] for family in family_results],
        server_rows=[family["server_row"] for family in family_results],
        request_counter_rows=[family["request_counter_row"] for family in family_results],
        completion_counter_rows=[family["completion_counter_row"] for family in family_results],
        teardown_rows=[family["teardown_row"] for family in family_results],
        rows=raw_rows,
        per_pair_results=per_pair_results(condition_rows, signed_rows),
        condition_rows=condition_rows,
        token_position_rows=token_rows,
        signed_response_rows=signed_rows,
        family_effect_rows=family_effects,
        held_source_effect_rows=held_effects,
        tie_rows=ties,
        prohibited_feature_rows=runtime_prohibited_feature_rows(),
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command boundary.
    """Run the controller, private scoring worker, or cold validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--score-worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--model-repository")
    parser.add_argument("--port", type=int)
    parser.add_argument("--status-path", type=Path)
    args = parser.parse_args(argv)
    if args.score_worker:
        if not all(
            (
                args.model_path,
                args.model_repository,
                args.port is not None,
                args.status_path,
            )
        ):
            parser.error("worker mode requires model path, repository, port, and status path")
        return _run_score_worker(
            model_path=str(args.model_path),
            model_repository=str(args.model_repository),
            port=int(args.port),
            status_path=args.status_path,
        )
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(
        run_date=args.date,
        result_path=args.result_path,
        checkpoint_root=args.checkpoint_root,
    )
    errors = validate_artifact(artifact)
    print(
        canonical_json(
            {
                "result_path": str(args.result_path),
                "expected_pair_count": artifact["expected_pair_count"],
                "observed_pair_count": artifact["observed_pair_count"],
                "expected_family_count": artifact["expected_family_count"],
                "observed_family_count": artifact["observed_family_count"],
                "intervention_surface_complete_score": artifact[
                    "intervention_surface_complete_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
