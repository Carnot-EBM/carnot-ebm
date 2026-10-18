"""Build a label-blind teacher-forced feature bank for frozen candidates.

The scorer receives candidate identifiers and text only. A separate controller
opens exact labels after all three model workers have exited. The experiment
does not sample answers and does not fit or select a verifier.

Spec refs: REQ-INF-6986 and SCENARIO-INF-6986-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6966_gguf_load_envelope_canary import (
    build_vram_release_row,
    embedded_tokenizer_probe,
    gpu_inventory,
    llama_cpp_probe,
    parse_offloaded_layers,
)
from carnot.experiment_6973_lease_aware_gguf_runtime import (
    _choose_free_port,
    _port_is_free,
    owned_process_absent,
    terminate_owned_process,
)
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.task_runtime_receipts import (
    capture_process_lineage,
    read_process_identity,
    sha256_file,
    write_json_atomic,
)


JsonDict = dict[str, Any]
EXPERIMENT_ID = "experiment_6986_three_family_contrast_features"
SCHEMA = "carnot.experiment_6986.three_family_contrast_features.v1"
RUN_DATE = "20260904"
RANDOM_SEED = 6_986_202_609_04
PREFERRED_QUANT = "Q4_K_M"
INFERENCE_SUBSTRATE = "live_local_llama_cpp_three_family_teacher_forced_cuda"
EXPECTED_FEATURE_ROW_COUNT = 414
VRAM_RELEASE_TOLERANCE_MB = 512
MODEL_TIMEOUT_S = 14_400.0
LEASE_TTL_S = MODEL_TIMEOUT_S + 600.0
POLL_INTERVAL_S = 0.5

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6986_three_family_contrast_features.json"
CHECKPOINT_ROOT = REPO_ROOT / "results/checkpoints/experiment_6986_three_family_contrast_features"
SOURCE_PATHS = {
    "exp6975": Path("results/experiment_6975_delayed_constraint_candidate_bank.json"),
    "exp6976": Path("results/experiment_6976_exact_candidate_certification.json"),
    "exp6984": Path("results/experiment_6984_exact_contrast_fixture.json"),
    "exp6985": Path("results/experiment_6985_chronological_constraint_stream.json"),
}
EXPECTED_SOURCE_HASHES = {
    "exp6975": "sha256:4a9faf7223d174729091248f8cb763cc6b76e481abe69bdadc19adfe7dac5455",
    "exp6976": "sha256:7d174415abe6b9c3777bc56bdbf0eaa38a37aba390c64685c667f625aa6e05b9",
    "exp6984": "sha256:15f9a9bb58ca7793966f2fbac548f6879a64417b50e31b0504078cfe6ea46a3f",
    "exp6985": "sha256:461421d4771b5d997799fd2b7cb2c9e2cdfd684fbf792d25b139fcc49c30bede",
}
EXPECTED_SOURCE_COUNTS = {"exp6984": 72, "exp6985": 48, "exp6976_transfer": 18}
REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SCORING_CONFIG: JsonDict = {
    "config_id": "teacher_forced_dual_cuda_ctx16384",
    "n_ctx": 16_384,
    "n_gpu_layers": -1,
    "n_batch": 2_048,
    "n_ubatch": 512,
    "main_gpu": 0,
    "split_mode": "layer",
    "tensor_split": [0.5, 0.5],
    "visible_devices": [0, 1],
    "logits_all": True,
    "tokenizer_source": "embedded_gguf",
    "sampling_performed": False,
}
FROZEN_CONTRAST_PROMPT = (
    "Continue with the exact frozen candidate text. Do not add commentary.\nFROZEN_CANDIDATE:\n"
)

FORBIDDEN_SCORING_FIELDS = frozenset(
    {
        "exact_label",
        "exact_labels",
        "expected_label",
        "certified_relation",
        "exact_semantic_success",
        "mutation_type",
        "mutation_types",
        "fault_family",
        "source_group",
        "source_group_id",
        "split",
        "future_window",
        "future_windows",
        "held_future_window",
        "held_future_window_rows",
        "family_scores",
        "model_scores",
        "other_family_scores",
    }
)
_MANIFEST_INPUT_FIELDS = frozenset(
    {"candidate_id", "prompt_text", "candidate_text", "source_block", "source_candidate_hash"}
)
_SCORER_INPUT_FIELDS = frozenset({"candidate_id", "prompt_text", "candidate_text"})

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "live_duration_s",
    "source_artifact_hashes",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes",
    "scoring_manifest_rows",
    "scoring_manifest_hash",
    "candidate_source_count_rows",
    "rows",
    "per_candidate_model_rows",
    "raw_token_rows",
    "sequence_feature_rows",
    "parser_feature_rows",
    "label_denial_rows",
    "process_isolation_rows",
    "gpu_runtime_rows",
    "lease_rows",
    "checkpoint_rows",
    "teardown_rows",
    "vram_release_rows",
    "joined_label_rows",
    "expected_feature_row_count",
    "observed_feature_row_count",
    "three_family_feature_bank_complete_score",
    "verifier_fit_performed",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for every field makes the evidence contract auditable.",
    "preconditions_checked": "Measured gates stop unsafe or fabricated model work.",
    "inference_substrate": "The exact substrate distinguishes live local CUDA scoring from simulation.",
    "duration_s": "Total wall time exposes truncated or synthetic execution.",
    "live_duration_s": "Model time separates live scoring from setup and joining.",
    "source_artifact_hashes": "Exact hashes bind the bank to the frozen candidate sources.",
    "MODEL_SPECS": "Exact model declarations prevent silent family substitution.",
    "models_used": "Ordered family identifiers make model coverage falsifiable.",
    "model_file_hashes": "File hashes bind scores to exact cached model bytes.",
    "scoring_manifest_rows": "Frozen inputs prevent candidate or prompt drift during scoring.",
    "scoring_manifest_hash": "One digest binds prompt, candidate, order, and scoring controls.",
    "candidate_source_count_rows": "Per-source counts prove the required 72, 48, and 18 composition.",
    "rows": "Terminal summary rows expose the full candidate-family cross product.",
    "per_candidate_model_rows": "One row per candidate and family prevents pooled evidence gaps.",
    "raw_token_rows": "Per-token scalar evidence permits feature replay without full vectors.",
    "sequence_feature_rows": "Sequence summaries retain the declared likelihood features.",
    "parser_feature_rows": "Structural counts provide label-free syntax covariates.",
    "label_denial_rows": "Denied-field receipts prove labels could not enter model inputs.",
    "process_isolation_rows": "Owned child receipts prevent foreign scores from entering the bank.",
    "gpu_runtime_rows": "PID-linked device samples prove CUDA execution for each family.",
    "lease_rows": "Owner-bound lease journals prove exclusive task authority.",
    "checkpoint_rows": "Block checkpoints preserve completed evidence after interruption.",
    "teardown_rows": "Process, port, and model-close receipts prove clean family exit.",
    "vram_release_rows": "Memory recovery prevents one family from contaminating the next.",
    "joined_label_rows": "A separate late join keeps oracle data out of raw model evidence.",
    "expected_feature_row_count": "The fixed 414 count makes completeness machine-checkable.",
    "observed_feature_row_count": "The observed count reveals missing or duplicate work.",
    "three_family_feature_bank_complete_score": "Completion measures evidence integrity, not predictive value.",
    "verifier_fit_performed": "False prevents feature collection from becoming hidden verifier selection.",
    "random_seed": "A fixed seed makes model setup and ordering repeatable.",
    "reproducibility_checksum": "A content digest detects later evidence mutation.",
    "gate_check_summary": "Expected and observed values make each failure actionable.",
    "verifier_is_oracle": "False states that feature extraction does not decide correctness.",
    "verdict_class": "A closed class keeps automation states unambiguous.",
    "honest_verdict": "A class-consistent prefix records the terminal outcome.",
}


class ManifestError(ValueError):
    """A manifest cannot prove that model inputs are label blind."""


class CheckpointError(ValueError):
    """A checkpoint cannot replay the frozen manifest or unique row keys."""


class LabelJoinError(ValueError):
    """Labels cannot open because family teardown is not complete."""


def canonical_json(value: Any) -> str:
    """Return stable JSON text for hashing and process inputs."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with the project prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the project prefix."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value in canonical form."""

    return sha256_text(canonical_json(value))


def resolve_model_specs(
    *,
    cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the required cached pair first and extend it to three families."""

    pair = cached_pair_func(gpu_indices=(0, 1)) or []
    pair_paths = {str(row.get("hf_id")): str(row.get("model_path") or "") for row in pair}
    rows: list[JsonDict] = []
    for model_id in REQUIRED_MODEL_IDS:
        path = pair_paths.get(model_id) or resolver(model_id, PREFERRED_QUANT) or ""
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": str(path),
                "gpu_indices": [0, 1],
                "headline_eligible": True,
                "preferred_quant": PREFERRED_QUANT,
                "resolution_method": (
                    "cached_sota_pair(gpu_indices=(0, 1))"
                    if model_id in pair_paths
                    else "resolve_cached_gguf exact family extension"
                ),
            }
        )
    return rows


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject substitutions, missing files, and single-device placement."""

    errors: list[str] = []
    if [row.get("hf_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_ids_mismatch")
    for row in rows:
        model_id = str(row.get("hf_id", ""))
        path = str(row.get("model_path", ""))
        if not path:
            errors.append(f"model_path_missing:{model_id}")
        elif Path(path).suffix.lower() != ".gguf" or "mmproj" in Path(path).name.lower():
            errors.append(f"model_path_not_primary_gguf:{model_id}")
        if row.get("gpu_indices") != [0, 1]:
            errors.append(f"dual_gpu_indices_missing:{model_id}")
        if row.get("headline_eligible") is not True:
            errors.append(f"headline_eligibility_missing:{model_id}")
    return errors


MODEL_SPECS = resolve_model_specs()


def _nested_items(value: Any, path: str = "$") -> list[tuple[str, str, Any]]:
    """Return every nested mapping key with its readable JSON path."""

    rows: list[tuple[str, str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            rows.append((str(key), child, item))
            rows.extend(_nested_items(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_nested_items(item, f"{path}[{index}]"))
    return rows


def label_blind_input_errors(value: Any) -> list[str]:
    """Find exact denied fields at any depth before a process can score."""

    return [
        f"denied_field:{path}"
        for key, path, _item in _nested_items(value)
        if key.casefold() in FORBIDDEN_SCORING_FIELDS
    ]


def read_source_artifacts(repo_root: Path) -> dict[str, JsonDict]:
    """Read the four pinned source artifacts without changing repository state."""

    return {
        source_id: json.loads((repo_root / relative).read_text(encoding="utf-8"))
        for source_id, relative in SOURCE_PATHS.items()
    }


def _transfer_candidate_id(attempt_key: str) -> str:
    return "transfer_" + sha256_text(attempt_key).removeprefix("sha256:")[:24]


def project_frozen_candidates(sources: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Project only identifiers and text from the frozen source artifacts."""

    rows: list[JsonDict] = []
    for source_block in ("exp6984", "exp6985"):
        source = sources[source_block]
        for candidate in source.get("per_candidate_rows", []):
            text = str(candidate["serialized_candidate"])
            source_hash_field = (
                "serialization_hash" if source_block == "exp6984" else "candidate_payload_hash"
            )
            rows.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "prompt_text": FROZEN_CONTRAST_PROMPT,
                    "candidate_text": text,
                    "source_block": source_block,
                    "source_candidate_hash": str(candidate[source_hash_field]),
                }
            )

    attempts = {
        str(row.get("attempt_key")): row for row in sources["exp6975"].get("per_attempt_rows", [])
    }
    for candidate in sources["exp6976"].get("per_candidate_rows", []):
        if candidate.get("parse_success") is not True:
            continue
        attempt_key = str(candidate["attempt_key"])
        attempt = attempts.get(attempt_key)
        if attempt is None:
            raise ManifestError(f"transfer_attempt_missing:{attempt_key}")
        text = str(attempt.get("candidate_raw_text", ""))
        rows.append(
            {
                "candidate_id": _transfer_candidate_id(attempt_key),
                "prompt_text": str(attempt.get("prompt", "")),
                "candidate_text": text,
                "source_block": "exp6976_transfer",
                "source_candidate_hash": str(candidate.get("raw_sha256", "")),
            }
        )
    return rows


def candidate_source_count_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Compare each frozen source block against its exact required count."""

    counts = Counter(str(row.get("source_block", "")) for row in rows)
    return [
        {
            "source_block": source_block,
            "expected": expected,
            "observed": counts[source_block],
            "passed": counts[source_block] == expected,
        }
        for source_block, expected in EXPECTED_SOURCE_COUNTS.items()
    ]


def build_scoring_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze a label-blind manifest after strict field and hash checks."""

    errors = label_blind_input_errors(rows)
    output_rows: list[JsonDict] = []
    seen: set[str] = set()
    for ordinal, source in enumerate(rows):
        unknown = set(source) - _MANIFEST_INPUT_FIELDS
        if unknown:
            errors.append(f"unknown_manifest_fields:{sorted(unknown)}")
        candidate_id = str(source.get("candidate_id", ""))
        prompt_text = str(source.get("prompt_text", ""))
        candidate_text = str(source.get("candidate_text", ""))
        source_hash = str(source.get("source_candidate_hash", ""))
        candidate_hash = sha256_text(candidate_text)
        if not candidate_id or candidate_id in seen:
            errors.append(f"candidate_id_invalid_or_duplicate:{candidate_id}")
        seen.add(candidate_id)
        if not prompt_text or not candidate_text:
            errors.append(f"empty_prompt_or_candidate:{candidate_id}")
        if candidate_hash != source_hash:
            errors.append(f"candidate_hash_mismatch:{candidate_id}")
        output_rows.append(
            {
                "ordinal": ordinal,
                "candidate_id": candidate_id,
                "prompt_text": prompt_text,
                "prompt_hash": sha256_text(prompt_text),
                "candidate_text": candidate_text,
                "candidate_hash": candidate_hash,
                "source_block": str(source.get("source_block", "")),
            }
        )
    if errors:
        raise ManifestError(";".join(errors))
    candidate_order_hash = sha256_json([row["candidate_id"] for row in output_rows])
    scoring_config_hash = sha256_json(SCORING_CONFIG)
    manifest_payload = {
        "rows": output_rows,
        "candidate_order_hash": candidate_order_hash,
        "scoring_config": deepcopy(SCORING_CONFIG),
        "scoring_config_hash": scoring_config_hash,
    }
    return {**manifest_payload, "scoring_manifest_hash": sha256_json(manifest_payload)}


def manifest_worker(input_path: Path, output_path: Path) -> int:
    """Build a manifest in a process that rejects every denied field."""

    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
        rows = value.get("rows", []) if isinstance(value, Mapping) else value
        manifest = build_scoring_manifest(rows)
        write_json_atomic(output_path, manifest)
        return 0
    except (OSError, json.JSONDecodeError, ManifestError, TypeError) as exc:
        write_json_atomic(
            output_path,
            {"passed": False, "errors": [part for part in str(exc).split(";") if part]},
        )
        return 2


def scoring_process_blocks(manifest_rows: Sequence[Mapping[str, Any]]) -> list[list[JsonDict]]:
    """Remove controller metadata before any model worker reads the roster."""

    blocks: list[list[JsonDict]] = []
    for source_block in EXPECTED_SOURCE_COUNTS:
        block = [
            {field: row[field] for field in _SCORER_INPUT_FIELDS}
            for row in manifest_rows
            if row.get("source_block") == source_block
        ]
        blocks.append(block)
    return blocks


def runtime_label_denial_rows() -> list[JsonDict]:
    """Mutate each denied key and prove that validation rejects it."""

    rows: list[JsonDict] = []
    for field in sorted(FORBIDDEN_SCORING_FIELDS):
        probe = {"candidate_id": "probe", "prompt_text": "p", "candidate_text": "c"}
        probe["nested"] = {field: "denied"}
        errors = label_blind_input_errors(probe)
        rows.append(
            {
                "mutation_field": field,
                "denied_path": f"$.nested.{field}",
                "errors": errors,
                "passed": any(field in error for error in errors),
            }
        )
    return rows


def tokenize_teacher_forced(model: Any, row: Mapping[str, Any]) -> JsonDict:
    """Tokenize prompt and candidate separately with one embedded tokenizer."""

    prompt_tokens = list(
        model.tokenize(str(row["prompt_text"]).encode("utf-8"), add_bos=True, special=True)
    )
    candidate_tokens = list(
        model.tokenize(str(row["candidate_text"]).encode("utf-8"), add_bos=False, special=True)
    )
    full_tokens = prompt_tokens + candidate_tokens
    aligned = bool(
        prompt_tokens
        and candidate_tokens
        and full_tokens[-len(candidate_tokens) :] == candidate_tokens
    )
    return {
        "tokenizer_source": "embedded_gguf",
        "prompt_token_ids": prompt_tokens,
        "candidate_token_ids": candidate_tokens,
        "full_token_ids": full_tokens,
        "prompt_token_count": len(prompt_tokens),
        "candidate_token_count": len(candidate_tokens),
        "full_token_count": len(full_tokens),
        "alignment_passed": aligned,
        "prompt_token_hash": sha256_json(prompt_tokens),
        "candidate_token_hash": sha256_json(candidate_tokens),
        "full_token_hash": sha256_json(full_tokens),
    }


def distribution_features(
    logits: np.ndarray,
    *,
    selected_token_id: int,
    previous_surprisal: float | None,
) -> JsonDict:
    """Reduce one full distribution to replayable scalars and a vector hash."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if selected_token_id < 0 or selected_token_id >= values.size:
        raise ValueError("selected_token_id_out_of_range")
    safe = np.where(np.isfinite(values), values, -1.0e30)
    maximum = float(np.max(safe))
    weights = np.exp(safe - maximum)
    weight_sum = float(np.sum(weights))
    probabilities = weights / weight_sum
    logsumexp = maximum + math.log(weight_sum)
    selected_logit = float(safe[selected_token_id])
    selected_log_probability = selected_logit - logsumexp
    surprisal = -selected_log_probability
    top_indices = np.argpartition(safe, -2)[-2:] if safe.size >= 2 else np.asarray([0, 0])
    top_values = np.sort(safe[top_indices])[::-1]
    top_probability = math.exp(float(top_values[0]) - logsumexp)
    second_probability = math.exp(float(top_values[1]) - logsumexp)
    entropy = float(logsumexp - np.sum(probabilities * safe))
    return {
        "selected_token_logit": selected_logit,
        "full_vocabulary_logsumexp": logsumexp,
        "selected_token_log_probability": selected_log_probability,
        "surprisal": surprisal,
        "token_entropy": entropy,
        "top_probability": top_probability,
        "top_probability_margin": top_probability - second_probability,
        "local_surprisal_change": (
            0.0 if previous_surprisal is None else surprisal - previous_surprisal
        ),
        "vocabulary_size": int(values.size),
        "full_logit_vector_hash": sha256_bytes(
            np.asarray(logits, dtype=np.float32).reshape(-1).tobytes(order="C")
        ),
    }


def _json_structure_counts(value: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if isinstance(value, Mapping):
        counts["object_count"] += 1
        counts["key_count"] += len(value)
        for item in value.values():
            counts.update(_json_structure_counts(item))
    elif isinstance(value, list):
        counts["array_count"] += 1
        counts["array_item_count"] += len(value)
        for item in value:
            counts.update(_json_structure_counts(item))
    elif value is None:
        counts["null_count"] += 1
    elif isinstance(value, bool):
        counts["boolean_count"] += 1
    elif isinstance(value, (int, float)):
        counts["number_count"] += 1
    elif isinstance(value, str):
        counts["string_count"] += 1
    return counts


def parser_structure_features(candidate_id: str, candidate_text: str) -> JsonDict:
    """Count syntax structure without deciding whether a candidate is correct."""

    fields = (
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
        "candidate_id": candidate_id,
        "json_parseable": parseable,
        "byte_count": len(candidate_text.encode("utf-8")),
        "line_count": candidate_text.count("\n") + 1,
        "brace_count": candidate_text.count("{") + candidate_text.count("}"),
        "bracket_count": candidate_text.count("[") + candidate_text.count("]"),
        **{field: int(counts[field]) for field in fields},
    }


def score_candidate(model: Any, row: Mapping[str, Any], *, model_id: str) -> JsonDict:
    """Teacher-force one sequence and retain each selected-token distribution."""

    tokenized = tokenize_teacher_forced(model, row)
    if tokenized["alignment_passed"] is not True:
        raise ValueError("tokenizer_alignment_failed")
    if tokenized["full_token_count"] > int(SCORING_CONFIG["n_ctx"]):
        raise ValueError("sequence_exceeds_context")
    model.reset()
    model.eval(tokenized["full_token_ids"])
    scores = np.asarray(model.scores)
    candidate_ids = tokenized["candidate_token_ids"]
    start = int(tokenized["prompt_token_count"]) - 1
    if start < 0 or start + len(candidate_ids) > scores.shape[0]:
        raise ValueError("logit_alignment_unavailable")

    raw_rows: list[JsonDict] = []
    previous_surprisal: float | None = None
    for candidate_position, token_id in enumerate(candidate_ids):
        sequence_position = start + candidate_position + 1
        features = distribution_features(
            scores[start + candidate_position],
            selected_token_id=int(token_id),
            previous_surprisal=previous_surprisal,
        )
        previous_surprisal = float(features["surprisal"])
        token_bytes = bytes(model.detokenize([int(token_id)]))
        raw_rows.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "candidate_hash": sha256_text(str(row["candidate_text"])),
                "model_id": model_id,
                "sequence_position": sequence_position,
                "candidate_token_position": candidate_position,
                "selected_token_id": int(token_id),
                "selected_token_bytes_hex": token_bytes.hex(),
                "selected_token_text": token_bytes.decode("utf-8", errors="replace"),
                **features,
            }
        )

    surprisals = [float(item["surprisal"]) for item in raw_rows]
    entropies = [float(item["token_entropy"]) for item in raw_rows]
    margins = [float(item["top_probability_margin"]) for item in raw_rows]
    changes = [float(item["local_surprisal_change"]) for item in raw_rows]
    candidate_hash = sha256_text(str(row["candidate_text"]))
    sequence_row = {
        "candidate_id": str(row["candidate_id"]),
        "candidate_hash": candidate_hash,
        "prompt_hash": sha256_text(str(row["prompt_text"])),
        "model_id": model_id,
        "terminal": True,
        "teacher_forced": True,
        "sampling_performed": False,
        "tokenizer_source": "embedded_gguf",
        "tokenizer_alignment": tokenized["alignment_passed"],
        "prompt_token_count": tokenized["prompt_token_count"],
        "candidate_token_count": tokenized["candidate_token_count"],
        "full_token_count": tokenized["full_token_count"],
        "prompt_token_hash": tokenized["prompt_token_hash"],
        "candidate_token_hash": tokenized["candidate_token_hash"],
        "full_token_hash": tokenized["full_token_hash"],
        "sequence_nll": float(sum(surprisals)),
        "length_normalized_nll": float(sum(surprisals) / len(surprisals)),
        "mean_token_entropy": float(sum(entropies) / len(entropies)),
        "mean_top_probability_margin": float(sum(margins) / len(margins)),
        "mean_local_surprisal_change": float(sum(changes) / len(changes)),
        "max_abs_local_surprisal_change": float(max(abs(value) for value in changes)),
        "raw_token_count": len(raw_rows),
        "raw_token_hash": sha256_json(raw_rows),
    }
    parser_row = {
        "model_id": model_id,
        **parser_structure_features(str(row["candidate_id"]), str(row["candidate_text"])),
    }
    return {
        "sequence_row": sequence_row,
        "raw_token_rows": raw_rows,
        "parser_row": parser_row,
    }


def completed_feature_keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """Return unique candidate-family keys or fail on a duplicate."""

    keys = [(str(row.get("candidate_id")), str(row.get("model_id"))) for row in rows]
    if len(keys) != len(set(keys)):
        raise CheckpointError("duplicate_feature_key")
    return set(keys)


def write_checkpoint(
    path: Path,
    *,
    manifest_hash: str,
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write one atomic checkpoint after duplicate-key validation."""

    completed_feature_keys(rows)
    payload = {
        "schema": "carnot.exp6986.checkpoint.v1",
        "manifest_hash": manifest_hash,
        "rows": [deepcopy(dict(row)) for row in rows],
    }
    write_json_atomic(path, payload)
    return {
        "path": str(path),
        "manifest_hash": manifest_hash,
        "row_count": len(rows),
        "checkpoint_hash": sha256_json(payload),
        "passed": True,
    }


def load_checkpoint(path: Path, *, manifest_hash: str) -> JsonDict:
    """Load one checkpoint only when its manifest and row keys replay."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifest_hash") != manifest_hash:
        raise CheckpointError("manifest_hash_mismatch")
    completed_feature_keys(value.get("rows", []))
    return value


def _teardown_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        len(rows) == len(REQUIRED_MODEL_IDS)
        and {str(row.get("model_id")) for row in rows} == set(REQUIRED_MODEL_IDS)
        and all(
            row.get("passed") is True
            and row.get("process_exit_code") == 0
            and row.get("owned_process_absent") is True
            and row.get("port_release_confirmed") is True
            and row.get("model_close_called") is True
            and row.get("signals_sent") == []
            for row in rows
        )
    )


def feature_bank_completion_errors(
    manifest_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    teardown_rows: Sequence[Mapping[str, Any]],
    vram_release_rows: Sequence[Mapping[str, Any]],
    *,
    label_denial_passed: bool,
    expected_candidate_count: int = 138,
) -> list[str]:
    """Reduce only integrity gates; predictive value is deliberately absent."""

    errors: list[str] = []
    expected_rows = expected_candidate_count * len(REQUIRED_MODEL_IDS)
    if len(manifest_rows) != expected_candidate_count:
        errors.append("manifest_candidate_count")
    if len(feature_rows) != expected_rows:
        errors.append("feature_row_count")
    try:
        keys = completed_feature_keys(feature_rows)
    except CheckpointError:
        keys = set()
        errors.append("duplicate_feature_key")
    expected_keys = {
        (str(candidate.get("candidate_id")), family)
        for candidate in manifest_rows
        for family in REQUIRED_MODEL_IDS
    }
    if keys != expected_keys:
        errors.append("family_completeness")
    candidate_hashes = {
        str(row.get("candidate_id")): str(row.get("candidate_hash")) for row in manifest_rows
    }
    if any(
        row.get("terminal") is not True
        or candidate_hashes.get(str(row.get("candidate_id"))) != row.get("candidate_hash")
        for row in feature_rows
    ):
        errors.append("candidate_hash_or_terminal_mismatch")
    if any(
        row.get("live_cuda") is not True or row.get("tokenizer_alignment") is not True
        for row in feature_rows
    ):
        errors.append("cuda_incomplete")
    if not label_denial_passed:
        errors.append("label_denial_failed")
    if not _teardown_complete(teardown_rows):
        errors.append("teardown_incomplete")
    if (
        len(vram_release_rows) != len(REQUIRED_MODEL_IDS)
        or {str(row.get("model_id")) for row in vram_release_rows} != set(REQUIRED_MODEL_IDS)
        or any(row.get("passed") is not True for row in vram_release_rows)
    ):
        errors.append("vram_release_incomplete")
    return list(dict.fromkeys(errors))


def join_labels(
    feature_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    teardown_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join labels by identity only after all family workers have exited."""

    if not _teardown_complete(teardown_rows):
        raise LabelJoinError("family_processes_not_exited")
    joined: list[JsonDict] = []
    for row in feature_rows:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in labels:
            raise LabelJoinError(f"label_missing:{candidate_id}")
        joined.append(
            {
                "candidate_id": candidate_id,
                "candidate_hash": row.get("candidate_hash"),
                "model_id": row.get("model_id"),
                **deepcopy(dict(labels[candidate_id])),
            }
        )
    return joined


def gate_check(
    check: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Record one check with explicit expected and observed values."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and promote the first failed expected-observed pair."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    live_duration_s: float,
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    model_file_hashes: Mapping[str, Any],
    scoring_manifest_rows: Sequence[Mapping[str, Any]] = (),
    scoring_manifest_hash: str | None = None,
    candidate_source_count_rows: Sequence[Mapping[str, Any]] = (),
    per_candidate_model_rows: Sequence[Mapping[str, Any]] = (),
    raw_token_rows: Sequence[Mapping[str, Any]] = (),
    sequence_feature_rows: Sequence[Mapping[str, Any]] = (),
    parser_feature_rows: Sequence[Mapping[str, Any]] = (),
    label_denial_rows: Sequence[Mapping[str, Any]] = (),
    process_isolation_rows: Sequence[Mapping[str, Any]] = (),
    gpu_runtime_rows: Sequence[Mapping[str, Any]] = (),
    lease_rows: Sequence[Mapping[str, Any]] = (),
    checkpoint_rows: Sequence[Mapping[str, Any]] = (),
    teardown_rows: Sequence[Mapping[str, Any]] = (),
    vram_release_rows: Sequence[Mapping[str, Any]] = (),
    joined_label_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the complete positive, partial, or blocked artifact schema."""

    manifest = [deepcopy(dict(row)) for row in scoring_manifest_rows]
    features = [deepcopy(dict(row)) for row in per_candidate_model_rows]
    sequence_rows = [deepcopy(dict(row)) for row in sequence_feature_rows]
    raw_rows = [deepcopy(dict(row)) for row in raw_token_rows]
    parser_rows = [deepcopy(dict(row)) for row in parser_feature_rows]
    denial_rows = [deepcopy(dict(row)) for row in label_denial_rows]
    isolation_rows = [deepcopy(dict(row)) for row in process_isolation_rows]
    teardown = [deepcopy(dict(row)) for row in teardown_rows]
    vram = [deepcopy(dict(row)) for row in vram_release_rows]
    checks = [deepcopy(dict(row)) for row in preconditions.get("checks", [])]
    all_preconditions = preconditions.get("all_passed") is True
    label_denial_passed = bool(
        denial_rows and all(row.get("passed") is True for row in denial_rows)
    )
    completion_errors = feature_bank_completion_errors(
        manifest,
        features,
        teardown,
        vram,
        label_denial_passed=label_denial_passed,
    )
    if len(sequence_rows) != len(features):
        completion_errors.append("sequence_feature_count")
    if len(parser_rows) != len(features):
        completion_errors.append("parser_feature_count")
    if sum(int(row.get("raw_token_count", 0) or 0) for row in features) != len(raw_rows):
        completion_errors.append("raw_token_count")
    if len(joined_label_rows) != len(features):
        completion_errors.append("joined_label_count")
    if (
        len(isolation_rows) != len(REQUIRED_MODEL_IDS)
        or {str(row.get("model_id")) for row in isolation_rows} != set(REQUIRED_MODEL_IDS)
        or any(row.get("passed") is not True for row in isolation_rows)
    ):
        completion_errors.append("process_isolation_incomplete")
    completion_errors = list(dict.fromkeys(completion_errors))
    complete = bool(all_preconditions and not completion_errors)
    if all_preconditions:
        checks.append(gate_check("feature_bank_completion", [], completion_errors))
    if not all_preconditions:
        verdict_class = "blocked"
        honest_verdict = "blocked_three_family_contrast_features"
    elif complete:
        verdict_class = "positive"
        honest_verdict = "complete_three_family_contrast_feature_bank"
    else:
        verdict_class = "partial"
        honest_verdict = "partial_three_family_contrast_features"

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "live_duration_s": float(live_duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
        "models_used": [str(row.get("hf_id")) for row in model_specs if row.get("hf_id")],
        "model_file_hashes": deepcopy(dict(model_file_hashes)),
        "scoring_manifest_rows": manifest,
        "scoring_manifest_hash": scoring_manifest_hash,
        "candidate_source_count_rows": [deepcopy(dict(row)) for row in candidate_source_count_rows],
        "rows": deepcopy(features),
        "per_candidate_model_rows": features,
        "raw_token_rows": raw_rows,
        "sequence_feature_rows": sequence_rows,
        "parser_feature_rows": parser_rows,
        "label_denial_rows": denial_rows,
        "process_isolation_rows": isolation_rows,
        "gpu_runtime_rows": [deepcopy(dict(row)) for row in gpu_runtime_rows],
        "lease_rows": [deepcopy(dict(row)) for row in lease_rows],
        "checkpoint_rows": [deepcopy(dict(row)) for row in checkpoint_rows],
        "teardown_rows": teardown,
        "vram_release_rows": vram,
        "joined_label_rows": [deepcopy(dict(row)) for row in joined_label_rows],
        "expected_feature_row_count": EXPECTED_FEATURE_ROW_COUNT,
        "observed_feature_row_count": len(features),
        "three_family_feature_bank_complete_score": int(complete),
        "verifier_fit_performed": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _contains_full_vectors(value: Any) -> bool:
    forbidden = {"logits", "full_vocabulary_logits", "full_logit_vector"}
    return any(key.casefold() in forbidden for key, _path, _item in _nested_items(value))


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, bare gates, and bounded raw evidence."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        errors.append(f"required_fields_missing:{sorted(missing)}")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if type(artifact.get("expected_feature_row_count")) is not int:
        errors.append("expected_feature_row_count_not_bare_int")
    if type(artifact.get("observed_feature_row_count")) is not int:
        errors.append("observed_feature_row_count_not_bare_int")
    if type(artifact.get("three_family_feature_bank_complete_score")) is not int:
        errors.append("completion_score_not_bare_int")
    if type(artifact.get("verifier_fit_performed")) is not bool:
        errors.append("verifier_fit_not_bare_bool")
    if type(artifact.get("verifier_is_oracle")) is not bool:
        errors.append("verifier_oracle_not_bare_bool")
    if artifact.get("expected_feature_row_count") != EXPECTED_FEATURE_ROW_COUNT:
        errors.append("expected_feature_row_count_mismatch")
    if label_blind_input_errors(artifact.get("scoring_manifest_rows", [])):
        errors.append("manifest_contains_denied_fields")
    if _contains_full_vectors(artifact.get("raw_token_rows", [])):
        errors.append("full_vocabulary_vector_present")
    score = artifact.get("three_family_feature_bank_complete_score")
    if score == 1:
        if artifact.get("observed_feature_row_count") != EXPECTED_FEATURE_ROW_COUNT:
            errors.append("positive_row_count_mismatch")
        if artifact.get("verdict_class") != "positive":
            errors.append("positive_verdict_class_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
            errors.append("positive_verdict_prefix_mismatch")
    elif artifact.get("verdict_class") == "blocked":
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_prefix_mismatch")
    elif artifact.get("verdict_class") == "partial" and not str(
        artifact.get("honest_verdict", "")
    ).startswith("partial_"):
        errors.append("partial_verdict_prefix_mismatch")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _storage_writable(path: Path) -> bool:  # pragma: no cover - host filesystem boundary.
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".write-probe-", dir=path)
        os.close(descriptor)
        Path(temporary).unlink()
        return True
    except OSError:
        return False


def _source_hashes(repo_root: Path) -> dict[str, str | None]:  # pragma: no cover
    return {
        source_id: sha256_file(repo_root / relative) for source_id, relative in SOURCE_PATHS.items()
    }


def _lease_preflight(
    devices: Sequence[Mapping[str, Any]], runtime_dir: Path
) -> JsonDict:  # pragma: no cover - live kernel-lock boundary.
    rows: list[JsonDict] = []
    for device in devices:
        uuid = str(device.get("uuid", ""))
        path = lease_api.journal_path_for(runtime_dir, uuid)
        if not path.exists():
            rows.append({"device_uuid": uuid, "classification": "absent", "free": True})
            continue
        try:
            document = lease_api.read_journal(path)
            owner = document.get("owner", {})
            live = bool(
                document.get("released") is not True
                and isinstance(owner, Mapping)
                and isinstance(owner.get("pid"), int)
                and isinstance(owner.get("pid_start_ticks"), int)
                and lease_api.process_start_matches(owner["pid"], owner["pid_start_ticks"])
            )
            rows.append(
                {
                    "device_uuid": uuid,
                    "classification": "live_foreign" if live else "released_or_stale",
                    "free": not live,
                    "journal_hash": sha256_json(document),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "device_uuid": uuid,
                    "classification": "unreadable",
                    "free": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"rows": rows, "free": len(rows) == 2 and all(row["free"] for row in rows)}


def collect_preconditions(
    *,
    repo_root: Path,
    model_specs: Sequence[Mapping[str, Any]],
    checkpoint_root: Path,
    lease_runtime_dir: Path,
) -> JsonDict:  # pragma: no cover - live host preflight.
    sources = read_source_artifacts(repo_root)
    projected = project_frozen_candidates(sources)
    counts = candidate_source_count_rows(projected)
    hashes = _source_hashes(repo_root)
    gpu = gpu_inventory()
    devices = list(gpu.get("devices", []))
    binding = llama_cpp_probe()
    lease = _lease_preflight(devices, lease_runtime_dir)
    tokenizers = [embedded_tokenizer_probe(row) for row in model_specs]
    specs_errors = model_spec_errors(model_specs)
    checks = [
        gate_check(
            "exp6984_contrast_fixture_complete_score",
            1,
            sources["exp6984"].get("contrast_fixture_complete_score"),
        ),
        gate_check(
            "exp6985_chronological_stream_ready_score",
            1,
            sources["exp6985"].get("chronological_stream_ready_score"),
        ),
        gate_check("exact_source_hashes", EXPECTED_SOURCE_HASHES, hashes),
        gate_check(
            "frozen_candidate_counts",
            EXPECTED_SOURCE_COUNTS,
            {row["source_block"]: row["observed"] for row in counts},
        ),
        gate_check("exact_model_specs", [], specs_errors),
        gate_check(
            "all_three_gguf_files",
            {family: True for family in REQUIRED_MODEL_IDS},
            {
                str(row.get("hf_id")): Path(str(row.get("model_path", ""))).is_file()
                for row in model_specs
            },
        ),
        gate_check(
            "two_cuda_devices",
            2,
            len(devices),
            passed=gpu.get("query_ok") is True and len(devices) == 2,
        ),
        gate_check(
            "cuda_capable_llama_cpp",
            {"importable": True, "gpu_offload": True},
            {"importable": binding.get("importable"), "gpu_offload": binding.get("gpu_offload")},
        ),
        gate_check("free_task_lease", True, lease.get("free")),
        gate_check("writable_checkpoints", True, _storage_writable(checkpoint_root)),
        gate_check(
            "embedded_tokenizers",
            {family: True for family in REQUIRED_MODEL_IDS},
            {str(row.get("model_id")): row.get("passed") for row in tokenizers},
        ),
    ]
    return {
        "all_passed": all(row["passed"] is True for row in checks),
        "checks": checks,
        "gpu_topology": gpu,
        "baseline_gpu_memory_rows": _memory_rows(gpu),
        "llama_cpp": binding,
        "lease_preflight": lease,
        "embedded_tokenizer_rows": tokenizers,
        "candidate_source_count_rows": counts,
    }


def _memory_rows(gpu: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "index": int(row.get("index", -1)),
            "uuid": str(row.get("uuid", "")),
            "memory_used_mb": int(row.get("memory_used_mb", 0) or 0),
            "memory_free_mb": int(row.get("memory_free_mb", 0) or 0),
        }
        for row in gpu.get("devices", [])
    ]


def _wait_for_vram_release(
    baseline_rows: Sequence[Mapping[str, Any]], model_id: str
) -> JsonDict:  # pragma: no cover - live CUDA boundary.
    deadline = time.monotonic() + 180.0
    latest = gpu_inventory()
    while True:
        after_rows = _memory_rows(latest)
        row = build_vram_release_row(
            model_id=model_id,
            baseline_rows=baseline_rows,
            after_rows=after_rows,
        )
        row["after_rows"] = after_rows
        row["max_residual_mb"] = max(0, int(row.get("max_increase_mb", 0) or 0))
        if row.get("passed") is True or time.monotonic() >= deadline:
            return row
        time.sleep(1.0)
        latest = gpu_inventory()


def _acquire_leases(
    model: Mapping[str, Any],
    devices: Sequence[Mapping[str, Any]],
    runtime_dir: Path,
) -> tuple[list[Any], list[JsonDict]]:  # pragma: no cover - live kernel-lock boundary.
    leases: list[Any] = []
    rows: list[JsonDict] = []
    try:
        for device in sorted(devices, key=lambda item: int(item.get("index", 0))):
            lease = lease_api.GpuLease.acquire(
                runtime_dir=runtime_dir,
                task_id=EXPERIMENT_ID,
                device_uuid=str(device["uuid"]),
                expected_model=str(model["model_path"]),
                vram_before_mb=int(device.get("memory_used_mb", 0) or 0),
                ttl_s=LEASE_TTL_S,
            )
            lease.transition("admitted")
            lease.transition("loading")
            owner = lease.owner_receipt()
            leases.append(lease)
            rows.append(
                {
                    "model_id": model["hf_id"],
                    "device_uuid": device["uuid"],
                    "task_id": EXPERIMENT_ID,
                    "owner_pid": owner.get("pid"),
                    "owner_pid_start_ticks": owner.get("pid_start_ticks"),
                    "owner_verified": owner.get("task_id") == EXPERIMENT_ID,
                    "journal_after_acquisition": deepcopy(lease.document),
                    "journal_after_release": None,
                    "release_receipt": {},
                    "signals_sent": [],
                    "consistent": False,
                }
            )
    except Exception:
        for lease in leases:
            lease.close()
        raise
    return leases, rows


def _release_leases(
    leases: Sequence[Any],
    rows: list[JsonDict],
    *,
    complete: bool,
    vram_release: Mapping[str, Any],
    exit_code: int,
) -> None:  # pragma: no cover - live kernel-lock boundary.
    after_by_uuid = {
        str(row.get("uuid")): int(row.get("memory_used_mb", 0) or 0)
        for row in vram_release.get("after_rows", [])
    }
    for lease, row in zip(leases, rows, strict=True):
        try:
            phase = str(lease.document.get("phase"))
            if phase in {"resident", "inferencing"}:
                lease.transition("unloading")
                phase = "unloading"
            if phase == "unloading" and vram_release.get("passed") is True:
                lease.transition(
                    "validating",
                    vram_mb=after_by_uuid.get(lease.device_uuid, 0),
                    exit_code=exit_code,
                    unload_observed=True,
                )
                lease.transition("terminal_complete" if complete else "terminal_blocked")
            elif phase in {"preflight", "admitted", "loading"}:
                lease.transition("terminal_blocked")
            row["release_receipt"] = lease.release()
            row["journal_after_release"] = deepcopy(lease.document)
            errors = lease_api.validate_journal_document(lease.document, check_freshness=False)
            row["consistent"] = bool(
                not errors
                and lease.document.get("released") is True
                and lease.document.get("phase") in lease_api.TERMINAL_PHASES
                and row["release_receipt"].get("signals_sent") == []
            )
            row["journal_validation_errors"] = errors
        except lease_api.LeaseError as exc:
            lease.close()
            row["journal_validation_errors"] = [f"{type(exc).__name__}: {exc}"]


def _score_worker(
    *,
    payload_path: Path,
    output_path: Path,
    ready_path: Path,
    port: int,
) -> int:  # pragma: no cover - live llama.cpp worker.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    model: Any = None
    close_called = False
    started = time.perf_counter()
    try:
        listener.bind(("127.0.0.1", int(port)))
        listener.listen(1)
        write_json_atomic(
            ready_path,
            {
                "pid": os.getpid(),
                "pid_start_ticks": lease_api.proc_start_ticks(os.getpid()),
                "port": port,
            },
        )
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        denied = label_blind_input_errors(payload.get("blocks", []))
        for block in payload.get("blocks", []):
            for row in block:
                unknown = set(row) - _SCORER_INPUT_FIELDS
                if unknown:
                    denied.append(f"unknown_scorer_fields:{sorted(unknown)}")
        if denied:
            raise ManifestError(";".join(denied))

        from llama_cpp import Llama

        model = Llama(
            model_path=str(payload["model_path"]),
            n_ctx=int(SCORING_CONFIG["n_ctx"]),
            n_gpu_layers=int(SCORING_CONFIG["n_gpu_layers"]),
            n_batch=int(SCORING_CONFIG["n_batch"]),
            n_ubatch=int(SCORING_CONFIG["n_ubatch"]),
            main_gpu=int(SCORING_CONFIG["main_gpu"]),
            split_mode=1,
            tensor_split=list(SCORING_CONFIG["tensor_split"]),
            logits_all=True,
            seed=RANDOM_SEED,
            use_mmap=True,
            use_mlock=False,
            verbose=True,
        )
        sequence_rows: list[JsonDict] = []
        raw_rows: list[JsonDict] = []
        parser_rows: list[JsonDict] = []
        checkpoint_rows: list[JsonDict] = []
        worker_root = Path(payload["worker_root"])
        worker_root.mkdir(parents=True, exist_ok=True)
        for block_index, block in enumerate(payload["blocks"]):
            block_path = worker_root / f"block_{block_index:02d}.json"
            if block_path.exists():
                stored = json.loads(block_path.read_text(encoding="utf-8"))
                if stored.get("manifest_hash") != payload["manifest_hash"]:
                    raise CheckpointError("manifest_hash_mismatch")
                block_result = stored
                recovered = True
            else:
                block_sequences: list[JsonDict] = []
                block_raw: list[JsonDict] = []
                block_parser: list[JsonDict] = []
                for row in block:
                    scored = score_candidate(model, row, model_id=str(payload["model_id"]))
                    block_sequences.append(scored["sequence_row"])
                    block_raw.extend(scored["raw_token_rows"])
                    block_parser.append(scored["parser_row"])
                block_result = {
                    "manifest_hash": payload["manifest_hash"],
                    "block_index": block_index,
                    "sequence_feature_rows": block_sequences,
                    "raw_token_rows": block_raw,
                    "parser_feature_rows": block_parser,
                }
                write_json_atomic(block_path, block_result)
                recovered = False
            sequence_rows.extend(block_result["sequence_feature_rows"])
            raw_rows.extend(block_result["raw_token_rows"])
            parser_rows.extend(block_result["parser_feature_rows"])
            checkpoint_rows.append(
                {
                    "model_id": payload["model_id"],
                    "block_index": block_index,
                    "path": str(block_path),
                    "manifest_hash": payload["manifest_hash"],
                    "checkpoint_hash": sha256_json(block_result),
                    "recovered": recovered,
                    "row_count": len(block_result["sequence_feature_rows"]),
                    "passed": True,
                }
            )
        model.close()
        close_called = True
        model = None
        gc.collect()
        output = {
            "terminal": True,
            "model_id": payload["model_id"],
            "model_close_called": close_called,
            "live_duration_s": time.perf_counter() - started,
            "sequence_feature_rows": sequence_rows,
            "raw_token_rows": raw_rows,
            "parser_feature_rows": parser_rows,
            "checkpoint_rows": checkpoint_rows,
            "scorer_input_field_names": sorted(_SCORER_INPUT_FIELDS),
            "denied_fields_visible": [],
            "sampling_performed": False,
        }
        write_json_atomic(output_path, output)
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json_atomic(
            output_path,
            {
                "terminal": True,
                "model_close_called": close_called,
                "live_duration_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "sequence_feature_rows": [],
                "raw_token_rows": [],
                "parser_feature_rows": [],
                "checkpoint_rows": [],
            },
        )
        return 1
    finally:
        if model is not None:
            try:
                model.close()
                close_called = True
            except Exception:  # noqa: BLE001
                pass
        listener.close()


def run_family_process(
    *,
    model: Mapping[str, Any],
    blocks: Sequence[Sequence[Mapping[str, Any]]],
    manifest_hash: str,
    work_root: Path,
    devices: Sequence[Mapping[str, Any]],
    lease_runtime_dir: Path,
) -> JsonDict:  # pragma: no cover - live subprocess and CUDA boundary.
    model_id = str(model["hf_id"])
    baseline = gpu_inventory()
    baseline_rows = _memory_rows(baseline)
    leases, lease_rows = _acquire_leases(model, devices, lease_runtime_dir)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()
    family_root = work_root / slug
    family_root.mkdir(parents=True, exist_ok=True)
    payload_path = family_root / "payload.json"
    output_path = family_root / "worker_output.json"
    ready_path = family_root / "ready.json"
    stdout_path = family_root / "stdout.log"
    stderr_path = family_root / "stderr.log"
    port = _choose_free_port()
    payload = {
        "model_id": model_id,
        "model_path": model["model_path"],
        "manifest_hash": manifest_hash,
        "blocks": [[deepcopy(dict(row)) for row in block] for block in blocks],
        "worker_root": str(family_root / "blocks"),
    }
    write_json_atomic(payload_path, payload)
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6986_three_family_contrast_features",
        "--score-worker-payload",
        str(payload_path),
        "--score-worker-output",
        str(output_path),
        "--score-worker-ready",
        str(ready_path),
        "--score-worker-port",
        str(port),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    samples: list[JsonDict] = []
    task_identity = read_process_identity(os.getpid()) or {}
    ownership: JsonDict = {"owned": False, "signals_sent": []}
    resident = False
    timed_out = False
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        start_ticks = lease_api.proc_start_ticks(process.pid)
        deadline = time.monotonic() + MODEL_TIMEOUT_S
        while process.poll() is None and time.monotonic() < deadline:
            sample = gpu_inventory()
            sample["monotonic_ns"] = time.monotonic_ns()
            samples.append(sample)
            if ready_path.exists():
                lineage = capture_process_lineage(process.pid, task_identity)
                ownership = {
                    **lineage,
                    "owned": lineage.get("owned") is True,
                    "signals_sent": ownership.get("signals_sent", []),
                }
            owned_uuids = {
                str(row.get("gpu_uuid"))
                for row in sample.get("processes", [])
                if row.get("pid") == process.pid
            }
            if not resident and owned_uuids == {str(row["uuid"]) for row in devices}:
                process_vram_by_uuid = {
                    str(device["uuid"]): sum(
                        int(row.get("used_memory_mb", 0) or 0)
                        for row in sample.get("processes", [])
                        if row.get("pid") == process.pid
                        and str(row.get("gpu_uuid")) == str(device["uuid"])
                    )
                    for device in devices
                }
                for lease in leases:
                    lease.transition(
                        "resident", vram_mb=process_vram_by_uuid.get(lease.device_uuid, 0)
                    )
                    lease.transition("inferencing")
                resident = True
            time.sleep(POLL_INTERVAL_S)
        if process.poll() is None:
            timed_out = True
            if terminate_owned_process(process.pid, os.getpid(), start_ticks):
                ownership["signals_sent"].append("SIGTERM")
        process.wait(timeout=60)

    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    layer_row = parse_offloaded_layers(stderr_text)
    worker = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {
            "terminal": True,
            "model_close_called": False,
            "live_duration_s": 0.0,
            "error": "worker_output_missing",
            "sequence_feature_rows": [],
            "raw_token_rows": [],
            "parser_feature_rows": [],
            "checkpoint_rows": [],
        }
    )
    release = _wait_for_vram_release(baseline_rows, model_id)
    absent = owned_process_absent(process.pid, start_ticks)
    port_released = _port_is_free(port)
    owned_samples = [
        {"monotonic_ns": sample["monotonic_ns"], **row}
        for sample in samples
        for row in sample.get("processes", [])
        if row.get("pid") == process.pid
    ]
    gpu_uuids = sorted({str(row.get("gpu_uuid")) for row in owned_samples})
    live_cuda = bool(layer_row.get("offloaded", 0) > 0 and len(gpu_uuids) == 2)
    complete = bool(
        process.returncode == 0
        and not timed_out
        and worker.get("model_close_called") is True
        and ownership.get("owned") is True
        and absent
        and port_released
        and release.get("passed") is True
        and live_cuda
    )
    _release_leases(
        leases,
        lease_rows,
        complete=complete,
        vram_release=release,
        exit_code=int(process.returncode or 0),
    )
    lease_complete = len(lease_rows) == 2 and all(
        row.get("consistent") is True for row in lease_rows
    )
    for row in worker.get("sequence_feature_rows", []):
        row["live_cuda"] = live_cuda
    teardown = {
        "model_id": model_id,
        "process_exit_code": process.returncode,
        "owned_process_absent": absent,
        "port": port,
        "port_release_confirmed": port_released,
        "model_close_called": worker.get("model_close_called"),
        "signals_sent": ownership.get("signals_sent", []),
        "passed": bool(complete and lease_complete and ownership.get("signals_sent") == []),
    }
    isolation = {
        "model_id": model_id,
        "process_owned": ownership.get("owned") is True,
        "scorer_input_field_names": worker.get("scorer_input_field_names", []),
        "denied_fields_visible": worker.get("denied_fields_visible", []),
        "another_family_scores_visible": False,
        "sampling_performed": worker.get("sampling_performed"),
        "input_payload_hash": sha256_json(payload["blocks"]),
        "process_receipt": ownership,
        "passed": bool(
            complete
            and worker.get("scorer_input_field_names") == sorted(_SCORER_INPUT_FIELDS)
            and worker.get("denied_fields_visible") == []
            and worker.get("sampling_performed") is False
        ),
    }
    return {
        "model_id": model_id,
        "complete": bool(complete and lease_complete),
        "live_duration_s": float(worker.get("live_duration_s", 0.0) or 0.0),
        "sequence_feature_rows": worker.get("sequence_feature_rows", []),
        "raw_token_rows": worker.get("raw_token_rows", []),
        "parser_feature_rows": worker.get("parser_feature_rows", []),
        "checkpoint_rows": worker.get("checkpoint_rows", []),
        "lease_rows": lease_rows,
        "process_isolation": isolation,
        "gpu_runtime": {
            "model_id": model_id,
            "live_cuda": live_cuda,
            "gpu_uuids_used": gpu_uuids,
            "offloaded_layers": layer_row.get("offloaded"),
            "total_layers": layer_row.get("total"),
            "owned_gpu_samples": owned_samples,
            "passed": live_cuda,
        },
        "teardown": teardown,
        "vram_release": {"model_id": model_id, **release},
        "worker_error": worker.get("error"),
        "backend_stdout_hash": sha256_file(stdout_path),
        "backend_stderr_hash": sha256_file(stderr_path),
    }


def _label_catalog(repo_root: Path) -> dict[str, JsonDict]:  # pragma: no cover - late label I/O.
    sources = read_source_artifacts(repo_root)
    labels: dict[str, JsonDict] = {}
    for row in sources["exp6984"].get("authority_agreement_rows", []):
        labels[str(row["candidate_id"])] = {
            "source_block": "exp6984",
            "exact_label": row.get("certified_relation"),
        }
    sealed_path = (
        repo_root
        / "results/raw/experiment_6985_chronological_constraint_stream"
        / str(sources["exp6985"]["sealed_label_path"])
    )
    for line in sealed_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        for candidate_id, exact_label in zip(
            event["candidate_ids"], event["exact_labels"], strict=True
        ):
            labels[str(candidate_id)] = {
                "source_block": "exp6985",
                "exact_label": exact_label,
                "label_commitment_hash": event["label_commitment_hash"],
            }
    for row in sources["exp6976"].get("per_candidate_rows", []):
        if row.get("parse_success") is True:
            labels[_transfer_candidate_id(str(row["attempt_key"]))] = {
                "source_block": "exp6976_transfer",
                "exact_label": row.get("expected_label"),
                "exact_success": row.get("exact_semantic_success"),
            }
    return labels


def _run_manifest_process(
    rows: Sequence[Mapping[str, Any]], work_root: Path
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover - process isolation boundary.
    input_path = work_root / "manifest_input.json"
    output_path = work_root / "scoring_manifest.json"
    write_json_atomic(input_path, {"rows": [deepcopy(dict(row)) for row in rows]})
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6986_three_family_contrast_features",
        "--manifest-worker-input",
        str(input_path),
        "--manifest-worker-output",
        str(output_path),
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    receipt = {
        "process": "label_denied_manifest_builder",
        "exit_code": result.returncode,
        "input_field_names": sorted(_MANIFEST_INPUT_FIELDS),
        "denied_fields_visible": [],
        "input_hash": sha256_file(input_path),
        "output_hash": sha256_file(output_path),
        "stderr_hash": sha256_text(result.stderr),
        "passed": result.returncode == 0 and manifest.get("scoring_manifest_hash") is not None,
    }
    return manifest, receipt


def run(
    *,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    result_path: Path = RESULT_PATH,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:  # pragma: no cover - required live E2E path.
    started = time.perf_counter()
    specs = [deepcopy(dict(row)) for row in (model_specs or MODEL_SPECS)]
    lease_runtime_dir = Path(
        os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases")
    )
    preconditions = collect_preconditions(
        repo_root=repo_root,
        model_specs=specs,
        checkpoint_root=checkpoint_root,
        lease_runtime_dir=lease_runtime_dir,
    )
    source_hashes = _source_hashes(repo_root)
    model_hashes = {
        str(row.get("hf_id")): sha256_file(str(row.get("model_path", "")))
        if Path(str(row.get("model_path", ""))).is_file()
        else None
        for row in specs
    }
    if preconditions.get("all_passed") is not True:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            live_duration_s=0.0,
            preconditions=preconditions,
            model_specs=specs,
            source_artifact_hashes=source_hashes,
            model_file_hashes=model_hashes,
            candidate_source_count_rows=preconditions.get("candidate_source_count_rows", []),
        )
        write_json_atomic(result_path, artifact)
        return artifact

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    sources = read_source_artifacts(repo_root)
    projected = project_frozen_candidates(sources)
    del sources
    manifest, manifest_receipt = _run_manifest_process(projected, checkpoint_root)
    if manifest_receipt["passed"] is not True:
        preconditions = {
            **preconditions,
            "all_passed": False,
            "checks": [
                *preconditions["checks"],
                gate_check("label_denied_manifest_process", True, False),
            ],
        }
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            live_duration_s=0.0,
            preconditions=preconditions,
            model_specs=specs,
            source_artifact_hashes=source_hashes,
            model_file_hashes=model_hashes,
            label_denial_rows=[manifest_receipt],
            candidate_source_count_rows=candidate_source_count_rows(projected),
        )
        write_json_atomic(result_path, artifact)
        return artifact

    blocks = scoring_process_blocks(manifest["rows"])
    denial_rows = [manifest_receipt, *runtime_label_denial_rows()]
    family_results: list[JsonDict] = []
    devices = list(preconditions["gpu_topology"]["devices"])
    for model in specs:
        family = run_family_process(
            model=model,
            blocks=blocks,
            manifest_hash=manifest["scoring_manifest_hash"],
            work_root=checkpoint_root / "workers",
            devices=devices,
            lease_runtime_dir=lease_runtime_dir,
        )
        family_results.append(family)
        if family.get("complete") is not True:
            break

    sequence_rows = [
        deepcopy(dict(row)) for family in family_results for row in family["sequence_feature_rows"]
    ]
    raw_rows = [
        deepcopy(dict(row)) for family in family_results for row in family["raw_token_rows"]
    ]
    parser_rows = [
        deepcopy(dict(row)) for family in family_results for row in family["parser_feature_rows"]
    ]
    teardown_rows = [deepcopy(dict(family["teardown"])) for family in family_results]
    feature_rows = [deepcopy(dict(row)) for row in sequence_rows]
    joined_rows: list[JsonDict] = []
    if _teardown_complete(teardown_rows):
        joined_rows = join_labels(feature_rows, _label_catalog(repo_root), teardown_rows)
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        live_duration_s=sum(float(family["live_duration_s"]) for family in family_results),
        preconditions=preconditions,
        model_specs=specs,
        source_artifact_hashes=source_hashes,
        model_file_hashes=model_hashes,
        scoring_manifest_rows=manifest["rows"],
        scoring_manifest_hash=manifest["scoring_manifest_hash"],
        candidate_source_count_rows=candidate_source_count_rows(projected),
        per_candidate_model_rows=feature_rows,
        raw_token_rows=raw_rows,
        sequence_feature_rows=sequence_rows,
        parser_feature_rows=parser_rows,
        label_denial_rows=denial_rows,
        process_isolation_rows=[family["process_isolation"] for family in family_results],
        gpu_runtime_rows=[family["gpu_runtime"] for family in family_results],
        lease_rows=[row for family in family_results for row in family["lease_rows"]],
        checkpoint_rows=[row for family in family_results for row in family["checkpoint_rows"]],
        teardown_rows=teardown_rows,
        vram_release_rows=[family["vram_release"] for family in family_results],
        joined_label_rows=joined_rows,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command boundary.
    """Run the controller or one private label-denied worker mode."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--manifest-worker-input", type=Path)
    parser.add_argument("--manifest-worker-output", type=Path)
    parser.add_argument("--score-worker-payload", type=Path)
    parser.add_argument("--score-worker-output", type=Path)
    parser.add_argument("--score-worker-ready", type=Path)
    parser.add_argument("--score-worker-port", type=int)
    args = parser.parse_args(argv)
    if args.manifest_worker_input is not None:
        if args.manifest_worker_output is None:
            parser.error("--manifest-worker-output is required")
        return manifest_worker(args.manifest_worker_input, args.manifest_worker_output)
    if args.score_worker_payload is not None:
        required = (args.score_worker_output, args.score_worker_ready, args.score_worker_port)
        if any(value is None for value in required):
            parser.error("score worker output, ready path, and port are required")
        return _score_worker(
            payload_path=args.score_worker_payload,
            output_path=args.score_worker_output,
            ready_path=args.score_worker_ready,
            port=args.score_worker_port,
        )
    run(run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
