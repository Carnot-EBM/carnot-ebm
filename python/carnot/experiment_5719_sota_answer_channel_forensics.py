"""Exp5719 mandated-GGUF answer-channel forensics.

Spec refs: REQ-VERIFY-5719, SCENARIO-VERIFY-5719.

This experiment diagnoses a boundary failure, not model quality. Exp5708 proved
that CUDA offload was real, but its raw-completion protocol mixed three risks:
no native chat template, a 32-token answer budget, and a newline stop that can
halt before the answer sentinel. Exp5719 keeps those failure modes visible on
small exact controls, then qualifies a single downstream answer protocol only
when deterministic controls pass on at least two mandated GGUF model families.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf


JsonDict = dict[str, Any]
GenerationRunner = Callable[[JsonDict, list[JsonDict], list[JsonDict], JsonDict], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5719_sota_answer_channel_forensics.json")
RAW_RESPONSE_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_5719_sota_answer_channel_forensics.responses.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5719_sota_answer_channel_forensics.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5719_sota_answer_channel_forensics.py")

SCHEMA = "carnot.experiment_5719.sota_answer_channel_forensics.v1"
MANIFEST_SCHEMA = SCHEMA + ".manifest"
EXPERIMENT = 5719
EXPERIMENT_ID = "experiment_5719_sota_answer_channel_forensics"
MILESTONE = "2026.07.511"
RUN_DATE = "20260719"
INFERENCE_SUBSTRATE = "local_llama_cpp_python_cuda_gguf_diagnostic"
SPEC_REFS = ("REQ-VERIFY-5719", "SCENARIO-VERIFY-5719")

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_MODEL_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)
N_GPU_LAYERS_REQUESTED = -1
RANDOM_SEEDS: JsonDict = {
    "control_seed": 5719001,
    "base_seed": 5719,
    "runner_seed": 5719002,
}

_REGISTRY = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
MODEL_SPECS: list[JsonDict] = [
    {
        "name": str(_REGISTRY.get(hf_id, {}).get("name") or hf_id.rsplit("/", 1)[-1]),
        "hf_id": hf_id,
        "model_repo_id": hf_id,
        "family": "",
        "role": str(_REGISTRY.get(hf_id, {}).get("role") or ""),
        "active_params_b": _REGISTRY.get(hf_id, {}).get("active_params_b"),
        "total_params_b": _REGISTRY.get(hf_id, {}).get("total_params_b"),
        "quantization": str(_REGISTRY.get(hf_id, {}).get("quantization") or "Q4_K_M"),
        "min_vram_gb": _REGISTRY.get(hf_id, {}).get("min_vram_gb"),
        "headline_eligible": True,
        "legacy_smoke_only": False,
    }
    for hf_id in MANDATED_MODEL_IDS
]
for _spec in MODEL_SPECS:
    _spec["family"] = _spec["name"].replace(".", "-").replace("_", "-").lower()

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Every gate field names the evidence boundary it protects.",
    "MODEL_SPECS": "The three mandated GGUF identities are explicit and cannot drift to legacy smoke models.",
    "resolved_model_receipts": "Cache paths, file sizes, and local presence are auditable before inference.",
    "model_hashes": "Weight bytes are bound to the artifact.",
    "gguf_filenames": "Exact GGUF filenames are visible.",
    "quantizations": "Observed quantization is recorded per weight file.",
    "llama_cpp_version": "The Python runtime can be reconstructed.",
    "llama_cpp_build_info": "CUDA build evidence is inspectable.",
    "native_chat_template_receipts": "Embedded chat-template provenance is preserved.",
    "cuda_device_receipts": "NVIDIA devices and memory snapshots are preserved.",
    "n_gpu_layers_offloaded": "Positive offload evidence is separate from intent.",
    "gpu_memory_before_mb": "CPU fallback cannot hide as a baseline.",
    "gpu_memory_peak_mb": "During-run GPU allocation is visible.",
    "gpu_memory_after_mb": "Cleanup evidence is visible.",
    "cuda_offload_authenticated": "Per-model CUDA eligibility is explicit.",
    "cuda_offload_authenticated_score": "The two-model CUDA gate scalar is mechanical.",
    "control_manifest": "Controls and expected answers are frozen before outcomes.",
    "protocol_matrix": "Completion/chat, stop policy, and budget arms are preregistered.",
    "raw_response_manifest_path": "Raw row evidence is lossless and replayable.",
    "raw_response_hashes": "Each response is byte-bound.",
    "finish_reason_counts": "Termination behavior is counted directly.",
    "truncation_count": "Length failures remain separate.",
    "missing_answer_count": "Sentinel omission remains separate.",
    "repetition_failure_count": "Repetition loops remain separate.",
    "parse_failure_count": "Parser failures remain separate from semantic errors.",
    "semantic_error_count": "Exact wrong answers remain separate from parse errors.",
    "validator_disagreement_count": "Independent validator mismatch blocks readiness.",
    "root_cause_attribution": "Exp 5708 failure causes are reported explicitly.",
    "qualified_protocol": "The downstream answer channel is frozen only from controls.",
    "qualified_model_ids": "The qualifying mandated models are visible.",
    "qualified_model_count": "The model denominator is honest.",
    "positive_control_parse_rate": "Positive-control parse success is measurable.",
    "answer_channel_ready_score": "Readiness is a strict mechanical gate.",
    "native_json_grammar_used": "The retired grammar path stays closed.",
    "external_scorer_used": "No external judge can decide labels.",
    "retired_runtime_used": "Retired runtimes cannot qualify the channel.",
    "inference_substrate": "Execution provenance is declared.",
    "random_seeds": "Sampling replay is stable.",
    "reproducibility_checksum": "The artifact can be replayed.",
    "honest_verdict": "Terminal state starts complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
PRIMARY_VALIDATOR_VERSION = "exp5719_primary_exact_control_validators_v1"
SECONDARY_VALIDATOR_VERSION = "exp5719_secondary_exact_control_validators_v1"
QUANT_RE = re.compile(
    r"(UD-)?(?:Q\d(?:_K_[A-Z]+|_[0-9A-Z]+)?|IQ\d_[A-Z]+|BF16|F16)",
    re.I,
)
OFFLOAD_RE = re.compile(r"offloaded\s+(?P<offloaded>\d+)\s*/\s*(?P<total>\d+)\s+layers", re.I)


class ManifestReplayError(ValueError):
    """Raised when the raw-response manifest no longer matches sealed hashes."""


def canonical_json(value: Any) -> str:
    """Serialize JSON data deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed SHA-256 digest for byte evidence."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local GGUF file in chunks so large weights stay streamable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def model_family(hf_id: str) -> str:
    """Return the stable per-model family label used in row identifiers."""

    if hf_id == QWEN_ID:
        return "qwen3-6-35b-a3b"
    if hf_id == GEMMA31_ID:
        return "gemma-4-31b-it"
    if hf_id == GEMMA26_ID:
        return "gemma-4-26b-a4b-it"
    return hf_id.rsplit("/", 1)[-1].replace("-GGUF", "").replace(".", "-").lower()


def extract_quantization(filename: str) -> str:
    """Read the observed quantization token from a GGUF filename."""

    matches = list(QUANT_RE.finditer(filename))
    return matches[-1].group(0) if matches else "unknown"


def _base_controls() -> list[JsonDict]:
    return [
        {
            "control_id": "pos-fsm-00",
            "family": "finite_label",
            "polarity": "positive",
            "prompt": "Finite-label exact control. Valid labels are A, B, C, and D. The evidence line states final_label=D. Return the final label.",
            "expected": {"kind": "answer", "answer": "D"},
            "validator_payload": {"kind": "finite_label", "answer": "D", "options": ["A", "B", "C", "D"]},
        },
        {
            "control_id": "pos-fsm-01",
            "family": "finite_label",
            "polarity": "positive",
            "prompt": "Finite-label exact control. Valid labels are RED, GREEN, and BLUE. The evidence line states final_label=GREEN. Return the final label.",
            "expected": {"kind": "answer", "answer": "GREEN"},
            "validator_payload": {"kind": "finite_label", "answer": "GREEN", "options": ["RED", "GREEN", "BLUE"]},
        },
        {
            "control_id": "pos-arith-00",
            "family": "arithmetic",
            "polarity": "positive",
            "prompt": "Arithmetic control. Compute 3 + 4.",
            "expected": {"kind": "answer", "answer": "7"},
            "validator_payload": {"kind": "arithmetic", "op": "add", "a": 3, "b": 4},
        },
        {
            "control_id": "pos-arith-01",
            "family": "arithmetic",
            "polarity": "positive",
            "prompt": "Arithmetic control. Compute 9 - 4.",
            "expected": {"kind": "answer", "answer": "5"},
            "validator_payload": {"kind": "arithmetic", "op": "sub", "a": 9, "b": 4},
        },
        {
            "control_id": "pos-arith-02",
            "family": "arithmetic",
            "polarity": "positive",
            "prompt": "Arithmetic control. Compute (5 * 3) mod 7.",
            "expected": {"kind": "answer", "answer": "1"},
            "validator_payload": {"kind": "arithmetic", "op": "mul_mod", "a": 5, "b": 3, "modulus": 7},
        },
        {
            "control_id": "pos-arith-03",
            "family": "arithmetic",
            "polarity": "positive",
            "prompt": "Arithmetic control. Compute 2 * 4.",
            "expected": {"kind": "answer", "answer": "8"},
            "validator_payload": {"kind": "arithmetic", "op": "mul", "a": 2, "b": 4},
        },
        {
            "control_id": "neg-malformed-00",
            "family": "malformed_envelope",
            "polarity": "negative",
            "prompt": "Malformed-control. The value is ZETA, but this row checks whether missing envelopes are detected.",
            "expected": {"kind": "answer", "answer": "ZETA"},
            "expected_failure": "malformed_envelope",
            "validator_payload": {"kind": "finite_label", "answer": "ZETA", "options": ["ZETA"]},
        },
        {
            "control_id": "neg-truncation-00",
            "family": "truncation",
            "polarity": "negative",
            "prompt": "Truncation-control. The value is LONGFORM, but low budgets should expose length termination.",
            "expected": {"kind": "answer", "answer": "LONGFORM"},
            "expected_failure": "length_truncation",
            "validator_payload": {"kind": "finite_label", "answer": "LONGFORM", "options": ["LONGFORM"]},
        },
        {
            "control_id": "neg-repetition-00",
            "family": "repetition",
            "polarity": "negative",
            "prompt": "Repetition-control. The value is OMEGA, but repeated text must be measured directly.",
            "expected": {"kind": "answer", "answer": "OMEGA"},
            "expected_failure": "repetition",
            "validator_payload": {"kind": "finite_label", "answer": "OMEGA", "options": ["OMEGA"]},
        },
        {
            "control_id": "neg-sentinel-00",
            "family": "sentinel_omission",
            "polarity": "negative",
            "prompt": "Sentinel-control. The value is KAPPA, and omission of the sentinel must remain visible.",
            "expected": {"kind": "answer", "answer": "KAPPA"},
            "expected_failure": "sentinel_omission",
            "validator_payload": {"kind": "finite_label", "answer": "KAPPA", "options": ["KAPPA"]},
        },
    ]


def freeze_control_manifest() -> list[JsonDict]:
    """Return the fixed per-model controls before any model outcome is read."""

    controls: list[JsonDict] = []
    sequence_index = 0
    for hf_id in MANDATED_MODEL_IDS:
        for base in _base_controls():
            preimage = {
                "model_hf_id": hf_id,
                "control_id": base["control_id"],
                "prompt": base["prompt"],
                "expected": base["expected"],
                "validator_payload": base["validator_payload"],
                "polarity": base["polarity"],
            }
            row = {
                **base,
                "sequence_index": sequence_index,
                "model_hf_id": hf_id,
                "model_family": model_family(hf_id),
                "control_uid": f"{model_family(hf_id)}::{base['control_id']}",
                "source": "exp5719_disjoint_answer_channel_controls_v1",
                "timestamp_utc": f"2026-07-19T00:{sequence_index:02d}:00Z",
                "pre_outcome_hash": sha256_json(preimage),
            }
            controls.append(row)
            sequence_index += 1
    return controls


def controls_by_model(controls: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    """Group frozen controls by mandated model ID."""

    grouped: dict[str, list[JsonDict]] = {hf_id: [] for hf_id in MANDATED_MODEL_IDS}
    for control in controls:
        grouped[str(control["model_hf_id"])].append(dict(control))
    return grouped


def freeze_protocol_matrix() -> list[JsonDict]:
    """Return the preregistered completion/chat boundary matrix."""

    protocols = [
        {
            "protocol_id": "exp5708_raw_completion_newline_32",
            "mode": "completion",
            "stop": ["\n"],
            "stop_policy": "newline_stop",
            "max_tokens": 32,
            "prompt_style": "exp5708_answer_line",
            "sentinel": "ANSWER",
            "adequate_answer_budget": False,
            "native_chat_template": False,
            "reason_then_final": False,
            "selection_candidate": False,
            "selection_rank": 99,
        },
        {
            "protocol_id": "raw_completion_eos_answer_budget",
            "mode": "completion",
            "stop": [],
            "stop_policy": "eos_only",
            "max_tokens": 96,
            "prompt_style": "answer_line",
            "sentinel": "ANSWER",
            "adequate_answer_budget": True,
            "native_chat_template": False,
            "reason_then_final": False,
            "selection_candidate": False,
            "selection_rank": 50,
        },
        {
            "protocol_id": "chat_native_newline_budget",
            "mode": "chat",
            "stop": ["\n"],
            "stop_policy": "newline_stop",
            "max_tokens": 96,
            "prompt_style": "final_line",
            "sentinel": "FINAL",
            "adequate_answer_budget": True,
            "native_chat_template": True,
            "reason_then_final": False,
            "selection_candidate": False,
            "selection_rank": 40,
        },
        {
            "protocol_id": "chat_native_eos_answer_budget",
            "mode": "chat",
            "stop": [],
            "stop_policy": "eos_only",
            "max_tokens": 96,
            "prompt_style": "final_line",
            "sentinel": "FINAL",
            "adequate_answer_budget": True,
            "native_chat_template": True,
            "reason_then_final": False,
            "selection_candidate": True,
            "selection_rank": 2,
        },
        {
            "protocol_id": "chat_reason_final_eos_budget",
            "mode": "chat",
            "stop": [],
            "stop_policy": "eos_only",
            "max_tokens": 128,
            "prompt_style": "reason_then_final",
            "sentinel": "FINAL",
            "adequate_answer_budget": True,
            "native_chat_template": True,
            "reason_then_final": True,
            "selection_candidate": True,
            "selection_rank": 1,
        },
    ]
    return [{**row, "protocol_hash": sha256_json(row)} for row in protocols]


def generation_config_for_protocol(protocol: Mapping[str, Any]) -> JsonDict:
    """Return llama.cpp sampling parameters for a protocol row."""

    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(protocol["max_tokens"]),
        "stop": list(protocol.get("stop") or []),
        "n_ctx": 2048,
        "n_batch": 256,
        "n_gpu_layers": N_GPU_LAYERS_REQUESTED,
        "native_json_grammar_used": False,
        "logit_bias_used": False,
    }


def expected_answer_text(control: Mapping[str, Any]) -> str:
    """Return the expected exact answer token for a frozen control."""

    return str(control["expected"]["answer"])


def prompt_for_control(control: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    """Build the protocol-specific prompt without relying on external templates."""

    task = str(control["prompt"])
    if protocol["sentinel"] == "ANSWER":
        return (
            "/no_think\nReturn exactly one line in this form: ANSWER: <value>. "
            "No hidden thinking process. Do not output JSON. Task: " + task
        )
    if protocol.get("reason_then_final") is True:
        return (
            "/no_think\nDo not write a hidden thinking process. Output exactly two lines:\n"
            "Reason: direct exact control\nFINAL: <value>\nDo not output JSON. Task: " + task
        )
    return (
        "/no_think\nReturn exactly one line in this form: FINAL: <value>. "
        "No other text. Do not output JSON. Task: " + task
    )


def parse_protocol_answer(raw_text: str, protocol: Mapping[str, Any]) -> JsonDict:
    """Extract the required protocol sentinel with deterministic regex parsing."""

    sentinel = str(protocol["sentinel"]).upper()
    pattern = re.compile(rf"(?im)^\s*{re.escape(sentinel)}\s*:\s*(?P<answer>[^\r\n]*)\s*$")
    matches = list(pattern.finditer(raw_text))
    if not matches:
        return {
            "parse_ok": False,
            "answer": "",
            "error": f"missing_{sentinel.lower()}",
            "sentinel": sentinel,
        }
    answer = matches[-1].group("answer").strip()
    if not answer:
        return {"parse_ok": False, "answer": "", "error": "empty_answer", "sentinel": sentinel}
    return {"parse_ok": True, "answer": answer, "error": "", "sentinel": sentinel}


def _expected_by_primary(control: Mapping[str, Any]) -> str:
    payload = control["validator_payload"]
    kind = str(payload["kind"])
    if kind == "finite_label":
        return str(payload["answer"])
    if kind == "arithmetic":
        op = str(payload["op"])
        a = int(payload["a"])
        b = int(payload["b"])
        if op == "add":
            return str(a + b)
        if op == "sub":
            return str(a - b)
        if op == "mul":
            return str(a * b)
        if op == "mul_mod":
            return str((a * b) % int(payload["modulus"]))
    raise ValueError(f"unknown validator payload: {kind}")


def _expected_by_secondary(control: Mapping[str, Any]) -> str:
    payload = control["validator_payload"]
    if payload["kind"] == "arithmetic" and payload["op"] == "mul_mod":
        domain = range(int(payload["modulus"]))
        value = (int(payload["a"]) * int(payload["b"])) % int(payload["modulus"])
        return str(next(candidate for candidate in domain if candidate == value))
    return str(control["expected"]["answer"])


def primary_validate_control(control: Mapping[str, Any], parsed: Mapping[str, Any]) -> JsonDict:
    """Validate one control with the primary deterministic implementation."""

    expected = _expected_by_primary(control)
    if parsed.get("parse_ok") is not True:
        return {
            "validator_version": PRIMARY_VALIDATOR_VERSION,
            "parse_ok": False,
            "label": False,
            "expected_answer": expected,
            "observed_answer": "",
            "error": str(parsed.get("error") or "parse_failure"),
        }
    observed = str(parsed["answer"])
    return {
        "validator_version": PRIMARY_VALIDATOR_VERSION,
        "parse_ok": True,
        "label": observed == expected,
        "expected_answer": expected,
        "observed_answer": observed,
        "error": "",
    }


def secondary_validate_control(control: Mapping[str, Any], parsed: Mapping[str, Any]) -> JsonDict:
    """Validate one control with a second exact implementation."""

    expected = _expected_by_secondary(control)
    if parsed.get("parse_ok") is not True:
        return {
            "validator_version": SECONDARY_VALIDATOR_VERSION,
            "parse_ok": False,
            "label": False,
            "expected_answer": expected,
            "observed_answer": "",
            "error": str(parsed.get("error") or "parse_failure"),
        }
    observed = str(parsed["answer"])
    return {
        "validator_version": SECONDARY_VALIDATOR_VERSION,
        "parse_ok": True,
        "label": observed == expected,
        "expected_answer": expected,
        "observed_answer": observed,
        "error": "",
    }


def repetition_metrics(raw_text: str) -> JsonDict:
    """Measure direct repetition telemetry without using a semantic judge."""

    words = re.findall(r"\S+", raw_text.lower())
    max_same_token_run = 0
    current_token = None
    current_run = 0
    for word in words:
        if word == current_token:
            current_run += 1
        else:
            current_token = word
            current_run = 1
        max_same_token_run = max(max_same_token_run, current_run)
    bigrams = [" ".join(words[i : i + 2]) for i in range(max(0, len(words) - 1))]
    repeated_bigram_max = max(Counter(bigrams).values(), default=0)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    repeated_line_max = max(Counter(lines).values(), default=0)
    failure = max_same_token_run >= 8 or repeated_bigram_max >= 4 or repeated_line_max >= 3
    return {
        "word_count": len(words),
        "max_same_token_run": max_same_token_run,
        "repeated_bigram_max": repeated_bigram_max,
        "repeated_line_max": repeated_line_max,
        "repetition_failure": failure,
        "bounded_repetition": not failure,
    }


def classify_failure_row(row: Mapping[str, Any]) -> list[str]:
    """Classify diagnosis failures without collapsing them into one parse flag."""

    classes: list[str] = []
    finish_reason = str(row.get("finish_reason") or "").lower()
    parser = dict(row.get("parser_result") or {})
    token_counts = dict(row.get("token_counts") or {})
    completion_tokens = int(token_counts.get("completion_tokens", 0) or 0)
    max_tokens = int(row.get("max_tokens", 0) or 0)
    if row.get("error"):
        classes.append("runtime_failure")
    if row.get("protocol_mode") == "completion" and parser.get("parse_ok") is not True:
        classes.append("template_mismatch")
    if finish_reason in {"length", "max_tokens", "truncated"} or (
        max_tokens > 0 and completion_tokens >= max_tokens and parser.get("parse_ok") is not True
    ):
        classes.append("length_truncation")
    if (
        finish_reason == "stop"
        and row.get("protocol_stop") == ["\n"]
        and parser.get("parse_ok") is not True
    ):
        classes.append("premature_stop")
    metrics = dict(row.get("repetition") or repetition_metrics(str(row.get("raw_text") or "")))
    if metrics.get("repetition_failure") is True:
        classes.append("repetition")
    if parser.get("parse_ok") is not True:
        error = str(parser.get("error") or "")
        if error.startswith("missing_"):
            classes.append("sentinel_omission")
        else:
            classes.append("malformed_envelope")
    primary = dict(row.get("primary_validation") or {})
    secondary = dict(row.get("secondary_validation") or {})
    if (
        row.get("control_polarity") == "positive"
        and parser.get("parse_ok") is True
        and primary.get("label") is not True
    ):
        classes.append("semantic_exact_error")
    if primary.get("label") != secondary.get("label") or primary.get("expected_answer") != secondary.get(
        "expected_answer"
    ):
        classes.append("validator_disagreement")
    return list(dict.fromkeys(classes))


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]] | None = None) -> list[JsonDict]:
    """Resolve and hash all mandated GGUFs without transformers tokenization."""

    sources = {str(row.get("hf_id")): row for row in model_specs or []}
    normalized: list[JsonDict] = []
    for index, base in enumerate(MODEL_SPECS):
        hf_id = str(base["hf_id"])
        source = sources.get(hf_id, {})
        resolved = str(
            source.get("model_path")
            or source.get("resolved_model_path")
            or resolve_cached_gguf(hf_id, str(base.get("quantization") or "Q4_K_M"))
            or ""
        )
        path = Path(resolved).expanduser() if resolved else Path()
        present = bool(resolved and path.is_file())
        filename = path.name if resolved else ""
        normalized.append(
            {
                **base,
                "sequence_index": index,
                "family": model_family(hf_id),
                "gpu": int(source.get("gpu", index % 2) or 0),
                "resolved_model_path": resolved,
                "model_path": resolved,
                "gguf_filename": filename,
                "model_hash": sha256_file(path) if present else "",
                "model_size_bytes": path.stat().st_size if present else 0,
                "quantization": extract_quantization(filename) if filename else str(base["quantization"]),
                "local_model_present": present,
                "headline_eligible": source.get("headline_eligible") is not False,
                "legacy_smoke_only": False,
            }
        )
    return normalized


def _runtime_row_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], JsonDict]:
    return {
        (
            str(row.get("model_hf_id")),
            str(row.get("control_id")),
            str(row.get("protocol_id")),
        ): dict(row)
        for row in rows
    }


def row_uid(control: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    """Return the stable raw-response manifest row ID."""

    return f"{control['control_uid']}::{protocol['protocol_id']}"


def build_manifest_rows(
    *,
    controls: Sequence[Mapping[str, Any]],
    protocol_matrix: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join preregistered controls with model outputs, parsers, and validators."""

    runtime_rows: list[JsonDict] = []
    for receipt in runtime_receipts:
        runtime_rows.extend(dict(row) for row in receipt.get("rows", []))
    raw_by_key = _runtime_row_map(runtime_rows)
    rows: list[JsonDict] = []
    previous_hash = ""
    sequence_index = 0
    for control in controls:
        for protocol in protocol_matrix:
            raw = raw_by_key.get(
                (str(control["model_hf_id"]), str(control["control_id"]), str(protocol["protocol_id"]))
            )
            missing = raw is None
            prompt = prompt_for_control(control, protocol)
            raw_text = "" if raw is None else str(raw.get("raw_text", ""))
            finish_reason = "missing" if raw is None else str(raw.get("finish_reason") or "")
            parsed = parse_protocol_answer(raw_text, protocol)
            if missing:
                parsed = {
                    "parse_ok": False,
                    "answer": "",
                    "error": "missing_generation",
                    "sentinel": str(protocol["sentinel"]),
                }
            primary = primary_validate_control(control, parsed)
            secondary = secondary_validate_control(control, parsed)
            repetition = repetition_metrics(raw_text)
            row: JsonDict = {
                "schema": MANIFEST_SCHEMA,
                "sequence_index": sequence_index,
                "row_uid": row_uid(control, protocol),
                "model_hf_id": str(control["model_hf_id"]),
                "model_family": str(control["model_family"]),
                "control_id": str(control["control_id"]),
                "control_uid": str(control["control_uid"]),
                "control_family": str(control["family"]),
                "control_polarity": str(control["polarity"]),
                "control_expected_failure": str(control.get("expected_failure") or ""),
                "protocol_id": str(protocol["protocol_id"]),
                "protocol_mode": str(protocol["mode"]),
                "protocol_stop": list(protocol.get("stop") or []),
                "protocol_hash": str(protocol["protocol_hash"]),
                "max_tokens": int(protocol["max_tokens"]),
                "prompt": prompt,
                "prompt_hash": sha256_text(prompt),
                "pre_outcome_hash": str(control["pre_outcome_hash"]),
                "native_template_hash": "" if raw is None else str(raw.get("template_hash") or ""),
                "raw_text": raw_text,
                "raw_response_hash": sha256_text(raw_text),
                "finish_reason": finish_reason,
                "missing_generation": missing,
                "token_counts": {} if raw is None else dict(raw.get("token_counts") or {}),
                "timing": {} if raw is None else dict(raw.get("timing") or {}),
                "seed": None if raw is None else raw.get("seed"),
                "generation_config": generation_config_for_protocol(protocol)
                if raw is None
                else dict(raw.get("generation_config") or generation_config_for_protocol(protocol)),
                "telemetry": {} if raw is None else dict(raw.get("telemetry") or {}),
                "repetition": repetition,
                "parser_result": parsed,
                "primary_validation": primary,
                "secondary_validation": secondary,
                "validator_disagreement": primary["label"] != secondary["label"]
                or primary["expected_answer"] != secondary["expected_answer"],
                "error": "" if raw is None else str(raw.get("error") or ""),
                "failure_classes": [],
                "previous_row_hash": previous_hash,
                "row_hash": "",
            }
            row["failure_classes"] = classify_failure_row(row)
            row["row_hash"] = manifest_row_hash(row)
            previous_hash = row["row_hash"]
            rows.append(row)
            sequence_index += 1
    return rows


def manifest_row_hash(row: Mapping[str, Any]) -> str:
    """Hash a manifest row while excluding its own hash field."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def write_manifest_rows(rows: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    """Write raw-response evidence as JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_manifest_rows(path: Path | str) -> list[JsonDict]:
    """Read a JSONL raw-response manifest from disk."""

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def verify_manifest_rows(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Replay row hashes, raw hashes, and the chronological manifest chain."""

    expected_hashes = dict(artifact.get("raw_response_hashes") or {})
    previous = ""
    for row in rows:
        if row.get("previous_row_hash") != previous:
            raise ManifestReplayError("previous_row_hash")
        raw_hash = sha256_text(str(row.get("raw_text", "")))
        if raw_hash != row.get("raw_response_hash"):
            raise ManifestReplayError("raw_response_hash")
        if expected_hashes.get(str(row.get("row_uid"))) != raw_hash:
            raise ManifestReplayError("raw_response_hash")
        if manifest_row_hash(row) != row.get("row_hash"):
            raise ManifestReplayError("row_hash")
        previous = str(row["row_hash"])
    return True


def raw_response_hashes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return row UID to raw-response hash mapping."""

    return {str(row["row_uid"]): str(row["raw_response_hash"]) for row in rows}


def _runtime_cuda_authenticated(receipt: Mapping[str, Any]) -> bool:
    return bool(
        receipt.get("cuda_offload_authenticated") is True
        and int(receipt.get("n_gpu_layers_offloaded") or 0) > 0
        and int(receipt.get("gpu_memory_peak_mb") or 0) > int(receipt.get("gpu_memory_before_mb") or 0)
    )


def _model_passes_protocol(
    *,
    hf_id: str,
    protocol_id: str,
    manifest_rows: Sequence[Mapping[str, Any]],
    cuda_authenticated: Mapping[str, bool],
) -> bool:
    positives = [
        row
        for row in manifest_rows
        if row.get("model_hf_id") == hf_id
        and row.get("protocol_id") == protocol_id
        and row.get("control_polarity") == "positive"
    ]
    if len(positives) < 6 or cuda_authenticated.get(hf_id) is not True:
        return False
    for row in positives:
        if row.get("missing_generation") is True:
            return False
        if row.get("validator_disagreement") is True:
            return False
        if row.get("parser_result", {}).get("parse_ok") is not True:
            return False
        if row.get("primary_validation", {}).get("label") is not True:
            return False
        if any(cls in row.get("failure_classes", []) for cls in ("length_truncation", "repetition")):
            return False
    return True


def select_qualified_protocol(
    *,
    protocol_matrix: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    cuda_authenticated: Mapping[str, bool],
) -> tuple[JsonDict | None, list[str]]:
    """Choose the preregistered protocol with at least two qualified families."""

    candidates = sorted(
        (dict(row) for row in protocol_matrix if row.get("selection_candidate") is True),
        key=lambda row: int(row["selection_rank"]),
    )
    best_partial: list[str] = []
    for protocol in candidates:
        qualified = [
            hf_id
            for hf_id in MANDATED_MODEL_IDS
            if _model_passes_protocol(
                hf_id=hf_id,
                protocol_id=str(protocol["protocol_id"]),
                manifest_rows=manifest_rows,
                cuda_authenticated=cuda_authenticated,
            )
        ]
        if len(qualified) > len(best_partial):
            best_partial = qualified
        if len(qualified) >= 2:
            return protocol, qualified
    return None, best_partial


def positive_control_parse_rate_for_selection(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    qualified_protocol: Mapping[str, Any] | None,
    qualified_model_ids: Sequence[str],
) -> float:
    """Measure parse success on selected positive controls only."""

    if not qualified_protocol or not qualified_model_ids:
        return 0.0
    rows = [
        row
        for row in manifest_rows
        if row.get("protocol_id") == qualified_protocol.get("protocol_id")
        and row.get("model_hf_id") in set(qualified_model_ids)
        and row.get("control_polarity") == "positive"
    ]
    if not rows:
        return 0.0
    parsed = sum(1 for row in rows if row.get("parser_result", {}).get("parse_ok") is True)
    return round(parsed / len(rows), 6)


def root_cause_attribution(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize the Exp5708 raw-completion control failure modes."""

    raw_rows = [row for row in rows if row.get("protocol_id") == "exp5708_raw_completion_newline_32"]
    counts = Counter(cls for row in raw_rows for cls in row.get("failure_classes", []))
    return {
        "exp5708_raw_completion_control": {
            "rows": len(raw_rows),
            "template_mismatch": int(counts.get("template_mismatch", 0)),
            "length_truncation": int(counts.get("length_truncation", 0)),
            "premature_stop": int(counts.get("premature_stop", 0)),
            "sentinel_omission": int(counts.get("sentinel_omission", 0)),
            "repetition": int(counts.get("repetition", 0)),
            "malformed_envelope": int(counts.get("malformed_envelope", 0)),
            "runtime_failure": int(counts.get("runtime_failure", 0)),
            "semantic_exact_error": int(counts.get("semantic_exact_error", 0)),
            "attribution": "raw completion plus newline stop and 32-token budget is diagnostic-only; native chat FINAL/EOS controls decide qualification.",
        }
    }


def cuda_offload_authenticated_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when at least two qualified models have CUDA evidence."""

    qualified = list(artifact.get("qualified_model_ids") or [])
    cuda_map = dict(artifact.get("cuda_offload_authenticated") or {})
    layer_map = dict(artifact.get("n_gpu_layers_offloaded") or {})
    before_map = dict(artifact.get("gpu_memory_before_mb") or {})
    peak_map = dict(artifact.get("gpu_memory_peak_mb") or {})
    count = 0
    for hf_id in qualified:
        if (
            cuda_map.get(hf_id) is True
            and int(layer_map.get(hf_id, 0) or 0) > 0
            and int(peak_map.get(hf_id, 0) or 0) > int(before_map.get(hf_id, 0) or 0)
        ):
            count += 1
    return 1.0 if count >= 2 else 0.0


def answer_channel_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when the selected answer channel satisfies all gates."""

    ready = bool(
        artifact.get("qualified_protocol")
        and int(artifact.get("qualified_model_count") or 0) >= 2
        and artifact.get("positive_control_parse_rate") == 1.0
        and artifact.get("cuda_offload_authenticated_score") == 1.0
        and artifact.get("native_json_grammar_used") is False
        and artifact.get("external_scorer_used") is False
        and artifact.get("retired_runtime_used") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    return 1.0 if ready else 0.0


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if artifact.get("qualified_protocol") in ({}, None):
        reasons.append("no_qualified_protocol")
    if int(artifact.get("qualified_model_count") or 0) < 2:
        reasons.append("fewer_than_two_qualified_models")
    if artifact.get("positive_control_parse_rate") != 1.0:
        reasons.append("positive_control_parse_rate_below_one")
    if artifact.get("cuda_offload_authenticated_score") != 1.0:
        reasons.append("cuda_offload_unauthenticated")
    if artifact.get("native_json_grammar_used") is not False:
        reasons.append("native_json_grammar_used")
    if artifact.get("external_scorer_used") is not False:
        reasons.append("external_scorer_used")
    if artifact.get("retired_runtime_used") is not False:
        reasons.append("retired_runtime_used")
    return reasons or ["answer_channel_gate_not_met"]


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict from mechanical readiness gates."""

    if float(artifact.get("answer_channel_ready_score") or 0.0) == 1.0:
        return "complete: answer_channel_protocol_qualified"
    return "blocked: " + ",".join(_blocked_reasons(artifact))


def build_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    raw_response_manifest_path: str,
    tests_added_or_reused: Sequence[str] = (),
) -> JsonDict:
    """Build the terminal Exp5719 artifact from sealed diagnostics."""

    specs = normalize_model_specs(model_specs)
    controls = freeze_control_manifest()
    protocols = freeze_protocol_matrix()
    receipts_by_model = {
        str(receipt.get("model_hf_id") or receipt.get("hf_id")): dict(receipt)
        for receipt in runtime_receipts
    }
    cuda_auth = {
        hf_id: _runtime_cuda_authenticated(receipts_by_model.get(hf_id, {}))
        for hf_id in MANDATED_MODEL_IDS
    }
    qualified_protocol, qualified_ids = select_qualified_protocol(
        protocol_matrix=protocols,
        manifest_rows=manifest_rows,
        cuda_authenticated=cuda_auth,
    )
    positive_parse_rate = positive_control_parse_rate_for_selection(
        manifest_rows=manifest_rows,
        qualified_protocol=qualified_protocol,
        qualified_model_ids=qualified_ids,
    )
    finish_reason_counts = dict(Counter(str(row.get("finish_reason") or "") for row in manifest_rows))
    failure_counts = Counter(cls for row in manifest_rows for cls in row.get("failure_classes", []))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "MODEL_SPECS": specs,
        "resolved_model_receipts": {
            spec["hf_id"]: {
                "resolved_model_path": spec["resolved_model_path"],
                "local_model_present": spec["local_model_present"],
                "model_size_bytes": spec["model_size_bytes"],
                "model_hash": spec["model_hash"],
            }
            for spec in specs
        },
        "model_hashes": {spec["hf_id"]: spec["model_hash"] for spec in specs},
        "gguf_filenames": {spec["hf_id"]: spec["gguf_filename"] for spec in specs},
        "quantizations": {spec["hf_id"]: spec["quantization"] for spec in specs},
        "llama_cpp_version": next(
            (str(row.get("llama_cpp_version")) for row in runtime_receipts if row.get("llama_cpp_version")),
            "",
        ),
        "llama_cpp_build_info": next(
            (dict(row.get("llama_cpp_build_info") or {}) for row in runtime_receipts if row.get("llama_cpp_build_info")),
            {},
        ),
        "native_chat_template_receipts": {
            hf_id: dict(receipts_by_model.get(hf_id, {}).get("native_chat_template_receipt") or {})
            for hf_id in MANDATED_MODEL_IDS
        },
        "cuda_device_receipts": {
            hf_id: dict(receipts_by_model.get(hf_id, {}).get("cuda_device_receipt") or {})
            for hf_id in MANDATED_MODEL_IDS
        },
        "n_gpu_layers_offloaded": {
            hf_id: int(receipts_by_model.get(hf_id, {}).get("n_gpu_layers_offloaded") or 0)
            for hf_id in MANDATED_MODEL_IDS
        },
        "gpu_memory_before_mb": {
            hf_id: int(receipts_by_model.get(hf_id, {}).get("gpu_memory_before_mb") or 0)
            for hf_id in MANDATED_MODEL_IDS
        },
        "gpu_memory_peak_mb": {
            hf_id: int(receipts_by_model.get(hf_id, {}).get("gpu_memory_peak_mb") or 0)
            for hf_id in MANDATED_MODEL_IDS
        },
        "gpu_memory_after_mb": {
            hf_id: int(receipts_by_model.get(hf_id, {}).get("gpu_memory_after_mb") or 0)
            for hf_id in MANDATED_MODEL_IDS
        },
        "cuda_offload_authenticated": cuda_auth,
        "cuda_offload_authenticated_score": 0.0,
        "control_manifest": controls,
        "protocol_matrix": protocols,
        "raw_response_manifest_path": raw_response_manifest_path,
        "raw_response_hashes": raw_response_hashes(manifest_rows),
        "finish_reason_counts": finish_reason_counts,
        "truncation_count": int(failure_counts.get("length_truncation", 0)),
        "missing_answer_count": int(failure_counts.get("sentinel_omission", 0)),
        "repetition_failure_count": int(failure_counts.get("repetition", 0)),
        "parse_failure_count": sum(1 for row in manifest_rows if row.get("parser_result", {}).get("parse_ok") is not True),
        "semantic_error_count": int(failure_counts.get("semantic_exact_error", 0)),
        "validator_disagreement_count": int(failure_counts.get("validator_disagreement", 0)),
        "root_cause_attribution": root_cause_attribution(manifest_rows),
        "qualified_protocol": qualified_protocol or {},
        "qualified_model_ids": qualified_ids,
        "qualified_model_count": len(qualified_ids),
        "positive_control_parse_rate": positive_parse_rate,
        "answer_channel_ready_score": 0.0,
        "native_json_grammar_used": False,
        "external_scorer_used": False,
        "retired_runtime_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "failure_class_counts": dict(failure_counts),
        "manifest_row_count": len(manifest_rows),
        "exact_validator_versions": {
            "primary": PRIMARY_VALIDATOR_VERSION,
            "secondary": SECONDARY_VALIDATOR_VERSION,
            "validator_authority": "deterministic_exact_controls",
        },
        "legacy_smoke_models": [
            {"hf_id": "Qwen/Qwen3.5-0.8B", "certificate_eligible": False},
            {"hf_id": "google/gemma-4-E4B-it", "certificate_eligible": False},
        ],
        "logit_bias_used": False,
        "xgrammar_used": False,
        "llguidance_used": False,
        "tests_added_or_reused": list(tests_added_or_reused),
    }
    artifact["cuda_offload_authenticated_score"] = cuda_offload_authenticated_score(artifact)
    artifact["answer_channel_ready_score"] = answer_channel_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["blocked_reasons"] = (
        [] if artifact["answer_channel_ready_score"] == 1.0 else _blocked_reasons(artifact)
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on schema drift or unsupported readiness claims."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if principles.get(field) != principle:
            raise ValueError("field_principles")
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(MANDATED_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    for forbidden in ("native_json_grammar_used", "external_scorer_used", "retired_runtime_used"):
        if artifact.get(forbidden) is not False:
            raise ValueError(forbidden)
    expected_cuda_score = cuda_offload_authenticated_score(artifact)
    if artifact.get("cuda_offload_authenticated_score") != expected_cuda_score:
        raise ValueError("cuda_offload_authenticated_score")
    expected_score = answer_channel_ready_score(artifact)
    if artifact.get("answer_channel_ready_score") != expected_score:
        raise ValueError("answer_channel_ready_score")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_score == 1.0 and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if expected_score == 0.0 and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _blocked_runtime_receipt(model_spec: Mapping[str, Any], reason: str) -> JsonDict:
    return {
        "model_hf_id": str(model_spec["hf_id"]),
        "llama_cpp_version": "",
        "llama_cpp_build_info": {"blocked_reason": reason},
        "native_chat_template_receipt": {
            "model_hf_id": str(model_spec["hf_id"]),
            "source": "unavailable",
            "template_hash": "",
            "template_preview": "",
        },
        "cuda_device_receipt": {"before": [], "peak": [], "after": []},
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": 0,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 0,
        "gpu_memory_after_mb": 0,
        "cuda_offload_authenticated": False,
        "offload_log_excerpt": "",
        "rows": [],
        "blocked_reason": reason,
    }


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    raw_response_manifest_path: Path | str = REPO_ROOT / RAW_RESPONSE_MANIFEST_RELATIVE_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    generation_runner: GenerationRunner | None = None,
    tests_added_or_reused: Sequence[str] = (),
    write: bool = True,
) -> JsonDict:
    """Run the diagnostic matrix or write an honest blocked artifact."""

    specs = normalize_model_specs(model_specs)
    controls = freeze_control_manifest()
    protocols = freeze_protocol_matrix()
    controls_by_hf_id = controls_by_model(controls)
    runner = generation_runner or default_generation_runner
    runtime_receipts: list[JsonDict] = []
    for spec in specs:
        if spec["local_model_present"] is not True:
            runtime_receipts.append(_blocked_runtime_receipt(spec, "mandated_gguf_missing"))
            continue
        receipt = runner(spec, controls_by_hf_id[str(spec["hf_id"])], protocols, dict(RANDOM_SEEDS))
        receipt.setdefault("model_hf_id", str(spec["hf_id"]))
        runtime_receipts.append(receipt)
    manifest_rows = build_manifest_rows(
        controls=controls,
        protocol_matrix=protocols,
        runtime_receipts=runtime_receipts,
    )
    artifact = build_artifact(
        model_specs=specs,
        runtime_receipts=runtime_receipts,
        manifest_rows=manifest_rows,
        raw_response_manifest_path=str(Path(raw_response_manifest_path)),
        tests_added_or_reused=tests_added_or_reused,
    )
    if write:
        write_manifest_rows(manifest_rows, raw_response_manifest_path)
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def parse_offloaded_layers(stderr_text: str) -> int:  # pragma: no cover - live telemetry helper.
    """Extract positive llama.cpp offload evidence from backend logs."""

    matches = list(OFFLOAD_RE.finditer(stderr_text))
    if not matches:
        return 0
    return max(int(match.group("offloaded")) for match in matches)


def _nvidia_smi_devices() -> list[JsonDict]:  # pragma: no cover - host dependent.
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(query, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return [{"error": str(exc)}]
    devices = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "memory_used_mb": int(parts[5]),
                }
            )
    return devices


def _gpu_used_total_mb() -> int:  # pragma: no cover - host dependent.
    return sum(int(row.get("memory_used_mb", 0) or 0) for row in _nvidia_smi_devices())


def default_generation_runner(
    model_spec: JsonDict,
    controls: list[JsonDict],
    protocol_matrix: list[JsonDict],
    random_seeds: JsonDict,
) -> JsonDict:  # pragma: no cover - host dependent live path.
    """Run one model's protocol matrix through llama-cpp-python in a child process."""

    devices_before = _nvidia_smi_devices()
    before_mb = _gpu_used_total_mb()
    worker_payload = {
        "model_spec": model_spec,
        "controls": controls,
        "protocol_matrix": protocol_matrix,
        "random_seeds": random_seeds,
    }
    worker_code = r'''
import gc
import importlib.metadata
import json
import re
import sys
import time

payload = json.load(sys.stdin)

def prompt_for_control(control, protocol):
    task = str(control["prompt"])
    if protocol["sentinel"] == "ANSWER":
        return "/no_think\nReturn exactly one line in this form: ANSWER: <value>. No hidden thinking process. Do not output JSON. Task: " + task
    if protocol.get("reason_then_final") is True:
        return "/no_think\nDo not write a hidden thinking process. Output exactly two lines:\nReason: direct exact control\nFINAL: <value>\nDo not output JSON. Task: " + task
    return "/no_think\nReturn exactly one line in this form: FINAL: <value>. No other text. Do not output JSON. Task: " + task

def generation_config_for_protocol(protocol):
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(protocol["max_tokens"]),
        "stop": list(protocol.get("stop") or []),
        "n_ctx": 2048,
        "n_batch": 256,
        "n_gpu_layers": -1,
        "native_json_grammar_used": False,
        "logit_bias_used": False,
    }

try:
    import llama_cpp
    from llama_cpp import Llama
    version = importlib.metadata.version("llama-cpp-python")
    system_info = ""
    try:
        raw_info = llama_cpp.llama_cpp.llama_print_system_info()
        system_info = raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
    except Exception as exc:
        system_info = f"system_info_unavailable: {exc}"
    llm = Llama(
        model_path=payload["model_spec"]["resolved_model_path"],
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=256,
        seed=int(payload["random_seeds"]["runner_seed"]),
        verbose=True,
    )
    metadata = dict(getattr(llm, "metadata", {}) or {})
    template = ""
    for key, value in metadata.items():
        if "chat_template" in str(key).lower():
            template = str(value)
            break
    rows = []
    for control_index, control in enumerate(payload["controls"]):
        for protocol in payload["protocol_matrix"]:
            prompt = prompt_for_control(control, protocol)
            config = generation_config_for_protocol(protocol)
            started = time.perf_counter()
            if protocol["mode"] == "chat":
                result = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=int(config["max_tokens"]),
                    temperature=0.0,
                    top_p=1.0,
                    stop=(list(config["stop"]) or None),
                )
                choice = result.get("choices", [{}])[0]
                raw_text = str(choice.get("message", {}).get("content", ""))
            else:
                result = llm(
                    prompt,
                    max_tokens=int(config["max_tokens"]),
                    temperature=0.0,
                    top_p=1.0,
                    stop=(list(config["stop"]) or None),
                    echo=False,
                )
                choice = result.get("choices", [{}])[0]
                raw_text = str(choice.get("text", ""))
            elapsed = time.perf_counter() - started
            usage = result.get("usage", {})
            rows.append({
                "model_hf_id": payload["model_spec"]["hf_id"],
                "control_id": control["control_id"],
                "protocol_id": protocol["protocol_id"],
                "prompt": prompt,
                "raw_text": raw_text,
                "finish_reason": str(choice.get("finish_reason", "")),
                "token_counts": {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                },
                "timing": {"load_s": 0.0, "generation_s": round(elapsed, 6)},
                "seed": int(payload["random_seeds"]["base_seed"]) + control_index,
                "generation_config": config,
                "telemetry": {},
                "template_hash": "",
                "error": "",
            })
    del llm
    gc.collect()
    print(json.dumps({
        "ok": True,
        "llama_cpp_version": version,
        "llama_cpp_build_info": {
            "cuda_backend": "CUDA" in system_info.upper(),
            "system_info": system_info,
            "module": getattr(llama_cpp, "__file__", ""),
        },
        "native_chat_template_receipt": {
            "model_hf_id": payload["model_spec"]["hf_id"],
            "source": "llama_cpp_embedded_metadata" if template else "metadata_missing",
            "template_text": template,
            "template_preview": template[:500],
        },
        "rows": rows,
    }, sort_keys=True))
except Exception as exc:
    print(json.dumps({"ok": False, "error": repr(exc), "rows": []}, sort_keys=True))
    raise
'''
    proc = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stop_monitor = threading.Event()
    samples: list[int] = []

    def _monitor() -> None:
        while not stop_monitor.is_set():
            samples.append(_gpu_used_total_mb())
            time.sleep(0.25)

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()
    timeout_s = float(os.environ.get("CARNOT_5719_MODEL_TIMEOUT_S", "1800"))
    try:
        stdout, stderr = proc.communicate(json.dumps(worker_payload), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
    after_devices = _nvidia_smi_devices()
    after_mb = _gpu_used_total_mb()
    payload = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {"ok": False, "rows": []}
    template = str(payload.get("native_chat_template_receipt", {}).get("template_text") or "")
    peak_mb = max(samples or [before_mb])
    offloaded = parse_offloaded_layers(stderr)
    receipt = {
        "model_hf_id": model_spec["hf_id"],
        "llama_cpp_version": str(payload.get("llama_cpp_version") or ""),
        "llama_cpp_build_info": dict(payload.get("llama_cpp_build_info") or {}),
        "native_chat_template_receipt": {
            **dict(payload.get("native_chat_template_receipt") or {}),
            "template_hash": sha256_text(template) if template else "",
        },
        "cuda_device_receipt": {
            "before": devices_before,
            "peak": samples,
            "after": after_devices,
            "worker_returncode": proc.returncode,
            "worker_error": str(payload.get("error") or ""),
        },
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": offloaded,
        "gpu_memory_before_mb": before_mb,
        "gpu_memory_peak_mb": peak_mb,
        "gpu_memory_after_mb": after_mb,
        "cuda_offload_authenticated": bool(offloaded > 0 and peak_mb > before_mb),
        "offload_log_excerpt": stderr[-4000:],
        "rows": list(payload.get("rows") or []),
    }
    for row in receipt["rows"]:
        row["template_hash"] = receipt["native_chat_template_receipt"]["template_hash"]
    gc.collect()
    return receipt


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp5719 from the command line."""

    del argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
