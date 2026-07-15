"""Exp5708 raw-response local-SOTA exact-constraint canary.

Spec refs: REQ-VERIFY-5708, SCENARIO-VERIFY-5708.

This experiment is a readiness canary, not a model-quality claim. It asks one
mandated local GGUF model for raw text, stores exactly what came back, and then
lets deterministic validators label the rows outside generation. The useful
result is whether the runtime and data stream are replayable enough for FR-11
learner access; a wrong model answer is a valid labeled row, while a missing,
truncated, or unparsable answer blocks readiness.
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
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf


JsonDict = dict[str, Any]
GenerationRunner = Callable[[JsonDict, list[JsonDict], JsonDict, JsonDict], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5708_sota_exact_constraint_canary.json")
ROW_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_5708_sota_exact_constraint_canary.rows.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5708_sota_exact_constraint_canary.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5708_sota_exact_constraint_canary.py")

SCHEMA = "carnot.experiment_5708.sota_exact_constraint_canary.v1"
EXPERIMENT = 5708
EXPERIMENT_ID = "experiment_5708_sota_exact_constraint_canary"
MILESTONE = "2026.07.510"
RUN_DATE = "20260715"
MODEL_REPO_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MODEL_HEADLINE = MODEL_REPO_ID
MODEL_NAME = "Gemma4-26B-A4B-it"
INFERENCE_SUBSTRATE = "local_llama_cpp_python_cuda_gguf"
N_GPU_LAYERS_REQUESTED = -1
PANEL_ROWS_PER_FAMILY = 10
SHADOW_PREFIX_COUNT = 25
REQUIRED_FAMILIES = (
    "exact_finite_state",
    "arithmetic_finite_domain",
    "hard_soft_preference",
    "format_stress",
    "trapqa_shortcut",
)
SPEC_REFS = ("REQ-VERIFY-5708", "SCENARIO-VERIFY-5708")
GENERATION_CONFIG: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 32,
    "stop": ["\n"],
    "n_ctx": 2048,
    "n_batch": 256,
    "n_gpu_layers": N_GPU_LAYERS_REQUESTED,
    "native_json_grammar_used": False,
}
RANDOM_SEEDS: JsonDict = {
    "panel_seed": 5708001,
    "base_seed": 5708,
    "runner_seed": 5708002,
}
MODEL_SPECS: list[JsonDict] = [
    {
        "headline": MODEL_HEADLINE,
        "name": MODEL_NAME,
        "hf_id": MODEL_REPO_ID,
        "model_repo_id": MODEL_REPO_ID,
        "quantization": "Q4_K_M",
        "role": "moe",
        "headline_eligible": True,
    }
]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "MODEL_SPECS",
    "resolved_model_path",
    "model_repo_id",
    "gguf_filename",
    "model_hash",
    "quantization",
    "llama_cpp_version",
    "llama_cpp_build_info",
    "cuda_device_receipt",
    "n_gpu_layers_requested",
    "n_gpu_layers_offloaded",
    "gpu_memory_before_mb",
    "gpu_memory_peak_mb",
    "gpu_memory_after_mb",
    "cuda_offload_authenticated",
    "cuda_offload_authenticated_score",
    "generation_config",
    "random_seeds",
    "preregistered_panel",
    "family_counts",
    "row_manifest_path",
    "raw_response_hashes",
    "missing_row_count",
    "parse_failure_count",
    "exact_validator_versions",
    "validator_disagreement_count",
    "shadow_prefix_hash",
    "sealed_suffix_hash",
    "stream_root_commitment",
    "headline_model_count",
    "sota_canary_ready_score",
    "legacy_smoke_only",
    "native_json_grammar_used",
    "external_scorer_used",
    "inference_substrate",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Every gate field names the evidence boundary it protects.",
    "MODEL_SPECS": "The single headline model identity is explicit and cannot drift to another GGUF or legacy model.",
    "resolved_model_path": "The local GGUF path is auditable without downloading.",
    "model_repo_id": "The mandated Hugging Face repository is explicit.",
    "gguf_filename": "The exact local weight filename is visible.",
    "model_hash": "Weight bytes are bound to the artifact.",
    "quantization": "The observed GGUF quantization is not inferred loosely.",
    "llama_cpp_version": "The Python runtime can be reconstructed.",
    "llama_cpp_build_info": "CUDA build evidence is inspectable.",
    "cuda_device_receipt": "NVIDIA device and reservation evidence is preserved.",
    "n_gpu_layers_requested": "Offload intent is explicit.",
    "n_gpu_layers_offloaded": "Positive offload evidence is separate from intent.",
    "gpu_memory_before_mb": "CPU-only fallback cannot hide as a baseline.",
    "gpu_memory_peak_mb": "During-run GPU allocation is visible.",
    "gpu_memory_after_mb": "Cleanup evidence is visible.",
    "cuda_offload_authenticated": "The bare CUDA gate is explicit.",
    "cuda_offload_authenticated_score": "The CUDA gate scalar is mechanical.",
    "generation_config": "Raw-response decoding can be replayed.",
    "random_seeds": "Sampling replay is stable.",
    "preregistered_panel": "Row coverage is frozen before outcomes.",
    "family_counts": "Coverage denominators are visible.",
    "row_manifest_path": "Raw row evidence is lossless and replayable.",
    "raw_response_hashes": "Each model answer is byte-bound.",
    "missing_row_count": "Missing generations fail visibly.",
    "parse_failure_count": "Unusable generations fail visibly.",
    "exact_validator_versions": "Label authority is versioned.",
    "validator_disagreement_count": "Independent validator failures block.",
    "shadow_prefix_hash": "Learner-visible prefix is sealed.",
    "sealed_suffix_hash": "Unopened suffix is sealed.",
    "stream_root_commitment": "Chronology is immutable.",
    "headline_model_count": "The model denominator is honest.",
    "sota_canary_ready_score": "Readiness is a strict mechanical gate.",
    "legacy_smoke_only": "Legacy paths stay non-headline.",
    "native_json_grammar_used": "The retired grammar path stays closed.",
    "external_scorer_used": "No external judge or scorer can decide labels.",
    "inference_substrate": "Execution provenance is declared.",
    "reproducibility_checksum": "The artifact can be replayed.",
    "honest_verdict": "Terminal state starts complete: or blocked:.",
}

PRIMARY_VALIDATOR_VERSION = "exp5708_primary_exact_validators_v1"
SECONDARY_VALIDATOR_VERSION = "exp5708_secondary_enumeration_validators_v1"
ANSWER_RE = re.compile(r"(?im)^\s*ANSWER\s*:\s*(?P<answer>.+?)\s*$")
OFFLOAD_RE = re.compile(r"offloaded\s+(?P<offloaded>\d+)\s*/\s*(?P<total>\d+)\s+layers", re.I)
QUANT_RE = re.compile(
    r"(UD-)?(?:Q\d(?:_K_[A-Z]+|_[0-9A-Z]+)?|IQ\d_[A-Z]+|BF16|F16)",
    re.I,
)


class ManifestReplayError(ValueError):
    """Raised when a row manifest no longer matches its sealed hashes."""


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


def extract_quantization(filename: str) -> str:
    """Read the quantization suffix from a GGUF filename."""

    matches = list(QUANT_RE.finditer(filename))
    return matches[-1].group(0) if matches else "unknown"


def expected_answer_text(row: Mapping[str, Any]) -> str:
    """Return the exact expected answer token for fixture rows and prompts."""

    expected = row["expected"]
    if expected["kind"] == "abstain":
        return "ABSTAIN"
    return str(expected["answer"])


def _prompt_header() -> str:
    return (
        "Return exactly one line in this form: ANSWER: <value>. "
        "Use ANSWER: ABSTAIN when hard constraints conflict. Do not output JSON."
    )


def _finite_state_row(i: int) -> JsonDict:
    states = ("A", "B", "C", "D")
    alphabet = ("left", "right", "hold")
    transitions = {
        "A": {"left": "B", "right": "C", "hold": "A"},
        "B": {"left": "D", "right": "A", "hold": "B"},
        "C": {"left": "A", "right": "D", "hold": "C"},
        "D": {"left": "C", "right": "B", "hold": "D"},
    }
    symbols = [alphabet[(i + j) % len(alphabet)] for j in range(3 + (i % 2))]
    start = states[i % len(states)]
    contradiction = i in {3, 8}
    final = start
    for symbol in symbols:
        final = transitions[final][symbol]
    expected = {"kind": "abstain"} if contradiction else {"kind": "answer", "answer": final}
    rule = f"Start at {start}; apply symbols {', '.join(symbols)}."
    if contradiction:
        rule += " Hard evidence also says the final state is both B and C."
    prompt = f"{_prompt_header()} Exact finite-state task. {rule}"
    return _row(
        row_id=f"efs-{i:02d}",
        family="exact_finite_state",
        prompt=prompt,
        expected=expected,
        payload={
            "start": start,
            "symbols": symbols,
            "transitions": transitions,
            "contradiction": contradiction,
        },
        shift=i in {5, 6},
        contradiction=contradiction,
    )


def _arithmetic_row(i: int) -> JsonDict:
    modulus = 7 + (i % 4)
    domain = list(range(modulus))
    a = 2 + (i % 3)
    b = 1 + (i % 5)
    target = (a * ((i + 2) % modulus) + b) % modulus
    contradiction = i in {2, 9}
    solutions = [x for x in domain if (a * x + b) % modulus == target]
    answer = solutions[0] if len(solutions) == 1 and not contradiction else None
    expected = {"kind": "abstain"} if answer is None else {"kind": "answer", "answer": answer}
    prompt = (
        f"{_prompt_header()} Arithmetic finite-domain task. "
        f"Find x in {domain} such that ({a}*x+{b}) mod {modulus} = {target}."
    )
    if contradiction:
        prompt += " A hard note additionally requires x to be outside the listed domain."
    return _row(
        row_id=f"afd-{i:02d}",
        family="arithmetic_finite_domain",
        prompt=prompt,
        expected=expected,
        payload={
            "domain": domain,
            "a": a,
            "b": b,
            "modulus": modulus,
            "target": target,
            "contradiction": contradiction,
        },
        shift=i in {4, 5},
        contradiction=contradiction,
    )


def _hard_soft_row(i: int) -> JsonDict:
    names = [f"plan-{i}-{suffix}" for suffix in ("A", "B", "C")]
    candidates = []
    for j, name in enumerate(names):
        hard_ok = not (i in {1, 7} and j != 2) and not (i == 8)
        soft_score = ((i + 3 * j) % 7) + j
        candidates.append({"name": name, "hard_ok": hard_ok, "soft_score": soft_score})
    feasible = [row for row in candidates if row["hard_ok"]]
    if feasible:
        best = sorted(feasible, key=lambda row: (-int(row["soft_score"]), str(row["name"])))[0]
        expected = {"kind": "answer", "answer": best["name"]}
    else:
        expected = {"kind": "abstain"}
    prompt = (
        f"{_prompt_header()} Hard-soft preference task. Hard-valid candidates and "
        f"soft scores are: {candidates}. Choose the hard-valid candidate with the "
        "largest soft score."
    )
    return _row(
        row_id=f"hsp-{i:02d}",
        family="hard_soft_preference",
        prompt=prompt,
        expected=expected,
        payload={"candidates": candidates},
        shift=i in {6, 7},
        contradiction=not feasible,
    )


def _format_row(i: int) -> JsonDict:
    target = f"K{i:02d}-ALPHA/{(i * 7) % 13:02d}"
    contradiction = i in {0, 6}
    expected = {"kind": "abstain"} if contradiction else {"kind": "answer", "answer": target}
    prompt = (
        f"{_prompt_header()} Format stress task. Copy the exact token after "
        f"FINAL_TOKEN, preserving case and punctuation. FINAL_TOKEN={target}."
    )
    if contradiction:
        prompt += f" Hard evidence also says FINAL_TOKEN=K{i:02d}-BETA/99."
    return _row(
        row_id=f"fmt-{i:02d}",
        family="format_stress",
        prompt=prompt,
        expected=expected,
        payload={"target": target, "contradiction": contradiction},
        shift=i in {3, 4},
        contradiction=contradiction,
    )


def _trapqa_row(i: int) -> JsonDict:
    shortcuts = ("Paris", "blue", "42", "largest", "left")
    evidences = ("Lyon", "green", "17", "smallest", "right")
    shortcut = shortcuts[i % len(shortcuts)]
    evidence = evidences[i % len(evidences)]
    contradiction = i in {4, 9}
    expected = {"kind": "abstain"} if contradiction else {"kind": "answer", "answer": evidence}
    prompt = (
        f"{_prompt_header()} TrapQA shortcut task. A salient shortcut suggests "
        f"{shortcut}, but exact evidence says the answer is {evidence}. Use exact evidence."
    )
    if contradiction:
        prompt += f" A second exact evidence line says the answer is not {evidence}."
    return _row(
        row_id=f"trap-{i:02d}",
        family="trapqa_shortcut",
        prompt=prompt,
        expected=expected,
        payload={"shortcut": shortcut, "evidence": evidence, "contradiction": contradiction},
        shift=i in {1, 2},
        contradiction=contradiction,
    )


def _row(
    *,
    row_id: str,
    family: str,
    prompt: str,
    expected: Mapping[str, Any],
    payload: Mapping[str, Any],
    shift: bool,
    contradiction: bool,
) -> JsonDict:
    index = int(row_id.rsplit("-", 1)[-1])
    return {
        "row_id": row_id,
        "family": family,
        "timestamp_utc": f"2026-07-15T00:{index:02d}:00Z",
        "prompt": prompt,
        "prompt_hash": sha256_text(prompt),
        "expected": dict(expected),
        "validator_payload": json.loads(canonical_json(payload)),
        "control_tags": {
            "answer": expected["kind"] == "answer",
            "abstention": expected["kind"] == "abstain",
            "shift": bool(shift),
            "contradiction": bool(contradiction),
        },
    }


def freeze_preregistered_panel() -> list[JsonDict]:
    """Return the fixed balanced panel before any model outcome is read."""

    builders = (
        _finite_state_row,
        _arithmetic_row,
        _hard_soft_row,
        _format_row,
        _trapqa_row,
    )
    rows_by_family = [[builder(i) for i in range(PANEL_ROWS_PER_FAMILY)] for builder in builders]
    interleaved: list[JsonDict] = []
    for i in range(PANEL_ROWS_PER_FAMILY):
        for family_rows in rows_by_family:
            interleaved.append(family_rows[i])
    return interleaved


def family_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count preregistered rows per required family."""

    return dict(Counter(str(row["family"]) for row in rows))


def preregistered_panel(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose row commitments without model outcomes."""

    compact = []
    for sequence_index, row in enumerate(rows):
        preimage = {
            "row_id": row["row_id"],
            "family": row["family"],
            "timestamp_utc": row["timestamp_utc"],
            "prompt": row["prompt"],
            "expected": row["expected"],
            "validator_payload": row["validator_payload"],
            "control_tags": row["control_tags"],
        }
        compact.append(
            {
                "sequence_index": sequence_index,
                "row_id": row["row_id"],
                "family": row["family"],
                "timestamp_utc": row["timestamp_utc"],
                "prompt_hash": row["prompt_hash"],
                "pre_outcome_hash": sha256_json(preimage),
                "control_tags": dict(row["control_tags"]),
            }
        )
    return compact


def commitment_hashes(panel_commitments: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Seal the shadow prefix and unopened suffix in chronological order."""

    prefix = list(panel_commitments[:SHADOW_PREFIX_COUNT])
    suffix = list(panel_commitments[SHADOW_PREFIX_COUNT:])
    shadow_prefix_hash = sha256_json(prefix)
    sealed_suffix_hash = sha256_json(suffix)
    stream_root_commitment = sha256_json(
        {
            "schema": SCHEMA,
            "model_repo_id": MODEL_REPO_ID,
            "split_index": SHADOW_PREFIX_COUNT,
            "panel_hash": sha256_json(list(panel_commitments)),
            "shadow_prefix_hash": shadow_prefix_hash,
            "sealed_suffix_hash": sealed_suffix_hash,
        }
    )
    return {
        "shadow_prefix_hash": shadow_prefix_hash,
        "sealed_suffix_hash": sealed_suffix_hash,
        "stream_root_commitment": stream_root_commitment,
    }


def parse_raw_answer(raw_text: str) -> JsonDict:
    """Extract the required raw-response answer line without using JSON grammar."""

    match = ANSWER_RE.search(raw_text)
    if match is None:
        return {"parse_ok": False, "answer": "", "error": "missing_answer_line"}
    answer = match.group("answer").strip()
    if not answer:
        return {"parse_ok": False, "answer": "", "error": "empty_answer"}
    return {"parse_ok": True, "answer": answer, "error": ""}


def _expected_by_primary(row: Mapping[str, Any]) -> str:
    family = row["family"]
    payload = row["validator_payload"]
    if family == "exact_finite_state":
        if payload["contradiction"]:
            return "ABSTAIN"
        state = str(payload["start"])
        for symbol in payload["symbols"]:
            state = payload["transitions"][state][symbol]
        return state
    if family == "arithmetic_finite_domain":
        if payload["contradiction"]:
            return "ABSTAIN"
        solutions = [
            x
            for x in payload["domain"]
            if (int(payload["a"]) * int(x) + int(payload["b"])) % int(payload["modulus"])
            == int(payload["target"])
        ]
        return str(solutions[0]) if len(solutions) == 1 else "ABSTAIN"
    if family == "hard_soft_preference":
        feasible = [candidate for candidate in payload["candidates"] if candidate["hard_ok"]]
        if not feasible:
            return "ABSTAIN"
        best = sorted(feasible, key=lambda row: (-int(row["soft_score"]), str(row["name"])))[0]
        return str(best["name"])
    if family == "format_stress":
        return "ABSTAIN" if payload["contradiction"] else str(payload["target"])
    if family == "trapqa_shortcut":
        return "ABSTAIN" if payload["contradiction"] else str(payload["evidence"])
    raise ValueError(f"unknown family: {family}")


def _expected_by_secondary(row: Mapping[str, Any]) -> str:
    family = row["family"]
    payload = row["validator_payload"]
    if family == "hard_soft_preference":
        best_name = None
        best_score = None
        for candidate in payload["candidates"]:
            if not candidate["hard_ok"]:
                continue
            score = int(candidate["soft_score"])
            name = str(candidate["name"])
            if best_score is None or score > best_score or (score == best_score and name < best_name):
                best_name = name
                best_score = score
        return best_name if best_name is not None else "ABSTAIN"
    if row["expected"]["kind"] == "abstain":
        return "ABSTAIN"
    return str(row["expected"]["answer"])


def primary_validate_row(row: Mapping[str, Any], parsed: Mapping[str, Any]) -> JsonDict:
    """Label one row using the primary exact validator implementation."""

    if parsed.get("parse_ok") is not True:
        return {
            "validator_version": PRIMARY_VALIDATOR_VERSION,
            "parse_ok": False,
            "label": False,
            "expected_answer": expected_answer_text(row),
            "observed_answer": "",
            "error": str(parsed.get("error") or "parse_failure"),
        }
    expected = _expected_by_primary(row)
    observed = str(parsed["answer"])
    return {
        "validator_version": PRIMARY_VALIDATOR_VERSION,
        "parse_ok": True,
        "label": observed == expected,
        "expected_answer": expected,
        "observed_answer": observed,
        "error": "",
    }


def secondary_validate_row(row: Mapping[str, Any], parsed: Mapping[str, Any]) -> JsonDict:
    """Double-check labels with a second enumeration-style implementation."""

    if parsed.get("parse_ok") is not True:
        return {
            "validator_version": SECONDARY_VALIDATOR_VERSION,
            "parse_ok": False,
            "label": False,
            "expected_answer": expected_answer_text(row),
            "observed_answer": "",
            "error": str(parsed.get("error") or "parse_failure"),
        }
    expected = _expected_by_secondary(row)
    observed = str(parsed["answer"])
    return {
        "validator_version": SECONDARY_VALIDATOR_VERSION,
        "parse_ok": True,
        "label": observed == expected,
        "expected_answer": expected,
        "observed_answer": observed,
        "error": "",
    }


def normalize_model_spec(resolved_model_path: Path | str | None = None) -> JsonDict:
    """Resolve and hash the mandated Gemma GGUF without using transformers."""

    registry = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
    registry_row = registry.get(MODEL_REPO_ID, {})
    resolved = str(resolved_model_path or resolve_cached_gguf(MODEL_REPO_ID, "Q4_K_M") or "")
    path = Path(resolved).expanduser() if resolved else Path()
    present = bool(resolved and path.is_file())
    filename = path.name if resolved else ""
    return {
        **MODEL_SPECS[0],
        "name": str(registry_row.get("name") or MODEL_NAME),
        "role": str(registry_row.get("role") or "moe"),
        "active_params_b": registry_row.get("active_params_b", 4.0),
        "total_params_b": registry_row.get("total_params_b", 26.0),
        "resolved_model_path": resolved,
        "model_path": resolved,
        "gguf_filename": filename,
        "model_hash": sha256_file(path) if present else "",
        "model_size_bytes": path.stat().st_size if present else 0,
        "quantization": extract_quantization(filename) if filename else "unknown",
        "local_model_present": present,
    }


def _runtime_row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {str(row.get("row_id")): dict(row) for row in rows if row.get("row_id")}


def build_manifest_rows(
    *,
    panel: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> list[JsonDict]:
    """Join preregistered rows with raw model outputs and exact labels."""

    raw_by_id = _runtime_row_map(runtime_receipt.get("rows", []))
    rows: list[JsonDict] = []
    previous_hash = ""
    compact_panel = preregistered_panel(panel)
    compact_by_id = {row["row_id"]: row for row in compact_panel}
    for sequence_index, row in enumerate(panel):
        row_id = str(row["row_id"])
        raw = raw_by_id.get(row_id)
        missing = raw is None
        raw_text = "" if raw is None else str(raw.get("raw_text", ""))
        finish_reason = "missing" if raw is None else str(raw.get("finish_reason", ""))
        truncated = finish_reason.lower() in {"length", "max_tokens", "truncated"}
        parsed = parse_raw_answer(raw_text)
        if missing:
            parsed = {"parse_ok": False, "answer": "", "error": "missing_generation"}
        elif truncated:
            parsed = {"parse_ok": False, "answer": parsed.get("answer", ""), "error": "truncated"}
        primary = primary_validate_row(row, parsed)
        secondary = secondary_validate_row(row, parsed)
        disagreement = (
            primary["label"] != secondary["label"]
            or primary["expected_answer"] != secondary["expected_answer"]
        )
        manifest_row: JsonDict = {
            "schema": SCHEMA + ".manifest",
            "sequence_index": sequence_index,
            "row_id": row_id,
            "family": row["family"],
            "timestamp_utc": row["timestamp_utc"],
            "prompt": row["prompt"],
            "prompt_hash": row["prompt_hash"],
            "pre_outcome_hash": compact_by_id[row_id]["pre_outcome_hash"],
            "model_repo_id": MODEL_REPO_ID,
            "gguf_filename": model_spec.get("gguf_filename", ""),
            "model_hash": model_spec.get("model_hash", ""),
            "raw_text": raw_text,
            "raw_response_hash": sha256_text(raw_text),
            "finish_reason": finish_reason,
            "truncated": truncated,
            "missing_generation": missing,
            "parse_ok": parsed["parse_ok"],
            "parse_error": parsed["error"],
            "parsed_answer": parsed.get("answer", ""),
            "primary_validation": primary,
            "secondary_validation": secondary,
            "validator_disagreement": disagreement,
            "token_counts": {} if raw is None else dict(raw.get("token_counts") or {}),
            "timing": {} if raw is None else dict(raw.get("timing") or {}),
            "seed": None if raw is None else raw.get("seed"),
            "generation_config": {} if raw is None else dict(raw.get("generation_config") or {}),
            "telemetry": {} if raw is None else dict(raw.get("telemetry") or {}),
            "previous_row_hash": previous_hash,
            "row_hash": "",
        }
        manifest_row["row_hash"] = manifest_row_hash(manifest_row)
        previous_hash = manifest_row["row_hash"]
        rows.append(manifest_row)
    return rows


def manifest_row_hash(row: Mapping[str, Any]) -> str:
    """Hash a manifest row while excluding its own hash field."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def write_manifest_rows(rows: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    """Write row evidence as append-only JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_manifest_rows(path: Path | str) -> list[JsonDict]:
    """Read a JSONL row manifest from disk."""

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def verify_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
) -> bool:
    """Replay row hashes, raw hashes, and the manifest chain."""

    expected_hashes = dict(artifact.get("raw_response_hashes") or {})
    previous = ""
    for row in rows:
        if row.get("previous_row_hash") != previous:
            raise ManifestReplayError("previous_row_hash")
        raw_hash = sha256_text(str(row.get("raw_text", "")))
        if raw_hash != row.get("raw_response_hash"):
            raise ManifestReplayError("raw_response_hash")
        if expected_hashes.get(str(row.get("row_id"))) != raw_hash:
            raise ManifestReplayError("raw_response_hash")
        if manifest_row_hash(row) != row.get("row_hash"):
            raise ManifestReplayError("row_hash")
        previous = str(row["row_hash"])
    return True


def verify_commitments(
    artifact: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Verify chronology commitments from artifact fields and manifest order."""

    panel = list(artifact.get("preregistered_panel") or [])
    hashes = commitment_hashes(panel)
    for field, value in hashes.items():
        if artifact.get(field) != value:
            raise ValueError(field)
    row_ids = [row["row_id"] for row in panel]
    manifest_ids = [row["row_id"] for row in manifest_rows]
    if row_ids != manifest_ids:
        raise ValueError("manifest_order")
    for panel_row, manifest_row in zip(panel, manifest_rows, strict=True):
        if panel_row["pre_outcome_hash"] != manifest_row["pre_outcome_hash"]:
            raise ValueError("pre_outcome_hash")
    return True


def raw_response_hashes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return row-id to raw-response hash mapping."""

    return {str(row["row_id"]): str(row["raw_response_hash"]) for row in rows}


def _runtime_cuda_authenticated(receipt: Mapping[str, Any]) -> bool:
    return bool(
        receipt.get("cuda_offload_authenticated") is True
        and int(receipt.get("n_gpu_layers_offloaded") or 0) > 0
        and int(receipt.get("gpu_memory_peak_mb") or 0) > int(receipt.get("gpu_memory_before_mb") or 0)
    )


def cuda_offload_authenticated_score(artifact_or_receipt: Mapping[str, Any]) -> float:
    """Return the mechanical CUDA scalar required by the artifact contract."""

    return 1.0 if _runtime_cuda_authenticated(artifact_or_receipt) else 0.0


def sota_canary_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when every data/runtime readiness gate is clean."""

    families = dict(artifact.get("family_counts") or {})
    family_gate = all(int(families.get(family, 0)) > 0 for family in REQUIRED_FAMILIES)
    ready = bool(
        artifact.get("cuda_offload_authenticated") is True
        and artifact.get("cuda_offload_authenticated_score") == 1.0
        and family_gate
        and int(artifact.get("headline_model_count") or 0) == 1
        and int(artifact.get("missing_row_count") or 0) == 0
        and int(artifact.get("parse_failure_count") or 0) == 0
        and int(artifact.get("validator_disagreement_count") or 0) == 0
        and artifact.get("commitments_verified") is True
        and artifact.get("legacy_smoke_only") is True
        and artifact.get("native_json_grammar_used") is False
        and artifact.get("external_scorer_used") is False
        and artifact.get("retired_runtime_used") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and len(artifact.get("raw_response_hashes") or {}) >= 48
    )
    return 1.0 if ready else 0.0


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if artifact.get("cuda_offload_authenticated_score") != 1.0:
        reasons.append("cuda_offload_unauthenticated")
    if int(artifact.get("missing_row_count") or 0) > 0:
        reasons.append("missing_rows")
    if int(artifact.get("parse_failure_count") or 0) > 0:
        reasons.append("parse_failures")
    if int(artifact.get("validator_disagreement_count") or 0) > 0:
        reasons.append("validator_disagreement")
    if artifact.get("commitments_verified") is not True:
        reasons.append("commitments_unverified")
    if artifact.get("native_json_grammar_used") is not False:
        reasons.append("native_json_grammar_used")
    if artifact.get("external_scorer_used") is not False:
        reasons.append("external_scorer_used")
    if artifact.get("retired_runtime_used") is not False:
        reasons.append("retired_runtime_used")
    return reasons or ["readiness_gate_not_met"]


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict string from mechanical gates."""

    if float(artifact.get("sota_canary_ready_score") or 0.0) == 1.0:
        return "complete: sota_exact_constraint_canary_ready"
    return "blocked: " + ",".join(_blocked_reasons(artifact))


def build_artifact(
    *,
    model_spec: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    row_manifest_path: str,
    tests_added_or_reused: Sequence[str] = (),
) -> JsonDict:
    """Build the terminal Exp5708 artifact from sealed row evidence."""

    panel = freeze_preregistered_panel()
    compact_panel = preregistered_panel(panel)
    commitments = commitment_hashes(compact_panel)
    counts = family_counts(panel)
    missing_count = sum(1 for row in manifest_rows if row.get("missing_generation") is True)
    parse_failure_count = sum(1 for row in manifest_rows if row.get("parse_ok") is not True)
    disagreement_count = sum(1 for row in manifest_rows if row.get("validator_disagreement") is True)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "MODEL_SPECS": [dict(model_spec)],
        "resolved_model_path": str(model_spec.get("resolved_model_path") or ""),
        "model_repo_id": MODEL_REPO_ID,
        "gguf_filename": str(model_spec.get("gguf_filename") or ""),
        "model_hash": str(model_spec.get("model_hash") or ""),
        "quantization": str(model_spec.get("quantization") or "unknown"),
        "llama_cpp_version": str(runtime_receipt.get("llama_cpp_version") or ""),
        "llama_cpp_build_info": dict(runtime_receipt.get("llama_cpp_build_info") or {}),
        "cuda_device_receipt": dict(runtime_receipt.get("cuda_device_receipt") or {}),
        "n_gpu_layers_requested": int(runtime_receipt.get("n_gpu_layers_requested") or 0),
        "n_gpu_layers_offloaded": int(runtime_receipt.get("n_gpu_layers_offloaded") or 0),
        "gpu_memory_before_mb": int(runtime_receipt.get("gpu_memory_before_mb") or 0),
        "gpu_memory_peak_mb": int(runtime_receipt.get("gpu_memory_peak_mb") or 0),
        "gpu_memory_after_mb": int(runtime_receipt.get("gpu_memory_after_mb") or 0),
        "cuda_offload_authenticated": _runtime_cuda_authenticated(runtime_receipt),
        "cuda_offload_authenticated_score": 0.0,
        "generation_config": dict(GENERATION_CONFIG),
        "random_seeds": dict(RANDOM_SEEDS),
        "preregistered_panel": compact_panel,
        "family_counts": counts,
        "row_manifest_path": row_manifest_path,
        "raw_response_hashes": raw_response_hashes(manifest_rows),
        "missing_row_count": missing_count,
        "parse_failure_count": parse_failure_count,
        "exact_validator_versions": {
            "primary": PRIMARY_VALIDATOR_VERSION,
            "secondary": SECONDARY_VALIDATOR_VERSION,
            "secondary_double_check_rows": len(manifest_rows),
        },
        "validator_disagreement_count": disagreement_count,
        "shadow_prefix_hash": commitments["shadow_prefix_hash"],
        "sealed_suffix_hash": commitments["sealed_suffix_hash"],
        "stream_root_commitment": commitments["stream_root_commitment"],
        "headline_model_count": 1,
        "sota_canary_ready_score": 0.0,
        "legacy_smoke_only": True,
        "native_json_grammar_used": False,
        "external_scorer_used": False,
        "retired_runtime_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "manifest_row_count": len(manifest_rows),
        "commitments_verified": True,
        "offload_log_excerpt": str(runtime_receipt.get("offload_log_excerpt") or "")[-2000:],
        "tests_added_or_reused": list(tests_added_or_reused),
    }
    artifact["cuda_offload_authenticated_score"] = cuda_offload_authenticated_score(artifact)
    artifact["sota_canary_ready_score"] = sota_canary_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["blocked_reasons"] = _blocked_reasons(artifact) if artifact["sota_canary_ready_score"] == 0.0 else []
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
    if artifact.get("model_repo_id") != MODEL_REPO_ID:
        raise ValueError("model_repo_id")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("headline_model_count") != 1:
        raise ValueError("headline_model_count")
    if artifact.get("legacy_smoke_only") is not True:
        raise ValueError("legacy_smoke_only")
    if artifact.get("native_json_grammar_used") is not False:
        raise ValueError("native_json_grammar_used")
    if artifact.get("external_scorer_used") is not False:
        raise ValueError("external_scorer_used")
    expected_cuda_score = cuda_offload_authenticated_score(artifact)
    if artifact.get("cuda_offload_authenticated_score") != expected_cuda_score:
        raise ValueError("cuda_offload_authenticated_score")
    expected_commitments = commitment_hashes(artifact.get("preregistered_panel") or [])
    for field, value in expected_commitments.items():
        if artifact.get(field) != value:
            raise ValueError(field)
    expected_score = sota_canary_ready_score(artifact)
    if artifact.get("sota_canary_ready_score") != expected_score:
        raise ValueError("sota_canary_ready_score")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_score == 1.0 and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if expected_score == 0.0 and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _blocked_runtime_receipt(reason: str) -> JsonDict:
    return {
        "llama_cpp_version": "",
        "llama_cpp_build_info": {"blocked_reason": reason},
        "cuda_device_receipt": {"devices": []},
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": 0,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 0,
        "gpu_memory_after_mb": 0,
        "cuda_offload_authenticated": False,
        "offload_log_excerpt": "",
        "rows": [],
    }


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_manifest_path: Path | str = REPO_ROOT / ROW_MANIFEST_RELATIVE_PATH,
    resolved_model_path: Path | str | None = None,
    generation_runner: GenerationRunner = None,
    tests_added_or_reused: Sequence[str] = (),
    write: bool = True,
) -> JsonDict:
    """Run the canary or write an honest blocked artifact."""

    runner = generation_runner or default_generation_runner
    model_spec = normalize_model_spec(resolved_model_path)
    panel = freeze_preregistered_panel()
    if model_spec["local_model_present"] is not True:
        runtime_receipt = _blocked_runtime_receipt("mandated_gguf_missing")
    else:
        runtime_receipt = runner(model_spec, panel, dict(GENERATION_CONFIG), dict(RANDOM_SEEDS))
    manifest_rows = build_manifest_rows(
        panel=panel,
        runtime_receipt=runtime_receipt,
        model_spec=model_spec,
    )
    artifact = build_artifact(
        model_spec=model_spec,
        runtime_receipt=runtime_receipt,
        manifest_rows=manifest_rows,
        row_manifest_path=str(Path(row_manifest_path)),
        tests_added_or_reused=tests_added_or_reused,
    )
    if write:
        write_manifest_rows(manifest_rows, row_manifest_path)
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
    devices = _nvidia_smi_devices()
    return sum(int(row.get("memory_used_mb", 0) or 0) for row in devices)


def _ram_available_mb() -> int:  # pragma: no cover - host dependent.
    try:
        info = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    match = re.search(r"MemAvailable:\s+(\d+)\s+kB", info)
    return int(int(match.group(1)) / 1024) if match else 0


def default_generation_runner(
    model_spec: JsonDict,
    panel: list[JsonDict],
    generation_config: JsonDict,
    random_seeds: JsonDict,
) -> JsonDict:  # pragma: no cover - host dependent live path.
    """Run the live llama-cpp-python raw-response stream in a child process."""

    devices_before = _nvidia_smi_devices()
    before_mb = _gpu_used_total_mb()
    worker_payload = {
        "model_path": model_spec["resolved_model_path"],
        "panel": panel,
        "generation_config": generation_config,
        "random_seeds": random_seeds,
    }
    worker_code = r'''
import gc
import importlib.metadata
import json
import time
import sys

payload = json.load(sys.stdin)
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
    config = payload["generation_config"]
    llm = Llama(
        model_path=payload["model_path"],
        n_gpu_layers=int(config["n_gpu_layers"]),
        n_ctx=int(config["n_ctx"]),
        n_batch=int(config["n_batch"]),
        seed=int(payload["random_seeds"]["runner_seed"]),
        verbose=True,
    )
    rows = []
    for index, row in enumerate(payload["panel"]):
        start = time.perf_counter()
        result = llm(
            row["prompt"],
            max_tokens=int(config["max_tokens"]),
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            stop=list(config["stop"]),
            echo=False,
        )
        elapsed = time.perf_counter() - start
        choice = result.get("choices", [{}])[0]
        text = str(choice.get("text", ""))
        usage = result.get("usage", {})
        rows.append({
            "row_id": row["row_id"],
            "prompt": row["prompt"],
            "raw_text": text,
            "finish_reason": str(choice.get("finish_reason", "")),
            "token_counts": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            "timing": {"load_s": 0.0, "generation_s": round(elapsed, 6)},
            "seed": int(payload["random_seeds"]["base_seed"]) + index,
            "generation_config": config,
            "model_hash": "",
            "telemetry": {},
        })
    del llm
    gc.collect()
    print(json.dumps({
        "ok": True,
        "llama_cpp_version": version,
        "llama_cpp_build_info": {
            "cuda_backend": "CUDA" in system_info.upper() or "GGML_CUDA" in system_info.upper(),
            "system_info": system_info,
            "module": getattr(llama_cpp, "__file__", ""),
        },
        "rows": rows,
    }, sort_keys=True))
except Exception as exc:
    print(json.dumps({"ok": False, "error": repr(exc), "rows": []}, sort_keys=True))
    raise
'''
    peak_mb = before_mb
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
        )
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(worker_payload))
        proc.stdin.close()
        start = time.time()
        stdout = ""
        try:
            while proc.poll() is None:
                peak_mb = max(peak_mb, _gpu_used_total_mb())
                if time.time() - start > 1800:
                    proc.kill()
                    raise TimeoutError("llama_cpp_python_canary_timeout")
                time.sleep(0.5)
            stdout = proc.stdout.read() if proc.stdout is not None else ""
        finally:
            if proc.poll() is None:
                proc.kill()
            stderr_file.seek(0)
            stderr_text = stderr_file.read()
    after_mb = _gpu_used_total_mb()
    devices_after = _nvidia_smi_devices()
    try:
        worker = json.loads(stdout.splitlines()[-1])
    except Exception as exc:
        worker = {"ok": False, "error": f"worker_json_parse_failed: {exc}", "rows": []}
    offloaded = parse_offloaded_layers(stderr_text)
    build_info = dict(worker.get("llama_cpp_build_info") or {})
    receipt = {
        "llama_cpp_version": str(worker.get("llama_cpp_version") or ""),
        "llama_cpp_build_info": build_info,
        "cuda_device_receipt": {
            "devices": devices_before,
            "devices_after": devices_after,
            "ram_reservation_mb": _ram_available_mb(),
            "vram_reservation_mb": max(0, int(model_spec.get("model_size_bytes", 0) / 1024 / 1024)),
            "worker_returncode": proc.returncode,
            "worker_error": worker.get("error", ""),
        },
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": offloaded,
        "gpu_memory_before_mb": before_mb,
        "gpu_memory_peak_mb": peak_mb,
        "gpu_memory_after_mb": after_mb,
        "cuda_offload_authenticated": bool(offloaded > 0 and peak_mb > before_mb),
        "offload_log_excerpt": stderr_text[-4000:],
        "rows": list(worker.get("rows") or []),
    }
    gc.collect()
    return receipt


def main() -> int:  # pragma: no cover - CLI wrapper.
    run(write=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
