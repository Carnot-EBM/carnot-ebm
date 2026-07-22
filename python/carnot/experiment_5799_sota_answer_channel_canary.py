"""Exp5799 matched SOTA answer-channel canary.

Spec refs: REQ-VERIFY-5799, SCENARIO-VERIFY-5799,
SCENARIO-VERIFY-5799-CONTROLS.

The canary spends a bounded amount of local GGUF runtime to answer one narrow
question: which per-model transport, if any, preserves the finite-choice answer
contract discovered by Exp5798.  It does not change the semantic task.  The
only accepted semantic surface remains a strict Exp5785 row-id plus candidate
label, and exact validators remain separate from parser failures.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

from carnot import experiment_5785_hardness_surface_fixture as fixture
from carnot import experiment_5786_sota_constraint_stream as stream
from carnot import experiment_5798_sota_answer_channel_diagnostic as diagnostic
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
EmitResponse = Callable[[Mapping[str, Any]], None]
CanaryRunner = Callable[[JsonDict, JsonDict, list[JsonDict], EmitResponse], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5799_sota_answer_channel_canary.json")
ROW_FILE_RELATIVE_PATH = Path("results/experiment_5799_sota_answer_channel_canary.rows.jsonl")
EXP5785_ARTIFACT_RELATIVE_PATH = fixture.RESULT_RELATIVE_PATH
EXP5785_ROWS_RELATIVE_PATH = fixture.ROW_FILE_RELATIVE_PATH
EXP5798_ARTIFACT_RELATIVE_PATH = diagnostic.RESULT_RELATIVE_PATH
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5799_sota_answer_channel_canary.py")

SCHEMA = "carnot.experiment_5799.sota_answer_channel_canary.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 5799
EXPERIMENT_ID = "experiment_5799_sota_answer_channel_canary"
MILESTONE = "2026.07.517"
RUN_DATE = "20260722"
INFERENCE_SUBSTRATE = "real_local_llama_cpp_cuda_gguf_generation_plus_exact_validation"
SEMANTIC_CONTRACT_ID = "strict_exp5785_row_id_candidate_label_line_v1"
EXACT_PARSER_ID = "exp5785_row_id_to_candidate_label_exact_parser"

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_MODEL_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)
DEFAULT_MAX_TOKENS = 128
DEFAULT_REASONING_BUDGET_TOKENS = 96
N_GPU_LAYERS_REQUESTED = -1
STOP_STRINGS = ("<|eot_id|>", "<stop>")
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5799,
    "runner_seed": 5799001,
    "fixture_seed": 5799002,
}
SAMPLING_CONFIG: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 40,
}
GENERATION_CONFIG: JsonDict = {
    "prompt_date": RUN_DATE,
    "temperature": SAMPLING_CONFIG["temperature"],
    "top_p": SAMPLING_CONFIG["top_p"],
    "top_k": SAMPLING_CONFIG["top_k"],
    "max_tokens": DEFAULT_MAX_TOKENS,
    "reasoning_budget_tokens": DEFAULT_REASONING_BUDGET_TOKENS,
    "n_ctx": 2048,
    "n_batch": 256,
    "n_gpu_layers": N_GPU_LAYERS_REQUESTED,
    "stop": list(STOP_STRINGS),
    "seed": RANDOM_SEEDS["runner_seed"],
    "semantic_contract_id": SEMANTIC_CONTRACT_ID,
    "parser": EXACT_PARSER_ID,
}
SPEC_REFS = (
    "REQ-VERIFY-5799",
    "SCENARIO-VERIFY-5799",
    "SCENARIO-VERIFY-5799-CONTROLS",
)
PRODUCER_GATE_FIELDS = (
    "qualified_real_sota_model_count",
    "answer_channel_ready_score",
    "raw_final_content_coverage",
    "exact_label_coverage",
    "parser_failure_rate",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "MODEL_SPECS",
    "models_used",
    "model_runtime_receipts",
    "gpu_offload_receipts",
    "embedded_template_receipts",
    "canary_fixture_hash",
    "independent_unit_count",
    "sample_size_justification",
    "mode_execution_matrix",
    "selected_transport_by_model",
    "raw_reasoning_coverage",
    "raw_final_content_coverage",
    "exact_label_coverage",
    "parser_failure_rate",
    "truncation_rate",
    "empty_final_content_rate",
    "invalid_candidate_rate",
    "exact_answer_error_rate",
    "protected_fact_distortion_count",
    "adversarial_control_results",
    "verified_outputs_per_second",
    "verified_outputs_per_token",
    "wasted_token_count",
    "checkpoint_resume_receipts",
    "qualified_real_sota_model_count",
    "answer_channel_ready_score",
    "producer_gate_fields",
    "row_file",
    "row_file_sha256",
    "inference_substrate",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5799_sota_answer_channel_canary.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5799_sota_answer_channel_canary.py -m pytest tests/python/test_experiment_5799_sota_answer_channel_canary.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5799_sota_answer_channel_canary.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

_REGISTRY = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
MODEL_SPECS: list[JsonDict] = []
for _index, _hf_id in enumerate(MANDATED_MODEL_IDS):
    _base = dict(_REGISTRY[_hf_id])
    MODEL_SPECS.append(
        {
            "name": _base["name"],
            "hf_id": _hf_id,
            "model_repo_id": _hf_id,
            "family": diagnostic.model_family(_hf_id),
            "role": _base["role"],
            "active_params_b": _base["active_params_b"],
            "total_params_b": _base["total_params_b"],
            "quantization": _base["quantization"],
            "min_vram_gb": _base["min_vram_gb"],
            "gpu": _index % 2,
            "headline_eligible": True,
            "legacy_smoke_only": False,
            "sequence_index": _index,
        }
    )


class ManifestReplayError(ValueError):
    """Raised when canary rows no longer match the sealed receipts."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks so large GGUF files remain streamable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _extract_quantization(filename: str, fallback: str) -> str:
    return stream._extract_quantization(filename, fallback)


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]] | None = None) -> list[JsonDict]:
    """Resolve and enrich the exact three mandated local SOTA GGUF specs."""

    overrides = {str(row.get("hf_id")): row for row in model_specs or []}
    normalized: list[JsonDict] = []
    for index, base in enumerate(MODEL_SPECS):
        hf_id = str(base["hf_id"])
        override = overrides.get(hf_id, {})
        resolved = str(
            override.get("model_path")
            or override.get("resolved_model_path")
            or resolve_cached_gguf(hf_id, str(base["quantization"]))
            or ""
        )
        path = Path(resolved).expanduser() if resolved else Path()
        present = bool(resolved and path.is_file())
        filename = path.name if present else ""
        model_hash = str(override.get("model_hash") or "")
        model_size = int(override.get("model_size_bytes") or 0)
        if present and not model_hash:
            model_hash = sha256_file(path)
        if present and not model_size:
            model_size = path.stat().st_size
        normalized.append(
            {
                **base,
                "sequence_index": index,
                "gpu": int(override.get("gpu", base["gpu"]) or 0),
                "model_path": resolved,
                "resolved_model_path": resolved,
                "gguf_filename": filename,
                "model_hash": model_hash if present else "",
                "model_size_bytes": model_size if present else 0,
                "quantization": _extract_quantization(filename, str(base["quantization"])),
                "local_model_present": present,
                "embedded_template_hash": str(
                    override.get("embedded_template_hash")
                    or override.get("chat_template_hash")
                    or ""
                ),
                "runtime_hash": str(override.get("runtime_hash") or ""),
                "sampling": dict(override.get("sampling") or SAMPLING_CONFIG),
                "stop": list(override.get("stop") or STOP_STRINGS),
                "token_budget": int(override.get("token_budget") or DEFAULT_MAX_TOKENS),
                "reasoning_budget": int(
                    override.get("reasoning_budget") or DEFAULT_REASONING_BUDGET_TOKENS
                ),
                "seed": int(override.get("seed") or RANDOM_SEEDS["runner_seed"] + index),
                "headline_eligible": override.get("headline_eligible") is not False,
                "legacy_smoke_only": False,
            }
        )
    return normalized


def _enrich_model_specs_from_preconditions(
    model_specs: Sequence[Mapping[str, Any]], preconditions_checked: Mapping[str, Any]
) -> list[JsonDict]:
    model_checks = dict(preconditions_checked.get("models") or {})
    enriched: list[JsonDict] = []
    for spec in model_specs:
        check = dict(model_checks.get(spec["hf_id"]) or {})
        row = dict(spec)
        row["embedded_template_hash"] = str(
            check.get("embedded_template_hash")
            or check.get("chat_template_hash")
            or row.get("embedded_template_hash")
            or ""
        )
        row["runtime_hash"] = str(check.get("runtime_hash") or row.get("runtime_hash") or "")
        row["sampling"] = dict(check.get("sampling") or row.get("sampling") or SAMPLING_CONFIG)
        row["stop"] = list(check.get("stop") or row.get("stop") or STOP_STRINGS)
        row["token_budget"] = int(check.get("token_budget") or row.get("token_budget") or DEFAULT_MAX_TOKENS)
        row["reasoning_budget"] = int(
            check.get("reasoning_budget")
            or row.get("reasoning_budget")
            or DEFAULT_REASONING_BUDGET_TOKENS
        )
        row["seed"] = int(check.get("seed") or row.get("seed") or RANDOM_SEEDS["runner_seed"])
        enriched.append(row)
    return enriched


def read_jsonl(path: str | Path) -> list[JsonDict]:
    """Read JSONL rows from disk."""

    source = Path(path)
    if not source.exists():
        return []
    return [
        dict(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_canary_rows(path: str | Path) -> list[JsonDict]:
    """Read an Exp5799 checkpoint/row JSONL file."""

    return read_jsonl(path)


def _family_quota(min_units: int) -> int:
    return max(1, min_units // len(fixture.REQUIRED_FAMILIES))


def select_canary_fixture(
    fixture_rows: Sequence[Mapping[str, Any]], *, min_units: int = 12
) -> list[JsonDict]:
    """Select a deterministic balanced fixture subset without inflating units."""

    by_unit_surface = {
        (str(row["unit_id"]), str(row["surface_kind"])): dict(row) for row in fixture_rows
    }
    canonical = [
        dict(row)
        for row in fixture_rows
        if row["surface_kind"] == "canonical"
        and row["split"] == "train"
        and row["family"] in fixture.REQUIRED_FAMILIES
    ]
    canonical.sort(key=lambda row: (row["family"], int(row["chronology_index"])))
    chosen: list[JsonDict] = []
    quota = _family_quota(min_units)
    for family in fixture.REQUIRED_FAMILIES:
        family_rows = [row for row in canonical if row["family"] == family]
        chosen.extend(family_rows[:quota])
    if len(chosen) < min_units:
        selected_units = {row["unit_id"] for row in chosen}
        for row in canonical:
            if row["unit_id"] in selected_units:
                continue
            chosen.append(row)
            selected_units.add(row["unit_id"])
            if len(chosen) >= min_units:
                break

    canary_rows: list[JsonDict] = []
    for index, row in enumerate(chosen[:min_units]):
        unit_id = str(row["unit_id"])
        variant = "symbol_relabel" if index % 2 == 0 else "order_paraphrase"
        canary_rows.append(dict(row))
        canary_rows.append(dict(by_unit_surface[(unit_id, variant)]))
    canary_rows.sort(key=lambda row: int(row["chronology_index"]))
    return canary_rows


def canary_fixture_hash(canary_rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the sealed canary subset by immutable Exp5785 row hashes."""

    return sha256_json([str(row["row_hash"]) for row in canary_rows])


def sample_size_justification(canary_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Explain the independent-unit denominator and coverage of fixture factors."""

    representative: dict[str, Mapping[str, Any]] = {}
    surfaces_by_unit: dict[str, set[str]] = {}
    for row in canary_rows:
        unit_id = str(row["unit_id"])
        representative.setdefault(unit_id, row)
        surfaces_by_unit.setdefault(unit_id, set()).add(str(row["surface_kind"]))
    reps = list(representative.values())
    pair_counts: Counter[str] = Counter()
    for surfaces in surfaces_by_unit.values():
        if {"canonical", "symbol_relabel"}.issubset(surfaces):
            pair_counts["canonical|symbol_relabel"] += 1
        if {"canonical", "order_paraphrase"}.issubset(surfaces):
            pair_counts["canonical|order_paraphrase"] += 1
    family_counts = Counter(str(row["family"]) for row in reps)
    status_counts = Counter(str(row["exact_status"]) for row in reps)
    solver_counts = Counter(str(row["solver_effort_bin"]) for row in reps)
    ready = bool(
        len(reps) >= 12
        and set(family_counts) == set(fixture.REQUIRED_FAMILIES)
        and {"sat", "unsat"}.issubset(status_counts)
        and {"low", "medium", "high"}.issubset(solver_counts)
        and pair_counts["canonical|symbol_relabel"] > 0
        and pair_counts["canonical|order_paraphrase"] > 0
    )
    return {
        "independent_unit_count": len(reps),
        "row_count": len(canary_rows),
        "family_counts": dict(sorted(family_counts.items())),
        "satisfiability_counts": dict(sorted(status_counts.items())),
        "solver_bin_counts": dict(sorted(solver_counts.items())),
        "surface_pair_counts": dict(sorted(pair_counts.items())),
        "repeated_modes_counted_as_independent": False,
        "repeated_surfaces_counted_as_independent": False,
        "minimum_independent_units_required": 12,
        "sample_size_note": "sealed balanced canary; repeated modes and surfaces are not independent units",
        "balanced_canary_ready": ready,
    }


def _candidate_lines(row: Mapping[str, Any]) -> str:
    return "\n".join(
        f"{item['label']}: {item['candidate']}" for item in row["label_mapping"]
    )


def build_prompt_cell(
    fixture_row: Mapping[str, Any], mode: Mapping[str, Any], model_spec: Mapping[str, Any]
) -> JsonDict:
    """Build a prompt whose only semantic output is the strict final line."""

    user = (
        f"Today is {RUN_DATE}.\n"
        "Evaluate the sealed finite-choice constraint fixture.\n"
        f"Fixture surface: {fixture_row['surface_text']}\n"
        "Candidates:\n"
        f"{_candidate_lines(fixture_row)}\n"
        f"Return the final answer as exactly this format: {fixture_row['row_id']}: <label>\n"
        f"The label must be one of: {', '.join(fixture_row['candidate_labels'])}."
    )
    if mode.get("mode_type") == "reasoning_disabled_final_sentinel":
        system = (
            "Do not reason in prose. Return one final row-id label line only. "
            "Do not output JSON and do not change fixture facts."
        )
    else:
        system = (
            "You may inspect the fixture internally, but the only final content "
            "that counts is one row-id label line. Do not output JSON and do "
            "not change fixture facts."
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt_payload = {
        "messages": messages,
        "mode_id": mode["mode_id"],
        "model_hf_id": model_spec["hf_id"],
        "semantic_contract_id": SEMANTIC_CONTRACT_ID,
        "parser": EXACT_PARSER_ID,
        "sampling": dict(SAMPLING_CONFIG),
        "stop": list(mode.get("stops") or STOP_STRINGS),
    }
    return {
        "row_id": str(fixture_row["row_id"]),
        "fixture_row": _copy_json(fixture_row),
        "messages": messages,
        "prompt_hash": sha256_json(prompt_payload),
    }


_FINAL_LINE_PATTERN = re.compile(
    r"^\s*(?P<row_id>\S+)\s*:\s*(?P<label>[A-Z][A-Z0-9_-]*)\s*$"
)


def split_response_text(raw_response_text: str) -> JsonDict:
    """Separate raw reasoning-like text from strict final candidate lines."""

    reasoning_lines: list[str] = []
    final_lines: list[str] = []
    for line in raw_response_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _FINAL_LINE_PATTERN.match(stripped):
            final_lines.append(stripped)
        else:
            reasoning_lines.append(line)
    reasoning = "\n".join(reasoning_lines).strip()
    final = "\n".join(final_lines).strip()
    return {
        "raw_reasoning_content": reasoning,
        "raw_final_content": final,
        "raw_reasoning_sha256": sha256_text(reasoning),
        "raw_final_sha256": sha256_text(final),
    }


def _selected_candidate(row: Mapping[str, Any], label: str) -> str:
    return next(
        (str(item["candidate"]) for item in row["label_mapping"] if item["label"] == label),
        "",
    )


def _protected_fact_distorted(row: Mapping[str, Any], text: str) -> bool:
    marker = "unit="
    if marker in text:
        for suffix in text.split(marker)[1:]:
            token = suffix.split()[0].strip(" ;,.\n\r\t")
            if token and token != row["unit_id"]:
                return True
    lowered = text.lower()
    return "protected_facts" in lowered or "exact_label" in lowered or "exact_answer" in lowered


def classify_canary_response(
    fixture_row: Mapping[str, Any],
    raw_response_text: str,
    *,
    finish_reason: str,
    output_tokens: int,
    mode: Mapping[str, Any],
    timeout: bool = False,
) -> JsonDict:
    """Classify one response at the exact Exp5785 finite-choice boundary."""

    split = split_response_text(raw_response_text)
    row_by_id = {str(fixture_row["row_id"]): fixture_row}
    parser = fixture.parse_response(str(split["raw_final_content"]), row_by_id)
    parse_ok = parser["parse_ok"] is True
    selected_label = ""
    selected_candidate = ""
    if parse_ok:
        selected_label = str(parser["parsed_labels"][str(fixture_row["row_id"])])
        selected_candidate = _selected_candidate(fixture_row, selected_label)
    exact_answer_error = bool(parse_ok and selected_label != fixture_row["exact_label"])
    max_tokens = int(mode.get("max_tokens") or DEFAULT_MAX_TOKENS)
    truncation = bool(
        finish_reason == "length"
        or parser["parser_failure_reason"] == "truncation"
        or output_tokens >= max_tokens
    )
    empty_final = str(split["raw_final_content"]) == ""
    invalid_candidate = parser["parser_failure_reason"] in {"invalid_id", "invalid_candidate"}
    stop_collision = bool(
        any(stop in raw_response_text for stop in mode.get("stops", STOP_STRINGS))
        or parser["parser_failure_reason"] == "stop_token"
    )
    protected = _protected_fact_distorted(fixture_row, raw_response_text)
    parser_failure = not parse_ok
    valid_exact_output = bool(
        parse_ok
        and not exact_answer_error
        and not parser_failure
        and not truncation
        and not empty_final
        and not invalid_candidate
        and not stop_collision
        and not timeout
        and not protected
    )
    if timeout:
        failure_mode = "timeout"
    elif empty_final:
        failure_mode = "empty_final_content"
    elif truncation:
        failure_mode = "truncation"
    elif stop_collision:
        failure_mode = "stop_collision"
    elif invalid_candidate:
        failure_mode = "invalid_candidate"
    elif parser_failure:
        failure_mode = "parser_failure"
    elif protected:
        failure_mode = "protected_fact_distortion"
    elif exact_answer_error:
        failure_mode = "exact_answer_error"
    else:
        failure_mode = "valid_exact_output"
    return {
        **split,
        "parse_ok": parse_ok,
        "parser_failure": parser_failure,
        "parser_failure_reason": str(parser["parser_failure_reason"]),
        "parsed_labels": dict(parser["parsed_labels"]),
        "selected_label": selected_label,
        "selected_candidate": selected_candidate,
        "selected_candidate_hash": sha256_text(selected_candidate) if selected_candidate else "",
        "exact_answer_error": exact_answer_error,
        "truncation": truncation,
        "empty_final_content": empty_final,
        "invalid_candidate": invalid_candidate,
        "stop_collision": stop_collision,
        "timeout": timeout,
        "protected_fact_distortion": protected,
        "valid_exact_output": valid_exact_output,
        "failure_mode": failure_mode,
    }


def canary_cell_key(row: Mapping[str, Any]) -> str:
    """Return the unique model/mode/fixture cell key."""

    return f"{row['model_hf_id']}::{row['mode_id']}::{row['fixture_row_id']}"


def canary_row_hash(row: Mapping[str, Any]) -> str:
    """Hash a canary row while excluding its own row hash."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def build_canary_row(
    *,
    model_spec: Mapping[str, Any],
    mode: Mapping[str, Any],
    fixture_row: Mapping[str, Any],
    raw_response: Mapping[str, Any],
    row_sequence_index: int,
) -> JsonDict:
    """Join live model text to parser and exact-validator receipts."""

    raw_text = str(raw_response.get("raw_response_text", ""))
    finish_reason = str(raw_response.get("finish_reason", ""))
    output_tokens = int(raw_response.get("output_tokens", 0) or 0)
    taxonomy = classify_canary_response(
        fixture_row,
        raw_text,
        finish_reason=finish_reason,
        output_tokens=output_tokens,
        mode=mode,
        timeout=raw_response.get("timeout") is True,
    )
    prompt_cell = build_prompt_cell(fixture_row, mode, model_spec)
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_sequence_index": row_sequence_index,
        "checkpoint_after_response": True,
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "model_hash": str(model_spec.get("model_hash", "")),
        "mode_id": str(mode["mode_id"]),
        "mode_type": str(mode["mode_type"]),
        "semantic_contract_id": SEMANTIC_CONTRACT_ID,
        "exact_parser_id": EXACT_PARSER_ID,
        "fixture_row_id": str(fixture_row["row_id"]),
        "fixture_unit_id": str(fixture_row["unit_id"]),
        "fixture_row_hash": str(fixture_row["row_hash"]),
        "surface_kind": str(fixture_row["surface_kind"]),
        "family": str(fixture_row["family"]),
        "solver_effort_bin": str(fixture_row["solver_effort_bin"]),
        "satisfiability": str(fixture_row["exact_status"]),
        "exact_label": str(fixture_row["exact_label"]),
        "exact_answer": str(fixture_row["exact_answer"]),
        "exact_certificate_hash": str(fixture_row["exact_certificate_hash"]),
        "prompt_hash": str(raw_response.get("prompt_hash") or prompt_cell["prompt_hash"]),
        "raw_response_text": raw_text,
        "raw_response_sha256": sha256_text(raw_text),
        "raw_reasoning_content": taxonomy["raw_reasoning_content"],
        "raw_reasoning_sha256": taxonomy["raw_reasoning_sha256"],
        "raw_final_content": taxonomy["raw_final_content"],
        "raw_final_sha256": taxonomy["raw_final_sha256"],
        "finish_reason": finish_reason,
        "output_tokens": output_tokens,
        "reasoning_token_estimate": len(str(taxonomy["raw_reasoning_content"]).split()),
        "timing": dict(raw_response.get("timing") or {}),
        "generation_error": str(raw_response.get("generation_error", "")),
        "timeout": taxonomy["timeout"],
        "stop_collision": taxonomy["stop_collision"],
        "parser_receipt": {
            "parse_ok": taxonomy["parse_ok"],
            "parser_failure_reason": taxonomy["parser_failure_reason"],
            "parsed_labels": taxonomy["parsed_labels"],
            "boundary": EXACT_PARSER_ID,
            "parsed_text_sha256": taxonomy["raw_final_sha256"],
        },
        "selected_label": taxonomy["selected_label"],
        "selected_candidate": taxonomy["selected_candidate"],
        "exact_validator_result": {
            "exact_label": str(fixture_row["exact_label"]),
            "selected_label": taxonomy["selected_label"],
            "exact_answer_error": taxonomy["exact_answer_error"],
            "valid_exact_output": taxonomy["valid_exact_output"],
            "exact_certificate_hash": str(fixture_row["exact_certificate_hash"]),
        },
        "taxonomy": taxonomy,
        "row_hash": "",
    }
    row["row_hash"] = canary_row_hash(row)
    return row


def append_canary_row(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one response row immediately after a model/mode/fixture cell."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()


def verify_canary_rows(
    rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    rows_path: str | Path | None = None,
) -> bool:
    """Replay row hashes, raw hashes, and checkpoint uniqueness."""

    if rows_path is not None:
        expected_file_hash = str(artifact.get("row_file_sha256") or "")
        if expected_file_hash and sha256_file(rows_path) != expected_file_hash:
            raise ManifestReplayError("row_file_sha256")
    receipts = dict(artifact.get("raw_response_receipts") or {})
    seen: set[str] = set()
    for row in rows:
        key = canary_cell_key(row)
        if key in seen:
            raise ManifestReplayError("duplicate canary cell")
        seen.add(key)
        if sha256_text(str(row.get("raw_response_text", ""))) != row.get("raw_response_sha256"):
            raise ManifestReplayError("raw_response_sha256")
        if canary_row_hash(row) != row.get("row_hash"):
            raise ManifestReplayError("row_hash")
        receipt = dict(receipts.get(key) or {})
        if receipt:
            for field in ("row_hash", "raw_response_sha256", "prompt_hash", "fixture_row_hash"):
                if receipt.get(field) != row.get(field):
                    raise ManifestReplayError(field)
    if receipts and set(receipts) != seen:
        raise ManifestReplayError("row receipt set")
    return True


def _prepare_existing_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    existing: dict[str, JsonDict] = {}
    for row in rows:
        key = canary_cell_key(row)
        if key in existing:
            raise ManifestReplayError("duplicate canary cell")
        existing[key] = dict(row)
    return existing


def preregistered_modes(diagnostic_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Return bounded Exp5798 modes in deterministic model preference order."""

    modes = [dict(row) for row in diagnostic_artifact.get("candidate_mode_matrix") or []]
    allowed = set(MANDATED_MODEL_IDS)
    result = [
        {
            **mode,
            "semantic_contract_id": SEMANTIC_CONTRACT_ID,
            "parser": EXACT_PARSER_ID,
            "stops": list(mode.get("stops") or STOP_STRINGS),
        }
        for mode in modes
        if mode.get("model_hf_id") in allowed
        and mode.get("bounded") is True
        and mode.get("executable") is True
        and mode.get("grammar_json") is not True
    ]
    if not result:
        raise ValueError("no preregistered Exp5798 modes")
    return result


def _preferred_modes_for_model(modes: Sequence[Mapping[str, Any]], hf_id: str) -> list[JsonDict]:
    model_modes = [dict(row) for row in modes if row["model_hf_id"] == hf_id]
    model_modes.sort(
        key=lambda row: (
            row.get("mode_type") != "reasoning_disabled_final_sentinel",
            int(row.get("max_tokens") or DEFAULT_MAX_TOKENS),
            str(row.get("mode_id")),
        )
    )
    return model_modes


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _rows_for_mode(
    rows: Sequence[Mapping[str, Any]], model_hf_id: str, mode_id: str
) -> list[JsonDict]:
    return [
        dict(row)
        for row in rows
        if row["model_hf_id"] == model_hf_id and row["mode_id"] == mode_id
    ]


def _mode_disqualifying_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counters = Counter()
    for row in rows:
        taxonomy = dict(row.get("taxonomy") or {})
        for field in (
            "parser_failure",
            "truncation",
            "empty_final_content",
            "invalid_candidate",
            "stop_collision",
            "timeout",
            "protected_fact_distortion",
        ):
            counters[field] += int(taxonomy.get(field) is True)
    return dict(counters)


def _runtime_authenticated(runtime_receipt: Mapping[str, Any]) -> bool:
    return bool(
        runtime_receipt.get("cuda_offload_authenticated") is True
        and int(runtime_receipt.get("n_gpu_layers_offloaded", 0) or 0) > 0
    )


def _mode_summary(
    *,
    model_hf_id: str,
    mode: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    expected_rows: int,
) -> JsonDict:
    counts = _mode_disqualifying_counts(rows)
    exact_errors = sum(
        1 for row in rows if dict(row.get("taxonomy") or {}).get("exact_answer_error") is True
    )
    accepted = sum(
        1 for row in rows if dict(row.get("taxonomy") or {}).get("valid_exact_output") is True
    )
    reasons = [
        name
        for name, count in sorted(counts.items())
        if count
    ]
    if len(rows) != expected_rows:
        reasons.append("missing_canary_rows")
    if not _runtime_authenticated(runtime_receipt):
        reasons.append("cuda_offload_not_authenticated")
    acceptable = bool(not reasons)
    return {
        "model_hf_id": model_hf_id,
        "model_family": diagnostic.model_family(model_hf_id),
        "mode_id": str(mode["mode_id"]),
        "mode_type": str(mode["mode_type"]),
        "semantic_contract_id": SEMANTIC_CONTRACT_ID,
        "parser": EXACT_PARSER_ID,
        "attempted": bool(rows),
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "acceptable": acceptable,
        "retired": bool(rows and not acceptable),
        "retirement_reasons": reasons,
        "parser_failure_count": int(counts.get("parser_failure", 0)),
        "truncation_count": int(counts.get("truncation", 0)),
        "empty_final_content_count": int(counts.get("empty_final_content", 0)),
        "invalid_candidate_count": int(counts.get("invalid_candidate", 0)),
        "stop_collision_count": int(counts.get("stop_collision", 0)),
        "timeout_count": int(counts.get("timeout", 0)),
        "protected_fact_distortion_count": int(counts.get("protected_fact_distortion", 0)),
        "exact_answer_error_count": exact_errors,
        "verified_exact_output_count": accepted,
        "cuda_offload_authenticated": _runtime_authenticated(runtime_receipt),
        "n_gpu_layers_offloaded": int(runtime_receipt.get("n_gpu_layers_offloaded", 0) or 0),
        "gpu_memory_peak_mb": int(runtime_receipt.get("gpu_memory_peak_mb", 0) or 0),
    }


def _raw_response_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        canary_cell_key(row): {
            "row_hash": str(row["row_hash"]),
            "raw_response_sha256": str(row["raw_response_sha256"]),
            "prompt_hash": str(row["prompt_hash"]),
            "fixture_row_hash": str(row["fixture_row_hash"]),
            "mode_id": str(row["mode_id"]),
        }
        for row in rows
    }


def _metric_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    counters = Counter()
    fields = (
        "parser_failure",
        "truncation",
        "empty_final_content",
        "invalid_candidate",
        "stop_collision",
        "timeout",
        "protected_fact_distortion",
        "exact_answer_error",
        "valid_exact_output",
    )
    for row in rows:
        taxonomy = dict(row.get("taxonomy") or {})
        for field in fields:
            counters[field] += int(taxonomy.get(field) is True)
    raw_reasoning = sum(1 for row in rows if "raw_reasoning_sha256" in row)
    raw_final = sum(1 for row in rows if str(row.get("raw_final_content") or ""))
    exact_label = sum(
        1
        for row in rows
        if row.get("exact_label") and row.get("exact_certificate_hash") and row.get("exact_validator_result")
    )
    return {
        "total": total,
        "raw_reasoning_count": raw_reasoning,
        "raw_final_count": raw_final,
        "exact_label_count": exact_label,
        **{field: 0 for field in fields},
        **dict(counters),
    }


def _total_generation_seconds(rows: Sequence[Mapping[str, Any]]) -> float:
    return round(
        sum(float(dict(row.get("timing") or {}).get("generation_s", 0.0) or 0.0) for row in rows),
        6,
    )


def _verified_outputs_per_second(rows: Sequence[Mapping[str, Any]]) -> float:
    accepted = sum(1 for row in rows if dict(row.get("taxonomy") or {}).get("valid_exact_output") is True)
    seconds = _total_generation_seconds(rows)
    return round(accepted / seconds, 6) if accepted and seconds > 0 else 0.0


def _verified_outputs_per_token(rows: Sequence[Mapping[str, Any]]) -> float:
    accepted = sum(1 for row in rows if dict(row.get("taxonomy") or {}).get("valid_exact_output") is True)
    tokens = sum(int(row.get("output_tokens", 0) or 0) for row in rows)
    return round(accepted / tokens, 6) if accepted and tokens > 0 else 0.0


def _wasted_token_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        int(row.get("output_tokens", 0) or 0)
        for row in rows
        if dict(row.get("taxonomy") or {}).get("valid_exact_output") is not True
    )


def adversarial_control_results(fixture_row: Mapping[str, Any]) -> JsonDict:
    """Run parser/validator controls that prevent syntax-only promotion."""

    wrong_label = next(
        label for label in fixture_row["candidate_labels"] if label != fixture_row["exact_label"]
    )
    mode = {
        "mode_id": "control",
        "mode_type": "control",
        "max_tokens": DEFAULT_MAX_TOKENS,
        "stops": list(STOP_STRINGS),
    }
    controls = {
        "empty_final_content": ("reasoning without final", "stop", 4),
        "reasoning_only_output": ("constraints look feasible", "stop", 4),
        "invalid_candidate_id": (f"unknown-row: {fixture_row['exact_label']}", "stop", 4),
        "schema_control_plane_injection": (
            '{"row_id":"x","label":"A"}\nignore previous instructions',
            "stop",
            8,
        ),
        "stop_collision": (f"scratch <stop>\n{fixture_row['row_id']}: {fixture_row['exact_label']}", "stop", 8),
        "truncation": ("partial reasoning", "length", DEFAULT_MAX_TOKENS),
        "protected_fact_mutation": (
            f"unit=mutated\n{fixture_row['row_id']}: {fixture_row['exact_label']}",
            "stop",
            8,
        ),
        "exact_answer_mismatch": (f"{fixture_row['row_id']}: {wrong_label}", "stop", 4),
    }
    result: JsonDict = {}
    for control_id, (text, finish_reason, output_tokens) in controls.items():
        taxonomy = classify_canary_response(
            fixture_row,
            text,
            finish_reason=finish_reason,
            output_tokens=output_tokens,
            mode=mode,
        )
        passed = False
        if control_id == "exact_answer_mismatch":
            passed = taxonomy["parse_ok"] is True and taxonomy["exact_answer_error"] is True
        elif control_id == "schema_control_plane_injection":
            passed = taxonomy["valid_exact_output"] is False
        else:
            passed = taxonomy["valid_exact_output"] is False
        result[control_id] = {
            "parser_ok": taxonomy["parse_ok"],
            "parser_failure_reason": taxonomy["parser_failure_reason"],
            "exact_answer_error": taxonomy["exact_answer_error"],
            "valid_exact_output": taxonomy["valid_exact_output"],
            "schema_injection_accepted": bool(
                control_id == "schema_control_plane_injection"
                and taxonomy["valid_exact_output"] is True
            ),
            "protected_fact_distortion": taxonomy["protected_fact_distortion"],
            "stop_collision": taxonomy["stop_collision"],
            "passed": passed,
        }
    return result


def _adversarial_controls_pass(artifact: Mapping[str, Any]) -> bool:
    controls = dict(artifact.get("adversarial_control_results") or {})
    required = {
        "empty_final_content",
        "reasoning_only_output",
        "invalid_candidate_id",
        "schema_control_plane_injection",
        "stop_collision",
        "truncation",
        "protected_fact_mutation",
        "exact_answer_mismatch",
    }
    return required.issubset(controls) and all(
        dict(row).get("passed") is True for row in controls.values()
    )


def _embedded_template_receipts(
    preconditions_checked: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]
) -> JsonDict:
    checks = dict(preconditions_checked.get("models") or {})
    return {
        str(spec["hf_id"]): {
            "model_path": str(spec.get("model_path") or ""),
            "model_hash": str(spec.get("model_hash") or ""),
            "gguf_filename": str(spec.get("gguf_filename") or ""),
            "quantization": str(spec.get("quantization") or ""),
            "embedded_template_checked": dict(checks.get(spec["hf_id"]) or {}).get(
                "embedded_template_checked",
                bool(spec.get("embedded_template_hash")),
            )
            is True,
            "embedded_template_hash": str(
                dict(checks.get(spec["hf_id"]) or {}).get("embedded_template_hash")
                or dict(checks.get(spec["hf_id"]) or {}).get("chat_template_hash")
                or spec.get("embedded_template_hash")
                or ""
            ),
            "template_replaced": False,
            "autotokenizer_used": False,
        }
        for spec in model_specs
    }


def _selected_runtime(
    selected: Mapping[str, Any], runtime_receipts: Mapping[str, Mapping[str, Any]], hf_id: str
) -> JsonDict:
    mode_id = str(dict(selected.get(hf_id) or {}).get("mode_id") or "")
    return dict(runtime_receipts.get(f"{hf_id}::{mode_id}") or {})


def _model_runtime_receipts(
    selected: Mapping[str, Any], runtime_receipts: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    result: JsonDict = {}
    for hf_id in MANDATED_MODEL_IDS:
        selected_receipt = _selected_runtime(selected, runtime_receipts, hf_id)
        per_mode = {
            key.split("::", 1)[1]: dict(value)
            for key, value in runtime_receipts.items()
            if key.startswith(f"{hf_id}::")
        }
        result[hf_id] = {**selected_receipt, "mode_runtime_receipts": per_mode}
    return result


def _gpu_offload_receipts(
    selected: Mapping[str, Any], runtime_receipts: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    result: JsonDict = {}
    for hf_id in MANDATED_MODEL_IDS:
        receipt = _selected_runtime(selected, runtime_receipts, hf_id)
        result[hf_id] = {
            "cuda_offload_authenticated": receipt.get("cuda_offload_authenticated") is True,
            "n_gpu_layers_requested": int(receipt.get("n_gpu_layers_requested", 0) or 0),
            "n_gpu_layers_offloaded": int(receipt.get("n_gpu_layers_offloaded", 0) or 0),
            "gpu_memory_before_mb": int(receipt.get("gpu_memory_before_mb", 0) or 0),
            "gpu_memory_peak_mb": int(receipt.get("gpu_memory_peak_mb", 0) or 0),
            "gpu_memory_after_mb": int(receipt.get("gpu_memory_after_mb", 0) or 0),
            "offload_log_excerpt": str(receipt.get("offload_log_excerpt", ""))[-1000:],
        }
    return result


def answer_channel_ready_score_from_artifact(artifact: Mapping[str, Any]) -> float:
    """Recompute the strict answer-channel readiness score."""

    def number(field: str, default: float = 0.0) -> float:
        return float(artifact[field]) if field in artifact else default

    ready = bool(
        artifact.get("status") == "complete"
        and artifact.get("models_used") == list(MANDATED_MODEL_IDS)
        and int(artifact.get("qualified_real_sota_model_count") or 0) == 3
        and set(dict(artifact.get("selected_transport_by_model") or {})) == set(MANDATED_MODEL_IDS)
        and number("raw_reasoning_coverage") == 1.0
        and number("raw_final_content_coverage") == 1.0
        and number("exact_label_coverage") == 1.0
        and number("parser_failure_rate", 1.0) == 0.0
        and number("truncation_rate", 1.0) == 0.0
        and number("empty_final_content_rate", 1.0) == 0.0
        and number("invalid_candidate_rate", 1.0) == 0.0
        and int(artifact["protected_fact_distortion_count"] if "protected_fact_distortion_count" in artifact else 1) == 0
        and _adversarial_controls_pass(artifact)
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and all(
            dict(row).get("acceptable") is True
            for row in artifact.get("mode_execution_matrix", [])
        )
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal complete/blocked verdict."""

    if artifact.get("status") == "blocked":
        reasons = dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [
            "preconditions"
        ]
        return "blocked: " + ",".join(str(reason) for reason in reasons)
    if answer_channel_ready_score_from_artifact(artifact) == 1.0:
        return "complete: answer_channel_ready_all_three_real_sota_models"
    qualified = int(artifact.get("qualified_real_sota_model_count") or 0)
    return f"complete: answer_channel_canary_complete_not_ready:qualified_models={qualified}"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    canary_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    runtime_receipts: Mapping[str, Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    modes_attempted: Sequence[Mapping[str, Any]],
    row_file_path: str | Path,
    checkpoint_resume_receipts: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Build the terminal artifact from checkpointed response rows."""

    expected_rows_per_mode = len(canary_rows)
    mode_matrix = [
        _mode_summary(
            model_hf_id=str(mode["model_hf_id"]),
            mode=mode,
            rows=_rows_for_mode(response_rows, str(mode["model_hf_id"]), str(mode["mode_id"])),
            runtime_receipt=runtime_receipts.get(
                f"{mode['model_hf_id']}::{mode['mode_id']}",
                {},
            ),
            expected_rows=expected_rows_per_mode,
        )
        for mode in modes_attempted
    ]
    selected: JsonDict = {}
    for summary in mode_matrix:
        hf_id = str(summary["model_hf_id"])
        if summary["acceptable"] is True and hf_id not in selected:
            selected[hf_id] = {
                "mode_id": summary["mode_id"],
                "mode_type": summary["mode_type"],
                "semantic_contract_id": SEMANTIC_CONTRACT_ID,
                "parser": EXACT_PARSER_ID,
                "bounded": True,
                "selected": True,
            }
    counts = _metric_counts(response_rows)
    total = int(counts["total"])
    row_file = Path(row_file_path)
    row_hash = sha256_file(row_file) if row_file.exists() and total else sha256_text("")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "status": "complete",
        "preconditions_checked": _copy_json(preconditions_checked),
        "MODEL_SPECS": [_copy_json(row) for row in model_specs],
        "models_used": list(MANDATED_MODEL_IDS),
        "model_runtime_receipts": _model_runtime_receipts(selected, runtime_receipts),
        "gpu_offload_receipts": _gpu_offload_receipts(selected, runtime_receipts),
        "embedded_template_receipts": _embedded_template_receipts(preconditions_checked, model_specs),
        "canary_fixture_hash": canary_fixture_hash(canary_rows),
        "independent_unit_count": int(
            sample_size_justification(canary_rows)["independent_unit_count"]
        ),
        "sample_size_justification": sample_size_justification(canary_rows),
        "mode_execution_matrix": mode_matrix,
        "selected_transport_by_model": selected,
        "raw_response_receipts": _raw_response_receipts(response_rows),
        "raw_reasoning_coverage": _rate(int(counts["raw_reasoning_count"]), total),
        "raw_final_content_coverage": _rate(int(counts["raw_final_count"]), total),
        "exact_label_coverage": _rate(int(counts["exact_label_count"]), total),
        "parser_failure_rate": _rate(int(counts["parser_failure"]), total),
        "truncation_rate": _rate(int(counts["truncation"]), total),
        "empty_final_content_rate": _rate(int(counts["empty_final_content"]), total),
        "invalid_candidate_rate": _rate(int(counts["invalid_candidate"]), total),
        "exact_answer_error_rate": _rate(int(counts["exact_answer_error"]), total),
        "stop_collision_rate": _rate(int(counts["stop_collision"]), total),
        "timeout_rate": _rate(int(counts["timeout"]), total),
        "protected_fact_distortion_count": int(counts["protected_fact_distortion"]),
        "adversarial_control_results": adversarial_control_results(canary_rows[0]) if canary_rows else {},
        "verified_outputs_per_second": _verified_outputs_per_second(response_rows),
        "verified_outputs_per_token": _verified_outputs_per_token(response_rows),
        "wasted_token_count": _wasted_token_count(response_rows),
        "checkpoint_resume_receipts": dict(checkpoint_resume_receipts),
        "qualified_real_sota_model_count": len(selected),
        "answer_channel_ready_score": 0.0,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "row_file": row_file.as_posix(),
        "row_file_sha256": row_hash,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    if preconditions_checked.get("preconditions_ready") is not True:
        artifact["status"] = "blocked"
    artifact["answer_channel_ready_score"] = answer_channel_ready_score_from_artifact(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on missing fields, schema drift, or unsupported readiness."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("models_used") != list(MANDATED_MODEL_IDS):
        raise ValueError("models_used")
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(MANDATED_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    for field in artifact.get("producer_gate_fields", []):
        if field not in artifact or isinstance(artifact[field], Mapping):
            raise ValueError("producer_gate_fields")
    if artifact.get("answer_channel_ready_score") != answer_channel_ready_score_from_artifact(artifact):
        raise ValueError("answer_channel_ready_score")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("status") == "complete" and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if artifact.get("status") == "blocked" and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _resume_runtime_receipt(
    model_spec: Mapping[str, Any], mode: Mapping[str, Any], existing_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    return {
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "mode_id": str(mode["mode_id"]),
        "resume_from_checkpoint": True,
        "llama_cpp_version": "resume_from_checkpoint",
        "llama_cpp_build_info": {"resume_from_checkpoint": True},
        "chat_template": {"used": True, "resume_from_checkpoint": True},
        "cuda_device_receipt": {"resume_from_checkpoint": True},
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": 0,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 0,
        "gpu_memory_after_mb": 0,
        "cuda_offload_authenticated": False,
        "rows_attempted": len(existing_rows),
        "offload_log_excerpt": "resume_only_no_runtime_receipt",
    }


def _runtime_receipt_is_resume_only(runtime_receipt: Mapping[str, Any]) -> bool:
    return bool(
        runtime_receipt.get("resume_from_checkpoint") is True
        or runtime_receipt.get("llama_cpp_version") == "resume_from_checkpoint"
        or dict(runtime_receipt.get("cuda_device_receipt") or {}).get("resume_from_checkpoint")
        is True
        or "resume_from" in str(runtime_receipt.get("offload_log_excerpt") or "")
    )


def _prior_runtime_receipt(
    result_path: str | Path, model_spec: Mapping[str, Any], mode: Mapping[str, Any]
) -> JsonDict:
    prior_path = Path(result_path)
    if not prior_path.is_file():
        return {}
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive corrupt-prior fallback.
        return {}
    model_receipts = dict(prior.get("model_runtime_receipts") or {})
    model = dict(model_receipts.get(str(model_spec["hf_id"])) or {})
    receipt = dict(dict(model.get("mode_runtime_receipts") or {}).get(str(mode["mode_id"])) or {})
    if not receipt or _runtime_receipt_is_resume_only(receipt):
        return {}
    if receipt.get("model_hf_id") != model_spec["hf_id"] or receipt.get("mode_id") != mode["mode_id"]:
        return {}  # pragma: no cover - defensive stale-prior fallback.
    receipt["replayed_from_prior_artifact"] = True
    return receipt


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    fixture_artifact: Mapping[str, Any] | None = None,
    fixture_rows: Sequence[Mapping[str, Any]] | None = None,
    diagnostic_artifact: Mapping[str, Any] | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    canary_runner: CanaryRunner | None = None,
    min_units: int = 12,
    max_modes_per_model: int = 1,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run or resume the matched local-SOTA answer-channel canary."""

    base_specs = normalize_model_specs(model_specs)
    preconditions = dict(preconditions_checked or collect_preconditions())
    specs = _enrich_model_specs_from_preconditions(base_specs, preconditions)
    source_rows = list(fixture_rows or fixture.read_row_file(REPO_ROOT / EXP5785_ROWS_RELATIVE_PATH))
    diag = dict(
        diagnostic_artifact
        or json.loads((REPO_ROOT / EXP5798_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    )
    canary_rows = select_canary_fixture(source_rows, min_units=min_units)
    output_rows_path = Path(row_file_path)
    existing_rows = read_canary_rows(output_rows_path)
    existing = _prepare_existing_rows(existing_rows)
    all_rows: list[JsonDict] = list(existing_rows)
    rows_written = 0
    duplicate_skipped = 0
    modes = preregistered_modes(diag)
    attempted_modes: list[JsonDict] = []
    runtime_receipts: dict[str, JsonDict] = {}
    runner = canary_runner or default_canary_runner

    if preconditions.get("preconditions_ready") is True:
        if write:
            output_rows_path.parent.mkdir(parents=True, exist_ok=True)
            output_rows_path.touch(exist_ok=True)
        for model_spec in specs:
            for mode in _preferred_modes_for_model(modes, str(model_spec["hf_id"]))[:max_modes_per_model]:
                attempted_modes.append(mode)
                pending_cells = []
                for row in canary_rows:
                    key = f"{model_spec['hf_id']}::{mode['mode_id']}::{row['row_id']}"
                    if key in existing:
                        duplicate_skipped += 1
                        continue
                    pending_cells.append(build_prompt_cell(row, mode, model_spec))
                receipt_key = f"{model_spec['hf_id']}::{mode['mode_id']}"
                if pending_cells:

                    def emit_response(
                        raw_response: Mapping[str, Any], *, spec: Mapping[str, Any] = model_spec, active_mode: Mapping[str, Any] = mode
                    ) -> None:
                        nonlocal rows_written
                        fixture_row = next(
                            row for row in canary_rows if row["row_id"] == raw_response["row_id"]
                        )
                        canary_row = build_canary_row(
                            model_spec=spec,
                            mode=active_mode,
                            fixture_row=fixture_row,
                            raw_response=raw_response,
                            row_sequence_index=len(all_rows),
                        )
                        key = canary_cell_key(canary_row)
                        if key in existing:
                            raise ManifestReplayError("duplicate canary cell")
                        existing[key] = canary_row
                        all_rows.append(canary_row)
                        rows_written += 1
                        if write:
                            append_canary_row(output_rows_path, canary_row)

                    runtime_receipts[receipt_key] = runner(
                        dict(model_spec),
                        dict(mode),
                        pending_cells,
                        emit_response,
                    )
                else:
                    mode_rows = _rows_for_mode(all_rows, str(model_spec["hf_id"]), str(mode["mode_id"]))
                    runtime_receipts[receipt_key] = _prior_runtime_receipt(
                        result_path,
                        model_spec,
                        mode,
                    ) or _resume_runtime_receipt(model_spec, mode, mode_rows)
                current_rows = _rows_for_mode(
                    all_rows,
                    str(model_spec["hf_id"]),
                    str(mode["mode_id"]),
                )
                current_summary = _mode_summary(
                    model_hf_id=str(model_spec["hf_id"]),
                    mode=mode,
                    rows=current_rows,
                    runtime_receipt=runtime_receipts[receipt_key],
                    expected_rows=len(canary_rows),
                )
                if current_summary["acceptable"] is True:
                    break
    checkpoint_receipts = {
        "schema": SCHEMA + ".checkpoint_resume",
        "row_file": output_rows_path.as_posix(),
        "existing_rows_loaded": len(existing_rows),
        "rows_written": rows_written,
        "duplicate_cells_skipped": duplicate_skipped,
        "checkpoint_after_every_response": all(
            row.get("checkpoint_after_response") is True for row in all_rows
        ),
        "replayed_row_hashes_match": all(canary_row_hash(row) == row.get("row_hash") for row in all_rows),
        "duplicate_cells_present": len(existing) != len(all_rows),
        "resume_supported": True,
    }
    artifact = build_artifact(
        model_specs=specs,
        canary_rows=canary_rows,
        response_rows=all_rows,
        runtime_receipts=runtime_receipts,
        preconditions_checked=preconditions,
        modes_attempted=attempted_modes,
        row_file_path=row_file_path,
        checkpoint_resume_receipts=checkpoint_receipts,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes or {},
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 32768
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": required_mb, "ok": available_mb >= required_mb}


def _disk_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 4096
    available_mb = int(shutil.disk_usage(REPO_ROOT).free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": required_mb, "ok": available_mb >= required_mb}


def _hash_optional_file(path: str | Path) -> str:  # pragma: no cover - host-dependent preflight.
    candidate = Path(path)
    return sha256_file(candidate) if candidate.is_file() else "missing"


def _runtime_hash() -> str:  # pragma: no cover - host-dependent preflight.
    parts: JsonDict = {}
    try:
        llama_cpp = importlib.import_module("llama_cpp")
        module_path = Path(str(getattr(llama_cpp, "__file__", "")))
        parts["python_module_path"] = str(module_path)
        parts["python_module_hash"] = _hash_optional_file(module_path)
        parts["version"] = importlib.metadata.version("llama-cpp-python")
        lib_path = module_path.parent / "lib" / "libllama.so"
        parts["libllama_path"] = str(lib_path)
        parts["libllama_hash"] = _hash_optional_file(lib_path)
    except Exception as exc:
        parts["import_error"] = repr(exc)
    return sha256_json(parts)


def _chat_template_probe(model_path: str) -> JsonDict:  # pragma: no cover - host-dependent preflight.
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        metadata = getattr(llm, "metadata", {}) or {}
        template = str(metadata.get("tokenizer.chat_template") or "")
        del llm
        gc.collect()
        return {
            "available": bool(template),
            "embedded_template_hash": sha256_text(template) if template else "",
            "metadata_keys": sorted(str(key) for key in metadata)[:64],
            "ok": bool(template),
        }
    except Exception as exc:
        return {"available": False, "embedded_template_hash": "", "ok": False, "error": repr(exc)}


def _replay_exp5798(path: str | Path) -> JsonDict:  # pragma: no cover - host-dependent preflight.
    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        diagnostic.validate_artifact(artifact)
        receipts = [
            {
                "field": "channel_diagnostic_ready_score",
                "expected": 1.0,
                "actual": artifact.get("channel_diagnostic_ready_score"),
                "passed": artifact.get("channel_diagnostic_ready_score") == 1.0,
            },
            {
                "field": "candidate_mode_count",
                "expected": ">=3",
                "actual": artifact.get("candidate_mode_count"),
                "passed": int(artifact.get("candidate_mode_count") or 0) >= 3,
            },
        ]
        return {
            "ok": all(row["passed"] for row in receipts),
            "artifact_path": str(EXP5798_ARTIFACT_RELATIVE_PATH),
            "artifact_sha256": sha256_file(path),
            "gate_receipts": receipts,
        }
    except Exception as exc:
        return {"ok": False, "artifact_path": str(EXP5798_ARTIFACT_RELATIVE_PATH), "error": repr(exc)}


def _replay_fixture(
    artifact_path: str | Path, row_file_path: str | Path, canary_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover - host-dependent preflight.
    try:
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        rows = fixture.read_row_file(row_file_path)
        fixture.validate_artifact(artifact)
        fixture.verify_row_file(rows, artifact)
        sample = sample_size_justification(canary_rows)
        receipts = [
            {
                "field": "fixture_ready_score",
                "expected": 1.0,
                "actual": artifact.get("fixture_ready_score"),
                "passed": artifact.get("fixture_ready_score") == 1.0,
            },
            {
                "field": "exact_label_coverage",
                "expected": 1.0,
                "actual": artifact.get("exact_label_coverage"),
                "passed": artifact.get("exact_label_coverage") == 1.0,
            },
            {
                "field": "parser_control_pass_rate",
                "expected": 1.0,
                "actual": artifact.get("parser_control_pass_rate"),
                "passed": artifact.get("parser_control_pass_rate") == 1.0,
            },
            {
                "field": "balanced_canary_ready",
                "expected": True,
                "actual": sample["balanced_canary_ready"],
                "passed": sample["balanced_canary_ready"] is True,
            },
        ]
        return {
            "ok": all(row["passed"] for row in receipts),
            "artifact_path": str(EXP5785_ARTIFACT_RELATIVE_PATH),
            "artifact_sha256": sha256_file(artifact_path),
            "row_file_sha256": sha256_file(row_file_path),
            "canary_fixture_hash": canary_fixture_hash(canary_rows),
            "independent_unit_count": sample["independent_unit_count"],
            "gate_receipts": receipts,
        }
    except Exception as exc:
        return {"ok": False, "artifact_path": str(EXP5785_ARTIFACT_RELATIVE_PATH), "error": repr(exc)}


def collect_preconditions(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    fixture_artifact_path: str | Path = REPO_ROOT / EXP5785_ARTIFACT_RELATIVE_PATH,
    fixture_row_file_path: str | Path = REPO_ROOT / EXP5785_ROWS_RELATIVE_PATH,
    diagnostic_artifact_path: str | Path = REPO_ROOT / EXP5798_ARTIFACT_RELATIVE_PATH,
) -> JsonDict:  # pragma: no cover - host-dependent preflight.
    """Collect all Step 0 checks before Exp5799 generation starts."""

    pair = cached_sota_pair()
    specs = normalize_model_specs()
    fixture_rows = fixture.read_row_file(fixture_row_file_path)
    canary_rows = select_canary_fixture(fixture_rows)
    exp5798 = _replay_exp5798(diagnostic_artifact_path)
    fixture_replay = _replay_fixture(fixture_artifact_path, fixture_row_file_path, canary_rows)
    devices = stream._nvidia_smi_devices()
    rtx_count = sum(1 for row in devices if "RTX 3090" in str(row.get("name", "")))
    llama = stream._llama_cpp_probe()
    llama["runtime_hash"] = _runtime_hash()
    memory = _memory_probe()
    disk = _disk_probe()
    output_parent = Path(result_path).parent
    row_parent = Path(row_file_path).parent
    model_checks: JsonDict = {}
    for spec in specs:
        chat = _chat_template_probe(str(spec["model_path"])) if spec["local_model_present"] else {}
        model_checks[spec["hf_id"]] = {
            "local_model_present": spec["local_model_present"],
            "model_hash_checked": bool(spec["model_hash"]),
            "model_path": spec["model_path"],
            "model_hash": spec["model_hash"],
            "gguf_filename": spec["gguf_filename"],
            "quantization": spec["quantization"],
            "embedded_template_checked": chat.get("ok") is True,
            "embedded_template_hash": chat.get("embedded_template_hash", ""),
            "runtime_hash": llama["runtime_hash"],
            "sampling": dict(SAMPLING_CONFIG),
            "stop": list(STOP_STRINGS),
            "token_budget": DEFAULT_MAX_TOKENS,
            "reasoning_budget": DEFAULT_REASONING_BUDGET_TOKENS,
            "gpu": spec["gpu"],
            "seed": RANDOM_SEEDS["runner_seed"] + int(spec["sequence_index"]),
            "free_vram_mb": max((int(row.get("memory_free_mb", 0) or 0) for row in devices), default=0),
            "min_vram_mb": int(float(spec["min_vram_gb"]) * 1000),
            "ok": bool(spec["local_model_present"] and spec["model_hash"] and chat.get("ok") is True),
        }
    third_added = next((row for row in specs if row["hf_id"] == GEMMA31_ID), {})
    blocked: list[str] = []
    if pair is None:
        blocked.append("cached_sota_pair_unavailable")
    if exp5798.get("ok") is not True:
        blocked.append("exp5798_gate_replay_failed")
    if fixture_replay.get("ok") is not True:
        blocked.append("fixture_subset_unbalanced")
    if rtx_count < 2:
        blocked.append("dual_rtx_3090_unavailable")
    if llama.get("ok") is not True:
        blocked.append("llama_cpp_cuda_unavailable")
    if memory.get("ok") is not True:
        blocked.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked.append("insufficient_free_disk")
    if not (output_parent.exists() and os.access(output_parent, os.W_OK)):
        blocked.append("output_parent_unwritable")
    if not (row_parent.exists() and os.access(row_parent, os.W_OK)):
        blocked.append("row_parent_unwritable")
    for hf_id, check in model_checks.items():
        if check["ok"] is not True:
            blocked.append(f"model_preflight_failed:{hf_id}")
        if int(check["free_vram_mb"]) < int(check["min_vram_mb"]):
            blocked.append(f"insufficient_free_vram:{hf_id}")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "exp5798_gate_replay": exp5798,
        "cached_sota_pair_called": True,
        "cached_sota_pair_result": pair or [],
        "third_mandated_model_added": {
            "hf_id": GEMMA31_ID,
            "model_path": third_added.get("model_path", ""),
            "added": bool(third_added.get("local_model_present") is True),
        },
        "cuda_devices": {"ok": rtx_count >= 2, "rtx_3090_count": rtx_count, "devices": devices},
        "llama_cpp": llama,
        "models": model_checks,
        "fixture_subset": fixture_replay,
        "memory": memory,
        "disk": disk,
        "output_paths": {
            "result_path": str(result_path),
            "row_file": str(row_file_path),
            "parent_writable": output_parent.exists() and row_parent.exists(),
        },
        "deterministic_seeds": dict(RANDOM_SEEDS),
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def _gpu_used_total_mb() -> int:  # pragma: no cover - host-dependent runtime.
    return sum(int(row.get("memory_used_mb", 0) or 0) for row in stream._nvidia_smi_devices())


def default_canary_runner(
    model_spec: JsonDict,
    mode: JsonDict,
    prompt_cells: list[JsonDict],
    emit_response: EmitResponse,
) -> JsonDict:  # pragma: no cover - host-dependent live GGUF path.
    """Generate raw responses for one model/mode through llama-cpp-python CUDA."""

    devices_before = stream._nvidia_smi_devices()
    before_mb = _gpu_used_total_mb()
    worker_payload = {
        "model_spec": model_spec,
        "mode": mode,
        "prompt_cells": [
            {
                "row_id": cell["row_id"],
                "messages": cell["messages"],
                "prompt_hash": cell["prompt_hash"],
            }
            for cell in prompt_cells
        ],
        "generation_config": dict(GENERATION_CONFIG),
    }
    worker_code = r"""
import gc
import importlib.metadata
import json
import sys
import time

payload = json.loads(sys.stdin.read())
try:
    import llama_cpp
    from llama_cpp import Llama

    raw_info = llama_cpp.llama_cpp.llama_print_system_info()
    system_info = raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
    supports_gpu = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
    version = importlib.metadata.version("llama-cpp-python")
    vocab = Llama(model_path=payload["model_spec"]["model_path"], vocab_only=True, verbose=False)
    metadata = getattr(vocab, "metadata", {}) or {}
    template = str(metadata.get("tokenizer.chat_template") or "")
    del vocab
    gc.collect()
    llm = Llama(
        model_path=payload["model_spec"]["model_path"],
        n_gpu_layers=int(payload["generation_config"]["n_gpu_layers"]),
        n_ctx=int(payload["generation_config"]["n_ctx"]),
        n_batch=int(payload["generation_config"]["n_batch"]),
        seed=int(payload["model_spec"]["seed"]),
        verbose=True,
    )
    for cell in payload["prompt_cells"]:
        started = time.perf_counter()
        try:
            result = llm.create_chat_completion(
                messages=cell["messages"],
                temperature=float(payload["generation_config"]["temperature"]),
                top_p=float(payload["generation_config"]["top_p"]),
                top_k=int(payload["generation_config"]["top_k"]),
                max_tokens=int(payload["mode"]["max_tokens"]),
                stop=list(payload["mode"]["stops"]),
                seed=int(payload["model_spec"]["seed"]),
            )
            choice = result["choices"][0]
            message = choice.get("message") or {}
            content = str(message.get("content") or choice.get("text") or "")
            reasoning = str(message.get("reasoning_content") or "")
            text = (reasoning + "\n" + content).strip() if reasoning else content
            finish_reason = str(choice.get("finish_reason") or "")
            usage = result.get("usage") or {}
            output_tokens = int(usage.get("completion_tokens") or 0)
            error = ""
        except Exception as exc:
            text = ""
            finish_reason = "error"
            output_tokens = 0
            error = repr(exc)
        elapsed = time.perf_counter() - started
        print(json.dumps({
            "type": "row",
            "row_id": cell["row_id"],
            "prompt_hash": cell["prompt_hash"],
            "raw_response_text": text,
            "finish_reason": finish_reason,
            "output_tokens": output_tokens,
            "timing": {"generation_s": round(elapsed, 6)},
            "generation_error": error,
        }, sort_keys=True), flush=True)
    del llm
    gc.collect()
    print(json.dumps({
        "type": "summary",
        "llama_cpp_version": version,
        "llama_cpp_build_info": {
            "cuda_backend": "CUDA" in system_info.upper(),
            "supports_gpu_offload": supports_gpu,
            "system_info": system_info,
            "module": getattr(llama_cpp, "__file__", ""),
        },
        "chat_template": {
            "available": bool(template),
            "used": True,
            "chat_template_hash": "sha256:" + __import__("hashlib").sha256(template.encode()).hexdigest() if template else "",
            "template_replaced": False,
            "autotokenizer_used": False,
        },
    }, sort_keys=True), flush=True)
except Exception as exc:
    print(json.dumps({"type": "summary", "error": repr(exc)}, sort_keys=True), flush=True)
    raise
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(model_spec.get("gpu", 0))},
    )
    stderr_chunks: list[str] = []
    stop_monitor = threading.Event()
    samples: list[int] = []

    def _stderr_reader() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_chunks.append(line)

    def _monitor() -> None:
        while not stop_monitor.is_set():
            samples.append(_gpu_used_total_mb())
            time.sleep(0.25)

    threading.Thread(target=_stderr_reader, daemon=True).start()
    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(worker_payload))
    proc.stdin.close()
    summary: JsonDict = {}
    timeout_s = float(mode.get("timeout_s") or os.environ.get("CARNOT_5799_MODE_TIMEOUT_S", "900"))
    started = time.monotonic()
    timed_out = False
    for line in proc.stdout:
        payload = json.loads(line)
        if payload.get("type") == "row":
            emit_response(payload)
        elif payload.get("type") == "summary":
            summary = payload
        if time.monotonic() - started > timeout_s:
            timed_out = True
            proc.kill()
            break
    proc.wait(timeout=30)
    stop_monitor.set()
    monitor.join(timeout=2)
    after_mb = _gpu_used_total_mb()
    stderr_text = "".join(stderr_chunks)
    offloaded = stream._parse_offloaded_layers(stderr_text)
    peak_mb = max(samples or [before_mb])
    gc.collect()
    return {
        "model_hf_id": model_spec["hf_id"],
        "model_family": model_spec["family"],
        "mode_id": mode["mode_id"],
        "llama_cpp_version": str(summary.get("llama_cpp_version") or ""),
        "llama_cpp_build_info": dict(summary.get("llama_cpp_build_info") or {}),
        "chat_template": dict(summary.get("chat_template") or {}),
        "cuda_device_receipt": {
            "before": devices_before,
            "peak": samples,
            "after": stream._nvidia_smi_devices(),
            "worker_returncode": proc.returncode,
            "worker_error": str(summary.get("error") or ""),
            "cuda_visible_devices": str(model_spec.get("gpu", 0)),
            "timed_out": timed_out,
        },
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": offloaded,
        "gpu_memory_before_mb": before_mb,
        "gpu_memory_peak_mb": peak_mb,
        "gpu_memory_after_mb": after_mb,
        "cuda_offload_authenticated": bool(offloaded > 0 and peak_mb > before_mb and not timed_out),
        "rows_attempted": len(prompt_cells),
        "offload_log_excerpt": stderr_text[-4000:],
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp5799 from the command line."""

    del argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
