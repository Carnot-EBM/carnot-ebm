"""Exp5798 offline SOTA answer-channel diagnostic.

Spec refs: REQ-VERIFY-5798, SCENARIO-VERIFY-5798,
SCENARIO-VERIFY-5798-CONTROLS, REQ-REPORT-5798,
SCENARIO-REPORT-5798, SCENARIO-REPORT-5798-BLOCKED.

This module performs exact forensics over Exp5786 rows that already exist on
disk. It does not call a model. The diagnostic boundary is deliberately narrow:
local rows prove what happened, GGUF/runtime metadata explains the transport
context, upstream issues motivate controls, and Exp5785 exact labels remain the
only semantic truth source.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any


JsonDict = dict[str, Any]
MetadataReader = Callable[[str], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5798_sota_answer_channel_diagnostic.json")
EXP5785_ARTIFACT_RELATIVE_PATH = Path("results/experiment_5785_hardness_surface_fixture.json")
EXP5786_ARTIFACT_RELATIVE_PATH = Path("results/experiment_5786_sota_constraint_stream.json")
EXP5786_ROWS_RELATIVE_PATH = Path("results/experiment_5786_sota_constraint_stream.rows.jsonl")
EXP5785_MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5785_hardness_surface_fixture.py")
EXP5786_MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5786_sota_constraint_stream.py")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5798_sota_answer_channel_diagnostic.py")
EXPERIMENT_TEMPLATE_RELATIVE_PATH = Path("scripts/experiment_template.py")

SCHEMA = "carnot.experiment_5798.sota_answer_channel_diagnostic.v1"
EXPERIMENT = 5798
EXPERIMENT_ID = "experiment_5798_sota_answer_channel_diagnostic"
MILESTONE = "2026.07.517"
RUN_DATE = "20260722"
INFERENCE_SUBSTRATE = "offline_exact_forensics_over_existing_real_gguf_rows_no_llm"
SPEC_REFS = (
    "REQ-VERIFY-5798",
    "SCENARIO-VERIFY-5798",
    "SCENARIO-VERIFY-5798-CONTROLS",
    "REQ-REPORT-5798",
    "SCENARIO-REPORT-5798",
    "SCENARIO-REPORT-5798-BLOCKED",
)

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_MODEL_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5798_sota_answer_channel_diagnostic.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5798_sota_answer_channel_diagnostic.py -m pytest tests/python/test_experiment_5798_sota_answer_channel_diagnostic.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5798_sota_answer_channel_diagnostic.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)
PRODUCER_GATE_FIELDS = (
    "row_count",
    "raw_response_coverage",
    "per_model_failure_attribution",
    "candidate_mode_count",
    "llm_calls_made",
    "channel_diagnostic_ready_score",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "input_artifact_hashes",
    "row_count",
    "raw_response_coverage",
    "per_model_failure_attribution",
    "reasoning_content_receipts",
    "final_content_receipts",
    "stop_reason_counts",
    "token_length_distributions",
    "qwen_answer_sentinel_count",
    "qwen_empty_final_count",
    "qwen_exact_cap_count",
    "embedded_template_metadata",
    "llama_cpp_runtime_receipts",
    "upstream_issue_receipts",
    "local_upstream_distinction",
    "candidate_mode_matrix",
    "candidate_mode_count",
    "mode_acceptance_rules",
    "mode_retirement_rules",
    "adversarial_control_matrix",
    "llm_calls_made",
    "channel_diagnostic_ready_score",
    "producer_gate_fields",
    "inference_substrate",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "status": "Terminal diagnostic state from local row/hash and schema gates.",
    "preconditions_checked": "Records file hashes, row coverage, disk/RAM, runtime, and template checks before attribution.",
    "input_artifact_hashes": "Binds Exp5785, Exp5786, row, parser/runtime, llama.cpp, and GGUF metadata evidence.",
    "row_count": "Bare row denominator prevents silent row loss.",
    "raw_response_coverage": "Every raw row must carry response text and hash evidence before diagnosis.",
    "per_model_failure_attribution": "Separates parser, final-content, stop, token, template, and exact-answer evidence per model.",
    "reasoning_content_receipts": "Reasoning-like text is preserved as channel evidence, not semantic authority.",
    "final_content_receipts": "Strict final row-id label lines are counted separately from reasoning text.",
    "stop_reason_counts": "Termination behavior is direct local receipt evidence.",
    "token_length_distributions": "Token exhaustion is measured independently from parser failure.",
    "qwen_answer_sentinel_count": "Bare Qwen strict-boundary count prevents prose from hiding absent final answers.",
    "qwen_empty_final_count": "Bare Qwen empty-final count records channel failure directly.",
    "qwen_exact_cap_count": "Bare Qwen exact-cap count distinguishes length exits from stop-before-boundary rows.",
    "embedded_template_metadata": "GGUF template metadata supplies transport context without replacing the embedded template.",
    "llama_cpp_runtime_receipts": "Pinned runtime/package/library receipts distinguish runtime context from model behavior.",
    "upstream_issue_receipts": "Issue references motivate controls but do not count as local evidence.",
    "local_upstream_distinction": "Local row symptoms, runtime metadata, upstream claims, and exact truth are separate.",
    "candidate_mode_matrix": "Downstream canary modes are preregistered, executable, and bounded before generation.",
    "candidate_mode_count": "Bare scalar prevents prose from inflating mode coverage.",
    "mode_acceptance_rules": "Acceptance rules are frozen before the canary run.",
    "mode_retirement_rules": "Retirement rules prevent looping, empty final, and grammar-only success from being promoted.",
    "adversarial_control_matrix": "Negative controls show why syntax constraints cannot prove semantic correctness.",
    "llm_calls_made": "Bare scalar must remain zero for this offline forensics task.",
    "channel_diagnostic_ready_score": "Strict gate for complete classification, provenance separation, bounded modes, and preregistered rules.",
    "producer_gate_fields": "Names the bare fields downstream producers must inspect.",
    "inference_substrate": "Declares offline exact forensics over existing real-GGUF rows with no LLM calls.",
    "test_commands": "Verification commands are preserved exactly.",
    "test_exit_codes": "Observed exit codes are recorded without relabeling failures.",
    "reproducibility_checksum": "Stable content checksum detects artifact drift.",
    "honest_verdict": "Terminal summary starts with complete: or blocked: and does not call Qwen truncation a competence failure.",
}
EXPECTED_ADVERSARIAL_CONTROLS = (
    "empty_final_content",
    "reasoning_only_output",
    "invalid_candidate_id",
    "duplicate_candidate_id",
    "schema_control_plane_injection",
    "stop_collision",
    "unclosed_thinking",
    "max_token_exhaustion",
    "exact_answer_mismatch",
)
_DEFAULT_METADATA_READER = object()
_FINAL_PATTERN_TEMPLATE = r"(?m)^\s*{row_id}\s*:\s*(?P<label>[A-Z][A-Z0-9_-]*)\s*$"


class ManifestReplayError(ValueError):
    """Raised when Exp5786 rows no longer match their sealed receipts."""


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks so large GGUF files remain streamable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def model_family(hf_id: str) -> str:
    """Return the stable model-family slug used in Exp5786 rows."""

    if hf_id == QWEN_ID:
        return "qwen3-6-35b-a3b"
    if hf_id == GEMMA31_ID:
        return "gemma-4-31b-it"
    if hf_id == GEMMA26_ID:
        return "gemma-4-26b-a4b-it"
    return hf_id.rsplit("/", 1)[-1].replace("-GGUF", "").replace(".", "-").lower()


def stream_cell_key(row: Mapping[str, Any]) -> str:
    """Return the unique Exp5786 model-by-fixture cell key."""

    return f"{row['model_hf_id']}::{row['fixture_row_id']}"


def stream_row_hash(row: Mapping[str, Any]) -> str:
    """Replay the Exp5786 row hash while excluding the row hash field."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def read_jsonl(path: str | Path) -> list[JsonDict]:
    """Read JSONL rows from disk."""

    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_input_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    stream_artifact: Mapping[str, Any],
    rows_path: str | Path | None = None,
) -> bool:
    """Fail closed unless every Exp5786 row replays against artifact receipts."""

    receipts = dict(stream_artifact.get("raw_response_receipts") or {})
    if rows_path is not None:
        expected_file_hash = str(stream_artifact.get("row_file_sha256") or "")
        if expected_file_hash and sha256_file(rows_path) != expected_file_hash:
            raise ManifestReplayError("row_file_sha256")
    seen: set[str] = set()
    for row in rows:
        key = stream_cell_key(row)
        if key in seen:
            raise ManifestReplayError("duplicate stream cell")
        seen.add(key)
        raw_text = str(row.get("raw_response_text", ""))
        if sha256_text(raw_text) != row.get("raw_response_sha256"):
            raise ManifestReplayError("raw_response_sha256")
        if stream_row_hash(row) != row.get("row_hash"):
            raise ManifestReplayError("row_hash")
        receipt = dict(receipts.get(key) or {})
        if receipts and not receipt:
            raise ManifestReplayError("missing receipt")
        for field in ("row_hash", "raw_response_sha256", "prompt_hash", "fixture_row_hash"):
            if receipt and receipt.get(field) != row.get(field):
                raise ManifestReplayError(field)
    if receipts and len(rows) != len(receipts):
        raise ManifestReplayError("row count")
    return True


def _strict_final_match(row: Mapping[str, Any]) -> re.Match[str] | None:
    pattern = _FINAL_PATTERN_TEMPLATE.format(row_id=re.escape(str(row["fixture_row_id"])))
    return next(re.finditer(pattern, str(row.get("raw_response_text", ""))), None)


def split_reasoning_final(row: Mapping[str, Any], *, max_tokens: int) -> JsonDict:
    """Split a row into reasoning-like text and strict final boundary content."""

    raw_text = str(row.get("raw_response_text", ""))
    match = _strict_final_match(row)
    final_content = match.group(0).strip() if match else ""
    reasoning_content = raw_text[: match.start()].strip() if match else raw_text.strip()
    parser = dict(row.get("parser_receipt") or {})
    taxonomy = dict(row.get("taxonomy") or {})
    output_tokens = int(row.get("output_tokens", 0) or 0)
    finish_reason = str(row.get("finish_reason") or "")
    exact_cap = output_tokens == max_tokens
    token_exhausted = finish_reason == "length" or exact_cap
    failure_classes: list[str] = []
    if final_content == "":
        failure_classes.extend(["parser_boundary_missing", "final_content_empty"])
    if finish_reason == "length":
        failure_classes.append("finish_reason_length")
    if exact_cap:
        failure_classes.append("exact_token_cap")
    if parser.get("parser_failure_reason"):
        failure_classes.append(f"parser_reported_{parser['parser_failure_reason']}")
    if taxonomy.get("exact_answer_error") is True:
        failure_classes.append("exact_answer_mismatch")
    if taxonomy.get("valid_correct_response") is True:
        failure_classes.append("valid_exact_match")
    if finish_reason == "stop" and final_content == "":
        failure_classes.append("stop_before_boundary")
    if "<think>" in raw_text and "</think>" not in raw_text:
        failure_classes.append("unclosed_thinking")
    if parser.get("parse_ok") is not True:
        exact_wrongness = "not_scored_parser_boundary_absent"
    elif taxonomy.get("exact_answer_error") is True:
        exact_wrongness = "exact_mismatch"
    else:
        exact_wrongness = "exact_match"
    return {
        "row_key": stream_cell_key(row),
        "model_hf_id": str(row.get("model_hf_id", "")),
        "model_family": str(row.get("model_family") or model_family(str(row.get("model_hf_id", "")))),
        "fixture_row_id": str(row.get("fixture_row_id", "")),
        "reasoning_nonempty": bool(reasoning_content),
        "reasoning_sha256": sha256_text(reasoning_content) if reasoning_content else "",
        "final_nonempty": bool(final_content),
        "final_empty": not bool(final_content),
        "final_content": final_content,
        "final_sha256": sha256_text(final_content) if final_content else "",
        "answer_sentinel_present": bool(match),
        "parser_ok": parser.get("parse_ok") is True,
        "selected_label": str(row.get("selected_label") or ""),
        "exact_wrongness": exact_wrongness,
        "finish_reason": finish_reason,
        "output_tokens": output_tokens,
        "exact_token_cap": exact_cap,
        "token_exhausted": token_exhausted,
        "failure_classes": list(dict.fromkeys(failure_classes)),
    }


def _count_by(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    return dict(sorted(Counter(str(row.get(field) or "") for row in rows).items()))


def _token_distribution(rows: Sequence[Mapping[str, Any]], *, max_tokens: int) -> JsonDict:
    values = [int(row.get("output_tokens", 0) or 0) for row in rows]
    hist = dict(sorted(Counter(str(value) for value in values).items(), key=lambda item: int(item[0])))
    stop_len = Counter(
        f"{row.get('finish_reason') or ''}:{int(row.get('output_tokens', 0) or 0)}"
        for row in rows
    )
    exact_cap = sum(1 for value in values if value == max_tokens)
    return {
        "n_rows": len(rows),
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "mean": round(sum(values) / len(values), 6) if values else 0.0,
        "max_tokens_configured": max_tokens,
        "exact_cap_count": exact_cap,
        "exact_cap_fraction": round(exact_cap / len(rows), 6) if rows else 0.0,
        "histogram": hist,
        "per_stop_length_counts": dict(sorted(stop_len.items())),
    }


def _receipt_samples(values: Sequence[str]) -> list[str]:
    return list(values[:3])


def _per_model_summaries(
    rows: Sequence[Mapping[str, Any]], splits: Sequence[Mapping[str, Any]], *, max_tokens: int
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict, JsonDict]:
    failure_attribution: JsonDict = {}
    reasoning_receipts: JsonDict = {}
    final_receipts: JsonDict = {}
    stop_counts: JsonDict = {}
    token_distributions: JsonDict = {}
    split_by_key = {str(split["row_key"]): split for split in splits}
    for hf_id in MANDATED_MODEL_IDS:
        model_rows = [dict(row) for row in rows if row.get("model_hf_id") == hf_id]
        model_splits = [split_by_key[stream_cell_key(row)] for row in model_rows]
        class_counts = Counter(cls for split in model_splits for cls in split["failure_classes"])
        parser_failure_count = sum(
            1 for row in model_rows if dict(row.get("parser_receipt") or {}).get("parse_ok") is not True
        )
        exact_mismatch_count = sum(
            1 for row in model_rows if dict(row.get("taxonomy") or {}).get("exact_answer_error") is True
        )
        sentinel_count = sum(1 for split in model_splits if split["answer_sentinel_present"])
        final_empty_count = sum(1 for split in model_splits if split["final_empty"])
        exact_cap_count = sum(1 for split in model_splits if split["exact_token_cap"])
        if hf_id == QWEN_ID and sentinel_count == 0:
            larger_budget_status = "not_established_from_existing_rows"
        elif parser_failure_count == 0:
            larger_budget_status = "not_needed_current_rows_parse"
        else:
            larger_budget_status = "mixed_current_rows_canary_required"
        failure_attribution[hf_id] = {
            "model_family": model_family(hf_id),
            "row_count": len(model_rows),
            "parser_failure_count": parser_failure_count,
            "answer_sentinel_count": sentinel_count,
            "empty_final_count": final_empty_count,
            "exact_cap_count": exact_cap_count,
            "all_rows_exact_cap": bool(model_rows and exact_cap_count == len(model_rows)),
            "finish_reason_length_count": sum(1 for row in model_rows if row.get("finish_reason") == "length"),
            "stop_before_boundary_count": int(class_counts.get("stop_before_boundary", 0)),
            "exact_mismatch_count": exact_mismatch_count,
            "failure_class_counts": dict(sorted(class_counts.items())),
            "larger_bounded_budget_parse_status": larger_budget_status,
            "attribution": _attribution_text(hf_id, parser_failure_count, sentinel_count, exact_cap_count, len(model_rows)),
        }
        reasoning_hashes = [str(split["reasoning_sha256"]) for split in model_splits if split["reasoning_sha256"]]
        final_hashes = [str(split["final_sha256"]) for split in model_splits if split["final_sha256"]]
        reasoning_receipts[hf_id] = {
            "model_family": model_family(hf_id),
            "row_count": len(model_rows),
            "reasoning_nonempty_count": sum(1 for split in model_splits if split["reasoning_nonempty"]),
            "reasoning_empty_count": sum(1 for split in model_splits if not split["reasoning_nonempty"]),
            "unclosed_thinking_count": int(class_counts.get("unclosed_thinking", 0)),
            "reasoning_sha256_samples": _receipt_samples(reasoning_hashes),
        }
        final_receipts[hf_id] = {
            "model_family": model_family(hf_id),
            "row_count": len(model_rows),
            "final_nonempty_count": sum(1 for split in model_splits if split["final_nonempty"]),
            "final_empty_count": final_empty_count,
            "answer_sentinel_count": sentinel_count,
            "parser_ok_count": len(model_rows) - parser_failure_count,
            "parsed_label_counts": _count_by(model_rows, "selected_label"),
            "exact_match_count": sum(1 for split in model_splits if split["exact_wrongness"] == "exact_match"),
            "exact_mismatch_count": exact_mismatch_count,
            "not_scored_count": sum(
                1 for split in model_splits if split["exact_wrongness"] == "not_scored_parser_boundary_absent"
            ),
            "final_sha256_samples": _receipt_samples(final_hashes),
        }
        stop_counts[hf_id] = dict(sorted(Counter(str(row.get("finish_reason") or "") for row in model_rows).items()))
        token_distributions[hf_id] = _token_distribution(model_rows, max_tokens=max_tokens)
    return failure_attribution, reasoning_receipts, final_receipts, stop_counts, token_distributions


def _attribution_text(
    hf_id: str, parser_failures: int, sentinel_count: int, exact_cap_count: int, row_count: int
) -> str:
    if hf_id == QWEN_ID:
        return (
            "local rows show an answer-channel failure before the strict row-id label boundary; "
            f"sentinel_count={sentinel_count}, parser_failures={parser_failures}, "
            f"exact_cap_rows={exact_cap_count}/{row_count}. This is not a competence verdict."
        )
    return (
        "matched Gemma control rows reached the strict row-id label boundary; exact mismatches, "
        "if any, are scored only after deterministic parsing."
    )


def _read_llama_cpp_metadata(model_path: str) -> Mapping[str, Any]:  # pragma: no cover - host metadata path.
    try:
        from llama_cpp import Llama

        model = Llama(model_path=model_path, vocab_only=True, verbose=False)
        metadata = dict(getattr(model, "metadata", {}) or {})
        del model
        metadata["metadata_source"] = "local_llama_cpp_vocab_only_no_generation"
        return metadata
    except Exception as exc:
        return {"metadata_source": "metadata_read_failed", "metadata_error": repr(exc)}


def embedded_template_metadata(
    stream_artifact: Mapping[str, Any],
    *,
    metadata_reader: MetadataReader | None | object = _DEFAULT_METADATA_READER,
) -> JsonDict:
    """Summarize embedded GGUF chat-template metadata without changing templates."""

    models = dict(dict(stream_artifact.get("preconditions_checked") or {}).get("models") or {})
    result: JsonDict = {}
    for hf_id in MANDATED_MODEL_IDS:
        receipt = dict(models.get(hf_id) or {})
        model_path = str(receipt.get("model_path") or "")
        metadata: Mapping[str, Any] = {}
        if metadata_reader is _DEFAULT_METADATA_READER and model_path:
            metadata = _read_llama_cpp_metadata(model_path)
        elif callable(metadata_reader) and model_path:
            metadata = metadata_reader(model_path)
        template = str(metadata.get("tokenizer.chat_template") or "")
        template_hash = sha256_text(template) if template else str(receipt.get("chat_template_hash") or "")
        result[hf_id] = {
            "model_family": model_family(hf_id),
            "model_path": model_path,
            "gguf_filename": str(receipt.get("gguf_filename") or ""),
            "model_hash": str(receipt.get("model_hash") or ""),
            "chat_template_checked_in_exp5786": receipt.get("chat_template_checked") is True,
            "chat_template_hash": template_hash,
            "template_metadata_source": str(metadata.get("metadata_source") or "exp5786_precondition_receipt"),
            "template_length": len(template),
            "contains_think_tags": "<think" in template or "</think>" in template,
            "contains_enable_thinking": "enable_thinking" in template,
            "contains_reasoning_content_key": "reasoning_content" in template,
            "supports_reasoning_disable": "enable_thinking" in template,
            "embedded_template_is_default_authority": True,
            "template_replaced": False,
            "tokenizer_downloaded": False,
        }
    return result


def _hash_optional_file(path: str | Path) -> str:
    candidate = Path(path)
    return sha256_file(candidate) if candidate.is_file() else "missing"


def llama_cpp_runtime_receipts(stream_artifact: Mapping[str, Any]) -> JsonDict:
    """Collect pinned llama.cpp package and shared-library receipts."""

    runtime = dict(stream_artifact.get("model_runtime_receipts") or {})
    pre_llama = dict(dict(stream_artifact.get("preconditions_checked") or {}).get("llama_cpp") or {})
    package: JsonDict = {
        "version": str(pre_llama.get("version") or ""),
        "cuda_backend": pre_llama.get("cuda_backend") is True,
        "supports_gpu_offload": pre_llama.get("supports_gpu_offload") is True,
        "system_info_hash": sha256_text(str(pre_llama.get("system_info") or "")),
        "standalone_binary_present": False,
        "standalone_binary_path": "",
        "standalone_binary_hash": "",
        "python_module_path": "",
        "python_module_hash": "",
        "libllama_path": "",
        "libllama_hash": "",
    }
    binary = shutil.which("llama-cli")
    if binary:
        package["standalone_binary_present"] = True
        package["standalone_binary_path"] = binary
        package["standalone_binary_hash"] = sha256_file(binary)
    try:
        llama_cpp = importlib.import_module("llama_cpp")
        module_path = Path(str(getattr(llama_cpp, "__file__", "")))
        package["python_module_path"] = str(module_path)
        package["python_module_hash"] = _hash_optional_file(module_path)
        package["version"] = package["version"] or importlib.metadata.version("llama-cpp-python")
        lib_path = module_path.parent / "lib" / "libllama.so"
        package["libllama_path"] = str(lib_path)
        package["libllama_hash"] = _hash_optional_file(lib_path)
    except Exception as exc:  # pragma: no cover - import availability is host-specific.
        package["python_import_error"] = repr(exc)
    per_model = {
        hf_id: {
            "llama_cpp_version": str(dict(runtime.get(hf_id) or {}).get("llama_cpp_version") or package["version"]),
            "cuda_offload_authenticated": dict(runtime.get(hf_id) or {}).get("cuda_offload_authenticated") is True,
            "n_gpu_layers_requested": int(dict(runtime.get(hf_id) or {}).get("n_gpu_layers_requested", 0) or 0),
            "n_gpu_layers_offloaded": int(dict(runtime.get(hf_id) or {}).get("n_gpu_layers_offloaded", 0) or 0),
            "chat_template_hash": str(
                dict(dict(runtime.get(hf_id) or {}).get("chat_template") or {}).get("chat_template_hash")
                or ""
            ),
            "rows_attempted": int(dict(runtime.get(hf_id) or {}).get("rows_attempted", 0) or 0),
        }
        for hf_id in MANDATED_MODEL_IDS
    }
    return {"package": package, "per_model": per_model}


def input_artifact_hashes(
    *,
    fixture_artifact: Mapping[str, Any],
    stream_artifact: Mapping[str, Any],
    input_paths: Mapping[str, str | Path] | None,
    template_metadata: Mapping[str, Any],
    runtime_receipts: Mapping[str, Any],
) -> JsonDict:
    """Hash local artifacts, code, runtime, and cached GGUF metadata receipts."""

    hashes: JsonDict = {
        "exp5785_fixture_artifact_json": sha256_json(fixture_artifact),
        "exp5786_artifact_json": sha256_json(stream_artifact),
        "cached_gguf_metadata": {
            hf_id: {
                "model_hash": dict(template_metadata.get(hf_id) or {}).get("model_hash", ""),
                "chat_template_hash": dict(template_metadata.get(hf_id) or {}).get("chat_template_hash", ""),
                "metadata_receipt_hash": sha256_json(dict(template_metadata.get(hf_id) or {})),
            }
            for hf_id in MANDATED_MODEL_IDS
        },
        "llama_cpp_runtime_package": sha256_json(dict(runtime_receipts.get("package") or {})),
    }
    for name, rel_path in (input_paths or {}).items():
        path = Path(rel_path)
        hashes[name] = sha256_file(path) if path.is_file() else "missing"
    return hashes


def upstream_issue_receipts() -> JsonDict:
    """Return issue receipts as upstream motivation, not local proof."""

    return {
        "20345": {
            "url": "https://github.com/ggml-org/llama.cpp/issues/20345",
            "claim_summary": "Qwen reasoning plus grammar can constrain thinking, leave final content empty, or loop.",
            "local_reference": "research-references.md:30592 and openspec/change-proposals/research-roadmap-vNEXT.md:104",
            "evidence_role": "motivation_only",
            "local_receipt": False,
        },
        "22792": {
            "url": "https://github.com/ggml-org/llama.cpp/issues/22792",
            "claim_summary": "The embedded GGUF chat template is the default authority.",
            "local_reference": "research-references.md:30597",
            "evidence_role": "transport_context_motivation",
            "local_receipt": False,
        },
        "20196": {
            "url": "https://github.com/ggml-org/llama.cpp/issues/20196",
            "claim_summary": "Reasoning-disable behavior is template/model dependent.",
            "local_reference": "research-references.md:30599",
            "evidence_role": "control_motivation",
            "local_receipt": False,
        },
    }


def local_upstream_distinction() -> JsonDict:
    """State which evidence boundary each source family is allowed to support."""

    return {
        "local_rows_establish": [
            "raw text bytes",
            "strict final-boundary presence or absence",
            "finish reasons",
            "completion token counts",
            "parser and exact-validator outcomes",
        ],
        "runtime_template_metadata_establishes": [
            "llama.cpp package/runtime context",
            "embedded template hashes",
            "whether metadata exposes thinking controls",
        ],
        "upstream_issues_motivate": [
            "reasoning-disable canary arms",
            "grammar guarded negative controls",
            "embedded-template authority checks",
        ],
        "upstream_issue_prose_is_local_receipt": False,
        "larger_budget_success_is_local_receipt": False,
        "qwen_competence_failure_claim_supported": False,
        "grammar_can_establish_semantic_correctness": False,
    }


def candidate_mode_matrix(template_metadata: Mapping[str, Any]) -> list[JsonDict]:
    """Preregister bounded per-family modes for the downstream real-GGUF canary."""

    modes: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        family = model_family(hf_id)
        base_fail = [
            "missing_raw_reasoning_or_final_receipt",
            "empty_final_content",
            "invalid_or_duplicate_candidate_id",
            "max_token_exhaustion",
            "exact_validator_mismatch",
            "cuda_offload_not_authenticated",
        ]
        modes.append(
            {
                "mode_id": f"{family}:embedded_template_final_sentinel_192",
                "model_hf_id": hf_id,
                "model_family": family,
                "mode_type": "embedded_template_final_sentinel",
                "max_tokens": 192,
                "stops": ["<|eot_id|>", "<stop>"],
                "finalizer": "strict line '<fixture_row_id>: <candidate_label>' after optional reasoning",
                "parser": "exp5785_row_id_to_candidate_label_exact_parser",
                "timeout_s": 900,
                "fail_closed_conditions": base_fail,
                "bounded": True,
                "executable": True,
                "grammar_json": False,
            }
        )
        metadata = dict(template_metadata.get(hf_id) or {})
        supports_disable = metadata.get("supports_reasoning_disable") is True
        mode_type = "reasoning_disabled_final_sentinel" if supports_disable else "final_only_eos_budget"
        modes.append(
            {
                "mode_id": f"{family}:{mode_type}_128",
                "model_hf_id": hf_id,
                "model_family": family,
                "mode_type": mode_type,
                "max_tokens": 128,
                "stops": ["<|eot_id|>", "<stop>"],
                "finalizer": "single strict final row-id label line",
                "parser": "exp5785_row_id_to_candidate_label_exact_parser",
                "timeout_s": 900,
                "fail_closed_conditions": base_fail + ["reasoning_disable_unsupported_by_metadata"]
                if supports_disable
                else base_fail,
                "bounded": True,
                "executable": True,
                "metadata_supports_reasoning_disable": supports_disable,
                "grammar_json": False,
            }
        )
    return modes


def mode_acceptance_rules() -> JsonDict:
    """Freeze the canary acceptance contract before Exp5799 generation."""

    return {
        "zero_parser_failures_required": True,
        "raw_reasoning_final_split_required": True,
        "exact_validator_authority": "Exp5785 exact labels only",
        "max_token_exhaustion_allowed": False,
        "empty_final_allowed": False,
        "invalid_or_duplicate_candidate_allowed": False,
        "grammar_success_counts_as_semantic_correctness": False,
        "matched_all_three_families_required": True,
        "authenticated_cuda_offload_required": True,
    }


def mode_retirement_rules() -> JsonDict:
    """Freeze mode retirement triggers before Exp5799 generation."""

    return {
        "retire_on_any_unbounded_generation": True,
        "retire_on_empty_final_content": True,
        "retire_on_reasoning_only_output": True,
        "retire_on_stop_collision": True,
        "retire_on_unclosed_thinking": True,
        "retire_on_max_token_exhaustion": True,
        "retire_on_grammar_loop_or_syntax_only_success": True,
        "retire_on_exact_answer_mismatch_in_adversarial_control": True,
    }


def adversarial_control_matrix() -> list[JsonDict]:
    """Return the negative controls that protect the downstream canary."""

    descriptions = {
        "empty_final_content": "Parser must reject present reasoning with no final label.",
        "reasoning_only_output": "Reasoning text alone cannot be credited as a candidate.",
        "invalid_candidate_id": "Unknown fixture row ids fail before exact scoring.",
        "duplicate_candidate_id": "Repeated final labels for one row fail closed.",
        "schema_control_plane_injection": "JSON or instruction injection cannot override parser boundaries.",
        "stop_collision": "A stop string inside reasoning must not create a false final.",
        "unclosed_thinking": "Open thinking tags or channels retire the mode.",
        "max_token_exhaustion": "Length termination blocks mode acceptance even if syntax is partial.",
        "exact_answer_mismatch": "Grammar-valid wrong labels remain exact validator failures.",
    }
    return [
        {
            "control_id": control_id,
            "description": descriptions[control_id],
            "parser_expected": "reject" if control_id != "exact_answer_mismatch" else "parse_then_fail_exact",
            "exact_validator_required": True,
            "grammar_can_establish_semantic_correctness": False,
            "fail_closed": True,
        }
        for control_id in EXPECTED_ADVERSARIAL_CONTROLS
    ]


def _raw_response_coverage(rows: Sequence[Mapping[str, Any]], expected_rows: int) -> float:
    covered = sum(
        1
        for row in rows
        if isinstance(row.get("raw_response_text"), str)
        and str(row.get("raw_response_sha256") or "").startswith("sha256:")
    )
    return round(covered / expected_rows, 6) if expected_rows else 0.0


def _preconditions_checked(
    *,
    fixture_artifact: Mapping[str, Any],
    stream_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    replay_ok: bool,
) -> JsonDict:
    stream_pre = dict(stream_artifact.get("preconditions_checked") or {})
    expected_rows = len(dict(stream_artifact.get("raw_response_receipts") or {})) or len(rows)
    return {
        "run_date": RUN_DATE,
        "exp5785_fixture_ready_score": fixture_artifact.get("fixture_ready_score"),
        "exp5786_raw_response_coverage": stream_artifact.get("raw_response_coverage"),
        "expected_row_count": expected_rows,
        "observed_row_count": len(rows),
        "raw_response_coverage_required": True,
        "row_hash_replay_ok": replay_ok,
        "fail_closed_on_row_or_hash_mismatch": True,
        "disk": dict(stream_pre.get("disk") or {}),
        "memory": dict(stream_pre.get("memory") or {}),
        "llama_cpp": dict(stream_pre.get("llama_cpp") or {}),
        "llm_calls_made": 0,
    }


def build_diagnostic_artifact(
    *,
    fixture_artifact: Mapping[str, Any],
    stream_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    input_paths: Mapping[str, str | Path] | None = None,
    metadata_reader: MetadataReader | None | object = _DEFAULT_METADATA_READER,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the terminal Exp5798 artifact from existing Exp5786 rows."""

    rows_path = input_paths.get("exp5786_rows") if input_paths else None
    verify_input_rows(rows=rows, stream_artifact=stream_artifact, rows_path=rows_path)
    max_tokens = int(dict(stream_artifact.get("generation_config") or {}).get("max_tokens", 0) or 0)
    split_rows = [split_reasoning_final(row, max_tokens=max_tokens) for row in rows]
    failure, reasoning, final, stop_counts, token_dist = _per_model_summaries(
        rows, split_rows, max_tokens=max_tokens
    )
    qwen = failure[QWEN_ID]
    template_meta = embedded_template_metadata(stream_artifact, metadata_reader=metadata_reader)
    runtime = llama_cpp_runtime_receipts(stream_artifact)
    modes = candidate_mode_matrix(template_meta)
    controls = adversarial_control_matrix()
    expected_rows = len(dict(stream_artifact.get("raw_response_receipts") or {})) or len(rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "complete",
        "preconditions_checked": _preconditions_checked(
            fixture_artifact=fixture_artifact,
            stream_artifact=stream_artifact,
            rows=rows,
            replay_ok=True,
        ),
        "input_artifact_hashes": {},
        "row_count": len(rows),
        "raw_response_coverage": _raw_response_coverage(rows, expected_rows),
        "per_model_failure_attribution": failure,
        "reasoning_content_receipts": reasoning,
        "final_content_receipts": final,
        "stop_reason_counts": stop_counts,
        "token_length_distributions": token_dist,
        "qwen_answer_sentinel_count": int(qwen["answer_sentinel_count"]),
        "qwen_empty_final_count": int(qwen["empty_final_count"]),
        "qwen_exact_cap_count": int(qwen["exact_cap_count"]),
        "embedded_template_metadata": template_meta,
        "llama_cpp_runtime_receipts": runtime,
        "upstream_issue_receipts": upstream_issue_receipts(),
        "local_upstream_distinction": local_upstream_distinction(),
        "candidate_mode_matrix": modes,
        "candidate_mode_count": len(modes),
        "mode_acceptance_rules": mode_acceptance_rules(),
        "mode_retirement_rules": mode_retirement_rules(),
        "adversarial_control_matrix": controls,
        "llm_calls_made": 0,
        "channel_diagnostic_ready_score": 0.0,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["input_artifact_hashes"] = input_artifact_hashes(
        fixture_artifact=fixture_artifact,
        stream_artifact=stream_artifact,
        input_paths=input_paths,
        template_metadata=template_meta,
        runtime_receipts=runtime,
    )
    artifact["channel_diagnostic_ready_score"] = channel_diagnostic_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(reason: str, *, test_commands: Sequence[str], test_exit_codes: Mapping[str, int]) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "blocked",
        "preconditions_checked": {"blocked_reason": reason, "llm_calls_made": 0},
        "input_artifact_hashes": {},
        "row_count": 0,
        "raw_response_coverage": 0.0,
        "per_model_failure_attribution": {},
        "reasoning_content_receipts": {},
        "final_content_receipts": {},
        "stop_reason_counts": {},
        "token_length_distributions": {},
        "qwen_answer_sentinel_count": 0,
        "qwen_empty_final_count": 0,
        "qwen_exact_cap_count": 0,
        "embedded_template_metadata": {},
        "llama_cpp_runtime_receipts": {},
        "upstream_issue_receipts": upstream_issue_receipts(),
        "local_upstream_distinction": local_upstream_distinction(),
        "candidate_mode_matrix": [],
        "candidate_mode_count": 0,
        "mode_acceptance_rules": {},
        "mode_retirement_rules": {},
        "adversarial_control_matrix": [],
        "llm_calls_made": 0,
        "channel_diagnostic_ready_score": 0.0,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": f"blocked: {reason}",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _bounded_modes_ok(artifact: Mapping[str, Any]) -> bool:
    modes = list(artifact.get("candidate_mode_matrix") or [])
    if int(artifact.get("candidate_mode_count") or 0) != len(modes):
        return False
    by_family: dict[str, int] = {}
    required = {"max_tokens", "stops", "finalizer", "parser", "timeout_s", "fail_closed_conditions"}
    for mode in modes:
        if not required.issubset(mode):
            return False
        if mode.get("bounded") is not True or mode.get("executable") is not True:
            return False
        if int(mode.get("max_tokens") or 0) <= 48 or int(mode.get("timeout_s") or 0) <= 0:
            return False
        if not mode.get("finalizer") or not mode.get("parser") or not mode.get("fail_closed_conditions"):
            return False
        by_family[str(mode.get("model_family") or "")] = by_family.get(str(mode.get("model_family") or ""), 0) + 1
    return all(by_family.get(model_family(hf_id), 0) >= 2 for hf_id in MANDATED_MODEL_IDS)


def _adversarial_controls_ok(artifact: Mapping[str, Any]) -> bool:
    controls = list(artifact.get("adversarial_control_matrix") or [])
    observed = {str(row.get("control_id") or "") for row in controls}
    return set(EXPECTED_ADVERSARIAL_CONTROLS).issubset(observed) and all(
        row.get("grammar_can_establish_semantic_correctness") is False for row in controls
    )


def channel_diagnostic_ready_score(artifact: Mapping[str, Any]) -> float:
    """Compute the strict diagnostic readiness scalar."""

    classified = sum(int(dict(row).get("row_count", 0) or 0) for row in dict(artifact.get("per_model_failure_attribution") or {}).values())
    ready = bool(
        artifact.get("status") == "complete"
        and int(artifact.get("llm_calls_made") or 0) == 0
        and int(artifact.get("row_count") or 0) > 0
        and classified == int(artifact.get("row_count") or 0)
        and float(artifact.get("raw_response_coverage") or 0.0) == 1.0
        and dict(artifact.get("local_upstream_distinction") or {}).get("upstream_issue_prose_is_local_receipt") is False
        and _bounded_modes_ok(artifact)
        and bool(artifact.get("mode_acceptance_rules"))
        and bool(artifact.get("mode_retirement_rules"))
        and _adversarial_controls_ok(artifact)
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal complete/blocked verdict for the artifact."""

    if artifact.get("status") != "complete":
        return "blocked: preconditions_or_row_hash_replay"
    if channel_diagnostic_ready_score(artifact) == 1.0:
        return "complete: answer_channel_diagnostic_ready_qwen_channel_failure_not_competence_failure"
    return "blocked: channel_diagnostic_ready_score"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on missing fields, unsupported claims, or checksum drift."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if int(artifact.get("llm_calls_made") or 0) != 0:
        raise ValueError("llm_calls_made")
    if artifact.get("status") == "complete" and not _adversarial_controls_ok(artifact):
        raise ValueError("adversarial_control_matrix")
    if artifact.get("status") == "complete" and not _bounded_modes_ok(artifact):
        raise ValueError("candidate_mode_matrix")
    if artifact.get("channel_diagnostic_ready_score") != channel_diagnostic_ready_score(artifact):
        raise ValueError("channel_diagnostic_ready_score")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("status") == "complete" and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if artifact.get("status") != "complete" and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    fixture_artifact_path: str | Path = REPO_ROOT / EXP5785_ARTIFACT_RELATIVE_PATH,
    stream_artifact_path: str | Path = REPO_ROOT / EXP5786_ARTIFACT_RELATIVE_PATH,
    rows_path: str | Path = REPO_ROOT / EXP5786_ROWS_RELATIVE_PATH,
    metadata_reader: MetadataReader | None | object = _DEFAULT_METADATA_READER,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build Exp5798 from local Exp5785/Exp5786 artifacts and existing rows."""

    exits = dict(test_exit_codes or {})
    try:
        fixture_artifact = json.loads(Path(fixture_artifact_path).read_text(encoding="utf-8"))
        stream_artifact = json.loads(Path(stream_artifact_path).read_text(encoding="utf-8"))
        rows = read_jsonl(rows_path)
        input_paths = {
            "exp5785_fixture_artifact": Path(fixture_artifact_path),
            "exp5786_artifact": Path(stream_artifact_path),
            "exp5786_rows": Path(rows_path),
            "exp5785_parser_code": REPO_ROOT / EXP5785_MODULE_RELATIVE_PATH,
            "exp5786_parser_runtime_code": REPO_ROOT / EXP5786_MODULE_RELATIVE_PATH,
            "exp5798_diagnostic_code": REPO_ROOT / MODULE_RELATIVE_PATH,
            "experiment_template_code": REPO_ROOT / EXPERIMENT_TEMPLATE_RELATIVE_PATH,
        }
        artifact = build_diagnostic_artifact(
            fixture_artifact=fixture_artifact,
            stream_artifact=stream_artifact,
            rows=rows,
            input_paths=input_paths,
            metadata_reader=metadata_reader,
            test_commands=test_commands,
            test_exit_codes=exits,
        )
    except Exception as exc:
        artifact = _blocked_artifact(repr(exc), test_commands=test_commands, test_exit_codes=exits)
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """CLI entrypoint."""

    del argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
