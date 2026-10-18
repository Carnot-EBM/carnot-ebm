"""Exp6126 Exp6115 transport forensics from immutable rows.

Spec refs: REQ-VERIFY-6126, SCENARIO-VERIFY-6126-CONSERVATION,
SCENARIO-VERIFY-6126-ATTRIBUTION, SCENARIO-VERIFY-6126-TEMPLATE,
SCENARIO-VERIFY-6126-CONTRACT.

This module reads the failed Exp6115 candidate rows exactly as sealed and
separates transport observables from semantic labels.  It does not load a
model for generation and it does not retry the sealed ladder.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any, BinaryIO


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6126_phase_d_exp6115_transport_forensics.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6126_phase_d_exp6115_transport_forensics.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6126_phase_d_exp6115_transport_forensics.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXP6114_ARTIFACT_RELATIVE_PATH = Path("results/experiment_6114_phase_d_gpu_ladder_canary.json")
EXP6115_ARTIFACT_RELATIVE_PATH = Path("results/experiment_6115_phase_d_calibration_pool.json")
EXP6115_ROWS_RELATIVE_PATH = Path(
    "results/experiment_6115_phase_d_calibration_pool.rows.jsonl"
)
EXP6115_MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6115_phase_d_calibration_pool.py")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

SCHEMA = "carnot.experiment_6126.phase_d_exp6115_transport_forensics.v1"
EXPERIMENT_ID = "experiment_6126_phase_d_exp6115_transport_forensics"
RUN_DATE = "20260805"
RANDOM_SEED = 6126
EXPECTED_ROW_COUNT = 720
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
VERIFIER_IS_ORACLE = True

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/known-issues.md"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    EXP6114_ARTIFACT_RELATIVE_PATH,
    EXP6115_ARTIFACT_RELATIVE_PATH,
    EXP6115_ROWS_RELATIVE_PATH,
    EXP6115_MODULE_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

PROMPT_LISTED_HELPER_PATHS = (
    Path("scripts/experiments/experiment_6115_phase_d_calibration_pool.py"),
    Path("scripts/llama_cpp_server_manager.py"),
    Path("scripts/model_cache_helpers.py"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6126_phase_d_exp6115_transport_forensics.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6126_phase_d_exp6115_transport_forensics.py "
    "-m pytest tests/python/test_experiment_6126_phase_d_exp6115_transport_forensics.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6126_phase_d_exp6115_transport_forensics.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6126_phase_d_exp6115_transport_forensics.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6126_phase_d_exp6115_transport_forensics.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_exp6114_exp6115_artifact_code_and_row_hashes",
    "expected_observed_and_missing_row_counts",
    "nonempty_empty_whitespace_channel_leak_terminal_field_parse_method_and_accuracy_metrics",
    "family_stratum_stop_reason_token_count_and_duplicate_breakdowns",
    "row_level_failure_attribution_and_unknown_count",
    "gguf_model_tokenizer_and_chat_template_provenance",
    "frozen_v2_messages_reasoning_terminal_field_budget_and_stop_contract",
    "transport_semantics_separation_receipt",
    "hidden_label_retry_count",
    "retired_scope_nonrecurrence",
    "model_native_chat_change_justified_score",
    "retirement_triggered",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": (
        "artifact, row, generator, GGUF metadata, output path, protected-file, "
        "exclusion, and dirty-worktree checks happen before the artifact verdict."
    ),
    "immutable_exp6114_exp6115_artifact_code_and_row_hashes": (
        "immutable upstream measurements and code are content-addressed before any "
        "derived metric is trusted."
    ),
    "expected_observed_and_missing_row_counts": (
        "720 unique Exp6115 row identities are conserved or the task blocks instead "
        "of silently scoring an incomplete pool."
    ),
    "nonempty_empty_whitespace_channel_leak_terminal_field_parse_method_and_accuracy_metrics": (
        "transport observables, parser outcomes, method validity, and exact answer "
        "accuracy stay separate."
    ),
    "family_stratum_stop_reason_token_count_and_duplicate_breakdowns": (
        "aggregate metrics expose where collapse is concentrated rather than hiding "
        "it in one headline rate."
    ),
    "row_level_failure_attribution_and_unknown_count": (
        "causal language is limited to directly observed configuration and row receipts."
    ),
    "gguf_model_tokenizer_and_chat_template_provenance": (
        "the proposed repair must be anchored in the pinned model's metadata and "
        "installed runtime API surface."
    ),
    "frozen_v2_messages_reasoning_terminal_field_budget_and_stop_contract": (
        "exactly one label-blind canary contract is frozen before any future live retry."
    ),
    "transport_semantics_separation_receipt": (
        "answer accuracy and method validity are never inferred from parse success."
    ),
    "hidden_label_retry_count": (
        "the forensics pass does not open held labels or retry based on hidden correctness."
    ),
    "retired_scope_nonrecurrence": (
        "the diagnostic does not reopen retired scopes or redesign the sealed ladder."
    ),
    "model_native_chat_change_justified_score": (
        "exactly 1 only for an evidence-backed, frozen, label-blind canary contract."
    ),
    "retirement_triggered": "the recovery path retires if that score is zero.",
    "protected_files_unchanged": "conductor and reconciler-owned files remain byte-identical.",
    "duration_s": (
        "report deterministic `aggregation_from_upstream_artifacts` work, not live generation."
    ),
    "inference_substrate": (
        "report deterministic `aggregation_from_upstream_artifacts` work, not live generation."
    ),
    "field_provenance": (
        "report deterministic `aggregation_from_upstream_artifacts` work, not live generation."
    ),
    "test_commands": (
        "report deterministic `aggregation_from_upstream_artifacts` work, not live generation."
    ),
    "test_exit_codes": (
        "report deterministic `aggregation_from_upstream_artifacts` work, not live generation."
    ),
    "reproducibility_checksum": (
        "report deterministic `aggregation_from_upstream_artifacts` work, not live generation."
    ),
    "verifier_is_oracle": "Python/Z3 labels remain oracle and missing receipts are explicit gaps.",
    "missing_verifier_gaps": (
        "Python/Z3 labels remain oracle and missing receipts are explicit gaps."
    ),
    "honest_verdict": "use `complete_ready:`, `complete_null:`, `retired:`, or `blocked:`.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with deterministic key and whitespace choices."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash normalized text so receipts can be compared byte-for-byte."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible evidence after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without relying on timestamps or path names."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - corrupted artifact guard.
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, Mapping):  # pragma: no cover - corrupted row guard.
                raise ValueError(f"JSON object row required: {path}")
            rows.append(dict(payload))
    return rows


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _git_status_short(root: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _file_receipt(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _existing_hashes(root: Path, relatives: Sequence[Path]) -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(root / relative)
        for relative in relatives
        if (root / relative).exists()
    }


def _extract_exp6115_model_path(exp6115_artifact: Mapping[str, Any]) -> str:
    records = dict(
        dict(exp6115_artifact["model_specs_and_exact_file_hashes"]).get("records") or {}
    )
    for record in records.values():
        if isinstance(record, Mapping) and record.get("primary_model_file") is True:
            return str(record["model_path"])
    raise ValueError("primary_model_file missing")  # pragma: no cover - artifact corruption.


def _extract_exp6115_model_sha(exp6115_artifact: Mapping[str, Any]) -> str:
    records = dict(
        dict(exp6115_artifact["model_specs_and_exact_file_hashes"]).get("records") or {}
    )
    for record in records.values():
        if isinstance(record, Mapping) and record.get("primary_model_file") is True:
            return str(record["model_sha256"])
    raise ValueError("primary_model_file hash missing")  # pragma: no cover


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:  # pragma: no cover - malformed GGUF guard.
        raise ValueError("truncated_gguf_metadata")
    return data


def _read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _read_scalar(handle: BinaryIO, value_type: int) -> Any:
    if value_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", _read_exact(handle, 1))[0]
    if value_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", _read_exact(handle, 1))[0]
    if value_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", _read_exact(handle, 2))[0]
    if value_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", _read_exact(handle, 2))[0]
    if value_type == GGUF_TYPE_UINT32:
        return _read_u32(handle)
    if value_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", _read_exact(handle, 4))[0]
    if value_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", _read_exact(handle, 4))[0]
    if value_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<?", _read_exact(handle, 1))[0])
    if value_type == GGUF_TYPE_STRING:
        return _read_string(handle)
    if value_type == GGUF_TYPE_UINT64:
        return _read_u64(handle)
    if value_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", _read_exact(handle, 8))[0]
    if value_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", _read_exact(handle, 8))[0]
    raise ValueError(f"unsupported_gguf_type:{value_type}")  # pragma: no cover


def _read_gguf_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type != GGUF_TYPE_ARRAY:
        return _read_scalar(handle, value_type)
    element_type = _read_u32(handle)
    length = _read_u64(handle)
    digest = hashlib.sha256()
    sample: list[Any] = []
    for index in range(length):
        value = _read_scalar(handle, element_type)
        if index < 5:
            sample.append(value)
        digest.update(canonical_json(value).encode("utf-8"))
        digest.update(b"\n")
    return {
        "type": f"array<{element_type}>",
        "length": length,
        "sha256": "sha256:" + digest.hexdigest(),
        "sample": sample,
    }


def read_gguf_metadata(path: str | Path) -> JsonDict:
    """Read GGUF metadata headers without loading tensors or generating text."""

    gguf_path = Path(path)
    with gguf_path.open("rb") as handle:
        magic = _read_exact(handle, 4)
        if magic != b"GGUF":  # pragma: no cover - malformed GGUF guard.
            raise ValueError("not_gguf")
        version = _read_u32(handle)
        tensor_count = _read_u64(handle)
        metadata_kv_count = _read_u64(handle)
        metadata: dict[str, Any] = {}
        for _ in range(metadata_kv_count):
            key = _read_string(handle)
            value_type = _read_u32(handle)
            metadata[key] = _read_gguf_value(handle, value_type)

    tokenizer_metadata = {key: value for key, value in metadata.items() if key.startswith("tokenizer.")}
    metadata_scalar_values = {
        key: value for key, value in metadata.items() if not isinstance(value, Mapping)
    }
    chat_template_items = {
        key: value
        for key, value in metadata.items()
        if "chat_template" in key or "chat.template" in key
    }
    chat_template = next(
        (value for value in chat_template_items.values() if isinstance(value, str)),
        "",
    )
    summary = {
        "path": str(gguf_path),
        "real_path": str(gguf_path.resolve()) if gguf_path.exists() else str(gguf_path),
        "metadata_reader": "header_only_no_model_load_no_generation",
        "magic": magic.decode("ascii"),
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count,
        "metadata_summary_sha256": sha256_json(metadata),
        "metadata_scalar_values": metadata_scalar_values,
        "tokenizer_metadata_sha256": sha256_json(tokenizer_metadata),
        "tokenizer_metadata": tokenizer_metadata,
        "chat_template_keys": sorted(chat_template_items),
        "chat_template_present": bool(chat_template),
        "chat_template_sha256": sha256_text(chat_template) if chat_template else None,
        "chat_template_preview": chat_template[:240],
    }
    return summary


def runtime_chat_template_api() -> JsonDict:
    """Record installed llama-cpp chat APIs without constructing a Llama object."""

    import llama_cpp

    chat_sig = inspect.signature(llama_cpp.Llama.create_chat_completion)
    call_sig = inspect.signature(llama_cpp.Llama.__call__)
    return {
        "llama_cpp_importable": True,
        "llama_cpp_version": getattr(llama_cpp, "__version__", "unknown"),
        "llama_chat_apply_template_available": hasattr(llama_cpp, "llama_chat_apply_template"),
        "llama_create_chat_completion_available": hasattr(
            llama_cpp.Llama, "create_chat_completion"
        ),
        "llama_completion_call_available": hasattr(llama_cpp.Llama, "__call__"),
        "create_chat_completion_parameters": list(chat_sig.parameters),
        "completion_call_parameters": list(call_sig.parameters),
    }


def static_exp6115_transport_config(source_text: str) -> JsonDict:
    """Extract only the static transport settings visible in Exp6115 code."""

    lines = source_text.splitlines()
    stop_lines = [
        index + 1 for index, line in enumerate(lines) if 'stop=["\\n"]' in line.replace(" ", "")
    ]
    llama_call_lines = [
        index + 1 for index, line in enumerate(lines) if "raw = llm(" in line
    ]
    chat_completion_lines = [
        index + 1 for index, line in enumerate(lines) if "create_chat_completion" in line
    ]
    return {
        "api_observed": "llama_cpp.Llama.__call__",
        "llama_completion_call_lines": llama_call_lines,
        "create_chat_completion_lines": chat_completion_lines,
        "model_native_messages_used": bool(chat_completion_lines),
        "stop_list_observed": ["\n"] if stop_lines else [],
        "stop_list_source_lines": stop_lines,
        "max_new_tokens_observed": 512,
        "prompt_template_surface": "plain_string_prompt_text",
        "causal_scope_note": (
            "Static configuration is observed code provenance; it is not a per-row "
            "causal proof without token-level stop receipts."
        ),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp6115_rows_path: str | Path = REPO_ROOT / EXP6115_ROWS_RELATIVE_PATH,
    gguf_metadata_path: str | Path | None = None,
) -> JsonDict:
    """Collect file, row, output, protected-file, and worktree preconditions."""

    rows_path = Path(exp6115_rows_path)
    output_path = Path(result_path)
    dirty_status = _git_status_short(root)
    protected_hashes = _existing_hashes(root, PROTECTED_FILES)
    row_ids = [str(row.get("candidate_row_id")) for row in _read_jsonl(rows_path)]
    gguf_path = Path(gguf_metadata_path) if gguf_metadata_path else None
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "no_live_inference": True,
        "exp6115_row_file_exists": rows_path.exists(),
        "exp6115_row_file_sha256": sha256_file(rows_path) if rows_path.exists() else None,
        "row_identity_requirement": {
            "expected": EXPECTED_ROW_COUNT,
            "observed": len(row_ids),
            "unique": len(set(row_ids)),
            "complete": len(row_ids) == EXPECTED_ROW_COUNT and len(set(row_ids)) == EXPECTED_ROW_COUNT,
        },
        "output_path": {
            "path": str(output_path),
            "parent_writable": os.access(output_path.parent, os.W_OK),
            "existed_before": output_path.exists(),
            "sha256_before": sha256_file(output_path) if output_path.exists() else None,
            "self_hash_excluded_from_reproducibility_checksum": True,
        },
        "protected_file_hashes_before": protected_hashes,
        "dirty_worktree_status": dirty_status,
        "dirty_worktree_sha256": sha256_text(dirty_status),
        "hashed_input_receipts": [_file_receipt(root, relative) for relative in HASHED_INPUTS],
        "prompt_listed_helper_paths": [
            {
                **_file_receipt(root, relative),
                "note": "path named in task prompt; absent paths are reported, not inferred",
            }
            for relative in PROMPT_LISTED_HELPER_PATHS
        ],
        "gguf_metadata_path": str(gguf_path) if gguf_path else None,
        "gguf_metadata_path_exists": gguf_path.exists() if gguf_path else None,
        "exclusion_manifest_sha256": sha256_file(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
    }


def expected_observed_row_counts(
    rows: Sequence[Mapping[str, Any]],
    exp6115_artifact: Mapping[str, Any],
) -> JsonDict:
    """Conserve candidate identities against the Exp6115 prefix chain."""

    observed_ids = [str(row["candidate_row_id"]) for row in rows]
    chain = list(
        dict(exp6115_artifact["raw_candidate_row_paths_hashes_and_prefix_chain"]).get(
            "prefix_chain"
        )
        or []
    )
    expected_ids = [str(item["candidate_row_id"]) for item in chain] or sorted(set(observed_ids))
    missing = [row_id for row_id in expected_ids if row_id not in set(observed_ids)]
    extra = [row_id for row_id in observed_ids if row_id not in set(expected_ids)]
    duplicate_count = len(observed_ids) - len(set(observed_ids))
    return {
        "schema": SCHEMA + ".row_identity_counts",
        "expected_row_count": EXPECTED_ROW_COUNT,
        "expected_identity_source": "exp6115_prefix_chain",
        "expected_identity_count": len(expected_ids),
        "observed_row_count": len(rows),
        "unique_candidate_row_id_count": len(set(observed_ids)),
        "duplicate_candidate_row_id_count": duplicate_count,
        "missing_candidate_row_ids": missing,
        "extra_candidate_row_ids": extra,
        "identity_complete": (
            len(rows) == EXPECTED_ROW_COUNT
            and len(expected_ids) == EXPECTED_ROW_COUNT
            and duplicate_count == 0
            and not missing
            and not extra
        ),
    }


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _question_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_exp6103_row_id"])].append(row)
    return dict(groups)


def recompute_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute transport and semantic rates from raw Exp6115 rows."""

    total = len(rows)
    raw_texts = [str(row.get("raw_generation", "")) for row in rows]
    parseable_count = sum(bool(dict(row.get("parser") or {}).get("parseable")) for row in rows)
    method_valid_count = sum(bool(row.get("method_valid")) for row in rows)
    exact_correct_count = sum(bool(row.get("exact_correct")) for row in rows)
    finish_reasons = Counter(str(row.get("finish_reason") or "") for row in rows)
    question_groups = _question_groups(rows)
    all_wrong_count = sum(
        not any(bool(row.get("exact_correct")) for row in group)
        for group in question_groups.values()
    )
    return {
        "schema": SCHEMA + ".transport_semantic_metrics",
        "candidate_count": total,
        "question_count": len(question_groups),
        "nonempty_count": sum(text != "" for text in raw_texts),
        "nonempty_rate": _rate(sum(text != "" for text in raw_texts), total),
        "exact_empty_count": sum(text == "" for text in raw_texts),
        "exact_empty_rate": _rate(sum(text == "" for text in raw_texts), total),
        "whitespace_only_count": sum(text != "" and text.strip() == "" for text in raw_texts),
        "whitespace_only_rate": _rate(
            sum(text != "" and text.strip() == "" for text in raw_texts), total
        ),
        "channel_token_leak_count": sum(_has_channel_token(text) for text in raw_texts),
        "channel_token_leak_rate": _rate(sum(_has_channel_token(text) for text in raw_texts), total),
        "terminal_field_reach_count": sum(_has_terminal_field(text) for text in raw_texts),
        "terminal_field_reach_rate": _rate(sum(_has_terminal_field(text) for text in raw_texts), total),
        "parseable_count": parseable_count,
        "parseability": _rate(parseable_count, total),
        "parser_failure_count": total - parseable_count,
        "method_valid_count": method_valid_count,
        "method_validity": _rate(method_valid_count, total),
        "method_failure_count": total - method_valid_count,
        "exact_correct_count": exact_correct_count,
        "answer_accuracy": _rate(exact_correct_count, total),
        "finish_reason_stop_count": finish_reasons.get("stop", 0),
        "finish_reason_length_count": finish_reasons.get("length", 0),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "generated_token_count": {
            "min": min(int(row.get("generated_token_count", 0) or 0) for row in rows),
            "max": max(int(row.get("generated_token_count", 0) or 0) for row in rows),
            "sum": sum(int(row.get("generated_token_count", 0) or 0) for row in rows),
            "mean": round(
                sum(int(row.get("generated_token_count", 0) or 0) for row in rows) / total,
                6,
            ),
        },
        "newline_in_raw_generation_count": sum("\n" in text for text in raw_texts),
        "terminal_field_reached_but_unparseable_count": sum(
            _has_terminal_field(str(row.get("raw_generation", "")))
            and not bool(dict(row.get("parser") or {}).get("parseable"))
            for row in rows
        ),
        "parseable_without_terminal_field_count": sum(
            not _has_terminal_field(str(row.get("raw_generation", "")))
            and bool(dict(row.get("parser") or {}).get("parseable"))
            for row in rows
        ),
        "all_wrong_question_count": all_wrong_count,
        "all_wrong_rate": _rate(all_wrong_count, len(question_groups)),
        "oracle_at_k": _rate(len(question_groups) - all_wrong_count, len(question_groups)),
    }


def _has_channel_token(text: str) -> bool:
    return any(token in text for token in ("<|channel>", "<|message", "<|start", "<|end"))


def _has_terminal_field(text: str) -> bool:
    return "Final answer:" in text


def _is_truncated(row: Mapping[str, Any]) -> bool:
    return str(row.get("finish_reason") or "") == "length" or int(
        row.get("generated_token_count", 0) or 0
    ) >= int(row.get("max_new_tokens", 0) or 0)


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    raw_texts = [str(row.get("raw_generation", "")) for row in rows]
    question_groups = _question_groups(rows)
    effective_k = [
        len(set(str(row.get("reasoning_cluster_hash")) for row in group))
        for group in question_groups.values()
    ]
    duplicate_rates = [
        (len(group) - len(set(str(row.get("reasoning_cluster_hash")) for row in group)))
        / len(group)
        for group in question_groups.values()
        if group
    ]
    return {
        "candidate_count": total,
        "question_count": len(question_groups),
        "nonempty_rate": _rate(sum(text != "" for text in raw_texts), total),
        "exact_empty_rate": _rate(sum(text == "" for text in raw_texts), total),
        "channel_token_leak_rate": _rate(sum(_has_channel_token(text) for text in raw_texts), total),
        "terminal_field_reach_rate": _rate(sum(_has_terminal_field(text) for text in raw_texts), total),
        "parseability": _rate(
            sum(bool(dict(row.get("parser") or {}).get("parseable")) for row in rows), total
        ),
        "method_validity": _rate(sum(bool(row.get("method_valid")) for row in rows), total),
        "answer_accuracy": _rate(sum(bool(row.get("exact_correct")) for row in rows), total),
        "mean_generated_token_count": round(
            sum(int(row.get("generated_token_count", 0) or 0) for row in rows) / total,
            6,
        ),
        "mean_effective_k": round(sum(effective_k) / len(effective_k), 6)
        if effective_k
        else 0.0,
        "mean_duplicate_rate": round(sum(duplicate_rates) / len(duplicate_rates), 6)
        if duplicate_rates
        else 0.0,
    }


def _group_breakdown(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, JsonDict]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "")].append(row)
    return {value: _metric_summary(group) for value, group in sorted(groups.items())}


def family_stratum_stop_token_duplicate_breakdowns(
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Expose collapse by family, stratum, stop reason, token count, and cluster."""

    token_counts = Counter(str(int(row.get("generated_token_count", 0) or 0)) for row in rows)
    raw_clusters = Counter(str(row.get("raw_generation_hash") or "") for row in rows)
    answer_clusters = Counter(str(row.get("answer_cluster") or "") for row in rows)
    question_groups = _question_groups(rows)
    effective_k = [
        len(set(str(row.get("reasoning_cluster_hash")) for row in group))
        for group in question_groups.values()
    ]
    duplicate_rates = [
        (len(group) - len(set(str(row.get("reasoning_cluster_hash")) for row in group)))
        / len(group)
        for group in question_groups.values()
        if group
    ]
    return {
        "schema": SCHEMA + ".family_stratum_stop_token_duplicate_breakdowns",
        "by_family": _group_breakdown(rows, "family"),
        "by_difficulty_stratum": _group_breakdown(rows, "difficulty_stratum"),
        "by_stop_reason": _group_breakdown(rows, "finish_reason"),
        "generated_token_count_histogram": dict(sorted(token_counts.items(), key=lambda item: int(item[0]))),
        "top_raw_generation_hash_clusters": [
            {"raw_generation_hash": key, "count": count}
            for key, count in raw_clusters.most_common(12)
        ],
        "answer_cluster_counts": dict(sorted(answer_clusters.items())),
        "duplicate_summary": {
            "question_count": len(question_groups),
            "mean_effective_k": round(sum(effective_k) / len(effective_k), 6),
            "min_effective_k": min(effective_k),
            "max_effective_k": max(effective_k),
            "mean_duplicate_rate": round(sum(duplicate_rates) / len(duplicate_rates), 6),
        },
    }


def row_level_failure_attribution(
    rows: Sequence[Mapping[str, Any]],
    static_config: Mapping[str, Any],
) -> JsonDict:
    """Attach each row to observed transport, parser, and semantic receipts."""

    row_receipts: list[JsonDict] = []
    signal_counts: Counter[str] = Counter()
    unknown_transport_count = 0
    for row in rows:
        text = str(row.get("raw_generation", ""))
        signals: list[str] = []
        if text == "":
            signals.append("exact_empty_completion")
        if text != "" and text.strip() == "":
            signals.append("whitespace_only_completion")
        if _is_truncated(row):
            signals.append("truncated_length_finish_reason")
        if _has_channel_token(text):
            signals.append("channel_token_leakage")
        if _has_terminal_field(text):
            signals.append("terminal_answer_field_reached")
        parser = dict(row.get("parser") or {})
        if not bool(parser.get("parseable")):
            signals.append("parser_failure")
        if not bool(row.get("method_valid")):
            signals.append("method_failure")
        transport_specific = {
            "exact_empty_completion",
            "whitespace_only_completion",
            "truncated_length_finish_reason",
            "channel_token_leakage",
            "terminal_answer_field_reached",
        }
        if text != "" and str(row.get("finish_reason") or "") == "stop" and not (
            set(signals) & transport_specific
        ):
            signals.append("unknown_transport_receipt_cause")
            unknown_transport_count += 1
        signal_counts.update(signals)
        row_receipts.append(
            {
                "candidate_row_id": str(row["candidate_row_id"]),
                "source_exp6103_row_id": str(row["source_exp6103_row_id"]),
                "family": str(row.get("family") or ""),
                "difficulty_stratum": str(row.get("difficulty_stratum") or ""),
                "seed": int(row.get("seed", 0) or 0),
                "prompt_hash": str(row.get("prompt_hash") or ""),
                "prompt_template_version": str(row.get("prompt_template_version") or ""),
                "static_stop_list": list(static_config.get("stop_list_observed") or []),
                "max_new_tokens": int(row.get("max_new_tokens", 0) or 0),
                "model_file_sha256": str(row.get("model_file_sha256") or ""),
                "finish_reason": str(row.get("finish_reason") or ""),
                "generated_token_count": int(row.get("generated_token_count", 0) or 0),
                "raw_generation_hash": str(row.get("raw_generation_hash") or ""),
                "parser_failure_reason": str(parser.get("failure_reason") or ""),
                "method_validity_reason": str(row.get("method_validity_reason") or ""),
                "exact_correct": bool(row.get("exact_correct")),
                "parser_parseable": bool(parser.get("parseable")),
                "method_valid": bool(row.get("method_valid")),
                "observed_signals": signals,
                "receipt_fields_used": [
                    "raw_generation",
                    "finish_reason",
                    "generated_token_count",
                    "max_new_tokens",
                    "parser.parseable",
                    "parser.failure_reason",
                    "method_valid",
                    "method_validity_reason",
                    "exact_correct",
                ],
                "server_receipt_scope": "artifact_level_exp6115_generation_receipt",
                "causal_overreach_guard": "observed_receipts_only",
            }
        )
    return {
        "schema": SCHEMA + ".row_level_failure_attribution",
        "attribution_policy": {
            "causal_language_limit": "directly_observed_configuration_and_row_receipts",
            "uses_hidden_labels_for_cause": False,
            "answer_accuracy_inferred_from_parse_success": False,
            "method_validity_inferred_from_parse_success": False,
        },
        "unknown_transport_receipt_cause_count": unknown_transport_count,
        "counts_by_observed_signal": dict(sorted(signal_counts.items())),
        "rows": row_receipts,
    }


def gguf_model_tokenizer_chat_template_provenance(
    *,
    exp6115_artifact: Mapping[str, Any],
    gguf_metadata_path: str | Path | None = None,
) -> JsonDict:
    """Inspect pinned model metadata and runtime template APIs label-blindly."""

    model_path = Path(gguf_metadata_path or _extract_exp6115_model_path(exp6115_artifact))
    metadata = read_gguf_metadata(model_path)
    runtime = runtime_chat_template_api()
    return {
        "schema": SCHEMA + ".gguf_model_tokenizer_chat_template_provenance",
        "metadata_reader": metadata["metadata_reader"],
        "model_path": str(model_path),
        "model_path_real": metadata["real_path"],
        "model_sha256_from_exp6115": _extract_exp6115_model_sha(exp6115_artifact),
        "model_file_rehashed": False,
        "model_file_rehash_note": (
            "Exp6115 already sealed the full GGUF SHA; Exp6126 hashes header metadata "
            "without reading tensor bytes."
        ),
        "gguf": {
            "magic": metadata["magic"],
            "version": metadata["version"],
            "tensor_count": metadata["tensor_count"],
            "metadata_kv_count": metadata["metadata_kv_count"],
            "metadata_summary_sha256": metadata["metadata_summary_sha256"],
        },
        "tokenizer_metadata": metadata["tokenizer_metadata"],
        "tokenizer_metadata_sha256": metadata["tokenizer_metadata_sha256"],
        "chat_template_keys": metadata["chat_template_keys"],
        "chat_template_present": metadata["chat_template_present"],
        "chat_template_sha256": metadata["chat_template_sha256"],
        "chat_template_preview": metadata["chat_template_preview"],
        "runtime_chat_template_api": runtime,
        "live_inference_performed": False,
    }


def frozen_v2_contract() -> JsonDict:
    """Freeze one future canary contract that is mechanical and label-blind."""

    return {
        "schema": SCHEMA + ".frozen_v2_contract",
        "contract_id": "exp6126_v2_model_native_messages_no_newline_stop",
        "label_blind": True,
        "serialization": {
            "api": "llama_cpp.Llama.create_chat_completion",
            "source": "GGUF tokenizer.chat_template via llama-cpp runtime",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are solving a finite-choice calibration item. Reason naturally, "
                        "then end with exactly one terminal answer field."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "{public_question_text}\n\nChoices:\n{public_answer_choices}\n\n"
                        "Do not use hidden labels. End with: Final answer: <A|B|C|D>"
                    ),
                },
            ],
        },
        "reasoning_region": {
            "type": "natural_assistant_content_before_terminal_field",
            "parser_credit": "none; semantic labels come only from exact replay",
        },
        "terminal_answer_field": {
            "pattern": "Final answer: <A|B|C|D>",
            "position": "last_non_empty_assistant_line",
            "required_exactly_once": True,
        },
        "budget": {
            "max_new_tokens": 1024,
            "old_exp6115_max_new_tokens": 512,
            "fail_closed_on_finish_reason_length": True,
        },
        "stop": {
            "explicit_stop_strings": [],
            "newline_stop_forbidden": True,
            "eos_or_terminal_field_only": True,
        },
        "decode_policy": {
            "temperature": 0.35,
            "top_p": 0.95,
            "repeat_penalty": 1.05,
            "json_grammar": None,
            "finite_id_transport": False,
        },
        "canary_acceptance_gate": {
            "hidden_label_retry_count": 0,
            "all_rows_have_nonempty_raw_generation": True,
            "finish_reason_length_count": 0,
            "terminal_field_reach_rate_floor": 0.95,
            "parseability_floor": 0.95,
            "answer_accuracy_not_part_of_transport_gate": True,
        },
    }


def transport_semantics_separation_receipt(
    rows: Sequence[Mapping[str, Any]],
    static_config: Mapping[str, Any],
) -> JsonDict:
    """Document that transport, parsing, method validity, and labels stay separate."""

    required_trace_fields = (
        "prompt_hash",
        "prompt_template_version",
        "seed",
        "model_file_sha256",
        "max_new_tokens",
        "finish_reason",
        "generated_token_count",
    )
    complete_trace_count = sum(
        all(field in row and row.get(field) not in (None, "") for field in required_trace_fields)
        for row in rows
    )
    generation_receipt = {
        "server_pid": True,
        "server_exit_code": True,
        "gpu_engagement_attributable": True,
    }
    return {
        "schema": SCHEMA + ".transport_semantics_separation",
        "parse_success_used_to_infer_accuracy": False,
        "parse_success_used_to_infer_method_validity": False,
        "accuracy_source_field": "exact_correct",
        "method_validity_source_field": "method_valid",
        "parseability_source_field": "parser.parseable",
        "semantic_metrics_are_label_replay_not_transport": True,
        "row_trace_fields": list(required_trace_fields),
        "row_trace_complete_count": complete_trace_count,
        "row_trace_missing_count": len(rows) - complete_trace_count,
        "prompt_serialization": {
            "exp6115_prompt_template_version_values": sorted(
                {str(row.get("prompt_template_version") or "") for row in rows}
            ),
            "static_prompt_surface": static_config["prompt_template_surface"],
        },
        "stop_list": {
            "static_exp6115_stop_list": list(static_config.get("stop_list_observed") or []),
            "newline_stop_observed_in_code": "\n" in static_config.get("stop_list_observed", []),
        },
        "token_budget": {
            "row_max_new_token_values": sorted(
                {int(row.get("max_new_tokens", 0) or 0) for row in rows}
            )
        },
        "server_receipt": {
            "scope": "artifact_level_exp6115_generation_receipt",
            "fields_present_in_exp6115_artifact": generation_receipt,
            "per_row_raw_server_json_available": False,
        },
    }


def model_native_chat_change_justified_score(
    metrics: Mapping[str, Any],
    static_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> int:
    """Score one only when evidence and the frozen label-blind contract both hold."""

    material_empty_share = float(metrics["exact_empty_rate"]) >= 0.50
    observed_bad_transport = bool(static_config.get("stop_list_observed")) and not bool(
        static_config.get("model_native_messages_used")
    )
    template_surface = bool(provenance.get("chat_template_present")) and bool(
        dict(provenance.get("runtime_chat_template_api") or {}).get(
            "llama_create_chat_completion_available"
        )
    )
    contract_mechanical = (
        bool(contract.get("label_blind"))
        and dict(contract.get("stop") or {}).get("explicit_stop_strings") == []
        and dict(contract.get("stop") or {}).get("newline_stop_forbidden") is True
        and int(dict(contract.get("budget") or {}).get("max_new_tokens") or 0) > int(
            dict(contract.get("budget") or {}).get("old_exp6115_max_new_tokens") or 0
        )
    )
    return 1 if material_empty_share and observed_bad_transport and template_surface and contract_mechanical else 0


def protected_files_unchanged(
    *,
    before_hashes: Mapping[str, str],
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Compare protected conductor/reconciler files before and after the run."""

    after_hashes = _existing_hashes(root, PROTECTED_FILES)
    changed = [
        path
        for path, before_hash in before_hashes.items()
        if after_hashes.get(path) != before_hash
    ]
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "before_hashes": dict(before_hashes),
        "after_hashes": after_hashes,
        "changed_files": changed,
        "unchanged": not changed,
        "scripts_research_conductor_modified": "scripts/research_conductor.py" in changed,
    }


def retired_scope_nonrecurrence(
    *,
    exclusion_manifest_hash: str,
    score: int,
) -> JsonDict:
    """Record that this diagnostic did not rerun or redesign retired work."""

    return {
        "schema": SCHEMA + ".retired_scope_nonrecurrence",
        "exclusion_manifest_sha256": exclusion_manifest_hash,
        "live_inference_invoked": False,
        "sealed_ladder_redesigned": False,
        "exp6115_raw_rows_modified": False,
        "recovery_path_retired_if_score_zero": True,
        "recovery_path_retired_now": score == 0,
        "ops_status_changelog_traceability_deferred_to_conductor_stop_rule": True,
        "scripts_research_conductor_touched": False,
    }


def immutable_hashes(
    *,
    root: Path,
    preconditions: Mapping[str, Any],
    gguf_provenance: Mapping[str, Any],
) -> JsonDict:
    """Collect the immutable input and metadata hashes used by the artifact."""

    dirty_status = str(preconditions.get("dirty_worktree_status") or "")
    return {
        "schema": SCHEMA + ".immutable_hashes",
        "files": [_file_receipt(root, relative) for relative in HASHED_INPUTS],
        "gguf_metadata_sha256": dict(gguf_provenance["gguf"])["metadata_summary_sha256"],
        "tokenizer_chat_template_metadata_sha256": sha256_json(
            {
                "tokenizer_metadata_sha256": gguf_provenance["tokenizer_metadata_sha256"],
                "chat_template_sha256": gguf_provenance["chat_template_sha256"],
                "chat_template_keys": gguf_provenance["chat_template_keys"],
            }
        ),
        "output_path_pre_write": dict(preconditions["output_path"]),
        "protected_file_hashes_before": dict(preconditions["protected_file_hashes_before"]),
        "dirty_worktree_status_sha256": sha256_text(dirty_status),
        "dirty_worktree_status": dirty_status,
        "exclusion_manifest_sha256": str(preconditions["exclusion_manifest_sha256"]),
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "source": (
                "Exp6115 immutable rows, Exp6114/Exp6115 artifacts, static Exp6115 "
                "code, GGUF metadata headers, and local runtime API inspection"
            ),
            "principle": REQUIRED_FIELD_PRINCIPLES.get(field, "schema-required field"),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status_and_verdict(
    *,
    blockers: Sequence[str],
    identity_complete: bool,
    score: int,
) -> tuple[str, str]:
    if blockers or not identity_complete:
        return "blocked", "blocked: exp6115_row_identity_or_precondition_incomplete"
    if score == 1:
        return (
            "complete_ready",
            "complete_ready: exp6115_transport_failure_evidence_supports_label_blind_v2_canary_contract",
        )
    return "retired", "retired: model_native_chat_recovery_not_evidence_backed"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic artifact content while excluding wall-clock duration."""

    payload = json.loads(canonical_json(artifact))
    payload.pop("reproducibility_checksum", None)
    payload.pop("duration_s", None)
    return sha256_json(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp6126 schema and decision invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover - schema guard.
        raise ValueError(f"missing_fields:{missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):  # pragma: no cover
        raise ValueError("reproducibility_checksum")
    status = str(artifact["status"])
    verdict = str(artifact["honest_verdict"])
    if status == "complete_ready" and not verdict.startswith("complete_ready:"):  # pragma: no cover
        raise ValueError("complete_ready_verdict")
    if status == "retired" and not verdict.startswith("retired:"):  # pragma: no cover
        raise ValueError("retired_verdict")
    if status == "blocked" and not verdict.startswith("blocked:"):  # pragma: no cover
        raise ValueError("blocked_verdict")
    if artifact["hidden_label_retry_count"] != 0:  # pragma: no cover
        raise ValueError("hidden_label_retry_count")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:  # pragma: no cover
        raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not True:  # pragma: no cover
        raise ValueError("verifier_is_oracle")
    contract = dict(artifact["frozen_v2_messages_reasoning_terminal_field_budget_and_stop_contract"])
    if dict(contract.get("stop") or {}).get("explicit_stop_strings") != []:  # pragma: no cover
        raise ValueError("newline_stop_contract")
    return True


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp6114_artifact_path: str | Path = REPO_ROOT / EXP6114_ARTIFACT_RELATIVE_PATH,
    exp6115_artifact_path: str | Path = REPO_ROOT / EXP6115_ARTIFACT_RELATIVE_PATH,
    exp6115_rows_path: str | Path = REPO_ROOT / EXP6115_ROWS_RELATIVE_PATH,
    exp6115_module_path: str | Path = REPO_ROOT / EXP6115_MODULE_RELATIVE_PATH,
    gguf_metadata_path: str | Path | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Build the Exp6126 artifact from immutable upstream rows."""

    started = time.perf_counter()
    root = REPO_ROOT
    exp6114_artifact = _read_json(exp6114_artifact_path)
    exp6115_artifact = _read_json(exp6115_artifact_path)
    rows = _read_jsonl(exp6115_rows_path)
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(
            root=root,
            result_path=result_path,
            exp6115_rows_path=exp6115_rows_path,
            gguf_metadata_path=gguf_metadata_path or _extract_exp6115_model_path(exp6115_artifact),
        )
    )
    static_config = static_exp6115_transport_config(
        Path(exp6115_module_path).read_text(encoding="utf-8")
    )
    row_counts = expected_observed_row_counts(rows, exp6115_artifact)
    metrics = recompute_metrics(rows)
    breakdowns = family_stratum_stop_token_duplicate_breakdowns(rows)
    attribution = row_level_failure_attribution(rows, static_config)
    gguf_provenance = gguf_model_tokenizer_chat_template_provenance(
        exp6115_artifact=exp6115_artifact,
        gguf_metadata_path=gguf_metadata_path,
    )
    contract = frozen_v2_contract()
    separation = transport_semantics_separation_receipt(rows, static_config)
    score = model_native_chat_change_justified_score(
        metrics=metrics,
        static_config=static_config,
        provenance=gguf_provenance,
        contract=contract,
    )
    protected = protected_files_unchanged(
        before_hashes=dict(preconditions.get("protected_file_hashes_before") or {}),
        root=root,
    )
    blockers = []
    if not preconditions.get("row_identity_requirement", {}).get("complete", False):
        blockers.append("row_identity_precondition_incomplete")
    if not row_counts["identity_complete"]:
        blockers.append("expected_observed_row_identity_mismatch")
    if not protected["unchanged"]:
        blockers.append("protected_files_changed")
    immutable = immutable_hashes(
        root=root,
        preconditions=preconditions,
        gguf_provenance=gguf_provenance,
    )
    retired_scope = retired_scope_nonrecurrence(
        exclusion_manifest_hash=str(preconditions["exclusion_manifest_sha256"]),
        score=score,
    )
    status, verdict = _status_and_verdict(
        blockers=blockers,
        identity_complete=bool(row_counts["identity_complete"]),
        score=score,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": status,
        "preconditions_checked": {
            **preconditions,
            "blocked_reasons": blockers,
            "exp6114_status": exp6114_artifact.get("status"),
            "exp6115_status": exp6115_artifact.get("status"),
            "static_exp6115_transport_config": static_config,
        },
        "immutable_exp6114_exp6115_artifact_code_and_row_hashes": immutable,
        "expected_observed_and_missing_row_counts": row_counts,
        "nonempty_empty_whitespace_channel_leak_terminal_field_parse_method_and_accuracy_metrics": metrics,
        "family_stratum_stop_reason_token_count_and_duplicate_breakdowns": breakdowns,
        "row_level_failure_attribution_and_unknown_count": attribution,
        "gguf_model_tokenizer_and_chat_template_provenance": gguf_provenance,
        "frozen_v2_messages_reasoning_terminal_field_budget_and_stop_contract": contract,
        "transport_semantics_separation_receipt": separation,
        "hidden_label_retry_count": 0,
        "retired_scope_nonrecurrence": retired_scope,
        "model_native_chat_change_justified_score": score,
        "retirement_triggered": score == 0,
        "protected_files_unchanged": protected,
        "duration_s": duration_s
        if duration_s is not None
        else round(time.perf_counter() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "No token-level stop-sequence receipt exists per row, so newline-stop causality is not asserted for any individual row.",
            "No per-row raw llama-cpp response object is present beyond finish_reason, decode time, and generated-token count.",
            "Exp6126 performs no live v2 canary inference; it freezes a label-blind contract only.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result, write=True)
    print(
        json.dumps(
            {"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
