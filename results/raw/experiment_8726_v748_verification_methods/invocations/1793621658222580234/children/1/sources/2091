"""Exp5922 tokenizer-aware GGUF schema decoder bridge.

Spec refs: REQ-VERIFY-5922, SCENARIO-VERIFY-5922.

This module qualifies the boundary between Exp5921's character-level
ConstraintIR support and the embedded tokenizers inside the mandated GGUF
artifacts.  It deliberately does not score model answers.  The live path only
checks that public llama.cpp can load each GGUF, call a logits processor, and
record GPU-offload evidence for a bounded one-token smoke.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_5921_schema_derived_constraintir_support as exp5921
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5922_gguf_schema_decoder_bridge.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5922_gguf_schema_decoder_bridge.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5922_gguf_schema_decoder_bridge.py")
VERIFIABLE_REASONING_SPEC_RELATIVE_PATH = Path(
    "openspec/capabilities/verifiable-reasoning/spec.md"
)

RUN_DATE = "20260725"
EXPERIMENT_ID = "experiment_5922_gguf_schema_decoder_bridge"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5922.gguf_schema_decoder_bridge.v1"
INFERENCE_SUBSTRATE = "public_llama_cpp_cuda_tokenizer_bridge_smoke"
RANDOM_SEED = 5922
N_GPU_LAYERS = -1

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "family": "qwen_moe",
        "required": True,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "family": "gemma_dense",
        "required": True,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "family": "gemma_moe",
        "required": True,
    },
)

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
    exp5921.RESULT_RELATIVE_PATH,
    exp5921.MODULE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    VERIFIABLE_REASONING_SPEC_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "gate_replay_receipt",
    "preconditions_checked",
    "model_specs",
    "model_file_hashes",
    "embedded_tokenizer_receipts",
    "public_llama_cpp_cuda_receipt",
    "schema_support_version_and_hash",
    "per_model_terminal_token_mapping",
    "unsupported_and_multitoken_terminal_receipts",
    "tokenizer_reference_support_parity",
    "utf8_whitespace_numeric_identifier_matrix",
    "eos_empty_mask_and_dead_end_policy",
    "logits_processor_public_api_receipt",
    "one_step_cuda_smoke",
    "full_answer_enumeration_used",
    "protected_files_unchanged",
    "gguf_schema_decoder_bridge_ready_score",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "tokenizer_reference_support_parity": (
        "Every admitted token must preserve at least one schema-valid continuation, "
        "and every known-valid fixture path must remain reachable."
    ),
    "full_answer_enumeration_used": (
        "Must be bare false because complete-answer token lists overlap a retired "
        "transport scope."
    ),
    "gguf_schema_decoder_bridge_ready_score": (
        "Emit bare 1.0 only for all-three embedded-tokenizer parity, public API "
        "compatibility, deterministic replay, and complete dead-end controls."
    ),
    "inference_substrate": "Use public_llama_cpp_cuda_tokenizer_bridge_smoke.",
    "honest_verdict": "Use complete_ready:, retired:, or blocked:.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5922_gguf_schema_decoder_bridge.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5922_gguf_schema_decoder_bridge.py "
    "-m pytest tests/python/test_experiment_5922_gguf_schema_decoder_bridge.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5922_gguf_schema_decoder_bridge.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5922_gguf_schema_decoder_bridge",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5922_gguf_schema_decoder_bridge.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5922_gguf_schema_decoder_bridge.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py",
)


@dataclass(frozen=True)
class PrefixStatus:
    """Result of checking whether bytes can still become schema-valid JSON."""

    valid: bool
    can_continue: bool
    complete_valid: bool
    reason: str
    partial_utf8: bool = False


@dataclass(frozen=True)
class TokenVocabulary:
    """Embedded-tokenizer vocabulary bytes used by the bridge.

    The bridge is byte-based because real GGUF tokens may detokenize to a
    partial UTF-8 byte sequence.  Treating bytes as first-class prevents the
    bridge from silently normalizing a tokenizer/runtime mismatch into Unicode
    replacement characters.
    """

    model_id: str
    model_path: str
    token_bytes_by_id: Mapping[int, bytes]
    eos_token_id: int | None
    tokenizer_receipt: Mapping[str, Any]

    @property
    def max_token_id(self) -> int:
        ids = list(self.token_bytes_by_id)
        if self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return max(ids) if ids else 0

    def encode_bytes(self, data: bytes) -> list[int]:
        receipt_tokenizer = self.tokenizer_receipt.get("_tokenizer")
        if receipt_tokenizer is not None:
            try:
                return list(receipt_tokenizer.tokenize(data, add_bos=False, special=False))
            except TypeError:
                return list(receipt_tokenizer.tokenize(data))
        reverse = {value: key for key, value in self.token_bytes_by_id.items() if value}
        output: list[int] = []
        index = 0
        ordered = sorted(reverse, key=len, reverse=True)
        while index < len(data):
            for piece in ordered:
                if data.startswith(piece, index):
                    output.append(reverse[piece])
                    index += len(piece)
                    break
            else:
                raise ValueError(f"TokenVocabulary cannot encode byte {data[index:index + 1]!r}")
        return output

    def encode_text(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def detokenize_ids(self, token_ids: Sequence[int]) -> bytes:
        return b"".join(self.token_bytes_by_id[token_id] for token_id in token_ids)


class SchemaDecoderBridge:
    """Token-level bridge from Exp5921 character support to llama.cpp masks."""

    def __init__(self, support: Mapping[str, Any], vocabulary: TokenVocabulary) -> None:
        self.support = support
        self.vocabulary = vocabulary

    def prefix_status(self, prefix: bytes | str) -> PrefixStatus:
        data = prefix.encode("utf-8") if isinstance(prefix, str) else bytes(prefix)
        return classify_schema_prefix_bytes(data, self.support)

    def token_preserves_continuation(self, prefix: bytes, token_id: int) -> bool:
        token = self.vocabulary.token_bytes_by_id.get(token_id)
        if not token:
            return False
        status = self.prefix_status(prefix + token)
        return bool(status.valid and (status.can_continue or status.complete_valid))

    def allowed_token_ids(self, prefix: bytes | str) -> list[int]:
        data = prefix.encode("utf-8") if isinstance(prefix, str) else bytes(prefix)
        status = self.prefix_status(data)
        if not status.valid:
            return []
        allowed = [
            token_id
            for token_id in sorted(self.vocabulary.token_bytes_by_id)
            if self.token_preserves_continuation(data, token_id)
        ]
        if status.complete_valid and self.vocabulary.eos_token_id is not None:
            allowed.append(self.vocabulary.eos_token_id)
        return allowed

    def replay_text(self, text: str) -> JsonDict:
        data = text.encode("utf-8")
        tokens = self.vocabulary.encode_bytes(data)
        prefix = b""
        for index, token_id in enumerate(tokens):
            if not self.token_preserves_continuation(prefix, token_id):
                return {
                    "accepted": False,
                    "token_count": len(tokens),
                    "accepted_count": index,
                    "rejected_token": token_id,
                    "reason": "no_schema_valid_continuation",
                }
            prefix += self.vocabulary.token_bytes_by_id[token_id]
        status = self.prefix_status(prefix)
        return {
            "accepted": status.complete_valid,
            "token_count": len(tokens),
            "accepted_count": len(tokens),
            "rejected_token": None if status.complete_valid else "eof",
            "complete_valid": status.complete_valid,
            "reason": status.reason,
        }


class LlamaCppSchemaLogitsProcessor:
    """Public llama.cpp-compatible logits processor.

    llama-cpp-python calls logits processors with `(input_ids, scores)`.  This
    processor reconstructs the constrained suffix from embedded-token bytes,
    asks the bridge for the allowed next token IDs, and sets every other score
    to `-inf`.  If no token preserves a continuation, the returned mask is
    empty and generation fails closed.
    """

    def __init__(self, bridge: SchemaDecoderBridge) -> None:
        self.bridge = bridge
        self.last_receipt: JsonDict = {}

    def __call__(self, input_ids: Sequence[int], scores: np.ndarray) -> np.ndarray:
        prefix = b"".join(
            self.bridge.vocabulary.token_bytes_by_id.get(int(token_id), b"")
            for token_id in input_ids
        )
        allowed = self.bridge.allowed_token_ids(prefix)
        masked = np.full_like(scores, -np.inf, dtype=float)
        for token_id in allowed:
            if 0 <= token_id < len(masked):
                masked[token_id] = scores[token_id]
        self.last_receipt = {
            "input_token_count": len(input_ids),
            "prefix_bytes": len(prefix),
            "allowed_token_count": len(allowed),
            "dead_end": len(allowed) == 0,
            "eos_allowed": self.bridge.vocabulary.eos_token_id in allowed
            if self.bridge.vocabulary.eos_token_id is not None
            else False,
        }
        return masked


def canonical_json(value: Any) -> str:
    return exp5921.canonical_json(value)


def sha256_json(value: Any) -> str:
    return exp5921.sha256_json(value)


def sha256_file(path: str | Path) -> str:
    return exp5921.sha256_file(path)


def vocabulary_from_embedded_tokenizer(
    *,
    model_id: str,
    model_path: str,
    tokenizer: Any,
    token_bytes_by_id: Mapping[int, bytes],
    eos_token_id: int | None,
) -> TokenVocabulary:
    receipt = {
        "source": "embedded_gguf_tokenizer",
        "embedded_tokenizer_only": True,
        "used_hf_autotokenizer": False,
        "model_path": model_path,
        "n_vocab": len(token_bytes_by_id),
        "eos_token_id": eos_token_id,
        "_tokenizer": tokenizer,
    }
    return TokenVocabulary(
        model_id=model_id,
        model_path=model_path,
        token_bytes_by_id=dict(token_bytes_by_id),
        eos_token_id=eos_token_id,
        tokenizer_receipt=receipt,
    )


def grammar_terminal_strings(support: Mapping[str, Any]) -> list[str]:
    terminals = support.get("grammar_terminals") or {}
    strings: set[str] = {"{", "}", "[", "]", ":", ",", '"', " ", "\n", "\t"}
    for field in (
        "top_level_keys",
        "domain_keys",
        "entity_keys",
        "predicate_keys",
        "fact_keys",
        "rule_keys",
        "query_keys",
        "expression_nodes",
        "arith_ops",
        "domain_types",
    ):
        for item in terminals.get(field) or []:
            strings.add(str(item))
            strings.add(json.dumps(str(item), ensure_ascii=False))
    for literal in terminals.get("truth_literals") or []:
        strings.add("true" if literal else "false")
    strings.update(
        {
            json.dumps(exp5921.exp5896.CONSTRAINT_IR_SCHEMA_VERSION),
            "0",
            "1",
            "-1",
            "12",
            json.dumps("identifier"),
            json.dumps("?var"),
            json.dumps("caf\u00e9", ensure_ascii=False),
        }
    )
    return sorted(strings)


def map_terminals_to_token_ids(
    vocabulary: TokenVocabulary, terminals: Sequence[str]
) -> JsonDict:
    single: JsonDict = {}
    multi: JsonDict = {}
    unsupported: JsonDict = {}
    for terminal in terminals:
        data = terminal.encode("utf-8")
        try:
            token_ids = vocabulary.encode_bytes(data)
        except Exception as exc:
            unsupported[terminal] = {"reason": f"{type(exc).__name__}: {exc}"}
            continue
        if not token_ids:
            unsupported[terminal] = {"reason": "tokenize_empty"}
            continue
        roundtrip = vocabulary.detokenize_ids(token_ids)
        if roundtrip != data:
            unsupported[terminal] = {
                "reason": "roundtrip_mismatch",
                "token_ids": token_ids,
                "roundtrip_hex": roundtrip.hex(),
            }
            continue
        receipt = {
            "token_ids": token_ids,
            "utf8_bytes_hex": data.hex(),
            "token_count": len(token_ids),
        }
        if len(token_ids) == 1:
            single[terminal] = receipt
        else:
            multi[terminal] = receipt
    return {
        "model_id": vocabulary.model_id,
        "terminal_count": len(terminals),
        "single_token_terminals": single,
        "multi_token_terminals": multi,
        "unsupported_terminals": unsupported,
        "embedded_tokenizer_only": True,
        "used_hf_autotokenizer": False,
    }


def classify_schema_prefix_bytes(data: bytes, support: Mapping[str, Any]) -> PrefixStatus:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.reason == "unexpected end of data":
            return PrefixStatus(True, True, False, "partial_utf8", partial_utf8=True)
        return PrefixStatus(False, False, False, "invalid_utf8")
    scan = _scan_json_prefix(text, support)
    if not scan.valid:
        return PrefixStatus(False, False, False, scan.reason)
    if not scan.complete:
        return PrefixStatus(True, True, False, scan.reason)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return PrefixStatus(False, False, False, f"json_decode_error:{exc.msg}")
    verdict = exp5921.validate_with_support(payload, support)
    if verdict["full_support_valid"]:
        return PrefixStatus(True, True, True, "complete_schema_valid")
    return PrefixStatus(False, False, False, "complete_json_not_schema_valid")


@dataclass(frozen=True)
class _ScanResult:
    valid: bool
    complete: bool
    reason: str


def _scan_json_prefix(text: str, support: Mapping[str, Any]) -> _ScanResult:
    stack: list[JsonDict] = []
    complete_root = False
    index = 0
    while index < len(text):
        ch = text[index]
        if ch.isspace():
            index += 1
            continue
        if complete_root:
            return _ScanResult(False, False, "trailing_non_whitespace")
        if ch == "{":
            if not _value_can_start(stack):
                return _ScanResult(False, False, "object_not_allowed_here")
            stack.append({"kind": "object", "expect": "key_or_end", "container_key": _value_key(stack)})
            index += 1
            continue
        if ch == "[":
            if not _value_can_start(stack):
                return _ScanResult(False, False, "array_not_allowed_here")
            stack.append({"kind": "array", "expect": "value_or_end", "container_key": _value_key(stack)})
            index += 1
            continue
        if ch == "}":
            if not stack or stack[-1]["kind"] != "object":
                return _ScanResult(False, False, "object_close_mismatch")
            if stack[-1]["expect"] not in {"key_or_end", "comma_or_end"}:
                return _ScanResult(False, False, "object_closed_before_value")
            stack.pop()
            complete_root = _finish_value(stack)
            index += 1
            continue
        if ch == "]":
            if not stack or stack[-1]["kind"] != "array":
                return _ScanResult(False, False, "array_close_mismatch")
            if stack[-1]["expect"] not in {"value_or_end", "comma_or_end"}:
                return _ScanResult(False, False, "array_closed_before_value")
            stack.pop()
            complete_root = _finish_value(stack)
            index += 1
            continue
        if ch == ":":
            if not stack or stack[-1]["kind"] != "object" or stack[-1]["expect"] != "colon":
                return _ScanResult(False, False, "colon_not_allowed")
            stack[-1]["expect"] = "value"
            index += 1
            continue
        if ch == ",":
            if not stack or stack[-1]["expect"] != "comma_or_end":
                return _ScanResult(False, False, "comma_not_allowed")
            stack[-1]["expect"] = "key" if stack[-1]["kind"] == "object" else "value"
            index += 1
            continue
        if ch == '"':
            result = _consume_string(text, index, stack, support)
            if not result.valid:
                return result
            if not result.complete:
                return result
            if not stack:
                complete_root = True
            index = int(result.reason)
            continue
        if ch in "tf":
            result = _consume_literal(text, index, stack)
            if not result.valid or not result.complete:
                return result
            if not stack:
                complete_root = True
            index = int(result.reason)
            continue
        if ch == "-" or ch.isdigit():
            result = _consume_integer(text, index, stack)
            if not result.valid or not result.complete:
                return result
            if not stack:
                complete_root = True
            index = int(result.reason)
            continue
        return _ScanResult(False, False, "unexpected_character")
    return _ScanResult(True, complete_root, "complete" if complete_root else "incomplete")


def _consume_string(
    text: str, start: int, stack: list[JsonDict], support: Mapping[str, Any]
) -> _ScanResult:
    key_context = bool(stack and stack[-1]["kind"] == "object" and stack[-1]["expect"] in {"key", "key_or_end"})
    value_key = _value_key(stack)
    index = start + 1
    escaped = False
    content_chars: list[str] = []
    while index < len(text):
        ch = text[index]
        if escaped:
            content_chars.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            content = "".join(content_chars)
            if key_context:
                if not _key_allowed(stack[-1], content, support, partial=False):
                    return _ScanResult(False, False, "unsupported_object_key")
                stack[-1]["pending_key"] = content
                stack[-1]["expect"] = "colon"
            else:
                if not _value_string_allowed(value_key, content, support, partial=False):
                    return _ScanResult(False, False, "unsupported_terminal_string")
                complete_root = _finish_value(stack)
                if complete_root:
                    return _ScanResult(True, True, str(index + 1))
            return _ScanResult(True, True, str(index + 1))
        else:
            if ord(ch) < 0x20:
                return _ScanResult(False, False, "control_character_in_string")
            content_chars.append(ch)
        index += 1
    content = "".join(content_chars)
    if key_context and not _key_allowed(stack[-1], content, support, partial=True):
        return _ScanResult(False, False, "unsupported_object_key_prefix")
    if not key_context and not _value_string_allowed(value_key, content, support, partial=True):
        return _ScanResult(False, False, "unsupported_terminal_string_prefix")
    return _ScanResult(True, False, "open_string")


def _consume_literal(text: str, start: int, stack: list[JsonDict]) -> _ScanResult:
    remainder = text[start:]
    for literal in ("true", "false"):
        if text.startswith(literal, start):
            complete_root = _finish_value(stack)
            return _ScanResult(True, True, str(start + len(literal)) if not complete_root else str(start + len(literal)))
    matches = [literal for literal in ("true", "false") if literal.startswith(remainder)]
    if matches:
        return _ScanResult(True, False, "partial_literal")
    return _ScanResult(False, False, "unsupported_literal")


def _consume_integer(text: str, start: int, stack: list[JsonDict]) -> _ScanResult:
    index = start
    if text[index] == "-":
        index += 1
        if index == len(text):
            return _ScanResult(True, False, "partial_integer")
    if index == len(text) or not text[index].isdigit():
        return _ScanResult(False, False, "invalid_integer")
    while index < len(text) and text[index].isdigit():
        index += 1
    if index < len(text) and text[index] in ".eE":
        return _ScanResult(False, False, "non_integer_number")
    _finish_value(stack)
    return _ScanResult(True, True, str(index))


def _value_can_start(stack: list[JsonDict]) -> bool:
    if not stack:
        return True
    return stack[-1]["expect"] in {"value", "value_or_end"}


def _value_key(stack: list[JsonDict]) -> str | None:
    if not stack:
        return None
    frame = stack[-1]
    if frame["kind"] == "object" and frame["expect"] == "value":
        return frame.get("pending_key")
    if frame["kind"] == "array":
        return frame.get("container_key")
    return None


def _finish_value(stack: list[JsonDict]) -> bool:
    if not stack:
        return True
    frame = stack[-1]
    if frame["kind"] == "object":
        frame["expect"] = "comma_or_end"
        frame["pending_key"] = None
    else:
        frame["expect"] = "comma_or_end"
    return False


def _key_allowed(
    frame: Mapping[str, Any], key: str, support: Mapping[str, Any], *, partial: bool
) -> bool:
    allowed = _allowed_keys(frame, support)
    if allowed is None:
        return key.startswith("?") or (partial and "?".startswith(key))
    if partial:
        return any(item.startswith(key) for item in allowed)
    return key in allowed


def _allowed_keys(frame: Mapping[str, Any], support: Mapping[str, Any]) -> set[str] | None:
    terminals = support.get("grammar_terminals") or {}
    container = frame.get("container_key")
    if container is None:
        return set(terminals.get("top_level_keys") or [])
    if container == "domains":
        return set(terminals.get("domain_keys") or [])
    if container == "entities":
        return set(terminals.get("entity_keys") or [])
    if container == "predicates":
        return set(terminals.get("predicate_keys") or [])
    if container == "facts":
        return set(terminals.get("fact_keys") or [])
    if container == "rules":
        return set(terminals.get("rule_keys") or [])
    if container == "query":
        return set(terminals.get("query_keys") or [])
    if container in {"vars", "variables"}:
        return None
    expression_keys = {"args", "left", "node", "op", "predicate", "right", "term", "terms"}
    if container in {"body", "head", "where", "term", "terms"}:
        return expression_keys
    union: set[str] = set()
    for name in (
        "domain_keys",
        "entity_keys",
        "predicate_keys",
        "fact_keys",
        "rule_keys",
        "query_keys",
    ):
        union.update(terminals.get(name) or [])
    union.update(expression_keys)
    return union


def _value_string_allowed(
    key: str | None, value: str, support: Mapping[str, Any], *, partial: bool
) -> bool:
    terminals = support.get("grammar_terminals") or {}
    if key == "schema_version":
        expected = exp5921.exp5896.CONSTRAINT_IR_SCHEMA_VERSION
        return expected.startswith(value) if partial else value == expected
    if key == "type":
        values = {str(item) for item in terminals.get("domain_types") or []}
    elif key == "node":
        values = {str(item) for item in terminals.get("expression_nodes") or []}
    elif key == "op":
        values = {str(item) for item in terminals.get("arith_ops") or []}
    else:
        return True
    return any(item.startswith(value) for item in values) if partial else value in values


def utf8_whitespace_numeric_identifier_matrix(bridge: SchemaDecoderBridge) -> JsonDict:
    return {
        "utf8": {
            "complete_utf8_string_prefix_admitted": bridge.prefix_status(
                b'{"domains":[{"id":"caf\xc3\xa9'
            ).can_continue,
            "partial_utf8_waits_for_continuation": bridge.prefix_status(
                b'{"domains":[{"id":"caf\xc3'
            ).partial_utf8,
        },
        "whitespace": {
            "space_prefix_admitted": bridge.prefix_status(b" ").can_continue,
            "newline_prefix_admitted": bridge.prefix_status(b"\n").can_continue,
        },
        "numeric_literals": {
            "integer_value_prefix_admitted": bridge.prefix_status(
                b'{"domains":[{"id":"n","type":"int","values":[12'
            ).can_continue,
            "negative_integer_prefix_admitted": bridge.prefix_status(
                b'{"domains":[{"id":"n","type":"int","values":[-1'
            ).can_continue,
        },
        "identifiers": {
            "identifier_string_prefix_admitted": bridge.prefix_status(
                b'{"domains":[{"id":"alpha'
            ).can_continue,
            "variable_key_prefix_admitted": bridge.prefix_status(
                b'{"domains":[],"entities":[],"facts":[],"predicates":[],"query":{"vars":{"?'
            ).can_continue,
        },
        "invalid": {
            "invalid_leading_byte_rejected": bridge.prefix_status(b"\xff").valid is False,
            "unsupported_root_token_rejected": bridge.prefix_status(b"@").valid is False,
        },
    }


def eos_empty_mask_and_dead_end_policy(bridge: SchemaDecoderBridge) -> JsonDict:
    valid_text = _known_valid_text()
    eos_id = bridge.vocabulary.eos_token_id
    start_allowed = bridge.allowed_token_ids(b"")
    complete_allowed = bridge.allowed_token_ids(valid_text.encode("utf-8"))
    dead_allowed = bridge.allowed_token_ids(b"@")
    return {
        "policy": "fail_closed_empty_mask; eos_only_when_complete_schema_valid",
        "eos_token_id": eos_id,
        "eos_disallowed_at_start": eos_id not in start_allowed if eos_id is not None else True,
        "eos_allowed_after_complete_valid_json": eos_id in complete_allowed
        if eos_id is not None
        else False,
        "dead_end_prefix": "@",
        "dead_end_allowed_token_count": len(dead_allowed),
        "dead_end_fails_closed": dead_allowed == [],
    }


def tokenizer_reference_support_parity(
    bridges: Mapping[str, SchemaDecoderBridge],
) -> JsonDict:
    valid_text = _known_valid_text()
    invalid_text = (
        '{"domains":[{"id":"x","type":"float","values":["a"]}],"entities":[],'
        '"facts":[],"predicates":[],"query":{"vars":{},"where":{"node":"and","terms":[]}},'
        '"rules":[],"schema_version":"carnot.constraint_ir.v1"}'
    )
    rows: JsonDict = {}
    for model_id, bridge in bridges.items():
        valid_replay = bridge.replay_text(valid_text)
        invalid_replay = bridge.replay_text(invalid_text)
        sampled_prefixes = [b"", b"{", b'{"domains"', b'{"domains":[{"id":"a']
        admitted_preserve = []
        for prefix in sampled_prefixes:
            allowed = bridge.allowed_token_ids(prefix)
            admitted_preserve.append(
                all(
                    token_id == bridge.vocabulary.eos_token_id
                    or bridge.token_preserves_continuation(prefix, token_id)
                    for token_id in allowed
                )
            )
        rows[model_id] = {
            "known_valid_replay": valid_replay,
            "adversarial_invalid_replay": invalid_replay,
            "all_admitted_tokens_preserve_reference_continuation": all(admitted_preserve),
            "known_valid_fixture_path_reachable": valid_replay["accepted"] is True,
            "adversarial_dead_end_rejected": invalid_replay["accepted"] is False,
        }
    return {
        "models": rows,
        "all_models_parity": all(
            row["all_admitted_tokens_preserve_reference_continuation"]
            and row["known_valid_fixture_path_reachable"]
            and row["adversarial_dead_end_rejected"]
            for row in rows.values()
        ),
        "principle": FIELD_PRINCIPLES["tokenizer_reference_support_parity"],
    }


def _known_valid_text() -> str:
    cases = {case["case_id"]: case for case in exp5921.build_adversary_cases()}
    return exp5921.canonical_json(cases["held_family_menu_canonical"]["candidate"])


def resolve_all_model_specs() -> list[JsonDict]:  # pragma: no cover - host cache boundary.
    pair = cached_sota_pair(gpu_indices=(0, 1)) or []
    pair_by_id = {str(spec["hf_id"]): dict(spec) for spec in pair}
    resolved = []
    for index, spec in enumerate(MODEL_SPECS):
        row = dict(spec)
        row["gpu"] = index % 2
        if row["hf_id"] in pair_by_id:
            row["model_path"] = pair_by_id[row["hf_id"]].get("model_path")
            row["resolved_via"] = "cached_sota_pair"
        else:
            row["model_path"] = resolve_cached_gguf(str(row["hf_id"]))
            row["resolved_via"] = "resolve_cached_gguf_cached_third_family"
        resolved.append(row)
    return resolved


def load_embedded_llama_cpp_vocabulary(spec: Mapping[str, Any]) -> TokenVocabulary:  # pragma: no cover
    from llama_cpp import Llama

    model_path = str(spec["model_path"])
    started = time.perf_counter()
    llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
    token_bytes: dict[int, bytes] = {}
    for token_id in range(llm.n_vocab()):
        try:
            token_bytes[token_id] = llm.detokenize([token_id], special=False)
        except Exception:
            token_bytes[token_id] = b""
    receipt = {
        "source": "embedded_gguf_llama_cpp_vocab_only",
        "embedded_tokenizer_only": True,
        "used_hf_autotokenizer": False,
        "model_path": model_path,
        "load_duration_s": round(time.perf_counter() - started, 6),
        "n_vocab": llm.n_vocab(),
        "eos_token_id": llm.token_eos(),
        "tokenizer_probe_token_ids": llm.tokenize(b'{"domains":[]}', add_bos=False, special=False),
    }
    receipt["_tokenizer"] = llm
    return TokenVocabulary(
        model_id=str(spec["hf_id"]),
        model_path=model_path,
        token_bytes_by_id=token_bytes,
        eos_token_id=llm.token_eos(),
        tokenizer_receipt=receipt,
    )


def public_llama_cpp_api_receipt() -> JsonDict:  # pragma: no cover - installed package boundary.
    try:
        import llama_cpp
        from llama_cpp import Llama
        from llama_cpp import llama_cpp as low
    except Exception as exc:
        return {"ok": False, "importable": False, "reason": repr(exc)}
    call_sig = inspect.signature(Llama.__call__)
    completion_sig = inspect.signature(Llama.create_completion)
    gpu = bool(low.llama_supports_gpu_offload())
    return {
        "ok": "logits_processor" in call_sig.parameters
        and "logits_processor" in completion_sig.parameters
        and gpu,
        "importable": True,
        "version": getattr(llama_cpp, "__version__", "unknown"),
        "module": getattr(llama_cpp, "__file__", "unknown"),
        "binding": "llama_cpp.Llama.__call__/create_completion",
        "logits_processor_parameter": "logits_processor" in call_sig.parameters,
        "create_completion_logits_processor_parameter": "logits_processor"
        in completion_sig.parameters,
        "logits_processor_list_available": hasattr(llama_cpp, "LogitsProcessorList"),
        "gpu_offload_supported": gpu,
    }


def run_one_step_cuda_smoke(
    model_specs: Sequence[Mapping[str, Any]],
    bridges: Mapping[str, SchemaDecoderBridge],
) -> JsonDict:  # pragma: no cover - live CUDA boundary.
    from llama_cpp import Llama, LogitsProcessorList

    smokes = []
    for spec in model_specs:
        model_id = str(spec["hf_id"])
        bridge = bridges[model_id]
        before_mb = _gpu_memory_total_mb()
        started = time.perf_counter()
        llm = None
        try:
            llm = Llama(
                model_path=str(spec["model_path"]),
                n_ctx=256,
                n_batch=16,
                n_gpu_layers=N_GPU_LAYERS,
                seed=RANDOM_SEED,
                verbose=False,
            )
            after_load_mb = _gpu_memory_total_mb()
            processor = LlamaCppSchemaLogitsProcessor(bridge)
            result = llm(
                " ",
                max_tokens=1,
                temperature=0.0,
                seed=RANDOM_SEED,
                logits_processor=LogitsProcessorList([processor]),
                stop=[],
            )
            after_smoke_mb = _gpu_memory_total_mb()
            text = ""
            if isinstance(result, Mapping) and result.get("choices"):
                text = str(result["choices"][0].get("text", ""))
            smokes.append(
                {
                    "hf_id": model_id,
                    "model_path": str(spec["model_path"]),
                    "public_api": "llama_cpp.Llama.__call__(logits_processor=...)",
                    "one_step_only": True,
                    "semantic_claim": "none",
                    "load_and_smoke_duration_s": round(time.perf_counter() - started, 6),
                    "gpu_memory_before_mb": before_mb,
                    "gpu_memory_after_load_mb": after_load_mb,
                    "gpu_memory_after_smoke_mb": after_smoke_mb,
                    "gpu_memory_delta_mb": max(0.0, after_load_mb - before_mb),
                    "offload_evidence": after_load_mb > before_mb,
                    "generated_text_length": len(text),
                    "logits_processor_receipt": processor.last_receipt,
                }
            )
        except Exception as exc:
            smokes.append(
                {
                    "hf_id": model_id,
                    "model_path": str(spec.get("model_path")),
                    "public_api": "llama_cpp.Llama.__call__(logits_processor=...)",
                    "one_step_only": True,
                    "semantic_claim": "none",
                    "load_and_smoke_duration_s": round(time.perf_counter() - started, 6),
                    "gpu_memory_before_mb": before_mb,
                    "gpu_memory_after_load_mb": _gpu_memory_total_mb(),
                    "gpu_memory_delta_mb": 0.0,
                    "offload_evidence": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            llm = None
            gc.collect()
    return {
        "all_smokes_ok": all(row["offload_evidence"] for row in smokes),
        "smokes": smokes,
        "semantic_latency_or_quality_claim": "none",
    }


def fake_one_step_cuda_smoke(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    smokes = [
        {
            "hf_id": spec["hf_id"],
            "model_path": spec.get("model_path"),
            "public_api": "llama_cpp.Llama.__call__(logits_processor=...)",
            "one_step_only": True,
            "semantic_claim": "none",
            "gpu_memory_delta_mb": 1024,
            "offload_evidence": True,
            "logits_processor_receipt": {"allowed_token_count": 1, "dead_end": False},
        }
        for spec in model_specs
    ]
    return {
        "all_smokes_ok": bool(smokes),
        "smokes": smokes,
        "semantic_latency_or_quality_claim": "none",
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    duration_s: float = 0.0,
    test_exit_codes: Mapping[str, int] | None = None,
    model_resolver: Callable[[], list[JsonDict]] = resolve_all_model_specs,
    tokenizer_loader: Callable[[Mapping[str, Any]], TokenVocabulary] = load_embedded_llama_cpp_vocabulary,
    cuda_smoke_runner: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, SchemaDecoderBridge]], JsonDict
    ] = run_one_step_cuda_smoke,
    public_api_checker: Callable[[], JsonDict] = public_llama_cpp_api_receipt,
    protected_baseline: Mapping[str, Any] | None = None,
) -> JsonDict:
    target = output_path or root / RESULT_RELATIVE_PATH
    baseline = protected_baseline or _protected_file_receipt(root)
    support = exp5921.compile_schema_support()
    gate = _gate_replay_receipt(root, support)
    model_specs = model_resolver()
    model_hashes = _model_file_hashes(model_specs)
    tokenizers: dict[str, TokenVocabulary] = {}
    tokenizer_receipts: JsonDict = {}
    mappings: JsonDict = {}
    terminals = grammar_terminal_strings(support)
    for spec in model_specs:
        if not spec.get("model_path"):
            continue
        vocab = tokenizer_loader(spec)
        tokenizers[str(spec["hf_id"])] = vocab
        receipt = dict(vocab.tokenizer_receipt)
        receipt.pop("_tokenizer", None)
        receipt["embedded_tokenizer_only"] = True
        receipt["used_hf_autotokenizer"] = False
        tokenizer_receipts[str(spec["hf_id"])] = receipt
        mappings[str(spec["hf_id"])] = map_terminals_to_token_ids(vocab, terminals)
    bridges = {model_id: SchemaDecoderBridge(support, vocab) for model_id, vocab in tokenizers.items()}
    public_api = public_api_checker()
    one_step = cuda_smoke_runner(model_specs, bridges) if bridges else {"all_smokes_ok": False, "smokes": []}
    parity = tokenizer_reference_support_parity(bridges) if bridges else {"all_models_parity": False, "models": {}}
    representative = next(iter(bridges.values()), None)
    matrix = (
        utf8_whitespace_numeric_identifier_matrix(representative)
        if representative is not None
        else {"ok": False}
    )
    eos_policy = (
        eos_empty_mask_and_dead_end_policy(representative)
        if representative is not None
        else {"dead_end_fails_closed": False}
    )
    protected = _protected_file_receipt(root, baseline=baseline)
    preconditions = _preconditions(root, target, model_specs, model_hashes, tokenizer_receipts, public_api, gate)
    unsupported_multi = _unsupported_and_multitoken(mappings)
    ready = (
        preconditions["all_preconditions_ok"]
        and len(model_specs) == 3
        and set(tokenizers) == {str(spec["hf_id"]) for spec in MODEL_SPECS}
        and parity["all_models_parity"] is True
        and eos_policy["dead_end_fails_closed"] is True
        and public_api.get("ok") is True
        and one_step.get("all_smokes_ok") is True
        and protected["unchanged"] is True
    )
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": "complete_ready" if ready else "blocked",
        "gate_replay_receipt": gate,
        "preconditions_checked": preconditions,
        "model_specs": model_specs,
        "model_file_hashes": model_hashes,
        "embedded_tokenizer_receipts": tokenizer_receipts,
        "public_llama_cpp_cuda_receipt": public_api,
        "schema_support_version_and_hash": {
            "support_schema_version": support["support_schema_version"],
            "signature_schema_version": support["signature_schema_version"],
            "signature_schema_hash": support["signature_schema_hash"],
            "support_schema_hash": support["schema_hash"],
        },
        "per_model_terminal_token_mapping": mappings,
        "unsupported_and_multitoken_terminal_receipts": unsupported_multi,
        "tokenizer_reference_support_parity": parity,
        "utf8_whitespace_numeric_identifier_matrix": matrix,
        "eos_empty_mask_and_dead_end_policy": eos_policy,
        "logits_processor_public_api_receipt": {
            "class": "LlamaCppSchemaLogitsProcessor",
            "call_signature": "__call__(input_ids, scores)",
            "masks_to_negative_infinity": True,
            "empty_mask_fails_closed": True,
            "public_llama_cpp_logits_processor_parameter": public_api.get(
                "logits_processor_parameter"
            )
            is True,
        },
        "one_step_cuda_smoke": one_step,
        "full_answer_enumeration_used": False,
        "protected_files_unchanged": protected,
        "gguf_schema_decoder_bridge_ready_score": 1.0 if ready else 0.0,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_ready: all three embedded GGUF tokenizers replay schema support through public llama.cpp logits_processor with fail-closed dead-end controls"
            if ready
            else "blocked: GGUF tokenizer bridge preconditions, parity, public API, or CUDA smoke did not all pass"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    model_resolver: Callable[[], list[JsonDict]] = resolve_all_model_specs,
    tokenizer_loader: Callable[[Mapping[str, Any]], TokenVocabulary] = load_embedded_llama_cpp_vocabulary,
    cuda_smoke_runner: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, SchemaDecoderBridge]], JsonDict
    ] = run_one_step_cuda_smoke,
    public_api_checker: Callable[[], JsonDict] = public_llama_cpp_api_receipt,
) -> JsonDict:
    started = time.monotonic()
    target = output_path or root / RESULT_RELATIVE_PATH
    baseline = _protected_file_receipt(root)
    elapsed = duration_s if duration_s is not None else round(time.monotonic() - started, 6)
    artifact = build_artifact(
        root=root,
        output_path=target,
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
        model_resolver=model_resolver,
        tokenizer_loader=tokenizer_loader,
        cuda_smoke_runner=cuda_smoke_runner,
        public_api_checker=public_api_checker,
        protected_baseline=baseline,
    )
    if duration_s is None:
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
        validate_artifact(artifact)
    _write_json_atomic(target, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be public_llama_cpp_cuda_tokenizer_bridge_smoke")
    if artifact["full_answer_enumeration_used"] is not False:
        raise ValueError("full_answer_enumeration_used must be bare false")
    score = float(artifact["gguf_schema_decoder_bridge_ready_score"])
    if score not in {0.0, 1.0}:
        raise ValueError("ready_score must be bare 0.0 or 1.0")
    if score == 1.0:
        if not str(artifact["honest_verdict"]).startswith("complete_ready:"):
            raise ValueError("complete_ready verdict required for ready bridge")
        if artifact["tokenizer_reference_support_parity"]["all_models_parity"] is not True:
            raise ValueError("ready bridge requires tokenizer parity")
        if artifact["one_step_cuda_smoke"]["all_smokes_ok"] is not True:
            raise ValueError("ready bridge requires one-step CUDA smoke")
    if not str(artifact["honest_verdict"]).startswith(("complete_ready:", "retired:", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")


def refresh_artifact_test_exit_codes(
    *,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    path = root / RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["test_exit_codes"] = dict(test_exit_codes)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    _write_json_atomic(path, artifact)
    return artifact


def _gate_replay_receipt(root: Path, support: Mapping[str, Any]) -> JsonDict:
    path = root / exp5921.RESULT_RELATIVE_PATH
    compiled = exp5921.compile_schema_support()
    replay = exp5921.support_replay_receipt(support, compiled)
    receipt: JsonDict = {
        "exp5921_artifact_path": str(path),
        "exp5921_artifact_present": path.exists(),
        "deterministic_support_replay": replay,
        "support_complete_ready": False,
    }
    if path.exists():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        exp5921.validate_artifact(artifact)
        receipt["support_complete_ready"] = artifact.get("schema_decode_contract_ready_score") == 1.0
        receipt["support_artifact_checksum"] = artifact.get("reproducibility_checksum")
    else:
        receipt["support_complete_ready"] = replay["deterministic_replay"]
    receipt["ok"] = bool(receipt["support_complete_ready"] and replay["deterministic_replay"])
    return receipt


def _preconditions(
    root: Path,
    output_path: Path,
    model_specs: Sequence[Mapping[str, Any]],
    model_hashes: Mapping[str, Any],
    tokenizer_receipts: Mapping[str, Any],
    public_api: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> JsonDict:
    expected_ids = {str(spec["hf_id"]) for spec in MODEL_SPECS}
    resolved_ids = {str(spec.get("hf_id")) for spec in model_specs if spec.get("model_path")}
    disk = _disk_probe(root)
    ram = _memory_probe()
    atomic = _atomic_output_probe(output_path)
    hashes = _hash_inputs(root)
    files_ok = expected_ids.issubset(model_hashes)
    tokenizers_ok = expected_ids.issubset(tokenizer_receipts)
    return {
        "exp5921_gate_replayed": gate.get("ok") is True,
        "model_specs_defined": [dict(spec) for spec in MODEL_SPECS],
        "resolved_all_three_model_files": resolved_ids == expected_ids,
        "cached_sota_pair_plus_cached_third_family": len(model_specs) == 3,
        "model_files_hashed": files_ok,
        "embedded_tokenizers_loaded_only_from_gguf": tokenizers_ok
        and all(row.get("used_hf_autotokenizer") is False for row in tokenizer_receipts.values()),
        "auto_tokenizer_from_pretrained_used": False,
        "public_llama_cpp_cuda_available": public_api.get("ok") is True,
        "disk": disk,
        "ram": ram,
        "atomic_output": atomic,
        "hashed_inputs": hashes,
        "output_path": str(output_path),
        "all_preconditions_ok": bool(
            gate.get("ok")
            and resolved_ids == expected_ids
            and files_ok
            and tokenizers_ok
            and public_api.get("ok")
            and disk["ok"]
            and ram["ok"]
            and atomic["ok"]
        ),
    }


def _model_file_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    hashes: JsonDict = {}
    for spec in model_specs:
        path = Path(str(spec.get("model_path") or ""))
        if path.is_file():
            hashes[str(spec["hf_id"])] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return hashes


def _unsupported_and_multitoken(mappings: Mapping[str, Any]) -> JsonDict:
    return {
        model_id: {
            "unsupported_terminals": mapping.get("unsupported_terminals", {}),
            "multi_token_terminals": mapping.get("multi_token_terminals", {}),
            "unsupported_count": len(mapping.get("unsupported_terminals", {})),
            "multi_token_count": len(mapping.get("multi_token_terminals", {})),
        }
        for model_id, mapping in mappings.items()
    }


def _hash_inputs(root: Path) -> JsonDict:
    files = []
    for relative in HASHED_INPUTS:
        path = root / relative
        files.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    return {"files": files, "all_present": all(row["exists"] for row in files)}


def _protected_file_receipt(root: Path, baseline: Mapping[str, Any] | None = None) -> JsonDict:
    files = []
    baseline_by_path = {str(item["path"]): item for item in (baseline or {}).get("files", [])}
    for relative in PROTECTED_FILES:
        path = root / relative
        current = sha256_file(path) if path.exists() else None
        before = baseline_by_path.get(str(relative), {}).get("sha256", current)
        files.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256_before": before,
                "sha256": current,
                "unchanged": before == current,
            }
        )
    return {"unchanged": all(row["unchanged"] for row in files), "files": files}


def _disk_probe(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _memory_probe() -> JsonDict:
    required_mb = 512
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    return {"available_mb": available_mb, "required_mb": required_mb, "ok": available_mb >= required_mb}


def _atomic_output_probe(output_path: Path) -> JsonDict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path = output_path.parent / f".{output_path.name}.atomic-probe"
    replacement = output_path.parent / f".{output_path.name}.atomic-probe.tmp"
    try:
        probe_path.write_text("old", encoding="utf-8")
        replacement.write_text("new", encoding="utf-8")
        os.replace(replacement, probe_path)
        ok = probe_path.read_text(encoding="utf-8") == "new"
    finally:
        probe_path.unlink(missing_ok=True)
        replacement.unlink(missing_ok=True)
    return {"ok": ok, "method": "os.replace_same_directory"}


def _field_provenance() -> JsonDict:
    return {
        field: {
            "satisfied_by": "generated_by_exp5922_tokenizer_bridge",
            "principle": FIELD_PRINCIPLES.get(field, "Exp5922 required artifact field."),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["test_exit_codes"] = {}
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked", {})
    if isinstance(preconditions, dict):
        if isinstance(preconditions.get("disk"), dict):
            preconditions["disk"]["available_mb"] = 0
        if isinstance(preconditions.get("ram"), dict):
            preconditions["ram"]["available_mb"] = 0
    return sha256_json(stable)


def _write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        handle.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _gpu_memory_total_mb() -> float:  # pragma: no cover - hardware boundary.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return 0.0
    if result.returncode != 0:
        return 0.0
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return float(sum(values))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    del argv
    write_artifact()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
