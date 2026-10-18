"""Requalify three frozen tokenizer bindings without model inference.

The old experiments hashed different receipt wrappers. This module instead
reduces both exact GGUF files to the semantic payload frozen by Exp6865. It
then compares native token IDs under one adversarial probe matrix.

Spec refs: REQ-INFERENCE-6866 and SCENARIO-INFERENCE-6866-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any
import unicodedata

from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    cached_sota_pair,
    flagship_dense,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/llm-ebm-inference/spec.md")
RESULT_PATH = Path("results/experiment_6866_canonical_tokenizer_binding_requalification.json")
SOURCE_PATHS = {
    "exp6850": Path("results/experiment_6850_three_family_scoring_admission_canary.json"),
    "exp6862": Path("results/experiment_6862_dual_side_semantic_contrast_bank.json"),
    "exp6863": Path(
        "results/experiment_6863_tokenizer_aware_semantic_contrast_preregistration.json"
    ),
    "exp6865": Path("results/experiment_6865_v601_evidence_method_change_contract.json"),
}

SCHEMA = "carnot.experiment_6866.canonical_tokenizer_binding_requalification.v1"
REDUCER_VERSION = "carnot.canonical_tokenizer_receipt_reducer.v1"
CANONICAL_PAYLOAD_SCHEMA_VERSION = "carnot.canonical_tokenizer_payload.v1"
INFERENCE_SUBSTRATE = "native_gguf_tokenization_without_inference"
RANDOM_SEED = 6866
BLOCKED_VERDICT = "complete_blocked_canonical_tokenizer_binding_requalification"
POSITIVE_VERDICT = "complete_positive_canonical_tokenizer_binding_requalification_ready"

MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
SPEC_REFS = (
    "REQ-INFERENCE-6866",
    "SCENARIO-INFERENCE-6866-CANONICAL-ORDER",
    "SCENARIO-INFERENCE-6866-SEMANTIC-DRIFT",
    "SCENARIO-INFERENCE-6866-TEXT-NORMALIZATION",
    "SCENARIO-INFERENCE-6866-OLD-RECEIPT-MIGRATION",
    "SCENARIO-INFERENCE-6866-PROBE-MATRIX",
    "SCENARIO-INFERENCE-6866-BLOCKED-ARTIFACT",
)

EXP6862_RAW_PROMPT_TEMPLATE = (
    "Exact semantic obligation program.\n"
    "PROGRAM_JSON_BEGIN\n{program_json}\nPROGRAM_JSON_END\n"
    "Return only the candidate sequence."
)
EXP6862_CANDIDATE_SEQUENCE_TEMPLATE = "CANDIDATE_JSON_BEGIN\n{candidate_json}\nCANDIDATE_JSON_END"
REQUIRED_PROBE_CLASSES = (
    "whitespace",
    "newline",
    "tab",
    "unicode_nfc",
    "unicode_nfd",
    "punctuation",
    "label_swap",
    "control_like_text",
    "literal_special_token_text",
    "empty_text",
    "chat_template_boundary",
    "exp6862_sequence_templates",
)
PROBE_SETTINGS = (
    {
        "setting_id": "plain_no_bos",
        "add_bos": False,
        "special": False,
        "utf8_error_mode": "replace",
        "unicode_normalization": "none",
    },
    {
        "setting_id": "plain_with_bos",
        "add_bos": True,
        "special": False,
        "utf8_error_mode": "replace",
        "unicode_normalization": "none",
    },
    {
        "setting_id": "special_no_bos",
        "add_bos": False,
        "special": True,
        "utf8_error_mode": "replace",
        "unicode_normalization": "none",
    },
    {
        "setting_id": "nfc_no_bos",
        "add_bos": False,
        "special": False,
        "utf8_error_mode": "replace",
        "unicode_normalization": "NFC",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "model_specs",
    "models_used",
    "model_artifact_hashes",
    "archived_receipt_rows",
    "live_receipt_rows",
    "source_field_maps",
    "canonical_payload_schema_version",
    "canonical_payload_hash_rows",
    "semantic_probe_manifest",
    "rows",
    "mismatch_witnesses",
    "append_only_correction_receipt",
    "token_likelihood_call_count",
    "generated_answer_count",
    "random_seed",
    "reproducibility_checksum",
    "canonical_tokenizer_binding_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "The version prevents silent consumer drift.",
    "experiment_id": "The fixed identifier binds this evidence to Exp6866.",
    "run_date": "The supplied execution date separates later cache states.",
    "status": "The status confirms a terminal artifact write.",
    "result_path": "The stable path is the append-only correction target.",
    "spec_refs": "The references connect each check to its requirement.",
    "field_principles": "Each field states why it exists.",
    "preconditions_checked": "Exact checks make each blocked result diagnosable.",
    "inference_substrate": "The substrate confirms tokenization without inference.",
    "duration_s": "Measured wall time makes the run auditable.",
    "source_artifact_hashes": "Source hashes bind the migration inputs to exact bytes.",
    "model_specs": "The list fixes all three required model repositories.",
    "models_used": "The list shows which native tokenizers were loaded.",
    "model_artifact_hashes": "Hashes and sizes prove exact model-file identity.",
    "archived_receipt_rows": "Rows recompute Exp6850 bindings through the new reducer.",
    "live_receipt_rows": "Rows recompute current bindings through the same reducer.",
    "source_field_maps": "Maps disclose every migrated or excluded wrapper field.",
    "canonical_payload_schema_version": "The value pins the Exp6865 semantic schema.",
    "canonical_payload_hash_rows": "Rows compare only canonical semantic content.",
    "semantic_probe_manifest": "The manifest freezes all adversarial token inputs.",
    "rows": "Each row compares one model, probe, and setting.",
    "mismatch_witnesses": "Typed witnesses preserve every blocking difference.",
    "append_only_correction_receipt": "The receipt corrects interpretation without rewriting history.",
    "token_likelihood_call_count": "Zero proves that no likelihood informed the result.",
    "generated_answer_count": "Zero proves that no model answer was generated.",
    "random_seed": "The fixed seed records deterministic ordering.",
    "reproducibility_checksum": "The checksum binds all stable result content.",
    "canonical_tokenizer_binding_ready_score": "Exp6867 consumes this exact readiness field.",
    "gate_check_summary": "The summary names exact expected and observed failures.",
    "verifier_is_oracle": "False prevents this reducer from becoming a truth oracle.",
    "verdict_class": "The closed class states the evidence boundary.",
    "honest_verdict": "The complete prefix gives one terminal disposition.",
}


def canonical_json(value: Any) -> str:
    """Return sorted compact UTF-8 JSON text for every semantic hash."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_sequence(values: Iterable[Any]) -> str:
    """Hash a sequence as canonical JSON without retaining a large list."""

    digest = hashlib.sha256()
    digest.update(b"[")
    for index, value in enumerate(values):
        if index:
            digest.update(b",")
        digest.update(canonical_json(value).encode("utf-8"))
    digest.update(b"]")
    return "sha256:" + digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def prepare_probe_bytes(
    value: str | bytes,
    *,
    normalization: str,
    utf8_error_mode: str = "replace",
) -> bytes:
    """Apply the frozen decoding and Unicode rule before tokenization."""

    if utf8_error_mode != "replace":
        raise ValueError("utf8_error_mode must be replace")
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
    if normalization not in {"none", "NFC", "NFD"}:
        raise ValueError(f"unsupported Unicode normalization: {normalization}")
    if normalization != "none":
        text = unicodedata.normalize(normalization, text)
    return text.encode("utf-8")


def semantic_probe_manifest() -> JsonDict:
    """Return one frozen matrix with Exp6862 sequence boundaries."""

    program_json = canonical_json({"authority": "required", "priority": 1})
    candidate_json = canonical_json({"accepted": True, "label": "A"})
    probes: list[JsonDict] = [
        {"probe_id": "whitespace", "probe_class": "whitespace", "text": "  lead  mid  tail  "},
        {"probe_id": "newline", "probe_class": "newline", "text": "line one\nline two\n"},
        {"probe_id": "tab", "probe_class": "tab", "text": "left\tmiddle\tright"},
        {
            "probe_id": "unicode_nfc",
            "probe_class": "unicode_nfc",
            "text": "caf\u00e9 \u00c5ngstr\u00f6m",
        },
        {
            "probe_id": "unicode_nfd",
            "probe_class": "unicode_nfd",
            "text": "cafe\u0301 A\u030angstro\u0308m",
        },
        {
            "probe_id": "punctuation",
            "probe_class": "punctuation",
            "text": "A/B: [x] -> {y}; !? ...",
        },
        {
            "probe_id": "label_swap",
            "probe_class": "label_swap",
            "text": "slot_0=B; slot_1=A; labels=[1,0]",
        },
        {
            "probe_id": "control_like_text",
            "probe_class": "control_like_text",
            "text": "\\x00\\r\\nBEGIN_JSON\\u0000END_JSON",
        },
        {
            "probe_id": "literal_special_token_text",
            "probe_class": "literal_special_token_text",
            "text": "<bos><eos><|im_start|><|im_end|><start_of_turn><end_of_turn>",
        },
        {"probe_id": "empty_text", "probe_class": "empty_text", "text": ""},
        {
            "probe_id": "chat_template_boundary",
            "probe_class": "chat_template_boundary",
            "text": "<|im_start|>user\nprobe<|im_end|>\n<start_of_turn>user\nprobe<end_of_turn>\n",
        },
        {
            "probe_id": "exp6862_raw_prompt_template",
            "probe_class": "exp6862_sequence_templates",
            "text": EXP6862_RAW_PROMPT_TEMPLATE.format(program_json=program_json),
        },
        {
            "probe_id": "exp6862_candidate_sequence_template",
            "probe_class": "exp6862_sequence_templates",
            "text": EXP6862_CANDIDATE_SEQUENCE_TEMPLATE.format(candidate_json=candidate_json),
        },
        {
            "probe_id": "invalid_utf8_replacement",
            "probe_class": "invalid_utf8_replacement",
            "text_bytes_hex": b"bad\xffutf8".hex(),
        },
    ]
    basis = {
        "schema": "carnot.semantic_token_probe_manifest.v1",
        "raw_prompt_template": EXP6862_RAW_PROMPT_TEMPLATE,
        "candidate_sequence_template": EXP6862_CANDIDATE_SEQUENCE_TEMPLATE,
        "probes": probes,
        "settings": list(PROBE_SETTINGS),
    }
    return {**basis, "manifest_sha256": sha256_json(basis)}


def run_probe_matrix(tokenizer: Any, manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Tokenize every frozen probe under every exact setting."""

    outputs: list[JsonDict] = []
    for probe in manifest.get("probes", []):
        raw: str | bytes
        if "text_bytes_hex" in probe:
            raw = bytes.fromhex(str(probe["text_bytes_hex"]))
        else:
            raw = str(probe.get("text", ""))
        for setting in PROBE_SETTINGS:
            prepared = prepare_probe_bytes(
                raw,
                normalization=str(setting["unicode_normalization"]),
                utf8_error_mode=str(setting["utf8_error_mode"]),
            )
            token_ids = tokenizer.tokenize(
                prepared,
                add_bos=bool(setting["add_bos"]),
                special=bool(setting["special"]),
            )
            outputs.append(
                {
                    "probe_id": probe["probe_id"],
                    "text_utf8_sha256": sha256_bytes(prepared),
                    "settings": deepcopy(setting),
                    "token_ids": [int(token) for token in token_ids],
                }
            )
    return outputs


def _source_field_map(source_receipt: Mapping[str, Any]) -> list[JsonDict]:
    """Explain why each old wrapper field is recomputed or excluded."""

    rows: list[JsonDict] = []
    for field in sorted(source_receipt):
        if field == "tokenizer_sha256" or field.endswith("receipt_hash"):
            disposition = "excluded_nested_hash"
            reason = "A nested receipt hash cannot define the new semantic payload."
        elif field in {
            "probe_token_ids",
            "special_tokens",
            "tokenize_settings",
            "chat_template_identity",
            "chat_template_present",
        }:
            disposition = "recomputed"
            reason = "The exact GGUF and frozen matrix supply this semantic value again."
        else:
            disposition = "excluded_wrapper_only"
            reason = "This source field is wrapper provenance, not Exp6865 semantic content."
        rows.append(
            {
                "source_field": str(field),
                "canonical_field": None,
                "disposition": disposition,
                "reason": reason,
            }
        )
    return rows


def _probe_setting_summary(probe_outputs: Sequence[Mapping[str, Any]], field: str) -> list[Any]:
    values = {
        row.get("settings", {}).get(field)
        for row in probe_outputs
        if isinstance(row.get("settings"), Mapping)
    }
    return sorted(values, key=lambda item: (str(type(item)), str(item)))


def reduce_tokenizer_receipt(
    evidence: Mapping[str, Any],
    probe_outputs: Sequence[Mapping[str, Any]],
    *,
    source_receipt: Mapping[str, Any],
    source_kind: str,
) -> JsonDict:
    """Reduce one exact GGUF capture to the Exp6865 V1 semantic payload."""

    empty_by_bos: dict[str, list[int]] = {}
    for row in probe_outputs:
        settings = row.get("settings")
        if row.get("probe_id") == "empty_text" and isinstance(settings, Mapping):
            empty_by_bos[f"requested_{str(bool(settings.get('add_bos'))).lower()}"] = [
                int(token) for token in row.get("token_ids", [])
            ]
    special = evidence.get("special_token_ids")
    special = special if isinstance(special, Mapping) else {}
    payload = {
        "semantic_vocabulary_metadata": {
            field: evidence.get(field)
            for field in (
                "tokenizer_model",
                "tokenizer_pretokenizer",
                "vocabulary_size",
                "token_pieces_sha256",
                "token_scores_sha256",
                "token_types_sha256",
                "merges_sha256",
            )
        },
        "special_token_ids": {
            field: special.get(field)
            for field in (
                "bos_token_id",
                "eos_token_id",
                "unknown_token_id",
                "padding_token_id",
                "mask_token_id",
                "separator_token_id",
            )
        },
        "add_bos_behavior": {
            "metadata_default": evidence.get("add_bos_metadata_default"),
            "requested": _probe_setting_summary(probe_outputs, "add_bos"),
            "observed_prefix_ids": empty_by_bos,
        },
        "chat_template_identity": {
            "present": bool(evidence.get("chat_template")),
            "template_sha256": sha256_bytes(
                str(evidence.get("chat_template") or "").encode("utf-8")
            ),
        },
        "tokenization_settings": {
            field: _probe_setting_summary(probe_outputs, field)
            for field in (
                "add_bos",
                "special",
                "utf8_error_mode",
                "unicode_normalization",
            )
        },
        "frozen_probe_outputs": [deepcopy(dict(row)) for row in probe_outputs],
    }
    return {
        "source_kind": source_kind,
        "reducer_version": REDUCER_VERSION,
        "canonical_payload_schema_version": CANONICAL_PAYLOAD_SCHEMA_VERSION,
        "canonical_payload": payload,
        "canonical_payload_sha256": sha256_json(payload),
        "source_field_names": sorted(str(field) for field in source_receipt),
        "source_field_map": _source_field_map(source_receipt),
    }


def _probe_map(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = payload.get("frozen_probe_outputs")
    rows = rows if isinstance(rows, list) else []
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        settings = row.get("settings")
        settings = settings if isinstance(settings, Mapping) else {}
        result[(str(row.get("probe_id")), str(settings.get("setting_id")))] = row
    return result


def _semantic_mismatch_reasons(
    archived_payload: Mapping[str, Any], live_payload: Mapping[str, Any]
) -> list[str]:
    """Return typed semantic drift reasons in a stable order."""

    reasons: list[str] = []
    sections = (
        ("semantic_vocabulary_metadata", "vocabulary_drift"),
        ("special_token_ids", "special_token_drift"),
        ("add_bos_behavior", "add_bos_drift"),
        ("chat_template_identity", "chat_template_drift"),
    )
    for section, prefix in sections:
        archived = archived_payload.get(section)
        live = live_payload.get(section)
        archived = archived if isinstance(archived, Mapping) else {}
        live = live if isinstance(live, Mapping) else {}
        for field in sorted(set(archived) | set(live)):
            if section == "add_bos_behavior" and field in {
                "observed_prefix_ids",
                "requested",
            }:
                continue
            if archived.get(field) != live.get(field):
                reasons.append(f"{prefix}:{field}")

    archived_probes = _probe_map(archived_payload)
    live_probes = _probe_map(live_payload)
    for probe_id, setting_id in sorted(set(archived_probes) | set(live_probes)):
        archived = archived_probes.get((probe_id, setting_id), {})
        live = live_probes.get((probe_id, setting_id), {})
        if archived.get("settings") != live.get("settings"):
            reasons.append(f"tokenization_setting_drift:{probe_id}:{setting_id}")
        elif archived.get("text_utf8_sha256") != live.get("text_utf8_sha256"):
            reasons.append(f"probe_text_drift:{probe_id}:{setting_id}")
        elif archived.get("token_ids") != live.get("token_ids"):
            reasons.append(f"token_output_drift:{probe_id}:{setting_id}")
    return reasons


def compare_reduced_receipts(archived: Mapping[str, Any], live: Mapping[str, Any]) -> JsonDict:
    """Classify semantic equality separately from wrapper-schema changes."""

    archived_payload = archived.get("canonical_payload")
    live_payload = live.get("canonical_payload")
    archived_payload = archived_payload if isinstance(archived_payload, Mapping) else {}
    live_payload = live_payload if isinstance(live_payload, Mapping) else {}
    reasons = _semantic_mismatch_reasons(archived_payload, live_payload)
    if reasons:
        status = "semantic_payload_drift"
    elif archived.get("source_kind") != live.get("source_kind") and archived.get(
        "source_field_names"
    ) != live.get("source_field_names"):
        status = "wrapper_schema_only"
    else:
        status = "semantic_payload_equal"
    return {
        "comparison_status": status,
        "mismatch_reasons": reasons,
        "blocks_model": bool(reasons),
    }


def snapshot_identity(model_path: str) -> str:
    """Extract the Hugging Face snapshot name from an exact cache path."""

    parts = Path(model_path).parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return ""
    return parts[index + 1] if index + 1 < len(parts) else ""


def model_binding_mismatch_reasons(
    current: Mapping[str, Any], frozen: Mapping[str, Any]
) -> list[str]:
    """Compare every required model-file identity field."""

    fields = (
        ("hf_id", "hub_id_drift"),
        ("sha256", "model_hash_drift"),
        ("size_bytes", "model_size_drift"),
        ("quantization", "quantization_drift"),
        ("snapshot_identity", "snapshot_identity_drift"),
    )
    return [reason for field, reason in fields if current.get(field) != frozen.get(field)]


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"check": check, "expected": expected, "observed": observed, "passed": bool(passed)}


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": [dict(row) for row in checks],
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def _read_sources(root: Path) -> tuple[dict[str, JsonDict], dict[str, str], list[JsonDict]]:
    documents: dict[str, JsonDict] = {}
    hashes: dict[str, str] = {}
    unreadable: list[JsonDict] = []
    for source_id, relative in SOURCE_PATHS.items():
        path = root / relative
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON object required")
            documents[source_id] = value
            hashes[source_id] = sha256_path(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            unreadable.append(
                {"source_id": source_id, "path": relative.as_posix(), "error": str(exc)}
            )
            hashes[source_id] = ""
    return documents, hashes, unreadable


def _frozen_model_rows(exp6850: Mapping[str, Any]) -> list[JsonDict]:
    specs_value = exp6850.get("model_specs")
    specs = specs_value if isinstance(specs_value, list) else []
    hashes_value = exp6850.get("model_artifact_hashes")
    hashes = hashes_value if isinstance(hashes_value, Mapping) else {}
    by_id = {str(row.get("hf_id")): row for row in specs if isinstance(row, Mapping)}
    rows: list[JsonDict] = []
    for hf_id in MODEL_SPECS:
        spec = by_id.get(hf_id, {})
        frozen_hash = hashes.get(hf_id)
        frozen_hash = frozen_hash if isinstance(frozen_hash, Mapping) else {}
        path = str(spec.get("model_path") or frozen_hash.get("path") or "")
        rows.append(
            {
                "hf_id": hf_id,
                "model_path": path,
                "sha256": spec.get("model_sha256") or frozen_hash.get("sha256"),
                "size_bytes": spec.get("model_size_bytes") or frozen_hash.get("size_bytes"),
                "quantization": spec.get("quantization"),
                "snapshot_identity": snapshot_identity(path),
            }
        )
    return rows


def resolve_model_files(exp6850: Mapping[str, Any]) -> list[JsonDict]:  # pragma: no cover
    """Call the pair resolver first, then extend it with flagship dense."""

    pair = cached_sota_pair() or []
    pair_by_id = {str(row.get("hf_id")): dict(row) for row in pair if isinstance(row, Mapping)}
    dense = flagship_dense()
    dense_path = resolve_cached_gguf(dense["hf_id"], dense["quantization"])
    if dense_path:
        pair_by_id[dense["hf_id"]] = {
            "hf_id": dense["hf_id"],
            "model_path": dense_path,
        }
    registry = {str(row["hf_id"]): row for row in SOTA_GGUF_MODELS}
    frozen = {row["hf_id"]: row for row in _frozen_model_rows(exp6850)}
    rows: list[JsonDict] = []
    for hf_id in MODEL_SPECS:
        source = pair_by_id.get(hf_id, {})
        path_text = str(source.get("model_path") or "")
        path = Path(path_text) if path_text else Path()
        present = bool(path_text and path.is_file())
        rows.append(
            {
                "hf_id": hf_id,
                "model_path": path_text,
                "sha256": sha256_path(path) if present else "",
                "size_bytes": path.stat().st_size if present else None,
                "quantization": registry[hf_id]["quantization"],
                "snapshot_identity": snapshot_identity(path_text),
                "cached_sota_pair_called": True,
                "flagship_dense_extension_called": hf_id == dense["hf_id"],
                "embedded_tokenizer_metadata": False,
                "frozen_path": frozen.get(hf_id, {}).get("model_path"),
            }
        )
    return rows


# GGUF uses small integer type tags in its metadata header. The reader stops
# before tensor bytes, so it does not load model weights into memory.
_GGUF_FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_exact(handle: Any, size: int) -> bytes:  # pragma: no cover - live GGUF boundary.
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("truncated GGUF metadata")
    return value


def _read_u32(handle: Any) -> int:  # pragma: no cover - live GGUF boundary.
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle: Any) -> int:  # pragma: no cover - live GGUF boundary.
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_gguf_string(handle: Any) -> str:  # pragma: no cover - live GGUF boundary.
    return _read_exact(handle, _read_u64(handle)).decode("utf-8", "replace")


def _read_gguf_scalar(handle: Any, value_type: int) -> Any:  # pragma: no cover
    if value_type == 8:
        return _read_gguf_string(handle)
    formats = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    if value_type not in formats:
        raise ValueError(f"unsupported GGUF scalar type: {value_type}")
    return struct.unpack(formats[value_type], _read_exact(handle, _GGUF_FIXED_SIZES[value_type]))[0]


def _skip_gguf_value(handle: Any, value_type: int) -> None:  # pragma: no cover
    if value_type == 8:
        handle.seek(_read_u64(handle), os.SEEK_CUR)
        return
    if value_type == 9:
        element_type = _read_u32(handle)
        count = _read_u64(handle)
        if element_type in _GGUF_FIXED_SIZES:
            handle.seek(_GGUF_FIXED_SIZES[element_type] * count, os.SEEK_CUR)
            return
        if element_type == 8:
            for _ in range(count):
                handle.seek(_read_u64(handle), os.SEEK_CUR)
            return
        raise ValueError(f"unsupported GGUF array type: {element_type}")
    handle.seek(_GGUF_FIXED_SIZES[value_type], os.SEEK_CUR)


def _read_tokenizer_gguf_metadata(path: str) -> JsonDict:  # pragma: no cover
    """Read only semantic tokenizer keys from the GGUF metadata header."""

    scalar_keys = {
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.mask_token_id",
        "tokenizer.ggml.separator_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.chat_template",
    }
    array_keys = {
        "tokenizer.ggml.tokens": "token_pieces_sha256",
        "tokenizer.ggml.scores": "token_scores_sha256",
        "tokenizer.ggml.token_type": "token_types_sha256",
        "tokenizer.ggml.merges": "merges_sha256",
    }
    result: JsonDict = {}
    with Path(path).open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ValueError("GGUF magic missing")
        version = _read_u32(handle)
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version: {version}")
        _read_u64(handle)
        key_count = _read_u64(handle)
        for _ in range(key_count):
            key = _read_gguf_string(handle)
            value_type = _read_u32(handle)
            if key in scalar_keys:
                result[key] = _read_gguf_scalar(handle, value_type)
            elif key in array_keys and value_type == 9:
                element_type = _read_u32(handle)
                count = _read_u64(handle)
                values = (_read_gguf_scalar(handle, element_type) for _ in range(count))
                result[array_keys[key]] = sha256_sequence(values)
                result[array_keys[key] + "_count"] = count
            else:
                _skip_gguf_value(handle, value_type)
    for field in array_keys.values():
        result.setdefault(field, sha256_sequence(()))
        result.setdefault(field + "_count", 0)
    return result


def _token_id(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


class NativeGgufTokenizer:  # pragma: no cover - exercised by the requested live run.
    """Load native vocabulary metadata and expose tokenization only."""

    def __init__(self, model_path: str) -> None:
        from llama_cpp import Llama

        metadata = _read_tokenizer_gguf_metadata(model_path)
        self._llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        model = self._llm._model
        self.evidence = {
            "tokenizer_model": metadata.get("tokenizer.ggml.model"),
            "tokenizer_pretokenizer": metadata.get("tokenizer.ggml.pre"),
            "vocabulary_size": int(model.n_vocab()),
            "token_pieces_sha256": metadata["token_pieces_sha256"],
            "token_scores_sha256": metadata["token_scores_sha256"],
            "token_types_sha256": metadata["token_types_sha256"],
            "merges_sha256": metadata["merges_sha256"],
            "special_token_ids": {
                "bos_token_id": _token_id(metadata.get("tokenizer.ggml.bos_token_id")),
                "eos_token_id": _token_id(metadata.get("tokenizer.ggml.eos_token_id")),
                "unknown_token_id": _token_id(metadata.get("tokenizer.ggml.unknown_token_id")),
                "padding_token_id": _token_id(metadata.get("tokenizer.ggml.padding_token_id")),
                "mask_token_id": _token_id(metadata.get("tokenizer.ggml.mask_token_id")),
                "separator_token_id": _token_id(metadata.get("tokenizer.ggml.separator_token_id")),
            },
            "add_bos_metadata_default": bool(model.add_bos_token()),
            "chat_template": str(metadata.get("tokenizer.chat_template") or ""),
        }

    def tokenize(self, text: bytes, *, add_bos: bool, special: bool) -> list[int]:
        return [int(token) for token in self._llm.tokenize(text, add_bos=add_bos, special=special)]

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()
        del self._llm
        gc.collect()


def _metadata_present(evidence: Mapping[str, Any]) -> bool:
    return bool(
        evidence.get("tokenizer_model")
        and int(evidence.get("vocabulary_size") or 0) > 0
        and str(evidence.get("token_pieces_sha256") or "").startswith("sha256:")
    )


def _base_artifact(run_date: str, duration_s: float) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": 6866,
        "run_date": run_date,
        "status": "complete",
        "result_path": RESULT_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "field_principles": {},
        "preconditions_checked": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "source_artifact_hashes": {},
        "model_specs": list(MODEL_SPECS),
        "models_used": [],
        "model_artifact_hashes": {},
        "archived_receipt_rows": [],
        "live_receipt_rows": [],
        "source_field_maps": [],
        "canonical_payload_schema_version": CANONICAL_PAYLOAD_SCHEMA_VERSION,
        "canonical_payload_hash_rows": [],
        "semantic_probe_manifest": semantic_probe_manifest(),
        "rows": [],
        "mismatch_witnesses": [],
        "append_only_correction_receipt": {
            "schema": "carnot.append_only_tokenizer_correction_receipt.v1",
            "correction_kind": "blocked_before_comparison",
            "old_artifacts_rewritten": False,
            "target_artifacts": [
                SOURCE_PATHS["exp6850"].as_posix(),
                SOURCE_PATHS["exp6863"].as_posix(),
            ],
            "correction_artifact": RESULT_PATH.as_posix(),
            "rows": [],
        },
        "token_likelihood_call_count": 0,
        "generated_answer_count": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "canonical_tokenizer_binding_ready_score": 0,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding measured time and annotations."""

    unsigned = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "field_principles", "reproducibility_checksum"}
    }
    return sha256_json(unsigned)


def _finish_artifact(artifact: JsonDict) -> JsonDict:
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves exact requalification evidence.")
        for key in artifact
    }
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]
    return artifact


def _receipt_by_model(artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    receipts = artifact.get("tokenizer_receipts")
    receipts = receipts if isinstance(receipts, list) else []
    return {str(row.get("hf_id")): row for row in receipts if isinstance(row, Mapping)}


def _probe_comparison_rows(
    hf_id: str, archived: Mapping[str, Any], live: Mapping[str, Any]
) -> tuple[list[JsonDict], list[JsonDict]]:
    archived_payload = archived.get("canonical_payload")
    live_payload = live.get("canonical_payload")
    archived_payload = archived_payload if isinstance(archived_payload, Mapping) else {}
    live_payload = live_payload if isinstance(live_payload, Mapping) else {}
    archived_map = _probe_map(archived_payload)
    live_map = _probe_map(live_payload)
    rows: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    for probe_id, setting_id in sorted(set(archived_map) | set(live_map)):
        old = archived_map.get((probe_id, setting_id), {})
        new = live_map.get((probe_id, setting_id), {})
        if old.get("settings") != new.get("settings"):
            reason = f"tokenization_setting_drift:{probe_id}:{setting_id}"
        elif old.get("text_utf8_sha256") != new.get("text_utf8_sha256"):
            reason = f"probe_text_drift:{probe_id}:{setting_id}"
        elif old.get("token_ids") != new.get("token_ids"):
            reason = f"token_output_drift:{probe_id}:{setting_id}"
        else:
            reason = None
        row = {
            "hf_id": hf_id,
            "probe_id": probe_id,
            "setting_id": setting_id,
            "settings": deepcopy(old.get("settings") or new.get("settings")),
            "text_utf8_sha256": old.get("text_utf8_sha256") or new.get("text_utf8_sha256"),
            "archived_token_ids": deepcopy(old.get("token_ids")),
            "live_token_ids": deepcopy(new.get("token_ids")),
            "archived_canonical_payload_sha256": archived.get("canonical_payload_sha256"),
            "live_canonical_payload_sha256": live.get("canonical_payload_sha256"),
            "comparison_status": "semantic_payload_drift" if reason else "semantic_payload_equal",
            "mismatch_reason": reason,
        }
        rows.append(row)
        if reason:
            witnesses.append(
                {
                    "hf_id": hf_id,
                    "probe_id": probe_id,
                    "setting_id": setting_id,
                    "reason": reason,
                    "expected": deepcopy(old.get("token_ids")),
                    "observed": deepcopy(new.get("token_ids")),
                }
            )
    return rows, witnesses


def build_artifact(
    root: Path,
    run_date: str,
    *,
    model_resolver: Callable[[Mapping[str, Any]], list[JsonDict]] = resolve_model_files,
    tokenizer_factory: Callable[[str], Any] = NativeGgufTokenizer,
) -> JsonDict:
    """Build a positive or complete blocked append-only receipt."""

    started = time.monotonic()
    artifact = _base_artifact(run_date, 0.0)
    documents, source_hashes, unreadable = _read_sources(root)
    artifact["source_artifact_hashes"] = {
        source_id: {"path": SOURCE_PATHS[source_id].as_posix(), "sha256": digest}
        for source_id, digest in source_hashes.items()
    }
    checks: list[JsonDict] = [
        _check(
            "required_source_readability",
            "all required artifacts readable",
            unreadable,
            not unreadable,
        )
    ]
    if unreadable:
        artifact["preconditions_checked"] = _gate_summary(checks)
        artifact["gate_check_summary"] = _gate_summary(checks)
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        return _finish_artifact(artifact)

    exp6850 = documents["exp6850"]
    exp6862 = documents["exp6862"]
    exp6863 = documents["exp6863"]
    exp6865 = documents["exp6865"]
    checks.append(
        _check(
            "v601_evidence_contract_ready_score",
            1,
            exp6865.get("v601_evidence_contract_ready_score"),
            exp6865.get("v601_evidence_contract_ready_score") == 1,
        )
    )
    schema = exp6865.get("canonical_tokenizer_payload_schema")
    schema = schema if isinstance(schema, Mapping) else {}
    checks.append(
        _check(
            "canonical_payload_schema_version",
            CANONICAL_PAYLOAD_SCHEMA_VERSION,
            schema.get("schema_id"),
            schema.get("schema_id") == CANONICAL_PAYLOAD_SCHEMA_VERSION,
        )
    )
    template = exp6862.get("frozen_sequence_template_manifest")
    template = template if isinstance(template, Mapping) else {}
    observed_templates = {
        "raw_prompt_template": template.get("raw_prompt_template"),
        "candidate_sequence_template": template.get("candidate_sequence_template"),
    }
    expected_templates = {
        "raw_prompt_template": EXP6862_RAW_PROMPT_TEMPLATE,
        "candidate_sequence_template": EXP6862_CANDIDATE_SEQUENCE_TEMPLATE,
    }
    checks.append(
        _check(
            "exp6862_frozen_sequence_templates",
            expected_templates,
            observed_templates,
            observed_templates == expected_templates,
        )
    )
    archived_receipts = _receipt_by_model(exp6850)
    live_source_receipts = _receipt_by_model(exp6863)
    checks.append(
        _check(
            "readable_exp6850_and_exp6863_receipts",
            list(MODEL_SPECS),
            {
                "exp6850": sorted(archived_receipts),
                "exp6863": sorted(live_source_receipts),
            },
            set(archived_receipts) == set(MODEL_SPECS)
            and set(live_source_receipts) == set(MODEL_SPECS),
        )
    )
    frozen_rows = _frozen_model_rows(exp6850)
    resolved_rows = model_resolver(exp6850)
    frozen_by_id = {str(row.get("hf_id")): row for row in frozen_rows}
    resolved_by_id = {str(row.get("hf_id")): row for row in resolved_rows}
    checks.append(
        _check(
            "all_three_exact_model_resolutions",
            list(MODEL_SPECS),
            sorted(resolved_by_id),
            set(resolved_by_id) == set(MODEL_SPECS),
        )
    )
    hash_cache: dict[str, str] = {}
    for hf_id in MODEL_SPECS:
        frozen = frozen_by_id.get(hf_id, {})
        current = resolved_by_id.get(hf_id, {})
        frozen_path = Path(str(frozen.get("model_path") or ""))
        if frozen_path.is_file():
            path_key = str(frozen_path)
            if path_key == str(current.get("model_path") or "") and current.get("sha256"):
                hash_cache[path_key] = str(current["sha256"])
            else:
                hash_cache.setdefault(path_key, sha256_path(frozen_path))
            actual_frozen = {
                **frozen,
                "sha256": hash_cache[path_key],
                "size_bytes": frozen_path.stat().st_size,
            }
        else:
            actual_frozen = {**frozen, "sha256": "", "size_bytes": None}
        frozen_errors = model_binding_mismatch_reasons(actual_frozen, frozen)
        current_errors = model_binding_mismatch_reasons(current, frozen)
        if current.get("cached_sota_pair_called") is not True:
            current_errors.append("cached_sota_pair_not_called")
        checks.append(
            _check(f"model.{hf_id}.archived_file_binding", [], frozen_errors, not frozen_errors)
        )
        checks.append(
            _check(f"model.{hf_id}.live_file_binding", [], current_errors, not current_errors)
        )

    preliminary = _gate_summary(checks)
    artifact["preconditions_checked"] = preliminary
    if not preliminary["passed"]:
        artifact["gate_check_summary"] = preliminary
        artifact["model_artifact_hashes"] = {
            hf_id: deepcopy(resolved_by_id.get(hf_id, {})) for hf_id in MODEL_SPECS
        }
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        return _finish_artifact(artifact)

    manifest = artifact["semantic_probe_manifest"]
    for hf_id in MODEL_SPECS:
        frozen = frozen_by_id[hf_id]
        current = resolved_by_id[hf_id]

        def capture(
            model_path: str,
            source_receipt: Mapping[str, Any],
            source_kind: str,
        ) -> tuple[JsonDict, bool]:
            tokenizer = tokenizer_factory(model_path)
            try:
                metadata_present = _metadata_present(tokenizer.evidence)
                outputs = run_probe_matrix(tokenizer, manifest)
                reduced = reduce_tokenizer_receipt(
                    tokenizer.evidence,
                    outputs,
                    source_receipt=source_receipt,
                    source_kind=source_kind,
                )
                return reduced, metadata_present
            finally:
                tokenizer.close()

        try:
            archived_reduced, archived_metadata = capture(
                str(frozen["model_path"]),
                archived_receipts[hf_id],
                "archived_exp6850_recomputed",
            )
            live_reduced, live_metadata = capture(
                str(current["model_path"]),
                live_source_receipts[hf_id],
                "live_exp6863_schema_recomputed",
            )
        except Exception as exc:  # noqa: BLE001 - a blocked artifact must survive native failures.
            checks.append(
                _check(
                    f"model.{hf_id}.native_tokenizer_load",
                    "native GGUF tokenizer and probe matrix complete",
                    f"{type(exc).__name__}:{exc}",
                    False,
                )
            )
            summary = _gate_summary(checks)
            artifact["preconditions_checked"] = summary
            artifact["gate_check_summary"] = summary
            artifact["model_artifact_hashes"] = {
                model_id: deepcopy(resolved_by_id.get(model_id, {})) for model_id in MODEL_SPECS
            }
            artifact["append_only_correction_receipt"]["correction_kind"] = "blocked_precondition"
            artifact["duration_s"] = round(time.monotonic() - started, 6)
            return _finish_artifact(artifact)

        checks.append(
            _check(
                f"model.{hf_id}.embedded_tokenizer_metadata",
                {"archived": True, "live": True},
                {"archived": archived_metadata, "live": live_metadata},
                archived_metadata and live_metadata,
            )
        )

        artifact["archived_receipt_rows"].append({"hf_id": hf_id, **archived_reduced})
        artifact["live_receipt_rows"].append({"hf_id": hf_id, **live_reduced})
        artifact["source_field_maps"].append(
            {
                "hf_id": hf_id,
                "archived": deepcopy(archived_reduced["source_field_map"]),
                "live": deepcopy(live_reduced["source_field_map"]),
            }
        )
        comparison = compare_reduced_receipts(archived_reduced, live_reduced)
        artifact["canonical_payload_hash_rows"].append(
            {
                "hf_id": hf_id,
                "reducer_version": REDUCER_VERSION,
                "archived_canonical_payload_sha256": archived_reduced["canonical_payload_sha256"],
                "live_canonical_payload_sha256": live_reduced["canonical_payload_sha256"],
                **comparison,
            }
        )
        probe_rows, probe_witnesses = _probe_comparison_rows(hf_id, archived_reduced, live_reduced)
        artifact["rows"].extend(probe_rows)
        artifact["mismatch_witnesses"].extend(probe_witnesses)
        probe_reason_set = {row["reason"] for row in probe_witnesses}
        for reason in comparison["mismatch_reasons"]:
            if reason not in probe_reason_set:
                artifact["mismatch_witnesses"].append(
                    {
                        "hf_id": hf_id,
                        "probe_id": None,
                        "setting_id": None,
                        "reason": reason,
                        "expected": archived_reduced["canonical_payload_sha256"],
                        "observed": live_reduced["canonical_payload_sha256"],
                    }
                )

    artifact["models_used"] = list(MODEL_SPECS)
    artifact["model_artifact_hashes"] = {
        hf_id: {
            "path": resolved_by_id[hf_id].get("model_path"),
            "sha256": resolved_by_id[hf_id].get("sha256"),
            "size_bytes": resolved_by_id[hf_id].get("size_bytes"),
            "quantization": resolved_by_id[hf_id].get("quantization"),
            "snapshot_identity": resolved_by_id[hf_id].get("snapshot_identity"),
        }
        for hf_id in MODEL_SPECS
    }
    all_semantic_equal = not artifact["mismatch_witnesses"]
    checks.append(
        _check(
            "canonical_semantic_and_probe_match",
            [],
            artifact["mismatch_witnesses"],
            all_semantic_equal,
        )
    )
    gate_summary = _gate_summary(checks)
    artifact["preconditions_checked"] = _gate_summary(checks[:-1])
    artifact["gate_check_summary"] = gate_summary
    ready = bool(gate_summary["passed"] and all_semantic_equal)
    artifact["canonical_tokenizer_binding_ready_score"] = int(ready)
    artifact["verdict_class"] = "positive" if ready else "blocked"
    artifact["honest_verdict"] = POSITIVE_VERDICT if ready else BLOCKED_VERDICT
    statuses = {row["comparison_status"] for row in artifact["canonical_payload_hash_rows"]}
    correction_kind = (
        "blocked_precondition"
        if not artifact["preconditions_checked"]["passed"]
        else "semantic_payload_drift_blocked"
        if not ready
        else "wrapper_schema_only"
        if "wrapper_schema_only" in statuses
        else "semantic_payload_equal"
    )
    artifact["append_only_correction_receipt"] = {
        "schema": "carnot.append_only_tokenizer_correction_receipt.v1",
        "correction_kind": correction_kind,
        "old_artifacts_rewritten": False,
        "target_artifacts": [
            SOURCE_PATHS["exp6850"].as_posix(),
            SOURCE_PATHS["exp6863"].as_posix(),
        ],
        "correction_artifact": RESULT_PATH.as_posix(),
        "rows": [
            {
                "hf_id": row["hf_id"],
                "comparison_status": row["comparison_status"],
                "archived_canonical_payload_sha256": row["archived_canonical_payload_sha256"],
                "live_canonical_payload_sha256": row["live_canonical_payload_sha256"],
            }
            for row in artifact["canonical_payload_hash_rows"]
        ],
    }
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    return _finish_artifact(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate the exact downstream gate and zero-inference contract."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_must_cover_every_field")
    if artifact.get("token_likelihood_call_count") != 0:
        errors.append("token_likelihood_call_count_must_be_zero")
    if artifact.get("generated_answer_count") != 0:
        errors.append("generated_answer_count_must_be_zero")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    if artifact.get("canonical_tokenizer_binding_ready_score") not in {0, 1}:
        errors.append("invalid_ready_score")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one artifact atomically without touching earlier results."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6866 and write only the requested append-only artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    args = parser.parse_args(argv)
    artifact = build_artifact(REPO_ROOT, args.date)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(";".join(errors))
    write_json_atomic(args.output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
