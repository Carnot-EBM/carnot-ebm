"""Freeze a decoder-aligned evidence schema without loading a model.

The prior pilot generated eight real responses, but its prompt omitted the
entity enum that its parser required. This module preserves those failures and
builds one explicit authority for the prompt, decoder grammar, and independent
post-generation validator.

Spec: REQ-REPORT-7616 and SCENARIO-REPORT-7616-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any

from carnot import experiment_7588_v663_evidence_protocol as evidence_protocol
from carnot import experiment_7602_v664_evidence_requalification as roles_protocol
from carnot import experiment_7603_v664_guarded_update_fixture as update_fixture
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.665"
EXPERIMENT_ID = "exp7616-v665-evidence-schema"
SCHEMA = "carnot.exp7616.v665.evidence_schema.v1"
RESULT_PATH = Path("results/experiment_7616_v665_evidence_schema.json")
RAW_DIR = Path("results/raw/experiment_7616_v665_evidence_schema")
MODULE_PATH = Path("python/carnot/experiment_7616_v665_evidence_schema.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7616_v665_evidence_schema.py")
TEST_PATH = Path("tests/python/test_experiment_7616_v665_evidence_schema.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXP7602_RESULT = Path("results/experiment_7602_v664_evidence_requalification.json")
EXP7602_PROTOCOL = Path("results/raw/experiment_7602_v664_evidence_requalification/protocol.json")
EXP7603_RESULT = Path("results/experiment_7603_v664_guarded_update_fixture.json")
EXP7604_RESULT = Path("results/experiment_7604_v664_evidence_pilot.json")
EXP7604_RUN = Path("results/raw/experiment_7604_v664_evidence_pilot/runs/1790251058-2137004")
MODEL_SPECS: list[str] = []
RANDOM_SEED = 7_616_001
MAX_POINTERS = 6
POINTER_FIELDS = (
    "response_sentence_id",
    "source_sentence_ids",
    "relation",
    "entity_type",
    "abstention_reason",
)
RELATIONS = ("supports", "contradicts", "unknown")
ENTITY_TYPES = (
    "person",
    "organization",
    "location",
    "date",
    "numeric",
    "code",
    "other",
    "none",
)
SUPPORTED_DECODER_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "enum",
        "const",
        "oneOf",
        "pattern",
    }
)
UNSUPPORTED_DECODER_KEYWORDS = ("uniqueItems",)
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "forward_calls_attempted": 0,
    "forward_calls_completed": 0,
    "forward_calls_failed": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}
TERMINAL_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def text_sha256(value: str) -> str:
    """Hash exact UTF-8 text so reconstruction checks bind every byte."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sentences(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep sentence identity and exact text once in model-visible input."""

    return [{"id": str(row["sentence_id"]), "text": str(row["text"])} for row in rows]


def build_canonical_input(record: Mapping[str, Any]) -> JsonDict:
    """Replace duplicate complete texts with ordered sentence arrays."""

    sides = (
        ("source", "complete_source", "source_sentences", "source_sha256"),
        ("question", "complete_question", "question_sentences", "question_sha256"),
        ("response", "complete_answer", "answer_sentences", "answer_sha256"),
    )
    model_input: JsonDict = {}
    hashes: JsonDict = {}
    for side, text_key, rows_key, hash_key in sides:
        text = record.get(text_key)
        rows = record.get(rows_key)
        if not isinstance(text, str) or not isinstance(rows, list):
            raise ValueError(f"canonical_{side}_absent")
        evidence_protocol.roundtrip_segments(text, rows)
        observed_hash = text_sha256(text)
        if record.get(hash_key) != observed_hash:
            raise ValueError(f"canonical_{side}_hash_invalid")
        model_input[f"{side}_sentences"] = _canonical_sentences(rows)
        hashes[side] = observed_hash
    result = {
        "component_hash": str(record.get("component_hash") or ""),
        "model_input": model_input,
        "reconstruction_hashes": hashes,
    }
    if not result["component_hash"]:
        raise ValueError("component_hash_absent")
    for side in ("source", "question", "response"):
        reconstruct_canonical_text(result, side)
    return result


def reconstruct_canonical_text(contract: Mapping[str, Any], side: str) -> str:
    """Join an ordered canonical array and verify its frozen content hash."""

    if side not in {"source", "question", "response"}:
        raise ValueError("canonical_side_invalid")
    model_input = contract.get("model_input")
    hashes = contract.get("reconstruction_hashes")
    rows = model_input.get(f"{side}_sentences") if isinstance(model_input, Mapping) else None
    if not isinstance(rows, list) or not isinstance(hashes, Mapping):
        raise ValueError(f"canonical_{side}_absent")
    value = "".join(str(row.get("text") or "") for row in rows if isinstance(row, Mapping))
    if len(rows) != sum(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"canonical_{side}_row_invalid")
    if text_sha256(value) != hashes.get(side):
        raise ValueError(f"canonical_{side}_reconstruction_invalid")
    return value


def build_schema_authority(record: Mapping[str, Any]) -> JsonDict:
    """Build the sole enum, field, branch, ID, and pointer-count authority."""

    contract = build_canonical_input(record)
    model_input = contract["model_input"]
    source_ids = [row["id"] for row in model_input["source_sentences"]]
    response_ids = [row["id"] for row in model_input["response_sentences"]]
    common_properties = {
        "response_sentence_id": {"type": "string", "enum": response_ids},
        "entity_type": {"type": "string", "enum": list(ENTITY_TYPES)},
    }
    linked = {
        "type": "object",
        "properties": {
            "response_sentence_id": common_properties["response_sentence_id"],
            "source_sentence_ids": {
                "type": "array",
                "items": {"type": "string", "enum": source_ids},
                "minItems": 1,
                "maxItems": MAX_POINTERS,
            },
            "relation": {"type": "string", "enum": ["supports", "contradicts"]},
            "entity_type": common_properties["entity_type"],
            "abstention_reason": {"const": ""},
        },
        "required": list(POINTER_FIELDS),
        "additionalProperties": False,
    }
    unknown = {
        "type": "object",
        "properties": {
            "response_sentence_id": common_properties["response_sentence_id"],
            "source_sentence_ids": {"type": "array", "maxItems": 0},
            "relation": {"const": "unknown"},
            "entity_type": common_properties["entity_type"],
            "abstention_reason": {"type": "string", "pattern": "^.+$"},
        },
        "required": list(POINTER_FIELDS),
        "additionalProperties": False,
    }
    decoder_schema = {
        "type": "array",
        "items": {"oneOf": [linked, unknown]},
        "maxItems": MAX_POINTERS,
    }
    return {
        "authority_schema": "carnot.evidence_pointer_authority.v1",
        "pointer_fields": list(POINTER_FIELDS),
        "relations": list(RELATIONS),
        "entity_types": list(ENTITY_TYPES),
        "maximum_pointers": MAX_POINTERS,
        "allowed_source_sentence_ids": source_ids,
        "allowed_response_sentence_ids": response_ids,
        "linked_relations": ["supports", "contradicts"],
        "unknown_relation": "unknown",
        "json_schema": decoder_schema,
        "supported_decoder_keywords": sorted(SUPPORTED_DECODER_KEYWORDS),
        "unsupported_decoder_keywords": list(UNSUPPORTED_DECODER_KEYWORDS),
        "unsupported_keyword_policy": {
            "uniqueItems": "independent_validator_rejects_duplicate_source_ids",
        },
        "independent_validator_required": True,
    }


def render_system_prompt(authority: Mapping[str, Any]) -> str:
    """Render every model instruction from the explicit schema authority."""

    fields = ", ".join(str(value) for value in authority["pointer_fields"])
    relations = ", ".join(str(value) for value in authority["relations"])
    entities = ", ".join(str(value) for value in authority["entity_types"])
    maximum = int(authority["maximum_pointers"])
    return (
        "Extract evidence pointers from the supplied canonical sentence arrays. "
        f"Return one JSON array with at most {maximum} items and no prose. "
        f"Every item has exactly these fields: {fields}. "
        f"relation must be one of: {relations}. entity_type must be one of: {entities}. "
        "Use only supplied sentence IDs. supports and contradicts require one or more "
        "source IDs and an empty abstention_reason. unknown requires no source IDs and "
        "a non-empty abstention_reason. Do not emit replacement text or reasoning. /no_think"
    )


def _schema_keywords(value: Any) -> set[str]:
    """Collect JSON-schema keywords while ignoring property names and enum values."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key != "properties":
                found.add(str(key))
            if key == "properties" and isinstance(item, Mapping):
                for child in item.values():
                    found.update(_schema_keywords(child))
            else:
                found.update(_schema_keywords(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_schema_keywords(item))
    return found


def compile_decoder_grammar(authority: Mapping[str, Any]) -> str:
    """Compile through the installed llama.cpp adapter after keyword validation."""

    decoder_schema = authority.get("json_schema")
    keywords = _schema_keywords(decoder_schema)
    unsupported = sorted(keywords - SUPPORTED_DECODER_KEYWORDS)
    if unsupported:
        raise ValueError(f"unsupported_schema_keyword:{unsupported[0]}")
    from llama_cpp.llama_grammar import json_schema_to_gbnf

    return json_schema_to_gbnf(json.dumps(decoder_schema, sort_keys=True))


def build_extraction_request(record: Mapping[str, Any]) -> JsonDict:
    """Build one decoder-constrained request over canonical arrays only."""

    contract = build_canonical_input(record)
    authority = build_schema_authority(record)
    return {
        "messages": [
            {"role": "system", "content": render_system_prompt(authority)},
            {
                "role": "user",
                "content": json.dumps(
                    contract["model_input"], ensure_ascii=False, separators=(",", ":")
                ),
            },
        ],
        "response_format": {"type": "json_object", "schema": authority["json_schema"]},
        "temperature": 0.0,
        "seed": RANDOM_SEED,
        "max_tokens": 512,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _unknown_row(response_id: str) -> JsonDict:
    return {
        "response_sentence_id": response_id,
        "source_sentence_ids": [],
        "relation": "unknown",
        "entity_type": "none",
        "abstention_reason": "unlinked_by_extractor",
        "proposed_link": False,
    }


def validate_evidence_output(
    record: Mapping[str, Any], response_text: str, *, finish_reason: str
) -> JsonDict:
    """Validate decoder output independently of grammar acceptance."""

    authority = build_schema_authority(record)
    if finish_reason == "length":
        return {"accepted": False, "error": "truncated_output", "evidence": []}
    try:
        proposals = json.loads(response_text)
    except json.JSONDecodeError:
        return {"accepted": False, "error": "truncated_json", "evidence": []}
    if not isinstance(proposals, list) or any(not isinstance(row, Mapping) for row in proposals):
        return {"accepted": False, "error": "evidence_output_not_array", "evidence": []}
    if len(proposals) > int(authority["maximum_pointers"]):
        return {"accepted": False, "error": "evidence_link_budget_exceeded", "evidence": []}
    response_ids = set(authority["allowed_response_sentence_ids"])
    source_ids = set(authority["allowed_source_sentence_ids"])
    normalized: dict[str, JsonDict] = {}
    try:
        for proposal in proposals:
            if set(proposal) != set(authority["pointer_fields"]):
                raise ValueError("evidence_output_fields_invalid")
            response_id = str(proposal.get("response_sentence_id") or "")
            if response_id not in response_ids or response_id in normalized:
                raise ValueError("response_sentence_id_invalid")
            links = proposal.get("source_sentence_ids")
            if (
                not isinstance(links, list)
                or len(links) != len(set(links))
                or any(str(value) not in source_ids for value in links)
            ):
                raise ValueError("source_sentence_id_invalid")
            relation = str(proposal.get("relation") or "")
            if relation not in authority["relations"]:
                raise ValueError("evidence_relation_invalid")
            entity = str(proposal.get("entity_type") or "")
            abstention = proposal.get("abstention_reason")
            if entity not in authority["entity_types"] or not isinstance(abstention, str):
                raise ValueError("evidence_output_value_invalid")
            if relation == authority["unknown_relation"]:
                if links or not abstention:
                    raise ValueError("unknown_relation_contract_invalid")
            elif not links or abstention:
                raise ValueError("linked_relation_contract_invalid")
            normalized[response_id] = {
                "response_sentence_id": response_id,
                "source_sentence_ids": [str(value) for value in links],
                "relation": relation,
                "entity_type": entity,
                "abstention_reason": abstention,
                "proposed_link": True,
            }
    except ValueError as error:
        return {"accepted": False, "error": str(error), "evidence": []}
    for response_id in response_ids - normalized.keys():
        normalized[response_id] = _unknown_row(response_id)
    return {
        "accepted": True,
        "error": None,
        "evidence": [normalized[key] for key in sorted(normalized)],
    }


def build_hashed_feature_row(record: Mapping[str, Any], outcome: Mapping[str, Any]) -> JsonDict:
    """Reduce accepted pointers to the existing eight features and bind the row."""

    if outcome.get("accepted") is not True:
        raise ValueError("accepted_evidence_required")
    contract = {
        "source_sentences": record["source_sentences"],
        "response_sentences": record["answer_sentences"],
    }
    reduced = evidence_protocol.reduce_evidence_features(
        contract, outcome["evidence"], raw_probability=0.5
    )
    features = reduced["features"]
    return {
        "component_hash": str(record["component_hash"]),
        "arm": "schema_fixture",
        "features": deepcopy(features),
        "feature_sha256": canonical_hash(features),
        "numerator": 1,
        "denominator": 1,
        "seed": RANDOM_SEED,
        "direction": "schema_conformance_only",
        "censored": False,
        "raw_provenance": "exact_canonical_fixture",
    }


def schema_case_rows(record: Mapping[str, Any]) -> list[JsonDict]:
    """Run every accepted and rejected conformance case through one validator."""

    linked = {
        "response_sentence_id": "R001",
        "source_sentence_ids": ["S001"],
        "relation": "supports",
        "entity_type": "other",
        "abstention_reason": "",
    }
    cases: list[tuple[str, Any, str, bool, str | None]] = [
        ("valid_link", [linked], "stop", True, None),
        (
            "invalid_enum",
            [{**linked, "entity_type": "code_snippet"}],
            "stop",
            False,
            "evidence_output_value_invalid",
        ),
        (
            "invalid_source_id",
            [{**linked, "source_sentence_ids": ["S999"]}],
            "stop",
            False,
            "source_sentence_id_invalid",
        ),
        (
            "invalid_response_id",
            [{**linked, "response_sentence_id": "R999"}],
            "stop",
            False,
            "response_sentence_id_invalid",
        ),
        (
            "extra_keys",
            [{**linked, "extra": True}],
            "stop",
            False,
            "evidence_output_fields_invalid",
        ),
        ("too_many_pointers", [linked] * 7, "stop", False, "evidence_link_budget_exceeded"),
        (
            "unknown_with_sources",
            [{**linked, "relation": "unknown", "abstention_reason": "missing"}],
            "stop",
            False,
            "unknown_relation_contract_invalid",
        ),
        (
            "linked_without_sources",
            [{**linked, "source_sentence_ids": []}],
            "stop",
            False,
            "linked_relation_contract_invalid",
        ),
        ("truncated_finish", [linked], "length", False, "truncated_output"),
        ("truncated_json", "[{", "stop", False, "truncated_json"),
        (
            "duplicate_source_id",
            [{**linked, "source_sentence_ids": ["S001", "S001"]}],
            "stop",
            False,
            "source_sentence_id_invalid",
        ),
    ]
    rows: list[JsonDict] = []
    for name, payload, finish_reason, expected, expected_error in cases:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        outcome = validate_evidence_output(record, text, finish_reason=finish_reason)
        observed = outcome["accepted"] is True
        rows.append(
            {
                "case": name,
                "expected_acceptance": expected,
                "observed_acceptance": observed,
                "expected_rejection": expected_error,
                "observed_rejection": outcome["error"],
                "passed": observed is expected and outcome["error"] == expected_error,
            }
        )
    return rows


def _load_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def replay_exp7604_outputs(root: Path) -> list[JsonDict]:
    """Authenticate and replay all eight outputs without changing their status."""

    resolved = root.resolve()
    artifact = _load_json(resolved / EXP7604_RESULT)
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:  # pragma: no cover - custody corruption.
        raise ValueError("exp7604_exactly_eight_rows_required")
    replay: list[JsonDict] = []
    for row in rows:
        index = int(row.get("pilot_index") or 0)
        request_path = resolved / EXP7604_RUN / "requests" / f"{index:02d}.json"
        response_path = resolved / EXP7604_RUN / "responses" / f"{index:02d}.json"
        if not request_path.is_file() or not response_path.is_file():  # pragma: no cover
            raise ValueError(f"exp7604_raw_pair_missing:{index}")
        request_hash = sha256_file(request_path)
        response_hash = sha256_file(response_path)
        raw_response = _load_json(response_path)
        choices = raw_response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message") if isinstance(choice, Mapping) else {}
        response_text = message.get("content") if isinstance(message, Mapping) else None
        if (  # pragma: no cover - byte mutation is checked before production reduction.
            request_hash != row.get("request_sha256")
            or response_hash != row.get("response_sha256")
            or response_text != row.get("response_text")
        ):
            raise ValueError(f"exp7604_raw_pair_hash_mismatch:{index}")
        outcome = validate_evidence_output(
            row["input_record"], str(response_text), finish_reason=str(row["finish_reason"])
        )
        observed = outcome.get("error")
        original = str(row.get("parser_error") or "")
        replay.append(
            {
                "pilot_index": index,
                "unit_id": str(row["component_hash"]),
                "arm": "historical_raw_replay",
                "original_parser_outcome": row.get("parser_outcome"),
                "original_parser_error": original,
                "observed_error": observed,
                "original_rejection_preserved": (
                    original == "ValueError:evidence_output_value_invalid"
                    and observed == "evidence_output_value_invalid"
                ),
                "semantic_verdict": None,
                "numerator": 0,
                "denominator": 1,
                "seed": int(row.get("seed") or 0),
                "direction": "historical_rejection_must_remain_rejected",
                "censoring": "invalid_output",
                "request_path": request_path.relative_to(resolved).as_posix(),
                "request_sha256": request_hash,
                "response_path": response_path.relative_to(resolved).as_posix(),
                "response_sha256": response_hash,
                "raw_provenance": "exp7604_immutable_raw_generation",
            }
        )
    if [row["pilot_index"] for row in replay] != list(range(1, 9)):  # pragma: no cover
        raise ValueError("exp7604_row_order_invalid")
    return replay


def authenticate_role_contract(root: Path) -> JsonDict:
    """Authenticate the Exp7602 manifest, role roster, and every sidecar."""

    resolved = root.resolve()
    result_path = resolved / EXP7602_RESULT
    protocol_path = resolved / EXP7602_PROTOCOL
    artifact = _load_json(result_path)
    protocol = _load_json(protocol_path)
    checksum_ok = artifact.get(
        "reproducibility_checksum"
    ) == roles_protocol.reproducibility_checksum(artifact)
    sidecars: list[JsonDict] = []
    for reader, values in sorted((protocol.get("reader_sidecars") or {}).items()):
        if not isinstance(values, Mapping):  # pragma: no cover - frozen manifest shape.
            continue
        for role, receipt in sorted(values.items()):
            path = resolved / str(receipt.get("path") or "missing")
            row_count = 0
            if path.is_file():
                with path.open(encoding="utf-8") as stream:
                    row_count = sum(1 for line in stream if line.strip())
            observed = {
                "bytes": path.stat().st_size if path.is_file() else 0,
                "rows": row_count,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            expected = {name: receipt.get(name) for name in ("bytes", "rows", "sha256")}
            sidecars.append(
                {
                    "reader": reader,
                    "role": role,
                    "path": path.relative_to(resolved).as_posix()
                    if path.is_relative_to(resolved)
                    else str(path),
                    "expected": expected,
                    "observed": observed,
                    "authenticated": observed == expected,
                }
            )
    roster = protocol.get("roster") or []
    scored = [row for row in roster if row.get("role") != "pilot"]
    pilots = [row for row in roster if row.get("role") == "pilot"]
    scored_ids = {str(row.get("component_hash")) for row in scored}
    pilot_ids = {str(row.get("component_hash")) for row in pilots}
    source_counts = protocol.get("source_role_counts") or {}
    role_counts = protocol.get("role_counts") or {}
    fit_counts = protocol.get("fit_partition_counts") or {}
    ready = bool(
        checksum_ok
        and artifact.get("evidence_protocol_ready_score") == 1
        and protocol.get("selection_salt") == evidence_protocol.SELECTION_SALT
        and sum(int(value) for value in source_counts.values()) == 480
        and len(scored) == 240
        and len(pilots) == 8
        and not (scored_ids & pilot_ids)
        and fit_counts == {"optimization": 64, "old_distribution_anchor": 16}
        and len(sidecars) == 12
        and all(row["authenticated"] for row in sidecars)
        and protocol.get("fresh_confirmatory_claim_allowed") is False
    )
    return {
        "ready": ready,
        "role_manifest_path": EXP7602_PROTOCOL.as_posix(),
        "role_manifest_sha256": sha256_file(protocol_path),
        "terminal_artifact_path": EXP7602_RESULT.as_posix(),
        "terminal_artifact_sha256": sha256_file(result_path),
        "terminal_checksum_authenticated": checksum_ok,
        "selection_salt": protocol.get("selection_salt"),
        "restored_group_count": sum(int(value) for value in source_counts.values()),
        "selected_scored_group_count": len(scored),
        "role_counts": deepcopy(role_counts),
        "fit_partition_counts": {
            "optimization": fit_counts.get("optimization"),
            "anchor": fit_counts.get("old_distribution_anchor"),
        },
        "pilot_count": len(pilots),
        "pilot_disjoint": not bool(scored_ids & pilot_ids),
        "fresh_confirmatory_claim_allowed": False,
        "sidecars": sidecars,
    }


def _llama_runtime_receipt() -> JsonDict:
    """Authenticate the installed schema compiler without loading model weights."""

    import llama_cpp
    from llama_cpp import llama_grammar

    module_path = Path(str(llama_cpp.__file__)).resolve()
    grammar_path = Path(str(llama_grammar.__file__)).resolve()
    return {
        "package": "llama-cpp-python",
        "version": str(llama_cpp.__version__),
        "module_path": str(module_path),
        "module_sha256": sha256_file(module_path),
        "grammar_adapter_path": str(grammar_path),
        "grammar_adapter_sha256": sha256_file(grammar_path),
        "model_loaded": False,
    }


def capture_configuration(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze the no-model schema rendering and validation configuration."""

    authorities = [build_schema_authority(record) for record in records]
    prompts = [render_system_prompt(authority) for authority in authorities]
    grammars = [compile_decoder_grammar(authority) for authority in authorities]
    return {
        "planned_inference_substrate_class": "no_model_load",
        "actual_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF",
        "temperature": 0.0,
        "seed": RANDOM_SEED,
        "maximum_pointers": MAX_POINTERS,
        "pointer_fields": list(POINTER_FIELDS),
        "relations": list(RELATIONS),
        "entity_types": list(ENTITY_TYPES),
        "prompt_sha256_by_component": [
            {
                "component_hash": str(record["component_hash"]),
                "sha256": text_sha256(prompt),
            }
            for record, prompt in zip(records, prompts, strict=True)
        ],
        "decoder_schema_sha256_by_component": [
            {
                "component_hash": str(record["component_hash"]),
                "sha256": canonical_hash(authority["json_schema"]),
            }
            for record, authority in zip(records, authorities, strict=True)
        ],
        "compiled_grammar_sha256_by_component": [
            {
                "component_hash": str(record["component_hash"]),
                "sha256": text_sha256(grammar),
            }
            for record, grammar in zip(records, grammars, strict=True)
        ],
        "unsupported_decoder_keywords": list(UNSUPPORTED_DECODER_KEYWORDS),
        "unsupported_keyword_policy": "independent_validator_fail_closed",
        "independent_validator": f"{__name__}.validate_evidence_output",
        "runtime": _llama_runtime_receipt(),
    }


def _write_frozen_json(path: Path, value: Any) -> None:
    """Write a sidecar once and reject a later byte-level contract change."""

    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"frozen_sidecar_conflict:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def freeze_protocol_sidecars(raw_dir: Path, records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze schema, configuration, and canonical model inputs."""

    authorities = [
        {
            "component_hash": str(record["component_hash"]),
            "authority": build_schema_authority(record),
            "system_prompt": render_system_prompt(build_schema_authority(record)),
            "compiled_grammar": compile_decoder_grammar(build_schema_authority(record)),
        }
        for record in records
    ]
    values = {
        "schema_authority": {
            "schema": "carnot.exp7616.schema_authority.v1",
            "authorities": authorities,
        },
        "capture_configuration": capture_configuration(records),
        "canonical_inputs": [build_canonical_input(record) for record in records],
    }
    receipts: list[JsonDict] = []
    for name, value in values.items():
        path = raw_dir / f"{name}.json"
        _write_frozen_json(path, value)
        receipts.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return receipts


def run_guarded_update_restart_e2e(root: Path, state_dir: Path) -> JsonDict:
    """Persist and reload the authenticated Exp7603 lifecycle without an LLM."""

    resolved = root.resolve()
    artifact_path = resolved / EXP7603_RESULT
    artifact = _load_json(artifact_path)
    checksum_ok = artifact.get(
        "reproducibility_checksum"
    ) == update_fixture.reproducibility_checksum(artifact)
    config = update_fixture.UpdateConfig()
    contract = artifact.get("update_rule_contract") or {}
    configuration_matches = bool(
        contract.get("feedback_lag") == config.lag
        and contract.get("parameter_bound") == config.parameter_bound
        and contract.get("anchor_brier_increase_max") == config.anchor_tolerance
        and contract.get("state_schema") == "carnot.guarded_update_state.v1"
    )
    _stream, anchors = update_fixture.load_authenticated_inputs(resolved)
    state_path = state_dir / "guarded-restart.json"
    lifecycle = update_fixture.UpdateLifecycle(
        config=config, arm="guarded", anchors=anchors, state_path=state_path
    )
    before = lifecycle.parameter_hash
    snapshot = lifecycle.persist()
    restarted = update_fixture.UpdateLifecycle.reload(
        state_path=state_path, config=config, arm="guarded", anchors=anchors
    )
    after = restarted.parameter_hash
    authenticated = bool(
        checksum_ok
        and configuration_matches
        and artifact.get("guarded_update_ready_score") == 1
        and artifact.get("verdict_class") == "circular_positive"
    )
    return {
        "authenticated": authenticated,
        "upstream_path": EXP7603_RESULT.as_posix(),
        "upstream_sha256": sha256_file(artifact_path),
        "upstream_reproducibility_checksum": artifact.get("reproducibility_checksum"),
        "checksum_authenticated": checksum_ok,
        "configuration": asdict(config),
        "configuration_matches": configuration_matches,
        "upstream_guarded_update_ready_score": artifact.get("guarded_update_ready_score"),
        "before_parameter_hash": before,
        "after_parameter_hash": after,
        "restart_parity": before == after,
        "snapshot": snapshot,
        "model_calls": 0,
        "fixture_positive_class": "circular_positive",
    }


def precondition(
    check: str,
    upstream: str,
    path: Path,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Record exact gate operands so an external block is actionable."""

    passed = {
        "eq": observed == expected,
        "contains": isinstance(observed, str) and str(expected) in observed,
        "gte": isinstance(observed, (int, float)) and observed >= expected,
    }[operator]
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
    }


def _source_row(root: Path, relative: Path, producer: str, source_class: str) -> JsonDict:
    path = root / relative
    return {
        "producer": producer,
        "source_class": source_class if path.is_file() else "missing_artifact",
        "path": relative.as_posix(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate named inputs and declared local tool versions before reduction."""

    from importlib.metadata import version

    resolved = root.resolve()
    named = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7588_v663_evidence_protocol.py"),
        Path("python/carnot/experiment_7602_v664_evidence_requalification.py"),
        Path("python/carnot/experiment_7603_v664_guarded_update_fixture.py"),
        Path("python/carnot/experiment_7604_v664_evidence_pilot.py"),
        EXP7602_RESULT,
        EXP7602_PROTOCOL,
        EXP7603_RESULT,
        EXP7604_RESULT,
        SPEC_PATH,
    )
    checks = [
        precondition(
            f"named_input:{relative.as_posix()}",
            "task_contract",
            resolved / relative,
            "filesystem.is_file_nonempty",
            "eq",
            True,
            (resolved / relative).is_file() and (resolved / relative).stat().st_size > 0,
        )
        for relative in named
    ]
    tool_versions = {
        "pytest": ("pytest", "9.0.3"),
        "ruff": ("ruff", "0.15.17"),
        "mypy": ("mypy", "2.2.0"),
        "llama-cpp-python": ("llama-cpp-python", "0.3.33"),
    }
    for label, (package, expected) in tool_versions.items():
        observed = version(package)
        checks.append(
            precondition(
                f"tool_version:{label}",
                "local_venv",
                resolved / ".venv",
                label,
                "eq",
                expected,
                observed,
            )
        )
    if any(row["passed"] is not True for row in checks):
        return checks, []
    spec_text = (resolved / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        precondition(
            "driving_requirement",
            "research_reporting_spec",
            resolved / SPEC_PATH,
            "REQ-*",
            "contains",
            "REQ-REPORT-7616",
            spec_text,
        )
    )
    role = authenticate_role_contract(resolved)
    checks.append(
        precondition(
            "exp7602_role_contract",
            "exp7602",
            resolved / EXP7602_PROTOCOL,
            "ready",
            "eq",
            True,
            role["ready"],
        )
    )
    for row in role["sidecars"]:
        checks.append(
            precondition(
                f"role_sidecar:{row['reader']}:{row['role']}",
                "exp7602",
                resolved / row["path"],
                "bytes_rows_sha256",
                "eq",
                row["expected"],
                row["observed"],
            )
        )
    exp7603 = _load_json(resolved / EXP7603_RESULT)
    recomputed = update_fixture.reproducibility_checksum(exp7603)
    checks.append(
        precondition(
            "exp7603_reproducibility_checksum",
            "exp7603",
            resolved / EXP7603_RESULT,
            "reproducibility_checksum",
            "eq",
            exp7603.get("reproducibility_checksum"),
            recomputed,
        )
    )
    rows = replay_exp7604_outputs(resolved)
    checks.append(
        precondition(
            "exp7604_original_rejections",
            "exp7604",
            resolved / EXP7604_RESULT,
            "rows[*].parser_error",
            "eq",
            {"count": 8, "error": "evidence_output_value_invalid"},
            {
                "count": len(rows),
                "error": (
                    "evidence_output_value_invalid"
                    if all(row["original_rejection_preserved"] for row in rows)
                    else "mismatch"
                ),
            },
        )
    )
    sources = [
        _source_row(resolved, EXP7602_RESULT, "exp7602_terminal", "authenticated_producer"),
        _source_row(resolved, EXP7602_PROTOCOL, "exp7602_role_manifest", "authenticated_producer"),
        _source_row(resolved, EXP7603_RESULT, "exp7603_lifecycle", "authenticated_producer"),
        _source_row(resolved, EXP7604_RESULT, "exp7604_terminal", "authenticated_producer"),
    ]
    for index in range(1, 9):
        for kind in ("requests", "responses"):
            relative = EXP7604_RUN / kind / f"{index:02d}.json"
            sources.append(
                _source_row(
                    resolved, relative, f"exp7604_raw_{kind[:-1]}", "immutable_raw_evidence"
                )
            )
    return checks, sources


def _acceptance_gates(valid: bool, ready: bool, retained: bool) -> list[JsonDict]:
    """Keep protocol readiness separate from benefit, retention, and freshness."""

    return [
        {
            "check": "authenticated_schema_inputs",
            "category": "validity",
            "condition": "all preconditions and affected validation pass",
            "passed": valid,
            "principle": "Only authenticated bytes and passing checks support protocol claims.",
        },
        {
            "check": "schema_role_lifecycle_ready",
            "category": "readiness",
            "condition": "schema, role, and guarded-update readiness scores all equal one",
            "passed": ready,
            "principle": "Protocol mechanics can be ready without establishing predictive benefit.",
        },
        {
            "check": "empirical_benefit",
            "category": "benefit",
            "condition": "fresh held-out evidence beats registered controls",
            "passed": False,
            "principle": "Exact fixtures and historical replay cannot establish learned advantage.",
        },
        {
            "check": "historical_row_retention",
            "category": "retention",
            "condition": "all eight attempted generations retain their original rejection",
            "passed": retained,
            "principle": "A schema repair must not erase or relabel prior failed evidence.",
        },
        {
            "check": "fresh_confirmatory_claim",
            "category": "freshness",
            "condition": "the evaluated groups were not historically exposed",
            "passed": False,
            "principle": "Immutable historical exposure keeps confirmation claims closed.",
        },
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]], result_path: Path) -> JsonDict:
    failed = [row for row in gates if row.get("passed") is not True]
    first = failed[0] if failed else None
    diagnostic = (
        {
            "check": first["check"],
            "upstream": EXPERIMENT_ID,
            "path": str(result_path.resolve()),
            "field": f"acceptance_gate_results.{first['category']}.passed",
            "operator": "eq",
            "expected": True,
            "observed": False,
            "passed": False,
            "principle": first["principle"],
        }
        if first
        else None
    )
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row["check"]) for row in failed],
        "first_failure": diagnostic,
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and isinstance(by_name[name].get("log_sha256"), str)
        for name in names
    )


def _adversarial_flag(receipts: Sequence[Mapping[str, Any]]) -> bool:
    receipt = next((row for row in receipts if row.get("name") == "adversarial_verify"), None)
    if not isinstance(receipt, Mapping):
        return False
    tail = str(receipt.get("output_tail") or "")
    match = re.search(r"Scanned\s+\d+\s+artifact\(s\);\s+(\d+)\s+flagged", tail)
    return bool(match and int(match.group(1)) > 0)


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    specific = {
        "honest_verdict": "A complete prefix marks finished work, not scientific benefit.",
        "verdict_class": "The closed class keeps protocol readiness distinct from a positive claim.",
        "flagged_adversarial": "Persist the terminal reader result; flagged evidence opens no gate.",
        "gate_check_summary": "Each failed gate exposes exact operands and its governing principle.",
        "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness remain separate.",
        "rows": "Each historical generation remains one independent unit with raw provenance.",
        "sample_size_budget": "Replays, views, and fixtures do not multiply independent samples.",
        "preconditions_checked": "Unavailable external inputs block dependent work without fabrication.",
        "inference_substrate": "This run performs CPU reduction and no current model invocation.",
        "inference_substrate_class": "The actual class is measured separately from historical GPU evidence.",
        "MODEL_SPECS": "No-model work uses an empty current model list.",
        "model_invoked": "Historical calls do not become current calls during replay.",
        "execution_venue": "The actual host is named; historical devices are only provenance.",
        "phase_spans": "Disjoint monotonic spans retain completed units and checkpoint positions.",
        "invocation_counts": "Loads, forwards, generations, and tokens are typed current-run zeros.",
        "duration_s": "Monotonic duration covers only current execution and is never padded.",
        "random_seed": "The deterministic fixture seed is recorded with its purpose.",
        "reproducibility_checksum": "Immutable inputs, configuration, and reductions bind the result.",
        "source_artifact_hashes": "Producers, raw evidence, current outputs, and absences stay distinct.",
        "validation_receipts": "Commands, exits, worktree paths, and log hashes remain auditable.",
        "verifier_is_oracle": "Exact fixtures prove mechanics only and cannot establish learned advantage.",
        "evidence_schema_ready_score": "One requires prompt, decoder, validator, and reconstruction conformance.",
        "role_contract_ready_score": "One requires exact disjoint roles and every authenticated sidecar.",
        "guarded_update_ready_score": "One requires authenticated lifecycle configuration and restart parity.",
        "schema_path": "The shared frozen authority path and byte hash are published together.",
        "role_manifest_path": "The manifest and each role sidecar path remain explicit downstream inputs.",
        "fresh_confirmatory_claim_allowed": "Historical exposure is immutable, so this remains false.",
        "schema_case_rows": "Every conformance case carries expected and observed acceptance or rejection.",
    }
    return {
        key: specific.get(key, "Persist this field so the terminal claim can be replayed.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash terminal content without recursively hashing the checksum itself."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _pilot_records(root: Path) -> list[JsonDict]:
    artifact = _load_json(root.resolve() / EXP7604_RESULT)
    rows = artifact.get("rows") or []
    if len(rows) != 8:  # pragma: no cover - authenticated upstream shape.
        raise ValueError("exp7604_exactly_eight_rows_required")
    return [deepcopy(dict(row["input_record"])) for row in rows]


def build_artifact(
    root: Path,
    *,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    role_receipt: Mapping[str, Any],
    lifecycle_receipt: Mapping[str, Any],
    frozen_sidecars: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a protocol-ready null without converting fixtures into benefit."""

    records = _pilot_records(root)
    cases = schema_case_rows(records[0])
    reconstructions = []
    for record in records:
        contract = build_canonical_input(record)
        reconstructions.append(
            {
                "component_hash": record["component_hash"],
                "source_sha256": text_sha256(reconstruct_canonical_text(contract, "source")),
                "question_sha256": text_sha256(reconstruct_canonical_text(contract, "question")),
                "response_sha256": text_sha256(reconstruct_canonical_text(contract, "response")),
                "exact_reconstruction": True,
                "sentence_order_preserved": True,
            }
        )
    valid_outcome = validate_evidence_output(
        records[0],
        json.dumps(
            [
                {
                    "response_sentence_id": records[0]["answer_sentences"][0]["sentence_id"],
                    "source_sentence_ids": [records[0]["source_sentences"][0]["sentence_id"]],
                    "relation": "supports",
                    "entity_type": "other",
                    "abstention_reason": "",
                }
            ]
        ),
        finish_reason="stop",
    )
    invalid_outcome = validate_evidence_output(
        records[0],
        json.dumps(
            [
                {
                    "response_sentence_id": records[0]["answer_sentences"][0]["sentence_id"],
                    "source_sentence_ids": [records[0]["source_sentences"][0]["sentence_id"]],
                    "relation": "supports",
                    "entity_type": "code_snippet",
                    "abstention_reason": "",
                }
            ]
        ),
        finish_reason="stop",
    )
    feature_row = build_hashed_feature_row(records[0], valid_outcome)
    schema_receipt = next(row for row in frozen_sidecars if row["name"] == "schema_authority")
    configuration_receipt = next(
        row for row in frozen_sidecars if row["name"] == "capture_configuration"
    )
    inputs_receipt = next(row for row in frozen_sidecars if row["name"] == "canonical_inputs")
    schema_ready = bool(
        all(row["passed"] for row in cases)
        and all(row["exact_reconstruction"] for row in reconstructions)
        and valid_outcome["accepted"] is True
        and invalid_outcome["accepted"] is False
        and schema_receipt.get("sha256")
    )
    role_ready = role_receipt.get("ready") is True
    lifecycle_ready = bool(
        lifecycle_receipt.get("authenticated") is True
        and lifecycle_receipt.get("restart_parity") is True
    )
    affected_valid = all(row.get("passed") is True for row in checks) and _receipts_pass(
        validation_receipts, validation_scope.REQUIRED_CHECK_NAMES
    )
    ready = schema_ready and role_ready and lifecycle_ready
    retained = len(replay_rows) == 8 and all(
        row.get("original_rejection_preserved") is True for row in replay_rows
    )
    gates = _acceptance_gates(affected_valid, ready, retained)
    terminal = [
        deepcopy(dict(row))
        for row in validation_receipts
        if row.get("name") in TERMINAL_CHECK_NAMES
    ]
    current_receipt = build_current_work_receipt(
        run_id=EXPERIMENT_ID,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_schema_reduction_historical_replay_no_llm",
        inference_substrate_details={
            "model_loaded": False,
            "historical_gpu_rows_replayed": 8,
            "decoder_grammar_compiled": True,
        },
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        phase_spans=phase_spans,
    )
    combined_sources = [deepcopy(dict(row)) for row in source_hashes]
    combined_sources.extend(
        {
            "producer": EXPERIMENT_ID,
            "source_class": "current_frozen_output",
            "path": row["path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in frozen_sidecars
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7616,
        "title": "Freeze a decoder-aligned evidence schema",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": "complete_null_evidence_schema_ready",
        "verdict_class": "null",
        "positive_claim": False,
        "flagged_adversarial": _adversarial_flag(validation_receipts),
        "gate_check_summary": _gate_summary(gates, root / RESULT_PATH),
        "acceptance_gate_results": gates,
        "rows": [deepcopy(dict(row)) for row in replay_rows],
        "sample_size_budget": {
            "intended": 8,
            "observed": 8,
            "excluded": 0,
            "censored": 8,
            "independent_unit": "exp7604_pilot_generation",
            "seeds_views_replays_multiply_independent_samples": False,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": "cpu_schema_reduction_historical_replay_no_llm",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF",
        "model_invoked": False,
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "physical_device": "host_cpu",
            "gpu_uuid": None,
            "gpu_used": False,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "duration_s": max(0.0, float(duration_s)),
        "random_seed": RANDOM_SEED,
        "random_seeds": [{"seed": RANDOM_SEED, "purpose": "deterministic_schema_fixture"}],
        "source_artifact_hashes": combined_sources,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "terminal_reader_outcomes": terminal,
        "current_work_receipt": current_receipt,
        "verifier_is_oracle": True,
        "evidence_schema_ready_score": int(schema_ready),
        "role_contract_ready_score": int(role_ready),
        "guarded_update_ready_score": int(lifecycle_ready),
        "schema_path": {
            "path": schema_receipt["path"],
            "sha256": schema_receipt["sha256"],
        },
        "capture_configuration_path": {
            "path": configuration_receipt["path"],
            "sha256": configuration_receipt["sha256"],
        },
        "canonical_inputs_path": {
            "path": inputs_receipt["path"],
            "sha256": inputs_receipt["sha256"],
        },
        "role_manifest_path": {
            "path": role_receipt["role_manifest_path"],
            "sha256": role_receipt["role_manifest_sha256"],
            "role_sidecar_paths": [row["path"] for row in role_receipt["sidecars"]],
        },
        "role_contract_receipt": deepcopy(dict(role_receipt)),
        "guarded_update_lifecycle_receipt": deepcopy(dict(lifecycle_receipt)),
        "schema_case_rows": cases,
        "reconstruction_receipts": reconstructions,
        "e2e_results": {
            "canonical_prompt_sha256": text_sha256(
                build_extraction_request(records[0])["messages"][1]["content"]
            ),
            "valid_parser_accepted": True,
            "invalid_parser_rejected": invalid_outcome["error"],
            "hashed_feature_row": feature_row,
            "guarded_update_restart_parity": lifecycle_receipt["restart_parity"],
            "model_calls": 0,
            "fixture_positive_class": "circular_positive",
            "protocol_readiness_class": "null",
        },
        "fresh_confirmatory_claim_allowed": False,
        "empirical_benefit_measured": False,
        "scope_retirement": {
            "schema_protocol_failure_retired": schema_ready,
            "scientific_hypothesis_retired": False,
            "principle": "Schema readiness retires only the exact protocol failure, not unmeasured science.",
        },
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "generator_weights_immutable": True,
        "default_promotion_authorized": False,
        "research_conductor_modified": False,
        "research_roadmap_modified": False,
        "applicable_e2e": "canonical_prompt_parser_feature_row_and_guarded_restart_no_llm",
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(checks: Sequence[Mapping[str, Any]], *, duration_s: float) -> JsonDict:
    """Publish an external block with exact diagnostics and no invented work."""

    failure = next(row for row in checks if row.get("passed") is not True)
    reason = re.sub(r"[^a-z0-9]+", "_", str(failure["check"]).lower()).strip("_")
    rows = [
        {
            "unit_id": f"unstarted-{index}",
            "arm": "historical_raw_replay",
            "numerator": 0,
            "denominator": 1,
            "seed": RANDOM_SEED,
            "direction": "historical_rejection_must_remain_rejected",
            "censoring": "external_precondition_block",
            "raw_provenance": None,
        }
        for index in range(1, 9)
    ]
    gates = _acceptance_gates(False, False, False)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7616,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete_blocked",
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "positive_claim": False,
        "flagged_adversarial": False,
        "gate_check_summary": {
            "passed": False,
            "failed_count": 1,
            "failed_checks": [failure["check"]],
            "first_failure": deepcopy(dict(failure)),
        },
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended": 8,
            "observed": 0,
            "excluded": 0,
            "censored": 8,
            "independent_unit": "exp7604_pilot_generation",
            "seeds_views_replays_multiply_independent_samples": False,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": "blocked_no_run",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate_class": "blocked_no_run",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "execution_venue": "host",
        "phase_spans": [],
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "duration_s": max(0.0, duration_s),
        "random_seed": RANDOM_SEED,
        "random_seeds": [{"seed": RANDOM_SEED, "purpose": "deterministic_schema_fixture"}],
        "source_artifact_hashes": [],
        "validation_receipts": [],
        "terminal_reader_outcomes": [],
        "verifier_is_oracle": True,
        "evidence_schema_ready_score": 0,
        "role_contract_ready_score": 0,
        "guarded_update_ready_score": 0,
        "schema_path": None,
        "role_manifest_path": None,
        "schema_case_rows": [],
        "fresh_confirmatory_claim_allowed": False,
        "empirical_benefit_measured": False,
        "external_publication_authorized": False,
        "generator_weights_immutable": True,
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(
    root: Path, work_dir: Path, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a complete fixture artifact through the same reusable reducers."""

    checks, sources = collect_preconditions(root)
    records = _pilot_records(root)
    replay = replay_exp7604_outputs(root)
    role = authenticate_role_contract(root)
    lifecycle = run_guarded_update_restart_e2e(root, work_dir / "state")
    sidecars = freeze_protocol_sidecars(work_dir, records)
    duration_ns = 10_000_000
    return build_artifact(
        root.resolve(),
        checks=checks,
        source_hashes=sources,
        replay_rows=replay,
        role_receipt=role,
        lifecycle_receipt=lifecycle,
        frozen_sidecars=sidecars,
        validation_receipts=validation_receipts,
        duration_s=duration_ns / 1_000_000_000,
        started_monotonic_ns=0,
        ended_monotonic_ns=duration_ns,
        phase_spans=[],
    )


def _resolve_artifact_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_artifact(
    value: object, *, root: Path = ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check claims, hashes, rows, gates, and no-model provenance."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_identity_mismatch")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_verdict_prefix_required")
    if artifact.get("verdict_class") not in TERMINAL_CLASSES:
        errors.append("verdict_class_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("model_specs_must_be_empty")
    if artifact.get("model_invoked") is not False:
        errors.append("model_invoked_must_be_false")
    counts = artifact.get("invocation_counts")
    if not isinstance(counts, Mapping) or counts != ZERO_INVOCATION_COUNTS:
        errors.append("current_model_calls_nonzero")
    if artifact.get("fresh_confirmatory_claim_allowed") is not False:
        errors.append("freshness_invalid")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_oracle_class_invalid")
    if artifact.get("flagged_adversarial") not in {True, False}:
        errors.append("adversarial_flag_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not principles.get(key) for key in artifact if key != "reproducibility_checksum"
    ):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        first = (artifact.get("gate_check_summary") or {}).get("first_failure")
        required = {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        if not isinstance(first, Mapping) or required - set(first):
            errors.append("blocked_gate_summary_incomplete")
        return list(dict.fromkeys(errors))
    if (
        artifact.get("planned_inference_substrate_class") != "no_model_load"
        or artifact.get("inference_substrate_class") != "no_model_load"
    ):
        errors.append("substrate_class_invalid")
    rows = artifact.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != 8
        or any(
            row.get("original_parser_error") != "ValueError:evidence_output_value_invalid"
            or row.get("observed_error") != "evidence_output_value_invalid"
            or row.get("original_rejection_preserved") is not True
            for row in rows
        )
    ):
        errors.append("historical_replay_invalid")
    cases = artifact.get("schema_case_rows")
    schema_ready = bool(
        isinstance(cases, list)
        and len(cases) >= 10
        and all(row.get("passed") is True for row in cases)
        and all(
            row.get("exact_reconstruction") is True
            for row in artifact.get("reconstruction_receipts") or []
        )
    )
    if artifact.get("evidence_schema_ready_score") != int(schema_ready):
        errors.append("schema_ready_score_mismatch")
    role = artifact.get("role_contract_receipt")
    role_ready = (
        isinstance(role, Mapping)
        and role.get("ready") is True
        and all(row.get("authenticated") is True for row in role.get("sidecars") or [])
    )
    if artifact.get("role_contract_ready_score") != int(role_ready):
        errors.append("role_ready_score_mismatch")
    lifecycle = artifact.get("guarded_update_lifecycle_receipt")
    lifecycle_ready = bool(
        isinstance(lifecycle, Mapping)
        and lifecycle.get("authenticated") is True
        and lifecycle.get("restart_parity") is True
    )
    if artifact.get("guarded_update_ready_score") != int(lifecycle_ready):
        errors.append("guarded_update_ready_score_mismatch")
    for name in ("schema_path", "capture_configuration_path", "canonical_inputs_path"):
        receipt = artifact.get(name)
        path = (
            _resolve_artifact_path(root, str(receipt.get("path") or ""))
            if isinstance(receipt, Mapping)
            else root / "missing"
        )
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            errors.append(f"{name}_hash_mismatch")
    gates = artifact.get("acceptance_gate_results") or []
    categories = {row.get("category") for row in gates if isinstance(row, Mapping)}
    if categories != {"validity", "readiness", "benefit", "retention", "freshness"}:
        errors.append("acceptance_gate_shape_invalid")
    elif next(row for row in gates if row["category"] == "benefit")["passed"] is not False:
        errors.append("benefit_gate_invalid")
    receipt = artifact.get("current_work_receipt")
    if not isinstance(receipt, Mapping) or validate_current_work_receipt(receipt, root=root):
        errors.append("current_work_receipt_invalid")
    if require_terminal and not _receipts_pass(
        artifact.get("validation_receipts") or [],
        (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES),
    ):
        errors.append("terminal_validation_incomplete")
    return list(dict.fromkeys(errors))


def independent_reduce(value: object, *, root: Path = ROOT) -> JsonDict:
    """Recompute comparison counts without trusting stored score summaries."""

    errors = validate_artifact(value, root=root, require_terminal=False)
    if not isinstance(value, Mapping):
        return {"passed": False, "errors": errors}
    rows = value.get("rows") or []
    cases = value.get("schema_case_rows") or []
    reconstruction = value.get("reconstruction_receipts") or []
    reduction = {
        "historical_unit_count": len(rows),
        "historical_rejection_count": sum(
            row.get("original_rejection_preserved") is True for row in rows
        ),
        "semantic_verdict_count": sum(row.get("semantic_verdict") is not None for row in rows),
        "schema_case_count": len(cases),
        "schema_case_pass_count": sum(row.get("passed") is True for row in cases),
        "reconstruction_count": len(reconstruction),
        "exact_reconstruction_count": sum(
            row.get("exact_reconstruction") is True for row in reconstruction
        ),
        "feature_row_sha256": (value.get("e2e_results") or {})
        .get("hashed_feature_row", {})
        .get("feature_sha256"),
    }
    expected = {
        "historical_unit_count": 8,
        "historical_rejection_count": 8,
        "semantic_verdict_count": 0,
        "schema_case_count": len(cases),
        "schema_case_pass_count": len(cases),
        "reconstruction_count": 8,
        "exact_reconstruction_count": 8,
        "feature_row_sha256": reduction["feature_row_sha256"],
    }
    passed = not errors and reduction == expected and bool(reduction["feature_row_sha256"])
    return {
        "mode": "independent_reduction",
        "passed": passed,
        "errors": errors,
        "reduction": reduction,
        "expected": expected,
        "comparative_metrics_recomputed_from_rows": True,
        "row_reduction_sha256": canonical_hash(rows),
    }


def cold_replay(path: Path, *, root: Path = ROOT) -> JsonDict:
    """Reload one exact candidate and validate it without model work."""

    value = _load_json(path)
    errors = validate_artifact(value, root=root, require_terminal=False)
    return {
        "mode": "cold_replay",
        "passed": not errors,
        "errors": errors,
        "historical_unit_count": len(value.get("rows") or []),
    }


def affected_file_manifest() -> JsonDict:
    """Freeze the exact files governed by this task's scoped checks."""

    return {
        "schema": "carnot.exp7616.affected_files.v1",
        "requirement": "REQ-REPORT-7616",
        "files": [
            SPEC_PATH.as_posix(),
            MODULE_PATH.as_posix(),
            WRAPPER_PATH.as_posix(),
            TEST_PATH.as_posix(),
        ],
        "test_paths": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
    }


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build focused serial tests, coverage, Ruff, mypy, and spec coverage."""

    (private_root / "pytest").mkdir(parents=True, exist_ok=True)
    return validation_scope.build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=private_root / "pytest",
        coverage_file=private_root / ".coverage-exp7616",
    )


def terminal_commands(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build independent readers for the exact unpublished candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = WRAPPER_PATH.as_posix()
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--root", str(root), "--cold-replay", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--root", str(root), "--independent-reduce", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
    ]


def _normalize_receipts(receipts: Sequence[Mapping[str, Any]], root: Path) -> list[JsonDict]:
    normalized: list[JsonDict] = []
    for row in receipts:
        value = deepcopy(dict(row))
        log_path = Path(str(value.get("log_path") or ""))
        if log_path.is_absolute() and log_path.is_relative_to(root):
            value["log_path"] = log_path.relative_to(root).as_posix()
        normalized.append(value)
    return normalized


def _terminal_receipts_equal(
    expected: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]
) -> bool:
    fields = ("name", "command_argv", "exit_code", "passed", "log_sha256")
    return len(expected) == len(observed) and all(
        all(left.get(field) == right.get(field) for field in fields)
        for left, right in zip(expected, observed, strict=True)
    )


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit one flushed phase boundary with truthful elapsed time."""

    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_ID,
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 3),
                **details,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _span(
    phase: str, started: float, phase_started: float, planned: int, completed: int
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "started_offset_s": phase_started - started,
        "ended_offset_s": ended - started,
        "duration_s": ended - phase_started,
        "planned_units": planned,
        "completed_units": completed,
        "pending_operation": None,
        "checkpoint_position": completed,
    }


def run_experiment(  # pragma: no cover - the declared CLI is the capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Authenticate, replay, validate, and atomically publish Exp7616."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    resolved = root.resolve()
    destination = output_path or resolved / RESULT_PATH
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "before", root=str(resolved))
    phase_started = time.monotonic()
    checks, sources = collect_preconditions(resolved)
    spans.append(_span("preconditions", started, phase_started, len(checks), len(checks)))
    failed = next((row for row in checks if row["passed"] is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(checks, duration_s=time.monotonic() - started)
        progress(started, "publish", "before_atomic_blocked", check=failed["check"])
        atomic_json(destination, blocked)
        progress(started, "publish", "after_atomic_blocked", check=failed["check"])
        return blocked
    progress(started, "preconditions", "after", completed_units=len(checks))

    for phase in ("model_load", "generation", "benchmark"):
        phase_started = time.monotonic()
        progress(started, phase, "before", planned_units=0)
        spans.append(_span(phase, started, phase_started, 0, 0))
        progress(started, phase, "after", completed_units=0)

    progress(started, "historical_replay", "before", planned_units=8)
    phase_started = time.monotonic()
    replay = replay_exp7604_outputs(resolved)
    records = _pilot_records(resolved)
    role = authenticate_role_contract(resolved)
    spans.append(_span("historical_replay", started, phase_started, 8, len(replay)))
    progress(started, "historical_replay", "after", completed_units=len(replay))

    raw_root = resolved / RAW_DIR
    progress(started, "schema_freeze", "before", planned_units=3)
    phase_started = time.monotonic()
    frozen = freeze_protocol_sidecars(raw_root, records)
    manifest_path = raw_root / "affected_validation_manifest.json"
    atomic_json(manifest_path, affected_file_manifest())
    sources.append(
        _source_row(
            resolved,
            manifest_path.relative_to(resolved),
            "exp7616_affected_manifest",
            "current_frozen_output",
        )
    )
    for relative in (SPEC_PATH, MODULE_PATH, WRAPPER_PATH, TEST_PATH):
        sources.append(
            _source_row(resolved, relative, "exp7616_current_source", "conductor_pre_gate_record")
        )
    spans.append(_span("schema_freeze", started, phase_started, 3, len(frozen)))
    progress(started, "schema_freeze", "after", completed_units=len(frozen))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7616-validation-", dir="/tmp"))
    progress(started, "guarded_restart_e2e", "before_benchmark", planned_units=1)
    phase_started = time.monotonic()
    lifecycle = run_guarded_update_restart_e2e(resolved, private_root / "state")
    spans.append(_span("guarded_restart_e2e", started, phase_started, 1, 1))
    progress(started, "guarded_restart_e2e", "after_benchmark", completed_units=1)

    commands = build_validation_commands(resolved, private_root)
    progress(started, "affected_validation", "before_subprocesses", planned_units=len(commands))
    phase_started = time.monotonic()
    affected = _normalize_receipts(
        validation_scope.run_commands(
            resolved,
            commands,
            log_dir=raw_root / "validation" / "affected",
            heartbeat_s=60.0,
        ),
        resolved,
    )
    spans.append(_span("affected_validation", started, phase_started, len(commands), len(affected)))
    progress(started, "affected_validation", "after_subprocesses", completed_units=len(affected))

    ended_ns = time.monotonic_ns()
    candidate = build_artifact(
        resolved,
        checks=checks,
        source_hashes=sources,
        replay_rows=replay,
        role_receipt=role,
        lifecycle_receipt=lifecycle,
        frozen_sidecars=frozen,
        validation_receipts=affected,
        duration_s=(ended_ns - started_ns) / 1_000_000_000,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=spans,
    )
    errors = validate_artifact(candidate, root=resolved, require_terminal=False)
    if errors:
        raise RuntimeError(f"candidate_invalid:{errors}")
    candidate_path = raw_root / "exact_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_plan = terminal_commands(resolved, candidate_path)
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        planned_units=len(terminal_plan),
    )
    phase_started = time.monotonic()
    terminal = _normalize_receipts(
        validation_scope.run_commands(
            resolved,
            terminal_plan,
            log_dir=raw_root / "validation" / "terminal",
            heartbeat_s=60.0,
        ),
        resolved,
    )
    spans.append(
        _span("terminal_validation", started, phase_started, len(terminal_plan), len(terminal))
    )
    progress(started, "terminal_validation", "after_subprocesses", completed_units=len(terminal))

    ended_ns = time.monotonic_ns()
    final = build_artifact(
        resolved,
        checks=checks,
        source_hashes=sources,
        replay_rows=replay,
        role_receipt=role,
        lifecycle_receipt=lifecycle,
        frozen_sidecars=frozen,
        validation_receipts=[*affected, *terminal],
        duration_s=(ended_ns - started_ns) / 1_000_000_000,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=spans,
    )
    atomic_json(candidate_path, final)
    errors = validate_artifact(final, root=resolved, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")

    progress(started, "exact_terminal_replay", "before_subprocesses", planned_units=4)
    exact = _normalize_receipts(
        validation_scope.run_commands(
            resolved,
            terminal_plan,
            log_dir=raw_root / "validation" / "terminal-exact",
            heartbeat_s=60.0,
        ),
        resolved,
    )
    if not _terminal_receipts_equal(terminal, exact):
        raise RuntimeError("exact_terminal_reader_receipts_changed")
    atomic_json(raw_root / "exact_terminal_reader_outcomes.json", {"outcomes": exact})
    progress(started, "exact_terminal_replay", "after_subprocesses", completed_units=len(exact))

    progress(started, "publish", "before_atomic", path=RESULT_PATH.as_posix())
    atomic_json(destination, final)
    progress(started, "publish", "after_atomic", path=RESULT_PATH.as_posix())
    return final


def _date_argument(value: str) -> str:
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def _output_argument(value: str) -> Path:
    path = Path(value)
    if path != RESULT_PATH:
        raise argparse.ArgumentTypeError(f"output must be {RESULT_PATH.as_posix()}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the producer and two bounded read-only terminal modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=_output_argument, default=RESULT_PATH)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--date", type=_date_argument)
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run Exp7616 or inspect one exact candidate without model work."""

    print("[exp7616] phase=startup event=flushed", flush=True)
    args = parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay is not None:
        result = cold_replay(args.cold_replay, root=root)
        print(json.dumps(result, sort_keys=True), flush=True)
        return int(result["passed"] is not True)
    if args.independent_reduce is not None:
        result = independent_reduce(_load_json(args.independent_reduce), root=root)
        print(json.dumps(result, sort_keys=True), flush=True)
        return int(result["passed"] is not True)
    artifact = run_experiment(root, args.date, output_path=root / args.output)
    print(
        json.dumps(
            {
                "result": args.output.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
                "evidence_schema_ready_score": artifact["evidence_schema_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
