"""Run one authenticated Qwen3.8 public-mention calibration canary.

The model gets separate source and claim calls for three fixed interfaces.
Private authority rows are opened only after all public requests are frozen.

Spec refs: REQ-VERIFY-7237 and SCENARIO-VERIFY-7237-*.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

from carnot import experiment_7208_v635_span_fixture as offset_fixture
from carnot import experiment_7209_v635_span_canary as live_runtime
from carnot import experiment_7236_v637_mention_fixture as mention_fixture
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
Tokenize = Callable[[bytes], Sequence[int]]

RUN_DATE = "20260912"
MILESTONE = "2026.09.637"
EXPERIMENT_ID = "exp7237-mention-canary"
TASK_ID = "experiment_7237_v637_mention_canary"
RANDOM_SEED = 7_237_001
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": QWEN_MODEL_ID, "quantization": QUANTIZATION}]

ARMS = ("original_offset", "explicit_schema_offset_control", "mention_pointer")
TOKEN_BUDGETS = {"source": 384, "claim": 128}
CONTEXT_TOKEN_BUDGET = 8192
MODEL_LOAD_CAP_S = 240.0
REQUEST_CAP_S = 90.0
INFERENCE_DEADLINE_S = 1800.0

RESULT_PATH = Path("results/experiment_7237_v637_mention_canary.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7237")
RAW_DIR = Path("results/raw/experiment_7237")
UPSTREAM_PATH = Path("results/experiment_7236_v637_mention_fixture.json")
PUBLIC_PATH = Path("results/raw/experiment_7236/public_manifest.json")
AUTHORITY_PATH = Path("results/raw/experiment_7236/authority_manifest.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7237_v637_mention_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7237_v637_mention_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7237_v637_mention_canary.py")

PINNED_UPSTREAM_SHA256 = "sha256:b5604654dcefe5755479137c930f0ff5266b4c064aa6701f5c72fd4dec8620e3"
PINNED_PUBLIC_SHA256 = "sha256:b6a11618b77431d6e6d80b449662ffe1845e6c0193097eae8bc95d388171019d"
PINNED_AUTHORITY_SHA256 = "sha256:7b19fdec3833f09805b210f51231365581269ec084dc09820e1b8b7f67c74edc"

UPSTREAM_EXPECTED_FIELDS: JsonDict = {
    "status": "complete",
    "run_date": RUN_DATE,
    "verdict_class": "circular_positive",
    "honest_verdict": (
        "complete_circular_positive_mention_fixture_ready_oracle_control_no_generalization_claim"
    ),
    "mention_fixture_ready_score": 1,
    "public_manifest_path": PUBLIC_PATH.as_posix(),
    "authority_manifest_path": AUTHORITY_PATH.as_posix(),
}

AUTHORITY_ONLY_FIELDS = {
    "condition_key",
    "entity_vocabulary",
    "exact_label",
    "generation_seed",
    "gold_claim_completion",
    "gold_source_completion",
    "random_seed",
    "relation_phrase",
}

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "sample_size_budget": "Predeclared independent units, attempted/completed/censored units, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked_* verdict names check, upstream, artifact_field, expected and observed value.",
    "verifier_is_oracle": "True when the verification authority also defines correctness; separate code is insufficient independence.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. External incompleteness is blocked.",
    "honest_verdict": "Completed findings start complete_ or complete:. External absence starts blocked_. Failed acceptance forbids positive.",
    "acceptance_gate_results": "Preserve each frozen criterion, actual value and pass/fail independently of task completion.",
    "inference_mode": "live_gpu only after actual CUDA work.",
    "runner_receipt": "Task-owned invocation identity and actual transport counts, not copied upstream receipts.",
    "raw_request_manifest": "Exact model, prompt, parameters and response hashes for every call.",
    "phase_spans": "Measured model load, generation, scoring and validation times.",
    "mention_canary_ready_score": "Authenticated pointer treatment passes the predeclared 7/8 parse and 6/8 semantic checks.",
    "transport_completed_calls": "All successful returned calls, independent of semantics.",
    "usable_calls": "Complete, parse-valid, non-truncated calls with required semantic content.",
    "per_unit_semantics": "All eight units in each of three arms, with source fidelity, claim fidelity, decision and abstention.",
    "frozen_capture_settings": "Immutable pointer and offset settings for the held-out run; no post-canary tuning.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

ORIGINAL_PROMPTS = deepcopy(live_runtime.PROMPT_TEMPLATES)
EXPLICIT_PROMPTS = {
    "source": (
        "Extract all stated relations from the source. Return one JSON object with keys "
        "outcome and relations. Each relation must use sentence_index, subject_start, "
        "subject_end, predicate, object_start, object_end, and polarity. Offsets are "
        "half-open UTF-8 byte offsets. Return unknown with an empty list when no exact "
        "relation is available.\nSOURCE:\n{input_text}"
    ),
    "claim": (
        "Extract the one stated relation from the claim. Return one JSON object with keys "
        "outcome and relations. The relation must use sentence_index, subject_start, "
        "subject_end, predicate, object_start, object_end, and polarity. Offsets are "
        "half-open UTF-8 byte offsets. Return unknown with an empty list when no exact "
        "relation is available.\nCLAIM:\n{input_text}"
    ),
}
POINTER_PROMPTS = {
    "source": (
        "Extract all stated relations from the source. Select subject_pointer and "
        "object_pointer only from the public mention table. Predict predicate and polarity. "
        "Return one JSON object with outcome and relations, or unknown with an empty list.\n"
        "SOURCE:\n{input_text}\nPUBLIC_MENTION_TABLE:\n{mention_table}"
    ),
    "claim": (
        "Extract the one stated relation from the claim. Select subject_pointer and "
        "object_pointer only from the public mention table. Predict predicate and polarity. "
        "Return one JSON object with outcome and relations, or unknown with an empty list.\n"
        "CLAIM:\n{input_text}\nPUBLIC_MENTION_TABLE:\n{mention_table}"
    ),
}

canonical_json = mention_fixture.canonical_json
sha256_bytes = mention_fixture.sha256_bytes
sha256_file = mention_fixture.sha256_file
unwrap_principle = mention_fixture.unwrap_principle_value
is_quarantined = live_runtime.is_quarantined
load_yaml = live_runtime.load_yaml
gate_row = live_runtime.gate_row
gate_summary = live_runtime.gate_summary


def sha256_json(value: Any) -> str:
    """Hash one structured value with the shared canonical JSON spelling."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding actual process clock observations."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "duration_s",
            "timestamps",
            "phase_spans",
            "reproducibility_checksum",
        }
    }
    return sha256_json(stable)


def load_calibration_manifests(
    public_path: Path, authority_path: Path
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Load the eight calibration bases while retaining public and private separation."""

    public = json.loads(public_path.read_text(encoding="utf-8"))
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if not isinstance(public, Mapping) or public.get("schema") != (
        "carnot.exp7236.public_mentions.v1"
    ):
        raise ValueError("public_manifest_schema")
    if not isinstance(authority, Mapping) or authority.get("schema") != (
        "carnot.exp7236.private_authority.v1"
    ):
        raise ValueError("authority_manifest_schema")
    public_rows = [
        deepcopy(row) for row in public.get("rows", []) if row.get("split") == "calibration"
    ]
    authority_rows = [
        deepcopy(row) for row in authority.get("rows", []) if row.get("split") == "calibration"
    ]
    if len(public_rows) != 8 or len(authority_rows) != 8:
        raise ValueError("calibration_denominator")
    public_ids = [str(row.get("unit_id")) for row in public_rows]
    authority_ids = [str(row.get("unit_id")) for row in authority_rows]
    if len(set(public_ids)) != 8 or public_ids != authority_ids:
        raise ValueError("calibration_identity")
    return public_rows, authority_rows


def unit_document(row: Mapping[str, Any], call_type: str) -> JsonDict:
    """Choose the fixed original public document without consulting authority labels."""

    variants = [item for item in row.get("variants", []) if item.get("variant") == "original"]
    if len(variants) != 1 or call_type not in TOKEN_BUDGETS:
        raise ValueError("public_original_variant")
    document = variants[0].get(call_type)
    if not isinstance(document, Mapping):
        raise ValueError("public_document")
    return deepcopy(dict(document))


def _enum_rule(values: Sequence[str]) -> str:
    """Render finite JSON string terminals for the native GBNF parser."""

    return " | ".join(json.dumps(json.dumps(value)) for value in values)


def _pointer_grammar(document: Mapping[str, Any], call_type: str) -> str:
    """Limit pointer fields to IDs present in one public mention table."""

    mentions = document.get("mentions")
    if call_type not in TOKEN_BUDGETS or not isinstance(mentions, list):
        raise ValueError("pointer_grammar_input")
    pointers = [str(row["mention_id"]) for row in mentions]
    if not pointers:
        raise ValueError("pointer_grammar_mentions")
    relation_list = (
        'relation | relation "," relation | relation "," relation "," relation | '
        'relation "," relation "," relation "," relation'
        if call_type == "source"
        else "relation"
    )
    unknown = canonical_json({"outcome": "unknown", "relations": []})
    return (
        "\n".join(
            [
                "root ::= unknown | known",
                f"unknown ::= {json.dumps(unknown)}",
                'known ::= "{\\"outcome\\":\\"known\\",\\"relations\\":[" relation-list "]}"',
                f"relation-list ::= {relation_list}",
                'relation ::= "{\\"object_pointer\\":" pointer '
                '",\\"polarity\\":" polarity '
                '",\\"predicate\\":" predicate '
                '",\\"subject_pointer\\":" pointer "}"',
                f"pointer ::= {_enum_rule(pointers)}",
                f"predicate ::= {_enum_rule(offset_fixture.PREDICATES)}",
                f"polarity ::= {_enum_rule(('positive', 'negative'))}",
            ]
        )
        + "\n"
    )


def compile_grammar(arm: str, document: Mapping[str, Any], call_type: str) -> JsonDict:
    """Build one public-only grammar for the selected representation."""

    text = str(document.get("text", "")).encode("utf-8")
    if arm == "mention_pointer":
        grammar = _pointer_grammar(document, call_type)
        return {
            "grammar": grammar,
            "grammar_sha256": sha256_bytes(grammar.encode("utf-8")),
            "reference_grammar_sha256": sha256_bytes(grammar.encode("utf-8")),
        }
    if arm not in ARMS:
        raise ValueError("representation_arm")
    grammar = offset_fixture.compile_grammar(text, call_type, "grammar_only")
    reference = offset_fixture.compile_grammar(text, call_type, "reference")
    return {
        "grammar": grammar["grammar"],
        "grammar_sha256": grammar["grammar_sha256"],
        "reference_grammar_sha256": reference["grammar_sha256"],
    }


def _prompt(arm: str, document: Mapping[str, Any], call_type: str) -> str:
    """Render the fixed interface from public text and optional public mentions."""

    text = str(document["text"])
    if arm == "original_offset":
        return ORIGINAL_PROMPTS[call_type].format(input_text=text)
    if arm == "explicit_schema_offset_control":
        return EXPLICIT_PROMPTS[call_type].format(input_text=text)
    if arm == "mention_pointer":
        table = canonical_json(document["mentions"])
        return POINTER_PROMPTS[call_type].format(input_text=text, mention_table=table)
    raise ValueError("representation_arm")


def build_schedule(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze 48 public requests before any private correctness field is scored."""

    if len(public_rows) != 8 or len(authority_rows) != 8:
        raise ValueError("calibration_denominator")
    if any(row.get("split") != "calibration" for row in public_rows) or any(
        row.get("split") != "calibration" for row in authority_rows
    ):
        raise ValueError("calibration_split")
    public_ids = [str(row.get("unit_id")) for row in public_rows]
    authority_ids = [str(row.get("unit_id")) for row in authority_rows]
    if len(set(public_ids)) != 8 or public_ids != authority_ids:
        raise ValueError("calibration_identity")

    schedule: list[JsonDict] = []
    for public in public_rows:
        unit_id = str(public["unit_id"])
        for arm in ARMS:
            for call_type in ("source", "claim"):
                document = unit_document(public, call_type)
                grammar = compile_grammar(arm, document, call_type)
                prompt = _prompt(arm, document, call_type)
                order = len(schedule)
                seed = RANDOM_SEED + order
                settings = {
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": seed,
                    "cache_prompt": False,
                }
                call_id = sha256_json(
                    {
                        "task": TASK_ID,
                        "unit_id": unit_id,
                        "arm": arm,
                        "call_type": call_type,
                        "seed": seed,
                    }
                )
                schedule.append(
                    {
                        "call_order": order,
                        "call_id": call_id,
                        "unit_id": unit_id,
                        "arm": arm,
                        "call_type": call_type,
                        "seed": seed,
                        "document": document,
                        "model_input": {
                            "text": document["text"],
                            "mentions": document["mentions"] if arm == "mention_pointer" else None,
                        },
                        "input_text": document["text"],
                        "input_sha256": sha256_bytes(str(document["text"]).encode("utf-8")),
                        "prompt": prompt,
                        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                        **grammar,
                        "output_token_budget": TOKEN_BUDGETS[call_type],
                        "context_token_budget": CONTEXT_TOKEN_BUDGET,
                        "request_timeout_s": REQUEST_CAP_S,
                        "decoding_parameters": settings,
                        "cold_request": True,
                    }
                )
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Rebuild the public schedule and name every changed property."""

    try:
        expected = build_schedule(public_rows, authority_rows)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"schedule_rebuild:{type(exc).__name__}:{exc}"]
    errors: list[str] = []
    if len(schedule) != len(expected):
        errors.append("schedule_count")
    for index, (observed, wanted) in enumerate(zip(schedule, expected, strict=False)):
        for field, expected_value in wanted.items():
            if observed.get(field) != expected_value:
                errors.append(f"call_{index}:{field}")
        if set(observed) - set(wanted):
            errors.append(f"call_{index}:extra_fields")
    return errors


def selection_receipt(
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Seal the public schedule before private labels are used for scoring."""

    return {
        "selection_frozen_before_inference": True,
        "selected_unit_ids": [str(row["unit_id"]) for row in public_rows],
        "selected_unit_count": len(public_rows),
        "schedule_sha256": sha256_json(list(schedule)),
        "public_rows_sha256": sha256_json(list(public_rows)),
        "authority_rows_sha256": sha256_json(list(authority_rows)),
        "authority_fields_in_model_schedule": sum(
            bool(set(row) & AUTHORITY_ONLY_FIELDS) for row in schedule
        ),
        "held_out_units_opened": 0,
        "retry_allowed": False,
    }


def upstream_gate_rows(
    upstream: Mapping[str, Any],
    upstream_bytes: bytes,
    public_bytes: bytes,
    authority_bytes: bytes,
    _manifest_bytes: bytes,
    exclusion_manifest: Any,
) -> list[JsonDict]:
    """Authenticate exact Exp7236 evidence and reject any quarantine override."""

    hashes = {
        "artifact": sha256_bytes(upstream_bytes),
        "public_manifest": sha256_bytes(public_bytes),
        "authority_manifest": sha256_bytes(authority_bytes),
    }
    expected_hashes = {
        "artifact": PINNED_UPSTREAM_SHA256,
        "public_manifest": PINNED_PUBLIC_SHA256,
        "authority_manifest": PINNED_AUTHORITY_SHA256,
    }
    rows = [
        gate_row(
            "exact_upstream_bytes",
            expected_hashes,
            hashes,
            hashes == expected_hashes,
            upstream="experiment_7236",
            field="artifact_and_manifest_bytes",
        )
    ]
    quarantined = is_quarantined(upstream)
    rows.append(
        gate_row(
            "structured_quarantine",
            False,
            quarantined,
            not quarantined,
            upstream="experiment_7236",
            field="flagged_adversarial|quarantined|fabricated",
        )
    )
    excluded = live_runtime._manifest_hits(
        exclusion_manifest,
        {"Exp7236", EXPERIMENT_ID, UPSTREAM_PATH.as_posix()},
    )
    rows.append(
        gate_row(
            "exclusion_manifest",
            False,
            excluded,
            not excluded,
            upstream="experiment_7236",
            field="experiment_ids",
        )
    )
    observed_fields = {
        field: unwrap_principle(upstream.get(field)) for field in UPSTREAM_EXPECTED_FIELDS
    }
    rows.append(
        gate_row(
            "producer_gate_fields",
            UPSTREAM_EXPECTED_FIELDS,
            observed_fields,
            observed_fields == UPSTREAM_EXPECTED_FIELDS,
            upstream="experiment_7236",
            field="terminal_fields",
        )
    )
    checksum_ok = upstream.get("reproducibility_checksum") == mention_fixture.artifact_checksum(
        upstream
    )
    rows.append(
        gate_row(
            "upstream_authentication",
            True,
            checksum_ok,
            checksum_ok,
            upstream="experiment_7236",
            field="reproducibility_checksum",
        )
    )
    try:
        public = json.loads(public_bytes)
        authority = json.loads(authority_bytes)
    except json.JSONDecodeError:
        public, authority = {}, {}
    artifact_hashes = upstream.get("source_artifact_hashes")
    observed_manifests = {
        "public_schema": public.get("schema"),
        "authority_schema": authority.get("schema"),
        "public_units": len(public.get("rows", [])),
        "authority_units": len(authority.get("rows", [])),
        "artifact_public_hash": artifact_hashes.get("public_manifest")
        if isinstance(artifact_hashes, Mapping)
        else None,
        "artifact_authority_hash": artifact_hashes.get("authority_manifest")
        if isinstance(artifact_hashes, Mapping)
        else None,
    }
    expected_manifests = {
        "public_schema": "carnot.exp7236.public_mentions.v1",
        "authority_schema": "carnot.exp7236.private_authority.v1",
        "public_units": 72,
        "authority_units": 72,
        "artifact_public_hash": PINNED_PUBLIC_SHA256,
        "artifact_authority_hash": PINNED_AUTHORITY_SHA256,
    }
    rows.append(
        gate_row(
            "manifest_authentication",
            expected_manifests,
            observed_manifests,
            observed_manifests == expected_manifests,
            upstream="experiment_7236",
            field="manifest_schema_counts_and_hashes",
        )
    )
    return rows


def render_gold_completion(
    document: Mapping[str, Any],
    pointer_completion: Mapping[str, Any],
    arm: str,
    call_type: str,
) -> JsonDict:
    """Create test authority output in the exact public interface shape."""

    if arm == "mention_pointer":
        return deepcopy(dict(pointer_completion))
    compiled = mention_fixture.compile_pointer_completion(document, pointer_completion, call_type)
    if compiled.get("outcome") != "known":
        return {"outcome": "unknown", "relations": []}
    return {
        "outcome": "known",
        "relations": [
            {field: relation[field] for field in offset_fixture.RELATION_FIELDS}
            for relation in compiled["relations"]
        ],
    }


def _completion_shape_valid(value: Any, arm: str, call_type: str) -> bool:
    """Check JSON structure separately from public-reference resolution."""

    if not isinstance(value, Mapping) or set(value) != {"outcome", "relations"}:
        return False
    outcome, relations = value.get("outcome"), value.get("relations")
    if outcome not in {"known", "unknown"} or not isinstance(relations, list):
        return False
    if outcome == "unknown":
        return not relations
    limit = 4 if call_type == "source" else 1
    if not 1 <= len(relations) <= limit:
        return False
    fields = (
        {"subject_pointer", "predicate", "object_pointer", "polarity"}
        if arm == "mention_pointer"
        else set(offset_fixture.RELATION_FIELDS)
    )
    return all(isinstance(row, Mapping) and set(row) == fields for row in relations)


def _compile_completion(sealed: Mapping[str, Any], parsed: Any) -> JsonDict:
    """Resolve one parsed output through its public representation compiler."""

    document = sealed["document"]
    if sealed["arm"] == "mention_pointer":
        return mention_fixture.compile_pointer_completion(
            document, parsed, str(sealed["call_type"])
        )
    return offset_fixture.compile_completion(
        str(document["text"]).encode("utf-8"),
        parsed,
        str(sealed["call_type"]),
        str(sealed["reference_grammar_sha256"]),
    )


def _decoded_bytes(value: Any) -> bytes | None:
    """Decode one retained byte string without repairing malformed evidence."""

    if not isinstance(value, str):
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error):
        return None


def _request_payload(sealed: Mapping[str, Any]) -> tuple[JsonDict, bytes]:
    """Use the shipped native request builder so sealed and sent bytes match."""

    return live_runtime._request_payload(sealed)


def build_completion_row(
    sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
) -> JsonDict:
    """Retain transport, syntax, representation, and provenance as separate facts."""

    raw = str(response.get("raw_completion") or "")
    finish_reason = response.get("finish_reason")
    truncated = finish_reason in {"length", "max_tokens"}
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parse_error = f"JSONDecodeError:{exc}"
    parse_valid = parse_error is None and _completion_shape_valid(
        parsed, str(sealed["arm"]), str(sealed["call_type"])
    )
    compiled = (
        _compile_completion(sealed, parsed)
        if parse_valid
        else {"outcome": "unknown", "relations": [], "errors": ["not_parse_valid"]}
    )
    explicit_unknown = bool(parse_valid and parsed.get("outcome") == "unknown")
    raw_request = response.get("raw_request")
    request_bytes = _decoded_bytes(response.get("raw_request_bytes_b64"))
    response_bytes = _decoded_bytes(response.get("raw_response_bytes_b64"))
    request_bytes_match = bool(
        isinstance(raw_request, Mapping)
        and request_bytes == canonical_json(raw_request).encode("utf-8")
    )
    raw_response = response.get("raw_response")
    decoded_response: Any = None
    if response_bytes:
        try:
            decoded_response = json.loads(response_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            decoded_response = None
    response_bytes_match = bool(
        isinstance(raw_response, Mapping)
        and isinstance(decoded_response, Mapping)
        and decoded_response == raw_response
    )
    transmitted_seed = raw_request.get("seed") if isinstance(raw_request, Mapping) else None
    seed_join_valid = (
        transmitted_seed
        == sealed.get("seed")
        == (sealed.get("decoding_parameters") or {}).get("seed")
    )
    transport_complete = bool(
        response.get("error") is None
        and response_bytes
        and response_bytes_match
        and finish_reason is not None
    )
    compile_errors = list(compiled.get("errors") or [])
    usable = bool(
        transport_complete
        and parse_valid
        and not truncated
        and not compile_errors
        and compiled.get("outcome") == "known"
        and request_bytes_match
        and seed_join_valid
    )
    failures = []
    if response.get("error"):
        failures.append(f"transport_error:{response['error']}")
    if parse_error:
        failures.append(parse_error)
    if parse_valid and not usable:
        failures.extend(compile_errors or (["explicit_unknown"] if explicit_unknown else []))
    if not request_bytes_match:
        failures.append("request_bytes_mismatch")
    if not response_bytes_match:
        failures.append("response_bytes_mismatch")
    if not seed_join_valid:
        failures.append("seed_join_mismatch")
    if truncated:
        failures.append("truncated")
    return {
        "call_order": sealed["call_order"],
        "call_id": sealed["call_id"],
        "unit_id": sealed["unit_id"],
        "arm": sealed["arm"],
        "call_type": sealed["call_type"],
        "seed": sealed["seed"],
        "transmitted_seed": transmitted_seed,
        "prompt": sealed["prompt"],
        "prompt_sha256": sealed["prompt_sha256"],
        "grammar_sha256": sealed["grammar_sha256"],
        "actual_parameters": deepcopy(dict(raw_request or {})),
        "raw_request_bytes_b64": response.get("raw_request_bytes_b64", ""),
        "request_bytes_sha256": sha256_bytes(request_bytes or b""),
        "request_bytes_match": request_bytes_match,
        "raw_response_bytes_b64": response.get("raw_response_bytes_b64", ""),
        "response_bytes_sha256": sha256_bytes(response_bytes or b""),
        "response_bytes_match": response_bytes_match,
        "raw_completion": raw,
        "raw_completion_sha256": sha256_bytes(raw.encode("utf-8")),
        "parsed_completion": deepcopy(parsed),
        "compiled_completion": compiled,
        "transport_complete": transport_complete,
        "parse_valid": parse_valid,
        "explicit_unknown": explicit_unknown,
        "usable": usable,
        "truncated": truncated,
        "finish_reason": finish_reason,
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "seed_join_valid": seed_join_valid,
        "server_pid": resource.get("server_pid"),
        "server_pid_start_ticks": resource.get("server_pid_start_ticks"),
        "gpu_uuid": resource.get("gpu_uuid"),
        "lease_id": resource.get("lease_id"),
        "cuda_offload_confirmed": resource.get("cuda_offload_confirmed") is True,
        "terminal_state": "complete" if transport_complete else "transport_error",
        "errors": list(dict.fromkeys(failures)),
    }


def replay_completion_rows(
    schedule: Sequence[Mapping[str, Any]], retained_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Reduce exact retained server bytes again without issuing model requests."""

    schedule_by_id = {str(row["call_id"]): row for row in schedule}
    replayed = []
    for retained in retained_rows:
        response_bytes = _decoded_bytes(retained["raw_response_bytes_b64"])
        raw_response = json.loads(response_bytes or b"null")
        replayed.append(
            build_completion_row(
                schedule_by_id[str(retained["call_id"])],
                {
                    "raw_request": deepcopy(retained["actual_parameters"]),
                    "raw_request_bytes_b64": retained["raw_request_bytes_b64"],
                    "raw_response": raw_response,
                    "raw_response_bytes_b64": retained["raw_response_bytes_b64"],
                    "raw_completion": retained["raw_completion"],
                    "prompt_tokens": retained["prompt_tokens"],
                    "completion_tokens": retained["completion_tokens"],
                    "finish_reason": retained["finish_reason"],
                    "latency_s": retained["latency_s"],
                    "error": None,
                },
                {
                    "server_pid": retained["server_pid"],
                    "server_pid_start_ticks": retained["server_pid_start_ticks"],
                    "gpu_uuid": retained["gpu_uuid"],
                    "lease_id": retained["lease_id"],
                    "cuda_offload_confirmed": retained["cuda_offload_confirmed"],
                },
            )
        )
    return replayed


def _semantic_view(relation: Mapping[str, Any]) -> JsonDict:
    """Compare predicted relation meaning without representation-specific coordinates."""

    return {
        "subject_surface": relation.get("subject_surface"),
        "predicate": relation.get("predicate"),
        "object_surface": relation.get("object_surface"),
        "polarity": relation.get("polarity"),
    }


def _expected_compiled(
    document: Mapping[str, Any], pointer: Mapping[str, Any], call_type: str
) -> JsonDict:
    """Resolve private gold pointers through the same public compiler for scoring."""

    return mention_fixture.compile_pointer_completion(document, pointer, call_type)


def score_semantics(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Score 24 unit-arm pairs without dropping failures or abstentions."""

    public_by_id = {str(row["unit_id"]): row for row in public_rows}
    authority_by_id = {str(row["unit_id"]): row for row in authority_rows}
    completion_by_key = {
        (str(row["unit_id"]), str(row["arm"]), str(row["call_type"])): row
        for row in completion_rows
    }
    rows: list[JsonDict] = []
    for unit_id in [str(row["unit_id"]) for row in public_rows]:
        public = public_by_id[unit_id]
        authority = authority_by_id[unit_id]
        authority_variant = authority["variants"][0]
        source_document = unit_document(public, "source")
        claim_document = unit_document(public, "claim")
        expected_source = _expected_compiled(
            source_document, authority_variant["gold_source_completion"], "source"
        )
        expected_claim = _expected_compiled(
            claim_document, authority_variant["gold_claim_completion"], "claim"
        )
        for arm in ARMS:
            source_row = completion_by_key.get((unit_id, arm, "source"))
            claim_row = completion_by_key.get((unit_id, arm, "claim"))
            missing = [
                call_type
                for call_type, row in (("source", source_row), ("claim", claim_row))
                if row is None
            ]
            compiled_source = (
                source_row["compiled_completion"]
                if source_row
                else {"outcome": "unknown", "relations": [], "errors": ["missing_call"]}
            )
            compiled_claim = (
                claim_row["compiled_completion"]
                if claim_row
                else {"outcome": "unknown", "relations": [], "errors": ["missing_call"]}
            )
            source_fidelity = bool(
                source_row
                and source_row.get("usable") is True
                and [_semantic_view(item) for item in compiled_source.get("relations", [])]
                == [_semantic_view(item) for item in expected_source.get("relations", [])]
            )
            claim_fidelity = bool(
                claim_row
                and claim_row.get("usable") is True
                and [_semantic_view(item) for item in compiled_claim.get("relations", [])]
                == [_semantic_view(item) for item in expected_claim.get("relations", [])]
            )
            offset_valid = bool(
                source_row
                and claim_row
                and source_row.get("usable") is True
                and claim_row.get("usable") is True
                and compiled_source.get("outcome") == "known"
                and compiled_claim.get("outcome") == "known"
            )
            mention_resolution = offset_valid if arm == "mention_pointer" else None
            execution = mention_fixture._execute_compiled_pair(
                source_document, compiled_source, compiled_claim
            )
            prediction = execution["decision"]
            expected = authority_variant["exact_label"]
            decision_correct = prediction == expected
            fully_correct = bool(
                offset_valid and source_fidelity and claim_fidelity and decision_correct
            )
            errors = [*missing, *execution.get("errors", [])]
            if not source_fidelity:
                errors.append("source_fidelity")
            if not claim_fidelity:
                errors.append("claim_fidelity")
            if not decision_correct:
                errors.append("decision")
            rows.append(
                {
                    "unit_id": unit_id,
                    "arm": arm,
                    "seed": source_row.get("seed") if source_row else None,
                    "condition": authority["condition_key"],
                    "expected_decision": expected,
                    "predicted_decision": prediction,
                    "offset_valid": offset_valid,
                    "mention_resolution": mention_resolution,
                    "source_fidelity": source_fidelity,
                    "claim_fidelity": claim_fidelity,
                    "decision_correct": decision_correct,
                    "fully_correct": fully_correct,
                    "abstention": execution["abstention"],
                    "false_accept": expected != "supported" and prediction == "supported",
                    "metric": int(fully_correct),
                    "error": None if fully_correct else ";".join(dict.fromkeys(errors)),
                    "source_call_id": source_row.get("call_id") if source_row else None,
                    "claim_call_id": claim_row.get("call_id") if claim_row else None,
                }
            )
    return rows


def readiness_receipt(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    *,
    provenance_errors: Sequence[str],
) -> JsonDict:
    """Apply the pointer-only syntax, fidelity, and negative-control gates."""

    pointer_calls = [row for row in completion_rows if row.get("arm") == "mention_pointer"]
    by_unit: dict[str, list[Mapping[str, Any]]] = {}
    for row in pointer_calls:
        by_unit.setdefault(str(row.get("unit_id")), []).append(row)
    complete_parse_units = sum(
        len(rows) == 2
        and {row.get("call_type") for row in rows} == {"source", "claim"}
        and all(
            row.get("transport_complete") is True
            and row.get("parse_valid") is True
            and row.get("truncated") is False
            for row in rows
        )
        for rows in by_unit.values()
    )
    usable_units = sum(
        len(rows) == 2 and all(row.get("usable") is True for row in rows)
        for rows in by_unit.values()
    )
    pointer_semantics = [row for row in semantic_rows if row.get("arm") == "mention_pointer"]
    correct = sum(row.get("fully_correct") is True for row in pointer_semantics)
    false_accepts = sum(
        row.get("condition") == "reversed" and row.get("false_accept") is True
        for row in pointer_semantics
    )
    schedule_ok = len(schedule) == 48 and len(completion_rows) == 48
    gates = {
        "authenticated_provenance": not provenance_errors,
        "fixed_schedule_complete": schedule_ok,
        "pointer_complete_parse_units_at_least_7": complete_parse_units >= 7,
        "pointer_usable_units_at_least_7": usable_units >= 7,
        "pointer_semantic_correct_units_at_least_6": correct >= 6,
        "negative_control_false_accepts_zero": false_accepts == 0,
    }
    return {
        "pointer_call_denominator": 16,
        "pointer_complete_parse_units": complete_parse_units,
        "pointer_usable_units": usable_units,
        "pointer_semantic_correct_units": correct,
        "negative_control_false_accepts": false_accepts,
        "provenance_errors": list(provenance_errors),
        "criteria": [
            {
                "criterion": name,
                "expected_value": True,
                "actual_value": passed,
                "passed": passed,
            }
            for name, passed in gates.items()
        ],
        "mention_canary_ready_score": int(all(gates.values())),
    }


def frozen_capture_settings() -> JsonDict:
    """Return the immutable settings eligible for the later held-out run."""

    return {
        "model": deepcopy(MODEL_SPECS[0]),
        "arms": list(ARMS),
        "token_budgets": deepcopy(TOKEN_BUDGETS),
        "temperature": 0.0,
        "top_k": 1,
        "seed_base": RANDOM_SEED,
        "retry_malformed": False,
        "model_load_cap_s": MODEL_LOAD_CAP_S,
        "request_cap_s": REQUEST_CAP_S,
        "inference_deadline_s": INFERENCE_DEADLINE_S,
        "pointer_prompt_sha256": sha256_json(POINTER_PROMPTS),
        "original_prompt_sha256": sha256_json(ORIGINAL_PROMPTS),
        "explicit_prompt_sha256": sha256_json(EXPLICIT_PROMPTS),
    }


def base_artifact(run_date: str) -> JsonDict:
    """Create every required field before the first fallible precondition."""

    return {
        "schema": {
            "name": "carnot.experiment_7237_v637_mention_canary",
            "version": 1,
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
        },
        "status": "running",
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": False,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_independent_units": 8,
            "planned_arms": 3,
            "planned_calls": 48,
            "planned_semantic_rows": 24,
            "attempted_calls": 0,
            "transport_completed_calls": 0,
            "censored_calls": 48,
            "completed_semantic_rows": 0,
            "stopping_rule": "run the fixed 48 calls once; do not retry malformed output",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "gate_check_summary": gate_summary(None),
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_exp7237_running",
        "acceptance_gate_results": [],
        "inference_mode": "not_run",
        "runner_receipt": {},
        "raw_request_manifest": {},
        "phase_spans": [],
        "mention_canary_ready_score": 0,
        "transport_completed_calls": 0,
        "usable_calls": 0,
        "per_unit_semantics": [],
        "frozen_capture_settings": frozen_capture_settings(),
        "timestamps": {"started_at_utc": utc_now(), "completed_at_utc": None},
        "schedule": [],
        "raw_rows": [],
        "selection_receipt": {},
        "readiness_receipt": {},
        "model_identity_receipt": {},
        "gpu_receipts": {},
        "validation_command_rows": [],
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Finish an external block without erasing any authentic invocation evidence."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    artifact["status"] = "blocked"
    artifact["verdict_class"] = "blocked"
    artifact["mention_canary_ready_score"] = 0
    artifact["gate_check_summary"] = gate_summary(failure)
    check_name = failure.get("check", "unknown") if failure else "unknown"
    artifact["honest_verdict"] = f"blocked_exp7237_{check_name}"
    if artifact.get("model_invoked") is not True:
        artifact["inference_substrate"] = "blocked_no_run"
        artifact["inference_substrate_class"] = "blocked_no_run"
        artifact["inference_mode"] = "not_run"
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def finalize_measured_artifact(
    artifact: JsonDict,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    provenance_errors: Sequence[str],
) -> JsonDict:
    """Finish a ready or terminal-null canary from the fixed denominators."""

    readiness = readiness_receipt(
        schedule, completion_rows, semantic_rows, provenance_errors=provenance_errors
    )
    transport = sum(row.get("transport_complete") is True for row in completion_rows)
    usable = sum(row.get("usable") is True for row in completion_rows)
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": "live_llm_inference",
            "inference_substrate_class": "model_bounded_generation",
            "inference_mode": "live_gpu",
            "model_invoked": transport > 0,
            "schedule": deepcopy(list(schedule)),
            "raw_rows": deepcopy(list(completion_rows)),
            "rows": deepcopy(list(semantic_rows)),
            "per_unit_semantics": deepcopy(list(semantic_rows)),
            "transport_completed_calls": transport,
            "usable_calls": usable,
            "readiness_receipt": readiness,
            "acceptance_gate_results": deepcopy(readiness["criteria"]),
            "mention_canary_ready_score": readiness["mention_canary_ready_score"],
            "duration_s": duration_s,
            "gate_check_summary": gate_summary(None),
            "raw_request_manifest": {
                "schedule_sha256": sha256_json(list(schedule)),
                "raw_row_count": len(completion_rows),
            },
        }
    )
    artifact["sample_size_budget"].update(
        {
            "attempted_calls": len(completion_rows),
            "transport_completed_calls": transport,
            "censored_calls": 48 - len(completion_rows),
            "completed_semantic_rows": len(semantic_rows),
        }
    )
    ready = readiness["mention_canary_ready_score"] == 1
    artifact["verdict_class"] = "circular_positive" if ready else "null"
    artifact["honest_verdict"] = (
        "complete_circular_positive_mention_canary_ready_scale_gate_only"
        if ready
        else "complete_null_mention_canary_not_ready_no_held_out_capture"
    )
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check terminal fields, denominators, readiness, and provenance."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if value.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("MODEL_SPECS")
    if value.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if value.get("execution_venue") != "host" or not value.get("execution_host"):
        errors.append("execution_identity")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("frozen_capture_settings") != frozen_capture_settings():
        errors.append("frozen_capture_settings")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    if value.get("status") == "blocked":
        summary = value.get("gate_check_summary")
        if value.get("verdict_class") != "blocked" or value.get("mention_canary_ready_score") != 0:
            errors.append("blocked_terminal_state")
        if not isinstance(summary, Mapping) or summary.get("passed") is not False:
            errors.append("gate_check_summary")
        if value.get("model_invoked") is not True and (
            value.get("inference_substrate") != "blocked_no_run"
            or value.get("inference_substrate_class") != "blocked_no_run"
        ):
            errors.append("blocked_substrate")
        return list(dict.fromkeys(errors))
    if value.get("status") != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))
    schedule = value.get("schedule")
    raw_rows = value.get("raw_rows")
    rows = value.get("rows")
    if (
        not isinstance(schedule, list)
        or not isinstance(raw_rows, list)
        or not isinstance(rows, list)
    ):
        errors.append("rows")
        return list(dict.fromkeys(errors))
    provenance_errors = []
    identity = value.get("model_identity_receipt")
    if isinstance(identity, Mapping):
        provenance_errors.extend(identity.get("identity_errors") or [])
    else:
        provenance_errors.append("model_identity")
    expected = readiness_receipt(schedule, raw_rows, rows, provenance_errors=provenance_errors)
    if value.get("readiness_receipt") != expected:
        errors.append("readiness_receipt")
    if value.get("acceptance_gate_results") != expected["criteria"]:
        errors.append("acceptance_gate_results")
    if value.get("mention_canary_ready_score") != expected["mention_canary_ready_score"]:
        errors.append("mention_canary_ready_score")
    transport = sum(row.get("transport_complete") is True for row in raw_rows)
    usable = sum(row.get("usable") is True for row in raw_rows)
    if value.get("transport_completed_calls") != transport:
        errors.append("transport_completed_calls")
    if value.get("usable_calls") != usable:
        errors.append("usable_calls")
    budget = value.get("sample_size_budget")
    if not isinstance(budget, Mapping) or (
        budget.get("planned_calls"),
        budget.get("attempted_calls"),
        budget.get("transport_completed_calls"),
        budget.get("censored_calls"),
        budget.get("completed_semantic_rows"),
    ) != (48, len(raw_rows), transport, 48 - len(raw_rows), len(rows)):
        errors.append("sample_size_budget")
    expected_class = "circular_positive" if expected["mention_canary_ready_score"] else "null"
    if value.get("verdict_class") != expected_class:
        errors.append("verdict_class")
    if (
        provenance_errors
        or value.get("model_invoked") is not True
        or value.get("inference_substrate") != "live_llm_inference"
        or value.get("inference_substrate_class") != "model_bounded_generation"
        or value.get("inference_mode") != "live_gpu"
        or not isinstance(value.get("gpu_receipts"), Mapping)
        or value["gpu_receipts"].get("provenance_ok") is not True
    ):
        errors.append("live_inference_provenance")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration < 10.0:
        errors.append("bounded_generation_duration_floor")
    if value.get("per_unit_semantics") != rows:
        errors.append("per_unit_semantics")
    manifest = value.get("raw_request_manifest")
    if not isinstance(manifest, Mapping) or (
        manifest.get("schedule_sha256") != sha256_json(schedule)
        or manifest.get("raw_row_count") != len(raw_rows)
    ):
        errors.append("raw_request_manifest")
    return list(dict.fromkeys(errors))


def write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:
    """Seal every exact request and response byte receipt without evaluator labels."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "carnot.exp7237.raw_requests.v1",
        "status": "complete" if len(completion_rows) == 48 else "partial",
        "model": deepcopy(dict(model_identity)),
        "raw_row_count": len(completion_rows),
        "schedule_sha256": sha256_json(list(schedule)),
        "authority_path_opened_by_model_worker": False,
        "retry_count": 0,
        "rows": [
            {
                "call_order": row.get("call_order"),
                "call_id": row.get("call_id"),
                "unit_id": row.get("unit_id"),
                "arm": row.get("arm"),
                "call_type": row.get("call_type"),
                "prompt": row.get("prompt"),
                "prompt_sha256": row.get("prompt_sha256"),
                "actual_parameters": deepcopy(row.get("actual_parameters")),
                "raw_request_bytes_b64": row.get("raw_request_bytes_b64"),
                "request_bytes_sha256": row.get("request_bytes_sha256"),
                "raw_response_bytes_b64": row.get("raw_response_bytes_b64"),
                "response_bytes_sha256": row.get("response_bytes_sha256"),
                "completion_tokens": row.get("completion_tokens"),
                "finish_reason": row.get("finish_reason"),
                "latency_s": row.get("latency_s"),
                "transport_complete": row.get("transport_complete"),
                "parse_valid": row.get("parse_valid"),
                "usable": row.get("usable"),
                "row_sha256": sha256_json(row),
            }
            for row in completion_rows
        ],
    }
    atomic_write_json(
        raw_dir / "raw_request_manifest.json",
        manifest,
        allow_override=False,
        sort_keys=True,
    )
    return manifest


def measure_token_budgets(schedule: Sequence[Mapping[str, Any]], tokenizer: Tokenize) -> JsonDict:
    """Measure public prompt sizes and conservative valid outputs with the GGUF tokenizer."""

    receipt: JsonDict = {"measurement_status": "measured_embedded_gguf_tokenizer"}
    all_fit = True
    for call_type in ("source", "claim"):
        calls = [row for row in schedule if row.get("call_type") == call_type]
        prompt_counts = [len(tokenizer(str(row["prompt"]).encode("utf-8"))) for row in calls]
        output_counts = [
            len(tokenizer(canonical_json({"outcome": "unknown", "relations": []}).encode("utf-8")))
            for _row in calls
        ]
        fits = bool(calls) and max(output_counts) <= TOKEN_BUDGETS[call_type]
        receipt[call_type] = {
            "form_count": len(calls),
            "maximum_prompt_tokens": max(prompt_counts),
            "maximum_minimal_output_tokens": max(output_counts),
            "output_budget": TOKEN_BUDGETS[call_type],
            "fits": fits,
        }
        all_fit = all_fit and fits
    receipt["all_forms_fit"] = all_fit
    return receipt


def _identity_errors(
    identity: Mapping[str, Any],
    gpu_receipts: Mapping[str, Any],
    completion_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Name any model, CUDA, process, request-byte, or seed join failure."""

    errors = []
    if identity.get("hf_id") != QWEN_MODEL_ID:
        errors.append("hf_id")
    if identity.get("quantization") != QUANTIZATION:
        errors.append("quantization")
    if not identity.get("revision") or not identity.get("gguf_sha256"):
        errors.append("gguf_revision_or_hash")
    if identity.get("embedded_chat_template_present") is not True:
        errors.append("embedded_chat_template")
    if gpu_receipts.get("provenance_ok") is not True:
        errors.append("actual_cuda_execution")
    server = gpu_receipts.get("server_identity")
    if not isinstance(server, Mapping) or any(
        row.get("server_pid") != server.get("pid")
        or row.get("server_pid_start_ticks") != server.get("start_time_ticks")
        for row in completion_rows
    ):
        errors.append("native_pid_identity")
    if any(
        row.get("seed_join_valid") is not True or row.get("request_bytes_match") is not True
        for row in completion_rows
    ):
        errors.append("request_seed_join")
    return errors


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live path inventory.
    """Hash every code, contract, public input, and private scoring input used here."""

    paths = {
        "agents": Path("AGENTS.md"),
        "claude": Path("CLAUDE.md"),
        "codex": Path("CODEX.md"),
        "research_program": Path("research-program.md"),
        "research_references": Path("research-references.md"),
        "exclusion_manifest": EXCLUSION_PATH,
        "e2e_test_plan": Path("ops/e2e-test-plan.md"),
        "verification_spec": SPEC_PATH,
        "upstream_artifact": UPSTREAM_PATH,
        "public_manifest": PUBLIC_PATH,
        "authority_manifest": AUTHORITY_PATH,
        "v636_canary": Path("python/carnot/experiment_7223_v636_span_canary.py"),
        "v635_runtime": Path("python/carnot/experiment_7209_v635_span_canary.py"),
        "mention_fixture": Path("python/carnot/experiment_7236_v637_mention_fixture.py"),
        "sota_models": Path("python/carnot/inference/sota_models.py"),
        "llama_server_supervisor": Path("python/carnot/inference/llama_server_supervisor.py"),
        "experiment_template": Path("scripts/experiment_template.py"),
        "module": MODULE_PATH,
        "entrypoint": WRAPPER_PATH,
        "focused_tests": TEST_PATH,
    }
    return {
        name: sha256_file(root / path) if (root / path).is_file() else "missing"
        for name, path in paths.items()
    }


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover
    """Flush truthful observed progress for the outer task monitor."""

    print(
        canonical_json({"experiment": 7237, "phase": phase, "event": event, **fields}),
        flush=True,
    )


def _collect_preflight(  # pragma: no cover - live host resource boundary.
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_dir: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], JsonDict]:
    """Reuse the shipped resource checks with exact Exp7236 inputs."""

    checks, public_rows, authority_rows, context = live_runtime._collect_preflight(
        root,
        run_date,
        result_path,
        checkpoint_dir,
        raw_dir,
        contract={
            "run_date": RUN_DATE,
            "upstream_path": UPSTREAM_PATH,
            "public_path": PUBLIC_PATH,
            "authority_path": AUTHORITY_PATH,
            "manifest_path": PUBLIC_PATH,
            "spec_path": SPEC_PATH,
            "module_path": MODULE_PATH,
            "wrapper_path": WRAPPER_PATH,
            "test_path": TEST_PATH,
            "spec_req": "REQ-VERIFY-7237",
            "expected_calls": 48,
            "expected_units": 8,
            "task_id": TASK_ID,
            "upstream_id": "experiment_7236",
            "split_id": "experiment_7236_calibration",
            "upstream_gate_rows": upstream_gate_rows,
            "load_split": load_calibration_manifests,
            "build_schedule": build_schedule,
            "schedule_errors": schedule_errors,
            "request_cap_s": REQUEST_CAP_S,
            "live_window_cap_s": INFERENCE_DEADLINE_S,
            "model_load_cap_s": MODEL_LOAD_CAP_S,
        },
    )
    if context.get("schedule"):
        context["selection_receipt"] = selection_receipt(
            public_rows, authority_rows, context["schedule"]
        )
        context["completion_builder"] = build_completion_row
    return checks, public_rows, authority_rows, context


def _checkpoint(
    path: Path,
    artifact: Mapping[str, Any],
    started: float,
    prior_duration_s: float = 0.0,
) -> None:  # pragma: no cover
    """Write provisional work only below the task checkpoint directory."""

    value = deepcopy(dict(artifact))
    value["duration_s"] = prior_duration_s + time.monotonic() - started
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def _terminal(  # pragma: no cover - exercised by the real producer replay.
    artifact: JsonDict,
    result_path: Path,
    checkpoint_path: Path,
    started: float,
    prior_duration_s: float = 0.0,
) -> JsonDict:
    """Cold-check once and atomically publish one stable terminal result."""

    _progress(8, "validation_start", path=str(result_path))
    validation_started = time.monotonic()
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["duration_s"] = prior_duration_s + time.monotonic() - started
    artifact["phase_spans"].append(
        {
            "phase": 8,
            "name": "cold_artifact_validation",
            "duration_s": time.monotonic() - validation_started,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    _progress(8, "validation_end", errors=errors)
    if errors:
        raise ValueError(f"invalid Exp7237 artifact: {errors}")
    _progress(9, "write_start", path=str(result_path))
    _checkpoint(checkpoint_path, artifact, started, prior_duration_s)
    atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
    _progress(9, "write_end", path=str(result_path))
    return artifact


def replay_terminal_artifact(  # pragma: no cover - real independent reducer E2E.
    root: Path | None = None, *, output_root: Path | None = None
) -> JsonDict:
    """Publish a corrected reduction of one complete sealed live capture."""

    replay_started = time.monotonic()
    repo = root or find_repo_root(start=__file__)
    destination = output_root or repo
    result_path = destination / RESULT_PATH
    checkpoint_path = destination / CHECKPOINT_DIR / "terminal_candidate.json"
    raw_dir = destination / RAW_DIR
    _progress(5, "benchmark_start", operation="independent_exact_byte_reducer")
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    prior_duration_s = float(artifact["duration_s"])
    schedule = list(artifact.get("schedule") or [])
    retained_rows = list(artifact.get("raw_rows") or [])
    public_rows, authority_rows = load_calibration_manifests(
        repo / PUBLIC_PATH, repo / AUTHORITY_PATH
    )
    authentication_errors = schedule_errors(schedule, public_rows, authority_rows)
    if len(retained_rows) != 48:
        authentication_errors.append("retained_row_count")
    for index, (sealed, retained) in enumerate(zip(schedule, retained_rows, strict=True)):
        raw_path = raw_dir / f"call_{index:02d}.json"
        captured = json.loads(raw_path.read_text(encoding="utf-8"))
        if captured.get("schedule") != sealed:
            authentication_errors.append(f"call_{index}:schedule")
        if captured.get("completion") != retained:
            authentication_errors.append(f"call_{index}:completion")
    upstream = json.loads((repo / UPSTREAM_PATH).read_text(encoding="utf-8"))
    upstream_checks = upstream_gate_rows(
        upstream,
        (repo / UPSTREAM_PATH).read_bytes(),
        (repo / PUBLIC_PATH).read_bytes(),
        (repo / AUTHORITY_PATH).read_bytes(),
        (repo / PUBLIC_PATH).read_bytes(),
        load_yaml(repo / EXCLUSION_PATH),
    )
    authentication_errors.extend(
        str(row.get("check")) for row in upstream_checks if row.get("passed") is not True
    )
    if authentication_errors:
        raise ValueError(f"sealed capture authentication failed: {authentication_errors}")

    completion_rows = replay_completion_rows(schedule, retained_rows)
    semantic_rows = score_semantics(schedule, completion_rows, public_rows, authority_rows)
    provenance_errors = _identity_errors(
        artifact["model_identity_receipt"], artifact["gpu_receipts"], completion_rows
    )
    artifact["model_identity_receipt"]["identity_errors"] = provenance_errors
    transport = sum(row.get("transport_complete") is True for row in completion_rows)
    checks = [
        deepcopy(row)
        for row in artifact["preconditions_checked"]
        if row.get("check") != "live_runtime_completion_and_cuda_provenance"
    ]
    checks.append(
        gate_row(
            "live_runtime_completion_and_cuda_provenance",
            {"rows": 48, "transport": 48, "runtime_error": None, "provenance_ok": True},
            {
                "rows": len(completion_rows),
                "transport": transport,
                "runtime_error": None,
                "provenance_ok": artifact["gpu_receipts"].get("provenance_ok"),
            },
            transport == 48 and not provenance_errors,
            upstream="owned_native_llama_server",
            field="live_capture",
        )
    )
    checks.append(
        gate_row(
            "sealed_response_reducer_authentication",
            {"scheduled": 48, "sealed": 48, "retry_count": 0},
            {"scheduled": len(schedule), "sealed": len(retained_rows), "retry_count": 0},
            len(schedule) == len(retained_rows) == 48,
            upstream="results/raw/experiment_7237/call_*.json",
            field="exact_retained_bytes",
        )
    )
    artifact["preconditions_checked"] = checks
    artifact["runner_receipt"].update(
        {
            "completed_transport": transport == 48,
            "transport_completed_calls": transport,
            "semantic_usable_calls": sum(row.get("usable") is True for row in completion_rows),
            "reducer_receipt": {
                "mode": "independent_exact_retained_response_bytes",
                "regenerated_calls": 0,
                "sealed_call_count": len(completion_rows),
                "transport_completed_calls": transport,
            },
        }
    )
    artifact["phase_spans"].append(
        {
            "phase": 5,
            "name": "independent_exact_byte_reducer",
            "duration_s": time.monotonic() - replay_started,
        }
    )
    finalize_measured_artifact(
        artifact,
        schedule,
        completion_rows,
        semantic_rows,
        duration_s=prior_duration_s + time.monotonic() - replay_started,
        provenance_errors=provenance_errors,
    )
    manifest = write_raw_manifest(
        raw_dir, schedule, completion_rows, artifact["model_identity_receipt"]
    )
    artifact["raw_request_manifest"] = manifest
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    artifact["source_artifact_hashes"]["raw_request_manifest"] = sha256_file(
        raw_dir / "raw_request_manifest.json"
    )
    for path in sorted(raw_dir.glob("call_*.json")):
        artifact["source_artifact_hashes"][path.stem] = sha256_file(path)
    _progress(
        5,
        "benchmark_end",
        semantic_rows=len(semantic_rows),
        transport_completed=transport,
        usable_calls=artifact["usable_calls"],
    )
    return _terminal(
        artifact,
        result_path,
        checkpoint_path,
        replay_started,
        prior_duration_s,
    )


def run_experiment(  # pragma: no cover - live GPU execution is replayed as E2E.
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    output_root: Path | None = None,
) -> JsonDict:
    """Run the finite live canary or preserve one exact external block."""

    _progress(0, "start", detail="checkpoint before every prerequisite")
    started = time.monotonic()
    repo = root or find_repo_root(start=__file__)
    destination = output_root or repo
    result_path = destination / RESULT_PATH
    checkpoint_dir = destination / CHECKPOINT_DIR
    raw_dir = destination / RAW_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "terminal_candidate.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("honest_verdict")
            == "blocked_exp7237_live_runtime_completion_and_cuda_provenance"
            and len(existing.get("schedule") or []) == 48
            and len(existing.get("raw_rows") or []) == 48
        ):
            _progress(0, "sealed_capture_replay", regenerated_calls=0)
            return replay_terminal_artifact(repo, output_root=destination)
    artifact = base_artifact(run_date)
    _checkpoint(checkpoint_path, artifact, started)
    _progress(0, "end", checkpoint=str(checkpoint_path))

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    phase_started = time.monotonic()
    _progress(1, "start", operation="preconditions_and_exp7236_authentication")
    checks, public_rows, authority_rows, context = _collect_preflight(
        repo, run_date, result_path, checkpoint_dir, raw_dir
    )
    artifact["phase_spans"].append(
        {
            "phase": 1,
            "name": "preconditions_and_exp7236_authentication",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    failure = next((row for row in checks if row.get("passed") is not True), None)
    _progress(1, "end", passed=failure is None)
    if failure is not None:
        finalize_blocked_artifact(artifact, checks, time.monotonic() - started)
        return _terminal(artifact, result_path, checkpoint_path, started)

    schedule = list(context["schedule"])
    artifact["schedule"] = deepcopy(schedule)
    artifact["selection_receipt"] = deepcopy(context["selection_receipt"])
    _progress(2, "start", operation="freeze_48_public_requests")
    atomic_write_json(
        raw_dir / "schedule.json",
        {"schedule": schedule},
        allow_override=False,
        sort_keys=True,
    )
    _checkpoint(checkpoint_path, artifact, started)
    _progress(2, "end", calls=len(schedule), units=8, arms=3)

    phase_started = time.monotonic()
    _progress(3, "model_load_start", operation="embedded_tokenizer_vocab_only")
    owner, tokenizer, tokenizer_load = live_runtime._load_embedded_tokenizer(
        Path(context["model_path"]), context["tokenizer_loader"]
    )
    _progress(
        3,
        "model_load_end",
        operation="embedded_tokenizer_vocab_only",
        available=tokenizer_load["embedded_tokenizer_available"],
    )
    artifact["phase_spans"].append(
        {
            "phase": 3,
            "name": "embedded_tokenizer_load",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    if tokenizer is None:
        check = gate_row(
            "embedded_tokenizer_load",
            True,
            False,
            False,
            upstream="cached_qwen_gguf",
            field="embedded_tokenizer",
        )
        checks.append(check)
        artifact["preconditions_checked"] = checks
        finalize_blocked_artifact(artifact, checks, time.monotonic() - started)
        return _terminal(artifact, result_path, checkpoint_path, started)
    artifact["token_budget_receipt"] = measure_token_budgets(schedule, tokenizer)
    owner.close()
    artifact["model_identity_receipt"] = deepcopy(context["model_identity"])
    artifact["model_identity_receipt"]["tokenizer_load_receipt"] = tokenizer_load
    _checkpoint(checkpoint_path, artifact, started)

    _progress(4, "benchmark_start", operation="fixed_48_call_mention_canary")
    capture = live_runtime._live_capture(context, checkpoint_dir, raw_dir, artifact["phase_spans"])
    completion_rows = list(capture["rows"])
    _progress(
        4,
        "benchmark_end",
        returned_rows=len(completion_rows),
        transport_completed=sum(row.get("transport_complete") is True for row in completion_rows),
        runtime_error=capture["runtime_error"],
    )

    _progress(5, "benchmark_start", operation="private_semantic_reduction")
    phase_started = time.monotonic()
    semantic_rows = score_semantics(schedule, completion_rows, public_rows, authority_rows)
    artifact["phase_spans"].append(
        {
            "phase": 5,
            "name": "private_semantic_reduction",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    _progress(5, "benchmark_end", semantic_rows=len(semantic_rows))

    artifact["gpu_receipts"] = deepcopy(capture["gpu_receipts"])
    artifact["runner_receipt"] = deepcopy(capture["runner_receipt"])
    transport = sum(row.get("transport_complete") is True for row in completion_rows)
    artifact["model_invoked"] = transport > 0
    if artifact["model_invoked"]:
        artifact["inference_substrate"] = "live_llm_inference"
        artifact["inference_substrate_class"] = "model_bounded_generation"
        artifact["inference_mode"] = "live_gpu"
    provenance_errors = _identity_errors(
        artifact["model_identity_receipt"], artifact["gpu_receipts"], completion_rows
    )
    artifact["model_identity_receipt"]["identity_errors"] = provenance_errors
    artifact["runner_receipt"].update(
        {
            "completed_transport": transport == 48,
            "transport_completed_calls": transport,
            "semantic_usable_calls": sum(row.get("usable") is True for row in completion_rows),
            "server_identity": deepcopy(artifact["gpu_receipts"].get("server_identity", {})),
            "cleanup_ok": artifact["gpu_receipts"].get("cleanup", {}).get("leak_free") is True,
        }
    )
    manifest = write_raw_manifest(
        raw_dir, schedule, completion_rows, artifact["model_identity_receipt"]
    )
    artifact["raw_request_manifest"] = manifest
    artifact["source_artifact_hashes"]["raw_request_manifest"] = sha256_file(
        raw_dir / "raw_request_manifest.json"
    )
    for path in sorted(raw_dir.glob("call_*.json")):
        artifact["source_artifact_hashes"][path.stem] = sha256_file(path)

    runtime_failed = bool(
        capture["runtime_error"]
        or len(completion_rows) != 48
        or transport != 48
        or capture["gpu_receipts"].get("provenance_ok") is not True
    )
    if runtime_failed:
        check = gate_row(
            "live_runtime_completion_and_cuda_provenance",
            {"rows": 48, "transport": 48, "runtime_error": None, "provenance_ok": True},
            {
                "rows": len(completion_rows),
                "transport": transport,
                "runtime_error": capture["runtime_error"],
                "provenance_ok": capture["gpu_receipts"].get("provenance_ok"),
            },
            False,
            upstream="owned_native_llama_server",
            field="live_capture",
        )
        checks.append(check)
        artifact["preconditions_checked"] = checks
        artifact["schedule"] = deepcopy(schedule)
        artifact["raw_rows"] = deepcopy(completion_rows)
        artifact["rows"] = deepcopy(semantic_rows)
        artifact["per_unit_semantics"] = deepcopy(semantic_rows)
        artifact["transport_completed_calls"] = transport
        artifact["usable_calls"] = sum(row.get("usable") is True for row in completion_rows)
        finalize_blocked_artifact(artifact, checks, time.monotonic() - started)
    else:
        finalize_measured_artifact(
            artifact,
            schedule,
            completion_rows,
            semantic_rows,
            duration_s=time.monotonic() - started,
            provenance_errors=provenance_errors,
        )
        artifact["raw_request_manifest"] = manifest
    return _terminal(artifact, result_path, checkpoint_path, started)


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V637 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canary and return success only for a cold-valid terminal result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    args = parser.parse_args(argv)
    artifact = run_experiment(run_date=args.date)
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7237] invalid artifact: {errors}", flush=True)
        return 1
    print(
        f"[exp7237] terminal verdict={artifact['honest_verdict']} "
        f"ready={artifact['mention_canary_ready_score']}",
        flush=True,
    )
    return 0
