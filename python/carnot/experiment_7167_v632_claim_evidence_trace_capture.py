"""Capture sealed claim and evidence traces from the mandated Qwen3.8 GGUF.

This experiment records model output and transport evidence. It does not score
an energy or claim that the model decision is correct. Exp7158 labels stay in
an authority sidecar and never enter a model prompt.

Spec refs: REQ-VERIFY-7167 and SCENARIO-VERIFY-7167-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any
from urllib import request

from carnot import gpu_lease_phase_journal as lease_api
from carnot import experiment_7158_v630_entity_evidence_fixture as fixture_7158
from carnot import experiment_7160_v631_qwen38_lease_diagnosis as preflight_7160
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    resolve_native_llama_server,
)
from carnot.experiment_7150_v628_grounding_preflight import (
    _gpu_snapshot,
    _wait_for_health,
    cuda_offload_receipt,
)
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    canonical_json,
    supervisor_contract,
)
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260909"
TASK_ID = "experiment_7167_v632_claim_evidence_trace_capture"
RANDOM_SEED = 7_167_202_609_09
RESULT_PATH = Path("results/experiment_7167_v632_claim_evidence_trace_capture.json")
FIXTURE_PATH = Path("results/experiment_7158_v630_entity_evidence_fixture.json")
RAW_DIR = Path("results/raw/experiment_7167_v632_claim_evidence_trace_capture")
RAW_MANIFEST_NAME = "raw_manifest.json"
AUTHORITY_SIDECAR_NAME = "authority_labels.json"
CHECKPOINT_NAME = "checkpoint_latest.json"
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7167_v632_claim_evidence_trace_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7167_v632_claim_evidence_trace_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7167_v632_claim_evidence_trace_capture.py")
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QWEN_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
QUANTIZATION = "Q4_K_M"
PINNED_FIXTURE_SHA256 = "sha256:957021034863e359c515c69d47fe44781841a7766e0442ad9a55255cdb1c1635"
INFERENCE_SUBSTRATE = "live_qwen38_claim_evidence_full_generation"
OUTPUT_TOKEN_BUDGET = 384
CONTEXT_TOKEN_BUDGET = 8_192
SOURCE_FAMILIES = tuple(
    family
    for family in fixture_7158.SOURCE_FAMILIES
    if fixture_7158.FAMILY_SPLITS[family] == "evaluation"
)
CONDITIONS = fixture_7158.CONDITIONS
MODEL_SPECS: list[JsonDict] = [
    {
        "hf_id": QWEN_MODEL_ID,
        "filename": QWEN_FILENAME,
        "quantization": QUANTIZATION,
        "role": "headline_full_generation",
    }
]
DECODING_PARAMETERS: JsonDict = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": RANDOM_SEED & 0x7FFFFFFF,
    "cache_prompt": False,
    "max_tokens": OUTPUT_TOKEN_BUDGET,
}
TRACE_SCHEMA: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim_entities",
        "claim_facts",
        "evidence_entities",
        "evidence_facts",
        "cited_source_span",
        "missing_fields",
        "direct_decision",
        "rationale",
    ],
    "properties": {
        "claim_entities": {"type": "array", "items": {"type": "object"}},
        "claim_facts": {"type": "array", "items": {"type": "object"}},
        "evidence_entities": {"type": "array", "items": {"type": "object"}},
        "evidence_facts": {"type": "array", "items": {"type": "object"}},
        "cited_source_span": {
            "type": "object",
            "additionalProperties": False,
            "required": ["start", "end", "text"],
            "properties": {
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 0},
                "text": {"type": "string"},
            },
        },
        "missing_fields": {"type": "array", "items": {"type": "string"}},
        "direct_decision": {
            "type": "string",
            "enum": ["supported", "unsupported", "abstain"],
        },
        "rationale": {"type": "string", "minLength": 1},
    },
}
PROMPT_TEMPLATE = """Analyze one claim against only the supplied evidence.
Keep claim localization, evidence structure, and the direct support decision separate.
Return one JSON object that follows the sealed response schema.
Use zero-based, end-exclusive offsets into EVIDENCE for cited_source_span.
Do not use outside knowledge. If the evidence cannot decide the claim, choose abstain.

FIXTURE_ID: {fixture_id}
PAIR_ID: {pair_id}
CLAIM:
{claim_text}

EVIDENCE:
{evidence_text}
"""

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "MODEL_SPECS",
    "sealed_schedule_rows",
    "generation_receipts",
    "claim_evidence_trace_rows",
    "parser_failure_rows",
    "gpu_telemetry_rows",
    "teardown_receipt",
    "claim_evidence_trace_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Field reasons separate authentic capture from later verifier value.",
    "status": "A terminal state prevents checkpoint-only evidence from passing.",
    "preconditions_checked": "Named fixture, cache, runner, GPU, lease, and storage checks prevent fallback.",
    "run_date": "The date binds volatile model and GPU state.",
    "inference_substrate": "Use live_qwen38_claim_evidence_full_generation.",
    "inference_substrate_class": "Use model_full_generation, or blocked_no_run before invocation, because 48 scored generations are real generative work.",
    "execution_venue": "Host and GPU UUID identify the local runtime.",
    "duration_s": "Measured wall time must reflect the full-generation run.",
    "source_artifact_hashes": "Hashes bind fixture, model, prompt, code, tests, and raw manifest.",
    "rows": "One row per fixture makes capture coverage and failures recheckable.",
    "MODEL_SPECS": "The sole entry proves use of unsloth/Qwen3.8-27B-GGUF Q4_K_M.",
    "sealed_schedule_rows": "Pre-output IDs, pairs, prompts, budgets, and order prove preregistration.",
    "generation_receipts": "Tokens, latency, raw hashes, PID, lease, and CUDA markers prove authentic generation.",
    "claim_evidence_trace_rows": "Per-unit structures, spans, parse states, direct decisions, and exact labels support CPU replay.",
    "parser_failure_rows": "Failed structured outputs remain visible in the denominator.",
    "gpu_telemetry_rows": "Before, during, and after snapshots attribute compute to the task-owned process.",
    "teardown_receipt": "PID, port, lease, and VRAM release prevent another orphan.",
    "claim_evidence_trace_ready_score": "One means all 48 replayable traces and receipts exist; it makes no verifier-value claim.",
    "random_seed": "A fixed seed controls row selection, decoding, and order.",
    "reproducibility_checksum": "The checksum detects fixture, model, prompt, raw-output, or row drift.",
    "gate_check_summary": "A blocked result names the exact failed precondition and observed resource state.",
    "verifier_is_oracle": "False records that this task captures candidates and does not authorize correctness.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial.",
    "honest_verdict": "Free text distinguishes complete transport from block or incomplete local work.",
}

_AUTHORITY_KEYS = frozenset(
    {
        "support_label",
        "expected_answer",
        "exact_label_rule",
        "exact_label",
        "authority_exact_label",
        "condition",
        "perturbation",
    }
)
_AUTHORITY_PROMPT_MARKERS = (
    "authority exact label",
    "authority_exact_label",
    "support_label",
    "expected_answer",
    "exact_label_rule",
    "controlled_fixture_condition",
    "known answer:",
)


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text so whitespace drift remains visible."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a canonical JSON projection for stable cross-process receipts."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a normal source file. Model weights use their cache object name."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Preserve both sides of one gate so a block can be reproduced."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the checksum that contains the hash."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def model_spec_errors(specs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject any model entry that differs from the single approved GGUF."""

    return [] if list(specs) == MODEL_SPECS else ["model_spec_mismatch"]


def _seeded_rank(*parts: str) -> str:
    payload = ":".join((str(RANDOM_SEED), *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_prompt(row: Mapping[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        fixture_id=row["fixture_id"],
        pair_id=row["pair_id"],
        claim_text=row["claim_text"],
        evidence_text=row["evidence_text"],
    )


def build_sealed_schedule(fixture: Mapping[str, Any]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Select 12 fixed pair groups without using labels in selection ranks.

    Each selected pair contributes four rows. A rotating condition window makes
    all nine controlled conditions differ in count by at most one.
    """

    rows = [dict(row) for row in fixture.get("rows", []) if row.get("split") == "evaluation"]
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("pair_id"))].append(row)
    family_pairs: dict[str, list[str]] = defaultdict(list)
    for pair_id, pair_rows in grouped.items():
        families = {str(row.get("source_family")) for row in pair_rows}
        conditions = {str(row.get("condition")) for row in pair_rows}
        if len(families) == 1 and set(CONDITIONS) <= conditions:
            family_pairs[next(iter(families))].append(pair_id)
    selected_pairs: list[str] = []
    for family in SOURCE_FAMILIES:
        ranked = sorted(family_pairs.get(family, []), key=lambda pair: _seeded_rank(family, pair))
        if len(ranked) < 6:
            raise ValueError(f"insufficient_complete_pairs:{family}")
        selected_pairs.extend(ranked[:6])
    selected_pairs.sort(key=lambda pair: _seeded_rank("pair_order", pair))

    schedule: list[JsonDict] = []
    authority: list[JsonDict] = []
    for group_index, pair_id in enumerate(selected_pairs):
        by_condition = {str(row["condition"]): row for row in grouped[pair_id]}
        chosen_conditions = [CONDITIONS[(4 * group_index + offset) % len(CONDITIONS)] for offset in range(4)]
        for condition in chosen_conditions:
            row = by_condition[condition]
            model_input = {
                "fixture_id": str(row["fixture_id"]),
                "pair_id": pair_id,
                "claim_text": str(row["claim_text"]),
                "evidence_text": str(row["evidence_text"]),
            }
            prompt = _render_prompt(row)
            schedule.append(
                {
                    "row_order": len(schedule),
                    "group_order": group_index,
                    "fixture_id": str(row["fixture_id"]),
                    "pair_id": pair_id,
                    "source_family": str(row["source_family"]),
                    "claim_text_sha256": str(row["claim_text_sha256"]),
                    "evidence_text_sha256": str(row["evidence_text_sha256"]),
                    "source_text_sha256": str(row["source_text_sha256"]),
                    "model_input": model_input,
                    "input_sha256": sha256_json(model_input),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
                    "response_schema": deepcopy(TRACE_SCHEMA),
                    "response_schema_sha256": sha256_json(TRACE_SCHEMA),
                    "decoding_parameters": deepcopy(DECODING_PARAMETERS),
                    "decoding_parameters_sha256": sha256_json(DECODING_PARAMETERS),
                    "output_token_budget": OUTPUT_TOKEN_BUDGET,
                }
            )
            authority.append(
                {
                    "row_order": len(authority),
                    "fixture_id": str(row["fixture_id"]),
                    "pair_id": pair_id,
                    "condition": condition,
                    "exact_label": str(row["support_label"]),
                    "exact_label_rule": str(row["exact_label_rule"]),
                    "expected_answer": str(row["expected_answer"]),
                }
            )
    return schedule, authority


def schedule_identity(schedule: Sequence[Mapping[str, Any]]) -> str:
    """Bind row order, inputs, prompt, schema, parameters, and token budgets."""

    return sha256_json(list(schedule))


def label_isolation_errors(schedule: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check model-visible fields for explicit fixture-authority leakage."""

    errors: list[str] = []
    for index, row in enumerate(schedule):
        model_input = row.get("model_input")
        if not isinstance(model_input, Mapping):
            errors.append(f"row_{index}:model_input_invalid")
            continue
        leaked_keys = sorted(_AUTHORITY_KEYS.intersection(model_input))
        if leaked_keys:
            errors.append(f"row_{index}:authority_keys_in_model_input:{','.join(leaked_keys)}")
        prompt = str(row.get("prompt", "")).lower()
        if any(marker in prompt for marker in _AUTHORITY_PROMPT_MARKERS):
            errors.append(f"row_{index}:authority_marker_in_prompt")
    return errors


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any] | None = None
) -> list[str]:
    """Rebuild the sealed schedule shape and optional fixture balance."""

    errors: list[str] = []
    if len(schedule) != 48:
        errors.append("schedule_row_count_mismatch")
    if len({row.get("fixture_id") for row in schedule}) != len(schedule):
        errors.append("schedule_fixture_ids_not_unique")
    pair_counts = Counter(str(row.get("pair_id")) for row in schedule)
    if len(pair_counts) != 12 or set(pair_counts.values()) != {4}:
        errors.append("schedule_pair_groups_mismatch")
    if [row.get("row_order") for row in schedule] != list(range(len(schedule))):
        errors.append("schedule_order_mismatch")
    family_counts = Counter(str(row.get("source_family")) for row in schedule)
    if family_counts != Counter({family: 24 for family in SOURCE_FAMILIES}):
        errors.append("schedule_source_balance_mismatch")
    for index, row in enumerate(schedule):
        model_input = row.get("model_input")
        if not isinstance(model_input, Mapping) or row.get("input_sha256") != sha256_json(model_input):
            errors.append(f"row_{index}:input_hash_mismatch")
        if row.get("prompt_sha256") != sha256_text(str(row.get("prompt", ""))):
            errors.append(f"row_{index}:prompt_hash_mismatch")
        if row.get("prompt_template_sha256") != sha256_text(PROMPT_TEMPLATE):
            errors.append(f"row_{index}:prompt_template_hash_mismatch")
        if row.get("response_schema") != TRACE_SCHEMA:
            errors.append(f"row_{index}:response_schema_mismatch")
        if row.get("response_schema_sha256") != sha256_json(TRACE_SCHEMA):
            errors.append(f"row_{index}:response_schema_hash_mismatch")
        if row.get("decoding_parameters") != DECODING_PARAMETERS:
            errors.append(f"row_{index}:decoding_parameters_mismatch")
        if row.get("decoding_parameters_sha256") != sha256_json(DECODING_PARAMETERS):
            errors.append(f"row_{index}:decoding_parameters_hash_mismatch")
        if row.get("output_token_budget") != OUTPUT_TOKEN_BUDGET:
            errors.append(f"row_{index}:token_budget_mismatch")
    errors.extend(label_isolation_errors(schedule))
    if fixture is not None:
        fixture_rows = {str(row["fixture_id"]): row for row in fixture.get("rows", [])}
        condition_counts = Counter(
            str(fixture_rows[str(row.get("fixture_id"))]["condition"])
            for row in schedule
            if str(row.get("fixture_id")) in fixture_rows
        )
        counts = [condition_counts[condition] for condition in CONDITIONS]
        if not counts or max(counts) - min(counts) > 1:
            errors.append("schedule_condition_balance_mismatch")
    return list(dict.fromkeys(errors))


def parse_structured_output(raw_output: str, model_input: Mapping[str, Any]) -> JsonDict:
    """Parse exact JSON and fail without extracting or repairing partial data."""

    try:
        value = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return {"parser_state": "failed", "parser_error": "invalid_json", "structured_fields": None}
    if not isinstance(value, dict):
        return {"parser_state": "failed", "parser_error": "root_not_object", "structured_fields": None}
    required = list(TRACE_SCHEMA["required"])
    for field in required:
        if field not in value:
            return {
                "parser_state": "failed",
                "parser_error": f"missing_field:{field}",
                "structured_fields": None,
            }
    if set(value) != set(required):
        return {"parser_state": "failed", "parser_error": "field_set_mismatch", "structured_fields": None}
    list_fields = (
        "claim_entities",
        "claim_facts",
        "evidence_entities",
        "evidence_facts",
        "missing_fields",
    )
    if any(not isinstance(value[field], list) for field in list_fields):
        return {"parser_state": "failed", "parser_error": "list_field_invalid", "structured_fields": None}
    if value["direct_decision"] not in {"supported", "unsupported", "abstain"}:
        return {"parser_state": "failed", "parser_error": "direct_decision_invalid", "structured_fields": None}
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        return {"parser_state": "failed", "parser_error": "rationale_invalid", "structured_fields": None}
    span = value["cited_source_span"]
    evidence = str(model_input.get("evidence_text", ""))
    if not isinstance(span, dict) or set(span) != {"start", "end", "text"}:
        return {"parser_state": "failed", "parser_error": "cited_source_span_invalid", "structured_fields": None}
    start = span.get("start")
    end = span.get("end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end < start
        or end > len(evidence)
        or evidence[start:end] != span.get("text")
    ):
        return {"parser_state": "failed", "parser_error": "cited_source_span_mismatch", "structured_fields": None}
    return {"parser_state": "valid", "parser_error": None, "structured_fields": value}


def build_trace_row(
    schedule_row: Mapping[str, Any],
    authority_row: Mapping[str, Any],
    response: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
) -> JsonDict:
    """Join one raw response to its sealed IDs after generation finishes."""

    raw_output = str(response.get("raw_output", ""))
    raw_response = deepcopy(response.get("raw_response", {}))
    parsed = parse_structured_output(raw_output, dict(schedule_row.get("model_input", {})))
    process = dict(resource_receipt.get("process", {}))
    cuda = dict(resource_receipt.get("cuda", {}))
    return {
        "row_order": schedule_row.get("row_order"),
        "fixture_id": schedule_row.get("fixture_id"),
        "pair_id": schedule_row.get("pair_id"),
        "source_family": schedule_row.get("source_family"),
        "input_sha256": schedule_row.get("input_sha256"),
        "prompt_sha256": schedule_row.get("prompt_sha256"),
        "source_text_sha256": schedule_row.get("source_text_sha256"),
        "claim_text_sha256": schedule_row.get("claim_text_sha256"),
        "evidence_text_sha256": schedule_row.get("evidence_text_sha256"),
        "raw_output": raw_output,
        "raw_output_sha256": sha256_text(raw_output),
        "raw_response": raw_response,
        "raw_response_sha256": sha256_json(raw_response),
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "generation_state": "failed" if response.get("error") else "complete",
        "generation_error": response.get("error"),
        **parsed,
        "authority_exact_label": authority_row.get("exact_label"),
        "authority_sidecar_join_sha256": sha256_json(dict(authority_row)),
        "pid": process.get("pid"),
        "port": process.get("port"),
        "lease_id": process.get("lease_id"),
        "gpu_uuid": process.get("gpu_uuid"),
        "cuda_placement_confirmed": cuda.get("cuda_placement_confirmed"),
        "task_owned_vram_mb": cuda.get("task_owned_vram_mb"),
    }


def trace_row_errors(row: Mapping[str, Any], schedule_row: Mapping[str, Any]) -> list[str]:
    """Recompute exact hashes and the parser state for one captured row."""

    errors: list[str] = []
    for field in (
        "row_order",
        "fixture_id",
        "pair_id",
        "source_family",
        "input_sha256",
        "prompt_sha256",
        "source_text_sha256",
        "claim_text_sha256",
        "evidence_text_sha256",
    ):
        if row.get(field) != schedule_row.get(field):
            errors.append(f"{field}_mismatch")
    if row.get("raw_output_sha256") != sha256_text(str(row.get("raw_output", ""))):
        errors.append("raw_output_hash_mismatch")
    if row.get("raw_response_sha256") != sha256_json(row.get("raw_response", {})):
        errors.append("raw_response_hash_mismatch")
    reparsed = parse_structured_output(
        str(row.get("raw_output", "")), dict(schedule_row.get("model_input", {}))
    )
    for field in ("parser_state", "parser_error", "structured_fields"):
        if row.get(field) != reparsed[field]:
            errors.append(f"{field}_mismatch")
    if row.get("authority_exact_label") not in {"supported", "unsupported"}:
        errors.append("authority_exact_label_invalid")
    if row.get("generation_state") not in {"complete", "failed"}:
        errors.append("generation_state_invalid")
    if not isinstance(row.get("latency_s"), (int, float)) or row.get("latency_s", -1) < 0:
        errors.append("latency_invalid")
    return errors


def generation_receipts(trace_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project per-call transport fields without dropping failed parses."""

    fields = (
        "row_order",
        "fixture_id",
        "prompt_tokens",
        "completion_tokens",
        "latency_s",
        "raw_output_sha256",
        "raw_response_sha256",
        "generation_state",
        "generation_error",
        "pid",
        "port",
        "lease_id",
        "gpu_uuid",
        "cuda_placement_confirmed",
        "task_owned_vram_mb",
    )
    return [{field: deepcopy(row.get(field)) for field in fields} for row in trace_rows]


def run_identity(schedule: Sequence[Mapping[str, Any]], *, manifest_sha256: str) -> JsonDict:
    """Bind checkpoint resume to every immutable pre-output input."""

    return {
        "schedule_sha256": schedule_identity(schedule),
        "model_specs_sha256": sha256_json(MODEL_SPECS),
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
        "response_schema_sha256": sha256_json(TRACE_SCHEMA),
        "decoding_parameters_sha256": sha256_json(DECODING_PARAMETERS),
        "fixture_sha256": PINNED_FIXTURE_SHA256,
        "manifest_sha256": manifest_sha256,
    }


def checkpoint_receipts(
    trace_rows: Sequence[Mapping[str, Any]], *, schedule_identity: str
) -> list[JsonDict]:
    """Describe every required four-row atomic checkpoint."""

    receipts: list[JsonDict] = []
    for row_count in range(4, len(trace_rows) + 1, 4):
        payload = {
            "row_count": row_count,
            "schedule_sha256": schedule_identity,
            "row_hashes": [sha256_json(dict(row)) for row in trace_rows[:row_count]],
        }
        receipts.append(
            {
                "row_count": row_count,
                "last_fixture_id": trace_rows[row_count - 1].get("fixture_id"),
                "schedule_sha256": schedule_identity,
                "checkpoint_sha256": sha256_json(payload),
                "atomic": True,
            }
        )
    return receipts


def write_checkpoint(
    path: Path, identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Atomically write only a complete four-row checkpoint boundary."""

    if not rows or len(rows) % 4:
        raise ValueError("checkpoint_row_cadence")
    payload = {"schema": "carnot.exp7167.checkpoint.v1", "identity": dict(identity), "rows": list(rows)}
    atomic_write_json(path, payload, allow_override=False, sort_keys=True)
    return payload


def resume_checkpoint(path: Path, expected_identity: Mapping[str, Any]) -> list[JsonDict]:
    """Resume only if the frozen run identity matches byte-for-byte."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != dict(expected_identity):
        raise ValueError("checkpoint_identity_mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows or len(rows) % 4:
        raise ValueError("checkpoint_row_cadence")
    return [dict(row) for row in rows]


def initialize_raw_storage(
    raw_dir: Path, schedule: Sequence[Mapping[str, Any]], authority: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Seal the raw manifest and authority sidecar before model output."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    authority_payload = {"schema": "carnot.exp7167.authority.v1", "rows": list(authority)}
    authority_hash = sha256_json(authority_payload)
    manifest_path = raw_dir / RAW_MANIFEST_NAME
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") != "running_unsealed":
            if (
                existing.get("schedule_sha256") != schedule_identity(schedule)
                or existing.get("authority_sha256") != authority_hash
            ):
                raise ValueError("raw_manifest_identity_mismatch")
            return existing
    atomic_write_json(
        raw_dir / AUTHORITY_SIDECAR_NAME,
        authority_payload,
        allow_override=False,
        sort_keys=True,
    )
    sealed = {
        "schema": "carnot.exp7167.raw_manifest.v1",
        "status": "sealed_before_generation",
        "schedule_rows": list(schedule),
        "schedule_sha256": schedule_identity(schedule),
        "model_specs": deepcopy(MODEL_SPECS),
        "model_specs_sha256": sha256_json(MODEL_SPECS),
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
        "response_schema_sha256": sha256_json(TRACE_SCHEMA),
        "decoding_parameters_sha256": sha256_json(DECODING_PARAMETERS),
        "authority_sidecar": AUTHORITY_SIDECAR_NAME,
        "authority_sha256": authority_hash,
        "raw_output_rows": [],
        "checkpoint_receipts": [],
    }
    sealed["seal_sha256"] = sha256_json(sealed)
    atomic_write_json(manifest_path, sealed, allow_override=False, sort_keys=True)
    return sealed


def resource_receipt_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Require exact model, task ownership, CUDA placement, and live VRAM."""

    errors: list[str] = []
    model = receipt.get("model")
    process = receipt.get("process")
    cuda = receipt.get("cuda")
    if not isinstance(model, Mapping):
        return ["model_resource_receipt_missing"]
    expected_model = {
        "repository": QWEN_MODEL_ID,
        "filename": QWEN_FILENAME,
        "quantization": QUANTIZATION,
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        errors.append("resource_model_identity_mismatch")
    if not model.get("revision") or not isinstance(model.get("bytes"), int) or model["bytes"] <= 0:
        errors.append("resource_model_provenance_incomplete")
    if not str(model.get("sha256", "")).startswith("sha256:"):
        errors.append("resource_model_hash_missing")
    if not model.get("runner_version"):
        errors.append("runner_version_missing")
    if model.get("embedded_tokenizer") is not True or model.get("embedded_chat_template") is not True:
        errors.append("embedded_model_metadata_not_confirmed")
    if not isinstance(process, Mapping) or process.get("owned_by_task") is not True:
        errors.append("task_process_not_owned")
    elif any(process.get(key) in (None, "") for key in ("pid", "port", "lease_id", "gpu_uuid")):
        errors.append("task_process_identity_incomplete")
    if not isinstance(cuda, Mapping) or cuda.get("cuda_placement_confirmed") is not True:
        errors.append("cuda_placement_not_confirmed")
    elif int(cuda.get("task_owned_vram_mb", 0) or 0) <= 0:
        errors.append("task_owned_vram_missing")
    if not isinstance(receipt.get("load_time_s"), (int, float)) or receipt.get("load_time_s", -1) < 0:
        errors.append("load_time_invalid")
    return errors


def teardown_errors(teardown: Mapping[str, Any], resource: Mapping[str, Any]) -> list[str]:
    """Prove cleanup matched the owned identity and touched no other process."""

    process = dict(resource.get("process", {}))
    errors: list[str] = []
    for key in ("pid", "port", "lease_id", "gpu_uuid"):
        if teardown.get(key) != process.get(key):
            errors.append(f"teardown_{key}_mismatch")
    for key in (
        "owned_identity_matched",
        "process_released",
        "port_released",
        "lease_released",
        "vram_released",
    ):
        if teardown.get(key) is not True:
            errors.append(f"teardown_{key}_missing")
    if teardown.get("unrelated_process_kill_count_delta") != 0:
        errors.append("teardown_touched_unrelated_process")
    return errors


def duration_errors(
    duration_s: float,
    resource: Mapping[str, Any],
    latencies: Sequence[float],
    substrate_class: str,
) -> list[str]:
    """Reject a full-generation claim shorter than its measured model work."""

    errors: list[str] = []
    if substrate_class != "model_full_generation":
        errors.append("complete_capture_wrong_substrate_class")
    measured_floor = float(resource.get("load_time_s", 0.0) or 0.0) + sum(
        float(value) for value in latencies
    )
    if float(duration_s) + 1e-9 < max(60.0, measured_floor):
        errors.append("duration_below_measured_work")
    return errors


def _generation_receipt_errors(
    receipts: Sequence[Mapping[str, Any]], trace_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    return [] if list(receipts) == generation_receipts(trace_rows) else ["generation_receipts_mismatch"]


def source_artifact_hashes(
    root: Path, *, raw_manifest: Path | None = None, model_sha256: str | None = None
) -> JsonDict:
    """Bind the fixture, prompt contract, implementation, tests, and raw log."""

    paths = {
        "fixture": root / FIXTURE_PATH,
        "module": root / MODULE_PATH,
        "wrapper": root / WRAPPER_PATH,
        "tests": root / TEST_PATH,
        "constraint_spec": root / SPEC_PATH,
        "experiment_template": root / "scripts/experiment_template.py",
        "sota_models": root / "python/carnot/inference/sota_models.py",
        "llama_server_supervisor": root / "python/carnot/inference/llama_server_supervisor.py",
    }
    hashes = {name: sha256_file(path) if path.is_file() else "missing" for name, path in paths.items()}
    manifest = raw_manifest or root / RAW_DIR / RAW_MANIFEST_NAME
    hashes.update(
        {
            "raw_manifest": sha256_file(manifest) if manifest.is_file() else "missing",
            "model_gguf": model_sha256 or "missing",
            "prompt_template": sha256_text(PROMPT_TEMPLATE),
            "response_schema": sha256_json(TRACE_SCHEMA),
            "decoding_parameters": sha256_json(DECODING_PARAMETERS),
        }
    )
    return hashes


def base_artifact(run_date: str, *, root: Path | None = None) -> JsonDict:
    """Create every required field before any resource gate can return."""

    repository = root or Path(__file__).resolve().parents[2]
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": {"host": socket.gethostname(), "gpu_uuid": None},
        "duration_s": 0.0,
        "source_artifact_hashes": source_artifact_hashes(repository),
        "rows": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "sealed_schedule_rows": [],
        "generation_receipts": [],
        "claim_evidence_trace_rows": [],
        "parser_failure_rows": [],
        "gpu_telemetry_rows": [],
        "teardown_receipt": {},
        "claim_evidence_trace_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_row("terminal_capture", True, False, False),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_claim_evidence_trace_capture",
        "checkpoint_receipts": [],
        "resource_receipt": {},
        "raw_manifest": {},
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def finish_blocked(
    artifact: Mapping[str, Any], preconditions: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Finish a stable pre-invocation failure without a partial verdict."""

    result = deepcopy(dict(artifact))
    checks = [dict(row) for row in preconditions]
    failure = next((row for row in checks if row.get("passed") is not True), None)
    if failure is None:
        failure = gate_row("blocked_without_failed_gate", False, True, False)
        checks.append(failure)
    result.update(
        {
            "status": "blocked",
            "preconditions_checked": checks,
            "inference_substrate_class": "blocked_no_run",
            "duration_s": float(duration_s),
            "claim_evidence_trace_ready_score": 0,
            "gate_check_summary": deepcopy(failure),
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failure['check']}",
        }
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def finalize_artifact(
    artifact: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    resource_receipt: Mapping[str, Any],
    gpu_rows: Sequence[Mapping[str, Any]],
    teardown: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build a positive transport result only from complete row receipts."""

    result = deepcopy(dict(artifact))
    traces = [dict(row) for row in trace_rows]
    receipts = generation_receipts(traces)
    expected_checkpoints = checkpoint_receipts(traces, schedule_identity=schedule_identity(schedule))
    complete = (
        all(row.get("passed") is True for row in preconditions)
        and not schedule_errors(schedule)
        and len(traces) == 48
        and all(row.get("generation_state") == "complete" for row in traces)
        and not resource_receipt_errors(resource_receipt)
        and not teardown_errors(teardown, resource_receipt)
        and list(checkpoints) == expected_checkpoints
        and not duration_errors(
            duration_s,
            resource_receipt,
            [float(row.get("latency_s", 0.0)) for row in traces],
            "model_full_generation",
        )
    )
    result.update(
        {
            "status": "complete" if complete else "partial",
            "preconditions_checked": [dict(row) for row in preconditions],
            "inference_substrate_class": "model_full_generation",
            "duration_s": float(duration_s),
            "rows": traces,
            "sealed_schedule_rows": [dict(row) for row in schedule],
            "generation_receipts": receipts,
            "claim_evidence_trace_rows": traces,
            "parser_failure_rows": [row for row in traces if row.get("parser_state") == "failed"],
            "gpu_telemetry_rows": [dict(row) for row in gpu_rows],
            "teardown_receipt": dict(teardown),
            "claim_evidence_trace_ready_score": int(complete),
            "gate_check_summary": gate_row("complete_transport_capture", True, complete, complete),
            "verdict_class": "positive" if complete else "partial",
            "honest_verdict": (
                "complete_positive_transport_capture_no_verifier_value_claim"
                if complete
                else "partial_incomplete_local_claim_evidence_capture"
            ),
            "checkpoint_receipts": [dict(row) for row in checkpoints],
            "resource_receipt": dict(resource_receipt),
        }
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def _load_artifact(value: Mapping[str, Any] | str | Path | object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, (str, Path)):
        return None
    try:
        loaded = json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def validate_artifact(
    value: Mapping[str, Any] | str | Path | object, *, check_source_hashes: bool = True
) -> list[str]:
    """Cold-check transport readiness from source and per-row evidence."""

    artifact = _load_artifact(value)
    if artifact is None:
        return ["artifact_unreadable"]
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"required_fields_missing:{','.join(missing)}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS:
        errors.extend(model_spec_errors(artifact.get("MODEL_SPECS", [])))
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not isinstance(artifact.get("duration_s"), (int, float)) or artifact.get("duration_s", -1) < 0:
        errors.append("duration_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")

    failed = next(
        (
            row
            for row in artifact.get("preconditions_checked", [])
            if isinstance(row, Mapping) and row.get("passed") is not True
        ),
        None,
    )
    if artifact.get("status") == "blocked":
        if failed is None:
            errors.append("blocked_failed_gate_missing")
        elif artifact.get("gate_check_summary") != failed:
            errors.append("blocked_gate_check_summary_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if artifact.get("claim_evidence_trace_ready_score") != 0:
            errors.append("blocked_readiness_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_honest_verdict_mismatch")
        return list(dict.fromkeys(errors))

    schedule = list(artifact.get("sealed_schedule_rows", []))
    traces = list(artifact.get("claim_evidence_trace_rows", []))
    fixture: Mapping[str, Any] | None = None
    root = find_repo_root()
    if check_source_hashes:
        try:
            fixture = json.loads((root / FIXTURE_PATH).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("fixture_source_unreadable")
    errors.extend(schedule_errors(schedule, fixture))
    if len(traces) != 48:
        errors.append("trace_row_count_mismatch")
    for index, (row, sealed) in enumerate(zip(traces, schedule, strict=False)):
        errors.extend(f"row_{index}:{error}" for error in trace_row_errors(row, sealed))
    if artifact.get("rows") != traces:
        errors.append("rows_projection_mismatch")
    errors.extend(_generation_receipt_errors(artifact.get("generation_receipts", []), traces))
    parser_failures = [row for row in traces if row.get("parser_state") == "failed"]
    if artifact.get("parser_failure_rows") != parser_failures:
        errors.append("parser_failure_rows_mismatch")
    resource = artifact.get("resource_receipt", {})
    errors.extend(resource_receipt_errors(resource))
    errors.extend(teardown_errors(artifact.get("teardown_receipt", {}), resource))
    expected_checkpoints = checkpoint_receipts(traces, schedule_identity=schedule_identity(schedule))
    if artifact.get("checkpoint_receipts") != expected_checkpoints:
        errors.append("checkpoint_receipts_mismatch")
    errors.extend(
        duration_errors(
            float(artifact.get("duration_s", 0.0) or 0.0),
            resource,
            [float(row.get("latency_s", 0.0) or 0.0) for row in traces],
            str(artifact.get("inference_substrate_class", "")),
        )
    )
    expected_ready = int(not errors and all(row.get("generation_state") == "complete" for row in traces))
    if artifact.get("claim_evidence_trace_ready_score") != expected_ready:
        errors.append("readiness_score_mismatch")
    if expected_ready:
        if artifact.get("status") != "complete" or artifact.get("verdict_class") != "positive":
            errors.append("complete_transport_state_mismatch")
    elif artifact.get("status") == "complete":
        errors.append("invalid_complete_state")

    if check_source_hashes and isinstance(artifact.get("source_artifact_hashes"), Mapping):
        manifest_value = artifact.get("raw_manifest", {})
        manifest_path = Path(str(manifest_value.get("path", root / RAW_DIR / RAW_MANIFEST_NAME)))
        model_hash = dict(resource.get("model", {})).get("sha256")
        expected_sources = source_artifact_hashes(
            root, raw_manifest=manifest_path, model_sha256=str(model_hash) if model_hash else None
        )
        if artifact.get("source_artifact_hashes") != expected_sources:
            errors.append("source_artifact_hashes_mismatch")
    return list(dict.fromkeys(errors))


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover
    """Print one machine-readable phase line and flush it immediately."""

    print(canonical_json({"phase": phase, "event": event, **fields}), flush=True)


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def _initialize_raw_shell(raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Write the raw-manifest shell before any host or fixture check."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    shell = {
        "schema": "carnot.exp7167.raw_manifest.v1",
        "status": "running_unsealed",
        "schedule_rows": [],
        "schedule_sha256": None,
        "model_specs": deepcopy(MODEL_SPECS),
        "model_specs_sha256": sha256_json(MODEL_SPECS),
        "prompt_template_sha256": sha256_text(PROMPT_TEMPLATE),
        "response_schema_sha256": sha256_json(TRACE_SCHEMA),
        "decoding_parameters_sha256": sha256_json(DECODING_PARAMETERS),
        "authority_sidecar": AUTHORITY_SIDECAR_NAME,
        "authority_sha256": None,
        "raw_output_rows": [],
        "checkpoint_receipts": [],
        "seal_sha256": None,
    }
    atomic_write_json(raw_dir / RAW_MANIFEST_NAME, shell, allow_override=False, sort_keys=True)
    return shell


def _collect_preflight(
    *, root: Path, run_date: str, result_path: Path, raw_dir: Path
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover
    """Check every named input and return the sealed schedule plus live state."""

    checks: list[JsonDict] = []
    context: JsonDict = {}
    checks.append(gate_row("run_date", RUN_DATE, run_date, run_date == RUN_DATE))
    fixture_path = root / FIXTURE_PATH
    fixture_hash = sha256_file(fixture_path) if fixture_path.is_file() else "missing"
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        fixture = {}
    fixture_observed = {
        "path": str(fixture_path),
        "sha256": fixture_hash,
        "status": fixture.get("status"),
        "ready_score": fixture.get("counterfactual_fixture_ready_score"),
        "row_count": len(fixture.get("rows", [])) if isinstance(fixture.get("rows"), list) else None,
    }
    fixture_ok = fixture_observed == {
        "path": str(fixture_path),
        "sha256": PINNED_FIXTURE_SHA256,
        "status": "complete",
        "ready_score": 1,
        "row_count": 648,
    }
    checks.append(
        gate_row(
            "exp7158_fixture_ready_and_exact",
            {
                "path": str(fixture_path),
                "sha256": PINNED_FIXTURE_SHA256,
                "status": "complete",
                "ready_score": 1,
                "row_count": 648,
            },
            fixture_observed,
            fixture_ok,
        )
    )

    schedule: list[JsonDict] = []
    authority: list[JsonDict] = []
    manifest: JsonDict = {}
    if fixture_ok:
        try:
            schedule, authority = build_sealed_schedule(fixture)
            selection_errors = schedule_errors(schedule, fixture)
        except (KeyError, TypeError, ValueError) as exc:
            selection_errors = [f"{type(exc).__name__}:{exc}"]
        checks.append(
            gate_row(
                "sealed_48_row_schedule",
                {"row_count": 48, "pair_count": 12, "errors": []},
                {
                    "row_count": len(schedule),
                    "pair_count": len({row.get("pair_id") for row in schedule}),
                    "errors": selection_errors,
                },
                not selection_errors,
            )
        )
        if not selection_errors:
            manifest = initialize_raw_storage(raw_dir, schedule, authority)
    else:
        checks.append(
            gate_row(
                "sealed_48_row_schedule",
                {"row_count": 48, "pair_count": 12, "errors": []},
                {"row_count": 0, "pair_count": 0, "errors": ["fixture_not_ready"]},
                False,
            )
        )

    path_state = {
        "module": (root / MODULE_PATH).is_file(),
        "wrapper": (root / WRAPPER_PATH).is_file(),
        "test": (root / TEST_PATH).is_file(),
        "spec_has_req": "REQ-VERIFY-7167" in (root / SPEC_PATH).read_text(encoding="utf-8")
        if (root / SPEC_PATH).is_file()
        else False,
    }
    checks.append(
        gate_row(
            "code_tests_and_spec",
            {key: True for key in path_state},
            path_state,
            all(path_state.values()),
        )
    )
    storage_state = {
        "raw_dir": str(raw_dir),
        "raw_dir_writable": raw_dir.is_dir() and os.access(raw_dir, os.W_OK),
        "raw_manifest_exists": (raw_dir / RAW_MANIFEST_NAME).is_file(),
        "result_path": str(result_path),
        "result_shell_exists": result_path.is_file(),
        "result_parent_writable": result_path.parent.is_dir()
        and os.access(result_path.parent, os.W_OK),
    }
    checks.append(
        gate_row(
            "writable_raw_storage_and_output",
            {
                **storage_state,
                "raw_dir_writable": True,
                "raw_manifest_exists": True,
                "result_shell_exists": True,
                "result_parent_writable": True,
            },
            storage_state,
            all(
                storage_state[key] is True
                for key in (
                    "raw_dir_writable",
                    "raw_manifest_exists",
                    "result_shell_exists",
                    "result_parent_writable",
                )
            ),
        )
    )
    lease_dir = preflight_7160.LEASE_RUNTIME_DIR
    lease_state = {
        "runtime_dir": str(lease_dir),
        "exists": lease_dir.is_dir(),
        "writable": lease_dir.is_dir() and os.access(lease_dir, os.W_OK),
        "acquire_callable": callable(lease_api.GpuLease.acquire),
    }
    checks.append(
        gate_row(
            "canonical_task_lease_support",
            {**lease_state, "exists": True, "writable": True, "acquire_callable": True},
            lease_state,
            lease_state["exists"] and lease_state["writable"] and lease_state["acquire_callable"],
        )
    )

    _progress(3, "subprocess_group_start", name="exact_cache_and_cuda_runner")
    cache_rows = preflight_7160.resolve_cache_identity()
    server_path = resolve_native_llama_server()
    runner_rows = preflight_7160.collect_runner_capabilities(server_path)
    _progress(3, "subprocess_group_end", name="exact_cache_and_cuda_runner")
    cache_errors = preflight_7160.cache_identity_errors(cache_rows)
    runner_errors = preflight_7160.runner_capability_errors(runner_rows)
    checks.append(gate_row("exact_qwen38_cache", [], cache_errors, not cache_errors))
    checks.append(gate_row("cuda_llama_runner", [], runner_errors, not runner_errors))

    _progress(4, "subprocess_group_start", name="gpu_process_and_lease_snapshot")
    process_rows, query_receipts = preflight_7160.collect_gpu_process_rows()
    lease_rows = preflight_7160.scan_lease_rows(lease_dir, process_rows)
    classified = preflight_7160.classify_process_rows(
        process_rows, lease_rows, current_task_id=TASK_ID
    )
    decision = preflight_7160.readiness_decision(
        classified, lease_rows, cache_rows, runner_rows
    )
    _progress(4, "subprocess_group_end", name="gpu_process_and_lease_snapshot")
    query_ok = all(row.get("returncode") == 0 for row in query_receipts)
    checks.append(gate_row("cuda_inventory_queries", True, query_ok, query_ok))
    process_free = {
        gpu_uuid
        for gpu_uuid in decision.get("available_gpu_uuids", [])
        if all(
            row.get("pid") is None
            for row in classified
            if row.get("gpu_uuid") == gpu_uuid
        )
    }
    idle_observed = {
        "available_gpu_uuids": sorted(process_free),
        "conflicting_processes": decision.get("conflicting_processes", []),
        "conflicting_lease_ids": decision.get("conflicting_lease_ids", []),
    }
    checks.append(
        gate_row(
            "idle_task_ownable_rtx_3090",
            {
                "minimum_count": 1,
                "no_compute_process": True,
                "no_conflicting_lease": True,
            },
            idle_observed,
            bool(process_free),
        )
    )
    context.update(
        {
            "fixture": fixture,
            "manifest": manifest,
            "cache_rows": cache_rows,
            "runner_rows": runner_rows,
            "server_path": server_path,
            "process_rows": classified,
            "lease_rows": lease_rows,
            "available_gpu_uuids": sorted(process_free),
        }
    )
    return checks, schedule, authority, context


def _free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(server: Path, model: Path, port: int) -> list[str]:  # pragma: no cover
    """Use one GPU and the tokenizer and chat template embedded in the GGUF."""

    return [
        str(server),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(CONTEXT_TOKEN_BUDGET),
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "none",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--jinja",
    ]


def _request_generation(port: int, schedule_row: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "Return only one JSON object. Use only the supplied claim and evidence.",
            },
            {"role": "user", "content": schedule_row["prompt"]},
        ],
        **DECODING_PARAMETERS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "claim_evidence_trace", "schema": TRACE_SCHEMA},
        },
    }
    encoded = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with request.urlopen(http_request, timeout=300.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = list(body.get("choices") or [{}])
    message = dict(choices[0].get("message") or {})
    usage = dict(body.get("usage") or {})
    return {
        "raw_output": str(message.get("content") or ""),
        "raw_response": body,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "latency_s": time.perf_counter() - started,
        "error": None,
    }


def _update_raw_manifest(
    raw_dir: Path,
    manifest: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    terminal: bool,
) -> JsonDict:  # pragma: no cover
    result = deepcopy(dict(manifest))
    result["status"] = "complete" if terminal else "checkpointed"
    result["raw_output_rows"] = [
        {
            "row_order": row.get("row_order"),
            "fixture_id": row.get("fixture_id"),
            "path": f"row_{int(row.get('row_order', 0)):03d}.json",
            "raw_output_sha256": row.get("raw_output_sha256"),
            "raw_response_sha256": row.get("raw_response_sha256"),
        }
        for row in traces
    ]
    result["checkpoint_receipts"] = [dict(row) for row in checkpoints]
    atomic_write_json(raw_dir / RAW_MANIFEST_NAME, result, allow_override=False, sort_keys=True)
    return result


def _port_released(port: int) -> bool:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _selected_device(context: Mapping[str, Any], gpu_uuid: str) -> JsonDict:  # pragma: no cover
    return next(
        dict(row)
        for row in context["process_rows"]
        if row.get("gpu_uuid") == gpu_uuid
    )


def _live_capture(
    *,
    schedule: Sequence[Mapping[str, Any]],
    authority: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    raw_dir: Path,
) -> tuple[list[JsonDict], JsonDict, list[JsonDict], JsonDict, list[JsonDict], JsonDict]:  # pragma: no cover
    """Own one server, generate 48 rows, checkpoint, and prove teardown."""

    gpu_uuid = str(context["available_gpu_uuids"][0])
    device = _selected_device(context, gpu_uuid)
    gpu_index = int(device["gpu_index"])
    cache = dict(context["cache_rows"][0])
    runner = dict(context["runner_rows"][0])
    port = _free_port()
    command = _server_command(Path(context["server_path"]), Path(cache["path"]), port)
    contract = supervisor_contract(
        outer_deadline_s=7_200,
        health_timeout_s=480,
        token_timeout_s=300,
        cleanup_grace_s=30,
        kill_after_cleanup_timeout_s=10,
        retry_budget=0,
        endurance_interval_s=0,
        endurance_sample_count=1,
    )
    supervisor = NativeLlamaServerSupervisor(command, raw_dir, contract)
    lease: lease_api.GpuLease | None = None
    traces: list[JsonDict] = []
    gpu_rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    resource: JsonDict = {}
    teardown: JsonDict = {}
    identity: JsonDict = {}
    load_started = 0.0
    load_time = 0.0
    manifest = dict(context["manifest"])
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    _progress(5, "phase_start", name="task_owned_model_and_generation")
    try:
        gpu_rows.append(_gpu_snapshot("before_model_load", phase=5))
        lease = lease_api.GpuLease.acquire(
            runtime_dir=preflight_7160.LEASE_RUNTIME_DIR,
            task_id=TASK_ID,
            device_uuid=gpu_uuid,
            expected_model=str(cache["path"]),
            vram_before_mb=int(device.get("gpu_memory_used_mb", 0) or 0),
            ttl_s=900.0,
        )
        lease.transition("admitted")
        lease.transition("loading")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        _progress(5, "subprocess_start", command=command)
        _progress(5, "model_load_start", model_id=QWEN_MODEL_ID, gpu_uuid=gpu_uuid)
        load_started = time.perf_counter()
        identity = supervisor.launch()
        health = _wait_for_health(supervisor, port, timeout_s=480.0)
        load_time = time.perf_counter() - load_started
        _progress(
            5,
            "model_load_end",
            model_id=QWEN_MODEL_ID,
            gpu_uuid=gpu_uuid,
            health=health.get("ok"),
            load_time_s=round(load_time, 6),
        )
        if health.get("ok") is not True:
            raise RuntimeError(f"model_health_failed:{health.get('classification')}")
        during = _gpu_snapshot("model_resident", phase=5)
        gpu_rows.append(during)
        server_log = supervisor.stderr_tail()
        cuda_source = cuda_offload_receipt(
            server_log, during, pid=int(identity["pid"]), command=command
        )
        owned_vram = int(cuda_source.get("owned_gpu_memory_mb", 0) or 0)
        resource = {
            "model": {
                "repository": QWEN_MODEL_ID,
                "filename": QWEN_FILENAME,
                "quantization": QUANTIZATION,
                "revision": cache.get("revision"),
                "bytes": cache.get("bytes"),
                "sha256": cache.get("sha256"),
                "path": cache.get("path"),
                "runner_version": runner.get("version"),
                "embedded_tokenizer": True,
                "embedded_chat_template": True,
                "server_log_sha256": sha256_text(server_log),
            },
            "process": {
                "pid": identity.get("pid"),
                "port": port,
                "lease_id": lease.lease_id,
                "gpu_uuid": gpu_uuid,
                "owned_by_task": identity.get("owned_by_task") is True,
                "process_group_id": identity.get("process_group_id"),
                "command_hash": identity.get("command_hash"),
                "lease_owner_receipt": lease.owner_receipt(),
            },
            "cuda": {
                "cuda_placement_confirmed": cuda_source.get("gpu_offload_confirmed") is True,
                "cuda_layers_offloaded": cuda_source.get("logged_offloaded_layers"),
                "cuda_markers": cuda_source.get("cuda_log_markers", []),
                "task_owned_vram_mb": owned_vram,
                "source_receipt": cuda_source,
            },
            "load_time_s": load_time,
        }
        lease.transition("resident", vram_mb=owned_vram)
        lease.transition("inferencing")
        identity_hash = run_identity(schedule, manifest_sha256=str(manifest["seal_sha256"]))
        latest = raw_dir / CHECKPOINT_NAME
        if latest.is_file():
            traces.extend(resume_checkpoint(latest, identity_hash))
            _progress(6, "checkpoint_resume", completed=len(traces), total=48)
        for batch_start in range(len(traces), len(schedule), 4):
            batch_end = min(batch_start + 4, len(schedule))
            _progress(6, "generation_batch_start", start=batch_start, end=batch_end)
            for index in range(batch_start, batch_end):
                sealed = schedule[index]
                _progress(6, "generation_start", row_order=index, fixture_id=sealed["fixture_id"])
                try:
                    response = _request_generation(port, sealed)
                except Exception as exc:  # noqa: BLE001 - the raw failure remains in its row.
                    response = {
                        "raw_output": "",
                        "raw_response": {},
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_s": 0.0,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                trace = build_trace_row(sealed, authority[index], response, resource)
                traces.append(trace)
                atomic_write_json(
                    raw_dir / f"row_{index:03d}.json",
                    {"schedule": dict(sealed), "trace": trace},
                    allow_override=False,
                    sort_keys=True,
                )
                _progress(
                    6,
                    "generation_end",
                    row_order=index,
                    fixture_id=sealed["fixture_id"],
                    parser_state=trace["parser_state"],
                    generation_state=trace["generation_state"],
                    completion_tokens=trace["completion_tokens"],
                )
            checkpoints = checkpoint_receipts(
                traces, schedule_identity=schedule_identity(schedule)
            )
            checkpoint_path = raw_dir / f"checkpoint_{len(traces):03d}.json"
            write_checkpoint(checkpoint_path, identity_hash, traces)
            write_checkpoint(latest, identity_hash, traces)
            manifest = _update_raw_manifest(
                raw_dir, manifest, traces, checkpoints, terminal=len(traces) == 48
            )
            lease.heartbeat()
            _progress(6, "generation_batch_end", completed=len(traces), total=48)
            _progress(6, "heartbeat", completed=len(traces), total=48)
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
        _progress(7, "teardown_start", pid=identity.get("pid"), port=port)
        cleanup = supervisor.cleanup()
        process_returncode = None
        if supervisor.proc is not None:
            try:
                process_returncode = supervisor.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process_returncode = supervisor.proc.poll()
        after = _gpu_snapshot("after_teardown", phase=7)
        gpu_rows.append(after)
        pid = identity.get("pid")
        pid_absent = not pid or not Path(f"/proc/{pid}").exists()
        vram_released = not any(
            app.get("pid") == pid and int(app.get("used_memory_mb", 0) or 0) > 0
            for app in after.get("compute_apps", [])
        )
        lease_receipt: JsonDict = {"released": False}
        if lease is not None:
            try:
                phase = str(lease.document.get("phase"))
                if phase in {"resident", "inferencing"}:
                    lease.transition("unloading")
                    phase = "unloading"
                if phase == "unloading":
                    lease.transition(
                        "validating",
                        vram_mb=int(device.get("gpu_memory_used_mb", 0) or 0),
                        exit_code=int(process_returncode or 0),
                        unload_observed=bool(cleanup.get("leak_free") and vram_released),
                    )
                    phase = "validating"
                if phase == "validating":
                    terminal_phase = (
                        "terminal_complete"
                        if len(traces) == 48
                        and all(row.get("generation_state") == "complete" for row in traces)
                        and cleanup.get("leak_free") is True
                        and vram_released
                        else "terminal_blocked"
                    )
                    lease.transition(terminal_phase)
                elif phase in {"preflight", "admitted", "loading"}:
                    lease.transition("terminal_blocked")
                lease_receipt = lease.release()
            except lease_api.LeaseError as exc:
                lease_receipt = {"released": False, "error": f"{type(exc).__name__}:{exc}"}
                lease.close()
        process = dict(resource.get("process", {}))
        teardown = {
            "pid": process.get("pid"),
            "port": process.get("port"),
            "lease_id": process.get("lease_id"),
            "gpu_uuid": process.get("gpu_uuid"),
            "owned_identity_matched": cleanup.get("action") != "refused",
            "process_released": cleanup.get("leak_free") is True and pid_absent,
            "port_released": _port_released(port),
            "lease_released": lease_receipt.get("released") is True,
            "vram_released": vram_released,
            "unrelated_process_kill_count_delta": cleanup.get(
                "unrelated_process_kill_count_delta", 0
            ),
            "cleanup": cleanup,
            "lease_release": lease_receipt,
            "process_returncode": process_returncode,
        }
        _progress(
            7,
            "teardown_end",
            pid=identity.get("pid"),
            port=port,
            process_released=teardown["process_released"],
            lease_released=teardown["lease_released"],
            vram_released=teardown["vram_released"],
        )
        _progress(5, "subprocess_end", command=command, returncode=process_returncode)
        _progress(5, "phase_end", name="task_owned_model_and_generation")
    return traces, resource, gpu_rows, teardown, checkpoints, manifest


def run_experiment(
    *, root: Path, run_date: str, result_path: Path, raw_dir: Path
) -> JsonDict:  # pragma: no cover
    """Run preflight once, then either block or capture under owned resources."""

    started = time.perf_counter()
    _progress(0, "phase_start", name="schema_complete_shell_and_raw_manifest")
    artifact = base_artifact(run_date, root=root)
    _progress(0, "artifact_write_start", path=str(result_path), state="running")
    _write_artifact(result_path, artifact)
    _progress(0, "artifact_write_end", path=str(result_path), state="running")
    _progress(0, "raw_manifest_write_start", path=str(raw_dir / RAW_MANIFEST_NAME))
    _initialize_raw_shell(raw_dir)
    _progress(0, "raw_manifest_write_end", path=str(raw_dir / RAW_MANIFEST_NAME))
    _progress(0, "phase_end", name="schema_complete_shell_and_raw_manifest")

    _progress(1, "phase_start", name="all_named_preconditions")
    checks, schedule, authority, context = _collect_preflight(
        root=root, run_date=run_date, result_path=result_path, raw_dir=raw_dir
    )
    artifact["sealed_schedule_rows"] = schedule
    manifest = dict(context.get("manifest", {}))
    artifact["raw_manifest"] = {
        "path": str(raw_dir / RAW_MANIFEST_NAME),
        "seal_sha256": manifest.get("seal_sha256"),
        "authority_sidecar": str(raw_dir / AUTHORITY_SIDECAR_NAME),
        "authority_sha256": manifest.get("authority_sha256"),
    }
    cache = list(context.get("cache_rows", []))
    model_hash = cache[0].get("sha256") if cache else None
    artifact["resource_receipt"] = {
        "model": {
            "repository": QWEN_MODEL_ID,
            "filename": QWEN_FILENAME,
            "quantization": QUANTIZATION,
            "revision": cache[0].get("revision") if cache else None,
            "bytes": cache[0].get("bytes") if cache else None,
            "sha256": model_hash,
            "runner_version": context.get("runner_rows", [{}])[0].get("version")
            if context.get("runner_rows")
            else None,
            "embedded_tokenizer": True,
            "embedded_chat_template": True,
        }
    }
    artifact["source_artifact_hashes"] = source_artifact_hashes(
        root,
        raw_manifest=raw_dir / RAW_MANIFEST_NAME,
        model_sha256=str(model_hash) if model_hash else None,
    )
    all_ready = bool(checks) and all(row.get("passed") is True for row in checks)
    _progress(1, "phase_end", name="all_named_preconditions", passed=all_ready)
    if not all_ready:
        result = finish_blocked(artifact, checks, duration_s=time.perf_counter() - started)
    else:
        _progress(2, "phase_start", name="sealed_schedule_before_model_output")
        _progress(
            2,
            "phase_end",
            name="sealed_schedule_before_model_output",
            row_count=len(schedule),
            pair_count=len({row["pair_id"] for row in schedule}),
        )
        traces, resource, gpu_rows, teardown, checkpoints, manifest = _live_capture(
            schedule=schedule,
            authority=authority,
            context=context,
            raw_dir=raw_dir,
        )
        artifact["execution_venue"] = {
            "host": socket.gethostname(),
            "gpu_uuid": dict(resource.get("process", {})).get("gpu_uuid"),
        }
        artifact["raw_manifest"] = {
            "path": str(raw_dir / RAW_MANIFEST_NAME),
            "seal_sha256": manifest.get("seal_sha256"),
            "authority_sidecar": str(raw_dir / AUTHORITY_SIDECAR_NAME),
            "authority_sha256": manifest.get("authority_sha256"),
        }
        artifact["source_artifact_hashes"] = source_artifact_hashes(
            root,
            raw_manifest=raw_dir / RAW_MANIFEST_NAME,
            model_sha256=str(dict(resource.get("model", {})).get("sha256") or ""),
        )
        result = finalize_artifact(
            artifact,
            preconditions=checks,
            schedule=schedule,
            trace_rows=traces,
            resource_receipt=resource,
            gpu_rows=gpu_rows,
            teardown=teardown,
            checkpoints=checkpoints,
            duration_s=time.perf_counter() - started,
        )

    _progress(8, "phase_start", name="final_validations_and_artifact")
    _progress(8, "artifact_write_start", path=str(result_path), state=result["status"])
    _write_artifact(result_path, result)
    _progress(8, "artifact_write_end", path=str(result_path), state=result["status"])
    _progress(8, "artifact_validation_start", path=str(result_path))
    errors = validate_artifact(result_path)
    _progress(8, "artifact_validation_end", valid=not errors, errors=errors)
    _progress(8, "phase_end", name="final_validations_and_artifact", status=result["status"])
    if errors:
        raise ValueError(f"terminal_artifact_invalid:{','.join(errors)}")
    return result


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        _progress(9, "subprocess_start", name="artifact_validator", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(9, "subprocess_end", name="artifact_validator", valid=not errors)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date != RUN_DATE:
        return 2
    root = find_repo_root()
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else root / args.raw_dir
    result_path.parent.mkdir(parents=True, exist_ok=True)
    run_experiment(root=root, run_date=args.date, result_path=result_path, raw_dir=raw_dir)
    return 0
