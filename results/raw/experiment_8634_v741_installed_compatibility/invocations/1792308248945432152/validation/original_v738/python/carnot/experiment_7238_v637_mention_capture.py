"""Capture the sealed V637 held-out mention-grounding corpus.

The model sees only public source and claim text. Private authority rows are
opened after the fixed 320-call schedule has terminal outcomes.

Spec refs: REQ-VERIFY-7238 and SCENARIO-VERIFY-7238-*.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

from carnot import experiment_7209_v635_span_canary as live_runtime
from carnot import experiment_7236_v637_mention_fixture as mention_fixture
from carnot import experiment_7237_v637_mention_canary as canary
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260912"
MILESTONE = "2026.09.637"
EXPERIMENT_ID = "exp7238-mention-capture"
TASK_ID = "experiment_7238_v637_mention_capture"
RANDOM_SEED = 7_238_001
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": QWEN_MODEL_ID, "quantization": QUANTIZATION}]

ARMS = ("mention_pointer", "explicit_schema_offset_control", "direct_judge")
REPRESENTATION_ARMS = ("mention_pointer", "explicit_schema_offset_control")
TOKEN_BUDGETS = {"source": 384, "claim": 128, "direct": 512}
CONTEXT_TOKEN_BUDGET = 8192
MODEL_LOAD_CAP_S = 240.0
REQUEST_CAP_S = 90.0
INFERENCE_DEADLINE_S = 3000.0
PLANNED_UNITS = 64
PLANNED_CALLS = 320
PLANNED_ROWS = 192

RESULT_PATH = Path("results/experiment_7238_v637_mention_capture.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7238")
RAW_DIR = Path("results/raw/experiment_7238")
CAPTURE_MANIFEST_PATH = RAW_DIR / "manifest.json"
FIXTURE_PATH = Path("results/experiment_7236_v637_mention_fixture.json")
UPSTREAM_PATH = Path("results/experiment_7237_v637_mention_canary.json")
PUBLIC_PATH = Path("results/raw/experiment_7236/public_manifest.json")
AUTHORITY_PATH = Path("results/raw/experiment_7236/authority_manifest.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7238_v637_mention_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7238_v637_mention_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7238_v637_mention_capture.py")

PINNED_FIXTURE_SHA256 = "sha256:b5604654dcefe5755479137c930f0ff5266b4c064aa6701f5c72fd4dec8620e3"
PINNED_CANARY_SHA256 = "sha256:e4944826d783d3bfc319899fd1bdd683e4fa36a6efe8aa053b22971bee7a5d75"
PINNED_PUBLIC_SHA256 = "sha256:b6a11618b77431d6e6d80b449662ffe1845e6c0193097eae8bc95d388171019d"
PINNED_AUTHORITY_SHA256 = "sha256:7b19fdec3833f09805b210f51231365581269ec084dc09820e1b8b7f67c74edc"

FIXTURE_EXPECTED_FIELDS: JsonDict = {
    "status": "complete",
    "run_date": RUN_DATE,
    "verdict_class": "circular_positive",
    "mention_fixture_ready_score": 1,
    "public_manifest_path": PUBLIC_PATH.as_posix(),
    "authority_manifest_path": AUTHORITY_PATH.as_posix(),
}
CANARY_EXPECTED_FIELDS: JsonDict = {
    "status": "complete",
    "run_date": RUN_DATE,
    "verdict_class": "circular_positive",
    "honest_verdict": "complete_circular_positive_mention_canary_ready_scale_gate_only",
    "mention_canary_ready_score": 1,
    "inference_substrate": "live_llm_inference",
    "inference_substrate_class": "model_bounded_generation",
    "inference_mode": "live_gpu",
}

AUTHORITY_ONLY_FIELDS = canary.AUTHORITY_ONLY_FIELDS

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
    "mention_capture_complete_score": "All 320 scheduled outcomes accounted for; bad or timed-out outputs are not removed.",
    "capture_manifest_path": "results/raw/experiment_7238/manifest.json binds every raw call to a sealed public unit.",
    "paired_unit_rows": "64 independent questions, three arms, final response provenance and missing-output penalties.",
    "decoding_cost_rows": "Per arm/unit prompt tokens, completion tokens, elapsed time and timeout status.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

DIRECT_PROMPT = (
    "Judge whether the claim is supported, contradicted, or unknown from the source. "
    "Return one JSON object with only the key decision.\nSOURCE:\n{source}\nCLAIM:\n{claim}"
)

canonical_json = mention_fixture.canonical_json
sha256_bytes = mention_fixture.sha256_bytes
sha256_file = mention_fixture.sha256_file
unwrap_principle = mention_fixture.unwrap_principle_value
is_quarantined = live_runtime.is_quarantined
load_yaml = live_runtime.load_yaml
gate_row = live_runtime.gate_row
gate_summary = live_runtime.gate_summary
_request_payload = live_runtime._request_payload


def sha256_json(value: Any) -> str:
    """Hash one value with the repository's stable JSON spelling."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding process-local clock values."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def load_held_out_manifests(
    public_path: Path, authority_path: Path
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Load the sealed held-out split without selecting on correctness labels."""

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
        deepcopy(row) for row in public.get("rows", []) if row.get("split") == "held_out"
    ]
    authority_rows = [
        deepcopy(row) for row in authority.get("rows", []) if row.get("split") == "held_out"
    ]
    if len(public_rows) != PLANNED_UNITS or len(authority_rows) != PLANNED_UNITS:
        raise ValueError("held_out_denominator")
    public_ids = [str(row.get("unit_id")) for row in public_rows]
    authority_ids = [str(row.get("unit_id")) for row in authority_rows]
    if len(set(public_ids)) != PLANNED_UNITS or public_ids != authority_ids:
        raise ValueError("held_out_identity")
    conditions = Counter(str(row.get("condition_key")) for row in authority_rows)
    if conditions != Counter(
        {"supported": 16, "reversed": 16, "joint_support": 16, "missing_support": 16}
    ):
        raise ValueError("held_out_balance")
    return public_rows, authority_rows


def _direct_grammar() -> JsonDict:
    """Constrain the baseline to one final judgment with no auxiliary score."""

    decisions = canary._enum_rule(("supported", "contradicted", "unknown"))
    grammar = (
        f'root ::= known\nknown ::= "{{\\"decision\\":" decision "}}"\ndecision ::= {decisions}\n'
    )
    digest = sha256_bytes(grammar.encode("utf-8"))
    return {
        "grammar": grammar,
        "grammar_sha256": digest,
        "reference_grammar_sha256": digest,
    }


def _paired_orders() -> list[tuple[str, str]]:
    """Freeze one deterministic arm order per unit before model inference."""

    generator = random.Random(RANDOM_SEED)
    orders = []
    for _ in range(PLANNED_UNITS):
        orders.append(
            REPRESENTATION_ARMS
            if generator.getrandbits(1) == 0
            else tuple(reversed(REPRESENTATION_ARMS))
        )
    return orders


def build_schedule(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze all 320 public requests before private semantic scoring."""

    if len(public_rows) != PLANNED_UNITS or len(authority_rows) != PLANNED_UNITS:
        raise ValueError("held_out_denominator")
    if any(row.get("split") != "held_out" for row in [*public_rows, *authority_rows]):
        raise ValueError("held_out_split")
    public_ids = [str(row.get("unit_id")) for row in public_rows]
    authority_ids = [str(row.get("unit_id")) for row in authority_rows]
    if len(set(public_ids)) != PLANNED_UNITS or public_ids != authority_ids:
        raise ValueError("held_out_identity")

    schedule: list[JsonDict] = []
    for public, arm_order in zip(public_rows, _paired_orders(), strict=True):
        unit_id = str(public["unit_id"])
        for arm in arm_order:
            for call_type in ("source", "claim"):
                document = canary.unit_document(public, call_type)
                grammar = canary.compile_grammar(arm, document, call_type)
                prompt = canary._prompt(arm, document, call_type)
                _append_schedule_row(
                    schedule,
                    unit_id=unit_id,
                    arm=arm,
                    call_type=call_type,
                    document=document,
                    model_input={
                        "text": document["text"],
                        "mentions": document["mentions"] if arm == "mention_pointer" else None,
                    },
                    prompt=prompt,
                    grammar=grammar,
                )
        source = canary.unit_document(public, "source")
        claim = canary.unit_document(public, "claim")
        _append_schedule_row(
            schedule,
            unit_id=unit_id,
            arm="direct_judge",
            call_type="direct",
            document={"source": source, "claim": claim},
            model_input={"source": source["text"], "claim": claim["text"]},
            prompt=DIRECT_PROMPT.format(source=source["text"], claim=claim["text"]),
            grammar=_direct_grammar(),
        )
    return schedule


def _append_schedule_row(
    schedule: list[JsonDict],
    *,
    unit_id: str,
    arm: str,
    call_type: str,
    document: Mapping[str, Any],
    model_input: Mapping[str, Any],
    prompt: str,
    grammar: Mapping[str, Any],
) -> None:
    """Append one call with settings and identity joined to the call seed."""

    order = len(schedule)
    seed = RANDOM_SEED + order
    settings = {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": seed,
        "cache_prompt": False,
    }
    schedule.append(
        {
            "call_order": order,
            "call_id": sha256_json(
                {
                    "task": TASK_ID,
                    "unit_id": unit_id,
                    "arm": arm,
                    "call_type": call_type,
                    "seed": seed,
                }
            ),
            "unit_id": unit_id,
            "arm": arm,
            "call_type": call_type,
            "seed": seed,
            "document": deepcopy(dict(document)),
            "model_input": deepcopy(dict(model_input)),
            "input_sha256": sha256_json(model_input),
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            **deepcopy(dict(grammar)),
            "output_token_budget": TOKEN_BUDGETS[call_type],
            "context_token_budget": CONTEXT_TOKEN_BUDGET,
            "request_timeout_s": REQUEST_CAP_S,
            "decoding_parameters": settings,
            "cold_request": True,
        }
    )


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Rebuild the frozen schedule and name each observed change."""

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
    """Seal the full held-out sample and randomized schedule before inference."""

    return {
        "selection_frozen_before_inference": True,
        "selected_unit_ids": [str(row["unit_id"]) for row in public_rows],
        "selected_unit_count": len(public_rows),
        "condition_counts": dict(
            sorted(Counter(str(row["condition_key"]) for row in authority_rows).items())
        ),
        "schedule_sha256": sha256_json(list(schedule)),
        "public_rows_sha256": sha256_json(list(public_rows)),
        "authority_rows_sha256": sha256_json(list(authority_rows)),
        "authority_fields_in_model_schedule": sum(
            bool(set(row) & AUTHORITY_ONLY_FIELDS) for row in schedule
        ),
        "paired_arm_orders": [list(order) for order in _paired_orders()],
        "retry_allowed": False,
    }


def upstream_gate_rows(
    canary_artifact: Mapping[str, Any],
    canary_bytes: bytes,
    public_bytes: bytes,
    authority_bytes: bytes,
    fixture_bytes: bytes,
    exclusion_manifest: Any,
) -> list[JsonDict]:
    """Authenticate both V637 gates and reject quarantine before invocation."""

    observed_hashes = {
        "fixture_artifact": sha256_bytes(fixture_bytes),
        "canary_artifact": sha256_bytes(canary_bytes),
        "public_manifest": sha256_bytes(public_bytes),
        "authority_manifest": sha256_bytes(authority_bytes),
    }
    expected_hashes = {
        "fixture_artifact": PINNED_FIXTURE_SHA256,
        "canary_artifact": PINNED_CANARY_SHA256,
        "public_manifest": PINNED_PUBLIC_SHA256,
        "authority_manifest": PINNED_AUTHORITY_SHA256,
    }
    rows = [
        gate_row(
            "exact_upstream_bytes",
            expected_hashes,
            observed_hashes,
            observed_hashes == expected_hashes,
            upstream="experiment_7236_and_7237",
            field="artifact_and_manifest_bytes",
        )
    ]
    try:
        fixture_artifact = json.loads(fixture_bytes)
        public = json.loads(public_bytes)
        authority = json.loads(authority_bytes)
    except json.JSONDecodeError:
        fixture_artifact, public, authority = {}, {}, {}
    quarantined = bool(
        is_quarantined(canary_artifact)
        or (isinstance(fixture_artifact, Mapping) and is_quarantined(fixture_artifact))
    )
    rows.append(
        gate_row(
            "structured_quarantine",
            False,
            quarantined,
            not quarantined,
            upstream="experiment_7236_and_7237",
            field="flagged_adversarial|quarantined|fabricated",
        )
    )
    excluded = live_runtime._manifest_hits(
        exclusion_manifest,
        {
            "Exp7236",
            "Exp7237",
            EXPERIMENT_ID,
            FIXTURE_PATH.as_posix(),
            UPSTREAM_PATH.as_posix(),
        },
    )
    rows.append(
        gate_row(
            "exclusion_manifest",
            False,
            excluded,
            not excluded,
            upstream="experiment_7236_and_7237",
            field="experiment_ids",
        )
    )
    fixture_fields = {
        field: unwrap_principle(fixture_artifact.get(field)) for field in FIXTURE_EXPECTED_FIELDS
    }
    canary_fields = {
        field: unwrap_principle(canary_artifact.get(field)) for field in CANARY_EXPECTED_FIELDS
    }
    rows.extend(
        [
            gate_row(
                "fixture_gate_fields",
                FIXTURE_EXPECTED_FIELDS,
                fixture_fields,
                fixture_fields == FIXTURE_EXPECTED_FIELDS,
                upstream="experiment_7236",
                field="terminal_fields",
            ),
            gate_row(
                "pointer_calibration_gate_fields",
                CANARY_EXPECTED_FIELDS,
                canary_fields,
                canary_fields == CANARY_EXPECTED_FIELDS,
                upstream="experiment_7237",
                field="terminal_fields",
            ),
        ]
    )
    fixture_checksum = bool(
        isinstance(fixture_artifact, Mapping)
        and fixture_artifact.get("reproducibility_checksum")
        == mention_fixture.artifact_checksum(fixture_artifact)
    )
    canary_exact_hash = observed_hashes["canary_artifact"] == PINNED_CANARY_SHA256
    canary_declared_checksum = isinstance(
        canary_artifact.get("reproducibility_checksum"), str
    ) and str(canary_artifact["reproducibility_checksum"]).startswith("sha256:")
    rows.append(
        gate_row(
            "upstream_authentication",
            {
                "experiment_7236_checksum": True,
                "experiment_7237_exact_file_hash": True,
                "experiment_7237_declared_checksum": True,
            },
            {
                "experiment_7236_checksum": fixture_checksum,
                "experiment_7237_exact_file_hash": canary_exact_hash,
                "experiment_7237_declared_checksum": canary_declared_checksum,
            },
            fixture_checksum and canary_exact_hash and canary_declared_checksum,
            upstream="experiment_7236_and_7237",
            field="reproducibility_checksum",
        )
    )
    manifest_observed = {
        "public_schema": public.get("schema"),
        "authority_schema": authority.get("schema"),
        "public_units": len(public.get("rows", [])),
        "authority_units": len(authority.get("rows", [])),
    }
    manifest_expected = {
        "public_schema": "carnot.exp7236.public_mentions.v1",
        "authority_schema": "carnot.exp7236.private_authority.v1",
        "public_units": 72,
        "authority_units": 72,
    }
    rows.append(
        gate_row(
            "manifest_authentication",
            manifest_expected,
            manifest_observed,
            manifest_observed == manifest_expected,
            upstream="experiment_7236",
            field="manifest_schema_and_counts",
        )
    )
    return rows


def _decoded_bytes(value: Any) -> bytes | None:
    """Decode retained bytes without repairing invalid base64 evidence."""

    return canary._decoded_bytes(value)


def _direct_shape_valid(value: Any) -> bool:
    """Accept only the one-key direct judgment contract."""

    return bool(
        isinstance(value, Mapping)
        and set(value) == {"decision"}
        and value.get("decision") in {"supported", "contradicted", "unknown"}
    )


def _row_hash(row: Mapping[str, Any]) -> str:
    """Hash one row without making its hash recursively self-referential."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def build_completion_row(
    sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
) -> JsonDict:
    """Preserve one attempted call, including every unusable terminal outcome."""

    if sealed.get("arm") != "direct_judge":
        row = canary.build_completion_row(sealed, response, resource)
        transport_error = response.get("error")
        timeout = bool(transport_error and "timeout" in str(transport_error).lower())
        row.update(
            {
                "attempted": True,
                "timeout": timeout,
                "transport_error": transport_error,
                "terminal_state": "timeout" if timeout else row["terminal_state"],
            }
        )
        row["row_sha256"] = _row_hash(row)
        return row

    raw = str(response.get("raw_completion") or "")
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parse_error = f"JSONDecodeError:{exc}"
    parse_valid = parse_error is None and _direct_shape_valid(parsed)
    finish_reason = response.get("finish_reason")
    truncated = finish_reason in {"length", "max_tokens"}
    request_bytes = _decoded_bytes(response.get("raw_request_bytes_b64"))
    response_bytes = _decoded_bytes(response.get("raw_response_bytes_b64"))
    raw_request = response.get("raw_request")
    request_match = bool(
        isinstance(raw_request, Mapping)
        and request_bytes == canonical_json(raw_request).encode("utf-8")
    )
    decoded_response: Any = None
    if response_bytes:
        try:
            decoded_response = json.loads(response_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            decoded_response = None
    response_match = bool(
        isinstance(response.get("raw_response"), Mapping)
        and isinstance(decoded_response, Mapping)
        and decoded_response == response.get("raw_response")
    )
    transmitted_seed = raw_request.get("seed") if isinstance(raw_request, Mapping) else None
    seed_match = (
        transmitted_seed
        == sealed.get("seed")
        == (sealed.get("decoding_parameters") or {}).get("seed")
    )
    transport_error = response.get("error")
    timeout = bool(transport_error and "timeout" in str(transport_error).lower())
    transport_complete = bool(
        transport_error is None and response_bytes and response_match and finish_reason is not None
    )
    usable = bool(
        transport_complete and parse_valid and not truncated and request_match and seed_match
    )
    errors = []
    if transport_error:
        errors.append(f"transport_error:{transport_error}")
    if parse_error:
        errors.append(parse_error)
    if not request_match:
        errors.append("request_bytes_mismatch")
    if not response_match:
        errors.append("response_bytes_mismatch")
    if not seed_match:
        errors.append("seed_join_mismatch")
    if truncated:
        errors.append("truncated")
    row: JsonDict = {
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
        "request_bytes_match": request_match,
        "raw_response_bytes_b64": response.get("raw_response_bytes_b64", ""),
        "response_bytes_sha256": sha256_bytes(response_bytes or b""),
        "response_bytes_match": response_match,
        "raw_completion": raw,
        "raw_completion_sha256": sha256_bytes(raw.encode("utf-8")),
        "parsed_completion": deepcopy(parsed),
        "compiled_completion": deepcopy(parsed) if parse_valid else None,
        "transport_complete": transport_complete,
        "parse_valid": parse_valid,
        "explicit_unknown": bool(parse_valid and parsed.get("decision") == "unknown"),
        "usable": usable,
        "truncated": truncated,
        "finish_reason": finish_reason,
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "seed_join_valid": seed_match,
        "server_pid": resource.get("server_pid"),
        "server_pid_start_ticks": resource.get("server_pid_start_ticks"),
        "gpu_uuid": resource.get("gpu_uuid"),
        "lease_id": resource.get("lease_id"),
        "cuda_offload_confirmed": resource.get("cuda_offload_confirmed") is True,
        "attempted": True,
        "timeout": timeout,
        "transport_error": transport_error,
        "terminal_state": "timeout"
        if timeout
        else ("complete" if transport_complete else "transport_error"),
        "errors": list(dict.fromkeys(errors)),
    }
    row["row_sha256"] = _row_hash(row)
    return row


def replay_completion_rows(
    schedule: Sequence[Mapping[str, Any]], retained_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Reduce exact retained response bytes again without another model call."""

    schedule_by_id = {str(row["call_id"]): row for row in schedule}
    replayed = []
    for retained in retained_rows:
        response_bytes = _decoded_bytes(retained.get("raw_response_bytes_b64"))
        raw_response = json.loads(response_bytes) if response_bytes else {}
        replayed.append(
            build_completion_row(
                schedule_by_id[str(retained["call_id"])],
                {
                    "raw_request": deepcopy(retained.get("actual_parameters") or {}),
                    "raw_request_bytes_b64": retained.get("raw_request_bytes_b64", ""),
                    "raw_response": raw_response,
                    "raw_response_bytes_b64": retained.get("raw_response_bytes_b64", ""),
                    "raw_completion": retained.get("raw_completion", ""),
                    "prompt_tokens": retained.get("prompt_tokens", 0),
                    "completion_tokens": retained.get("completion_tokens", 0),
                    "finish_reason": retained.get("finish_reason"),
                    "latency_s": retained.get("latency_s", 0.0),
                    "error": retained.get("transport_error"),
                },
                {
                    "server_pid": retained.get("server_pid"),
                    "server_pid_start_ticks": retained.get("server_pid_start_ticks"),
                    "gpu_uuid": retained.get("gpu_uuid"),
                    "lease_id": retained.get("lease_id"),
                    "cuda_offload_confirmed": retained.get("cuda_offload_confirmed"),
                },
            )
        )
    return replayed


def _semantic_view(relation: Mapping[str, Any]) -> JsonDict:
    """Compare relation meaning without representation-specific coordinates."""

    return canary._semantic_view(relation)


def score_semantics(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Score all 192 unit-arm rows and penalize every missing output."""

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
        private = authority["variants"][0]
        expected = private["exact_label"]
        source_document = canary.unit_document(public, "source")
        claim_document = canary.unit_document(public, "claim")
        expected_source = canary._expected_compiled(
            source_document, private["gold_source_completion"], "source"
        )
        expected_claim = canary._expected_compiled(
            claim_document, private["gold_claim_completion"], "claim"
        )
        for arm in ARMS:
            if arm == "direct_judge":
                direct = completion_by_key.get((unit_id, arm, "direct"))
                prediction = (
                    direct.get("parsed_completion", {}).get("decision")
                    if direct and isinstance(direct.get("parsed_completion"), Mapping)
                    else "unknown"
                )
                usable = bool(direct and direct.get("usable") is True)
                decision_correct = prediction == expected
                fully_correct = usable and decision_correct
                missing = direct is None
                rows.append(
                    _paired_row(
                        unit_id,
                        arm,
                        authority,
                        expected,
                        prediction,
                        representation_valid=usable,
                        source_fidelity=None,
                        claim_fidelity=None,
                        decision_correct=decision_correct,
                        fully_correct=fully_correct,
                        abstention=prediction == "unknown",
                        missing=missing,
                        source_call_id=None,
                        claim_call_id=None,
                        direct_call_id=direct.get("call_id") if direct else None,
                    )
                )
                continue
            source = completion_by_key.get((unit_id, arm, "source"))
            claim = completion_by_key.get((unit_id, arm, "claim"))
            compiled_source = (
                source["compiled_completion"]
                if source
                else {"outcome": "unknown", "relations": [], "errors": ["missing_call"]}
            )
            compiled_claim = (
                claim["compiled_completion"]
                if claim
                else {"outcome": "unknown", "relations": [], "errors": ["missing_call"]}
            )
            source_fidelity = bool(
                source
                and source.get("usable") is True
                and [_semantic_view(item) for item in compiled_source.get("relations", [])]
                == [_semantic_view(item) for item in expected_source.get("relations", [])]
            )
            claim_fidelity = bool(
                claim
                and claim.get("usable") is True
                and [_semantic_view(item) for item in compiled_claim.get("relations", [])]
                == [_semantic_view(item) for item in expected_claim.get("relations", [])]
            )
            representation_valid = bool(
                source and claim and source.get("usable") is True and claim.get("usable") is True
            )
            execution = mention_fixture._execute_compiled_pair(
                source_document, compiled_source, compiled_claim
            )
            prediction = execution["decision"]
            decision_correct = prediction == expected
            fully_correct = bool(
                representation_valid and source_fidelity and claim_fidelity and decision_correct
            )
            rows.append(
                _paired_row(
                    unit_id,
                    arm,
                    authority,
                    expected,
                    prediction,
                    representation_valid=representation_valid,
                    source_fidelity=source_fidelity,
                    claim_fidelity=claim_fidelity,
                    decision_correct=decision_correct,
                    fully_correct=fully_correct,
                    abstention=execution["abstention"],
                    missing=source is None or claim is None,
                    source_call_id=source.get("call_id") if source else None,
                    claim_call_id=claim.get("call_id") if claim else None,
                    direct_call_id=None,
                )
            )
    return rows


def _paired_row(
    unit_id: str,
    arm: str,
    authority: Mapping[str, Any],
    expected: str,
    prediction: str,
    *,
    representation_valid: bool,
    source_fidelity: bool | None,
    claim_fidelity: bool | None,
    decision_correct: bool,
    fully_correct: bool,
    abstention: bool,
    missing: bool,
    source_call_id: Any,
    claim_call_id: Any,
    direct_call_id: Any,
) -> JsonDict:
    """Build one fixed-denominator comparison row with response provenance."""

    return {
        "unit_id": unit_id,
        "arm": arm,
        "condition": authority["condition_key"],
        "expected_decision": expected,
        "predicted_decision": prediction,
        "representation_valid": representation_valid,
        "mention_resolution": representation_valid if arm == "mention_pointer" else None,
        "source_fidelity": source_fidelity,
        "claim_fidelity": claim_fidelity,
        "decision_correct": decision_correct,
        "fully_correct": fully_correct,
        "abstention": abstention,
        "false_accept": expected != "supported" and prediction == "supported",
        "missing_output_penalty": missing,
        "metric": int(fully_correct),
        "error": None
        if fully_correct
        else ("missing_output" if missing else "incorrect_or_unusable"),
        "final_response_provenance": {
            "source_call_id": source_call_id,
            "claim_call_id": claim_call_id,
            "direct_call_id": direct_call_id,
        },
    }


def decoding_cost_rows(completion_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project per-call token, time, timeout, and terminal costs."""

    return [
        {
            "call_order": row.get("call_order"),
            "call_id": row.get("call_id"),
            "unit_id": row.get("unit_id"),
            "arm": row.get("arm"),
            "call_type": row.get("call_type"),
            "prompt_tokens": row.get("prompt_tokens"),
            "completion_tokens": row.get("completion_tokens"),
            "elapsed_s": row.get("latency_s"),
            "timeout": row.get("timeout") is True,
            "transport_complete": row.get("transport_complete") is True,
            "terminal_state": row.get("terminal_state"),
        }
        for row in completion_rows
    ]


def completeness_receipt(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    *,
    provenance_errors: Sequence[str],
) -> JsonDict:
    """Account for every scheduled outcome without filtering quality failures."""

    schedule_ids = [str(row.get("call_id")) for row in schedule]
    row_ids = [str(row.get("call_id")) for row in completion_rows]
    terminal_states = {"complete", "transport_error", "timeout"}
    authentic = sum(
        row.get("attempted") is True
        and row.get("request_bytes_match") is True
        and row.get("seed_join_valid") is True
        and row.get("terminal_state") in terminal_states
        for row in completion_rows
    )
    transport = sum(row.get("transport_complete") is True for row in completion_rows)
    timeouts = sum(row.get("timeout") is True for row in completion_rows)
    exact_schedule = bool(
        len(schedule_ids) == PLANNED_CALLS
        and len(set(schedule_ids)) == PLANNED_CALLS
        and len(row_ids) == PLANNED_CALLS
        and len(set(row_ids)) == PLANNED_CALLS
        and set(row_ids) == set(schedule_ids)
    )
    gates = {
        "authenticated_provenance": not provenance_errors,
        "fixed_320_call_schedule": len(schedule) == PLANNED_CALLS,
        "every_scheduled_call_has_one_terminal_outcome": exact_schedule,
        "all_terminal_outcomes_authentic": authentic == PLANNED_CALLS,
    }
    return {
        "scheduled_calls": PLANNED_CALLS,
        "attempted_calls": len(completion_rows),
        "terminal_outcomes": sum(
            row.get("terminal_state") in terminal_states for row in completion_rows
        ),
        "authentic_terminal_outcomes": authentic,
        "transport_completed_calls": transport,
        "censored_calls": PLANNED_CALLS - transport,
        "timeout_calls": timeouts,
        "unattempted_calls": PLANNED_CALLS - len(completion_rows),
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
        "mention_capture_complete_score": int(all(gates.values())),
    }


def frozen_capture_settings() -> JsonDict:
    """Return the immutable model, budget, order, and stopping contract."""

    return {
        "model": deepcopy(MODEL_SPECS[0]),
        "arms": list(ARMS),
        "representation_token_budgets": {"source": 384, "claim": 128},
        "direct_token_budget": 512,
        "temperature": 0.0,
        "top_k": 1,
        "seed_base": RANDOM_SEED,
        "retry_malformed": False,
        "model_load_cap_s": MODEL_LOAD_CAP_S,
        "request_cap_s": REQUEST_CAP_S,
        "inference_deadline_s": INFERENCE_DEADLINE_S,
        "pointer_prompt_sha256": sha256_json(canary.POINTER_PROMPTS),
        "explicit_prompt_sha256": sha256_json(canary.EXPLICIT_PROMPTS),
        "direct_prompt_sha256": sha256_bytes(DIRECT_PROMPT.encode("utf-8")),
        "paired_arm_orders": [list(order) for order in _paired_orders()],
    }


def checkpoint_identity(
    schedule: Sequence[Mapping[str, Any]],
    public_sha256: str,
    authority_sha256: str,
    model_sha256: str,
) -> JsonDict:
    """Bind resume permission to model, settings, inputs, and the full schedule."""

    return {
        "schema": "carnot.exp7238.resume.v1",
        "schedule_sha256": sha256_json(list(schedule)),
        "public_sha256": public_sha256,
        "authority_sha256": authority_sha256,
        "model_sha256": model_sha256,
        "settings_sha256": sha256_json(frozen_capture_settings()),
    }


def write_resume_checkpoint(
    path: Path, identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write resumable terminal rows only below the checkpoint directory."""

    atomic_write_json(
        path,
        {
            "identity": deepcopy(dict(identity)),
            "row_count": len(rows),
            "rows": deepcopy(list(rows)),
        },
        allow_override=False,
        sort_keys=True,
    )


def resume_checkpoint(
    path: Path,
    expected_identity: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Load only exact authentic rows from a matching capture checkpoint."""

    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("identity") != expected_identity:
        raise ValueError("checkpoint_identity")
    rows = value.get("rows")
    if not isinstance(rows, list) or value.get("row_count") != len(rows):
        raise ValueError("checkpoint_rows")
    schedule_by_id = {str(row["call_id"]): row for row in schedule}
    seen: set[str] = set()
    for row in rows:
        call_id = str(row.get("call_id"))
        sealed = schedule_by_id.get(call_id)
        if call_id in seen or sealed is None:
            raise ValueError("checkpoint_call_id")
        seen.add(call_id)
        if row.get("row_sha256") != _row_hash(row):
            raise ValueError("checkpoint_row_hash")
        for field in ("call_order", "unit_id", "arm", "call_type", "seed"):
            if row.get(field) != sealed.get(field):
                raise ValueError(f"checkpoint_schedule_join:{field}")
    return deepcopy(rows)


def missing_schedule_rows(
    schedule: Sequence[Mapping[str, Any]], completed: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Return only calls with no retained terminal row."""

    complete_ids = {str(row.get("call_id")) for row in completed}
    return [deepcopy(dict(row)) for row in schedule if str(row.get("call_id")) not in complete_ids]


def base_artifact(run_date: str) -> JsonDict:
    """Create every required field before the first fallible precondition."""

    return {
        "schema": {
            "name": "carnot.experiment_7238_v637_mention_capture",
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
            "planned_independent_units": PLANNED_UNITS,
            "planned_arms": len(ARMS),
            "planned_calls": PLANNED_CALLS,
            "planned_comparison_rows": PLANNED_ROWS,
            "attempted_calls": 0,
            "transport_completed_calls": 0,
            "terminal_outcomes": 0,
            "censored_calls": PLANNED_CALLS,
            "unattempted_calls": PLANNED_CALLS,
            "completed_comparison_rows": 0,
            "stopping_rule": (
                "attempt each fixed call once within 3000 seconds; preserve every terminal "
                "outcome and never retry malformed output"
            ),
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "gate_check_summary": gate_summary(None),
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_exp7238_running_checkpoint_only",
        "acceptance_gate_results": [],
        "inference_mode": "not_run",
        "runner_receipt": {},
        "raw_request_manifest": {},
        "phase_spans": [],
        "mention_capture_complete_score": 0,
        "capture_manifest_path": CAPTURE_MANIFEST_PATH.as_posix(),
        "paired_unit_rows": [],
        "decoding_cost_rows": [],
        "timestamps": {"started_at_utc": utc_now(), "completed_at_utc": None},
        "schedule": [],
        "raw_rows": [],
        "selection_receipt": {},
        "completeness_receipt": {},
        "frozen_capture_settings": frozen_capture_settings(),
        "feasibility_projection": {},
        "model_identity_receipt": {},
        "gpu_receipts": {},
        "validation_command_rows": [],
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Finish an external pre-invocation block without claiming a null result."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    artifact["status"] = "blocked"
    artifact["verdict_class"] = "blocked"
    artifact["mention_capture_complete_score"] = 0
    artifact["gate_check_summary"] = gate_summary(failure)
    check_name = failure.get("check", "unknown") if failure else "unknown"
    artifact["honest_verdict"] = f"blocked_exp7238_{check_name}"
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
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    provenance_errors: Sequence[str],
) -> JsonDict:
    """Finish the full capture without promoting execution into scientific value."""

    receipt = completeness_receipt(schedule, completion_rows, provenance_errors=provenance_errors)
    costs = decoding_cost_rows(completion_rows)
    transport = receipt["transport_completed_calls"]
    usable = sum(row.get("usable") is True for row in completion_rows)
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": "live_llm_inference",
            "inference_substrate_class": "model_full_generation",
            "inference_mode": "live_gpu",
            "model_invoked": len(completion_rows) > 0,
            "schedule": deepcopy(list(schedule)),
            "raw_rows": deepcopy(list(completion_rows)),
            "rows": deepcopy(list(paired_rows)),
            "paired_unit_rows": deepcopy(list(paired_rows)),
            "decoding_cost_rows": costs,
            "completeness_receipt": receipt,
            "acceptance_gate_results": deepcopy(receipt["criteria"]),
            "mention_capture_complete_score": receipt["mention_capture_complete_score"],
            "duration_s": duration_s,
            "gate_check_summary": gate_summary(None),
            "raw_request_manifest": {
                "path": CAPTURE_MANIFEST_PATH.as_posix(),
                "schedule_sha256": sha256_json(list(schedule)),
                "raw_row_count": len(completion_rows),
            },
        }
    )
    artifact["runner_receipt"].update(
        {
            "scheduled_calls": PLANNED_CALLS,
            "attempted_calls": len(completion_rows),
            "transport_completed_calls": transport,
            "semantic_usable_calls": usable,
            "completed_transport": transport == PLANNED_CALLS,
            "completed_invocation_outcomes": len(completion_rows) == PLANNED_CALLS,
        }
    )
    artifact["sample_size_budget"].update(
        {
            "attempted_calls": len(completion_rows),
            "transport_completed_calls": transport,
            "terminal_outcomes": receipt["terminal_outcomes"],
            "censored_calls": receipt["censored_calls"],
            "unattempted_calls": receipt["unattempted_calls"],
            "completed_comparison_rows": len(paired_rows),
        }
    )
    complete = receipt["mention_capture_complete_score"] == 1
    artifact["verdict_class"] = "null" if complete else "disqualified"
    artifact["honest_verdict"] = (
        "complete_null_mention_capture_complete_pending_independent_semantic_audit"
        if complete
        else "complete_disqualified_mention_capture_provenance_incomplete"
    )
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check terminal fields, denominators, costs, and provenance."""

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
    if value.get("capture_manifest_path") != CAPTURE_MANIFEST_PATH.as_posix():
        errors.append("capture_manifest_path")
    if value.get("frozen_capture_settings") != frozen_capture_settings():
        errors.append("frozen_capture_settings")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    if value.get("status") == "blocked":
        summary = value.get("gate_check_summary")
        if (
            value.get("verdict_class") != "blocked"
            or value.get("mention_capture_complete_score") != 0
        ):
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
    paired = value.get("paired_unit_rows")
    costs = value.get("decoding_cost_rows")
    if not all(isinstance(item, list) for item in (schedule, raw_rows, paired, costs)):
        errors.append("rows")
        return list(dict.fromkeys(errors))
    provenance_errors = []
    identity = value.get("model_identity_receipt")
    if isinstance(identity, Mapping):
        provenance_errors.extend(identity.get("identity_errors") or [])
    else:
        provenance_errors.append("model_identity")
    receipt = completeness_receipt(schedule, raw_rows, provenance_errors=provenance_errors)
    if value.get("completeness_receipt") != receipt:
        errors.append("completeness_receipt")
    if value.get("acceptance_gate_results") != receipt["criteria"]:
        errors.append("acceptance_gate_results")
    if value.get("mention_capture_complete_score") != receipt["mention_capture_complete_score"]:
        errors.append("mention_capture_complete_score")
    if value.get("rows") != paired or len(paired) != PLANNED_ROWS:
        errors.append("paired_unit_rows")
    expected_costs = decoding_cost_rows(raw_rows)
    if costs != expected_costs or len(costs) != PLANNED_CALLS:
        errors.append("decoding_cost_rows")
    budget = value.get("sample_size_budget")
    if not isinstance(budget, Mapping) or (
        budget.get("planned_calls"),
        budget.get("attempted_calls"),
        budget.get("transport_completed_calls"),
        budget.get("terminal_outcomes"),
        budget.get("censored_calls"),
        budget.get("unattempted_calls"),
        budget.get("completed_comparison_rows"),
    ) != (
        PLANNED_CALLS,
        len(raw_rows),
        receipt["transport_completed_calls"],
        receipt["terminal_outcomes"],
        receipt["censored_calls"],
        receipt["unattempted_calls"],
        len(paired),
    ):
        errors.append("sample_size_budget")
    expected_class = "null" if receipt["mention_capture_complete_score"] else "disqualified"
    if value.get("verdict_class") != expected_class:
        errors.append("verdict_class")
    if (
        value.get("model_invoked") is not True
        or value.get("inference_substrate") != "live_llm_inference"
        or value.get("inference_substrate_class") != "model_full_generation"
        or value.get("inference_mode") != "live_gpu"
        or not isinstance(value.get("gpu_receipts"), Mapping)
        or value["gpu_receipts"].get("provenance_ok") is not True
    ):
        errors.append("live_inference_provenance")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration < 60.0:
        errors.append("full_generation_duration_floor")
    manifest = value.get("raw_request_manifest")
    if not isinstance(manifest, Mapping) or (
        manifest.get("path") != CAPTURE_MANIFEST_PATH.as_posix()
        or manifest.get("schedule_sha256") != sha256_json(schedule)
        or manifest.get("raw_row_count") != len(raw_rows)
    ):
        errors.append("raw_request_manifest")
    return list(dict.fromkeys(errors))


def attach_validation_receipts(
    artifact: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach exact validation outcomes without changing scientific evidence."""

    if validate_artifact(artifact):
        raise ValueError("validation_receipt_source_artifact")
    required = {"command", "exit_code", "classification", "summary"}
    if any(set(row) != required for row in rows):
        raise ValueError("validation_receipt_schema")
    value = deepcopy(dict(artifact))
    value["validation_command_rows"] = deepcopy(list(rows))
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def write_capture_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:
    """Bind every raw outcome to its sealed public unit and exact bytes."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "carnot.exp7238.capture.v1",
        "status": "complete" if len(completion_rows) == PLANNED_CALLS else "partial",
        "model": deepcopy(dict(model_identity)),
        "scheduled_calls": PLANNED_CALLS,
        "attempted_calls": len(completion_rows),
        "terminal_outcomes": len(completion_rows),
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
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "finish_reason": row.get("finish_reason"),
                "latency_s": row.get("latency_s"),
                "timeout": row.get("timeout"),
                "terminal_state": row.get("terminal_state"),
                "transport_complete": row.get("transport_complete"),
                "parse_valid": row.get("parse_valid"),
                "usable": row.get("usable"),
                "row_sha256": row.get("row_sha256"),
            }
            for row in completion_rows
        ],
    }
    atomic_write_json(raw_dir / "manifest.json", manifest, allow_override=False, sort_keys=True)
    return manifest


def feasibility_projection(canary_artifact: Mapping[str, Any]) -> JsonDict:
    """Project held-out cost from the measured 48-call canary before loading."""

    spans = list(canary_artifact.get("phase_spans") or [])
    generation = sum(
        float(row.get("duration_s", 0.0) or 0.0)
        for row in spans
        if row.get("name") == "generation_aggregate"
    )
    measured_calls = int(canary_artifact.get("transport_completed_calls", 0) or 0)
    per_call = generation / measured_calls if measured_calls else 0.0
    projected_generation = per_call * PLANNED_CALLS
    return {
        "source": UPSTREAM_PATH.as_posix(),
        "measured_calls": measured_calls,
        "measured_generation_s": generation,
        "measured_generation_s_per_call": per_call,
        "projected_generation_s": projected_generation,
        "startup_allowance_s": MODEL_LOAD_CAP_S,
        "projected_total_s": projected_generation + MODEL_LOAD_CAP_S,
        "inference_budget_s": INFERENCE_DEADLINE_S,
        "projected_feasible": bool(
            measured_calls == 48 and projected_generation + MODEL_LOAD_CAP_S <= INFERENCE_DEADLINE_S
        ),
    }


def _identity_errors(
    identity: Mapping[str, Any],
    gpu_receipts: Mapping[str, Any],
    completion_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Name model, CUDA, process, request-byte, and seed provenance failures."""

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
    if any(
        not row.get("server_pid")
        or not row.get("server_pid_start_ticks")
        or not row.get("gpu_uuid")
        or row.get("cuda_offload_confirmed") is not True
        for row in completion_rows
    ):
        errors.append("per_call_cuda_process_identity")
    if any(
        row.get("seed_join_valid") is not True or row.get("request_bytes_match") is not True
        for row in completion_rows
    ):
        errors.append("request_seed_join")
    return errors


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live source inventory.
    """Hash all code, public inputs, private inputs, and upstream artifacts."""

    paths = {
        "agents": Path("AGENTS.md"),
        "claude": Path("CLAUDE.md"),
        "codex": Path("CODEX.md"),
        "research_program": Path("research-program.md"),
        "research_references": Path("research-references.md"),
        "exclusion_manifest": EXCLUSION_PATH,
        "e2e_test_plan": Path("ops/e2e-test-plan.md"),
        "verification_spec": SPEC_PATH,
        "fixture_artifact": FIXTURE_PATH,
        "canary_artifact": UPSTREAM_PATH,
        "public_manifest": PUBLIC_PATH,
        "authority_manifest": AUTHORITY_PATH,
        "v636_canary": Path("python/carnot/experiment_7223_v636_span_canary.py"),
        "v634_capture": Path("python/carnot/experiment_7196_v634_qwen_atomic_capture.py"),
        "mention_fixture": Path("python/carnot/experiment_7236_v637_mention_fixture.py"),
        "mention_canary": Path("python/carnot/experiment_7237_v637_mention_canary.py"),
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
    """Flush truthful progress at each phase and long operation boundary."""

    print(
        canonical_json({"experiment": 7238, "phase": phase, "event": event, **fields}),
        flush=True,
    )


def _collect_preflight(  # pragma: no cover - live host resource boundary.
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_dir: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], JsonDict]:
    """Reuse the shipped live checks with exact V637 upstream inputs."""

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
            "manifest_path": FIXTURE_PATH,
            "spec_path": SPEC_PATH,
            "module_path": MODULE_PATH,
            "wrapper_path": WRAPPER_PATH,
            "test_path": TEST_PATH,
            "spec_req": "REQ-VERIFY-7238",
            "expected_calls": PLANNED_CALLS,
            "expected_units": PLANNED_UNITS,
            "task_id": TASK_ID,
            "upstream_id": "experiment_7237",
            "split_id": "experiment_7236_held_out",
            "upstream_gate_rows": upstream_gate_rows,
            "load_split": load_held_out_manifests,
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
    path: Path, artifact: Mapping[str, Any], started: float
) -> None:  # pragma: no cover
    """Write provisional state only below the task checkpoint directory."""

    value = deepcopy(dict(artifact))
    value["duration_s"] = time.monotonic() - started
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def _terminal(  # pragma: no cover - exercised by the real producer.
    artifact: JsonDict, result_path: Path, checkpoint_path: Path, started: float
) -> JsonDict:
    """Cold-check and atomically publish one terminal artifact."""

    _progress(8, "validation_start", path=str(result_path))
    validation_started = time.monotonic()
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["duration_s"] = time.monotonic() - started
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
        raise ValueError(f"invalid Exp7238 artifact: {errors}")
    _progress(9, "write_start", path=str(result_path))
    _checkpoint(checkpoint_path, artifact, started)
    atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
    _progress(9, "write_end", path=str(result_path))
    return artifact


def replay_terminal_artifact(root: Path | None = None) -> JsonDict:  # pragma: no cover
    """Independently replay all retained request and response bytes without inference."""

    repo = root or find_repo_root(start=__file__)
    _progress(7, "benchmark_start", operation="independent_exact_byte_reducer")
    artifact = json.loads((repo / RESULT_PATH).read_text(encoding="utf-8"))
    if artifact.get("status") == "blocked":
        upstream = json.loads((repo / UPSTREAM_PATH).read_text(encoding="utf-8"))
        checks = upstream_gate_rows(
            upstream,
            (repo / UPSTREAM_PATH).read_bytes(),
            (repo / PUBLIC_PATH).read_bytes(),
            (repo / AUTHORITY_PATH).read_bytes(),
            (repo / FIXTURE_PATH).read_bytes(),
            load_yaml(repo / EXCLUSION_PATH),
        )
        failed = [row for row in checks if row.get("passed") is not True]
        if [row.get("check") for row in failed] != ["structured_quarantine"]:
            raise ValueError("blocked_upstream_replay")
        if validate_artifact(artifact):
            raise ValueError("terminal_artifact_validation")
        _progress(
            7,
            "benchmark_end",
            operation="blocked_precondition_reducer",
            regenerated_calls=0,
        )
        return artifact
    public_rows, authority_rows = load_held_out_manifests(repo / PUBLIC_PATH, repo / AUTHORITY_PATH)
    schedule = list(artifact.get("schedule") or [])
    if schedule_errors(schedule, public_rows, authority_rows):
        raise ValueError("sealed_schedule_authentication")
    retained = list(artifact.get("raw_rows") or [])
    if len(retained) != PLANNED_CALLS:
        raise ValueError("retained_row_count")
    for sealed, row in zip(schedule, retained, strict=True):
        raw_path = repo / RAW_DIR / f"call_{int(sealed['call_order']):02d}.json"
        captured = json.loads(raw_path.read_text(encoding="utf-8"))
        if captured.get("schedule") != sealed or captured.get("completion") != row:
            raise ValueError(f"sealed_call_authentication:{sealed['call_order']}")
    replayed = replay_completion_rows(schedule, retained)
    paired = score_semantics(schedule, replayed, public_rows, authority_rows)
    if paired != artifact.get("paired_unit_rows"):
        raise ValueError("independent_semantic_reduction")
    if decoding_cost_rows(replayed) != artifact.get("decoding_cost_rows"):
        raise ValueError("independent_cost_reduction")
    if validate_artifact(artifact):
        raise ValueError("terminal_artifact_validation")
    _progress(7, "benchmark_end", rows=len(paired), calls=len(replayed))
    return artifact


def run_experiment(  # pragma: no cover - live CUDA path is verified by E2E replay.
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    output_root: Path | None = None,
) -> JsonDict:
    """Run the finite full capture or preserve one exact external block."""

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
    resume_path = checkpoint_dir / "resume.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(existing)
        if not errors:
            _progress(0, "existing_terminal", verdict=existing["honest_verdict"])
            return existing
        raise ValueError(f"existing Exp7238 artifact is invalid: {errors}")
    artifact = base_artifact(run_date)
    _checkpoint(checkpoint_path, artifact, started)
    _progress(0, "end", checkpoint=str(checkpoint_path))

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    phase_started = time.monotonic()
    _progress(1, "start", operation="preconditions_and_v637_authentication")
    checks, public_rows, authority_rows, context = _collect_preflight(
        repo, run_date, result_path, checkpoint_dir, raw_dir
    )
    artifact["phase_spans"].append(
        {
            "phase": 1,
            "name": "preconditions_and_v637_authentication",
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
    artifact["feasibility_projection"] = feasibility_projection(
        json.loads((repo / UPSTREAM_PATH).read_text(encoding="utf-8"))
    )
    _progress(
        2,
        "start",
        operation="freeze_320_public_requests",
        projected_feasible=artifact["feasibility_projection"]["projected_feasible"],
        projected_total_s=round(artifact["feasibility_projection"]["projected_total_s"], 6),
    )
    atomic_write_json(
        raw_dir / "schedule.json",
        {"schedule": schedule, "selection_receipt": artifact["selection_receipt"]},
        allow_override=False,
        sort_keys=True,
    )
    _checkpoint(checkpoint_path, artifact, started)
    _progress(2, "end", calls=len(schedule), units=PLANNED_UNITS, arms=len(ARMS))

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
        checks.append(
            gate_row(
                "embedded_tokenizer_load",
                True,
                False,
                False,
                upstream="cached_qwen_gguf",
                field="embedded_tokenizer",
            )
        )
        artifact["preconditions_checked"] = checks
        finalize_blocked_artifact(artifact, checks, time.monotonic() - started)
        return _terminal(artifact, result_path, checkpoint_path, started)
    artifact["token_budget_receipt"] = _measure_token_budgets(schedule, tokenizer)
    owner.close()
    artifact["model_identity_receipt"] = deepcopy(context["model_identity"])
    artifact["model_identity_receipt"]["tokenizer_load_receipt"] = tokenizer_load
    identity = checkpoint_identity(
        schedule,
        sha256_file(repo / PUBLIC_PATH),
        sha256_file(repo / AUTHORITY_PATH),
        str(context["model_identity"]["gguf_sha256"]),
    )
    prior_rows = resume_checkpoint(resume_path, identity, schedule)
    missing = missing_schedule_rows(schedule, prior_rows)
    _progress(4, "resume", retained_calls=len(prior_rows), missing_calls=len(missing))
    _checkpoint(checkpoint_path, artifact, started)

    capture_context = deepcopy(context)
    capture_context["schedule"] = missing
    capture_context["completion_builder"] = build_completion_row
    _progress(5, "benchmark_start", operation="fixed_320_call_held_out_capture")
    capture = live_runtime._live_capture(
        capture_context, checkpoint_dir, raw_dir, artifact["phase_spans"]
    )
    completion_rows = sorted(
        [*prior_rows, *list(capture["rows"])], key=lambda row: int(row["call_order"])
    )
    write_resume_checkpoint(resume_path, identity, completion_rows)
    _progress(
        5,
        "benchmark_end",
        returned_rows=len(completion_rows),
        transport_completed=sum(row.get("transport_complete") is True for row in completion_rows),
        runtime_error=capture["runtime_error"],
    )
    artifact["gpu_receipts"] = deepcopy(capture["gpu_receipts"])
    artifact["runner_receipt"] = deepcopy(capture["runner_receipt"])
    artifact["model_invoked"] = len(completion_rows) > 0
    if artifact["model_invoked"]:
        artifact["inference_substrate"] = "live_llm_inference"
        artifact["inference_substrate_class"] = "model_full_generation"
        artifact["inference_mode"] = "live_gpu"
    artifact["raw_rows"] = deepcopy(completion_rows)
    artifact["sample_size_budget"]["attempted_calls"] = len(completion_rows)
    _checkpoint(checkpoint_path, artifact, started)
    if len(completion_rows) != PLANNED_CALLS:
        artifact["honest_verdict"] = "partial_exp7238_resumable_capture"
        artifact["runner_receipt"]["runtime_error"] = capture["runtime_error"]
        _checkpoint(checkpoint_path, artifact, started)
        raise RuntimeError(
            f"Exp7238 capture partial: {len(completion_rows)}/{PLANNED_CALLS}; resume checkpoint retained"
        )

    _progress(6, "benchmark_start", operation="private_semantic_reduction")
    phase_started = time.monotonic()
    paired_rows = score_semantics(schedule, completion_rows, public_rows, authority_rows)
    artifact["phase_spans"].append(
        {
            "phase": 6,
            "name": "private_semantic_reduction",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    _progress(6, "benchmark_end", comparison_rows=len(paired_rows))
    provenance_errors = _identity_errors(
        artifact["model_identity_receipt"], artifact["gpu_receipts"], completion_rows
    )
    artifact["model_identity_receipt"]["identity_errors"] = provenance_errors
    artifact["runner_receipt"].update(
        {
            "server_identity": deepcopy(artifact["gpu_receipts"].get("server_identity", {})),
            "cleanup_ok": artifact["gpu_receipts"].get("cleanup", {}).get("leak_free") is True,
            "runtime_error": capture["runtime_error"],
        }
    )
    _progress(7, "write_start", operation="raw_capture_manifest")
    manifest = write_capture_manifest(
        raw_dir, schedule, completion_rows, artifact["model_identity_receipt"]
    )
    _progress(7, "write_end", operation="raw_capture_manifest", rows=len(manifest["rows"]))
    finalize_measured_artifact(
        artifact,
        schedule,
        completion_rows,
        paired_rows,
        duration_s=time.monotonic() - started,
        provenance_errors=provenance_errors,
    )
    artifact["raw_request_manifest"] = {
        "path": CAPTURE_MANIFEST_PATH.as_posix(),
        "schedule_sha256": manifest["schedule_sha256"],
        "raw_row_count": len(manifest["rows"]),
        "manifest_sha256": sha256_file(raw_dir / "manifest.json"),
    }
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    artifact["source_artifact_hashes"]["capture_manifest"] = sha256_file(raw_dir / "manifest.json")
    artifact["source_artifact_hashes"]["schedule"] = sha256_file(raw_dir / "schedule.json")
    for path in sorted(raw_dir.glob("call_*.json")):
        artifact["source_artifact_hashes"][path.stem] = sha256_file(path)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return _terminal(artifact, result_path, checkpoint_path, started)


def _measure_token_budgets(schedule: Sequence[Mapping[str, Any]], tokenizer: Any) -> JsonDict:
    """Measure each public prompt class with the embedded GGUF tokenizer."""

    receipt: JsonDict = {"measurement_status": "measured_embedded_gguf_tokenizer"}
    all_fit = True
    for call_type in ("source", "claim", "direct"):
        calls = [row for row in schedule if row.get("call_type") == call_type]
        prompt_counts = [len(tokenizer(str(row["prompt"]).encode("utf-8"))) for row in calls]
        output = (
            {"decision": "unknown"}
            if call_type == "direct"
            else {"outcome": "unknown", "relations": []}
        )
        output_count = len(tokenizer(canonical_json(output).encode("utf-8")))
        fits = bool(calls) and output_count <= TOKEN_BUDGETS[call_type]
        receipt[call_type] = {
            "form_count": len(calls),
            "maximum_prompt_tokens": max(prompt_counts),
            "maximum_minimal_output_tokens": output_count,
            "output_budget": TOKEN_BUDGETS[call_type],
            "fits": fits,
        }
        all_fit = all_fit and fits
    receipt["all_forms_fit"] = all_fit
    return receipt


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V637 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the capture and return success only for a cold-valid terminal result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    args = parser.parse_args(argv)
    artifact = run_experiment(run_date=args.date)
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7238] invalid artifact: {errors}", flush=True)
        return 1
    print(
        f"[exp7238] terminal verdict={artifact['honest_verdict']} "
        f"complete={artifact['mention_capture_complete_score']}",
        flush=True,
    )
    return 0
