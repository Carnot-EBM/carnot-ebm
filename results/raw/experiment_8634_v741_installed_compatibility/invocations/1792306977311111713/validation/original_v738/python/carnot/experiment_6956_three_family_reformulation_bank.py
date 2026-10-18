"""Acquire a frozen local-GGUF bank of reformulation mapping proposals.

Spec refs: REQ-VERIFY-6956 and SCENARIO-VERIFY-6956-*.

This module deliberately separates acquisition from exact certification. It
shows each model only two public formulations, stores the unedited response
before parsing it, and never imports or calls either Exp6955 exact engine. A
later task may certify these frozen bytes; this task measures only whether the
planned proposal bank was durably acquired.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import resolve_cached_gguf
from carnot.task_runtime_receipts import sha256_file, write_json_atomic


JsonDict = dict[str, Any]
InferenceCall = Callable[[Mapping[str, Any]], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ARTIFACT_PATH = REPO_ROOT / "results/experiment_6955_reformulation_fixture.json"
FIXTURE_CHECKPOINT_PATH = (
    REPO_ROOT / "results/checkpoints/experiment_6955_reformulation_fixture_corpus.json"
)
RUNTIME_QUALIFICATION_PATH = (
    REPO_ROOT / "results/experiment_6928_sota_runtime_receipt_qualification.json"
)
RESULT_PATH = REPO_ROOT / "results/experiment_6956_three_family_reformulation_bank.json"
CHECKPOINT_PATH = (
    REPO_ROOT
    / "results/checkpoints/experiment_6956_three_family_reformulation_bank.checkpoint.json"
)

SCHEMA_VERSION = "carnot.exp6956.three_family_reformulation_bank.v1"
CHECKPOINT_SCHEMA_VERSION = "carnot.exp6956.reformulation_bank_checkpoint.v1"
PLAN_SCHEMA_VERSION = "carnot.exp6956.reformulation_bank_plan.v1"
INFERENCE_SUBSTRATE = "task_owned_local_gguf_reformulation_mapping_generation"
TASK_ID = "exp6956-three-family-reformulation-bank"
RANDOM_SEED = 695620260903
OUTPUT_TOKEN_CAP = 512
CONTEXT_SIZE = 4096
MODEL_CALL_TIMEOUT_S = 240.0
MODEL_LOAD_TIMEOUT_S = 240.0
MIN_DISK_FREE_BYTES = 1_000_000_000
EXPECTED_ATTEMPT_COUNT = 162

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "family": "qwen3.6_moe",
        "quantization": "Q4_K_M",
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "family": "gemma4_dense",
        "quantization": "Q4_K_M",
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "family": "gemma4_moe",
        "quantization": "Q4_K_M",
    },
)

# Six IDs from each held-out generator template keep the family counts equal.
# IDs 0..4 are equivalent fixture rows and ID 5 is a single-edit hard negative,
# but those labels are intentionally absent from every object returned by
# ``load_public_pairs`` and from every prompt.
HELD_OUT_PAIR_IDS = (
    "0-3-0",
    "0-3-1",
    "0-3-2",
    "0-3-3",
    "0-3-4",
    "0-3-5",
    "1-3-0",
    "1-3-1",
    "1-3-2",
    "1-3-3",
    "1-3-4",
    "1-3-5",
    "2-3-0",
    "2-3-1",
    "2-3-2",
    "2-3-3",
    "2-3-4",
    "2-3-5",
)

PROMPT_VARIANTS: tuple[JsonDict, ...] = (
    {
        "prompt_variant_id": "direct_affine",
        "instruction": "Infer the explicit affine variable and objective correspondence.",
    },
    {
        "prompt_variant_id": "domain_first",
        "instruction": "Compare variable domains first, then state the proposed correspondence.",
    },
    {
        "prompt_variant_id": "objective_first",
        "instruction": "Compare objective direction and scale first, then state the correspondence.",
    },
)

DECODING_SETTINGS = {
    "temperature": 0.35,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "max_tokens": OUTPUT_TOKEN_CAP,
}

MAPPING_RESPONSE_SCHEMA = {
    "mapping": {
        "schema_version": "carnot.reformulation_mapping.v1",
        "variables": [
            {
                "source": "string",
                "target": "string",
                "scale": "rational string",
                "offset": "rational string",
            }
        ],
        "domain_clauses": [
            {
                "source": "string",
                "target": "string",
                "source_lower": "rational string",
                "source_upper": "rational string",
                "target_lower": "rational string",
                "target_upper": "rational string",
            }
        ],
        "objective": {
            "source_direction": "min or max",
            "target_direction": "min or max",
            "scale": "nonzero rational string",
            "offset": "rational string",
        },
        "claimed_relation": "equivalent or non_equivalent",
    },
    "confidence": "optional number from 0 to 1",
    "rationale": "optional short string",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "model_specs",
    "model_rows",
    "pair_rows",
    "attempt_rows",
    "prompt_variant_rows",
    "candidate_group_rows",
    "raw_output_rows",
    "parse_rows",
    "schema_rows",
    "confidence_rows",
    "rationale_rows",
    "diversity_rows",
    "hidden_label_isolation_rows",
    "checkpoint_rows",
    "model_lifecycle_rows",
    "task_runtime_receipt",
    "random_seed",
    "reproducibility_checksum",
    "reformulation_bank_complete_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A scientific reason per field makes the evidence contract reviewable.",
    "preconditions_checked": "Fail-closed host checks prevent a CPU or partial-cache run from posing as the bank.",
    "inference_substrate": "The fixed substrate declaration excludes remote and exact-checker generation.",
    "duration_s": "Measured wall time distinguishes a live acquisition from a static artifact rewrite.",
    "source_artifact_hashes": "Content hashes bind proposals to the exact fixture, runtime contract, and code.",
    "rows": "The complete attempt surface lets downstream audits ignore potentially wrong aggregates.",
    "model_specs": "Concrete current GGUF identities exclude legacy smoke models from headline rows.",
    "model_rows": "Per-model counts expose family-specific acquisition failures.",
    "pair_rows": "Per-pair counts prove the frozen held-out roster was not selectively sampled.",
    "attempt_rows": "One terminal row per frozen key keeps every failure in the denominator.",
    "prompt_variant_rows": "Variant counts prove all three independent prompt views were attempted.",
    "candidate_group_rows": "Group receipts prove three candidates exist for every model-pair cell.",
    "raw_output_rows": "Unedited text and token IDs preserve the candidate before interpretation.",
    "parse_rows": "Parse outcomes remain separate from raw generation and semantic correctness.",
    "schema_rows": "Strict schema decisions expose malformed mappings without repair.",
    "confidence_rows": "Self-reported confidence stays visible but has no authority over completion.",
    "rationale_rows": "Optional rationales remain auditable without becoming exact evidence.",
    "diversity_rows": "Raw-hash groups reveal copied candidates without deleting them.",
    "hidden_label_isolation_rows": "Per-prompt audits prove exact labels and later evidence were not shown.",
    "checkpoint_rows": "Two durable stages prove raw bytes existed before parsing.",
    "model_lifecycle_rows": "Sequential load and teardown receipts prevent cross-model state contamination.",
    "task_runtime_receipt": "Task-owned call and lifecycle hashes bind runtime evidence to this acquisition.",
    "random_seed": "A fixed seed roster makes stochastic attempts reproducible and distinguishable.",
    "reproducibility_checksum": "A timing-free digest detects drift in the frozen plan and acquired bytes.",
    "reformulation_bank_complete_score": "The binary score measures terminal coverage and lifecycle closure only.",
    "gate_check_summary": "Expected and observed values make blocked or partial outcomes actionable.",
    "verifier_is_oracle": "False prevents parsing or self-reports from becoming semantic truth.",
    "verdict_class": "The closed class separates a complete null bank from positive science evidence.",
    "honest_verdict": "A terminal prefix gives automation a stable summary of the acquisition outcome.",
}

_TOP_LEVEL_KEYS = {"mapping", "confidence", "rationale"}
_MAPPING_KEYS = {"schema_version", "variables", "domain_clauses", "objective", "claimed_relation"}
_VARIABLE_KEYS = {"source", "target", "scale", "offset"}
_DOMAIN_KEYS = {
    "source",
    "target",
    "source_lower",
    "source_upper",
    "target_lower",
    "target_upper",
}
_OBJECTIVE_KEYS = {"source_direction", "target_direction", "scale", "offset"}
_FORBIDDEN_PROMPT_KEYS = (
    "expected_label",
    "exact_label",
    "canonical_mapping",
    "feasibility_witness",
    "objective_order_witness",
    "counterexample",
    "z3_label",
    "enumeration_label",
    "mapping_hash",
)
_FORBIDDEN_PROMPT_PHRASES = (
    "other candidate",
    "prior candidate",
    "previous candidate",
    "solver output",
    "later memory",
)


class BankError(RuntimeError):
    """Name a fail-closed proposal-bank contract violation."""


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes and exact aggregate comparison."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return the project spelling of a SHA-256 digest over UTF-8 text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(value))


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Build one precondition row whose pass decision is exact equality."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retain expected and observed values for each failed precondition."""

    return [
        {
            "failed_check": row.get("check"),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in checks
        if row.get("passed") is not True
    ]


def load_public_pairs(fixture: Mapping[str, Any]) -> list[JsonDict]:
    """Extract only public formulations for the preregistered held-out IDs.

    Exp6955 stores labels, mappings, and witnesses alongside its public
    formulation rows. This function never copies those authority fields into
    the returned objects, so prompt construction has no label-bearing input.
    """

    formulations: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in fixture.get("formulation_rows", []):
        if isinstance(row, Mapping):
            formulations[(str(row.get("pair_id")), str(row.get("side")))] = row
    pair_metadata = {
        str(row.get("pair_id")): row
        for row in fixture.get("pair_rows", [])
        if isinstance(row, Mapping)
    }
    public: list[JsonDict] = []
    for pair_id in HELD_OUT_PAIR_IDS:
        meta = pair_metadata.get(pair_id)
        source = formulations.get((pair_id, "source"))
        target = formulations.get((pair_id, "target"))
        if meta is None or meta.get("split") != "held_out" or source is None or target is None:
            raise BankError(f"held_out_pair_unavailable:{pair_id}")
        public.append(
            {
                "pair_id": pair_id,
                "family": str(meta.get("family")),
                "source_formulation": deepcopy(source.get("formulation")),
                "target_formulation": deepcopy(target.get("formulation")),
            }
        )
    family_counts = Counter(row["family"] for row in public)
    if sorted(family_counts.values()) != [6, 6, 6]:
        raise BankError(f"held_out_family_balance:{dict(family_counts)}")
    return public


def build_prompt(pair: Mapping[str, Any], variant: Mapping[str, Any]) -> str:
    """Render one label-free request for mapping JSON and optional self-report."""

    public_input = {
        "pair_id": pair["pair_id"],
        "source_formulation": pair["source_formulation"],
        "target_formulation": pair["target_formulation"],
    }
    return (
        "/no_think\n"
        "Propose a semantic mapping between the two bounded optimization formulations. "
        f"{variant['instruction']} Do not solve either optimization problem. "
        "Return exactly one JSON object and no markdown. Use rational numbers as strings. "
        "The confidence and rationale fields are optional; keep rationale under 240 characters.\n"
        f"PUBLIC_INPUT={canonical_json(public_input)}\n"
        f"RESPONSE_SCHEMA={canonical_json(MAPPING_RESPONSE_SCHEMA)}"
    )


def audit_prompt(prompt: str) -> list[str]:
    """Reject exact-authority keys and candidate-bearing phrases in prompt bytes."""

    lowered = prompt.lower()
    failures = [
        f"forbidden_prompt_key:{key}" for key in _FORBIDDEN_PROMPT_KEYS if f'"{key}"' in lowered
    ]
    failures.extend(
        f"forbidden_prompt_phrase:{phrase}"
        for phrase in _FORBIDDEN_PROMPT_PHRASES
        if phrase in lowered
    )
    return failures


def _plan_hash(plan: Mapping[str, Any]) -> str:
    """Hash a plan without its self-referential digest."""

    return sha256_json({key: value for key, value in plan.items() if key != "plan_sha256"})


def freeze_plan(
    public_pairs: Sequence[Mapping[str, Any]],
    *,
    model_specs: Sequence[Mapping[str, Any]] = MODEL_SPECS,
    prompt_variants: Sequence[Mapping[str, Any]] = PROMPT_VARIANTS,
) -> JsonDict:
    """Freeze the complete model, pair, variant, seed, and prompt roster."""

    attempts: list[JsonDict] = []
    ordinal = 0
    for model in model_specs:
        for pair in public_pairs:
            for variant in prompt_variants:
                prompt = build_prompt(pair, variant)
                failures = audit_prompt(prompt)
                attempt_key = "|".join(
                    (
                        str(model["hf_id"]),
                        str(pair["pair_id"]),
                        str(variant["prompt_variant_id"]),
                    )
                )
                attempts.append(
                    {
                        "attempt_key": attempt_key,
                        "ordinal": ordinal,
                        "hf_id": str(model["hf_id"]),
                        "model_family": str(model["family"]),
                        "pair_id": str(pair["pair_id"]),
                        "problem_family": str(pair["family"]),
                        "prompt_variant_id": str(variant["prompt_variant_id"]),
                        "random_seed": RANDOM_SEED + ordinal,
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "hidden_label_isolation_errors": failures,
                        "source_formulation": deepcopy(pair["source_formulation"]),
                        "target_formulation": deepcopy(pair["target_formulation"]),
                    }
                )
                ordinal += 1
    plan: JsonDict = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "pair_ids": [str(row["pair_id"]) for row in public_pairs],
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "prompt_variants": [deepcopy(dict(row)) for row in prompt_variants],
        "decoding_settings": dict(DECODING_SETTINGS),
        "mapping_response_schema": deepcopy(MAPPING_RESPONSE_SCHEMA),
        "attempts": attempts,
        "plan_sha256": "",
    }
    plan["plan_sha256"] = _plan_hash(plan)
    return plan


def _names(formulation: Mapping[str, Any]) -> list[str]:
    """Read declared variable names for structural mapping validation."""

    return [str(row.get("name")) for row in formulation.get("variables", [])]


def _mapping_schema_error(mapping: Any, pair: Mapping[str, Any]) -> str | None:
    """Return one stable structural rejection without modifying the mapping."""

    if not isinstance(mapping, Mapping) or set(mapping) != _MAPPING_KEYS:
        return "mapping_keys"
    if mapping.get("schema_version") != "carnot.reformulation_mapping.v1":
        return "mapping_schema_version"
    variables = mapping.get("variables")
    domains = mapping.get("domain_clauses")
    objective = mapping.get("objective")
    if not isinstance(variables, list) or any(
        not isinstance(row, Mapping) or set(row) != _VARIABLE_KEYS for row in variables
    ):
        return "variable_rows"
    if not isinstance(domains, list) or any(
        not isinstance(row, Mapping) or set(row) != _DOMAIN_KEYS for row in domains
    ):
        return "domain_clause_rows"
    if not isinstance(objective, Mapping) or set(objective) != _OBJECTIVE_KEYS:
        return "objective_keys"
    source_names = _names(pair["source_formulation"])
    target_names = _names(pair["target_formulation"])
    mapped_sources = [row.get("source") for row in variables]
    mapped_targets = [row.get("target") for row in variables]
    if sorted(mapped_sources) != sorted(source_names) or len(set(mapped_sources)) != len(
        mapped_sources
    ):
        return "source_variable_roster"
    if sorted(mapped_targets) != sorted(target_names) or len(set(mapped_targets)) != len(
        mapped_targets
    ):
        return "target_variable_roster"
    domain_sources = [row.get("source") for row in domains]
    domain_targets = [row.get("target") for row in domains]
    if sorted(domain_sources) != sorted(source_names) or len(set(domain_sources)) != len(
        domain_sources
    ):
        return "source_domain_roster"
    if sorted(domain_targets) != sorted(target_names) or len(set(domain_targets)) != len(
        domain_targets
    ):
        return "target_domain_roster"
    rational_fields = [
        *(row.get(field) for row in variables for field in ("scale", "offset")),
        *(
            row.get(field)
            for row in domains
            for field in ("source_lower", "source_upper", "target_lower", "target_upper")
        ),
        objective.get("scale"),
        objective.get("offset"),
    ]
    if any(not isinstance(value, str) or not value for value in rational_fields):
        return "rational_string_required"
    if objective.get("source_direction") not in {"min", "max"} or objective.get(
        "target_direction"
    ) not in {"min", "max"}:
        return "objective_direction"
    if mapping.get("claimed_relation") not in {"equivalent", "non_equivalent"}:
        return "claimed_relation"
    return None


def parse_candidate(raw_text: str, pair: Mapping[str, Any]) -> JsonDict:
    """Parse strict JSON and record validity without extracting or repairing text."""

    if not raw_text:
        return {
            "json_valid": False,
            "schema_valid": False,
            "failure_reason": "empty_output",
            "parsed_candidate": None,
            "confidence": None,
            "rationale": None,
        }
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return {
            "json_valid": False,
            "schema_valid": False,
            "failure_reason": "malformed_json",
            "parsed_candidate": None,
            "confidence": None,
            "rationale": None,
        }
    if not isinstance(parsed, Mapping):
        reason = "response_object_required"
    elif "mapping" not in parsed or not set(parsed) <= _TOP_LEVEL_KEYS:
        reason = "response_keys"
    else:
        confidence = parsed.get("confidence")
        rationale = parsed.get("rationale")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            reason = "confidence_range"
        elif rationale is not None and (not isinstance(rationale, str) or len(rationale) > 240):
            reason = "rationale_length"
        else:
            reason = _mapping_schema_error(parsed["mapping"], pair)
    return {
        "json_valid": True,
        "schema_valid": reason is None,
        "failure_reason": reason,
        "parsed_candidate": deepcopy(parsed),
        "confidence": parsed.get("confidence") if isinstance(parsed, Mapping) else None,
        "rationale": parsed.get("rationale") if isinstance(parsed, Mapping) else None,
    }


def _checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    """Hash a checkpoint without its self-referential digest."""

    return sha256_json(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )


def new_checkpoint(
    plan: Mapping[str, Any],
    *,
    model_hashes: Mapping[str, str],
    tokenizer_bindings: Mapping[str, str],
) -> JsonDict:
    """Create an empty checkpoint bound to every immutable acquisition input."""

    checkpoint: JsonDict = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "model_hashes": dict(model_hashes),
        "tokenizer_bindings": dict(tokenizer_bindings),
        "attempt_rows": [],
        "model_lifecycle_rows": [],
        "checkpoint_sha256": "",
    }
    checkpoint["checkpoint_sha256"] = _checkpoint_hash(checkpoint)
    return checkpoint


def write_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> None:
    """Atomically publish a checkpoint after recomputing its content digest."""

    payload = deepcopy(dict(checkpoint))
    payload["checkpoint_sha256"] = _checkpoint_hash(payload)
    write_json_atomic(path, payload)


def load_checkpoint(
    path: str | Path,
    *,
    plan: Mapping[str, Any],
    model_hashes: Mapping[str, str],
    tokenizer_bindings: Mapping[str, str],
) -> JsonDict:
    """Load one checkpoint and reject corruption or any immutable-input drift."""

    target = Path(path)
    try:
        checkpoint = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise BankError("checkpoint_malformed_json") from exc
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise BankError("checkpoint_schema_mismatch")
    if checkpoint.get("checkpoint_sha256") != _checkpoint_hash(checkpoint):
        raise BankError("checkpoint_checksum_invalid")
    if checkpoint.get("plan_sha256") != plan.get("plan_sha256"):
        raise BankError("plan_hash_drift")
    if checkpoint.get("model_hashes") != dict(model_hashes):
        raise BankError("model_hash_drift")
    if checkpoint.get("tokenizer_bindings") != dict(tokenizer_bindings):
        raise BankError("tokenizer_binding_drift")
    keys = [str(row.get("attempt_key")) for row in checkpoint.get("attempt_rows", [])]
    if len(keys) != len(set(keys)):
        raise BankError("checkpoint_duplicate_attempt_key")
    return checkpoint


def persist_raw_attempt(
    checkpoint: Mapping[str, Any],
    attempt: Mapping[str, Any],
    result: Mapping[str, Any],
) -> JsonDict:
    """Add the raw-only stage; parsing is intentionally absent from this write."""

    payload = deepcopy(dict(checkpoint))
    rows = payload.setdefault("attempt_rows", [])
    key = str(attempt["attempt_key"])
    if any(row.get("attempt_key") == key for row in rows):
        raise BankError(f"duplicate_attempt_key:{key}")
    raw_text = str(result.get("raw_text") or "")
    receipt = deepcopy(dict(result.get("runtime_receipt") or {}))
    receipt.update(
        {
            "task_id": TASK_ID,
            "attempt_key": key,
            "hf_id": attempt["hf_id"],
            "prompt_sha256": attempt["prompt_sha256"],
            "raw_sha256": sha256_text(raw_text),
        }
    )
    rows.append(
        {
            "attempt_key": key,
            "ordinal": attempt["ordinal"],
            "hf_id": attempt["hf_id"],
            "model_family": attempt["model_family"],
            "pair_id": attempt["pair_id"],
            "problem_family": attempt["problem_family"],
            "prompt_variant_id": attempt["prompt_variant_id"],
            "random_seed": attempt["random_seed"],
            "prompt_sha256": attempt["prompt_sha256"],
            "source_formulation": deepcopy(attempt["source_formulation"]),
            "target_formulation": deepcopy(attempt["target_formulation"]),
            "hidden_label_isolation_errors": list(attempt.get("hidden_label_isolation_errors", [])),
            "raw_text": raw_text,
            "raw_sha256": sha256_text(raw_text),
            "output_token_ids": [int(value) for value in result.get("output_token_ids", [])],
            "prompt_token_ids": [int(value) for value in result.get("prompt_token_ids", [])],
            "latency_s": float(result.get("latency_s", 0.0) or 0.0),
            "call_status": str(result.get("call_status") or "runner_failure"),
            "failure_reason": result.get("failure_reason"),
            "runtime_receipt": receipt,
            "raw_durable": True,
            "parse_state": "pending",
            "parse": None,
            "terminal": False,
        }
    )
    payload["checkpoint_sha256"] = _checkpoint_hash(payload)
    return payload


def finalize_pending_attempt(checkpoint: Mapping[str, Any], attempt_key: str) -> JsonDict:
    """Parse one already-durable raw row and mark its second checkpoint stage."""

    payload = deepcopy(dict(checkpoint))
    row = next(
        (
            item
            for item in payload.get("attempt_rows", [])
            if item.get("attempt_key") == attempt_key
        ),
        None,
    )
    if row is None:
        raise BankError(f"attempt_not_found:{attempt_key}")
    if row.get("terminal") is True:
        return payload
    if row.get("raw_durable") is not True or row.get("parse_state") != "pending":
        raise BankError(f"raw_stage_not_durable:{attempt_key}")
    if row.get("call_status") != "complete":
        parse = {
            "json_valid": False,
            "schema_valid": False,
            "failure_reason": row.get("failure_reason") or row.get("call_status"),
            "parsed_candidate": None,
            "confidence": None,
            "rationale": None,
        }
    else:
        parse = parse_candidate(
            str(row.get("raw_text", "")),
            {
                "source_formulation": row["source_formulation"],
                "target_formulation": row["target_formulation"],
            },
        )
    row["parse"] = parse
    row["parse_state"] = "terminal"
    row["terminal"] = True
    payload["checkpoint_sha256"] = _checkpoint_hash(payload)
    return payload


def _failure_result(status: str, reason: str, *, latency_s: float) -> JsonDict:
    """Create a raw-stage result for a failed call without inventing tokens."""

    return {
        "raw_text": "",
        "output_token_ids": [],
        "prompt_token_ids": [],
        "latency_s": latency_s,
        "call_status": status,
        "failure_reason": reason,
        "runtime_receipt": {},
    }


def resume_attempts(
    plan: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    model_hashes: Mapping[str, str],
    tokenizer_bindings: Mapping[str, str],
    infer: InferenceCall,
) -> JsonDict:
    """Complete missing stages while never regenerating an existing raw row."""

    checkpoint = load_checkpoint(
        checkpoint_path,
        plan=plan,
        model_hashes=model_hashes,
        tokenizer_bindings=tokenizer_bindings,
    )
    for attempt in plan["attempts"]:
        key = str(attempt["attempt_key"])
        current = next(
            (row for row in checkpoint["attempt_rows"] if row.get("attempt_key") == key), None
        )
        if current is not None and current.get("terminal") is True:
            continue
        if current is None:
            started = time.perf_counter()
            try:
                result = infer(attempt)
                if not isinstance(result, Mapping):
                    raise TypeError("inference result must be a mapping")
            except TimeoutError:
                result = _failure_result(
                    "timeout", "timeout", latency_s=time.perf_counter() - started
                )
            except Exception as exc:
                result = _failure_result(
                    "runner_failure",
                    f"{type(exc).__name__}:{exc}",
                    latency_s=time.perf_counter() - started,
                )
            checkpoint = persist_raw_attempt(checkpoint, attempt, result)
            write_checkpoint(checkpoint_path, checkpoint)
        checkpoint = finalize_pending_attempt(checkpoint, key)
        write_checkpoint(checkpoint_path, checkpoint)
    checkpoint["checkpoint_sha256"] = _checkpoint_hash(checkpoint)
    return checkpoint


def diversity_rows(attempt_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Group all raw hashes and flag repeated non-empty candidate bytes."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in attempt_rows:
        grouped[str(row.get("raw_sha256"))].append(row)
    return [
        {
            "raw_sha256": raw_hash,
            "attempt_keys": sorted(str(row.get("attempt_key")) for row in members),
            "attempt_count": len(members),
            "empty_output": all(not str(row.get("raw_text", "")) for row in members),
            "duplicate_nonempty": len(members) > 1
            and any(bool(str(row.get("raw_text", ""))) for row in members),
        }
        for raw_hash, members in sorted(grouped.items())
    ]


def lifecycle_errors(
    lifecycle_rows: Sequence[Mapping[str, Any]], model_specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Validate one closed, hash-bound, sequential lifecycle per model."""

    errors: list[str] = []
    expected_ids = [str(row["hf_id"]) for row in model_specs]
    observed_ids = [str(row.get("hf_id")) for row in lifecycle_rows]
    if observed_ids != expected_ids:
        errors.append("model_lifecycle_roster_or_order")
    by_id = {str(row.get("hf_id")): row for row in lifecycle_rows}
    for spec in model_specs:
        hf_id = str(spec["hf_id"])
        row = by_id.get(hf_id, {})
        if row.get("model_sha256") != spec.get("model_sha256"):
            errors.append(f"lifecycle_model_hash_drift:{hf_id}")
        if row.get("cuda_offload_authenticated") is not True:
            errors.append(f"cuda_offload_not_authenticated:{hf_id}")
        if row.get("model_closed") is not True:
            errors.append(f"model_not_closed:{hf_id}")
        if row.get("cuda_context_released") is not True:
            errors.append(f"cuda_context_not_released:{hf_id}")
        if row.get("process_exit_confirmed") is not True:
            errors.append(f"process_exit_not_confirmed:{hf_id}")
        if row.get("process_reaped") is not True:
            errors.append(f"process_not_reaped:{hf_id}")
    for left, right in zip(lifecycle_rows, lifecycle_rows[1:], strict=False):
        try:
            if int(right["load_start_ns"]) < int(left["close_end_ns"]):
                errors.append("model_lifecycle_overlap")
                break
        except (KeyError, TypeError, ValueError):
            errors.append("model_lifecycle_interval_invalid")
            break
    return errors


def completion_errors(
    plan: Mapping[str, Any],
    attempt_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Recompute completeness without consulting parse accuracy or confidence."""

    errors: list[str] = []
    expected = [str(row["attempt_key"]) for row in plan["attempts"]]
    observed = [str(row.get("attempt_key")) for row in attempt_rows]
    if Counter(observed) != Counter(expected):
        errors.append("attempt_key_roster_mismatch")
    if any(row.get("terminal") is not True for row in attempt_rows):
        errors.append("nonterminal_attempt")
    if any(row.get("raw_durable") is not True for row in attempt_rows):
        errors.append("raw_stage_not_durable")
    if any(row.get("hidden_label_isolation_errors") for row in attempt_rows):
        errors.append("hidden_label_isolation_failure")
    errors.extend(lifecycle_errors(lifecycle_rows, plan["model_specs"]))
    return errors


def _aggregate_rows(plan: Mapping[str, Any], attempt_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Derive every report table directly from terminal attempt rows."""

    def summarize(key: str, values: Sequence[str]) -> list[JsonDict]:
        rows: list[JsonDict] = []
        for value in values:
            members = [row for row in attempt_rows if str(row.get(key)) == value]
            parse_count = sum(bool((row.get("parse") or {}).get("json_valid")) for row in members)
            schema_count = sum(
                bool((row.get("parse") or {}).get("schema_valid")) for row in members
            )
            confidences = [
                float(row["parse"]["confidence"])
                for row in members
                if isinstance(row.get("parse"), Mapping)
                and isinstance(row["parse"].get("confidence"), (int, float))
                and not isinstance(row["parse"].get("confidence"), bool)
            ]
            rows.append(
                {
                    key: value,
                    "attempt_count": len(members),
                    "terminal_count": sum(row.get("terminal") is True for row in members),
                    "json_parse_count": parse_count,
                    "schema_valid_count": schema_count,
                    "parse_rate": parse_count / len(members) if members else None,
                    "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
                    "complete": len(members) > 0
                    and all(row.get("terminal") is True for row in members),
                }
            )
        return rows

    model_ids = [str(row["hf_id"]) for row in plan["model_specs"]]
    pair_ids = [str(value) for value in plan["pair_ids"]]
    variant_ids = [str(row["prompt_variant_id"]) for row in plan["prompt_variants"]]
    groups: list[JsonDict] = []
    for hf_id in model_ids:
        for pair_id in pair_ids:
            members = [
                row
                for row in attempt_rows
                if row.get("hf_id") == hf_id and row.get("pair_id") == pair_id
            ]
            groups.append(
                {
                    "hf_id": hf_id,
                    "pair_id": pair_id,
                    "expected_prompt_variants": variant_ids,
                    "observed_prompt_variants": sorted(
                        str(row.get("prompt_variant_id")) for row in members
                    ),
                    "attempt_count": len(members),
                    "complete": len(members) == len(variant_ids)
                    and all(row.get("terminal") is True for row in members),
                }
            )
    return {
        "model_rows": summarize("hf_id", model_ids),
        "pair_rows": summarize("pair_id", pair_ids),
        "prompt_variant_rows": summarize("prompt_variant_id", variant_ids),
        "candidate_group_rows": groups,
    }


def _projection_rows(attempt_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build audit projections while preserving attempt identity in every table."""

    raw_rows: list[JsonDict] = []
    parse_rows: list[JsonDict] = []
    schema_rows: list[JsonDict] = []
    confidence_rows: list[JsonDict] = []
    rationale_rows: list[JsonDict] = []
    isolation_rows: list[JsonDict] = []
    checkpoint_rows: list[JsonDict] = []
    for row in attempt_rows:
        key = row.get("attempt_key")
        parse = row.get("parse") if isinstance(row.get("parse"), Mapping) else {}
        raw_rows.append(
            {
                "attempt_key": key,
                "raw_text": row.get("raw_text"),
                "raw_sha256": row.get("raw_sha256"),
                "prompt_token_ids": deepcopy(row.get("prompt_token_ids", [])),
                "output_token_ids": deepcopy(row.get("output_token_ids", [])),
                "call_status": row.get("call_status"),
                "failure_reason": row.get("failure_reason"),
            }
        )
        parse_rows.append(
            {
                "attempt_key": key,
                "json_valid": parse.get("json_valid"),
                "failure_reason": parse.get("failure_reason"),
                "parsed_candidate": deepcopy(parse.get("parsed_candidate")),
            }
        )
        schema_rows.append(
            {
                "attempt_key": key,
                "schema_valid": parse.get("schema_valid"),
                "failure_reason": parse.get("failure_reason"),
            }
        )
        confidence_rows.append(
            {
                "attempt_key": key,
                "confidence": parse.get("confidence"),
                "self_report_only": True,
            }
        )
        rationale_rows.append(
            {
                "attempt_key": key,
                "rationale": parse.get("rationale"),
                "self_report_only": True,
            }
        )
        isolation_rows.append(
            {
                "attempt_key": key,
                "prompt_sha256": row.get("prompt_sha256"),
                "failures": deepcopy(row.get("hidden_label_isolation_errors", [])),
                "passed": not row.get("hidden_label_isolation_errors"),
            }
        )
        checkpoint_rows.append(
            {
                "attempt_key": key,
                "raw_stage_durable": row.get("raw_durable") is True,
                "raw_sha256": row.get("raw_sha256"),
                "parse_stage_durable": row.get("terminal") is True,
                "parse_state": row.get("parse_state"),
            }
        )
    return {
        "raw_output_rows": raw_rows,
        "parse_rows": parse_rows,
        "schema_rows": schema_rows,
        "confidence_rows": confidence_rows,
        "rationale_rows": rationale_rows,
        "hidden_label_isolation_rows": isolation_rows,
        "checkpoint_rows": checkpoint_rows,
    }


def _task_runtime_receipt(
    attempt_rows: Sequence[Mapping[str, Any]], lifecycle_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Bind per-call receipts and final process teardown into one task receipt."""

    call_rows = [deepcopy(dict(row.get("runtime_receipt") or {})) for row in attempt_rows]
    content = {
        "schema_version": "carnot.exp6956.task_runtime_receipt.v1",
        "task_id": TASK_ID,
        "call_rows": call_rows,
        "model_lifecycle_rows": [deepcopy(dict(row)) for row in lifecycle_rows],
    }
    content["receipt_sha256"] = sha256_json(content)
    return content


def _reproducibility_checksum(
    plan: Mapping[str, Any],
    attempt_rows: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> str:
    """Build a timing-free checksum over the plan, candidate bytes, and bindings."""

    return sha256_json(
        {
            "random_seed": RANDOM_SEED,
            "plan_sha256": plan.get("plan_sha256"),
            "attempts": [
                {
                    "attempt_key": row.get("attempt_key"),
                    "raw_sha256": row.get("raw_sha256"),
                    "output_token_ids": row.get("output_token_ids"),
                    "call_status": row.get("call_status"),
                    "parse": row.get("parse"),
                }
                for row in attempt_rows
            ],
            "model_lifecycle_bindings": [
                {
                    "hf_id": row.get("hf_id"),
                    "model_sha256": row.get("model_sha256"),
                    "child_pid": row.get("child_pid"),
                }
                for row in lifecycle_rows
            ],
            "source_artifact_hashes": dict(source_hashes),
        }
    )


def blocked_artifact(
    run_date: str,
    preconditions_checked: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    source_artifact_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the full required schema when preflight cannot authorize inference."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": 6956,
        "run_date": run_date,
        "status": "blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 9),
        "source_artifact_hashes": dict(source_artifact_hashes or {}),
        "rows": [],
        "model_specs": [deepcopy(dict(row)) for row in MODEL_SPECS],
        "model_rows": [],
        "pair_rows": [],
        "attempt_rows": [],
        "prompt_variant_rows": [],
        "candidate_group_rows": [],
        "raw_output_rows": [],
        "parse_rows": [],
        "schema_rows": [],
        "confidence_rows": [],
        "rationale_rows": [],
        "diversity_rows": [],
        "hidden_label_isolation_rows": [],
        "checkpoint_rows": [],
        "model_lifecycle_rows": [],
        "task_runtime_receipt": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": sha256_json(
            {
                "run_date": run_date,
                "random_seed": RANDOM_SEED,
                "preconditions": list(preconditions_checked),
            }
        ),
        "reformulation_bank_complete_score": 0,
        "gate_check_summary": gate_summary(preconditions_checked),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_three_family_reformulation_bank",
    }
    return artifact


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    lifecycle_rows: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
) -> JsonDict:
    """Reduce only durable rows into the terminal completeness artifact."""

    attempts = [deepcopy(dict(row)) for row in checkpoint.get("attempt_rows", [])]
    lifecycles = [deepcopy(dict(row)) for row in lifecycle_rows]
    aggregates = _aggregate_rows(plan, attempts)
    projections = _projection_rows(attempts)
    errors = completion_errors(plan, attempts, lifecycles)
    complete = not errors
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": 6956,
        "run_date": run_date,
        "status": "complete" if complete else "partial",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 9),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "rows": attempts,
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        **aggregates,
        "attempt_rows": attempts,
        **projections,
        "diversity_rows": diversity_rows(attempts),
        "model_lifecycle_rows": lifecycles,
        "task_runtime_receipt": _task_runtime_receipt(attempts, lifecycles),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _reproducibility_checksum(
            plan, attempts, lifecycles, source_artifact_hashes
        ),
        "reformulation_bank_complete_score": int(complete),
        "gate_check_summary": [
            {"failed_check": error, "expected_value": "pass", "observed_value": "fail"}
            for error in errors
        ],
        "verifier_is_oracle": False,
        "verdict_class": "null" if complete else "partial",
        "honest_verdict": "complete_reformulation_proposal_bank_frozen"
        if complete
        else "partial_reformulation_proposal_bank",
    }
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Independently recompute required fields, aggregates, and completion score."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
        return errors
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles):
        errors.append("field_principles_incomplete")
    attempts = artifact.get("attempt_rows", [])
    model_specs = artifact.get("model_specs", [])
    pair_ids = [str(row.get("pair_id")) for row in artifact.get("pair_rows", [])]
    variant_ids = [
        str(row.get("prompt_variant_id")) for row in artifact.get("prompt_variant_rows", [])
    ]
    plan = {
        "attempts": [
            {
                "attempt_key": row.get("attempt_key"),
            }
            for row in attempts
        ],
        "model_specs": model_specs,
        "pair_ids": pair_ids,
        "prompt_variants": [{"prompt_variant_id": value} for value in variant_ids],
    }
    expected_aggregates = _aggregate_rows(plan, attempts)
    for field in ("model_rows", "pair_rows", "prompt_variant_rows", "candidate_group_rows"):
        if artifact.get(field) != expected_aggregates[field]:
            errors.append(f"{field}_mismatch")
    score_errors = completion_errors(plan, attempts, artifact.get("model_lifecycle_rows", []))
    expected_score = int(not score_errors)
    if artifact.get("reformulation_bank_complete_score") != expected_score:
        errors.append("completion_score_mismatch")
    receipt = artifact.get("task_runtime_receipt", {})
    if isinstance(receipt, Mapping) and receipt:
        receipt_payload = dict(receipt)
        stored = receipt_payload.pop("receipt_sha256", None)
        if stored != sha256_json(receipt_payload):
            errors.append("task_runtime_receipt_hash_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    verdict = str(artifact.get("honest_verdict", ""))
    verdict_class = artifact.get("verdict_class")
    if verdict_class == "null" and not verdict.startswith("complete_"):
        errors.append("honest_verdict_prefix_mismatch")
    if verdict_class == "blocked" and not verdict.startswith("blocked_"):
        errors.append("honest_verdict_prefix_mismatch")
    if verdict_class == "partial" and not verdict.startswith("partial_"):
        errors.append("honest_verdict_prefix_mismatch")
    return errors


def _read_json(path: Path) -> JsonDict:
    """Read one JSON object or raise a stable preflight error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BankError(f"json_unavailable:{path}") from exc
    if not isinstance(value, dict):
        raise BankError(f"json_object_required:{path}")
    return value


def _resolve_models() -> list[JsonDict]:  # pragma: no cover - reads the multi-gigabyte host cache.
    """Resolve and hash exactly the three mandated local GGUF files."""

    rows: list[JsonDict] = []
    for spec in MODEL_SPECS:
        resolved = resolve_cached_gguf(str(spec["hf_id"]), str(spec["quantization"]))
        path = Path(resolved).resolve() if resolved else None
        rows.append(
            {
                **deepcopy(spec),
                "model_path": str(path) if path and path.is_file() else None,
                "model_sha256": sha256_file(path) if path and path.is_file() else None,
                "size_bytes": path.stat().st_size if path and path.is_file() else 0,
                "cache_state": "hit" if path and path.is_file() else "miss",
            }
        )
    return rows


def _vocab_probe(model: Mapping[str, Any]) -> JsonDict:  # pragma: no cover - host GGUF boundary.
    """Use only the tokenizer embedded in the resolved GGUF file."""

    try:
        from llama_cpp import Llama

        tokenizer = Llama(model_path=str(model["model_path"]), vocab_only=True, verbose=False)
        probe_ids = [
            int(value)
            for value in tokenizer.tokenize(
                b'{"mapping":{"schema_version":"carnot.reformulation_mapping.v1"}}',
                add_bos=False,
                special=False,
            )
        ]
        vocabulary_size = int(tokenizer._model.n_vocab())
        tokenizer.close()
        del tokenizer
        return {
            "hf_id": model["hf_id"],
            "source": "native_embedded_gguf_llama_cpp_vocab_only",
            "loadable": bool(probe_ids),
            "probe_token_ids": probe_ids,
            "vocabulary_size": vocabulary_size,
            "binding_sha256": sha256_json(
                {
                    "model_sha256": model["model_sha256"],
                    "probe_token_ids": probe_ids,
                    "vocabulary_size": vocabulary_size,
                }
            ),
            "used_hf_autotokenizer": False,
        }
    except Exception as exc:
        return {
            "hf_id": model["hf_id"],
            "source": "native_embedded_gguf_llama_cpp_vocab_only",
            "loadable": False,
            "probe_token_ids": [],
            "vocabulary_size": None,
            "binding_sha256": None,
            "used_hf_autotokenizer": False,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _gpu_rows() -> list[JsonDict]:  # pragma: no cover - host hardware boundary.
    """Return stable UUID evidence for the two local CUDA devices."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows: list[JsonDict] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(float(parts[3])),
                    "memory_free_mb": int(float(parts[4])),
                }
            )
        except ValueError:
            continue
    return rows


def _llama_cuda_status() -> JsonDict:  # pragma: no cover - installed runtime boundary.
    """Confirm the selected llama.cpp binding was compiled for CUDA offload."""

    try:
        import llama_cpp
        from llama_cpp import llama_cpp as backend

        return {
            "importable": True,
            "version": getattr(llama_cpp, "__version__", "unknown"),
            "module_path": str(Path(llama_cpp.__file__).resolve()),
            "supports_gpu_offload": bool(backend.llama_supports_gpu_offload()),
        }
    except Exception as exc:
        return {
            "importable": False,
            "version": None,
            "module_path": None,
            "supports_gpu_offload": False,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _atomic_path_probe(path: Path) -> JsonDict:  # pragma: no cover - host filesystem boundary.
    """Prove same-directory temporary write and rename work at the checkpoint path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    probe = path.parent / f".{path.name}.atomic-probe-{os.getpid()}"
    try:
        write_json_atomic(probe, {"probe": True})
        observed = json.loads(probe.read_text(encoding="utf-8")) == {"probe": True}
        return {"path": str(path), "atomic_write_readback": observed}
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "atomic_write_readback": False, "error": repr(exc)}
    finally:
        probe.unlink(missing_ok=True)


def _source_hashes() -> JsonDict:  # pragma: no cover - repository file boundary.
    """Bind the terminal artifact to its exact local source inputs."""

    paths = {
        "fixture_artifact": FIXTURE_ARTIFACT_PATH,
        "fixture_checkpoint": FIXTURE_CHECKPOINT_PATH,
        "runtime_qualification": RUNTIME_QUALIFICATION_PATH,
        "verification_spec": REPO_ROOT / "openspec/capabilities/verification/spec.md",
        "module": Path(__file__),
        "test": REPO_ROOT / "tests/python/test_experiment_6956_three_family_reformulation_bank.py",
        "wrapper": REPO_ROOT
        / "scripts/experiments/experiment_6956_three_family_reformulation_bank.py",
        "runtime_helper": REPO_ROOT / "python/carnot/task_runtime_receipts.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def collect_preconditions(  # pragma: no cover - live host integration boundary.
    *, output_path: Path = RESULT_PATH, checkpoint_path: Path = CHECKPOINT_PATH
) -> JsonDict:
    """Collect every upstream, cache, tokenizer, CUDA, disk, and atomic-write gate."""

    checks: list[JsonDict] = []
    fixture: JsonDict = {}
    public_pairs: list[JsonDict] = []
    try:
        fixture = _read_json(FIXTURE_ARTIFACT_PATH)
        fixture_score = fixture.get("reformulation_fixture_ready_score")
    except BankError as exc:
        fixture_score = f"{type(exc).__name__}:{exc}"
    checks.append(gate_check("reformulation_fixture_ready_score", 1, fixture_score))
    checks.append(gate_check("fixture_checkpoint_exists", True, FIXTURE_CHECKPOINT_PATH.is_file()))
    if fixture_score == 1:
        try:
            public_pairs = load_public_pairs(fixture)
            pair_observed: Any = len(public_pairs)
        except BankError as exc:
            pair_observed = f"{type(exc).__name__}:{exc}"
        checks.append(gate_check("frozen_held_out_pair_count", 18, pair_observed))
    else:
        checks.append(gate_check("frozen_held_out_pair_count", 18, 0))

    models = _resolve_models()
    checks.append(
        gate_check("three_cached_gguf_files", 3, sum(row["cache_state"] == "hit" for row in models))
    )
    tokenizer_rows = [_vocab_probe(row) for row in models if row["cache_state"] == "hit"]
    checks.append(
        gate_check(
            "three_vocab_only_probes",
            3,
            sum(row.get("loadable") is True for row in tokenizer_rows),
        )
    )
    llama_status = _llama_cuda_status()
    checks.append(gate_check("llama_cpp_cuda_offload", True, llama_status["supports_gpu_offload"]))
    gpus = _gpu_rows()
    authenticated_gpus = [
        row
        for row in gpus
        if str(row.get("uuid", "")).startswith("GPU-")
        and "RTX 3090" in str(row.get("name", ""))
        and int(row.get("memory_total_mb", 0)) >= 23_000
    ][:2]
    checks.append(gate_check("two_authenticated_cuda_devices", 2, len(authenticated_gpus)))

    qualification: JsonDict = {}
    try:
        qualification = _read_json(RUNTIME_QUALIFICATION_PATH)
        qualification_score = qualification.get("sota_runtime_receipt_ready_score")
    except BankError as exc:
        qualification_score = f"{type(exc).__name__}:{exc}"
    checks.append(gate_check("authenticated_cuda_runtime_receipt", 1, qualification_score))
    qualified_hashes = {
        str(row.get("hf_id")): row.get("model_sha256")
        for row in qualification.get("model_file_rows", [])
        if isinstance(row, Mapping)
    }
    current_hashes = {str(row["hf_id"]): row.get("model_sha256") for row in models}
    checks.append(
        gate_check(
            "qualified_model_hashes_current",
            current_hashes,
            {hf_id: qualified_hashes.get(hf_id) for hf_id in current_hashes},
        )
    )
    disk = shutil.disk_usage(output_path.parent)
    checks.append(gate_check("disk_free_bytes_at_least", True, disk.free >= MIN_DISK_FREE_BYTES))
    atomic = _atomic_path_probe(checkpoint_path)
    checks.append(gate_check("atomic_per_attempt_recovery", True, atomic["atomic_write_readback"]))
    output_atomic = _atomic_path_probe(output_path)
    checks.append(gate_check("atomic_result_write", True, output_atomic["atomic_write_readback"]))
    return {
        "checks": checks,
        "fixture": fixture,
        "public_pairs": public_pairs,
        "model_specs": models,
        "tokenizer_rows": tokenizer_rows,
        "tokenizer_bindings": {
            str(row["hf_id"]): str(row["binding_sha256"]) for row in tokenizer_rows
        },
        "model_hashes": current_hashes,
        "gpus": authenticated_gpus,
        "llama_cpp": llama_status,
        "runtime_qualification": qualification,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "atomic_checkpoint_probe": atomic,
        "atomic_result_probe": output_atomic,
        "source_artifact_hashes": _source_hashes(),
    }


def _emit_worker_event(event: Mapping[str, Any]) -> None:  # pragma: no cover - child protocol.
    """Write one newline-delimited JSON event without mixing it with llama.cpp logs."""

    sys.stdout.write(json.dumps(dict(event), sort_keys=True) + "\n")
    sys.stdout.flush()


def worker_main(model_path: str) -> int:  # pragma: no cover - exercised by the live command.
    """Hold one GGUF in a child process and answer bounded parent commands."""

    from llama_cpp import Llama

    model: Any = None
    _emit_worker_event({"event": "started", "pid": os.getpid()})
    for line in sys.stdin:
        try:
            command = json.loads(line)
            action = command.get("action")
            if action == "load":
                model = Llama(
                    model_path=model_path,
                    n_ctx=CONTEXT_SIZE,
                    n_gpu_layers=-1,
                    tensor_split=[0.5, 0.5],
                    seed=RANDOM_SEED,
                    verbose=False,
                )
                _emit_worker_event({"event": "loaded", "pid": os.getpid()})
            elif action == "generate":
                if model is None:
                    raise RuntimeError("model_not_loaded")
                prompt = str(command["prompt"])
                response = model.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "Return exactly the requested JSON object and no markdown.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(DECODING_SETTINGS["temperature"]),
                    top_p=float(DECODING_SETTINGS["top_p"]),
                    repeat_penalty=float(DECODING_SETTINGS["repeat_penalty"]),
                    max_tokens=int(DECODING_SETTINGS["max_tokens"]),
                    seed=int(command["seed"]),
                )
                raw_text = str(response["choices"][0]["message"]["content"] or "")
                _emit_worker_event(
                    {
                        "event": "generated",
                        "raw_text": raw_text,
                        "output_token_ids": [
                            int(value)
                            for value in model.tokenize(
                                raw_text.encode("utf-8"), add_bos=False, special=True
                            )
                        ],
                        "prompt_token_ids": [
                            int(value)
                            for value in model.tokenize(
                                prompt.encode("utf-8"), add_bos=False, special=True
                            )
                        ],
                        "usage": response.get("usage", {}),
                    }
                )
            elif action == "close":
                if model is not None:
                    model.close()
                    model = None
                gc.collect()
                _emit_worker_event({"event": "closed", "pid": os.getpid()})
                return 0
            else:
                raise RuntimeError(f"unknown_worker_action:{action}")
        except Exception as exc:
            _emit_worker_event(
                {"event": "error", "error": f"{type(exc).__name__}:{exc}", "pid": os.getpid()}
            )
    return 1


def _send_worker(
    process: subprocess.Popen[str], payload: Mapping[str, Any]
) -> None:  # pragma: no cover
    """Send one command to the owned child or fail if its input pipe closed."""

    if process.stdin is None:
        raise BankError("worker_stdin_unavailable")
    process.stdin.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    process.stdin.flush()


def _wait_worker(
    process: subprocess.Popen[str], expected: str, timeout_s: float
) -> JsonDict:  # pragma: no cover
    """Wait for one child event with a true wall-clock timeout."""

    if process.stdout is None:
        raise BankError("worker_stdout_unavailable")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([process.stdout], [], [], min(1.0, remaining))
        if not readable:
            if process.poll() is not None:
                raise BankError(f"worker_exited:{process.returncode}")
            continue
        line = process.stdout.readline()
        if not line:
            raise BankError(f"worker_stream_closed:{process.poll()}")
        event = json.loads(line)
        if event.get("event") == "error":
            raise BankError(str(event.get("error")))
        if event.get("event") == expected:
            return event
    raise TimeoutError(f"worker_timeout:{expected}")


def _pid_gpu_samples(
    pid: int, gpus: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:  # pragma: no cover
    """Join the owned child PID to CUDA UUID and resident-memory evidence."""

    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        result = subprocess.CompletedProcess(command, 1, "", "")
    residency: dict[str, int] = defaultdict(int)
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                residency[parts[1]] += int(float(parts[2]))
            except ValueError:
                continue
    return [
        {
            "pid": pid,
            "gpu_uuid": str(gpu["uuid"]),
            "pid_memory_mb": residency[str(gpu["uuid"])],
            "sampled_monotonic_ns": time.monotonic_ns(),
        }
        for gpu in gpus
    ]


def _terminate_and_reap(process: subprocess.Popen[str]) -> tuple[bool, bool]:  # pragma: no cover
    """Narrowly terminate only the child created by this task, then reap it."""

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    return process.poll() is not None, process.returncode is not None


def _run_model_attempts(  # pragma: no cover - live GGUF/CUDA integration boundary.
    model: Mapping[str, Any],
    model_plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    model_hashes: Mapping[str, str],
    tokenizer_bindings: Mapping[str, str],
    gpus: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, JsonDict]:
    """Load one child model, acquire its missing attempts, and fully tear it down."""

    command = [sys.executable, "-m", __name__, "--worker", "--model-path", str(model["model_path"])]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    stderr_path = checkpoint_path.parent / f"experiment_6956_{model['family']}.stderr.log"
    load_start = time.monotonic_ns()
    process: subprocess.Popen[str] | None = None
    load_end = load_start
    close_start = load_start
    close_end = load_start
    loaded = False
    closed = False
    exit_confirmed = False
    reaped = False
    samples: list[JsonDict] = []
    runner_error: str | None = None
    checkpoint_payload = deepcopy(dict(checkpoint))
    with stderr_path.open("a", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            _wait_worker(process, "started", 15.0)
            _send_worker(process, {"action": "load"})
            _wait_worker(process, "loaded", MODEL_LOAD_TIMEOUT_S)
            load_end = time.monotonic_ns()
            loaded = True
            samples = _pid_gpu_samples(process.pid, gpus)
            offload_authenticated = len(samples) == len(gpus) and all(
                int(row["pid_memory_mb"]) > 0 for row in samples
            )
            if not offload_authenticated:
                raise BankError("owned_cuda_offload_not_observed")

            def infer(attempt: Mapping[str, Any]) -> JsonDict:
                call_start_ns = time.monotonic_ns()
                wall_start = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                _send_worker(
                    process,
                    {
                        "action": "generate",
                        "prompt": attempt["prompt"],
                        "seed": attempt["random_seed"],
                    },
                )
                event = _wait_worker(process, "generated", MODEL_CALL_TIMEOUT_S)
                call_end_ns = time.monotonic_ns()
                return {
                    "raw_text": event.get("raw_text", ""),
                    "output_token_ids": event.get("output_token_ids", []),
                    "prompt_token_ids": event.get("prompt_token_ids", []),
                    "latency_s": (call_end_ns - call_start_ns) / 1_000_000_000,
                    "call_status": "complete",
                    "failure_reason": None,
                    "runtime_receipt": {
                        "child_pid": process.pid,
                        "parent_pid": os.getpid(),
                        "model_sha256": model["model_sha256"],
                        "model_path": model["model_path"],
                        "gpu_uuids": [str(gpu["uuid"]) for gpu in gpus],
                        "cuda_samples": deepcopy(samples),
                        "monotonic_start_ns": call_start_ns,
                        "monotonic_end_ns": call_end_ns,
                        "wall_clock_start": wall_start,
                        "wall_clock_end": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        "usage": deepcopy(event.get("usage", {})),
                    },
                }

            durable = deepcopy(checkpoint_payload)
            write_checkpoint(checkpoint_path, durable)
            checkpoint_payload = resume_attempts(
                model_plan,
                checkpoint_path=checkpoint_path,
                model_hashes=model_hashes,
                tokenizer_bindings=tokenizer_bindings,
                infer=infer,
            )
            close_start = time.monotonic_ns()
            _send_worker(process, {"action": "close"})
            _wait_worker(process, "closed", 30.0)
            process.wait(timeout=30.0)
            close_end = time.monotonic_ns()
            closed = True
            exit_confirmed = process.returncode == 0
            reaped = process.poll() is not None
        except Exception as exc:
            runner_error = f"{type(exc).__name__}:{exc}"
        finally:
            if process is not None:
                terminated, waited = _terminate_and_reap(process)
                exit_confirmed = exit_confirmed or terminated
                reaped = reaped or waited
                close_end = max(close_end, time.monotonic_ns())
    post_samples = _pid_gpu_samples(process.pid, gpus) if process is not None else []
    cuda_released = bool(post_samples) and all(
        int(row["pid_memory_mb"]) == 0 for row in post_samples
    )
    lifecycle = {
        "hf_id": model["hf_id"],
        "model_family": model["family"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "child_pid": process.pid if process is not None else None,
        "load_start_ns": load_start,
        "load_end_ns": load_end,
        "close_start_ns": close_start,
        "close_end_ns": close_end,
        "model_loaded": loaded,
        "cuda_offload_authenticated": loaded
        and len(samples) == len(gpus)
        and all(int(row["pid_memory_mb"]) > 0 for row in samples),
        "load_gpu_samples": samples,
        "model_closed": closed,
        "cuda_context_released": cuda_released,
        "post_teardown_gpu_samples": post_samples,
        "process_exit_confirmed": exit_confirmed,
        "process_reaped": reaped,
        "returncode": process.returncode if process is not None else None,
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "runner_error": runner_error,
    }
    lifecycles = [
        deepcopy(dict(row))
        for row in checkpoint_payload.get("model_lifecycle_rows", [])
        if row.get("hf_id") != model["hf_id"]
    ]
    lifecycles.append(lifecycle)
    checkpoint_payload["model_lifecycle_rows"] = lifecycles
    write_checkpoint(checkpoint_path, checkpoint_payload)
    return checkpoint_payload, lifecycle


def run_live_acquisition(  # pragma: no cover - live GGUF/CUDA integration boundary.
    plan: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    model_hashes: Mapping[str, str],
    tokenizer_bindings: Mapping[str, str],
    model_specs: Sequence[Mapping[str, Any]],
    gpus: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Resume durable rows, then load each still-needed model exactly one at a time."""

    if checkpoint_path.is_file():
        checkpoint = load_checkpoint(
            checkpoint_path,
            plan=plan,
            model_hashes=model_hashes,
            tokenizer_bindings=tokenizer_bindings,
        )
    else:
        checkpoint = new_checkpoint(
            plan, model_hashes=model_hashes, tokenizer_bindings=tokenizer_bindings
        )
        write_checkpoint(checkpoint_path, checkpoint)

    # Raw rows left pending by an interruption are parsed before any model load.
    for row in list(checkpoint["attempt_rows"]):
        if row.get("terminal") is not True:
            checkpoint = finalize_pending_attempt(checkpoint, str(row["attempt_key"]))
            write_checkpoint(checkpoint_path, checkpoint)

    terminal_keys = {
        str(row["attempt_key"]) for row in checkpoint["attempt_rows"] if row.get("terminal") is True
    }
    for model in model_specs:
        model_attempts = [row for row in plan["attempts"] if row["hf_id"] == model["hf_id"]]
        if all(str(row["attempt_key"]) in terminal_keys for row in model_attempts):
            continue
        model_plan = deepcopy(dict(plan))
        model_plan["attempts"] = model_attempts
        # Keep the original plan hash because the checkpoint is bound to the full 162-key roster.
        model_plan["plan_sha256"] = plan["plan_sha256"]
        checkpoint, _lifecycle = _run_model_attempts(
            model,
            model_plan,
            checkpoint,
            checkpoint_path=checkpoint_path,
            model_hashes=model_hashes,
            tokenizer_bindings=tokenizer_bindings,
            gpus=gpus,
        )
        terminal_keys = {
            str(row["attempt_key"])
            for row in checkpoint["attempt_rows"]
            if row.get("terminal") is True
        }
    return checkpoint


def run(  # pragma: no cover - command integration is intentionally live and expensive.
    *,
    run_date: str,
    output_path: Path = RESULT_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> JsonDict:
    """Run preflight, acquire every missing proposal, validate, and atomically write."""

    started = time.perf_counter()
    preflight = collect_preconditions(output_path=output_path, checkpoint_path=checkpoint_path)
    checks = preflight["checks"]
    if any(row.get("passed") is not True for row in checks):
        artifact = blocked_artifact(
            run_date,
            checks,
            duration_s=time.perf_counter() - started,
            source_artifact_hashes=preflight["source_artifact_hashes"],
        )
        write_json_atomic(output_path, artifact)
        return artifact
    plan = freeze_plan(preflight["public_pairs"], model_specs=preflight["model_specs"])
    if len(plan["attempts"]) != EXPECTED_ATTEMPT_COUNT:
        raise BankError(f"headline_attempt_budget:{len(plan['attempts'])}")
    checkpoint = run_live_acquisition(
        plan,
        checkpoint_path=checkpoint_path,
        model_hashes=preflight["model_hashes"],
        tokenizer_bindings=preflight["tokenizer_bindings"],
        model_specs=preflight["model_specs"],
        gpus=preflight["gpus"],
    )
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        preconditions_checked=checks,
        plan=plan,
        checkpoint=checkpoint,
        model_specs=preflight["model_specs"],
        lifecycle_rows=checkpoint.get("model_lifecycle_rows", []),
        source_artifact_hashes=preflight["source_artifact_hashes"],
    )
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        artifact["status"] = "partial"
        artifact["reformulation_bank_complete_score"] = 0
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = "partial_reformulation_proposal_bank"
        artifact["gate_check_summary"].extend(
            {
                "failed_check": error,
                "expected_value": "pass",
                "observed_value": "fail",
            }
            for error in validation_errors
        )
    write_json_atomic(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Dispatch the internal model worker or the required dated experiment command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-path")
    args = parser.parse_args(argv)
    if args.worker:
        if not args.model_path:
            parser.error("--worker requires --model-path")
        return worker_main(args.model_path)
    if not args.date:
        parser.error("--date is required")
    artifact = run(
        run_date=args.date,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "reformulation_bank_complete_score": artifact["reformulation_bank_complete_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module worker entry point.
    raise SystemExit(main())
