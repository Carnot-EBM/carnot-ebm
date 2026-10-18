"""Build the delayed-constraint three-schedule candidate bank.

The module acquires candidates only. It keeps exact labels and exact engines
outside the generation process. It stores raw bytes and token-level energy
statistics before a syntax-only parser runs.

Spec refs: REQ-INF-6975 and SCENARIO-INF-6975-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np

from carnot.experiment_6966_gguf_load_envelope_canary import (
    gpu_inventory,
    llama_cpp_probe,
)
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.task_runtime_receipts import sha256_file, write_json_atomic


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_6975_delayed_constraint_candidate_bank"
SCHEMA = "carnot.experiment_6975.delayed_constraint_candidate_bank.v1"
RUN_DATE = "20260904"
RANDOM_SEED = 6_975_202_609_04
INFERENCE_SUBSTRATE = "live_local_llama_cpp_three_family_delayed_constraint_generation"
RESULT_PATH = REPO_ROOT / "results/experiment_6975_delayed_constraint_candidate_bank.json"
CHECKPOINT_PATH = (
    REPO_ROOT
    / "results/checkpoints/experiment_6975_delayed_constraint_candidate_bank.checkpoint.json"
)
EXP6973_PATH = REPO_ROOT / "results/experiment_6973_lease_aware_gguf_runtime.json"
EXP6974_PATH = REPO_ROOT / "results/experiment_6974_claim_provenance_duration_lint.json"
EXP6967_PATH = REPO_ROOT / "results/experiment_6967_certified_error_headroom_fixture.json"
EXP6969_PATH = REPO_ROOT / "results/experiment_6969_error_structured_prompt_bank.json"
EXP5923_PATH = REPO_ROOT / "results/experiment_5923_sota_schema_supported_constraintir_ab.json"
EXPECTED_EXP6967_SHA256 = "sha256:1685ad1bff1b82aae3a17f80d341e0593d99879809bb9afb3268060100e54fee"
EXPECTED_ATTEMPT_COUNT = 108
PREFERRED_QUANT = "Q4_K_M"
CONTEXT_SIZE = 8_192
MODEL_LOAD_TIMEOUT_S = 300.0
PAIR_BLOCK_TIMEOUT_S = 900.0

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
FORMULATION_FAMILIES = (
    "bounded_integer_linear",
    "boolean_cardinality",
    "bounded_piecewise_linear",
)
SELECTED_SPLITS = ("calibration", "heldout")
TRIGGER_TEXT = "CONSTRAINTIR_BEGIN"

DECODING_SETTINGS: JsonDict = {
    "temperature": 0.35,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.05,
    "completion_token_cap": 128,
    "planning_token_cap": 48,
    "certificate_token_cap": 80,
}

CONSTRAINT_IR_SCHEMA: JsonDict = {
    "type": "object",
    "properties": {
        "schema_version": {
            "type": "string",
            "const": "carnot.constraint_ir.mapping.v1",
        },
        "variable_map": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "scale": {"type": "string"},
                    "offset": {"type": "string"},
                },
                "required": ["source", "target", "scale", "offset"],
                "additionalProperties": False,
            },
        },
        "objective_map": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["same", "reversed"]},
                "scale": {"type": "string"},
                "offset": {"type": "string"},
            },
            "required": ["direction", "scale", "offset"],
            "additionalProperties": False,
        },
    },
    "required": ["schema_version", "variable_map", "objective_map"],
    "additionalProperties": False,
}

SCHEDULE_ROWS: list[JsonDict] = [
    {
        "schedule_id": "direct",
        "schedule_text": (
            "Constrain generation from the first emitted token. Emit only one "
            "ConstraintIR certificate JSON object."
        ),
        "phase_order": ["direct_certificate"],
        "constraint_start": "token_zero",
        "stop_rules": ["embedded_eog", "total_token_cap_128"],
    },
    {
        "schedule_id": "trigger_switched",
        "schedule_text": (
            "Reason freely about the public formulations. Emit the exact line "
            f"{TRIGGER_TEXT} when the mapping is unambiguous. Constrain only the "
            "certificate tail after that trigger."
        ),
        "phase_order": ["free_reasoning", "certificate_tail"],
        "constraint_start": f"after_exact_trigger:{TRIGGER_TEXT}",
        "stop_rules": [
            f"free_phase_exact_trigger:{TRIGGER_TEXT}",
            "embedded_eog",
            "total_token_cap_128",
        ],
    },
    {
        "schedule_id": "draft_conditioned",
        "schedule_text": (
            "Generate one unconstrained semantic draft first. Then condition a "
            "new constrained ConstraintIR certificate tail on that same draft."
        ),
        "phase_order": ["unconstrained_draft", "certificate_tail"],
        "constraint_start": "after_own_draft",
        "stop_rules": ["embedded_eog", "total_token_cap_128"],
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "live_duration_s",
    "source_artifact_hashes",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes",
    "gpu_runtime_rows",
    "schedule_rows",
    "schedule_hashes",
    "selected_pair_rows",
    "split_hash",
    "rows",
    "per_attempt_rows",
    "raw_output_rows",
    "parser_diagnostic_rows",
    "energy_trace_rows",
    "token_span_rows",
    "checkpoint_rows",
    "teardown_rows",
    "split_isolation_rows",
    "expected_attempt_count",
    "observed_attempt_count",
    "candidate_bank_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned schema lets cold validation reject incompatible evidence.",
    "experiment_id": "A stable task identity prevents evidence from another run entering the bank.",
    "run_date": "The fixed execution date makes protocol changes visible.",
    "field_principles": "A reason for each field makes the scientific contract reviewable.",
    "preconditions_checked": "Bare gates stop generation when upstream or host evidence drifts.",
    "inference_substrate": "The substrate states that local llama.cpp CUDA generation ran.",
    "duration_s": "Total wall time exposes incomplete or synthetic acquisition.",
    "live_duration_s": "Generation time separates live model work from setup and parsing.",
    "source_artifact_hashes": "Hashes bind the bank to its exact fixture and runtime evidence.",
    "MODEL_SPECS": "Exact declarations exclude legacy models from headline rows.",
    "models_used": "The observed model roster exposes missing family execution.",
    "model_file_hashes": "File hashes prevent silent model-byte substitution.",
    "gpu_runtime_rows": "PID-linked device rows distinguish CUDA use from CPU fallback.",
    "schedule_rows": "Frozen mechanism text prevents a schedule from changing during acquisition.",
    "schedule_hashes": "Content hashes make schedule drift mechanically detectable.",
    "selected_pair_rows": "Public pair rows prove balanced selection without carrying labels.",
    "split_hash": "One digest binds pair IDs, split assignments, and order.",
    "rows": "Compact terminal rows expose the complete attempt denominator.",
    "per_attempt_rows": "Full attempt evidence lets validators ignore incorrect aggregates.",
    "raw_output_rows": "Exact raw phases preserve malformed and truncated model output.",
    "parser_diagnostic_rows": "Syntax outcomes remain separate from semantic correctness.",
    "energy_trace_rows": "Adjacent-step scalars support later energy recomputation.",
    "token_span_rows": "Byte offsets bind each scalar row to emitted raw bytes.",
    "checkpoint_rows": "Ordered stages prove raw evidence became durable before parsing.",
    "teardown_rows": "Process exit rows prove one model ended before the next started.",
    "split_isolation_rows": "Isolation checks prevent label, future-row, and cross-arm leakage.",
    "expected_attempt_count": "The preregistered denominator prevents selective omission.",
    "observed_attempt_count": "The measured denominator exposes missing or duplicate attempts.",
    "candidate_bank_complete_score": "One requires all 108 terminal rows and sufficient traces.",
    "random_seed": "A frozen seed makes every stochastic attempt addressable.",
    "reproducibility_checksum": "A timing-free digest detects drift in plans and raw bytes.",
    "gate_check_summary": "Expected and observed values make every blocked gate actionable.",
    "verifier_is_oracle": "False states that this acquisition does not decide correctness.",
    "verdict_class": "A closed class prevents success text from hiding a partial run.",
    "honest_verdict": "A class-consistent prefix gives automation a stable terminal state.",
}

_FORBIDDEN_PROMPT_TERMS = (
    "expected_label",
    "exact label",
    "exact witness",
    "feasibility witness",
    "solver outcome",
    "solver feedback",
    "z3",
    "enumeration label",
    "chronological",
    "held-future",
    "held_future",
    "another schedule's result",
    "other schedule result",
    "candidate answer id",
    "answer menu",
)
_CONSTRAINT_IR_KEYS = {"schema_version", "variable_map", "objective_map"}
_VARIABLE_MAP_KEYS = {"source", "target", "scale", "offset"}
_OBJECTIVE_MAP_KEYS = {"direction", "scale", "offset"}


class CandidateBankError(RuntimeError):
    """Name a fail-closed candidate-bank contract violation."""


def canonical_json(value: Any) -> str:
    """Serialize one value with stable ordering for hashes and equality."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    """Return the repository spelling of a SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text without changing its bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash one JSON value after canonical serialization."""

    return sha256_text(canonical_json(value))


def resolve_model_specs(
    *,
    cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the canonical cached pair before adding the third family."""

    pair = cached_pair_func(gpu_indices=(0, 1)) or []
    pair_paths = {str(row.get("hf_id")): str(row.get("model_path") or "") for row in pair}
    rows: list[JsonDict] = []
    for model_id in REQUIRED_MODEL_IDS:
        model_path = pair_paths.get(model_id) or resolver(model_id, PREFERRED_QUANT) or ""
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": model_path,
                "gpu_indices": [0, 1],
                "headline_eligible": True,
                "preferred_quant": PREFERRED_QUANT,
                "resolution_method": (
                    "cached_sota_pair(gpu_indices=(0, 1))"
                    if model_id in pair_paths
                    else "resolve_cached_gguf exact family extension"
                ),
            }
        )
    return rows


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject model substitution, projector files, and non-dual-GPU placement."""

    errors: list[str] = []
    if [row.get("hf_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_ids_mismatch")
    for row in rows:
        model_id = str(row.get("hf_id", ""))
        model_path = str(row.get("model_path", ""))
        if not model_path:
            errors.append(f"model_path_missing:{model_id}")
        elif (
            Path(model_path).suffix.lower() != ".gguf" or "mmproj" in Path(model_path).name.lower()
        ):
            errors.append(f"model_path_not_primary_gguf:{model_id}")
        if row.get("gpu_indices") != [0, 1]:
            errors.append(f"dual_gpu_indices_missing:{model_id}")
        if row.get("headline_eligible") is not True:
            errors.append(f"headline_eligibility_missing:{model_id}")
    return errors


MODEL_SPECS = resolve_model_specs()


def gate_check(
    check: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Record one gate with exact expected and observed values."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every gate and the first failed expected-observed pair."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "checks": rows,
        "passed": failed is None,
    }


def _public_pair(row: Mapping[str, Any]) -> JsonDict:
    """Copy only fields that the fixture marks as prompt-visible."""

    return {
        "record_id": str(row["record_id"]),
        "split": str(row["split"]),
        "pair_id": str(row["pair_id"]),
        "formulation_family": str(row["formulation_family"]),
        "source_formulation": deepcopy(row["source_formulation"]),
        "target_formulation": deepcopy(row["target_formulation"]),
        "prompt_record_hash": str(row["prompt_record_hash"]),
    }


def eligible_public_pairs(fixture: Mapping[str, Any]) -> list[JsonDict]:
    """Return hash-valid public rows from only the two admitted splits."""

    result: list[JsonDict] = []
    seen: set[str] = set()
    for source in fixture.get("prompt_visible_rows", []):
        if not isinstance(source, Mapping):
            continue
        if source.get("split") not in SELECTED_SPLITS:
            continue
        if source.get("formulation_family") not in FORMULATION_FAMILIES:
            continue
        required = {
            "record_id",
            "split",
            "pair_id",
            "formulation_family",
            "source_formulation",
            "target_formulation",
            "prompt_record_hash",
        }
        if not required <= set(source):
            continue
        pair_id = str(source["pair_id"])
        if pair_id in seen:
            continue
        unhashed = {
            key: deepcopy(value) for key, value in source.items() if key != "prompt_record_hash"
        }
        if source.get("prompt_record_hash") != sha256_json(unhashed):
            continue
        if not isinstance(source.get("source_formulation"), Mapping) or not isinstance(
            source.get("target_formulation"), Mapping
        ):
            continue
        seen.add(pair_id)
        result.append(_public_pair(source))
    return result


def select_balanced_pairs(fixture: Mapping[str, Any]) -> list[JsonDict]:
    """Select two deterministic public pairs for each split-family cell."""

    eligible = eligible_public_pairs(fixture)
    selected: list[JsonDict] = []
    for split in SELECTED_SPLITS:
        for family in FORMULATION_FAMILIES:
            cell = [
                row
                for row in eligible
                if row["split"] == split and row["formulation_family"] == family
            ]
            if len(cell) < 2:
                raise CandidateBankError(
                    f"eligible_pair_cell_too_small:{split}:{family}:{len(cell)}"
                )
            ordered = sorted(
                cell,
                key=lambda row: (
                    sha256_json(
                        {
                            "selection_seed": RANDOM_SEED,
                            "split": split,
                            "family": family,
                            "pair_id": row["pair_id"],
                            "prompt_record_hash": row["prompt_record_hash"],
                        }
                    ),
                    row["pair_id"],
                ),
            )
            selected.extend(deepcopy(ordered[:2]))
    return selected


def compute_split_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind selected pair order, public bytes, and split assignments."""

    return sha256_json([deepcopy(dict(row)) for row in rows])


def audit_prompt(prompt: str) -> list[str]:
    """Reject authority, future-row, finite-answer, and cross-arm terms."""

    lowered = prompt.lower()
    return [f"forbidden_prompt_term:{term}" for term in _FORBIDDEN_PROMPT_TERMS if term in lowered]


def build_prompt(pair: Mapping[str, Any], schedule: Mapping[str, Any]) -> str:
    """Render one fixed public prompt without exact or cross-schedule evidence."""

    public_input = {
        "pair_id": pair["pair_id"],
        "source_formulation": pair["source_formulation"],
        "target_formulation": pair["target_formulation"],
    }
    prompt = (
        "Infer one semantic correspondence between these public bounded formulations. "
        "Use rational strings for scale and offset. Do not solve the optimization problems.\n"
        f"SCHEDULE={schedule['schedule_text']}\n"
        f"PUBLIC_PAIR={canonical_json(public_input)}\n"
        f"CONSTRAINT_IR_JSON_SCHEMA={canonical_json(CONSTRAINT_IR_SCHEMA)}"
    )
    failures = audit_prompt(prompt)
    if failures:
        raise CandidateBankError(f"prompt_isolation_failed:{failures}")
    return prompt


def build_split_isolation_rows(
    selected_pairs: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build replayable checks for split and prompt isolation."""

    by_split = {
        split: {str(row["pair_id"]) for row in selected_pairs if row.get("split") == split}
        for split in SELECTED_SPLITS
    }
    prompt_errors = [
        {"attempt_key": row.get("attempt_key"), "errors": audit_prompt(str(row.get("prompt", "")))}
        for row in attempts
        if audit_prompt(str(row.get("prompt", "")))
    ]
    return [
        {
            "check": "pair_ids_disjoint",
            "passed": not (by_split["calibration"] & by_split["heldout"]),
            "observed_value": sorted(by_split["calibration"] & by_split["heldout"]),
        },
        {
            "check": "held_future_excluded",
            "passed": all(row.get("split") in SELECTED_SPLITS for row in selected_pairs),
            "observed_value": sorted(
                {
                    str(row.get("split"))
                    for row in selected_pairs
                    if row.get("split") not in SELECTED_SPLITS
                }
            ),
        },
        {
            "check": "prompt_authority_terms_absent",
            "passed": not prompt_errors,
            "observed_value": prompt_errors,
        },
        {
            "check": "other_schedule_outputs_absent",
            "passed": all("prior_schedule_output" not in row for row in attempts),
            "observed_value": sum("prior_schedule_output" in row for row in attempts),
        },
    ]


def _plan_hash(plan: Mapping[str, Any]) -> str:
    """Hash the plan without its self-referential field."""

    return sha256_json({key: value for key, value in plan.items() if key != "plan_sha256"})


def freeze_plan(
    selected_pairs: Sequence[Mapping[str, Any]],
    *,
    model_specs: Sequence[Mapping[str, Any]] = MODEL_SPECS,
) -> JsonDict:
    """Freeze pair order, schedules, seeds, prompts, stops, and model identity."""

    schedule_hashes = {
        str(row["schedule_id"]): sha256_text(str(row["schedule_text"])) for row in SCHEDULE_ROWS
    }
    attempts: list[JsonDict] = []
    ordinal = 0
    for model in model_specs:
        for pair in selected_pairs:
            for schedule in SCHEDULE_ROWS:
                prompt = build_prompt(pair, schedule)
                attempt_key = "|".join(
                    (str(model["hf_id"]), str(pair["pair_id"]), str(schedule["schedule_id"]))
                )
                attempts.append(
                    {
                        "attempt_key": attempt_key,
                        "ordinal": ordinal,
                        "hf_id": str(model["hf_id"]),
                        "pair_id": str(pair["pair_id"]),
                        "split": str(pair["split"]),
                        "formulation_family": str(pair["formulation_family"]),
                        "schedule_id": str(schedule["schedule_id"]),
                        "schedule_hash": schedule_hashes[str(schedule["schedule_id"])],
                        "random_seed": RANDOM_SEED + ordinal,
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "source_formulation": deepcopy(pair["source_formulation"]),
                        "target_formulation": deepcopy(pair["target_formulation"]),
                    }
                )
                ordinal += 1
    isolation = build_split_isolation_rows(selected_pairs, attempts)
    plan: JsonDict = {
        "schema": "carnot.experiment_6975.frozen_plan.v1",
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "selected_pair_rows": [deepcopy(dict(row)) for row in selected_pairs],
        "split_hash": compute_split_hash(selected_pairs),
        "schedule_rows": deepcopy(SCHEDULE_ROWS),
        "schedule_hashes": schedule_hashes,
        "decoding_settings": deepcopy(DECODING_SETTINGS),
        "trigger_text": TRIGGER_TEXT,
        "constraint_ir_schema": deepcopy(CONSTRAINT_IR_SCHEMA),
        "attempts": attempts,
        "split_isolation_rows": isolation,
        "plan_sha256": "",
    }
    plan["plan_sha256"] = _plan_hash(plan)
    return plan


def energy_statistics(logits: np.ndarray, *, emitted_token_id: int) -> JsonDict:
    """Reduce one full-vocabulary logit vector to sufficient scalar evidence."""

    raw = np.asarray(logits, dtype=np.float32).reshape(-1)
    if emitted_token_id < 0 or emitted_token_id >= raw.size:
        raise CandidateBankError(f"emitted_token_id_out_of_range:{emitted_token_id}")
    finite = np.isfinite(raw)
    if not bool(np.any(finite)):
        raise CandidateBankError("no_finite_logits")
    values = raw.astype(np.float64)
    maximum = float(np.max(values[finite]))
    weights = np.zeros(values.shape, dtype=np.float64)
    weights[finite] = np.exp(values[finite] - maximum)
    partition = float(np.sum(weights))
    logsumexp = maximum + math.log(partition)
    probabilities = weights / partition
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    selected = float(values[emitted_token_id])
    return {
        "emitted_token_id": int(emitted_token_id),
        "selected_token_logit": selected,
        "full_vocabulary_logsumexp": logsumexp,
        "selected_token_logprob": selected - logsumexp,
        "entropy": entropy,
        "top_probability": float(np.max(probabilities)),
        "full_vocabulary_size": int(raw.size),
        "finite_logit_count": int(np.sum(finite)),
        "full_vocabulary_logits_sha256": sha256_bytes(raw.astype("<f4", copy=False).tobytes()),
    }


def parse_syntax(raw_text: str) -> JsonDict:
    """Report JSON and ConstraintIR shape without checking semantic truth."""

    if not raw_text:
        return {
            "json_valid": False,
            "object_valid": False,
            "constraintir_shape_valid": False,
            "syntax_reason": "empty_output",
        }
    try:
        value = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return {
            "json_valid": False,
            "object_valid": False,
            "constraintir_shape_valid": False,
            "syntax_reason": "malformed_json",
        }
    if not isinstance(value, Mapping):
        return {
            "json_valid": True,
            "object_valid": False,
            "constraintir_shape_valid": False,
            "syntax_reason": "json_object_required",
        }
    reason: str | None = None
    if set(value) != _CONSTRAINT_IR_KEYS:
        reason = "constraintir_keys"
    elif value.get("schema_version") != "carnot.constraint_ir.mapping.v1":
        reason = "constraintir_schema_version"
    elif not isinstance(value.get("variable_map"), list) or any(
        not isinstance(row, Mapping) or set(row) != _VARIABLE_MAP_KEYS
        for row in value.get("variable_map", [])
    ):
        reason = "variable_map_shape"
    elif (
        not isinstance(value.get("objective_map"), Mapping)
        or set(value.get("objective_map", {})) != _OBJECTIVE_MAP_KEYS
    ):
        reason = "objective_map_shape"
    else:
        objective = value["objective_map"]
        if objective.get("direction") not in {"same", "reversed"}:
            reason = "objective_direction_syntax"
        elif any(
            not isinstance(row.get(field), str)
            for row in value["variable_map"]
            for field in _VARIABLE_MAP_KEYS
        ) or any(not isinstance(objective.get(field), str) for field in ("scale", "offset")):
            reason = "string_field_syntax"
    return {
        "json_valid": True,
        "object_valid": True,
        "constraintir_shape_valid": reason is None,
        "syntax_reason": reason,
    }


def _checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    """Hash a checkpoint without its self-referential digest."""

    return sha256_json(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )


def new_checkpoint(
    plan: Mapping[str, Any],
    *,
    model_file_hashes: Mapping[str, Any],
) -> JsonDict:
    """Create the raw-first state machine bound to every immutable input."""

    checkpoint: JsonDict = {
        "schema": "carnot.experiment_6975.checkpoint.v1",
        "plan": deepcopy(dict(plan)),
        "plan_sha256": str(plan["plan_sha256"]),
        "model_file_hashes": deepcopy(dict(model_file_hashes)),
        "attempt_rows": [],
        "gpu_runtime_rows": [],
        "teardown_rows": [],
        "checkpoint_rows": [
            {
                "sequence": 1,
                "stage": "plan_frozen",
                "plan_sha256": str(plan["plan_sha256"]),
                "prompt_count": len(plan.get("attempts", [])),
            }
        ],
        "checkpoint_sha256": "",
    }
    checkpoint["checkpoint_sha256"] = _checkpoint_hash(checkpoint)
    return checkpoint


def write_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    """Publish one complete checkpoint through atomic replacement."""

    document = deepcopy(dict(checkpoint))
    document["checkpoint_sha256"] = _checkpoint_hash(document)
    write_json_atomic(path, document)


def load_checkpoint(
    path: Path,
    *,
    plan: Mapping[str, Any],
    model_file_hashes: Mapping[str, Any],
) -> JsonDict:
    """Load a checkpoint and reject corruption or immutable-input drift."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateBankError(f"checkpoint_unreadable:{type(exc).__name__}") from exc
    if not isinstance(value, Mapping):
        raise CandidateBankError("checkpoint_object_required")
    checkpoint = deepcopy(dict(value))
    if checkpoint.get("checkpoint_sha256") != _checkpoint_hash(checkpoint):
        raise CandidateBankError("checkpoint_checksum_mismatch")
    if plan.get("plan_sha256") != _plan_hash(plan) or checkpoint.get("plan_sha256") != plan.get(
        "plan_sha256"
    ):
        raise CandidateBankError("checkpoint_plan_drift")
    if checkpoint.get("model_file_hashes") != dict(model_file_hashes):
        raise CandidateBankError("checkpoint_model_hash_drift")
    return checkpoint


def _next_sequence(checkpoint: Mapping[str, Any]) -> int:
    """Return the next durable event sequence."""

    return 1 + max(
        (int(row.get("sequence", 0)) for row in checkpoint.get("checkpoint_rows", [])),
        default=0,
    )


def _raw_attempt_row(
    attempt: Mapping[str, Any], result: Mapping[str, Any], sequence: int
) -> JsonDict:
    """Build a raw-only row without calling the parser."""

    phases = [deepcopy(dict(row)) for row in result.get("phase_outputs", [])]
    for phase in phases:
        try:
            raw_bytes = bytes.fromhex(str(phase.get("raw_utf8_hex", "")))
        except ValueError as exc:
            raise CandidateBankError("raw_phase_hex_invalid") from exc
        if phase.get("raw_text") != raw_bytes.decode("utf-8", errors="replace"):
            raise CandidateBankError("raw_phase_text_mismatch")
        if phase.get("raw_sha256") != sha256_bytes(raw_bytes):
            raise CandidateBankError("raw_phase_hash_mismatch")
    candidate_raw = str(result.get("candidate_raw_text", ""))
    if result.get("candidate_raw_sha256") != sha256_text(candidate_raw):
        raise CandidateBankError("candidate_raw_hash_mismatch")
    energy_rows = [
        deepcopy(dict(energy)) for phase in phases for energy in phase.get("energy_rows", [])
    ]
    span_rows = [
        deepcopy(dict(span)) for phase in phases for span in phase.get("token_span_rows", [])
    ]
    return {
        "attempt_key": str(attempt["attempt_key"]),
        "ordinal": int(attempt["ordinal"]),
        "hf_id": str(attempt["hf_id"]),
        "pair_id": str(attempt["pair_id"]),
        "split": str(attempt["split"]),
        "formulation_family": str(attempt["formulation_family"]),
        "schedule_id": str(attempt["schedule_id"]),
        "schedule_hash": str(attempt["schedule_hash"]),
        "random_seed": int(attempt["random_seed"]),
        "prompt": str(attempt["prompt"]),
        "prompt_sha256": str(attempt["prompt_sha256"]),
        "call_status": str(result.get("call_status", "exception")),
        "failure_reason": result.get("failure_reason"),
        "phase_outputs": phases,
        "candidate_phase_id": result.get("candidate_phase_id"),
        "candidate_raw_text": candidate_raw,
        "candidate_raw_sha256": str(result["candidate_raw_sha256"]),
        "finish_reason": result.get("finish_reason"),
        "truncated": bool(result.get("truncated", False)),
        "exception_type": result.get("exception_type"),
        "exception_message": result.get("exception_message"),
        "live_duration_s": float(result.get("live_duration_s", 0.0) or 0.0),
        "energy_trace_rows": energy_rows,
        "token_span_rows": span_rows,
        "raw_durable": True,
        "raw_durable_sequence": sequence,
        "terminal": False,
    }


def persist_raw_pair_block(
    checkpoint: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Add one pair block as raw-only rows before any syntax parse."""

    if len(attempts) != len(results):
        raise CandidateBankError("pair_block_result_count_mismatch")
    durable = deepcopy(dict(checkpoint))
    existing = {str(row.get("attempt_key")) for row in durable.get("attempt_rows", [])}
    sequence = _next_sequence(durable)
    new_rows: list[JsonDict] = []
    for attempt, result in zip(attempts, results, strict=True):
        attempt_key = str(attempt["attempt_key"])
        if attempt_key in existing:
            raise CandidateBankError(f"duplicate_raw_attempt:{attempt_key}")
        if result.get("attempt_key") != attempt_key:
            raise CandidateBankError(f"raw_attempt_key_mismatch:{attempt_key}")
        new_rows.append(_raw_attempt_row(attempt, result, sequence))
        existing.add(attempt_key)
    durable.setdefault("attempt_rows", []).extend(new_rows)
    durable.setdefault("checkpoint_rows", []).append(
        {
            "sequence": sequence,
            "stage": "raw_before_parse",
            "attempt_keys": [row["attempt_key"] for row in new_rows],
            "raw_hashes": [row["candidate_raw_sha256"] for row in new_rows],
        }
    )
    durable["checkpoint_sha256"] = _checkpoint_hash(durable)
    return durable


def finalize_pair_block(checkpoint: Mapping[str, Any], attempt_keys: Sequence[str]) -> JsonDict:
    """Add syntax diagnostics only after the raw checkpoint stage exists."""

    durable = deepcopy(dict(checkpoint))
    wanted = set(attempt_keys)
    sequence = _next_sequence(durable)
    found: set[str] = set()
    for row in durable.get("attempt_rows", []):
        attempt_key = str(row.get("attempt_key"))
        if attempt_key not in wanted:
            continue
        found.add(attempt_key)
        if row.get("raw_durable") is not True or "parser_diagnostic" in row:
            raise CandidateBankError(f"raw_first_state_invalid:{attempt_key}")
        row["parser_diagnostic"] = parse_syntax(str(row.get("candidate_raw_text", "")))
        row["parse_sequence"] = sequence
        row["terminal"] = True
    if found != wanted:
        raise CandidateBankError(f"pair_block_attempt_missing:{sorted(wanted - found)}")
    durable.setdefault("checkpoint_rows", []).append(
        {
            "sequence": sequence,
            "stage": "pair_block_terminal",
            "attempt_keys": list(attempt_keys),
        }
    )
    durable["checkpoint_sha256"] = _checkpoint_hash(durable)
    return durable


def _sha256_format(value: Any) -> bool:
    """Return true for one lower-case repository SHA-256 string."""

    text = str(value)
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in text[7:])
    )


def _energy_trace_sufficient(row: Mapping[str, Any]) -> bool:
    """Check token, scalar, vector-hash, and byte-span one-to-one evidence."""

    phases = row.get("phase_outputs", [])
    energy = row.get("energy_trace_rows", [])
    spans = row.get("token_span_rows", [])
    if not isinstance(phases, list) or not isinstance(energy, list) or not isinstance(spans, list):
        return False
    tokens = [int(token) for phase in phases for token in phase.get("token_ids", [])]
    if len(tokens) != len(energy) or len(tokens) != len(spans):
        return False
    if [entry.get("attempt_step_index") for entry in energy] != list(range(len(tokens))):
        return False
    if [entry.get("attempt_step_index") for entry in spans] != list(range(len(tokens))):
        return False
    scalar_fields = (
        "selected_token_logit",
        "full_vocabulary_logsumexp",
        "selected_token_logprob",
        "entropy",
        "top_probability",
    )
    for token, scalar, span in zip(tokens, energy, spans, strict=True):
        if scalar.get("emitted_token_id") != token or span.get("emitted_token_id") != token:
            return False
        if any(
            isinstance(scalar.get(field), bool)
            or not isinstance(scalar.get(field), (int, float))
            or not math.isfinite(float(scalar[field]))
            for field in scalar_fields
        ):
            return False
        if not 0.0 <= float(scalar["top_probability"]) <= 1.0:
            return False
        if int(scalar.get("full_vocabulary_size", 0) or 0) <= 0:
            return False
        if not _sha256_format(scalar.get("full_vocabulary_logits_sha256")):
            return False
        if "full_vocabulary_logits" in scalar or "logits" in scalar:
            return False
        if not all(
            isinstance(span.get(field), int)
            for field in (
                "phase_byte_start",
                "phase_byte_end",
                "attempt_byte_start",
                "attempt_byte_end",
            )
        ):
            return False
        if span["phase_byte_start"] > span["phase_byte_end"]:
            return False
        if span["attempt_byte_start"] > span["attempt_byte_end"]:
            return False
    for phase in phases:
        try:
            raw_bytes = bytes.fromhex(str(phase.get("raw_utf8_hex", "")))
        except ValueError:
            return False
        if phase.get("raw_sha256") != sha256_bytes(raw_bytes):
            return False
        phase_spans = phase.get("token_span_rows", [])
        if phase_spans and phase_spans[-1].get("phase_byte_end") != len(raw_bytes):
            return False
    return True


def completion_errors(
    *,
    plan: Mapping[str, Any],
    attempt_rows: Sequence[Mapping[str, Any]],
    gpu_runtime_rows: Sequence[Mapping[str, Any]],
    schedule_rows: Sequence[Mapping[str, Any]],
    split_isolation_rows: Sequence[Mapping[str, Any]],
    teardown_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Recompute completion without labels, solvers, or syntax success rates."""

    errors: list[str] = []
    expected_keys = [str(row["attempt_key"]) for row in plan.get("attempts", [])]
    observed_keys = [str(row.get("attempt_key")) for row in attempt_rows]
    if len(attempt_rows) != EXPECTED_ATTEMPT_COUNT or len(expected_keys) != EXPECTED_ATTEMPT_COUNT:
        errors.append("attempt_count_mismatch")
    if observed_keys != expected_keys or len(set(observed_keys)) != len(observed_keys):
        errors.append("attempt_roster_mismatch")
    if any(row.get("terminal") is not True for row in attempt_rows):
        errors.append("nonterminal_attempt")
    expected_schedules = deepcopy(SCHEDULE_ROWS)
    if list(schedule_rows) != expected_schedules:
        errors.append("schedule_rows_mismatch")
    observed_hashes = {
        str(row.get("schedule_id")): sha256_text(str(row.get("schedule_text", "")))
        for row in schedule_rows
    }
    if observed_hashes != plan.get("schedule_hashes"):
        errors.append("schedule_hashes_mismatch")
    gpu_ids = [row.get("hf_id") for row in gpu_runtime_rows]
    if gpu_ids != list(REQUIRED_MODEL_IDS) or any(
        row.get("used_cuda") is not True
        or row.get("gpu_indices") != [0, 1]
        or len(set(row.get("gpu_uuids", []))) != 2
        for row in gpu_runtime_rows
    ):
        errors.append("cuda_model_roster_mismatch")
    teardown_ids = [row.get("hf_id") for row in teardown_rows]
    if teardown_ids != list(REQUIRED_MODEL_IDS) or any(
        row.get("process_exit_code") != 0
        or row.get("process_reaped") is not True
        or row.get("model_closed") is not True
        for row in teardown_rows
    ):
        errors.append("teardown_roster_mismatch")
    required_isolation = {
        "pair_ids_disjoint",
        "held_future_excluded",
        "prompt_authority_terms_absent",
        "other_schedule_outputs_absent",
    }
    if {row.get("check") for row in split_isolation_rows} != required_isolation or any(
        row.get("passed") is not True for row in split_isolation_rows
    ):
        errors.append("split_isolation_failed")
    for row in attempt_rows:
        attempt_key = str(row.get("attempt_key"))
        if row.get("raw_durable") is not True or not isinstance(
            row.get("raw_durable_sequence"), int
        ):
            errors.append(f"raw_not_durable:{attempt_key}")
        if not isinstance(row.get("parse_sequence"), int) or int(
            row.get("parse_sequence", 0)
        ) <= int(row.get("raw_durable_sequence", 0)):
            errors.append(f"parse_preceded_raw:{attempt_key}")
        if row.get("candidate_raw_sha256") != sha256_text(str(row.get("candidate_raw_text", ""))):
            errors.append(f"candidate_raw_hash_mismatch:{attempt_key}")
        if not _energy_trace_sufficient(row):
            errors.append(f"energy_trace_insufficient:{attempt_key}")
    return list(dict.fromkeys(errors))


def _summary_rows(attempt_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project one compact terminal record per attempt."""

    fields = (
        "attempt_key",
        "ordinal",
        "hf_id",
        "pair_id",
        "split",
        "formulation_family",
        "schedule_id",
        "call_status",
        "failure_reason",
        "finish_reason",
        "truncated",
        "terminal",
        "candidate_raw_sha256",
        "raw_durable_sequence",
        "parse_sequence",
    )
    return [{field: deepcopy(row.get(field)) for field in fields} for row in attempt_rows]


def _raw_output_rows(attempt_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project raw phases without parser fields."""

    return [
        {
            "attempt_key": row.get("attempt_key"),
            "candidate_phase_id": row.get("candidate_phase_id"),
            "candidate_raw_text": row.get("candidate_raw_text"),
            "candidate_raw_sha256": row.get("candidate_raw_sha256"),
            "phase_outputs": deepcopy(row.get("phase_outputs", [])),
            "raw_durable_sequence": row.get("raw_durable_sequence"),
        }
        for row in attempt_rows
    ]


def _parser_rows(attempt_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project syntax diagnostics while retaining attempt identity."""

    return [
        {
            "attempt_key": row.get("attempt_key"),
            "schedule_id": row.get("schedule_id"),
            "parser_diagnostic": deepcopy(row.get("parser_diagnostic", {})),
            "parse_sequence": row.get("parse_sequence"),
        }
        for row in attempt_rows
    ]


def _trace_projection(attempt_rows: Sequence[Mapping[str, Any]], field: str) -> list[JsonDict]:
    """Flatten token evidence while retaining model, pair, and schedule identity."""

    return [
        {
            "attempt_key": attempt.get("attempt_key"),
            "hf_id": attempt.get("hf_id"),
            "pair_id": attempt.get("pair_id"),
            "schedule_id": attempt.get("schedule_id"),
            **deepcopy(dict(row)),
        }
        for attempt in attempt_rows
        for row in attempt.get(field, [])
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash frozen inputs, raw bytes, and logit hashes without runtime timing."""

    return sha256_json(
        {
            "schema": artifact.get("schema"),
            "run_date": artifact.get("run_date"),
            "source_artifact_hashes": artifact.get("source_artifact_hashes"),
            "model_file_hashes": artifact.get("model_file_hashes"),
            "schedule_hashes": artifact.get("schedule_hashes"),
            "split_hash": artifact.get("split_hash"),
            "attempts": [
                {
                    "attempt_key": row.get("attempt_key"),
                    "prompt_sha256": row.get("prompt_sha256"),
                    "candidate_raw_sha256": row.get("candidate_raw_sha256"),
                    "phase_hashes": [
                        phase.get("raw_sha256") for phase in row.get("phase_outputs", [])
                    ],
                    "logit_hashes": [
                        trace.get("full_vocabulary_logits_sha256")
                        for trace in row.get("energy_trace_rows", [])
                    ],
                }
                for row in artifact.get("per_attempt_rows", [])
            ],
            "random_seed": artifact.get("random_seed"),
        }
    )


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    plan: Mapping[str, Any],
    attempt_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    model_file_hashes: Mapping[str, Any],
    gpu_runtime_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    teardown_rows: Sequence[Mapping[str, Any]],
    split_isolation_rows: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    source_artifact_hashes: Mapping[str, Any],
) -> JsonDict:
    """Build all artifact projections from durable attempt rows."""

    attempts = [deepcopy(dict(row)) for row in attempt_rows]
    gpu_rows = [deepcopy(dict(row)) for row in gpu_runtime_rows]
    teardown = [deepcopy(dict(row)) for row in teardown_rows]
    isolation = [deepcopy(dict(row)) for row in split_isolation_rows]
    complete_errors = completion_errors(
        plan=plan,
        attempt_rows=attempts,
        gpu_runtime_rows=gpu_rows,
        schedule_rows=plan.get("schedule_rows", []),
        split_isolation_rows=isolation,
        teardown_rows=teardown,
    )
    score = int(preconditions_checked.get("all_passed") is True and not complete_errors)
    if preconditions_checked.get("all_passed") is not True:
        verdict_class = "blocked"
        honest_verdict = "blocked_delayed_constraint_candidate_bank"
    elif score == 1:
        verdict_class = "positive"
        honest_verdict = "complete: delayed-constraint candidate bank acquired without selection"
    else:
        verdict_class = "partial"
        honest_verdict = "partial_delayed_constraint_candidate_bank"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(run_date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions_checked)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "live_duration_s": sum(float(row.get("live_duration_s", 0.0)) for row in attempts),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
        "models_used": [str(row.get("hf_id")) for row in gpu_rows],
        "model_file_hashes": deepcopy(dict(model_file_hashes)),
        "gpu_runtime_rows": gpu_rows,
        "schedule_rows": deepcopy(plan.get("schedule_rows", [])),
        "schedule_hashes": deepcopy(dict(plan.get("schedule_hashes", {}))),
        "selected_pair_rows": deepcopy(list(plan.get("selected_pair_rows", []))),
        "split_hash": str(plan.get("split_hash", compute_split_hash([]))),
        "rows": _summary_rows(attempts),
        "per_attempt_rows": attempts,
        "raw_output_rows": _raw_output_rows(attempts),
        "parser_diagnostic_rows": _parser_rows(attempts),
        "energy_trace_rows": _trace_projection(attempts, "energy_trace_rows"),
        "token_span_rows": _trace_projection(attempts, "token_span_rows"),
        "checkpoint_rows": [deepcopy(dict(row)) for row in checkpoint_rows],
        "teardown_rows": teardown,
        "split_isolation_rows": isolation,
        "expected_attempt_count": EXPECTED_ATTEMPT_COUNT,
        "observed_attempt_count": len(attempts),
        "candidate_bank_complete_score": score,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": (
            gate_summary(preconditions_checked.get("checks", []))
            if preconditions_checked.get("all_passed") is not True
            else {
                "failed_check": complete_errors[0] if complete_errors else None,
                "expected_value": "all completion checks pass",
                "observed_value": (
                    complete_errors[0] if complete_errors else "all completion checks pass"
                ),
                "checks": deepcopy(list(preconditions_checked.get("checks", []))),
                "passed": not complete_errors,
            }
        ),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
) -> JsonDict:
    """Build the complete required schema when preflight blocks generation."""

    empty_plan = {
        "schedule_rows": deepcopy(SCHEDULE_ROWS),
        "schedule_hashes": {
            str(row["schedule_id"]): sha256_text(str(row["schedule_text"])) for row in SCHEDULE_ROWS
        },
        "selected_pair_rows": [],
        "split_hash": compute_split_hash([]),
        "attempts": [],
    }
    return build_artifact(
        run_date=run_date,
        duration_s=duration_s,
        plan=empty_plan,
        attempt_rows=[],
        model_specs=model_specs,
        model_file_hashes={},
        gpu_runtime_rows=[],
        checkpoint_rows=[],
        teardown_rows=[],
        split_isolation_rows=[],
        preconditions_checked={
            "all_passed": False,
            "checks": [deepcopy(dict(row)) for row in checks],
        },
        source_artifact_hashes=source_artifact_hashes,
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-recompute schema, projections, hashes, score, and verdict."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(f"missing_field:{field}" for field in missing)
    if missing:
        return errors
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("expected_attempt_count") != EXPECTED_ATTEMPT_COUNT:
        errors.append("expected_attempt_count_mismatch")
    if type(artifact.get("candidate_bank_complete_score")) is not int:
        errors.append("candidate_bank_complete_score_not_bare_int")
    attempts = artifact.get("per_attempt_rows", [])
    if artifact.get("observed_attempt_count") != len(attempts):
        errors.append("observed_attempt_count_mismatch")
    if artifact.get("rows") != _summary_rows(attempts):
        errors.append("rows_projection_mismatch")
    if artifact.get("raw_output_rows") != _raw_output_rows(attempts):
        errors.append("raw_output_rows_projection_mismatch")
    if artifact.get("parser_diagnostic_rows") != _parser_rows(attempts):
        errors.append("parser_diagnostic_rows_projection_mismatch")
    if artifact.get("energy_trace_rows") != _trace_projection(attempts, "energy_trace_rows"):
        errors.append("energy_trace_rows_projection_mismatch")
    if artifact.get("token_span_rows") != _trace_projection(attempts, "token_span_rows"):
        errors.append("token_span_rows_projection_mismatch")
    schedule_hashes = {
        str(row.get("schedule_id")): sha256_text(str(row.get("schedule_text", "")))
        for row in artifact.get("schedule_rows", [])
    }
    if schedule_hashes != artifact.get("schedule_hashes"):
        errors.append("schedule_hashes_mismatch")
    if artifact.get("split_hash") != compute_split_hash(artifact.get("selected_pair_rows", [])):
        errors.append("split_hash_mismatch")
    preflight = artifact.get("preconditions_checked", {})
    if preflight.get("all_passed") is True:
        plan = {
            "attempts": [
                {"attempt_key": row.get("attempt_key")}
                for row in sorted(attempts, key=lambda item: int(item.get("ordinal", -1)))
            ],
            "schedule_hashes": artifact.get("schedule_hashes"),
        }
        complete = completion_errors(
            plan=plan,
            attempt_rows=attempts,
            gpu_runtime_rows=artifact.get("gpu_runtime_rows", []),
            schedule_rows=artifact.get("schedule_rows", []),
            split_isolation_rows=artifact.get("split_isolation_rows", []),
            teardown_rows=artifact.get("teardown_rows", []),
        )
        expected_score = int(not complete)
        if artifact.get("candidate_bank_complete_score") != expected_score:
            errors.append("candidate_bank_complete_score_mismatch")
        if expected_score == 1:
            if artifact.get("models_used") != list(REQUIRED_MODEL_IDS):
                errors.append("models_used_mismatch")
            if artifact.get("verdict_class") != "positive" or not str(
                artifact.get("honest_verdict", "")
            ).startswith("complete:"):
                errors.append("positive_verdict_mismatch")
        elif artifact.get("verdict_class") != "partial" or not str(
            artifact.get("honest_verdict", "")
        ).startswith("partial_"):
            errors.append("partial_verdict_mismatch")
    else:
        summary = artifact.get("gate_check_summary", {})
        if artifact.get("candidate_bank_complete_score") != 0:
            errors.append("blocked_score_mismatch")
        if (
            artifact.get("verdict_class") != "blocked"
            or artifact.get("honest_verdict") != "blocked_delayed_constraint_candidate_bank"
        ):
            errors.append("blocked_verdict_mismatch")
        if not isinstance(summary, Mapping) or not all(
            field in summary for field in ("failed_check", "expected_value", "observed_value")
        ):
            errors.append("blocked_gate_summary_incomplete")
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
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _read_json(path: Path) -> JsonDict:
    """Read one required JSON object through a stable type boundary."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateBankError(
            f"required_json_unreadable:{path.name}:{type(exc).__name__}"
        ) from exc
    if not isinstance(value, Mapping):
        raise CandidateBankError(f"required_json_object_expected:{path.name}")
    return deepcopy(dict(value))


def _atomic_path_probe(path: Path) -> bool:  # pragma: no cover - host filesystem boundary.
    """Prove same-directory write and replacement without changing the target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    first: Path | None = None
    second: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            first = Path(handle.name)
            handle.write(b"first")
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            second = Path(handle.name)
            handle.write(b"second")
        os.replace(second, first)
        second = None
        return first.read_bytes() == b"second"
    except OSError:
        return False
    finally:
        for candidate in (first, second):
            if candidate is not None:
                candidate.unlink(missing_ok=True)


def _source_hashes() -> JsonDict:  # pragma: no cover - repository file boundary.
    """Hash all frozen sources that define generation or interpretation."""

    paths = {
        "experiment_6973": EXP6973_PATH,
        "experiment_6974": EXP6974_PATH,
        "experiment_6967": EXP6967_PATH,
        "experiment_6969": EXP6969_PATH,
        "experiment_5923": EXP5923_PATH,
        "experiment_template": REPO_ROOT / "scripts/experiment_template.py",
        "sota_models": REPO_ROOT / "python/carnot/inference/sota_models.py",
        "gemma4_quantized_loader": REPO_ROOT / "python/carnot/pipeline/gemma4_quantized_loader.py",
        "experiment_6956": REPO_ROOT
        / "python/carnot/experiment_6956_three_family_reformulation_bank.py",
        "experiment_6957": REPO_ROOT / "python/carnot/experiment_6957_smt_mapping_certification.py",
        "module": Path(__file__),
        "wrapper": REPO_ROOT
        / "scripts/experiments/experiment_6975_delayed_constraint_candidate_bank.py",
        "tests": REPO_ROOT
        / "tests/python/test_experiment_6975_delayed_constraint_candidate_bank.py",
        "spec": REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def collect_preconditions(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    upstream_paths: Mapping[str, Path] | None = None,
    json_reader: Callable[[Path], JsonDict] = _read_json,
    file_hasher: Callable[[Path], str | None] = sha256_file,
    gpu_probe: Callable[[], JsonDict] = gpu_inventory,
    llama_probe: Callable[[], JsonDict] = llama_cpp_probe,
    writable_probe: Callable[[Path], bool] = _atomic_path_probe,
    model_hasher: Callable[[Path], str | None] = sha256_file,
) -> JsonDict:
    """Check upstream scores, exact fixture bytes, pairs, files, CUDA, and storage."""

    paths = dict(
        upstream_paths
        or {"exp6973": EXP6973_PATH, "exp6974": EXP6974_PATH, "exp6967": EXP6967_PATH}
    )
    documents: dict[str, JsonDict] = {}
    read_errors: dict[str, str] = {}
    for name, path in paths.items():
        try:
            documents[name] = json_reader(path)
        except CandidateBankError as exc:
            documents[name] = {}
            read_errors[name] = str(exc)
    exp6973 = documents.get("exp6973", {})
    exp6974 = documents.get("exp6974", {})
    fixture = documents.get("exp6967", {})
    runtime_score = exp6973.get("lease_aware_runtime_ready_score")
    fixture_score = exp6974.get("fixture_admissibility_ready_score")
    fixture_hash = file_hasher(paths["exp6967"]) if "exp6967" not in read_errors else None
    eligible = eligible_public_pairs(fixture)
    counts = Counter(str(row["split"]) for row in eligible)
    selected: list[JsonDict] = []
    selection_error: str | None = None
    try:
        selected = select_balanced_pairs(fixture)
    except CandidateBankError as exc:
        selection_error = str(exc)
    spec_errors = model_spec_errors(model_specs)
    files = {
        str(row.get("hf_id")): Path(str(row.get("model_path", ""))).is_file() for row in model_specs
    }
    gpu = gpu_probe()
    devices = list(gpu.get("devices", []))
    llama = llama_probe()
    checkpoint_writable = writable_probe(checkpoint_path)
    checks = [
        gate_check(
            "lease_aware_runtime_ready_score",
            1,
            runtime_score if "exp6973" not in read_errors else read_errors["exp6973"],
            passed=type(runtime_score) is int and runtime_score == 1,
        ),
        gate_check(
            "fixture_admissibility_ready_score",
            1,
            fixture_score if "exp6974" not in read_errors else read_errors["exp6974"],
            passed=type(fixture_score) is int and fixture_score == 1,
        ),
        gate_check(
            "exact_exp6967_source_hash",
            EXPECTED_EXP6967_SHA256,
            fixture_hash,
        ),
        gate_check(
            "eligible_calibration_pair_count_at_least",
            ">=6",
            counts.get("calibration", 0),
            passed=counts.get("calibration", 0) >= 6,
        ),
        gate_check(
            "eligible_heldout_pair_count_at_least",
            ">=6",
            counts.get("heldout", 0),
            passed=counts.get("heldout", 0) >= 6,
        ),
        gate_check(
            "balanced_selected_pair_count",
            12,
            selection_error if selection_error else len(selected),
            passed=selection_error is None and len(selected) == 12,
        ),
        gate_check("exact_model_specs", [], spec_errors),
        gate_check(
            "all_three_model_files",
            {model_id: True for model_id in REQUIRED_MODEL_IDS},
            files,
            passed=len(files) == 3 and all(files.values()),
        ),
        gate_check(
            "exact_two_cuda_devices",
            2,
            len(devices),
            passed=gpu.get("query_ok") is True
            and len(devices) == 2
            and all(str(row.get("uuid", "")).startswith("GPU-") for row in devices),
        ),
        gate_check(
            "llama_cpp_cuda_offload",
            {"importable": True, "gpu_offload": True},
            llama,
            passed=llama.get("importable") is True and llama.get("gpu_offload") is True,
        ),
        gate_check("checkpoint_writable", True, checkpoint_writable),
    ]
    upstream_hashes = exp6973.get("model_file_hashes", {})
    if isinstance(upstream_hashes, Mapping) and set(upstream_hashes) == set(REQUIRED_MODEL_IDS):
        model_hashes = deepcopy(dict(upstream_hashes))
    else:
        model_hashes = {
            str(row["hf_id"]): model_hasher(Path(str(row["model_path"])))
            for row in model_specs
            if Path(str(row.get("model_path", ""))).is_file()
        }
    source_hashes = {name: file_hasher(path) for name, path in paths.items()}
    if upstream_paths is None:
        source_hashes = _source_hashes()
    isolation = build_split_isolation_rows(selected, [])
    return {
        "all_passed": all(row.get("passed") is True for row in checks),
        "checks": checks,
        "source_artifact_hashes": source_hashes,
        "model_file_hashes": model_hashes,
        "selected_pair_rows": selected,
        "split_isolation_rows": isolation,
        "gpu_topology": gpu,
        "llama_cpp": llama,
    }


def _format_chat_prompt(model: Any, user_prompt: str) -> tuple[str, list[int]]:  # pragma: no cover
    """Apply the chat template embedded in the same GGUF used for generation."""

    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    template = model.metadata.get("tokenizer.chat_template")
    if not isinstance(template, str) or not template:
        raise CandidateBankError("embedded_chat_template_missing")
    eos = model.detokenize([model.token_eos()]).decode("utf-8", errors="ignore")
    bos = model.detokenize([model.token_bos()]).decode("utf-8", errors="ignore")
    formatter = Jinja2ChatFormatter(
        template=template,
        eos_token=eos,
        bos_token=bos,
        stop_token_ids=[model.token_eos()],
    )
    formatted = formatter(
        messages=[
            {
                "role": "system",
                "content": (
                    "Follow the stated schedule. Do not use external evidence. "
                    "A JSON certificate is a proposal, not a correctness decision."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
    )
    tokens = model.tokenize(formatted.prompt.encode("utf-8"), add_bos=False, special=True)
    if not tokens:
        raise CandidateBankError("formatted_prompt_has_no_tokens")
    return formatted.prompt, [int(token) for token in tokens]


def _grammar() -> Any:  # pragma: no cover - live llama.cpp grammar boundary.
    """Create fresh grammar state for one constrained phase."""

    from llama_cpp import LlamaGrammar

    return LlamaGrammar.from_json_schema(canonical_json(CONSTRAINT_IR_SCHEMA), verbose=False)


def _generate_phase(  # pragma: no cover - live token generation boundary.
    model: Any,
    *,
    phase_id: str,
    prompt: str,
    seed: int,
    max_tokens: int,
    constrained: bool,
    stop_strings: Sequence[str] = (),
    attempt_step_offset: int = 0,
    attempt_byte_offset: int = 0,
) -> JsonDict:
    """Generate one phase and capture adjacent raw-logit statistics."""

    from llama_cpp import llama_cpp

    formatted_prompt, prompt_tokens = _format_chat_prompt(model, prompt)
    model.reset()
    model.set_seed(seed)
    token_ids: list[int] = []
    raw = b""
    energy_rows: list[JsonDict] = []
    span_rows: list[JsonDict] = []
    finish_reason = "length"
    started = time.monotonic_ns()
    generator = model.generate(
        prompt_tokens,
        top_k=int(DECODING_SETTINGS["top_k"]),
        top_p=float(DECODING_SETTINGS["top_p"]),
        temp=float(DECODING_SETTINGS["temperature"]),
        repeat_penalty=float(DECODING_SETTINGS["repeat_penalty"]),
        reset=True,
        grammar=_grammar() if constrained else None,
    )
    for token in generator:
        if llama_cpp.llama_vocab_is_eog(model._model.vocab, token):
            finish_reason = "eos"
            break
        pointer = model._ctx.get_logits()
        logits = np.ctypeslib.as_array(pointer, shape=(model.n_vocab(),)).copy()
        stats = energy_statistics(logits, emitted_token_id=int(token))
        previous = prompt_tokens + token_ids
        piece = bytes(model.detokenize([int(token)], prev_tokens=previous))
        phase_start = len(raw)
        raw += piece
        phase_end = len(raw)
        phase_step = len(token_ids)
        attempt_step = attempt_step_offset + phase_step
        stats.update(
            {
                "phase_id": phase_id,
                "phase_step_index": phase_step,
                "attempt_step_index": attempt_step,
            }
        )
        energy_rows.append(stats)
        span_rows.append(
            {
                "phase_id": phase_id,
                "phase_step_index": phase_step,
                "attempt_step_index": attempt_step,
                "emitted_token_id": int(token),
                "phase_byte_start": phase_start,
                "phase_byte_end": phase_end,
                "attempt_byte_start": attempt_byte_offset + phase_start,
                "attempt_byte_end": attempt_byte_offset + phase_end,
            }
        )
        token_ids.append(int(token))
        if any(stop.encode("utf-8") in raw for stop in stop_strings):
            finish_reason = "frozen_stop"
            break
        if len(token_ids) >= max_tokens:
            finish_reason = "length"
            break
    ended = time.monotonic_ns()
    text = raw.decode("utf-8", errors="replace")
    return {
        "phase_id": phase_id,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "formatted_prompt_sha256": sha256_text(formatted_prompt),
        "prompt_token_count": len(prompt_tokens),
        "raw_text": text,
        "raw_utf8_hex": raw.hex(),
        "raw_sha256": sha256_bytes(raw),
        "finish_reason": finish_reason,
        "token_ids": token_ids,
        "energy_rows": energy_rows,
        "token_span_rows": span_rows,
        "live_duration_s": (ended - started) / 1_000_000_000,
    }


def _tail_prompt(attempt: Mapping[str, Any], context_name: str, context: str) -> str:
    """Bind a constrained tail only to evidence from its own schedule."""

    return (
        f"{attempt['prompt']}\n"
        f"{context_name}={canonical_json(context)}\n"
        "Now emit only the ConstraintIR certificate JSON object."
    )


def _run_schedule(model: Any, attempt: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Run one direct, trigger-switched, or own-draft-conditioned attempt."""

    started = time.monotonic_ns()
    phases: list[JsonDict] = []
    schedule_id = str(attempt["schedule_id"])
    candidate_phase_id: str | None = None
    candidate_raw = ""
    if schedule_id == "direct":
        phase = _generate_phase(
            model,
            phase_id="direct_certificate",
            prompt=str(attempt["prompt"]),
            seed=int(attempt["random_seed"]),
            max_tokens=int(DECODING_SETTINGS["completion_token_cap"]),
            constrained=True,
        )
        phases.append(phase)
        candidate_phase_id = "direct_certificate"
        candidate_raw = str(phase["raw_text"])
    elif schedule_id == "trigger_switched":
        free = _generate_phase(
            model,
            phase_id="free_reasoning",
            prompt=str(attempt["prompt"]),
            seed=int(attempt["random_seed"]),
            max_tokens=int(DECODING_SETTINGS["planning_token_cap"]),
            constrained=False,
            stop_strings=[TRIGGER_TEXT],
        )
        phases.append(free)
        if TRIGGER_TEXT in str(free["raw_text"]):
            tail = _generate_phase(
                model,
                phase_id="certificate_tail",
                prompt=_tail_prompt(attempt, "OWN_TRIGGERED_REASONING", str(free["raw_text"])),
                seed=int(attempt["random_seed"]) + 1,
                max_tokens=min(
                    int(DECODING_SETTINGS["certificate_token_cap"]),
                    int(DECODING_SETTINGS["completion_token_cap"]) - len(free["token_ids"]),
                ),
                constrained=True,
                attempt_step_offset=len(free["token_ids"]),
                attempt_byte_offset=len(bytes.fromhex(free["raw_utf8_hex"])),
            )
            phases.append(tail)
            candidate_phase_id = "certificate_tail"
            candidate_raw = str(tail["raw_text"])
    elif schedule_id == "draft_conditioned":
        draft = _generate_phase(
            model,
            phase_id="unconstrained_draft",
            prompt=str(attempt["prompt"]),
            seed=int(attempt["random_seed"]),
            max_tokens=int(DECODING_SETTINGS["planning_token_cap"]),
            constrained=False,
        )
        phases.append(draft)
        tail = _generate_phase(
            model,
            phase_id="certificate_tail",
            prompt=_tail_prompt(attempt, "OWN_UNCONSTRAINED_DRAFT", str(draft["raw_text"])),
            seed=int(attempt["random_seed"]) + 1,
            max_tokens=min(
                int(DECODING_SETTINGS["certificate_token_cap"]),
                int(DECODING_SETTINGS["completion_token_cap"]) - len(draft["token_ids"]),
            ),
            constrained=True,
            attempt_step_offset=len(draft["token_ids"]),
            attempt_byte_offset=len(bytes.fromhex(draft["raw_utf8_hex"])),
        )
        phases.append(tail)
        candidate_phase_id = "certificate_tail"
        candidate_raw = str(tail["raw_text"])
    else:
        raise CandidateBankError(f"unknown_schedule:{schedule_id}")
    token_count = sum(len(phase["token_ids"]) for phase in phases)
    if token_count > int(DECODING_SETTINGS["completion_token_cap"]):
        raise CandidateBankError(f"completion_token_cap_exceeded:{token_count}")
    finish_reason = phases[-1]["finish_reason"] if phases else "trigger_missing"
    return {
        "attempt_key": attempt["attempt_key"],
        "call_status": "complete",
        "failure_reason": None if candidate_phase_id else "trigger_missing",
        "phase_outputs": phases,
        "candidate_phase_id": candidate_phase_id,
        "candidate_raw_text": candidate_raw,
        "candidate_raw_sha256": sha256_text(candidate_raw),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "exception_type": None,
        "exception_message": None,
        "live_duration_s": (time.monotonic_ns() - started) / 1_000_000_000,
    }


def _failure_result(attempt: Mapping[str, Any], exc: BaseException) -> JsonDict:
    """Keep an exception attempt without inventing raw tokens or logits."""

    return {
        "attempt_key": attempt["attempt_key"],
        "call_status": "timeout" if isinstance(exc, TimeoutError) else "exception",
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "phase_outputs": [],
        "candidate_phase_id": None,
        "candidate_raw_text": "",
        "candidate_raw_sha256": sha256_text(""),
        "finish_reason": None,
        "truncated": False,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "live_duration_s": 0.0,
    }


def _emit_worker_event(event: Mapping[str, Any]) -> None:  # pragma: no cover
    """Write one JSON event without mixing it with backend stderr."""

    sys.stdout.write(json.dumps(dict(event), sort_keys=True) + "\n")
    sys.stdout.flush()


def worker_main(model_path: str) -> int:  # pragma: no cover - live child boundary.
    """Load one GGUF and execute pair blocks until the parent requests close."""

    from llama_cpp import Llama

    model: Any = None
    _emit_worker_event({"event": "started", "pid": os.getpid()})
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_SIZE,
            n_gpu_layers=-1,
            n_batch=512,
            n_ubatch=512,
            main_gpu=0,
            split_mode=1,
            tensor_split=[0.5, 0.5],
            logits_all=False,
            seed=RANDOM_SEED,
            verbose=True,
        )
        _emit_worker_event({"event": "loaded", "pid": os.getpid()})
        for line in sys.stdin:
            command = json.loads(line)
            action = command.get("action")
            if action == "generate_pair_block":
                results = []
                for attempt in command.get("attempts", []):
                    try:
                        results.append(_run_schedule(model, attempt))
                    except Exception as exc:  # noqa: BLE001 - failures remain data rows.
                        results.append(_failure_result(attempt, exc))
                _emit_worker_event(
                    {
                        "event": "pair_block_generated",
                        "pair_id": command.get("pair_id"),
                        "results": results,
                    }
                )
            elif action == "close":
                model.close()
                model = None
                gc.collect()
                _emit_worker_event({"event": "closed", "pid": os.getpid()})
                return 0
            else:
                raise CandidateBankError(f"unknown_worker_action:{action}")
        return 1
    except Exception as exc:  # noqa: BLE001 - parent needs the exact backend failure.
        _emit_worker_event(
            {
                "event": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    finally:
        if model is not None:
            model.close()
        gc.collect()


def _send_worker(
    process: subprocess.Popen[str], payload: Mapping[str, Any]
) -> None:  # pragma: no cover
    """Send one command to the exact child owned by this task."""

    if process.stdin is None:
        raise CandidateBankError("worker_stdin_missing")
    process.stdin.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    process.stdin.flush()


def _wait_worker(  # pragma: no cover - live process boundary.
    process: subprocess.Popen[str], expected_event: str, timeout_s: float
) -> JsonDict:
    """Wait with a wall-clock bound and retain unexpected worker errors."""

    if process.stdout is None:
        raise CandidateBankError("worker_stdout_missing")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([process.stdout], [], [], remaining)
        if not readable:
            break
        line = process.stdout.readline()
        if not line:
            break
        event = json.loads(line)
        if event.get("event") == "error":
            raise CandidateBankError(str(event.get("error")))
        if event.get("event") == expected_event:
            return deepcopy(dict(event))
    raise TimeoutError(f"worker_event_timeout:{expected_event}")


def _pid_gpu_samples(pid: int) -> list[JsonDict]:  # pragma: no cover - host GPU boundary.
    """Join the owned child PID to current NVIDIA process rows."""

    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0].isdigit() or int(parts[0]) != pid:
            continue
        rows.append(
            {
                "pid": pid,
                "gpu_uuid": parts[1],
                "used_memory_mb": int(parts[2]),
            }
        )
    return rows


def _terminate_and_reap(process: subprocess.Popen[str]) -> None:  # pragma: no cover
    """Terminate only the child created by this controller, then reap it."""

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    elif process.returncode is None:
        process.wait(timeout=30)


def _run_model_family(  # pragma: no cover - live GGUF and CUDA integration boundary.
    *,
    model_spec: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> JsonDict:
    """Load one task-owned model, batch schedules by pair, then tear it down."""

    durable = deepcopy(dict(checkpoint))
    command = [
        sys.executable,
        "-m",
        __name__,
        "--worker",
        "--model-path",
        str(model_spec["model_path"]),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    stderr_path = checkpoint_path.parent / (
        "experiment_6975_" + str(model_spec["hf_id"]).split("/")[-1] + ".stderr.log"
    )
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[str] | None = None
    loaded = False
    closed = False
    load_started = time.monotonic_ns()
    try:
        with stderr_path.open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
                env=env,
            )
            _wait_worker(process, "started", 20.0)
            _wait_worker(process, "loaded", MODEL_LOAD_TIMEOUT_S)
            loaded = True
            samples = _pid_gpu_samples(process.pid)
            gpu_uuids = sorted({str(row["gpu_uuid"]) for row in samples})
            gpu_row = {
                "hf_id": model_spec["hf_id"],
                "process_pid": process.pid,
                "gpu_indices": [0, 1],
                "gpu_uuids": gpu_uuids,
                "samples": samples,
                "used_cuda": len(gpu_uuids) == 2
                and all(int(row["used_memory_mb"]) > 0 for row in samples),
            }
            durable.setdefault("gpu_runtime_rows", []).append(gpu_row)
            existing = {str(row.get("attempt_key")) for row in durable.get("attempt_rows", [])}
            model_attempts = [
                row for row in plan["attempts"] if row["hf_id"] == model_spec["hf_id"]
            ]
            pair_ids = list(dict.fromkeys(str(row["pair_id"]) for row in model_attempts))
            for pair_id in pair_ids:
                block = [
                    row
                    for row in model_attempts
                    if row["pair_id"] == pair_id and row["attempt_key"] not in existing
                ]
                if not block:
                    continue
                try:
                    _send_worker(
                        process,
                        {"action": "generate_pair_block", "pair_id": pair_id, "attempts": block},
                    )
                    event = _wait_worker(process, "pair_block_generated", PAIR_BLOCK_TIMEOUT_S)
                    results = event.get("results", [])
                except Exception as exc:  # noqa: BLE001 - preserve all launched attempts.
                    results = [_failure_result(attempt, exc) for attempt in block]
                durable = persist_raw_pair_block(durable, block, results)
                write_checkpoint(checkpoint_path, durable)
                keys = [str(row["attempt_key"]) for row in block]
                durable = finalize_pair_block(durable, keys)
                write_checkpoint(checkpoint_path, durable)
                existing.update(keys)
            _send_worker(process, {"action": "close"})
            _wait_worker(process, "closed", 60.0)
            process.wait(timeout=60.0)
            closed = process.returncode == 0
    except Exception:
        closed = False
    finally:
        if process is not None:
            _terminate_and_reap(process)
    teardown = {
        "hf_id": model_spec["hf_id"],
        "process_pid": process.pid if process is not None else None,
        "load_started_ns": load_started,
        "loaded": loaded,
        "model_closed": closed,
        "process_exit_code": process.returncode if process is not None else None,
        "process_reaped": process is not None and process.poll() is not None,
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    durable.setdefault("teardown_rows", []).append(teardown)
    sequence = _next_sequence(durable)
    durable.setdefault("checkpoint_rows", []).append(
        {
            "sequence": sequence,
            "stage": "family_teardown",
            "hf_id": model_spec["hf_id"],
            "process_exit_code": teardown["process_exit_code"],
        }
    )
    durable["checkpoint_sha256"] = _checkpoint_hash(durable)
    write_checkpoint(checkpoint_path, durable)
    return durable


def run_live_acquisition(  # pragma: no cover - required live E2E boundary.
    *,
    plan: Mapping[str, Any],
    checkpoint_path: Path,
    model_specs: Sequence[Mapping[str, Any]],
    model_file_hashes: Mapping[str, Any],
) -> JsonDict:
    """Resume raw-first state, then run each still-needed model sequentially."""

    if checkpoint_path.is_file():
        checkpoint = load_checkpoint(
            checkpoint_path,
            plan=plan,
            model_file_hashes=model_file_hashes,
        )
    else:
        checkpoint = new_checkpoint(plan, model_file_hashes=model_file_hashes)
        write_checkpoint(checkpoint_path, checkpoint)
    pending = [
        str(row["attempt_key"])
        for row in checkpoint.get("attempt_rows", [])
        if row.get("terminal") is not True
    ]
    if pending:
        checkpoint = finalize_pair_block(checkpoint, pending)
        write_checkpoint(checkpoint_path, checkpoint)
    terminal = {
        str(row["attempt_key"])
        for row in checkpoint.get("attempt_rows", [])
        if row.get("terminal") is True
    }
    teardown_ids = {str(row.get("hf_id")) for row in checkpoint.get("teardown_rows", [])}
    for model in model_specs:
        keys = {
            str(row["attempt_key"]) for row in plan["attempts"] if row["hf_id"] == model["hf_id"]
        }
        if keys <= terminal and model["hf_id"] in teardown_ids:
            continue
        checkpoint = _run_model_family(
            model_spec=model,
            plan=plan,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )
        terminal = {
            str(row["attempt_key"])
            for row in checkpoint.get("attempt_rows", [])
            if row.get("terminal") is True
        }
        teardown_ids.add(str(model["hf_id"]))
    return checkpoint


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preflight_fn: Callable[[Sequence[Mapping[str, Any]], Path], JsonDict] | None = None,
    acquisition_fn: Callable[..., JsonDict] = run_live_acquisition,
) -> JsonDict:
    """Preflight, acquire durable attempts, validate, and write the artifact."""

    started = time.perf_counter()
    specs = [deepcopy(dict(row)) for row in (model_specs or MODEL_SPECS)]
    preflight = (
        collect_preconditions(model_specs=specs, checkpoint_path=checkpoint_path)
        if preflight_fn is None
        else preflight_fn(specs, checkpoint_path)
    )
    if preflight.get("all_passed") is not True:
        artifact = blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            checks=preflight.get("checks", []),
            model_specs=specs,
            source_artifact_hashes=preflight.get("source_artifact_hashes", {}),
        )
        errors = validate_artifact(artifact)
        if errors:
            raise CandidateBankError(f"blocked_artifact_validation_failed:{errors}")
        write_json_atomic(result_path, artifact)
        return artifact
    plan = freeze_plan(preflight["selected_pair_rows"], model_specs=specs)
    if len(plan["attempts"]) != EXPECTED_ATTEMPT_COUNT:
        raise CandidateBankError(f"attempt_budget_mismatch:{len(plan['attempts'])}")
    checkpoint = acquisition_fn(
        plan=plan,
        checkpoint_path=checkpoint_path,
        model_specs=specs,
        model_file_hashes=preflight["model_file_hashes"],
    )
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        plan=plan,
        attempt_rows=checkpoint.get("attempt_rows", []),
        model_specs=specs,
        model_file_hashes=preflight["model_file_hashes"],
        gpu_runtime_rows=checkpoint.get("gpu_runtime_rows", []),
        checkpoint_rows=checkpoint.get("checkpoint_rows", []),
        teardown_rows=checkpoint.get("teardown_rows", []),
        split_isolation_rows=plan["split_isolation_rows"],
        preconditions_checked=preflight,
        source_artifact_hashes=preflight["source_artifact_hashes"],
    )
    errors = validate_artifact(artifact)
    if errors:
        raise CandidateBankError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command surface.
    """Dispatch the required dated command or private worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        if not args.model_path:
            parser.error("--worker requires --model-path")
        return worker_main(args.model_path)
    if args.validate:
        artifact = _read_json(args.result_path)
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(
        run_date=args.date,
        result_path=args.result_path,
        checkpoint_path=args.checkpoint_path,
    )
    print(
        canonical_json(
            {
                "honest_verdict": artifact["honest_verdict"],
                "observed_attempt_count": artifact["observed_attempt_count"],
                "candidate_bank_complete_score": artifact["candidate_bank_complete_score"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module worker surface.
    raise SystemExit(main())
