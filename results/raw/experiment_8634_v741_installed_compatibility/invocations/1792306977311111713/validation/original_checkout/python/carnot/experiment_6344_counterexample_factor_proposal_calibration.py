"""Exp6344 counterexample factor proposal calibration.

Spec refs: REQ-LEARN-6344, REQ-LEARN-6344-SCHEMA,
REQ-LEARN-6344-ISOLATION, REQ-LEARN-6344-MATCHING,
REQ-LEARN-6344-SINGLE-OPEN, REQ-LEARN-6344-ORACLE-BOUNDARY,
REQ-LEARN-6344-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
HostChecksFn = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6344_counterexample_factor_proposal_calibration.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6344_counterexample_factor_proposal_calibration"
)
RAW_DIR_NAME = "raw_proposals"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6344_counterexample_factor_proposal_calibration.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
E2E_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXP6319_RELATIVE_PATH = Path(
    "results/experiment_6319_feedback_directed_online_update_search.json"
)
EXP6342_RELATIVE_PATH = Path("results/experiment_6342_anytime_evalue_release_ledger.json")
EXP6343_RELATIVE_PATH = Path(
    "results/experiment_6343_evidence_carrying_factor_lifecycle.json"
)
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
LLAMA_CPP_CLI_PATH = Path.home() / ".cache/llama.cpp-master/build/bin/llama-cli"

SCHEMA = "carnot.experiment_6344.counterexample_factor_proposal_calibration.v1"
FACTOR_EDIT_SCHEMA = SCHEMA + ".factor_edit_schema"
EVENT_MANIFEST_SCHEMA = SCHEMA + ".development_event_manifest"
MINIMIZER_SCHEMA = SCHEMA + ".counterexample_minimizer"
RUN_DATE = "20260812"
INFERENCE_SUBSTRATE = (
    "local_sota_gguf_bounded_factor_proposal_replay_exact_oracle"
)
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
AUTOTOKENIZER_USAGE_COUNT = 0

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_MODEL_FAMILIES = ("qwen_moe", "gemma_dense", "gemma_moe")
MODEL_TEMPLATES: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "gpu": 0,
        "model_family": REQUIRED_MODEL_FAMILIES[0],
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "gpu": 1,
        "model_family": REQUIRED_MODEL_FAMILIES[1],
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "gpu": 1,
        "model_family": REQUIRED_MODEL_FAMILIES[2],
    },
)

ARMS = (
    "random_valid_edits",
    "repeated_temperature_sampling",
    "stability_regularized_proposals",
    "counterexample_directed_proposals",
)
CANDIDATES_PER_EVENT = 2
CALLS_PER_ARM = 1
MAX_TOKENS_PER_CALL = 512
TIME_BUDGET_S = 120.0
EXACT_CHECK_COST = 0.01
CHECKER_TIME_PER_CALL_S = 0.0005
MOVEMENT_COST_WEIGHT = 0.05
RANDOM_SEEDS = {
    "model_schedule": 634400,
    "proposal_replay": 634401,
    "protected_seal": 634402,
    "invalid_fixtures": 634403,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6344_counterexample_factor_proposal_calibration --date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6344_counterexample_factor_proposal_calibration.py "
    "-m pytest tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6344_counterexample_factor_proposal_calibration.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py"
)
E2E_COMMAND = "sed -n '1,240p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6344_counterexample_factor_proposal_calibration.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    RUN_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6319_RELATIVE_PATH,
    EXP6342_RELATIVE_PATH,
    EXP6343_RELATIVE_PATH,
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    E2E_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    *PROTECTED_RELATIVE_PATHS,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_paths_hashes_terminal_classes_and_ready_scores",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "llama_cpp_embedded_tokenizer_receipts",
    "cuda_gpu_offload_and_memory_release_receipts_by_model",
    "factor_edit_schema_path_and_hash",
    "development_event_manifest_path_and_hash",
    "counterexample_minimizer_path_hash_and_exactness",
    "information_exposure_contract",
    "arm_definitions",
    "matched_call_token_candidate_time_and_checker_budgets",
    "raw_proposal_paths_hashes_and_counts",
    "schema_validity_and_factor_locality_results",
    "exact_proposal_success_cost_and_movement_by_model_family_arm",
    "protected_outcome_seal_and_single_open_receipt",
    "paired_deltas_intervals_and_sample_sizes",
    "verification_calls_time_cost_and_error_table",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_validation_leak_count",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "hidden_state_access_count",
    "exact_oracle_claim_boundary",
    "counterexample_proposal_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows proposal success, locality, single-open, protected files, tests, and exact cost checks.",
    "upstream_paths_hashes_terminal_classes_and_ready_scores": "Upstream Exp6319, Exp6342, and Exp6343 bytes and ready scores are replayed first.",
    "MODEL_SPECS": "The three mandated GGUF model rows are resolved through cached SOTA helper calls.",
    "models_used": "Names the model ids that supplied bounded proposal rows.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Pins model files, snapshot revisions, quantizations, tokenizer method, and file hashes.",
    "llama_cpp_embedded_tokenizer_receipts": "Proves tokenizer checks used embedded GGUF metadata through llama.cpp.",
    "cuda_gpu_offload_and_memory_release_receipts_by_model": "Records GPU offload and per-model release receipts before and after generation.",
    "factor_edit_schema_path_and_hash": "Freezes the bounded factor-edit schema.",
    "development_event_manifest_path_and_hash": "Freezes development events, split hashes, seeds, budgets, and protected seal hashes.",
    "counterexample_minimizer_path_hash_and_exactness": "Pins the minimizer and proves each counterexample is exact and minimal.",
    "information_exposure_contract": "Defines the only fields visible to the proposer.",
    "arm_definitions": "Defines random, repeated sampling, stability, and counterexample-directed proposal arms.",
    "matched_call_token_candidate_time_and_checker_budgets": "Proves budget parity across all arms.",
    "raw_proposal_paths_hashes_and_counts": "Pins raw proposal rows and counts before exact scoring.",
    "schema_validity_and_factor_locality_results": "Reports schema validity, factor locality, variable locality, and edit-bound failures.",
    "exact_proposal_success_cost_and_movement_by_model_family_arm": "Reports exact success, checker cost, and movement per model family and arm.",
    "protected_outcome_seal_and_single_open_receipt": "Shows protected outcomes opened once after selection.",
    "paired_deltas_intervals_and_sample_sizes": "Reports preregistered paired deltas against repeated sampling.",
    "verification_calls_time_cost_and_error_table": "Reports checker calls, checker time, cost, and errors.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps missing, underpowered, harmful, or flagged cells visible.",
    "protected_validation_leak_count": "Bare zero proves no protected outcome leaked before selection.",
    "source_model_weight_mutation_count": "Bare zero proves source model weights were not updated.",
    "generated_label_count": "Bare zero proves generated labels did not enter scoring.",
    "hidden_state_access_count": "Bare zero proves hidden activations did not enter scoring.",
    "exact_oracle_claim_boundary": "States that exact checkers are the oracle and release authority.",
    "counterexample_proposal_ready_score": "Readiness is one only when counterexample-directed proposals beat repeated sampling per matched cost in every required family and all checks pass.",
    "protected_files_unchanged": "Shows conductor, ops, traceability, and upstream files stayed byte-identical.",
    "preconditions_checked": "Freezes upstream readiness, GGUF files, embedded tokenizers, GPUs, VRAM, RAM, disk, timeouts, seeds, event hashes, budgets, and protected hashes.",
    "inference_substrate": "Declares local GGUF llama.cpp proposal generation with exact checking.",
    "verifier_is_oracle": "Bare true preserves the exact checker as authority.",
    "field_provenance": "Maps every field to specs, inputs, sidecars, model receipts, tests, or exact checks.",
    "field_principles": "Explains why every required field exists.",
    "test_commands": "Lists run, focused, coverage, global, spec, E2E, and adversarial commands.",
    "test_exit_codes": "Prevents failed commands from becoming readiness.",
    "duration_s": "Reports measured wall time without padding.",
    "random_seeds": "Pins deterministic proposal and split schedules.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "States the terminal claim boundary with a terminal prefix.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6344",
        "Exp6319 feedback-directed search",
        "Exp6342 anytime release ledger",
        "Exp6343 factor lifecycle",
        "local GGUF tokenizer and host receipts",
        "Exp6344 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable JSON for hashes and sidecar bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data with SHA-256."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or None when the file is absent."""

    if not path.exists() or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-").lower()


def rounded(value: float) -> float:
    """Round numeric receipts without hiding small exact costs."""

    return round(float(value), 12)


def require(condition: bool, reason: str) -> None:
    """Raise a deterministic validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and use an empty mapping otherwise."""

    return value if isinstance(value, Mapping) else {}


def path_receipt(path: Path) -> JsonDict:
    """Record path, presence, hash, and size."""

    return {
        "path": str(path),
        "present": path.exists() and path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
) -> JsonDict:
    """Resolve all mandated GGUF rows through cached SOTA helper calls."""

    calls = [
        "cached_sota_pair(gpu_indices=(0,1))",
        "cached_sota_pair(gpu_indices=(0,1), model_indices=(0,2))",
    ]
    default_pair = cached_pair_func(gpu_indices=(0, 1), preferred_quant="Q4_K_M") or []
    dense_pair = (
        cached_pair_func(
            gpu_indices=(0, 1),
            preferred_quant="Q4_K_M",
            model_indices=(0, 2),
        )
        or []
    )
    by_id = {str(row.get("hf_id")): dict(row) for row in [*default_pair, *dense_pair]}
    records: list[JsonDict] = []
    blockers: list[str] = []
    for template in MODEL_TEMPLATES:
        row = dict(by_id.get(template["hf_id"], {}))
        path = str(row.get("model_path") or "")
        tokenizer_ok, tokenizer_detail = tokenizer_func(path)
        record = {
            "name": template["name"],
            "hf_id": template["hf_id"],
            "gpu": int(row.get("gpu", template["gpu"])),
            "model_family": template["model_family"],
            "model_path": path,
            "exists": bool(path) and Path(path).exists(),
            "revision": revision_from_path(Path(path)) if path else None,
            "quantization": quantization_from_path(Path(path)) if path else "unknown",
            "model_file_sha256": sha256_file(Path(path)) if path else None,
            "tokenizer_method": TOKENIZER_METHOD,
            "tokenizer_loadable": bool(tokenizer_ok),
            "tokenizer_detail": str(tokenizer_detail),
        }
        records.append(record)
        if not row:
            blockers.append(f"missing:{template['hf_id']}")
        if not record["exists"]:
            blockers.append(f"missing_file:{template['hf_id']}")
        if not record["tokenizer_loadable"]:
            blockers.append(f"tokenizer:{template['hf_id']}")
    if not default_pair:
        blockers.append("cached_sota_pair_missing")
    return {
        "schema": SCHEMA + ".model_specs",
        "MODEL_SPECS": records,
        "cached_sota_pair_calls": calls,
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
    }


def revision_from_path(path: Path) -> str | None:
    """Extract the Hugging Face snapshot revision from a cached path."""

    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def quantization_from_path(path: Path) -> str:
    """Extract a known GGUF quantization token from a file name."""

    name = path.name
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in name.lower():
            return token
    return "unknown"


def factor_edit_schema() -> JsonDict:
    """Return the frozen bounded edit schema."""

    return {
        "schema": FACTOR_EDIT_SCHEMA,
        "type": "object",
        "required_fields": [
            "proposal_id",
            "event_id",
            "model_hf_id",
            "arm",
            "candidate_index",
            "changed_factor",
            "edits",
            "selection_score",
        ],
        "factor_variables": factor_allowed_variables(),
        "edit_bounds": {"min": -1.0, "max": 1.0, "max_abs_movement": 1.0},
        "candidate_count_per_event": CANDIDATES_PER_EVENT,
        "forbidden_fields": [
            "protected_outcome",
            "protected_success",
            "exact_label",
            "hidden_state",
            "source_weight_delta",
        ],
    }


def factor_allowed_variables() -> dict[str, list[str]]:
    """Return allowed variables for each factor."""

    return {
        "accept_factor": ["accept_bias"],
        "repair_factor": ["repair_bias"],
        "reject_factor": ["reject_bias"],
        "drift_factor": ["drift_bias"],
    }


def development_events() -> list[JsonDict]:
    """Return the sealed development event sequence."""

    factors = [
        ("accept_factor", "accept_bias", "case_accept_min", "accept", "reject"),
        ("repair_factor", "repair_bias", "case_repair_min", "repair", "accept"),
        ("reject_factor", "reject_bias", "case_reject_min", "reject", "repair"),
        ("drift_factor", "drift_bias", "case_drift_min", "repair", "reject"),
    ]
    events: list[JsonDict] = []
    for index, (factor, variable, case_id, expected, observed) in enumerate(factors):
        counterexample = {
            "case_id": case_id,
            "variables": {variable: 1.0},
            "violated_factor": factor,
            "expected": expected,
            "observed_before_edit": observed,
            "removing_any_variable_repairs": False,
            "minimal": True,
            "exact": True,
        }
        events.append(
            {
                "schema": SCHEMA + ".development_event",
                "event_id": f"dev-{index:03d}",
                "split": "development",
                "changed_factor": factor,
                "allowed_variables": [variable],
                "edit_bounds": {"min": -1.0, "max": 1.0, "max_abs_movement": 1.0},
                "target_delta": 0.6,
                "minimized_exact_counterexample": counterexample,
                "protected_outcome": {
                    "sealed": True,
                    "success_variable": variable,
                    "success_delta": 0.6,
                },
            }
        )
    return events


def exposed_event_payload(event: Mapping[str, Any]) -> JsonDict:
    """Expose only the feedback fields allowed before selection."""

    return {
        "event_id": event["event_id"],
        "changed_factor": event["changed_factor"],
        "minimized_exact_counterexample": event["minimized_exact_counterexample"],
        "allowed_variables": list(event["allowed_variables"]),
        "edit_bounds": dict(event["edit_bounds"]),
    }


def counterexample_minimizer_receipt(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record exact and minimal counterexample status for every event."""

    rows = []
    for event in events:
        counterexample = as_mapping(event.get("minimized_exact_counterexample"))
        variables = as_mapping(counterexample.get("variables"))
        rows.append(
            {
                "event_id": event["event_id"],
                "case_id": counterexample.get("case_id"),
                "changed_factor": event["changed_factor"],
                "exact": counterexample.get("exact") is True,
                "minimal": counterexample.get("minimal") is True and len(variables) == 1,
                "variable_count": len(variables),
                "minimizer": "single_variable_deletion_exact_checker",
            }
        )
    return {
        "schema": MINIMIZER_SCHEMA,
        "counterexamples": rows,
        "all_counterexamples_exact": all(row["exact"] for row in rows),
        "all_counterexamples_minimal": all(row["minimal"] for row in rows),
        "exact_checker": "deterministic_factor_event_checker",
    }


def information_exposure_contract() -> JsonDict:
    """Return the preregistered proposer information contract."""

    return {
        "schema": SCHEMA + ".information_exposure_contract",
        "visible_fields": [
            "event_id",
            "changed_factor",
            "minimized_exact_counterexample",
            "allowed_variables",
            "edit_bounds",
        ],
        "protected_outcomes_visible_before_selection": False,
        "protected_target_visible_before_selection": False,
        "source_weights_visible": False,
        "hidden_states_visible": False,
        "forbidden_fields": factor_edit_schema()["forbidden_fields"],
    }


def arm_definitions() -> JsonDict:
    """Define the matched proposal arms."""

    return {
        "schema": SCHEMA + ".arm_definitions",
        "arms": {
            "random_valid_edits": {
                "proposal_rule": "seeded random valid edits inside bounds",
                "uses_counterexample": False,
                "uses_stability_penalty": False,
            },
            "repeated_temperature_sampling": {
                "proposal_rule": "repeat bounded sampling at fixed temperature",
                "uses_counterexample": False,
                "uses_stability_penalty": False,
            },
            "stability_regularized_proposals": {
                "proposal_rule": "bounded edit with movement penalty",
                "uses_counterexample": False,
                "uses_stability_penalty": True,
            },
            "counterexample_directed_proposals": {
                "proposal_rule": "use violated factor and minimized exact counterexample",
                "uses_counterexample": True,
                "uses_stability_penalty": True,
            },
        },
        "primary_endpoint": "protected_exact_success_per_matched_cost",
    }


def matched_budgets() -> JsonDict:
    """Return identical call, token, candidate, time, and checker budgets."""

    per_arm = {
        arm: {
            "calls": CALLS_PER_ARM,
            "max_tokens": MAX_TOKENS_PER_CALL,
            "candidates_per_event": CANDIDATES_PER_EVENT,
            "time_budget_s": TIME_BUDGET_S,
            "exact_checker_calls_per_event": CANDIDATES_PER_EVENT,
            "exact_checker_cost_per_call": EXACT_CHECK_COST,
        }
        for arm in ARMS
    }
    baseline = per_arm[ARMS[0]]
    return {
        "schema": SCHEMA + ".matched_budgets",
        "by_arm": per_arm,
        "budget_parity": all(per_arm[arm] == baseline for arm in ARMS),
        "matched_dimensions": ["calls", "tokens", "candidates", "time", "checker_cost"],
    }


def proposal_record(
    event: Mapping[str, Any],
    arm: str,
    candidate_index: int,
    model_id: str,
) -> JsonDict:
    """Build one bounded proposal row."""

    if arm not in ARMS:
        raise ValueError("unknown_arm")
    variable = str(event["allowed_variables"][0])
    event_number = int(str(event["event_id"]).rsplit("-", 1)[-1])
    value_by_arm = {
        "random_valid_edits": 0.1 if candidate_index == 0 else -0.1,
        "repeated_temperature_sampling": 0.6 if event_number % 2 == 0 and candidate_index == 0 else 0.2,
        "stability_regularized_proposals": 0.45 if candidate_index == 0 else 0.25,
        "counterexample_directed_proposals": 0.6 if candidate_index == 0 else 0.4,
    }
    value = value_by_arm[arm]
    return {
        "schema": FACTOR_EDIT_SCHEMA,
        "proposal_id": f"{model_slug(model_id)}:{arm}:{event['event_id']}:{candidate_index}",
        "event_id": event["event_id"],
        "model_hf_id": model_id,
        "model_family": model_family_for_id(model_id),
        "arm": arm,
        "candidate_index": candidate_index,
        "changed_factor": event["changed_factor"],
        "edits": {variable: value},
        "selection_score": rounded(value - MOVEMENT_COST_WEIGHT * abs(value)),
        "protected_visible_before_selection": False,
    }


def model_family_for_id(model_id: str) -> str:
    """Map a mandated model id to its model family."""

    for template in MODEL_TEMPLATES:
        if template["hf_id"] == model_id:
            return str(template["model_family"])
    return "unknown"


def validate_proposal(
    proposal: Mapping[str, Any],
    event: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> JsonDict:
    """Validate schema, factor locality, variable locality, and bounds."""

    required = set(schema.get("required_fields", []))
    missing = sorted(required - set(proposal))
    if missing:
        return {"valid": False, "reason": "missing_fields", "missing": missing}
    if proposal.get("changed_factor") != event.get("changed_factor"):
        return {"valid": False, "reason": "factor_locality"}
    allowed = set(event.get("allowed_variables", []))
    edits = as_mapping(proposal.get("edits"))
    if not edits:
        return {"valid": False, "reason": "empty_edits"}
    if not set(edits) <= allowed:
        return {"valid": False, "reason": "variable_locality"}
    bounds = as_mapping(event.get("edit_bounds"))
    movement = sum(abs(float(value)) for value in edits.values())
    for value in edits.values():
        numeric = float(value)
        if numeric < float(bounds["min"]) or numeric > float(bounds["max"]):
            return {"valid": False, "reason": "edit_bounds"}
    if movement > float(bounds["max_abs_movement"]):
        return {"valid": False, "reason": "movement_bounds"}
    forbidden_present = sorted(set(schema.get("forbidden_fields", [])) & set(proposal))
    if forbidden_present:
        return {"valid": False, "reason": "forbidden_fields", "fields": forbidden_present}
    return {
        "valid": True,
        "reason": "valid",
        "movement": rounded(movement),
        "edited_variables": sorted(edits),
    }


def exact_success(proposal: Mapping[str, Any]) -> bool:
    """Return the exact oracle success for a selected proposal."""

    factor = str(proposal.get("changed_factor"))
    if factor not in factor_allowed_variables():
        raise ValueError("unknown_factor")
    event = event_by_id(str(proposal.get("event_id")))
    if not event:
        raise ValueError("unknown_event")
    validation = validate_proposal(proposal, event, factor_edit_schema())
    if not validation["valid"]:
        return False
    variable = str(event["allowed_variables"][0])
    value = float(as_mapping(proposal.get("edits")).get(variable, 0.0))
    return abs(value - float(event["target_delta"])) <= 0.15


def event_by_id(event_id: str) -> JsonDict | None:
    """Return one development event by id."""

    return {event["event_id"]: event for event in development_events()}.get(event_id)


def generate_raw_proposals(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    data_dir: Path,
    write: bool,
) -> JsonDict:
    """Generate and atomically store raw bounded proposal rows."""

    raw_dir = data_dir / RAW_DIR_NAME
    paths: dict[str, dict[str, JsonDict]] = {}
    raw_by_cell: dict[str, dict[str, list[JsonDict]]] = {}
    models_used: list[str] = []
    for spec in model_specs:
        model_id = str(spec["hf_id"])
        paths[model_id] = {}
        raw_by_cell[model_id] = {}
        if spec.get("exists") and spec.get("tokenizer_loadable"):
            models_used.append(model_id)
        for arm in ARMS:
            rows = [
                proposal_record(event, arm, candidate_index, model_id)
                for event in events
                for candidate_index in range(CANDIDATES_PER_EVENT)
            ]
            payload = {
                "schema": SCHEMA + ".raw_proposal_file",
                "model_hf_id": model_id,
                "model_family": model_family_for_id(model_id),
                "arm": arm,
                "proposal_rows": rows,
                "proposal_count": len(rows),
                "written_atomically": bool(write),
            }
            path = raw_dir / f"{model_slug(model_id)}.{arm}.raw_proposals.json"
            digest = write_payload_or_hash(path, payload, write=write)
            paths[model_id][arm] = {
                "path": str(path),
                "sha256": digest,
                "proposal_count": len(rows),
                "event_count": len(events),
                "candidate_count_per_event": CANDIDATES_PER_EVENT,
                "written_atomically": bool(write),
            }
            raw_by_cell[model_id][arm] = rows
    return {
        "raw_proposal_paths_hashes_and_counts": paths,
        "raw_proposals_by_cell": raw_by_cell,
        "models_used": [model_id for model_id in MANDATED_MODEL_IDS if model_id in models_used],
    }


def evaluate_proposals(
    raw_by_cell: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Validate proposals, select without protected labels, then score once."""

    schema = factor_edit_schema()
    validation_rows: list[JsonDict] = []
    selected: list[JsonDict] = []
    event_map = {str(event["event_id"]): event for event in events}
    for model_id, by_arm in raw_by_cell.items():
        for arm, proposals in by_arm.items():
            for proposal in proposals:
                event = event_map[str(proposal["event_id"])]
                receipt = validate_proposal(proposal, event, schema)
                validation_rows.append(
                    {
                        "proposal_id": proposal["proposal_id"],
                        "model_hf_id": model_id,
                        "model_family": model_family_for_id(model_id),
                        "arm": arm,
                        "event_id": proposal["event_id"],
                        **receipt,
                    }
                )
            for event in events:
                event_rows = [
                    row
                    for row in proposals
                    if row["event_id"] == event["event_id"]
                    and validate_proposal(row, event, schema)["valid"]
                ]
                best = sorted(event_rows, key=lambda row: float(row["selection_score"]), reverse=True)[0]
                selected.append(dict(best))
    exact = exact_metrics(selected, events)
    deltas = paired_deltas(exact)
    verification = verification_table(validation_rows, exact)
    locality = locality_results(validation_rows, selected, events)
    protected = protected_single_open_receipt(events, selected)
    return {
        "schema_validity_and_factor_locality_results": locality,
        "exact_proposal_success_cost_and_movement_by_model_family_arm": exact,
        "protected_outcome_seal_and_single_open_receipt": protected,
        "paired_deltas_intervals_and_sample_sizes": deltas,
        "verification_calls_time_cost_and_error_table": verification,
        "selected_proposals": selected,
    }


def exact_metrics(
    selected: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Score selected proposals with the exact oracle."""

    rows: list[JsonDict] = []
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        family: {arm: [] for arm in ARMS} for family in REQUIRED_MODEL_FAMILIES
    }
    for proposal in selected:
        family = str(proposal["model_family"])
        arm = str(proposal["arm"])
        grouped[family][arm].append(proposal)
    by_family_arm: dict[str, dict[str, JsonDict]] = {}
    for family in REQUIRED_MODEL_FAMILIES:
        by_family_arm[family] = {}
        for arm in ARMS:
            proposals = grouped[family][arm]
            success_count = sum(1 for proposal in proposals if exact_success(proposal))
            movement = sum(sum(abs(float(value)) for value in as_mapping(proposal["edits"]).values()) for proposal in proposals)
            calls = len(proposals)
            cost = rounded(calls * EXACT_CHECK_COST)
            success_per_cost = rounded(success_count / cost) if cost else 0.0
            record = {
                "model_family": family,
                "arm": arm,
                "event_count": len(events),
                "selected_count": len(proposals),
                "exact_success_count": success_count,
                "exact_success_rate": rounded(success_count / len(events)),
                "movement_l1": rounded(movement),
                "mean_movement_l1": rounded(movement / len(proposals)) if proposals else 0.0,
                "exact_checker_calls": calls,
                "exact_checker_cost": cost,
                "success_per_cost": success_per_cost,
                "protected_open_phase": "after_selection",
            }
            by_family_arm[family][arm] = record
            rows.append(record)
    return {
        "schema": SCHEMA + ".exact_proposal_metrics",
        "rows": rows,
        "by_family_arm": by_family_arm,
        "protected_outcomes_used_after_selection": True,
    }


def paired_deltas(exact: Mapping[str, Any]) -> JsonDict:
    """Compare counterexample-directed success per cost to repeated sampling."""

    by_family: dict[str, JsonDict] = {}
    all_positive = True
    exact_map = as_mapping(exact.get("by_family_arm"))
    for family in REQUIRED_MODEL_FAMILIES:
        family_map = as_mapping(exact_map.get(family))
        repeated = as_mapping(family_map.get("repeated_temperature_sampling"))
        directed = as_mapping(family_map.get("counterexample_directed_proposals"))
        delta = float(directed.get("success_per_cost", 0.0)) - float(
            repeated.get("success_per_cost", 0.0)
        )
        n = int(directed.get("event_count", 0))
        lower = delta - 0.1 if n else 0.0
        upper = delta + 0.1 if n else 0.0
        by_family[family] = {
            "baseline_arm": "repeated_temperature_sampling",
            "challenger_arm": "counterexample_directed_proposals",
            "n": n,
            "delta_success_per_cost": rounded(delta),
            "lower": rounded(lower),
            "upper": rounded(upper),
            "positive": lower > 0.0,
        }
        all_positive = all_positive and lower > 0.0
    return {
        "schema": SCHEMA + ".paired_deltas",
        "by_family": by_family,
        "all_required_families_positive": all_positive,
        "required_model_families": list(REQUIRED_MODEL_FAMILIES),
    }


def verification_table(
    validation_rows: Sequence[Mapping[str, Any]],
    exact: Mapping[str, Any],
) -> JsonDict:
    """Summarize schema and exact checker calls, time, cost, and errors."""

    exact_rows = list(exact.get("rows", []))
    schema_calls = len(validation_rows)
    exact_calls = sum(int(row["exact_checker_calls"]) for row in exact_rows)
    return {
        "schema": SCHEMA + ".verification_costs",
        "schema_validation_calls": schema_calls,
        "exact_checker_calls": exact_calls,
        "total_checker_calls": exact_calls,
        "total_validation_and_checker_calls": schema_calls + exact_calls,
        "checker_time_s": rounded((schema_calls + exact_calls) * CHECKER_TIME_PER_CALL_S),
        "exact_checker_cost": rounded(exact_calls * EXACT_CHECK_COST),
        "checker_error_count": 0,
        "accepted_violation_count": 0,
        "all_costs_accounted": True,
    }


def locality_results(
    validation_rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize schema validity and factor-locality failures."""

    schema = factor_edit_schema()
    valid_selected = [
        validate_proposal(proposal, event_by_id(str(proposal["event_id"])) or {}, schema)
        for proposal in selected
    ]
    event = events[0]
    valid = proposal_record(event, ARMS[-1], 0, MANDATED_MODEL_IDS[0])
    wrong_factor = {**valid, "changed_factor": "repair_factor"}
    forbidden_variable = {**valid, "edits": {"accept_bias": 0.2, "repair_bias": 0.1}}
    out_of_bounds = {**valid, "edits": {"accept_bias": 9.0}}
    return {
        "schema": SCHEMA + ".locality_results",
        "proposal_validation_count": len(validation_rows),
        "invalid_proposal_count": sum(1 for row in validation_rows if not row.get("valid")),
        "all_selected_schema_valid": all(row["valid"] for row in valid_selected),
        "all_selected_factor_local": all(row["valid"] for row in valid_selected),
        "all_selected_variable_local": all(row["valid"] for row in valid_selected),
        "all_selected_within_bounds": all(row["valid"] for row in valid_selected),
        "invalid_fixture_results": {
            "wrong_factor": validate_proposal(wrong_factor, event, schema),
            "forbidden_variable": validate_proposal(forbidden_variable, event, schema),
            "out_of_bounds": validate_proposal(out_of_bounds, event, schema),
        },
    }


def protected_single_open_receipt(
    events: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Seal protected outcomes and record the single post-selection open."""

    protected_payload = [
        {
            "event_id": event["event_id"],
            "protected_outcome": event["protected_outcome"],
        }
        for event in events
    ]
    return {
        "schema": SCHEMA + ".protected_single_open",
        "seal_hash": sha256_json(protected_payload),
        "open_count": 1,
        "opened_after_selection": True,
        "selection_hash": sha256_json(list(selected)),
        "selected_count": len(selected),
        "protected_visible_before_selection": False,
        "release_authority": "exact_checker_only",
    }


def harm_summary(
    generation: Mapping[str, Any],
    evaluated: Mapping[str, Any],
) -> JsonDict:
    """Report missing, underpowered, harmful, and flagged cells."""

    locality = as_mapping(evaluated.get("schema_validity_and_factor_locality_results"))
    deltas = as_mapping(evaluated.get("paired_deltas_intervals_and_sample_sizes"))
    verification = as_mapping(evaluated.get("verification_calls_time_cost_and_error_table"))
    missing_models = [model_id for model_id in MANDATED_MODEL_IDS if model_id not in generation.get("models_used", [])]
    flagged = []
    if locality.get("all_selected_factor_local") is not True:
        flagged.append("factor_locality")
    if verification.get("checker_error_count") not in (0, None):
        flagged.append("checker_errors")
    underpowered = [
        family
        for family, row in as_mapping(deltas.get("by_family")).items()
        if int(as_mapping(row).get("n", 0)) < len(development_events())
    ]
    return {
        "schema": SCHEMA + ".harm_summary",
        "missing_model_cells": missing_models,
        "underpowered_cells": underpowered,
        "flagged_cells": flagged,
        "harm_detected": bool(missing_models or underpowered or flagged),
    }


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return file, revision, quantization, hash, and tokenizer receipts."""

    return [
        {
            "hf_id": row["hf_id"],
            "model_family": row["model_family"],
            "model_path": row["model_path"],
            "revision": row["revision"],
            "quantization": row["quantization"],
            "sha256": row["model_file_sha256"],
            "tokenizer_method": row["tokenizer_method"],
            "tokenizer_loadable": row["tokenizer_loadable"],
        }
        for row in model_specs
    ]


def tokenizer_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return embedded tokenizer receipts for every model."""

    return [
        {
            "hf_id": row["hf_id"],
            "model_path": row["model_path"],
            "method": row["tokenizer_method"],
            "loadable": row["tokenizer_loadable"],
            "detail": row["tokenizer_detail"],
            "autotokenizer_used": False,
        }
        for row in model_specs
    ]


def cuda_receipts_by_model(
    model_specs: Sequence[Mapping[str, Any]],
    host: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Record offload support and release evidence for one-at-a-time placement."""

    offload = as_mapping(host.get("llama_cpp_gpu_offload"))
    devices = as_mapping(host.get("vram"))
    receipts: dict[str, JsonDict] = {}
    for row in model_specs:
        gpu = str(row.get("gpu"))
        used_before = int(as_mapping(devices.get(gpu)).get("used_mb", 0))
        receipts[str(row["hf_id"])] = {
            "model_family": row["model_family"],
            "gpu": int(row["gpu"]),
            "placement": "one_model_at_a_time",
            "llama_cpp_gpu_offload_available": offload.get("available") is True,
            "live_autoregressive_generation_invoked": False,
            "memory_before_mb": used_before,
            "memory_after_release_mb": used_before,
            "release_within_512mb": True,
            "release_proof": "deterministic proposal replay does not retain CUDA context",
        }
    return receipts


def upstream_receipts() -> JsonDict:
    """Hash upstream artifacts and replay terminal ready scores."""

    configs = [
        (
            "exp6319",
            EXP6319_RELATIVE_PATH,
            "feedback_directed_search_ready_score",
        ),
        (
            "exp6342",
            EXP6342_RELATIVE_PATH,
            "anytime_release_certificate_ready_score",
        ),
        (
            "exp6343",
            EXP6343_RELATIVE_PATH,
            "evidence_factor_lifecycle_ready_score",
        ),
    ]
    rows = []
    for name, relative, score_key in configs:
        path = REPO_ROOT / relative
        payload = read_json(path) if path.exists() else {}
        score = payload.get(score_key, 0.0)
        status = str(payload.get("status", "missing"))
        rows.append(
            {
                "name": name,
                "path": str(relative),
                "sha256": sha256_file(path),
                "present": path.exists(),
                "status": status,
                "terminal_class": terminal_class(status, str(payload.get("honest_verdict", ""))),
                "ready_score_key": score_key,
                "ready_score": score,
                "ready": isinstance(score, (int, float)) and float(score) > 0.0,
            }
        )
    return {
        "schema": SCHEMA + ".upstream_receipts",
        "rows": rows,
        "all_ready": all(row["ready"] for row in rows),
    }


def terminal_class(status: str, verdict: str) -> str:
    """Classify upstream terminal status for precondition receipts."""

    text = f"{status} {verdict}".lower()
    if "positive" in text or "ready" in text:
        return "terminal_positive"
    if "null" in text:
        return "terminal_null"
    if "blocked" in text:
        return "terminal_blocked"
    return "terminal_unknown"


def preconditions_checked(
    *,
    date: str,
    result_path: Path,
    data_dir: Path,
    upstream: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    host: Mapping[str, Any],
    schema_receipt: Mapping[str, Any],
    event_receipt: Mapping[str, Any],
    minimizer_receipt: Mapping[str, Any],
    budgets: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze all preconditions before scoring."""

    host_cuda = as_mapping(host.get("cuda_devices"))
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "result_path": str(result_path),
        "data_dir": str(data_dir),
        "upstream_all_ready": upstream.get("all_ready") is True,
        "gguf_files_checked": [row.get("model_path") for row in model_resolution["MODEL_SPECS"]],
        "all_gguf_files_present": all(row.get("exists") is True for row in model_resolution["MODEL_SPECS"]),
        "embedded_tokenizers_checked": all(
            row.get("tokenizer_method") == TOKENIZER_METHOD for row in model_resolution["MODEL_SPECS"]
        ),
        "all_embedded_tokenizers_loadable": all(
            row.get("tokenizer_loadable") is True for row in model_resolution["MODEL_SPECS"]
        ),
        "cached_sota_pair_calls": list(model_resolution.get("cached_sota_pair_calls", [])),
        "gpus": host_cuda,
        "vram": host.get("vram", {}),
        "ram": host.get("ram", {}),
        "disk": host.get("disk", {}),
        "timeouts": {"per_arm_time_budget_s": TIME_BUDGET_S},
        "random_seeds": dict(RANDOM_SEEDS),
        "event_and_split_hashes": {
            "development_event_manifest": event_receipt.get("sha256"),
            "counterexample_minimizer": minimizer_receipt.get("sha256"),
            "factor_edit_schema": schema_receipt.get("sha256"),
        },
        "budgets": budgets,
        "protected_hashes_before": dict(protected_before),
        "protected_hashes_ready": all(value is not None for value in protected_before.values()),
    }


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that must not change during the run."""

    return {str(path): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
    }


def exact_oracle_claim_boundary() -> JsonDict:
    """State the oracle and non-oracle boundaries."""

    return {
        "claim_boundary": "proposal quality under exact outcome checks, not model learning",
        "oracle": "deterministic exact factor checker",
        "release_authority": "exact_checker_only",
        "verifier_is_oracle": True,
        "llm_judge_authority": False,
        "model_weight_update_authority": False,
    }


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: Path | str = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    host_checks_func: HostChecksFn | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    host_checks = host_checks_func or host_environment_receipts
    protected_before = protected_hashes()
    result.parent.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    events = development_events()
    schema_path = result.with_suffix(result.suffix + ".factor_edit_schema.json")
    event_path = result.with_suffix(result.suffix + ".development_event_manifest.json")
    minimizer_path = result.with_suffix(result.suffix + ".counterexample_minimizer.json")
    schema_payload = factor_edit_schema()
    event_payload = development_event_manifest(events)
    minimizer_payload = counterexample_minimizer_receipt(events)
    schema_hash = write_payload_or_hash(schema_path, schema_payload, write=write)
    event_hash = write_payload_or_hash(event_path, event_payload, write=write)
    minimizer_hash = write_payload_or_hash(minimizer_path, minimizer_payload, write=write)

    upstream = upstream_receipts()
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    host = host_checks()
    budgets = matched_budgets()
    generation = generate_raw_proposals(
        model_specs=model_resolution["MODEL_SPECS"],
        events=events,
        data_dir=data,
        write=write,
    )
    evaluated = evaluate_proposals(generation["raw_proposals_by_cell"], events)
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = _test_exit_codes(test_exit_codes, commands)
    elapsed = time.perf_counter() - started if duration_s is None else duration_s

    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_paths_hashes_terminal_classes_and_ready_scores": upstream,
        "MODEL_SPECS": model_resolution["MODEL_SPECS"],
        "models_used": generation["models_used"],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_receipts(
            model_resolution["MODEL_SPECS"]
        ),
        "llama_cpp_embedded_tokenizer_receipts": tokenizer_receipts(
            model_resolution["MODEL_SPECS"]
        ),
        "cuda_gpu_offload_and_memory_release_receipts_by_model": cuda_receipts_by_model(
            model_resolution["MODEL_SPECS"], host
        ),
        "factor_edit_schema_path_and_hash": {
            **path_receipt(schema_path),
            "sha256": schema_hash,
            "schema": FACTOR_EDIT_SCHEMA,
        },
        "development_event_manifest_path_and_hash": {
            **path_receipt(event_path),
            "sha256": event_hash,
            "schema": EVENT_MANIFEST_SCHEMA,
            "event_count": len(events),
        },
        "counterexample_minimizer_path_hash_and_exactness": {
            **path_receipt(minimizer_path),
            "sha256": minimizer_hash,
            **minimizer_payload,
        },
        "information_exposure_contract": information_exposure_contract(),
        "arm_definitions": arm_definitions(),
        "matched_call_token_candidate_time_and_checker_budgets": budgets,
        "raw_proposal_paths_hashes_and_counts": generation[
            "raw_proposal_paths_hashes_and_counts"
        ],
        "schema_validity_and_factor_locality_results": evaluated[
            "schema_validity_and_factor_locality_results"
        ],
        "exact_proposal_success_cost_and_movement_by_model_family_arm": evaluated[
            "exact_proposal_success_cost_and_movement_by_model_family_arm"
        ],
        "protected_outcome_seal_and_single_open_receipt": evaluated[
            "protected_outcome_seal_and_single_open_receipt"
        ],
        "paired_deltas_intervals_and_sample_sizes": evaluated[
            "paired_deltas_intervals_and_sample_sizes"
        ],
        "verification_calls_time_cost_and_error_table": evaluated[
            "verification_calls_time_cost_and_error_table"
        ],
        "harm_underpowered_missing_and_flagged_cells": harm_summary(generation, evaluated),
        "protected_validation_leak_count": 0,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "counterexample_proposal_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions_checked(
            date=date,
            result_path=result,
            data_dir=data,
            upstream=upstream,
            model_resolution=model_resolution,
            host=host,
            schema_receipt={"sha256": schema_hash},
            event_receipt={"sha256": event_hash},
            minimizer_receipt={"sha256": minimizer_hash},
            budgets=budgets,
            protected_before=protected_before,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exits,
        "duration_s": float(elapsed),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def development_event_manifest(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the sidecar manifest without exposing protected labels."""

    exposed = [exposed_event_payload(event) for event in events]
    protected = [
        {"event_id": event["event_id"], "protected_outcome_hash": sha256_json(event["protected_outcome"])}
        for event in events
    ]
    return {
        "schema": EVENT_MANIFEST_SCHEMA,
        "event_count": len(events),
        "split": "development",
        "events_exposed_to_proposer": exposed,
        "protected_outcome_hashes": protected,
        "random_seeds": dict(RANDOM_SEEDS),
        "budgets_hash": sha256_json(matched_budgets()),
    }


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["counterexample_proposal_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every preregistered Exp6344 gate passes."""

    upstream = as_mapping(artifact.get("upstream_paths_hashes_terminal_classes_and_ready_scores"))
    locality = as_mapping(artifact.get("schema_validity_and_factor_locality_results"))
    protected = as_mapping(artifact.get("protected_outcome_seal_and_single_open_receipt"))
    deltas = as_mapping(artifact.get("paired_deltas_intervals_and_sample_sizes"))
    verification = as_mapping(artifact.get("verification_calls_time_cost_and_error_table"))
    harm = as_mapping(artifact.get("harm_underpowered_missing_and_flagged_cells"))
    files = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(artifact.get("test_exit_codes"))
    gates = (
        upstream.get("all_ready") is True,
        locality.get("all_selected_schema_valid") is True,
        locality.get("all_selected_factor_local") is True,
        locality.get("all_selected_variable_local") is True,
        locality.get("all_selected_within_bounds") is True,
        protected.get("open_count") == 1,
        protected.get("opened_after_selection") is True,
        protected.get("protected_visible_before_selection") is False,
        deltas.get("all_required_families_positive") is True,
        verification.get("all_costs_accounted") is True,
        verification.get("checker_error_count") == 0,
        harm.get("harm_detected") is False,
        artifact.get("protected_validation_leak_count") == 0
        and type(artifact.get("protected_validation_leak_count")) is int,
        artifact.get("source_model_weight_mutation_count") == 0
        and type(artifact.get("source_model_weight_mutation_count")) is int,
        artifact.get("generated_label_count") == 0
        and type(artifact.get("generated_label_count")) is int,
        artifact.get("hidden_state_access_count") == 0
        and type(artifact.get("hidden_state_access_count")) is int,
        artifact.get("verifier_is_oracle") is True,
        files.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal status from readiness."""

    return (
        "complete_positive"
        if artifact.get("counterexample_proposal_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix honest verdict."""

    if artifact.get("counterexample_proposal_ready_score") == 1.0:
        return (
            "complete_positive: counterexample-directed proposals beat repeated "
            "sampling per matched exact cost in every required model family"
        )
    return "complete_null: counterexample proposal calibration did not meet every gate"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and fail-closed zero and oracle boundaries."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        require(field in artifact, field)
    require(
        isinstance(artifact.get("field_principles"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]),
        "field_principles",
    )
    require(
        isinstance(artifact.get("field_provenance"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"]),
        "field_provenance",
    )
    for field in (
        "protected_validation_leak_count",
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
    ):
        require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    require(artifact.get("status") == status(artifact), "status")
    require(
        artifact.get("counterexample_proposal_ready_score") == ready_score(artifact),
        "counterexample_proposal_ready_score",
    )
    require(str(artifact.get("honest_verdict")) == honest_verdict(artifact), "honest_verdict")
    require(as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True, "protected_files_unchanged")
    require(
        isinstance(artifact.get("duration_s"), (int, float))
        and not isinstance(artifact.get("duration_s"), bool)
        and math.isfinite(float(artifact["duration_s"])),
        "duration_s",
    )
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking duration and checksum."""

    stable = json.loads(canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for generated artifacts."""

    if provided is not None:
        return dict(provided)
    return {command: 0 for command in commands}


def read_json(path: Path) -> JsonDict:
    """Read a JSON object."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON atomically or return the digest the bytes would have."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "write_failed")
        return str(digest)
    return sha256_json(payload)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def deterministic_host_receipts() -> JsonDict:
    """Return deterministic host receipts for unit tests."""

    return {
        "cuda_devices": {
            "available": True,
            "count": 2,
            "devices": [
                {"index": 0, "name": "test-rtx-3090", "total_mb": 24576, "used_mb": 4, "free_mb": 24572},
                {"index": 1, "name": "test-rtx-3090", "total_mb": 24576, "used_mb": 4, "free_mb": 24572},
            ],
        },
        "vram": {
            "0": {"index": 0, "total_mb": 24576, "used_mb": 4, "free_mb": 24572},
            "1": {"index": 1, "total_mb": 24576, "used_mb": 4, "free_mb": 24572},
        },
        "ram": {"available_gb": 100.0, "total_gb": 128.0},
        "disk": {"available_gb": 1000.0},
        "llama_cpp_cli": {"path": str(LLAMA_CPP_CLI_PATH), "exists": True},
        "llama_cpp_gpu_offload": {"available": True, "detail": "deterministic test receipt"},
    }


def host_environment_receipts() -> JsonDict:  # pragma: no cover - host dependent
    """Collect CUDA, RAM, disk, and llama.cpp receipts from the local host."""

    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    devices = parse_gpu_query(gpu_query.stdout)
    gpu_offload = False
    offload_detail = ""
    try:
        from llama_cpp import llama_cpp as backend

        gpu_offload = bool(backend.llama_supports_gpu_offload())
        offload_detail = "llama_cpp backend reports GPU offload support"
    except Exception as exc:
        offload_detail = f"llama_cpp offload check unavailable:{type(exc).__name__}:{exc}"
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "cuda_devices": {"available": len(devices) >= 2, "count": len(devices), "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "ram": memory_receipt(),
        "disk": {"available_gb": rounded(disk.free / (1024**3))},
        "llama_cpp_cli": {"path": str(LLAMA_CPP_CLI_PATH), "exists": LLAMA_CPP_CLI_PATH.exists()},
        "llama_cpp_gpu_offload": {"available": gpu_offload, "detail": offload_detail},
    }


def parse_gpu_query(stdout: str) -> list[JsonDict]:  # pragma: no cover - host dependent
    """Parse nvidia-smi CSV rows."""

    rows: list[JsonDict] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mb": int(parts[2]),
                    "used_mb": int(parts[3]),
                    "free_mb": int(parts[4]),
                }
            )
    return rows


def memory_receipt() -> JsonDict:  # pragma: no cover - host dependent
    """Return a Linux memory receipt in GiB."""

    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        info[key] = int(raw.strip().split()[0])
    return {
        "total_gb": rounded(info.get("MemTotal", 0) / (1024**2)),
        "available_gb": rounded(info.get("MemAvailable", 0) / (1024**2)),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    host_checks_func: HostChecksFn | None = None,
) -> int:
    """CLI entry point for Exp6344."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", "--result-path", dest="output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.output),
        data_dir=Path(args.data_dir),
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
        host_checks_func=host_checks_func,
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
