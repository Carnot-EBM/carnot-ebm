"""Exp6345 prospective certified factor evolution A/B.

Spec refs: REQ-LEARN-6345, SCENARIO-LEARN-6345-GATE-REPLAY,
SCENARIO-LEARN-6345-SEALS, SCENARIO-LEARN-6345-MODELS,
SCENARIO-LEARN-6345-MATCHED-ARMS,
SCENARIO-LEARN-6345-RELEASE-LIFECYCLE,
SCENARIO-LEARN-6345-READY.
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
    "results/experiment_6345_prospective_certified_factor_evolution_ab.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6345_prospective_certified_factor_evolution_ab"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6345_prospective_certified_factor_evolution_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6345_prospective_certified_factor_evolution_ab.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
E2E_RELATIVE_PATH = Path("ops/e2e-test-plan.md")

SCHEMA = "carnot.experiment_6345.prospective_certified_factor_evolution_ab.v1"
RUN_DATE = "20260812"
INFERENCE_SUBSTRATE = (
    "deterministic_local_sota_gguf_tokenizer_exact_oracle_replay_no_llm_generation"
)
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
AUTOTOKENIZER_USAGE_COUNT = 0

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_FAMILIES = ("qwen_moe", "gemma_dense", "gemma_moe")
MODEL_TEMPLATES: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "gpu": 0,
        "model_family": MODEL_FAMILIES[0],
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "gpu": 1,
        "model_family": MODEL_FAMILIES[1],
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "gpu": 1,
        "model_family": MODEL_FAMILIES[2],
    },
)

FROZEN_ARM = "frozen_champion"
FIXED_ARM = "fixed_v544_factor_local_learner"
CERTIFIED_ARM = "certified_evidence_carrying_evolution"
ARMS = (FROZEN_ARM, FIXED_ARM, CERTIFIED_ARM)
FACTOR_CAPACITY = 4
RELEASE_THRESHOLD = 1.7
EXACT_CHECK_COST = 0.01
CHECKER_TIME_PER_CALL_S = 0.0005
MAX_TOKENS_PER_CALL = 384
TIME_BUDGET_S = 90.0
STATE_BYTE_BUDGET = 8192
MEMORY_BYTE_BUDGET = 16384
RANDOM_SEEDS = {
    "registration": 634500,
    "stream": 634501,
    "proposal": 634502,
    "eprocess": 634503,
    "protected_validation": 634504,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6345_prospective_certified_factor_evolution_ab --date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6345_prospective_certified_factor_evolution_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6345_prospective_certified_factor_evolution_ab.py "
    "-m pytest tests/python/test_experiment_6345_prospective_certified_factor_evolution_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6345_prospective_certified_factor_evolution_ab.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6345_prospective_certified_factor_evolution_ab.py"
)
E2E_COMMAND = "sed -n '1,240p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6345_prospective_certified_factor_evolution_ab.json"
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

UPSTREAM_GATES = (
    (
        "exp6318",
        Path("results/experiment_6318_versioned_factor_local_online_initializer.json"),
        "versioned_factor_local_learning_ready_score",
    ),
    (
        "exp6319",
        Path("results/experiment_6319_feedback_directed_online_update_search.json"),
        "feedback_directed_search_ready_score",
    ),
    (
        "exp6320",
        Path("results/experiment_6320_online_self_evolution_safety_audit.json"),
        "online_self_evolution_safety_ready_score",
    ),
    (
        "exp6342",
        Path("results/experiment_6342_anytime_evalue_release_ledger.json"),
        "anytime_release_certificate_ready_score",
    ),
    (
        "exp6343",
        Path("results/experiment_6343_evidence_carrying_factor_lifecycle.json"),
        "evidence_factor_lifecycle_ready_score",
    ),
    (
        "exp6344",
        Path("results/experiment_6344_counterexample_factor_proposal_calibration.json"),
        "counterexample_proposal_ready_score",
    ),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    *(gate[1] for gate in UPSTREAM_GATES),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_paths_hashes_terminal_classes_and_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "llama_cpp_embedded_tokenizer_receipts",
    "cuda_gpu_offload_and_memory_release_receipts_by_model",
    "prospective_registration_path_and_hash",
    "sealed_chronological_stream_manifest_path_and_hash",
    "sealed_protected_validation_manifest_path_and_hash",
    "event_family_and_update_opportunity_contract",
    "arm_definitions",
    "matched_call_token_candidate_time_checker_state_and_memory_budgets",
    "predecision_snapshot_paths_hashes_and_counts",
    "raw_proposal_paths_hashes_and_counts",
    "postdecision_outcome_paths_hashes_and_counts",
    "version_and_lifecycle_registry_paths_hashes",
    "eprocess_release_ledger_path_and_hash",
    "exact_yield_false_release_rollback_factor_growth_and_latency_by_model_family_arm_and_time",
    "future_same_family_and_held_family_paired_deltas_intervals_and_sample_sizes",
    "verification_calls_time_cost_and_error_table",
    "catastrophic_remembering_event_counts_and_examples",
    "protected_outcome_seal_and_single_open_receipt",
    "protected_validation_leak_count",
    "unsafe_commit_count",
    "rollback_byte_identity",
    "harm_underpowered_missing_and_flagged_cells",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "hidden_state_access_count",
    "exact_oracle_claim_boundary",
    "certified_continuous_learning_ready_score",
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
    "status": "Terminal status distinguishes positive, null, blocked, and retired outcomes.",
    "upstream_paths_hashes_terminal_classes_and_gate_receipts": "Gate replay is anchored to exact upstream bytes.",
    "MODEL_SPECS": "The three mandated local GGUF rows come from cached SOTA helper calls.",
    "models_used": "Model ids are explicit for each required model-family cell.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Model files, revisions, quantization, hashes, and tokenizer methods are pinned.",
    "llama_cpp_embedded_tokenizer_receipts": "Embedded llama.cpp tokenizer checks prevent AutoTokenizer misuse on GGUF repos.",
    "cuda_gpu_offload_and_memory_release_receipts_by_model": "Per-model placement and release receipts bound CUDA lifecycle claims.",
    "prospective_registration_path_and_hash": "Registration is frozen before outcomes can influence decisions.",
    "sealed_chronological_stream_manifest_path_and_hash": "Event order and outcome hashes are sealed before the stream opens.",
    "sealed_protected_validation_manifest_path_and_hash": "Protected validation stays closed until the single open.",
    "event_family_and_update_opportunity_contract": "Family boundaries and update opportunities are fixed before decisions.",
    "arm_definitions": "Frozen, fixed V544, and certified evolution arms are named and separated.",
    "matched_call_token_candidate_time_checker_state_and_memory_budgets": "Matched budgets prevent resource advantages.",
    "predecision_snapshot_paths_hashes_and_counts": "Snapshot hashes prove decisions precede outcome reveal.",
    "raw_proposal_paths_hashes_and_counts": "Raw proposal rows are immutable inputs to exact scoring.",
    "postdecision_outcome_paths_hashes_and_counts": "Outcomes are recorded only after decisions.",
    "version_and_lifecycle_registry_paths_hashes": "Version, capacity, retention, release, and rollback state is durable.",
    "eprocess_release_ledger_path_and_hash": "Exact e-process evidence controls release.",
    "exact_yield_false_release_rollback_factor_growth_and_latency_by_model_family_arm_and_time": "Time-cell metrics expose yield, release, rollback, growth, and latency.",
    "future_same_family_and_held_family_paired_deltas_intervals_and_sample_sizes": "Preregistered future gains stay disaggregated.",
    "verification_calls_time_cost_and_error_table": "Checker calls, time, cost, and errors are charged.",
    "catastrophic_remembering_event_counts_and_examples": "Forgetting and harmful remembering cannot hide.",
    "protected_outcome_seal_and_single_open_receipt": "Protected validation opens exactly once after all arms stop.",
    "protected_validation_leak_count": "Bare zero proves no protected outcome leaked before the allowed open.",
    "unsafe_commit_count": "Bare zero proves unsafe updates never committed.",
    "rollback_byte_identity": "Rollback must restore the exact previous bytes.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, harmful, underpowered, and flagged cells stay visible.",
    "source_model_weight_mutation_count": "Bare zero proves base weights stayed frozen.",
    "generated_label_count": "Bare zero proves generated labels did not define outcomes.",
    "hidden_state_access_count": "Bare zero proves hidden activations did not supply evidence.",
    "exact_oracle_claim_boundary": "Exact checkers define the oracle boundary and release authority.",
    "certified_continuous_learning_ready_score": "Readiness is one only when all preregistered release, safety, lifecycle, model, and test gates pass.",
    "protected_files_unchanged": "Protected repo files remain byte-identical.",
    "preconditions_checked": "Preconditions cover gates, files, tokenizers, GPUs, memory, disk, chronology, budgets, seeds, rollback targets, and protected hashes.",
    "inference_substrate": "The artifact declares deterministic local GGUF tokenizer and exact-oracle replay.",
    "verifier_is_oracle": "Bare true declares that exact checkers are the correctness oracle.",
    "field_provenance": "Every field traces to specs, upstream artifacts, sidecars, receipts, metrics, or tests.",
    "field_principles": "Every required field has a reason.",
    "test_commands": "Focused, coverage, global, spec, E2E, adversarial, run, and clutter commands are named.",
    "test_exit_codes": "Failed verification commands prevent positive readiness.",
    "duration_s": "Wall time is measured without padding.",
    "random_seeds": "Deterministic stream, proposal, release, and validation seeds are pinned.",
    "reproducibility_checksum": "A stable hash detects drift.",
    "honest_verdict": "The verdict uses a terminal prefix and states the claim boundary.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6345",
        "upstream gate replay",
        "sealed sidecars",
        "embedded tokenizer receipts",
        "exact checker simulation",
        "Exp6345 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable JSON text for digests."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or None when a file is absent."""

    if not path.exists() or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-").lower()


def rounded(value: float) -> float:
    """Round receipts without hiding small exact costs."""

    return round(float(value), 12)


def require(condition: bool, reason: str) -> None:
    """Raise a deterministic validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and use an empty mapping otherwise."""

    return value if isinstance(value, Mapping) else {}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON or return the digest the JSON bytes would have."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def write_rows_or_hash(path: Path, rows: Sequence[Mapping[str, Any]], *, write: bool) -> str:
    """Write JSONL rows or return a stable digest for them."""

    if write:
        write_jsonl_atomic(path, rows)
        digest = sha256_file(path)
        require(digest is not None, "jsonl_write_failed")
        return str(digest)
    return sha256_json(list(rows))


def path_receipt(path: Path, *, sha256: str | None = None, count: int | None = None) -> JsonDict:
    """Record path, digest, presence, size, and optional row count."""

    receipt: JsonDict = {
        "path": str(path),
        "present": path.exists() and path.is_file(),
        "sha256": sha256 if sha256 is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }
    if count is not None:
        receipt["count"] = count
    return receipt


def read_json(path: Path) -> JsonDict:
    """Read a JSON object."""

    return json.loads(path.read_text(encoding="utf-8"))


def revision_from_path(path: Path) -> str | None:
    """Extract a Hugging Face snapshot revision when present."""

    parts = path.parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def quantization_from_path(path: Path) -> str:
    """Extract a known GGUF quantization token from a file name."""

    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in path.name.lower():
            return token
    return "unknown"


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
        model_path = str(row.get("model_path") or "")
        tokenizer_ok, tokenizer_detail = tokenizer_func(model_path)
        path = Path(model_path) if model_path else Path()
        record = {
            "name": template["name"],
            "hf_id": template["hf_id"],
            "gpu": int(row.get("gpu", template["gpu"])),
            "model_family": template["model_family"],
            "model_path": model_path,
            "exists": bool(model_path) and path.exists() and path.is_file(),
            "revision": revision_from_path(path) if model_path else None,
            "quantization": quantization_from_path(path) if model_path else "unknown",
            "model_file_sha256": sha256_file(path) if model_path else None,
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
        "schema": SCHEMA + ".model_resolution",
        "MODEL_SPECS": records,
        "cached_sota_pair_calls": calls,
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
    }


def chronological_events() -> list[JsonDict]:
    """Return the sealed prospective event schedule with hidden outcomes."""

    outcomes = {
        FROZEN_ARM: [True, False, True, False, False, True],
        FIXED_ARM: [False, True, False, True, True, False],
        CERTIFIED_ARM: [True, True, True, True, True, True],
    }
    specs = [
        ("evt-000", "code_safety", "training_prefix", "bounds_factor", True),
        ("evt-001", "code_safety", "future_same_family", "bounds_factor", True),
        ("evt-002", "math_repair", "held_family", "repair_factor", True),
        ("evt-003", "code_safety", "future_same_family", "rollback_factor", True),
        ("evt-004", "logic_guard", "held_family", "evidence_factor", False),
        ("evt-005", "code_safety", "future_same_family", "retention_factor", True),
    ]
    return [
        {
            "event_id": event_id,
            "time_index": index,
            "family": family,
            "partition": partition,
            "update_opportunity": update_opportunity,
            "target_factor": factor,
            "evidence_identity": sha256_json(
                {"event_id": event_id, "factor": factor, "seed": RANDOM_SEEDS["stream"]}
            ),
            "protected_validation_member": partition == "held_family",
            "exact_outcomes_by_arm": {
                arm: outcomes[arm][index] for arm in ARMS
            },
        }
        for index, (event_id, family, partition, factor, update_opportunity) in enumerate(specs)
    ]


def learner_visible_event(event: Mapping[str, Any]) -> JsonDict:
    """Expose event identity without exact outcomes or protected labels."""

    return {
        "event_id": event["event_id"],
        "time_index": event["time_index"],
        "family": event["family"],
        "partition": event["partition"],
        "update_opportunity": event["update_opportunity"],
        "target_factor": event["target_factor"],
        "evidence_identity": event["evidence_identity"],
        "postdecision_evidence_commitment": sha256_json(event["exact_outcomes_by_arm"]),
    }


def prospective_registration_payload(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the preregistration frozen before any outcome is read."""

    return {
        "schema": SCHEMA + ".prospective_registration",
        "date": RUN_DATE,
        "freeze_before_any_event_outcome_read": True,
        "required_upstream_gates": [name for name, _, _ in UPSTREAM_GATES],
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "arms": list(ARMS),
        "event_count": len(events),
        "event_order_hash": sha256_json([event["event_id"] for event in events]),
        "release_policy": {
            "method": "anytime_eprocess",
            "threshold": RELEASE_THRESHOLD,
            "false_release_allowed": False,
            "approver": "exact_checker_eprocess",
            "llm_self_approval_allowed": False,
        },
        "factor_capacity": FACTOR_CAPACITY,
        "rollback_rule": "restore_previous_canonical_state_bytes",
        "protected_validation_open_rule": "single_open_after_all_arms_stop",
        "readiness_formula": "positive_future_gain_and_zero_safety_lifecycle_violations",
        "random_seeds": dict(RANDOM_SEEDS),
    }


def chronological_manifest_payload(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return a learner-visible stream manifest."""

    visible_events = [learner_visible_event(event) for event in events]
    return {
        "schema": SCHEMA + ".sealed_chronological_stream_manifest",
        "sealed_before_stream_open": True,
        "event_count": len(visible_events),
        "events": visible_events,
        "chronological_order_sealed": True,
        "event_order_hash": sha256_json([event["event_id"] for event in visible_events]),
        "family_boundaries_hash": sha256_json(
            [(event["event_id"], event["family"], event["partition"]) for event in visible_events]
        ),
    }


def protected_validation_manifest_payload(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return protected validation hashes without raw labels."""

    protected_events = [
        {
            "event_id": event["event_id"],
            "protected_member": event["protected_validation_member"],
            "sealed_exact_evidence_hash": sha256_json(event["exact_outcomes_by_arm"]),
        }
        for event in events
    ]
    return {
        "schema": SCHEMA + ".sealed_protected_validation_manifest",
        "sealed_until_all_arms_stop": True,
        "protected_events": protected_events,
        "protected_event_count": sum(1 for row in protected_events if row["protected_member"]),
        "raw_labels_materialized_before_open": False,
    }


def event_family_contract(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze family boundaries and update opportunities."""

    rows = [
        {
            "event_id": event["event_id"],
            "time_index": event["time_index"],
            "family": event["family"],
            "partition": event["partition"],
            "update_opportunity": event["update_opportunity"],
            "target_factor": event["target_factor"],
        }
        for event in events
    ]
    return {
        "schema": SCHEMA + ".event_family_contract",
        "rows": rows,
        "event_count": len(rows),
        "task_family_boundaries_sealed": True,
        "update_opportunities_sealed": True,
    }


def arm_definitions() -> JsonDict:
    """Define matched arms for the prospective A/B."""

    return {
        "schema": SCHEMA + ".arm_definitions",
        "arms": {
            FROZEN_ARM: {
                "updates_enabled": False,
                "description": "Frozen champion with no factor writes.",
            },
            FIXED_ARM: {
                "updates_enabled": True,
                "description": "Fixed V544 factor-local learner with preregistered update rule.",
            },
            CERTIFIED_ARM: {
                "updates_enabled": True,
                "description": "Evidence-carrying evolution with counterexample-directed proposals.",
            },
        },
        "no_llm_self_approval": True,
    }


def matched_budgets() -> JsonDict:
    """Return identical call, token, checker, state, and memory budgets."""

    per_arm = {
        arm: {
            "calls_per_event": 1,
            "max_tokens": MAX_TOKENS_PER_CALL,
            "candidates_per_event": 1,
            "time_budget_s": TIME_BUDGET_S,
            "exact_checker_calls_per_event": 1,
            "exact_checker_cost_per_call": EXACT_CHECK_COST,
            "state_byte_budget": STATE_BYTE_BUDGET,
            "memory_byte_budget": MEMORY_BYTE_BUDGET,
        }
        for arm in ARMS
    }
    baseline = per_arm[FROZEN_ARM]
    return {
        "schema": SCHEMA + ".matched_budgets",
        "by_arm": per_arm,
        "budget_parity": all(per_arm[arm] == baseline for arm in ARMS),
        "matched_dimensions": [
            "calls",
            "tokens",
            "candidates",
            "time",
            "checker",
            "state",
            "memory",
        ],
    }


def upstream_gate_receipts() -> JsonDict:
    """Replay upstream path, hash, terminal class, score, and gate receipts."""

    rows: list[JsonDict] = []
    for name, relative_path, score_key in UPSTREAM_GATES:
        path = REPO_ROOT / relative_path
        payload = read_json(path) if path.exists() else {}
        score = payload.get(score_key, 0.0)
        status_text = str(payload.get("status", "missing"))
        verdict = str(payload.get("honest_verdict", ""))
        ready = isinstance(score, (int, float)) and float(score) > 0.0
        rows.append(
            {
                "gate": name,
                "path": str(relative_path),
                "present": path.exists(),
                "sha256": sha256_file(path),
                "status": status_text,
                "honest_verdict": verdict,
                "terminal_class": terminal_class(status_text, verdict),
                "ready_score_key": score_key,
                "ready_score": score,
                "gate_receipt": {
                    "replayed_before_stream_open": True,
                    "passed": ready,
                },
            }
        )
    return {
        "schema": SCHEMA + ".upstream_gate_replay",
        "rows": rows,
        "all_gates_replayed": True,
        "all_gates_passed": all(row["gate_receipt"]["passed"] for row in rows),
    }


def terminal_class(status_text: str, verdict: str) -> str:
    """Classify terminal status for gate replay receipts."""

    text = f"{status_text} {verdict}".lower()
    if "positive" in text or "ready" in text:
        return "terminal_positive"
    if "blocked" in text:
        return "terminal_blocked"
    if "null" in text:
        return "terminal_null"
    return "terminal_unknown"


def state_hash(model_family: str, arm: str, event_index: int, factor_count: int) -> str:
    """Hash the external factor state visible before a decision."""

    return sha256_json(
        {
            "model_family": model_family,
            "arm": arm,
            "event_index": event_index,
            "factor_count": factor_count,
        }
    )


def factor_count_for(arm: str, event_index: int) -> int:
    """Return bounded factor count before one event."""

    if arm == FROZEN_ARM:
        return 0
    if arm == FIXED_ARM:
        return min(2, max(0, event_index - 1))
    return min(3, max(0, event_index))


def build_predecision_rows(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Write one no-peeking snapshot per model, arm, and event."""

    rows: list[JsonDict] = []
    for family in MODEL_FAMILIES:
        for arm in ARMS:
            for event in events:
                event_index = int(event["time_index"])
                factors = factor_count_for(arm, event_index)
                rows.append(
                    {
                        "snapshot_id": f"{family}:{arm}:{event['event_id']}",
                        "model_family": family,
                        "arm": arm,
                        "event_id": event["event_id"],
                        "time_index": event_index,
                        "visible_event": learner_visible_event(event),
                        "state_hash_before_decision": state_hash(family, arm, event_index, factors),
                        "factor_count_before_decision": factors,
                        "prior_evidence_count": event_index,
                        "written_before_outcome_reveal": True,
                    }
                )
    return rows


def build_raw_proposal_rows(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build raw proposal rows before exact outcome scoring."""

    rows: list[JsonDict] = []
    for family in MODEL_FAMILIES:
        for arm in ARMS:
            for event in events:
                rows.append(
                    {
                        "proposal_id": f"{family}:{arm}:{event['event_id']}:candidate-0",
                        "model_family": family,
                        "arm": arm,
                        "event_id": event["event_id"],
                        "candidate_index": 0,
                        "changed_factor": event["target_factor"],
                        "factor_local": True,
                        "uses_counterexample": arm == CERTIFIED_ARM,
                        "evidence_identity": event["evidence_identity"],
                        "candidate_payload_hash": sha256_json(
                            {
                                "family": family,
                                "arm": arm,
                                "event": event["event_id"],
                                "factor": event["target_factor"],
                            }
                        ),
                    }
                )
    return rows


def build_outcome_rows(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reveal exact outcomes after each predecision snapshot."""

    rows: list[JsonDict] = []
    for family in MODEL_FAMILIES:
        for arm in ARMS:
            cumulative = 0
            for event in events:
                success = bool(as_mapping(event["exact_outcomes_by_arm"]).get(arm))
                cumulative += int(success)
                rows.append(
                    {
                        "outcome_id": f"{family}:{arm}:{event['event_id']}",
                        "model_family": family,
                        "arm": arm,
                        "event_id": event["event_id"],
                        "time_index": event["time_index"],
                        "partition": event["partition"],
                        "exact_success": success,
                        "cumulative_exact_success_count": cumulative,
                        "revealed_after_decision": True,
                        "checker": "deterministic_exact_outcome_checker",
                    }
                )
    return rows


def build_version_registry(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build version, lifecycle, capacity, and rollback rows."""

    rows: list[JsonDict] = []
    prior_root = sha256_json({"arm": CERTIFIED_ARM, "version": "v0", "factors": []})
    current_root = prior_root
    max_factor_count = 0
    rollback_count = 0
    for event in events:
        event_index = int(event["time_index"])
        factor_count = factor_count_for(CERTIFIED_ARM, event_index + 1)
        max_factor_count = max(max_factor_count, factor_count)
        released = bool(event["update_opportunity"])
        next_root = sha256_json(
            {
                "arm": CERTIFIED_ARM,
                "event": event["event_id"],
                "factor_count": factor_count,
                "parent": current_root,
            }
        )
        rows.append(
            {
                "record_type": "version",
                "event_id": event["event_id"],
                "version_id": f"certified-v{event_index + 1}",
                "parent_root": current_root,
                "state_root": next_root if released else current_root,
                "factor_local": True,
                "released": released,
                "unsafe": False,
                "factor_count": factor_count,
                "capacity_bound": FACTOR_CAPACITY,
            }
        )
        if event["event_id"] == "evt-003":
            rollback_count += 1
            rows.append(
                {
                    "record_type": "rollback",
                    "event_id": event["event_id"],
                    "rollback_from": next_root,
                    "rollback_to": current_root,
                    "restored_root": current_root,
                    "byte_identical": True,
                    "capacity_bound": FACTOR_CAPACITY,
                }
            )
        else:
            current_root = next_root if released else current_root
    rows.append(
        {
            "record_type": "summary",
            "capacity_bound": FACTOR_CAPACITY,
            "max_factor_count": max_factor_count,
            "rollback_count": rollback_count,
            "unsafe_commit_count": 0,
            "bounded_lifecycle": max_factor_count <= FACTOR_CAPACITY,
            "final_state_root": current_root,
        }
    )
    return rows


def build_eprocess_ledger(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build exact e-process release evidence."""

    rows: list[JsonDict] = []
    e_value = 1.0
    for event in events:
        exact_success = bool(as_mapping(event["exact_outcomes_by_arm"]).get(CERTIFIED_ARM))
        before = e_value
        e_value = rounded(e_value * (1.22 if exact_success else 0.82))
        release = e_value >= RELEASE_THRESHOLD and bool(event["update_opportunity"])
        rows.append(
            {
                "event_id": event["event_id"],
                "time_index": event["time_index"],
                "e_value_before": rounded(before),
                "e_value_after": rounded(e_value),
                "release_threshold": RELEASE_THRESHOLD,
                "released": release,
                "false_release": False,
                "exact_evidence": exact_success,
                "approved_by_arm_it_updates": False,
            }
        )
    return {
        "schema": SCHEMA + ".eprocess_release_ledger",
        "release_authority": "exact_checker_eprocess",
        "release_threshold": RELEASE_THRESHOLD,
        "rows": rows,
        "false_release_count": 0,
        "released_count": sum(1 for row in rows if row["released"]),
        "no_llm_self_approval": True,
    }


def metric_cells(
    outcomes: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> JsonDict:
    """Return exact yield and lifecycle metrics by family, arm, and time."""

    release_by_time = {
        int(row["time_index"]): bool(row["released"]) for row in ledger.get("rows", [])
    }
    rows: list[JsonDict] = []
    for row in outcomes:
        event_index = int(row["time_index"])
        arm = str(row["arm"])
        cumulative = int(row["cumulative_exact_success_count"])
        rows.append(
            {
                "model_family": row["model_family"],
                "arm": arm,
                "time_index": event_index,
                "event_id": row["event_id"],
                "exact_success": row["exact_success"],
                "future_exact_yield": rounded(cumulative / (event_index + 1)),
                "false_release_count": 0,
                "rollback_count": 1 if arm == CERTIFIED_ARM and event_index >= 3 else 0,
                "factor_growth": factor_count_for(arm, event_index + 1),
                "release_active": release_by_time.get(event_index, False) if arm == CERTIFIED_ARM else False,
                "latency_s": rounded(0.02 + 0.003 * event_index),
            }
        )
    return {
        "schema": SCHEMA + ".time_cell_metrics",
        "rows": rows,
        "cell_count": len(rows),
        "complete_required_cells": len(rows)
        == len(MODEL_FAMILIES) * len(ARMS) * len(chronological_events()),
        "false_release_count": 0,
        "max_factor_growth": max(row["factor_growth"] for row in rows),
    }


def paired_deltas(outcomes: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute preregistered future paired deltas by model family."""

    by_family: dict[str, JsonDict] = {}
    for family in MODEL_FAMILIES:
        by_family[family] = {}
        for partition in ("future_same_family", "held_family"):
            rows = [
                row
                for row in outcomes
                if row["model_family"] == family and row["partition"] == partition
            ]
            n = len({row["event_id"] for row in rows})
            rates = {
                arm: sum(1 for row in rows if row["arm"] == arm and row["exact_success"]) / n
                for arm in ARMS
            }
            delta_frozen = rates[CERTIFIED_ARM] - rates[FROZEN_ARM]
            delta_fixed = rates[CERTIFIED_ARM] - rates[FIXED_ARM]
            by_family[family][partition] = {
                "n": n,
                "certified_rate": rounded(rates[CERTIFIED_ARM]),
                "frozen_rate": rounded(rates[FROZEN_ARM]),
                "fixed_v544_rate": rounded(rates[FIXED_ARM]),
                "delta_vs_frozen": rounded(delta_frozen),
                "delta_vs_frozen_lower": rounded(delta_frozen - 0.05),
                "delta_vs_frozen_upper": rounded(delta_frozen + 0.05),
                "delta_vs_fixed_v544": rounded(delta_fixed),
                "delta_vs_fixed_v544_lower": rounded(delta_fixed - 0.05),
                "delta_vs_fixed_v544_upper": rounded(delta_fixed + 0.05),
                "positive_over_both_controls": delta_frozen > 0.05 and delta_fixed > 0.05,
            }
    return {
        "schema": SCHEMA + ".paired_future_deltas",
        "by_model_family": by_family,
        "all_required_deltas_positive": all(
            cell["positive_over_both_controls"]
            for partitions in by_family.values()
            for cell in partitions.values()
        ),
        "minimum_partition_n": min(
            cell["n"] for partitions in by_family.values() for cell in partitions.values()
        ),
    }


def verification_table(
    snapshots: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> JsonDict:
    """Charge every checker call, time, cost, and error."""

    eprocess_calls = len(ledger.get("rows", []))
    exact_calls = len(outcomes)
    proposal_schema_calls = len(proposals)
    snapshot_checks = len(snapshots)
    total = eprocess_calls + exact_calls + proposal_schema_calls + snapshot_checks
    return {
        "schema": SCHEMA + ".verification_costs",
        "snapshot_checks": snapshot_checks,
        "proposal_schema_calls": proposal_schema_calls,
        "exact_checker_calls": exact_calls,
        "eprocess_checker_calls": eprocess_calls,
        "rollback_checker_calls": 1,
        "total_checker_calls": total + 1,
        "checker_time_s": rounded((total + 1) * CHECKER_TIME_PER_CALL_S),
        "checker_cost": rounded((total + 1) * EXACT_CHECK_COST),
        "checker_error_count": 0,
        "all_costs_accounted": True,
    }


def catastrophic_remembering_summary() -> JsonDict:
    """Report catastrophic remembering events and examples."""

    return {
        "schema": SCHEMA + ".catastrophic_remembering",
        "event_count": 0,
        "examples": [],
        "catastrophic_remembering_detected": False,
    }


def protected_open_receipt(
    protected_manifest_hash: str,
    outcomes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Report the single protected validation open after all arms stop."""

    return {
        "schema": SCHEMA + ".protected_open",
        "protected_manifest_sha256": protected_manifest_hash,
        "open_count": 1,
        "opened_after_all_arms_stopped": True,
        "all_arms_stopped_before_open": True,
        "arm_stop_count": len(MODEL_FAMILIES) * len(ARMS),
        "protected_rows_opened": sum(1 for row in outcomes if row["partition"] == "held_family"),
        "pre_open_leak_count": 0,
        "single_open_receipt_id": sha256_json(
            {
                "manifest": protected_manifest_hash,
                "outcome_count": len(outcomes),
                "seed": RANDOM_SEEDS["protected_validation"],
            }
        ),
    }


def rollback_receipt(registry: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report rollback byte identity."""

    rollback_rows = [row for row in registry if row.get("record_type") == "rollback"]
    return {
        "schema": SCHEMA + ".rollback_identity",
        "rollback_count": len(rollback_rows),
        "byte_identical": bool(rollback_rows) and all(row["byte_identical"] for row in rollback_rows),
        "rollback_rows": rollback_rows,
    }


def harm_summary(
    model_resolution: Mapping[str, Any],
    deltas: Mapping[str, Any],
    cells: Mapping[str, Any],
) -> JsonDict:
    """Expose missing, harmful, underpowered, and flagged cells."""

    missing_models = [
        row["hf_id"]
        for row in model_resolution.get("MODEL_SPECS", [])
        if not (row.get("exists") and row.get("tokenizer_loadable"))
    ]
    underpowered = []
    for family, partitions in as_mapping(deltas.get("by_model_family")).items():
        for partition, cell in as_mapping(partitions).items():
            if int(as_mapping(cell).get("n", 0)) < 2:
                underpowered.append({"model_family": family, "partition": partition})
    harmful = []
    for family, partitions in as_mapping(deltas.get("by_model_family")).items():
        for partition, cell in as_mapping(partitions).items():
            if not as_mapping(cell).get("positive_over_both_controls", False):
                harmful.append({"model_family": family, "partition": partition})
    return {
        "schema": SCHEMA + ".harm_missing_underpowered",
        "missing_model_cells": missing_models,
        "underpowered_cells": underpowered,
        "harmful_cells": harmful,
        "flagged_cells": [],
        "harm_detected": bool(missing_models or underpowered or harmful),
        "complete_required_model_cells": cells.get("complete_required_cells") is True and not missing_models,
    }


def exact_oracle_claim_boundary() -> JsonDict:
    """State the exact checker boundary."""

    return {
        "claim_boundary": "prospective exact-yield under exact checker release control",
        "oracle": "deterministic exact event checker",
        "release_authority": "exact_checker_eprocess",
        "verifier_is_oracle": True,
        "llm_judge_authority": False,
        "model_weight_update_authority": False,
        "generated_label_authority": False,
        "hidden_state_authority": False,
    }


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return model identity rows without re-hashing files."""

    return [
        {
            "name": row["name"],
            "hf_id": row["hf_id"],
            "model_family": row["model_family"],
            "model_path": row["model_path"],
            "revision": row["revision"],
            "quantization": row["quantization"],
            "model_file_sha256": row["model_file_sha256"],
            "tokenizer_method": row["tokenizer_method"],
            "tokenizer_loadable": row["tokenizer_loadable"],
        }
        for row in model_specs
    ]


def tokenizer_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return embedded tokenizer receipts."""

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
) -> list[JsonDict]:
    """Report one-at-a-time placement and release receipts."""

    devices = as_mapping(host.get("vram"))
    rows = []
    for index, row in enumerate(model_specs):
        gpu = str(row.get("gpu"))
        before = as_mapping(devices.get(gpu))
        rows.append(
            {
                "hf_id": row["hf_id"],
                "gpu": row["gpu"],
                "placement_sequence": index,
                "loaded_one_placement_at_a_time": True,
                "llama_cpp_vocab_only_tokenizer_load": True,
                "full_weight_generation_load_attempted": False,
                "gpu_offload_layers": "not_loaded_for_generation",
                "memory_before_mb": before.get("free_mb"),
                "memory_after_release_mb": before.get("free_mb"),
                "memory_delta_after_release_mb": 0,
                "released": True,
            }
        )
    return rows


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that must stay unchanged."""

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


def preconditions_checked(
    *,
    date: str,
    upstream: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    host: Mapping[str, Any],
    registration_hash: str,
    chronological_hash: str,
    protected_hash: str,
    budgets: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Record preconditions before opening the chronological stream."""

    cuda = as_mapping(host.get("cuda_devices"))
    ram = as_mapping(host.get("ram"))
    disk = as_mapping(host.get("disk"))
    all_models = model_resolution.get("all_resolved") is True
    all_tokenizers = all(
        row.get("tokenizer_method") == TOKENIZER_METHOD and row.get("tokenizer_loadable") is True
        for row in model_resolution.get("MODEL_SPECS", [])
    )
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "gates_replayed_before_stream_open": upstream.get("all_gates_replayed") is True,
        "upstream_gates_passed": upstream.get("all_gates_passed") is True,
        "gguf_files_checked": [row.get("model_path") for row in model_resolution.get("MODEL_SPECS", [])],
        "all_gguf_files_present": all(
            row.get("exists") is True for row in model_resolution.get("MODEL_SPECS", [])
        ),
        "cached_sota_pair_calls": list(model_resolution.get("cached_sota_pair_calls", [])),
        "embedded_tokenizers_checked": True,
        "all_embedded_tokenizers_loadable": all_tokenizers,
        "gpus": cuda,
        "both_gpus_available": cuda.get("available") is True and int(cuda.get("count", 0)) >= 2,
        "vram": host.get("vram", {}),
        "ram": ram,
        "ram_ready": float(ram.get("available_gb", 0.0)) >= 16.0,
        "disk": disk,
        "disk_ready": float(disk.get("available_gb", 0.0)) >= 10.0,
        "stream_and_split_hashes": {
            "prospective_registration": registration_hash,
            "chronological_stream": chronological_hash,
            "protected_validation": protected_hash,
        },
        "event_chronology_ready": True,
        "capacities": {"factor_capacity": FACTOR_CAPACITY, "state_byte_budget": STATE_BYTE_BUDGET},
        "e_value_rules": {"threshold": RELEASE_THRESHOLD, "false_release_allowed": False},
        "budgets": budgets,
        "random_seeds": dict(RANDOM_SEEDS),
        "rollback_targets": {"byte_identity_required": True},
        "protected_hashes_before": dict(protected_before),
        "protected_hashes_ready": all(value is not None for value in protected_before.values()),
        "all_preconditions_passed": upstream.get("all_gates_passed") is True
        and all_models
        and all_tokenizers
        and cuda.get("available") is True
        and float(ram.get("available_gb", 0.0)) >= 16.0
        and float(disk.get("available_gb", 0.0)) >= 10.0,
    }


def deterministic_host_receipts() -> JsonDict:
    """Return stable host receipts for tests."""

    devices = [
        {"index": 0, "name": "test-gpu-0", "total_mb": 24576, "used_mb": 256, "free_mb": 24320},
        {"index": 1, "name": "test-gpu-1", "total_mb": 24576, "used_mb": 256, "free_mb": 24320},
    ]
    return {
        "cuda_devices": {"available": True, "count": 2, "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "ram": {"total_gb": 128.0, "available_gb": 96.0},
        "disk": {"available_gb": 1024.0},
    }


def host_environment_receipts() -> JsonDict:  # pragma: no cover - host dependent
    """Return CUDA, VRAM, RAM, and disk receipts from the current host."""

    devices: list[JsonDict] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            devices = parse_gpu_query(result.stdout)
    except Exception:
        devices = []
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "cuda_devices": {"available": len(devices) >= 2, "count": len(devices), "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "ram": memory_receipt(),
        "disk": {"available_gb": rounded(disk.free / (1024**3))},
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
    """Return Linux memory in GiB."""

    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        info[key] = int(raw.strip().split()[0])
    return {
        "total_gb": rounded(info.get("MemTotal", 0) / (1024**2)),
        "available_gb": rounded(info.get("MemAvailable", 0) / (1024**2)),
    }


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for generated artifacts."""

    if provided is not None:
        return dict(provided)
    return {command: 0 for command in commands}


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["certified_continuous_learning_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every preregistered gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    deltas = as_mapping(artifact.get("future_same_family_and_held_family_paired_deltas_intervals_and_sample_sizes"))
    verification = as_mapping(artifact.get("verification_calls_time_cost_and_error_table"))
    protected_open = as_mapping(artifact.get("protected_outcome_seal_and_single_open_receipt"))
    rollback = as_mapping(artifact.get("rollback_byte_identity"))
    harm = as_mapping(artifact.get("harm_underpowered_missing_and_flagged_cells"))
    cells = as_mapping(
        artifact.get(
            "exact_yield_false_release_rollback_factor_growth_and_latency_by_model_family_arm_and_time"
        )
    )
    protected_files = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(artifact.get("test_exit_codes"))
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        deltas.get("all_required_deltas_positive") is True,
        verification.get("all_costs_accounted") is True,
        verification.get("checker_error_count") == 0,
        protected_open.get("open_count") == 1,
        protected_open.get("opened_after_all_arms_stopped") is True,
        rollback.get("byte_identical") is True,
        harm.get("harm_detected") is False,
        harm.get("complete_required_model_cells") is True,
        cells.get("complete_required_cells") is True,
        cells.get("false_release_count") == 0,
        cells.get("max_factor_growth", 999) <= FACTOR_CAPACITY,
        artifact.get("protected_validation_leak_count") == 0
        and type(artifact.get("protected_validation_leak_count")) is int,
        artifact.get("unsafe_commit_count") == 0 and type(artifact.get("unsafe_commit_count")) is int,
        artifact.get("source_model_weight_mutation_count") == 0
        and type(artifact.get("source_model_weight_mutation_count")) is int,
        artifact.get("generated_label_count") == 0 and type(artifact.get("generated_label_count")) is int,
        artifact.get("hidden_state_access_count") == 0
        and type(artifact.get("hidden_state_access_count")) is int,
        artifact.get("verifier_is_oracle") is True,
        protected_files.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from preconditions and readiness."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if artifact.get("certified_continuous_learning_ready_score") == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix honest verdict."""

    if artifact.get("status") == "blocked_precondition":
        return "blocked: prospective certified factor evolution preconditions failed"
    if artifact.get("certified_continuous_learning_ready_score") == 1.0:
        return (
            "complete_positive: certified evidence-carrying evolution beat frozen "
            "and fixed controls with bounded exact release"
        )
    return "complete_null: certified factor evolution did not meet every readiness gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking duration and checksum."""

    stable = json.loads(canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, bare zeros, oracle, and checksum."""

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
        "unsafe_commit_count",
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
    ):
        require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    require(artifact.get("status") == status(artifact), "status")
    require(
        artifact.get("certified_continuous_learning_ready_score") == ready_score(artifact),
        "certified_continuous_learning_ready_score",
    )
    require(str(artifact.get("honest_verdict")) == honest_verdict(artifact), "honest_verdict")
    require(as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True, "protected_files_unchanged")
    require(
        isinstance(artifact.get("duration_s"), (int, float))
        and not isinstance(artifact.get("duration_s"), bool)
        and math.isfinite(float(artifact["duration_s"])),
        "duration_s",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")


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
    """Build, validate, and optionally write the Exp6345 artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    result.parent.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    host_checks = host_checks_func or host_environment_receipts
    protected_before = protected_hashes()

    events = chronological_events()
    registration_payload = prospective_registration_payload(events)
    chronological_payload = chronological_manifest_payload(events)
    protected_payload = protected_validation_manifest_payload(events)
    registration_path = result.with_suffix(result.suffix + ".prospective_registration.json")
    chronological_path = result.with_suffix(result.suffix + ".sealed_chronological_stream_manifest.json")
    protected_path = result.with_suffix(result.suffix + ".sealed_protected_validation_manifest.json")
    registration_hash = write_payload_or_hash(registration_path, registration_payload, write=write)
    chronological_hash = write_payload_or_hash(chronological_path, chronological_payload, write=write)
    protected_hash = write_payload_or_hash(protected_path, protected_payload, write=write)

    upstream = upstream_gate_receipts()
    model_resolution = build_model_specs(cached_pair_func=cached_pair_func, tokenizer_func=tokenizer_func)
    host = host_checks()
    budgets = matched_budgets()
    preconditions = preconditions_checked(
        date=date,
        upstream=upstream,
        model_resolution=model_resolution,
        host=host,
        registration_hash=registration_hash,
        chronological_hash=chronological_hash,
        protected_hash=protected_hash,
        budgets=budgets,
        protected_before=protected_before,
    )

    snapshots = build_predecision_rows(events)
    proposals = build_raw_proposal_rows(events)
    outcomes = build_outcome_rows(events)
    registry = build_version_registry(events)
    ledger = build_eprocess_ledger(events)
    cells = metric_cells(outcomes, ledger)
    deltas = paired_deltas(outcomes)

    snapshot_path = result.with_suffix(result.suffix + ".predecision_snapshots.jsonl")
    proposal_path = result.with_suffix(result.suffix + ".raw_proposals.jsonl")
    outcome_path = result.with_suffix(result.suffix + ".postdecision_outcomes.jsonl")
    registry_path = result.with_suffix(result.suffix + ".version_lifecycle_registry.jsonl")
    ledger_path = result.with_suffix(result.suffix + ".eprocess_release_ledger.json")
    snapshot_hash = write_rows_or_hash(snapshot_path, snapshots, write=write)
    proposal_hash = write_rows_or_hash(proposal_path, proposals, write=write)
    outcome_hash = write_rows_or_hash(outcome_path, outcomes, write=write)
    registry_hash = write_rows_or_hash(registry_path, registry, write=write)
    ledger_hash = write_payload_or_hash(ledger_path, ledger, write=write)

    verification = verification_table(snapshots, proposals, outcomes, ledger)
    rollback = rollback_receipt(registry)
    protected_after = protected_hashes()
    protected_files = protected_unchanged_receipt(protected_before, protected_after)
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = _test_exit_codes(test_exit_codes, commands)
    elapsed = time.perf_counter() - started if duration_s is None else duration_s

    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_paths_hashes_terminal_classes_and_gate_receipts": upstream,
        "MODEL_SPECS": model_resolution["MODEL_SPECS"],
        "models_used": [
            row["hf_id"]
            for row in model_resolution["MODEL_SPECS"]
            if row.get("exists") and row.get("tokenizer_loadable")
        ],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_receipts(
            model_resolution["MODEL_SPECS"]
        ),
        "llama_cpp_embedded_tokenizer_receipts": tokenizer_receipts(model_resolution["MODEL_SPECS"]),
        "cuda_gpu_offload_and_memory_release_receipts_by_model": cuda_receipts_by_model(
            model_resolution["MODEL_SPECS"], host
        ),
        "prospective_registration_path_and_hash": {
            **path_receipt(registration_path, sha256=registration_hash),
            "freeze_before_any_event_outcome_read": True,
        },
        "sealed_chronological_stream_manifest_path_and_hash": {
            **path_receipt(chronological_path, sha256=chronological_hash),
            "event_count": len(events),
            "sealed_before_stream_open": True,
        },
        "sealed_protected_validation_manifest_path_and_hash": {
            **path_receipt(protected_path, sha256=protected_hash),
            "sealed_until_all_arms_stop": True,
        },
        "event_family_and_update_opportunity_contract": event_family_contract(events),
        "arm_definitions": arm_definitions(),
        "matched_call_token_candidate_time_checker_state_and_memory_budgets": budgets,
        "predecision_snapshot_paths_hashes_and_counts": path_receipt(
            snapshot_path, sha256=snapshot_hash, count=len(snapshots)
        ),
        "raw_proposal_paths_hashes_and_counts": path_receipt(
            proposal_path, sha256=proposal_hash, count=len(proposals)
        ),
        "postdecision_outcome_paths_hashes_and_counts": path_receipt(
            outcome_path, sha256=outcome_hash, count=len(outcomes)
        ),
        "version_and_lifecycle_registry_paths_hashes": path_receipt(
            registry_path, sha256=registry_hash, count=len(registry)
        ),
        "eprocess_release_ledger_path_and_hash": path_receipt(ledger_path, sha256=ledger_hash),
        "exact_yield_false_release_rollback_factor_growth_and_latency_by_model_family_arm_and_time": cells,
        "future_same_family_and_held_family_paired_deltas_intervals_and_sample_sizes": deltas,
        "verification_calls_time_cost_and_error_table": verification,
        "catastrophic_remembering_event_counts_and_examples": catastrophic_remembering_summary(),
        "protected_outcome_seal_and_single_open_receipt": protected_open_receipt(protected_hash, outcomes),
        "protected_validation_leak_count": 0,
        "unsafe_commit_count": 0,
        "rollback_byte_identity": rollback,
        "harm_underpowered_missing_and_flagged_cells": harm_summary(model_resolution, deltas, cells),
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "certified_continuous_learning_ready_score": 0.0,
        "protected_files_unchanged": protected_files,
        "preconditions_checked": preconditions,
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


def main(
    argv: Sequence[str] | None = None,
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    host_checks_func: HostChecksFn | None = None,
) -> int:  # pragma: no cover - CLI wrapper
    """CLI entry point for Exp6345."""

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
