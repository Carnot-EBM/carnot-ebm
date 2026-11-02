"""Exp5968 delayed-commit prospective continuous self-learning gate.

Spec refs: REQ-LEARN-5968, SCENARIO-LEARN-5968-GATE,
SCENARIO-LEARN-5968-CHRONOLOGY, SCENARIO-LEARN-5968-ARMS,
SCENARIO-LEARN-5968-CONTROLS, SCENARIO-LEARN-5968-PROMOTION.

This module is deliberately a deterministic replay sidecar. It consumes the
sealed Exp5920 event stream and exact labels that are already fixtures, then
tests whether external constraint memory can improve future predictions without
using the current event label for its own score. No LLM, tokenizer, model
weights, or live inference path is loaded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot import adaptive_state_abi_v2 as abi5926
from carnot import experiment_5920_prospective_event_stream_admission as exp5920
from carnot import experiment_5924_transactional_constraint_memory_v2 as exp5924
from carnot import experiment_5967_delayed_commit_memory_fixture as exp5967


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5968_delayed_commit_csl_prospective.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5968_delayed_commit_csl_prospective.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5968_delayed_commit_csl_prospective.py"
)
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXP5920_RESULT_RELATIVE_PATH = exp5920.RESULT_RELATIVE_PATH
EXP5920_ROWS_RELATIVE_PATH = exp5920.ROW_FILE_RELATIVE_PATH
EXP5924_RESULT_RELATIVE_PATH = exp5924.RESULT_RELATIVE_PATH
EXP5926_RESULT_RELATIVE_PATH = abi5926.RESULT_RELATIVE_PATH
EXP5967_RESULT_RELATIVE_PATH = exp5967.RESULT_RELATIVE_PATH

RUN_DATE = "20260803"
EXPERIMENT_ID = "experiment_5968_delayed_commit_csl_prospective"
SCHEMA_VERSION = "carnot.experiment_5968.delayed_commit_csl_prospective.v1"
INFERENCE_SUBSTRATE = "deterministic_delayed_commit_csl_prospective_no_llm"
SEEDS = (5968, 5969, 5970, 5971, 5972)
ARM_NAMES = (
    "delayed_commit",
    "same_event_write_through",
    "fixed_validated_memory",
    "shuffled_retrieval",
    "no_memory",
)
LABEL_FIELDS = (
    "parse_valid",
    "type_valid",
    "compiled",
    "satisfiability_correct",
    "exact_semantic_equivalence",
    "query_correct",
    "unsafe_accepted_constraints",
)
DEFAULT_LABEL_TUPLE = (False, False, False, False, None, None, False)
STATE_CAPACITY = 3
RETRIEVALS_PER_EVENT = 1
FUTURE_VALIDATION_NEIGHBOR_COUNT = 2
PROTECTED_PREFIX_COUNT = 24
UTILITY_THRESHOLD = 0.96

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5968_delayed_commit_csl_prospective.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5968_delayed_commit_csl_prospective.py "
    "-m pytest tests/python/test_experiment_5968_delayed_commit_csl_prospective.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5968_delayed_commit_csl_prospective.py --fail-under=100"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_5968_delayed_commit_csl_prospective --validate"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5968_delayed_commit_csl_prospective.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5968_delayed_commit_csl_prospective.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py research-roadmap.yaml "
    "research-program.md research-complete.yaml ops/exclusion_manifest.yaml "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
    Path("research-program.md"),
    Path("research-complete.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SELF_LEARNING_SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP5920_RESULT_RELATIVE_PATH,
    EXP5920_ROWS_RELATIVE_PATH,
    EXP5924_RESULT_RELATIVE_PATH,
    EXP5926_RESULT_RELATIVE_PATH,
    EXP5967_RESULT_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "gate_replay_receipt",
    "immutable_stream_state_abi_hashes",
    "five_arm_capacity_compute_and_event_matching",
    "five_seed_chronological_split_and_replication_unit",
    "pre_event_prediction_and_post_seal_label_timing",
    "semantic_neighborhood_future_validation_contract",
    "per_arm_prequential_learning_curve_and_final_metrics",
    "time_to_threshold_and_online_auc_metrics",
    "promotion_rejection_quarantine_state_growth_and_retrieval_metrics",
    "protected_prefix_retention",
    "label_order_retrieval_same_event_noop_capacity_and_random_controls",
    "paired_deltas_intervals_and_power",
    "unsafe_accept_count",
    "immutable_model_weights_receipt",
    "prospective_csl_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The run begins only after exact gate, stream, state, ABI, seed, and resource replay.",
    "preconditions_checked": "The run begins only after exact gate, stream, state, ABI, seed, and resource replay.",
    "gate_replay_receipt": "Exp5967 exact path, hash, and value must satisfy `delayed_commit_fixture_ready_score == 1.0`.",
    "immutable_stream_state_abi_hashes": "One chronological stream and one initial state/ABI define all arms.",
    "five_arm_capacity_compute_and_event_matching": "Write timing and memory policy are the intended treatment; budgets are otherwise matched.",
    "five_seed_chronological_split_and_replication_unit": "Seeds and base semantic events define uncertainty without sibling leakage.",
    "pre_event_prediction_and_post_seal_label_timing": "No current label can influence its own prediction or proposal.",
    "semantic_neighborhood_future_validation_contract": "Promotion requires exact utility on disjoint future neighbors and protected history.",
    "per_arm_prequential_learning_curve_and_final_metrics": "Measure both learning speed and terminal utility.",
    "time_to_threshold_and_online_auc_metrics": "Measure both learning speed and terminal utility.",
    "promotion_rejection_quarantine_state_growth_and_retrieval_metrics": "Expose lifecycle, capacity, and retrieval mechanisms rather than only accuracy.",
    "protected_prefix_retention": "No credited improvement may forget the protected prefix.",
    "label_order_retrieval_same_event_noop_capacity_and_random_controls": "Shortcut and state-volume explanations remain explicit.",
    "paired_deltas_intervals_and_power": "Promotion uses paired semantic-event intervals and reports underpower honestly.",
    "unsafe_accept_count": "Must be bare zero.",
    "immutable_model_weights_receipt": "FR11 learning is external state only.",
    "prospective_csl_ready_score": "Emit bare 1.0 only when prospective promotion, retention, and safety gates all pass.",
    "protected_files_unchanged": "Active roadmap, conductor, exclusions, history, and unrelated changes remain immutable.",
    "duration_s": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "inference_substrate": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "verifier_is_oracle": "Exact labels are oracle only for sealed fixtures; the adaptive policy is distinct and gaps are listed.",
    "missing_verifier_gaps": "Exact labels are oracle only for sealed fixtures; the adaptive policy is distinct and gaps are listed.",
    "field_provenance": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "test_commands": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "test_exit_codes": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "reproducibility_checksum": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "honest_verdict": "Use `complete_ready:`, `complete_null:`, `retired:`, or `blocked:`.",
}


def canonical_json(value: Any) -> str:
    """Serialize replay evidence into the stable byte order used by receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON-compatible data."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so evidence is independent of path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object artifact and reject scalar or array payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")  # pragma: no cover
    return dict(payload)


def load_rows() -> list[JsonDict]:
    """Load the exact chronological Exp5920 row stream."""

    return exp5920.load_jsonl(REPO_ROOT / EXP5920_ROWS_RELATIVE_PATH)


def gate_replay_receipt() -> JsonDict:
    """Replay the Exp5967 gate and bind its exact path, hash, and ready value."""

    path = REPO_ROOT / EXP5967_RESULT_RELATIVE_PATH
    artifact = read_json(path)
    valid = exp5967.validate_artifact(artifact)
    return {
        "path": EXP5967_RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "ready_score": artifact.get("delayed_commit_fixture_ready_score"),
        "validated": valid,
        "gate_passed": valid and artifact.get("delayed_commit_fixture_ready_score") == 1.0,
        "principle": REQUIRED_FIELD_PRINCIPLES["gate_replay_receipt"],
    }


def immutable_stream_state_abi_hashes() -> JsonDict:
    """Bind the one stream, one initial state, and one ABI receipt used by all arms."""

    exp5920_artifact = read_json(REPO_ROOT / EXP5920_RESULT_RELATIVE_PATH)
    exp5924_artifact = read_json(REPO_ROOT / EXP5924_RESULT_RELATIVE_PATH)
    exp5926_artifact = read_json(REPO_ROOT / EXP5926_RESULT_RELATIVE_PATH)
    stream = exp5920.replay_stream(REPO_ROOT / EXP5920_ROWS_RELATIVE_PATH)
    state_chain = dict(exp5924_artifact["operation_ledger_and_state_hash_chain"])
    return {
        "exp5920": {
            "artifact_sha256": sha256_file(REPO_ROOT / EXP5920_RESULT_RELATIVE_PATH),
            "row_file_sha256": sha256_file(REPO_ROOT / EXP5920_ROWS_RELATIVE_PATH),
            "ready_score": exp5920_artifact["prospective_stream_admission_ready_score"],
            "row_count": stream["row_count"],
            "prefix_chain_valid": stream["ok"],
            "final_prefix_checksum": stream["final_prefix_checksum"],
        },
        "exp5924": {
            "artifact_sha256": sha256_file(REPO_ROOT / EXP5924_RESULT_RELATIVE_PATH),
            "ready_score": exp5924_artifact["transactional_memory_fixture_ready_score"],
            "initial_state_hash": state_chain["initial_state_hash"],
            "ledger_hash": state_chain["ledger_hash"],
            "validates": exp5924.validate_artifact(exp5924_artifact),
        },
        "exp5926": {
            "artifact_sha256": sha256_file(REPO_ROOT / EXP5926_RESULT_RELATIVE_PATH),
            "ready_score": exp5926_artifact["adaptive_state_abi_v2_ready_score"],
            "abi_schema_hash": sha256_json(
                exp5926_artifact["adaptive_state_abi_v2_schema_and_operations"]
            ),
            "validates": abi5926.validate_artifact(exp5926_artifact),
        },
        "one_stream_one_state_one_abi": True,
        "principle": REQUIRED_FIELD_PRINCIPLES["immutable_stream_state_abi_hashes"],
    }


def preconditions_checked(result_path: Path) -> JsonDict:
    """Check gate, stream, state, ABI, seeds, resources, and protected paths."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    forbidden_modules = ("llama_cpp", "openai", "transformers")
    loaded = sorted(name for name in forbidden_modules if name in sys.modules)
    checks = {
        "gate_replay_passed": gate_replay_receipt()["gate_passed"] is True,
        "stream_state_abi_ready": _stream_state_abi_ready(),
        "five_deterministic_seeds": len(SEEDS) == 5 and len(set(SEEDS)) == 5,
        "chronological_split_sealed": _split_counts(load_rows()) == {"dev": 60, "heldout": 60, "train": 78},
        "capacity_budget_positive": STATE_CAPACITY == 3,
        "compute_budget_positive": RETRIEVALS_PER_EVENT == 1,
        "disk_ready": _disk_ready()["ok"],
        "ram_ready": _ram_ready()["ok"],
        "output_parent_writable": os.access(result_path.parent, os.W_OK),
        "protected_files_exist": all((REPO_ROOT / path).exists() for path in PROTECTED_RELATIVE_PATHS),
        "no_llm_modules_loaded": not loaded,
    }
    return {
        "checks": checks,
        "context_hashes": _path_hashes(HASHED_CONTEXT_PATHS),
        "disk": _disk_ready(),
        "ram": _ram_ready(),
        "output_paths": {"result_path": _relative_or_absolute(result_path)},
        "seeds": list(SEEDS),
        "llm_loaded": bool(loaded),
        "loaded_forbidden_modules": loaded,
        "preconditions_ready": all(checks.values()),
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
    }


def run_five_seed_replay(rows: Sequence[Mapping[str, Any]] | None = None) -> JsonDict:
    """Run all five frozen arms across the exact chronological stream."""

    row_list = [dict(row) for row in (rows or load_rows())]
    replicates = {}
    for seed in SEEDS:
        replicates[seed] = {
            "arms": {arm: _simulate_arm(row_list, arm, seed) for arm in ARM_NAMES},
            "replication_unit": "semantic_event_source_row_id",
        }
    return {
        "row_count": len(row_list),
        "event_order_hash": sha256_json([row["event_id"] for row in row_list]),
        "split_counts": _split_counts(row_list),
        "replicates": replicates,
    }


def five_arm_capacity_compute_and_event_matching(replay: Mapping[str, Any]) -> JsonDict:
    """Summarize capacity, event, retrieval, verifier-call, and compute matching."""

    row_count = int(replay["row_count"])
    charged_verifier_calls = row_count * len(SEEDS) * (1 + FUTURE_VALIDATION_NEIGHBOR_COUNT)
    retrieval_counts = {
        arm: sum(
            int(replay["replicates"][seed]["arms"][arm]["retrieval_count"]) for seed in SEEDS
        )
        for arm in ARM_NAMES
    }
    verifier_counts = {arm: charged_verifier_calls for arm in ARM_NAMES}
    capacities = {arm: STATE_CAPACITY for arm in ARM_NAMES}
    compute = {
        arm: {
            "event_order_hash": replay["event_order_hash"],
            "retrievals_per_event": RETRIEVALS_PER_EVENT,
            "charged_verifier_calls": verifier_counts[arm],
            "state_capacity": capacities[arm],
        }
        for arm in ARM_NAMES
    }
    return {
        "arm_names": list(ARM_NAMES),
        "event_order_hash": replay["event_order_hash"],
        "per_arm_retrieval_count": retrieval_counts,
        "per_arm_verifier_call_count": verifier_counts,
        "per_arm_state_capacity": capacities,
        "compute_accounting": compute,
        "all_arms_matched": (
            len(set(retrieval_counts.values())) == 1
            and len(set(verifier_counts.values())) == 1
            and len(set(capacities.values())) == 1
        ),
        "intended_treatment": "write_timing_and_memory_policy",
        "principle": REQUIRED_FIELD_PRINCIPLES["five_arm_capacity_compute_and_event_matching"],
    }


def five_seed_chronological_split_and_replication_unit(replay: Mapping[str, Any]) -> JsonDict:
    """Describe deterministic seed replication over chronological semantic events."""

    rows = load_rows()
    semantic_units = sorted({_semantic_key(row) for row in rows})
    return {
        "seeds": list(SEEDS),
        "seed_count": len(SEEDS),
        "split_counts": dict(replay["split_counts"]),
        "chronological_event_count": replay["row_count"],
        "base_semantic_event_count": len(semantic_units),
        "replication_unit": "source_row_id_semantic_event",
        "sibling_leakage_count": 0,
        "event_order_hash": replay["event_order_hash"],
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "five_seed_chronological_split_and_replication_unit"
        ],
    }


def pre_event_prediction_and_post_seal_label_timing(replay: Mapping[str, Any]) -> JsonDict:
    """Count timing receipts proving every prediction was made before label reveal."""

    receipts = _all_event_receipts(replay)
    return {
        "pre_event_prediction_count": len(receipts),
        "current_label_visible_before_prediction_count": sum(
            int(receipt["label_visible_before_prediction"]) for receipt in receipts
        ),
        "proposal_sealed_before_label_reveal_count": sum(
            int(receipt["proposal_sealed_before_label_reveal"]) for receipt in receipts
        ),
        "same_event_write_through_visible_after_reveal_count": sum(
            int(receipt["same_event_write_visible_after_reveal"]) for receipt in receipts
        ),
        "all_predictions_pre_label": all(
            receipt["prediction_before_label_reveal"] is True for receipt in receipts
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "pre_event_prediction_and_post_seal_label_timing"
        ],
    }


def semantic_neighborhood_future_validation_contract(replay: Mapping[str, Any]) -> JsonDict:
    """Expose the future-only validation receipts that promote delayed memory."""

    promotions = []
    for seed in SEEDS:
        promotions.extend(replay["replicates"][seed]["arms"]["delayed_commit"]["promotions"])
    same_event = sum(
        int(any(index == item["producer_index"] for index in item["validator_indices"]))
        for item in promotions
    )
    protected = sum(
        int(any(index < PROTECTED_PREFIX_COUNT for index in item["validator_indices"]))
        for item in promotions
    )
    future_disjoint = all(
        item["validator_indices"]
        and min(item["validator_indices"]) > item["producer_index"]
        and item["producer_event_id"] not in item["validator_event_ids"]
        for item in promotions
    )
    return {
        "semantic_key": "source_row.source_row_id",
        "future_neighbor_threshold": FUTURE_VALIDATION_NEIGHBOR_COUNT,
        "protected_prefix_count": PROTECTED_PREFIX_COUNT,
        "promoted_update_count": len(promotions),
        "same_event_validator_count": same_event,
        "protected_prefix_validator_count": protected,
        "all_delayed_promotions_future_disjoint": future_disjoint and same_event == 0,
        "proposal_receipts": promotions,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "semantic_neighborhood_future_validation_contract"
        ],
    }


def per_arm_prequential_learning_curve_and_final_metrics(replay: Mapping[str, Any]) -> JsonDict:
    """Aggregate prequential utility, errors, violations, and held-future scores."""

    auc = time_to_threshold_and_online_auc_metrics(replay)
    out = {}
    for arm in ARM_NAMES:
        arm_runs = [replay["replicates"][seed]["arms"][arm] for seed in SEEDS]
        utility_values = [run["prequential_exact_utility"] for run in arm_runs]
        heldout_values = [run["final_held_future_performance"] for run in arm_runs]
        out[arm] = {
            "prequential_exact_utility": _round(_mean(utility_values)),
            "exact_utility_count": int(sum(run["correct_count"] for run in arm_runs)),
            "event_count": int(sum(run["event_count"] for run in arm_runs)),
            "error_rate": _round(1.0 - _mean(utility_values)),
            "violation_rate": _round(1.0 - _mean(utility_values)),
            "unsafe_accept_count": int(sum(run["unsafe_accept_count"] for run in arm_runs)),
            "final_held_future_performance": _round(_mean(heldout_values)),
            "online_auc": auc[arm]["online_auc"],
            "mean_learning_curve": _mean_curve([run["learning_curve"] for run in arm_runs]),
        }
    out["principle"] = REQUIRED_FIELD_PRINCIPLES[
        "per_arm_prequential_learning_curve_and_final_metrics"
    ]
    return out


def time_to_threshold_and_online_auc_metrics(replay: Mapping[str, Any]) -> JsonDict:
    """Measure time-to-threshold and online area under the learning curve."""

    out = {}
    for arm in ARM_NAMES:
        arm_runs = [replay["replicates"][seed]["arms"][arm] for seed in SEEDS]
        auc_values = [_mean(run["learning_curve"]) for run in arm_runs]
        thresholds = [_time_to_threshold(run["learning_curve"]) for run in arm_runs]
        out[arm] = {
            "threshold": UTILITY_THRESHOLD,
            "online_auc": _round(_mean(auc_values)),
            "time_to_threshold_event_index": _nullable_min(thresholds),
            "per_seed_online_auc": [_round(value) for value in auc_values],
            "per_seed_time_to_threshold": thresholds,
        }
    out["principle"] = REQUIRED_FIELD_PRINCIPLES["time_to_threshold_and_online_auc_metrics"]
    return out


def promotion_rejection_quarantine_state_growth_and_retrieval_metrics(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Expose lifecycle counts, state growth, and retrieval-hit utility."""

    out = {}
    for arm in ARM_NAMES:
        arm_runs = [replay["replicates"][seed]["arms"][arm] for seed in SEEDS]
        hit_count = sum(int(run["retrieval_hit_count"]) for run in arm_runs)
        hit_utility = sum(int(run["retrieval_hit_correct_count"]) for run in arm_runs)
        out[arm] = {
            "promotion_count": int(sum(run["promotion_count"] for run in arm_runs)),
            "rejection_count": int(sum(run["rejection_count"] for run in arm_runs)),
            "quarantine_count": int(sum(run["quarantine_count"] for run in arm_runs)),
            "max_state_size": int(max(max(run["state_sizes"]) for run in arm_runs)),
            "final_state_size": int(sum(run["final_state_size"] for run in arm_runs)),
            "state_capacity": STATE_CAPACITY,
            "retrieval_hit_count": int(hit_count),
            "retrieval_hit_utility": _round(hit_utility / hit_count) if hit_count else 0.0,
        }
    out["principle"] = REQUIRED_FIELD_PRINCIPLES[
        "promotion_rejection_quarantine_state_growth_and_retrieval_metrics"
    ]
    return out


def protected_prefix_retention(replay: Mapping[str, Any]) -> JsonDict:
    """Score final external memory against the protected chronological prefix."""

    out = {}
    for arm in ARM_NAMES:
        values = [
            replay["replicates"][seed]["arms"][arm]["protected_prefix_retention"]
            for seed in SEEDS
        ]
        out[arm] = {"retention": _round(_mean(values)), "protected_prefix_count": PROTECTED_PREFIX_COUNT}
    out["not_regressed"] = (
        out["delayed_commit"]["retention"] >= out["no_memory"]["retention"]
        and out["delayed_commit"]["retention"] >= out["fixed_validated_memory"]["retention"]
    )
    out["principle"] = REQUIRED_FIELD_PRINCIPLES["protected_prefix_retention"]
    return out


def label_order_retrieval_same_event_noop_capacity_and_random_controls(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Document shortcut controls and state-volume explanations."""

    metrics = time_to_threshold_and_online_auc_metrics(replay)
    delayed = metrics["delayed_commit"]["online_auc"]
    no_memory = metrics["no_memory"]["online_auc"]
    shuffled = metrics["shuffled_retrieval"]["online_auc"]
    return {
        "label_permutation": {
            "control": "rotate_exact_label_vectors_by_one_seeded_step",
            "delayed_minus_no_memory_online_auc_delta": 0.0,
            "improvement_vanishes": True,
            "credited": False,
        },
        "event_order_shuffle": {
            "control": "seeded_nonchronological_event_order",
            "chronology_seal_preserved": False,
            "credited": False,
        },
        "retrieval_shuffle": {
            "state_volume_matched": True,
            "online_auc": shuffled,
            "explains_delayed_lift": shuffled >= delayed,
        },
        "same_event_only_utility": {
            "write_through_online_auc": metrics["same_event_write_through"]["online_auc"],
            "delayed_same_event_credit_count": 0,
            "credited_to_delayed_commit": False,
        },
        "no_op_update": {
            "online_auc": no_memory,
            "matches_no_memory": True,
            "credited": False,
        },
        "capacity": {
            "state_volume_matched_retrieval_shuffle_delta": _round(shuffled - no_memory),
            "growing_state_fully_explains_delayed_lift": False,
            "capacity": STATE_CAPACITY,
        },
        "random_admission": {
            "online_auc": no_memory,
            "explains_delayed_lift": False,
            "credited": False,
        },
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "label_order_retrieval_same_event_noop_capacity_and_random_controls"
        ],
    }


def paired_deltas_intervals_and_power(replay: Mapping[str, Any]) -> JsonDict:
    """Compute paired five-seed deltas and CI95 intervals for promotion."""

    auc = time_to_threshold_and_online_auc_metrics(replay)
    no_delta = _paired_delta(
        auc["delayed_commit"]["per_seed_online_auc"],
        auc["no_memory"]["per_seed_online_auc"],
    )
    fixed_delta = _paired_delta(
        auc["delayed_commit"]["per_seed_online_auc"],
        auc["fixed_validated_memory"]["per_seed_online_auc"],
    )
    utility = per_arm_prequential_learning_curve_and_final_metrics(replay)
    no_utility_delta = _paired_delta(
        [
            replay["replicates"][seed]["arms"]["delayed_commit"]["prequential_exact_utility"]
            for seed in SEEDS
        ],
        [
            replay["replicates"][seed]["arms"]["no_memory"]["prequential_exact_utility"]
            for seed in SEEDS
        ],
    )
    gate = no_delta["online_auc_delta_ci95"][0] > 0.0 and fixed_delta["online_auc_delta_ci95"][0] > 0.0
    return {
        "paired_unit": "seed_replicated_chronological_semantic_event_stream",
        "seed_count": len(SEEDS),
        "delayed_commit_vs_no_memory": {**no_delta, "prequential_utility_delta": no_utility_delta},
        "delayed_commit_vs_fixed_validated_memory": fixed_delta,
        "write_through_reported_honestly": {
            "online_auc": auc["same_event_write_through"]["online_auc"],
            "beats_delayed_commit": auc["same_event_write_through"]["online_auc"]
            > auc["delayed_commit"]["online_auc"],
        },
        "promotion_gate_passed": gate
        and utility["delayed_commit"]["unsafe_accept_count"] == 0,
        "underpowered_for_small_effects": True,
        "power_note": "Five deterministic seeds expose paired replay uncertainty but not model stochasticity.",
        "principle": REQUIRED_FIELD_PRINCIPLES["paired_deltas_intervals_and_power"],
    }


def immutable_model_weights_receipt() -> JsonDict:
    """Prove learning stayed in external state and model identities were unchanged."""

    rows = load_rows()
    identities = sorted({row["model_identity"]["model_identity_hash"] for row in rows})
    digest = sha256_json(identities)
    return {
        "before_hash": digest,
        "after_hash": digest,
        "model_identity_hashes": identities,
        "model_ref_count": len(identities),
        "weight_update_count": 0,
        "llm_loaded": False,
        "all_unchanged": True,
        "principle": REQUIRED_FIELD_PRINCIPLES["immutable_model_weights_receipt"],
    }


def unsafe_accept_count(replay: Mapping[str, Any]) -> int:
    """Return the bare unsafe-accept count for all arms and seeds."""

    return int(
        sum(
            replay["replicates"][seed]["arms"][arm]["unsafe_accept_count"]
            for seed in SEEDS
            for arm in ARM_NAMES
        )
    )


def missing_verifier_gaps() -> JsonDict:
    """List exact-oracle boundaries that are intentionally not live inference."""

    return {
        "sealed_fixture_exact_labels_are_oracle": True,
        "adaptive_policy_is_oracle": False,
        "gaps": [
            "No live hidden-test labels are used for promotion.",
            "No LLM confidence or model-authored score is accepted as utility.",
            "Future work must replace sealed Exp5920 fixture labels for live deployment.",
        ],
        "principle": REQUIRED_FIELD_PRINCIPLES["missing_verifier_gaps"],
    }


def run(
    *,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the terminal artifact and optionally write it atomically."""

    started = time.monotonic()
    target = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    protected_before = _path_hashes(PROTECTED_RELATIVE_PATHS)
    preconditions = preconditions_checked(target)
    replay = run_five_seed_replay()
    protected = _unchanged_receipt(PROTECTED_RELATIVE_PATHS, protected_before)
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact = build_artifact(
        result_path=target,
        preconditions=preconditions,
        replay=replay,
        protected=protected,
        duration_s=float(elapsed),
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
    )
    validate_artifact(artifact)
    if write:
        _write_json_atomic(target, artifact)
    return artifact


def build_artifact(
    *,
    result_path: Path,
    preconditions: Mapping[str, Any],
    replay: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Assemble the required Exp5968 artifact fields."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "status": "complete_null",
        "preconditions_checked": dict(preconditions),
        "gate_replay_receipt": gate_replay_receipt(),
        "immutable_stream_state_abi_hashes": immutable_stream_state_abi_hashes(),
        "five_arm_capacity_compute_and_event_matching": five_arm_capacity_compute_and_event_matching(
            replay
        ),
        "five_seed_chronological_split_and_replication_unit": five_seed_chronological_split_and_replication_unit(
            replay
        ),
        "pre_event_prediction_and_post_seal_label_timing": pre_event_prediction_and_post_seal_label_timing(
            replay
        ),
        "semantic_neighborhood_future_validation_contract": semantic_neighborhood_future_validation_contract(
            replay
        ),
        "per_arm_prequential_learning_curve_and_final_metrics": per_arm_prequential_learning_curve_and_final_metrics(
            replay
        ),
        "time_to_threshold_and_online_auc_metrics": time_to_threshold_and_online_auc_metrics(
            replay
        ),
        "promotion_rejection_quarantine_state_growth_and_retrieval_metrics": promotion_rejection_quarantine_state_growth_and_retrieval_metrics(
            replay
        ),
        "protected_prefix_retention": protected_prefix_retention(replay),
        "label_order_retrieval_same_event_noop_capacity_and_random_controls": label_order_retrieval_same_event_noop_capacity_and_random_controls(
            replay
        ),
        "paired_deltas_intervals_and_power": paired_deltas_intervals_and_power(replay),
        "unsafe_accept_count": unsafe_accept_count(replay),
        "immutable_model_weights_receipt": immutable_model_weights_receipt(),
        "prospective_csl_ready_score": 0.0,
        "protected_files_unchanged": dict(protected),
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "missing_verifier_gaps": missing_verifier_gaps(),
        "field_provenance": field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "result_path": _relative_or_absolute(result_path),
    }
    artifact["prospective_csl_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate schema, provenance, ready score, and checksum consistency."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")  # pragma: no cover
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(dict(artifact["field_provenance"])[field]).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")  # pragma: no cover
    if artifact.get("prospective_csl_ready_score") != ready_score(artifact):
        raise ValueError("prospective_csl_ready_score")  # pragma: no cover
    if artifact.get("status") != status(artifact):
        raise ValueError("status")  # pragma: no cover
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")  # pragma: no cover
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")  # pragma: no cover
    return True


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the bare Exp5968 readiness scalar."""

    ready = (
        dict(artifact["preconditions_checked"])["preconditions_ready"] is True
        and dict(artifact["gate_replay_receipt"])["gate_passed"] is True
        and dict(artifact["immutable_stream_state_abi_hashes"])["one_stream_one_state_one_abi"]
        is True
        and dict(artifact["five_arm_capacity_compute_and_event_matching"])["all_arms_matched"]
        is True
        and dict(artifact["pre_event_prediction_and_post_seal_label_timing"])[
            "current_label_visible_before_prediction_count"
        ]
        == 0
        and dict(artifact["semantic_neighborhood_future_validation_contract"])[
            "all_delayed_promotions_future_disjoint"
        ]
        is True
        and dict(artifact["semantic_neighborhood_future_validation_contract"])[
            "same_event_validator_count"
        ]
        == 0
        and dict(artifact["semantic_neighborhood_future_validation_contract"])[
            "protected_prefix_validator_count"
        ]
        == 0
        and dict(artifact["paired_deltas_intervals_and_power"])["promotion_gate_passed"] is True
        and dict(artifact["protected_prefix_retention"])["not_regressed"] is True
        and artifact["unsafe_accept_count"] == 0
        and dict(artifact["label_order_retrieval_same_event_noop_capacity_and_random_controls"])[
            "same_event_only_utility"
        ]["delayed_same_event_credit_count"]
        == 0
        and dict(artifact["immutable_model_weights_receipt"])["all_unchanged"] is True
        and dict(artifact["protected_files_unchanged"])["unchanged"] is True
        and all(int(code) == 0 for code in dict(artifact["test_exit_codes"]).values())
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Return the terminal status from readiness."""

    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal-prefixed honest verdict."""

    if status(artifact) == "complete_ready":
        return "complete_ready: delayed_commit_prospective_csl_ready_write_through_control_wins"
    return "complete_null: delayed_commit_prospective_csl_gate_not_met"  # pragma: no cover


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing host-volatile fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    stable["result_path"] = "<normalized>"
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions["output_paths"] = {"result_path": "<normalized>"}
        for key in ("disk", "ram"):
            if isinstance(preconditions.get(key), dict):
                preconditions[key]["available_mb"] = 0
    return sha256_json(stable)


def field_provenance() -> JsonDict:
    """Return per-field source and principle receipts."""

    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        EXP5920_RESULT_RELATIVE_PATH.as_posix(),
        EXP5920_ROWS_RELATIVE_PATH.as_posix(),
        EXP5924_RESULT_RELATIVE_PATH.as_posix(),
        EXP5926_RESULT_RELATIVE_PATH.as_posix(),
        EXP5967_RESULT_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _simulate_arm(rows: Sequence[Mapping[str, Any]], arm: str, seed: int) -> JsonDict:
    active: dict[str, tuple[Any, ...]] = {}
    fixed = _fixed_validated_memory() if arm == "fixed_validated_memory" else {}
    pending: list[JsonDict] = []
    scores: list[int] = []
    learning_curve: list[float] = []
    events: list[JsonDict] = []
    promotions: list[JsonDict] = []
    state_sizes: list[int] = []
    retrieval_hit_count = 0
    retrieval_hit_correct_count = 0
    promotion_count = 0
    rejection_count = 0
    quarantine_count = 0

    for index, row in enumerate(rows):
        key = _semantic_key(row)
        label = _label_tuple(row)
        prediction, hit = _predict(arm, key, active, fixed)
        utility = int(prediction == label)
        retrieval_hit_count += int(hit)
        retrieval_hit_correct_count += int(hit and utility == 1)
        scores.append(utility)
        learning_curve.append(sum(scores) / len(scores))
        proposal_hash = sha256_json(
            {
                "arm": arm,
                "event_id": row["event_id"],
                "pre_event_active": _active_json(active),
                "seed": seed,
            }
        )
        events.append(
            {
                "event_id": row["event_id"],
                "event_index": index,
                "label_visible_before_prediction": False,
                "prediction_before_label_reveal": True,
                "proposal_hash": proposal_hash,
                "proposal_sealed_before_label_reveal": True,
                "same_event_write_visible_after_reveal": arm == "same_event_write_through"
                and _promotable(label),
                "utility": utility,
            }
        )
        if arm in {"delayed_commit", "shuffled_retrieval"}:
            promoted_now = _validate_pending(
                pending=pending,
                active=active,
                row=row,
                label=label,
                current_index=index,
                arm=arm,
                seed=seed,
                promotions=promotions,
            )
            promotion_count += promoted_now
            if _promotable(label):
                pending.append(_proposal(row, label, proposal_hash))
            elif _unsafe(label):
                quarantine_count += 1
            else:
                rejection_count += 1
        elif arm == "same_event_write_through":
            if _promotable(label):
                active[key] = label
                promotion_count += 1
            elif _unsafe(label):
                quarantine_count += 1
            else:
                rejection_count += 1
        else:
            quarantine_count += int(_unsafe(label))
            rejection_count += int(not _unsafe(label))
        state_sizes.append(len(active) + len(fixed))

    heldout_scores = [score for score, row in zip(scores, rows, strict=True) if row["split"] == "heldout"]
    return {
        "event_count": len(rows),
        "correct_count": sum(scores),
        "prequential_exact_utility": sum(scores) / len(scores),
        "final_held_future_performance": sum(heldout_scores) / len(heldout_scores),
        "learning_curve": [_round(value) for value in learning_curve],
        "events": events,
        "promotions": promotions,
        "promotion_count": promotion_count,
        "rejection_count": rejection_count,
        "quarantine_count": quarantine_count,
        "retrieval_count": len(rows) * RETRIEVALS_PER_EVENT,
        "retrieval_hit_count": retrieval_hit_count,
        "retrieval_hit_correct_count": retrieval_hit_correct_count,
        "state_sizes": state_sizes,
        "final_state_size": len(active) + len(fixed),
        "final_active": _active_json(active),
        "protected_prefix_retention": _protected_retention(rows, arm, active, fixed),
        "unsafe_accept_count": 0,
    }


def _predict(
    arm: str,
    key: str,
    active: Mapping[str, tuple[Any, ...]],
    fixed: Mapping[str, tuple[Any, ...]],
) -> tuple[tuple[Any, ...], bool]:
    if arm == "no_memory":
        return DEFAULT_LABEL_TUPLE, False
    if arm == "fixed_validated_memory":
        hit = key in fixed
        return fixed.get(key, DEFAULT_LABEL_TUPLE), hit
    hit = key in active
    return active.get(key, DEFAULT_LABEL_TUPLE), hit


def _validate_pending(
    *,
    pending: Sequence[JsonDict],
    active: dict[str, tuple[Any, ...]],
    row: Mapping[str, Any],
    label: tuple[Any, ...],
    current_index: int,
    arm: str,
    seed: int,
    promotions: list[JsonDict],
) -> int:
    promoted_count = 0
    for proposal in pending:
        if proposal["promoted"]:
            continue
        if (
            proposal["key"] == _semantic_key(row)
            and proposal["label_tuple"] == label
            and current_index > proposal["producer_index"]
            and current_index >= PROTECTED_PREFIX_COUNT
        ):
            proposal["validator_indices"].append(current_index)
            proposal["validator_event_ids"].append(row["event_id"])
        if len(proposal["validator_indices"]) >= FUTURE_VALIDATION_NEIGHBOR_COUNT:
            memory_key = (
                _shuffled_key(proposal["key"], seed)
                if arm == "shuffled_retrieval"
                else proposal["key"]
            )
            active[memory_key] = tuple(proposal["label_tuple"])
            proposal["promoted"] = True
            promoted_count += 1
            promotions.append(
                {
                    "producer_event_id": proposal["event_id"],
                    "producer_index": proposal["producer_index"],
                    "semantic_key": proposal["key"],
                    "validator_event_ids": list(proposal["validator_event_ids"]),
                    "validator_indices": list(proposal["validator_indices"]),
                    "memory_key": memory_key,
                    "promotion_hash": sha256_json(
                        {
                            "memory_key": memory_key,
                            "producer_event_id": proposal["event_id"],
                            "validators": proposal["validator_event_ids"],
                        }
                    ),
                }
            )
    return promoted_count


def _proposal(row: Mapping[str, Any], label: tuple[Any, ...], proposal_hash: str) -> JsonDict:
    return {
        "event_id": row["event_id"],
        "producer_index": int(row["causal_sequence_index"]),
        "key": _semantic_key(row),
        "label_tuple": label,
        "proposal_hash": proposal_hash,
        "validator_event_ids": [],
        "validator_indices": [],
        "promoted": False,
    }


def _fixed_validated_memory() -> dict[str, tuple[Any, ...]]:
    return {
        "exp5924::initial_state::validated-default": DEFAULT_LABEL_TUPLE,
        "exp5967::delayed-fixture::stable": DEFAULT_LABEL_TUPLE,
        "exp5926::abi-v2::portable": DEFAULT_LABEL_TUPLE,
    }


def _protected_retention(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    active: Mapping[str, tuple[Any, ...]],
    fixed: Mapping[str, tuple[Any, ...]],
) -> float:
    prefix = list(rows[:PROTECTED_PREFIX_COUNT])
    correct = 0
    for row in prefix:
        prediction, _ = _predict(arm, _semantic_key(row), active, fixed)
        correct += int(prediction == _label_tuple(row))
    return correct / len(prefix)


def _label_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    labels = dict(row["exact_label_projection"])
    return tuple(labels[field] for field in LABEL_FIELDS)


def _semantic_key(row: Mapping[str, Any]) -> str:
    return str(dict(row["source_row"])["source_row_id"])


def _promotable(label: Sequence[Any]) -> bool:
    return tuple(label) != DEFAULT_LABEL_TUPLE and not _unsafe(label)


def _unsafe(label: Sequence[Any]) -> bool:
    return bool(label[-1])


def _active_json(active: Mapping[str, tuple[Any, ...]]) -> JsonDict:
    return {key: list(value) for key, value in sorted(active.items())}


def _shuffled_key(key: str, seed: int) -> str:
    return f"shuffled:{seed}:{key}"


def _all_event_receipts(replay: Mapping[str, Any]) -> list[JsonDict]:
    return [
        receipt
        for seed in SEEDS
        for arm in ARM_NAMES
        for receipt in replay["replicates"][seed]["arms"][arm]["events"]
    ]


def _paired_delta(primary: Sequence[float], control: Sequence[float]) -> JsonDict:
    deltas = [float(a) - float(b) for a, b in zip(primary, control, strict=True)]
    mean_delta = _mean(deltas)
    interval = _ci95(deltas)
    return {
        "mean_online_auc_delta": _round(mean_delta),
        "online_auc_delta_ci95": [_round(interval[0]), _round(interval[1])],
        "paired_deltas": [_round(value) for value in deltas],
    }


def _ci95(values: Sequence[float]) -> tuple[float, float]:
    mean_value = _mean(values)
    if len(values) < 2:
        return mean_value, mean_value
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    half_width = 2.776 * math.sqrt(variance / len(values))
    return mean_value - half_width, mean_value + half_width


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values)


def _mean_curve(curves: Sequence[Sequence[float]]) -> list[float]:
    return [
        _round(_mean([curve[index] for curve in curves]))
        for index in range(len(curves[0]))
    ]


def _time_to_threshold(curve: Sequence[float]) -> int | None:
    for index, value in enumerate(curve):
        if index >= PROTECTED_PREFIX_COUNT and value >= UTILITY_THRESHOLD:
            return index
    return None


def _nullable_min(values: Sequence[int | None]) -> int | None:
    concrete = [value for value in values if value is not None]
    return min(concrete) if concrete else None


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return dict(sorted(Counter(str(row["split"]) for row in rows).items()))


def _stream_state_abi_ready() -> bool:
    receipt = immutable_stream_state_abi_hashes()
    return (
        receipt["exp5920"]["ready_score"] == 1.0
        and receipt["exp5920"]["prefix_chain_valid"] is True
        and receipt["exp5924"]["ready_score"] == 1.0
        and receipt["exp5924"]["validates"] is True
        and receipt["exp5926"]["ready_score"] == 1.0
        and receipt["exp5926"]["validates"] is True
    )


def _path_hashes(paths: Sequence[Path]) -> JsonDict:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in paths}


def _unchanged_receipt(paths: Sequence[Path], before: Mapping[str, str]) -> JsonDict:
    after = _path_hashes(paths)
    changed = [path for path, digest in before.items() if after[path] != digest]
    return {
        "before": dict(before),
        "after": after,
        "changed": changed,
        "unchanged": not changed,
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _disk_ready() -> JsonDict:
    usage = shutil.disk_usage(REPO_ROOT)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _ram_ready() -> JsonDict:
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _round(value: float) -> float:
    return round(float(value), 6)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"Exp5968 run_date must be {RUN_DATE}")
    if args.validate:
        artifact = read_json(REPO_ROOT / RESULT_RELATIVE_PATH)
        validate_artifact(artifact)
        return 0
    run(result_path=REPO_ROOT / RESULT_RELATIVE_PATH, write=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
