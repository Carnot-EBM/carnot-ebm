"""Measure next-frame support from real adapter-withheld E3 trajectories.

REQ-ARC-WMTE-7597. The measurement uses only public frames, runtime legal
actions, selected actions, level boundaries, and public termination. It does
not load a model or infer a solve from an already-cleared public game.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import yaml

from carnot.experiment_7589_v663_arc_output_boundary import (
    exercise_learning_lifecycle,
    full_frame_hash,
)
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    build_scoped_commands,
    run_commands,
)

Json = dict[str, Any]
EXPERIMENT_ID = 7597
MILESTONE = "2026.09.663"
RUN_DATE = "20260924"
REQUIREMENT_ID = "REQ-ARC-WMTE-7597"
SCHEMA = "carnot.experiment_7597_v663_arc_history_generalization.v1"
RAW_EPISODE_SCHEMA = "carnot.exp7597.raw_public_episode.v1"
GAMES = ("su15", "sp80", "ft09", "sb26", "g50t", "dc22")
SEEDS = (7_597_001, 7_597_002)
HISTORY_LENGTHS = (0, 1, 2, 4)
ACTION_LIMIT = 600
MIN_MATCHED_KEYS = 20
MIN_QUALIFYING_GAMES = 3
BOOTSTRAP_SEED = 7_597_003
MODEL_SPECS: list[Json] = []

RESULT_REL = Path("results/experiment_7597_v663_arc_history_generalization.json")
RAW_REL = Path("results/raw/experiment_7597_v663_arc_history_generalization")
MODULE_REL = Path("python/carnot/experiment_7597_v663_arc_history_generalization.py")
TEST_REL = Path("tests/python/test_experiment_7597_v663_arc_history_generalization.py")
WRAPPER_REL = Path("scripts/experiments/experiment_7597_v663_arc_history_generalization.py")
AGENT_REL = Path("python/carnot/agentic/arc_competition_agent.py")
SPEC_REL = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_REL = Path("ops/arc_solve_registry.yaml")
REQUIRED_SCOPED_CHECKS = REQUIRED_CHECK_NAMES

REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "sample_size_budget",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "validation_receipts",
    "verifier_is_oracle",
    "field_principles",
    "arc_history_measurement_ready_score",
    "history_support_ready_score",
    "per_game_results",
    "solve_provenance",
    "trajectory_supervisor",
    "live_llm_invoked",
}

FIELD_PRINCIPLES = {
    "honest_verdict": "Use a complete_ terminal prefix; execution completion does not prove benefit.",
    "verdict_class": "Use exactly one closed verdict class; only unfinished owned work is partial.",
    "flagged_adversarial": "Persist the exact terminal reader outcome; flagged evidence opens no gate.",
    "gate_check_summary": "Every blocked verdict names check, upstream, path, field, operator, expected, and observed.",
    "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness remain separate.",
    "rows": "Each episode and history arm retains absolute operands, missingness, and provenance.",
    "sample_size_budget": "Games are independent units; seeds and windows do not multiply clusters.",
    "inference_substrate": "The value states the actual live-agent offline runtime without an LLM.",
    "inference_substrate_class": "Actual and planned classes remain separate; blocked_no_run means zero model work.",
    "MODEL_SPECS": "No current LLM call means the model list is empty.",
    "invocation_counts": "Loads, forwards, generations, and tokens are counted independently.",
    "duration_s": "Monotonic current phase time is never padded or inherited.",
    "random_seed": "Every stochastic episode and bootstrap stage has an explicit seed.",
    "reproducibility_checksum": "One digest binds raw evidence, configuration, and terminal reduction.",
    "source_artifact_hashes": "Authenticated producers remain distinct from missing and pre-gate inputs.",
    "validation_receipts": "Commands, worktree, exits, and raw log hashes bind validation claims.",
    "verifier_is_oracle": "Exact frame equality is truth for this interface audit, not a learned verifier claim.",
    "field_principles": "One-line reasons preserve field meaning outside the task prompt.",
    "arc_history_measurement_ready_score": "One requires twelve causal live-path episode receipts and fresh reduction.",
    "history_support_ready_score": "One requires twenty matched repeated keys on at least three games.",
    "per_game_results": "Every game, seed, and history length retains support, conflict, boundaries, and censoring.",
    "solve_provenance": "Incidental progress is live-agent self-discovery; registered levels remain duplicates.",
    "trajectory_supervisor": "Only actual fired and helped counts are reported; no firing means no refinement.",
    "live_llm_invoked": "False because this experiment measures the observation interface, not generation.",
}


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Emit a flushed monotonic phase boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7597] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so removed evidence changes identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes for custody receipts."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def frame_sha256(frame: Any) -> str:
    """Hash the full public frame, including array shape and dtype."""

    return full_frame_hash(frame)


def frozen_episode_plan() -> list[Json]:
    """Freeze the twelve live E3 episodes without starting environment work."""

    return [
        {
            "episode_id": f"{game}:{seed}",
            "game": game,
            "seed": seed,
            "policy": "E3AgentPolicy",
            "action_limit": ACTION_LIMIT,
            "adapter_withheld": True,
            "stored_solutions_withheld": True,
            "induction_disabled": True,
            "live_llm_invoked": False,
        }
        for game in GAMES
        for seed in SEEDS
    ]


def _action_kind(action: Mapping[str, Any]) -> Any:
    return action.get("kind")


def _causal_key(
    observation_sha256: str,
    action: Mapping[str, Any],
    level: int,
    history: Sequence[Mapping[str, Any]],
    length: int,
) -> str:
    prior = list(history[-length:]) if length else []
    return canonical_hash(
        {
            "history_length": length,
            "prior_observation_actions": prior,
            "current_observation_sha256": observation_sha256,
            "current_action": deepcopy(dict(action)),
            "observed_level": int(level),
        }
    )


def validate_raw_episode(raw: Mapping[str, Any]) -> list[str]:
    """Reject any episode that is not an adapter-withheld, no-model E3 run."""

    errors: list[str] = []
    expected = {
        "schema": RAW_EPISODE_SCHEMA,
        "policy": "E3AgentPolicy",
        "action_limit": ACTION_LIMIT,
        "induction_disabled": True,
        "adapter_withheld": True,
        "stored_solutions_withheld": True,
        "game_source_read": False,
        "hidden_state_read": False,
        "offline_ground_truth_bfs": False,
        "live_llm_invoked": False,
    }
    labels = {
        "policy": "wrong_policy",
        "induction_disabled": "induction_on",
        "adapter_withheld": "adapter_used",
        "stored_solutions_withheld": "stored_solution",
        "game_source_read": "source_read",
        "hidden_state_read": "hidden_state",
        "offline_ground_truth_bfs": "ground_truth",
        "live_llm_invoked": "llm",
        "qualified_observer": "observer_missing",
    }
    for field, wanted in expected.items():
        if raw.get(field) != wanted:
            errors.append(labels.get(field, f"{field}_mismatch"))
    observer = raw.get("qualified_observer")
    if not isinstance(observer, Mapping) or observer.get("enabled") is not True:
        errors.append("observer_missing")
    if raw.get("game") not in GAMES:
        errors.append("game_not_frozen")
    if raw.get("seed") not in SEEDS:
        errors.append("seed_not_frozen")
    steps = raw.get("steps")
    if not isinstance(steps, list) or len(steps) > ACTION_LIMIT:
        errors.append("steps_invalid")
        return list(dict.fromkeys(errors))
    for index, row in enumerate(steps):
        if not isinstance(row, Mapping) or row.get("action_index") != index:
            errors.append(f"action_index_invalid:{index}")
            continue
        observation = row.get("observation")
        if observation is not None:
            if not isinstance(observation, Mapping) or "frame" not in observation:
                errors.append(f"observation_invalid:{index}")
            elif "legal_actions" not in observation:
                errors.append(f"legal_actions_missing:{index}")
        outcome = row.get("outcome")
        if outcome is not None and (not isinstance(outcome, Mapping) or "frame" not in outcome):
            errors.append(f"outcome_invalid:{index}")
        if not isinstance(row.get("action"), Mapping):
            errors.append(f"action_invalid:{index}")
    return list(dict.fromkeys(errors))


def _key_summary(targets_by_key: Mapping[str, Counter[str]]) -> Json:
    key_rows: list[Json] = []
    repeated_occurrences = 0
    conflict_numerator = 0
    conflict_denominator = 0
    for key, targets in targets_by_key.items():
        support = int(sum(targets.values()))
        modal = max(targets.values())
        numerator = support - int(modal)
        if support == 1:
            status = "unknown_support"
        elif len(targets) == 1:
            status = "repeated_consistent"
        else:
            status = "conflicting"
        if support >= 2:
            repeated_occurrences += support
            conflict_numerator += numerator
            conflict_denominator += support
        distribution = [
            {"target_sha256": target, "count": int(count)}
            for target, count in sorted(targets.items(), key=lambda item: (-item[1], item[0]))
        ]
        key_rows.append(
            {
                "key_sha256": key,
                "support_count": support,
                "target_distribution": distribution,
                "status": status,
                "conditional_conflict_numerator": numerator if support >= 2 else None,
                "conditional_conflict_denominator": support if support >= 2 else None,
                "conditional_conflict_rate": numerator / support if support >= 2 else None,
            }
        )
    key_rows.sort(key=lambda row: row["key_sha256"])
    return {
        "key_count": len(key_rows),
        "repeated_key_count": sum(row["support_count"] >= 2 for row in key_rows),
        "singleton_unknown_key_count": sum(row["support_count"] == 1 for row in key_rows),
        "conflicting_key_count": sum(row["status"] == "conflicting" for row in key_rows),
        "repeated_occurrence_count": repeated_occurrences,
        "conflict_numerator": conflict_numerator,
        "conflict_denominator": conflict_denominator,
        "conditional_conflict_rate": (
            conflict_numerator / conflict_denominator if conflict_denominator else None
        ),
        "key_rows": key_rows,
    }


def reduce_episode(raw: Mapping[str, Any]) -> Json:
    """Reduce one raw public trajectory without consulting stored scores."""

    errors = validate_raw_episode(raw)
    if errors:
        raise ValueError("raw_episode_invalid:" + ",".join(errors))
    history: list[Json] = []
    supports: dict[str, dict[str, Counter[str]]] = {str(length): {} for length in HISTORY_LENGTHS}
    transition_rows: list[Json] = []
    reset_count = 0
    level_boundary_count = 0
    for raw_step in raw["steps"]:
        action = deepcopy(dict(raw_step["action"]))
        reset = bool(raw_step.get("reset_boundary") or _action_kind(action) == "RESET")
        if reset:
            reset_count += 1
            history.clear()
        observation = raw_step.get("observation")
        outcome = raw_step.get("outcome")
        if observation is None or outcome is None:
            continue
        observation_sha256 = frame_sha256(observation["frame"])
        target_sha256 = frame_sha256(outcome["frame"])
        level_before = int(observation.get("level", 0))
        level_after = int(outcome.get("level", level_before))
        keys = {
            str(length): _causal_key(observation_sha256, action, level_before, history, length)
            for length in HISTORY_LENGTHS
        }
        used = {
            str(length): min(length, len(history)) if length else 0 for length in HISTORY_LENGTHS
        }
        for label, key in keys.items():
            counter = supports[label].setdefault(key, Counter())
            counter[target_sha256] += 1
        level_boundary = bool(raw_step.get("level_boundary") or level_after != level_before)
        level_boundary_count += int(level_boundary)
        transition_rows.append(
            {
                "action_index": int(raw_step["action_index"]),
                "observation_sha256": observation_sha256,
                "target_sha256": target_sha256,
                "action": action,
                "keys": keys,
                "history_items_used": used,
                "level_before": level_before,
                "level_after": level_after,
                "reset_boundary": reset,
                "level_boundary": level_boundary,
            }
        )
        history.append({"observation_sha256": observation_sha256, "action": action})
        history = history[-max(HISTORY_LENGTHS) :]
        if reset or level_boundary:
            history.clear()
    per_history = {label: _key_summary(by_key) for label, by_key in supports.items()}
    eligible = len(transition_rows)
    for result in per_history.values():
        result["repeat_coverage"] = (
            result["repeated_occurrence_count"] / eligible if eligible else None
        )
    return {
        "episode_id": raw["episode_id"],
        "game": raw["game"],
        "seed": int(raw["seed"]),
        "action_opportunities": len(raw["steps"]),
        "eligible_transition_count": eligible,
        "reset_count": reset_count,
        "level_boundary_count": level_boundary_count,
        "legal_action_receipt_count": sum(
            int(row.get("observation") is not None) for row in raw["steps"]
        ),
        "termination": deepcopy(raw.get("termination")),
        "trajectory_supervisor": reduce_trajectory_supervisor(raw["trajectory_supervisor"]),
        "transition_rows": transition_rows,
        "per_history": per_history,
    }


def _key_row_map(reduction: Mapping[str, Any], label: str) -> dict[str, Json]:
    return {
        str(row["key_sha256"]): dict(row) for row in reduction["per_history"][label]["key_rows"]
    }


def _modal_target(row: Mapping[str, Any]) -> str:
    distribution = list(row.get("target_distribution") or [])
    if not distribution:
        return ""
    return str(distribution[0]["target_sha256"])


def compare_seed_reductions(first: Mapping[str, Any], second: Mapping[str, Any]) -> Json:
    """Use seed one for candidates and seed two for supported predictions."""

    if first.get("game") != second.get("game"):
        raise ValueError("seed_game_mismatch")
    if (first.get("seed"), second.get("seed")) != SEEDS:
        raise ValueError("seed_order_mismatch")
    per_history: dict[str, Json] = {}
    candidate_maps: dict[str, dict[str, Json]] = {}
    evaluation_maps: dict[str, dict[str, Json]] = {}
    for length in HISTORY_LENGTHS:
        label = str(length)
        candidates = _key_row_map(first, label)
        evaluation = _key_row_map(second, label)
        candidate_maps[label] = candidates
        evaluation_maps[label] = evaluation
        exact = conflicts = supported = unseen = singleton = 0
        seen_supported_keys: set[str] = set()
        for transition in second["transition_rows"]:
            key = str(transition["keys"][label])
            candidate = candidates.get(key)
            if candidate is None:
                unseen += 1
                continue
            if int(candidate["support_count"]) < 2:
                singleton += 1
                continue
            supported += 1
            seen_supported_keys.add(key)
            predicted = _modal_target(candidate)
            matched = predicted == transition["target_sha256"]
            exact += int(matched)
            conflicts += int(not matched)
        per_history[label] = {
            "history_length": length,
            "candidate_key_count": len(candidates),
            "repeated_candidate_key_count": sum(
                int(row["support_count"] >= 2) for row in candidates.values()
            ),
            "repeated_supported_key_count": len(seen_supported_keys),
            "supported_predictive_denominator": supported,
            "exact_prediction_numerator": exact,
            "cross_seed_conflict_numerator": conflicts,
            "cross_seed_conflict_rate": conflicts / supported if supported else None,
            "unseen_key_censored_count": unseen,
            "singleton_candidate_censored_count": singleton,
            "seed_1_within_conflict_numerator": first["per_history"][label]["conflict_numerator"],
            "seed_1_within_conflict_denominator": first["per_history"][label][
                "conflict_denominator"
            ],
            "seed_2_within_conflict_numerator": second["per_history"][label]["conflict_numerator"],
            "seed_2_within_conflict_denominator": second["per_history"][label][
                "conflict_denominator"
            ],
        }

    matched_transitions: list[Mapping[str, Any]] = []
    for transition in second["transition_rows"]:
        if _action_kind(transition["action"]) == "RESET":
            continue
        supported_everywhere = True
        for length in HISTORY_LENGTHS:
            label = str(length)
            key = str(transition["keys"][label])
            candidate = candidate_maps[label].get(key)
            evaluated = evaluation_maps[label].get(key)
            if (
                candidate is None
                or evaluated is None
                or int(candidate["support_count"]) < 2
                or int(evaluated["support_count"]) < 2
            ):
                supported_everywhere = False
                break
        if supported_everywhere:
            matched_transitions.append(transition)

    matched_rows: dict[str, Json] = {}
    distinct_counts: list[int] = []
    for length in HISTORY_LENGTHS:
        label = str(length)
        conflict = 0
        keys: set[str] = set()
        for transition in matched_transitions:
            key = str(transition["keys"][label])
            keys.add(key)
            conflict += int(
                _modal_target(candidate_maps[label][key]) != transition["target_sha256"]
            )
        distinct_counts.append(len(keys))
        matched_rows[label] = {
            "history_length": length,
            "transition_count": len(matched_transitions),
            "repeated_key_count": len(keys),
            "conflict_numerator": conflict,
            "conflict_denominator": len(matched_transitions),
            "conflict_rate": (conflict / len(matched_transitions) if matched_transitions else None),
        }
    floor_count = min(distinct_counts) if distinct_counts else 0
    eligible = floor_count >= MIN_MATCHED_KEYS
    return {
        "game": first["game"],
        "candidate_seed": first["seed"],
        "evaluation_seed": second["seed"],
        "per_history": per_history,
        "matched_common_support": {
            "transition_count": len(matched_transitions),
            "transition_indices": [row["action_index"] for row in matched_transitions],
            "support_floor_key_count": floor_count,
            "minimum_required": MIN_MATCHED_KEYS,
            "eligible_for_cross_game": eligible,
            "reason": ("support_floor_met" if eligible else "insufficient_matched_repeat_support"),
            "per_history": matched_rows,
        },
    }


def cross_game_history_comparison(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    n_bootstrap: int = 2_000,
) -> Json:
    """Compare matched history arms with games as the only bootstrap clusters."""

    qualifying = [
        row
        for row in comparisons
        if row.get("matched_common_support", {}).get("eligible_for_cross_game") is True
    ]
    games = [str(row["game"]) for row in qualifying]
    ready = len(qualifying) >= MIN_QUALIFYING_GAMES
    base = {
        "ready": ready,
        "reason": "support_floor_met" if ready else "insufficient_support",
        "qualifying_games": games,
        "qualifying_game_count": len(games),
        "required_game_count": MIN_QUALIFYING_GAMES,
        "minimum_matched_repeated_keys_per_game": MIN_MATCHED_KEYS,
        "independent_cluster_count": len(games),
        "cluster_unit": "game",
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_draw_count": int(n_bootstrap) if ready else 0,
        "bootstrap_differences": None,
    }
    if not ready:
        return base
    rng = random.Random(bootstrap_seed)
    output: dict[str, Json] = {}
    for length in (1, 2, 4):
        differences = []
        for row in qualifying:
            matched = row["matched_common_support"]["per_history"]
            differences.append(
                float(matched["0"]["conflict_rate"]) - float(matched[str(length)]["conflict_rate"])
            )
        draws = [
            sum(rng.choice(differences) for _ in differences) / len(differences)
            for _ in range(n_bootstrap)
        ]
        ordered = sorted(draws)
        output[str(length)] = {
            "comparison": "history_0_conflict_rate_minus_longer_history",
            "cluster_unit": "game",
            "game_count": len(differences),
            "mean_difference": sum(differences) / len(differences),
            "bootstrap_p025": ordered[int(0.025 * (len(ordered) - 1))],
            "bootstrap_p975": ordered[int(0.975 * (len(ordered) - 1))],
            "descriptive_only": True,
        }
    base["bootstrap_differences"] = output
    return base


def reduce_trajectory_supervisor(receipt: Mapping[str, Any]) -> Json:
    """Report only measured arm outcomes; an empty ledger supports no refinement."""

    outcomes = receipt.get("arm_outcomes")
    outcomes = outcomes if isinstance(outcomes, Mapping) else {}
    fired = sum(int(row.get("fired", 0)) for row in outcomes.values() if isinstance(row, Mapping))
    helped = sum(int(row.get("helped", 0)) for row in outcomes.values() if isinstance(row, Mapping))
    return {
        "fired": fired,
        "helped": helped,
        "supervisor_refinement_supported": False,
        "reason": (
            "no_firings_nothing_to_refine"
            if fired == 0
            else "measurement_only_no_arm_change_authorized"
        ),
    }


def independent_reduce_rows(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Reduce row operands without treating missing support as a perfect score."""

    by_arm: dict[str, Json] = {}
    signs = {"positive": 0, "zero": 0, "negative": 0, "missing": 0}
    for row in rows:
        numerator = int(row.get("numerator") or 0)
        denominator = int(row.get("denominator") or 0)
        if numerator < 0 or denominator < 0 or numerator > denominator:
            raise ValueError(f"invalid_row_operands:{row.get('unit')}:{row.get('arm')}")
        arm = str(row.get("arm"))
        reduced = by_arm.setdefault(
            arm, {"numerator": 0, "denominator": 0, "row_count": 0, "missing_count": 0}
        )
        reduced["numerator"] += numerator
        reduced["denominator"] += denominator
        reduced["row_count"] += 1
        reduced["missing_count"] += int(row.get("missing") is True)
        metric = row.get("absolute_metric")
        if metric is None:
            signs["missing"] += 1
        elif float(metric) > 0:
            signs["positive"] += 1
        elif float(metric) < 0:
            signs["negative"] += 1
        else:
            signs["zero"] += 1
    return {
        "row_count": len(rows),
        "missing_count": sum(int(row.get("missing") is True) for row in rows),
        "censored_count": sum(int(row.get("censored") is True) for row in rows),
        "sign_counts": signs,
        "by_arm": by_arm,
    }


def _per_game_rows(reductions: Sequence[Mapping[str, Any]]) -> list[Json]:
    rows: list[Json] = []
    for episode in reductions:
        for length in HISTORY_LENGTHS:
            summary = episode["per_history"][str(length)]
            denominator = int(summary["conflict_denominator"])
            numerator = int(summary["conflict_numerator"])
            missing = denominator == 0
            rows.append(
                {
                    "unit": f"{episode['game']}:{episode['seed']}",
                    "arm": f"history_{length}",
                    "game": episode["game"],
                    "seed": int(episode["seed"]),
                    "history_length": length,
                    "absolute_metric": (numerator / denominator if denominator else None),
                    "numerator": numerator,
                    "denominator": denominator,
                    "direction": "lower_is_better",
                    "missing": missing,
                    "censored": missing,
                    "provenance": "qualified_public_frame_action_observer",
                    "key_count": summary["key_count"],
                    "repeated_key_count": summary["repeated_key_count"],
                    "singleton_unknown_key_count": summary["singleton_unknown_key_count"],
                    "repeat_coverage": summary["repeat_coverage"],
                    "actions": episode["action_opportunities"],
                    "eligible_transitions": episode["eligible_transition_count"],
                    "levels": episode["level_boundary_count"],
                    "resets": episode["reset_count"],
                    "censoring_reason": "no_repeated_keys" if missing else None,
                }
            )
    return rows


def _support_curve(reductions: Sequence[Mapping[str, Any]]) -> list[Json]:
    curve: list[Json] = []
    for length in HISTORY_LENGTHS:
        label = str(length)
        repeated_keys = sum(row["per_history"][label]["repeated_key_count"] for row in reductions)
        total_keys = sum(row["per_history"][label]["key_count"] for row in reductions)
        repeated_occurrences = sum(
            row["per_history"][label]["repeated_occurrence_count"] for row in reductions
        )
        transitions = sum(row["eligible_transition_count"] for row in reductions)
        numerator = sum(row["per_history"][label]["conflict_numerator"] for row in reductions)
        denominator = sum(row["per_history"][label]["conflict_denominator"] for row in reductions)
        curve.append(
            {
                "history_length": length,
                "key_count": total_keys,
                "repeated_key_count": repeated_keys,
                "repeat_coverage": repeated_occurrences / transitions if transitions else None,
                "conflict_numerator": numerator,
                "conflict_denominator": denominator,
                "conditional_conflict_rate": numerator / denominator if denominator else None,
                "singleton_purity_is_determinism_evidence": False,
            }
        )
    return curve


def reduce_raw_episode_receipts(receipts: Sequence[Mapping[str, Any]]) -> Json:
    """Authenticate and independently reduce all twelve raw episode files."""

    errors: list[str] = []
    reductions: list[Json] = []
    expected_units = {(game, seed) for game in GAMES for seed in SEEDS}
    observed_units: set[tuple[str, int]] = set()
    for receipt in receipts:
        game = str(receipt.get("game"))
        seed = int(receipt.get("seed", -1))
        label = f"{game}:{seed}"
        path = Path(str(receipt.get("path") or ""))
        if not path.is_file():
            errors.append(f"raw_missing:{label}")
            continue
        observed_hash = sha256_file(path)
        if observed_hash != receipt.get("sha256"):
            errors.append(f"raw_hash_mismatch:{label}")
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw_errors = validate_raw_episode(raw)
            if raw_errors:
                errors.extend(f"raw_invalid:{label}:{error}" for error in raw_errors)
                continue
            reduced = reduce_episode(raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"raw_reduce_error:{label}:{type(exc).__name__}")
            continue
        if (reduced["game"], reduced["seed"]) != (game, seed):
            errors.append(f"raw_identity_mismatch:{label}")
            continue
        reductions.append(reduced)
        observed_units.add((game, seed))
    missing_units = sorted(expected_units - observed_units)
    errors.extend(f"episode_missing:{game}:{seed}" for game, seed in missing_units)
    extra_units = sorted(observed_units - expected_units)
    errors.extend(f"episode_extra:{game}:{seed}" for game, seed in extra_units)

    comparisons: list[Json] = []
    by_unit = {(row["game"], row["seed"]): row for row in reductions}
    for game in GAMES:
        first = by_unit.get((game, SEEDS[0]))
        second = by_unit.get((game, SEEDS[1]))
        if first is not None and second is not None:
            comparisons.append(compare_seed_reductions(first, second))
    rows = _per_game_rows(reductions)
    cross_game = cross_game_history_comparison(comparisons, bootstrap_seed=BOOTSTRAP_SEED)
    valid = not errors and observed_units == expected_units and len(reductions) == 12
    return {
        "valid": valid,
        "errors": list(dict.fromkeys(errors)),
        "episode_count": len(reductions),
        "per_episode_reductions": reductions,
        "per_game_comparisons": comparisons,
        "per_game_results": rows,
        "rows": deepcopy(rows),
        "independent_reduction": independent_reduce_rows(rows),
        "support_versus_conflict_curve": _support_curve(reductions),
        "cross_game_comparison": cross_game,
        "arc_history_measurement_ready_score": int(valid),
        "history_support_ready_score": int(valid and cross_game["ready"]),
        "trajectory_supervisor": {
            "fired": sum(row["trajectory_supervisor"]["fired"] for row in reductions),
            "helped": sum(row["trajectory_supervisor"]["helped"] for row in reductions),
            "supervisor_refinement_supported": False,
            "reason": (
                "no_firings_nothing_to_refine"
                if sum(row["trajectory_supervisor"]["fired"] for row in reductions) == 0
                else "measurement_only_no_arm_change_authorized"
            ),
        },
    }


PREREQUISITE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7589_v663_arc_output_boundary.py"),
    AGENT_REL,
    Path("scripts/arc_leaderboard_eval.py"),
    REGISTRY_REL,
    Path("python/carnot/experiment_7580_v662_arc_verifier_support.py"),
    Path("python/carnot/experiment_7562_v661_arc_plan_lineage.py"),
    SPEC_REL,
)


def collect_preconditions(root: Path, private_root: Path) -> tuple[list[Json], dict[str, str]]:
    """Authenticate custody, requirements, registry entries, and private ownership."""

    resolved_root = Path(root).resolve()
    checks: list[Json] = []
    hashes: dict[str, str] = {}
    for relative in PREREQUISITE_PATHS:
        path = resolved_root / relative
        present = path.is_file()
        checks.append(
            {
                "check": "source_custody",
                "upstream": "worktree",
                "path": relative.as_posix(),
                "field": "is_file",
                "op": "==",
                "expected": True,
                "observed": present,
                "passed": present,
            }
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)
    spec_text = (resolved_root / SPEC_REL).read_text(encoding="utf-8")
    present = REQUIREMENT_ID in spec_text
    checks.append(
        {
            "check": "capability_requirement",
            "upstream": "OpenSpec",
            "path": SPEC_REL.as_posix(),
            "field": REQUIREMENT_ID,
            "op": "contains",
            "expected": True,
            "observed": present,
            "passed": present,
        }
    )
    registry = yaml.safe_load((resolved_root / REGISTRY_REL).read_text(encoding="utf-8"))
    registered = sorted(
        row.get("game") for row in registry.get("games", []) if row.get("game") in GAMES
    )
    checks.append(
        {
            "check": "registry_precheck",
            "upstream": "arc_solve_registry",
            "path": REGISTRY_REL.as_posix(),
            "field": "games_already_reproduced_and_ineligible_for_new_credit",
            "op": "==",
            "expected": sorted(GAMES),
            "observed": registered,
            "passed": registered == sorted(GAMES),
        }
    )
    private = Path(private_root).resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    results = (resolved_root / "results").resolve()
    owned = private.is_relative_to(temporary) and not private.is_relative_to(results)
    checks.append(
        {
            "check": "resource_ownership",
            "upstream": "current_exp7597_process",
            "path": str(private),
            "field": "task_owned_private_root_outside_results",
            "op": "==",
            "expected": True,
            "observed": owned,
            "passed": owned,
        }
    )
    checks.append(
        {
            "check": "model_call_declaration",
            "upstream": "current_exp7597_task",
            "path": MODULE_REL.as_posix(),
            "field": "MODEL_SPECS",
            "op": "==",
            "expected": [],
            "observed": MODEL_SPECS,
            "passed": MODEL_SPECS == [],
        }
    )
    return checks, hashes


def _gate(passed: bool, principle: str, expected: Any, observed: Any) -> Json:
    return {
        "passed": bool(passed),
        "principle": principle,
        "expected": expected,
        "observed": observed,
    }


def _zero_invocations() -> Json:
    return {
        **ZERO_INVOCATION_COUNTS,
        "forward_calls_attempted": 0,
        "forward_calls_completed": 0,
        "forward_calls_failed": 0,
        "forward_calls_cancelled": 0,
        "forward_calls_in_flight": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    copied = deepcopy(dict(artifact))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    check: str,
    upstream: str,
    path: str,
    field: str,
    op: str,
    expected: Any,
    observed: Any,
) -> Json:
    """Build a complete blocked artifact with exact failed operands."""

    clean = "".join(character if character.isalnum() else "_" for character in check)
    summary = {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "honest_verdict": f"complete_blocked_{clean}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": summary,
        "acceptance_gate_results": {
            name: _gate(False, principle, True, False)
            for name, principle in {
                "validity": "A failed upstream check blocks dependent validity.",
                "readiness": "Blocked work cannot establish live-path measurement readiness.",
                "benefit": "No benefit claim exists without complete support evidence.",
                "retention": "No lifecycle claim exists without owned execution.",
                "freshness": "Missing current custody cannot be replaced with history.",
            }.items()
        },
        "rows": [],
        "sample_size_budget": {
            "intended_independent_units": 6,
            "observed_independent_units": 0,
            "excluded_independent_units": 0,
            "censored_independent_units": 6,
            "seeds_or_windows_multiply_source_groups": False,
        },
        "inference_substrate": "precondition_check_only_no_model_work",
        "inference_substrate_class": "blocked_no_run",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "live_llm_invoked": False,
        "model_invoked": False,
        "invocation_counts": _zero_invocations(),
        "duration_s": float(duration_s),
        "random_seed": {"episodes": list(SEEDS), "bootstrap": BOOTSTRAP_SEED},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {
            "authenticated_sources": {},
            "missing_producers": [summary],
            "conductor_pre_gate_artifact": {"present": False, "used": False},
        },
        "validation_receipts": [],
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "arc_history_measurement_ready_score": 0,
        "history_support_ready_score": 0,
        "per_game_results": [],
        "solve_provenance": "live_agent_self_discovery",
        "trajectory_supervisor": {
            "fired": 0,
            "helped": 0,
            "supervisor_refinement_supported": False,
            "reason": "blocked_no_run",
        },
        "new_solve_claimed": False,
        "official_leaderboard_score_claimed": False,
        "production_defaults_changed": False,
        "acceptance_threshold_changed": False,
        "qwen_thinking_behavior_changed": False,
        "supervisor_arms_changed": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


E2E_NAMES = (
    "e2e_009",
    "e2e_010",
    "e2e_011",
    "e2e_012",
    "e2e_013",
    "llm_off_environment_smoke",
)
TERMINAL_NAMES = (
    "declared_entrypoint",
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def _receipt_names_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return set(names) <= set(by_name) and all(by_name[name].get("passed") is True for name in names)


def build_artifact(
    *,
    repo_root: Path,
    run_date: str,
    duration_s: float,
    panel: Mapping[str, Any],
    raw_episode_receipts: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    validation_receipts: Sequence[Mapping[str, Any]],
    e2e_receipts: Sequence[Mapping[str, Any]],
    terminal_receipts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]] = (),
    phase_spans: Sequence[Mapping[str, Any]] = (),
    flagged_adversarial: bool = False,
) -> Json:
    """Build the terminal observability result without a policy-benefit claim."""

    measurement_ready = int(panel.get("arc_history_measurement_ready_score") == 1)
    support_ready = int(panel.get("history_support_ready_score") == 1)
    validation_ok = _receipt_names_pass(validation_receipts, REQUIRED_SCOPED_CHECKS)
    e2e_ok = _receipt_names_pass(e2e_receipts, E2E_NAMES)
    terminal_ok = not terminal_receipts or _receipt_names_pass(terminal_receipts, TERMINAL_NAMES)
    validity = bool(panel.get("valid") and validation_ok and e2e_ok and terminal_ok)
    retention = lifecycle.get("passed") is True
    freshness = bool(source_hashes) and all(
        Path(str(row.get("path") or "")).is_file() for row in raw_episode_receipts
    )
    readiness = measurement_ready == 1
    gates = {
        "validity": _gate(
            validity and not flagged_adversarial,
            "All raw reductions, scoped checks, E2Es, and terminal readers must pass exact bytes.",
            True,
            validity and not flagged_adversarial,
        ),
        "readiness": _gate(
            readiness,
            "All twelve causal E3 episode receipts and independent reduction are required.",
            {"arc_history_measurement_ready_score": 1},
            {"arc_history_measurement_ready_score": measurement_ready},
        ),
        "benefit": _gate(
            False,
            "Observed support can justify later history features but cannot prove policy benefit or Markov state.",
            "separate_authorized_policy_intervention",
            "not_run",
        ),
        "retention": _gate(
            retention,
            "Delayed learning counts only after persist, reload, and duplicate rejection.",
            True,
            retention,
        ),
        "freshness": _gate(
            freshness,
            "Current source and raw-log hashes prevent inherited evidence from entering the result.",
            True,
            freshness,
        ),
    }
    operational = ("validity", "readiness", "retention", "freshness")
    failures = [name for name in operational if gates[name]["passed"] is not True]
    if failures:
        verdict = "complete_disqualified_required_validation_or_episode_failure"
        verdict_class = "disqualified"
    elif support_ready:
        verdict = "complete_null_history_support_measured_no_markov_or_policy_benefit_proof"
        verdict_class = "null"
    else:
        verdict = "complete_null_insufficient_history_support"
        verdict_class = "null"
    rows = deepcopy(list(panel.get("rows") or []))
    supervisors = deepcopy(dict(panel.get("trajectory_supervisor") or {}))
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "gate_check_summary": {
            "passed": not failures,
            "failed_count": len(failures),
            "failed_checks": failures,
            "history_support_floor_met": bool(support_ready),
            "benefit_gate_intentionally_closed": True,
        },
        "acceptance_gate_results": gates,
        "rows": rows,
        "independent_reduction": independent_reduce_rows(rows),
        "sample_size_budget": {
            "intended_independent_units": len(GAMES),
            "observed_independent_units": len(
                {row.get("game") for row in panel.get("per_episode_reductions") or []}
            ),
            "excluded_independent_units": 0,
            "censored_independent_units": sum(
                int(not comparison["matched_common_support"]["eligible_for_cross_game"])
                for comparison in panel.get("per_game_comparisons") or []
            ),
            "episode_count": len(raw_episode_receipts),
            "action_opportunity_ceiling": len(GAMES) * len(SEEDS) * ACTION_LIMIT,
            "seeds_or_windows_multiply_source_groups": False,
        },
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "historical_model_identity": [],
        "live_llm_invoked": False,
        "model_invoked": False,
        "invocation_counts": _zero_invocations(),
        "duration_s": float(duration_s),
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {"episodes": list(SEEDS), "bootstrap": BOOTSTRAP_SEED},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {
            "authenticated_sources": dict(source_hashes),
            "missing_producers": [],
            "conductor_pre_gate_artifact": {"present": False, "used": False},
        },
        "validation_receipts": [
            *deepcopy(list(validation_receipts)),
            *deepcopy(list(e2e_receipts)),
            *deepcopy(list(terminal_receipts)),
        ],
        "scoped_validation_receipts": deepcopy(list(validation_receipts)),
        "e2e_receipts": deepcopy(list(e2e_receipts)),
        "terminal_validation_receipts": deepcopy(list(terminal_receipts)),
        "preconditions_checked": deepcopy(list(preconditions_checked)),
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "arc_history_measurement_ready_score": measurement_ready,
        "history_support_ready_score": support_ready,
        "per_game_results": deepcopy(list(panel.get("per_game_results") or [])),
        "per_episode_reductions": deepcopy(list(panel.get("per_episode_reductions") or [])),
        "per_game_comparisons": deepcopy(list(panel.get("per_game_comparisons") or [])),
        "support_versus_conflict_curve": deepcopy(
            list(panel.get("support_versus_conflict_curve") or [])
        ),
        "cross_game_comparison": deepcopy(dict(panel.get("cross_game_comparison") or {})),
        "raw_episode_receipts": deepcopy(list(raw_episode_receipts)),
        "solve_provenance": "live_agent_self_discovery",
        "trajectory_supervisor": supervisors,
        "learning_lifecycle": deepcopy(dict(lifecycle)),
        "new_solve_claimed": False,
        "registered_levels_are_duplicate_history": True,
        "official_leaderboard_score_claimed": False,
        "beneficial_policy_change_claimed": False,
        "production_defaults_changed": False,
        "acceptance_threshold_changed": False,
        "qwen_thinking_behavior_changed": False,
        "supervisor_arms_changed": False,
        "learned_probabilities_reported": False,
        "exact_raw_frame_truth_reported": True,
        "zero_observed_conflicts_proves_markov_state": False,
        "external_publication_authorized": False,
        "generator_weight_change_authorized": False,
        "default_promotion_authorized": False,
        "prior_verdict_disposition": {
            "prior_experiment": 7589,
            "prior_verdict": "complete_null_output_boundary_and_history_observer_ready_no_benefit_claim",
            "literal_prior_verdict_repeated": False,
            "retire_if_same_verdict": False,
            "scientific_hypothesis_retired": False,
        },
        "repository_root": str(Path(repo_root).resolve()),
        "methodology_note": (
            "Conflict rates use exact raw next-frame hashes. Singleton keys are unknown. "
            "No observed conflict is not proof of determinism or a Markov state."
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, no-model accounting, rows, scores, and raw custody."""

    errors: list[str] = []
    if artifact.get("experiment_id") != EXPERIMENT_ID or artifact.get("schema") != SCHEMA:
        errors.append("identity")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("MODEL_SPECS")
    if artifact.get("live_llm_invoked") is not False:
        errors.append("live_llm_invoked")
    counts = artifact.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(value != 0 for value in counts.values()):
        errors.append("invocation_counts")
    if not REQUIRED_FIELD_PRINCIPLES <= set(artifact.get("field_principles") or {}):
        errors.append("field_principles")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    for unchanged in (
        "new_solve_claimed",
        "official_leaderboard_score_claimed",
        "production_defaults_changed",
        "acceptance_threshold_changed",
        "qwen_thinking_behavior_changed",
        "supervisor_arms_changed",
    ):
        if artifact.get(unchanged) is not False:
            errors.append(unchanged)

    if artifact.get("verdict_class") == "blocked":
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if set(artifact.get("gate_check_summary") or {}) != required:
            errors.append("gate_check_summary")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("inference_substrate_class")
        if artifact.get("arc_history_measurement_ready_score") != 0:
            errors.append("arc_history_measurement_ready_score")
        if artifact.get("history_support_ready_score") != 0:
            errors.append("history_support_ready_score")
        return list(dict.fromkeys(errors))

    if artifact.get("inference_substrate") != (
        "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    ):
        errors.append("inference_substrate")
    if artifact.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class")
    if artifact.get("planned_inference_substrate_class") != "no_model_load":
        errors.append("planned_inference_substrate_class")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 48:
        errors.append("rows")
    else:
        try:
            reduced = independent_reduce_rows(rows)
        except (TypeError, ValueError):
            errors.append("independent_reduction")
        else:
            if reduced != artifact.get("independent_reduction"):
                errors.append("independent_reduction")
    per_game = artifact.get("per_game_results")
    if per_game != rows:
        errors.append("per_game_results")
    raw_receipts = artifact.get("raw_episode_receipts")
    raw_valid = isinstance(raw_receipts, list) and len(raw_receipts) == 12
    if raw_valid:
        for receipt in raw_receipts:
            path = Path(str(receipt.get("path") or ""))
            if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
                raw_valid = False
                break
    if artifact.get("arc_history_measurement_ready_score") != int(raw_valid):
        errors.append("arc_history_measurement_ready_score")
    cross_game = artifact.get("cross_game_comparison")
    support_ready = bool(isinstance(cross_game, Mapping) and cross_game.get("ready") is True)
    if artifact.get("history_support_ready_score") != int(raw_valid and support_ready):
        errors.append("history_support_ready_score")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, Mapping) or set(gates) != {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }:
        errors.append("acceptance_gate_results")
    else:
        if any(not isinstance(row, Mapping) or not row.get("principle") for row in gates.values()):
            errors.append("gate_principles")
        if gates["readiness"].get("passed") is not raw_valid:
            errors.append("readiness_gate")
        if gates["benefit"].get("passed") is not False:
            errors.append("benefit_gate")
    budget = artifact.get("sample_size_budget")
    if not isinstance(budget, Mapping) or budget.get("observed_independent_units") != 6:
        errors.append("sample_size_budget")
    return list(dict.fromkeys(errors))


def _synthetic_raw_episode(game: str, seed: int) -> Json:
    """Create a small valid episode for unit tests, never for scientific evidence."""

    def public(value: int) -> Json:
        return {
            "frame": [[[value, value + 1], [value + 2, value + 3]]],
            "level": 0,
            "legal_actions": ["ACTION1", "RESET"],
            "termination": "NOT_FINISHED",
        }

    steps: list[Json] = [
        {
            "action_index": 0,
            "observation": None,
            "action": {"kind": "RESET", "coordinates": None},
            "outcome": public(0),
            "reset_boundary": True,
            "level_boundary": False,
            "terminated": False,
        }
    ]
    for index in range(1, 6):
        steps.append(
            {
                "action_index": index,
                "observation": public(index),
                "action": {"kind": 1, "coordinates": None},
                "outcome": public(index + 1),
                "reset_boundary": False,
                "level_boundary": False,
                "terminated": False,
            }
        )
    return {
        "schema": RAW_EPISODE_SCHEMA,
        "episode_id": f"{game}:{seed}",
        "game": game,
        "seed": seed,
        "policy": "E3AgentPolicy",
        "action_limit": ACTION_LIMIT,
        "induction_disabled": True,
        "adapter_withheld": True,
        "stored_solutions_withheld": True,
        "game_source_read": False,
        "hidden_state_read": False,
        "offline_ground_truth_bfs": False,
        "live_llm_invoked": False,
        "steps": steps,
        "trajectory_supervisor": {
            "enabled": True,
            "mode": "shadow",
            "arm_outcomes": {},
            "redirects": [],
        },
        "qualified_observer": {"enabled": True, "error_count": 0},
        "termination": {"reason": "fixture_end", "action_opportunities": len(steps)},
    }


def _mock_receipt(name: str, root: Path) -> Json:
    return {
        "name": name,
        "command": f"mock {name}",
        "command_argv": ["mock", name],
        "scope": "test_fixture",
        "worktree": str(root.resolve()),
        "exit_code": 0,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": canonical_hash(name),
        "passed": True,
        "timed_out": False,
        "output_tail": "passed",
    }


def build_test_artifact(tmp_path: Path) -> Json:
    """Build a compact valid terminal artifact without a game or subprocess."""

    root = Path(__file__).resolve().parents[2]
    receipts: list[Json] = []
    for game in GAMES:
        for seed in SEEDS:
            raw = _synthetic_raw_episode(game, seed)
            path = tmp_path / "episodes" / f"{game}-{seed}.json"
            atomic_json(path, raw)
            receipts.append(
                {
                    "episode_id": raw["episode_id"],
                    "game": game,
                    "seed": seed,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "process_exit_code": 0,
                    "copied_after_process_exit": True,
                }
            )
    panel = reduce_raw_episode_receipts(receipts)
    lifecycle = exercise_learning_lifecycle(tmp_path / "learning")
    validation = [_mock_receipt(name, root) for name in REQUIRED_SCOPED_CHECKS]
    e2e = [_mock_receipt(name, root) for name in E2E_NAMES]
    terminal = [_mock_receipt(name, root) for name in TERMINAL_NAMES]
    return build_artifact(
        repo_root=root,
        run_date=RUN_DATE,
        duration_s=1.0,
        panel=panel,
        raw_episode_receipts=receipts,
        lifecycle=lifecycle,
        source_hashes={MODULE_REL.as_posix(): sha256_file(root / MODULE_REL)},
        validation_receipts=validation,
        e2e_receipts=e2e,
        terminal_receipts=terminal,
    )


def build_validation_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Freeze serial pytest, coverage, Ruff, mypy, and spec checks."""

    coverage_file = private_root / "coverage" / ".coverage"
    commands = build_scoped_commands(
        root,
        [TEST_REL.as_posix()],
        [MODULE_REL.as_posix()],
        static_paths=[WRAPPER_REL.as_posix()],
        basetemp=private_root / "pytest",
        coverage_file=coverage_file,
    )
    transformed: list[CommandSpec] = []
    for command in commands:
        argv = tuple(item for item in command.argv if not item.startswith("--data-file="))
        if command.name in {"changed_module_coverage", "changed_module_coverage_report"}:
            argv = ("/usr/bin/env", f"COVERAGE_FILE={coverage_file}", *argv)
        transformed.append(CommandSpec(command.name, argv, command.scope, command.timeout_s))
    return transformed


def build_e2e_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Declare unchanged E2E-009..013 and one private LLM-off environment smoke."""

    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    targets = {
        "e2e_009": ("tests/python/test_arc_induction_state_persistence.py",),
        "e2e_010": ("tests/python/test_arc_tool_grammar_transport.py",),
        "e2e_011": ("tests/python/test_arc_decision_telemetry.py",),
        "e2e_012": (
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7492_e6_timed_cost_profile.py",
            "tests/python/test_arc_decision_telemetry.py",
        ),
        "e2e_013": (
            "tests/python/test_arc_decision_telemetry.py",
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
            "tests/python/test_semif_arc_readout_eval.py",
        ),
    }
    commands = [
        CommandSpec(
            name,
            (pytest, *common, f"--basetemp={private_root / name}", *paths, "-q"),
            name.replace("_", "-").upper(),
            900.0,
        )
        for name, paths in targets.items()
    ]
    foreign = private_root / "foreign-cwd"
    commands.append(
        CommandSpec(
            "llm_off_environment_smoke",
            (
                "/usr/bin/env",
                "-C",
                str(foreign),
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                python,
                "-u",
                str(root / "scripts/arc_loop_solve.py"),
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(foreign / "r11l-smoke.json"),
            ),
            "private LLM-off real E3 environment smoke",
            300.0,
        )
    )
    return commands


def build_terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Declare read-only terminal readers for the exact private candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_REL)
    common = ("--root", str(root), "--date", RUN_DATE)
    return [
        CommandSpec(
            "declared_entrypoint",
            (python, "-u", wrapper, *common, "--validate", str(candidate)),
            "declared read-only entrypoint",
            300.0,
        ),
        CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, *common, "--cold-replay", str(candidate)),
            "fresh-process raw-log replay",
            300.0,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, *common, "--independent-reduce", str(candidate)),
            "independent row and raw-log reduction",
            300.0,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact terminal candidate",
            300.0,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact terminal candidate",
            300.0,
        ),
    ]


def cold_replay(path: Path) -> list[str]:
    """Reload exact raw logs and reproduce every stored history reduction."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_artifact(artifact)
    receipts = artifact.get("raw_episode_receipts")
    if not isinstance(receipts, list):
        return list(dict.fromkeys([*errors, "raw_episode_receipts"]))
    panel = reduce_raw_episode_receipts(receipts)
    comparisons = {
        "rows": artifact.get("rows"),
        "per_episode_reductions": artifact.get("per_episode_reductions"),
        "per_game_comparisons": artifact.get("per_game_comparisons"),
        "support_versus_conflict_curve": artifact.get("support_versus_conflict_curve"),
        "cross_game_comparison": artifact.get("cross_game_comparison"),
    }
    for field, stored in comparisons.items():
        if panel[field] != stored:
            errors.append(f"cold_{field}_mismatch")
    return list(dict.fromkeys(errors))


def independent_replay(path: Path) -> list[str]:
    """Recompute row signs, counts, missingness, censoring, and raw support."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        return ["rows"]
    errors: list[str] = []
    try:
        reduced = independent_reduce_rows(rows)
    except (TypeError, ValueError) as exc:
        return [f"independent_reduction:{exc}"]
    if reduced != artifact.get("independent_reduction"):
        errors.append("independent_reduction")
    receipts = artifact.get("raw_episode_receipts")
    if not isinstance(receipts, list):
        errors.append("raw_episode_receipts")
    else:
        panel = reduce_raw_episode_receipts(receipts)
        if panel["rows"] != rows:
            errors.append("raw_rows")
    return errors


def _jsonable(value: Any) -> Any:  # pragma: no cover - live episode serialization
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@contextlib.contextmanager
def _temporary_env(changes: Mapping[str, str | None]):  # pragma: no cover
    previous = {name: os.environ.get(name) for name in changes}
    try:
        for name, value in changes.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _DisabledProposer:  # pragma: no cover - real episode guard
    """Make an accidental induction call visible without loading a model."""

    def __init__(self) -> None:
        self.calls = 0

    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        self.calls += 1
        raise RuntimeError("exp7597_induction_must_remain_disabled")


def _public_observation(frame: Any) -> Json | None:  # pragma: no cover
    if frame is None:
        return None
    from carnot.agentic.arc_competition_agent import _level_of

    visible = _jsonable(getattr(frame, "frame", []))
    actions = getattr(frame, "available_actions", []) or []
    state = getattr(frame, "state", None)
    return {
        "frame": visible,
        "level": int(_level_of(frame)),
        "legal_actions": [str(getattr(action, "name", action)) for action in actions],
        "termination": str(getattr(state, "name", state)),
    }


def _selected_action(kind: Any, data: Any) -> Json:  # pragma: no cover
    coordinates = None
    if isinstance(data, Mapping) and ("x" in data or "y" in data):
        coordinates = {key: _jsonable(data.get(key)) for key in ("x", "y") if key in data}
    return {"kind": _jsonable(kind), "coordinates": coordinates, "data": _jsonable(data)}


def run_live_episode(
    root: Path,
    game: str,
    seed: int,
    output_path: Path,
    *,
    action_limit: int = ACTION_LIMIT,
) -> Json:  # pragma: no cover - isolated real-environment worker
    """Run one real adapter-withheld E3 episode and write one private raw log."""

    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    if game not in GAMES or seed not in SEEDS or action_limit != ACTION_LIMIT:
        raise ValueError("episode_not_in_frozen_plan")
    resolved_output = Path(output_path).resolve()
    if resolved_output.is_relative_to((Path(root).resolve() / "results").resolve()):
        raise ValueError("live_episode_output_must_be_private")
    started = time.monotonic()
    random.seed(seed)
    np.random.seed(seed)
    changes = {
        "CARNOT_ARC_DISABLE_INDUCTION": "1",
        "CARNOT_ARC_OBSERVABLE_ALIAS_OBSERVER": "1",
        "CARNOT_ARC_ACTION_PROVENANCE": None,
        "CARNOT_ARC_DECISION_TELEMETRY": None,
    }
    with _temporary_env(changes):
        proposer = _DisabledProposer()
        policy = E3AgentPolicy(game, proposer=proposer)
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        frames: list[Any] = []
        latest = None
        steps: list[Json] = []
        reason = "action_limit"
        last_heartbeat = time.monotonic()
        for action_index in range(action_limit):
            if policy.is_done(frames, latest):
                reason = "policy_done"
                break
            before = _public_observation(latest)
            kind, data = policy.next_move(frames, latest)
            action = _selected_action(kind, data)
            if kind == "RESET":
                next_frame = env.reset()
            elif kind is None:
                next_frame = None
                reason = "policy_returned_none"
            else:
                next_frame = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
            after = _public_observation(next_frame)
            level_before = int(before["level"]) if before is not None else 0
            level_after = int(after["level"]) if after is not None else level_before
            steps.append(
                {
                    "action_index": action_index,
                    "observation": before,
                    "action": action,
                    "outcome": after,
                    "reset_boundary": kind == "RESET",
                    "level_boundary": before is not None and level_after != level_before,
                    "terminated": next_frame is None,
                }
            )
            latest = next_frame
            if latest is not None:
                frames.append(latest)
            now = time.monotonic()
            if now - last_heartbeat >= 60.0:
                progress(
                    started,
                    "episode_loop",
                    "heartbeat",
                    game=game,
                    seed=seed,
                    completed_actions=len(steps),
                    pending_operation="E3_policy_and_offline_environment_step",
                )
                last_heartbeat = now
            if kind is None:
                break
        observer = policy.observable_aliasing_diagnostics()
        supervisor = policy.trajectory_supervisor_diagnostics()
        if proposer.calls:
            raise RuntimeError(f"unexpected_proposer_calls:{proposer.calls}")
        raw: Json = {
            "schema": RAW_EPISODE_SCHEMA,
            "episode_id": f"{game}:{seed}",
            "game": game,
            "seed": seed,
            "policy": "E3AgentPolicy",
            "action_limit": action_limit,
            "induction_disabled": True,
            "adapter_withheld": True,
            "stored_solutions_withheld": True,
            "game_source_read": False,
            "hidden_state_read": False,
            "offline_ground_truth_bfs": False,
            "live_llm_invoked": False,
            "steps": steps,
            "qualified_observer": observer,
            "trajectory_supervisor": supervisor,
            "termination": {
                "reason": reason,
                "action_opportunities": len(steps),
                "public_state": None
                if latest is None
                else _public_observation(latest)["termination"],
            },
            "duration_s": time.monotonic() - started,
            "model_call_counts": _zero_invocations(),
        }
        raw_errors = validate_raw_episode(raw)
        if raw_errors:
            raise ValueError("live_raw_episode_invalid:" + ",".join(raw_errors))
        atomic_json(resolved_output, raw)
        return raw


def build_episode_commands(root: Path, private_root: Path) -> list[CommandSpec]:  # pragma: no cover
    """Build one bounded fresh process per frozen real E3 episode."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_REL)
    commands: list[CommandSpec] = []
    for unit in frozen_episode_plan():
        output = private_root / "episodes" / f"{unit['game']}-{unit['seed']}.json"
        commands.append(
            CommandSpec(
                f"episode_{unit['game']}_{unit['seed']}",
                (
                    python,
                    "-u",
                    wrapper,
                    "--root",
                    str(root),
                    "--date",
                    RUN_DATE,
                    "--run-episode",
                    "--game",
                    str(unit["game"]),
                    "--seed",
                    str(unit["seed"]),
                    "--raw-output",
                    str(output),
                ),
                "one real adapter-withheld E3 episode",
                900.0,
            )
        )
    return commands


def prepare_command_parent(command: CommandSpec) -> None:  # pragma: no cover
    """Create private output parents before a bounded child starts."""

    for index, argument in enumerate(command.argv):
        if argument.startswith("--basetemp=") or argument.startswith("COVERAGE_FILE="):
            Path(argument.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        elif argument == "-C" and index + 1 < len(command.argv):
            Path(command.argv[index + 1]).mkdir(parents=True, exist_ok=True)
        elif argument in {"--output", "--raw-output", "--reduction-output"}:
            if index + 1 < len(command.argv):
                Path(command.argv[index + 1]).parent.mkdir(parents=True, exist_ok=True)


def run_prepared_commands(
    root: Path,
    commands: Sequence[CommandSpec],
    *,
    log_dir: Path,
) -> list[Json]:  # pragma: no cover
    """Stream one child at a time with truthful 60-second heartbeats."""

    receipts: list[Json] = []
    for index, command in enumerate(commands, start=1):
        prepare_command_parent(command)
        print(
            f"[exp7597-subprocess] event=before name={command.name} "
            f"completed_units={index - 1} total_units={len(commands)}",
            flush=True,
        )
        rows = run_commands(
            root,
            [command],
            log_dir=log_dir / command.name,
            heartbeat_s=60.0,
        )
        for row in rows:
            row["worktree"] = str(root.resolve())
        receipts.extend(rows)
        print(
            f"[exp7597-subprocess] event=after name={command.name} "
            f"completed_units={index} total_units={len(commands)} "
            f"passed={all(row.get('passed') is True for row in rows)}",
            flush=True,
        )
    return receipts


def _copy_closed_file(source: Path, destination: Path) -> Json:  # pragma: no cover
    """Copy one closed private file and verify exact byte identity."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    private_hash = sha256_file(source)
    durable_hash = sha256_file(destination)
    if private_hash != durable_hash:
        raise ValueError(f"post_exit_copy_hash_mismatch:{source}")
    return {
        "private_path": str(source),
        "private_sha256": private_hash,
        "path": str(destination),
        "sha256": durable_hash,
        "copied_after_process_exit": True,
    }


def copy_episode_evidence(
    root: Path,
    private_root: Path,
    episode_processes: Sequence[Mapping[str, Any]],
) -> list[Json]:  # pragma: no cover
    """Copy raw episodes only after every owning episode process has exited."""

    by_name = {str(row.get("name")): row for row in episode_processes}
    receipts: list[Json] = []
    for unit in frozen_episode_plan():
        name = f"episode_{unit['game']}_{unit['seed']}"
        process = by_name.get(name, {})
        source = private_root / "episodes" / f"{unit['game']}-{unit['seed']}.json"
        if process.get("passed") is not True or not source.is_file():
            continue
        destination = root / RAW_REL / "episodes" / source.name
        copied = _copy_closed_file(source, destination)
        receipts.append(
            {
                "episode_id": unit["episode_id"],
                "game": unit["game"],
                "seed": unit["seed"],
                **copied,
                "process_exit_code": process.get("exit_code"),
                "process_log_sha256": process.get("log_sha256"),
            }
        )
    return receipts


def copy_validation_logs(
    root: Path,
    receipts: Sequence[Mapping[str, Any]],
    *,
    group: str,
) -> list[Json]:  # pragma: no cover
    """Copy closed subprocess logs into durable raw evidence."""

    copied: list[Json] = []
    for index, original in enumerate(receipts):
        row = deepcopy(dict(original))
        source = Path(str(row["log_path"]))
        if not source.is_absolute():
            source = root / source
        destination = root / RAW_REL / "validation" / group / f"{index:02d}_{row['name']}.log"
        receipt = _copy_closed_file(source, destination)
        row["private_log_path"] = receipt["private_path"]
        row["private_log_sha256"] = receipt["private_sha256"]
        row["log_path"] = str(destination.relative_to(root))
        row["log_sha256"] = receipt["sha256"]
        row["copied_after_process_exit"] = True
        copied.append(row)
    return copied


def reduce_raw_manifest(manifest_path: Path, output_path: Path) -> Json:  # pragma: no cover
    """Fresh-process entrypoint for the twelve exact raw episode receipts."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    receipts = manifest.get("raw_episode_receipts")
    if not isinstance(receipts, list):
        raise ValueError("raw_episode_receipts_required")
    panel = reduce_raw_episode_receipts(receipts)
    atomic_json(Path(output_path), panel)
    return panel


def run_fresh_raw_reduction(
    root: Path,
    private_root: Path,
    raw_receipts: Sequence[Mapping[str, Any]],
) -> tuple[Json, Json]:  # pragma: no cover
    """Reduce all raw logs in a new interpreter and bind its command receipt."""

    manifest_path = private_root / "reduction" / "raw-manifest.json"
    reduction_path = private_root / "reduction" / "panel-reduction.json"
    atomic_json(manifest_path, {"raw_episode_receipts": list(raw_receipts)})
    command = CommandSpec(
        "fresh_process_raw_reduction",
        (
            str(root / ".venv/bin/python"),
            "-u",
            str(root / WRAPPER_REL),
            "--root",
            str(root),
            "--date",
            RUN_DATE,
            "--reduce-raw-manifest",
            str(manifest_path),
            "--reduction-output",
            str(reduction_path),
        ),
        "fresh process independently reduces all twelve raw logs",
        300.0,
    )
    rows = run_prepared_commands(root, [command], log_dir=private_root / "logs" / "raw-reduction")
    receipt = rows[0]
    if receipt.get("passed") is not True or not reduction_path.is_file():
        raise RuntimeError("fresh_process_raw_reduction_failed")
    panel = json.loads(reduction_path.read_text(encoding="utf-8"))
    if panel != reduce_raw_episode_receipts(raw_receipts):
        raise RuntimeError("fresh_process_raw_reduction_mismatch")
    receipt["manifest_sha256"] = sha256_file(manifest_path)
    receipt["reduction_sha256"] = sha256_file(reduction_path)
    return panel, receipt


def _source_hashes(root: Path, prerequisite_hashes: Mapping[str, str]) -> dict[str, str]:
    hashes = dict(prerequisite_hashes)
    for relative in (MODULE_REL, TEST_REL, WRAPPER_REL, AGENT_REL, SPEC_REL, REGISTRY_REL):
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = sha256_file(path)
    return hashes


def _phase_span(phase: str, phase_started: float, run_started: float) -> Json:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "started_s": phase_started - run_started,
        "ended_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def run_experiment(
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_REL,
) -> Json:  # pragma: no cover - declared integration entrypoint
    """Run twelve E3 episodes, fresh reduction, validation, and atomic publication."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    expected_root = Path(__file__).resolve().parents[2]
    if root != expected_root:
        raise ValueError(f"root_mismatch:{root}:{expected_root}")
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    target = output_path if output_path.is_absolute() else root / output_path
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7597-v663-", dir="/tmp")).resolve()
    spans: list[Json] = []
    progress(started, "startup", "begin", root=root, private_root=private_root)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    preconditions, prerequisite_hashes = collect_preconditions(root, private_root)
    failed = [row for row in preconditions if row.get("passed") is not True]
    spans.append(_phase_span("preconditions", phase_started, started))
    progress(started, "preconditions", "after", checks=len(preconditions), failed=len(failed))
    if failed:
        first = failed[0]
        blocked = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.monotonic() - started,
            check=str(first["check"]),
            upstream=str(first["upstream"]),
            path=str(first["path"]),
            field=str(first["field"]),
            op=str(first["op"]),
            expected=first["expected"],
            observed=first["observed"],
        )
        blocked["preconditions_checked"] = deepcopy(preconditions)
        blocked["phase_spans"] = spans
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        progress(started, "publication", "before_atomic_blocked", path=target)
        atomic_json(target, blocked)
        progress(started, "publication", "after_atomic_blocked", path=target)
        return blocked

    phase_started = time.monotonic()
    episode_commands = build_episode_commands(root, private_root)
    progress(started, "live_episodes", "before_subprocesses", units=len(episode_commands))
    episode_private = run_prepared_commands(
        root,
        episode_commands,
        log_dir=private_root / "logs" / "episodes",
    )
    spans.append(_phase_span("live_episodes", phase_started, started))
    progress(
        started,
        "live_episodes",
        "after_subprocesses",
        passed=sum(int(row.get("passed") is True) for row in episode_private),
        units=len(episode_private),
    )

    phase_started = time.monotonic()
    progress(started, "evidence_copy", "before", units=len(episode_private))
    raw_episode_receipts = copy_episode_evidence(root, private_root, episode_private)
    episode_logs = copy_validation_logs(root, episode_private, group="episodes")
    spans.append(_phase_span("evidence_copy", phase_started, started))
    progress(
        started,
        "evidence_copy",
        "after",
        raw_episodes=len(raw_episode_receipts),
        logs=len(episode_logs),
    )

    phase_started = time.monotonic()
    progress(started, "fresh_raw_reduction", "before_subprocess")
    panel, fresh_reduction_private = run_fresh_raw_reduction(
        root, private_root, raw_episode_receipts
    )
    fresh_reduction_logs = copy_validation_logs(
        root, [fresh_reduction_private], group="raw-reduction"
    )
    reduction_copy = _copy_closed_file(
        private_root / "reduction" / "panel-reduction.json",
        root / RAW_REL / "panel-reduction.json",
    )
    spans.append(_phase_span("fresh_raw_reduction", phase_started, started))
    progress(
        started,
        "fresh_raw_reduction",
        "after_subprocess",
        episodes=panel["episode_count"],
        valid=panel["valid"],
        support_ready=panel["history_support_ready_score"],
    )

    phase_started = time.monotonic()
    progress(started, "learning_lifecycle", "before")
    lifecycle = exercise_learning_lifecycle(private_root / "learning")
    spans.append(_phase_span("learning_lifecycle", phase_started, started))
    progress(started, "learning_lifecycle", "after", passed=lifecycle["passed"])

    phase_started = time.monotonic()
    validation_commands = build_validation_commands(root, private_root / "validation")
    progress(started, "scoped_validation", "before_subprocesses", units=len(validation_commands))
    validation_private = run_prepared_commands(
        root,
        validation_commands,
        log_dir=private_root / "logs" / "validation",
    )
    validation = copy_validation_logs(root, validation_private, group="scoped")
    spans.append(_phase_span("scoped_validation", phase_started, started))
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        passed=_receipt_names_pass(validation, REQUIRED_SCOPED_CHECKS),
    )

    phase_started = time.monotonic()
    e2e_commands = build_e2e_commands(root, private_root / "e2e")
    progress(started, "arc_e2e", "before_subprocesses", units=len(e2e_commands))
    e2e_private = run_prepared_commands(
        root,
        e2e_commands,
        log_dir=private_root / "logs" / "e2e",
    )
    e2e = copy_validation_logs(root, e2e_private, group="e2e")
    spans.append(_phase_span("arc_e2e", phase_started, started))
    progress(
        started,
        "arc_e2e",
        "after_subprocesses",
        passed=_receipt_names_pass(e2e, E2E_NAMES),
    )

    source_hashes = _source_hashes(root, prerequisite_hashes)
    for receipt in raw_episode_receipts:
        path = Path(str(receipt["path"]))
        source_hashes[path.relative_to(root).as_posix()] = str(receipt["sha256"])
    source_hashes[RAW_REL.joinpath("panel-reduction.json").as_posix()] = str(
        reduction_copy["sha256"]
    )
    base_validation = [*episode_logs, *fresh_reduction_logs, *validation]
    candidate = build_artifact(
        repo_root=root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        panel=panel,
        raw_episode_receipts=raw_episode_receipts,
        lifecycle=lifecycle,
        source_hashes=source_hashes,
        validation_receipts=base_validation,
        e2e_receipts=e2e,
        terminal_receipts=[],
        preconditions_checked=preconditions,
        phase_spans=spans,
    )
    candidate_errors = validate_artifact(candidate)
    if candidate_errors:
        raise RuntimeError("terminal_candidate_invalid:" + ",".join(candidate_errors))
    candidate_path = private_root / "terminal" / "candidate.json"
    atomic_json(candidate_path, candidate)
    candidate_sha256 = sha256_file(candidate_path)

    phase_started = time.monotonic()
    terminal_commands = build_terminal_commands(root, candidate_path)
    progress(
        started,
        "terminal_readers",
        "before_subprocesses",
        units=len(terminal_commands),
        candidate_sha256=candidate_sha256,
    )
    terminal_private = run_prepared_commands(
        root,
        terminal_commands,
        log_dir=private_root / "logs" / "terminal",
    )
    terminal = copy_validation_logs(root, terminal_private, group="terminal")
    spans.append(_phase_span("terminal_readers", phase_started, started))
    terminal_passed = _receipt_names_pass(terminal, TERMINAL_NAMES)
    adversarial = next((row for row in terminal if row.get("name") == "adversarial_verify"), {})
    flagged = adversarial.get("passed") is not True or "CRITICAL" in str(
        adversarial.get("output_tail") or ""
    )
    progress(
        started,
        "terminal_readers",
        "after_subprocesses",
        passed=terminal_passed,
        flagged_adversarial=flagged,
    )
    candidate_copy = _copy_closed_file(
        candidate_path,
        root / RAW_REL / "terminal-candidate.json",
    )
    source_hashes[RAW_REL.joinpath("terminal-candidate.json").as_posix()] = str(
        candidate_copy["sha256"]
    )

    final = build_artifact(
        repo_root=root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        panel=panel,
        raw_episode_receipts=raw_episode_receipts,
        lifecycle=lifecycle,
        source_hashes=source_hashes,
        validation_receipts=base_validation,
        e2e_receipts=e2e,
        terminal_receipts=terminal,
        preconditions_checked=preconditions,
        phase_spans=spans,
        flagged_adversarial=flagged,
    )
    final["affected_file_validation_manifest"] = {
        "schema": "carnot.exp7597.affected_validation_manifest.v1",
        "test_paths": [TEST_REL.as_posix()],
        "changed_modules": [MODULE_REL.as_posix()],
        "static_paths": [WRAPPER_REL.as_posix()],
        "spec_path": SPEC_REL.as_posix(),
    }
    final["terminal_reader_outcomes"] = {
        row["name"]: {
            "exit_code": row["exit_code"],
            "passed": row["passed"],
            "log_sha256": row["log_sha256"],
        }
        for row in terminal
    }
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    final_errors = validate_artifact(final)
    if final_errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(final_errors))
    progress(started, "publication", "before_atomic", path=target)
    atomic_json(root / RAW_REL / "terminal-validation-receipts.json", {"receipts": terminal})
    atomic_json(target, final)
    progress(
        started,
        "publication",
        "after_atomic",
        path=target,
        verdict=final["honest_verdict"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_REL)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate", type=Path)
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    modes.add_argument("--run-episode", action="store_true")
    modes.add_argument("--reduce-raw-manifest", type=Path)
    parser.add_argument("--game", choices=GAMES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--reduction-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.validate is not None:
        errors = validate_artifact(json.loads(args.validate.read_text(encoding="utf-8")))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(args.independent_reduce)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.run_episode:
        if args.game is None or args.seed is None or args.raw_output is None:
            raise SystemExit("--run-episode requires --game, --seed, and --raw-output")
        started = time.monotonic()
        progress(started, "episode", "before", game=args.game, seed=args.seed)
        raw = run_live_episode(args.root, args.game, args.seed, args.raw_output)
        progress(
            started,
            "episode",
            "after",
            game=args.game,
            seed=args.seed,
            actions=len(raw["steps"]),
        )
        return 0
    if args.reduce_raw_manifest is not None:
        if args.reduction_output is None:
            raise SystemExit("--reduce-raw-manifest requires --reduction-output")
        reduce_raw_manifest(args.reduce_raw_manifest, args.reduction_output)
        return 0
    run_experiment(args.root, args.date, output_path=args.output)
    return 0
