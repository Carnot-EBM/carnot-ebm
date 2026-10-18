"""Measure hidden-history effects with matched replay interventions.

REQ-ARC-WMTE-7611. Two naturally discovered action prefixes qualify only when
their public target frame and coordinate-aware action are identical. Fresh
replays then separate within-prefix instability from a repeatable history effect.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
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
EXPERIMENT_ID = 7611
MILESTONE = "2026.09.664"
RUN_DATE = "20260924"
REQUIREMENT_ID = "REQ-ARC-WMTE-7611"
SCHEMA = "carnot.experiment_7611_v664_arc_matched_support.v1"
RAW_EPISODE_SCHEMA = "carnot.exp7611.raw_public_episode.v1"
GAMES = ("su15", "sp80", "ft09", "sb26", "g50t", "dc22")
SEEDS = (7_612_001, 7_612_002)
HISTORY_LENGTHS = (0, 1, 2, 4)
ACTION_LIMIT = 600
PREFIX_ACTION_CAP = 128
MAX_MATCHED_KEYS_PER_GAME = 20
MODEL_SPECS: list[Json] = []

RESULT_REL = Path("results/experiment_7611_v664_arc_matched_support.json")
RAW_REL = Path("results/raw/experiment_7611_v664_arc_matched_support")
MODULE_REL = Path("python/carnot/experiment_7611_v664_arc_matched_support.py")
TEST_REL = Path("tests/python/test_experiment_7611_v664_arc_matched_support.py")
WRAPPER_REL = Path("scripts/experiments/experiment_7611_v664_arc_matched_support.py")
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
    "matched_support_ready_score",
    "matched_protocol_path",
    "solve_provenance",
    "supervisor_scope",
}

FIELD_PRINCIPLES = {
    "honest_verdict": "Use a complete_ terminal prefix; completion alone is not scientific benefit.",
    "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Persist the terminal reader result; flagged evidence never opens readiness.",
    "gate_check_summary": "Every block names check, upstream, path, field, operator, expected, and observed.",
    "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness have separate results.",
    "rows": "Each independent game and arm keeps absolute operands, seed, direction, censoring, and provenance.",
    "sample_size_budget": "Repeated seeds and replays never multiply independent games or matched keys.",
    "inference_substrate": "The value describes this current no-LLM offline-arcade execution.",
    "inference_substrate_class": "Planned and actual classes stay separate; blocked_no_run means no model work.",
    "MODEL_SPECS": "A no-model measurement uses an empty model list.",
    "invocation_counts": "Current loads, forwards, generations, and tokens are counted separately.",
    "duration_s": "Current monotonic time is measured and never padded or inherited.",
    "random_seed": "Every stochastic collector stage has an explicit seed.",
    "reproducibility_checksum": "The digest binds configuration, immutable raw evidence, and reduction.",
    "source_artifact_hashes": "Producer artifacts, pre-gate records, and missing producers remain distinct.",
    "validation_receipts": "Commands, exits, worktree, log hashes, and terminal readers bind validation.",
    "verifier_is_oracle": "Exact fixtures calibrate the protocol and cannot prove learned semantic correctness.",
    "field_principles": "One-line reasons travel with each governed field.",
    "matched_support_ready_score": "One means matched-key, fresh-reset, and provenance fixtures conform only.",
    "matched_protocol_path": "The frozen roster, prefix cap, target key, and game unit define the protocol.",
    "solve_provenance": "Runtime prefixes are live agent self-discovery; no new game solve is claimed.",
    "supervisor_scope": "Only current fired and helped ledger counts support refinement.",
}


def repo_root() -> Path:
    """Resolve this checkout without relying on the caller's current directory."""

    return Path(__file__).resolve().parents[2]


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Emit a flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7611] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def frame_sha256(observation: Mapping[str, Any]) -> str:
    """Hash the raw public frame while retaining array shape and dtype."""

    return full_frame_hash(observation["frame"])


def zero_invocations() -> Json:
    value = deepcopy(ZERO_INVOCATION_COUNTS)
    value.update({"forward_calls": 0, "input_tokens": 0, "output_tokens": 0})
    return value


def canonical_action(action: Mapping[str, Any]) -> Json:
    """Retain the selected action and coordinates in a stable replay shape."""

    coordinates = action.get("coordinates")
    data = action.get("data")
    if coordinates is None and isinstance(data, Mapping) and ("x" in data or "y" in data):
        coordinates = {key: data.get(key) for key in ("x", "y") if key in data}
    return {
        "kind": action.get("kind"),
        "coordinates": deepcopy(coordinates),
        "data": deepcopy(data),
    }


def _legal_action_names(observation: Mapping[str, Any]) -> set[str]:
    return {str(item).replace("ACTION", "") for item in observation.get("legal_actions") or []}


def action_is_legal(observation: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    kind = action.get("kind")
    if kind == "RESET":
        return True
    return str(kind).replace("ACTION", "") in _legal_action_names(observation)


def target_key(game: str, observation: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    """Build the frozen intervention key without consulting the future outcome."""

    return canonical_hash(
        {
            "game": game,
            "level": int(observation.get("level", 0)),
            "current_frame_sha256": frame_sha256(observation),
            "legal_action": canonical_action(action),
        }
    )


def _history_items(prefix: Sequence[Mapping[str, Any]]) -> list[Json]:
    return [_history_item(row) for row in prefix]


def _history_item(row: Mapping[str, Any]) -> Json:
    observation = row.get("observation")
    outcome = row.get("outcome")
    return {
        "observation_sha256": None
        if not isinstance(observation, Mapping)
        else frame_sha256(observation),
        "action": canonical_action(row["action"]),
        "outcome_sha256": None if not isinstance(outcome, Mapping) else frame_sha256(outcome),
        "reset_boundary": bool(row.get("reset_boundary")),
        "level_boundary": bool(row.get("level_boundary")),
    }


def _history_key(target: str, items: Sequence[Mapping[str, Any]], length: int) -> str:
    prior = [] if length == 0 else list(items[-length:])
    return canonical_hash({"history_length": length, "target_key": target, "prior": prior})


def validate_natural_episode(raw: Mapping[str, Any]) -> list[str]:
    """Reject collection rows that could contain source, adapter, or model knowledge."""

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
        "per_game_masks_used": False,
        "live_llm_invoked": False,
    }
    for field, expected_value in expected.items():
        if raw.get(field) != expected_value:
            errors.append(field)
    if raw.get("game") not in GAMES:
        errors.append("game_not_frozen")
    if raw.get("seed") not in SEEDS:
        errors.append("seed_not_frozen")
    steps = raw.get("steps")
    if not isinstance(steps, list) or len(steps) > ACTION_LIMIT:
        return [*errors, "steps_invalid"]
    for index, row in enumerate(steps):
        if not isinstance(row, Mapping) or row.get("action_index") != index:
            errors.append(f"action_index_invalid:{index}")
            continue
        action = row.get("action")
        if not isinstance(action, Mapping):
            errors.append(f"action_invalid:{index}")
            continue
        observation = row.get("observation")
        if isinstance(observation, Mapping) and not action_is_legal(observation, action):
            errors.append(f"illegal_target_action:{index}")
    return list(dict.fromkeys(errors))


def _occurrence(
    raw: Mapping[str, Any],
    target_index: int,
    history_items: Sequence[Mapping[str, Any]] | None = None,
) -> Json:
    row = raw["steps"][target_index]
    observation = row["observation"]
    action = canonical_action(row["action"])
    prefix = list(raw["steps"][:target_index])
    key = target_key(str(raw["game"]), observation, action)
    items = list(history_items) if history_items is not None else _history_items(prefix)
    return {
        "episode_id": raw["episode_id"],
        "game": raw["game"],
        "seed": int(raw["seed"]),
        "target_action_index": target_index,
        "prefix_action_count": len(prefix),
        "prefix_steps": prefix,
        "target_observation": deepcopy(observation),
        "target_action": action,
        "target_key": key,
        "full_history_sha256": canonical_hash(items),
        "history_keys": {
            str(length): _history_key(key, items, length) for length in HISTORY_LENGTHS
        },
    }


def select_matched_prefixes(
    episodes: Sequence[Mapping[str, Any]], *, max_per_game: int = MAX_MATCHED_KEYS_PER_GAME
) -> Json:
    """Select distinct histories by pre-outcome hashes only.

    The function intentionally never copies or hashes the target outcome. A
    future-frame mutation therefore cannot change the selected cohort.
    """

    grouped: dict[tuple[str, str], list[Json]] = defaultdict(list)
    eligible_occurrences = 0
    long_censored = 0
    invalid_occurrences = 0
    for raw in episodes:
        errors = validate_natural_episode(raw)
        if errors:
            raise ValueError("raw_episode_invalid:" + ",".join(errors))
        history_items: list[Json] = []
        for index, row in enumerate(raw["steps"]):
            observation = row.get("observation")
            action = row.get("action")
            if not isinstance(observation, Mapping) or not isinstance(action, Mapping):
                history_items.append(_history_item(row))
                continue
            if action.get("kind") == "RESET" or not action_is_legal(observation, action):
                invalid_occurrences += 1
                history_items.append(_history_item(row))
                continue
            eligible_occurrences += 1
            if index > PREFIX_ACTION_CAP:
                long_censored += 1
                continue
            occurrence = _occurrence(raw, index, history_items)
            grouped[(str(raw["game"]), occurrence["target_key"])].append(occurrence)
            history_items.append(_history_item(row))

    singleton_keys = 0
    candidate_pairs: list[Json] = []
    repeated_same_history_keys = 0
    for (game, key), occurrences in grouped.items():
        histories: dict[str, Json] = {}
        for occurrence in sorted(
            occurrences,
            key=lambda row: (
                str(row["full_history_sha256"]),
                int(row["seed"]),
                int(row["target_action_index"]),
            ),
        ):
            histories.setdefault(str(occurrence["full_history_sha256"]), occurrence)
        if len(occurrences) < 2:
            singleton_keys += 1
            continue
        if len(histories) < 2:
            repeated_same_history_keys += 1
            continue
        left, right = list(histories.values())[:2]
        pair = {
            "game": game,
            "target_key": key,
            "selection_hash": canonical_hash({"game": game, "target_key": key}),
            "target_action": deepcopy(left["target_action"]),
            "target_observation_sha256": frame_sha256(left["target_observation"]),
            "histories_distinct": left["full_history_sha256"] != right["full_history_sha256"],
            "history_keys": {
                str(length): [left["history_keys"][str(length)], right["history_keys"][str(length)]]
                for length in HISTORY_LENGTHS
            },
            "prefixes": [left, right],
        }
        candidate_pairs.append(pair)

    selected: list[Json] = []
    by_game: dict[str, list[Json]] = defaultdict(list)
    for pair in candidate_pairs:
        by_game[str(pair["game"])].append(pair)
    hash_cap_censored = 0
    for game in sorted(by_game):
        ordered = sorted(by_game[game], key=lambda row: row["selection_hash"])
        selected.extend(ordered[:max_per_game])
        hash_cap_censored += max(0, len(ordered) - max_per_game)
    selection_identity = [
        {
            "game": row["game"],
            "target_key": row["target_key"],
            "prefixes": [prefix["full_history_sha256"] for prefix in row["prefixes"]],
        }
        for row in selected
    ]
    return {
        "selected_pairs": selected,
        "selection_checksum": canonical_hash(selection_identity),
        "selection_uses_future_outcomes": False,
        "natural_trajectory_denominator": {
            "episode_count": len(episodes),
            "eligible_target_occurrences": eligible_occurrences,
            "distinct_pre_outcome_keys": len(grouped),
            "singleton_keys": singleton_keys,
            "repeated_same_history_keys": repeated_same_history_keys,
            "different_history_candidate_keys": len(candidate_pairs),
            "selected_matched_keys": len(selected),
            "long_prefix_censored_occurrences": long_censored,
            "hash_cap_censored_keys": hash_cap_censored,
            "invalid_target_occurrences": invalid_occurrences,
        },
    }


def _snapshot_matches(
    expected: Mapping[str, Any] | None, observed: Mapping[str, Any] | None
) -> bool:
    if expected is None or observed is None:
        return expected is observed
    return (
        frame_sha256(expected) == frame_sha256(observed)
        and int(expected.get("level", 0)) == int(observed.get("level", 0))
        and _legal_action_names(expected) == _legal_action_names(observed)
    )


def replay_prefix_once(
    occurrence: Mapping[str, Any],
    environment_factory: Callable[[], Any],
    execute_action: Callable[[Any, Mapping[str, Any]], Mapping[str, Any] | None],
) -> Json:
    """Reconstruct one prefix in a new environment and apply its target action."""

    environment = environment_factory()
    latest: Mapping[str, Any] | None = None
    reset_count = 0
    boundary_count = 0
    for index, step in enumerate(occurrence["prefix_steps"]):
        expected_before = step.get("observation")
        if not _snapshot_matches(expected_before, latest):
            return {
                "replayable": False,
                "exclusion_reason": f"observation_mismatch:{index}",
                "verified_reset_boundaries": reset_count,
                "verified_level_boundaries": boundary_count,
            }
        action = canonical_action(step["action"])
        actual_reset = action["kind"] == "RESET"
        if actual_reset != bool(step.get("reset_boundary")):
            return {
                "replayable": False,
                "exclusion_reason": f"reset_boundary_mismatch:{index}",
                "verified_reset_boundaries": reset_count,
                "verified_level_boundaries": boundary_count,
            }
        latest = execute_action(environment, action)
        expected_after = step.get("outcome")
        if not _snapshot_matches(expected_after, latest):
            return {
                "replayable": False,
                "exclusion_reason": f"outcome_mismatch:{index}",
                "verified_reset_boundaries": reset_count,
                "verified_level_boundaries": boundary_count,
            }
        level_before = int(expected_before.get("level", 0)) if expected_before else 0
        level_after = (
            int(expected_after.get("level", level_before)) if expected_after else level_before
        )
        actual_boundary = expected_before is not None and level_after != level_before
        if actual_boundary != bool(step.get("level_boundary")):
            return {
                "replayable": False,
                "exclusion_reason": f"level_boundary_mismatch:{index}",
                "verified_reset_boundaries": reset_count,
                "verified_level_boundaries": boundary_count,
            }
        reset_count += int(actual_reset)
        boundary_count += int(actual_boundary)

    target_observation = occurrence["target_observation"]
    target_action_value = canonical_action(occurrence["target_action"])
    if not _snapshot_matches(target_observation, latest):
        return {
            "replayable": False,
            "exclusion_reason": "target_observation_mismatch",
            "verified_reset_boundaries": reset_count,
            "verified_level_boundaries": boundary_count,
        }
    if not action_is_legal(target_observation, target_action_value):
        return {
            "replayable": False,
            "exclusion_reason": "target_action_not_legal",
            "verified_reset_boundaries": reset_count,
            "verified_level_boundaries": boundary_count,
        }
    if occurrence["target_key"] != target_key(
        str(occurrence.get("game") or ""), target_observation, target_action_value
    ) and occurrence.get("game"):
        return {
            "replayable": False,
            "exclusion_reason": "target_key_mismatch",
            "verified_reset_boundaries": reset_count,
            "verified_level_boundaries": boundary_count,
        }
    outcome = execute_action(environment, target_action_value)
    return {
        "replayable": outcome is not None,
        "exclusion_reason": None if outcome is not None else "target_terminated_without_frame",
        "verified_reset_boundaries": reset_count,
        "verified_level_boundaries": boundary_count,
        "target_outcome_sha256": None if outcome is None else frame_sha256(outcome),
        "target_level_after": None if outcome is None else int(outcome.get("level", 0)),
    }


def replay_matched_pair(
    pair: Mapping[str, Any],
    environment_factory: Callable[[], Any],
    execute_action: Callable[[Any, Mapping[str, Any]], Mapping[str, Any] | None],
) -> Json:
    """Replay both histories twice so stability and disagreement are separate."""

    outcomes: list[list[Json]] = []
    for prefix in pair["prefixes"]:
        occurrence = dict(prefix)
        occurrence["game"] = pair["game"]
        outcomes.append(
            [
                replay_prefix_once(occurrence, environment_factory, execute_action),
                replay_prefix_once(occurrence, environment_factory, execute_action),
            ]
        )
    replayable = all(row["replayable"] for group in outcomes for row in group)
    stable = [
        replayable
        and group[0].get("target_outcome_sha256") == group[1].get("target_outcome_sha256")
        for group in outcomes
    ]
    disagreement = bool(
        all(stable)
        and outcomes[0][0].get("target_outcome_sha256")
        != outcomes[1][0].get("target_outcome_sha256")
    )
    if not replayable:
        exclusion = "unreplayable_prefix"
    elif not all(stable):
        exclusion = "unstable_prefix"
    else:
        exclusion = None
    return {
        "game": pair["game"],
        "target_key": pair["target_key"],
        "history_keys": deepcopy(pair["history_keys"]),
        "histories_distinct": bool(pair["histories_distinct"]),
        "fresh_environment_count": 4,
        "prefix_replays": outcomes,
        "left_within_prefix_stable": stable[0],
        "right_within_prefix_stable": stable[1],
        "cross_history_disagreement": disagreement,
        "history_disambiguation_witness": bool(
            pair["histories_distinct"] and all(stable) and disagreement
        ),
        "replay_exclusion": exclusion,
    }


def reduce_measurements(
    selection: Mapping[str, Any], measurements: Sequence[Mapping[str, Any]]
) -> Json:
    """Keep natural collection and replay intervention denominators independent."""

    replayed = len(measurements)
    unreplayable = sum(row.get("replay_exclusion") == "unreplayable_prefix" for row in measurements)
    unstable = sum(row.get("replay_exclusion") == "unstable_prefix" for row in measurements)
    stable_rows = [row for row in measurements if row.get("replay_exclusion") is None]
    witnesses = sum(row.get("history_disambiguation_witness") is True for row in stable_rows)
    stable_agreements = sum(row.get("cross_history_disagreement") is False for row in stable_rows)
    return {
        "natural_trajectory_denominator": deepcopy(selection["natural_trajectory_denominator"]),
        "replay_intervention_denominator": {
            "intended_matched_keys": replayed,
            "replayed_matched_keys": replayed,
            "stable_matched_keys": len(stable_rows),
            "excluded_unreplayable_keys": unreplayable,
            "excluded_unstable_keys": unstable,
            "fresh_environment_count": sum(
                int(row.get("fresh_environment_count", 0)) for row in measurements
            ),
        },
        "primary_h0_vs_h1": {
            "witness_numerator": witnesses,
            "stable_denominator": len(stable_rows),
            "stable_agreement_count": stable_agreements,
            "direction": "history_disambiguation_witness_rate_h1_over_h0",
            "rate": witnesses / len(stable_rows) if stable_rows else None,
        },
        "descriptive_history_lengths": [2, 4],
        "measurements": deepcopy(list(measurements)),
    }


def _fixture_frame(
    value: int, *, level: int = 0, legal: Sequence[str] = ("1", "2", "3", "6")
) -> Json:
    return {
        "frame": [[value]],
        "level": level,
        "legal_actions": list(legal),
        "termination": "NOT_FINISHED",
    }


def _fixture_action(kind: int | str, x: int | None = None, y: int | None = None) -> Json:
    data = None if x is None else {"x": x, "y": y}
    return {"kind": kind, "coordinates": deepcopy(data), "data": deepcopy(data)}


def _fixture_step(index: int, before: Any, action: Json, after: Any, **flags: Any) -> Json:
    return {
        "action_index": index,
        "observation": deepcopy(before),
        "action": deepcopy(action),
        "outcome": deepcopy(after),
        "reset_boundary": bool(flags.get("reset")),
        "level_boundary": bool(flags.get("boundary")),
        "terminated": after is None,
    }


def _synthetic_episode(game: str, seed: int, branch: int) -> Json:
    start = _fixture_frame(1)
    shared = _fixture_frame(2, legal=("6",))
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
        "per_game_masks_used": False,
        "live_llm_invoked": False,
        "steps": [
            _fixture_step(0, None, _fixture_action("RESET"), start, reset=True),
            _fixture_step(1, start, _fixture_action(branch), shared),
            _fixture_step(2, shared, _fixture_action(6, 4, 5), _fixture_frame(7 + branch)),
        ],
        "trajectory_supervisor": {"would_have_arm_outcomes": {}, "actions_observed": 2},
        "termination": {"reason": "fixture", "action_opportunities": 3},
        "model_call_counts": zero_invocations(),
    }


class _HiddenFixtureEnvironment:
    def __init__(self, instability: int = 0) -> None:
        self.hidden = 0
        self.instability = instability

    def reset(self) -> Json:
        self.hidden = 0
        return _fixture_frame(1)

    def step(self, action: Mapping[str, Any]) -> Json:
        if action["kind"] in (1, 2):
            self.hidden = int(action["kind"])
            return _fixture_frame(2, legal=("6",))
        return _fixture_frame(7 + self.hidden + self.instability)


def _fixture_execute(environment: Any, action: Mapping[str, Any]) -> Json:
    return environment.reset() if action["kind"] == "RESET" else environment.step(action)


def measure_protocol_fixtures() -> Json:
    """Exercise protocol mechanics without treating oracle rows as empirical gain."""

    episodes = [
        _synthetic_episode("su15", SEEDS[0], 1),
        _synthetic_episode("su15", SEEDS[1], 2),
    ]
    selection = select_matched_prefixes(episodes)
    pair = selection["selected_pairs"][0]
    witness = replay_matched_pair(pair, _HiddenFixtureEnvironment, _fixture_execute)
    instability = iter((0, 0, 0, 1))
    unstable = replay_matched_pair(
        pair,
        lambda: _HiddenFixtureEnvironment(next(instability)),
        _fixture_execute,
    )

    class BoundaryEnvironment:
        def __init__(self) -> None:
            self.level = 0

        def reset(self) -> Json:
            self.level = 0
            return _fixture_frame(1, legal=("3",))

        def step(self, action: Mapping[str, Any]) -> Json:
            self.level += 1
            return _fixture_frame(1 + self.level, level=self.level, legal=("3",))

    start = _fixture_frame(1, legal=("3",))
    level_one = _fixture_frame(2, level=1, legal=("3",))
    boundary_occurrence = {
        "prefix_steps": [
            _fixture_step(0, None, _fixture_action("RESET"), start, reset=True),
            _fixture_step(1, start, _fixture_action(3), level_one, boundary=True),
        ],
        "target_observation": level_one,
        "target_action": _fixture_action(3),
        "target_key": target_key("sb26", level_one, _fixture_action(3)),
        "game": "sb26",
    }
    boundary = replay_prefix_once(boundary_occurrence, BoundaryEnvironment, _fixture_execute)
    singleton = select_matched_prefixes([_synthetic_episode("sp80", SEEDS[0], 1)])
    fixture_results = {
        "coordinate_action": pair["target_action"]["coordinates"] == {"x": 4, "y": 5},
        "same_action_level_boundary": bool(
            boundary["replayable"] and boundary["verified_level_boundaries"] == 1
        ),
        "singleton": singleton["natural_trajectory_denominator"]["singleton_keys"] == 2,
        "stateful_reset": all(
            replay["verified_reset_boundaries"] == 1
            for group in witness["prefix_replays"]
            for replay in group
        ),
        "two_hidden_histories": witness["history_disambiguation_witness"] is True,
        "unstable_prefix": unstable["replay_exclusion"] == "unstable_prefix",
    }
    return {
        "fixture_results": fixture_results,
        "matched_support_ready_score": int(all(fixture_results.values())),
        "verdict_class": "circular_positive",
        "verifier_is_oracle": True,
        "methodology_note": "Exact fixture outcomes calibrate the protocol and are not empirical benefit.",
    }


def _gate(result: bool, principle: str, expected: Any, observed: Any) -> Json:
    return {
        "principle": principle,
        "result": bool(result),
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
    }


def _checksum_payload(artifact: Mapping[str, Any]) -> Json:
    payload = deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    payload.pop("duration_s", None)
    payload.pop("phase_spans", None)
    payload.pop("validation_receipts", None)
    return payload


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return canonical_hash(_checksum_payload(artifact))


def _supervisor_scope(episodes: Sequence[Mapping[str, Any]]) -> Json:
    fired = helped = 0
    for episode in episodes:
        receipt = episode.get("trajectory_supervisor") or {}
        for outcome in (receipt.get("would_have_arm_outcomes") or {}).values():
            if isinstance(outcome, Mapping):
                fired += int(outcome.get("fired", 0))
                helped += int(outcome.get("helped", 0))
    return {
        "fired_count": fired,
        "helped_count": helped,
        "refinement_supported": fired > 0 and helped > 0,
        "no_firings_means_no_refinement": fired == 0,
        "arm_table_changed": False,
    }


def _rows_from_reduction(reduction: Mapping[str, Any], games: Sequence[str]) -> list[Json]:
    primary = reduction["primary_h0_vs_h1"]
    rows: list[Json] = []
    for game in games:
        measurements = [row for row in reduction["measurements"] if row["game"] == game]
        stable = [row for row in measurements if row.get("replay_exclusion") is None]
        witnesses = sum(row.get("history_disambiguation_witness") is True for row in stable)
        for length in HISTORY_LENGTHS:
            row = {
                "unit": game,
                "arm": f"h{length}",
                "history_length": length,
                "absolute_metric": "stable_history_disagreement_count",
                "numerator": witnesses if length else 0,
                "denominator": len(stable),
                "rate": (witnesses / len(stable) if stable else None)
                if length
                else 0.0
                if stable
                else None,
                "seed": list(SEEDS),
                "direction": primary["direction"]
                if length == 1
                else "descriptive"
                if length
                else "reference",
                "censoring": {
                    "unreplayable": sum(
                        row.get("replay_exclusion") == "unreplayable_prefix" for row in measurements
                    ),
                    "unstable": sum(
                        row.get("replay_exclusion") == "unstable_prefix" for row in measurements
                    ),
                },
                "provenance": "fresh_prefix_replay" if measurements else "fixture_only",
            }
            row["operand_checksum"] = canonical_hash(
                {
                    key: row[key]
                    for key in ("unit", "arm", "numerator", "denominator", "seed", "censoring")
                }
            )
            rows.append(row)
    for length in HISTORY_LENGTHS:
        fixture_row = {
            "unit": "protocol_fixture",
            "arm": f"h{length}",
            "history_length": length,
            "absolute_metric": "collector_conformance",
            "numerator": 1,
            "denominator": 1,
            "rate": 1.0,
            "seed": [],
            "direction": "fixture_calibration_not_empirical_benefit",
            "censoring": {"unreplayable": 0, "unstable": 0},
            "provenance": "oracle_defined_fixture",
            "independent_unit_class": "fixture_not_empirical_game",
        }
        fixture_row["operand_checksum"] = canonical_hash(
            {
                key: fixture_row[key]
                for key in ("unit", "arm", "numerator", "denominator", "seed", "censoring")
            }
        )
        rows.append(fixture_row)
    return rows


def build_artifact(
    *,
    root: Path,
    run_date: str,
    duration_s: float,
    episodes: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    reduction: Mapping[str, Any],
    fixtures: Mapping[str, Any],
    registry_precheck: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    lifecycle: Mapping[str, Any] | None = None,
    phase_spans: Sequence[Mapping[str, Any]] = (),
) -> Json:
    """Build a terminal null unless a separate empirical benefit gate exists."""

    ready = fixtures.get("matched_support_ready_score") == 1
    validation_ok = not validation_receipts or all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in validation_receipts
    )
    empirical_rows = list(reduction.get("measurements") or [])
    fresh = bool(source_hashes)
    retention = (lifecycle or {}).get("passed", True) is True
    gates = {
        "validity": _gate(
            ready and validation_ok,
            "All protocol fixtures and declared checks must pass before empirical rows are readable.",
            True,
            ready and validation_ok,
        ),
        "readiness": _gate(
            ready,
            "Matched-key, fresh-reset, and provenance fixtures define collector readiness only.",
            1,
            int(ready),
        ),
        "benefit": _gate(
            False,
            "Empirical benefit requires a separate passing policy-outcome gate.",
            "separate_empirical_gate",
            "not_run",
        ),
        "retention": _gate(
            retention,
            "Persist, reload, update, release, and duplicate rejection must pass.",
            True,
            retention,
        ),
        "freshness": _gate(
            fresh, "Current source and raw evidence hashes prevent inherited outcomes.", True, fresh
        ),
    }
    observed_games = sorted({str(row.get("game")) for row in episodes})
    matched_games = {str(row.get("game")) for row in selection.get("selected_pairs") or []}
    rows = _rows_from_reduction(reduction, observed_games or ["fixture"])
    natural = deepcopy(reduction["natural_trajectory_denominator"])
    intervention = deepcopy(reduction["replay_intervention_denominator"])
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "honest_verdict": (
            "complete_null_matched_prefix_fixture_ready_empirical_benefit_not_established"
            if validation_ok
            else "complete_disqualified_required_validation_failed"
        ),
        "verdict_class": "null" if validation_ok else "disqualified",
        "flagged_adversarial": False,
        "gate_check_summary": {
            "passed": all(
                gates[name]["result"]
                for name in ("validity", "readiness", "retention", "freshness")
            ),
            "benefit_gate_intentionally_closed": True,
            "empirical_matched_key_count": len(empirical_rows),
        },
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended_independent_units": len(GAMES),
            "observed_independent_units": len(observed_games),
            "excluded_independent_units": len(set(GAMES) - set(observed_games)),
            "censored_independent_units": len(set(observed_games) - matched_games),
            "censored_prefix_occurrences": natural.get("long_prefix_censored_occurrences", 0),
            "intended_episode_count": len(GAMES) * len(SEEDS),
            "observed_episode_count": len(episodes),
            "selected_matched_keys": natural.get("selected_matched_keys", 0),
            "stable_replay_keys": intervention.get("stable_matched_keys", 0),
            "repeated_seeds_or_replays_multiply_units": False,
        },
        "natural_trajectory_denominator": natural,
        "replay_intervention_denominator": intervention,
        "primary_h0_vs_h1": deepcopy(reduction["primary_h0_vs_h1"]),
        "descriptive_history_lengths": [2, 4],
        "measurements": deepcopy(empirical_rows),
        "protocol_fixtures": deepcopy(dict(fixtures)),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "actual_inference_substrate_class": "no_model_load",
        "no_model_load": True,
        "MODEL_SPECS": [],
        "historical_model_identity": [],
        "model_invoked": False,
        "live_llm_invoked": False,
        "invocation_counts": zero_invocations(),
        "duration_s": float(duration_s),
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {"policy_episodes": list(SEEDS), "selection": "sha256_ascending"},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {
            "authenticated_sources": dict(source_hashes),
            "producer_artifacts": [],
            "conductor_pre_gate_records": [],
            "missing_producers": [],
        },
        "validation_receipts": deepcopy(list(validation_receipts)),
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "matched_support_ready_score": int(ready),
        "matched_protocol_path": {
            "games": list(GAMES),
            "seeds": list(SEEDS),
            "actions_per_episode": ACTION_LIMIT,
            "prefix_action_cap": PREFIX_ACTION_CAP,
            "matched_keys_per_game_cap": MAX_MATCHED_KEYS_PER_GAME,
            "target_key_fields": [
                "game",
                "level",
                "exact_raw_current_frame_hash",
                "legal_action_with_coordinates",
            ],
            "primary_history_comparison": [0, 1],
            "descriptive_history_lengths": [2, 4],
            "independent_unit": "game",
        },
        "solve_provenance": "live_agent_self_discovery",
        "new_game_level_solve_claimed": False,
        "registry_precheck": deepcopy(list(registry_precheck)),
        "registered_public_levels_are_development_history": True,
        "supervisor_scope": _supervisor_scope(episodes),
        "learning_lifecycle": deepcopy(dict(lifecycle or {"passed": True})),
        "raw_exp7597_used_as_measurement": False,
        "policy_choice_changed": False,
        "supervisor_arm_table_changed": False,
        "hud_masks_changed": False,
        "thinking_settings_changed": False,
        "acceptance_gates_changed": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "prior_verdict_disposition": {
            "experiment": 7597,
            "verdict": "complete_null_insufficient_history_support",
            "same_verdict_repeated": False,
            "exact_scope_retired": False,
            "scientific_hypothesis_retired": False,
        },
        "repository_root": str(Path(root).resolve()),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(tmp_path: Path) -> Json:
    """Build a small terminal artifact without a model, network, or real game."""

    episodes = [
        _synthetic_episode("su15", SEEDS[0], 1),
        _synthetic_episode("su15", SEEDS[1], 2),
    ]
    selection = select_matched_prefixes(episodes)
    measurement = replay_matched_pair(
        selection["selected_pairs"][0], _HiddenFixtureEnvironment, _fixture_execute
    )
    reduction = reduce_measurements(selection, [measurement])
    lifecycle = exercise_learning_lifecycle(tmp_path / "lifecycle")
    return build_artifact(
        root=repo_root(),
        run_date=RUN_DATE,
        duration_s=0.25,
        episodes=episodes,
        selection=selection,
        reduction=reduction,
        fixtures=measure_protocol_fixtures(),
        registry_precheck=[
            {"game": game, "present": True, "classification": "development_history"}
            for game in GAMES
        ],
        source_hashes={MODULE_REL.as_posix(): sha256_file(repo_root() / MODULE_REL)},
        lifecycle=lifecycle,
    )


def build_blocked_artifact(
    private_root: Path, failed_checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> Json:
    """Return one complete blocked record without pretending partial measurement."""

    artifact = build_test_artifact(private_root)
    first = failed_checks[0] if failed_checks else {"check": "unknown_precondition"}
    artifact.update(
        {
            "honest_verdict": f"complete_blocked_{first['check']}",
            "verdict_class": "blocked",
            "gate_check_summary": deepcopy(list(failed_checks)),
            "matched_support_ready_score": 0,
            "inference_substrate_class": "blocked_no_run",
            "planned_inference_substrate_class": "no_model_load",
            "actual_inference_substrate_class": "blocked_no_run",
            "duration_s": float(duration_s),
            "measurements": [],
        }
    )
    artifact["acceptance_gate_results"]["validity"] = _gate(
        False,
        "All named upstream checks must pass before collection starts.",
        True,
        False,
    )
    artifact["acceptance_gate_results"]["readiness"] = _gate(
        False,
        "Blocked runs create no readiness evidence.",
        1,
        0,
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, no-model accounting, row operands, gates, and hashes."""

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
    if artifact.get("MODEL_SPECS") != [] or artifact.get("no_model_load") is not True:
        errors.append("MODEL_SPECS")
    if artifact.get("model_invoked") is not False or artifact.get("live_llm_invoked") is not False:
        errors.append("model_invoked")
    counts = artifact.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(value != 0 for value in counts.values()):
        errors.append("invocation_counts")
    if not REQUIRED_FIELD_PRINCIPLES <= set(artifact.get("field_principles") or {}):
        errors.append("field_principles")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, Mapping) or set(gates) != {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }:
        errors.append("acceptance_gate_results")
    elif any(not isinstance(row, Mapping) or not row.get("principle") for row in gates.values()):
        errors.append("gate_principles")
    if artifact.get("verdict_class") == "blocked":
        summary = artifact.get("gate_check_summary")
        required = {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        if (
            not isinstance(summary, list)
            or not summary
            or any(not isinstance(row, Mapping) or not required <= set(row) for row in summary)
        ):
            errors.append("blocked_gate_check_summary")
        actual = artifact.get("actual_inference_substrate_class")
        if actual != "blocked_no_run":
            errors.append("blocked_substrate")
    for index, row in enumerate(artifact.get("rows") or []):
        if not isinstance(row, Mapping):
            errors.append(f"row_invalid:{index}")
            continue
        expected = canonical_hash(
            {
                key: row.get(key)
                for key in ("unit", "arm", "numerator", "denominator", "seed", "censoring")
            }
        )
        if row.get("operand_checksum") != expected:
            errors.append(f"row_operand_mismatch:{index}")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path) -> list[str]:
    """Reload the terminal bytes and validate every self-contained operand."""

    with Path(path).open(encoding="utf-8") as stream:
        return validate_artifact(json.load(stream))


def independent_replay(path: Path) -> list[str]:
    """Independently recompute row rates and intervention denominator totals."""

    with Path(path).open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    errors = validate_artifact(artifact)
    for index, row in enumerate(artifact.get("rows") or []):
        numerator = row.get("numerator")
        denominator = row.get("denominator")
        rate = row.get("rate")
        expected = numerator / denominator if denominator else None
        if rate != expected:
            errors.append(f"row_rate_mismatch:{index}")
    intervention = artifact.get("replay_intervention_denominator") or {}
    if intervention.get("fresh_environment_count") != 4 * intervention.get(
        "replayed_matched_keys", 0
    ):
        errors.append("fresh_environment_count")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Freeze explicit tests, changed module, CLI, and private validation outputs."""

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
        argv = tuple(part for part in command.argv if not part.startswith("--data-file="))
        if command.name in {"changed_module_coverage", "changed_module_coverage_report"}:
            argv = ("/usr/bin/env", f"COVERAGE_FILE={coverage_file}", *argv)
        transformed.append(CommandSpec(command.name, argv, command.scope, command.timeout_s))
    return transformed


def build_e2e_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Declare applicable ARC CPU E2Es and one private foreign-CWD smoke."""

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
            "foreign_cwd_llm_off_smoke",
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
            "private foreign-CWD LLM-off environment smoke",
            300.0,
        )
    )
    return commands


def build_terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Declare read-only readers for the exact terminal candidate bytes."""

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
            "fresh-process cold replay",
            300.0,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, *common, "--independent-reduce", str(candidate)),
            "independent row reduction",
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


def _all_passed(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return set(names) <= set(by_name) and all(by_name[name].get("passed") is True for name in names)


PREREQUISITE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7597_v663_arc_history_generalization.py"),
    Path("python/carnot/experiment_7589_v663_arc_output_boundary.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_solver_kit.py"),
    REGISTRY_REL,
    SPEC_REL,
    MODULE_REL,
)


def collect_preconditions(
    root: Path, private_root: Path
) -> tuple[list[Json], dict[str, str], list[Json]]:
    """Authenticate named inputs and classify every registered game as history."""

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
                "operator": "==",
                "expected": True,
                "observed": present,
                "passed": present,
            }
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)
    spec_path = resolved_root / SPEC_REL
    spec_has_requirement = spec_path.is_file() and REQUIREMENT_ID in spec_path.read_text(
        encoding="utf-8"
    )
    checks.append(
        {
            "check": "capability_requirement",
            "upstream": "OpenSpec",
            "path": SPEC_REL.as_posix(),
            "field": REQUIREMENT_ID,
            "operator": "contains",
            "expected": True,
            "observed": spec_has_requirement,
            "passed": spec_has_requirement,
        }
    )
    registry_path = resolved_root / REGISTRY_REL
    registry = (
        yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {}
    )
    entries = {
        str(row.get("game")): row for row in registry.get("games", []) if isinstance(row, Mapping)
    }
    registry_rows = [
        {
            "game": game,
            "present": game in entries,
            "levels_reproduced": entries.get(game, {}).get("levels_reproduced"),
            "classification": "development_history",
            "eligible_for_new_solve_credit": False,
        }
        for game in GAMES
    ]
    observed_games = sorted(row["game"] for row in registry_rows if row["present"])
    checks.append(
        {
            "check": "registry_precheck",
            "upstream": "ops/arc_solve_registry.yaml",
            "path": REGISTRY_REL.as_posix(),
            "field": "games_present_as_development_history",
            "operator": "==",
            "expected": sorted(GAMES),
            "observed": observed_games,
            "passed": observed_games == sorted(GAMES),
        }
    )
    private = Path(private_root).resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    results = (resolved_root / "results").resolve()
    owned = private.is_relative_to(temporary) and not private.is_relative_to(results)
    checks.append(
        {
            "check": "resource_ownership",
            "upstream": "current_exp7611_process",
            "path": str(private),
            "field": "private_root_outside_results",
            "operator": "==",
            "expected": True,
            "observed": owned,
            "passed": owned,
        }
    )
    checks.append(
        {
            "check": "model_call_declaration",
            "upstream": "current_exp7611_task",
            "path": MODULE_REL.as_posix(),
            "field": "MODEL_SPECS",
            "operator": "==",
            "expected": [],
            "observed": MODEL_SPECS,
            "passed": MODEL_SPECS == [],
        }
    )
    return checks, hashes, registry_rows


def _jsonable(value: Any) -> Any:  # pragma: no cover - live serialization
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
def _temporary_env(changes: Mapping[str, str | None]):  # pragma: no cover - live guard
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
    def __init__(self) -> None:
        self.calls = 0

    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        self.calls += 1
        raise RuntimeError("exp7611_induction_must_remain_disabled")


def _public_observation(frame: Any) -> Json | None:  # pragma: no cover - real frame seam
    if frame is None:
        return None
    from carnot.agentic.arc_competition_agent import _level_of

    actions = getattr(frame, "available_actions", []) or []
    state = getattr(frame, "state", None)
    return {
        "frame": _jsonable(getattr(frame, "frame", [])),
        "level": int(_level_of(frame)),
        "legal_actions": [str(getattr(action, "name", action)) for action in actions],
        "termination": str(getattr(state, "name", state)),
    }


def _selected_action(kind: Any, data: Any) -> Json:  # pragma: no cover - live action seam
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
) -> Json:  # pragma: no cover - isolated environment worker
    """Run one real policy episode while exposing only public runtime data."""

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
    with _temporary_env(
        {
            "CARNOT_ARC_DISABLE_INDUCTION": "1",
            "CARNOT_ARC_ACTION_PROVENANCE": None,
            "CARNOT_ARC_DECISION_TELEMETRY": None,
        }
    ):
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
            before_level = int(before["level"]) if before else 0
            after_level = int(after["level"]) if after else before_level
            steps.append(
                {
                    "action_index": action_index,
                    "observation": before,
                    "action": action,
                    "outcome": after,
                    "reset_boundary": kind == "RESET",
                    "level_boundary": before is not None and before_level != after_level,
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
                    pending_operation="policy_and_environment_step",
                )
                last_heartbeat = now
            if kind is None:
                break
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
            "per_game_masks_used": False,
            "live_llm_invoked": False,
            "steps": steps,
            "trajectory_supervisor": policy.trajectory_supervisor_diagnostics(),
            "termination": {
                "reason": reason,
                "action_opportunities": len(steps),
                "public_state": None
                if latest is None
                else _public_observation(latest)["termination"],
            },
            "duration_s": time.monotonic() - started,
            "model_call_counts": zero_invocations(),
        }
        errors = validate_natural_episode(raw)
        if errors:
            raise ValueError("live_raw_episode_invalid:" + ",".join(errors))
        atomic_json(resolved_output, raw)
        return raw


def frozen_episode_plan() -> list[Json]:
    return [
        {
            "episode_id": f"{game}:{seed}",
            "game": game,
            "seed": seed,
            "action_limit": ACTION_LIMIT,
            "policy": "E3AgentPolicy",
        }
        for game in GAMES
        for seed in SEEDS
    ]


def build_episode_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Run each natural trajectory in its own bounded process."""

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
                "one adapter-withheld no-model E3 natural trajectory",
                900.0,
            )
        )
    return commands


def prepare_command_parent(command: CommandSpec) -> None:
    """Create only private output parents named by one bounded command."""

    for index, argument in enumerate(command.argv):
        if argument.startswith("--basetemp="):
            Path(argument.split("=", 1)[1]).mkdir(parents=True, exist_ok=True)
        if argument.startswith("COVERAGE_FILE="):
            Path(argument.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        if argument in {"--raw-output", "--output"} and index + 1 < len(command.argv):
            Path(command.argv[index + 1]).parent.mkdir(parents=True, exist_ok=True)
    if "-C" in command.argv:
        Path(command.argv[command.argv.index("-C") + 1]).mkdir(parents=True, exist_ok=True)


def run_prepared_commands(
    root: Path, commands: Sequence[CommandSpec], *, log_dir: Path | None = None
) -> list[Json]:
    """Prepare private parents, then stream bounded subprocesses with heartbeats."""

    receipts: list[Json] = []
    for command in commands:
        prepare_command_parent(command)
        destination = log_dir or Path(tempfile.mkdtemp(prefix="carnot-exp7611-logs-"))
        receipts.extend(run_commands(root, [command], log_dir=destination, heartbeat_s=60.0))
    return receipts


class _RealReplayEnvironment:  # pragma: no cover - real replay seam
    def __init__(self, game: str) -> None:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        self.environment = arc.make(game, scorecard_id=arc.open_scorecard())

    def reset(self) -> Json:
        frame = _public_observation(self.environment.reset())
        if frame is None:
            raise RuntimeError("reset_returned_no_public_frame")
        return frame

    def step(self, action: Mapping[str, Any]) -> Json | None:
        from arcengine import GameAction

        kind = action["kind"]
        data = deepcopy(action.get("data"))
        return _public_observation(
            self.environment.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
        )


def replay_selected_pairs(
    selection: Mapping[str, Any], started: float
) -> list[Json]:  # pragma: no cover
    """Replay selected keys and report every completed key immediately."""

    rows: list[Json] = []
    last_heartbeat = time.monotonic()
    pairs = list(selection.get("selected_pairs") or [])
    for index, pair in enumerate(pairs):
        game = str(pair["game"])
        row = replay_matched_pair(
            pair,
            lambda game=game: _RealReplayEnvironment(game),
            _fixture_execute,
        )
        rows.append(row)
        progress(
            started,
            "replay_interventions",
            "unit_complete",
            completed=index + 1,
            total=len(pairs),
            game=game,
        )
        if time.monotonic() - last_heartbeat >= 60.0:
            progress(
                started,
                "replay_interventions",
                "heartbeat",
                completed=index + 1,
                total=len(pairs),
                pending_operation="fresh_prefix_replay",
            )
            last_heartbeat = time.monotonic()
    return rows


def _copy_closed_file(source: Path, destination: Path) -> Json:  # pragma: no cover - run custody
    """Copy one closed file and reject a byte mismatch."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    if sha256_file(source) != sha256_file(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"evidence_copy_hash_mismatch:{source}")
    temporary.replace(destination)
    return {
        "path": destination,
        "sha256": sha256_file(destination),
        "byte_count": destination.stat().st_size,
    }


def copy_episode_evidence(
    root: Path, private_root: Path
) -> tuple[list[Json], list[Json]]:  # pragma: no cover
    episodes: list[Json] = []
    receipts: list[Json] = []
    for unit in frozen_episode_plan():
        name = f"{unit['game']}-{unit['seed']}.json"
        source = private_root / "episodes" / name
        destination = root / RAW_REL / "episodes" / name
        copied = _copy_closed_file(source, destination)
        raw = json.loads(destination.read_text(encoding="utf-8"))
        errors = validate_natural_episode(raw)
        if errors:
            raise RuntimeError("copied_episode_invalid:" + ",".join(errors))
        episodes.append(raw)
        receipts.append(
            {
                "game": unit["game"],
                "seed": unit["seed"],
                "path": destination.relative_to(root).as_posix(),
                "sha256": copied["sha256"],
                "byte_count": copied["byte_count"],
                "scope": "current_exp7611_natural_trajectory",
            }
        )
    return episodes, receipts


def load_episode_checkpoint(root: Path) -> tuple[list[Json], list[Json]] | None:  # pragma: no cover
    """Resume all completed units only when the full frozen panel validates."""

    episodes: list[Json] = []
    receipts: list[Json] = []
    for unit in frozen_episode_plan():
        path = root / RAW_REL / "episodes" / f"{unit['game']}-{unit['seed']}.json"
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if validate_natural_episode(raw):
            return None
        episodes.append(raw)
        receipts.append(
            {
                "game": unit["game"],
                "seed": unit["seed"],
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
                "scope": "current_exp7611_natural_trajectory_checkpoint",
            }
        )
    return episodes, receipts


def copy_command_logs(
    root: Path, receipts: Sequence[Mapping[str, Any]], *, group: str
) -> list[Json]:  # pragma: no cover - run custody
    copied_receipts: list[Json] = []
    for index, receipt in enumerate(receipts):
        source = Path(str(receipt["log_path"]))
        if not source.is_absolute():
            source = root / source
        destination = root / RAW_REL / "logs" / group / f"{index:02d}_{receipt['name']}.log"
        copied = _copy_closed_file(source, destination)
        row = deepcopy(dict(receipt))
        row["private_log_path"] = str(source)
        row["log_path"] = destination.relative_to(root).as_posix()
        row["log_sha256"] = copied["sha256"]
        copied_receipts.append(row)
    return copied_receipts


def _phase_span(phase: str, phase_started: float, run_started: float) -> Json:
    ended = time.monotonic()
    return {
        "phase": phase,
        "started_s": phase_started - run_started,
        "ended_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def _selection_summary(selection: Mapping[str, Any]) -> Json:
    return {
        "selection_checksum": selection["selection_checksum"],
        "selection_uses_future_outcomes": False,
        "natural_trajectory_denominator": deepcopy(selection["natural_trajectory_denominator"]),
        "selected_pairs": [
            {
                "game": row["game"],
                "target_key": row["target_key"],
                "selection_hash": row["selection_hash"],
                "target_action": deepcopy(row["target_action"]),
                "target_observation_sha256": row["target_observation_sha256"],
                "histories_distinct": row["histories_distinct"],
                "history_keys": deepcopy(row["history_keys"]),
                "prefixes": [
                    {
                        "episode_id": prefix["episode_id"],
                        "seed": prefix["seed"],
                        "target_action_index": prefix["target_action_index"],
                        "prefix_action_count": prefix["prefix_action_count"],
                        "full_history_sha256": prefix["full_history_sha256"],
                    }
                    for prefix in row["prefixes"]
                ],
            }
            for row in selection["selected_pairs"]
        ],
    }


def _source_hashes(
    root: Path, prerequisites: Mapping[str, str], raw_receipts: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    hashes = dict(prerequisites)
    for relative in (MODULE_REL, TEST_REL, WRAPPER_REL, SPEC_REL, REGISTRY_REL):
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = sha256_file(path)
    for receipt in raw_receipts:
        hashes[str(receipt["path"])] = str(receipt["sha256"])
    return hashes


def run_experiment(
    root_path: Path, run_date: str, *, output_path: Path = RESULT_REL
) -> Json:  # pragma: no cover - declared integration entrypoint
    """Collect fresh trajectories, replay matched prefixes, validate, and publish."""

    started = time.monotonic()
    root = Path(root_path).resolve()
    if root != repo_root():
        raise ValueError(f"root_mismatch:{root}:{repo_root()}")
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    target = output_path if output_path.is_absolute() else root / output_path
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7611-v664-", dir="/tmp")).resolve()
    spans: list[Json] = []
    progress(started, "startup", "begin", root=root, private_root=private_root)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    checks, prerequisite_hashes, registry_rows = collect_preconditions(root, private_root)
    failed = [row for row in checks if row.get("passed") is not True]
    spans.append(_phase_span("preconditions", phase_started, started))
    progress(started, "preconditions", "after", checks=len(checks), failed=len(failed))
    if failed:
        blocked = build_blocked_artifact(
            private_root, failed, duration_s=time.monotonic() - started
        )
        blocked["preconditions_checked"] = deepcopy(checks)
        blocked["registry_precheck"] = registry_rows
        blocked["phase_spans"] = spans
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        progress(started, "publication", "before_atomic_blocked", path=target)
        atomic_json(target, blocked)
        progress(started, "publication", "after_atomic_blocked", path=target)
        return blocked

    phase_started = time.monotonic()
    progress(started, "natural_episodes", "checkpoint_probe_before")
    checkpoint = load_episode_checkpoint(root)
    progress(started, "natural_episodes", "checkpoint_probe_after", resumed=checkpoint is not None)
    if checkpoint is None:
        episode_commands = build_episode_commands(root, private_root)
        progress(started, "natural_episodes", "before_subprocesses", units=len(episode_commands))
        episode_private = run_prepared_commands(
            root, episode_commands, log_dir=private_root / "logs" / "episodes"
        )
        progress(
            started,
            "natural_episodes",
            "after_subprocesses",
            passed=sum(row.get("passed") is True for row in episode_private),
            units=len(episode_private),
        )
        if not all(row.get("passed") is True for row in episode_private):
            raise RuntimeError("natural_episode_subprocess_failed")
        progress(started, "evidence_copy", "before", units=len(episode_private))
        episodes, raw_receipts = copy_episode_evidence(root, private_root)
        episode_logs = copy_command_logs(root, episode_private, group="episodes")
        progress(started, "evidence_copy", "after", units=len(episodes))
    else:
        episodes, raw_receipts = checkpoint
        episode_logs = []
        progress(started, "natural_episodes", "checkpoint_resumed", units=len(episodes))
    spans.append(_phase_span("natural_episodes", phase_started, started))

    phase_started = time.monotonic()
    progress(started, "natural_selection", "before", episodes=len(episodes))
    selection = select_matched_prefixes(episodes)
    selection_path = root / RAW_REL / "natural-selection.json"
    atomic_json(selection_path, _selection_summary(selection))
    spans.append(_phase_span("natural_selection", phase_started, started))
    progress(
        started,
        "natural_selection",
        "after",
        selected=selection["natural_trajectory_denominator"]["selected_matched_keys"],
    )

    phase_started = time.monotonic()
    progress(started, "replay_interventions", "before", units=len(selection["selected_pairs"]))
    measurements = replay_selected_pairs(selection, started)
    reduction = reduce_measurements(selection, measurements)
    atomic_json(root / RAW_REL / "replay-reduction.json", reduction)
    spans.append(_phase_span("replay_interventions", phase_started, started))
    progress(started, "replay_interventions", "after", units=len(measurements))

    phase_started = time.monotonic()
    progress(started, "protocol_fixtures", "before")
    fixtures = measure_protocol_fixtures()
    spans.append(_phase_span("protocol_fixtures", phase_started, started))
    progress(started, "protocol_fixtures", "after", ready=fixtures["matched_support_ready_score"])

    phase_started = time.monotonic()
    progress(started, "learning_lifecycle", "before")
    lifecycle = exercise_learning_lifecycle(private_root / "learning")
    spans.append(_phase_span("learning_lifecycle", phase_started, started))
    progress(started, "learning_lifecycle", "after", passed=lifecycle["passed"])

    phase_started = time.monotonic()
    validation_commands = build_validation_commands(root, private_root / "validation")
    progress(started, "scoped_validation", "before_subprocesses", units=len(validation_commands))
    validation_private = run_prepared_commands(
        root, validation_commands, log_dir=private_root / "logs" / "validation"
    )
    validation = copy_command_logs(root, validation_private, group="scoped")
    spans.append(_phase_span("scoped_validation", phase_started, started))
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        passed=_all_passed(validation, REQUIRED_SCOPED_CHECKS),
    )

    phase_started = time.monotonic()
    e2e_commands = build_e2e_commands(root, private_root / "e2e")
    progress(started, "arc_e2e", "before_subprocesses", units=len(e2e_commands))
    e2e_private = run_prepared_commands(root, e2e_commands, log_dir=private_root / "logs" / "e2e")
    e2e = copy_command_logs(root, e2e_private, group="e2e")
    spans.append(_phase_span("arc_e2e", phase_started, started))
    progress(
        started,
        "arc_e2e",
        "after_subprocesses",
        passed=all(row.get("passed") is True for row in e2e),
    )

    source_hashes = _source_hashes(root, prerequisite_hashes, raw_receipts)
    for evidence_path in (selection_path, root / RAW_REL / "replay-reduction.json"):
        source_hashes[evidence_path.relative_to(root).as_posix()] = sha256_file(evidence_path)
    base_receipts = [*episode_logs, *validation, *e2e]
    candidate = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        episodes=episodes,
        selection=selection,
        reduction=reduction,
        fixtures=fixtures,
        registry_precheck=registry_rows,
        source_hashes=source_hashes,
        validation_receipts=base_receipts,
        lifecycle=lifecycle,
        phase_spans=spans,
    )
    candidate["preconditions_checked"] = deepcopy(checks)
    candidate["raw_episode_receipts"] = deepcopy(raw_receipts)
    candidate["affected_file_validation_manifest"] = {
        "schema": "carnot.exp7611.affected_validation_manifest.v1",
        "test_paths": [TEST_REL.as_posix()],
        "changed_modules": [MODULE_REL.as_posix()],
        "static_paths": [WRAPPER_REL.as_posix()],
        "spec_path": SPEC_REL.as_posix(),
    }
    candidate["reproducibility_checksum"] = reproducibility_checksum(candidate)
    errors = validate_artifact(candidate)
    if errors:
        raise RuntimeError("preliminary_candidate_invalid:" + ",".join(errors))
    preliminary_path = private_root / "terminal" / "preliminary-candidate.json"
    atomic_json(preliminary_path, candidate)

    phase_started = time.monotonic()
    progress(
        started,
        "terminal_readers_preliminary",
        "before_subprocesses",
        units=5,
        candidate_sha256=sha256_file(preliminary_path),
    )
    preliminary_private = run_prepared_commands(
        root,
        build_terminal_commands(root, preliminary_path),
        log_dir=private_root / "logs" / "terminal-preliminary",
    )
    preliminary = copy_command_logs(root, preliminary_private, group="terminal-preliminary")
    spans.append(_phase_span("terminal_readers_preliminary", phase_started, started))
    progress(
        started,
        "terminal_readers_preliminary",
        "after_subprocesses",
        passed=all(row.get("passed") is True for row in preliminary),
    )

    final = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=time.monotonic() - started,
        episodes=episodes,
        selection=selection,
        reduction=reduction,
        fixtures=fixtures,
        registry_precheck=registry_rows,
        source_hashes=source_hashes,
        validation_receipts=[*base_receipts, *preliminary],
        lifecycle=lifecycle,
        phase_spans=spans,
    )
    adversarial = next((row for row in preliminary if row.get("name") == "adversarial_verify"), {})
    flagged = "CRITICAL" in str(adversarial.get("output_tail") or "")
    final["flagged_adversarial"] = flagged
    if flagged:
        final["matched_support_ready_score"] = 0
        final["acceptance_gate_results"]["readiness"]["result"] = False
    final["preconditions_checked"] = deepcopy(checks)
    final["raw_episode_receipts"] = deepcopy(raw_receipts)
    final["affected_file_validation_manifest"] = deepcopy(
        candidate["affected_file_validation_manifest"]
    )
    final["terminal_reader_outcomes"] = {
        row["name"]: {
            "exit_code": row["exit_code"],
            "passed": row["passed"],
            "log_sha256": row["log_sha256"],
        }
        for row in preliminary
    }
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    final_path = private_root / "terminal" / "final-candidate.json"
    atomic_json(final_path, final)

    phase_started = time.monotonic()
    progress(
        started,
        "terminal_readers_exact",
        "before_subprocesses",
        units=5,
        candidate_sha256=sha256_file(final_path),
    )
    exact_private = run_prepared_commands(
        root,
        build_terminal_commands(root, final_path),
        log_dir=private_root / "logs" / "terminal-exact",
    )
    exact = copy_command_logs(root, exact_private, group="terminal-exact")
    spans.append(_phase_span("terminal_readers_exact", phase_started, started))
    exact_passed = all(row.get("passed") is True for row in exact)
    progress(started, "terminal_readers_exact", "after_subprocesses", passed=exact_passed)
    atomic_json(root / RAW_REL / "terminal-exact-reader-outcomes.json", {"receipts": exact})
    if not exact_passed:
        raise RuntimeError("exact_terminal_reader_failed")

    progress(started, "publication", "before_atomic", path=target)
    _copy_closed_file(final_path, target)
    progress(
        started,
        "publication",
        "after_atomic",
        path=target,
        verdict=final["honest_verdict"],
        sha256=sha256_file(target),
    )
    return final


def _date_argument(value: str) -> str:
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_REL)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate", type=Path)
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    modes.add_argument("--run-episode", action="store_true")
    parser.add_argument("--game", choices=GAMES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--raw-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
        run_live_episode(args.root, args.game, args.seed, args.raw_output)
        return 0
    run_experiment(args.root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
