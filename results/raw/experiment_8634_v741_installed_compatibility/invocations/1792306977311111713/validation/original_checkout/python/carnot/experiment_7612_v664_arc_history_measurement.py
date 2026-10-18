"""Measure reusable history support with the frozen matched-prefix protocol.

REQ-ARC-WMTE-7612. This module keeps the tested Experiment 7611 collector and
adds the fresh-evidence custody, support floor, and terminal reporting required
for the final measurement. It does not introduce a game-specific mechanism.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_7611_v664_arc_matched_support as base
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec

Json = dict[str, Any]
EXPERIMENT_ID = 7612
MILESTONE = "2026.09.664"
RUN_DATE = "20260924"
REQUIREMENT_ID = "REQ-ARC-WMTE-7612"
SCHEMA = "carnot.experiment_7612_v664_arc_history_measurement.v1"
RAW_EPISODE_SCHEMA = "carnot.exp7612.raw_public_episode.v1"
GAMES = base.GAMES
SEEDS = base.SEEDS
HISTORY_LENGTHS = base.HISTORY_LENGTHS
ACTION_LIMIT = base.ACTION_LIMIT
PREFIX_ACTION_CAP = base.PREFIX_ACTION_CAP
MAX_MATCHED_KEYS_PER_GAME = base.MAX_MATCHED_KEYS_PER_GAME
EPISODE_WATCHDOG_S = 120.0
TOTAL_EPISODE_REPLAY_BUDGET_S = 1800.0
STABLE_KEYS_PER_GAME_FLOOR = 20
GAME_SUPPORT_FLOOR = 3
MODEL_SPECS: list[Json] = []
no_model_load = True

RESULT_REL = Path("results/experiment_7612_v664_arc_history_measurement.json")
RAW_REL = Path("results/raw/experiment_7612_v664_arc_history_measurement")
MODULE_REL = Path("python/carnot/experiment_7612_v664_arc_history_measurement.py")
TEST_REL = Path("tests/python/test_experiment_7612_v664_arc_history_measurement.py")
WRAPPER_REL = Path("scripts/experiments/experiment_7612_v664_arc_history_measurement.py")
SPEC_REL = base.SPEC_REL
REGISTRY_REL = base.REGISTRY_REL
PRIOR_RESULT_REL = Path("results/experiment_7611_v664_arc_matched_support.json")
PRIOR_MODULE_REL = base.MODULE_REL
PRIOR_SCHEMA = "carnot.experiment_7611_v664_arc_matched_support.v1"

EXPECTED_PRIOR_PROTOCOL = {
    "actions_per_episode": 600,
    "descriptive_history_lengths": [2, 4],
    "games": list(GAMES),
    "independent_unit": "game",
    "matched_keys_per_game_cap": 20,
    "prefix_action_cap": 128,
    "primary_history_comparison": [0, 1],
    "seeds": list(SEEDS),
    "target_key_fields": [
        "game",
        "level",
        "exact_raw_current_frame_hash",
        "legal_action_with_coordinates",
    ],
}

FIELD_PRINCIPLES = {
    **base.FIELD_PRINCIPLES,
    "arc_measurement_ready_score": "One records complete truthful runtime and replay outcomes, independent of support.",
    "history_support_score": "One requires at least 20 stable matched keys in each of at least three games.",
    "per_game_results": "Natural and intervention denominators, witnesses, exclusions, censorship, and hashes stay separate.",
    "trajectory_supervisor": "Only actual fired, helped, and unredirected stagnation counts support refinement.",
}
REQUIRED_FIELD_PRINCIPLES = set(base.REQUIRED_FIELD_PRINCIPLES) | {
    "arc_measurement_ready_score",
    "history_support_score",
    "per_game_results",
    "trajectory_supervisor",
}

PREREQUISITE_PATHS = tuple(path for path in base.PREREQUISITE_PATHS if path != base.MODULE_REL) + (
    PRIOR_MODULE_REL,
    PRIOR_RESULT_REL,
    MODULE_REL,
    TEST_REL,
    WRAPPER_REL,
)

_ORIGINAL_BUILD_ARTIFACT = base.build_artifact
_ORIGINAL_BUILD_BLOCKED_ARTIFACT = base.build_blocked_artifact
_ORIGINAL_BUILD_EPISODE_COMMANDS = base.build_episode_commands
_ORIGINAL_COLLECT_PRECONDITIONS = base.collect_preconditions
_ORIGINAL_COPY_EPISODE_EVIDENCE = base.copy_episode_evidence
_ORIGINAL_INDEPENDENT_REPLAY = base.independent_replay
_ORIGINAL_LOAD_EPISODE_CHECKPOINT = base.load_episode_checkpoint
_ORIGINAL_VALIDATE_ARTIFACT = base.validate_artifact
_ORIGINAL_VALIDATE_NATURAL_EPISODE = base.validate_natural_episode
_ORIGINAL_REDUCE_MEASUREMENTS = base.reduce_measurements
_ORIGINAL_RUN_LIVE_EPISODE = base.run_live_episode


def repo_root() -> Path:
    """Resolve the worktree from this reusable module, not the caller's directory."""

    return Path(__file__).resolve().parents[2]


canonical_hash = base.canonical_hash
sha256_file = base.sha256_file
reproducibility_checksum = base.reproducibility_checksum
replay_matched_pair = base.replay_matched_pair
progress = base.progress
zero_invocations = base.zero_invocations


@contextmanager
def _configured_base() -> Iterator[None]:
    """Apply the new experiment identity only while reused behavior executes."""

    overrides = {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "MILESTONE": MILESTONE,
        "RUN_DATE": RUN_DATE,
        "REQUIREMENT_ID": REQUIREMENT_ID,
        "SCHEMA": SCHEMA,
        "RAW_EPISODE_SCHEMA": RAW_EPISODE_SCHEMA,
        "RESULT_REL": RESULT_REL,
        "RAW_REL": RAW_REL,
        "MODULE_REL": MODULE_REL,
        "TEST_REL": TEST_REL,
        "WRAPPER_REL": WRAPPER_REL,
        "FIELD_PRINCIPLES": FIELD_PRINCIPLES,
        "REQUIRED_FIELD_PRINCIPLES": REQUIRED_FIELD_PRINCIPLES,
        "PREREQUISITE_PATHS": PREREQUISITE_PATHS,
        "build_artifact": build_artifact,
        "build_blocked_artifact": build_blocked_artifact,
        "build_episode_commands": build_episode_commands,
        "collect_preconditions": collect_preconditions,
        "copy_episode_evidence": copy_episode_evidence,
        "load_episode_checkpoint": load_episode_checkpoint,
        "reduce_measurements": reduce_measurements,
        "replay_selected_pairs": replay_selected_pairs,
        "run_live_episode": run_live_episode,
        "validate_artifact": validate_artifact,
        "cold_replay": cold_replay,
        "independent_replay": independent_replay,
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def authenticate_exp7611_protocol(root: Path) -> tuple[Json, Json]:
    """Authenticate the prior protocol without importing its outcomes as evidence."""

    path = Path(root).resolve() / PRIOR_RESULT_REL
    observed: Any = None
    valid_artifact = False
    if path.is_file():
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            observed = artifact.get("matched_protocol_path")
            valid_artifact = (
                artifact.get("experiment_id") == 7611
                and artifact.get("schema") == PRIOR_SCHEMA
                and str(artifact.get("honest_verdict") or "").startswith("complete_")
                and artifact.get("reproducibility_checksum")
                == base.reproducibility_checksum(artifact)
            )
        except (OSError, ValueError, TypeError):
            valid_artifact = False
    passed = valid_artifact and observed == EXPECTED_PRIOR_PROTOCOL
    check = {
        "check": "exp7611_protocol",
        "upstream": "experiment_7611",
        "path": PRIOR_RESULT_REL.as_posix(),
        "field": "matched_protocol_path",
        "operator": "==",
        "expected": deepcopy(EXPECTED_PRIOR_PROTOCOL),
        "observed": deepcopy(observed),
        "passed": passed,
    }
    receipt = {
        "path": PRIOR_RESULT_REL.as_posix(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }
    return check, receipt


def collect_preconditions(
    root: Path, private_root: Path
) -> tuple[list[Json], dict[str, str], list[Json]]:
    """Add exact Exp7611 protocol authentication to the reused source checks."""

    with _configured_base():
        checks, hashes, registry_rows = _ORIGINAL_COLLECT_PRECONDITIONS(root, private_root)
    protocol_check, receipt = authenticate_exp7611_protocol(root)
    checks.append(protocol_check)
    if receipt["sha256"] is not None:
        hashes[receipt["path"]] = receipt["sha256"]
    return checks, hashes, registry_rows


def validate_natural_episode(raw: Mapping[str, Any]) -> list[str]:
    """Require the generic policy source stage in addition to Exp7611 custody."""

    with _configured_base():
        errors = _ORIGINAL_VALIDATE_NATURAL_EPISODE(raw)
    for index, step in enumerate(raw.get("steps") or []):
        if not isinstance(step, Mapping) or not step.get("source_stage"):
            errors.append(f"source_stage_missing:{index}")
    return list(dict.fromkeys(errors))


def select_matched_prefixes(
    episodes: Sequence[Mapping[str, Any]], *, max_per_game: int = MAX_MATCHED_KEYS_PER_GAME
) -> Json:
    """Reuse hash-only selection after enforcing current-evidence fields."""

    for raw in episodes:
        errors = validate_natural_episode(raw)
        if errors:
            raise ValueError("raw_episode_invalid:" + ",".join(errors))
    with _configured_base():
        return base.select_matched_prefixes(episodes, max_per_game=max_per_game)


def reduce_measurements(
    selection: Mapping[str, Any], measurements: Sequence[Mapping[str, Any]]
) -> Json:
    """Keep selected, completed, and time-censored replay probes distinct."""

    reduction = _ORIGINAL_REDUCE_MEASUREMENTS(selection, measurements)
    intended = len(selection.get("selected_pairs") or [])
    completed = len(measurements)
    intervention = reduction["replay_intervention_denominator"]
    intervention["intended_matched_keys"] = intended
    intervention["replayed_matched_keys"] = completed
    intervention["unstarted_time_censored_keys"] = max(0, intended - completed)
    return reduction


def reduce_per_game(
    episodes: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    reduction: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> list[Json]:
    """Expose natural coverage and replay outcomes for each independent game."""

    selected = list(selection.get("selected_pairs") or [])
    reduced = list(reduction.get("measurements") or [])
    results: list[Json] = []
    for game in GAMES:
        game_episodes = [row for row in episodes if row.get("game") == game]
        natural = select_matched_prefixes(game_episodes)["natural_trajectory_denominator"]
        intended = [row for row in selected if row.get("game") == game]
        measured = [row for row in reduced if row.get("game") == game]
        stable = [row for row in measured if row.get("replay_exclusion") is None]
        witnesses = sum(row.get("history_disambiguation_witness") is True for row in stable)
        game_hashes = {
            path: digest
            for path, digest in source_hashes.items()
            if f"/episodes/{game}-" in f"/{path}"
        }
        results.append(
            {
                "game": game,
                "natural_coverage": deepcopy(natural),
                "intervention_coverage": {
                    "intended_matched_keys": len(intended),
                    "replayed_matched_keys": len(measured),
                    "stable_matched_keys": len(stable),
                    "witness_count": witnesses,
                    "stable_agreement_count": len(stable) - witnesses,
                    "excluded_unreplayable_keys": sum(
                        row.get("replay_exclusion") == "unreplayable_prefix" for row in measured
                    ),
                    "excluded_unstable_keys": sum(
                        row.get("replay_exclusion") == "unstable_prefix" for row in measured
                    ),
                },
                "time_censoring": {
                    "episode_timeouts": 0,
                    "unstarted_replay_keys": max(0, len(intended) - len(measured)),
                    "over_length_prefix_occurrences": natural["long_prefix_censored_occurrences"],
                },
                "source_hashes": game_hashes,
                "solve_provenance": "live_agent_self_discovery",
            }
        )
    return results


def reduce_support(per_game: Sequence[Mapping[str, Any]]) -> Json:
    """Create a game-cluster estimate only after the preregistered support floor."""

    supported = [
        row
        for row in per_game
        if int((row.get("intervention_coverage") or {}).get("stable_matched_keys", 0))
        >= STABLE_KEYS_PER_GAME_FLOOR
    ]
    ready = len(supported) >= GAME_SUPPORT_FLOOR
    rates = [
        row["intervention_coverage"]["witness_count"]
        / row["intervention_coverage"]["stable_matched_keys"]
        for row in supported
    ]
    mean = round(sum(rates) / len(rates), 12) if ready else None
    interval = None
    if ready and mean is not None:
        variance = sum((value - mean) ** 2 for value in rates) / (len(rates) - 1)
        margin = 1.96 * math.sqrt(variance / len(rates))
        interval = [round(max(0.0, mean - margin), 12), round(min(1.0, mean + margin), 12)]
    return {
        "stable_keys_per_game_floor": STABLE_KEYS_PER_GAME_FLOOR,
        "game_support_floor": GAME_SUPPORT_FLOOR,
        "games_meeting_stable_key_floor": [str(row["game"]) for row in supported],
        "history_support_score": int(ready),
        "game_cluster_rate": mean,
        "game_cluster_interval_95": interval,
        "independent_validity": "valid_current_measurement" if per_game else "no_current_units",
        "support_disposition": "supported" if ready else "insufficient_matched_support",
    }


def _trajectory_supervisor(episodes: Sequence[Mapping[str, Any]]) -> Json:
    """Reduce only observed supervisor outcomes from the current episodes."""

    fired = helped = stagnations = 0
    arms: dict[str, Json] = {}
    for episode in episodes:
        receipt = episode.get("trajectory_supervisor") or {}
        stagnations += int(receipt.get("stagnations_unredirected", 0))
        for name, outcome in (receipt.get("would_have_arm_outcomes") or {}).items():
            if not isinstance(outcome, Mapping):
                continue
            arm = arms.setdefault(str(name), {"fired": 0, "helped": 0})
            arm["fired"] += int(outcome.get("fired", 0))
            arm["helped"] += int(outcome.get("helped", 0))
            fired += int(outcome.get("fired", 0))
            helped += int(outcome.get("helped", 0))
    return {
        "fired_count": fired,
        "helped_count": helped,
        "stagnations_unredirected": stagnations,
        "per_arm": arms,
        "refinement_supported": fired > 0 and helped > 0,
        "empty_ledger_requires_no_refinement": fired == 0,
    }


def build_artifact(**kwargs: Any) -> Json:
    """Upgrade the reused artifact with measurement and support as separate gates."""

    with _configured_base():
        artifact = _ORIGINAL_BUILD_ARTIFACT(**kwargs)
    episodes = list(kwargs["episodes"])
    selection = kwargs["selection"]
    reduction = kwargs["reduction"]
    source_hashes = kwargs["source_hashes"]
    per_game = reduce_per_game(episodes, selection, reduction, source_hashes)
    support = reduce_support(per_game)
    intervention = reduction["replay_intervention_denominator"]
    replay_accounted = intervention["intended_matched_keys"] == (
        intervention["replayed_matched_keys"] + intervention.get("unstarted_time_censored_keys", 0)
    )
    episode_ids = {str(row.get("episode_id")) for row in episodes}
    measurement_ready = (
        len(episode_ids) == len(GAMES) * len(SEEDS)
        and kwargs["fixtures"].get("matched_support_ready_score") == 1
        and replay_accounted
    )
    support_ready = support["history_support_score"] == 1
    prior_hash = source_hashes.get(PRIOR_RESULT_REL.as_posix())
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "honest_verdict": (
                "complete_null_matched_support_observed_benefit_not_established"
                if support_ready
                else "complete_null_insufficient_matched_support"
            ),
            "verdict_class": "null",
            "arc_measurement_ready_score": int(measurement_ready),
            "history_support_score": int(support_ready),
            "per_game_results": per_game,
            "support_reduction": support,
            "cross_game_history_claim": (
                {
                    "rate": support["game_cluster_rate"],
                    "interval_95": support["game_cluster_interval_95"],
                    "scope": support["games_meeting_stable_key_floor"],
                    "claim": "between_history_disagreement_on_stable_matched_keys",
                }
                if support_ready
                else None
            ),
            "trajectory_supervisor": _trajectory_supervisor(episodes),
            "field_principles": dict(FIELD_PRINCIPLES),
            "raw_exp7597_used_as_measurement": False,
            "raw_exp7611_used_as_measurement": False,
            "exp7611_protocol_authenticated": prior_hash is not None,
            "work_budget": {
                "total_episode_replay_budget_s": TOTAL_EPISODE_REPLAY_BUDGET_S,
                "episode_watchdog_s": EPISODE_WATCHDOG_S,
                "completed_episodes": len(episodes),
                "unstarted_episodes": max(0, len(GAMES) * len(SEEDS) - len(episodes)),
                "completed_replay_keys": intervention["replayed_matched_keys"],
                "unstarted_replay_keys": intervention.get("unstarted_time_censored_keys", 0),
            },
        }
    )
    artifact["matched_protocol_path"].update(
        {
            "total_episode_replay_budget_s": TOTAL_EPISODE_REPLAY_BUDGET_S,
            "episode_watchdog_s": EPISODE_WATCHDOG_S,
            "support_floor": {
                "stable_keys_per_game": STABLE_KEYS_PER_GAME_FLOOR,
                "games": GAME_SUPPORT_FLOOR,
            },
        }
    )
    artifact["sample_size_budget"].update(
        {
            "intended_matched_keys": intervention["intended_matched_keys"],
            "observed_matched_keys": intervention["replayed_matched_keys"],
            "excluded_matched_keys": intervention["excluded_unreplayable_keys"]
            + intervention["excluded_unstable_keys"],
            "censored_matched_keys": intervention.get("unstarted_time_censored_keys", 0),
        }
    )
    artifact["acceptance_gate_results"]["readiness"] = base._gate(
        measurement_ready,
        "Complete truthful episode and replay accounting defines measurement readiness.",
        1,
        int(measurement_ready),
    )
    artifact["acceptance_gate_results"]["benefit"] = base._gate(
        False,
        "History support is not a policy-benefit gate and exact fixtures are circular positives.",
        "separate_empirical_benefit_gate",
        "not_run",
    )
    artifact["gate_check_summary"] = {
        "passed": measurement_ready,
        "independent_validity": support["independent_validity"],
        "history_support_floor_passed": support_ready,
        "benefit_gate_intentionally_closed": True,
        "empirical_matched_key_count": intervention["replayed_matched_keys"],
    }
    artifact["source_artifact_hashes"]["conductor_pre_gate_records"] = (
        []
        if prior_hash is None
        else [
            {
                "experiment_id": 7611,
                "path": PRIOR_RESULT_REL.as_posix(),
                "sha256": prior_hash,
                "fields_imported": ["matched_protocol_path"],
                "outcomes_imported": False,
            }
        ]
    )
    artifact["prior_verdict_disposition"] = {
        "experiment": 7611,
        "verdict": "complete_null_matched_prefix_fixture_ready_empirical_benefit_not_established",
        "same_verdict_repeated": False,
        "same_insufficient_support_disposition_repeated": not support_ready,
        "exact_scope_retired": not support_ready,
        "scientific_hypothesis_retired": False,
        "reopen_requires": "different_representation_or_sampling_mechanism",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    private_root: Path, failed_checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> Json:
    """Return a complete external block with no partial measurement claim."""

    with _configured_base():
        saved = base.build_artifact
        base.build_artifact = _ORIGINAL_BUILD_ARTIFACT
        try:
            artifact = _ORIGINAL_BUILD_BLOCKED_ARTIFACT(
                private_root, failed_checks, duration_s=duration_s
            )
        finally:
            base.build_artifact = saved
    first = failed_checks[0] if failed_checks else {"check": "unknown_precondition"}
    artifact.update(
        {
            "honest_verdict": f"complete_blocked_{first['check']}",
            "verdict_class": "blocked",
            "gate_check_summary": deepcopy(list(failed_checks)),
            "arc_measurement_ready_score": 0,
            "history_support_score": 0,
            "per_game_results": [],
            "support_reduction": reduce_support([]),
            "cross_game_history_claim": None,
            "trajectory_supervisor": _trajectory_supervisor([]),
            "field_principles": dict(FIELD_PRINCIPLES),
            "raw_exp7611_used_as_measurement": False,
            "exp7611_protocol_authenticated": False,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Check terminal identity, support suppression, and required measurement fields."""

    with _configured_base():
        errors = _ORIGINAL_VALIDATE_ARTIFACT(artifact)
    required = {
        "arc_measurement_ready_score",
        "history_support_score",
        "per_game_results",
        "trajectory_supervisor",
    }
    if not required <= set(artifact):
        errors.append("measurement_fields")
    support = artifact.get("support_reduction") or {}
    if artifact.get("history_support_score") != support.get("history_support_score"):
        errors.append("history_support_score")
    expected_support = reduce_support(artifact.get("per_game_results") or [])
    if artifact.get("history_support_score") != expected_support["history_support_score"]:
        errors.append("history_support_score")
    if artifact.get("history_support_score") == 0:
        if artifact.get("cross_game_history_claim") is not None:
            errors.append("unsupported_cross_game_claim")
        if (
            support.get("game_cluster_rate") is not None
            or support.get("game_cluster_interval_95") is not None
        ):
            errors.append("unsupported_cluster_interval")
    if artifact.get("verdict_class") == "null" and artifact.get("history_support_score") == 0:
        if artifact.get("honest_verdict") != "complete_null_insufficient_matched_support":
            errors.append("support_verdict")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path) -> list[str]:
    """Reload raw trajectories, authenticate bytes, and reproduce natural coverage."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_artifact(artifact)
    receipts = artifact.get("raw_episode_receipts") or []
    episodes: list[Json] = []
    root = Path(str(artifact.get("repository_root") or repo_root())).resolve()
    for index, receipt in enumerate(receipts):
        raw_path = root / str(receipt.get("path"))
        if not raw_path.is_file() or sha256_file(raw_path) != receipt.get("sha256"):
            errors.append(f"raw_episode_custody:{index}")
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_errors = validate_natural_episode(raw)
        errors.extend(f"raw_episode:{index}:{error}" for error in raw_errors)
        if not raw_errors:
            episodes.append(raw)
    if episodes:
        selected = select_matched_prefixes(episodes)
        if selected["natural_trajectory_denominator"] != artifact.get(
            "natural_trajectory_denominator"
        ):
            errors.append("cold_natural_reduction")
    return list(dict.fromkeys(errors))


def independent_replay(path: Path) -> list[str]:
    """Recompute row rates and each per-game support operand independently."""

    with _configured_base():
        errors = _ORIGINAL_INDEPENDENT_REPLAY(path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = reduce_support(artifact.get("per_game_results") or [])
    if expected != artifact.get("support_reduction"):
        errors.append("per_game_support_reduction")
    return list(dict.fromkeys(errors))


def build_test_artifact(tmp_path: Path) -> Json:
    """Build all twelve synthetic units without using them as scientific evidence."""

    episodes: list[Json] = []
    for game in GAMES:
        episodes.extend(
            [
                base._synthetic_episode(game, SEEDS[0], 1),
                base._synthetic_episode(game, SEEDS[1], 2),
            ]
        )
    for episode in episodes:
        episode["schema"] = RAW_EPISODE_SCHEMA
        for step in episode["steps"]:
            step["source_stage"] = "e3_agent_policy"
    selection = select_matched_prefixes(episodes)
    measurements = [
        replay_matched_pair(pair, base._HiddenFixtureEnvironment, base._fixture_execute)
        for pair in selection["selected_pairs"]
    ]
    return build_artifact(
        root=repo_root(),
        run_date=RUN_DATE,
        duration_s=0.25,
        episodes=episodes,
        selection=selection,
        reduction=reduce_measurements(selection, measurements),
        fixtures=base.measure_protocol_fixtures(),
        registry_precheck=[],
        source_hashes={PRIOR_RESULT_REL.as_posix(): "sha256:test-prior"},
        lifecycle={"passed": True},
    )


def run_live_episode(
    root: Path,
    game: str,
    seed: int,
    output_path: Path,
    *,
    action_limit: int = ACTION_LIMIT,
) -> Json:  # pragma: no cover - isolated live worker
    """Run the reused live policy and add its generic source-stage receipt."""

    raw = _ORIGINAL_RUN_LIVE_EPISODE(root, game, seed, output_path, action_limit=action_limit)
    raw["schema"] = RAW_EPISODE_SCHEMA
    for step in raw["steps"]:
        step["source_stage"] = "e3_agent_policy"
    errors = validate_natural_episode(raw)
    if errors:
        raise ValueError("live_raw_episode_invalid:" + ",".join(errors))
    atomic_json(Path(output_path).resolve(), raw)
    return raw


def build_episode_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Use one 120-second watchdog for each isolated fresh policy episode."""

    with _configured_base():
        commands = _ORIGINAL_BUILD_EPISODE_COMMANDS(root, private_root)
    return [
        CommandSpec(command.name, command.argv, command.scope, EPISODE_WATCHDOG_S)
        for command in commands
    ]


def copy_episode_evidence(
    root: Path, private_root: Path
) -> tuple[list[Json], list[Json]]:  # pragma: no cover - live custody seam
    """Keep reused copying behavior while naming the current experiment scope."""

    episodes, receipts = _ORIGINAL_COPY_EPISODE_EVIDENCE(root, private_root)
    for receipt in receipts:
        receipt["scope"] = "current_exp7612_natural_trajectory"
    return episodes, receipts


def load_episode_checkpoint(
    root: Path,
) -> tuple[list[Json], list[Json]] | None:  # pragma: no cover - live resume seam
    """Accept only the complete Exp7612 checkpoint panel."""

    checkpoint = _ORIGINAL_LOAD_EPISODE_CHECKPOINT(root)
    if checkpoint is not None:
        for receipt in checkpoint[1]:
            receipt["scope"] = "current_exp7612_natural_trajectory_checkpoint"
    return checkpoint


def replay_selected_pairs(
    selection: Mapping[str, Any], started: float
) -> list[Json]:  # pragma: no cover - real replay seam
    """Checkpoint each completed key and stop new work at the total budget."""

    rows: list[Json] = []
    pairs = list(selection.get("selected_pairs") or [])
    checkpoint_root = repo_root() / RAW_REL / "replay-checkpoints"
    for index, pair in enumerate(pairs):
        if time.monotonic() - started >= TOTAL_EPISODE_REPLAY_BUDGET_S:
            progress(
                started,
                "replay_interventions",
                "budget_censored",
                completed=index,
                total=len(pairs),
            )
            break
        game = str(pair["game"])
        row = replay_matched_pair(
            pair,
            lambda game=game: base._RealReplayEnvironment(game),
            base._fixture_execute,
        )
        rows.append(row)
        atomic_json(checkpoint_root / f"{game}-{pair['target_key'].split(':')[-1]}.json", row)
        progress(
            started,
            "replay_interventions",
            "unit_complete",
            completed=index + 1,
            total=len(pairs),
            game=game,
        )
    return rows


def run_experiment(
    root_path: Path, run_date: str, *, output_path: Path = RESULT_REL
) -> Json:  # pragma: no cover - declared integration entrypoint
    """Run the existing collector under the authenticated Exp7612 identity."""

    with _configured_base():
        return base.run_experiment(root_path, run_date, output_path=output_path)


def parse_args(argv: Sequence[str] | None = None) -> Any:  # pragma: no cover - CLI parser
    """Expose the reused bounded command surface with Exp7612 defaults."""

    with _configured_base():
        return base.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """Delegate CLI modes while the reused module carries this experiment identity."""

    with _configured_base():
        return base.main(argv)


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
