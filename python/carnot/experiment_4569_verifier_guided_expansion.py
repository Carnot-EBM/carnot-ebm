"""Experiment 4569: verifier-guided frontier expansion.

Spec refs: REQ-CAPSTONE-4569, SCENARIO-CAPSTONE-4569,
SCENARIO-CAPSTONE-4569-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot import experiment_4550_honest_sprint_metric as exp4550
from carnot import experiment_4556_verifier_router_generic_transfer as exp4556
from carnot.agentic.arc_discriminative_router import (
    DEFAULT_CHECKPOINT_RELATIVE_PATH,
    RandomExpansionPriority,
    checkpoint_sha256,
    dominant_feature_family_from_checkpoint,
    load_cross_game_discriminative_expansion_priority,
)


JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]
VariantRunnerFactory = Callable[[str], VariantRunner]

RESULT_RELATIVE_PATH = "results/experiment_4569_verifier_guided_expansion.json"
EXPERIMENT_ID = "experiment_4569_verifier_guided_expansion"
SCHEMA = "carnot.exp4569.verifier_guided_expansion.v1"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4569
BASELINE_REFERENCE_RATE = 0.04
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
DEFAULT_BOOTSTRAPS = 1000
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "generic_transfer_rate_with_expansion",
    "generic_transfer_rate_baseline",
    "transfer_delta",
    "transfer_ci",
    "expanded_states_to_goal_with_vs_without",
    "winner_generated",
    "expansions_used",
    "random_priority_control_passed",
    "false_negative_risk_checked",
    "null_delta_methodology_note",
    "solve_rate_preserved",
    "chosen_submitted_config",
    "missing_verifier_gaps",
    "offline_reproduced",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: verifier_guided_expansion_generic_transfer_<n>_above_0.04 "
            "OR complete: verifier_guided_expansion_no_value_honest_null_generation_gap_sharpened."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- the verifier scores offline frontier "
            "candidates, no LLM load (1s floor)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the DiscriminativeVerifier guides search by a LEARNED cross-game "
            "signal, oracle-DISTINCT from the executable win-check."
        )
    },
    "generic_transfer_rate_with_expansion": {
        "principle": (
            "the HEADLINE -- held-out variant transfer WITH verifier-guided expansion; > 0.04 "
            "with CI-excl-baseline is the non-circular live-value evidence."
        )
    },
    "generic_transfer_rate_baseline": {
        "principle": "0.04 -- the .420/.421 B1 bare-solver baseline, measured the SAME way."
    },
    "transfer_delta": {
        "principle": (
            "with_expansion - baseline, emitted explicitly so a null (0.0) is annotated, "
            "not a control==best TAUTOLOGY false-positive."
        )
    },
    "transfer_ci": {
        "principle": (
            "bootstrap CI on the transfer delta; a claim above baseline requires the CI to exclude "
            "the baseline."
        )
    },
    "expanded_states_to_goal_with_vs_without": {
        "principle": (
            "the search-efficiency evidence -- did verifier-guided expansion reach the goal in "
            "fewer expanded states."
        )
    },
    "winner_generated": {"principle": "did the search GENERATE the winning candidate at all."},
    "expansions_used": {
        "principle": (
            "the bounded expansion budget consumed -- proves the Scaling-Flaws guard held and "
            "bounds wall-clock."
        )
    },
    "random_priority_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- verifier-expansion must beat random-priority on the SAME variants."
        )
    },
    "false_negative_risk_checked": {
        "principle": "a no-value null is valid only if the random-priority control passed."
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when transfer_delta==0.0 -- states the equality is an honest no-value null, "
            "not a measurement bug."
        )
    },
    "solve_rate_preserved": {"principle": "HARD gate -- expansion must NOT drop solve-rate."},
    "chosen_submitted_config": {
        "principle": "enable verifier-guided expansion on a live win; unchanged on null."
    },
    "missing_verifier_gaps": {
        "principle": "if no value, the residual generation gap."
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent verifier/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, DiscriminativeVerifier loadable); "
            "pre-empts missing-resource fabrication."
        )
    },
}


def _attempt_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and attempt.get("solved") is True


def _first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_arcade") is not True:
        return "offline_arcade"
    if preconditions.get("arc_value_learner_discriminative_import") is not True:
        return "discriminative_verifier_import"
    if preconditions.get("trained_verifier_loadable") is not True:
        return "trained_verifier"
    if preconditions.get("cross_game_corpus_loadable") is not True:
        return "cross_game_corpus"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission"
    return None


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary
    checks = dict(exp4556.check_preconditions(root))
    checkpoint = Path(root) / DEFAULT_CHECKPOINT_RELATIVE_PATH
    checks["trained_verifier_sha256"] = checkpoint_sha256(checkpoint)
    return checks


def _mode_priority(mode: str, *, root: Path) -> Any | None:  # pragma: no cover - live boundary
    if mode == "verifier_expansion":
        return load_cross_game_discriminative_expansion_priority(root=root)
    if mode == "random_priority":
        return RandomExpansionPriority(seed=RANDOM_SEED)
    return None


def make_variant_runner(  # pragma: no cover - ARC runtime boundary
    mode: str, *, root: Path | str = REPO_ROOT
) -> VariantRunner:
    """Run a manufactured variant with optional frontier-expansion priority."""

    root_path = Path(root)
    priority = _mode_priority(mode, root=root_path)

    def run(game: str, spec: Mapping[str, Any], budget: int) -> JsonDict:
        from arcengine import GameAction
        from carnot import experiment_4472_variant_generic_transfer_benchmark_v4 as v4
        from carnot.agentic import arc_solver_kit as kit
        from carnot.agentic.arc_competition_agent import CarnotAgentPolicy, _level_of
        from carnot.agentic.arc_variant_generator import VariantEnv

        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        env = VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))
        policy = CarnotAgentPolicy(
            game,
            {},
            force_explore=True,
            value_head=priority,
            value_weight=1.0 if priority is not None else 0.0,
            search_mode="best_first",
        )
        frames: list[Any] = []
        latest = None
        labels: list[str] = []
        actions = 0
        start_level: int | None = None
        reached = 0
        for _index in range(int(budget)):
            if policy.is_done(frames, latest):
                break
            kind, data = policy.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
                if labels:
                    labels.append("RESET")
            elif kind is None:
                break
            else:
                real_data = v4._remap_reflected_data(data, spec.get("reflect"))
                latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
                labels.append(v4._action_label(int(kind), real_data))
                actions += 1
            if start_level is None:
                start_level = _level_of(latest)
            frames.append(latest)
            reached = _level_of(latest)
            if start_level is not None and reached > start_level:
                break
            if latest is None:
                break
        claimed = reached if start_level is not None and reached > start_level else 0
        gate: JsonDict = {
            "game": game,
            "reached_level": 0,
            "claimed_level": claimed,
            "reproduced": False,
            "mode": "offline_reproduction_gate_no_solution",
        }
        if claimed > 0 and labels:
            gate = dict(kit.reproduce(game, labels, v4._apply_action_label, claimed_level=claimed))
        solved = bool(gate.get("reproduced")) and int(gate.get("reached_level") or 0) >= claimed >= 1
        graph_size = len(getattr(getattr(policy, "explorer", None), "graph", {}) or {})
        return {
            "game": game,
            "variant_signature": spec["variant_signature"],
            "variant": int(spec["variant"]),
            "kind": spec["kind"],
            "reflect": spec.get("reflect"),
            "attempted": True,
            "solved": solved,
            "winner_generated": solved,
            "reached_level": int(gate.get("reached_level") or reached) if solved else reached,
            "actions": actions,
            "expanded_states_to_goal": graph_size if solved else None,
            "expansions_used": graph_size,
            "max_expansions": int(budget),
            "solution_labels": labels if solved else [],
            "reproduction_gate": gate,
            "blocked_reason": "",
            "expansion_priority_mode": mode,
        }

    return run


def default_variant_runner_factory(mode: str) -> VariantRunner:  # pragma: no cover - live boundary
    return make_variant_runner(mode, root=REPO_ROOT)


def _solved_by_signature(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(attempt.get("variant_signature")): attempt
        for attempt in attempts
        if attempt.get("attempted") is True
    }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _median_attempt_field(attempts: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        float(attempt.get(field))
        for attempt in attempts
        if _attempt_solved(attempt) and attempt.get(field) is not None
    ]
    return _median(values)


def _paired_bootstrap_delta_ci(
    baseline_attempts: Sequence[Mapping[str, Any]],
    verifier_attempts: Sequence[Mapping[str, Any]],
    *,
    random_seed: int,
    n_bootstrap: int,
) -> list[float]:
    baseline = _solved_by_signature(baseline_attempts)
    verifier = _solved_by_signature(verifier_attempts)
    keys = sorted(set(baseline) & set(verifier))
    if not keys:
        return [0.0, 0.0]
    deltas = [
        (1.0 if _attempt_solved(verifier[key]) else 0.0)
        - (1.0 if _attempt_solved(baseline[key]) else 0.0)
        for key in keys
    ]
    if n_bootstrap <= 0:
        mean = sum(deltas) / len(deltas)
        return [round(mean, 10), round(mean, 10)]
    rng = random.Random(random_seed)
    samples = []
    n = len(deltas)
    for _index in range(int(n_bootstrap)):
        total = 0.0
        for _sample in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(lo), 10), round(float(hi), 10)]


def _checksum(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "generic_transfer_rate_with_expansion": artifact.get(
            "generic_transfer_rate_with_expansion"
        ),
        "generic_transfer_rate_baseline": artifact.get("generic_transfer_rate_baseline"),
        "transfer_delta": artifact.get("transfer_delta"),
        "transfer_ci": artifact.get("transfer_ci"),
        "expanded_states_to_goal_with_vs_without": artifact.get(
            "expanded_states_to_goal_with_vs_without"
        ),
        "winner_generated": artifact.get("winner_generated"),
        "expansions_used": artifact.get("expansions_used"),
        "variant_plan": artifact.get("variant_plan"),
        "preconditions_checked": artifact.get("preconditions_checked"),
    }


def _measurement(
    *,
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
    runner: VariantRunner,
) -> JsonDict:
    measured = exp4550.measure_generic_transfer_over_variants(
        public_games=public_games,
        variant_ids=variant_ids,
        budget=budget,
        variant_runner=runner,
    )
    attempts = measured["variant_attempts"]
    measured["median_expanded_states_to_goal"] = _median_attempt_field(
        attempts, "expanded_states_to_goal"
    )
    measured["median_actions_to_goal"] = _median_attempt_field(attempts, "actions")
    measured["max_expansions_used"] = max(
        [int(attempt.get("expansions_used") or 0) for attempt in attempts] or [0]
    )
    return measured


def _blocked_artifact(
    *,
    resource: str,
    preconditions: Mapping[str, Any],
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
    dominant_weight_family: str,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4569",
            "SCENARIO-CAPSTONE-4569",
            "SCENARIO-CAPSTONE-4569-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"complete: blocked_{resource}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "generic_transfer_rate_with_expansion": 0.0,
        "generic_transfer_rate_baseline": BASELINE_REFERENCE_RATE,
        "generic_transfer_baseline_reference": BASELINE_REFERENCE_RATE,
        "transfer_delta": 0.0,
        "transfer_ci": [0.0, 0.0],
        "expanded_states_to_goal_with_vs_without": {},
        "winner_generated": {"with_expansion": False, "generated_count": 0},
        "expansions_used": {"max_expansions": int(budget), "with_expansion": 0},
        "random_priority_control_passed": False,
        "false_negative_risk_checked": False,
        "null_delta_methodology_note": (
            "blocked before measurement; no verifier-guided expansion delta was fabricated."
        ),
        "solve_rate_preserved": False,
        "chosen_submitted_config": "unchanged",
        "missing_verifier_gaps": [
            f"blocked_{resource}; strongest_weight_family={dominant_weight_family}"
        ],
        "offline_reproduced": False,
        "preconditions_checked": dict(preconditions),
        "variant_plan": {
            "public_games": sorted(str(game) for game in public_games),
            "public_game_count": len(public_games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "max_expansions": int(budget),
        },
        "baseline_measurement": {},
        "verifier_expansion_measurement": {},
        "random_priority_measurement": {},
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def _newly_solved_reproduced(
    baseline_attempts: Sequence[Mapping[str, Any]],
    verifier_attempts: Sequence[Mapping[str, Any]],
) -> tuple[list[str], bool]:
    baseline_by_sig = _solved_by_signature(baseline_attempts)
    newly_solved = []
    newly_reproduced = []
    for attempt in verifier_attempts:
        signature = str(attempt.get("variant_signature"))
        if _attempt_solved(attempt) and not _attempt_solved(baseline_by_sig.get(signature, {})):
            newly_solved.append(signature)
            gate = attempt.get("reproduction_gate")
            newly_reproduced.append(isinstance(gate, Mapping) and gate.get("reproduced") is True)
    return sorted(newly_solved), (all(newly_reproduced) if newly_reproduced else True)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    preconditions_checked: Mapping[str, Any] | None = None,
    variant_runner_factory: VariantRunnerFactory = default_variant_runner_factory,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    dominant_weight_family: str | None = None,
) -> JsonDict:
    started = time.monotonic()
    root_path = Path(root)
    preconditions = dict(preconditions_checked or check_preconditions(root_path))
    games = list(public_games or preconditions.get("offline_env_public_games") or [])
    checkpoint = root_path / DEFAULT_CHECKPOINT_RELATIVE_PATH
    if dominant_weight_family is None:
        try:
            dominant_weight_family = dominant_feature_family_from_checkpoint(checkpoint)
        except Exception:
            dominant_weight_family = "unknown"

    miss = _first_precondition_miss(preconditions)
    if miss:
        return _blocked_artifact(
            resource=miss,
            preconditions=preconditions,
            public_games=games,
            variant_ids=variant_ids,
            budget=budget,
            dominant_weight_family=dominant_weight_family,
        )

    baseline = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("baseline"),
    )
    verifier = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("verifier_expansion"),
    )
    random_control = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("random_priority"),
    )

    baseline_rate = float(baseline["generic_transfer_rate_over_variants"])
    verifier_rate = float(verifier["generic_transfer_rate_over_variants"])
    random_rate = float(random_control["generic_transfer_rate_over_variants"])
    delta = round(verifier_rate - baseline_rate, 10)
    delta_ci = _paired_bootstrap_delta_ci(
        baseline["variant_attempts"],
        verifier["variant_attempts"],
        random_seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    solve_rate_preserved = verifier_rate >= baseline_rate
    verifier_expanded = verifier["median_expanded_states_to_goal"]
    baseline_expanded = baseline["median_expanded_states_to_goal"]
    random_expanded = random_control["median_expanded_states_to_goal"]
    strictly_lower = (
        verifier_expanded is not None
        and baseline_expanded is not None
        and float(verifier_expanded) < float(baseline_expanded)
    )
    beats_random_by_rate = verifier_rate > random_rate
    beats_random_by_expansions = (
        verifier_rate >= random_rate
        and verifier_expanded is not None
        and random_expanded is not None
        and float(verifier_expanded) < float(random_expanded)
    )
    random_priority_control_passed = bool(beats_random_by_rate or beats_random_by_expansions)

    newly_solved, offline_reproduced = _newly_solved_reproduced(
        baseline["variant_attempts"],
        verifier["variant_attempts"],
    )
    attempted = int(verifier["variant_attempts_count"])
    solved_count = int(verifier["variant_solved_count"])
    winner_generated = {
        "with_expansion": solved_count > 0,
        "without_expansion": int(baseline["variant_solved_count"]) > 0,
        "random_priority": int(random_control["variant_solved_count"]) > 0,
        "generated_count": solved_count,
        "attempted_count": attempted,
        "not_generated_count": max(0, attempted - solved_count),
        "newly_solved_variants": newly_solved,
    }
    expansion_summary = {
        "with_expansion_median": verifier_expanded,
        "without_expansion_median": baseline_expanded,
        "random_priority_median": random_expanded,
        "actions_to_goal_with_expansion_median": verifier["median_actions_to_goal"],
        "actions_to_goal_without_expansion_median": baseline["median_actions_to_goal"],
        "actions_to_goal_random_priority_median": random_control["median_actions_to_goal"],
        "strictly_lower_than_without": bool(strictly_lower),
    }
    transfer_win = (
        verifier_rate > BASELINE_REFERENCE_RATE
        and solve_rate_preserved
        and random_priority_control_passed
        and offline_reproduced
        and delta_ci[0] > 0.0
    )
    efficiency_win = (
        solve_rate_preserved
        and random_priority_control_passed
        and offline_reproduced
        and strictly_lower
    )
    wins = bool(transfer_win or efficiency_win)
    unresolved = max(0, attempted - solved_count)
    null_note = (
        "transfer_delta==0.0 is an honest no-value null under the paired same-variant "
        "measurement, not a control==best tautology."
        if delta == 0.0
        else ""
    )
    missing_gaps = (
        []
        if wins
        else [
            (
                "verifier_guided_expansion_no_value_added; "
                f"winner_not_generated_for={unresolved}; "
                f"strongest_weight_family={dominant_weight_family}"
            )
        ]
    )
    if transfer_win:
        verdict = f"success: verifier_guided_expansion_generic_transfer_{verifier_rate:.3f}_above_0.04"
    elif efficiency_win:
        verdict = "success: verifier_guided_expansion_expanded_states_lower_solve_rate_preserved"
    else:
        verdict = "complete: verifier_guided_expansion_no_value_honest_null_generation_gap_sharpened"

    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4569",
            "SCENARIO-CAPSTONE-4569",
            "SCENARIO-CAPSTONE-4569-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "generic_transfer_rate_with_expansion": verifier_rate,
        "generic_transfer_rate_baseline": baseline_rate,
        "generic_transfer_baseline_reference": BASELINE_REFERENCE_RATE,
        "transfer_delta": delta,
        "transfer_ci": delta_ci,
        "expanded_states_to_goal_with_vs_without": expansion_summary,
        "winner_generated": winner_generated,
        "expansions_used": {
            "max_expansions": int(budget),
            "with_expansion_max": int(verifier["max_expansions_used"]),
            "without_expansion_max": int(baseline["max_expansions_used"]),
            "random_priority_max": int(random_control["max_expansions_used"]),
        },
        "random_priority_transfer_rate": random_rate,
        "random_priority_control_passed": random_priority_control_passed,
        "false_negative_risk_checked": random_priority_control_passed,
        "null_delta_methodology_note": null_note,
        "solve_rate_preserved": solve_rate_preserved,
        "chosen_submitted_config": "enable_verifier_guided_expansion" if wins else "unchanged",
        "missing_verifier_gaps": missing_gaps,
        "dominant_verifier_weight_family": dominant_weight_family,
        "offline_reproduced": offline_reproduced,
        "preconditions_checked": preconditions,
        "variant_plan": {
            "public_games": sorted(str(game) for game in games),
            "public_game_count": len(games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "max_expansions": int(budget),
            "runner": "generic_solver_offline_variant_env",
            "arms": ["baseline", "verifier_expansion", "random_priority"],
        },
        "baseline_measurement": baseline,
        "verifier_expansion_measurement": verifier,
        "random_priority_measurement": random_control,
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.monotonic() - started, 6),
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    for field in (
        "generic_transfer_rate_with_expansion",
        "generic_transfer_rate_baseline",
        "transfer_delta",
    ):
        if not isinstance(artifact.get(field), float):
            errors.append(f"{field} must be a bare float")
    for field in (
        "random_priority_control_passed",
        "false_negative_risk_checked",
        "solve_rate_preserved",
        "offline_reproduced",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare bool")
    ci = artifact.get("transfer_ci")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or not all(isinstance(value, float) for value in ci)
    ):
        errors.append("transfer_ci must be [float, float]")
    if artifact.get("transfer_delta") == 0.0 and not artifact.get(
        "null_delta_methodology_note"
    ):
        errors.append("null_delta_methodology_note required for zero delta")
    for field in ("expanded_states_to_goal_with_vs_without", "winner_generated", "expansions_used"):
        if not isinstance(artifact.get(field), Mapping):
            errors.append(f"{field} must be a mapping")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles missing")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in principles:
                errors.append(f"missing field principle for {field}")
    return errors


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    artifact: Mapping[str, Any] | None = None,
) -> Path:
    root_path = Path(root)
    payload = dict(artifact or build_artifact(root_path))
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    path = root_path / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    root: Path | str = REPO_ROOT,
    *,
    write: bool = True,
) -> JsonDict:
    artifact = build_artifact(root)
    if write:
        write_artifact(root, artifact=artifact)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
