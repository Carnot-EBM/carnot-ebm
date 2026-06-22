"""Experiment 4582: early-play mechanic feature router transfer.

Spec refs: REQ-CAPSTONE-4582, SCENARIO-CAPSTONE-4582,
SCENARIO-CAPSTONE-4582-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import os
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
from carnot.agentic import arc_solve_learning as learning


JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]
VariantRunnerFactory = Callable[[str], VariantRunner]

RESULT_RELATIVE_PATH = "results/experiment_4582_feature_router_transfer.json"
EXPERIMENT_ID = "experiment_4582_feature_router_transfer"
SCHEMA = "carnot.exp4582.feature_router_transfer.v1"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4582
BASELINE_REFERENCE_RATE = 0.04
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
DEFAULT_BOOTSTRAPS = 1000
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "generic_transfer_rate_with_router",
    "generic_transfer_rate_baseline",
    "transfer_delta",
    "transfer_ci",
    "median_actions_to_first_levelup_with_router",
    "actions_delta",
    "winner_generated",
    "random_route_control_passed",
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
            "terminal prefix; success: feature_router_generic_transfer_<n>_above_0.04 OR "
            "complete: feature_router_no_value_honest_null_transfer_gap_sharpened."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- the router classifies + routes over "
            "offline variants, no LLM load (1s floor)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the router selects an approach by a LEARNED early-play signal, "
            "oracle-DISTINCT from the executable win-check."
        )
    },
    "generic_transfer_rate_with_router": {
        "principle": (
            "the HEADLINE -- held-out variant transfer WITH the feature-router; > 0.04 with "
            "CI-excl-baseline is the non-circular live-value evidence."
        )
    },
    "generic_transfer_rate_baseline": {
        "principle": "0.04 -- the .420/.421 B1 default-approach baseline, measured the SAME way."
    },
    "transfer_delta": {
        "principle": "with_router - baseline, emitted explicitly so a null is annotated."
    },
    "transfer_ci": {
        "principle": (
            "bootstrap CI on the transfer delta; a claim above baseline requires the CI to exclude "
            "the baseline."
        )
    },
    "median_actions_to_first_levelup_with_router": {
        "principle": "ACTION cost WITH routing -- the leaderboard tiebreaker."
    },
    "actions_delta": {
        "principle": "baseline_actions - with_router (positive = fewer actions)."
    },
    "winner_generated": {
        "principle": "did routing to the right approach GENERATE the winning candidate."
    },
    "random_route_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- the router must beat random-route on the SAME variants."
        )
    },
    "false_negative_risk_checked": {
        "principle": "MUST be true with the random-route control run."
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when transfer_delta==0.0 -- states the equality is an honest no-value null, "
            "not a measurement bug."
        )
    },
    "solve_rate_preserved": {"principle": "HARD gate -- routing must NOT drop solve-rate."},
    "chosen_submitted_config": {
        "principle": (
            "what (if anything) is recommended for SUBMITTED_AGENT_CONFIG (enable the feature-router) "
            "-- the A6 input; 'unchanged' if null."
        )
    },
    "missing_verifier_gaps": {
        "principle": "if no value, the residual generation gap by mechanic class."
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent router/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, recommend_approach importable); "
            "pre-empts missing-resource fabrication."
        )
    },
}

_VARIANT_WIRED_APPROACHES = {
    "default_graph_explore",
    "systematic_bfs",
    "diversity_graph_explore",
}


def _attempt_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and attempt.get("solved") is True


def _median(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(float(value) for value in values))


def _median_actions(attempts: Sequence[Mapping[str, Any]]) -> float | None:
    return _median(exp4550.agent_actions_to_first_levelup(attempts))


def _first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_arcade") is not True:
        return "offline_arcade"
    if preconditions.get("recommend_approach_importable") is not True:
        return "recommend_approach_import"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission"
    return None


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary
    root_path = Path(root)
    checks: JsonDict = {
        "offline_arcade": False,
        "recommend_approach_importable": False,
        "offline_env_public_games": exp4550._public_games(root_path),
        "leaderboard_submission": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic.arc_solve_learning import recommend_approach  # noqa: F401

        checks["recommend_approach_importable"] = True
    except Exception as exc:
        checks["recommend_approach_import_error"] = f"{type(exc).__name__}: {exc}"
    checks["ok"] = _first_precondition_miss(checks) is None
    return checks


@contextmanager
def _temporary_diversity(enabled: bool):  # pragma: no cover - process env boundary
    old_value = os.environ.get("CARNOT_ARC_EXPLORE_DIVERSITY")
    if enabled:
        os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = "1"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("CARNOT_ARC_EXPLORE_DIVERSITY", None)
        else:
            os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = old_value


def _probe_variant_signature(
    game: str, spec: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - ARC runtime boundary
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_variant_generator import VariantEnv

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env = VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))
    return learning.probe_early_play_signature(env, k=8)


def _random_route(game: str, spec: Mapping[str, Any]) -> JsonDict:
    choices = sorted(learning.FEATURE_ROUTER_APPROACHES)
    rng = random.Random(f"{RANDOM_SEED}:{game}:{spec.get('variant_signature')}")
    approach = choices[rng.randrange(len(choices))]
    return {
        "enabled": True,
        "mechanic_class": "random_route_control",
        "approach": approach,
        "confidence": 0.0,
        "policy_source": "deterministic_random_control",
        "signature": {},
        "approach_descriptor": dict(learning.FEATURE_ROUTER_APPROACHES[approach]),
        "verifier_is_oracle": False,
        "no_regression_fallback": "arc_graph_explore.graph_explore_solve_v2",
    }


def _route_for_variant(
    mode: str,
    game: str,
    spec: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> JsonDict:
    if mode == "random_route":
        return _random_route(game, spec)
    if mode != "feature_router":
        return {
            "enabled": False,
            "mechanic_class": "baseline_default",
            "approach": "default_graph_explore",
            "confidence": 0.0,
            "policy_source": "baseline",
            "signature": {},
            "approach_descriptor": dict(
                learning.FEATURE_ROUTER_APPROACHES["default_graph_explore"]
            ),
            "verifier_is_oracle": False,
            "no_regression_fallback": "arc_graph_explore.graph_explore_solve_v2",
        }
    try:
        signature = _probe_variant_signature(game, spec)
    except Exception as exc:
        signature = {
            "probe_count": 0,
            "keyboard_effect_count": 0,
            "click_effect_count": 0,
            "avatar_motion_present": False,
            "cell_connect": False,
            "hidden_carry_state": False,
            "config_toggle": False,
            "probe_error": f"{type(exc).__name__}: {exc}",
            "llm_used": False,
        }
    return learning.route_feature_approach(signature, policy=policy)


def make_variant_runner(  # pragma: no cover - ARC runtime boundary
    mode: str, *, root: Path | str = REPO_ROOT
) -> VariantRunner:
    """Run one manufactured variant under baseline, feature-router, or random-route mode."""

    _root_path = Path(root)
    policy = learning.learn_feature_router_policy()

    def run(game: str, spec: Mapping[str, Any], budget: int) -> JsonDict:
        route = _route_for_variant(mode, game, spec, policy=policy)
        selected = str(route.get("approach") or "default_graph_explore")
        executed = selected if selected in _VARIANT_WIRED_APPROACHES else "default_graph_explore"
        with _temporary_diversity(executed == "diversity_graph_explore"):
            attempt = dict(exp4550.default_variant_runner(game, spec, budget))
        attempt["feature_router_mode"] = mode
        attempt["selected_feature_route"] = route
        attempt["selected_approach"] = selected
        attempt["executed_approach"] = executed
        attempt["approach_variant_wired"] = selected in _VARIANT_WIRED_APPROACHES
        return attempt

    return run


def default_variant_runner_factory(mode: str) -> VariantRunner:  # pragma: no cover - live boundary
    return make_variant_runner(mode, root=REPO_ROOT)


def _solved_by_signature(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(attempt.get("variant_signature")): attempt
        for attempt in attempts
        if attempt.get("attempted") is True
    }


def _paired_bootstrap_delta_ci(
    baseline_attempts: Sequence[Mapping[str, Any]],
    router_attempts: Sequence[Mapping[str, Any]],
    *,
    random_seed: int,
    n_bootstrap: int,
) -> list[float]:
    baseline = _solved_by_signature(baseline_attempts)
    router = _solved_by_signature(router_attempts)
    keys = sorted(set(baseline) & set(router))
    if not keys:
        return [0.0, 0.0]
    deltas = [
        (1.0 if _attempt_solved(router[key]) else 0.0)
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
        "generic_transfer_rate_with_router": artifact.get("generic_transfer_rate_with_router"),
        "generic_transfer_rate_baseline": artifact.get("generic_transfer_rate_baseline"),
        "transfer_delta": artifact.get("transfer_delta"),
        "transfer_ci": artifact.get("transfer_ci"),
        "actions_delta": artifact.get("actions_delta"),
        "winner_generated": artifact.get("winner_generated"),
        "variant_plan": artifact.get("variant_plan"),
        "newly_solved_variants": artifact.get("newly_solved_variants"),
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
    measured["median_actions_to_first_levelup"] = _median_actions(measured["variant_attempts"])
    return measured


def _blocked_artifact(
    *,
    resource: str,
    preconditions: Mapping[str, Any],
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4582",
            "SCENARIO-CAPSTONE-4582",
            "SCENARIO-CAPSTONE-4582-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"complete: blocked_{resource}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "generic_transfer_rate_with_router": 0.0,
        "generic_transfer_rate_baseline": BASELINE_REFERENCE_RATE,
        "generic_transfer_baseline_reference": BASELINE_REFERENCE_RATE,
        "transfer_delta": 0.0,
        "transfer_ci": [0.0, 0.0],
        "median_actions_to_first_levelup_with_router": None,
        "median_actions_to_first_levelup_baseline": None,
        "actions_delta": 0.0,
        "winner_generated": {"with_router": False, "generated_count": 0},
        "random_route_transfer_rate": 0.0,
        "random_route_control_passed": False,
        "false_negative_risk_checked": False,
        "null_delta_methodology_note": (
            "blocked before measurement; no feature-router delta was fabricated."
        ),
        "solve_rate_preserved": False,
        "chosen_submitted_config": "unchanged",
        "missing_verifier_gaps": [f"blocked_{resource}"],
        "offline_reproduced": False,
        "newly_solved_variants": [],
        "preconditions_checked": dict(preconditions),
        "variant_plan": {
            "public_games": sorted(str(game) for game in public_games),
            "public_game_count": len(public_games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "budget": int(budget),
        },
        "baseline_measurement": {},
        "feature_router_measurement": {},
        "random_route_measurement": {},
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def _newly_solved_reproduced(
    baseline_attempts: Sequence[Mapping[str, Any]],
    router_attempts: Sequence[Mapping[str, Any]],
) -> tuple[list[str], bool]:
    baseline_by_sig = _solved_by_signature(baseline_attempts)
    newly_solved = []
    newly_reproduced = []
    for attempt in router_attempts:
        signature = str(attempt.get("variant_signature"))
        if _attempt_solved(attempt) and not _attempt_solved(baseline_by_sig.get(signature, {})):
            newly_solved.append(signature)
            gate = attempt.get("reproduction_gate")
            newly_reproduced.append(isinstance(gate, Mapping) and gate.get("reproduced") is True)
    return sorted(newly_solved), (all(newly_reproduced) if newly_reproduced else True)


def _dominant_route_gaps(attempts: Sequence[Mapping[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        if _attempt_solved(attempt):
            continue
        route = attempt.get("selected_feature_route")
        mechanic = "unknown"
        approach = str(attempt.get("selected_approach") or "default_graph_explore")
        wired = attempt.get("approach_variant_wired")
        if isinstance(route, Mapping):
            mechanic = str(route.get("mechanic_class") or mechanic)
        key = f"{mechanic}:{approach}:variant_wired={bool(wired)}"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return []
    return [
        f"feature_router_residual_generation_gap {key} unsolved_count={count}"
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    preconditions_checked: Mapping[str, Any] | None = None,
    variant_runner_factory: VariantRunnerFactory = default_variant_runner_factory,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
) -> JsonDict:
    started = time.monotonic()
    root_path = Path(root)
    preconditions = dict(preconditions_checked or check_preconditions(root_path))
    games = list(public_games or preconditions.get("offline_env_public_games") or [])
    miss = _first_precondition_miss(preconditions)
    if miss:
        return _blocked_artifact(
            resource=miss,
            preconditions=preconditions,
            public_games=games,
            variant_ids=variant_ids,
            budget=budget,
        )

    baseline = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("baseline"),
    )
    router = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("feature_router"),
    )
    random_control = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("random_route"),
    )

    baseline_rate = float(baseline["generic_transfer_rate_over_variants"])
    router_rate = float(router["generic_transfer_rate_over_variants"])
    random_rate = float(random_control["generic_transfer_rate_over_variants"])
    delta = round(router_rate - baseline_rate, 10)
    delta_ci = _paired_bootstrap_delta_ci(
        baseline["variant_attempts"],
        router["variant_attempts"],
        random_seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    baseline_actions = baseline["median_actions_to_first_levelup"]
    router_actions = router["median_actions_to_first_levelup"]
    random_actions = random_control["median_actions_to_first_levelup"]
    actions_delta = (
        round(float(baseline_actions) - float(router_actions), 10)
        if baseline_actions is not None and router_actions is not None
        else 0.0
    )
    solve_rate_preserved = router_rate >= baseline_rate
    beats_random_by_rate = router_rate > random_rate
    beats_random_by_actions = (
        router_rate >= random_rate
        and router_actions is not None
        and random_actions is not None
        and float(router_actions) < float(random_actions)
    )
    random_route_control_passed = bool(beats_random_by_rate or beats_random_by_actions)
    newly_solved, offline_reproduced = _newly_solved_reproduced(
        baseline["variant_attempts"], router["variant_attempts"]
    )
    attempted = int(router["variant_attempts_count"])
    solved_count = int(router["variant_solved_count"])
    transfer_win = (
        router_rate > BASELINE_REFERENCE_RATE
        and solve_rate_preserved
        and random_route_control_passed
        and offline_reproduced
        and delta_ci[0] > 0.0
    )
    action_win = (
        actions_delta > 0.0
        and solve_rate_preserved
        and random_route_control_passed
        and offline_reproduced
    )
    wins = bool(transfer_win or action_win)
    if delta == 0.0 and random_route_control_passed:
        null_note = (
            "transfer_delta==0.0 is an honest no-value null under the paired same-variant "
            "measurement, not a measurement bug."
        )
    elif delta == 0.0:
        null_note = (
            "transfer_delta==0.0 under the paired same-variant measurement, but "
            "random_route_control_passed=false, so the no-value null is not closed."
        )
    else:
        null_note = ""
    gaps = [] if wins else _dominant_route_gaps(router["variant_attempts"])
    if not gaps and not wins:
        gaps = ["feature_router_no_value_added; no newly generated winning variant"]
    if transfer_win:
        verdict = f"success: feature_router_generic_transfer_{router_rate:.3f}_above_0.04"
    elif action_win:
        verdict = "success: feature_router_actions_to_first_levelup_lower_solve_rate_preserved"
    elif random_route_control_passed:
        verdict = "complete: feature_router_no_value_honest_null_transfer_gap_sharpened"
    else:
        verdict = "complete: feature_router_no_value_control_failed_false_negative_risk_open"

    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4582",
            "SCENARIO-CAPSTONE-4582",
            "SCENARIO-CAPSTONE-4582-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "generic_transfer_rate_with_router": router_rate,
        "generic_transfer_rate_baseline": baseline_rate,
        "generic_transfer_baseline_reference": BASELINE_REFERENCE_RATE,
        "transfer_delta": delta,
        "transfer_ci": delta_ci,
        "median_actions_to_first_levelup_with_router": router_actions,
        "median_actions_to_first_levelup_baseline": baseline_actions,
        "median_actions_to_first_levelup_random_route": random_actions,
        "actions_delta": actions_delta,
        "winner_generated": {
            "with_router": solved_count > 0,
            "without_router": int(baseline["variant_solved_count"]) > 0,
            "random_route": int(random_control["variant_solved_count"]) > 0,
            "generated_count": solved_count,
            "attempted_count": attempted,
            "not_generated_count": max(0, attempted - solved_count),
            "newly_solved_variants": newly_solved,
        },
        "random_route_transfer_rate": random_rate,
        "random_route_control_passed": random_route_control_passed,
        "false_negative_risk_checked": True,
        "null_delta_methodology_note": null_note,
        "solve_rate_preserved": solve_rate_preserved,
        "chosen_submitted_config": "enable_feature_router" if wins else "unchanged",
        "missing_verifier_gaps": gaps,
        "offline_reproduced": offline_reproduced,
        "newly_solved_variants": newly_solved,
        "preconditions_checked": preconditions,
        "variant_plan": {
            "public_games": sorted(str(game) for game in games),
            "public_game_count": len(games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "budget": int(budget),
            "runner": "feature_router_over_generic_solver_offline_variant_env",
            "arms": ["baseline", "feature_router", "random_route"],
            "value_head_best_first_expansion": False,
        },
        "feature_router_policy": learning.learn_feature_router_policy(),
        "baseline_measurement": baseline,
        "feature_router_measurement": router,
        "random_route_measurement": random_control,
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
        "generic_transfer_rate_with_router",
        "generic_transfer_rate_baseline",
        "transfer_delta",
        "actions_delta",
    ):
        if not isinstance(artifact.get(field), float):
            errors.append(f"{field} must be a bare float")
    for field in (
        "random_route_control_passed",
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
        errors.append("null_delta_methodology_note required for zero transfer_delta")
    if not isinstance(artifact.get("winner_generated"), Mapping):
        errors.append("winner_generated must be a mapping")
    if not isinstance(artifact.get("missing_verifier_gaps"), list):
        errors.append("missing_verifier_gaps must be a list")
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
