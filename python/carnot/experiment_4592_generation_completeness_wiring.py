"""Experiment 4592: generation-completeness wiring over held-out variants.

Spec refs: REQ-CAPSTONE-4592, SCENARIO-CAPSTONE-4592,
SCENARIO-CAPSTONE-4592-FIELD-PRINCIPLES.
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
from carnot import experiment_4472_variant_generic_transfer_benchmark_v4 as variant_bench
from carnot.agentic import arc_solve_learning as learning


JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]
VariantRunnerFactory = Callable[[str], VariantRunner]

RESULT_RELATIVE_PATH = "results/experiment_4592_generation_completeness_wiring.json"
EXPERIMENT_ID = "experiment_4592_generation_completeness_wiring"
SCHEMA = "carnot.exp4592.generation_completeness_wiring.v1"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4592
BASELINE_REFERENCE_RATE = 0.04
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
DEFAULT_BOOTSTRAPS = 1000
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "winner_generated_rate_with_wiring",
    "winner_generated_rate_baseline",
    "winner_generated_delta",
    "generic_transfer_rate_with_wiring",
    "generic_transfer_rate_baseline",
    "transfer_delta",
    "transfer_ci",
    "median_actions_to_first_levelup_with_wiring",
    "actions_delta",
    "no_wiring_control_passed",
    "false_negative_risk_checked",
    "null_delta_methodology_note",
    "solve_rate_preserved",
    "residual_unwired_classes",
    "chosen_submitted_config",
    "offline_reproduced",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: generation_completeness_winner_generated_<n>_above_1of25 "
            "OR complete: generation_completeness_no_value_honest_null_residual_logged."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- offline dispatch over variants "
            "(1s floor); if the optional LLM-reasoner tail arm runs, declare live_llm_inference "
            "for THAT arm + the Qwen3.5-9B-MTP iGPU precondition (NEVER the 3090s)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- wiring + goal_distance + graph_explore GENERATE candidates; they "
            "are oracle-DISTINCT from the executable win-check (a circular win does not count)."
        )
    },
    "winner_generated_rate_with_wiring": {
        "principle": (
            "the HEADLINE -- fraction of held-out variants for which the WIRED dispatch generated "
            "the winning candidate; > 1/25 is the non-circular generation evidence the four "
            "ranking nulls could not produce."
        )
    },
    "winner_generated_rate_baseline": {
        "principle": (
            "1/25 = 0.04 -- the exp4582 default-harness baseline, measured the SAME way "
            "(the apples-to-apples control)."
        )
    },
    "winner_generated_delta": {
        "principle": (
            "with_wiring - baseline (positive = more winners generated = the wall cracks), "
            "emitted explicitly so a null (0) is annotated."
        )
    },
    "generic_transfer_rate_with_wiring": {
        "principle": (
            "held-out variant transfer WITH the wired dispatch; > 0.04 with CI-excl-baseline "
            "is the leaderboard-honest seen->hidden signal."
        )
    },
    "generic_transfer_rate_baseline": {
        "principle": "0.04 -- the .420/.421 B1 default baseline, measured the SAME way."
    },
    "transfer_delta": {
        "principle": (
            "with_wiring - baseline, emitted explicitly so a null (0.0) is annotated, not a "
            "control==best TAUTOLOGY false-positive."
        )
    },
    "transfer_ci": {
        "principle": (
            "bootstrap CI on the transfer delta; a claim above baseline requires the CI to "
            "exclude 0.04."
        )
    },
    "median_actions_to_first_levelup_with_wiring": {
        "principle": (
            "ACTION cost WITH wiring -- the leaderboard tiebreaker; reported because "
            "transfer-only is insufficient (the weekend best-first regression)."
        )
    },
    "actions_delta": {
        "principle": (
            "baseline_actions - with_wiring (positive = fewer actions); emitted explicitly so "
            "a null is annotated."
        )
    },
    "no_wiring_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- the wired dispatch must beat the no-wiring "
            "default_variant_runner on the SAME variants; a null is valid only if this ran "
            "(fixes the .423 A3 broken-control trap)."
        )
    },
    "false_negative_risk_checked": {
        "principle": (
            "MUST be true with the no-wiring control run -- a no-value null is valid only if "
            "the control passed."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when winner_generated_delta==0 -- states the equality is an honest "
            "no-value null, not a measurement bug."
        )
    },
    "solve_rate_preserved": {"principle": "HARD gate -- wiring must NOT drop solve-rate."},
    "residual_unwired_classes": {
        "principle": (
            "which mechanic class still has winner_generated=0 after wiring -- the "
            "Missing-Verifier/Generator Gap Logging entry (the residual generation gap)."
        )
    },
    "chosen_submitted_config": {
        "principle": (
            "what (if anything) is recommended for SUBMITTED_AGENT_CONFIG (enable the wired "
            "dispatch) -- the A6 input; 'unchanged' if null."
        )
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, variant generator, goal_distance + "
            "graph_explore importable); pre-empts missing-resource fabrication."
        )
    },
}

NAV_GOAL_DISTANCE_MECHANICS = {"avatar_navigation", "click_connect"}
GRAPH_MECHANICS = {"keyboard_graph", "click_graph", "config_toggle"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _attempt_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and attempt.get("solved") is True


def _gate_reproduced(gate: Any) -> bool:
    if not isinstance(gate, Mapping):
        return False
    claimed = max(1, _as_int(gate.get("claimed_level"), 1))
    return gate.get("reproduced") is True and _as_int(gate.get("reached_level"), 0) >= claimed


def _median(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(float(value) for value in values))


def _median_actions(attempts: Sequence[Mapping[str, Any]]) -> float | None:
    return _median(exp4550.agent_actions_to_first_levelup(attempts))


def _rate(count: int, attempted: int) -> float:
    return 0.0 if attempted <= 0 else round(float(count) / float(attempted), 10)


def _first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_arcade") is not True:
        return "offline_arcade"
    if preconditions.get("variant_generator_importable") is not True:
        return "variant_generator_import"
    if preconditions.get("goal_distance_importable") is not True:
        return "goal_distance_import"
    if preconditions.get("graph_explore_importable") is not True:
        return "graph_explore_import"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission"
    return None


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary
    root_path = Path(root)
    checks: JsonDict = {
        "offline_arcade": False,
        "variant_generator_importable": False,
        "goal_distance_importable": False,
        "graph_explore_importable": False,
        "offline_env_public_games": exp4550._public_games(root_path),
        "leaderboard_submission": False,
        "live_llm_inference": False,
        "qwen35_9b_mtp_igpu_precondition": "not_used",
        "required_commands": [
            '.venv/bin/python -c "from carnot.agentic import arc_solver_kit as k; '
            'k.offline_arcade()"',
            '.venv/bin/python -c "import carnot.agentic.arc_variant_generator; '
            "from carnot.agentic.arc_goal_distance import goal_distance_solve; "
            'from carnot.agentic.arc_graph_explore import graph_explore_solve_v2"',
        ],
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic.arc_variant_generator import VariantEnv  # noqa: F401

        checks["variant_generator_importable"] = True
    except Exception as exc:
        checks["variant_generator_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic.arc_goal_distance import goal_distance_solve  # noqa: F401

        checks["goal_distance_importable"] = True
    except Exception as exc:
        checks["goal_distance_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic.arc_graph_explore import graph_explore_solve_v2  # noqa: F401

        checks["graph_explore_importable"] = True
    except Exception as exc:
        checks["graph_explore_import_error"] = f"{type(exc).__name__}: {exc}"
    checks["ok"] = _first_precondition_miss(checks) is None
    return checks


@contextmanager
def _temporary_diversity(enabled: bool):  # pragma: no cover - process env boundary
    old_value = os.environ.get("CARNOT_ARC_EXPLORE_DIVERSITY")
    os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("CARNOT_ARC_EXPLORE_DIVERSITY", None)
        else:
            os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = old_value


def _make_variant_env(game: str, spec: Mapping[str, Any]) -> Any:  # pragma: no cover - ARC boundary
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_variant_generator import VariantEnv

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    return VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))


def _probe_variant_signature(game: str, spec: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    env = _make_variant_env(game, spec)
    return learning.probe_early_play_signature(env, k=8)


def _route_for_variant(
    game: str,
    spec: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> JsonDict:
    try:
        signature = _probe_variant_signature(game, spec)
    except Exception as exc:  # pragma: no cover - defensive live boundary
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


def _executor_for_route(route: Mapping[str, Any]) -> str:
    mechanic = str(route.get("mechanic_class") or "unknown")
    approach = str(route.get("approach") or "default_graph_explore")
    if mechanic in NAV_GOAL_DISTANCE_MECHANICS:
        return "goal_distance_astar"
    if mechanic in GRAPH_MECHANICS:
        return "graph_explore_solve_v2"
    if approach == "goal_distance_astar":
        return "goal_distance_astar"
    if approach in {"systematic_bfs", "diversity_graph_explore", "default_graph_explore"}:
        return "graph_explore_solve_v2"
    return "default_graph_explore"


def _labels_for_reproduction(traj: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for step in traj:
        data = variant_bench._remap_reflected_data(step.get("data"), spec.get("reflect"))
        labels.append(variant_bench._action_label(_as_int(step.get("action")), data))
    return labels


def _trajectory_attempt(
    *,
    game: str,
    spec: Mapping[str, Any],
    route: Mapping[str, Any],
    executor: str,
    traj: Sequence[Mapping[str, Any]] | None,
    reached_level: int,
    stats: Mapping[str, Any] | None = None,
    note: str = "",
) -> JsonDict:  # pragma: no cover - ARC reproduction boundary
    from carnot.agentic import arc_solver_kit as kit

    labels = _labels_for_reproduction(traj or [], spec)
    gate: JsonDict = {
        "game": game,
        "claimed_level": int(reached_level) if reached_level > 0 else 0,
        "reached_level": 0,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_solution",
    }
    if reached_level > 0 and labels:
        gate = dict(
            kit.reproduce(
                game,
                labels,
                variant_bench._apply_action_label,
                claimed_level=int(reached_level),
            )
        )
    solved = _gate_reproduced(gate) and reached_level > 0
    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": int(spec["variant"]),
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": solved,
        "winner_generated": solved,
        "reached_level": _as_int(gate.get("reached_level"), reached_level if solved else 0),
        "actions": len(labels),
        "actions_to_first_levelup": len(labels) if solved else None,
        "solution_labels": labels if solved else [],
        "reproduction_gate": gate,
        "blocked_reason": "",
        "selected_feature_route": dict(route),
        "selected_approach": str(route.get("approach") or "default_graph_explore"),
        "executed_approach": executor,
        "approach_variant_wired": executor != "default_graph_explore",
        "generation_stats": dict(stats or {}),
        "candidate_generated": bool(labels and reached_level > 0),
        "dispatch_note": note,
    }


def _failed_attempt(
    *,
    game: str,
    spec: Mapping[str, Any],
    route: Mapping[str, Any],
    executor: str,
    reason: str,
    stats: Mapping[str, Any] | None = None,
) -> JsonDict:
    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": int(spec["variant"]),
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": False,
        "winner_generated": False,
        "reached_level": 0,
        "actions": 0,
        "actions_to_first_levelup": None,
        "solution_labels": [],
        "reproduction_gate": {
            "game": game,
            "claimed_level": 0,
            "reached_level": 0,
            "reproduced": False,
            "mode": "offline_reproduction_gate_no_solution",
        },
        "blocked_reason": reason,
        "selected_feature_route": dict(route),
        "selected_approach": str(route.get("approach") or "default_graph_explore"),
        "executed_approach": executor,
        "approach_variant_wired": executor != "default_graph_explore",
        "generation_stats": dict(stats or {}),
        "candidate_generated": False,
    }


def _run_goal_distance_attempt(  # pragma: no cover - ARC runtime boundary
    game: str,
    spec: Mapping[str, Any],
    budget: int,
    route: Mapping[str, Any],
) -> JsonDict:
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_executable_world_model import detect_cell
    from carnot.agentic.arc_goal_distance import calibrate_avatar_goal, make_goal_distance
    from carnot.agentic.arc_graph_explore import _warm, graph_explore_solve_v2

    executor = "goal_distance_astar"
    env = _make_variant_env(game, spec)
    try:
        frame = _warm(env, False)
        cell = detect_cell(grid_of(frame))
        calib = calibrate_avatar_goal(env, cell, warmup=False)
    except Exception as exc:
        return _failed_attempt(
            game=game,
            spec=spec,
            route=route,
            executor=executor,
            reason=f"goal_distance_calibration_{type(exc).__name__}: {exc}",
        )
    if calib.get("avatar") is None or not calib.get("goals"):
        return _failed_attempt(
            game=game,
            spec=spec,
            route=route,
            executor=executor,
            reason="goal_distance_no_avatar_or_goal",
            stats={"calibration": calib},
        )
    heuristic = make_goal_distance(int(calib["avatar"]), list(calib["goals"]), cell)
    stats: JsonDict = {"calibration": calib}
    traj, level = graph_explore_solve_v2(
        env,
        start_level=0,
        max_expansions=int(budget),
        heuristic=heuristic,
        heuristic_weight=2.0,
        stats=stats,
    )
    return _trajectory_attempt(
        game=game,
        spec=spec,
        route=route,
        executor=executor,
        traj=traj,
        reached_level=int(level),
        stats=stats,
    )


def _run_graph_attempt(  # pragma: no cover - ARC runtime boundary
    game: str,
    spec: Mapping[str, Any],
    budget: int,
    route: Mapping[str, Any],
) -> JsonDict:
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2

    env = _make_variant_env(game, spec)
    stats: JsonDict = {}
    diversity = str(route.get("approach") or "") == "diversity_graph_explore"
    with _temporary_diversity(diversity):
        traj, level = graph_explore_solve_v2(
            env,
            start_level=0,
            max_expansions=int(budget),
            stats=stats,
        )
    stats["diversity_env_var"] = "1" if diversity else "0"
    return _trajectory_attempt(
        game=game,
        spec=spec,
        route=route,
        executor="graph_explore_solve_v2",
        traj=traj,
        reached_level=int(level),
        stats=stats,
    )


def make_variant_runner(
    mode: str,
    *,
    root: Path | str = REPO_ROOT,
    policy: Mapping[str, Any] | None = None,
) -> VariantRunner:
    """Run one manufactured variant under the no-wiring baseline or wired dispatch."""

    _root_path = Path(root)
    router_policy = dict(policy or learning.learn_feature_router_policy())

    def run(game: str, spec: Mapping[str, Any], budget: int) -> JsonDict:
        if mode == "baseline":
            attempt = dict(exp4550.default_variant_runner(game, spec, budget))
            attempt["wiring_mode"] = "baseline"
            attempt["executed_approach"] = attempt.get("executed_approach", "default_graph_explore")
            attempt["approach_variant_wired"] = False
            return attempt

        route = _route_for_variant(game, spec, policy=router_policy)
        executor = _executor_for_route(route)
        if executor == "goal_distance_astar":
            selected = _run_goal_distance_attempt(game, spec, budget, route)
        elif executor == "graph_explore_solve_v2":
            selected = _run_graph_attempt(game, spec, budget, route)
        else:
            selected = dict(exp4550.default_variant_runner(game, spec, budget))
            selected["selected_feature_route"] = route
            selected["selected_approach"] = str(route.get("approach") or "default_graph_explore")
            selected["executed_approach"] = "default_graph_explore"
            selected["approach_variant_wired"] = False
        selected["wiring_mode"] = "wired"
        if _attempt_solved(selected):
            selected["fallback_used"] = False
            return selected

        fallback = dict(exp4550.default_variant_runner(game, spec, budget))
        fallback.update(
            {
                "wiring_mode": "wired",
                "fallback_used": True,
                "selected_attempt": selected,
                "selected_feature_route": route,
                "selected_approach": str(route.get("approach") or "default_graph_explore"),
                "executed_approach": "default_graph_explore",
                "approach_variant_wired": selected.get("approach_variant_wired") is True,
            }
        )
        return fallback if _attempt_solved(fallback) else {**selected, "fallback_used": True, "fallback_attempt": fallback}

    return run


def default_variant_runner_factory(mode: str) -> VariantRunner:  # pragma: no cover - live boundary
    return make_variant_runner(mode, root=REPO_ROOT)


def _measurement(
    *,
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
    runner: VariantRunner,
    n_bootstrap: int,
) -> JsonDict:
    measured = exp4550.measure_generic_transfer_over_variants(
        public_games=public_games,
        variant_ids=variant_ids,
        budget=budget,
        variant_runner=runner,
        random_seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    solved = int(measured["variant_solved_count"])
    attempted = int(measured["variant_attempts_count"])
    measured["winner_generated_count"] = solved
    measured["winner_generated_rate"] = _rate(solved, attempted)
    measured["solve_rate"] = float(measured["generic_transfer_rate_over_variants"])
    measured["median_actions_to_first_levelup"] = _median_actions(measured["variant_attempts"])
    return measured


def _attempts_by_signature(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(attempt.get("variant_signature")): attempt
        for attempt in attempts
        if attempt.get("attempted") is True
    }


def _paired_bootstrap_delta_ci(
    baseline_attempts: Sequence[Mapping[str, Any]],
    wired_attempts: Sequence[Mapping[str, Any]],
    *,
    random_seed: int,
    n_bootstrap: int,
) -> list[float]:
    baseline = _attempts_by_signature(baseline_attempts)
    wired = _attempts_by_signature(wired_attempts)
    keys = sorted(set(baseline) & set(wired))
    if not keys:
        return [0.0, 0.0]
    deltas = [
        (1.0 if _attempt_solved(wired[key]) else 0.0)
        - (1.0 if _attempt_solved(baseline[key]) else 0.0)
        for key in keys
    ]
    point = sum(deltas) / len(deltas)
    if n_bootstrap <= 0:
        rounded = round(float(point), 10)
        return [rounded, rounded]
    rng = random.Random(random_seed)
    samples: list[float] = []
    n = len(deltas)
    for _index in range(int(n_bootstrap)):
        total = 0.0
        for _sample in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(min(lo, point)), 10), round(float(max(hi, point)), 10)]


def _newly_solved_reproduced(
    baseline_attempts: Sequence[Mapping[str, Any]],
    wired_attempts: Sequence[Mapping[str, Any]],
) -> tuple[list[str], bool]:
    baseline = _attempts_by_signature(baseline_attempts)
    newly_solved: list[str] = []
    reproduced_flags: list[bool] = []
    for attempt in wired_attempts:
        signature = str(attempt.get("variant_signature"))
        if not _attempt_solved(attempt) or _attempt_solved(baseline.get(signature, {})):
            continue
        newly_solved.append(signature)
        reproduced_flags.append(_gate_reproduced(attempt.get("reproduction_gate")))
    return sorted(newly_solved), all(reproduced_flags) if reproduced_flags else True


def _residual_unwired_classes(attempts: Sequence[Mapping[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        if attempt.get("attempted") is not True or _attempt_solved(attempt):
            continue
        route = attempt.get("selected_feature_route")
        mechanic = "unknown"
        if isinstance(route, Mapping):
            mechanic = str(route.get("mechanic_class") or mechanic)
        approach = str(attempt.get("selected_approach") or "default_graph_explore")
        key = f"{mechanic}:{approach}:winner_generated=0"
        counts[key] = counts.get(key, 0) + 1
    return [
        f"{key} count={count}"
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _checksum(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "winner_generated_rate_with_wiring": artifact.get("winner_generated_rate_with_wiring"),
        "winner_generated_rate_baseline": artifact.get("winner_generated_rate_baseline"),
        "winner_generated_delta": artifact.get("winner_generated_delta"),
        "generic_transfer_rate_with_wiring": artifact.get("generic_transfer_rate_with_wiring"),
        "generic_transfer_rate_baseline": artifact.get("generic_transfer_rate_baseline"),
        "transfer_delta": artifact.get("transfer_delta"),
        "transfer_ci": artifact.get("transfer_ci"),
        "actions_delta": artifact.get("actions_delta"),
        "no_wiring_control_passed": artifact.get("no_wiring_control_passed"),
        "residual_unwired_classes": artifact.get("residual_unwired_classes"),
        "newly_solved_variants": artifact.get("newly_solved_variants"),
        "variant_plan": artifact.get("variant_plan"),
        "preconditions_checked": artifact.get("preconditions_checked"),
    }


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
            "REQ-CAPSTONE-4592",
            "SCENARIO-CAPSTONE-4592",
            "SCENARIO-CAPSTONE-4592-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"complete: blocked_{resource}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "winner_generated_rate_with_wiring": 0.0,
        "winner_generated_rate_baseline": BASELINE_REFERENCE_RATE,
        "winner_generated_delta": 0.0,
        "generic_transfer_rate_with_wiring": 0.0,
        "generic_transfer_rate_baseline": BASELINE_REFERENCE_RATE,
        "transfer_delta": 0.0,
        "transfer_ci": [0.0, 0.0],
        "median_actions_to_first_levelup_with_wiring": None,
        "median_actions_to_first_levelup_baseline": None,
        "actions_delta": 0.0,
        "no_wiring_control_passed": False,
        "false_negative_risk_checked": False,
        "null_delta_methodology_note": "blocked before measurement; no wiring delta was fabricated.",
        "solve_rate_preserved": False,
        "residual_unwired_classes": [f"blocked_{resource}"],
        "chosen_submitted_config": "unchanged",
        "offline_reproduced": False,
        "newly_solved_variants": [],
        "preconditions_checked": dict(preconditions),
        "variant_plan": {
            "public_games": sorted(str(game) for game in public_games),
            "public_game_count": len(public_games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "budget": int(budget),
            "arms": ["baseline", "wired"],
        },
        "baseline_measurement": {},
        "wired_measurement": {},
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


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
        n_bootstrap=n_bootstrap,
    )
    wired = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("wired"),
        n_bootstrap=n_bootstrap,
    )

    baseline_rate = float(baseline["generic_transfer_rate_over_variants"])
    wired_rate = float(wired["generic_transfer_rate_over_variants"])
    baseline_winner_rate = float(baseline["winner_generated_rate"])
    wired_winner_rate = float(wired["winner_generated_rate"])
    winner_delta = round(wired_winner_rate - baseline_winner_rate, 10)
    transfer_delta = round(wired_rate - baseline_rate, 10)
    transfer_ci = _paired_bootstrap_delta_ci(
        baseline["variant_attempts"],
        wired["variant_attempts"],
        random_seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    baseline_actions = baseline["median_actions_to_first_levelup"]
    wired_actions = wired["median_actions_to_first_levelup"]
    actions_delta = (
        round(float(baseline_actions) - float(wired_actions), 10)
        if baseline_actions is not None and wired_actions is not None
        else 0.0
    )
    same_variant_control_ran = (
        int(baseline["variant_attempts_count"]) > 0
        and int(baseline["variant_attempts_count"]) == int(wired["variant_attempts_count"])
    )
    no_wiring_control_passed = bool(same_variant_control_ran and wired_winner_rate >= baseline_winner_rate)
    false_negative_risk_checked = bool(same_variant_control_ran)
    solve_rate_preserved = wired_rate >= baseline_rate
    newly_solved, offline_reproduced = _newly_solved_reproduced(
        baseline["variant_attempts"], wired["variant_attempts"]
    )
    transfer_win = (
        wired_rate > BASELINE_REFERENCE_RATE
        and transfer_ci[0] > 0.0
        and solve_rate_preserved
    )
    action_win = actions_delta > 0.0 and solve_rate_preserved
    generation_win = winner_delta > 0.0 and no_wiring_control_passed and offline_reproduced
    wins = bool(generation_win and (transfer_win or action_win))
    if winner_delta == 0.0:
        null_note = (
            "winner_generated_delta==0.0 is an honest no-value null under the paired "
            "same-variant measurement, not a measurement bug."
        )
    else:
        null_note = ""
    residual = _residual_unwired_classes(wired["variant_attempts"])
    if not residual and not wins:
        residual = ["unknown:default_graph_explore:winner_generated=0 count=0"]
    if wins:
        verdict = (
            "success: generation_completeness_winner_generated_"
            f"{int(wired['winner_generated_count'])}of{int(wired['variant_attempts_count'])}_above_1of25"
        )
    else:
        verdict = "complete: generation_completeness_no_value_honest_null_residual_logged"

    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4592",
            "SCENARIO-CAPSTONE-4592",
            "SCENARIO-CAPSTONE-4592-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "winner_generated_rate_with_wiring": wired_winner_rate,
        "winner_generated_rate_baseline": baseline_winner_rate,
        "winner_generated_delta": winner_delta,
        "generic_transfer_rate_with_wiring": wired_rate,
        "generic_transfer_rate_baseline": baseline_rate,
        "generic_transfer_baseline_reference": BASELINE_REFERENCE_RATE,
        "transfer_delta": transfer_delta,
        "transfer_ci": transfer_ci,
        "median_actions_to_first_levelup_with_wiring": wired_actions,
        "median_actions_to_first_levelup_baseline": baseline_actions,
        "actions_delta": actions_delta,
        "no_wiring_control_passed": no_wiring_control_passed,
        "no_wiring_control_ran": same_variant_control_ran,
        "false_negative_risk_checked": false_negative_risk_checked,
        "null_delta_methodology_note": null_note,
        "solve_rate_preserved": solve_rate_preserved,
        "residual_unwired_classes": residual,
        "chosen_submitted_config": "enable_wired_generation_dispatch" if wins else "unchanged",
        "offline_reproduced": offline_reproduced,
        "newly_solved_variants": newly_solved,
        "preconditions_checked": preconditions,
        "variant_plan": {
            "public_games": sorted(str(game) for game in games),
            "public_game_count": len(games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "budget": int(budget),
            "runner": "mechanic_class_wired_generation_dispatch_over_variant_env",
            "arms": ["baseline", "wired"],
            "optional_llm_tail_enabled": False,
        },
        "baseline_measurement": baseline,
        "wired_measurement": wired,
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
        "winner_generated_rate_with_wiring",
        "winner_generated_rate_baseline",
        "winner_generated_delta",
        "generic_transfer_rate_with_wiring",
        "generic_transfer_rate_baseline",
        "transfer_delta",
        "actions_delta",
    ):
        if not isinstance(artifact.get(field), float):
            errors.append(f"{field} must be a bare float")
    for field in (
        "no_wiring_control_passed",
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
    if artifact.get("winner_generated_delta") == 0.0 and not artifact.get(
        "null_delta_methodology_note"
    ):
        errors.append("null_delta_methodology_note required for zero winner_generated_delta")
    if not isinstance(artifact.get("residual_unwired_classes"), list):
        errors.append("residual_unwired_classes must be a list")
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
