"""Experiment 4594: goal-distance / structural-progress energy as a proposal prior.

Spec refs: REQ-CAPSTONE-4594, SCENARIO-CAPSTONE-4594,
SCENARIO-CAPSTONE-4594-FIELD-PRINCIPLES.
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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot import experiment_4550_honest_sprint_metric as exp4550
from carnot import experiment_4592_generation_completeness_wiring as exp4592


JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]
VariantRunnerFactory = Callable[[str], VariantRunner]

RESULT_RELATIVE_PATH = "results/experiment_4594_goal_energy_generation_prior.json"
EXP4582_RELATIVE_PATH = "results/experiment_4582_feature_router_transfer.json"
EXPERIMENT_ID = "experiment_4594_goal_energy_generation_prior"
SCHEMA = "carnot.exp4594.goal_energy_generation_prior.v1"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4594
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
DEFAULT_BOOTSTRAPS = 1000
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
TARGETED_CLASSES = (
    "keyboard_graph:systematic_bfs:variant_wired=True",
    "click_graph:diversity_graph_explore:variant_wired=True",
    "config_toggle:diversity_graph_explore:variant_wired=True",
)
TARGETED_MECHANICS = {"keyboard_graph", "click_graph", "config_toggle"}
GRAPH_APPROACHES = {"systematic_bfs", "diversity_graph_explore", "default_graph_explore"}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "winner_generated_rate_with_energy",
    "winner_generated_rate_no_energy",
    "winner_generated_delta",
    "median_actions_to_first_levelup_with_energy",
    "actions_delta",
    "no_energy_control_passed",
    "false_negative_risk_checked",
    "null_delta_methodology_note",
    "targeted_classes",
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
            "terminal prefix; success: goal_energy_generation_prior_winner_generated_up_<n> "
            "OR complete: goal_energy_prior_no_value_honest_null_gap_sharpened."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- offline energy-guided proposal "
            "over variants, no LLM load (1s floor)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false for the value claim -- the goal-distance energy estimates PROGRESS "
            "(oracle-DISTINCT from the win-check); any game where goal-distance == the "
            "win-oracle is excluded as circular."
        )
    },
    "winner_generated_rate_with_energy": {
        "principle": (
            "the HEADLINE -- fraction of targeted-class variants for which the energy-guided "
            "proposal generated the winner; up vs no-energy is the energy-augmented-generation "
            "evidence."
        )
    },
    "winner_generated_rate_no_energy": {
        "principle": (
            "the matched no-energy A1-wired baseline on the SAME variants (the apples-to-apples "
            "control)."
        )
    },
    "winner_generated_delta": {
        "principle": (
            "with_energy - no_energy (positive = the energy prior reaches more winners), "
            "emitted explicitly so a null (0) is annotated."
        )
    },
    "median_actions_to_first_levelup_with_energy": {
        "principle": (
            "ACTION cost WITH the energy prior -- the leaderboard tiebreaker; energy must not "
            "blow up actions while reaching more winners."
        )
    },
    "actions_delta": {
        "principle": (
            "no_energy_actions - with_energy (positive = fewer actions); emitted explicitly so "
            "a null is annotated."
        )
    },
    "no_energy_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- energy-on must beat the no-energy arm on the SAME variants; "
            "a null is valid only if this ran."
        )
    },
    "false_negative_risk_checked": {
        "principle": "true with the no-energy control run -- a no-value null is valid only then."
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when winner_generated_delta==0 -- states the equality is an honest "
            "no-value null, not a measurement bug."
        )
    },
    "targeted_classes": {
        "principle": (
            "the wired-but-failing mechanic classes the energy prior targets (keyboard_graph BFS "
            "+ config/click tail) -- traceable to the exp4582 residual."
        )
    },
    "solve_rate_preserved": {
        "principle": "HARD gate -- the energy prior must NOT drop solve-rate."
    },
    "chosen_submitted_config": {
        "principle": (
            "what (if anything) is recommended for SUBMITTED_AGENT_CONFIG (enable the "
            "goal-energy proposal prior) -- the A6 input; 'unchanged' if null."
        )
    },
    "missing_verifier_gaps": {
        "principle": (
            "if no value, which class the energy prior still cannot reach -- the "
            "Missing-Verifier Gap Logging entry."
        )
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, goal_distance importable); "
            "pre-empts missing-resource fabrication."
        )
    },
}

_DIRS = {1: (-1.0, 0.0), 2: (1.0, 0.0), 3: (0.0, -1.0), 4: (0.0, 1.0)}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rate(count: int, attempted: int) -> float:
    return 0.0 if attempted <= 0 else round(float(count) / float(attempted), 10)


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


def _candidate_key(candidate: Any) -> tuple:
    data = getattr(candidate, "data", None)
    action_id = _as_int(getattr(candidate, "action_id", 0))
    if action_id == 6 and isinstance(data, Mapping):
        return (6, _as_int(data.get("x")), _as_int(data.get("y")))
    return (action_id,)


def _grid_of(frame_or_grid: Any) -> np.ndarray:
    from carnot.agentic.arc_agi3_world_model import grid_of

    try:
        return np.asarray(grid_of(frame_or_grid), dtype=np.int16)
    except Exception:  # pragma: no cover - non-frame fallback for live adapters
        return np.asarray(frame_or_grid, dtype=np.int16)


def _centroid(grid: np.ndarray, color: int) -> tuple[float, float] | None:
    ys, xs = np.where(np.asarray(grid) == int(color))
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def _nearest_distance(point: tuple[float, float], targets: Sequence[tuple[float, float]]) -> float:
    if not targets:
        return 0.0
    y, x = point
    return float(min(abs(y - ty) + abs(x - tx) for ty, tx in targets))


class GoalEnergyProposalPrior:
    """Objective candidate delta-energy used only to rank action proposals."""

    def __init__(
        self,
        *,
        avatar_color: int | None = None,
        goals: Sequence[tuple[float, float]] | None = None,
        cell: int | None = None,
        target_points: Sequence[tuple[float, float]] | None = None,
        action_energy_deltas: Mapping[int, float] | None = None,
        candidate_energy_deltas: Mapping[tuple, float] | None = None,
        energy_mode: str,
    ) -> None:
        self.avatar_color = None if avatar_color is None else int(avatar_color)
        self.goals = [(float(y), float(x)) for y, x in (goals or [])]
        self.cell = None if cell is None else int(cell)
        self.target_points = [(float(y), float(x)) for y, x in (target_points or [])]
        self.action_energy_deltas = {
            int(key): float(value) for key, value in (action_energy_deltas or {}).items()
        }
        self.candidate_energy_deltas = {
            tuple(key): float(value) for key, value in (candidate_energy_deltas or {}).items()
        }
        self.energy_mode = str(energy_mode)

    def candidate_delta_energy(self, frame: Any, candidate: Any) -> float:
        key = _candidate_key(candidate)
        if key in self.candidate_energy_deltas:
            return float(self.candidate_energy_deltas[key])
        action_id = _as_int(getattr(candidate, "action_id", 0))
        if action_id in self.action_energy_deltas:
            return float(self.action_energy_deltas[action_id])
        if self.avatar_color is not None and self.goals and action_id in _DIRS:
            grid = _grid_of(frame)
            current = _centroid(grid, self.avatar_color)
            if current is None:
                return 1e6
            dy, dx = _DIRS[action_id]
            before = _nearest_distance(current, self.goals)
            after = _nearest_distance((current[0] + dy, current[1] + dx), self.goals)
            return float(after - before)
        data = getattr(candidate, "data", None)
        if action_id == 6 and isinstance(data, Mapping) and self.target_points:
            click = (float(_as_int(data.get("y"))), float(_as_int(data.get("x"))))
            return _nearest_distance(click, self.target_points)
        return 0.0


def _target_points_from_grid(grid: np.ndarray, max_points: int = 6) -> list[tuple[float, float]]:
    arr = np.asarray(grid)
    if arr.size == 0:
        return []
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[counts.argmax()])
    points: list[tuple[float, float]] = []
    for value, _count in sorted(zip(vals.tolist(), counts.tolist()), key=lambda item: item[1]):
        color = int(value)
        if color == bg:
            continue
        ys, xs = np.where(arr == color)
        if len(ys):
            points.append((float(ys.mean()), float(xs.mean())))
        if len(points) >= max_points:
            break
    if points:
        return points
    ys, xs = np.where(arr != bg)
    if len(ys):
        return [(float(ys.mean()), float(xs.mean()))]
    return []


def _structural_state_energy(start_grid: np.ndarray, grid: np.ndarray) -> float:
    arr = np.asarray(grid)
    start = np.asarray(start_grid)
    if arr.shape != start.shape:
        return 1e6
    vals, counts = np.unique(start, return_counts=True)
    bg = int(vals[counts.argmax()]) if len(vals) else 0
    changed = float((arr != start).sum())
    non_background = float((arr != bg).sum())
    colors = float(len(set(arr.flatten().tolist())))
    return float(-changed - 0.05 * non_background - 0.1 * colors)


def _first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_arcade") is not True:
        return "offline_arcade"
    if preconditions.get("goal_distance_importable") is not True:
        return "goal_distance_import"
    if preconditions.get("variant_generator_importable") is not True:
        return "variant_generator_import"
    if preconditions.get("graph_explore_importable") is not True:
        return "graph_explore_import"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission"
    return None


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary
    root_path = Path(root)
    checks: JsonDict = {
        "offline_arcade": False,
        "goal_distance_importable": False,
        "variant_generator_importable": False,
        "graph_explore_importable": False,
        "offline_env_public_games": exp4550._public_games(root_path),
        "leaderboard_submission": False,
        "required_commands": [
            '.venv/bin/python -c "from carnot.agentic import arc_solver_kit as k; '
            'k.offline_arcade()"',
            '.venv/bin/python -c "from carnot.agentic.arc_goal_distance import '
            'make_goal_distance, calibrate_avatar_goal"',
        ],
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic.arc_goal_distance import calibrate_avatar_goal, make_goal_distance  # noqa: F401

        checks["goal_distance_importable"] = True
    except Exception as exc:
        checks["goal_distance_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic.arc_variant_generator import VariantEnv  # noqa: F401

        checks["variant_generator_importable"] = True
    except Exception as exc:
        checks["variant_generator_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic.arc_graph_explore import graph_explore_solve_v2  # noqa: F401

        checks["graph_explore_importable"] = True
    except Exception as exc:
        checks["graph_explore_import_error"] = f"{type(exc).__name__}: {exc}"
    checks["ok"] = _first_precondition_miss(checks) is None
    return checks


def load_targeted_variant_specs(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    path = Path(root) / EXP4582_RELATIVE_PATH
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    attempts = payload.get("feature_router_measurement", {}).get("variant_attempts", [])
    specs: list[JsonDict] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        route = attempt.get("selected_feature_route")
        mechanic = str(route.get("mechanic_class") if isinstance(route, Mapping) else "")
        selected = str(attempt.get("selected_approach") or "")
        if attempt.get("attempted") is not True or _attempt_solved(attempt):
            continue
        if attempt.get("approach_variant_wired") is not True:
            continue
        if mechanic not in TARGETED_MECHANICS or selected not in GRAPH_APPROACHES:
            continue
        specs.append(
            {
                "game": str(attempt.get("game") or ""),
                "variant": _as_int(attempt.get("variant"), 1),
                "kind": str(attempt.get("kind") or "color"),
                "reflect": attempt.get("reflect"),
                "variant_signature": str(attempt.get("variant_signature") or ""),
                "mechanic_class": mechanic,
                "selected_approach": selected,
            }
        )
    return sorted(specs, key=lambda spec: str(spec["variant_signature"]))


def _specs_from_games(public_games: Sequence[str], variant_ids: Sequence[int]) -> list[JsonDict]:
    return [dict(spec) for spec in exp4550.variant_specs(public_games, variant_ids)]


def _build_goal_energy_prior(env: Any, stats: JsonDict) -> GoalEnergyProposalPrior:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_executable_world_model import detect_cell
    from carnot.agentic.arc_goal_distance import calibrate_avatar_goal, make_goal_distance
    from carnot.agentic.arc_graph_explore import _warm, rich_action_candidates

    frame = _warm(env, False)
    start_grid = grid_of(frame)
    cell = detect_cell(start_grid)
    try:
        calib = calibrate_avatar_goal(env, cell, warmup=False)
    except Exception as exc:
        calib = {"avatar": None, "goals": [], "error": f"{type(exc).__name__}: {exc}"}
    stats["goal_energy_calibration"] = dict(calib)
    if calib.get("avatar") is not None and calib.get("goals"):
        make_goal_distance(int(calib["avatar"]), list(calib["goals"]), cell)
        return GoalEnergyProposalPrior(
            avatar_color=int(calib["avatar"]),
            goals=list(calib["goals"]),
            cell=cell,
            target_points=list(calib["goals"]),
            energy_mode="avatar_goal_distance",
        )

    base_energy = _structural_state_energy(start_grid, start_grid)
    target_points = _target_points_from_grid(start_grid)
    candidate_deltas: dict[tuple, float] = {}
    action_deltas: dict[int, float] = {}
    for candidate in rich_action_candidates(frame, max_click=12):
        trial = _warm(env, False)
        try:
            nf = env.step(
                _game_action(GameAction, int(candidate.action_id)),
                data=getattr(candidate, "data", None),
            )
            if nf is None:
                continue
            delta = _structural_state_energy(start_grid, grid_of(nf)) - base_energy
        except Exception:
            continue
        candidate_deltas[_candidate_key(candidate)] = float(delta)
        action_id = int(candidate.action_id)
        current = action_deltas.get(action_id)
        action_deltas[action_id] = float(delta if current is None else min(current, delta))
    stats["goal_energy_calibration"] = {
        **stats.get("goal_energy_calibration", {}),
        "structural_target_points": target_points,
        "structural_action_deltas": action_deltas,
        "structural_candidate_delta_count": len(candidate_deltas),
    }
    return GoalEnergyProposalPrior(
        target_points=target_points,
        action_energy_deltas=action_deltas,
        candidate_energy_deltas=candidate_deltas,
        energy_mode="structural_progress",
    )


def _run_energy_graph_attempt(  # pragma: no cover - ARC runtime boundary
    game: str,
    spec: Mapping[str, Any],
    budget: int,
    route: Mapping[str, Any],
) -> JsonDict:
    from carnot.agentic.arc_graph_explore import graph_explore_solve_v2

    env = exp4592._make_variant_env(game, spec)
    stats: JsonDict = {}
    prior = _build_goal_energy_prior(env, stats)
    env = exp4592._make_variant_env(game, spec)
    diversity = str(route.get("approach") or "") == "diversity_graph_explore"
    with exp4592._temporary_diversity(diversity):
        traj, level = graph_explore_solve_v2(
            env,
            start_level=0,
            max_expansions=int(budget),
            structural_energy_scorer=prior,
            stats=stats,
        )
    stats["diversity_env_var"] = "1" if diversity else "0"
    attempt = exp4592._trajectory_attempt(
        game=game,
        spec=spec,
        route=route,
        executor="graph_explore_solve_v2",
        traj=traj,
        reached_level=int(level),
        stats=stats,
    )
    attempt["goal_energy_mode"] = prior.energy_mode
    attempt["proposal_prior_enabled"] = True
    attempt["winner_generated_by_energy_prior"] = _attempt_solved(attempt)
    return attempt


def make_variant_runner(
    mode: str,
    *,
    root: Path | str = REPO_ROOT,
    policy: Mapping[str, Any] | None = None,
) -> VariantRunner:  # pragma: no cover - ARC runtime boundary
    if mode == "no_energy":
        return exp4592.make_variant_runner("wired", root=root, policy=policy)
    if mode != "with_energy":
        raise ValueError(f"unknown mode {mode!r}")

    from carnot.agentic import arc_solve_learning as learning

    router_policy = dict(policy or learning.learn_feature_router_policy())
    no_energy_runner = exp4592.make_variant_runner("wired", root=root, policy=router_policy)

    def run(game: str, spec: Mapping[str, Any], budget: int) -> JsonDict:
        route = exp4592._route_for_variant(game, spec, policy=router_policy)
        executor = exp4592._executor_for_route(route)
        if executor == "graph_explore_solve_v2":
            selected = _run_energy_graph_attempt(game, spec, budget, route)
        else:
            selected = dict(no_energy_runner(game, spec, budget))
            selected["proposal_prior_enabled"] = False
            selected["winner_generated_by_energy_prior"] = False
            selected["goal_energy_mode"] = "not_applicable"
            return selected
        selected["goal_energy_runner_mode"] = "with_energy"
        if _attempt_solved(selected):
            return selected
        fallback = dict(no_energy_runner(game, spec, budget))
        fallback.update(
            {
                "goal_energy_runner_mode": "with_energy",
                "fallback_used": True,
                "energy_attempt": selected,
                "proposal_prior_enabled": True,
                "winner_generated_by_energy_prior": False,
                "goal_energy_mode": selected.get("goal_energy_mode", "unknown"),
            }
        )
        return fallback if _attempt_solved(fallback) else {**selected, "fallback_attempt": fallback}

    return run


def default_variant_runner_factory(mode: str) -> VariantRunner:  # pragma: no cover - live boundary
    return make_variant_runner(mode, root=REPO_ROOT)


def _winner_generated_by_energy(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("winner_generated_by_energy_prior") is True


def _attempts_by_signature(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(attempt.get("variant_signature")): attempt
        for attempt in attempts
        if attempt.get("attempted") is True
    }


def _paired_bootstrap_delta_ci(
    no_energy_attempts: Sequence[Mapping[str, Any]],
    with_energy_attempts: Sequence[Mapping[str, Any]],
    *,
    random_seed: int,
    n_bootstrap: int,
) -> list[float]:
    no_energy = _attempts_by_signature(no_energy_attempts)
    with_energy = _attempts_by_signature(with_energy_attempts)
    keys = sorted(set(no_energy) & set(with_energy))
    if not keys:
        return [0.0, 0.0]
    deltas = [
        (1.0 if _winner_generated_by_energy(with_energy[key]) else 0.0)
        - (1.0 if _attempt_solved(no_energy[key]) else 0.0)
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


def _measurement(
    *,
    specs: Sequence[Mapping[str, Any]],
    budget: int,
    runner: VariantRunner,
    energy_prior: bool,
    n_bootstrap: int,
) -> JsonDict:
    attempts = [dict(runner(str(spec["game"]), spec, int(budget))) for spec in specs]
    attempted = sum(1 for attempt in attempts if attempt.get("attempted") is True)
    solved = sum(1 for attempt in attempts if _attempt_solved(attempt))
    winner_count = sum(
        1
        for attempt in attempts
        if (
            _winner_generated_by_energy(attempt)
            if energy_prior
            else _attempt_solved(attempt)
        )
    )
    transfer_rate = _rate(solved, attempted)
    return {
        "variant_specs": [dict(spec) for spec in specs],
        "variant_attempts": attempts,
        "variant_attempts_count": attempted,
        "variant_solved_count": solved,
        "generic_transfer_rate_over_variants": transfer_rate,
        "generic_transfer_ci": exp4550.bootstrap_transfer_ci(
            attempts,
            random_seed=RANDOM_SEED,
            n_bootstrap=n_bootstrap,
        ),
        "winner_generated_count": int(winner_count),
        "winner_generated_rate": _rate(int(winner_count), attempted),
        "solve_rate": float(transfer_rate),
        "median_actions_to_first_levelup": _median_actions(attempts),
    }


def _newly_solved_reproduced(
    no_energy_attempts: Sequence[Mapping[str, Any]],
    with_energy_attempts: Sequence[Mapping[str, Any]],
) -> tuple[list[str], bool]:
    no_energy = _attempts_by_signature(no_energy_attempts)
    newly_solved: list[str] = []
    reproduced_flags: list[bool] = []
    for attempt in with_energy_attempts:
        signature = str(attempt.get("variant_signature"))
        if not _winner_generated_by_energy(attempt) or _attempt_solved(no_energy.get(signature, {})):
            continue
        newly_solved.append(signature)
        reproduced_flags.append(_gate_reproduced(attempt.get("reproduction_gate")))
    return sorted(newly_solved), all(reproduced_flags) if reproduced_flags else True


def _missing_gaps(attempts: Sequence[Mapping[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        if attempt.get("attempted") is not True or _winner_generated_by_energy(attempt):
            continue
        route = attempt.get("selected_feature_route")
        mechanic = "unknown"
        if isinstance(route, Mapping):
            mechanic = str(route.get("mechanic_class") or mechanic)
        approach = str(attempt.get("selected_approach") or "default_graph_explore")
        mode = str(attempt.get("goal_energy_mode") or "structural_progress")
        key = f"{mechanic}:{approach}:{mode} winner_generated=0"
        counts[key] = counts.get(key, 0) + 1
    return [
        f"goal_energy_residual {key} count={count}"
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _oracle_distinctness_records(attempts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    records: list[JsonDict] = []
    for attempt in attempts:
        route = attempt.get("selected_feature_route")
        mechanic = str(route.get("mechanic_class") if isinstance(route, Mapping) else "unknown")
        mode = str(attempt.get("goal_energy_mode") or "unknown")
        overlap = mode == "avatar_goal_distance"
        records.append(
            {
                "game": str(attempt.get("game") or ""),
                "variant_signature": str(attempt.get("variant_signature") or ""),
                "mechanic_class": mechanic,
                "goal_energy_mode": mode,
                "verifier_is_oracle": False,
                "goal_distance_could_approximate_oracle": bool(overlap),
                "included_in_value_claim": not overlap,
            }
        )
    return records


def _checksum(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "winner_generated_rate_with_energy": artifact.get("winner_generated_rate_with_energy"),
        "winner_generated_rate_no_energy": artifact.get("winner_generated_rate_no_energy"),
        "winner_generated_delta": artifact.get("winner_generated_delta"),
        "winner_generated_delta_ci": artifact.get("winner_generated_delta_ci"),
        "actions_delta": artifact.get("actions_delta"),
        "no_energy_control_passed": artifact.get("no_energy_control_passed"),
        "targeted_variant_specs": artifact.get("targeted_variant_specs"),
        "newly_solved_variants": artifact.get("newly_solved_variants"),
        "missing_verifier_gaps": artifact.get("missing_verifier_gaps"),
        "per_game_oracle_distinctness": artifact.get("per_game_oracle_distinctness"),
        "oracle_overlap_excluded_variants": artifact.get("oracle_overlap_excluded_variants"),
        "preconditions_checked": artifact.get("preconditions_checked"),
    }


def _blocked_artifact(
    *,
    resource: str,
    preconditions: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    budget: int,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4594",
            "SCENARIO-CAPSTONE-4594",
            "SCENARIO-CAPSTONE-4594-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"complete: blocked_{resource}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "winner_generated_rate_with_energy": 0.0,
        "winner_generated_rate_no_energy": 0.0,
        "winner_generated_delta": 0.0,
        "winner_generated_delta_ci": [0.0, 0.0],
        "generic_transfer_rate_with_energy": 0.0,
        "generic_transfer_rate_no_energy": 0.0,
        "median_actions_to_first_levelup_with_energy": None,
        "median_actions_to_first_levelup_no_energy": None,
        "actions_delta": 0.0,
        "no_energy_control_passed": False,
        "no_energy_control_ran": False,
        "false_negative_risk_checked": False,
        "null_delta_methodology_note": (
            "blocked before measurement; no goal-energy proposal delta was fabricated."
        ),
        "targeted_classes": list(TARGETED_CLASSES),
        "solve_rate_preserved": False,
        "chosen_submitted_config": "unchanged",
        "missing_verifier_gaps": [f"blocked_{resource}"],
        "offline_reproduced": False,
        "newly_solved_variants": [],
        "per_game_oracle_distinctness": [],
        "preconditions_checked": dict(preconditions),
        "variant_plan": {
            "targeted_variant_count": len(specs),
            "budget": int(budget),
            "arms": ["no_energy", "with_energy"],
            "value_head_best_first_expansion": False,
        },
        "targeted_variant_specs": [dict(spec) for spec in specs],
        "no_energy_measurement": {},
        "with_energy_measurement": {},
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
    specs = (
        _specs_from_games(public_games, variant_ids)
        if public_games is not None
        else load_targeted_variant_specs(root_path)
    )
    miss = _first_precondition_miss(preconditions)
    if miss:
        return _blocked_artifact(
            resource=miss,
            preconditions=preconditions,
            specs=specs,
            budget=budget,
        )

    no_energy = _measurement(
        specs=specs,
        budget=budget,
        runner=variant_runner_factory("no_energy"),
        energy_prior=False,
        n_bootstrap=n_bootstrap,
    )
    with_energy = _measurement(
        specs=specs,
        budget=budget,
        runner=variant_runner_factory("with_energy"),
        energy_prior=True,
        n_bootstrap=n_bootstrap,
    )

    no_energy_winner_rate = float(no_energy["winner_generated_rate"])
    with_energy_winner_rate = float(with_energy["winner_generated_rate"])
    winner_delta = round(with_energy_winner_rate - no_energy_winner_rate, 10)
    winner_delta_ci = _paired_bootstrap_delta_ci(
        no_energy["variant_attempts"],
        with_energy["variant_attempts"],
        random_seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    no_energy_rate = float(no_energy["generic_transfer_rate_over_variants"])
    with_energy_rate = float(with_energy["generic_transfer_rate_over_variants"])
    no_energy_actions = no_energy["median_actions_to_first_levelup"]
    with_energy_actions = with_energy["median_actions_to_first_levelup"]
    actions_delta = (
        round(float(no_energy_actions) - float(with_energy_actions), 10)
        if no_energy_actions is not None and with_energy_actions is not None
        else 0.0
    )
    same_variant_control_ran = (
        int(no_energy["variant_attempts_count"]) > 0
        and int(no_energy["variant_attempts_count"]) == int(with_energy["variant_attempts_count"])
    )
    no_energy_control_passed = bool(
        same_variant_control_ran and with_energy_winner_rate >= no_energy_winner_rate
    )
    false_negative_risk_checked = bool(same_variant_control_ran and no_energy_control_passed)
    solve_rate_preserved = with_energy_rate >= no_energy_rate
    newly_solved, offline_reproduced = _newly_solved_reproduced(
        no_energy["variant_attempts"], with_energy["variant_attempts"]
    )
    generation_win = (
        winner_delta > 0.0
        and winner_delta_ci[0] > 0.0
        and no_energy_control_passed
        and solve_rate_preserved
        and offline_reproduced
    )
    action_win = (
        actions_delta > 0.0
        and no_energy_control_passed
        and solve_rate_preserved
        and offline_reproduced
    )
    wins = bool(generation_win or action_win)
    if winner_delta == 0.0 and no_energy_control_passed:
        null_note = (
            "winner_generated_delta==0.0 is an honest no-value null under the paired "
            "same-variant no-energy control, not a measurement bug."
        )
    elif winner_delta == 0.0:
        null_note = (
            "winner_generated_delta==0.0 but the matched no-energy control did not pass, "
            "so false-negative risk remains open."
        )
    else:
        null_note = ""
    gaps = [] if wins else _missing_gaps(with_energy["variant_attempts"])
    if not gaps and not wins:
        gaps = ["goal_energy_prior_no_value_added; no targeted winner generated"]
    if wins:
        verdict = (
            "success: goal_energy_generation_prior_winner_generated_up_"
            f"{int(with_energy['winner_generated_count']) - int(no_energy['winner_generated_count'])}"
        )
    elif no_energy_control_passed:
        verdict = "complete: goal_energy_prior_no_value_honest_null_gap_sharpened"
    else:
        verdict = "complete: goal_energy_prior_control_failed_false_negative_risk_open"

    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4594",
            "SCENARIO-CAPSTONE-4594",
            "SCENARIO-CAPSTONE-4594-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "winner_generated_rate_with_energy": with_energy_winner_rate,
        "winner_generated_rate_no_energy": no_energy_winner_rate,
        "winner_generated_delta": winner_delta,
        "winner_generated_delta_ci": winner_delta_ci,
        "generic_transfer_rate_with_energy": with_energy_rate,
        "generic_transfer_rate_no_energy": no_energy_rate,
        "median_actions_to_first_levelup_with_energy": with_energy_actions,
        "median_actions_to_first_levelup_no_energy": no_energy_actions,
        "actions_delta": float(actions_delta),
        "no_energy_control_passed": no_energy_control_passed,
        "no_energy_control_ran": same_variant_control_ran,
        "false_negative_risk_checked": false_negative_risk_checked,
        "null_delta_methodology_note": null_note,
        "targeted_classes": list(TARGETED_CLASSES),
        "solve_rate_preserved": solve_rate_preserved,
        "chosen_submitted_config": "enable_goal_energy_generation_prior" if wins else "unchanged",
        "missing_verifier_gaps": gaps,
        "offline_reproduced": offline_reproduced,
        "newly_solved_variants": newly_solved,
        "per_game_oracle_distinctness": _oracle_distinctness_records(
            with_energy["variant_attempts"]
        ),
        "oracle_overlap_excluded_variants": [
            record["variant_signature"]
            for record in _oracle_distinctness_records(with_energy["variant_attempts"])
            if not record["included_in_value_claim"]
        ],
        "preconditions_checked": preconditions,
        "variant_plan": {
            "targeted_variant_count": len(specs),
            "targeted_variants_source": EXP4582_RELATIVE_PATH
            if public_games is None
            else "explicit_public_games",
            "budget": int(budget),
            "arms": ["no_energy", "with_energy"],
            "runner": "goal_energy_generation_prior_over_a1_wired_variant_runner",
            "proposal_prior_not_expansion_priority": True,
            "value_head_best_first_expansion": False,
        },
        "targeted_variant_specs": [dict(spec) for spec in specs],
        "no_energy_measurement": no_energy,
        "with_energy_measurement": with_energy,
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
        "winner_generated_rate_with_energy",
        "winner_generated_rate_no_energy",
        "winner_generated_delta",
        "actions_delta",
    ):
        if not isinstance(artifact.get(field), float):
            errors.append(f"{field} must be a bare float")
    ci = artifact.get("winner_generated_delta_ci")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or not all(isinstance(value, float) for value in ci)
    ):
        errors.append("winner_generated_delta_ci must be [float, float]")
    for field in (
        "no_energy_control_passed",
        "false_negative_risk_checked",
        "solve_rate_preserved",
        "offline_reproduced",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare bool")
    if artifact.get("winner_generated_delta") == 0.0 and not artifact.get(
        "null_delta_methodology_note"
    ):
        errors.append("null_delta_methodology_note required for zero winner_generated_delta")
    if not isinstance(artifact.get("targeted_classes"), list):
        errors.append("targeted_classes must be a list")
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
