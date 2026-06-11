"""Bounded heuristic search over a verified ARC-AGI-3 world model.

Spec refs: REQ-PHASE4-030, SCENARIO-PHASE4-030.

The planner here does not learn dynamics.  It asks a caller-provided verifier
world model to expand legal next states, ranks those states with a coded
distance-to-goal proxy, and returns a plan only when the supplied goal predicate
fires.  Keeping transition expansion outside this module lets Exp 4021 use the
real offline environment copy as the simulator while unit tests use a tiny
deterministic graph.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "new_levels_solved_this_task",
    "wall_was_search_not_representation",
    "nodes_expanded",
    "heuristic_used",
    "real_env_confirmed",
    "inference_substrate",
)

TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")
DEFAULT_HEURISTIC_NAME = "coded_unmet_targets_plus_manhattan_progress"
DEFAULT_OUTPUT_ARTIFACT = Path("results/experiment_4021_heuristic_search_over_verified_wm.json")


@dataclass(frozen=True)
class SearchResult:
    """Planner output with enough provenance to avoid fabricating a solve."""

    solved: bool
    actions: list[Any]
    nodes_expanded: int
    final_state: Any
    bottleneck: str
    max_expansions: int


NextStates = Callable[[Any], Iterable[tuple[Any, Any]]]
GoalPredicate = Callable[[Any], bool]
Heuristic = Callable[[Any], float]


def coded_goal_distance_heuristic(state: Any) -> float:
    """Return an OOD-stable coded distance proxy for ARC wall search.

    The strongest signal is the number of unsatisfied goal components because
    Exp 4020's goal predicate is defined around that field.  Manhattan distance
    and remaining progress are smaller tie-breakers, so a state that satisfies
    one target outranks a state that merely moves close to a target without
    reducing unmet components.
    """

    if not isinstance(state, dict):
        return 0.0
    unmet = float(state.get("unsatisfied_targets", state.get("unmet_goal_components", 0)) or 0)
    manhattan = float(state.get("manhattan_to_target", state.get("remaining_distance", 0)) or 0)
    progress_gap = float(state.get("progress_bar_delta_to_goal", 0) or 0)
    return unmet * 1000.0 + manhattan + max(0.0, progress_gap)


def _state_key(state: Any) -> str:
    try:
        return json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(state)


def best_first_search(
    start_state: Any,
    *,
    next_states: NextStates,
    is_goal: GoalPredicate,
    heuristic: Heuristic = coded_goal_distance_heuristic,
    max_expansions: int = 50000,
) -> SearchResult:
    """Run bounded best-first search over verifier-expanded successor states.

    The expansion bound is hard.  If no goal is found, the returned bottleneck
    states whether the frontier was exhausted or the bound stopped the search.
    The caller must still execute any plan in the real environment before
    claiming a solved level.
    """

    max_expansions = max(0, int(max_expansions))
    if is_goal(start_state):
        return SearchResult(True, [], 0, start_state, "", max_expansions)

    counter = 0
    start_key = _state_key(start_state)
    best_cost: dict[str, int] = {start_key: 0}
    frontier: list[tuple[float, int, int, Any, list[Any]]] = [
        (float(heuristic(start_state)), 0, counter, start_state, [])
    ]
    nodes_expanded = 0
    last_state = start_state

    while frontier and nodes_expanded < max_expansions:
        _, cost, _, state, actions = heapq.heappop(frontier)
        key = _state_key(state)
        if cost != best_cost.get(key):  # pragma: no cover - defensive if queue entries become stale
            continue
        last_state = state
        if is_goal(state):
            return SearchResult(True, actions, nodes_expanded, state, "", max_expansions)

        nodes_expanded += 1
        for action, child in next_states(state):
            child_cost = cost + 1
            child_key = _state_key(child)
            if child_cost >= best_cost.get(child_key, 1_000_000_000):
                continue
            best_cost[child_key] = child_cost
            counter += 1
            priority = child_cost + float(heuristic(child))
            heapq.heappush(frontier, (priority, child_cost, counter, child, actions + [action]))

    bottleneck = "expansion_bound_exhausted" if frontier else "frontier_exhausted"
    return SearchResult(False, [], nodes_expanded, last_state, bottleneck, max_expansions)


def _verdict_game(game: str) -> str:
    return str(game).split("-")[0]


def build_search_artifact(
    result: SearchResult,
    *,
    game: str,
    target_level: int,
    prior_level: int,
    real_env_confirmed: bool,
    heuristic_used: str = DEFAULT_HEURISTIC_NAME,
    inference_substrate: str,
    duration_s: float,
    random_seed: int = 4021,
    diagnosis: str = "",
) -> dict[str, Any]:
    """Build the required Exp 4021 terminal artifact from planner evidence."""

    solved_and_confirmed = bool(result.solved and real_env_confirmed)
    short_game = _verdict_game(game)
    target_level = int(target_level)
    prior_level = int(prior_level)
    bottleneck = result.bottleneck or "real_env_confirmation_failed"
    if solved_and_confirmed:
        verdict = f"complete: search_layer_solved_{short_game}_L{target_level}_real_env_confirmed"
        new_levels = max(1, target_level - prior_level)
    else:
        verdict = f"complete: search_layer_no_solve_{short_game}_L{target_level}_{bottleneck}"
        new_levels = 0

    return {
        "experiment": "experiment_4021_heuristic_search_over_verified_wm",
        "title": "arc3_heuristic_search_over_verified_world_model",
        "game": str(game),
        "target_level": target_level,
        "prior_wall_level": prior_level,
        "honest_verdict": verdict,
        "new_levels_solved_this_task": int(new_levels),
        "wall_was_search_not_representation": bool(solved_and_confirmed),
        "nodes_expanded": int(result.nodes_expanded),
        "max_expansions": int(result.max_expansions),
        "heuristic_used": str(heuristic_used),
        "real_env_confirmed": bool(real_env_confirmed),
        "inference_substrate": str(inference_substrate),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "search_found_plan": bool(result.solved),
        "search_advanced_past_single_step_stall": bool(solved_and_confirmed),
        "action_plan": result.actions,
        "action_count": len(result.actions),
        "bottleneck": "" if solved_and_confirmed else bottleneck,
        "representation_vs_search_diagnosis": diagnosis
        or (
            "search over the code world model solved the wall, so representation was not the bottleneck"
            if solved_and_confirmed
            else "bounded search did not produce a real-env-confirmed solve"
        ),
        "inference_substrate_principle": (
            "verifier simulator expands T(s,a); Exp 4020 sandboxed is_goal tests terminal states"
        ),
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Return human-readable schema errors for the required Exp 4021 fields."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be a terminal-prefix string")

    for field in ("new_levels_solved_this_task", "nodes_expanded"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")

    for field in ("wall_was_search_not_representation", "real_env_confirmed"):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")

    for field in ("heuristic_used", "inference_substrate"):
        if field in artifact and type(artifact[field]) is not str:
            errors.append(f"{field} must be a bare string")

    if (
        "nodes_expanded" in artifact
        and "max_expansions" in artifact
        and isinstance(artifact["nodes_expanded"], int)
        and isinstance(artifact["max_expansions"], int)
        and artifact["nodes_expanded"] > artifact["max_expansions"]
    ):
        errors.append("nodes_expanded must not exceed max_expansions")

    return errors


def write_artifact(artifact: dict[str, Any], path: Path = DEFAULT_OUTPUT_ARTIFACT) -> Path:
    """Write stable JSON so downstream reconciliation can diff the experiment."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
