"""Exp 4353: learned ARC action-cost heuristic for action-efficient A* plans.

Spec refs: REQ-LEARN-4353, SCENARIO-LEARN-4353.

This is deliberately not the retired cross-game transfer line.  It trains a
small per-game CPU regressor from reproduced solve-trace states to remaining
env-actions-to-win, then uses that regressor in `OfflineSolver` as `g + h`
path-cost search.  The comparison target is the prior search-compute value-head
mode: greedy verifier routing can cut search states but still return a longer
plan.  The action-cost arm only counts when the full plan reproduces offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_value_learner import collect_trajectory_data


REPO = Path(__file__).resolve().parents[2]
OUTPUT_REL = Path("results/experiment_4353_learned_action_cost_heuristic_efficiency.json")
ENTRYPOINT_REL = Path("results/experiment_4353_learned_action_cost_heuristic_efficiency.py")
REGISTRY_REL = Path("ops/arc_solve_registry.yaml")
RANDOM_SEED = 4353
MIN_REPRODUCED_LEVELS = 5
GAP_ID = "GAP-4353"
INFERENCE_SUBSTRATE = "cpu_offline_arc_agi3_per_game_learned_action_cost_astar"
SPEC_REFS = ["REQ-LEARN-4353", "SCENARIO-LEARN-4353"]

LP85_GAME_ID = "lp85-305b61c3"
LP85_DEPTH = {1: 20, 2: 70, 3: 90}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "action_efficiency_improves",
    "held_out_actions_baseline",
    "held_out_actions_learned",
    "positive_control_passed",
    "reproduction_gated",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An action-efficiency improvement (the learned "
        "heuristic lowers env-actions-to-solve on held-out levels) and an "
        "honest null with the positive control passing (the value head already "
        "captures it) are BOTH decision-grade."
    ),
    "action_efficiency_improves": (
        "BARE bool: the capstone reads this; true iff held-out "
        "actions-to-solve is REDUCED with the learned A* path-cost heuristic vs "
        "the baseline planner AND the positive control confirms headroom "
        "existed (not a degenerate no-headroom null)."
    ),
    "held_out_actions_baseline": (
        "BARE int: total env-actions-to-solve across held-out levels with the "
        "baseline planner -- the efficiency baseline."
    ),
    "held_out_actions_learned": (
        "BARE int: total env-actions-to-solve across held-out levels with the "
        "learned A* heuristic -- the efficiency result (lower is better; the "
        "north-star action-efficiency metric)."
    ),
    "positive_control_passed": (
        "BARE bool: a level with known optimal action-count confirms headroom "
        "existed, so a null is 'the value head already captures it', not 'no "
        "headroom' (FALSE_NEGATIVE_RISK guard)."
    ),
    "reproduction_gated": (
        "BARE bool: true iff every counted plan still passes "
        "arc_solver_kit.reproduce -- an action-minimal plan that does not "
        "reproduce does NOT count."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the learned action-cost heuristic is not the "
        "executable oracle."
    ),
    "preconditions_checked": (
        "Records the solve-trace availability + TRM-stand-down; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
    "random_seed": (
        "Determinism precondition for the heuristic training + the held-out "
        "split + the planning."
    ),
    "reproducibility_checksum": (
        "Hash of the training corpus + the held-out split + the heuristic "
        "config; lets a third party re-run."
    ),
    "model_specs": (
        "CPU regression architecture, train/held-out split, path-cost planner "
        "config, and trace/reproduction substrate."
    ),
}


def _json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:  # pragma: no cover - file hashing boundary
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class ActionCostRegressor:
    """Tiny ridge regressor for `h(state) ~= minimal env-actions-to-win`."""

    def __init__(self, *, ridge: float = 1e-6) -> None:
        self.ridge = float(ridge)
        self.w: np.ndarray | None = None
        self.n_samples = 0

    def fit(self, rows: Sequence[Sequence[float]], targets: Sequence[float]) -> "ActionCostRegressor":
        if not rows:
            raise ValueError("cannot train action-cost regressor with no rows")
        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("action-cost feature rows must be a 2-D matrix")
        if len(y) != x.shape[0]:
            raise ValueError("action-cost targets must match feature row count")
        design = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])
        penalty = self.ridge * np.eye(design.shape[1], dtype=np.float64)
        self.w = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        self.n_samples = int(design.shape[0])
        return self

    def predict(self, row: Sequence[float]) -> float:
        if self.w is None:
            return 0.0
        x = np.asarray([float(value) for value in row] + [1.0], dtype=np.float64)
        return float(max(0.0, x @ self.w))

    def rounded_weights(self) -> list[float]:
        if self.w is None:
            return []
        return [round(float(value), 12) for value in self.w.tolist()]

    def model_summary(self) -> dict[str, Any]:
        return {
            "architecture": "linear ridge regression with bias",
            "target": "minimal env-actions-to-win",
            "n_samples": int(self.n_samples),
            "ridge": float(self.ridge),
            "training_compute": "CPU numpy.linalg.solve",
            "weights": self.rounded_weights(),
            "llm_weight_mutation": False,
        }


class StateActionCostHeuristic:
    """Callable adapter from an ARC game object to a learned action-cost estimate."""

    def __init__(self, regressor: ActionCostRegressor, featurize: Callable[[Any], Sequence[float]]) -> None:
        self.regressor = regressor
        self.featurize = featurize

    def __call__(self, game: Any, _frame: Any | None = None) -> float:
        return self.regressor.predict(self.featurize(game))


def lp85_action_labels(env: Any) -> list[str]:  # pragma: no cover - offline SDK boundary
    """Discover lp85 click buttons from the current offline env layout."""

    from carnot.experiment_4179_arc_incremental_progress import discover_click_buttons

    return [
        json.dumps({"x": int(button["x"]), "y": int(button["y"])}, sort_keys=True)
        for button in discover_click_buttons(env)
    ]


def lp85_apply(env: Any, label: str, _frame: Any) -> Any:  # pragma: no cover - offline SDK boundary
    from arcengine import GameAction

    action = json.loads(label)
    return env.step(GameAction.ACTION6, data={"x": int(action["x"]), "y": int(action["y"])})


def lp85_state_key(game: Any) -> tuple[tuple[Any, ...], ...]:  # pragma: no cover - offline SDK boundary
    from carnot.experiment_4179_arc_incremental_progress import _goal_key

    return _goal_key(game)


def _lp85_dists(game: Any) -> list[float]:  # pragma: no cover - offline SDK boundary
    from carnot.experiment_4179_arc_incremental_progress import _goal_key, _target_goal_key

    actual = _goal_key(game)
    target = _target_goal_key(game)
    by_type: dict[Any, list[tuple[int, int]]] = defaultdict(list)
    for item_type, x, y in actual:
        by_type[item_type].append((x, y))
    distances = []
    for item_type, target_x, target_y in target:
        candidates = by_type.get(item_type, [])
        distances.append(
            min((abs(target_x - x) + abs(target_y - y) for x, y in candidates), default=1000.0)
        )
    return distances


def lp85_featurize(game: Any) -> list[float]:  # pragma: no cover - offline SDK boundary
    """Generic numeric state features used by the learned action-cost head."""

    distances = _lp85_dists(game)
    n = len(distances) or 1
    total = float(sum(distances))
    unsatisfied = float(sum(1 for distance in distances if distance > 0))
    return [
        total,
        unsatisfied,
        total / n,
        float(max(distances) if distances else 0.0),
        float(n),
    ]


def _new_lp85_solver(
    verifier: Callable[[Any], float] | None,
    *,
    path_cost_weight: float = 0.0,
    max_nodes: int = 60000,
) -> kit.OfflineSolver:  # pragma: no cover - offline SDK boundary
    return kit.OfflineSolver(
        LP85_GAME_ID,
        lp85_action_labels,
        lp85_apply,
        lp85_state_key,
        verifier=verifier,
        path_cost_weight=path_cost_weight,
        max_nodes=max_nodes,
    )


def _make_lp85_env() -> Any:  # pragma: no cover - offline SDK boundary
    arcade = kit.offline_arcade()
    return arcade.make(LP85_GAME_ID, scorecard_id=arcade.open_scorecard())


def _reproduce_lp85(solution: Sequence[str], *, claimed_level: int = 3) -> dict[str, Any]:  # pragma: no cover - offline SDK boundary
    return kit.reproduce(LP85_GAME_ID, solution, lp85_apply, claimed_level=claimed_level)


def trace_level_sources(repo: Path = REPO) -> list[dict[str, Any]]:  # pragma: no cover - filesystem preflight
    """REQ-LEARN-4353-1: enumerate loadable solved-level action traces."""

    sources: list[dict[str, Any]] = []
    for path in sorted((repo / "results").glob("arc_explore_trajectory_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        trajectory = payload.get("trajectory")
        reached = int(payload.get("reached_level") or 0)
        game = str(payload.get("game") or path.stem.removeprefix("arc_explore_trajectory_"))
        if isinstance(trajectory, list) and trajectory and reached > 0:
            for level_index in range(1, reached + 1):
                sources.append(
                    {
                        "level_id": f"{game}:L{level_index}",
                        "game": game,
                        "source": str(path.relative_to(repo)),
                        "action_count": len(trajectory),
                    }
                )

    lp85_path = repo / "results" / "arc3_lp85_offline_resolve.json"
    try:
        lp85_payload = json.loads(lp85_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lp85_payload = {}
    lp85_solution = lp85_payload.get("solution")
    lp85_reached = int(lp85_payload.get("reached_level") or 0)
    if isinstance(lp85_solution, list) and lp85_solution and lp85_reached > 0:
        for level_index in range(1, lp85_reached + 1):
            sources.append(
                {
                    "level_id": f"lp85:L{level_index}",
                    "game": "lp85",
                    "source": str(lp85_path.relative_to(repo)),
                    "action_count": len(lp85_solution),
                }
            )
    return sources


def build_preconditions(repo: Path = REPO) -> dict[str, Any]:  # pragma: no cover - filesystem preflight
    """REQ-LEARN-4353-1: record trace availability and TRM stand-down."""

    registry_path = repo / REGISTRY_REL
    registry_text = registry_path.read_text(encoding="utf-8") if registry_path.exists() else ""
    sources = trace_level_sources(repo)
    return {
        "registry_path": REGISTRY_REL.as_posix(),
        "registry_present": registry_path.exists(),
        "registry_sha256": _sha256_file(registry_path) if registry_path.exists() else "",
        "registry_mentions_reproduced": "reproducibility: reproduced" in registry_text,
        "usable_reproduced_level_count": len(sources),
        "usable_level_ids": [source["level_id"] for source in sources],
        "trace_sources": sorted({source["source"] for source in sources}),
        "minimum_reproduced_levels": MIN_REPRODUCED_LEVELS,
        "trm_training_stood_down": True,
        "research_conductor_modified": False,
        "offline_cpu_only": True,
    }


def _train_lp85_action_cost() -> tuple[ActionCostRegressor, list[str], list[list[float]], list[float]]:  # pragma: no cover - offline SDK boundary
    """REQ-LEARN-4353-2: train from lp85 L1-L2 reproduced solve states."""

    env = _make_lp85_env()
    solver = _new_lp85_solver(None)
    prefix: list[str] = []
    rows: list[list[float]] = []
    targets: list[float] = []
    for level in (1, 2):
        path, _states = solver.solve_level(env, level - 1, prefix, LP85_DEPTH[level])
        if path is None:
            raise RuntimeError(f"lp85 L{level} training solve failed")
        level_rows, level_targets = collect_trajectory_data(env, solver, prefix, path, lp85_featurize)
        rows.extend(level_rows)
        targets.extend(level_targets)
        prefix.extend(path)
    return ActionCostRegressor().fit(rows, targets), prefix, rows, targets


def evaluate_lp85_heldout_l3() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:  # pragma: no cover - offline SDK boundary
    """REQ-LEARN-4353: held-out L3 action-count comparison, reproduction-gated."""

    regressor, prefix, train_rows, train_targets = _train_lp85_action_cost()
    heuristic = StateActionCostHeuristic(regressor, lp85_featurize)

    env = _make_lp85_env()
    baseline_solver = _new_lp85_solver(heuristic, path_cost_weight=0.0)
    baseline_path, baseline_states = baseline_solver.solve_level(env, 2, prefix, LP85_DEPTH[3])

    env = _make_lp85_env()
    learned_solver = _new_lp85_solver(heuristic, path_cost_weight=1.0)
    learned_path, learned_states = learned_solver.solve_level(env, 2, prefix, LP85_DEPTH[3])

    env = _make_lp85_env()
    bfs_solver = _new_lp85_solver(None, path_cost_weight=0.0)
    bfs_path, bfs_states = bfs_solver.solve_level(env, 2, prefix, LP85_DEPTH[3])

    baseline_full = prefix + list(baseline_path or [])
    learned_full = prefix + list(learned_path or [])
    bfs_full = prefix + list(bfs_path or [])
    baseline_repro = _reproduce_lp85(baseline_full, claimed_level=3) if baseline_path else {"reproduced": False}
    learned_repro = _reproduce_lp85(learned_full, claimed_level=3) if learned_path else {"reproduced": False}
    bfs_repro = _reproduce_lp85(bfs_full, claimed_level=3) if bfs_path else {"reproduced": False}

    baseline_actions = len(baseline_path or [])
    learned_actions = len(learned_path or [])
    known_short_actions = len(bfs_path or [])
    row = {
        "held_out_level_id": "lp85:L3",
        "game": "lp85",
        "target_level": 3,
        "baseline_planner": "greedy_learned_value_head_h_only",
        "learned_planner": "astar_path_cost_g_plus_learned_h",
        "baseline_actions": int(baseline_actions),
        "learned_actions": int(learned_actions),
        "known_short_actions": int(known_short_actions),
        "baseline_full_replay_actions": int(len(baseline_full)),
        "learned_full_replay_actions": int(len(learned_full)),
        "positive_control_full_replay_actions": int(len(bfs_full)),
        "baseline_states_expanded": int(baseline_states),
        "learned_states_expanded": int(learned_states),
        "positive_control_states_expanded": int(bfs_states),
        "baseline_reproduced": bool(baseline_repro.get("reproduced")),
        "learned_reproduced": bool(learned_repro.get("reproduced")),
        "positive_control_reproduced": bool(bfs_repro.get("reproduced")),
        "headroom_exists": bool(
            bfs_path
            and bfs_repro.get("reproduced")
            and baseline_path
            and baseline_actions > known_short_actions
        ),
        "baseline_reproduction_gate": baseline_repro,
        "learned_reproduction_gate": learned_repro,
        "positive_control_reproduction_gate": bfs_repro,
    }
    split_spec = {
        "split_axis": "lp85_levels",
        "train_level_ids": ["lp85:L1", "lp85:L2"],
        "held_out_level_ids": ["lp85:L3"],
        "positive_control": {
            "level_id": "lp85:L3",
            "planner": "plain_bfs_shortest_path",
            "known_short_actions": int(known_short_actions),
            "reproduced": bool(bfs_repro.get("reproduced")),
        },
        "held_out_rows_excluded_from_training": True,
    }
    model_specs = {
        "action_cost_regressor": regressor.model_summary(),
        "feature_names": ["total_goal_distance", "unsatisfied_goals", "mean_goal_distance", "max_goal_distance", "n_goals"],
        "training_corpus": {
            "source": "offline-reproduced lp85 L1-L2 solver traces",
            "n_rows": len(train_rows),
            "targets": [float(target) for target in train_targets],
        },
        "planner_comparison": {
            "baseline": "OfflineSolver verifier priority h(state), path_cost_weight=0.0",
            "learned": "OfflineSolver A* priority g(path_len)+h(state), path_cost_weight=1.0",
            "action_count_metric": "held-out L3 suffix env-actions; reproduction gate replays full L0-L3 path",
        },
    }
    return [row], split_spec, model_specs


def summarize_action_efficiency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """REQ-LEARN-4353-5/6: aggregate bare action-efficiency gate fields."""

    held_out_actions_baseline = sum(int(row.get("baseline_actions", 0) or 0) for row in rows)
    held_out_actions_learned = sum(int(row.get("learned_actions", 0) or 0) for row in rows)
    reproduction_gated = bool(rows) and all(
        bool(row.get("baseline_reproduced")) and bool(row.get("learned_reproduced"))
        for row in rows
    )
    positive_control_passed = bool(rows) and any(bool(row.get("headroom_exists")) for row in rows)
    action_efficiency_improves = bool(
        reproduction_gated
        and positive_control_passed
        and held_out_actions_learned < held_out_actions_baseline
    )
    return {
        "action_efficiency_improves": action_efficiency_improves,
        "held_out_actions_baseline": int(held_out_actions_baseline),
        "held_out_actions_learned": int(held_out_actions_learned),
        "positive_control_passed": positive_control_passed,
        "reproduction_gated": reproduction_gated,
    }


def _verdict(summary: Mapping[str, Any]) -> str:
    if summary.get("action_efficiency_improves") is True:
        baseline = int(summary.get("held_out_actions_baseline", 0) or 0)
        learned = int(summary.get("held_out_actions_learned", 0) or 0)
        return f"success: learned_action_cost_reduces_actions_{baseline}_to_{learned}"
    if summary.get("positive_control_passed") is True:
        return "complete: learned_action_cost_no_reduction_positive_control_passed"
    return "complete: learned_action_cost_no_reduction_positive_control_failed"


def _missing_gap_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if bool(row.get("baseline_reproduced"))
        and bool(row.get("learned_reproduced"))
        and int(row.get("learned_actions", 0) or 0) >= int(row.get("baseline_actions", 0) or 0)
    ]


def _gap_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gap_rows = _missing_gap_rows(rows)
    if not gap_rows:
        return []
    return [
        {
            "gap_id": GAP_ID,
            "held_out_level_ids": [str(row.get("held_out_level_id")) for row in gap_rows],
            "failure_mode": (
                "learned action-cost heuristic did not reduce env-actions-to-solve "
                "versus the baseline planner on reproduced held-out levels"
            ),
            "missing_discriminator": "state feature that predicts shorter action plans, not just lower search energy",
            "candidate_design": "richer per-game action-cost features or exact path-cost labels from more reproduced levels",
            "priority": "medium",
        }
    ]


def build_blocked_artifact(
    *,
    usable_levels: Sequence[str],
    missing_sources: Sequence[str],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4353-BLOCKED: terminal artifact for insufficient traces."""

    checksum_payload = {
        "usable_levels": list(usable_levels),
        "missing_sources": list(missing_sources),
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4353_learned_action_cost_heuristic_efficiency",
        "title": "learned_action_cost_heuristic_efficiency",
        "honest_verdict": "blocked_insufficient_solve_traces",
        "action_efficiency_improves": False,
        "held_out_actions_baseline": 0,
        "held_out_actions_learned": 0,
        "positive_control_passed": False,
        "reproduction_gated": False,
        "verifier_is_oracle": False,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _json_hash(checksum_payload),
        "model_specs": {
            "blocked_reason": "insufficient_solve_traces",
            "usable_levels": list(usable_levels),
            "missing_sources": list(missing_sources),
            "minimum_reproduced_levels": MIN_REPRODUCED_LEVELS,
            "value_head": "not_trained",
            "llm_weight_mutation": False,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "held_out_level_rows": [],
        "missing_verifier_gaps": [],
        "acceptance_gate_passed": True,
    }


def build_complete_artifact(
    *,
    held_out_rows: Sequence[Mapping[str, Any]],
    split_spec: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    adversarial_verify: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4353: construct the reproduction-gated result artifact."""

    rows = [dict(row) for row in held_out_rows]
    summary = summarize_action_efficiency(rows)
    checksum_payload = {
        "held_out_rows": rows,
        "split_spec": dict(split_spec),
        "model_specs": dict(model_specs),
        "preconditions_checked": dict(preconditions_checked),
        "summary": summary,
        "random_seed": RANDOM_SEED,
        "heuristic_config": {"path_cost_weight": 1.0, "regressor": "linear_ridge"},
    }
    artifact = {
        "experiment": "experiment_4353_learned_action_cost_heuristic_efficiency",
        "title": "learned_action_cost_heuristic_efficiency",
        **summary,
        "honest_verdict": _verdict(summary),
        "verifier_is_oracle": False,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _json_hash(checksum_payload),
        "model_specs": {
            "module": "python/carnot/experiment_4353_learned_action_cost_heuristic_efficiency.py",
            "offline_solver": "python/carnot/agentic/arc_solver_kit.py:OfflineSolver(path_cost_weight)",
            "split": dict(split_spec),
            "heuristic": dict(model_specs),
            "verifier_is_oracle": False,
            "llm_weight_mutation": False,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "held_out_level_rows": rows,
        "missing_verifier_gaps": _gap_payload(rows),
        "adversarial_verify": dict(adversarial_verify or {}),
        "methodology_note": (
            "CPU-only per-game action-cost heuristic. The held-out L3 rows are "
            "excluded from training; the learned arm uses g+h path-cost search, "
            "and both arms are counted only after full offline reproduction."
        ),
        "acceptance_gate_passed": True,
    }
    return artifact


def _is_bare_int(value: Any) -> bool:
    return type(value) is int


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """SCENARIO-LEARN-4353: validate required bare fields and gates."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")
    if type(artifact.get("action_efficiency_improves")) is not bool:
        errors.append("action_efficiency_improves must be a bare bool")
    if not _is_bare_int(artifact.get("held_out_actions_baseline")):
        errors.append("held_out_actions_baseline must be a bare int")
    if not _is_bare_int(artifact.get("held_out_actions_learned")):
        errors.append("held_out_actions_learned must be a bare int")
    if type(artifact.get("positive_control_passed")) is not bool:
        errors.append("positive_control_passed must be a bare bool")
    if type(artifact.get("reproduction_gated")) is not bool:
        errors.append("reproduction_gated must be a bare bool")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be the bare bool false")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be an object")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs must be an object")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be an object")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if principles.get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles mismatch for {field}")
    if artifact.get("action_efficiency_improves") is True:
        if artifact.get("positive_control_passed") is not True:
            errors.append("action_efficiency_improves requires positive_control_passed=true")
        if artifact.get("reproduction_gated") is not True:
            errors.append("action_efficiency_improves requires reproduction_gated=true")
        baseline = artifact.get("held_out_actions_baseline")
        learned = artifact.get("held_out_actions_learned")
        if not (_is_bare_int(baseline) and _is_bare_int(learned) and int(learned) < int(baseline)):
            errors.append("action_efficiency_improves requires learned actions < baseline actions")
    return errors


def ensure_gap_logged(repo: Path, artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-4353-7: append unreduced held-out levels to the gap ledger."""

    gaps = artifact.get("missing_verifier_gaps")
    if not isinstance(gaps, list) or not gaps:
        return
    gap_path = repo / "ops" / "verifier_gaps.md"
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    text = gap_path.read_text(encoding="utf-8") if gap_path.exists() else "# Verifier Gaps\n\n"
    if GAP_ID in text:
        return
    level_ids = []
    for gap in gaps:
        if isinstance(gap, Mapping):
            level_ids.extend(str(level) for level in gap.get("held_out_level_ids", []) or [])
    entry = (
        f"\n### {GAP_ID}: ARC action-cost heuristic residual\n"
        "- status: open\n"
        f"- evidence: `{OUTPUT_REL.as_posix()}` reports unreduced held-out levels: "
        f"{', '.join(level_ids) or 'unknown'}.\n"
        "- failure mode: the learned action-cost heuristic did not reduce "
        "env-actions-to-solve for every reproduction-gated held-out level.\n"
        "- missing discriminator: state features that distinguish shorter valid "
        "plans from search-energy progress alone.\n"
        "- candidate design: train on more per-game reproduced levels with "
        "exact shortest-path labels and richer action-effect features.\n"
        "- priority: medium\n"
    )
    gap_path.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")


def run_adversarial_verify(repo: Path) -> dict[str, Any]:  # pragma: no cover - subprocess boundary
    """REQ-LEARN-4353-8: run artifact verification after writing the JSON."""

    output = repo / OUTPUT_REL
    cmd = [sys.executable, str(repo / "scripts" / "adversarial_verify.py"), str(output), "--json"]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    try:
        report = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        report = {"stdout": completed.stdout, "stderr": completed.stderr}
    flagged_count = int(report.get("flagged_count", 0) or 0)
    return {
        "status": "clean" if completed.returncode == 0 and flagged_count == 0 else "flagged",
        "returncode": int(completed.returncode),
        "flagged_count": flagged_count,
        "reports": report.get("reports", []),
    }


def _write_artifact(repo: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover - filesystem boundary
    output = repo / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate(repo: Path = REPO) -> dict[str, Any]:  # pragma: no cover - offline SDK boundary
    started = time.time()
    preconditions = build_preconditions(repo)
    usable_levels = list(preconditions.get("usable_level_ids", []) or [])
    if int(preconditions.get("usable_reproduced_level_count", 0) or 0) < MIN_REPRODUCED_LEVELS:
        return build_blocked_artifact(
            usable_levels=[str(level) for level in usable_levels],
            missing_sources=[],
            preconditions_checked=preconditions,
            duration_s=time.time() - started,
        )

    held_out_rows, split_spec, model_specs = evaluate_lp85_heldout_l3()
    return build_complete_artifact(
        held_out_rows=held_out_rows,
        split_spec=split_spec,
        model_specs=model_specs,
        preconditions_checked=preconditions,
        duration_s=time.time() - started,
    )


def run(*, repo: Path = REPO, write: bool = True) -> dict[str, Any]:  # pragma: no cover - CLI/integration boundary
    artifact = evaluate(repo)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_artifact(repo, artifact)
        artifact = dict(artifact)
        if not artifact["honest_verdict"].startswith("blocked_"):
            artifact["adversarial_verify"] = run_adversarial_verify(repo)
        _write_artifact(repo, artifact)
        ensure_gap_logged(repo, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
