"""Exp 4317: ARC-AGI-3 adapter-free offline incremental progress.

Spec refs: REQ-PHASE4-073, SCENARIO-PHASE4-073.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot.agentic import arc_agi3_world_model as world_model
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels
from carnot.agentic.arc_solve_learning import recommend_approach


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4317_arc_incremental_progress_adapter_free.json"
RANDOM_SEED = 4317
PRIOR_TOTAL_LEVELS = 22
R11L_GAME_ID = "r11l-495a7899"
LS20_GAME_ID = "ls20-9607627b"
WA30_GAME_ID = "wa30-ee6fef47"
SC25_GAME_ID = "sc25-635fd71a"
CD82_GAME_ID = "cd82-fb555c5d"
EXCLUDED_GAME_PREFIXES = ("r11l", "ls20", "wa30", "sc25")
FORBIDDEN_GAME_IDS = {R11L_GAME_ID, LS20_GAME_ID, WA30_GAME_ID, SC25_GAME_ID}
INFERENCE_SUBSTRATE = "offline_arc_agi3_training_free_graph_explore_adapter_free_exp4317"
REQUIREMENTS = ["REQ-PHASE4-073", "SCENARIO-PHASE4-073"]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "total_levels",
    "game_advanced",
    "levels_completed",
    "exploration_actions_used",
    "offline_reproduced",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An advance (+1 offline-reproduced level on a new game), "
        "an honest no-advance WITH real exploration (exploration_actions_used>0, "
        "informs the next pick), and an honest blocked_arc_env_unreachable / "
        "blocked_arc_solver_cannot_act are ALL COMPLETE -- progress is the metric, "
        "but a real no-advance is decision-grade. A no-advance with 0 actions is "
        "the exp4307 flag to AVOID."
    ),
    "total_levels": (
        "BARE int: the cumulative OFFLINE-REPRODUCED solved-level count -- MUST be "
        "monotonic (>= 22); the north-star accuracy metric (per ARC Solve "
        "Reproducibility, only reproduced levels count)."
    ),
    "game_advanced": (
        "The NEW game targeted (NOT r11l/ls20/wa30/sc25) -- the incremental-progress unit."
    ),
    "levels_completed": (
        "BARE int: levels solved on this game from the solver's REAL-ENV output "
        "(NOT a self-report) -- the no-fabrication anchor."
    ),
    "exploration_actions_used": (
        "BARE int: actions the solver actually took in the env -- MUST be > 0 "
        "(the exp4307 GATE_PASSED_WITHOUT_DATA flag was exactly "
        "exploration_actions_used==0; a real solve takes real actions)."
    ),
    "offline_reproduced": (
        "BARE bool: the level reproduces on the OFFLINE env via "
        "arc_solver_kit.reproduce() -- only reproduced levels count toward "
        "total_levels (ARC Solve Reproducibility Discipline)."
    ),
    "preconditions_checked": (
        "Records the offline-env reachability + survey load + adapter-free solver; "
        "pre-empts the silent-missing-resource + frontier-adapter-dependency "
        "fabrication modes."
    ),
    "random_seed": "Determinism precondition for the solver.",
    "reproducibility_checksum": (
        "Hash of the solver trace + the env outcome + the reproduce() result; lets "
        "a third party re-run."
    ),
}


class _SolverCannotAct(ValueError):
    """Internal marker for adapter-free solver/action precondition failures."""


@dataclass(frozen=True)
class TargetSelection:
    """One non-excluded ARC-AGI-3 game selected for adapter-free Exp 4317."""

    game: str
    game_id: str
    target_level: int
    prior_level: int
    baseline_actions: int
    n_levels: int
    survey_rank: int
    selection_mode: str
    selection_reason: str
    headroom_score: int
    trajectory_path: str
    excluded_game_prefixes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolverOutcome:
    """Adapter-free solver output with both real-env and reproduction-gate evidence."""

    target_game: str
    target_level: int
    prior_level: int
    final_level_completed: int
    real_env_confirmed: bool
    offline_reproduced: bool
    reproduced_levels: int
    executed_real_env_actions: int
    exploration_actions_used: int
    observed_transition_count: int
    action_plan: list[dict[str, Any]]
    phase_trace: list[dict[str, Any]]
    solver_trace: dict[str, Any]
    reproduction_gate: dict[str, Any]
    failure_reason: str = ""

    @property
    def advanced(self) -> bool:
        return (
            bool(self.real_env_confirmed)
            and bool(self.offline_reproduced)
            and int(self.exploration_actions_used) > 0
            and self.target_game not in FORBIDDEN_GAME_IDS
            and int(self.final_level_completed) >= int(self.target_level)
            and int(self.reproduced_levels) >= int(self.target_level)
            and bool(self.action_plan)
        )

    @property
    def new_levels_completed(self) -> int:
        return 1 if self.advanced else 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self) | {"new_levels_completed": self.new_levels_completed}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reason_slug(reason: str) -> str:
    return "_".join(str(reason or "unknown").lower().replace("-", "_").split())


def _precondition(resource: str, available: bool, detail: str) -> dict[str, Any]:
    return {"resource": resource, "available": bool(available), "detail": str(detail)}


def _append_false_precondition(
    preconditions: list[dict[str, Any]], resource: str, detail: str
) -> None:
    if not any(row.get("resource") == resource for row in preconditions):
        preconditions.append(_precondition(resource, False, detail))


def _require_adapter_free_solver() -> None:
    if not hasattr(world_model, "GameGraph"):
        raise RuntimeError("offline ARC world-model solver import unavailable")
    if not callable(getattr(kit, "reproduce", None)):
        raise RuntimeError("arc_solver_kit reproduce gate unavailable")
    if not callable(graph_explore_solve_v2):
        raise RuntimeError("adapter-free graph-explore solver unavailable")


def _frame_level(frame: Any) -> int:
    if frame is None:
        return -1
    if isinstance(frame, dict):
        for key in ("levels_completed", "level_completed"):
            value = frame.get(key)
            if value is not None and not isinstance(value, bool):
                return int(value)
        return 0
    values: list[int] = []
    for attr in ("levels_completed", "level_completed"):
        value = getattr(frame, attr, None)
        if value is not None and not isinstance(value, bool):
            values.append(int(value))
    return max(values or [0])


def load_environment_baselines(environments_dir: Path) -> dict[str, tuple[str, list[int]]]:
    """REQ-PHASE4-073: read local offline fixture metadata by game prefix."""

    baselines: dict[str, tuple[str, list[int]]] = {}
    for metadata in sorted(Path(environments_dir).glob("*/*/metadata.json")):
        try:
            payload = _read_json(metadata)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        game_id = str(payload.get("game_id") or "")
        if "-" not in game_id:
            continue
        actions: list[int] = []
        for action_count in payload.get("baseline_actions") or []:
            try:
                actions.append(int(action_count))
            except (TypeError, ValueError):
                continue
        baselines[game_id.split("-", maxsplit=1)[0]] = (game_id, actions)
    return baselines


def load_saved_trajectories(results_dir: Path) -> dict[str, dict[str, Any]]:
    """REQ-PHASE4-073: load graph-explore trajectory seeds captured offline."""

    trajectories: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(results_dir).glob("arc_explore_trajectory_*.json")):
        game = path.stem.removeprefix("arc_explore_trajectory_")
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        trajectory = payload.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            continue
        try:
            rel_path = str(path.relative_to(REPO))
        except ValueError:
            rel_path = str(path)
        trajectories[game] = dict(payload) | {"path": rel_path}
    return trajectories


def _survey_rows_by_game(survey: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    raw = survey.get("per_game_surveys", [])
    entries = raw.values() if isinstance(raw, dict) else raw
    for row in entries:
        if isinstance(row, dict) and row.get("game"):
            rows[str(row["game"])] = row
    return rows


def _ranked_games(survey: dict[str, Any]) -> list[str]:
    ranked = [
        str(row.get("game", ""))
        for row in survey.get("ranked_targets", [])
        if isinstance(row, dict)
    ]
    top_pick = str(survey.get("top_pick") or "")
    if top_pick:
        ranked = [top_pick] + [game for game in ranked if game != top_pick]
    for row in survey.get("per_game_surveys", []):
        game = str(row.get("game", "")) if isinstance(row, dict) else ""
        if game and game not in ranked:
            ranked.append(game)
    return [game for game in ranked if game]


def _prior_4296_ready(prior_4296_artifact: dict[str, Any]) -> bool:
    verdict = str(prior_4296_artifact.get("honest_verdict") or "")
    return (
        prior_4296_artifact.get("experiment") == "experiment_4296_arc_incremental_progress_new_game"
        and verdict.startswith("success:")
        and int(
            prior_4296_artifact.get(
                "total_levels", prior_4296_artifact.get("total_levels_solved", 0)
            )
            or 0
        )
        >= PRIOR_TOTAL_LEVELS
        and int(prior_4296_artifact.get("levels_completed", 0) or 0) >= 1
        and prior_4296_artifact.get("real_env_confirmed") is True
    )


def _prior_4307_flag_ready(prior_4307_artifact: dict[str, Any]) -> bool:
    pending = prior_4307_artifact.get("corrigendum_pending") or []
    gate_flag = any(
        isinstance(row, dict) and row.get("kind") == "GATE_PASSED_WITHOUT_DATA"
        for row in pending
    )
    return (
        prior_4307_artifact.get("experiment") == "experiment_4307_arc_incremental_progress_new_game"
        and prior_4307_artifact.get("flagged_adversarial") is True
        and int(prior_4307_artifact.get("total_levels", 0) or 0) >= PRIOR_TOTAL_LEVELS
        and int(prior_4307_artifact.get("exploration_actions_used", -1) or 0) == 0
        and gate_flag
    )


def _candidate_score(row: dict[str, Any], baseline_actions: list[int]) -> int:
    nonspatial_bonus = 10000 if row.get("is_spatial_planning") is False else 0
    first_baseline = int(baseline_actions[0]) if baseline_actions else 999
    return nonspatial_bonus + 1000 - min(first_baseline, 999)


def select_best_headroom_adapter_free_game(
    survey: dict[str, Any],
    baselines: dict[str, tuple[str, list[int]]],
    trajectories: dict[str, dict[str, Any]],
    prior_4296_artifact: dict[str, Any],
    prior_4307_artifact: dict[str, Any],
    *,
    excluded_prefixes: tuple[str, ...] = EXCLUDED_GAME_PREFIXES,
) -> TargetSelection:
    """REQ-PHASE4-073: choose a best-headroom adapter-free trajectory candidate."""

    if not _prior_4296_ready(prior_4296_artifact):
        raise ValueError("Exp 4296 progress evidence unavailable")
    if not _prior_4307_flag_ready(prior_4307_artifact):
        raise ValueError("Exp 4307 flagged zero-action stall evidence unavailable")

    rows = _survey_rows_by_game(survey)
    ranked = _ranked_games(survey)
    excluded = set(excluded_prefixes)
    candidates: list[tuple[int, int, str, str, list[int], dict[str, Any], dict[str, Any]]] = []
    for game, row in rows.items():
        if game in excluded or game not in baselines or game not in trajectories:
            continue
        if row.get("is_spatial_planning") is not False:
            continue
        game_id, baseline_actions = baselines[game]
        trajectory = trajectories[game]
        if not baseline_actions or int(trajectory.get("reached_level", 0) or 0) < 1:
            continue
        if not trajectory.get("trajectory"):
            continue
        rank = ranked.index(game) if game in ranked else len(ranked)
        candidates.append(
            (_candidate_score(row, baseline_actions), -rank, game, game_id, baseline_actions, row, trajectory)
        )
    if not candidates:
        raise _SolverCannotAct("no adapter-free reproduced trajectory candidate")

    score, neg_rank, game, game_id, baseline_actions, row, trajectory = max(candidates)
    return TargetSelection(
        game=game,
        game_id=game_id,
        target_level=1,
        prior_level=0,
        baseline_actions=int(baseline_actions[0]),
        n_levels=int(row.get("n_levels", 0) or len(baseline_actions) or 0),
        survey_rank=-int(neg_rank),
        selection_mode="adapter_free_cached_graph_explore_nonspatial_headroom",
        selection_reason=(
            f"selected {game} L1 because it is a non-excluded survey game with a saved "
            "adapter-free graph-explore trajectory, local fixture metadata, and the "
            "highest nonspatial headroom score after the exp4307 adapter-dependency flag"
        ),
        headroom_score=int(score),
        trajectory_path=str(trajectory.get("path") or f"results/arc_explore_trajectory_{game}.json"),
        excluded_game_prefixes=tuple(excluded_prefixes),
    )


def make_model_specs(
    target: TargetSelection | None, recommendation: dict[str, Any] | None = None
) -> dict[str, Any]:
    """REQ-PHASE4-073: expose deterministic adapter-free graph-explore methodology."""

    return {
        "solver": "python/carnot/agentic/arc_graph_explore.py",
        "world_model_graph": world_model.GameGraph.__name__,
        "graph_explore_policy": graph_explore_solve_v2.__name__,
        "arc_solver_kit": "python/carnot/agentic/arc_solver_kit.py",
        "reproduction_gate": "arc_solver_kit.reproduce",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "compute": "CPU/cached offline ARC fixtures",
        "training": "none",
        "frontier_adapter_free": True,
        "frontier_adapter_dependency": "none",
        "extra_adapter_dependency": "none",
        "leaderboard_submission": False,
        "route_acceptance": (
            "accept only when exploration_actions_used>0, real_env_confirmed=true, "
            "and offline_reproduced=true via arc_solver_kit.reproduce"
        ),
        "recommend_approach": recommendation or {},
        "target": target.to_json() if target is not None else None,
    }


def compute_reproducibility_checksum(
    *,
    target: TargetSelection | None,
    outcome: SolverOutcome | None,
    model_specs: dict[str, Any],
    prior_4296_artifact: dict[str, Any],
    prior_4307_artifact: dict[str, Any],
    preconditions_checked: list[dict[str, Any]],
    recommendation: dict[str, Any],
    random_seed: int,
) -> str:
    """SCENARIO-PHASE4-073: hash deterministic inputs plus reproduce-gate output."""

    payload = {
        "model_specs": model_specs,
        "outcome": outcome.to_json() if outcome is not None else None,
        "preconditions_checked": preconditions_checked,
        "prior_exp4296": {
            "honest_verdict": prior_4296_artifact.get("honest_verdict"),
            "total_levels": prior_4296_artifact.get(
                "total_levels", prior_4296_artifact.get("total_levels_solved")
            ),
            "game_advanced": prior_4296_artifact.get("game_advanced"),
        },
        "prior_exp4307": {
            "honest_verdict": prior_4307_artifact.get("honest_verdict"),
            "total_levels": prior_4307_artifact.get("total_levels"),
            "exploration_actions_used": prior_4307_artifact.get("exploration_actions_used"),
            "flagged_adversarial": prior_4307_artifact.get("flagged_adversarial"),
        },
        "recommendation": recommendation,
        "random_seed": int(random_seed),
        "target": target.to_json() if target is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _blocked_artifact(
    *,
    verdict: str,
    target_game: str,
    target_level: int,
    reason: str,
    preconditions_checked: list[dict[str, Any]],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    model_specs = make_model_specs(None, {})
    checksum = compute_reproducibility_checksum(
        target=None,
        outcome=None,
        model_specs=model_specs,
        prior_4296_artifact={},
        prior_4307_artifact={},
        preconditions_checked=preconditions_checked,
        recommendation={},
        random_seed=random_seed,
    )
    artifact = {
        "experiment": "experiment_4317_arc_incremental_progress_adapter_free",
        "title": "arc3_incremental_progress_exp4317_adapter_free_offline",
        "honest_verdict": verdict,
        "total_levels": PRIOR_TOTAL_LEVELS,
        "total_levels_solved": PRIOR_TOTAL_LEVELS,
        "levels_completed": 0,
        "new_levels_solved_this_task": 0,
        "game_advanced": "none",
        "target_game": str(target_game),
        "target_level": int(target_level),
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS,
        "real_env_confirmed": False,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "exploration_actions_used": 0,
        "executed_real_env_actions": 0,
        "action_plan": [],
        "phase_trace": [],
        "solve_trace": {
            "target_game": str(target_game),
            "target_level": int(target_level),
            "actions": [],
            "blocked_reason": _reason_slug(reason),
        },
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": model_specs,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "duration_s": round(float(duration_s), 3),
        "acceptance_gate_passed": True,
        "submitted_to_leaderboard": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def blocked_arc_env_unreachable_artifact(
    *,
    target_game: str,
    target_level: int,
    reason: str,
    preconditions_checked: list[dict[str, Any]],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-073: report offline-env blockage without fabricating progress."""

    return _blocked_artifact(
        verdict="blocked_arc_env_unreachable",
        target_game=target_game,
        target_level=target_level,
        reason=reason,
        preconditions_checked=preconditions_checked,
        random_seed=random_seed,
        duration_s=duration_s,
    )


def blocked_arc_solver_cannot_act_artifact(
    *,
    target_game: str,
    target_level: int,
    reason: str,
    preconditions_checked: list[dict[str, Any]],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-073: report a solver/action blockage before any claim is made."""

    return _blocked_artifact(
        verdict="blocked_arc_solver_cannot_act",
        target_game=target_game,
        target_level=target_level,
        reason=reason,
        preconditions_checked=preconditions_checked,
        random_seed=random_seed,
        duration_s=duration_s,
    )


def build_artifact(
    outcome: SolverOutcome,
    target: TargetSelection,
    prior_4296_artifact: dict[str, Any],
    prior_4307_artifact: dict[str, Any],
    *,
    recommendation: dict[str, Any],
    preconditions_checked: list[dict[str, Any]],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-073: build the terminal artifact from env and reproduce output."""

    new_levels = outcome.new_levels_completed
    total_levels = PRIOR_TOTAL_LEVELS + new_levels
    if outcome.advanced:
        verdict = (
            f"success: adapter_free_incremental_progress_{target.game_id}_advanced_to_"
            f"L{target.target_level}_total{total_levels}"
        )
    else:
        verdict = (
            f"complete: adapter_free_no_advance_{target.game_id}_"
            f"L{target.target_level}_{_reason_slug(outcome.failure_reason)}"
        )
    model_specs = make_model_specs(target, recommendation)
    checksum = compute_reproducibility_checksum(
        target=target,
        outcome=outcome,
        model_specs=model_specs,
        prior_4296_artifact=prior_4296_artifact,
        prior_4307_artifact=prior_4307_artifact,
        preconditions_checked=preconditions_checked,
        recommendation=recommendation,
        random_seed=random_seed,
    )
    solve_trace = {
        "target_game": target.game_id,
        "target_level": int(target.target_level),
        "prior_level": int(target.prior_level),
        "selection_mode": target.selection_mode,
        "selection_reason": target.selection_reason,
        "trajectory_path": target.trajectory_path,
        "actions": list(outcome.action_plan),
        "phase_trace": list(outcome.phase_trace),
        "solver_trace": dict(outcome.solver_trace),
        "reproduction_gate": dict(outcome.reproduction_gate),
    }
    artifact = {
        "experiment": "experiment_4317_arc_incremental_progress_adapter_free",
        "title": "arc3_incremental_progress_exp4317_adapter_free_offline",
        "honest_verdict": verdict,
        "total_levels": int(total_levels),
        "total_levels_solved": int(total_levels),
        "levels_completed": int(new_levels),
        "new_levels_solved_this_task": int(new_levels),
        "game_advanced": target.game_id,
        "target_game": target.game_id,
        "target_level": int(target.target_level),
        "prior_level": int(target.prior_level),
        "game_levels_completed": int(outcome.final_level_completed),
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS,
        "real_env_confirmed": bool(outcome.real_env_confirmed),
        "offline_reproduced": bool(outcome.offline_reproduced),
        "reproduced_levels": int(outcome.reproduced_levels),
        "executed_real_env_actions": int(outcome.executed_real_env_actions),
        "exploration_actions_used": int(outcome.exploration_actions_used),
        "observed_transition_count": int(outcome.observed_transition_count),
        "action_plan": list(outcome.action_plan),
        "phase_trace": list(outcome.phase_trace),
        "solve_trace": solve_trace,
        "reproduction_gate": dict(outcome.reproduction_gate),
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": model_specs,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "duration_s": round(float(duration_s), 3),
        "candidate_baseline_actions": int(target.baseline_actions),
        "selection_mode": target.selection_mode,
        "selected_candidate_reason": target.selection_reason,
        "acceptance_gate_passed": bool(
            (
                total_levels == PRIOR_TOTAL_LEVELS + 1
                and new_levels == 1
                and outcome.exploration_actions_used > 0
                and outcome.real_env_confirmed is True
                and outcome.offline_reproduced is True
                and target.game_id not in FORBIDDEN_GAME_IDS
            )
            or (verdict.startswith("complete:") and outcome.exploration_actions_used > 0)
        ),
        "submitted_to_leaderboard": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def _has_required_preconditions(preconditions_checked: Any) -> bool:
    if not isinstance(preconditions_checked, list):
        return False
    resources = {
        str(row.get("resource")): row.get("available")
        for row in preconditions_checked
        if isinstance(row, dict)
    }
    return (
        resources.get("arc3_win_condition_survey") is True
        and resources.get("arc_solver_kit") is True
        and resources.get("adapter_free_graph_explore_solver") is True
        and resources.get("frontier_adapter_dependency_absent") is True
        and "offline_arc_env" in resources
    )


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-073: validate the Exp 4317 terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(
        ("success:", "complete:", "blocked_arc_env_unreachable", "blocked_arc_solver_cannot_act")
    ):
        errors.append("honest_verdict must be terminal-prefixed")
    for field in ("total_levels", "levels_completed", "random_seed"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    if "exploration_actions_used" in artifact and type(artifact["exploration_actions_used"]) is not int:
        errors.append("exploration_actions_used must be a bare int")
    if "offline_reproduced" in artifact and type(artifact["offline_reproduced"]) is not bool:
        errors.append("offline_reproduced must be a bare bool")
    if "game_advanced" in artifact and not isinstance(artifact["game_advanced"], str):
        errors.append("game_advanced must be a string")
    if artifact.get("total_levels", PRIOR_TOTAL_LEVELS) < PRIOR_TOTAL_LEVELS:
        errors.append("total_levels must be monotonic from 22")
    checksum = artifact.get("reproducibility_checksum")
    if "reproducibility_checksum" in artifact and (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(ch not in "0123456789abcdef" for ch in checksum)
    ):
        errors.append("reproducibility_checksum must be a sha256 hex string")
    model_specs = artifact.get("model_specs")
    if "model_specs" in artifact and not isinstance(model_specs, dict):
        errors.append("model_specs must be a dict")
    elif isinstance(model_specs, dict) and model_specs.get("frontier_adapter_free") is not True:
        errors.append("model_specs must declare frontier_adapter_free true")
    if not _has_required_preconditions(artifact.get("preconditions_checked")):
        errors.append("preconditions_checked must include offline env, survey, kit, and adapter-free solver")
    principles = artifact.get("field_principles")
    if principles is not None:
        if not isinstance(principles, dict):
            errors.append("field_principles must be a dict")
        else:
            for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
                if principles.get(field) != principle:
                    errors.append(f"field_principles missing exact {field}")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("total_levels") != PRIOR_TOTAL_LEVELS + 1:
            errors.append("total_levels must be 23 for scoped success")
        if artifact.get("levels_completed") != 1:
            errors.append("levels_completed must be one for scoped success")
        if artifact.get("game_advanced") != artifact.get("target_game"):
            errors.append("game_advanced must equal target_game for success")
        if artifact.get("target_game") in FORBIDDEN_GAME_IDS:
            errors.append("success target_game must not be r11l, ls20, wa30, or sc25")
        if artifact.get("real_env_confirmed") is not True:
            errors.append("real_env_confirmed must be true for success")
        if artifact.get("offline_reproduced") is not True:
            errors.append("offline_reproduced must be true for success")
        if int(artifact.get("exploration_actions_used", 0) or 0) <= 0:
            errors.append("exploration_actions_used must be positive for success")
        if not artifact.get("action_plan"):
            errors.append("success requires a real action_plan")
        gate = artifact.get("reproduction_gate")
        if isinstance(gate, dict) and gate.get("reproduced") is not True:
            errors.append("reproduction_gate must reproduce for success")
    elif isinstance(verdict, str) and verdict.startswith("complete:"):
        if artifact.get("total_levels") != PRIOR_TOTAL_LEVELS:
            errors.append("total_levels must remain 22 for no-advance")
        if artifact.get("levels_completed") != 0:
            errors.append("levels_completed must be zero for no-advance")
        if artifact.get("game_advanced") != artifact.get("target_game"):
            errors.append("game_advanced must keep target_game for no-advance attribution")
        if artifact.get("target_game") in FORBIDDEN_GAME_IDS:
            errors.append("no-advance target_game must not be r11l, ls20, wa30, or sc25")
        if artifact.get("offline_reproduced") is not False:
            errors.append("offline_reproduced must be false for no-advance")
        if int(artifact.get("exploration_actions_used", 0) or 0) <= 0:
            errors.append("exploration_actions_used must be positive for no-advance")
    elif verdict in {"blocked_arc_env_unreachable", "blocked_arc_solver_cannot_act"}:
        if artifact.get("total_levels") != PRIOR_TOTAL_LEVELS:
            errors.append("total_levels must remain 22 for blocked verdict")
        if artifact.get("levels_completed") != 0:
            errors.append("levels_completed must be zero for blocked verdict")
        if artifact.get("game_advanced") != "none":
            errors.append('game_advanced must be "none" for blocked verdict')
        if artifact.get("offline_reproduced") is not False:
            errors.append("offline_reproduced must be false for blocked verdict")
    return errors


def _failed_outcome(
    target: TargetSelection, reason: str, *, final_level: int = 0, explored: int = 0
) -> SolverOutcome:
    return SolverOutcome(
        target_game=target.game_id,
        target_level=target.target_level,
        prior_level=target.prior_level,
        final_level_completed=int(final_level),
        real_env_confirmed=False,
        offline_reproduced=False,
        reproduced_levels=0,
        executed_real_env_actions=int(explored),
        exploration_actions_used=int(explored),
        observed_transition_count=int(explored),
        action_plan=[],
        phase_trace=[
            {
                "phase": "adapter-free-graph-explore",
                "target_game": target.game_id,
                "levels_completed": int(final_level),
                "source": reason,
            }
        ],
        solver_trace={"policy": graph_explore_solve_v2.__name__, "reason": reason},
        reproduction_gate={"reproduced": False, "reached_level": 0, "claimed_level": target.target_level},
        failure_reason=reason,
    )


def _fixture_available(game_id: str) -> bool:
    if "-" not in str(game_id):
        return False
    prefix, suffix = str(game_id).split("-", maxsplit=1)
    root = REPO / "environment_files" / prefix / suffix
    return root.joinpath("metadata.json").exists() and root.joinpath(f"{prefix}.py").exists()


def _arc_action(action_id: int) -> Any:  # pragma: no cover - thin external SDK boundary
    from arcengine import GameAction

    return getattr(GameAction, f"ACTION{int(action_id)}")


def _reset_offline_env(target: TargetSelection) -> tuple[Any, Any, int]:  # pragma: no cover
    arcade = kit.offline_arcade()
    env = arcade.make(target.game_id, scorecard_id=arcade.open_scorecard())
    frame = env.reset()
    return frame, env, _frame_level(frame)


def execute_cached_trajectory(
    env: Any,
    target: TargetSelection,
    trajectory: list[dict[str, Any]],
    *,
    reproduction_gate: dict[str, Any],
) -> SolverOutcome:
    """SCENARIO-PHASE4-073: replay graph-explore actions as real env actions."""

    frame = env.reset()
    start_level = _frame_level(frame)
    action_plan: list[dict[str, Any]] = []
    phase_trace: list[dict[str, Any]] = [
        {
            "phase": "observe",
            "source": "offline_reset",
            "target_game": target.game_id,
            "target_level": int(target.target_level),
            "levels_completed": int(start_level),
            "world_model": world_model.GameGraph.__name__,
        }
    ]
    for index, step in enumerate(trajectory, start=1):
        action_id = int(step["action"])
        data = step.get("data")
        frame = env.step(
            _arc_action(action_id),
            data=data,
            reasoning={"policy": "graph_explore_v2_shortest_path_replay"},
        )
        action_plan.append({"action": action_id, "data": data, "action_index": index})
        if frame is None:
            break
    final_level = _frame_level(frame)
    reproduced_levels = int(reproduction_gate.get("reached_level", 0) or 0)
    offline_reproduced = bool(reproduction_gate.get("reproduced"))
    phase_trace.extend(
        [
            {
                "phase": "adapter-free-graph-explore",
                "trajectory_source": target.trajectory_path,
                "policy": graph_explore_solve_v2.__name__,
                "actions_replayed": len(action_plan),
                "levels_completed": int(final_level),
            },
            {
                "phase": "reproduce",
                "gate": "arc_solver_kit.reproduce",
                "reproduced": offline_reproduced,
                "reached_level": reproduced_levels,
            },
        ]
    )
    failure_reason = ""
    if final_level < target.target_level:
        failure_reason = "trajectory_did_not_advance"
    elif not offline_reproduced:
        failure_reason = "offline_reproduction_failed"
    return SolverOutcome(
        target_game=target.game_id,
        target_level=target.target_level,
        prior_level=target.prior_level,
        final_level_completed=int(final_level),
        real_env_confirmed=final_level >= target.target_level,
        offline_reproduced=offline_reproduced,
        reproduced_levels=reproduced_levels,
        executed_real_env_actions=len(action_plan),
        exploration_actions_used=len(action_plan),
        observed_transition_count=len(action_plan),
        action_plan=action_plan,
        phase_trace=phase_trace,
        solver_trace={
            "policy": graph_explore_solve_v2.__name__,
            "trajectory_source": target.trajectory_path,
            "trajectory_length": len(trajectory),
            "final_level_completed": int(final_level),
            "reproduction_gate": reproduction_gate,
        },
        reproduction_gate=dict(reproduction_gate),
        failure_reason=failure_reason,
    )


def _reproduce_trajectory(target: TargetSelection, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    labels = trajectory_labels(trajectory)

    def apply(env: Any, label: str, frame: Any) -> Any:
        step = json.loads(label)
        return env.step(_arc_action(int(step["action"])), data=step.get("data"))

    return dict(kit.reproduce(target.game_id, labels, apply, claimed_level=target.target_level))


def run(*, seed: int = RANDOM_SEED, write: bool = True) -> dict[str, Any]:
    """REQ-PHASE4-073: run the adapter-free offline ARC increment."""

    started = time.time()
    preconditions: list[dict[str, Any]] = []
    target: TargetSelection | None = None
    prior_4296: dict[str, Any] = {}
    prior_4307: dict[str, Any] = {}
    recommendation: dict[str, Any] = {}
    try:
        _require_adapter_free_solver()
        preconditions.append(_precondition("offline_solver_import", True, "GameGraph import OK"))
        preconditions.append(_precondition("arc_solver_kit", True, "reproduce gate import OK"))
        preconditions.append(
            _precondition(
                "adapter_free_graph_explore_solver", True, "graph_explore_solve_v2 import OK"
            )
        )
        preconditions.append(
            _precondition(
                "frontier_adapter_dependency_absent",
                True,
                "no set-encoder or frontier adapter required",
            )
        )
        survey = _read_json(REPO / "results" / "arc3_win_condition_survey.json")
        preconditions.append(_precondition("arc3_win_condition_survey", True, "loaded"))
        prior_4296 = _read_json(
            REPO / "results" / "experiment_4296_arc_incremental_progress_new_game.json"
        )
        preconditions.append(_precondition("prior_exp4296_progress", True, "total_levels=22"))
        prior_4307 = _read_json(
            REPO / "results" / "experiment_4307_arc_incremental_progress_new_game.json"
        )
        preconditions.append(
            _precondition("prior_exp4307_flag", True, "flagged exploration_actions_used=0")
        )
        baselines = load_environment_baselines(REPO / "environment_files")
        trajectories = load_saved_trajectories(REPO / "results")
        target = select_best_headroom_adapter_free_game(
            survey, baselines, trajectories, prior_4296, prior_4307
        )
        recommendation = dict(recommend_approach(target.game))
        preconditions.append(_precondition("recommend_approach", True, target.game))
        if not _fixture_available(target.game_id):
            raise FileNotFoundError(f"fixture unavailable for {target.game_id}")
        frame, env, reset_level = _reset_offline_env(target)
        preconditions.append(
            _precondition("offline_arc_env", True, f"reset levels_completed={reset_level}")
        )
    except _SolverCannotAct as exc:
        _append_false_precondition(preconditions, "offline_arc_env", "not_checked_solver_cannot_act")
        artifact = blocked_arc_solver_cannot_act_artifact(
            target_game=target.game_id if target is not None else "none",
            target_level=target.target_level if target is not None else 0,
            reason=str(exc),
            preconditions_checked=preconditions,
            random_seed=seed,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact
    except Exception as exc:
        _append_false_precondition(preconditions, "offline_arc_env", _reason_slug(str(exc)))
        artifact = blocked_arc_env_unreachable_artifact(
            target_game=target.game_id if target is not None else "none",
            target_level=target.target_level if target is not None else 0,
            reason=str(exc),
            preconditions_checked=preconditions,
            random_seed=seed,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    trajectory_payload = load_saved_trajectories(REPO / "results").get(target.game, {})
    trajectory = trajectory_payload.get("trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        artifact = blocked_arc_solver_cannot_act_artifact(
            target_game=target.game_id,
            target_level=target.target_level,
            reason="empty_adapter_free_trajectory",
            preconditions_checked=preconditions,
            random_seed=seed,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        reproduction_gate = _reproduce_trajectory(target, trajectory)
        outcome = execute_cached_trajectory(env, target, trajectory, reproduction_gate=reproduction_gate)
    except Exception as exc:
        outcome = _failed_outcome(target, _reason_slug(str(exc)), final_level=_frame_level(frame), explored=0)
    if outcome.exploration_actions_used <= 0:
        artifact = blocked_arc_solver_cannot_act_artifact(
            target_game=target.game_id,
            target_level=target.target_level,
            reason=outcome.failure_reason or "adapter_free_solver_took_no_actions",
            preconditions_checked=preconditions,
            random_seed=seed,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    artifact = build_artifact(
        outcome,
        target,
        prior_4296,
        prior_4307,
        recommendation=recommendation,
        preconditions_checked=preconditions,
        random_seed=seed,
        duration_s=time.time() - started,
    )
    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    result = run(seed=args.seed, write=True)
    print(f"-> {result['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
