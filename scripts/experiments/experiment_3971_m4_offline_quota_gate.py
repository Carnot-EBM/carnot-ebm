"""Exp 3971: ARC-AGI-3 M4 offline quota-gate readiness package.

Spec traces: REQ-PHASE4-016, SCENARIO-PHASE4-016.

This script never submits online. It only measures the offline hybrid policy
against same-game no-induction baselines and emits the operator-only readiness
verdict for a future scored run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import arc3_offline_eval  # noqa: E402


OUTFILE = REPO / "results" / "experiment_3971_m4_offline_quota_gate.json"
INFERENCE_SUBSTRATE = "offline_arc_agi3_hybrid_policy_quota_gate_local_env"
DEFAULT_RANDOM_SEED = 3971
DEFAULT_BUDGET_FACTOR = 3.0
DEFAULT_BUDGET_CAP = 3000
PRIOR_SUBMITTED_LEVELS = 0
SOLVED_GAME_EXTRAS = ("sc25-635fd71a",)
DEFAULT_START_HERE_GAMES = (
    "vc33-5430563c",
    "lp85-305b61c3",
    "sb26-7fbdac44",
    "tu93-0768757b",
    "s5i5-18d95033",
    "bp35-0a0ad940",
    "r11l-495a7899",
    "su15-1944f8ab",
    "sc25-635fd71a",
)
DOCUMENTED_SOTA_CONTEXT = {
    "ewm_rhae_pct": 58.12,
    "ewm_games_fully_solved": 15,
    "graph_explore_levels": "median 30/52 across 6 games",
    "frontier_llm_rhae_pct_lt": 0.4,
    "trm_arc_agi3_result": None,
    "source": "research-references.md 2026-06-09 ARC-AGI-3 SOTA refresh",
}


def _artifact_levels(artifact: dict[str, Any]) -> int:
    return int(artifact.get("ACCURACY_total_levels_solved", 0) or 0)


def _artifact_efficiency(artifact: dict[str, Any]) -> float | None:
    value = artifact.get("EFFICIENCY_mean_action_ratio_on_solved")
    return None if value is None else float(value)


def load_start_here_games(repo: Path = REPO) -> list[str]:
    path = repo / "results" / "arc_agi3_game_characterization.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        games = [str(row["game_id"]) for row in data.get("start_here_top8", []) if row.get("game_id")]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        games = list(DEFAULT_START_HERE_GAMES)

    for game_id in SOLVED_GAME_EXTRAS:
        if game_id not in games:
            games.append(game_id)
    return games


def check_offline_env_available() -> bool:
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arcade = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
    if not arcade.get_environments():
        return False
    env = arcade.make("r11l-495a7899")
    env.reset()
    return True


def write_artifact(artifact: dict[str, Any], path: Path = OUTFILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_blocked_artifact(*, games: list[str], seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_3971_m4_offline_quota_gate",
        "title": "arc3_m4_offline_quota_gate",
        "hybrid_accuracy_levels_solved": 0,
        "baseline_accuracy_levels_solved": 0,
        "hybrid_efficiency_ratio": None,
        "quota_gate_cleared": False,
        "scored_run_ready_for_operator": False,
        "random_seed": int(seed),
        "honest_verdict": "blocked_arc_offline_env_unavailable",
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "games": games,
        "prior_submitted_levels": PRIOR_SUBMITTED_LEVELS,
        "submitted_to_leaderboard": False,
        "documented_sota_context": DOCUMENTED_SOTA_CONTEXT,
        "operator_only_note": "Offline readiness package only; no online/scored ARC-AGI-3 submission was run.",
    }


def build_readiness_artifact(
    *,
    games: list[str],
    seed: int,
    budget_factor: float,
    budget_cap: int,
    hybrid: dict[str, Any],
    random_baseline: dict[str, Any],
    object_click_baseline: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    hybrid_levels = _artifact_levels(hybrid)
    random_levels = _artifact_levels(random_baseline)
    object_click_levels = _artifact_levels(object_click_baseline)
    baseline_levels = max(random_levels, object_click_levels)
    beats_prior = hybrid_levels > PRIOR_SUBMITTED_LEVELS
    beats_baseline = hybrid_levels > baseline_levels
    quota_gate_cleared = beats_prior and beats_baseline
    target_to_clear = max(PRIOR_SUBMITTED_LEVELS + 1, baseline_levels + 1)
    gap_to_clearing = max(0, target_to_clear - hybrid_levels)

    if quota_gate_cleared:
        verdict = (
            f"success: quota_gate_cleared_hybrid_levels{hybrid_levels}"
            f"_baseline{baseline_levels}_prior{PRIOR_SUBMITTED_LEVELS}_operator_ready"
        )
    elif not beats_prior:
        verdict = f"complete: quota_gate_not_cleared_prior0_gap{gap_to_clearing}"
    else:
        verdict = f"complete: quota_gate_not_cleared_baseline_gap{gap_to_clearing}"

    return {
        "experiment": "experiment_3971_m4_offline_quota_gate",
        "title": "arc3_m4_offline_quota_gate",
        "hybrid_accuracy_levels_solved": hybrid_levels,
        "baseline_accuracy_levels_solved": baseline_levels,
        "hybrid_efficiency_ratio": _artifact_efficiency(hybrid),
        "quota_gate_cleared": quota_gate_cleared,
        "scored_run_ready_for_operator": quota_gate_cleared,
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "games": games,
        "n_games": len(games),
        "budget_factor": float(budget_factor),
        "budget_cap": int(budget_cap),
        "prior_submitted_levels": PRIOR_SUBMITTED_LEVELS,
        "gap_to_clearing_levels": int(gap_to_clearing),
        "random_baseline_accuracy_levels_solved": random_levels,
        "object_click_baseline_accuracy_levels_solved": object_click_levels,
        "hybrid_policy_artifact": hybrid,
        "random_baseline_artifact": random_baseline,
        "object_click_baseline_artifact": object_click_baseline,
        "documented_sota_context": DOCUMENTED_SOTA_CONTEXT,
        "submitted_to_leaderboard": False,
        "operator_only_note": "Offline readiness package only; no online/scored ARC-AGI-3 submission was run.",
    }


def run(
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    budget_factor: float = DEFAULT_BUDGET_FACTOR,
    budget_cap: int = DEFAULT_BUDGET_CAP,
    write: bool = True,
    output_path: Path = OUTFILE,
    evaluator: Callable[..., dict[str, Any]] = arc3_offline_eval.run,
    precondition_checker: Callable[[], bool] = check_offline_env_available,
) -> dict[str, Any]:
    started = time.time()
    games = load_start_here_games(REPO)

    try:
        offline_available = bool(precondition_checker())
    except Exception:
        offline_available = False
    if not offline_available:
        artifact = build_blocked_artifact(games=games, seed=seed, duration_s=time.time() - started)
        if write:
            write_artifact(artifact, output_path)
        return artifact

    common = {
        "n_games": len(games),
        "budget_factor": budget_factor,
        "budget_cap": budget_cap,
        "seed": seed,
        "write": write,
        "games": games,
    }
    hybrid = evaluator(policy_name="hybrid", **common)
    random_baseline = evaluator(policy_name="random", **common)
    object_click_baseline = evaluator(policy_name="object_click", **common)
    artifact = build_readiness_artifact(
        games=games,
        seed=seed,
        budget_factor=budget_factor,
        budget_cap=budget_cap,
        hybrid=hybrid,
        random_baseline=random_baseline,
        object_click_baseline=object_click_baseline,
        duration_s=time.time() - started,
    )
    if write:
        write_artifact(artifact, output_path)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--budget-factor", type=float, default=DEFAULT_BUDGET_FACTOR)
    parser.add_argument("--budget-cap", type=int, default=DEFAULT_BUDGET_CAP)
    args = parser.parse_args()
    artifact = run(seed=args.seed, budget_factor=args.budget_factor, budget_cap=args.budget_cap)
    print(f"-> {artifact['honest_verdict']}")
    print(
        "   hybrid_levels="
        f"{artifact['hybrid_accuracy_levels_solved']} baseline_levels={artifact['baseline_accuracy_levels_solved']} "
        f"efficiency={artifact['hybrid_efficiency_ratio']} operator_ready={artifact['scored_run_ready_for_operator']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
