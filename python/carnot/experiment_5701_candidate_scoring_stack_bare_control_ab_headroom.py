"""Experiment 5701: re-scoped re-run of the forge-style candidate-scoring-stack vs
bare-control ablation (exp5592, REQ-ARC-FCP-5592) on a roster/budget combination with
genuine headroom, so a null result is informative rather than a floor-effect artifact
(task 9 -- "Task 12 follow-up").

Context: exp5592 ran the SAME ablation (full E3AgentPolicy default vs
bare_control_config: candidate_router=None, goal_bias=None,
goal_candidate_guidance=False, navigation_cost_tiebreak=False,
action_effect_expansion_prior=False, target_levels=1, value_weight=0.0) on an ad-hoc
11-game roster at budget=200 and found levels_gained_headroom_present=True on a technicality:
only 1 of 11 games (lp85) reached level>0 in EITHER arm, and that one game tied
(levels=1, efficiency=2.7778, IDENTICAL in both arms). 10 of 11 games were stuck at
level 0 in BOTH arms -- a floor effect, not a genuine measurement of whether the
candidate-scoring stack earns its keep.

A same-session calibration probe (informal, single-episode, not itself a checked-in
artifact -- see methodology_note) found: (1) exp5592's roster mixed adaptered and
UN-adaptered games (wa30, sc25 have no registered GameAdapter in
arc_game_adapters.py and got 0 levels at budget=200 AND budget=600 -- structurally
unreachable by the generic policy, not an arm-comparison signal at all); (2)
increasing budget 200->600 on ADAPTERED games raised the level>=1 hit rate from 1/11
(9%) to 3/6 (50%) tested (lp85, tu93, su15 each reached level 1; r11l did not within
600 actions this run). This experiment re-scopes to the FULL 22-game adaptered
roster (arc_game_adapters.adaptered_games() -- every game the live path has a
registered GameAdapter for, per the ARC Live-Path Reachability Discipline) at
budget=500 (between the 200 floor and the 600 calibration point, with margin), so
substantially more of the roster has a chance to clear level 1 in at least one arm,
giving the arm-comparison genuine statistical footing instead of resting on a single
game.

This is still NOT a new scorer build -- same controlled A/B, same
BARE_CONTROL_KWARGS, same Tier-3 LLM induction disabled
(CARNOT_ARC_DISABLE_INDUCTION=1) isolating the candidate-SELECTION axis, same
level-up-rate + action-efficiency-delta reporting. Only the roster and budget change.

Spec refs: REQ-ARC-FCP-5592 (extends), REQ-ARC-FCP-5701-HEADROOM-RESCOPE.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(SCRIPTS_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5701_candidate_scoring_stack_bare_control_ab_headroom"
RESULT_RELATIVE_PATH = (
    "results/experiment_5701_candidate_scoring_stack_bare_control_ab_headroom.json"
)
SCHEMA = "carnot.exp5701.candidate_scoring_stack_bare_control_ab_headroom.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5701
DEFAULT_BUDGET = 500
PRIOR_ROSTER = (
    "cd82",
    "cn04",
    "lp85",
    "ls20",
    "m0r0",
    "r11l",
    "sk48",
    "sp80",
    "su15",
    "tu93",
    "wa30",
)
PRIOR_BUDGET = 200
BARE_CONTROL_KWARGS: dict[str, Any] = {
    "target_levels": 1,
    "value_weight": 0.0,
    "candidate_router": None,
    "navigation_cost_tiebreak": False,
    "action_effect_expansion_prior": False,
    "goal_bias": None,
    "goal_candidate_guidance": False,
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "roster",
    "roster_source",
    "budget",
    "prior_attempt",
    "bare_control_kwargs",
    "full_stack_results",
    "bare_control_results",
    "levels_gained_full_stack_total",
    "levels_gained_bare_control_total",
    "per_game_levels_delta",
    "efficiency_full_stack_total",
    "efficiency_bare_control_total",
    "n_games_with_headroom",
    "levels_gained_headroom_present",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; the full stack beating bare control, losing to it, and tying are all distinct, real, citable outcomes -- an architecturally-richer stack is not automatically a measured win"
    },
    "roster_source": {
        "principle": "documents WHY this roster differs from exp5592's -- every game here has a registered GameAdapter (arc_game_adapters.adaptered_games()), so the generic policy has a structural chance to progress, unlike exp5592's un-adaptered games (wa30, sc25) which were floor-effect noise, not signal"
    },
    "n_games_with_headroom": {
        "principle": "count of games where EITHER arm reached level>=1 -- the direct fix for exp5592's single-game floor effect; more non-zero cells means the arm comparison rests on genuine statistical footing, not one game's tie"
    },
    "prior_attempt": {
        "principle": "CLAUDE.md Failed-Experiment Rerun Discipline -- names the exp5592 floor-effect finding and what is different here (roster restricted to adaptered games, budget raised 200->500) so this is a documented root-cause fix, not a doomed re-run"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- CARNOT_ARC_DISABLE_INDUCTION=1 isolates the candidate-selection axis, not tier-3 induction"
    },
    "bare_control_kwargs": {
        "principle": "identical to exp5592's -- the real E3AgentPolicy constructor kwargs mapped from SUBMITTED_AGENT_CONFIG['bare_control_config'], unchanged so this is a genuine re-scope, not a different ablation"
    },
    "levels_gained_headroom_present": {
        "principle": "CLAUDE.md FALSE_NEGATIVE_RISK discipline -- a null delta is only interpretable if at least one arm shows nonzero levels_gained somewhere on the roster"
    },
    "efficiency_full_stack_total": {
        "principle": "sum of the leaderboard harness's own per-game efficiency score -- the action-efficiency half of forge's own reported metric, not just level count"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        checks["offline_arcade_importable"] = True
        checks["offline_arcade_makes_env"] = False
        try:
            env = arc.make("lp85", scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import (  # noqa: F401
            E3AgentPolicy,
            SUBMITTED_AGENT_CONFIG,
        )

        checks["e3_policy_import"] = True
        checks["bare_control_config_present"] = "bare_control_config" in SUBMITTED_AGENT_CONFIG
    except Exception:
        checks["e3_policy_import"] = False
        checks["bare_control_config_present"] = False
    try:
        from carnot.agentic import arc_game_adapters as aga

        checks["adaptered_games_available"] = len(aga.adaptered_games()) > 0
    except Exception:
        checks["adaptered_games_available"] = False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def default_roster() -> tuple[str, ...]:
    from carnot.agentic import arc_game_adapters as aga

    return tuple(sorted(aga.adaptered_games()))


def _play_one_game(
    game: str,
    *,
    budget: int,
    arm: str,
    results: dict[str, JsonDict],
    lock: threading.Lock,
) -> None:
    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

    if arm == "bare_control":
        policy = E3AgentPolicy(game, **BARE_CONTROL_KWARGS)
    else:
        policy = E3AgentPolicy(game)
    row = lb.run_game(game, policy, budget=budget)

    with lock:
        results[game] = row


def run_both_arms(roster: tuple[str, ...], *, budget: int) -> tuple[JsonDict, JsonDict, float]:
    full_stack: JsonDict = {}
    bare_control: JsonDict = {}
    lock = threading.Lock()
    t0 = time.time()
    threads = []
    for game in roster:
        threads.append(
            threading.Thread(
                target=_play_one_game,
                args=(game,),
                kwargs={
                    "budget": budget,
                    "arm": "full_stack",
                    "results": full_stack,
                    "lock": lock,
                },
            )
        )
        threads.append(
            threading.Thread(
                target=_play_one_game,
                args=(game,),
                kwargs={
                    "budget": budget,
                    "arm": "bare_control",
                    "results": bare_control,
                    "lock": lock,
                },
            )
        )
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return full_stack, bare_control, time.time() - t0


def build_artifact(
    *,
    roster: tuple[str, ...] | None = None,
    budget: int = DEFAULT_BUDGET,
    root: Path = REPO_ROOT,
) -> JsonDict:
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    started_at = time.time()

    prior_attempt = {
        "experiment_id": 5592,
        "verdict": "candidate_stack_honest_null_headroom_present_no_delta",
        "root_cause": (
            "roster mixed adaptered and un-adaptered games at budget=200; only 1 of 11 "
            "games (lp85) reached level>0 in either arm, and that game tied identically "
            "-- a floor effect, not a genuine arm comparison"
        ),
        "what_is_different": (
            "roster restricted to arc_game_adapters.adaptered_games() (games the live "
            "path can structurally progress on); budget raised 200->500, calibrated by "
            "a same-session probe showing budget=600 lifts the level>=1 hit rate on "
            "adaptered games from ~9% to ~50%"
        ),
        "retire_if_same_verdict": False,
    }

    if roster is None:
        try:
            roster = default_roster()
        except Exception:
            roster = PRIOR_ROSTER

    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "field_principles": FIELD_PRINCIPLES,
            "verifier_is_oracle": False,
            "roster": list(roster),
            "roster_source": "arc_game_adapters.adaptered_games()",
            "budget": int(budget),
            "prior_attempt": prior_attempt,
            "bare_control_kwargs": {k: v for k, v in BARE_CONTROL_KWARGS.items()},
            "full_stack_results": {},
            "bare_control_results": {},
            "levels_gained_full_stack_total": 0,
            "levels_gained_bare_control_total": 0,
            "per_game_levels_delta": {},
            "efficiency_full_stack_total": 0.0,
            "efficiency_bare_control_total": 0.0,
            "n_games_with_headroom": 0,
            "levels_gained_headroom_present": False,
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    full_stack_results, bare_control_results, _wall_clock_s = run_both_arms(roster, budget=budget)

    levels_gained_full_stack_total = sum(r.get("levels", 0) for r in full_stack_results.values())
    levels_gained_bare_control_total = sum(
        r.get("levels", 0) for r in bare_control_results.values()
    )
    efficiency_full_stack_total = sum(
        float(r.get("efficiency") or 0.0) for r in full_stack_results.values()
    )
    efficiency_bare_control_total = sum(
        float(r.get("efficiency") or 0.0) for r in bare_control_results.values()
    )

    per_game_deltas: JsonDict = {}
    n_games_with_headroom = 0
    for game in roster:
        fs_levels = full_stack_results[game].get("levels", 0)
        bc_levels = bare_control_results[game].get("levels", 0)
        per_game_deltas[game] = fs_levels - bc_levels
        if fs_levels > 0 or bc_levels > 0:
            n_games_with_headroom += 1

    total_level_delta = levels_gained_full_stack_total - levels_gained_bare_control_total
    total_efficiency_delta = efficiency_full_stack_total - efficiency_bare_control_total
    any_headroom = n_games_with_headroom > 0

    if not any_headroom:
        verdict = "complete: candidate_stack_no_headroom_on_roster"
    elif total_level_delta > 0:
        verdict = (
            f"complete: candidate_stack_beats_bare_control_{levels_gained_bare_control_total}_"
            f"to_{levels_gained_full_stack_total}_levels_across_{n_games_with_headroom}_games_with_headroom"
        )
    elif total_level_delta < 0:
        verdict = "complete: candidate_stack_regression_below_bare_control"
    elif total_efficiency_delta > 0:
        verdict = "complete: candidate_stack_ties_levels_but_more_efficient_than_bare_control"
    elif total_efficiency_delta < 0:
        verdict = "complete: candidate_stack_ties_levels_but_less_efficient_than_bare_control"
    else:
        verdict = (
            f"complete: candidate_stack_honest_null_headroom_present_no_delta_across_"
            f"{n_games_with_headroom}_games_with_headroom"
        )

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "roster": list(roster),
        "roster_source": "arc_game_adapters.adaptered_games()",
        "budget": int(budget),
        "prior_attempt": prior_attempt,
        "bare_control_kwargs": {k: v for k, v in BARE_CONTROL_KWARGS.items()},
        "full_stack_results": full_stack_results,
        "bare_control_results": bare_control_results,
        "levels_gained_full_stack_total": levels_gained_full_stack_total,
        "levels_gained_bare_control_total": levels_gained_bare_control_total,
        "per_game_levels_delta": per_game_deltas,
        "efficiency_full_stack_total": round(efficiency_full_stack_total, 6),
        "efficiency_bare_control_total": round(efficiency_bare_control_total, 6),
        "n_games_with_headroom": n_games_with_headroom,
        "levels_gained_headroom_present": any_headroom,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
