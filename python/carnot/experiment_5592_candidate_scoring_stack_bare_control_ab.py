"""Experiment 5592: does our live candidate-scoring stack (candidate_router,
DAgger value head, goal-energy candidate guidance, navigation-cost tiebreak)
earn its keep, measured with the ablation methodology the 3rd-place ARC-AGI-3
team ("forge") used on their own architecturally-equivalent arbiter slot?

Context: docs/research-notes/arc-agi3-milestone1-winners-sota-ingestion-
2026-07-11.md (O2) found forge's `_select_candidate_with_arbiter` -- a
second-LLM-call candidate judge -- was DISABLED in their winning config for
cost, kept only as a hand-tuned static-score fallback. That is architecturally
the exact slot our own verifier-routed search already fills, except our
stack is real (a real cross-game-trained discriminative router + a
DAgger-trained value head + goal-energy candidate guidance), not a
disabled-for-cost LLM judge. The codebase already defines a
`bare_control_config` (SUBMITTED_AGENT_CONFIG["bare_control_config"]) --
the exact on/off toggle forge's own ablation used -- but no experiment had
ever run it against the real full-stack default and reported the delta.

This is NOT a new scorer build (per the task's own framing) -- a controlled
A/B, matched action budget, matched games, FULL stack (the real live
default, unmodified) vs BARE control (candidate_router=None,
goal_bias=None, goal_candidate_guidance=False, navigation_cost_tiebreak=
False, action_effect_expansion_prior=False, target_levels=1,
value_weight=0.0 -- the exact bare_control_config knobs mapped to their
real E3AgentPolicy constructor kwargs), reporting level-up rate and
action-efficiency delta. Tier-3 LLM induction is disabled
(CARNOT_ARC_DISABLE_INDUCTION=1) so this isolates the candidate-SELECTION
axis forge's ablation targeted, not induction.

operator_override: "2026-07-12 operator directive (standing): explicit
decision to port the energy-scorer opportunity into our own E3AgentPolicy
stack rather than fork forge's codebase -- not a doomed-rerun risk, this is
new measurement work with no prior attempt on file." (per
ops/known-issues.md task 12's own override text, carried forward here since
this task matches that scope exactly.)

Spec refs: REQ-ARC-FCP-5592, SCENARIO-ARC-FCP-5592-STACK-VS-BARE-DELTA.
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

EXPERIMENT_ID = "experiment_5592_candidate_scoring_stack_bare_control_ab"
RESULT_RELATIVE_PATH = "results/experiment_5592_candidate_scoring_stack_bare_control_ab.json"
SCHEMA = "carnot.exp5592.candidate_scoring_stack_bare_control_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5592
DEFAULT_BUDGET = 200
DEFAULT_ROSTER = (
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
    "budget",
    "bare_control_kwargs",
    "full_stack_results",
    "bare_control_results",
    "levels_gained_full_stack_total",
    "levels_gained_bare_control_total",
    "per_game_levels_delta",
    "efficiency_full_stack_total",
    "efficiency_bare_control_total",
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
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- CARNOT_ARC_DISABLE_INDUCTION=1 isolates the candidate-selection axis forge's ablation targeted, not tier-3 induction"
    },
    "bare_control_kwargs": {
        "principle": "the real E3AgentPolicy constructor kwargs mapped from SUBMITTED_AGENT_CONFIG['bare_control_config'] -- documents exactly what was ablated, matching forge's own on/off toggle"
    },
    "levels_gained_headroom_present": {
        "principle": "CLAUDE.md FALSE_NEGATIVE_RISK discipline -- a null delta is only interpretable if at least one arm shows nonzero levels_gained somewhere on the roster"
    },
    "efficiency_full_stack_total": {
        "principle": "sum of the leaderboard harness's own per-game efficiency score (matches arc_agi.scorecard.EnvironmentScoreCalculator) -- the action-efficiency half of forge's own reported metric, not just level count"
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
            env = arc.make(DEFAULT_ROSTER[0], scorecard_id=arc.open_scorecard())
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
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    budget: int = DEFAULT_BUDGET,
    root: Path = REPO_ROOT,
) -> JsonDict:
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    started_at = time.time()
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
            "budget": int(budget),
            "bare_control_kwargs": {k: v for k, v in BARE_CONTROL_KWARGS.items()},
            "full_stack_results": {},
            "bare_control_results": {},
            "levels_gained_full_stack_total": 0,
            "levels_gained_bare_control_total": 0,
            "per_game_levels_delta": {},
            "efficiency_full_stack_total": 0.0,
            "efficiency_bare_control_total": 0.0,
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
    for game in roster:
        delta = full_stack_results[game].get("levels", 0) - bare_control_results[game].get(
            "levels", 0
        )
        per_game_deltas[game] = delta

    total_level_delta = levels_gained_full_stack_total - levels_gained_bare_control_total
    total_efficiency_delta = efficiency_full_stack_total - efficiency_bare_control_total
    any_headroom = any(
        full_stack_results[g].get("levels", 0) > 0 or bare_control_results[g].get("levels", 0) > 0
        for g in roster
    )

    if not any_headroom:
        verdict = "complete: candidate_stack_no_headroom_on_roster"
    elif total_level_delta > 0:
        verdict = (
            f"complete: candidate_stack_beats_bare_control_{levels_gained_bare_control_total}_"
            f"to_{levels_gained_full_stack_total}_levels"
        )
    elif total_level_delta < 0:
        verdict = "complete: candidate_stack_regression_below_bare_control"
    elif total_efficiency_delta > 0:
        verdict = "complete: candidate_stack_ties_levels_but_more_efficient_than_bare_control"
    elif total_efficiency_delta < 0:
        verdict = "complete: candidate_stack_ties_levels_but_less_efficient_than_bare_control"
    else:
        verdict = "complete: candidate_stack_honest_null_headroom_present_no_delta"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "roster": list(roster),
        "budget": int(budget),
        "bare_control_kwargs": {k: v for k, v in BARE_CONTROL_KWARGS.items()},
        "full_stack_results": full_stack_results,
        "bare_control_results": bare_control_results,
        "levels_gained_full_stack_total": levels_gained_full_stack_total,
        "levels_gained_bare_control_total": levels_gained_bare_control_total,
        "per_game_levels_delta": per_game_deltas,
        "efficiency_full_stack_total": round(efficiency_full_stack_total, 6),
        "efficiency_bare_control_total": round(efficiency_bare_control_total, 6),
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
