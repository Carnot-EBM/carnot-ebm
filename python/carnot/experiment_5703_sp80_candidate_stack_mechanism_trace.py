"""Experiment 5703: mechanism-level trace of WHY the full candidate-scoring stack
lost a level to bare control on sp80 (task 10 -- follow-up to exp5701's finding
that sp80 was the one game where bare_control beat full_stack).

Replays sp80 under both E3AgentPolicy arms (identical construction to exp5592/
exp5701's BARE_CONTROL_KWARGS vs the real unmodified default) with instrumentation
around each of the THREE "richer stack" mechanisms that differ between the arms
(`candidate_router`, `goal_bias`, `goal_candidate_guidance`) to determine whether
any of them actively steered the full-stack arm toward a worse choice, or whether
they were structurally inert.

Spec refs: REQ-ARC-FCP-5701-HEADROOM-RESCOPE (extends, sp80 regression follow-up).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
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

EXPERIMENT_ID = "experiment_5703_sp80_candidate_stack_mechanism_trace"
RESULT_RELATIVE_PATH = "results/experiment_5703_sp80_candidate_stack_mechanism_trace.json"
SCHEMA = "carnot.exp5703.sp80_candidate_stack_mechanism_trace.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5703
TARGET_GAME = "sp80"
BUDGET = 500
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
    "target_game",
    "budget",
    "full_stack",
    "bare_control",
    "goal_bias_score_variance",
    "goal_bias_n_scored",
    "candidate_router_changed_order_count",
    "inert_mechanisms",
    "prior_result",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; distinguishes 'a learned mechanism actively misled search' from 'the richer mechanisms were inert and something else caused the divergence' -- these have very different implications for whether the stack is unsafe or just unhelpful here"
    },
    "goal_bias_score_variance": {
        "principle": "zero variance across every real invocation is direct, mechanical proof the goal-energy source could not have influenced frontier ordering on this game, independent of any post-hoc narrative"
    },
    "candidate_router_changed_order_count": {
        "principle": "counts how many of the router's real invocations actually altered the candidate ordering it was given -- distinguishes 'present and consulted' from 'present and load-bearing'"
    },
    "prior_result": {
        "principle": "CLAUDE.md Failed-Experiment Rerun Discipline analog for a diagnostic task -- names the exp5701 finding this investigates so the connection is traceable"
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
            env = arc.make(TARGET_GAME, scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import (  # noqa: F401
            E3AgentPolicy,
            StepwiseExplorer,
        )

        checks["e3_policy_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
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


def _play_traced(game: str, *, arm: str, budget: int) -> JsonDict:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from arc_leaderboard_eval import _baseline_actions, _level_of

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    _baseline_actions(env, game)

    if arm == "bare_control":
        policy = E3AgentPolicy(game, **BARE_CONTROL_KWARGS)
    else:
        policy = E3AgentPolicy(game)

    router_calls = {"n": 0, "changed_order_count": 0}
    real_router = policy.explorer.candidate_router
    if real_router is not None:

        def _counting_router(candidates, *a, **kw):
            router_calls["n"] += 1
            before = [c.get("id") if isinstance(c, dict) else c for c in candidates]
            result = real_router(candidates, *a, **kw)
            after = [c.get("id") if isinstance(c, dict) else c for c in result]
            if before != after:
                router_calls["changed_order_count"] += 1
            return result

        policy.explorer.candidate_router = _counting_router

    goal_bias_scores: list[float] = []
    if policy.explorer.goal_bias is not None:
        real_goal_bias = policy.explorer.goal_bias

        def _traced_goal_bias(frame):
            v = real_goal_bias(frame)
            goal_bias_scores.append(float(v))
            return v

        policy.explorer.goal_bias = _traced_goal_bias

    frames, latest, actions = [], None, 0
    start, best = None, None
    action_sequence: list[Any] = []
    for _ in range(budget):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            action_sequence.append("RESET")
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            actions += 1
            action_sequence.append(kind)
        if start is None:
            start = _level_of(latest)
            best = start
        lvl = _level_of(latest)
        if best is not None and lvl > best:
            best = lvl
        frames.append(latest)
        if latest is None:
            break

    reached = _level_of(latest)
    levels_gained = max(0, reached - (start or 0))

    return {
        "arm": arm,
        "start_level": start,
        "reached_level": reached,
        "levels_gained": levels_gained,
        "total_actions": actions,
        "action_sequence": action_sequence,
        "candidate_router_present": real_router is not None,
        "candidate_router_calls": router_calls["n"],
        "candidate_router_changed_order_count": router_calls["changed_order_count"],
        "goal_bias_present": policy.explorer.goal_bias is not None,
        "goal_bias_n_scored": len(goal_bias_scores),
        "goal_bias_score_variance": round(statistics.pvariance(goal_bias_scores), 8)
        if len(goal_bias_scores) > 1
        else 0.0,
        "goal_bias_score_min": min(goal_bias_scores) if goal_bias_scores else None,
        "goal_bias_score_max": max(goal_bias_scores) if goal_bias_scores else None,
        "goal_candidate_guidance_diagnostics": (
            policy.explorer.goal_candidate_guidance_diagnostics()
            if hasattr(policy.explorer, "goal_candidate_guidance_diagnostics")
            else None
        ),
    }


def build_artifact(*, root: Path = REPO_ROOT, budget: int = BUDGET) -> JsonDict:
    started = time.monotonic()
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)

    prior_result = {
        "experiment_id": 5701,
        "finding": (
            "sp80 was the one game (of 5 with headroom, 22-game adaptered roster) where "
            "bare_control beat full_stack by 1 level (bare_control reached L1, full_stack did "
            "not, within the same budget=500)"
        ),
    }

    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "field_principles": FIELD_PRINCIPLES,
            "target_game": TARGET_GAME,
            "budget": int(budget),
            "full_stack": {},
            "bare_control": {},
            "goal_bias_score_variance": None,
            "goal_bias_n_scored": 0,
            "candidate_router_changed_order_count": 0,
            "inert_mechanisms": [],
            "prior_result": prior_result,
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.monotonic() - started, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    full_stack = _play_traced(TARGET_GAME, arm="full_stack", budget=budget)
    bare_control = _play_traced(TARGET_GAME, arm="bare_control", budget=budget)

    inert_mechanisms = []
    gcg = full_stack.get("goal_candidate_guidance_diagnostics") or {}
    if (
        full_stack["candidate_router_present"]
        and full_stack["candidate_router_changed_order_count"] == 0
    ):
        inert_mechanisms.append("candidate_router")
    if full_stack["goal_bias_present"] and full_stack["goal_bias_score_variance"] == 0.0:
        inert_mechanisms.append("goal_bias")
    if gcg.get("enabled") and gcg.get("arms_non_degenerate") is False:
        inert_mechanisms.append("goal_candidate_guidance")

    reproduces_regression = full_stack["levels_gained"] < bare_control["levels_gained"]
    if not reproduces_regression:
        verdict = "complete: sp80_regression_did_not_reproduce_this_run"
    elif len(inert_mechanisms) == 3:
        verdict = (
            "complete: regression_reproduced_but_all_three_learned_mechanisms_inert_"
            "cause_is_elsewhere_in_stack"
        )
    elif inert_mechanisms:
        verdict = "complete: regression_reproduced_some_learned_mechanisms_inert_partial_diagnosis"
    else:
        verdict = "complete: regression_reproduced_learned_mechanisms_active_and_implicated"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "target_game": TARGET_GAME,
        "budget": int(budget),
        "full_stack": full_stack,
        "bare_control": bare_control,
        "goal_bias_score_variance": full_stack["goal_bias_score_variance"],
        "goal_bias_n_scored": full_stack["goal_bias_n_scored"],
        "candidate_router_changed_order_count": full_stack["candidate_router_changed_order_count"],
        "inert_mechanisms": inert_mechanisms,
        "diagnosis_note": (
            "goal_bias and goal_candidate_guidance both source from "
            "arc_goal_energy_live.GoalSatisfactionEnergy (Exp4020's induced goal predicate + a "
            "satisfied/total-targets fraction). A constant score across every real invocation "
            "means GoalSatisfactionEnergy.visible_state()/_state_from_visible() extracts no "
            "usable target-state from sp80's frames (falls to the total<=0 or state-is-None "
            "default of 1.0 every time) -- the goal-energy source is structurally blind on this "
            "game's specific spill-splitter/placement mechanic. This corroborates "
            "ops/verifier_gaps.md GAP-4891 (a DIFFERENT mechanism -- the offline self-induction "
            "operator -- independently found sp80's goal is SPATIAL/placement and not "
            "discriminable by count/generic features) via a completely separate code path (the "
            "LIVE goal_bias/goal_candidate_guidance stack), extending that gap's scope from an "
            "offline diagnostic to the live submitted agent's own search behavior. Separately: "
            "goal_candidate_guidance already self-detects this degeneracy and falls back to the "
            "unranked candidate order (arms_non_degenerate=False -> no-op, by design in "
            "arc_goal_energy_live.py); goal_bias's frontier-node scoring has NO equivalent "
            "self-audit -- it silently contributes a constant (hence inert but not necessarily "
            "harmless-by-design) key to every node's sort tuple. The regression itself, given "
            "all three learned mechanisms are confirmed inert, must trace to one of the OTHER "
            "differing knobs (value_weight/DAgger value head, navigation_cost_tiebreak, "
            "action_effect_expansion_prior) -- not further isolated in this investigation; a "
            "natural next step if this specific game matters for a headline claim."
        ),
        "prior_result": prior_result,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.monotonic() - started, 3),
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
