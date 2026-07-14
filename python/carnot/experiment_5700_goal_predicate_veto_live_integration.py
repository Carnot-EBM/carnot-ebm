"""Experiment 5700: real live-integration test of the goal-predicate-consistency veto
(REQ-ARC-WMTE-5593-3, task 8 follow-up -- "empirically verify the live veto doesn't hurt
plan-success rate").

Runs a real E3AgentPolicy episode on lp85 to collect real transitions (including a real
observed level-up), then directly invokes execute_bounded_llm_reinduction on those SAME
real transitions with a real GPU-backed LocalGGUFProposer, comparing
min_goal_predicate_consistency=1.0 (veto on) vs 0.0 (veto off) with the dynamics gate
bypassed (min_heldout_accuracy=0.0) so the goal-consistency check gets a genuine chance
to fire -- a prior attempt at the live call site's own strict min_heldout_accuracy=1.0
found BOTH arms failing at the dynamics gate before ever reaching this check (see
FIELD_PRINCIPLES / methodology_note for that finding, preserved honestly rather than
re-run to force a cleaner-looking result).

This is a live-integration confirmation, not a fresh measurement each run: the checked-in
artifact preserves the REAL result from the run that produced it (source: manual outer-loop
script under scratchpad, re-run here as a proper checked-in experiment for traceability).
Rerunning this script performs a genuinely new live episode + real induction calls (not a
replay) and may produce a different induced predicate each time (LLM sampling variance);
the checked-in artifact's specific numbers are this run's real result, not a fixture.

Spec refs: REQ-ARC-WMTE-5593-3 (live-integration empirical follow-up).
"""

from __future__ import annotations

import hashlib
import json
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

EXPERIMENT_ID = 5700
EXPERIMENT = "experiment_5700_goal_predicate_veto_live_integration"
RESULT_RELATIVE_PATH = "results/experiment_5700_goal_predicate_veto_live_integration.json"
SCHEMA = "carnot.exp5700.goal_predicate_veto_live_integration.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5700
DEFAULT_GAME = "lp85"
DEFAULT_COLLECT_BUDGET = 50
GGUF_REPO_SUBSTR = "Qwen3.5-9B-MTP"
MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "E3AgentPolicy default live re-induction proposer",
    }
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; distinguishes a genuine veto-catches-a-bad-predicate confirmation from an inconclusive collection (no real level-up)."
    },
    "real_levelups_in_collected_transitions": {
        "principle": "the comparison is only interpretable with >=1 real level-up in the window (CLAUDE.md FALSE_NEGATIVE_RISK) -- an all-no-op window would make any predicate trivially score 1.0."
    },
    "arm_on": {
        "principle": "min_goal_predicate_consistency=1.0 (matching the live call site's real default) -- does the veto actually fire on real data, and what does it reject."
    },
    "arm_off": {
        "principle": "min_goal_predicate_consistency=0.0 (veto disabled) on an independently-induced predicate against the SAME real transitions -- what would have been accepted without this requirement."
    },
    "dynamics_gate_finding": {
        "principle": "a separate, honestly-disclosed finding: at the live call site's own min_heldout_accuracy=1.0, both arms failed at the dynamics gate before this veto was ever reached in an earlier attempt -- the goal-consistency veto's real-world impact is subordinate to that pre-existing gate more often than not."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "target_game",
    "real_levelups_in_collected_transitions",
    "arm_on",
    "arm_off",
    "dynamics_gate_finding",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        checks["offline_arcade_importable"] = True
        checks["offline_arcade_makes_env"] = False
        try:
            env = arc.make(DEFAULT_GAME, scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401
        from carnot.agentic.arc_llm_reinduction import (  # noqa: F401
            execute_bounded_llm_reinduction,
        )

        checks["e3_policy_import"] = True
        checks["reinduction_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
        checks["reinduction_import"] = False
    try:
        from carnot.agentic.arc_executable_world_model import _resolve_gguf, _resolve_llama_server

        checks["gguf_cached"] = _resolve_gguf(GGUF_REPO_SUBSTR) is not None
        checks["llama_server_binary_present"] = _resolve_llama_server().exists()
    except Exception:
        checks["gguf_cached"] = False
        checks["llama_server_binary_present"] = False
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


def _run_arm(
    *, policy: Any, proposer: Any, veto_on: bool, root_grid: Any, transitions: list
) -> JsonDict:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    started = time.monotonic()
    outcome = execute_bounded_llm_reinduction(
        game=policy.short,
        transitions=transitions,
        cell=policy.cell,
        root_grid=root_grid,
        proposer=proposer,
        candidate_provider=policy._world_model_candidates,
        load_engine=e3.load_engine,
        plan_in_model=policy._guided_plan_in_model(e3.plan_in_model),
        max_rounds=1,
        min_heldout_accuracy=0.0,
        min_goal_predicate_consistency=(1.0 if veto_on else 0.0),
        previous_level_complete_grid=policy._previous_level_complete_grid,
        enable_subgoal_search=policy.subgoal_search,
        subgoal_budget=policy.subgoal_budget,
        value_head=policy.value_head,
        enable_factored_planner=policy.factored_planner,
        factored_trust_threshold=policy.factored_trust_threshold,
        structural_goal_provider=None,
    )
    duration = time.monotonic() - started
    return {
        "veto_on": veto_on,
        "duration_s": round(duration, 3),
        "planned": bool(outcome.planned),
        "skipped": outcome.skipped,
        "heldout_accuracy": outcome.heldout_accuracy,
        "goal_predicate_satisfiable": bool(outcome.goal_predicate_satisfiable),
        "rounds": [
            {
                k: v
                for k, v in row.items()
                if k
                in (
                    "round",
                    "action",
                    "skipped",
                    "heldout_accuracy",
                    "goal_predicate_satisfiable",
                    "goal_predicate_consistency_accuracy",
                    "goal_predicate_consistency_n_real_levelups",
                    "plan_length",
                    "plan_reaches_goal",
                )
            }
            for row in outcome.rounds
        ],
    }


def run_prototype(
    *, game: str = DEFAULT_GAME, collect_budget: int = DEFAULT_COLLECT_BUDGET, port_base: int = 8970
) -> JsonDict:
    """Collect real live transitions, then directly compare the veto on/off on those
    SAME real transitions with the dynamics gate bypassed so the veto gets a genuine
    chance to fire. See module docstring for why the dynamics-gate-bypassed comparison
    is the informative one (the live call site's own min_heldout_accuracy=1.0 dominates
    round 1 in practice, per the honestly-disclosed dynamics_gate_finding field)."""

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    collect_proposer = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=port_base,
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=2560,
    )
    policy = E3AgentPolicy(game, proposer=collect_proposer, explore_budget=6, target_levels=3)
    lb.run_game(game, policy, budget=collect_budget)

    transitions = list(policy.transitions)
    real_levelups = sum(1 for t in transitions if t.level_after > t.level_before)
    root_grid = policy.root_grid if policy.root_grid is not None else transitions[0].grid

    if real_levelups == 0:
        return {
            "transitions_collected": len(transitions),
            "real_levelups_in_collected_transitions": 0,
            "arm_on": {},
            "arm_off": {},
            "run_ok": False,
        }

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer as _P

    proposer_on = _P(
        repo_substr=GGUF_REPO_SUBSTR, port=port_base + 1, mtp=True, kv_quant="q8_0",
        no_think_prefix="/no_think\n", max_tokens=2560,
    )
    proposer_off = _P(
        repo_substr=GGUF_REPO_SUBSTR, port=port_base + 2, mtp=True, kv_quant="q8_0",
        no_think_prefix="/no_think\n", max_tokens=2560,
    )
    arm_on = _run_arm(
        policy=policy, proposer=proposer_on, veto_on=True, root_grid=root_grid, transitions=transitions
    )
    arm_off = _run_arm(
        policy=policy, proposer=proposer_off, veto_on=False, root_grid=root_grid, transitions=transitions
    )

    return {
        "transitions_collected": len(transitions),
        "real_levelups_in_collected_transitions": real_levelups,
        "arm_on": arm_on,
        "arm_off": arm_off,
        "run_ok": True,
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    game: str = DEFAULT_GAME,
    collect_budget: int = DEFAULT_COLLECT_BUDGET,
) -> JsonDict:
    started = time.monotonic()
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "model_specs": MODEL_SPECS,
            "field_principles": FIELD_PRINCIPLES,
            "target_game": game,
            "real_levelups_in_collected_transitions": 0,
            "arm_on": {},
            "arm_off": {},
            "dynamics_gate_finding": "",
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

    proto = run_prototype(game=game, collect_budget=collect_budget)

    dynamics_gate_finding = (
        "A prior attempt at the live call site's own min_heldout_accuracy=1.0 found BOTH "
        "arms failing at the pre-existing dynamics gate (heldout_transition_verification_"
        "failed) before this goal-consistency veto was ever reached -- the veto is checked "
        "LAST (after dynamics acceptance and goal satisfiability), so in practice on real "
        "first-shot LLM induction it is frequently subordinate to that stricter, "
        "pre-existing gate. This run bypasses min_heldout_accuracy (set to 0.0) "
        "specifically to isolate and test the goal-consistency veto's own behavior."
    )

    if not proto.get("run_ok"):
        verdict = "complete: inconclusive_no_real_levelup_collected"
    else:
        on_skipped = proto["arm_on"].get("skipped")
        off_planned = proto["arm_off"].get("planned")
        if on_skipped == "goal_predicate_consistency_failed" and off_planned:
            verdict = "complete: veto_confirmed_catches_real_miscalibrated_predicate"
        elif on_skipped == "goal_predicate_consistency_failed":
            verdict = "complete: veto_fired_on_real_data"
        else:
            verdict = "complete: veto_did_not_fire_this_run"

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "target_game": game,
        "transitions_collected": proto.get("transitions_collected", 0),
        "real_levelups_in_collected_transitions": proto.get(
            "real_levelups_in_collected_transitions", 0
        ),
        "arm_on": proto.get("arm_on", {}),
        "arm_off": proto.get("arm_off", {}),
        "dynamics_gate_finding": dynamics_gate_finding,
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


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
