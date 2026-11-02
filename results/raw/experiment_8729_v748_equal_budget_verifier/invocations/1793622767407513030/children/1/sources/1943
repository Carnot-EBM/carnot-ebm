"""Experiment 5704: live A/B testing whether the live pipeline's
`min_heldout_accuracy=1.0` dynamics gate is too strict to be useful (task 13 --
follow-up to task 11's corpus survey finding that only 12.6% of real induction
rounds ever clear that exact bar).

Collects real transitions on a game with an observed real level-up (reusing
exp5700's collection methodology: lp85, budget=50, real GPU-backed
`LocalGGUFProposer`), then runs N INDEPENDENT fresh single-round real induction
attempts (`execute_bounded_llm_reinduction`, `max_rounds=1`, a fresh proposer
per attempt to avoid a known proposer-connection-reuse issue) against the SAME
real transitions with `min_heldout_accuracy=0.0` (gate bypassed so every
attempt's real `heldout_accuracy` is observed regardless of outcome).

For each attempt, this experiment asks: would the CURRENT strict threshold
(1.0) accept it? Would a RELAXED threshold (0.7, chosen as a plausible
"good but imperfect" bar, not a hyperparameter search) accept it? An attempt
that the relaxed threshold accepts but the strict one rejects is the
interesting case -- for those, this experiment checks whether the resulting
plan (`outcome.planned`, `plan_reaches_goal`) was ACTUALLY good, or whether
the strict threshold was correctly protecting against a plan that would have
failed anyway.

This is exploratory and honest either way: it does NOT presuppose relaxing
the threshold helps. If no attempt lands in the interesting [0.7, 1.0) band
across N attempts, the result is reported as inconclusive, not forced into a
false positive or negative.

Spec refs: REQ-ARC-WMTE-5593-4 (extends, dynamics-gate calibration follow-up).
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

EXPERIMENT_ID = "experiment_5704_dynamics_gate_relaxed_threshold_ab"
RESULT_RELATIVE_PATH = "results/experiment_5704_dynamics_gate_relaxed_threshold_ab.json"
SCHEMA = "carnot.exp5704.dynamics_gate_relaxed_threshold_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5704
DEFAULT_GAME = "lp85"
DEFAULT_COLLECT_BUDGET = 50
DEFAULT_N_ATTEMPTS = 5
STRICT_THRESHOLD = 1.0
RELAXED_THRESHOLD = 0.7
GGUF_REPO_SUBSTR = "Qwen3.5-9B-MTP"
MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "E3AgentPolicy default live re-induction proposer",
    }
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "target_game",
    "n_attempts",
    "strict_threshold",
    "relaxed_threshold",
    "attempts",
    "n_strict_accepts",
    "n_relaxed_only_accepts",
    "relaxed_only_accepts_with_good_plan",
    "real_levelups_in_collected_transitions",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; distinguishes 'relaxing the threshold unlocks a real usable plan the strict gate discards' from 'the strict gate is correctly protecting against bad plans' from 'inconclusive -- no attempt landed in the interesting band' -- these are different, non-interchangeable findings"
    },
    "n_relaxed_only_accepts": {
        "principle": "count of real attempts that would be accepted under the relaxed bar but rejected under the strict bar -- the direct evidence for whether the strict threshold discards recoverable attempts"
    },
    "relaxed_only_accepts_with_good_plan": {
        "principle": "of the relaxed-only accepts, how many produced a plan that ALSO passed plan_reaches_goal (in-model verification) -- an accepted-but-useless model would not support loosening the gate"
    },
    "real_levelups_in_collected_transitions": {
        "principle": "CLAUDE.md FALSE_NEGATIVE_RISK discipline -- the comparison is only interpretable with >=1 real level-up in the collection window"
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


def _run_attempt(
    *, attempt_index: int, policy: Any, root_grid: Any, transitions: list, port: int
) -> JsonDict:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    proposer = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=port,
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=2560,
    )
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
        previous_level_complete_grid=policy._previous_level_complete_grid,
        enable_subgoal_search=policy.subgoal_search,
        subgoal_budget=policy.subgoal_budget,
        value_head=policy.value_head,
        enable_factored_planner=policy.factored_planner,
        factored_trust_threshold=policy.factored_trust_threshold,
        structural_goal_provider=None,
    )
    duration = time.monotonic() - started
    heldout_accuracy = float(outcome.heldout_accuracy or 0.0)
    round0 = outcome.rounds[0] if outcome.rounds else {}
    return {
        "attempt_index": attempt_index,
        "duration_s": round(duration, 3),
        "heldout_accuracy": heldout_accuracy,
        "accepted_by_strict": heldout_accuracy >= STRICT_THRESHOLD,
        "accepted_by_relaxed": heldout_accuracy >= RELAXED_THRESHOLD,
        "planned": bool(outcome.planned),
        "goal_predicate_satisfiable": bool(outcome.goal_predicate_satisfiable),
        "plan_reaches_goal": bool(round0.get("plan_reaches_goal", False)),
        "skipped": outcome.skipped,
    }


def run_prototype(
    *,
    game: str = DEFAULT_GAME,
    collect_budget: int = DEFAULT_COLLECT_BUDGET,
    n_attempts: int = DEFAULT_N_ATTEMPTS,
    port_base: int = 8980,
) -> JsonDict:
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
            "attempts": [],
            "run_ok": False,
        }

    attempts = []
    for i in range(n_attempts):
        attempts.append(
            _run_attempt(
                attempt_index=i,
                policy=policy,
                root_grid=root_grid,
                transitions=transitions,
                port=port_base + 1 + i,
            )
        )

    return {
        "transitions_collected": len(transitions),
        "real_levelups_in_collected_transitions": real_levelups,
        "attempts": attempts,
        "run_ok": True,
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    game: str = DEFAULT_GAME,
    collect_budget: int = DEFAULT_COLLECT_BUDGET,
    n_attempts: int = DEFAULT_N_ATTEMPTS,
) -> JsonDict:
    started = time.monotonic()
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "model_specs": MODEL_SPECS,
            "field_principles": FIELD_PRINCIPLES,
            "target_game": game,
            "n_attempts": 0,
            "strict_threshold": STRICT_THRESHOLD,
            "relaxed_threshold": RELAXED_THRESHOLD,
            "attempts": [],
            "n_strict_accepts": 0,
            "n_relaxed_only_accepts": 0,
            "relaxed_only_accepts_with_good_plan": 0,
            "real_levelups_in_collected_transitions": 0,
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

    proto = run_prototype(game=game, collect_budget=collect_budget, n_attempts=n_attempts)

    if not proto.get("run_ok"):
        verdict = "complete: inconclusive_no_real_levelup_collected"
        n_strict = n_relaxed_only = n_relaxed_only_good = 0
    else:
        attempts = proto["attempts"]
        n_strict = sum(1 for a in attempts if a["accepted_by_strict"])
        relaxed_only = [
            a for a in attempts if a["accepted_by_relaxed"] and not a["accepted_by_strict"]
        ]
        n_relaxed_only = len(relaxed_only)
        n_relaxed_only_good = sum(1 for a in relaxed_only if a["plan_reaches_goal"])

        if n_relaxed_only == 0:
            verdict = "complete: inconclusive_no_attempt_in_relaxed_only_band"
        elif n_relaxed_only_good > 0:
            verdict = (
                f"complete: relaxed_threshold_unlocks_{n_relaxed_only_good}_of_{n_relaxed_only}_"
                f"good_plans_strict_gate_discards"
            )
        else:
            verdict = (
                f"complete: relaxed_threshold_accepts_{n_relaxed_only}_attempts_but_"
                f"none_produce_a_good_plan_strict_gate_correctly_protective"
            )

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "target_game": game,
        "n_attempts": n_attempts,
        "strict_threshold": STRICT_THRESHOLD,
        "relaxed_threshold": RELAXED_THRESHOLD,
        "attempts": proto.get("attempts", []),
        "n_strict_accepts": n_strict,
        "n_relaxed_only_accepts": n_relaxed_only,
        "relaxed_only_accepts_with_good_plan": n_relaxed_only_good,
        "real_levelups_in_collected_transitions": proto.get(
            "real_levelups_in_collected_transitions", 0
        ),
        "transitions_collected": proto.get("transitions_collected", 0),
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
