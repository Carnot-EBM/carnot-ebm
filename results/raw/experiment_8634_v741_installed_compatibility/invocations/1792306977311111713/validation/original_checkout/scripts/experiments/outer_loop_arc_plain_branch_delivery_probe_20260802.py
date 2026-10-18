"""Plain-branch delivery repair: does a GATE-PASSING engine install a plan on the
branch a HIDDEN game must take? The anatomy's only gate-passing observation was dc22
(hidden-state branch); HIDDEN_STATE_GAME_IDS is a hardcoded public-game tuple, so a
hidden Kaggle game always takes the PLAIN branch. Proven at the CALLEE, not the call site."""

import inspect
import json
import os
import sys
import time

sys.path.insert(0, "python")
sys.path.insert(0, "scripts/experiments")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from types import SimpleNamespace
from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_executable_world_model as e3


def make_lookup_oracle(transitions):
    """Memorize the agent's OWN observed transitions and replay them exactly.
    Deliberately an ORACLE: its job is to clear the plain gate so the POST-GATE
    channel can be observed, NOT to be producible by the generator."""
    table = {}
    for t in transitions:
        table[(np.asarray(t.grid).tobytes(), t.action, str(t.data))] = np.asarray(t.next_grid)

    def engine(grid, action, data):
        g = np.asarray(grid)
        return table.get((g.tobytes(), action, str(data)), g.copy()).copy()

    return engine


def make_reachable_goal(transitions, k):
    """Goal = the state actually observed k steps in. A plan provably exists in the
    oracle's model (replay the first k actions), so an empty plan means the channel
    is closed, not that the goal was unreachable."""
    target = np.asarray(transitions[k].next_grid).tobytes()

    def is_level_complete(grid):
        return np.asarray(grid).tobytes() == target

    return is_level_complete


def probe(game, seed, engine_kind, goal_kind, n=120):
    t0 = time.time()
    transitions, cell = e3.collect_transitions(game, n=n, seed=seed)
    root = np.asarray(transitions[0].grid)
    if engine_kind == "lookup_oracle":
        engine = make_lookup_oracle(transitions)
    else:
        engine = lambda g, a, d: np.asarray(g).copy()
    goal = make_reachable_goal(transitions, 2) if goal_kind == "reachable" else (lambda g: False)

    w = {
        "plan_in_model_calls": 0,
        "callers": [],
        "returned_plan_lengths": [],
        "reached_post_gate_call_site": False,
        "engine_identity_matches_injected": None,
        "goal_identity_matches_injected": None,
        "post_gate_caller": None,
    }
    real_pim, real_load = e3.plan_in_model, e3.load_engine

    def instrumented(engine_arg, is_done_arg, start_grid, **kw):
        w["plan_in_model_calls"] += 1
        frames = [f"{fr.function}:{fr.lineno}" for fr in inspect.stack()[1:3]]
        w["callers"].append(frames)
        if engine_arg is engine:
            w["engine_identity_matches_injected"] = True
            w["goal_identity_matches_injected"] = is_done_arg is goal
            w["reached_post_gate_call_site"] = True
            w["post_gate_caller"] = frames
        out = real_pim(engine_arg, is_done_arg, start_grid, **kw)
        w["returned_plan_lengths"].append(len(out or []))
        return out

    e3.plan_in_model = instrumented
    e3.load_engine = lambda g: (engine, goal)  # store never written: results/arc_e3 untouched
    os.environ["CARNOT_ARC_STALL_REFACTOR_LOOP"] = "1"
    try:
        policy = agent.E3AgentPolicy(
            game,
            proposer=SimpleNamespace(
                model_specs="INJECTED_NO_LLM_no_model_invoked",
                induce=lambda *a, **k: (True, ""),
                refactor=lambda *a, **k: (True, ""),
                include_playbook_exemplars=False,
            ),
            target_levels=2,
        )
        policy.root_grid = root
        policy.cell = cell
        policy.transitions = list(transitions)
        policy._episode_transition_start = 0
        policy._pending_induction_reason = "stall"
        policy._induce_and_plan()
        att = policy.induction_attempts[-1]
        plan_len = len(policy.plan or [])
    finally:
        e3.plan_in_model, e3.load_engine = real_pim, real_load
        os.environ.pop("CARNOT_ARC_STALL_REFACTOR_LOOP", None)
    acc = att.get("verify_accuracy")
    return {
        "game": game,
        "seed": seed,
        "engine": engine_kind,
        "goal": goal_kind,
        "hidden_state_branch": game in agent.HIDDEN_STATE_GAME_IDS,
        "verify_accuracy": acc,
        "verify_cell_recall": att.get("verify_cell_recall"),
        "trust_metric": att.get("trust_metric"),
        "gate_passed": bool(acc is not None and float(acc) >= 0.5),
        "attempt_planned": bool(att.get("planned")),
        "attempt_skipped": att.get("skipped"),
        "attempt_plan_length": att.get("plan_length"),
        "policy_plan_installed_len": plan_len,
        "callee_witness": w,
        "elapsed_s": round(time.time() - t0, 3),
    }


if __name__ == "__main__":
    rows = []
    for seed in (0, 1, 2):
        rows.append(probe("tu93", seed, "lookup_oracle", "reachable"))
        rows.append(probe("tu93", seed, "identity", "reachable"))
    rows.append(probe("tu93", 0, "lookup_oracle", "unreachable"))
    for r in rows:
        print(
            "%-14s s%s goal=%-11s acc=%-7s gate=%-5s planned=%-5s plan=%s post_gate=%s %s"
            % (
                r["engine"],
                r["seed"],
                r["goal"],
                r["verify_accuracy"],
                r["gate_passed"],
                r["attempt_planned"],
                r["policy_plan_installed_len"],
                r["callee_witness"]["reached_post_gate_call_site"],
                r["callee_witness"]["post_gate_caller"],
            )
        )
    json.dump(
        rows, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1, default=str
    )
