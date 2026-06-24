"""MULTI-LEVEL DEEPENING DIAGNOSTIC: locate WHERE the live agent's L1->L2 transition stalls.

Context (.428 B1): live first-win rate ~0.59 but live_multi_level_solve_rate ~0.04 -- the agent
reaches the FIRST level-up on a fresh game but rarely deepens to L2/L3. Every existing harness
(exp4628) used target_levels=1 AND broke at the first level-up, so NOTHING has ever instrumented
the L1->L2 transition in the live agent. This does.

Method: instantiate the LIVE SUBMITTED-config E3AgentPolicy (real CNN action-effect expansion prior +
value head + candidate router = the .427 bridge-crossing config that actually runs on Kaggle) but with
target_levels=5 and NO early break, on games where L2+ is KNOWN reachable (we reproduce them offline).
Roll out to a generous budget; record the action index at EACH level-up. This separates two hypotheses:
  (H1) real capability wall -- exploration+energy guidance simply cannot find the 2nd win, OR
  (H2) no-headroom measurement artifact -- the L2 transition needs the LLM proposer (goal induction)
       that the matched-offline noop arm disables, so a noop run stalling at L1 is EXPECTED not a wall.

Arm: NoOpProposer (exploration + CNN prior + value head, NO LLM induction). If exploration ALONE
deepens on some games -> H1 is wrong for those (exploration suffices). If it universally stalls at L1
-> the 2nd-win generation needs goal-induction (the proposer), pointing the lever at the proposer not
the explorer. CPU-forced (the value-head/CNN are small; no LLM in the noop arm).
"""

from __future__ import annotations

import json
import os
import sys
import time

# ARM selection. noop = exploration + CNN expansion prior + value head only (CPU, no LLM induction).
# real = the LIVE submitted agent with the real LocalGGUFProposer (LLM goal/rule induction); needs a GPU.
_ARM = os.environ.get("MULTILEVEL_ARM", "noop")
if _ARM != "real":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from arcengine import GameAction  # noqa: E402
from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of  # noqa: E402
from carnot.experiment_4628_dense_curiosity_progress_loop import _NoOpProposer  # noqa: E402


def _gid(arc, short):
    for e in arc.get_environments():
        g = getattr(e, "game_id", "")
        if g.split("-")[0] == short:
            return str(g)
    raise RuntimeError(f"{short} unavailable")


def diagnose(arc, short, budget, target_levels=5):
    """Roll the LIVE submitted-config policy (noop proposer) to budget; record per-level-up action idx."""
    gid = _gid(arc, short)
    env = arc.make(gid, scorecard_id=arc.open_scorecard())
    # SUBMITTED defaults EXCEPT: target_levels raised so it won't stop at L1.
    # noop arm -> NoOpProposer (exploration-only); real arm -> proposer=None lazily loads LocalGGUFProposer.
    proposer = None if _ARM == "real" else _NoOpProposer()
    policy = E3AgentPolicy(gid, proposer=proposer, target_levels=int(target_levels))
    frames: list = []
    latest = None
    start_level = None
    reached = 0
    actions = 0
    levelup_at = {}  # level -> action index where first reached
    t0 = time.time()
    for _ in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            actions += 1
        if latest is None:
            break
        lvl = _level_of(latest)
        if start_level is None:
            start_level = lvl
        rel = lvl - (start_level or 0)
        if rel > reached:
            reached = rel
            levelup_at.setdefault(rel, actions)
        frames.append(latest)
    return {
        "game": short,
        "max_rel_level": int(reached),
        "levelup_at_action": {str(k): v for k, v in sorted(levelup_at.items())},
        "actions_used": actions,
        "budget": int(budget),
        "exhausted_budget": actions >= int(budget) - 1,
        "explorer_explored_out": bool(getattr(policy.explorer, "explored_out", False)),
        "state_coverage": int(len(getattr(policy.explorer, "graph", {}) or {})),
        # induction diagnostics: did the explore->induce->execute cascade actually fire?
        "induction_fired": bool(getattr(policy, "induced", False)),
        "final_phase": str(getattr(policy, "phase", "?")),
        "transitions_collected": int(len(getattr(policy, "transitions", []) or [])),
        "plan_len": int(len(getattr(policy, "plan", []) or [])),
        "stalled_at_L1": reached == 1,
        "reached_L2_plus": reached >= 2,
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> int:
    # Multi-level-reachable games (repro_levels>=2 per registry): exploration-only deepening candidates.
    games = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["vc33", "sc25", "tn36", "cd82", "sp80", "lp85", "su15", "tu93", "m0r0"]
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    arc = kit.offline_arcade()
    out = {"budget": budget, "arm": _ARM, "games": {}}
    _outfile = f"results/proto_multilevel_diag_{_ARM}.json"
    print(f"== multi-level deepening diag: LIVE E3 (arm={_ARM}, target_levels=5), budget={budget} ==",
          flush=True)
    for short in games:
        try:
            r = diagnose(arc, short, budget)
        except Exception as e:
            out["games"][short] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {short:6} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        out["games"][short] = r
        print(f"  {short:6} maxL={r['max_rel_level']} at={r['levelup_at_action']} "
              f"acts={r['actions_used']}/{budget} induced={r['induction_fired']} "
              f"phase={r['final_phase']} plan={r['plan_len']} cov={r['state_coverage']} "
              f"({r['wall_s']}s)", flush=True)
    reached2 = sorted(g for g, v in out["games"].items() if v.get("reached_L2_plus"))
    stalled = sorted(g for g, v in out["games"].items() if v.get("stalled_at_L1"))
    out["reached_L2_plus"] = reached2
    out["stalled_at_L1"] = stalled
    out["VERDICT"] = (
        "EXPLORATION_ALONE_DEEPENS_on_" + ",".join(reached2) if reached2
        else "EXPLORATION_NEVER_DEEPENS_2nd_win_needs_goal_induction_proposer"
    )
    json.dump(out, open(_outfile, "w"), indent=2)
    print(f"\nreached_L2+={reached2}  stalled_at_L1={stalled}\nVERDICT={out['VERDICT']}\n-> {_outfile}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
