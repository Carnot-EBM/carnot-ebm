#!/usr/bin/env python3
"""ONE GAME, ONE BUDGET: record whether `_win_transition` is available at every LIVE induce call.

WHAT THIS IS. The live scored path is `make_carnot_agent -> E3AgentPolicy.next_move`. The only
place the 2026-08-01 win-transition change can alter behaviour is the induce call at
`arc_competition_agent.py:6433`, which passes `self._win_transition` to the proposer. A win
transition only EXISTS after a level-up (`_begin_level_goal_episode` is its sole writer), so
before measuring what the change DOES, this measures how often it is even REACHED.

HOW IT AVOIDS A GPU. The proposer is `experiment_4605._NoOpProposer` -- the project's own
`llm_off` arm definition (results/first_win_llm_on_20260727 `arm_definitions`), not a bespoke
stub. The full shipped `_induce_and_plan` body runs, including the `self._proposer().induce(...,
win_transition=self._win_transition)` call site under test; only the generator's answer is
absent. That corpus measured `induction_attempts_planned == 0` on 224/224 cells and every LLM-on
arm BIT-IDENTICAL to its llm_off control, which is why the llm_off trajectory is the right
CPU-only stand-in here -- and it is still a stand-in, stated as such in the artifact.

NOTHING TRACKED IS WRITTEN. `arc_leaderboard_eval.run_game` writes no files (only its `main()`
does, and `main()` is never called); output goes to the path given on the command line.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main() -> int:
    game = sys.argv[1]
    budget = int(sys.argv[2])
    out_path = Path(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20260802
    # THE REACHABLE SEED. `random.seed`/`np.random.seed` above CANNOT reach the agent's
    # stochasticity: every RNG in the explorer is `random.Random(<constructor default>)`
    # (arc_competition_agent.py:1310 `frontier_discipline_seed=20260724`, :1397
    # `Random(20260621)`). Varying argv[4] alone therefore produces an A/A with an
    # identically-zero floor BY CONSTRUCTION, which is a design bug, not a noise estimate.
    # argv[5] varies the seed the agent actually consumes. Default = the shipped value, so
    # a 4-argument invocation is byte-for-byte the shipped configuration.
    fd_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20260724

    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    import arc_leaderboard_eval as ev
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG, E3AgentPolicy
    from carnot.experiment_4605_live_integration_scored_agent import _NoOpProposer

    calls: list[dict] = []
    original = E3AgentPolicy._induce_and_plan

    def instrumented(self):  # noqa: ANN001
        # Read BEFORE delegating: `_induce_and_plan` appends its own attempt row, and the
        # question is the state at the moment the shipped code reads `self._win_transition`.
        calls.append(
            {
                "call_index": len(calls),
                "win_transition_available": self._win_transition is not None,
                # The independent structural discriminator. `_begin_level_goal_episode` is the
                # sole writer of BOTH `_win_transition` and `level_induction_events`, so these
                # two must agree unless the empty-transitions guard fired. Recorded separately
                # so a disagreement is VISIBLE rather than assumed away.
                "n_level_induction_events_before": len(self.level_induction_events),
                "pending_induction_reason": self._pending_induction_reason,
                "goal_level": self._current_goal_level,
                "start_level": self._start_level,
                "observed_level": self._observed_level,
                "explorer_best_level": getattr(self.explorer, "best_level", None),
                "explorer_start_level": getattr(self.explorer, "start_level", None),
                "n_transitions_total": len(self.transitions),
                "episode_transition_start": int(self._episode_transition_start),
                "n_active_transitions": len(self._active_transitions()),
            }
        )
        return original(self)

    E3AgentPolicy._induce_and_plan = instrumented
    try:
        policy = E3AgentPolicy(
            game,
            proposer=_NoOpProposer(),
            target_levels=int(SUBMITTED_AGENT_CONFIG["target_levels"]),
            value_weight=float(SUBMITTED_AGENT_CONFIG["value_weight"]),
            search_mode=str(SUBMITTED_AGENT_CONFIG["search_mode"]),
            lazy_value_top_k=int(SUBMITTED_AGENT_CONFIG["lazy_value_top_k"]),
            frontier_batch_size=SUBMITTED_AGENT_CONFIG["frontier_batch_size"],
            navigation_cost_tiebreak=bool(SUBMITTED_AGENT_CONFIG["navigation_cost_tiebreak"]),
            similarity_retrieval=bool(SUBMITTED_AGENT_CONFIG["matm_similarity_retrieval_enabled"]),
            frontier_discipline_seed=fd_seed,
        )
        t0 = time.time()
        row = ev.run_game(game, policy, budget=budget, variant=0)
        elapsed = time.time() - t0
        witness = {}
        try:
            witness = policy.generator_liveness_witness()
        except Exception as exc:
            witness = {"witness_error": repr(exc)[:200]}
        out = {
            "game": game,
            "budget": budget,
            "seed": seed,
            "frontier_discipline_seed": fd_seed,
            "elapsed_s": round(elapsed, 3),
            "levels": row.get("levels"),
            "reached": row.get("reached"),
            "actions": row.get("actions"),
            "induce_calls": calls,
            "n_induce_calls": len(calls),
            "n_with_win_transition": sum(1 for c in calls if c["win_transition_available"]),
            "level_induction_events": policy.level_induction_events,
            "induction_attempts_reasons": [a.get("reason") for a in policy.induction_attempts],
            "induction_attempts_skipped": [a.get("skipped") for a in policy.induction_attempts],
            "induction_attempts_n": witness.get("induction_attempts_n"),
            "induction_attempts_planned": witness.get("induction_attempts_planned"),
            "llm_enabled": witness.get("llm_enabled"),
            "error": "",
        }
    except Exception as exc:  # a crashed game must be VISIBLE, never an implicit zero
        import traceback

        out = {
            "game": game,
            "budget": budget,
            "seed": seed,
            "frontier_discipline_seed": fd_seed,
            "error": repr(exc)[:400],
            "traceback": traceback.format_exc()[-2000:],
            "induce_calls": calls,
            "n_induce_calls": len(calls),
            "n_with_win_transition": sum(1 for c in calls if c["win_transition_available"]),
        }
    finally:
        E3AgentPolicy._induce_and_plan = original

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(
        f"{game} budget={budget} levels={out.get('levels')} induce={out.get('n_induce_calls')} "
        f"win_avail={out.get('n_with_win_transition')} err={bool(out.get('error'))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
