#!/usr/bin/env python3
"""DIRECT CHECK: does `win_transition` ever arrive at the proposer as non-None on the live path?

The exposure analysis infers "this call reached arc_competition_agent.py:6433" from the skip
string. An inference is not a measurement, and this project has already shipped a confident wrong
answer built on one. So this instruments the RECEIVING END: the proposer's own `induce`, which is
the single object both call sites hand their arguments to. It records, per call, whether the
`win_transition` keyword arrived non-None AND which module made the call, read off the stack.

Run: verify_kwarg.py <game> <budget> <out.json>. CPU only, NoOp proposer, offline arcade.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main() -> int:
    game, budget, out_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    import random

    random.seed(20260802)
    try:
        import numpy as np

        np.random.seed(20260802)
    except Exception:
        pass

    import arc_leaderboard_eval as ev
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG, E3AgentPolicy

    seen: list[dict] = []

    class RecordingNoOpProposer:
        """Byte-equivalent to experiment_4605._NoOpProposer, plus a record of what it was handed."""

        include_playbook_exemplars = False

        def induce(self, *args, **kwargs):
            stack = traceback.extract_stack()[:-1]
            caller = stack[-1] if stack else None
            seen.append(
                {
                    "call_index": len(seen),
                    "win_transition_kwarg_present": "win_transition" in kwargs,
                    "win_transition_is_not_none": kwargs.get("win_transition") is not None,
                    "previous_level_complete_grid_is_not_none": (
                        kwargs.get("previous_level_complete_grid") is not None
                    ),
                    "kwargs_keys": sorted(kwargs),
                    "caller_file": (Path(caller.filename).name if caller else None),
                    "caller_lineno": (caller.lineno if caller else None),
                    "caller_func": (caller.name if caller else None),
                }
            )
            return False, "disabled_no_live_llm"

        def refactor(self, *_a, **_k):
            return False, "disabled_no_live_llm"

        def world_model_candidates(self, _game):
            return []

    policy = E3AgentPolicy(
        game,
        proposer=RecordingNoOpProposer(),
        target_levels=int(SUBMITTED_AGENT_CONFIG["target_levels"]),
        value_weight=float(SUBMITTED_AGENT_CONFIG["value_weight"]),
        search_mode=str(SUBMITTED_AGENT_CONFIG["search_mode"]),
        lazy_value_top_k=int(SUBMITTED_AGENT_CONFIG["lazy_value_top_k"]),
        frontier_batch_size=SUBMITTED_AGENT_CONFIG["frontier_batch_size"],
        navigation_cost_tiebreak=bool(SUBMITTED_AGENT_CONFIG["navigation_cost_tiebreak"]),
        similarity_retrieval=bool(SUBMITTED_AGENT_CONFIG["matm_similarity_retrieval_enabled"]),
    )
    row = ev.run_game(game, policy, budget=budget, variant=0)
    out = {
        "game": game,
        "budget": budget,
        "levels": row.get("levels"),
        "actions": row.get("actions"),
        "n_proposer_induce_calls": len(seen),
        "n_with_win_transition_kwarg_not_none": sum(
            1 for s in seen if s["win_transition_is_not_none"]
        ),
        "proposer_induce_calls": seen,
        "agent_level_induction_events": policy.level_induction_events,
        "agent_induction_attempts": [
            {k: a.get(k) for k in ("reason", "skipped", "planned", "goal_level")}
            for a in policy.induction_attempts
        ],
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(
        f"{game} b={budget} levels={out['levels']} proposer_induce_calls={len(seen)} "
        f"win_kwarg_not_none={out['n_with_win_transition_kwarg_not_none']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
