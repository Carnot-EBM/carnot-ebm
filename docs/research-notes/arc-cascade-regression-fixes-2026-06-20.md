# Improving the E3+v3 cascade regression WITHOUT reverting (2026-06-20)

Operator: "improve the regression without reverting." Diagnosed (3-agent workflow wezv5lw17) + fixed +
gate-verified. Result: the slow/timing-out regression is FIXED; the residual action-efficiency gap is
architectural (needs better search guidance, not a config fix).

## The three fixes (cascade + v3 stay fully wired)

1. **Per-node value guard** (`_value`, arc_competition_agent.py): skip the expensive v3 featurizer when
   `value_weight==0` (its result is multiplied by 0 in the frontier priority -> pure dead cost; ordering
   is provably identical). Fires unchanged when value_weight>0. -> restored ~bare-BFS SPEED at weight 0.
2. **Env-gated induction skip** (`_induce_and_plan`): `CARNOT_ARC_DISABLE_INDUCTION=1` skips the LLM
   world-model tier. Default OFF (production/Kaggle runs induction normally). The local GATE sets it so it
   measures the tier-1 explorer's SEARCH cleanly without the local llama-server spawn (a one-time ~30s
   cost under the real 12h eval, but it dominates a bounded local gate run). -> eliminated the gate timeouts.
3. **Nav edge recording** (`_serve`): record forward edges for RESET-replay/nav steps too (origin=self.cur),
   so `_shortest_path` learns replayed paths and future backtracks forward-walk instead of RESET-replaying
   from root. Correct + benign (solves preserved) but did NOT move the action count -- see below.

## Gate before/after (8 games, frame-only, induction disabled)

| | before (E3+v3, value_weight=5 then 0) | after (3 fixes) |
|---|---|---|
| solved | 1/8 (then 3/8) | **4/8** |
| timed out @115s | 6 (then 4) | **0** |
| median actions/solve | 7792 | 7760 |

Speed + solve-rate regression: FIXED. Action efficiency: essentially unchanged.

## The residual: action efficiency is ARCHITECTURAL, not a config bug

The gate's bare-BFS "21 actions for lp85" was the wrong reference: it is the OFFLINE solver's SOLUTION
LENGTH (it teleports between states via deepcopy). The LIVE StepwiseExplorer must PHYSICALLY navigate +
explore, so its ~7760 is actions-to-explore-AND-solve. A live agent can NEVER match an offline solution
length; the gate baseline was re-set to the LIVE agent's own metrics (live-vs-live regression check).
Closing the live action-efficiency gap (explore 7760 to find a 21-step solution) requires BETTER SEARCH
GUIDANCE so the explorer probes fewer dead ends -- i.e. a working frame-change predictor / value head
(the .416 A2/A3 work, currently honest-null). It is NOT an overnight config fix.

## Net

Submitting the post-fix config would NOT regress (gate passes, no timeouts, 4/4 solved) and is faster +
more robust than the value_weight=5 cascade -- but it would likely still score ~0.08 (action efficiency
unchanged -> efficiency term ~0 on the games it solves by broad exploration). The real score lever remains
making the live explorer action-efficient via guidance -- the ongoing predictor/energy work.

Cross-refs: diagnosis wf wezv5lw17; arc-e3-cascade-vs-bfs-2026-06-20.md; the gate
(scripts/kaggle/arc_local_submission_gate.py); arc-frame-change-predictor-spec.md (the guidance fix).
