# ARC-AGI-3 live agent — the canonical path (disambiguation)

**Why this doc (the 2026-06-19 "0.08 incident"):** the offline eval improved STRONGER opt-in configs
(`explorer_bf` unlocked cn04) while the SUBMITTED default shipped bare BFS — and nobody noticed, because
"better" lived behind an opt-in eval flag and the headline metric was banked-replay levels, not the
submitted path. There are **13 overlapping solve/agent entrypoints** in `python/carnot/agentic/`; this doc
names the ONE that ships and marks the rest, so the ambiguity that hid the gap can't hide it again.

## THE submitted agent (the only thing scored)

```
scripts/kaggle/submission_kernel/main.py
  -> make_carnot_agent(Agent)                      # cascade=True (default)
     -> E3AgentPolicy                              # verifier-routed cascade: explorer -> E3 induction
        -> StepwiseExplorer                        # tier-1 (CPU, frame-only)
        -> LocalGGUFProposer (Qwen3.5-9B-MTP)      # tier-3, on stall
```

The exact shipped config is `arc_competition_agent.py:SUBMITTED_AGENT_CONFIG` — the **single source of
truth**. `tests/python/test_arc_submitted_agent_parity.py` asserts the live default matches it AND that the
`router_wired` / `world_model_dsl_wired` flags reflect real imports. **Improvements to the live agent go
HERE (update the config + the default), NEVER to an opt-in-only eval flag.**

## Dev / eval-only paths — NOT what ships (do not confuse for the submission)

| Path | What it is | Status |
|---|---|---|
| `make_carnot_agent(cascade=False)` → `CarnotAgentPolicy(load_solutions())` | recognize-and-replay banked KNOWN-game solutions | dev-only — "useless on the hidden eval"; never submitted |
| `arc_leaderboard_eval.py --policy explorer_vh / explorer_bf` | value-head-routed / best-first explorer (the modes that unlocked cn04) | **opt-in eval modes — STRONGER than the shipped default; this is the divergence to fix (.414 A1: make the best the default)** |
| `arc_strategy_router.py`, `arc_world_model_dsl.py` | stronger generic solvers | EXIST but NOT imported by the submission path (.414 A1 wires them) |
| `arc_value_learner.py` cross-game value head | steps-to-go regressor | inert in the default (`value_weight=0.0`); per-game discriminative head is the shipped lever (.414 A3) |

## The prevention rules (enforced)

1. **The strongest measured config MUST be the submitted default.** Opt-in eval flags are for sweeps only;
   the best config is never allowed to live opt-in. Enforced by the parity test + the per-milestone
   measurement gate (.414 A1 reports the held-out generic solve-rate of the EXACT submitted default,
   frame-only, as the headline — not `reproducible_total_levels`).
2. **`reproducible_total_levels` is NOT the headline.** It measures banked replays of KNOWN games (≈0 on the
   hidden eval). The headline is the submitted default's held-out generic solve-rate + action efficiency.
3. **`SUBMITTED_AGENT_CONFIG` + the parity test are the gate.** Any change to the live explorer defaults, or
   wiring a stronger solver, must update the config dict in the same commit, or CI fails.

## Cleanup backlog (disambiguation, queued)

- Fold the `explorer_bf` / `explorer_vh` strength into the E3AgentPolicy DEFAULT (remove opt-in-only strength). [.414 A1]
- Mark `cascade=False` banked-replay clearly dev-only / remove it from the submission surface.
- Collapse the 13 overlapping entrypoints toward one documented live-agent constructor + the dev/eval harness.

Cross-refs: `ops/verifier_gaps.md` GAP-LIVE-INTEGRATION; `research-roadmap-next.yaml` A1; `results/arc_offline_to_live_bridge_v2.json` (the 8/32 measurement).
