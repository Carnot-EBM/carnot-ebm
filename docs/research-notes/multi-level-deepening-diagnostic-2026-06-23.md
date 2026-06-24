# Multi-level live deepening: the 0.0 "wall" is a HARNESS ARTIFACT, and exploration is not the bottleneck (outer-loop, 2026-06-23)

Operator: "let's iterate on multi-level live deepening." The `.428 capstone surfaced a NEW wall —
`live_multi_level_solve_rate = 0.0` vs first-win-rate `0.59` — and the conductor's `.429 pivot aims
generation-guidance levers (A1 value-routing discriminator GUIDES live, A2 energy-as-fitness QD
evolution) at it. Before brainstorming a lever (the null-prone pattern), this diagnoses WHERE the
L1->L2 transition actually fails. Two findings; the first is the load-bearing one.

## FINDING 1 (headline): `live_multi_level_solve_rate = 0.0` is a MEASUREMENT ARTIFACT, not a measured wall

`scripts`/`experiment_4646_live_multi_level_solve_rate_metric.py:compute_live_multi_level_solve_rate`
computes `depth>=2 attempts / all attempts`. Its INPUT attempts come from the A1/A2 artifacts
(exp4640/exp4641), which are produced by the `experiment_4628` rollout `run_variant_attempt`. That
rollout **BREAKS at the first level-up**:

```python
# experiment_4628_dense_curiosity_progress_loop.py  (run_variant_attempt)
if start_level is not None and reached > start_level:
    actions_to_first = actions
    break        # <-- terminates the episode at the FIRST level-up
```

with `target_levels=1`. So every source attempt's `reached_level` is **capped at start+1 by
construction**. `depth>=2` is therefore *structurally impossible* in the metric's input — the
histogram is `depth_0=24, depth_1=26, depth_2=0`, and `multi_level_attempt_count = 0`. The rate is
`0/50 = 0.0` **no matter how capable the agent is**. The metric never let a single attempt try for a
2nd level-up.

This is exactly the corpus/harness-artifact trap CLAUDE.md's "ARC-AGI-3 IS a Live Hidden-Game
Discovery Agent" framing rule warns about ("an offline null may be a HARNESS/CORPUS ARTIFACT, not a
true limitation… verify the offline harness faithfully reproduces what the LIVE agent sees" — the
`n_win_states=0` precedent). **The `.429 generation levers will be evaluated against a metric that
cannot move**, because the rollout that feeds it stops at L1. The harness must be fixed (target_levels
> 1 + no early break + real proposer) before `live_multi_level_solve_rate` can measure anything.

## FINDING 2: the deepening MACHINERY exists (`level_up_reinduction`) and is gated on the PROPOSER

`E3AgentPolicy.next_move` (`arc_competition_agent.py`) has an explicit multi-level path: after a
level-up it re-enters induction with `reason="level_up_reinduction"` to induce the NEXT level's goal
predicate (it does not stop at L1 — `is_done` only fires at `best_level >= start_level +
target_levels`). So the agent is *designed* to deepen; the question is whether its generation
(induction via the LLM proposer) can find the 2nd win. This is gated on the proposer: a no-op proposer
re-induces nothing and falls back to exploration.

## FINDING 3: exploration ALONE (CNN expansion prior + value head, no LLM induction) never deepens

`scripts/experiments/proto_multilevel_diag.py` runs the LIVE SUBMITTED-config `E3AgentPolicy`
(real `.427 CNN action-effect expansion prior + value head + candidate router) with `target_levels=5`
and **no early break**, on the 9 multi-level-reachable games, budget 3000. Arm = NoOpProposer
(exploration only, no LLM induction; CPU). Result (`results/proto_multilevel_diag_noop.json`):

| game | max rel level | L1 reached at action | actions used | states explored past L1 | explored_out |
|---|---|---|---|---|---|
| lp85 | 1 | 20 | 2842/3000 | 633 | no |
| sc25 | 1 | 1660 | 2925/3000 | 1843 | no |
| vc33 | 1 | 1758 | 2913/3000 | 460 | no |
| tn36 | 0 | — | 2934/3000 | 316 | no |
| cd82 | 0 | — | 2951/3000 | 417 | no |
| sp80 | 0 | — | 2903/3000 | 389 | no |
| su15 | 0 | — | 2929/3000 | 260 | no |
| tu93 | 0 | — | 2916/3000 | 1101 | no |
| m0r0 | 0 | — | 2961/3000 | 564 | no |

**0/9 reach L2.** The 3 games that reach L1 then explore 460–1843 MORE distinct states over 1300–2800
more actions and never find L2 — not budget-starved (used ~2900/3000), not explored-out. The explorer
is doing its job (broad new-state coverage); it simply is **not** the mechanism that produces a 2nd
win. The 6 games that never reach L1 need an induced per-game mechanic even for L1.

**Honest caveat:** the no-op arm CANNOT deepen by construction on games whose L2 needs induction — so
"exploration never deepens" is *expected*, not a surprise. Its value is the negative control: it rules
OUT "more/better exploration" as the lever and points squarely at goal/rule INDUCTION (the proposer /
world-model) — which is exactly where the `.429 generation-guidance pivot is aimed. Registry context
reinforces this: all 9 games reach L2+ ONLY via hand-built per-game `GameAdapter`s (development_proxy,
e.g. tn36's 6-bit program-editor RE, m0r0's 93-action sequence) — the GENERIC agent reaching L2 is
genuinely unproven AND, per Finding 1, has never been measured.

## FINDING 4 (the genuine measurement, first ever): real LLM induction fires at L1 but yields NO L2 plan

`MULTILEVEL_ARM=real` ran the live cascade with a real `LocalGGUFProposer` on lp85, budget 800
(`results/proto_multilevel_diag_real.json`). The FIRST time the live agent has ever been rolled past
L1 with real LLM induction. Result:

```
lp85  maxL=1  L1@action20  induced=True  final_phase=explore  plan_len=0  transitions=792  wall=176.6s
```

**Induction FIRED (induced=True, 792 transitions, 176s vs the 3.4s no-op arm = real LLM calls
happened) but produced an EMPTY PLAN (plan_len=0).** Per `next_move`, an empty plan after induce
routes the agent back to `phase=explore` — so it re-induced repeatedly across the episode, never once
produced an actionable sequence to the 2nd win, and stayed at L1. **The L1->L2 stall is in INDUCTION:
the proposer cannot induce a goal-predicate + plan that reaches L2 from the L1-boundary evidence.**
That is the generation-guidance wall, one level deep — exactly what the `.429 A1/A2 levers target.

**Two measurement-hygiene caveats (do not over-read this single run):**
1. **LLM-reuse confound (gemma, not Qwen).** `LocalGGUFProposer.port` defaults to `8919`, and a
   persistent **gemma-4-12B-it** llama-server (PPID 1, 1d+ uptime, ROCm iGPU) was already healthy on
   8919, so `_ensure_server()` REUSED it rather than spawning Qwen3.5-9B-MTP. The induction therefore
   ran on gemma-4-12B, not the frozen-stack Qwen. This is *conservative* for the finding (gemma-12B ≥
   Qwen-9B in raw capability, so if gemma can't induce an L2 plan, Qwen is unlikely to) — but a clean
   Qwen-on-a-fresh-port re-run is the right confirmation. **It is also a real local-measurement bug:
   any "Qwen" proposer run on this box silently talks to whatever server squats port 8919.** On Kaggle
   (fresh env) this doesn't bite; in local dev it confounds every generation measurement.
2. **n=1 game.** lp85 only. The 3 L1-reaching games (lp85/sc25/vc33) should all get the real arm; the
   6 that don't even reach L1 with exploration need induction for L1 too.

## Bottom line / recommended iteration

1. **Fix the metric harness FIRST** (the actionable lever): make the live multi-level rollout use
   `target_levels >= 2`, drop the break-at-first-win, AND give the proposer a non-colliding port (or
   verify the right server). Until then `live_multi_level_solve_rate` is a constant 0.0 that no `.429
   lever can move. Flagging this to the conductor/operator is the highest-value output here.
2. The wall is in GENERATION/INDUCTION at the L1->L2 boundary (proposer fires, plan empty), NOT
   exploration — consistent with the whole-session diagnosis and the `.429 pivot. The lever should
   improve L2-goal-predicate induction + planning at the level boundary, not exploration breadth.
3. Fix the port-8919 server-reuse hygiene bug so local generation measurements use the declared model.

## Bottom line / recommended iteration

1. **Fix the metric harness FIRST** (this is the actionable lever): make the live multi-level rollout
   use `target_levels >= 2` and drop the break-at-first-win, with the REAL proposer. Until then
   `live_multi_level_solve_rate` is a constant 0.0 that no `.429 lever can move — flagging this to the
   conductor/operator is the highest-value output of this diagnostic.
2. The wall (if Finding 4 confirms it) is in GENERATION/INDUCTION at the L1->L2 boundary, NOT
   exploration — consistent with the whole-session diagnosis (generation-guidance is the live wall)
   and the `.429 pivot direction.

Code: `scripts/experiments/proto_multilevel_diag.py`; data: `results/proto_multilevel_diag_noop.json`
(+ `_real.json`). Cross-refs: `results/experiment_4646_live_multi_level_solve_rate_metric.json`
(the degenerate metric), `h2h-just-explore-vs-bare-explorer-2026-06-23.md` (the prior thread),
`ops/arc_solve_registry.yaml` (the per-game adapter context).
