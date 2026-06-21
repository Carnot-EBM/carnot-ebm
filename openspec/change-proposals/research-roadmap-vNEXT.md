# Research Roadmap 2026.06.421 — OPERATIONALIZE THE VERIFIER WIN + ATTACK THE FIRST-CONTACT CEILING

**Milestone:** 2026.06.421
**Planned by:** outer-loop (Claude Opus 4.8 planner, 2026-06-21)
**Sprint:** ARC-AGI-3 submission sprint through **2026-06-30** (CLAUDE.md forcing function; 9 days out)
**Prior milestone doc:** this file supersedes the .420 design doc in place.

---

## 1. What the previous milestone (2026.06.420) proved

.420 brought the FROZEN sprint LLM (Qwen3.5-9B-MTP) into per-level GOAL RE-INDUCTION as the
proposer, with a verifier-guided refinement loop. The honest scorecard (capstone exp4554:
`complete: llm_proposer_null_efficiency_unmoved_barrier_refined`):

| Phase | Result | Bottom line |
|---|---|---|
| **A1** LLM-proposer re-induction (HEADLINE) | **NULL — positive_control FAILED** (exp4544) | The live Qwen proposer produced **ZERO reachable plans** (`llm_proposer_value.count=0`, `rate=0.0`), even on a fixture with a KNOWN reachable plan. `false_negative_risk_checked: false`. This is a **broken-integration** signal, not an exhausted idea. `core_efficiency` unchanged at 2.0074. |
| **A2** cross-game verifier discrimination | **WIN** (exp4545) | **LOO-AUROC 0.674, CI [0.606, 0.745] — excludes 0.5.** In-sample 0.871, positive control passed. The **first oracle-distinct verifier-moat result**: the DiscriminativeVerifier beats chance on a game it never saw (`verifier_is_oracle: false`). |
| **A3** level-up bank | **WIN** (exp4546) | su15 L1→L2 offline-reproduced. `reproducible_total_levels` 51→**52**. |
| **A4** frame-change CNN (action efficiency) | **NULL** (exp4547) | Held-out median-actions-to-first-levelup already at the **floor (1.0 == 1.0)**; the CNN learned the signal (positive control passed) but the metric had no headroom. Solve-rate preserved. |
| **A5** integration | **NULL** (exp4548) | No lever raised `core_efficiency`. SUBMITTED config unchanged. |
| **A6** primitive persist + transfer | **NULL** (exp4549) | Transfer re-induced predicates on sc25/tr87/tu93 but produced **no reachable deeper-level plan** (all `reachable_plan_produced: False`); no new bank. |
| **B1** honest sprint metric (SHIPPED) | exp4550 | **`generic_transfer_rate_over_variants = 0.04`.** The 52 banked levels are mostly **known-game replays (~0 on the hidden eval)** — the bank count is largely a mirage; generic transfer is the real leaderboard signal. |

**The three load-bearing facts for .421:**

1. **`core_efficiency` has been stuck at 2.0074 for THREE consecutive milestones (.418, .419, .420).**
   The "reach a deeper CORE level via per-level re-induction" lever has not moved. The .420 failure
   mode is specific and informative: the proposer's **free-form plan output is unreachable** (positive
   control failed). This is a 4th-attempt doomed-rerun risk unless the proposer mechanism is changed.

2. **A2 is the one genuine breakthrough — and it is unexploited.** A cross-game verifier that beats
   LOO chance is exactly the oracle-distinct moat north-star §5 is built around. But it is still a
   **bench result**: it has never been wired into the LIVE solver to prove it adds VALUE. That is the
   highest-leverage move available.

3. **The honest leaderboard ceiling is `generic_transfer_rate = 0.04`, not the bank count.** Confirmed
   independently this sweep: the ARC-AGI-3 leaderboard metric is **first-contact action efficiency on
   UNSEEN environments** (frontier models sit at 0.37%; arXiv:2603.24621). Banking more KNOWN-game
   levels barely moves it. The score-movers are GAP-LIVE-INTEGRATION (the submitted agent) and
   GAP-ARCH-FEATURES (cross-game verifier transfer — which A2 just cracked).

## 2. The .421 pivot (in one sentence)

**Stop billing "reach a deeper CORE level via re-induction" as the headline (3× null); instead
OPERATIONALIZE the now-above-chance cross-game discriminative verifier (A2's win) inside the LIVE
generic solver and measure it against the honest `generic_transfer_rate` ceiling (0.04) — turning the
oracle-distinct moat into a live first-contact capability — while keeping ONE re-scoped executable-
world-model proposer attempt (Family-B, gated on its positive control passing first, retire-if-same)
and the reliable level-up bank engine.**

This respects every standing discipline: ARC stays the majority; `reproducible_total_levels` grows
monotonically (A3 + A4 banks); ≥1 level-up attempt (A3); 2 reserved infra; 1 hardware-continuity;
1 SOTA-ingestion; all experiments codex/gpt-5.5 (planner/retro stay Claude Opus); live stack frozen.

## 3. Architecture — where each .421 task plugs in

```
                         ARC-AGI-3 first-contact solver (the SUBMITTED agent)
                         make_carnot_agent -> E3AgentPolicy
   +----------------------------------------------------------------------------------+
   |  perception        candidate gen          ROUTING / RANKING        induction      |
   |  (frame->features)  rich_action_candidates  +-------------------+  _induce_and_plan|
   |  cross_game_features_v3 ------------------>  | A1: Discriminative|  +-------------+ |
   |  (arc_value_learner)  graph_explore_solve_v2 | Verifier as the   |  | A2: Family-B| |
   |       ^              (arc_graph_explore)     | LIVE candidate    |  | executable  | |
   |  A4: hidden-field ---+                       | ROUTER (oracle-   |  | world-model | |
   |  state-hash probe                            | distinct moat)    |  | proposer +  | |
   |  (deepen ka59/ar25/ft09 L2)                  +---------+---------+  | CEGIS loop  | |
   +-----------------------------------------------------+--------------+------+-------+ |
                                                         |                     |         |
                       measured against ----------> generic_transfer_rate_over_variants  |
                       (B1 metric, baseline 0.04) + core_efficiency (baseline 2.0074)    |
                                                         |                     |         |
            A5 integration: wire winners into SUBMITTED_AGENT_CONFIG ----------+         |
            A6 self-learning: persist the winning primitive -> arc_solver_kit + registry |
            A3 level-up bank: rotate-deepen one game (offline_reproduced gate)           |
```

- **A1 (HEADLINE):** the A2 DiscriminativeVerifier (`arc_value_learner.py:497`, features `:394`)
  becomes a **LIVE candidate router** inside `rich_action_candidates` / `graph_explore_solve_v2`.
  Oracle-distinct (`verifier_is_oracle: false`). Measured on `generic_transfer_rate_over_variants`.
- **A2:** the failed re-induction proposer is re-built as a **Family-B executable Python world-model
  inducer** (arXiv:2605.05138) with held-out transition verification + bounded counterexample-guided
  refinement (arXiv:2606.11521), **gated first on positive_control_passed=True** (the .420 break).
- **A4:** hidden-field state-hash probing (GAP-ARCH-GRID-ONLY-STATE) to deepen the L2-stalled
  hidden-state games (ka59 step counter / ar25 undo stack / ft09) — a different bank source.
- **A3:** rotate-deepen one game; the Level-Up Attempt Guarantee anchor.
- **A5/A6:** integration + self-learning reuse.

## 4. Phases & tasks (12)

| # | id | Phase | What | Gate |
|---|---|---|---|---|
| 0 | exp4555 | Transition | archive .420 -> activate .421; record true close-state | YAML parses + pre-test green |
| 1 | exp4556 | **A1 HEADLINE** | DiscriminativeVerifier as LIVE candidate router; measure on generic_transfer (0.04) | generic_transfer up w/ CI **OR** measured no-value char. + positive control; `verifier_is_oracle: false` |
| 2 | exp4557 | A2 | Family-B executable world-model proposer (re-scope of 3x null re-induction) | **positive_control_passed=True FIRST**, then deeper CORE level / efficiency; retire-if-same |
| 3 | exp4558 | A3 | rotate-deepen one game (level-up bank) | `offline_reproduced` new level |
| 4 | exp4559 | A4 | hidden-field state-hash probe (deepen ka59/ar25/ft09 L2) | new L2 reproduced **OR** sharpened hidden-field gap + positive control |
| 5 | exp4560 | A5 | integration: wire winners into SUBMITTED agent; re-measure | core_efficiency + generic_transfer end-to-end; parity green |
| 6 | exp4561 | A6 | persist winning primitive + cross-game transfer | transfer measured; reproduced levels count |
| 7 | exp4562 | B1 infra | make `generic_transfer_rate` the co-headline capstone metric + harden variant CI | asserting tests |
| 8 | exp4563 | B2 infra | guard: positive_control_FAILED proposer task != valid null (false-neg-risk-open) | asserting tests |
| 9 | exp4564 | C hardware | per-board continuity audit (KV260 SSH / GateMate USB / PolarFire SSH) | per-board reachability |
| 10 | exp4565 | D SOTA | ingest verifier-as-cross-task-router + executable-inducer SOTA; flag for .422 | real arXiv IDs, no fabrication |
| 11 | exp4566 | E capstone | scorecard: did A1 raise generic_transfer? did A2 pass its control? both metrics | aggregation; skip flagged |

## 5. Dependency graph

```
exp4555 (transition)
   |-> exp4556 (A1 verifier-router) ----------+
   |-> exp4557 (A2 executable proposer) -------+
   |-> exp4558 (A3 level-up bank)              +-> exp4560 (A5 integration; gated on A1/A2/A4 deltas)
   |-> exp4559 (A4 hidden-state probe) --------+                |
   |                                                            +-> exp4561 (A6 persist+transfer; gated on A1/A2)
   |-> exp4562 (B1 metric)                                      |
   |-> exp4563 (B2 guard)                                       |
   |-> exp4564 (C hardware)                                     |
   |-> exp4565 (D SOTA) ----------------------------------------+-> exp4566 (E capstone, reads all)
```

A1/A2/A3/A4 are independent (parallel-safe). A5 gates on A1/A2/A4 producing a positive delta; A6 gates
on A1/A2 producing a persistable primitive. The capstone reads everything.

## 6. Hardware requirements

- **iGPU (Radeon 890M)** for the Qwen3.5-9B-MTP generator in A2 (NEVER the RTX 3090s — frozen
  live-generator selection, project_arc_live_generator).
- **CPU / offline arcade** for A1/A3/A4 (verifier scoring + offline-reproduction; no LLM load).
- **Attached boards** (C): KV260 (`ssh kria`), GateMate (`openFPGALoader -c dirtyJtag --detect`),
  PolarFire (`ssh polarfire`) — SSH/USB reachability only; KV260 SSH-not-SD-card.

## 7. Discipline compliance

- **Failed-Experiment Rerun:** A2 carries `prior_failures` (exp4544 + exp4533, all four sub-fields,
  `retire_if_same_verdict: true`) — the 4th and final re-induction attempt unless its positive control
  passes. The mechanism is genuinely different (executable Python world-model + held-out verification
  vs free-form plan) and the gate is tightened (positive-control-first).
- **Circularity / Oracle-Distinctness:** A1 sets `verifier_is_oracle: false` (a learned ranking signal,
  NOT the executable win-check) — a circular win does not count.
- **Operator-override** (false-positive scope-match classes only) on the routine transition (0),
  versioned-lineage continuations (A1/A3/A5/A6), hardware (C), and SOTA-ingestion (D).
- **Adversarial rigor:** every ARC efficiency/transfer claim emits an explicit delta + null-delta note
  (TAUTOLOGY carve-out), a positive control, and FALSE_NEGATIVE_RISK guard; `inference_substrate` is a
  REQUIRED ARTIFACT FIELD on every task; SOTA cites real arXiv IDs.
- **Sprint compliance:** ARC majority (A1-A6 + capstone); monotonic `reproducible_total_levels`
  (A3 + A4); >=1 level-up attempt (A3); 2 infra (B1/B2); 1 hardware (C); 1 SOTA (D); codex/gpt-5.5.
