# Research Roadmap — Milestone 2026.06.422

**ACTION EFFICIENCY: ship the leaderboard's #1 lever (a CNN clickability / action-effect predictor) re-aimed at actions-to-first-levelup, + verifier-guided candidate EXPANSION (fix the "winner-not-in-pool" root cause)**

- **Planned by:** outer-loop (Claude Opus 4.8 planner), 2026-06-21
- **Milestone:** 2026.06.422 (CalVer: June 2026, seq 422)
- **Prior milestone:** 2026.06.421 (OPERATIONALIZE THE VERIFIER WIN — near-total honest null)
- **Sprint context:** ARC-AGI-3 Kaggle submission sprint through **2026-06-30** (9 days). ARC-majority milestone per the Submission Sprint Forcing Function (CLAUDE.md).

---

## 1. What the previous milestone (.421) proved

.421 set out to OPERATIONALIZE the one .420 win (the cross-game DiscriminativeVerifier, LOO-AUROC 0.674,
oracle-distinct) by wiring it into the LIVE solver, and to make a 4th-and-final re-induction attempt. The
capstone (exp4566, `complete: verifier_router_null_reinduction_retired_or_refined`) is a **near-total
honest null**:

| Phase | Result | Status |
|---|---|---|
| A1 (HEADLINE) verifier-router → generic_transfer | `generic_transfer_delta = 0.0`; `random_router_control_passed = false`; `solve_rate_preserved = false` | **NULL — failed positive control, flagged_adversarial, EXCLUDED from capstone** |
| A2 (4th re-induction) executable world-model proposer | `positive_control_passed = false` (proposer failed before CORE measurement) | **NULL → re-induction lever RETIRED (`reinduction_retired: true`)** |
| A3 level-up bank | target m0r0 L2 already banked → 0 new | no new level |
| A4 hidden-field state probe | `offline_reproduced = false`, 0 banked | no new level |
| A5 integration | `DURATION_TOO_SHORT` (36.7s, compute markers, no model_specs) + false_negative_risk | **flagged_adversarial, EXCLUDED** |
| A6 self-learning transfer | `verifier_router_candidate_ranking_operator` persisted; `ordering_gain = 0` on tu93/tr87/sc25 | **transfer null** |
| B1 co-headline metric | `generic_transfer_rate_over_variants = 0.04`, CI [0.0, 0.1], 2/50 variants solved | shipped (clean) |

**The plateau is now structural.** `core_efficiency` has been pinned at **2.0074** for FOUR milestones
(.418/.419/.420/.421); `generic_transfer` at **0.04** for two; `reproducible_total_levels` at **52**.

**The decisive diagnostic (A6):** a verifier that **RE-RANKS a fixed candidate pool** has `ordering_gain=0`
because **the winning candidate is never IN the pool** ("no offline-reproduced target candidate for the
ranker to promote"). The verifier moat (oracle-distinct, real as a discriminator) is **useless as a
re-ranker** on first-contact unseen games. The bottleneck is **candidate GENERATION / exploration**, not
ranking. Re-running the re-ranker is churn; the lever is the generator.

---

## 2. The strategic pivot — ACTION EFFICIENCY

Two independent signals converge on the same lever:

1. **The leaderboard winners** (operator 2026-06-20 dive,
   `docs/research-notes/arc-leaderboard-competitive-intel-2026-06-20.md`): the leader **StochasticGoose /
   Tufa Labs (12.58%)** and the top public agents all win on a **learned CNN action-effect / clickability
   predictor** — predict which actions/clicks cause a FRAME CHANGE, stop wasting the budget on no-ops. Action
   efficiency (`min(human/agent,1)²`) is the scoring metric; we have **0.08**, the leader **1.21**. This is
   the **operator-flagged #1 score lever** (GAP-ARCH-FRAME-CHANGE-PREDICTOR).

2. **Our own .421 root cause:** the verifier can't help when the winning candidate isn't generated. Better
   candidate GENERATION (action-efficient exploration) puts the winner in the pool.

**The blocker is cleared.** exp4490 (the frame-change predictor) was `blocked_human_replay_corpus_not_cached`.
The corpus is now LOCAL: `data/arc_public_demo_human_replay_corpus` (14,672 rows, CC BY 4.0) +
`data/arc_transition_corpus/*.npz` (14,628 self-captured transitions, 25 public games). The outer-loop's
`arc_pretrain_prior` USED this data but aimed at the WRONG target — exact next-cell DYNAMICS prediction
(`warm_wins 0/5`, **0% accuracy both scratch and warm**). The leaderboard target is COARSER: a clickability
heatmap + per-action effect probability, evaluated by **actions-to-first-levelup**, not pixel-perfect
dynamics. .422 re-aims at the right target.

**.422 = two complementary attacks on action efficiency / generic_transfer, plus the mandatory level-up bank:**

- **A1 (HEADLINE):** the clickability / action-effect predictor — better candidate ORDERING by P(frame-change).
- **A2:** the cross-game DiscriminativeVerifier as a FRONTIER-EXPANSION priority — grows the search so the
  winner is GENERATED (fixes the A6 root cause). Oracle-distinct (`verifier_is_oracle: false`).

These are the two halves the frame-change spec's energy-augmentation later fuses (`P(frame_change)·(−ΔE)`);
.422 measures each in isolation, A5 integrates the winners.

---

## 3. Architecture — where the two levers wire in

```
                    ARC-AGI-3 first-contact solve (UNSEEN game / variant)
                                       │
                    ┌──────────────────┴───────────────────┐
                    │  live explorer (graph_explore_solve_v2 /          │
                    │  rich_action_candidates) — the GENERATOR          │
                    └──────────────────┬───────────────────┘
                                       │  enumerated candidate actions / clicks
          ┌────────────────────────────┼────────────────────────────┐
          ▼                            ▼                            ▼
 ┌────────────────────┐  ┌───────────────────────────┐  ┌────────────────────┐
 │ A1: clickability /  │  │ A2: cross-game            │  │ existing salience   │
 │ action-effect CNN   │  │ DiscriminativeVerifier    │  │ ordering (baseline) │
 │ predict(frame) ->   │  │ as FRONTIER-EXPANSION     │  │                     │
 │ (click_heatmap,     │  │ priority (best-first):    │  │                     │
 │  dir_change[5])     │  │ GROWS the search toward   │  │                     │
 │ RANK by P(change)   │  │ the goal (oracle-distinct)│  │                     │
 └─────────┬──────────┘  └────────────┬──────────────┘  └────────────────────┘
           │                          │
           └────────────┬─────────────┘
                        ▼
     candidate ORDER + frontier EXPANSION  →  winner more likely GENERATED & tried EARLY
                        │
                        ▼
     METRIC: median actions-to-first-levelup ↓ (action efficiency ↑)  +  generic_transfer ↑
                        │
                        ▼
     A5 integration → SUBMITTED agent   •   A6 persist as cross-game action-effect memory
```

- **A1 is a LEARNED action-model** (a different paradigm from the verifier-energy thesis). It COMPLEMENTS:
  it makes the tier-1 explorer action-efficient; the verifier/energy still routes + grounds. Trained
  self-supervised on the local corpora, CPU forward pass at eval (sub-ms, live-legal, frame-only). **No
  LLM, zero quota.**
- **A2 keeps the verifier thesis central** — the oracle-distinct moat result, now used where it can add value
  (growing the frontier), not where it can't (re-ranking a pool without the winner).

---

## 4. Phase descriptions

### PHASE 0 — Transition (exp4567)
Archive .421 → activate .422; assert YAML parses + smart-subset pre-test gate green; record the TRUE .421
close-state (4-milestone efficiency plateau 2.0074; generic_transfer 0.04; reproducible_total_levels 52;
re-induction RETIRED; A6 root cause = winner-not-in-pool). `aggregation_from_upstream_artifacts`.

### PHASE A — ARC NORTH STAR (majority; the sprint)
- **A1 (HEADLINE, exp4568):** CNN clickability / action-effect predictor. Train `predict(frame) ->
  (click_heatmap[H×W], directional_change[5])` on the LOCAL 14.6k human-replay + transition corpora; label =
  did the action change the frame (binary) + where. Wire into `rich_action_candidates` to RANK candidates by
  predicted change. **Gate:** held-out median actions-to-first-levelup STRICTLY lower than blind BFS (+ a
  learnable-clickability positive control + FALSE_NEGATIVE_RISK guard); must NOT drop solve-rate.
  `verifier_is_oracle: false`. Re-aimed at the clickability/efficiency target (NOT the 0% dynamics target).
- **A2 (exp4569):** the cross-game DiscriminativeVerifier (LOO-AUROC 0.674) as a FRONTIER-EXPANSION priority
  in best-first search — grow the candidate set toward the goal so the winner is GENERATED. **Gate:**
  verifier-guided expansion raises generic_transfer or reaches the goal in fewer expanded states, vs a
  random-priority positive control; bounded expansion (Scaling-Flaws guard). `verifier_is_oracle: false`.
  Fixes the A6 ordering_gain=0 root cause.
- **A3 (LEVEL-UP GUARANTEE, exp4570):** bank +1 NEW reproducible level — rotate to a game NOT deepened
  recently (cn04/sk48/ar25 L1→L2, graph/world-model family). Gate: offline_reproduced. 52 → 53+.
- **A4 (exp4571):** hidden-field state-hash probing, re-scoped to the SINGLE game with a named readable
  register (ka59 StepCounter), gate-FIRST on a state-disambiguation positive control (so a still-aliased
  search retires cleanly). The deepening-tail fix (GAP-ARCH-GRID-ONLY-STATE).
- **A5 (exp4572):** integration — wire whatever RAISED a real metric (A1 ordering, A2 expansion, A4 bank)
  into the SUBMITTED agent; re-measure BOTH metrics (action efficiency + generic_transfer). Honest null if
  nothing rose. Keep `test_arc_submitted_agent_parity.py` green.
- **A6 (SELF-LEARNING, exp4573):** persist the milestone's winning primitive (the clickability predictor as
  a reusable cross-game action-effect MEMORY, per PersistentAEM) into `arc_solver_kit` + `arc_solve_registry`;
  measure CROSS-GAME transfer. (research-program.md mandates ≥1 continuous-self-learning experiment — Tier-3
  predictive verification / Tier-2 persistent memory.)

### PHASE B — Reserved infrastructure (2 slots)
- **B1 (exp4574):** make **action efficiency** (`median_actions_to_first_levelup` + `min(human/agent,1)²`,
  with the human baseline from the replay corpus) a CO-HEADLINE capstone metric alongside generic_transfer +
  reproducible_total_levels. This is the metric A1 moves; it is currently unreported. Asserting tests.
- **B2 (exp4575):** substrate-declaration guard for learned-CNN action-model artifacts — the recurring
  `DURATION_TOO_SHORT` false-positive (it quarantined the .421 A5; a fast CPU CNN forward pass + torch
  markers trips the live-model 60s floor). Ensure a fast-but-real CNN action-effect artifact is recognized,
  + a regression assert. Asserting tests.

### PHASE C — Hardware continuity (1 slot, exp4576)
Per-board reachability audit: KV260 (SSH only, never host SD card), GateMate (USB detect), PolarFire (SSH).
Honest `blocked_<board>_<reason>` per board.

### PHASE D — SOTA ingestion (1 slot, exp4577)
Ingest SOTA on learned action-effect / clickability models for interactive agents + verifier-guided
candidate EXPANSION / PRM generalization (arXiv:2502.18407 AgentRM, 2504.16828 ThinkPRM, 2502.00271
scaling-flaws, the carried 2602.01070 / 2601.22607), mapped onto the A1/A2 headlines. Reliable channel only;
/deep-research BANNED in the autonomous loop. Flag the strongest method for .423.

### PHASE E — Capstone (1 slot, exp4578)
The scorecard: did the clickability predictor (A1) reduce median actions-to-first-levelup / raise
generic_transfer above 0.04? Did verifier-guided expansion (A2) add value (fix the winner-not-in-pool root
cause)? Did A3/A4 grow reproducible_total_levels? Report all THREE co-headline metrics (bank count + generic
transfer w/ CI + action efficiency). Skip flagged_adversarial except the annotated null-delta carve-out;
honor the B2 substrate guard + the .421 B2 positive-control-failed guard.

---

## 5. Dependency graph

```
exp4567 (PHASE 0 transition)
   ├─> exp4568 (A1 clickability predictor)  ─────────────┐
   ├─> exp4569 (A2 verifier-guided expansion) ───────────┤
   ├─> exp4570 (A3 level-up bank)            ─────────────┤
   ├─> exp4571 (A4 hidden-field probe)       ─────────────┤
   ├─> exp4574 (B1 action-efficiency metric) ──┐          │
   ├─> exp4575 (B2 substrate guard)            │          │
   ├─> exp4576 (C hardware)                    │          │
   └─> exp4577 (D SOTA ingestion)              │          │
                                               ▼          ▼
        exp4572 (A5 integration) <── reads A1/A2/A4 winners, B1 metric
                 │
                 ▼
        exp4573 (A6 self-learning / persist + transfer) <── persists A1/A2 primitive
                 │
                 ▼
        exp4578 (E CAPSTONE .422) <── aggregates A1–A6 + B1 (skips flagged; honors B2 guards)
```

A1–A4, B1–B2, C, D run independently after the transition. A5 reads the A1/A2/A4 winners + the B1 metric;
A6 persists the winning primitive; E aggregates. No task `requires:` a retired exp_id.

---

## 6. Hardware requirements

- **A1 / A2 / A3 / A4 / A6:** CPU + offline arcade; A1 trains a small CNN (torch 2.11 in venv; CPU or iGPU
  Radeon 890M — NEVER the RTX 3090s for the live path). No LLM, zero quota — these are the deadline-friendly,
  leaderboard-proven non-LLM levers.
- **A5:** offline arcade end-to-end; if the integrated config invokes the live generator, it is the frozen
  **Qwen3.5-9B-MTP** on the iGPU (project memory `arc_live_generator`), never the 3090s.
- **B1 / B2 / D / E:** CPU only (metrics, guards, literature, aggregation).
- **C:** SSH/USB to the attached boards (KV260, GateMate, PolarFire).

No new hardware required. No SOTA GGUF MODEL_SPECS required — the .422 core levers are CNN/search/offline
and deliberately LLM-free (quota conservation + the score lever is the non-LLM action-effect model).

---

## 7. Success metrics (the honest scorecard for E)

| Metric | Baseline (.421) | .422 target |
|---|---|---|
| **median actions-to-first-levelup (held-out)** | unmeasured (NEW co-headline, B1) | A1: STRICTLY lower than blind BFS, CI-backed |
| **generic_transfer_rate_over_variants** | 0.04 | A1/A2: STRICTLY > 0.04, CI excludes baseline |
| **reproducible_total_levels** | 52 | A3 (+A4): 53+ (monotonic) |
| **action-efficiency `min(human/agent,1)²`** | unreported | reported with human baseline (B1) |

**Discipline gates honored:** Verdict Terminal-Prefix; Inference-Substrate Declaration (in REQUIRED ARTIFACT
FIELDS so the agent EMITS it — the .410 lesson); Pre-Launch Preconditions; Principle-Annotated Fields;
Circularity / Oracle-Distinctness (`verifier_is_oracle: false` on A1/A2); Failed-Experiment Rerun +
Exclusion-Manifest (prior_failures on A1/A2/A4, operator_override on routine/continuation tasks);
Missing-Verifier Gap Logging; ARC Level-Up Attempt Guarantee (A3); ARC Solve Reproducibility (offline gate);
Submission Sprint Forcing Function (ARC-majority, monotonic levels, codex experiments / Opus planner+retro).

---

## 8. Why this milestone advances the headline (not churn)

Per north-star §1, every milestone must advance the headline claim, close a gate, or test a load-bearing
unproven link. .422:

- **A1** attacks action efficiency — the actual leaderboard metric — with the proven #1 technique, finally
  unblocked (corpus local) and correctly aimed (clickability, not the 0% dynamics target). Direct headline.
- **A2** fixes the diagnosed .421 root cause (winner-not-in-pool) by using the oracle-distinct verifier where
  it can add value (frontier expansion). Direct headline.
- **A3** banks a new reproducible level (monotonic growth). Guarantee.
- It does NOT re-run the RETIRED re-induction lever, nor re-run the verifier as a re-ranker (the .421 null).

If both A1 and A2 null with passing positive controls, the milestone still produces (a) the new action-
efficiency co-headline metric (B1), (b) a banked level (A3), and (c) a sharpened generation-gap diagnosis —
honest progress, not churn.
