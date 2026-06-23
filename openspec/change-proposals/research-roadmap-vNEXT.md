# Research Roadmap — Milestone 2026.06.425

**Status:** PROPOSED (pre-staged by the outer-loop Claude Opus 4.8 planner, 2026-06-23)
**Milestone doc for:** `research-roadmap-next.yaml` (milestone `2026.06.425`)
**Theme:** PIVOT FROM GENERATION (exhausted) TO THE VERIFIER / WORLD-MODEL SIDE — fix the
degenerate world-model trust gate that caps the scored agent at 0.08, and wire the
transfer-validated discriminative verifier into the SCORED agent.

---

## 0. ARC-AGI-3 submission sprint (active through 2026-06-30)

Per CLAUDE.md "ARC-AGI-3 Submission Sprint Forcing Function," every milestone through
2026-06-30 reserves the MAJORITY of its non-reserved slots for ARC-AGI-3 live-solving
progress, with `reproducible_total_levels` monotonically growing, ≥1 level-up attempt
that BANKS a new reproducible level (ARC Level-Up Attempt Guarantee), 2 reserved infra
slots, 1 per-attached-board hardware slot, and 1 SOTA-ingestion slot. All experiments are
`agent_type: codex` / `gpt-5.5`; planner and retro stay Claude Opus 4.8. **7 days to the
deadline.** The frozen live generator is Qwen3.5-9B-MTP on the AMD iGPU (NEVER the 3090s).

---

## 1. What the previous milestones proved (.420–.424)

**The reliable engine.** The standing self-play loop banks +1 reproducible level per
milestone: su15 L2 (.420), cn04 L2 (.422), ar25 L2 (.423), ft09 L2 (.424). Authoritative
state: **`reproducible_total_levels = 55`, 24 games, provisional = 1** (`ops/arc_solve_registry.yaml`).
The operator-resubmit package is at 55 levels; the last submitted scorecard was **33 levels**
(2026-06-21) — the standing gate to beat.

**The generation wall is exhausted (FIVE consecutive nulls).** The bottleneck is candidate
GENERATION on first contact, not selection — and every generation/selection lever has now
nulled:

| Milestone | Lever | Result |
|---|---|---|
| .421 A6 | re-rank a fixed candidate pool | ordering_gain = 0 |
| .422 A2 | verifier-guided frontier EXPANSION | regressed −0.04 |
| .423 A3 | feature-ROUTER (classify mechanic → route) | generic_transfer flat 0.04; winner_generated 1/25 |
| .424 A1 | WIRE the toolkit into the variant generation harness | quarantined; winner_generated 2/25; transfer flat 0.04 |
| .424 A3 | objective goal-distance ENERGY as a GENERATION prior | honest-null |

Both .424 generation levers carried `retire_if_same_verdict: true` and produced the same
verdict — so **"generation-completeness-wiring" and "energy-generation-prior" are retired
scopes.** Re-proposing them is forbidden.

**The decisive un-addressed fact.** Per `docs/research-notes/arc-008-wall-root-cause-2026-06-21.md`,
the 0.08 Kaggle ceiling has a single binding cause: **every world-model path is gated out by
the exact-full-grid-match `WorldModelVerifier`** (TTT 0/5; e3 LLM induction 0/6,
model-size-independent), so the scored `E3AgentPolicy` ALWAYS falls back to the bare explorer
floor (= 0.08). Separately, GAP-LIVE-INTEGRATION records that the scored agent ships bare BFS
+ a 0/6-value LLM tier, `target_levels=1`, `value_weight=0.0`, and **never imports** the
strategy router or the transfer-validated discriminative verifier — so the 55 reproduced
levels are "largely a leaderboard mirage." Neither the trust-gate fix nor the live-integration
wiring has been attempted as an experiment.

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The scored agent does not use Carnot's verifier (GAP-LIVE-INTEGRATION).** The
   transfer-validated cross_game_features_v3 DiscriminativeVerifier (LOO-AUROC **0.674**,
   above chance, oracle-distinct) exists but is not imported by `arc_competition_agent.py`.
   The single most direct move on the leaderboard score is to wire it (+ the strategy router,
   higher `target_levels`, forward-edge nav) into the SCORED `E3AgentPolicy`.

2. **The one oracle-distinct EBM slot on the ARC critical path is unfilled
   (GAP-ARCH-WORLD-MODEL-TRUST-ENERGY / GAP-WM-TRUST-GATE).** The binary `accuracy < 0.5` /
   exact-full-grid-match trust gate is degenerate (an identity engine scores 0.725 and
   false-passes on click-heavy games; an induced model that mispredicts one no-op cell is
   rejected). Replacing it with a CHANGE-WEIGHTED, non-degenerate consistency score AND a
   learned trust energy that ranks induced models by HELD-OUT generalization is the operator's
   energy-augmented graft #3 — the moat — and the documented 0.08-wall root-cause fix.

3. **Generalization across novel games.** Frontier <1%, all preview winners <13%; the named
   wall is GENERALIZATION. The literature (VFScale; "verifiers generalize better than
   generators"; World-in-World) says the verifier/energy side transfers where learned
   generation does not. The .425 pivot follows that evidence: invest in the verifier/energy
   side, not the exhausted generation side.

---

## 3. The .425 program (architecture)

```
                       SCORED LIVE AGENT (the deliverable; arc_competition_agent.py:E3AgentPolicy)
                                            |
        +-----------------------------------+--------------------------------------+
        |                                   |                                      |
   A1 TRUST ENERGY                     A2 LIVE INTEGRATION                    A6 INTEGRATION
   (oracle-distinct EBM moat:          (GAP-LIVE-INTEGRATION:                (consolidate whatever
    fix the degenerate                  wire the 0.674 discriminative         raised a metric into
    WorldModelVerifier gate ->          verifier + strategy router +          SUBMITTED_AGENT_CONFIG;
    change-weighted + held-out          raise target_levels +                 re-measure end-to-end;
    + learned trust energy;             forward-edge nav into the             parity test green)
    unblocks WM-induction 0/6)          SCORED explorer scoring)
        |                                   |                                      |
        +-------------> measured on held-out variants (first-win-rate, efficiency, <+
                        world_model_trust_pass_rate, live-submittable count)
                                            |
   A3 LEVEL-UP + SELF-PLAY  --------->  bank +1 reproducible level (55->56+) + train/checkpoint
   (the reliable engine; rotate target)    the learned verifier on pos/neg traces
                                            |
   A4 SCORE --> fold A3 bank into the refreshed operator-resubmit package (live-submittable > 33; operator-only)
   A5 SELF-LEARNING --> persist the milestone's winning primitive (A1 trust energy / A2 wired verifier) + measure cross-game transfer

   RESERVED: B1 co-headline metric (world_model_trust_pass_rate) . B2 adversarial_verify guard
             (degenerate/circular world-model-trust false-pass) . C hardware (per board) . D SOTA-ingestion . E capstone
```

**Why this is rule-compliant (ARC Live-Path Reachability Discipline, 2026-06-22).** A1 and A2
both IMPROVE the live path (`arc_competition_agent.py` + `WorldModelVerifier`) — they do not
build a parallel solver the live agent cannot reach. A3 is the development-proxy self-play loop
(`arc_loop_solve` + a hand `GameAdapter`); its `solve_provenance` is declared `development_proxy`
honestly. No task claims an outer-loop hand-RE solve.

### Phase A — ARC north star (operator-mandatory)

- **A1 (HEADLINE; oracle-distinct EBM moat) — World-model TRUST ENERGY.** Replace the degenerate
  full-grid-match / binary-`accuracy<0.5` `WorldModelVerifier` gate with (a) a CHANGE-WEIGHTED
  consistency score over grid-CHANGING transitions only + a non-degeneracy requirement (>=1
  correctly-predicted real change), and (b) a learned/calibrated TRUST ENERGY that ranks induced
  world-models by HELD-OUT (not prefix) misprediction. Gate: a world-model path now passes the
  trust gate AND is USED by the planner on hidden-state games where it currently 0/6 fails, with
  first-win-rate/efficiency up on held-out variants vs the binary-gate baseline (matched control;
  positive control + FALSE_NEGATIVE_RISK guard). `verifier_is_oracle: false` (ranks by held-out
  generalization, not by running the executable win-check). This is graft #3, the documented
  0.08-wall fix, and it unblocks the executable-WM-induction generator the .424 ingestion flagged.

- **A2 (highest mandatory; GAP-LIVE-INTEGRATION) — wire the transfer-validated stack into the
  SCORED agent.** Import the cross_game_features_v3 DiscriminativeVerifier (LOO-AUROC 0.674) into
  the `E3AgentPolicy` explorer scoring; import the strategy router; raise `SUBMITTED_TARGET_LEVELS`
  above 1; replace RESET-replay navigation with forward-edge `_shortest_path`. Measure end-to-end
  on held-out variants (first-win-rate, median actions-to-first-levelup, solve-rate) vs the bare
  config (matched control). Keep `test_arc_submitted_agent_parity.py` green. `verifier_is_oracle: false`.

- **A3 (LEVEL-UP ATTEMPT GUARANTEE + self-play every milestone).** Run the standing loop to bank
  +1 NEW reproducible level (registry-precheck: a game+level NOT already reproduced; rotate —
  prefer sk48 L1->L2, else a shallow L1 game; SKIP recorded dead-ends: ka59 hidden-register stall,
  cd82/sp80/su15 L3) AND train+checkpoint the learned verifier on this run's pos/neg traces.
  `solve_provenance: development_proxy`. Gate: `offline_reproduced` -> `reproducible_total_levels`
  55 -> 56+.

- **A4 (SCORE — keep the package fresh; operator-only).** Fold A3's bank (+ any A1/A2 newly-solved
  variant) into the refreshed operator-resubmit package; re-validate every claimed level
  offline-reproduces; live-submittable count stays STRICTLY > 33. Submission is OPERATOR-ONLY —
  this task PREPARES + offline-validates only and emits `ready_for_operator_submit`.

- **A5 (SELF-LEARNING + REUSE).** Persist the milestone's winning primitive (A1 trust-energy
  operator, or A2 wired-verifier integration helper) into `arc_solver_kit` + `arc_solve_registry`
  (ARC Solve Reproducibility + Solver-Reuse Discipline) and measure CROSS-GAME TRANSFER on 2-3
  untuned games. `verifier_is_oracle: false`.

- **A6 (INTEGRATION + HEADLINE METRIC).** Consolidate whatever RAISED a real metric (A1
  trust-pass-rate/first-win, A2 first-win/efficiency, A3 new bank) into `SUBMITTED_AGENT_CONFIG`;
  re-measure end-to-end on world_model_trust_pass_rate + first-win-rate + live-submittable count;
  keep parity green; honest null if nothing rose. `verifier_is_oracle: false`.

### Phase B — reserved infrastructure (2 slots)

- **B1 — co-headline metric `world_model_trust_pass_rate`.** The fraction of hidden-state games
  for which a world-model path passes the (new, change-weighted) trust gate AND is used by the
  planner — the direct measure of whether A1 cracked the 0.08-wall root cause. Reported side-by-side
  with `reproducible_total_levels`, live-submittable count, generic_transfer, winner_generated_rate,
  and action efficiency. Asserting tests.

- **B2 — degenerate/circular world-model-trust guard in `adversarial_verify.py`.** An ARC artifact
  claiming a world-model trust pass MUST declare `verifier_is_oracle: false` AND show >=1
  correctly-predicted grid-CHANGING transition (non-degeneracy) — else a degenerate identity-engine
  false-pass (the GAP-WM-TRUST-GATE failure mode) or a circular trust claim is flagged. Reader/guard
  only; does NOT change any solver. Asserting tests.

### Phase C — hardware continuity (1 per attached board)

- **C — per-board reachability audit:** KV260 (SSH reachability ONLY, never host SD card),
  GateMate (USB detect), PolarFire (SSH). Honest `blocked_<board>_<reason>` if a board is down.

### Phase D — SOTA ingestion (1 slot)

- **D — ingest world-model-trust / verifier-generalization SOTA** mapped onto A1 (trust energy)
  + A2 (live integration): VFScale (2502.01989), World-in-World (2510.18135), WMPO (2511.09515),
  Grounding Generated Videos in Feasible Plans (2602.01960), contrastive combinatorial generalization
  (2508.13113 / 2510.01853 / ConRep4CO), Executable World Models (2605.05138). Real arXiv IDs only;
  `/deep-research` BANNED in the autonomous loop (low-concurrency WebSearch/WebFetch + sweep helpers).

### Phase E — capstone

- **E — the .425 scorecard:** did A1 (trust energy) make a world-model path pass the trust gate +
  get used where it currently 0/6 fails (cracking the 0.08-wall root cause)? Did A2 (live integration)
  raise first-win-rate/efficiency on held-out variants? Did A3 bank +1 (55->56+)? Is the package
  operator-resubmit-ready above 33 (A4)? Report ALL co-headline metrics. Skip `flagged_adversarial`
  except the mechanical null-delta carve-out (.424 B2); honor the offline-arc-METHODOLOGY +
  learned-CNN-DURATION + positive-control-failed guards.

---

## 4. Dependency graph

```
exp4603 (phase0: archive .424 -> activate .425)
   +-> exp4604 (A1 trust energy) ------+
   +-> exp4605 (A2 live integration) --+  (A1 lands its WorldModelVerifier change before A2 wires the verifier+router)
   +-> exp4606 (A3 level-up self-play; INDEPENDENT — the guarantee holds even if A1/A2 null)
                                        |
   exp4604, exp4605, exp4606 ----------+-> exp4607 (A4 refresh package: folds A3 bank + A1/A2 variant solves)
                                        +-> exp4608 (A5 persist winning primitive + transfer)
                                        +-> exp4609 (A6 integration: wire winners into SUBMITTED_AGENT_CONFIG)
   exp4610 (B1 co-headline metric)  -- parallel reserved infra
   exp4611 (B2 adversarial_verify guard) -- parallel reserved infra
   exp4612 (C hardware)             -- parallel
   exp4613 (D SOTA ingestion)       -- parallel
   exp4604..exp4613 -------------------> exp4614 (E capstone .425 scorecard)
```

A3 is deliberately INDEPENDENT of A1/A2 so the ARC Level-Up Attempt Guarantee holds regardless
of whether the headline levers null. A4/A5/A6 gate on the A-phase artifacts but degrade
gracefully (aggregate what exists).

---

## 5. Hardware requirements

- **A1/A2/A3/A4/A5/A6/B1/B2:** CPU + the offline arcade simulator (`arc_solver_kit.offline_arcade`,
  zero quota, deterministic). No 3090. If any task invokes the live LLM proposer for a residual
  arm, it runs on the AMD iGPU Qwen3.5-9B-MTP (NEVER the 3090s) and declares `live_llm_inference`
  for that arm only.
- **C (hardware continuity):** SSH to `kria` (KV260) and `polarfire`; `openFPGALoader` USB detect
  for GateMate. `hardware_smoke` substrate.
- **D (SOTA ingestion):** network for arXiv / Semantic-Scholar; `aggregation_from_upstream_artifacts`.

---

## 6. Discipline compliance checklist

- ARC sprint: majority ARC (A1-A6); >=1 level-up bank (A3); 2 infra (B1/B2); 1 hardware (C);
  1 SOTA-ingestion (D); all experiments codex/gpt-5.5; planner/retro Opus.
- ARC Live-Path Reachability: A1/A2 improve the live path; A3 declares `solve_provenance: development_proxy`;
  registry-precheck on A3 (target a level NOT already reproduced); no outer-loop-RE solve claims.
- Circularity/Oracle-Distinctness: A1/A2/A5/A6 declare `verifier_is_oracle: false`; B2 guards the
  circular/degenerate trust-pass.
- NOT re-proposing retired generation scopes (generation-completeness-wiring, energy-generation-prior).
  A1/A2 are new-scope mandatory-priority pickups (GAP-ARCH-WORLD-MODEL-TRUST-ENERGY / GAP-LIVE-INTEGRATION),
  each cleared by an `operator_override:` citing the standing directive.
- Verdict terminal-prefix, principle-annotated artifact fields, pre-launch PRECONDITIONS,
  inference-substrate declaration, missing-verifier gap logging, operator-only submission: all honored.
- CalVer: `2026.06.425` (June, derived from today's UTC date).

**Cross-refs:** `docs/research-notes/arc-energy-augmented-strategy.md` (the spine),
`docs/research-notes/arc-008-wall-root-cause-2026-06-21.md` (the root cause A1 fixes),
`ops/verifier_gaps.md` (GAP-LIVE-INTEGRATION, GAP-ARCH-WORLD-MODEL-TRUST-ENERGY, GAP-WM-TRUST-GATE),
`ops/arc_solve_registry.yaml` (reproducible_total_levels=55), `research-references.md` (the .425 pre-sweep).
