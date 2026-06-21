# Research Roadmap — Milestone 2026.06.420

**Theme: TEACH THE PROPOSER TO INDUCE — bring the LLM generator into per-level
GOAL RE-INDUCTION. The proposer, not the trigger, is the bottleneck.**

Planned by: outer-loop planner (Claude Opus 4.8), 2026-06-21.
Sprint: ARC-AGI-3 submission sprint through 2026-06-30 (CLAUDE.md "ARC-AGI-3 Submission
Sprint Forcing Function"). Majority ARC; >=1 level-up attempt (ARC Level-Up Attempt
Guarantee); 2 reserved infra; 1 per-board hardware; 1 SOTA-ingestion. ALL experiments
`agent_type: codex` / `gpt-5.5`; planner + retro stay Claude Opus (operator quality choice).

---

## What .419 proved (and the pivot that makes .420)

`.419 BUILT the per-level GOAL RE-INDUCTION the `.418 A2 diagnosis prescribed (detect the
level boundary -> re-induce the L_{n+1} win-predicate -> route the frontier toward it ->
raise `target_levels` past 1). It NULLED on the headline, and in nulling it located the next
barrier down — precisely.

| `.419 phase | result | what it means for `.420 |
|---|---|---|
| A1 per-level re-induction | NULL: `core_efficiency` stayed 2.0074 | The re-induction MECHANISM is correctly wired (it triggers on a level-up), **but the OFFLINE DSL-only proposer cannot induce a reachable L_{n+1} plan.** The load-bearing finding (below). |
| A2 energy-trust next-level routing | NULL (flagged DURATION_TOO_SHORT, characterized) | Routed in the L1 frame with no reachable L2 plan to aim at — same root cause as A1: nothing to route toward. |
| A3 level-up attempt | **SUCCESS: sp80 L2 offline-reproduced** | `reproducible_total_levels` 50 -> 51. The Level-Up Guarantee was met (via a per-game GameAdapter, NOT the generic agent). |
| A4 integration | NULL (`efficiency_moved: false`) | No lever raised `core_efficiency`; submitted config unchanged. |
| A5 primitive transfer | representation generalized, 0 levels banked | The re-induction primitive re-induces a DIFFERENT L_{n+1} predicate on sc25/tr87/tu93 (representation transfers) but still produces no reachable PLAN. Same barrier. |

### The .419 A1 barrier refinement (exp4533) — the load-bearing finding

> `barrier_refinement`: **"post_level_reinduction_triggered_but_no_reachable_l2_plan;
> offline_dsl_attempt_outcomes=['proposer_failed_or_missing_root']."**
> `model_specs`: **"offline_dsl_induction_no_llm"**

The per-level trace tells the whole story: on lp85 the agent completes L0 (20 actions vs human
17), arrives at L1, then takes **0 actions at L1** — `induction_attempts: []`,
`reinduction_events: []`, `barrier_hint: "reinduction registered but offline DSL-only
induction did not produce a reachable post-L1 plan"`. **The trigger fires; the PROPOSER comes
up empty.** A1 ran the offline DSL inducer with NO LLM, and the DSL grammar cannot induce the
L2 goal/transition rule from the post-transition frames.

### This converges with the 2026-06-19 GAP-LIVE-INTEGRATION operator audit

The operator's step-back audit found the 0.08 score's real ceiling is the **SUBMITTED agent,
not the solver research**: it ships **bare BFS, `target_levels=1`, `value_weight=0.0`, and an
LLM tier with 0/6 measured value** (`make_carnot_agent -> E3AgentPolicy`). The local gate even
sets `CARNOT_ARC_DISABLE_INDUCTION=1`, so per-level efficiency is measured on the BARE explorer
with no L2 mechanic at all. The LLM proposer (`LocalGGUFProposer(repo_substr="Qwen3.5-9B-MTP")`,
already wired in `_proposer()`) is the one component that could supply the proposal distribution
the offline DSL lacks — and its value has never been measured. **The .419 barrier and the
operator audit name the same gap.**

### The .419 SOTA-ingestion (PHASE D) already flagged the fix for .420

`research-studying.md:flagged_for_v420` reads: *"Family-B executable re-induction loop for each
level-up, with separate GOAL-vs-dynamics candidates, adaptive behavior tests for goal-shift
detection, and a bounded refinement loop around exp4533."* The discover->ingest->plan loop
closed: the literature pass points exactly at the LLM-proposer headline below.

---

## The .420 bet: the GENERATOR does the induction; the energy VERIFIER routes/verifies

This is the project's HYBRID architecture (north-star §0/§5) applied to the ARC critical path.
Energy-as-generator is closed-negative; the GENERATOR is the learned model (here the FROZEN
sprint LLM, Qwen3.5-9B-MTP). The energy ensemble is the VERIFIER. The .419 null is the predicted
failure of DSL/energy-as-generator; the fix is to bring the LLM generator in as the proposer and
keep the energy as the oracle-distinct router/verifier.

```
   level-up detected (levels_completed bumps)
            |
            v
   [GENERATOR]  Qwen3.5-9B-MTP proposes, from post-transition frames:
                (a) the L_{n+1} GOAL predicate         (separate GOAL candidate)
                (b) the L_{n+1} transition/world-model (separate DYNAMICS candidate)
                (c) a candidate action PLAN toward the goal
            |
            v
   [VERIFIER]  world-model TRUST ENERGY (verifier_is_oracle: false) ranks the
               induced candidates by held-out generalization; the execution
               check scores whether the plan reaches the goal
            |
       plan reaches L_{n+1}?  --no-->  bounded refinement loop (<=K rounds):
            |  yes                      feed the verifier's counterexample back
            v                           to the LLM (ALGO / counterexample-guided)
   advance into L_{n+1}  ==>  core_efficiency rises above 2.0074
```

Literature anchors (filed in research-references.md this milestone): Executable World Models for
ARC-AGI-3 (arXiv:2605.05138, the SOTA Family-B inducer+verifier), ARC-AGI-3 Tech Report
(arXiv:2603.24621, names goal-acquisition as the gap), ALGO (arXiv:2305.14591) and LLM Priors
for ERM over Programs (arXiv:2510.14331, LLM-as-proposal-distribution + execution selection),
Procedural Refinement / Counterexample-Guided Learning (arXiv:2603.20334, 2606.11521, the
bounded verifier-feedback refinement loop).

---

## The three biggest gaps this milestone attacks

1. **The proposer/generator gap (HEADLINE, A1).** The generic per-level re-induction proposer is
   offline-DSL-only and produces empty plans. Bring the LLM generator in; MEASURE its value
   (turn the "0/6 measured value LLM tier" into a measured proposer); bounded verifier-guided
   refinement. Score lever: `core_efficiency` > 2.0074 on a CORE game.

2. **The oracle-distinct verifier-moat gap (A2).** The cross-game DiscriminativeVerifier is
   in-sample 0.726 but **LOO-AUROC 0.503 == chance** (GAP-ARCH-FEATURES, 2026-06-19). Frame-only
   order-1 features do not transfer. Add relational / Δframe / action-conditioned / predicate-
   distance features and re-run the LOO gate. This is the verifier's open, oracle-distinct claim
   (north-star §5): can it discriminate above chance on a game it never saw?

3. **The action-efficiency gap (A4).** Our agent is effect-blind (centroid-click + RESET-replay);
   the leaderboard leaders win on a CNN action-effect/clickability predictor
   (GAP-ARCH-FRAME-CHANGE-PREDICTOR, 2026-06-20). The 14,672 labeled human-replay transitions are
   now cached (`.416 B1 exp4495). Train the CNN; wire into `rich_action_candidates`; gate on
   held-out median actions-to-first-levelup STRICTLY below blind BFS at preserved solve-rate.

---

## Phases

### PHASE 0 — TRANSITION (exp4543)
Archive .419 -> activate .420; assert the YAML parses + the smart-subset pre-test gate is green;
RECORD the true .419 close-state (proposer is the bottleneck; A3 banked sp80 L2;
`reproducible_total_levels`=51; `efficiency_moved`=false).

### PHASE A — ARC NORTH STAR (majority; operator MANDATORY)
- **A1 (exp4544, HEADLINE):** LLM-generator-as-L_{n+1}-proposer + bounded verifier-guided
  refinement. Qwen3.5-9B-MTP proposes the GOAL predicate + DYNAMICS + plan on a level-up; the
  world-model trust energy ranks candidates; a bounded refinement loop retries on a verifier
  counterexample. Measure WITH the LLM proposer vs the offline-DSL baseline (the .419 A1
  control). `live_llm_inference` (iGPU, NEVER the 3090s). Gate: a CORE game reaches L2
  (`core_efficiency` STRICTLY > 2.0074) at preserved CORE solves, OR a measured proposer-value
  characterization (the LLM produces a reachable L2 plan/predicate the DSL could not).
- **A2 (exp4545, ORACLE-DISTINCT MOAT):** richer cross-game verifier features -> re-run the
  DiscriminativeVerifier LOO-AUROC gate. `verifier_is_oracle: false`. Gate: LOO-AUROC STRICTLY
  > 0.5 with CI excluding 0.5 + positive control + FALSE_NEGATIVE_RISK.
- **A3 (exp4546, LEVEL-UP GUARANTEE):** bank +1 NEW reproducible level — deepen a shallow game
  (su15/cn04/sk48 L1->L2) or first-contact rotation, via `arc_loop_solve`. `offline_reproduced`.
- **A4 (exp4547, ACTION EFFICIENCY):** train the CNN frame-change/clickability predictor on the
  cached human-replay corpus; wire into `rich_action_candidates`. Gate: held-out median
  actions-to-first-levelup STRICTLY lower than blind BFS at preserved solve-rate.
- **A5 (exp4548, INTEGRATION + HEADLINE METRIC):** wire whatever RAISED `core_efficiency` (A1 LLM
  proposer / A4 CNN ranker) into `SUBMITTED_AGENT_CONFIG`; re-measure end-to-end on the per-level
  gate; keep parity green. Honest null if nothing raised it.
- **A6 (exp4549, SELF-LEARNING + transfer):** persist the LLM-proposer re-induction +
  bounded-refinement primitive to `arc_solver_kit` / registry; measure cross-game transfer
  (Tier-2 constraint memory; ARC reuse discipline).

### PHASE B — RESERVED INFRA (2 slots)
- **B1 (exp4550):** honest sprint-metric reporting — wire the SHIPPED variant benchmark
  (`arc_leaderboard_eval --variant/--reflect`) into the capstone metric so
  `reproducible_total_levels` is reported ALONGSIDE `generic_transfer_rate_over_variants`,
  surfacing the "mirage" (banked replays of known games ~0 on the hidden eval). Asserting tests.
- **B2 (exp4551):** offline-eval / live-submission PROPOSER PARITY guard — a regression assert
  that the offline `core_efficiency` harness measures the SAME proposer/config the SUBMITTED
  agent ships (or explicitly flags the `CARNOT_ARC_DISABLE_INDUCTION` discrepancy), so the
  offline measurement never silently understates the real agent (the .419 gap: offline ran
  no-LLM while the live agent has the LLM proposer). Asserting tests.

### PHASE C — HARDWARE CONTINUITY (exp4552)
Per-board reachability audit: KV260 (SSH ONLY, never host SD card), GateMate (USB detect),
PolarFire (SSH). Honest `blocked_<board>_<reason>` per board.

### PHASE D — SOTA-INGESTION (exp4553)
Ingest SOTA on LLM-as-world-model-inducer + verifier-guided refinement loops + intra-episode
goal-shift detection for ARC; map onto the A1 LLM-proposer headline; emit a SOTA->experiment note
with real arXiv IDs + flag the strongest method for .421.

### PHASE E — CAPSTONE (exp4554)
The per-level efficiency scorecard: did the LLM proposer raise `core_efficiency` above 2.0074 (a
CORE game reach a deeper level)? Did verifier discrimination beat chance (A2)? Did action
efficiency improve (A4)? Did `reproducible_total_levels` grow (A3/A6)? Skip `flagged_adversarial`
EXCEPT the annotated control-vs-treatment null-delta carve-out (the .419 B2 robustness).

---

## Dependency graph

```
exp4543 (transition)
   |
   +--> exp4544 (A1 LLM proposer) ----+--> exp4548 (A5 integration) --+
   +--> exp4545 (A2 discrimination)   |                               |
   +--> exp4546 (A3 level-up)         |                               |
   +--> exp4547 (A4 frame-change) ----+                               |
   +--> exp4544 (A1) --> exp4549 (A6 self-learning transfer) ---------+
   +--> exp4550 (B1 metric), exp4551 (B2 parity), exp4552 (C hw), exp4553 (D sota)
                                                                       |
                                                                       v
                                                              exp4554 (E capstone)
```

## Hardware requirements

- **A1 (exp4544) + A5 (exp4548):** live LLM inference — Qwen3.5-9B-MTP GGUF (cached:
  `unsloth/Qwen3.5-9B-MTP-GGUF`) on the **iGPU (Radeon 890M), NEVER the RTX 3090s** (per
  [[project_arc_live_generator]] + the 16GB Kaggle constraint); llama_cpp 0.3.29 (present).
- **A4 (exp4547):** the cached human-replay corpus (`data/arc_public_demo_human_replay_corpus`,
  staged license-clean by `.416 exp4495); CPU/iGPU CNN training.
- **C (exp4552):** KV260 (`ssh kria`), GateMate (DirtyJTAG USB), PolarFire (`ssh polarfire`).
- All other phases: CPU / offline arcade (`arc_solver_kit.offline_arcade()`), zero quota.

## Discipline compliance

- **Codex-Default-v2:** all experiments `agent_type: codex` / `gpt-5.5`; planner/retro stay Opus.
- **Verdict Terminal-Prefix:** every `honest_verdict` starts `complete:`/`success:`/`shipped:`.
- **Principle-Annotated Fields + Inference-Substrate + Pre-Launch Preconditions:** every task.
- **Failed-Experiment Rerun:** A1 carries `prior_failures` vs exp4533 (offline-DSL proposer ->
  LLM proposer); A4 carries `prior_failures` vs exp4490 (corpus-not-cached -> now staged).
- **Exclusion-Manifest Cross-Check:** A2/A3/A5/A6 + transition/infra/hw/sota/capstone carry
  `operator_override` for false-positive scope-matches against retired exps (cross-game VALUE
  transfer exp4318/4331/4342 etc. — A2/A6 are DISCRIMINATION / primitive-reuse, not value transfer).
- **Circularity / Oracle-Distinctness:** A2 declares `verifier_is_oracle: false`.
- **ARC Level-Up Attempt Guarantee:** A3 banks a NEW reproducible level (offline-reproduced).
- **ARC Solve Reproducibility + Reuse:** A3/A6 update `ops/arc_solve_registry.yaml`; only
  offline-reproduced levels count toward `reproducible_total_levels`.
