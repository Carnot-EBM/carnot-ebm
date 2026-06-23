# Research Roadmap — Milestone 2026.06.426

**Planned by:** outer-loop (Claude Opus 4.8 planner, 2026-06-23).
**Milestone doc for:** `research-roadmap-next.yaml` (activate as the .426 roadmap).
**One-line thesis:** The ARC representation is SOLVED and the dense-gradient
positive is PROVEN; the single binding constraint is now the **offline→live
bridge** — a 0.725-LOO-AUROC discriminator / a 7.6x-offline SpatialValueNet
Q-head both *regress* the live depth-first explorer. .426 **disambiguates why**
(compute-cost vs distribution-shift vs calibration) and **graduates** the
position-preserving value head from prototype to the LIVE path, replacing the
linear verifier that "actively misled," measured on the SCORED agent.

---

## 0. ARC-AGI-3 submission sprint (active through 2026-06-30)

Per CLAUDE.md **ARC-AGI-3 Submission Sprint Forcing Function**, this milestone
sits inside the operator submission sprint (7 days to the 2026-06-30 deadline).
The contract is honored:

- **Majority ARC**, monotonic `reproducible_total_levels` (currently **55**).
- **≥1 level-up attempt that BANKS** a new reproducible level (Phase A3 — the
  ARC Level-Up Attempt Guarantee; `scripts/arc_levelup_guarantee_lint.py`).
- **2 reserved infra slots** (B1 co-headline bridge metric, B2 adversarial_verify
  hardening); **1 per-attached-board hardware slot** (C); **1 SOTA-ingestion
  slot** (D).
- **All experiments `agent_type: codex` / `gpt-5.5`**; planner + retro stay
  Claude Opus 4.8 (operator quality choice).
- **Live generator FROZEN:** Qwen3.5-9B-MTP on the iGPU (NEVER the 3090s)
  ([[project_arc_live_generator]]). .426 builds on this stack; no model
  re-litigation.
- **Submission stays operator-only** (External Publication). Tasks PREPARE +
  offline-validate; they never submit.

---

## 1. What the previous milestones proved (.421–.425), and the decisive re-diagnosis

The .421–.424 milestones exhausted the **generation** side (5 consecutive nulls:
re-rank, verifier-expansion, router, generation-wiring, energy-prior; those
scopes are RETIRED). .425 then PIVOTED to the **verifier / world-model** side and
also landed flat — but it produced the decisive diagnostic that steers .426:

| .425 phase | Result | Reading |
|---|---|---|
| **A1** world-model TRUST ENERGY (change-weighted + held-out gate) | **QUARANTINED** — verdict claimed `trust_pass_rate 0→1.0, first_win 0→1.0` but `flagged_adversarial=True` (DURATION_TOO_SHORT 0.44s); the capstone correctly excluded it | A trust gate going 0/6→6/6 in 0.44s is a degenerate trivially-passing gate, not a crack. Excluded from the headline. |
| **A2** live integration (0.674 LINEAR verifier as tie-breaker + router + forward nav into the SCORED agent) | **HONEST NULL** — `first_win_delta=0.0`, `actions_delta=0.0`, `solve_rate` stuck `0.04` bare==integrated | Wiring a verifier naively into the live agent does NOT help. The linear discriminative head earns no place. |
| **A3** self-play deepen (dc22 L1→L2) | **NO BANK** — `reached_level=1`, `reproduced=False` | The level-up guarantee was attempted but did not bank. |
| **A4** | package refreshed, `live_submittable=55`, `ready_for_operator_submit=True` (beats the standing 33-level scorecard) | The score asset is intact. |
| **Capstone** | `complete: pivot_characterized_capability_grew_55_to_55` (delta=0) | The pivot did NOT crack the 0.08 wall. |

**The decisive re-diagnosis (outer-loop, 2026-06-23, worktree-isolated
measurement — commit `4b7782d41`,
`docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md`):**

- The architecture analysis that drove .425 claimed the binding constraint was
  the **representation** ("features at chance, LOO-AUROC 0.503"). **A fresh run
  of the dedicated harness refutes that.** `cross_game_features_v3` (the LIVE
  features, `arc_value_learner.py:394`) gets **LOO-AUROC 0.725** (CI [0.649,
  0.806]); the frame-Δ/relational features the analysis "recommended adding" are
  **already implemented** and already lift 0.515→0.725. **"Add more features" is
  not the lever.**
- **The real binding constraint is the OFFLINE→LIVE BRIDGE.** A 0.725-AUROC
  discriminator *regressed* the live search (`value_weight` reverted 5.0→0.0,
  `arc_competition_agent.py:60` — "slower than bare BFS, solved fewer games, the
  25-game sim timed out"). The same pattern is in
  `results/arc_offline_to_live_bridge_v2.json`: the value head "unlocks cn04
  offline best-first but does NOT transfer to the live depth-first explorer."
- **Three candidate causes, to disambiguate (this is the explicit next step):**
  1. **COMPUTE-COST** (the live comment points here): computing the value per
     frontier node slows the bounded-time search → fewer nodes → fewer solves.
     Fix = cheap/cached/incremental features, or apply the value head only at
     decision points, not every node.
  2. **DISTRIBUTION-SHIFT**: the value is trained on *winning-path* states but
     the live frontier is *off-path* states it never saw → ~chance where it
     matters. Fix = train on the search distribution (DAgger-style), or use it
     as bounded pruning not an A* value.
  3. **CALIBRATION**: a 0.725 *ranking* is not a usable A* cost; a wrong rank on
     the decisive node misroutes a depth-first search. Fix = isotonic/Platt
     calibration to a cost.

**The ONE genuine positive from the 2026-06-22 sprint (VERIFIER-AS-Q-HEAD).** A
learned **position-preserving** `SpatialValueNet` (4×4 pool, not the global-pool
version that "discards spatial position") supplies the dense per-step gradient
that goal-induction never provided. `experiment_value_q_head_v4`: tuned
`heuristic_weight~10` routes ls20-L1 in **233 vs 1777 expansions = 7.6x**
(monotonic in weight) on clean-navigation games. **But it is stuck in
`scripts/experiments/`** and `scripts/arc_loop_solve.py` STILL warm-starts the
LIVE search with the LINEAR `LearnedVerifier` we have now SHOWN "cannot route the
live search." The flagged shippable upgrade (`.425+`): wire the SpatialValueNet
+ tuned weight into the live path. **This is the .426 build.**

---

## 2. The three biggest gaps between current state and the PRD/north-star vision

1. **GAP-OFFLINE-LIVE-BRIDGE (the headline gap).** A decent offline
   discriminator (0.725) and a strong offline router (7.6x) make the LIVE search
   *worse*. Until we know WHY (compute/shift/calibration) and FIX it, every
   verifier improvement dies at the bridge — exactly what .425 A2 showed. This is
   Carnot's value-add (the verifier) failing to reach the deliverable (the live
   agent). **Closing it is the north star's "efficiency" axis made real.**
2. **GAP-LIVE-INTEGRATION (`ops/verifier_gaps.md:2420`).** The SCORED
   `E3AgentPolicy` ships `value_weight=0.0` (value head inert), `target_levels=1`,
   and a value head with ~0 OOD transfer — "the SUBMITTED agent runs a weaker
   generic path than the repo's own research." The 55 reproduced levels are a
   leaderboard mirage until the scored agent actually uses Carnot's verifier.
3. **Deepening / cross-level gradient.** A per-level value has no signal for the
   next level's different goal (v5 null; v6 milestone-distance relabel was
   inconclusive on a no-headroom seed). Banking NEW levels (vs re-banking known
   ones) is the monotonic-progress test the sprint demands.

---

## 3. The .426 program (architecture)

```
              OFFLINE (proven)                 ||      THE BRIDGE (.426)      ||     LIVE (the deliverable)
  cross_game_features_v3  LOO-AUROC 0.725      ||                            ||  E3AgentPolicy (SCORED)
  SpatialValueNet (4x4)   7.6x routing offline ||   A1: WHICH cause binds?   ||  arc_loop_solve (dev twin)
                                               ||     compute / shift /      ||
   [ today: regresses the live search ]  ------>>     calibration  --------->>  A2: graduated value head
   [ linear LearnedVerifier "actively misled"] ||   A2: apply the fix        ||      first-win / actions UP
                                               ||     + GRADUATE the Q-head   ||      vs linear & bare (controls)
                                               ||     (live-path-reachable)   ||  A3: bank +1 level (55->56+)
```

**Phase A — ARC north star (operator-mandatory; the majority of the milestone)**

- **A1 (exp4616) — HEADLINE: DISAMBIGUATE the offline→live bridge.** A controlled
  three-arm experiment that isolates compute-cost vs distribution-shift vs
  calibration as the cause the 0.725 / 7.6x-offline value head regresses the live
  depth-first explorer. Each arm has a matched control (bare BFS + the value head
  as-is). Deliverable: the BINDING sub-cause + the indicated fix, with the
  diagnosis traceable (per-arm node-count, off-path AUROC, rank→cost calibration
  error). `verifier_is_oracle: false`. This is the diagnostic `bridge_v1/v2`
  plumbed-but-never-isolated.
- **A2 (exp4617) — HIGHEST MANDATORY: GRADUATE the SpatialValueNet Q-head to the
  LIVE path.** Move the position-preserving SpatialValueNet (4×4 pool) out of
  `scripts/experiments/` into a `python/carnot/agentic/` module that is **in the
  live import closure** (`arc_loop_solve.py` warm-start; fed to `E3AgentPolicy`)
  — NOT an orphaned solver (CLAUDE.md ARC Live-Path Reachability Discipline,
  `arc_orphan_solver_lint.py`). Replace the LINEAR `LearnedVerifier` warm-start;
  apply the **A1-diagnosed fix** (decision-point-only eval for compute-cost /
  DAgger search-distribution retraining for distribution-shift / isotonic
  calibration for ranking→cost). Measure LIVE first-win-rate + actions-to-first-
  levelup vs (i) the linear-verifier baseline and (ii) bare BFS (matched
  controls; bootstrap CI). Keep `test_arc_submitted_agent_parity.py` green.
  `verifier_is_oracle: false`.
- **A3 (exp4618) — LEVEL-UP GUARANTEE + SELF-PLAY (the BANK).** Run the standing
  self-play loop to bank +1 NEW reproducible level (55→56+) on a rotated
  clean-navigation game where the graduated Q-head routes faster (prefer
  **sk48 L1→L2**; alternatives wa30/ls20/lf52/re86/bp35 L1→L2). Skip the recorded
  dead-ends (ka59 hidden-register, dc22-L2 just-failed, cd82-L3/sp80-L3/su15-L3).
  Train+checkpoint the learned verifier on the run's pos/neg traces. INDEPENDENT
  of A1/A2 so the guarantee holds even if they null. Gate: `offline_reproduced`.
- **A4 (exp4619) — SCORE: keep the package operator-resubmit-ready.** Fold A3's
  bank (+ any A2/A3 new solves) into the refreshed package; re-validate every
  claimed level offline-reproduces; live-submittable count stays STRICTLY > 33.
  Submission operator-only.
- **A5 (exp4620) — SELF-LEARNING + REUSE.** Persist the milestone's winning
  primitive (the graduated value-head bridge operator, OR the calibration helper)
  into `arc_solver_kit` + `arc_solve_registry` (Solver-Reuse Discipline) and
  measure CROSS-GAME TRANSFER to 2–3 untuned games. `verifier_is_oracle: false`.
- **A6 (exp4621) — INTEGRATION + HEADLINE METRIC.** Consolidate whatever raised a
  real metric (A2 graduated value head; A3 bank) into `SUBMITTED_AGENT_CONFIG` +
  the refreshed package; re-measure end-to-end. If NONE raised a clean
  control-passed metric, keep the bare config + honest null. Parity test green.

**Phase B — reserved infrastructure (2 slots)**

- **B1 (exp4622) — co-headline BRIDGE metric.** Canonical
  `offline_to_live_transfer_ratio`: the LIVE first-win/efficiency lift
  attributable to the value head, reported **side-by-side with the offline
  LOO-AUROC** so the offline→live gap is explicit and tracked every milestone
  (the direct measure of whether .426 crossed the bridge — the analogue of
  .425's `world_model_trust_pass_rate`). Mechanical aggregation; asserting tests.
- **B2 (exp4623) — adversarial_verify hardening (two reader-side guards).**
  (1) an **offline-vs-live overclaim guard**: an ARC value/verifier artifact
  claiming a LIVE search win MUST report a measured LIVE metric, not substitute an
  offline AUROC (the exact 0.503-vs-0.725 / offline-vs-live confusion the
  outer-loop just corrected). (2) a **calibrated cheap-learned-value substrate
  floor** so a legitimate fast CNN/linear value-head scoring run over cached
  candidates is not DURATION_TOO_SHORT false-flagged (the .425 A1 0.44s
  regression fixture), while a no-methodology fast run STILL fires. Asserting
  tests; edits `scripts/adversarial_verify.py` only (never the conductor).

**Phase C — hardware continuity (1 per attached board)**

- **C (exp4624)** — per-board reachability audit: KV260 (SSH reachability ONLY,
  never host SD card), GateMate (`openFPGALoader --detect`), PolarFire (SSH).
  Honest `blocked_<board>_<reason>` per board. Lightweight during the sprint.

**Phase D — SOTA ingestion (1 slot)**

- **D (exp4625)** — ingest the **offline→live transfer / distribution-shift /
  calibration** SOTA mapped onto A1/A2 and fed forward to .427: DAgger / dataset
  aggregation (arXiv:1011.0686), isotonic/Platt calibration, learned-heuristic
  search (DeepCubeA/Q* arXiv:2102.04518, SLOPE arXiv:2406.04935), goal-conditioned
  value (GoFAR arXiv:2206.03023). Real arXiv IDs only; reliable channel
  (sweep helpers + low-concurrency WebSearch/WebFetch); `/deep-research` banned.

**Phase E — capstone**

- **E (exp4626)** — the .426 scorecard: did we CROSS the bridge (A1 named the
  binding cause + A2 graduated the value head and raised LIVE first-win/efficiency
  on the SCORED agent vs the linear baseline)? Did A3 bank +1 (55→56+)? Report all
  co-headline metrics (`offline_to_live_transfer_ratio`,
  `reproducible_total_levels`, live-submittable, first-win-rate, action
  efficiency). Skip `flagged_adversarial`; honor the positive-control +
  FALSE_NEGATIVE_RISK + .425-B2 TAUTOLOGY carve-out guards.
  `verifier_is_oracle: false` on every value claim.

---

## 4. Dependency graph

```
exp4615 PHASE 0 (archive .425 -> activate .426)
   |
exp4616 A1  DISAMBIGUATE bridge cause (compute / shift / calibration)
   |            \
   v             v  (A2 reads A1's diagnosed cause)
exp4617 A2  GRADUATE SpatialValueNet to live path + apply fix + measure on SCORED agent
   |
exp4618 A3  self-play BANK +1 level (independent; guarantee holds even if A1/A2 null)
   |
exp4619 A4  refresh package (folds A3 bank + A2 solves)         exp4622 B1  bridge co-headline metric
   |                                                            exp4623 B2  adversarial_verify hardening
exp4620 A5  persist winning primitive + cross-game transfer     exp4624 C   hardware reachability
   |                                                            exp4625 D   SOTA ingestion -> .427
exp4621 A6  integrate winners into SUBMITTED_AGENT_CONFIG
   |
exp4626 E   CAPSTONE scorecard (aggregates A1-A6 + B1/B2; skips flagged_adversarial)
```

A1→A2 is informational (A2 reads A1's diagnosed cause; not hard-gated — A2
graduates the value head regardless, defaulting to the compute-cost fix the live
comment points at if A1 is inconclusive). A3 is independent of A1/A2 so the
level-up guarantee holds unconditionally. A4/A6 fold the upstream winners. E
aggregates everything via `summarize_artifact.py`, excluding flagged artifacts.

---

## 5. Hardware requirements

- **iGPU (Radeon 890M) ONLY** for any optional live LLM-induction arm
  (Qwen3.5-9B-MTP, the frozen generator). **NEVER the dual RTX 3090s** during the
  sprint.
- A1/A2/A3/A5/A6 are `verifier_ensemble_against_cached_candidates` /
  offline-arcade CPU work (the SpatialValueNet is CPU-trained, mirror-ready per
  decentralization Rule 3). B1/B2/D/E are `aggregation_from_upstream_artifacts`.
- C is `hardware_smoke` (SSH/USB reachability on the three attached boards).

---

## 6. Discipline compliance checklist

- **ARC sprint:** majority-ARC (A1–A6 of 12 tasks); monotonic
  `reproducible_total_levels` target (A3 55→56+); 2 infra (B1/B2) + 1 hardware
  (C) + 1 SOTA-ingestion (D) reserved; codex experiments, Opus planner/retro;
  frozen generator. ✓
- **ARC Level-Up Attempt Guarantee:** A3 is a BANK attempt (gate
  `offline_reproduced`); `arc_levelup_guarantee_lint.py` passes. ✓
- **ARC Live-Path Reachability:** A2 graduates the value head INTO the live
  import closure (not an orphaned `scripts/experiments/` solver);
  `solve_provenance` declared on every solve-claiming task. ✓
- **Circularity / Oracle-Distinctness:** `verifier_is_oracle: false` on every
  value claim (the SpatialValueNet is a learned value, oracle-DISTINCT from the
  executable win-check). ✓
- **Failed-Experiment Rerun:** A1/A2/A3 carry `prior_failures:` blocks (bridge
  v1/v2; exp4605 .425 A2 + q-head v4/v5; exp4606 .425 A3) with the forward
  difference + `retire_if_same_verdict: true`; routine continuations carry
  `operator_override:`. ✓
- **Pre-Launch Preconditions:** every task opens with a PRECONDITIONS step. ✓
- **Verdict Terminal-Prefix + Principle-Annotated Fields + Inference-Substrate
  Declaration:** all honored in the YAML. ✓
- **Public Documentation Discipline / Operator-Only Publication:** no autonomous
  edits to the landing page; submission stays operator-only. ✓
