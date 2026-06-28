# Research Roadmap v456 — Final-Stretch Sprint (locked deliverable) + Post-6/30 Verifier-Moat Pivot to Turnkey

**Milestone:** 2026.06.456
**Planned by:** outer-loop (Claude Opus 4.8 planner), 2026-06-28
**Predecessor:** 2026.06.455 (FINAL PRE-DEADLINE SPRINT — locked deliverable)
**Sprint:** ARC-AGI-3 submission sprint, ACTIVE through 2026-06-30 (CLAUDE.md forcing function) — `.456` is a final-stretch milestone in the 6/28–6/30 window.

---

## 1. What `.455` proved (the locked deliverable, re-confirmed; the well is drying)

`.455` executed the locked deliverable for the 6/30 deadline. Capstone (exp4945):
`complete_capstone_v455_submission_ready_levels_69_heldout_0.04_package_ready_pivot_executable_7_1`.
The honest read:

| Lane | Result |
|---|---|
| A1 deepen lf52 L2→L3 | **NO-BANK** — `no_grounded_l3_delta` |
| A2 deepen sb26 L2→L3 | **NO-BANK** — `no_grounded_l3_delta` |
| → `reproducible_total_levels` | **stayed 69** (2nd consecutive flat milestone — the L2→L3 deepen well is drying on recently-rotated targets) |
| A4 held-out go/no-go | **CLEAN full-25 first-win = 0.04**, `flag_resolved=true`, CI [0,0] (the hidden-state wall; the `0.04==0.04` TAUTOLOGY is a *warn* = the honest null) — carried from exp4928 (anti-churn) |
| A3 self-play (ar25) | checkpoint refreshed (FR-11) BUT artifact **CRITICAL-flagged `DURATION_TOO_SHORT`** (declared `live_llm_inference` 60s floor; ran the offline reproduction gate in 0.0001s) → capstone correctly skipped it. **Substrate-declaration bug to fix.** |
| D distributional-energy-verifier | **ADVANCED to EXECUTABLE** (exp4940, `pivot_executable_on_7_1=true`); B1 `pivot_readiness_trustworthy=true` |
| B1 audit | banks_trustworthy=true (no banks to count); pivot_readiness_trustworthy=true |
| B2 submission package | **READY** (15.146 GB < 16 GB, frozen Qwen3.5-9B-MTP iGPU stack, operator-only) |
| B3 stamping | **`blocked_insufficient_v455_mtime_window`** again (asserts n_arms≥10 but runs before all arms land). **Gate-too-strict bug to relax.** |
| C KV260 | reachable (terminal, overlay `carnot_ising_v2_n64`) |

**Consequence:** the `.453` B1-trusted `WALL_IS_HIDDEN_STATE` closure stands. Two consecutive
flat deepen milestones + the held-out 0.04 confirm the two open scored axes (deepening + first-win)
are at their practical ceiling on the recently-rotated targets. **The deliverable LOCKS to the
current ~0.05 first-win agent + the publishable FoVer paper (`paper_ready=true`).** Per the operator
handoff: **do NOT queue representation #5; do NOT reopen any nulled/retired lever** (energy-as-ARC
[CONCLUDED null], macro/horizon-collapse, click-heatmap, trust-gate, MATM similarity-retrieval
[NULLED `.454`], TTT-on-code, local code inducers, decision-need targets, action-prefix latents,
coverage/exploration/selection/perception-from-grid).

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The 6/30 deadline deliverables stay green through every remaining sprint milestone.** Package
   ready + a clean 0.04 go/no-go — `.456` re-confirms both (the deadline lane) without churn (carry
   the settled number; re-build the package only to confirm it still loads).
2. **The mandatory sprint disciplines still apply through 6/30.** The Level-Up Attempt Guarantee
   (≥1 banking attempt) + self-play-every-milestone (FR-11) are not suspended because the L2→L3 well
   is drying — `.456` makes those passes on FRESH rotated targets, and tries a **deeper L3→L4 lane**
   (more plausible headroom than another exhausted L2→L3), recording honest no-bank if dry.
3. **The verifier-moat — Carnot's core post-ARC value-add — must be TURNKEY for 7/1.** `.453`/`.454`
   scaffolded and `.455` D made the distributional-energy-verifier EXECUTABLE; `.456` D finalizes it
   to **turnkey** — the post-6/30 first experiment is one command away (real data loaders for ≥1
   SC-not-saturated domain + the three columns dry-run + a pre-staged post-sprint first-experiment
   pointer), still WITHOUT running the real benchmark (majority-ARC governs through 6/30).

---

## 3. `.456` design — execute the locked deliverable, make the pivot turnkey, fix the two recurring infra bugs

`.456` is a **final-stretch sprint milestone (6/28–30).** Its honest headline is **submission
readiness for the locked deliverable + post-6/30 verifier-moat pivot made turnkey**, NOT a new
research fork. The genuine NEW value is concentrated in **D** (pivot → turnkey) and the two infra
fixes (A3 substrate declaration, B3 window gate); the ARC arms (A1/A2/A3/A4) satisfy the mandatory
sprint disciplines and attempt monotonic growth across two depth regimes on fresh targets.

```
                 reproducible_total_levels = 69 (no .455 bank)   held-out first-win = 0.04 (clean full-25)
                                     |
        +----------------------------+----------------------------------+
        |  DEEPEN (mandatory >=1 attempt; well drying)   POST-6/30 PIVOT|
        |  A1 fresh DEEPER L3->L4 (ar25/ft09/cn04)   D  distributional- |
        |  A2 fresh L2->L3 (cd82/vc33/bp35/re86)        energy-verifier |
        |     (Level-Up Guarantee floor)                -> TURNKEY      |
        +----------------------------+----------------------------------+
                                     |
              A3 self-play (FR-11 learned verifier, HONEST substrate)
              A4 held-out FINAL go/no-go (carry/confirm .455 clean full-25 0.04 — anti-churn)
                                     |
   B2 FINAL submission package harden + operator checklist (the deadline deliverable)
   B1 audit (banks real + pivot-readiness oracle-distinct)   B3 stamping (relaxed window gate)
   C KV260 SSH continuity
                                     |
                  E capstone — submission readiness scorecard + post-6/30 TURNKEY pivot handoff
```

### Allocation (mirrors the `.455` reserved-slot contract: 3 infra + 1 hardware + 1 SOTA/pivot)

| Phase | Task | Track | Agent | Why |
|---|---|---|---|---|
| 0 | exp4946 archive .455 → activate .456 | transition | codex | record the close-state (levels 69, 0.04, package ready, pivot executable 7/1) |
| A1 | exp4947 DEEPEN fresh **L3→L4** target (HEADLINE level-bank) | arc-north-star | codex | majority-ARC level-growth attempt in a deeper regime (more headroom than the drying L2→L3); honest no-bank if dry |
| A2 | exp4948 DEEPEN fresh **L2→L3** target (Level-Up Guarantee) | arc-north-star | codex | ≥1 banking-attempt floor; hedge across both depth regimes |
| A3 | exp4949 self-play (rotated banked game, **HONEST substrate**) | arc-north-star | codex | FR-11 continuous self-learning; fixes the `.455` DURATION_TOO_SHORT flag |
| A4 | exp4950 held-out FINAL go/no-go (carry/confirm 0.04) | arc-north-star | codex | the operator's 6/30 number; anti-churn carry/resume; deliverable-first max_turns≤50 |
| D | exp4951 distributional-energy-verifier → TURNKEY + SOTA refresh | sota-ingestion | codex | the post-6/30 verifier-moat headline, made one-command for 7/1 |
| B1 | exp4952 audit banks + pivot-readiness | infra | codex | banks real/non-duplicate; D oracle-distinct + honest (ordered AFTER A1/A2/D) |
| B2 | exp4953 FINAL submission package harden | infra | codex | the deadline deliverable; submits=false |
| B3 | exp4954 stamping backfill + **relaxed window** | infra | codex | retro top action; the relaxed gate stops the recurring false-block |
| C | exp4955 KV260 SSH continuity | hardware | codex | Hardware-Task Continuity (SSH-only) |
| E | exp4956 capstone .456 | capstone | codex | submission-readiness scorecard + TURNKEY post-6/30 pivot handoff |

**Continuous self-learning (research-program.md requirement):** A3 is the explicit Tier-3 / FR-11
experiment — the learned ARC verifier trains on the self-play traces and checkpoints, improving
across runs. D readies a *second* self-learning surface for the post-sprint headline (the
distributional energy verifier's learned quality-scorer ensemble).

### The two recurring-infra fixes `.456` lands

1. **A3 substrate declaration (`.455` exp4938 DURATION_TOO_SHORT critical flag).** Self-play's
   reproduction gate + checkpoint training run OFFLINE (`offline_reproduction_gate_no_quota`, no LLM,
   ~0.0001s), but `.455` declared `inference_substrate: live_llm_inference` (60s floor) → critical
   flag → capstone-skipped. `.456` A3 declares the **honest** substrate
   (`verifier_ensemble_against_cached_candidates`, 1s floor) when no LLM runs, and only
   `live_llm_inference` if the generator actually runs ≥60s. The artifact stops false-flagging and
   the capstone counts it.
2. **B3 window gate (`.455` exp4943 `blocked_insufficient_v455_mtime_window`).** The fallback asserts
   `n_arms≥10` but runs before all 11 arms land. `.456` B3 relaxes to the arms PRESENT at run time
   (`n_arms≥7`, `wall_minutes>0`) anchored on the activation commit, so it emits a NON-zero partial
   window instead of blocking. (The full operator wire of the conductor remains the operator's job;
   the autonomous loop cannot edit `research_conductor.py`.)

### Ordering (keeps the `.455` fix)

**D (exp4951) is ordered before B1 (exp4952)** so the audit can read the pivot-readiness artifact;
the capstone (last) reads B1's trust flags. A1/A2 also precede B1 so it can audit any banks.

### Disciplines honored

- **ARC Live-Path Reachability:** every ARC task improves a LIVE-reachable mechanism
  (`arc_loop_solve` / `GameAdapter` / `E3AgentPolicy`); `arc_orphan_solver_lint` must pass.
  `solve_provenance` declared on every solve task (`live_agent_self_discovery` for A1/A2/A3;
  `development_proxy` for A4).
- **ARC Solve Reproducibility:** banks count only through `arc_solver_kit.reproduce`;
  registry-precheck before any solve (no re-solving an already-reproduced level → duplicate is
  CRITICAL). Honest no-bank rotation dead-ends are the EXPECTED outcome on a drying well — never
  fabricate a bank.
- **ARC Incremental-Progress Scoping:** +1 level per game; rotated targets; no "FULL solve."
- **ARC Level-Up Attempt Guarantee:** A1 + A2 are two banking attempts (≥1 floor; lint passes).
- **Circularity / Oracle-Distinctness:** D's distributional-energy-verifier design target is
  oracle-distinct (`verifier_is_oracle: false`) and makes NO moat-proven claim (it is
  readiness/design only); B1 audits this.
- **Operator-Only External Publication:** B2 prepares the package + checklist; `submits=false`.
  D states the post-6/30 validation gate but does NOT execute the experiment.
- **Pre-Launch Preconditions:** every compute-bound task gates on arcade/generator/SSH; GPU fix
  (2026-06-27) — offline induction accepts the conductor's GPU-0 CUDA generator, NOT iGPU-pinned;
  the LIVE submission stack (B2) stays frozen Qwen3.5-9B-MTP on the iGPU.
- **Failed-Experiment Rerun / Exclusion-Manifest:** no nulled/retired lever is re-proposed; routine
  continuations carry `operator_override` (the standing 2026-06-19 ARC-sprint directive).
- **SOTA-Ingestion Cycle:** D is the reserved SOTA-ingestion slot (the headline track is the
  bleeding-edge verifier-moat); it ingests via low-concurrency channels (NOT `/deep-research`)
  and cites real arXiv IDs.
- **Codex-Default v2:** all experiments `agent_type: codex`/`gpt-5.5`; planner + retro stay Opus.
- **Verdict Terminal-Prefix + Principle-Annotated fields:** every task.

### What `.456` does NOT do (closed/retired — do not re-propose)

Representation #5 / any new world-model fork; energy-as-ARC-lever (CONCLUDED negative — no live ARC
value); macro/horizon-collapse, click-heatmap generator, trust-gate flip, MATM similarity-retrieval
(NULLED `.454`) (all empirically retired); TTT-on-code engine; stronger local code inducers;
decision-need targets; action-prefix latents; coverage/exploration/selection/perception-from-grid.
First-win-wall chasing is closed.

---

## 4. Dependency graph

```
exp4946 (transition)
   ├─> exp4947 A1 deepen L3->L4 ─┐
   ├─> exp4948 A2 deepen L2->L3 ─┤
   ├─> exp4949 A3 self-play       ├─> exp4952 B1 audit ─┐
   ├─> exp4950 A4 held-out        │   (after A1/A2/D)    │
   ├─> exp4951 D  pivot turnkey ──┘                      │
   ├─> exp4953 B2 package ─────────────────────────────────┤
   ├─> exp4954 B3 stamping ────────────────────────────────┤
   └─> exp4955 C  KV260 ───────────────────────────────────┤
                                                            v
                                            exp4956 E capstone .456
```

A1/A2/D feed B1 (the audit gates trust in the banks + the pivot-readiness). All arms feed the
capstone, which skips any `flagged_adversarial` upstream per the fabrication gate.

---

## 5. Hardware requirements

- **Offline ARC induction (A1/A2/A3/A4):** conductor's dedicated **GPU-0 CUDA llama-server**
  (`CARNOT_ARC_GENERATOR_CUDA_GPU=0`, card ≥13 GB free) OR the iGPU HIP server. Per the 2026-06-27
  GPU-allocation directive, offline induction is NOT iGPU-pinned. Do NOT block merely because
  `CUDA_VISIBLE_DEVICES` is set. Most deepens are offline search + reproduction gate (no LLM) —
  declare the honest substrate.
- **LIVE submission stack (B2):** FROZEN — Qwen3.5-9B-MTP on the **iGPU** (Kaggle ~16 GB parity);
  never the 3090s for the live stack.
- **D (pivot turnkey):** aggregation + a small dry-run only (no full benchmark); reads the exp4922
  scaffold + exp4940 executable spec + the structured-domain slice.
- **KV260 (C):** SSH-reachability only (`ssh kria 'true'`); NEVER a host SD-card / block-device
  precondition.

---

## 6. SOTA anchoring the post-6/30 verifier-moat pivot (re-confirmed; filed in research-references.md)

- **arXiv:2605.18871 — Distributional Energy-Based Models for Uncertainty-Aware Structured LLM
  Reasoning** (May 2026). Decomposed energy = a learned quality scorer (heterogeneous LoRA ensemble
  on one frozen encoder, ~3% trainable) + deterministic analytical constraint penalties; the ensemble
  MEAN ranks, the STDDEV abstains (two-pass regen/abstain). **Matches SC on GSM8K (97.0%) and EXCEEDS
  SC on MuSR (>64.4%)** — the oracle-distinct, SC-not-saturated win Carnot's moat needs, landing the
  exact Carnot thesis. **This is the post-6/30 pivot experiment.**
- **arXiv:2504.16828 — THINKPRM.** A generative reasoning verifier that beats SC at equal compute
  using ~1% of process labels — corroborates the verifier-moat direction.
- **arXiv:2502.01989 — VFScale.** An intrinsic energy-based verifier whose dense reward is
  near-oracle — corroborates energy-as-verifier as a near-perfect dense reward.

---

## 7. Post-6/30 handoff (the pivot is now TURNKEY)

The sprint retires 2026-06-30. The locked deliverable (~0.05 agent + FoVer paper) is the ARC outcome.
The **post-sprint headline is the verifier-moat**: execute the distributional-energy-verifier
experiment (arXiv:2605.18871) on a non-saturated structured-reasoning domain (MuSR / TravelPlanner)
where self-consistency is NOT near-ceiling — validation gate: "distributional energy verifier beats
self-consistency, CI95 excluding zero, no model-identity shortcut, oracle-distinct
(`verifier_is_oracle: false`)." `.456` D makes it **turnkey** (real loaders + three-column dry-run +
a pre-staged post-sprint first-experiment pointer; `pivot_executable_on_7_1=true`, gated on B1
`pivot_readiness_trustworthy`); the capstone states the handoff so the loop pivots cleanly the instant
the sprint retires. This is OFF-ARC structured reasoning, NOT energy-as-ARC (that program concluded
null).
