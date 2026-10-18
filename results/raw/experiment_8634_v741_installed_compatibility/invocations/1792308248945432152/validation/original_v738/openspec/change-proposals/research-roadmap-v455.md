# Research Roadmap v455 — Final Pre-Deadline Sprint + Post-6/30 Verifier-Moat Pivot Readiness

**Milestone:** 2026.06.455
**Planned by:** outer-loop (Claude Opus 4.8 planner), 2026-06-28
**Predecessor:** 2026.06.454 (POST-ARC-CLOSURE SUBMISSION MAXIMIZATION)
**Sprint:** ARC-AGI-3 submission sprint, ACTIVE through 2026-06-30 (CLAUDE.md forcing function) — `.455` is the FINAL or second-to-last milestone before the deadline.

---

## 1. What `.454` proved (the locked deliverable, confirmed)

`.454` executed the locked deliverable for the 6/30 deadline. Capstone (exp4934):
`complete_capstone_v454_submission_maximized_levels_69_heldout_0.04_package_ready_efficiency_null`.
The honest read:

| Lane | Result |
|---|---|
| A1 deepen sp80 L2→L3 | **NO-BANK** — no grounded next-level delta |
| A2 deepen su15 L2→L3 | **NO-BANK** — no grounded next-level delta |
| → `reproducible_total_levels` | **stayed 69** (the deepen well is drying on recently-rotated targets) |
| A4 held-out go/no-go | **CLEAN full-25 first-win = 0.04**, `flag_resolved=true` (the hidden-state wall confirmed; the TAUTOLOGY was a *warn* only because `0.04==0.04` baseline = the honest null) |
| A3 self-play | bp35 checkpoint refreshed (FR-11) |
| D MATM similarity-keyed retrieval | **NULLED + RETIRED** (`complete_matm_similarity_retrieval_no_efficiency_gain_retired`) |
| B2 submission package | **READY** (15.146 GB < 16 GB, frozen Qwen3.5-9B-MTP iGPU stack, operator-only) |
| B1 audit | `blocked_experiment_4933_..._missing` — **ordering bug**: the audit (exp4929) ran *before* D (exp4933) existed |
| B3 / C | stamping applied; KV260 reachable (terminal) |

**Consequence:** the `.453` B1-trusted `WALL_IS_HIDDEN_STATE` closure stands. Both `.454`
deepens no-banked and the MATM efficiency lever nulled, so the two open scored axes (deepening
+ action efficiency) are at or near their practical ceiling on the recently-rotated targets.
**The deliverable LOCKS to the current ~0.05 first-win agent + the publishable FoVer paper
(`paper_ready=true`).** Per the operator handoff: **do NOT queue representation #5; do NOT
reopen any nulled/retired lever** (energy-as-ARC, macro/horizon-collapse, click-heatmap,
trust-gate, MATM similarity-retrieval, TTA-on-code, local code inducers, decision-need targets,
action-prefix latents, coverage/exploration/selection/perception-from-grid).

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The 6/30 deadline deliverables need a FINAL confirmation.** The package is ready and the
   go/no-go number is clean (0.04), but `.455` is the last window before the deadline — a FINAL
   package harden + a FINAL (anti-churn) go/no-go confirmation are the deadline lane.
2. **The mandatory sprint disciplines still apply through 6/30.** The Level-Up Attempt Guarantee
   (≥1 banking attempt) + self-play-every-milestone (FR-11) are not suspended just because the
   well is drying — `.455` makes those passes on FRESH rotated targets, recording honest no-bank
   rotation dead-ends if dry (no fabrication).
3. **The verifier-moat — Carnot's core post-ARC value-add — has no EXECUTABLE post-sprint
   experiment yet.** ARC closure makes the FoVer/verifier-moat program the next headline. `.453`
   D scaffolded the distributional-energy-verifier harness; `.454` D kept it a scaffold. `.455` D
   ADVANCES it to EXECUTABLE so the real experiment runs the instant the sprint retires (7/1).
   New SOTA (this planning round) sharpens it decisively: **arXiv:2605.18871 BEATS self-consistency
   on MuSR** — the oracle-distinct, SC-not-saturated win the moat needs — using the EXACT Carnot
   thesis (learned quality scorer + deterministic constraint penalties + uncertainty/abstention).

---

## 3. `.455` design — execute the locked deliverable, ready the pivot, do not chase the closed wall

`.455` is the **final (or second-to-last) milestone before 6/30.** Its honest headline is
**submission readiness for the locked deliverable + post-6/30 verifier-moat pivot readiness**,
NOT a new research fork. The genuine NEW value is concentrated in **D** (pivot → executable) and
**B2** (final package); the ARC arms (A1/A2/A3/A4) satisfy the mandatory sprint disciplines and
attempt monotonic growth on fresh targets.

```
                 reproducible_total_levels = 69 (no .454 bank)   held-out first-win = 0.04 (clean full-25)
                                     |
        +----------------------------+----------------------------------+
        |  DEEPEN (mandatory ≥1 attempt; well drying)   POST-6/30 PIVOT |
        |  A1 fresh deepen (cd82/lf52/...)   D  distributional-energy-  |
        |  A2 fresh deepen (Level-Up Guar.)     verifier → EXECUTABLE   |
        +----------------------------+----------------------------------+
                                     |
              A3 self-play (FR-11 learned verifier, every milestone)
              A4 held-out FINAL go/no-go (carry/confirm .454 clean full-25 0.04 — anti-churn)
                                     |
   B2 FINAL submission package harden + operator checklist (the deadline deliverable)
   B1 audit (banks real + pivot-readiness oracle-distinct)   B3 stamping backfill/readiness
   C KV260 SSH continuity
                                     |
                  E capstone — submission readiness scorecard + post-6/30 EXECUTABLE pivot handoff
```

### Allocation (mirrors the `.454` reserved-slot contract: 3 infra + 1 hardware + 1 SOTA/pivot)

| Phase | Task | Track | Agent | Why |
|---|---|---|---|---|
| 0 | exp4935 archive .454 → activate .455 | transition | codex | record the close-state (levels 69, 0.04, package ready, efficiency null) |
| A1 | exp4936 DEEPEN fresh grounded target (HEADLINE level-bank) | arc-north-star | codex | majority-ARC level-growth attempt (no rep #5); honest no-bank if dry |
| A2 | exp4937 DEEPEN different fresh target (Level-Up Guarantee) | arc-north-star | codex | ≥1 banking attempt floor; hedge the drying well |
| A3 | exp4938 self-play (rotated banked game) | arc-north-star | codex | FR-11 continuous self-learning, every milestone |
| A4 | exp4939 held-out FINAL go/no-go (carry/confirm 0.04) | arc-north-star | codex | the operator's 6/30 number; anti-churn carry/resume |
| D | exp4940 distributional-energy-verifier → EXECUTABLE + SOTA-ingest | sota-ingestion | codex | the post-6/30 verifier-moat headline, readied to run 7/1 |
| B1 | exp4941 audit banks + pivot-readiness | infra | codex | banks real/non-duplicate; D oracle-distinct + honest (ordered AFTER D — fixes the .454 bug) |
| B2 | exp4942 FINAL submission package harden | infra | codex | the deadline deliverable; submits=false |
| B3 | exp4943 stamping backfill + wiring readiness | infra | codex | retro top action; closes duration_s=None for .455 arms |
| C | exp4944 KV260 SSH continuity | hardware | codex | Hardware-Task Continuity (SSH-only) |
| E | exp4945 capstone .455 | capstone | codex | submission-readiness scorecard + EXECUTABLE post-6/30 pivot handoff |

**Continuous self-learning (research-program.md requirement):** A3 is the explicit Tier-3 / FR-11
experiment — the learned ARC verifier trains on the self-play traces and checkpoints, improving
across runs. D readies a *second* self-learning surface for the post-sprint headline (the
distributional energy verifier's learned quality-scorer ensemble).

### Ordering fix (the `.454` bug)

In `.454`, B1 (exp4929, position 6) ran *before* D (exp4933, position 11), so the audit could not
read D's artifact and emitted `blocked_..._missing`. `.455` orders **D (exp4940) before B1
(exp4941)** so the audit reads the pivot-readiness artifact; the capstone (last) reads B1's trust
flags.

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
- **Failed-Experiment Rerun:** the `.454` MATM efficiency lever NULLED+RETIRED is NOT re-proposed;
  D replaces it with the post-6/30 pivot readiness.
- **SOTA-Ingestion Cycle:** D is the reserved SOTA-ingestion slot (the headline track is the
  bleeding-edge verifier-moat); it ingests via low-concurrency channels (NOT `/deep-research`)
  and cites real arXiv IDs.
- **Codex-Default v2:** all experiments `agent_type: codex`/`gpt-5.5`; planner + retro stay Opus.
- **Verdict Terminal-Prefix + Principle-Annotated fields:** every task.

### What `.455` does NOT do (closed/retired — do not re-propose)

Representation #5 / any new world-model fork; energy-as-ARC-lever (CONCLUDED negative);
macro/horizon-collapse, click-heatmap generator, trust-gate flip, **MATM similarity-retrieval
(NULLED `.454`)** (all empirically retired); TTT-on-code engine; stronger local code inducers;
decision-need targets; action-prefix latents; coverage/exploration/selection/perception-from-grid.
First-win-wall chasing is closed.

---

## 4. Dependency graph

```
exp4935 (transition)
   ├─> exp4936 A1 deepen ─────┐
   ├─> exp4937 A2 deepen ─────┤
   ├─> exp4938 A3 self-play   ├─> exp4941 B1 audit ─┐
   ├─> exp4939 A4 held-out    │   (after A1/A2/D)    │
   ├─> exp4940 D  pivot exec ─┘                      │
   ├─> exp4942 B2 package ────────────────────────────┤
   ├─> exp4943 B3 stamping ───────────────────────────┤
   └─> exp4944 C  KV260 ──────────────────────────────┤
                                                       v
                                            exp4945 E capstone .455
```

A1/A2/D feed B1 (the audit gates trust in the banks + the pivot-readiness). All arms feed the
capstone, which skips any `flagged_adversarial` upstream per the fabrication gate.

---

## 5. Hardware requirements

- **Offline ARC induction (A1/A2/A3/A4):** conductor's dedicated **GPU-0 CUDA llama-server**
  (`CARNOT_ARC_GENERATOR_CUDA_GPU=0`, card ≥13 GB free) OR the iGPU HIP server. Per the 2026-06-27
  GPU-allocation directive, offline induction is NOT iGPU-pinned. Do NOT block merely because
  `CUDA_VISIBLE_DEVICES` is set.
- **LIVE submission stack (B2):** FROZEN — Qwen3.5-9B-MTP on the **iGPU** (Kaggle ~16 GB parity);
  never the 3090s for the live stack.
- **D (pivot readiness):** aggregation + a small dry-run only (no full benchmark); reads the
  exp4922 scaffold + the structured-domain slice.
- **KV260 (C):** SSH-reachability only (`ssh kria 'true'`); NEVER a host SD-card / block-device
  precondition.

---

## 6. SOTA ingested this planning round (filed in research-references.md)

The post-6/30 verifier-moat pivot is now anchored on three 2025–2026 papers:

- **arXiv:2605.18871 — Distributional Energy-Based Models for Uncertainty-Aware Structured LLM
  Reasoning** (May 2026). Decomposed energy = a learned quality scorer (heterogeneous LoRA
  ensemble on one frozen encoder, ~3% trainable) + deterministic analytical constraint penalties;
  the ensemble MEAN ranks, the STDDEV abstains (two-pass regen/abstain). **Matches SC on GSM8K
  (97.0%, beats Math-Shepherd PRM 94.2% / EORM 90.7%) and EXCEEDS SC on MuSR (>64.4%).** This is
  the oracle-distinct, SC-not-saturated win Carnot's moat needs, and independently lands the exact
  Carnot thesis. **This is the post-6/30 pivot experiment.**
- **arXiv:2504.16828 — Process Reward Models That Think (THINKPRM).** A generative reasoning
  verifier that beats SC at equal compute and beats LLM-as-judge + discriminative verifiers using
  ~1% of process labels — corroborates the verifier-moat direction.
- **arXiv:2502.01989 — VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable
  Diffusion.** An intrinsic energy-based verifier whose dense reward is near-oracle — corroborates
  energy-as-verifier as a near-perfect dense reward.

---

## 7. Post-6/30 handoff (the pivot is now EXECUTABLE, not just scaffolded)

The sprint retires 2026-06-30. The locked deliverable (~0.05 agent + FoVer paper) is the ARC
outcome. The **post-sprint headline is the verifier-moat**: execute the distributional-energy-
verifier experiment (arXiv:2605.18871) on a non-saturated structured-reasoning domain (MuSR /
TravelPlanner) where self-consistency is NOT near-ceiling — validation gate: "distributional
energy verifier beats self-consistency, CI95 excluding zero, no model-identity shortcut,
oracle-distinct (`verifier_is_oracle: false`)." `.455` D advances the scaffold to EXECUTABLE
(`pivot_executable_on_7_1=true`, gated on B1 `pivot_readiness_trustworthy`); the capstone states
the handoff so the loop pivots cleanly the instant the sprint retires.
