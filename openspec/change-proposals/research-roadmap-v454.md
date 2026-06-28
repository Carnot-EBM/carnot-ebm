# Research Roadmap v454 — Post-ARC-Closure Submission Maximization (final pre-deadline milestone)

**Milestone:** 2026.06.454
**Planned by:** outer-loop (Claude Opus 4.8 planner), 2026-06-28
**Predecessor:** 2026.06.453 (FINAL ARC CLOSURE DIAGNOSTIC — reached `WALL_IS_HIDDEN_STATE`)
**Sprint:** ARC-AGI-3 submission sprint, ACTIVE through 2026-06-30 (CLAUDE.md forcing function)

---

## 1. What `.453` proved (the strategic inflection)

`.453` ran the FINAL ARC closure diagnostic and **mechanistically closed the ARC first-win
question** as a B1-trusted result:

- **A1 (exp4914, causal-state-abstraction wall diagnostic, arXiv:2401.12497, live 60.0s):**
  verdict `complete_causal_abstraction_hidden_state_representation_invariant_closure` →
  **`WALL_IS_HIDDEN_STATE`**. The minimal causal abstraction needed to predict changed-cell
  value + progress-to-goal on the failed first-win games **requires a hidden variable**
  (winning-prefix-order state) that the ARC interface cannot expose. The solved-game positive
  control classified OBSERVABLE (non-degenerate).
- **B1 (exp4918) audited it trustworthy:** all six trust gates true (real transitions, not a
  value table, observable claims verified readable, positive control observable, oracle-distinct
  + planner-blind, numbers match the fork).
- **Capstone (exp4923):** `complete_capstone_v453_wall_is_hidden_state_arc_closure`.

**Consequence:** the live first-win wall is **representation-invariant by construction**. No
offline representation over observable inputs recovers the discriminating variable — this closes
the multi-milestone world-model fork (.431→.453). **The deliverable LOCKS to the current ~0.05
first-win agent + the publishable FoVer verifier-ensemble paper (paper_ready=true).** Per the
capstone and the operator handoff: **do NOT queue representation #5; do NOT reopen any nulled
generation/perception/selection/energy lever.**

`.453` also delivered, on the scored lanes:

| Lane | Result |
|---|---|
| A2 level-up | **cn04 L2→L3 banked** (`live_agent_self_discovery`, reproduction-gated) → `reproducible_total_levels` **68 → 69** (first bank in 4 milestones) |
| A3 self-play | bp35 checkpoint refreshed (FR-11 continuous self-learning) |
| A4 held-out go/no-go | genuine-live PARTIAL (21/25 games, 84 attempts, 3627 s, soft-budget stop) — **NOT flagged_adversarial** (the .452→.453 stamping fix resolved the 3-milestone recurring flag); 4 games remain |
| B2 submission package | `success_submission_package_ready_final_pre_deadline` (15.146 GB < 16 GB, frozen Qwen3.5-9B-MTP iGPU stack) |
| B3 retro infra | shipped the mtime-fallback + runtime-stamping modules + wiring proposal (operator wire pending — the autonomous loop cannot edit the conductor) |
| C KV260 | reachable (graduated terminal, SSH-only) |
| D SOTA | distributional-energy-verifier pivot scaffold built (arXiv:2605.18871) |

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The locked deliverable is not yet maximized for the 6/30 submission.** ARC is closed, but the
   *scored* submission can still improve along the two axes that remain open: **deepening**
   (more reproduced levels → more games the agent can score) and **action efficiency** (the
   leaderboard squares `(human_actions/agent_actions)` → a fewer-actions improvement is squared
   in the score). Neither is at its ceiling.
2. **The held-out go/no-go number is incomplete.** `.453` A4 finished only 21/25 games before its
   soft-budget stop. The operator's 6/30 decision needs the CLEAN full-25-game rate + CI — a
   resume-to-finish, not a fresh run.
3. **The verifier-moat (Carnot's core value-add) has no executed post-sprint experiment yet.**
   ARC closure makes the FoVer/verifier-moat program the next headline. `.453` D scaffolded the
   distributional-energy-verifier harness (arXiv:2605.18871 — which independently lands the exact
   Carnot thesis: learned quality scorer + deterministic constraint penalties + uncertainty). `.454`
   keeps it a scaffold (majority-ARC governs through 6/30) but readies it to execute the instant
   the sprint retires.

---

## 3. `.454` design — execute the locked deliverable, do not chase the closed wall

`.454` is the **last (or second-to-last) milestone before the 6/30 deadline.** Its honest headline
is **submission-score maximization for the locked deliverable**, NOT a new research fork. The
majority-ARC allocation shifts from first-win-wall work (closed) to the two open scored axes:

```
                       reproducible_total_levels = 69 (post cn04)
                                     |
        +----------------------------+----------------------------+
        |  DEEPEN (more scored games)       EFFICIENCY (squared)  |
        |  A1 sp80 L2->L3  (headline)       D  MATM similarity-    |
        |  A2 su15 L2->L3  (guarantee)         keyed retrieval     |
        +----------------------------+----------------------------+
                                     |
              A3 self-play (FR-11 learned verifier, every milestone)
              A4 held-out RESUME 21/25 -> 25/25 (CLEAN 6/30 go/no-go)
                                     |
   B2 FINAL submission package harden + operator checklist (the deadline deliverable)
   B1 audit (banks real + efficiency oracle-distinct)   B3 stamping backfill/readiness
   C KV260 SSH continuity        D scaffold the post-6/30 distributional-energy-verifier pivot
                                     |
                          E capstone — submission readiness scorecard + post-6/30 handoff
```

### Allocation (mirrors the `.453` reserved-slot contract: 3 infra + 1 hardware + 1 SOTA/efficiency)

| Phase | Task | Track | Agent | Why |
|---|---|---|---|---|
| 0 | exp4924 archive .453 → activate .454 | transition | codex | record the closure close-state |
| A1 | exp4925 DEEPEN sp80 L2→L3 (HEADLINE level-bank) | arc-north-star | codex | majority-ARC level-growth (no rep #5) |
| A2 | exp4926 DEEPEN su15 L2→L3 (Level-Up Guarantee) | arc-north-star | codex | ≥1 banking attempt; maximize monotonic growth before deadline |
| A3 | exp4927 self-play (rotated banked game) | arc-north-star | codex | FR-11 continuous self-learning, every milestone |
| A4 | exp4928 held-out RESUME 21/25 → 25/25 | arc-north-star | codex | CLEAN 6/30 go/no-go (deadline lane) |
| B1 | exp4929 audit banks + efficiency | infra | codex | banks real/non-duplicate; D oracle-distinct |
| B2 | exp4930 FINAL submission package harden | infra | codex | the deadline deliverable; submits=false |
| B3 | exp4931 stamping backfill + wiring readiness | infra | codex | retro top-3; closes duration_s=None for .454 arms |
| C | exp4932 KV260 SSH continuity | hardware | codex | Hardware-Task Continuity (SSH-only) |
| D | exp4933 MATM similarity-keyed retrieval (efficiency) + DEV pivot scaffold | sota-ingestion | codex | the squared-scored efficiency lever (operator-flagged) |
| E | exp4934 capstone .454 | capstone | codex | submission-readiness scorecard + post-6/30 handoff |

**Continuous self-learning (research-program.md requirement):** A3 is the explicit Tier-3 / FR-11
experiment — the learned ARC verifier trains on the self-play traces and checkpoints, improving
across runs. D's MATM retrieval is a second self-learning surface (the agent reuses its own
within-game rollouts).

### Disciplines honored

- **ARC Live-Path Reachability:** every ARC task improves a LIVE-reachable mechanism
  (`arc_loop_solve` / `GameAdapter` / `StepwiseExplorer` / `E3AgentPolicy`); `arc_orphan_solver_lint`
  must pass. No parallel solver the live agent cannot reach. `solve_provenance` declared on every
  solve task (`live_agent_self_discovery` for A1/A2/A3; `development_proxy` for A4/D measurements).
- **ARC Solve Reproducibility:** banks count only through `arc_solver_kit.reproduce`; registry-precheck
  before any solve (no re-solving an already-reproduced level → duplicate is CRITICAL).
- **ARC Incremental-Progress Scoping:** +1 level per game; rotated targets; no "FULL solve."
- **ARC Level-Up Attempt Guarantee:** A1 + A2 are two banking attempts (≥1 floor).
- **Circularity / Oracle-Distinctness:** D's MATM retrieval scores candidates via the verifier
  router (`verifier_is_oracle: false`, oracle-distinct) — energy-as-router, not oracle.
- **Operator-Only External Publication:** B2 prepares the package + checklist; `submits=false`.
- **Pre-Launch Preconditions:** every compute-bound task gates on arcade/generator/SSH; GPU fix
  (2026-06-27) — offline induction accepts the conductor's GPU-0 CUDA generator, NOT iGPU-pinned;
  the LIVE submission stack (B2) stays frozen Qwen3.5-9B-MTP on the iGPU.
- **Codex-Default v2:** all experiments `agent_type: codex`/`gpt-5.5`; planner + retro stay Opus
  (operator's sprint quality choice).
- **Verdict Terminal-Prefix + Principle-Annotated fields:** every task.

### What `.454` does NOT do (closed/retired — do not re-propose)

Representation #5 / any new world-model fork; energy-as-ARC-lever (CONCLUDED negative 2026-06-26);
macro/horizon-collapse, click-heatmap generator, trust-gate flip (all empirically retired); the
TTT-on-code engine; stronger local code inducers; decision-need targets; action-prefix latents;
coverage/exploration/selection/perception-from-grid. First-win-wall chasing is closed.

---

## 4. Dependency graph

```
exp4924 (transition)
   ├─> exp4925 A1 deepen sp80 ─┐
   ├─> exp4926 A2 deepen su15 ─┤
   ├─> exp4927 A3 self-play    ├─> exp4929 B1 audit ─┐
   ├─> exp4928 A4 held-out     │                      │
   ├─> exp4933 D  MATM eff ────┘                      │
   ├─> exp4930 B2 package ────────────────────────────┤
   ├─> exp4931 B3 stamping ───────────────────────────┤
   └─> exp4932 C  KV260 ──────────────────────────────┤
                                                       v
                                            exp4934 E capstone .454
```

A1/A2/A3/D feed B1 (the audit gates trust in the headline numbers). All arms feed the capstone,
which skips any `flagged_adversarial` upstream per the fabrication gate.

---

## 5. Hardware requirements

- **Offline ARC induction (A1/A2/A3/A4/D):** conductor's dedicated **GPU-0 CUDA llama-server**
  (`CARNOT_ARC_GENERATOR_CUDA_GPU=0`, card ≥13 GB free) OR the iGPU HIP server; health-check via
  `_ensure_server()`. Per the 2026-06-27 GPU-allocation directive, offline induction is NOT
  iGPU-pinned. Do NOT block merely because `CUDA_VISIBLE_DEVICES` is set.
- **LIVE submission stack (B2):** FROZEN — Qwen3.5-9B-MTP on the **iGPU** (Kaggle ~16 GB parity);
  never the 3090s for the live stack.
- **KV260 (C):** SSH-reachability only (`ssh kria 'true'`); NEVER a host SD-card / block-device
  precondition.

---

## 6. Post-6/30 handoff (stated now so the loop pivots cleanly)

The sprint retires 2026-06-30. The locked deliverable (~0.05 agent + FoVer paper) is the ARC
outcome. The **post-sprint headline is the verifier-moat**: execute the distributional-energy-verifier
experiment (arXiv:2605.18871) on a non-saturated structured-reasoning domain (MuSR / TravelPlanner /
TACO / Knights&Knaves) where self-consistency is NOT near-ceiling — the validation gate is
"distributional energy verifier beats self-consistency, CI95 excluding zero, no model-identity
shortcut, oracle-distinct." `.454` D readies (does not start) this; the capstone states the handoff.
