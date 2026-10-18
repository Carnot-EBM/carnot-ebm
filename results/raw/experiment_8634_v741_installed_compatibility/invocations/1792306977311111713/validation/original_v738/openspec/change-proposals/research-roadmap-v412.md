# Research Roadmap v412 — 2026.06.412

**CLOSE the three OPEN ARC generic-solver gaps (dc22 first-contact, tr87
glyph-rewrite, sc25 cast-grid) with counterexample-guided induction, BANK
sc25's 4 provisional levels, re-measure the falsifiable generic_loo_solve_count,
and PREP an updated operator submission package** (offline 39 levels ≫ the
13-level prior submitted baseline).

- **Milestone:** 2026.06.412
- **Planner:** outer-loop Claude Opus 4.8 (operator quota-conserve 2026-06-19
  sprint — planner/retro stay Claude; ALL experiments `agent_type: codex` /
  `gpt-5.5`). This is a pre-staged roadmap; the conductor's planner is SKIPPED.
- **Live generator (FROZEN for the sprint):** Qwen3.5-9B-MTP on the iGPU
  ([[project_arc_live_generator]]); **NEVER the 3090s** (conductor-sacrosanct).
- **Headline (FROZEN):** `paper_ready=True` (FoVer 0.9131; G1–G4 met). No
  publication work this milestone; the ARC submission sprint is the priority.
- **No leaderboard submission in-loop** — operator-only (Operator-Only External
  Publication Discipline). A6 PREPARES + VALIDATES a package; it does not submit.

---

## 0. Why this milestone (the forcing function + the open gaps)

The **ARC-AGI-3 Submission Sprint Forcing Function** (CLAUDE.md, MANDATORY
through **2026-06-30**) mandates the MAJORITY of every milestone on ARC
live-game-solving progress, with `reproducible_total_levels` growing
monotonically. This milestone allocates **6 of 6 discretionary slots (A1–A6)**
to ARC live-solving, plus the reserved 2 infra / 1 hardware-continuity / 1
SOTA-ingestion slots.

### What .411 proved (`results/experiment_4453_capstone_v411.json`)

`generic_solver_gap_state: partial`. The example corpus + new generic operators
moved the needle but did not close the gap:

| .411 result | Outcome |
|---|---|
| g50t L1 banked cleanly (exp4443) | +1 game/level → registry **38/19** |
| GENERIC `config_rule_verifier` operator (exp4444) | ft09 re-solved generically; **dc22 NOT grounded** (open) |
| GENERIC `object_motion_world_model` operator (exp4445) | ar25 + ka59 residuals closed; world-model acc 0.25→1.0 with examples |
| Generic first-contact bank (exp4446) | vc33 L1 banked (routed→s5i5) → registry **39/20** |
| LILO documented primitive library (exp4447) | library_coverage 1.0, retrieval p@1 1.0, zero constant leaks |
| LOO generic-solve benchmark v2 (exp4448) | **generic_loo_solve_count 2 → 5** (loo_gate_passed) |

**Registry now: 39 reproducible levels / 20 games** (`ops/arc_solve_registry.yaml`),
`provisional_total_levels: 5`.

### The .412 build backlog (the capstone's `next_backlog.open_gap_ids`)

Three generic gaps stay OPEN — all of the form *"the generic operator could
not GROUND/INDUCE the rule":*

1. **GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT** — dc22 (config/toggle maze;
   `is_spatial_planning=false`, medium) did not ground via the .411
   `config_rule_verifier`. The **last OPEN first-contact game**; a +1 game/level
   opportunity (39/20 → **40/21**).
2. **GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER**
   — tr87 (the deepest single-game solve, 6 levels) re-solves only via its hand
   adapter. A GENERIC glyph-rewrite operator lets unseen config-substitution
   games be solved (raises generic_loo_solve_count 5→6).
3. **GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER** —
   sc25 re-solves only via its hand recipe. A GENERIC cast-grid/phase-FSM
   world-model operator closes the residual **and can BANK sc25's 4 PROVISIONAL
   levels** (L2–L5 are live-recorded but NOT offline-reproduced — the single
   biggest reproduced-level opportunity in the registry).

### The fresh lever (`research-references.md` .412 sweep, 2026-06-19)

All three gaps are grounding/induction failures. The unifying fresh SOTA lever
is **counterexample-guided inductive synthesis** (arXiv:2309.16436 LLM+SAT
CEGIS; arXiv:2606.11521 counterexample-guided reasoning agents): when a generic
operator proposes a predicate that does NOT reproduce on the offline env, feed
the rejecting execution-state back as a counterexample and re-induce
(propose → ground → refute → re-propose). Wired into the dc22 config-rule
operator (A1), the tr87 glyph-rewrite operator (A2, SOAR-style self-improvement
arXiv:2507.14172), and the sc25 phase-FSM world model (A3, Executable World
Models arXiv:2605.05138 / Loop-OWM arXiv:2606.12316). This composes with the
.411 LILO documented library (the operator retrieves a documented primitive
before re-inducing).

---

## 1. The three biggest gaps between current state and PRD vision

1. **Generic first-contact still cannot GROUND a brand-new mechanic.** The .411
   operators close LOO residuals on KNOWN games and bank config-routed vc33, but
   dc22 — a genuinely-different toggle/maze mechanic — did not ground. The
   held-out Kaggle leaderboard tests exactly this. **A1 + A4** attack it
   (dc22 via CEGIS-grounded config-rule; a never-attempted survey game via the
   full generic stack).
2. **Provisional levels are not reproduced.** sc25 has 4 live-recorded levels
   (L2–L5) that do NOT pass the offline reproduction gate, so they do not count
   toward the submission metric. **A3** banks them via a generic cast-grid
   world-model operator (reproduction-gated).
3. **The operator submission package is stale at 13 levels** while offline is at
   39. The operator must submit MULTIPLE times before 2026-06-30, improving each
   time. **A6** prepares + validates an env-matched replay package the operator
   can submit to beat the 13-level baseline (operator-only submit).

---

## 2. Architecture (where .412 touches the stack)

```
                 ARC-AGI-3 live solver (the sprint's deliverable)
                 ─────────────────────────────────────────────────
  recommend_approach (transfer-routing)         arc_solve_learning.py
        │                                          (route unseen → nearest recipe)
        ▼
  retrieve_primitives(digest)  ◄── LILO documented library (exp4447, .411)
        │                          [A2/A3 ADD glyph_rewrite + cast-grid ops]
        ▼
  GENERIC OPERATORS  ─────────────────────────  arc_solver_kit.py
   • config_rule_verifier (exp4444)  ── A1: + CEGIS counterexample grounding → dc22
   • object_motion_world_model (exp4445)
   • glyph_rewrite_rule_verifier  ◄── A2 NEW (close tr87 LOO residual)
   • cast_grid_phase_fsm_world_model ◄── A3 NEW (close sc25 LOO residual + bank L2–L5)
        │
        ▼
  propose-then-GROUND (verifier_is_oracle=true)   Qwen3.5-9B-MTP (iGPU; FROZEN)
        │   live induction; the offline env is the oracle
        ▼
  arc_solver_kit.reproduce  ──────────────────  THE GATE (only reproduced levels count)
        │
        ▼
  ops/arc_solve_registry.yaml  +  ops/verifier_gaps.md   (capture-don't-waste)
        │
        ▼
  A6: env-matched offline-reproduced REPLAY package  →  operator submits (operator-only)
```

**Self-learning (PRD FR-11 / Continuous Self-Learning Tier 2):** every new
generic operator + documented-library entry compounds — the system gets better
at unseen games by accumulating reusable, retrievable, counterexample-hardened
operators, NOT by transferring learned value weights (that lineage is RETIRED,
exp4342). A5 re-measures the falsifiable compounding metric (`generic_loo_solve_count`).

---

## 3. Phases & experiments (12 tasks, exp4454–exp4465)

### PHASE 0 — TRANSITION
- **exp4454** — archive .411 → activate .412; assert YAML parses + smart-subset
  pre-test gate green; record the TRUE .411 close-state (39/20,
  generic_loo_solve_count=5, 3 open gaps).

### PHASE A — GENERIC LIVE SOLVING (operator MANDATORY; the headline; 6 of 6 discretionary slots)
- **exp4455 (A1)** — SOLVE **dc22 L1**: extend the .411 `config_rule_verifier`
  with **counterexample-guided grounding** (re-induce from the rejecting
  execution-state when a predicate does not reproduce), per-game RE the
  toggle/maze delta, register a `GameAdapter`, drive the OfflineSolver, bank L1
  → +1 game/level (39/20 → **40/21**). Closes GAP-4423-DC22.
- **exp4456 (A2)** — GENERIC **glyph-rewrite-rule verifier operator**:
  generalize tr87's hand `glyph_rewrite_matcher` into a composable
  `glyph_rewrite_rule_verifier` operator (SOAR-style self-improving rule
  synthesis) so tr87 re-solves WITHOUT its own adapter. Closes
  GAP-4432-LOO-TR87 (raises generic_loo_solve_count). 0 new levels (re-solve).
- **exp4457 (A3)** — GENERIC **cast-grid phase-FSM world-model operator** +
  **BANK sc25 provisional L2–L5**: generalize sc25's per-game world model into a
  two-phase-FSM (toggle cast-grid → navigate to exit) operator, drive it to
  REPRODUCE sc25's live-recorded L2+ (provisional → reproduced,
  reproduction-gated). Closes GAP-4432-LOO-SC25; **+1..+4 levels** (the biggest
  banking opportunity). Incremental-Progress-scoped: gate = ≥1 new sc25 level.
- **exp4458 (A4)** — FIRST-CONTACT a **NEW never-attempted survey game** (sb26 /
  bp35 / lf52 / re86; sb26 is the most config-amenable) via the full generic
  stack (transfer-routing + documented-library retrieval + the new operators) →
  +1 game/level if solved, else terminal `complete:` with the residual logged as
  a missing-verifier gap.
- **exp4459 (A5)** — LEAVE-ONE-OUT generic-solve benchmark **v3**: re-measure
  `generic_loo_solve_count` after A2 (tr87) + A3 (sc25) land. v2 was 5/7; target
  ≥6. The falsifiable progress metric (counts NO new levels — a re-measurement).
- **exp4460 (A6)** — SUBMISSION PACKAGE PREP (operator-only submit): validate +
  assemble the env-matched offline-reproduced REPLAY package across all
  reproduced games/levels (≥39, ≫ the 13-level prior submitted baseline) so the
  operator can submit to beat it. `submission_package_ready: bool`; **does NOT
  submit**.

### PHASE B — RESERVED INFRA (2 slots)
- **exp4461 (B1)** — registry/gaps hygiene + GAP-4 execution regression guard +
  reconcile .412; verify capstone `verifier_is_oracle` stamping stays durable.
- **exp4462 (B2)** — provisional-vs-reproduced **count-integrity lint** +
  submission-replay env-match guard: a CI/pre-commit guard that (a) a level is
  never counted as `reproduced` unless `arc_solver_kit.reproduce` passes, and
  (b) the A6 replay package's actions actually reproduce on the env layout. Pins
  the sprint's monotonic metric against silent inflation.

### PHASE C — HARDWARE CONTINUITY (1 per attached board)
- **exp4463 (C)** — KV260 (**SSH, NOT SD-card**) + GateMate + PolarFire forward
  step or precondition-gated audit; never forget the FPGAs.

### PHASE D — SOTA-INGESTION (.412 bleeding-edge track)
- **exp4464 (D)** — ingest SOTA for **counterexample-guided generic rule/world-
  model induction** for interactive games; map the strongest 3–5 methods onto
  the .413 roadmap. RELIABLE channel only (never /deep-research); every method
  cites a VERIFIED arXiv ID.

### PHASE E — CAPSTONE
- **exp4465 (E)** — the .412 scorecard + the GENERIC-SOLVER-GAP DECISION (did A1–A3
  close gaps + raise generic_loo_solve_count + bank levels?) + the SUBMISSION-
  READINESS decision (is the operator package ready to beat 13?). G1–G4 gate
  stays True (FROZEN).

---

## 4. Dependency graph

```
exp4454 (transition)
   │
   ├─► exp4455 (A1 dc22 solve) ─────────────┐
   ├─► exp4456 (A2 tr87 glyph-rewrite op) ──┤
   ├─► exp4457 (A3 sc25 cast-grid op+bank) ─┤
   ├─► exp4458 (A4 new-game first contact) ─┤
   │                                        ▼
   │                              exp4459 (A5 LOO v3 — reads A2/A3 operators)
   │                                        │
   ├─► exp4460 (A6 submission package) ─────┤  (reads the reconciled registry)
   ├─► exp4461 (B1 hygiene) ────────────────┤
   ├─► exp4462 (B2 count-integrity lint) ───┤
   ├─► exp4463 (C hardware) ────────────────┤
   ├─► exp4464 (D SOTA-ingestion) ──────────┤
   │                                        ▼
   └────────────────────────────► exp4465 (E capstone — aggregates A1–A6, B, D)
```

A5/A6/B1/E read upstream artifacts; they use the robust aggregate-available
helper (read available, report gaps — never hard-block-all-False if one is
missing). No `requires:` chain references a retired exp_id.

---

## 5. Hardware requirements

| Resource | Use | Discipline |
|---|---|---|
| iGPU (Radeon 890M) | Qwen3.5-9B-MTP live generator for ARC induction (A1–A5) | FROZEN; **NEVER the 3090s** ([[project_arc_live_generator]]) |
| CPU | offline arcade sim (deterministic, zero-quota), reproduction gate, hygiene, lints | all ARC solves are CPU-reproducible |
| KV260 (ssh `kria`) | hardware-continuity forward step / audit | SSH-reachability ONLY (KV260 SSH-Not-SD-Card) |
| GateMate / PolarFire | opportunistic forward step / audit | `nextpnr-himbaechel`; ssh `polarfire` |

No 3090 inference anywhere in this milestone. No new SOTA-GGUF model is needed:
the ARC sprint freezes the generator to Qwen3.5-9B-MTP; the non-ARC tasks
(hygiene, lint, SOTA-ingestion, capstone, hardware) use no LLM.

---

## 6. Disciplines applied (the .410/.411 lessons, enforced this milestone)

- **`inference_substrate` is a REQUIRED ARTIFACT FIELD on EVERY ARC solve/scoring
  task** — `adversarial_verify` reads it from the ARTIFACT, so the agent must
  EMIT it (the .410 g50t false-positive-quarantine class). `live_llm_inference`
  when the Qwen induction runs live (>60s), else
  `verifier_ensemble_against_cached_candidates` (≥1s floor) — never `None`.
  exp4462 CI-guards this.
- **Reproduction-gated** — only `arc_solver_kit.reproduce`-passing levels count;
  provisional ≠ reproduced (ARC Solve Reproducibility Discipline).
- **Incremental-Progress-scoped** — +1..+n levels on ONE game; never "solve them
  all"; each task's gate is ≥1 new reproduced level OR a closed LOO residual.
- **verifier_is_oracle** — every ARC solve is execution-grounded
  (`verifier_is_oracle=true`); the capstone declares `false` for itself so
  CIRCULAR_MOAT_OVERCLAIM does not fire (Circularity Discipline).
- **Terminal-prefixed honest_verdict** — `complete:/success:/passed:/shipped:`; a
  routed-no-level run is `complete:` (NOT `partial:` — the exp4423 FAIL-loop fix).
- **Pre-Launch Preconditions** — every solve task checks offline env files + the
  live generator (GGUF cached OR iGPU llama-server, NEVER 3090s) BEFORE any
  induction; on a miss, `blocked_<resource>`, no fabrication.
- **Operator-Only External Publication** — A6 prepares + validates; it does NOT
  submit. No leaderboard/arxiv calls anywhere in-loop.
- **Codex-Default sprint routing** — ALL experiments `agent_type: codex` /
  `gpt-5.5`; planner/retro stay Claude Opus 4.8.

---

## 7. Falsifiable milestone gate (the .412 DECISION)

The capstone (exp4465) answers, from non-flagged artifacts only:

1. Did **dc22 ground + bank L1** (A1) → registry 40/21? (or honest residual)
2. Did the **tr87 glyph-rewrite operator** (A2) close GAP-4432-LOO-TR87?
3. Did the **sc25 cast-grid operator** (A3) bank ≥1 provisional level + close
   GAP-4432-LOO-SC25?
4. Did **A4** bank a never-attempted game?
5. Did **generic_loo_solve_count** (A5) rise above 5?
6. Is the **operator submission package** (A6) ready to beat 13 levels?

**Monotonic sprint metric:** `reproducible_total_levels` ≥ 39 (target 40–44 with
dc22 + sc25 deepening). A flat/lower count with honest residuals logged is a REAL
finding, not a failure — but the design targets growth on dc22 (A1) and the 4
sc25 provisional levels (A3).
