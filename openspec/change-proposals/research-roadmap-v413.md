# Research Roadmap v413 — 2026.06.413

**EXECUTE the counterexample-guided induction `.412 designed but could not run:
fix the pytest-precondition infra bug that BLOCKED dc22, BANK sc25's 4
PROVISIONAL levels (the single biggest reproduced-level opportunity), and
finally GROW `reproducible_total_levels` off the 39 that `.412 left flat — plus
the operator-mandated MANUFACTURED-variant generic-transfer benchmark.**

- **Milestone:** 2026.06.413
- **Milestone doc:** `openspec/change-proposals/research-roadmap-v413.md`
- **Planned:** 2026-06-19 (Claude Opus 4.8 planner; experiments stay codex/gpt-5.5
  per the ARC Submission Sprint Forcing Function — operator quota-conserve).
- **North star:** solve ARC-AGI-3 accurately AND efficiently (`ops/north-star.md` §0).
- **Sprint window:** the ARC-AGI-3 Submission Sprint Forcing Function is ACTIVE
  through **2026-06-30**; the MAJORITY of this milestone is ARC live-solving and
  `reproducible_total_levels` must grow monotonically.

---

## 1. What `.412 proved (and where it stalled)

`.412 set out to close the three open generic-solver gaps via
counterexample-guided induction and bank new levels. It **moved generic
capability but banked ZERO new reproduced levels** — the metric that matters for
the sprint (`reproducible_total_levels` stayed **39 / 20 games**, exactly the
`.411 close value). The capstone (`results/experiment_4465_capstone_v412.json`)
honestly recorded `generic_solver_gap_state: partial`.

| `.412 task | Goal | Outcome |
|---|---|---|
| A2 exp4456 — generic glyph-rewrite operator | close GAP-4432-LOO-TR87 | **CLOSED** — tr87 re-solves generically without its hand adapter (`tr87_resolved_generically: true`) |
| A5 exp4459 — LOO benchmark v3 | raise `generic_loo_solve_count` | **5 → 6** (loo gate passed) |
| A6 exp4460 — submission package prep | validate a >13-level package | **READY** — 39-level env-matched replay package assembled (`submission_package_ready: true`, not submitted) |
| **A1 exp4455 — SOLVE dc22 L1** | +1 game/level | **BLOCKED, never attempted** — `blocked_baseline_pytest_coverage` |
| **A3 exp4457 — BANK sc25 L2-L5** | +1..+4 levels (biggest opportunity) | **NO ARTIFACT** — task produced nothing; sc25 still `levels_reproduced: 1`, `levels_live_recorded: 5` |
| **A4 exp4458 — first-contact sb26** | +1 game/level | honest negative — `complete: ...routed_no_new_level` (missing operator) |

**The decisive finding: two of the three level-banking failures were
OPERATIONAL, not research dead-ends.**

1. **dc22 (exp4455) was blocked by a brittle precondition, never attempted.**
   Its PRECONDITIONS ran `.venv/bin/pytest -k "config_rule or arc_solver_kit" -q`,
   which inherits the global `--cov-fail-under=99` from `pyproject.toml addopts`.
   A `-k` subset can never cover 99% of the package, so the command exited 1
   ("Coverage failure: total of 26.71% is less than fail-under=99%") and the
   codex agent treated exit-1 as a hard block. The `.411 SUCCESSFUL solve
   (exp4446) used `--no-cov` and ran clean. **The counterexample-guided config-rule
   grounding loop that `.412 designed for dc22 has therefore NEVER actually run.**

2. **sc25 deeper-bank (exp4457) produced no artifact at all.** The task bundled
   "generalize a generic cast-grid phase-FSM operator" AND "bank sc25's
   provisional L2-L5" into one 150-turn task — over-scoped, and likely hit the
   same precondition block — so it skipped and wrote nothing. **sc25's L2-L5 are
   already live-recorded; banking them is the single biggest reproduced-level
   opportunity in the corpus (`provisional_total_levels: 5`).**

3. **sb26 (exp4458)** is a genuine, honest negative: the solver correctly routed
   sb26 → s5i5 / `config_rule_verifier`, but that operator does not fit sb26's
   ordered color-match-item-to-slot-with-undo mechanic. The residual was logged
   as `GAP-4458-SB26` (`missing_color_match_slot_sequence_verifier`).

**The SOTA-ingestion slot (exp4464) flagged the exact lever forward to `.413:**
counterexample-guided re-induction from rejecting execution states
(arXiv:2606.11521; SMT-checked CEGIS predecessor arXiv:2309.16436), with SOAR
(2507.14172), neurally-guided induction (2411.17708), Executable World Models
(2605.05138) + Loop-OWM (2606.12316) as the supporting toolbox.

**Frozen headline (unchanged):** `paper_ready: True` — G1∧G2∧G3∧G4 met (FoVer
0.9131, independently reproduced; `ops/north-star.md` §2). No leaderboard
submission in-loop (operator-only).

---

## 2. The three biggest gaps between current state and the PRD vision

1. **`reproducible_total_levels` has been flat at 39 for a full milestone.** The
   sprint's only headline metric did not move. The fix is not "design a cleverer
   operator" — `.412 already designed the right approach — it is to **make the
   designed approach actually execute** (fix the precondition) and to **harvest
   the level inventory that already exists** (sc25's 5 live-recorded levels). This
   is the highest-leverage thing the project can do before the 2026-06-30 deadline.

2. **A single operational class of bug (coverage-gated smoke preconditions) is
   silently eating level-bank attempts.** dc22 was blocked, sc25 likely the same.
   This will keep recurring on every ARC solve task until a durable guard exists.
   `.413 ships the `--no-cov` smoke-precondition helper + a lint so the dc22-class
   block never happens again.

3. **Generic transfer is still measured on a tiny 2/7 LOO set.** The operator
   shipped a MANUFACTURED-variant generator (color-permutation / reflection,
   mechanic-preserving, guaranteed-solvable) precisely to give a real,
   rule-legal, 25-games × N-variants generalization benchmark — the closest
   legitimate proxy to the held-out OOD eval. Wiring it in is an operator-directed
   MANDATORY-NEXT-MILESTONE-for-`.413 task (`ops/known-issues.md` 2026-06-19).

---

## 3. Architecture (unchanged; this milestone executes against it)

```
                ARC-AGI-3 offline arcade (environment_files/, 25 games)
                                   │  deterministic sim, zero quota
                                   ▼
   recommend_approach(game)  ──►  transfer-routing to the closest SOLVED recipe
   (arc_solve_learning.py)        + retrieve_primitives (LILO documented library)
                                   │
                                   ▼
   GENERIC OPERATORS (arc_solver_kit.py)  ──── few-shot from the grounded
     config_rule_verifier · glyph_rewrite_rule_verifier (NEW .412) ·            example corpus
     object_motion_world_model · cast_grid_phase_fsm_world_model (NEW .413) ·
     color_match_slot_sequence_verifier (NEW .413)
                                   │  propose predicate / world-model
                                   ▼
   COUNTEREXAMPLE-GUIDED GROUNDING (the .413 lever, arXiv:2606.11521/2309.16436)
     propose ─► ground on offline env ─► on REJECTION feed the rejecting
     execution-state back ─► re-induce (bounded budget)   [verifier_is_oracle=true]
                                   │
                                   ▼
   OfflineSolver (verifier-routed best-first)  ──►  arc_solver_kit.reproduce
                                   │  REPRODUCTION GATE: only offline-reproduced
                                   ▼              levels count toward the metric
   ops/arc_solve_registry.yaml  (reproducible_total_levels / per-game gotchas)
                                   │
                                   ▼
   operator-only submission package (env-matched replay; beats the 13 baseline)
```

The live generator is **Qwen3.5-9B-MTP** (`[[project_arc_live_generator]]`) on the
iGPU llama-server — **NEVER the 3090s** (conductor-sacrosanct). The energy
verifier GROUNDS the LLM-proposed predicate/world-model (execution-grounded,
`verifier_is_oracle: true`) — this is a circular/execution-grounded ARC solve,
NOT an oracle-distinct moat headline (Circularity Discipline).

---

## 4. Phases

**PHASE 0 — TRANSITION (exp4466).** Archive `.412 → activate `.413; record the
TRUE `.412 close-state (39/20, gap `partial`, LOO v3 = 6, paper_ready frozen; the
3 open gaps + the dc22 precondition residual).

**PHASE A — GENERIC LIVE SOLVING (operator MANDATORY; the headline; 7 of 7
discretionary slots).**
- **A1 (exp4467) — SOLVE dc22 L1.** Counterexample-guided config-rule grounding,
  with the CORRECTED `--no-cov` precondition. +1 game/level (39/20 → 40/21).
  Closes GAP-4423-DC22. *(level-up attempt; prior_failures exp4455.)*
- **A2 (exp4468) — BANK sc25 provisional L2-L5.** Drive sc25's OWN cast-grid
  world model deeper through the reproduction gate. +1..+4 levels (target up to
  44). The biggest reproduced-level opportunity. *(level-up attempt;
  prior_failures exp4457.)*
- **A3 (exp4469) — GENERIC cast-grid phase-FSM operator.** Generalize sc25's
  world model into a composable two-phase-FSM operator so sc25 re-solves L1
  WITHOUT its hand recipe. Closes GAP-4432-LOO-SC25 (raises
  `generic_loo_solve_count`). *(generic capability; operator_override.)*
- **A4 (exp4470) — BUILD `color_match_slot_sequence_verifier` + SOLVE sb26 L1.**
  Build the specific missing operator the `.412 sb26 negative identified, then
  drive the solve. +1 game/level if banked. *(level-up attempt; prior_failures
  exp4458.)*
- **A5 (exp4471) — FIRST-CONTACT a NEW never-attempted game.** Target ROTATION
  (bp35 / lf52 / re86) via the full generic stack + counterexample-guided
  grounding. +1 game/level or a terminal `complete:` with the residual logged.
  *(level-up attempt; prior_failures exp4458.)*
- **A6 (exp4472) — MANUFACTURED-variant generic-transfer benchmark v4
  (operator-MANDATED).** Wire the SHIPPED variant generator into the
  LOO/generic-transfer benchmark: score the generic solver on 25 games × N
  variants, report `generic_transfer_rate_over_variants`, AND re-measure
  `generic_loo_solve_count_v4` (v3 = 6; target ≥ 7 after A3). *(operator-directed
  2026-06-19; operator_override.)*
- **A7 (exp4473) — SUBMISSION PACKAGE PREP refresh (operator-only submit).**
  Re-validate + refresh the env-matched replay package after the new levels bank
  (target > 39 ≫ the 13 baseline). Does NOT submit. *(operator_override.)*

**PHASE B — RESERVED INFRA (2 slots).**
- **B1 (exp4474)** — registry/gaps hygiene + GAP-4 execution regression guard +
  reconcile `.413 (standing).
- **B2 (exp4475)** — the durable fix for the `.412 churn root cause: a shared
  `--no-cov` ARC precondition smoke helper + a lint/CI guard that FLAGS any ARC
  experiment whose pytest precondition lacks `--no-cov`, plus extend the
  count-integrity lint for the sc25 provisional→reproduced transition.

**PHASE C — HARDWARE CONTINUITY (1 per attached board, exp4476).** KV260 (SSH,
not SD-card) + GateMate + PolarFire forward step or audit.

**PHASE D — SOTA-INGESTION (1 slot, exp4477).** Ingest SOTA for the `.413
bleeding-edge headline (executing CEGIS induction at scale + the
program-induction precision/agreement frontier from GAP-4/GAP-5); map the
strongest method onto the `.414 roadmap.

**PHASE E — CAPSTONE (exp4478).** The `.413 scorecard + the DECISION: did
`reproducible_total_levels` finally GROW off 39 (dc22 banked? sc25 deeper
banked?), did `generic_loo_solve_count` rise, what is
`generic_transfer_rate_over_variants`, is the submission package refreshed? +
the `.414 build backlog.

---

## 5. Dependency graph

```
exp4466 (transition)
   ├─► exp4467 (dc22 L1)            ─┐
   ├─► exp4468 (sc25 bank L2-L5)    ─┤
   ├─► exp4469 (generic cast-grid)  ─┤ (A3 builds on A2's banked sc25 levels)
   ├─► exp4470 (color-match + sb26) ─┤
   ├─► exp4471 (new-game first-contact)─┤
   ├─► exp4472 (variant benchmark v4)  ─┤ (reads A3 closure + the new operators)
   └─► exp4473 (submission refresh)    ─┘ (reads all banked levels)
   exp4474 (hygiene + GAP-4 guard)  ◄── reconciles A1-A7
   exp4475 (precondition --no-cov lint + count-integrity)  (durable fix; independent)
   exp4476 (hardware continuity)    (independent)
   exp4477 (SOTA ingestion .413)    (independent; feeds .414)
   exp4478 (capstone .413)          ◄── aggregates A1-A7 + B1 + B2
```

The conductor runs in listed order; A3 benefits from A2 landing first, A6/A7/E
aggregate the upstream artifacts via the robust aggregate-available helper (no
hard-block-all-False if one is missing).

---

## 6. Hardware requirements

- **Live generator:** Qwen3.5-9B-MTP GGUF on the iGPU llama-server (5.9 GB Q4).
  NEVER the dual RTX 3090s (reserved for the conductor; sacrosanct).
- **ARC offline arcade:** CPU-only deterministic sim (`environment_files/`),
  zero quota. All solve/reproduction work is CPU + iGPU.
- **FPGA continuity (exp4476):** KV260 via `ssh kria` (SSH-reachability is the
  ONLY valid precondition — host SD-card mechanism is FORBIDDEN); GateMate via
  `openFPGALoader -c dirtyJtag --detect` + `nextpnr-himbaechel`; PolarFire via
  `ssh polarfire`.

---

## 7. Disciplines honored

- **ARC-AGI-3 Submission Sprint Forcing Function** (through 2026-06-30): 7 of 7
  discretionary slots on ARC live-solving; `reproducible_total_levels` must grow.
- **ARC Level-Up Attempt Guarantee:** 4 level-up attempts (A1 dc22, A2 sc25
  deeper, A4 sb26, A5 new game) — far above the ≥1 floor; targets rotate across
  dc22 + sc25 + sb26 + a never-attempted game. Verified with
  `scripts/arc_levelup_guarantee_lint.py`.
- **ARC-AGI-3 Incremental-Progress Scoping:** every solve targets +1..+n levels
  on ONE game (sc25 banks L2 first, then deeper); no "solve them all" task.
- **ARC Solve Reproducibility:** every solve is reproduction-gated
  (`arc_solver_kit.reproduce`); only offline-reproduced levels count; registry +
  gaps updated.
- **Inference-Substrate Declaration:** `inference_substrate` is a REQUIRED
  ARTIFACT FIELD on EVERY ARC solve/scoring task (the agent must EMIT it).
- **Failed-Experiment Rerun Discipline + Exclusion-Manifest Cross-Check:** every
  scope-matching task carries `prior_failures:` (all 4 sub-fields) or an
  auditable `operator_override:`.
- **Verdict Terminal-Prefix Discipline:** every `honest_verdict` starts with
  `complete:`/`success:`/`passed:`/`shipped:` (a routed-no-level run is
  `complete:`, never `partial:`).
- **Pre-Launch Preconditions + the .413 fix:** every compute-bound task has a
  PRECONDITIONS step 0; the pytest smoke check uses `--no-cov` (the exp4446
  pattern) so the dc22-class coverage block cannot recur.
- **Circularity / Oracle-Distinctness:** ARC execution-grounded solves carry
  `verifier_is_oracle: true` (NOT a moat headline); the capstone itself is
  `verifier_is_oracle: false`.
- **Operator-Only External Publication:** A7 PREPARES the submission package;
  it NEVER submits.
- **Codex-Default v2 / Sprint routing:** all experiments `agent_type: codex` /
  `gpt-5.5`; planner + retro stay Claude Opus 4.8.
- **Reserved slots:** 2 infra (B1, B2) + 1 hardware (C) + 1 SOTA-ingestion (D).
