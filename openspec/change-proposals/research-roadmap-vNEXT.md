# Research Roadmap v436 — VALID-TEST THE GUIDANCE-CLASS GENERATION LEVERS

**Milestone:** 2026.06.436
**Planned:** 2026-06-25 (Claude Opus 4.8, outer-loop planner)
**Predecessor:** 2026.06.435 (STOP SELECTING, START ACTING)
**Sprint:** ARC-AGI-3 Submission Sprint (MANDATORY through 2026-06-30 — **5 days to the ARC Prize Milestone #1 deadline**)

---

## TL;DR

`.435` was the **11th consecutive `bridge_crossed_for_solve=False` milestone**, but its real
output was a sharp **trust-correction** that reframes the entire search:

- **A1 (the leader's online action-learning driver) — the ONLY `.435` lever VALIDLY tested
  (`arms_non_degenerate=True`) — genuinely nulled.** 66 online Adam steps, distinct per-arm
  action distributions, coordinate head differs from frozen, yet `online_warm_vs_frozen_delta=0.0`
  (all arms at the 0.04 baseline). The operator's highest-leverage 2026-06-24 bet is **answered: NO**.
  → the online-action-learning lever **RETIRES**.
- **A2 (active-probe disambiguation) tested DEAD CODE** (`probe_actions_taken=0`,
  `hypothesis_posterior_built=False`, `reason: probe_mechanism_did_not_run`).
- **B1 silent-bug audit (operator-mandated): 12 nulls audited, 5 must reopen.** The decisive finding —
  the **guidance-class GENERATION levers were ALL tested on dead/byte-identical code** and are
  therefore **untrustworthy nulls that were never validly tested**: P2 **goal-energy generation**
  (cloned cached baseline), P3 **energy-fitness QD** (byte-identical arms), P1 Go-Explore amortized prior
  (`(1,64,64)` archive), P4 hierarchical subgoal (empty decomposition).

**`.436` valid-tests the two strongest guidance-class generation levers**, each with the
non-degeneracy gate applied **INLINE** (the `.435`-A1 discipline) and the null-delta carve-out markers
emitted so an honest no-lift null is **not quarantined**:

- **A1 (HEADLINE): goal-energy candidate generation** — reopen of exp4640, the operator's explicit
  guidance-class redirect target. The goal-energy arm must score **REAL candidate states** (not cloned
  baseline) and generate a candidate pool that DIFFERS from baseline before any lift is measured.
- **A2: energy-fitness QD generation** — reopen of exp4653, with **distinct QD + random-mutation
  candidate pools** (not byte-identical arms). The MAP-Elites-with-gradients family applied to making
  the winning L1 candidate appear.

Plus the standing sprint pipeline: A3 self-play **bank +1 (64→65)** + verifier train, A4 held-out
first-win readiness + submission-config confirm, A5 persist, A6 integration. Two **INFRA** tasks
that fix the two `.435` adversarial-verify escapes (don't quarantine honest generation nulls;
flag the A2-class declared-but-unrun mechanism). C KV260 continuity. D maps the `.437` SOTA frontier.
E capstone + HEADLINE DECISION.

---

## What `.435` proved (the inputs to this plan)

| Lever | Verdict | Read |
|---|---|---|
| **A1 online driver** (exp4726) | `online_action_learning_no_first_win_lift_residual_online_signal_genuinely_too_sparse`; `arms_non_degenerate=True`; `delta=0.0` | **Validly tested → genuine null → RETIRE.** The leader's mechanism does not lift first-win. (Quarantined by a TAUTOLOGY false-positive — INFRA-1 fixes the carve-out.) |
| **A2 active probe** (exp4727) | `active_probe_no_new_level_residual_budget_insufficient`; `probe_actions_taken=0` | **Tested DEAD CODE** — the probe never ran. Reopen deferred to `.437` (tracked); INFRA-1 adds the guard. |
| **A3 self-play** (exp4728) | `success: ar25_L3_offline_reproduced` | **Banked +1 → registry 63→64.** The self-play loop works for deepening. |
| **B1 audit** (exp4725) | `silent_bug_audit_12_nulls_5_must_reopen` | **Guidance-class generation levers tested dead code** → the `.436` reopen list (P1–P4). |
| **B2 guard** (exp4732) | shipped `LEVER_EXERCISE_EVIDENCE_DEGENERATE` | Mechanizes the dead-code catch going forward (missed A2's 0-probe case — INFRA-1 extends it). |
| **A4 readiness** | **no artifact** (codex 4800s cap) | `.436` A4 uses the checkpoint/resume fix so the SCORE survives the cap. |
| **D ingestion** (exp4734) | `flagged_for_v436`: epistemic-MCTS + causal-probe | Carried to `.437` (the `.436` reopens are the more-grounded headline). |
| **Capstone** (exp4735) | `bridge_crossed_for_solve=False` (11th); `reproducible_total_levels_delta=1`; `paper_ready=True` | Capability grew (64), bridge not crossed. `next_milestone_fallback.b1_reopen_list` = the `.436` plan. |

**Triply-grounded direction.** (1) operator redirect 2026-06-23: "the 0.04 wall is generation-GUIDANCE,
NOT depth/coverage — concentrate on the guidance class (goal-energy / expansion-prior)"; (2) B1's audit:
the guidance-class generation levers tested dead code → must reopen; (3) the energy-config-space memory:
"the live agent IS an energy-refinement loop; the wall is make-a-winner-appear, not select."

---

## Architecture — the live agent + the `.436` generation valid-tests

```
                ARC-AGI-3 LIVE HIDDEN-GAME DISCOVERY AGENT  (the scored deliverable)
   +----------------------------------------------------------------------------------+
   |  E3AgentPolicy  (arc_competition_agent.py -- the SCORED Kaggle agent)             |
   |     StepwiseExplorer -- online world-model induction (arc_live_ttt / LocalGGUF)   |
   |        |                       |  gated by WorldModelVerifier                     |
   |        v                       v                                                  |
   |   CANDIDATE GENERATION ---->  verifier-routed best-first search --> plan_in_model  |
   |   (the 0.04 WALL: the winning L1 candidate is PRESENT but the                      |
   |    generator's proposal distribution under-weights it -- make it APPEAR)          |
   +----------------------------------------------------------------------------------+
        ^                                   ^
        |  A1 (HEADLINE)                    |  A2
   +----+-------------------+        +------+------------------------+
   | GOAL-ENERGY CANDIDATE  |        | ENERGY-FITNESS QD GENERATION  |
   | GENERATION (reopen 4640)|       | (reopen 4653)                 |
   | score REAL candidate    |       | distinct QD + random-mutation |
   | states by graded        |       | pools; energy=fitness; evolve |
   | goal-distance energy;    |       | the pool toward the winner    |
   | bias proposals toward    |       | (MAP-Elites + gradients)      |
   | low-goal-energy states   |       |                               |
   +---------+---------------+        +----------+--------------------+
             |  verifier_is_oracle:false (oracle-distinct: scores candidates, never the win-check)
             v                                    v
   +------------------------------------------------------------------+
   |  NON-DEGENERACY GATE (INLINE, the .435-A1 discipline + B2)         |
   |  PROVE first: distinct per-candidate scores, non-zero variance,    |
   |  a candidate pool that DIFFERS from baseline.                      |
   |  Degenerate -> *_generation_arms_degenerate_confirmed_harness_bug  |
   |  (a BUG to fix, NOT a capability null). Non-degenerate + zero lift |
   |  -> honest null WITH null_delta markers (not quarantined).         |
   +------------------------------------------------------------------+
```

**Discipline (CLAUDE.md ARC Live-Path Reachability + Circularity):** both A1 and A2 modify the LIVE
`E3AgentPolicy` candidate-generation path (in the scored agent's import closure — `arc_orphan_solver_lint`
+ `test_arc_submitted_agent_parity` stay green). The goal-energy / QD-fitness signals are
**oracle-distinct** (they score candidate states; they do NOT run the executable win-check) →
`verifier_is_oracle:false`, gate-eligible. `solve_provenance: live_agent_self_discovery`.

---

## Phases & dependency graph

```
  exp4736  PHASE 0   transition (archive .435 -> activate .436)            [codex, 40t]
     |
     +--> exp4737  A1  goal-energy candidate generation (HEADLINE, reopen 4640)   [live, 160t]
     +--> exp4738  A2  energy-fitness QD generation (reopen 4653)                 [live, 160t]
     +--> exp4739  A3  self-play bank +1 (64->65) + verifier train               [150t]   (INDEPENDENT -- Level-Up Guarantee)
     +--> exp4743  B1(INFRA-1)  adversarial_verify carve-out hardening           [60t]    (fixes the .435 A1+A2 escapes; runs EARLY)
     +--> exp4744  B2(INFRA-2)  submission-package readiness validation          [60t]    (deadline-critical)
     +--> exp4745  C   KV260 hardware continuity                                 [60t]
     +--> exp4746  D   SOTA-ingestion -> .437 frontier                           [50t]
                       |
   (A1,A2 done) --> exp4740  A4  held-out first-win readiness + submission-config [60t]
   (A1,A2 done) --> exp4741  A5  persist strongest .436 generation primitive      [60t]
   (A1,A2 done) --> exp4742  A6  integration gate                                 [60t]
                       |
   (all done)  --> exp4747  E   CAPSTONE .436 + HEADLINE DECISION                 [60t]
```

- **A1 / A2** are the genuinely-independent guidance-class generation valid-tests (different mechanisms:
  goal-distance energy scoring vs energy-as-fitness QD evolution).
- **A3** is INDEPENDENT of A1/A2 so the **ARC Level-Up Attempt Guarantee** (≥1 banked level) holds even if both null.
- **B1 (INFRA-1)** runs EARLY: it fixes the carve-out so A1/A2's honest nulls are not quarantined like `.435` A1 was.
- **A4/A5/A6** read A1/A2's `chosen_submitted_config`; **E** aggregates everything.

---

## Sprint-forcing-function compliance (CLAUDE.md, through 2026-06-30)

| Requirement | This milestone |
|---|---|
| Majority ARC live-solving | A1–A6 (6 of 12 tasks) — the generation valid-tests + self-play + readiness + persist + integration |
| ≥1 level-up attempt that BANKS a new reproducible level | **A3** (64→65), plus A1/A2 L2 gates (`arc_levelup_guarantee_lint.py` ≥1) |
| Self-play EVERY milestone (train+checkpoint the verifier) | **A3** trains + checkpoints `models/arc_verifier_<game>.json` |
| 2 reserved infra | **B1** (carve-out hardening) + **B2** (submission-package readiness) |
| 1 per-board hardware | **C** (KV260) |
| 1 SOTA-ingestion | **D** (epistemic-MCTS + causal-probe + MATM efficiency → `.437`) |
| All experiments codex/gpt-5.5; planner+retro Claude Opus | ✓ |
| Frozen live generator = Qwen3.5-9B-MTP on the iGPU (port-8919 gemma-squat guard) | ✓ (A1/A2 construct the proposer on a FREE port + `/props`-verify Qwen) |

---

## Continuous self-learning (research-program.md requirement)

**A3** is the self-learning experiment: the standing `arc_loop_solve` loop banks a level AND
trains+checkpoints the learned verifier on the run's positive (steps-to-go) + negative (dead-end)
traces — the Phase-3 self-improving-verifier program, run every milestone. **A5** persists the
strongest `.436` generation primitive into reusable `arc_solver_kit` scaffolding so the LIVE agent
reuses it on hidden games (knowledge captured as reusable scaffolding, verified by the reproduction
gate, compounding across games).

---

## Hardware requirements

- **A1/A2 (live):** iGPU (Radeon 890M) for the frozen **Qwen3.5-9B-MTP** generator (NEVER the 3090s);
  CUDA (RTX 3090) for any offline training arm; the Kaggle path is CPU under a 12h/600-RPM cap —
  A1/A2 measure CPU candidate-scoring latency.
- **C (KV260):** SSH-reachability of the board (`ssh kria`), NEVER host SD-card device nodes.
- No new hardware required.

---

## What is RETIRED / must NOT be re-proposed

- The `.435` A1 **online-action-learning driver** — validly tested (`arms_non_degenerate=True`),
  genuine null → retired. Do NOT re-run verbatim.
- The `.434` static present-winner ranker (as-built); the `.434` single-frame structural-alignment L2 goal.
- **Macro-action / horizon-collapse** + **click-heatmap-as-generator** — empirical nulls (2026-06-23);
  the 0.04 wall is generation-GUIDANCE, not depth or coverage.
- Re-running the dead-code generation nulls VERBATIM — `.436` A1/A2 reopen them WITH the non-degeneracy
  gate inline + REAL candidate scoring / distinct pools (the structural difference).

## Deferred to `.437` (tracked by the capstone fallback + INFRA-1 ledger)

- P1 Go-Explore amortized prior reopen (frame-grid fix landed; rerun with positive archive cells).
- P4 hierarchical subgoal reopen (non-empty decomposition).
- A2 active-probe reopen (the `.435` probe never ran).
- D's `.437` frontier: epistemic-MCTS probe planner + factored causal probe bank + MATM similarity-keyed retrieval.
