# Research Roadmap v245: Phase 4 Option B + arXiv Submission + Ensemble v7b + Hardware Flash + JEPA Real Evaluation

**Milestone:** 2026.05.245
**Previous Milestone:** 2026.05.244
**Date:** 2026-05-19
**Status:** PROPOSED
**ID allocation:** Milestone .244 used exp2530–exp2542; .245 starts at **exp2543**.

---

## What Milestone .244 Proved

Milestone .244 had a **severe execution-layer gap**: 5 of 13 planned tasks (exp2530–exp2534,
the first five in the queue) produced no artifacts. The tasks that did complete (exp2535–exp2542,
except exp2538 which was honestly blocked) were all later in the roadmap.

**Three genuine successes from .244:**

1. **arXiv LaTeX compile fixed** (exp2536): tectonic compiled docs/arxiv-paper/main.tex cleanly;
   abstract trimmed 522 to 205 words. The three-milestone LaTeX/abstract blocker is resolved.
2. **GateMate bitstream generated** (exp2537): rtl/gatemate_ising_n16.cfg (16 392 bytes,
   514.67 MHz max F, 33/40 960 CPE_LT). The yosys to nextpnr LUT mismatch turned out to be
   a false premise. Physical flash via DirtyJTAG is the only remaining step.
3. **FR-11 Tier 3 JEPA fast-path integrated** (exp2539): JEPAFastPathPredictor wired into
   VerifyRepairPipeline with 14 passing tests. Caveat: fast_path_rate=1.0 on synthetic corpus.

**What .244 did NOT accomplish:**

- exp2531 (IsingVerifier.energy() implementation) — no artifact; stub class remains
- exp2532 (Phase 4 ARM-EBM v4) — gated on exp2531; never ran
- exp2533 (Ensemble v7b, Group D) — no artifact; 0.9607 regression unfixed for third consecutive milestone
- exp2534 (Adaptive conformal v2) — gated on exp2533; never ran

**Root-cause hypothesis for execution-layer gap:** All five failed tasks were at the front of
the milestone with relatively high complexity for codex in the allotted turns. The archive task
(exp2530) may have found research-complete.yaml already partially updated. IsingVerifier (exp2531)
required writing multi-method Python code. Without exp2531 artifacts, exp2532 was gate-blocked.
exp2533 (no gate) may have failed because calibration code location was hard to locate in 45 turns.

**Fix for .245:** simpler archive task with explicit preconditions, minimal IsingVerifier
specification with code spelled out verbatim, and reduced task complexity at the front of queue.

---

## Three Biggest Gaps Entering .245

| # | Gap | .245 Fix |
|---|-----|---------|
| 1 | Phase 4 blocked_precondition x 4 milestones | Take Option (b): expand §4 with honest negative, redefine gate-3 as phase4_resolved |
| 2 | Ensemble v7b deferred x 3 milestones | Front-load at exp2546 with explicit calibration code pointers |
| 3 | Execution-layer gap (5/13 tasks failed) | Simplify front tasks; spell out IsingVerifier code verbatim |

---

## Architecture: .245 Phase Structure

```
Phase 0 (Admin):
  exp2543 — Archive .244 + diagnose execution gap + activate .245

Phase 1 (arXiv Unblock):
  exp2544 — Phase 4 Option B: expand §4 with honest negative subsection
  exp2545 — IsingVerifier.energy(text) implementation (verifier fix, not Phase 4)

Phase 2 (Ensemble Expansion — deferred 3 milestones):
  exp2546 — Ensemble v7b: Tier 0r Group D (fix 0.9607 regression)
  exp2547 — Adaptive conformal v2 + ACSE [gated on exp2546.ensemble_v7b_auroc >= 0.975]

Phase 3 (Verifier Real-Corpus Validation):
  exp2548 — Tier 0r/0s/0u real-corpus AUROC validation (all synthetic-only currently)
  exp2549 — Tier 0v HalluField verifier prototype (arXiv:2509.10753)

Phase 4 (Self-Learning):
  exp2550 — JEPA fast-path real-corpus evaluation [continuous_self_learning_task: true]

Phase 5 (Hardware — MANDATORY per CLAUDE.md):
  exp2551 — GateMate physical flash + KV260 SD card flash (combined hardware task)

Phase 6 (Paper + arXiv):
  exp2552 — Paper-v6 final write-through (ensemble v7b results, real-corpus, new cites)
  exp2553 — arXiv final submission package v3

Phase 7 (Synthesis):
  exp2554 — Capstone v245 (NO HARD GATE)
  exp2555 — Retro v245
```

**Dependency graph:**
- exp2543 unlocks .245 milestone state
- exp2544 provides §4 honest negative; exp2553 reads this
- exp2545 provides working IsingVerifier; independent of Phase 4 path
- exp2546 provides ensemble v7b auroc; exp2547 is gated on exp2546.ensemble_v7b_auroc >= 0.975
- exp2547, exp2548, exp2549, exp2550 all feed exp2552 (paper write-through)
- exp2552 feeds exp2553 (arXiv package)
- exp2553 feeds exp2554 (capstone)
- exp2554 feeds exp2555 (retro)
- exp2551 is independent of all above (hardware; no upstream gates)

---

## Operator Decision: Phase 4 Option B

The .244 capstone and retro both recommend **Option (b)**:

> Accept Phase 4 as empirically unsupported. Expand paper §4 with a full honest
> negative-result subsection citing exp2486, exp2508, exp2519, and the four-milestone
> history. Submit paper-v6 on Phase 1 (verify-repair) + ensemble AUROC results alone.

**Rationale:**
- Option (a) (try again with IsingVerifier) has declining prior probability after 4
  consecutive attempts. IsingVerifier.energy() measures arithmetic consistency, which
  measures something different from ARM-EBM token-level free energy.
- Option (b) allows arXiv submission this milestone. Gate-3 is redefined as
  phase4_resolved = (phase4_validated_any OR phase4_honest_negative_documented).
- Option (c) loses the theoretical EBM framing that differentiates Carnot.

---

## New arXiv Findings from .245 Planning Sweep

| arXiv ID | Title (abbreviated) | Carnot Relevance |
|---------|---------------------|-----------------|
| arXiv:2509.10753 | HalluField — field-theoretic hallucination detection | Tier 0v prototype |
| arXiv:2512.18730 | RL-tuned LMs as EBMs (RLVR theoretical lens) | paper-v6 §3 theoretical cite |
| arXiv:2604.16217 | Conformal prediction via internal representations | adaptive conformal enhancement |
| arXiv:2604.17109 | Fully parallel Ising machine on FPGA | KV260 hardware precedent |
| arXiv:2605.09515 | Game-theoretic FEP in LLM attention heads | Phase 4 negative §4 grounding |

All five added to research-references.md.

---

## Hardware State Entering .245

| Board | State | .245 Task |
|-------|-------|-----------|
| KV260 | hwh generated; /dev/sda /dev/sdb detected; PYNQ URL unreachable | exp2551: manual download path; attempt flash |
| GateMate A1-EVB-2M | Bitstream ready (16 392 bytes); DirtyJTAG USB 1209:c0ca onboard | exp2551: openFPGALoader + smoke test |
| PolarFire SoC | TERMINAL (exp2501) | none (opportunistic) |

---

## Agent Routing

| Task | Agent | Justification |
|------|-------|--------------|
| exp2543–exp2550, exp2552–exp2553, exp2555 | codex/gpt-5.5 | mechanical, well-defined patterns |
| exp2551 hardware flash | claude/opus | FPGA hardware multi-step interaction requires Claude |
| exp2554 capstone | claude/opus | cross-artifact synthesis under ambiguity |

Routing: 11 codex (84.6%), 2 claude/opus (15.4%) — within the CLAUDE.md 2/13 guideline.

---

## Continuous Self-Learning Mandate

exp2550 (JEPA fast-path real-corpus evaluation) is the continuous_self_learning_task for this
milestone. Synthetic fast_path_rate=1.0 in exp2539 is useless. This task tests the predictor
against real logprob-labeled responses to find a discrimination threshold that enables meaningful
fast-path filtering (target fast_path_rate between 0.3 and 0.8 on real corpus).

---

## Acceptance Summary

1. phase4_honest_negative_documented == True (exp2544)
2. ising_verifier_text_energy_implemented == True (exp2545)
3. ensemble_v7b_auroc >= 0.970 (exp2546)
4. At least one hardware terminal state advanced (exp2551)
5. arxiv_ready == True (exp2553)
6. Capstone runs (exp2554 — NO HARD GATE)
