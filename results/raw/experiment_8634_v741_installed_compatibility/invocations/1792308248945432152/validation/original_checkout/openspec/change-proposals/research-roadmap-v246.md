# Research Roadmap v246: Post-arXiv Paper Integrity + Hardware Terminal + Ensemble Expansion v8

**Milestone:** 2026.05.246
**Previous Milestone:** 2026.05.245
**Date:** 2026-05-20
**Status:** PROPOSED
**ID allocation:** Milestone .245 used exp2543–exp2555; .246 starts at **exp2556**.

---

## What Milestone .245 Proved

Milestone .245 was a landmark: **arxiv_ready=True for the first time** across the
.240–.245 research track. Nine of eleven capstone-input experiments reached terminal verdicts.

**Five wins from .245:**

1. **arxiv_ready=True** (exp2553): All 4 gates satisfied — Phase 1 ship (PyPI), paper integrity
   audit, Phase 4 resolved via Option B (honest negative), AUROC adversarially verified. Operator
   recommendation: submit_now.
2. **Phase 4 Option B executed** (exp2544): §4.4 honest negative subsection landed in main.tex,
   citing exp2486/2508/2519/2532 across four milestones. Gate-3 redefined as
   `phase4_resolved = validated_any OR honest_negative_documented` — satisfied.
3. **Ensemble v7b AUROC 0.9857** (exp2546): Three-milestone 0.9607 regression resolved by routing
   Tier0r to Group D (proof-path). 5 seeds, std=0.0175. New headline AUROC.
4. **JEPA discrimination improved** (exp2550): JEPAFastPathPredictor fast_path_rate in [0.30, 0.80]
   on balanced corpus — useful discrimination from the synthetic-only 1.0 baseline.
5. **IsingVerifier implemented** (exp2545): IsingVerifier().energy(text) regex arithmetic checker
   fully implemented and test-passing.

**What .245 did NOT accomplish (execution gaps):**

- exp2549 (Tier0v HalluField): blocked_carnot_import_failed — PYTHONPATH issue; hedge-energy
  never ran.
- exp2550 (JEPA): fast_path_rate improvement achieved but acceptance_gate_passed=true with
  unsafe_fast_path_rate=0 — GATE_PASSED_WITHOUT_DATA adversarial flag.

**Critical finding from real-corpus validation (exp2548):**

Real-corpus validation revealed a severe synthetic-vs-real AUROC gap:

| Verifier | Synthetic AUROC (claimed) | Real-Corpus AUROC (FoVer, n=6548) |
|----------|--------------------------|----------------------------------|
| Tier 0r (Curry-Howard) | 0.8256 | **0.9414** — solid |
| Tier 0s (HalluGuard NTK) | 1.0 | **0.3758** — severe gap |
| Tier 0u (Logical Consistency) | 0.96 | **0.5360** — severe gap |

Tier0s AUROC=1.0 and Tier0u AUROC=0.96 appear in the paper as currently-reported verifier
numbers. **These must be corrected before the operator submits to arXiv.** Publishing inflated
synthetic AUROCs would damage scientific credibility.

---

## Three Biggest Gaps Entering .246

| # | Gap | .246 Fix |
|---|-----|---------|
| 1 | tier0s/tier0u inflated synthetic AUROC in paper-v6 | exp2557: errata pass — update §5 with real-corpus values; remove tier0s/tier0u from headline claims |
| 2 | GateMate strtol parse error; KV260 SD media absent | exp2559: CC1-toolchain regeneration OR openFPGALoader HEAD build; exp2560: KV260 operator docs |
| 3 | Verifier expansion blocked (Tier0v, Tier0t not yet in ensemble) | exp2561/2562/2563 → exp2563 Ensemble v8 |

---

## Architecture: .246 Phase Structure

```
Phase 0 (Admin):
  exp2556 — Archive .245 + activate .246

Phase 1 (Paper Integrity — highest priority):
  exp2557 — Paper-v6 errata: retire tier0s/tier0u synthetic AUROCs, report real-corpus values
  exp2558 — arXiv Final Package v4 (after errata; operator submits) [gated on exp2557]

Phase 2 (Hardware — MANDATORY per CLAUDE.md):
  exp2559 — GateMate .cfg format fix (CC1 toolchain or openFPGALoader HEAD) [claude+opus]
  exp2560 — KV260 operator flash docs update [codex]

Phase 3 (Verifier Expansion):
  exp2561 — Tier 0t Dynamical System prototype (arXiv:2605.05134)
  exp2562 — Tier 0v HalluField retry + Tier 0w DiffuTruth (arXiv:2602.11364) [combined]
  exp2563 — Ensemble v8: integrate viable new verifiers [gated on ≥1 new verifier AUROC>0.60]

Phase 4 (Calibration + Self-Learning):
  exp2564 — Feasibility-Aware Conformal Calibration (arXiv:2603.22966 MRL)
  exp2565 — JEPA real FoVer training (n=6548; continuous_self_learning_task)

Phase 5 (External Benchmark):
  exp2566 — HalluScan benchmark evaluation (arXiv:2605.02443; peer comparison)

Phase 6 (Synthesis):
  exp2567 — Capstone v246 [claude+opus; NO HARD GATE]
  exp2568 — Retro v246 [codex]
```

**Dependency graph:**

```
exp2556 (archive)
  → exp2557 (paper errata) → exp2558 (arxiv package v4)
  → exp2559 (gatemate fix) [independent]
  → exp2560 (kv260 docs) [independent]
  → exp2561 (tier0t) → exp2563 (ensemble v8) [gated ≥1 new verifier viable]
  → exp2562 (tier0v+tier0w) → exp2563 (ensemble v8)
  → exp2564 (feasibility-aware conformal) [independent]
  → exp2565 (jepa training) [independent]
  → exp2566 (halluscan eval) [independent]
  → exp2567 (capstone) [reads all above]
  → exp2568 (retro)
```

---

## Hardware Requirements

| Board | Current State | .246 Target |
|-------|--------------|-------------|
| Cologne Chip GateMate A1-EVB-2M | openFPGALoader strtol parse error on .cfg; JTAG chain verified | Regenerate .cfg via CC1 toolchain OR build openFPGALoader HEAD; flash bitstream; smoke test |
| Xilinx KV260 | SD media absent; PYNQ URL unreachable | Updated operator docs with confirmed-reachable URLs; device write test |
| Microchip PolarFire SoC | **TERMINAL** (exp2501) | Optional/opportunistic only |

---

## Decentralization Compliance (Rules 1–7)

| Rule | Compliance |
|------|-----------|
| 1. Local-first open models | All experiments use FoVer corpus or local logprob features; no closed-weight required |
| 2. Closed models optional | No closed-weight dependencies in new verifiers |
| 3. Distribution mirroring | No new artifacts to publish; paper arXiv submission is operator-side |
| 4. Multiple integration surfaces | No changes to surface count |
| 5. Hardware portability | GateMate + KV260 tracks advance sovereignty hardware story |
| 6. Data minimization | No external LLM calls in scope |
| 7. No vendor abstractions in core | All new code stays in `carnot/verify/` or `carnot/pipeline/` |

---

## Exclusion Manifest Cross-Check

Retired experiment IDs in `ops/exclusion_manifest.yaml`: exp2091, exp260, exp308, exp309,
exp346, exp380-383, exp410, exp425, exp491, exp527, exp603, exp627.

None of the .246 task scopes match any retired experiment ID by name or deliverable shape.
No retired experiment IDs appear in any task's `requires:` chain.

---

## Failed-Experiment Rerun Compliance

| New Task | Prior Failure | Scope Match | How Addressed |
|----------|-------------|-------------|---------------|
| exp2562 (Tier0v retry) | exp2549 (blocked_carnot_import_failed) | MATCH: same verifier, same technique | Fixed by PYTHONPATH-aware precondition; sys.path.insert before carnot import |
| exp2558 (arXiv pkg v4) | exp2553 (complete: arxiv_ready=True) | Non-failure; routine update | Different milestone, different errata incorporated |
| All others | No prior failures with matching scope | — | — |

---

## Agent Routing

| Agent | Count | Fraction | Tasks |
|-------|-------|---------|-------|
| codex / gpt-5.5 | 11 | 84.6% | exp2556–2558, 2560–2566, 2568 |
| claude + opus | 2 | 15.4% | exp2559 (hardware multi-step), exp2567 (capstone) |

**requires_claude: true justification:**
- exp2559: Hardware multi-step coordination (openFPGALoader error diagnosis + CC1 toolchain install
  + alternative bitstream generation + physical flash attempt); meets all 3 positive criteria:
  (1) hardware tasks have confirmed codex failure history on this board; (2) multi-file coordination
  across rtl/, tools/, operator artifact; (3) open-ended hardware diagnosis under uncertainty.
- exp2567: Multi-artifact synthesis across 12 experiments + cross-phase reasoning; meets all 3 criteria.
