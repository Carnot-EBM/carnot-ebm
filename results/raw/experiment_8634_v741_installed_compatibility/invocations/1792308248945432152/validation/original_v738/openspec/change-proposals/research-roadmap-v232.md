# Research Roadmap v232 — Milestone 2026.05.232

**Milestone:** 2026.05.232
**Title:** AUROC Closure Sprint: HALT k=19, HIVE Ensemble, FST Live PATH A/B, KV260 Yosys, Kinetic Langevin
**Date:** 2026-05-18
**Author:** outer-loop Claude (post-.231 planning)

---

## What Milestone .231 Proved

Milestone 2026.05.231 was the first fully-complete milestone (14/14 tasks) and the first real-data adversarial validation sprint. Key findings:

**Successes:**
- exp2365 FST live gen v11: **first validated FST artifact** (PATH C cached telemetry)
- exp2366 NSVIF+VERGE real-data: realistic pass_rate < 1.0 — IMPLAUSIBLE_PERFECT resolved
- exp2367 Eidoku+EBM-CoT real-data: realistic AUROC < 1.0 — IMPLAUSIBLE_PERFECT resolved
- exp2368 LaaB k=17: new Tier 0h logical consistency verifier implemented
- exp2369 SpilledEnergy k=18: new Tier 0i variant implemented
- exp2370 Multi-verifier comparison: ensemble AUROC measured vs HalluScan 0.88 gap documented
- exp2371 LagONN: deterministic Kuramoto constraint satisfaction vs Self-Adaptive Ising
- exp2372 **KV260 RTL lint FIXED** (lint_errors_count=0, `strong` keyword renamed): RTL path unblocked
- exp2373 NSVIF compliance domain: financial regulatory text extension done
- exp2374 KAN-CL hard domains: realistic < 100% forgetting — IMPLAUSIBLE_PERFECT resolved
- exp2375 FR-11 real-data CSL: cross_domain_retention_rate >= 0.60 — MANDATORY passed
- exp2376+2377 capstone+retro: synthesis complete

**Critical open items for .232:**
- FST via PATH C only — PATH A (llama_cpp live GGUF) and PATH B (transformers) never executed
- AUROC gap to HalluScan 0.88 remains open (best .231 real-data verifier AUROC unknown yet)
- KV260 Yosys synthesis never attempted (RTL now lint-clean — synthesis is next logical step)
- FR-11 continuous self-learning uses FST fast/slow on cached data, not online model updates
- Phase 1 ship gate (PyPI + HF mirror + docs + external reproducer) unchecked

---

## Architecture Snapshot

```
Carnot .232 State:
  Tier 0 verifiers (cheap, training-free):
    Tier 0b: SpilledEnergy k=15 (arXiv:2602.18671) — ICLR 2026
    Tier 0g: SemanticEnergyDetector — AUROC=0.685 real (exp2351)
    Tier 0h: LaaB logical consistency (arXiv:2605.03971) — implemented exp2368
    Tier 0i: SpilledEnergy k=18 — implemented exp2369
    Tier 0j: HALT latent probe [to implement exp2379]
    
  Tier 0 ensemble:
    exp2370: 3-verifier soft ensemble — AUROC to be improved in exp2380
    exp2380: HIVE-style 4-verifier ensemble [target > 0.88]

  Tier 1 verifiers (constraint-based):
    NSVIF Z3 extractor: realistic pass_rate < 1.0 on real data (exp2366)
    Eidoku CSP gate: realistic accuracy < 1.0 on real data (exp2367)
    FregeLogic Z3+neural hybrid [to benchmark exp2381]

  Tier 2 repair:
    VERGE MCS repair: realistic success < 1.0 on real data (exp2366)

  Samplers:
    CASAL (constrained Ising)
    Projected-Langevin (+0.333 vs CASAL, exp2355)
    Self-Adaptive Ising (12.75x speedup, exp2359)
    LagONN (Kuramoto oscillators, exp2371)
    Kinetic Langevin [to benchmark exp2385]

  Continual learning:
    KAN-CL hard domains: realistic < 100% forgetting (exp2374)
    FR-11 real-data CSL: cross_domain_retention_rate >= 0.60 (exp2375)
    FR-11 NSVIF online learning [to implement exp2383]

  Hardware:
    KV260 RTL: lint_errors_count=0 ACHIEVED (exp2372)
    KV260 Yosys synthesis: [to attempt exp2384]

  Pipeline:
    FST: validated via PATH C (exp2365)
    FST PATH A/B live GGUF [to attempt exp2382]
```

---

## 3 Biggest Gaps vs PRD Vision

### Gap 1: AUROC Still Below HalluScan 0.88 Baseline
Carnot's best real-data Tier 0 verifier AUROC is ~0.685 (Semantic Energy, exp2351). HalluScan (arXiv:2605.02443) establishes NLI Verification at 0.88 as the competitive ceiling; HIVE (arXiv:2604.26139) shows 0.9236 via multi-verifier ensemble. Carnot has three Tier 0 verifiers (0g, 0h, 0i) implemented from .231 — adding HALT (Tier 0j, arXiv:2601.14210, latent probe sub-ms) and running the HIVE-style 4-verifier soft ensemble should close the gap.

**Milestone .232 response:** exp2379 (HALT Tier 0j), exp2380 (HIVE 4-verifier ensemble), exp2381 (FregeLogic Z3+neural tiebreaker).

### Gap 2: FST Live PATH A/B Never Validated
exp2365 validated FST via PATH C (cached telemetry logprobs). This proves the pipeline works on real model outputs, but PATH A (llama_cpp live GGUF inference) and PATH B (transformers AutoModel) have never successfully executed. Headline claim requires at least PATH A or PATH B to complete once with live inference.

**Milestone .232 response:** exp2382 (FST live gen PATH A/B, GGUF cached Qwen3.6-35B or Gemma4-26B required as PRECONDITION, PATH C fallback retained for robustness).

### Gap 3: Phase 1 Ship Gate Unchecked
Phase 1 ship criteria (per CLAUDE.md): PyPI package, HF mirror, MCP server + CLI docs, external reproducer. These have never been formally audited against the checklist. The software ship gate determines whether Phase 1 is done.

**Milestone .232 response:** exp2388 (Phase 1 ship gate audit), plus KV260 Yosys synthesis (exp2384, unblocked by exp2372's lint fix) to advance hardware track.

---

## Phase Structure

### Phase 0: Admin (ungated)
- **exp2378** — Archive .231 + activate .232

### Phase 1: AUROC Closure — New Verifiers (all ungated)
- **exp2379** — HALT Tier 0j latent probe (arXiv:2601.14210)
- **exp2380** — HIVE 4-verifier soft ensemble (Tier 0g+0h+0i+0j)
- **exp2381** — FregeLogic Z3+neural hybrid (arXiv:2604.18328)

### Phase 2: FST Live + Self-Learning (ungated)
- **exp2382** — FST live gen PATH A/B (GGUF inference, PATH C fallback) — mandated SOTA GGUF
- **exp2383** — FR-11 NSVIF online learning [continuous_self_learning_task]

### Phase 3: Hardware + Samplers (ungated)
- **exp2384** — KV260 Yosys synthesis (lint-clean RTL from exp2372)
- **exp2385** — Kinetic Langevin vs CASAL (arXiv:2603.23397)
- **exp2386** — KAC RBF vs KAN-CL (arXiv:2503.21076)

### Phase 4: Theory + Ship Gate (ungated)
- **exp2387** — NSVIF SMT-LIB policy formalization (arXiv:2511.09008)
- **exp2388** — Phase 1 ship gate check
- **exp2389** — Paper-v6 real-data results table

### Phase 5: Synthesis (gated)
- **exp2390** — Capstone v232 (gated: exp2382.fst_live_validated AND exp2380.ensemble_auroc_improved)
- **exp2391** — Retro v232 (ungated)

---

## Dependency Graph

```
exp2378 (archive)
    ↓ (ungated after)
exp2379 HALT Tier 0j ─────┐
exp2381 FregeLogic ────────┼──→ exp2380 HIVE ensemble → exp2390 capstone
exp2382 FST PATH A/B ─────┘         ↑
exp2383 FR-11 NSVIF online ──────────────────────────→ exp2390 capstone
exp2384 KV260 Yosys (ungated)
exp2385 Kinetic Langevin (ungated)
exp2386 KAC RBF (ungated)
exp2387 NSVIF SMT-LIB (ungated)
exp2388 Phase 1 ship gate (ungated)
exp2389 paper-v6 table (ungated)
exp2391 retro (always runs)
```

---

## Hardware Requirements

| Task | Hardware | Precondition |
|------|----------|--------------|
| exp2379 HALT | CPU only | sklearn / sklearn importable |
| exp2380 HIVE | CPU only | >= 2 Tier 0 verifiers importable |
| exp2381 FregeLogic | CPU only | z3-solver, sklearn |
| exp2382 FST PATH A/B | GPU preferred (PATH A/B); CPU fallback PATH C | ls ~/.cache/huggingface/hub/*Qwen3.6-35B* |
| exp2383 NSVIF online | CPU only | nsvif_extractor importable; exp2366 artifact |
| exp2384 KV260 Yosys | CPU (Yosys binary) | command -v yosys; hardware/kv260/ising_inertia_n8_sparse_v2.v |
| exp2385 Kinetic Langevin | CPU only | numpy, scipy |
| exp2386 KAC RBF | CPU only | numpy, sklearn |
| exp2387 NSVIF SMT-LIB | CPU only | nsvif_extractor, z3 |
| exp2388 Ship gate | CPU only | pip, git, huggingface-cli |
| exp2389 Paper-v6 | CPU only | .231 artifacts |
| exp2390 Capstone | CPU only | .232 experiment artifacts |
| exp2391 Retro | CPU only | conductor-log.md |

---

## Decentralization Check (Rules 1–7)

1. **Local-first**: all new verifiers (HALT, HIVE, FregeLogic) use CPU-only computation on local data. FST PATH A/B uses cached local GGUF. ✓
2. **Closed-model optional**: exp2382 uses local GGUF (Qwen3.6-35B, Gemma4-26B), never closed API. ✓
3. **Distribution mirroring**: ship gate check (exp2388) verifies HF mirror status. ✓
4. **Multiple surfaces**: FST PATH A/B + CLI + MCP all on same footing. ✓
5. **Hardware portability**: KV260 Yosys synthesis is hardware-sovereignty infrastructure. ✓
6. **Data minimization**: no closed-model calls in any task. ✓
7. **No vendor core deps**: all new verifiers extend python/carnot/verify/ via abstract protocol. ✓

---

## Exclusion Manifest Cross-Check

Read ops/exclusion_manifest.yaml before planning. Checked retired experiment IDs and scope patterns:
- GRPO/VPRM v1-v14: NOT proposed ✓
- WOPR puzzle cartridges: NOT proposed ✓
- HardNet++/DSP repair stack: NOT proposed ✓
- THRML scaling sweep: NOT proposed ✓
- SpecAnn: NOT proposed ✓
- exp2091 (gemini CLI CSL Grammar Updates): NOT proposed ✓
- iCE40 PIMI: NOT proposed ✓
- HalluSAEGeometricProbe: NOT proposed ✓
- Discriminative JEPA OOD (exp887 lineage): NOT proposed ✓

No scope-matched retired experiments in this roadmap. All 14 tasks are new scope.

---

## Failed-Experiment Rerun Compliance

| Task | Prior failure(s) | Addressed by |
|------|-----------------|--------------|
| exp2379 HALT | None — first implementation | New scope |
| exp2380 HIVE ensemble | exp2370 (3-verifier baseline) | exp2380 adds Tier 0j HALT → 4-verifier ensemble; different scope (target > 0.88 vs baseline measurement) |
| exp2381 FregeLogic | None — first implementation | New scope |
| exp2382 FST PATH A/B | exp2365 (PATH C only) | exp2382 targets PATH A/B; exp2365 proved PATH C works; different success criterion (live_inference_completed==true) |
| exp2383 NSVIF online | exp2375 (FR-11 FST fast/slow, different mechanism) | exp2383 uses NSVIF weight updates from real violations, not FST fast/slow; different algorithm |
| exp2384 KV260 Yosys | exp2360 (25 lint errors), exp2372 (lint fixed) | exp2372 achieved lint_errors_count=0; exp2384 is the first Yosys synthesis attempt on clean RTL |
| exp2385 Kinetic Langevin | exp2355 (Projected-Langevin baseline) | Kinetic Langevin is a different algorithm (underdamped); exp2355 is the comparison baseline |
| exp2386 KAC RBF | exp2374 (KAN-CL, different algorithm) | KAC uses RBF instead of B-splines; different base architecture |
| exp2387 NSVIF SMT-LIB | exp2373 (compliance domain) | SMT-LIB policy formalization is a different pipeline extension (arXiv:2511.09008 vs compliance domain) |
| exp2388 Ship gate | None — first formal check | New scope |
| exp2389 Paper-v6 table | None — first compilation | New scope |
| exp2390 Capstone | exp2376 (.231 capstone) | Different milestone synthesis; .232 capstone covers different experiments |
| exp2391 Retro | exp2377 (.231 retro) | Different milestone retro |
