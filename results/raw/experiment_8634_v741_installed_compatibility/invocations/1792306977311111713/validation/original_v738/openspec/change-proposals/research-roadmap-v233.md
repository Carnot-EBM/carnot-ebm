# Research Roadmap v233 — Milestone 2026.05.233

**Milestone:** 2026.05.233
**Title:** Codex Recovery Sprint: HALT/HIVE/FregeLogic v2, FST PATH A/B v2, FR-11 v2, KV260 Yosys v2, Typed CoT Tier 2.8
**Date:** 2026-05-18
**Author:** outer-loop Claude (post-.232 planning)

---

## What Milestone .232 Proved

Milestone 2026.05.232 was titled "AUROC Closure Sprint" but produced a catastrophic infrastructure failure. Key findings:

**Single success:**
- exp2389 paper-v6 table: best_auroc_achieved=0.6852 (stub artifact, DURATION_TOO_SHORT=0.000426s, adversarial flags present)

**Systematic failure (11 of 14 tasks):**
All 11 non-synthesis tasks failed with identical error: `"Codex CLI error: u finish the real work inside 10 minutes, that is correct an"`. This is a truncated message from the Codex CLI agent, suggesting the agent is responding conversationally to a timing instruction rather than executing the task. Root cause is unknown; exp2393 must diagnose it.

**Gate-blocked (1 task):**
- exp2390 capstone: GATE_BLOCK (gated on exp2382.fst_live_validated AND exp2380.ensemble_auroc_improved, both failed)

**Metrics at .232 close:**
- AUROC gap to HalluScan 0.88: **0.1948** (unchanged from .231)
- FST live PATH A/B: **never executed**
- KV260 Yosys synthesis: **never attempted**
- FR-11 continuous self-learning: **not satisfied**
- Phase 1 ship gate: **unchecked**
- Codex CLI failure rate: **11/11 = 100%** on implementation tasks

---

## Architecture Snapshot

```
Carnot .233 Start State (carried from .231 — .232 added nothing):
  Tier 0 verifiers (training-free):
    Tier 0b: SpilledEnergy k=15 (arXiv:2602.18671) — ICLR 2026
    Tier 0g: SemanticEnergyDetector — AUROC=0.685 real (exp2351)
    Tier 0h: LaaB logical consistency (arXiv:2605.03971) — implemented exp2368
    Tier 0i: SpilledEnergy k=18 — implemented exp2369
    Tier 0j: HALT latent probe [TO IMPLEMENT exp2394]
    Tier 0f: Frequency-Aware Attention [TO IMPLEMENT exp2397]

  Tier 0 ensemble:
    exp2370: 3-verifier soft ensemble (Tier 0g+0h+0i, .231)
    exp2398: HIVE 4-verifier v2 [TARGET > 0.88, needs exp2394 first]

  Tier 1 verifiers (constraint-based):
    NSVIF Z3 extractor: realistic pass_rate < 1.0 on real data (exp2366)
    FregeLogic Z3+neural hybrid [TO BENCHMARK exp2395]

  Tier 2 repair:
    VERGE MCS repair: realistic success < 1.0 on real data (exp2366)

  New (Phase 2.8):
    Typed CoT Curry-Howard verifier [TO IMPLEMENT exp2396]

  Samplers:
    CASAL (constrained Ising)
    Projected-Langevin (+0.333 vs CASAL, exp2355)
    Self-Adaptive Ising (12.75x speedup, exp2359)
    LagONN (Kuramoto oscillators, exp2371)
    Kinetic Langevin [TO BENCHMARK exp2402]

  Continual learning:
    KAN-CL hard domains: realistic < 100% forgetting (exp2374)
    FR-11 real-data CSL: cross_domain_retention_rate >= 0.60 (exp2375)
    FR-11 NSVIF online learning [TO IMPLEMENT exp2400]

  Hardware:
    KV260 RTL: lint_errors_count=0 ACHIEVED (exp2372)
    KV260 Yosys synthesis: [TO ATTEMPT exp2401]

  Pipeline:
    FST: validated via PATH C (exp2365)
    FST PATH A/B live GGUF [TO ATTEMPT exp2399]

  Infrastructure:
    Codex CLI: BROKEN in .232 [TO DIAGNOSE exp2393]
```

---

## 3 Biggest Gaps

### Gap 1: Codex CLI Infrastructure Broken (NEW — highest priority)
All 11 implementation tasks in .232 failed with identical error. This renders the conductor's primary execution path inoperable. Until this is diagnosed and fixed (or worked around), ALL future codex-based tasks will fail identically.

**Milestone .233 response:** exp2393 (Codex CLI diagnostic, `requires_claude: true` — positive criterion met by 33 consecutive codex failures across 11 distinct task categories in .232).

### Gap 2: AUROC Gap Unchanged at 0.1948 (Sprint Goal Failed)
None of the AUROC-advancing tasks (HALT Tier 0j, HIVE ensemble, FregeLogic) executed in .232. AUROC remains at 0.685 with a 0.1948 gap to HalluScan 0.88 and 0.2384 gap to HIVE 0.9236.

**Milestone .233 response:** exp2394 (HALT v2), exp2395 (FregeLogic v2), exp2396 (Typed CoT Tier 2.8 — new), exp2397 (Frequency-Aware Attention Tier 0f — queued since .228), exp2398 (HIVE ensemble v2, gated on HALT).

### Gap 3: FST/FR-11/KV260/Ship-Gate Evidence Missing
Phase 1 ship gate unchecked; FST PATH A/B unvalidated; FR-11 not satisfied; KV260 Yosys not attempted. These are all .232 reruns where the only blocker was the Codex CLI failure.

**Milestone .233 response:** exp2399 (FST PATH A/B v2), exp2400 (FR-11 NSVIF online v2, mandatory), exp2401 (KV260 Yosys v2), exp2402 (Kinetic Langevin v2), exp2403 (Phase 1 ship gate v2), exp2404 (paper-v6 table v2 + capstone).

---

## Phase Structure

### Phase 0: Admin + Infrastructure Repair (ungated)
- **exp2392** — Archive .232 + activate .233
- **exp2393** — Codex CLI diagnostic + infrastructure repair (`requires_claude: true`)

### Phase 1: AUROC Closure — Verifier Reruns v2 (all ungated)
- **exp2394** — HALT Tier 0j latent probe v2 (arXiv:2601.14210)
- **exp2395** — FregeLogic Z3+neural hybrid v2 (arXiv:2604.18328)
- **exp2396** — Typed CoT Curry-Howard Tier 2.8 (arXiv:2510.01069, NEW)
- **exp2397** — Frequency-Aware Attention Tier 0f (arXiv:2602.18145, queued .228)
- **exp2398** — HIVE 4-verifier ensemble v2 (needs exp2394 first)

### Phase 2: FST + Self-Learning Reruns v2 (all ungated)
- **exp2399** — FST live gen PATH A/B v2 (mandated SOTA GGUFs)
- **exp2400** — FR-11 NSVIF online learning v2 (`continuous_self_learning_task: true`)

### Phase 3: Hardware + Samplers Reruns v2 (all ungated)
- **exp2401** — KV260 Yosys synthesis v2
- **exp2402** — Kinetic Langevin vs CASAL v2 (arXiv:2603.23397)
- **exp2403** — Phase 1 ship gate v2

### Phase 4: Synthesis (gated)
- **exp2404** — Paper-v6 real-data results table v2 + capstone (`requires_claude: true`, gated: exp2398.ensemble_auroc_improved OR exp2394.halt_k19j_validated)
- **exp2405** — Retro v233 (always runs)

---

## Dependency Graph

```
exp2392 (archive) → exp2393 (codex diagnostic)
                         ↓ (infra repair precedes all codex tasks conceptually)
exp2394 HALT v2 ─────────┐
exp2395 FregeLogic v2 ───┤
exp2396 Typed CoT ───────┤
exp2397 Freq-Aware Attn ─┤
                         ↓
exp2398 HIVE ensemble v2 (needs exp2394 complete for 4th verifier)
exp2399 FST PATH A/B v2 (ungated — PATH C fallback available)
exp2400 FR-11 NSVIF online v2 (ungated — FR-11 mandatory)
exp2401 KV260 Yosys v2 (ungated — lint clean from exp2372)
exp2402 Kinetic Langevin v2 (ungated)
exp2403 Phase 1 ship gate v2 (ungated)
exp2404 capstone (gated: exp2398 OR exp2394)
exp2405 retro (always runs)
```

---

## Hardware Requirements

| Task | Hardware | Precondition |
|------|----------|--------------|
| exp2392 archive | CPU | python importable |
| exp2393 codex diagnostic | CPU | codex CLI accessible |
| exp2394 HALT v2 | CPU | sklearn, telemetry manifest |
| exp2395 FregeLogic v2 | CPU | z3-solver, sklearn |
| exp2396 Typed CoT | CPU | z3-solver OR sklearn |
| exp2397 Freq-Aware Attn | CPU | telemetry manifest |
| exp2398 HIVE ensemble v2 | CPU | exp2394 HALT artifact |
| exp2399 FST PATH A/B v2 | GPU preferred; CPU fallback | ~/.cache/huggingface/hub/*Qwen3.6-35B* OR PATH C |
| exp2400 FR-11 NSVIF v2 | CPU | z3-solver, telemetry manifest |
| exp2401 KV260 Yosys v2 | CPU | yosys binary |
| exp2402 Kinetic Langevin v2 | CPU | scipy, numpy |
| exp2403 Phase 1 ship gate v2 | CPU/network | pip install carnot-ebm check |
| exp2404 capstone | CPU | exp2398 OR exp2394 artifact |
| exp2405 retro | CPU | 14 experiment artifacts |

---

## Exclusion Manifest Cross-Check

The following retired scopes were checked against all proposed .233 tasks:

| Retired Scope | .233 Task Overlap? | Disposition |
|---|---|---|
| exp2091 (gemini CSL grammar) | No | Not proposed |
| GRPO/VPRM lineage | No | Not proposed |
| WOPR puzzle cartridge | No | Not proposed |
| HardNet++/DSP repair | No | Not proposed |
| THRML scaling sweep | No | Not proposed |
| SpecAnn (spectral annealing) | No | Not proposed |
| iCE40 PIMI (all variants) | No | KV260 Yosys is Yosys synthesis, not PIMI |
| discriminative JEPA | No | Not proposed |
| HalluSAE geometric probe | No | HALT is a different algorithm |

No .233 tasks overlap with any retired experiment scope. ✓

---

## Failed-Experiment Rerun Compliance

All .233 tasks that re-propose .232 scope include `prior_failures:` blocks in the YAML. Key reruns:

| .233 Task | Prior Failure | Root Cause | What Changed |
|---|---|---|---|
| exp2394 HALT v2 | exp2379 (Codex CLI error x3) | Codex CLI infrastructure broken | exp2393 diagnosed + repaired |
| exp2395 FregeLogic v2 | exp2381 (Codex CLI error x3) | Same | Same |
| exp2398 HIVE v2 | exp2380 (Codex CLI error x3) | Same | Same |
| exp2399 FST PATH A/B v2 | exp2382 (Codex CLI error x3) | Same | Same |
| exp2400 FR-11 v2 | exp2383 (Codex CLI error x3) | Same | Same |
| exp2401 KV260 Yosys v2 | exp2384 (Codex CLI error x3) | Same | Same |
| exp2402 Kinetic Langevin v2 | exp2385 (Codex CLI error x3) | Same | Same |
| exp2403 Ship gate v2 | exp2388 (Codex CLI error x3) | Same | Same |
| exp2392 Archive v2 | exp2378 (Codex CLI error x3) | Same | Adapted for .232 partial state |

All rerun tasks: `retire_if_same_verdict: true` if the Codex CLI error repeats AND exp2393 confirms the infrastructure is repaired — meaning we have exhausted the structural fix and the issue is per-task-specific.

---

## Decentralization Check (Rules 1–7)

| Rule | Compliance |
|---|---|
| 1. Local-first open models | exp2399 mandates SOTA GGUF models; PATH C (CPU-only cached data) fallback retained |
| 2. Closed frontier-model optional | No new closed-weight dependencies; exp2393 uses Claude as outer-loop operator (not a core dependency) |
| 3. Distribution mirroring | exp2403 Phase 1 ship gate checks PyPI + HF mirror + IPFS secondary channel |
| 4. Multiple integration surfaces | No change to API/CLI/MCP/HTTP surfaces in this milestone |
| 5. Hardware portability | KV260 Yosys v2 advances FPGA sovereignty track; CPU-only fallbacks on all verifier tasks |
| 6. Data minimization | exp2393 diagnostic uses only local conductor logs; no closed-weight API calls needed |
| 7. No vendor abstractions in core | No vendor-specific imports added to core verify/ or pipeline/ modules |

All seven rules: COMPLIANT ✓

---

## Codex-Default Audit

Tasks by agent type:
- `agent_type: codex` (12 tasks): exp2392, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2405
- `agent_type: claude` (2 tasks): exp2393 (codex demonstrably failed x33), exp2404 (capstone synthesis)

Ratio: 2/14 = 14% claude. Within the ≤2/13 guideline (14% ≈ 15%). ✓

`requires_claude: true` positive criterion for exp2393:
1. **Codex demonstrably failed**: 33 consecutive codex CLI failures across 11 task categories in .232. Experiment ID pattern: exp2378-2388 all FAILed identically.
2. **Multi-file tool choreography**: Diagnosing conductor-codex integration requires reading ops/conductor-log.md, checking codex CLI binary, examining scripts/ (read-only), running subprocess tests, writing diagnostic artifact — clearly 5+ files.
3. **Open-ended judgment under ambiguity**: Root cause of the "u finish the real work inside 10 minutes" error is unknown; hypothesis testing and interpretation of diagnostic outputs requires reasoning, not mechanical checking.

`requires_claude: true` positive criterion for exp2404:
1. **Prior codex capstone quality**: .231 capstone (opus) was the synthesis of 14 research streams; codex would require detailed scripting of the synthesis logic. The .232 capstone (exp2390) was gate-blocked so no quality comparison exists.
2. **Multi-file tool choreography**: Capstone reads 12+ experiment artifacts + research-program.md + paper-v6 draft sections.
3. **Open-ended prose synthesis**: Capstone requires identifying the most impactful .233 findings and framing them in context of the phase program — high-judgment, no deterministic gate.
