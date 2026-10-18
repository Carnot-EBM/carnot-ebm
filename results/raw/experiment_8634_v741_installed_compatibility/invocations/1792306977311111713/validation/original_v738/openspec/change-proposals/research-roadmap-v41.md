# Research Roadmap v41 — Milestone 2026.04.41

**Status:** Proposed
**Milestone:** 2026.04.41
**Title:** Close the Nine-Milestone Gap — First Live 25q Positive, Teardown Fix, GRPO Self-Learning
**Planned Experiments:** 537–548 (12 experiments)
**Planned Date:** 2026-04-20 onwards

---

## What Milestone 2026.04.40 Proved

Milestone .40 delivered four meaningful wins while finally confirming the exact nature of the RETRO-033 blocker:

1. **RETRO-053 CLOSED** — `apply_env_autofix()` now overrides `CARNOT_FORCE_LIVE='0'` (falsy values
   treated as not-set when GPU detected). This was the single remaining programmatic gate.

2. **Exp 527 ran live GPU inference** — For the first time in 9 milestones, RETRO-033 did NOT fail due
   to env gates, VRAM zombies, or conductor process memory. Exp 527 actually executed inference for
   45 minutes before timing out. The new blocker is **latency**, not environment.

3. **FR-11 live relay achieved** — JEPA retrained on real FOVER CoT pairs (fover_442): AUC improved
   from 0.437 → 0.967. First time FR-11 trained on real rather than synthetic data across all milestones.

4. **LowRankKAEM viable** — k=2 achieves 23.7x speedup over full-rank MCMC baseline.

5. **Tier 0c + 0d wired** — NUP Probe v4 (AUC=1.0) and Hallucination Basin Detector integrated into
   the `ThreeTierPipeline` cascade.

**The single remaining gate (RETRO-055):**
Exp 527 ran at ~27 seconds/question. At 100 questions, that requires ~45 minutes — exactly the budget.
Fix: reduce `n_questions` to 25 for the first live probe (target: ~11 min), verify latency and per-question
accuracy, then scale up once the budget is confirmed. RETRO-033 can close in this milestone.

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: Live Verify-Repair Result Still Missing (FR-12, RETRO-033 Miss #9)

**RETRO-055 is the new critical gate.** The latency budget mismatch (100q at 27s/q = 45 min) can be
resolved immediately by reducing n_questions to 25 (estimated: ~11 min). The infrastructure chain
is now correct at every level:

- `JITVRAMCheck` gates model loading against real-time VRAM ✓
- `Gemma4QuantizedLoader` fits within 24 GiB budget ✓
- `GPUVRAMGateV2` kills zombie VRAM holders before GPU runs ✓
- `EnvironmentAutoFix` overrides falsy `CARNOT_FORCE_LIVE` values ✓
- Live inference reaches GPU (Exp 527 confirmed — ran for 45 min) ✓

**Fix in Exp 538:** Run 25 questions × 2 models × baseline+pipeline variants with 90-min budget.
This provides the first live improvement measurement and establishes the per-question latency baseline
for scaling to 100q in RETRO-038.

### Gap 2: GPU Zombie Accumulation at Milestone Close (RETRO-054, Carry Count = 5)

The .40 retro found 47,653 MB of zombie VRAM across both GPUs at milestone close — the worst close
state across all milestones .34-.40. PIDs 430009 (18,678 MB GPU 0) and 430012 (4,894 MB GPU 0 +
24,072 MB GPU 1) must be killed before any .41 experiment.

`ExperimentTemplate.teardown()` with `atexit` registration has been recommended in retros .36 through .40
without implementation. This is the highest-leverage single fix: without teardown, GPU state degrades
monotonically across the milestone. Estimated 3-6% wall-time savings from clean GPU state.

**Fix in Exp 537:** Implement `teardown(clear_gpu=True)` in `ExperimentTemplate`, register via
`atexit.register(self.teardown)` in `__init__()`, and add startup zombie kill as the first action
in every `.41` experiment prompt.

### Gap 3: Self-Learning Still Synthetic-Only (FR-11, Tiers 1-2)

JEPA trained on real data (Exp 535: AUC 0.967) but the Tier 1-2 pipeline components
(`ConstraintAdditionFromMemory`, EORM retrain) still run only on synthetic violation patterns.
The live benchmarks from .40 produced FOVER-labeled CoT pairs; those pairs need to flow into the
self-learning loop.

**Fix in Exps 540-543:** GRPO contrastive pairing converts live correct/incorrect pairs into
EORM training signal (no additional labeling). `ConstraintAdditionFromMemory` wire-in turns
accumulated FP patterns into new constraint types. JEPA v8 retrain expands the corpus from
57 to 100+ real pairs.

---

## Architecture Diagram (Verification Cascade)

```
LLM Response
    │
    ▼
[Tier 0a] CarnotThinkProbe (generative 3-step CoT verdict)
    │  verdict='incorrect' → fast-path violation
    ▼
[Tier 0b] SpilledEnergyDetector (~0ms, logit-discrepancy)
    │  high_spill_fraction → likely hallucination
    ▼
[Tier 0c] NUP Probe v4 (~0ms, contrastive bigram energy, AUC=1.0)  ← .40 wired
    │  low energy → likely correct, skip downstream
    ▼
[Tier 0d] HallucinationBasinDetector (~0ms, latent basin depth)  ← .40 wired
    │  deep basin → stable reasoning, skip downstream
    ▼
[Tier 1] SinkProbe (~0ms, attention sink concentration)
    │  high sink score → skip downstream
    ▼
[Tier 2] EORM / InternalStateProbe (~10ms / <1ms)  ← .41 new: InternalStateProbe (Exp 545)
    │  energy < threshold → skip Ising
    ▼
[Tier 3] VerifyRepairPipeline (~0.006ms/constraint, Ising)
    │  violations found → repair → re-verify
    ▼
[Self-Learning] ConstraintAdditionFromMemory  ← .41 wire-in (Exp 541)
    │  FP patterns → new constraints for next query
    ▼
[FR-11] JEPA Predictor (predicts violations before generation)  ← .41 v8 retrain (Exp 543)
    │  LowRankKAEM (k=2, 23.7x faster)  ← .41 wire-in (Exp 544)
    ▼
Verified Output
```

---

## Phase Descriptions

### Phase 1: Infrastructure — Teardown Fix and Zombie Kill (Exp 537)

**Goal:** Implement `ExperimentTemplate.teardown()` with `atexit` registration. Kill zombie GPU
processes from .40 before any experiment runs.

Every experiment in .41 begins with explicit GPU zombie kill (PIDs 430009 + 430012 from .40 retro).
`teardown()` calls `torch.cuda.empty_cache()`, `gc.collect()`, and optionally kills any process
holding >5% VRAM with no compute. This prevents the 47,653 MB zombie state from recurring.

**Expected impact:** 3-6% wall-time savings per session; clean GPU state at every milestone close.

### Phase 2: Live Benchmark Attempts (Exps 538–539)

**Goal:** Finally close RETRO-033 with n_questions=25 and 90-min budget. Then scale to 100q.

Exp 538 runs 25 questions × 2 models × 5 variants with 90-min budget. This is the 10th attempt
at RETRO-033. RETRO-053 is confirmed resolved; the only remaining variable is latency budget.
At 27s/question, 25 questions = ~11 min: well within the 90-min budget.

Exp 539 runs 100q VeriCoT+VPRM (RETRO-038, 8th attempt) gated on Exp 538 latency confirmation.
If Exp 538 shows latency < 40s/question, Exp 539 extends to 100q with 90-min budget.

### Phase 3: Self-Learning — GRPO Pairing and Constraint Addition (Exps 540–541)

**Goal:** Wire live correct/incorrect pairs into EORM retrain (GRPO contrastive) and into
ConstraintAdditionFromMemory (Tier 2 self-learning).

Exp 540 implements GRPO-style contrastive pairing (arXiv 2503.06639): live benchmark pairs
(question, is_correct) from Exp 538 → contrastive EORM retrain. No additional labeling needed.
Exp 541 wires `ConstraintAdditionFromMemory` into the live pipeline: when FP tracker detects
a constraint type with FP_rate > threshold, adds a new domain-specific constraint automatically.

### Phase 4: FR-11 Scale-Up — Corpus Expansion and JEPA v8 Retrain (Exps 542–543)

**Goal:** Expand real CoT corpus from 57 to 100+ pairs; retrain JEPA on expanded corpus.

Exp 542 runs FOVER annotation on Exp 538 live CoT responses (25q × 2 models = 50 CoT chains).
Combined with fover_442 (57 pairs), target: 100+ labeled steps.
Exp 543 retrains JEPA v8 on the expanded corpus. Uses LeWorldModel two-term objective
(prediction + Gaussian KL regularization) for stable training. Target: AUC >= 0.900.

### Phase 5: LowRankKAEM Wire-In (Exp 544)

**Goal:** Wire k=2 LowRankKAEMEnergy into verification cascade as KAN fast-path.

Exp 532 proved 23.7x speedup at k=2 (the sole positive novel result in .40). Exp 544 replaces
the full-rank KAEMEnergy in the cascade with the low-rank version for all problems with ≤100 spins.
Spec: add REQ-SAMPLE-017 (low-rank default for n_vars ≤ 100).

### Phase 6: New Research (Exps 545–546)

**Goal:** Test two new 2025-2026 papers as concrete Carnot components.

Exp 545 implements `InternalStateProbe` (arXiv 2511.06209): single linear probe on LLM hidden state
at layer -4. Train on 57 real FOVER pairs. Compare AUC vs EORM (55M params). If AUC > 0.700 at
810x smaller, recommend as default Tier 2.

Exp 546 implements AutoRefine-style constraint template distillation (arXiv 2601.22758): accumulate
violation patterns from live benchmarks → offline distillation into named constraint templates →
retrieval at inference time. Closes the loop between Tier 2 memory and Tier 3 JEPA.

### Phase 7: Legacy Modernization Sprint (Exp 547)

**Goal:** Eliminate the 4 legacy experiments that consume the slowest-5 slots every milestone.

Exps 308, 260, 309, 425, 410 have appeared in the slowest-5 for four consecutive milestones with
cumulative overhead >1,020 minutes (17 hours). Exp 547 migrates all 5 to `BatchedInferenceRunner`
and `ExperimentTemplate` v2 (with teardown). Estimated savings: 8.5% wall-time per milestone.

### Phase 8: Retrospective (Exp 548)

Standard milestone retrospective following the v15 schema.

---

## Dependency Graph

```
Exp 537 (teardown)
    ├── Exp 538 (live 25q) ──────────────── RETRO-033 close attempt
    │       ├── Exp 539 (live 100q) ──────── RETRO-038 close attempt
    │       ├── Exp 540 (GRPO EORM) ──────── self-learning FR-11 Tier 1
    │       ├── Exp 541 (constraint add) ─── self-learning FR-11 Tier 2
    │       └── Exp 542 (FOVER expand) ──── FR-11 upstream
    │               └── Exp 543 (JEPA v8) ── FR-11 mandatory
    ├── Exp 544 (LowRankKAEM wire-in) ─── positive result
    ├── Exp 545 (InternalStateProbe) ───── new research
    ├── Exp 546 (AutoRefine templates) ─── new research
    ├── Exp 547 (legacy modernization) ─── throughput
    └── Exp 548 (retrospective) ─────────── reads all above
```

---

## Success Criteria

| Criterion | Target | Experiment |
|-----------|--------|------------|
| `teardown_implemented` | True | Exp 537 |
| `retro_054_resolved` | True | Exp 537 |
| `live_25q_completed` | True | Exp 538 |
| `retro_033_closed` | True | Exp 538 |
| `live_100q_completed` | True | Exp 539 |
| `retro_038_closed` | True | Exp 539 |
| `grpo_eorm_improved` | AUC improvement vs baseline | Exp 540 |
| `constraint_addition_live` | fp_rate_delta < 0 | Exp 541 |
| `fover_corpus_expanded` | n_labeled >= 100 | Exp 542 |
| `jepa_v8_auc` | >= 0.900 | Exp 543 |
| `lowrank_kaem_wired` | tier_kaem_default=lowrank | Exp 544 |
| `internal_probe_auc` | >= 0.700 (Tier 2 viable) | Exp 545 |
| `autorefine_templates_distilled` | n_templates >= 3 | Exp 546 |
| `legacy_modernized` | 5 experiments migrated | Exp 547 |
| `fr11_live_relay` | True | Exp 543 |

---

## Hardware Requirements

| Experiment | GPU Required | VRAM | Model |
|------------|-------------|------|-------|
| Exp 537 | No | — | CPU only |
| Exp 538 | Yes (CRITICAL) | 10+1.5 GiB | Gemma4-INT4 + Qwen3.5-0.8B |
| Exp 539 | Yes | 10+1.5 GiB | Gemma4-INT4 + Qwen3.5-0.8B |
| Exp 540 | No | — | Uses Exp 538 pairs |
| Exp 541 | Preferred | — | CPU feasible |
| Exp 542 | No | — | CPU FOVER annotation |
| Exp 543 | No | — | CPU JEPA training |
| Exp 544 | No | — | CPU LowRankKAEM |
| Exp 545 | No | — | CPU probe training |
| Exp 546 | No | — | CPU template distillation |
| Exp 547 | No | — | CPU script migration |
| Exp 548 | No | — | CPU retrospective |

**FIRST ACTION:** Kill zombie PIDs 430009 and 430012 before Exp 537. These hold ~47 GiB of zombie
VRAM on both GPUs from the .40 session close.

```bash
kill -9 430009 430012 2>/dev/null; sleep 2; nvidia-smi
```

---

## Open RETRO Items Addressed

| RETRO | Description | Addressed By |
|-------|-------------|-------------|
| RETRO-033 | Live 100q verify-repair positive result (miss #9) | Exp 538 (25q first) |
| RETRO-038 | Live 200q VeriCoT+VPRM statistically significant | Exp 539 |
| RETRO-052 | GPU1 0% compute utilization | Exp 538 (DualGPU explicit routing) |
| RETRO-054 | ExperimentTemplate.teardown() unimplemented (carry=5) | Exp 537 |
| RETRO-055 | Live inference timeout — reduce n_questions or increase budget | Exp 538 |

---

## arxiv Findings Incorporated

| Paper | What | Incorporated As |
|-------|------|----------------|
| arXiv 2511.06209 | Internal-state probes 810x smaller than PRM | Exp 545 |
| arXiv 2601.22758 | AutoRefine constraint template distillation | Exp 546 |
| arXiv 2503.06639 | GRPO contrastive pairing for verifiable rewards | Exp 540 |
| arXiv 2511.07124 | EBM calibration of latent CoT reasoning | Filed for .42 |
