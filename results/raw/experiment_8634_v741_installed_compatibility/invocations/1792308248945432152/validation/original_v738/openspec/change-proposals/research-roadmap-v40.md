# Research Roadmap v40 — Milestone 2026.04.40

**Status:** Proposed
**Milestone:** 2026.04.40
**Title:** Fix the Last Gate — Eighth Attempt, First Live Positive
**Planned Experiments:** 526–536 (11 experiments)
**Planned Date:** 2026-04-19 onwards

---

## What Milestone 2026.04.39 Proved

Milestone .39 delivered three meaningful algorithmic wins and resolved the final infrastructure
blocker in the RETRO-033 chain — but was still blocked from live benchmarks by a one-line bug
in `env_autofix`.

1. **RETRO-051 CLOSED** — `JITVRAMCheck` implemented and wired into both `Gemma4QuantizedLoader`
   and `GemmaTransformersLoader`. Just-in-time VRAM gating with one retry now converts silent
   CUDA OOM into a diagnostic fast-fail. The infrastructure chain that caused six consecutive
   benchmark deferrals is now complete.

2. **RETRO-049 CLOSED** — NUP Probe v4 contrastive training achieved AUC=1.0 (vs BCE AUC=0.40).
   The correct learning objective for EBM constraint scoring is energy-gap margin loss, not binary
   cross-entropy. This validates the core architectural principle: optimize E(incorrect)-E(correct)
   directly, not classification boundaries.

3. **RETRO-039 CLOSED (negative result)** — GSM-Symbolic adversarial thesis definitively rejected by
   Exp 516. Carnot's improvement is NOT larger under adversarial numeric perturbations. The Ising
   constraint verification is sensitive to surface form. This is an honest research finding.

4. **LeWorldModel JEPA** — AUC=0.972 with 274x variance reduction vs standard BCE training. Stable
   training objective confirmed. FR-11 live relay confirmed.

5. **Hallucination Basin Detector** — viable as Tier 0d. New detection modality using latent-space
   basin depth added to the verification cascade.

**The one remaining gate (RETRO-053):**
`apply_env_autofix()` checks for presence of `CARNOT_FORCE_LIVE` and skips injection when the
variable exists, even when its value is `'0'`. The live experiments gate on truthiness — `'0'` is
falsy — so the eighth consecutive benchmark attempt (Exp 514) deferred in under one second because
`env_autofix` silently left `CARNOT_FORCE_LIVE='0'` in the environment.

**Fix:** In `apply_env_autofix()`, when `gpu_detected=True`, treat `CARNOT_FORCE_LIVE` values of
`None`, `''`, `'0'`, `'false'`, `'False'` as equivalent to not-set and inject `'1'`. This is a
single conditional change.

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: CARNOT_FORCE_LIVE='0' Not Overridden (FR-12 — Verifiable Reasoning)

**RETRO-053 is the single remaining programmatic gate before live benchmarks execute.**

Seven consecutive milestones (.33 through .39) ended with RETRO-033 open. The progression was:
GPU offline → env not propagating → zombie VRAM → conductor VRAM → VRAM budget → runtime VRAM state
→ CARNOT_FORCE_LIVE='0' not overridden.

The infrastructure is now correct at every other level:
- `JITVRAMCheck` (Exp 513) gates model loading against real-time VRAM state ✓
- `Gemma4QuantizedLoader` (Exp 500) fits within 24 GiB budget ✓
- `GPUVRAMGateV2` (Exp 487) kills zombie VRAM holders before GPU runs ✓
- `EnvironmentAutoFix` (Exp 413) injects `CARNOT_FORCE_LIVE=1` when var is absent ✓

The single gap: `env_autofix` treats `'0'` as a user-intentional override and skips injection.
Fixing this in Exp 526 enables Exps 527 and 528 to run on the live GPU for the first time in
seven milestones.

**Evidence from Exp 524 retro:**
```json
"env_autofix": {
  "gpu_detected": true,
  "carnot_force_live_was_set": true,
  "auto_fix_applied": false,
  "final_env_value": "0"
}
```

### Gap 2: GPU 1 at 0% Forward-Pass Compute (Hardware Efficiency)

Exp 517 (controlled DualGPU test) confirmed `gpu1_compute_pct=0.0` in live GPU mode. The
`build_zombie_fix_strategy()` device assignment is wired in the template, but no live benchmark
script has executed a forward pass on `cuda:1` and verified utilization. The dual RTX 3090 system
is running at 50% GPU capacity.

**Fix in Exp 529:** Write a minimal script that explicitly loads Qwen3.5-0.8B with
`device_map={'': 'cuda:1'}`, runs 10 inference passes, and measures `nvmlDeviceGetUtilizationRates()`
on `cuda:1` during inference. If GPU 1 utilization > 10%, the hardware is working — the template's
routing fix is sufficient. If still 0%, diagnose whether `device_map` propagation is broken at the
`transformers` level.

This is distinct from Exp 517 (which diagnosed the problem). Exp 529 fixes it and verifies the fix.

### Gap 3: Validated Components Not Wired into Live Pipeline (FR-12)

Three components validated in .39 are CPU-tested but not wired into the `VerifyRepairPipeline`:

| Component | .39 Result | Gap |
|-----------|-----------|-----|
| NUP Probe v4 | AUC=1.0 (contrastive) | Not in cascade; AUC validates it for Tier 0c |
| Hallucination Basin Detector | viable_tier0d | Not in cascade; adds new detection modality |
| LeWorldModel JEPA | AUC=0.972, 274x variance reduction | Not used in live inference path |

**Fix in Exp 530:** Wire NUP Probe v4 and Hallucination Basin Detector into `ThreeTierPipeline`
as optional Tier 0c and Tier 0d respectively. Update `_bmad/architecture.md` Verification Pipeline
Tiers table.

---

## Phase Descriptions

### Phase 1: Fix the Env Gate (Exp 526)
One-liner fix to `apply_env_autofix()`. This is the single gate before live benchmarks.
RETRO-053 targeted. No GPU required. Expected duration: 15 min.

### Phase 2: Live Credibility Benchmarks (Exps 527–528)
Eighth attempt at RETRO-033 (100q live precision) and seventh attempt at RETRO-038 (200q
statistically significant). Both use JIT VRAM gating and the Gemma4-INT4 + Qwen3.5-0.8B dual
model configuration. Each writes CoT pairs for JEPA retrain. Expected duration: 120-150 min each
(GPU-required). RETRO-033 closes if Exp 527 `honest_verdict='retro_033_closed'`.

### Phase 3: GPU1 Activation Fix (Exp 529)
Explicit `cuda:1` routing for one model, with utilization measurement. Closes RETRO-052 if
`gpu1_compute_pct > 10.0`. Expected duration: 30 min (GPU-required).

### Phase 4: Production Wiring (Exps 530–531)
Exp 530 wires NUP Probe v4 (Tier 0c) and Hallucination Basin Detector (Tier 0d) into the live
verification cascade. Exp 531 implements EORM-as-PRM adaptive rectification sampling (arXiv
2504.01317) — generate K candidates, score with EORM, select minimum-energy. Both CPU-feasible
with synthetic data, live inference optional.

### Phase 5: New Research (Exps 532–534)
Three independent research experiments:
- **Exp 532:** Low-Rank KAEMEnergy (arXiv 2604.04384) — SVD projection to 2-11 singular
  components. Hypothesis: k=11 achieves >95% of full-rank AUC at 10-100x fewer operations.
- **Exp 533:** COLD Decoding energy-guided token selection (arXiv 2202.11705 + 2604.14862) —
  minimal energy-guided generation: score K candidate next tokens with IsingEBM, select
  minimum-energy. Compare violation rate vs unconstrained.
- **Exp 534:** Potts Machine Verifier (arXiv 2602.04200) — q=3 Potts sampler for 3-state
  constraint encoding (correct/partial/violated). Benchmark vs binary IsingEBM AUROC.

### Phase 6: Self-Learning — FR-11 Mandatory (Exp 535)
JEPA live retrain v7 using LeWorldModelJEPATrainer on CoT pairs from Exps 527-528 if available,
with fallback to prior real data. Target: AUC >= 0.800 on held-out pairs. FR-11 closes if
`fr11_live_relay=True`.

### Phase 7: Retrospective (Exp 536)
Milestone retrospective. Key question: did RETRO-033 finally close after eight attempts?

---

## Dependency Graph

```
Exp 526 (env_autofix fix)
    └── Exp 527 (live 100q v8) ──────────────────────────┐
    └── Exp 528 (live 200q v7) ──────────────────────────┤
                                                          │
    └── Exp 529 (GPU1 fix) ────────────── independent    │
                                                          │
Exp 530 (pipeline wiring)  ──────────── independent      │
Exp 531 (EORM PRM)         ──────────── independent      │
Exp 532 (low-rank KAEM)    ──────────── independent      │
Exp 533 (COLD decoding)    ──────────── independent      │
Exp 534 (Potts machine)    ──────────── independent      │
                                                          │
Exp 535 (JEPA v7) ← Exps 527+528 CoT pairs (preferred) ─┘
    └── fallback: prior real data from Exp 442/522

Exp 536 (retrospective) ← all above
```

---

## Architecture Diagram (Updated Verification Cascade)

```
Input: LLM response text
         │
         ▼
[Tier 0a] CarnotThinkProbe — generative CoT pre-filter (optional)
         │ if verdict='incorrect' → fast-path violation
         ▼
[Tier 0b] SpilledEnergyDetector — logit-discrepancy energy
         │ if high_spill_fraction <= threshold → pass through
         ▼
[Tier 0c] NUP Probe v4 — contrastive energy-gap probe (NEW in .40)
         │ if basin_risk_score <= threshold → pass through
         ▼
[Tier 0d] HallucinationBasinDetector — latent-space basin depth (NEW in .40)
         │ if basin_risk_score <= threshold → pass through
         ▼
[Tier 1]  SinkProbe — attention sink concentration
         │ if mean_sink_score >= sink_threshold → pass through
         ▼
[Tier 2]  EORM — CoT energy reward model (55M params, ~10ms)
         │ if energy < eorm_threshold → pass through
         ▼
[Tier 3]  Ising + VeriCoT + VPRM — full constraint verification
         │ if violations found → BoltzmannRepairBridge → repaired response
         ▼
Output: verified + repaired response
```

---

## Hardware Requirements

| Experiment | GPU Required | Notes |
|-----------|-------------|-------|
| 526 | No | CPU-only env fix |
| 527 | YES (cuda:0+1) | Gemma4-INT4 on cuda:0, Qwen3.5-0.8B on cuda:1 |
| 528 | YES (cuda:0+1) | Same as 527 |
| 529 | YES (cuda:1) | Explicit cuda:1 routing verification |
| 530 | No | CPU-only wiring |
| 531 | No | CPU-only with synthetic |
| 532 | No | CPU-only SVD benchmark |
| 533 | No | CPU-only generation test |
| 534 | No | CPU-only Potts sampler |
| 535 | No (preferred: cot_pairs from 527/528) | LeWorldModel JEPA training |
| 536 | No | Retrospective |

**KV260 FPGA:** Exp 534 (Potts machine) is FPGA-relevant. If KV260 bitfile is available,
the Potts sampler can be benchmarked against the FPGA backend. Not required for Exp 534 to run.

**AMD XDNA NPU:** Still blocked (ninja + openblas not installed). No NPU experiments in .40.
Human must install `sudo pacman -S ninja openblas` before Exp 511 can be retried.

---

## Success Criteria

| Criterion | Source | Target |
|-----------|--------|--------|
| retro_053_resolved | Exp 526 | env_autofix overrides '0' values |
| retro_033_closed | Exp 527 | live GPU, is_positive=True |
| retro_038_closed | Exp 528 | Wilson CI lower > 0, live GPU |
| gpu1_compute_verified | Exp 529 | gpu1_compute_pct > 10% |
| tier0c_wired | Exp 530 | NUP Probe v4 in pipeline cascade |
| tier0d_wired | Exp 530 | Basin Detector in pipeline cascade |
| lowrank_kaem_viable | Exp 532 | k=11 AUC > 95% full-rank |
| cold_decoding_viable | Exp 533 | violation_rate_delta < 0 |
| potts_viable | Exp 534 | potts_auroc > ising_auroc |
| fr11_live_relay | Exp 535 | final_auc >= 0.800, real data |
| milestone_complete | Exp 536 | retro written |

---

## arxiv Papers Incorporated

| Paper | arXiv ID | Incorporated As |
|-------|---------|----------------|
| Adaptive Rectification Sampling | 2504.01317 | Exp 531 (EORM as test-time PRM) |
| Potts Machine Mean-Field Constraints | 2602.04200 | Exp 534 (Potts verifier) |
| COLD Decoding (energy-based constrained generation) | 2202.11705 | Exp 533 (energy-guided token selection) |
| Constrained Decoding Near-Zero Overhead | 2604.14862 | Exp 533 (context) |
| Low-Rank Logit Energy Landscape | 2604.04384 | Exp 532 (low-rank KAEM) |
| GRPO Verifiable Rewards as Contrastive | 2503.06639 | Filed for .41 (contrastive training data) |
| IR³ Contrastive IRL | 2602.19416 | Filed for .41 (reward hacking detection) |
| AutoRefine Trajectory Distillation | 2601.22758 | Filed for .41 (Tier 2 retrieval) |
