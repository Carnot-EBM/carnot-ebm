# Carnot Research Roadmap v38: Live Numbers Confirmed, FR-11 Real-Data Validation, Spilled Energy Pre-Filter

**Created:** 2026-04-16
**Milestone:** 2026.04.32
**Status:** Planned (activates when milestone 2026.04.31 retrospective completes)
**Supersedes:** Milestone 2026.04.31 — "EnvironmentAutoFix, Complete Purge, First Live Numbers, VPRM Architecture"
**Informed by:** Exps 413–424, operational retrospective 2026.04.31, v37 carry-forwards
**External inputs (new in v38):**
- Spilled Energy (2602.18671) — logit-discrepancy hallucination detection; extends Semantic Energy Scorer
- KAEM (2506.14167) — KAN energy models with exact inverse-transform sampling; no MCMC needed
- GPU Oscillator Ising (2505.22631) — GPU-native oscillator Ising machine; CUDA-accelerated constraint sampling
- DSP feasibility channel (2604.02350) — differentiable constraint propagation with soft feasibility signal
- FoVer/FOVER (2505.15960) — Z3/Isabelle automatic step-level error annotation for PRM training
- RLVR (2506.14245) — verifiable reward RL; EORM as reward signal for LLM constraint extraction policy

---

## What 2026.04.31 Proved

| Approach | Experiments | Verdict | Key Finding |
|----------|-------------|---------|-------------|
| EnvironmentAutoFix (RETRO-022 workaround) | 413 | **COMPLETE** | auto_fix_applied; GPU hardware present; CARNOT_FORCE_LIVE now self-injects |
| CIKANEnergy re-implement | 414 | **COMPLETE** | Valid Python; RETRO-023 entry closed for this file |
| JitRL constraint memory | 415 | **COMPLETE** | Valid Python; threshold modulation > reweighting confirmed on synthetic |
| Safety KAN classifier | 416 | **COMPLETE** | Valid Python; first Tier B product implemented |
| Semantic Energy Scorer | 417 | **COMPLETE** | Valid Python; logit-space Boltzmann pre-filter implemented |
| CRANE extractor | 418 | **COMPLETE** | CPU-only regex + confidence gate; zero GPU dependency for extraction |
| Live precision pipeline (CRANE) | 419 | **PARTIAL** | Script implemented; live run in-progress at retro write time (GPU0 at 88% util) |
| Live HumanEval | 420 | **PARTIAL** | Script implemented; blocked at Gate 0 (env_not_propagating) |
| Live adversarial GSM8K | 421 | **PARTIAL** | Script implemented; blocked at Gate 0 |
| VPRM step labels | 422 | **PARTIAL** | Script implemented; live CoT pairs needed |
| EORM+JEPA retrain live | 423 | **PARTIAL** | Script implemented; live pairs needed |
| Retro + NPU | 424 | **PARTIAL** | Retro written; NPU prereq check pending |

**Milestone-level conclusion:**

Six CPU-only modules (CIKANEnergy, JitRL memory, Safety KAN, Semantic Energy, CRANE, EnvironmentAutoFix)
were implemented and tested. The apply_env_autofix() workaround (Exp 413) confirmed the GPU is
physically present and now self-configures CARNOT_FORCE_LIVE=1 at experiment startup. Exp 419 was
observed actively running on GPU0 at 88% utilization during the operational retrospective, confirming
that at least one live inference run has started. However, Exps 420–424 remain blocked (partial status).

RETRO-003 (conductor timeout) remains unimplemented at 17+ milestones. RETRO-025 (GPU1 idle VRAM)
was first observed this milestone. Both must be addressed as the first two experiments.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Zero published credible live benchmark numbers (CRITICAL)

**Status:** All headline results are still either simulated or pending live execution. Exp 419 ran
live during the retro, but the JSON shows `status: partial` — the full 200-question run across 5
variants × 2 models may not have completed (GPU0 at 82C under 144+ minutes of load).

**Impact:** Without credible live numbers, Carnot cannot make any publishable claims. The 16 model
READMEs on HuggingFace point to pip install carnot but have no verified improvement numbers. This
blocks commercial credibility.

**Fix:** Confirm Exp 419 results or re-run with RETRO-025 fixed (GPU1 scheduling). Execute Exps
420–421 (HumanEval and adversarial). Publish results to HuggingFace model cards after confirmation.

### Gap 2: FR-11 self-learning loop never confirmed on real data (HIGH)

**Status:** Six consecutive milestone failures. The infrastructure (JitRL memory, EORM, JEPA,
SelfLearningRelay) is all implemented and tested on synthetic data. The missing ingredient is live
CoT pairs from real LLM inference on GSM8K/HumanEval with ground-truth annotations.

**Fix:** Once Exp 419 produces live inference data, use the FOVER approach (arXiv 2505.15960) to
annotate each CoT step with Z3-derived correctness labels. These become EORM training pairs. Then
retrain EORM on these real pairs and confirm AUC > 0.5 (vs 0.5 baseline from synthetic-only).

### Gap 3: No Tier B products shipped beyond Safety KAN (MEDIUM)

**Status:** Safety KAN (Exp 416) is implemented but not published. Compliance checker and multi-agent
arbiter (both Tier B in research-program.md) have no implementation.

**Fix:** Implement compliance checker as the next Tier B product. It reuses the same KAN/Ising
energy tiers and constraint template library (Exp 343) — just with domain-specific constraint types
(regulatory, financial, medical). Ship alongside Safety KAN on HuggingFace.

---

## Architecture Diagram (Current State after 2026.04.31)

```
                        ┌─────────────────────────────────┐
                        │         INPUT PIPELINE          │
                        │  LLM response (text / code)     │
                        └────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │  TIER 0: FAST PRE-FILTERS (NEW)     │
                    │  ┌─────────────────────────────────┐│
                    │  │  SemanticEnergyScorer (Exp 417) ││  logit-space Boltzmann energy
                    │  │  SpilledEnergyDetector (v38 NEW)││  logit discrepancy (arXiv 2602.18671)
                    │  │  SinkProbe (Exp 348)            ││  attention sink concentration
                    │  └──────────────┬──────────────────┘│
                    │  LOW energy → SKIP to output        │
                    │  HIGH energy → continue to Tier 1   │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │  EXTRACTION LAYER                   │
                    │  ┌──────────────────────────────┐   │
                    │  │ CRANEExtractionGate (Exp 418)│   │  CPU-only regex + confidence
                    │  │ LLMConstraintExtractor (366) │   │  LLM-parsed structural claims
                    │  │ LLMz3Formalizer (Exp 357)   │   │  Z3/SMT symbolic formalization
                    │  │ AutoExtractor (auto-routes)  │   │  routes to best available
                    │  └──────────────┬───────────────┘   │
                    └────────────────┬────────────────────┘
                                     │
          ┌──────────────────────────▼──────────────────────────┐
          │              THREE-TIER VERIFICATION                 │
          │  ┌────────────────────────────────────────────────┐  │
          │  │ Tier 1: EORM (Exp 346+)  attention-sink probe │  │  fast, ~0.1ms
          │  │ Tier 2: KAN (Exp 96) / CIKAN (Exp 414)        │  │  medium, ~1ms
          │  │ Tier 3: IsingEBM + ThreeTierPipeline (360)    │  │  thorough, ~6ms
          │  └────────────────────────────────────────────────┘  │
          │  JitRL memory (Exp 415) adjusts thresholds at query  │
          └──────────────────────────┬──────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │  REPAIR LAYER                       │
                    │  VerifyRepairPipeline (Exp 74-75)   │
                    │  VERGE iterative Z3 refinement      │
                    │  ConstraintTemplateLibrary (343)    │
                    │  SAVeRVerifier for multi-turn       │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │  SELF-LEARNING RELAY (FR-11)        │
                    │  Tier 1: JitRL threshold modulation │
                    │  Tier 2: CaseMemory template wiring │
                    │  Tier 3: EORM/JEPA retrain on pairs │
                    │  FOVER: Z3-label CoT steps (v38 NEW)│
                    └────────────────────────────────────┘
```

---

## Phase Descriptions

### Phase 1: Infrastructure Hardening (Experiments 425–426)

**Mandatory first.** Two retro items blocking efficient operation.

**Exp 425 — Conductor timeout (RETRO-003, 17+ milestones):**
Implement a Python watchdog class that wraps every experiment and kills the process after a
configurable timeout (default 45 minutes). The watchdog runs in a background thread, emits a
RETRO log entry on timeout, and writes a partial result JSON so the conductor can continue.
This has been deferred 17+ consecutive milestones. The retro captured PID 3509070 at 144+ minutes
with GPU0 at 82C. Non-negotiable first experiment.

**Exp 426 — DualGPURunner GPU-1 scheduling fix + temperature guard (RETRO-025):**
Fix the DualGPURunner to assert GPU-1 utilization > 0% within 60s of model load. Add a
temperature guard in ExperimentTemplate.setup_gpu() that warns and reduces batch size when
any GPU exceeds 80C. Fix the zombie VRAM allocation pattern (Exp 419 held 1786MB on GPU1 at 0%).

### Phase 2: Confirm Live Benchmark Results (Experiments 427–429)

**Validate the first credible headline numbers.**

All three experiments use apply_env_autofix() at startup and gate on Exp 413 verdict.
They execute the scripts already implemented in Exps 419–421 but with infrastructure fixes
from Phase 1 in place (RETRO-025 fix means GPU1 is actually used).

**Exp 427:** Re-confirm or re-run Exp 419 precision live benchmark.
**Exp 428:** Execute Exp 420 HumanEval live benchmark.
**Exp 429:** Execute Exp 421 adversarial GSM8K live benchmark.

Success criterion: at least one experiment produces `honest_verdict='live_improvement'` and
`inference_mode='live_gpu'`. This is the credible headline number Carnot has been waiting 8+
milestones to produce.

### Phase 3: FR-11 Self-Learning with Real Data (Experiments 430–432)

**Close RETRO-024 by giving the self-learning relay actual training data.**

The pipeline: live inference (Phase 2) produces CoT steps → Z3 annotates each step with
correctness labels (FOVER pattern) → EORM retrains on (partial_cot, step_label) pairs →
SelfLearningRelay confirms AUC improvement on held-out data.

**Exp 430:** FOVER Z3 step annotation pipeline — parse Exp 427 CoT responses into steps,
annotate each step with Z3 arithmetic verification, produce labeled (step, correct/incorrect) pairs.
This generates the training data that five consecutive milestones lacked.

**Exp 431:** EORM + JEPA retrain on Exp 430 labeled pairs (vs Exp 423 synthetic-only baseline).
Target: after_auc > before_auc (0.5), honest_verdict='real_data_improvement'.

**Exp 432:** JitRL live validation — run 100 live GSM8K questions through the full JitRL memory
pipeline (Exp 415) with real constraint history from Exp 427. Measure FP reduction vs baseline.
This is the Tier 1 self-learning live validation that completes FR-11 requirements.

### Phase 4: New Capabilities (Experiments 433–436)

**Advance the product roadmap and hardware path.**

**Exp 433:** Spilled Energy integration (arXiv 2602.18671). Implement SpilledEnergyDetector
as Tier 0 pre-filter alongside SemanticEnergyScorer. Compare skip rates and FN rates.
This is the highest-impact new capability from the arxiv scan — free signal from logits.

**Exp 434:** Compliance checker (Tier B product). Implement ComplianceEnergyChecker with
domain-specific constraint types (financial: no unauthorized advice; medical: no clinical
recommendations; legal: no binding statements). Uses ConstraintTemplateLibrary with new
compliance templates. Produce results/experiment_434_compliance_checker.json with AUC > 0.7.

**Exp 435:** AMD XDNA NPU unblock. Check if ninja and openblas are now installed. If yes,
run the source build path from Exp 292/303/314/335. If still blocked, generate actionable
install instructions and measure what's achievable with the existing ONNX model path.

**Exp 436:** Operational retrospective (milestone 2026.04.32). Compute timing stats, assess
whether live_numbers_confirmed and fr11_relay_confirmed are True. Flag new RETRO items.

---

## Dependency Graph

```
Exp 425 (timeout watchdog)
  └─→ Exp 426 (GPU1 fix)
        └─→ Exp 427 (precision live)
              ├─→ Exp 428 (humaneval live)
              ├─→ Exp 429 (adversarial live)
              └─→ Exp 430 (FOVER Z3 labels from live CoT pairs)
                    └─→ Exp 431 (EORM+JEPA retrain on real pairs)
                          └─→ Exp 432 (JitRL live validation)

Exp 433 (spilled energy)  ← independent of live GPU (CPU-based on logits)
Exp 434 (compliance checker)  ← independent, CPU-only KAN
Exp 435 (NPU unblock)  ← independent, blocked by human prereq install
Exp 436 (retrospective)  ← depends on all prior experiments
```

---

## Success Criteria for Milestone 2026.04.32

| Criterion | Target | Experiment |
|-----------|--------|-----------|
| conductor_timeout_implemented | True | Exp 425 |
| gpu1_zombie_fixed | True | Exp 426 |
| live_numbers_confirmed | At least 1 experiment with honest_verdict='live_improvement' | Exp 427-429 |
| fr11_relay_confirmed | after_auc > before_auc on real CoT pairs | Exp 431 |
| jitrl_live_validated | fp_reduction_pct > 0 on real data | Exp 432 |
| spilled_energy_viable | skip_rate > 20% with FN < 5% | Exp 433 |
| compliance_checker_works | test_auroc > 0.70 | Exp 434 |
| npu_prereqs_installed | honest_verdict != 'blocked_prereq' | Exp 435 |

---

## Hardware Requirements

- 2x RTX 3090 (48GB VRAM) — live inference on Gemma4-E4B-it (GPU0) + Qwen3.5-0.8B (GPU1)
- AMD XDNA NPU — needs ninja + openblas install (human action required before Exp 435)
- Kria KV260 FPGA — not required this milestone (bitfile still pending)
- CPU — all CPU experiments (Exp 425, 426, 430, 433, 434)

**Human action required before milestone starts:**
1. Install ninja: `sudo pacman -S ninja` (or `sudo apt install ninja-build`)
2. Install openblas: `sudo pacman -S openblas` (or `sudo apt install libopenblas-dev`)
   (These unblock Exp 435, AMD XDNA NPU)
3. Verify `source scripts/session_startup.sh` runs before conductor session
4. Physical cooling check: GPU0 reached 82C in prior session — clean heatsinks before
   starting 3+ hour live inference sessions

---

## Papers Driving This Milestone

| Paper | ID | Experiment |
|-------|----|-----------|
| Spilled Energy in LLMs | 2602.18671 | Exp 433 |
| FoVer formal step annotation | 2505.15960 | Exp 430 |
| VPRM process reward models | 2601.17223 | Exp 430, 431 |
| JitRL continual learning | 2601.18510 | Exp 432 |
| GPU oscillator Ising | 2505.22631 | (Tier B, future) |
| KAEM exact-sampling KAN | 2506.14167 | (Tier B, future) |
| ARM-EBM bijection | 2512.15605 | Theoretical foundation |
| CRANE alternating decoding | 2502.09061 | Baseline (Exp 418 already done) |
| AMD XDNA IRON toolchain | 2504.03083 | Exp 435 |
