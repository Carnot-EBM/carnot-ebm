# Carnot Research Roadmap v39: First Live Results, ThinkPRM Bridge, Boltzmann-GPT Repair

**Created:** 2026-04-17
**Milestone:** 2026.04.33
**Status:** Planned (activates when milestone 2026.04.32 retrospective completes)
**Supersedes:** Milestone 2026.04.32 — "Live Numbers Confirmed, FR-11 Real-Data Validation, Spilled Energy Pre-Filter"
**Informed by:** Exps 425-436, operational retrospective 2026.04.32, v38 carry-forwards
**External inputs (new in v39):**
- ThinkPRM (2504.16828) — generative CoT verifier; 1% supervision labels, SOTA on MATH-500
- Boltzmann-GPT (2601.17094) — DBM world models bridge to language generation; energy-guided repair
- Energy Matching (2504.10612, NeurIPS 2025) — flow + EBM unified; Phase 3 continuous generation
- Generative Thermodynamic Computing (2506.15121) — Langevin dynamics from free energy landscape
- Process Reward Agents (2604.09482) — decoupled step-level reward; IsingEBM as PRA

---

## What 2026.04.32 Proved

| Approach | Experiments | Verdict | Key Finding |
|----------|-------------|---------|-------------|
| ExperimentTimeoutWatchdog (RETRO-003) | 425 | **CLOSED** | Watchdog implemented; 45-min hard cap; RETRO-003 finally resolved after 17+ milestones |
| DualGPU health check (RETRO-025) | 426 | **PARTIAL** | Zombie confirmed (GPU1 0% util, 1786MB VRAM). Not fixed — diagnosis only |
| Live precision benchmark | 427 | **SCAFFOLDING_ONLY** | Harness exists; live run blocked by 45-min conductor budget (RETRO-026) |
| Live HumanEval | 428 | **SCAFFOLDING_ONLY** | Harness exists; blocked by budget (RETRO-026) |
| Live adversarial GSM8K | 429 | **SCAFFOLDING_ONLY** | Harness exists; blocked by budget (RETRO-026) |
| FOVER Z3 annotation | 430 | **COMPLETE** | FOVERAnnotator implemented; synthetic fallback (Exp 427 not live) |
| EORM+JEPA live retrain | 431 | **PARTIAL** | Harness exists; retro_024_closed=False; needs Exp 427 live data |
| JitRL live validation | 432 | **SYNTHETIC_FALLBACK** | 33.71% FP reduction on synthetic; live deferred |
| SpilledEnergyDetector (Tier 0) | 433 | **NO RESULT JSON** | Module exists; conductor silently dropped (RETRO-027) |
| ComplianceEnergyChecker | 434 | **NO RESULT JSON** | Module exists; conductor silently dropped (RETRO-027) |
| AMD NPU unblock (5th) | 435 | **NO RESULT JSON** | Script exists; dropped by conductor (RETRO-027) |
| Kona Phase 3 seed (continuous EBM) | 435a | **PARTIAL_MATCH** | L2=2.69 (target <0.1), sign_agreement=0.8. Gradient descent converges but imprecisely |

**Milestone-level conclusion:**

The watchdog (RETRO-003) finally shipped after 17+ milestones. The core problem now has two parts:
RETRO-026 (live benchmarks need more than 45 min per subagent budget) and RETRO-027 (conductor
silently dropped Exps 433, 434, 435 with no result JSONs). Until RETRO-026 is fixed, live
benchmark numbers will never arrive. The fix is structural: break 200-question benchmarks into
50-question micro-batches that each fit in 45 minutes.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Still zero published live benchmark numbers — 7th consecutive milestone (CRITICAL)

**Status:** All three live benchmark harnesses (precision, HumanEval, adversarial) are
scaffolding_only after 7 consecutive milestones. RETRO-026 is the direct cause: the conductor
subagent budget (~45 min) is too small for 200 questions × 5 variants × 2 models.

**Root cause:** Benchmarks were designed at full scale (200q × 5 variants × 2 models = 2000 LLM
calls). Each call is ~5-15s on GPU. 2000 calls × 10s = 20,000s = 333 min. Even with batching and
parallelism, this exceeds a 45-min subagent budget by 7x.

**Fix:** Redesign benchmarks as micro-batches: 50 questions × 3 variants × 2 models = 300 calls
≈ 50 min. Use LongRunBenchmarkExecutor (Exp 437) to split into 50q batches, each checkpointed.
The first 50-question result is publishable — it beats "no result" in every dimension.

### Gap 2: FR-11 self-learning unconfirmed on real data — 8th consecutive milestone carry (HIGH)

**Status:** EORM/JEPA retrained only on synthetic data (AUC = 0.5 baseline). FOVER annotator
(Exp 430) is implemented but produced synthetic_fallback because Exp 427 had no live results.
The chain is: live inference → FOVER annotation → real training pairs → EORM retrain → AUC > 0.5.
Only the live inference step (step 1) is missing.

**Fix:** Once Exp 439 produces even 50 questions of live inference data, Exp 442 (FOVER annotation)
can produce real training pairs. Exp 443 (EORM retrain) can then retrain on real data. With 50
questions and ~15 annotatable steps per question, we get ~750 training pairs — enough for retrain.

### Gap 3: Phase 3 continuous energy landscape imprecise (MEDIUM)

**Status:** Exp 435a (Kona Phase 3 seed): L2=2.69, sign_agreement=0.8. Target was L2<0.1.
The continuous EBM converges to the right sign pattern (80%) but not to the correct magnitude.
Root cause: gradient descent on E(x) = -0.5*x^T*J*x - h^T*x with tanh squashing gets stuck in
local minima. The Ising ground state (discrete) is not a global minimum of the continuous landscape.

**Fix:** Use Langevin dynamics (gradient descent + noise) instead of pure gradient descent.
The noise term prevents local minimum trapping. Also: initialize the continuous model from the
Ising discrete solution (warm start), then use Langevin to refine. Theory from Generative
Thermodynamic Computing (arXiv 2506.15121) supports this approach.

---

## Architecture Diagram (Current State after 2026.04.32)

```
                        ┌─────────────────────────────────┐
                        │         INPUT PIPELINE          │
                        │  LLM response (text / code)     │
                        └────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │  TIER 0: FAST PRE-FILTERS           │
                    │  ┌─────────────────────────────────┐│
                    │  │  SemanticEnergyScorer (Exp 417) ││ logit-space Boltzmann energy
                    │  │  SpilledEnergyDetector (Exp 433)││ logit discrepancy (2602.18671)
                    │  │  SinkProbe (Exp 348)            ││ attention sink concentration
                    │  │  CarnotThinkProbe (Exp 444 NEW) ││ CoT pre-verification (2504.16828)
                    │  └──────────────┬──────────────────┘│
                    │  LOW energy → SKIP to output        │
                    │  HIGH energy → continue to Tier 1   │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │  TIER 1: ISING CONSTRAINT CHECK     │
                    │  VerifyRepairPipeline               │
                    │  CRANEExtractionGate (Exp 418)      │
                    │  JitRLConstraintMemory (Exp 415)    │
                    │  PerModelFPTracker (Exp 333)        │
                    │  ConstraintTemplateLibrary (Exp 343)│
                    └────────────────┬────────────────────┘
                                     │ violation detected
                    ┌────────────────▼────────────────────┐
                    │  TIER 2: KAN ENERGY + EORM          │
                    │  KANEnergy / CIKANEnergy (Exp 414)  │
                    │  EORMModel (Exp 346; retrain 431)   │
                    │  ComplianceEnergyChecker (Exp 434)  │
                    │  BoltzmannRepairBridge (Exp 445 NEW)│
                    └────────────────┬────────────────────┘
                                     │ high energy confirmed
                    ┌────────────────▼────────────────────┐
                    │  TIER 3: REPAIR + SELF-LEARNING     │
                    │  JEPA Predictor (Exp 291/347)       │
                    │  FOVERAnnotator (Exp 430)           │
                    │  SelfLearningRelay (Exp 361)        │
                    │  SessionMemory (Exp 345)            │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │  PHASE 3 SEED: CONTINUOUS EBM       │
                    │  ContinuousEBM (Exp 435a)           │
                    │  Energy Matching (Exp 446 NEW)      │
                    │  KAEMEnergy exact sampling (447 NEW)│
                    └─────────────────────────────────────┘
```

---

## Phase Descriptions

### Phase 1: Infrastructure — Fix the Root Cause of Scaffolding-Only Benchmarks

The direct cause of 7 consecutive scaffolding-only milestones is benchmark scope exceeding
subagent budget. This phase fixes the structural problem with LongRunBenchmarkExecutor and
attempts to resolve the GPU1 zombie scheduling that inflates runtimes.

**Exp 437:** LongRunBenchmarkExecutor (RETRO-026 fix)
- Implements `LongRunBenchmarkExecutor` that splits any benchmark into 50-question batches
- Each batch fits in 45 min; checkpoints between batches; resumes from checkpoint on retry
- CPU-only implementation; always produces a result JSON
- Enables all three pending live benchmarks (Exps 439-441) to complete within budget

**Exp 438:** GPU1 Zombie Root-Cause Fix (RETRO-025)
- Diagnoses WHY GPU1 allocates VRAM but computes nothing (0% utilization)
- Attempts fix: explicit device_map={"": "cuda:1"}, force model to GPU1 device
- If fix works: retests DualGPURunner with zombie detection cleared
- If fix fails: documents root cause, recommends fallback to GPU0-only mode

### Phase 2: Live Benchmarks — 50-Question Micro-Batches

Using LongRunBenchmarkExecutor from Phase 1, run SMALLER benchmarks (50q each) that fit in 45 min.
50 questions × 3 variants × 2 models = 300 LLM calls ≈ 45 min. Each produces a credible live result.

**Exp 439:** Live Precision Micro-Benchmark (50q × 3 variants × 2 models)
- Models: Gemma4-E4B-it (GPU0), Qwen3.5-0.8B (GPU1)
- Variants: BASELINE, CONFIDENCE_ADAPTIVE, FULL_STACK (3 vs 5 from Exp 427)
- Primary outcome: signed_improvement on live data — first credible headline
- Secondary outcome: CRANE detection rate on live IT model outputs
- Deliverable: first publishable verify-repair accuracy number on live inference

**Exp 440:** Live HumanEval Micro-Benchmark (50 problems × 2 models)
- Models: Gemma4-E4B-it (GPU0), Qwen3.5-0.8B (GPU1)
- Confirm or refute Exp 226's +3.0pp on live HumanEval
- Code verification is the most structurally sound extraction (execute code, not regex)
- Expected: positive result since CodeExtractor uses execution, not regex pattern matching

**Exp 441:** Live Adversarial GSM8K (50q × 3 conditions × 2 models)
- Conditions: standard / irrelevant-sentence injection / repaired (Apple arXiv 2410.05229)
- Primary check: adversarial_drop > 0 (LLM drops, as Apple showed) AND repair_improvement > 0
- This is Carnot's thesis experiment — proof that constraint verification is robust to adversarial noise

### Phase 3: FR-11 Real-Data Validation — Close the 8-Milestone Carry

Once Exp 439 produces live CoT data, use it to close RETRO-024.

**Exp 442:** FOVER Live Annotation (uses Exp 439 live data)
- Runs FOVERAnnotator on Exp 439's live CoT responses
- Produces real (step, Z3_label) pairs for EORM training
- honest_verdict='real_data_labeled' if ≥20 real pairs; 'synthetic_fallback' if not
- Deliverable: results/fover_labeled_steps_live.json — real training data

**Exp 443:** EORM+JEPA Live Retrain (uses Exp 442 real pairs)
- Retrains EORM on Exp 442 real pairs via contrastive divergence
- Reports before_auc (baseline 0.5) vs after_auc
- honest_verdict='real_data_improvement' if AUC > 0.5 AND n_real_pairs ≥ 10
- Closes RETRO-024 if honest_verdict='real_data_improvement'

### Phase 4: New Research — ThinkPRM Bridge + Boltzmann-GPT + Phase 3

**Exp 444:** CarnotThinkProbe — ThinkPRM-style CoT Verifier (arXiv 2504.16828)
- Implements a lightweight secondary verification CoT using Qwen3.5-0.8B
- Given an LLM response, generates a 3-step check ("Step 1: Extract the arithmetic claim. Step 2: Verify it. Step 3: Verdict.")
- If secondary CoT concludes "incorrect," flag as violation immediately (skip Ising)
- Only run Ising for "uncertain" or "correct" secondary verdicts (slow-path)
- Measures: skip_rate (how often ThinkProbe replaces Ising), precision (ThinkProbe-only FP rate)
- CPU-only with Qwen CI-stub; always produces result

**Exp 445:** BoltzmannRepairBridge — DBM Energy → LLM Repair Direction (arXiv 2601.17094)
- Maps Carnot's IsingEBM ground state configuration to an LLM embedding steering direction
- Inspired by Boltzmann-GPT's DBM-to-adapter-to-LLM bridge architecture
- IsingEBM finds low-energy (constraint-satisfying) spin configuration
- BoltzmannRepairBridge projects the configuration into embedding space → repair suggestion
- Replaces the "ask LLM to fix error" repair step with an energy-guided alternative
- CPU-only; measures whether energy-guided repair is more targeted than prompt-based repair

**Exp 446:** Energy Matching for ContinuousEBM (arXiv 2504.10612) — Phase 3 seed v2
- Implements Energy Matching trajectory sampling for ContinuousEBM (Exp 435a)
- Flow from noise to data via energy gradient, with thermodynamic Langevin noise (arXiv 2506.15121)
- Goal: improve Exp 435a's L2=2.69 to L2<0.5 using flow-guided sampling
- Compares three sampling algorithms: gradient descent (Exp 435a), Langevin dynamics, Energy Matching
- Deliverable: continuous energy landscape that better matches Ising ground state

**Exp 447:** KAEMEnergy — Exact Inverse-Transform Sampling (arXiv 2506.14167)
- Implements KAEMEnergy as a fast-path KAN variant with exact sampling (no MCMC)
- KAEM imposes univariate latent structure enabling closed-form inverse-transform sampling
- Measures latency reduction vs IsingEBM (target: 10x faster for sub-100-variable problems)
- CPU-only; KAEM structure + CIKAN boundary priors from CIKANEnergy (Exp 414)
- Compares: KAEM exact vs Ising MCMC vs KANEnergy gradient on same constraint tasks

### Phase 5: Continuous Self-Learning (MANDATORY per research-program.md)

**Exp 448:** Tier 2 Cross-Session Constraint Memory Relay
- Tests that constraint templates learned in one session persist and improve the NEXT session
- Uses SessionMemory (Exp 345) + ConstraintTemplateLibrary (Exp 343) + SelfLearningRelay (Exp 361)
- Run three sequential sessions on arithmetic problems:
  - Session 1: baseline (no prior memory), record carry_check violations
  - Session 2: load Session 1 memory, apply carry_check template, measure FP reduction
  - Session 3: load Session 2 memory, measure further improvement or plateau
- honest_verdict='cross_session_improvement' if Session 2 < Session 1 FP rate
- This is the Tier 2 continuous self-learning validation the system has never done end-to-end

### Retrospective

**Exp 449:** Operational Retrospective — Milestone 2026.04.33
- Standard retrospective following Exp 424/436 pattern
- Headline question: "Did we FINALLY get live GPU benchmark numbers?"
- RETRO items to close: RETRO-026 (long-running executor), RETRO-027 (silent drop detection)
- New RETRO items if any

---

## Dependency Graph

```
Exp 437 (LongRunBenchmarkExecutor)
    ↓ enables
Exp 439 (Live Precision Micro) ← Exp 438 (GPU1 Fix, informational)
    ↓ produces live CoT data
Exp 442 (FOVER Live Annotation)
    ↓ produces real training pairs
Exp 443 (EORM+JEPA Live Retrain)

Exp 440 (Live HumanEval) — independent of Exp 439
Exp 441 (Live Adversarial GSM8K) — independent

Exp 444 (CarnotThinkProbe) — CPU-only, independent
Exp 445 (BoltzmannRepairBridge) — CPU-only, independent
Exp 446 (Energy Matching ContinuousEBM) — Phase 3, independent
Exp 447 (KAEMEnergy) — CPU-only, independent
Exp 448 (Cross-Session Memory Relay) — CPU-only, independent

Exp 449 (Retrospective) — depends on all prior experiments
```

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Exp 437 (LongRunBenchmarkExecutor) | CPU only | Infrastructure — always runs |
| Exp 438 (GPU1 Zombie Fix) | GPU optional | Diagnosis runs on CPU; fix attempt needs GPU |
| Exp 439 (Live Precision) | 2x RTX 3090 | Gemma4-E4B-it + Qwen3.5-0.8B simultaneously |
| Exp 440 (Live HumanEval) | 2x RTX 3090 | Code generation + repair |
| Exp 441 (Live Adversarial) | 2x RTX 3090 | Standard + adversarial conditions |
| Exp 442 (FOVER Annotation) | CPU only | Z3 is CPU-native; loads Exp 439 JSON |
| Exp 443 (EORM+JEPA Retrain) | CPU only | JAX CD training on CPU |
| Exp 444 (CarnotThinkProbe) | CPU stub / GPU optional | Qwen3.5-0.8B for live; CI stub for tests |
| Exp 445 (BoltzmannRepairBridge) | CPU only | Embedding projection math |
| Exp 446 (Energy Matching) | CPU only | Phase 3 seed; JAX gradient flow |
| Exp 447 (KAEMEnergy) | CPU only | Exact sampling; no MCMC |
| Exp 448 (Cross-Session Memory) | CPU only | In-memory session simulation |
| Exp 449 (Retrospective) | CPU only | Always runs |

**KV260 FPGA:** Bitfile needed for bring-up (see research-hardware-wishlist.md). Deferred until
bitfile is built/flashed. Estimated arrival of working bitfile: human action required.

**AMD XDNA NPU:** Install `sudo pacman -S ninja && sudo pacman -S openblas` to unblock.
Alternatively: `pip install mlir-aie` for IRON toolchain path. 5th consecutive milestone carry.

---

## Success Criteria for Milestone 2026.04.33

| Criterion | Target | Notes |
|-----------|--------|-------|
| retro_026_resolved | True | LongRunBenchmarkExecutor implemented |
| retro_027_resolved | True | Silent experiment drop detection added |
| retro_025_resolved | True or partial | GPU1 zombie fix attempted with honest verdict |
| live_precision_result | honest_verdict in (live_improvement, live_no_improvement) | 50q micro-batch |
| live_humaneval_result | honest_verdict in (code_verification_positive, code_no_improvement) | 50 problems |
| live_adversarial_result | honest_verdict in (improvement_positive, neutral, degradation_positive) | Not 'blocked' |
| fover_real_data_labeled | honest_verdict='real_data_labeled' | Needs Exp 439 live data |
| fr11_relay_confirmed | retro_024_closed=True | Needs Exp 442 real pairs |
| think_probe_viable | skip_rate > 0.15 AND tp_rate > 0.70 | ThinkPRM pre-filter |
| boltzmann_repair_works | repair_energy_improvement > 0 | DBM-guided repair better than random |
| continuous_energy_improved | L2 < 0.5 (vs 2.69 in Exp 435a) | Phase 3 seed improvement |
| kaem_faster | latency_ratio < 0.1 (KAEM 10x faster than Ising MCMC) | Exact sampling speedup |
| cross_session_improvement | session2_fp < session1_fp | Tier 2 continuous learning |

---

## Key Papers for This Milestone

| Paper | ArXiv | Experiment | Key Insight |
|-------|-------|-----------|-------------|
| ThinkPRM | 2504.16828 | Exp 444 | CoT verification pre-filter; 1% supervision labels |
| Boltzmann-GPT | 2601.17094 | Exp 445 | DBM → LLM adapter; energy-guided repair direction |
| Energy Matching | 2504.10612 | Exp 446 | Flow + EBM; continuous reasoning generation |
| Generative Thermo Computing | 2506.15121 | Exp 446 | Langevin from free energy; thermodynamic sampling |
| KAEM | 2506.14167 | Exp 447 | KAN exact inverse-transform; eliminates MCMC |
| RLVR | 2506.14245 | FR-11 background | Verifiable rewards teach correct reasoning |
| Apple GSM8K | 2410.05229 | Exp 441 | Irrelevant-sentence injection; Carnot's thesis |
| VPRM | 2601.17223 | Exp 442/443 | Verifiable step-level PRM via symbolic checking |
| Process Reward Agents | 2604.09482 | Roadmap | Decoupled step reward; IsingEBM as PRA |
