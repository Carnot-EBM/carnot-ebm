# Carnot Research Roadmap v26: FPGA Hardware Bring-up, Apple Adversarial GSM8K, and Spilled-Energy Hallucination Detection

**Created:** 2026-04-14
**Milestone:** 2026.04.20
**Status:** Active (milestone 2026.04.19 complete)
**Supersedes:** Milestone 2026.04.19 — "GPU-Accelerated Inference, Calibrated Verification, and Live Benchmark Completion"
**Informed by:** Exp 258-270, operational retrospectives 2026-04-13 (×2), Exp 259 CUDA ORT inversion finding, Exp 260 cpu_only_blocked result, v25 pre-plan
**External inputs:** Spilled Energy (2602.18671, ICLR 2026), Semantic Energy (2508.14496), FactNet (2602.03417), Quantum-FPGA Ising (2604.04606), LagONN (2505.07179), Apple adversarial (2410.05229), KAEM (2506.14167), Hybrid FPGA Decomposition (2602.15985), VERGE (2601.20055), Z3 Policy Verification (2603.20449), Probabilistic NS Layer (2503.19466), Conformal LLM (2603.22966), Denoising Thermodynamic (2510.23972), KANELÉ (2512.12850), EBM-CoT (2511.07124)

---

## What 2026.04.19 Proved

| Approach | Experiments | Finding |
|----------|-------------|---------|
| DualGPURunner wiring | 258 | Harness built and tested (35 tests), but only 2 of 280 experiments (0.7%) actually invoked it — both RTX 3090s idle for 99% of the milestone. |
| CUDA ORT for PredictiveVerifier | 259 | **INVERTED:** CUDA ORT is 5.49× SLOWER than CPU ORT for the 9→1 gate at batch_size=1 (47.3 µs vs 8.6 µs). Kernel launch overhead dominates. GPU advantage only appears at batch_size ≥ 32. |
| Solver-routed semantic benchmark | 260 | **Still blocked:** `inference_mode: "cpu_only_blocked"`. Despite DualGPURunner available, benchmark ran on CPU only — 180s stall timeout killed every long inference experiment. |
| Calibration corpus | 262 | 450 rows collected, but prefix_fraction feature importance ≈ 0.507 (near-random). CPU-inference token patterns don't predict violations. Corpus not discriminative for calibration. |
| Self-learning replay v3 | 266 | Deliverable shows Exp 255 metadata — pass-through of prior work because Exp 263 and 264 both SKIP'd. Constraint addition still uses description-text proxies. |
| HuggingFace README updates | 267 | **Completed:** 16 per-token EBM model READMEs updated with Phase 1 status banner. |
| AMD XDNA NPU | 269 | **Failed 3 times** with 180s silence stall. VitisAI EP `.so` files present but onnxruntime handshake blocked. |
| HF joint model publish | 268 | SKIP'd 3 times — carry-forward to Exp 282. |

**Milestone-level conclusion:** Every infrastructure layer is now built (DualGPU, CUDA ORT, formal claim verifier, process integrity, predictive gate, FPGA design, calibration corpus). The gap is execution quality: long-running inference experiments stall at 180s because the GPU was never wired as the DEFAULT, and the self-learning calibration experiments were SKIP'd because their dependencies were also SKIP'd. The next milestone must wire GPU from experiment 1 and use per-question checkpointing on every experiment that touches live inference.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: No credible positive verify-repair result on real GPU inference after 260 experiments

The core product claim — "EBM-verified LLM outputs, structurally correct by design" — has no supporting evidence from a complete live benchmark. Exp 260 `inference_mode: "cpu_only_blocked"` is the symptom. The Apple adversarial GSM8K benchmark (arXiv 2410.05229) has been identified as the most credible possible demonstration since Exp 139 research scan, and still has never been run. This is the single most important outstanding experiment.

### Gap 2: FPGA hardware path deferred 2 milestones — KV260 is in hand with no bitfile

Exp 228 (software design), Exp 242 (blocker artifact, no bitfile) — both completed. The quantum-inspired sparse Ising paper (arXiv 2604.04606) provides a directly applicable design. KANELÉ (arXiv 2512.12850) shows KANs on FPGAs are hardware-efficient via LUT evaluation. The hardware path is the long-term differentiator (TSU abstraction, Tier 2 FPGA pattern matching). Every milestone deferred costs real research time.

### Gap 3: Spilled energy is a fast, extraction-free hallucination signal that doesn't require regex or SMT

Spilled Energy (ICLR 2026, arXiv 2602.18671) reframes autoregressive LLMs as EBMs and detects factual errors from logit discrepancy alone — no extraction required. This directly addresses the bottleneck that has blocked every positive result: ArithmeticExtractor found 0 violations on Gemma4-E4B-it, FormalClaimVerifier abstains on "not_formalizable" claims. Spilled energy reads from the model's own logits. Semantic Energy (arXiv 2508.14496) catches a complementary class of confident-but-wrong errors. Together they form a fast, extraction-free first-pass filter. EBM-CoT (arXiv 2511.07124) shows energy-based calibration improves chain-of-thought consistency — directly applicable to the JEPA self-learning tier.

---

## Promising 2025-2026 Inputs Adopted in v26

- **Spilled Energy (2602.18671, ICLR 2026):** Detects hallucinations via logit_energy − output_energy discrepancy. No extraction required. Fast first-pass signal before Ising verification. Directly informs Exp 274.
- **Semantic Energy (2508.14496):** Boltzmann energy from logit distributions; catches confident-but-wrong outputs that spilled energy misses. Complementary signal. Informs Exp 275.
- **FactNet (2602.03417):** 1.7B atomic assertions, 92.1% grounding precision. Each triple maps to a Carnot ConstraintTerm. Informs Exp 276.
- **Quantum-inspired FPGA Ising (2604.04606):** Sparse spin connectivity, 6× faster than simulated annealing, 1600-spin vs 400-spin baseline. Directly applicable to KV260 FpgaBackend. Informs Exp 278.
- **LagONN (2505.07179):** Lagrangian oscillatory NNs escape infeasible local minima in Ising problems. Addresses Exp 46b failure mode. Informs FPGA backend.
- **KANELÉ (2512.12850):** KANs on FPGAs via LUT evaluation. Hardware path for `carnot-kan` tier. Informs KV260 FpgaBackend extension for KAN energy.
- **Denoising Thermodynamic (2510.23972):** DTM architecture more hardware-efficient than raw EBMs for FPGA. Alternative to pure Ising for KV260.
- **EBM-CoT (2511.07124):** Energy-based calibration for implicit chain-of-thought. Informs JEPA training design (Exp 280).
- **VERGE (2601.20055):** Z3-based iterative LLM reasoning refinement. Feedback loop architecture informs Carnot's self-learning design.
- **Z3 Policy Verification (2603.20449):** NL→SMT-LIB-2.0 for tool-call constraints using Z3. Addresses constraint extraction bottleneck directly.
- **Conformal LLM (2603.22966):** Set-valued prediction with statistical coverage guarantees. Informs PredictiveVerifier threshold calibration.
- **Apple adversarial GSM8K (2410.05229):** 65% accuracy drop with irrelevant sentences + number swaps. "This is our thesis" (research-program.md). Still unrun.

---

## v26 Hypothesis

If Carnot (1) wires DualGPURunner as the DEFAULT from experiment 1, (2) runs the Apple adversarial GSM8K benchmark with live GPU inference, (3) implements SpilledEnergyExtractor as an extraction-free hallucination signal, (4) brings up the KV260 FPGA overlay for the first real hardware Ising sample, and (5) trains a JEPA predictor on GPU-inference calibration data and deploys it as a calibrated Tier 3 gate, then the milestone will produce:

1. The first **external-benchmark-backed evidence** that Carnot catches errors that break all other approaches
2. A **logit-based hallucination signal** that works on any IT model without regex or SMT extraction
3. The first **real FPGA Ising sample**, validated against the CPU baseline
4. A **calibrated JEPA Tier 3 gate** with measured fast-path hit rate and TP/FP rates on real GPU inference

---

## v26 Architecture: GPU-First → Fast Signal → FPGA Hardware

```
Benchmark Item (Apple adversarial GSM8K, 1,319 problems)
      |
      v
┌──────────────────────────────────────────────────────────────────┐
│ DualGPURunner — DEFAULT from Exp 271                             │
│  Qwen → GPU 0 (RTX 3090 #0)  |  Gemma → GPU 1 (RTX 3090 #1)    │
│  Batched inference 8-16/pass, per-question checkpointing         │
│  180s hard timeout per inference call (emit partial, not stall)  │
│  Target: ≤ 3s/case (validated in Exp 258 harness)               │
└───────────────────────────────────┬──────────────────────────────┘
                                    |
                    ┌───────────────┴──────────────────┐
                    v                                  v
┌───────────────────────────┐          ┌───────────────────────────┐
│ SpilledEnergyExtractor    │          │ SemanticEnergyExtractor    │
│  (Exp 274)                │          │  (Exp 275)                 │
│  logit_energy-output_energy│         │  Boltzmann logit energy    │
│  Fast: no KB, no regex    │          │  Catches confident+wrong   │
│  AUROC target: ≥ 0.65     │          │  Complementary to spill    │
└──────────────┬────────────┘          └──────────────┬────────────┘
               └────────────────┬──────────────────────┘
                                v
┌──────────────────────────────────────────────────────────────────┐
│ FormalClaimVerifier (existing) — only invoked when fast gate fires│
│  Route: arithmetic / cardinality / boolean-entailment / abstain  │
└───────────────────────────────────┬──────────────────────────────┘
                                    |
                      Apple adversarial result (Exp 273)
                      Spilled energy benchmark (Exp 276)

Hardware Path:                 Self-Learning Path:
KV260 + PYNQ                  JEPA training on GPU data
      |                              |
FpgaBackend (Exp 278)         Exp 280 (calibrated gate)
      |                              |
FPGA Ising benchmark          A/B held-out validation
(Exp 279)                     Tier 3 fast-path ≥ 30%
```

---

## Phase 87: GPU Benchmark Completion — Deferred Carry-Forwards (Experiments 271-273)

**Process mandate for this phase:** The 180s stall timeout killed Exp 261-269. Every experiment that touches live inference MUST: (a) wire DualGPURunner at the start (not as a separate harness setup step), (b) checkpoint every 10 questions, (c) emit a partial artifact with `stall_at` field on timeout rather than blocking the conductor indefinitely.

### Exp 271: Full 164-problem HumanEval with Qwen via GPU (carry-forward from Exp 261)

**Deliverable:** `results/experiment_271_results.json`

Complete the full 164-problem HumanEval benchmark for Qwen/Qwen3.5-0.8B using the GPU harness from Exp 258. Carry-forward from Exp 261 (failed 3 times with 180s stalls). Wire DualGPURunner at the start of this experiment, not as a harness-setup prerequisite. Use the PBT + spec-aware + process-aware verifier stack (Exp 250). Checkpoint every 10 problems. If inference stalls on any problem, emit a partial honest artifact with `completed_problems`, `stall_at_problem_idx`, and the completed results. Do not stall the conductor. Report pass@1 at each stage: baseline, pbt_verify_only, spec_aware_verify_only, verify_repair. Include cross-model comparison against Exp 226 (Gemma full 164-problem). Use Qwen/Qwen3.5-0.8B and google/gemma-4-E4B-it on GPU only — CPU fallback is permitted if VRAM unavailable but must be labeled `inference_mode: "cpu_fallback"` in the artifact.

### Exp 272: Apple adversarial GSM8K dataset preparation and GPU baseline

**Deliverable:** `data/research/gsm8k_adversarial_272.jsonl`

Implement the Apple adversarial GSM8K generator (arXiv 2410.05229): (1) number substitution — same logical structure, swapped operands; (2) irrelevant-sentence injection — one contextually plausible but mathematically irrelevant sentence inserted. Create adversarial variants of the 200 Exp 219 cohort questions. Run GPU baseline (no verification) on both variants with Qwen3.5-0.8B and Gemma4-E4B-it via the Exp 258 GPU harness. Checkpoint every 10 questions. Deliverable is the JSONL corpus of adversarial variants plus the baseline accuracy numbers. Primary check: does adding an irrelevant sentence cause ≥15pp accuracy drop, replicating Apple's finding? Report both adversarial variant types separately.

### Exp 273: Apple adversarial GSM8K with verify-repair — the credibility benchmark

**Deliverable:** `results/experiment_273_results.json`

Run the full verify-repair pipeline on the Exp 272 adversarial corpus. Test both adversarial variant types × both models × all three modes (baseline, verify_only, verify_repair). Wire DualGPURunner from the start. Checkpoint every 10 questions. Primary hypothesis (research-program.md): Carnot's verify-repair improvement should be LARGER on adversarial variants than on standard GSM8K (Exp 260), because adversarial variants contain more real arithmetic errors for Ising to catch. Secondary hypothesis: the irrelevant-sentence variant shows less verify-only degradation because FormalClaimVerifier extracts only verifiable arithmetic claims and ignores irrelevant context. Report: accuracy delta adversarial vs standard per mode, whether verify-repair improvement is larger on adversarial (primary criterion), and comparison against Exp 260. This is the single most important credibility experiment in Carnot's research history.

---

## Phase 88: Spilled Energy and Semantic Energy Hallucination Detection (Experiments 274-276)

The ICLR 2026 paper (arXiv 2602.18671) detects hallucinations via "spilled energy" — the discrepancy between logit energy (pre-softmax) and output energy (post-softmax) — without requiring any external knowledge base or regex extraction. Factually incorrect outputs have systematically higher spilled energy because the model distributes probability mass across incorrect alternatives. Semantic Energy (arXiv 2508.14496) catches a complementary class: confident-but-wrong outputs where entropy is low but the top prediction is wrong.

Both signals read directly from LLM generation logits. Together they form a two-stage fast filter: spilled energy catches uncertain outputs, semantic energy catches overconfident wrong outputs. Only when either gate fires does the expensive FormalClaimVerifier run. This directly addresses the extraction bottleneck: no regex, no SMT, no KB required.

### Exp 274: SpilledEnergyExtractor implementation

**Deliverable:** `python/carnot/pipeline/spilled_energy_extractor.py`

Implement `SpilledEnergyExtractor` as an additive ConstraintExtractor that reads from LLM generation logits. Compute per-token spilled energy = sum(softmax_probs * log_softmax_probs) − max(log_softmax_probs): this is the discrepancy between entropy and max-logit energy. Aggregate over response tokens: mean, max, p95. Expose as `SpilledEnergyResult` with: per_token_spilled (list), mean_spilled, max_spilled, p95_spilled, suspected_hallucination (bool, threshold configurable). Add `verify_spilled_energy()` method to `VerifyRepairPipeline` as opt-in entry point — does not replace existing `verify()` path. Write tests first covering spilled energy computation, thresholding, pipeline integration, edge cases (empty, single-token, uniform logits, saved logits from file). 100% targeted module coverage required. References: arXiv 2602.18671.

### Exp 275: SemanticEnergyExtractor implementation

**Deliverable:** `python/carnot/pipeline/semantic_energy_extractor.py`

Implement `SemanticEnergyExtractor` based on Boltzmann energy from logit distributions (arXiv 2508.14496). The semantic energy is: E_semantic = -log sum_i(exp(logit_i / T)) where T is a temperature parameter. High semantic energy = the model's top prediction is anomalously confident relative to context. Combine with spilled energy in a `DualEnergyGate` that fires if EITHER signal exceeds its threshold. Add `verify_dual_energy()` to `VerifyRepairPipeline`. Benchmark both extractors on the Exp 273 adversarial and Exp 260 standard results (retrospectively from saved logits). Report: AUROC for each signal separately and combined, precision/recall at optimal threshold, and which error categories each catches that FormalClaimVerifier misses. Key question: does the dual-energy gate identify cases where FormalClaimVerifier abstains? Write tests first. 100% coverage.

### Exp 276: FactNet knowledge-grounded constraint extraction prototype

**Deliverable:** `python/carnot/pipeline/factnet_extractor.py`

Implement `FactNetExtractor` prototype querying a local slice of FactNet (arXiv 2602.03417, 1.7B atomic assertions, 92.1% grounding precision). Use the top 10K most-cited triples for common GSM8K entities (numeric values, units, math facts). Each (subject, predicate, object) triple becomes a Carnot `ConstraintTerm`. Implement `verify_factual_claims(response_text) → List[ConstraintTerm]` using entity extraction + local KB lookup. Write tests covering triple parsing, entity extraction, constraint generation, pipeline integration. Research prototype: report precision on a 30-case sample from Exp 273 adversarial corpus; production-quality recall not required. Label all results `prototype_quality: true`.

---

## Phase 89: FPGA Hardware Bring-up (Experiments 277-279)

The Kria KV260 was ordered in milestone 2026.04.18, arrived, and has been deferred from two prior milestones. Exp 228 designed the 4096-spin sparse Ising sampler and AXI-Lite register map. Exp 242 produced a blocker artifact (no bitfile configured). This phase closes the hardware gap.

The quantum-inspired sparse Ising paper (arXiv 2604.04606) provides a directly applicable design: sparse connectivity matching Carnot's clause-graph masking (Exp 61), quantum-inspired annealing schedule (6× faster than simulated annealing, 4× scale increase to 1600 spins). KANELÉ (arXiv 2512.12850) shows KAN splines are hardware-efficient via FPGA LUT evaluation — relevant to a future `carnot-kan` hardware tier. Denoising Thermodynamic Models (arXiv 2510.23972) suggest an alternative to pure Ising that may be more FPGA-efficient.

### Exp 277: KV260 FPGA overlay bring-up validation

**Deliverable:** `results/experiment_277_results.json`

Attempt KV260 FPGA overlay bring-up. Steps: (1) configure `CARNOT_KV260_BITFILE` to the Carnot Ising bitstream (if synthesized) or PYNQ base overlay (to validate the stack); (2) load overlay via PYNQ Python API; (3) exercise the AXI-Lite register map (write/read coupling matrix fields, verify round-trip latency); (4) trigger one Ising Gibbs sweep and readback sampled spin state. Report: overlay load latency, register round-trip latency, whether sampled state is valid (all spins ∈ {+1,-1}), and execution_path ("hardware", "software_model", or "blocked"). If bitstream not yet synthesized, use PYNQ base overlay to validate the stack and report as "software_model" with exact next steps. Do NOT fabricate hardware timing numbers. Do NOT stall: set a 60s timeout and emit blocker on timeout.

### Exp 278: FpgaBackend implementation with quantum-inspired sparse Ising schedule

**Deliverable:** `python/carnot/samplers/fpga_backend.py`

Implement `FpgaBackend` as a concrete `SamplerBackend` (Exp 71 protocol). Steps: (1) load coupling matrix J and bias vector h from an `IsingEBM` instance; (2) quantize to KV260 Q8.8 fixed-point (from Exp 228 design); (3) apply quantum-inspired sparse connectivity (arXiv 2604.04606): keep top-K couplings by magnitude, matching Exp 61 clause-graph masking; (4) serialize to AXI-Lite register map schema from Exp 228; (5) if CARNOT_KV260_BITFILE set, send over PYNQ AXI and readback; else invoke `ParallelIsingSampler` as software-model fallback. Include optional `use_lagrangian_penalty` flag (LagONN, arXiv 2505.07179) to escape infeasible local minima. Write tests first for quantization, sparse connectivity, register serialization, hardware/software dispatch, and LagONN penalty. 100% targeted module coverage. Note the KANELÉ approach (arXiv 2512.12850) as future extension comment in the code (KAN LUT evaluation for the `carnot-kan` tier).

### Exp 279: FPGA vs CPU Ising benchmark (hardware or software-model)

**Deliverable:** `results/experiment_279_results.json`

Benchmark `FpgaBackend` against `ParallelIsingSampler` (CPU) on three problem sizes: 100 spins, 500 spins, 1000 spins. For each: measure samples/second, energy convergence quality (final energy vs ground truth), and whether the quantum-inspired sparse schedule improves convergence vs dense CPU sampling. If KV260 hardware available (CARNOT_KV260_BITFILE set): report hardware latency explicitly labeled `execution_path: "hardware"`. Otherwise: software-model timing labeled `execution_path: "software_model"`. Compare against Exp 228 software-model baseline. Report whether the sparse Ising design from arXiv 2604.04606 reproduces its claimed 6× speedup in software simulation. Include a LagONN penalty convergence comparison (with vs without penalty on a known-infeasible problem). Do NOT stall: 60s timeout per benchmark configuration.

---

## Phase 90: JEPA Self-Training, NPU, HuggingFace Publishing, and Retro (Experiments 280-283)

### Exp 280: JEPA predictor training on GPU-inference data — Tier 3 self-learning

**Deliverable:** `results/experiment_280_results.json`

**This is the required continuous self-learning experiment for this milestone.** Train the JEPA predictor on GPU-inference calibration data from the Exp 273 Apple adversarial run (logits must be saved during Exp 273 at 25/50/75% prefix fractions, enriched with Exp 273's adversarial structure which is more violation-dense than standard GSM8K). The Exp 262 corpus (CPU inference, near-random feature importance) is insufficient. New data from Exp 273 GPU inference should be more discriminative because: (a) GPU inference is faster and more consistent; (b) adversarial variants have higher violation density; (c) richer token features (spilled energy and semantic energy from Exp 274-275 added as features).

Implement via EBM-CoT approach (arXiv 2511.07124): train an EBM that refines latent thought representations. Apply isotonic regression calibration. Target operating zone: fast-path hit rate ≥ 30%, true-violation detection rate ≥ 60%, FP rate ≤ 20% (4/δ bound framework, arXiv 2512.02080). Deploy calibrated gate as default `PredictiveVerifier`. Run a 50-case A/B test on Exp 273 held-out questions: calibrated vs uncalibrated gate. Apply conformal coverage bounds (arXiv 2603.22966) to report statistically valid confidence intervals on the gate's precision/recall. Report: calibrated fast-path rate, TP/FP rates, net accuracy delta on A/B held-out, and conformal coverage intervals. State clearly if fast-path ≥ 30% is not achieved.

### Exp 281: AMD XDNA NPU enablement — onnxruntime source build approach

**Deliverable:** `results/experiment_281_results.json`

Third and final structured attempt at AMD XDNA NPU (previous 3 all stalled at 180s). Take a different approach: build onnxruntime 1.20.1 from source with `-Donnxruntime_USE_VITISAI=ON` using the VitisAI EP source in `~/github.com/amd/RyzenAI-SW/`. Steps: (1) install cmake, ninja, openblas build dependencies; (2) configure onnxruntime 1.20.1 with VitisAI EP; (3) build in `.venv-npu/` Python 3.12; (4) if build succeeds, load `results/jepa_predictor_146.onnx` with VitisAIExecutionProvider and benchmark vs CPU ORT (8.6 µs Exp 257 baseline). **Hard constraint: 45-minute build timeout.** If exceeded, emit blocker artifact immediately with exact step, error log, and next action. Do NOT let the build stall the conductor silently.

### Exp 282: Publish Exp 66 joint model and FormalClaimVerifier to HuggingFace

**Deliverable:** `results/experiment_282_results.json`

Carry-forward from Exp 268 (SKIP'd 3 times). Publish to huggingface.co/Carnot-EBM: (1) the Exp 66 differentiable constraint model (embedding + Ising → score, 1.0 AUROC) as `safetensors` with README: "proof-of-concept, not production quality"; (2) the FormalClaimVerifier as ONNX (arithmetic + comparison routes) + Python bundle (remaining routes), README explaining solver routing, abstention policy, standalone usage. Tag as release `v0.2.0-research`. Use `huggingface_hub` Python library. Log HF artifact URLs and push status in results artifact. If credentials not configured, emit clear blocker with `huggingface-cli login` instructions. Do NOT stall silently — check credentials first.

### Exp 283: Operational retrospective for milestone 2026.04.20

**Deliverable:** `results/operational_retro_2026_04_20.json`

Generate the process efficiency analysis for milestone 2026.04.20. Specifically audit whether the 2026.04.19 retro action items were resolved — the 2026.04.19 retro showed 100% carry-over from the 2026.04.18 retro (zero action items resolved). Track: (1) was DualGPURunner wired from Exp 1? (2) did per-question checkpointing prevent stall losses? (3) was CUDA ORT batch_size ≥ 32 tested? (4) did the Apple adversarial benchmark complete? (5) did the carry-over rate from 2026.04.19 retro improve below 100%? Report: total wall time, experiments/hour, GPU utilization per-experiment (not just milestone-end), and updated action items for 2026.04.21 with explicit "resolved/deferred/new" tracking.

---

## Phase Summary

| Phase | Experiments | Theme | Key Success Criterion |
|-------|-------------|-------|----------------------|
| 87 | 271-273 | GPU benchmark completion + Apple adversarial | Exp 273: adversarial verify-repair Δ > standard GSM8K Δ |
| 88 | 274-276 | Spilled energy + semantic energy + FactNet | Exp 275: dual-energy AUROC ≥ 0.65 on Exp 273 adversarial |
| 89 | 277-279 | FPGA hardware bring-up | Exp 277: first non-"blocked" KV260 execution_path result |
| 90 | 280-283 | Tier 3 JEPA + NPU + HuggingFace + retro | Exp 280: calibrated gate fast-path ≥ 30%, FP ≤ 20% |

---

## Hardware Requirements

| Hardware | Experiments | Status |
|----------|-------------|--------|
| 2× RTX 3090 (CUDA) | 271, 272, 273, 275, 276, 280 | Available; must wire from Exp 1, not just harness setup |
| Kria KV260 FPGA | 277, 278, 279 | Available; CARNOT_KV260_BITFILE path must be configured at milestone start |
| AMD XDNA NPU (kernel module loaded) | 281 | VitisAI EP .so present; needs onnxruntime source build with VitisAI |
| HuggingFace CLI + credentials | 282 | May need `huggingface-cli login` at milestone start |
| ONNX export (Exp 66) | 282 | Exp 66 safetensors artifact exists in results/ |

---

## Dependency Graph

```
Exp 271 (HumanEval Qwen GPU)          [GPU required, carry-forward]
Exp 272 (Apple adversarial dataset)   [GPU required, independent]
    └── Exp 273 (Apple adversarial + verify-repair, saves logits)
              ├── Exp 275 (dual-energy benchmark — uses Exp 273 logits)
              └── Exp 280 (JEPA training — uses Exp 273 GPU-inference data)
Exp 274 (SpilledEnergyExtractor) ─────── Exp 275 (dual-energy benchmark)
Exp 275 (SemanticEnergyExtractor) ───── Exp 275 (same deliverable, extend Exp 274)
Exp 276 (FactNetExtractor)             [independent prototype]
Exp 277 (KV260 bring-up) ─────────── Exp 278 (FpgaBackend) ── Exp 279 (benchmark)
Exp 281 (AMD XDNA NPU)                [independent, 45min build timeout]
Exp 282 (HF publish)                  [independent]
Exp 283 (retro)                       [depends on all prior]
```

---

## What This Milestone Does NOT Include

- **PredictiveVerifier CUDA ORT batched benchmark** — Exp 259 proved CUDA is slower at batch_size=1. A batched benchmark (batch_size ≥ 32) would require inference pipeline changes. Defer to 2026.04.21 when Tier 3 is proven useful.
- **DOMINO grammar-constrained generation** — long-term path, depends on completing the current autoregressive benchmark first.
- **Conformal prediction (TECP)** — incorporated into Exp 280 as coverage bounds; not a separate experiment.
- **ΠNet projection repair** — valid future direction (arXiv 2508.10480); Langevin repair sufficient for now.
- **Z3/SMT constraint extraction (VERGE, NSVIF)** — deferred to 2026.04.21; spilled energy is a faster win for the extraction problem and must be proven first.
- **Contrastive Decoding** — deferred; no benchmark infrastructure for it yet.

---

## Key Lessons from 2026.04.19 to Apply Immediately

1. **Wire DualGPURunner from Exp 1** — not as a separate harness setup step. The retro showed 99% GPU idle.
2. **Per-question checkpointing on every long benchmark** — stall at 180s killed Exp 261-269. Checkpoint every 10 questions, resume gracefully.
3. **Hard timeout on build/inference** — Exp 281 NPU build: 45-minute timeout; emit blocker immediately.
4. **GPU cleanup hook between experiments** — `torch.cuda.empty_cache()` after every experiment that loads a model.
5. **Credential check at milestone start** — Exp 282 HF publish: check `huggingface-cli whoami` before attempting any uploads.
6. **Save logits during Exp 273** — SpilledEnergyExtractor (Exp 274) and JEPA training (Exp 280) both need logit data from Exp 273. This must be explicit in the Exp 273 prompt.
