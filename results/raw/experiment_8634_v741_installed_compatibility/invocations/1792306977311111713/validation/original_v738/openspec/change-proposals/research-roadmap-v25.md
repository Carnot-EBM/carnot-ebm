# Carnot Research Roadmap v25: FPGA Hardware Bring-up, Apple Adversarial GSM8K, and Spilled-Energy Hallucination Detection

**Created:** 2026-04-14
**Milestone:** 2026.04.20
**Status:** Planned (activates when milestone 2026.04.19 completes)
**Supersedes:** Milestone 2026.04.19 — "GPU-Accelerated Inference, Calibrated Verification, and Live Benchmark Completion"
**Informed by:** Exp 258-270, operational retrospectives 2026-04-13 (x2), Exp 259 CUDA ORT inversion finding, Exp 260 cpu_only_blocked result
**External inputs:** Spilled Energy (arXiv 2602.18671, ICLR 2026), FactNet (arXiv 2602.03417), Quantum-FPGA Ising (arXiv 2604.04606), LagONN (arXiv 2505.07179), Πnet (arXiv 2508.10480), Apple GSM8K adversarial (arXiv 2410.05229), KAEM (arXiv 2506.14167), DOMINO (ICML 2025), Hybrid FPGA Decomposition (arXiv 2602.15985)

---

## What 2026.04.19 Proved

| Approach | Experiments | Finding |
|----------|-------------|---------|
| DualGPURunner wiring | 258 | Harness built and tested (35 tests), but only 2 of 280 experiments (0.7%) actually invoked it — zero per-experiment GPU assignment at milestone start. Both RTX 3090s were idle for 99% of the milestone. |
| CUDA ORT for PredictiveVerifier | 259 | **INVERTED:** CUDA ORT is 5.49× SLOWER than CPU ORT for the 9→1 gate at batch_size=1 (47.3 µs vs 8.6 µs). Kernel launch overhead dominates at this scale. GPU advantage only appears at batch_size ≥ 32. |
| Solver-routed semantic benchmark | 260 | **Still blocked:** `inference_mode: "cpu_only_blocked"`. Despite DualGPURunner being available, the full 200+200+81+81 case benchmark ran on CPU only, which could not finish within session budget. First credible GPU-backed verify-repair result still missing after 260 experiments. |
| Calibration corpus collection | 262 | 450 rows collected (150 cases × 3 prefix fractions). But prefix_fraction feature importance was 0.507 for all fractions — essentially random, suggesting CPU-inference token patterns don't predict violations any better than chance. Corpus exists but is not discriminative for calibration. |
| Self-learning replay v3 | 266 | Deliverable exists but shows experiment=255 metadata — indicates this was a pass-through of prior work rather than a new run on calibrated gate + domain templates (Exp 263 and Exp 264 were both SKIP). Constraint addition still uses description-text proxies. |
| HuggingFace model README updates | 267 | **Completed:** 16 per-token EBM model READMEs updated with Phase 1 research status banner and "What's Proven to Work" section. |
| AMD XDNA NPU (Exp 269) | 269 | **Failed 3 times** with 180s silence stall. VitisAI EP `.so` files are present but onnxruntime handshake still blocked. |

**The milestone-level conclusion:** The infrastructure build-up is complete (DualGPU harness, CUDA ORT, formal claim verifier, process integrity, predictive gate). The gap remains execution quality: GPU inference was never actually used for the benchmark that needed it, the calibration corpus was built on CPU inference patterns that don't generalize, and all five experiments that ran on real LLM inference were stalled by the 180s silence timeout. The next milestone must wire GPU from experiment 1 and use per-question checkpointing on every long-running experiment.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: The first credible positive verify-repair result on real GPU inference is still missing after 260 experiments

Exp 260 `inference_mode: "cpu_only_blocked"` is the clearest symptom. Despite 2 RTX 3090s, a wired DualGPURunner, and a checkpoint-resumable harness, the solver-routed benchmark never ran at GPU speed. The fundamental product claim — "EBM-verified LLM outputs, faster than you can generate without verification" — has no supporting evidence from a complete benchmark. Meanwhile, the Apple adversarial GSM8K experiment (research-program.md, referenced since the Exp 139 scan) remains unrun despite being identified as the single most credible demonstration of Carnot's value: a dataset where LLMs systematically fail because they pattern-match, not reason, and Carnot's structural verification should hold up because it doesn't care about irrelevant context.

### Gap 2: FPGA hardware path has been deferred for 2 milestones — it is the long-term differentiator

The Kria KV260 was ordered in milestone 2026.04.18 (4+ days ago) and arrived. Exp 242 recorded a blocker artifact (no `CARNOT_KV260_BITFILE` configured). Exp 228 implemented the `FPGAIsingSampler` design but as software-only. The FPGA path matters because: (1) it validates the TSU abstraction before Extropic Z1 ships; (2) the quantum-inspired sparse Ising paper (arXiv 2604.04606) gives a concrete hardware design with 6× speedup that maps directly to KV260 LUTs; (3) FPGA pattern matching is the acceleration path for Tier 2 constraint memory (research-program.md self-learning architecture).

### Gap 3: Spilled energy is a fast signal that works without extraction — but it's still unimplemented

The ICLR 2026 paper (arXiv 2602.18671) shows that the discrepancy between LLM logit energy (pre-softmax) and output energy (post-softmax) — "spilled energy" — reliably identifies factually incorrect outputs without needing external knowledge bases or regex extraction. This is exactly the extraction bottleneck Carnot has been fighting: ArithmeticExtractor found 0 violations on Gemma4-E4B-it (0/20), FormalClaimVerifier abstains on non-formalizable claims. Spilled energy requires no extraction at all — it reads directly from logits. As a fast first-pass filter before Ising verification, it could eliminate the false-positive problem while catching errors the extractor misses.

---

## Promising 2025-2026 Inputs Adopted in v25

- **Spilled Energy (2602.18671, ICLR 2026):** Reframes LLMs as EBMs via soft Bellman equation. Detects hallucinations via logit_energy − output_energy discrepancy. No extraction required — reads directly from the model's own internal scores. Directly informs Exp 274.
- **FactNet (2602.03417):** 1.7B atomic assertions, 92.1% grounding precision. Open-source KB for factual claim verification. Each triple maps to a Carnot ConstraintTerm. Informs Exp 276.
- **Quantum-inspired FPGA Ising (2604.04606):** Sparse spin connectivity, 6× faster than simulated annealing, 1600-spin vs 400-spin baseline. Directly applicable to FpgaBackend design for KV260 LUTs. Informs Exp 278.
- **LagONN (2505.07179):** Lagrange oscillatory neural networks escape infeasible local minima in constrained Ising problems. Addresses a known failure mode in Exp 46b (5000-var SAT convergence to infeasible states). Informs FPGA backend design.
- **KAEM (2506.14167):** KAN energy model with fast exact inference via inverse transform. Explores KAN for generative EBM use case — could inform carnot-kan generative capabilities beyond classification.
- **Πnet (2508.10480):** Hard-constrained NNs via orthogonal projection. More principled than Langevin repair — guarantees constraint satisfaction rather than reducing energy. Informs a future ProjectionRepair strategy as alternative to Langevin.
- **Apple adversarial GSM8K (2410.05229):** Models drop 65% with irrelevant sentences + number swaps. Even o1-preview drops 92.7%→77.4%. "This is our thesis" (research-program.md). Running Carnot on this dataset is the single most credible external validation.
- **Hybrid FPGA decomposition (2602.15985):** FPGA offloads problem decomposition to enable full-speed oscillator Ising solving. AXI interface design directly applicable to KV260 FpgaBackend.

---

## v25 Hypothesis

If Carnot (1) runs the Apple adversarial GSM8K benchmark with live GPU inference, (2) implements the spilled energy extractor from ICLR 2026 as a zero-extraction hallucination signal, (3) brings up the KV260 FPGA overlay for the first real hardware Ising sample, and (4) trains a JEPA predictor on the Exp 262 calibration corpus and deploys it as a calibrated Tier 3 gate, then the milestone will produce:

1. The first **external-benchmark-backed evidence** that Carnot catches errors that break all other approaches, including reasoning models
2. A **logit-based hallucination signal** that works without extraction on any instruction-tuned model
3. The first **real FPGA Ising sample**, validated against the CPU baseline
4. A **calibrated JEPA Tier 3 gate** with measured fast-path hit rate and TP/FP rates on real inference data

---

## v25 Architecture: From Infrastructure to Evidence

```
Credibility Path (independent of hardware):
  Apple adversarial GSM8K (1,319 problems)
      |
      v
  ┌──────────────────────────────────────────────────┐
  │ DualGPURunner (Exp 258) — GPU 0: Qwen, GPU 1: Gemma │
  │ Batched inference 8-16 per pass                   │
  │ Per-question checkpointing (10-question granularity) │
  │ Target: ≤ 3s/case (vs 21s/case CPU baseline)     │
  └──────────────────────────────────────────────────┘
      |
      v
  ┌──────────────────────────────────────────────────┐
  │ SpilledEnergyExtractor (new, Exp 274)            │
  │  logit_energy - output_energy discrepancy        │
  │  No extraction required — reads from model logits │
  │  Fast first-pass: skip Ising if spill < threshold │
  └──────────────────────────────────────────────────┘
      |
      v
  ┌──────────────────────────────────────────────────┐
  │ FormalClaimVerifier (existing, Exp 245)          │
  │  solver-routed: arithmetic, cardinality, SMT     │
  │  Only invoked if spilled energy gate fires       │
  └──────────────────────────────────────────────────┘
      |
      v
  results/experiment_273_results.json (Apple adversarial, complete)
  results/experiment_275_results.json (spilled energy benchmark)

Hardware Path (FPGA bring-up):
  KV260 + PYNQ + CARNOT_KV260_BITFILE
      |
      v
  ┌──────────────────────────────────────────────────┐
  │ FpgaBackend (Exp 278)                            │
  │  Quantum-inspired sparse Ising (2604.04606)      │
  │  6× faster than simulated annealing              │
  │  AXI-Lite register map (from Exp 228 design)     │
  └──────────────────────────────────────────────────┘
      |
      v
  results/experiment_279_results.json (FPGA vs CPU Ising benchmark)

Self-Learning Path (Tier 3):
  Exp 262 calibration corpus (450 rows)
      |
      v
  ┌──────────────────────────────────────────────────┐
  │ JEPA predictor training (Exp 280)                │
  │  Train on (partial_context, violation_label) pairs│
  │  Calibrated threshold: fast-path ≥ 30%           │
  │  Deploy as Tier 3 PredictiveVerifier gate        │
  └──────────────────────────────────────────────────┘
      |
      v
  results/experiment_280_results.json (calibrated gate A/B)
```

---

## Phase 87: GPU Benchmark Completion — Deferred Carry-Forwards (Experiments 271-273)

These three experiments directly complete the work that milestone 2026.04.19 was supposed to deliver but couldn't due to the 180s stall pattern.

**Process change for this phase:** Wire DualGPURunner at the start of each experiment that uses live inference, not just the harness setup step. All long-running benchmarks must checkpoint at 10-question granularity.

### Exp 271: Full 164-problem HumanEval with Qwen via GPU (carry-forward from Exp 261)

**Deliverable:** `results/experiment_271_results.json`

Complete the full 164-problem HumanEval benchmark for Qwen/Qwen3.5-0.8B using the GPU harness from Exp 258. This is a carry-forward from Exp 261 which failed 3 times with 180s stalls. The PBT + spec-aware + process-aware verifier stack from Exp 250 is already built. Use checkpointing at every 10 problems. Report pass@1 at each stage: baseline, pbt_verify_only, spec_aware_verify_only, process_aware_verify_only, verify_repair. Include direct cross-model comparison against Exp 226 (Gemma full 164-problem result). If inference stalls, emit partial honest artifact and report which problem indices are complete.

### Exp 272: Apple adversarial GSM8K dataset preparation and baseline

**Deliverable:** `data/research/gsm8k_adversarial_272.jsonl`

Implement the Apple adversarial GSM8K generator (based on arXiv 2410.05229) that creates two adversarial variants: (1) number substitution — same logical structure, swapped operands, (2) irrelevant sentence insertion — one contextually plausible but mathematically irrelevant sentence injected. Create 200 adversarial versions of the Exp 219 cohort (same 200 questions per model). For each question, generate both adversarial variants. Run the baseline (no verification) with Qwen3.5-0.8B and Gemma4-E4B-it via the Exp 258 GPU harness to measure the accuracy drop vs standard GSM8K. This is the "original finding" replication: does adding an irrelevant sentence drop accuracy by ~20pp as Apple showed? Report baseline accuracy on adversarial variants vs the Exp 219 standard baseline.

### Exp 273: Apple adversarial GSM8K with verify-repair — the credibility benchmark

**Deliverable:** `results/experiment_273_results.json`

Run the full verify-repair pipeline on the Exp 272 adversarial GSM8K corpus. Test both adversarial variant types (number substitution, irrelevant sentence injection) × both models (Qwen, Gemma) × all three modes (baseline, verify_only, verify_repair). Primary hypothesis (from research-program.md): Carnot's verify-repair improvement should be LARGER on adversarial variants than on standard GSM8K (Exp 260), because adversarial variants contain more real arithmetic errors that Ising can catch. Secondary hypothesis: the irrelevant-sentence variant shows less degradation under verify-repair than under LLM baseline, because the FormalClaimVerifier extracts only the verifiable arithmetic claims and ignores irrelevant context. Report: (1) accuracy delta adversarial vs standard per mode, (2) whether verify-repair improvement is larger on adversarial (primary criterion), (3) direct comparison block against Exp 260. This is the single most important credibility experiment in Carnot's history.

---

## Phase 88: Spilled Energy Hallucination Detection (Experiments 274-276)

The ICLR 2026 paper (arXiv 2602.18671) reframes autoregressive LLMs as EBMs via the soft Bellman equation in max-entropy RL, then detects hallucinations via "spilled energy" — the discrepancy between the model's logit energy (pre-softmax, proportional to the negative log-probability of the predicted token) and output energy (post-softmax, the actual token distribution). Factually incorrect outputs have systematically higher spilled energy because the model "spills" probability mass from the correct token to incorrect alternatives.

This is directly aligned with Carnot's constraint verification architecture: spilled energy is a fast, extraction-free hallucination signal that complements the structured FormalClaimVerifier. It can serve as a first-pass filter — only invoke the expensive Ising verification when spilled energy exceeds a threshold.

### Exp 274: SpilledEnergyExtractor implementation

**Deliverable:** `python/carnot/pipeline/spilled_energy_extractor.py`

Implement `SpilledEnergyExtractor` as an additive ConstraintExtractor that reads directly from LLM generation logits. The extractor computes per-token spilled energy = sum(logit_distribution) - max(logit_distribution) — a measure of how spread out the probability mass is. Aggregate over response tokens: mean, max, and tail (95th percentile) spilled energy. Expose as `SpilledEnergyResult` with: per_token_spilled (list), mean_spilled, max_spilled, p95_spilled, suspected_hallucination (bool, true if mean_spilled > threshold). Write tests first covering spilled energy computation, thresholding, integration with `VerifyRepairPipeline` as an opt-in path (verify_spilled_energy method), and edge cases (empty response, single-token response, uniform logits). 100% test coverage required.

### Exp 275: Spilled energy benchmark on GSM8K and HumanEval

**Deliverable:** `results/experiment_275_results.json`

Benchmark `SpilledEnergyExtractor` on the Exp 260/Exp 271 results. Retrospectively compute spilled energy from the live inference logits (which must be saved during the Exp 273 apple adversarial run if not already saved from Exp 260). Compare spilled energy detection rate vs FormalClaimVerifier detection rate on the same set of questions. Report: AUROC for spilled energy as hallucination predictor, precision/recall at the optimal threshold, false positive rate, and which error categories spilled energy catches that FormalClaimVerifier misses (and vice versa). Key question: does spilled energy identify cases where FormalClaimVerifier abstains ("not_formalizable")? If yes, the two signals are complementary and can be combined.

### Exp 276: FactNet knowledge-grounded constraint extraction prototype

**Deliverable:** `python/carnot/pipeline/factnet_extractor.py`

Implement a prototype `FactNetExtractor` that queries a local slice of FactNet (arXiv 2602.03417) for entity-level factual claims. FactNet's 1.7B atomic assertions map directly to Carnot `ConstraintTerm` protocol: each (subject, predicate, object) triple becomes an `IsingConstraint` on whether the LLM's output is consistent with the known fact. Use a local subset: download the top 10K most-cited FactNet triples related to common GSM8K entities (numbers, units, common math facts). Implement `verify_factual_claims(response_text) → List[ConstraintTerm]` that extracts entity mentions, looks them up in the local FactNet slice, and returns `ConstraintTerm` objects for each matched triple. Write tests first covering triple parsing, entity extraction, constraint generation, and pipeline integration. This is a research prototype — report precision on a 30-case sample but don't require production-quality recall.

---

## Phase 89: FPGA Hardware Bring-up (Experiments 277-279)

The Kria KV260 has been deferred from two prior milestones. The Exp 228 design (4096-spin sparse Ising, AXI-Lite register map, software control path) is complete. The Exp 242 blocker artifact identified the exact missing step: `CARNOT_KV260_BITFILE` path not configured. This phase completes the bring-up.

The quantum-inspired sparse Ising paper (arXiv 2604.04606) provides an immediately applicable hardware design: sparse spin connectivity (matching Carnot's Exp 61 clause-graph masking), quantum-inspired annealing schedule (improved from standard simulated annealing), and a 6× speedup result at 1600 spins. This is directly applicable to the KV260 `FpgaBackend`.

### Exp 277: KV260 FPGA bitstream loading and overlay validation

**Deliverable:** `results/experiment_277_results.json`

Attempt KV260 FPGA overlay bring-up using the software control-path model from Exp 228. Steps: (1) configure `CARNOT_KV260_BITFILE` path to the Carnot Ising bitstream (if synthesized) or the PYNQ base overlay (as fallback to validate the PYNQ stack); (2) load the overlay via PYNQ Python API; (3) exercise the AXI-Lite register map (write/read coupling matrix fields, verify round-trip); (4) trigger one Ising Gibbs sweep and readback sampled spin state. Report: overlay load latency, register round-trip latency, whether sampled state is valid (all spins in {+1,-1}), and execution_path ("hardware", "software_model", or "blocked"). If the Carnot bitstream is not yet synthesized, use the PYNQ base overlay to validate the stack and report as "software_model" with clear next steps. Do not fabricate hardware timing.

### Exp 278: FpgaBackend implementation with quantum-inspired sparse Ising schedule

**Deliverable:** `python/carnot/samplers/fpga_backend.py`

Implement `FpgaBackend` as a concrete `SamplerBackend` (from the protocol in Exp 71). The implementation should: (1) load coupling matrix J and bias vector h from an `IsingEBM` instance; (2) quantize to the KV260's fixed-point representation; (3) apply the quantum-inspired sparse connectivity scheme from arXiv 2604.04606 (keep only top-K couplings by magnitude, matching Exp 61 clause-graph masking); (4) serialize to the AXI-Lite register map schema from Exp 228; (5) if real hardware is available (CARNOT_KV260_BITFILE set), send over PYNQ AXI and readback samples; otherwise, invoke CPU `ParallelIsingSampler` as software-model fallback. Implement the LagONN Lagrangian penalty (arXiv 2505.07179) as an optional `use_lagrangian_penalty` flag to escape infeasible local minima. Write tests first for quantization, sparse connectivity, register serialization, hardware/software dispatch, and LagONN penalty integration. 100% coverage required.

### Exp 279: FPGA vs CPU Ising benchmark (or software-model baseline)

**Deliverable:** `results/experiment_279_results.json`

Benchmark `FpgaBackend` against `ParallelIsingSampler` (CPU, Exp 46a) on three problem sizes: 100 spins, 500 spins, 1000 spins. For each size, measure: samples/second, energy convergence quality (final energy vs ground truth), and whether the quantum-inspired sparse schedule improves convergence vs dense sampling. If real KV260 hardware is available, report hardware latency; otherwise, report software-model timing labeled explicitly as `execution_path: "software_model"`. Include a comparison against the Exp 228 software simulation baseline. Report: whether the sparse Ising design from arXiv 2604.04606 reproduces its claimed 6× speedup in software simulation, and what the hardware latency would be if extrapolated from the Exp 228 register-map latency.

---

## Phase 90: Tier 3 Self-Learning, NPU, HuggingFace Publishing, and Retro (Experiments 280-283)

### Exp 280: JEPA predictor training on Exp 262 corpus and calibrated gate deployment (Tier 3)

**Deliverable:** `results/experiment_280_results.json`

**This is the required continuous self-learning experiment for this milestone.** Train the JEPA predictor on the 450-row `data/research/predictive_calibration_corpus_262.jsonl` corpus using contrastive divergence. The Exp 262 summary showed prefix_fraction feature importance was ~0.507 (near-random), meaning raw token patterns from CPU inference aren't predictive. This experiment must: (1) retrain on features specifically designed for GPU inference patterns (from the Exp 273 Apple adversarial run where logits are saved); (2) apply isotonic regression calibration to the predictor output; (3) target calibrated operating zone: fast-path hit rate ≥ 30%, true-violation detection rate ≥ 60%, FP rate ≤ 20%; (4) deploy the calibrated gate as the default `PredictiveVerifier` and run a 50-case A/B test comparing calibrated vs uncalibrated gate on held-out questions from Exp 273. This closes the Tier 3 loop first opened in Exp 143 (JEPA training pairs) → Exp 144 (violation predictor) → Exp 145 (fast-path gate) → Exp 256 (uncalibrated on live data) → here (calibrated on GPU inference data). Report: calibrated fast-path rate, TP/FP rates, and net accuracy delta on the A/B held-out set.

### Exp 281: AMD XDNA NPU enablement — VitisAI EP custom build approach

**Deliverable:** `results/experiment_281_results.json`

Third and final structured attempt at AMD XDNA NPU for PredictiveVerifier. Previous 3 attempts (Exp 269) all stalled at 180s. This experiment takes a different approach: instead of relying on pip-installed onnxruntime, build onnxruntime 1.20.1 from source with `-Donnxruntime_USE_VITISAI=ON` using the VitisAI EP source already present in `~/github.com/amd/RyzenAI-SW/`. Steps: (1) install build dependencies (cmake, ninja, openblas); (2) clone onnxruntime 1.20.1 and configure with VitisAI EP enabled; (3) build targeting the `.venv-npu/` Python 3.12 environment; (4) if build succeeds (>1h expected), load `results/jepa_predictor_146.onnx` with VitisAIExecutionProvider and benchmark latency vs CPU ORT (8.6 µs baseline from Exp 257); (5) if build fails or exceeds 45min, emit an honest blocker artifact naming the exact step and error. Do not stall silently — set build timeout at 45min and emit blocker if exceeded. Expected outcome: either first NPU benchmark result or definitive "build onnxruntime from source is required" blocker with exact commands.

### Exp 282: Publish Exp 66 joint model and FormalClaimVerifier ONNX to HuggingFace

**Deliverable:** `results/experiment_282_results.json`

Carry-forward from Exp 268 (SKIP, 3 failures in 2026.04.19). Publish two new artifacts to huggingface.co/Carnot-EBM: (1) the Exp 66 differentiable constraint model (embedding + Ising → score, 1.0 AUROC) as a `safetensors` artifact with a README stating "proof-of-concept demonstrating the approach, not production quality"; (2) the FormalClaimVerifier exported as ONNX for the arithmetic and comparison routes, with a Python bundle for remaining routes, and a README explaining solver routing, abstention policy, and standalone usage. Tag the collection as release `v0.2.0-research`. Log HF artifact URLs and model card stats in the results artifact. Use `huggingface-cli upload` for both artifacts.

### Exp 283: Operational retrospective for milestone 2026.04.20

**Deliverable:** `results/operational_retro_2026_04_20.json`

Generate the process efficiency analysis for milestone 2026.04.20. Specifically measure whether the action items from the 2026.04.19 retro were actually resolved (100% carry-over rate was the key finding). Track: (1) did DualGPURunner wire from experiment 1? (2) did per-question checkpointing prevent stall losses? (3) was CUDA ORT batch_size ≥ 32 tested for PredictiveVerifier (the threshold where GPU becomes faster)? (4) did the Apple adversarial benchmark complete in a single session? Report total wall time, experiments/hour, GPU utilization per experiment (not just end-of-milestone), and updated recommendations for milestone 2026.04.21. The retro must evaluate PROCESS quality (did we learn from the last two retros?) not just OUTCOME quality.

---

## Phase Summary

| Phase | Experiments | Theme | Key Success Criterion |
|-------|-------------|-------|----------------------|
| 87 | 271-273 | GPU benchmark completion + Apple adversarial | Exp 273 completes: adversarial verify-repair delta > standard GSM8K delta |
| 88 | 274-276 | Spilled energy + FactNet hallucination detection | SpilledEnergyExtractor AUROC > 0.6 on Exp 275 benchmark |
| 89 | 277-279 | FPGA hardware bring-up | Exp 277: first non-blocked KV260 execution_path result |
| 90 | 280-283 | Tier 3 self-learning + NPU + HF + retro | Exp 280: calibrated gate fast-path ≥ 30%, FP ≤ 20% |

---

## Hardware Requirements

| Hardware | Experiments | Status |
|----------|-------------|--------|
| 2× RTX 3090 (CUDA) | 271, 272, 273, 275, 280 | Available; must wire from Exp 1, not just harness setup |
| Kria KV260 FPGA | 277, 278, 279 | Available; CARNOT_KV260_BITFILE path must be configured at milestone start |
| AMD XDNA NPU (kernel module loaded) | 281 | VitisAI EP .so present; needs onnxruntime source build with VitisAI |
| HuggingFace CLI + credentials | 282 | May need `huggingface-cli login` at milestone start |
| ONNX export (from Exp 66) | 282 | Exp 66 safetensors artifact exists in results/ |

---

## Dependency Graph

```
Exp 271 (HumanEval Qwen GPU)          [independent, GPU required]
Exp 272 (Apple adversarial dataset)   [independent]
    └── Exp 273 (Apple adversarial + verify-repair) ── Exp 275 (spilled energy benchmark)
                                                    └── Exp 280 (JEPA calibration, uses logits from Exp 273)
Exp 274 (SpilledEnergyExtractor) ────────────────── Exp 275
Exp 276 (FactNet extractor)            [independent]
Exp 277 (KV260 bring-up) ──────────── Exp 278 (FpgaBackend) ── Exp 279 (FPGA benchmark)
Exp 281 (AMD XDNA NPU)                 [independent, 45min build timeout]
Exp 282 (HF publish)                   [independent]
Exp 283 (retro)                        [depends on all prior]
```

---

## What This Milestone Does NOT Include

- **PredictiveVerifier CUDA ORT batched benchmark** — Exp 259 proved CUDA is slower at batch_size=1. A batched benchmark (batch_size ≥ 32) would require changes to the inference pipeline. Defer to 2026.04.21 when PredictiveVerifier Tier 3 is proven useful at all.
- **Conformal prediction calibration (TECP)** — deferred until JEPA predictor training (Exp 280) proves the gate is discriminative; TECP adds coverage guarantees on top of a working gate.
- **DOMINO grammar-constrained generation** — long-term path, depends on completing the current autoregressive benchmark.
- **Exp 263 calibration with isotonic regression** — incorporated into Exp 280 (JEPA training on GPU inference features), not a separate experiment.
- **EB-JEPA / THRML integration** — still depends on self-learning calibration being proven; carry to 2026.04.21 or later.
- **ΠNet projection repair** — valid future direction (arXiv 2508.10480) but Langevin repair works well enough; defer to repair pipeline improvement milestone.

---

## Key Lessons from 2026.04.19 to Apply Immediately

1. **Wire DualGPURunner from Exp 1** — not just as a harness. The operational retro showed 99% of experiments ran single-GPU because the harness was wired in Exp 258 but experiments 259-269 didn't use it.
2. **Per-question checkpointing on every long benchmark** — "stalled after 180s silence" killed Exp 261-264, 266-270 in 2026.04.19. Checkpoint every 10 questions, resume gracefully.
3. **Set explicit build/inference timeouts** — Exp 281 NPU build has a 45-minute timeout; emit blocker immediately when exceeded rather than stalling conductor.
4. **GPU cleanup hook between experiments** — Exp 258 included `empty_cache_between_runs()` but it was never called in the actual experiment sequence.
5. **Provenance auto-sync** — run `scripts/validate-reconciliation.sh` as a post-experiment hook rather than a per-milestone manual step.
