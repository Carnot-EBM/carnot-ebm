# Carnot Research Roadmap v24: GPU-Accelerated Inference, Calibrated Verification, and Live Benchmark Completion

**Created:** 2026-04-13
**Milestone:** 2026.04.19
**Status:** Planned (activates when milestone 2026.04.18 completes)
**Supersedes:** Milestone 2026.04.18 — "Formal Claim Verification, Process Integrity, and Predictive Self-Learning"
**Informed by:** Exp 247, Exp 251, Exp 256, Exp 257, operational retrospective 2026-04-13
**External inputs:** The 4/δ Bound (2512.02080), EORM (2505.14999), Self-Play Info Gain (2603.02218), PSV (2512.18160), TECP conformal prediction (MDPI 2025), DOMINO grammar decoding (ICML 2025), Sol-Ver (2502.14948), Sparse Ising Machines (Nature Comms 2024), Stochastic Ising Advantage (2504.18359)

---

## What 2026.04.18 Proved

| Approach | Experiments | Finding |
|----------|-------------|---------|
| Solver-routed formal claim verification | 244-247 | FormalClaimVerifier built and integrated cleanly, but live benchmark **blocked** at 18/200 cases (21s/case × 843 total cases ≈ 5 hours on CPU — exceeds conductor session budget). Infrastructure is solid; throughput is the blocker. |
| Process integrity verification | 248-251 | ProcessVerifier catches "right-for-wrong-reasons" defects (Qwen: 3 cases, Gemma: 2 cases in 30-case cohort) and produced one Gemma repair convergence that Exp 238 missed. But it adds **zero pass@1 lift** at the gating stage. Useful for audit/safety; not for accuracy improvement alone. |
| Calibrated self-learning A/B | 252-256 | PredictiveVerifier gate routed **all cases to FAST_PATH** when uncalibrated on live data, eliminating FPs but also missing real errors (−1.7pp). Constraint addition matched description-text proxies instead of real inference tokens, adding +4 FPs with zero new successes. Root cause: gate never trained/calibrated on live GSM8K distribution; templates too liberal. |
| PredictiveVerifier hardware path | 257 | CPU NumPy: 41.8 µs/call. ONNX CPU ORT: 5.8 µs/call (7.1× faster). CUDA ORT **blocked** — requires `pip install onnxruntime-gpu` (not default wheel). AMD XDNA NPU **blocked** — Python 3.14 unsupported by VitisAI EP. Both are solvable, not fundamental. |

**The milestone-level conclusion:** Carnot has now built all the infrastructure layers — formal claim routing, process integrity, predictive gating, constraint addition, warm model server, DualGPURunner. The gap is not capability; it is calibration, throughput, and wiring. The formal claim benchmark that should prove Carnot's value was never able to complete because CPU inference is 7× too slow. The self-learning system failed because the gate was never calibrated on the inference distribution it needs to operate over. The next milestone fixes both.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: The first credible positive verify-repair result on real IT models is still missing

After 257 experiments across 18 milestones, Carnot has never produced a completed, statistically meaningful positive verify-repair improvement on a real instruction-tuned model. The solver-routed semantic benchmark (Exp 247) is the most principled attempt yet but ran out of runtime budget after 18/200 cases. The core enabler is throughput: going from 21s/case (CPU) to ~2-3s/case (GPU with batching) makes the full benchmark feasible in a single conductor session. **This is the primary deliverable of this milestone.**

### Gap 2: Self-learning produces no held-out gain because the gate was never calibrated on live data

Exp 256 showed the PredictiveVerifier routed everything to FAST_PATH (uncalibrated). The fix is conceptually simple: collect (partial_response → final_violation_label) pairs from live GPU inference and fit a calibrated threshold. The 4/δ bound (2512.02080) provides a formal framework for this calibration. The Self-Play Info Gain paper (2603.02218) explains why constraint templates matched proxy text (no real learnable signal) — the fix is mining live inference tokens, not description text. Both are actionable, bounded experiments.

### Gap 3: Hardware acceleration for the verification pipeline has known, unblocked solutions that have not been executed

- **CUDA ORT:** one command (`pip install onnxruntime-gpu`) unblocks 10-100× speedup for PredictiveVerifier
- **AMD XDNA NPU:** Python 3.12 venv with onnxruntime 1.20.1 already set up; VitisAI EP `.so` files already present in `~/github.com/amd/RyzenAI-SW/`; needs one linking/path experiment
- **DualGPURunner:** built in Exp 224b, never wired to the benchmark harness; estimated +15% wall-time reduction per operational retrospective
- **Inference batching:** 8-16 per pass; estimated +10% wall-time reduction; not implemented in any live harness

Solving these moves from "infrastructure exists but blocked" to "infrastructure deployed and benchmarked."

---

## Promising 2025-2026 Inputs Adopted in v24

- **4/δ Bound (2512.02080):** formal Markov chain model for multi-stage verification pipelines gives principled calibration strategy for the PredictiveVerifier. Identifies three operating zones and parameterized threshold selection. Directly informs Exp 262.
- **TECP conformal prediction (MDPI 2025):** token-entropy conformal prediction gives finite-sample coverage guarantees without retraining. Candidate calibration backend for PredictiveVerifier to replace static thresholds.
- **EORM (2505.14999):** lightweight 55M-parameter energy-based ranking verifier achieves 90.7% GSM8K with Llama 3 8B. Establishes a published EBM baseline that Carnot should compare against after calibration.
- **Self-Play Info Gain (2603.02218):** root-cause analysis for why constraint templates failed — they provided no learnable information gain because they matched proxy text rather than real inference tokens. Prescribes live-token mining and asymmetric co-evolution.
- **PSV (2512.18160):** formal verification as self-play signal (9.6× pass@1 improvement) — motivates using FormalClaimVerifier verdicts as the learning signal for constraint addition rather than loose templates.
- **DOMINO grammar decoding (ICML 2025):** 17.71× faster grammar-constrained preprocessing with speculative decoding compatibility — long-term path to enforce formal claim templates during generation rather than post-hoc, eliminating the extraction bottleneck entirely.
- **Sol-Ver (2502.14948):** bidirectional quality-gating for solver+verifier self-play prevents error propagation — informs mutual verification between constraint templates and live tokens.
- **Sparse Ising + stochastic advantage papers:** confirm sparse constraint graph formulation is the right design for KV260/TSU hardware. The PredictiveVerifier energy gate maps to a sparse Ising problem class where hardware would win.

---

## v24 Hypothesis

If Carnot (1) completes the solver-routed semantic benchmark with GPU inference, (2) calibrates the PredictiveVerifier gate on live GSM8K data, and (3) fixes constraint addition to use domain-specific live inference tokens rather than description-text proxies, it should be able to produce:

1. the first **completed, statistically valid** solver-routed semantic benchmark result (Qwen3.5-0.8B and Gemma4-E4B-it, 200+ GSM8K cases, all three modes),
2. a calibrated self-learning A/B result showing whether Tier 1/2/3 learning produces real held-out gain when the gate is properly tuned,
3. a deployed CUDA ORT path for PredictiveVerifier (not just a CPU benchmark), and
4. the first HuggingFace artifact publications from Carnot research.

---

## v24 Architecture: GPU Throughput → Calibrated Gate → Live Self-Learning

```
Benchmark Item
      |
      v
┌──────────────────────────────────────────────────────────────────┐
│ DualGPURunner (Exp 258)                                          │
│  Qwen → GPU 0 (RTX 3090 #0)  |  Gemma → GPU 1 (RTX 3090 #1)    │
│  Batched inference (8-16/pass)                                   │
│  ~2-3s/case target (vs 21s/case CPU baseline)                    │
└───────────────────────────────────┬──────────────────────────────┘
                                    |
                                    v
┌──────────────────────────────────────────────────────────────────┐
│ Shared Benchmark Harness (Exp 258 upgrade)                       │
│  resume Exp 247 checkpoints → complete 200 GSM8K + 81 ctir       │
│  Exp 260: full solver-routed semantic benchmark (first complete) │
│  Exp 261: full 164-problem HumanEval (Qwen via GPU)              │
└───────────────────────────────────┬──────────────────────────────┘
                                    |
                                    v
┌──────────────────────────────────────────────────────────────────┐
│ PredictiveVerifier (Tier 3)                                      │
│  Exp 259: install onnxruntime-gpu → CUDA ORT path                │
│  Exp 262: collect live calibration corpus from GPU inference     │
│  Exp 263: calibrate gate via conformal prediction (4/δ bound)    │
│  Exp 269: AMD XDNA NPU path (Python 3.12 venv + VitisAI EP)     │
└───────────────────────────────────┬──────────────────────────────┘
                                    |
                                    v
┌──────────────────────────────────────────────────────────────────┐
│ Self-Learning (Tier 1/2/3 combined)                              │
│  Exp 264: domain constraint templates from live inference tokens │
│  Exp 265: constraint addition wired to formal solver verdicts    │
│  Exp 266: self-learning A/B v3 (calibrated gate + real templates)│
└───────────────────────────────────┬──────────────────────────────┘
                                    |
                                    v
┌──────────────────────────────────────────────────────────────────┐
│ HuggingFace Publishing (research-program.md "NOW" priorities)    │
│  Exp 267: update 16 model READMEs → point to pip install carnot │
│  Exp 268: publish Exp 66 joint model + FormalClaimVerifier ONNX  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 83: GPU Inference Pipeline and Benchmark Completion (Experiments 258-261)

The operational retrospective identified sequential dual-model loading and missing inference batching as the top two bottlenecks (+25% estimated wall-time together). This phase addresses both, then applies them to complete the experiments blocked in 2026.04.18.

### Exp 258: Wire DualGPURunner to live benchmark harness with batched inference

**Deliverable:** `scripts/experiment_258_dual_gpu_harness.py`

Wire the DualGPURunner (built in Exp 224b at `python/carnot/inference/dual_gpu.py`) and warm model server (Exp 224a at `python/carnot/inference/model_server.py`) to the shared benchmark harness so that Qwen3.5-0.8B runs on GPU 0 and Gemma4-E4B-it runs on GPU 1 simultaneously for all dual-model experiments. Add inference batching at batch_size=8 or 16. Add automatic GPU memory cleanup (torch.cuda.empty_cache()) between runs. The harness should wrap the existing experiment_218_live_dual_model_suite.py interface so existing benchmark runners can opt in without full rewrites. Verify throughput reaches ≤3s/case target on both models. Write tests first for GPU assignment, batching, cleanup, and harness interface.

### Exp 259: onnxruntime-gpu CUDA EP unlock and PredictiveVerifier benchmark

**Deliverable:** `results/experiment_259_results.json`

Install onnxruntime-gpu (`pip install onnxruntime-gpu --extra-index-url https://download.pytorch.org/whl/cu121`) in the project venv, verify that CUDAExecutionProvider is listed in ort.get_available_providers(), export or reload the PredictiveVerifier ONNX model (check `results/jepa_predictor_146.onnx`), and benchmark CUDA ORT vs CPU ORT (5.8 µs baseline from Exp 257). Report latency (µs/call), throughput (calls/s), GPU memory overhead (MB), and model export compatibility. If CUDA ORT requires re-export, regenerate from `python/carnot/pipeline/predictive_verifier.py`. Emit honest blocker artifact if GPU setup fails — do not fabricate numbers.

### Exp 260: Complete solver-routed semantic benchmark with GPU inference

**Deliverable:** `results/experiment_260_results.json`

Execute the full solver-routed semantic benchmark using the Exp 258 DualGPURunner harness. Resume from Exp 247 checkpoints (18/200 Qwen baseline cases already complete at `results/checkpoints/experiment_246/`). Cover all cells: 200 GSM8K × {baseline, verify_only, verify_repair} × {Qwen/Qwen3.5-0.8B, google/gemma-4-E4B-it} and 81 constraint_ir × 3 modes × 2 models. Report route-level evidence (solver routes that fired, abstain rates), per-model false positive budget, and whether verify-only is non-harmful for at least one model. This is the **primary deliverable of the milestone** — the first statistically complete solver-routed benchmark. Keep checkpointing at 10-case granularity. Do not fabricate cells if any remain blocked — emit a partial artifact with honest status.

### Exp 261: Full 164-problem HumanEval benchmark with Qwen via GPU

**Deliverable:** `results/experiment_261_results.json`

Run the full 164-problem HumanEval benchmark using the GPU harness from Exp 258, for Qwen/Qwen3.5-0.8B only (Gemma 164-problem result already exists in Exp 226). Use the PBT + spec-aware + process-aware verifier stack from Exp 250. Report pass@1 at each stage: baseline, pbt_verify_only, spec_aware_verify_only, process_aware_verify_only, verify_repair. This generates the missing paired Qwen full-164 result alongside Exp 226 to enable a direct cross-model comparison at scale.

---

## Phase 84: Predictive Verifier Calibration (Experiments 262-263)

The Exp 256 root cause: the PredictiveVerifier gate was never calibrated on the live GSM8K inference distribution. It routed 100% of cases to FAST_PATH because its internal confidence scores were uncalibrated. This phase collects a calibration corpus and fits a calibrated threshold using the 4/δ bound framework.

### Exp 262: Live calibration corpus for PredictiveVerifier from GPU inference

**Deliverable:** `data/research/predictive_calibration_corpus_262.jsonl`

Using the GPU harness from Exp 258, run a targeted calibration collection pass over 200 GSM8K questions with Qwen/Qwen3.5-0.8B. For each question, capture: the first N-token prefix (partial response at 25%, 50%, 75% of final length), the full response, and the final violation label from FormalClaimVerifier (did it flag a violation?). Extract domain-specific token patterns (explicit equation tokens, arithmetic operator sequences, digit density) from the partial contexts, keyed by whether a final violation occurred. This creates a ground-truth (partial_context_features, full_violation_label) corpus with provenance to Exp 260 case IDs. Store as JSONL with schema: case_id, prefix_fraction, token_pattern_features, violation_label, provenance_exp.

### Exp 263: Calibrate PredictiveVerifier and run calibrated self-learning A/B benchmark

**Deliverable:** `results/experiment_263_results.json`

Use the Exp 262 corpus to calibrate the PredictiveVerifier threshold via isotonic regression. Target: fast-path hit rate ≥ 30%, true-violation detection rate ≥ 60%, FP rate ≤ 20% on held-out portion. Reference the 4/δ bound paper (arXiv 2512.02080) for operating zone identification. After calibration, run the full self-learning A/B benchmark (same structure as Exp 256: no_learning, case_memory_plus_policy, constraint_addition, predictive_gate, combined) using the calibrated gate and the Exp 260 result as the evaluation cohort. Primary success condition: any strategy achieves positive held-out gain with ≤ Exp 241 false positives. Report secondary metrics: verification spend, fast-path hit rate, per-domain lift, per-model lift.

---

## Phase 85: Domain-Specific Constraint Templates and Tier 2 Self-Learning (Experiments 264-266)

Exp 256 showed constraint templates matched description-text proxies (full English strings) not real inference tokens. The Self-Play Info Gain paper (2603.02218) prescribes live-token mining for learnable signal. This phase rebuilds constraint templates from GPU inference data and wires constraint addition to formal solver verdicts.

### Exp 264: Domain-specific constraint template extraction from live inference tokens

**Deliverable:** `data/research/domain_constraint_templates_264.jsonl`

Mine the Exp 262 calibration corpus and Exp 260 results for domain-specific token patterns that correlate with FormalClaimVerifier violations. For arithmetic: explicit equation tokens (digits, arithmetic operators, equals signs) and their positional context. For cardinality: count-word and quantity-noun patterns. For set-membership: list-enumeration and element-identification token sequences. Store templates as (domain, token_pattern_regex, associated_claim_route, corpus_precision, corpus_recall, model_specificity) rows. Use FormalClaimVerifier verdicts from Exp 260 as ground-truth labels (PSV approach: formal verification as the learning signal). Enforce minimum corpus_precision ≥ 0.50 before inclusion — no liberal matching allowed. Report template counts, precision distribution, and which models each template is specific to.

### Exp 265: Constraint addition module wired to formal solver verdicts

**Deliverable:** `python/carnot/pipeline/constraint_addition.py` (updated)

Update constraint_addition.py to: (1) use Exp 264 domain templates as the source for new constraint candidates (replacing description-text proxy matching); (2) wire the addition decision to FormalClaimVerifier verdict history (from case_memory or Exp 260 checkpoints) — when memory detects a recurring violation type (e.g., arithmetic carry errors in Qwen3.5-0.8B multi-step addition), ADD a new FormalClaimVerifier route extension, not just a weight upward; (3) apply mutual verification gating (Sol-Ver approach): a template is only promoted if it fires on ≥ 3 known-violation cases AND fires on < 20% of known-correct cases in the calibration corpus. Write tests first covering template promotion logic, mutual verification gating, and integration with VerifyRepairPipeline.

### Exp 266: Self-learning replay v3 with calibrated gate and domain templates

**Deliverable:** `results/experiment_266_results.json`

Run the chronological self-learning replay (Exp 241/256 structure) using the calibrated PredictiveVerifier gate (Exp 263) and domain-specific constraint templates (Exp 264/265). Test four strategies: no_learning, case_memory_plus_policy, calibrated_constraint_addition, calibrated_predictive_gate_plus_addition. Primary success condition: positive held-out gain with ≤ Exp 241 (8) false positives on at least one strategy. Include direct numeric comparison blocks against Exp 241 and Exp 256. State clearly if the primary condition is not met. Preserve per-strategy fast-path hit rate, latency, and domain-level breakdown.

---

## Phase 86: HuggingFace Publishing and Hardware Enablement (Experiments 267-270)

Three of the six HuggingFace publishing milestones in research-program.md are marked "NOW" and have been deferred for multiple milestones. This phase executes them. Plus AMD XDNA NPU enablement (`.so` libraries are already present).

### Exp 267: Update 16 HuggingFace model READMEs

**Deliverable:** `results/experiment_267_results.json`

Update the READMEs of all 16 existing Carnot-EBM models on huggingface.co/Carnot-EBM. Add a status banner clarifying: (1) the per-token activation EBMs are Phase 1 research artifacts that detect confidence, not correctness; (2) users should use `pip install carnot` for the production pipeline with FormalClaimVerifier, MCP server, and PBT code verification; (3) a brief "what's proven to work" section referencing verified capabilities. Do not delete existing content — prepend the status banner and append the updated context section. Use `huggingface-cli` to push each update. Log each model's HF repo URL and push status in the results artifact. Write a simple script to batch the updates.

### Exp 268: Publish Exp 66 joint model and FormalClaimVerifier ONNX to HuggingFace

**Deliverable:** `results/experiment_268_results.json`

Publish two new artifacts to huggingface.co/Carnot-EBM: (1) the Exp 66 differentiable constraint model (embedding + Ising → score, 1.0 AUROC) as a safetensors artifact with a README stating "proof-of-concept demonstrating the approach, not production quality"; (2) the FormalClaimVerifier exported as an ONNX model for the ONNX-compatible components (arithmetic and comparison routes), with a Python module bundle for the remaining routes, and a README explaining solver routing, abstention policy, and how to use it standalone. Tag the collection as release `v0.2.0-research`. Log HF artifact URLs and model card stats in the results artifact.

### Exp 269: AMD XDNA NPU enablement for PredictiveVerifier

**Deliverable:** `results/experiment_269_results.json`

Attempt to enable the AMD XDNA NPU for PredictiveVerifier inference using the already-present VitisAI EP files in `~/github.com/amd/RyzenAI-SW/`. Steps: (1) activate `.venv-npu/` (Python 3.12); (2) configure LD_LIBRARY_PATH to include the VitisAI EP `.so` files and attempt to load `onnxruntime` with VitisAIExecutionProvider; (3) load `results/jepa_predictor_146.onnx`; (4) benchmark latency vs CPU ORT (5.8 µs Exp 257 baseline) and CUDA ORT (Exp 259). If NPU setup succeeds: report throughput (calls/s), power consumption (if measurable), and routing quality differences. If blocked: emit honest blocker naming the exact missing component and exact next action to unblock. Do not fabricate NPU numbers.

### Exp 270: Operational retrospective for milestone 2026.04.19

**Deliverable:** `results/operational_retro_2026_04_19.json`

Generate the process efficiency analysis for this milestone. Measure whether DualGPURunner wiring + inference batching achieved the predicted 25% wall-time reduction vs the 2026.04.18 baseline (3,889 minutes total, 14.2 min/experiment average). Enable continuous GPU monitoring (`gpu_monitor.py --loop`) at the start of this milestone to record actual GPU utilization rather than estimating it post-hoc. Document: total wall time, experiments/hour, slowest experiments, GPU utilization distribution, and whether the Exp 260 benchmark completion proves the throughput hypothesis. Generate updated recommendations for milestone 2026.04.20.

---

## Phase Summary

| Phase | Experiments | Theme | Key Success Criterion |
|-------|-------------|-------|----------------------|
| 83 | 258-261 | GPU inference pipeline + benchmark completion | Exp 260 completes: valid solver-routed results for both models |
| 84 | 262-263 | PredictiveVerifier calibration | Calibrated gate: fast-path ≥30%, detection ≥60%, FP ≤20% |
| 85 | 264-266 | Domain templates + Tier 2 self-learning | Any strategy: positive held-out gain vs Exp 241 baseline |
| 86 | 267-270 | HuggingFace publishing + hardware | ≥2 new HF artifacts; 16 READMEs updated |

---

## Hardware Requirements

| Hardware | Experiments | Status |
|----------|-------------|--------|
| 2× RTX 3090 (CUDA) | 258-266 | Available; DualGPURunner wiring needed |
| onnxruntime-gpu pip wheel | 259, 263 | One `pip install onnxruntime-gpu` away |
| AMD XDNA NPU (kernel module loaded) | 269 | VitisAI EP .so present in RyzenAI-SW; Python 3.12 venv ready |
| HuggingFace CLI + account | 267-268 | May need `huggingface-cli login` at milestone start |
| AMD KV260 FPGA | Retro mention | Available; overlay bring-up deferred to 2026.04.20 |

---

## Dependency Graph

```
Exp 258 (DualGPURunner harness)
    ├── Exp 260 (solver-routed semantic benchmark, complete)
    │       └── Exp 262 (calibration corpus from GPU inference)
    │               └── Exp 263 (calibrate gate + A/B v2)
    │                       └── Exp 266 (self-learning v3)
    └── Exp 261 (HumanEval 164-problem Qwen GPU)
Exp 259 (onnxruntime-gpu) ──────────────────────────── Exp 263
Exp 264 (domain templates) ── Exp 265 (constraint addition) ── Exp 266
Exp 267 (HF READMEs) [independent]
Exp 268 (HF publish) [independent, cite Exp 260 in README]
Exp 269 (AMD XDNA NPU) [independent]
Exp 270 (retro) [depends on all prior]
```

---

## What This Milestone Does NOT Include

- **KV260 FPGA overlay bring-up** — deferred to 2026.04.20; DualGPURunner and CUDA ORT deliver faster wins with lower risk. KV260 earmarked for a dedicated hardware phase.
- **Gemma3 / Qwen4 model upgrades** — not yet available at HuggingFace at planning time; models stay as Qwen/Qwen3.5-0.8B and google/gemma-4-E4B-it per research-program.md.
- **DOMINO / grammar-constrained generation** — long-term path, depends on completing the current autoregressive benchmark first.
- **EB-JEPA / THRML integration** — depends on self-learning calibration being proven. Post-milestone 2026.04.20 at earliest.
