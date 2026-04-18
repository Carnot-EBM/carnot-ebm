# Carnot Research Roadmap v40: VeriCoT Extraction, EBM-CoT Calibration, First Positive Numbers

**Created:** 2026-04-18
**Milestone:** 2026.04.34
**Status:** Planned (activates when milestone 2026.04.33 retrospective completes)
**Supersedes:** Milestone 2026.04.33 — "First Live Results, ThinkPRM Bridge, Boltzmann-GPT Repair"
**Informed by:** Exps 437-449, operational retrospective 2026.04.33, v39 carry-forwards
**External inputs (new in v40):**
- VeriCoT (arXiv 2511.04662) — neuro-symbolic CoT validation via FOL + Z3; 46% pass rate improvement
- VPRM (arXiv 2601.17223) — rule-based process verifiers; 20% F1 gain; no neural judge needed
- LSEBMCL (arXiv 2501.05495) — EBM-based cross-session continual learning; prevents forgetting
- Semantic Energy (arXiv 2508.14496) — Boltzmann logit energy for hallucination; 13% AUROC gain
- Gemma4 llama.cpp bug (llama.cpp#21516) — infinite <unused> tokens; root cause of RETRO-028
- AMD XDNA IRON (arXiv 2504.03083) — bare-metal NPU via mlir-aie; no ninja/openblas required

---

## What 2026.04.33 Proved

| Approach | Experiments | Verdict | Key Finding |
|----------|-------------|---------|-------------|
| LongRunBenchmarkExecutor (RETRO-026) | 437 | **CLOSED** | Batch executor splits benchmarks into 50q chunks, checkpoints each |
| GPU1 zombie fix (RETRO-025) | 438 | **FIX SHIPPED** | Explicit device_map fixes zombie allocation; live verification pending |
| Live precision micro (50q × 3 × 2) | 439 | **LIVE — NO IMPROVEMENT** | Baseline 14% (Qwen), 0% (Gemma4); no signed improvement in any variant |
| Live HumanEval micro (50 × 2) | 440 | **LIVE — NO IMPROVEMENT** | pass@1=0.0 both models; code_no_improvement |
| Live adversarial micro (50q × 3) | 441 | **DEGRADATION POSITIVE** | 14pp adversarial drop (Qwen), 0% repair recovery |
| FOVER live annotation | 442 | **FIRST REAL DATA** | 57 real labeled CoT pairs (30 correct, 27 incorrect); real_data_labeled |
| EORM+JEPA live retrain | 443 | **RETRO-024 CLOSED** | JEPA AUC 0.457→0.571 on real data; first FR-11 real relay |
| CarnotThinkProbe | 444 | **TIMED OUT** | 20-min budget insufficient; RETRO-029 opened |
| BoltzmannRepairBridge | 445 | **REPAIR ENERGY POSITIVE** | 100% repair success on synthetic; ready for live testing |
| Energy Matching ContinuousEBM | 446 | **NO RESULT JSON** | Script ran, result file missing; RETRO-030 opened |
| KAEMEnergy exact sampling | 447 | **BELOW THRESHOLD** | 1.29x speedup vs MCMC (< 5x target); RETRO-031 opened |
| Cross-session Tier 2 relay | 448 | **NO IMPROVEMENT** | SessionMemory + ConstraintTemplates insufficient; need constraint ADDITION |

**Milestone-level conclusion:**

For the first time after 7 consecutive scaffolding-only milestones, live GPU benchmarks ran and
returned real numbers. Results are honest negatives — no improvement — which is MORE valuable than
scaffolding artifacts because they reveal the actual failure modes:

1. **Gemma4-E4B-it 0% accuracy is a tokenizer bug, not an EBM failure.** GitHub issue
   llama.cpp#21516 documents infinite `<unused>` token generation. The model never produced
   valid text. RETRO-028 is a false negative — Gemma4 CAN produce correct answers but the
   loader was broken.

2. **Qwen3.5-0.8B 14% accuracy with no improvement** means the pipeline runs on real hardware
   but the verify-repair loop has 0% net effect. The ArithmeticExtractor regex still finds 0
   violations on instruction-tuned models (Gemma4 was 0/20 in Exp 203; Qwen likely similar).
   Without detected violations, verify-repair does nothing.

3. **Cross-session relay no improvement** (Exp 448) because we're reweighting existing
   constraints rather than ADDING new ones. research-program.md explicitly identified this:
   "The fix: constraint ADDITION from memory patterns, not just weight changes."

4. **JEPA AUC 0.457→0.571 is real progress** — first time any self-learning metric improved on
   live data. The FR-11 relay IS closing, just slowly. More real training data (from Exp 451's
   live run) will continue improving it.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Constraint Extraction Still Broken for IT Models (CRITICAL)

**Status:** Live benchmarks run and return real numbers, but verify-repair has 0% net effect
because the extraction pipeline finds 0 violations on instruction-tuned model outputs. The
ArithmeticExtractor regex matches `a + b = c` format that base models write, but IT models
write reasoning in natural language ("the total is 47 plus 28, which gives us 75").

**Root cause (Gemma4):** llama.cpp tokenizer bug causes infinite `<unused>` tokens — the model
never ran. Fix: use transformers library directly.

**Root cause (Qwen3.5-0.8B):** The extraction pipeline is regex-based. IT models don't write
equations in the expected format. The LLMExtractor (Exp 366) and VeriCoT-style formalization
are the right fixes but haven't been live-tested yet.

**Fix:** (1) Fix Gemma4 loading via transformers loader. (2) Implement VeriCoT step validation
(arXiv 2511.04662) + VPRM arithmetic rule verifiers (arXiv 2601.17223) as the primary extraction
front-end. These work on natural language reasoning, not regex patterns.

### Gap 2: Self-Learning Not Improving Across Sessions (HIGH)

**Status:** Exp 448 (cross-session relay) showed no_improvement. The cross-session mechanism
stores constraint templates in SessionMemory but Session 2 can't leverage them because there
is no replay mechanism — the templates are loaded but don't activate for new constraint patterns.

**Root cause:** Constraint template REWEIGHTING was proven ineffective (Exp 134: fixed=adaptive
across 500 questions). The correct approach is constraint ADDITION — when memory detects "carry
errors are common," ADD a carry-check constraint rather than upweight existing ones. This is
stated explicitly in research-program.md Goal #1.

**Fix:** (1) Implement Tier 1 constraint addition from memory patterns (carry-check, sign-check
templates added automatically when memory detects the pattern). (2) Add LSEBMCL-style EBM
replay (arXiv 2501.05495) for cross-session warm-starting. (3) Apply EBM-CoT calibration
(arXiv 2511.07124) to EORM hidden state to improve AUC beyond 0.571.

### Gap 3: Hardware Path Stalled — KV260 and NPU Unused (MEDIUM)

**Status:** KV260 FPGA arrived months ago but CARNOT_KV260_BITFILE is never set — the board
sits idle. AMD XDNA NPU blocked for 5 consecutive milestones by missing ninja/openblas packages.
KAEM is only 1.29x faster than MCMC (below 5x threshold).

**Root cause:** FPGA bitfile generation is undocumented. NPU prerequisite installation is
a 1-command fix but hasn't been done. KAEM was only benchmarked at n_vars ≤ 100 — the crossover
point where it beats MCMC is likely n_vars > 200 (where MCMC mixing time dominates).

**Fix:** (1) Profile KAEM at n_vars = 100–1000 to find the crossover point. (2) Attempt the
AMD XDNA NPU via IRON toolchain (pip install mlir-aie, no cmake/ninja needed, arXiv 2504.03083).
(3) Document KV260 bitfile synthesis steps and create a placeholder bitfile for bring-up testing.

---

## Architecture After This Milestone

```
Input LLM Response
    │
    ├─[Tier 0a: CarnotThinkProbe] ─── CoT verdict (incorrect → fast-path)
    │                                  (arXiv 2504.16828, Exp 444/455)
    │
    ├─[Tier 0b: SpilledEnergyDetector] ─── Pre-softmax logit energy
    │                                       (arXiv 2602.18671, Exp 433)
    │
    ├─[NEW: VeriCoT Step Validator] ─── FOL formalization + Z3 per-step check
    │                                    (arXiv 2511.04662, Exp 453)
    │
    ├─[NEW: VPRM Rule-Based Verifier] ─── Deterministic arithmetic rules
    │                                      (arXiv 2601.17223, Exp 454)
    │
    ├─[Tier 1: SinkProbe] ─── Attention sink concentration
    │                          (arXiv 2604.10697, Exp 348)
    │
    ├─[Tier 2: EORM (EBM-CoT calibrated)] ─── CoT energy reward model
    │                                           (arXiv 2505.14999 + 2511.07124, Exp 443/458)
    │
    └─[Tier 3: Ising/KAN/KAEM] ─── Full constraint verification + BoltzmannRepair
                                     (Exps 61/96/447, BoltzmannRepairBridge Exp 445)
                                              │
                                    [Self-Learning Loop]
                                    Tier 1: Constraint ADDITION (Exp 456)
                                    Tier 2: LSEBMCL EBM replay (Exp 457)
                                    Tier 3: EORM retrain on real data
```

---

## Phase Descriptions

### Phase 1 — Fix the Zero-Accuracy Problem (Exps 450-452)

These experiments address the three RETRO items that prevent any positive benchmark number.
RETRO-028 (Gemma4 0% accuracy) is the highest-impact fix — if Gemma4 works, we have a model
with 80%+ baseline accuracy (consistent with published Gemma4 numbers) where verify-repair
has real errors to catch. RETRO-030 (Exp 446 silent drop) is a quick fix. Exp 451 is the
first experiment likely to produce a positive verify-repair number.

**Exp 450** — Gemma4 Tokenizer Diagnosis and Fix (RETRO-028 closure)
: Diagnose the llama.cpp tokenizer bug (infinite `<unused>` tokens). Implement
  `GemmaTransformersLoader` that uses HuggingFace transformers directly rather than any
  llama.cpp backend. Verify the model produces valid text on 10 GSM8K questions.
  Deliverable: `results/experiment_450_gemma4_fix.json`

**Exp 451** — Live Precision Re-Run Post-Fix (first positive number target)
: Re-run the Exp 439 micro-benchmark harness with the fixed Gemma4 loader. Expected Gemma4
  baseline: 75-80% (published performance). Verify-repair should show improvement because
  even IT models make mistakes, and CRANE extraction captures structured claims.
  Deliverable: `results/experiment_451_live_precision_postfix.json`

**Exp 452** — Energy Matching v2 (RETRO-030 closure)
: Re-run the Exp 446 energy matching script. Add explicit result file existence check to
  ensure `results/experiment_452_energy_matching_v2.json` is written even on partial run.
  Also add Phase 3 continuous EBM improvement tracking.
  Deliverable: `results/experiment_452_energy_matching_v2.json`

### Phase 2 — Better Extraction for IT Models (Exps 453-455)

These experiments implement new extraction approaches that work on instruction-tuned model
outputs, bypassing the ArithmeticExtractor regex failure. VeriCoT (arXiv 2511.04662) and
VPRM (arXiv 2601.17223) are the primary new approaches. CarnotThinkProbe (RETRO-029) gets
a redesigned budget.

**Exp 453** — VeriCoT Step Validator (arXiv 2511.04662)
: Implement `VeriCoTStepValidator`: LLM extracts FOL premises from each CoT step, Z3 verifies
  logical consistency. The key advantage over ArithmeticExtractor: works on natural-language
  reasoning ("the total is 47 plus 28, which gives 75" → Z3 asserts 47+28=75 → UNSAT).
  Deliverable: `results/experiment_453_vericot_validator.json`

**Exp 454** — VPRM Arithmetic Rule Verifier (arXiv 2601.17223)
: Implement `VPRMArithmeticVerifier`: rule-based step checkers (addition, multiplication,
  percentage, unit consistency). Pure Python arithmetic rules, no ML judge. Deterministic,
  transparent, immune to reward hacking. Compare detection rate vs ArithmeticExtractor on
  GSM8K with Gemma4-E4B-it (post-fix).
  Deliverable: `results/experiment_454_vprm_verifier.json`

**Exp 455** — Think Probe v2 with Partial Verdicts (RETRO-029 closure)
: Redesign CarnotThinkProbe for 60-minute budget with partial verdict support. If the full
  benchmark doesn't complete, emit a partial result with the completed fraction.
  Add incremental checkpointing every 10 questions.
  Deliverable: `results/experiment_455_think_probe_v2.json`

### Phase 3 — Self-Learning with Constraint Addition (Exps 456-458)

These experiments implement the critical fix identified in research-program.md: constraint
ADDITION from memory patterns, not weight reweighting. Exp 456 implements the direct fix
for Exp 448's failure. Exp 457 adds LSEBMCL-style EBM replay. Exp 458 applies EBM-CoT
calibration to improve EORM accuracy beyond 0.571.

**Exp 456** — Tier 1 Constraint Addition from Memory (research-program.md Goal #1)
: Implement `ConstraintAdditionFromMemory`: when CaseMemory detects ≥5 instances of a
  violation type (carry/sign/unit/comparison), GENERATE a new ConstraintTerm and ADD it
  to the active pipeline — not just upweight existing ones. Test: Session 1 (50q carry errors)
  → Session 2 (carry-check constraint auto-added) → Session 2 FP rate measured.
  Deliverable: `results/experiment_456_constraint_addition.json`

**Exp 457** — LSEBMCL Cross-Session EBM Replay (arXiv 2501.05495)
: Implement `LSEBMConstraintReplayer`: train a small Ising EBM on violation type distributions
  from Session 1, then replay N synthetic violations from the EBM to warm-start Session 2's
  template library. Compare to Exp 448 (no improvement) and Exp 456 (constraint addition).
  Deliverable: `results/experiment_457_lsebmcl_replay.json`

**Exp 458** — EBM-CoT Latent Thought Calibration (arXiv 2511.07124)
: Implement `EBMCoTCalibrator`: apply Langevin dynamics to EORM hidden state encodings before
  scoring. The calibration moves hidden states toward lower-energy (higher-consistency) regions
  before EORM assigns the final score. Train on real labeled CoT pairs from Exp 443.
  Target: EORM AUC > 0.600 (vs 0.571 baseline).
  Deliverable: `results/experiment_458_ebm_cot_calibration.json`

### Phase 4 — Hardware + Scale (Exps 459-460)

These experiments advance the hardware path and profile KAEM at the scale where it shows
clear advantages over MCMC. The AMD XDNA IRON toolchain (arXiv 2504.03083) offers a
pip-install-only path that bypasses the 5-milestone ninja/openblas blockage.

**Exp 459** — KAEM Large-Variable Crossover Benchmark (RETRO-031 closure)
: Profile KAEMEnergy at n_vars = {50, 100, 200, 500, 1000}. Expected crossover: KAEM beats
  MCMC at n_vars > 200 where MCMC mixing time grows super-linearly. Report speedup curve
  and identify the crossover point.
  Deliverable: `results/experiment_459_kaem_large_vars.json`

**Exp 460** — AMD XDNA IRON NPU Unblock (arXiv 2504.03083)
: Install mlir-aie via pip (no cmake/ninja needed). Use IRON toolchain to compile a small
  Ising energy kernel for the AMD NPU. Test whether the JEPA predictor ONNX model
  (results/jepa_predictor_291.onnx) runs faster on NPU than CPU.
  Deliverable: `results/experiment_460_npu_iron.json`

### Phase 5 — Retrospective (Exp 461)

**Exp 461** — Milestone 2026.04.34 Retrospective
: Evaluate milestone results. Headline question: "Did we get the first POSITIVE verify-repair
  number after two milestones of honest negatives?" Close RETRO-028, RETRO-029, RETRO-030,
  RETRO-031 where applicable. Open new RETRO items.
  Deliverable: `results/operational_retro_2026_04_34.json`

---

## Dependency Graph

```
Exp 450 → Exp 451 (need Gemma4 fix before live benchmark)
Exp 452 (independent — re-run energy matching)
Exp 453 → Exp 454 (VeriCoT then VPRM — both extraction, share FOL/Z3 infrastructure)
Exp 455 (independent — think probe redesign)
Exp 456 → Exp 457 (constraint addition first, then EBM replay on top)
Exp 458 (depends on Exp 443 real training data — already exists)
Exp 459 (independent — KAEM scale profiling)
Exp 460 (independent — NPU unblock)
Exp 461 (depends on all Exps 450-460) — retrospective
```

---

## Success Criteria

| Criterion | Threshold | Source Experiment |
|-----------|-----------|-------------------|
| retro_028_resolved | Gemma4 produces valid text (>5% accuracy) | Exp 450 |
| first_positive_number | signed_improvement > 0 on live GPU | Exp 451 |
| retro_030_resolved | Exp 452 result JSON exists | Exp 452 |
| vericot_detection_rate | detect_rate > ArithmeticExtractor on IT model | Exp 453 |
| vprm_detection_rate | detect_rate > ArithmeticExtractor on IT model | Exp 454 |
| retro_029_resolved | think_probe_v2 produces partial_or_full verdict | Exp 455 |
| constraint_addition_works | Session 2 fp_rate < Session 1 fp_rate | Exp 456 |
| lsebmcl_improves_relay | LSEBMCL fp_rate < Exp 448 baseline | Exp 457 |
| eorm_auc_improved | EORM AUC > 0.600 (vs 0.571 Exp 443) | Exp 458 |
| kaem_crossover_found | crossover_n_vars > 0 (speedup > 5x at some n) | Exp 459 |
| npu_iron_runs | ONNX model executes on NPU | Exp 460 |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|------------|----------|-------|
| Exp 450 | CPU | Diagnostic only, no GPU |
| Exp 451 | 2x RTX 3090 | CARNOT_FORCE_LIVE=1 required; Gemma4 on GPU0, Qwen on GPU1 |
| Exp 452-460 | CPU | All JAX CPU mode; no GPU needed |
| Exp 461 | CPU | Retrospective analysis only |

---

## Open RETRO Items Targeted for Closure

| RETRO | Priority | Description | Target Experiment |
|-------|----------|-------------|-------------------|
| RETRO-028 | P0 | Gemma4-E4B-it 0% accuracy (tokenizer bug) | Exp 450 |
| RETRO-029 | P1 | Think probe timed out at 20 min | Exp 455 |
| RETRO-030 | P1 | Exp 446 silent drop (no result JSON) | Exp 452 |
| RETRO-031 | P2 | KAEM 1.29x speedup (< 5x threshold) | Exp 459 |

---

## Research Gaps Targeted (research-program.md alignment)

| Goal | research-program.md Section | Experiment |
|------|------------------------------|------------|
| Rebuild constraint extraction for IT models | Goal #1 (HIGHEST PRIORITY) | Exps 453, 454 |
| Establish real baselines with IT models | Goal #2 (CREDIBILITY) | Exps 450, 451 |
| Code verification on live GPU | Goal #3 (MOST LIKELY TO WORK) | Exp 451 |
| Tier 1 constraint addition | Self-Learning Goal | Exp 456 |
| LSEBMCL cross-session learning | Tier 2 Self-Learning | Exp 457 |
| JEPA predictive verification (Tier 3) | Tier 3 Self-Learning | Exp 458 |
| FPGA/hardware progress | Goal #5 | Exps 459, 460 |
