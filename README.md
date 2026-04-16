# Carnot

**Open-source Energy Based Model framework — Rust + Python/JAX**

Carnot uses Energy-Based Models to **verify and repair LLM outputs**. It extracts constraints from any response, checks them formally (Z3 SMT, property-based testing, energy scoring), and repairs violations via LLM feedback. All headline results are from live GPU inference.

**Headline results:** +3.0pp on 164-problem HumanEval (statistically significant), +4.9pp on typed constraint verification, 86% false positive reduction via self-learning, 99.3% code bug detection rate. See the [technical report](docs/technical-report.md) for the full 411+ experiment analysis.

**What ships today:** `pip install carnot` -- verify any LLM output in 5 lines of Python. CLI, MCP server for Claude Code, and full API docs. Four energy model tiers (KAN, Ising, Gibbs, Boltzmann) with hardware acceleration paths (FPGA, D-Wave quantum annealing, Extropic TSU).

## Install

```bash
# Python (3.11+)
pip install -e ".[dev]"

# Verify it works
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --test "(7,13):1"
```

> GPU: `pip install carnot[cuda]` for CUDA 12. On AMD/ROCm, use `JAX_PLATFORMS=cpu`.
> Rust bindings (optional): `pip install carnot[rust]` with Rust toolchain installed.

### Quick start (Python API)

```python
from carnot.pipeline import VerifyRepairPipeline

pipeline = VerifyRepairPipeline()

# Correct answer — passes verification
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 42")
print(result.verified)    # True

# Wrong answer — caught by constraint extraction
result = pipeline.verify("What is 15 + 27?", "15 + 27 = 43")
print(result.verified)    # False
print(result.violations)  # [ConstraintResult: "15 + 27 = 43 (correct: 42)"]
```

## The Problem with LLMs

Large language models generate text by predicting the most probable next token. This produces fluent output, but it's fundamentally guessing — there is no mechanism to verify that the output is logically consistent, physically valid, or factually correct. When an early token is wrong, the error cascades irrecoverably. This is why LLMs hallucinate: they optimize for plausibility, not truth.

## Why Energy Based Models?

EBMs take a fundamentally different approach. Instead of generating outputs sequentially, they assign a scalar energy to every possible configuration of variables. Low energy = valid/consistent; high energy = invalid/contradictory. Inference is optimization: find the configuration that minimizes energy across all constraints simultaneously.

This enables capabilities that autoregressive models structurally cannot provide:

- **Verifiable reasoning** — mathematically prove a solution satisfies all constraints by showing it sits at an energy minimum
- **Surgical error correction** — when a constraint is violated, gradient descent fixes the broken part without discarding the rest
- **Autonomous self-improvement** — the energy function is an objective ground truth that cannot be gamed, enabling closed-loop self-learning without human feedback
- **Hardware acceleration** — energy landscapes map directly to thermodynamic sampling hardware (Extropic TSU), promising 10,000x efficiency gains

## How Carnot Uses EBMs (Introspection, Not Fine-Tuning)

**Carnot never modifies the LLM's weights.** The target language model remains completely frozen throughout all experiments and deployment. Instead, Carnot works by introspecting the LLM's existing internal representations:

1. **Logprob-based methods** — read the LLM's own per-token log-probabilities as an energy signal. The model is already an EBM (per the ARM↔EBM bijection); we simply read the energy it already computes.
2. **Activation-based methods** — extract hidden state activations from a frozen forward pass (`output_hidden_states=True`), then train a small separate EBM classifier (a lightweight Gibbs model, typically [1024→256→64→1]) on those extracted features via Noise Contrastive Estimation.
3. **Structural verification** — execute the LLM's generated code against test cases. No model weights involved at all.

The "training" in Carnot refers to training the small EBM classifier on activation features extracted from a frozen LLM — not gradient descent on the LLM itself. This is fundamentally different from fine-tuning, RLHF, or DPO, which modify the language model's parameters. Carnot's approach is closer to probing or introspection: we observe what the model already knows internally and build a lightweight detector on top of it.

## The Path to Self-Learning

Carnot is designed from the ground up to support an automated self-improvement loop (LLM proposes, energy function evaluates):

1. **Propose** — candidate improvements to architecture, training, or hyperparameters are prototyped in Python/JAX
2. **Evaluate** — the energy landscape on held-out data serves as the objective judge (did energy decrease? real improvement. did it not? rejected.)
3. **Deploy** — proven improvements are transpiled to Rust for production performance
4. **Repeat** — the loop runs without human supervision, with safety guardrails

The EBM itself is the evaluator. No LLM needed to judge quality — the math provides ground truth.

## Key Results (411 experiments, 16 completed milestones, 17th in progress)

All benchmark results below are from **live GPU inference**. Simulated and software-model artifacts remain in the repo, but they are labeled explicitly and are not mixed into the headline tables. See the [technical report](docs/technical-report.md) for the full history including what didn't work.

### Simulation vs Reality

Provenance snapshot: **15 live GPU artifacts**, **5 simulated artifacts**, **95 unverified artifacts**, and **1 software-model artifact** (Exp 228, software simulation). Only the live GPU subset informs the benchmark tables below.

Note: Milestone 2026.05.20 (Exps 351-364) discovered that `CARNOT_FORCE_LIVE` was never being set by the conductor (RETRO-012), which caused three consecutive milestones of silent simulated fallback despite both RTX 3090s being live-capable. Milestone 2026.05.27 (Exps 365-376) closed RETRO-012/013/014 but live GPU remained unconfirmed for a fourth consecutive milestone (RETRO-015 critical). Milestone 2026.06.03 (Exps 377-389) fixed the infrastructure (Exp 377: LiveGPUGate + session_startup.sh export), but the GPU node was offline during the conductor session — live GPU unconfirmed for a fifth consecutive milestone (RETRO-019 critical). Milestone 2026.06.10 (Exps 390-402, 16th milestone) ran entirely in "deliverable already exists" fast-path mode — GPU node offline for a SIXTH consecutive milestone; RETRO-022 critical human escalation opened (cloud GPU or power on RTX 3090 node required). Milestone 2026.06.17 (17th, in progress): Exp 404 confirmed GPU hardware IS present (`is_live_capable=True`) — the only remaining blocker is running `source scripts/session_startup.sh` before the next conductor session to propagate `CARNOT_FORCE_LIVE=1`. Exps 410-411 live harnesses implemented and blocked correctly (no simulated fallback). All live GPU result counts above reflect artifacts generated before this bug was identified.

## PBT Verification

Carnot's strongest live evidence is now the Hypothesis-backed code-verification path. The full Gemma run is positive and statistically significant; the seeded Qwen transfer check is intentionally honest about a flat repair delta while still showing additive verifier signal.

| Live PBT artifact | Baseline | +Carnot | Added verifier signal | Experiment |
|-------------------|----------|---------|-----------------------|------------|
| Full HumanEval 164 (Gemma4) | 11.6% | 14.6% | 144/145 wrong baselines detected; 6 official-test misses caught; 5 repairs | Exp 226 |
| Seeded HumanEval 30 (Qwen3.5-0.8B) | 23.3% | 23.3% | 17/23 wrong baselines detected; 2 official-test misses caught; 0 repairs | Exp 227 |
| Dual-model HumanEval 50 | 18.0% / 10.0% | 20.0% / 12.0% | 144/145 wrong-code detections across the paired slice | Exp 220 |

### Code verification (strongest domain)

| Benchmark | Baseline | +Carnot | Delta | Experiment |
|-----------|----------|---------|-------|------------|
| HumanEval 164 problems (PBT) | 11.6% | 14.6% | **+3.0pp** [+0.6, +6.1] CI | Exp 226 |
| HumanEval 30 problems (PBT, seeded Qwen cohort) | 23.3% | 23.3% | +0.0pp; 2 harness misses caught | Exp 227 |
| HumanEval 50 problems (PBT) | 18.0% / 10.0% | 20.0% / 12.0% | +2.0pp both models | Exp 220 |
| HumanEval 30 problems (execution) | 16.7% | 20.0% | +3.3pp | Exp 208 |

PBT detects 99.3% of wrong code (144/145) on the full Gemma run and catches 6 official-test misses. Exp 227 shows the same verifier still adds signal cross-model even when the repair loop itself stays flat.
On deterministic HumanEval-style probes, the same Hypothesis-backed verifier also catches **5/5** under-specified bugs that execution-only checks miss while preserving **5/5** matching correct solutions (Exp 224).

### Constraint verification (math/instruction)

| Benchmark | Baseline | +Carnot | Delta | Experiment |
|-----------|----------|---------|-------|------------|
| Typed IR constraints (81 tasks) | 61.7% | 66.7% | **+4.9pp** (Gemma4) | Exp 221 |
| GSM8K semantic v2 (200 questions) | 46.5% | 47.5% | +1.0pp (Gemma4); verify-only still unjustified | Exp 235 |
| GSM8K arithmetic (100 questions) | 91.0% | 91.0% | 0.0pp | Exp 206 |

Semantic verifier v2 reuses the fixed Exp 219 cohort and trims Qwen false positives from **7** to **4**, but verify-only still hurts on both models and Gemma now carries **26** unnecessary repair triggers (Exp 235). The main win is better calibration and cleaner abstention, not a solved live semantic benchmark yet.

### Self-learning

| Metric | Without learning | With learning | Improvement | Experiment |
|--------|-----------------|---------------|-------------|------------|
| Held-out success | 34.48% | 34.48% | Flat across all four strategies; primary success **not met** | Exp 241 |
| Retrieval quality | 0.0% hit / 0.0% precision | 32.1% hit / 43.6% precision | Better case reuse without extra wins or false positives | Exp 241 |

On **116** held-out cases against **344** learning cases, `no_learning`, `tracker_only`, `case_memory`, and `case_memory_plus_policy` all finish at **34.48%** success with **8** false positives. The latest positive signal is narrower and more honest than Exp 223: `case_memory` materially improves retrieval hit rate and precision, but those better matches still do not convert into held-out task gains under the zero-additional-false-positive budget.

### Latest follow-ons

- **Semantic calibration + live rerun (Exp 232 / Exp 235):** the new calibration corpus contains **568** rows (**155 TP / 33 FP / 221 FN / 159 TN**), and the semantic-verifier-v2 rerun keeps verify-only explicitly unjustified even after cleaner thresholds and abstention logic.
- **Spec-aware code verification (Exp 236 / VERIFY-036):** the explicit code-spec corpus now covers **164** HumanEval tasks with **194** trace links, **8** official-test-miss traces, and **5** repaired traces, and the packaged verifier can now combine official tests, PBT, and explicit spec clauses through `verify_generated_code_with_specs()`.
- **Chronological replay v2 (Exp 241 / VERIFY-038 / VERIFY-039 / VERIFY-040):** richer case keys plus compiled policy context improve held-out retrieval to **32.1%** hit rate and **43.6%** precision, but the primary task-gain success condition is still honestly **not met**.
- **KV260 host / overlay round-trip (Exp 242):** the checked-in artifact is intentionally blocked in this environment because no `CARNOT_KV260_BITFILE` path was configured; `mode="auto"` still resolves to CPU fallback instead of fabricating board timings.
- **Sampler-backed replay (Exp 243):** CPU reranking stays neutral across **460** saved semantic and code repair cases, while the KV260-backed path remains blocked by the same missing bitfile setup.
- **Formal claim corpus (Exp 244 / VERIFY-041):** **2,545** provenance-bearing rows from live traces — **1,243** solver-routable (arithmetic, boolean-entailment, set-membership, execution-oracle, cardinality, comparison) and **1,302** explicitly `not_formalizable`.
- **Process integrity corpus (Exp 248):** **849** rows across five labels (`right_answer_wrong_process`, `wrong_answer_partially_sound_process`, `unsupported_step`, `repair_fixed_outcome_only`, `repair_fixed_process_and_outcome`, `clean`), built from Exp 235 and Exp 238 live traces.
- **Process-aware verification comparison (Exp 251):** process verification added **0** rejections beyond the spec-aware gate on both models, but caught **5** `outcome_correct_process_invalid` cases (Qwen=3, Gemma=2) across **143** combined defect instances.
- **Predictive verifier hardware benchmark (Exp 257):** ONNX CPUExecutionProvider runs at **5.8 µs/call** (**7.1×** faster than CPU NumPy at **41.8 µs/call**); CUDA ORT and AMD XDNA NPU paths remain blocked by missing toolchain.
- **PrefillUncertaintyProbe (Exp 295 / REQ-VERIFY-080):** entropy-based pre-generation hallucination gate (arXiv 2603.19562). High entropy → `high_risk=True` → trigger full verification before any tokens are generated. Black-box, no gradient access required.
- **ConstraintGenerator from CaseMemory (Exp 300 / REQ-LEARN-010/011):** converts CaseMemory violation patterns into new constraints when observed_precision ≥ 0.85 (soundness bound, arXiv 2603.03538). Purely additive — never removes existing constraints.
- **Confidence-weighted repair gating (Exp 301 / REQ-VERIFY-081/082):** EBM energy-derived confidence scores gate the repair loop. Violations below threshold=0.8 are suppressed, fixing Exp 184's 0% net improvement from false-positive repairs.
- **Integrated Tier 1+2 self-learning benchmark (Exp 302):** first end-to-end run combining Exp 301 confidence-weighted gating and Exp 300 ConstraintGenerator on 100 questions (2 × 50 batches). Reports honest signed `improvement_delta`; negative values not hidden.
- **AMD XDNA NPU Unblock (Exp 303):** full source-build + inference benchmark path ready. Currently `blocked_prereq` (ninja + openblas missing). Run `sudo pacman -S ninja openblas` then re-run Exp 303 to auto-advance.
- **FCV LIVE on HuggingFace (Exp 304):** `Carnot-EBM/carnot-formal-claim-verifier-v1` is now live. Python API credential fallback resolved the Exp 293 CLI blocker.
- **Experiment template + batching harness (Exp 306 / REQ-VERIFY-083/084):** `ExperimentTemplate` + `BatchedInferenceRunner` eliminate 15–20 min cold-start per experiment. Template overhead validated at **0.0001 s** (target < 0.5 s).
- **Conductor hardening (Exp 325 / REQ-INFRA-001/002):** 45-min hard timeout wrapper (`run_experiment_with_timeout.sh`) and test-first stub generation (`generate_test_stub()`). Estimated 27% wall-time speedup per milestone.
- **DualGPUMonitor (Exp 326 / REQ-INFRA-003/004):** zombie process detection + idle-GPU checks wired into `setup_gpu()`; CI-safe.
- **Live GPU full-scale benchmark (Exp 328):** Qwen3.5-0.8B 27.5%, Gemma4-E4B-it 26.3% on adversarial GSM8K all-variant (`inference_mode=live_gpu`). Live accuracy ~10% below simulated baseline (honest divergence).
- **Live relay benchmark (Exp 329):** four-tier self-learning relay on live GPU; `improvement_1to3 = -6.1%` (negative relay signal, research concern; honest signed delta).
- **Live HuggingFace publish (Exp 330 / REQ-PUBLISH-004):** 16 per-token EBM READMEs updated with live-GPU benchmark numbers. FCV README updated. Joint-constraint placeholder created.
- **Confidence-weighted repair — dual-signal FP reduction (Exp 332 / REQ-VERIFY-083/084/085):** expression specificity + Ising variance gate. FPs avoided: **86.7%** (13/15), TPs preserved: **100.0%** (15/15), `GATE_EFFECTIVE`.
- **Model-adaptive constraint thresholds + selective CaseMemory (Exp 333 / REQ-LEARN-015/016):** `PerModelFPTracker` auto-disables range_check for qwen3.5-0.8b (fp_rate=0.73 > tp_rate=0.27 after 15 obs). `SelectiveConsolidation` (ATLAS arXiv 2511.01093) at consolidation_ratio=0.60 (`ADAPTIVE_PASS_ATLAS_PARTIAL`).
- **VERGE-style iterative Z3 refinement (Exp 334 / REQ-REPAIR-012/013):** `VergeRefiner` identifies the specific failed Z3 assertion and repairs only that step; 3-iteration max loop.
- **CoT Circuit Verifier (Exp 336 / REQ-EXTRACT-015/016):** `CoTCircuitVerifier` extracts a computational dependency graph from chain-of-thought responses and checks structural consistency (value-carryover mismatches, cycles). Catches error classes that arithmetic regex and Z3 miss.
- **Milestone 2026.05.06 retrospective (Exp 337):** 12 experiments, 293 total min, mean 24.4 min/exp. **Actual speedup: 39.9%** vs prior milestone baseline (exceeds 27% estimate). All 4 prior action items resolved.

### Milestone 2026.05.13 (Exps 338-348)

- **Host prereqs registry + DualGPU auto-assignment (Exp 338 / REQ-INFRA-006/007):** `HostPrereqsRegistry` checks packages before experiment launch; `DualGPURunner` selects the idle GPU automatically. Closes RETRO-005/006.
- **Pre-session startup health check (Exp 339 / REQ-INFRA-008):** `scripts/session_startup.sh` (--dry-run / --kill-zombies) detects GPU count and zombie processes before any experiment. Closes RETRO-007/008.
- **Live full-precision pipeline benchmark (Exp 340 / REQ-BENCH-003):** first honest measurement of combined precision stack (5 variants × 2 models × 200 GSM8K). `PipelineVariant` enum: BASELINE, CONFIDENCE_ONLY, CONFIDENCE_ADAPTIVE, CONFIDENCE_ADAPTIVE_VERGE, FULL_STACK. `compute_signed_improvement` (honest signed delta, no clamping). CI-safe simulated mode; blocked on GPU failure.
- **HumanEval code verification benchmark (Exp 341 / REQ-BENCH-004):** `CodeExtractor + VerifyRepairPipeline` on 50 HumanEval-style problems. `compute_pass_at_1`, `compute_pass_at_1_after_repair`, `build_humaneval_artifact` (schema `carnot.humaneval_benchmark.v1`).
- **ConstraintTemplateLibrary — Tier 2 constraint addition (Exp 343 / REQ-LEARN-017/018):** 4 built-in Eidoku-taxonomy templates (`carry_check`, `sign_check`, `unit_consistency`, `comparison_direction`). Patterns observed above `min_frequency` threshold activate the template and inject new constraints additively into the pipeline.
- **CaseMemory → ConstraintTemplateLibrary wiring + benchmark (Exp 344 / REQ-LEARN-019):** `CaseMemoryTemplateWiring.on_violation_recorded()` fires `observe_pattern()` on every recorded violation. Constraint-addition benchmark (200 simulated questions, seed=42): Control group shows 0% detection; Treatment group confirms **positive improvement_delta** after `carry_check` activates at 5 violations. Hypothesis confirmed.
- **SessionMemory — multi-session persistence (Exp 345 / REQ-LEARN-020/021):** `SessionMemory(storage_dir, model_id)` serialises `CaseMemory`, `ConstraintTemplateLibrary`, and `PerModelFPTracker` to disk across process restarts. `VerifyRepairPipeline` gains optional `session_memory` param and `close()` save method.
- **EORM CoT energy reward model (Exp 346 / REQ-LEARN-022/023):** pure-JAX transformer encoder (`EORMModel`, `EORMTrainer`) implementing arXiv 2505.14999. Contrastive hinge loss: `max(0, E_correct - E_incorrect + margin)`. AUC-ROC evaluation; saves `results/eorm_model_346.safetensors`.
- **JEPA real-data retrain on live violation pairs (Exp 347 / REQ-LEARN-024):** `extract_violation_pairs` word-tokenises Exp 340 responses and splits at `prefix_fraction=0.5`. `JEPARetrainer` trains with binary BCE loss and evaluates trapezoidal AUC-ROC. CI-safe synthetic fallback (50 pairs) when live GPU data unavailable.
- **SinkProbe attention-sink pre-filter (Exp 348 / REQ-VERIFY-086/087):** implements arXiv 2604.10697 as the first gate in the three-tier pipeline (SinkProbe → EORM → Ising). `compute_sink_concentration` accepts (n_heads, seq_len, seq_len) attention tensors; `SinkProbe(threshold=0.3)` sets `is_uncertain = mean_sink_score < threshold`. Simulated benchmark: **skip_rate=60%, FNR=0%, TNR=100%** — 60% fewer Ising calls with no false negatives.

### Milestone 2026.05.20 (Exps 351-364)

- **Live GPU Diagnostic — silent fallback bug fixed (Exp 352 / REQ-INFRA-014):** Diagnosed and fixed the critical bug where `CARNOT_FORCE_LIVE=1` was silently ignored by the conductor, causing Exps 340/341/346/347 to run in simulated mode despite live-capable RTX 3090s. `LiveGPUDiagnostic` now raises `RuntimeError` when forced-live prewarm fails instead of silently falling through.
- **Live GPU Smoke Test Gate (Exp 353 / REQ-BENCH-005):** `run_smoke_test()` gates any benchmark experiment launch; CI-safe when `CARNOT_FORCE_LIVE` not set; raises on live GPU unavailability when forced.
- **Adversarial GSM8K harness + execution (Exp 354/355 / REQ-BENCH-006/007):** Apple adversarial GSM8K benchmark (arXiv 2410.05229). Three-condition runner: standard / adversarial / repaired-adversarial. `honest_verdict=improvement_positive` gated on `inference_mode==live_gpu AND repair_improvement>0` — never emitted for simulated results. Live execution pending RETRO-012 fix.
- **LLMz3Formalizer — LLM-guided Z3 formalization (Exp 357 / REQ-EXTRACT-019/020):** Implements arXiv 2601.04675 (80% Z3 success rate improvement via task decomposition). `LLMz3Formalizer` extracts Z3 constraints via structured LLM prompting with sandboxed execution (restricted `__import__`, `print→StringIO`).
- **Three-tier pipeline complete (Exp 360 / REQ-VERIFY-088):** `ThreeTierPipeline(SinkProbe → EORM → Ising)` with early-exit at each tier. `verify()` returns `(verified, tier_used, energy)`. Simulated: 30% fewer Ising calls from SinkProbe alone; 60%+ combined skip rate.
- **Three-tier self-learning relay (Exp 361 / REQ-LEARN-026/027):** End-to-end relay across all three tiers. Simulated run: batch1_accuracy=0.60 → batch4_accuracy=0.72 (`improved=True`); all 4 Tier 2 templates activated. `honest_verdict=synthetic_only` — live GPU required for `learning_confirmed`.
- **SAVeR multi-turn verification wrapper (Exp 362 / REQ-AGENT-001/002):** `SAVeRVerifier` implements the arXiv 2604.08401 auditor-before-commit loop for multi-step agent reasoning chains. Goal #4 from research-program.md complete.
- **EORM real-data retrain (Exp 359 / REQ-LEARN-025):** `retrain_mode=synthetic_only` (5 real HumanEval pairs with unique question IDs — no cross-pair contrastive triples). `honest_verdict=synthetic_only`. Fixed `_pairs_to_contrastive_triples` bug: synthetic question IDs now routed to shared pool.
- **ModelServer + TensorRT + DualGPU wiring (Exp 364):** Infrastructure wiring — ModelServer, TensorRT, and DualGPU inference acceleration integrated into all benchmark harnesses for consistent hardware-accelerated testing.
- **Milestone 2026.05.20 retrospective (Exp 363):** 11/12 experiments ran (Exp 356 LLMExtractor skipped). 366 min total, mean 33.3 min/exp. RETRO-012 (CARNOT_FORCE_LIVE bug) is the critical blocker for all live GPU headline results.

### Milestone 2026.05.27 (Exps 365-376)

- **RETRO-012/013/014 close (Exp 365):** `scripts/conductor_gpu_env.sh` created with `CARNOT_FORCE_LIVE=1`; RetroJSONEnforcer pattern established for mandatory result JSON production; LLMExtractor gap documented. RETRO-012/013/014 formally closed.
- **LLMConstraintExtractor (Exp 366):** Second LLM call extracts structured arithmetic claims from free-form IT model output. `LLMConstraintExtractor` added to `python/carnot/pipeline/llm_extractor.py`; prompted extraction with fallback to regex; 100% module coverage.
- **Live extraction comparison harness (Exp 367):** Three-way head-to-head: `ArithmeticExtractor` vs `LLMConstraintExtractor` vs `LLMz3Formalizer` on 30 GSM8K questions with Gemma4-E4B-it (GPU0) + Qwen3.5-0.8B (GPU1). Hard `CARNOT_FORCE_LIVE=1` gate; `honest_verdict=live_gpu_winner` only when all results confirmed live. Live run pending.
- **Live precision pipeline benchmark v2 (Exp 368):** Hard CARNOT_FORCE_LIVE gate; schema=`carnot.precision_benchmark.v2`; `honest_verdict=live_improvement` only when `inference_mode==live_gpu AND signed_improvement>0`. First credible precision-stack headline number pending live run. 74 tests pass.
- **Live HumanEval benchmark v2 (Exp 369):** Hard CARNOT_FORCE_LIVE gate (3-stage: env + diagnose_live_gpu + model_load); `CodeExtractor + VerifyRepairPipeline`; PBT via determinism/idempotency checks; `honest_verdict=code_verification_positive` only on live GPU with positive signed improvement. 69 tests pass.
- **Live adversarial GSM8K benchmark v2 (Exp 370):** Hard CARNOT_FORCE_LIVE gate (`diagnose_live_gpu_or_raise()` — raises RuntimeError, NO simulated fallback); three conditions (standard/adversarial/repaired-adversarial) with LLMConstraintExtractor for repair; `honest_verdict=improvement_positive` gated on `inference_mode==live_gpu`. 23 tests pass; `SCENARIO-BENCH-022` added to spec.
- **EORM retrain on real pairs v2 (Exp 371):** Retrains EORM model on real CoT pairs from Exp 368/369/370; `eorm_model_371_real.safetensors` when live GPU available; priority fallback: 371_real → 346_synthetic in downstream experiments.
- **Three-tier pipeline live benchmark v2 (Exp 373):** Hard `CARNOT_FORCE_LIVE=1` gate via `diagnose_live_gpu()`; Beta-mixture approximate attention (realistic sink distribution vs Exp 360 binary); `compute_honest_verdict()` with 4-branch conservative reporting; `artifact_type=carnot.three_tier_benchmark.v2`. 80 tests pass; `SCENARIO-VERIFY-118/119` added to spec.
- **FR-11 self-learning relay live run (Exp 374):** Three-tier online self-learning relay with real models; `learning_confirmed` verdict gated on `inference_mode==live_gpu`; honest_verdict=synthetic_only without live GPU.
- **CIKAN energy layer (Exp 375):** Clifford-Informed KAN-Ising hybrid energy model (`CIKANEnergy`) for geometric constraint representation. Note: `cikan_energy.py` delivered as JSON not Python — RETRO-018 opened for reimplementation.
- **Milestone 2026.05.27 retrospective (Exp 376):** 11 experiments (Exps 365-375), mean=22.7 min/exp (apparent speedup from fast-fail blocked experiments, not genuine GPU work). `live_gpu_confirmed=False` for FOURTH consecutive milestone — RETRO-015 (critical) opened. New RETRO-015/016/017/018. 78 tests pass. `results/operational_retro_2026_05_27.json`.

### Milestone 2026.06.03 (Exps 377-389) — 15th Milestone

- **Live GPU infrastructure fix (Exp 377):** `LiveGPUGate` class + `session_startup.sh` export of `CARNOT_FORCE_LIVE=1`. Formally closes RETRO-015 at the infrastructure level. GPU node was offline during session — RETRO-019 escalation opened for execution-environment failure.
- **Combined EORM+JEPA retrain (Exp 383):** Trains EORM on contrastive triples and JEPA on binary violation pairs from live CoT pairs (Exps 379-382). `schema=carnot.combined_retrain.v1`; `honest_verdict=insufficient_pairs` (Exps 379-382 live files empty — RETRO-015 upstream). `eorm_model_383_real.safetensors` + `jepa_predictor_383_real.safetensors` written when pairs available. 41 tests pass.
- **Precision / HumanEval / adversarial / extraction harnesses (Exps 379-382):** Scripts created with hard `CARNOT_FORCE_LIVE=1` gates; all returned `status='partial'` because GPU node was offline. Live run pending once GPU is confirmed online.
- **Milestone 2026.06.03 retrospective (Exp 389):** 12 experiments (Exps 377-388, with Exps 378/386/387 missing due to session interruption), mean=19.9 min/exp. `live_gpu_confirmed=False` for FIFTH consecutive milestone. RETRO-019 (GPU node offline), RETRO-020 (CIKAN not implemented), RETRO-021 (FR-11 relay third carry) opened. RETRO-015 closed at infrastructure level. 115 tests pass. `results/operational_retro_2026_06_03.json`.

### Milestone 2026.06.10 (Exps 390-402, Exp 403 retro) — 16th Milestone

- **GPU preflight gate (Exp 390):** `scripts/experiment_390_gpu_preflight.py` created. GPU NOT confirmed live — RETRO-019 unresolved (script confirmed present, GPU node still offline).
- **JitRL constraint memory (Exp 392), Safety KAN classifier (Exp 393):** No result JSONs — fast-path did not execute inference code.
- **Precision / HumanEval / adversarial / extraction v3 harnesses (Exps 394-397):** All returned `status='partial'` — GPU node offline for SIXTH consecutive milestone.
- **FR-11 self-learning relay (Exp 399):** Partial — `honest_verdict='learning_confirmed'` NOT achieved; FOURTH consecutive miss (RETRO-024 opened).
- **Milestone 2026.06.10 retrospective (Exp 403):** 13 experiments (Exps 390-402), mean=7.5 min/exp. All experiments ran in "deliverable already exists" fast-path mode — no actual inference work. `live_gpu_confirmed=False` for SIXTH consecutive milestone. RETRO-022 (CRITICAL HUMAN ESCALATION: GPU node must be powered on or cloud GPU rented before next milestone), RETRO-023 (CIKANEnergy third consecutive failure — corrupt JSON fast-path), RETRO-024 (FR-11 relay fourth carry) opened. 138 tests pass. `results/operational_retro_2026_06_10.json`.

### Milestone 2026.06.17 (Exps 404-411, in progress) — 17th Milestone

- **Deliverable content validator + GPU preflight v2 (Exp 404):** `DeliverableContentValidator` implemented in `python/carnot/pipeline/deliverable_validator.py` with `ast.parse()` + `json.loads()` pre-check. Root cause of RETRO-023 (corrupt JSON fast-path) formally fixed. Preflight v2 result: `honest_verdict=env_not_propagating` — GPU hardware IS present (`is_live_capable=True`), but `source scripts/session_startup.sh` was not run before the conductor session. 53 tests pass.
- **Live precision pipeline v3 harness (Exp 410):** Preflight gate detected `env_not_propagating` and correctly blocked without simulation fallback. 34 tests pass. No inference executed.
- **Live HumanEval v3 harness (Exp 411):** Same preflight gate path as Exp 410. 44 tests pass; 4-gate sequence implemented (preflight JSON check → LiveGPUGate → setup_gpu → model load). Full suite: **3,058 passed, 2 pre-existing failures**.
- **Path to first live results:** Run `source scripts/session_startup.sh` before the next conductor session. Exp 404 confirms the GPU hardware is present and will be live-capable once the env variable propagates.

### HuggingFace Published Models (Exp 293 / v0.2.0-research)
> **Exp 304 (2026-04-14):** Upload confirmed. Credentials verified via Python API. FCV artifact live at https://huggingface.co/Carnot-EBM/carnot-formal-claim-verifier-v1.


Two Phase 1 research artifacts are published to [Carnot-EBM](https://huggingface.co/Carnot-EBM) on HuggingFace Hub:

| Artifact | Repo | Format | Notes |
|----------|------|--------|-------|
| Exp 66 joint EBM + Ising | [Carnot-EBM/carnot-joint-constraint-v1](https://huggingface.co/Carnot-EBM/carnot-joint-constraint-v1) | safetensors | Phase 1 prototype. 1.0 AUROC on held-out validation (simulated training). |
| FormalClaimVerifier | [Carnot-EBM/carnot-formal-claim-verifier-v1](https://huggingface.co/Carnot-EBM/carnot-formal-claim-verifier-v1) | ONNX + Python | Arithmetic and comparison routes as ONNX (opset 13); set_membership + boolean_entailment as pure Python. |

> Both are tagged `v0.2.0-research`. Phase 1 research prototypes — not production quality.

### Revalidation Sweep (Exp 271-279)

Re-ran 9 promising pre-provenance experiments with live or live-representative data and modern extractors. 6 CONFIRMED, 2 INCONCLUSIVE, 0 definitively ruled out.

| Exp | Approach | Classification | Key Result |
|-----|----------|----------------|------------|
| 271 | GlobalConsistencyChecker multi-turn | **CONFIRMED** | 100% detection, 0% FP, 1.91ms/call — matches synthetic baseline |
| 272 | Tier 1 self-learning on live traces | INCONCLUSIVE | 86% FP reduction (7→1) confirmed; task-success rate flat at 32.7% |
| 273 | Agent rollback verification | **CONFIRMED** | 100% rollback success + 100% violation detection (canned outputs) |
| 274 | FactualKBExtractor on IT model | **CONFIRMED** | 45% coverage (target 40%), 100% accuracy (target 75%) |
| 275 | Adaptive KAN on live traces | **CONFIRMED** | AUROC 0.991 on Exp 219-221 traces; AMR pruned 17 params, 0 AUROC gain |
| 276 | Z3+LLM+semantic on GSM8K | **CONFIRMED** | Z3+LLM: 80% detection / 0% FP; semantic: 0% detection / 20% FP for arithmetic |
| 277 | Combined verification signals | INCONCLUSIVE | Conductor OK, 3068 tests pass, but results JSON absent — needs re-run |
| 278 | Cross-session constraint memory | **CONFIRMED** | 100% warm hit rate, 0% FP unseen slice, session boundary verified, avg score 95.67 |
| 279 | Adversarial number-swapped GSM8K | **CONFIRMED** | Stale detection 100%, fresh-wrong 0%, FP 20%, lift +40pp |

Full results: `results/revalidation_sweep_271_279_summary.json`.

### Infrastructure

| Component | Result | Experiment |
|-----------|--------|------------|
| Three extractors | Regex 5/91 FP, Z3 3/91 FP, LLM 1/91 FP | Exp 206-207 |
| Verify latency | 0.006ms per constraint check | Exp 102 |
| Parallel Ising sampler | 183x faster than thrml | Exp 102 |
| CoT monitorability | Free-form 100% parseable, JSON 18% | Exp 213 |
| Dual-GPU parallel | 1.14x speedup (Thunderbolt bottleneck) | Exp 225 |

### What works on test sets but fails in practice

| Approach | Test Accuracy | Practical Result | Why It Fails |
|----------|-------------|-----------------|-------------|
| Per-token EBM (best) | 88.5% | 50% on real questions | Detects confidence, not correctness |
| Multi-layer concat | 81.3% | Not tested in deployment | Same fundamental limitation |
| Activation steering | 0% effect | N/A | Statistical ≠ causal |
| Cross-model transfer | ~50% (chance) | N/A | Model-specific representations |
| Cross-domain training | 70.8% (worse) | N/A | Domain-specific signals |

**The core problem:** activation-based EBMs measure how confident the model is, not whether it's right. A model that confidently says "Neil Armstrong walked on Mars" produces activations indistinguishable from "Neil Armstrong walked on the Moon." The EBM rewards confident hallucination and penalizes correct hedging — the exact opposite of what a hallucination detector should do.

See the [technical report](docs/technical-report.md) for the full research record.

## 14 Principles Learned

Hard-won lessons from the activation-based phase of a research program that now spans 411 experiments across 17 milestones and 16 model families. These negative results are the project's primary contribution — they document what doesn't work and why, saving other researchers months of dead ends.

### What works
1. **The model's own logprobs are the best energy.** No external EBM needed for rejection sampling — the LLM's own confidence is already an energy function. Simple, practical, +10%.
2. **Different energy signals dominate in different domains.** Logprobs for QA, structural tests for code. The composite combines both and is never worse than either alone.
3. **Multi-layer concatenation improves test-set detection by ~6%.** Concatenating activations from layers 4+12+24 achieves 81.3% vs 75.5% for the final layer alone.

### What doesn't work (and why)
4. **Activation EBMs detect confidence, not correctness.** The fundamental limitation. Confident hallucinations produce activations indistinguishable from confident correct answers. Test-set accuracy (75-88%) does not translate to practical detection (50%).
5. **Instruction tuning compresses the hallucination signal.** Base models: 86.8%. Instruction-tuned: 75.0%. RLHF makes models sound confident even when wrong.
6. **Chain-of-thought compresses it further.** Disabling thinking improves detection from 61.3% → 75.5%. Thinking makes hidden states more uniform.
7. **Statistical difference ≠ causal influence.** A direction that separates correct from hallucinated activations does NOT steer the model when injected during generation.
8. **Adversarial questions defeat post-hoc detection.** On TruthfulQA, neither logprob nor EBM rejection improves over greedy.
9. **Hallucination representations are model-specific.** Cross-model transfer is at chance (~50%). Each model needs its own EBM.
10. **EBM detection is domain-specific.** Mixing datasets hurts (70.8% < 75.5%). Mixing temperatures hurts. Train on your target domain only.
11. **Normalization doesn't enable transfer.** Z-score, L2, and PCA whitening all destroy signal without improving cross-domain or cross-model transfer.
12. **Upstream question-level detection is weak.** The model's representation of the question partially predicts hallucination (62.6%) but not usefully.

### Scaling observations
13. **EBM accuracy scales with model size** within a family. Qwen3.5: 75.5% (0.8B) → 88.5% (27B). But this is test-set accuracy — the confidence-vs-correctness problem applies at all scales.
14. **MoE architectures vary wildly.** Qwen3.5-35B has 256 genuinely specialized experts (0.008 overlap). Mixtral has 8 near-identical experts (0.997 overlap). Fundamentally different knowledge organization.

## Tools

### CLI

```bash
pip install -e .
carnot verify examples/math_funcs.py --func gcd --test "(12,8):4" --test "(7,13):1"
```

### MCP Server

Configure with `cp .mcp.json.example .mcp.json` for Claude Code integration. The hardened stdio JSON-RPC server now exposes **7** tools: `verify_code`, `verify_with_properties`, `verify_code_with_pbt`, `verify_llm_output`, `verify_and_repair`, `list_domains`, and `health_check`. These tools perform **Python code verification** and pipeline checks — they do not implement the activation-based EBM hallucination detection described in the research sections.

See [docs/usage-guide.md](docs/usage-guide.md) for detailed setup and usage instructions.

## Model Tiers

| Tier | Name | Scale | Use Case |
|------|------|-------|----------|
| Large | **Boltzmann** | Deep residual + attention | Research frontiers, large-scale generation |
| Medium | **Gibbs** | Multi-layer MLP (2-4 hidden) | Applied ML, complex pattern learning |
| Efficient | **KAN** | Learnable B-spline edges | **Default for verification** — best accuracy/param ratio (0.994 AUROC, 2.3K params) |
| Hardware | **Ising** | Pairwise quadratic | **Real-time sampling + FPGA/TSU** — direct hardware mapping, fastest parallel Gibbs |

All tiers implement the same `EnergyFunction` trait (Rust) / protocol (Python), so algorithms written against the interface work with any tier.

**When to use which:** KAN is the default for constraint verification (most accurate per parameter). Ising is for real-time guided decoding and hardware deployment (fastest sampling, maps to physical p-bits). They complement each other — KAN for accuracy, Ising for speed.

### Hardware Path

The current hardware track is the [FPGA Ising design](docs/fpga-ising-design.md) for a KV260-class sparse **4,096-spin** backend. Exp 228 validates the AXI-Lite upload/trigger/readback contract in **software simulation** with the new `FPGAIsingSampler` backend; the checked-in `fpga_sim` timing (`0.824549s` on a 128-spin sparse problem) is explicitly a software-model artifact, not a synthesized FPGA throughput claim. Exp 242 now attempts the real KV260 round trip and records the honest blocker in this environment: no `CARNOT_KV260_BITFILE` path was configured, so no board timings were fabricated. Exp 243 then replays **460** saved semantic and code repair cases through the sampler path: the CPU reranker leaves top-1 quality flat at **30.2%**, leaves verifier precision flat at **30.65%**, and keeps the KV260-backed path blocked in the same environment.

## Architecture

```
carnot/
├── crates/                        # Rust workspace
│   ├── carnot-core/               # EnergyFunction trait, types, serialization
│   ├── carnot-ising/              # Ising tier: E(x) = -0.5 x^T J x - b^T x
│   ├── carnot-gibbs/              # Gibbs tier: multi-layer energy network
│   ├── carnot-boltzmann/          # Boltzmann tier: deep residual energy network
│   ├── carnot-samplers/           # Langevin dynamics + HMC samplers
│   ├── carnot-training/           # CD-k, score matching, optimizers
│   └── carnot-python/             # PyO3 bindings
├── python/carnot/                 # Python/JAX package
│   ├── core/                      # Energy function protocol, model state
│   ├── models/                    # Ising, Gibbs, Boltzmann in JAX
│   ├── samplers/                  # Langevin, HMC, ParallelIsingSampler
│   ├── training/                  # JAX training loops with Optax
│   ├── pipeline/                  # VerifyRepairPipeline, extractors, errors
│   ├── mcp/                       # MCP server for Claude Code integration
│   ├── verify/                    # ComposedEnergy, ConstraintTerm, repair
│   └── inference/                 # EBM loader, composite scorer, LLM solver
├── openspec/capabilities/         # Specification-driven contracts
│   ├── core-ebm/                  # REQ-CORE-*, SCENARIO-CORE-*
│   ├── model-tiers/               # REQ-TIER-*, SCENARIO-TIER-*
│   └── training-inference/        # REQ-TRAIN-*, REQ-SAMPLE-*
├── _bmad/                         # Strategic docs (PRD, architecture, traceability)
└── ops/                           # Operational status, changelog, test results
```

## Quick Start

### Rust

```bash
cargo build --workspace --exclude carnot-python
cargo test --workspace --exclude carnot-python
```

### Python

```bash
pip install -e ".[dev]"
pytest tests/python
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Development Philosophy

Carnot follows **spec-anchored development**:

1. **Spec first** — every feature starts as REQ-* and SCENARIO-* in OpenSpec
2. **Tests trace to specs** — every test references the requirement it verifies
3. **100% coverage** — code coverage and spec coverage enforced by pre-commit hooks
4. **Dual implementation** — Rust for performance, Python/JAX for research iteration
5. **Cross-language interop** — safetensors serialization + PyO3 bindings

See [CLAUDE.md](CLAUDE.md) for the full development workflow.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Core compute (Rust) | ndarray, rayon |
| Core compute (Python) | JAX, Flax, Optax |
| Python-Rust bridge | PyO3, maturin |
| Serialization | safetensors |
| Testing | cargo test, pytest, cargo-tarpaulin |
| Linting | rustfmt, clippy, ruff, mypy (strict) |

## Research References

Papers and resources that have informed Carnot's design and direction.

### Foundational EBM Theory

- [LeCun et al. (2006) — A Tutorial on Energy-Based Learning](https://web.stanford.edu/class/cs379c/archive/2012/suggested_reading_list/documents/LeCunetal06.pdf) — The foundational EBM tutorial establishing energy functions as a unifying framework for ML
- [Gutmann & Hyvarinen (2010) — Noise-Contrastive Estimation](https://proceedings.mlr.press/v9/gutmann10a.html) — NCE training for EBMs, used in Carnot's `nce_loss()`

### EBM Architecture and Scaling

- [Energy-Based Transformers are Scalable Learners and Thinkers (2025)](https://arxiv.org/abs/2507.02092) — EBTs: train transformers to assign energy to (input, prediction) pairs, infer via gradient descent. 35% faster scaling than Transformer++. Validates Carnot's verify-and-repair architecture at transformer scale.
- [Autoregressive Language Models are Secretly EBMs (2025)](https://arxiv.org/abs/2512.15605) — Explicit bijection between ARMs and EBMs via soft Bellman equation. Every LLM is already an EBM. Theoretical foundation for extracting energy signals directly from LLM logits.
- [Learning EBMs by Self-Normalising the Likelihood (2025)](https://arxiv.org/abs/2503.07021) — SNL: single learnable parameter for partition function. Lower bound of log-likelihood, concave for exponential families. Potential alternative to NCE for training learned verifiers.

### EBM + LLM Hallucination Detection

- [Semantic Energy: Detecting LLM Hallucination Beyond Entropy (2025)](https://arxiv.org/abs/2508.14496) — Energy = negative logit from penultimate layer. High energy = hallucination. 4-5% AUROC improvement over entropy methods. Directly applicable to Carnot's verification pipeline.
- [Spilled Energy in Large Language Models (2026)](https://arxiv.org/abs/2602.18671) — Energy-based analysis of LLM internals for hallucination detection.
- [Energy-Based Calibration for Implicit Chain-of-Thought (2025)](https://arxiv.org/abs/2511.07124) — EBM-CoT: refine latent reasoning toward low-energy regions. Gradient descent on reasoning trajectories.

### EBM for Physical Systems

- [Hybrid EBMs for Physical AI: Port-Hamiltonian Dynamics (2026)](https://arxiv.org/abs/2604.00277) — Separates visible (dynamical) from hidden (feedforward) layers. Absorbing invariant sets for stability. Validates Carnot's architecture of constraint evaluation (feedforward) + repair dynamics (gradient descent).
- [Cognitively Inspired Energy-Based World Models (2024)](https://arxiv.org/abs/2406.08862) — EBMs as cognitive world models.

### Agent Skill Learning

- [Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills (2026)](https://arxiv.org/abs/2603.25158) — Parallel analyst sub-agents extract lessons from execution traces, hierarchical consolidation merges them. Integrated into Carnot's autoresearch as the Trace2Skill learning layer.

### Open-Source EBM Frameworks

| Framework | Org | Language | Focus |
|-----------|-----|----------|-------|
| [EB-JEPA](https://github.com/facebookresearch/eb_jepa) | Meta FAIR | PyTorch | Self-supervised world modeling (JEPA) |
| [THRML](https://github.com/extropic-ai/thrml) | Extropic | JAX | Probabilistic graphical models for TSU hardware |
| [TorchEBM](https://github.com/soran-ghaderi/torchebm) | Independent | PyTorch | General-purpose EBM toolkit |
| [mini-ebm](https://github.com/yataobian/mini-ebm) | Educational | PyTorch | Minimal educational EBM implementation |
| [Kona 1.0](https://logicalintelligence.com/kona-ebms-energy-based-models) | Logical Intelligence | — | Continuous latent reasoning via EBMs |
| [UvA Deep Energy Models Tutorial](https://github.com/phlippe/uvadlc_notebooks) | UvA | PyTorch | Tutorial 8: deep energy-based models |
| [Equilibrium Matching](https://energy-based-model.github.io/) | — | — | EBM training via equilibrium matching |

### Hardware

- [FPGA Ising design](docs/fpga-ising-design.md) — KV260-class sparse **4,096-spin** overlay contract with `FPGAIsingSampler`, `SoftwareFPGAOverlay`, and AXI-Lite upload/trigger/readback semantics. Provenance: **software simulation** (Exp 228), not a live hardware-speed claim.
- [Extropic TSU/XTR-0](https://extropic.ai/writing/inside-x0-and-xtr-0) — Thermodynamic Sampling Unit for native EBM inference in hardware

## Pre-trained Models

16 per-token EBM models are available on HuggingFace at [huggingface.co/Carnot-EBM](https://huggingface.co/Carnot-EBM).

**Important caveat:** These models achieve 75-88% accuracy on held-out TruthfulQA test sets, but this metric is misleading. In practical deployment (8 real questions), the EBM agreed with ground truth only 50% of the time. The EBMs detect model *confidence*, not *correctness* — they are research artifacts for studying activation-space structure, not production hallucination detectors. See the [practical test results](scripts/experiment_practical_mcp_test.py).

| Model | Test Set Accuracy | Source Model | Notes |
|-------|----------|-------------|-------|
| `per-token-ebm-qwen35-27b-nothink` | 88.5% | Qwen3.5-27B | Highest test accuracy |
| `per-token-ebm-gemma4-e2b-nothink` | 86.8% | Gemma 4 E2B (base) | Best base model |
| `per-token-ebm-qwen35-9b-nothink` | 85.8% | Qwen3.5-9B | |
| `per-token-ebm-qwen35-35b-nothink` | 84.5% | Qwen3.5-35B-A3B | MoE, 256 experts |
| ... | 73-84% | 11 more models | See HuggingFace |

## License

Apache 2.0 — see [LICENSE](LICENSE).
