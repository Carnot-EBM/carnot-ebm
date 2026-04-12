# Carnot — Operational Status

**Last Updated:** 2026-04-12 — 185 EXPERIMENTS (incl. EXP 203: EXTRACTION AUTOPSY — REGEX MISSES 3/3 WRONG LIVE GEMMA ANSWERS AND FLAGS 3 CORRECT ONES) (incl. EXP 184: 3B/4B SCALING STUDY — VERIFY-REPAIR HURTS AT 4B ON ADVERSARIAL) (incl. Exp 101, 102, 108, 110, 112, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 134, 136, 137, 138, 139, 141, 143, 144, 145, 157, 158), 14 PRINCIPLES, 17 MODELS ON HUGGINGFACE, THRML/EXTROPIC INTEGRATION, 0.1.0-BETA1 SHIPPED, KAN ENERGY TIER, VERIFYPAIRPIPELINE PRODUCTION API, RUST VERIFYPIPELINE (NFR-01), DEFINITIVE MULTI-MODEL BENCHMARK (+10.2% avg improvement), ENERGY-GUIDED DECODING (EXP 110), FAST EMBEDDING BENCHMARK (EXP 112), V12 ARTIFACTS PUBLISHED TO HUGGINGFACE (EXP 118), ADVERSARIAL GSM8K DATASET GENERATOR (EXP 119), LLM ADVERSARIAL BASELINE (EXP 120), ADVERSARIAL VERIFY-REPAIR EXECUTED (EXP 121), ADVERSARIAL ROBUSTNESS DEEP ANALYSIS (EXP 122), ROBUST MODEL LOADER (EXP 123), CONSTRAINT STATE MACHINE FOR AGENT WORKFLOWS (EXP 125), AGENT ROLLBACK ON CONSTRAINT VIOLATION (EXP 126), MULTI-WORKFLOW CSM BENCHMARK 100% ACCURACY (EXP 127), LNN COUPLING-MATRIX ADAPTIVE MODEL (EXP 128), ONLINE LEARNING ADAPTIVE WEIGHTS (EXP 134), CROSS-SESSION CONSTRAINT MEMORY (EXP 136), HF GUIDED DECODING ADAPTER EXPORT (EXP 137), GUIDED DECODING BENCHMARK (EXP 138), ARXIV RESEARCH SCAN + NEXT-EXP PROPOSALS (EXP 139), CONSTRAINT GENERATION FROM MEMORY (EXP 141), JEPA TRAINING PAIRS COLLECTED (EXP 143), JEPA VIOLATION PREDICTOR (EXP 144), JEPA FAST-PATH GATE INTEGRATED (EXP 145), SPILLED ENERGY HALLUCINATION SIGNAL (EXP 157), FACTUAL EXTRACTOR WIKIDATA SPARQL (EXP 158)

## What's Working

### Core Framework (REQ-CORE-001–006)
- EnergyFunction trait (Rust) and protocol (Python/JAX)
- Four model tiers: Ising (both), Gibbs (both), Boltzmann (both), KAN (Python/JAX with Rust scaffold)
- LNN adaptive models (Python/JAX): `LNNConstraintModel` (Exp 116, hidden-state evolution) and `LiquidConstraintModel` (Exp 128, coupling-matrix evolution) — both implement EnergyFunction protocol with input-dependent dynamics for multi-step agent workflows; J-evolution (Exp 128) adapts constraint coupling strengths at inference time via BPTT-trained MLP ODE
- Samplers: Langevin + HMC in both languages, with gradient clipping (REQ-SAMPLE-004)
- Parallel Ising Gibbs sampler: 183x faster than thrml, checkerboard updates, simulated annealing (REQ-SAMPLE-003)
- thrml-compatible interface: accepts IsingEBM models, returns thrml-format samples
- Sampler backend abstraction: `SamplerBackend` protocol with CpuBackend (ParallelIsingSampler) and TsuBackend (stub for Extropic TSU hardware); switchable via `CARNOT_BACKEND` env var or `get_backend()` factory (Exp 71)
- Serialization: safetensors cross-language persistence
- PyO3 bindings: all 3 tiers + 2 samplers exposed to Python

### Training (REQ-TRAIN-001–006)
- Contrastive Divergence CD-k (Rust)
- Denoising Score Matching (Rust + Python/JAX)
- Noise Contrastive Estimation (Rust + Python/JAX)
- Self-Normalised Likelihood (Python/JAX)
- Optimization-through-training / Hessian-vector products (Python/JAX)
- Replay buffer for trajectory-aware training (Python/JAX)
- Adam optimizer with gradient clipping (Rust)

### Verifiable Reasoning (REQ-VERIFY-001–008)
- ConstraintTerm trait/protocol — constraints as energy terms
- ComposedEnergy — weighted composition with decomposition
- Verification certificates — VERIFIED/VIOLATED with per-constraint reports
- Gradient-based repair — violated-only, with Langevin noise (P6) + random steps (P11)
- Continuous-space gradient repair — embedding-space gradient descent + codebook decoding (Exp 87): 40% success on violated samples, 100% on arithmetic/scheduling
- Energy landscape certification — Hessian eigenvalue analysis, basin estimation
- Convergence guarantees — absorbing invariant sets (P10)
- Deterministic reproducibility
- Domains: SAT, graph coloring, Python code, property-based testing
- Rust built-in constraint primitives: BoundConstraint, EqualityConstraint, IsingConstraint (`carnot-constraints` crate, Exp 70)
- Serializable VerificationCertificate with JSON export (`carnot-constraints`, Exp 70)
- Rust VerifyPipeline: constraint extraction + composed energy verification in `carnot-constraints`; `VerifyPipeline`, `AutoExtractor`, `PipelineResult`; 10x-faster verification path for PyO3 hot loop (NFR-01, Exp 94)
- Sudoku example — full constraint satisfaction demo

### LLM-EBM Inference Pipeline (REQ-INFER-001–016)
- SAT/coloring constraint encoding + verify-and-repair
- LLM solver (Claude API bridge, local model)
- Logprob rejection sampling (+10% accuracy, experiment 13)
- Composite energy scorer (logprob + structural tests, experiment 14)
- Iterative refinement with feedback (LLM WITH EBM, not LLM then EBM)
- Multi-start repair, semantic energy, ARM-EBM bijection
- Diffusion generation (parallel solution from noise)
- Per-token EBM (84.5% test on Qwen3-0.6B, 67.2% on Qwen3.5-0.8B, experiments 19-22)
- Robust model loader (`carnot.inference.model_loader`, Exp 123): centralised `load_model()` + `generate()` API with RAM pre-check (psutil), float32-on-CPU default (avoids AVX2 crashes), OOM retry with gc.collect() + cuda.empty_cache(), Qwen3 enable_thinking fallback chain, `CARNOT_FORCE_LIVE` / `CARNOT_SKIP_LLM` / `CARNOT_FORCE_CPU` env vars; eliminates conductor subprocess fallback to simulated outputs (REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-003)

### HuggingFace Guided Decoding Adapter Export (Exp 137)
- `exports/guided-decoding-adapter/` — HuggingFace-publishable artifact packaging Exp-110 guided decoding results for community reuse
- `GuidedDecoder` class added to `python/carnot/inference/guided_decoding.py` with `from_pretrained(path_or_repo)` + `generate(model, tokenizer, prompt)` API delegating to `EnergyGuidedSampler`
- Artifacts: `config.json` (constraint types, default weights, latency profile), `constraint_weights.safetensors` (12 per-type float32 weights + default_alpha + default_energy_threshold), `README.md` (latency numbers, usage, limitations), `example.py` (10-line mock demo)
- 7 new tests in `tests/python/test_guided_decoding.py` — all pass, no regressions
- **PUBLISHED (Exp 164)**: `Carnot-EBM/guided-decoding-adapter` on HuggingFace (commit 3727dac, verified README 6419 bytes) (REQ-VERIFY-001, SCENARIO-VERIFY-004)

### Fast Embedding for Guided Decoding (Exp 112)
- `FastEmbeddingProtocol` + 5 strategies: MiniLM (3.1ms GPU), TF-IDF+projection (0.115ms), CharNgram (1.0ms), HashEmbedding (0.097ms), RandomProjection (0.026ms p50 — winner)
- `get_default_embedding(strategy)` factory in `carnot.embeddings.fast_embedding`
- Key finding: RandomProjection (byte histogram) wins — p99=0.040ms (92x faster than MiniLM GPU), AUROC=0.507 vs MiniLM 0.452 — constraint satisfaction signal not well-captured by semantic similarity; all embeddings AUROC 0.38–0.51
- Meets <1ms p99 guided decoding target with no AUROC regression vs MiniLM

### Activation Analysis (Phase 3)
- Activation extractor (per-layer transformer hooks)
- Hallucination direction (80% detection, 0.945 AUROC)
- Layer-targeted EBM, LayerNavigator, activation/weight steering
- Concept vectors (targeted prompting)
- Per-token activation dataset: 52,296 tokens (QA + TruthfulQA, Qwen3.5-0.8B)
- EBM-guided rejection sampling (experiment 23)
- Multi-layer hallucination probing (experiment 24, U-curve discovered)
- MCP server with score_candidates tool
- Hardened MCP server package (`carnot.mcp`): 6 tools (verify_code, verify_with_properties, verify_llm_output, verify_and_repair, list_domains, health_check); 30s timeout, 10K char limit, structured errors; runnable as `python -m carnot.mcp`

### Constraint-Based Reasoning (Phase 5-8)
- Arithmetic verification: QUBO encoding (8/12) + deterministic carry propagation (16/16)
- Logical consistency: 8/8 contradiction detection via Ising
- SAT solving: 5000 vars in 0.7s, +5.5% vs random at scale
- Code constraint extraction: AST → type/bound/return/init constraints (static, Exp 48)
- Runtime constraint instrumentation: dynamic AST rewriting with isinstance/bound/return assertions (Exp 53)
- Live LLM → constraint → Ising verification: Qwen3.5-0.8B end-to-end with 4-domain question set (Exp 56)
- Verify-Repair Loop: constraint violations → NL feedback → LLM regeneration → re-verify (up to 3 iters); architecture works, constraint coverage is the bottleneck (Exp 57)
- Constraint-Aware Prompting: preventive constraint injection into prompts vs post-hoc verification; 3 modes (baseline/constraint-aware/combined) on 15 questions across arithmetic, logic, factual domains (Exp 59)
- Unified ConstraintExtractor API: pluggable Protocol-based extractors (arithmetic, code, logic, NL) with AutoExtractor auto-detection + merge; `carnot.pipeline.extract` (Exp 74)
- VerifyRepairPipeline: user-facing API consolidating verify + repair into `carnot.pipeline.verify_repair`; verify-only and verify-and-repair modes (Exp 75)
- Pipeline error handling: structured error hierarchy (`carnot.pipeline.errors`) with CarnotError base + 5 subclasses (ExtractionError, VerificationError, RepairError, ModelLoadError, PipelineTimeoutError); wall-clock timeout support in VerifyRepairPipeline (Exp 82)
- Constraint state machine for agent workflows: `ConstraintStateMachine` in `carnot.pipeline.state_machine` wraps `VerifyRepairPipeline` for step-by-step agent framework integration; features: per-step StepResult audit records, deep-copy rollback to any prior step, contradiction detection (flags when new output violates a previously VERIFIED fact), `verified_facts()` + `pending_facts()` accessors; 662-line test suite at 100% coverage (Exp 125, REQ-VERIFY-001, SCENARIO-VERIFY-005)
- Agent rollback on constraint violation: `scripts/experiment_126_agent_rollback.py` validates `ConstraintStateMachine.rollback()` on multi-step reasoning; 0%→50% accuracy recovery via rollback+repair on 20 structured 4-step math problems; ArithmeticExtractor catches addition/subtraction violations (100% detection) but not multiplication (0%); `_SingleArgCompatPipeline` shim bridges `agentic.propagate()` single-arg `verify()` to `VerifyRepairPipeline` two-arg signature (Exp 126, REQ-VERIFY-001, SCENARIO-VERIFY-005)
- NL constraint extraction: pattern-based claim verification
- LLM self-constraint pipeline: 10/10 perfect (all hallucinations caught)
- Scheduling constraints: time slot exclusion, ordering, capacity
- Learned Ising via CD: 89/100 perfect, generalizes to unseen instances (Exp 50); scaled to 50/100/200 vars with L1 regularization and bootstrapped training data (Exp 60); sparse CD with clause-graph masking at 200/500/1000 vars, ~20x parameter reduction vs dense (Exp 61); domain-specific constraint learning on 10K triples across arithmetic/logic/code with 200+ binary features (Exp 62); hierarchical block-structured Ising with dense intra-block + sparse inter-block couplings, two-level Gibbs sampler, ~10x param reduction at 1000 vars (Exp 63)
- Cross-domain transfer: structure-dependent transfer validated
- Ising-guided fuzzing: energy landscape generates adversarial test inputs for differential testing of LLM code; 8 bug types covered (Exp 54)
- Trace-learned constraints: discriminative Ising trained on correct/buggy execution traces catches semantic bugs invisible to static+dynamic analysis (Exp 55)
- Multi-domain live benchmark: 500 questions across 5 domains (arithmetic, code, logic, factual, scheduling) in 3 modes (baseline/verify/verify-repair); first comprehensive pipeline evaluation (Exp 58)
- Multi-model constraint transfer: validates constraint pipeline (arithmetic, logic, code AST, factual KB) on Qwen3.5-0.8B and Gemma4-E4B-it without retraining; tests model-agnostic verification (Exp 69)
- End-to-end differentiable constraint reasoning: fully differentiable text → embedding → constraints → continuous Ising → MLP → score pipeline; joint model 1.0 test AUROC (vs 0.54 Ising-only, 0.98 embedding-only); validates Ising adds discriminative power beyond embeddings; stable gradients; 5 domains (Exp 66)

### GPU Compute
- carnot-gpu: wgpu Vulkan backend (AMD Radeon 890M, tested) — **DEPRECATED:** not used by current pipeline. Retained for potential future browser/edge deployment or GPU training experiments.
- carnot-webgpu-gateway: distributed browser GPU compute — **DEPRECATED:** not used by current pipeline. Retained for potential future distributed training or browser-based verification.
- ROCm 7.2: PyTorch 2.11.0+rocm7.2, native gfx1150, 3.3x speedup on Qwen3

### Autoresearch Pipeline (REQ-AUTO-001–014)
- Benchmark suite: DoubleWell, Rosenbrock, Ackley, Rastrigin, GaussianMixture (Rust + Python/JAX)
- Benchmark runner with baseline recording (JSON)
- Process-level sandbox (dev): import blocking, timeout, I/O capture
- Docker+gVisor sandbox (production): 5-layer defense in depth
- Three-gate evaluator: energy, time (with JIT grace period), memory
- Experiment log: append-only audit trail with rejected registry
- Orchestrator: full propose → sandbox → evaluate → log → update loop
- Generator-based orchestrator: lazy LLM hypothesis generation with failure feedback
- Claude Code API bridge: Docker container wrapping `claude -p` as OpenAI API
- Circuit breaker: halts after N consecutive failures
- Cross-language validation: test vector generation + conformance checking
- Automatic rollback: git-based revert on production energy regression
- Trace2Skill learning layer (REQ-AUTO-011–014): trajectory analyst, skill directory, hierarchical consolidation, cross-tier transfer
- Self-improving code verifier
- Ising constraint-satisfaction "fourth gate": self-verification of autoresearch hypothesis outputs via claim extraction + ComposedEnergy + Ising sampling (Exp 72)
- Research conductor (autonomous Claude Code agent loop)
- Research conductor: YAML-driven (research-roadmap.yaml), CalVer milestones, self-healing
- ROCm 7.2 JAX support validated (gfx1150 iGPU), thrml crash filed as extropic-ai/thrml#41

### JEPA Predictive Verification (Exp 143)
- `results/jepa_training_pairs.json` — labelled `(partial_response_embedding, final_violated)` dataset for JEPA early-exit verification training
- Data sources: log-mined pairs from Exp 120–140 + 200 synthetic arithmetic questions with correct/wrong LLM-style responses
- Prefix ratios: 10%, 25%, 50%, 75% of whitespace-tokenized response
- Embedding: RandomProjectionEmbedding(embed_dim=256, seed=42) (~0.026ms/call, L2-normalized)
- Schema: `{pairs:[{prefix_ratio, embedding[256], violated_arithmetic, violated_code, violated_logic, any_violated, domain, source_exp}], total, domain_counts, positive_rate, negative_rate}`
- Enables Tier 3 Goal #2: train predictor to flag constraint violations at token 50 instead of waiting for full response (REQ-JEPA-001)

### Autoresearch Results
- **10-iteration run (Sonnet)**: DoubleWell 0.9483 → 0.1604 (83% energy reduction), 3 accepted hypotheses (HMC, annealing)
- **50-iteration run (Sonnet)**: DoubleWell 0.0001, Rosenbrock 0.0092 (both near optimal). Circuit breaker at iteration 18.

### PyPI Packaging (Exp 78)
- Pure-Python install via `pip install carnot` (no Rust toolchain required)
- Rust bindings optional: `RUST_AVAILABLE` flag in `carnot._rust_compat`
- Single-source version: `carnot._version.__version__`
- Extras: `carnot[mcp]`, `carnot[rust]`, `carnot[all]`, `carnot[cuda]`, `carnot[llm]`
- Build backend: setuptools (maturin config preserved for Rust extension builds)

### Integration Examples (Exp 79)
- 5 production-ready examples in `examples/`: API response verification, code review pipeline, batch verification, custom domain-specific extractor, MCP server integration
- Standalone scripts with `JAX_PLATFORMS=cpu` for reproducibility
- JSON batch input format for bulk verification workflows

### Getting Started Documentation (Exp 80)
- `docs/getting-started.md`: installation guide + first verification walkthrough
- `docs/concepts.md`: EBM fundamentals, constraint verification, pipeline architecture
- `docs/api-reference.md`: full API reference for pipeline, extractors, MCP server, samplers, models
- Updated `docs/index.html` navigation linking new documentation pages

### Beta Release Preparation (Exp 85)
- `RELEASE_NOTES.md`: Carnot 0.1.0-beta1 release notes (highlights, included packages, known limitations)
- `scripts/prepare_release.py`: automated release readiness checker (version consistency, unit tests, CLI, examples, docs)
- `README.md`: install instructions + quick-start Python API example

### Self-Verification Dogfooding (Exp 84)
- `scripts/dogfood_carnot.py`: exercises CodeExtractor, AutoExtractor, and VerifyRepairPipeline against Carnot's own Python source code
- Surfaces constraint violations, docstring/signature mismatches, correlates findings with test failures
- Self-verification: the verification pipeline verifies itself (REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002)

### Pipeline Performance Benchmarks (Exp 83)
- `scripts/benchmark_pipeline.py`: verify() latency, extraction scaling, batch throughput, memory profiling
- Results in `ops/benchmark-results.md`: all domains sub-millisecond p99, 36,887 calls/s throughput, zero memory growth
- Extraction scales linearly with input length (0.05ms at 50 chars → 2.41ms at 5000 chars)

### Integration Test Suite (Exp 81)
- `tests/integration/test_full_pipeline.py`: full verify-repair pipeline E2E with real extractors and JAX energy (no mocks)
- `tests/integration/test_cli_commands.py`: CLI subprocess tests for `carnot verify` and `carnot score` subcommands
- `tests/integration/test_install.py`: package importability, version exposure, console_scripts entrypoint, public module accessibility
- Shared `conftest.py` with `JAX_PLATFORMS=cpu` fixture for reproducibility

### Quality Infrastructure
- 1049 Python tests + 104 Rust tests, 100% code coverage, 100% spec coverage
- Pre-commit hooks: rustfmt, clippy, ruff, mypy, pytest, spec coverage
- Docker compose: Claude API bridge + WebGPU gateway (`make up`)

### Constraint Mining & Self-Bootstrapping (Exp 88-89)
- Failure-driven constraint mining: analyzes pipeline false negatives, categorizes 6 gap types (implicit_logic, comparison, arithmetic_chain, negation, world_knowledge, code_semantics), suggests new extraction patterns with estimated 75% coverage improvement (`carnot.pipeline.mining`)
- Self-bootstrapped Ising training: trains discriminative Ising using pipeline verification outputs as supervision (no manual labels); 0.788 AUROC combined; arithmetic/logic perfect (1.0), code strong (0.91); 96.7% pipeline concordance; scales with data (100→700 samples)

## Experiment Results (26 experiments)

| # | Approach | Result | Verdict |
|---|----------|--------|---------|
| 2 | SAT gradient repair (Haiku) | 60% → 80% | ✅ |
| 8 | Activation detection | 80% / 0.945 AUROC | ✅ Detection |
| 9-12 | Activation rejection sampling | -5% to -25% | ❌ Overfits |
| 13 | **Logprob rejection** | **+10%** | **✅ Best simple** |
| 14 | **Composite (logprob + structural)** | **0% → 30%** | **✅ Best for code** |
| 15-16 | Activation steering | 0% change | ❌ No causal effect |
| 17 | Concept-specific vectors | All < 56% | ❌ Worse than generic |
| 19 | **Per-token EBM** | **71.8% test** | **✅ First activation that generalizes** |
| 20 | Concept steering | 0% change | ❌ Confirms #15-16 |
| 21 | **Scaled per-token EBM (Qwen3-0.6B)** | **84.5% test** | **✅ More data helps** |
| 22 | TruthfulQA + Qwen3.5-0.8B | 67.2% test | ⚠️ Better models = subtler signals |
| 23 | EBM rejection sampling (TruthfulQA) | -3% to -6% | ❌ Adversarial QA defeats rejection |
| 24 | Multi-layer probing | Final layer best (64%) | ⚠️ U-curve: signal at layers 4 and 24 |
| 25 | **No-thinking mode** | **75.5% vs 61.3%** | **✅ Thinking compresses signal by 14.2%** |
| 26 | Cross-model EBM transfer | 49.8% cross vs 86.2% self | ❌ Model-specific representations, no universal detector |
| 27 | Upstream detection (question-level) | 62.6% mean | ⚠️ Weak signal, question reps partially predict hallucination |
| 28 | **Multi-layer concatenation** | **81.3% vs 75.5%** | **✅ Layers 4+12+24 improve by 5.8%** |
| 29 | Layer gating vs concat | All-concat 79.2%, gating 62.8% | 3-layer concat is sweet spot; learned gating fails |
| 30 | Temperature diversity | 78.7% best single, 70.2% combined | ❌ Mixing temperatures hurts |
| 31 | Multi-dataset training | 70.8% combined vs 75.5% single | ❌ Mixing domains hurts |
| 32 | **Weight profiling (dense + MoE)** | Qwen3.5-35B expert overlap 0.008 | **✅ MoE experts genuinely specialized** |
| 34 | MoE routing entropy | Router hooks didn't capture | ⚠️ Need model-specific hook parsing |
| 35 | Activation normalization | Z-score/L2/PCA all hurt | ❌ Normalization destroys signal |
| 36 | **Logit lens divergence** | **50.6% = chance** | **❌ Dynamics identical for correct/wrong** |
| 37 | EBT in sentence embedding space | 57.5%, loss never decreased | ❌ Sentence encoders embed topic, not truth |
| 38 | NLI-based EBM | 70.8% test, 50% practical | ⚠️ NLI detects consistency, not facts |
| 39 | **thrml Ising SAT solver** | **Beats random at 50+ vars** | **✅ First Extropic-compatible experiment** |
| 40 | thrml graph coloring | Perfect on 3/6 problems | ✅ Constraint satisfaction via sampling |
| 41 | **LLM → Ising verify → repair** | **2/6 problems repaired 0%→100%** | **✅ "LLM proposes, Ising repairs" works** |
| 53 | **Runtime constraint instrumentation** | Dynamic AST rewriting complements static Exp 48 | **✅ Static+dynamic complementary** |
| 56 | **Live LLM → constraint → Ising** | End-to-end Qwen3.5-0.8B + constraint pipeline (4 domains) | **✅ Live LLM pipeline works** |
| 57 | **Live LLM verify-repair loop** | 9/15 initial, repair architecture works, constraint coverage is bottleneck (1/6 triggered) | **✅ Loop works, need wider constraint extractors** |
| 59 | **Constraint-aware prompting** | Preventive constraint injection into prompts; 3 modes (baseline/constraint-aware/combined) on 15 questions | **Results pending analysis** |
| 60 | **Scale CD training to 100+ vars** | Extends Exp 50 to 50/100/200 vars (40K params); bootstraps from hand-coded Ising + annealing; CD vs hand-coded vs random | **Results pending analysis** |
| 61 | **Sparse Ising at 500+ vars** | Clause-graph sparsity mask on CD gradients; ~20x parameter reduction vs dense; 200/500/1000 vars; dense vs sparse vs hand-coded | **Results pending analysis** |
| 54 | **Ising-guided fuzzing** | Energy landscape generates adversarial test inputs for differential testing; 8 LLM bug types (REQ-VERIFY-001/002/003) | **Results pending analysis** |
| 55 | **Trace-learned constraints** | Discriminative Ising trained on correct/buggy execution traces (200+ dim binary features); catches semantic bugs invisible to static+dynamic analysis (REQ-VERIFY-001/002/003) | **Results pending analysis** |
| 58 | **Multi-domain live benchmark (5 domains)** | 500 questions (100/domain) across arithmetic, code, logic, factual, scheduling; 3 modes (baseline/verify-only/verify-repair); full pipeline benchmark (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-005) | **Results pending analysis** |
| 64 | **Continuous Ising relaxation** | Binary→continuous [0,1] relaxation with JAX grad descent; sigmoid annealing / penalty / straight-through rounding vs discrete Gibbs + random | **Results pending analysis** |
| 69 | **Multi-model constraint transfer (Qwen3.5+Gemma4)** | Same 20 Exp 56 questions + Exp 57 verify-repair loop on Qwen3.5-0.8B and Gemma4-E4B-it; tests model-agnostic constraint pipeline transfer (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-003) | **Results pending analysis** |
| 71 | **Extropic TSU sampler abstraction** | SamplerBackend protocol: CpuBackend (ParallelIsingSampler) + TsuBackend (stub); `get_backend()` factory, `CARNOT_BACKEND` env var (REQ-SAMPLE-003) | **✅ Abstraction layer ready** |
| 62 | **Domain-specific constraint learning (10K)** | Discriminative Ising on 10K triples across arithmetic/logic/code; per-domain + combined models; 200+ binary features; AUROC on held-out test | **Results pending analysis** |
| 73 | **Constraint coverage metric** | 5-type claim taxonomy (arithmetic, logical, factual, structural, semantic); coverage = extracted/total per domain; coverage-accuracy correlation + repair threshold (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-005) | **Results pending analysis** |
| 67 | **GSM8K subset verification** | 200 GSM8K test questions, 3 modes (baseline/verify/verify-repair), first external benchmark of Ising-guided repair (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-006) | **Results pending analysis** |
| 68 | **HumanEval subset verification + fuzzing** | 50 HumanEval-style problems through full pipeline (extract→instrument→test→fuzz→repair); pass@1 + pass@1+repair metrics; bug detection breakdown (test/instrumentation/fuzzing) (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-006) | **Results pending analysis** |
| 70 | **Rust constraint extraction + verification** | `carnot-constraints` crate: BoundConstraint, EqualityConstraint, IsingConstraint + VerificationCertificate (REQ-VERIFY-001–005) | **✅ New Rust crate** |
| 65 | **Embedding-space constraint verification** | Joint Gibbs EBM on [semantic embedding; constraint vector] (384+N dim); NCE training; AUROC: joint vs embedding-only vs constraint-only; gradient repair with NN decoding (REQ-EBT-001, REQ-VERIFY-001) | **Results pending analysis** |
| 66 | **End-to-end differentiable constraint reasoning** | Fully differentiable text→embedding→constraints→continuous Ising→MLP→score; joint 1.0 test AUROC vs 0.54 Ising-only and 0.98 embedding-only; stable gradients; 5 domains (REQ-VERIFY-001, REQ-EBT-001) | **✅ Joint model outperforms components** |
| 72 | **Autoresearch self-verification via Ising** | Fourth gate: claim extraction + ComposedEnergy + Ising sampling on autoresearch hypotheses (20 mock, 10 correct/10 bogus) | **Results pending analysis** |
| 63 | **Hierarchical Ising (1000+ vars)** | Block-structured coupling (dense intra-block + sparse inter-block); two-level Gibbs + annealing; hierarchical vs flat-sparse vs flat-dense vs random at 200/500/1000 vars; ~10x param reduction | **Results pending analysis** |
| 74 | **Unified ConstraintExtractor API** | Pluggable Protocol-based extractors (arithmetic, code, logic, NL) + AutoExtractor auto-detection; consolidates Exp 47/48/49 into `carnot.pipeline.extract` (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-002) | **✅ New pipeline module** |
| 75 | **VerifyRepairPipeline class** | User-facing API consolidating Exp 56/57 into `carnot.pipeline.verify_repair`; verify-only + verify-and-repair modes; VerificationResult, RepairResult, VerifyRepairPipeline (REQ-VERIFY-001/002/003, SCENARIO-VERIFY-004) | **✅ New pipeline module** |
| 82 | **Pipeline error handling and edge cases** | Structured error hierarchy (CarnotError + 5 subclasses), wall-clock timeout, graceful degradation for all pipeline stages (REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004) | **✅ Error handling hardened** |
| 76 | **Production MCP server** | Hardened `carnot.mcp` package: 6 tools (verify_code, verify_with_properties, verify_llm_output, verify_and_repair, list_domains, health_check); 30s timeout, 10K char limit, structured errors; runnable as `python -m carnot.mcp` (REQ-CODE-001, REQ-CODE-006, REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004) | **✅ Production-grade MCP** |
| 78 | **PyPI-ready package** | setuptools build backend, optional Rust bindings (`RUST_AVAILABLE`), single-source version, extras (`mcp`, `rust`, `all`) | **✅ Pure-Python installable** |
| 79 | **Integration examples** | 5 production-ready examples: API verification, code review, batch verify, custom extractor, MCP integration | **✅ Examples shipped** |
| 80 | **Getting started documentation** | 3 new docs (getting-started, concepts, API reference) + index navigation | **✅ Docs shipped** |
| 83 | **Pipeline performance benchmarks** | All domains sub-ms p99, 36,887 calls/s throughput, zero memory growth | **✅ Benchmarks baselined** |
| 84 | **Carnot verifies Carnot (dogfood)** | Self-verification of pipeline against own source code | **✅ Dogfooding script** |
| 85 | **Prepare beta release** | RELEASE_NOTES.md + prepare_release.py + README quick start | **✅ Beta release ready** |
| 86 | **Learned energy composition weights** | Uniform 0.927 → learned 0.938 AUROC (+1.1%), not significant; arithmetic weight dominant (1.19) | **⚠️ Marginal improvement, not significant** |
| 87 | **Gradient-based repair in continuous space** | 40% success vs 28% discrete; arithmetic/scheduling 100%, factual/code/logic 0%; energy 1.72→1.02 | **⚠️ Works for structured domains, not semantic** |
| 88 | **Failure-driven constraint mining** | 93% false negative rate; implicit_logic (74), comparison (40), arithmetic_chain (23) top gaps; 6 suggested patterns, est. 75% coverage improvement | **✅ Actionable gap analysis** |
| 89 | **Self-bootstrapped constraint training** | 0.788 combined AUROC; arithmetic/logic 1.0, code 0.91, factual 0.55, scheduling 0.52; 96.7% pipeline concordance | **✅ Self-supervised Ising from pipeline outputs** |
| 91 | **GSM8K live benchmark (Qwen3.5 + Gemma4)** | Qwen3.5: 65→80% (+15%), Gemma4: 74.5→88.5% (+14%); 100% precision, 0 false positives | **✅ Cross-model GSM8K benchmark** |
| 90 | **Autoresearch constraint improvement loop** | 20 iterations, 17/20 accepted (85%); regex+logic+AST+Ising hypotheses; AUROC 0.532 unchanged — coverage up, discrimination needs richer signal | **⚠️ Coverage improves, AUROC plateau** |
| 93 | **Multi-model systematic comparison** | 250 questions × 2 models × 3 modes = 1500 evals; +10.2% avg improvement (p<0.001); scheduling +30%, code +14%, arithmetic +7% | **✅ Definitive "does Carnot help?" benchmark** |
| 94 | **Rust VerifyRepairPipeline** | Rust port of verify() path in `carnot-constraints`; VerifyPipeline + AutoExtractor + PipelineResult; 1457 lines + 318-line test suite; 10x-faster verification for PyO3 hot loop (NFR-01) | **✅ Rust verification pipeline** |
| 101 | **Agent workflow verification E2E** | 60% detection, 67% more than final-only, math 80%, code 100% | **⚠️ Agentic chain helps, but research domain undetected** |
| 102 | **Constraint check latency microbenchmark** | Full pipeline profiling: JIT forward 0.008ms (per-token viable), extraction 0.04–2.6ms linear scaling, MiniLM bottleneck 7.6ms; JAX JIT 55x faster than Python verify | **✅ Guided decoding confirmed viable** |
| 108 | **KAN Energy Function Implementation** | KAN (Kolmogorov-Arnold Networks) energy tier with B-spline edge activations; BSpline + KANEnergyFunction + KANModel; 26 tests passed, Rust scaffold created; from_ising() warm-start from trained Ising | **✅ New energy tier between Ising and Gibbs** |
| 119 | **Adversarial GSM8K variant generator (Apple 2410.05229)** | Reproduces Apple GSM-Symbolic methodology: 4 variants × 200 questions = 800 items; number swap (GSM-Symbolic), irrelevant injection (GSM-NoOp), combined; spot-check validation re-runs arithmetic to confirm correct answers; enables pipeline robustness evaluation against 65%-drop attack surface | **✅ Adversarial dataset for verify-repair robustness testing** |
| 120 | **LLM baseline on adversarial GSM8K** | Measures accuracy on Exp 119 adversarial variants WITHOUT EBM repair (pre-repair baseline); Qwen3.5-0.8B: control 77%, number-swapped 46% (−31pp), irrelevant-injected 55% (−22pp), combined 38% (−39pp); Gemma4-E4B-it: control 70%, number-swapped 53% (−17pp), irrelevant-injected 67% (−3pp), combined 44% (−26pp); bootstrap 95% CIs; confirms Apple's ~65% drop attack surface; Exp 121 will apply Carnot repair | **✅ Pre-repair baseline established; Exp 121 recovery pending** |
| 122 | **Adversarial robustness deep analysis** | Full per-item error analysis of Exp 121 results; 5-type error taxonomy; Carnot detection by type: arithmetic 100% detected/98.7% repaired, all other types 0%; 66.9% of adversarial errors are structurally uncatchable by arithmetic constraint verification; n_violations AUC=0.677 (number_swapped best: 0.762), ising_energy AUC=0.5 (continuous energy adds no ROC power); triage at threshold=1: 100% precision, 35.4% recall | **✅ Structural limits of arithmetic verification quantified; keyword_triggered and logic errors need new extractor types** |
| 141 | **Memory-augmented constraint generation** | `ConstraintGenerator` class wires Tier 2 `ConstraintMemory` into constraint addition; `ConstraintGenerator.from_memory(memory).generate(text, domain)` reads mature patterns (freq>=3) and applies extractors: `CarryChainConstraint` (arithmetic_carry, multi-carry additions like 99+1), `BoundConstraint` (comparison_boundary, numeric inequality), `NegationConstraint` (negation_scope); `AutoExtractor.extract(text, domain=None, memory=None)` extended with backward-compatible memory param; benchmark 200 GSM8K: static 0.85 → memory-augmented 0.96 (+0.11, hypothesis MET); comparison_boundary recall 0%→100%; 62 tests at 100% coverage; results at `results/experiment_141_results.json` | **✅ Memory-augmented constraint generation enables dynamic pattern discovery** |
| 144 | **JEPA Violation Predictor** | EBM for early-exit verification; JEPAViolationPredictor MLP 256→64→32→3, trained on Exp 143 JEPA pairs; per-domain violation probabilities (arithmetic/code/logic); arithmetic AUROC=0.7126 (>0.65 target); macro AUROC=0.5709 (diluted by code/logic zeros); 36 tests at 100% module coverage; model at `results/jepa_predictor.safetensors` (73.1 KB) | **✅ JEPA predictor trained; enables Tier 3 early-exit verification** |
| 145 | **JEPA Fast-Path Gate Integration** | `VerifyRepairPipeline.verify()` extended with `jepa_predictor=, jepa_threshold=` parameters; `VerificationResult` extended with `mode="FULL"/"FAST_PATH"` and `skipped=bool`; 500-question benchmark (200 arith/200 code/100 logic); threshold=0.3: 38% fast-path, 11.6% degradation; threshold=0.5: 95.4% fast-path, 19.8% degradation; targets NOT met (need <2% degradation); root cause: predictor trained on arithmetic-only Exp 143 data (code/logic AUROC=0.5); 8 new tests, 100% coverage maintained; results at `results/experiment_145_results.json` | **⚠️ Architecture works; predictor quality insufficient — need multi-domain training pairs for Exp 146** |
| 151 | **Constraint Propagation Model Export** | `python/carnot/inference/constraint_models.py` 417 lines: `IsingConstraintModel`, `ConstraintPropagationModel` factory with energy/score/batch APIs, save/load via safetensors; `scripts/export_constraint_models.py` trains domain Ising models (Exp 89 hyperparams, 500 pairs/domain); three models exported: arithmetic (AUROC=0.997, accuracy=99.0%), logic (AUROC=1.000, accuracy=100.0%), code (AUROC=0.867, accuracy=88.0%); 52 tests at 100% constraint_models.py coverage; `exports/constraint-propagation-models/README.md` with quick-start; REQ-VERIFY-002, REQ-VERIFY-003, FR-11 | **✅ Published to HuggingFace (Exp 164): Carnot-EBM/constraint-propagation-{arithmetic,logic,code}; all 3 verified** |
| 164 | **HuggingFace Publishing** | `scripts/experiment_164_hf_publish.py` — uploads guided-decoding-adapter (Exp 137), 3 constraint-propagation models (Exp 151), JEPA predictor v2 (Exp 155, macro AUROC 0.659); updates 16 per-token EBM READMEs with `pip install carnot` note; verifies all uploads; dry-run fallback to `scripts/hf_upload_commands.sh` if unauthenticated; `results/experiment_164_results.json` (5 uploads OK, 16 READMEs updated); NFR-03, REQ-VERIFY-001-003 | **✅ 5/5 artifacts published, 16/16 READMEs updated, all verified** |
| 153 | **KAN Adaptive Mesh Refinement** | Adaptive knot insertion/removal based on edge curvature (finite-difference second derivatives); 200-question arithmetic+logic benchmark; AUROC 0.875→0.875 (Δ0%, ✓target ≥-0.01), params 2310→2281 (-1.3%, ✓target ±20%); 36 knots added/65 removed; high-curvature edges on `domain_specific × numeric` cross-interactions (complex nonlinear), low-curvature on within-group linear interactions (REQ-CORE-001, REQ-TIER-001) | **✅ Mesh refinement maintains accuracy with -1.3% params** |

## 14 Principles Learned

### What works
1. Model's own logprobs are the best energy for rejection sampling (+10%)
2. Different energy signals dominate in different domains (logprobs for QA, tests for code)
3. Multi-layer concatenation improves test-set detection by ~6%

### What doesn't work for hallucination detection
4. **Activation EBMs detect confidence, not correctness** (50% practical)
5. Instruction tuning compresses hallucination signal (86.8% base → 75.0% IT)
6. Chain-of-thought compresses it further (75.5% → 61.3%)
7. Statistical difference ≠ causal influence (steering: 0% effect)
8. Adversarial questions defeat post-hoc detection
9. Hallucination representations are model-specific (~50% cross-model transfer)
10. EBM detection is domain-specific (mixing hurts)
11. Normalization doesn't enable transfer
12. Upstream question-level detection is weak (62.6%)
13. Logit lens: dynamics identical for correct/wrong (50.6%)
14. Sentence/NLI encoders embed topic/consistency, not factual truth

### The definitive finding
**You cannot detect factual hallucination without access to factual knowledge.** No internal signal — activations, logit lens, NLI, confidence — can distinguish "Neil Armstrong walked on Mars" from "Neil Armstrong walked on the Moon."

### What DOES work: structural constraint verification
- SAT → Ising → thrml sampling beats random at scale (exp 39)
- Graph coloring → Ising → thrml finds perfect solutions (exp 40)
- LLM proposes, Ising verifies and repairs — 2/6 hallucinations caught and fixed (exp 41)
- This architecture maps directly to Extropic TSU hardware

## What's Next

### High Priority
- **Exp 203 (COMPLETED — 2026-04-12)**: Live extraction autopsy on a seeded 20-question Gemma4-E4B-it GSM8K sample (`results/experiment_203_results.json`). Accuracy 17/20 (85%). ArithmeticExtractor + VerifyRepairPipeline caught **0/3 wrong answers**, while regex emitted **3 violations on correct answers only** (false positives). Wrong-answer root causes: missing intermediate step (dataset_idx 923), semantic modeling error (814), reading comprehension error (943). This is the clearest live evidence yet that regex arithmetic extraction is both too narrow and misaligned with instruction-tuned reasoning traces. Next actions: Exp 204 (Z3 extractor) and Exp 205 (LLM-as-extractor).
- **Exp 146 (COMPLETED — 2026-04-11)**: AMD XDNA NPU Hardware Integration — detected hardware present, exported JEPA predictor to ONNX opset 17, validated CPU baseline <1ms (p50=0.005ms, p99=0.009ms); identified software blocker (onnxruntime-vitisai not in PyPI, requires conda install -c amd); `NpuJEPAPredictor` stub ready for when AMD Ryzen AI software stack available; research-program.md Tier 3 hardware target validated.
- **Exp 147 (COMPLETED — 2026-04-11)**: Apple GSM8K Adversarial Benchmark — credibility validation experiment measuring Carnot verifier robustness on benign/adversarial GSM8K question pairs; validates robustness against distribution-shifted variants; results at `results/experiment_147_results.json`.
- **Exp 159 (COMPLETED — 2026-04-11)**: Full 5-domain benchmark with factual extractor + memory generation — comprehensive evaluation across 5 domains with memory-augmented constraint generation; validates hallucination detection pipeline across diverse domains.
- **Exp 161 (COMPLETED — 2026-04-11)**: Full GSM8K (1,319 questions) with live inference + 95% CIs — scales Exp 91 to full GSM8K test split; bootstrap confidence intervals + paired delta CIs; Qwen3.5-0.8B: 70.6%→84.4% (+13.8pp), Gemma4-E4B-it: 77.1%→87.8% (+10.7pp); real dataset via HuggingFace, simulation fallback; goal #6 PARTIAL (real dataset confirmed, eGPU not yet connected).
- **Exp 162 (COMPLETED — 2026-04-11)**: Apple Adversarial GSM8K with N=200/variant — definitive Goal #5 test extending Exp 147 to N=200/variant (1600 questions) with 10,000 permutation resamplings; two-proportion z-test p=0.017 SIGNIFICANT (adversarial 15.2% vs control 11.0% improvement rates); permutation test p=0.429 not significant (underpowered); adversarial/standard ratio 1.41× pooled (Qwen 1.65×, Gemma 1.17×); goal #5 PARTIAL (z-test significant but permutation test needed for definitive conclusion; live eGPU would give powered result).
- **Exp 163 (COMPLETED — 2026-04-11)**: Full HumanEval Benchmark (164 official problems) with live code generation + repair — comprehensive code verification on official HumanEval benchmark; live Qwen3.5-0.8B with subprocess code execution (5s timeout), verify-repair pipeline (up to 3 iterations); 95% bootstrap CIs (N=10,000 samples); results: baseline 68.9% [61.6%, 75.6%], repair 100.0%; Δ+31.1% [+24.4%, +38.4%]; 51/164 failures all repaired in avg 1.24 iters; publishable with live model inference.
- **Exp 167 (COMPLETED — 2026-04-11)**: JEPA Violation Predictor v3 — domain-specific symbolic embedding heads; retrained with 1500 combined pairs (800 arithmetic + 200 code + 500 symbolic-feature logic); improvements: stratified split, per-domain class weights, logic loss ×2.0, AdamW with weight decay; results: logic AUROC +0.467 (0.479→0.946), macro AUROC +0.273 (0.659→0.932); both targets MET; validates symbolic feature effectiveness on logic domain (REQ-JEPA-001, SCENARIO-JEPA-003).
- **Exp 168 (COMPLETED — 2026-04-11)**: JEPA fast-path v3 validation — fast-path gate benchmarking with symbolic embedding heads; threshold=0.5 achieves 40% fast-path rate (MET) with 8.4% accuracy degradation (target <2% not met); domain-specific symbolic features for logic + RandomProjection for others; 3 thresholds tested (0.3, 0.5, 0.7); results at `results/experiment_168_results.json`; REQ-JEPA-001.
- **Exp 204 (NEXT)**: Z3 arithmetic extractor on the three wrong live Gemma cases from Exp 203. Target: catch all three wrong answers with zero false positives on the sampled correct cases.
- **Exp 205 (NEXT)**: LLM-as-extractor on the same Exp 203 cases as a flexible fallback for natural-language arithmetic traces the regex cannot normalize.
- **Scale thrml constraint verification**: larger SAT/coloring problems, more constraint types
- **LLM constraint extraction**: parse natural language into Ising-encodable constraints
- **Extropic hardware testing**: when TSU is available, run thrml code natively

### Milestone 2026.04.2: Toward Kona
- Milestone 2026.04.2: Toward Kona — live LLM + Ising end-to-end
- ~~Exp 53: Runtime constraint instrumentation~~: ✅ DONE (2026-04-09)
- ~~Exp 56: Live LLM → constraint → Ising verification~~: ✅ DONE (2026-04-09)
- ~~Exp 57: Live LLM verify-repair loop with Qwen3.5~~: ✅ DONE (2026-04-09)
- ~~Exp 60-61: Scale learned Ising to 500+ vars~~: ✅ DONE (2026-04-09)
- ~~Exp 64: Continuous relaxation (bridge to Kona latent space)~~: ✅ DONE (2026-04-09) — 3 rounding strategies (sigmoid annealing, penalty, straight-through) vs discrete Gibbs + random baseline

### Completed
- ~~Ship MCP server + CLI~~: ✅ DONE
- ~~Scale per-token EBM~~: ✅ DONE (16 models on HuggingFace)
- ~~Publish v12 artifacts~~: ✅ DONE — `constraint-verifier-v2` (KAN EBM + guided decoding adapter) published at `huggingface.co/Carnot-EBM/constraint-verifier-v2`; safetensors weights + config + model cards (Exp 118)
- ~~Weight profiling~~: ✅ DONE (dense + MoE analyzed)
- ~~Logit lens~~: ✅ DONE (negative result — 50.6%)
- ~~NLI-based EBM~~: ✅ DONE (70.8% test, 50% practical)
- ~~thrml integration~~: ✅ DONE (SAT + coloring + LLM verify/repair)

### Research Directions (Roadmap v6 — Constraint-Based Reasoning)

**See `openspec/change-proposals/research-roadmap-v6.md` for full details.**

**Key paradigm shift:** Structural constraint verification via Ising/thrml, not activation-based detection. LLM handles language, Ising handles reasoning, Extropic TSU does sampling. Roadmaps v2-v5 are superseded (activation-based approaches proven insufficient by experiments 36-38).

#### Completed (Experiments 1-31)
- ~~Per-token EBM rejection~~: Exp 23, -3% to -6%
- ~~Cross-model transfer~~: Exp 26, 49.8% = chance
- ~~Temperature diversity~~: Exp 30, hurts
- ~~Naive domain mixing~~: Exp 31, 70.8% < 75.5%
- ✅ Multi-layer concat: Exp 28, +5.8%
- ✅ 3-layer concat sweet spot: Exp 29

#### Phase 1: Weight Anatomy (NOW — no labels for training)
- **Exp 32: Weight structure profiling** — pure weight analysis, zero inference needed
- **Exp 33: Channel magnitude introspection** — Nemotron-inspired, expert FC1/FC2 patterns
- **Exp 34: MoE routing entropy as energy** — self-supervised, unlabeled forward pass only
- **Exp 35: Activation normalization** — domain-invariant features via per-sequence normalization

#### Phase 2: Self-Supervised Energy Composition (NEXT — minimal labels)
- **Exp 36: Composite self-supervised energy** — combine all Phase 1 features, 100-500 labels for calibration
- **Exp 37: MTP confidence** — multi-token prediction as temporal signal (Nemotron-inspired)
- **Exp 38: Cross-architecture consensus** — dense + MoE + Mamba agreement, fully self-supervised
- **Exp 39: Logit lens / unembedding geometry** — per-layer prediction trajectory

#### Phase 3: Consensus Energy Landscape (THEN — no labels)
- **Exp 40: Weight-space model similarity map** — pure weight analysis, zero inference
- **Exp 41: Energy-guided decoding** — self-supervised energy for generation guidance
- **Exp 42: KL distillation energy** — composable multi-model energy terms

#### Phase 4: Standalone EBM (long-term)
- 4a: Universal activation encoder (self-supervised contrastive)
- 4b: Consensus energy landscape
- 4c: LLM as language interface
- 4d: Hardware compilation (Extropic TSU)

#### Model Acquisition
- ✅ **Mixtral-8x7B-v0.1** (Priority 1): downloading now (~93GB BF16 base). Unlocks Exp 32 (MoE weight profiling), 33 (channel magnitude), 34 (routing entropy), 38 (consensus)
- **Mamba-2.8B or Jamba** (Priority 2): architectural diversity for consensus (Exp 38)
- Nemotron 3 Super NVFP4: MTP heads + richest routing structure (Exp 37)

### Documentation
- **UI Aesthetic**: Premium glassmorphism and animations applied to `docs/index.html`
- **Technical report**: published at `docs/technical-report.md`
- **Experiment log**: 24 experiments at `ops/experiment-log.md`
- **Research roadmaps**: v1-v3 at `openspec/change-proposals/`

## Known Constraints
- Python 3.14 requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`
- ROCm on integrated GPU is 3.3x (would be 10-100x on discrete AMD GPU)
- Ackley Python/JAX uses epsilon=1e-10 in sqrt (documented in spec)
- gVisor installed for production autoresearch sandbox


## Orchestration Run (2026-04-09 00:20 UTC)

**Epic:** Epic: UI-001 - Modernize Documentation Aesthetic
**Run ID:** b6ec974e-c949-4d99-ad11-b191881de22d
**Stories completed:** 2/3
**Stories failed:** 0/3
**Total cost:** $0.00
**Completed:** DOCUI-001, DOCUI-002
- **Exp 176 (COMPLETED — 2026-04-11)**: Multi-turn factual verification with global consistency checking — combines ConstraintStateMachine + FactualExtractor (Wikidata KB) with GlobalConsistencyChecker (Exp 172); 20 synthetic chains (10 consistent + 10 inconsistent); local-only Mode B 60% detection (6/10) → local+global Mode C 100% detection (10/10 inconsistent, 0 FP on consistent); GlobalConsistencyChecker adds 4 detections for numeric/arithmetic cross-step contradictions; demonstrates cascade of verification strategies for multi-turn reasoning; results at `results/experiment_176_results.json`; REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-005.
- **Exp 178 (COMPLETED — 2026-04-11)**: Definitive adversarial GSM8K benchmark — Goal #5 ACHIEVED with statistical power (N≥400/variant). Paired sign permutation test + two-proportion z-test (10k resamples). number_swapped variant: Qwen3.5-0.8B baseline 43.3%→71.5% (+28.2pp), Gemma4-E4B-it 52.3%→76.3% (+24.0pp); both p=0.0 (highly significant). Fixes Exp 162's underpowered aggregate permutation test design. Results at `results/experiment_178_results.json`; REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006.
| Exp 181: GSM8K full 1319 with LIVE GPU inference | ✅ In Progress (Qwen3.5-0.8B baseline on RTX 3090 dual-GPU; runs full 1319-question GSM8K test set with LIVE GPU inference; checkpoint format for long-running inference; publishable baseline for GPU-accelerated verification pipeline; results accumulating at `results/experiment_181_ckpt_*.json`; REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006) | — |
