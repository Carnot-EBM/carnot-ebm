# Carnot — Operational Status

**Last Updated:** 2026-04-09 — 64 EXPERIMENTS, 14 PRINCIPLES, 16 MODELS ON HUGGINGFACE, THRML/EXTROPIC INTEGRATION

## What's Working

### Core Framework (REQ-CORE-001–006)
- EnergyFunction trait (Rust) and protocol (Python/JAX)
- Three model tiers: Ising (both), Gibbs (both), Boltzmann (both)
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
- Energy landscape certification — Hessian eigenvalue analysis, basin estimation
- Convergence guarantees — absorbing invariant sets (P10)
- Deterministic reproducibility
- Domains: SAT, graph coloring, Python code, property-based testing
- Rust built-in constraint primitives: BoundConstraint, EqualityConstraint, IsingConstraint (`carnot-constraints` crate, Exp 70)
- Serializable VerificationCertificate with JSON export (`carnot-constraints`, Exp 70)
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
- NL constraint extraction: pattern-based claim verification
- LLM self-constraint pipeline: 10/10 perfect (all hallucinations caught)
- Scheduling constraints: time slot exclusion, ordering, capacity
- Learned Ising via CD: 89/100 perfect, generalizes to unseen instances (Exp 50); scaled to 50/100/200 vars with L1 regularization and bootstrapped training data (Exp 60); sparse CD with clause-graph masking at 200/500/1000 vars, ~20x parameter reduction vs dense (Exp 61); domain-specific constraint learning on 10K triples across arithmetic/logic/code with 200+ binary features (Exp 62); hierarchical block-structured Ising with dense intra-block + sparse inter-block couplings, two-level Gibbs sampler, ~10x param reduction at 1000 vars (Exp 63)
- Cross-domain transfer: structure-dependent transfer validated
- Ising-guided fuzzing: energy landscape generates adversarial test inputs for differential testing of LLM code; 8 bug types covered (Exp 54)
- Trace-learned constraints: discriminative Ising trained on correct/buggy execution traces catches semantic bugs invisible to static+dynamic analysis (Exp 55)
- Multi-domain live benchmark: 500 questions across 5 domains (arithmetic, code, logic, factual, scheduling) in 3 modes (baseline/verify/verify-repair); first comprehensive pipeline evaluation (Exp 58)
- Multi-model constraint transfer: validates constraint pipeline (arithmetic, logic, code AST, factual KB) on Qwen3.5-0.8B and Gemma4-E4B-it without retraining; tests model-agnostic verification (Exp 69)

### GPU Compute
- carnot-gpu: wgpu Vulkan backend (AMD Radeon 890M, tested)
- carnot-webgpu-gateway: distributed browser GPU compute
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
