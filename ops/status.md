# Carnot — Operational Status

**Last Updated:** 2026-04-07 — 29 EXPERIMENTS, 13 PRINCIPLES, 12+ MODELS ON HUGGINGFACE

## What's Working

### Core Framework (REQ-CORE-001–006)
- EnergyFunction trait (Rust) and protocol (Python/JAX)
- Three model tiers: Ising (both), Gibbs (both), Boltzmann (both)
- Samplers: Langevin + HMC in both languages, with gradient clipping (REQ-SAMPLE-004)
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
- Research conductor (autonomous Claude Code agent loop)

### Autoresearch Results
- **10-iteration run (Sonnet)**: DoubleWell 0.9483 → 0.1604 (83% energy reduction), 3 accepted hypotheses (HMC, annealing)
- **50-iteration run (Sonnet)**: DoubleWell 0.0001, Rosenbrock 0.0092 (both near optimal). Circuit breaker at iteration 18.

### Quality Infrastructure
- 1049 Python tests + 104 Rust tests, 100% code coverage, 100% spec coverage
- Pre-commit hooks: rustfmt, clippy, ruff, mypy, pytest, spec coverage
- Docker compose: Claude API bridge + WebGPU gateway (`make up`)

## Experiment Results (24 experiments)

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

## 13 Principles Learned

1. Simpler is better in small-data regimes
2. Token-level features > sequence-level (mean-pooling kills signal)
3. Model's own logprobs are the best energy
4. Overfitting is the main enemy when examples < dimensions
5. Extract features from generated tokens, not prompts
6. Different energy signals dominate in different domains
7. Statistical difference ≠ causal influence
8. Instruction tuning compresses the hallucination signal (84.5% base → 67.2% tuned)
9. Adversarial questions defeat post-hoc detection (logprob/EBM/composite all fail on TruthfulQA)
10. Chain-of-thought compresses hallucination signal (75.5% without thinking vs 61.3% with)
11. Hallucination representations are model-specific (~50% cross-model transfer = chance)
12. Multi-layer concatenation improves detection by ~6% (layers 4+12+24 = 81.3% vs last-only 75.5%)
13. Upstream question-level detection is weak (62.6% mean) but model-dependent

## What's Next

### High Priority
- **Ship MCP server + CLI**: ✅ DONE — MCP server (3 tools: verify_code, verify_with_properties, score_candidates), CLI (`carnot verify`), tested E2E
- **Scale per-token EBM**: ✅ DONE — 52,296 tokens from QA + TruthfulQA (Qwen3.5-0.8B), 67.2% test accuracy
- **E2E-001: Rust training pipeline test**: Only remaining E2E test gap

### Medium Priority
- **GitHub public mirror**: open-source visibility
- **GPU-accelerated experiments**: ROCm 7.2 native gfx1150 ready (3.3x speedup), update experiment scripts to use `.cuda()`
- **Larger local model**: test with Qwen3-4B or 8B (67GB unified memory available)

### Research Directions (Roadmap v5 — Weight-First EBM)

**See `openspec/change-proposals/research-roadmap-v5.md` for full details.**

**Key paradigm shift:** Derive hallucination signal from frozen weight structure + unlabeled forward passes. Labeled data is for validation, not training. 10 of 11 new experiments need zero training labels.

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
