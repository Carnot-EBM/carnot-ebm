# Carnot — Operational Status

**Last Updated:** 2026-04-06 — ALL RESEARCH COMPLETE, 20 EXPERIMENTS, TECHNICAL REPORT PUBLISHED

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
- Per-token EBM (71.8% test accuracy, experiment 19)

### Activation Analysis (Phase 3)
- Activation extractor (per-layer transformer hooks)
- Hallucination direction (80% detection, 0.945 AUROC)
- Layer-targeted EBM, LayerNavigator, activation/weight steering
- Concept vectors (targeted prompting)
- Per-token activation dataset (1860 tokens)

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

## Experiment Results (20 experiments)

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

## 7 Principles Learned

1. Simpler is better in small-data regimes
2. Token-level features > sequence-level (mean-pooling kills signal)
3. Model's own logprobs are the best energy
4. Overfitting is the main enemy when examples < dimensions
5. Extract features from generated tokens, not prompts
6. Different energy signals dominate in different domains
7. Statistical difference ≠ causal influence

## What's Next

### High Priority
- **Ship MCP server + CLI**: tools built (`tools/verify-mcp/server.py`, `scripts/carnot_cli.py`), need real-world testing
- **Scale per-token EBM**: train on 1000+ QA dataset (`data/qa_dataset_1000.json` generated), test on harder tasks
- **E2E-001: Rust training pipeline test**: Only remaining E2E test gap

### Medium Priority
- **GitHub public mirror**: open-source visibility
- **GPU-accelerated experiments**: ROCm 7.2 native gfx1150 ready (3.3x speedup), update experiment scripts to use `.cuda()`
- **Larger local model**: test with Qwen3-4B or 8B (67GB unified memory available)

### Research Directions
- **Per-token EBM rejection sampling**: use the 71.8%-accurate per-token EBM for candidate selection (not yet tested)
- **Concept-specific vectors with MORE data**: current concept prompts underperformed generic direction, may need more diverse prompting
- **EBT training on real data**: minimal EBT exists (`python/carnot/models/ebt.py`), needs training on large dataset

### Documentation
- **Technical report**: published at `docs/technical-report.md`
- **Experiment log**: 20 experiments at `ops/experiment-log.md`
- **Research roadmaps**: v1-v3 at `openspec/change-proposals/`

## Known Constraints
- Python 3.14 requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`
- ROCm on integrated GPU is 3.3x (would be 10-100x on discrete AMD GPU)
- Ackley Python/JAX uses epsilon=1e-10 in sqrt (documented in spec)
- gVisor installed for production autoresearch sandbox
