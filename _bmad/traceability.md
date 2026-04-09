# Carnot — Traceability Matrix

**Last Updated:** 2026-04-09 (reconciled with codebase)

## Functional Requirements → Implementation Status

| FR ID | Description | Spec | Tests | Impl | Status |
|-------|------------|------|-------|------|--------|
| FR-01 | Core EBM Framework | `openspec/capabilities/core-ebm/spec.md` | 10 Rust, 6 Python | Rust + Python | Implemented |
| FR-02 | Boltzmann Tier | `openspec/capabilities/model-tiers/spec.md` | 7 Rust, Python | Rust + Python | Implemented |
| FR-03 | Gibbs Tier | `openspec/capabilities/model-tiers/spec.md` | 7 Rust, Python | Rust + Python | Implemented |
| FR-04 | Ising Tier | `openspec/capabilities/model-tiers/spec.md` | 10 Rust, 12 Python | Rust + Python | Implemented |
| FR-05 | Dual-Language Impl | `openspec/capabilities/core-ebm/spec.md` | PyO3 integration | Rust + Python + bindings | Implemented |
| FR-06 | Training Pipeline | `openspec/capabilities/training-inference/spec.md` | 30+ Python | CD-k, DSM, NCE, SNL, optimization-training, replay buffer | Implemented |
| FR-07 | Inference Pipeline | `openspec/capabilities/training-inference/spec.md` | 20+ Python | Langevin + HMC + gradient clipping | Implemented |
| FR-08 | Interoperability | `openspec/capabilities/core-ebm/spec.md` | 24 PyO3 | PyO3 bindings all tiers + samplers | Implemented |
| FR-09 | Test Coverage | N/A (process) | 104 Rust + 1093 Python | pre-commit + CI | Implemented |
| FR-10 | Spec-Driven Dev | N/A (process) | spec_coverage.py | pre-commit + CI | Implemented |
| FR-11 | Autonomous Self-Learning | `openspec/capabilities/autoresearch/spec.md` | 100+ Python | Sandbox, evaluator, orchestrator, Trace2Skill, conductor | Implemented |
| FR-12 | Verifiable Reasoning | `openspec/capabilities/verifiable-reasoning/spec.md` | 40+ Python | Constraints, repair, SAT, coloring, code, property tests, convergence | Implemented |
| FR-13 | LLM-EBM Inference | `openspec/capabilities/llm-ebm-inference/spec.md` | 170+ Python | Composite scorer, iterative refinement, logprob rejection, EBM rejection, multi-start, semantic energy, ARM-EBM bridge, diffusion, reasoning energy | Implemented |
| FR-14 | Code Verification | `openspec/capabilities/code-verification/spec.md` | 50+ Python | Type/exception/test constraints, code embeddings (bag-of-tokens + AST), learned verifier, self-improving loop | Implemented |
| FR-15 | Activation Analysis | `openspec/capabilities/llm-ebm-inference/spec.md` | 110+ Python | Activation extractor, hallucination direction, layer EBM, LayerNavigator, steering, weight steering, concept vectors, multi-layer probing | Implemented |
| FR-18 | MCP Server + CLI | `openspec/capabilities/llm-ebm-inference/spec.md` | 19 Python | MCP verify_code, verify_with_properties, score_candidates; CLI carnot verify | Implemented |
| FR-16 | GPU Compute | N/A | 4 Rust | wgpu Vulkan backend, WebGPU gateway, ROCm 7.2 native gfx1150 | Implemented |
| FR-17 | Documentation UI | `openspec/capabilities/documentation-ui/spec.md` | 1 Python | Premium aesthetic, glassmorphism, animations | Implemented |

## Non-Functional Requirements

| NFR ID | Description | Verified By | Status |
|--------|------------|-------------|--------|
| NFR-01 | Performance (10x Rust vs Python) | Benchmark suite + GPU | Implemented (wgpu + ROCm 7.2) |
| NFR-02 | Safety (no unsafe in public API) | clippy + security-auditor agent | Verified |
| NFR-03 | Documentation | Technical report + inline docs | Implemented |
| NFR-04 | CI/CD pre-commit hooks | Gitea Actions + pre-commit config | Implemented |

## Research Validation

| Experiment | Result | Paper Reference |
|-----------|--------|----------------|
| Logprob rejection (+10%) | ✅ Validated | Semantic Energy (arxiv 2508.14496) |
| Composite scoring (0% → 30%) | ✅ Novel | — |
| Per-token EBM (71.8%) | ✅ Novel | — |
| Activation steering (0%) | ❌ Negative result | Emotion Concepts (Anthropic 2025) |
| SAT gradient repair (60% → 80%) | ✅ Validated | — |
| Per-token EBM scaled (84.5%) | ✅ Novel | — |
| Instruction tuning compression (84.5% → 67.2%) | ✅ Novel (Principle 8) | — |
| EBM rejection on adversarial QA (-3%) | ❌ Negative result (Principle 9) | — |
| Multi-layer probing (U-curve) | ✅ Novel | — |
| Thinking compression (75.5% vs 61.3%) | ✅ Novel (Principle 10) | — |
| Exp 42b: Arithmetic QUBO encoding | ✅ Complete | — |
| Exp 42c: Deterministic carry chain verification | ✅ Complete | — |
| Exp 44: Scheduling constraints | ✅ Complete | — |
| Exp 45: Logical consistency verification | ✅ Complete | — |
| Exp 46b: Scale SAT to 5000 vars | ✅ Complete | — |
| Exp 47: LLM self-constraint extraction | ✅ Complete | — |
| Exp 48: Code → constraint extraction | ✅ Complete | — |
| Exp 49: NL → constraint extraction | ✅ Complete | — |
| Exp 50: Learn SAT couplings via CD | ✅ Complete | — |
| Exp 51: Learn from LLM (correct/wrong) pairs | ✅ Complete | — |
| Exp 52: Transfer Ising across domains | ✅ Complete | — |
| Exp 53: Runtime constraint instrumentation | ✅ Complete | — |
| Exp 56: Live LLM → constraint → Ising verification | ✅ Complete | — |
| Exp 57: Live LLM verify-repair loop | ✅ Complete (9/15 initial, loop works, constraint coverage bottleneck) | — |
| Exp 59: Constraint-aware prompting | ✅ Complete (preventive constraint injection, SCENARIO-VERIFY-005) | — |
| Exp 60: Scale CD training to 100+ vars | ✅ Complete (50/100/200 vars, up to 40K params, bootstrapped training) | — |
| Exp 61: Sparse Ising at 500+ vars | ✅ Complete (clause-graph sparsity mask, ~20x param reduction, 200/500/1000 vars) | — |
| Exp 54: Ising-guided fuzzing | ✅ Complete (adversarial test input generation via Ising energy, 8 LLM bug types, REQ-VERIFY-001/002/003) | — |
| Exp 55: Learn constraints from execution traces | ✅ Complete (discriminative Ising on correct/buggy execution traces, catches semantic bugs, REQ-VERIFY-001/002/003) | — |
| Exp 58: Multi-domain live benchmark (5 domains) | ✅ Complete (500 questions, 5 domains, 3 modes: baseline/verify-only/verify-repair, REQ-VERIFY-001/002/003, SCENARIO-VERIFY-005) | — |
| Exp 64: Continuous Ising relaxation | ✅ Complete (binary→continuous [0,1], sigmoid/penalty/straight-through rounding, bridges to Kona latent space) | — |
| Exp 69: Multi-model constraint transfer (Qwen3.5+Gemma4) | ✅ Complete (same constraint pipeline on Qwen3.5-0.8B + Gemma4-E4B-it, tests model-agnostic verification, REQ-VERIFY-001/002/003, SCENARIO-VERIFY-003) | — |
| Exp 71: Extropic TSU sampler abstraction | ✅ Complete (SamplerBackend protocol, CpuBackend + TsuBackend stub, get_backend() factory, CARNOT_BACKEND env var, REQ-SAMPLE-003) | — |
| Exp 73: Constraint coverage metric | ✅ Complete (5-type claim taxonomy, coverage = extracted/total per domain+type, coverage-accuracy correlation, repair threshold analysis, REQ-VERIFY-001/002/003, SCENARIO-VERIFY-005) | — |
| Parallel Ising Gibbs sampler | ✅ 183x faster than thrml (572x at 500 vars) | — |
