# Carnot — Traceability Matrix

**Last Updated:** 2026-04-06 (reconciled with codebase)

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
| FR-09 | Test Coverage | N/A (process) | 104 Rust + 1049 Python | pre-commit + CI | Implemented |
| FR-10 | Spec-Driven Dev | N/A (process) | spec_coverage.py | pre-commit + CI | Implemented |
| FR-11 | Autonomous Self-Learning | `openspec/capabilities/autoresearch/spec.md` | 100+ Python | Sandbox, evaluator, orchestrator, Trace2Skill, conductor | Implemented |
| FR-12 | Verifiable Reasoning | `openspec/capabilities/verifiable-reasoning/spec.md` | 40+ Python | Constraints, repair, SAT, coloring, code, property tests, convergence | Implemented |
| FR-13 | LLM-EBM Inference | `openspec/capabilities/llm-ebm-inference/spec.md` | 150+ Python | Composite scorer, iterative refinement, logprob rejection, multi-start, semantic energy, ARM-EBM bridge, diffusion, reasoning energy | Implemented |
| FR-14 | Code Verification | `openspec/capabilities/code-verification/spec.md` | 50+ Python | Type/exception/test constraints, code embeddings (bag-of-tokens + AST), learned verifier, self-improving loop | Implemented |
| FR-15 | Activation Analysis | `openspec/capabilities/llm-ebm-inference/spec.md` | 100+ Python | Activation extractor, hallucination direction, layer EBM, LayerNavigator, steering, weight steering, concept vectors | Implemented |
| FR-16 | GPU Compute | N/A | 4 Rust | wgpu Vulkan backend, WebGPU gateway, ROCm 7.2 native gfx1150 | Implemented |

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
