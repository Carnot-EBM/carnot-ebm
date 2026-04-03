# Carnot — Traceability Matrix

**Last Updated:** 2026-04-03 (reconciled with codebase)

## Functional Requirements → Implementation Status

| FR ID | Description | Spec | Tests | Impl | Status |
|-------|------------|------|-------|------|--------|
| FR-01 | Core EBM Framework | `openspec/capabilities/core-ebm/spec.md` | 10 Rust, 6 Python | Rust + Python | Implemented |
| FR-02 | Boltzmann Tier | `openspec/capabilities/model-tiers/spec.md` | 7 Rust | Rust only | Partial (Python missing) |
| FR-03 | Gibbs Tier | `openspec/capabilities/model-tiers/spec.md` | 7 Rust | Rust + partial Python | Partial |
| FR-04 | Ising Tier | `openspec/capabilities/model-tiers/spec.md` | 10 Rust, 12 Python | Rust + Python | Implemented |
| FR-05 | Dual-Language Impl | `openspec/capabilities/core-ebm/spec.md` | - | Rust + Python (Boltzmann Python missing) | Partial |
| FR-06 | Training Pipeline | `openspec/capabilities/training-inference/spec.md` | 10 Rust, 13 Python | CD-k (Rust), DSM (Rust+Python), NCE missing | Partial |
| FR-07 | Inference Pipeline | `openspec/capabilities/training-inference/spec.md` | 6 Rust, 8 Python | Langevin + HMC both languages | Implemented |
| FR-08 | Interoperability | `openspec/capabilities/core-ebm/spec.md` | - | PyO3 bindings for all 3 tiers + 2 samplers | Partial |
| FR-09 | Test Coverage | N/A (process) | 93 Rust, 125 Python = 218 | pre-commit + CI | Implemented |
| FR-10 | Spec-Driven Dev | N/A (process) | spec_coverage.py | pre-commit + CI | Implemented |
| FR-11 | Autonomous Self-Learning | `openspec/capabilities/autoresearch/spec.md` | 20 Python | Sandbox, evaluator, orchestrator, Docker+gVisor | Partial (rollback, composition missing) |
| FR-12 | Verifiable Reasoning | `openspec/capabilities/verifiable-reasoning/spec.md` | 17 Rust, 15 Python | Constraints, composition, verification, repair, Sudoku | Implemented (landscape cert missing) |

## Non-Functional Requirements

| NFR ID | Description | Verified By | Status |
|--------|------------|-------------|--------|
| NFR-01 | Performance (10x Rust vs Python) | Benchmark suite | Partial (benchmarks exist, no formal perf comparison) |
| NFR-02 | Safety (no unsafe in public API) | clippy + security-auditor agent | Verified (no unsafe in library code) |
| NFR-03 | Documentation | Comprehensive inline docs (4,475 lines added) | Implemented |
| NFR-04 | CI/CD pre-commit hooks | Gitea Actions + pre-commit config | Implemented |
