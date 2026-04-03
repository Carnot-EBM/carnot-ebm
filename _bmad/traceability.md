# Carnot — Traceability Matrix

**Last Updated:** 2026-04-03

## Functional Requirements → Implementation Status

| FR ID | Description | Spec | Tests | Impl | Status |
|-------|------------|------|-------|------|--------|
| FR-01 | Core EBM Framework | `openspec/capabilities/core-ebm/spec.md` | 10 Rust, 4 Python | Rust + Python | Implemented |
| FR-02 | Boltzmann Tier | `openspec/capabilities/model-tiers/spec.md` | 6 Rust | Rust | Partial |
| FR-03 | Gibbs Tier | `openspec/capabilities/model-tiers/spec.md` | 7 Rust | Rust | Partial |
| FR-04 | Ising Tier | `openspec/capabilities/model-tiers/spec.md` | 10 Rust, 7 Python | Rust + Python | Implemented |
| FR-05 | Dual-Language Impl | `openspec/capabilities/core-ebm/spec.md` | - | Rust + Python | Partial |
| FR-06 | Training Pipeline | `openspec/capabilities/training-inference/spec.md` | 5 Rust | Rust (CD-k) | Partial |
| FR-07 | Inference Pipeline | `openspec/capabilities/training-inference/spec.md` | 6 Rust, 4 Python | Rust + Python | Partial |
| FR-08 | Interoperability | `openspec/capabilities/core-ebm/spec.md` | - | PyO3 skeleton | Not Started |
| FR-09 | Test Coverage | N/A (process) | 44 Rust tests | pre-commit | In Progress |
| FR-10 | Spec-Driven Dev | N/A (process) | spec_coverage.py | pre-commit | In Progress |
| FR-11 | Autonomous Self-Learning | TBD | - | - | Spec'd |
| FR-12 | Verifiable Reasoning | TBD | - | - | Spec'd |

## Non-Functional Requirements

| NFR ID | Description | Verified By | Status |
|--------|------------|-------------|--------|
| NFR-01 | Performance (10x Rust vs Python) | Benchmark suite | Not Started |
| NFR-02 | Safety (no unsafe in public API) | clippy + manual review | Not Started |
| NFR-03 | Documentation | cargo doc + sphinx | Not Started |
| NFR-04 | CI/CD pre-commit hooks | pre-commit config | Implemented |
