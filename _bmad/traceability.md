# Carnot — Traceability Matrix

**Last Updated:** 2026-04-03

## Functional Requirements → Implementation Status

| FR ID | Description | Spec | Tests | Impl | Status |
|-------|------------|------|-------|------|--------|
| FR-01 | Core EBM Framework | `openspec/capabilities/core-ebm/spec.md` | - | - | Spec'd |
| FR-02 | Boltzmann Tier | `openspec/capabilities/model-tiers/spec.md` | - | - | Spec'd |
| FR-03 | Gibbs Tier | `openspec/capabilities/model-tiers/spec.md` | - | - | Spec'd |
| FR-04 | Ising Tier | `openspec/capabilities/model-tiers/spec.md` | - | - | Spec'd |
| FR-05 | Dual-Language Impl | `openspec/capabilities/core-ebm/spec.md` | - | - | Spec'd |
| FR-06 | Training Pipeline | `openspec/capabilities/training-inference/spec.md` | - | - | Spec'd |
| FR-07 | Inference Pipeline | `openspec/capabilities/training-inference/spec.md` | - | - | Spec'd |
| FR-08 | Interoperability | `openspec/capabilities/core-ebm/spec.md` | - | - | Spec'd |
| FR-09 | Test Coverage | N/A (process) | - | - | In Progress |
| FR-10 | Spec-Driven Dev | N/A (process) | - | - | In Progress |

## Non-Functional Requirements

| NFR ID | Description | Verified By | Status |
|--------|------------|-------------|--------|
| NFR-01 | Performance (10x Rust vs Python) | Benchmark suite | Not Started |
| NFR-02 | Safety (no unsafe in public API) | clippy + manual review | Not Started |
| NFR-03 | Documentation | cargo doc + sphinx | Not Started |
| NFR-04 | CI/CD pre-commit hooks | pre-commit config | Not Started |
