# Carnot — Product Requirements Document

**Last Updated:** 2026-04-03

## Vision

Carnot is an open-source Energy Based Model (EBM) framework providing production-grade implementations across three capability tiers. It bridges the gap between research-oriented Python/JAX code and deployment-ready Rust implementations, enabling EBM practitioners to move from experimentation to production with a single unified framework.

## Problem Statement

Energy Based Models are a foundational class of generative models, yet existing implementations are fragmented across research codebases with no clear path to production deployment. Researchers need fast iteration in Python/JAX; production systems need the performance and safety guarantees of Rust. No framework bridges both worlds with spec-driven quality guarantees.

## Model Tiers

| Tier | Name | Scale | Target Use Case |
|------|------|-------|-----------------|
| Large | **Boltzmann** | Full-scale EBM | Research frontiers, large-scale generation |
| Medium | **Gibbs** | Mid-scale EBM | Applied ML, fine-tuning, domain adaptation |
| Small | **Ising** | Lightweight EBM | Edge deployment, embedded systems, teaching |

## Functional Requirements

### FR-01: Core EBM Framework
The system shall provide a core energy function abstraction that all tiers implement.

### FR-02: Boltzmann Tier
The system shall provide a full-scale EBM implementation suitable for research and large-scale generation.

### FR-03: Gibbs Tier
The system shall provide a mid-scale EBM implementation suitable for applied ML and domain adaptation.

### FR-04: Ising Tier
The system shall provide a lightweight EBM implementation suitable for edge deployment and teaching.

### FR-05: Dual-Language Implementation
The system shall provide implementations in both Rust (for production performance) and Python/JAX (for research iteration).

### FR-06: Training Pipeline
The system shall provide training pipelines including contrastive divergence, score matching, and noise-contrastive estimation.

### FR-07: Inference Pipeline
The system shall provide inference via MCMC sampling (Langevin dynamics, HMC) with configurable samplers.

### FR-08: Interoperability
The Python and Rust implementations shall be interoperable via PyO3 bindings, allowing Rust cores to be called from Python.

### FR-09: Test Coverage
All code shall have 100% test coverage. All tests shall have 100% spec coverage (every test traces to a REQ-* or SCENARIO-*).

### FR-10: Spec-Driven Development
All code shall be derived from OpenSpec specifications. No code exists without a driving spec.

## Non-Functional Requirements

### NFR-01: Performance
Rust implementations shall achieve at minimum 10x throughput improvement over equivalent Python implementations for inference workloads.

### NFR-02: Safety
All Rust code shall compile with no `unsafe` blocks in public APIs. All Python code shall pass mypy strict type checking.

### NFR-03: Documentation
All public APIs shall have documentation with examples. All capabilities shall have spec.md and design.md.

### NFR-04: CI/CD
Pre-commit hooks shall enforce linting, formatting, and test coverage on every commit.

## Success Metrics

- 100% spec coverage (every line of code traces to a requirement)
- 100% test coverage (every line of code is exercised by tests)
- All three tiers functional with training + inference
- PyO3 bindings enabling Python-to-Rust interop
- Pre-commit hooks passing on every commit
