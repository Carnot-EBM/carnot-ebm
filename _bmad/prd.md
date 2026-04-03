# Carnot — Product Requirements Document

**Last Updated:** 2026-04-03

## Vision

Carnot is an open-source Energy Based Model (EBM) framework built to escape the fundamental limitations of autoregressive large language models. Where LLMs generate outputs by sequentially predicting the most probable next token — a process that produces fluent but unreliable text riddled with hallucinations — EBMs reason by minimizing energy across entire configurations simultaneously, enforcing logical consistency and physical constraints mathematically rather than statistically.

The long-term goal is **autonomous directed self-learning**: a system that can evaluate its own reasoning against a deterministic energy landscape, identify where constraints are violated, and improve itself without human supervision. EBMs provide the mathematical foundation for this because the energy function serves as an objective ground truth — low energy means the configuration satisfies all constraints, high energy means it doesn't. There is no guessing.

Carnot provides production-grade EBM implementations across three capability tiers in both Rust (for deployment performance) and Python/JAX (for research iteration), with the explicit aim of building toward self-improving AI systems grounded in verifiable reasoning rather than probabilistic token prediction.

## Problem Statement

### The Hallucination Problem

Large language models are fundamentally probabilistic text generators. They predict syntax, not semantics. When tasked with complex reasoning, constraint satisfaction, or mission-critical decision-making, they hallucinate — producing outputs that are syntactically plausible but logically wrong. This is not a bug to be fixed with more data or RLHF; it is an architectural limitation of autoregressive generation.

Energy Based Models offer a structural solution. Instead of committing to outputs token-by-token (where an early error cascades irrecoverably), EBMs evaluate complete configurations holistically. When a constraint is violated, gradient descent on the energy landscape can surgically fix the broken part without discarding the rest. This enables verifiable reasoning: you can mathematically prove that a solution satisfies all constraints by showing its energy is at a global minimum.

### The Self-Learning Problem

Current AI systems improve only through human-curated training data and reward signals. True autonomous self-improvement requires a system that can:

1. **Generate hypotheses** about how to improve its own architecture or parameters
2. **Evaluate those hypotheses** against an objective function that cannot be gamed
3. **Incorporate improvements** that provably reduce error

EBMs provide the evaluation mechanism: the energy function is the objective truth. A proposed improvement either lowers the energy on held-out data (real improvement) or it doesn't (rejected). This creates a closed loop for autonomous directed self-learning that doesn't depend on human feedback.

### The Implementation Gap

Despite these advantages, existing open-source EBM implementations are fragmented across research codebases with no clear path to production deployment. Researchers need fast iteration in Python/JAX; production systems need the performance and safety guarantees of Rust. No framework bridges both worlds with spec-driven quality guarantees — and none is designed from the ground up for self-improvement.

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

### FR-11: Autonomous Self-Learning Loop
The system shall support an autoresearch pipeline where:
- Candidate improvements (architectures, training algorithms, hyperparameters) are proposed and prototyped in Python/JAX
- The energy landscape serves as the objective evaluator — improvements must demonstrably lower energy on held-out validation data
- Proven improvements are transpiled to Rust for production performance
- The loop operates without human supervision, with safety guardrails (immutable validation data, rollback thresholds, execution timeouts)

### FR-12: Verifiable Reasoning
The system shall support constraint satisfaction verification: given a configuration and a set of constraints encoded in the energy function, the system shall report which constraints are satisfied (low energy) and which are violated (high energy), enabling deterministic verification of reasoning outputs.

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
