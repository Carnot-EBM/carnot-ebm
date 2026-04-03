# Autoresearch Capability Specification

**Capability:** autoresearch
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-11

## Overview

Defines Carnot's autonomous self-improvement loop: a pipeline where the system proposes modifications to its own architecture, training algorithms, or hyperparameters, evaluates them against objective energy-based benchmarks, and incorporates proven improvements — all without human supervision.

The fundamental insight: because the energy function provides mathematical ground truth (energy decreased = real improvement, energy didn't = rejected), the system has an objective evaluator that cannot be gamed. This is the property that makes autonomous self-improvement safe and convergent, unlike LLM-based self-evaluation which inherits all the hallucination problems of the base model.

## Requirements

### REQ-AUTO-001: Benchmark Suite

The system shall provide a standard benchmark suite of energy landscape problems with known optimal solutions, including:
- **Analytical benchmarks**: DoubleWell, Rosenbrock, Ackley, Rastrigin (known global minima)
- **Structured benchmarks**: Sudoku (constraint satisfaction), graph coloring, scheduling
- **Statistical benchmarks**: Gaussian mixture models (known distribution parameters)

Each benchmark shall define:
- Energy function
- Known optimal energy value (or bound)
- Evaluation metrics (convergence speed, final energy, sample quality, wall-clock time)
- Pass/fail thresholds

### REQ-AUTO-002: Baseline Registry

The system shall maintain a registry of baseline performance metrics for the current production models and algorithms, stored as versioned JSON files. Each entry records:
- Benchmark name
- Algorithm/model configuration
- Metrics (final energy, convergence steps, wall-clock time, memory usage)
- Git commit hash of the implementation
- Timestamp

### REQ-AUTO-003: Hypothesis Generation Interface

The system shall define a structured format for improvement hypotheses:
- **Target**: what is being modified (architecture, sampler, training algorithm, hyperparameter)
- **Rationale**: why this might improve performance (optional, for human-readable logging)
- **Specification**: a complete, executable description of the change (Python/JAX code or configuration diff)
- **Expected impact**: which benchmark metrics should improve and by how much
- **Risk assessment**: which metrics might degrade

### REQ-AUTO-004: Sandbox Execution

The system shall execute hypothesis code in an isolated environment with:
- Read-only access to immutable validation datasets
- Hard timeout (configurable, default 30 minutes)
- Memory limit (configurable, default 16GB)
- No network access
- No write access to production code or data
- Captured stdout/stderr and metrics output

### REQ-AUTO-005: Evaluation Protocol

The system shall evaluate sandbox results against baselines using:
- **Primary gate**: benchmark energy must be <= baseline energy (improvement or at least no regression)
- **Secondary gate**: wall-clock time must be <= 2x baseline time (not catastrophically slower)
- **Tertiary gate**: memory usage must be <= 2x baseline usage
- All three gates must pass for the hypothesis to advance

### REQ-AUTO-006: Cross-Language Validation

For hypotheses that pass evaluation in Python/JAX:
- The system shall support transpilation to Rust (initially agent-assisted, eventually automated)
- The Rust implementation must produce energy values within floating-point tolerance of the JAX implementation on the same inputs
- The Rust implementation must meet or exceed the JAX implementation's wall-clock performance

### REQ-AUTO-007: Rollback Mechanism

The system shall support automatic rollback:
- If production energy metrics degrade by more than a configurable threshold (default 5%) over a monitoring window (default 1 hour), automatically revert to the previous version
- All reverted changes are logged with the regression metrics
- Reverted hypotheses are added to a "rejected" registry to prevent re-proposal

### REQ-AUTO-008: Experiment Logging

The system shall maintain a structured experiment log recording:
- Hypothesis (full specification)
- Sandbox results (metrics, stdout, stderr)
- Evaluation verdict (pass/fail per gate)
- If transpiled: Rust validation results
- If deployed: production monitoring results
- If rolled back: regression metrics and reason

### REQ-AUTO-009: Safety Invariants

The following invariants shall hold at all times during autoresearch:
- Validation data is never modified by any component
- Production model can always be restored to the last known-good state within 60 seconds
- No hypothesis can modify the evaluation protocol or benchmark definitions
- No hypothesis can modify its own evaluation criteria
- The system shall halt and alert if it detects more than N consecutive failures (configurable, default 10)

### REQ-AUTO-010: Improvement Composition

When multiple hypotheses independently pass evaluation, the system shall:
- Test them in combination (improvements may conflict)
- Apply the combination only if joint evaluation passes all gates
- If conflicting, rank by primary gate improvement magnitude and apply the best

## Scenarios

### SCENARIO-AUTO-001: Successful Self-Improvement Cycle

**Given** a baseline Ising model trained with CD-1, achieving energy -5.2 on the DoubleWell benchmark
**When** the autoresearch loop proposes switching to Denoising Score Matching
**And** the sandbox evaluates DSM on DoubleWell, achieving energy -5.8
**And** wall-clock time is 0.7x baseline
**Then** the hypothesis passes all three gates
**And** it is queued for Rust transpilation
**And** after Rust validation matches JAX output, it is merged as the new baseline
**And** the baseline registry is updated with the new metrics

### SCENARIO-AUTO-002: Rejected Hypothesis

**Given** a baseline Langevin sampler with step_size=0.01
**When** the autoresearch loop proposes step_size=0.5
**And** the sandbox evaluates this, producing divergent (NaN) energies
**Then** the hypothesis fails the primary gate
**And** it is logged as rejected with reason "divergent energy"
**And** the production model is unaffected

### SCENARIO-AUTO-003: Automatic Rollback

**Given** a hypothesis that passed sandbox evaluation
**And** was transpiled to Rust and deployed
**When** production energy metrics degrade by 8% over the monitoring window
**Then** the system automatically reverts to the previous version
**And** logs the regression with metrics
**And** adds the hypothesis to the rejected registry

### SCENARIO-AUTO-004: Safety Invariant Enforcement

**Given** a hypothesis that attempts to write to the validation dataset
**When** it is executed in the sandbox
**Then** the write is blocked by the sandbox filesystem policy
**And** the hypothesis is immediately terminated and flagged as unsafe

### SCENARIO-AUTO-005: Consecutive Failure Halt

**Given** a configuration with max_consecutive_failures=10
**When** 10 consecutive hypotheses fail evaluation
**Then** the autoresearch loop halts
**And** an alert is generated with the failure log
**And** the system waits for human review before resuming

### SCENARIO-AUTO-006: Improvement Composition

**Given** Hypothesis A improves sampler convergence by 15%
**And** Hypothesis B improves training loss by 10%
**When** both pass individual evaluation
**And** combined evaluation also passes all gates
**Then** both improvements are merged as a single update
**And** the baseline registry reflects the combined metrics

### SCENARIO-AUTO-007: Benchmark Regression Prevention

**Given** a hypothesis that improves DoubleWell energy by 5%
**But** degrades Rosenbrock energy by 3%
**When** evaluated across the full benchmark suite
**Then** the hypothesis is flagged for review (mixed results)
**And** is not auto-merged (requires human decision on the tradeoff)

## Implementation Status

| Requirement | Rust | Python | Tests |
|------------|------|--------|-------|
| REQ-AUTO-001 | Not Started | Not Started | Not Started |
| REQ-AUTO-002 | Not Started | Not Started | Not Started |
| REQ-AUTO-003 | Not Started | Not Started | Not Started |
| REQ-AUTO-004 | Not Started | Not Started | Not Started |
| REQ-AUTO-005 | Not Started | Not Started | Not Started |
| REQ-AUTO-006 | Not Started | Not Started | Not Started |
| REQ-AUTO-007 | Not Started | Not Started | Not Started |
| REQ-AUTO-008 | Not Started | Not Started | Not Started |
| REQ-AUTO-009 | Not Started | Not Started | Not Started |
| REQ-AUTO-010 | Not Started | Not Started | Not Started |
