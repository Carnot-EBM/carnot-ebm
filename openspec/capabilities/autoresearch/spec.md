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

**Cross-language note:** The Ackley benchmark in Python/JAX adds a small epsilon (1e-10) inside the sqrt to prevent NaN gradients from jax.grad at the origin (d/dx sqrt(0) is undefined). The Rust implementation uses numerical gradients instead. Energy values may differ by up to ~1e-5 near the origin. This is an intentional implementation divergence documented here rather than in the code alone.

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

### REQ-AUTO-011: Trajectory Analysis

The system shall provide parallel analyst sub-agents that extract structured lessons from experiment outcomes:
- **Error analysts** receive a failed experiment's full trajectory (hypothesis code, sandbox metrics, evaluation verdict, error messages) and diagnose the root cause via LLM reasoning (e.g., "gradient explosion due to Rosenbrock curvature exceeding step size")
- **Success analysts** receive an accepted experiment's trajectory and extract the generalizable optimization pattern (e.g., "annealing schedules improve convergence on multi-basin landscapes")
- Analysts run in parallel via thread pool and produce structured `Lesson` objects with: title, description, concrete examples, confidence score, applicable benchmarks, model tier, and lesson type
- Analyst dispatch is configurable (can be disabled to save LLM cost)

### REQ-AUTO-012: Skill Directory

The system shall maintain a persistent, evolving optimization playbook (skill directory) that accumulates lessons across iterations:
- **SKILL.md**: Natural-language optimization guide, periodically rewritten by LLM from accumulated lessons
- **scripts/**: Proven sampler configurations and code snippets extracted from successful hypotheses
- **references/**: Benchmark-specific edge cases and niche patterns (low-frequency lessons)
- **lessons.json**: Structured lesson store with confidence scores and metadata
- The skill directory shall be serialized as `to_prompt_context()` and injected into the hypothesis generator's prompt, replacing the shallow `recent_failures` list with structured knowledge
- Maximum lesson count shall be configurable (default 200) to prevent unbounded growth

### REQ-AUTO-013: Hierarchical Lesson Consolidation

The system shall consolidate raw lessons into a conflict-free set via hierarchical tree-reduction:
- Group lessons into batches of configurable size (default 32)
- For each batch, use LLM to: deduplicate equivalent lessons (merging confidence scores), resolve contradictory lessons (keeping the better-supported one), and extract cross-cutting meta-patterns
- Repeat reduction until a single batch remains (L = ceil(log_batch(N)) levels)
- Filter lessons below a configurable minimum confidence threshold (default 0.3)
- Consolidation runs periodically (configurable interval, default every 5 iterations)

### REQ-AUTO-014: Cross-Tier Skill Transfer

The system shall support transfer of optimization knowledge across model tiers:
- Lessons learned on fast-to-evaluate tiers (Ising) shall be available when generating hypotheses for slower tiers (Gibbs, Boltzmann)
- Each lesson is tagged with its originating model tier and applicable benchmarks
- The `to_prompt_context()` method accepts a target model tier and includes relevant lessons from other tiers
- Tier-specific edge cases are stored in the references subdirectory, not propagated as general lessons

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

### SCENARIO-AUTO-008: Error Analyst Diagnoses Gradient Explosion

**Given** a hypothesis that attempted Langevin sampling on Rosenbrock with step_size=0.1
**And** the sandbox produced NaN energies after 47 steps
**When** the error analyst receives the full experiment trajectory
**Then** it produces a Lesson with title "Gradient explosion on steep landscapes"
**And** description identifies the `100*(x[i+1]-x[i]^2)^2` term curvature as the root cause
**And** applicable_benchmarks includes "rosenbrock"
**And** confidence is >= 0.7 (clear diagnosis from the execution trace)

### SCENARIO-AUTO-009: Success Analyst Extracts Annealing Pattern

**Given** a hypothesis that used step_size annealing from 0.1 to 0.001 over 5000 steps
**And** it was accepted with 30% energy improvement on DoubleWell
**When** the success analyst receives the full experiment trajectory
**Then** it produces a Lesson with title "Step-size annealing for multi-basin landscapes"
**And** the description generalizes beyond the specific parameters to the annealing principle
**And** lesson_type is "success_pattern"

### SCENARIO-AUTO-010: Lessons Consolidated Across 10 Iterations

**Given** 15 raw lessons accumulated over 10 iterations (8 error, 7 success)
**And** 3 pairs of near-duplicate lessons (e.g., both say "small step sizes prevent divergence")
**When** hierarchical consolidation runs
**Then** duplicates are merged (confidence increased)
**And** the consolidated set has fewer lessons than the input
**And** contradictory lessons (e.g., "use large step size" vs "use small step size") are resolved
**And** lessons below min_confidence are filtered out

### SCENARIO-AUTO-011: Ising Skill Transfers to Gibbs Model

**Given** a skill directory with 5 lessons learned from Ising model experiments
**And** one lesson is "HMC outperforms Langevin on narrow-valley landscapes" (model_tier="ising")
**When** generating hypotheses for the Gibbs model tier
**Then** `to_prompt_context(model_tier="gibbs")` includes the Ising HMC lesson
**And** the generator's prompt contains this cross-tier knowledge
**And** the generated hypothesis tries HMC on the Gibbs tier

## Implementation Status

| Requirement | Rust | Python | Tests |
|------------|------|--------|-------|
| REQ-AUTO-001 | Implemented | Partial | 14 Rust |
| REQ-AUTO-002 | Partial | Implemented | 3 Python |
| REQ-AUTO-003 | Not Started | Implemented | Integration |
| REQ-AUTO-004 | Not Started | Implemented | 13 + 21 Python |
| REQ-AUTO-005 | Not Started | Implemented | 7 Python |
| REQ-AUTO-006 | Not Started | Not Started | Not Started |
| REQ-AUTO-007 | Not Started | Not Started | Not Started |
| REQ-AUTO-008 | Not Started | Implemented | 5 Python |
| REQ-AUTO-009 | Not Started | Implemented | 4 Python |
| REQ-AUTO-010 | Not Started | Not Started | Not Started |
| REQ-AUTO-011 | N/A | Implemented | 10+ Python |
| REQ-AUTO-012 | N/A | Implemented | 11+ Python |
| REQ-AUTO-013 | N/A | Implemented | 9+ Python |
| REQ-AUTO-014 | N/A | Implemented | Integration |
