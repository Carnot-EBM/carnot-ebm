# Verifiable Reasoning Capability Specification

**Capability:** verifiable-reasoning
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-12

## Overview

Defines how Carnot provides deterministic, verifiable reasoning through energy-based constraint satisfaction. Where LLMs produce outputs that are plausible but unverifiable, Carnot evaluates configurations against an energy function that encodes hard constraints — producing outputs that are provably correct or precisely diagnosable when wrong.

This capability is the foundation for the anti-hallucination guarantee: a configuration at an energy minimum satisfies all encoded constraints by mathematical necessity, not by statistical likelihood.

## Requirements

### REQ-VERIFY-001: Constraint Encoding

The system shall support encoding logical, physical, and domain constraints as terms in an energy function, where:
- Each constraint contributes a non-negative energy term
- A satisfied constraint contributes zero (or near-zero) energy
- A violated constraint contributes energy proportional to the severity of the violation
- The total energy is the sum of all constraint terms

### REQ-VERIFY-002: Constraint Decomposition

The system shall provide per-constraint energy decomposition: given a configuration, report the individual energy contribution of each constraint term, enabling identification of which specific constraints are satisfied and which are violated.

### REQ-VERIFY-003: Verification Certificate

The system shall produce a verification result for a given configuration containing:
- Total energy (scalar)
- Per-constraint energy breakdown (named map)
- Boolean satisfaction flag per constraint (energy below threshold)
- Overall verdict: VERIFIED (all constraints satisfied) or VIOLATED (with list of failing constraints)

### REQ-VERIFY-004: Constraint Composition

The system shall support composing constraint sets from independent sources:
- Multiple constraint functions can be combined into a single energy function
- Constraints can be weighted to express relative importance
- New constraints can be added without modifying existing ones

### REQ-VERIFY-005: Gradient-Based Repair

When a configuration violates one or more constraints, the system shall support targeted repair via gradient descent on the violated constraint terms only, without perturbing the parts of the configuration that already satisfy their constraints.

### REQ-VERIFY-006: Energy Landscape Certification

The system shall provide tools to analyze the energy landscape around a solution:
- Local minimum verification (Hessian eigenvalue analysis)
- Basin of attraction estimation (perturbation sensitivity)
- Saddle point detection

### REQ-VERIFY-007: Deterministic Reproducibility

Given identical inputs, model parameters, and random seeds, the system shall produce bit-identical energy values and verification results across runs (within the same language implementation).

### REQ-VERIFY-008: Extraction Autopsy Records

The system shall support experiment-grade extraction autopsies for live model
responses, where:
- Full response text is preserved for each sampled question
- Arithmetic extractor matches are recorded with exact matched expressions and
  verification verdicts
- Wrong answers can be annotated with a diagnosed failure category and a
  recommended extraction approach that would have caught the error
- The autopsy result is serializable to JSON for offline review and follow-on
  extractor development

### REQ-VERIFY-009: SMT-Backed Arithmetic Extraction

The system shall support an SMT-backed arithmetic extractor for instruction-
tuned reasoning traces, where:
- Explicit equations and verbal arithmetic steps (for example, "half of 48")
  are normalized into solver constraints
- Multi-step chains can reference values derived earlier in the response
- Approximate claims (for example, "about 150") are represented as bounded
  numeric ranges rather than exact-equality checks
- Each extracted step records whether the solver found it satisfiable and, if
  not, the violation includes the claimed result, the solver-derived result,
  and the source step text

### REQ-VERIFY-010: LLM-Assisted Arithmetic Claim Extraction

The system shall support an auxiliary LLM-based arithmetic extractor for
free-form reasoning traces, where:
- A small language model is prompted with the response text only
- The prompt requests canonical claim lines in the form
  `CLAIM: a OP b = c`
- The extractor parses zero or more claim lines, ignoring malformed model
  output without crashing
- Each extracted claim is verified by deterministic arithmetic and records the
  operands, operator, claimed result, computed result, satisfaction verdict,
  raw claim text, and extraction latency
- Each extracted claim is returned as a `ConstraintResult` that can flow
  through the existing `VerifyRepairPipeline` / `ComposedEnergy` path

### REQ-VERIFY-011: Constraint IR Benchmark Corpus

The repository shall provide a deterministic workflow that writes
`data/research/constraint_ir_benchmark_211.jsonl`, where:
- the benchmark contains between 80 and 120 examples
- every record includes `prompt`, `gold_atomic_constraints`,
  `constraint_types`, `expected_verifier_path`, `expected_answer_schema`, and
  `free_form_reasoning_monitorable`
- the corpus includes live GSM8K semantic or question-grounding failures drawn
  from Exp 203 / 206 / 207
- the corpus includes multi-constraint instruction-following prompts inspired
  by VIFBench, ConstraintBench, CFBench, FollowBench, or RealInstruct-style
  task shapes
- the corpus includes code prompts whose requirements can be represented as
  typed properties rather than only free-form textual judgments
- the overall mix spans literal constraints, compositional constraints, and
  semantic/question-grounding constraints

### REQ-VERIFY-012: Constraint IR Benchmark Summary

The same workflow shall write `results/experiment_211_results.json`, where:
- the artifact records the fixed Exp 211 run date used for the benchmark
- the summary reports counts by source family, constraint type, verifier path,
  answer-schema type, and reasoning-monitorability flag
- the summary records simple coverage checks confirming the benchmark stays in
  the 80-120 example range and covers the required benchmark slices
- re-running the workflow rewrites the JSONL and summary artifacts
  deterministically without duplicate records

### REQ-VERIFY-013: Monitorability Audit Artifact

The repository shall provide a workflow that writes
`results/experiment_213_results.json`, where:
- the workflow evaluates `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it`
  against a representative subset of `data/research/constraint_ir_benchmark_211.jsonl`
- each model is evaluated in three response modes: free-form reasoning,
  answer-only terse output, and structured JSON or typed reasoning scaffold
- the artifact records per-model and per-mode measurements for parseability,
  constraint coverage, semantic visibility of the true error, answer quality,
  token cost, and latency
- the artifact preserves the fixed Exp 213 run date `20260412` and the
  benchmark subset composition used for the audit
- re-running the workflow refreshes the artifact in place rather than
  appending duplicate runs

### REQ-VERIFY-014: Monitorability Fallback Policy

The same workflow shall write `results/monitorability_policy_213.json`, where:
- the policy states when Carnot should request structured reasoning, accept
  terse output, or distrust free-form traces
- the policy is derived from the measured Exp 213 audit metrics rather than
  hard-coded without evidence
- the policy includes model-family and task-shape guidance when the audit
  shows materially different monitorability behavior across response modes
- the policy remains machine-readable and deterministic given the measured
  audit summary

### REQ-JEPA-002: Tier 3 Fast-Path Gate

The `VerifyRepairPipeline.verify()` method shall support an optional JEPA predictor gate that:
- Accepts `jepa_predictor` (optional `JEPAViolationPredictor`) and `jepa_threshold` (float, default 0.5)
- When a predictor is provided, embeds the first 50 whitespace-split tokens of the response and queries the predictor for per-domain violation probabilities
- If `max(probabilities) < jepa_threshold`, returns a `VerificationResult` with `mode="FAST_PATH"`, `skipped=True`, and `verified=True` (optimistic low-risk assumption)
- When no predictor is provided, behaves identically to the original full-pipeline path (backward compatible)
- `VerificationResult` shall include `mode: str = "FULL"` and `skipped: bool = False` fields (defaulting to full-path semantics for all existing callers)

## Scenarios

### SCENARIO-VERIFY-001: Sudoku Constraint Satisfaction

**Given** a 9x9 Sudoku puzzle encoded as an energy function where:
- Row uniqueness constraint: E_row = sum of penalties for duplicate values in each row
- Column uniqueness constraint: E_col = same for columns
- Box uniqueness constraint: E_box = same for 3x3 boxes
- Given clue constraint: E_clue = penalty for deviating from provided clues
**When** the system verifies a candidate solution
**Then** it reports per-constraint energy (E_row, E_col, E_box, E_clue)
**And** a correct solution has total energy = 0 and verdict = VERIFIED
**And** an incorrect solution has energy > 0 with specific failing constraints identified

### SCENARIO-VERIFY-002: Gradient-Based Repair

**Given** a Sudoku configuration with 3 constraint violations
**When** gradient-based repair is applied for N steps
**Then** the number of violations decreases monotonically (on average)
**And** previously satisfied constraints remain satisfied (energy does not increase for non-violated terms)

### SCENARIO-VERIFY-003: Constraint Decomposition

**Given** an energy function composed of 4 independent constraint terms
**When** per-constraint decomposition is requested for a configuration
**Then** the sum of individual constraint energies equals the total energy (within floating-point tolerance)
**And** each constraint is identified by name

### SCENARIO-VERIFY-004: Constraint Composition

**Given** two independently defined constraint sets A and B
**When** they are composed into a single energy function with weights [1.0, 2.0]
**Then** the composed energy equals E_A(x) + 2.0 * E_B(x)
**And** verification reports decomposed energies for both A and B sub-constraints

### SCENARIO-VERIFY-005: Local Minimum Verification

**Given** a configuration that the sampler reports as converged
**When** local minimum verification is requested
**Then** the system reports whether all Hessian eigenvalues are positive (true local minimum)
**And** estimates the basin of attraction radius via perturbation analysis

### SCENARIO-VERIFY-006: Deterministic Reproducibility

**Given** a model and input configuration
**When** energy is computed twice with the same random seed
**Then** the results are bit-identical

### SCENARIO-VERIFY-007: Comparison with LLM Output

**Given** a constraint satisfaction problem (e.g., scheduling, planning)
**When** an LLM-generated solution and a Carnot-minimized solution are both verified
**Then** the Carnot solution has energy = 0 (all constraints satisfied)
**And** the LLM solution has energy > 0 in at least some constraint terms (hallucinated violations)
**And** the specific violations in the LLM output are identified by constraint name

### SCENARIO-VERIFY-008: Live Extraction Autopsy

**Given** 20 live GSM8K responses from an instruction-tuned model
**When** the extraction-autopsy workflow runs
**Then** it stores each question, full response, extracted final answer,
correctness, and arithmetic extractor matches
**And** each wrong answer includes a diagnosed failure category and a proposed
extraction strategy that would have caught the reasoning error
**And** at least three correct answers are preserved as contrast cases

### SCENARIO-VERIFY-009: SMT Verification of Arithmetic Chains

**Given** a chain-of-thought response containing explicit equations, verbal
arithmetic steps, and approximate values
**When** the SMT-backed arithmetic extractor verifies the response
**Then** each arithmetic step is returned with a satisfiable/unsatisfiable
verdict
**And** unsatisfiable steps include the solver-derived correction and the
offending source step
**And** correct chains produce zero arithmetic false positives

### SCENARIO-VERIFY-010: LLM Extractor Recovers Natural-Language Arithmetic

**Given** a response whose arithmetic is expressed in natural language rather
than explicit regex-readable equations
**When** the LLM-assisted arithmetic extractor is run
**Then** it emits canonical `CLAIM: a OP b = c` lines for the verifiable steps
**And** wrong extracted claims are surfaced as arithmetic violations by the
existing `VerifyRepairPipeline`
**And** the extractor records per-response latency for the auxiliary LLM call

### SCENARIO-VERIFY-011: Exp 211 Benchmark Writes the Curated Corpus

**Given** the Exp 211 benchmark workflow is run
**When** it materializes the benchmark artifacts
**Then** `data/research/constraint_ir_benchmark_211.jsonl` is written with
between 80 and 120 records
**And** each record contains the required prompt, constraint, verifier-path,
answer-schema, and monitorability fields
**And** the corpus includes live GSM8K semantic failures plus instruction and
code-oriented prompt constraints

### SCENARIO-VERIFY-012: Exp 211 Rerun Is Deterministic

**Given** the Exp 211 benchmark artifacts already exist
**When** the workflow runs again
**Then** the JSONL corpus is rewritten in the same example order
**And** `results/experiment_211_results.json` is refreshed in place
**And** the summary still reports the same benchmark counts and coverage checks

### SCENARIO-VERIFY-013: Exp 213 Audit Writes Both Artifacts

**Given** the Exp 211 benchmark exists
**When** the Exp 213 monitorability-audit workflow runs
**Then** `results/experiment_213_results.json` is written with per-model and
per-mode metrics for parseability, constraint coverage, semantic visibility,
answer quality, token cost, and latency
**And** `results/monitorability_policy_213.json` is written with structured
fallback guidance
**And** the audit records the fixed Exp 213 run date and the representative
subset that was evaluated

### SCENARIO-VERIFY-014: Policy Prefers Structured Or Terse Output When Needed

**Given** the Exp 213 audit shows that one or more response modes have weak
monitorability or poor parseability on a task slice
**When** the fallback policy is derived from the audit summary
**Then** the policy recommends structured reasoning when explicit intermediate
state is useful and reliably parseable
**And** the policy recommends terse output when reasoning text adds little
visibility relative to cost
**And** the policy marks free-form traces as distrusted when their observed
monitorability is too weak to support verifier decisions

## Implementation Status

| Requirement | Rust | Python | Tests |
|------------|------|--------|-------|
| REQ-VERIFY-001 | Implemented | Implemented | 3 Rust + 3 Python |
| REQ-VERIFY-002 | Implemented | Implemented | 2 Rust + 1 Python |
| REQ-VERIFY-003 | Implemented | Implemented | 2 Rust + 2 Python |
| REQ-VERIFY-004 | Implemented | Implemented | 2 Rust + 2 Python |
| REQ-VERIFY-005 | Implemented | Implemented | 2 Rust + 2 Python |
| REQ-VERIFY-006 | Not Started | Not Started | Not Started |
| REQ-VERIFY-007 | Implemented | Implemented | 1 Rust + 2 Python |
| REQ-VERIFY-008 | Not Started | Implemented | 10 Python |
| REQ-VERIFY-009 | Not Started | Implemented | 29 + paired live benchmark Python |
| REQ-VERIFY-010 | Not Started | Implemented | 14 + paired live benchmark Python |
| REQ-VERIFY-011 | Not Started | Implemented | Exp 211 benchmark generator + artifact tests |
| REQ-VERIFY-012 | Not Started | Implemented | Exp 211 benchmark generator + artifact tests |
| REQ-VERIFY-013 | Not Started | Implemented | 9 Python + live Exp 213 audit artifact |
| REQ-VERIFY-014 | Not Started | Implemented | 9 Python + live Exp 213 policy artifact |
| REQ-JEPA-002 | Not Started | Implemented | 8 Python |
