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

### REQ-VERIFY-015: Typed Reasoning Intermediate Representation

The repository shall provide a typed reasoning intermediate representation in
`python/carnot/pipeline/typed_reasoning.py`, where:
- the IR can represent user constraints extracted from the prompt
- the IR can represent ordered reasoning steps extracted from the response
- the IR can represent atomic claims attached to individual reasoning steps
- the IR can represent the final answer in a normalized verifier-friendly form
- the IR records extraction provenance including the extraction method and a
  fixed parser version string `20260412`

### REQ-VERIFY-016: Dual-Path Typed Reasoning Extraction And Validation

The same module shall provide a dual-path extractor for typed reasoning, where:
- direct structured JSON emitted by a model is parsed into the typed IR when
  the response contains a machine-readable reasoning payload
- plain-text responses fall back to deterministic post-hoc parsing of prompt
  constraints, reasoning steps, atomic claims, and the final answer
- malformed or internally inconsistent IR payloads fail validation
  deterministically rather than silently producing partial invalid state
- the extractor remains backward compatible with the existing
  `ConstraintExtractor` pipeline and does not require changes to existing
  extractors

### REQ-VERIFY-017: Deterministic Typed Reasoning Serialization And Pipeline Hook

The same capability shall expose deterministic serialization and pipeline
integration for the typed reasoning IR, where:
- the IR supports `to_dict()` / `from_dict()` plus deterministic JSON
  serialization and deserialization helpers
- validation checks include unique identifiers plus referential integrity
  between steps, claims, and the final answer
- `VerifyRepairPipeline` can surface the typed IR without changing the
  existing verification behavior or breaking current callers
- later verifiers can consume the serialized IR without depending on the
  original raw response formatting

### REQ-VERIFY-018: Semantic Failure Corpus Artifact

The repository shall provide a deterministic workflow that writes
`data/research/semantic_failure_corpus_214.jsonl`, where:
- the corpus contains at least 60 labeled failure cases
- the cases combine live traces from Exp 203 / 206 / 207 with targeted
  follow-up prompts informed by the live semantic and code-path failures
  reviewed in Exp 208
- every record includes `example_id`, `source_type`, `source_refs`, `domain`,
  `prompt`, `response`, `gold_diagnosis`, `expected_verifier_signal`, and
  `structured_reasoning_helpful`
- `gold_diagnosis` records the primary taxonomy label plus a concise
  explanation of why the response is wrong in verifier-friendly terms
- `expected_verifier_signal` records the verifier path or signal family that a
  future semantic verifier should surface for the case
- the taxonomy separates question-grounding failures, omitted premises,
  entity/quantity binding errors, unit/aggregation errors, genuine arithmetic
  slips, and code-specific oracle/property misses
- the record layout stays easy to convert into later unit tests without
  depending on the original result-artifact schemas

### REQ-VERIFY-019: Semantic Failure Corpus Summary

The same workflow shall write `results/experiment_214_results.json`, where:
- the artifact records the fixed Exp 214 run date `20260412`
- the summary reports counts by source type, source artifact, domain,
  taxonomy label, expected verifier signal, and
  `structured_reasoning_helpful`
- the summary records coverage checks confirming the corpus has at least 60
  cases and covers every required taxonomy bucket
- re-running the workflow refreshes the JSONL and summary artifacts in place
  without duplicate records or order drift

### REQ-VERIFY-020: Semantic Grounding Claim Decomposition And Deterministic Alignment

The repository shall provide a semantic grounding verifier in
`python/carnot/pipeline/semantic_grounding.py`, where:
- the verifier decomposes a question-response pair into prompt clauses and
  atomic response claims, reusing the typed reasoning IR when it is available
- the verifier extracts prompt-side entities, quantities, and required answer
  targets using deterministic rules rather than hidden chain-of-thought
- the verifier aligns response claims against prompt-side entities,
  quantities, and answer targets, with first-layer checks for entity coverage,
  quantity or premise coverage, answer-target mismatch, and unsupported
  references or assumptions when the evidence is strong
- the verifier returns structured violations that preserve the violated prompt
  clause, claim identifier, and machine-readable metadata and can be converted
  into `ConstraintResult` objects for `VerifyRepairPipeline`
- the deterministic layer is conservative enough that obviously grounded
  correct answers do not get flagged in normal live verify-repair usage

### REQ-VERIFY-021: Optional Semantic Refinement And Pipeline Integration

The same capability shall expose optional refinement and pipeline integration,
where:
- the verifier accepts an optional refinement hook for ambiguous cases, but the
  hook operates on structured summaries of prompt clauses, claims, and current
  violations rather than requiring hidden chain-of-thought
- `VerifyRepairPipeline` integrates semantic grounding additively with the
  existing extractor path, so semantic violations can fail verification even
  when the response is internally arithmetic-consistent
- the pipeline remains backward compatible for existing callers that ignore the
  new semantic-grounding analysis
- later repair or audit stages can inspect the semantic-grounding result or the
  pipeline-compatible violation objects without depending on raw prose parsing

### REQ-VERIFY-022: Structured Reasoning Emission Schema And Model Prompts

The repository shall provide a structured reasoning emission path in
`python/carnot/pipeline/structured_reasoning.py`, where:
- the emitted schema covers prompt constraints, ordered reasoning steps,
  atomic claims, and the final answer in a form that can be consumed by the
  typed reasoning IR
- prompt helpers exist for `Qwen/Qwen3.5-0.8B` and
  `google/gemma-4-E4B-it`
- the prompts request monitorable structured output without forcing verbose or
  hidden chain-of-thought
- the schema stays minimal enough that later verifier stages can inspect the
  emitted state deterministically

### REQ-VERIFY-023: Structured Emission Validation, Retry, And Fallback

The same module shall validate structured reasoning emissions before they are
trusted, where:
- malformed JSON, missing required schema sections, or typed-IR validation
  failures are rejected deterministically
- the emission path retries at least once with explicit correction feedback
  when a structured response is malformed
- the emission path can fall back to a caller-provided non-structured
  generation path when retries are exhausted
- fallback responses can still be converted into typed reasoning via the
  existing dual-path extractor when possible

### REQ-VERIFY-024: Exp 213 Policy-Gated Structured Emission

The same module shall integrate `results/monitorability_policy_213.json`, where:
- structured prompting is used only when the Exp 213 policy recommends
  `structured_json` for the task slice
- non-structured tasks continue to use the existing generation path rather than
  paying the structured-output token cost by default
- unsupported or unknown model families do not trigger the structured path
  unless an explicit prompt helper is available
- `VerifyRepairPipeline` can expose the structured emission path through an
  additive entry point without breaking existing verification or repair flows

### REQ-VERIFY-025: Shared Dual-Model Live Benchmark Harness

The repository shall provide a checkpointed live benchmark harness in
`scripts/experiment_218_live_dual_model_suite.py`, where:
- the CLI supports exactly three benchmark values:
  `gsm8k_semantic`, `humaneval_property`, and `constraint_ir`
- the harness supports exactly two target models:
  `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it`
- each benchmark runs the same high-level mode sequence:
  `baseline`, `verify_only`, and `verify_repair`
- the sampled case order is deterministic for a fixed sample seed
- each sampled case records a shared prompt seed that is reused across the
  three high-level modes so later comparisons stay paired
- long runs can resume from checkpoints scoped by benchmark, model, and mode
  without silently discarding already-completed case results

### REQ-VERIFY-026: Stable Paired Result Schema For Exp 219-221

The same harness shall emit a stable paired result schema that later live
artifacts can consume directly, where:
- the top-level payload records the fixed Exp 218 run date `20260412`
- the payload records benchmark metadata, the sampled cohort manifest, and the
  ordered paired runs for every model and high-level mode
- every cohort case records stable identifiers plus the shared prompt seeds
  for `baseline`, `verify_only`, and `verify_repair`
- every paired run records the benchmark name, model name, model hf id, mode,
  per-case results, and a mode-level summary
- re-running the harness with matching cohort metadata refreshes the output in
  place rather than appending duplicate runs or scrambling case order

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

### SCENARIO-VERIFY-015: Direct JSON Reasoning Payload Parses Into Typed IR

**Given** a model response that emits structured JSON with explicit user
constraints, reasoning steps, claims, and a final answer
**When** the typed reasoning extractor parses the response
**Then** it records `direct_json` provenance
**And** it preserves the typed sections in a validated IR
**And** deterministic JSON serialization round-trips back to the same IR

### SCENARIO-VERIFY-016: Plain-Text Reasoning Falls Back To Post-Hoc Parsing

**Given** a plain-text response with reasoning lines and a final answer marker
**When** the typed reasoning extractor parses the response
**Then** it records `fallback_text` provenance
**And** it derives prompt constraints, ordered reasoning steps, atomic claims,
  and the normalized final answer
**And** the resulting IR validates without requiring structured JSON input

### SCENARIO-VERIFY-017: VerifyRepairPipeline Surfaces Typed Reasoning Backward Compatibly

**Given** an existing verification call that still uses the current
constraint extractors
**When** `VerifyRepairPipeline.verify()` runs on the same response
**Then** the verification verdict remains backward compatible
**And** the result can additionally surface the typed reasoning IR for later
  verifier stages
**And** existing callers that ignore the new IR continue to work unchanged

### SCENARIO-VERIFY-018: Exp 214 Workflow Writes The Semantic Failure Corpus

**Given** the curated live failure artifacts from Exp 203 / 206 / 207 and the
Exp 208 code-path failure review are available
**When** the Exp 214 semantic-failure-corpus workflow runs
**Then** `data/research/semantic_failure_corpus_214.jsonl` is written with at
least 60 records
**And** each record contains the required prompt, response, diagnosis,
verifier-signal, and structured-reasoning fields
**And** the taxonomy coverage spans all six required failure categories

### SCENARIO-VERIFY-019: Exp 214 Rerun Refreshes The Corpus Deterministically

**Given** the Exp 214 corpus and summary already exist
**When** the workflow runs again
**Then** the JSONL corpus is rewritten in the same example order
**And** `results/experiment_214_results.json` is refreshed in place
**And** the summary still reports the same coverage checks and taxonomy counts

### SCENARIO-VERIFY-020: Exp 214 Semantic Omission Is Flagged Without Arithmetic Contradiction

**Given** an Exp 214 word-problem response whose arithmetic steps are locally
consistent but which omits a prompt premise needed by the asked-for answer
**When** the semantic grounding verifier runs
**Then** it emits one or more structured semantic-grounding violations
**And** at least one violation points to the missing prompt clause or answer
target mismatch rather than a generic arithmetic failure
**And** the exported pipeline-compatible constraints are sufficient to mark the
response unverified

### SCENARIO-VERIFY-021: VerifyRepairPipeline Catches The Wrong Question Answered Correctly

**Given** an Exp 214 question-grounding failure where the response computes a
related quantity correctly but answers the wrong target
**When** `VerifyRepairPipeline.verify()` runs
**Then** verification fails even if no arithmetic contradiction is extracted
**And** the surfaced violation is tagged as semantic grounding rather than
arithmetic
**And** optional refinement is not required to catch the case

### SCENARIO-VERIFY-022: Clean Structured Emission Produces Direct Typed IR

**Given** a supported model is asked for structured reasoning on a task slice
where the Exp 213 policy recommends `structured_json`
**When** the structured reasoning controller receives a well-formed emission
**Then** the emission validates as direct JSON against the required schema
**And** the resulting typed reasoning IR preserves constraints, steps, claims,
  and the final answer
**And** the controller reports that the structured path succeeded without
  falling back

### SCENARIO-VERIFY-023: Malformed Structured Emission Retries Then Falls Back

**Given** a supported model first emits malformed structured output for a task
slice that requests `structured_json`
**When** the structured reasoning controller validates the response
**Then** it records the validation failure deterministically
**And** it retries with explicit schema-correction feedback
**And** if the structured output remains malformed, it falls back to the
  caller-provided non-structured generation path instead of crashing

### SCENARIO-VERIFY-024: Policy Gate Avoids Structured Cost On Terse Tasks

**Given** the Exp 213 policy recommends a non-structured mode for a task slice
such as `code_typed_properties`
**When** the structured reasoning controller is asked to emit a response for
that task slice
**Then** it does not call the structured generation path
**And** it returns the caller-provided fallback response path instead
**And** existing `VerifyRepairPipeline` callers remain backward compatible
  unless they opt into the additive structured entry point

### SCENARIO-VERIFY-025: Resume Skips Completed Benchmark Cells Without Breaking Pairing

**Given** the Exp 218 harness already has a checkpoint for one
benchmark-model-mode cell with completed case results
**When** the same harness is run again with the same benchmark, sample seed,
and sampled case identifiers
**Then** the completed case results are reused rather than regenerated
**And** only the unfinished cases are executed
**And** the final paired run preserves the original case order and prompt
  seeds

### SCENARIO-VERIFY-026: Shared Prompt Seeds And Stable Payload Survive Re-Runs

**Given** the Exp 218 harness materializes a paired output artifact for one of
`gsm8k_semantic`, `humaneval_property`, or `constraint_ir`
**When** the harness is run again with the same benchmark, sample size, and
sample seed
**Then** each cohort case keeps the same shared prompt seeds across
  `baseline`, `verify_only`, and `verify_repair`
**And** the top-level payload still records the same benchmark manifest and
  ordered paired runs
**And** the refreshed artifact does not duplicate cases or mode entries

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
| REQ-VERIFY-015 | Not Started | Implemented | Typed reasoning IR tests |
| REQ-VERIFY-016 | Not Started | Implemented | Typed reasoning IR tests |
| REQ-VERIFY-017 | Not Started | Implemented | Typed reasoning IR tests + pipeline integration tests |
| REQ-VERIFY-018 | Not Started | Implemented | Exp 214 semantic failure corpus tests |
| REQ-VERIFY-019 | Not Started | Implemented | Exp 214 semantic failure corpus tests |
| REQ-VERIFY-020 | Not Started | Implemented | Semantic grounding verifier tests |
| REQ-VERIFY-021 | Not Started | Implemented | Semantic grounding verifier + pipeline integration tests |
| REQ-VERIFY-022 | Not Started | Implemented | Structured reasoning emission tests |
| REQ-VERIFY-023 | Not Started | Implemented | Structured reasoning retry + fallback tests |
| REQ-VERIFY-024 | Not Started | Implemented | Structured reasoning policy + pipeline entrypoint tests |
| REQ-VERIFY-025 | Not Started | Implemented | Exp 218 harness CLI + checkpoint tests |
| REQ-VERIFY-026 | Not Started | Implemented | Exp 218 stable payload + prompt seed tests |
| REQ-JEPA-002 | Not Started | Implemented | 8 Python |
