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

### REQ-VERIFY-027: Exp 219 GSM8K Semantic Artifact Metrics And Trace Data

The same harness shall support writing `results/experiment_219_results.json`
for the `gsm8k_semantic` benchmark, where:
- the top-level payload records experiment id `219` when the output path is
  `results/experiment_219_results.json`
- the artifact preserves the fixed run date `20260412` plus live-run metadata
  needed to interpret the benchmark, including the forced-live setting,
  checkpoint directory, max-repair limit, and monitorability-policy source
- each model summary reports baseline, verify-only, and verify-repair accuracy
  together with paired deltas, semantic-violation detections, false positives,
  parse coverage, repair yield, and latency/token overhead
- each per-question result preserves the raw response plus verifier-visible
  artifacts needed for later trace learning, including typed-reasoning parse
  status, semantic-grounding violations, and repair history when repairs occur
- re-running the benchmark with matching cohort metadata refreshes the same
  artifact in place without changing the sampled case order

### REQ-VERIFY-028: Exp 220 HumanEval Property Artifact Metrics And Trace Data

The same harness shall support writing `results/experiment_220_results.json`
for the `humaneval_property` benchmark, where:
- the top-level payload records experiment id `220` when the output path is
  `results/experiment_220_results.json`
- each model summary compares the same paired cohort across baseline pass@1,
  execution-only accepted pass@1, execution-plus-property accepted pass@1, and
  full verify-repair pass@1
- each model summary reports wrong-answer detections, false positives,
  prompt-derived property violations, repair success rate, and verification or
  repair latency needed to compare execution-only checks against the stronger
  property-derived verifier
- the artifact explicitly records whether prompt-derived properties catch bugs
  that the official HumanEval harness alone did not reject
- each per-problem result preserves the generated code, execution-only
  findings, execution-plus-property findings, and repair traces needed for
  later self-learning and audit analysis
- re-running the benchmark with matching cohort metadata refreshes the same
  artifact in place without changing the sampled case order

### REQ-VERIFY-029: Exp 221 Constraint IR Artifact Metrics And Failure Taxonomy

The same harness shall support writing `results/experiment_221_results.json`
for the `constraint_ir` benchmark, where:
- the top-level payload records experiment id `221` when the output path is
  `results/experiment_221_results.json`
- the harness deterministically enriches Exp 211 benchmark rows with the task
  slices needed to apply the Exp 213 response-mode policy across live GSM8K,
  instruction-following, and code-typed-property cases
- each model summary reports parse success, constraint extraction coverage,
  exact satisfaction, partial satisfaction, semantic-violation detection, and
  repair yield over the paired cohort
- each model summary breaks failures down by constraint type and by observed
  output style so the artifact makes literal versus semantic versus
  search-or-optimization-limited errors visible without relying on hidden
  chain-of-thought
- each per-case result preserves the raw response, observed output style,
  deterministic scoring breakdown, and any explicitly labeled heuristic or
  model-assisted judge path used during evaluation
- deterministic code scoring keeps any executed prompt-derived code probe within
  a bounded timeout so one non-terminating answer cannot block the artifact
  refresh
- re-running the benchmark with matching cohort metadata refreshes the same
  artifact in place without changing the sampled case order

### REQ-VERIFY-030: Live Trace Ingestion And Provenance Gate

The repository shall provide a live-trace ingestion workflow for Exp 222,
where:
- the workflow reads the checked-in live artifacts
  `results/experiment_219_results.json`,
  `results/experiment_220_results.json`, and
  `results/experiment_221_results.json`
- case-level verifier outcomes are normalized into trace events that preserve
  the source experiment, benchmark, model, case identifier, response mode,
  derived error taxonomy, and provenance confidence
- only high-confidence, non-ambiguous true-positive traces are admitted into
  learned memory
- false positives, false negatives, and ambiguous traces are preserved as
  quarantined provenance records rather than silently contaminating memory
- re-running the workflow on the same source artifacts rebuilds the same
  normalized trace set deterministically

### REQ-VERIFY-031: Exp 222 Live Constraint Memory And Repair Patch Artifacts

The same workflow shall write `results/constraint_memory_live_222.json`, where:
- the artifact remains compatible with the existing `ConstraintMemory` pattern
  schema so learned patterns can be loaded by later memory-aware code
- the artifact adds accepted and quarantined live-trace provenance records
  alongside the learned pattern table
- the artifact records reusable repair snippets or prompt patches derived from
  live verify-repair histories, together with support counts and observed
  repair outcomes when available
- re-running the workflow refreshes the artifact in place without duplicating
  pattern or snippet records

### REQ-VERIFY-032: Exp 222 Reliability, Retrieval, And Policy Update Summary

The same workflow shall write `results/experiment_222_results.json`, where:
- the artifact records the fixed Exp 222 run date `20260412`
- the summary reports model- and benchmark-specific verifier reliability stats,
  including true positives, false positives, false negatives, true negatives,
  precision, and recall
- the summary reports memory growth, chronological retrieval usefulness, and
  the precision of reused learned patterns
- the summary emits machine-readable monitorability-policy update suggestions
  grounded in live trace evidence rather than simulated traces
- low-confidence or quarantined traces do not contribute to memory-growth or
  policy-update counts as if they were accepted learning events

### REQ-VERIFY-033: Exp 223 Held-Out Chronological Replay Cohorts

The repository shall provide an Exp 223 replay workflow over the checked-in
live artifacts from Exp 219, Exp 220, and Exp 221, where:
- the workflow deterministically reconstructs per-case replay items for the
  `verify_only`, `baseline`, and `verify_repair` modes of each live benchmark
- the workflow derives deterministic held-out slices from each of Exp 219,
  Exp 220, and Exp 221 so evaluation measures reuse rather than memorization
- held-out replay items are evaluated in chronological order without being used
  to update the tracker or memory state for later held-out items
- only non-held-out live traces may contribute learning updates during the
  replay

### REQ-VERIFY-034: Exp 223 Strategy Comparison And Over-Time Metrics

The Exp 223 workflow shall write `results/experiment_223_results.json`, where:
- the artifact records the fixed Exp 223 run date `20260412`
- the workflow compares `no_learning`, `tracker_only`, and
  `tracker_plus_memory` replay strategies on the same held-out items
- the summary reports held-out task metrics over time for GSM8K accuracy,
  HumanEval pass-rate, and prompt-side exact constraint satisfaction
- the summary reports held-out false-positive counts and whether each learning
  strategy stayed within a zero-additional-false-positive regression budget
  relative to `no_learning`
- the summary reports retrieval hit rate and retrieval precision for any
  strategy that reuses live-derived memory patterns

### REQ-VERIFY-035: Exp 223 Live-Only Provenance And Transfer Analysis

The same artifact shall make the learning provenance explicit, where:
- every tracker update and memory-pattern update is traceable to prior
  non-held-out live traces only
- the artifact reports per-model transfer effects, including when prior traces
  from one model materially help or gate decisions for the other model on
  held-out cases
- the artifact distinguishes same-model support from cross-model support for
  tracker and memory decisions
- improvements are not attributed to simulated traces, synthetic labels, or
  prebuilt full-corpus memory snapshots that include held-out cases

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

### SCENARIO-VERIFY-027: Exp 219 GSM8K Artifact Records Semantic Metrics And Trace Evidence

**Given** the shared harness runs `gsm8k_semantic` with output path
`results/experiment_219_results.json`
**When** the Exp 219 artifact is written
**Then** the top-level payload records experiment id `219`
**And** each model summary includes accuracy, paired deltas, semantic
  violations detected, false positives, parse coverage, repair yield, and
  latency/token overhead
**And** each per-question result preserves the raw response together with
  typed-reasoning parse status, semantic-grounding evidence, and repair
  history needed for later trace learning

### SCENARIO-VERIFY-028: Exp 220 HumanEval Artifact Separates Execution-Only And Property Verification

**Given** the shared harness runs `humaneval_property` with output path
`results/experiment_220_results.json`
**When** the Exp 220 artifact is written
**Then** the top-level payload records experiment id `220`
**And** each model summary includes the baseline, execution-only,
  execution-plus-property, and verify-repair comparisons with pass@1,
  property-violation totals, repair success rate, and latency
**And** the artifact records whether prompt-derived properties caught a bug
  that the official HumanEval tests alone would have accepted
**And** each per-problem result preserves the generated candidate code plus
  execution and repair traces needed for later self-learning

### SCENARIO-VERIFY-029: Exp 221 Constraint IR Artifact Records Prompt-Side Constraint Metrics

**Given** the shared harness runs `constraint_ir` with output path
`results/experiment_221_results.json`
**When** the Exp 221 artifact is written
**Then** the top-level payload records experiment id `221`
**And** each model summary includes parse success, extraction coverage,
  exact satisfaction, partial satisfaction, semantic-violation counts,
  repair yield, and failure breakdowns by constraint type and output style
**And** each per-case result preserves the raw response together with the
  deterministic constraint-scoring breakdown, observed output style, and any
  explicitly labeled heuristic or model-assisted judging metadata used during
  evaluation
**And** a non-terminating code answer is marked as a bounded scoring failure
  instead of stalling the benchmark refresh

### SCENARIO-VERIFY-030: Ambiguous Live Traces Stay Quarantined

**Given** a live trace whose verifier signal is ambiguous or is contradicted by
the benchmark outcome
**When** the Exp 222 ingestion workflow runs
**Then** the trace is preserved with its provenance and exclusion reason
**And** it does not increment any learned `ConstraintMemory` pattern counts

### SCENARIO-VERIFY-031: Repeated High-Confidence Live Patterns Become Reusable

**Given** three or more high-confidence live traces with the same benchmark,
domain, and error pattern
**And** a later live trace exposes the same error pattern again
**When** the Exp 222 ingestion workflow runs in chronological replay order
**Then** the learned pattern is promoted into the memory artifact
**And** the later trace is counted as a retrieval-usefulness hit for the
  promoted pattern

### SCENARIO-VERIFY-032: Live Repair Histories Yield Reusable Prompt Patches

**Given** verify-repair histories from Exp 220 or Exp 221 preserve repeated
repair prompts for the same failure shape
**When** the Exp 222 ingestion workflow consolidates the live traces
**Then** it emits a reusable repair snippet or prompt patch with support and
  observed outcome counts
**And** the snippet remains traceable to the source experiment, model, and
  case histories that produced it

### SCENARIO-VERIFY-033: Held-Out Replay Does Not Learn From Evaluation Cases

**Given** deterministic held-out slices from Exp 219, Exp 220, and Exp 221
**When** the Exp 223 workflow replays the live traces chronologically
**Then** held-out cases are evaluated in order
**And** only earlier non-held-out live traces may update the tracker or memory
  state seen by those held-out cases
**And** rerunning the workflow yields the same held-out case order and metrics

### SCENARIO-VERIFY-034: Tracker Gating Avoids Harmful False-Positive Escalation

**Given** earlier live traces show that a detected error type is noisy on the
same benchmark or domain
**When** the `tracker_only` strategy reaches a later held-out case with the
same detected error type
**Then** it may suppress the repair escalation for that held-out case
**And** the held-out false-positive count does not increase relative to
  `no_learning`

### SCENARIO-VERIFY-035: Live Memory Reuse Shows Cross-Model Transfer

**Given** mature live memory patterns were learned from non-held-out traces for
one model
**And** a later held-out case from the other model exposes the same failure
shape
**When** the `tracker_plus_memory` strategy replays that held-out case
**Then** the artifact records a memory retrieval hit
**And** any changed decision is attributed to cross-model live support rather
  than same-case memorization
**And** the resulting held-out outcome remains traceable to the live source
  cases that supplied the reused pattern

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
| REQ-VERIFY-027 | Not Started | Implemented | Exp 219 artifact + harness regression tests |
| REQ-VERIFY-028 | Not Started | Implemented | Exp 220 artifact + harness regression tests |
| REQ-VERIFY-029 | Not Started | Implemented | Exp 221 artifact + harness regression tests |
| REQ-VERIFY-030 | Not Started | Implemented | Exp 222 live trace memory tests |
| REQ-VERIFY-031 | Not Started | Implemented | Exp 222 live trace memory tests + artifact refresh |
| REQ-VERIFY-032 | Not Started | Implemented | Exp 222 reliability, retrieval, and policy update tests |
| REQ-VERIFY-033 | Not Started | Implemented | Exp 223 held-out replay tests + artifact refresh |
| REQ-VERIFY-034 | Not Started | Implemented | Exp 223 held-out replay tests + artifact refresh |
| REQ-VERIFY-035 | Not Started | Implemented | Exp 223 held-out replay tests + artifact refresh |
| REQ-JEPA-002 | Not Started | Implemented | 8 Python |
