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

### REQ-JEPA-003: Apple Adversarial Retrain with Energy Features and Conformal Calibration

`experiment_291_jepa_apple_retrain.py` shall retrain the `PredictiveVerifier` gate on Apple adversarial GPU data (Exp 282/283) with the following requirements:

- **Feature extraction**: For each logit array (real or synthetic fallback), extract per-prefix-fraction (25/50/75/100%) features: `mean_spilled`, `max_spilled`, `p95_spilled` (via `SpilledEnergyExtractor`), `semantic_energy` (via `SemanticEnergyExtractor`), `mean_logit`, `max_logit`, `variant_type_encoded` (standard=0, number_swap=1, irrelevant=2). Synthetic fallback records shall include `"synthetic_training": true` in their metadata.
- **Isotonic calibration**: Apply EBM-CoT isotonic regression calibration (arXiv 2511.07124) to the trained model's raw scores.
- **Conformal prediction intervals**: Compute conformal intervals at α=0.1 (arXiv 2603.22966) providing a 90% coverage guarantee on TP/FP rate estimates.
- **Evaluation**: Chronological 80/20 train/held-out split. Measure fast-path hit rate (gate predicts safe → skip Ising), TP rate (gate fires on real violations), FP rate (gate fires on correct outputs). Run 50-case A/B comparison (calibrated vs uncalibrated gate).
- **Targets**: fast-path ≥ 30%, TP ≥ 60%, FP ≤ 20%. Must report clearly if targets are not met.
- **ONNX export**: Save retrained model as `results/jepa_predictor_291.onnx` for Exp 292 NPU testing.

### REQ-JEPA-004: JEPA MLP Retrain on Real Apple Adversarial Logits

`experiment_307_jepa_real_training.py` shall retrain a 3-layer MLP JEPA violation predictor directly on real logit arrays from Exps 294/295, replacing synthetic training data:

- **Pair extraction**: `extract_training_pairs(logit_dir, results_json)` reads `logits_294_*.npy` and `logits_295_*.npy` files and the Exp 295 results JSON to produce `(partial_logit_mean, violation_label)` pairs where `violation_label=1` when `violation_detected=True` in the Exp 295 results for that question.
- **Minimum pair count**: Raises `ValueError` if fewer than 50 pairs are found (insufficient data guard).
- **Train/val split**: 80/20 random split; val set is held out and never seen during training.
- **MLP architecture**: Input = mean logit vector (vocab_size-dim); Hidden = 128-dim ReLU; Output = single energy scalar. Loss = BCE(sigmoid(energy), violation_label). Optimizer = Adam, lr=1e-3, 50 epochs.
- **Training metrics**: `train_jepa_on_pairs(pairs, epochs, lr)` returns dict with `train_loss`, `val_loss`, `val_tp`, `val_fp` per epoch.
- **Convergence requirement**: `val_loss` at epoch N < `val_loss` at epoch 1 (model must improve).
- **ONNX export**: Saved to `results/jepa_predictor_307.onnx`, loadable with onnxruntime.
- **Honest fallback**: If logit files are missing, emit a `blocked` artifact listing exact missing paths.
- **Artifact schema**: `experiment=307`, `training_source="real_logits"`, `n_pairs`, `split` description, `val_tp`, `val_fp`, `skip_rate`, `onnx_path`.

### REQ-JEPA-005: JEPA Gate Reduces Ising Invocations

`JepaGate` shall serve as a fast-path filter before full Ising verification in
`VerifyRepairPipeline.verify_with_gate()`, where:

- **Interface**: `JepaGate(onnx_path, threshold=0.5, enabled=True)` — dataclass-style constructor.
- **Lazy ONNX load**: The ONNX Runtime `InferenceSession` is created on the first call to `predict()`, not at construction time (avoids load overhead when gate is disabled).
- **predict(logit_mean)**: Accepts a 1-D numpy array of mean logit values, runs the ONNX session, and returns `sigmoid(raw_output)` as a float in [0, 1]. When `enabled=False`, returns 1.0 unconditionally (always run Ising).
- **should_skip(logit_mean)**: Returns `True` when `predict(logit_mean) < threshold` (energy low — safe to skip Ising); otherwise `False`.
- **to_dict()**: Returns `{"onnx_path": str, "threshold": float, "enabled": bool}` for artifact serialization.
- **verify_with_gate()**: Additive method on `VerifyRepairPipeline` accepting `jepa_gate=None` kwarg. When gate is provided and `should_skip()` is True, returns a `VerificationResult` with `violations=[]`, `gate_decision="skip"`, `gate_energy=<float>`, `ising_skipped=True`. When gate says verify, runs full Ising and tags the result with `gate_decision="verify"`.
- **Skip rate**: The benchmark (Exp 308) must report `skip_rate = n_skipped / n_total` per threshold.
- **TP rate target**: At least one threshold in `[0.3, 0.5, 0.7]` must achieve `skip_rate ≥ 0.30` AND `TP_rate ≥ 0.85` on the 50-question benchmark corpus.

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

### REQ-VERIFY-036: Warm Multi-Model Inference Server Lifecycle

The repository shall provide a warm inference server in
`python/carnot/inference/model_server.py`, where:
- `ModelServer` accepts one or more HuggingFace model identifiers and loads
  them eagerly when the server starts
- the server keeps loaded models warm for repeated inference requests instead
  of re-loading weights for every experiment call
- the default loader path requests `device="cuda"` so warm models stay on GPU
  when CUDA is available, while preserving the existing loader fallback and
  `CARNOT_FORCE_CPU` override behavior
- the server exposes a context-manager API so callers can scope startup and
  shutdown with `with ModelServer([...]) as server: ...`
- the server validates `batch_size` deterministically with a default of `8`
  and rejects values outside `1..16`
- shutting the server down clears loaded model references, stops the worker
  loop, and attempts GPU memory cleanup via `gc.collect()` plus
  `torch.cuda.empty_cache()` when torch is available

### REQ-VERIFY-037: Queued Batched Generation And Health Reporting

The same module shall batch queued generation requests, where:
- requests are accepted through an internal queue serviced by a dedicated
  worker thread or process
- compatible requests for the same loaded model are coalesced into a single
  forward pass up to the configured batch-size limit
- the default batching path applies each prompt's chat template, tokenizes the
  resulting prompt list with padding, and issues one `model.generate(...)`
  call for each executed batch instead of looping prompt-by-prompt
- `generate_batch()` returns one response per input question in the original
  question order even when multiple callers are batched together
- the server records batch metrics including total requests served, total
  batches executed, average batch size, and max observed batch size
- `health_check()` reports loaded models, queue depth, batch metrics, and a
  GPU-memory snapshot when torch CUDA APIs are available

### REQ-VERIFY-038: Model Loader Integration And Warm-Server Benchmarking

`python/carnot/inference/model_loader.py` shall integrate with the warm server,
where:
- callers can register or clear a running `ModelServer` instance without
  breaking the existing `load_model()` / `generate()` API
- when a compatible server is registered, `load_model()` returns a lightweight
  server-backed handle instead of loading a fresh HuggingFace model
- `generate()` detects the server-backed handle and routes generation through
  the running server while preserving the existing caller contract
- `python/carnot/inference/model_server.py` provides a deterministic benchmark
  helper for `50` questions that compares repeated cold-load generation
  against warm-server generation and reports elapsed times plus speedup

### SCENARIO-VERIFY-036: Batched Requests Preserve Per-Question Results

**Given** a running `ModelServer` with a loaded model and `batch_size=8`
**And** multiple queued questions targeting the same model
**When** the worker coalesces those questions into one forward pass
**Then** each caller receives exactly one response for each submitted question
**And** the responses remain aligned with the original per-question order
**And** the default batching helper issues one padded batched
  `model.generate(...)` call for that executed batch
**And** the server records one executed batch with the observed batch size

### SCENARIO-VERIFY-037: Graceful Shutdown Releases Warm Resources

**Given** a running `ModelServer` has loaded one or more models
**When** the caller exits the context manager or calls `shutdown()`
**Then** the worker loop stops accepting new work
**And** loaded model references are cleared
**And** the server attempts garbage collection and GPU cache cleanup without
  raising when CUDA APIs are unavailable

### SCENARIO-VERIFY-038: Registered Server Avoids Repeated Cold Loads

**Given** a running `ModelServer` is registered with
`carnot.inference.model_loader`
**And** a benchmark compares `50` repeated question generations using
per-call cold loads versus the registered warm server
**When** the benchmark runs with deterministic injected clocks or backends
**Then** the reported cold and warm elapsed times are reproducible
**And** the benchmark reports a speedup ratio of
  `cold_elapsed_seconds / warm_elapsed_seconds`
**And** `load_model()` returns a server-backed handle instead of loading the
  same model again

### REQ-VERIFY-039: Optional TensorRT-LLM Backend With Cached Engines

The repository shall provide an optional TensorRT-LLM backend in
`python/carnot/inference/tensorrt_backend.py`, where:
- `TRTLLMBackend` accepts a HuggingFace model identifier plus backend options
  and exposes deterministic single-prompt and batched generation methods
  compatible with the existing HuggingFace generate contract
- the backend keeps TensorRT engines in an on-disk cache keyed by model name,
  quantization mode, and build parameters so repeated loads reuse an existing
  engine instead of rebuilding it
- before attempting a build, the backend checks the engine cache for a
  compatible engine and loads that engine directly when present
- when a cached engine is absent, the backend can build a TensorRT engine from
  the HuggingFace checkpoint and persist metadata describing the cached build
- the backend supports `fp16` and `int8` quantization modes
- if `tensorrt_llm` is unavailable, CUDA execution is not usable, or the build
  or engine load fails, the backend reports the failure in a structured way so
  callers can fall back to the existing HuggingFace path without breaking the
  public inference API

### REQ-VERIFY-040: Warm Server Preference And TensorRT Benchmarking

The warm inference path shall prefer the TensorRT-LLM backend when possible,
where:
- `python/carnot/inference/model_server.py` prefers a ready TensorRT-LLM
  backend before falling back to the existing HuggingFace loader
- the default preference preserves the existing `CARNOT_FORCE_CPU` override
  and degrades to HuggingFace when TensorRT-LLM is unavailable or unhealthy
- the repository provides a deterministic benchmark helper for `50` questions
  that compares HuggingFace generation against TensorRT-LLM generation and
  reports elapsed times plus speedup when the TensorRT backend is available
- the same benchmark helper returns a structured unavailable/fallback result
  instead of raising when TensorRT-LLM cannot be used in the local environment

### SCENARIO-VERIFY-039: Cached TensorRT Engine Avoids Rebuild

**Given** a compatible TensorRT engine for
`Qwen/Qwen3.5-0.8B` or `google/gemma-4-E4B-it` already exists in the engine
cache for the requested quantization mode
**When** the backend is asked to initialize the same model again
**Then** it loads the cached engine instead of rebuilding from the
  HuggingFace checkpoint
**And** the cached-engine metadata shows the matching model name,
  quantization mode, and build parameters

### SCENARIO-VERIFY-040: TensorRT Failure Falls Back To HuggingFace

**Given** TensorRT-LLM is not installed, CUDA execution is disabled, or the
engine build or load fails for a requested model
**When** the warm inference path tries to initialize that model
**Then** the failure is captured as backend status or metadata rather than an
  uncaught exception
**And** `ModelServer` falls back to the existing HuggingFace loader for that
  model
**And** caller-visible generation still succeeds through the fallback path

### SCENARIO-VERIFY-041: HF Versus TensorRT Benchmark Reports Speedup

**Given** deterministic HuggingFace and TensorRT-LLM backends for the same
  model and a list of `50` benchmark prompts
**When** the benchmark helper runs both backends on those prompts
**Then** the reported HuggingFace and TensorRT elapsed times are reproducible
**And** the speedup is reported as
  `huggingface_elapsed_seconds / tensorrt_elapsed_seconds`
**And** if TensorRT-LLM is unavailable, the result is marked unavailable with
  a fallback reason instead of raising

### REQ-VERIFY-041: Dual-GPU Paired Benchmark Execution

The repository shall provide a dual-GPU paired benchmark helper in
`python/carnot/inference/dual_gpu.py`, where:
- `DualGPURunner` accepts exactly two model specs that preserve the existing
  Exp 218 model ordering and expose per-model task execution through a single
  runner API
- when at least two CUDA devices are available and both models are estimated
  below `7B` parameters, the runner loads the first model onto `cuda:0` and
  the second model onto `cuda:1`
- the runner executes the two per-model benchmark tasks in parallel via
  threads and returns one ordered result per model together with
  device-assignment metadata and elapsed time
- when a model is estimated at `7B` parameters or larger, that model is
  loaded with `device_map="auto"` and the runner falls back to sequential
  execution rather than attempting overlapping cross-GPU loads
- `python/carnot/inference/model_loader.py` accepts explicit CUDA device
  strings such as `cuda:0` and `cuda:1`, and also accepts
  `device_map="auto"` without breaking the existing default CPU/CUDA behavior
- `scripts/experiment_218_live_dual_model_suite.py` adds a `--parallel` flag
  that opts into `DualGPURunner` when two GPUs are present and otherwise
  preserves the existing sequential harness path and deterministic artifact
  ordering

### SCENARIO-VERIFY-042: Exp 218 Parallel Mode Preserves Paired Results

**Given** the Exp 218 live dual-model harness is run with `--parallel`
**And** two CUDA devices are available
**And** both target models are estimated below `7B` parameters
**When** the harness executes one benchmark cohort
**Then** the first model is assigned to `cuda:0`
**And** the second model is assigned to `cuda:1`
**And** the two per-model benchmark tasks run through the shared
  `DualGPURunner`
**And** the resulting paired runs and summary statistics remain ordered by
  model and mode as they would in the sequential harness
**And** if one of the requested models is estimated at `7B` parameters or
  larger, that model is loaded with `device_map="auto"` instead of a fixed
  single-GPU assignment

### REQ-VERIFY-042: Exp 232 Semantic Calibration Corpus

The repository shall provide a deterministic Exp 232 workflow that writes
`data/research/semantic_calibration_corpus_232.jsonl`, where:
- the corpus is built primarily from checked-in verify-only live artifacts
  `results/experiment_219_results.json` and
  `results/experiment_221_results.json`
- the live portion preserves true-positive, false-positive, false-negative,
  and true-negative rows from the semantic artifact, and preserves the
  available true-positive and true-negative rows from the prompt-side artifact
- targeted follow-up rows are added only where the checked-in prompt-side
  artifact lacks outcome coverage needed for calibration analysis, rather than
  replacing the live rows with synthetic-only examples
- every record includes prompt and response text, gold error label,
  verifier-detected label, outcome bucket, violation-family labeling,
  answer-target alignment, premise coverage, claim granularity, repairability
  hint, threshold-sweep fields, and provenance back to the source artifact or
  the follow-up gap it fills
- the threshold-sweep fields include at least one deterministic numeric score
  plus the raw signal features needed for later precision-recall analysis
- re-running the workflow refreshes the JSONL artifact deterministically
  without duplicate rows or order drift

### REQ-VERIFY-043: Exp 232 Calibration Summary And Coverage Checks

The same workflow shall write `results/experiment_232_results.json`, where:
- the artifact records the fixed Exp 232 run date `20260413`
- the summary reports counts by source type, source artifact, benchmark,
  task slice, model, outcome bucket, and violation family
- the summary reports the live-versus-follow-up breakdown and the score-range
  metadata needed to sanity-check later threshold sweeps
- the summary records coverage checks confirming that the live semantic slice
  contributed all four confusion buckets and that any prompt-side missing
  buckets were filled only by targeted follow-up rows
- re-running the workflow refreshes the summary artifact in place without
  appending duplicate runs

### SCENARIO-VERIFY-043: Live Calibration Rows Preserve Provenance And Scores

**Given** the checked-in Exp 219 and Exp 221 verify-only artifacts
**When** the Exp 232 workflow rebuilds the calibration corpus
**Then** every live row carries stable provenance to the source experiment,
  case identifier, model, and source artifact
**And** every row includes the deterministic score and raw signal features
  needed for threshold sweeps
**And** the confusion-bucket label for each live row matches the checked-in
  artifact verdict versus the checked-in gold outcome

### SCENARIO-VERIFY-044: Prompt-Side Gap Follow-Ups Stay Minimal And Traceable

**Given** the checked-in Exp 221 prompt-side artifact lacks one or more
  confusion buckets needed for calibration
**When** the Exp 232 workflow adds follow-up rows
**Then** the added rows cover only the missing prompt-side buckets
**And** each follow-up row records the gap it fills and the checked-in prompt
  artifact it was derived from
**And** the summary artifact reports the follow-up count separately from the
  live-artifact count
**And** re-running the workflow does not change the follow-up ordering
  or duplicate them

### REQ-VERIFY-044: Exp 233 Output-Policy Refresh Benchmark

The repository shall provide a deterministic Exp 233 workflow that writes
`results/experiment_233_results.json`, where:
- the workflow evaluates exactly `Qwen/Qwen3.5-0.8B` and
  `google/gemma-4-E4B-it`
- the workflow evaluates at least the four response modes
  `free_form_reasoning`, `answer_only_terse`, `minimal_json`, and
  `grammar_gated_json`
- the audited cohort includes a mixed slice covering semantic GSM8K cases,
  prompt-side constraint cases from Exp 211, code-typed-property cases from
  Exp 211, and deterministic spec-grounded cases derived from checked-in repo
  files
- the artifact records per-model, per-task-slice, and per-mode measurements
  for parse success, retry rate, answer quality, exact satisfaction, partial
  satisfaction, claim coverage, token cost, latency, and repair usefulness
- the artifact preserves the fixed Exp 233 run date `20260413` and the exact
  cohort composition used for the audit
- re-running the workflow refreshes the artifact in place deterministically
  without duplicate rows or order drift

### REQ-VERIFY-045: Exp 233 Minimal-Schema Output Policy And Routing

The same workflow shall write a refreshed machine-readable policy artifact,
where:
- the policy is derived from the measured Exp 233 benchmark metrics rather
  than hard-coded preferences
- the policy stays task-gated and prefers the smallest response contract that
  still preserves verifier-visible evidence for that task slice
- the policy may recommend `minimal_json` or `grammar_gated_json` only for
  task slices where the measured evidence or parse trade-off justifies JSON
- task slices where Exp 213 already showed structured output hurts remain able
  to stay on `answer_only_terse` unless Exp 233 evidence materially changes
  that conclusion
- later experiments and pipeline helpers can consume the refreshed policy
  directly for response-mode routing without requiring manual artifact edits

### SCENARIO-VERIFY-045: Routing Prefers Minimal JSON Only On Supported Slices

**Given** an Exp 233 policy artifact with per-task-slice recommendations
**When** a later caller asks for the recommended response mode for a live
  semantic, prompt-side, code, or spec-grounded task slice
**Then** the caller receives the Exp 233 recommended mode for that slice
**And** unsupported or unknown slices fall back deterministically to
  `answer_only_terse`
**And** JSON-oriented modes are not forced onto code or surface-only slices
  when the policy recommends a non-JSON alternative

### SCENARIO-VERIFY-046: Exp 233 Artifact Regeneration Is Deterministic

**Given** a fixed benchmark cohort and deterministic mocked live responses
**When** the Exp 233 workflow is run twice against the same temporary repo
**Then** it rewrites the same results and policy artifacts in place
**And** the recorded run date remains `20260413`
**And** the ordered response rows and derived summary metrics remain stable
**And** the policy artifact stays machine-readable for later consumers

### REQ-VERIFY-046: Claim-Isolated Semantic Verification With Calibrated Confidence

The repository shall provide a second-generation semantic verifier in
`python/carnot/pipeline/semantic_verifier_v2.py`, where:
- the verifier reuses typed reasoning extraction when it is available and
  falls back to the existing deterministic semantic-grounding claim
  decomposition when it is not
- the verifier isolates answer-bearing claims rather than collapsing the full
  response into one coarse response-level verdict
- each isolated claim records answer-target coverage, premise-support
  coverage, monitorability evidence, and a calibrated semantic-error
  probability
- the calibration thresholds are derived from the checked-in Exp 232 corpus
  `data/research/semantic_calibration_corpus_232.jsonl`
- the verifier consults the refreshed Exp 233 policy artifact
  `results/output_policy_233.json` so structured evidence can raise
  monitorability confidence only on task slices where the measured policy says
  JSON evidence is actually useful
- the overall semantic verdict is one of `supported`, `violated`, or
  `abstain` rather than forcing every weak-evidence case into a hard yes/no
  label

### REQ-VERIFY-047: Additive Pipeline Hook And Deterministic Serialization For Semantic Verifier V2

The same capability shall expose additive pipeline integration and stable
serialization, where:
- `VerifyRepairPipeline` exposes a dedicated
  `verify_semantic_verifier_v2()` entry point and surfaces the structured v2
  result on `VerificationResult` without breaking current callers that ignore
  the new field
- the pipeline only turns semantic-verifier-v2 findings into failing
  `ConstraintResult` objects when the v2 verdict is `violated`
- when the v2 verdict is `abstain`, legacy semantic-grounding detail remains
  inspectable but the weak-evidence case does not automatically become a live
  semantic false positive
- the structured v2 result supports deterministic `to_dict()` and `to_json()`
  helpers with fixed run-date metadata `20260413`
- later audits and benchmark serializers can consume the v2 result without
  depending on raw prose parsing

### SCENARIO-VERIFY-047: Strong Semantic Mismatch Produces A Calibrated Violation

**Given** a grounded arithmetic-style word problem whose response solves a
  related sub-problem but misses the requested answer target
**And** the response exposes enough claim-level evidence for Carnot to align
  the focus claim and its supporting premises
**When** `semantic_verifier_v2.py` evaluates the question-response pair
**Then** the focus claim is isolated explicitly
**And** the result records high answer-target or premise-miss probability for
  that focus claim
**And** the overall verdict is `violated`
**And** the result converts that finding into pipeline-compatible failing
  `ConstraintResult` objects

### SCENARIO-VERIFY-048: Weak Semantic Evidence Produces Abstain Instead Of A Forced Failure

**Given** a semantic question whose response is a terse bare answer with weak
  claim-level grounding evidence
**When** `semantic_verifier_v2.py` evaluates the pair
**Then** the result records low monitorability confidence
**And** the overall verdict is `abstain`
**And** the pipeline does not convert that weak-evidence case into a failing
  semantic verifier constraint automatically

### SCENARIO-VERIFY-049: Semantic Verifier V2 Serialization And Pipeline Wiring Are Deterministic

**Given** a fixed question-response pair and the checked-in Exp 232 and
  Exp 233 artifacts
**When** the v2 semantic verifier is run twice with the same inputs
**Then** it produces the same deterministic `to_dict()` and `to_json()`
  payloads
**And** `VerifyRepairPipeline` exposes the same structured v2 result through
  its additive entry point and on the final `VerificationResult`
**And** only `violated` v2 results become failing pipeline constraints
  automatically

### REQ-VERIFY-048: Exp 235 Live GSM8K Semantic V2 Artifact

The repository shall provide a live Exp 235 workflow in
`scripts/experiment_235_gsm8k_semantic_v2.py`, where:
- the workflow writes `results/experiment_235_results.json`
- the workflow evaluates exactly `Qwen/Qwen3.5-0.8B` and
  `google/gemma-4-E4B-it`
- the workflow reuses the checked-in Exp 219 GSM8K cohort manifest, including
  the ordered case identifiers and shared prompt seeds, so the comparison stays
  paired to the original live benchmark
- the artifact preserves run-date metadata `20260413` and records the refreshed
  Exp 233 policy source plus the semantic-verifier-v2 path used for the run
- each model summary reports baseline, verify-only, and verify-repair accuracy
  together with detected wrong answers, false positives, repair yield, and
  calibrated semantic confidence summaries
- each verify-only and verify-repair case preserves the raw response plus
  verifier-visible semantic traces, including the structured semantic-verifier-v2
  result, per-case false-positive labeling, and repair history when repairs occur
- the workflow uses checkpointed execution so resumed runs do not duplicate
  already completed case results or scramble the paired cohort order

### REQ-VERIFY-049: Exp 235 Direct Comparison Against Exp 219

The same workflow shall write an explicit comparison block against
`results/experiment_219_results.json`, where:
- the comparison records whether the Exp 235 cohort exactly matches the checked-in
  Exp 219 cohort
- the comparison reports per-model deltas for verify-only and verify-repair
  accuracy, false positives, wrong-answer detections, repair yield, and
  calibrated semantic confidence summaries relative to Exp 219
- the comparison makes an explicit machine-readable judgment about whether the
  false-positive budget improved enough to justify the new verify-only path for
  each model
- the comparison records any honest blockers or missing live cells rather than
  fabricating completed results

### REQ-VERIFY-050: Additive Case-Memory Schema And Deterministic Normalization

The repository shall provide a case-based memory module in
`python/carnot/pipeline/case_memory.py`, where:
- the module defines a reusable case schema that can represent semantic
  verification traces, prompt-side constraint traces, and code-verification
  traces without collapsing them into one vague domain-wide pattern bucket
- each normalized case records model name, benchmark slice, violation types and
  families, a deterministic prompt sketch, any normalized property names,
  repair outcome, confidence, and explicit provenance
- prompt sketches are derived with a deterministic text-normalization path that
  stays cheap enough for CPU-side inference-speed use and can fall back to
  verifier trace text when the original prompt is not available
- repair outcome normalization distinguishes at least `improved`,
  `regressed`, `unchanged_success`, `unchanged_failure`, and `unknown`
- cases expose deterministic serialization helpers so the normalized schema can
  round-trip without losing provenance or confidence fields

### REQ-VERIFY-051: Cheap Case Retrieval And Additive Replay Integration

The same module and replay path shall support additive case retrieval, where:
- retrieval keys combine model name, benchmark slice, violation family, prompt
  sketch, property names, and desired repair outcome rather than only coarse
  domain-wide pattern names
- retrieval ranking is deterministic, explainable, and CPU-cheap, using
  exact-field matches and normalized token overlap rather than embeddings or
  stochastic search
- retrieval results preserve matched-field explanations, support counts,
  confidence, and source provenance so later policy updates can explain why a
  case was reused
- `python/carnot/pipeline/self_learning_replay.py` can consult the case memory
  additively when the older pattern-memory path has no exact match, without
  breaking the existing pattern-memory decision path or the Exp 222 / Exp 223
  artifact schema
- existing replay callers that ignore the new case-memory data continue to work
  unchanged

### SCENARIO-VERIFY-050: Exp 235 Reuses The Checked-In Exp 219 Cohort Exactly

**Given** the checked-in `results/experiment_219_results.json` cohort manifest
**When** the Exp 235 workflow rebuilds its live benchmark payload
**Then** the Exp 235 cohort reuses the same ordered case identifiers and prompt
  seeds from Exp 219
**And** the artifact records that the comparison stayed on the paired Exp 219
  cohort rather than sampling a fresh GSM8K slice

### SCENARIO-VERIFY-051: Exp 235 Comparison Summary States Whether False Positives Improved Enough

**Given** Exp 235 model summaries and the checked-in Exp 219 model summaries
**When** the comparison block is built
**Then** the artifact reports the per-model false-positive and accuracy deltas
  directly against Exp 219
**And** the artifact states whether the new verify-only path is justified for
  each model under the observed false-positive budget
**And** any missing run cells are recorded as blockers instead of being
  represented as successful live results

### SCENARIO-VERIFY-052: Semantic And Code Traces Normalize Into Specific Cases

**Given** one semantic verification trace and one code-verification trace
**When** the case-memory module normalizes them
**Then** each case preserves its benchmark slice, violation families, prompt
  sketch, repair outcome, confidence, and provenance
**And** the code trace records any normalized property names while the semantic
  trace does not collapse into the same case key

### SCENARIO-VERIFY-053: Retrieval Prefers Specific Prompt And Property Matches

**Given** multiple stored cases from the same benchmark slice
**When** a query matches one case on model, violation family, prompt sketch,
  and property names but only matches another case on the coarse slice
**Then** retrieval ranks the more specific case first
**And** the ranked result reports which fields matched and which source cases
  supported the reuse

### SCENARIO-VERIFY-054: Case Memory Serialization Round-Trips Deterministically

**Given** a case memory containing aggregated support from multiple traces
**When** it is serialized and loaded again
**Then** the stored support counts, confidence values, and provenance records
  round-trip without duplication or order drift
**And** retrieval after the round-trip yields the same ranked case keys

### SCENARIO-VERIFY-055: Replay Falls Back To Case Memory When Pattern Names Diverge

**Given** the held-out replay has prior live support for a code or semantic
  failure shape but the later held-out case uses a different coarse
  `error_type` label
**When** the older pattern-memory lookup misses the exact `error_type`
**Then** the replay path can still reuse a more specific matching case from the
  additive case memory
**And** the existing replay outputs remain backward compatible for callers that
  do not inspect the new case-memory details

### REQ-VERIFY-052: Self-Learning Policy Compiler And Provenance-Bearing Update Schema

The repository shall provide a learned self-learning policy compiler in
`python/carnot/pipeline/self_learning_policy.py`, where:
- accepted repair snippets and high-precision case-memory evidence are compiled
  into deterministic policy updates rather than free-form prose advice
- the compiled update types include at least verifier-threshold overrides,
  property-budget updates, repair-prompt patches, and routing hints
- every compiled update records explicit provenance back to the source repair
  snippet, case-memory entry, or tracker snapshot that justified the update
- compilation remains deterministic for the same case-memory entries, accepted
  repairs, and tracker statistics, including stable update identifiers and
  ordering
- the module exposes deterministic serialization helpers so compiled policy
  updates round-trip without losing evidence, support counts, or provenance

### REQ-VERIFY-053: Machine-Readable Policy Artifact And Additive Runtime Context

The same module shall expose a machine-readable policy artifact plus additive
runtime integration helpers, where:
- the compiled artifact records fixed run-date metadata `20260413`, summary
  counts, and deterministic lists of the compiled update types
- the artifact stays small and machine-readable so later replay or benchmark
  workflows can load it without depending on raw benchmark result schemas
- additive runtime helpers can combine compiled policy updates with the current
  `ConstraintTracker` statistics and `CaseMemory` retrieval results without
  replacing either path
- the additive runtime context reports which tracker signals, case-memory
  matches, and compiled policy updates were active for a given query shape
- callers that ignore the new policy artifact or runtime helpers continue to
  work unchanged

### SCENARIO-VERIFY-056: High-Precision Cases Compile Into Deterministic Policy Updates

**Given** a case memory containing high-confidence semantic and code cases with
  repeated support
**When** the self-learning policy compiler runs over those entries
**Then** it emits deterministic verifier-threshold overrides, property-budget
  updates, and routing hints keyed to the specific model and benchmark slice
**And** each compiled update records the support, confidence, and source case
  provenance that justified it

### SCENARIO-VERIFY-057: Accepted Repairs Become Reusable Prompt Patches

**Given** accepted repair snippets derived from repeated live repair histories
**When** the self-learning policy compiler consolidates them
**Then** it emits deterministic repair-prompt patches with support counts,
  observed success rates, and provenance back to the source repair histories
**And** the resulting patch artifact remains machine-readable for later replay
  work

### SCENARIO-VERIFY-058: Policy Artifact Serialization Round-Trips Deterministically

**Given** a compiled self-learning policy containing all supported update types
**When** it is serialized and loaded again
**Then** the run-date metadata, update ordering, support counts, and provenance
  records round-trip without duplication or order drift
**And** later runtime lookups over the restored artifact yield the same matched
  policy updates

### SCENARIO-VERIFY-059: Runtime Policy Context Stays Additive To Tracker And Memory

**Given** an existing `ConstraintTracker`, additive `CaseMemory`, and a compiled
  self-learning policy artifact
**When** runtime context is requested for a new query shape
**Then** the returned context preserves the current tracker statistics and
  case-memory matches
**And** it adds any matching threshold overrides, property-budget updates,
  repair-prompt patches, and routing hints without replacing the original paths
**And** callers can explain which sources drove the final policy context

### REQ-VERIFY-054: Exp 241 Chronological Replay V2 Over Semantic And Code Artifacts

The repository shall provide an Exp 241 replay workflow for continuous
self-learning in `python/carnot/pipeline/self_learning_replay.py` plus
`scripts/experiment_241_self_learning_replay_v2.py`, where:
- the workflow replays checked-in semantic traces from
  `results/experiment_235_results.json` and checked-in code traces from
  `results/experiment_238_results.json`
- the replay can normalize both the paired-run Exp 235 schema and the
  model-run Exp 238 schema into a shared chronological replay case format
- the workflow holds out the final chronological slice from each source
  artifact so evaluation cases are never used to update tracker, case-memory,
  accepted-repair, or compiled-policy state
- the replay compares exactly four conditions on the same held-out cases:
  `no_learning`, `tracker_only`, `case_memory`, and
  `case_memory_plus_policy`
- the `case_memory_plus_policy` condition compiles runtime policy updates only
  from prior non-held-out live evidence, using additive case memory plus
  accepted repairs rather than prebuilt full-corpus artifacts that include
  held-out cases
- Exp 223 callers and artifacts remain backward compatible without needing to
  adopt the Exp 241 sources or strategy names

### REQ-VERIFY-055: Exp 241 Artifact Summary, Comparison, And Success Decision

The same workflow shall write `results/experiment_241_results.json`, where:
- the artifact records the fixed Exp 241 run date `20260413`
- the summary reports held-out task success, held-out false positives,
  retrieval hit rate, retrieval precision, and latency overhead for every
  replay condition
- the summary includes a direct machine-readable comparison against
  `results/experiment_223_results.json`, including success-rate and
  false-positive deltas relative to Exp 223's comparable strategies
- the artifact states the primary success condition explicitly as
  `real_held_out_task_gain_with_no_extra_false_positives`
- the artifact records whether that primary success condition was met and, when
  it was not, says so clearly rather than implying success from secondary
  metrics alone
- re-running the workflow refreshes the artifact in place without duplicate
  held-out decisions or order drift

### REQ-VERIFY-056: Exp 244 Formal Claim Corpus From Live Semantic And Prompt-Side Traces

The repository shall provide a deterministic Exp 244 workflow in
`scripts/experiment_244_formal_claim_corpus.py`, where:
- the workflow reads the checked-in live semantic traces from
  `results/experiment_235_results.json` and the checked-in live prompt-side
  traces from `results/experiment_221_results.json`
- the workflow may enrich those live traces with checked-in source metadata
  from `data/research/constraint_ir_benchmark_211.jsonl` and
  `data/research/semantic_failure_corpus_214.jsonl`, but it does not invent a
  fresh synthetic benchmark as the primary data source
- the workflow writes `data/research/formal_claim_corpus_244.jsonl`
- every corpus row records the source prompt, source response, typed claim
  text, normalized relation type, operands or bound variables, target quantity
  or entity, candidate solver route, gold verdict, and provenance back to the
  source run or source artifact
- rows that cannot be normalized safely preserve an explicit
  `formalization_status` such as `abstain` or `not_formalizable` rather than
  pretending every live claim is solver-ready
- when the source trace already reveals a localized failure seed, the row also
  records minimal-correction-subset-style metadata identifying the broken
  premise, claim, or prompt-side constraint ids

### REQ-VERIFY-057: Exp 244 Summary Artifact And Deterministic Regeneration

The same workflow shall write `results/experiment_244_results.json`, where:
- the artifact records the fixed Exp 244 run date `20260413`
- the summary reports total claim rows, counts by candidate solver route,
  counts by `formalization_status`, counts by gold verdict, and a source
  breakdown by contributing artifact family
- the summary reports formalizable-versus-abstain or
  not-formalizable rates clearly enough to show how much of the checked-in
  live trace inventory is already solver-routable
- re-running the workflow refreshes both the JSONL corpus and the summary
  artifact in place without duplicate rows, provenance drift, or order drift

### SCENARIO-VERIFY-060: Final Chronological Slice Stays Honest Across Artifact Shapes

**Given** Exp 235 semantic traces and Exp 238 code traces in their checked-in
  artifact schemas
**When** Exp 241 constructs replay cases and holds out the final chronological
  slice from each source artifact
**Then** the held-out replay mixes both semantic and code traces in one
  evaluation stream
**And** none of those held-out cases contribute tracker, case-memory, accepted
  repair, or compiled-policy updates for later held-out decisions
**And** the older Exp 223 replay artifact contract remains available for
  callers that still use the pre-Exp-241 path

### SCENARIO-VERIFY-061: Policy Updates Can Change Replay Decisions Beyond Raw Retrieval

**Given** non-held-out live cases that produce repeated high-precision
  case-memory support and accepted repair evidence
**When** the Exp 241 replay evaluates the same held-out case under
  `case_memory` and `case_memory_plus_policy`
**Then** the policy-aware branch can make a different decision from raw
  case-memory retrieval alone
**And** the artifact records which compiled routing hints, threshold updates,
  repair patches, or case matches were active for that held-out decision

### SCENARIO-VERIFY-062: Exp 241 Summary States The Honest Outcome Relative To Exp 223

**Given** the Exp 241 replay summary and the checked-in Exp 223 replay artifact
**When** the Exp 241 artifact is written
**Then** it reports retrieval, latency, task-success, and false-positive
  comparisons against Exp 223 in machine-readable form
**And** it states plainly whether the primary success condition
  `real_held_out_task_gain_with_no_extra_false_positives` was met
**And** a gain in secondary metrics alone does not get reported as the primary
  milestone win when held-out task gain is absent

### REQ-VERIFY-058: Solver-Routed Formal Claim Verification

The repository shall provide a formal claim verifier in
`python/carnot/pipeline/formal_claim_verifier.py`, where:
- the verifier accepts a typed claim with at least `claim_id`, `claim_text`,
  `candidate_solver_route`, `formalization_status`, `relation_type`,
  `operands`, and `target` fields and routes it to a deterministic checker
- supported solver routes are `arithmetic`, `comparison`, `cardinality`,
  `set_membership`, and `boolean_entailment`; any other route or any claim
  whose `formalization_status` is not `formalized` receives an explicit
  `abstain` verdict rather than a heuristic guess
- each route executes a deterministic, side-effect-free check using only the
  normalized fields present in the claim; no hidden chain-of-thought or
  external model calls are required
- the verifier returns a structured `FormalClaimVerdict` for each claim
  containing: `claim_id`, `verdict` (`supported`/`violated`/`abstain`),
  `route` (the solver route selected or `abstain`), `failure_detail`
  (machine-readable localization of the specific failed operand, bound, or
  relation when `verdict == violated`), and `run_date`
- the verifier supports batch input producing a `FormalClaimBatchResult` that
  records all per-claim verdicts, aggregate counts by verdict and route, and
  deterministic JSON serialization
- all verdict, route, and failure-detail fields use string literals drawn from
  a fixed closed vocabulary so downstream tools can rely on exact string
  equality without fragile parsing

### REQ-VERIFY-059: Formal Claim Verifier Pipeline Integration

The same module shall integrate additively into `VerifyRepairPipeline`, where:
- `VerifyRepairPipeline` exposes a `verify_formal_claims` entry point that
  accepts a list of typed claim dicts and returns a `FormalClaimBatchResult`
- the entry point does not change the existing `verify()` behavior or break
  any current caller
- `VerificationResult` optionally carries a `formal_claims` field holding the
  `FormalClaimBatchResult` when the entry point was invoked
- the `FormalClaimBatchResult` supports deterministic `to_dict()` / `to_json()`
  serialization with sort-key ordering so two runs on the same input produce
  bit-identical JSON output

### REQ-VERIFY-061: Process-Integrity Verifier For Typed Reasoning And Code-Repair Traces

The repository shall provide a process-integrity verifier in
`python/carnot/pipeline/process_verifier.py`, where:
- the verifier accepts a typed corpus row (reasoning or code-repair) and
  returns a structured `ProcessVerificationResult` that is independent of
  the outcome verdict so callers can detect "correct answer, invalid process"
- supported defect kinds are `unsupported_step`, `missing_premise_jump`,
  `contradictory_intermediate`, `outcome_correct_process_invalid`,
  `repair_regression`, and `repair_stall`; each defect records a kind, a
  human-readable detail string, an optional step reference, and machine-
  readable evidence fields
- unsupported_step is raised when `n_unsupported_claims > 0` in
  `process_evidence`
- missing_premise_jump is raised when `max_premise_support < 1.0` and at
  least one unsupported claim exists
- contradictory_intermediate is raised when `verifier_verdict == "violated"`
  in `process_evidence`
- outcome_correct_process_invalid is raised when `outcome_label == "correct"`
  and `process_label` indicates an invalid process
  (`"right_answer_wrong_process"`)
- repair_regression is raised when the corpus row carries a `repair_context`
  with `prior_outcome == "correct"` and the current `outcome_label ==
  "incorrect"`
- repair_stall is raised when `repair_context.prior_outcome == "incorrect"`
  and `outcome_label == "incorrect"`
- the verifier additionally accepts a `TypedReasoningIR` directly and
  cross-references its claim-to-step grounding to surface unsupported and
  missing-premise defects without requiring pre-computed corpus evidence
- `process_valid` is `True` only when zero defects are detected
- `run_date` is the fixed string `"20260413"` embedded in every result for
  traceability
- `to_dict()` / `to_json()` serialization is deterministic with sorted keys
  so two runs on the same input produce bit-identical output

### REQ-VERIFY-062: Process-Integrity Verifier Pipeline Integration

The process-integrity verifier shall integrate additively into
`VerifyRepairPipeline`, where:
- `VerifyRepairPipeline` exposes a `verify_process_integrity` entry point
  that accepts a corpus row dict and returns a `ProcessVerificationResult`
- the entry point does not change the existing `verify()` behavior or break
  any current caller
- `VerificationResult` can optionally carry a `process_integrity` field
  holding the `ProcessVerificationResult` when the entry point was invoked
- existing verify and verify_and_repair callers that ignore the new field
  continue to work unchanged

### REQ-VERIFY-060: Constraint Addition From Memory Patterns

The repository shall provide a constraint-addition compiler in
`python/carnot/pipeline/constraint_addition.py`, where:
- the compiler accepts a `CaseMemory` instance and produces a
  `ConstraintAdditionResult` that is independent of any existing policy
  artifact — it adds new constraint templates rather than replacing
  threshold or routing decisions already compiled by `SelfLearningPolicy`
- the compiler qualifies entries by minimum support (`min_support`, default 3)
  and minimum confidence (`min_confidence`, default 0.85) so only recurring,
  high-confidence failure families generate templates
- each `ConstraintTemplate` carries a `kind` drawn from three CPU-cheap
  varieties: `text_pattern_guard` (substring presence/absence checks),
  `budget_addition` (additional verifier passes for the failure family), and
  `verifier_guard_clause` (a named gate the pipeline can check before invoking
  the heavier verifier)
- every `ConstraintTemplate` is provenance-bearing: it records the originating
  case entry fingerprints, source experiment numbers, failure family, support
  and confidence at compile time, and the fixed compile date `"20260413"`
- the `ConstraintAdditionResult` serializes deterministically via `to_dict()`
  and `from_dict()` such that two calls on the same `CaseMemory` produce
  byte-identical JSON output when serialized with `json.dumps(..., sort_keys=True)`
- a `ConstraintAdditionRegistry` class wraps a `ConstraintAdditionResult` and
  exposes `lookup(model_name, benchmark_slice, failure_family)` returning the
  sorted tuple of matching templates, and `apply(model_name, benchmark_slice,
  violation_types, response_text)` returning the subset of templates whose
  guard patterns match `response_text` or whose budget/guard kind applies
- the compiler and registry have no dependency on JAX, PyTorch, or any
  numerical library — they are pure Python dataclass and string operations
  safe to import on any CPU-only deployment
- the module integrates additively: `SelfLearningPolicyCompiler` callers need
  not change, and `CaseMemory` is read-only inside the compiler

### SCENARIO-VERIFY-070: Pattern-To-Constraint Compilation From Mature Failure Family

**Given** a `CaseMemory` containing at least 3 entries for the same violation
  family with support ≥ 3 and confidence ≥ 0.85
**When** `ConstraintAdditionCompiler().compile(case_memory)` is called
**Then** the result contains at least one `ConstraintTemplate` whose
  `failure_family` matches the recurring family
**And** the template `kind` is one of `text_pattern_guard`,
  `budget_addition`, or `verifier_guard_clause`
**And** `template.support >= 3` and `template.confidence >= 0.85`

### SCENARIO-VERIFY-071: Every Template Is Provenance-Bearing

**Given** a compiled `ConstraintAdditionResult` with at least one template
**When** any `template.provenance` is inspected
**Then** each `ConstraintProvenance` record carries non-empty
  `source_case_ids`, a non-empty `failure_family`, `support > 0`,
  `confidence > 0.0`, and `compiled_date == "20260413"`

### SCENARIO-VERIFY-072: Serialization Is Deterministic Across Two Calls

**Given** a `CaseMemory` with qualifying entries
**When** the compiler runs twice and both results are serialized with
  `json.dumps(result.to_dict(), sort_keys=True)`
**Then** both serialized strings are byte-identical

### SCENARIO-VERIFY-073: Additive Integration Does Not Modify Case Memory Or Policy

**Given** a populated `CaseMemory` and a compiled `SelfLearningPolicy`
**When** `ConstraintAdditionCompiler().compile(case_memory)` is called
**Then** `len(case_memory)` is unchanged
**And** the returned `ConstraintAdditionResult` contains no `ThresholdOverride`
  or `RoutingHint` objects — it only contains `ConstraintTemplate` objects

### SCENARIO-VERIFY-074: Registry Lookup Returns Only Matching Templates

**Given** a `ConstraintAdditionRegistry` loaded from a compiled result
  containing templates for two different failure families
**When** `registry.lookup(model_name, benchmark_slice, "semantic")` is called
**Then** only templates whose `failure_family == "semantic"` are returned
**And** the returned tuple is sorted by `template_id`

### REQ-VERIFY-063: Apple Adversarial GSM8K Dataset Generation

The repository shall provide a deterministic adversarial dataset generator
`scripts/experiment_281_apple_adversarial_dataset.py`, where:
- the generator loads exactly 200 cohort questions from
  `results/experiment_219_results.json` (the live GSM8K semantic benchmark cohort)
- for each cohort question two adversarial variants are generated:
  - `number_swap`: numeric operands in the original question text are replaced
    with different values drawn from a seeded RNG (seed base `281_000 + i`),
    producing a question with the same logical structure but a different correct
    answer; the swap must change at least one number
  - `irrelevant_sentence`: one contextually plausible but mathematically
    irrelevant sentence is inserted at a random position in the original
    question; the correct answer is unchanged
- each output row carries the fields: `question_id`, `original_question`,
  `original_answer`, `variant_type`, `variant_question`, `variant_answer`,
  `provenance`
- the generator writes 400 rows (200 questions × 2 variants) to
  `data/research/gsm8k_adversarial_281.jsonl` (one JSON object per line)
- results metadata is written to `results/experiment_281_results.json`
  including row count, seed, variant-type counts, and a coverage summary
- no live model inference is performed — the module is pure dataset generation
- seeds are drawn from the `281_000+` range to avoid collision with Exp 119
  seeds (`119`) and Exp 279 seeds (`279_000+`)
- the generator is reproducible: running it twice with the same inputs produces
  byte-identical output files

### SCENARIO-VERIFY-078: Number-Swap Variant Changes The Answer

**Given** a cohort question with a known ground-truth answer
**When** the `number_swap` variant is generated with seed `281_000 + i`
**Then** at least one numeric value in `variant_question` differs from
  `original_question`
**And** `variant_answer != original_answer` for the majority of cohort rows
  (the swap produces a genuinely different arithmetic result in most cases)

### SCENARIO-VERIFY-079: Irrelevant-Sentence Variant Preserves The Answer

**Given** a cohort question with a known ground-truth answer
**When** the `irrelevant_sentence` variant is generated
**Then** the `variant_answer` equals `original_answer` for every row
**And** `variant_question` contains at least one sentence not present in
  `original_question`
**And** the inserted sentence contains a numeric value that does not appear in
  the original question (to maximize distractor potential)

### REQ-VERIFY-064: Apple Adversarial GSM8K GPU Baseline Inference With Logit Saving

The repository shall provide `scripts/experiment_282_apple_baseline_gpu.py` that runs
baseline (no-verification) inference on the Exp 281 adversarial corpus, where:
- two models (Qwen3.5-0.8B on GPU 0, Gemma4-E4B-it on GPU 1) are dispatched via DualGPURunner
- three question variants are evaluated per model: `standard`, `number_swap`, `irrelevant_sentence`
- logit tensors are saved at 25/50/75/100% prefix fractions as `.npy` files under `data/research/`
- checkpoints are written every 10 questions (REQ-VERIFY-065)
- a 60 s hard timeout fires per inference call (REQ-VERIFY-066)
- results are written to `results/experiment_282_results.json` with `ARTIFACT_SCHEMA` fields

### REQ-VERIFY-065: Checkpoint-Resume Support For Long Inference Runs

The repository shall support resuming interrupted inference runs by:
- writing a checkpoint every `CHECKPOINT_INTERVAL` (10) questions to a per-(model, variant) JSON file
- on restart, loading the checkpoint and skipping already-completed question IDs
- never re-calling the model for a question_id that already has a result in the checkpoint

### REQ-VERIFY-066: 60-Second Hard Timeout Per Inference Call With Partial Artifact

The repository shall enforce a 60-second hard timeout per inference call, where:
- if `generate_fn` raises `TimeoutError`, a partial artifact is written with `partial=True`
  and a `stall_at` field identifying `"model_name:variant_type:question_id"`
- a completed run writes `partial=False` and `stall_at=None`

### REQ-VERIFY-067: Logit Tensors Saved At Prefix Fractions For JEPA Training

The repository shall save logit tensors at prefix fractions 25/50/75/100% of each
(model, variant_type) combination as NumPy `.npy` files where each file stores a
1-D object array of per-question `(seq_len, vocab_size)` tensors.

### REQ-VERIFY-068: Verify-Repair 12-Cell Benchmark On Apple Adversarial Corpus

The repository shall provide `scripts/experiment_283_apple_verify_repair.py` that runs
three inference modes on the Exp 281 adversarial corpus, where:
- modes: `baseline` (no verification), `verify_only`, `verify_repair`
- variant types: `number_swap`, `irrelevant_sentence` (both from the adversarial corpus)
- models: Qwen3.5-0.8B (GPU 0), Gemma4-E4B-it (GPU 1) — dispatched via DualGPURunner at startup
- this produces 12 benchmark cells: 3 modes × 2 variant_types × 2 models
- per-question record fields: `mode`, `variant_type`, `model`, `correct`,
  `violation_detected`, `repaired`, `logit_path`, `semantic_grounding_fired`,
  `formal_claim_fired`
- logit tensors are saved at 25/50/75/100% prefix fractions to
  `data/research/logits_283_{model_slug}_{variant}_{pct}pct.npy`
- checkpoints are written every 10 questions; 60 s hard timeout per inference call
- a partial artifact with `stall_at` is emitted on timeout
- `inference_mode` is `"live_gpu"` when `CARNOT_FORCE_LIVE=1`, else `"mock"`

### REQ-VERIFY-069: Verify-Repair Improvement Delta Computation And Primary Criterion

The artifact produced by Exp 283 shall include an `improvement_deltas` field where:
- `delta(mode, variant)` = `accuracy(mode, variant)` − `accuracy(baseline, variant)` for each model
- the primary criterion `larger_improvement_on_number_swap` is `True` when
  `delta(verify_repair, number_swap) > delta(verify_repair, standard)` for at least one model,
  where `standard` accuracy is read from the Exp 282 baseline artifact
- comparison references: Exp 260 (standard GSM8K), Exp 235 (semantic v2 cohort),
  Exp 282 (Apple adversarial baseline)

### REQ-VERIFY-070: Logit Saving Hook For Verify-Repair Pipeline (Exp 291 JEPA Training)

The Exp 283 runner shall capture logits at the initial `baseline` generation step of each
question and save them at the standard prefix fractions so they are compatible with the
Exp 291 JEPA training pipeline.

### REQ-VERIFY-071: Partial Artifact With stall_at On 60-Second Timeout

Identical semantics to REQ-VERIFY-066, applied to the Exp 283 verify-repair runner:
a `TimeoutError` from any inference call (baseline generation or repair generation)
causes a partial artifact to be emitted with `stall_at = "model:mode:variant:question_id"`.

### REQ-VERIFY-072: DualGPU Dispatch At Startup For Exp 283

The Exp 283 runner shall wire DualGPUBenchmarkHarness at object construction time
(before any data loading) so GPU slots are reserved immediately, matching the
Exp 282 pattern (SCENARIO-VERIFY-082).

### SCENARIO-VERIFY-084: 12-Cell Result Structure Contains All Required Fields

**Given** a completed (or partial) Exp 283 run
**When** the artifact JSON is loaded
**Then** `results` contains entries for every combination of (model, mode, variant_type)
**And** each entry has keys `correct`, `total`, `accuracy`, `violation_detected_count`,
  `repaired_count`
**And** `improvement_deltas` contains per-model delta values for each (mode, variant_type) pair

### SCENARIO-VERIFY-085: Verify-Repair Improvement Is Larger On number_swap Than Standard

**Given** baseline and verify_repair accuracy for both number_swap and standard variants
**When** improvement deltas are computed as `accuracy(verify_repair) − accuracy(baseline)`
**Then** `delta(verify_repair, number_swap) > delta(verify_repair, standard)` shall hold
  for at least one model in a successful experiment
**And** the artifact records this as `primary_criterion_met: True`

### SCENARIO-VERIFY-086: DualGPU Dispatch Assigns Qwen GPU 0 Gemma GPU 1

**Given** Exp 283 MODEL_SPECS
**When** the runner is constructed
**Then** Qwen3.5-0.8B is assigned to GPU 0 and Gemma4-E4B-it to GPU 1

### SCENARIO-VERIFY-087: Logit Files Saved At Each Prefix Fraction For Exp 283

**Given** an Exp 283 run with at least 4 questions per (model, variant_type) pair
**When** the run completes
**Then** `.npy` files exist at 25/50/75/100% prefix fractions in `data/research/`
**And** each file stores a 1-D object array of per-question `(seq_len, vocab_size)` tensors

### REQ-VERIFY-073: Apple Adversarial Analysis And Classification (Exp 284)

The repository shall provide `scripts/experiment_284_apple_analysis.py` that:
- loads `results/experiment_282_results.json` (baseline) and
  `results/experiment_283_results.json` (verify-repair) when available
- answers the five key questions defined in SCENARIO-VERIFY-088
- classifies the overall result as one of: `CONFIRMED` (primary criterion met),
  `PARTIAL` (some improvement but primary criterion not met), `RULED_OUT`
  (no improvement on number_swap), or `INCONCLUSIVE` (results files missing,
  stall detected, or insufficient data)
- writes `results/experiment_284_results.json` with the analysis

### REQ-VERIFY-074: compute_delta Function

`compute_delta(baseline_acc, mode_acc)` shall return `mode_acc - baseline_acc` as a
float rounded to four decimal places.  Both inputs must be floats in [0, 1].

### REQ-VERIFY-075: classify_result Function

`classify_result(primary_met, partial_improvement, stall_detected)` shall return:
- `"INCONCLUSIVE"` when `stall_detected` is True or both booleans are False
  with no positive delta present
- `"CONFIRMED"` when `primary_met` is True
- `"PARTIAL"` when `partial_improvement` is True and `primary_met` is False
- `"RULED_OUT"` otherwise

### REQ-VERIFY-076: Spilled Energy Hallucination Detector

The system shall provide a `SpilledEnergyExtractor` in
`python/carnot/pipeline/spilled_energy_extractor.py` that detects hallucinations
from raw LLM logit arrays without any constraint extraction, where:
- `SpilledEnergyResult` dataclass holds `per_token_spilled` (1-D float64 array),
  `mean_spilled`, `max_spilled`, `p95_spilled`, `lookahead_energy` (float),
  `suspected_hallucination` (bool), `threshold_used` (float), and a
  `to_dict()` method that serializes all fields to JSON-compatible types
- `compute_spilled_energy(logits: np.ndarray) -> SpilledEnergyResult` accepts a
  2-D float64 array of shape `(T, V)` (T tokens, V vocab size) and computes
  per-token spilled energy as `H(softmax(logit_t)) − max(log_softmax(logit_t))`
  (entropy minus max log-probability contribution), then aggregates mean, max,
  and 95th-percentile statistics
- `compute_lookahead_energy(logits: np.ndarray) -> float` returns the
  AR-EBM lookahead energy `−mean(log P(token_t | prefix_t))` over all response
  tokens, computed as `−mean(max(log_softmax(logit_t)))` as an approximation
  when ground-truth token indices are unavailable
- `SpilledEnergyExtractor` class exposes `extract_from_array(logits, threshold)`
  and `extract_from_file(path, threshold)` where the file is a `.npy` binary;
  both return `SpilledEnergyResult` with `suspected_hallucination=True` when
  `mean_spilled > threshold`
- `VerifyRepairPipeline.verify_spilled_energy(logits_path)` is an additive
  entry point that accepts either a file path string or a numpy array and
  returns `SpilledEnergyResult`; it does not affect existing `verify()` or
  `verify_and_repair()` callers

### SCENARIO-VERIFY-093: Spilled Energy Detects Peaked Logits

**Given** a logit array where one vocab token dominates with high probability
**When** `compute_spilled_energy(logits)` is called
**Then** `per_token_spilled` values are positive (entropy exceeds max contribution
  only modestly for peaked distributions, but non-zero spill is present)
**And** `mean_spilled` and `max_spilled` are positive floats
**And** `SpilledEnergyResult.to_dict()` round-trips to JSON without error

### SCENARIO-VERIFY-094: Uniform Logits Produce Near-Zero Spill

**Given** a logit array where all vocab entries are equal (uniform distribution)
**When** `compute_spilled_energy(logits)` is called
**Then** `per_token_spilled` values are near zero (entropy ≈ max log-prob magnitude)
**And** `suspected_hallucination` is False when threshold is the default 1.0
**And** `compute_lookahead_energy(logits)` returns `log(vocab_size)` approximately

### SCENARIO-VERIFY-088: Five Key Questions Answered In Exp 284

**Given** Exp 282 and Exp 283 result artifacts (or their absence)
**When** `experiment_284_apple_analysis.py` runs
**Then** the output JSON contains answers to all five key questions:
  1. `apple_drop_replicated` — True iff number_swap caused ≥15 pp accuracy drop vs standard
  2. `verify_repair_delta_larger_on_swap` — True iff Δ(verify_repair, number_swap) > Δ(verify_repair, standard) for at least one model
  3. `irrelevant_sentence_ignored` — True iff irrelevant_sentence accuracy drop < 5 pp vs standard
  4. `extractor_firing_summary` — dict mapping extractor name to fire count across all questions
  5. `dual_model_consistent` — True iff Qwen and Gemma produce the same classification for the overall result

### SCENARIO-VERIFY-089: INCONCLUSIVE Classification When Results Files Missing

**Given** `results/experiment_282_results.json` or `results/experiment_283_results.json`
  does not exist
**When** `experiment_284_apple_analysis.py` runs
**Then** `classification` equals `"INCONCLUSIVE"`
**And** `missing_artifacts` lists the missing file names
**And** docs are NOT updated (no file writes to docs/)

### SCENARIO-VERIFY-090: compare_vs_exp235 Returns Comparison Dict

**Given** `number_swap_acc` (float in [0,1]) and `exp235_gemma_verify_repair_acc=0.475`
**When** `compare_vs_exp235(number_swap_acc, exp235_acc)` is called
**Then** the returned dict contains `delta`, `better_than_exp235` (bool), and
  `exp235_reference_acc` (float)

### SCENARIO-VERIFY-091: compute_delta Rounds To Four Decimal Places

**Given** baseline_acc=0.4650 and mode_acc=0.4750
**When** `compute_delta(0.465, 0.475)` is called
**Then** the result equals `0.01` (rounded to four decimal places)

### SCENARIO-VERIFY-092: Artifact Schema For Exp 284

**Given** a completed Exp 284 analysis run
**When** the artifact JSON is loaded
**Then** it contains: `experiment`, `run_date`, `classification`, `missing_artifacts`,
  `five_questions`, `exp235_comparison`, `exp279_comparison`, and `analysis_notes`

### REQ-VERIFY-077: Semantic Energy Extractor And Dual Energy Gate

The system shall provide a `SemanticEnergyExtractor` and `DualEnergyGate` in
`python/carnot/pipeline/semantic_energy_extractor.py` that detect overconfident-wrong
outputs via the semantic energy signal from arXiv 2508.14496, where:
- `SemanticEnergyResult` dataclass holds `semantic_energy` (float), `temperature` (float),
  `overconfident_flag` (bool), `threshold_used` (float), `per_token_semantic` (1-D float64
  array), and a `to_dict()` method serializing all fields to JSON-compatible types
- `compute_semantic_energy(logits: np.ndarray, T: float = 1.0) -> float` accepts a 2-D
  float64 array of shape `(T, V)` and computes the mean negative log-partition function
  as `mean_t(−log(∑_i exp(logit_t_i / T)))` using the log-sum-exp trick for numerical
  stability; raises `ValueError` for non-2-D input, empty token count, or `T <= 0`
- `SemanticEnergyExtractor` class holds a `threshold` and `temperature`; its `extract(logits)`
  method returns `SemanticEnergyResult` with `overconfident_flag=True` when
  `semantic_energy < threshold`; its `calibrate(logits_corpus, labels)` method fits an
  isotonic regression (decreasing: lower energy → higher P(wrong)) and sets `self.threshold`
  to the crossing energy where P(wrong) drops below 0.5
- `DualEnergyResult` dataclass holds `spilled_result`, `semantic_result`,
  `gate_fired` (bool), `trigger_signal` (Literal["spilled","semantic","both","none"]),
  and `calibration_threshold_used` (float)
- `DualEnergyGate` class exposes `calibrate(logits_corpus, labels)` (delegates to
  its internal `SemanticEnergyExtractor`) and `fire(spilled_result, semantic_result)
  → DualEnergyResult` where `gate_fired = spilled.suspected_hallucination OR
  semantic.overconfident_flag` and `trigger_signal` reflects which signal(s) fired
- `VerifyRepairPipeline.verify_dual_energy(logits)` is an additive entry point accepting
  a numpy array or .npy file path, running both extractors and returning `DualEnergyResult`;
  it does not affect existing `verify()` or `verify_and_repair()` callers

### SCENARIO-VERIFY-095: Semantic Energy Fires On Overconfident Logits

**Given** a logit array where one vocab token dominates with an extreme logit value
(very peaked, very low entropy)
**When** `SemanticEnergyExtractor(threshold=-0.5).extract(logits)` is called
**Then** `overconfident_flag` is True (semantic_energy < −0.5 for highly peaked logits)
**And** `semantic_energy` is a large negative float (high confidence = low partition energy)
**And** higher temperature T produces a less negative (higher) semantic_energy, showing
T controls confidence sensitivity

### SCENARIO-VERIFY-096: DualEnergyGate Fires When Either Signal Triggers

**Given** a `DualEnergyGate` with calibrated spilled and semantic thresholds
**When** `gate.fire(spilled_result, semantic_result)` is called with various combinations
**Then** `gate_fired` is True if EITHER `spilled_result.suspected_hallucination` or
  `semantic_result.overconfident_flag` is True
**And** `trigger_signal` equals `"both"` when both fire, `"spilled"` when only spilled
  fires, `"semantic"` when only semantic fires, and `"none"` when neither fires
**And** `DualEnergyResult.to_dict()` round-trips through JSON without error

### REQ-VERIFY-078: Dual-Energy Gate Benchmark On Apple Adversarial Corpus

The repository shall provide `scripts/experiment_287_dual_energy_benchmark.py` that:
- Loads `data/research/gsm8k_adversarial_281.jsonl` (400 rows: 200 number_swap + 200
  irrelevant_sentence)
- If `data/research/logits_282_*.npy` files are present, loads real GPU logits; otherwise
  generates synthetic logits labelled `synthetic_logits: true`, calibrated to the Exp 219
  accuracy distribution (Qwen 21.5% / Gemma 37.5%)
- Calibrates the `DualEnergyGate` semantic threshold from a held-out calibration corpus
  before applying it to the adversarial benchmark
- Computes `SpilledEnergyResult` and `SemanticEnergyResult` for every response in the
  adversarial corpus
- Runs a threshold sweep (101 steps) recording precision, recall, and F1 for the spilled
  signal, semantic signal, and combined gate at every cut-point
- Computes scikit-learn AUROC for: spilled-only, semantic-only, and dual-energy combined
- **Primary criterion**: dual-energy combined AUROC ≥ 0.65 on the adversarial corpus
- Reports per-variant-type breakdown (number_swap vs. irrelevant_sentence) showing which
  signal fires predominantly on which error class
- Reports signal attribution counts: "spilled", "semantic", "both", "none"
- Gap-filler analysis: at the threshold achieving ≥ 0.65 precision on the adversarial
  corpus, reports what fraction of the 1302 not_formalizable Exp 244 claims would be
  covered by the gate
- Writes `results/experiment_287_results.json` with `primary_criterion_met` bool and all
  computed metrics
- `run_date` is the fixed string `"20260414"` embedded in every result record

### SCENARIO-VERIFY-097: AUROC Computation Is Correct

**Given** a list of binary error labels and a list of continuous scores
**When** `compute_auroc(scores, labels)` is called
**Then** perfect predictions (all errors scored above all non-errors) return AUROC = 1.0
**And** random predictions (all scores identical) return AUROC ≈ 0.5
**And** the function delegates to `sklearn.metrics.roc_auc_score` without data leakage

### SCENARIO-VERIFY-098: Threshold Sweep Monotonicity

**Given** a threshold sweep over 101 steps from min to max score on the adversarial corpus
**When** the threshold increases from 0.0 toward max
**Then** precision is non-decreasing and recall is non-increasing across sweep steps
**And** every step record contains `threshold`, `precision`, `recall`, and `f1` keys
**And** at least one threshold achieves precision ≥ 0.65 for the combined gate

### SCENARIO-VERIFY-099: Per-Variant-Type Signal Attribution

**Given** the dual-energy gate applied to 200 number_swap variants and 200 irrelevant_sentence variants
**When** trigger_signal is recorded for each fired result
**Then** number_swap errors produce a higher fraction of "semantic" or "both" trigger signals
  than irrelevant_sentence errors
**And** irrelevant_sentence errors produce a higher fraction of "spilled" or "both" trigger signals
  than number_swap errors

### SCENARIO-VERIFY-100: Gap-Filler Coverage Analysis

**Given** the 1302 not_formalizable rows in `data/research/formal_claim_corpus_244.jsonl`
  where FCV abstains
**When** the dual-energy gate is applied at the threshold achieving ≥ 0.65 precision on
  the adversarial corpus
**Then** `gap_filler_coverage_fraction` is a float in [0.0, 1.0]
**And** the result reports `n_not_formalizable` = 1302 and `coverage_at_target_precision`
  as the fraction of those rows that the gate would flag

### REQ-VERIFY-079: GPU Pre-Warm Health-Check For Live Inference

The repository shall provide a `model_prewarm(model_name, hf_id, gpu_id, *, health_prompt, timeout_seconds)` function in `scripts/experiment_294_gpu_baseline_apple.py` that:
- Loads the specified model onto `cuda:{gpu_id}` (or CPU fallback) with a configurable timeout
- Runs a single health-check prompt (default `"What is 2+2?"`) and checks that the response is non-empty
- Returns a `PrewarmResult` dataclass with fields: `model_name`, `gpu_id`, `load_time_s` (float), `health_ok` (bool), `stall_root_cause` (str: `"lazy_load_stall"` / `"cuda_oom"` / `"unknown"` / `None`)
- If the load stalls beyond `timeout_seconds` (default 15 s), sets `health_ok=False` and `stall_root_cause="lazy_load_stall"`
- If a `torch.cuda.OutOfMemoryError` is raised during load, sets `stall_root_cause="cuda_oom"`
- If an unknown exception is raised, sets `stall_root_cause="unknown"`
- Records `pre_warm_time_s` and `pre_warm_status` fields in the final artifact

The root cause for the recurring GPU stall in Exps 282/283 is lazy model loading: `AutoModelForCausalLM.from_pretrained()` is called inside the per-question inference closure, so the first question triggers a full model load (several seconds) inside the 60 s inference timeout. By pre-warming both models before the timed benchmark loop, the stall is eliminated.

### SCENARIO-VERIFY-101: Pre-Warm Health-Check Returns True On Fast Mock Load

**Given** a `model_prewarm()` call with a mock load function that returns in <1 s
**When** the health prompt produces a non-empty response
**Then** `PrewarmResult.health_ok` is `True`
**And** `PrewarmResult.stall_root_cause` is `None`
**And** `PrewarmResult.load_time_s` is a non-negative float

### SCENARIO-VERIFY-102: Pre-Warm Health-Check Returns False On Timeout

**Given** a `model_prewarm()` call with a mock load function that sleeps beyond `timeout_seconds`
**When** the timeout fires before the model responds
**Then** `PrewarmResult.health_ok` is `False`
**And** `PrewarmResult.stall_root_cause` is `"lazy_load_stall"`

### REQ-VERIFY-080: Prefill Uncertainty Probe

The system shall provide a `PrefillUncertaintyProbe` in
`python/carnot/pipeline/prefill_uncertainty_probe.py` that detects hallucination
RISK before any tokens are generated, using only the first-pass logit distribution
(black-box friendly — no gradient access required):

- `PrefillUncertaintyResult`: dataclass with fields:
  - `uncertainty_score` (float): entropy-based uncertainty measure over the first-token
    distribution (Shannon entropy normalised to [0, 1] by dividing by log(V))
  - `conjugate_bound` (float): Cauchy-Schwarz upper bound approximated from input norm
    and gradient norm proxies (both derived from logit magnitude statistics)
  - `high_risk` (bool): True when `uncertainty_score > threshold`
  - `threshold_exceeded` (bool): alias for `high_risk` (explicit name for pipeline routing)
  - `n_tokens` (int): number of vocabulary tokens in the logit distribution
  - `computation_method` (str): `"entropy_approximation"` (black-box) or
    `"embedding_variance"` (white-box embedding variance)
- `compute_input_uncertainty(embeddings)`: accepts a 2-D float64 array of shape
  (T, D) (token embeddings), returns the variance of per-token L2 norms as an
  uncertainty proxy (high variance → model "stretched" over diverse representations)
- `compute_conjugate_bound(input_norm, gradient_norm)`: returns
  `input_norm * gradient_norm` (Cauchy-Schwarz factor; the squared inner product
  is bounded by this product of norms)
- `compute_prompt_uncertainty(logits_first_pass, threshold)`: accepts a 1-D or 2-D
  float64 logit array for the first-pass token distribution (shape (V,) or (1, V)),
  computes normalised Shannon entropy of the softmax distribution as the uncertainty
  score, and returns a `PrefillUncertaintyResult`
- `PrefillUncertaintyProbe.probe(logits_first_pass, threshold)` → `PrefillUncertaintyResult`
- `VerifyRepairPipeline.check_prefill_uncertainty(logits_first_pass, threshold=0.5)`
  returns a dict with keys `skip_verification` (bool), `reason` (str),
  and `result` (PrefillUncertaintyResult); when `high_risk` is False it sets
  `skip_verification=True` and `reason="low_uncertainty"` (fast-path skip)

Theoretical basis (arXiv 2603.19562, Neural Uncertainty Principle, Mar 2026):
adversarial vulnerability and hallucination share a geometric origin — input and
loss-gradient are conjugate observables with an irreducible uncertainty bound
(Heisenberg analogue). High normalised entropy of the first-token distribution
correlates with high gradient-norm uncertainty, making it a viable prefill-stage
hallucination gate.

### SCENARIO-VERIFY-103: PrefillUncertaintyProbe Flags High-Entropy Logits As High Risk

**Given** a logit array of shape (1, V) with near-uniform distribution (all logits equal)
**When** `PrefillUncertaintyProbe().probe(logits, threshold=0.5)` is called
**Then** `result.uncertainty_score` is close to 1.0 (maximum entropy → max uncertainty)
**And** `result.high_risk` is True
**And** `result.threshold_exceeded` is True
**And** `result.computation_method` is `"entropy_approximation"`
**And** `VerifyRepairPipeline.check_prefill_uncertainty(logits, threshold=0.5)` returns
  `skip_verification=False` (high risk — do NOT skip)

### SCENARIO-VERIFY-104: PrefillUncertaintyProbe Fast-Path Skip For Low-Entropy Logits

**Given** a logit array of shape (1, V) with a highly peaked distribution (one token dominates)
**When** `PrefillUncertaintyProbe().probe(logits, threshold=0.5)` is called
**Then** `result.uncertainty_score` is close to 0.0 (near-zero entropy → low uncertainty)
**And** `result.high_risk` is False
**And** `result.threshold_exceeded` is False
**And** `VerifyRepairPipeline.check_prefill_uncertainty(logits, threshold=0.5)` returns
  `skip_verification=True` and `reason="low_uncertainty"`
**And** edge cases (empty logits, single-vocab, all-same embeddings) return valid results
  without raising exceptions

### REQ-PRED-001: Predictive Verifier Feature Extraction

The repository shall provide a predictive verifier module in
`python/carnot/pipeline/predictive_verifier.py`, where:
- a `PredictiveFeatures` dataclass holds structured scalar features extracted
  from a partial response: `token_count`, `char_count`, `numeric_density`,
  `operator_density`, `json_parseable`, `n_claims`, `has_final_answer`,
  `domain_code` (0.0/1.0), and `prior_confidence` (float in [0,1])
- `extract_features(partial_response, domain, prior_confidence)` returns a
  `PredictiveFeatures` instance deterministically without calling any LLM or
  external service
- the feature vector exported via `PredictiveFeatures.to_array()` is a 1-D
  NumPy float32 array of fixed length so downstream ONNX or NPU runtimes
  receive a stable input shape
- extraction is purely rule-based and safe to call on any CPU-only deployment
  without JAX, PyTorch, or any numerical library
- `run_date` is the fixed string `"20260413"` embedded in every result record

### REQ-PRED-002: Calibrated Predictive Gate Decision

The same module shall expose a calibrated gate that produces a routing
decision from extracted features, where:
- a `GateDecision` dataclass carries `should_skip` (bool), `confidence`
  (float in [0,1]), `threshold` (float), `route` (one of `"FAST_PATH"` or
  `"FULL"`), `domain_probs` (dict mapping domain name to float), and
  `feature_summary` (dict of feature name to value) plus `run_date`
- `PredictiveVerifier.predict(features, threshold)` returns a `GateDecision`
  using a deterministic NumPy logistic gate whose weights can be updated via
  `calibrate(corpus_rows)` without touching the JAX MLP in `jepa_predictor.py`
- when `confidence < threshold` the gate returns `should_skip=True` and
  `route="FAST_PATH"` (low-risk optimistic assumption)
- when `confidence >= threshold` the gate returns `should_skip=False` and
  `route="FULL"` (take the expensive verification path)
- `GateDecision.to_dict()` and `GateDecision.to_json()` serialize
  deterministically with sorted keys for audit-trail reproducibility

### REQ-PRED-003: ONNX Export Readiness

The same module shall expose an ONNX-compatible export path, where:
- `PredictiveVerifier.export_onnx(path)` writes a self-contained ONNX model
  file whose single input is the feature vector produced by
  `PredictiveFeatures.to_array()` and whose single output is a scalar
  violation-probability float32
- when the `onnx` package is not installed the helper raises `ImportError`
  with a clear installation hint rather than crashing silently
- the gate weights (bias and coefficients) are serializable to / from a
  `.safetensors` file via `save(path)` / `load(path)` so parameters trained
  from a live corpus can be checkpointed and replayed deterministically
- the module itself has no hard dependency on `onnx` at import time; the
  dependency is deferred to the `export_onnx` call so the module remains safe
  to import on every deployment

### REQ-PRED-004: Additive Pipeline Integration

The predictive verifier shall integrate additively into
`VerifyRepairPipeline`, where:
- `VerifyRepairPipeline` can accept a `PredictiveVerifier` instance via its
  `verify()` call or as a drop-in for the existing `jepa_predictor` parameter
  by satisfying the same `predict(embedding) -> dict[str, float]` duck-type
  interface
- passing `predictive_verifier=None` (the default) leaves the existing
  verification behavior completely unchanged
- `VerificationResult` returned through the fast path preserves
  `mode="FAST_PATH"`, `skipped=True`, and includes a `predictive_gate`
  key in the certificate dict carrying the serialized `GateDecision`
- the `PredictiveVerifier` is additive: it does not change the behavior of
  the existing `JEPAViolationPredictor`, the constraint extractor, or any
  other pipeline component

### SCENARIO-PRED-001: Low-Confidence Partial Response Takes Fast Path

**Given** a partial response with very few tokens and no numeric content
**When** `PredictiveVerifier().gate(partial_response, threshold=0.5)` is called
**Then** `decision.confidence` is below 0.5 and `decision.should_skip` is True
**And** `decision.route == "FAST_PATH"`
**And** `decision.feature_summary` contains all extracted feature names

### SCENARIO-PRED-002: High-Confidence Violation Triggers Full Path

**Given** a partial response with dense numeric content and arithmetic operators
**When** `PredictiveVerifier().gate(partial_response, threshold=0.5)` is called
**Then** `decision.route == "FULL"` and `decision.should_skip` is False

### SCENARIO-PRED-003: Gate Decision Serializes Deterministically

**Given** a `GateDecision` with any field values
**When** `to_json()` is called twice
**Then** both calls produce byte-identical JSON output

### SCENARIO-PRED-004: Calibration Updates Gate Without Breaking Defaults

**Given** a list of corpus rows from `predictive_verification_corpus_252.jsonl`
**When** `PredictiveVerifier().calibrate(corpus_rows)` is called
**Then** subsequent calls to `gate()` use the updated weights
**And** the calibration does not change the module's public API

### SCENARIO-JEPA-006: Feature Extraction From Adversarial Logits

**Given** a (T, V) logit array from an Apple adversarial GPU run (or synthetic fallback)
**When** `extract_apple_features(logits, prefix_fraction, variant_type)` is called
**Then** it returns a dict with keys: `mean_spilled`, `max_spilled`, `p95_spilled`,
  `semantic_energy`, `mean_logit`, `max_logit`, `variant_type_encoded`, `prefix_fraction`
**And** `variant_type_encoded` equals 0 for "standard", 1 for "number_swap", 2 for "irrelevant"
**And** synthetic fallback data includes `"synthetic_training": true` in its metadata record

### REQ-VERIFY-081: Confidence-Weighted Violation Scoring

The system shall provide a `ConfidenceVerifier` in
`python/carnot/pipeline/confidence_verifier.py` that converts binary
violated/not-violated flags into continuous confidence scores, where:

- `ViolationConfidence`: dataclass with fields:
  - `constraint_id` (str): identifier matching the source ConstraintResult description
  - `energy_delta` (float): absolute energy contribution from this violation
  - `confidence_score` (float in [0, 1]): sigmoid-normalised confidence that this
    is a real violation, derived from `energy_delta / temperature`
  - `confidence_class` (str): `"HIGH"` when score ≥ 0.8, `"MEDIUM"` when 0.5 ≤ score < 0.8,
    `"LOW"` when score < 0.5
  - `repair_recommended` (bool): True only when `confidence_score ≥ repair_threshold`
  - `evidence` (dict): metadata from the source constraint (e.g. claimed vs. correct result)
- `confidence_from_energy(energy_score, temperature=1.0)`: sigmoid normalisation
  `1 / (1 + exp(-energy_score / temperature))` → [0, 1] confidence value; never
  raises on inf or NaN inputs — clamps to [0, 1]
- `ConfidenceVerifier.verify_with_confidence(response, extractor, threshold=0.8)`:
  runs `extractor.extract(response, "auto")`, computes a `ViolationConfidence` for
  each violated constraint, and returns a list of `ViolationConfidence` objects
- The number of items with `repair_recommended=True` is always ≤ the total number of
  detected violations
- For arithmetic violations, `|operand_sum - claimed_result| > 2 * temperature`
  maps to confidence_class `"HIGH"` (error larger than 2σ is definitively wrong)

Theoretical basis (arXiv 2602.03979, Likelihood-Based Reward Designs):
log-probability (EBM energy score) is a continuous signal for repair decisions,
more informative than binary correct/incorrect, and correlates with error severity.

Spec: REQ-VERIFY-081

### REQ-VERIFY-082: Repair Gate With Confidence Threshold

The `VerifyRepairPipeline` shall expose a confidence-aware repair method where:

- `verify_and_repair_confident(question, response=None, domain=None, threshold=0.8)`:
  runs the standard verify pass, then filters violations to only those with
  `confidence_score ≥ threshold` before entering the repair loop
- When no violations exceed the threshold, returns immediately with
  `repaired=False` even if binary violations exist (high-confidence gate blocks
  false-positive repairs that plagued Exp 184)
- The method is additive: it does NOT change `verify_and_repair()` behaviour

Spec: REQ-VERIFY-082

### SCENARIO-VERIFY-105: HIGH-Confidence Arithmetic Violation Triggers Repair

**Given** a response containing `"47 + 28 = 76"` (off by 1 — unambiguous error)
**When** `ConfidenceVerifier().verify_with_confidence(response, extractor, threshold=0.8)` is called
**Then** exactly one `ViolationConfidence` is returned
**And** `confidence_class == "HIGH"` and `repair_recommended == True`
**And** `confidence_score >= 0.8`

### SCENARIO-VERIFY-106: LOW-Confidence Violation Does Not Trigger Repair

**Given** a response with a tiny arithmetic deviation (energy_delta near zero)
**When** `ConfidenceVerifier().verify_with_confidence(response, extractor, threshold=0.8)` is called
**Then** the returned `ViolationConfidence` has `confidence_class == "LOW"` or `"MEDIUM"`
**And** `repair_recommended == False`

### SCENARIO-VERIFY-107: repair_gate Blocks Repair Below Threshold

**Given** `confidence_score = 0.65` and `threshold = 0.8`
**When** `repair_gate(0.65, threshold=0.8)` is called
**Then** the result is `False` (do not repair)

### SCENARIO-VERIFY-108: verify_and_repair_confident Skips Repair On Low-Confidence Violations

**Given** a pipeline with no model and a response that has binary violations but all
  with `confidence_score < threshold`
**When** `pipeline.verify_and_repair_confident(question, response, threshold=0.8)` is called
**Then** the result has `repaired == False` and `verified == False`
**And** the low-confidence violations were not passed to the repair loop

### SCENARIO-JEPA-007: Calibrated Gate Meets Targets on Held-Out Set

**Given** the PredictiveVerifier retrained on Apple adversarial features (Exp 291)
**When** applied to the chronological 20% held-out set with isotonic calibration
**Then** fast-path hit rate ≥ 30% (gate says "probably fine", skip Ising)
**And** TP rate ≥ 60% (gate fires on real violations)
**And** FP rate ≤ 20% (gate fires on correct outputs)
**And** conformal intervals at α=0.1 are reported for TP and FP rates
**And** the result clearly states TARGETS_MET or TARGETS_NOT_MET
**And** the model is exported to `results/jepa_predictor_291.onnx`

### SCENARIO-JEPA-008: Real Logit Pairs Extracted From Apple Adversarial Files

**Given** `logits_294_*.npy` and `logits_295_*.npy` files exist in `data/research/`
**And** Exp 295 results JSON contains `violation_detected` per question
**When** `extract_training_pairs(logit_dir, results_json)` is called
**Then** each returned pair has a mean logit vector as input and a 0/1 violation label
**And** at least 50 pairs are returned (else `ValueError` is raised)
**And** labels are sourced from Exp 295 `violation_detected` field, not synthetic logic
**And** 80% of pairs are used for training and 20% for validation (held out)

### SCENARIO-JEPA-009: MLP JEPA Predictor Converges and Exports to ONNX

**Given** at least 50 (partial_logit_mean, violation_label) pairs
**When** `train_jepa_on_pairs(pairs, epochs=50, lr=1e-3)` is run
**Then** `val_loss` at epoch 50 < `val_loss` at epoch 1 (model improves)
**And** training metrics dict contains `train_loss`, `val_loss`, `val_tp`, `val_fp` per epoch
**And** the trained model is exported to `results/jepa_predictor_307.onnx`
**And** the ONNX file is loadable with onnxruntime
**And** if logit files are absent the script emits a `blocked` artifact with exact missing paths

### REQ-LEARN-012: Tier 3 Online Threshold Adaptation

The system shall provide a `ThresholdAdapter` class that adjusts the JEPA gate threshold
after each batch of questions based on observed false-positive rate and skip rate, so the
gate self-corrects during a live benchmark without manual tuning:

- `ThresholdAdapter(initial, fp_threshold, min_skip)` — construct with initial threshold,
  FP-rate trigger (e.g. 0.05), and minimum skip-rate target (e.g. 0.10).
- `adapt(fp_rate, skip_rate) → float` — returns a new threshold value:
  - If `fp_rate > fp_threshold`: **increase** threshold by 0.05 (gate is too aggressive).
  - Elif `skip_rate < min_skip`: **decrease** threshold by 0.05 (gate is too conservative).
  - Else: no change.
  - Result is clamped to `[0.1, 0.9]`.
- The `ThresholdAdapter` instance persists across batches and exposes the current threshold
  via `self.threshold`.
- The Tier 3 benchmark records a `threshold_history` list of per-batch threshold values.

Spec: REQ-LEARN-012, SCENARIO-LEARN-019, SCENARIO-LEARN-020

### SCENARIO-LEARN-019: ThresholdAdapter Increases Threshold When FP Rate Exceeds Limit

**Given** a `ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)`
**When** `adapt(fp_rate=0.10, skip_rate=0.30)` is called (FP rate above limit)
**Then** the returned threshold is `0.55` (increased by 0.05)
**And** `adapter.threshold == 0.55`

### SCENARIO-LEARN-020: ThresholdAdapter Decreases Threshold When Skip Rate Below Minimum

**Given** a `ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)`
**When** `adapt(fp_rate=0.02, skip_rate=0.05)` is called (skip rate below minimum, FP OK)
**Then** the returned threshold is `0.45` (decreased by 0.05)
**And** `adapter.threshold == 0.45`

### REQ-LEARN-013: Four-Tier Continuous Self-Learning Relay Benchmark

The system shall provide a four-tier relay benchmark (Exp 318) that runs all four
self-learning tiers in sequence on a single 100-question benchmark, demonstrating
the full continuous self-learning loop:

- **Tier 1:** Per-constraint precision tracker (online weight updates via `ConfidenceVerifier`)
- **Tier 2:** `CaseMemory` → `ConstraintGenerator` (constraint addition from violation patterns)
- **Tier 3:** JEPA fast-path gate (predict violations, skip Ising if low energy)
- **Z3 gate:** `NL2Z3Extractor` → UNSAT triggers Ising; SAT skips

The relay design uses 3 batches of 33 questions each:
- **Batch 1 (warmup):** All tiers passive, collect data, record `batch1_accuracy`
- **Batch 2 (Tier 1+2 active):** Constraint adaptation enabled, record `batch2_accuracy`
- **Batch 3 (all tiers):** Full relay — JepaGate → Z3GatedRepair → Ising if UNSAT,
  record `batch3_accuracy`, `jepa_skip_rate`, `z3_sat_rate`

Primary metric: `improvement_1to3 = batch3_accuracy - batch1_accuracy` (honest signed delta).

Artifact schema: `experiment=318`, `schema="carnot.self_learning_relay.v1"`.

Data structures:
- `RelayBatchResult(batch_id, accuracy, n_questions, tiers_active, constraint_delta)`
  where `tiers_active` is `["tier1"]` / `["tier1","tier2"]` / `["tier1","tier2","tier3","z3"]`
  and `constraint_delta` is `n_constraints_after - n_constraints_before` for that batch.
- `RelayArtifact` containing `batch1`, `batch2`, `batch3`, `improvement_1to2`,
  `improvement_1to3` — all signed floats, never clamped.

Honest reporting invariants:
- Any batch may have negative improvement versus batch1; no fabrication.
- `improvement_1to2` and `improvement_1to3` reflect raw signed deltas.

### SCENARIO-LEARN-021: Relay Batch Tiers Active List

**Given** a `RelayBatchResult` for batch 1 (warmup)
**Then** `tiers_active == ["tier1"]`
**Given** a `RelayBatchResult` for batch 2 (Tier 1+2)
**Then** `tiers_active == ["tier1", "tier2"]`
**Given** a `RelayBatchResult` for batch 3 (all tiers)
**Then** `tiers_active == ["tier1", "tier2", "tier3", "z3"]`

### SCENARIO-LEARN-022: Relay Artifact Honest Signed Improvement

**Given** a `RelayArtifact` where batch3_accuracy < batch1_accuracy
**Then** `improvement_1to3 < 0.0` (negative improvement is reported, not clamped)
**And** `improvement_1to3 == batch3_accuracy - batch1_accuracy` exactly

### SCENARIO-JEPA-010: Gate Below Threshold Skips Ising

**Given** a `JepaGate` with `threshold=0.5` and `enabled=True`
**And** a logit mean vector whose ONNX-predicted energy is 0.3
**When** `verify_with_gate(question, response, domain, jepa_gate=gate)` is called
**Then** the returned `VerificationResult` has `ising_skipped=True`
**And** `gate_decision == "skip"`
**And** `gate_energy == 0.3`
**And** `violations == []`

### SCENARIO-JEPA-011: Gate Above Threshold Runs Full Ising

**Given** a `JepaGate` with `threshold=0.5` and `enabled=True`
**And** a logit mean vector whose ONNX-predicted energy is 0.8
**When** `verify_with_gate(question, response, domain, jepa_gate=gate)` is called
**Then** the returned `VerificationResult` has `ising_skipped=False`
**And** `gate_decision == "verify"`
**And** the full Ising pipeline ran

### SCENARIO-VERIFY-065: Clean Reasoning Trace Produces No Defects

**Given** a corpus row whose `process_label` is `"clean"` and whose
  `process_evidence` has `n_unsupported_claims == 0`,
  `max_premise_support == 1.0`, and `verifier_verdict != "violated"`
**When** the process verifier runs on the row
**Then** `process_valid` is `True` and `defects` is empty

### SCENARIO-VERIFY-066: Unsupported Claim Triggers Defect Detection

**Given** a corpus row with `n_unsupported_claims > 0` in `process_evidence`
**When** the process verifier runs
**Then** at least one defect with `kind == "unsupported_step"` is present and
  `process_valid` is `False`

### SCENARIO-VERIFY-067: Right-Answer Wrong-Process Pattern Is Flagged

**Given** a corpus row with `outcome_label == "correct"` and
  `process_label == "right_answer_wrong_process"`
**When** the process verifier runs
**Then** at least one defect with
  `kind == "outcome_correct_process_invalid"` is present

### SCENARIO-VERIFY-068: Repair Regression Detected In Code-Repair Trace

**Given** a corpus row with `repair_context.prior_outcome == "correct"` and
  `outcome_label == "incorrect"`
**When** the process verifier runs
**Then** a defect with `kind == "repair_regression"` is present

### SCENARIO-VERIFY-069: Process Verification Serializes Deterministically

**Given** a corpus row that produces one or more defects
**When** `to_json()` is called twice on the same `ProcessVerificationResult`
**Then** both calls produce the same byte-for-byte JSON output

### SCENARIO-VERIFY-063: Live Claims Normalize Conservatively Instead Of Guessing

**Given** checked-in live semantic and prompt-side traces from Exp 235 and
  Exp 221
**When** Exp 244 converts those traces into formal-claim rows
**Then** rows that match deterministic arithmetic, comparison, cardinality,
  set-membership, boolean-entailment, or execution-oracle patterns record the
  narrowest candidate solver route that fits the source evidence
**And** rows without a safe deterministic normalization are labeled
  `abstain` or `not_formalizable` instead of receiving an invented route
**And** every row preserves source prompt, response, and provenance back to
  the originating checked-in artifact

### SCENARIO-VERIFY-064: Provenance And Localization Stay Deterministic Across Regeneration

**Given** the checked-in Exp 235 and Exp 221 artifacts plus the checked-in
  Exp 211 and Exp 214 source metadata
**When** Exp 244 is run multiple times with the same inputs
**Then** `data/research/formal_claim_corpus_244.jsonl` and
  `results/experiment_244_results.json` are rewritten deterministically with
  the same row order and summary counts
**And** prompt-side violated constraints and semantic missing-premise signals
  preserve minimal-correction-subset-style localization metadata when the
  source trace exposes it
**And** the summary still reports route counts, formalizable-versus-abstain
  rates, and source-family breakdowns from the regenerated corpus

### REQ-EXTRACT-010: NL2Z3 Chain-of-Thought Constraint Extraction

The system shall provide an `NL2Z3Extractor` that detects internal inconsistencies in
chain-of-thought responses by translating natural-language arithmetic reasoning into Z3
Python assertions and running the SMT solver, where:

- A second LLM call prompts a small auxiliary model to generate self-contained Z3 Python
  code that checks the arithmetic consistency of the response's reasoning steps.
- The generated code is executed in a sandboxed subprocess with a 2-second timeout.
- If Z3 reports `unsat`, the response contains an internally inconsistent reasoning chain
  and a `ConstraintViolation` of type `"z3_unsat"` is returned.
- If the LLM is unavailable (CI environment where `CARNOT_FORCE_LIVE` is not set),
  the extractor returns a `Z3Result` with `sat_status="unknown"` without crashing.
- The extractor implements the `ConstraintExtractor` protocol and integrates with the
  existing `VerifyRepairPipeline`.

Spec: REQ-EXTRACT-010, SCENARIO-EXTRACT-020, SCENARIO-EXTRACT-021

### REQ-EXTRACT-011: Z3 UNSAT Violation Detection

The system shall represent the result of an NL2Z3 check as a `Z3Result` dataclass with:
- `sat_status`: one of `"sat"`, `"unsat"`, `"unknown"`, or `"error"`.
- `z3_code`: the generated Z3 Python code (empty string if LLM unavailable).
- `runtime_ms`: wall-clock time in milliseconds for subprocess execution.
- `violations_found`: `True` if and only if `sat_status == "unsat"`.
- `error_message`: optional string describing any execution error.

A `Z3Result` with `sat_status="unsat"` is the sole trigger for a `z3_unsat`
constraint violation in the pipeline. `sat`, `unknown`, and `error` do not
generate violations.

Spec: REQ-EXTRACT-011, SCENARIO-EXTRACT-022, SCENARIO-EXTRACT-023, SCENARIO-EXTRACT-024

### SCENARIO-EXTRACT-020: NL2Z3 Detects Internally Inconsistent Reasoning

**Given** a chain-of-thought response that contains an arithmetic contradiction
  (for example, "There are 10 apples. We eat 3. There are 8 left.")
**When** `NL2Z3Extractor.extract(question, response)` is called with a live LLM
**Then** the extractor emits Z3 Python code that encodes the contradiction
**And** Z3 returns `unsat`
**And** the returned list contains one `ConstraintViolation` with `type="z3_unsat"`

### SCENARIO-EXTRACT-021: NL2Z3 Returns Empty List for Consistent Reasoning

**Given** a chain-of-thought response whose arithmetic is internally consistent
**When** `NL2Z3Extractor.extract(question, response)` is called
**Then** Z3 returns `sat`
**And** the returned list is empty

### SCENARIO-EXTRACT-022: Z3Result is_violation True Only for UNSAT

**Given** a `Z3Result` with `sat_status="unsat"`
**When** `result.violations_found` is checked
**Then** it is `True`
**And** for `sat_status` values of `"sat"`, `"unknown"`, or `"error"`,
  `violations_found` is `False`

### SCENARIO-EXTRACT-023: run_z3_code Handles Subprocess Timeout

**Given** Z3 Python code that contains an infinite loop or very long computation
**When** `run_z3_code(code, timeout_s=2.0)` is called
**Then** the subprocess is killed after 2 seconds
**And** the returned `Z3Result` has `sat_status="unknown"` and `runtime_ms >= 2000`

### SCENARIO-EXTRACT-024: NL2Z3Extractor Degrades Gracefully Without LLM

**Given** an environment where `CARNOT_FORCE_LIVE` is not set (CI mode)
**When** `NL2Z3Extractor.extract(question, response)` is called
**Then** no LLM call is made
**And** the returned list is empty
**And** the internal `Z3Result` has `sat_status="unknown"` and `z3_code=""`

### REQ-EXTRACT-012: Extractor Head-to-Head Benchmark

The repository shall provide a deterministic benchmark comparing all three
extractor implementations (ArithmeticExtractor, LLMExtractor, NL2Z3Extractor)
on a labeled corpus, where:
- The corpus contains at least 30 responses: at least 15 correct (no errors)
  and at least 15 incorrect (known arithmetic or reasoning errors), labeled
  deterministically so the benchmark is reproducible in CI without GPU access
- For each extractor and each response, the benchmark records:
  `violation_detected` (bool), `runtime_ms`, and `error` (if any)
- **False Positive (FP):** `violation_detected=True` on a CORRECT response
- **True Positive (TP):** `violation_detected=True` on an INCORRECT response
- FP rate = n_fp / n_correct_responses; TP rate = n_tp / n_incorrect_responses
- The winner is the extractor with the lowest FP rate while TP rate > 0;
  if all extractors have TP rate = 0, the extractor with the lowest FP rate wins
- Results are written to `results/experiment_311_extractor_benchmark.json`
  with keys: `experiment`, `extractors`, `fp_tp_table`, `winner`,
  `recommendation`, `inference_mode`
- Any extractor with TP rate = 0 is reported truthfully — never suppressed

Spec: REQ-EXTRACT-012, SCENARIO-EXTRACT-025, SCENARIO-EXTRACT-026

### SCENARIO-EXTRACT-025: Benchmark Correctly Classifies FP and TP

**Given** a corpus of 15 correct and 15 incorrect labeled responses
**And** an extractor that flags 2 correct responses and 5 incorrect responses
**When** `compute_fp_rate(rows)` and `compute_tp_rate(rows)` are called
**Then** `fp_rate = 2/15` and `tp_rate = 5/15`
**And** `BenchmarkResult` records these rates with `n_total = 30`

### SCENARIO-EXTRACT-026: Benchmark Reports Extractor with TP=0 Truthfully

**Given** a benchmark run where ArithmeticExtractor detects zero violations
  on all 15 incorrect responses
**When** the benchmark selects the winner
**Then** ArithmeticExtractor is reported with `tp_rate = 0.0` without suppression
**And** the winner selection prefers any extractor with TP > 0 over one with TP = 0

### REQ-REPAIR-010: Z3-Gated Repair Pipeline

The system shall support a Z3-gated repair pipeline that uses the NL2Z3Extractor as a
cheap first-pass gate before invoking the expensive Ising repair loop, where:
- The gate runs NL2Z3Extractor on every (question, response, domain) triple
- When Z3 returns `sat`, the Ising repair loop is SKIPPED (response is presumed correct)
- When Z3 returns `unsat`, the Ising repair loop is TRIGGERED (contradiction confirmed)
- When Z3 returns `unknown` or `error`, the pipeline falls back to confidence-weighted
  Ising repair (existing path from REQ-VERIFY-082)
- Results are captured in a `Z3GatedRepairResult` dataclass with fields:
  `z3_status`, `z3_code`, `ising_triggered`, `ising_violations`,
  `repair_attempted`, `repaired`, `improvement`, `runtime_ms`
- The `skip_rate` (fraction of questions where Z3 SAT skipped Ising) is computable
  from a collection of `Z3GatedRepairResult` records

Spec: REQ-REPAIR-010, SCENARIO-REPAIR-020, SCENARIO-REPAIR-021

### REQ-REPAIR-011: Z3 SAT Skip Path

The Z3-gated pipeline shall implement a fast-exit path when Z3 returns `sat`, where:
- `ising_triggered` is `False`
- `repair_attempted` is `False`
- `repaired` is `False`
- `improvement` is `0`
- The result is returned immediately without calling ConfidenceVerifier or the LLM

This path is the primary cost-reduction mechanism: Z3 SAT questions consume only the
Z3 gate overhead, not the full Ising + LLM repair cost.

Spec: REQ-REPAIR-011, SCENARIO-REPAIR-022

### SCENARIO-REPAIR-020: Z3 UNSAT Triggers Ising Repair

**Given** a response with internally inconsistent arithmetic
**And** NL2Z3Extractor returns `sat_status="unsat"` for that response
**When** `Z3GatedRepair.repair(question, response, domain)` is called
**Then** `result.z3_status == "unsat"`
**And** `result.ising_triggered is True`
**And** `result.repair_attempted is True`

### SCENARIO-REPAIR-021: Z3 Unknown Falls Back to Confidence-Weighted Ising

**Given** a response where NL2Z3Extractor returns `sat_status="unknown"`
  (e.g., CI mode with no LLM available)
**When** `Z3GatedRepair.repair(question, response, domain)` is called
**Then** `result.z3_status == "unknown"`
**And** `result.ising_triggered is True`
**And** the confidence-weighted ConfidenceVerifier path is invoked as fallback

### SCENARIO-REPAIR-022: Z3 SAT Skips Ising

**Given** a response that NL2Z3Extractor evaluates as consistent (`sat_status="sat"`)
**When** `Z3GatedRepair.repair(question, response, domain)` is called
**Then** `result.z3_status == "sat"`
**And** `result.ising_triggered is False`
**And** `result.repair_attempted is False`
**And** `result.repaired is False`
**And** the result is returned in under 1 ms (no Ising or LLM calls)

### SCENARIO-REPAIR-023: 30-Question Benchmark Reports Honest net_improvement

**Given** a 30-question corpus (at least 10 correct + at least 10 incorrect responses)
**When** `experiment_312_z3_gated_benchmark.py` is run in CI mode (no live LLM)
**Then** the artifact is written to `results/experiment_312_z3_gated_results.json`
**And** the artifact includes `z3_gate_skip_rate`, `ising_trigger_rate`,
  `net_accuracy_improvement` (which may be `0` — reported honestly)
**And** `z3_gate_skip_rate + ising_trigger_rate == 1.0` (every question takes exactly one path)
**And** the artifact schema includes `experiment=312`

### REQ-BENCH-001: Full-Scale Credible Benchmark With 95% Confidence Intervals

The system shall provide a full-scale benchmark script (`experiment_315_fullscale_benchmark.py`)
that measures Carnot verify-repair accuracy against published baselines with statistical rigor:

- GSM8K corpus: 400 questions from Apple adversarial dataset (number_swap, irrelevant_sentence
  variants) plus standard HuggingFace GSM8K; deterministic synthetic fallback when unavailable
- HumanEval corpus: 50 problems; PBT execution-based pass@1 evaluation
- Models: Qwen3.5-0.8B (GPU 0) and Gemma4-E4B-it (GPU 1); DualGPURunner pattern
- Modes: baseline (no verification), verify_only (violations recorded, no repair),
  verify_repair (ConfidenceVerifier threshold=0.8), z3_gated (Z3GatedRepair; skipped if unavailable)
- Confidence intervals: 95% Wilson CI on all accuracy numbers
- Published baselines recorded in artifact: Qwen3.5-0.8B ~25%, Gemma4-E4B-it ~80% on GSM8K
- BatchedInferenceRunner(batch_size=8) per Exp 306 pattern; checkpoint every 50 questions
- Artifact schema: `carnot.fullscale_benchmark.v1` with per_model_results,
  per_variant_results, published_baselines, inference_mode fields
- Results written to `results/experiment_316_fullscale_results.json` (executed in Exp 316)

Spec: REQ-BENCH-001, SCENARIO-BENCH-001, SCENARIO-BENCH-002

### SCENARIO-BENCH-001: Wilson CI Bounds Accuracy Numbers

**Given** a benchmark run with N=400 GSM8K questions and n_correct correct answers
**When** `wilson_interval(n_correct, 400)` is called at 95% confidence
**Then** the result is a tuple `(lower, upper)` with `0 <= lower <= accuracy <= upper <= 1.0`
**And** width `(upper - lower) <= 0.05` for any accuracy between 0.3 and 0.7 at N=400
**And** the interval is narrower than a Wald interval at small n (CI property)

### SCENARIO-BENCH-002: Benchmark Emits Simulated Artifact When GPU Unavailable

**Given** no GPU is available (or `--simulated` flag is set)
**When** `experiment_315_fullscale_benchmark.py` is run with `--n_gsm8k 10 --n_humaneval 5`
**Then** the artifact is written to the output path with `inference_mode="simulated"`
**And** all four modes (baseline, verify_only, verify_repair, z3_gated) are recorded
**And** the artifact contains `per_model_results` for both Qwen3.5-0.8B and Gemma4-E4B-it
**And** all accuracy values have `ci_lower` and `ci_upper` fields
**And** `schema` equals `"carnot.fullscale_benchmark.v1"`

### REQ-INFRA-001: Conductor Timeout Wrapper

The research conductor SHALL be invokable via a wrapper shell script
`scripts/run_experiment_with_timeout.sh` that enforces a configurable hard
timeout on every `claude -p` invocation, where:

- **Hard cap**: Timeout defaults to 45 minutes (RETRO-001 action item carried
  forward from milestones 2026.04.22 and 2026.04.29).
- **Configurability**: The limit is read from the environment variable
  `CARNOT_CONDUCTOR_TIMEOUT_MINUTES` (integer); if unset, defaults to 45.
- **Kill signal**: Uses `timeout -k 60s <N>m "$@"` so that if the grace-kill
  does not work within 60 s a SIGKILL is sent.
- **Exit code**: The script exits with the subprocess exit code unchanged;
  when `timeout` fires, exit code 124 (standard Unix timeout sentinel) is
  propagated and a human-readable message is printed to stderr.
- **No conductor modification**: `scripts/research_conductor.py` MUST NOT be
  modified; the wrapper calls it as a subprocess argument.
- **Rationale**: Exp 308 post-test failure loop consumed 138 minutes; a 45 min
  cap would have saved 93 minutes and prevented the loop from running to the
  50-turn max.

### REQ-INFRA-002: Test-First Stub Generation

`ExperimentTemplate` SHALL provide a `generate_test_stub()` method that writes a
pytest skeleton BEFORE any implementation begins, enforcing test-first
development (NEW-001 action item), where:

- **Signature**: `generate_test_stub(test_file_path: str, module_to_test: str = "") -> str`
- **Idempotent**: If `test_file_path` already exists, log a warning and return
  the path unchanged — never overwrite an existing test file.
- **Content**: The generated skeleton includes:
  - A header comment `# AUTO-GENERATED by ExperimentTemplate.generate_test_stub() — replace me`
  - A traceability comment `# REQ-INFRA-002: test-first enforcement`
  - An import of `module_to_test` when provided (empty string = no import)
  - A class `TestExpNNNPlaceholder` containing one `test_placeholder_stub(self)` that
    asserts True (makes the test-runner green while the real tests are being written)
- **Valid Python**: The skeleton MUST parse without error via `ast.parse()`.
- **Permissions**: Written with mode 0o644.
- **Return value**: The path string to the written (or existing) file.
- **Rationale**: 23.5% post-test failure rate observed in 2026.04.29 milestone
  was driven partly by tests being written after (or skipped during) implementation.

## Scenarios

### SCENARIO-INFRA-001: Conductor Wrapper Times Out Correctly

**Given** `scripts/run_experiment_with_timeout.sh` exists and is executable
**And** `CARNOT_CONDUCTOR_TIMEOUT_MINUTES=1` is set in the environment
**When** the wrapper is invoked with `sleep 120` as the command
**Then** the process is killed after approximately 60 seconds
**And** the wrapper exits with code 124
**And** stderr contains "CONDUCTOR TIMEOUT after 1 min"

### SCENARIO-INFRA-002: Test Stub Is Written Once (Idempotency)

**Given** `ExperimentTemplate(325, "Hardening", "results/exp_325.json")` is constructed
**When** `generate_test_stub("/tmp/test_exp325.py", "scripts.experiment_template")` is called twice
**Then** the file is written on the first call
**And** on the second call the method returns the path unchanged without overwriting the file
**And** the content after both calls is identical to what was written on the first call

### SCENARIO-INFRA-003: Generated Stub Parses as Valid Python

**Given** `ExperimentTemplate(325, "Hardening", "results/exp_325.json")` is constructed
**When** `generate_test_stub("/tmp/test_exp325_new.py", "scripts.experiment_template")` is called
**Then** the file is written to disk
**And** `ast.parse(open(path).read())` succeeds without raising `SyntaxError`
**And** the file contains the string "REQ-INFRA-002"
**And** the file contains a class whose name starts with "TestExp"

### REQ-INFRA-003: GPU Zombie Process Detection

**Status:** Implemented (Exp 326)

At experiment start, the monitoring subsystem must detect GPU processes that hold VRAM but
show zero compute utilisation ("zombie" processes).  These processes were observed in the
2026.04.29 retrospective (PIDs 2592400/2595103, ~1050 MB VRAM each, 0% utilisation) and
caused downstream experiments to fail silently or stall.

**Acceptance criteria:**
- A process is classified as a zombie when `utilization_pct == 0` AND `vram_mb > 100`.
- `DualGPUMonitor.detect_zombies()` returns only zombie processes.
- When `nvidia-smi` is absent (CI environments), `list_gpu_processes()` returns `[]`
  without raising an exception.

### REQ-INFRA-004: Dual-GPU Utilisation Check

**Status:** Implemented (Exp 326)

`ExperimentTemplate.setup_gpu()` must verify that both GPUs are active before timed
inference begins.  Exp 219/221 ran two models sequentially on GPU 0 while GPU 1 sat idle
(195 min vs ~90 min estimated for parallel execution).

**Acceptance criteria:**
- `DualGPUMonitor.check_dual_gpu_health()` returns a dict with keys:
  `n_gpus_detected`, `n_zombies`, `idle_gpus` (list of gpu indices with 0 active processes),
  `all_healthy` (bool).
- `all_healthy` is `True` only when `n_gpus_detected >= 2`, `n_zombies == 0`, and
  `len(idle_gpus) == 0`.
- `ExperimentTemplate.setup_gpu()` adds `gpu_monitor_results` to its returned dict.
  Existing callers are unaffected (additive change only).
- When `CARNOT_FORCE_LIVE=0` (CI mode), `check_dual_gpu_health()` returns a synthetic
  healthy dict without running `nvidia-smi`.

### SCENARIO-INFRA-004: Zombie Detection from nvidia-smi Output

**Given** `nvidia-smi` reports two processes on GPU 0 with 600 MB VRAM each at 0% utilisation
**And** one process on GPU 1 with 50 MB VRAM at 15% utilisation (not a zombie)
**When** `DualGPUMonitor.detect_zombies()` is called
**Then** exactly the two 0%-utilisation, >100 MB processes are returned
**And** the 50 MB / 15% process is excluded

### SCENARIO-INFRA-005: Healthy Dual-GPU Configuration

**Given** two GPUs are detected
**And** each GPU has at least one active process (utilisation > 0 or vram_mb <= 100)
**And** no zombies are present
**When** `DualGPUMonitor.check_dual_gpu_health()` is called
**Then** `all_healthy` is `True`
**And** `idle_gpus` is `[]`
**And** `n_zombies` is `0`

### SCENARIO-INFRA-006: CI-Safe Fallback When nvidia-smi is Absent

**Given** `nvidia-smi` is not installed on the system
**When** `DualGPUMonitor.list_gpu_processes()` is called
**Then** an empty list is returned without raising an exception
**And** `check_dual_gpu_health()` returns `{"all_healthy": False, "n_gpus_detected": 0,
"n_zombies": 0, "idle_gpus": []}`

### REQ-INFRA-005: Pre-Experiment Dependency Audit

**Status:** Implemented (Exp 327)

Before an experiment begins, the infrastructure must verify that all files listed under
"EXISTING CODE TO READ FIRST" in the experiment prompt actually exist on disk.  Missing
files were identified in the 2026.04.29 retrospective (NEW-002) as a root cause of ~5%
wall-time overhead from retry loops — experiments fail mid-run when they try to read
result files from prior experiments that were never completed.

**Acceptance criteria:**
- `extract_required_files(prompt, project_root)` parses lines between
  "EXISTING CODE TO READ FIRST:" and the next blank-line-or-TASK: boundary.
- Each bullet line is stripped of its `- ` prefix and any explanatory comment
  (text after ` — ` or ` # `).
- `{project_root}` placeholder in paths is substituted with the actual repo root.
- Relative paths are resolved to absolute paths using `project_root`.
- `check_dependencies(prompt, project_root)` returns a `DependencyAudit` dataclass with
  `required_files`, `missing_files`, and `all_present` (bool).
- `build_blocked_artifact(audit)` returns a dict including `missing_files` and
  `next_action` advice for the conductor.
- When invoked as a CLI, exits with code 0 (all present) or code 1 (missing files).
- `load_experiment_prompt(yaml_path, exp_id)` loads a prompt from a roadmap YAML by
  matching `exp_id` in the task `id` field.

### SCENARIO-INFRA-007: All Dependencies Present

**Given** a prompt with an "EXISTING CODE TO READ FIRST:" section listing two files
**And** both files exist on disk
**When** `check_dependencies(prompt, project_root)` is called
**Then** `audit.all_present` is `True`
**And** `audit.missing_files` is `[]`
**And** `audit.required_files` has length 2

### SCENARIO-INFRA-008: Missing Dependency Detected Before Experiment Runs

**Given** a prompt listing a file that does not exist (e.g. `results/experiment_999_results.json`)
**When** `check_dependencies(prompt, project_root)` is called
**Then** `audit.all_present` is `False`
**And** the missing path appears in `audit.missing_files`
**And** `build_blocked_artifact(audit)` returns a dict with key `missing_files`
**And** the dict contains a `next_action` field with remediation advice

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
| REQ-VERIFY-036 | Not Started | Implemented | Warm `ModelServer` lifecycle + export tests |
| REQ-VERIFY-037 | Not Started | Implemented | Queued batching, health reporting, and shutdown-path tests |
| REQ-VERIFY-038 | Not Started | Implemented | Model-loader server-handle + deterministic benchmark tests |
| REQ-VERIFY-039 | Not Started | Implemented | TensorRT backend cache/build/fallback tests |
| REQ-VERIFY-040 | Not Started | Implemented | Warm-server preference + HF-vs-TRT benchmark tests |
| REQ-VERIFY-041 | Not Started | Implemented | DualGPU runner + loader/harness dispatch tests |
| REQ-VERIFY-042 | Not Started | Implemented | Exp 232 calibration corpus tests + artifact refresh |
| REQ-VERIFY-043 | Not Started | Implemented | Exp 232 summary/coverage tests + artifact refresh |
| REQ-VERIFY-044 | Not Started | Implemented | Exp 233 artifact + policy refresh tests |
| REQ-VERIFY-045 | Not Started | Implemented | Structured reasoning policy + Exp 233 routing tests |
| REQ-VERIFY-046 | Not Started | Implemented | Semantic verifier v2 claim-isolation tests |
| REQ-VERIFY-047 | Not Started | Implemented | Semantic verifier v2 pipeline + serialization tests |
| REQ-VERIFY-048 | Not Started | Implemented | Exp 235 cohort reuse + semantic-v2 artifact tests |
| REQ-VERIFY-049 | Not Started | Implemented | Exp 235 comparison summary + blocker reporting tests |
| REQ-VERIFY-050 | Not Started | Implemented | Case memory normalization + serialization tests |
| REQ-VERIFY-051 | Not Started | Implemented | Case memory retrieval + replay integration tests |
| REQ-VERIFY-052 | Not Started | Implemented | Self-learning policy compilation + provenance tests |
| REQ-VERIFY-053 | Not Started | Implemented | Self-learning policy serialization + additive runtime-context tests |
| REQ-VERIFY-054 | Not Started | Implemented | Exp 241 replay v2 tests + artifact refresh |
| REQ-VERIFY-055 | Not Started | Implemented | Exp 241 summary/comparison tests + artifact refresh |
| REQ-VERIFY-056 | Not Started | Implemented | Exp 244 formal-claim corpus tests |
| REQ-VERIFY-057 | Not Started | Implemented | Exp 244 summary/provenance tests + artifact refresh |
| REQ-VERIFY-058 | Not Started | Implemented | Formal claim verifier route + abstain + serialization tests |
| REQ-VERIFY-059 | Not Started | Implemented | Formal claim verifier pipeline integration tests |
| REQ-VERIFY-060 | Not Started | Implemented | Constraint addition compilation + provenance + serialization + registry tests |
| REQ-VERIFY-061 | Not Started | Implemented | Process verifier defect detection + serialization tests |
| REQ-VERIFY-062 | Not Started | Implemented | Process verifier pipeline integration tests |
| REQ-VERIFY-063 | Not Started | Implemented | Apple adversarial GSM8K dataset generation tests |
| REQ-VERIFY-064 | Not Started | Implemented | Apple adversarial baseline inference + logit saving tests (Exp 282) |
| REQ-VERIFY-065 | Not Started | Implemented | Checkpoint-resume support tests (Exp 282) |
| REQ-VERIFY-066 | Not Started | Implemented | 60s hard timeout + partial artifact tests (Exp 282) |
| REQ-VERIFY-067 | Not Started | Implemented | Logit fraction saving tests (Exp 282) |
| REQ-VERIFY-068 | Not Started | Implemented | 12-cell verify-repair benchmark tests (Exp 283) |
| REQ-VERIFY-069 | Not Started | Implemented | Improvement delta computation + primary criterion tests (Exp 283) |
| REQ-VERIFY-070 | Not Started | Implemented | Logit saving hook for verify-repair pipeline tests (Exp 283) |
| REQ-VERIFY-071 | Not Started | Implemented | Partial artifact stall_at tests (Exp 283) |
| REQ-VERIFY-072 | Not Started | Implemented | DualGPU dispatch at startup tests (Exp 283) |
| REQ-VERIFY-076 | Not Started | Implemented | Spilled energy extractor + pipeline integration tests |
| REQ-VERIFY-077 | Not Started | Implemented | Semantic energy extractor + DualEnergyGate + pipeline integration tests |
| REQ-VERIFY-078 | Not Started | Implemented | Dual-energy gate benchmark on Apple adversarial corpus (Exp 287) |
| REQ-VERIFY-079 | Not Started | Not Started | GPU pre-warm health-check for live inference (Exp 294) |
| REQ-VERIFY-080 | Not Started | Implemented | PrefillUncertaintyProbe + pipeline integration + 35 Python tests (SCENARIO-103/104) |
| REQ-JEPA-002 | Not Started | Implemented | 8 Python |
| REQ-JEPA-003 | Not Started | Implemented | Exp 291 Apple adversarial retrain + conformal calibration |
| REQ-JEPA-004 | Not Started | Implemented | Exp 307 JEPA MLP real-logit retrain; blocked until logits_294/295 files present |
| REQ-JEPA-005 | Not Started | Implemented | JepaGate fast-path + verify_with_gate() + Exp 308 benchmark |
| REQ-PRED-001 | Not Started | Implemented | Feature extraction + serialization tests |
| REQ-PRED-002 | Not Started | Implemented | Calibrated gate + serialization tests |
| REQ-PRED-003 | Not Started | Implemented | ONNX export helper + safetensors round-trip tests |
| REQ-PRED-004 | Not Started | Implemented | Additive pipeline integration tests |
| REQ-VERIFY-081 | Not Started | Implemented | Confidence-weighted violation scoring + ConfidenceVerifier tests (SCENARIO-105/106/107/108) |
| REQ-VERIFY-082 | Not Started | Implemented | Repair gate with confidence threshold + pipeline integration tests |
| REQ-LEARN-012 | Not Started | Implemented | ThresholdAdapter online adaptation + Tier 3 benchmark (SCENARIO-LEARN-019/020, Exp 309) |
| REQ-LEARN-013 | Not Started | In Progress | Four-tier relay benchmark — RelayBatchResult + RelayArtifact (SCENARIO-LEARN-021/022, Exp 318) |
| REQ-EXTRACT-010 | Not Started | Implemented | NL2Z3Extractor + Z3Result + pipeline integration tests (SCENARIO-EXTRACT-020/021, Exp 310) |
| REQ-EXTRACT-011 | Not Started | Implemented | Z3Result dataclass + is_violation + subprocess timeout tests (SCENARIO-EXTRACT-022/023/024, Exp 310) |
| REQ-EXTRACT-012 | Not Started | Implemented | Extractor benchmark corpus + FP/TP metrics + winner selection (SCENARIO-EXTRACT-025/026, Exp 311) |
| REQ-REPAIR-010 | Not Started | Implemented | Z3GatedRepair + Z3GatedRepairResult + pipeline integration tests (SCENARIO-REPAIR-020/021, Exp 312) |
| REQ-REPAIR-011 | Not Started | Implemented | Z3 SAT fast-exit path tests (SCENARIO-REPAIR-022, Exp 312) |
| REQ-BENCH-001 | Not Started | Script written (Exp 315) | Full-scale benchmark script with 95% Wilson CI; execution in Exp 316 |
| REQ-INFRA-001 | N/A | Implemented | run_experiment_with_timeout.sh + timeout_wrapper_exists() tests (SCENARIO-INFRA-001, Exp 325) |
| REQ-INFRA-002 | N/A | Implemented | generate_test_stub() idempotency + ast.parse tests (SCENARIO-INFRA-002/003, Exp 325) |
| REQ-INFRA-003 | N/A | Implemented | DualGPUMonitor zombie detection + CI-safe fallback tests (SCENARIO-INFRA-004/006, Exp 326) |
| REQ-INFRA-004 | N/A | Implemented | check_dual_gpu_health() + setup_gpu() integration tests (SCENARIO-INFRA-005/006, Exp 326) |
| REQ-INFRA-005 | N/A | Implemented | DependencyAudit + extract_required_files + check_dependencies + CLI tests (SCENARIO-INFRA-007/008, Exp 327) |
