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

### SCENARIO-EXP303-A: Prereq Check Detects Missing Packages With Install Command

**Given** a system where ninja and/or openblas are not installed
**When** `prereq_status()` is called
**Then** the returned dict contains `ninja_installed`, `openblas_installed`,
  `cmake_version`, `cmake_sufficient`, `ryzen_ai_sw_present`, and `vitisai_so_present`
**And** when a package is missing, the corresponding `*_install_command` key
  provides the exact install command for the detected distro (Arch or Debian/Ubuntu)
**And** `all_met` is False when any required package is absent

### SCENARIO-EXP303-B: Source Build Path Attempts Build With Timeout And Log Tail

**Given** all prereqs are available (ninja, openblas, cmake >= 3.26, XRT, VitisAI)
**When** `attempt_ort_source_build(build_dir, timeout_s=600)` is called
**Then** the returned dict contains `success`, `duration_seconds`, and either
  `whl_path` (on success) or `error_summary` + `build_log_tail` + `timeout_exceeded`
  (on failure)
**And** `timeout_exceeded` is True only when the subprocess exceeded the timeout
  (not on compile errors), allowing the researcher to distinguish the two failure modes

### SCENARIO-EXP303-C: Inference Benchmark Reports NPU vs CPU Latency

**Given** a successful ORT build with VitisAI EP available and a valid ONNX model
**When** the inference benchmark runs WARMUP_CALLS=20 + TIMED_CALLS=100
**Then** the artifact records `npu_latency_us`, `cpu_latency_us`, `speedup_factor`,
  `provider_used` (containing "VitisAI"), and `timed_calls >= 100`
**And** `speedup_factor` equals `cpu_latency_us / npu_latency_us` within 5% tolerance

### SCENARIO-EXP303-D: Honest Labeling — Null Inference Result On All Blocked Paths

**Given** any execution path other than `npu_working`
**When** the artifact is written
**Then** `inference_result` is null (not a dict with fabricated latency values)
**And** `build_outcome` is null when `execution_path == "blocked_prereq"`
  (no build was attempted, so no build result should appear)

### SCENARIO-EXP303-E: Exp 335 Still Blocked — Prereq State Unchanged From Exp 314

**Given** Exp 335 runs and ninja and/or openblas are still not installed
**When** `prereq_status()` returns `all_met=False`
**Then** `honest_verdict` is `"blocked_prereq"`
**And** `prereq_changes_vs_exp314` maps each missing package to `"still_missing"`
**And** `blocked_reason` names the exact install command(s) needed
**And** the artifact is written to `results/experiment_335_npu_build.json` with no
  `build_attempt_result` key (no build was attempted)

### SCENARIO-EXP303-F: Exp 335 Build Attempted — Prereqs Now Met

**Given** Exp 335 runs and all required prereqs (ninja, openblas, cmake, XRT, VitisAI)
  are present
**When** `attempt_ort_source_build("/tmp/ort_build_335", timeout_s=600)` is called
**Then** `honest_verdict` is one of `"build_failed"`, `"inference_success"`, or `"timeout"`
  (never `"blocked_prereq"` when prereqs are met)
**And** `prereq_changes_vs_exp314` maps each newly available package to `"now_available"`
**And** `build_attempt_result` is present in the artifact and contains `success`,
  `duration_seconds`, and either `whl_path` (success) or `error_summary` (failure)
**And** `npu_inference_result` is present and non-null only when `honest_verdict ==
  "inference_success"`

### SCENARIO-EXP303-G: IRON Toolchain Check As VitisAI Alternative When Prereqs Missing

**Given** Exp 435 runs and ninja and/or openblas are still not installed
  (VitisAI source build path remains blocked)
**When** `check_iron_toolchain_available()` is called
**Then** the function returns `True` if the `mlir_aie` Python package is importable,
  `False` otherwise — determined without attempting a full build
**And** the artifact includes `iron_path_tested` and `iron_path_succeeded` booleans
**And** if IRON is available, a minimal 16x16 GEMM kernel dispatch is attempted
**And** `honest_verdict` is `"npu_ready_iron_path"` when IRON dispatch succeeds,
  `"blocked_prereq"` when neither VitisAI nor IRON paths are available
**And** `install_commands` in the artifact provides arch_linux and ubuntu commands
  for each missing prerequisite so the human can unblock without re-reading prior artifacts

### REQ-PRED-005: NPU Unblock Status Audit — IRON Toolchain Viability

The NPU unblock status audit (Exp 435) shall:
- Report whether ninja and openblas are installed via subprocess-based checks
  (not assumptions), making the prereq state observable from the artifact alone
- Report whether the IRON toolchain (mlir-aie Python bindings) is available as an
  alternative to VitisAI EP when the source-build prereqs are missing
- Provide an `honest_verdict` field with exactly three possible values:
  `"npu_ready_iron_path"` (IRON dispatch succeeded),
  `"npu_ready_vitisai_path"` (VitisAI prereqs met and build attempted),
  `"blocked_prereq"` (neither path available)
- Never fabricate latency or speedup numbers; set `iron_speedup_vs_cpu` to `null`
  when no successful dispatch occurred
- Include `install_commands` dict with `arch_linux` and `ubuntu` keys listing
  exact shell commands for each missing package so the human can unblock in one step

Spec traces: SCENARIO-EXP303-G (Exp 435)

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

### REQ-VERIFY-083: Expression Specificity Confidence Signal

The system shall provide `compute_expression_confidence(violation_text: str) → float`
in `python/carnot/pipeline/confidence_weighted_repair.py`, where:

- The function returns a float in [0, 1] representing how specifically the violation
  text indicates a real arithmetic error (as opposed to an approximate or intermediate result)
- Exact arithmetic expressions (e.g. `"47+28=76"`) → score ≥ 0.90
- Integer mismatch with specific values present → score ≥ 0.75
- Approximate/qualitative language (e.g. `"approximately 150"`) → score ≤ 0.40
- No numeric content → score ≤ 0.15
- The score is determined by regex patterns detecting: explicit arithmetic operators with
  two operands and a stated result; approximate language markers (approximately, about,
  roughly, ~); intermediate-step language (step, then, so, later); and numeric specificity
- The function never raises on any string input — unknown patterns return a conservative low score

Theoretical basis: Exp 331 identified VALID_INTERMEDIATE as the primary FP category.
Approximate and intermediate-step violations have lower expression specificity and should
be deprioritised before invoking expensive LLM repair.

Spec: REQ-VERIFY-083

### REQ-VERIFY-084: Partition Function Variance Confidence Signal

The system shall provide `compute_energy_variance_confidence(energies: list[float]) → float`
in `python/carnot/pipeline/confidence_weighted_repair.py`, where:

- The function accepts a list of scalar energies from multiple Ising sampler runs
- Low variance (samples agree) → high confidence (violation is stable / not noise)
- High variance (samples disagree) → low confidence (may be sampling noise)
- Implemented via coefficient of variation: `cv = std(energies) / (mean(energies) + 1e-8)`
  and `confidence = 1 / (1 + cv)`
- Empty list or single-element list → returns 0.5 (uninformative prior)
- All-zero energies → returns 0.5 (division by near-zero handled via the 1e-8 guard)

Theoretical basis: arXiv 2504.13134 proposes partition function variance as a confidence
signal — high variance across Gibbs samples means uncertain constraint (high temperature or
degenerate energy landscape). Low variance means the Ising model consistently agrees the
configuration is wrong.

Spec: REQ-VERIFY-084

### REQ-VERIFY-085: Confidence-Weighted Repair With Dual Signals

The system shall provide `ConfidenceWeightedRepair` in
`python/carnot/pipeline/confidence_weighted_repair.py`, where:

- `ViolationConfidence` dataclass with fields:
  - `expression_confidence` (float): score from REQ-VERIFY-083
  - `energy_variance_confidence` (float): score from REQ-VERIFY-084
  - `combined_confidence` (float): geometric mean of the two signals
  - `is_high_confidence` (bool): True when `combined_confidence >= min_confidence`
  - `min_confidence` (float): threshold used for the is_high_confidence predicate
- `ConfidenceRepairResult` dataclass with fields:
  - `violations_found` (int): total violations extracted
  - `violations_above_threshold` (int): violations with combined_confidence >= threshold
  - `repair_triggered` (bool): True when at least one violation exceeded the threshold
  - `improvement` (int): 1 if repair succeeded and improved the response, 0 otherwise
- `ConfidenceWeightedRepair(pipeline, n_samples=5, min_confidence=0.8)`:
  - `repair(question, response, domain) → ConfidenceRepairResult`:
    1. Run extractor to get all violations
    2. For each violation: compute expression_confidence from violation text
    3. Run Ising n_samples times to get energy distribution → energy_variance_confidence
    4. Compute combined_confidence = geometric mean of both signals
    5. Only trigger repair if combined_confidence >= min_confidence for at least one violation
    6. Returns ConfidenceRepairResult with full accounting
- `VerifyRepairPipeline.verify_repair_confidence_weighted(question, response, domain,
  min_confidence=0.8, n_samples=5) → ConfidenceRepairResult`: additive integration method;
  delegates to ConfidenceWeightedRepair; does NOT modify verify_and_repair() behaviour

Spec: REQ-VERIFY-085

### REQ-VERIFY-086: SinkProbe Attention-Sink Pre-Filter

The system shall provide a SinkProbe component that detects factual uncertainty
via attention-sink concentration analysis (arXiv 2604.10697), where:
- A SinkTokenType enum identifies common sink tokens: BOS, EOS, PERIOD, COMMA
- A SinkConcentration dataclass records per-head sink scores, mean, and max over
  a captured attention matrix
- compute_sink_concentration(attention_matrix, sink_positions) accepts an
  attention tensor of shape (n_heads, seq_len, seq_len) and a list of sink
  token positions, and returns a SinkConcentration by computing the fraction
  of attention mass each head places on the sink positions
- A SinkProbeResult dataclass records the sink_concentration, an is_uncertain
  boolean, and a should_skip_verification boolean
- SinkProbe(threshold, sink_token_types) exposes:
  - score(attention_matrix, sink_positions) → SinkConcentration
  - decide(sink_concentration) → SinkProbeResult, where
    is_uncertain = mean_sink_score < threshold (low sink = uncertain output)
    should_skip_verification = not is_uncertain
  - benchmark(responses_with_attention, correctness_labels) → dict with keys
    skip_rate, false_negative_rate, true_negative_rate
- The component is CI-safe: compute_sink_concentration operates on arbitrary
  jnp arrays without requiring a real language model

Spec: REQ-VERIFY-086

### REQ-VERIFY-087: SinkProbe Threshold Configuration

The system shall support configurable sink-concentration thresholds for the
SinkProbe pre-filter, where:
- The default threshold is 0.3 (responses with mean_sink_score >= 0.3 are
  considered confident and skip expensive Ising verification)
- The threshold can be overridden at construction time for calibration
- A benchmark helper returns skip_rate, false_negative_rate, and
  true_negative_rate so the threshold can be tuned empirically
- The threshold is documented as a calibration parameter, not a fixed constant

Spec: REQ-VERIFY-087

### SCENARIO-VERIFY-113: High Sink Concentration Indicates Confident Response

**Given** an attention matrix where every head places >= 0.5 of its mass on
  a BOS sink token at position 0
**When** `SinkProbe(threshold=0.3).decide(concentration)` is called
**Then** `is_uncertain == False`
**And** `should_skip_verification == True`
**And** the expensive Ising verification step is bypassed on the fast path

### SCENARIO-VERIFY-114: Low Sink Concentration Triggers Full Verification

**Given** an attention matrix where heads distribute attention nearly uniformly
  (mean_sink_score < 0.1)
**When** `SinkProbe(threshold=0.3).decide(concentration)` is called
**Then** `is_uncertain == True`
**And** `should_skip_verification == False`
**And** Ising verification proceeds for this response

### SCENARIO-VERIFY-115: SinkProbe Benchmark Reports Accurate Skip And Error Rates

**Given** 10 synthetic responses: 6 correct (high sink) and 4 incorrect (low sink)
**When** `SinkProbe(threshold=0.3).benchmark(responses_with_attention,
  correctness_labels)` is called with matching labels
**Then** skip_rate is in [0.0, 1.0]
**And** false_negative_rate is in [0.0, 1.0]  (skipped wrong responses / total wrong)
**And** true_negative_rate is in [0.0, 1.0]   (skipped correct responses / total correct)

### REQ-VERIFY-088: Three-Tier Pipeline Benchmark

The system shall provide a combined three-tier verification pipeline that:
- Applies SinkProbe (attention-sink pre-filter, ~0 ms) as the first gate
- Applies EORM (energy-based CoT ranker, ~10 ms) as the second gate
- Falls back to Ising constraint verification (full accuracy) for remaining responses
- Exposes `verify(response, attention_matrix=None, question=None)` returning `(verified: bool, tier_used: str, energy: float)`
- Exposes `benchmark(responses, ground_truth)` returning a `ThreeTierPipelineResult`
- When `attention_matrix=None`, the SinkProbe gate is bypassed (all responses proceed to EORM)

Spec: REQ-VERIFY-088
SCENARIO-VERIFY-116, SCENARIO-VERIFY-117

### SCENARIO-VERIFY-116: Three-Tier Pipeline Routes High-Sink Responses Directly To Fast Path

**Given** a `ThreeTierPipeline` with `sink_threshold=0.3` and `eorm_threshold=0.5`
**And** a response with an attention matrix where mean_sink_score >= 0.3
**When** `pipeline.verify(response, attention_matrix=attn)` is called
**Then** `tier_used == "sink_probe"`
**And** `verified == True`
**And** the EORM and Ising stages are NOT called for this response

### SCENARIO-VERIFY-117: Three-Tier Pipeline Benchmark Measures Skip Rate And Accuracy

**Given** 100 synthetic responses (30% correct with high sink concentration, 70% wrong)
**When** `pipeline.benchmark(responses, ground_truth)` is called
**Then** `result.total_skip_rate` is in [0.0, 1.0]
**And** `result.fn_rate` is in [0.0, 1.0]  (wrong responses incorrectly cleared)
**And** `result.ising_calls_saved_pct` equals `total_skip_rate * 100`
**And** `result.throughput_qps` is > 0.0

### SCENARIO-VERIFY-118: Three-Tier Live GPU Benchmark Captures Real Attention Matrices

**Given** `CARNOT_FORCE_LIVE=1` is set and a Gemma4-E4B-it model is loaded on GPU
**And** 50 GSM8K questions are run with `output_attentions=True`
**When** `experiment_373_three_tier_live.run_experiment()` is called
**Then** `real_attention_matrices_used == True` in the artifact
**And** `inference_mode == "live_gpu"` in the artifact
**And** `skip_rate_sink_probe` + `skip_rate_eorm` + `(1 - total_skip_rate)` sums to 1.0
**And** `ising_calls_saved_pct == total_skip_rate * 100`

### SCENARIO-VERIFY-119: Three-Tier Live Benchmark Honest Verdict Requires Low FN Rate

**Given** a completed live three-tier benchmark run
**When** `total_skip_rate > 0.3` AND `fn_rate < 0.05`
**Then** `honest_verdict == "throughput_gain_live"`
**And** the artifact contains `eorm_model_used` field indicating which EORM was loaded
**When** `total_skip_rate <= 0.3` OR `fn_rate >= 0.05`
**Then** `honest_verdict != "throughput_gain_live"` (conservative: only claims gain when both conditions met)

### SCENARIO-AGENT-001: SAVeR Commits Clean Steps And Propagates Facts

**Given** a SAVeRVerifier with a stub pipeline (no violations)
**And** a 3-step chain: compute subtotal → apply discount → compute final price
**When** `verifier.run_chain(steps, initial_state)` is called
**Then** all 3 `AgentStep` objects have `committed=True`
**And** `constraint_state.accumulated_facts` contains 3 entries after the chain
**And** `verifier.compute_faithfulness(steps)` returns 1.0

### SCENARIO-AGENT-002: SAVeR Blocks A Step That Violates Accumulated Constraints After Max Repairs

**Given** a SAVeRVerifier with `max_repair_attempts=2` and a mock pipeline
**And** the mock pipeline returns violations on every call for step 2 (even after repair)
**When** `verifier.propose_step(question, action_cot, constraint_state)` is called for step 2
**Then** the returned `AgentStep` has `committed=False`
**And** `repair_attempts` equals 2 (the maximum)
**And** `constraint_state.accumulated_facts` is NOT updated (blocked step does not add facts)
**And** `compute_faithfulness([step])` returns 0.0

### SCENARIO-AGENT-003: SAVeR Repairs A Violating Step Before Committing

**Given** a SAVeRVerifier with `max_repair_attempts=3` and a mock pipeline
**And** the mock pipeline returns violations on first verify, then passes on repair
**When** `verifier.propose_step(question, action_cot, constraint_state)` is called
**Then** the returned `AgentStep` has `committed=True`
**And** `repaired_action` is set (not None) reflecting the repaired response
**And** `constraint_violations` captured the initial violation descriptions
**And** `repair_attempts` is >= 1

### SCENARIO-VERIFY-109: Exact Arithmetic Expression Gets High Expression Confidence

**Given** violation_text = `"47+28=76"` (explicit arithmetic with wrong result)
**When** `compute_expression_confidence("47+28=76")` is called
**Then** the returned score is >= 0.90
**And** the score reflects high specificity — an exact arithmetic expression with two
  operands, an operator, and a stated result

### SCENARIO-VERIFY-110: Approximate Language Gets Low Expression Confidence

**Given** violation_text = `"the intermediate result is approximately 150"`
**When** `compute_expression_confidence("the intermediate result is approximately 150")` is called
**Then** the returned score is <= 0.40
**And** the score reflects low specificity — approximate/qualitative language deprioritises repair

### SCENARIO-VERIFY-111: Low Variance Energies Give High Variance Confidence

**Given** energies = [2.0, 2.1, 1.9, 2.05, 1.95] (low coefficient of variation)
**When** `compute_energy_variance_confidence([2.0, 2.1, 1.9, 2.05, 1.95])` is called
**Then** the returned confidence is > 0.8
**And** the score reflects that samples consistently agree the violation is real

### SCENARIO-VERIFY-112: High Variance Energies Give Low Variance Confidence

**Given** energies = [0.1, 5.0, 0.2, 8.0, 0.05] (high coefficient of variation)
**When** `compute_energy_variance_confidence([0.1, 5.0, 0.2, 8.0, 0.05])` is called
**Then** the returned confidence is < 0.5
**And** the score reflects that samples disagree — violation may be sampling noise

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

### REQ-LEARN-014: Four-Tier Relay Live GPU Validation

**Priority:** High
**Experiment:** Exp 329

The system MUST re-run the Exp 318 four-tier continuous self-learning relay benchmark
with live GPU inference (`CARNOT_FORCE_LIVE=1`) to determine whether the relay stack
produces `improvement_1to3 > 0` on real model outputs.

Requirements:
- REQ-LEARN-014-1: The wrapper script (Exp 329) MUST invoke `experiment_318_self_learning_relay.py`
  with `CARNOT_FORCE_LIVE=1` to obtain live model outputs.
- REQ-LEARN-014-2: The resulting artifact MUST have `inference_mode="live_gpu"`.
  If the GPU is unavailable the wrapper MUST emit `status="blocked"` and never silently
  fall back to simulated results.
- REQ-LEARN-014-3: `improvement_1to3` MUST be reported as a signed float (not clamped).
  Negative results are valid and must appear in the artifact.
- REQ-LEARN-014-4: The artifact MUST include a `simulation_comparison` dict with keys
  `batch1_accuracy_delta`, `batch3_accuracy_delta`, and `improvement_delta` computed
  against the Exp 318 simulated baseline.
- REQ-LEARN-014-5: `jepa_skip_rate_live` MUST be captured from the live run and included
  in the wrapper artifact alongside `jepa_skip_rate_simulated` for comparison.

### SCENARIO-LEARN-023: Live Relay Artifact Rejected If Not Live GPU

**Given** a result artifact loaded from an Exp 318 re-run
**When** `validate_relay_live(result)` is called
**And** `result["inference_mode"] != "live_gpu"`
**Then** a `ValueError` is raised describing the rejected mode and instructing the caller
  to re-run with `CARNOT_FORCE_LIVE=1`

### SCENARIO-LEARN-024: Simulation Comparison Keys Present And Signed

**Given** a live relay result and a simulated relay result (Exp 318 baseline)
**When** `compare_relay_to_simulated(live, simulated)` is called
**Then** the returned dict contains keys `batch1_accuracy_delta`, `batch3_accuracy_delta`,
  and `improvement_delta`
**And** each value equals `live[key] - simulated[key]` (signed, not absolute)
**And** negative deltas are preserved without clamping

### REQ-LEARN-015: Model-Adaptive Constraint Thresholds

The system shall track false-positive (FP) and true-positive (TP) rates per
(model_id, constraint_type) pair and disable constraint types for a given model
when their FP rate exceeds their TP rate, provided a minimum observation count
has been reached. This self-calibration ensures the verify-repair pipeline
improves accuracy over time without manual per-model tuning.

Rationale: Exp 331 FP autopsy showed that certain constraint types (e.g.,
NL2Z3 range checks) have systematically high FP rates on small models like
Qwen3.5-0.8B but low FP rates on larger models. Disabling noisy constraint
types per-model recovers the net positive gain of the repair pipeline.

- REQ-LEARN-015-1: `PerModelFPTracker.update(model_id, constraint_type, was_fp, was_tp)` MUST
  update running counts for the (model_id, constraint_type) pair atomically.
- REQ-LEARN-015-2: `PerModelFPTracker.should_disable(model_id, constraint_type)` MUST
  return True when fp_rate > tp_rate AND n_observations >= min_observations.
- REQ-LEARN-015-3: `PerModelFPTracker.get_active_constraint_types(model_id)` MUST
  return a frozenset of constraint types not currently disabled.
- REQ-LEARN-015-4: `ModelAdaptiveThresholds.extract(question, response, model_id)` MUST
  call the wrapped extractor and filter out violations whose constraint_type is
  disabled for model_id.
- REQ-LEARN-015-5: `PerModelFPTracker` MUST support `to_dict()` / `from_dict()` for
  persistence across experiment runs.

Spec: REQ-LEARN-015, SCENARIO-LEARN-025, SCENARIO-LEARN-026

### REQ-LEARN-016: Selective CaseMemory Consolidation

Following ATLAS (arXiv 2511.01093) selective memory consolidation, the system
shall retain only high-contrast interactions in CaseMemory — those where the
verified violation energy disagrees with the model's confidence direction. This
reduces memory footprint while maximising the information value of retained cases.

Rationale: Low-contrast traces (where verification agreed with the model's
apparent confidence) provide weak learning signal. ATLAS shows ~60% memory
reduction with no precision loss when only high-contrast cases are retained.

- REQ-LEARN-016-1: `SelectiveConsolidation.should_retain(verified_violation_energy,
  model_confidence_score)` MUST return True when the two signals disagree
  (high energy + low confidence, or low energy + high confidence indicates
  a surprising interaction worth retaining).
- REQ-LEARN-016-2: `SelectiveConsolidation.consolidation_ratio(all_traces, retained)` MUST
  return the fraction of traces retained (target 0.3–0.5).
- REQ-LEARN-016-3: `CaseMemory.add_trace_selective(trace, min_contrast)` MUST only
  store the trace when `abs(violation_energy - model_confidence) > min_contrast`.
- REQ-LEARN-016-4: `add_trace_selective` MUST return True when the trace was stored,
  False when it was discarded.

Spec: REQ-LEARN-016, SCENARIO-LEARN-027, SCENARIO-LEARN-028

### REQ-LEARN-017: Constraint Template Addition from Memory Patterns

When CaseMemory detects that a specific error pattern is common for a given model
(e.g., "carry errors appear in 40% of Qwen3.5-0.8B arithmetic responses"), the
system shall ADD a new constraint template that checks that error type — rather than
upweighting existing constraints that do not cover the error.

Rationale: Exp 134 proved that precision-based reweighting (upweighting existing
constraint types when they fire) does NOT improve accuracy — fixed vs. adaptive
strategies are equivalent across 500 questions. Root cause: if the existing
constraint set doesn't cover the real error type, upweighting amplifies noise.
The fix is constraint ADDITION, not reweighting. (arXiv 2603.03538, arXiv 2512.20664)

- REQ-LEARN-017-1: `ConstraintTemplateLibrary.observe_pattern(pattern_key, model_id,
  count=1)` MUST increment the per-(pattern_key, model_id) observation count.
- REQ-LEARN-017-2: `ConstraintTemplateLibrary.get_active_templates(model_id)` MUST
  return only templates where cumulative observed count >= template.min_frequency.
- REQ-LEARN-017-3: `ConstraintTemplateLibrary.apply_active_templates(response, model_id)`
  MUST call template_fn for each active template and merge all returned constraints.
  Each called template's `is_active` is set True and `activation_count` is incremented.

Spec: REQ-LEARN-017, SCENARIO-LEARN-029, SCENARIO-LEARN-030, SCENARIO-LEARN-031,
      SCENARIO-LEARN-032

### REQ-LEARN-018: Constraint Template Persistence and Built-in Templates

The system shall support serialization and built-in template registration for the
ConstraintTemplateLibrary, enabling persistence of learned observation counts across
experiment runs.

- REQ-LEARN-018-1: `ConstraintTemplateLibrary.to_dict()` / `from_dict()` MUST
  serialize/restore observation counts. Template functions (callables) are NOT
  serialized; the caller calls `register_builtin_templates()` after `from_dict()`.
- REQ-LEARN-018-2: `register_builtin_templates()` MUST register four standard
  templates from the Eidoku taxonomy: `carry_check` (min_freq=5), `sign_check`
  (min_freq=5), `unit_consistency` (min_freq=3), `comparison_direction` (min_freq=5).
- REQ-LEARN-018-3: All built-in template functions MUST be CI-safe: return [] when
  the response contains no parseable arithmetic of the relevant type.

Spec: REQ-LEARN-018, SCENARIO-LEARN-029, SCENARIO-LEARN-030, SCENARIO-LEARN-031,
      SCENARIO-LEARN-032

### REQ-LEARN-019: CaseMemory to ConstraintTemplateLibrary Wiring

The system shall provide a `CaseMemoryTemplateWiring` class that bridges CaseMemory
violation events to `ConstraintTemplateLibrary.observe_pattern()` calls, forming
the Tier 2 → Tier 1 feedback loop.

- REQ-LEARN-019-1: `CaseMemoryTemplateWiring.__init__(library)` MUST accept a
  `ConstraintTemplateLibrary` instance and store it for use in event handlers.
- REQ-LEARN-019-2: `CaseMemoryTemplateWiring.on_violation_recorded(violation_type,
  model_id)` MUST map the violation_type string to a pattern_key via
  `violation_type_to_pattern_key()` and call `library.observe_pattern(pattern_key,
  model_id)` with count=1.
- REQ-LEARN-019-3: `CaseMemoryTemplateWiring.violation_type_to_pattern_key(
  violation_type)` MUST implement the canonical mapping:
    - "carry_error" or any type containing "carry" → "carry_check"
    - "sign_error" or any type containing "sign" → "sign_check"
    - "unit_error" or any type containing "unit" → "unit_consistency"
    - "comparison_error" or any type containing "comparison" → "comparison_direction"
    - All other types → the violation_type unchanged (pass-through)
- REQ-LEARN-019-4: The mapping MUST be case-insensitive (e.g., "CARRY_ERROR"
  maps to "carry_check").

Spec: REQ-LEARN-019, SCENARIO-LEARN-033, SCENARIO-LEARN-034

### REQ-LEARN-020: Session State Persistence

The system shall provide a `SessionMemory` class that serialises and restores
`CaseMemory`, `ConstraintTemplateLibrary`, and `PerModelFPTracker` across
process restarts so that learned patterns accumulate over time.

- REQ-LEARN-020-1: `SessionMemory(storage_dir: str, model_id: str)` MUST accept
  a filesystem directory and a model identifier as construction parameters.
- REQ-LEARN-020-2: `SessionMemory.save(case_memory, template_library, fp_tracker)`
  MUST serialise all three components to `(storage_dir)/(model_id)/session_state.json`
  as a single JSON document conforming to schema `"carnot.session_memory.v1"` with
  keys `"case_memory"`, `"template_library"`, `"fp_tracker"`, and `"saved_at"`
  (ISO 8601 UTC timestamp).
- REQ-LEARN-020-3: `save()` MUST be idempotent: calling it multiple times
  overwrites the existing file rather than appending.
- REQ-LEARN-020-4: `SessionMemory.load()` MUST return a tuple
  `(CaseMemory, ConstraintTemplateLibrary, PerModelFPTracker)` when saved state
  exists, or `None` when no state file is present for the given `model_id`.
- REQ-LEARN-020-5: `load()` MUST be CI-safe: return `None` (not raise) when the
  state file is missing, empty, or contains malformed JSON.
- REQ-LEARN-020-6: `SessionMemory.exists()` MUST return `True` if and only if the
  expected state file is present on disk.
- REQ-LEARN-020-7: `SessionMemory.clear()` MUST delete the state file for the
  given `model_id` if it exists, and be a no-op if the file does not exist.
- REQ-LEARN-020-8: `SessionMemory.list_sessions(storage_dir)` (static/class method)
  MUST return a sorted list of `model_id` strings for which a
  `session_state.json` file exists under `storage_dir`; return `[]` when
  `storage_dir` does not exist.

Spec: REQ-LEARN-020, SCENARIO-LEARN-035, SCENARIO-LEARN-036, SCENARIO-LEARN-037

### REQ-LEARN-021: Per-Model Session State Isolation

Each model's session state MUST be stored in its own subdirectory so that
saving state for model A never overwrites or corrupts state for model B.

- REQ-LEARN-021-1: The storage path for a given `model_id` MUST be
  `(storage_dir)/(model_id)/session_state.json`.  Slashes in `model_id` (e.g.
  `"org/model-name"`) MUST be escaped to a safe filesystem character (`"__"`)
  so that no nested directories are created unintentionally.
- REQ-LEARN-021-2: `VerifyRepairPipeline` MUST accept an optional
  `session_memory: SessionMemory | None = None` constructor parameter.  When
  provided, the pipeline MUST call `session_memory.load()` during `__init__`
  and restore whichever components were returned.
- REQ-LEARN-021-3: `VerifyRepairPipeline.close()` MUST save current state via
  `session_memory.save(...)` when `session_memory` is set, and be a no-op
  otherwise.

Spec: REQ-LEARN-021, SCENARIO-LEARN-035, SCENARIO-LEARN-036, SCENARIO-LEARN-037

### SCENARIO-LEARN-035: SessionMemory Save and Load Round-Trip

**Given** a `SessionMemory(storage_dir="/tmp/test_sessions", model_id="test-model")`
**And** a `CaseMemory` with 3 recorded cases
**And** a `ConstraintTemplateLibrary` with 1 observation recorded
**And** a `PerModelFPTracker` with 1 FP recorded
**When** `save(case_memory, template_library, fp_tracker)` is called
**Then** `exists()` returns `True`
**And** `load()` returns a tuple whose `CaseMemory` has the same number of entries
**And** the loaded `PerModelFPTracker` has the same stats

### SCENARIO-LEARN-036: SessionMemory Returns None When No State Exists

**Given** a `SessionMemory(storage_dir="/tmp/no_such_dir", model_id="missing-model")`
**When** `load()` is called
**Then** the result is `None`
**And** `exists()` returns `False`

### SCENARIO-LEARN-037: SessionMemory list_sessions Returns All Saved Model IDs

**Given** saved states for model_ids `["model-a", "model-b"]` under the same `storage_dir`
**When** `SessionMemory.list_sessions(storage_dir)` is called
**Then** the result is `["model-a", "model-b"]` (sorted)
**And** `list_sessions` returns `[]` when `storage_dir` does not exist

### SCENARIO-LEARN-025: PerModelFPTracker Disables High-FP Constraint Type

**Given** a `PerModelFPTracker(min_observations=10)`
**And** 12 observations for model "qwen3.5-0.8b" / constraint_type "range_check"
  where 9 are FP and 3 are TP (fp_rate=0.75, tp_rate=0.25)
**When** `should_disable("qwen3.5-0.8b", "range_check")` is called
**Then** the result is `True`

### SCENARIO-LEARN-026: PerModelFPTracker Keeps Low-FP Constraint Type Active

**Given** a `PerModelFPTracker(min_observations=10)`
**And** 12 observations for model "qwen3.5-0.8b" / constraint_type "arithmetic"
  where 3 are FP and 9 are TP (fp_rate=0.25, tp_rate=0.75)
**When** `should_disable("qwen3.5-0.8b", "arithmetic")` is called
**Then** the result is `False`
**And** `get_active_constraint_types("qwen3.5-0.8b")` includes "arithmetic"

### SCENARIO-LEARN-027: SelectiveConsolidation Retains High-Contrast Trace

**Given** a `SelectiveConsolidation()` instance
**And** a trace with `verified_violation_energy=0.9` and `model_confidence_score=0.1`
  (high energy but model was not confident — surprising disagreement)
**When** `should_retain(0.9, 0.1)` is called
**Then** the result is `True` (contrast = 0.8 > default threshold 0.5)

### SCENARIO-LEARN-028: CaseMemory add_trace_selective Discards Low-Contrast Trace

**Given** a `CaseMemory` instance and `min_contrast=0.5`
**And** a trace with `violation_energy=0.6` and `model_confidence=0.55`
  (contrast = 0.05 < 0.5)
**When** `add_trace_selective(trace, min_contrast=0.5)` is called
**Then** the method returns `False` and the memory length is unchanged

### SCENARIO-LEARN-029: carry_check_template Detects Carry Error

**Given** a response string `"24 × 3 = 62"` (correct answer is 72)
**When** `carry_check_template("24 × 3 = 62")` is called
**Then** the result contains one ConstraintResult with `constraint_type="carry_check"`,
  `metadata["satisfied"]=False`, `metadata["correct"]=72`, `metadata["claimed"]=62`

### SCENARIO-LEARN-030: sign_check_template Detects Sign Error

**Given** a response string `"(-3) × (-4) = -12"` (should be positive)
**When** `sign_check_template("(-3) × (-4) = -12")` is called
**Then** the result contains one ConstraintResult with `constraint_type="sign_check"`,
  `metadata["satisfied"]=False`, `metadata["claimed"]=-12.0`

### SCENARIO-LEARN-031: unit_consistency_template Detects Incompatible Units

**Given** a response string containing both "5 kg" and "200 g"
**When** `unit_consistency_template(response)` is called
**Then** the result contains a ConstraintResult with `constraint_type="unit_consistency"`,
  `metadata["satisfied"]=False`, and `metadata["inconsistent_pair"]` containing "kg" and "g"

### SCENARIO-LEARN-032: comparison_direction_template Detects Direction Inconsistency

**Given** a response string `"Since 50 > 30, we compute 50 - 30 = -20."`
**When** `comparison_direction_template(response)` is called
**Then** the result contains one ConstraintResult with `constraint_type="comparison_direction"`,
  `metadata["satisfied"]=False` (z = -20 is not positive)

### SCENARIO-LEARN-033: CaseMemoryTemplateWiring Maps carry_error to carry_check

**Given** a `CaseMemoryTemplateWiring` wrapping a `ConstraintTemplateLibrary`
**When** `on_violation_recorded("carry_error", "qwen3.5-0.8b")` is called
**Then** `library._observations[("carry_check", "qwen3.5-0.8b")]` is incremented by 1
**And** `violation_type_to_pattern_key("carry_error")` returns `"carry_check"`

### SCENARIO-LEARN-034: CaseMemoryTemplateWiring Unknown Type Passes Through

**Given** a `CaseMemoryTemplateWiring` wrapping a `ConstraintTemplateLibrary`
**When** `on_violation_recorded("range_check", "model_a")` is called
  (a type with no canonical mapping)
**Then** `library._observations[("range_check", "model_a")]` is incremented by 1
**And** `violation_type_to_pattern_key("range_check")` returns `"range_check"` unchanged

### REQ-LEARN-022: EORM CoT Energy Reward Model

The system shall provide an `EORMModel` class in `python/carnot/models/eorm.py`
that scores complete chain-of-thought (CoT) responses by energy, enabling
best-of-N selection by choosing the lowest-energy candidate.

Architecture follows arXiv 2505.14999: a small transformer encoder that reads
the full (question, CoT response) pair and outputs a scalar energy — lower
energy signals higher-quality reasoning.

- REQ-LEARN-022-1: A `CoTEnergyInput` dataclass MUST hold `response_text: str`
  and `question_text: str`.
- REQ-LEARN-022-2: `EORMModel.__init__(embed_dim=128, n_heads=4, n_layers=2,
  max_seq_len=512, vocab_size=4096)` MUST initialize a JAX transformer encoder
  whose parameter count is at most 100 million.
- REQ-LEARN-022-3: `EORMModel.energy(cot_input: CoTEnergyInput) -> float` MUST
  return a finite scalar.  Lower energy means the model considers the CoT
  response more correct.
- REQ-LEARN-022-4: `EORMModel.rank(responses: list[str], question: str) ->
  list[int]` MUST return index positions sorted ascending by energy (lowest
  energy = best response = first in the returned list).
- REQ-LEARN-022-5: `EORMModel.save(path)` / `EORMModel.load(path)` MUST
  serialize and restore all parameters using safetensors, with architecture
  config stored in an adjacent `_config.json` sidecar file.
- REQ-LEARN-022-6: `EORMModel.n_params` property MUST return the total
  trainable parameter count as an integer.

Spec: REQ-LEARN-022, SCENARIO-LEARN-038, SCENARIO-LEARN-039

### REQ-LEARN-023: EORM Contrastive Ranking Loss and Trainer

The system shall provide an `EORMTrainer` class in `python/carnot/models/eorm.py`
that trains `EORMModel` via contrastive (hinge) loss on (correct, incorrect)
response pairs derived from live benchmark data.

- REQ-LEARN-023-1: `EORMTrainer.__init__(model: EORMModel, lr: float = 1e-4)`
  MUST store the model and learning rate.
- REQ-LEARN-023-2: `EORMTrainer.contrastive_loss(correct_energy: float,
  incorrect_energy: float, margin: float = 1.0) -> float` MUST implement hinge
  loss: `max(0, correct_energy - incorrect_energy + margin)`.  Loss is zero
  when the incorrect response has energy at least `margin` higher than the
  correct response.
- REQ-LEARN-023-3: `EORMTrainer.train_step(correct_response: str,
  incorrect_response: str, question: str) -> float` MUST compute contrastive
  loss, compute parameter gradients via `jax.value_and_grad`, apply a gradient
  descent update to `model.params`, and return the scalar loss value.
- REQ-LEARN-023-4: `EORMTrainer.train_epoch(pairs: list[tuple[str, str, str]],
  batch_size: int = 16) -> float` MUST iterate over the pairs list in
  `batch_size` chunks, call `train_step` for each pair, and return the mean
  loss over all pairs in the epoch.

Spec: REQ-LEARN-023, SCENARIO-LEARN-040

### SCENARIO-LEARN-038: EORMModel Energy Is Finite for CoTEnergyInput

**Given** an `EORMModel` with default parameters (embed_dim=128, n_heads=4,
  n_layers=2)
**And** a `CoTEnergyInput(question_text="What is 2+2?", response_text="It is 4.")`
**When** `model.energy(cot_input)` is called
**Then** the returned value is a finite float
**And** `model.n_params` is a positive integer no greater than 100_000_000

### SCENARIO-LEARN-039: EORMModel rank Returns Indices Sorted by Energy

**Given** an `EORMModel` with default parameters
**And** responses `["bad answer", "good answer", "mediocre answer"]`
**When** `model.rank(responses, question="2+2?")` is called
**Then** the returned list contains exactly the indices [0, 1, 2] in some order
**And** the energies at those indices are non-decreasing

### SCENARIO-LEARN-040: EORMTrainer Contrastive Loss Is Zero When Margin Already Met

**Given** an `EORMTrainer` with default `margin=1.0`
**When** `contrastive_loss(correct_energy=2.0, incorrect_energy=4.0)` is called
**Then** the returned loss is 0.0 (incorrect has energy > correct + margin)
**When** `contrastive_loss(correct_energy=4.0, incorrect_energy=2.0)` is called
**Then** the returned loss is 3.0 (hinge: max(0, 4.0 - 2.0 + 1.0))

### REQ-LEARN-024: JEPA Real-Data Retrain on Live Violation Pairs

Given live GPU inference data (real model responses with ground-truth correctness labels),
the system shall retrain the JEPA predictor on real `(partial_response, violation_flag)` pairs
to enable preemptive verification — predicting whether a response will contain a constraint
violation from only its first N tokens, before generation completes.

- REQ-LEARN-024-1: A `ViolationPair` dataclass SHALL carry `partial_response` (first N tokens of a response as a string), `full_response` (complete response), `has_violation` (bool), `model_id` (str), and `question_id` (str).
- REQ-LEARN-024-2: `extract_violation_pairs(live_results: dict, prefix_fraction: float = 0.5) -> list[ViolationPair]` SHALL split each response at `prefix_fraction * len(tokens)` (word-tokenized) to produce the partial prefix; `has_violation = not correct` where `correct` is the ground-truth label from the live results dict.
- REQ-LEARN-024-3: When `live_results` is `None` or empty, `extract_violation_pairs` SHALL return 50 synthetic pairs (deterministic, seed-based) so CI can exercise all code paths without GPU data.
- REQ-LEARN-024-4: `JEPARetrainer.__init__(jepa_model, lr: float = 1e-4)` SHALL accept a `ContextPredictionEnergy` model and a scalar learning rate.
- REQ-LEARN-024-5: `JEPARetrainer.binary_ce_loss(predicted_energy: float, has_violation: bool) -> float` SHALL return the binary cross-entropy loss treating high energy as predicting a violation.
- REQ-LEARN-024-6: `JEPARetrainer.train_epoch(pairs: list[ViolationPair], batch_size: int = 8) -> float` SHALL train for one epoch over the pairs in mini-batches and return the mean loss.
- REQ-LEARN-024-7: `JEPARetrainer.evaluate_auc_roc(pairs: list[ViolationPair]) -> float` SHALL return the AUC-ROC of the model's violation predictions on the given pairs (using sklearn or a pure-numpy trapezoidal approximation).
- REQ-LEARN-024-8: `build_retrain_artifact(before_auc: float, after_auc: float, n_pairs: int) -> dict` SHALL return a dict with keys `before_auc`, `after_auc`, `auc_improvement` (signed float), `n_pairs`, `schema_version="carnot.jepa_retrain.v1"`.

Spec: REQ-LEARN-024, SCENARIO-LEARN-041, SCENARIO-LEARN-042

### SCENARIO-LEARN-041: extract_violation_pairs Splits at prefix_fraction

**Given** a live results dict with 10 responses, each 20 words long, and `prefix_fraction=0.5`
**When** `extract_violation_pairs(live_results, prefix_fraction=0.5)` is called
**Then** each `ViolationPair.partial_response` contains exactly the first 10 words
**And** `has_violation` equals `not correct` for each pair
**And** the returned list has exactly 10 entries

### SCENARIO-LEARN-042: extract_violation_pairs Returns Synthetic Pairs When No Live Data

**Given** `live_results=None`
**When** `extract_violation_pairs(None)` is called
**Then** a list of exactly 50 `ViolationPair` objects is returned
**And** each pair has a non-empty `partial_response`, a valid `has_violation` bool,
  a non-empty `model_id`, and a non-empty `question_id`
**And** the function is deterministic (same output on repeated calls)

### REQ-LEARN-025: EORM Real-Data Retrain on Live CoT Pairs

Given experiment result files that may contain real (question, response, correctness) triples
from live GPU inference runs, the system shall load those pairs, merge them with any available
synthetic corpus from the Exp 346 EORM training, retrain the EORM model, and report AUC-ROC
before and after retrain to measure improvement from real data.

- REQ-LEARN-025-1: `load_real_cot_pairs(result_files: list[str]) -> list[ViolationPair]` SHALL read each JSON file, extract entries with a `response` (or equivalent) field and a `correct` (or `passed_tests`) boolean, and return a flat list of `ViolationPair` objects. Missing files, missing keys, and empty lists SHALL be handled gracefully (skip silently, return empty list).
- REQ-LEARN-025-2: `merge_cot_corpora(real_pairs: list[ViolationPair], synthetic_pairs: list[ViolationPair], max_real: int = 300, max_synthetic: int = 100) -> list[ViolationPair]` SHALL prefer real pairs; use up to `max_real` real pairs first, then fill with up to `max_synthetic` synthetic pairs if fewer than `max_real` real pairs are available.
- REQ-LEARN-025-3: An `EORMRetrainResult` dataclass SHALL carry `n_real_pairs` (int), `n_synthetic_pairs` (int), `before_auc` (float), `after_auc` (float), `auc_improvement` (float), `retrain_mode` (str: "real_data" or "synthetic_only"), and `model_path` (str).
- REQ-LEARN-025-4: `build_retrain_artifact(result: EORMRetrainResult) -> dict` SHALL return a dict with `schema="carnot.eorm_retrain.v1"`, `retrain_mode`, `auc_improvement`, and `honest_verdict` ("real_data_improvement" when `retrain_mode="real_data"` and `auc_improvement > 0`, "real_data_no_improvement" when `retrain_mode="real_data"` and `auc_improvement <= 0`, or "synthetic_only" when `retrain_mode="synthetic_only"`).

Spec: REQ-LEARN-025, SCENARIO-LEARN-043, SCENARIO-LEARN-044

### SCENARIO-LEARN-043: load_real_cot_pairs Handles Missing Files and Keys Gracefully

**Given** a list of file paths where some do not exist and others exist but lack a `responses` key
**When** `load_real_cot_pairs(result_files)` is called
**Then** no exception is raised
**And** only valid (response, correctness) entries from files that exist and have the key are returned
**And** the returned list may be empty if no valid data is found

### SCENARIO-LEARN-044: merge_cot_corpora Prefers Real Pairs and Fills with Synthetic

**Given** 30 real pairs and 100 synthetic pairs, with `max_real=300` and `max_synthetic=100`
**When** `merge_cot_corpora(real_pairs, synthetic_pairs, max_real=300, max_synthetic=100)` is called
**Then** all 30 real pairs appear in the result (real count is below max_real)
**And** up to 100 synthetic pairs are appended to fill the corpus
**And** the total length is 130 (30 real + 100 synthetic)

### REQ-LEARN-026: Tier 1+2+3 Self-Learning Relay

The system SHALL implement a `SelfLearningRelay` that runs all three self-learning tiers
in concert on real (or synthetic) model outputs and measures whether accuracy improves
across batches.

- REQ-LEARN-026-1: A `SelfLearningBatchResult` dataclass SHALL carry `batch_id` (int), `n_questions` (int), `accuracy` (float), `n_tier1_updates` (int), `n_tier2_templates_active` (int), `tier3_gate_auc` (float), and `cumulative_accuracy` (float).
- REQ-LEARN-026-2: `SelfLearningRelay.__init__(pipeline, template_library, fp_tracker, eorm_model)` SHALL accept a `ThreeTierPipeline`, a `ConstraintTemplateLibrary`, a `PerModelFPTracker`, and an `EORMModel`.
- REQ-LEARN-026-3: `SelfLearningRelay.run_batch(questions: list[str], ground_truth: list[bool], model_id: str) -> SelfLearningBatchResult` SHALL: (a) score each (question, response) pair with the EORM model; (b) update the `PerModelFPTracker` after each question (Tier 1 online weight updates); (c) call `CaseMemoryTemplateWiring.on_violation_recorded()` for each incorrect response (Tier 2 pattern accumulation); (d) compute EORM gate AUC-ROC on the batch (Tier 3 gate accuracy); (e) append the result to the internal trajectory.
- REQ-LEARN-026-4: `SelfLearningRelay.learning_trajectory() -> list[SelfLearningBatchResult]` SHALL return a copy of all batch results accumulated so far.
- REQ-LEARN-026-5: `cumulative_accuracy` in each `SelfLearningBatchResult` SHALL be computed as total correct across all batches so far (including the current batch) divided by total questions.

Spec: REQ-LEARN-026, SCENARIO-LEARN-045, SCENARIO-LEARN-046

### REQ-LEARN-027: Cross-Batch Learning Improvement Measurement

The system SHALL provide utilities to measure whether the self-learning relay improves
accuracy across batches.

- REQ-LEARN-027-1: `compute_learning_improvement(trajectory: list[SelfLearningBatchResult]) -> tuple[float, float, bool]` SHALL return `(batch1_accuracy, batch4_accuracy, improved)` where `improved` is True when `batch4_accuracy > batch1_accuracy`. When the trajectory has fewer than 4 batches, the last available batch is used as the final batch. When the trajectory is empty, all values are 0.0 and `improved` is False.
- REQ-LEARN-027-2: `build_relay_artifact(trajectory: list[SelfLearningBatchResult], learning_improvement: tuple[float, float, bool], *, inference_mode: str = "cpu_synthetic") -> dict` SHALL return a dict with `schema="carnot.self_learning_relay.v1"`, a `trajectory` list (each batch serialized as a dict), `batch1_accuracy`, `batch4_accuracy`, `improved` (bool), `inference_mode`, and `honest_verdict` — where `honest_verdict` is `"learning_confirmed"` only when `improved=True` AND `inference_mode=="live_gpu"`, otherwise `"synthetic_only"` or `"no_improvement"` as appropriate.

Spec: REQ-LEARN-027, SCENARIO-LEARN-047

### REQ-LEARN-036: JEPA Retrain on 200+ Real CoT Pairs (Exp 472)

The system SHALL provide JEPARetrainResult capturing before/after AUC-ROC
when JEPA is retrained on an expanded dataset of real chain-of-thought pairs.

- REQ-LEARN-036-1: `JEPARetrainResult(n_pairs: int, before_auc: float, after_auc: float)` SHALL be
  a dataclass with `auc_improvement: float` (= after_auc - before_auc) and
  `target_met: bool` (True when after_auc > 0.700).
- REQ-LEARN-036-2: Exp 472 retrains JEPA on all available real CoT pairs (Exp 443 57 pairs +
  Exp 464/467 up to 200 additional), capped at 300 total.
- REQ-LEARN-036-3: The retrain targets after_auc > 0.700 on the held-out test split.

### SCENARIO-LEARN-064: JEPARetrainResult Target Met

**Given** `JEPARetrainResult(n_pairs=200, before_auc=0.571, after_auc=0.720)`
**When** `target_met` is evaluated
**Then** `target_met` is True (0.720 > 0.700)
**And** `auc_improvement` is approximately 0.149

### REQ-LEARN-037: Cross-Session Tier 2 Relay

**Purpose:** Validate that constraint templates learned in Session N persist to Session N+1 and reduce false positive (FP) rate on the same error domain. Within-session learning is washed out at process restart; persistent memory accumulates signal across queries and sessions.

**Requirements:**
- REQ-LEARN-037-1: `CrossSessionResult(session_id: int, n_questions: int, fp_rate: float, n_templates_active: int, n_templates_loaded_from_prior: int)` SHALL be a dataclass capturing per-session relay metrics.
- REQ-LEARN-037-2: `simulate_session(session_id: int, questions: list, prior_memory_path: str | None, memory_dir: str) -> CrossSessionResult` SHALL: (a) load the prior session's `ConstraintTemplateLibrary` from `SessionMemory` when `prior_memory_path` is not None; (b) run `VerifyRepairPipeline` on questions with the loaded template library; (c) record violations and FP rate; (d) save the updated `ConstraintTemplateLibrary` to `SessionMemory` for this session; (e) return `CrossSessionResult`.
- REQ-LEARN-037-3: `compute_relay_verdict(sessions: list[CrossSessionResult]) -> str` SHALL return `"cross_session_improvement"` when `sessions[1].fp_rate < sessions[0].fp_rate`, `"no_improvement"` when `sessions[1].fp_rate >= sessions[0].fp_rate`, and `"insufficient_data"` when `len(sessions) < 2`.

Spec: REQ-LEARN-037, SCENARIO-LEARN-066, SCENARIO-LEARN-067, SCENARIO-LEARN-068

### REQ-LEARN-038: Session Memory Template Round-Trip

**Purpose:** Guarantee that `SessionMemory.save()` / `SessionMemory.load()` preserves the full `ConstraintTemplateLibrary` observation state without loss.

**Requirements:**
- REQ-LEARN-038-1: After `SessionMemory.save(case_memory, template_library, fp_tracker)` and `SessionMemory.load()`, the loaded `ConstraintTemplateLibrary` SHALL have identical `_observations` to the saved library.
- REQ-LEARN-038-2: After `register_builtin_templates()` is called on the loaded library, all four built-in templates SHALL be active for models that crossed their `min_frequency` threshold in the saved state.

Spec: REQ-LEARN-038, SCENARIO-LEARN-067

### SCENARIO-LEARN-045: SelfLearningRelay Accumulates Tier 1 Updates Per Question

**Given** a `SelfLearningRelay` with a stub `ThreeTierPipeline`
**And** a batch of 25 synthetic questions with 15 correct and 10 incorrect answers
**When** `run_batch(questions, ground_truth, model_id)` is called
**Then** the returned `SelfLearningBatchResult` has `n_tier1_updates == 25`
**And** `accuracy == 0.6` (15/25)
**And** `n_questions == 25`

### SCENARIO-LEARN-046: SelfLearningRelay Tier 2 Templates Activate After Pattern Threshold

**Given** a `SelfLearningRelay` with a `ConstraintTemplateLibrary` that has the carry_check template (min_frequency=5)
**And** 6 incorrect responses have been processed with violation_type "carry_error"
**When** `run_batch()` is called and we query `n_tier2_templates_active`
**Then** `n_tier2_templates_active >= 1` (the carry_check template has crossed its threshold)

### SCENARIO-LEARN-047: compute_learning_improvement Returns Improved When Batch4 > Batch1

**Given** a trajectory of 4 `SelfLearningBatchResult` objects with accuracies [0.60, 0.65, 0.70, 0.75]
**When** `compute_learning_improvement(trajectory)` is called
**Then** `batch1_accuracy == 0.60`
**And** `batch4_accuracy == 0.75`
**And** `improved == True`

### SCENARIO-LEARN-048: EORM Real-Data Retrain Achieves AUC Improvement Over Synthetic Baseline

**Given** an `EORMModel` loaded or freshly initialized as the baseline
**And** at least 50 real (question, response, correctness) pairs loaded from live experiment result files
**When** `run_experiment()` for Exp 371 is executed with those pairs
**Then** the artifact has `retrain_mode == "real_data"`
**And** `n_real_pairs >= 50`
**And** `auc_improvement == after_auc - before_auc` (signed, no clamping)
**And** `honest_verdict == "real_data_improvement"` when `auc_improvement > 0`
**And** `honest_verdict == "real_data_no_improvement"` when `auc_improvement <= 0`
**And** `honest_verdict == "insufficient_real_pairs"` when `n_real_pairs < 50`

Spec: REQ-LEARN-025, SCENARIO-LEARN-048

### SCENARIO-LEARN-049: SelfLearningRelay Live GPU Run Emits honest_verdict=no_improvement When Batch4 <= Batch1

**Given** a `SelfLearningRelay` running with `inference_mode="live_gpu"`
**And** batch4 accuracy is not strictly greater than batch1 accuracy
**When** `build_relay_artifact(trajectory, improvement, inference_mode="live_gpu")` is called
**Then** `honest_verdict == "no_improvement"`
**And** `inference_mode == "live_gpu"`

Spec: REQ-LEARN-027, SCENARIO-LEARN-047

### SCENARIO-LEARN-050: Live Self-Learning Relay Confirms Accuracy Improvement on Real GPU Inference

**Given** `CARNOT_FORCE_LIVE=1` is set in the environment
**And** Gemma4-E4B-it is loaded and healthy on GPU 0
**And** 100 GSM8K questions are loaded in 4 batches of 25
**And** the EORM model is loaded (from Exp 371 if available, else Exp 359 or fresh)
**When** `scripts/experiment_374_self_learning_relay_live.py` is executed
**Then** the artifact has `schema == "carnot.self_learning_relay.v2"`
**And** `inference_mode == "live_gpu"`
**And** `learning_trajectory` has 4 entries, one per batch
**And** each trajectory entry has `batch_id`, `n_questions==25`, `accuracy`, `n_tier1_updates==25`, `n_tier2_templates_active`, `tier3_gate_auc`
**And** `improved == (batch4_accuracy > batch1_accuracy)`
**And** `honest_verdict == "learning_confirmed"` when `improved==True AND inference_mode=="live_gpu"`
**And** `honest_verdict == "no_improvement"` when `improved==False`
**And** `eorm_source` is one of `["exp371_real", "exp359_real", "synthetic_fallback"]`

Spec: REQ-LEARN-026, REQ-LEARN-027, SCENARIO-LEARN-050

### SCENARIO-LEARN-066: Session 2 Loads Session 1 Templates

**Given** `simulate_session(session_id=0, questions=q0, prior_memory_path=None, memory_dir=d)` has completed
**And** `simulate_session(session_id=1, questions=q1, prior_memory_path=d/session_0, memory_dir=d)` is called
**When** the session completes
**Then** `CrossSessionResult.n_templates_loaded_from_prior > 0`
**And** the template library for session 1 started with the observation counts from session 0

Spec: REQ-LEARN-037, SCENARIO-LEARN-066

### SCENARIO-LEARN-067: SessionMemory Round-Trip Preserves Template Library

**Given** a `ConstraintTemplateLibrary` with observations for `("carry_check", "test-model")` >= `min_frequency`
**When** `SessionMemory.save()` and then `SessionMemory.load()` are called
**Then** the loaded library has identical observation counts to the original
**And** after `register_builtin_templates()`, `get_active_templates("test-model")` returns the `carry_check` template

Spec: REQ-LEARN-038, SCENARIO-LEARN-067

### SCENARIO-LEARN-068: compute_relay_verdict Returns insufficient_data With < 2 Sessions

**Given** `compute_relay_verdict([])` or `compute_relay_verdict([single_session])` is called
**When** the function evaluates the list length
**Then** it returns `"insufficient_data"`

Spec: REQ-LEARN-037, SCENARIO-LEARN-068

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

### REQ-EXTRACT-013: FP Categorization for Broken Verify-Repair Cases

The system shall support systematic categorization of false-positive violations that
caused verify-repair to hurt (produce a worse answer than baseline), where:
- A `FPCategory` enum defines exactly five categories:
  - `VALID_INTERMEDIATE`: the extractor flagged an arithmetic step that is correct
    in context (e.g. an intermediate result, not the final answer)
  - `PRECISION_LIMIT`: a correct computation was flagged because of rounding or
    floating-point approximation (e.g. `1/3 ≈ 0.33`)
  - `REGEX_ARTIFACT`: `ArithmeticExtractor` matched a wrong substring (e.g. a date,
    a phone number, or an incidental numeral that is not an arithmetic claim)
  - `REPAIR_DEGRADATION`: the violation was real but the repair step made the answer
    worse than leaving the original response unchanged
  - `UNCATEGORIZED`: none of the above patterns apply
- An `AutopsyCase` dataclass captures: `question`, `baseline_answer`,
  `vr_answer`, `correct_answer`, `violations_flagged` (list[str]),
  `fp_category` (FPCategory), `evidence` (str)
- `categorize_fp(case: AutopsyCase) -> FPCategory` inspects `violations_flagged`
  and `evidence` to assign one of the five categories deterministically
- `load_broken_cases(results_path: str) -> list[AutopsyCase]` loads cases where
  `vr_answer != correct_answer AND baseline_answer == correct_answer` from a
  fullscale benchmark result file; returns an empty list (not an error) when the
  file has no per-question data
- `compute_category_distribution(cases: list[AutopsyCase]) -> dict[FPCategory, int]`
  returns a count of cases per category

Spec: REQ-EXTRACT-013, SCENARIO-EXTRACT-027, SCENARIO-EXTRACT-028

### REQ-EXTRACT-014: FP Autopsy Corpus and Experiment Artifact

The system shall provide a deterministic FP autopsy workflow (`experiment_331`) that:
- Loads broken cases from Exp 328 live results (primary) or Exp 316 simulated
  fallback (when per-question data is absent from Exp 328)
- For each broken case, re-runs all three extractors (ArithmeticExtractor,
  LLMExtractor, NL2Z3Extractor) and records violation text and extractor output
- Emits an honest `inconclusive` artifact (not a failure) when `n_broken_cases < 5`
- Computes `primary_fp_type` as the most common FPCategory across all broken cases
- Maps `primary_fp_type` to a `recommended_fix`:
  - `VALID_INTERMEDIATE` → `"Confidence-weighted repair: add expression confidence filter"`
  - `PRECISION_LIMIT` → `"Model-adaptive threshold: increase ArithmeticExtractor tolerance"`
  - `REGEX_ARTIFACT` → `"NL2Z3 as primary: regex causes false positives on IT responses"`
  - `REPAIR_DEGRADATION` → `"Constrain repair: only accept if repaired energy < violation energy"`
  - `UNCATEGORIZED` → `"Manual review required: insufficient signal for automated fix"`
- Writes artifact to `results/experiment_331_fp_autopsy.json` with schema:
  `experiment=331`, `n_broken_cases`, `category_distribution`, `primary_fp_type`,
  `recommended_fix`, `sample_cases` (first 5, anonymized), `status`

Spec: REQ-EXTRACT-014, SCENARIO-EXTRACT-029, SCENARIO-EXTRACT-030

### SCENARIO-EXTRACT-027: FP Categorization Assigns VALID_INTERMEDIATE

**Given** an `AutopsyCase` where `violations_flagged` contains an expression that
  evaluates to a value that appears verbatim in the response as an intermediate step
  (e.g. "step 1: 10 - 3 = 7") and `correct_answer` is a different final value
**When** `categorize_fp(case)` is called
**Then** it returns `FPCategory.VALID_INTERMEDIATE`
**And** the `evidence` field records which expression triggered the flag

### SCENARIO-EXTRACT-028: FP Categorization Assigns REGEX_ARTIFACT

**Given** an `AutopsyCase` where `violations_flagged` contains a match whose
  `a`, `b`, `claimed_result` do not correspond to any arithmetic step in the
  question or response (e.g. a year like "2024 - 3 = 2021" matched incidentally)
**When** `categorize_fp(case)` is called
**Then** it returns `FPCategory.REGEX_ARTIFACT`

### SCENARIO-EXTRACT-029: Autopsy Emits Inconclusive When n < 5

**Given** a results file from which `load_broken_cases()` returns fewer than 5 cases
**When** `experiment_331_fp_autopsy.py` is executed
**Then** the artifact has `status="inconclusive"` and `n_broken_cases < 5`
**And** the artifact still contains `category_distribution` and `primary_fp_type`
  (which may be `UNCATEGORIZED`) rather than omitting these fields

### SCENARIO-EXTRACT-030: Autopsy Recommends Fix Matching Primary FP Type

**Given** an autopsy corpus where the majority of broken cases are `REGEX_ARTIFACT`
**When** `experiment_331_fp_autopsy.py` computes `recommended_fix`
**Then** `recommended_fix` equals
  `"NL2Z3 as primary: regex causes false positives on IT responses"`
**And** the artifact records `primary_fp_type="REGEX_ARTIFACT"`

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

### REQ-EXTRACT-015: CoT Circuit Extraction

The system shall support a circuit-based reasoning verifier (CoTCircuitVerifier, arXiv 2510.09312)
that extracts a computational dependency graph from a chain-of-thought response and checks
structural consistency, where:
- The response is parsed into a list of `CoTStep` objects: each step has a `step_id` (int),
  `text` (the raw step text), `input_refs` (list of step IDs this step depends on), and
  `output_value` (the last numeric result extracted from the step, as float or None)
- `is_final_answer` is True for the last step detected
- Step boundaries are detected via regex matching: "Step N:", numbered lines ("1."), and
  discourse markers ("First,", "Then,", "Next,", "Finally,")
- `input_refs` are populated by scanning each step for back-references ("from step N",
  "the result from step N", "(N)", "step N") pointing to earlier steps
- `output_value` is the last float/int found in the step text; None if no numeric result
  is present in the step

Spec: REQ-EXTRACT-015, SCENARIO-EXTRACT-031, SCENARIO-EXTRACT-032

### REQ-EXTRACT-016: CoT Structural Consistency Verification

The system shall detect structural inconsistencies (broken links) in a CoT dependency graph,
where:
- A `CoTCircuit` aggregates `steps` (list[CoTStep]), `has_cycle` (bool), and
  `broken_links` (list of (downstream_step_id, upstream_step_id, expected_value, actual_value))
- `has_cycle` is True when any step references a later step (logically impossible in a valid
  CoT — downstream steps cannot be inputs to upstream steps)
- A "broken link" exists when step i declares `input_refs=[j, ...]` and step j's `output_value`
  is not None but does not equal the value that step i uses for it (within `tolerance`)
- `find_broken_links(steps, tolerance)` returns a list of 4-tuples for each broken link:
  `(downstream_step_id, upstream_step_id, expected_value_str, actual_value_str)`
- `CoTCircuitVerifier(tolerance=0.01).verify(response)` returns a `CoTCircuit`
- `CoTCircuitVerifier.extract(question, response)` implements the `ConstraintExtractor`
  protocol: returns one `ConstraintResult(constraint_type="circuit_broken_link", ...)` per
  broken link; returns empty list when the circuit has no broken links
- Single-step responses produce a circuit with no broken links
- Steps with no numeric output_value are skipped in link checking

Spec: REQ-EXTRACT-016, SCENARIO-EXTRACT-033, SCENARIO-EXTRACT-034, SCENARIO-EXTRACT-035

### SCENARIO-EXTRACT-031: CoT Step Extraction Detects Step Boundaries

**Given** a multi-step response with boundaries marked "Step 1:", "Step 2:", "Step 3:"
**When** `extract_cot_steps(response)` is called
**Then** three `CoTStep` objects are returned, one per step
**And** `steps[2].is_final_answer` is True (last step)
**And** each step's `text` contains only the content of that step

### SCENARIO-EXTRACT-032: CoT Step Extraction Captures output_value and input_refs

**Given** a step containing "from step 1, we get 15. Adding 20 gives us 35."
  and step 1 had output_value=15.0
**When** `extract_cot_steps` parses this step
**Then** `output_value` is 35.0 (last numeric result in step)
**And** `input_refs` contains 1 (reference to step 1)

### SCENARIO-EXTRACT-033: Broken Link Detected When Step Uses Wrong Upstream Value

**Given** step 1 with output_value=10.0
**And** step 2 with input_refs=[1] and text using "from step 1 (12)"
  where 12 != 10 (outside tolerance)
**When** `build_circuit(steps)` is called
**Then** `circuit.broken_links` contains one entry:
  `(2, 1, "12", "10.0")` (downstream=2, upstream=1, expected=12, actual=10.0)

### SCENARIO-EXTRACT-034: Single-Step Response Produces Empty Broken Links

**Given** a response with only one step
**When** `CoTCircuitVerifier.verify(response)` is called
**Then** `circuit.broken_links` is empty
**And** `circuit.has_cycle` is False

### SCENARIO-EXTRACT-035: CoTCircuitVerifier.extract Returns ConstraintResult Per Broken Link

**Given** a response with two broken links
**When** `CoTCircuitVerifier.extract(question, response)` is called
**Then** two `ConstraintResult` objects are returned
**And** each has `constraint_type == "circuit_broken_link"`
**And** the `description` field names both the downstream and upstream step IDs
**And** a consistent response returns an empty list

### REQ-EXTRACT-017: Live Extractor Comparison Benchmark

The system shall support a live extractor comparison that runs all four extractor types
(ArithmeticExtractor, NL2Z3Extractor, VergeRefiner-as-extractor, CoTCircuitVerifier) on
the same set of live IT model responses and computes:
- Per-extractor violation_rate, estimated_precision, fp_category_distribution
- Pairwise agreement matrix: which extractor pairs flag the same responses
- A recommended_extractor: the extractor with highest estimated_precision (tie: first)

The comparison shall use the Exp 331 FP taxonomy (VALID_INTERMEDIATE, CORRECT_STEP,
PRECISION_LIMIT, REGEX_ARTIFACT, REPAIR_DEGRADATION) as the reference for precision
estimation. When live responses are unavailable, synthetic fallback responses are used
with inference_mode="simulated" clearly recorded in the artifact.

Spec: REQ-EXTRACT-017, SCENARIO-EXTRACT-036, SCENARIO-EXTRACT-037

### SCENARIO-EXTRACT-036: ExtractorResult Captures Per-Extractor Violation Statistics

**Given** a list of 50 responses and a set of four extractors
**When** `compare_extractors(responses, extractors)` is called
**Then** one `ExtractorResult` is returned per extractor
**And** each `ExtractorResult` has `extractor_name`, `n_responses_checked`,
  `n_violations_found`, `violation_rate`, `estimated_precision`, `fp_categories` (dict)
**And** `violation_rate` equals `n_violations_found / n_responses_checked`
**And** `estimated_precision` is in [0.0, 1.0]

### SCENARIO-EXTRACT-037: build_comparison_artifact Selects Recommended Extractor

**Given** a list of `ExtractorResult` objects with varying `estimated_precision` values
**When** `build_comparison_artifact(results)` is called
**Then** the artifact has `comparison_schema == "carnot.extractor_comparison.v1"`
**And** `best_precision` equals the maximum `estimated_precision` across all results
**And** `recommended_extractor` names the extractor with the highest `estimated_precision`
**And** when two extractors tie, the first one in the input list is chosen

### REQ-EXTRACT-018: LLM Extractor for Instruction-Tuned Format Responses

The system shall support an `LLMExtractor` that uses an auxiliary LLM call to canonicalize
arithmetic claims from instruction-tuned (IT) format responses (markdown, numbered steps,
mixed prose), where NL2Z3Extractor's single-pass approach fails due to format ambiguity.

Spec: REQ-EXTRACT-018, SCENARIO-EXTRACT-036, SCENARIO-EXTRACT-037

### REQ-EXTRACT-019: LLM-Guided Z3 Formalization

The system shall support an `LLMz3Formalizer` that separates the formalization step from
the extraction step, where:
- The LLM receives the raw response text and outputs ONLY Python z3-syntax assertion strings
  (not prose, not explanations)
- A second pass runs the Z3 snippet via `exec()` in a restricted sandbox
- The sandbox allows ONLY the `z3` module; attempts to import `os`, `sys`, `subprocess`
  or any other module raise `NameError`
- Results are captured in a `Z3FormalizationResult` dataclass with fields:
  `z3_code`, `z3_result`, `n_assertions`, `is_sat`, `formalization_mode`,
  `source_response_length`
- When `llm_caller` is `None` (CI mode), a hardcoded stub Z3 snippet is used for testing
  without any LLM call
- `formalization_mode` is `"llm"` when a real LLM caller is used, `"ci_stub"` otherwise

Inspired by arXiv 2601.04675 (LLM-guided SMT): 80% improvement in Z3 success rate when
an LLM rewrites ambiguous arithmetic into explicit Z3 assertion syntax.

Spec: REQ-EXTRACT-019, SCENARIO-EXTRACT-039, SCENARIO-EXTRACT-040, SCENARIO-EXTRACT-041

### REQ-EXTRACT-020: Zero-False-Positive Z3 Path

The system shall support a zero-false-positive verification path where:
- `LLMz3Formalizer` only reports a violation when Z3 returns `unsat`
- `sat` and `unknown` and `error` results do NOT produce violations
- This ensures FP rate is bounded by Z3 soundness (zero by design for correct Z3 models)
- The `fp_rate` field in the experiment artifact reflects this property

Spec: REQ-EXTRACT-020, SCENARIO-EXTRACT-039, SCENARIO-EXTRACT-041

### SCENARIO-EXTRACT-039: Z3FormalizationResult Captures Full Formalization Metadata

**Given** a `LLMz3Formalizer` with `llm_caller=None` (CI stub mode)
**When** `formalizer.formalize(question, response)` is called
**Then** the result is a `Z3FormalizationResult` instance
**And** `z3_code` is a non-empty Python string
**And** `z3_result` is one of `"sat"`, `"unsat"`, `"unknown"`, `"error"`
**And** `n_assertions` is a non-negative integer counting `s.add(...)` calls
**And** `is_sat` is `True` iff `z3_result == "sat"`
**And** `formalization_mode` is `"ci_stub"`
**And** `source_response_length` equals `len(response)`

### SCENARIO-EXTRACT-040: LLMz3Formalizer Sandbox Blocks Forbidden Imports

**Given** a `LLMz3Formalizer` with an `llm_caller` that returns code containing
  `import os` or `import sys` or `import subprocess`
**When** `formalizer.formalize(question, response)` is called
**Then** the exec sandbox raises `NameError` for the forbidden import
**And** the result has `z3_result == "error"` (sandbox blocked the code)
**And** no exception propagates to the caller

### SCENARIO-EXTRACT-041: LLMz3Formalizer Detects Arithmetic Contradiction

**Given** a `LLMz3Formalizer` with an `llm_caller` that returns Z3 code asserting
  `x == 5` and `x == 6` simultaneously
**When** `formalizer.formalize(question, "The answer is 5. Also the answer is 6.")` is called
**Then** `z3_result == "unsat"`
**And** `is_sat` is `False`
**And** `n_assertions >= 2`
**And** `formalization_mode == "llm"`

### REQ-EXTRACT-021: Comparative Extraction Benchmark

The system shall support a comparative extraction benchmark that:
- Runs multiple extractors (ArithmeticExtractor, LLMConstraintExtractor, LLMz3Formalizer)
  against the same set of LLM responses from a live instruction-tuned model
- For each response labelled as wrong: records whether the extractor detected a violation (TP)
- For each response labelled as correct: records whether the extractor raised a false alarm (FP)
- Computes `detection_rate = TP / (TP + FN)` and `false_positive_rate = FP / (FP + TN)`
  for each extractor
- Returns an `ExtractionBenchmarkResult` per extractor and a comparison artifact with:
  `winner`, `improvement_over_arithmetic_extractor`, `honest_verdict`
- `honest_verdict == "live_gpu_llm_extractor_wins"` only when `inference_mode == "live_gpu"`
  AND LLMConstraintExtractor detection_rate > ArithmeticExtractor detection_rate
- All other conditions yield a non-winning verdict string (e.g. `"simulated_no_verdict"`,
  `"live_gpu_no_improvement"`)

Spec: REQ-EXTRACT-021, SCENARIO-EXTRACT-042, SCENARIO-EXTRACT-043

### SCENARIO-EXTRACT-042: ExtractionBenchmarkResult Captures Per-Extractor Metrics

**Given** a set of 10 LLM responses (5 wrong, 5 correct) with known ground truth
**When** `run_extraction_benchmark(extractor, questions, ground_truth_wrong, inference_fn)`
  is called
**Then** the result is an `ExtractionBenchmarkResult` instance
**And** `n_questions == 10`
**And** `n_violations_found` is a non-negative integer <= `n_questions`
**And** `n_true_positives` is a non-negative integer <= 5
**And** `n_false_positives` is a non-negative integer <= 5
**And** `detection_rate` is a float in [0.0, 1.0]
**And** `inference_mode` is one of `"live_gpu"`, `"simulated"`
**And** `extractor_name` is a non-empty string

### SCENARIO-EXTRACT-043: build_extraction_comparison_artifact Enforces Honest Verdict

**Given** three `ExtractionBenchmarkResult` objects with `inference_mode="simulated"`
**When** `build_extraction_comparison_artifact(results)` is called
**Then** the returned dict has keys `winner`, `improvement_over_arithmetic_extractor`,
  `honest_verdict`, `per_extractor_results`
**And** `honest_verdict != "live_gpu_llm_extractor_wins"` (simulated mode cannot claim a win)

**Given** three `ExtractionBenchmarkResult` objects with `inference_mode="live_gpu"` and
  LLMConstraintExtractor having strictly higher detection_rate than ArithmeticExtractor
**When** `build_extraction_comparison_artifact(results)` is called
**Then** `honest_verdict == "live_gpu_llm_extractor_wins"`

### REQ-EXTRACT-023: Live Extraction Comparison Benchmark

The system shall support a live GPU extraction comparison benchmark that:
- Runs all three extractors (ArithmeticExtractor, LLMConstraintExtractor, LLMz3Formalizer)
  on live Gemma4-E4B-it responses to GSM8K questions
- Uses `inference_mode="live_gpu"` exclusively when `CARNOT_FORCE_LIVE=1`; produces a
  blocked artifact and stops if live GPU is unavailable
- Captures per-extractor metrics in `ExtractorComparisonResult` (extractor_name,
  n_questions, n_correct_questions, n_wrong_questions, n_true_positives, n_false_positives,
  detection_rate, fp_rate, inference_mode)
- Determines an `honest_verdict`:
  - `"live_gpu_winner"` only when ALL results have `inference_mode="live_gpu"`
  - `"simulated_no_verdict"` when ANY result has `inference_mode != "live_gpu"`
- Uses `build_extractor_comparison_artifact(results)` to produce a dict with
  schema `"carnot.extraction_comparison.v1"`, winner selected by highest detection_rate
  with fp_rate as tiebreak

Spec: REQ-EXTRACT-023, SCENARIO-EXTRACT-047, SCENARIO-EXTRACT-048

### SCENARIO-EXTRACT-047: ExtractorComparisonResult Captures Live Per-Extractor Metrics

**Given** an extraction run in `inference_mode="live_gpu"` on 30 GSM8K questions
**When** `run_extractor_comparison()` produces an `ExtractorComparisonResult` per extractor
**Then** each result has `n_questions`, `n_correct_questions`, `n_wrong_questions`,
  `n_true_positives`, `n_false_positives`, `detection_rate`, `fp_rate`, `inference_mode`
**And** `n_correct_questions + n_wrong_questions == n_questions`
**And** `inference_mode == "live_gpu"`

### SCENARIO-EXTRACT-048: build_extractor_comparison_artifact Enforces Live Honest Verdict

**Given** a list of `ExtractorComparisonResult` objects
**When** all results have `inference_mode="live_gpu"`
**Then** `build_extractor_comparison_artifact(results)["honest_verdict"]` may be
  `"live_gpu_winner"` if a clear winner exists, or `"live_gpu_no_winner"` if tied
**When** any result has `inference_mode != "live_gpu"`
**Then** `honest_verdict == "simulated_no_verdict"` (simulated mode cannot claim a win)
**And** `schema == "carnot.extraction_comparison.v1"`

### REQ-EXTRACT-024: VeriCoT FOL Premise Extraction From Natural-Language CoT Steps

The system shall provide a `VeriCoTStepValidator` in
`python/carnot/extraction/vericot_validator.py` that extracts First-Order Logic (FOL)
premises from natural-language Chain-of-Thought reasoning steps, where:
- Each reasoning step is parsed to extract arithmetic claims written in free-form prose
  (for example "the total is 47 plus 28, which gives 75") rather than equation format
- Each extracted claim is represented as a `FOLPremise(expression, source_step)` where
  `expression` is a Z3-compatible assertion string
- `FOLPremise.to_z3_assertion()` converts the expression into a `z3.BoolRef` or `None`
  when the expression cannot be formalized
- In production mode (`use_mock=False`) a small LLM (default `Qwen/Qwen3.5-0.8B`) is
  prompted to emit Z3-compatible assertion strings; in test mode (`use_mock=True`) a
  rule-based extractor is used without any model call

**Why FOL formalization beats regex (IT model incompatibility):** Instruction-tuned models
write arithmetic in prose like "47 plus 28, which gives 75" — regex patterns tuned for
"47 + 28 = 75" find zero matches on these traces. Formalizing each step into FOL lets us
delegate the prose-to-formula translation to an LLM that understands natural language, then
hand the formalized constraints to a sound solver that cannot be fooled by surface wording.

Spec: REQ-EXTRACT-024, SCENARIO-EXTRACT-049, SCENARIO-EXTRACT-050

### REQ-EXTRACT-025: Z3 UNSAT As VeriCoT Violation Signal

The system shall provide Z3-backed consistency checking for each extracted `FOLPremise`
list, where:
- `VeriCoTStepValidator.verify_step(step_text)` returns a `StepVerdict` with `status`
  in `{'sat', 'unsat', 'unknown'}`
- The solver adds all `FOLPremise.to_z3_assertion()` results for a step as conjunction
- `status='unsat'` means the conjunction of premises is unsatisfiable — the step contains
  an internal contradiction that signals an arithmetic error
- `status='sat'` means the step is internally consistent (no violation detected)
- `status='unknown'` is returned when no premises could be extracted from the step
- `detect_violations(cot_text)` splits a full CoT into steps and returns the list of
  `StepVerdict` objects whose `status='unsat'`

**Why Z3 (not a neural judge):** Z3 is a sound SMT solver — it cannot hallucinate a
"violation" on a correct constraint. A neural judge can be reward-hacked or produce
inconsistent verdicts across runs. Z3's UNSAT guarantee is deterministic: if it says
UNSAT, the premises are provably inconsistent given the arithmetic axioms, every time.

Spec: REQ-EXTRACT-025, SCENARIO-EXTRACT-051

### REQ-EXTRACT-026: VeriCoT Detection Improvement Over ArithmeticExtractor

The system shall demonstrate that `VeriCoTStepValidator` detects violations in
instruction-tuned model outputs that `ArithmeticExtractor` misses, where:
- `ArithmeticExtractor` returns zero violations on IT-model natural-language reasoning
  traces (no "A + B = C" patterns present)
- `VeriCoTStepValidator(use_mock=True).detect_violations(cot_text)` returns at least
  one violation on the same traces when they contain arithmetic errors
- Experiment 453 runs this comparison on 20 hardcoded IT-style samples (10 correct,
  10 with arithmetic errors) and records `baseline_detected`, `vericot_detected`,
  `improvement_rate`, and `honest_verdict` in
  `results/experiment_453_vericot_validator.json`

Spec: REQ-EXTRACT-026, SCENARIO-EXTRACT-049, SCENARIO-EXTRACT-050, SCENARIO-EXTRACT-051

### SCENARIO-EXTRACT-049: VeriCoTStepValidator Detects Error In IT Natural-Language Step

**Given** the step text "the total is 47 plus 28, which gives 76"
**When** `VeriCoTStepValidator(use_mock=True).verify_step(step_text)` is called
**Then** `verdict.status == 'unsat'`
**And** `len(verdict.fol_premises) >= 1`
**And** `ArithmeticExtractor().extract(step_text)` returns an empty list

### SCENARIO-EXTRACT-050: VeriCoTStepValidator Returns SAT For Correct IT Step

**Given** the step text "the total is 47 plus 28, which gives 75"
**When** `VeriCoTStepValidator(use_mock=True).verify_step(step_text)` is called
**Then** `verdict.status == 'sat'`
**And** `ArithmeticExtractor().extract(step_text)` returns an empty list

### SCENARIO-EXTRACT-051: Exp 453 Improvement Rate Is Positive

**Given** 20 IT-style reasoning samples (10 correct, 10 with arithmetic errors)
**When** Exp 453 runs ArithmeticExtractor and VeriCoTStepValidator(use_mock=True) on all 20
**Then** `baseline_detected == 0` (ArithmeticExtractor finds no violations)
**And** `vericot_detected >= 1` (VeriCoT finds at least one violation)
**And** `honest_verdict == 'vericot_better'`

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

### REQ-REPAIR-012: VERGE-Style Iterative Z3 Refinement

The system shall provide a `VergeRefiner` component that implements iterative SMT-guided
step-level repair of chain-of-thought responses, inspired by VERGE (arXiv 2601.20055):

- Initial Z3 check: if SAT, return immediately with no iterations (response is consistent).
- On UNSAT: extract the specific failed assertion from the Z3 code using
  `extract_failed_assertion(z3_result)`.
- Build a targeted repair prompt asking the LLM to correct ONLY the step that produced the
  failed assertion (`build_step_repair_prompt`).
- Patch the response with the repaired step and re-run Z3 on the patched response.
- Repeat until SAT or `max_iterations` reached.
- Each iteration is recorded as a `VergeIteration` dataclass capturing: `iteration_n`,
  `assertion_failed`, `step_text`, `repair_prompt`, `repaired_step`, `new_z3_result`,
  `resolved` (bool — True when Z3 reached SAT on this iteration).
- Return `(final_response, iteration_log)` from `refine(question, response)`.

Spec: REQ-REPAIR-012, SCENARIO-REPAIR-024, SCENARIO-REPAIR-025

### REQ-REPAIR-013: Step Isolation for Targeted Repair

The system shall provide `extract_failed_assertion(z3_result: Z3Result) -> str | None`
that identifies the specific assertion in the Z3 code that caused UNSAT:

- Parse z3_code for `s.add(...)` calls using regex; return the first match.
- Fallback: if no `s.add(...)` found, scan for any `assert` keyword or `z3.solve(...)` args.
- If z3_code is empty or no assertion found, return `None`.
- When called with a SAT or non-unsat result, return `None` (no failure to report).
- The returned string is the assertion body text (what was inside `s.add(...)`).

Spec: REQ-REPAIR-013, SCENARIO-REPAIR-026, SCENARIO-REPAIR-027

### REQ-REPAIR-014: BoltzmannRepairBridge Maps Ising Ground-State to LLM Repair Direction

The BoltzmannRepairBridge shall map a low-energy Ising spin configuration (the "ground
state" found by simulated annealing) to a repair direction in LLM embedding space via a
trained linear adapter.

**Rationale (Boltzmann-GPT arXiv 2601.17094):**
The DBM world model and the LLM live in separate representation spaces. The adapter
bridges them: it is trained on (spin_config, target_embedding) pairs so that low-energy
spin configurations project to embeddings that favour constraint-satisfying outputs.
This replaces the naive "ask LLM to fix error" step with an energy-guided direction.

**Hardware path:** The adapter is a single matrix multiply — GPU/NPU native, <1ms.

Spec: REQ-REPAIR-014, SCENARIO-REPAIR-028, SCENARIO-REPAIR-029, SCENARIO-REPAIR-030

### REQ-REPAIR-015: Repair Direction Reduces Ising Energy on Held-Out Constraint Instances

The BoltzmannRepairBridge shall produce repair directions whose Ising energy
(energy_after) is less than or equal to the energy of the original constraint
state (energy_before) for at least 60% of sampled constraint instances.

**Rationale:** If energy_after >= energy_before systematically, the bridge adds no value
over random repair. The 60% threshold is the minimum bar for "energy-guided" repair to
be distinguishable from noise. Honest verdict is written based on observed rate.

Spec: REQ-REPAIR-015, SCENARIO-REPAIR-029

### SCENARIO-REPAIR-028: BoltzmannRepairBridge Returns RepairDirection Dataclass

**Given** a BoltzmannRepairBridge constructed from a 16-variable IsingModel and a
  trained LinearSpinAdapter(spin_dim=16, embed_dim=128)
**When** `bridge.get_repair_direction(constraint_state)` is called with a dict
  containing at least one constraint violation
**Then** the returned object is a RepairDirection with fields:
  - `spin_config`: jnp.ndarray of shape (16,) with values in {-1.0, +1.0}
  - `embedding_projection`: jnp.ndarray of shape (128,)
  - `energy_before`: float
  - `energy_after`: float

### SCENARIO-REPAIR-029: Energy After Is Less Than Or Equal To Baseline For Random State

**Given** a BoltzmannRepairBridge with a 16-variable IsingModel (seed=42)
**When** `get_repair_direction` is called 100 times with random constraint states
**Then** `energy_after <= energy_before` for all 100 calls
  (because energy_after is the low-energy Ising ground state; energy_before is
  a random starting configuration — ground state is always at or below random)

### SCENARIO-REPAIR-030: LinearSpinAdapter Trains Without Error

**Given** a LinearSpinAdapter(spin_dim=16, embed_dim=128)
**And** 50 synthetic (spin_config, target_embedding) training pairs
**When** `adapter.train(spin_configs, target_embeddings, n_epochs=50)` is called
**Then** the returned MSE loss is a non-negative finite float
**And** `adapter.project(spin_config)` returns a jnp.ndarray of shape (128,) for
  any input spin_config of shape (16,) with values in {-1.0, +1.0}

### SCENARIO-REPAIR-024: VERGE SAT Fast Path

**Given** a question and response where the Z3 check immediately returns SAT
**When** `VergeRefiner.refine(question, response)` is called
**Then** the returned iteration log is empty (no refinement iterations needed)
**And** the returned final response is identical to the input response

### SCENARIO-REPAIR-025: VERGE Iterative Repair Converges to SAT

**Given** a question and a response whose Z3 check returns UNSAT on the first check
**And** a mock LLM caller that returns a corrected step
**And** the re-verification of the patched response returns SAT
**When** `VergeRefiner.refine(question, response)` is called
**Then** exactly one `VergeIteration` is returned
**And** `iteration.resolved == True`
**And** `iteration.new_z3_result.sat_status == "sat"`
**And** the returned final response contains the repaired step text

### SCENARIO-REPAIR-026: extract_failed_assertion Returns First s.add() Body

**Given** a Z3Result with `sat_status="unsat"` and `z3_code` containing `s.add(x + y == 10)`
**When** `extract_failed_assertion(z3_result)` is called
**Then** the returned string is `"x + y == 10"` (the assertion body)

### SCENARIO-REPAIR-027: extract_failed_assertion Returns None for SAT

**Given** a Z3Result with `sat_status="sat"` (no contradiction)
**When** `extract_failed_assertion(z3_result)` is called
**Then** `None` is returned (no failed assertion to surface)

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

### REQ-BENCH-002: Live GPU Benchmark Result With Provenance

The system shall provide a live GPU benchmark execution wrapper (`experiment_328`) that:

- Executes `experiment_315_fullscale_benchmark.py` with `CARNOT_FORCE_LIVE=1` and records
  results with `inference_mode="live_gpu"` only — results with any other mode are rejected
- Validates loaded results via `validate_live_result(result)` which raises `ValueError`
  if `inference_mode != "live_gpu"`
- Computes simulation divergence: `compare_to_simulated(live, simulated)` → per-model,
  per-mode accuracy delta between live result and Exp 316 simulated result
- Computes baseline deviation: `compare_to_published_baseline(result, baselines)` →
  per-model deviation from published accuracy (Qwen3.5-0.8B: 0.25, Gemma4-E4B-it: 0.80)
- Assesses whether each model is within acceptable range via
  `within_expected_range(accuracy, baseline, tolerance=0.15)` → bool
- Emits a wrapper artifact with schema `carnot.live_fullscale_benchmark.v1` containing:
  `experiment=328`, `inference_mode="live_gpu"`, `simulation_divergence`,
  `baseline_deviation`, `primary_result_path`
- If GPU is unavailable (OOM, driver error, insufficient VRAM), emits an honest
  `status="blocked"` artifact with `gpu_diagnostics` dict — does NOT fall back to
  simulation and label it as live
- Results written to `results/experiment_328_live_fullscale_results.json`

### SCENARIO-BENCH-003: Live Result Rejected If Not Live GPU

**Given** a benchmark artifact with `inference_mode="simulated"`
**When** `validate_live_result(result)` is called
**Then** a `ValueError` is raised with a message indicating the mode is not "live_gpu"

**Given** a benchmark artifact with `inference_mode="live_gpu"`
**When** `validate_live_result(result)` is called
**Then** no exception is raised

### SCENARIO-BENCH-004: Simulation Divergence And Baseline Deviation Computed Correctly

**Given** a live result with Qwen3.5-0.8B baseline accuracy 0.27 and a simulated
  result with Qwen3.5-0.8B baseline accuracy 0.34
**When** `compare_to_simulated(live, simulated)` is called
**Then** the returned dict contains `Qwen3.5-0.8B.baseline.all.delta == 0.27 - 0.34 == -0.07`

**Given** a live result with Qwen3.5-0.8B baseline accuracy 0.27 and
  `published_baselines = {"Qwen3.5-0.8B": 0.25}`
**When** `compare_to_published_baseline(live, published_baselines)` is called
**Then** the returned dict contains `Qwen3.5-0.8B.deviation == 0.02`
**And** `within_expected_range(0.27, 0.25, tolerance=0.15)` returns `True`

### REQ-BENCH-003: Live Full Precision Pipeline Benchmark

The system shall provide a live benchmark script (`experiment_340`) that measures the
combined effect of the full precision stack (Exp 332–336 components) on real LLM output
from instruction-tuned models on GSM8K, where:

- **PrecisionStackResult** dataclass captures per-variant benchmark data with fields:
  `model_id`, `n_questions`, `baseline_accuracy`, `precision_stack_accuracy`,
  `signed_improvement`, `pipeline_variant`, `inference_mode`
- **compute_signed_improvement(baseline_acc, stack_acc) → float**: returns the honest
  signed delta (stack_acc − baseline_acc) without clamping — negative values are preserved
- **PipelineVariant** enum has five members: `BASELINE`, `CONFIDENCE_ONLY`,
  `CONFIDENCE_ADAPTIVE`, `CONFIDENCE_ADAPTIVE_VERGE`, `FULL_STACK`
- **build_precision_benchmark_artifact(results: list[PrecisionStackResult]) → dict**:
  produces `precision_schema="carnot.precision_benchmark.v1"`, `headline_result` from the
  FULL_STACK result for Gemma4-E4B-it, and `inference_mode` from the result list
- **headline_result**: the `signed_improvement` of `FULL_STACK` on Gemma4-E4B-it;
  if `signed_improvement > 0`, `headline_label` is set to `"first_positive_live_it_result"`
- **CI-safe simulated mode**: when `CARNOT_FORCE_LIVE=0`, all pipeline variants produce
  `inference_mode="simulated"` with `honest_verdict="simulated_only"` in the artifact

Spec: REQ-BENCH-003, SCENARIO-BENCH-007, SCENARIO-BENCH-008, SCENARIO-BENCH-009

### REQ-BENCH-004: Live HumanEval Code Verification Benchmark

The system shall provide a live benchmark script (`experiment_341`) that measures
Carnot's code verification pipeline on HumanEval-style problems with Gemma4-E4B-it,
where:

- **HumanEvalResult** dataclass captures per-problem benchmark data with fields:
  `problem_id`, `generated_code`, `passed_tests`, `violations_found`,
  `repair_attempted`, `final_code`, `final_passed_tests`
- **compute_pass_at_1(results: list[HumanEvalResult]) → float**: returns the fraction
  of problems where `passed_tests=True` before any repair (baseline pass@1)
- **compute_pass_at_1_after_repair(results: list[HumanEvalResult]) → float**: returns the
  fraction of problems where `final_passed_tests=True` after the verify-repair loop
- **build_humaneval_artifact(results: list[HumanEvalResult], inference_mode: str) → dict**:
  produces `humaneval_schema="carnot.humaneval_benchmark.v1"`, a `headline_improvement` field
  (pass@1_after_repair − pass@1_before_repair), and `headline_label`
- **headline_label**: if `headline_improvement > 0`, set to `"code_verification_positive"`;
  otherwise `"no_improvement"`
- **CI-safe simulated mode**: when `CARNOT_FORCE_LIVE=0`, use synthetic code snippets
  (no LLM call); every result has `inference_mode="simulated"` in the artifact
- **Live mode**: when `CARNOT_FORCE_LIVE=1`, use Gemma4-E4B-it for code generation
  and run the full CodeExtractor + VerifyRepairPipeline on each problem

Spec: REQ-BENCH-004, SCENARIO-BENCH-010, SCENARIO-BENCH-011

### SCENARIO-BENCH-010: HumanEvalResult Dataclass And compute_pass_at_1

**Given** a `HumanEvalResult` is constructed with `passed_tests=True` and
  `final_passed_tests=True`
**When** `compute_pass_at_1([result])` is called
**Then** the result is `1.0`

**Given** two results where one has `passed_tests=True` and one has `passed_tests=False`
**When** `compute_pass_at_1(results)` is called
**Then** the result is `0.5`

**Given** a result with `passed_tests=False` and `final_passed_tests=True`
**When** `compute_pass_at_1_after_repair([result])` is called
**Then** the result is `1.0`

### SCENARIO-BENCH-011: build_humaneval_artifact Headline And CI-Safe Mode

**Given** results where pass@1_before=0.5 and pass@1_after=0.6
**When** `build_humaneval_artifact(results, "live_gpu")` is called
**Then** the artifact has `humaneval_schema="carnot.humaneval_benchmark.v1"`
**And** `headline_improvement` is approximately `0.1`
**And** `headline_label == "code_verification_positive"`

**Given** results where pass@1_before=0.6 and pass@1_after=0.5
**When** `build_humaneval_artifact(results, "live_gpu")` is called
**Then** `headline_label != "code_verification_positive"`

**Given** `CARNOT_FORCE_LIVE` is not set (or set to `"0"`)
**When** the experiment runs all problems
**Then** the artifact contains `inference_mode="simulated"`
**And** every result uses synthetic code snippets without calling any LLM

### REQ-BENCH-005: Live GPU Smoke Test Gate

Before running any benchmark experiment, the system SHALL provide a minimal live
GPU smoke test that:
- Runs N questions (default 5) through a specified model with `CARNOT_FORCE_LIVE=1`
- Returns a `SmokeTestResult` dataclass recording `inference_mode`, `n_questions`,
  `n_answered`, `elapsed_s`, `model_id`, `is_live`, and `blocked_reason`
- Raises `RuntimeError` if `CARNOT_FORCE_LIVE=1` and `inference_mode != "live_gpu"`
  (explicit fail, not silent fallback to simulated)
- Returns `SmokeTestResult(is_live=False, inference_mode="ci_skip",
  blocked_reason="CARNOT_FORCE_LIVE not set")` when `CARNOT_FORCE_LIVE` is not set
- Produces a `build_smoke_test_artifact(result)` with `schema="carnot.smoke_test.v1"`
  and `honest_verdict` of `"live_confirmed"`, `"blocked_simulated"`, or
  `"blocked_error"`

Spec: REQ-BENCH-005, SCENARIO-BENCH-012, SCENARIO-BENCH-013

### SCENARIO-BENCH-012: SmokeTestResult Dataclass And CI-Skip Path

**Given** `CARNOT_FORCE_LIVE` is not set in the environment
**When** `run_smoke_test("google/gemma-4-E4B-it", n_questions=5)` is called
**Then** it returns a `SmokeTestResult` with `is_live=False`
**And** `inference_mode="ci_skip"`
**And** `blocked_reason="CARNOT_FORCE_LIVE not set"`
**And** no `RuntimeError` is raised

**Given** a `SmokeTestResult` with `is_live=False` and `inference_mode="ci_skip"`
**When** `build_smoke_test_artifact(result)` is called
**Then** the artifact has `schema="carnot.smoke_test.v1"`
**And** `honest_verdict="blocked_simulated"`

### SCENARIO-BENCH-013: Live GPU Smoke Test Raises On Simulated Fallback

**Given** `CARNOT_FORCE_LIVE=1` is set in the environment
**And** the model pre-warm fails (GPU unavailable or model not loadable)
**When** `run_smoke_test("google/gemma-4-E4B-it", n_questions=5)` is called
**Then** a `RuntimeError` is raised
**And** the error message describes the failure layer

**Given** a `SmokeTestResult` with `is_live=True` and `inference_mode="live_gpu"`
**When** `build_smoke_test_artifact(result)` is called
**Then** the artifact has `honest_verdict="live_confirmed"`

### SCENARIO-BENCH-007: PrecisionStackResult Dataclass And compute_signed_improvement

**Given** `baseline_accuracy=0.50` and `precision_stack_accuracy=0.65`
**When** `compute_signed_improvement(0.50, 0.65)` is called
**Then** the result is `0.15` (positive improvement)

**Given** `baseline_accuracy=0.65` and `precision_stack_accuracy=0.60`
**When** `compute_signed_improvement(0.65, 0.60)` is called
**Then** the result is `-0.05` (negative — no clamping; honest)

**Given** a `PrecisionStackResult` is constructed with all required fields
**When** `signed_improvement` is set to `0.10`
**Then** the dataclass is accessible and serializable to dict via `dataclasses.asdict()`

### SCENARIO-BENCH-008: build_precision_benchmark_artifact Headline And Schema

**Given** a list of `PrecisionStackResult` objects including one with
  `pipeline_variant=PipelineVariant.FULL_STACK`, `model_id="Gemma4-E4B-it"`,
  and `signed_improvement=0.08`
**When** `build_precision_benchmark_artifact(results)` is called
**Then** the artifact has `precision_schema="carnot.precision_benchmark.v1"`
**And** `headline_result.signed_improvement == 0.08`
**And** `headline_result.headline_label == "first_positive_live_it_result"`

**Given** a `FULL_STACK` result for Gemma4-E4B-it with `signed_improvement=-0.03`
**When** `build_precision_benchmark_artifact(results)` is called
**Then** `headline_result` does NOT contain `headline_label == "first_positive_live_it_result"`

### SCENARIO-BENCH-009: CI-Safe Simulated Mode With Honest Verdict

**Given** `CARNOT_FORCE_LIVE` is not set (or set to `"0"`)
**When** the experiment runs all five pipeline variants
**Then** every `PrecisionStackResult` has `inference_mode="simulated"`
**And** the artifact contains `honest_verdict="simulated_only"`

**Given** a list of results all with `inference_mode="simulated"`
**When** `build_precision_benchmark_artifact(results)` is called
**Then** the artifact's `inference_mode` is `"simulated"`

### REQ-BENCH-006: Adversarial GSM8K Benchmark Harness

The repository shall provide a harness for the Apple adversarial GSM8K benchmark
(arXiv 2410.05229) that:
- Loads or synthesizes a set of GSM8K math questions
- Appends one irrelevant distractor sentence per question from a fixed pool of 20
  distractors, using a seeded PRNG for reproducibility
- Runs each question through Carnot's ArithmeticExtractor-based verify-repair pipeline
  under both standard (original) and adversarial (distractor-appended) conditions
- Reports standard_accuracy, adversarial_accuracy, accuracy_drop, and
  repaired_adversarial_accuracy to measure whether structural parsing is robust to
  irrelevant-sentence noise

The hypothesis: because ArithmeticExtractor parses explicit equation tokens — not
context words — the Ising energy should be invariant to appended irrelevant text,
making Carnot immune to the ≤65% accuracy drop observed in frontier LLMs.

Spec: REQ-BENCH-006, SCENARIO-BENCH-014, SCENARIO-BENCH-015, SCENARIO-BENCH-016

### REQ-BENCH-007: Irrelevant-Sentence Robustness Invariant

The adversarial GSM8K result artifact shall include a boolean
`robustness_invariant_holds` field that is `True` when
`adversarial_accuracy >= standard_accuracy - 0.05`, indicating that Carnot's
structural parsing did not degrade meaningfully under distractor-sentence injection.
This invariant is the primary research claim for the Apple adversarial GSM8K result.

Spec: REQ-BENCH-007, SCENARIO-BENCH-014, SCENARIO-BENCH-016

### SCENARIO-BENCH-014: AdversarialGSMQuestion Dataclass And build_adversarial_questions

**Given** a list of `n` GSM8K question strings and a fixed `seed=42`
**When** `build_adversarial_questions(original_questions, seed=42)` is called
**Then** it returns a list of `n` `AdversarialGSMQuestion` objects
**And** each `adversarial_question` contains the original question text
**And** each `adversarial_question` ends with one sentence from `DISTRACTOR_SENTENCES`
**And** the same seed always produces the same assignment of distractors to questions

**Given** `AdversarialGSMQuestion` is constructed with all five required fields
**When** `dataclasses.asdict()` is called on the result
**Then** a dict is returned with keys `question_id`, `original_question`,
  `adversarial_question`, `ground_truth_answer`, and `irrelevant_sentence`

### SCENARIO-BENCH-015: AdversarialBenchmarkResult And compute_adversarial_results

**Given** `standard_results` with 8/10 correct, `adversarial_results` with 6/10 correct,
  `repaired_results` with 7/10 correct
**When** `compute_adversarial_results(standard_results, adversarial_results, repaired_results)`
  is called
**Then** the returned `AdversarialBenchmarkResult` has
  `standard_accuracy=0.80`, `adversarial_accuracy=0.60`, `accuracy_drop=0.20`,
  `repaired_adversarial_accuracy=0.70`, `repair_improvement=0.10`

**Given** `AdversarialBenchmarkResult` constructed with all required fields
**When** `dataclasses.asdict()` is called
**Then** a dict is returned with all six fields present

### SCENARIO-BENCH-016: build_adversarial_artifact Schema And Honest Verdict

**Given** an `AdversarialBenchmarkResult` with `inference_mode="simulated"`
**When** `build_adversarial_artifact(result)` is called
**Then** the artifact has `schema="carnot.adversarial_gsm8k.v1"`
**And** `honest_verdict` is one of `"improvement_positive"`, `"degradation_positive"`,
  or `"blocked_simulated"`
**And** when `inference_mode="simulated"` the `honest_verdict` is `"blocked_simulated"`

**Given** an `AdversarialBenchmarkResult` with `inference_mode="live_gpu"` and
  `repair_improvement > 0`
**When** `build_adversarial_artifact(result)` is called
**Then** `honest_verdict="improvement_positive"`

**Given** an `AdversarialBenchmarkResult` with `inference_mode="live_gpu"` and
  `repair_improvement <= 0` and `accuracy_drop > 0`
**When** `build_adversarial_artifact(result)` is called
**Then** `honest_verdict="degradation_positive"`

### SCENARIO-BENCH-017: run_adversarial_benchmark CI-Safe Mode

**Given** `CARNOT_FORCE_LIVE` is not set (or set to any value other than `"1"`)
**When** `run_adversarial_benchmark(model_id, questions, pipeline, batch_size=8)` is called
**Then** it returns an `AdversarialBenchmarkResult` with `inference_mode="simulated"`
**And** `honest_verdict` in the resulting artifact is `"blocked_simulated"`
**And** no live model loading or GPU inference is attempted

### SCENARIO-BENCH-018: run_adversarial_benchmark honest_verdict Logic

**Given** a completed `AdversarialBenchmarkResult` with `inference_mode="live_gpu"` and
  `repair_improvement > 0`
**When** `build_adversarial_artifact(result)` is called
**Then** `honest_verdict` is `"improvement_positive"`

**Given** a completed `AdversarialBenchmarkResult` with `inference_mode` other than `"live_gpu"`
**When** `build_adversarial_artifact(result)` is called
**Then** `honest_verdict` is `"blocked_simulated"` (never `"improvement_positive"`)

**Given** a completed `AdversarialBenchmarkResult` with `inference_mode="live_gpu"` and
  `repair_improvement <= 0` and `accuracy_drop <= 0`
**When** `build_adversarial_artifact(result)` is called
**Then** `honest_verdict` is `"neutral"`

### SCENARIO-BENCH-019: Exp 355 Per-Model Results And Headline Artifact

**Given** Exp 355 runs with N questions per model across standard + adversarial + repair conditions
**When** the artifact is built with `schema="carnot.adversarial_gsm8k.v1"`
**Then** the artifact contains a `per_model_results` list with one entry per model
**And** each entry contains `model_id`, `standard_accuracy`, `adversarial_accuracy`,
  `accuracy_drop`, `repaired_adversarial_accuracy`, `repair_improvement`, `n_questions`
**And** the artifact contains a `headline_result` dict with aggregate `honest_verdict`
**And** `honest_verdict` at top level is `"improvement_positive"` only when
  `repair_improvement > 0` AND `inference_mode == "live_gpu"` for at least one model

### SCENARIO-BENCH-020: Exp 368 Live GPU Confirmed Criterion

**Given** Exp 368 runs the precision pipeline benchmark with `CARNOT_FORCE_LIVE=1`
  and `diagnose_live_gpu` returns `is_live_capable=True`
**When** the artifact is built
**Then** `inference_mode == "live_gpu"` in the artifact
**And** `honest_verdict == "live_improvement"` iff `signed_improvement > 0`
  AND `inference_mode == "live_gpu"` for the headline FULL_STACK Gemma4-E4B-it result
**And** if `diagnose_live_gpu` returns `is_live_capable=False`, a blocked artifact
  is emitted immediately with `status="blocked"` and `inference_mode` set to the
  first failure layer name — the experiment MUST NOT fall through to simulated mode

Spec: REQ-BENCH-003, SCENARIO-BENCH-020

### SCENARIO-BENCH-021: Exp 369 Live HumanEval Code Verification Pass@1 After Repair

**Given** Exp 369 runs with `CARNOT_FORCE_LIVE=1` and `diagnose_live_gpu` returns
  `is_live_capable=True`
**When** Gemma4-E4B-it generates solutions for 50 HumanEval problems and the
  CodeExtractor + VerifyRepairPipeline runs on all initially-failing solutions
**Then** the artifact has `schema="carnot.humaneval_benchmark.v2"` and
  `inference_mode="live_gpu"`
**And** `honest_verdict == "code_verification_positive"` iff
  `inference_mode == "live_gpu"` AND `signed_improvement > 0`
**And** if `diagnose_live_gpu` returns `is_live_capable=False`, a blocked artifact
  is emitted immediately with `status="blocked"` — the experiment MUST NOT fall
  through to simulated mode
**And** `pbt_bugs_found` is an integer recording how many solutions that passed
  official tests were subsequently found buggy by property-based testing

Spec: REQ-BENCH-004, SCENARIO-BENCH-021

### SCENARIO-BENCH-022: Exp 370 Live-Confirmed Adversarial GSM8K Criterion

**Given** Exp 370 runs the adversarial GSM8K benchmark with `CARNOT_FORCE_LIVE=1`
  and `diagnose_live_gpu()` confirms at least one GPU is live-capable
**When** the artifact is built with `schema="carnot.adversarial_gsm8k.v2"`
**Then** `inference_mode == "live_gpu"` in the artifact
**And** `honest_verdict` is one of `"improvement_positive"`, `"degradation_positive"`,
  or `"neutral"` — never `"blocked_simulated"`
**And** if `diagnose_live_gpu()` raises `RuntimeError` (GPU not live), the experiment
  raises immediately — it MUST NOT silently fall through to simulated mode
**And** the artifact contains `per_model_results` with at minimum one entry for
  Gemma4-E4B-it and one for Qwen3.5-0.8B
**And** the repair condition uses `LLMConstraintExtractor` with live Qwen3.5-0.8B
  rather than regex-only `ArithmeticExtractor`
**And** `robustness_invariant_holds` is `True` iff
  `adversarial_accuracy >= standard_accuracy - 0.05`

Spec: REQ-BENCH-006, REQ-BENCH-007, SCENARIO-BENCH-022

### REQ-BENCH-009: Live Precision Micro-Benchmark (50q × 3 variants × 2 models)

The repository SHALL provide a live GPU precision micro-benchmark (Exp 439) that:

- Evaluates 50 GSM8K questions per variant × 3 variants × 2 models = 300 LLM calls.
- Fits within a 45-minute ExperimentTimeoutWatchdog budget (the Exp 437 fix).
- Tests three ablation variants: BASELINE (no verify), CRANE_ONLY (CRANEExtractionGate
  + one-shot repair), and FULL_STACK (CRANE + JitRLConstraintMemory + energy gate).
- Produces a ``MicroPrecisionResult`` per (model, variant) pair with fields:
  ``model_id``, ``variant``, ``n_questions``, ``baseline_accuracy``,
  ``variant_accuracy``, ``signed_improvement``, ``crane_detection_rate``,
  ``inference_mode``.
- Builds a ``carnot.precision_micro.v1`` artifact via ``build_micro_precision_artifact``
  with ``honest_verdict`` in {'live_improvement', 'live_no_improvement', 'blocked'}.
- Requires ``inference_mode='live_gpu'`` for any non-blocked verdict — simulated
  results may never produce a live headline claim.
- Saves full CoT responses to ``results/experiment_439_live_cot.json`` for
  downstream FOVER annotation (Exp 442).

### SCENARIO-BENCH-025: Exp 439 Live Improvement Criterion

**Given** Exp 439 runs on live dual-RTX-3090 hardware with CARNOT_FORCE_LIVE=1
**And** all 6 (model, variant) batches complete within the 45-minute watchdog
**When** the artifact is assembled
**Then** ``honest_verdict`` is 'live_improvement' iff best non-baseline
  ``signed_improvement`` > 0
**And** ``headline_result`` identifies the best (model, variant) by signed_improvement
**And** ``inference_mode`` is 'live_gpu'
**And** ``per_model_results`` contains all 6 (model, variant) entries

### SCENARIO-BENCH-026: Exp 439 Blocked Artifact on Gate Failure

**Given** Exp 439 runs without CARNOT_FORCE_LIVE=1 (CI mode)
**Or** the LiveGPUGate check fails
**When** the artifact is assembled
**Then** ``honest_verdict`` is 'blocked'
**And** ``headline_result`` is None
**And** ``inference_mode`` is 'blocked'
**And** ``per_model_results`` is an empty list

Spec: REQ-BENCH-009, SCENARIO-BENCH-025, SCENARIO-BENCH-026

### REQ-BENCH-010: Live HumanEval Micro-Benchmark (50 problems × 2 models)

The repository SHALL provide a live GPU HumanEval micro-benchmark (Exp 440) that:

- Evaluates 50 HumanEval problems per model × 2 models = 100 LLM calls.
- Fits within a 45-minute ExperimentTimeoutWatchdog budget using LongRunBenchmarkExecutor
  with batch_size=25 (two 25-problem batches per model).
- For each model: generates code with the LLM, runs official test cases, then applies
  CodeExtractor + VerifyRepairPipeline on failing solutions, and re-evaluates.
- Runs PBT determinism/idempotency checks on solutions that pass official tests.
- Produces a ``MicroHumanEvalResult`` per model with fields: ``model_id``,
  ``n_problems``, ``pass_at_1_before``, ``pass_at_1_after``, ``signed_improvement``,
  ``pbt_bugs_found``, ``inference_mode``.
- Builds a ``carnot.humaneval_micro.v1`` artifact via ``build_micro_humaneval_artifact``
  with ``honest_verdict`` in {'code_verification_positive', 'code_no_improvement', 'blocked'}.
- Requires ``inference_mode='live_gpu'`` for any non-blocked verdict — simulated
  results may never produce a live headline claim.
- Imports ALL CodeExtractor and HumanEval helpers from Exp 369 and Exp 428 — no
  duplication of tested code.

### SCENARIO-BENCH-027: Exp 440 Live Improvement Criterion

**Given** Exp 440 runs on live dual-RTX-3090 hardware with CARNOT_FORCE_LIVE=1
**And** all 2 model × 2 batch runs complete within the 45-minute watchdog
**When** the artifact is assembled
**Then** ``honest_verdict`` is 'code_verification_positive' iff at least one model
  shows ``signed_improvement`` > 0
**And** ``headline_result`` identifies the best model by signed_improvement
**And** ``inference_mode`` is 'live_gpu'
**And** ``per_model_results`` contains one entry per model

### SCENARIO-BENCH-028: Exp 440 Blocked Artifact on Gate Failure

**Given** Exp 440 runs without CARNOT_FORCE_LIVE=1 (CI mode)
**Or** the LiveGPUGate check fails
**When** the artifact is assembled
**Then** ``honest_verdict`` is 'blocked'
**And** ``headline_result`` is None
**And** ``inference_mode`` is 'blocked'
**And** ``per_model_results`` is an empty list

Spec: REQ-BENCH-010, SCENARIO-BENCH-027, SCENARIO-BENCH-028

### REQ-BENCH-011: Live Adversarial GSM8K Micro-Benchmark (50q × 3 conditions × 2 models)

The repository SHALL provide a live GPU adversarial GSM8K micro-benchmark (Exp 441) that:

- Evaluates 50 GSM8K questions per model × 2 models across 3 conditions:
  1. **standard**    — original (clean) question, no verify-repair.
  2. **adversarial** — distractor-appended question (one irrelevant sentence), no repair.
  3. **repaired**    — distractor-appended question + VerifyRepairPipeline.
- Fits within a 45-minute ExperimentTimeoutWatchdog budget using LongRunBenchmarkExecutor
  with batch_size=50 (one 50-question batch per condition).
- Imports AdversarialGSMQuestion and build_adversarial_questions from adversarial_gsm8k.py
  (Exp 354 code) — no duplication.
- Imports all inference helpers (load_gsm8k_questions, _extract_answer, _is_correct,
  _call_model, _build_per_model_result, _compute_top_level_verdict) from Exp 355 — no
  duplication.
- Produces a ``MicroAdversarialResult`` per model with fields: ``model_id``, ``n_questions``,
  ``standard_accuracy``, ``adversarial_accuracy``, ``repaired_accuracy``,
  ``adversarial_drop_pct``, ``repair_improvement_pct``, ``inference_mode``.
- Builds a ``carnot.adversarial_micro.v1`` artifact via ``build_micro_adversarial_artifact``
  with ``honest_verdict`` in {'improvement_positive', 'degradation_positive', 'neutral',
  'blocked'}.
- Sets ``robustness_claim=True`` iff ``repair_improvement_pct > 0`` AND
  ``adversarial_drop_pct > 5`` (at least one model triggered Apple-paper-style degradation
  AND repair recovered it).
- Requires ``inference_mode='live_gpu'`` for any non-blocked verdict.

### SCENARIO-BENCH-029: Exp 441 Live Improvement Criterion

**Given** Exp 441 runs on live dual-RTX-3090 hardware with CARNOT_FORCE_LIVE=1
**And** all model × condition runs complete within the 45-minute watchdog
**When** the artifact is assembled
**Then** ``honest_verdict`` is 'improvement_positive' iff at least one model shows
  ``repair_improvement_pct > 0``
**And** ``robustness_claim`` is True iff ``repair_improvement_pct > 0`` AND
  ``adversarial_drop_pct > 5`` (for any model)
**And** ``inference_mode`` is 'live_gpu'
**And** ``per_model_results`` contains one entry per model

### SCENARIO-BENCH-030: Exp 441 Blocked Artifact on Gate Failure

**Given** Exp 441 runs without CARNOT_FORCE_LIVE=1 (CI mode)
**Or** the LiveGPUGate check fails
**When** the artifact is assembled
**Then** ``honest_verdict`` is 'blocked'
**And** ``robustness_claim`` is False
**And** ``inference_mode`` is 'blocked'
**And** ``per_model_results`` is an empty list

Spec: REQ-BENCH-011, SCENARIO-BENCH-029, SCENARIO-BENCH-030

### REQ-BENCH-012: Live Precision Post-Fix Benchmark — GemmaTransformersLoader Backend

The live precision benchmark (Exp 451, post-RETRO-028 fix) SHALL support
GemmaTransformersLoader as the Gemma4 inference backend, where:

- Gemma4-E4B-it is loaded exclusively via ``GemmaTransformersLoader`` (HuggingFace
  transformers), never via llama.cpp, to avoid the token-id=14 infinite ``<unused8>``
  bug confirmed in RETRO-028.
- The artifact records ``gemma4_loader='GemmaTransformersLoader'`` so auditors can
  confirm the llama.cpp path was not taken.
- If ``GemmaTransformersLoader.is_valid_output()`` returns False for any Gemma4
  response, the response is treated as a generation failure (not counted as correct).
- Qwen3.5-0.8B continues to use the standard HF pipeline loader (no change from
  Exp 439).

### REQ-BENCH-013: LivePrecisionResult — Signed Improvement Data Class

The benchmark SHALL provide a ``LivePrecisionResult`` data class in
``python/carnot/pipeline/live_precision_result.py``, where:

- ``model_id: str`` — the model name/ID being benchmarked.
- ``pre_accuracy: float`` — baseline accuracy (no verify-repair pipeline).
- ``post_accuracy: float`` — accuracy after verify-repair pipeline is applied.
- ``signed_improvement`` (computed property) — ``post_accuracy - pre_accuracy``.
  Negative values are honest regression signals — never clamp or abs() this field.
- ``is_positive`` (computed property) — ``True`` iff ``signed_improvement > 0``.
- Both properties are pure Python (no JAX dependency) so they work in CPU-only
  test environments without GPU hardware.

### SCENARIO-BENCH-031: Exp 451 First Positive Verify-Repair Number

**Given** Exp 451 runs on live dual-RTX-3090 hardware with CARNOT_FORCE_LIVE=1
**And** Gemma4-E4B-it is loaded via GemmaTransformersLoader (RETRO-028 fix)
**And** 50 GSM8K questions are evaluated for both Gemma4-E4B-it and Qwen3.5-0.8B
**When** baseline and pipeline variant runs complete
**Then** the artifact records ``LivePrecisionResult`` dicts for both models
**And** ``first_positive_number`` is True iff at least one model's
  ``LivePrecisionResult.is_positive`` is True
**And** ``honest_verdict`` is ``'first_positive'`` iff ``first_positive_number``
  is True, else ``'no_improvement_v2'``
**And** ``gemma4_loader`` is ``'GemmaTransformersLoader'`` in the artifact
**And** ``schema`` is ``'carnot.live_precision.v2'``

### SCENARIO-BENCH-032: Exp 451 Deferred Artifact When GPU Unavailable

**Given** Exp 451 runs without live GPU hardware (CARNOT_FORCE_LIVE not set)
**Or** setup_gpu() reports not all_healthy
**When** the artifact is assembled
**Then** ``status`` is ``'gpu_required'``
**And** ``honest_verdict`` is ``'deferred_to_gpu'``
**And** the artifact does NOT contain fabricated accuracy numbers

Spec: REQ-BENCH-012, REQ-BENCH-013, SCENARIO-BENCH-031, SCENARIO-BENCH-032

### REQ-BENCH-014: Live Precision Benchmark Runs 100 GSM8K Questions Per Model

The live precision benchmark (Exp 464) SHALL run 100 GSM8K questions per model
(stratified 50 easy + 50 hard) to provide better statistical confidence than
the 50q run in Exp 451.  Both models (Gemma4-E4B-it and Qwen3.5-0.8B) must each
complete 100 questions in both baseline and pipeline variants.

**Why 100q over 50q:** At 50 questions, a 5pp difference (e.g. 40% vs 45%) has
a 95% confidence interval of roughly ±14pp, overlapping zero.  At 100 questions
the same difference has a ±10pp interval, narrowing the uncertainty enough to make
directional claims credible.  Wilson score CI is reported in the artifact.

**Implementation Status:** Done (Exp 464)

### REQ-BENCH-015: Integrated Extraction Stack

The live precision pipeline SHALL use an IntegratedExtractor that applies:
1. VeriCoTStepValidator (FOL + Z3 satisfiability checking) — primary extractor.
2. VPRMArithmeticVerifier (deterministic rule-based) — secondary extractor.
3. ArithmeticExtractor (regex equation extractor) — optional fallback.

Violations from VeriCoT and VPRM are merged into a single list.
The extractor_used field in the artifact records which extractors produced
violations (comma-separated names).

**Implementation Status:** Done (Exp 464)

### REQ-BENCH-016: DualGPURunner Assigns Models to Dedicated GPUs

Exp 464 SHALL use DualGPUAssigner to pin:
- Gemma4-E4B-it to cuda:0
- Qwen3.5-0.8B to cuda:1

This prevents the GPU 1 idle waste documented in RETRO-034.

**Implementation Status:** Done (Exp 464)

### SCENARIO-BENCH-033: Exp 464 Deferred Artifact When GPU Unavailable

**Given** CARNOT_FORCE_LIVE is not set (CPU/CI environment)
**When** experiment_464 runs
**Then** the artifact is written with status='gpu_required' and honest_verdict='deferred_to_gpu'
**And** the artifact contains schema='carnot.live_precision.v3'
**And** DeliverableGuard.assert_written() passes

### SCENARIO-BENCH-034: Precision100qResult Wilson CI Is Credible

**Given** a Precision100qResult with n_questions=100 and post_accuracy=0.50
**When** confidence_interval_95 is computed
**Then** the interval is within (0.40, 0.60) — i.e. ±10pp at 50% hit rate

**Given** a Precision100qResult with n_questions=100 and post_accuracy=0.05
**When** confidence_interval_95 is computed
**Then** the interval spans less than 0.10 (Wilson score shrinks at extremes)

### SCENARIO-BENCH-035: IntegratedExtractor Calls VeriCoT Then VPRM

**Given** an IntegratedExtractor with mocked vericot and vprm
**When** extract(cot_text) is called
**Then** vericot.detect_violations is called first
**And** vprm.detect_violations is called second
**And** the returned violations are the union of both results

Spec: REQ-BENCH-014, REQ-BENCH-015, REQ-BENCH-016,
      SCENARIO-BENCH-033, SCENARIO-BENCH-034, SCENARIO-BENCH-035

### REQ-BENCH-017: VeriCoT + VPRM + CRANE Integrated as Primary Extraction Front-End

The VerifyRepairPipeline SHALL use IntegratedExtractor (VeriCoTStepValidator +
VPRMArithmeticVerifier) as the primary extraction front-end for all live GPU
benchmark runs.  The integrated stack replaces the legacy ArithmeticExtractor-only
path for IT model evaluation.

**Implementation Status:** Implemented (Exp 467)

### REQ-BENCH-018: 200q Benchmark Uses DualGPURunner with batch_size=8

The Exp 467 benchmark SHALL run 200 GSM8K questions per model using DualGPURunner
with batch_size=8.  Both models (Gemma4-E4B-it→cuda:0, Qwen3.5-0.8B→cuda:1) run
in parallel, each with their own inference loop.

**Implementation Status:** Implemented (Exp 467)

### REQ-BENCH-019: 200q Result Includes Wilson 95% CI

The Live200qResult data class SHALL compute and expose a Wilson score 95% confidence
interval on post_accuracy.  At n=200, the CI half-width SHALL be ≤ 0.035 (3.5pp),
giving statistically credible directional claims.

**Implementation Status:** Implemented (Exp 467)

### SCENARIO-BENCH-036: Live200qResult CI Half-Width ≤ 3.5pp at n=200

**Given** a Live200qResult with n=200 and post_accuracy=0.05 (near boundary)
**When** ci_95 is computed
**Then** (ci_95[1] - ci_95[0]) / 2 <= 0.035

### SCENARIO-BENCH-037: is_statistically_positive Requires Lower CI Bound > 0

**Given** a Live200qResult with signed_improvement > 0
**When** the lower bound of ci_95 > 0
**Then** is_statistically_positive == True
**And** when lower bound <= 0, is_statistically_positive == False

### SCENARIO-BENCH-038: Exp 467 Writes Deliverable JSON and CoT Pairs

**Given** Exp 467 completes (live or gpu_required)
**When** the experiment exits
**Then** results/experiment_467_live_200q_integrated.json exists on disk
**And** DeliverableGuard.assert_written() passes without raising

Spec: REQ-BENCH-017, REQ-BENCH-018, REQ-BENCH-019,
      SCENARIO-BENCH-036, SCENARIO-BENCH-037, SCENARIO-BENCH-038

### REQ-BENCH-020: Adversarial Benchmark Loads from apple/GSM-Symbolic

The adversarial benchmark shall attempt to load the `apple/GSM-Symbolic` HuggingFace
dataset (main split, first 50 questions) and fall back to a hardcoded set of 50
adversarial-style questions with irrelevant context sentences when the dataset is
unavailable.  The artifact shall record `data_source='huggingface'` or
`data_source='hardcoded_fallback'` so downstream analysis can trace provenance.

**Implementation Status:** Done (Exp 468)

### REQ-BENCH-021: Three-Condition Comparison: Standard Baseline, Adversarial Baseline, Adversarial with Carnot

The adversarial benchmark shall run three conditions for each model:
- Condition A: standard GSM8K baseline (same 50q subset as Exp 464)
- Condition B: adversarial variant baseline (no verify-repair pipeline)
- Condition C: adversarial variant with Carnot verify-repair (IntegratedExtractor)

The artifact shall record per-condition accuracy and the signed improvement for each
transition (A→B measuring adversarial drop, B→C measuring Carnot recovery).

**Implementation Status:** Done (Exp 468)

### REQ-BENCH-022: thesis_confirmed When Adversarial Improvement > Standard Improvement

`AdversarialBenchmarkResult.thesis_confirmed` shall be `True` when
`carnot_adversarial_improvement > carnot_standard_improvement` (Carnot's benefit is
larger on adversarial variants than on standard GSM8K).  The artifact field
`honest_verdict` shall reflect: `'thesis_confirmed'` / `'thesis_partial'` (any
improvement > 0) / `'thesis_not_confirmed'`.

**Implementation Status:** Done (Exp 468)

### SCENARIO-BENCH-039: adversarial_drop Is Positive When Adversarial Accuracy Is Lower

**Given** an `AdversarialBenchmarkResult` with standard_acc=0.70 and adversarial_baseline_acc=0.55
**When** `adversarial_drop` is computed
**Then** `adversarial_drop == 0.15` (positive = accuracy degradation on adversarial)

### SCENARIO-BENCH-040: thesis_confirmed Requires Adversarial Improvement Exceeds Standard Improvement

**Given** an `AdversarialBenchmarkResult` with:
  - standard_acc=0.70, adversarial_baseline_acc=0.55, adversarial_carnot_acc=0.65
  - carnot_standard_improvement=0.05 (from condition A)
**When** `thesis_confirmed` is evaluated
**Then** `thesis_confirmed == True` (adversarial improvement 0.10 > standard improvement 0.05)

**Given** carnot_standard_improvement=0.15 (larger than adversarial improvement 0.10)
**Then** `thesis_confirmed == False`

### SCENARIO-BENCH-041: Exp 468 Writes Deliverable with schema='carnot.adversarial_benchmark.v2'

**Given** Exp 468 completes (live or gpu_required)
**When** the experiment exits
**Then** `results/experiment_468_gsm_symbolic_adversarial.json` exists on disk
**And** the artifact contains `schema='carnot.adversarial_benchmark.v2'`
**And** `data_source` is either `'huggingface'` or `'hardcoded_fallback'`
**And** `DeliverableGuard.assert_written()` passes without raising

Spec: REQ-BENCH-020, REQ-BENCH-021, REQ-BENCH-022,
      SCENARIO-BENCH-039, SCENARIO-BENCH-040, SCENARIO-BENCH-041

### REQ-BENCH-023: HumanEval Live Uses CodeExtractor + VeriCoT for Code Verification

The live HumanEval benchmark (Exp 469) SHALL use CodeExtractor for structural analysis
and VeriCoTStepValidator for logical consistency checking of implementation reasoning
steps, rather than ArithmeticExtractor (which does not apply to code verification tasks).

The pipeline SHALL:
- Extract structural violations from generated code using CodeExtractor
- Validate reasoning-step logical consistency using VeriCoTStepValidator
- Obtain repair direction from BoltzmannRepairBridge when violations are detected
- Re-generate or re-execute code after repair to produce a final verdict

Rationale: Exp 440 showed 0% improvement because ArithmeticExtractor found no violations
in code — it is designed for arithmetic text, not Python function bodies. CodeExtractor
operates on structural properties (syntax, type, execution output) and is the correct
extractor for HumanEval-style code verification.

### REQ-BENCH-024: Code Verification Pipeline: Generate → Extract → Verify → Repair → Re-Execute

The code verification pipeline SHALL implement the following stages per problem:

1. **Generate**: produce a Python function using the live LLM (Gemma4-E4B-it)
2. **Extract**: run CodeExtractor on the generated code to find structural violations
3. **Verify**: run VeriCoTStepValidator on the docstring reasoning steps
4. **Repair**: if violations detected, call BoltzmannRepairBridge.get_repair_direction()
   to obtain a repair hint, then re-generate or patch the code
5. **Re-Execute**: run the official HumanEval test cases on the repaired code

Results SHALL be aggregated into HumanEvalLiveResult with:
- baseline_pass_at_1: fraction passing before the pipeline
- pipeline_pass_at_1: fraction passing after the pipeline
- signed_improvement: pipeline_pass_at_1 − baseline_pass_at_1
- honest_verdict: 'code_verification_positive' only when live GPU AND improvement > 0

### SCENARIO-BENCH-042: CodeVerificationResult Improvement and Regression Flags

**Given** a CodeVerificationResult for a single HumanEval problem
**When** `pipeline_passed=True` and `baseline_passed=False`
**Then** `improvement=True` and `regression=False`

**When** `baseline_passed=True` and `pipeline_passed=False`
**Then** `regression=True` and `improvement=False`

**When** both are True or both are False
**Then** both `improvement` and `regression` are False

### SCENARIO-BENCH-043: Exp 469 Writes Deliverable with schema='carnot.humaneval.live.v2'

**Given** Exp 469 completes (live or gpu_required)
**When** the experiment exits
**Then** `results/experiment_469_humaneval_live_vericot.json` exists on disk
**And** the artifact contains `schema` list including `'carnot.humaneval.live.v2'` key or
  the artifact `honest_verdict` is one of: `'code_verification_positive'`,
  `'code_no_improvement'`, `'gpu_required'`
**And** `DeliverableGuard.assert_written()` passes without raising

Spec: REQ-BENCH-023, REQ-BENCH-024,
      SCENARIO-BENCH-042, SCENARIO-BENCH-043

### REQ-INFRA-001: Conductor Timeout Wrapper

The research conductor SHALL be invokable via a wrapper shell script
`scripts/run_experiment_with_timeout.sh` that enforces a configurable hard
timeout on every `claude -p` invocation, where:

- **Hard cap**: Timeout defaults to 45 minutes (RETRO-001 action item carried
  forward from milestones 2026.04.22 and 2026.04.23).
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
- **Rationale**: 23.5% post-test failure rate observed in 2026.04.23 milestone
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
2026.04.23 retrospective (PIDs 2592400/2595103, ~1050 MB VRAM each, 0% utilisation) and
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
files were identified in the 2026.04.23 retrospective (NEW-002) as a root cause of ~5%
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

### REQ-INFRA-006: Host Prerequisites Registry

**Status:** Implemented (Exp 338)

The repository must maintain a machine-readable registry of host-level prerequisites
(`ops/host-prereqs.md`) so that experiments blocked by missing system packages can
be diagnosed immediately without repeating independent discovery.

Root cause (RETRO-006): AMD XDNA NPU experiments (Exps 292, 303, 314, 335) each
independently discovered the same two missing packages (`ninja`, `openblas`), wasting
approximately 4 experiment slots.

**Acceptance criteria:**
- `ops/host-prereqs.md` exists and contains a markdown table with columns:
  `Package`, `Check Command`, `Install (Arch)`, `Install (Debian)`, `Required For`.
- `HostPrereqRegistry` loads the table from `ops/host-prereqs.md` at construction time.
- `HostPrereqRegistry.check_prereqs(experiment_class)` returns a `list[str]` of package
  names whose check commands fail (non-zero exit or command not found) and that are
  required for the given experiment class.
- `HostPrereqRegistry.check_prereqs()` (no class filter) checks all registered packages.
- Check commands are run via `subprocess.run` with `timeout=5`; `FileNotFoundError` and
  non-zero returncode both count as "missing".
- The registry never raises when a check command is absent — it logs a warning and marks
  the package as missing gracefully.
- At minimum, the registry includes: `ninja`, `openblas`, `nvidia-smi`, `yosys`,
  `nextpnr-xilinx`, and the `CARNOT_FORCE_LIVE` environment variable check.

### REQ-INFRA-007: DualGPU Auto-Assignment as Default

**Status:** Implemented (Exp 338)

`ExperimentTemplate.setup_gpu()` must automatically assign models to separate GPU
indices when two or more model specs are provided and `CARNOT_FORCE_LIVE=1`.
This closes RETRO-004: `DualGPURunner` was implemented in Exp 326 but two-model
experiments continued to run sequentially on GPU 0.

**Acceptance criteria:**
- When `len(model_specs) >= 2` and `CARNOT_FORCE_LIVE=1`:
  - `model_specs[i]['gpu']` is set to `i` (zero-based, up to n_gpus_detected).
  - If only 1 GPU is detected, all models are assigned to GPU 0 and a warning is logged
    with the text "RETRO-004 warning".
- `setup_gpu()` returns `dual_gpu_auto_assigned: bool` as an additional key.
  Existing keys (`all_healthy`, `models`, `prewarm_time_s`, `gpu_monitor_results`) are
  preserved without modification.
- When `CARNOT_FORCE_LIVE=0` (CI mode), auto-assignment is skipped and
  `dual_gpu_auto_assigned` is `False`.
- When `len(model_specs) < 2`, auto-assignment is skipped and
  `dual_gpu_auto_assigned` is `False`.

### SCENARIO-INFRA-009: Registry Loads ops/host-prereqs.md

**Given** `ops/host-prereqs.md` exists with a valid markdown table
**When** `HostPrereqRegistry()` is constructed
**Then** at least 4 package entries are loaded
**And** each entry has `package`, `check_command`, `required_for` fields
**And** construction does not raise

### SCENARIO-INFRA-010: Missing Package Detected by Check Command

**Given** a package whose check command exits non-zero (or is not found on the system)
**When** `HostPrereqRegistry().check_prereqs(experiment_class="npu")` is called
**Then** the package name appears in the returned list
**And** no exception is raised

### SCENARIO-INFRA-011: DualGPU Auto-Assignment with Two Model Specs

**Given** `CARNOT_FORCE_LIVE=1`
**And** `model_specs` has exactly two entries, both with `gpu=0`
**And** `DualGPUMonitor._get_gpu_count()` returns 2
**When** `ExperimentTemplate.setup_gpu(model_specs)` is called (with mock prewarm_fn)
**Then** `model_specs[0]['gpu']` is 0
**And** `model_specs[1]['gpu']` is 1
**And** the returned dict has `dual_gpu_auto_assigned=True`

### REQ-INFRA-008: Pre-Session Startup Health Check

**Status:** Implemented (Exp 339)

Before launching a research session or experiment run, a startup health check must:
- Detect how many NVIDIA GPUs are visible via `nvidia-smi`.
- Identify zombie GPU processes (0% utilisation, >100 MiB VRAM) via `DualGPUMonitor`.
- Optionally kill zombie processes (only when `--kill-zombies` flag is passed).
- Print a single-line summary: `SESSION STARTUP: n_gpus=X zombies=Y killed=Z all_healthy=T/F`.
- Degrade gracefully when `nvidia-smi` is absent (CI environments): report `n_gpus=0`.
- Exit 0 always (health check, not a blocking requirement).

**Acceptance criteria:**
- `scripts/session_startup.sh --dry-run` prints the summary line without killing any process.
- `parse_session_startup_output(output)` returns a dict with keys:
  `n_gpus_detected`, `n_zombies_found`, `n_zombies_killed`, `all_healthy`.
- `all_healthy` is `True` iff `n_gpus_detected >= 2` AND `n_zombies_found == 0`.
- When `nvidia-smi` is absent, `n_gpus_detected=0`, `all_healthy=False`, no exception raised.
- `run_session_startup(dry_run=True)` never kills any process.

### SCENARIO-INFRA-012: Dry-Run Reports GPUs and Zombies Without Killing

**Given** `scripts/session_startup.sh` exists and is executable
**And** the system has 0, 1, or 2 GPUs (including CI with no GPUs)
**When** `run_session_startup(dry_run=True)` is called
**Then** the returned dict contains keys `n_gpus_detected`, `n_zombies_found`, `n_zombies_killed`, `all_healthy`
**And** `n_zombies_killed` is 0 (dry-run never kills)
**And** no `kill -9` command is issued

### SCENARIO-INFRA-013: CI-Safe Degradation When nvidia-smi Absent

**Given** `nvidia-smi` is not installed on the system
**When** `run_session_startup(dry_run=True)` is called
**Then** `n_gpus_detected` is 0
**And** `all_healthy` is `False`
**And** no exception is raised

---

### REQ-INFRA-014: Live GPU Failure Must Be Explicit, Not Silent Fallback

When `CARNOT_FORCE_LIVE=1` is set in the environment, the inference pipeline MUST NOT
silently fall back to simulated mode if GPU inference cannot be established.  A silent
fallback produces artifacts that appear to contain live GPU measurements but are actually
synthetic, invalidating every benchmark result without any signal to the researcher.

Instead:

- `diagnose_live_gpu(model_ids)` must inspect each failure layer in sequence
  (CUDA visibility, PyTorch CUDA availability, `CARNOT_FORCE_LIVE` env var, model loadability)
  and return a `LiveGPUDiagnostic` dataclass with `is_live_capable: bool` and
  `failure_reason: str` identifying the first failed layer.
- `ExperimentTemplate.setup_gpu()` must call `diagnose_live_gpu()` when
  `CARNOT_FORCE_LIVE=1` and any model pre-warm fails.  If `not is_live_capable`, it
  must raise `RuntimeError("Live GPU required but unavailable: <failure_reason>")`.
- `diagnose_live_gpu()` is CI-safe: it never raises; it returns a diagnostic even when
  no GPU is present.

Rationale: Exps 340, 341, 346, 347 all ran in simulated mode despite `CARNOT_FORCE_LIVE=1`,
producing four consecutive milestones of meaningless benchmark results.  Both RTX 3090s
were idle throughout.  Silent fallback is a correctness bug, not a usability feature.

### SCENARIO-INFRA-014: Live GPU Diagnostic Reports First Failure Layer

**Given** `CARNOT_FORCE_LIVE=1` is set
**And** `nvidia-smi` is absent (or reports 0 GPUs)
**When** `diagnose_live_gpu(["Qwen/Qwen3.5-0.8B"])` is called
**Then** `result.is_live_capable` is `False`
**And** `result.cuda_visible` is `False`
**And** `result.failure_reason` contains `"cuda_visible"`
**And** no exception is raised

**Given** `CARNOT_FORCE_LIVE=1` is set
**And** CUDA is visible but `torch.cuda.is_available()` returns `False`
**When** `diagnose_live_gpu(["Qwen/Qwen3.5-0.8B"])` is called
**Then** `result.is_live_capable` is `False`
**And** `result.torch_available` is `False`
**And** `result.failure_reason` contains `"torch_cuda"`

**Given** `CARNOT_FORCE_LIVE=1` is set, CUDA visible, torch CUDA available,
**And** model `"nonexistent/model-xyz"` is not cached and cannot be downloaded
**When** `diagnose_live_gpu(["nonexistent/model-xyz"])` is called
**Then** `result.is_live_capable` is `False`
**And** `result.model_loadable` is `False`
**And** `result.failure_reason` contains `"model_loadable"`

### SCENARIO-INFRA-015: setup_gpu Raises When CARNOT_FORCE_LIVE=1 and GPU Unavailable

**Given** `CARNOT_FORCE_LIVE=1` is set
**And** all model pre-warms fail (`health_ok=False` for every model)
**When** `ExperimentTemplate.setup_gpu(model_specs)` is called
**Then** a `RuntimeError` is raised
**And** the error message starts with `"Live GPU required but unavailable:"`
**And** the message contains the `failure_reason` from `diagnose_live_gpu()`

**Given** `CARNOT_FORCE_LIVE=0` (CI mode)
**And** all model pre-warms fail
**When** `ExperimentTemplate.setup_gpu(model_specs)` is called
**Then** no exception is raised
**And** the returned dict has `all_healthy=False`

### REQ-INFRA-015: Conductor GPU Environment Propagation

The project shall provide a shell environment script at `scripts/conductor_gpu_env.sh`
that propagates `CARNOT_FORCE_LIVE=1` into conductor subprocess environments for all
GPU-tagged experiments, where:
- The script is sourced (not executed) before launching any experiment tagged `requires_gpu=True`
- The script contains `export CARNOT_FORCE_LIVE=1` as an unconditional top-level export
- The script contains a comment identifying it as the RETRO-012 fix
- A `ConductorEnvFix` dataclass records `env_script_path`, `exports` (dict of variable names
  to values), `apply_cmd` (the shell source command), and `is_documented` (bool)
- `build_conductor_env_fix(project_root)` creates the script and returns the dataclass
- `verify_env_script_exports(path)` returns `True` iff the script contains
  `export CARNOT_FORCE_LIVE=1`

### SCENARIO-INFRA-016: Env Script Created and Contains Correct Export

**Given** `build_conductor_env_fix(project_root)` is called with a valid project root
**When** the function completes
**Then** `scripts/conductor_gpu_env.sh` exists under `project_root`
**And** the file contains `export CARNOT_FORCE_LIVE=1`
**And** the returned `ConductorEnvFix.exports` dict maps `"CARNOT_FORCE_LIVE"` to `"1"`
**And** `ConductorEnvFix.apply_cmd` is `"source scripts/conductor_gpu_env.sh"`
**And** `verify_env_script_exports(path)` returns `True` for the created script

### SCENARIO-INFRA-017: verify_env_script_exports Detects Missing Export

**Given** a file that does NOT contain `export CARNOT_FORCE_LIVE=1`
**When** `verify_env_script_exports(path)` is called
**Then** the function returns `False`

**Given** a path that does not exist
**When** `verify_env_script_exports(path)` is called
**Then** the function returns `False`

### REQ-INFRA-016: Mandatory Result JSON Per Experiment

Every experiment script shall produce a result JSON at
`results/experiment_NNN_<slug>.json` before exiting, where:
- `RetroJSONEnforcer.check_result_json_exists(exp_id, results_dir)` returns `True` iff
  at least one file matching `experiment_NNN_*.json` exists in `results_dir`
- `RetroJSONEnforcer.audit_missing_jsons(exp_ids, results_dir)` returns the list of
  experiment IDs (as ints) for which no matching result JSON exists
- This enforcer is used in the RETRO-014 close to document the gap for experiments
  357, 358, and 362 and to prevent recurrence in future experiments

### SCENARIO-INFRA-018: audit_missing_jsons Reports Missing and Present Correctly

**Given** a results directory containing `experiment_357_llm_z3_formalizer.json`
**And** no file for experiment 358 or 362
**When** `RetroJSONEnforcer().audit_missing_jsons([357, 358, 362], results_dir)` is called
**Then** the returned list is `[358, 362]`
**And** `check_result_json_exists(357, results_dir)` returns `True`
**And** `check_result_json_exists(358, results_dir)` returns `False`

### REQ-INFRA-017: Session Startup Must Export CARNOT_FORCE_LIVE=1

`scripts/session_startup.sh` must exist at repo root and, when sourced, must:
- Source `scripts/conductor_gpu_env.sh` if the file exists
- Export `CARNOT_FORCE_LIVE=1` directly as a belt-and-suspenders measure
- Print a human-readable confirmation: `[session_startup] CARNOT_FORCE_LIVE=1 exported at <ISO-timestamp>`

`build_session_startup_script(project_root)` returns the shell script content as a string.
`check_session_startup_exists(project_root)` returns `True` iff the file exists.

### REQ-INFRA-018: LiveGPUGate Must Be Called at Start of Any GPU Experiment

A `LiveGPUGate` class in `python/carnot/pipeline/live_gpu_gate.py` provides a hard gate:
- `LiveGPUGate.check_env_var()` returns `True` iff `CARNOT_FORCE_LIVE == "1"` in environment
- `LiveGPUGate.check_gpu_live()` returns `True` iff `diagnose_live_gpu().is_live_capable`
- `LiveGPUGate.require_live(model_ids=[])` raises `RuntimeError` if env var missing or GPU not ready
- `LiveGPUGate.require_live_or_blocked(tmpl, model_ids)` returns blocked artifact dict on failure,
  `None` on success — for experiments that prefer a blocked result to an exception
- `LiveGPUGate.verify_subprocess_env_propagation(env_var)` spawns a subprocess and verifies
  the env var is inherited; returns `True` iff the var is present in subprocess env

### SCENARIO-INFRA-019: LiveGPUGate Raises When Env Var Missing

**Given** `CARNOT_FORCE_LIVE` is not set in the environment
**When** `LiveGPUGate.require_live()` is called
**Then** a `RuntimeError` is raised containing "CARNOT_FORCE_LIVE not set"

### SCENARIO-INFRA-020: LiveGPUGate Raises When GPU Not Live

**Given** `CARNOT_FORCE_LIVE=1` is set
**And** `diagnose_live_gpu()` returns `is_live_capable=False`
**When** `LiveGPUGate.require_live()` is called
**Then** a `RuntimeError` is raised containing "is_live_capable=False"

### SCENARIO-INFRA-021: Subprocess Inherits CARNOT_FORCE_LIVE from Parent

**Given** `CARNOT_FORCE_LIVE=1` is set in the current process environment
**When** `LiveGPUGate.verify_subprocess_env_propagation("CARNOT_FORCE_LIVE")` is called
**Then** the function returns `True`
**And** the spawned subprocess's environment contains `CARNOT_FORCE_LIVE=1`

### REQ-INFRA-019: DeliverableContentValidator Must Guard Conductor Fast-Path

The conductor fast-path must not accept a deliverable file unless it contains valid Python
source code.  A file that exists but contains JSON or other non-Python content must be
detected, logged as corrupt, and deleted so that the producing experiment re-runs.

**Acceptance criteria:**
- `DeliverableContentValidator.is_valid_python(path)` returns `True` only when `ast.parse()`
  succeeds on the file contents.
- `DeliverableContentValidator.validate_and_clear(path)` deletes the file and returns `False`
  when `is_valid_python()` returns `False`.
- `DeliverableContentValidator.audit_known_corrupt_files(project_root)` returns a
  `{path: status}` dict with `status` in `{'valid_python', 'corrupt_json', 'missing'}` for
  each of the five known corrupt modules (RETRO-023 affected files).

Spec: SCENARIO-INFRA-022, SCENARIO-INFRA-023 (Exp 404)

### REQ-INFRA-020: Cloud GPU Setup Script Generated When Local GPU Unavailable

When the local GPU is not live-capable, the preflight experiment must generate a shell
script at `scripts/setup_cloud_gpu.sh` containing provisioning commands for Lambda Labs,
vast.ai, and RunPod so the operator has a one-step path to cloud GPU.

**Acceptance criteria:**
- `build_cloud_gpu_instructions()` returns a `CloudGPUInstructions` dataclass with
  `lambda_command`, `vastai_command`, `runpod_command`, and `estimated_cost_per_hour_usd`.
- `generate_cloud_gpu_script(instructions, output_path)` writes a shell script containing
  all three provider commands.
- The generated script is referenced in the experiment artifact and the ops status document.

Spec: SCENARIO-INFRA-024 (Exp 404)

### SCENARIO-INFRA-022: DeliverableContentValidator Rejects JSON File as Invalid Python

**Given** a file exists on disk containing only JSON (e.g. `{"experiment": 375}`)
**When** `DeliverableContentValidator.is_valid_python(path)` is called
**Then** the function returns `False`

**Given** the same JSON file
**When** `DeliverableContentValidator.validate_and_clear(path)` is called
**Then** the function logs a warning, deletes the file, and returns `False`

### SCENARIO-INFRA-023: DeliverableContentValidator Accepts Valid Python Module

**Given** a file exists on disk containing syntactically correct Python source code
**When** `DeliverableContentValidator.is_valid_python(path)` is called
**Then** the function returns `True`

**When** `DeliverableContentValidator.validate_and_clear(path)` is called
**Then** the function returns `True` and does NOT delete the file

### SCENARIO-INFRA-024: Cloud GPU Script Generated on Preflight Failure

**Given** `run_gpu_preflight()` returns `is_live_capable=False`
**When** Exp 404 preflight v2 runs
**Then** `scripts/setup_cloud_gpu.sh` is written containing Lambda Labs, vast.ai, and RunPod
  provisioning commands
**And** `cloud_gpu_script_generated=True` appears in the result artifact

### REQ-INFRA-021: EnvironmentAutoFix Self-Injects CARNOT_FORCE_LIVE=1 When GPU Hardware is Detected

When GPU hardware is present (as detected by `torch.cuda.is_available()`) and
`CARNOT_FORCE_LIVE` is absent from `os.environ`, the `apply_env_autofix()` function
must set `os.environ['CARNOT_FORCE_LIVE'] = '1'` before returning.  This allows each
experiment script to self-configure without requiring the human shell to propagate the
var through the conductor's `subprocess.run()` call — the root cause of RETRO-022 (seven
consecutive milestones with blocked live GPU experiments).

**Acceptance criteria:**
- `apply_env_autofix()` returns an `EnvironmentAutoFix` dataclass with fields:
  `gpu_detected`, `carnot_force_live_was_set`, `auto_fix_applied`, `final_env_value`.
- When GPU is detected and var was absent: `auto_fix_applied=True`,
  `final_env_value='1'`, `os.environ['CARNOT_FORCE_LIVE'] == '1'` after the call.
- When GPU is not detected: `auto_fix_applied=False`, no env mutation.
- When var was already set: `auto_fix_applied=False`, `carnot_force_live_was_set=True`.

Spec: SCENARIO-INFRA-025, SCENARIO-INFRA-026, SCENARIO-INFRA-027 (Exp 413)

### REQ-INFRA-022: EnvironmentAutoFix Logs a Warning When Applied

When `apply_env_autofix()` applies the fix (sets `CARNOT_FORCE_LIVE=1`), it must emit a
`WARNING`-level log message via the standard Python `logging` module so the human
operator can observe that env propagation is still broken and the workaround was
activated.  Silent auto-fix would mask the underlying infrastructure problem.

**Acceptance criteria:**
- Log message contains the string `"EnvironmentAutoFix applied CARNOT_FORCE_LIVE=1"`.
- Log message is emitted at `logging.WARNING` level.
- No warning is emitted when `auto_fix_applied=False`.

Spec: SCENARIO-INFRA-026 (Exp 413)

### SCENARIO-INFRA-025: EnvironmentAutoFix No-Op When No GPU

**Given** `torch.cuda.is_available()` returns `False` (or torch is not importable)
**When** `apply_env_autofix()` is called
**Then** `auto_fix_applied=False`
**And** `os.environ` is not mutated
**And** `honest_verdict` in the artifact is `'gpu_not_detected'`
**And** `retro_022_resolved=False` in the artifact

### SCENARIO-INFRA-026: EnvironmentAutoFix Applies Fix and Logs Warning

**Given** `torch.cuda.is_available()` returns `True`
**And** `CARNOT_FORCE_LIVE` is absent from `os.environ`
**When** `apply_env_autofix()` is called
**Then** `auto_fix_applied=True`
**And** `os.environ['CARNOT_FORCE_LIVE'] == '1'`
**And** a WARNING log containing `"EnvironmentAutoFix applied CARNOT_FORCE_LIVE=1"` is emitted
**And** `honest_verdict` in the artifact is `'auto_fix_applied'`
**And** `retro_022_resolved=True` in the artifact

### SCENARIO-INFRA-027: EnvironmentAutoFix No-Op When Var Already Set

**Given** `torch.cuda.is_available()` returns `True`
**And** `CARNOT_FORCE_LIVE=1` is already in `os.environ`
**When** `apply_env_autofix()` is called
**Then** `auto_fix_applied=False`
**And** `carnot_force_live_was_set=True`
**And** no warning is emitted
**And** `honest_verdict` in the artifact is `'gpu_detected_env_was_correct'`
**And** `retro_022_resolved=True` in the artifact

### REQ-INFRA-023: ExperimentTimeoutWatchdog Kills Experiment After Configurable Timeout

The `ExperimentTimeoutWatchdog` class must monitor an experiment's wall-clock runtime
and, when the elapsed time exceeds the configured timeout, call `sys.exit(1)` after
writing a partial result JSON to disk.  This closes RETRO-003: PID 3509070 ran 144+
minutes with GPU0 at 82C when a 45-minute cap would have freed the GPU 99 minutes early.

**Acceptance criteria:**
- `ExperimentTimeoutWatchdog(experiment_id, timeout_minutes, result_path)` starts a
  background thread timer that calls `_on_timeout()` after `timeout_minutes` wall-clock
  minutes.
- `start()` activates the timer and records `_start_time`.
- `stop()` cancels the timer before it fires (normal completion path).
- `is_active()` returns `True` while running and not yet timed out.
- `elapsed_minutes()` returns float wall-clock minutes since `start()`.
- `_on_timeout()` writes a partial result JSON (if `result_path` is set) then calls
  `sys.exit(1)`.
- Context manager: `__enter__` calls `start()`, `__exit__` calls `stop()`.
- `ExperimentTimeoutResult` dataclass has fields: `experiment_id`, `timeout_minutes`,
  `elapsed_minutes`, `timed_out`, `partial_result_path`.

Spec: SCENARIO-INFRA-028, SCENARIO-INFRA-029, SCENARIO-INFRA-030 (Exp 425)

### REQ-INFRA-024: ExperimentTimeoutWatchdog Default Timeout Is 45 Minutes

The default timeout for `ExperimentTimeoutWatchdog` must be 45 minutes.  This value
was derived from the PID 3509070 case: the experiment ran 144 minutes (3.2x over budget)
and GPU0 reached 82C.  A 45-minute cap would have freed the GPU 99 minutes early.

`get_timeout_minutes()` reads `CARNOT_CONDUCTOR_TIMEOUT_MINUTES` from the environment.
When the env var is absent or empty, it returns 45.  When present, it returns the integer
value of the env var.

**Acceptance criteria:**
- `get_timeout_minutes()` returns 45 when `CARNOT_CONDUCTOR_TIMEOUT_MINUTES` is unset.
- `get_timeout_minutes()` returns the integer value of `CARNOT_CONDUCTOR_TIMEOUT_MINUTES`
  when set.
- `ExperimentTimeoutWatchdog` default `timeout_minutes` parameter is 45.

Spec: SCENARIO-INFRA-030 (Exp 425)

### SCENARIO-INFRA-028: Watchdog Fires After Timeout

**Given** an `ExperimentTimeoutWatchdog` is created with `timeout_minutes=0.01` (0.6 s)
**And** `start()` is called
**When** 0.6 s elapses without `stop()` being called
**Then** `_on_timeout()` fires
**And** if `result_path` is set, a JSON file is written at that path with `timed_out=True`
**And** `sys.exit(1)` is called

### SCENARIO-INFRA-029: Stop Before Timeout Prevents Firing

**Given** an `ExperimentTimeoutWatchdog` is created with `timeout_minutes=0.1`
**And** `start()` is called
**When** `stop()` is called before the timeout elapses
**Then** `_on_timeout()` is never called
**And** no `sys.exit` is invoked

### SCENARIO-INFRA-030: get_timeout_minutes Reads Env Var

**Given** `CARNOT_CONDUCTOR_TIMEOUT_MINUTES` is not set
**When** `get_timeout_minutes()` is called
**Then** the return value is `45`

**Given** `CARNOT_CONDUCTOR_TIMEOUT_MINUTES=30` is in the environment
**When** `get_timeout_minutes()` is called
**Then** the return value is `30`

### REQ-INFRA-025: DualGPUHealthCheck Asserts GPU1 Utilization Within 60 Seconds of Model Load

After models are loaded in a dual-GPU experiment, `check_dual_gpu_health()` must detect
whether GPU1 is a "zombie" — a GPU that has allocated VRAM (>500 MB) but is doing zero
compute (util < 1%).  This pattern was observed in RETRO-025: PID 3509070 held 1786 MB on
GPU1 at 0% utilization for 144+ minutes while GPU0 ran at 88%.

`DualGPUHealthResult` carries structured fields:
- `gpu0_util_pct`, `gpu1_util_pct`: utilization percentages from nvidia-smi or pynvml
- `gpu0_temp_c`, `gpu1_temp_c`: temperature in Celsius
- `gpu1_is_zombie`: `True` when `gpu1_vram_mb > 500` AND `gpu1_util_pct < 1`
- `temperature_warning`: `True` when any GPU temperature exceeds 80C
- `recommended_batch_size_factor`: `0.75` when `temperature_warning=True`, else `1.0`

`check_dual_gpu_health(timeout_seconds=60)` uses pynvml (preferred) or falls back to
`nvidia-smi` subprocess.  When neither is available (CI), it returns safe mock defaults
(all zeros, no zombie, no temperature warning) and never raises.

`build_gpu_fix_artifact(health, prior_retro_path)` builds a JSON-serializable dict with:
- `schema='carnot.dual_gpu_fix.v1'`
- `honest_verdict`: `'zombie_detected'` if `gpu1_is_zombie` else `'gpu1_healthy'`
- `retro_025_status`: `'zombie_confirmed'` if `gpu1_is_zombie` else `'zombie_cleared'`
- All fields from `DualGPUHealthResult`

**Acceptance criteria:**
- When `gpu1_util_pct=0` and `gpu1_vram_mb>500`: `gpu1_is_zombie=True`
- When pynvml and nvidia-smi are both unavailable: returns safe mock defaults, no exception
- `build_gpu_fix_artifact` sets `honest_verdict='zombie_detected'` when `gpu1_is_zombie=True`

Spec: SCENARIO-INFRA-031, SCENARIO-INFRA-032 (Exp 426)

### REQ-INFRA-026: ExperimentTemplate.setup_gpu() Reduces Batch Size When GPU Temperature Exceeds 80C

After pre-warm completes, `ExperimentTemplate.setup_gpu()` must call
`check_dual_gpu_health()` and include the result under the `'dual_gpu_health'` key in
its return dict.

When `temperature_warning=True` (any GPU > 80C at pre-warm time):
- Log a WARNING: `'GPU temp > 80C — reducing batch_size by 25%'`
- The `dual_gpu_health` result's `recommended_batch_size_factor` will be `0.75`

When `gpu1_is_zombie=True`:
- Log a WARNING: `'GPU1 allocated but idle — DualGPURunner may not be scheduling GPU1.'`

This requirement was motivated by GPU0 reaching 82C (within 1-3C of RTX 3090 throttle
threshold 83-85C) during the RETRO-025 incident.  The 25% batch reduction gives the GPU
thermal headroom to avoid the throttle boundary.

**Acceptance criteria:**
- `setup_gpu()` return dict always contains `'dual_gpu_health'` key after call
- When `temperature_warning=True`: WARNING logged, `recommended_batch_size_factor=0.75`
- When `gpu1_is_zombie=True`: WARNING logged identifying zombie
- CI fallback (no nvidia-smi): `dual_gpu_health` still present with safe defaults

Spec: SCENARIO-INFRA-033 (Exp 426)

### SCENARIO-INFRA-031: GPU1 Zombie Detected When VRAM High and Utilization Zero

**Given** a `DualGPUHealthResult` with `gpu1_vram_mb=1786` and `gpu1_util_pct=0`
**When** `gpu1_is_zombie` is evaluated
**Then** `gpu1_is_zombie=True`
**And** `build_gpu_fix_artifact` produces `honest_verdict='zombie_detected'`
**And** `retro_025_status='zombie_confirmed'`

### SCENARIO-INFRA-032: CI Mode Returns Safe Defaults When nvidia-smi Unavailable

**Given** pynvml is not installed and nvidia-smi is not found on PATH
**When** `check_dual_gpu_health()` is called
**Then** it returns a `DualGPUHealthResult` with all utilization/temperature at 0
**And** `gpu1_is_zombie=False`
**And** `temperature_warning=False`
**And** no exception is raised

### SCENARIO-INFRA-033: Temperature Warning Triggers Batch Size Reduction

**Given** any GPU temperature exceeds 80C at pre-warm time
**When** `check_dual_gpu_health()` is evaluated
**Then** `temperature_warning=True`
**And** `recommended_batch_size_factor=0.75`
**And** `setup_gpu()` logs a WARNING about temperature and batch reduction

### REQ-INFRA-027: LongRunBenchmarkExecutor Splits Benchmarks Into Checkpointed Batches

**Description:** Any benchmark with more questions than the batch size must be automatically
partitioned into fixed-size batches. Each batch is executed independently and checkpointed to
disk on completion (or on timeout). The final result is assembled from all batch checkpoint
files. This closes RETRO-026: 200-question benchmarks previously exceeded the 45-minute
ExperimentTimeoutWatchdog budget, producing scaffolding_only artifacts.

**Acceptance Criteria:**
- `LongRunBenchmarkExecutor.partition(questions)` splits a list into batches of at most `batch_size`
- `run_batch(batch, inference_fn, watchdog_timeout_minutes)` uses ExperimentTimeoutWatchdog internally
- Each batch is checkpointed via `save_batch()` on completion or partial timeout
- `assemble(batches)` builds a `LongRunBenchmarkResult` from all batch checkpoint files
- `honest_verdict='complete'` iff all batches fully completed; `'partial_N_of_M'` otherwise

Spec: SCENARIO-INFRA-034, SCENARIO-INFRA-035, SCENARIO-INFRA-036 (Exp 437)

### REQ-INFRA-028: LongRunBenchmarkExecutor Default Batch Size Is 50, Configurable

**Description:** The default batch size of 50 is derived from the budget arithmetic:
50 questions × ~10s/question × 5 variants × 2 models = ~8,333s ≈ 139 min per batch,
but with parallelism a single batch of 50 (1 model, 1 variant) takes ≤10 min, comfortably
under the 40-minute per-batch watchdog limit. The batch size is overridable via
`CARNOT_BENCH_BATCH_SIZE` to support both large and small benchmark runs.

**Acceptance Criteria:**
- `get_batch_size()` returns 50 when `CARNOT_BENCH_BATCH_SIZE` is absent
- `get_batch_size()` returns the integer value of `CARNOT_BENCH_BATCH_SIZE` when set
- `LongRunBenchmarkExecutor(batch_size=N)` overrides the default for that instance

Spec: SCENARIO-INFRA-036 (Exp 437)

### SCENARIO-INFRA-034: 120 Questions Split Into Three Batches Correctly

**Given** 120 synthetic questions and a `LongRunBenchmarkExecutor(batch_size=50)`
**When** `executor.partition(questions)` is called
**Then** 3 `BenchmarkBatch` objects are returned
**And** batches have sizes [50, 50, 20] in order
**And** `batch_id` is 0, 1, 2 respectively
**And** `start_idx` / `end_idx` correctly index into the original list

### SCENARIO-INFRA-035: Partial Batch Assembly Produces Honest Partial Verdict

**Given** 3 batches where only batch 0 and batch 1 completed (batch 2 status='pending')
**When** `executor.assemble(batches)` is called
**Then** `result.honest_verdict == 'partial_2_of_3'`
**And** `result.completed_batches == 2`
**And** `result.n_batches == 3`

### SCENARIO-INFRA-036: Full Batch Assembly Produces Complete Verdict

**Given** 3 batches all with status='complete'
**When** `executor.assemble(batches)` is called
**Then** `result.honest_verdict == 'complete'`
**And** `result.completed_batches == 3`

### REQ-INFRA-029: DualGPURunner Uses Explicit device_map When Assigning Models to GPUs

When `ExperimentTemplate.setup_gpu()` is called with `len(model_specs) >= 2` AND
`CARNOT_FORCE_LIVE=1` AND `n_gpus >= 2`, the framework shall use
`device_map={'': 'cuda:N'}` (explicit per-GPU assignment) instead of
`device_map='auto'` for each model spec.

Rationale (RETRO-025): `device_map='auto'` allows CUDA to allocate VRAM on GPU1 for
layer offloading while the actual forward pass stays on GPU0.  This produces the zombie
pattern observed in RETRO-025 — GPU1 holds 1786 MB at 0% utilization for 144+ minutes.
Explicit assignment pins each model to a single GPU, preventing cross-device VRAM spill.

When running in CI mode (no CUDA) or with a single GPU, `device_map='auto'` is
preserved as the fallback — it is only replaced in the explicit dual-GPU live path.

Each model spec entry gains a `device_map` key populated by `build_zombie_fix_strategy()`,
and `ExperimentTemplate.setup_gpu()` logs:
`'Using explicit device assignment to prevent GPU1 zombie allocation'`

Spec: SCENARIO-INFRA-037, SCENARIO-INFRA-038 (Exp 438)

### REQ-INFRA-030: ZombieFixResult Captures Pre/Post State of GPU1 Zombie Fix

The framework shall provide a `ZombieFixResult` dataclass capturing:
- `gpu0_model_id` / `gpu1_model_id` — model IDs loaded on each GPU
- `gpu0_device_map` / `gpu1_device_map` — the actual device_map strings used
- `fix_applied` — True when explicit device_map was used (not 'auto')
- `post_fix_gpu1_util_pct` — GPU1 utilization after model load (None in CI mode)
- `honest_verdict` — one of: 'fix_applied_and_verified' / 'fix_applied_unverified' / 'ci_mode'

And a `build_zombie_fix_artifact(result)` that emits `schema='carnot.gpu1_zombie_fix.v1'`
with `retro_025_status` in ('fix_applied_and_verified', 'fix_applied_unverified', 'ci_mode').

Spec: SCENARIO-INFRA-037, SCENARIO-INFRA-038 (Exp 438)

### SCENARIO-INFRA-037: Dual-GPU Live Mode Uses Explicit Device Maps

**Given** `CARNOT_FORCE_LIVE=1` and 2 GPUs detected
**When** `build_zombie_fix_strategy(n_gpus=2, model_ids=['ModelA', 'ModelB'])` is called
**Then** the returned dict maps 'ModelA' to `{'': 'cuda:0'}` (NOT 'auto')
**And** maps 'ModelB' to `{'': 'cuda:1'}` (NOT 'auto')
**And** no model spec uses `device_map='auto'`

### SCENARIO-INFRA-038: Single-GPU / CI Mode Falls Back to device_map='auto'

**Given** CI mode (CUDA unavailable) or `n_gpus=1`
**When** `build_zombie_fix_strategy(n_gpus=1, model_ids=['ModelA'])` is called
**Then** the returned dict maps 'ModelA' to `'auto'`
**And** no explicit `cuda:N` assignment is made

### REQ-INFRA-031: Experiment Scripts Must Use Atomic Write for Result Files

All experiment scripts that write a JSON result file shall use the atomic write pattern:
write to a `.tmp` file first, then `os.rename()` to the final path. This prevents
partial writes from producing a zero-byte or truncated result file when an exception
occurs mid-write (RETRO-030 root cause).

**Rationale (RETRO-030):** Exp 446 exited with status 0 but produced no result file.
The root cause was that an exception occurred after the `open()` call but before
`json.dump()` completed, leaving no file on disk.  The watchdog could not detect this
because the process exited cleanly.  Atomic write (write-to-tmp + rename) ensures that
the final path either does not exist or contains a complete, valid JSON document.

**Why os.rename() is atomic on POSIX:** The kernel guarantees that `rename(2)` is
atomic — the destination path either points to the old file or the new file; there is
no window where it is missing or partially written.  This is not guaranteed on all
filesystems (e.g. NFS mounts without lock support), but is reliable on ext4/xfs.

**Implementation Status:** Done (Exp 452)

Spec: SCENARIO-INFRA-039, SCENARIO-INFRA-040 (Exp 452)

### REQ-INFRA-032: Conductor Verifies Deliverable File Exists After Run

After each experiment process completes, the conductor (or the experiment script
itself) shall check that the deliverable result file exists on disk.  If the file is
absent, the run is treated as a silent failure and the experiment is re-queued or
flagged as BLOCKED.

**Rationale (RETRO-030):** Exp 446 exited with status 0 but the watchdog reported
success because it only checked the process exit code, not file existence.  A file-
existence check would have surfaced the missing artifact immediately.

**Implementation Status:** Done (Exp 452) — experiment script self-checks via
`AtomicResultWriter.verify_exists()` and raises `RuntimeError` if the file is absent.

Spec: SCENARIO-INFRA-039, SCENARIO-INFRA-040 (Exp 452)

### SCENARIO-INFRA-039: AtomicResultWriter Writes and Verifies Result File

**Given** an `AtomicResultWriter` with a valid output path
**When** `writer.write({"key": "value"})` is called
**Then** the file exists at the output path with valid JSON content
**And** `writer.verify_exists()` returns `True`

### SCENARIO-INFRA-040: AtomicResultWriter Does Not Corrupt Existing File on Partial Write

**Given** an existing result file at the output path
**And** an `AtomicResultWriter` targeting the same path
**When** an exception is raised during `write()` before the rename completes
**Then** the original file at the output path is unchanged
**And** the `.tmp` file (if any) is the only partial artifact

---

## Operational Retrospective Requirements (REQ-RETRO-*)

These requirements govern the per-milestone operational retrospectives produced by
the research conductor.  Each retro summarises wall-time data, identifies bottlenecks,
audits whether prior action items were resolved, and proposes new improvements.

### REQ-RETRO-001: Milestone 2026.04.23 Retrospective

The conductor must produce `results/operational_retro_2026_04_23.json` with schema
`operational_retro_v1`.  The artifact must record `n_experiments`, `total_wall_time_minutes`,
`avg_minutes_per_experiment`, `bottlenecks_identified` (≥ 1 entry), `action_items`
(≥ 1 entry), `carry_over_from_previous_retro`, and `estimated_next_milestone_speedup_pct`.

### SCENARIO-RETRO-001: Retro Artifact Schema Conforms to Contract

**Given** a completed milestone with ≥ 10 experiments
**When** the retro script runs
**Then** the artifact is valid JSON with all required top-level keys present and correctly typed
**And** `schema` is a non-empty string
**And** `milestone` identifies the correct date label

### SCENARIO-RETRO-002: Action Items Carry Forward Unresolved Items

**Given** action items RETRO-001 and RETRO-002 were not implemented in the prior milestone
**When** the 2026.04.23 retro runs
**Then** both items appear in `action_items` with `status == "carried_forward"`
**And** at least one NEW-* item appears with `status == "new"`

---

### REQ-RETRO-002: Milestone 2026.04.23 (v2) Full-History Retrospective

An extended version of the 2026.04.23 retro that covers the full 359-experiment history.
The artifact uses schema `operational_retro_v2` and adds `gpu_utilization_analysis`,
`meta_reflection`, and `post_test_failure_rate_assessment` fields.

### SCENARIO-RETRO-003: GPU Utilisation Analysis Included

**Given** nvidia-smi data captured at retro time
**When** the v2 retro artifact is loaded
**Then** `gpu_utilization_analysis.current_state_at_retro` is present
**And** `zombie_process_assessment` is a non-empty string

### SCENARIO-RETRO-004: Speedup Estimate Reflects Action Item Impact

**Given** action items with `estimated_impact_pct` fields
**When** the retro computes `estimated_time_savings_pct`
**Then** the value equals the sum of individual item impacts (capped at 100)
**And** a `savings_breakdown` dict maps each item ID to its individual estimate

---

### REQ-RETRO-003: Milestone 2026.04.24 Retrospective

The conductor must produce `results/operational_retro_2026_04_24.json` with schema
`carnot.operational_retro.v1`.  The artifact must record:

- `n_experiments`: int in [10, 20]
- `total_wall_time_min`: positive float
- `mean_time_per_exp_min`: total / n (consistent)
- `slowest_experiment`: string identifying the highest-duration experiment
- `retro_001_resolved`: bool — was the 45-min timeout wrapper (RETRO-001) actually
  shipped and active this milestone?
- `retro_002_resolved`: bool — was DualGPUMonitor (RETRO-002) shipped and active?
- `actual_speedup_pct`: float — measured wall-time improvement vs prior milestone
  (40.6 min/exp baseline); negative values are valid and mean regression
- `estimated_next_milestone_speedup_pct`: float ≥ 0 — forward-looking estimate; can be 0
- `carry_over`: list of dicts `{id, description, resolved: bool}` from the 2026.04.23 retro
- `action_items`: list of dicts with `{id, description, estimated_impact_pct}`

### SCENARIO-RETRO-005: All Prior Action Items Resolved Flag Correctly

**Given** RETRO-001 was implemented in Exp 325 and RETRO-002 in Exp 326
**When** the 2026.04.24 retro artifact is loaded
**Then** `retro_001_resolved` is `True`
**And** `retro_002_resolved` is `True`
**And** every entry in `carry_over` has `resolved: True`

### SCENARIO-RETRO-006: Actual Speedup Reflects Measured Data

**Given** prior milestone mean was 40.6 min/exp and this milestone mean is derived from
conductor log timestamps for Exps 325–336
**When** `actual_speedup_pct` is computed
**Then** the value equals `(40.6 − mean_time_per_exp_min) / 40.6 × 100` rounded to 1 dp
**And** `actual_speedup_pct` is positive (this milestone ran faster than the prior one)

### REQ-AGENT-001: Multi-Turn Constraint State Propagation

The system shall propagate constraint state across multiple agent reasoning steps, where:
- Each step produces an `AgentStep` record capturing: `step_id`, `question`,
  `proposed_action`, `action_cot`, `constraint_violations`, `repaired_action`,
  `committed`, and `repair_attempts`
- A `ConstraintState` dataclass carries `step_id`, `active_constraints` (list of
  constraint descriptions from prior committed steps), `accumulated_facts` (list of
  action strings committed so far), `facts_established` (count of committed facts),
  and `model_id` (chain identifier)
- When a step commits, its final action is appended to `accumulated_facts` and
  `facts_established` is incremented
- When a step is blocked (committed=False), `accumulated_facts` is NOT updated

### REQ-AGENT-002: Commit-Gate Via SAVeR Auditor Loop

The system shall gate each agent action behind a SAVeR auditor loop, where:
- `SAVeRVerifier(pipeline, max_repair_attempts=3)` wraps a `VerifyRepairPipeline`
- `propose_step(question, action_cot, constraint_state)` runs
  `VerifyRepairPipeline.verify_and_repair()` on the action's chain-of-thought
- If verification succeeds immediately: `committed=True`, `repair_attempts=0`
- If violations exist but repair succeeds within `max_repair_attempts`:
  `committed=True`, `repaired_action` set, `repair_attempts >= 1`
- If violations persist after all repair attempts: `committed=False`,
  `repaired_action=None`, the step is blocked
- When `pipeline` is `None` (CI-safe mode): all steps are approved immediately
  (`committed=True`, `repair_attempts=0`) without any verification calls
- `run_chain(steps, initial_state)` runs a full multi-step reasoning chain,
  propagating `ConstraintState` across steps
- `compute_faithfulness(steps)` returns the fraction of steps where `committed=True`
- `build_saver_artifact(steps, faithfulness)` serializes results with
  `schema="carnot.saver_verifier.v1"`

---

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
| REQ-PRED-005 | Not Started | Implemented | NPU unblock status audit — IRON toolchain viability (SCENARIO-EXP303-G, Exp 435) |
| REQ-VERIFY-081 | Not Started | Implemented | Confidence-weighted violation scoring + ConfidenceVerifier tests (SCENARIO-105/106/107/108) |
| REQ-VERIFY-082 | Not Started | Implemented | Repair gate with confidence threshold + pipeline integration tests |
| REQ-VERIFY-083 | Not Started | Implemented | Expression specificity confidence signal (SCENARIO-109/110, Exp 332) |
| REQ-VERIFY-084 | Not Started | Implemented | Partition function variance confidence signal (SCENARIO-111/112, Exp 332) |
| REQ-VERIFY-085 | Not Started | Implemented | Dual-signal confidence-weighted repair + ConfidenceRepairResult (SCENARIO-109–112, Exp 332) |
| REQ-LEARN-012 | Not Started | Implemented | ThresholdAdapter online adaptation + Tier 3 benchmark (SCENARIO-LEARN-019/020, Exp 309) |
| REQ-LEARN-013 | Not Started | In Progress | Four-tier relay benchmark — RelayBatchResult + RelayArtifact (SCENARIO-LEARN-021/022, Exp 318) |
| REQ-LEARN-014 | Not Started | In Progress | Four-tier relay live GPU validation — wrapper artifact with simulation_comparison + jepa_skip_rate_live (SCENARIO-LEARN-023/024, Exp 329) |
| REQ-LEARN-015 | Not Started | Implemented | Model-adaptive constraint thresholds — PerModelFPTracker + ModelAdaptiveThresholds + 43 tests (SCENARIO-LEARN-025/026, Exp 333) |
| REQ-LEARN-016 | Not Started | Implemented | Selective CaseMemory consolidation — SelectiveConsolidation + add_trace_selective + 43 tests (SCENARIO-LEARN-027/028, Exp 333) |
| REQ-LEARN-017 | Not Started | Implemented | Constraint template addition from memory patterns — ConstraintTemplateLibrary + 4 builtin templates (SCENARIO-LEARN-029/030/031/032, Exp 343) |
| REQ-LEARN-018 | Not Started | Implemented | Constraint template persistence + builtin registry — to_dict/from_dict + register_builtin_templates (SCENARIO-LEARN-029/030/031/032, Exp 343) |
| REQ-LEARN-019 | Not Started | Implemented | CaseMemory to ConstraintTemplateLibrary wiring — CaseMemoryTemplateWiring [violation_type_to_pattern_key + on_violation_recorded]; constraint addition benchmark shows improvement_delta>0 (SCENARIO-LEARN-033/034, Exp 344) |
| REQ-LEARN-020 | Not Started | Implemented | Session state persistence — SessionMemory save/load/exists/clear/list_sessions + JSON schema v1 (SCENARIO-LEARN-035/036/037, Exp 345) |
| REQ-LEARN-021 | Not Started | Implemented | Per-model session isolation — scoped subdirs + VerifyRepairPipeline session_memory param + close() (SCENARIO-LEARN-035/036/037, Exp 345) |
| REQ-LEARN-022 | Not Started | Implemented | EORM CoT energy reward model — EORMModel + CoTEnergyInput + energy/rank/save/load/n_params (SCENARIO-LEARN-038/039, Exp 346) |
| REQ-LEARN-023 | Not Started | Implemented | EORM contrastive ranking loss — EORMTrainer + contrastive_loss + train_step + train_epoch (SCENARIO-LEARN-040, Exp 346) |
| REQ-LEARN-024 | Not Started | Implemented | JEPA real-data retrain — ViolationPair + extract_violation_pairs + JEPARetrainer + build_retrain_artifact (SCENARIO-LEARN-041/042, Exp 347) |
| REQ-LEARN-025 | Implemented | Verified | EORM real-data retrain — load_real_cot_pairs + merge_cot_corpora + EORMRetrainResult + build_retrain_artifact (SCENARIO-LEARN-043/044, Exp 359). Run 20260415: retrain_mode=synthetic_only (5 real pairs from Exp 341 HumanEval; Exps 340/355 still simulated), before_auc=0.500, after_auc=0.500, honest_verdict=synthetic_only. Live GPU required for real_data_improvement verdict. |
| REQ-LEARN-026 | Not Started | Implemented | Tier 1+2+3 self-learning relay — SelfLearningBatchResult + SelfLearningRelay + _compute_auc_roc + run_batch + learning_trajectory (SCENARIO-LEARN-045/046, Exp 361). Tier 1: PerModelFPTracker.update() per question. Tier 2: CaseMemoryTemplateWiring.on_violation_recorded() per incorrect response, all 4 templates activated (carry/sign/unit/comparison). Tier 3: EORM gate AUC-ROC per batch. 54 tests, 100% module coverage. |
| REQ-LEARN-027 | Not Started | Implemented | Cross-batch learning improvement — compute_learning_improvement + build_relay_artifact with honest_verdict gate (SCENARIO-LEARN-047, Exp 361). Run 20260415: batch1=0.600, batch4=0.720, improved=True, honest_verdict=synthetic_only. "learning_confirmed" requires live_gpu + improved=True. |
| REQ-EXTRACT-010 | Not Started | Implemented | NL2Z3Extractor + Z3Result + pipeline integration tests (SCENARIO-EXTRACT-020/021, Exp 310) |
| REQ-EXTRACT-011 | Not Started | Implemented | Z3Result dataclass + is_violation + subprocess timeout tests (SCENARIO-EXTRACT-022/023/024, Exp 310) |
| REQ-EXTRACT-012 | Not Started | Implemented | Extractor benchmark corpus + FP/TP metrics + winner selection (SCENARIO-EXTRACT-025/026, Exp 311) |
| REQ-EXTRACT-013 | Not Started | Implemented | FPCategory enum + AutopsyCase + categorize_fp + load_broken_cases + compute_category_distribution (SCENARIO-EXTRACT-027/028, Exp 331) |
| REQ-EXTRACT-014 | Not Started | Implemented | FP autopsy experiment + inconclusive artifact + recommended_fix mapping (SCENARIO-EXTRACT-029/030, Exp 331) |
| REQ-EXTRACT-015 | Not Started | Implemented | CoTCircuitVerifier + CoTStep + extract_cot_steps (SCENARIO-EXTRACT-031/032, Exp 336) |
| REQ-EXTRACT-016 | Not Started | Implemented | CoTCircuit + build_circuit + find_broken_links + broken link detection (SCENARIO-EXTRACT-033/034/035, Exp 336) |
| REQ-EXTRACT-017 | Not Started | Implemented | ExtractorResult + compare_extractors + build_comparison_artifact; live 4-way extractor comparison on IT model responses (SCENARIO-EXTRACT-036/037, Exp 342) |
| REQ-EXTRACT-019 | Not Started | Implemented | LLMz3Formalizer + Z3FormalizationResult + exec sandbox + CI stub mode (SCENARIO-EXTRACT-039/040/041, Exp 357) |
| REQ-EXTRACT-020 | Not Started | Implemented | Zero-FP Z3 path: only unsat raises violation; fp_rate in experiment artifact (SCENARIO-EXTRACT-039/041, Exp 357) |
| REQ-EXTRACT-021 | Not Started | Implemented | ExtractionBenchmarkResult + run_extraction_benchmark + build_extraction_comparison_artifact; comparative benchmark with honest_verdict gate (SCENARIO-EXTRACT-042/043, Exp 358) |
| REQ-EXTRACT-023 | Not Started | Implemented | ExtractorComparisonResult + run_extractor_comparison + build_extractor_comparison_artifact (schema=carnot.extraction_comparison.v1); honest_verdict=live_gpu_winner gated on ALL results live_gpu; 42 tests 100% coverage (SCENARIO-EXTRACT-047/048, Exp 367) |
| REQ-EXTRACT-024 | Not Started | Not Started | VeriCoTStepValidator + FOLPremise + FOLPremise.to_z3_assertion() — FOL premise extraction from natural-language CoT steps; use_mock=True for test isolation (SCENARIO-EXTRACT-049/050, Exp 453) |
| REQ-EXTRACT-025 | Not Started | Not Started | StepVerdict + verify_step() + detect_violations() — Z3 UNSAT as violation signal for IT model outputs (SCENARIO-EXTRACT-051, Exp 453) |
| REQ-EXTRACT-026 | Not Started | Not Started | Exp 453 head-to-head: ArithmeticExtractor=0 violations vs VeriCoT>=1 on IT natural-language arithmetic errors (SCENARIO-EXTRACT-049/050/051, Exp 453) |
| REQ-REPAIR-010 | Not Started | Implemented | Z3GatedRepair + Z3GatedRepairResult + pipeline integration tests (SCENARIO-REPAIR-020/021, Exp 312) |
| REQ-REPAIR-011 | Not Started | Implemented | Z3 SAT fast-exit path tests (SCENARIO-REPAIR-022, Exp 312) |
| REQ-BENCH-001 | Not Started | Script written (Exp 315) | Full-scale benchmark script with 95% Wilson CI; execution in Exp 316 |
| REQ-BENCH-002 | Not Started | Implemented (Exp 328) | Live GPU benchmark result with inference_mode="live_gpu"; wrapper artifact with simulation_divergence + baseline_deviation; honest blocked artifact when GPU unavailable |
| REQ-BENCH-003 | Not Started | Implemented (Exp 340) | PrecisionStackResult + PipelineVariant + compute_signed_improvement + build_precision_benchmark_artifact; live full precision stack benchmark (SCENARIO-BENCH-007/008/009) |
| REQ-BENCH-004 | Not Started | Implemented (Exp 341) | HumanEvalResult + compute_pass_at_1 + compute_pass_at_1_after_repair + build_humaneval_artifact; live HumanEval code verification benchmark (SCENARIO-BENCH-010/011) |
| REQ-BENCH-005 | Not Started | Implemented (Exp 353) | SmokeTestResult + run_smoke_test + build_smoke_test_artifact; live GPU smoke test gate with CI-skip path and RuntimeError on simulated fallback (SCENARIO-BENCH-012/013) |
| REQ-BENCH-006 | Not Started | Implemented (Exp 354) | AdversarialGSMQuestion + build_adversarial_questions + AdversarialBenchmarkResult + compute_adversarial_results + build_adversarial_artifact; adversarial GSM8K harness with 20-distractor pool and CI-safe synthetic mode (SCENARIO-BENCH-014/015/016) |
| REQ-BENCH-007 | Not Started | Implemented (Exp 354) | robustness_invariant_holds field in build_adversarial_artifact; invariant is True when adversarial_accuracy >= standard_accuracy - 0.05 (SCENARIO-BENCH-014/016) |
| REQ-BENCH-009 | Not Started | Implemented (Exp 439) | MicroPrecisionResult + build_micro_precision_artifact (schema carnot.precision_micro.v1); 50q × 3 variants × 2 models micro-benchmark with LongRunBenchmarkExecutor; CoT log for FOVER (SCENARIO-BENCH-025/026) |
| REQ-BENCH-010 | Not Started | Scaffolding (Exp 440) | MicroHumanEvalResult + build_micro_humaneval_artifact (schema carnot.humaneval_micro.v1); 50 problems × 2 models; blocked at gate (SCENARIO-BENCH-027/028) |
| REQ-BENCH-011 | Not Started | Implemented (Exp 441) | MicroAdversarialResult + build_micro_adversarial_artifact (schema carnot.adversarial_micro.v1); 50q × 3 conditions × 2 models micro-benchmark; robustness_claim field; LongRunBenchmarkExecutor (SCENARIO-BENCH-029/030) |
| REQ-BENCH-012 | Not Started | Implemented (Exp 451) | GemmaTransformersLoader backend for Gemma4 in live precision post-fix benchmark; gemma4_loader artifact field; is_valid_output gating (SCENARIO-BENCH-031/032) |
| REQ-BENCH-013 | Not Started | Implemented (Exp 451) | LivePrecisionResult(model_id, pre_accuracy, post_accuracy) + signed_improvement + is_positive; schema carnot.live_precision.v2; honest_verdict first_positive/no_improvement_v2 (SCENARIO-BENCH-031/032) |
| REQ-BENCH-014 | Not Started | Implemented (Exp 464) | 100q stratified GSM8K benchmark per model; Wilson 95% CI; Precision100qResult (SCENARIO-BENCH-033/034) |
| REQ-BENCH-015 | Not Started | Implemented (Exp 464) | IntegratedExtractor(VeriCoTStepValidator + VPRMArithmeticVerifier + fallback); extractor_used field (SCENARIO-BENCH-035) |
| REQ-BENCH-016 | Not Started | Implemented (Exp 464) | DualGPUAssigner pins Gemma4-E4B-it→cuda:0, Qwen3.5-0.8B→cuda:1 (REQ-INFRA-034) |
| REQ-INFRA-001 | N/A | Implemented | run_experiment_with_timeout.sh + timeout_wrapper_exists() tests (SCENARIO-INFRA-001, Exp 325) |
| REQ-INFRA-002 | N/A | Implemented | generate_test_stub() idempotency + ast.parse tests (SCENARIO-INFRA-002/003, Exp 325) |
| REQ-INFRA-003 | N/A | Implemented | DualGPUMonitor zombie detection + CI-safe fallback tests (SCENARIO-INFRA-004/006, Exp 326) |
| REQ-INFRA-004 | N/A | Implemented | check_dual_gpu_health() + setup_gpu() integration tests (SCENARIO-INFRA-005/006, Exp 326) |
| REQ-INFRA-005 | N/A | Implemented | DependencyAudit + extract_required_files + check_dependencies + CLI tests (SCENARIO-INFRA-007/008, Exp 327) |
| REQ-RETRO-001 | N/A | Implemented | operational_retro_2026_04_23.json schema + action items + carry_over tests (SCENARIO-RETRO-001/002, Exp 319) |
| REQ-RETRO-002 | N/A | Implemented | operational_retro_2026_04_23.json v2 + gpu_utilization_analysis tests (SCENARIO-RETRO-003/004, Exp 319) |
| REQ-RETRO-003 | N/A | Implemented | operational_retro_2026_04_24.json + retro_001_resolved + actual_speedup_pct tests (SCENARIO-RETRO-005/006, Exp 337) |
| REQ-INFRA-006 | N/A | Implemented | HostPrereqRegistry + ops/host-prereqs.md + check_prereqs() tests (SCENARIO-INFRA-009/010, Exp 338) |
| REQ-INFRA-007 | N/A | Implemented | DualGPU auto-assignment in setup_gpu() + dual_gpu_auto_assigned key tests (SCENARIO-INFRA-011, Exp 338) |
| REQ-INFRA-008 | N/A | Implemented | session_startup.sh + session_startup.py + parse_session_startup_output + run_session_startup tests (SCENARIO-INFRA-012/013, Exp 339) |
| REQ-INFRA-014 | N/A | Implemented | LiveGPUDiagnostic + diagnose_live_gpu() + setup_gpu() RuntimeError on CARNOT_FORCE_LIVE=1 failure (SCENARIO-INFRA-014/015, Exp 352) |
| REQ-INFRA-015 | N/A | Implemented | ConductorEnvFix + build_conductor_env_fix + verify_env_script_exports + scripts/conductor_gpu_env.sh (SCENARIO-INFRA-016/017, Exp 365). Closes RETRO-012. |
| REQ-INFRA-016 | N/A | Implemented | RetroJSONEnforcer + check_result_json_exists + audit_missing_jsons (SCENARIO-INFRA-018, Exp 365). Closes RETRO-014. |
| REQ-INFRA-017 | N/A | Implemented | session_startup.sh + build_session_startup_script + check_session_startup_exists (SCENARIO-INFRA-019/020/021, Exp 377). Part of RETRO-015 fix. |
| REQ-INFRA-018 | N/A | Implemented | LiveGPUGate + check_env_var + check_gpu_live + require_live + require_live_or_blocked + verify_subprocess_env_propagation (SCENARIO-INFRA-019/020/021, Exp 377). Closes RETRO-015. |
| REQ-INFRA-019 | N/A | Implemented | DeliverableContentValidator + is_valid_python + validate_and_clear + audit_known_corrupt_files (SCENARIO-INFRA-022/023, Exp 404). Closes RETRO-023. |
| REQ-INFRA-020 | N/A | Implemented | CloudGPUInstructions + build_cloud_gpu_instructions + generate_cloud_gpu_script + scripts/setup_cloud_gpu.sh (SCENARIO-INFRA-024, Exp 404). |
| REQ-INFRA-021 | N/A | Implemented | EnvironmentAutoFix + apply_env_autofix + build_env_autofix_artifact (SCENARIO-INFRA-025/026/027, Exp 413). Closes RETRO-022 (workaround). |
| REQ-INFRA-022 | N/A | Implemented | apply_env_autofix() logs WARNING when fix applied (SCENARIO-INFRA-026, Exp 413). |
| REQ-INFRA-023 | N/A | Implemented | ExperimentTimeoutWatchdog + ExperimentTimeoutResult + build_timeout_artifact (SCENARIO-INFRA-028/029, Exp 425). Closes RETRO-003 (17+ milestones). |
| REQ-INFRA-024 | N/A | Implemented | get_timeout_minutes() — default 45 min, configurable via CARNOT_CONDUCTOR_TIMEOUT_MINUTES (SCENARIO-INFRA-030, Exp 425). |
| REQ-INFRA-025 | N/A | Implemented | DualGPUHealthResult + check_dual_gpu_health + build_gpu_fix_artifact (SCENARIO-INFRA-031/032, Exp 426). Closes RETRO-025 (GPU1 zombie detection). |
| REQ-INFRA-026 | N/A | Implemented | setup_gpu() calls check_dual_gpu_health() + dual_gpu_health key + temp-guard WARNING + batch_size_factor=0.75 (SCENARIO-INFRA-033, Exp 426). |
| REQ-INFRA-027 | N/A | Implemented | LongRunBenchmarkExecutor + BenchmarkBatch + LongRunBenchmarkResult + save/load/assemble (SCENARIO-INFRA-034/035/036, Exp 437). Closes RETRO-026. |
| REQ-INFRA-028 | N/A | Implemented | get_batch_size() reads CARNOT_BENCH_BATCH_SIZE (default 50) (SCENARIO-INFRA-036, Exp 437). |
| REQ-INFRA-029 | N/A | Implemented | build_zombie_fix_strategy() + explicit device_map in setup_gpu() for dual-GPU live path (SCENARIO-INFRA-037/038, Exp 438). Closes RETRO-025 (GPU1 zombie scheduling). |
| REQ-INFRA-030 | N/A | Implemented | ZombieFixResult dataclass + build_zombie_fix_artifact() (schema carnot.gpu1_zombie_fix.v1) (Exp 438). |
| REQ-INFRA-031 | N/A | Implemented | AtomicResultWriter.write() — json.dumps + .tmp write + os.rename (POSIX-atomic) in python/carnot/pipeline/atomic_writer.py (SCENARIO-INFRA-039/040, Exp 452). Closes RETRO-030. |
| REQ-INFRA-032 | N/A | Implemented | AtomicResultWriter.verify_exists() — post-write file existence check; Exp 452 raises RuntimeError if absent (SCENARIO-INFRA-039, Exp 452). Closes RETRO-030. |
| REQ-VERIFY-086 | Not Started | Implemented | SinkProbe attention-sink pre-filter + SinkConcentration + SinkProbeResult + compute_sink_concentration (SCENARIO-VERIFY-113/114/115, Exp 348) |
| REQ-VERIFY-087 | Not Started | Implemented | SinkProbe threshold configuration + benchmark() skip/FNR/TNR reporting (SCENARIO-VERIFY-113/114/115, Exp 348) |
| REQ-VERIFY-088 | Not Started | Implemented | Three-tier pipeline benchmark — ThreeTierPipeline + ThreeTierPipelineResult + verify/benchmark + build_three_tier_artifact (SCENARIO-VERIFY-116/117, Exp 360); live GPU benchmark on real attention matrices (SCENARIO-VERIFY-118/119, Exp 373) |
| REQ-VERIFY-092 | Not Started | Implemented | SpilledEnergyDetector per-token logit-discrepancy hallucination signal — SpilledEnergyDetector + SpilledEnergyDetectorResult + SpilledEnergyToken + compute_detector_spilled_energy (SCENARIO-VERIFY-123/124, Exp 433) |
| REQ-VERIFY-093 | Not Started | Implemented | SpilledEnergyDetector CI-safe text mode — score_from_text() deterministic hash-based proxy (SCENARIO-VERIFY-125, Exp 433) |
| REQ-VERIFY-094 | Not Started | Implemented | CarnotThinkProbe generative 3-step CoT verifier — ThinkVerdict + ThinkProbeResult + build_think_probe_prompt + parse_think_probe_output + CarnotThinkProbe.probe/benchmark (SCENARIO-VERIFY-126/127/128, Exp 444) |
| REQ-VERIFY-095 | Not Started | Implemented | CarnotThinkProbe CI-safe stub — returns uncertain verdict without GPU, enabling CI-only testing (SCENARIO-VERIFY-126, Exp 444) |
| REQ-AGENT-001 | Not Started | Implemented | SAVeR multi-turn constraint state propagation — ConstraintState dataclass carrying accumulated_facts, active_constraints, facts_established, model_id across steps (SCENARIO-AGENT-001/002/003, Exp 362) |
| REQ-AGENT-002 | Not Started | Implemented | SAVeR commit-gate — propose_step() blocks action when constraint violations survive max_repair_attempts; committed=False prevents fact accumulation (SCENARIO-AGENT-001/002/003, Exp 362) |

### REQ-VERIFY-092: SpilledEnergyDetector Per-Token Logit-Discrepancy Hallucination Signal

**Description:** Implements the "spilled energy" signal from arXiv 2602.18671 (ICLR 2026).
For each generated token position t, spilled energy = log(sum_j exp(logit_j/T)) - sum_j p_j * logit_j,
where p_j = softmax(logit_j/T). This equals the gap between the partition function (free energy)
and the expected logit value — the "intensity discarded" by softmax normalization.
High per-token spilled energy indicates the model's logit space is uncertain (spread across many
tokens), which correlates with hallucination risk. Unlike semantic entropy (post-hoc, full response),
spilled energy is measurable per-token during streaming generation.

**Acceptance Criteria:**
- `SpilledEnergyDetector.score(logits_per_token)` computes per-token spilled energy and returns
  `SpilledEnergyDetectorResult` with mean, max, high_spill_fraction, should_verify
- `should_verify=True` iff `high_spill_fraction > high_spill_fraction_threshold`
- `high_spill_fraction` = fraction of tokens with spilled_energy > spill_threshold
- Default thresholds: spill_threshold=2.0, high_spill_fraction_threshold=0.2
- Hardware path: logit operations are native GPU tensor ops, ~0.01ms per token

Spec: REQ-VERIFY-092

### REQ-VERIFY-093: SpilledEnergyDetector CI-Safe Text Mode

**Description:** When logits are not available (CI environment, offline scoring),
SpilledEnergyDetector.score_from_text(response_text) provides a deterministic hash-based
proxy that returns a valid SpilledEnergyDetectorResult without GPU access.
The hash is deterministic so repeated calls on the same text produce the same result.

**Acceptance Criteria:**
- `score_from_text(response_text: str)` returns `SpilledEnergyDetectorResult` without JAX/GPU
- Result is deterministic: same input text → same output
- The hash-based energy proxy is in a plausible range (not always triggering or always skipping)

Spec: REQ-VERIFY-093

### SCENARIO-VERIFY-123: Uncertain Logits Trigger should_verify

**Given** a logit array where all tokens have equal logits (uniform distribution)
**When** `SpilledEnergyDetector().score(logits)` is called
**Then** spilled energy per token equals approximately log(vocab_size) (high)
**And** `result.should_verify == True` when high_spill_fraction > threshold
**And** `result.mean_spilled > 0`

### SCENARIO-VERIFY-124: Confident Logits Do Not Trigger should_verify

**Given** a logit array where one token has a very large logit (peaked distribution, T=1)
**When** `SpilledEnergyDetector().score(logits)` is called with spill_threshold=2.0
**Then** spilled energy per token is approximately 0 (low)
**And** `result.should_verify == False` (confident model, skip verification)

### SCENARIO-VERIFY-125: CI Text Mode Returns Valid Result Without GPU

**Given** a response text string, no GPU hardware available
**When** `SpilledEnergyDetector().score_from_text(response_text)` is called
**Then** a valid `SpilledEnergyDetectorResult` is returned
**And** `result.should_verify` is a bool
**And** calling twice with the same text returns the same result (deterministic)

### REQ-VERIFY-094: CarnotThinkProbe Generative 3-Step CoT Verifier

**Description:** Implements a generative Process Reward Model pre-filter (ThinkPRM,
arXiv 2504.16828, OpenAI April 2025). Given an LLM response, a secondary Qwen3.5-0.8B
generates a 3-step verification chain-of-thought:
  Step 1: Extract the arithmetic/logical claim.
  Step 2: Check if the claim is correct.
  Step 3: State verdict (incorrect / uncertain / correct).

If the CoT concludes 'incorrect', Ising verification is skipped immediately (fast-path).
Only 'uncertain' or 'correct' verdicts proceed to Ising. This creates a fast-path/slow-path
architecture that is dramatically more sample-efficient than discriminative classifiers
(ThinkPRM achieves SOTA on MATH-500 and AIME '24 with 1% of supervision labels).

**Acceptance Criteria:**
- `ThinkVerdict(verdict: Literal['incorrect','uncertain','correct'], confidence: float, reasoning_steps: list[str])`
- `ThinkProbeResult(response_text: str, verdict: ThinkVerdict, should_run_ising: bool, latency_ms: float)`
- `build_think_probe_prompt(response: str) -> str` returns 3-step verification prompt
- `parse_think_probe_output(output: str) -> ThinkVerdict` parses VERDICT: line; fallback 'uncertain'
- `CarnotThinkProbe(llm_caller=None, confidence_threshold=0.8).probe(response) -> ThinkProbeResult`
- `should_run_ising=False` iff verdict=='incorrect' (fast-path skip)
- `benchmark(responses, ground_truth) -> dict` with skip_rate, tp_rate, fp_rate

Spec: REQ-VERIFY-094
SCENARIO-VERIFY-126, SCENARIO-VERIFY-127, SCENARIO-VERIFY-128

### REQ-VERIFY-095: CarnotThinkProbe CI-Safe Stub

**Description:** When `llm_caller=None`, CarnotThinkProbe.probe() returns a ThinkProbeResult
with `ThinkVerdict('uncertain', 0.5, [])` without requiring GPU or network access.
This enables CI testing of the fast-path/slow-path routing logic without real LLM inference.

**Acceptance Criteria:**
- `CarnotThinkProbe(llm_caller=None).probe(response)` returns `ThinkProbeResult` with
  `verdict.verdict == 'uncertain'` and `should_run_ising == True`
- No GPU, no network access required
- Result is deterministic

Spec: REQ-VERIFY-095
SCENARIO-VERIFY-126

### SCENARIO-VERIFY-126: CI Stub Returns Uncertain Without GPU

**Given** no GPU and `llm_caller=None`
**When** `CarnotThinkProbe().probe("any response text")` is called
**Then** `result.verdict.verdict == 'uncertain'`
**And** `result.should_run_ising == True`
**And** `result.verdict.confidence == 0.5`
**And** no exception is raised

### SCENARIO-VERIFY-127: Incorrect Verdict Triggers Fast-Path Skip

**Given** an `llm_caller` that returns output containing `VERDICT: incorrect`
**When** `CarnotThinkProbe(llm_caller=mock_caller).probe(response)` is called
**Then** `result.verdict.verdict == 'incorrect'`
**And** `result.should_run_ising == False` (fast-path: skip Ising)

### SCENARIO-VERIFY-128: Benchmark Reports Skip Rate And Error Rates

**Given** a corpus of 50 correct + 50 wrong responses with a deterministic mock LLM caller
**When** `CarnotThinkProbe(llm_caller=mock).benchmark(responses, ground_truth)` is called
**Then** `result['skip_rate']` is a float in [0.0, 1.0]
**And** `result['tp_rate']` is the fraction of wrong responses flagged as incorrect
**And** `result['fp_rate']` is the fraction of correct responses wrongly flagged as incorrect

### REQ-PROBE-005: ThinkProbeV2 60-Minute Budget

ThinkProbeV2 shall run the full 50-question × 2-model benchmark within a 60-minute wall-clock
budget (55 minutes internal, 5 minutes reserved for artifact write and cleanup). This resolves
RETRO-029 where the 20-minute budget caused a complete timeout with no result.

**Acceptance Criteria:**
- `ThinkProbeV2(budget_minutes=60)` is constructable with no error
- The internal budget used during `run()` is `budget_minutes * 60 * 0.916` seconds
  (i.e. 55 of 60 minutes, leaving 5 minutes for artifact write)
- When all questions are answered within budget, `result.status == 'complete'`

Spec: REQ-PROBE-005
SCENARIO-PROBE-010

### REQ-PROBE-006: ThinkProbeV2 Partial Verdict Mode

When a `ThinkProbeV2.run()` call times out before all questions are answered, it shall
emit a `ThinkProbeV2Result` with `status='partial'` rather than raising an exception.
The result shall faithfully describe the completed fraction so the conductor and operator
can assess progress without re-running the full experiment.

**Why honest negatives instead of silent timeout:**
RETRO-029 lost the entire Exp 444 run when the 20-minute watchdog fired `sys.exit(1)`.
No partial data survived. A `ThinkProbeV2Result` with `honest_verdict='partial_30_of_50'`
is far more useful than an empty result file — it tells the researcher exactly how far the
run got and allows continuation checkpointing in a future run.

**Acceptance Criteria:**
- `ThinkProbeV2Result.is_partial` is `True` when `n_completed < n_total`
- `ThinkProbeV2Result.completion_fraction` equals `n_completed / n_total`
- `ThinkProbeV2Result.honest_verdict` is `'complete'` when `n_completed == n_total`
- `ThinkProbeV2Result.honest_verdict` is `'partial_{n_completed}_of_{n_total}'` when partial
- `ThinkProbeV2Result.honest_verdict` is `'timeout_no_data'` when `n_completed == 0`
- Timeout during `run()` does NOT raise — returns partial result instead

Spec: REQ-PROBE-006
SCENARIO-PROBE-011

### REQ-PROBE-007: ThinkProbeV2 Incremental Checkpoint Every 10 Questions

ThinkProbeV2 shall write an incremental checkpoint after every 10 completed questions.
This ensures that a timeout after question 40 preserves 40 answers rather than losing
the entire run (RETRO-029 root cause: no checkpoint → zero recovery on timeout).

**Acceptance Criteria:**
- `ThinkProbeV2(checkpoint_interval=10)` is constructable
- After every `checkpoint_interval` questions, `_checkpoint(results_so_far, step)` is called
- Checkpoint is written to `results/checkpoints/experiment_455/checkpoint.json`
- On timeout at question 40, checkpoint at step 40 exists and contains 40 answers

Spec: REQ-PROBE-007
SCENARIO-PROBE-012

### SCENARIO-PROBE-010: ThinkProbeV2 Completes Within 60-Minute Budget

**Given** `ThinkProbeV2(budget_minutes=60)` and a fast mock `inference_fn` (< 1 ms per call)
**When** `run(questions_50, inference_fn)` is called
**Then** `result.status == 'complete'`
**And** `result.n_completed == 50`
**And** `result.n_total == 50`
**And** `result.completion_fraction == 1.0`
**And** `result.honest_verdict == 'complete'`

### SCENARIO-PROBE-011: ThinkProbeV2 Emits Partial Result On Timeout

**Given** `ThinkProbeV2(budget_minutes=0.0001)` (sub-second budget) and a slow `inference_fn`
  that sleeps 1 second per question
**When** `run(questions_50, inference_fn)` is called
**Then** no exception is raised
**And** `result.n_completed < 50`
**And** `result.is_partial == True`
**And** `result.honest_verdict` starts with `'partial_'` or equals `'timeout_no_data'`

### SCENARIO-PROBE-012: ThinkProbeV2 Writes Checkpoint Every 10 Questions

**Given** `ThinkProbeV2(budget_minutes=60, checkpoint_interval=10)` with a fast `inference_fn`
  and a mock `_checkpoint` recorder
**When** `run(questions_50, inference_fn)` is called and completes
**Then** `_checkpoint` was called at least once for every 10 completed questions
**And** each checkpoint call received `step` equal to a multiple of `checkpoint_interval`

### REQ-PROBE-008: ThinkProbeV2 Live Run Uses DeliverableGuard

When Experiment 465 runs ThinkProbeV2 on live GPU, it shall use `DeliverableGuard` to
assert that `results/experiment_465_think_probe_live.json` is present on disk as the
FINAL assertion in `main()`.  This closes the RETRO-032/033/036 hole where the conductor
reported success but the deliverable file was absent.

Spec: REQ-PROBE-008
SCENARIO-PROBE-013

### REQ-PROBE-009: Live Think Probe Result Includes inference_mode Field

The artifact written by Experiment 465 shall include the field
`inference_mode='live_gpu'` to distinguish it from deferred/simulated runs.
`LiveThinkProbeResult` extends `ThinkProbeV2Result` with `inference_mode: str`,
`model_id: str`, and `gpu_used: str` fields.

Spec: REQ-PROBE-009
SCENARIO-PROBE-014

### SCENARIO-PROBE-013: DeliverableGuard Raises On Missing File

**Given** `DeliverableGuard('results/experiment_465_think_probe_live.json')`
**When** `assert_written()` is called and the file does not exist
**Then** `FileNotFoundError` is raised with a message citing RETRO-032/033/036

### SCENARIO-PROBE-014: LiveThinkProbeResult is_partial When n_completed < n_total

**Given** `LiveThinkProbeResult(n_completed=30, n_total=50, results=[], inference_mode='live_gpu', model_id='google/gemma-4-E4B-it', gpu_used='cuda:0')`
**When** `is_partial` is evaluated
**Then** `is_partial is True`
**And** `inference_mode == 'live_gpu'`

### REQ-PROBE-010: ThinkProbeV2 Live Run Uses GPUVRAMGate Before Model Load

Before loading a Gemma model for a ThinkProbeV2 live GPU run, the experiment shall
invoke `GPUVRAMGate(min_free_gb=8.0).__enter__()` to ensure sufficient VRAM is
available.  If the gate raises `GPUVRAMInsufficientError`, the experiment shall emit
an artifact with `status='gpu_vram_insufficient'` and exit without loading the model.

This requirement closes RETRO-042: Exp 465 deferred because zombie VRAM prevented
model load.  The gate kills zombies and waits before failing, giving the experiment
the best possible chance of a clean load.

**Implementation Status:** Done (Exp 482)

Spec: REQ-PROBE-010

### REQ-PROBE-011: ThinkProbeV2 Live Result Requires n_completed >= 40 for Viable Verdict

A `ThinkProbeLiveV3Result` is considered viable (think_probe_viable) when:
- `completion_fraction >= 0.80`  (at least 40 of 50 questions completed)
- `tp_rate >= 0.70`              (true-positive flag rate on the 50-question corpus)
- `fp_rate <= 0.20`              (false-positive flag rate on the 50-question corpus)

The `is_viable` property encodes this three-way threshold.  A partial run with
`n_completed < 40` is NOT viable regardless of tp_rate/fp_rate, because the
statistical power is too low to trust the rate estimates.

**Implementation Status:** Done (Exp 482)

Spec: REQ-PROBE-011

### SCENARIO-PROBE-015: ThinkProbeLiveV3Result is_viable When n_completed=42

**Given** `ThinkProbeLiveV3Result(inference_mode='live_gpu', model_id='google/gemma-4-E4B-it', n_completed=42, n_total=50, gpu_vram_gate_fired=True, skip_rate=0.0, tp_rate=0.80, fp_rate=0.15)`
**When** `is_viable` is evaluated
**Then** `is_viable is True`
**And** `retro_036_closed is True`

### SCENARIO-PROBE-016: ThinkProbeLiveV3Result is_viable False When n_completed=30

**Given** `ThinkProbeLiveV3Result(inference_mode='live_gpu', model_id='google/gemma-4-E4B-it', n_completed=30, n_total=50, gpu_vram_gate_fired=True, skip_rate=0.1, tp_rate=0.80, fp_rate=0.10)`
**When** `is_viable` is evaluated
**Then** `is_viable is False`  (completion_fraction=0.60 < 0.80 threshold)

---

## Phase 3 Seed Requirements (Exp 435a — NOT production, exploratory only)

> **Phase 3 seed work.** These requirements plant a concrete bridge from discrete Ising
> (Phase 1 sweet spot) to continuous energy landscapes (Kona's domain).  They are
> intentionally minimal — CPU-only, no GPU, <30 min.  Do NOT gate any production
> capability on these requirements.

### REQ-KONA-001: Continuous EBM Minimum Recovery

The continuous EBM shall recover the same energy minimum as the Ising baseline on a
matching 10-variable constraint problem, where "same" is defined as:

- L2 distance between the continuous minimiser and the Ising ground-state sample is
  ≤ 0.1 (after tanh squashing of continuous variables to [-1, 1])
- Sign agreement fraction (fraction of coordinates with identical sign) is > 0.9

This requirement validates that a gradient-descent–based continuous sampler and a
discrete simulated-annealing sampler converge to the same region of the energy landscape
when initialised from the same Ising coupling matrix J and bias h.

**Rationale:** Phase 3 (Kona parity) requires non-autoregressive reasoning over
continuous latent variables.  This seed experiment proves the energy function is
consistent across the discrete↔continuous boundary — without that invariant, the
Phase 3 architecture has no foundation.

Spec: REQ-KONA-001, SCENARIO-KONA-001, SCENARIO-KONA-002

### SCENARIO-KONA-001: Continuous Minimiser Within L2 Tolerance

**Given** a random 10-variable Ising model with sparse couplings (density ≈ 0.3) and
  seed 42
**When** simulated annealing produces an approximate Ising ground-state sample AND
  gradient descent on E(x) = -0.5*x^T*J*x - h^T*x with tanh squashing produces a
  continuous minimiser
**Then** `compare_minima(ising_sample, continuous_sample)['l2_distance'] <= 0.1`
**And** `compare_minima(...)['sign_agreement'] > 0.9`

### SCENARIO-KONA-002: Honest Verdict Written To Artifact

**Given** the experiment has run to completion (whether it matched or not)
**When** `build_kona_artifact()` is called with the comparison result
**Then** `artifact['schema'] == 'carnot.kona_seed.v1'`
**And** `artifact['honest_verdict']` is one of:
  - `'continuous_matches_ising'` (L2 < 0.1 AND sign_agreement > 0.9)
  - `'partial_match'` (sign_agreement > 0.7)
  - `'failed_to_match'` (otherwise)

### REQ-KONA-002: Langevin Dynamics Sampling Achieves L2 < 0.5 vs Ising Ground State

The `sample_langevin` function shall produce a continuous-variable energy minimum that is
within L2 < 0.5 of the discrete Ising ground-state sample on the 10-variable reference
problem (seed 42, density 0.3), averaged over at least 20 independent trials.

Langevin dynamics: x_{t+1} = x_t - lr * grad_E(x_t) + noise_scale * sqrt(2*lr) * eps_t,
where eps_t ~ N(0, I).  Temperature schedule ('cosine', 'linear', or 'constant') controls
noise annealing.

**Rationale:** Exp 435a showed gradient descent alone achieves L2=2.69 (partial match
only). Langevin noise injects thermal fluctuations that help escape local energy minima,
mimicking how physical sampling in thermodynamic systems discovers lower-energy states.
This is the mechanism behind Generative Thermodynamic Computing (arXiv 2506.15121).

**Implementation Status:** Done (Exp 446)

### REQ-KONA-003: Energy Matching Trajectory Converges Faster Than Gradient Descent

The `sample_energy_matching` function shall converge to a lower-energy state than
`sample_continuous` (gradient descent) using fewer total gradient evaluations, on the
10-variable reference problem averaged over 20 trials.

Energy Matching trajectory: x = x - step_size * grad_E(x) / (||grad_E(x)|| + eps)
(normalised gradient flow).  Initialised from Gaussian noise, this implements a
constant-speed flow toward the energy minimum regardless of gradient magnitude
(arXiv 2504.10612, NeurIPS 2025).

**Rationale:** Standard gradient descent slows near flat regions and overshoots near
steep ones.  Normalised gradient flow maintains constant convergence speed, providing
more predictable and efficient energy minimisation — a key property for Kona's
non-autoregressive reasoning over continuous latent spaces.

**Implementation Status:** Done (Exp 446)

### SCENARIO-KONA-003: Langevin Noise Escapes Local Minima

**Given** a 10-variable Ising model (seed 42, density 0.3) with a known Ising ground state
**When** `sample_langevin(model, n_steps=2000, lr=0.005, noise_scale=0.1,
  temp_schedule='cosine')` is run 20 independent trials
**Then** mean L2 distance to Ising ground state is < 0.5
**And** mean sign_agreement > 0.7

### SCENARIO-KONA-004: Energy Matching Normalized Gradient Flow

**Given** a ContinuousEBM with known coupling J and bias h
**When** `sample_energy_matching(model, n_steps=1000, n_flow_steps=10)` is called
**Then** the output is an array of shape (n,) with values in (-1, 1) due to tanh squashing
**And** the output energy E(x) is lower than the energy of a Gaussian-initialised point

### SCENARIO-KONA-005: compare_samplers Reports All Three Sampler Results

**Given** a ContinuousEBM fitted from an Ising model and its discrete ground state
**When** `compare_samplers(model, ising_ground_state, n_trials=10)` is called
**Then** the result dict has keys 'gradient_descent', 'langevin', 'energy_matching'
**And** each sub-dict has keys 'mean_l2', 'std_l2', 'mean_sign_agreement'
**And** 'best_sampler' identifies the sampler with the lowest mean_l2
**And** the artifact is serialisable to JSON without error

## Gemma4 Loader Requirements (RETRO-028 Fix)

### REQ-LOADER-001: GemmaTransformersLoader Uses HuggingFace Transformers

The `GemmaTransformersLoader` class shall load Gemma4 models exclusively via
`AutoModelForCausalLM.from_pretrained` (HuggingFace transformers library), never via
any llama.cpp backend.

**Rationale:** llama.cpp tokenizer bug (GitHub issue llama.cpp#21516) causes Gemma4
models to emit infinite `<unused8>` tokens (token ID 14) instead of valid text.
The model never actually ran in Exp 439 — the 0.0% GSM8K accuracy is a false negative.
Published Gemma4 accuracy on GSM8K is 75-80%.  Using transformers directly bypasses the
buggy llama.cpp tokenizer entirely.

**Implementation Status:** Done (Exp 450)

### REQ-LOADER-002: GemmaTransformersLoader Validates Output for Unused Tokens

The `GemmaTransformersLoader.is_valid_output(text: str) -> bool` method shall return
`False` if the generated text consists entirely of `<unused>` tokens (any sequence of
`<unusedN>` tokens with no other content), and `True` otherwise.

**Rationale:** The llama.cpp#21516 bug produces silent failure — the model appears to
run but every generated token is `<unused8>` (token_id=14 in the Gemma4 tokenizer).
Without explicit validation, a caller would observe 100% "generation success" but 0%
accuracy because every response is garbage.

**Implementation Status:** Done (Exp 450)

### SCENARIO-LOADER-001: is_valid_output Rejects All-Unused Token Text

**Given** a string consisting entirely of `<unused>` tokens (e.g. `<unused><unused><unused>`)
**When** `GemmaTransformersLoader.is_valid_output(text)` is called
**Then** the method returns `False`

### SCENARIO-LOADER-002: is_valid_output Accepts Normal Text

**Given** a string containing real natural language (e.g. `The answer is 42.`)
**When** `GemmaTransformersLoader.is_valid_output(text)` is called
**Then** the method returns `True`

## Gemma4 Quantized Loader Requirements (Exp 500, RETRO-048)

### REQ-LOADER-003: Gemma4QuantizedLoader Loads GGUF Q4_K_M Checkpoint via llama-cpp-python

`Gemma4QuantizedLoader(model_path, n_gpu_layers=-1, max_tokens=512)` shall load a
GGUF Q4_K_M checkpoint using llama-cpp-python with `n_gpu_layers=-1` for full GPU
offload.  `load()` returns `True` on success.  When llama-cpp-python is not installed,
the CI stub path returns `True` (mocked) so the test suite passes without GPU hardware.

**Why Q4_K_M:** The conductor process holds ~9 GiB of GPU 0 VRAM.  Gemma4 at FP16
requires 14.89 GiB — conductor + FP16 = ~24.7 GiB, exceeding the 24 GiB RTX 3090.
Q4_K_M quantization reduces Gemma4 to ~8-10 GiB; conductor + quantized model = ~18 GiB,
fitting within 24 GiB with ~6 GiB headroom.  This unblocks RETRO-048 (five consecutive
deferred milestones).

**Implementation Status:** Done (Exp 500)

### REQ-LOADER-004: Gemma4QuantizedLoader Validates VRAM Usage After Load

`Gemma4QuantizedLoader.vram_usage_gb()` shall return the GPU 0 VRAM consumed by the
loaded model via pynvml.  `is_within_budget(max_gb=10.0)` shall return `True` iff
`vram_usage_gb() <= max_gb`.

**Implementation Status:** Done (Exp 500)

### REQ-LOADER-005: Gemma4QuantizedLoader.accuracy_check Validates Quantization Quality

`Gemma4QuantizedLoader.accuracy_check(n_questions=10)` shall run `n_questions` GSM8K
sample questions through the loaded model and return the fraction answered correctly
as a float in [0, 1].  A result >= 0.60 is accepted as passing quantization quality.

**Implementation Status:** Done (Exp 500)

### SCENARIO-LOADER-003: Gemma4QuantizedLoader.load Returns True on Success

**Given** a valid GGUF model path (or CI stub when llama-cpp-python is not installed)
**When** `Gemma4QuantizedLoader(model_path).load()` is called
**Then** it returns `True`

### SCENARIO-LOADER-004: is_within_budget Returns Correct Result Against Budget

**Given** `vram_usage_gb()` returns 9.5
**When** `is_within_budget(10.0)` is evaluated
**Then** it returns `True`

**Given** `vram_usage_gb()` returns 10.5
**When** `is_within_budget(10.0)` is evaluated
**Then** it returns `False`

### SCENARIO-LOADER-005: accuracy_check Returns Float in [0, 1]

**Given** a loaded Gemma4QuantizedLoader (real or CI stub)
**When** `accuracy_check(n_questions=10)` is called
**Then** the return value is a float in [0.0, 1.0]

## VPRM Arithmetic Verifier Requirements (Exp 454)

### REQ-EXTRACT-027: VPRMArithmeticVerifier Implements Rule-Based Arithmetic Step Checks

The system shall provide `VPRMArithmeticVerifier` with six deterministic rule families:
addition, subtraction, multiplication, division, percentage, and unit consistency.
Each rule matches IT-model prose patterns (e.g. "47 plus 28 equals 75") without any
LLM call and verifies the stated result against the arithmetically computed result.

The abstract base `ArithmeticRule` defines `check(step_text: str) -> RuleVerdict | None`,
returning `None` when the rule's pattern is not present in the step text, and a
`RuleVerdict` when the pattern matches (whether the arithmetic is correct or not).

`RuleVerdict` carries: `rule_name`, `passed`, `computed_value`, `stated_value`,
`error_magnitude` (all Optional where not applicable, e.g. unit consistency).

**Rationale:** Deterministic rules are immune to reward hacking — a neural judge can be
fooled with adversarial inputs; an arithmetic identity check cannot.  arXiv 2601.17223
reports 20% F1 gain over neural process reward models using this approach.

**Implementation Status:** Done (Exp 454)

Spec: REQ-EXTRACT-027, SCENARIO-EXTRACT-052, SCENARIO-EXTRACT-053

### REQ-EXTRACT-028: VPRM Rules Are Deterministic — Same Input Always Produces Same Verdict

`VPRMArithmeticVerifier.verify_step(step_text)` and
`VPRMArithmeticVerifier.detect_violations(cot_text)` shall be deterministic:
for identical input text, the output is always identical across runs, processes,
and Python interpreter restarts.  No random seeds, no model sampling, no I/O.

**Rationale:** Determinism is the defining property that makes VPRM immune to reward
hacking.  A verifier that produces different verdicts for the same input cannot be
trusted as ground truth.

**Implementation Status:** Done (Exp 454)

Spec: REQ-EXTRACT-028, SCENARIO-EXTRACT-052

### REQ-EXTRACT-029: VPRM Detects IT Model Arithmetic Errors With No LLM Call

`VPRMArithmeticVerifier` shall detect arithmetic errors in IT-model-style CoT steps
(written in natural prose) without invoking any LLM, without loading any model weights,
and without network access.  The entire verification pipeline runs in pure Python on CPU.

`VPRMArithmeticVerifier.f1_score(ground_truth, predicted)` shall compute standard binary
F1 score, matching sklearn convention (returning 0.0 when the denominator is zero).

Experiment 454 measures F1 improvement over `ArithmeticExtractor` on 20 hardcoded
IT-style samples (10 correct, 10 with errors):
- Baseline (ArithmeticExtractor): F1 = 0.0 (regex finds no prose patterns)
- VPRM: F1 = 1.0 (all 10 errors detected, 0 false positives)
- f1_improvement = 1.0, honest_verdict = 'vprm_better'

**Implementation Status:** Done (Exp 454)

Spec: REQ-EXTRACT-029, SCENARIO-EXTRACT-054

### SCENARIO-EXTRACT-052: AdditionRule Detects '47 plus 28 equals 76' As Failed

**Given** the step text "47 plus 28 equals 76"
**When** `AdditionRule().check(step_text)` is called
**Then** the verdict is not None
**And** `verdict.passed == False`
**And** `verdict.computed_value == 75.0`
**And** `verdict.stated_value == 76.0`
**And** `verdict.error_magnitude == 1.0`

### SCENARIO-EXTRACT-053: VPRMArithmeticVerifier Detects Violations ArithmeticExtractor Misses

**Given** the step text "47 plus 28 equals 76" (arithmetic error in IT prose)
**When** `ArithmeticExtractor().extract(step_text)` is called
**Then** it returns an empty list (prose format not matched by equation regex)
**When** `VPRMArithmeticVerifier().detect_violations(step_text)` is called
**Then** it returns at least one failed `RuleVerdict` with `passed=False`

### SCENARIO-EXTRACT-054: VPRM F1 Score Exceeds Baseline On 20-Sample IT Corpus

**Given** 20 IT-model-style CoT steps (10 correct, 10 with arithmetic errors)
**When** Exp 454 runs ArithmeticExtractor and VPRMArithmeticVerifier on all 20
**Then** `baseline_f1 == 0.0` (ArithmeticExtractor finds no violations in IT prose)
**And** `vprm_f1 == 1.0` (VPRM detects all 10 errors with 0 false positives)
**And** `f1_improvement == 1.0`
**And** `honest_verdict == 'vprm_better'`

### REQ-SELFLEARN-010: ConstraintAdditionFromMemory Monitors and Adds Constraints

The repository shall provide `ConstraintAdditionFromMemory` in
`python/carnot/pipeline/constraint_addition.py` with the following contract:

- `observe(violation_type: str, step_text: str) -> None` — record one instance of
  a violation type with its step text (examples capped at 5 per type).
- `check_and_add(pipeline=None) -> list[str]` — for each violation type whose
  observation count is >= threshold, look up the canonical constraint name from the
  Eidoku taxonomy mapping and add it (idempotent — same constraint not added twice).
  Returns sorted list of constraint names added in this call.
- `get_pattern_counts() -> dict[str, int]` — snapshot of current counts per type.
- `get_patterns() -> list[ViolationPattern]` — full list of observed violation patterns
  sorted by type.
- `ViolationPattern(type, count, example_steps)` — dataclass carrying one violation
  family's evidence record.

**Rationale:** Exp 134 proved that reweighting existing constraints (adaptive vs fixed)
produces identical F1 on 500 questions.  The constraint VOCABULARY must be EXPANDED when
a new error class is detected; reweighting the existing vocabulary amplifies noise.
See research-program.md Goal #1.

**Implementation Status:** Done (Exp 456)

Spec: REQ-SELFLEARN-010, SCENARIO-SELFLEARN-010

### REQ-SELFLEARN-011: Threshold Configurable via CARNOT_ADDITION_THRESHOLD

The default activation threshold for `ConstraintAdditionFromMemory` is 5 observations.
This value is configurable via the `CARNOT_ADDITION_THRESHOLD` environment variable.
An explicit `threshold` constructor argument takes precedence over the environment
variable.

**Rationale:** Five observations means the pattern has appeared across multiple distinct
responses, providing high confidence that the error class is genuinely recurring rather
than a noise spike.  The env var allows researchers to tune this without code changes.

**Implementation Status:** Done (Exp 456)

Spec: REQ-SELFLEARN-011, SCENARIO-SELFLEARN-011

### REQ-SELFLEARN-012: Session 2 FP Rate Decreases After Constraint Addition

When the two-session cross-session relay is run:
- Session 1 (no carry constraint): pipeline cannot detect carry errors → FP rate ≈ 1.0.
- Feed Session 1 violation observations into `ConstraintAdditionFromMemory`.
- `check_and_add()` returns `['carry_check_constraint']`.
- Session 2 (carry constraint active): pipeline detects carry errors → FP rate ≈ 0.0.
- `fp_rate_delta = session2_fp_rate - session1_fp_rate < 0` (improvement).
- `honest_verdict = 'improvement'`.

**Implementation Status:** Done (Exp 456)

Spec: REQ-SELFLEARN-012, SCENARIO-SELFLEARN-012

### SCENARIO-SELFLEARN-010: After 5 Carry Observations, check_and_add Returns Carry Constraint

**Given** a `ConstraintAdditionFromMemory(threshold=5)` instance
**And** 5 calls to `observe('carry', step_text)` have been made
**When** `check_and_add()` is called
**Then** the return value is `['carry_check_constraint']`

### SCENARIO-SELFLEARN-011: Below Threshold Returns Empty List

**Given** a `ConstraintAdditionFromMemory(threshold=5)` instance
**And** only 4 calls to `observe('carry', step_text)` have been made
**When** `check_and_add()` is called
**Then** the return value is `[]`
**And** a second call to `check_and_add()` after 5+ observations does not re-add
  a constraint that was already added (idempotency guarantee)

### SCENARIO-SELFLEARN-012: Exp 456 Two-Session Relay Achieves improvement Verdict

**Given** the Exp 456 relay is run with 50 synthetic carry-error questions
**When** Session 1 runs with no carry constraint
**Then** `session1_fp_rate == 1.0` (all carry errors missed)
**When** ConstraintAdditionFromMemory feeds 50 carry observations (>> threshold=5)
**Then** `check_and_add()` returns `['carry_check_constraint']`
**When** Session 2 runs with carry_check_constraint active
**Then** `session2_fp_rate == 0.0` (all carry errors detected)
**And** `fp_rate_delta == -1.0`
**And** `honest_verdict == 'improvement'`

### REQ-SELFLEARN-013: LSEBMConstraintReplayer Trains an EBM on Session 1 Violation Distribution

The `LSEBMConstraintReplayer` shall:
- Accept a list of violation type strings from Session 1.
- Compute a frequency distribution over violation types.
- Train a small Ising EBM on the violation distribution using contrastive divergence (n_iter steps).
- Store the trained EBM for later sampling.
- Be pure Python + JAX, CPU-only, completing in < 5 seconds for ≤ 100 iterations.

**Why LSEBM replay (arXiv 2501.05495):**
    LSEBM-CL (Lifelong Sequence EBM with Continual Learning) prevents catastrophic
    forgetting by training a generative EBM on task data and replaying synthetic samples
    when starting a new task.  Here the "tasks" are sessions and the "data" are violation
    type distributions.  Without replay, Session 2 starts cold: no prior knowledge of
    which error types are common.  With replay, Session 2 knows 'carry errors are likely'
    before any real questions arrive.

**Implementation Status:** Done (Exp 457)

Spec: REQ-SELFLEARN-013, SCENARIO-SELFLEARN-013

### REQ-SELFLEARN-014: LSEBMConstraintReplayer Generates N Synthetic Violations to Warm-Start Session 2

The `LSEBMConstraintReplayer.generate_replay(n)` shall:
- Sample N synthetic violation type strings from the trained EBM.
- Return a list of violation type strings drawn from the learned distribution.
- Guarantee that violation types returned are from the vocabulary seen during `fit()`.
- The `warm_start(memory: SessionMemory)` method shall inject the synthetic violations
  into the session memory's template observation counts and return the count of templates
  warm-started (i.e., number of distinct violation types injected).

**How warm-start differs from Exp 448's template loading:**
    Exp 448 directly loaded Session 1 templates into Session 2 — exact replay of stored
    state.  LSEBMCL replay uses the EBM as a generative model to produce NEW synthetic
    instances drawn from the learned distribution.  This means Session 2's warm-start
    is probabilistic (not deterministic) and does not require storing the full template
    library — only the compact EBM parameters.  The EBM also generalises: it can
    interpolate between observed violation types, potentially discovering boundary regions
    Exp 448's exact-copy approach could not reach.

**Implementation Status:** Done (Exp 457)

Spec: REQ-SELFLEARN-014, SCENARIO-SELFLEARN-014

### REQ-SELFLEARN-015: LSEBMCL Replay Improves Session 2 FP Rate vs Exp 448 Baseline

In the Exp 457 two-session relay:
- Session 1 baseline FP rate shall be recorded (equivalent to Exp 448 Session 0).
- After fitting `LSEBMConstraintReplayer` on Session 1 violations and calling `warm_start`,
  Session 2 shall be run with the warm-started template library.
- The `lsebmcl_fp_rate` shall be compared to the `exp448_fp_rate` baseline.
- The artifact shall report `honest_verdict = 'lsebmcl_better'` if `lsebmcl_fp_rate < exp448_fp_rate`,
  otherwise `'no_improvement'`.
- The constraint addition FP rate from Exp 456 shall also be recorded for three-way comparison.

**Implementation Status:** Done (Exp 457)

Spec: REQ-SELFLEARN-015, SCENARIO-SELFLEARN-015

### SCENARIO-SELFLEARN-013: fit + generate Produces Violation Types Seen in Training

**Given** a `LSEBMConstraintReplayer(n_replay=20, ebm_n_iter=100)` instance
**And** `fit(['carry', 'carry', 'carry', 'sign', 'carry'])` has been called
**When** `generate_replay(10)` is called
**Then** all returned strings are in `{'carry', 'sign'}`
**And** no string is returned that was not seen during fit

### SCENARIO-SELFLEARN-014: warm_start Returns Count > 0 After Fitting on Carry-Error Session

**Given** a `LSEBMConstraintReplayer` fitted on a list of 50 carry-error violations
**And** a `SessionMemory` with no prior state
**When** `warm_start(memory)` is called
**Then** the return value is > 0
**And** `generate_replay(20)` returns 20 violation type strings

### SCENARIO-SELFLEARN-015: Exp 457 Three-Way Comparison Produces honest_verdict

**Given** the Exp 457 relay is run with 50 synthetic carry-error questions
**When** Session 1 produces violations
**And** LSEBMConstraintReplayer is fit on those violations
**And** warm_start is called before Session 2
**When** Session 2 runs
**Then** the artifact contains `exp448_fp_rate`, `lsebmcl_fp_rate`, `constraint_add_fp_rate`
**And** `honest_verdict` is one of `'lsebmcl_better'` or `'no_improvement'`

### REQ-SELFLEARN-016: PPSConstraintLearner Isolates Constraint Weight Updates by Domain

The `PPSConstraintLearner` shall:
- Support three constraint domains: `ARITHMETIC`, `CODE`, `LOGICAL` (via `ConstraintDomain` enum).
- Maintain a separate `DomainParameterPartition` for each domain.
- When `fit_domain(domain, violations)` is called, ONLY the partition for that domain shall be updated.
- Partitions for all other domains shall remain numerically unchanged.

**Why domain isolation (arXiv 2512.15658 — PPSEBM):**
    LSEBMCL (Exp 457) uses a single parameter space for all constraint types.
    When arithmetic sessions and code sessions interleave, gradient updates bleed across
    domain boundaries — improving arithmetic detection can degrade code detection.
    PPSEBM fixes this by giving each domain an isolated parameter partition: updates
    to the arithmetic partition cannot affect the code or logical partitions.

**Implementation Status:** Done (Exp 470)

Spec: REQ-SELFLEARN-016, SCENARIO-SELFLEARN-016

### REQ-SELFLEARN-017: EBM Generates Synthetic Boundary Violations per Domain to Reinforce Partition Isolation

The `PPSConstraintLearner` shall implement `generate_boundary_violations(domain, n)`:
- Use the domain's `LSEBMConstraintReplayer` to generate `n` synthetic violation strings.
- Boundary violations are not random: the EBM encodes the learned distribution of violations
  for that domain, so generated samples stress-test the partition walls at the distribution boundary.
- The method shall return exactly `n` violation strings, all drawn from the domain's training vocabulary.

**Why boundary violations from EBM (not random):**
    Random violations would not reflect the domain's actual error distribution.
    The EBM encodes learned co-occurrence patterns (e.g., carry + sign errors tend to
    co-occur in arithmetic). Boundary violations sampled from near the distribution boundary
    are the hardest cases for the partition — the ones most likely to cause cross-domain bleed.
    Using EBM samples for stress-testing is the PPSEBM key insight.

**Implementation Status:** Done (Exp 470)

Spec: REQ-SELFLEARN-017, SCENARIO-SELFLEARN-017

### REQ-SELFLEARN-018: Partition Isolation Score > 0.8 After Independent Domain Training

The `PartitionIsolationScore` shall:
- Accept a list of `DomainParameterPartition` objects.
- Compute `score()` as the minimum cosine distance between gradient update vectors from any two distinct partitions.
- Implement `is_isolated(threshold=0.8)` returning `True` iff `score() >= threshold`.

**Why cosine distance as the isolation metric:**
    Cosine similarity measures the DIRECTION of gradient updates, not their magnitude.
    If two partitions are being updated in the same direction, their gradients are correlated —
    they share signal and cannot be truly isolated.  Cosine DISTANCE = 1 - cosine_similarity,
    so high cosine distance (near 1.0) means the gradients point in orthogonal or opposite
    directions — i.e., the partitions are learning independently.  A threshold of 0.8 means
    at most 0.2 of shared directional signal between any two domain partitions.

**Implementation Status:** Done (Exp 470)

Spec: REQ-SELFLEARN-018, SCENARIO-SELFLEARN-018

### SCENARIO-SELFLEARN-016: fit_domain(ARITHMETIC) Does NOT Change CODE or LOGICAL Partition Weights

**Given** a `PPSConstraintLearner` with partitions for ARITHMETIC, CODE, LOGICAL
**And** CODE and LOGICAL partitions have recorded their initial weight arrays
**When** `fit_domain(ConstraintDomain.ARITHMETIC, ['carry', 'carry', 'sign'])` is called
**Then** the CODE partition weights are bit-for-bit identical to their initial values
**And** the LOGICAL partition weights are bit-for-bit identical to their initial values

**Implementation Status:** Done (Exp 470)

### SCENARIO-SELFLEARN-017: generate_boundary_violations Returns Domain Vocabulary Strings

**Given** a `PPSConstraintLearner` fitted on ARITHMETIC domain with violations `['carry', 'sign']`
**When** `generate_boundary_violations(ConstraintDomain.ARITHMETIC, 10)` is called
**Then** exactly 10 strings are returned
**And** all strings are in `{'carry', 'sign'}`

**Implementation Status:** Done (Exp 470)

### SCENARIO-SELFLEARN-018: partition_isolation_score > 0.8 After Training 3 Domains Independently

**Given** a `PPSConstraintLearner` with three domains
**And** each domain has been fitted on its own violation set independently
**When** `PartitionIsolationScore(partitions).score()` is computed
**Then** the score is >= 0.8 (gradients are in sufficiently different directions)
**And** `is_isolated(threshold=0.8)` returns `True`

**Implementation Status:** Done (Exp 470)

### REQ-EORM-005: EBMCoTCalibrator Applies Langevin Dynamics to EORM Hidden State Before Scoring

The `EBMCoTCalibrator` shall:
- Accept an `EORMModel` and a configurable `n_langevin_steps` (default 10) at construction.
- Implement `calibrate_hidden(hidden: jnp.ndarray) -> jnp.ndarray`: run N steps of Langevin dynamics
  on the pooled EORM hidden state, moving it toward lower-energy regions of the EBM manifold.
- Implement `score(cot_input: CoTEnergyInput) -> float`: extract pooled hidden state from EORM,
  calibrate via Langevin, and return energy of calibrated representation.
- The Langevin update rule: `h_{t+1} = h_t - (ε/2) * ∇_h E(h_t) + sqrt(ε) * ξ_t`, `ξ_t ~ N(0, I)`.

**Why Langevin calibration (arXiv 2511.07124 — EBM-CoT):**
    Token-level encoding introduces noise into the pooled hidden state.  Langevin dynamics
    slide the representation along the energy gradient toward the nearest low-energy region
    on the EBM manifold, removing token-level noise while preserving semantic structure.
    This improves the discriminability of the energy readout between correct and incorrect CoT.

**Implementation Status:** Done (Exp 458)

Spec: REQ-EORM-005, SCENARIO-EORM-010

### REQ-EORM-006: EBMCoTCalibrator Improves EORM AUC vs Uncalibrated Baseline

The `EBMCoTCalibrator` shall:
- Implement `calibrated_auc(examples: list[dict]) -> float`: compute AUC-ROC of calibrated
  energy scores on labeled (question_text, response_text, label) examples.
- The calibrated AUC shall be reported alongside the uncalibrated baseline in the Exp 458 artifact.
- The artifact shall contain `baseline_auc`, `calibrated_auc`, `auc_improvement`, `target_met`,
  and `honest_verdict` (one of `'target_met'`, `'improvement'`, `'regression'`).

**Implementation Status:** Done (Exp 458)

Spec: REQ-EORM-006, SCENARIO-EORM-011

### REQ-EORM-007: EBMCoTCalibrator n_langevin_steps Configurable with Default 10

`EBMCoTCalibrator.__init__(eorm, n_langevin_steps: int = 10, step_size: float = 0.01)` shall:
- Accept `n_langevin_steps` as a named parameter with default value 10.
- Accept `step_size` (Langevin ε) with default value 0.01.
- Both parameters shall be accessible as instance attributes.

**Why 10 steps default:** Empirical sweet spot from arXiv 2511.07124 — sufficient to move off
the high-energy initial point without over-smoothing the representation.

**Implementation Status:** Done (Exp 458)

Spec: REQ-EORM-007

### SCENARIO-EORM-010: Calibrated Hidden Has Lower Energy Than Uncalibrated in Expectation

**Given** an `EBMCoTCalibrator` with `n_langevin_steps=20` and `step_size=0.1`
**And** an EORMModel with `out_weight = [2, 0, 0, ...]` (strong gradient signal)
**And** initial hidden state `h = [0, 0, ..., 0]` (energy ≈ 0)
**When** `calibrate_hidden(h)` is called 50 times with different random seeds
**Then** the mean calibrated energy across all 50 runs is < initial energy
**And** the Langevin drift term consistently decreases energy in expectation

### SCENARIO-EORM-011: calibrated_auc Does Not Regress Below Baseline on Synthetic Pairs

**Given** an `EBMCoTCalibrator` wrapping a randomly-initialized EORMModel
**And** 40 synthetic labeled CoT pairs (20 correct, 20 incorrect)
**When** `calibrated_auc(examples)` is called
**Then** the result is in [0.0, 1.0]
**And** the calibrated AUC is >= baseline uncalibrated AUC - 0.05 (regression tolerance)

### REQ-EORM-008: EBMCoTCalibratorV3 Uses 50 Langevin Steps by Default

`EBMCoTCalibratorV3.__init__(eorm, n_langevin_steps: int = 50, ...)` shall:
- Default `n_langevin_steps` to 50 (vs. 10 in v2).
- Accept `n_langevin_steps` as a configurable parameter.

**Why 50 steps (RETRO-034):** Exp 458 (v2) used 10 Langevin steps and reached AUC 0.5554 vs
the 0.600 target.  Root cause: 10 steps is insufficient for full hidden-state relaxation to the
low-energy manifold.  Increasing to 50 steps gives the dynamics enough time to reach a stable
low-energy region, improving discriminability between correct and incorrect CoT representations.

**Implementation Status:** Done (Exp 466)

Spec: REQ-EORM-008

### REQ-EORM-009: EP Coupling Update Applies Free/Clamped Phase Spin Correlations

`EPCouplingUpdate.update_couplings(J, free_spins, clamped_spins) -> jnp.ndarray` shall:
- Compute free-phase spin correlation: `free_corr[i,j] = mean(free_spins[:,i] * free_spins[:,j])`
- Compute clamped-phase spin correlation: `clamped_corr[i,j] = mean(clamped_spins[:,i] * clamped_spins[:,j])`
- Apply Equilibrium Propagation update: `J_new = J + η * (free_corr - clamped_corr)`
- Require no backpropagation — the update is a local Hebbian rule derived from OIM physics.

**Why EP coupling update (arXiv 2510.12934):** Equilibrium Propagation (OIM/EP) trains coupling
matrices using only two steady states (free phase and clamped phase), producing a weight update
that is equivalent to contrastive Hebbian learning.  No gradient backpropagation is needed:
the update is a local rule that can be computed from observable spin correlations.
This aligns with Phase 2 hardware goals — OIM machines implement EP natively.

**Implementation Status:** Done (Exp 466)

Spec: REQ-EORM-009

### REQ-EORM-010: SyntheticCoTPairGenerator Supplements Real Pairs to n_total=150

`SyntheticCoTPairGenerator(ebm, n_samples: int = 100).generate() -> list[tuple[str, bool]]` shall:
- Return exactly `n_samples` synthetic (cot_text, is_correct) pairs.
- Alternate between correct (is_correct=True) and incorrect (is_correct=False) pairs.

**Why synthetic augmentation (RETRO-034):** Exp 458 had only 57 real CoT pairs from one source
(Exp 443).  57 pairs is insufficient to train a calibrator that generalizes: the model overfits
to a narrow slice of the CoT style distribution.  Augmenting to 150 pairs (57 real + 93 synthetic)
improves generalization and gives the EP coupling update enough variation to learn meaningful
spin correlations.

**Implementation Status:** Done (Exp 466)

Spec: REQ-EORM-010

### SCENARIO-EORM-012: EPCouplingUpdate Produces Different J After Update

**Given** an `EPCouplingUpdate(learning_rate=0.01)` and a random coupling matrix J
**And** random free_spins and clamped_spins with different correlation structures
**When** `update_couplings(J, free_spins, clamped_spins)` is called
**Then** the returned J_new differs from the input J (update was applied)
**And** the difference has the correct sign: `J_new - J = η * (free_corr - clamped_corr)`

### SCENARIO-EORM-013: EBMCoTCalibratorV3 with 50 Steps Reaches Lower Energy Than with 10

**Given** an `EBMCoTCalibratorV3(eorm, n_langevin_steps=50)` and v2 with `n_langevin_steps=10`
**And** identical initial hidden state and random seed
**When** `calibrate_hidden(h)` is called on both
**Then** the mean energy after 50 steps is lower than after 10 steps (stronger relaxation)

### SCENARIO-EORM-014: EBMCoTCalibratorV3 calibrated_auc >= v2 Baseline on Same Synthetic Data

**Given** an `EBMCoTCalibratorV3` with EP coupling update enabled
**And** the same 40 synthetic labeled CoT pairs used in SCENARIO-EORM-011
**When** `calibrated_auc(examples)` is called
**Then** the AUC >= 0.5 (above random chance)
**And** AUC >= v2 baseline AUC minus 0.05 (no regression from v2)

### REQ-KAEM-005: KAEMEnergy Crossover Profiling Spans n_vars 50-1000

The `SpeedupProfile` class shall profile KAEMEnergy exact sampling versus
IsingEBM MCMC baseline across n_vars in [50, 100, 200, 500, 1000], producing
per-size latency records with speedup ratios, to empirically locate the crossover
point where KAEM becomes faster than MCMC.

**Why this range matters:** Exp 447 benchmarked n_vars ≤ 100 and found only 1.29x
speedup. RETRO-031 identified that the crossover is expected at n_vars > 200 where
MCMC mixing time grows super-linearly. This requirement mandates extending profiling
into that regime.

**Implementation Status:** Done (Exp 459)

Spec: REQ-KAEM-005, SCENARIO-KAEM-010

### REQ-KAEM-006: Crossover n_vars Defined as First n_vars Where kaem_time < mcmc_time

The `SpeedupProfile.crossover_n_vars()` method shall return the first n_vars in the
profiled list where speedup > 1.0 (i.e., KAEM is faster than MCMC). Returns None
when KAEM is never faster in the profiled range.

**Why exact definition matters:** Crossover is a discrete event in the profiled list,
not an interpolated float. Returning the first n_vars where KAEM wins gives a concrete,
actionable threshold for callers to switch samplers.

**Implementation Status:** Done (Exp 459)

Spec: REQ-KAEM-006, SCENARIO-KAEM-011

### SCENARIO-KAEM-010: SpeedupProfile Finds Crossover at n_vars=200 When KAEM Wins There

**Given** a SpeedupProfile with n_vars_list=[50, 100, 200, 500, 1000]
**And** kaem_times=[10, 12, 8, 6, 5] (ms)
**And** mcmc_times=[8, 10, 20, 50, 100] (ms)
**When** `crossover_n_vars()` is called
**Then** the result is 200 (first n_vars where mcmc_time > kaem_time)

### SCENARIO-KAEM-011: SpeedupProfile Returns None When KAEM Never Faster

**Given** a SpeedupProfile with n_vars_list=[50, 100, 200]
**And** kaem_times=[10, 15, 25] (ms, always slower)
**And** mcmc_times=[8, 10, 20] (ms, always faster)
**When** `crossover_n_vars()` is called
**Then** the result is None (KAEM never beats MCMC in the profiled range)

## Hardware Acceleration: AMD XDNA NPU via IRON Toolchain (Exp 460)

### REQ-HARDWARE-010: IRON Toolchain Installs via pip install mlir-aie

The `NPUEnvironment.install_iron()` method shall attempt to install the IRON
toolchain (arXiv 2504.03083) via `pip install mlir-aie` and return True on
success, False on failure.

**Why IRON unblocks NPU:** The traditional mlir-aie build requires cmake, ninja,
and openblas — none of which are available in the Carnot build environment.  IRON
(Integrated Runtime for Open NPUs) provides a pip-installable wheel that requires
ONLY Python and pip, eliminating 5 milestones of build-system blockage.

**Reference:** arXiv 2504.03083 (IRON: AMD XDNA NPU toolchain)

**Implementation Status:** Done (Exp 460)

Spec: REQ-HARDWARE-010, SCENARIO-HARDWARE-010

### REQ-HARDWARE-011: NPURunner Attempts JEPA ONNX Execution on AMD XDNA NPU

The `IRONRunner` class shall accept an ONNX model path, compile it to an AIE
kernel via `mlir_aie` (when IRON is installed and an NPU device is present),
and execute inference.  When the NPU is unavailable, it shall transparently
fall back to onnxruntime CPU inference so experiments run on all hardware.

**Why this matters:** The JEPA predictor ONNX model (results/jepa_predictor_291.onnx)
is the target workload for NPU acceleration.  Demonstrating NPU execution of this
model establishes the hardware path for production self-learning acceleration.

**Implementation Status:** Done (Exp 460)

Spec: REQ-HARDWARE-011, SCENARIO-HARDWARE-011

### REQ-HARDWARE-012: NPURunner Falls Back to CPU ONNX When NPU Unavailable

The `IRONRunner.benchmark()` method shall always return a valid timing result.
The `NPURunner.honest_verdict` field shall distinguish `'npu_executed'` from
`'cpu_fallback'` so callers can determine whether a real NPU was used.
CPU fallback is required because most developers lack NPU hardware.

**Implementation Status:** Done (Exp 460)

Spec: REQ-HARDWARE-012, SCENARIO-HARDWARE-012

### SCENARIO-HARDWARE-010: iron_available Returns False When mlir_aie Not Installed

**Given** mlir_aie is not installed in the test environment
**When** `NPUEnvironment.iron_available()` is called
**Then** the result is False

### SCENARIO-HARDWARE-011: IRONRunner Falls Back to CPU When NPU Device Absent

**Given** no /dev/accel0 device exists (most dev machines)
**When** `IRONRunner.benchmark(inputs, n_runs=3)` is called
**Then** a float timing result is returned (CPU onnxruntime fallback)
**And** `npu_device_present()` returns False

### SCENARIO-HARDWARE-012: honest_verdict Distinguishes npu_executed from cpu_fallback

**Given** an IRONRunner on a machine without NPU hardware
**When** benchmark is called and completes
**Then** the artifact field `honest_verdict` is `'cpu_baseline_only'`
**And** `npu_executed` is False in the artifact

---

## KV260 FPGA Ising Sampler v2 (Exp 471)

### REQ-HARDWARE-013: 128-Spin Sparsified Ising Sampler Verilog RTL

`hardware/kv260/ising_sampler_128_sparse.v` shall implement a 128-spin Ising
sampler with:
1. AXI-Lite slave interface for coupling matrix writes (0x4000 base) and spin reads (0x8010+).
2. Sparsified coupling matrix with 10-bit signed precision per entry.
3. 32-bit Galois LFSR for hardware-efficient pseudorandom thermal noise.
4. Metropolis update rule per spin (clock-driven).

**Why 128 spins:** Fits in KV260's ~256K LUTs with room for control logic.
**Why sparsified:** arXiv 2604.04606 shows 6x speedup vs SA, 4x larger scale at 90% sparsity.
**Why LFSR:** Two LUTs in Xilinx fabric, adequate for thermal noise uncorrelated between spins.
**Why 10-bit precision:** arXiv 2505.02103 validates 10-bit sufficient for EP-trained Ising machines.

**Implementation Status:** Done (Exp 471)

Spec: REQ-HARDWARE-013, SCENARIO-HARDWARE-013

### REQ-HARDWARE-014: FpgaBackend CPU Simulation Mode

`FpgaBackend` (in `carnot.hardware.fpga_backend`) shall:
1. Default to `simulation_mode=True` when `bitfile_path` is `None`.
2. In simulation mode, use `ParallelIsingSampler` as a drop-in CPU backend.
3. Return samples of shape `(n_samples, n_spins)` with dtype `bool` from `sample()`.

**Why simulation fallback:** Most contributors and all CI runners lack a KV260.
CPU simulation enables correctness testing without hardware.

**Implementation Status:** Done (Exp 471)

Spec: REQ-HARDWARE-014, SCENARIO-HARDWARE-014

### REQ-HARDWARE-015: EP Coupling Update Uses 10-Bit Precision and POSIX-Atomic Write

`FpgaBackend.update_couplings(new_J)` shall:
1. Encode the coupling matrix as 10-bit signed integers (range -511..511).
2. Write the encoded matrix to disk using a POSIX-atomic rename
   (write to a temp file in the same directory, then `os.rename()` to the target).

**Why POSIX atomic:** Prevents the KV260 PetaLinux driver from reading a torn
(partially-written) coupling matrix during the EP outer loop update.

**Implementation Status:** Done (Exp 471)

Spec: REQ-HARDWARE-015, SCENARIO-HARDWARE-015

### SCENARIO-HARDWARE-013: SparsifiedIsingConfig Generates Correct Sparse Matrix

**Given** `SparsifiedIsingConfig(n_spins=64, sparsity=0.9, seed=10)`
**When** `coupling_matrix()` is called
**Then** the result has shape `(64, 64)`, dtype `float32`, zero diagonal, and symmetric
**And** approximately 90% of off-diagonal entries are zero (within ±10% tolerance)
**And** `n_edges()` returns fewer edges than the dense (sparsity=0) version

### SCENARIO-HARDWARE-014: FpgaBackend Simulation Sample Returns Correct Shape

**Given** `FpgaBackend(simulation_mode=True)`
**When** `sample(SparsifiedIsingConfig(n_spins=8, sparsity=0.5), n_samples=10)` is called
**Then** the result has shape `(10, 8)` and dtype `bool`

**Given** `FpgaBackend(bitfile_path=None)`
**When** constructed
**Then** `simulation_mode` is `True` (bitfile_path=None forces simulation)

### SCENARIO-HARDWARE-015: update_couplings Uses Atomic Rename

**Given** `FpgaBackend(simulation_mode=True)` with a writable coupling cache path
**When** `update_couplings(new_J)` is called
**Then** exactly one `os.rename()` call is made (src in same dir as dst)
**And** the destination file contains the coupling matrix as a loadable `.npy` file

---

## Infrastructure Hardening (Exp 462)

### REQ-INFRA-033: ExperimentTemplate.assert_deliverable_written() Raises on Missing File

`ExperimentTemplate.assert_deliverable_written()` shall raise `FileNotFoundError`
(with a message citing RETRO-032/033/036) if the deliverable JSON path passed to
`ExperimentTemplate.__init__()` does not exist on disk when the method is called.
The method shall be called as the FINAL line of every experiment's `main()`.

**Why this matters (RETRO-032, RETRO-033, RETRO-036):** Three consecutive milestones
ended with missing result JSON files.  `build_result()` builds a dict in memory but
does NOT write it.  If the write was skipped, the file was silently absent and the
retrospective question could not be answered.  This assertion turns a silent omission
into an observable crash.

**Implementation Status:** Done (Exp 462)

Spec: REQ-INFRA-033, SCENARIO-INFRA-041

### REQ-INFRA-034: ExperimentTemplate.setup_gpu() Uses DualGPUAssigner for Dual-Model Experiments

`ExperimentTemplate.setup_gpu()` shall call `DualGPUAssigner(model_specs, n_gpus).assign()`
when `len(model_specs) >= 2` and `CARNOT_FORCE_LIVE=1` and `n_gpus >= 2`.
The assigner shall inject `device_map={'': 'cuda:N'}` into each spec so that
model N runs exclusively on GPU N with no cross-device layer offloading.

**Why this matters (RETRO-034, milestone .34):** GPU 1 (RTX 3090, 24 GB VRAM) was
idle the ENTIRE milestone .34.  DualGPURunner existed but was not wired in.
Every dual-model experiment ran sequentially on GPU 0 — 48 GB wasted.

**Implementation Status:** Done (Exp 462)

Spec: REQ-INFRA-034, SCENARIO-INFRA-042

### REQ-INFRA-035: DocOnlyClassifier Skips Full Test Suite for Doc-Only Diffs

`DocOnlyClassifier.is_doc_only_diff(changed_files)` shall return `True` if and
only if every file in `changed_files` has a `.md` extension or lives under
`docs/`, `ops/`, `_bmad/`, or `openspec/`.  CI pipelines may use this to skip
the full Rust+Python suite and run only ruff+mypy for doc-only commits.

**Why this matters:** The full 3900+ test suite takes 60-90s.  At 80+ doc-only
commits per milestone, this wastes 80-120 minutes.  Doc-only classification
recovers that time for actual research runs.

**Implementation Status:** Done (Exp 462)

Spec: REQ-INFRA-035, SCENARIO-INFRA-043

### SCENARIO-INFRA-041: assert_written() Raises When Deliverable Is Absent

**Given** an `ExperimentTemplate` with deliverable path `results/experiment_999.json`
**When** `assert_deliverable_written()` is called before the file is written
**Then** `FileNotFoundError` is raised with a message containing the file path and RETRO context

### SCENARIO-INFRA-042: DualGPUAssigner Assigns cuda:0 and cuda:1 in Live Mode

**Given** `DualGPUAssigner` with 2 model specs, n_gpus=2, CARNOT_FORCE_LIVE=1
**When** `assign()` is called
**Then** specs[0]['device_map'] == {'': 'cuda:0'} and specs[1]['device_map'] == {'': 'cuda:1'}
**And** `is_dual_gpu_eligible()` returns False when CARNOT_FORCE_LIVE is unset

### SCENARIO-INFRA-043: DocOnlyClassifier Correctly Classifies Diffs

**Given** changed_files = ['ops/status.md', '_bmad/prd.md']
**When** `is_doc_only_diff()` is called
**Then** returns True

**Given** changed_files = ['python/carnot/models/ising.py']
**When** `is_doc_only_diff()` is called
**Then** returns False

### REQ-INFRA-036: ConductorSessionHealthCheck Runs at Session Start

`ConductorSessionHealthCheck.run()` shall verify at conductor session startup:
1. `CARNOT_FORCE_LIVE` is propagated (calls `apply_env_autofix()` first).
2. Both GPUs are under 200 MB VRAM (idle — no zombie processes holding memory).
3. No GPU temperature exceeds 80°C (thermal gate).
4. No zombie GPU processes exist (VRAM > 500 MB, age > 300 s, util 0%).
Auto-remediates when `auto_remediate=True` (production default).

**Why this matters (RETRO-034, milestone .34):** Three zombie processes held 23,795 MB
on GPU 0 for 11.5 hours.  A session-start check would have caught and killed them
before Experiment 1 ran, recovering the GPU for the entire milestone.

**Implementation Status:** Done (Exp 463)

Spec: REQ-INFRA-036, SCENARIO-INFRA-044

### REQ-INFRA-037: Health Check Kills Zombie GPU Processes Before First Experiment

When `auto_remediate=True`, `ConductorSessionHealthCheck` shall call `SIGKILL`
on any GPU-attached process with VRAM > 500 MB and wall-clock age > 300 seconds,
then re-check GPU health after a 2-second CUDA context release window.

**Implementation Status:** Done (Exp 463)

Spec: REQ-INFRA-037, SCENARIO-INFRA-045

### REQ-INFRA-038: Thermal Gate Pauses Conductor When GPU >= 80°C

When any GPU reports temperature >= 80°C, `SessionHealthResult.honest_verdict`
shall be `'session_thermal_blocked'` and `conductor_session_health.py` shall
`sys.exit(1)` to prevent the conductor from spawning experiments into an
already-stressed thermal state.

**Why 80°C:** RTX 3090 self-throttles at 83°C; sustained operation above 80°C
shortens hardware life.  RETRO-034 recorded GPU 0 at 82°C during runaway experiments.

**Implementation Status:** Done (Exp 463)

Spec: REQ-INFRA-038, SCENARIO-INFRA-046

### SCENARIO-INFRA-044: ConductorSessionHealthCheck Returns SessionHealthResult in CI

**Given** `ConductorSessionHealthCheck(auto_remediate=False)` (CI mode)
**When** `run()` is called
**Then** returns a `SessionHealthResult` with `honest_verdict` in
  {'session_healthy', 'session_remediated', 'session_thermal_blocked'}
**And** no processes are killed (auto_remediate=False)

### SCENARIO-INFRA-045: Zombie Process Detected and Killed in Live Mode

**Given** a GPU process holding > 500 MB VRAM with wall-clock age > 300 seconds
**When** `ConductorSessionHealthCheck(auto_remediate=True).run()` is called
**Then** the process receives SIGKILL
**And** `SessionHealthResult.zombies_killed >= 1`
**And** `SessionHealthResult.honest_verdict == 'session_remediated'`

### SCENARIO-INFRA-046: Thermal Gate Blocks Conductor

**Given** a GPU reporting temperature >= 80°C
**When** `ConductorSessionHealthCheck.run()` is called
**Then** `SessionHealthResult.thermal_ok == False`
**And** `SessionHealthResult.honest_verdict == 'session_thermal_blocked'`
**And** `conductor_session_health.py` exits with code 1

### REQ-BENCH-025: Live 100q Benchmark Uses GPUVRAMGate Before Model Load

The Exp 476 live precision benchmark shall use `GPUVRAMGate(min_free_gb=8.0, wait_seconds=60)`
as a context manager before any model load.  The gate shall kill zombie processes that hold
VRAM at 0% utilisation, then wait up to 60 seconds for 8 GB free VRAM on each GPU.  If VRAM
cannot be freed within 60 seconds, the experiment shall write an artifact with
`status='gpu_vram_insufficient'` and call `assert_deliverable_written()` before returning.

**Why GPUVRAMGate (RETRO-037, RETRO-042):** Exp 464 was deferred with `deferred_to_gpu`
because mid-session zombie processes held 23.8 GB on GPU 0 at 0% utilisation.  The session-
start health check (Exp 463) cannot prevent mid-session accumulation; only a per-experiment
gate can.

**Implementation Status:** Done (Exp 476)

Spec: REQ-BENCH-025, SCENARIO-BENCH-044

### REQ-BENCH-026: DualGPURunner Assigns Gemma4-E4B-it to cuda:0 and Qwen3.5-0.8B to cuda:1

`DualGPUAssigner` shall assign `Gemma4-E4B-it` to `cuda:0` and `Qwen3.5-0.8B` to `cuda:1`
for concurrent inference.  Both models shall be loaded simultaneously before the benchmark
begins.  Each model gets a dedicated GPU, preventing the RETRO-025 cross-GPU layer sharing
problem that occurs with `device_map='auto'`.

**Implementation Status:** Done (Exp 476)

Spec: REQ-BENCH-026, SCENARIO-BENCH-045

### REQ-BENCH-027: Benchmark Writes 100 CoT Pairs to results/exp476_cot_pairs.json for JEPA Retrain

For every benchmark question, the Exp 476 harness shall collect a CoT pair
`{question, cot_text, correct}` via `CoTPairCollector` and flush atomically to
`results/exp476_cot_pairs.json`.  The pair count shall be reported in the primary artifact
under `cot_pairs_written`.  These pairs feed the JEPA retrain pipeline (Exp 477).

**Implementation Status:** Done (Exp 476)

Spec: REQ-BENCH-027, SCENARIO-BENCH-046

### SCENARIO-BENCH-044: GPUVRAMGate Fires and Kills Zombies Before Model Load

**Given** `GPUVRAMGate(min_free_gb=8.0, wait_seconds=60)` is entered
**When** free VRAM on any GPU is below 8 GB
**Then** the gate kills zombie processes holding VRAM at 0% utilisation
**And** waits up to 60 seconds for VRAM to free
**And** proceeds if VRAM becomes available, or raises `GPUVRAMInsufficientError` otherwise

### SCENARIO-BENCH-045: DualGPUAssigner Pins Each Model to Its Own GPU

**Given** two models [Gemma4-E4B-it, Qwen3.5-0.8B] and `n_gpus=2`
**When** `DualGPUAssigner.assign()` is called
**Then** Gemma4-E4B-it receives `device_map={'': 'cuda:0'}`
**And** Qwen3.5-0.8B receives `device_map={'': 'cuda:1'}`

### SCENARIO-BENCH-046: CoTPairCollector Flushes Pairs Atomically

**Given** `CoTPairCollector(output_path)` with N pairs added
**When** `flush()` is called
**Then** a valid JSON file is written atomically at `output_path`
**And** the file contains exactly N pairs with keys: model, question, cot_text, correct
**And** `flush()` returns N (the number of pairs written)

Spec: REQ-BENCH-025, REQ-BENCH-026, REQ-BENCH-027,
      SCENARIO-BENCH-044, SCENARIO-BENCH-045, SCENARIO-BENCH-046

### REQ-BENCH-028: 200q Benchmark v2 Uses GPUVRAMGate Before Each Model Load

The 200-question live benchmark v2 (Exp 478) shall invoke `GPUVRAMGate(min_free_gb=8.0)`
as a context manager before any model is loaded, so that zombie VRAM held by prior
experiments cannot silently prevent model initialisation.  If the gate raises
`GPUVRAMInsufficientError`, the experiment shall write an honest artifact with
`status='gpu_vram_insufficient'` and exit cleanly.

### REQ-BENCH-029: 200q Benchmark v2 Uses DualGPURunner with Gemma4→cuda:0 and Qwen→cuda:1

The 200-question live benchmark v2 shall use `DualGPUAssigner` to pin
Gemma4-E4B-it to `cuda:0` and Qwen3.5-0.8B to `cuda:1` so both models run on
dedicated GPUs simultaneously rather than competing for GPU 0.

### REQ-BENCH-030: 200q v2 Result Includes Wilson 95% CI and is_statistically_positive Flag

`Live200qV2Result` shall expose:

- `ci_95_wilson: tuple[float, float]` — Wilson score CI on post_acc.
  At n=200 the full width shall be < 0.07 for any p in [0,1].
- `is_statistically_positive: bool` — True iff the lower bound of the Wald CI
  for the *signed improvement* exceeds zero.  A 2pp improvement at n=200 with
  typical pre_acc ≈ 0.70 shall evaluate to False (not statistically significant).

### SCENARIO-BENCH-047: Wilson CI Width at n=200

**Given** `Live200qV2Result` with `post_acc=0.05` and `n=200`
**When** `ci_95_wilson` is computed
**Then** the full interval width (`upper - lower`) is less than 0.07

### SCENARIO-BENCH-048: DualGPUAssigner Assigns Gemma4→cuda:0 and Qwen→cuda:1 in v2

**Given** model specs `[Gemma4-E4B-it, Qwen3.5-0.8B]` and `n_gpus=2`
**When** `DualGPUAssigner.assign()` is called
**Then** Gemma4-E4B-it has `device_map={'': 'cuda:0'}`
**And** Qwen3.5-0.8B has `device_map={'': 'cuda:1'}`

### SCENARIO-BENCH-049: is_statistically_positive=False for Small Improvement at n=200

**Given** `Live200qV2Result` with `pre_acc=0.70`, `post_acc=0.72`, `n=200`
(signed_improvement = 0.02)
**When** `is_statistically_positive` is evaluated
**Then** the result is `False` because the Wald CI for the improvement includes zero

Spec: REQ-BENCH-028, REQ-BENCH-029, REQ-BENCH-030,
      SCENARIO-BENCH-047, SCENARIO-BENCH-048, SCENARIO-BENCH-049

### REQ-BENCH-031: Adversarial Benchmark Uses GPUVRAMGate Before Model Load

The GSM-Symbolic adversarial live benchmark (Exp 479) shall invoke
`GPUVRAMGate(min_free_gb=8.0)` as a context manager before any model is loaded.
If the gate raises `GPUVRAMInsufficientError`, the experiment shall write an honest
artifact with `status='gpu_vram_insufficient'` and exit cleanly via
`assert_deliverable_written()`.

**Implementation Status:** Done (Exp 479)

### REQ-BENCH-032: Adversarial Benchmark Runs Three Conditions

The GSM-Symbolic adversarial live benchmark shall evaluate each model across three
conditions:
- Condition A: standard GSM8K baseline (LLM only, no pipeline)
- Condition B: adversarial GSM-Symbolic baseline (LLM only, no pipeline)
- Condition C: adversarial GSM-Symbolic + Carnot verify-repair (`IntegratedExtractor`)

All three conditions use 50 questions.  Results are aggregated into `AdversarialV2Result`
(alias for `AdversarialBenchmarkResult`) per model, with `data_source` recording
`'huggingface'` or `'hardcoded_fallback'`.

**Implementation Status:** Done (Exp 479)

### REQ-BENCH-033: thesis_confirmed=True When Adversarial Improvement Exceeds Standard

`AdversarialV2Result.thesis_confirmed` shall be `True` when
`carnot_adversarial_improvement > carnot_standard_improvement`.  The top-level artifact
field `thesis_confirmed` shall be `True` when any model's result confirms the thesis.
The `honest_verdict` field shall be:
- `'thesis_confirmed'` when `thesis_confirmed=True`
- `'thesis_partial'` when any `carnot_adversarial_improvement > 0` but thesis not fully confirmed
- `'thesis_not_confirmed'` otherwise

**Implementation Status:** Done (Exp 479)

### SCENARIO-BENCH-050: GPUVRAMGate Blocks Exp 479 When VRAM Insufficient

**Given** `GPUVRAMGate(min_free_gb=8.0)` is entered in Exp 479
**When** free VRAM on any GPU cannot reach 8 GB after zombie kill + wait
**Then** the experiment writes `status='gpu_vram_insufficient'` artifact
**And** `assert_deliverable_written()` confirms the file exists

### SCENARIO-BENCH-051: Three Conditions Are Evaluated Per Model in Exp 479

**Given** Exp 479 runs on live GPU hardware
**When** both Gemma4-E4B-it and Qwen3.5-0.8B are loaded
**Then** each model is evaluated on condition A (standard), B (adversarial), and C (adversarial+Carnot)
**And** each model's `AdversarialV2Result` records `adversarial_drop`, `carnot_adversarial_improvement`, `thesis_confirmed`

### SCENARIO-BENCH-052: thesis_confirmed=True When Adversarial Improvement (0.10) > Standard (0.05)

**Given** `AdversarialV2Result` with `adversarial_carnot_acc=0.65`, `adversarial_baseline_acc=0.55`, `carnot_standard_improvement=0.05`
**When** `thesis_confirmed` is evaluated
**Then** `carnot_adversarial_improvement = 0.10 > 0.05 = carnot_standard_improvement`
**And** `thesis_confirmed is True`

Spec: REQ-BENCH-031, REQ-BENCH-032, REQ-BENCH-033,
      SCENARIO-BENCH-050, SCENARIO-BENCH-051, SCENARIO-BENCH-052

---

### REQ-INFRA-045: DualGPUHarness.apply() Assigns cuda:0/cuda:1 to Dual-Model Specs

`DualGPUHarness.apply(model_specs)` shall inject `gpu=0, device_map={'': 'cuda:0'}` into
the first spec and `gpu=1, device_map={'': 'cuda:1'}` into the second spec when
`is_eligible` is True (`n_gpus >= 2` and `live_mode=True`).

When `is_eligible` is False (CI mode or single GPU), `apply()` returns `model_specs`
unchanged — no device maps are injected.

**Motivation (RETRO-041):** GPU 1 (RTX 3090) was idle for three consecutive milestones.
DualGPURunner existed in experiment_template.py but no benchmark harness called it.
DualGPUHarness closes the adoption gap without requiring per-script rewrites.

**Implementation Status:** Done (Exp 480)

### REQ-INFRA-046: HarnessAudit.scan() Returns Findings for Dual-Model Scripts Missing cuda:1

`HarnessAudit(scripts_dir).scan()` shall return a list of `AuditFinding` records, one per
Python script in `scripts_dir` that:
- Contains two or more model-load indicators (`hf_id` occurrences or `_load_*` function calls), AND
- Does NOT contain the literal string `'cuda:1'`.

Such scripts have `needs_fix=True`.  Scripts that already contain `'cuda:1'` have
`has_cuda1_assignment=True` and `needs_fix=False`.

**Implementation Status:** Done (Exp 480)

### SCENARIO-INFRA-053: DualGPUHarness Assigns cuda:0 and cuda:1 in Live Mode

**Given** `DualGPUHarness(n_gpus=2, live_mode=True)` is constructed
**When** `apply([spec_A, spec_B])` is called
**Then** `spec_A['gpu'] == 0` and `spec_A['device_map'] == {'': 'cuda:0'}`
**And** `spec_B['gpu'] == 1` and `spec_B['device_map'] == {'': 'cuda:1'}`

**Given** `DualGPUHarness(n_gpus=2, live_mode=False)` is constructed (CI mode)
**When** `apply([spec_A, spec_B])` is called
**Then** specs are returned unchanged (no gpu or device_map injected)

### SCENARIO-INFRA-054: HarnessAudit Flags Dual-Model Scripts Without cuda:1

**Given** a scripts directory containing `exp_dual.py` with two `hf_id` keys but no `'cuda:1'`
**When** `HarnessAudit(scripts_dir).scan()` is called
**Then** one `AuditFinding` is returned with `has_dual_model_load=True`, `has_cuda1_assignment=False`, `needs_fix=True`

**Given** a scripts directory containing `exp_good.py` with two `hf_id` keys AND `'cuda:1'`
**When** `HarnessAudit(scripts_dir).scan()` is called
**Then** one `AuditFinding` is returned with `has_cuda1_assignment=True`, `needs_fix=False`

Spec: REQ-INFRA-045, REQ-INFRA-046,
      SCENARIO-INFRA-053, SCENARIO-INFRA-054

### REQ-INFRA-047: BatchingEnforcementAudit Scans Scripts for Sequential Question Loops

`BatchingEnforcementAudit(scripts_dir).scan()` shall scan all `*.py` files in `scripts_dir`
and return a list of `BatchingViolation` records for every file that contains a sequential
question loop (``for <var> in <questions_collection>:``) without using `BatchedInferenceRunner`.

**Motivation (RETRO-041 batching sub-item):** Sequential inference at batch_size=1 is 3-5x
slower than batched inference because each question pays full CUDA kernel-launch overhead
and uses fewer than 10 of 80-144 GPU streaming multiprocessors.  BatchedInferenceRunner
has been available since Exp 437 but was not consistently adopted.  The .35 retrospective
estimated 5% wall time savings (~250 min per 5000-min milestone) from enforcement.

Each `BatchingViolation` has:
- `script_path` — path to the offending script
- `line_no` — 1-based line number of the sequential loop
- `pattern` — matched source line (stripped)
- `severity` — `'high'` when no `BatchedInferenceRunner` is present; `'medium'` when it is
- `is_high_severity` — True when severity == 'high'

**Implementation Status:** Done (Exp 481)

### REQ-INFRA-048: Standard Batch Sizes for Known Task Types

`BatchingEnforcementAudit.recommended_batch_size(task_type)` shall return:
- `'gsm8k'`    → 8  (longer arithmetic prompts; predictable VRAM on 24 GB RTX cards)
- `'humaneval'`→ 4  (code generation produces variable-length output; batch=4 avoids OOM)
- any other   → 8  (default; recovers most throughput gap vs. batch_size=1)

**Implementation Status:** Done (Exp 481)

### SCENARIO-INFRA-055: BatchingEnforcementAudit Detects Sequential Loop Violation

**Given** a scripts directory containing `exp_seq.py` with `for q in questions:` and no `BatchedInferenceRunner`
**When** `BatchingEnforcementAudit(scripts_dir).scan()` is called
**Then** at least one `BatchingViolation` is returned with `is_high_severity=True`

### SCENARIO-INFRA-056: recommended_batch_size Returns Correct Values

**Given** `BatchingEnforcementAudit(scripts_dir)`
**When** `recommended_batch_size('gsm8k')` is called
**Then** 8 is returned

**When** `recommended_batch_size('humaneval')` is called
**Then** 4 is returned

**When** `recommended_batch_size('unknown_task')` is called
**Then** 8 is returned (default)

Spec: REQ-INFRA-047, REQ-INFRA-048,
      SCENARIO-INFRA-055, SCENARIO-INFRA-056

### REQ-VERIFY-096: NUPProbe Computes Per-Step Continuation Entropy

`NUPProbe.score(cot_text, logprobs)` shall compute the Shannon entropy of the
continuation distribution as a proxy for constraint violation likelihood.

When `logprobs` is supplied, Shannon entropy of the token distribution is returned.
When `logprobs` is absent or empty, character-level Shannon entropy of `cot_text` is
returned as a structural proxy (the character-entropy fallback).

`ContinuationEntropy.from_logprobs(logprobs, threshold)` shall compute the Shannon entropy
and set `is_high_entropy = entropy > threshold` (default threshold 1.5 nats).

**Motivation (arXiv 2603.19562 — Neural Uncertainty Principle, 2026):**
    Hallucination is an under-constrained continuation problem.  High continuation entropy
    = multiple paths nearly equally plausible = constraint violation likely.  Low entropy =
    peaked distribution = continuation forced by prior context = likely correct.
    This is mathematically compatible with Carnot's formulation: high energy = high entropy.

**Implementation Status:** Done (Exp 484)

### REQ-VERIFY-097: NUPProbe AUC > 0.700 Qualifies as Tier 0c

`NUPProbeResult.is_viable_tier_0c` shall be `True` when `auc > 0.700` on held-out
labeled CoT pairs, qualifying NUPProbe for insertion as Tier 0c in `ThreeTierPipeline`.

Tier 0c is positioned before SpilledEnergyDetector (Tier 0b) and CarnotThinkProbe (Tier 0a)
because NUPProbe requires zero LLM calls and zero Ising sampling — it is pure arithmetic on
log-probabilities.  Earlier cascade position = higher skip rate = lower total cost at scale.

When the current live data lacks token logprobs (n_with_logprobs=0), the character-entropy
fallback is used and the AUC may be below 0.700 — this is reported as
`honest_verdict='improvement_below_threshold'`.  The probe's Tier 0c promotion is deferred
until token logprobs are available.

**Implementation Status:** Done (Exp 484); AUC=0.600 on 11 held-out pairs with char-entropy
fallback (logprobs absent from current live data).

### SCENARIO-VERIFY-129: NUPProbe High Entropy Predicts Violation

**Given** a CoT step with a uniform token distribution over 20 tokens (logprobs=[0.0]*20,
entropy=ln(20)≈3.0 nats)
**When** `NUPProbe(entropy_threshold=1.5).predict_violation(text, logprobs)` is called
**Then** `True` is returned (entropy 3.0 > threshold 1.5)

**Given** a CoT step with a peaked distribution (logprobs=[0.0, -100.0], entropy≈0)
**When** `NUPProbe(entropy_threshold=1.5).predict_violation(text, logprobs)` is called
**Then** `False` is returned (entropy ≈ 0 < threshold 1.5)

### SCENARIO-VERIFY-130: NUPProbe.evaluate_auc Returns Float in [0, 1]

**Given** a list of labeled CoT pairs (mix of 'correct' and 'incorrect' labels)
**When** `NUPProbe.evaluate_auc(pairs)` is called
**Then** a float in [0.0, 1.0] is returned

**Given** fewer than 2 pairs
**When** `NUPProbe.evaluate_auc(pairs)` is called
**Then** 0.5 is returned (chance level, AUC undefined)

Spec: REQ-VERIFY-096, REQ-VERIFY-097,
      SCENARIO-VERIFY-129, SCENARIO-VERIFY-130

### REQ-INFRA-049: GPUVRAMGateV2 Kills Zombie Processes Before Checking Free VRAM

`GPUVRAMGateV2(min_free_gb, wait_seconds, zombie_drain_sleep_seconds=15, kill_first=True)`
shall implement a kill-first VRAM guard that eliminates the RETRO-044 race condition.

When `kill_first=True` (default), `ensure_vram_available(gpu_index)` shall:
1. Call `kill_zombies(gpu_index)` unconditionally — remove zombie processes first
2. Sleep `zombie_drain_sleep_seconds` seconds — allow the GPU driver to flush VRAM
   held by the killed processes (RTX 3090 driver drains within 10-15 s; 15 s default)
3. Call `check_vram(gpu_index)` — read VRAM after the drain window has elapsed
4. If still below threshold: enter the `wait_seconds` polling loop

When `kill_first=False`, V1 (GPUVRAMGate) check-first behavior is reproduced for
backward compatibility only.

**Motivation (RETRO-044):** GPUVRAMGate (V1, Exp 474) checked VRAM first, then killed
zombies.  The GPU driver holds the killed context's VRAM for 5-15 s during drain.  The
subsequent poll fired during this window, saw VRAM still held, and deferred.  Exps 476,
478, and 479 all hit `gpu_vram_insufficient` for this single ordering mistake.

**Implementation Status:** Done (Exp 487)

### REQ-INFRA-050: GPUVRAMGateV2 Replaces GPUVRAMGate in ExperimentTemplate.setup_gpu()

`ExperimentTemplate.setup_gpu()` shall use `GPUVRAMGateV2(kill_first=True)` instead of
`GPUVRAMGate` when `requires_gpu=True`.

**Implementation Status:** Done (Exp 487)

### REQ-INFRA-051: ExperimentTemplate.setup_gpu() Imports GPUVRAMGateV2 with kill_first=True Default

`ExperimentTemplate.setup_gpu()` shall import `GPUVRAMGateV2` from
`carnot.pipeline.gpu_vram_gate_v2` and use `kill_first=True` by default.

`GPUVRAMGateV2` shall also be exported from `carnot.pipeline.__init__` alongside
`GPUVRAMGate`, `GPUVRAMInsufficientError`, and `VRAMStatus`.

**Implementation Status:** Done (Exp 487)

### SCENARIO-INFRA-057: GPUVRAMGateV2 kill_first=True Calls kill_zombies Before check_vram

**Given** `GPUVRAMGateV2(kill_first=True)` with a mocked GPU
**When** `ensure_vram_available(0)` is called
**Then** `kill_zombies()` is called before the first `check_vram()` call
**And** `time.sleep(zombie_drain_sleep_seconds)` is called after `kill_zombies()`

### SCENARIO-INFRA-058: GPUVRAMGateV2 kill_first=False Preserves V1 Check-First Ordering

**Given** `GPUVRAMGateV2(kill_first=False)` with a healthy GPU
**When** `ensure_vram_available(0)` is called
**Then** `check_vram()` is called before `kill_zombies()`
**And** `time.sleep` is NOT called (no drain logic in V1 path)

### SCENARIO-INFRA-059: GPUVRAMGateV2 Is a No-Op on CPU-Only Machines

**Given** `GPUVRAMGateV2()` on a machine with zero NVML-detected GPUs (`_n_gpus() == 0`)
**When** the context manager is entered
**Then** no error is raised and `__enter__` returns self
**And** `ensure_vram_available` returns True when `check_vram` returns total_mb==0

Spec: REQ-INFRA-049, REQ-INFRA-050, REQ-INFRA-051,
      SCENARIO-INFRA-057, SCENARIO-INFRA-058, SCENARIO-INFRA-059

---

## Exp 488: Live 100q Precision v5 — GPUVRAMGateV2 + Explicit cuda:0/cuda:1

### REQ-BENCH-034: 100q v5 Uses GPUVRAMGateV2 with kill_first=True Before Model Load

Experiment 488 shall run `GPUVRAMGateV2(min_free_gb=8.0, kill_first=True)` as the first
gate before any model load, ensuring zombie processes are killed and VRAM is confirmed free
before allocation.

**Implementation Status:** Done (Exp 488)

### REQ-BENCH-035: 100q v5 Assigns Models to Explicit CUDA Devices via DualGPUHarness

Experiment 488 shall assign `google/gemma-4-E4B-it` to `cuda:0` and `Qwen/Qwen3.5-0.8B`
to `cuda:1` explicitly via `DualGPUHarness.apply()`, never using `device_map='auto'`.
Explicit device assignment prevents VRAM contention when two large models are loaded
simultaneously.

**Implementation Status:** Done (Exp 488)

### REQ-BENCH-036: 100q v5 Writes CoT Pairs for NUP Probe v2 Retrain

Experiment 488 shall write 100 CoT pairs per model (200 total) to
`results/exp488_cot_pairs.json` for use in NUP Probe v2 (Exp 496) retraining.
Each pair contains (model, question, cot_text, correct).

**Implementation Status:** Done (Exp 488)

### SCENARIO-BENCH-053: GPUVRAMGateV2 Fires Before Model Load in Exp 488

**Given** Experiment 488 running on a machine with CUDA GPUs
**When** the experiment starts
**Then** `GPUVRAMGateV2(kill_first=True).__enter__()` is called before any model is loaded
**And** if VRAM is insufficient after gate, a deferred artifact with
`honest_verdict='deferred_retro_033'` is written and the experiment exits without loading models

### SCENARIO-BENCH-054: Exp 488 Assigns Gemma4 to cuda:0 and Qwen to cuda:1

**Given** `DualGPUHarness(n_gpus=2, live_mode=True)` and two model specs
**When** `apply()` is called with `[{'name': 'google/gemma-4-E4B-it'}, {'name': 'Qwen/Qwen3.5-0.8B'}]`
**Then** the first spec receives `gpu=0` and `device_map={'': 'cuda:0'}`
**And** the second spec receives `gpu=1` and `device_map={'': 'cuda:1'}`

### SCENARIO-BENCH-055: Exp 488 Writes Valid Deliverable with All Schema Fields

**Given** a successful Exp 488 run
**When** the experiment completes
**Then** `results/experiment_488_live_100q_precision_v5.json` exists and contains:
- `schema='carnot.live_precision.v5'`
- `gemma4_result` with `signed_improvement`, `ci_95_wilson`, `is_positive`
- `qwen_result` with the same fields
- `cot_pairs_written >= 0`
- `gpu_vram_gate_v2_fired=True`
- `dual_gpu_explicit_assignment=True`
- `retro_033_closed` (bool)
- `honest_verdict` in `{'retro_033_closed_positive', 'retro_033_closed_negative', 'deferred_retro_033'}`

---

## Exp 489: Live 200q VeriCoT+VPRM v3 — GPUVRAMGateV2 + DualGPUHarness (RETRO-038)

### REQ-BENCH-037: 200q v3 Uses GPUVRAMGateV2(kill_first=True) Before Model Load

Experiment 489 shall run `GPUVRAMGateV2(min_free_gb=8.0, kill_first=True)` as the first
gate before any model load.  Kill-first order (Exp 487) eliminates the RETRO-044 race
condition where zombie VRAM was checked before the driver drained killed contexts.

**Implementation Status:** Done (Exp 489)

### REQ-BENCH-038: 200q v3 Uses DualGPUHarness with Explicit cuda:0/cuda:1

Experiment 489 shall use `DualGPUHarness(n_gpus=2, live_mode=True).apply(model_specs)`
to assign `google/gemma-4-E4B-it` to `cuda:0` and `Qwen/Qwen3.5-0.8B` to `cuda:1`
explicitly, preventing VRAM contention from dual-model same-GPU loading.

**Implementation Status:** Done (Exp 489)

### REQ-BENCH-039: 200q v3 Result Includes Wilson 95% CI and is_statistically_positive Flag

`Live200qV3Result` shall expose:
- `ci_95_wilson`: Wilson score 95% CI on `post_acc`; width < 0.07 at n=200 for any p.
- `is_statistically_positive`: True iff the lower bound of the Wald CI for the
  improvement (post_acc − pre_acc) exceeds 0.0 strictly.
- A Wilson 95% CI at n=200 with p=0.05 shall have interval width < 0.07.
- `is_statistically_positive` shall be False when `signed_improvement=0.02` and `n=200`.

**Implementation Status:** Done (Exp 489)

### SCENARIO-BENCH-056: Wilson CI Width < 0.07 at n=200

**Given** `Live200qV3Result` with `n=200` and `post_acc=0.05`
**When** `ci_95_wilson` is computed
**Then** the interval width `(upper - lower)` is less than `0.07`

### SCENARIO-BENCH-057: Signed Improvement Computed Correctly

**Given** `Live200qV3Result` with `pre_acc=0.70`, `post_acc=0.80`
**When** `signed_improvement` is accessed
**Then** it equals `0.10` (unclamped; negative values allowed for honest reporting)

### SCENARIO-BENCH-058: is_statistically_positive=False for Small Improvements

**Given** `Live200qV3Result` with `pre_acc=0.70`, `post_acc=0.72`, `n=200`
**When** `is_statistically_positive` is evaluated
**Then** it is `False` because the Wald CI lower bound is approximately -0.069 < 0.0

---

## Exp 490: GSM-Symbolic Adversarial v3 — GPUVRAMGateV2 + thesis confirmation (RETRO-039)

### REQ-BENCH-040: Adversarial v3 Uses GPUVRAMGateV2(kill_first=True) Before Model Load

Experiment 490 shall run `GPUVRAMGateV2(min_free_gb=8.0, kill_first=True)` as the first
gate before any model load.  Kill-first order eliminates the RETRO-044 race condition where
zombie VRAM was checked before the driver drained killed contexts.

**Implementation Status:** Done (Exp 490)

### REQ-BENCH-041: Adversarial v3 Runs Three-Condition Test

Experiment 490 shall run all three benchmark conditions per model:
- Condition A: 50 standard GSM8K questions, LLM only, no Carnot pipeline
- Condition B: 50 adversarial GSM-Symbolic questions, LLM only, no pipeline
- Condition C: 50 adversarial GSM-Symbolic questions with Carnot verify-repair (IntegratedExtractor)

**Implementation Status:** Done (Exp 490)

### REQ-BENCH-042: thesis_confirmed=True When carnot_adversarial_improvement > carnot_standard_improvement

`AdversarialV3Result.thesis_confirmed` shall be True when `carnot_adversarial_improvement`
strictly exceeds `carnot_standard_improvement`.  This is the headline result: Carnot's
benefit is larger on adversarial variants than on standard ones, confirming that EBM
constraint verification is immune to irrelevant-sentence injection.

**Implementation Status:** Done (Exp 490)

### SCENARIO-BENCH-059: GPUVRAMGateV2 Fires Before Model Load in Exp 490

**Given** Experiment 490 running on a machine with CUDA GPUs
**When** the experiment starts
**Then** `GPUVRAMGateV2(kill_first=True).__enter__()` is called before any model is loaded
**And** if VRAM is insufficient after gate, a deferred artifact with
`honest_verdict='gpu_vram_insufficient'` is written and the experiment exits without loading models

### SCENARIO-BENCH-060: thesis_confirmed=True When Adversarial Improvement Exceeds Standard

**Given** `AdversarialV3Result` with `adversarial_carnot_acc=0.65`, `adversarial_baseline_acc=0.55`,
`carnot_standard_improvement=0.05`
**When** `thesis_confirmed` is evaluated
**Then** it is `True` because `carnot_adversarial_improvement=0.10 > carnot_standard_improvement=0.05`

### SCENARIO-BENCH-061: adversarial_drop Computed as standard_acc - adversarial_baseline_acc

**Given** `AdversarialV3Result` with `standard_acc=0.70`, `adversarial_baseline_acc=0.55`
**When** `adversarial_drop` is accessed
**Then** it equals `0.15` (positive means LLM regressed on adversarial, replicating Apple finding)

---

## Exp 493: Batching Pre-commit Hook — RETRO-045

### REQ-INFRA-052: Batching-Check Pre-commit Hook

The repository shall install a `batching-check` pre-commit hook that runs
`BatchingEnforcementAudit` on staged `scripts/*.py` files and exits 1 if any new
high-severity violations (sequential loops without BatchedInferenceRunner) are found in
the staged diff.

**Why:** Writing a batching standard (Exp 481, 77 violations documented) without enforcement
is ineffective.  Every new experiment script added by the conductor can introduce new violations
silently.  A pre-commit hook makes enforcement automatic: the conductor cannot commit a new
script with sequential loops without first fixing them.

**Implementation Status:** Done (Exp 493)

### REQ-INFRA-053: Batching-Check Hook is Idempotent

The `batching-check` hook shall be idempotent: existing violations in previously committed
files do not block future commits.  Only NEW violations introduced in the currently staged
diff are flagged.  A staged file that is NOT new (already committed with violations) is
treated as pre-existing and is not re-flagged.

**Why:** A non-idempotent hook would block every commit until all 77 existing violations
are fixed — which is infeasible in a single pass.  The idempotency rule allows the project
to accumulate fixes incrementally without blocking unrelated work.

**Implementation Status:** Done (Exp 493)

### SCENARIO-INFRA-060: New Script with Sequential Loop Blocked by Hook

**Given** a new Python script staged for commit containing `for q in questions:`
  without any `BatchedInferenceRunner` reference
**When** the `batching-check` pre-commit hook runs
**Then** it exits 1 and prints a violation report

### SCENARIO-INFRA-061: Compliant Script Not Blocked by Hook

**Given** a new Python script staged for commit that uses `BatchedInferenceRunner`
  or contains no sequential question loops
**When** the `batching-check` pre-commit hook runs
**Then** it exits 0 and does not block the commit

---

## Exp 495: DualGPU Harness Patch — RETRO-041 execution (milestone .36 GPU 1 at 11%)

### REQ-INFRA-057: HarnessPatcher.patch_script() Rewrites device_map='auto' to DualGPUHarness.apply()

`HarnessPatcher.patch_script(path)` shall read the Python source at `path` and apply
the following transformation for dual-model scripts flagged by `HarnessAudit`:

1. If the source already contains `cuda:1`, return `HarnessPatchResult(was_patched=False)` — no-op.
2. If the source contains `device_map='auto'` or `device_map="auto"`, replace the first
   occurrence with `device_map={'': 'cuda:0'}` and subsequent occurrences with
   `device_map={'': 'cuda:1'}`, then write the patched source back to disk.
3. If no `device_map='auto'` pattern is found, inject a module-level
   `DualGPUHarness.from_env().apply(MODEL_SPECS)` call block at the end of the file
   so that the second model is pinned to `cuda:1` at runtime.
4. Return `HarnessPatchResult(was_patched=True, error=None)` on success, or
   `HarnessPatchResult(was_patched=False, error=<message>)` if no patchable pattern
   was found.

**Why:** Documentation of 53 violations (Exp 480) without execution did not change
GPU 1 utilization — milestone .36 still measured 11%.  Auto-patch propagates the fix
to all identified scripts immediately without requiring manual rewrites.

**Implementation Status:** Done (Exp 495)

### REQ-INFRA-058: HarnessPatcher.patch_all() Applies Patch to All needs_fix Findings

`HarnessPatcher.patch_all(findings)` shall iterate over all `AuditFinding` entries
with `needs_fix=True` and call `patch_script(finding.script_path)` for each, collecting
and returning the list of `HarnessPatchResult` objects.  The method shall report
`n_patched` (count where `was_patched=True`) and proceed even when individual patches
fail (error is captured in the result, not raised).

**Implementation Status:** Done (Exp 495)

### SCENARIO-INFRA-065: patch_script Replaces device_map='auto' with Explicit CUDA Assignment

**Given** a Python script containing `device_map='auto'` twice (two model loads)
**When** `HarnessPatcher.patch_script(path)` is called
**Then** the source is rewritten so the first occurrence becomes `device_map={'': 'cuda:0'}`
  and the second becomes `device_map={'': 'cuda:1'}`
**And** `HarnessPatchResult.was_patched` is `True` and `error` is `None`
**And** `HarnessAudit.scan()` on the patched script returns `needs_fix=False`

### SCENARIO-INFRA-066: verify_clean Returns 0 After patch_all on All Violations

**Given** a scripts directory containing only dual-model scripts with `needs_fix=True`
**When** `HarnessPatcher.patch_all(findings)` is applied to all findings
**Then** `HarnessPatcher.verify_clean(scripts_dir)` returns `0`
**And** every `HarnessPatchResult.success` in the returned list is `True`

## Exp 496: NUP Probe v2 — Bayesian Semantic Entropy

### REQ-VERIFY-098: BayesianEntropyEstimator Returns EntropyEstimate with Credible Interval

`BayesianEntropyEstimator.estimate(logprobs)` shall compute a Bayesian credible interval
over the Shannon entropy of the token probability distribution via a Beta-conjugate posterior
on the token probability mass, returning an `EntropyEstimate` with fields:
- `mean`: float — posterior mean entropy in nats
- `lower_ci`: float — lower bound of the (1 - alpha) credible interval
- `upper_ci`: float — upper bound of the (1 - alpha) credible interval
- `n_samples`: int — number of logprob samples used

**Why Beta posterior:**
    Token probabilities are bounded in [0, 1].  The Beta distribution is the conjugate prior
    for Bernoulli/Binomial likelihoods, making it the natural posterior for token-level
    probabilities.  The conjugate update is exact and O(n), requiring no MCMC.

**Why credible interval instead of point estimate:**
    A point-estimate entropy threshold triggers false positives on texts whose entropy is
    near the threshold but uncertain.  The credible interval separates 'confidently high
    entropy' (lower_ci > threshold) from 'uncertain' (lower_ci <= threshold <= upper_ci),
    reducing FP rate for ambiguous cases (arXiv 2603.22812, AAAI 2026 oral).

**Implementation Status:** Done (Exp 496)

### REQ-VERIFY-099: NUPProbeV2.predict_violation Uses upper_ci Threshold

`NUPProbeV2.predict_violation(cot_text, logprobs)` shall return `True` only when
`EntropyEstimate.lower_ci > hallucination_threshold` (i.e., the continuation is
*confidently* high-entropy), not when entropy is merely uncertain.  Cases where
`lower_ci <= hallucination_threshold <= upper_ci` are classified as indeterminate
and shall return `False` (no violation predicted), reducing false positives.

**Implementation Status:** Done (Exp 496)

### REQ-VERIFY-100: NUPProbeV2 AUC > 0.700 Qualifies as Tier 0c in ThreeTierPipeline

`NUPProbeV2.evaluate_auc(labeled_pairs)` shall return a float in [0, 1] representing
the ROC-AUC of `EntropyEstimate.mean` scores against ground-truth violation labels.
`NUPProbeV2Result.is_viable_tier_0c` shall be `True` when `auc > 0.700`.

**Implementation Status:** Done (Exp 496)

### SCENARIO-VERIFY-131: EntropyEstimate.is_confidently_high Only When lower_ci Exceeds Threshold

**Given** an `EntropyEstimate` with `lower_ci=2.0` and `threshold=1.5`
**When** `estimate.is_confidently_high(threshold=1.5)` is evaluated
**Then** it returns `True`

**Given** an `EntropyEstimate` with `lower_ci=1.0`, `upper_ci=2.0`, and `threshold=1.5`
**When** `estimate.is_confidently_high(threshold=1.5)` is evaluated
**Then** it returns `False` (uncertain region — lower_ci does not exceed threshold)

### SCENARIO-VERIFY-132: BayesianEntropyEstimator.estimate Gives Wide CI for Uniform Distribution

**Given** a uniform logprob distribution over N tokens (all logprobs equal)
**When** `BayesianEntropyEstimator(confidence_level=0.95).estimate(logprobs)` is called
**Then** `estimate.upper_ci > estimate.mean > estimate.lower_ci` (non-degenerate interval)
**And** `estimate.mean > 0.0` (positive entropy for N > 1)

### SCENARIO-VERIFY-133: NUPProbeV2.evaluate_auc Returns Float in [0, 1]

**Given** a list of labeled CoT pairs with 'step_text'/'cot_text' and 'label' fields
**When** `NUPProbeV2(hallucination_threshold=1.5).evaluate_auc(labeled_pairs)` is called
**Then** the return value is a float in [0.0, 1.0]

## Exp 501: Conductor CPU Routing — VRAMBudgetLedger

### REQ-INFRA-054: VRAMBudgetLedger Reads YAML Manifest and Produces Feasibility Forecast

`VRAMBudgetLedger` shall maintain a registry of experiment VRAM requirements
(registered via `add_experiment(exp_id, required_gb)`) and produce a feasibility
forecast for each planned experiment given the GPU's total VRAM minus the
conductor process's own footprint (`conductor_vram_gb`).  `to_yaml()` serializes
the ledger state for embedding in conductor milestone manifests.

**Why the conductor footprint is a first-class parameter:**
    The conductor process holds ~9 GiB GPU VRAM in JAX-GPU mode.  GPUVRAMGateV2
    cannot subtract this — it reads pynvml free VRAM which already has the conductor's
    allocation subtracted, but the gate's threshold does not know how large the incoming
    model is.  The ledger makes the conductor's footprint explicit at planning time,
    converting silent OOM into fast-fail with actionable root cause.

**Implementation Status:** Done (Exp 501)

### REQ-INFRA-055: VRAMBudgetLedger.check_feasibility() Returns VRAMForecast

`VRAMBudgetLedger.check_feasibility(exp_id)` shall return a `VRAMForecast` with:
- `is_feasible: bool` — True iff `required_gb <= available_gb`
- `required_gb: float` — experiment's registered VRAM requirement
- `available_gb: float` — `gpu_total_gb - conductor_vram_gb`
- `blocking_experiment: str | None` — None when feasible; `exp_id` when not feasible
- `headroom_gb: float` (property) — `available_gb - required_gb` (signed)

`check_all()` shall return one `VRAMForecast` per registered experiment in
registration order.

**Implementation Status:** Done (Exp 501)

### REQ-INFRA-056: VRAMBudgetLedger Supports CPU-Routing Mode

`VRAMBudgetLedger(conductor_vram_gb=0.0)` shall model a conductor routed to CPU
(JAX_PLATFORMS=cpu), where the conductor process holds 0 GiB GPU VRAM and the
full `gpu_total_gb` is available for experiment models.  This converts previously
infeasible experiments (where `required_gb > gpu_total_gb - 9.0`) into feasible
ones (where `required_gb <= gpu_total_gb`).

**Implementation Status:** Done (Exp 501)

### SCENARIO-INFRA-062: Feasible When Conductor=9 + Model=9 < Total=24

**Given** a `VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)`
**And** an experiment registered with `required_gb=9.0`
**When** `check_feasibility(exp_id)` is called
**Then** `VRAMForecast.is_feasible` is `True`
**And** `VRAMForecast.blocking_experiment` is `None`
**And** `VRAMForecast.headroom_gb` is `6.0`

### SCENARIO-INFRA-063: Not Feasible When Conductor=9 + Model=16 > Total=24

**Given** a `VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)`
**And** an experiment registered with `required_gb=16.0`
**When** `check_feasibility(exp_id)` is called
**Then** `VRAMForecast.is_feasible` is `False`
**And** `VRAMForecast.blocking_experiment` equals `exp_id`
**And** `VRAMForecast.headroom_gb` is `-1.0`

### SCENARIO-INFRA-064: CPU-Routing Mode Makes Previously Infeasible Experiments Feasible

**Given** a GPU-routed ledger `VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)`
**And** an experiment with `required_gb=18.0` that is **not feasible** under GPU routing
**When** a CPU-routed ledger `VRAMBudgetLedger(conductor_vram_gb=0.0, gpu_total_gb=24.0)`
  checks the same experiment
**Then** `VRAMForecast.is_feasible` is `True` (18.0 <= 24.0)
**And** the GPU-routed forecast has `is_feasible=False` (18.0 > 15.0)

---

## Exp 502: Live 100q Precision v6 — Gemma4QuantizedLoader + VRAMBudgetLedger (RETRO-033 close)

### REQ-BENCH-043: 100q v6 Uses Gemma4QuantizedLoader and GPUVRAMGateV2 Before Model Load

Experiment 502 shall use `Gemma4QuantizedLoader` (GGUF Q4_K_M, fits in 8-10 GiB) for
Gemma4 inference, and shall run `GPUVRAMGateV2(min_free_gb=6.0, kill_first=True)` as the
first gate before any model load.  `VRAMBudgetLedger` shall pre-check feasibility before
the gate fires.

**Implementation Status:** Done (Exp 502)

### REQ-BENCH-044: 100q v6 Assigns Gemma4-INT4 to cuda:0 and Qwen3.5-0.8B to cuda:1 via DualGPUHarness

Experiment 502 shall use `DualGPUHarness.apply()` to pin Gemma4-INT4 to `cuda:0` and
`Qwen3.5-0.8B` to `cuda:1` explicitly.  The dual-GPU assignment shall be recorded in the
artifact as `dual_gpu_explicit=True` (when eligible) to confirm the assignment occurred.

**Implementation Status:** Done (Exp 502)

### REQ-BENCH-045: 100q v6 Writes CoT Pairs to results/exp502_cot_pairs.json for JEPA Retraining

Experiment 502 shall collect Chain-of-Thought pairs during the pipeline pass for each model
and flush them atomically to `results/exp502_cot_pairs.json`.  The artifact shall include
`cot_pairs_written: int` recording the count.  At least 50 pairs per model are required to
support Exp 510 JEPA retrain (100 pairs minimum total).

**Implementation Status:** Done (Exp 502)

### SCENARIO-BENCH-062: GPUVRAMGateV2 Fires Before Model Load in Exp 502

**Given** Experiment 502 running on a machine with CUDA GPUs
**When** the experiment starts
**Then** `GPUVRAMGateV2(min_free_gb=6.0, kill_first=True).__enter__()` is called before any model is loaded
**And** if VRAM is insufficient after gate, a deferred artifact with
`honest_verdict='deferred_retro_033_v6'` is written and the experiment exits without loading models

### SCENARIO-BENCH-063: Exp 502 Assigns Gemma4-INT4 to cuda:0 and Qwen to cuda:1

**Given** Experiment 502 running with two RTX 3090 GPUs and `CARNOT_FORCE_LIVE=1`
**When** `DualGPUHarness.apply()` is called on the model specs
**Then** the Gemma4-INT4 spec has `gpu=0` and `device_map={'': 'cuda:0'}`
**And** the Qwen3.5-0.8B spec has `gpu=1` and `device_map={'': 'cuda:1'}`

### SCENARIO-BENCH-064: Exp 502 Writes Valid Deliverable with All Schema Fields

**Given** Experiment 502 completing successfully on live GPU hardware
**When** the artifact JSON is written
**Then** it contains schema='carnot.live_precision.v6', gemma4_result, qwen_result,
cot_pairs_written, gemma4_quantized=True, dual_gpu_explicit, retro_033_closed, and honest_verdict
**And** `DeliverableGuard.assert_written()` passes as the final call

---

## Exp 503: Live 200q VeriCoT+VPRM v4 — Gemma4QuantizedLoader resolves RETRO-038

### REQ-BENCH-046: 200q v4 Uses Gemma4QuantizedLoader and GPUVRAMGateV2 Before Model Load

Experiment 503 shall use `Gemma4QuantizedLoader` (GGUF Q4_K_M, ~9 GiB) for Gemma4
inference and shall run `GPUVRAMGateV2(min_free_gb=6.0, kill_first=True)` as the first
gate before any model load.  `VRAMBudgetLedger` shall pre-check feasibility (exp503
requires ~10 GiB) before the gate fires.

**Implementation Status:** Done (Exp 503)

### REQ-BENCH-047: 200q v4 Reports Wilson 95% CI and is_statistically_positive Flag

Experiment 503 shall compute `Live200qV4Result` per model with `ci_95_wilson` (Wilson
score 95% CI on post_acc) and `is_statistically_positive` (True iff the lower bound of
the Wald CI for the improvement > 0.0).  A credible positive result closes RETRO-038.

**Implementation Status:** Done (Exp 503)

### REQ-BENCH-048: 200q v4 Writes CoT Pairs to results/exp503_cot_pairs.json

Experiment 503 shall collect Chain-of-Thought pairs during the pipeline pass for each
model and flush them atomically to `results/exp503_cot_pairs.json`.  The artifact shall
include `cot_pairs_written: int`.  At least 100 pairs per model are required to support
Exp 510 JEPA retrain (200 pairs minimum total).

**Implementation Status:** Done (Exp 503)

### SCENARIO-BENCH-065: GPUVRAMGateV2 Fires Before Model Load in Exp 503

**Given** Experiment 503 running on a machine with CUDA GPUs
**When** the experiment starts
**Then** `GPUVRAMGateV2(min_free_gb=6.0, kill_first=True).__enter__()` is called before any model is loaded
**And** if VRAM is insufficient after gate, a deferred artifact with
`honest_verdict='deferred_retro_038_v4'` is written and the experiment exits without loading models

### SCENARIO-BENCH-066: Exp 503 is_statistically_positive Is False for 2pp Improvement at n=200

**Given** a `Live200qV4Result` with `pre_acc=0.70`, `post_acc=0.72`, `n=200`
**When** `is_statistically_positive` is evaluated
**Then** it returns `False`
**Because** SE ≈ 0.045, lower = 0.02 - 1.96*0.045 ≈ -0.069 < 0

### SCENARIO-BENCH-067: Exp 503 Writes Valid Deliverable with All Schema Fields

**Given** Experiment 503 completing successfully on live GPU hardware
**When** the artifact JSON is written
**Then** it contains schema='carnot.live_benchmark.v7', gemma4_result, qwen_result,
cot_pairs_written, gemma4_quantized=True, retro_038_closed, and honest_verdict
**And** `DeliverableGuard.assert_written()` passes as the final call


## Exp 504: GSM-Symbolic Adversarial v4 — Gemma4QuantizedLoader RETRO-039 Robustness Claim

### REQ-BENCH-049: Adversarial v4 Uses Gemma4QuantizedLoader and GPUVRAMGateV2

Experiment 504 shall use `Gemma4QuantizedLoader` (GGUF Q4_K_M, ~9 GiB) for Gemma4
inference and shall run `GPUVRAMGateV2(min_free_gb=6.0, kill_first=True)` before any
model load.  `VRAMBudgetLedger` shall pre-check feasibility before the gate fires.

**Implementation Status:** Done (Exp 504)

### REQ-BENCH-050: Adversarial v4 Measures robustness_delta per Model

Experiment 504 shall compute `AdversarialV4Result` per model with:
  - `standard_drop_baseline`: accuracy drop of the raw LLM under distractor injection
  - `standard_drop_pipeline`: accuracy drop of the Carnot pipeline under distractor injection
  - `robustness_delta = standard_drop_baseline - standard_drop_pipeline`
  - `carnot_more_robust = robustness_delta > 0`

A positive `robustness_delta` is the headline RETRO-039 credibility result.

**Implementation Status:** Done (Exp 504)

### REQ-BENCH-051: Adversarial v4 Reports All Four Accuracy Conditions

Experiment 504 shall report four accuracy values per model:
`standard_baseline`, `standard_pipeline`, `adversarial_baseline`, `adversarial_pipeline`.
The `to_dict()` method shall include all four raw accuracies plus all derived metrics.

**Implementation Status:** Done (Exp 504)

### SCENARIO-BENCH-068: robustness_delta > 0 When Pipeline Drop < Baseline Drop

**Given** an `AdversarialV4Result` with `standard_drop_baseline=0.15` and `standard_drop_pipeline=0.08`
**When** `robustness_delta` is evaluated
**Then** it returns `0.07` (15pp baseline drop minus 8pp pipeline drop)
**And** `carnot_more_robust` returns `True`

### SCENARIO-BENCH-069: carnot_more_robust=True Confirms RETRO-039 Thesis

**Given** a model where the baseline drops 15pp but the Carnot pipeline drops only 8pp
**When** `carnot_more_robust` is evaluated
**Then** it returns `True`
**And** the experiment artifact carries `retro_039_confirmed=True` and `honest_verdict='thesis_confirmed'`

### SCENARIO-BENCH-070: to_dict() Contains All Required Schema Fields

**Given** any `AdversarialV4Result` instance
**When** `to_dict()` is called
**Then** the result contains all twelve fields: `model_id`, `n`, `standard_baseline`,
`standard_pipeline`, `adversarial_baseline`, `adversarial_pipeline`, `standard_improvement`,
`adversarial_improvement`, `standard_drop_baseline`, `standard_drop_pipeline`,
`robustness_delta`, `carnot_more_robust`
**And** the dict is JSON-serializable without error

---

## Exp 516: GSM-Symbolic Adversarial v5 — RETRO-039 Robustness Claim (Four Consecutive Misses)

### REQ-BENCH-052: compute_robustness_delta Measures Carnot Robustness Advantage

`compute_robustness_delta(baseline_std, baseline_adv, pipeline_std, pipeline_adv)` shall
return `(baseline_std - baseline_adv) - (pipeline_std - pipeline_adv)`.  A positive value
means Carnot degrades LESS than the raw LLM under distractor injection (RETRO-039 thesis
confirmed).  No clamping — negative values are honest research findings.

**Implementation Status:** Done (Exp 516)

### REQ-BENCH-053: build_adversarial_v5_artifact Produces schema=carnot.adversarial_v5.v1

`build_adversarial_v5_artifact(results, inference_mode)` shall assemble the Exp 516 result
artifact with all required fields:

  - `baseline_standard_accuracy`, `baseline_adversarial_accuracy`
  - `pipeline_standard_accuracy`, `pipeline_adversarial_accuracy`
  - `robustness_delta` (computed via `compute_robustness_delta`)
  - `retro_039_confirmed = robustness_delta > 0 and inference_mode == 'live_gpu'`
  - `honest_verdict`: `'thesis_confirmed'` | `'thesis_rejected'` | `'gpu_required'`

**Implementation Status:** Done (Exp 516)

### SCENARIO-BENCH-037: compute_robustness_delta Returns Positive When Carnot More Robust

**Given** `baseline_std=0.80`, `baseline_adv=0.65`, `pipeline_std=0.78`, `pipeline_adv=0.70`
**When** `compute_robustness_delta` is called
**Then** it returns `0.07` (baseline drops 0.15, pipeline drops 0.08, delta = 0.07)
**And** a positive delta confirms the RETRO-039 thesis

**Implementation Status:** Done (Exp 516)

### SCENARIO-BENCH-038: build_adversarial_v5_artifact Selects honest_verdict from inference_mode and Delta

**Given** `robustness_delta > 0` and `inference_mode='live_gpu'`
**When** `build_adversarial_v5_artifact` is called
**Then** `honest_verdict == 'thesis_confirmed'` and `retro_039_confirmed == True`

**Given** `robustness_delta <= 0` and `inference_mode='live_gpu'`
**When** `build_adversarial_v5_artifact` is called
**Then** `honest_verdict == 'thesis_rejected'` and `retro_039_confirmed == False`

**Given** `inference_mode != 'live_gpu'` (e.g. `'simulated'`, `'gpu_required'`)
**When** `build_adversarial_v5_artifact` is called
**Then** `honest_verdict == 'gpu_required'` and `retro_039_confirmed == False`

**Implementation Status:** Done (Exp 516)

---

## REQ-INFRA-059: DualGPUSweep Patches All Dual-Model Scripts Missing DualGPUHarness

### REQ-INFRA-059: DualGPUSweep Patches All Dual-Model Scripts Missing DualGPUHarness

`DualGPUSweep` (Exp 505) shall scan all `scripts/experiment_*.py` files for dual-model
patterns (two or more `hf_id` occurrences) that do not already import `DualGPUHarness`.
For each matched script it shall append the standard DualGPUHarness injection block so
that `DualGPUHarness.from_env().apply(MODEL_SPECS)` is called at module level when
`CARNOT_FORCE_LIVE=1` is set.  The sweep is a best-effort retroactive fix addressing the
RETRO-041 finding that GPU 1 contributed 0% forward-pass compute across all milestones.

**Implementation Status:** Done (Exp 505)

### REQ-INFRA-060: DualGPUSweep Reports Patch Metrics via DualGPUSweepResult

`DualGPUSweep` (Exp 505) shall produce a `DualGPUSweepResult` dataclass with:
- `n_scripts_found`: total scripts matching the dual-model pattern (regardless of patch status)
- `n_scripts_patched`: scripts that were actually modified (injection block appended)
- `n_scripts_skipped`: scripts that already imported `DualGPUHarness` (no modification needed)
- `patch_manifest`: list of script filenames that were patched

`DualGPUSweepResult.patch_rate` shall return `n_scripts_patched / n_scripts_found` (0.0 when
n_scripts_found == 0).  `DualGPUSweepResult.to_dict()` shall return a JSON-serializable dict
with all four fields plus `patch_rate`.

**Implementation Status:** Done (Exp 505)

### SCENARIO-INFRA-067: patch_rate=1.0 When All Found Scripts Are Patched

**Given** a `DualGPUSweepResult` with `n_scripts_found=3`, `n_scripts_patched=3`, `n_scripts_skipped=0`
**When** `patch_rate` is evaluated
**Then** it returns `1.0`
**And** `to_dict()` contains `patch_rate=1.0` and `patch_manifest` with three entries

### SCENARIO-INFRA-068: Skipped Scripts Are Those Already Using DualGPUHarness

**Given** a `DualGPUSweepResult` with `n_scripts_found=5`, `n_scripts_patched=3`, `n_scripts_skipped=2`
**When** `patch_rate` is evaluated
**Then** it returns `0.6` (3 patched out of 5 found)
**And** `to_dict()['n_scripts_skipped']` equals `2`
**And** `to_dict()['patch_manifest']` contains only the three patched script names

---

## Exp 506: BoltzmannSemanticEnergy Tier 0d — Semantic Clustering + Boltzmann-Weighted Energy

### REQ-VERIFY-101: BoltzmannSemanticEnergy Clusters Token Logits into Semantic Clusters via Cosine-Similarity K-Means

`BoltzmannSemanticEnergy` shall accept `n_clusters: int` and `temperature: float` parameters.
Its `cluster(token_logits: dict[str, float]) -> list[SemanticCluster]` method shall:
- Compute a character-level embedding for each token (cosine-similarity proxy for semantic similarity)
- Group tokens into `n_clusters` clusters via k-means
- For each cluster compute `mean_logit` (average of member logits) and `cluster_energy`
  (Boltzmann energy: `-mean_logit / temperature`)

**Implementation Status:** Done (Exp 506)

### REQ-VERIFY-102: BoltzmannSemanticEnergy.score() Returns a Hallucination Score in [0, 1]

`BoltzmannSemanticEnergy.score(response: str, token_logits: dict[str, float]) -> float`
shall return a float in `[0.0, 1.0]` where `1.0` indicates high confidence of hallucination
and `0.0` indicates low hallucination risk.  The score is derived from the Boltzmann-weighted
sum of cluster energies, normalised via sigmoid so it maps monotonically to `[0, 1]`.

**Implementation Status:** Done (Exp 506)

### REQ-VERIFY-103: BoltzmannSemanticEnergy Is Tier 0d in the Verification Cascade

`BoltzmannSemanticEnergy` is positioned as Tier 0d in the Carnot verification cascade,
inserted between Tier 0b SpilledEnergy and Tier 1 SinkProbe.  Its `benchmark()` method
shall return a dict with at minimum `auroc` and `skip_rate` keys.

**Implementation Status:** Done (Exp 506)

### SCENARIO-VERIFY-134: score() Returns Float in [0, 1]

**Given** a `BoltzmannSemanticEnergy` instance
**And** any valid `token_logits` dict
**When** `score()` is called
**Then** the returned value is a float in the closed interval `[0.0, 1.0]`

### SCENARIO-VERIFY-135: High-Logit-Variance Input Yields Higher Score Than Low-Variance

**Given** two `token_logits` dicts — one with uniform low logits (confident model) and one with high-variance logits (uncertain model)
**When** `score()` is called on both
**Then** the high-variance input returns a strictly higher score than the low-variance input

### SCENARIO-VERIFY-136: benchmark() on 10 Synthetic Returns auroc Key

**Given** a `BoltzmannSemanticEnergy` instance
**And** a list of 10 synthetic `(response, token_logits)` pairs with known ground truth
**When** `benchmark()` is called
**Then** the returned dict contains the key `auroc` with a float value

---

## REQ-VERIFY-104: CLAPFeatureExtractor Constructs Cross-Layer Activation Tensor Features

**Requirement:** CLAPFeatureExtractor constructs a (n_layers, n_tokens, hidden_dim) activation tensor from the last n_layers residual stream layers and computes per-token softmax entropy, top-k concentration, and cross-layer L2-norm variance features.

**Rationale:** NUP Probe v1 and v2 used sequence-level aggregates (character entropy, Bayesian SE) that average away the token-level hallucination signal. CLAP's insight (arXiv 2509.09700) is that hallucination fingerprints are LOCAL in the residual stream — a specific layer and token position where the model becomes uncertain. Cross-layer activation features capture this locality.

**Acceptance criteria:**
- extract_features(activations) accepts shape (n_layers, n_tokens, hidden_dim)
- Returns CLAPFeatures with per_token_entropy, topk_concentration, cross_layer_variance each of shape (n_tokens,)
- Raises ValueError for non-3D input or n_layers mismatch

Spec: REQ-VERIFY-104

### REQ-VERIFY-105: CLAPFeatures.to_feature_vector Returns Normalised 1D Array

**Requirement:** CLAPFeatures.to_feature_vector() flattens and z-score normalises the three feature arrays into a single 1D np.ndarray of length 3 * n_tokens.

**Acceptance criteria:**
- Output length == 3 * n_tokens
- No NaN for any valid input
- Constant-value input yields zero vector (degenerate std path)

Spec: REQ-VERIFY-105

### REQ-VERIFY-106: NUPProbeV3 Trained on Real CoT Pairs Achieves AUC >= 0.700

**Requirement:** NUPProbeV3 (logistic classifier on CLAP features) trained on real CoT pairs from Exps 502-503 achieves AUC >= 0.700 on held-out pairs, qualifying for Tier 0c promotion and closing RETRO-049.

**Acceptance criteria:**
- fit(pairs, labels) trains a logistic classifier on CLAP feature vectors
- predict(features) returns float in [0, 1]
- evaluate(pairs, labels) returns dict with 'auroc' key in [0, 1]
- On real GPU data: auroc >= 0.700

Spec: REQ-VERIFY-106

### SCENARIO-VERIFY-137: CLAPFeatureExtractor on (4, 20, 768) Returns CLAPFeatures

**Given** a CLAPFeatureExtractor with n_layers=4
**And** a random activation tensor of shape (4, 20, 768)
**When** extract_features() is called
**Then** it returns a CLAPFeatures instance with all three arrays of shape (20,)

### SCENARIO-VERIFY-138: NUPProbeV3.fit on 20 Synthetic Pairs Converges

**Given** a NUPProbeV3 instance
**And** 20 synthetic (activations, label) pairs
**When** fit() is called
**Then** _is_fitted is True and _weights has the correct shape

### SCENARIO-VERIFY-139: NUPProbeV3.evaluate Returns auroc Key in [0, 1]

**Given** a fitted NUPProbeV3 instance
**And** held-out (activations, label) pairs
**When** evaluate() is called
**Then** the returned dict contains 'auroc' with a float in [0, 1]

---

## REQ-INFRA-061: NPUEntropyProbe Exports Per-Token Entropy to ONNX and Loads via VitisAI EP

**Requirement:** NPUEntropyProbe exports the softmax entropy computation graph (softmax(logits) followed by H(p) = -sum(p*log(p))) to ONNX format and attempts to load it via the onnxruntime VitisAI execution provider for AMD XDNA NPU inference.

**Rationale:** Per-token entropy is O(vocab_size=50k) parallel floating-point ops per token. The AMD XDNA NPU's spatial AI Engine array can compute this in a single pass across the vocabulary dimension, potentially achieving <5ms/token latency. If NPU latency < LLM generation latency (5-50ms/token at 20-100 tokens/sec), the entropy probe can pipeline with generation and add zero overhead to Tier 0c filtering.

**Acceptance criteria:**
- export_onnx(path) creates a valid ONNX file at the specified path
- load_vitisai(onnx_path) returns True when VitisAI EP is available, False otherwise
- compute_entropy(activations) returns per-token entropy array of shape (seq_len,)

**Implementation Status:** Done (Exp 511)

---

## REQ-INFRA-062: NPUEntropyProbe.benchmark() Measures Latency in ms/token on NPU vs CPU

**Requirement:** NPUEntropyProbe.benchmark(n_trials) measures wall-clock entropy computation latency in milliseconds per token on both NPU (if available) and CPU baseline, returning an NPUBenchmarkResult with both measurements and the speedup ratio.

**Acceptance criteria:**
- benchmark(n_trials=100) returns NPUBenchmarkResult
- cpu_latency_ms is always measured (no NPU required)
- npu_latency_ms is None when NPU is unavailable
- speedup_ratio = cpu_latency_ms / npu_latency_ms when both available, else None
- npu_viable = True iff npu_available and speedup_ratio >= 2.0

**Implementation Status:** Done (Exp 511)

---

## REQ-INFRA-063: NPUEntropyProbe Emits honest_verdict='npu_not_available' With Setup Instructions

**Requirement:** When the VitisAI execution provider is not installed or fails to load, NPUEntropyProbe.benchmark() returns NPUBenchmarkResult(npu_available=False) and the experiment script emits honest_verdict='npu_not_available' along with step-by-step VitisAI EP installation instructions. It must NOT fail silently.

**Acceptance criteria:**
- NPUBenchmarkResult.npu_available=False when VitisAI EP is absent
- NPUBenchmarkResult.npu_viable=False when npu_available=False
- NPUBenchmarkResult.speedup_ratio=None when npu_available=False
- Experiment artifact includes setup_instructions string when npu_available=False

**Implementation Status:** Done (Exp 511)

---

### SCENARIO-INFRA-070: NPUBenchmarkResult.npu_viable False When NPU Unavailable

**Given** an NPUBenchmarkResult with npu_available=False
**When** npu_viable is accessed
**Then** it returns False

### SCENARIO-INFRA-071: NPUBenchmarkResult.speedup_ratio None When NPU Unavailable

**Given** an NPUBenchmarkResult with npu_available=False
**When** speedup_ratio is accessed
**Then** it returns None

### SCENARIO-INFRA-072: export_onnx Creates File at Given Path

**Given** an NPUEntropyProbe instance
**And** a temporary file path
**When** export_onnx(path) is called
**Then** the file exists at the given path

---

## REQ-INFRA-064: JITVRAMCheck Queries Real-Time VRAM Immediately Before Each model.load() Call

**Requirement:** JITVRAMCheck.gate_model_load(model_id, required_gb) queries pynvml for current free VRAM on the target device immediately before each model.load() invocation — not at script startup. The result is returned as a JITVRAMResult with is_cleared, available_gb, attempts, and wait_applied fields.

**Rationale:** Planning-time VRAM forecasts (VRAMBudgetLedger, Exp 500) are computed once at startup. By the time model.load() actually runs, the VRAM state may have changed (conductor loaded a model, prior model still allocated). This staleness caused CUDA OOM in Exps 502/503/504 despite a passing planning-time forecast. JIT queries resolve this by checking in the same call frame as the load.

**Acceptance criteria:**
- gate_model_load(model_id, required_gb) returns JITVRAMResult
- is_cleared=True iff available_gb >= required_gb at check time
- available_gb reflects the reading from the last pynvml query
- attempts=1 when first check passes

**Implementation Status:** Done (Exp 513)

---

## REQ-INFRA-065: JITVRAMCheck Retries Once After retry_wait_s=30 If Insufficient

**Requirement:** If the first VRAM check fails (available_gb < required_gb), JITVRAMCheck.gate_model_load() waits retry_wait_s seconds (default 30) and queries pynvml once more. The result includes attempts=2 and wait_applied=True. If the second check also fails, is_cleared=False is returned and the caller must abort the load.

**Acceptance criteria:**
- attempts=2 and wait_applied=True when first check fails
- is_cleared=True if second check passes
- is_cleared=False if second check also fails
- Only one retry — never more than 2 attempts

**Implementation Status:** Done (Exp 513)

---

## REQ-INFRA-066: JITVRAMCheck CI Stub Returns is_cleared=True When pynvml Not Installed

**Requirement:** When pynvml is not installed (CI, CPU-only machines), JITVRAMCheck.get_available_gb() returns 24.0 GB so that gate_model_load() always returns is_cleared=True in CI. This prevents CI from being blocked by missing GPU hardware.

**Acceptance criteria:**
- get_available_gb() returns 24.0 when pynvml ImportError occurs
- gate_model_load() returns is_cleared=True in CI stub mode
- CI stub does not raise exceptions

**Implementation Status:** Done (Exp 513)

---

### SCENARIO-INFRA-073: gate_model_load Returns is_cleared=True When available_gb >= required_gb

**Given** a JITVRAMCheck on device 0
**And** available_gb=20.0 at query time
**When** gate_model_load("model", required_gb=10.0) is called
**Then** JITVRAMResult.is_cleared is True and attempts=1 and wait_applied=False

### SCENARIO-INFRA-074: gate_model_load Returns attempts=2 and wait_applied=True When First Check Fails

**Given** a JITVRAMCheck on device 0
**And** available_gb=8.0 on first check and available_gb=12.0 on second check
**When** gate_model_load("model", required_gb=10.0, retry_wait_s=30) is called
**Then** JITVRAMResult.attempts=2, wait_applied=True, is_cleared=True, available_gb=12.0

### SCENARIO-INFRA-075: gate_model_load Returns is_cleared=False When Both Checks Fail

**Given** a JITVRAMCheck on device 0
**And** available_gb=5.0 on both checks
**When** gate_model_load("model", required_gb=10.0) is called
**Then** JITVRAMResult.is_cleared=False and attempts=2 and wait_applied=True

---

### REQ-BENCH-014: Live 100q Benchmark with JIT VRAM Gating (Exp 514)

The experiment shall run a live 100-question GSM8K precision benchmark where:
- `JITVRAMCheck` gates every model.load() call immediately before it fires (RETRO-051 fix)
- Gemma4-INT4 is loaded on cuda:0 gated by `JITVRAMCheck(0).gate_model_load('gemma4-int4', required_gb=10.0)`
- Qwen3.5-0.8B is loaded on cuda:1 gated by `JITVRAMCheck(1).gate_model_load('qwen3.5-0.8b', required_gb=1.5)`
- If either gate returns `is_cleared=False`, the experiment writes a deferred artifact rather than crashing with CUDA OOM
- The artifact includes `jit_vram_check_applied=True` so downstream tooling can confirm the fix was active
- `retro_033_closed=True` iff `inference_mode=='live_gpu'` and `pipeline_accuracy > baseline_accuracy`

**Implementation Status:** Done (Exp 514)

### REQ-BENCH-015: CoT Pairs Written to exp514_cot_pairs.json for JEPA Retrain (Exp 514)

The experiment shall write 100 FOVER-format CoT pairs to `results/exp514_cot_pairs.json` where:
- Each pair contains: `question` (str), `cot_text` (str), `correct` (bool), `model_id` (str)
- The file is written atomically (tmp-rename) to prevent partial writes on interrupt
- The artifact field `cot_pairs_written` records the path when pairs are successfully written, else `null`

**Implementation Status:** Done (Exp 514)

### SCENARIO-BENCH-033: JIT VRAM Gate Blocks Load When VRAM Insufficient in Exp 514

**Given** `JITVRAMCheck(0).gate_model_load('gemma4-int4', required_gb=10.0)` returns `is_cleared=False`
**When** `load_jit_gated_model(Gemma4QuantizedLoader, 'gemma4-int4', 10.0, 0)` is called
**Then** the function returns `None` without calling `loader.load()`
**And** the experiment writes a deferred artifact with `inference_mode='gpu_required'`

### SCENARIO-BENCH-034: write_cot_pairs Produces Valid FOVER JSON

**Given** a list of 10 CoT response strings and 10 boolean labels
**When** `write_cot_pairs(responses, labels, path)` is called
**Then** the file at `path` is valid JSON containing 10 entries
**And** each entry has keys `question`, `cot_text`, `correct`, `model_id`
**And** the function returns 10

---

## Exp 515: Live 200q VeriCoT+VPRM v5 (RETRO-038 fifth attempt)

### REQ-BENCH-016: 200q Wilson 95% CI Lower > 0 Constitutes First Publishable Claim

**Requirement:** The 200q live benchmark must compute Wilson 95% CI on the pipeline improvement
delta and report `is_statistically_positive=(wilson_95ci_lower > 0)`. When this flag is True
and `inference_mode=='live_gpu'`, the experiment must set `retro_038_closed=True` and
`honest_verdict='first_publishable_claim'`.

**Rationale:** Wilson CI lower bound > 0 at n=200 provides the statistical power needed for
a publishable credibility claim. This is the first gating criterion for RETRO-038 closure.

**Acceptance:** `wilson_95ci_lower > 0` observed under live GPU inference on 200 GSM8K questions.

Spec: REQ-BENCH-016, SCENARIO-BENCH-035, SCENARIO-BENCH-036

### SCENARIO-BENCH-035: compute_wilson_ci Returns Correct Bounds at n=200

**Given** `n_successes=120`, `n_total=200`, `confidence=0.95`
**When** `compute_wilson_ci(120, 200)` is called
**Then** the returned `(lower, upper)` tuple satisfies `lower > 0.53` and `upper < 0.67`
**And** both bounds are in [0.0, 1.0]
**And** `lower < 0.6 < upper` (CI straddles the observed proportion)

### SCENARIO-BENCH-036: is_statistically_positive Returns True iff Lower Bound > 0

**Given** `wilson_lower_bound > 0` (e.g. 0.001)
**When** `is_statistically_positive(wilson_lower_bound)` is called
**Then** the function returns `True`

**Given** `wilson_lower_bound == 0.0`
**When** `is_statistically_positive(0.0)` is called
**Then** the function returns `False`

**Given** `wilson_lower_bound < 0` (e.g. −0.001, should not arise in practice but guarded)
**When** `is_statistically_positive(−0.001)` is called
**Then** the function returns `False`

---

## Exp 525: ExpandedGPUReaper (RETRO-033 root cause fix)

### REQ-INFRA-067: ExpandedGPUReaper Kills Out-of-Subtree GPU Processes Above Age and VRAM Thresholds

**Requirement:** `ExpandedGPUReaper.reap()` must scan `nvidia-smi --query-compute-apps` output
and kill any process satisfying ALL of: (a) `used_memory_mb >= min_vram_mb`, (b) not a
descendant of the current conductor process subtree, and (c) `age_s >= min_age_s`.

**Rationale:** Stale pytest orphan processes (pattern: `python3 -u -c ...` children of a now-dead
pytest run) are indistinguishable by name from legitimate conductor subagent children, so
name-based whitelists (GPUVRAMGateV2, JITVRAMCheck) silently miss them.  A subtree check
uses the kernel's actual parent-child relationships, which are exact and cannot be spoofed.

**Acceptance:** reap() returns ExpandedGPUReapResult with killed list non-empty and
total_vram_freed_mb > 0 when eligible orphan processes are present.

Spec: REQ-INFRA-067, SCENARIO-INFRA-076, SCENARIO-INFRA-077

### REQ-INFRA-068: ExpandedGPUReaper Integrates Into GPUVRAMGateV2 as Pre-Kill Pass

**Requirement:** GPUVRAMGateV2 must optionally accept an ExpandedGPUReaper instance and invoke
`reaper.reap()` BEFORE its own pattern-based kill pass.

**Rationale:** The expanded reaper's broader heuristic catches orphans that the name-based
killer misses.  Running it first frees more VRAM before the threshold check, reducing the
probability of a false `gpu_vram_insufficient` deferral.

**Acceptance:** (Integration step — not implemented in Exp 525; requires human review per
CLAUDE.md.  This REQ documents the intended integration point.)

Spec: REQ-INFRA-068

### REQ-INFRA-069: ExpandedGPUReaper Provides a pyxrt-Free Reaper Path

**Requirement:** `expanded_gpu_reaper.py` must not import pyxrt, pynvml, or any NPU/GPU Python
binding.  All GPU process enumeration must use subprocess calls to `nvidia-smi` and `ps`.

**Rationale:** The NPU stack (pyxrt) is not always installed.  A GPU VRAM reclamation utility
that requires the full NPU stack would be unusable on standard CUDA-only hosts.

**Acceptance:** `python3 -c "from carnot.pipeline.expanded_gpu_reaper import ExpandedGPUReaper"`
succeeds on a host with only CUDA drivers (no pyxrt, no pynvml).

Spec: REQ-INFRA-069, SCENARIO-INFRA-078

### SCENARIO-INFRA-076: reap() Returns dry_run_candidate Entries When dry_run=True

**Given** `ExpandedGPUReaperConfig(min_vram_mb=1024, min_age_s=1800, dry_run=True)`
**And** one GPU process: pid=555, used_memory_mb=4096, age_s=3600, not in subtree
**When** `ExpandedGPUReaper(cfg).reap()` is called
**Then** `result.honest_verdict == 'reap_dry_run_complete'`
**And** `result.killed == []`
**And** `result.skipped[0]['reason'] == 'dry_run_candidate'`
**And** `result.total_vram_freed_mb == 0`

**Implementation Status:** Done (Exp 525)

### SCENARIO-INFRA-077: reap() Kills Eligible Process and Reports VRAM Freed

**Given** `ExpandedGPUReaperConfig(min_vram_mb=1024, min_age_s=1800, dry_run=False)`
**And** one GPU process: pid=777, used_memory_mb=6144, age_s=7200, not in subtree
**When** `ExpandedGPUReaper(cfg).reap()` is called
**Then** `os.kill(777, SIGKILL)` is invoked
**And** `result.killed[0]['action'] == 'killed'`
**And** `result.total_vram_freed_mb == 6144`
**And** `result.honest_verdict == 'reap_complete'`

**Implementation Status:** Done (Exp 525)

### SCENARIO-INFRA-078: reap() Returns no_nvidia_smi_no_reap When nvidia-smi Not in PATH

**Given** `shutil.which('nvidia-smi')` returns `None`
**When** `ExpandedGPUReaper().reap()` is called
**Then** `result.honest_verdict == 'no_nvidia_smi_no_reap'`
**And** `result.killed == []` and `result.skipped == []`

**Implementation Status:** Done (Exp 525)

---

## Exp 517: Controlled DualGPU Test (RETRO-052 resolution)

### REQ-INFRA-070: DualGPUControlledTest Measures GPU 1 Compute Utilization During Inference

**Requirement:** `sample_gpu_utilization(device_ids, n_samples, interval_s)` must poll
`nvmlDeviceGetUtilizationRates()` for each device_id `n_samples` times and return the
mean compute utilization percentage per device.  `run_dual_inference(model_a, model_b,
prompts)` must run both models simultaneously in threads.  `DualGPUTestResult` must carry
`gpu0_compute_pct`, `gpu1_compute_pct`, `n_samples_run`, `inference_mode`, and
`honest_verdict`.

**Rationale:** RETRO-052: the DualGPU sweep (Exp 505) found n_scripts_patched=0, yet GPU 1
remained at 0% compute.  Either all scripts were already patched (and another root cause
exists) or the sweep's detection missed eligible scripts.  This experiment answers the
question definitively: does forward-pass compute actually run on GPU 1?  It loads one model
per GPU simultaneously, then samples utilization during parallel inference.

**Acceptance:** The experiment writes `results/experiment_517_dual_gpu_controlled_test.json`
with `gpu1_utilization_verified` and `retro_052_status` fields.  CI stub returns
`{0: 0.0, 1: 0.0}` for both devices so the test suite passes without GPU hardware.

Spec: REQ-INFRA-070, SCENARIO-INFRA-079, SCENARIO-INFRA-080

### SCENARIO-INFRA-079: sample_gpu_utilization Returns Mean Per Device Over n_samples

**Given** `pynvml.nvmlDeviceGetUtilizationRates()` returns `.gpu == 50` for device 0
  and `.gpu == 30` for device 1 on every call
**When** `sample_gpu_utilization([0, 1], n_samples=4, interval_s=0.0)` is called
**Then** the returned dict has key 0 with value 50.0 and key 1 with value 30.0
**And** pynvml was called exactly 4 times per device

**CI stub:** When pynvml is not installed, returns `{device_id: 0.0}` for all device_ids
so the test suite can verify the data contract without GPU hardware.

**Implementation Status:** Done (Exp 517)

### SCENARIO-INFRA-080: DualGPUTestResult honest_verdict Reflects Inference Mode and GPU 1 Activity

**Given** `inference_mode='live_gpu'` and `gpu1_compute_pct > 10.0`
**When** `DualGPUTestResult` is constructed
**Then** `honest_verdict == 'gpu1_active'`

**Given** `inference_mode='live_gpu'` and `gpu1_compute_pct <= 10.0`
**When** `DualGPUTestResult` is constructed
**Then** `honest_verdict == 'gpu1_idle'`

**Given** `inference_mode='gpu_required'` (GPU not available)
**When** `DualGPUTestResult` is constructed
**Then** `honest_verdict == 'gpu_required'`

**Implementation Status:** Done (Exp 517)


### REQ-VERIFY-107: HallucinationBasinDetector Estimates Basin Depth from Hidden States

**Requirement:** `estimate_basin_depth(hidden_state, energy_fn, n_perturbations, perturbation_scale)`
must compute `energy_at_x = energy_fn(hidden_state)`, then generate `n_perturbations` random
perturbations at `perturbation_scale` magnitude, evaluate `energy_fn` at each perturbed point,
and return `energy_at_x - min(perturbed_energies)`.  Positive depth means a local minimum
(deep basin = low hallucination risk); near-zero or negative depth means a saddle or flat
region (shallow basin = high hallucination risk).

**Rationale:** arXiv 2604.04743 shows correct reasoning follows deep-basin attractors in LLM
latent space while hallucinated reasoning drifts into shallow basins with high escape
probability.  Basin depth is computable from hidden-state trajectories without additional
model passes — it only requires a fast energy proxy callable.

**Acceptance:** `estimate_basin_depth` returns a float.  For a global-minimum state
(all zeros, quadratic energy) depth is positive.  For a saddle-point state depth is near zero.

Spec: REQ-VERIFY-107, SCENARIO-VERIFY-140, SCENARIO-VERIFY-141

### REQ-VERIFY-108: HallucinationBasinDetector.detect Returns BasinEstimate with basin_risk_score

**Requirement:** `HallucinationBasinDetector.detect(hidden_states)` must:
- Compute per-timestep basin depth over all rows of `hidden_states`
- Compute `mean_depth = mean(depth_t over T timesteps)`
- Compute `escape_probability = sigmoid(-mean_depth)` (low depth → high escape probability)
- Compute `basin_risk_score = 1.0 - sigmoid(mean_depth)` (high score = shallow basin = risky)
- Return a `BasinEstimate` with all three fields

A `basin_risk_score > 0.5` indicates hallucination risk; `< 0.5` indicates low risk.

**Rationale:** The sigmoid of basin depth maps depth to a probability-like risk score in [0, 1].
Deep basins (positive depth) yield low risk scores; shallow basins (near-zero depth) yield
high risk scores.  This is a Tier 0d candidate positioned after SpilledEnergy (Tier 0b)
in the verification cascade.

**Acceptance:** Deep-basin trajectories (synthetic depth ≈ 2.0) give `basin_risk_score < 0.5`.
Shallow-basin trajectories (synthetic depth ≈ 0.1) give `basin_risk_score > 0.5`.

Spec: REQ-VERIFY-108, SCENARIO-VERIFY-142

### SCENARIO-VERIFY-140: estimate_basin_depth Returns Positive Depth for Global Minimum

**Given** a quadratic energy function `energy_fn(x) = sum(x**2)` (minimum at origin)
**And** a hidden state `x = zeros(8)` (at the global minimum)
**When** `estimate_basin_depth(x, energy_fn, n_perturbations=8, perturbation_scale=0.1)` is called
**Then** the returned depth is >= 0.0
**Because** perturbations move away from the minimum, so `min(perturbed_energies) >= energy_at_x`

**Implementation Status:** Done (Exp 521)

### SCENARIO-VERIFY-141: estimate_basin_depth Returns Near-Zero Depth for Saddle Point

**Given** a linear energy function `energy_fn(x) = x[0]` (no minimum — flat along all but one axis)
**And** a hidden state `x = zeros(8)`
**When** `estimate_basin_depth(x, energy_fn, n_perturbations=8, perturbation_scale=0.1)` is called
**Then** the returned depth is close to 0.0 (within 0.2 of zero in absolute value)
**Because** linear functions have no local minima; perturbations in the gradient direction
reduce energy while perturbations against the gradient increase it — overall the minimum
of perturbed energies is roughly equal to the current energy.

**Implementation Status:** Done (Exp 521)

### SCENARIO-VERIFY-142: detect on Deep-Basin Trajectories Gives basin_risk_score < 0.5

**Given** a quadratic energy function and 10 hidden states all near the global minimum
**When** `HallucinationBasinDetector(energy_fn).detect(hidden_states)` is called
**Then** `basin_estimate.basin_risk_score < 0.5`
**And** `basin_estimate.escape_probability < 0.5`
**And** `basin_estimate.basin_depth >= 0.0`

**Given** the same detector and 10 hidden states near a saddle point (flat region)
**When** `detect(hidden_states)` is called
**Then** `basin_estimate.basin_risk_score > 0.5`

**Implementation Status:** Done (Exp 521)


### REQ-VERIFY-109: NUPProbeV4 Contrastive Margin Loss Maximises E(incorrect)-E(correct) Gap

**Requirement:** `ContrastivePairLoss(margin).loss(energy_incorrect, energy_correct)` must return
`max(0, margin - (energy_incorrect - energy_correct))`.  `NUPProbeV4.train_contrastive(correct_steps,
incorrect_steps, n_epochs)` must minimise this loss over all (correct, incorrect) pairs using SGD,
pushing `E(incorrect) - E(correct) >= margin` for all training pairs.

**Rationale:** Binary cross-entropy (BCE) optimises a classification boundary — it is indifferent to
the magnitude of the energy gap between correct and incorrect steps.  When correct and incorrect steps
embed similarly (subtle hallucinations), BCE cannot distinguish them.  Contrastive margin loss directly
optimises the EBM verification invariant: `E(incorrect) >> E(correct)`.  The margin enforces a minimum
energy difference, not just correct ordering.

**Acceptance:** `loss(2.0, 1.0)` == 0.0 (gap meets margin).  `loss(2.0, 2.0)` == margin (gap is zero).
`train_contrastive` returns a dict with keys `converged`, `final_loss`, `final_auc`, `loss_history`.

Spec: REQ-VERIFY-109, SCENARIO-VERIFY-143, SCENARIO-VERIFY-144

### REQ-VERIFY-110: NUPProbeV4 AUC >= 0.700 Qualifies as Tier 0c

**Requirement:** `NUPProbeV4.evaluate_auc(correct_steps, incorrect_steps)` must return AUROC >= 0.700
on real CoT step data after contrastive training.  AUROC >= 0.700 means `tier0c_promoted=True` and
`retro_049_closed=True` in the Exp 523 artifact.

**Rationale:** AUC 0.700 is the Tier 0c threshold established in REQ-VERIFY-097.  Contrastive training
is the fix for the RETRO-049 root cause (BCE objective mismatch).

**Acceptance:** Exp 523 deliverable has `final_auc >= 0.700` and `tier0c_promoted=True`.

Spec: REQ-VERIFY-110, SCENARIO-VERIFY-145

### SCENARIO-VERIFY-143: ContrastivePairLoss Returns Zero When Gap Meets Margin

**Given** `ContrastivePairLoss(margin=1.0)` and `energy_incorrect=2.0`, `energy_correct=1.0`
**When** `loss(energy_incorrect, energy_correct)` is called
**Then** the result is 0.0 (gap of 1.0 equals margin — constraint satisfied)

**Implementation Status:** Done (Exp 523)

### SCENARIO-VERIFY-144: ContrastivePairLoss Returns Margin When Energies Equal

**Given** `ContrastivePairLoss(margin=1.0)` and `energy_incorrect=2.0`, `energy_correct=2.0`
**When** `loss(energy_incorrect, energy_correct)` is called
**Then** the result is 1.0 (gap is zero — maximum deficit equal to margin)

**Implementation Status:** Done (Exp 523)

### SCENARIO-VERIFY-145: NUPProbeV4.train_contrastive Returns converged and final_auc in [0, 1]

**Given** a `NUPProbeV4` with default parameters
**And** lists of correct and incorrect CoT steps (at least 2 each)
**When** `train_contrastive(correct_steps, incorrect_steps, n_epochs=50)` is called
**Then** the returned dict has `converged` (bool), `final_auc` in [0.0, 1.0], `final_loss >= 0`,
and `loss_history` with `n_epochs` entries

**Implementation Status:** Done (Exp 523)
