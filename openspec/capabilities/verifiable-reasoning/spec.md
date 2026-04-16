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

---

## Operational Retrospective Requirements (REQ-RETRO-*)

These requirements govern the per-milestone operational retrospectives produced by
the research conductor.  Each retro summarises wall-time data, identifies bottlenecks,
audits whether prior action items were resolved, and proposes new improvements.

### REQ-RETRO-001: Milestone 2026.04.29 Retrospective

The conductor must produce `results/operational_retro_2026_04_29.json` with schema
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
**When** the 2026.04.29 retro runs
**Then** both items appear in `action_items` with `status == "carried_forward"`
**And** at least one NEW-* item appears with `status == "new"`

---

### REQ-RETRO-002: Milestone 2026.04.29 (v2) Full-History Retrospective

An extended version of the 2026.04.29 retro that covers the full 359-experiment history.
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

### REQ-RETRO-003: Milestone 2026.05.06 Retrospective

The conductor must produce `results/operational_retro_2026_05_06.json` with schema
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
- `carry_over`: list of dicts `{id, description, resolved: bool}` from the 2026.04.29 retro
- `action_items`: list of dicts with `{id, description, estimated_impact_pct}`

### SCENARIO-RETRO-005: All Prior Action Items Resolved Flag Correctly

**Given** RETRO-001 was implemented in Exp 325 and RETRO-002 in Exp 326
**When** the 2026.05.06 retro artifact is loaded
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
| REQ-REPAIR-010 | Not Started | Implemented | Z3GatedRepair + Z3GatedRepairResult + pipeline integration tests (SCENARIO-REPAIR-020/021, Exp 312) |
| REQ-REPAIR-011 | Not Started | Implemented | Z3 SAT fast-exit path tests (SCENARIO-REPAIR-022, Exp 312) |
| REQ-BENCH-001 | Not Started | Script written (Exp 315) | Full-scale benchmark script with 95% Wilson CI; execution in Exp 316 |
| REQ-BENCH-002 | Not Started | Implemented (Exp 328) | Live GPU benchmark result with inference_mode="live_gpu"; wrapper artifact with simulation_divergence + baseline_deviation; honest blocked artifact when GPU unavailable |
| REQ-BENCH-003 | Not Started | Implemented (Exp 340) | PrecisionStackResult + PipelineVariant + compute_signed_improvement + build_precision_benchmark_artifact; live full precision stack benchmark (SCENARIO-BENCH-007/008/009) |
| REQ-BENCH-004 | Not Started | Implemented (Exp 341) | HumanEvalResult + compute_pass_at_1 + compute_pass_at_1_after_repair + build_humaneval_artifact; live HumanEval code verification benchmark (SCENARIO-BENCH-010/011) |
| REQ-BENCH-005 | Not Started | Implemented (Exp 353) | SmokeTestResult + run_smoke_test + build_smoke_test_artifact; live GPU smoke test gate with CI-skip path and RuntimeError on simulated fallback (SCENARIO-BENCH-012/013) |
| REQ-BENCH-006 | Not Started | Implemented (Exp 354) | AdversarialGSMQuestion + build_adversarial_questions + AdversarialBenchmarkResult + compute_adversarial_results + build_adversarial_artifact; adversarial GSM8K harness with 20-distractor pool and CI-safe synthetic mode (SCENARIO-BENCH-014/015/016) |
| REQ-BENCH-007 | Not Started | Implemented (Exp 354) | robustness_invariant_holds field in build_adversarial_artifact; invariant is True when adversarial_accuracy >= standard_accuracy - 0.05 (SCENARIO-BENCH-014/016) |
| REQ-INFRA-001 | N/A | Implemented | run_experiment_with_timeout.sh + timeout_wrapper_exists() tests (SCENARIO-INFRA-001, Exp 325) |
| REQ-INFRA-002 | N/A | Implemented | generate_test_stub() idempotency + ast.parse tests (SCENARIO-INFRA-002/003, Exp 325) |
| REQ-INFRA-003 | N/A | Implemented | DualGPUMonitor zombie detection + CI-safe fallback tests (SCENARIO-INFRA-004/006, Exp 326) |
| REQ-INFRA-004 | N/A | Implemented | check_dual_gpu_health() + setup_gpu() integration tests (SCENARIO-INFRA-005/006, Exp 326) |
| REQ-INFRA-005 | N/A | Implemented | DependencyAudit + extract_required_files + check_dependencies + CLI tests (SCENARIO-INFRA-007/008, Exp 327) |
| REQ-RETRO-001 | N/A | Implemented | operational_retro_2026_04_29.json schema + action items + carry_over tests (SCENARIO-RETRO-001/002, Exp 319) |
| REQ-RETRO-002 | N/A | Implemented | operational_retro_2026_04_29.json v2 + gpu_utilization_analysis tests (SCENARIO-RETRO-003/004, Exp 319) |
| REQ-RETRO-003 | N/A | Implemented | operational_retro_2026_05_06.json + retro_001_resolved + actual_speedup_pct tests (SCENARIO-RETRO-005/006, Exp 337) |
| REQ-INFRA-006 | N/A | Implemented | HostPrereqRegistry + ops/host-prereqs.md + check_prereqs() tests (SCENARIO-INFRA-009/010, Exp 338) |
| REQ-INFRA-007 | N/A | Implemented | DualGPU auto-assignment in setup_gpu() + dual_gpu_auto_assigned key tests (SCENARIO-INFRA-011, Exp 338) |
| REQ-INFRA-008 | N/A | Implemented | session_startup.sh + session_startup.py + parse_session_startup_output + run_session_startup tests (SCENARIO-INFRA-012/013, Exp 339) |
| REQ-INFRA-014 | N/A | Implemented | LiveGPUDiagnostic + diagnose_live_gpu() + setup_gpu() RuntimeError on CARNOT_FORCE_LIVE=1 failure (SCENARIO-INFRA-014/015, Exp 352) |
| REQ-INFRA-015 | N/A | Implemented | ConductorEnvFix + build_conductor_env_fix + verify_env_script_exports + scripts/conductor_gpu_env.sh (SCENARIO-INFRA-016/017, Exp 365). Closes RETRO-012. |
| REQ-INFRA-016 | N/A | Implemented | RetroJSONEnforcer + check_result_json_exists + audit_missing_jsons (SCENARIO-INFRA-018, Exp 365). Closes RETRO-014. |
| REQ-INFRA-017 | N/A | Implemented | session_startup.sh + build_session_startup_script + check_session_startup_exists (SCENARIO-INFRA-019/020/021, Exp 377). Part of RETRO-015 fix. |
| REQ-INFRA-018 | N/A | Implemented | LiveGPUGate + check_env_var + check_gpu_live + require_live + require_live_or_blocked + verify_subprocess_env_propagation (SCENARIO-INFRA-019/020/021, Exp 377). Closes RETRO-015. |
| REQ-VERIFY-086 | Not Started | Implemented | SinkProbe attention-sink pre-filter + SinkConcentration + SinkProbeResult + compute_sink_concentration (SCENARIO-VERIFY-113/114/115, Exp 348) |
| REQ-VERIFY-087 | Not Started | Implemented | SinkProbe threshold configuration + benchmark() skip/FNR/TNR reporting (SCENARIO-VERIFY-113/114/115, Exp 348) |
| REQ-VERIFY-088 | Not Started | Implemented | Three-tier pipeline benchmark — ThreeTierPipeline + ThreeTierPipelineResult + verify/benchmark + build_three_tier_artifact (SCENARIO-VERIFY-116/117, Exp 360); live GPU benchmark on real attention matrices (SCENARIO-VERIFY-118/119, Exp 373) |
| REQ-AGENT-001 | Not Started | Implemented | SAVeR multi-turn constraint state propagation — ConstraintState dataclass carrying accumulated_facts, active_constraints, facts_established, model_id across steps (SCENARIO-AGENT-001/002/003, Exp 362) |
| REQ-AGENT-002 | Not Started | Implemented | SAVeR commit-gate — propose_step() blocks action when constraint violations survive max_repair_attempts; committed=False prevents fact accumulation (SCENARIO-AGENT-001/002/003, Exp 362) |
