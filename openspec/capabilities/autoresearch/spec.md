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

### REQ-LEARN-010: Constraint Addition from CaseMemory Patterns

When CaseMemory has accumulated error patterns for a violation family with support ≥ 3, the
system shall generate and add new IsingConstraint types to the active extractor set:
- `extract_patterns(case_memory, min_support=3)` groups CaseMemory entries by violation_family,
  computes observed_precision = improved_repairs / total_flagged per family, returns ConstraintPattern
  objects for families meeting the support threshold.
- `generate_arithmetic_constraint(pattern)` maps families to targeted constraint types:
  "carry_error" → carry-propagation check; "sign_error" → sign-consistency check;
  "magnitude_error" → order-of-magnitude check; unknown families → generic learned check.
- `add_to_extractor(extractor, constraint)` appends the generated constraint to the extractor's
  `_dynamic_constraints` list without removing or modifying any existing constraints.
- `constraint_already_exists(extractor, constraint_id)` prevents duplicate insertion.

### REQ-LEARN-011: Soundness Bound for Constraint Addition

Constraint addition shall be gated by a soundness bound derived from arXiv 2603.03538
(CoT Verifier Online Learnability):
- `soundness_filter(patterns, min_precision=0.85)` returns only patterns where
  observed_precision ≥ min_precision (85% of flagged cases were confirmed real errors).
- Patterns below min_precision are explicitly logged as "rejected_soundness" in the
  `ConstraintGenerator.generation_log`, not silently dropped.
- The 0.85 threshold ensures that generated constraints have high precision — they rarely
  flag correct answers — preserving the soundness of downstream verification.

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

### SCENARIO-LEARN-015: extract_patterns Groups CaseMemory by Violation Family

**Given** a CaseMemory with 5 entries for violation_family "carry_error" (3 "improved", 2 "unchanged_failure")
**When** `extract_patterns(case_memory, min_support=3)` is called
**Then** one ConstraintPattern is returned for family "carry_error"
**And** observed_precision = 3/5 = 0.60
**And** support_count = 5
**And** families below min_support=3 are absent from the result

### SCENARIO-LEARN-016: soundness_filter Rejects Low-Precision Patterns

**Given** two ConstraintPatterns: one with observed_precision=0.90 and one with 0.60
**When** `soundness_filter(patterns, min_precision=0.85)` is called
**Then** only the pattern with precision 0.90 is returned
**And** the rejected pattern is NOT silently dropped — it must be tracked by the caller

### SCENARIO-LEARN-017: generate_arithmetic_constraint Maps Families to Constraint Types

**Given** ConstraintPatterns for families "carry_error", "sign_error", "magnitude_error"
**When** `generate_arithmetic_constraint(pattern)` is called for each
**Then** "carry_error" yields a LearnedConstraint with constraint_id "learned:carry_error"
**And** "sign_error" yields constraint_id "learned:sign_error"
**And** "magnitude_error" yields constraint_id "learned:magnitude_error"
**And** each LearnedConstraint has a human-readable description of what it checks

### SCENARIO-LEARN-018: ConstraintGenerator Orchestrates and Logs All Outcomes

**Given** a CaseMemory with high-precision carry_error (precision=0.92), low-precision sign_error (0.60),
and an already-existing magnitude_error constraint in the extractor
**When** `ConstraintGenerator().generate_from_memory(case_memory, extractor)` is called
**Then** generation_log["carry_check:carry_error"] == "added"
**And** generation_log["sign_consistency:sign_error"] == "rejected_soundness"
**And** generation_log["magnitude_check:magnitude_error"] == "already_exists"
**And** only carry_error is added to extractor._dynamic_constraints

### REQ-LEARN-030: FOVER-Style Z3 Step Annotation

The system shall provide a `FOVERAnnotator` that implements FoVer-style annotation
(arXiv 2505.15960) of chain-of-thought reasoning steps with Z3-verified correctness labels:

- Parse a CoT response into discrete steps using `parse_cot_into_steps(response) -> list[FOVERCoTStep]`.
  Split on numbered steps ("1.", "2.") or "Step N:" patterns.
- For each step, extract any claimed arithmetic equation using the same `_INLINE_EQ` pattern as
  CRANEExtractionGate.
- Annotate each step via `annotate_step_with_z3(step) -> FOVERCoTStep`:
  - If no equation: `z3_label='not_verifiable'`, `z3_confidence=0.0`
  - If equation present: run `_exec_z3_snippet` on an inline Z3 assertion.
  - `z3_label='correct'` when Z3 returns 'sat'; `z3_label='incorrect'` when Z3 returns 'unsat'.
  - `z3_confidence=1.0` for a complete equation (all three operands present); `0.5` otherwise.
- `FOVERAnnotator(z3_timeout_seconds=5)` wraps the above with corpus-level batching and
  training-pair output.

Rationale: Z3 annotations are deterministic, scalable, and formally correct — unlike
human labels which require expensive annotation at scale. This is the training signal
that was missing for FR-11 (EORM/JEPA retrains on synthetic-only data).

Spec refs: arXiv 2505.15960 (FoVer), arXiv 2601.17223 (VPRM)

### REQ-LEARN-032: EORM Retrain on FOVER-Labeled Real Pairs

EORM retrained on FOVER-labeled (correct_step, incorrect_step) contrastive pairs from
real LLM inference shall achieve AUC-ROC > 0.5 on the held-out 20% test split.

- `load_fover_pairs(path) -> list[dict]`: load FOVER labeled steps from JSON; filter by
  `label in ('correct', 'incorrect')` and `confidence >= 0.3`.
- `fover_pairs_to_contrastive(pairs) -> list[tuple[jnp.ndarray, jnp.ndarray]]`: convert
  co-occurring (correct, incorrect) pairs on the same question into (positive, negative)
  contrastive tensors for EORM training.
- If `n_real_pairs < 10`, fall back to synthetic data; set `honest_verdict='synthetic_only'`.
- `honest_verdict='real_data_improvement'` requires both `after_auc > before_auc` AND
  `n_real_pairs >= 10`.
- Retro-024 is closed when `honest_verdict='real_data_improvement'`.

Spec refs: arXiv 2505.14999 (EORM), RETRO-024

### REQ-LEARN-033: JEPA Predictor Retrain on FOVER Pairs

The JEPA predictor shall be retrained on the same FOVER-labeled pairs, treating each
labeled step as a (partial_step_prefix, violation_occurred) pair:

- `label='incorrect'` maps to `has_violation=True`
- `label='correct'` maps to `has_violation=False`
- Use existing `JEPARetrainer.train_epoch()` and `evaluate_auc_roc()`.
- Save the retrained model to `jepa_431_real.safetensors`.

### REQ-LEARN-031: FOVER Training Pair Export

The annotated (step, label) pairs shall be written to `results/fover_labeled_steps.json`
for use as training targets for EORM (Exp 431). The export format:

- Schema: `carnot.fover_labels.v1`
- Each pair: `{question_id, step_text, label, confidence}`
- Filter: only include steps where `z3_label in ('correct', 'incorrect')` AND
  `z3_confidence >= 0.3`. Steps labeled `not_verifiable` are excluded from training
  pairs because they provide no learning signal.
- `FOVERAnnotator.to_training_pairs(annotated) -> list[dict]` performs this filtering.

## Scenarios

### SCENARIO-LEARN-054: Numbered Step Parsing

**Given** a CoT response with text "1. First step. 2. Second step. 3. Third step."
**When** `parse_cot_into_steps(response)` is called
**Then** exactly 3 `FOVERCoTStep` objects are returned
**And** each step has the correct `step_idx` (0, 1, 2)

### SCENARIO-LEARN-055: Z3 Correct Equation Label

**Given** a `FOVERCoTStep` with `claimed_equation="2 + 3 = 5"`
**When** `annotate_step_with_z3(step)` is called
**Then** the returned step has `z3_label='correct'`
**And** `z3_confidence >= 0.3`

### SCENARIO-LEARN-056: Z3 Incorrect Equation Label

**Given** a `FOVERCoTStep` with `claimed_equation="2 + 3 = 6"`
**When** `annotate_step_with_z3(step)` is called
**Then** the returned step has `z3_label='incorrect'`

### SCENARIO-LEARN-057: FOVER Pair Loading with Confidence Filter

**Given** a FOVER-labeled JSON file with 100 pairs (80 with confidence >= 0.3, 20 with confidence < 0.3)
**When** `load_fover_pairs(path)` is called
**Then** exactly 80 pairs are returned (confidence filter applied)
**And** all returned pairs have `label in ('correct', 'incorrect')`

### SCENARIO-LEARN-058: Contrastive Tensor Conversion

**Given** a list of FOVER pairs including co-occurring correct and incorrect steps for the same question
**When** `fover_pairs_to_contrastive(pairs)` is called
**Then** a list of (positive, negative) tensor tuples is returned
**And** each tuple contains two `jnp.ndarray` vectors (one per step text)
**And** pairs from different questions are NOT cross-matched

### SCENARIO-LEARN-059: Retrain Verdict Computation

**Given** `before_auc=0.5`, `after_auc=0.62`, `n_real_pairs=25`
**When** `compute_retrain_verdict(0.5, 0.62, 25)` is called
**Then** the result is `'real_data_improvement'`

**Given** `n_real_pairs=5` (below threshold)
**When** `compute_retrain_verdict(0.5, 0.8, 5)` is called
**Then** the result is `'synthetic_only'`

**Given** `before_auc=0.62`, `after_auc=0.55`, `n_real_pairs=25`
**When** `compute_retrain_verdict(0.62, 0.55, 25)` is called
**Then** the result is `'real_data_no_improvement'`

### REQ-LEARN-034: JitRL Memory Validated on Real Data Produces Measurable FP Reduction

The system shall validate JitRL constraint memory on real GSM8K model output data
(from Exp 427) and produce a measurable false-positive reduction percentage.

The validation shall:
- Load real violation records from an Exp 427 result file (or fall back to synthetic
  if unavailable).
- Split records into a warm-up set (first 50) and a validation set (last 50).
- Feed warm-up records into ``JitRLConstraintMemory.record()`` to build per-domain
  threshold history.
- Compare fp_rate before JitRL (no threshold adaptation) vs after (with adapted
  thresholds applied to the validation set).
- Compute ``fp_reduction_pct = (before_fp - after_fp) / before_fp * 100`` (0 if
  before_fp == 0).
- Produce an honest verdict:
  - ``'live_fp_reduction'``   — source='live' AND fp_reduction_pct > 0
  - ``'live_no_reduction'``   — source='live' AND fp_reduction_pct <= 0
  - ``'synthetic_fallback'``  — source='synthetic' (Exp 427 unavailable)

Spec: Exp 432

### SCENARIO-LEARN-060: JitRL Record Raises Threshold for FP Domain

**Given** a ``JitRLConstraintMemory`` with base_threshold=0.5, lr=0.02
**When** ``record('rate_problems', violation_energy=0.6, was_fp=True)`` is called
**Then** ``threshold('rate_problems')`` returns 0.52
**And** the history list has length 1

### SCENARIO-LEARN-061: JitRL Validation Artifact Schema

**Given** ``before_fp=0.3``, ``after_fp=0.2``, ``n_questions=50``, ``source='live'``
**When** ``build_jitrl_validation_artifact(0.3, 0.2, 50, 'live')`` is called
**Then** the result has ``schema='carnot.jitrl_validation.v1'``
**And** ``fp_reduction_pct`` is approximately 33.33
**And** ``honest_verdict`` is ``'live_fp_reduction'``

**Given** ``before_fp=0.0``, ``after_fp=0.0``, ``n_questions=50``, ``source='synthetic'``
**When** ``build_jitrl_validation_artifact(0.0, 0.0, 50, 'synthetic')`` is called
**Then** ``fp_reduction_pct`` is 0
**And** ``honest_verdict`` is ``'synthetic_fallback'``

### REQ-LEARN-035: FOVER Annotator Runs on Live GPU CoT Data

The system shall run FOVERAnnotator on live GPU CoT responses (from Exp 439) and produce
labeled training pairs sufficient for downstream EORM/JEPA training:

- Load ``results/experiment_439_live_cot.json``; if present and the companion
  ``results/experiment_439_live_precision_micro.json`` confirms ``inference_mode='live_gpu'``,
  treat all cot_responses as real data (``source='live'``).
- If the live CoT file is absent or the companion does not confirm live_gpu, fall back to
  generating 100 synthetic GSM8K-style CoT responses (``source='synthetic'``).
- Run ``FOVERAnnotator.annotate_corpus(responses)`` then ``to_training_pairs(annotated, responses)``.
- Write labeled pairs to ``results/fover_labeled_steps_live.json``
  (separate file from Exp 430's ``fover_labeled_steps.json``).
- Honest verdict:

  - ``'real_data_labeled'``      — source='live' AND n_labeled >= 20
  - ``'real_data_insufficient'`` — source='live' AND n_labeled < 20
  - ``'synthetic_fallback'``     — source='synthetic'

Spec: Exp 442

### SCENARIO-LEARN-062: Live CoT Annotation Produces Labeled Pairs

**Given** ``results/experiment_439_live_cot.json`` contains 300 cot_responses
**And** the companion confirms ``inference_mode='live_gpu'``
**When** ``build_live_fover_artifact(result)`` is called with source='live' and n_labeled>=20
**Then** ``honest_verdict`` is ``'real_data_labeled'``
**And** ``schema`` is ``'carnot.fover_live.v1'``
**And** ``labeling_rate`` equals ``n_labeled / n_steps_found``

### SCENARIO-LEARN-063: Synthetic Fallback When Live Data Absent

**Given** ``results/experiment_439_live_cot.json`` is absent or does not confirm live_gpu
**When** ``build_live_fover_artifact(result)`` is called with source='synthetic'
**Then** ``honest_verdict`` is ``'synthetic_fallback'``
**And** ``schema`` is ``'carnot.fover_live.v1'``

### REQ-LEARN-036: EORM Retrained on Real FOVER Pairs Achieves AUC > 0.50

The system shall retrain EORM on ≥20 real FOVER-labeled (step_text, label) pairs sourced from
live GPU CoT annotation (Exp 442) and evaluate on a held-out 20% test split:

- Load ``results/fover_labeled_steps_live.json`` (Exp 442 output).
- If n_labeled < 20: use synthetic fallback (``honest_verdict='real_data_insufficient'``).
- Evaluate EORM ``before_auc`` on held-out 20% split.
- Retrain EORM for 150 epochs on contrastive triples derived from real pairs.
- Also retrain JEPA predictor on (step_prefix, violation_flag) pairs.
- Evaluate ``after_auc`` on same held-out split.
- Use ``compute_retrain_verdict_v2(before_auc, after_auc, n_real_pairs, source)`` to produce verdict:

  - ``'real_data_improvement'``   — source='live' AND after_auc > before_auc AND n_real_pairs >= 20
  - ``'real_data_no_improvement'``— source='live' AND after_auc <= before_auc AND n_real_pairs >= 20
  - ``'real_data_insufficient'``  — source='live' AND n_real_pairs < 20
  - ``'synthetic_only'``          — source='synthetic'

- Save models: ``results/eorm_443_live.safetensors``, ``results/jepa_443_live.safetensors``.
- Set ``retro_024_closed=True`` iff ``honest_verdict='real_data_improvement'``.
- Emit artifact ``schema='carnot.eorm_jepa_retrain.v3'``.

Spec: Exp 443

### SCENARIO-LEARN-064: EORM Retrain on Real Pairs Closes RETRO-024

**Given** ``results/fover_labeled_steps_live.json`` contains ≥20 labeled pairs (n_correct > 0, n_incorrect > 0)
**When** ``compute_retrain_verdict_v2(before_auc, after_auc, n_real_pairs=57, source='live')`` is called
**And** after_auc > before_auc
**Then** verdict is ``'real_data_improvement'``
**And** ``retro_024_closed=True`` in the Exp 443 artifact

### SCENARIO-LEARN-065: Synthetic Fallback When Pairs Insufficient

**Given** n_real_pairs < 20 for Exp 443
**When** ``compute_retrain_verdict_v2(before_auc, after_auc, n_real_pairs=5, source='live')`` is called
**Then** verdict is ``'real_data_insufficient'``
**And** ``retro_024_closed=False`` in the artifact

### REQ-LEARN-037: CoTPairQualityFilter Rejects Low-Quality Training Pairs

The system shall provide a ``CoTPairQualityFilter`` that rejects CoT training pairs
where ``arithmetic_coverage < 0.3`` OR ``label_confidence < 0.7``.

This filter addresses RETRO-040: Exp 472 AUC regressed from 0.667 to 0.400 because
54 real CoT pairs included partially-verifiable steps labeled with low confidence.
Quality gating before training prevents noisy supervision from degrading the JEPA
energy landscape.

Spec: Exp 477

### REQ-LEARN-038: JEPAQualityAugmentor Generates EBM-Guided Synthetic Pairs

The system shall provide a ``JEPAQualityAugmentor`` that generates synthetic CoT
training pairs by sampling from the Ising model's violation distribution (high-energy
spin configurations = incorrect, low-energy = correct).

This is NOT random synthetic generation. The EBM's own energy landscape provides a
principled source of violations: the Ising coupling matrix encodes learned constraint
interactions, so sampling near energy maxima produces examples representative of the
actual failure modes the JEPA must predict.

Spec: Exp 477, FR-11 JEPA, RETRO-040

### REQ-LEARN-039: JEPA Retrained on Quality-Gated Corpus Targets AUC > 0.700

The system shall retrain the JEPA model on a quality-gated corpus (filtered real pairs
+ EBM-guided synthetic pairs) and target AUC > 0.700 on a held-out 20% test set.

Regression baseline is AUC = 0.400 (Exp 472). Recovery threshold is AUC > 0.571
(Exp 443 level). RETRO-040 is closed when AUC > 0.600.

Spec: Exp 477, FR-11 JEPA, RETRO-040

### SCENARIO-LEARN-066: CoTPairQualityFilter Rejects Low-Coverage Pairs

**Given** a CoT pair with arithmetic_coverage=0.2 and label_confidence=0.8
**When** ``CoTPairQualityFilter(min_coverage=0.3, min_confidence=0.7).filter([pair])``
**Then** the returned list is empty (pair rejected for coverage < 0.3)

### SCENARIO-LEARN-067: JEPAQualityAugmentor Violation Pairs Are Labeled Incorrect

**Given** a small Ising model (input_dim=8)
**When** ``JEPAQualityAugmentor(ising_model, n_samples=10).generate_violation_pairs()``
**Then** all returned pairs have ``correct=False``
**And** the energy of their spin configurations is above the model's mean energy

### SCENARIO-LEARN-068: JEPARetrainV2Result Regression Recovery Detection

**Given** ``before_auc=0.400``, ``after_auc=0.620``
**When** ``JEPARetrainV2Result(n_pairs_raw=57, n_pairs_filtered=30, n_synthetic=170, before_auc=0.400, after_auc=0.620)``
**Then** ``regression_recovered=True`` (after_auc > 0.571)
**And** ``retro_040_closed=True`` (after_auc > 0.600)
**And** ``target_met=False`` (after_auc <= 0.700)

### REQ-LEARN-040: JEPACurriculumTrainer Stage 1 Trains on High-Confidence Pairs

``JEPACurriculumTrainer.train(pairs)`` stage 1 shall train the EORM model exclusively on
pairs where ``label_confidence >= high_conf_threshold`` (default 0.85), establishing a
stable baseline before full-distribution exposure.  This prevents the majority-class
collapse that caused the 0.281 AUC regression in Exp 477.

Spec: Exp 492, RETRO-040, arXiv 2509.14252

### REQ-LEARN-041: JEPACurriculumTrainer Stage 2 Fine-Tunes on All Pairs Unfiltered

``JEPACurriculumTrainer.train(pairs)`` stage 2 shall fine-tune the model (warm-started from
stage 1) on ALL available pairs without any confidence gate.  The information loss from
quality-gate filtering is the root cause of the regression — stage 2 recovers that lost
information after stage 1 provides a stable anchor.

Spec: Exp 492, RETRO-040

### REQ-LEARN-042: JEPACurriculumTrainer Stage 3 Augments to n_total >= 200 and Validates AUC

``JEPACurriculumTrainer.train(pairs)`` stage 3 shall augment the training corpus with
EBM-guided synthetic pairs (via ``JEPAQualityAugmentor``) until the total corpus size
reaches at least 200 pairs, then train for ``n_stage3_epochs`` epochs.
``JEPACurriculumTrainer.get_final_auc(held_out_pairs)`` shall return AUC on any held-out set.
``JEPARetrainV3Result.target_met`` shall be True iff ``after_auc > 0.600``.

Spec: Exp 492, RETRO-040

### SCENARIO-LEARN-069: Stage 1 Excludes Low-Confidence Pairs

**Given** a corpus with pairs of varying ``label_confidence`` (some above 0.85, some below)
**When** ``JEPACurriculumTrainer(high_conf_threshold=0.85).train(pairs)`` stage 1
**Then** stage 1 ``n_pairs`` counts only pairs with ``label_confidence >= 0.85``
**And** no pair with ``label_confidence < 0.85`` enters stage 1 training

### SCENARIO-LEARN-070: JEPARetrainV3Result.regression_recovered When after_auc > 0.400

**Given** ``before_auc=0.281``, ``after_auc=0.450``
**When** ``JEPARetrainV3Result(n_pairs_raw=57, curriculum_stages=[...], before_auc=0.281, after_auc=0.450)``
**Then** ``regression_recovered=True`` (after_auc > 0.400, reversing the quality-gate collapse)
**And** ``target_met=False`` (after_auc <= 0.600, not yet at closure bar)

### SCENARIO-RETRO-032: Milestone 2026.04.32 Retrospective Complete

**Given** result JSONs for Exps 425-435 (partial: some results are scaffolding_only or absent)
**When** ``MilestoneRetro2026_04_32`` is computed from available result files
**Then** the artifact has ``schema='carnot.operational_retro.v6'``
**And** ``milestone='2026.04.32'``
**And** ``conductor_timeout_implemented`` reflects whether experiment_watchdog.py is present
**And** ``gpu1_zombie_fixed`` reflects Exp 426 retro_025_status
**And** ``live_numbers_confirmed`` is True only if at least one live benchmark produced a signed improvement
**And** ``fr11_relay_confirmed`` reflects Exp 431 retro_024_closed
**And** all boolean fields are derived exclusively from result JSON files, never asserted without provenance

### SCENARIO-RETRO-033: Milestone 2026.04.33 Retrospective Complete

**Given** result JSONs for Exps 437-448 (partial: Exp 446 absent, Exp 444 timed out)
**When** ``MilestoneRetro2026_04_33`` is computed from available result files
**Then** the artifact has ``schema='carnot.operational_retro.v7'``
**And** ``milestone='2026.04.33'``
**And** ``retro_026_resolved`` reflects Exp 437 retro_026_resolved flag
**And** ``retro_025_resolved`` reflects whether Exp 438 fix_applied is True
**And** ``live_precision_result`` is the honest_verdict from Exp 439 (first live GPU precision run)
**And** ``live_humaneval_result`` is the honest_verdict from Exp 440 (first live GPU HumanEval run)
**And** ``live_adversarial_result`` is the honest_verdict from Exp 441 (first live GPU adversarial run)
**And** ``fr11_relay_confirmed`` reflects Exp 443 retro_024_closed
**And** ``think_probe_viable`` is False when Exp 444 timed_out=True
**And** ``kaem_faster`` is True only when Exp 447 mean_speedup > 5
**And** ``cross_session_improvement`` is True only when Exp 448 honest_verdict indicates improvement
**And** all boolean fields are derived exclusively from result JSON files, never asserted without provenance
**And** the headline confirms live GPU benchmark numbers were obtained for the first time after 7 consecutive scaffolding-only milestones

### REQ-SELFLEARN-019: PPSConstraintLearner Maintains Partition Isolation on Naturally-Interleaved Real Violations

The system shall maintain ``partition_isolation_score > 0.7`` on naturally-interleaved
real violation sequences from live CoT data.  "Naturally interleaved" means steps are
processed in their original occurrence order — NOT sorted by domain.  The threshold is
0.7 (vs 0.8 on synthetic data) to account for real-data noise in domain label assignment.

RETRO-043: Exp 470 used independent domain batches (easy mode).  This requirement
closes RETRO-043 by validating under the harder interleaved condition.

Spec: Exp 485, FR-11, RETRO-043, arXiv 2512.15658

### REQ-SELFLEARN-020: Interleaved Validation Uses FOVERAnnotator-Labeled Steps in Natural Order

The system shall use FOVERAnnotator-labeled steps in the order they occurred in the live
CoT chain (not sorted by domain) when validating PPSConstraintLearner.  Training batches
drawn from the natural-order sequence will contain mixed domains, stressing the partition
walls.

Spec: Exp 485, FR-11, RETRO-043

### SCENARIO-SELFLEARN-019: InterleavedViolationSequence Alternating Steps Have Rate 1.0

**Given** a sequence of steps alternating between arithmetic and code domains
**When** ``InterleavedViolationSequence(steps)``
**Then** ``interleaving_rate == 1.0``
**And** every adjacent pair in ``domain_sequence`` has a different domain label

### SCENARIO-SELFLEARN-020: PPSEBMRealValidationResult Isolation Maintained at 0.75

**Given** ``isolation_score_after=0.75``, ``synthetic_isolation_score=1.0``
**When** ``PPSEBMRealValidationResult(n_steps=57, interleaving_rate=0.4, isolation_score_before=1.0, isolation_score_after=0.75, fp_rate_real=0.05)``
**Then** ``isolation_maintained=True`` (0.75 > 0.7)
**And** ``better_than_synthetic=False`` (0.75 < 1.0)

### REQ-DIAG-001: JEPA Corpus Diagnostic Analysis

``JEPACurriculumDiagnostic.analyze_corpus(quality_filter)`` shall compute the following
metrics for the quality-gated corpus:
- ``label_imbalance_ratio``: ratio of correct to incorrect steps in the filtered corpus
- ``filter_rate``: fraction of raw pairs that passed the quality filter
- ``n_pairs_remaining`` (``n_pairs_filtered``): count of pairs surviving the filter

These metrics together diagnose the RETRO-040 regression: excessive filtering combined
with label imbalance causes JEPA majority-class collapse and AUC below 0.5.

Spec: Exp 491, RETRO-040

### REQ-DIAG-002: JEPA Curriculum Regime Simulation

``JEPACurriculumDiagnostic.simulate_regime(regime, n_epochs=100)`` shall train a JEPA
(EORM) model on pairs selected/ordered by the given regime for n_epochs passes, then
return AUC on a held-out 20% set.

Supported regimes:
- ``'all_pairs'``: train on all pairs — baseline
- ``'quality_gated'``: train only on pairs with label_confidence >= 0.7 — reproduces Exp 477
- ``'curriculum_high_to_low'``: sort by decreasing label_confidence — curriculum learning
- ``'random_50pct'``: random 50% sample — isolates size effect from quality effect

Return value is a float in [0, 1]. AUC < 0.5 indicates the model learned an inverted signal.

Spec: Exp 491, RETRO-040

### SCENARIO-DIAG-001: CorpusAnalysis is_imbalanced for Heavily Correct Corpus

**Given** ``n_correct=80``, ``n_incorrect=5``, ``label_imbalance_ratio=16.0``
**When** ``CorpusAnalysis.is_imbalanced``
**Then** ``is_imbalanced=True`` (16.0 > 3.0)

### SCENARIO-DIAG-002: CorpusAnalysis diagnosis='imbalance' When Filtered Corpus is Imbalanced

**Given** ``is_imbalanced=True`` and ``n_pairs_filtered >= 5``
**When** ``CorpusAnalysis.diagnosis``
**Then** ``diagnosis='imbalance'``

This is the Exp 477 root cause: the quality gate removed 73% of pairs, leaving a corpus
with many more correct than incorrect steps. The JEPA model collapsed to majority-class
prediction (predict all correct), yielding AUC = 0.281 (actively wrong, not just random).

### SCENARIO-DIAG-003: simulate_regime Returns Float in [0, 1]

**Given** a ``JEPACurriculumDiagnostic`` with labeled pairs
**When** ``simulate_regime(regime)`` for any valid regime string
**Then** return value is a ``float`` in ``[0, 1]``

### REQ-SELFLEARN-021: SuRePriorityReplay Ranks Violations by EBM Energy Surprise

SuRePriorityReplay (arXiv 2511.22367) shall rank constraint violations in the replay
buffer by EBM energy surprise — energy relative to the per-domain running mean — and
select the top-k highest-surprise violations for replay at each update step.

Uniform random replay wastes replay budget on easy examples the model already handles
well. Surprise-driven selection (SuRe) prioritizes examples near the knowledge
boundary — where catastrophic forgetting is most likely. arXiv 2511.22367 shows +5%
accuracy improvement in continual learning from NLL-based surprise prioritization.

In the EBM domain, high Ising energy = multiple competing low-energy configurations =
the constraint is ambiguous for the current model = high cognitive surprise. This is a
direct translation of NLL-based surprise to the EBM domain.

Spec: Exp 497, FR-11, arXiv 2511.22367

### REQ-SELFLEARN-022: SuRePriorityReplay Maintains Partition Isolation > 0.7

SuRePriorityReplay combined with PPSConstraintLearner shall maintain
partition_isolation_score > 0.7 on interleaved real violation sequences from
fover_labeled_steps_live.json (57 steps). The isolation score with SuRe replay
shall be compared against uniform random replay as a baseline.

Spec: Exp 497, FR-11, arXiv 2511.22367

### SCENARIO-SELFLEARN-021: SuRePriorityReplay Returns Highest-Surprise Items First

**Given** a SuRePriorityReplay with violations at energies [0.1, 0.9, 0.5, 0.8]
**When** domain_mean_energy = 0.4 and get_replay_batch(n=2) is called
**Then** returns the two violations with energies 0.9 and 0.8 (highest surprise)
**And** surprise_scores are sorted descending

### SCENARIO-SELFLEARN-022: SuReReplayResult.sure_better True When SuRe Improves Isolation

**Given** ``isolation_score_uniform=0.72``, ``isolation_score_sure=0.85``
**When** ``SuReReplayResult(n_violations_processed=57, n_replay_items=17, isolation_score_uniform=0.72, isolation_score_sure=0.85)``
**Then** ``sure_better=True`` (0.85 > 0.72)
**And** ``isolation_improvement=0.13`` (0.85 - 0.72)

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
| REQ-LEARN-010 | N/A | Implemented | 22 Python |
| REQ-LEARN-011 | N/A | Implemented | 22 Python |
| REQ-LEARN-030 | N/A | Implemented | 10+ Python |
| REQ-LEARN-031 | N/A | Implemented | 10+ Python |
| REQ-LEARN-032 | N/A | Implemented | 10+ Python |
| REQ-LEARN-033 | N/A | Implemented | 10+ Python |
| REQ-LEARN-034 | N/A | Implemented | 10+ Python |
| REQ-LEARN-035 | N/A | Implemented | 10+ Python |
| REQ-LEARN-036 | N/A | Implemented | 10+ Python |
| REQ-LEARN-037 | N/A | Implemented | Python |
| REQ-LEARN-038 | N/A | Implemented | Python |
| REQ-LEARN-039 | N/A | Implemented | Python |
| REQ-LEARN-040 | N/A | Implemented | Python |
| REQ-LEARN-041 | N/A | Implemented | Python |
| REQ-LEARN-042 | N/A | Implemented | Python |
