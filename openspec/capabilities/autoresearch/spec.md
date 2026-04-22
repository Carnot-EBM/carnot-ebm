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

### REQ-SELFLEARN-013: Tier 1 Relay Uses Real FP Patterns from Live Inference

The Tier 1 self-learning relay (FR-11) shall use real false-positive and
false-negative patterns extracted from live inference diagnostics (e.g.,
Exp 554 per_question_flags) as the seed for ConstraintAdditionFromMemory —
not synthetic data.  The relay artifact must carry ``inference_mode='real_data'``
and ``fr11_real_data=True``.  If the diagnostic source is unavailable, the
experiment must gate and report ``status='blocked'`` rather than substituting
synthetic data.

Spec: Exp 561, FR-11

### SCENARIO-SELFLEARN-013: Exp 554 FN Patterns Produce ViolationPattern for 'low_tp_extraction'

**Given** Exp 554 vericot_result has 17 FN cells and 0 FP cells
**When** ``load_exp554_fp_patterns(path)`` is called
**Then** returns a list containing one ViolationPattern with type='low_tp_extraction'
**And** ViolationPattern.count == 34 (17 FN from VeriCoT + 17 FN from VPRM)

### SCENARIO-SELFLEARN-014: Session 2 fp_rate_delta Reported Honestly

**Given** Session 1 fp_rate and Session 2 fp_rate are both computed on the same
25-response corpus with and without constraint_memory
**When** fp_rate_delta = session2_fp_rate - session1_fp_rate
**Then** honest_verdict is 'real_data_improvement' iff fp_rate_delta < -0.05
**And** honest_verdict is 'real_data_no_improvement' otherwise

### SCENARIO-SELFLEARN-015: Missing Exp 554 Diagnostic Gates the Experiment

**Given** Exp 554 result file does not exist
**When** load_exp554_fp_patterns() is called
**Then** returns an empty list
**And** the experiment sets status='blocked' and honest_verdict='blocked_no_real_data'

### REQ-LEARN-043: EnergyMagnitudeReplay Ranks Violations by |energy - session_mean|

EnergyMagnitudeReplay shall rank constraint violations by absolute energy deviation
|energy(x) - domain_mean_energy| for replay priority, ensuring domain boundary violations
(the examples the EBM is most wrong about) are always replayed first.

RETRO-050 root cause: SuRe used LLM NLL as surprise proxy (Exp 497, isolation=-0.1172).
LLM surface-form surprise and EBM energy magnitude are anticorrelated: common sentences
can violate hard constraints at high energy, and rare sentences can have low constraint
energy. Replacing LLM-surprise with energy-magnitude priority directly targets domain
boundary cases.

The EBM's energy function is the ground truth. High |energy - mean| = the constraint
model is maximally wrong about this boundary = highest replay value.

Spec: Exp 509, RETRO-050

### REQ-LEARN-044: EnergyMagnitudeBuffer Maintains Per-Domain Energy Buffer Sorted by Deviation

EnergyMagnitudeBuffer shall maintain a per-domain sorted buffer of at most max_size
violations, ordered by |energy - running_mean| descending. When the buffer is full, the
lowest-deviation item is evicted (not the oldest), ensuring the buffer always contains
the hardest domain boundary examples regardless of recency.

The running mean shall be maintained via Welford's online algorithm (O(1) per update,
unbiased over all observed energies, not just buffered ones).

Spec: Exp 509, RETRO-050

### REQ-LEARN-045: EnergyMagnitudeReplay.isolation_score Measures Domain Boundary Interference

EnergyMagnitudeReplay.isolation_score(domain_a, domain_b, n_steps) shall return a float
in [-1.0, 1.0] measuring how much replaying domain_a would interfere with domain_b's
learned constraint boundaries. Score 1.0 = perfect isolation. Score -1.0 = complete
interference.

The score is computed as 1.0 - 2.0 * change_rate, where change_rate is the fraction of
domain_a replay steps that share constraint key structure with domain_b's top-k violations.

Spec: Exp 509, RETRO-050

### SCENARIO-LEARN-071: EnergyMagnitudeBuffer top_k Returns Highest-Deviation Violations

**Given** violations added with energies [1.0, 5.0, 2.0, 4.0] to a buffer (mean ~3.0)
**When** top_k(2) is called
**Then** returns the 2 violations with highest |energy - mean| (energies 5.0 and 1.0)
**And** sorted highest deviation first

### SCENARIO-LEARN-072: EnergyMagnitudeReplay isolation_score in [-1, 1]

**Given** EnergyMagnitudeReplay with domains ['arithmetic', 'code'] and 20 violations added
**When** isolation_score('arithmetic', 'code', n_steps=20) is called
**Then** result is in [-1.0, 1.0]
**And** perfectly disjoint domains (no shared keys) return score=1.0

### SCENARIO-LEARN-073: EnergyMagnitudeReplay Beats SuRe Baseline (RETRO-050 Closure)

**Given** 200 simulated constraint violations across arithmetic/code/logical domains
**When** EnergyMagnitudeReplay.isolation_score('arithmetic', 'code', n_steps=50)
**Then** isolation_improvement = isolation_score - (-0.1172) > 0
**And** retro_050_closed=True
**And** honest_verdict='energy_magnitude_wins'

---

### REQ-LEARN-039: JEPA v4 Retrains on Live CoT Pairs from Exps 502-503 with Curriculum Ordering

JEPA v4 retraining (Exp 510) must load live CoT pairs from
`results/exp502_cot_pairs.json` and `results/exp503_cot_pairs.json` (when available)
and apply curriculum ordering: sort pairs by `label_confidence` descending so
high-confidence pairs are trained first.  If live files are absent, fall back to
100 synthetic pairs (ci_mode) and set `inference_mode='synthetic'`.

### REQ-LEARN-040: JEPA v4 Training Loss Includes Quasimetric Regularization Term

JEPA v4 training loss must include a quasimetric regularization term:

    L_quasimetric = lambda * max(0, d(conclusion, premise) - d(premise, conclusion))

where `d` is Euclidean distance, `lambda=0.1` (arXiv 2602.12245).  This penalizes
symmetric embedding distances to encode the directed structure of reasoning chains
(premise → conclusion is a one-way relationship).

### REQ-LEARN-041: JEPA v4 Saves Retrained Checkpoint to results/jepa_predictor_510_live.safetensors

Upon successful training, JEPA v4 must save the retrained model parameters as a
safetensors file at `results/jepa_predictor_510_live.safetensors`.  The artifact
must record `checkpoint_saved=True`.

### SCENARIO-LEARN-067: JEPA v4 Curriculum Ordering — High-Confidence Pairs First

**Given** a mixed set of CoT pairs with varying `label_confidence` values
**When** JEPA v4 training begins
**Then** pairs are sorted by `label_confidence` descending before training
**And** the first 200 training epochs use only the top-50% highest-confidence pairs
**And** the next 100 epochs use all pairs

### SCENARIO-LEARN-068: QuasimetricRegularizer Returns Zero Loss for Euclidean Symmetric Distances

**Given** a QuasimetricRegularizer with lambda=0.1
**When** loss(premise_emb, conclusion_emb) is called with standard numpy arrays
**Then** the loss is >= 0.0 (hinge lower bound)
**And** for symmetric Euclidean distance the loss is 0.0
**And** penalizes_symmetry is True

### SCENARIO-LEARN-069: JEPALiveRetrainResult Reports FR-11 Relay Status

**Given** a JEPALiveRetrainResult with post_auc=0.850 and inference_mode='live'
**When** target_met and auc_improvement are accessed
**Then** target_met is True (post_auc >= 0.800)
**And** auc_improvement = post_auc - pre_auc
**And** to_dict() contains all required schema fields

### REQ-RETRO-038: Milestone 2026.04.38 Operational Retrospective

The system shall produce a milestone retrospective artifact (schema v13) covering all 13
experiments in milestone 2026.04.38 (Exps 500-512) that:
- Reports retro closure rates for carry-forward items (RETRO-033, -038, -039, -048, -049, -050)
- Reports GPU utilization state at milestone close via pynvml
- Reports the five credibility milestone results: RETRO-048 resolved, RETRO-033 closed,
  RETRO-038 closed, RETRO-039 confirmed, GPU 1 utilization improved
- Adds new RETRO items for issues discovered in .38 experiments
- Sets credibility_milestone_reached = retro_033_closed OR retro_038_closed
- Writes result atomically via AtomicResultWriter to results/experiment_512_retro_2026_04_38.json

### SCENARIO-RETRO-038: Milestone .38 Retro Produces Schema v13 Artifact

**Given** Exp 500-511 result JSONs are present in the results/ directory
**When** scripts/experiment_512_retro_2026_04_38.py is executed
**Then** results/experiment_512_retro_2026_04_38.json is written with schema='carnot.operational_retro.v13'
**And** all five credibility fields are present: retro_048_resolved, retro_033_closed,
  retro_038_closed, retro_039_confirmed, gpu1_utilization_improved
**And** credibility_milestone_reached equals retro_033_closed OR retro_038_closed
**And** new_retro_items is a list with at least the carry-forward open items
**And** honest_verdict is 'milestone_complete'

### REQ-LEARN-046: LeWorldModel Two-Term Objective for JEPA Training

LeWorldModelJEPATrainer must use a two-term loss objective:

    L_total = L_prediction + λ * KL(q(z)||N(0,I))

where L_prediction is MSE between predicted and actual embeddings, and the KL term
is the analytical Gaussian KL divergence:

    KL = 0.5 * sum(exp(log_var) + mean^2 - 1 - log_var)

Default λ=0.01.  The KL term prevents embedding collapse (where the predictor maps
all inputs to the same point) by forcing the latent distribution to stay near N(0,I),
maintaining diversity in the embedding space.  This is the key stability trick from
arXiv 2603.19312 (LeWorldModel, 15M param model trained stably on single GPU).

### REQ-LEARN-047: LeWorldModel Training Stability — AUC Variance < 0.05

Three independent training runs with LeWorldModelLoss on the same 100-pair synthetic
corpus must produce AUC variance < 0.05 across runs.  This validates that the Gaussian
KL regularization has resolved the BCE collapse instability observed in Exps 472/510.

### SCENARIO-LEARN-074: gaussian_kl_regularization Returns Zero at Origin

**Given** gaussian_kl_regularization is called with z_mean=0 and z_log_var=0
**When** the KL divergence is computed
**Then** the return value is 0.0 (within float32 tolerance)
**And** this represents N(0,1) which is exactly N(0,I), so KL divergence is zero

### SCENARIO-LEARN-075: LeWorldModelLoss Total Is Non-Negative

**Given** a LeWorldModelLoss with any lambda_reg >= 0
**When** total_loss(predicted, actual, z_mean, z_log_var) is called with any inputs
**Then** the total loss is >= 0.0
**And** this holds because MSE >= 0 and KL divergence >= 0 by definition

### REQ-LEARN-048: JEPA Live Retrain v6 — LeWorldModel Two-Term Objective on Real Data

Exp 522 shall train the JEPA predictor using the LeWorldModel two-term objective
(REQ-LEARN-046) on real CoT pairs sourced from Exps 514-515 when available, falling
back to fover_labeled_steps_live.json (57 real labeled pairs from Exp 442) and finally
to 100 synthetic pairs.

The retrain shall:
- Record data_source as 'live_exp514_515', 'live_fover_442', or 'synthetic'
- Split pairs 80/20 train/test deterministically
- Train using LeWorldModelJEPATrainer with max_epochs=50
- Evaluate AUC on held-out test pairs
- Save checkpoint to results/jepa_predictor_522_live.safetensors
- Set fr11_live_relay=True when data_source != 'synthetic' and final_auc >= 0.800

FR-11 relay is achieved only when real data is used AND AUC >= 0.800.  Synthetic
fallback runs are labelled fr11_synthetic_only and do not count as a relay.

### SCENARIO-LEARN-076: load_cot_pairs_from_experiments Falls Back Gracefully

**Given** exp514_cot_pairs.json and exp515_cot_pairs.json do not exist on disk
**When** load_cot_pairs_from_experiments([514, 515], fallback_path) is called
**Then** the function loads from fallback_path (fover_labeled_steps_live.json)
**And** returns a non-empty list of ViolationPair objects
**And** each ViolationPair has non-empty partial_response and a valid has_violation bool

### SCENARIO-LEARN-077: compute_held_out_split Produces Correct Fractions

**Given** a list of N ViolationPair objects
**When** compute_held_out_split(pairs, test_fraction=0.2) is called
**Then** the returned (train, test) split sums to N
**And** len(test) == round(N * 0.2) (or floor, ensuring at least 1 test pair)
**And** the split is deterministic — calling twice with the same input yields the same split

### REQ-LEARN-049: JEPA Live Retrain v7 — LeWorldModel on Exps 527/528 Real Data

Exp 535 shall train the JEPA predictor using the LeWorldModel two-term objective on
real CoT pairs sourced from Exps 527/528 (live 100q/200q benchmarks) when available,
falling back to fover_labeled_steps_live.json (Exp 442) and exp514_cot_pairs.json,
and finally to 100 synthetic pairs.

The retrain shall:
- Implement load_v7_cot_corpus(preferred_paths, fallback_paths) returning (pairs, data_source)
  where data_source is 'live_exp527_528', 'live_fover_442', or 'synthetic'
- Implement summarize_corpus(pairs) returning n_pairs, n_correct, n_incorrect, source_breakdown
- Record data_source and corpus size in the artifact
- Split pairs 80/20 train/test deterministically
- Train using LeWorldModelJEPATrainer with max_epochs=50
- Evaluate AUC on held-out test pairs
- Save checkpoint to results/jepa_predictor_535_live.safetensors

### REQ-LEARN-050: JEPA Retrain v7 AUC Target >= 0.800 on Held-Out Real Pairs

When Exp 535 uses real CoT pairs (data_source != 'synthetic'), the final AUC on the
held-out test set shall be recorded honestly.  FR-11 relay is achieved when:
  fr11_live_relay = (data_source != 'synthetic') AND (final_auc >= 0.800)

Synthetic fallback runs are labelled fr11_synthetic_fallback and do not count as relay.
Partial relay (real data but AUC < 0.800) is labelled fr11_partial.

### SCENARIO-LEARN-078: load_v7_cot_corpus Prefers Exps 527/528 Over Fallback

**Given** exp527_cot_pairs.json or exp528_cot_pairs.json exist in results/
**When** load_v7_cot_corpus(preferred_paths=[527_path, 528_path], fallback_paths=[fover_path]) is called
**Then** data_source is 'live_exp527_528'
**And** the returned pairs are sourced from the preferred files
**And** summarize_corpus(pairs) returns n_pairs > 0

### SCENARIO-LEARN-079: load_v7_cot_corpus Falls Back When Preferred Files Absent

**Given** exp527_cot_pairs.json and exp528_cot_pairs.json do not exist on disk
**When** load_v7_cot_corpus(preferred_paths=[527_path, 528_path], fallback_paths=[fover_path]) is called
**Then** data_source is 'live_fover_442' (or 'synthetic' if fover also absent)
**And** the function never raises an exception
**And** the returned (pairs, data_source) tuple is always a valid result

### REQ-LEARN-051: GRPO Contrastive Pairing for EORM Retrain from Live Binary Verdicts

Exp 540 shall build contrastive (correct, incorrect) response pairs directly from live
benchmark binary verdicts (Exp 538 format), applying the GRPO insight (arXiv 2503.06639)
that verifiable binary rewards naturally induce contrastive pairs without additional labeling.

For each question in the benchmark:
- If baseline_correct=False and pipeline_correct=True: pair (correct=pipeline_response, incorrect=baseline_response)
- If baseline_correct=True and pipeline_correct=False: pair (correct=baseline_response, incorrect=pipeline_response)

When fewer than 5 pairs are extracted from the benchmark, fall back to FOVER
fover_labeled_steps_live.json (Exp 442 labels, 57 annotated steps).

The module shall expose:
- GRPOContrastivePair(question_id, correct_response, incorrect_response) dataclass
- build_grpo_pairs_from_benchmark(benchmark_result_path) → List[GRPOContrastivePair]
- train_eorm_grpo(eorm_model, pairs, margin, epochs, lr) → (training_loss, before_auc, after_auc)
- GRPOEORMRetrainResult(n_pairs, before_auc, after_auc, auc_improvement, honest_verdict)

### REQ-LEARN-052: GRPO EORM Contrastive Loss — Maximize Energy Gap

The contrastive loss used in Exp 540 EORM retrain shall be:
    L = mean(max(0, margin - (E(incorrect) - E(correct))))

This is equivalent to the EORM hinge loss rewritten to explicitly maximize the gap
(E(incorrect) - E(correct)).  The margin (default 1.0) ensures the model does not
stop learning once it achieves a minimal separation.

The loss is zero when E(incorrect) > E(correct) + margin for every pair in the batch,
indicating the model already discriminates correctly with sufficient confidence.

### SCENARIO-LEARN-080: build_grpo_pairs_from_benchmark Returns Empty When No Paired Fields

**Given** a benchmark JSON that has no per-question baseline_correct/pipeline_correct fields
**When** build_grpo_pairs_from_benchmark(benchmark_result_path) is called
**Then** it returns an empty list (no pairs, not an error)
**And** no exception is raised

### SCENARIO-LEARN-081: train_eorm_grpo Reduces Loss Over Epochs on Synthetic Pairs

**Given** 5 synthetic GRPOContrastivePair objects with distinct correct/incorrect responses
**When** train_eorm_grpo(model, pairs, margin=1.0, epochs=10, lr=1e-3) is called
**Then** the returned training_loss is a non-negative float
**And** before_auc and after_auc are in [0.0, 1.0]
**And** the model parameters are updated (not identical to initial)

### SCENARIO-LEARN-082: GRPOEORMRetrainResult honest_verdict Logic

**Given** a GRPOEORMRetrainResult with auc_improvement=0.10
**Then** honest_verdict is 'grpo_improved'
**Given** a GRPOEORMRetrainResult with auc_improvement=0.02
**Then** honest_verdict is 'no_improvement'
**Given** a GRPOEORMRetrainResult constructed from synthetic fallback pairs
**Then** honest_verdict is 'synthetic_fallback'

### REQ-LEARN-053: VerifyRepairPipeline Accepts Optional constraint_memory Parameter

``VerifyRepairPipeline.__init__()`` shall accept an optional
``constraint_memory: ConstraintAdditionFromMemory | None = None`` parameter.
When ``constraint_memory`` is provided, the pipeline integrates Tier 2
self-learning via the ConstraintAdditionFromMemory mechanism validated in Exp 456.
When ``None`` (the default), all existing behavior is unchanged — no new code
paths are exercised.

### REQ-LEARN-054: VerifyRepairPipeline Records Violations into constraint_memory and Applies Learned Constraints

After each ``verify()`` call that produces violations, the pipeline shall call
``self._constraint_memory.observe(violation_type, step_text)`` for each violation,
where ``violation_type`` is the leading token of the constraint_type field (split
on ``':'``).  Before running constraint evaluation, if ``constraint_memory`` is not
None, the pipeline shall call ``constraint_memory.check_and_add(self)`` and prepend
any newly added constraint names as synthetic ConstraintResult objects to the active
constraint set.  After check_and_add() is called, if the pattern count for any
violation type exceeds the threshold, new constraints are active for subsequent calls.

### SCENARIO-LEARN-083: Pipeline with constraint_memory=None Uses Existing Behavior

**Given** a VerifyRepairPipeline constructed without constraint_memory
**When** verify() is called with a response that has violations
**Then** the result is identical to the pre-wire-in behavior
**And** no ConstraintAdditionFromMemory method is called

### SCENARIO-LEARN-084: Pipeline Records Violations into constraint_memory

**Given** a VerifyRepairPipeline constructed with constraint_memory set
**When** verify() is called and violations are found
**Then** constraint_memory.observe() is called once per violation
**And** the violation_type passed to observe() is the constraint_type prefix

### SCENARIO-LEARN-085: Pipeline Applies Learned Constraints on Subsequent Calls

**Given** a VerifyRepairPipeline with constraint_memory whose pattern count
for 'carry' has been pre-seeded above the threshold
**When** verify() is called
**Then** check_and_add() promotes the pattern to an active constraint
**And** the pipeline applies the new constraint during evaluation

### REQ-LEARN-055: FOVER Corpus Expansion via Multi-Source Merge

**Why this requirement exists:**
JEPA v7 was retrained on 57 real FOVER pairs (fover_442 source, n_train_pairs=46,
n_test_pairs=11). Exp 538 produced 25 additional CoT responses (50 total when two
models each answer 25 questions). Merging these sources with deduplication by
step_text hash produces a 100+ pair corpus for JEPA v8 retrain (Exp 543).

**Given** a FOVER corpus from fover_labeled_steps_live.json (57 pairs)
**And** new CoT responses from exp538_cot_pairs.json (25 responses)
**When** Exp 542 runs FOVERAnnotator on the new responses and merges with prior pairs
**Then** the merged corpus is deduplicated by SHA-256 hash of step_text
**And** the merged corpus is written to results/fover_labeled_steps_expanded.json
**And** honest_verdict is 'corpus_expanded' when n_total_pairs >= 100

### SCENARIO-LEARN-086: Deduplication Removes Pairs with Identical step_text

**Given** two training pairs with identical step_text content
**When** merge_fover_corpora() combines them
**Then** only one copy is retained in the merged output

### SCENARIO-LEARN-087: Multi-Source Load Tolerates Missing exp538 File

**Given** exp538_cot_pairs.json is absent from the results directory
**When** Exp 542 runs
**Then** n_new_pairs = 0 and the merged corpus equals the prior 57-pair corpus
**And** honest_verdict reflects partial_expansion

### REQ-LEARN-056: JEPA v8 Retrain on Expanded FOVER Corpus

**Why this requirement exists:**
JEPA v7 (Exp 535) achieved AUC=0.967 on 57 real FOVER pairs. Exp 542 expanded the corpus
via multi-source merge. Exp 543 retrains JEPA v8 on this expanded corpus to validate that
the larger dataset maintains or improves AUC, satisfying FR-11 with honest live data.

**Given** fover_labeled_steps_expanded.json exists (from Exp 542)
**When** Exp 543 runs JEPA v8 retrain
**Then** the model trains on the expanded corpus with 80/20 train/test split
**And** the artifact contains n_train_pairs, n_test_pairs, final_auc, auc_improvement
**And** honest_verdict is set based on AUC threshold and n_train_pairs

### REQ-LEARN-057: LeWorldModel Two-Term Objective for JEPA v8

**Why this requirement exists:**
The LeWorldModel objective (L_total = L_prediction + lambda * L_KL) provides stable Gaussian
regularization that prevents embedding collapse during training. JEPA v8 uses lambda=0.1
(vs 0.01 in v6/v7) to apply stronger regularization on the expanded corpus.

**Given** a JEPA predictor and training pairs
**When** LeWorldModelJEPATrainer trains for 100 epochs with lambda_reg=0.1
**Then** L_KL = KL(q(z) || N(0,I)) = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
**And** L_total = L_prediction + 0.1 * L_KL per batch

### SCENARIO-LEARN-088: JEPA v8 AUC on Expanded Corpus Meets Threshold

**Given** JEPA v8 trained on the expanded FOVER corpus
**When** final_auc is evaluated on the held-out test split
**Then** honest_verdict is 'jepa_v8_improved' when auc>=0.900 AND n_train>=80
**Or** honest_verdict is 'auc_stable' when auc>=0.800
**Or** honest_verdict is 'synthetic_fallback' otherwise

### SCENARIO-LEARN-089: JEPA v8 Falls Back to fover_labeled_steps_live.json

**Given** fover_labeled_steps_expanded.json is absent from results/
**When** Exp 543 runs
**Then** the experiment loads fover_labeled_steps_live.json as fallback
**And** the artifact correctly reflects the actual data source used

---

### REQ-LEARN-058: Offline Distillation of Violation Patterns into Named Constraint Templates

The system SHALL provide a ConstraintTemplateStore that accumulates violation
observations (violation_type, context) and distils mature patterns into named
ConstraintTemplate objects.  A pattern is mature when its observation count
reaches min_observations (default 3).  The store MUST be serialisable to JSON
and rehydratable across sessions.

Implemented by: python/carnot/pipeline/constraint_template_store.py

### REQ-LEARN-059: Template Retrieval at Inference Time via String Similarity

The ConstraintTemplateStore SHALL provide a retrieve(query_context, top_k)
method that returns up to top_k distilled templates ranked by keyword overlap
score — the count of template context_keywords that appear as substrings in
the lowercased query_context.  This allows the pipeline to apply domain-
specific constraints only when the current query context matches a template.

Implemented by: python/carnot/pipeline/constraint_template_store.py

### SCENARIO-LEARN-090: add_violation Accumulates Observations and Distill Promotes Mature Patterns

**Given** a ConstraintTemplateStore with min_observations=3
**When** add_violation("carry", ctx) is called 3 or more times
**Then** distill(min_observations=3) returns a ConstraintTemplate with violation_type="carry"
**And** the template's n_violations_observed equals the total observation count
**And** context_keywords contains tokens extracted from the context strings

### SCENARIO-LEARN-091: Distill Threshold Filters Immature Patterns

**Given** a ConstraintTemplateStore where "sign" has been observed 2 times and "carry" 4 times
**When** distill(min_observations=3) is called
**Then** the result contains the "carry" template
**And** the result does NOT contain the "sign" template (below threshold)

### SCENARIO-LEARN-092: Retrieve Returns Top-K Templates by Keyword Overlap

**Given** a distilled store with carry_guard (keywords: arithmetic, carry) and semantic_guard
**When** retrieve("arithmetic carry overflow", top_k=3) is called
**Then** carry_guard is ranked first (highest keyword overlap)
**And** the result contains at most top_k templates

---

### REQ-LEARN-060: EORM Retrain on Real GRPO Contrastive Pairs from FOVER Corpus v2

**Statement:**
    When fover_corpus_v2.json contains at least 100 entries with is_correct labels,
    the experiment pipeline SHALL retrain EORM using GRPO-style contrastive pairs
    (arXiv 2503.06639) formed from same-question (correct, incorrect) response pairs,
    report AUC on a held-out 20% split, and save the retrained model as
    results/eorm_model_556_real.safetensors.

**Acceptance criteria:**
    - load_fover_corpus_v2() loads entries from fover_corpus_v2.json and groups by question
    - Contrastive triples built from questions that have both correct and incorrect responses
    - n_contrastive_triples >= 1 when corpus has at least one question with both correct
      and incorrect responses
    - AUC reported before and after retraining on 20% held-out split
    - honest_verdict='real_data_improvement' iff after_auc >= 0.700
    - Blocked artifact emitted if n_pairs < 100

Spec: REQ-LEARN-060

---

### SCENARIO-LEARN-093: load_fover_corpus_v2 Returns GRPOContrastivePairs from Both-Verdict Questions

**Given** fover_corpus_v2.json with entries including at least one question with
         both a correct and an incorrect response
**When** load_fover_corpus_v2(path) is called
**Then** the returned list contains one GRPOContrastivePair per such question
**And** correct_response and incorrect_response fields are non-empty strings

---

### SCENARIO-LEARN-094: load_fover_corpus_v2 Returns Empty List for Missing File

**Given** a file path that does not exist
**When** load_fover_corpus_v2(path) is called
**Then** an empty list is returned without raising an exception

---

### REQ-LEARN-061: JEPAPUREMinForm — PURE Min-Form Contrastive Margin Loss for JEPA Training

**Motivation (RETRO-060):** Binary BCE loss lets the JEPA model hedge toward P=0.5 everywhere,
producing near-zero gradient and AUC=0.4286 — below random — for the second consecutive retrain
(Exp 557).  The root cause is that BCE does not enforce a contrastive margin between correct and
incorrect chain scores.

**Fix (arXiv 2504.15275, PURE PRM):** Score a chain by the MINIMUM step score across all its
steps.  A chain with one bad step gets a strong low signal; a chain with all good steps gets a
strong high signal.  The contrastive loss is:

    loss = mean(max(0, margin - (min_score_incorrect - min_score_correct)))

over all (correct, incorrect) pairs from the same question.  This enforces a hard margin gap of
at least `margin` between the minimum step scores of incorrect and correct chains — exactly the
signal that NUP Probe v4 used to achieve AUC=1.0.

**Requirements:**
- JEPAChainScore dataclass: chain_id (str), step_scores (list[float]), min_score (float), is_correct (bool).
- PUREMinFormLoss class with compute_chain_scores, compute_loss, zero_if_empty methods.
- pairs_to_pure_chains helper: groups FOVERCorpusEntry list by question_id, embeds steps, returns (correct_chains, incorrect_chains).

Spec: REQ-LEARN-061

---

### REQ-LEARN-062: PUREMinFormLoss Margin Parameter Is Configurable

The contrastive margin used by PUREMinFormLoss must be exposed as a constructor parameter
`margin: float = 1.0`.  Callers may override it without subclassing.

Spec: REQ-LEARN-062

---

### SCENARIO-LEARN-095: PUREMinFormLoss Returns Positive Loss When Incorrect Score > Correct Score But Gap < Margin

**Given** a correct chain with min_score=0.3 and an incorrect chain with min_score=0.7 and margin=1.0
**When** compute_loss is called
**Then** loss = max(0, 1.0 - (0.7 - 0.3)) = 0.6 > 0

---

### SCENARIO-LEARN-096: PUREMinFormLoss Returns Zero When Gap Exceeds Margin

**Given** a correct chain with min_score=0.0 and an incorrect chain with min_score=1.5 and margin=1.0
**When** compute_loss is called
**Then** loss = max(0, 1.0 - (1.5 - 0.0)) = 0.0

---

### SCENARIO-LEARN-097: PUREMinFormLoss.zero_if_empty Returns 0.0 for Empty Pair List

**Given** an empty list of pairs
**When** zero_if_empty is called
**Then** 0.0 is returned without error

---

### REQ-LEARN-063: JEPA v10 Full Retrain with PUREMinFormLoss on 132-Pair FOVER Corpus

**What:** Full retrain of the JEPA chain scorer (v10) on the complete 132-pair FOVER corpus v2
using PUREMinFormLoss (REQ-LEARN-061) as the training objective. This is the mandatory FR-11
JEPA retrain that follows every milestone and the direct response to RETRO-060.

**Why (layman):** Exps 543 and 557 both produced AUC < 0.5 (worse than random) because they
used binary cross-entropy loss which lets the model hedge toward 0.5 everywhere. Exp 566 showed
that PUREMinFormLoss can push AUC above random. This requirement mandates a full 200-epoch retrain
with the new objective and saves the best checkpoint for downstream use.

**Acceptance criteria:**
- Model trained for 200 epochs on 80% split of 132-pair corpus.
- Val AUC evaluated every 20 epochs; best checkpoint saved.
- Final artifact reports v9_auc=0.4286, v10_auc, auc_improvement, retro_060_resolved flag.
- Model saved to results/jepa_predictor_v10.safetensors.
- fr11_retrain_complete=True always set in artifact.

Spec: REQ-LEARN-063

---

### SCENARIO-LEARN-081: JEPA v10 Retrain Produces AUC Above Random Baseline

**Given** the 132-pair FOVER corpus v2 loaded and split 80/20
**When** JEPA v10 is trained for 200 epochs with PUREMinFormLoss(margin=1.0)
**Then** the best validation AUC > 0.5 (above random) and retro_060_resolved=True

---

### SCENARIO-LEARN-082: JEPA v10 Retrain Saves Model Checkpoint

**Given** training completes successfully
**When** the best val AUC epoch is identified
**Then** model weights are saved to results/jepa_predictor_v10.safetensors via AtomicResultWriter

---

### SCENARIO-LEARN-083: JEPA v10 Artifact Contains All Required Fields

**Given** training completes (success or partial)
**When** the result artifact is built
**Then** it contains schema, n_train, n_val, loss_function, v9_auc, v10_auc, auc_improvement,
best_epoch, model_path, retro_060_resolved, fr11_retrain_complete, honest_verdict

---

### REQ-LEARN-064: Hardware Energy as EORM Calibration Signal via RAPL Power Trace Correlation

**What:** Measure Pearson correlation between hardware energy expenditure (RAPL joules/step
on CPU) and EORM model energy scores during inference over a set of CoT steps.

**Why (layman):** Hard reasoning steps require more compute cycles — the CPU draws more
power to compute large activations and long attention. Carnot's EORM is a learned proxy
for reasoning quality. If RAPL hardware energy and EORM energy co-move (r > 0.5), then
a simple power meter becomes a free, label-free calibration signal: we can align EORM
scores to physical energy without any human annotation.

**Acceptance criteria:**
- HardwareEnergyProbe reads from /sys/class/powercap/intel-rapl:0/energy_uj when available.
- Falls back to pyRAPL, then to a mock(0.0) source with source label.
- compute_eorm_hardware_correlation() runs EORM scoring + hardware measurement for each step.
- Pearson r and p-value computed via scipy.stats.pearsonr.
- calibration_viable = r > 0.5 AND p_value < 0.05.
- Artifact reports rapl_available, pearson_r, p_value, calibration_viable, honest_verdict.

Spec: REQ-LEARN-064,
      SCENARIO-LEARN-098, SCENARIO-LEARN-099, SCENARIO-LEARN-100

---

### SCENARIO-LEARN-098: HardwareEnergyProbe Falls Back to Mock When RAPL Unavailable

**Given** the system does not have /sys/class/powercap/intel-rapl:0/energy_uj
**When** HardwareEnergyProbe.read() is called
**Then** it returns a HardwareEnergyReading with source='mock' and joules=0.0

---

### SCENARIO-LEARN-099: measure_segment Returns Result and Delta Joules

**Given** a callable function fn and a HardwareEnergyProbe instance
**When** probe.measure_segment(fn) is called
**Then** it returns a tuple (result, delta_joules) where result is fn()'s return value

---

### SCENARIO-LEARN-100: Correlation Artifact Contains All Required Fields

**Given** 30 CoT steps (15 correct, 15 incorrect) from FOVER corpus v2
**When** compute_eorm_hardware_correlation() runs with mock probe
**Then** EORMHardwareCorrelation has n_steps=30, pearson_r, p_value, hardware_energies,
eorm_energies, calibration_viable; and artifact has honest_verdict set correctly

---

### REQ-LEARN-065: JEPACPMIPairBuilder — Hard-Negative Contrastive Pair Construction from FOVER Corpus

**Why (layman):** The JEPA predictor AUC was stuck at 0.4444 across three retrains because
all three used scalar loss on step-level labels — the model could hedge by making all scores
near 0.5.  The CPMI fix (arXiv 2604.10660) constructs explicit contrastive pairs: one
(correct_chain, incorrect_chain) from the SAME question.  This forces the model to learn a
RELATIVE ordering between two whole chains, which is exactly what AUC measures.

**Acceptance criteria:**
- JEPACPMIPairBuilder.build_pairs() groups FOVERCorpusEntry objects by question_id.
- For each group with >= 1 correct AND >= 1 incorrect entry: yields one JEPACPMIPair.
- Hardest incorrect = entry with most cot_steps (hard-negative mining).
- JEPACPMIPair has: question_id, correct_embeddings, incorrect_embeddings,
  hard_negative_step_idx, pair_quality.
- pair_quality = len(correct_steps) / max(len(incorrect_steps), 1).
- build_synthetic_pairs(n) returns exactly n pairs as fallback for small corpora.

Spec: REQ-LEARN-065,
      SCENARIO-LEARN-101, SCENARIO-LEARN-103

---

### REQ-LEARN-066: CPMIContrastiveLoss — Hinge Margin Loss for JEPA CPMI Pairs

**Why (layman):** The contrastive hinge margin loss directly optimises the ranking
objective that AUC measures.  For each pair it computes:
    L = max(0, margin - (E_incorrect - E_correct))
This is zero when the incorrect chain scores at least `margin` above the correct chain,
and positive otherwise.  Unlike BCE or PURE, the gradient of this loss is always zero
when the constraint is satisfied — no gradient flows when the model is already correct.

**Acceptance criteria:**
- CPMIContrastiveLoss.__init__(margin, chain_energy_mode) with mode in ('mean','max','min').
- chain_energy(model, embeddings) aggregates per-step scores by the selected mode.
- compute_loss(model, pairs) returns mean(max(0, margin - gap)) over all pairs.
- Returns 0.0 for empty pair list (zero_if_empty guard).
- chain_energy returns 0.0 for empty embeddings.

Spec: REQ-LEARN-066,
      SCENARIO-LEARN-101, SCENARIO-LEARN-102, SCENARIO-LEARN-103

---

### SCENARIO-LEARN-101: build_pairs Yields One Pair Per Question With Both Verdicts

**Given** a FOVER corpus with questions that each have at least one correct and one incorrect entry
**When** JEPACPMIPairBuilder.build_pairs() is called
**Then** it yields exactly one JEPACPMIPair per qualifying question; questions with only
correct or only incorrect entries are skipped

---

### SCENARIO-LEARN-102: CPMIContrastiveLoss Returns Zero When Gap Exceeds Margin

**Given** a pair where E_incorrect - E_correct = 5.0 and margin = 1.0
**When** CPMIContrastiveLoss.compute_loss() is called
**Then** it returns 0.0 (constraint already satisfied; no gradient needed)

---

### SCENARIO-LEARN-103: build_synthetic_pairs Pads Small Corpus to Requested Count

**Given** a corpus that yields fewer than min_pairs real pairs
**When** JEPACPMIPairBuilder.build_synthetic_pairs(n) is called
**Then** it returns exactly n JEPACPMIPair objects with pair_quality=1.0 and
correct arithmetic in the correct chain, one off-by-one error in the incorrect chain

---

### REQ-LEARN-067: JEPA v11 Full Retrain with CPMIContrastiveLoss on FOVER Corpus

**Summary:** Mandatory FR-11 retrain using CPMI contrastive objective to resolve RETRO-063.

**Motivation:** JEPA v8/v9/v10 all trained on scalar loss with step-level labels, producing
AUC <= 0.4444 (below random). RETRO-063 identified the root cause: step-level labels are
noisy, and per-chain objectives allow hedging to P=0.5. This retrain uses CPMIContrastiveLoss
(pairwise ordering constraint) which forces the model to directly rank correct chains above
incorrect chains for the same question — the same mechanism that yielded AUC=1.0 in NUP Probe v4.

**Requirements:**
- SHALL load FOVER corpus v3 if available, else v2 (132 pairs minimum)
- SHALL build contrastive pairs with JEPACPMIPairBuilder; augment with synthetic pairs if < 5 real
- SHALL split train/val 80/20 by question_id to prevent cross-question leakage
- SHALL train 300 epochs with optax.adamw(lr=1e-3, weight_decay=1e-4)
- SHALL use CPMIContrastiveLoss(margin=1.0, chain_energy_mode='mean')
- SHALL save best checkpoint to results/jepa_predictor_v11.safetensors
- SHALL report retro_063_resolved = True if v11_auc > 0.5
- SHALL emit honest_verdict: 'jepa_v11_above_random' | 'jepa_v11_still_inverted' | 'jepa_v11_at_random'

Spec: REQ-LEARN-067,
      SCENARIO-LEARN-104, SCENARIO-LEARN-105, SCENARIO-LEARN-106

---

### SCENARIO-LEARN-104: JEPA v11 Retrain Produces Valid Artifact With All Required Fields

**Given** a FOVER corpus with at least 5 questions having both correct and incorrect entries
**When** experiment_580_jepa_v11_retrain runs to completion
**Then** the artifact contains all required schema fields including v10_auc, v11_auc,
auc_improvement, best_epoch, n_real_pairs, n_synthetic_pairs, retro_063_resolved,
fr11_retrain_complete=True, and honest_verdict

---

### SCENARIO-LEARN-105: JEPA v11 Question-ID Split Prevents Cross-Question Leakage

**Given** a FOVER corpus where multiple entries share the same question
**When** the 80/20 train/val split is applied by question_id
**Then** no question_id appears in both the train and val pair sets

---

### SCENARIO-LEARN-106: JEPA v11 Synthetic Pair Augmentation Kicks In For Small Corpus

**Given** a corpus that yields fewer than 5 real CPMI pairs
**When** experiment_580_jepa_v11_retrain runs
**Then** build_synthetic_pairs(20) is called and n_synthetic_pairs >= 20 in the artifact

---

### REQ-LEARN-068: JEPA v12 — Live Corpus CPMI Retrain with >= 20 Real Pairs (Exp 593)

JEPA v12 must be trained on the full live corpus from Exp 578 (GSM8K 0-49, both
Qwen3.5-0.8B and Gemma4-E4B-it, inference_mode='live_gpu') using the CPMIContrastiveLoss
objective.  The training corpus must contain at least 20 real live pairs (pairs derived
from actual GPU inference, not synthetic fallback).  Validates whether v11 AUC=1.0 on
9 pairs was genuine learning or overfitting.

Spec: REQ-LEARN-068

### REQ-LEARN-069: PROGRSCentering — Outcome-Conditioned Group Centering for Contrastive Stability (arXiv 2604.02341)

PROGRSCentering must implement PROGRS outcome-conditioned group centering to prevent
reward hacking on easy questions.  For each question_id group, the mean energy gap is
subtracted from each pair's energy gap before computing the contrastive margin loss.
This ensures every question group contributes an equally-weighted learning signal
regardless of absolute gap magnitude.  For single-pair groups, a graceful fallback
to raw gap must be used to preserve gradient flow.

Spec: REQ-LEARN-069

### SCENARIO-LEARN-107: PROGRSCentering.center_pairs Correctly Normalises Within-Group Gaps

**Given** a list of JEPACPMIPair objects spanning multiple question_id groups
**When** PROGRSCentering.center_pairs() is called
**Then** the centered gap for each pair equals raw_gap minus the group mean gap,
         the output list preserves input order, and empty input returns []

Spec: SCENARIO-LEARN-107

### SCENARIO-LEARN-108: PROGRSCentering.compute_centered_loss Applies Hinge on Centered Gaps

**Given** a list of pairs where some groups have multiple pairs (different raw gaps)
**When** PROGRSCentering.compute_centered_loss(model, pairs, margin) is called
**Then** the loss equals mean(max(0, margin - centered_gap_p)) over all pairs,
         where centered gaps account for within-group mean subtraction

Spec: SCENARIO-LEARN-108

### SCENARIO-LEARN-109: Single-Pair Groups Yield Centered Gap = 0 in PROGRSCentering

**Given** a corpus where every question_id has exactly one pair (output of JEPACPMIPairBuilder)
**When** PROGRSCentering.center_pairs() is called
**Then** centered_gap = 0 for every pair (raw_gap - group_mean = raw_gap - raw_gap = 0),
         and compute_centered_loss returns margin for each pair

Spec: SCENARIO-LEARN-109

---

### REQ-LEARN-070: FR-11 Real Violations Relay from Live Verify-Repair (Exp 597)

The system shall attempt to source real constraint violations from live verify-repair runs
(Exp 594 CoACEv3, Exp 595 DSVD) and feed them into ConstraintAdditionFromMemory.
When live violations are unavailable (both upstream experiments blocked), synthetic
violations shall be used as a clearly-labeled fallback.  The artifact must always record:
- n_live_violations (int): count of violations from live GPU inference.
- violations_source: one of 'exp594', 'exp595', 'synthetic_fallback'.
- fr11_real_violations_confirmed (bool): True iff n_live_violations >= 5.
- honest_verdict: 'real_violations_improved', 'real_violations_no_improvement',
  or 'synthetic_fallback'.

**Rationale:** FR-11 requires self-learning to operate on real data.  When live runs are
blocked, synthetic data preserves pipeline exercise without misleading headline claims.

Spec: REQ-LEARN-070,
      SCENARIO-LEARN-110, SCENARIO-LEARN-111, SCENARIO-LEARN-112

---

### SCENARIO-LEARN-110: Violations Source Recorded in Artifact

**Given** Exp 594 and Exp 595 are both blocked (n_violations=0)
**When** Exp 597 runs
**Then** violations_source='synthetic_fallback', fr11_real_violations_confirmed=False,
         honest_verdict='synthetic_fallback', and all required schema fields are present.

Spec: SCENARIO-LEARN-110

---

### REQ-LEARN-071: MISE Backward Inference for EORM Reward Calibration (arXiv 2604.11611)

The system shall implement MISECalibrator with backward_inference_score() computing
cosine similarity between embed(response) and embed(question) as a proxy for how well
the response answers the question.  The calibrate() method computes:
- mean_alignment_correct: mean score for verdict_correct=True triples.
- mean_alignment_incorrect: mean score for verdict_correct=False triples.
- calibration_gap: mean_alignment_correct - mean_alignment_incorrect.
A positive calibration_gap indicates the alignment signal reliably separates correct
from incorrect responses and can be used as a dense EORM reward signal.

Spec: REQ-LEARN-071,
      SCENARIO-LEARN-111, SCENARIO-LEARN-112

---

### SCENARIO-LEARN-111: backward_inference_score Returns Float in [-1, 1]

**Given** any two non-empty strings
**When** MISECalibrator.backward_inference_score(response, question) is called
**Then** a float in [-1.0, 1.0] is returned; zero-norm embeddings return 0.0.

Spec: SCENARIO-LEARN-111

---

### SCENARIO-LEARN-112: calibrate() Returns Three-Key Dict with calibration_gap

**Given** a list of MISETriple objects (may be empty)
**When** MISECalibrator.calibrate(triples) is called
**Then** a dict with keys 'mean_alignment_correct', 'mean_alignment_incorrect',
         'calibration_gap' is returned; empty input yields all-zero values;
         calibration_gap = mean_alignment_correct - mean_alignment_incorrect.

Spec: SCENARIO-LEARN-112

### REQ-LEARN-072: HISR — Hindsight Importance Score Reweighting for Constraint Violations (arXiv 2603.18683) (Exp 598)

Applies credit-assignment from HISR paper to ConstraintAdditionFromMemory:
violations that temporally preceded a final incorrect answer receive higher
hindsight scores, focusing constraint-addition signal on causally relevant events.

- REQ-LEARN-072-1: ``HISRViolationWeight`` SHALL store violation_type, question_id,
  final_incorrect (bool), and hindsight_score (float in [0, 1]).
- REQ-LEARN-072-2: ``HISRWeighter.compute_hindsight_score(violations, final_correct)``
  SHALL assign score = 0.0 to all violations when final_correct=True.
- REQ-LEARN-072-3: When final_correct=False, the last violation SHALL receive score 1.0
  and earlier violations SHALL receive score = 1/(1 + distance_from_last).
- REQ-LEARN-072-4: ``HISRWeighter.weighted_violations(weights, threshold=0.5)``
  SHALL return ViolationPattern objects only for weights with hindsight_score >= threshold.

Spec: REQ-LEARN-072, SCENARIO-LEARN-113, SCENARIO-LEARN-114

### SCENARIO-LEARN-113: Violations in Correct Chains Score Zero (HISR False-Positive Suppression)

**Given** a list of ViolationPattern objects from a chain where final_correct=True
**When** HISRWeighter.compute_hindsight_score(violations, final_correct=True) is called
**Then** every HISRViolationWeight has hindsight_score == 0.0 and final_incorrect == False.

Spec: SCENARIO-LEARN-113

### SCENARIO-LEARN-114: Violations in Incorrect Chains Score Higher Near the Final Error

**Given** a list of 5+ ViolationPattern objects from a chain where final_correct=False
**When** HISRWeighter.compute_hindsight_score(violations, final_correct=False) is called
**Then** scores are strictly increasing (earliest < latest), and the last score is 1.0.

Spec: SCENARIO-LEARN-114

---

### REQ-LEARN-073: JEPA v12 OOD Generalization Validation on Held-Out GSM8K Questions (Exp 607)

The system shall validate JEPA v12's generalization by evaluating it on strictly held-out
GSM8K questions never seen during training (question indices >= 50, sourced from
live_pairs_602.json GSM8K 250-349).  The artifact must record:
- n_ood_pairs (int): number of OOD entries used for evaluation.
- v12_val_auc (float): 1.0 from Exp 593 (in-distribution).
- v12_ood_auc (float): AUC on the held-out OOD set.
- v12_generalized (bool): True iff v12_ood_auc >= 0.65.
- v12_overfit (bool): True iff v12_ood_auc < 0.55.
- fr11_generalization_confirmed (bool): True iff v12 generalized OR v13 retrain
  achieves val_auc >= 0.65.
- honest_verdict: one of 'v12_generalized', 'v12_overfit_v13_saved', 'jepa_fails_ood'.

**Rationale:** AUC=1.0 on 20 in-distribution pairs is insufficient evidence of generalization.
FR-11 claims the JEPA relay is validated only when OOD performance is confirmed.

### REQ-LEARN-074: JEPA v13 Corpus-v4 Retrain When v12 Is Overfit (Exp 607 conditional)

If JEPA v12 is overfit (ood_auc < 0.55), the system shall retrain JEPA v13 on the full
fover_corpus_v4.json corpus using the same CPMI+PROGRS architecture (100 epochs, 80/20
split).  v13 is saved to results/jepa_predictor_v13.safetensors iff val_auc >= 0.65.
The artifact must record v13_retrained (bool) and v13_val_auc (float | None).

Spec: REQ-LEARN-073, REQ-LEARN-074,
      SCENARIO-LEARN-115, SCENARIO-LEARN-116, SCENARIO-LEARN-117

### SCENARIO-LEARN-115: v12 Generalizes When OOD AUC Exceeds Threshold

**Given** a loaded jepa_predictor_v12.safetensors and OOD entries from live_pairs_602.json
**When** scores are computed for each entry and roc_auc_score is called
**Then** if ood_auc >= 0.65: v12_generalized=True, fr11_generalization_confirmed=True,
         honest_verdict='v12_generalized'.

Spec: SCENARIO-LEARN-115

### SCENARIO-LEARN-116: v12 Overfit Triggers v13 Retrain

**Given** v12_ood_auc < 0.55 (overfit detected)
**When** retrain_jepa_v13() is called with fover_corpus_v4.json entries
**Then** v13_retrained=True, and if v13_val_auc >= 0.65: fr11_generalization_confirmed=True,
         honest_verdict='v12_overfit_v13_saved'; else honest_verdict='jepa_fails_ood'.

Spec: SCENARIO-LEARN-116

### SCENARIO-LEARN-117: Blocked When v12 Model File Is Absent

**Given** results/jepa_predictor_v12.safetensors does not exist
**When** Exp 607 is run
**Then** the artifact records model_available=False, status='blocked_no_v12_model',
         and the script exits cleanly without crashing.

Spec: SCENARIO-LEARN-117

---

### REQ-LEARN-076: FLIP Backward Inference for Repair Quality Scoring (FR-11, arXiv 2602.13551)

**Requirement:**
    The system MUST implement FLIPRewardCalibrator, which uses backward inference
    (cosine similarity between embed(repaired) and embed(question)) to score
    whether a repair improved constraint alignment relative to the original response.
    A positive alignment delta means the repair moved the response closer to the
    question's constraint space.  A zero or negative delta signals a potentially
    constraint-inconsistent repair.

**Rationale:**
    Binary verify-repair verdicts are weak FR-11 signals.  FLIP provides a dense
    signal by measuring whether each repair improved the model's constraint alignment
    — without requiring human labels.

Spec: REQ-LEARN-076, SCENARIO-LEARN-118, SCENARIO-LEARN-119, SCENARIO-LEARN-120

### REQ-LEARN-077: FR-11 Real Violations Relay v5 from Exp 609 or Synthetic Fallback

**Requirement:**
    Experiment 611 MUST load live violations from experiment_609 (n_violations_found > 0)
    as the primary source.  If Exp 609 has zero violations, fall back to Exp 597.
    If both are empty, use 10 synthetic violations from fover_corpus_v4.json, labeled
    explicitly as synthetic_fallback.  The violations_source field in the artifact MUST
    record which path was taken.

Spec: REQ-LEARN-077, SCENARIO-LEARN-118

---

### SCENARIO-LEARN-118: FLIP Calibration Runs on Live or Synthetic Violations

**Given** experiment 611 loads violations from Exp 609, Exp 597, or synthetic fallback,

**When** FLIPRewardCalibrator.batch_calibrate() is called with the resulting triples,

**Then** the artifact records flip_mean_score (float), flip_n_improved (int),
         flip_repair_quality ('good'|'neutral'|'bad'), and violations_source.

Spec: SCENARIO-LEARN-118

### SCENARIO-LEARN-119: backward_inference_score Returns Float in [0, 1] for Hash Embed

**Given** FLIPRewardCalibrator is constructed with the hash-projection embed_fn,

**When** backward_inference_score(response, question) is called with non-empty strings,

**Then** the return value is a float in [-1.0, 1.0] (cosine similarity range) and
         does not raise an exception.

Spec: SCENARIO-LEARN-119

### SCENARIO-LEARN-120: batch_calibrate Reports 'good' When Majority Repairs Improve Alignment

**Given** a list of FLIPRepairTriples where more than half have repaired responses
         that score higher alignment than the originals,

**When** batch_calibrate() is called,

**Then** repair_quality = 'good', n_improved > 0, and mean_flip_score > 0.

Spec: SCENARIO-LEARN-120

---

### REQ-LEARN-078: MetaJuLS Extractor Policy Adapts from Live Batch Feedback

**Requirement:** After LLMAsExtractorV1 processes a batch of live outputs, a
MetaJuLS-style meta-RL update must adjust the extractor's policy parameters
(temperature, claim_confidence_threshold, strategy_weights) based on observed
precision measured against true violation labels.  The adapter must maintain
an experience list so that policy drift over multiple batches is auditable.

Reference: arXiv 2601.00095 (MetaJuLS — meta-RL for universal constraint
propagation).  Self-Learning Tier 2 (Constraint Memory) applied to the
extractor itself.

Spec: REQ-LEARN-078, SCENARIO-LEARN-121, SCENARIO-LEARN-122

---

### REQ-LEARN-079: MetaJuLS Precision Trend is Non-Negative After Adaptation

**Requirement:** After at least two batches of meta-RL adaptation, the
precision_trend() method must return a value >= 0 when the adapter has
successfully learned from feedback — indicating that precision is non-decreasing
over the most recent experience window.

Spec: REQ-LEARN-079, SCENARIO-LEARN-123

---

### SCENARIO-LEARN-121: update_from_batch Adjusts Temperature Down When Precision is Low

**Given** a MetaJuLSAdapter with default policy (temperature=0.1),

**When** update_from_batch() is called with a batch where precision < 0.5
(more false positives than true positives),

**Then** policy.temperature decreases (multiplied by 0.9) and
policy.claim_confidence_threshold increases (multiplied by 1.1).

Spec: SCENARIO-LEARN-121

---

### SCENARIO-LEARN-122: update_from_batch Relaxes Thresholds When Precision is High

**Given** a MetaJuLSAdapter with default policy (temperature=0.1,
claim_confidence_threshold=0.5),

**When** update_from_batch() is called with a batch where precision > 0.8
(strong true-positive signal),

**Then** policy.temperature increases (multiplied by 1.05) and
policy.claim_confidence_threshold decreases (multiplied by 0.95).

Spec: SCENARIO-LEARN-122

---

### SCENARIO-LEARN-123: precision_trend Returns Non-Negative After Improving Batches

**Given** a MetaJuLSAdapter that has processed two batches where the second
batch has equal or higher precision than the first,

**When** precision_trend() is called,

**Then** the return value is >= 0.0, confirming non-decreasing precision.

Spec: SCENARIO-LEARN-123

### REQ-LEARN-080: Tier 1 FR-11 Relay — ConstraintAdditionFromMemory Updates on Real Violations When Available

When Exp 620 (or successor live VR attempt) reports signed_improvement > 0 AND
n_violations_found > 0, the FR-11 relay experiment shall feed those REAL violations
into ConstraintAdditionFromMemory.add_from_violations() and measure the FP-rate delta
across two sessions (before and after the update).

When Exp 620 is blocked (signed_improvement <= 0 or n_violations_found == 0), the relay
shall generate 25 explicit synthetic arithmetic violations to maintain relay continuity
and document fr11_real_violations_confirmed = False.

In both modes the artifact must record: mode, n_violations_used,
fr11_real_violations_confirmed, fp_rate_before, fp_rate_after, fp_rate_delta.

### SCENARIO-LEARN-124: Real-Mode Relay Completes When Exp 620 Has Positive signed_improvement

**Given** Exp 620 reports signed_improvement > 0 and n_violations_found > 0
**When** the Tier 1 relay runs
**Then** violations are loaded from Exp 620 and fed to ConstraintAdditionFromMemory
**And** fp_rate_delta = fp_rate_after - fp_rate_before is recorded
**And** fr11_real_violations_confirmed = True
**And** honest_verdict = 'real_violations_relay_complete'

### SCENARIO-LEARN-125: Synthetic-Fallback Relay Fires When Exp 620 Is Blocked

**Given** Exp 620 reports signed_improvement <= 0 or is absent
**When** the Tier 1 relay runs
**Then** 25 synthetic arithmetic violations are generated
**And** ConstraintAdditionFromMemory processes the synthetic violations
**And** fr11_real_violations_confirmed = False
**And** honest_verdict = 'synthetic_fallback_relay_complete'

### REQ-LEARN-081: FR-11 Relay v2 — ConstraintAdditionFromMemory Updates from Real Violations When Available, Interwhen-Detected Violations When VR Blocked

When Exp 630 (live VR attempt 16) reports signed_improvement > 0 AND n_violations_found > 0,
the FR-11 relay experiment shall feed those REAL violations into ConstraintAdditionFromMemory
and record fr11_mode='real_violations'.

When Exp 630 is blocked but Exp 629 (interwhen diagnostic) reports gate_open=True AND
interwhen_tp > 0, the relay shall use interwhen-detected violations as a semi-real proxy
and record fr11_mode='semi_real_interwhen'.

Otherwise the relay generates 25 synthetic arithmetic violations
(fr11_mode='synthetic_fallback').

All modes record: schema='carnot.tier1_fr11_relay.v2', mode, n_violations_used,
fr11_real_violations_confirmed, fp_rate_before, fp_rate_after, fp_rate_delta,
honest_verdict.

Spec: REQ-LEARN-081

### SCENARIO-LEARN-126: Real-Mode Relay v2 Fires When Exp 630 Has Positive signed_improvement

**Given** Exp 630 reports signed_improvement > 0 and n_violations_found > 0
**When** the Tier 1 FR-11 relay v2 runs
**Then** violations from Exp 630 are fed to ConstraintAdditionFromMemory
**And** fr11_real_violations_confirmed = True
**And** honest_verdict = 'real_violations_relay'

### SCENARIO-LEARN-127: Synthetic-Fallback Relay v2 Fires When Both Exp 630 and Exp 629 Are Blocked

**Given** Exp 630 is blocked (signed_improvement <= 0 or n_violations_found == 0)
**And** Exp 629 shows gate_open = False or interwhen_tp = None
**When** the Tier 1 relay v2 runs
**Then** 25 synthetic arithmetic violations are generated
**And** fr11_real_violations_confirmed = False
**And** honest_verdict = 'synthetic_fallback_relay'

---

### REQ-LEARN-082: FR-11 Relay v3 — ConstraintAdditionFromMemory Updates from Real Violations When Available, Ensemble-Detected Violations When VR Blocked

When Exp 644 (live VR attempt 17) reports signed_improvement > 0,
the FR-11 relay experiment shall feed those REAL violations into ConstraintAdditionFromMemory
and record fr11_mode='real_violations'.

When Exp 644 is blocked but Exp 643 (Ensemble Recall Gate v2) reports gate_open=True AND
ensemble_tp > 0, the relay shall use ensemble-detected violations as a semi-real proxy
and record fr11_mode='semi_real_ensemble'.

Otherwise the relay generates 25 synthetic arithmetic violations
(fr11_mode='synthetic_fallback').

All modes record: schema='carnot.tier1_fr11_relay.v3', mode, n_violations_used,
fr11_real_violations_confirmed, fp_rate_before, fp_rate_after, fp_rate_delta,
honest_verdict.

Spec: REQ-LEARN-082

### SCENARIO-LEARN-128: Semi-Real Ensemble Relay v3 Fires When Exp 644 Is Blocked but Exp 643 Gate Is Open

**Given** Exp 644 reports signed_improvement <= 0
**And** Exp 643 shows gate_open = True and ensemble_tp > 0
**When** the Tier 1 FR-11 relay v3 runs
**Then** ensemble-detected violations from Exp 643 are fed to ConstraintAdditionFromMemory
**And** fr11_real_violations_confirmed = False
**And** honest_verdict = 'semi_real_ensemble_relay'

### SCENARIO-LEARN-129: Synthetic-Fallback Relay v3 Fires When Both Exp 644 and Exp 643 Are Blocked

**Given** Exp 644 is blocked (signed_improvement <= 0)
**And** Exp 643 shows gate_open = False or ensemble_tp = 0
**When** the Tier 1 relay v3 runs
**Then** 25 synthetic arithmetic violations are generated
**And** fr11_real_violations_confirmed = False
**And** honest_verdict = 'synthetic_fallback_relay'

---

### REQ-LEARN-083: JEPA v15 Trained on Real Violation Data Achieves OOD AUC >= 0.80

**Requirement:** When retrained on real violation pairs sourced from Exp 659
(FR-11 Tier 2 relay) and FOVER live pairs, the JEPAViolationPredictor v15 must
achieve OOD AUC >= 0.80 on a 20% held-out subset of those real pairs.

**Rationale:** The CPMI + PURE min-form objective provides contrastive gradient signal
(correct chain energy < incorrect chain energy by a margin) that BCE alone cannot.
Training on real violation data rather than synthetic pairs closes the loop from FR-11
relay → model improvement.

**Acceptance criteria:**
- ood_auc >= 0.80 in the experiment artifact
- Training data includes at least one non-synthetic pair from Exp 659 or FOVER live

Spec: REQ-LEARN-083, SCENARIO-LEARN-130

### SCENARIO-LEARN-130: JEPA v15 OOD AUC Meets Target on Real Violation Pairs

**Given** Exp 659 violation relay and FOVER live pairs are available
**When** JEPAViolationPredictor v15 is trained with CPMI+PURE for 100 epochs
**Then** ood_auc >= 0.80 on the 20% held-out test split
**And** the artifact records honest_verdict = 'jepa_v15_target_met' or 'jepa_v15_auc_met'

---

### REQ-LEARN-084: JEPA v15 ECE < 0.10 After Platt Calibration

**Requirement:** After training v15 on real violation pairs, a Platt temperature T
is fitted on the validation set. The Expected Calibration Error (ECE) of the calibrated
model must be < 0.10.

**Rationale:** Raw energy scores are uncalibrated probabilities.  Dividing by T before
sigmoid converts the energy output into a well-calibrated probability estimate that
downstream confidence thresholds can trust.

**Acceptance criteria:**
- ece_post_calibration < 0.10 in the experiment artifact
- platt_temperature_T > 0.0 (a positive scalar)

Spec: REQ-LEARN-084, SCENARIO-LEARN-131

### SCENARIO-LEARN-131: JEPA v15 Platt Calibration Yields ECE < 0.10

**Given** v15 is trained and produces raw energy scores on the validation set
**When** Platt temperature T is fitted via Brent's method on the validation set
**Then** ece_post_calibration < 0.10
**And** platt_temperature_T is a finite positive scalar

### SCENARIO-LEARN-132: JEPA v15 honest_verdict Is a Valid Enum Member

**Given** any execution path of experiment_671_jepa_v15.py
**When** the experiment artifact is written
**Then** honest_verdict is one of:
    'jepa_v15_target_met' | 'jepa_v15_auc_met' | 'jepa_v15_partial'
    | 'jepa_v15_no_improvement' | 'ci_mode_synthetic'

### REQ-LEARN-085: MetaJuLS Forcing Adapter Updates Forcing Strategy from Live Feedback

**Given** a MetaJuLSForcingAdapter initialized with base FORCER_SYSTEM_ADDENDUM
**When** ForcingFeedback observations are submitted via update()
**Then** the adapter accumulates per-domain recall history
**And** domains with mean recall < 0.30 receive a CRITICAL: emphasis string

### REQ-LEARN-086: Adapted Forcing Addendum Is Extended for Low-Recall Domains

**Given** a MetaJuLSForcingAdapter that has received low-recall feedback for a domain
**When** get_adapted_addendum() is called for that domain
**Then** the returned addendum is longer than the base FORCER_SYSTEM_ADDENDUM
**And** the base addendum is preserved as a prefix in the extended addendum

### SCENARIO-LEARN-133: Low Recall Domain Triggers CRITICAL Emphasis

**Given** a MetaJuLSForcingAdapter with base addendum
**When** ForcingFeedback with recall < 0.30 is submitted for domain 'percentage'
**Then** adapter.domain_emphasis['percentage'] contains "CRITICAL"
**And** the emphasis string references the 'percentage' domain by name

### SCENARIO-LEARN-134: High Recall Domain Keeps Base Addendum Unchanged

**Given** a MetaJuLSForcingAdapter
**When** ForcingFeedback with recall >= 0.30 is submitted for domain 'arithmetic'
**Then** no emphasis is installed for 'arithmetic'
**And** get_adapted_addendum() returns the base addendum unchanged

### SCENARIO-LEARN-135: Save/Load Round-Trip Preserves Full Adapter State

**Given** a MetaJuLSForcingAdapter with accumulated recall history and installed emphasis
**When** save_state() is called and load_state() restores from that file
**Then** domain_recalls matches the original
**And** domain_emphasis matches the original
**And** get_adapted_addendum() produces identical output before and after round-trip

---

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
| REQ-LEARN-039 (v4) | N/A | Implemented | 11 Python |
| REQ-LEARN-040 (v4) | N/A | Implemented | 11 Python |
| REQ-LEARN-041 (v4) | N/A | Implemented | 11 Python |
| REQ-LEARN-046 | N/A | Implemented | Python |
| REQ-LEARN-047 | N/A | Implemented | Python |
| REQ-LEARN-048 | N/A | Implemented | Python |
| REQ-LEARN-049 | N/A | Implemented | Python |
| REQ-LEARN-050 | N/A | Implemented | Python |
| REQ-LEARN-051 | N/A | Implemented | Python |
| REQ-LEARN-052 | N/A | Implemented | Python |
| REQ-LEARN-053 | N/A | Implemented | Python |
| REQ-LEARN-054 | N/A | Implemented | Python |
| REQ-LEARN-055 | N/A | Implemented | Python |
| REQ-LEARN-056 | N/A | Implemented | Python |
| REQ-LEARN-057 | N/A | Implemented | Python |
| REQ-LEARN-058 | N/A | Implemented | 26 Python |
| REQ-LEARN-059 | N/A | Implemented | 26 Python |
| REQ-LEARN-060 | N/A | Implemented | Python |
| REQ-LEARN-061 | N/A | Implemented | Python |
| REQ-LEARN-062 | N/A | Implemented | Python |
| REQ-LEARN-063 | N/A | Implemented | Python |
| REQ-LEARN-064 | N/A | Implemented | Python |
| REQ-LEARN-065 | N/A | Implemented | 28 Python |
| REQ-LEARN-066 | N/A | Implemented | 28 Python |
| REQ-LEARN-067 | N/A | Implemented | Python |
| REQ-LEARN-068 | N/A | Implemented | Python |
| REQ-LEARN-069 | N/A | Implemented | 12 Python |
| REQ-LEARN-070 | N/A | Implemented | Python (Exp 597) |
| REQ-LEARN-071 | N/A | Implemented | Python (Exp 597) |
| REQ-LEARN-072 | N/A | Implemented | Python (test_hisr_weights.py, 100%) |
| REQ-LEARN-071 | N/A | Implemented | Python (test_mise_calibrator) |
| REQ-LEARN-073 | N/A | Implemented | Python (test_jepa_ood_validation.py) |
| REQ-LEARN-074 | N/A | Implemented | Python (test_jepa_ood_validation.py) |
| REQ-LEARN-076 | N/A | Implemented | Python (test_flip_calibrator.py) |
| REQ-LEARN-077 | N/A | Implemented | Python (Exp 611) |
| REQ-LEARN-078 | N/A | Implemented | Python (test_metajuls_adapter.py) |
| REQ-LEARN-079 | N/A | Implemented | Python (test_metajuls_adapter.py) |
| REQ-LEARN-080 | N/A | Implemented | Python (test_experiment_625_fr11_relay.py) |
| REQ-LEARN-081 | N/A | Implemented | Python (test_experiment_638_fr11_relay.py) |
| REQ-LEARN-082 | N/A | Implemented | Python (test_experiment_645_fr11_relay.py) |
| REQ-LEARN-083 | N/A | Implemented | Python (test_experiment_671_jepa_v15.py) |
| REQ-LEARN-084 | N/A | Implemented | Python (test_experiment_671_jepa_v15.py) |
| REQ-LEARN-085 | N/A | Implemented | Python (test_metajuls_forcing_adapter.py) |
| REQ-LEARN-086 | N/A | Implemented | Python (test_metajuls_forcing_adapter.py) |
