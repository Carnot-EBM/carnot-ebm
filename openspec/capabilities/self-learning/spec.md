# Self-Learning Capability Specification

**Capability:** self-learning
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-11, FR-12

## Overview

Defines how the Carnot pipeline improves its own constraint pool quality through
self-play: generating responses, verifying them, and feeding verified violations
back as training signal.  This file covers PSV-PaCoRe (Parallel Chain with
Reward-shaped Constraints) extensions that increase the diversity of the
violation pool by running multiple chains at different temperatures.

The hypothesis being tested: single-chain PSV saturates after ~10 iterations
because the same 10 questions produce similar errors each round, so the
constraint pool accumulates FP noise instead of diverse violation signal.
Two chains with different generation temperatures produce qualitatively different
errors, giving the constraint learner more varied examples to learn from.

---

## REQ-LEARN-020: PSV-PaCoRe K=2 Parallel Chains Improve Constraint Pool Quality

**Given** a PSV self-play loop that has degraded (fp_rate_trend_slope > 0) due to
         insufficient violation diversity in single-chain mode
**When** two parallel chains (A and B) run on the SAME questions with different
         generation temperatures (temp_A=0.7, temp_B=1.0)
**Then** the combined violation pool from both chains contains more diverse error
         patterns than either chain alone
**And** the fp_rate_trend_slope measured over 10 iterations is lower than the
        single-chain baseline (Exp 697 slope=+0.004242)

### REQ-LEARN-020 Sub-requirements

- REQ-LEARN-020-1: `PSVPaCoReRunner.__init__` SHALL accept
  `inference_fn`, `verify_fn`, `n_iterations`, and `n_questions` parameters.
- REQ-LEARN-020-2: `PSVPaCoReRunner.run_iteration` SHALL call `inference_fn`
  TWICE per question: once with `temp_a` and once with `temp_b`, producing two
  independent responses per question.
- REQ-LEARN-020-3: `PSVPaCoReRunner.run_10_iterations` SHALL run `n_iterations`
  iterations (default 10), accumulating violations from both chains into a
  shared constraint pool list.
- REQ-LEARN-020-4: `IterationResult.fp_rate_estimate` SHALL be computed as
  `n_violations / n_questions` where `n_violations` counts questions where
  BOTH chain A and chain B responses were flagged as violations by `verify_fn`.
  This is the conservative (strict) FP rate estimate.

Spec: REQ-LEARN-020, SCENARIO-LEARN-020

---

## REQ-LEARN-021: Energy-Merge Selects Lower-Violation-Energy Response Per Question

**Given** two candidate responses for the same question (one from chain A, one
         from chain B)
**When** the energy-merge step runs
**Then** the response with LOWER violation energy (fewer constraint violations)
         is selected as the "best response" for that question
**And** if both responses have equal energy, chain A's response is selected
        (tie-break by chain index)

### REQ-LEARN-021 Sub-requirements

- REQ-LEARN-021-1: `PSVPaCoReRunner.run_iteration` SHALL call `verify_fn` on
  both the chain-A response and the chain-B response, producing two boolean
  labels per question.
- REQ-LEARN-021-2: The energy proxy for a response SHALL be `0.0` when
  `verify_fn` returns `True` (no violations detected) and `1.0` when
  `verify_fn` returns `False` (violation detected).
  This binary proxy is sufficient because PSV only produces a binary label.
- REQ-LEARN-021-3: `IterationResult.best_responses` SHALL be a list of
  strings, one per question, containing the lower-energy response (or chain-A
  response on tie).
- REQ-LEARN-021-4: `IterationResult.all_violations` SHALL contain ALL violation
  pairs `(question, response)` from BOTH chains — not just the "best" responses.
  Both chains contribute to the constraint pool regardless of which response
  was selected as "best".

Spec: REQ-LEARN-021, SCENARIO-LEARN-021

---

## SCENARIO-LEARN-020: PaCoRe Energy-Merge Selects Correct Response

**Given** two responses for the same question where chain A is incorrect
         (verify_fn=False) and chain B is correct (verify_fn=True)
**When** `run_iteration` performs the energy-merge step
**Then** chain B's response (correct, energy=0.0) is selected over chain A's
         (incorrect, energy=1.0)
**And** the violation from chain A is still recorded in `all_violations`

---

## SCENARIO-LEARN-021: PaCoRe Violation Pool Contains Both Chain Violations

**Given** n=3 questions where:
         - question 0: chain A correct, chain B incorrect
         - question 1: chain A incorrect, chain B correct
         - question 2: both chains correct
**When** `run_iteration` completes
**Then** `all_violations` contains exactly 2 entries
         (one from chain A on question 1, one from chain B on question 0)
**And** `best_responses` contains 3 strings (one per question)
**And** `fp_rate_estimate` = 0.0 (no question had BOTH chains flagged)

---

## Implementation Status

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-020 | N/A | Implemented | Python (test_experiment_709_psv_pacore_k2.py) |
| REQ-LEARN-021 | N/A | Implemented | Python (test_experiment_709_psv_pacore_k2.py) |

---

## REQ-LEARN-022: FR-11 Tier 2 JEPA v17 Violations Update ConstraintTemplateLibrary

**Given** the JEPA v17 cascade gate is open (cascade_gate_open=True in Exp 705)
**When** Exp 713 runs the FR-11 Tier 2 relay
**Then** all verified OOD violations from the JEPA v17 cascade are extracted
**And** each violation is wired into ViolationPatternLibrary via add_template()
**And** n_patterns_added > 0 and fr11_tier_advancement == 2

### REQ-LEARN-022 Sub-requirements

- REQ-LEARN-022-1: `run_experiment` SHALL detect cascade_gate_open from Exp 705 artifact.
- REQ-LEARN-022-2: When cascade_gate_open=True, source SHALL be "jepa_v17_cascade_violations".
- REQ-LEARN-022-3: Each violation SHALL produce at least one ViolationPatternEntry in the library.
- REQ-LEARN-022-4: honest_verdict SHALL be "fr11_tier2_real_violations" when JEPA source is used.

---

## REQ-LEARN-023: FR-11 Fallback — Exp 694 Qwen Violations When JEPA Cascade Blocked

**Given** the JEPA v17 cascade gate is closed (cascade_gate_open=False in Exp 705)
**When** Exp 713 runs the FR-11 Tier 2 relay
**Then** Exp 694 Qwen3.5-0.8B violation data (signed_improvement=1.0, 200q scale) is used
**And** confirmed true-positive violations from Exp 694 are wired into ViolationPatternLibrary
**And** fr11_tier_advancement == 2 and honest_verdict == "fr11_tier2_fallback_relay"

### REQ-LEARN-023 Sub-requirements

- REQ-LEARN-023-1: `run_experiment` SHALL detect cascade_gate_open=False and select fallback.
- REQ-LEARN-023-2: When fallback, source SHALL be "exp694_qwen_fallback".
- REQ-LEARN-023-3: n_violations SHALL equal the count of confirmed repairs extracted from 694 data.
- REQ-LEARN-023-4: honest_verdict SHALL be "fr11_tier2_fallback_relay" for fallback source.

---

## SCENARIO-LEARN-022: JEPA Gate Open — Real Violations Wired

**Given** cascade_gate_open=True in Exp 705 artifact
**When** run_experiment is called
**Then** source == "jepa_v17_cascade_violations"
**And** honest_verdict == "fr11_tier2_real_violations"
**And** fr11_tier_advancement == 2

---

## SCENARIO-LEARN-023: JEPA Gate Closed — Fallback Relay Wired

**Given** cascade_gate_open=False in Exp 705 artifact
**When** run_experiment is called with fallback_exp694 data
**Then** source == "exp694_qwen_fallback"
**And** honest_verdict == "fr11_tier2_fallback_relay"
**And** fr11_tier_advancement == 2
**And** n_violations > 0

---

## Implementation Status (continued)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-022 | Implemented | Python (test_experiment_713_fr11_tier2_relay.py) |
| REQ-LEARN-023 | Implemented | Python (test_experiment_713_fr11_tier2_relay.py) |

---

## REQ-PSV-005: PSV Question Pool Must Contain >= 100 Questions and Sample Randomly Per Iteration

**Given** a PSV self-play loop running for multiple iterations
**When** the question pool has fewer than 100 questions (e.g. 10 fixed questions)
**Then** the loop overfits to those specific questions after ~10 iterations, causing
         the constraint weights to memorize question-specific surface patterns instead
         of generalizable violation signals, and fp_rate_trend_slope becomes positive
         (degrading) on held-out evaluations

**Given** a PSV self-play loop with a pool of >= 100 questions sampled randomly each iteration
**When** any single iteration samples 10 questions from the 100-question pool
**Then** repeated sampling prevents any individual question from dominating weight updates,
         the constraint weights learn more general violation patterns, and fp_rate_trend_slope
         remains <= 0 (improving or stable) over 20 iterations

**Required behavior:**
- The question pool passed to PSV self-play MUST contain >= 100 distinct questions.
- Each iteration MUST sample a fresh random subset of questions (no fixed pool reuse).
- Violation of this requirement causes monotonically increasing fp_rate_trend_slope
  (confirmed experimentally in Exp 722: condition A slope > 0, condition B slope <= 0).

### SCENARIO-PSV-005: Pool Exhaustion Controlled Experiment

**Given** two PSV conditions run for 20 iterations each:
         - Condition A: fixed 10-question pool (baseline degradation)
         - Condition B: 100-question pool, 10 sampled randomly per iteration
**When** fp_rate_trend_slope is computed via linear regression across 20 iterations
**Then** condition_A_slope > 0 (degrading, pool exhaustion confirmed)
**And** condition_B_slope < condition_A_slope (rotating pool slows or reverses degradation)
**And** honest_verdict is one of:
         - "pool_exhaustion_confirmed": condition_A_slope > 0 AND condition_B_slope < 0
         - "pool_exhaustion_not_confirmed": condition_B_slope also > 0
         - "pool_exhaustion_ambiguous": abs(condition_B_slope) < 0.0001

---

## Implementation Status (PSV-005)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-PSV-005 | Implemented | Python (test_experiment_722_psv_pool_exhaustion.py) |

---

## REQ-FR11-001: FR11EventBus Delivers ViolationEvents to All Subscribers Within 200ms

**Given** a ViolationEvent is emitted by Tier 2.1 JEPAReasonerProbe
**When** FR11EventBus.publish(event) is called
**Then** all registered subscribers receive the event within 200ms wall-clock time
**And** events_acked is incremented for each subscriber that completes without error

### REQ-FR11-001 Sub-requirements

- REQ-FR11-001-1: FR11EventBus.subscribe(fn) SHALL register fn as a subscriber.
- REQ-FR11-001-2: FR11EventBus.publish(event) SHALL call all subscribers synchronously.
- REQ-FR11-001-3: events_acked SHALL equal events_published * len(subscribers) when
  all subscribers complete without error.

---

## REQ-FR11-002: PerModelFPTracker Increments constraint_weight by 0.01 Per Violation

**Given** a ViolationEvent for constraint_type T is received by PerModelFPTracker.on_violation()
**When** the update fires (not throttled)
**Then** constraint_weight[T] is incremented by 0.01
**And** constraint_weight[T] is capped at 2.0

---

## REQ-FR11-003: SessionMemory Caches Violations and Calls observe_pattern After 5

**Given** ViolationEvents for constraint_type T are received by SessionMemory.on_violation()
**When** the cumulative count for T reaches 5
**Then** ConstraintTemplateLibrary.observe_pattern(T, model_id, count) is called
**And** violations_by_type[T] and violations_by_domain[domain] are incremented on each event

---

## REQ-FR11-004: Weight Update Throttle — Max 1 Tier 1 Update Per 10 Queries

**Given** ViolationEvents arrive in bursts
**When** PerModelFPTracker.on_violation() is called
**Then** constraint_weight is only updated when violation_call_count % 10 == 0
**And** all other calls are skipped (logged as throttled)

---

## SCENARIO-FR11-001: FR11EventBus Delivers to Two Subscribers

**Given** FR11EventBus has two subscribers (fp_tracker.on_violation, session_memory.on_violation)
**When** one ViolationEvent is published
**Then** events_published == 1 AND events_acked == 2

---

## SCENARIO-FR11-002: FR11EventBus Latency Under 200ms

**Given** a ViolationEvent with 2 in-memory subscribers
**When** measure_publish_latency_ms() is called
**Then** the returned latency is < 200ms

---

## SCENARIO-FR11-003: SessionMemory Triggers observe_pattern at 5th Violation

**Given** 4 prior violations of type "carry_check" are cached in SessionMemory
**When** a 5th ViolationEvent of type "carry_check" arrives
**Then** ConstraintTemplateLibrary.observe_pattern("carry_check", model_id, 5) is called

---

## SCENARIO-FR11-004: Weight Throttle Skips Updates 1 Through 9

**Given** 10 consecutive ViolationEvents arrive
**When** on_violation() is called 10 times
**Then** only call #10 (count % 10 == 0) updates constraint_weight
**And** calls #1-9 are skipped

---

## Implementation Status (FR-11)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-FR11-001 | Implemented (fr11_event_bus.py) | test_experiment_734_fr11_relay.py |
| REQ-FR11-002 | Implemented (adaptive_thresholds.py) | test_experiment_734_fr11_relay.py |
| REQ-FR11-003 | Implemented (session_memory.py) | test_experiment_734_fr11_relay.py |
| REQ-FR11-004 | Implemented (adaptive_thresholds.py) | test_experiment_734_fr11_relay.py |

---

## REQ-SAMPLE-020: AdaptivePSVSampler Must Stop Early When Energy Variance Converges

**Given** an AdaptivePSVSampler configured with K_min, K_max, and variance_threshold
**When** sample_until_convergent(question) is called
**Then** sampling stops as soon as energy_variance < variance_threshold across current samples
**And** sampling MUST NOT stop before K_min samples have been collected
**And** sampling MUST stop at K_max even if variance has not dropped below threshold

### REQ-SAMPLE-020 Sub-requirements

- REQ-SAMPLE-020-1: `AdaptiveSamplerConfig` SHALL accept `K_min=2`, `K_max=8`, and
  `variance_threshold=0.05` as configurable parameters with these defaults.
- REQ-SAMPLE-020-2: `AdaptivePSVSampler.sample_until_convergent` SHALL collect at least
  `K_min` samples before evaluating the stopping criterion.
- REQ-SAMPLE-020-3: The stopping criterion is `variance(energy_scores) < variance_threshold`
  where variance is population variance of all energy scores collected so far.
- REQ-SAMPLE-020-4: `AdaptiveSampleResult.k_used` SHALL equal the number of samples
  actually collected (between K_min and K_max inclusive).

---

## REQ-SAMPLE-021: sample_reduction_fraction Must Be Computed as 1 - mean_samples / K_max

**Given** N questions processed by AdaptivePSVSampler
**When** sample_reduction_fraction is computed
**Then** sample_reduction_fraction = 1 - mean(k_used_per_question) / K_max
**And** the target efficiency gain is sample_reduction_fraction >= 0.30

### REQ-SAMPLE-021 Sub-requirements

- REQ-SAMPLE-021-1: mean_samples_used SHALL be the arithmetic mean of k_used across all questions.
- REQ-SAMPLE-021-2: sample_reduction_fraction SHALL be in the range [0.0, 1.0].
- REQ-SAMPLE-021-3: When all questions require K_max samples, sample_reduction_fraction == 0.0.
- REQ-SAMPLE-021-4: When all questions stop at K_min, sample_reduction_fraction == 1 - K_min/K_max.

---

## SCENARIO-SAMPLE-030: Adaptive Sampler Stops Early When Variance Below Threshold

**Given** a question where all K_min energy scores are nearly identical (variance < 0.01)
**When** sample_until_convergent is called with variance_threshold=0.05
**Then** k_used == K_min (stopped at minimum)
**And** the returned energy_scores list has K_min entries

---

## SCENARIO-SAMPLE-031: Adaptive Sampler Uses All K_max Samples When Variance Stays High

**Given** a question where energy scores vary widely across all K_max samples (variance > 0.10)
**When** sample_until_convergent is called with variance_threshold=0.05
**Then** k_used == K_max (no early stop)
**And** the returned energy_scores list has K_max entries

---

## Implementation Status (SAMPLE)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-SAMPLE-020 | Implemented (adaptive_psv_sampler.py) | test_experiment_774_adaptive_bayesian_psv.py |
| REQ-SAMPLE-021 | Implemented (adaptive_psv_sampler.py) | test_experiment_774_adaptive_bayesian_psv.py |

---

## REQ-PSV-010: PSV Diagnosis Experiments Must Test At Least 2 Falsifiable Hypotheses

**Given** PSV self-play has degraded for 2+ consecutive milestones
**When** a diagnosis experiment is designed to identify root cause
**Then** the experiment MUST test at least 2 distinct, falsifiable hypotheses with controlled conditions
**And** each hypothesis must have a measurable outcome (slope, rate, or classification metric)
**And** the gate file MUST record which hypothesis was confirmed or that neither was confirmed

Sub-requirements:
- REQ-PSV-010-1: Each condition MUST vary exactly ONE independent variable vs. the control.
- REQ-PSV-010-2: honest_verdict MUST be one of the predefined falsifiable outcomes (not "unknown" unless ALL hypotheses are ruled out).
- REQ-PSV-010-3: The gate file MUST record root_cause_hypothesis so downstream experiments can branch on it.

**Implementation Status:** Implemented (scripts/experiment_736_psv_specialization.py, Exp 736)

---

## REQ-PSV-011: PSV Verifier Training Data Must Include At Least 3 Distinct Domains

**Given** PSV self-play shows increasing fp_rate (specialization degradation pattern)
**When** the verifier is retrained or evaluated
**Then** the training/evaluation data MUST include questions from at least 3 distinct domains
**And** domains must be qualitatively different (e.g., arithmetic, logical, planning — not arithmetic sub-variants)
**And** fp_rate MUST be measured separately per domain so per-domain degradation is detectable

Sub-requirements:
- REQ-PSV-011-1: Domains MUST include at least one non-arithmetic domain (logical or planning).
- REQ-PSV-011-2: The domain pool loader MUST accept a list of (domain_name, question_list) tuples.
- REQ-PSV-011-3: Per-condition slopes MUST be reported in the artifact for each domain condition tested.

**Implementation Status:** Implemented (scripts/experiment_736_psv_specialization.py, Exp 736)

---

## SCENARIO-PSV-010: Exp 736 Condition B Confirms Domain Diversity Hypothesis

**Given** Exp 736 runs Condition B (domain-diverse pool: GSM8K + MATH-Algebra + ARC-Challenge)
**When** fp_rate_slope is computed over 20 iterations for Condition B
**Then** IF condition_b_slope < condition_a_slope THEN gate="pass" with root_cause="constraint_specialization"
**And** the gate file records fix="domain_diversity"

**Spec traces:** REQ-PSV-010, REQ-PSV-011
**Implementation Status:** Implemented (Exp 736)

---

## SCENARIO-PSV-011: Exp 736 Condition C Confirms Verifier Generalization Hypothesis

**Given** Exp 736 runs Condition C (domain-generic verifier weights, held-out GSM8K questions)
**When** fp_rate_slope is computed over 20 iterations for Condition C
**Then** IF condition_c_slope < condition_a_slope THEN gate="pass_verifier" with root_cause="constraint_specialization_verifier"
**And** the gate file records fix="domain_generic_verifier"

**Spec traces:** REQ-PSV-010, REQ-PSV-011
**Implementation Status:** Implemented (Exp 736)

---

---

## REQ-PSV-012: PSV Monitoring MUST Run for at Least 60 Total Iterations to Confirm Recovery Sustainability

**Given** Exp 737 confirmed domain-diverse recovery over 30 iterations (fp_rate_slope = -0.00131257)
**When** the domain-diverse fix is applied for 30 additional iterations (iterations 31-60)
**Then** fp_rate_slope over the new 30 iterations MUST be measured and compared to Exp 737's slope
**And** honest_verdict MUST be one of:
  - "psv_recovery_sustained" — new30 slope < 0 OR abs(new30 slope) < 0.0001
  - "psv_recovery_decelerating" — new30 slope >= 0 AND new30 slope < condition_a slope (slowing but not reversed)
  - "psv_recovery_relapse" — new30 slope > condition_a slope (root cause deeper than specialization)

Sub-requirements:
- REQ-PSV-012-1: The 30 new iterations MUST use the same domain_pool as Exp 737 (GSM8K 10q + MATH-Algebra 5q + ARC-Challenge 5q per iteration).
- REQ-PSV-012-2: fp_rate_slope_all60 MUST be computed over all 60 iterations (Exp 737 fp_rates + new fp_rates).
- REQ-PSV-012-3: The artifact MUST record both fp_rate_slope_new30 and fp_rate_slope_all60 for traceability.

**Implementation Status:** Implemented (scripts/experiment_749_psv_monitoring.py, Exp 749)

---

## SCENARIO-PSV-012: Exp 749 Confirms Recovery is Sustained at 60 Iterations

**Given** Exp 737 produced fp_rate_slope = -0.00131257 over 30 iterations
**When** Exp 749 runs 30 more iterations (31-60) with the same domain-diverse pool
**Then** fp_rate_slope_new30 is computed and compared to fp_rate_slope_737
**And** honest_verdict is determined:
  - "psv_recovery_sustained" if new30 slope < 0 OR abs(new30 slope) < 0.0001
  - "psv_recovery_decelerating" if new30 slope >= 0 AND new30 slope < condition_a slope
  - "psv_recovery_relapse" if new30 slope > condition_a slope

**Spec traces:** REQ-PSV-012
**Implementation Status:** Implemented (Exp 749)

---

## Implementation Status (PSV-010/011/012)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-PSV-010 | Implemented (scripts/experiment_736_psv_specialization.py) | tests/python/test_experiment_736_psv_specialization.py |
| REQ-PSV-011 | Implemented (scripts/experiment_736_psv_specialization.py) | tests/python/test_experiment_736_psv_specialization.py |
| REQ-PSV-012 | Implemented (scripts/experiment_749_psv_monitoring.py) | tests/python/test_experiment_749_psv_monitoring.py |

---

## REQ-PSV-013: PSV Relapse Root Cause MUST Be Identified Via Controlled Test of All Three Hypotheses Before Fix

**Given** PSV self-play has relapsed (fp_rate_slope_new30 > 0) for a third consecutive milestone
**When** three competing architectural hypotheses exist (memory contamination, coupling overwrite, curriculum collapse)
**Then** a controlled diagnostic MUST be run that tests each hypothesis in isolation before applying any fix

Sub-requirements:
- REQ-PSV-013-1: test_hypothesis_a() MUST inject exactly 20% corrupted repairs into memory and run >= 30 self-play steps, measuring fp_rate_slope. Slope > 0 confirms.
- REQ-PSV-013-2: test_hypothesis_b() MUST freeze the coupling matrix before self-play, run >= 30 steps, and measure fp_rate_slope. Slope <= 0 when frozen confirms that self-play overwrite is the cause.
- REQ-PSV-013-3: test_hypothesis_c() MUST use a zero-diversity question set (10 unique questions repeated N times), run >= 30 steps, and measure fp_rate_slope. Slope > 0 confirms curriculum collapse.
- REQ-PSV-013-4: honest_verdict MUST be one of: "hypothesis_a_confirmed", "hypothesis_b_confirmed", "hypothesis_c_confirmed", "multiple_hypotheses", or "diagnosis_inconclusive".
- REQ-PSV-013-5: The artifact MUST record per-hypothesis slopes (hypothesis_a_slope, hypothesis_b_slope, hypothesis_c_slope) and primary_hypothesis for downstream fix selection.

**Implementation Status:** Implemented (scripts/experiment_755_psv_relapse_diagnosis.py, Exp 755)

---

## SCENARIO-PSV-020: Exp 755 Diagnoses PSV Relapse Root Cause From Three Hypotheses

**Given** PSV fp_rate_slope_new30=+0.00110 observed in retro (Exp 753, milestone .57)
**When** PSVDiagnostic runs test_hypothesis_a/b/c on synthetic GSM8K-style data (100q, CPU-only)
**Then**:
  - hypothesis_a_slope > 0 (memory contamination causes deterioration)
  - hypothesis_b_slope <= 0 when coupling frozen (coupling overwrite hypothesis supported)
  - hypothesis_c_slope > 0 (curriculum collapse causes deterioration)
  - primary_hypothesis is set to the single confirmed cause or "multiple_hypotheses"
  - honest_verdict is NOT "diagnosis_inconclusive" unless all three slopes are flat

**Spec traces:** REQ-PSV-013
**Implementation Status:** Implemented (Exp 755)

---

## Implementation Status (PSV-013)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-PSV-013 | Implemented (python/carnot/pipeline/psv_diagnostic.py, scripts/experiment_755_psv_relapse_diagnosis.py) | tests/python/test_experiment_755_psv_diagnosis.py |

---

## REQ-PSV-014: SRSA Memory Gate — SessionMemory.write_with_verification MUST Reject Unverified Repairs

**Given** a repair text is produced by the self-play loop
**When** SessionMemory.write_with_verification(repair_text, constraint_type) is called
**Then** VPRMArithmeticVerifier.detect_violations(repair_text) MUST be called first
**And** if any violations are detected (list non-empty), the repair MUST be discarded (return False)
**And** only if no violations are detected, the repair is accepted (return True, then write)

This closes the SRSA memory contamination pathway: incorrect repairs that survive the LLM
generation step cannot corrupt the session memory pool.

Sub-requirements:
- REQ-PSV-014-1: write_with_verification MUST call detect_violations before any write.
- REQ-PSV-014-2: write_with_verification MUST return False and not call write() when violations are found.
- REQ-PSV-014-3: write_with_verification MUST return True after calling write() when no violations.

**Implementation Status:** Implemented (python/carnot/pipeline/session_memory.py, Exp 756)

---

## REQ-PSV-015: PPSEBM Constraint Freezing — Low-Variance Coupling Entries MUST NOT Be Updated

**Given** a SelfLearningRelay has run at least 30 self-play steps
**When** _freeze_stable_constraints() is called
**Then** for each constraint type, energy variance over the last 30 steps is computed
**And** if variance < FREEZE_THRESHOLD (0.01), that constraint type is marked as frozen
**And** frozen constraints are skipped in PerModelFPTracker.update() during self-play

This prevents self-play's high-LR updates from overwriting CD-learned coupling entries.

Sub-requirements:
- REQ-PSV-015-1: _freeze_stable_constraints() MUST read the last 30 fp_tracker update records.
- REQ-PSV-015-2: FREEZE_THRESHOLD is 0.01 (energy variance units).
- REQ-PSV-015-3: Frozen constraint types are stored in SelfLearningRelay._frozen_constraints set.

**Implementation Status:** Implemented (python/carnot/pipeline/self_learning_relay.py, Exp 756)

---

## REQ-PSV-016: Sustained Recovery REQUIRES fp_rate_slope < 0 in BOTH Windows (Steps 0-30 AND 30-60)

**Given** Exp 756 applies the RETRO-PSV-RELAPSE architectural fix
**When** 60 self-play steps are run and fp_rate is measured at steps 0, 10, 20, 30, 40, 50, 60
**Then** fp_rate_slope MUST be < 0 in BOTH windows:
  - window1 (steps 0-30): slope over first 31 measurements < 0
  - window2 (steps 30-60): slope over last 31 measurements < 0
**And** recovery_sustained = True ONLY when both window slopes are negative
**And** honest_verdict MUST be "recovery_sustained" only when recovery_sustained=True

This is stricter than Exps 697 and 737 which each showed window1 slope < 0 but failed window2.

Sub-requirements:
- REQ-PSV-016-1: Both window slopes MUST be independently computed via OLS.
- REQ-PSV-016-2: "recovery_partial" honest_verdict is used when window1 slope < 0 but window2 >= 0 (old pattern).
- REQ-PSV-016-3: "recovery_failed" is used when both slopes >= 0.

**Implementation Status:** Implemented (scripts/experiment_756_psv_srsa_gate.py, Exp 756)

---

## SCENARIO-PSV-021: write_with_verification Rejects Repair with Arithmetic Error

**Given** a repair text containing "3 plus 4 equals 8" (incorrect: 3+4=7, not 8)
**When** SessionMemory.write_with_verification(repair_text, "addition") is called
**Then** the method returns False (rejected) and no write occurs

**Spec traces:** REQ-PSV-014, REQ-PSV-014-1, REQ-PSV-014-2

---

## SCENARIO-PSV-022: _freeze_stable_constraints Skips Update for Frozen Constraint

**Given** SelfLearningRelay has 30 steps where "verification" type has variance < 0.01
**When** _freeze_stable_constraints() is called and then run_batch() is called
**Then** "verification" constraint type is in _frozen_constraints
**And** fp_tracker.update() is NOT called for frozen constraint types

**Spec traces:** REQ-PSV-015, REQ-PSV-015-1, REQ-PSV-015-2, REQ-PSV-015-3

---

## SCENARIO-PSV-023: recovery_sustained Requires Both Windows Negative

**Given** fp_rate_series of length 61 (steps 0 through 60)
**When** window1_slope (steps 0-30) = -0.001 AND window2_slope (steps 30-60) = +0.002
**Then** recovery_sustained = False (window2 is non-negative)
**And** honest_verdict = "recovery_partial" (old pattern from Exp 697/737)

**Given** window1_slope = -0.001 AND window2_slope = -0.0005
**Then** recovery_sustained = True
**And** honest_verdict = "recovery_sustained"

**Spec traces:** REQ-PSV-016, REQ-PSV-016-1, REQ-PSV-016-2

---

## Implementation Status (PSV-014/015/016)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-PSV-014 | Implemented (python/carnot/pipeline/session_memory.py) | tests/python/test_experiment_756_psv_srsa_gate.py |
| REQ-PSV-015 | Implemented (python/carnot/pipeline/self_learning_relay.py) | tests/python/test_experiment_756_psv_srsa_gate.py |
| REQ-PSV-016 | Implemented (scripts/experiment_756_psv_srsa_gate.py) | tests/python/test_experiment_756_psv_srsa_gate.py |

---

## FR-11 Formal Closure — Implementation Status: OPERATIONAL

**Status:** OPERATIONAL as of Milestone 2026.04.56 (2026-04-22).

FR-11 (Autonomous Self-Learning Loop) is formally closed.  The requirement is
satisfied end-to-end: violation events from Tier 2.1 (JEPAReasonerProbe) are
relayed through FR11EventBus to Tier 1 (PerModelFPTracker) and Tier 2
(SessionMemory), and SessionMemory state persists across sessions.

**Closing experiments:**
- Exp 734 (fr11_relay_operational=True): FR11EventBus delivers ViolationEvents
  to all subscribers within 200ms; relay_events_acked >= 1 (confirmed 100 events).
- Exp 738 (fr11_tier2_relay_functional=True): SessionMemory cross-session persist
  confirmed; templates_replayed_in_s2=1, precision improves across sessions.
- Exp 732: 5-fold CV probe AUC = 0.993 (signal quality gate for FR-11 Tier 2.1).

**Evidence summary:**

| Metric | Value | Source |
|--------|-------|--------|
| relay_events_acked | 100 | Exp 734 |
| relay_latency_p99_ms | < 200 | Exp 734 |
| fr11_relay_operational | True | Exp 734 |
| fr11_tier2_relay_functional | True | Exp 738 |
| templates_replayed_in_s2 | >= 1 | Exp 738 |
| probe_5fold_auc | 0.993 | Exp 732 |

**Formal closure certificate:** results/fr11_closure_certificate.json
**Closure experiment:** scripts/experiment_741_fr11_formal_closure.py

---

## REQ-FR11-005: SessionMemory MUST Persist Violation Type Mappings Across Sessions

**Given** a SessionMemory that has accumulated violation_type counts during a pipeline run
**When** SessionMemory.persist(filepath) is called at session_end
**Then** a JSON file is written at filepath containing violation_type → count mappings
  AND template_keys listing which types crossed the observation threshold (count >= 5)
**When** SessionMemory.load(filepath, template_library) is called at next session_start
**Then** ConstraintTemplateLibrary.replay_template(key, model_id) is called for each template_key
  AND the template is immediately active for that model without re-accumulating violations

Sub-requirements:
- REQ-FR11-005-1: persist() SHALL write schema "carnot.session_memory_relay.v1".
- REQ-FR11-005-2: load() SHALL return the number of templates replayed (0 on missing file).
- REQ-FR11-005-3: load() SHALL never raise — missing or corrupt files return 0 silently.
- REQ-FR11-005-4: replay_template() SHALL set observation count >= min_frequency for that template/model pair.

**Implementation Status:** Implemented (python/carnot/pipeline/session_memory.py, python/carnot/pipeline/constraint_template_library.py, Exp 738)

### SCENARIO-FR11-005: Cross-Session Template Replay Fires in S2

Given: Session S1 accumulates 5 carry_check violations and calls persist().
When: Session S2 calls load() and then calls get_active_templates(model_id).
Then:
  - "carry_check" template is in the active list.
  - templates_replayed == 1.
  - No S2 violations needed to re-activate the template.

**Spec traces:** REQ-FR11-005
**Implementation Status:** Implemented (Exp 738)

---

## REQ-FR11-006: 3-Session Simulation MUST Show Monotonically Non-Decreasing Precision

**Given** sessions S1, S2, S3 run the same 20 questions through the cascade with FR-11 relay
**When** cross-session memory is active (persist at S1 end, load at S2 start; persist at S2 end, load at S3 start)
**Then** precision_s1 <= precision_s2 <= precision_s3 (non-decreasing)
  OR fr11_tier2_relay_functional = True if any template from S1 fires in S2

Sub-requirements:
- REQ-FR11-006-1: At least one template from S1 SHALL be replayed in S2 (templates_replayed_in_s2 > 0).
- REQ-FR11-006-2: Precision is measured as fraction of flagged queries that are true violations.
- REQ-FR11-006-3: fr11_tier2_relay_functional SHALL be True when templates_replayed_in_s2 > 0.

**Implementation Status:** Implemented (scripts/experiment_738_step_probe_tier2_memory.py, Exp 738)

### SCENARIO-FR11-006: 3-Session Precision Monotonically Improves

Given: S1 runs 20q and persists carry_check (5 violations).
When: S2 loads the persisted state and runs same 20q.
Then:
  - templates_replayed_in_s2 >= 1.
  - precision_s2 >= precision_s1.
  - After S2 persist + S3 load: precision_s3 >= precision_s2.

**Spec traces:** REQ-FR11-006
**Implementation Status:** Implemented (Exp 738)

---

## REQ-FR11-007: PerModelFPTracker MUST Expose get_weight_state() for Convergence Auditing

**Given** PerModelFPTracker has processed at least one ViolationEvent via on_violation()
**When** get_weight_state() is called
**Then** it returns a dict mapping each constraint_type that has been seen to a WeightState
  with weight (float), update_count (int), and last_updated_at (ISO-8601 timestamp).
  Constraint types with zero updates are excluded from the result.

Sub-requirements:
- REQ-FR11-007-1: get_weight_state() SHALL return a dict[str, WeightState].
- REQ-FR11-007-2: WeightState SHALL include weight, update_count, and last_updated_at.
- REQ-FR11-007-3: Constraint types that were throttled (not updated) SHALL appear only if
  they have at least one un-throttled update (update_count > 0).

**Implementation Status:** Implemented (python/carnot/pipeline/adaptive_thresholds.py, Exp 747)

### SCENARIO-FR11-007: get_weight_state Returns All Observed Constraint Types

Given: 30 arithmetic + 15 logical + 5 code ViolationEvents injected via on_violation().
When: get_weight_state() is called after all events.
Then:
  - All three constraint types appear in the result.
  - arithmetic weight > 1.0 (most updates).
  - Each WeightState has update_count >= 1.

**Spec traces:** REQ-FR11-007
**Implementation Status:** Implemented (Exp 747)

---

## REQ-FR11-008: Weight Convergence Audit MUST Run After >= 50 Relay Events

**Given** the FR-11 relay has been operational and processed at least 50 ViolationEvents
**When** get_weight_state() is called and weight_ratio = max_weight / min_weight is computed
**Then** weight_ratio >= 2.0 indicates the relay is discriminating between constraint types
  (arithmetic should be highest for GSM8K domain).

Sub-requirements:
- REQ-FR11-008-1: Audit SHALL classify verdict as one of: tier1_weights_converging,
  tier1_weights_uniform, tier1_weights_inverted, tier1_weights_no_data.
- REQ-FR11-008-2: tier1_weights_converging requires update_count_ratio
  (max_update_count / min_update_count) >= 2.0 AND arithmetic_weight > logical_weight.
  Note: weight_ratio (max_weight / min_weight) is reported separately but is not used
  for the verdict because weights increment by 0.01 and are bounded at 2.0 — with 50
  events at 10x throttle the weight ratio is at most ~1.02, making 2.0 unachievable.
  update_count_ratio reflects the true discrimination signal.
- REQ-FR11-008-3: disabled_constraints list SHALL include any constraint_type with weight < 0.02.

**Implementation Status:** Implemented (scripts/experiment_747_tier1_weight_audit.py, Exp 747)

### SCENARIO-FR11-008: Convergence Audit Classifies Weight Distribution Correctly

Given: 30 arithmetic + 15 logical + 5 code events injected (expected: arithmetic > logical > code).
When: audit runs with weight_ratio = max_weight / min_weight.
Then:
  - honest_verdict == "tier1_weights_converging".
  - arithmetic_weight > logical_weight (expected_ordering_correct == True).
  - disabled_constraints is empty (no weight near zero).

**Spec traces:** REQ-FR11-008
**Implementation Status:** Implemented (Exp 747)

---

## Implementation Status (FR11-005/006/007/008)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-FR11-005 | Implemented (session_memory.py, constraint_template_library.py) | tests/python/test_experiment_738_step_probe_memory.py |
| REQ-FR11-006 | Implemented (scripts/experiment_738_step_probe_tier2_memory.py) | tests/python/test_experiment_738_step_probe_memory.py |
| REQ-FR11-007 | Implemented (adaptive_thresholds.py) | tests/python/test_experiment_747_weight_audit.py |
| REQ-FR11-008 | Implemented (scripts/experiment_747_tier1_weight_audit.py) | tests/python/test_experiment_747_weight_audit.py |

---

## REQ-FR11-009: 10-Session Memory Stress Test MUST Show Non-Decreasing Precision or Report Plateau

**Given** a 10-session cross-session memory simulation with 20 questions per session (200q total)
**When** each session Si loads persisted state from S(i-1) via SessionMemory.load_relay() and then
  runs 20 questions through the cascade with FR-11 fully wired, then calls SessionMemory.persist()
**Then** the precision series [precision_s1, ..., precision_s10] is monotonically non-decreasing
  OR the first session where improvement plateaus (delta < 0.001) is reported as plateau_session
  AND the system MUST NOT show regression (any session < prior session by > 0.01)

Sub-requirements:
- REQ-FR11-009-1: The experiment SHALL measure precision at sessions 1, 3, 5, 7, 10.
- REQ-FR11-009-2: is_monotonically_non_decreasing SHALL be True when all(s[i] >= s[i-1]).
- REQ-FR11-009-3: plateau_session SHALL be the first session index where delta < 0.001
  (None when precision is still rising at S10).
- REQ-FR11-009-4: honest_verdict SHALL be one of: "tier2_memory_monotonic_gain",
  "tier2_memory_plateau_at_s{N}", or "tier2_memory_regression".

**Implementation Status:** Implemented (scripts/experiment_748_cross_session_memory_10session.py, Exp 748)

---

## REQ-FR11-010: Templates Replayed Count MUST Be > 0 For Sessions S2 Through S10

**Given** a multi-session simulation with cross-session memory active
**When** each session Si (i >= 2) calls SessionMemory.load_relay() at session start
**Then** templates_replayed_in_si > 0 for every session from S2 to S10
  (at least one template from the prior session is immediately active)

Sub-requirements:
- REQ-FR11-010-1: The cumulative templates_replayed list SHALL have length 9 (one per S2..S10).
- REQ-FR11-010-2: Each entry in templates_replayed_per_session SHALL be >= 1.
- REQ-FR11-010-3: total_templates_at_s10 SHALL be the ConstraintTemplateLibrary total template count after S10.

**Implementation Status:** Implemented (scripts/experiment_748_cross_session_memory_10session.py, Exp 748)

---

## SCENARIO-FR11-009: 10-Session Precision Is Non-Decreasing

**Given** 10 sessions run with cross-session memory, 20q each, different question slices
**When** precision is computed as fraction of truly-violated constraints detected correctly
**Then** precision_s10 >= precision_s1 (overall improvement from start to end)
**And** is_monotonically_non_decreasing is True OR plateau_session is reported (both are valid wins)

**Spec traces:** REQ-FR11-009
**Implementation Status:** Implemented (Exp 748)

---

## SCENARIO-FR11-010: Templates Replayed in Every Session S2-S10

**Given** Session S1 accumulates at least 5 violations of one constraint type and calls persist()
**When** Sessions S2 through S10 each call load_relay() at start
**Then** templates_replayed > 0 in every subsequent session (pre-warmed constraint knowledge active)

**Spec traces:** REQ-FR11-010
**Implementation Status:** Implemented (Exp 748)

---

## Implementation Status (FR11-009/010)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-FR11-009 | Implemented (scripts/experiment_748_cross_session_memory_10session.py) | tests/python/test_experiment_748_10session_memory.py |
| REQ-FR11-010 | Implemented (scripts/experiment_748_cross_session_memory_10session.py) | tests/python/test_experiment_748_10session_memory.py |

---

## REQ-LEARN-040: ConstraintAdditionEngine MUST Generate New Constraints From Session Memory

**Given** a SessionMemory instance that has accumulated ≥3 instances of the same violation pattern
**When** ConstraintAdditionEngine.scan_for_patterns() is called
**Then** patterns with count >= min_count (default 3) are returned as ConstraintPattern objects
**And** generate_constraint(pattern) returns a ConstraintTerm for each recognised violation_type
**And** inject_into_pipeline(pipeline) appends each generated ConstraintTerm to
         pipeline.active_constraints before the next session starts

### REQ-LEARN-040 Sub-requirements

- REQ-LEARN-040-1: `ConstraintAdditionEngine.__init__` SHALL accept `session_memory` and
  optional `min_count=3` parameters.
- REQ-LEARN-040-2: `scan_for_patterns()` SHALL read `_violations_by_type` from `session_memory`
  and return `ConstraintPattern` objects where count >= min_count.
- REQ-LEARN-040-3: `generate_constraint(pattern)` SHALL return a `ConstraintTerm` instance for
  `carry_error`, `sign_error`, `unit_error`, and `comparison_error` violation types.
- REQ-LEARN-040-4: `inject_into_pipeline(pipeline)` SHALL NOT add duplicate constraints
  (checked by ConstraintTerm name) and SHALL return the count of newly injected constraints.

---

## REQ-LEARN-041: Precision Across 10 Sessions MUST Be Monotonically Non-Decreasing

**Given** a 10-session run with ConstraintAdditionEngine injecting constraints after each session
**When** precision is measured per session as fraction of actual violations correctly detected
**Then** precision_s10 >= precision_s1 (no regression from first to last session)
**And** the `monotonic_non_decreasing` flag in the artifact is True when no session-to-session
         regression occurs

### REQ-LEARN-041 Sub-requirements

- REQ-LEARN-041-1: `monotonic_non_decreasing` SHALL be True when every session's precision
  is >= the previous session's precision (allowing equal).
- REQ-LEARN-041-2: `precision_s1` and `precision_s10` SHALL be float values in [0.0, 1.0].
- REQ-LEARN-041-3: Any precision regression (session i+1 < session i) is a gate failure
  for the `constraint_addition_works` verdict.

---

## SCENARIO-LEARN-080: Patterns Accumulate to Threshold and Trigger Injection

**Given** a SessionMemory with `_violations_by_type = {"carry_error": 5, "sign_error": 2}`
**When** ConstraintAdditionEngine(session_memory, min_count=3).scan_for_patterns() is called
**Then** exactly one ConstraintPattern is returned (carry_error, count=5)
**And** sign_error is excluded because count=2 < min_count=3

---

## SCENARIO-LEARN-081: Duplicate Injection Is Prevented

**Given** a pipeline with `active_constraints` already containing a carry_check constraint
**When** inject_into_pipeline is called again with the same session memory
**Then** the carry_check constraint is NOT added a second time
**And** inject_into_pipeline returns 0 (zero new injections)

---

## Implementation Status (REQ-LEARN-040/041)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-040 | Pending | Pending |
| REQ-LEARN-041 | Pending | Pending |

---

## REQ-LEARN-042: PPSConstraintSelector MUST Compute Energy Variance Per Coupling

**Given** a self-play adaptation loop that updates coupling weights J_(ij) in the
         VerifyRepairPipeline after each question
**When** the PPSConstraintSelector is active
**Then** ``CouplingVarianceTracker`` MUST compute the energy variance of each coupling's
         contribution over a rolling window of W=30 questions
**And** couplings with variance < FREEZE_THRESHOLD (0.01) MUST NOT be updated during
        self-play adaptation (their gradient entries are zeroed before the weight update)

### REQ-LEARN-042 Sub-requirements

- REQ-LEARN-042-1: ``CouplingVarianceTracker.__init__`` SHALL accept ``n_couplings``
  and ``window_size=30`` parameters.
- REQ-LEARN-042-2: ``CouplingVarianceTracker.update(coupling_contributions)`` SHALL
  record the energy contribution of each coupling for one question into a rolling
  deque of length ``window_size``.
- REQ-LEARN-042-3: ``CouplingVarianceTracker.get_variance()`` SHALL return a 1-D
  float64 array of population variance per coupling over the current window.
  Couplings with fewer than 2 observations SHALL return variance 0.0 (frozen by default).
- REQ-LEARN-042-4: ``CouplingVarianceTracker.get_frozen_mask(freeze_threshold=0.01)``
  SHALL return a boolean array where True = coupling is frozen (variance < threshold).
- REQ-LEARN-042-5: ``PPSConstraintSelector.apply_mask(gradient)`` SHALL zero all
  gradient entries corresponding to frozen couplings and return a new array (not
  mutate the input).
- REQ-LEARN-042-6: ``PPSConstraintSelector.frozen_count()`` SHALL return the integer
  count of currently frozen couplings.

---

## SCENARIO-LEARN-082: PPS Freezing Prevents Gradient Updates on Settled Couplings

**Given** a CouplingVarianceTracker with n_couplings=3 and window_size=30
**And** 10 updates have been recorded where coupling 0 is always 1.0 (variance=0)
        and coupling 1 alternates 0/1 (variance=0.25)
**When** PPSConstraintSelector.apply_mask([5.0, 3.0, -2.0]) is called with threshold=0.01
**Then** the returned gradient has 0.0 at index 0 (frozen: variance < 0.01)
**And** the returned gradient has 3.0 at index 1 (not frozen: variance >= 0.01)
**And** the input gradient array is not mutated

---

## Implementation Status (REQ-LEARN-042)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-042 | Implemented (python/carnot/pipeline/pps_constraint_selector.py) | Implemented (tests/python/test_experiment_762_ppsebm_constraint_select.py) |

---

## REQ-PROBE-010: DualPathwayProbe MUST Include Question-Anchored and Answer-Anchored Sub-Probes

**Given** a FoVer v2 labeled dataset of (question_context, step_text, label) triples
**When** MixtureOfProbes is instantiated and trained
**Then** it MUST include a QuestionAnchoredProbe (2-layer MLP on question-side tokens)
**And** it MUST include an AnswerAnchoredProbe (2-layer MLP on answer-side tokens)
**And** the GateNetwork MUST be a 2-layer MLP trained jointly with both sub-probes

### REQ-PROBE-010 Sub-requirements

- REQ-PROBE-010-1: QuestionAnchoredProbe SHALL accept hidden_dim=128 and output_dim=1.
- REQ-PROBE-010-2: AnswerAnchoredProbe SHALL accept hidden_dim=128 and output_dim=1.
- REQ-PROBE-010-3: GateNetwork SHALL accept input_dim=2, output_dim=1.
- REQ-PROBE-010-4: MixtureOfProbes.train() SHALL train all three components end-to-end
  via a single Adam optimizer and BCELoss.

---

## REQ-PROBE-011: Dual-Pathway AUROC on FoVer v2 Test Split MUST Be Reported

**Given** a trained MixtureOfProbes evaluated on the held-out 20% test split of FoVer v2
**When** MixtureOfProbes.evaluate_auroc() is called on test predictions and labels
**Then** the AUROC value MUST be recorded in the experiment artifact
**And** if AUROC >= 0.993, the dual-pathway probe supersedes single-pathway JEPAReasonerProbe
**And** a confidence interval caveat MUST be noted when test_set_size < 20

---

## SCENARIO-PROBE-020: Mixture-of-Probes Trains Without Error

**Given** 45 labeled FoVer v2 steps (80% train split)
**When** MixtureOfProbes.train(labeled_steps, n_epochs=100) is called
**Then** training completes without raising an exception
**And** the returned dict contains "final_loss" (float) and "n_train" (int == 45)

---

## SCENARIO-PROBE-021: Dual-Pathway AUROC Is Computed and Reported

**Given** a trained MixtureOfProbes and a 12-sample test split
**When** MixtureOfProbes.evaluate_auroc(scores, labels) is called
**Then** the result is a float in [0.0, 1.0]
**And** the experiment artifact contains auroc, precision, recall, honest_verdict

---

## Implementation Status (REQ-PROBE-010, REQ-PROBE-011)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-PROBE-010 | Implemented (python/carnot/pipeline/dual_pathway_probe.py) | Implemented (tests/python/test_experiment_763_dual_pathway_probe.py) |
| REQ-PROBE-011 | Implemented (scripts/experiment_763_dual_pathway_probe.py) | Implemented (tests/python/test_experiment_763_dual_pathway_probe.py) |

---

## REQ-LEARN-043: JEPA v19 MUST Train Exclusively on Real Accumulated Violation Data

JEPA v19 MUST be trained exclusively on real accumulated violation data from live GPU
experiments (Exps 742, 759, 760, and fover_labeled_steps_live).  No synthetic data
injection is permitted.  The experiment artifact MUST list data_sources and n_real_pairs.

---

## REQ-LEARN-044: JEPA v19 OOD AUC MUST Exceed 0.75 for Tier 3 Deployment

**Given** JEPA v19 trained on real accumulated violation data
**When** evaluated on OOD questions (GSM8K 800-999, never seen in training data)
**Then** the OOD AUC MUST exceed 0.75 for Tier 3 deployment eligibility
**Otherwise** a new RETRO-JEPA-OOD retrospective MUST be filed

---

## SCENARIO-LEARN-085: JEPA v19 Trains on Real Labeled Steps

**Given** real labeled FoVer steps (step_text, label) from live GPU experiments
**When** MultiStepJEPAv19.train(step_sequences, labels, n_epochs=200) is called
**Then** training completes without error
**And** the final training loss is a finite positive float

---

## SCENARIO-LEARN-086: JEPA v19 OOD AUC Reported in Artifact

**Given** a trained MultiStepJEPAv19 and OOD evaluation set (GSM8K 800-999 range)
**When** the experiment artifact is written
**Then** it MUST contain ood_auc, in_dist_auc, n_real_pairs, data_sources, honest_verdict
**And** honest_verdict MUST be one of: jepa_v19_ood_viable, jepa_v19_improving,
  jepa_v19_still_below_random, jepa_v19_insufficient_data

---

---

## REQ-LEARN-045: MultiStepJEPAv19 MUST Pool TF-IDF Embeddings Across n_steps=3 Steps

MultiStepJEPAv19 MUST pool TF-IDF embeddings across exactly n_steps=3 consecutive CoT
steps for each training sample.  Fewer steps are zero-padded; extra steps beyond n_steps
are ignored.  The pooling operation is element-wise max across the step dimension.

---

## SCENARIO-LEARN-087: MultiStepJEPAv19 Pools Exactly n_steps Step Embeddings

**Given** a MultiStepJEPAv19 with n_steps=3
**When** forward(steps) is called with 1, 2, or 3 step strings
**Then** the pooled embedding always has length equal to vocab_size
**And** steps fewer than n_steps are zero-padded before pooling

---

## Implementation Status (REQ-LEARN-043, REQ-LEARN-044, REQ-LEARN-045)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-043 | Implemented (python/carnot/samplers/jepa_v19.py) | Implemented (tests/python/test_experiment_770_jepa_v19_predictive.py) |
| REQ-LEARN-044 | Implemented (scripts/experiment_770_jepa_v19_predictive.py) | Implemented (tests/python/test_experiment_770_jepa_v19_predictive.py) |
| REQ-LEARN-045 | Implemented (python/carnot/samplers/jepa_v19.py) | Implemented (tests/python/test_experiment_770_jepa_v19_predictive.py) |

---

## REQ-EBRM-001: EBRM MUST Model Energy as E(response, reward) via 2-Layer MLP

EBRM (arXiv 2504.13134) implementation MUST model energy as E(response, reward) using a
2-layer MLP trained with margin loss (max(0, margin - (E_neg - E_pos))).
MUST be compared to EORM on the same FoVer v2 test split.

---

## REQ-EBRM-002: EORM vs EBRM Comparison MUST Report AUC and Architectural Advantage

The EORM vs EBRM comparison MUST report in-distribution AUC for both models.
If EORM > EBRM on step-level tasks, log architectural_advantage=True.
Honest verdict MUST be one of: eorm_outperforms_ebrm, ebrm_competitive,
ebrm_outperforms_eorm, insufficient_data.

---

## SCENARIO-EBRM-001: EBRM Trains on FoVer Steps and Reports AUC

**Given** 57 FoVer labeled CoT steps from results/fover_labeled_steps_live.json
**When** EBRMEnergy is trained for 200 epochs on the 80% training split
**Then** AUROC is computed on the 20% test split
**And** the artifact contains ebrm_auc, eorm_auc, auroc_delta, architectural_advantage

---

## SCENARIO-EBRM-002: Architectural Advantage Logged When EORM Outperforms EBRM

**Given** eorm_auc (from Exp 732) and ebrm_auc (computed on FoVer test split)
**When** eorm_auc > ebrm_auc
**Then** architectural_advantage MUST be True in the result artifact
**And** honest_verdict MUST be eorm_outperforms_ebrm or ebrm_competitive

---

## Implementation Status (REQ-EBRM-001, REQ-EBRM-002)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-EBRM-001 | Implemented (python/carnot/pipeline/ebrm_baseline.py) | Implemented (tests/python/test_experiment_771_ebrm_comparison.py) |
| REQ-EBRM-002 | Implemented (scripts/experiment_771_ebrm_comparison.py) | Implemented (tests/python/test_experiment_771_ebrm_comparison.py) |

---

## REQ-LEARN-048: JEPA v20 Live Data Collection MUST Use kill_gpu_zombies() Before Model Load

Live data collection for JEPA v20 training corpus expansion MUST call
``kill_gpu_zombies(gpu_index=0)`` before any model load when ``CARNOT_FORCE_LIVE=1``.
The labeled pairs MUST be written to ``results/fover_labeled_steps_live_v2.json``
WITHOUT modifying ``results/fover_labeled_steps_live.json`` (the Exp 442 baseline file).

---

## REQ-LEARN-049: JEPA v20 Live Data Collection MUST Report n_labeled Honestly

The artifact MUST report ``n_labeled`` as the exact count of labeled pairs written
to ``fover_labeled_steps_live_v2.json``.
``labeling_rate = n_labeled / n_steps_found`` MUST be computed and reported (0.0 when
``n_steps_found == 0``).  The target is ``n_labeled >= 80`` new real pairs; honest
under-reporting is required when the target is not met.

---

## SCENARIO-LEARN-092: kill_gpu_zombies() Called Before Model Load

**Given** ``CARNOT_FORCE_LIVE=1`` is set in the environment
**When** Exp 781 runs
**Then** ``kill_gpu_zombies(gpu_index=0)`` MUST be called before any model load
**And** the call order MUST be: apply_env_autofix → kill_gpu_zombies → setup_gpu

---

## SCENARIO-LEARN-093: Labeled Pairs Written to Separate v2 File

**Given** 100 GSM8K questions answered by Qwen3.5-0.8B
**When** FOVERAnnotator annotates the responses
**Then** labeled pairs MUST be written to ``results/fover_labeled_steps_live_v2.json``
**And** ``results/fover_labeled_steps_live.json`` MUST NOT be modified
**And** honest_verdict MUST be ``"real_data_collected_sufficient"`` iff
  ``n_labeled >= 80`` AND ``inference_mode="live_gpu"``

---

## Implementation Status (REQ-LEARN-048, REQ-LEARN-049)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-048 | Implemented (scripts/experiment_781_jepa_v20_data_collection.py) | Implemented (tests/python/test_experiment_781_jepa_v20_data_collection.py) |
| REQ-LEARN-049 | Implemented (scripts/experiment_781_jepa_v20_data_collection.py) | Implemented (tests/python/test_experiment_781_jepa_v20_data_collection.py) |

---

## REQ-LEARN-050: EDUPRMStepSelector MUST Compute Bootstrap Variance for Step Selection

``EDUPRMStepSelector`` MUST compute prediction variance over ``N_BOOTSTRAP=10``
bootstrap samples of a TF-IDF (128-dim) + LogisticRegression classifier.
MUST select the top 30% of steps by variance (highest-uncertainty steps first).
Bootstrap classifiers MUST be trained on resamples with replacement; predictions
recorded on the FULL corpus.

---

## REQ-LEARN-051: EDUPRMStepSelector MUST Write Selected Corpus and Report Metrics

The selected corpus MUST be written to ``results/fover_edu_prm_selected.json``
in the same format as ``fover_labeled_steps_live.json``.
The artifact MUST report ``uncertainty_selected_pct`` (fraction of corpus selected)
and ``diversity_score`` (positive_fraction of selected labels).
``diversity_delta=True`` MUST be reported when the selected set is more balanced
(closer to 0.5) than uniform selection.

---

## SCENARIO-LEARN-094: EDU-PRM Selects Top 30% Highest-Variance Steps

**Given** a pooled FoVer corpus of N labeled steps
**When** ``EDUPRMStepSelector.select()`` is called
**Then** exactly ``ceil(N * 0.30)`` steps MUST be returned
**And** each returned index MUST have variance >= every non-returned index

---

## SCENARIO-LEARN-095: Diversity Score Near 0.5 Indicates Balanced Hard Examples

**Given** a set of selected step labels
**When** ``diversity_score(selected_labels)`` is computed
**Then** it MUST equal ``sum(label==1) / len(labels)``
**And** ``diversity_delta=True`` when ``|diversity_score - 0.5| < |uniform_diversity - 0.5|``

---

## Implementation Status (REQ-LEARN-050, REQ-LEARN-051)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-050 | Implemented (python/carnot/pipeline/edu_prm_selector.py) | Implemented (tests/python/test_experiment_782_edu_prm_step_selection.py) |
| REQ-LEARN-051 | Implemented (scripts/experiment_782_edu_prm_step_selection.py) | Implemented (tests/python/test_experiment_782_edu_prm_step_selection.py) |

---

## REQ-LEARN-052: JEPA v20 MUST Train Exclusively on Real Accumulated Data with EDU-PRM Preference

JEPA v20 MUST be trained exclusively on real accumulated data from live GPU experiments.
EDU-PRM selected corpus (``fover_edu_prm_selected.json``) MUST be preferred over uniform
sampling when available; ``data_source`` MUST be ``"edu_prm_selected"`` in that case.
Synthetic data injection is NOT allowed under any circumstances.
Class-weight balancing (``weight_positive = n_negative / n_positive`` in BCE loss) MUST
be applied to correct for potential label imbalance in the selected corpus.

---

## REQ-LEARN-053: JEPA v20 OOD AUC MUST Exceed 0.75 to Gate Tier 3.5 Deployment

JEPA v20 OOD AUC MUST exceed 0.75 to unlock Tier 3.5 deployment in Exp 784.
``ood_auc_delta_vs_v19`` MUST equal ``ood_auc - 0.5667`` (v19 baseline).
If OOD AUC <= 0.75, RETRO-JEPA-OOD-V20 MUST be filed.
``honest_verdict`` MUST be one of:
  - ``"jepa_v20_ood_viable"`` if ood_auc > 0.75
  - ``"jepa_v20_improving"`` if 0.60 < ood_auc <= 0.75
  - ``"jepa_v20_below_v19"`` if ood_auc <= 0.5667
  - ``"jepa_v20_insufficient_data"`` if n_training_pairs < 30

---

## SCENARIO-LEARN-096: JEPA v20 Uses EDU-PRM Corpus When Available

**Given** ``results/fover_edu_prm_selected.json`` exists
**When** Exp 783 collects training data
**Then** ``data_source`` MUST equal ``"edu_prm_selected"``
**And** ``n_training_pairs`` MUST equal the number of items in that file

---

## SCENARIO-LEARN-097: JEPA v20 Class Weighting Corrects Label Imbalance in BCE

**Given** a training corpus with n_positive positives and n_negative negatives
**When** ``MultiStepJEPAv20.train()`` computes BCE loss
**Then** each positive example's loss MUST be multiplied by ``n_negative / n_positive``
**And** ``class_weight_used`` MUST be ``True`` in the artifact

---

## Implementation Status (REQ-LEARN-052, REQ-LEARN-053)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-052 | Implemented (python/carnot/samplers/jepa_v20.py, scripts/experiment_783_jepa_v20_retrain.py) | Implemented (tests/python/test_experiment_783_jepa_v20_retrain.py) |
| REQ-LEARN-053 | Implemented (scripts/experiment_783_jepa_v20_retrain.py) | Implemented (tests/python/test_experiment_783_jepa_v20_retrain.py) |
