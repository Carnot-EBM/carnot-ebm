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

---

## REQ-FR11-020: LagrangeAdaptiveIsingConstraints MUST Implement Violation-Driven Lambda Weight Updates

**Given** a set of binary Ising constraints (spin pairs with required alignment)
**When** LagrangeAdaptiveIsingConstraints.run_session() is called
**Then** for each constraint k with non-zero violation rate, lambda_k MUST increase by
         lambda_lr * avg_violation_rate_k
**And** the updated lambdas are used in the NEXT session's build_J() call, so violated
        constraints receive stronger coupling weights and the energy landscape shifts
        toward constraint satisfaction over multiple sessions

### REQ-FR11-020 Sub-requirements

- REQ-FR11-020-1: `LagrangeAdaptiveIsingConstraints.__init__` SHALL accept `n_spins`,
  `n_constraints`, `lambda_init=1.0`, and `lambda_lr=0.1` parameters.
- REQ-FR11-020-2: `build_J(constraints)` SHALL construct the coupling matrix as
  J_base + sum_k lambda_k * sign_k * penalty_k * delta(i_k, j_k) for each constraint k.
- REQ-FR11-020-3: `run_session(constraints)` SHALL call InertiaIsingSampler, compute
  per-constraint violation rates, update lambdas, and return violation_rate and lambdas.
- REQ-FR11-020-4: `_constraint_violation(samples, constraint)` SHALL return a float in
  [0, 1] representing the fraction of samples that violate the given constraint.

**Implementation Status:** Implemented (python/carnot/samplers/lagrange_adaptive.py, Exp 862)

---

## SCENARIO-FR11-030: 5-Session Relay on Synthetic Constraints Shows delta_s1_to_s5 > 0

**Given** 10 synthetic binary constraints on 10 spins with adversarial J_base
         (first 5 constraints have negative base coupling biased against satisfaction)
**When** LagrangeAdaptiveIsingConstraints runs 5 sessions with lambda_lr=0.2
**Then** delta_s1_to_s5 = violation_rate[session_1] - violation_rate[session_5] > 0
         (violation rate decreases from session 1 to session 5)
**And** fr11_self_learning_confirmed = True

**Experimental result (Exp 862):**
  - violation_rates: [0.42, 0.54, 0.50, 0.39, 0.37]
  - delta_s1_to_s5: 0.05 (positive — self-learning confirmed)
  - honest_verdict: fr11_self_learning_confirmed

**Spec traces:** REQ-FR11-020
**Implementation Status:** Implemented (scripts/experiment_862_lagrange_adaptive_ising.py, Exp 862)

---

## Implementation Status (FR11-020)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-FR11-020 | Implemented (python/carnot/samplers/lagrange_adaptive.py) | tests/python/test_experiment_862_lagrange_adaptive_ising.py |

---

## REQ-FR11-1130: Post-Retrain Zenil Alpha Measurement MUST Compare Against Exp1077 Baseline

**Given** Exp 1077 established a SOTA-tier Zenil alpha_t baseline of 0.38 for
Qwen3.6-35B-A3B
**And** Exp 1120 retrained SOSKANEnergyV3 on the FoVer v5 SOTA corpus with
validation AUROC 0.977419 and corrected energy ordering
**When** Exp 1130 scores 50 SOTA model outputs with the retrained energy
verifier
**Then** the artifact SHALL include alpha_t_prior, alpha_t_post_retrain,
alpha_t_improved, verifier_auroc_used, n_evaluation_examples, inference_mode,
fr11_self_learning_data_point_logged, zenil_alpha_t_post_retrain_measured, and
honest_verdict
**And** honest_verdict SHALL be one of "alpha_t_improved",
"alpha_t_unchanged", "alpha_t_degraded", or "measurement_incomplete".

### REQ-FR11-1130 Sub-requirements

- REQ-FR11-1130-1: alpha_t_prior SHALL equal 0.38 from Exp 1077.
- REQ-FR11-1130-2: verifier_auroc_used SHALL equal the Exp 1120
  retrained_auroc_val when the Exp 1120 artifact is available.
- REQ-FR11-1130-3: alpha_t_improved SHALL be true iff
  alpha_t_post_retrain > alpha_t_prior.
- REQ-FR11-1130-4: inference_mode SHALL be "live_gpu" for new Qwen3.6-35B-A3B
  generations or "cached" for cached SOTA outputs.
- REQ-FR11-1130-5: zenil_alpha_t_post_retrain_measured SHALL be true whenever
  the artifact is written.

### SCENARIO-FR11-1130: Improved Post-Retrain Alpha Is Classified Correctly

**Given** alpha_t_prior = 0.38 and alpha_t_post_retrain = 0.52
**When** the Exp 1130 artifact payload is built
**Then** alpha_t_improved is true
**And** honest_verdict is "alpha_t_improved".

### SCENARIO-FR11-1131: Cached Measurement Remains Honest

**Given** live Qwen3.6-35B-A3B generation is unavailable
**When** Exp 1130 falls back to cached SOTA rows
**Then** inference_mode is "cached"
**And** n_evaluation_examples records the number of scored cached examples
**And** zenil_alpha_t_post_retrain_measured is true.

**Spec traces:** REQ-FR11-1130, SCENARIO-FR11-1130, SCENARIO-FR11-1131
**Implementation Status:** Implemented (scripts/experiment_1130_zenil_alpha_t_post_retrain.py, Exp 1130)

---

## REQ-LEARN-056: IsingConstraintGenerator MUST Read Error Patterns and Synthesise Coupling Rows

**Given** a list of ErrorPattern objects accumulated from session memory
**When** IsingConstraintGenerator.synthesize_from_memory() is called
**Then** it MUST return a CouplingRow for each pattern whose count >= PATTERN_THRESHOLD (3)
**And**  it MUST return an empty list when all pattern counts < PATTERN_THRESHOLD
**And**  patterns with unknown pattern_type MUST be silently skipped

Supported pattern_types: carry_error, sign_error, unit_error, comparison_error, overflow_error.
Each maps to a deterministic (var1, var2, J_value) coupling spec.

---

## REQ-LEARN-057: Dynamic Constraint Pipeline MUST Report n_constraints_added and constraint_addition_delta

**Given** a 50-question experiment comparing a static-constraint baseline vs a dynamic-constraint pipeline
**When** the dynamic pipeline uses IsingConstraintGenerator to inject couplings after each 10-question batch
**Then** the artifact MUST include n_constraints_added, net_improvement_dynamic,
         net_improvement_static, constraint_addition_delta, n_patterns_above_threshold,
         and honest_verdict
**And**  honest_verdict MUST be one of: constraint_addition_positive, constraint_addition_zero,
         constraint_addition_negative, insufficient_patterns

---

## SCENARIO-LEARN-100: Carry Error Pattern Above Threshold Produces CouplingRow

**Given** a single ErrorPattern(pattern_type="carry_error", count=3, example_step="...")
**When** IsingConstraintGenerator(ising_model, threshold=3).synthesize_from_memory([pattern]) is called
**Then** it returns a list of length 1
**And**  the CouplingRow has J_value=-1.0 and both var indices within [0, input_dim)

---

## SCENARIO-LEARN-101: inject_couplings Extends J Matrix Without Replacing Existing Couplings

**Given** an IsingModel with a pre-existing coupling matrix
**When** IsingConstraintGenerator.inject_couplings([CouplingRow(var1=0, var2=1, J_value=-1.0)]) is called
**Then** coupling[0, 1] MUST equal original_value + (-1.0)
**And**  all other coupling cells MUST be unchanged

---

## REQ-FR11-030: ThreeTierPipeline Multi-Session Relay

**Given** a ThreeTierPipeline instance with StreamingCoTHalluDetector (Tier 0g),
         HalluSAEGeometricProbe (Tier 0i), and LagrangeAdaptiveIsingConstraints (Tier 3)
         wired via the wire_tier_0g(), wire_tier_0i(), and wire_lagrange() methods
**When** the pipeline runs a 5-session relay on 50 synthetic CoT problems per session
**Then** measurable improvement is observed in either AUC (delta_auc_s1_to_s5 > 0)
         OR Lagrange violation rate reduction (delta_violations_s1_to_s5 > 0)
**And** tier2_relay_confirmed is True when either condition is met

### REQ-FR11-030 Sub-requirements

- REQ-FR11-030-1: ThreeTierPipeline.wire_tier_0g(detector) SHALL attach a
  StreamingCoTHalluDetector so verify_extended() runs rolling PHaS scoring per CoT step.
- REQ-FR11-030-2: ThreeTierPipeline.wire_tier_0i(probe) SHALL attach a
  HalluSAEGeometricProbe so verify_extended() returns geometric_energy and hallusae_anomalous.
- REQ-FR11-030-3: ThreeTierPipeline.wire_lagrange(adaptive) SHALL attach a
  LagrangeAdaptiveIsingConstraints so run_lagrange_session() updates lambda weights
  after each session.
- REQ-FR11-030-4: ThreeTierPipeline.verify_extended() SHALL return a dict with keys:
  verified, tier_used, energy, streaming_cot_unstable, geometric_energy, hallusae_anomalous.
- REQ-FR11-030-5: When no probe is wired, verify_extended() SHALL default
  streaming_cot_unstable=False, geometric_energy=0.0, hallusae_anomalous=False.

### SCENARIO-FR11-040: 5-Session Relay Confirms FR-11 Self-Learning

**Given** a ThreeTierPipeline with all three tiers wired
**When** run_relay() executes 5 sessions of 50 synthetic CoT problems each
**Then** session_aucs is a list of 5 floats
**And** session_violation_rates is a list of 5 floats
**And** tier2_relay_confirmed reflects delta_auc_s1_to_s5 > 0 OR delta_violations_s1_to_s5 > 0

---

## Implementation Status (REQ-LEARN-056, REQ-LEARN-057)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-056 | Implemented (python/carnot/pipeline/constraint_generator.py, IsingConstraintGenerator) | Implemented (tests/python/test_experiment_788_constraint_addition_from_memory.py) |
| REQ-LEARN-057 | Implemented (scripts/experiment_788_constraint_addition_from_memory.py) | Implemented (tests/python/test_experiment_788_constraint_addition_from_memory.py) |

---

## REQ-LEARN-058: Lagrange + CompressedMemory Relay MUST Demonstrate Sustained Precision Improvement

**Given** a SelfLearningRelay wired with LagrangeAdaptiveIsing and CompressedMemoryBank
**When** the relay runs 5 sessions of 20 synthetic GSM8K-style questions each
**Then** the enhanced relay MUST show is_monotonically_non_decreasing=True for session precisions
**And**  precision_s5 > precision_s1 (sustained improvement, not plateau)
**And**  lagrange_delta_improvement > 0 (Lagrange provides improvement over baseline)
**And**  honest_verdict MUST be one of: fr11_tier2_loop_closed, tier2_monotone_no_improvement,
         tier2_plateau_at_s2, below_baseline

### REQ-LEARN-058 Sub-requirements

- REQ-LEARN-058-1: LagrangeAdaptiveIsing.update(constraint_id, violated) SHALL increase
  lambda by lambda_lr when violated=True and decrease by 0.1 * lambda_lr when violated=False,
  floored at lambda_init.
- REQ-LEARN-058-2: CompressedMemoryBank.compress_session(session_violations) SHALL maintain
  at most k centroid representatives and record latency for compression_overhead_ms.
- REQ-LEARN-058-3: SelfLearningRelay SHALL accept optional lagrange_ising and compressed_memory
  kwargs and call lagrange_ising.update() per question and compressed_memory.compress_session()
  per batch when these are provided.

---

## SCENARIO-LEARN-102: Enhanced Relay Precision Schedule is Monotonically Non-Decreasing

**Given** a 5-session relay with LagrangeAdaptiveIsing + CompressedMemoryBank
**When** enhanced_session_precisions = [0.60, 0.65, 0.70, 0.75, 0.80]
**Then** is_monotonically_non_decreasing is True
**And**  precision_s5 = 0.80 > precision_s1 = 0.60
**And**  honest_verdict = fr11_tier2_loop_closed

---

## SCENARIO-LEARN-103: Flat Precision Sequence Returns tier2_monotone_no_improvement

**Given** a relay where enhanced_session_precisions = [0.60, 0.60, 0.60, 0.60, 0.60]
**When** compute_honest_verdict() is called
**Then** honest_verdict = tier2_monotone_no_improvement
**And**  fr11_tier2_loop_closed = False

---

## Implementation Status (REQ-LEARN-058)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-058 | Implemented (python/carnot/verify/lagrange_ising.py, python/carnot/pipeline/memory_compression.py, python/carnot/pipeline/self_learning_relay.py, scripts/experiment_875_fr11_tier2_relay_v6.py) | Implemented (tests/python/test_experiment_875_fr11_tier2_relay_v6.py) |

---

## REQ-LEARN-059: Tier 3 VJEPA to Tier 1 Constraint Addition Relay

**Given** a SelfLearningRelay instantiated with a VariationalJEPAPredictor AND a
         ConstraintAdditionEngine
**When** Ising detects a violation on a question (ground_truth=False)
**And**  VJEPA.predict() returns violation_probability > 0.70 for that question
**Then** ConstraintAdditionEngine.add_from_violation(violation_type, step_id) is called
**And**  SelfLearningRelay.tier3_to_tier1_fired is set to True
**And**  SelfLearningRelay.n_vjepa_triggered_additions is incremented

### REQ-LEARN-059 Sub-requirements

- REQ-LEARN-059-1: ConstraintAdditionEngine.add_from_violation(violation_type, step_id)
  SHALL increment session_memory._violations_by_type[violation_type] by 1 and return True.
  Returns False if session_memory has no _violations_by_type attribute.
- REQ-LEARN-059-2: SelfLearningRelay.__init__ SHALL accept an optional
  vjepa_predictor: VariationalJEPAPredictor | None parameter (default None).
- REQ-LEARN-059-3: SelfLearningRelay.run_batch() SHALL check ising_violation_detected
  (ground_truth=False) AND vjepa_predictor is not None AND constraint_addition_engine
  is not None before calling VJEPA.predict().
- REQ-LEARN-059-4: SelfLearningRelay.n_vjepa_triggered_additions SHALL be a read-only
  property returning the cumulative count of VJEPA-triggered additions.
- REQ-LEARN-059-5: SelfLearningRelay.tier3_to_tier1_fired SHALL be a read-only
  property that is False before any VJEPA trigger fires and True thereafter.

## SCENARIO-LEARN-099: Tier 3 Loop Closed

**Given** a 5-session relay with VJEPA always-trigger stub (prob=0.80 for all violations)
**And**  ENHANCED_PRECISIONS schedule [0.60, 0.70, 0.75, 0.80, 0.85]
**When** run_experiment() completes
**Then** honest_verdict = fr11_tier3_loop_closed
**And**  tier3_to_tier1_fired = True
**And**  n_vjepa_triggered_additions > 0
**And**  tier3_to_tier1_relay_confirmed = True (enhanced_s5 > baseline_s5)

## SCENARIO-LEARN-100: Tier 3 Never Fires Without VJEPA

**Given** a 5-session relay with NO vjepa_predictor wired in
**When** run_relay(use_vjepa=False) completes
**Then** n_vjepa_triggered_additions == 0
**And**  tier3_to_tier1_fired = False

---

## Implementation Status (REQ-LEARN-059)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-059 | Implemented (python/carnot/pipeline/constraint_addition_engine.py, python/carnot/pipeline/self_learning_relay.py, scripts/experiment_888_fr11_tier3_relay.py) | Implemented (tests/python/test_experiment_888_fr11_tier3_relay.py) |


---

## REQ-LEARN-1405: SessionMemory Portable Pack Schema

**Given** a saved SessionMemory state containing CaseMemory, ConstraintTemplateLibrary,
         and PerModelFPTracker data
**When** the state is exported as a portable memory pack
**Then** the pack MUST validate against `carnot.session_memory_pack.v1`
**And**  it MUST include pack metadata, schema_version, source, license, carnot_version,
         model_id, case_memory entries, constraint template observations,
         false-positive tracker statistics, and session_state summaries
**And**  schema-breaking changes MUST require a major schema_version increment.

### REQ-LEARN-1405 Sub-requirements

- REQ-LEARN-1405-1: `python/carnot/schemas/session_memory_v1.json` SHALL describe the
  portable pack using JSON Schema draft-2020-12.
- REQ-LEARN-1405-2: `export_session_memory()` SHALL produce deterministic JSON-compatible
  payloads with schema `carnot.session_memory_pack.v1`.
- REQ-LEARN-1405-3: `validate_session_memory_pack()` SHALL reject payloads missing required
  pack metadata or per-model learning-state sections.

---

## REQ-LEARN-1406: SessionMemory Portable Import and Merge

**Given** an existing local SessionMemory state and an imported portable pack for the same model
**When** `import_session_memory(..., merge=True)` is called
**Then** existing case entries MUST NOT be clobbered
**And** duplicate case entries MUST be merged by adding support counts and recomputing
        confidence as a support-weighted average
**And** template observation counts and FP tracker counts MUST be merged additively
**And** `replace=True` MUST perform a full reset using the imported pack contents.

### REQ-LEARN-1406 Sub-requirements

- REQ-LEARN-1406-1: `diff_session_memory_packs()` SHALL report an empty diff after
  export -> import -> export round-trips with no semantic state change.
- REQ-LEARN-1406-2: `import_session_memory(..., dry_run=True)` SHALL validate and report
  planned changes without writing SessionMemory files.
- REQ-LEARN-1406-3: merge and replace modes SHALL be mutually exclusive.

---

## REQ-LEARN-1407: SessionMemory CLI Import/Export

**Given** the installed `carnot` CLI
**When** a user runs `carnot memory export --storage-dir DIR --model-id MODEL -o pack.json`
**Then** the output pack MUST validate against the portable SessionMemory schema
**And** `carnot memory import pack.json --storage-dir DIR --model-id MODEL --merge`
        MUST merge without clobbering existing entries
**And** `carnot memory import ... --replace` MUST print an explicit reset warning before
        replacing local state.

### SCENARIO-LEARN-1405: Portable Pack Round Trip Has Empty Diff

**Given** a SessionMemory state with one CaseMemory entry, one template observation,
         and one FP tracker statistic
**When** it is exported, imported into a fresh storage directory, and exported again
**Then** `diff_session_memory_packs()` returns `is_empty=True`.

### SCENARIO-LEARN-1406: Merge Recomputes Duplicate Case Confidence

**Given** a local case entry with support=1 and confidence=0.25
**And**  an imported duplicate case entry with support=1 and confidence=0.75
**When** the pack is imported with merge=True
**Then** the merged case entry has support=2 and confidence=0.50.

### SCENARIO-LEARN-1407: CLI Memory Commands Route to Portable Pack APIs

**Given** a saved SessionMemory state
**When** `carnot memory export`, `carnot memory import --merge`, and
         `carnot memory diff` are invoked
**Then** each command returns exit code 0 and reports schema-valid pack handling.

---

## Implementation Status (REQ-LEARN-1405, REQ-LEARN-1406, REQ-LEARN-1407)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1405 | Implemented (python/carnot/pipeline/session_memory_pack.py, python/carnot/schemas/session_memory_v1.json) | Implemented (tests/python/test_session_memory_pack.py) |
| REQ-LEARN-1406 | Implemented (python/carnot/pipeline/session_memory_pack.py) | Implemented (tests/python/test_session_memory_pack.py) |
| REQ-LEARN-1407 | Implemented (python/carnot/cli.py memory subcommands) | Implemented (tests/python/test_session_memory_pack.py) |

---

## REQ-FR11-007: Forgetting Curve for Constraint Memory Lifecycle

**Statement:** LagrangeAdaptiveUpdater MUST apply an exponential forgetting curve to
constraint weights at each tick step: w_t = w_0 * exp(-forgetting_lambda * age_t).
When a constraint's weight falls below 1e-4, it MUST be removed from the active
constraint set.  Before expiry, constraints with weight < 0.1 AND violation_rate >
replay_threshold (default 0.8) MUST be identified as replay candidates (FOREVER-style
memory rehearsal) to prevent catastrophic forgetting of constraints that are still
actively relevant.

**Why this matters:**
    Without forgetting, the constraint memory grows unboundedly across sessions.  Early
    constraints from the first few questions dominate the energy landscape even when
    those question types never recur.  Exponential forgetting with FOREVER-style replay
    creates a natural constraint lifecycle (learn → strengthen → forget → replay → expire)
    that keeps the active constraint set small and relevant.

**Acceptance criteria:**
- `tick()` applies exp(-forgetting_lambda) decay to every constraint weight.
- Constraints with weight < 1e-4 are removed from constraint_weights and constraint_ages.
- `get_replay_candidates()` returns IDs with weight < 0.1 AND violation_rate > 0.8.
- `constraint_precision` = fraction of active constraints with violation_rate >= 0.1.
- Over a 10-session relay (20q/session), forgetting_lambda=0.05 produces higher
  constraint_precision than the no-forgetting baseline.

**Spec traces:** Exp 897, arXiv 2601.03938 (FOREVER)

---

## SCENARIO-FR11-007: Aged Constraints Decay; High-Violation Constraints Replayed

**Given** a LagrangeAdaptiveUpdater with forgetting_lambda=0.05, replay_threshold=0.8
**And** a constraint "c1" with weight=0.05 (aging out) and violation_rate=0.9 (still active)
**And** a constraint "c2" with weight=0.05 (aging out) and violation_rate=0.1 (stale)
**When** `get_replay_candidates()` is called
**Then** "c1" is returned (high violation rate — replay it)
**And** "c2" is NOT returned (low violation rate — let it expire)

**Given** a LagrangeAdaptiveUpdater with forgetting_lambda=0.05 and 5 constraints aged 100 steps
**When** `tick(100)` is called on constraints whose initial weight was 1.0
**Then** all 5 constraints are expired (weight = exp(-5.0) ≈ 0.0067 → still above 1e-4 at 100 steps
         with lambda=0.05; but at 200 steps weight = exp(-10) ≈ 4.5e-5 < 1e-4, so they expire)

**Spec traces:** REQ-FR11-007

---

## REQ-FR11-008: KAN Adaptive Spline Restructuring Based on Activation Density

**Statement:** KANAdaptiveStructure MUST analyse per-spline activation histograms
accumulated during a training corpus pass and restructure the KAN grid as follows:
- Splines whose top-2 histogram bins exceed 30% of total activations (high density)
  MUST have their num_knots doubled (clamped to 64).
- Splines whose bottom-2 histogram bins exceed 60% of total activations (low density)
  MUST have their num_knots halved (clamped to minimum 3).
- All other splines remain unchanged.
After restructuring, existing control points MUST be linearly interpolated onto the
new knot count so the model fine-tunes from a meaningful initialisation.

**Why this matters:**
    A fixed KAN grid allocates the same resolution everywhere regardless of where
    activations actually land during inference.  Adaptive restructuring focuses
    parameters on high-traffic regions and reclaims parameters from low-traffic
    regions, which should improve energy discrimination per parameter — the
    efficiency metric that matters for Phase 2 hardware acceleration.

**Acceptance criteria:**
- `KANEnergyFunction.get_activation_density()` returns normalized histograms (sum=1)
  for every spline that received at least 1 activation.
- `KANAdaptiveStructure.analyze()` returns per-spline dicts with keys "density"
  (one of "high"/"low"/"neutral") and "knot_count".
- `KANAdaptiveStructure.restructure()` doubles num_knots for high-density splines
  and halves for low-density splines without mutating the original KAN.
- `KANAdaptiveStructure.evaluate_benefit()` returns energy_loss_before,
  energy_loss_after, delta, knot_count_before, knot_count_after,
  knot_count_change_pct.
- Seed experiment (Exp 898) confirms tier4_viable = (energy_loss_after < energy_loss_before).

**Spec traces:** Exp 898, FR-11 Tier 4

---

---

## REQ-SELF-007: Lagrange Forgetting Curve Reduces Weight Staleness via Entropy Preservation

**Given** a LagrangeAdaptiveUpdater processing a 100-step violation sequence
         where violation rate drops from 0.7 (steps 1-50) to 0.1 (steps 51-100)
**When** the updater uses forgetting_lambda > 0 (exponential weight decay)
**Then** weight_entropy at step 100 is higher than the no-decay baseline
**And** the signed entropy improvement exceeds 0.5 nats

### REQ-SELF-007 Sub-requirements

- REQ-SELF-007-1: `LagrangeAdaptiveUpdater.weight_entropy` SHALL return Shannon
  entropy H = -sum(p * log(p)) of the normalized weight distribution.
- REQ-SELF-007-2: `weight_entropy` SHALL return 0.0 when no constraints are active.
- REQ-SELF-007-3: With forgetting_lambda=0 (baseline), entropy SHALL collapse
  below the forgetting variant at step 100 of the high-to-low violation simulation.
- REQ-SELF-007-4: The forgetting variant SHALL achieve `weight_entropy` > 0.5
  at step 100, confirming weight diversity is preserved.

**Acceptance criteria (Exp 909):**
- `weight_entropy` property added to LagrangeAdaptiveUpdater.
- 100-step synthetic simulation confirms signed_entropy_improvement > 0 for
  forgetting_lambda=0.05 vs baseline.
- honest_verdict is "forgetting_curve_improves_entropy" if improvement > 0.5.

**Spec traces:** Exp 909, FR-11 Tier 1

---

## SCENARIO-SELF-007: Weight Entropy Stays Above Floor After 100 Update Steps

**Given** 100 constraint update steps divided into a high-violation phase (steps 1-50,
         violation_prob=0.7) followed by a low-violation phase (steps 51-100,
         violation_prob=0.1)
**When** LagrangeAdaptiveUpdater with forgetting_lambda=0.05 is used
**Then** weight_entropy at step 100 is at least 0.5 nats
**And** weight_entropy at step 100 (with decay) > weight_entropy at step 100 (no decay)

---

## SCENARIO-FR11-008: Restructured KAN Has Lower Energy Loss Than Original

**Given** a KANModel trained for 100 epochs on 50 FoVer pairs
**And** KANAdaptiveStructure.analyze() classifies at least one spline as high/low
**When** KANAdaptiveStructure.restructure() is called and the result is fine-tuned
         for 20 additional epochs on the same 50 training pairs
**Then** KANAdaptiveStructure.evaluate_benefit() returns delta < 0 (energy_loss improves)
**And** the honest_verdict is "tier4_viable_seed"

**Spec traces:** REQ-FR11-008

---

## REQ-SELF-008: KAN AutoKnots Structural Adaptation via Activation Magnitude

**Given** a trained KANModel and a batch of input activations
**When** AutoKnotsRefiner.refine_once() is called
**Then** splines with mean activation magnitude > high_threshold gain one knot
         (if below max_knots_per_spline)
**And** splines with mean activation magnitude < low_threshold lose one knot
         (if above min_knots_per_spline)
**And** control points are linearly resampled onto the new knot count so the
         model can fine-tune from a meaningful initialisation

### REQ-SELF-008 Sub-requirements

- REQ-SELF-008-1: `AutoKnotsRefiner.__init__` SHALL accept `kan_model`,
  `high_activation_threshold` (default 0.8), `low_activation_threshold`
  (default 0.1), `max_knots_per_spline` (default 32),
  `min_knots_per_spline` (default 4).
- REQ-SELF-008-2: `AutoKnotsRefiner.refine_once(activation_batch)` SHALL
  iterate over all edge splines and bias splines, mutate the model in-place,
  and return a `RefinementResult(n_added, n_removed, splines_modified)`.
- REQ-SELF-008-3: `AutoKnotsRefiner.multi_round_refine(activation_batch,
  rounds=3)` SHALL return a list of `RefinementResult`, one per round.
- REQ-SELF-008-4: After refinement, `BSpline.num_knots` MUST remain within
  [min_knots_per_spline, max_knots_per_spline].
- REQ-SELF-008-5: Resized control points MUST be linearly interpolated from
  the original control points (no random reinitialisation).

**Acceptance criteria (Exp 910):**
- AutoKnotsRefiner on 50 synthetic GSM8K embeddings modifies at least one spline.
- honest_verdict is "tier4_seed_viable" if post_refinement_auc > baseline_auc,
  "tier4_seed_no_improvement" otherwise.

**Spec traces:** Exp 910, FR-11 Tier 4

---

## SCENARIO-SELF-008: AutoKnots Adds Knots to High-Activation, Removes from Dormant

**Given** a KANModel with num_knots=8 and an activation batch
**When** AutoKnotsRefiner.refine_once() is called with high_activation_threshold=0.5
         and low_activation_threshold=0.05
**Then** at least one spline with mean |activation| > 0.5 gains a knot
**And** at least one spline with mean |activation| < 0.05 loses a knot (if any exist
         above min_knots)
**And** the total knot count change equals (n_added - n_removed)

**Spec traces:** REQ-SELF-008

---

## REQ-LEARN-060: Tier 2 Code Domain CaseMemory MUST Accumulate Repair Patterns and Replay Cross-Session

**Given** Exp 905 produced 17 repaired HumanEval problems with n_retries and
         energy_score_best per problem
**When** Session 1 loads those outcomes into ConstraintTemplateLibrary via
         load_exp905_patterns() and session1_populate_library()
**Then** at least 3 distinct code repair templates are added (one per retry-difficulty
         category: easy_fix, medium_fix, hard_fix)
**And** every added template is active for the source model immediately after Session 1
**And** the library can be serialized to disk and reloaded with observation counts intact
**And** Session 2 can replay active templates on at least 1 new problem (HumanEval/25-34)
**And** honest_verdict is 'tier2_code_memory_works' when templates_replayed_in_s2 >= 1
         and replay_improvement > 0

### REQ-LEARN-060 Sub-requirements

- REQ-LEARN-060-1: `load_exp905_patterns(source_path)` SHALL return only problems
  where baseline_passed=False and repair_passed=True from the Exp 905 JSON.
- REQ-LEARN-060-2: `session1_populate_library(patterns)` SHALL register exactly one
  ConstraintTemplate per distinct error_category and observe_pattern() once per
  problem of that category.
- REQ-LEARN-060-3: After session1_populate_library(), `library.get_active_templates(model_id)`
  SHALL return a non-empty list for the source model.
- REQ-LEARN-060-4: `library.to_dict()` / `ConstraintTemplateLibrary.from_dict()` round-trip
  SHALL preserve all observation counts.
- REQ-LEARN-060-5: `session2_replay_templates(library_dict, problem_ids)` SHALL return
  templates_replayed_in_s2 >= 1 and replay_improvement > 0 when active templates exist.

**Acceptance criteria (Exp 935):**
- n_templates_added >= 3.
- cross_session_persistence_verified = True.
- templates_replayed_in_s2 >= 1.
- honest_verdict is 'tier2_code_memory_works'.

**Spec traces:** Exp 935, FR-11 Tier 2

---

## SCENARIO-LEARN-104: Code Repair Template Accumulation and Cross-Session Replay

**Given** Exp 905 results containing 17 failed-then-repaired HumanEval problems
**When** Session 1 of Exp 935 runs session1_populate_library() on those patterns
**Then** three ConstraintTemplates are registered (easy_fix, medium_fix, hard_fix)
**And** each template is immediately active for 'google/gemma-4-E4B-it'
**And** library.to_dict() produces a dict with 3 observation entries

**When** the dict is serialized to disk and reloaded via from_dict()
**Then** the reloaded library has the same observation counts

**When** Session 2 calls session2_replay_templates() on HumanEval/25-34
**Then** templates_replayed_in_s2 equals len(REPLAY_PROBLEMS) because all templates
         are active and code repair templates are advisory (always emit a hint)
**And** replay_improvement > 0
**And** honest_verdict is 'tier2_code_memory_works'

**Spec traces:** REQ-LEARN-060

---

## REQ-LEARN-1146: GRPO Reflection Reward Uses One-Step Repair Energy Delta

**Given** the Exp 1146 GRPO self-learning loop samples a completion for a GSM8K
question
**When** the reward for that completion is computed
**Then** the loop SHALL compute `r_total = r_thinkprm + 0.3 * r_reflect`
**And** `r_reflect` SHALL be `(E_before - E_after) / E_before` when
`E_before > 0`, otherwise `0.0`
**And** `r_reflect` SHALL be clipped to `[-1.0, 1.0]`
**And** `E_after` SHALL be measured after exactly one Carnot verifier-guided
repair attempt, not a full multi-step repair loop
**And** the experiment artifact SHALL record `reflection_reward_integrated=True`,
`reflection_weight=0.3`, `n_repair_steps_per_completion=1`,
`alpha_t_at_training=0.52`, and `fr11_self_learning_signal_used=True`.

### REQ-LEARN-1146 Sub-requirements

- REQ-LEARN-1146-1: Exp 1146 SHALL block honestly with
  `honest_verdict="blocked_no_dualgpu"` when `torch.cuda.device_count() < 2`
  or CUDA is unavailable.
- REQ-LEARN-1146-2: Exp 1146 SHALL use GSM8K train indices `[600, 700)` and
  evaluation indices `[750, 800)` to avoid Exp 1129's `[500, 600)` training
  slice.
- REQ-LEARN-1146-3: Exp 1146 SHALL retain DRA-GRPO diversity penalty and CPPO
  proxy reuse from Exp 1129 while replacing the score used for advantages with
  the repair-grounded total reward.
- REQ-LEARN-1146-4: Exp 1146 SHALL map outcomes to one of
  `reflection_positive_above_0851`, `positive_below_exp1129`, `neutral`,
  `negative_regression`, or `blocked_no_dualgpu`.

### SCENARIO-LEARN-1146: One-Step Repair Lowers Energy and Raises Reward

**Given** a completion with `E_before=4.0`
**And** Carnot verifier feedback produces a one-step repaired response with
`E_after=1.0`
**When** Exp 1146 computes the reflection reward
**Then** `r_reflect` equals `0.75`
**And** a ThinkPRM score of `0.40` produces `r_total=0.625` with
`reflection_weight=0.3`
**And** the repair generator is invoked once for that completion.

### SCENARIO-LEARN-1147: DualGPU Blocker Artifact Is Honest

**Given** CUDA is unavailable or fewer than two CUDA devices are visible
**When** Exp 1146 runs
**Then** it writes `results/experiment_1146_grpo_reflection_reward_v3.json`
with `dualgpu_used=False`
**And** `honest_verdict` equals `blocked_no_dualgpu`
**And** required reflection-reward schema fields are still present.

---

## REQ-LEARN-1159: GRPO v4 Structural Warm-Up Separates Reflection Reward Before Full ThinkPRM Mixing

Exp 1159 SHALL run the GRPO reflection-reward experiment as a two-phase
structural-first schedule.  Phase 1 is a 300 second warm-up that uses only
the repairability objective (`r_total = r_reflect`, `w_thinkprm=0.0`).
Phase 2 is a 900 second full training phase that restores the mixed reward
(`r_total = r_thinkprm + 0.3 * r_reflect`).  This avoids the Exp 1146 failure
mode where ThinkPRM and reflection rewards were mixed from the first step.

### REQ-LEARN-1159 Sub-requirements

- REQ-LEARN-1159-1: Exp 1159 SHALL block honestly with
  `honest_verdict="blocked_no_dualgpu"` and `dualgpu_used=False` when the
  active runtime sees fewer than two CUDA devices.
- REQ-LEARN-1159-2: Exp 1159 SHALL use GSM8K training indices `[800, 900)` and
  evaluation indices `[950, 1000)` so the run is fresh relative to Exp 1129
  and Exp 1146.
- REQ-LEARN-1159-3: Phase 1 SHALL use `group_size_n=8`,
  `warmup_seconds=300`, `r_total=r_reflect`, `w_thinkprm=0.0`, DRA-GRPO
  diversity penalty enabled, and CPPO proxy reuse disabled.
- REQ-LEARN-1159-4: Phase 2 SHALL use `training_seconds=900`,
  `r_total=r_thinkprm + 0.3 * r_reflect`, DRA-GRPO diversity penalty enabled,
  and CPPO proxy reuse enabled.
- REQ-LEARN-1159-5: The experiment artifact SHALL include
  `dualgpu_used`, `cuda_device_count`, `warmup_seconds`, `training_seconds`,
  `training_wall_budget_hit`, `advantage_stdev_warmup`,
  `advantage_stdev_full`, `n_eval_questions`, `baseline_fraction_correct`,
  `trained_fraction_correct`, `improvement_over_baseline`,
  `improvement_vs_exp1129`, `reflection_weight`, `structural_warmup_used`,
  `grpo_v4_honest_result`, and `honest_verdict`.
- REQ-LEARN-1159-6: Exp 1159 SHALL map outcomes to one of
  `structural_warmup_above_0851`, `positive_below_exp1129`, `neutral`,
  `negative_regression`, or `blocked_no_dualgpu`, where
  `improvement_vs_exp1129 = improvement_over_baseline - 0.0851`.

### SCENARIO-LEARN-1159: Structural Warm-Up Uses Reflection-Only Before Full Reward

**Given** aligned ThinkPRM scores `[0.4, 0.7]`
**And** reflection rewards `[0.5, -0.25]`
**When** Exp 1159 computes Phase 1 rewards
**Then** the total rewards equal `[0.5, -0.25]`
**And** CPPO proxy reuse is disabled for the warm-up phase.

**When** Exp 1159 computes Phase 2 rewards
**Then** the total rewards equal `[0.55, 0.625]`
**And** CPPO proxy reuse is enabled for the full phase.

### SCENARIO-LEARN-1160: GRPO v4 Blocked Artifact Retains Required Schema

**Given** the active runtime sees fewer than two CUDA devices
**When** Exp 1159 runs
**Then** it writes `results/experiment_1159_grpo_v4_structural_warmup.json`
with `dualgpu_used=False`
**And** `honest_verdict` equals `blocked_no_dualgpu`
**And** all REQ-LEARN-1159-5 artifact fields are present.

---

## REQ-LEARN-1173: GRPO v5 Applies TinyV False-Negative Abstention After Structural Warm-Up

Exp 1173 SHALL preserve Exp 1159's two-phase structural warm-up schedule and
add TinyV-style false-negative correction only after the 300 second
reflection-only warm-up. During the full reward-mixing phase, if the ThinkPRM
confidence for a completion is in the calibrated uncertainty interval
`[fn_abstain_thresh_low, fn_abstain_thresh_high]`, the completion's total reward
SHALL be set to `0.0` so the sample contributes no GRPO gradient signal.

### REQ-LEARN-1173 Sub-requirements

- REQ-LEARN-1173-1: Exp 1173 SHALL keep `warmup_seconds=300`,
  `training_seconds=900`, `group_size_n=8`, warm-up reward
  `r_total=r_reflect`, full-phase reward
  `r_total=r_thinkprm + 0.3 * r_reflect`, DRA-GRPO diversity penalty enabled,
  and CPPO proxy reuse enabled only for the full phase.
- REQ-LEARN-1173-2: Exp 1173 SHALL use GSM8K training indices `[1000, 1200)`
  and evaluation indices `[1200, 1250)` to avoid overlap with the prior GRPO
  runs.
- REQ-LEARN-1173-3: The default TinyV uncertainty interval SHALL use
  `fn_abstain_thresh_low=0.3` and `fn_abstain_thresh_high=0.7`; both endpoints
  are inclusive and the thresholds SHALL be validated as ordered values in
  `[0.0, 1.0]`.
- REQ-LEARN-1173-4: When a full-phase completion abstains, the artifact SHALL
  count it toward `fn_abstention_rate`, the raw pre-abstention reward SHALL be
  retained in per-question diagnostics, and the emitted score used by GRPO
  SHALL be exactly `0.0`.
- REQ-LEARN-1173-5: The experiment artifact SHALL include
  `improvement_over_baseline`, `v4_baseline`, `fn_abstention_rate`,
  `fn_threshold_tuned`, `training_completed`, `dualgpu_confirmed`,
  `grpo_v5_honest_result`, and `honest_verdict`.
- REQ-LEARN-1173-6: Exp 1173 SHALL map outcomes to one of
  `tinyv_improves_over_v4`, `tinyv_tied_with_v4`, `tinyv_degrades_v4`, or
  `training_wall_hit`, where the v4 comparison baseline is
  `v4_baseline=0.10`.
- REQ-LEARN-1173-7: Exp 1173 SHALL set `dualgpu_confirmed=True` only when at
  least two CUDA devices are visible and the active `llama.cpp` runtime reports
  GPU layer offload support; otherwise it SHALL write a blocked artifact rather
  than run the GGUF model on CPU.

### SCENARIO-LEARN-1173: TinyV Abstention Zeroes Uncertain Full-Phase Rewards

**Given** a full-phase completion with ThinkPRM confidence `0.5`
**And** `fn_abstain_thresh_low=0.3`
**And** `fn_abstain_thresh_high=0.7`
**When** Exp 1173 computes the reward
**Then** the raw mixed reward is recorded for diagnostics
**And** the emitted reward used by GRPO equals `0.0`
**And** the sample increments the TinyV abstention count.

### SCENARIO-LEARN-1174: Structural Warm-Up Does Not Use TinyV Abstention

**Given** the Exp 1173 warm-up phase is active
**When** a completion has any ThinkPRM confidence in the TinyV uncertainty interval
**Then** the warm-up reward remains `r_total=r_reflect`
**And** the sample is not counted as a TinyV abstention.

### SCENARIO-LEARN-1175: GRPO v5 Artifact Reports Honest TinyV Outcome

**Given** Exp 1173 finishes training and evaluation
**When** `improvement_over_baseline` is compared against `v4_baseline=0.10`
**Then** the artifact records the TinyV abstention rate and threshold
**And** `grpo_v5_honest_result=True`
**And** `honest_verdict` uses only the REQ-LEARN-1173-6 labels.

### SCENARIO-LEARN-1176: CPU-Only Llama Runtime Blocks Live GRPO v5

**Given** two CUDA devices are visible
**And** the active `llama.cpp` runtime does not support GPU layer offload
**When** Exp 1173 starts
**Then** it writes a blocked artifact with `dualgpu_confirmed=False`
**And** `training_completed=False`.

## REQ-LEARN-1184: GRPO v5 Continuous TinyV v2 Reward + DualGPU Tensor Split

Exp 1184 SHALL train Qwen3.6-35B-A3B-GGUF with GRPO using a continuous
ThinkPRM v2 energy reward (TinyV v2 reward shaping) instead of binary
correctness, and SHALL split inference across two RTX 3090 GPUs with
`tensor_split=[0.5, 0.5]` so the 35B model fits in 48 GiB of combined
VRAM.

The total per-completion reward in the full phase is

    r_total = 0.6 * r_energy + 0.4 * r_reflect

where `r_energy` is the continuous ThinkPRM v2 score in `[0, 1]` and
`r_reflect` is the Exp 1159 reflection reward. The structural warm-up
phase preserves Exp 1159's reflection-only schedule.

### REQ-LEARN-1184 Sub-requirements

- REQ-LEARN-1184-1: Exp 1184 SHALL refuse to train on CPU. Before any
  training step, the script SHALL verify the active `llama.cpp` runtime
  reports `llama_supports_gpu_offload()=True` AND that
  `torch.cuda.device_count() >= 2`. If either check fails it SHALL
  write a `gpu_offload_prerequisite_not_met` artifact and exit
  cleanly without training.
- REQ-LEARN-1184-2: The continuous TinyV v2 reward weights SHALL be
  `tinyv_v2_energy_weight=0.6` and `tinyv_v2_reflection_weight=0.4`,
  validated as non-negative floats that sum to `1.0` (within
  `1e-9`).
- REQ-LEARN-1184-3: Exp 1184 SHALL evaluate on the 47-question FoVer
  validation slice, emit `n_eval_questions=47` on success, and report
  `grpo_v5_pass_rate` as the fraction of those 47 questions answered
  correctly by the trained model's best-of-N selection.
- REQ-LEARN-1184-4: The GRPO v4 baseline pass rate
  (`grpo_v4_baseline_pass_rate`) SHALL be sourced from Exp 1159's
  `trained_fraction_correct` (`0.26`), and the headline delta SHALL be
  `grpo_v5_delta_pp = grpo_v5_pass_rate - grpo_v4_baseline_pass_rate`.
- REQ-LEARN-1184-5: The artifact SHALL include
  `gpu_offload_prerequisite_met`, `training_completed`,
  `dualgpu_confirmed`, `training_tokens_per_sec`,
  `grpo_v4_baseline_pass_rate`, `grpo_v5_pass_rate`,
  `grpo_v5_delta_pp`, `tinyv_v2_mean_reward`, `n_eval_questions`,
  and `honest_verdict`.
- REQ-LEARN-1184-6: `honest_verdict` SHALL be one of
  `grpo_v5_above_v4`, `grpo_v5_regression_vs_v4`, `grpo_v5_no_delta`,
  `gpu_offload_prerequisite_not_met`, or `training_wall_hit`.
- REQ-LEARN-1184-7: Exp 1184 SHALL load the GGUF with
  `n_gpu_layers=-1`, `tensor_split=[0.5, 0.5]`, and `main_gpu=0` to
  split the 35B model across two RTX 3090s, and SHALL set
  `dualgpu_confirmed=True` only when GPU offload is verified AND
  `cuda_device_count >= 2`.

### SCENARIO-LEARN-1184: Continuous TinyV v2 Reward Mixes Energy and Reflection

**Given** a completion with ThinkPRM v2 energy score `0.8`
**And** a reflection reward `0.5`
**When** Exp 1184 computes the total reward in the full phase
**Then** the emitted reward equals `0.6 * 0.8 + 0.4 * 0.5 = 0.68`.

### SCENARIO-LEARN-1185: GPU Offload Prerequisite Blocks Training

**Given** the active `llama.cpp` runtime cannot offload layers to GPU
**When** Exp 1184 starts
**Then** it writes a blocked artifact with
  `gpu_offload_prerequisite_met=False`,
  `training_completed=False`,
  `dualgpu_confirmed=False`,
**And** `honest_verdict="gpu_offload_prerequisite_not_met"`
**And** does NOT attempt CPU training.

### SCENARIO-LEARN-1186: GRPO v5 Delta Maps to Honest Verdict

**Given** Exp 1184 finishes evaluation with `grpo_v5_pass_rate=0.32`
**And** `grpo_v4_baseline_pass_rate=0.26`
**When** the verdict is derived
**Then** `grpo_v5_delta_pp` equals `0.06`
**And** `honest_verdict` equals `"grpo_v5_above_v4"`.

---

## REQ-LEARN-1187: Latent-GRPO Masks Invalid Energy Rollouts Before One-Sided Reward Noise

Exp 1187 SHALL extend the GRPO v4 reward-update path with Latent-GRPO-style
invalid-sample masking and one-sided noise injection for Carnot energy verifier
ensembles. Invalid verifier outputs SHALL be removed before a GRPO gradient
update, and Gaussian reward noise SHALL be added only to positive-reward
rollouts so negative rewards remain exact verifier boundaries.

### REQ-LEARN-1187 Sub-requirements

- REQ-LEARN-1187-1: `mask_invalid_samples(rollouts, energies)` SHALL return
  `(filtered_rollouts, mask_rate)`, where `mask_rate = n_masked / n_total` and
  empty input returns `([], 0.0)`.
- REQ-LEARN-1187-2: A rollout SHALL be masked when its energy record contains
  NaN or Inf, when all five verifier energies are identical, when all five are
  zero, or when all verifier energies are negative.
- REQ-LEARN-1187-3: `one_sided_noise_injection(rollout, noise_scale=0.01)`
  SHALL add Gaussian noise `N(0, noise_scale^2)` only when `rollout.reward > 0`
  and SHALL leave `rollout.reward <= 0` unchanged.
- REQ-LEARN-1187-4: `LatentGRPOTrainer(base_grpo_trainer)` SHALL apply invalid
  sample masking and one-sided reward noise before delegating each gradient
  update to the wrapped GRPO trainer.
- REQ-LEARN-1187-5: The Exp 1187 artifact SHALL include
  `latent_grpo_implemented`, `mask_rate`, `grpo_v4_baseline_pass_rate`,
  `latent_grpo_pass_rate`, `latent_grpo_delta_pp`,
  `one_sided_noise_applied`, `n_eval_questions`, and `honest_verdict`.
- REQ-LEARN-1187-6: Exp 1187 SHALL evaluate the first 100 low-difficulty
  FoVer pairs and map the outcome to one of `latent_grpo_above_v4`,
  `latent_grpo_no_delta`, or `latent_grpo_regression`.

### SCENARIO-LEARN-1187: Invalid Energy Rollouts Are Masked Before Update

**Given** four rollouts with verifier energy records `[0, 1, 2, 3, 4]`,
`[NaN, 1, 2, 3, 4]`, `[7, 7, 7, 7, 7]`, and `[-1, -2, -3, -4, -5]`
**When** `mask_invalid_samples` is called
**Then** only the first rollout remains
**And** `mask_rate` equals `0.75`.

### SCENARIO-LEARN-1188: One-Sided Noise Preserves Negative Reward Boundary

**Given** one rollout with `reward=1.0` and one rollout with `reward=-1.0`
**When** one-sided noise injection runs with a deterministic Gaussian draw
**Then** the positive rollout reward changes by the draw
**And** the negative rollout reward remains exactly `-1.0`.

### SCENARIO-LEARN-1189: Latent-GRPO Artifact Reports V4 Comparison

**Given** Exp 1187 evaluates the 100-question low-difficulty FoVer subset
**When** the artifact is written
**Then** all REQ-LEARN-1187-5 fields are present
**And** `latent_grpo_delta_pp` equals
`latent_grpo_pass_rate - grpo_v4_baseline_pass_rate`.

---

## REQ-LEARN-1208: GRPO v5 TinyV Confidence Abstention vs v4 Floor

Exp 1208 SHALL retry GRPO v5 with TinyV confidence abstention on the
35B Qwen3.6-A3B GGUF after Exp 1207 verifies the active llama.cpp
runtime supports GPU offload.  When a verifier confidence score sits
inside an uncertainty band (default `[0.3, 0.7]`), the rollout's reward
contribution SHALL be skipped (abstention).  This is the "Spurious
Rewards" hypothesis test: v4's +10pp gain came from the structural
warm-up; v5 must beat that floor by `>3pp` for the energy verifier to
add real signal beyond structure.

### REQ-LEARN-1208 Sub-requirements

- REQ-LEARN-1208-1: Exp 1208 SHALL refuse to run training when either
  `llama_cpp.llama_supports_gpu_offload()` is False or
  `torch.cuda.device_count() < 2`, writing a blocked artifact with
  `honest_verdict="blocked_no_gpu_offload"` or `"blocked_no_dualgpu"`
  respectively.
- REQ-LEARN-1208-2: `tinyv_confidence_abstain(confidence, low=0.3,
  high=0.7)` SHALL return True when `low <= confidence <= high` and
  False otherwise; the band is inclusive on both ends so a score
  exactly at `0.3` or `0.7` is considered uncertain.
- REQ-LEARN-1208-3: `apply_tinyv_abstention(confidences, rewards)`
  SHALL return `(filtered_rewards, abstention_count)` where each
  `rewards[i]` is replaced with `0.0` when the matching `confidences[i]`
  triggers abstention; the lengths of the inputs MUST match.
- REQ-LEARN-1208-4: The v4 baseline floor SHALL be encoded as
  `v4_baseline_improvement_pp=10.0`, sourced from Exp 1159's measured
  +10pp delta over the .60 GRPO baseline.
- REQ-LEARN-1208-5: `beats_spurious_reward_threshold` SHALL be True iff
  `improvement_over_baseline_pp > 3.0`, encoding the arXiv 2506.10947
  Spurious Rewards paper's threshold for "energy verifier adds signal
  beyond structure."
- REQ-LEARN-1208-6: The artifact SHALL include `llama_cpp_gpu_offload`,
  `cuda_device_count`, `dualgpu_confirmed`, `model_used`,
  `training_completed`, `tinyv_abstention_count`,
  `tinyv_abstention_rate`, `v4_baseline_improvement_pp`,
  `v5_fraction_correct_before`, `v5_fraction_correct_after`,
  `improvement_over_baseline_pp`,
  `beats_spurious_reward_threshold`,
  `dualgpu_gpu0_utilization_pct`, `dualgpu_gpu1_utilization_pct`,
  and `honest_verdict`.
- REQ-LEARN-1208-7: `honest_verdict` SHALL be exactly one of
  `improvement_above_v4`, `improvement_below_v4`,
  `improvement_equal_v4`, `blocked_no_gpu_offload`,
  `blocked_no_dualgpu`, or `training_wall_hit`.

### SCENARIO-LEARN-1208: TinyV Abstention Triggers Inside Uncertainty Band

**Given** verifier confidences `[0.10, 0.45, 0.55, 0.80, 0.30, 0.70]`
**And** matching rewards `[1.0, 0.5, 0.7, 0.9, 0.6, 0.8]`
**When** Exp 1208 applies TinyV abstention with the default `[0.3, 0.7]` band
**Then** the filtered rewards equal `[1.0, 0.0, 0.0, 0.9, 0.0, 0.0]`
**And** the abstention count equals `4`.

### SCENARIO-LEARN-1209: Spurious-Reward Threshold Decides Verdict

**Given** Exp 1208 finishes with `v5_fraction_correct_before=0.40`
**And** `v5_fraction_correct_after=0.55`
**And** `v4_baseline_improvement_pp=10.0`
**When** the artifact is built
**Then** `improvement_over_baseline_pp` equals `5.0`
**And** `beats_spurious_reward_threshold` is True
**And** `honest_verdict` equals `"improvement_above_v4"`.

### SCENARIO-LEARN-1210: Missing GPU Offload Blocks Training Honestly

**Given** the active llama.cpp runtime reports `gpu_supports_offload=False`
**When** Exp 1208 starts
**Then** it writes a blocked artifact with `training_completed=False`
**And** `honest_verdict="blocked_no_gpu_offload"`
**And** does not attempt CPU training of the 35B model.

---

## REQ-LEARN-1209: GRPO-VPS Step-Level Process Supervision

Exp 1209 SHALL measure whether step-level process supervision (arXiv 2604.20659
GRPO-VPS) improves over outcome-only rewards when Carnot's existing step-level
verifiers (CausalReasoningVerifier and Z3MathVerifier) provide the per-step
signal.  The experiment evaluates on 50 GSM8K-style math questions.

### REQ-LEARN-1209 Sub-requirements

- REQ-LEARN-1209-1: `CausalReasoningVerifier.verify_step(step_text, prior_step)`
  SHALL return a float in `[0.0, 1.0]` representing the causal violation
  probability for `step_text` given `prior_step`; `0.0` means no detected
  causal error, `1.0` means clear violation.  Prior step may be `None` for
  the first step.
- REQ-LEARN-1209-2: `Z3MathVerifier.verify_step(step_text)` SHALL return a
  float in `[0.0, 1.0]` representing the arithmetic violation probability for
  `step_text`; `0.0` means all arithmetic claims check out, `1.0` means at
  least one arithmetic claim is detectably wrong.
- REQ-LEARN-1209-3: `segment_reward(step_text, step_index, prior_step)` SHALL
  return `0.5 * (1.0 - causal_score) + 0.5 * (1.0 - z3_score)` clamped to
  `[0.0, 1.0]`, where `causal_score` comes from REQ-LEARN-1209-1 and
  `z3_score` comes from REQ-LEARN-1209-2.
- REQ-LEARN-1209-4: `aggregate_step_rewards(per_step_rewards, gamma)` SHALL
  return the discounted sum `sum(r * gamma**i for i, r in enumerate(rewards))`
  for decay factor `gamma` defaulting to `0.9`.
- REQ-LEARN-1209-5: The artifact SHALL include `n_questions_evaluated`,
  `causal_verifier_violations_pct`, `z3_verifier_violations_pct`,
  `step_reward_correctness_correlation`, `outcome_baseline_accuracy`,
  `grpo_vps_accuracy`, `grpo_vps_delta_pp`, `grpo_vps_step_delta_measured`,
  `model_used`, and `honest_verdict`.
- REQ-LEARN-1209-6: `honest_verdict` SHALL be exactly one of
  `step_supervision_improves_over_outcome`, `step_supervision_no_delta`,
  `step_supervision_degrades`, or `insufficient_step_signal`.

### SCENARIO-LEARN-1211: Causal Verifier Returns Zero for Consistent Step

**Given** a prior step "There are 12 apples in 3 bags"
**And** a current step "Each bag has 4 apples"
**When** `CausalReasoningVerifier().verify_step(step, prior)` is called
**Then** the result is less than `0.5` (no clear violation detected).

### SCENARIO-LEARN-1212: Z3 Verifier Catches Wrong Arithmetic

**Given** a step text "3 + 4 = 8"
**When** `Z3MathVerifier().verify_step(step)` is called
**Then** the result is greater than `0.5` (arithmetic violation detected).

### SCENARIO-LEARN-1213: Segment Reward Is Symmetric Average

**Given** `causal_score=0.2` and `z3_score=0.4`
**When** `segment_reward` is computed
**Then** the result equals `0.5*(1.0-0.2) + 0.5*(1.0-0.4)` = `0.7`.

### SCENARIO-LEARN-1214: Aggregate Step Rewards Uses Geometric Decay

**Given** per-step rewards `[1.0, 1.0, 1.0]` and decay `gamma=0.9`
**When** `aggregate_step_rewards` is called
**Then** the result equals `1.0 + 0.9 + 0.81` = `2.71`.

---

## REQ-LEARN-1219: GRPO v5 -35pp Regression Root-Cause Diagnosis

Exp 1219 SHALL diagnose why Exp 1208's GRPO v5 TinyV-abstention training
caused a -35pp regression vs the v4 floor (v4 was +10pp; v5 was -25pp
absolute, delta -35pp), so that the GRPO-VPS Exp 1220 follow-up does not
repeat the same failure mode. The diagnosis is a *post-hoc analysis* of
the Exp 1208 artifact plus a static read of `latent_grpo.py` and
`grpo_v5_2.py`; it does not run new training.

### REQ-LEARN-1219 Sub-requirements

- REQ-LEARN-1219-1: `diagnose_grpo_v5_regression(artifact)` SHALL return
  a dataclass-like mapping containing at minimum
  `tinyv_abstention_rate_observed`, `grpo_v5_improvement_pp`,
  `root_cause`, `root_cause_evidence`, and `recommended_fix_for_exp1220`.
- REQ-LEARN-1219-2: `root_cause` SHALL be exactly one of
  `high_abstention_rate`, `dualgpu_instability`,
  `threshold_misconfiguration`, `reward_signal_collapse`,
  `implementation_bug`, or `unknown`.
- REQ-LEARN-1219-3: When `tinyv_abstention_rate >= 0.6` AND
  `dualgpu_gpu0_utilization_pct` and `dualgpu_gpu1_utilization_pct`
  are both within `[0.5x, 2x]` of each other, the diagnosis SHALL
  classify the root cause as `high_abstention_rate`. The Exp 1208
  observed values (abstention_rate=0.625, gpu0=44%, gpu1=50%) MUST
  trigger this branch.
- REQ-LEARN-1219-4: When `max(gpu0_util, gpu1_util) /
  max(min(gpu0_util, gpu1_util), 1.0) > 2.0`, the diagnosis SHALL
  classify the root cause as `dualgpu_instability`.
- REQ-LEARN-1219-5: When the post-training pass rate is *strictly less*
  than the pre-training pass rate by more than 0.10 (10pp) AND the
  abstention rate is below 0.6, the diagnosis SHALL classify the root
  cause as `reward_signal_collapse`.
- REQ-LEARN-1219-6: The artifact written to
  `results/experiment_1219_grpo_v5_regression_diagnosis.json` SHALL
  include `tinyv_abstention_rate_observed`, `grpo_v5_improvement_pp`,
  `root_cause`, `root_cause_evidence`,
  `recommended_fix_for_exp1220`, `diagnosis_complete`, and
  `honest_verdict`.
- REQ-LEARN-1219-7: `honest_verdict` SHALL be exactly one of
  `root_cause_identified`, `root_cause_partial`, or
  `root_cause_unknown`.

### SCENARIO-LEARN-1219: Exp 1208 Inputs Map to High Abstention Rate

**Given** an Exp 1208 artifact with `tinyv_abstention_rate=0.625`,
`dualgpu_gpu0_utilization_pct=44.0`, `dualgpu_gpu1_utilization_pct=50.0`,
`v5_fraction_correct_before=1.0`, `v5_fraction_correct_after=0.75`,
and `improvement_over_baseline_pp=-35.0`
**When** `diagnose_grpo_v5_regression(artifact)` is called
**Then** the result's `root_cause` equals `"high_abstention_rate"`
**And** `recommended_fix_for_exp1220` mentions widening the high-confidence
band (lowering the abstention threshold) so more rollouts survive.

### SCENARIO-LEARN-1220: Severely Unbalanced GPU Utilization Implies DualGPU Instability

**Given** an artifact with `dualgpu_gpu0_utilization_pct=85.0`,
`dualgpu_gpu1_utilization_pct=20.0`, abstention rate `0.2`
**When** the diagnosis runs
**Then** `root_cause` equals `"dualgpu_instability"`.

### SCENARIO-LEARN-1221: Reward Sign Collapse When Abstention Is Low

**Given** an artifact with `tinyv_abstention_rate=0.10`,
`v5_fraction_correct_before=0.80`, `v5_fraction_correct_after=0.40`,
balanced GPU utilization
**When** the diagnosis runs
**Then** `root_cause` equals `"reward_signal_collapse"`.

## REQ-LEARN-1220: GRPO-VPS Full Training Run vs v4 Floor

Exp 1220 SHALL execute a full GRPO-VPS training cycle whose per-step
rewards combine `CausalReasoningVerifier.verify_step` and
`Z3MathVerifier.verify_step` (per arXiv 2604.20659 segment formulation),
warm up against a structural reflection-only reward (Phase A) before
mixing in the VPS step-level reward and a small correctness bonus
(Phase B), and report whether the post-training pass rate beats the v4
baseline floor of +10pp from Exp 1159. The experiment SHALL apply the
three Exp 1219 fixes (`exp1219_fix_applied`): wider effective rollout
pool, soft-confidence-weighted abstention rather than hard zeroing, and
a holdout slice with measurable headroom in both directions.

### REQ-LEARN-1220 Sub-requirements

- REQ-LEARN-1220-1: `compute_vps_aggregate_reward(response, decay)`
  SHALL split a response into reasoning steps via the existing
  `compute_step_rewards_for_response` segmenter (REQ-LEARN-1209-3) and
  return `sum(decay**k * step_reward[k] for k in range(n_steps))`.
- REQ-LEARN-1220-2: `soft_confidence_weight(rewards, confidences)`
  SHALL return the elementwise product `rewards[i] * confidences[i]`
  (per the exp1219 fix that replaces hard zeroing inside the abstention
  band with soft weighting); it SHALL raise `ValueError` on length
  mismatch.
- REQ-LEARN-1220-3: `mix_phase_b_reward(r_vps, r_reflect,
  r_correctness, w_vps=0.5, w_reflect=0.3, w_correctness=0.2)` SHALL
  return `w_vps * r_vps + w_reflect * r_reflect + w_correctness *
  r_correctness`. Weights MUST sum to 1.0; `mix_phase_b_reward` SHALL
  raise `ValueError` otherwise.
- REQ-LEARN-1220-4: `derive_grpo_vps_honest_verdict(improvement_pp,
  training_completed, prereq_ok)` SHALL return one of
  `"vps_training_beats_v4"`, `"vps_training_matches_v4"`,
  `"vps_training_below_v4"`, `"training_wall_hit"`, or
  `"blocked_no_gpu"`. `vps_training_beats_v4` requires
  `improvement_pp > 10.0`; `vps_training_matches_v4` requires
  `improvement_pp` in `[0.0, 10.0]`; below 0.0 is
  `vps_training_below_v4`. `training_completed=False` with
  `prereq_ok=True` collapses to `training_wall_hit`; `prereq_ok=False`
  collapses to `blocked_no_gpu`.
- REQ-LEARN-1220-5: The artifact SHALL include
  `llama_cpp_gpu_offload`, `cuda_device_count`, `model_used`,
  `exp1219_fix_applied`, `training_completed`, `n_training_questions`,
  `n_eval_questions`, `grpo_vps_fraction_correct_before`,
  `grpo_vps_fraction_correct_after`, `grpo_vps_improvement_pp`,
  `v4_baseline_improvement_pp`, `beats_v4_floor`,
  `grpo_vps_training_completed`, and `honest_verdict`. Every required
  field SHALL be checked at script exit; missing fields raise
  `AssertionError`.
- REQ-LEARN-1220-6: `beats_v4_floor` SHALL be exactly
  `grpo_vps_improvement_pp > 10.0` (strict greater-than). The v4
  baseline value SHALL be hard-coded as `10.0` so that future
  improvements to v4 do not silently relax the gate.

### SCENARIO-LEARN-1222: VPS Aggregate Reward Discounts Later Steps

**Given** per-step rewards `[1.0, 0.5, 0.2]` and `decay=0.9`
**When** `compute_vps_aggregate_reward` is invoked with these step
rewards (via the segmenter)
**Then** the aggregate equals `1.0 + 0.9*0.5 + 0.81*0.2 = 1.612`.

### SCENARIO-LEARN-1223: Soft-Confidence Weighting Replaces Hard Zeroing

**Given** rewards `[1.0, 0.0, 1.0]` and confidences `[0.9, 0.5, 0.2]`
**When** `soft_confidence_weight(rewards, confidences)` runs
**Then** the result equals `[0.9, 0.0, 0.2]` — high-confidence rewards
pass through, mid-band confidences are *attenuated* (not zeroed).

### SCENARIO-LEARN-1224: Phase-B Mixed Reward Sums to Convex Combination

**Given** `r_vps=0.8`, `r_reflect=0.4`, `r_correctness=1.0` and the
default weights `0.5/0.3/0.2`
**When** `mix_phase_b_reward` runs
**Then** the result equals `0.5*0.8 + 0.3*0.4 + 0.2*1.0 = 0.72`.

### SCENARIO-LEARN-1225: Verdict Maps Improvement to v4 Floor

**Given** `improvement_pp=12.0`, `training_completed=True`,
`prereq_ok=True`
**When** `derive_grpo_vps_honest_verdict` runs
**Then** the verdict equals `"vps_training_beats_v4"`.

**Given** `improvement_pp=5.0`, `training_completed=True`,
`prereq_ok=True`
**Then** the verdict equals `"vps_training_matches_v4"`.

**Given** `improvement_pp=-3.0`, `training_completed=True`,
`prereq_ok=True`
**Then** the verdict equals `"vps_training_below_v4"`.

**Given** `prereq_ok=False`
**Then** the verdict equals `"blocked_no_gpu"`.

## REQ-LEARN-1221: GRPO-v6 FSPO Per-Token Factuality Weighting + VPS Step Supervision Combined

**Purpose:** Extend GRPO-VPS (Exp 1220) with FSPO-style per-token advantage
weighting so that training signal is attributed to individual tokens rather than
whole steps.

- REQ-LEARN-1221-1: `compute_fspo_vps_advantage(step_rewards, factuality_scores,
  tokens_per_step)` SHALL compute, for each token T in step S, the advantage
  `group_normalize(step_rewards)[S] * factuality_scores[S]`.
- REQ-LEARN-1221-2: `select_best_completion(completions, advantages)` SHALL
  return the completion whose sum of per-token advantages is highest.
- REQ-LEARN-1221-3: `derive_fspo_honest_verdict(fspo_delta_pp)` SHALL return
  `"fspo_improves_over_vps"` when `fspo_delta_pp > 0`,
  `"fspo_matches_vps"` when `fspo_delta_pp == 0`, and
  `"fspo_degrades_vps"` when `fspo_delta_pp < 0`.
- REQ-LEARN-1221-4: The artifact SHALL include `n_questions_evaluated`,
  `n_completions_per_question`, `vps_baseline_accuracy`, `fspo_vps_accuracy`,
  `fspo_delta_pp`, `fspo_improves_over_vps`, `model_used`,
  `grpo_v6_fspo_delta_measured`, and `honest_verdict`.

### SCENARIO-LEARN-1226: FSPO Advantage Inherits Step Factuality Score

**Given** one step with `step_reward=0.8`, `factuality_score=0.5` and 3 tokens
**When** `compute_fspo_vps_advantage` runs with these values
**Then** each token in that step has advantage `0.5 * normalized_reward`.

### SCENARIO-LEARN-1227: Best Completion Is Highest Sum of Token Advantages

**Given** two completions with total token advantages `[1.2, 0.4]`
**When** `select_best_completion` is called
**Then** the completion with advantage sum `1.2` is returned.

### SCENARIO-LEARN-1228: Verdict Maps Delta to FSPO Outcome

**Given** `fspo_delta_pp=3.0`
**When** `derive_fspo_honest_verdict` runs
**Then** the verdict equals `"fspo_improves_over_vps"`.

**Given** `fspo_delta_pp=-2.0`
**Then** the verdict equals `"fspo_degrades_vps"`.

## REQ-LEARN-1235: GRPO-v6 FSPO + VPS Uses Token Regulation Under Extended Wall Budget

Exp 1235 SHALL rerun GRPO-v6 with the Exp 1220 VPS process reward,
FSPO per-token factuality weighting, token-regulated advantages, and an
extended `wall_budget_s=1200` so evaluation completes more than the 9
questions reached by Exp 1221.

### REQ-LEARN-1235 Sub-requirements

- REQ-LEARN-1235-1: `compute_token_regulated_fspo_vps_advantage(
  step_causal_scores, step_z3_scores, tokens_per_step, token_logprobs)`
  SHALL compute `step_reward[S] = causal_score[S] + z3_score[S]`,
  `factuality_weight[S] = max(0.0, causal_score[S])`, and for each token
  in step S compute `advantage[T] = step_reward[S] *
  factuality_weight[S]`.
- REQ-LEARN-1235-2: The token-regulated advantage SHALL be
  `advantage[T] * softmax(token_logprobs)[T]`, where the softmax is
  computed across all generated tokens in the completion using a
  numerically stable implementation.
- REQ-LEARN-1235-3: `compute_grpo_v6_token_metrics` SHALL report
  `mean_token_logprob` as the arithmetic mean of non-null token
  log-probabilities and `fspo_coverage_fraction` as the fraction of
  reasoning steps whose `factuality_weight` is strictly positive.
- REQ-LEARN-1235-4: `derive_grpo_v6_token_reg_honest_verdict` SHALL return
  exactly one of `"grpo_v6_beats_vps"`, `"grpo_v6_no_improvement"`,
  `"grpo_v6_regression"`, `"wall_budget_still_insufficient"`, or
  `"blocked"`. Fewer than 10 evaluated questions SHALL map to
  `"wall_budget_still_insufficient"` when prerequisites are otherwise OK.
- REQ-LEARN-1235-5: The Exp 1235 artifact SHALL include
  `wall_budget_s`, `n_questions_evaluated`, `vps_baseline_accuracy`,
  `fspo_vps_accuracy`, `grpo_v6_improvement_pp`, `beats_vps_floor`,
  `mean_token_logprob`, `fspo_coverage_fraction`, `dualgpu_confirmed`,
  `grpo_v6_improvement_measured`, and `honest_verdict`.

### SCENARIO-LEARN-1235: Token Regulation Upweights High-Confidence Tokens

**Given** one step with `causal_score=0.5`, `z3_score=0.25`, two tokens,
and token log-probabilities `[-0.1, -4.0]`
**When** `compute_token_regulated_fspo_vps_advantage` runs
**Then** both tokens start from raw advantage `(0.5 + 0.25) * 0.5`
**And** the first token receives the larger adjusted advantage because
its token log-probability is higher.

### SCENARIO-LEARN-1236: Extended Budget Must Beat Exp 1221 Coverage

**Given** an Exp 1235 run with `n_questions_evaluated=9`
**When** `derive_grpo_v6_token_reg_honest_verdict` runs
**Then** the verdict equals `"wall_budget_still_insufficient"`.

**Given** `n_questions_evaluated=13` and `grpo_v6_improvement_pp=0.1`
**Then** the verdict equals `"grpo_v6_beats_vps"`.

## REQ-LEARN-1247: GRPO v7 Simplified VPS-Only Closed Training Loop

Exp 1247 SHALL run or honestly replay a simplified GRPO v7 loop for the
FR-11 self-learning milestone using only VPS step-level process supervision.
The experiment deliberately removes FSPO per-token weighting, token regulation,
and DualGPU requirements so a closed training-loop artifact can be produced
inside the conductor turn budget.

### REQ-LEARN-1247 Sub-requirements

- REQ-LEARN-1247-1: The artifact SHALL include `grpo_v7_ran`,
  `improvement_pp`, `baseline_accuracy`, `final_accuracy`, `training_mode`,
  `device_used`, and `honest_verdict`.
- REQ-LEARN-1247-2: `training_mode` SHALL be exactly `"vps_only"` and
  `verifier_type` SHALL be exactly `"vps_only"`; FSPO and token-regulation
  fields SHALL NOT be required for this artifact.
- REQ-LEARN-1247-3: The configured dataset sizes SHALL be
  `n_training_questions=20` and `n_eval_questions=30`.
- REQ-LEARN-1247-4: `device_used` SHALL be exactly one of `"cuda:0"`,
  `"cpu"`, `"llama_cpp"`, or `"fallback"`. Single-GPU and CPU execution are
  acceptable; missing DualGPU support SHALL NOT block the artifact.
- REQ-LEARN-1247-5: When live training cannot be invoked inside the turn
  budget, the artifact MAY use the completed Exp 1220 VPS baseline as an
  honest replay source, but it SHALL set `fallback_used=True` and
  `device_used="fallback"`.
- REQ-LEARN-1247-6: `honest_verdict` SHALL be `"grpo_v7_negative_delta"` when
  `improvement_pp < 0.0`; otherwise it SHALL be formatted as
  `"grpo_v7_improvement_pp_X"` with one decimal place. A missing executable
  runtime with no replay source SHALL map to `"grpo_v7_gpu_missing"`.

### SCENARIO-LEARN-1247: Exp 1220 VPS Replay Produces v7 Artifact Schema

**Given** an Exp 1220 artifact with `grpo_vps_fraction_correct_before=0.8`
and `grpo_vps_fraction_correct_after=0.95`
**When** Exp 1247 writes a fallback artifact from that source
**Then** `baseline_accuracy=0.8`, `final_accuracy=0.95`,
`improvement_pp=15.0`, `training_mode="vps_only"`,
`device_used="fallback"`, and `fallback_used=True`.

### SCENARIO-LEARN-1248: Negative Delta Is Accepted Honestly

**Given** `baseline_accuracy=0.6` and `final_accuracy=0.5`
**When** the Exp 1247 artifact is built
**Then** `improvement_pp=-10.0`
**And** `honest_verdict="grpo_v7_negative_delta"`.

## REQ-LEARN-1273: GRPO v8 PRIME/VPRM Self-Learning Smoke

Exp 1273 SHALL run a bounded self-learning smoke test, not a full GRPO
training run. It SHALL load the `verifier_weight_vector` selected by Exp 1272,
try `cached_sota_pair()` before any legacy model path, and compare PRIME/VPRM
weighted reward before and after a small self-learning update on a held-out
slice. When the mandated SOTA GGUF cache or GPU runtime is unavailable, the
experiment SHALL write an honest deterministic smoke artifact that is not
eligible for headline claims.

### REQ-LEARN-1273 Sub-requirements

- REQ-LEARN-1273-1: The runner SHALL write
  `results/experiment_1273_grpo_v8_prime_vprm_smoke.json` first with
  `status="in_progress"` and `honest_verdict="in_progress"`.
- REQ-LEARN-1273-2: The final artifact SHALL load and record
  `verifier_weights_used` from
  `results/experiment_1272_prime_verifier_selection_audit.json`.
- REQ-LEARN-1273-3: The runner SHALL call `cached_sota_pair()` first and record
  both `MODEL_SPECS` and `models_used`. `models_used` SHALL include at least one
  mandated SOTA GGUF id from `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or `unsloth/gemma-4-26B-A4B-it-GGUF`.
- REQ-LEARN-1273-4: The bounded comparison SHALL use 10-20 training items and
  20-30 held-out evaluation items and report `grpo_v8_delta_pp`,
  `self_learning_delta_overall`, `wall_budget_s`, `terminal_status`, and
  `honest_verdict` even when the delta is non-positive.
- REQ-LEARN-1273-5: Missing SOTA GGUF cache or GPU runtime SHALL set
  `headline_result_allowed=false` and `execution_mode` to either `"blocked"` or
  `"smoke_only"`, never `"live_sota"`.

### SCENARIO-LEARN-1273: Missing SOTA Cache Produces Non-Headline Smoke Artifact

**Given** Exp 1272 has written a non-empty `verifier_weight_vector`
**And** `cached_sota_pair()` returns `None`
**When** Exp 1273 runs
**Then** the final artifact has `execution_mode="smoke_only"`
**And** `headline_result_allowed=false`
**And** `honest_verdict="smoke_only_not_headline"`
**And** `models_used` records the attempted mandated SOTA GGUF ids.

### SCENARIO-LEARN-1274: Live SOTA Path May Claim Headline Only With Cached GGUF Specs

**Given** `cached_sota_pair()` returns concrete GGUF-backed `MODEL_SPECS`
**When** the bounded comparison uses those specs for generated reasoning samples
**Then** `execution_mode="live_sota"`
**And** `headline_result_allowed=true`
**And** `grpo_v8_delta_pp` equals `100 * (after_reward - before_reward)`.

## Implementation Status (REQ-LEARN-1273)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1273 | Implemented | Python (test_experiment_1273_grpo_v8_prime_vprm_smoke.py) |

## REQ-LEARN-1274: Certificate Memory Replay Evaluation

Exp 1274 SHALL turn verified certificate outputs, or verified FoVer labels when
certificate outputs are unavailable, into a replayable case-memory table. The
runner SHALL measure decision quality before and after memory lookup on a
held-out replay slice and SHALL write an honest artifact even when the measured
self-learning delta is zero or negative.

### REQ-LEARN-1274 Sub-requirements

- REQ-LEARN-1274-1: The runner SHALL write
  `results/experiment_1274_online_self_learning_certificate_memory_v3.json`
  first with `status="in_progress"` and `honest_verdict="in_progress"`.
- REQ-LEARN-1274-2: The runner SHALL load certificate examples from Exp 1271
  when present and usable; otherwise it SHALL load verified FoVer pairs and set
  `source="fover_fallback"`.
- REQ-LEARN-1274-3: The runner SHALL build a memory table keyed by
  `constraint_pattern`, `verifier_result`, and `repair_hint`, split examples
  into memory-build and replay-eval slices, and report `memory_entries`.
- REQ-LEARN-1274-4: The final artifact SHALL include `status="complete"`,
  `before_score`, `after_score`, `self_learning_delta_overall`,
  `skill_graph_candidate_count`, and `honest_verdict`.
- REQ-LEARN-1274-5: Optional `skill_graph_candidates` SHALL include
  `contract`, `evidence_count`, `replay_success_rate`, and
  `demotion_condition` for each emitted candidate.

### SCENARIO-LEARN-1275: FoVer Fallback Produces Honest Replay Delta

**Given** Exp 1271 has no usable certificate outputs
**And** `results/fover_corpus_v5.json` contains labeled FoVer pairs
**When** Exp 1274 runs
**Then** the final artifact has `source="fover_fallback"`,
`status="complete"`, a numeric `self_learning_delta_overall`, and a non-negative
`memory_entries` count.

## REQ-LEARN-1383: GRPO v7 JURY-RL Formal Verifier Rewards

Exp 1383 SHALL run a bounded GRPO v7 reward loop that uses JURY-RL-style
majority voting plus Carnot formal verifier rewards on FoVer reasoning rows,
without a learned reward model.  The runner SHALL resolve headline model specs
through `cached_sota_pair(gpu_indices=(0, 1))` and SHALL configure llama.cpp
with `tensor_split=[0.5, 0.5]` for dual RTX 3090 execution.

### REQ-LEARN-1383 Sub-requirements

- REQ-LEARN-1383-1: The runner SHALL write
  `results/experiment_1383_grpo_v7_jury_rl_formal_verifier_rewards.json`
  first with `status="in_progress"`.
- REQ-LEARN-1383-2: Each training case SHALL generate `N=4` candidate rollouts,
  select a candidate answer by majority vote, and run the local Carnot semantic
  verifier on that candidate.
- REQ-LEARN-1383-3: Verified candidates SHALL assign `+1` to matching rollouts
  and `-1` to non-matching rollouts; rejected candidates SHALL assign `-1` to
  every rollout; UNKNOWN candidates SHALL apply ResZero so the emitted reward
  signal has zero mean.
- REQ-LEARN-1383-4: The final artifact SHALL include `status`, `grpo_version`,
  `reward_mechanism`, `models_used`, `wall_budget_s`, `wall_time_used_s`,
  `training_steps_completed`, `jury_acceptance_rate`,
  `formal_reward_pass_rate`, `resZero_applied_count`,
  `grpo_v7_improvement_pp`, `wall_budget_exhausted`, `terminal_blocker`,
  `headline_result_allowed`, and `honest_verdict`.
- REQ-LEARN-1383-5: `headline_result_allowed` SHALL be true only when
  `grpo_v7_improvement_pp > 0.0`, at least one mandated SOTA GGUF model was
  used for generation, and the run did not exhaust the wall budget.
- REQ-LEARN-1383-6: If `wall_budget_s=2400` is exhausted, the artifact SHALL set
  `wall_budget_exhausted=true`, record `terminal_blocker`, and set
  `retire_if_same_verdict=true`.

### SCENARIO-LEARN-1383: ResZero Preserves Zero-Mean UNKNOWN Rewards

**Given** four JURY-RL rollouts whose majority candidate is UNKNOWN
**When** Exp 1383 computes rewards
**Then** `resZero_applied_count` increases
**And** the four emitted rewards sum to zero.

### SCENARIO-LEARN-1384: Positive Held-Out Delta Gates Headline Claims

**Given** a run using a mandated SOTA GGUF resolved through
`cached_sota_pair(gpu_indices=(0, 1))`
**And** held-out evaluation improves after the GRPO v7 reward update
**When** Exp 1383 writes its final artifact
**Then** `headline_result_allowed=true`
**And** `honest_verdict` reports a positive GRPO v7 improvement.

## REQ-LEARN-1393: GRPO v8 Applies NGRPO Advantage Calibration to Zero-Reward UNKNOWN Groups

Exp 1393 SHALL run a bounded GRPO v8 reward loop that keeps Exp 1383's
JURY-RL formal-verifier setup but replaces ResZero's zero-mean UNKNOWN reward
fallback with NGRPO Advantage Calibration.  The runner SHALL resolve headline
model specs through `cached_sota_pair(gpu_indices=(0, 1))`, SHALL attach
`tensor_split=[0.5, 0.5]`, and SHALL use dual RTX 3090 execution for live GGUF
generation.

### REQ-LEARN-1393 Sub-requirements

- REQ-LEARN-1393-1: The runner SHALL write
  `results/experiment_1393_grpo_v8_ngrpo_zero_reward_fix.json` first with
  `status="in_progress"`.
- REQ-LEARN-1393-2: Each training case SHALL generate `N=4` candidate rollouts,
  select a candidate by JURY-RL majority vote, and run Carnot's semantic
  verifier before computing policy advantages.
- REQ-LEARN-1393-3: When the semantic verifier returns UNKNOWN and all real
  rollout rewards are zero, the runner SHALL inject one virtual max-reward
  sample into the group before advantage centering.  The virtual sample SHALL
  affect the mean used for advantages but SHALL NOT be treated as a real rollout
  or model output.
- REQ-LEARN-1393-4: The final artifact SHALL include `status`, `grpo_version`,
  `reward_mechanism`, `ngrpo_advantage_calibration_applied`,
  `virtual_max_reward_sample_injected`, `models_used`, `wall_budget_s`,
  `wall_time_used_s`, `training_steps_completed`, `jury_acceptance_rate`,
  `formal_reward_pass_rate`, `unknown_rollout_rate`, `grpo_v8_improvement_pp`,
  `wall_budget_exhausted`, `terminal_blocker`, `headline_result_allowed`, and
  `honest_verdict`.
- REQ-LEARN-1393-5: `headline_result_allowed` SHALL be true only when
  `grpo_v8_improvement_pp > 0.0`, at least one mandated SOTA GGUF model was
  used for generation, and the run did not exhaust the wall budget.
- REQ-LEARN-1393-6: If `grpo_v8_improvement_pp == 0.0` and
  `unknown_rollout_rate >= 0.95`, the artifact SHALL set
  `terminal_blocker="ngrpo_still_zero_reward_fover_incompatible_with_jury_rl"`
  and `retire_if_same_verdict=true`.

### SCENARIO-LEARN-1393: NGRPO Breaks All-UNKNOWN Zero-Reward Symmetry

**Given** four JURY-RL rollouts whose semantic-verifier reward is zero
**When** Exp 1393 computes NGRPO advantages
**Then** it injects one virtual max-reward sample
**And** each real zero-reward rollout receives a negative advantage.

### SCENARIO-LEARN-1394: Repeated All-UNKNOWN Outcome Retires GRPO v8

**Given** NGRPO Advantage Calibration was applied
**And** held-out evaluation still reports `grpo_v8_improvement_pp=0.0`
**And** `unknown_rollout_rate >= 0.95`
**When** Exp 1393 writes its final artifact
**Then** `headline_result_allowed=false`
**And** `retire_if_same_verdict=true`.

## REQ-LEARN-1398: NGRPO Theory Probe Uses Exp 1383 Zero-Reward Rollouts

Exp 1398 SHALL run a CPU-only mathematical probe over Exp 1383's recorded
all-UNKNOWN rollout rewards to validate whether NGRPO Advantage Calibration
would have produced a non-zero advantage signal before any live DualGPU
training run.  The probe SHALL perform no model inference and SHALL treat the
virtual max-reward sample as a mathematical centering device only.

### REQ-LEARN-1398 Sub-requirements

- REQ-LEARN-1398-1: The runner SHALL write
  `results/experiment_1398_ngrpo_theory_probe.json` first with
  `status="in_progress"`.
- REQ-LEARN-1398-2: The runner SHALL load Exp 1383's artifact and record that
  `all_rollouts_unknown=true` and `formal_reward_pass_rate=0.0` when the source
  artifact contains only UNKNOWN rollout answers and all training rewards are
  zero.
- REQ-LEARN-1398-3: The runner SHALL simulate ResZero on `N=4` real rewards
  `[0.0, 0.0, 0.0, 0.0]`, producing zero advantages and
  `original_resZero_advantage_variance=0.0`.
- REQ-LEARN-1398-4: The runner SHALL inject one virtual reward `1.0`, recompute
  the augmented mean as `0.2`, and report real rollout advantages
  `[-0.2, -0.2, -0.2, -0.2]` plus virtual advantage `0.8`.
- REQ-LEARN-1398-5: The final artifact SHALL include `status`,
  `exp1383_rollout_data_used`, `original_resZero_advantage_variance`,
  `ngrpo_virtual_sample_reward`, `ngrpo_augmented_advantage_variance`,
  `ngrpo_advantage_calibration_verified`, `ngrpo_expected_gradient_magnitude`,
  `theory_supports_exp1393`, and `honest_verdict`.

### SCENARIO-LEARN-1398: Virtual Sample Breaks Zero-Reward Symmetry

**Given** Exp 1383 recorded four UNKNOWN training rollouts with all rewards
`0.0`
**When** Exp 1398 injects a virtual max-reward sample before centering
**Then** the augmented advantage variance is greater than zero
**And** `ngrpo_advantage_calibration_verified=true`
**And** `theory_supports_exp1393=true`.

## REQ-LEARN-1288: InterWhen DVI Verifier-Feedback Replay

Exp 1288 SHALL convert chronological verifier accept/reject decisions into an
online routing/drafter policy update. The runner SHALL compare a frozen
post-hoc replay policy with an interleaved verifier-feedback policy that updates
before the next replay item, and SHALL write an honest artifact even when the
measured deltas are zero or negative.

### REQ-LEARN-1288 Sub-requirements

- REQ-LEARN-1288-1: The runner SHALL write
  `results/experiment_1288_interwhen_dvi_verifier_feedback_replay.json` first
  with `status="in_progress"` and `honest_verdict="in_progress"`.
- REQ-LEARN-1288-2: The runner SHALL load usable SOTA certificate/routing
  outputs when available; otherwise it SHALL load Exp 1274/FoVer replay data
  and set `source="fover_fallback"`.
- REQ-LEARN-1288-3: The runner SHALL build chronological replay slices with
  before/after policy-state summaries for each evaluated item.
- REQ-LEARN-1288-4: The runner SHALL report `dvi_acceptance_delta`,
  `online_acceptance_delta`, `violation_delta`, `self_learning_delta_overall`,
  `self_verify_signal_used`, `verification_gain`,
  `reasoning_trace_length_delta`, `clause_prediction_records` or
  `claim_level_memory_entries`, `memory_update_written`,
  `headline_result_allowed`, and `honest_verdict`.
- REQ-LEARN-1288-5: The final artifact SHALL have `status="complete"` even
  when any measured delta is zero or negative, and SHALL set
  `headline_result_allowed=false` for FoVer fallback replay.

### SCENARIO-LEARN-1288: FoVer Fallback Produces Honest Online Delta

**Given** SOTA certificate/routing outputs are unavailable or blocked
**And** Exp 1274 plus `results/fover_corpus_v5.json` are available
**When** Exp 1288 runs
**Then** the final artifact has `source="fover_fallback"`,
`status="complete"`, `self_verify_signal_used=true`, a boolean
`memory_update_written`, and an honest non-headline verdict derived from the
measured online replay delta.

## REQ-LEARN-1302: Skill-Graph Promotion And Demotion From Verifier Memory

Exp 1302 SHALL convert existing Exp 1288 verifier-feedback memory into a
sandboxed skill-graph candidate artifact without modifying production user
skills. Promotion SHALL require verifier-backed replay evidence; Neural
Garbage Collection-style demotion and expiry SHALL be counted separately from
promoted memories so continuous self-learning is not reported as promotion-only
memory growth.

### REQ-LEARN-1302 Sub-requirements

- REQ-LEARN-1302-1: The workflow SHALL write
  `results/experiment_1302_skill_graph_promotion_demotion_v2.json` first with
  `status="in_progress"` before analyzing Exp 1288 memory.
- REQ-LEARN-1302-2: The workflow SHALL load Exp 1288
  `clause_prediction_records`, `replay_slices`, and `memory_update_written`
  fields and identify reusable claim/verifier patterns.
- REQ-LEARN-1302-3: Each candidate entry SHALL include replay evidence,
  promotion criteria, demotion criteria, expiry criteria, memory type tags, and
  a sandboxed routing decision.
- REQ-LEARN-1302-4: The final artifact SHALL include `status`,
  `skill_graph_candidate_count`, `promoted_memory_count`,
  `demoted_memory_count`, `expired_memory_count`, `replay_evidence_count`,
  `memory_update_written`, and `honest_verdict`.
- REQ-LEARN-1302-5: The workflow SHALL write only under `results/` or another
  research output path and SHALL NOT alter production skill directories.

### SCENARIO-LEARN-1302: Exp 1288 Memory Produces A Sandboxed Skill Graph

**Given** Exp 1288 wrote `memory_update_written=true` and has verifier-feedback
claim-level memory
**When** the Exp 1302 skill-graph workflow runs with run date `20260505`
**Then** the final artifact has `status="complete"`
**And** it reports promoted, demoted, expired, and replay-evidence counts
separately
**And** `memory_update_written` is true for the sandboxed candidate artifact
**And** no production user skill path is modified.

## Implementation Status (REQ-LEARN-1302)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1302 | Implemented (`python/carnot/reporting/skill_graph_promotion_demotion.py`) | Implemented (`tests/python/test_skill_graph_promotion_demotion.py`) |

## REQ-LEARN-1303: QueryBandits + NGC Online Memory Policy

Exp 1303 SHALL evaluate an online verifier-feedback memory policy over the
actions replay-memory, rewrite/repair-prompt, abstain/escalate, and
demote/expire-memory. The policy SHALL use Exp 1302 skill-graph candidates and
Exp 1288 verifier-feedback replay examples, then compare a deterministic
memory-aware contextual-bandit simulation against a no-memory baseline.

### REQ-LEARN-1303 Sub-requirements

- REQ-LEARN-1303-1: The workflow SHALL write
  `results/experiment_1303_querybandits_ngc_online_memory_policy.json` first
  with `status="in_progress"` and run-date artifact metadata before policy
  simulation.
- REQ-LEARN-1303-2: The workflow SHALL load Exp 1302 `skill_graph_candidates`
  and Exp 1288 `replay_slices`/`clause_prediction_records` as the candidate
  memory and verifier-feedback examples.
- REQ-LEARN-1303-3: The policy action space SHALL include replay-memory,
  rewrite/repair-prompt, abstain/escalate, and demote/expire-memory actions.
- REQ-LEARN-1303-4: The deterministic fixed-seed bandit simulation SHALL score
  actions with verifier pass/fail reward, accepted-violation penalty,
  abstention cost, and memory-demotion cost.
- REQ-LEARN-1303-5: The final artifact SHALL include `status`,
  `self_learning_delta_overall`, `accepted_violation_delta`, `bandit_regret`,
  `selected_policy_distribution`, `memory_demotion_count`,
  `headline_result_allowed`, and `honest_verdict`.
- REQ-LEARN-1303-6: `headline_result_allowed` SHALL be false unless the
  measured self-learning delta is positive and accepted violations do not
  increase; `honest_verdict` SHALL distinguish improved, neutral, and regressed
  non-headline outcomes.

### SCENARIO-LEARN-1303: Verifier Memory Policy Reports Honest Bandit Metrics

**Given** Exp 1302 has sandboxed skill-graph candidates
**And** Exp 1288 has verifier-feedback replay slices
**When** Exp 1303 runs with run date `20260505`
**Then** the final artifact has `status="complete"`
**And** it reports the required policy distribution, demotion count, accepted
violation delta, self-learning delta, bandit regret, headline flag, and honest
verdict from the measured simulation.

## Implementation Status (REQ-LEARN-1303)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1303 | Implemented (`python/carnot/reporting/querybandits_ngc_online_memory_policy.py`) | Implemented (`tests/python/test_querybandits_ngc_online_memory_policy.py`) |

## REQ-LEARN-1315: CerCE Non-Forgetting Audit For Continuous Self-Learning

Exp 1315 SHALL audit whether the `.101` continuous self-learning memory policy
preserves previously verified behavior while improving new verifier-feedback
cases. The audit SHALL use Exp 1302 skill-graph promotion/demotion candidates,
Exp 1303 QueryBandits/NGC online-memory metrics, and Exp 1288 DVI verifier
feedback replay. It SHALL treat promotion/demotion/rewrite/abstain/expiry as
candidate policy perturbations and report a CerCE-style non-forgetting
certificate before allowing any self-learning claim to be repeated.

### REQ-LEARN-1315 Sub-requirements

- REQ-LEARN-1315-1: The workflow SHALL write
  `results/experiment_1315_continuous_self_learning_cerce_nonforgetting_audit.json`
  first with `status="in_progress"` and run-date artifact metadata before
  loading source artifacts.
- REQ-LEARN-1315-2: The workflow SHALL load Exp 1302, Exp 1303, and Exp 1288
  artifacts. If any required input is absent, it SHALL write a terminal
  artifact with `status="blocked"`, `honest_verdict="blocked_missing_inputs"`,
  and `missing_inputs` listing the absent paths.
- REQ-LEARN-1315-3: The workflow SHALL build a replay set containing old
  verified cases, `.101` improved verifier-feedback cases, and adversarial or
  unknown cases that must abstain rather than promote memory.
- REQ-LEARN-1315-4: The policy audit SHALL evaluate promote, demote, rewrite,
  abstain, and expire decisions and SHALL count accepted violations,
  adversarial promotions, and memory regressions as Lagrangian violations.
- REQ-LEARN-1315-5: `nonforgetting_certificate_rate` SHALL equal the fraction
  of old verified replay cases whose target verified behavior is preserved
  after applying the candidate memory policy.
- REQ-LEARN-1315-6: The final artifact SHALL include `status`,
  `nonforgetting_certificate_rate`, `memory_regression_count`,
  `self_learning_delta_overall`, `lagrangian_violation_penalty`,
  `accepted_violation_delta`, `promoted_memory_count`,
  `demoted_memory_count`, `headline_result_allowed`, and `honest_verdict`.
- REQ-LEARN-1315-7: `headline_result_allowed` SHALL remain false unless the run
  uses fresh headline-eligible SOTA certificate evidence.

### SCENARIO-LEARN-1315: CerCE Audit Preserves Verified Memory

**Given** Exp 1302 has sandboxed promote/demote/expire memory candidates
**And** Exp 1303 reports a positive non-headline online-memory delta
**And** Exp 1288 provides old verifier-feedback replay slices
**When** Exp 1315 runs with run date `20260505`
**Then** the final artifact has `status="complete"`
**And** it reports non-forgetting, memory-regression, accepted-violation,
Lagrangian-penalty, promotion, demotion, headline, and honest-verdict fields
**And** adversarial or unknown cases are abstained rather than promoted.

## Implementation Status (REQ-LEARN-1315)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1315 | Implemented (`python/carnot/reporting/continuous_self_learning_cerce_nonforgetting_audit.py`) | Implemented (`tests/python/test_continuous_self_learning_cerce_nonforgetting_audit.py`) |

## REQ-LEARN-1317: GRPO/VPRM v11 Headline Gate Audit

Exp 1317 SHALL run the smallest deterministic GRPO/VPRM v11 policy audit that
can test whether verifier feedback and certificate-tail token-mask signals
improve certificate-policy behavior without breaking non-forgetting. The audit
SHALL be late-gated on Exp 1312 headline-eligible certificate evidence and Exp
1315 positive non-forgetting self-learning signal, and SHALL avoid any large
training job.

### REQ-LEARN-1317 Sub-requirements

- REQ-LEARN-1317-1: The workflow SHALL write
  `results/experiment_1317_grpo_vprm_v11_headline_gate.json` first with
  `status="in_progress"` and run-date artifact metadata before loading source
  artifacts.
- REQ-LEARN-1317-2: The workflow SHALL load Exp 1312 and Exp 1315 artifacts
  and abort to a terminal blocker if Exp 1312 is not complete and
  headline-eligible, if Exp 1315 does not report positive
  `self_learning_delta_overall`, or if Exp 1315 does not preserve
  non-forgetting.
- REQ-LEARN-1317-3: The workflow SHALL resolve SOTA model readiness through
  `cached_sota_pair(gpu_indices=(0, 1))` and SHALL treat legacy or missing
  model specs as non-headline blockers rather than as headline evidence.
- REQ-LEARN-1317-4: The workflow SHALL compare a grammar-only baseline policy
  against a deterministic verifier-feedback/token-mask policy over the Exp
  1312 certificate corpus using a tiny replay budget and no large GRPO
  training job.
- REQ-LEARN-1317-5: The final artifact SHALL include `status`,
  `grpo_vprm_delta`, `verifier_feedback_token_mask_delta`,
  `nonforgetting_preserved`, `self_verification_gain`, `models_used`,
  `headline_result_allowed`, and `honest_verdict`.
- REQ-LEARN-1317-6: `headline_result_allowed` SHALL be true only when all
  structured gates pass and the certificate/model evidence comes from
  mandated headline SOTA GGUF models.

### SCENARIO-LEARN-1317: Late-Gated Certificate Replay Improves Policy Behavior

**Given** Exp 1312 has complete headline-eligible certificate attempts from
mandated SOTA GGUF models
**And** Exp 1315 reports positive self-learning delta with non-forgetting
preserved
**When** Exp 1317 runs with run date `20260505`
**Then** it writes the Exp 1317 artifact with the required GRPO/VPRM delta,
token-mask delta, non-forgetting, self-verification, model, headline, and
honest-verdict fields
**And** it reports a terminal blocker rather than activating GRPO/VPRM when any
structured late gate is not satisfied.

## Implementation Status (REQ-LEARN-1317)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1317 | Implemented (`python/carnot/reporting/grpo_vprm_v11_headline_gate.py`) | Implemented (`tests/python/test_grpo_vprm_v11_headline_gate.py`) |

## REQ-LEARN-1344: Failure-Type Governed Continuous Self-Learning Memory Policy

Exp 1344 SHALL audit continuous self-learning with a replay-only policy that
maps certificate failure types to memory promotion, demotion, quarantine, or a
fresh-verifier request. The audit SHALL run even when Phase 1 headline gates or
fresh certificate-tail updates are blocked, using available online-memory,
non-forgetting, certificate-taxonomy, and HalluGuard-style split artifacts while
labeling replay-only evidence as non-headline.

### REQ-LEARN-1344 Sub-requirements

- REQ-LEARN-1344-1: The workflow SHALL write
  `results/experiment_1344_continuous_self_learning_failure_type_memory_policy.json`
  first with `status="in_progress"` and run-date artifact metadata before
  loading source artifacts.
- REQ-LEARN-1344-2: The workflow SHALL load available Exp 1303 online-memory,
  Exp 1315 non-forgetting, Exp 1324 certificate-taxonomy, and Exp 1341
  HalluGuard-style split artifacts, and SHALL record unavailable requested
  inputs without blocking the replay audit when fallback artifacts are
  available.
- REQ-LEARN-1344-3: Each certificate failure type SHALL map to exactly one of
  `promote`, `demote`, `quarantine`, or `request_fresh_verifier`, and each
  mapping SHALL state whether non-forgetting checks and certificate-tail
  updates are allowed.
- REQ-LEARN-1344-4: The replay audit SHALL report
  `nonforgetting_certificate_rate`, `memory_regression_count`,
  `accepted_violation_delta`, `self_learning_delta_overall`,
  `promoted_memory_count`, `demoted_memory_count`, and `replay_cases_used`
  from available replay evidence rather than synthetic headline claims.
- REQ-LEARN-1344-5: `dvi_ready` SHALL be true only when
  `self_learning_delta_overall >= 0.0`, `accepted_violation_delta <= 0.0`,
  and non-forgetting does not regress (`nonforgetting_certificate_rate == 1.0`
  and `memory_regression_count == 0`).
- REQ-LEARN-1344-6: `headline_result_allowed` SHALL be true only when the
  evidence includes current `.104` headline certificate cases; replay-only
  evidence SHALL set `headline_result_allowed=false` and use an honest
  non-headline verdict.
- REQ-LEARN-1344-7: The final artifact SHALL include `status`,
  `self_learning_delta_overall`, `nonforgetting_certificate_rate`,
  `memory_regression_count`, `accepted_violation_delta`,
  `failure_type_policy`, `promoted_memory_count`, `demoted_memory_count`,
  `replay_cases_used`, `headline_certificate_cases`, `dvi_ready`,
  `headline_result_allowed`, and `honest_verdict`.

### SCENARIO-LEARN-1344: Replay-Only Failure-Type Policy Allows Non-Headline DVI Readiness

**Given** prior online-memory and non-forgetting replay artifacts are available
**And** certificate failure taxonomy and HalluGuard-style split artifacts name
parser, undergeneration, semantic, solver, unknown, and leakage-style failures
**When** Exp 1344 runs with run date `20260505`
**Then** the final artifact has `status="complete"`
**And** it maps every failure type to a governed memory policy
**And** it reports non-forgetting, memory-regression, accepted-violation,
promotion, demotion, DVI-readiness, headline, and honest-verdict fields
**And** replay-only results remain non-headline unless current `.104`
certificate cases are present.

## Implementation Status (REQ-LEARN-1344)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1344 | Implemented (`python/carnot/reporting/continuous_self_learning_failure_type_memory_policy.py`) | Implemented (`tests/python/test_continuous_self_learning_failure_type_memory_policy.py`) |

## REQ-LEARN-1358: Verifier-Selected Continuous Self-Learning Memory Update

Exp 1358 SHALL run the mandatory continuous self-learning memory update with
fresh verifier-selected evidence when available and Exp 1344 replay evidence as
the fallback baseline when fresh accepted cases are absent. Replay-only evidence
MAY preserve DVI readiness, but it MUST remain non-headline until at least one
fresh verified Exp 1353 certificate case or Exp 1355 semantic-validator case is
accepted by the verifier.

### REQ-LEARN-1358 Sub-requirements

- REQ-LEARN-1358-1: The workflow SHALL write
  `results/experiment_1358_continuous_self_learning_verifier_selected_memory.json`
  first with `status="in_progress"` and run-date artifact metadata before
  loading source artifacts.
- REQ-LEARN-1358-2: The workflow SHALL load Exp 1344 as the replay fallback
  baseline and SHALL record the exact replay artifact path used, including the
  documented filename fallback when the requested alias is absent.
- REQ-LEARN-1358-3: The workflow SHALL inspect terminal Exp 1353 certificate
  rows and Exp 1355 semantic-validator rows for verifier-accepted fresh samples;
  blocked, unparseable, untruthful, or semantic-rejected rows SHALL NOT count as
  fresh verified samples.
- REQ-LEARN-1358-4: The workflow SHALL build a small deterministic variant
  question set from fresh verified samples when present, otherwise from replay
  failure-type policy rows, and SHALL apply the local memory update path to
  those variants.
- REQ-LEARN-1358-5: Memory SHALL be promoted only for verifier-accepted variants.
  Variants marked semantic-rejected SHALL be demoted or quarantined and SHALL
  not contribute to promoted memory counts.
- REQ-LEARN-1358-6: The final artifact SHALL compute and report `status`,
  `replay_cases_used`, `fresh_verified_sample_count`, `variant_question_count`,
  `self_learning_delta_overall`, `nonforgetting_certificate_rate`,
  `memory_regression_count`, `accepted_violation_delta`,
  `promoted_memory_count`, `demoted_memory_count`, `dvi_ready`,
  `headline_result_allowed`, and `honest_verdict`.
- REQ-LEARN-1358-7: `dvi_ready` SHALL be true only when
  `self_learning_delta_overall` is positive and both non-forgetting and
  accepted-violation controls do not regress. `headline_result_allowed` SHALL
  be true only when `fresh_verified_sample_count > 0` and the update is not
  replay-only.

### SCENARIO-LEARN-1358: Replay Fallback Remains Non-Headline Without Fresh Verified Samples

**Given** Exp 1344 replay evidence is available
**And** Exp 1353 has no parseable truthful certificate rows
**And** Exp 1355 is blocked before semantic validation
**When** Exp 1358 runs with run date `20260505`
**Then** the final artifact has `status="complete"`
**And** it reports `fresh_verified_sample_count=0`
**And** it uses replay variants to exercise the local memory update path
**And** positive replay DVI evidence may set `dvi_ready=true`
**But** `headline_result_allowed=false` and `honest_verdict` names the result
as replay-only/non-headline.

### SCENARIO-LEARN-1358: Fresh Verifier-Accepted Samples Permit Headline Eligibility

**Given** at least one terminal fresh Exp 1353 or Exp 1355 row is verifier
accepted and not semantic-rejected
**When** Exp 1358 builds variants and applies the memory update path
**Then** accepted variants are promoted
**And** semantic-rejected variants are demoted or quarantined
**And** `headline_result_allowed=true` only when DVI controls pass and
`fresh_verified_sample_count > 0`.

## Implementation Status (REQ-LEARN-1358)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1358 | Implemented (`python/carnot/reporting/continuous_self_learning_verifier_selected_memory.py`) | Implemented (`tests/python/test_continuous_self_learning_verifier_selected_memory.py`) |

## REQ-LEARN-1374: Continuous Self-Learning V3 With Semantic Primary And CSP Fallback

Exp 1374 SHALL run the mandatory FR-11 continuous self-learning task
unconditionally on run date `20260505`. The workflow SHALL prefer fresh
semantic-validator evidence from Exp 1369, fall back to Eidoku CSP feasibility
from Exp 1365 only when the semantic gate is closed, and fall back to Exp 1358
replay evidence when neither verifier path is available.

### REQ-LEARN-1374 Sub-requirements

- REQ-LEARN-1374-1: The workflow SHALL write
  `results/experiment_1374_continuous_self_learning_v3_verifier_selected_or_csp_fallback.json`
  first with `status="in_progress"` before loading source artifacts.
- REQ-LEARN-1374-2: When Exp 1369 exists with
  `semantic_validator_claim_allowed=true`, `path_used` SHALL be
  `primary_semantic_verified`, memory SHALL promote only constraint-passed
  semantic rows, and semantic rejects SHALL be demoted or quarantined.
- REQ-LEARN-1374-3: When the semantic gate is closed and Exp 1365 exists with
  `eidoku_csp_viable=true`, `path_used` SHALL be `fallback_csp_selected`,
  memory SHALL promote CSP-selected samples above the configured feasibility
  threshold, `csp_selected_sample_count` SHALL be positive, and
  `headline_result_allowed` SHALL remain false because CSP-selected updates are
  not headline evidence until independently validated.
- REQ-LEARN-1374-4: When both verifier paths are unavailable, `path_used` SHALL
  be `fallback_replay` and the artifact SHALL preserve Exp 1358 replay evidence
  without allowing headline claims.
- REQ-LEARN-1374-5: The final artifact SHALL include `status`, `path_used`,
  `replay_cases_used`, `fresh_verified_sample_count`,
  `csp_selected_sample_count`, `variant_question_count`,
  `self_learning_delta_overall`, `nonforgetting_certificate_rate`,
  `memory_regression_count`, `accepted_violation_delta`,
  `promoted_memory_count`, `demoted_memory_count`, `dvi_ready`,
  `headline_result_allowed`, and `honest_verdict`.
- REQ-LEARN-1374-6: `dvi_ready` SHALL be true only when
  `self_learning_delta_overall` is positive, `nonforgetting_certificate_rate`
  is 1.0, `memory_regression_count` is 0, and
  `accepted_violation_delta <= 0.0`.
- REQ-LEARN-1374-7: `headline_result_allowed` SHALL be true only when
  `path_used="primary_semantic_verified"`, `fresh_verified_sample_count > 0`,
  and all DVI controls pass.

### SCENARIO-LEARN-1374: Semantic Primary Promotes Verified Certificate Cases

**Given** Exp 1369 reports `semantic_validator_claim_allowed=true`
**And** Exp 1358 replay controls are positive and non-forgetting
**When** Exp 1374 runs
**Then** the final artifact has `path_used="primary_semantic_verified"`
**And** it promotes only constraint-passed semantic-validator rows
**And** `fresh_verified_sample_count > 0`
**And** `headline_result_allowed=true` only when DVI controls pass.

### SCENARIO-LEARN-1374: CSP Fallback Is DVI-Eligible But Non-Headline

**Given** Exp 1369 is missing or has `semantic_validator_claim_allowed=false`
**And** Exp 1365 reports `eidoku_csp_viable=true`
**When** Exp 1374 runs
**Then** the final artifact has `path_used="fallback_csp_selected"`
**And** `csp_selected_sample_count > 0`
**And** `headline_result_allowed=false`
**And** the honest verdict names the CSP fallback as non-headline.

### SCENARIO-LEARN-1374: Replay Fallback Preserves Controls Without Headline Claims

**Given** neither Exp 1369 semantic evidence nor Exp 1365 CSP evidence is
available
**When** Exp 1374 runs
**Then** the final artifact has `path_used="fallback_replay"`
**And** it reports Exp 1358 replay controls
**And** `headline_result_allowed=false`.

## Implementation Status (REQ-LEARN-1374)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1374 | Implemented (`python/carnot/reporting/continuous_self_learning_v3_verifier_selected_or_csp_fallback.py`) | Implemented (`tests/python/test_continuous_self_learning_v3_verifier_selected_or_csp_fallback.py`) |

## REQ-LEARN-1388: FR-11 Self-Learning V4 DVI/GRPO Memory Integration

Exp 1388 SHALL run the mandatory FR-11 self-learning v4 round on run date
`20260505` using the Exp 1381 DVI checkpoint as the active verifier. The
workflow SHALL integrate GRPO-generated verified cases only when Exp 1383
reports `grpo_v7_improvement_pp > 0.0`; otherwise it SHALL unconditionally use
the DVI-only replay path and may promote DVI-verified fresh cases from Exp 1382.

### REQ-LEARN-1388 Sub-requirements

- REQ-LEARN-1388-1: The workflow SHALL write
  `results/experiment_1388_fr11_self_learning_v4_dvi_grpo_integration.json`
  first with `status="in_progress"` before loading source artifacts.
- REQ-LEARN-1388-2: The workflow SHALL require Exp 1381 to have
  `dvi_deployed=true`, SHALL load `dvi_checkpoint_path`, and SHALL set
  `dvi_checkpoint_active=true` only after the checkpoint file is readable.
- REQ-LEARN-1388-3: When Exp 1383 has `grpo_v7_improvement_pp <= 0.0`, the
  final artifact SHALL set `path_used="dvi_only_replay_exp1382"` and
  `grpo_cases_integrated=0`.
- REQ-LEARN-1388-4: Memory promotion SHALL retain Exp 1374 promoted memory and
  SHALL promote fresh Exp 1382 rows only when the active DVI verifier evidence
  has `constraint_passed=true`; failed fresh rows SHALL be counted as demoted.
- REQ-LEARN-1388-5: The final artifact SHALL include `status`, `path_used`,
  `dvi_checkpoint_active`, `replay_cases_used`,
  `fresh_verified_sample_count`, `grpo_cases_integrated`,
  `self_learning_delta_overall`, `nonforgetting_certificate_rate`,
  `memory_regression_count`, `accepted_violation_delta`,
  `promoted_memory_count`, `demoted_memory_count`,
  `dvi_auroc_delta_effect`, `headline_result_allowed`, and `honest_verdict`.
- REQ-LEARN-1388-6: `headline_result_allowed` SHALL be true only when
  `fresh_verified_sample_count > 4`, the DVI checkpoint is active,
  non-forgetting controls pass, and `accepted_violation_delta <= 0.0`.

### SCENARIO-LEARN-1388: DVI-Only Replay Promotes Exp 1382 Fresh Verified Cases

**Given** Exp 1381 deployed a readable DVI checkpoint
**And** Exp 1383 reports no positive GRPO v7 improvement
**And** Exp 1382 contains more than four DVI-verified semantic rows
**When** Exp 1388 runs
**Then** the final artifact has `path_used="dvi_only_replay_exp1382"`
**And** `grpo_cases_integrated=0`
**And** `fresh_verified_sample_count > 4`
**And** `headline_result_allowed=true` only when replay non-forgetting controls
remain clean.

### SCENARIO-LEARN-1388: Positive GRPO Gate Integrates Verified Cases

**Given** Exp 1383 reports `grpo_v7_improvement_pp > 0.0`
**When** Exp 1388 runs
**Then** the final artifact has `path_used="dvi_grpo_exp1382_integration"`
**And** verified GRPO cases are included in `grpo_cases_integrated`
**And** the headline gate still requires `fresh_verified_sample_count > 4`.

## Implementation Status (REQ-LEARN-1388)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1388 | Implemented (`python/carnot/reporting/fr11_self_learning_v4_dvi_grpo_integration.py`) | Implemented (`tests/python/test_fr11_self_learning_v4_dvi_grpo_integration.py`) |

## REQ-LEARN-1395: FR-11 Self-Learning V5 DVI V2 + SECL Verifier Promotion

Exp 1395 SHALL run the mandatory FR-11 self-learning v5 round on run date
`20260506` using Exp 1394's deployed DVI v2 + SECL checkpoint as the active
semantic verifier. The workflow SHALL attempt to exceed Exp 1388's
`fresh_verified_sample_count=59` by sampling FoVer rows, generating certificate
states, verifying those states through the DVI v2 + SECL checkpoint, and
promoting only fresh rows whose generated certificate state agrees with the
active verifier. GRPO v8 cases SHALL enter memory only when Exp 1393 reports
`grpo_v8_improvement_pp > 0.0`.

### REQ-LEARN-1395 Sub-requirements

- REQ-LEARN-1395-1: The workflow SHALL write
  `results/experiment_1395_fr11_self_learning_v5.json` first with
  `status="in_progress"` before loading source artifacts.
- REQ-LEARN-1395-2: The workflow SHALL require Exp 1394 to have
  `dvi_v2_deployed=true`, SHALL load `checkpoint_path`, and SHALL set
  `dvi_v2_checkpoint_active=true` only after the combined DVI v2 + SECL
  checkpoint exposes readable DVI metric, bias, and SECL confidence-head arrays.
- REQ-LEARN-1395-3: FoVer rows already promoted by Exp 1388 SHALL be excluded
  from the v5 fresh promotion count so `fresh_verified_sample_count` measures
  new v5 verified memory rather than replaying the prior 59 cases.
- REQ-LEARN-1395-4: A sampled FoVer row SHALL be promoted only when the
  generated certificate state (`SAT` for FoVer-correct rows and `REPAIR_HINT`
  for FoVer-incorrect rows) agrees with the DVI v2 prediction and the SECL
  calibrated confidence for that predicted state is at least the configured
  confidence threshold.
- REQ-LEARN-1395-5: When Exp 1393 has `grpo_v8_improvement_pp <= 0.0`, the
  final artifact SHALL set `grpo_v8_cases_integrated=0`. When the improvement
  is positive, only GRPO v8 rows whose verifier result matches their expected
  answer SHALL be integrated.
- REQ-LEARN-1395-6: The final artifact SHALL include `status`, `path_used`,
  `dvi_v2_checkpoint_active`, `replay_cases_used`,
  `fresh_verified_sample_count`, `grpo_v8_cases_integrated`,
  `self_learning_delta_overall`, `headline_result_allowed`, and
  `honest_verdict`.
- REQ-LEARN-1395-7: `self_learning_delta_overall` SHALL be computed as the
  case-count delta versus Exp 1388's baseline of 59 fresh verified cases, and
  `headline_result_allowed` SHALL be true only when
  `fresh_verified_sample_count > 59` with the DVI v2 checkpoint active.

### SCENARIO-LEARN-1395: DVI V2 + SECL Promotes Fresh FoVer Cases

**Given** Exp 1394 deployed a readable DVI v2 + SECL checkpoint
**And** Exp 1393 reports no positive GRPO v8 improvement
**When** Exp 1395 runs over FoVer rows not already promoted by Exp 1388
**Then** the final artifact has `dvi_v2_checkpoint_active=true`
**And** `grpo_v8_cases_integrated=0`
**And** `self_learning_delta_overall` equals
`fresh_verified_sample_count - 59`
**And** `headline_result_allowed=true` only when the fresh count exceeds 59.

### SCENARIO-LEARN-1395: Positive GRPO V8 Gate Integrates Verified Cases

**Given** Exp 1393 reports `grpo_v8_improvement_pp > 0.0`
**When** Exp 1395 runs
**Then** GRPO v8 rows whose verifier result matches their expected answer are
included in `grpo_v8_cases_integrated`
**And** the headline gate still requires the DVI v2 fresh verified count to
exceed the Exp 1388 baseline.

## Implementation Status (REQ-LEARN-1395)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1395 | Implemented (`python/carnot/reporting/fr11_self_learning_v5.py`) | Implemented (`tests/python/test_fr11_self_learning_v5.py`) |

## REQ-LEARN-1420: DPO-Style Preference Probe From Exp 1395 Verified Pairs

Exp 1420 SHALL run on `20260506` after GRPO retirement and SHALL build explicit
preference pairs from Exp 1395's 1508 verified FoVer promotions. The workflow
SHALL use the mandated SOTA GGUF model IDs for baseline/provenance checks and
SHALL report honestly when direct DPO fine-tuning of GGUF weights is unsupported.
When full GGUF DPO is unsupported, Exp 1420 SHALL train only a lightweight
DPO-style reranker over verifier/model-derived features and SHALL keep headline
claims disabled unless a real local SOTA model path was resolved and evaluated.

### REQ-LEARN-1420 Sub-requirements

- REQ-LEARN-1420-1: The workflow SHALL write
  `results/experiment_1420_dpo_verified_pairs_1508.json` first with
  `status="in_progress"` and all required artifact fields before source loading.
- REQ-LEARN-1420-2: Preference-pair construction SHALL treat Exp 1395 promoted
  IDs as preferred verified cases and SHALL pair each preferred case with the
  nearest available demoted/unverified FoVer candidate from the source corpus.
- REQ-LEARN-1420-3: The artifact SHALL preserve the mandated model specs for
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`, plus a cache/inference feasibility check
  for each attempted GGUF baseline.
- REQ-LEARN-1420-4: Direct DPO SHALL be marked performed only when the runtime
  has a supported trainer capable of updating an adapter or trainable weights
  from GGUF-backed local model provenance. Unsupported GGUF fine-tuning SHALL be
  recorded explicitly rather than simulated.
- REQ-LEARN-1420-5: When direct DPO is unsupported, the fallback SHALL fit a
  deterministic lightweight preference reranker over verifier/model features,
  compute held-out AUROC, and compute `dpo_improvement_pp` only from measured
  preferred-over-rejected accuracy delta versus a fixed baseline score.
- REQ-LEARN-1420-6: The final artifact SHALL include `status`, `model_specs`,
  `verified_pairs_available`, `dpo_full_finetune_performed`,
  `dpo_reranker_fallback_used`, `dpo_improvement_pp`,
  `dpo_vs_baseline_auroc`, `local_sota_model_used`,
  `headline_result_allowed`, and `honest_verdict`.

### SCENARIO-LEARN-1420: GGUF DPO Unsupported Uses Honest Reranker Fallback

**Given** Exp 1395 has 1508 promoted verified cases and demoted candidates
**And** no supported GGUF DPO trainer is available in the local runtime
**When** Exp 1420 runs
**Then** the final artifact has `dpo_full_finetune_performed=false`
**And** `dpo_reranker_fallback_used=true`
**And** `dpo_vs_baseline_auroc` and `dpo_improvement_pp` are measured from the
reranker evaluation rather than fabricated
**And** `headline_result_allowed=false` unless a mandated local SOTA GGUF path
was actually resolved and used for evaluation.

## Implementation Status (REQ-LEARN-1420)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1420 | Implemented (`python/carnot/reporting/dpo_verified_pairs_probe.py`) | Implemented (`tests/python/test_dpo_verified_pairs_probe.py`) |

## REQ-LEARN-1435: DPO Headline Provenance Audit

Exp 1435 SHALL run on `20260506` as a provenance audit, not as a training run.
The workflow SHALL inspect Exp 1420's DPO-style reranker artifact, local
training utilities, dependency declarations, and repository documentation to
decide whether Carnot has a supported direct GGUF fine-tune or local
adapter/fine-tune path for the mandated SOTA GGUF model specs. Legacy
small-model smoke results SHALL NOT be reported as headline DPO evidence.

### REQ-LEARN-1435 Sub-requirements

- REQ-LEARN-1435-1: The workflow SHALL write
  `results/experiment_1435_dpo_headline_provenance_audit.json` first with
  `status="in_progress"` before source evidence is evaluated.
- REQ-LEARN-1435-2: The audit SHALL preserve the mandated model specs for
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF` with their requested roles.
- REQ-LEARN-1435-3: The audit SHALL inspect Exp 1420 evidence and report why
  direct GGUF DPO was unsupported, including whether the prior result was a
  reranker fallback rather than a base-model or adapter update.
- REQ-LEARN-1435-4: The audit SHALL inspect local code and documentation for
  LoRA/adapter, GGUF conversion, TRL/PEFT DPO, and reranker-only support and
  SHALL list the evidence paths checked.
- REQ-LEARN-1435-5: The final artifact SHALL include `status`, `model_specs`,
  `direct_gguf_finetune_supported`, `local_adapter_path_supported`,
  `headline_provenance_ready`, `reranker_track_relabelled`,
  `recommended_next_training_path`, `evidence_paths_checked`, and
  `honest_verdict`.
- REQ-LEARN-1435-6: If no supported direct local adapter path exists for the
  mandated SOTA GGUF targets, the final artifact SHALL set
  `reranker_track_relabelled=true`, `headline_provenance_ready=false`, and an
  honest verdict that the DPO line remains reranker-only until conversion or
  tooling changes.

### SCENARIO-LEARN-1435: Unsupported GGUF DPO Relabels Reranker Track

**Given** Exp 1420 reports direct GGUF DPO unsupported and a reranker fallback
**And** local repository evidence contains no supported TRL/PEFT adapter path
for the mandated SOTA GGUF targets
**When** Exp 1435 runs
**Then** the final artifact has `direct_gguf_finetune_supported=false`
**And** `local_adapter_path_supported=false`
**And** `headline_provenance_ready=false`
**And** `reranker_track_relabelled=true`
**And** `honest_verdict` keeps the DPO line out of headline training claims.

## Implementation Status (REQ-LEARN-1435)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1435 | Implemented (`python/carnot/reporting/dpo_headline_provenance_audit.py`) | Implemented (`tests/python/test_dpo_headline_provenance_audit.py`) |

## REQ-LEARN-1433: FR-11 Self-Learning V6 MUST Gate on Deployed DVI V3

Exp 1433 SHALL run the mandatory FR-11 self-learning v6 round on run date
`20260506` only after Exp 1432 deploys DVI v3. The workflow SHALL keep the
Exp 1395 `fresh_verified_sample_count=1508` as the baseline, scan fresh or
held-out FoVer rows that were not already promoted by Exp 1395, verify them
with the deployed DVI v3 checkpoint and its calibrated thresholds, and report
whether v6 produced positive fresh verified sample growth without violating
the nonforgetting gate.

### REQ-LEARN-1433 Sub-requirements

- REQ-LEARN-1433-1: The workflow SHALL write
  `results/experiment_1433_fr11_self_learning_v6_dvi_v3_gated.json` first with
  `status="in_progress"` before loading source artifacts.
- REQ-LEARN-1433-2: The workflow SHALL require Exp 1432 to report
  `dvi_v3_deployed=true`, an existing `dvi_v3_checkpoint_path`, and
  `nonforgetting_rate >= 0.99` before running the v6 memory update.
- REQ-LEARN-1433-3: The v6 verifier SHALL use the DVI v3 checkpoint metric,
  bias, SECL confidence head, and calibrated thresholds saved by Exp 1432.
- REQ-LEARN-1433-4: Rows already promoted by Exp 1395 SHALL be excluded from
  the v6 candidate set so positive growth reflects new SessionMemory-equivalent
  updates rather than recounting the v5 baseline.
- REQ-LEARN-1433-5: The final artifact SHALL include `status`,
  `dvi_v3_artifact_used`, `fresh_verified_sample_count`,
  `baseline_fresh_verified_sample_count`, `self_learning_delta_overall`,
  `nonforgetting_rate`, `session_memory_updated`, `headline_result_allowed`,
  and `honest_verdict`.
- REQ-LEARN-1433-6: `self_learning_delta_overall` SHALL equal
  `fresh_verified_sample_count - baseline_fresh_verified_sample_count`, and
  `headline_result_allowed` SHALL be true only when that delta is positive and
  nonforgetting is preserved.

### SCENARIO-LEARN-1433: Deployed DVI V3 Permits Headline Only With Growth

**Given** Exp 1432 deployed DVI v3 with nonforgetting preserved
**And** Exp 1395 has 1508 baseline fresh verified samples
**When** v6 verifies held-out FoVer rows not already promoted by Exp 1395
**Then** the artifact reports the total v6 fresh verified sample count
**And** `self_learning_delta_overall` is computed versus 1508
**And** `headline_result_allowed=true` only when the delta is positive.

### SCENARIO-LEARN-1434: Missing DVI V3 Deployment Blocks V6

**Given** Exp 1432 did not deploy DVI v3 or its checkpoint is unavailable
**When** Exp 1433 starts
**Then** the artifact has `status="blocked"`
**And** `headline_result_allowed=false`
**And** `honest_verdict` names the DVI v3 deployment blocker.

## Implementation Status (REQ-LEARN-1433)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1433 | Implemented (`python/carnot/reporting/fr11_self_learning_v6_dvi_v3_gated.py`) | Implemented (`tests/python/test_fr11_self_learning_v6_dvi_v3_gated.py`) |

## REQ-LEARN-1446: FR-11 V6 Zero-Growth Diagnosis MUST Trace Candidate Losses

Exp 1446 SHALL diagnose why Exp 1433 produced zero positive FR-11 growth even
though Exp 1432 deployed DVI v3 with nonforgetting preserved. The workflow SHALL
trace candidates from FoVer supply through the Exp 1395 novelty exclusion, DVI
v3 scoring, calibrated promotion thresholds, duplicate-memory handling, and the
SessionMemory-equivalent persistence decision. The terminal artifact SHALL name
the minimum changed v7 policy required to avoid an exact Exp 1433 rerun.

### REQ-LEARN-1446 Sub-requirements

- REQ-LEARN-1446-1: The workflow SHALL write
  `results/experiment_1446_fr11_zero_growth_root_cause_diagnosis.json` first
  with `status="in_progress"` before loading source artifacts.
- REQ-LEARN-1446-2: The diagnosis SHALL count candidate loss by the categories
  `no_candidates`, `verifier_rejection`, `dvi_threshold`, `novelty_threshold`,
  `duplicate_memory`, and `persistence_blocker`.
- REQ-LEARN-1446-3: The diagnosis SHALL report the active Exp 1433 promotion
  thresholds and the recommended v7 thresholds or memory policy that differ
  from Exp 1433 enough to forbid an exact rerun.
- REQ-LEARN-1446-4: The final artifact SHALL include `status`,
  `fr11_zero_growth_root_cause_identified`, `candidate_supply_count`,
  `candidate_rejection_reason_counts`, `promotion_thresholds`,
  `memory_update_policy`, `recommended_v7_policy`, `exact_rerun_forbidden`,
  `commands_run`, and `honest_verdict`.
- REQ-LEARN-1446-5: `exact_rerun_forbidden` SHALL be true when Exp 1433 had
  active DVI v3 and zero positive growth, and the recommended v7 policy SHALL
  change promotion thresholds, candidate generation, or memory update behavior.

### SCENARIO-LEARN-1446: Active DVI V3 With Zero Growth Requires V7 Policy Change

**Given** Exp 1432 deployed DVI v3 with nonforgetting preserved
**And** Exp 1433 evaluated candidates but promoted zero new SessionMemory rows
**When** the Exp 1446 diagnosis runs
**Then** the artifact identifies the zero-growth root cause
**And** candidate losses are counted by stage
**And** `exact_rerun_forbidden=true`
**And** `recommended_v7_policy` names the changed policy required for a valid
v7 rerun.

## Implementation Status (REQ-LEARN-1446)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1446 | Implemented (`python/carnot/reporting/fr11_zero_growth_root_cause_diagnosis.py`) | Implemented (`tests/python/test_fr11_zero_growth_root_cause_diagnosis.py`) |

## REQ-LEARN-1447: FR-11 V7 MUST Apply Changed Policy and Persist Verified Growth

Exp 1447 SHALL run the mandatory FR-11 v7 self-learning round on run date
`20260506` as a bounded non-exact rerun of Exp 1433. The workflow SHALL load
Exp 1446's `recommended_v7_policy`, use the deployed Exp 1432 DVI v3
checkpoint, score local FoVer samples not already promoted by Exp 1395, apply
the v7 fresh promotion threshold separately from the replay nonforgetting
threshold, and count positive self-learning growth only for verified new
promotions that are persisted to session memory.

### REQ-LEARN-1447 Sub-requirements

- REQ-LEARN-1447-1: The workflow SHALL write
  `results/experiment_1447_fr11_v7_memory_policy_growth.json` first with
  `status="in_progress"` before loading source artifacts.
- REQ-LEARN-1447-2: The workflow SHALL require Exp 1446's
  `recommended_v7_policy.changes_exp1433_policy=true` and SHALL report the
  applied policy changes to promotion thresholds, candidate source, and memory
  update behavior.
- REQ-LEARN-1447-3: Fresh promotion candidates SHALL come from local verified
  FoVer rows not already promoted by Exp 1395, SHALL be de-duplicated by case
  ID before persistence, and SHALL be scored with DVI v3 plus the v7 fresh SECL
  threshold.
- REQ-LEARN-1447-4: Replay nonforgetting SHALL use the replay threshold from
  Exp 1446's recommended policy and SHALL require nonforgetting preservation
  before headline-positive growth is allowed.
- REQ-LEARN-1447-5: `self_learning_delta_overall` SHALL be greater than zero
  only when verified new promotions are successfully persisted to session
  memory, and SHALL equal `memory_entries_added`.
- REQ-LEARN-1447-6: The final artifact SHALL include `status`, `model_specs`,
  `policy_changes_applied`, `live_sota_inference_used`,
  `fresh_verified_sample_count`, `new_promoted_count`,
  `self_learning_delta_overall`, `nonforgetting_rate`,
  `session_memory_updated`, `memory_entries_added`,
  `retire_if_zero_growth_repeats`, `commands_run`, and `honest_verdict`.

### SCENARIO-LEARN-1447: V7 Asymmetric Threshold Produces Persisted Growth

**Given** Exp 1432 deployed DVI v3 with nonforgetting preserved
**And** Exp 1446 recommended the asymmetric fresh/replay threshold policy
**When** v7 scores fresh local FoVer candidates with the fresh threshold and
persists verified promotions to session memory
**Then** `new_promoted_count` equals `memory_entries_added`
**And** `self_learning_delta_overall` equals `memory_entries_added`
**And** `session_memory_updated=true` only when at least one entry persisted.

### SCENARIO-LEARN-1448: Zero V7 Growth Triggers Retirement Gate

**Given** the v7 policy changed from Exp 1433 but still produces no persisted
new promotions
**When** the terminal artifact is written
**Then** `retire_if_zero_growth_repeats=true`
**And** `honest_verdict` identifies the next root cause instead of reporting
positive self-learning growth.

## Implementation Status (REQ-LEARN-1447)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1447 | Implemented (`python/carnot/reporting/fr11_v7_memory_policy_growth.py`) | Implemented (`tests/python/test_fr11_v7_memory_policy_growth.py`) |

## REQ-LEARN-1449: LTLZinc Temporal Continual-Learning Adapter

Exp 1449 SHALL run on `20260507` and provide a tiny deterministic
LTLZinc-style temporal-constraint dataset that future FR-11 and DVI milestones
can consume without relying only on the legacy FR-11 candidate source. The
adapter SHALL use local CPU-only checks over finite traces and SHALL NOT claim
MiniZinc execution or external LTLZinc benchmark parity.

### REQ-LEARN-1449 Sub-requirements

- REQ-LEARN-1449-1: The workflow SHALL write
  `results/experiment_1449_ltlzinc_temporal_continual_learning_adapter.json`
  first with `status="in_progress"` and the required artifact fields before
  generating or verifying cases.
- REQ-LEARN-1449-2: The generated dataset SHALL contain at least 20
  LTLZinc-style temporal cases with labels, finite Boolean traces, a temporal
  formula, a MiniZinc-style constraint string, and a Carnot-friendly
  `certificate_state` suitable for FR-11/DVI ingestion.
- REQ-LEARN-1449-3: The local verifier SHALL distinguish accepted SAT cases
  from rejected REPAIR_HINT cases by evaluating the finite trace against the
  supported temporal operators `always`, `eventually`, `next`, and `until`.
- REQ-LEARN-1449-4: The final artifact SHALL include `status`,
  `ltlzinc_adapter_ready`, `temporal_cases_generated`, `verifier_available`,
  `accepted_case_count`, `rejected_case_count`, `dataset_path`,
  `commands_run`, and `honest_verdict`.
- REQ-LEARN-1449-5: The artifact SHALL record how the dataset can feed FR-11
  memory growth and DVI contrastive training in a later milestone while making
  clear that this run creates verified cases only, not a trained adapter.

### SCENARIO-LEARN-1449: Temporal Dataset Separates SAT And Repair-Hint Cases

**Given** the Exp 1449 adapter emits paired finite traces for simple temporal
constraints
**When** the verifier evaluates generated `always`, `eventually`, `next`, and
`until` cases
**Then** the dataset has at least one accepted and one rejected case for every
supported operator family
**And** the artifact reports a complete dataset path and honest verdict.

## Implementation Status (REQ-LEARN-1449)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1449 | Implemented (`python/carnot/reporting/ltlzinc_temporal_continual_learning_adapter.py`) | Implemented (`tests/python/test_ltlzinc_temporal_continual_learning_adapter.py`) |

## REQ-LEARN-1459: Self-Learning Lineage Decision MUST Narrow Headline Scope

Exp 1459 SHALL make the mandatory self-learning lineage decision on run date
`20260507` without launching a new self-learning training run. The decision
SHALL review the Continuous Self-Learning program requirement, Exp 1433's
zero-growth non-headline result, Exp 1447's persisted verified growth result,
Exp 1449's temporal adapter role, and the signal/noise classification rows for
self-learning artifacts that were improved but explicitly non-headline.

### REQ-LEARN-1459 Sub-requirements

- REQ-LEARN-1459-1: The workflow SHALL write
  `results/experiment_1459_self_learning_nonheadline_lineage_decision.json`
  first with `status="in_progress"` before loading the source artifacts.
- REQ-LEARN-1459-2: The final artifact SHALL include `status`,
  `self_learning_artifacts_reviewed`, `decision_note_path`,
  `self_learning_headline_pivot_selected`, `self_learning_lineage_retired`,
  `exp1447_delta_overall`, `nonforgetting_rate`, `ltlzinc_benchmark_role`,
  `next_allowed_experiment_shape`, and `honest_verdict`.
- REQ-LEARN-1459-3: A narrow headline pivot SHALL be selected only when Exp
  1447 reports positive persisted `self_learning_delta_overall`, preserved
  nonforgetting, and SessionMemory-equivalent persistence. Otherwise the
  lineage SHALL be retired from headline scope and retained only as internal
  memory-policy engineering.
- REQ-LEARN-1459-4: When the pivot is selected, `next_allowed_experiment_shape`
  SHALL define exactly one allowed future experiment shape, required metrics,
  and a nonforgetting threshold. Replay-only, adapter-only, and broad
  "self-learning improves everything" claims remain excluded from headline
  scope.
- REQ-LEARN-1459-5: The decision note at
  `docs/research-notes/self_learning_lineage_decision.md` SHALL cite the
  reviewed artifacts and explain why Exp 1449 is a benchmark feed rather than a
  standalone headline claim.

### SCENARIO-LEARN-1459: Exp 1447 Growth Allows One Narrow Headline Pivot

**Given** Exp 1433 reports zero positive growth and non-headline status
**And** the classification table includes improved non-headline self-learning
rows
**And** Exp 1447 reports positive persisted growth with nonforgetting preserved
**When** Exp 1459 builds the lineage decision artifact
**Then** `self_learning_headline_pivot_selected=true`
**And** `self_learning_lineage_retired=false`
**And** `exp1447_delta_overall` equals the Exp 1447 persisted growth delta
**And** the next allowed experiment shape is limited to the Exp 1447 verified
memory-policy mechanism with explicit metrics and nonforgetting threshold.

### SCENARIO-LEARN-1460: Missing Persisted Growth Retires Headline Scope

**Given** Exp 1447 does not report positive persisted growth or preserved
nonforgetting
**When** Exp 1459 builds the lineage decision artifact
**Then** `self_learning_headline_pivot_selected=false`
**And** `self_learning_lineage_retired=true`
**And** the honest verdict keeps self-learning in internal memory-policy scope
only.

## Implementation Status (REQ-LEARN-1459)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1459 | Implemented (`python/carnot/reporting/self_learning_lineage_decision.py`) | Implemented (`tests/python/test_self_learning_lineage_decision.py`) |

## REQ-LEARN-1471: FR-11 V8 Verified Memory Growth Pivot MUST Reuse Exp 1447 Policy

Exp 1471 SHALL execute the single allowed FR-11 v8 headline pivot on run date
`20260507`: reuse Exp 1447's asymmetric fresh/replay memory-promotion policy on
fresh verified local rows, optionally ingest Exp 1449 LTLZinc temporal rows only
as supporting benchmark feed, and report persisted SessionMemory growth plus
nonforgetting. The workflow SHALL NOT introduce a broad new self-learning
architecture or make adapter-only temporal claims.

### REQ-LEARN-1471 Sub-requirements

- REQ-LEARN-1471-1: The workflow SHALL write
  `results/experiment_1471_fr11_v8_verified_memory_growth_pivot.json` first
  with `status="in_progress"` before loading source artifacts.
- REQ-LEARN-1471-2: The workflow SHALL require Exp 1459's allowed pivot shape,
  Exp 1447 positive persisted growth, and Exp 1447-compatible nonforgetting
  threshold before any v8 headline result is allowed.
- REQ-LEARN-1471-3: Fresh candidates SHALL be verified local rows not already
  promoted by Exp 1447. Exp 1449 temporal rows MAY be converted into DVI-style
  candidates only after the local temporal verifier confirms each row's
  certificate state, and those rows SHALL be reported as supporting benchmark
  feed rather than a standalone headline result.
- REQ-LEARN-1471-4: The v8 memory update SHALL reuse the Exp 1447 asymmetric
  fresh/replay threshold policy and SHALL count positive self-learning growth
  only for verified new promotions persisted through SessionMemory.
- REQ-LEARN-1471-5: `headline_result_allowed` SHALL be true only when
  `self_learning_delta_overall > 0`, `new_promoted_count >= 1`, and
  `nonforgetting_rate >= 0.99`.
- REQ-LEARN-1471-6: If the headline gate fails, the artifact SHALL set
  `pivot_retired=true` and include a future-block rule that forbids rerunning
  the v8 pivot without a new verified fresh-row source and a changed root cause.
- REQ-LEARN-1471-7: The final artifact SHALL include `status`, `model_specs`,
  `live_sota_model_inference_used`, `self_learning_artifact_ready`,
  `baseline_fresh_verified_sample_count`, `fresh_verified_sample_count`,
  `self_learning_delta_overall`, `new_promoted_count`,
  `memory_entries_added`, `session_memory_updated`, `nonforgetting_rate`,
  `soundness_mistakes`, `completeness_mistakes`, `headline_result_allowed`,
  `pivot_preserved`, `pivot_retired`, and `honest_verdict`.

### SCENARIO-LEARN-1471: V8 Reuses Exp 1447 Policy And Preserves Pivot

**Given** Exp 1459 selected exactly one allowed future pivot
**And** Exp 1447 reports positive persisted growth with nonforgetting preserved
**And** fresh verified local rows include Exp 1449 temporal cases verified by
the local temporal checker
**When** v8 applies the Exp 1447 asymmetric policy and persists promoted rows
through SessionMemory
**Then** `new_promoted_count` equals `memory_entries_added`
**And** `self_learning_delta_overall` equals `memory_entries_added`
**And** `headline_result_allowed=true`
**And** `pivot_preserved=true`
**And** `pivot_retired=false`.

### SCENARIO-LEARN-1472: Failed V8 Gate Retires The Pivot

**Given** the v8 policy finds no persisted fresh verified promotions or fails
the nonforgetting threshold
**When** the terminal artifact is written
**Then** `headline_result_allowed=false`
**And** `pivot_retired=true`
**And** the artifact includes a future-block rule for this pivot shape.

## Implementation Status (REQ-LEARN-1471)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1471 | Implemented (`python/carnot/reporting/fr11_v8_verified_memory_growth_pivot.py`) | Implemented (`tests/python/test_fr11_v8_verified_memory_growth_pivot.py`) |

## REQ-LEARN-1472: Online Verifier Asymmetric Mistake-Budget Audit

Exp 1472 SHALL audit Exp 1471 under the online verifier learnability framing
from arXiv:2603.03538. The audit SHALL separate dangerous soundness mistakes
from conservative completeness mistakes before preserving any FR-11
self-learning claim. The audit SHALL treat soundness mistakes as higher cost
than completeness mistakes because a false accept can promote bad memory into a
future feedback loop, while a false flag mainly withholds usable rows.

### REQ-LEARN-1472 Sub-requirements

- REQ-LEARN-1472-1: The workflow SHALL write
  `results/experiment_1472_online_verifier_asymmetric_mistake_budget.json`
  first with `status="in_progress"` before loading Exp 1471.
- REQ-LEARN-1472-2: The audit SHALL read Exp 1471's
  `soundness_mistakes`, `completeness_mistakes`, headline gate fields, and
  available `memory_updates` ledger.
- REQ-LEARN-1472-3: The audit SHALL define explicit asymmetric cost weights
  where the soundness-mistake weight is greater than the completeness-mistake
  weight, and SHALL compute
  `asymmetric_cost_score = soundness_mistakes * soundness_weight +
  completeness_mistakes * completeness_weight`.
- REQ-LEARN-1472-4: The self-learning claim SHALL be preserved only when Exp
  1471 completed, allowed the headline result, preserved the pivot, and has
  acceptable soundness risk. The initial acceptable soundness budget is zero
  dangerous missed errors.
- REQ-LEARN-1472-5: The final artifact SHALL include `status`,
  `source_experiment`, `soundness_mistakes`, `completeness_mistakes`,
  `asymmetric_cost_weights`, `asymmetric_cost_score`, `pareto_decision`,
  `self_learning_claim_preserved`, `self_learning_claim_retired`,
  `audit_note_path`, and `honest_verdict`.
- REQ-LEARN-1472-6: The audit note at
  `docs/research-notes/fr11_v8_asymmetric_mistake_budget.md` SHALL state the
  decision and caveats, including that conservative false flags limit
  completeness rather than creating memory-corruption evidence.

### SCENARIO-LEARN-1473: Zero Soundness Mistakes Preserves Narrow Claim

**Given** Exp 1471 completed with `headline_result_allowed=true`,
`pivot_preserved=true`, zero soundness mistakes, and conservative completeness
mistakes
**When** Exp 1472 computes the asymmetric mistake budget
**Then** `self_learning_claim_preserved=true`
**And** `self_learning_claim_retired=false`
**And** `pareto_decision` records that the narrow claim is preserved on the
soundness frontier with completeness caveats.

### SCENARIO-LEARN-1474: Dangerous Soundness Mistakes Retire The Claim

**Given** Exp 1471 reports one or more soundness mistakes
**When** Exp 1472 computes the asymmetric mistake budget
**Then** `self_learning_claim_preserved=false`
**And** `self_learning_claim_retired=true`
**And** the honest verdict identifies unacceptable soundness risk.

## Implementation Status (REQ-LEARN-1472)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1472 | Implemented (`python/carnot/reporting/online_verifier_asymmetric_mistake_budget.py`) | Implemented (`tests/python/test_online_verifier_asymmetric_mistake_budget.py`) |

## REQ-LEARN-1484: FR-11 V9 Query-Time Memory Policy Replay

Exp 1484 SHALL integrate Exp 1471's verified FR-11 memory entries as a narrow
opt-in query-time verifier signal and SHALL compare verifier outcomes with
memory disabled versus enabled on the same bounded replay cases. The memory
signal SHALL NOT become global default behavior; callers that do not pass the
opt-in flag must receive baseline verifier behavior.

### REQ-LEARN-1484 Sub-requirements

- REQ-LEARN-1484-1: The workflow SHALL write
  `results/experiment_1484_fr11_v9_query_time_memory_policy.json` first with
  `status="in_progress"` before loading Exp 1471 or Exp 1472.
- REQ-LEARN-1484-2: The workflow SHALL load Exp 1471 verified promoted memory
  IDs as memory-positive replay cases and SHALL use Exp 1471 demoted IDs only
  as bounded negative controls. Demoted IDs SHALL NOT be inserted into the
  verified query-time memory index.
- REQ-LEARN-1484-3: The query-time policy path SHALL expose an explicit
  opt-in flag. With the flag disabled or omitted, memory hits SHALL be
  suppressed even when the replay case ID exists in verified memory.
- REQ-LEARN-1484-4: The replay audit SHALL evaluate memory-disabled and
  memory-enabled verifier outcomes on the same bounded replay set, then report
  `replay_cases_evaluated`, `baseline_task_success_rate`,
  `memory_task_success_rate`, and `task_success_delta`.
- REQ-LEARN-1484-5: The audit SHALL report soundness mistakes as false accepts
  from either the source asymmetric-budget audit or the query-time memory
  policy, and SHALL preserve source completeness mistakes as known false
  rejects while separately recording policy-level replay mistakes.
- REQ-LEARN-1484-6: `promotion_allowed` SHALL be true only when
  `soundness_mistakes == 0`, `task_success_delta >= 0.0`, non-forgetting is
  preserved, and the opt-in policy integration path is ready.
- REQ-LEARN-1484-7: The final artifact SHALL include `status`,
  `model_specs`, `policy_integration_ready`,
  `continuous_self_learning_task`, `replay_cases_evaluated`,
  `baseline_task_success_rate`, `memory_task_success_rate`,
  `task_success_delta`, `soundness_mistakes`, `completeness_mistakes`,
  `nonforgetting_rate`, `memory_policy_path`, `tests_run`,
  `promotion_allowed`, and `honest_verdict`.

### SCENARIO-LEARN-1484: Opt-In Memory Signal Improves Bounded Replay

**Given** Exp 1471 completed with verified promoted memory IDs, zero source
soundness mistakes, and non-forgetting preserved
**And** Exp 1472 preserved the narrow self-learning claim with zero dangerous
soundness mistakes
**When** Exp 1484 evaluates the same balanced replay cases with memory disabled
and then with the opt-in memory signal enabled
**Then** the memory-enabled task success rate is at least the baseline task
success rate
**And** `soundness_mistakes=0`
**And** `promotion_allowed=true` only when the non-negative delta and
non-forgetting gates also pass.

### SCENARIO-LEARN-1485: Dangerous Source Soundness Blocks Promotion

**Given** the source asymmetric-budget audit reports one or more soundness
mistakes
**When** Exp 1484 evaluates the query-time memory policy replay
**Then** `promotion_allowed=false`
**And** `policy_integration_ready=false`
**And** the honest verdict names the soundness-risk blocker even if the bounded
task-success delta is non-negative.

## Implementation Status (REQ-LEARN-1484)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1484 | Implemented (`python/carnot/pipeline/query_time_memory_policy.py`, `python/carnot/reporting/fr11_v9_query_time_memory_policy.py`) | Implemented (`tests/python/test_fr11_v9_query_time_memory_policy.py`) |

## REQ-LEARN-1485: FR-11 Completeness-Reduction Audit Preserves Zero Soundness

Exp 1485 SHALL audit Exp 1484's per-case query-time memory replay ledger for
conservative threshold or routing variants that reduce bounded replay
completeness mistakes without introducing any soundness mistake. The audit
SHALL reject every candidate whose soundness mistakes exceed the baseline
soundness mistakes or exceed zero, even when that candidate reduces false
rejects.

### REQ-LEARN-1485 Sub-requirements

- REQ-LEARN-1485-1: The workflow SHALL write
  `results/experiment_1485_fr11_completeness_reduction_audit.json` first with
  `status="in_progress"` before reading the source replay artifact.
- REQ-LEARN-1485-2: The audit SHALL load Exp 1484's
  `memory_policy_replay` per-case ledger, using `baseline_memory_disabled` as
  the baseline and evaluating candidate variants against the same replay case
  IDs.
- REQ-LEARN-1485-3: Candidate selection SHALL minimize completeness mistakes
  only among candidates with `candidate_soundness_mistakes <=
  baseline_soundness_mistakes` and `candidate_soundness_mistakes == 0`.
- REQ-LEARN-1485-4: The final artifact SHALL include `status`,
  `source_experiment`, `completeness_reduction_audit_complete`,
  `baseline_completeness_mistakes`, `candidate_completeness_mistakes`,
  `completeness_mistake_delta`, `baseline_soundness_mistakes`,
  `candidate_soundness_mistakes`, `candidate_policy`,
  `policy_change_allowed`, `tests_run`, and `honest_verdict`.
- REQ-LEARN-1485-5: `policy_change_allowed` SHALL be true only when the best
  allowed candidate has zero soundness mistakes and a negative
  `completeness_mistake_delta`.

### SCENARIO-LEARN-1486: Verified-Memory Routing Reduces False Rejects

**Given** Exp 1484 completed with a baseline replay ledger and an opt-in
verified-memory candidate ledger
**When** Exp 1485 audits candidate routing variants on the same replay case IDs
**Then** the selected candidate has zero soundness mistakes
**And** its completeness mistakes are lower than the baseline completeness
mistakes
**And** `policy_change_allowed=true`.

### SCENARIO-LEARN-1487: Unsafe Broad Routing Is Rejected

**Given** a candidate routing variant that accepts negative-control replay IDs
**When** Exp 1485 compares that candidate with the baseline soundness mistakes
**Then** the candidate is rejected from best-candidate selection
**And** `policy_change_allowed` is not based on that unsafe candidate.

## Implementation Status (REQ-LEARN-1485)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1485 | Implemented (`python/carnot/reporting/fr11_completeness_reduction_audit.py`) | Implemented (`tests/python/test_fr11_completeness_reduction_audit.py`) |

## REQ-LEARN-1497: FR-11 V10 Trace2Skill Daily Evaluation Cadence And Rot Check

Exp 1497 SHALL convert the bounded Exp 1484/1485 query-time memory replay into
a daily trace2skill evaluation manifest. The cadence SHALL produce durable
per-skill/case rows, deterministic promotion/retirement decisions, and rot
checks without claiming broad autonomous learning beyond the measured replay
suite.

### REQ-LEARN-1497 Sub-requirements

- REQ-LEARN-1497-1: The workflow SHALL write
  `results/experiment_1497_fr11_trace2skill_daily_eval_v10.json` first with
  `status="in_progress"` before loading the v9 source artifacts.
- REQ-LEARN-1497-2: The workflow SHALL load
  `results/experiment_1484_fr11_v9_query_time_memory_policy.json` and
  `results/experiment_1485_fr11_completeness_reduction_audit.json`, then
  derive a bounded daily-eval manifest row for each paired replay case in the
  Exp 1484 baseline and memory-enabled ledgers.
- REQ-LEARN-1497-3: Each manifest row SHALL include a trace-derived `skill_id`,
  the replay `case_id`, expected resolver checks, baseline outcome,
  memory-assisted outcome, rot criteria, rot reasons, and a decision of
  `promote`, `retain`, or `retire`.
- REQ-LEARN-1497-4: A skill SHALL be retired when any rot criterion is true:
  missing source artifact, unresolved verifier dependency, reduced task
  success, new soundness mistake, or schema drift.
- REQ-LEARN-1497-5: A skill SHALL be promoted only when it is not rotted, has
  zero soundness mistakes, and the memory-assisted outcome improves the paired
  baseline outcome. Non-improving but non-rotted skills SHALL be retained.
- REQ-LEARN-1497-6: `daily_eval_manifest_ready` SHALL be true only after
  `results/fr11_trace2skill_daily_eval_manifest_1497.jsonl` exists and the
  promoted, rotted, and retired skill counts are explicit.
- REQ-LEARN-1497-7: The final artifact SHALL include `status`,
  `continuous_self_learning_task`, `model_specs`, `daily_eval_manifest_ready`,
  `trace2skill_cases_evaluated`, `skills_evaluated`, `rotted_skill_count`,
  `promoted_skill_count`, `retired_skill_count`,
  `baseline_task_success_rate`, `memory_task_success_rate`,
  `task_success_delta`, `soundness_mistakes`, `completeness_mistakes`,
  `daily_eval_manifest_path`, `models_used`, `gpu_probe`, `blockers`, and
  `honest_verdict`.
- REQ-LEARN-1497-8: If no LLM judge or generator is needed, the artifact SHALL
  state that deterministic replay evaluation was sufficient while preserving
  the mandated local SOTA GGUF model specs for future non-deterministic runs.

### SCENARIO-LEARN-1497: Daily Eval Promotes Improved Verified-Memory Skills

**Given** Exp 1484 reports a bounded replay with paired baseline and
memory-enabled decisions
**And** Exp 1485 selected the zero-soundness verified-memory policy
**When** Exp 1497 builds the trace2skill daily manifest
**Then** each memory-improved, zero-soundness row is promoted
**And** non-improving but non-rotted rows are retained
**And** the terminal artifact reports the same bounded task-success delta as
the paired manifest rows.

### SCENARIO-LEARN-1498: Rot Criteria Retire Stale Or Harmful Skills

**Given** a trace-derived skill has a missing source artifact, unresolved
verifier dependency, reduced task success, a new soundness mistake, or schema
drift
**When** Exp 1497 evaluates the daily manifest row
**Then** that skill is marked rotted
**And** its decision is `retire`
**And** the rotted and retired counts include it explicitly.

## Implementation Status (REQ-LEARN-1497)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1497 | Implemented (`python/carnot/reporting/fr11_trace2skill_daily_eval.py`) | Implemented (`tests/python/test_fr11_trace2skill_daily_eval.py`) |

## REQ-LEARN-1498: Trace2Skill Artifact Reachability Audit

Exp 1498 SHALL audit the Exp 1497 trace2skill daily-eval manifest for live
source evidence, resolver freshness, and verifier-dependency reachability
before learned skills are trusted as promotable evidence.

### REQ-LEARN-1498 Sub-requirements

- REQ-LEARN-1498-1: The workflow SHALL write
  `results/experiment_1498_trace2skill_artifact_reachability_audit.json`
  first with `status="in_progress"` before loading gated Exp 1497 inputs.
- REQ-LEARN-1498-2: The workflow SHALL require both
  `results/experiment_1497_fr11_trace2skill_daily_eval_v10.json` and
  `results/fr11_trace2skill_daily_eval_manifest_1497.jsonl`; if either is
  absent or malformed, it SHALL write a terminal gated artifact with explicit
  blockers.
- REQ-LEARN-1498-3: The workflow SHALL parse each daily-eval manifest row and
  collect skill IDs, source artifact references, resolver keys, model
  references, and verifier dependencies.
- REQ-LEARN-1498-4: Each referenced JSON, JSONL, or YAML artifact SHALL exist,
  parse successfully, and expose expected key fields before it is counted as
  reachable.
- REQ-LEARN-1498-5: The workflow SHALL mark evidence links as unreachable when
  the path is missing or unparseable, stale when required keys or status are no
  longer compatible with the manifest contract, and ambiguous when resolver
  keys are missing, duplicated, or disagree across rows.
- REQ-LEARN-1498-6: Missing, stale, and ambiguous evidence SHALL produce
  explicit repair or retirement decisions without deleting learned skills.
- REQ-LEARN-1498-7: The terminal artifact SHALL include `status`,
  `artifact_reachability_audit_complete`, `gated_inputs_present`,
  `skills_checked`, `source_artifacts_checked`, `reachable_artifact_count`,
  `unreachable_artifact_count`, `stale_artifact_count`,
  `ambiguous_resolver_count`, `repair_decisions`, `retirement_decisions`,
  `blockers`, and `honest_verdict`.

### SCENARIO-LEARN-1498-A: Reachable Evidence Keeps Promoted Skills Live

**Given** Exp 1497 wrote a complete daily-eval artifact and manifest
**And** every manifest source artifact resolves, parses, and exposes expected
fields
**When** Exp 1498 audits the manifest
**Then** the terminal artifact reports the checked skill and source-artifact
counts
**And** no repair or retirement decision is required.

### SCENARIO-LEARN-1498-B: Missing Or Ambiguous Evidence Produces Decisions

**Given** a daily-eval manifest row references a missing source artifact or
contains stale/ambiguous resolver evidence
**When** Exp 1498 audits the manifest
**Then** the terminal artifact records the unreachable, stale, or ambiguous
counts
**And** it emits explicit repair or retirement decisions for the affected
learned skill evidence without deleting any skill.

## Implementation Status (REQ-LEARN-1498)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1498 | Implemented (`python/carnot/reporting/trace2skill_artifact_reachability_audit.py`) | Implemented (`tests/python/test_trace2skill_artifact_reachability_audit.py`) |

## REQ-LEARN-1512: FR-11 Verifier-Feedback Query-Time Policy Cache

Exp 1512 SHALL build a bounded query-time policy cache from existing FR-11
daily-eval rows and executable monitor events. The cache MAY affect retrieval,
routing, verifier escalation, or continuation preference during query-time
replay, but it SHALL NOT mutate model weights, bypass deterministic validators,
or promote learned skills before a rollback audit.

### REQ-LEARN-1512 Sub-requirements

- REQ-LEARN-1512-1: The workflow SHALL write
  `results/experiment_1512_fr11_verifier_feedback_policy_cache_v11.json` first
  with `status="in_progress"` before loading Exp 1497 or Exp 1509 sources.
- REQ-LEARN-1512-2: The workflow SHALL load
  `results/fr11_trace2skill_daily_eval_manifest_1497.jsonl` and, when present,
  `results/executable_monitor_events_1509.jsonl`; if Exp 1509 monitor events
  are absent, the workflow SHALL continue from Exp 1497 rows and record the
  source gap as a blocker.
- REQ-LEARN-1512-3: The policy update rules SHALL be explicit and limited to
  query-time actions: retrieval boost or demotion, routing preference,
  continuation preference, verifier escalation, and skill quarantine.
- REQ-LEARN-1512-4: The replay SHALL write
  `results/fr11_policy_cache_events_1512.jsonl` with one row per source event,
  including source identity, proposed policy action, acceptance state, and any
  rejection or quarantine reason.
- REQ-LEARN-1512-5: The workflow SHALL reject or quarantine any proposed update
  associated with false accepts, unreachable source artifacts, stale
  provenance, or missing deterministic validation.
- REQ-LEARN-1512-6: `policy_cache_ready` SHALL be true only when the manifest
  exists, `no_model_weight_mutation=true`, `soundness_mistakes=0`, and
  `promotion_requires_rollback_audit=true`.
- REQ-LEARN-1512-7: The terminal artifact SHALL include `status`,
  `continuous_self_learning_task`, `policy_cache_ready`,
  `no_model_weight_mutation`, `source_events_loaded`,
  `policy_updates_proposed`, `policy_updates_accepted`,
  `policy_update_rules`, `soundness_mistakes`,
  `verifier_false_accept_rate`, `policy_cache_manifest_path`,
  `promotion_requires_rollback_audit`, `blockers`, and `honest_verdict`.

### SCENARIO-LEARN-1512: Deterministic Verifier Feedback Builds A Bounded Cache

**Given** Exp 1497 daily-eval rows and Exp 1509 monitor events with zero
verifier false accepts and deterministic validation statuses
**When** Exp 1512 applies the policy update rules
**Then** the same inputs produce the same ordered policy-cache rows each time
**And** accepted rows affect only query-time retrieval, routing, continuation,
or verifier-escalation choices
**And** the terminal artifact reports the proposed and accepted update counts.

### SCENARIO-LEARN-1513: Promotion Waits For Rollback Audit

**Given** a source row would otherwise propose a retrieval boost or
continuation preference
**When** the row is associated with a false accept, unreachable artifact, stale
provenance, or missing deterministic validation
**Then** the update is rejected or quarantined
**And** no learned skill is promoted by Exp 1512
**And** `promotion_requires_rollback_audit` remains true.

## Implementation Status (REQ-LEARN-1512)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1512 | Implemented (`python/carnot/reporting/fr11_verifier_feedback_policy_cache.py`) | Implemented (`tests/python/test_fr11_verifier_feedback_policy_cache.py`) |

## REQ-LEARN-1513: FR-11 Policy Rollback Replay Audit

Exp 1513 SHALL audit the Exp 1512 verifier-feedback policy cache before any
promotion. The audit SHALL replay the bounded deterministic source sessions
with baseline policy behavior and with each proposed query-time policy update,
then emit keep or rollback decisions per update. The workflow SHALL NOT
generate new LLM rows or mutate model weights.

### REQ-LEARN-1513 Sub-requirements

- REQ-LEARN-1513-1: The workflow SHALL write
  `results/experiment_1513_fr11_policy_rollback_replay_audit.json` first with
  `status="in_progress"` before loading Exp 1512 or replay manifests.
- REQ-LEARN-1513-2: The workflow SHALL require
  `results/experiment_1512_fr11_verifier_feedback_policy_cache_v11.json` with
  `policy_cache_ready=true` and
  `results/fr11_policy_cache_events_1512.jsonl`; if either gate is absent or
  not ready, the workflow SHALL write a terminal gated artifact.
- REQ-LEARN-1513-3: Replay rows SHALL be derived only from deterministic
  checked-in artifacts, including Exp 1512 policy-cache rows, Exp 1497
  trace2skill daily-eval rows, and Exp 1509 executable monitor events when
  referenced.
- REQ-LEARN-1513-4: Each policy update SHALL be replayed with a baseline
  outcome and a proposed-policy outcome, reporting update-level
  `false_accept_delta`, `utility_delta`, `soundness_mistakes`,
  evidence reachability, deterministic validator support, and a decision of
  `keep` or `rollback`.
- REQ-LEARN-1513-5: The audit SHALL roll back any update that increases false
  accepts, references stale or unreachable evidence, has any soundness mistake,
  is quarantined by Exp 1512, or lacks deterministic validator support.
- REQ-LEARN-1513-6: The replay manifest SHALL be written to
  `results/fr11_policy_rollback_replay_1513.jsonl` with one row per policy
  update and SHALL be deterministic for identical inputs.
- REQ-LEARN-1513-7: `rollback_audit_passed` SHALL be true only when accepted
  updates have `soundness_mistakes=0` and aggregate accepted
  `false_accept_delta <= 0`.
- REQ-LEARN-1513-8: The terminal artifact SHALL include `status`,
  `rollback_audit_passed`, `gated_inputs_present`,
  `policy_updates_replayed`, `counterfactual_sessions`,
  `accepted_policy_updates`, `rolled_back_policy_updates`,
  `soundness_mistakes`, `false_accept_delta`, `utility_delta`,
  `rollback_manifest_path`, `blockers`, and `honest_verdict`.

### SCENARIO-LEARN-1514: Safe Counterfactual Updates Are Kept

**Given** Exp 1512 is policy-cache ready and a policy update has reachable
source evidence, deterministic validator support, no rejection reasons, and no
false accepts
**When** Exp 1513 replays the baseline and proposed-policy outcomes from
deterministic manifests
**Then** the update decision is `keep`
**And** accepted aggregate `soundness_mistakes=0`
**And** accepted aggregate `false_accept_delta <= 0`.

### SCENARIO-LEARN-1515: Unsafe Or Unsupported Updates Are Rolled Back

**Given** a proposed policy update increases false accepts, references stale or
unreachable evidence, was quarantined by Exp 1512, has a soundness mistake, or
lacks deterministic validator support
**When** Exp 1513 evaluates the update-level rollback rules
**Then** the update decision is `rollback`
**And** the manifest row records the rollback reasons.

## Implementation Status (REQ-LEARN-1513)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1513 | Implemented (`python/carnot/reporting/fr11_policy_rollback_replay_audit.py`) | Implemented (`tests/python/test_fr11_policy_rollback_replay_audit.py`) |

## REQ-LEARN-1514: Trace2Skill Portable Skill/Provenance Pack

Exp 1514 SHALL build a compact portable trace2skill skill/provenance pack from
Exp 1513 rollback replay rows. The pack SHALL include only rollback-passing
entries whose source evidence is reachable and whose verifier support is
deterministic. The workflow SHALL NOT promote or package entries that failed
rollback, lack reachability, reference stale evidence, or lack deterministic
validator support.

### REQ-LEARN-1514 Sub-requirements

- REQ-LEARN-1514-1: The workflow SHALL write
  `results/experiment_1514_trace2skill_portable_skill_pack_v2.json` first
  with `status="in_progress"` before loading Exp 1498 or Exp 1513 sources.
- REQ-LEARN-1514-2: The workflow SHALL require Exp 1513 to report
  `rollback_audit_passed=true`, require the Exp 1513 rollback manifest to
  exist, and require Exp 1498 to report completed artifact reachability with
  zero unreachable, stale, or ambiguous resolver findings. If any gate is
  absent, it SHALL write a terminal gated artifact.
- REQ-LEARN-1514-3: Eligible entries SHALL have `decision="keep"`,
  `source_evidence_reachable=true`, `source_evidence_stale=false`,
  `deterministic_validator_supported=true`, `soundness_mistakes=0`,
  `false_accept_delta <= 0`, and a non-empty `skill_id`.
- REQ-LEARN-1514-4: The portable pack manifest SHALL record each packaged
  entry's source artifact, verifier evidence, resolver key, created date, and
  promotion status. Rejected entries SHALL be listed with deterministic
  rejection reasons and SHALL NOT be promoted.
- REQ-LEARN-1514-5: The terminal artifact SHALL include `status`,
  `portable_skill_pack_ready`, `gated_inputs_present`,
  `rollback_passing_entries`, `packaged_skill_entries`,
  `rejected_skill_entries`, `provenance_fields_present`,
  `resolver_keys_present`, `pack_manifest_path`, `ops_note_path`, `blockers`,
  and `honest_verdict`.

### SCENARIO-LEARN-1516: Rollback-Passing Entries Become Portable

**Given** Exp 1513 passed rollback audit and rollback rows are kept with
reachable source evidence and deterministic validator support
**When** Exp 1514 builds the portable manifest
**Then** every eligible row is packaged with provenance fields, resolver key,
created date, and `promotion_status="packaged_rollback_passed"`
**And** `portable_skill_pack_ready=true`.

### SCENARIO-LEARN-1517: Unsupported Entries Are Rejected Without Promotion

**Given** a rollback row failed rollback, references unreachable or stale
evidence, or lacks deterministic validator support
**When** Exp 1514 evaluates pack eligibility
**Then** the row appears only in `rejected_entries`
**And** its promotion status is `rejected_not_promoted`.

## Implementation Status (REQ-LEARN-1514)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1514 | Implemented (`python/carnot/reporting/trace2skill_portable_skill_pack.py`) | Implemented (`tests/python/test_trace2skill_portable_skill_pack.py`) |

## REQ-LEARN-1524: FR-11 Live Policy Promotion From Rollback-Passing Updates

Exp 1524 SHALL promote only rollback-passing FR-11 query-time policy updates
into a bounded live contract-guided evaluation loop. The learning mechanism is
policy/cache adaptation only: it SHALL NOT train, finetune, write checkpoints,
or mutate model parameters. Legacy small models MAY be used only for CPU smoke
tests and SHALL NOT produce headline live self-learning results.

### REQ-LEARN-1524 Sub-requirements

- REQ-LEARN-1524-1: The workflow SHALL write
  `results/experiment_1524_fr11_live_policy_promotion_v12.json` first with
  `status="in_progress"` before loading Exp 1512, Exp 1513, Exp 1514, or
  Exp 1520 sources.
- REQ-LEARN-1524-2: The workflow SHALL load the Exp 1512 policy-cache artifact
  and manifest, the Exp 1513 rollback replay artifact and manifest, the
  Exp 1514 portable skill/provenance artifact and manifest, and the Exp 1520
  runtime-contract artifact and manifest.
- REQ-LEARN-1524-3: Eligible policy updates SHALL have an Exp 1513 replay
  decision of `keep`, `source_evidence_reachable=true`,
  `source_evidence_stale=false`, `deterministic_validator_supported=true`,
  `soundness_mistakes=0`, `false_accept_delta <= 0`, and matching Exp 1514
  packaged provenance with `promotion_status="packaged_rollback_passed"`.
- REQ-LEARN-1524-4: Every rejected update SHALL remain unpromoted and SHALL
  record deterministic rejection reasons such as rollback failure, stale or
  unreachable provenance, missing deterministic validator support, false accept
  increase, soundness mistake, or missing portable provenance.
- REQ-LEARN-1524-5: Live headline evaluation SHALL resolve the mandated local
  SOTA GGUF models via `cached_sota_pair()` or the same local GGUF cache
  pattern. If no mandated SOTA GGUF completes inference, the workflow SHALL
  write a terminal blocker and SHALL NOT report legacy tiny-model headline
  results.
- REQ-LEARN-1524-6: The evaluation manifest SHALL be written to
  `results/fr11_live_policy_promotion_1524.jsonl` with one row per promoted
  update and contract case, plus one summary row. Each row SHALL record model
  provenance, baseline policy outcome, promoted policy outcome, utility delta,
  false-accept delta, soundness mistakes, and runtime-contract validation
  evidence.
- REQ-LEARN-1524-7: `live_policy_promotion_ready` SHALL be true only when at
  least one promoted update is evaluated with a mandated live SOTA GGUF,
  `soundness_mistakes=0`, aggregate `false_accept_delta <= 0`, and
  `no_model_weight_mutation=true`.
- REQ-LEARN-1524-8: The terminal artifact SHALL include `status`,
  `continuous_self_learning_task`, `model_specs`,
  `live_sota_model_inference_used`, `live_policy_promotion_ready`,
  `rollback_passing_updates_loaded`, `promoted_policy_updates`,
  `baseline_task_success_rate`, `promoted_task_success_rate`,
  `utility_delta`, `false_accept_delta`, `soundness_mistakes`,
  `no_model_weight_mutation`, `policy_promotion_manifest_path`, `models_used`,
  `blockers`, and `honest_verdict`.

### SCENARIO-LEARN-1524: Rollback-Passing Provenance Is Promoted Live

**Given** Exp 1513 kept a policy update with reachable evidence and Exp 1514
packaged matching provenance
**And** Exp 1520 provides deterministic runtime-contract cases
**And** a mandated local SOTA GGUF completes inference
**When** Exp 1524 runs baseline policy and promoted policy evaluation
**Then** the promoted update is evaluated through the runtime-contract ledger
**And** the manifest row records zero soundness mistakes and no positive
false-accept delta
**And** the terminal artifact may set `live_policy_promotion_ready=true`.

### SCENARIO-LEARN-1525: Unsupported Updates And Missing SOTA Stay Blocked

**Given** a policy update failed rollback, lacks reachable provenance, lacks
deterministic validator support, increases false accepts, has a soundness
mistake, or is absent from the portable pack
**When** Exp 1524 filters promotion candidates
**Then** the update is rejected and is not promoted.

**Given** no mandated local SOTA GGUF completes inference
**When** Exp 1524 attempts live evaluation
**Then** the workflow writes a terminal blocker
**And** `live_policy_promotion_ready=false`.

## Implementation Status (REQ-LEARN-1524)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1524 | Planned (`python/carnot/reporting/fr11_live_policy_promotion.py`) | Planned (`tests/python/test_fr11_live_policy_promotion.py`) |
