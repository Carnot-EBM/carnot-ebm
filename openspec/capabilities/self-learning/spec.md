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

## REQ-LEARN-2357: FR-11 Multidomain Fast-Slow Retention

**Given** a Fast-Slow trainer with verifier-initialized slow constraints and
online fast verifier context
**When** the trainer processes 30 verified arithmetic examples, 30 verified code
examples, and 30 verified logic examples with fixed seed 42
**Then** the trainer SHALL keep slow weights unchanged, update fast constraints
from verifier-approved examples, and report
`results/experiment_2357_fr11_multidomain.json`.

### REQ-LEARN-2357 Sub-requirements

- REQ-LEARN-2357-1: `carnot.learning.fast_slow.FastSlowTrainer` SHALL expose
  `slow_weights`, `fast_weights`, `update_fast(query, verification_result)`,
  and `predict(query)`.
- REQ-LEARN-2357-2: The experiment SHALL use exactly 3 domains: arithmetic,
  code, and logic.
- REQ-LEARN-2357-3: Each domain SHALL use 30 verified training queries and 10
  holdout queries.
- REQ-LEARN-2357-4: The artifact SHALL include `honest_verdict`,
  `fr11_multidomain_passed`, `cross_domain_retention_rate`,
  `continuous_self_learning_validated`, `n_domains`, and `random_seed`.
- REQ-LEARN-2357-5: `fr11_multidomain_passed` SHALL be true exactly when
  `cross_domain_retention_rate >= 0.75`.

### SCENARIO-LEARN-2357: FST Retains Prior Domains After Later Training

**Given** arithmetic fast constraints have been learned and then code fast
constraints have been learned
**When** arithmetic holdout accuracy is measured again
**Then** the retained arithmetic accuracy contributes to
`cross_domain_retention_rate`.

**Given** logic fast constraints have subsequently been learned
**When** arithmetic and code holdouts are measured again
**Then** their mean retained accuracy across prior-domain checks is at least
0.75 for an FR-11 pass.

Spec: REQ-LEARN-2357, SCENARIO-LEARN-2357

## Implementation Status (REQ-LEARN-2357)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-2357 | Implemented (`python/carnot/learning/fast_slow.py`, `python/carnot/reporting/fr11_multidomain_fst_retention.py`) | Python (`tests/python/test_experiment_2357_fr11_multidomain.py`) |

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

## REQ-LEARN-2151: NLA Confidence Gate on FR-11 Policy Retention

**Given** the FR-11 self-learning loop proposes a policy candidate
**When** the NLA-gated retention gate evaluates the candidate
**Then** the candidate is retained ONLY when `nla_confidence > 0.7`
**And** candidates with `nla_confidence <= 0.7` are rejected before entering the policy pool
**And** `soundness_mistakes` remains 0 after all retained candidates are evaluated

This gate prevents overfit or mode-collapsed policies from entering the
active pool.  High NLA confidence means the SAE can faithfully reconstruct
model activations for the candidate repair — a proxy for well-understood,
non-hallucinated feature directions.

### REQ-LEARN-2151 Sub-requirements

- REQ-LEARN-2151-1: `NLAGatedSelfLearner.__init__` SHALL accept `nla_threshold`
  (default 0.7) and `max_iterations` (default 10).
- REQ-LEARN-2151-2: `NLAGatedSelfLearner.run_iteration` SHALL retain each
  candidate whose `nla_confidence` is STRICTLY GREATER than `nla_threshold`.
  Candidates at or below the threshold are counted in `n_rejected_by_nla`.
- REQ-LEARN-2151-3: `NLAGatedSelfLearner.run_loop` SHALL run AT MOST
  `max_iterations` iterations regardless of how many batches are supplied.
- REQ-LEARN-2151-4: `NLAGatedLoopResult` SHALL accumulate `total_retained`,
  `total_rejected`, and `soundness_mistakes` across all iterations.

---

## SCENARIO-LEARN-2151: Mixed-Confidence Batch Selects High-Confidence Candidates Only

**Given** a batch of 5 candidates where 3 have `nla_confidence > 0.7`
**When** `run_iteration` is called
**Then** exactly 3 candidates are retained
**And** 2 candidates are rejected
**And** `soundness_mistakes` == 0

---

## SCENARIO-LEARN-2152: soundness_mistakes Stays Zero When Retained Candidates Are Correct

**Given** a 10-iteration loop where every retained candidate has `is_correct=True`
**When** the loop completes
**Then** `soundness_mistakes` == 0 across the entire run
**And** `iterations_run` == 10

---

## Implementation Status (NLA Gate)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-2151 | Implemented (pipeline/nla_gated_self_learning.py) | test_nla_gated_self_learning.py |

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

---

## REQ-LEARN-1594: CerCE Certificate Ledger For FR-11 Policy Promotion

Exp 1594 SHALL build a CerCE-style certificate ledger for FR-11 policy
promotion.  The ledger SHALL preserve constraint-violation evidence from the
FR-11 update loop and promotion evaluation rows so a policy update can only be
marked promotion-safe when its certificate shows non-forgetting, zero accepted
violations, and no positive false-accept delta.  The mechanism is query-time
policy bookkeeping only: it SHALL NOT train, finetune, write checkpoints, or
mutate model parameters.

### REQ-LEARN-1594 Sub-requirements

- REQ-LEARN-1594-1: The workflow SHALL write
  `results/experiment_1594_cerce_ledger.json` first with
  `status="in_progress"` before loading promotion, rollback, or FR-11 event
  inputs.
- REQ-LEARN-1594-2: The ledger SHALL record one normalized certificate row per
  policy update and constraint case. Each row SHALL include `policy_update_id`,
  `constraint_id`, `constraint_type`, `source`, `baseline_violation`,
  `promoted_violation`, `accepted_violation`, `false_accept_delta`,
  `soundness_mistake`, and a deterministic `certificate_id`.
- REQ-LEARN-1594-3: The ledger SHALL expose an FR-11 event subscriber hook so
  `ViolationEvent` records from the update loop increment constraint-violation
  counts without bypassing the existing EventBus semantics.
- REQ-LEARN-1594-4: A policy certificate SHALL be promotion-safe only when every
  row for that policy has `accepted_violation=false`,
  `soundness_mistake=false`, and aggregate `false_accept_delta <= 0`.
- REQ-LEARN-1594-5: `nonforgetting_certificate_rate` SHALL equal the fraction
  of policies whose certificate is promotion-safe.  `cerce_ledger_ready` SHALL
  be true only when at least one policy certificate exists and the rate is
  exactly 1.0.
- REQ-LEARN-1594-6: The terminal artifact SHALL include `status`, `schema`,
  `continuous_self_learning_task`, `cerce_ledger_ready`,
  `policy_certificates_evaluated`, `constraint_violation_records`,
  `fr11_events_recorded`, `accepted_violation_count`,
  `false_accept_delta`, `nonforgetting_certificate_rate`,
  `promotion_safe_policy_updates`, `blocked_policy_updates`,
  `ledger_rows`, `blockers`, and `honest_verdict`.

### SCENARIO-LEARN-1594: FR-11 Events And Promotion Rows Produce Safe Certificates

**Given** FR-11 emits constraint-violation events for a policy update
**And** the promotion evaluation rows show zero accepted violations, zero
soundness mistakes, and no positive false-accept delta
**When** Exp 1594 builds the CerCE ledger
**Then** the policy certificate is promotion-safe
**And** `cerce_ledger_ready=true`
**And** the terminal artifact records the policy id in
`promotion_safe_policy_updates`.

### SCENARIO-LEARN-1595: Accepted Violations Block Policy Promotion

**Given** a promotion evaluation row accepts a constraint case whose expected
contract label is reject
**When** Exp 1594 builds the CerCE ledger
**Then** the policy certificate is not promotion-safe
**And** the policy id is recorded in `blocked_policy_updates`
**And** `cerce_ledger_ready=false`.

## Implementation Status (REQ-LEARN-1594)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1594 | Implemented (`python/carnot/training/cerce_certificate_ledger.py`) | Implemented (`tests/python/test_experiment_1594_cerce_ledger.py`) |

---

## REQ-LEARN-1595: CerCE Ledger Pre/Post Bounds Check

Exp 1595 SHALL execute a pre/post bounds check on the CerCE ledger using local models. The workflow SHALL run simulated policy updates against the ledger and reject them if the bound worsens. It SHALL write the terminal artifact to `results/experiment_1595_cerce_bounds.json`.

### REQ-LEARN-1595 Sub-requirements

- REQ-LEARN-1595-1: The workflow SHALL write `results/experiment_1595_cerce_bounds.json` first with `status="in_progress"`.
- REQ-LEARN-1595-2: The workflow SHALL run simulated policy updates against the CerCE ledger bounds.
- REQ-LEARN-1595-3: If a simulated update worsens the bounds, the update SHALL be rejected.
- REQ-LEARN-1595-4: The terminal artifact SHALL include `status`, `schema`, `continuous_self_learning_task`, `bounds_check_passed`, `simulated_updates_run`, `rejected_updates`, and `honest_verdict`.

### SCENARIO-LEARN-1595-A: Worsened Bound Rejects Update

**Given** a simulated policy update
**When** Exp 1595 runs the bounds check and the bound worsens
**Then** the update is rejected
**And** it is recorded in `rejected_updates`.

### SCENARIO-LEARN-1595-B: Valid Bounds Pass Check

**Given** simulated policy updates where the bound does not worsen
**When** Exp 1595 runs the bounds check
**Then** the updates are not rejected
**And** `bounds_check_passed=true` and `honest_verdict` is `complete: cerce_bounds_checked`.

## Implementation Status (REQ-LEARN-1595)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1595 | Implemented (`python/carnot/training/cerce_bounds_check.py`) | Implemented (`tests/python/test_experiment_1595_cerce_bounds.py`) |

---

## REQ-LEARN-1608: FR-11 Continuous Self-Learning CerCE Scale Run

Exp 1608 SHALL run the FR-11 continuous self-learning loop over exactly 1000
deterministic examples while enforcing CerCE no-forgetting bounds.  The run
SHALL use the CerCE certificate ledger to replay prior certificates and to
record every new example-level update before claiming utility.  A terminal
artifact SHALL be written to `results/experiment_1608_fr11_cerce.json`, and
the artifact SHALL include `continuous_self_learning_task`.

### REQ-LEARN-1608 Sub-requirements

- REQ-LEARN-1608-1: The workflow SHALL write
  `results/experiment_1608_fr11_cerce.json` first with `status="in_progress"`
  before loading prior CerCE certificates or running the training loop.
- REQ-LEARN-1608-2: The training loop SHALL process exactly 1000 examples and
  SHALL record one CerCE ledger row per example-level update.
- REQ-LEARN-1608-3: The workflow SHALL replay prior CerCE certificate rows from
  `results/experiment_1594_cerce_ledger.json` into the ledger before evaluating
  the new policy updates.
- REQ-LEARN-1608-4: `utility_delta` SHALL be strictly positive only when the
  promoted self-learning policy has a higher success rate than the baseline on
  the 1000-example loop.
- REQ-LEARN-1608-5: The terminal artifact SHALL report a safe complete result
  only when the prior and new certificate rows have zero accepted violations,
  zero soundness mistakes, non-positive false-accept delta, and
  `nonforgetting_certificate_rate == 1.0`.
- REQ-LEARN-1608-6: The terminal artifact SHALL include `status`, `schema`,
  `continuous_self_learning_task`, `examples_processed`, `utility_delta`,
  `cerce_ledger_ready`, `bounds_check_passed`, `past_certificates_checked`,
  `accepted_violation_count`, `false_accept_delta`,
  `nonforgetting_certificate_rate`, `no_model_weight_mutation`, and
  `honest_verdict`.

### SCENARIO-LEARN-1608: 1000-Example Scale Run Preserves No-Forgetting

**Given** a prior CerCE ledger whose policy certificates are promotion-safe
**When** Exp 1608 runs the FR-11 self-learning loop over 1000 examples
**Then** the terminal artifact reports `examples_processed=1000`
**And** `utility_delta > 0`
**And** `cerce_ledger_ready=true`
**And** `nonforgetting_certificate_rate=1.0`
**And** `continuous_self_learning_task=true`.

### SCENARIO-LEARN-1609: Worsened Certificate Blocks The Scale Run

**Given** a prior or new certificate row accepts a violation, introduces a
soundness mistake, or increases false accepts
**When** Exp 1608 evaluates the CerCE ledger
**Then** the terminal artifact is blocked
**And** `cerce_ledger_ready=false`
**And** the unsafe policy id is listed in `blocked_policy_updates`.

## Implementation Status (REQ-LEARN-1608)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1608 | Implemented (`python/carnot/training/fr11_cerce_scale.py`) | Implemented (`tests/python/test_experiment_1608_fr11_cerce.py`) |

---

## REQ-LEARN-1630: LTLZinc Temporal Retention Benchmark

Exp 1630 SHALL generate a deterministic LTLZinc-style temporal-constraint
benchmark for FR-11 query-time learning non-forgetting checks.  The benchmark
SHALL exercise the existing additive `CaseMemory` retrieval path by inserting
temporal constraints, applying additional query-time update rows, and then
verifying that the original temporal constraints can still be retrieved and
locally checked.

### REQ-LEARN-1630 Sub-requirements

- REQ-LEARN-1630-1: The workflow SHALL write
  `results/experiment_1630_ltlzinc.json` with a terminal JSON artifact.
- REQ-LEARN-1630-2: The generated benchmark cases SHALL include
  LTLZinc-style finite Boolean traces, temporal formulas, MiniZinc-style
  constraint strings, expected satisfaction labels, and deterministic case IDs.
- REQ-LEARN-1630-3: Each benchmark case SHALL be converted into a
  `CaseRecord` shape compatible with `CaseMemory`, and the benchmark SHALL
  score a case as retained only when the local temporal verifier agrees with
  the expected label and retrieval after later updates returns the same case
  key.
- REQ-LEARN-1630-4: The terminal artifact SHALL include `status`, `schema`,
  `experiment_id`, `benchmark_size`, `pass_rate`, `retained_case_count`,
  `memory_entry_count`, `case_results`, and `honest_verdict`.
- REQ-LEARN-1630-5: A complete benchmark result SHALL require
  `benchmark_size > 0`, `pass_rate == 1.0`, and every retained case result to
  cite the temporal operator and retrieved CaseMemory fingerprint.
- REQ-LEARN-1630-6: The repository SHALL provide a deterministic generator for
  `data/ltlzinc_benchmark.json` that reads `data/constraint_templates.json`,
  emits a top-level schema/version/provenance summary, and writes stable JSON
  with `case_count`, `anchor_case_count`, `update_case_count`,
  `sat_case_count`, `repair_hint_case_count`, `supported_operators`, and
  `cases`.
- REQ-LEARN-1630-7: Each reusable benchmark case SHALL include a deterministic
  `case_id`, source template provenance, non-forgetting phase, temporal
  operator, LTL-style formula, MiniZinc-style constraint, finite Boolean trace,
  expected satisfaction label, certificate state, DVI label, verifier path, and
  retention metadata so evaluation code can distinguish original anchors from
  later updates.

### SCENARIO-LEARN-1630: Temporal Constraints Survive Query-Time Updates

**Given** finite-trace `always`, `eventually`, `next`, and `until` temporal
constraints have been recorded into `CaseMemory`
**When** additional temporal update cases are recorded and the original cases
are replayed through the retention benchmark
**Then** every original case is locally verified against its expected label
**And** retrieval returns the same temporal CaseMemory key for each original
case
**And** the terminal artifact reports `benchmark_size` and `pass_rate`.

### SCENARIO-LEARN-1630-JSON: Reusable Temporal Benchmark JSON Is Stable

**Given** `data/constraint_templates.json` contains deterministic source
template IDs
**When** `scripts/generate_ltlzinc.py` writes
`data/ltlzinc_benchmark.json`
**Then** the JSON contains balanced SAT and REPAIR_HINT temporal cases for
`always`, `eventually`, `next`, and `until`
**And** every case can be validated by the local LTLZinc finite-trace verifier
against its expected label
**And** anchor cases are explicitly marked for retention checks after update
cases are applied.

## Implementation Status (REQ-LEARN-1630)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1630 | Implemented (`scripts/experiment_1630_ltlzinc.py`) | Implemented (`tests/python/test_experiment_1630_ltlzinc.py`) |
| REQ-LEARN-1630-6/7 | Implemented (`scripts/generate_ltlzinc.py`) | Implemented (`tests/python/test_generate_ltlzinc.py`) |

---

## REQ-LEARN-1752: LTLZinc Spatial Reasoning Benchmark Expansion

Exp 1752 SHALL broaden the reusable LTLZinc continual-learning curriculum from
finite temporal traces into deterministic spatial reasoning cases.  The spatial
benchmark SHALL focus on topological map routing constraints so later FR-11 and
DVI checks can replay route-validity, obstacle-avoidance, waypoint, and
simple-path violations without calling an external MiniZinc runtime.

### REQ-LEARN-1752 Sub-requirements

- REQ-LEARN-1752-1: The workflow SHALL write
  `results/experiment_1752_spatial.json` with a terminal JSON artifact.
- REQ-LEARN-1752-2: The workflow SHALL create or refresh
  `data/ltlzinc_spatial_benchmark.json` with exactly 100 deterministic spatial
  reasoning cases derived from the existing `data/ltlzinc_benchmark.json`
  provenance.
- REQ-LEARN-1752-3: The spatial cases SHALL include topological map routing
  constraints with deterministic case IDs, map IDs, node/edge topology,
  blocked edges, route traces, LTLZinc-style formulas, MiniZinc-style
  constraint strings, expected satisfaction labels, certificate states, DVI
  labels, verifier metadata, retention metadata, and source benchmark
  provenance.
- REQ-LEARN-1752-4: The case families SHALL cover route existence, blocked
  edge avoidance, mandatory waypoint visits, forbidden-zone avoidance, and
  simple-path/no-revisit checks, with balanced SAT and REPAIR_HINT rows per
  family.
- REQ-LEARN-1752-5: The terminal artifact SHALL include `status`, `schema`,
  `experiment_id`, `benchmark_path`, `spatial_case_count`,
  `validated_case_count`, `sat_case_count`, `repair_hint_case_count`,
  `family_counts`, `commands_run`, and `honest_verdict`.
- REQ-LEARN-1752-6: A complete artifact SHALL require exactly 100 spatial
  cases, every local spatial verifier result to match the expected label, and
  balanced SAT and REPAIR_HINT totals.

### SCENARIO-LEARN-1752: Spatial Routes Broaden LTLZinc Replay

**Given** the existing temporal LTLZinc benchmark exists as provenance
**When** `scripts/experiment_1752_expand_ltlzinc.py` writes
`data/ltlzinc_spatial_benchmark.json` and
`results/experiment_1752_spatial.json`
**Then** the spatial benchmark contains 100 deterministic topological routing
cases
**And** every case is validated by the local spatial verifier
**And** every supported spatial family contains both SAT and REPAIR_HINT rows
**And** the terminal artifact reports a complete verdict only when those
conditions hold.

## Implementation Status (REQ-LEARN-1752)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1752 | Implemented (`scripts/experiment_1752_expand_ltlzinc.py`) | Implemented (`tests/python/test_experiment_1752_expand_ltlzinc.py`) |

---

## REQ-LEARN-1753: Continual Stability Evaluation Loop

Exp 1753 SHALL run a continual stability evaluation loop over the expanded LTLZinc spatial dataset using `unsloth/gemma-4-31B-it-GGUF`.

### REQ-LEARN-1753 Sub-requirements

- REQ-LEARN-1753-1: The workflow SHALL create `scripts/experiment_1753_continual_stability.py`.
- REQ-LEARN-1753-2: The workflow SHALL run the evaluation over the expanded dataset.
- REQ-LEARN-1753-3: The workflow SHALL write the results to `results/experiment_1753_continual.json`.

### SCENARIO-LEARN-1753: Continual Stability Evaluation
**Given** the expanded LTLZinc spatial dataset
**When** `scripts/experiment_1753_continual_stability.py` runs
**Then** it evaluates stability using `unsloth/gemma-4-31B-it-GGUF`
**And** it outputs a complete artifact to `results/experiment_1753_continual.json`.

## Implementation Status (REQ-LEARN-1753)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1753 | Implemented (`scripts/experiment_1753_continual_stability.py`) | Implemented (`tests/python/test_experiment_1753_continual_stability.py`) |

---

## REQ-LEARN-1539: FR-11 External-Feedback Skill Graph Promotion

Exp 1539 SHALL convert rollback-passing FR-11 query-time policy updates into an
auditable skill graph whose promoted nodes cite external deterministic verifier
feedback, replay evidence, and explicit skill lineage.  The workflow SHALL NOT
train, finetune, write checkpoints, or mutate model weights.  Positive headline
self-learning readiness SHALL require strictly positive task utility and zero
soundness mistakes; a safe zero-utility result SHALL remain complete but SHALL
NOT claim positive-utility promotion readiness.

### REQ-LEARN-1539 Sub-requirements

- REQ-LEARN-1539-1: The workflow SHALL write
  `results/experiment_1539_fr11_external_feedback_skill_promotion_v13.json`
  first with `status="in_progress"` before loading Exp 1524 live-policy
  promotion sources, Exp 1513 rollback replay sources, or Exp 1538
  residual-drift ledger sources.
- REQ-LEARN-1539-2: Candidate updates SHALL be extracted with explicit
  query-time policy inputs, expected verifier outputs, model outputs or output
  hashes, verifier rewards, rollback evidence, replay evidence, and skill
  lineage.
- REQ-LEARN-1539-3: Skill graph nodes SHALL be promoted only when the update
  has external deterministic verifier feedback, rollback replay evidence,
  no positive false-accept delta, no soundness mistakes, and no model-weight
  mutation.  Self-feedback alone SHALL NOT satisfy this requirement.
- REQ-LEARN-1539-4: The skill graph artifact SHALL be written under `results/`
  or `ops/`, and each promoted node SHALL cite the source artifacts and row
  identifiers that support the node.
- REQ-LEARN-1539-5: The rollback plan SHALL identify each promoted node,
  the policy update it can disable, and deterministic triggers for demotion
  or rollback, including soundness mistakes, false-accept increases, missing
  verifier feedback, stale provenance, or replay failure.
- REQ-LEARN-1539-6: Baseline and promoted task success rates SHALL be computed
  on a bounded mandated SOTA headline model task set from Exp 1524 live
  promotion evidence.  Legacy small GGUFs MAY NOT become headline-result
  models for this artifact.
- REQ-LEARN-1539-7: `positive_utility_promotion_ready` SHALL be true only
  when `utility_delta > 0`, `soundness_mistakes == 0`, at least one promoted
  skill-graph node has external verifier feedback, live mandated SOTA
  inference evidence is present, and `no_model_weight_mutation=true`.
- REQ-LEARN-1539-8: The terminal artifact SHALL include `status`,
  `milestone`, `continuous_self_learning_task`,
  `fr11_external_feedback_ready`, `positive_utility_promotion_ready`,
  `model_specs`, `live_sota_model_inference_used`, `skill_graph_path`,
  `candidate_updates`, `externally_verified_updates`, `promoted_updates`,
  `baseline_task_success_rate`, `promoted_task_success_rate`,
  `utility_delta`, `soundness_mistakes`, `no_model_weight_mutation`,
  `rollback_plan_path`, `focused_tests_passed`, and `honest_verdict`.

### SCENARIO-LEARN-1539: External Verifier Feedback Promotes An Auditable Node

**Given** an Exp 1524 live-policy evaluation row for a mandated SOTA GGUF
**And** the row records deterministic runtime-contract verifier feedback
with zero false accepts and zero soundness mistakes
**And** rollback replay evidence kept the same policy update
**When** Exp 1539 builds the FR-11 skill graph
**Then** the candidate update becomes a promoted skill-graph node
**And** the node records its policy input interface, expected verifier output,
verifier reward, replay evidence, source artifacts, and rollback plan handle.

### SCENARIO-LEARN-1540: Positive Utility Is Required For Headline Readiness

**Given** a promoted external-feedback skill node with zero soundness mistakes
**And** live SOTA evidence where promoted task success does not exceed baseline
task success
**When** Exp 1539 writes the terminal artifact
**Then** `fr11_external_feedback_ready=true`
**And** `positive_utility_promotion_ready=false`
**And** `honest_verdict` reports the safety-only result without claiming
positive utility.

### SCENARIO-LEARN-1541: Self-Feedback Or Unsafe Replay Rolls Back

**Given** a candidate update that lacks external deterministic verifier
feedback, increases false accepts, records a soundness mistake, or lacks
rollback replay evidence
**When** Exp 1539 evaluates promotion eligibility
**Then** the update is not promoted
**And** the rollback plan records the deterministic demotion or rollback
trigger.

## Implementation Status (REQ-LEARN-1539)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1539 | Implemented (`python/carnot/reporting/fr11_external_feedback_skill_promotion.py`) | Implemented (`tests/python/test_fr11_external_feedback_skill_promotion.py`) |

---

## REQ-LEARN-1555: FR-11 Positive Utility Or Retire Gate

Exp 1555 SHALL run the .119 continuous self-learning gate over the .118
external-feedback skill graph plus any available .119 repair or scale artifacts.
The gate SHALL promote only externally verified skill or policy updates, SHALL
reject self-feedback-only updates, SHALL NOT mutate model weights, and SHALL
compute held-out verifier-backed baseline and post-promotion utility.  If the
measured `utility_delta` is not strictly positive, the artifact SHALL retire the
positive-utility self-learning headline instead of reporting promotion success.

### REQ-LEARN-1555 Sub-requirements

- REQ-LEARN-1555-1: The workflow SHALL write
  `results/experiment_1555_fr11_positive_utility_or_retire_v14.json` first with
  `status="in_progress"` before loading predecessor evidence.
- REQ-LEARN-1555-2: Candidate updates SHALL be selected only from externally
  verified successes or deterministic failure repairs.  Candidates whose only
  evidence is model self-feedback SHALL be rejected and SHALL NOT appear in
  `skill_updates_promoted`.
- REQ-LEARN-1555-3: Every promoted update SHALL pass pre-promotion replay and
  post-promotion replay checks, SHALL have zero false accepts, SHALL have zero
  soundness mistakes, and SHALL declare `no_model_weight_mutation=true`.
- REQ-LEARN-1555-4: `baseline_utility`, `post_promotion_utility`, and
  `utility_delta` SHALL be computed on a held-out verifier-backed replay set
  derived from checked-in experiment artifacts.
- REQ-LEARN-1555-5: `positive_utility_achieved` SHALL be true only when
  `utility_delta > 0`, `soundness_mistakes == 0`, at least one update is
  promoted, external feedback is used, and model weights are not mutated.
- REQ-LEARN-1555-6: If `utility_delta <= 0`, the artifact SHALL set
  `positive_utility_claim_retired=true`.
- REQ-LEARN-1555-7: The terminal artifact SHALL include `status`, `milestone`,
  `continuous_self_learning_task`, `fr11_positive_utility_gate_ready`,
  `model_specs`, `live_sota_model_inference_used`, `no_model_weight_mutation`,
  `external_feedback_used`, `self_feedback_only_rejected`,
  `candidate_skill_updates`, `skill_updates_promoted`, `replay_cases`,
  `replay_pass_rate`, `soundness_mistakes`, `baseline_utility`,
  `post_promotion_utility`, `utility_delta`, `positive_utility_achieved`,
  `positive_utility_claim_retired`, `skill_graph_path`,
  `focused_tests_passed`, and `honest_verdict`.

### SCENARIO-LEARN-1555: Externally Verified Repair Promotion Demonstrates Utility

**Given** Exp 1552 residual-drift repair rows with accepted deterministic replay
repairs and zero false accepts
**When** Exp 1555 builds candidate updates from available .119 repair evidence
**Then** the repair policy candidate is eligible for promotion
**And** the held-out replay set records a post-promotion utility greater than
baseline utility when accepted repairs pass deterministic replay.

### SCENARIO-LEARN-1556: Zero Or Negative Utility Retires The Headline

**Given** externally verified candidates whose post-promotion utility does not
exceed baseline utility
**When** Exp 1555 writes the terminal artifact
**Then** `positive_utility_achieved=false`
**And** `positive_utility_claim_retired=true`
**And** `honest_verdict` uses a terminal prefix while reporting the retirement.

### SCENARIO-LEARN-1557: Self-Feedback Or Unsafe Replay Is Rejected

**Given** a candidate with self-feedback-only evidence, model-weight mutation,
missing replay, replay failure, false accepts, or soundness mistakes
**When** Exp 1555 evaluates promotion eligibility
**Then** the candidate is not promoted
**And** its deterministic rejection reason is recorded for audit.

## Implementation Status (REQ-LEARN-1555)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1555 | Implemented (`python/carnot/reporting/fr11_positive_utility_or_retire.py`) | Implemented (`tests/python/test_fr11_positive_utility_or_retire.py`) |

---

## REQ-LEARN-1568: FR-11 v14 Retained Policy Mode-Collapse Audit

Exp 1568 SHALL audit the retained v14 policies that passed the Exp 1555
Positive-Utility-or-Retire gate for anti-exploration mode-collapse predictors.
The audit SHALL snapshot the retained policy set from the Exp 1555 terminal
artifact and v14 skill graph, measure every available retained policy against
checked-in held-out or replay evidence, and SHALL NOT fabricate missing policy
count, pre-RL baseline, fresh k=8 reward groups, or OOD adversarial accuracy
evidence when the source artifacts do not contain it.

### REQ-LEARN-1568 Sub-requirements

- REQ-LEARN-1568-1: The workflow SHALL write
  `results/experiment_1568_fr11_v14_retained_mode_collapse_audit.json` with
  `status="complete"` and an `honest_verdict` prefixed by `complete:`.
- REQ-LEARN-1568-2: The retained policy snapshot SHALL include exactly the
  v14 policies with `promotion_decision="promote"` or equivalent retained-node
  status from Exp 1555; if fewer than the target N=5 exist, the artifact SHALL
  report the actual count and a source limitation.
- REQ-LEARN-1568-3: For each retained policy, the audit SHALL report the four
  requested predictors: token-entropy drop versus pre-RL baseline, boilerplate
  fraction versus a training/reference corpus, k=8 reward-variance collapse,
  and adversarial OOD accuracy regression.  Predictors whose source evidence is
  unavailable SHALL be marked unavailable rather than confirmed.
- REQ-LEARN-1568-4: A predictor SHALL be confirmed only under the requested
  gates: token-entropy drop >= 0.5 nats per token, boilerplate fraction >= 0.30,
  reward variance collapsed to a single mode, or OOD adversarial accuracy worse
  than the pre-RL baseline.
- REQ-LEARN-1568-5: `mode_collapse_confirmed_count` SHALL count retained
  policies with at least two confirmed predictors, and
  `mode_collapse_confirmed_percent` SHALL equal that count divided by
  `retained_policies_audited_count`.
- REQ-LEARN-1568-6: `reversal_recommended_count` SHALL equal the confirmed
  retained-policy count only when at least 50% of audited retained policies show
  at least two confirmed predictors; otherwise it SHALL be zero.
- REQ-LEARN-1568-7: The terminal artifact SHALL include
  `mode_collapse_audit_complete`, `retained_policies_audited_count`,
  `mode_collapse_confirmed_count`, `reversal_recommended_count`,
  `retained_policy_audits`, `source_limitations`, `spec`, and
  `honest_verdict`.

### SCENARIO-LEARN-1568: Multiple Predictors Flag Retained Policies

**Given** retained v14 policies with held-out generated repairs, reference
training text, k=8 reward groups, and OOD adversarial baseline/post accuracy
evidence
**When** Exp 1568 evaluates the anti-mode-collapse gates
**Then** any retained policy with at least two confirmed predictors contributes
to `mode_collapse_confirmed_count`
**And** reversal recommendations are emitted only if the confirmed share is at
least 50% of audited retained policies.

### SCENARIO-LEARN-1569: Missing Audit Evidence Remains Honest

**Given** the Exp 1555 retained-policy artifacts contain fewer than five retained
policies or omit pre-RL, fresh k=8, or OOD adversarial evidence
**When** Exp 1568 writes the terminal artifact
**Then** it reports the actual audited count and source limitations
**And** unavailable predictors are not counted as confirmed.

## Implementation Status (REQ-LEARN-1568)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1568 | Implemented (`python/carnot/reporting/fr11_v14_retained_mode_collapse_audit.py`) | Implemented (`tests/python/test_fr11_v14_retained_mode_collapse_audit.py`) |

---

## REQ-LEARN-1581: FR-11 v15 Lambda-GRPO Retention Reversal

Exp 1581 SHALL execute the .121 continuous self-learning correction path for
the v14 retention that Exp 1568 flagged as mode-collapsed.  The workflow SHALL
replay `policy:residual_drift_repair:1552` against held-out checked-in evidence,
apply the lambda-GRPO loss-normalization correction when a local trainer path is
available, otherwise write a deterministic simulator showing the corrected
non-collapsed weighting, and SHALL reverse the retained v14 policy only if the
fresh replay still confirms the Exp 1568 mode-collapse criteria.

### REQ-LEARN-1581 Sub-requirements

- REQ-LEARN-1581-1: The workflow SHALL write
  `results/experiment_1581_fr11_v15_lambda_grpo_retention_reversal.json` with
  `status="in_progress"` before replaying predecessor artifacts.
- REQ-LEARN-1581-2: The replay SHALL target
  `policy:residual_drift_repair:1552` only when Exp 1568 recommended that
  policy for retention reversal, and SHALL report whether the flagged policy
  was replayed.
- REQ-LEARN-1581-3: The replay SHALL measure the Exp 1568 predictors on
  held-out accepted repair rows, including entropy preservation, boilerplate
  fraction delta, OOD accuracy proxy, reward-variance collapse, and soundness
  mistakes.
- REQ-LEARN-1581-4: The lambda-GRPO correction path SHALL normalize per-case
  weights so uniform rewards do not collapse all update mass onto boilerplate
  repairs.  If the live trainer dependency is unavailable, the workflow SHALL
  mark `implementation_deferred=true`, set `lambda_grpo_simulated_only=true`,
  and include deterministic simulator evidence for the corrected weighting.
- REQ-LEARN-1581-5: `retention_reversal_applied` SHALL be true only when the
  flagged policy was replayed, soundness mistakes are zero, at least two
  replay predictors still confirm mode collapse, no model weights were mutated,
  and the lambda-GRPO correction path has either been implemented locally or
  deterministically simulated.
- REQ-LEARN-1581-6: The terminal artifact SHALL include `status`,
  `continuous_self_learning_task`, `flagged_policy_replayed`,
  `retention_reversal_applied`, `lambda_grpo_patch_implemented`,
  `lambda_grpo_simulated_only`, `soundness_mistakes`,
  `entropy_preservation_rate`, `boilerplate_fraction_delta`,
  `fr11_v15_decision_ready`, and `honest_verdict`.

### SCENARIO-LEARN-1581: Replay-Confirmed Collapse Reverses Retention

**Given** Exp 1568 recommended `policy:residual_drift_repair:1552` for reversal
**And** held-out repair replay still shows at least two mode-collapse predictors
with zero soundness mistakes
**When** Exp 1581 writes the terminal artifact
**Then** `flagged_policy_replayed=true`
**And** `retention_reversal_applied=true`
**And** the artifact records lambda-GRPO correction evidence without mutating
model weights.

### SCENARIO-LEARN-1582: Unconfirmed Collapse Does Not Reverse Retention

**Given** a flagged retained policy whose fresh replay has fewer than two
confirmed mode-collapse predictors or nonzero soundness mistakes
**When** Exp 1581 evaluates the retention reversal gate
**Then** `retention_reversal_applied=false`
**And** `honest_verdict` reports that reversal was blocked by the replay.

## Implementation Status (REQ-LEARN-1581)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1581 | Implemented (`python/carnot/reporting/fr11_v15_lambda_grpo_retention_reversal.py`) | Implemented (`tests/python/test_fr11_v15_lambda_grpo_retention_reversal.py`) |

## REQ-LEARN-1624: Offline Consolidation via Spectral Overlap Pruning

**Given** a set of active energy constraints
**When** the offline consolidation phase runs
**Then** redundant constraints MUST be removed using spectral overlap pruning
**And** the pruned network MUST yield identical energy minima on a hold-out set.

### REQ-LEARN-1624 Sub-requirements
- REQ-LEARN-1624-1: Pruning SHALL calculate spectral overlap between constraint embeddings or matrices.
- REQ-LEARN-1624-2: The test MUST validate energy minima on a hold-out set.
- REQ-LEARN-1624-3: Artifact MUST be saved to results/experiment_1624_adaptive_reconfig.json.

---

## REQ-LEARN-1635: Label-Free ConsFormer Refiner on FoVer CSP Rows

Exp 1635 SHALL train a deterministic, CPU-only ConsFormer-style refiner on
FoVer constraint-satisfaction rows without using labels for the training
objective.  The refiner SHALL turn each FoVer step into local CSP features,
apply a lightweight transformer-style self-attention refinement pass, and
evaluate the refined satisfaction score against held-out FoVer labels only for
reporting.

### REQ-LEARN-1635 Sub-requirements

- REQ-LEARN-1635-1: The workflow SHALL implement
  `scripts/experiment_1635_consformer.py` and write
  `results/experiment_1635_consformer.json`.
- REQ-LEARN-1635-2: FoVer loading SHALL accept JSONL rows with `step_text` and
  `label`, skip malformed or unsupported rows, and preserve deterministic row
  ordering.
- REQ-LEARN-1635-3: Training SHALL fit the refiner's scoring statistics from
  unlabeled CSP features only; held-out FoVer labels SHALL be used only by the
  evaluation routine that computes accuracy.
- REQ-LEARN-1635-4: The terminal artifact SHALL include `status`, `schema`,
  `experiment_id`, `spec_refs`, `dataset_rows`, `train_rows`, `eval_rows`,
  `refiner_accuracy`, `baseline_accuracy`, `label_free_training`,
  `tests_run`, and `honest_verdict`.
- REQ-LEARN-1635-5: A complete artifact SHALL require at least one train row,
  at least one eval row, `label_free_training=true`, and
  `0.0 <= refiner_accuracy <= 1.0`.

### SCENARIO-LEARN-1635: ConsFormer Refiner Writes FoVer Accuracy

**Given** a FoVer JSONL corpus containing correct and incorrect step rows
**When** Exp 1635 trains the label-free ConsFormer-style refiner and evaluates
the held-out split
**Then** it writes `results/experiment_1635_consformer.json`
**And** the artifact records `refiner_accuracy` and `baseline_accuracy`
**And** `refiner_accuracy` is computed from labels that were not used by the
training objective.

## Implementation Status (REQ-LEARN-1635)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1635 | Implemented (`scripts/experiment_1635_consformer.py`) | Implemented (`tests/python/test_experiment_1635_consformer.py`) |

---

## REQ-LEARN-1659: SMGI Certified Update Gates For FR-11 CerCE Ledger

Exp 1659 SHALL implement reusable SMGI certified-update gates for FR-11 CerCE
ledger policy updates in `python/carnot/pipeline/smgi_updates.py`.  A certified
update SHALL be query-time bookkeeping only: it MAY update portable memory and
policy metadata after certification, but it SHALL NOT train, finetune, write
checkpoints, or mutate model weights.  The terminal certification report SHALL
be written to `results/experiment_1659_smgi_certified_updates.json` when the
module is run for the current roadmap gate.

### REQ-LEARN-1659 Sub-requirements

- REQ-LEARN-1659-1: The module SHALL emit a deterministic JSON-compatible
  certification report with `status`, `schema`, `experiment_id`,
  `continuous_self_learning_task`, `smgi_certified_update_ready`,
  `certified_update_success`, `cerce_ledger_ready`,
  `policy_certificates_evaluated`, `accepted_violation_count`,
  `false_accept_delta`, `soundness_mistakes`,
  `nonforgetting_certificate_rate`, `promotion_safe_policy_updates`,
  `blocked_policy_updates`, `certified_updates`, `rejected_updates`,
  `blockers`, and `honest_verdict`.
- REQ-LEARN-1659-2: The CerCE ledger gate SHALL pass only when the source
  ledger is complete, `cerce_ledger_ready=true`, at least one policy
  certificate exists, every certificate is promotion-safe, zero accepted
  violations and zero soundness mistakes are present, aggregate
  `false_accept_delta <= 0`, and `nonforgetting_certificate_rate == 1.0`.
- REQ-LEARN-1659-3: The SMGI update gate SHALL certify a candidate only when
  its policy id is promotion-safe in the CerCE ledger, its certificate id
  matches the ledger certificate, it cites prior and updated SessionMemory
  state hashes, it has retained every replayed case, it has zero replay
  failures, it has non-negative utility delta, and it declares
  `no_model_weight_mutation=true`.
- REQ-LEARN-1659-4: If no explicit candidates are provided, the module SHALL
  derive deterministic certification candidates from promotion-safe CerCE
  certificates so downstream FR-11 experiments can gate on the existing ledger
  without bypassing the same checks.
- REQ-LEARN-1659-5: Validation SHALL fail closed: missing required report
  fields, impossible complete reports, unsafe ledgers, replay failures,
  negative utility, missing hashes, certificate mismatches, or model-weight
  mutation SHALL produce `status="blocked"` and SHALL NOT set
  `certified_update_success=true`.

### SCENARIO-LEARN-1659: Promotion-Safe CerCE Row Becomes Certified Update

**Given** an FR-11 CerCE ledger whose policy certificate is promotion-safe
**And** an SMGI update candidate that cites matching certificate and memory
hashes with zero replay failures and no model-weight mutation
**When** Exp 1659 evaluates the certified-update gates
**Then** the report is complete
**And** `smgi_certified_update_ready=true`
**And** `certified_update_success=true`
**And** the policy id appears in `certified_updates`.

### SCENARIO-LEARN-1660: Unsafe CerCE Or Replay Evidence Blocks Certification

**Given** a CerCE ledger or SMGI update candidate with accepted violations,
positive false-accept delta, soundness mistakes, replay failures, missing
memory hashes, negative utility, certificate mismatch, or model-weight mutation
**When** Exp 1659 evaluates the certified-update gates
**Then** the report is blocked
**And** `certified_update_success=false`
**And** the unsafe policy id appears in `rejected_updates`.

## Implementation Status (REQ-LEARN-1659)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1659 | Implemented (`python/carnot/pipeline/smgi_updates.py`) | Implemented (`tests/python/test_smgi_updates.py`) |

---

## REQ-LEARN-1664: E2E Plan Coverage For EBRM Scorers And SMGI Updates

Exp 1664 SHALL update `ops/e2e-test-plan.md` with deterministic end-to-end
test scenarios for the new EBRM scorer/hardware path and the SMGI certified
update path.  The updater SHALL be idempotent, SHALL preserve the existing E2E
plan content, and SHALL write `results/experiment_1664_e2e_plan.json`.

### REQ-LEARN-1664 Sub-requirements

- REQ-LEARN-1664-1: The updater SHALL add an EBRM scorer E2E scenario that
  references Exp 1656 CPU trace scoring, Exp 1657 KV260 q=3 Potts binding, and
  Exp 1658 CPU-vs-KV260 SOTA trace evaluation.
- REQ-LEARN-1664-2: The updater SHALL add an SMGI certified-update E2E
  scenario that references Exp 1659 gates, CerCE certificate evidence, replay
  retention, SessionMemory hash changes, and `no_model_weight_mutation=true`.
- REQ-LEARN-1664-3: The updater SHALL validate source artifacts before marking
  the Exp 1664 artifact complete; missing or incomplete EBRM/SMGI evidence SHALL
  produce `status="blocked"` with explicit blockers.
- REQ-LEARN-1664-4: Re-running the updater SHALL NOT duplicate E2E sections or
  rewrite the plan when the required scenarios are already present.
- REQ-LEARN-1664-5: The terminal artifact SHALL include `status`, `schema`,
  `experiment_id`, `run_date`, `plan_path`, `e2e_sections_added`,
  `e2e_section_ids`, `ebrm_e2e_ready`, `smgi_e2e_ready`, `source_artifacts`,
  `plan_hash_before`, `plan_hash_after`, `plan_updated`, `spec_traces`,
  `tests_run`, `blockers`, and `honest_verdict`.

### SCENARIO-LEARN-1664: EBRM And SMGI E2E Plan Entries Are Stable

**Given** the current E2E plan and complete Exp 1656, 1657, 1658, and 1659
artifacts
**When** Exp 1664 updates the plan
**Then** the plan contains one EBRM scorer/hardware scenario and one SMGI
certified-update verification scenario
**And** the terminal artifact is complete with source evidence and spec traces.

### SCENARIO-LEARN-1665: Missing Source Evidence Blocks Completion

**Given** missing or incomplete Exp 1656, 1657, 1658, or 1659 artifacts
**When** Exp 1664 evaluates source evidence
**Then** it records explicit blockers
**And** it does not mark `ebrm_e2e_ready` or `smgi_e2e_ready` true without the
required evidence.

## Implementation Status (REQ-LEARN-1664)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1664 | Implemented (`scripts/experiment_1664_e2e.py`) | Implemented (`tests/python/test_experiment_1664_e2e.py`) |

---

## REQ-LEARN-1668: Pipeline CerCE Ledger For FR-11 Promotion Bounds

Exp 1668 SHALL provide a reusable CerCE-style promotion ledger in
`python/carnot/pipeline/cerce_ledger.py` so FR-11 policy and memory updates can
be certified before promotion.  The ledger SHALL compare pre-update and
post-update constraint-violation bounds, verify replay retention, and reject any
candidate whose post-update bound is worse than its pre-update bound.  The
mechanism is query-time bookkeeping only: it SHALL NOT train, finetune, write
checkpoints, or mutate model weights.  The terminal artifact SHALL be written to
`results/experiment_1668_cerce.json` and SHALL include `nonforgetting_rate`.

### REQ-LEARN-1668 Sub-requirements

- REQ-LEARN-1668-1: The module SHALL expose deterministic JSON-compatible
  policy certificates for simulated or real memory updates, including
  `policy_update_id`, `certificate_id`, `pre_violation_bound`,
  `post_violation_bound`, `violation_bound_delta`,
  `replay_retention_rate`, `replay_retention_passed`, and
  `promotion_safe`.
- REQ-LEARN-1668-2: The pre/post bound gate SHALL mark a candidate unsafe when
  any replay case has `post_violation_bound > pre_violation_bound` or when the
  aggregate post-update violation bound is greater than the aggregate
  pre-update violation bound.
- REQ-LEARN-1668-3: The replay-retention gate SHALL pass only when at least one
  replay case is present, every replayed case is retained, and no replay case
  reports a replay failure.
- REQ-LEARN-1668-4: The terminal artifact SHALL report
  `policy_certificates_evaluated`, `promotion_safe_policy_updates`,
  `blocked_policy_updates`, `accepted_violation_count`,
  `pre_violation_bound`, `post_violation_bound`, `violation_bound_delta`,
  `replay_retention_rate`, `nonforgetting_rate`, `certificates`, `blockers`,
  and `honest_verdict`.
- REQ-LEARN-1668-5: The promotion gate SHALL fail closed: worsened bounds,
  missing replay cases, replay failures, dropped replay cases, unchanged memory
  hashes, or model-weight mutation SHALL prevent `cerce_ledger_ready=true`.

### SCENARIO-LEARN-1668: Simulated Memory Update Preserves Bounds

**Given** an FR-11 simulated memory update with retained replay cases
**And** each post-update constraint-violation bound is less than or equal to the
matching pre-update bound
**When** the pipeline CerCE promotion ledger evaluates the update
**Then** the policy certificate is promotion-safe
**And** `cerce_ledger_ready=true`
**And** `nonforgetting_rate=1.0`.

### SCENARIO-LEARN-1669: Worsened Bound Blocks Policy Promotion

**Given** an FR-11 simulated memory update with a replay case whose
post-update constraint-violation bound is greater than its pre-update bound
**When** the pipeline CerCE promotion ledger evaluates the update
**Then** the update is rejected
**And** the policy id appears in `blocked_policy_updates`
**And** `cerce_ledger_ready=false`.

## Implementation Status (REQ-LEARN-1668)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1668 | Implemented (`python/carnot/pipeline/cerce_ledger.py`) | Implemented (`tests/python/test_cerce_ledger.py`) |

---

## REQ-LEARN-1669: LTLZinc CerCE Continual-Learning Adapter

Exp 1669 SHALL provide a deterministic LTLZinc-style continual-learning adapter
in `python/carnot/pipeline/ltlzinc_adapter.py`.  The adapter SHALL generate
finite temporal-constraint replay cases with LTL-style formulas and
MiniZinc-style constraint strings, evaluate the replay cases through the
pipeline CerCE ledger, and write
`results/experiment_1669_ltlzinc.json`.  The run SHALL be query-time
bookkeeping only: it SHALL NOT train, finetune, write checkpoints, or mutate
model weights.

### REQ-LEARN-1669 Sub-requirements

- REQ-LEARN-1669-1: The adapter SHALL generate deterministic cases for the
  temporal operators `always`, `eventually`, `next`, and `until`; each case
  SHALL include `case_id`, `temporal_operator`, finite Boolean `trace`,
  `expected_satisfied`, `ltl_formula`, and `minizinc_constraint`.
- REQ-LEARN-1669-2: The local verifier SHALL evaluate generated finite traces
  against the supported temporal operators and reject malformed or mislabeled
  rows before CerCE replay evidence is built.
- REQ-LEARN-1669-3: The adapter SHALL convert temporal cases into CerCE
  `ReplayCase` evidence, preserving one replay row per temporal case and using
  non-worsening pre/post violation bounds for retained cases.
- REQ-LEARN-1669-4: The benchmark SHALL call the CerCE promotion gate and
  report `forgetting_rate`, `cerce_nonforgetting_rate`,
  `replay_retention_rate`, `accepted_violation_count`,
  `policy_certificates_evaluated`, and per-case replay outcomes.
- REQ-LEARN-1669-5: The terminal artifact SHALL include `status`, `schema`,
  `experiment_id`, `ltlzinc_adapter_ready`, `temporal_cases_generated`,
  `cerce_ledger_ready`, `promotion_gate_passed`, `forgetting_rate`,
  `cerce_nonforgetting_rate`, `ledger_artifact`, `case_results`, `blockers`,
  `tests_run`, and `honest_verdict`.

### SCENARIO-LEARN-1669a: LTLZinc Temporal Cases Do Not Forget Under CerCE

**Given** deterministic temporal replay cases for `always`, `eventually`,
`next`, and `until`
**When** the LTLZinc adapter verifies the cases and evaluates their replay
evidence through the CerCE ledger
**Then** the terminal artifact reports `ltlzinc_adapter_ready=true`
**And** `forgetting_rate=0.0`
**And** `cerce_nonforgetting_rate=1.0`.

## Implementation Status (REQ-LEARN-1669)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-1669 | Implemented (`python/carnot/pipeline/ltlzinc_adapter.py`) | Implemented (`tests/python/test_ltlzinc_adapter.py`) |

## REQ-SELFPLAY-1685: Live SOTA GGUF Evaluation

The autonomous constraint discovery loop MUST be validated against mandated SOTA GGUFs.
The experiment SHALL generate 10 math reasoning traces, identify a hallucination,
auto-generate a constraint, and re-evaluate to confirm repair.
The deliverable SHALL be written to `results/experiment_1685_live_sota.json`.

### SCENARIO-SELFPLAY-1685: End-to-End Validation on SOTA Model

**Given** a pipeline configured to use `unsloth/Qwen3.6-35B-A3B-GGUF`
**When** the discovery loop is executed
**Then** it generates 10 traces
**And** successfully identifies at least one hallucination and confirms repair.

---

## REQ-FR11-040: FR-11 Continual Learning Replay Buffer with Structural Diversity

**Given** the FR-11 constraint self-discovery loop generating constraint violations
**When** the continual learning replay buffer is updated and sampled
**Then** it MUST prioritize structural constraint diversity (e.g. maintaining a diverse set of constraint types) to prevent catastrophic forgetting of rare constraint types.

### REQ-FR11-040 Sub-requirements
- REQ-FR11-040-1: The replay buffer MUST categorize samples by their structural constraint type.
- REQ-FR11-040-2: When adding samples, the buffer MUST ensure representation of diverse structural constraint types up to a max capacity per type.
- REQ-FR11-040-3: Sampling MUST return a batch that preserves structural diversity by sampling evenly across available constraint types.

### SCENARIO-FR11-040: Structural Diversity Preserved
**Given** a replay buffer flooded with constraint type A
**When** a constraint type B is added and the buffer is sampled
**Then** constraint type B is sampled at a proportionally higher rate than its naive frequency, preserving structural diversity.

---

## REQ-FR11-041: Semantic Pruning for FR-11 Replay Buffer

**Given** the FR-11 continual learning replay buffer
**When** constraint rules are added that are semantically redundant to existing rules
**Then** a semantic pruning mechanism MUST identify and remove redundant structural constraint rules to reduce semantic redundancy.

### REQ-FR11-041 Sub-requirements
- REQ-FR11-041-1: The pruning mechanism MUST compute a similarity score between structural constraint rules.
- REQ-FR11-041-2: Redundant rules exceeding a similarity threshold MUST be pruned from the replay buffer.

### SCENARIO-FR11-041: Semantic Pruning of Redundant Rules
**Given** an FR-11 replay buffer containing a structural constraint rule X
**When** a semantically equivalent structural constraint rule Y is added
**Then** the pruning mechanism calculates a similarity score above the threshold and removes rule Y (or X) to prevent redundancy.

---

## REQ-LEARN-101: Continuous Online Updater for CIKAN Models

**Given** a deployed CIKAN verifier
**When** the verification loop encounters streaming violations or valid responses
**Then** an online updater mechanism (e.g. SGD/AdamW) MUST adjust the CIKAN residual head weights to minimize binary cross-entropy loss against the verified outcome.

### REQ-LEARN-101 Sub-requirements
- REQ-LEARN-101-1: The updater MUST support stochastic updates based on single streaming observations.
- REQ-LEARN-101-2: The verification loop MUST route features and verified labels (violation vs. valid) to the updater.

### SCENARIO-LEARN-101: Online CIKAN Verification Loop
**Given** an instantiated `OnlineUpdater` and `VerificationLoop`
**When** a synthetic stream of alternating valid and violation samples is processed
**Then** the CIKAN residual energy for valid samples MUST decrease and the residual energy for violation samples MUST increase.

## REQ-LEARN-1725: E2E Pipeline with SOTA, FourierCSP, CIKAN, and Online Updater

**Given** the need to run an end-to-end continuous learning pipeline
**When** Experiment 1725 is executed
**Then** it MUST instantiate a SOTA model (`unsloth/Qwen3.6-35B-A3B-GGUF` or `unsloth/gemma-4-31B-it-GGUF`)
**And** it MUST execute a 50-problem stream where constraints are generated, parsed via FourierCSP, verified via CIKAN, and updated via Online Updater
**And** the experiment artifact MUST contain adaptation rate, model_used, n_processed, and honest_verdict
**And** honest_verdict MUST be `e2e_pipeline_successful` or `e2e_pipeline_failed`.

### SCENARIO-LEARN-1725: Adaptation Rate Measured on 50-Problem Stream

**Given** a SOTA model stream of 50 problems
**When** the pipeline processes the stream
**Then** the Online Updater adapts to the verified violations
**And** the `adaptation_rate` (updated/processed) is computed and written to `results/experiment_1725_e2e_cikan.json`.

---

## REQ-LEARN-102: Asynchronous Telemetry Streamer

**Given** a continuous self-learning pipeline
**When** active inference sessions generate violation feedback
**Then** an asynchronous telemetry streamer MUST queue and record `VerificationResult` instances (both successes and failures) without blocking the main inference loop.

### REQ-LEARN-102 Sub-requirements
- REQ-LEARN-102-1: `TelemetryStreamer` MUST implement a non-blocking queue.
- REQ-LEARN-102-2: It MUST record `VerificationResult` objects.
- REQ-LEARN-102-3: A load test script MUST verify the asynchronous operation and report `experiment_1738_telemetry.json` with appropriate telemetry load metrics.

### SCENARIO-LEARN-102: Async Streamer Load Test
**Given** an instantiated `TelemetryStreamer`
**When** multiple simulated inference threads submit results
**Then** the streamer processes them asynchronously without dropping items and outputs the results JSON.

## REQ-LEARN-1754: Continual Memory Semantic Distillation

**Given** a ContinualMemory containing memory states
**When** distill(n_clusters) is called
**Then** memory states MUST be grouped by semantic clustering
**And** only representative vectors for each cluster MUST be retained

### SCENARIO-LEARN-1754: Memory states are pruned to requested clusters
**Given** 100 memory states representing 5 semantic groups
**When** distill(n_clusters=5) is called
**Then** exactly 5 states remain

## REQ-LEARN-1820: Continuous Online Distillation for MoE Routers

**Given** the need for continuous self-learning via online distillation (arXiv:2604.08912)
**When** the MoE distillation loop runs during inference
**Then** an online replay buffer MUST be established to fine-tune router logits
**And** distillation loss MUST be logged to results/experiment_1820_moe_distill.json
**And** MODEL_SPECS MUST include "unsloth/Qwen3.6-35B-A3B-GGUF"

### REQ-LEARN-1820 Sub-requirements

- REQ-LEARN-1820-1: `moe_distill.py` SHALL implement the replay buffer and fine-tune router logits.
- REQ-LEARN-1820-2: The artifact SHALL include required schema fields (e.g. distillation_loss, honest_verdict).

## REQ-LEARN-1832: Zero-Constraint Violation Guarantee for Self-Learning

**Given** the COCOM pipeline running continuous self-learning updates
**When** a new safety constraint is evaluated and yields a violation (constraint_value > safety_margin)
**Then** the pipeline MUST enforce a zero-constraint violation guarantee by applying a corrective gradient step
**And** the corrective step MUST oppose the constraint gradient proportionally to the violation magnitude

### REQ-LEARN-1832 Sub-requirements

- REQ-LEARN-1832-1: `COCOMPipeline.update` SHALL accept optional `constraint_value` (default 0.0) and `safety_margin` (default 0.0) parameters.
- REQ-LEARN-1832-2: If `constraint_value > safety_margin`, a corrective step `(constraint_value - safety_margin) * normalized_constraint_grad` SHALL be added to the objective gradient `v`.
- REQ-LEARN-1832-3: The update MUST still perform null-space projection for prior memory constraints.

### SCENARIO-LEARN-1832: Corrective Step Applied on Constraint Violation

**Given** a constraint violation where `constraint_value = 0.5` and `safety_margin = 0.1`
**When** `COCOMPipeline.update` is called
**Then** the gradient `v` receives a correction of magnitude `0.4` along `constraint_grad`
**And** the parameters are updated in the direction of `-constraint_grad` to actively reduce the violation.

## REQ-LEARN-1854: Verification Learning (VL) Proxy for Constraint Satisfaction

**Given** unlabelled continuous stream of constraints and model generations
**When** the Verification Learning proxy evaluates the generations natively
**Then** it MUST score constraint satisfaction directly without external labeled targets
**And** the validation MUST write the proxy result to `results/experiment_1854_vl_proxy.json` with an `honest_verdict` and proxy scores.

### REQ-LEARN-1854 Sub-requirements

- REQ-LEARN-1854-1: `verification_learning.py` SHALL implement a `VerificationLearningProxy` with a proxy loss function based on constraint satisfaction.
- REQ-LEARN-1854-2: The proxy loss function MUST evaluate unlabelled data natively.

### SCENARIO-LEARN-1854: Verification Learning Proxy Execution

**Given** an initialized `VerificationLearningProxy`
**When** unlabelled model generation data is scored against constraints
**Then** it calculates a native score and writes the `results/experiment_1854_vl_proxy.json` successfully.


## REQ-LEARN-1986: Validator-Tree Promotion Ledger for FR-11 Non-Forgetting

**Given** an FR-11 self-learning promotion loop
**When** the validator-tree ledger evaluates a proposed model promotion
**Then** it MUST emit `promotion_gate_passed=true` only if `utility_delta > 0` and non-forgetting conditions hold
**And** it MUST document `soundness_mistakes` and `completeness_mistakes`
**And** write the results to `results/experiment_1986_fr11_routing_without_forgetting.json`.

### SCENARIO-LEARN-1986: Validator-Tree Promotion Ledger Gate

**Given** an initialized `ValidatorTreeLedger`
**When** a promotion is evaluated with positive utility and no forgetting
**Then** it emits `promotion_gate_passed=true` and records the mistakes.

## REQ-LEARN-1725: E2E Pipeline with SOTA, FourierCSP, CIKAN, and Online Updater

**Given** the need to run an end-to-end continuous learning pipeline
**When** Experiment 1725 is executed
**Then** it MUST instantiate a SOTA model (`unsloth/Qwen3.6-35B-A3B-GGUF` or `unsloth/gemma-4-31B-it-GGUF`)
**And** it MUST execute a 50-problem stream where constraints are generated, parsed via FourierCSP, verified via CIKAN, and updated via Online Updater
**And** the experiment artifact MUST contain adaptation rate, model_used, n_processed, and honest_verdict
**And** honest_verdict MUST be `e2e_pipeline_successful` or `e2e_pipeline_failed`.

### SCENARIO-LEARN-1725: Adaptation Rate Measured on 50-Problem Stream

**Given** a SOTA model stream of 50 problems
**When** the pipeline processes the stream
**Then** the Online Updater adapts to the verified violations
**And** the `adaptation_rate` (updated/processed) is computed and written to `results/experiment_1725_e2e_cikan.json`.


---

## REQ-FR11-1681: ETS Policy Evaluation in FR-11 Loop

**Statement:** The FR-11 self-learning loop MUST replace RLHF with Energy-Term Transition Probabilities (ETS) for policy evaluation.
- The self-learning promotion function MUST incorporate test-time energy scaling based on ETS.
- Test-time compute MUST scale with measured energy uncertainty.

**Acceptance criteria:**
- `EtsPolicyEvaluator.scale_test_time_compute(base_compute, uncertainty)` returns scaled compute proportional to uncertainty.
- `EtsPolicyEvaluator.promote_policy(candidate_policy, transition_probabilities, uncertainty)` returns a promotion decision incorporating energy scaling.
- Experiment 1681 confirms ETS promotion works and writes output to `results/experiment_1681_ets_policy.json`.

**Spec traces:** Exp 1681, FR-11 Tier 1

---

## SCENARIO-FR11-1681: Compute Scales with Uncertainty

**Given** an `EtsPolicyEvaluator` and a base compute limit
**When** test-time energy uncertainty is high
**Then** test-time compute is scaled up proportionally
**And** promotion decision accounts for ETS.


## REQ-FR11-1683: FR-11 Policy Soundness Audit MUST Validate Non-forgetting

**Statement:** The FR-11 continuous self-learning loop MUST enforce zero soundness mistakes and maintain a strict non-forgetting rate via a rollback-passing audit.

### REQ-FR11-1683 Sub-requirements

- REQ-FR11-1683-1: `FR11Audit.audit_rollback_passing(artifact_path)` SHALL read the previous experiment artifact.
- REQ-FR11-1683-2: The audit SHALL emit `soundness_mistakes`, `completeness_mistakes`, and `nonforgetting_rate`.
- REQ-FR11-1683-3: For a blocked artifact, it SHALL assume 0 soundness mistakes, 0 completeness mistakes, and 1.0 nonforgetting rate.

### SCENARIO-FR11-1683: Rollback-Passing Audit of Blocked Experiment

**Given** the artifact from Exp 1682 is blocked,
**When** `FR11Audit.audit_rollback_passing()` is executed on it,
**Then** the audit MUST return `soundness_mistakes=0`, `completeness_mistakes=0`, and `nonforgetting_rate=1.0`.

**Spec traces:** REQ-FR11-1683

---

## REQ-LOOP-001: EBFT Autoresearch Loop Measures Energy Delta on Validation Set

**Statement:** The EBFT autoresearch loop SHALL load a dataset of verification trajectories,
apply the EBFT loss to fine-tune a small energy model over multiple steps, measure the
energy on a held-out validation set before and after training, and emit a results artifact
with `baseline_energy`, `final_energy`, `energy_delta`, and `acceptance_gate_passed`.

### REQ-LOOP-001 Sub-requirements

- REQ-LOOP-001-1: `EBFTAutoResearchLoop.__init__` SHALL accept `model_spec`, `n_train_steps`,
  `batch_size`, `lr`, and `seed` parameters.
- REQ-LOOP-001-2: `EBFTAutoResearchLoop.build_dataset` SHALL return `(train_seqs, val_seqs)` as
  JAX arrays of shape `(N, feature_dim)` drawn from a deterministic fixed constraint corpus.
- REQ-LOOP-001-3: `EBFTAutoResearchLoop.measure_energy` SHALL return a scalar float equal to
  the mean L2 energy of a batch of sequences under the current energy model parameters.
- REQ-LOOP-001-4: `EBFTAutoResearchLoop.run` SHALL return a dict with keys `baseline_energy`,
  `final_energy`, `energy_delta`, and `acceptance_gate_passed`; `acceptance_gate_passed` is
  True iff `energy_delta > 0` (energy decreased from baseline to final).

### SCENARIO-LOOP-001: Energy Decreases After Training on Expert Trajectories

**Given** a small energy model initialized with random weights
**When** the EBFT loop trains for `n_train_steps` steps on expert verification trajectories
**Then** the final validation energy is lower than the baseline validation energy
**And** `acceptance_gate_passed` is True

**Spec traces:** REQ-LOOP-001

## REQ-CSL-1778: Establish Continuous Self-Learning Baseline

**Given** the need for continuous self-learning
**When** the baseline script evaluates traces
**Then** it MUST apply current static skill scoring
**And** establish a baseline for retention/forgetting
**And** the output artifact MUST include `schema: "carnot.csl.baseline.v1"` and `baseline_soundness_mistakes` (int).

### SCENARIO-CSL-1778: Baseline evaluation produces sound output
**Given** multi-turn traces are available
**When** the CSL baseline evaluator runs
**Then** it produces a JSON artifact with `schema` equal to "carnot.csl.baseline.v1" and `baseline_soundness_mistakes` as an integer.

## REQ-CSL-1779: Implement non-forgetting soundness check logic

**Given** the need for continuous self-learning
**When** the check script evaluates synthetic trace updates
**Then** it MUST validate soundness
**And** the output artifact MUST include `schema: "carnot.csl.check.v1"` and `check_implemented` (bool).

### SCENARIO-CSL-1779: Non-forgetting soundness check produces correct artifact
**Given** synthetic trace updates are available
**When** the CSL check evaluator runs
**Then** it produces a JSON artifact with `schema` equal to "carnot.csl.check.v1" and `check_implemented` as True.

## REQ-CSL-1780: Run FR-11 CSL loop with unsloth/Qwen3.6-35B-A3B-GGUF

**Given** the non-forgetting framework
**When** the CSL loop script executes the self-learning loop with model `unsloth/Qwen3.6-35B-A3B-GGUF`
**Then** it MUST produce a JSON artifact in `results/experiment_1780_csl_loop.json`
**And** the output artifact MUST include `schema: "carnot.csl.loop.v1"`, `utility_delta` (float), and `soundness_mistakes` (int).

### SCENARIO-CSL-1780: FR-11 CSL loop evaluates utility and soundness
**Given** a loop execution environment
**When** the CSL loop evaluator runs
**Then** it produces a JSON artifact with `schema` equal to "carnot.csl.loop.v1", a float `utility_delta`, and an integer `soundness_mistakes`.

## REQ-LEARN-1741: FR-11 Self-Distillation Memory for Continuous Self-Learning

**Given** the FR-11 continuous self-learning loop running over n_queries verification calls
**When** SelfLearningTracker stores past violated constraints in a DistillationMemory replay buffer
**Then** the system MUST compute a distillation_loss (KL divergence from replay to current weights)
**And** utility_delta (late-window mean precision minus early-window mean precision) MUST be >= 0
**And** the artifact MUST be written to `results/experiment_1741_fr11_self_distillation.json`

This implements the self-distillation-prevents-forgetting mechanism from arXiv:2601.19897.
The replay buffer acts as the "teacher" distribution; the ConstraintTracker precision weights
are the "student".  Keeping KL(teacher || student) low prevents the system from forgetting
historically reliable constraint violation patterns as new queries arrive.

### REQ-LEARN-1741 Sub-requirements

- REQ-LEARN-1741-1: `SelfLearningTracker` in `python/carnot/pipeline/self_learning.py` SHALL
  maintain a `DistillationMemory` replay buffer and a `ConstraintTracker`.
- REQ-LEARN-1741-2: `SelfLearningTracker.record_query()` SHALL update both structures per query.
- REQ-LEARN-1741-3: `SelfLearningTracker.distillation_loss()` SHALL return KL(teacher || student).
- REQ-LEARN-1741-4: `run_fr11_distillation_loop(n_queries=50)` SHALL run 50 synthetic queries
  and return a dict including `utility_delta`, `utility_non_decreasing`, and `replay_buffer_size`.
- REQ-LEARN-1741-5: The artifact MUST include `status: "complete"`,
  `continuous_self_learning_task: true`, `utility_delta: float`, and
  `honest_verdict` starting with `complete:` or `complete_`.

### SCENARIO-LEARN-1741: Self-distillation prevents catastrophic forgetting

**Given** a SelfLearningTracker running 50 queries with progressively improving precision
**When** run_fr11_distillation_loop() completes
**Then** `utility_non_decreasing` MUST be true (late-window utility >= early-window utility)
**And** `distillation_loss` MUST be a non-negative float
**And** all 5 constraint types MUST be observed

## Implementation Status (REQ-LEARN-1741)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-1741 | Implemented (`python/carnot/pipeline/self_learning.py`) | Implemented (`tests/python/test_experiment_1741_fr11_self_distillation.py`) |

## REQ-LEARN-1827: Muon-OGD Spectral Orthogonal Gradient Projection

**Given** the FR-11 continual learning stack
**When** optimizing parameters across multiple sequential tasks
**Then** the stack MUST support Muon-OGD (arXiv 2604.14818) to project gradients orthogonally to previously learned knowledge, protecting prior constraints from catastrophic forgetting.

### SCENARIO-LEARN-1827: Muon-OGD Orthogonal Projection
**Given** an initialized Muon-OGD optimizer
**When** sequential gradient updates are applied
**Then** updates MUST be projected orthogonally against the running memory matrix of prior task gradients.

## REQ-LEARN-2066: EBRM Training Loop via Contrastive Divergence

**Summary:** Carnot SHALL implement a lightweight Energy-Based Reward Model (EBRM) training loop using Contrastive Divergence on synthetic logical derivations.

- REQ-LEARN-2066-1: The repository SHALL provide `python/carnot/training/ebrm_trainer.py`.
- REQ-LEARN-2066-2: The module SHALL contain an `EBRMTrainer` class using a small JAX MLP to act as the Energy Reward Model.
- REQ-LEARN-2066-3: The trainer SHALL implement a Contrastive Divergence step taking synthetic logical traces and generating an energy loss.
- REQ-LEARN-2066-4: A script `scripts/experiment_2066_ebrm_training.py` SHALL be implemented to train the artifact.
- REQ-LEARN-2066-5: The script SHALL output an artifact `results/experiment_2066_ebrm_training.json` with `ebrm_trained=true`.

### SCENARIO-LEARN-2066: EBRM Trainer executes CD Loop

**Given** the `EBRMTrainer` initialized with a small JAX MLP
**When** the `train()` method is called with synthetic logical traces
**Then** the energy model weights MUST be updated using Contrastive Divergence
**And** it outputs a complete deliverable to `results/experiment_2066_ebrm_training.json` with `ebrm_trained=true`.

### REQ-LEARN-2127: AdamFLIP Integration into CSL Loop

AdamFLIP SHALL apply adaptive momentum feedback linearization to solve hard constraints robustly. It SHALL be integrated into the CSL loop to update constraint residuals as a feedback input.

## REQ-LEARN-2136: Substrate Shifting CSL
**Given** the Continuous Self-Learning (CSL) loop
**When** mode concentration is detected and substrate_shift parameters are provided
**Then** the CSL loop MUST translate the underlying KAN LUT grids by applying the energy grid translation parameters
**And** the experiment artifact MUST report `substrate_shifting_ready=True` and `integrated_with_kan_tiers=True`.

### SCENARIO-LEARN-2136: Substrate Shifting Applied
**Given** grid_translation parameters
**When** run_csl_loop is executed with substrate_shift
**Then** the parameters MUST be translated and the artifact saved.


## REQ-LEARN-2152: CSL Loop Incorporates PREM Intrinsic Rewards

**Given** a Continuous Self-Learning (CSL) loop
**When** PREM intrinsic rewards are provided during the step
**Then** the optimizer parameters MUST be shifted by the intrinsic rewards
**And** the run_csl_loop output MUST reflect that PREM intrinsic rewards were applied

### SCENARIO-LEARN-2152: PREM Intrinsic Rewards Shift CSL Parameters

**Given** a CSL loop
**When** run_csl_loop is called with prem_intrinsic_reward
**Then** the returned dict includes 'prem_intrinsic_applied': True
**And** the parameters are updated accordingly.

## REQ-LEARN-2241: FR-11 Fast-Slow Training Evaluation Artifact

**Given** the FR-11 continuous self-learning milestone rescue task
**When** the Exp 2241 FST evaluation runs
**Then** it MUST verify that `python/carnot/training/fast_slow.py` can be
imported directly before running the experiment
**And** it MUST use a 30-example FoVer-class synthetic arithmetic CoT corpus
without external LLM API calls
**And** it MUST compare a 5-iteration parameter-only verifier-weight RL regime
against a 5-iteration Fast-Slow Training regime with frozen slow weights and
verifier-output-summary fast-weight updates
**And** it MUST write `results/experiment_2241_fr11_fst_eval.json` with
`continuous_self_learning_task=true`, `sample_efficiency_ratio`,
`kl_drift_ratio`, `utility_delta`, `n_corpus`, `preconditions_checked`,
`fr11_fst_eval_passed`, and `honest_verdict`.

### REQ-LEARN-2241 Sub-requirements

- REQ-LEARN-2241-1: If direct import of `fast_slow.py` fails, the artifact
  SHALL be written with `honest_verdict="blocked_fst_module_missing"`.
- REQ-LEARN-2241-2: `fr11_fst_eval_passed` SHALL be true only when
  `sample_efficiency_ratio >= 2.0` and `kl_drift_ratio <= 0.5`.
- REQ-LEARN-2241-3: Passing artifacts SHALL use an `honest_verdict` that starts
  with `complete:`.
- REQ-LEARN-2241-4: The artifact SHALL record enough regime history for the
  sample-efficiency and KL-drift ratios to be independently audited.

### SCENARIO-LEARN-2241: FST Beats Parameter-Only RL on Synthetic FoVer Arithmetic

**Given** a deterministic 30-example synthetic arithmetic CoT corpus
**When** Exp 2241 evaluates parameter-only RL and FST for 5 iterations each
**Then** the FST regime reaches the energy-reduction threshold at least twice
as quickly as the parameter-only regime
**And** the FST regime's verifier-parameter KL drift is at most half the
parameter-only regime's drift
**And** the artifact reports `continuous_self_learning_task=true`.

## Implementation Status (REQ-LEARN-2241)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2241 | Implemented (`python/carnot/reporting/fr11_fst_eval.py`, `scripts/experiment_2241_fr11_fst_eval.py`) | Implemented (`tests/python/test_experiment_2241_fr11_fst_eval.py`) |

## REQ-LEARN-2242: ActFocus + FST SOTA GGUF Evaluation Artifact

**Given** the ActFocus token-level energy reweighting task and the FR-11
Fast-Slow Training verifier loop
**When** the Exp 2242 combined evaluation runs
**Then** it MUST directly import `python/carnot/training/actfocus.py` and
`python/carnot/training/fast_slow.py` before evaluation
**And** it MUST call the cached SOTA GGUF resolver path before any fallback
model-spec construction
**And** it MUST run a deterministic 20-example reasoning corpus through an
FST-enabled verify-repair loop whose fast-update retention is scored from
ActFocus action-token energy variance
**And** it MUST write `results/experiment_2242_actfocus_fst.json` with
`honest_verdict`, `actfocus_fst_validated`, `models_used`,
`energy_variance_correlation`, `fast_weight_retention_rate`,
`energy_reduction`, `preconditions_checked`, and `duration_s`.

### REQ-LEARN-2242 Sub-requirements

- REQ-LEARN-2242-1: If direct import of `actfocus.py` fails, the artifact SHALL
  be written with `honest_verdict="blocked_actfocus_missing"`.
- REQ-LEARN-2242-2: If direct import of `fast_slow.py` fails, the artifact
  SHALL be written with `honest_verdict="blocked_fst_missing"`.
- REQ-LEARN-2242-3: If no Qwen or Gemma local HF cache entry is present, the
  artifact SHALL be written with `honest_verdict="blocked_model_not_cached"`.
- REQ-LEARN-2242-4: `MODEL_SPECS` SHALL include at least one of
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- REQ-LEARN-2242-5: `actfocus_fst_validated` SHALL be true only when
  `fast_weight_retention_rate >= 0.85`.

### SCENARIO-LEARN-2242: ActFocus Action Variance Retains Useful FST Updates

**Given** a deterministic 20-example reasoning corpus with answer-action
tokens and reasoning tokens
**When** ActFocus scores token-energy traces and FST consumes failed verifier
summaries as fast weights
**Then** action-token energy variance is positively correlated with useful
fast-weight updates
**And** the artifact reports whether the retention gate reached 0.85.

## REQ-LEARN-2844: LoopUS-Style External FR-11 Recurrence Pilot

**Given** Exp 2836 reports `sota_runtime_ready=true` and a mandated SOTA GGUF
path
**When** the Exp 2844 continuous self-learning pilot runs
**Then** it SHALL select exactly 25 FoVer and 25 MBPP examples with a
deterministic seed
**And** it SHALL run at most three external recurrence loops per example without
modifying model weights
**And** each later loop SHALL receive localized Carnot energy-violation feedback
from the previous attempt
**And** it SHALL stop early when the answer passes or energy improvement is
below the configured convergence threshold
**And** it SHALL write
`results/experiment_2844_loopus_fr11_self_learning_pilot.json` with
`honest_verdict`, `continuous_self_learning_task`, `n_examples`,
`mean_energy_delta_loop0_to_final`, `correctness_delta`, `early_exit_rate`,
`per_example_trace`, `model_specs`, `preconditions_checked`, and `duration_s`.

### REQ-LEARN-2844 Sub-requirements

- REQ-LEARN-2844-1: The runner SHALL load Exp 2836, require
  `sota_runtime_ready=true`, and record `selected_python` plus the selected
  mandated GGUF path in `model_specs`.
- REQ-LEARN-2844-2: If FoVer, MBPP, verifier, SOTA model-path, or live
  recurrence backend preconditions fail, the runner SHALL write a blocked
  artifact whose `honest_verdict` starts with `blocked_`, whose
  `n_examples=0`, and whose delta fields are explicit zero sentinels rather
  than inferred measurements.
- REQ-LEARN-2844-3: `run_recurrence_pilot()` SHALL compute
  `mean_energy_delta_loop0_to_final` as mean loop0 energy minus final-loop
  energy, `correctness_delta` as final correctness rate minus loop0
  correctness rate, and `early_exit_rate` as the fraction of examples using
  fewer than three loops.
- REQ-LEARN-2844-4: Every `per_example_trace` row SHALL include corpus,
  example id, per-loop energy, localized feedback, final correctness, early
  exit reason, and token cost.

### SCENARIO-LEARN-2844: Recurrence Lowers Energy and Reports Correctness Delta

**Given** a deterministic mixed FoVer/MBPP sample and an attached generation
backend that improves after localized energy feedback
**When** the Exp 2844 pilot runs
**Then** the success artifact starts with `complete:`
**And** `mean_energy_delta_loop0_to_final` is positive when final energy is lower
than loop0 energy
**And** `correctness_delta` is positive when recurrence repairs initially wrong
answers
**And** at least one trace exits before loop2 when energy converges or the answer
passes.

### SCENARIO-LEARN-2844-BLOCKED: Missing Recurrence Backend Blocks Without Deltas

**Given** Exp 2836, FoVer, MBPP, and verifier preconditions are available
**But** no live recurrence backend is attached
**When** the Exp 2844 pilot runs
**Then** `honest_verdict` is `blocked_live_recurrence_backend`
**And** `n_examples=0`, `per_example_trace=[]`, and the delta fields are zero
sentinels with a methodology note explaining that no recurrence was measured.

## Implementation Status (REQ-LEARN-2844)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2844 | Implemented (`python/carnot/eval/loopus_fr11_self_learning_pilot.py`, `scripts/experiment_2844_loopus_fr11_self_learning_pilot.py`) | Implemented (`tests/python/test_experiment_2844_loopus_fr11_self_learning_pilot.py`) |

## REQ-LEARN-2857: LoopUS FR-11 Self-Learning Pilot v2 Gated By Exp 2856

**Given** Exp 2856 has written
`results/experiment_2856_loopus_recurrence_backend_adapter.json` with
`live_recurrence_backend_ready=true`
**When** the Exp 2857 continuous self-learning pilot runs
**Then** it SHALL use the selected Exp 2856 recurrence backend to evaluate at
least 50 mixed FoVer, code, and factual examples
**And** it SHALL run initial answer generation, Carnot energy feedback,
bounded external recurrence refinement, and adaptive early exit for at most
three loops without mutating model weights
**And** it SHALL write
`results/experiment_2857_loopus_fr11_self_learning_v2.json` with
`honest_verdict`, `continuous_self_learning_task`, `fr11_self_learning_ready`,
`n_examples`, `source_counts`, `max_loops`, `recurrence_success_rate`,
`energy_delta_mean`, `correctness_delta`, `per_loop_energy_summary`,
`early_exit_summary`, `memory_hash_before`, `memory_hash_after`,
`no_model_weight_mutation`, `model_specs`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`, `duration_s`,
`adversarial_verify_passed`, `adversarial_verify_flags`, and `run_date`.

### REQ-LEARN-2857 Sub-requirements

- REQ-LEARN-2857-1: Before selecting examples or running recurrence, the runner
  SHALL record the repository directory precondition, execute
  `jq -e '.live_recurrence_backend_ready == true'
  results/experiment_2856_loopus_recurrence_backend_adapter.json`, and execute
  `.venv/bin/python3 -c "import json;
  p='results/experiment_2856_loopus_recurrence_backend_adapter.json';
  print(json.load(open(p))['backend_module_path'])"`.
- REQ-LEARN-2857-2: If any Exp 2856 precondition fails, the runner SHALL write a
  terminal `blocked_<resource>` artifact with `n_examples=0`, empty source and
  per-loop summaries, zero delta sentinels, precondition evidence, and no
  inferred recurrence metrics.
- REQ-LEARN-2857-3: For a successful gated run, aggregate energy and correctness
  deltas SHALL be computed from measured per-example traces, and
  `recurrence_success_rate` SHALL be the fraction of examples whose final
  energy is lower than initial energy or whose correctness improved.
- REQ-LEARN-2857-4: The artifact SHALL record memory/session hashes before and
  after any permitted FR-11 counter or memory update and SHALL keep
  `no_model_weight_mutation=true`.

### SCENARIO-LEARN-2857-BLOCKED: Missing Exp 2856 Blocks Without Pilot Metrics

**Given** `results/experiment_2856_loopus_recurrence_backend_adapter.json` is
missing or does not report `live_recurrence_backend_ready=true`
**When** the Exp 2857 runner executes
**Then** `honest_verdict` starts with `blocked_`
**And** `fr11_self_learning_ready=false`, `n_examples=0`,
`recurrence_success_rate=0.0`, `energy_delta_mean=0.0`, and
`correctness_delta=0.0`
**And** the artifact records the failed Exp 2856 command evidence and an
adversarial verification flag describing the failed precondition.

### SCENARIO-LEARN-2857: Successful Gated Run Reports Recurrence Effects

**Given** Exp 2856 reports a ready mandated SOTA recurrence backend and the
measurement layer returns per-example traces with energy and correctness before
and after recurrence
**When** the Exp 2857 runner completes
**Then** the artifact starts with `complete:`
**And** it reports the measured source counts, per-loop energy summary,
early-exit summary, final energy delta mean, correctness delta, recurrence
success rate, memory hashes, model specs, and reproducibility checksum.

## Implementation Status (REQ-LEARN-2857)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2857 | Implemented (`python/carnot/eval/loopus_fr11_self_learning_v2.py`, `scripts/experiment_2857_loopus_fr11_self_learning_v2.py`) | Implemented (`tests/python/test_experiment_2857_loopus_fr11_self_learning_v2.py`) |

## REQ-LEARN-2868: Offline Recurrence Backend Adapter for Verifier Trace Replay

**Given** FR-11 LoopUS live recurrence is blocked by the missing backend artifact
**When** Exp 2868 selects an offline recurrence backend
**Then** the backend SHALL accept existing verifier trace rows without invoking a
live LLM
**And** it SHALL emit deterministic revised energy summaries from the supplied
trace rows
**And** it SHALL expose a stable `backend_module_path` for downstream Exp 2869
gating
**And** it SHALL report `live_recurrence_backend_ready=false`,
`live_model_invoked=false`, and `no_model_weight_mutation=true`.

### REQ-LEARN-2868 Sub-requirements

- REQ-LEARN-2868-1: `OfflineRecurrenceReplayBackend.replay(rows)` SHALL accept
  an iterable of mapping-like verifier trace rows and return a mapping with
  `backend_module_path`, `backend_api`, `n_rows`, `per_example_trace`,
  `per_loop_energy_summary`, `energy_delta_mean`, `replay_smoke_passed`,
  `live_model_invoked=false`, and `no_model_weight_mutation=true`.
- REQ-LEARN-2868-2: Each replay trace row SHALL include `example_id`,
  `energy_before`, `energy_after_each_loop`, `correctness_before`,
  `correctness_after`, and `early_exit_reason`. Missing optional source fields
  SHALL default to `"verifier_trace"`.
- REQ-LEARN-2868-3: `run_experiment()` SHALL write
  `results/experiment_2868_offline_recurrence_backend_adapter_v2.json` with the
  required Exp 2868 artifact fields, including `honest_verdict`,
  `offline_recurrence_backend_ready`, `live_recurrence_backend_ready`,
  `backend_module_path`, `backend_api`, `replay_smoke_passed`,
  `live_model_invoked`, `no_model_weight_mutation`, `preconditions_checked`,
  `tests_run`, `random_seed`, `reproducibility_checksum`, `field_principles`,
  `run_date`, and `duration_s`.

### SCENARIO-LEARN-2868: Offline Replay Backend Emits Stable Energy Summary

**Given** two deterministic verifier trace rows with initial verifier energy and
revised verifier energy
**When** `OfflineRecurrenceReplayBackend.replay()` is called
**Then** the backend returns a stable import path, a positive energy delta for
improved rows, loop-level energy means, and mutation flags proving no live model
or model weights were touched.

## Implementation Status (REQ-LEARN-2868)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2868 | Implemented (`python/carnot/eval/offline_recurrence_backend_adapter_v2.py`) | Implemented (`tests/python/test_experiment_2868_offline_recurrence_backend_adapter_v2.py`) |

## REQ-LEARN-2869: FR-11 Continuous Self-Learning Offline Replay

**Given** Exp 2868 exposes an offline recurrence backend and Exp 2865 marks
FoVer plus HaluEval/FEVER as clean corpus rows
**When** Exp 2869 runs the bounded FR-11 continuous self-learning replay
**Then** it SHALL import the Exp 2868 `backend_module_path`, select a
deterministic small replay corpus from the clean Exp 2865 rows, hash relevant
constraint memory/session state before the run, replay at most three verifier
feedback loops with early exit on no improvement, write only permitted
constraint-memory replay state, hash memory/session state after the run, and
write `results/experiment_2869_fr11_continuous_self_learning_replay_v3.json`.
**And** it SHALL report energy, correctness, recurrence-success, forgetting,
memory-hash, source-count, provenance, and no-mutation fields without invoking a
live LLM or mutating model weights.

### REQ-LEARN-2869 Sub-requirements

- REQ-LEARN-2869-1: The runner SHALL load
  `results/experiment_2868_offline_recurrence_backend_adapter_v2.json`, import
  its `backend_module_path`, and record the imported path in
  `offline_recurrence_backend_used`.
- REQ-LEARN-2869-2: The runner SHALL load
  `results/experiment_2865_cross_corpus_matrix_v5.json`, use only corpus rows
  whose status is `"clean"`, and select a deterministic bounded replay corpus
  controlled by `random_seed`.
- REQ-LEARN-2869-3: The runner SHALL hash relevant constraint memory/session
  files before and after replay and SHALL verify that changed memory files are a
  subset of the Exp 2869 permitted memory-state path.
- REQ-LEARN-2869-4: The runner SHALL replay at most three loops, stop each
  example when no energy improvement occurs, and compute
  `energy_delta_mean`, `correctness_delta`, `recurrence_success_rate`,
  `per_loop_energy_summary`, and `forgetting_regression_count` from the
  backend-normalized traces.
- REQ-LEARN-2869-5: The artifact SHALL include `honest_verdict`,
  `continuous_self_learning_task=true`, `fr11_self_learning_ready`,
  `offline_recurrence_backend_used`, `live_model_invoked=false`,
  `no_model_weight_mutation=true`, `n_examples`, `max_loops`,
  `recurrence_success_rate`, `energy_delta_mean`, `correctness_delta`,
  `forgetting_regression_count`, `memory_hash_before`, `memory_hash_after`,
  `per_loop_energy_summary`, `source_counts`, `preconditions_checked`,
  `random_seed`, `reproducibility_checksum`, `field_principles`, `run_date`, and
  `duration_s`.

### SCENARIO-LEARN-2869: Offline Verifier-Feedback Replay Updates Constraint Memory

**Given** Exp 2868 is importable and Exp 2865 lists clean FoVer and
HaluEval/FEVER rows
**When** the Exp 2869 runner replays the deterministic small corpus through the
offline backend
**Then** the artifact starts with `complete:`
**And** energy decreases on replayed verifier-feedback rows while correctness is
not inferred from live repair
**And** memory hashes are recorded before and after the permitted memory update
**And** `live_model_invoked=false`, `no_model_weight_mutation=true`, and
`forgetting_regression_count=0`.

### SCENARIO-LEARN-2869-BLOCKED: Missing Offline Backend Blocks Without Fake Metrics

**Given** the Exp 2868 backend artifact is missing or cannot be imported
**When** the Exp 2869 runner executes
**Then** `honest_verdict` starts with `blocked_`
**And** `fr11_self_learning_ready=false`, `n_examples=0`,
`recurrence_success_rate=0.0`, `energy_delta_mean=0.0`,
`correctness_delta=0.0`, and memory hashes are explicit blocked sentinels.

## Implementation Status (REQ-LEARN-2869)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2869 | Implemented (`python/carnot/eval/fr11_continuous_self_learning_replay_v3.py`) | Implemented (`tests/python/test_experiment_2869_fr11_continuous_self_learning_replay_v3.py`) |

## REQ-LEARN-2881: FR-11 RecMem-Style Recurrence-Triggered Consolidation

**Given** Exp 2868 exposes the offline recurrence backend and Exp 2869 provides
the prior FR-11 replay artifact
**When** Exp 2881 ingests deterministic offline verifier-feedback events
**Then** it SHALL store each event in a subconscious buffer using only
deterministic verifier metadata and lightweight local text features
**And** it SHALL trigger consolidation only for sustained recurrence clusters of
semantically similar verifier failures or constraint motifs
**And** it SHALL avoid live LLM calls while reporting memory drift, duplicate,
contradiction, non-forgetting, and token-cost proxy metrics.

### REQ-LEARN-2881 Sub-requirements

- REQ-LEARN-2881-1: `SubconsciousRecurrenceBuffer` SHALL ingest replay events
  without consolidating them eagerly and SHALL derive deterministic motif
  features from source, verifier failure, localized violation, and
  early-exit metadata.
- REQ-LEARN-2881-2: `RecurrenceDetector` SHALL form recurrence clusters when
  event-feature similarity is at least `recurrence_threshold` and SHALL require
  sustained support before a cluster is consolidation-ready.
- REQ-LEARN-2881-3: The consolidation workflow SHALL write only a dedicated
  Exp 2881 memory artifact, hash memory state before and after the write, and
  report duplicate rate, contradiction rate, and non-forgetting regressions
  over the prior replay rows.
- REQ-LEARN-2881-4: The token-cost proxy SHALL count eager consolidation calls
  avoided, bytes or token-like units before and after consolidation, and
  threshold sensitivity without using a live LLM.
- REQ-LEARN-2881-5: The terminal artifact
  `results/experiment_2881_fr11_recmem_recurrence_trigger_v1.json` SHALL
  include `honest_verdict`, `continuous_self_learning_task=true`,
  `recmem_trigger_ready`, `source_artifacts`, `recurrence_threshold`,
  `n_events_ingested`, `n_recurrence_clusters`, `n_consolidations_triggered`,
  `eager_consolidations_avoided`, `token_reduction_proxy_pct`,
  `memory_hash_before`, `memory_hash_after`, `contradiction_rate`,
  `duplicate_rate`, `forgetting_regression_count`, `live_llm_called=false`,
  `tests_run`, `field_principles`, `run_date`, and `duration_s`.

### SCENARIO-LEARN-2881: Recurring Failure Motifs Trigger Consolidation

**Given** offline replay events contain repeated factuality-mismatch verifier
failures plus non-recurring clean rows
**When** the recurrence detector runs at the configured threshold
**Then** only the recurring failure motif cluster is consolidated
**And** eager consolidation calls are avoided for the remaining buffered events
**And** memory hashes differ after the dedicated consolidation memory write.

### SCENARIO-LEARN-2881-GUARD: Drift Guards Block Unsafe Consolidation

**Given** a candidate cluster contains contradictory before/after correctness
metadata for the same motif or regresses prior replay energy
**When** Exp 2881 evaluates drift guards
**Then** the artifact reports a non-zero contradiction or forgetting metric and
`recmem_trigger_ready=false`.

## Implementation Status (REQ-LEARN-2881)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2881 | Implemented (`python/carnot/eval/fr11_recmem_recurrence_trigger_v1.py`) | Implemented (`tests/python/test_experiment_2881_fr11_recmem_recurrence_trigger_v1.py`) |

## REQ-LEARN-2882: FR-11 RecMem Replay Scale-Up Comparison

**Given** Exp 2869 provides the prior FR-11 offline replay result and Exp 2881
provides a RecMem-style recurrence trigger
**When** Exp 2882 runs a bounded offline FR-11 replay scale-up
**Then** it SHALL select a deterministic local labeled replay set using the same
seed and examples for eager replay and recurrence-triggered replay
**And** it SHALL compare energy, correctness, AUROC where both label classes
exist, token cost, memory drift, and forgetting without calling a live LLM or
mutating model weights.

### REQ-LEARN-2882 Sub-requirements

- REQ-LEARN-2882-1: The runner SHALL load
  `results/experiment_2869_fr11_continuous_self_learning_replay_v3.json` and
  `results/experiment_2881_fr11_recmem_recurrence_trigger_v1.json`, require
  their ready flags before claiming a complete run, and list them in
  `source_artifacts`.
- REQ-LEARN-2882-2: The runner SHALL select the largest deterministic local
  labeled replay set allowed by its bounded example budget, target at least 50
  examples, and report `n_examples` plus `target_examples_met`.
- REQ-LEARN-2882-3: The eager and RecMem-triggered replay paths SHALL use
  identical selected examples, `random_seed`, and offline recurrence backend
  rows; the RecMem path SHALL apply replay effects only to examples in
  recurrence-ready clusters.
- REQ-LEARN-2882-4: The artifact SHALL compute `energy_delta_mean`,
  `correctness_delta`, `auroc_delta`, `token_reduction_pct`,
  `memory_drift_score`, and `forgetting_regression_count` from the eager versus
  RecMem-triggered comparison, and SHALL include enough auxiliary eager/RecMem
  metrics to audit the comparison.
- REQ-LEARN-2882-5: The terminal artifact
  `results/experiment_2882_fr11_recmem_replay_scaleup_v1.json` SHALL include
  `honest_verdict`, `continuous_self_learning_task=true`,
  `recmem_replay_scaleup_ready`, `source_artifacts`, `n_examples`,
  `target_examples_met`, `energy_delta_mean`, `correctness_delta`,
  `auroc_delta`, `token_reduction_pct`, `memory_drift_score`,
  `forgetting_regression_count`, `model_weights_mutated=false`,
  `live_llm_called=false`, `random_seed`, `field_principles`,
  `run_date="20260522"`, and `duration_s`.

### SCENARIO-LEARN-2882: RecMem Trigger Matches Eager Replay at Lower Token Cost

**Given** at least 50 local clean labeled FoVer/HaluEval/FEVER replay rows
**When** Exp 2882 runs eager replay and RecMem-triggered replay over the same
selected rows
**Then** the artifact starts with `complete:`
**And** RecMem-triggered replay reports the same non-forgetting result as eager
replay while reducing token cost
**And** `model_weights_mutated=false`, `live_llm_called=false`, and
`forgetting_regression_count=0`.

### SCENARIO-LEARN-2882-BLOCKED: Missing Source Artifacts Block Without Fake Metrics

**Given** Exp 2869 or Exp 2881 is missing or not ready
**When** Exp 2882 runs
**Then** `honest_verdict` starts with `blocked_`
**And** `recmem_replay_scaleup_ready=false`, `n_examples=0`,
`energy_delta_mean=0.0`, `correctness_delta=0.0`, `auroc_delta=0.0`,
`token_reduction_pct=0.0`, `memory_drift_score=0.0`, and
`forgetting_regression_count=0`.

## Implementation Status (REQ-LEARN-2882)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2882 | Planned (`python/carnot/eval/fr11_recmem_replay_scaleup_v1.py`) | Planned (`tests/python/test_experiment_2882_fr11_recmem_replay_scaleup_v1.py`) |

## REQ-LEARN-2887: FR-11 Fast-Slow Memory Corrigendum For RecMem Scale-Up

**Given** Exp 2869 provides offline FR-11 replay rows, Exp 2881 provides a
clean RecMem recurrence trigger, and Exp 2882 was adversarially flagged for
retroactive eager-vs-RecMem metric parity
**When** Exp 2887 runs the bounded corrigendum
**Then** it SHALL load Exp 2869, Exp 2881, and Exp 2882, explain the exact
Exp 2882 flag cause, select the same deterministic local labeled replay set
for every compared policy, and compare eager replay, causal
recurrence-triggered replay, and a Memini-style fast/slow external-memory
baseline without mutating model weights or calling a live LLM.
**And** it SHALL write
`results/experiment_2887_fr11_fast_slow_memory_corrigendum_v2.json` with
per-policy energy, correctness, AUROC, token, duplicate, contradiction, memory
drift, and forgetting metrics.

### REQ-LEARN-2887 Sub-requirements

- REQ-LEARN-2887-1: The runner SHALL require
  `results/experiment_2869_fr11_continuous_self_learning_replay_v3.json`,
  `results/experiment_2881_fr11_recmem_recurrence_trigger_v1.json`, and
  `results/experiment_2882_fr11_recmem_replay_scaleup_v1.json` before claiming
  a complete corrigendum, and SHALL list them in `source_artifacts`.
- REQ-LEARN-2887-2: The runner SHALL select a deterministic local labeled
  replay set with target >= 50 examples, record selected-example and source-file
  checksums, and report why if fewer examples are available.
- REQ-LEARN-2887-3: The eager, causal RecMem, and fast/slow-memory policies
  SHALL consume identical selected examples and seeds. The causal RecMem policy
  SHALL apply replay only after recurrence support already exists, never
  retroactively to examples that created the trigger.
- REQ-LEARN-2887-4: The fast/slow-memory baseline SHALL maintain fast episodic
  association strength, slow consolidated edge strength, and selective
  forgetting for contradictory or regressive evidence using external memory
  state only.
- REQ-LEARN-2887-5: `fr11_scaleup_clean` SHALL be true only when the target
  example count is met, no live LLM or model-weight mutation occurs, the
  current per-policy metrics are non-tautological, and drift/forgetting guards
  pass.

### SCENARIO-LEARN-2887: Corrigendum Separates Replay Policies Causally

**Given** at least 50 local clean labeled FoVer/HaluEval/FEVER replay rows
**When** Exp 2887 compares eager replay, causal RecMem replay, and fast/slow
memory over the same selected examples
**Then** eager replay can apply every offline replay effect immediately
**And** causal RecMem waits for prior recurrence support
**And** fast/slow memory can apply fast episodic associations before slow
consolidation while retaining selective forgetting guards
**And** the artifact records per-policy metric dictionaries instead of a single
eager-vs-RecMem scalar.

### SCENARIO-LEARN-2887-GUARD: Fast-Slow Memory Forgets Unsafe Edges

**Given** a fast/slow memory edge has accumulated recurring positive evidence
**When** contradictory or regressive evidence for the same motif arrives
**Then** the fast association and slow consolidated edge are weakened
**And** the policy reports non-zero drift or forgetting instead of preserving an
unsafe memory update.

## Implementation Status (REQ-LEARN-2887)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2887 | Implemented (`python/carnot/eval/fr11_fast_slow_memory_corrigendum_v2.py`) | Implemented (`tests/python/test_experiment_2887_fr11_fast_slow_memory_corrigendum_v2.py`) |

## REQ-LEARN-2906: FR-11 KV260 Hardware-Accelerated Replay Pilot

**Given** Exp 2882 provides the bounded CPU replay scale-up artifact, Exp 2887
provides the clean fast/slow memory corrigendum for that replay path, and Exp
2898 provides a live KV260 board dispatch artifact
**When** Exp 2906 builds the hardware-accelerated replay pilot artifact
**Then** it SHALL aggregate only those upstream artifacts, validate that the CPU
replay path and live KV260 dispatch path both passed their required gates, and
write `results/experiment_2906_fr11_hardware_accelerated_replay_pilot_v1.json`.
**And** it SHALL explicitly report that this is a pilot-only dispatch-path
validation, not a hardware performance comparison.

### REQ-LEARN-2906 Sub-requirements

- REQ-LEARN-2906-1: The runner SHALL require
  `results/experiment_2882_fr11_recmem_replay_scaleup_v1.json`,
  `results/experiment_2887_fr11_fast_slow_memory_corrigendum_v2.json`, and
  `results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json`
  before setting `dispatch_path_validated=true`.
- REQ-LEARN-2906-2: The runner SHALL validate the replay side by requiring
  `recmem_replay_scaleup_ready=true`, `n_examples >= 50`,
  `live_llm_called=false`, `model_weights_mutated=false`, and
  `fr11_scaleup_clean=true` in the corrigendum artifact.
- REQ-LEARN-2906-3: The runner SHALL validate the KV260 side by requiring a
  terminal complete Exp 2898 verdict, `inference_substrate="hardware_smoke"`,
  all listed preconditions available, a non-empty loaded Carnot overlay,
  at least one `/dev/uio*` device, a board-selected UIO path, a bitstream
  SHA256, and three positive per-seed board result rows.
- REQ-LEARN-2906-4: The terminal artifact SHALL include `honest_verdict`,
  `inference_substrate="aggregation_from_upstream_artifacts"`,
  `dispatch_path_validated`, `cited_upstream_artifacts`, `duration_s`,
  `pilot_only=true`, `no_hardware_performance_claim=true`, and enough imported
  field summaries to audit the path without re-running the board.

### SCENARIO-LEARN-2906: KV260 Replay Pilot Validates Dispatch Without Performance Claim

**Given** the Exp 2882, Exp 2887, and Exp 2898 artifacts all exist and satisfy
their replay and board-dispatch gates
**When** Exp 2906 runs
**Then** it writes the pilot artifact with `dispatch_path_validated=true`,
`inference_substrate="aggregation_from_upstream_artifacts"`, upstream artifact
SHA256 citations, `pilot_only=true`, and `no_hardware_performance_claim=true`.

### SCENARIO-LEARN-2906-BLOCKED: Missing Or Failed Upstream Blocks The Pilot

**Given** any required upstream artifact is missing or fails its replay or KV260
dispatch gate
**When** Exp 2906 runs
**Then** it writes a blocked artifact with `dispatch_path_validated=false`,
the failed gate names, `inference_substrate="aggregation_from_upstream_artifacts"`,
and no hardware performance comparison fields.

## Implementation Status (REQ-LEARN-2906)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2906 | Implemented (`python/carnot/eval/fr11_hardware_accelerated_replay_pilot_v1.py`) | Implemented (`tests/python/test_experiment_2906_fr11_hardware_accelerated_replay_pilot_v1.py`) |

## REQ-LEARN-2918: FR-11 Verifiable Process Rewards Replay Update

**Given** Exp 2911 has labeled code-generation candidates with deterministic
syntax/static/runtime verifier outcomes and Exp 2912 has produced a ready
same-basis CPU Gibbs baseline for the KV260 problem basis
**When** Exp 2918 runs a bounded FR-11 replay experiment
**Then** it SHALL construct a deterministic replay corpus from verified code
candidates, failed code-hallucination labels, matched sampler measurements, and
prior FR-11 memory examples.
**And** it SHALL derive process-reward weights from verifier evidence rather
than imitation likelihood.
**And** it SHALL perform an online external-memory replay update when no
trainable model component is available, measuring before/after correctness,
energy proxy, replay priority, duplicate or contradiction rate, and forgetting
on held-out prior rows.

### REQ-LEARN-2918 Sub-requirements

- REQ-LEARN-2918-1: The runner SHALL require
  `results/experiment_2911_code_hallucination_taxonomy_verifier_v1.json` with
  `code_hallucination_verifier_ready=true` and
  `results/experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json` with
  `same_basis_cpu_baseline_ready=true`; if either gate fails it SHALL write a
  blocked artifact and exit without claiming an online update.
- REQ-LEARN-2918-2: Code process rewards SHALL be deterministic functions of
  syntax success, static hallucination labels, runtime/test pass status, and
  failed hallucination labels.
- REQ-LEARN-2918-3: Hardware process rewards SHALL be deterministic functions
  of same-basis checks, final-energy proxy, latency proxy, and matched
  sampler-trajectory metadata.
- REQ-LEARN-2918-4: The online update SHALL mutate only an explicit replay
  scheduler or priority table unless a real trainable component is provided,
  and the artifact SHALL report that scope honestly via
  `online_update_performed` and `replay_scheduler_updated`.
- REQ-LEARN-2918-5: The deliverable artifact SHALL include
  `honest_verdict`, `online_self_learning_ready`, `fr11_requirement_targeted`,
  `process_reward_definition`, `replay_corpus_summary`,
  `online_update_performed`, `replay_scheduler_updated`, `delta_overall`,
  `delta_energy_proxy`, `forgetting_rate`, `contradiction_rate_before`,
  `contradiction_rate_after`, `pdi_proxy`, `hardware_replay_used`,
  `inference_substrate`, `duration_s`, and `run_date`.

### SCENARIO-LEARN-2918: Verified Replay Evidence Updates Scheduler Priorities

**Given** ready Exp 2911 and Exp 2912 artifacts and prior FR-11 memory rows
**When** Exp 2918 builds the replay corpus and applies verifier-derived process
reward weights
**Then** verified code and same-basis sampler trajectories receive higher
replay priority than failed hallucination or high-energy/latency rows
**And** the artifact reports non-negative overall and energy-proxy deltas
without mutating model weights.

### SCENARIO-LEARN-2918-BLOCKED: Missing Prerequisite Fails Closed

**Given** either the Exp 2911 code-hallucination verifier or Exp 2912
same-basis CPU baseline artifact is missing or not ready
**When** Exp 2918 runs
**Then** it writes the required artifact schema with
`online_self_learning_ready=false`, `online_update_performed=false`, and an
`honest_verdict` beginning with `blocked_`.

## Implementation Status (REQ-LEARN-2918)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2918 | Implemented (`python/carnot/eval/fr11_verifiable_process_rewards_self_learning_v1.py`) | Implemented (`tests/python/test_experiment_2918_fr11_verifiable_process_rewards_self_learning_v1.py`) |

## REQ-LEARN-2933: KAN/KAC Per-Knot Structural Memory Self-Learning Probe

**Given** a deterministic synthetic stream of verified constraint examples with
train/holdout splits and `random_seed=2933`
**When** Exp 2933 compares a no-update baseline, a replay-scheduler-only
baseline, and a KAN/KAC-inspired per-knot or RBF-importance updater
**Then** the probe SHALL update bounded local structural memory only, measure
pre-update and post-update utility on new constraints, and measure forgetting on
previous constraints under a predeclared threshold.
**And** it SHALL write
`results/experiment_2933_kan_cl_per_knot_self_learning_v1.json` with
`honest_verdict`, `kan_cl_self_learning_ready`,
`continuous_self_learning_targeted=true`, `random_seed=2933`,
`dataset_manifest`, `baselines`, `kan_update_config`,
`utility_delta_vs_replay_only`, `energy_proxy_delta`, `forgetting_rate`,
`forgetting_threshold`, `non_forgetting_passed`,
`updated_knot_or_rbf_count`, `tests_run`,
`inference_substrate="local_training_simulation"`, `duration_s`, and
`run_date="20260523"`.

### REQ-LEARN-2933 Sub-requirements

- REQ-LEARN-2933-1: The dataset generator SHALL be deterministic for seed 2933
  and SHALL report train/holdout counts, constraint IDs, and RBF center count in
  `dataset_manifest`.
- REQ-LEARN-2933-2: The no-update and replay-scheduler-only baselines SHALL be
  evaluated on the same holdout splits as the KAN/KAC update path.
- REQ-LEARN-2933-3: The KAN/KAC path SHALL update per-knot or RBF-center
  importance from exact verifier labels or deterministic energy proxies and
  SHALL report the number of updated knots or centers.
- REQ-LEARN-2933-4: The headline ready flag SHALL be false when forgetting on
  previous constraints exceeds the predeclared threshold.

### SCENARIO-LEARN-2933: RBF Importance Improves New Constraints Without Forgetting

**Given** the seed-2933 synthetic constraint stream
**When** the RBF-importance structural memory processes each verified training
split and evaluates holdout utility before and after each update
**Then** its final holdout utility and energy proxy improve over the
replay-scheduler-only baseline
**And** `non_forgetting_passed=true` when the measured forgetting rate stays at
or below `forgetting_threshold`.

### SCENARIO-LEARN-2933-GUARD: Forgetting Threshold Gates The Headline

**Given** an Exp 2933 artifact payload with measured forgetting above the
predeclared threshold
**When** the artifact validator checks headline readiness
**Then** `kan_cl_self_learning_ready=false` and `non_forgetting_passed=false`.

## Implementation Status (REQ-LEARN-2933)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2933 | Implemented (`python/carnot/eval/fr11_kan_cl_per_knot_self_learning_v1.py`) | Implemented (`tests/python/test_experiment_2933_kan_cl_per_knot_self_learning_v1.py`) |

## REQ-LEARN-2947: FR-11 Continuation Replay Curriculum Scheduler

**Given** Exp 2918 provides verifier process-reward replay evidence, Exp 2933
provides structural-memory self-learning evidence, and Exp 2942 provides the
current continuation hardware boundary
**When** Exp 2947 pilots the FR-11 continuation replay scheduler
**Then** it SHALL aggregate only upstream artifact fields, not invoke a live
model or mutate model weights.
**And** it SHALL allocate replay counts through a named non-uniform curriculum
instead of flat-uniform sampling.
**And** it SHALL write
`results/experiment_2947_fr11_continuation_replay_curriculum_v1.json` with
`honest_verdict`, `inference_substrate="aggregation_from_upstream_artifacts"`,
`curriculum_schedule_used`, `replay_count_distribution`,
`cited_upstream_artifacts`, and `duration_s`.

### REQ-LEARN-2947 Sub-requirements

- REQ-LEARN-2947-1: The scheduler SHALL cite every upstream artifact it imports
  with path, SHA256, and imported field names.
- REQ-LEARN-2947-2: The replay allocation SHALL be deterministic for the same
  upstream artifacts and replay budget.
- REQ-LEARN-2947-3: The replay allocation SHALL fail closed when required
  upstream artifacts are missing, malformed, or not ready.
- REQ-LEARN-2947-4: The replay count distribution SHALL contain at least three
  curriculum buckets and SHALL not assign the same replay count to every bucket
  when the upstream evidence has non-uniform signal.

### SCENARIO-LEARN-2947: Curriculum Replaces Flat-Uniform Replay

**Given** ready Exp 2918, Exp 2933, and Exp 2942 artifacts
**When** Exp 2947 computes the continuation replay schedule
**Then** structural-memory, process-reward, continuation-boundary, and
retention buckets receive deterministic non-uniform replay counts
**And** the artifact reports that the schedule was built by upstream-artifact
aggregation.

### SCENARIO-LEARN-2947-BLOCKED: Dirty Upstream Blocks The Curriculum

**Given** a required upstream artifact is missing, malformed, or not ready
**When** Exp 2947 runs
**Then** it writes the required artifact schema with an `honest_verdict`
beginning with `blocked_` and an empty `replay_count_distribution`.

## Implementation Status (REQ-LEARN-2947)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2947 | Implemented (`python/carnot/eval/fr11_continuation_replay_curriculum_v1.py`) | Implemented (`tests/python/test_experiment_2947_fr11_continuation_replay_curriculum_v1.py`) |

## REQ-LEARN-2954: FR-11 Utility-Gated Replay Curriculum With Rollback

**Given** Exp 2947 provides the non-uniform replay pilot, Exp 2946 provides a
checked-in code-generation continuation protocol, and Exp 2940 provides
verifier-grounded code-status energy evidence
**When** Exp 2954 evaluates replay policies offline
**Then** it SHALL define deterministic train/replay, held-out utility, and
stable forgetting-guard slices from the existing artifacts.
**And** it SHALL compare flat replay, the Exp 2947 non-uniform replay pilot,
and a utility-gated replay update.
**And** it SHALL update replay weights only when verifier-grounded held-out
utility improves over the non-uniform pilot and SHALL roll back the candidate
weights whenever the stable forgetting guard degrades.
**And** it SHALL write
`results/experiment_2954_fr11_utility_gated_replay_curriculum_v2.json` with
`honest_verdict`, `continuous_self_learning_task=true`,
`self_learning_utility_artifact_ready`, `source_artifacts`,
`replay_policies_compared`, `heldout_utility_baseline`,
`heldout_utility_after`, `heldout_utility_delta`,
`self_learning_utility_positive`, `forgetting_guard_metric_before`,
`forgetting_guard_metric_after`, `forgetting_guard_passed`,
`rollback_triggered`, `update_rule`,
`inference_substrate="aggregation_from_upstream_artifacts"`, and
`duration_s`.

### REQ-LEARN-2954 Sub-requirements

- REQ-LEARN-2954-1: The evaluator SHALL cite every required upstream artifact
  it imports with path, SHA256, role, and imported field names.
- REQ-LEARN-2954-2: The train/replay, held-out utility, and forgetting-guard
  slices SHALL be deterministic for the same Exp 2946 protocol rows.
- REQ-LEARN-2954-3: Replay-policy utility SHALL be computed only from
  verifier outcomes, code-status energy, and repair taxonomy; the evaluator
  SHALL NOT invoke a live model or mutate closed-model weights.
- REQ-LEARN-2954-4: The utility-gated update SHALL retain the previous
  non-uniform weights when held-out utility does not improve or when the
  forgetting guard degrades.
- REQ-LEARN-2954-5: Soft-Radial Projection MAY be mentioned only as future
  differentiable constraint-preserving design context, not as an implemented
  update mechanism in this artifact.

### SCENARIO-LEARN-2954: Utility Gate Accepts Non-Forgetting Replay Update

**Given** the checked-in Exp 2947, Exp 2946, and Exp 2940 artifacts are present
**When** Exp 2954 builds the offline replay policy comparison
**Then** the utility-gated replay policy reports held-out utility before and
after the candidate update
**And** the candidate weights are accepted only if held-out utility improves and
the forgetting guard passes.

### SCENARIO-LEARN-2954-ROLLBACK: Forgetting Guard Rolls Back Candidate Weights

**Given** a candidate replay update improves the held-out utility metric but
reduces the stable forgetting-guard metric
**When** the utility gate evaluates the candidate
**Then** the update is rolled back to the baseline weights and
`rollback_triggered=true`.

## Implementation Status (REQ-LEARN-2954)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2954 | Implemented (`python/carnot/eval/utility_gated_replay_curriculum_v2.py`) | Implemented (`tests/python/test_experiment_2954_fr11_utility_gated_replay_curriculum_v2.py`) |

## REQ-LEARN-2969: Non-Tautological FR-11 Utility-Gated Replay Re-Evaluation

**Given** Exp 2954 reported a positive FR-11 utility-gated replay result but
was flagged for possible tautological replay or held-out leakage
**When** Exp 2969 re-evaluates the replay policy using only checked-in upstream
artifacts
**Then** it SHALL require Exp 2954, Exp 2952, Exp 2959, and matrix v12 artifacts
to exist and expose enough row-level or summary metrics to define disjoint
train, replay, held-out, and forgetting-guard slices.
**And** it SHALL write a blocked artifact naming missing sources or fields when
those preconditions fail.
**And** it SHALL compare frozen baseline, random replay, prior `.278`
utility-gated replay, and a new non-tautological utility-gated replay policy.
**And** it SHALL mark `non_tautological_self_learning_ready=true` only when
held-out utility improves over frozen and random baselines, the negative-control
update does not improve held-out utility, leakage checks pass, and stable code,
logic, and threshold-policy forgetting guards pass.

The terminal artifact SHALL be
`results/experiment_2969_fr11_non_tautological_utility_gate_v3.json` and SHALL
include `honest_verdict`, `continuous_self_learning_task=true`,
`non_tautological_self_learning_ready`, `source_artifacts`,
`split_checksums`, `leakage_check_passed`, `replay_policies_compared`,
`frozen_heldout_utility`, `random_replay_heldout_utility`,
`prior_utility_gated_heldout_utility`, `new_heldout_utility`,
`heldout_utility_delta_vs_random`, `negative_control_delta`,
`forgetting_guard_passed`, `rollback_triggered`, `update_rule`,
`model_specs_if_live_llm_used`, `inference_substrate="aggregation_from_upstream_artifacts"`,
and measured `duration_s`.

### REQ-LEARN-2969 Sub-requirements

- REQ-LEARN-2969-1: The evaluator SHALL cite every required upstream artifact
  with path, SHA256, role, required flag, presence, and imported field names.
- REQ-LEARN-2969-2: Split checksums SHALL separately hash split IDs and reward
  source IDs so the artifact can show that replay rewards are not computed from
  held-out evaluation targets.
- REQ-LEARN-2969-3: The frozen baseline, random replay baseline, prior Exp 2954
  utility-gated policy, negative-control update, and new non-tautological update
  SHALL be scored on the same held-out utility slice.
- REQ-LEARN-2969-4: The negative-control reward SHALL be intentionally
  uninformative and SHALL fail the readiness gate if it improves held-out
  utility.
- REQ-LEARN-2969-5: Forgetting guards SHALL cover stable code, logic, and
  threshold-policy slices; any guard degradation SHALL roll back the candidate
  update before reporting final readiness.
- REQ-LEARN-2969-6: The evaluator SHALL NOT invoke a live model. If a future
  live proposal path is used, it SHALL populate `model_specs_if_live_llm_used`
  with exact mandated SOTA GGUF IDs.

### SCENARIO-LEARN-2969: Non-Tautological Replay Passes Only With Disjoint Evidence

**Given** checked-in Exp 2954, Exp 2952, Exp 2959, and matrix v12 artifacts with
enough metrics to define disjoint slices
**When** Exp 2969 builds the replay comparison
**Then** it reports split checksums, passes the leakage check, scores frozen,
random, prior utility-gated, and new non-tautological policies on the held-out
slice, and sets `non_tautological_self_learning_ready=true` only when the new
policy beats frozen and random while the negative control fails to improve.

### SCENARIO-LEARN-2969-ROLLBACK: Forgetting Guard Rolls Back Candidate Update

**Given** a candidate replay update improves held-out utility but degrades any
stable code, logic, or threshold-policy forgetting guard
**When** Exp 2969 evaluates the candidate
**Then** the update is rolled back, `forgetting_guard_passed=false`,
`rollback_triggered=true`, and readiness remains false.

### SCENARIO-LEARN-2969-BLOCKED: Missing Slice Evidence Fails Closed

**Given** a required upstream artifact or required split-defining metric is
missing
**When** Exp 2969 runs
**Then** it writes the required artifact schema with `honest_verdict` beginning
with `blocked_`, `non_tautological_self_learning_ready=false`,
`leakage_check_passed=false`, and `missing_fields` naming the absent evidence.

## Implementation Status (REQ-LEARN-2969)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2969 | Implemented (`python/carnot/eval/fr11_non_tautological_utility_gate_v3.py`) | Implemented (`tests/python/test_experiment_2969_fr11_non_tautological_utility_gate_v3.py`) |

## REQ-LEARN-2970: KAN Constraint-Memory Forgetting Guard Audit

**Given** Exp 2969 reports `non_tautological_self_learning_ready=true` and the
local KAN/per-knot self-learning helpers are importable
**When** Exp 2970 audits KAN-style constraint memory on a bounded deterministic
fixture
**Then** it SHALL compare frozen, eager update, per-knot importance update, and
adapter-style update policies on the same current-domain and held-out old-domain
slices.
**And** it SHALL compute old-domain forgetting deltas by policy and select a
policy only when current-domain utility improves without exceeding the declared
forgetting threshold.
**And** it SHALL record RM/BOP/NABS-style hardware-cost fields when they are
derivable from checked-in KAN complexity artifacts.
**And** it SHALL mark high-dimensional extrapolation out of scope, make no
synthesis claim, make no analog acceleration claim, and use
`inference_substrate="deterministic_wiring"`.

The terminal artifact SHALL be
`results/experiment_2970_kan_forgetting_guard_memory_audit_v1.json` and SHALL
include `honest_verdict`, `kan_forgetting_guard_ready`, `source_artifacts`,
`policies_compared`, `current_domain_utility`, `old_domain_utility`,
`forgetting_delta_by_policy`, `selected_policy`,
`high_dimensional_claim_allowed=false`, `hardware_cost_fields`,
`no_synthesis_claim=true`, `no_analog_claim=true`, `files_changed`,
`inference_substrate="deterministic_wiring"`, and measured `duration_s`.

### REQ-LEARN-2970 Sub-requirements

- REQ-LEARN-2970-1: The audit SHALL fail closed with a blocked artifact when
  Exp 2969 is missing, malformed, or not ready.
- REQ-LEARN-2970-2: The policy comparison SHALL use the same deterministic
  constraint-memory fixture for every policy and SHALL evaluate current-domain
  and held-out old-domain utility separately.
- REQ-LEARN-2970-3: The eager policy SHALL expose any old-domain utility
  regression as a positive forgetting delta rather than hiding it in aggregate
  utility.
- REQ-LEARN-2970-4: The per-knot and adapter-style policies SHALL preserve
  old-domain guard slices before they can be selected.
- REQ-LEARN-2970-5: Hardware-cost fields SHALL be copied or derived from local
  RM/BOP/NABS accounting artifacts only; missing cost evidence SHALL not create
  any synthesis, timing-closure, FPGA, or analog KAN claim.

### SCENARIO-LEARN-2970: Per-Knot Memory Learns Current Constraints Without Forgetting

**Given** the seed-2933 KAN constraint-memory fixture split into old and current
constraint domains
**When** the four policy updates are evaluated
**Then** frozen memory retains old-domain utility but does not learn the current
domain, eager updating improves current-domain utility while reporting a
positive forgetting delta, and the selected per-knot or adapter-style policy
improves current-domain utility while keeping old-domain forgetting within the
threshold.

### SCENARIO-LEARN-2970-BLOCKED: Missing Readiness Evidence Fails Closed

**Given** Exp 2969 is absent or `non_tautological_self_learning_ready` is not
true
**When** Exp 2970 runs
**Then** it writes the required schema with `kan_forgetting_guard_ready=false`,
an `honest_verdict` beginning with `blocked_`, no selected policy, and
`high_dimensional_claim_allowed=false`.

## Implementation Status (REQ-LEARN-2970)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2970 | Implemented (`python/carnot/eval/fr11_kan_forgetting_guard_memory_audit_v1.py`) | Implemented (`tests/python/test_experiment_2970_kan_forgetting_guard_memory_audit_v1.py`) |

## REQ-LEARN-2982: FR-11 Independent-Metric Utility Gate Re-Run

**Given** Exp 2969 reported FR-11 replay readiness but matrix v13 still flagged
the result because the independent held-out metric evidence was weak
**When** Exp 2982 re-runs the continuous self-learning gate from checked-in
artifacts
**Then** it SHALL identify Exp 2969's update-selection metric separately from
the reported held-out improvement metrics.
**And** it SHALL evaluate at least three independent metrics such as pass@1,
solver-verified accuracy, schema or syntax failure, and verifier
false-accept rate.
**And** it SHALL compare frozen baseline, random replay, previous FR-11 replay,
and a new independent-metric replay policy under deterministic reset controls.
**And** it SHALL require a negative control that does not improve, a passing
Exp 2969 forgetting guard, and bounded Exp 2970 KAN memory evidence before it
can set `fr11_independent_self_learning_ready=true`.
**And** it SHALL use
`inference_substrate="aggregation_and_deterministic_replay"` without invoking a
live model or claiming KAN acceleration.

The terminal artifact SHALL be
`results/experiment_2982_fr11_independent_metric_utility_gate_v4.json` and
SHALL include `honest_verdict`, `continuous_self_learning_task=true`,
`fr11_independent_metrics_evaluated`,
`fr11_independent_self_learning_ready`, `update_selection_metric`,
`independent_metrics`, `frozen_baseline_metrics`, `random_replay_metrics`,
`prior_fr11_metrics`, `new_replay_metrics`,
`heldout_independent_delta_vs_random`, `negative_control_delta`,
`forgetting_guard_passed`, `leakage_audit`, `no_identical_metric_flag`,
`inference_substrate="aggregation_and_deterministic_replay"`, and measured
`duration_s`.

### REQ-LEARN-2982 Sub-requirements

- REQ-LEARN-2982-1: The evaluator SHALL fail closed when Exp 2969 readiness or
  Exp 2970 bounded forgetting-guard evidence is missing.
- REQ-LEARN-2982-2: `update_selection_metric` SHALL name the Exp 2969 utility
  used for update selection and SHALL NOT be identical to any reported
  independent held-out metric name.
- REQ-LEARN-2982-3: Held-out independent deltas SHALL be directional: higher is
  better for accuracy metrics and lower is better for failure-rate metrics.
- REQ-LEARN-2982-4: The negative control SHALL be deterministic, uninformative,
  and SHALL fail readiness if it improves over random replay on any independent
  held-out metric.
- REQ-LEARN-2982-5: KAN evidence from Exp 2970 MAY be used only as bounded
  memory-forgetting evidence and SHALL NOT be reported as KAN acceleration,
  synthesis, or analog evidence.

### SCENARIO-LEARN-2982: Independent Metrics Pass With Guarded Replay

**Given** Exp 2969 and Exp 2970 artifacts are present and ready
**When** Exp 2982 scores frozen, random, prior FR-11, and independent-metric
replay policies
**Then** it reports independent held-out deltas versus random replay
**And** sets `fr11_independent_self_learning_ready=true` only when the new
policy improves the independent metrics, the negative control does not improve,
and forgetting guards pass.

### SCENARIO-LEARN-2982-BLOCKED: Missing Preconditions Fail Closed

**Given** Exp 2969 or Exp 2970 readiness evidence is absent or malformed
**When** Exp 2982 runs
**Then** it writes the required artifact schema with an `honest_verdict`
beginning with `blocked_`, `fr11_independent_metrics_evaluated=false`, and
`fr11_independent_self_learning_ready=false`.

## Implementation Status (REQ-LEARN-2982)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2982 | Implemented (`python/carnot/eval/fr11_independent_metric_utility_gate_v4.py`) | Implemented (`tests/python/test_experiment_2982_fr11_independent_metric_utility_gate_v4.py`) |

## REQ-LEARN-2983: Trace-To-Skill Repair Memory Pilot

**Given** Exp 2976 reports `trace_execution_plan_ready=true` and checked-in
repair or partial-monitor artifacts contain row-level trace evidence
**When** Exp 2983 runs the continuous self-learning trace-to-skill memory pilot
**Then** it SHALL convert failed repair traces and deterministic partial-monitor
events into candidate skill memories with a `trace_to_skill_memory` schema.
**And** every memory SHALL include `failure_signature`, `verifier_feedback`,
`minimal_fix_pattern`, `applicability_conditions`, and
`forbidden_label_leakage`.
**And** it SHALL evaluate frozen/no-memory, random-memory, and trace-memory
replay on held-out repair or formalization tasks whose IDs are disjoint from
the memory extraction sources.
**And** it SHALL keep the update selector distinct from the reported held-out
utility metric, include a negative control, and set `leakage_flag=false` only
when no memory contains held-out labels, expected answers, pass vectors, or
held-out task IDs.
**And** it SHALL use
`inference_substrate="artifact_replay_and_optional_live_llm"` without reporting
a headline result unless fresh live inference used at least one mandated SOTA
GGUF model.

The terminal artifact SHALL be
`results/experiment_2983_trace_to_skill_repair_memory_pilot_v1.json` and SHALL
include `honest_verdict`, `continuous_self_learning_task=true`,
`trace_to_skill_memory_ready`, `headline_result`, `pilot_source`,
`models_used`, `mandatory_headline_model_ids`, `memory_schema`,
`extracted_memory_count`, `heldout_task_count`, `no_memory_metrics`,
`random_memory_metrics`, `trace_memory_metrics`,
`heldout_skill_reuse_delta`, `leakage_flag`, `negative_control_delta`,
`inference_substrate="artifact_replay_and_optional_live_llm"`, and measured
`duration_s`.

### REQ-LEARN-2983 Sub-requirements

- REQ-LEARN-2983-1: The evaluator SHALL fail closed when Exp 2976 is missing
  or `trace_execution_plan_ready` is not true.
- REQ-LEARN-2983-2: The evaluator SHALL prefer Exp 2977 repair traces when the
  artifact exists; if Exp 2977 is absent, it SHALL use Exp 2964 repair traces
  and set `pilot_source=".279"`.
- REQ-LEARN-2983-3: Candidate memories SHALL be extracted only from failed
  repair rows or partial-monitor event schemas and SHALL NOT copy pass/fail
  labels, held-out task IDs, or exact expected answers into the memory body.
- REQ-LEARN-2983-4: Held-out evaluation SHALL compare no-memory,
  random-memory, and trace-memory conditions with a deterministic negative
  control and SHALL report a positive `heldout_skill_reuse_delta` only when
  trace-memory replay improves over random-memory replay.
- REQ-LEARN-2983-5: `trace_to_skill_memory_ready` SHALL be true only when the
  schema is usable, `heldout_skill_reuse_delta > 0`, the negative control does
  not improve, and `leakage_flag=false`.

### SCENARIO-LEARN-2983: Trace Memories Improve Held-Out Replay Without Leakage

**Given** Exp 2976 is ready, Exp 2977 or Exp 2964 provides failed repair rows,
and Exp 2968 provides deterministic partial-monitor events
**When** Exp 2983 builds candidate skill memories and evaluates held-out replay
**Then** the artifact reports a usable memory schema, non-zero extracted memory
count, disjoint held-out task count, no label leakage, non-improving negative
control, positive trace-memory delta over random-memory replay, and
`trace_to_skill_memory_ready=true`.

### SCENARIO-LEARN-2983-BLOCKED: Missing Protocol Readiness Fails Closed

**Given** Exp 2976 is missing or does not report
`trace_execution_plan_ready=true`
**When** Exp 2983 runs
**Then** it writes the required artifact schema with `trace_to_skill_memory_ready=false`,
`heldout_task_count=0`, `extracted_memory_count=0`, `leakage_flag=true`, and an
`honest_verdict` beginning with `blocked_`.

## Implementation Status (REQ-LEARN-2983)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2983 | Implemented (`python/carnot/eval/trace_to_skill_repair_memory_pilot_v1.py`) | Implemented (`tests/python/test_experiment_2983_trace_to_skill_repair_memory_pilot_v1.py`) |

## REQ-LEARN-2995: FR-11 Verifier-Grounded Trace Memory v2

**Given** Exp 2982 preserves the independent FR-11 metric boundary, Exp 2983
is only a trace-memory pilot, Exp 2992 records fresh solver-backed
formalization transcripts, and Exp 2994 records deterministic validator-tree
successes and failures
**When** Exp 2995 builds verifier-grounded trace memory v2
**Then** it SHALL extract candidate memories from Exp 2992 solver transcripts
and Exp 2994 validator-tree feedback only when the candidate includes
process-verifiable evidence from exact checks.
**And** it SHALL use a bounded update or memory-selection rule that selects
memories by utility metrics such as process verification score and trace
evidence density, not by the reported independent held-out metrics.
**And** it SHALL evaluate held-out pass@1, solver-verified accuracy,
syntax/schema failure, verifier false accepts, and forgetting with the
independent metric direction rules established by Exp 2982.
**And** it SHALL include deterministic negative controls where trace memory
selection is disabled or made uninformative, and SHALL refuse promotion if the
selection utility is identical to the independent metrics or the controls
improve equally.

The terminal artifact SHALL be
`results/experiment_2995_fr11_verifier_grounded_trace_memory_v2.json` and
SHALL include `independent_self_learning_boundary_preserved`,
`continuous_self_learning_task=true`, `trace_memory_ready`,
`n_trace_memories`, `independent_metric_names`, `utility_metric_names`,
`no_identical_metric_flag`, `negative_control_deltas`,
`forgetting_guard_passed`, `heldout_metric_deltas`, and `honest_verdict`.

### REQ-LEARN-2995 Sub-requirements

- REQ-LEARN-2995-1: The evaluator SHALL fail closed when Exp 2982, Exp 2992,
  or Exp 2994 readiness evidence is absent or malformed.
- REQ-LEARN-2995-2: Candidate memories SHALL include explicit exact-check
  evidence from Z3, runtime JSON parsing, Python AST parsing, or deterministic
  validator-tree feedback, and SHALL NOT copy held-out labels or answers into
  memory bodies.
- REQ-LEARN-2995-3: The bounded selection rule SHALL be disableable for a
  random/control condition and SHALL select by utility metrics that are
  disjoint from held-out metric names.
- REQ-LEARN-2995-4: Held-out deltas SHALL use Exp 2982 directional semantics:
  higher is better for pass@1 and solver-verified accuracy, while lower is
  better for syntax/schema failure and verifier false accepts.
- REQ-LEARN-2995-5: `trace_memory_ready` SHALL be true only when at least one
  verifier-grounded memory is selected, independent held-out deltas improve,
  negative controls do not improve equally, the forgetting guard passes, and
  `no_identical_metric_flag=true`.

### SCENARIO-LEARN-2995: Verifier-Grounded Memories Improve Held-Out Verification

**Given** ready Exp 2982, reproduced Exp 2992 solver provenance, and ready Exp
2994 validator-tree feedback
**When** Exp 2995 extracts and selects bounded trace memories
**Then** it writes the required artifact fields, reports non-zero
`n_trace_memories`, preserves distinct utility and independent metric names,
reports positive held-out directional deltas, reports non-improving negative
controls, passes the forgetting guard, and sets `trace_memory_ready=true`.

### SCENARIO-LEARN-2995-BLOCKED: Missing Verifier Evidence Fails Closed

**Given** any required upstream verifier artifact is missing or not ready
**When** Exp 2995 runs
**Then** it writes the required artifact fields with `trace_memory_ready=false`,
`n_trace_memories=0`, empty held-out deltas, non-ready forgetting evidence, and
an `honest_verdict` beginning with `blocked_`.

## Implementation Status (REQ-LEARN-2995)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-2995 | Implemented (`python/carnot/eval/fr11_verifier_grounded_trace_memory_v2.py`) | Implemented (`tests/python/test_experiment_2995_fr11_verifier_grounded_trace_memory_v2.py`) |

## REQ-LEARN-3007: FR-11 Attractor-Inspired Trace Memory Stability

**Given** Exp 2995 made verifier-grounded trace memory ready in a narrow
machine-gated sense, Exp 3005 produced an exact-checked validator-tree corpus,
and Exp 3006 produced a fixed-point energy diagnostic without making a native
EqR claim
**When** Exp 3007 stress-tests FR-11 trace memory across repeated replay/update
cycles
**Then** it SHALL build a trace-memory candidate set from the Exp 2995 selected
memories, Exp 3005 exact validator rows, and Exp 3006 diagnostic rows.
**And** it SHALL accept memory updates only when independent exact verifier
evidence and held-out verifier-task performance support the update.
**And** it SHALL evaluate convergence and bounded drift across repeated
replay/update cycles without treating self-reported memory utility as success.
**And** it SHALL include negative controls for irrelevant traces, contradicted
constraints, and shuffled validator labels.
**And** it SHALL include forgetting checks on held-out verifier tasks that
should not degrade after accepted memory updates.
**And** it SHALL state that the diagnostic is attractor-inspired only and does
not claim a native local attractor model.

The terminal artifact SHALL be
`results/experiment_3007_fr11_attractor_trace_memory_stability_v1.json` and
SHALL include `trace_memory_stability_ready`,
`continuous_self_learning_task`,
`independent_self_learning_boundary_preserved`, `n_memory_candidates`,
`convergence_guard_passed`, `drift_guard_passed`,
`negative_control_rejected`, `forgetting_guard_passed`, `heldout_delta`,
`native_attractor_model_claim_made`, and `honest_verdict`.

### REQ-LEARN-3007 Sub-requirements

- REQ-LEARN-3007-1: The evaluator SHALL fail closed when Exp 2995, Exp 3005, or
  Exp 3006 readiness evidence is absent or malformed.
- REQ-LEARN-3007-2: Candidate memories SHALL record their source experiment,
  exact verifier authorities, verifier label/signature, and whether the
  evidence is machine-checked.  Candidate extraction SHALL exclude LLM-only
  judgments.
- REQ-LEARN-3007-3: Update acceptance SHALL be controlled by exact verifier
  evidence and held-out verifier-task score.  Self-reported memory utility
  fields SHALL be recorded as non-authoritative and SHALL NOT be counted as the
  promotion success metric.
- REQ-LEARN-3007-4: The convergence guard SHALL require repeated replay/update
  cycles to stabilize their accepted memory IDs and held-out score.
- REQ-LEARN-3007-5: The drift guard SHALL reject unrelated signatures,
  contradicted constraints, and validator labels that no longer match the exact
  validator evidence.
- REQ-LEARN-3007-6: The forgetting guard SHALL compare held-out verifier-task
  score before and after accepted memory updates and SHALL fail if the update
  degrades the retained held-out task score.
- REQ-LEARN-3007-7: `trace_memory_stability_ready` SHALL be true only when the
  independent self-learning boundary is preserved, memory candidates exist,
  convergence and drift guards pass, negative controls are rejected, the
  forgetting guard passes, `heldout_delta > 0`, and
  `native_attractor_model_claim_made=false`.

### SCENARIO-LEARN-3007: Stable Trace Memory Attractor Diagnostic Passes

**Given** ready Exp 2995, Exp 3005, and Exp 3006 artifacts
**When** Exp 3007 builds exact-verifier memory candidates and runs repeated
replay/update cycles
**Then** the accepted memory set converges, drift remains bounded, negative
controls are rejected, held-out verifier-task score improves, forgetting score
does not degrade, and `trace_memory_stability_ready=true`.

### SCENARIO-LEARN-3007-BLOCKED: Missing Stability Evidence Fails Closed

**Given** any required upstream artifact is missing or not ready
**When** Exp 3007 runs
**Then** it writes the required artifact fields with
`trace_memory_stability_ready=false`, `n_memory_candidates=0`,
`heldout_delta=0.0`, failed guards, and an `honest_verdict` beginning with
`blocked_`.

## Implementation Status (REQ-LEARN-3007)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3007 | Implemented (`python/carnot/eval/fr11_attractor_trace_memory_stability_v1.py`) | Implemented (`tests/python/test_experiment_3007_fr11_attractor_trace_memory_stability_v1.py`) |

## REQ-LEARN-3020: DVI Verifier-Feedback Self-Learning Controller

**Given** Exp 3017 supplies exact instruction validator trees, Exp 3018
supplies BEAVER-style cached frontier certificate rows, Exp 3019 supplies a
feasibility-channel de-tautology diagnostic, and Exp 3007 supplies a stable
FR-11 trace-memory forgetting guard
**When** Exp 3020 builds a bounded Draft-Verify-Improve-inspired controller
over cached traces
**Then** it SHALL update only transparent controller weights or routing
decisions from exact verifier accept/reject events.
**And** each accepted update SHALL improve an independent held-out exact
validator utility metric while preserving the forgetting guard.
**And** it SHALL compare the learned controller against no-learning,
replay-only, random-update, and negative-control conditions.
**And** it SHALL write an inspectable controller config, replay transcript, and
terminal artifact without claiming native LLM fine-tuning or self-graded
promotion.

The terminal artifact SHALL be
`results/experiment_3020_dvi_verifier_feedback_self_learning_controller_v1.json`
and SHALL include `verifier_feedback_controller_ready`,
`continuous_self_learning_task=true`,
`independent_self_learning_boundary_preserved`,
`controller_config_path`, `n_replay_items`, `heldout_delta`,
`negative_control_delta`, `forgetting_guard_passed`, `drift_guard_passed`,
`tautology_risk_flag`, `native_llm_training_claim_made`, and
`honest_verdict`.

### REQ-LEARN-3020 Sub-requirements

- REQ-LEARN-3020-1: The evaluator SHALL fail closed when any source artifact
  or manifest/table required from Exp 3017, Exp 3018, Exp 3019, or Exp 3007 is
  missing, malformed, or not ready.
- REQ-LEARN-3020-2: Replay items SHALL be built from exact validator/certificate
  evidence and SHALL record their source experiment, item ID, partition, exact
  verifier feedback, machine-check status, and non-label feature vector.
- REQ-LEARN-3020-3: The controller SHALL update only bounded inspectable
  weights; it SHALL not mutate model weights and SHALL set
  `native_llm_training_claim_made=false`.
- REQ-LEARN-3020-4: A proposed update SHALL be accepted only when exact
  verifier feedback is machine checked, independent held-out utility improves,
  the forgetting score does not degrade, and drift controls reject unrelated or
  contradicted evidence.
- REQ-LEARN-3020-5: `verifier_feedback_controller_ready` SHALL be true only
  when `heldout_delta > 0`, `negative_control_delta <= 0`,
  `forgetting_guard_passed=true`, `drift_guard_passed=true`,
  `tautology_risk_flag=false`, `native_llm_training_claim_made=false`, and the
  independent self-learning boundary is preserved.

### SCENARIO-LEARN-3020: Cached Verifier Feedback Improves Held-Out Utility

**Given** ready Exp 3017, Exp 3018, Exp 3019, and Exp 3007 evidence
**When** Exp 3020 replays exact accept/reject events through the bounded
controller
**Then** the controller writes its config and replay transcript, reports a
positive held-out delta over no-learning/replay-only/random controls, reports
non-improving negative controls, preserves forgetting and drift guards, makes
no native LLM training claim, and sets
`verifier_feedback_controller_ready=true`.

### SCENARIO-LEARN-3020-BLOCKED: Missing Source Evidence Fails Closed

**Given** any required source evidence is missing, malformed, or not ready
**When** Exp 3020 runs
**Then** it writes the required artifact fields with
`verifier_feedback_controller_ready=false`, `n_replay_items=0`,
`heldout_delta=0.0`, failed guards, `native_llm_training_claim_made=false`, and
an `honest_verdict` beginning with `blocked_`.

## Implementation Status (REQ-LEARN-3020)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3020 | Implemented (`python/carnot/eval/dvi_verifier_feedback_self_learning_controller_v1.py`) | Implemented (`tests/python/test_experiment_3020_dvi_verifier_feedback_self_learning_controller_v1.py`) |

## REQ-LEARN-3032: FR-11 Held-Out DVI Replay Must Reject Tautological Gains

**Given** Exp 3019 reports a feasibility-channel de-tautology diagnostic with
known tautology risk and Exp 3020 reports a bounded DVI verifier-feedback
controller over cached exact traces
**When** Exp 3032 replays the learned controller on held-out exact verifier
traces that were not used for controller updates
**Then** it SHALL measure held-out feasible/infeasible separation, false
positive rate, false negative rate, and tautology exposure without live LLM
inference.
**And** the proposer/controller-update role SHALL receive only the verifier
feedback history or the persisted bounded controller weights.
**And** the checker role SHALL receive only held-out exact claim features,
expected exact authorities, and expected verifier labels, not the controller's
original rationale or training transcript rows as grading evidence.
**And** at least one negative control SHALL shuffle verifier feedback labels or
withhold exact authority features.
**And** it SHALL not claim model-weight learning unless model weights are
actually trained.

The terminal artifact SHALL be
`results/experiment_3032_fr11_heldout_dvi_replay_v2.json` and SHALL include
`fr11_heldout_replay_ready`, `continuous_self_learning_tested`,
`heldout_trace_count`, `feasible_infeasible_auc_delta`,
`shuffled_feedback_delta`, `false_positive_delta`, `false_negative_delta`,
`tautology_risk_cleared`, `information_asymmetry_enforced`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3032 Sub-requirements

- REQ-LEARN-3032-1: The evaluator SHALL fail closed when Exp 3019 or Exp 3020
  artifacts are missing, malformed, or not terminal, or when their diagnostic
  table, controller config, or replay transcript cannot be loaded.
- REQ-LEARN-3032-2: Held-out traces SHALL be exact candidate-frontier rows from
  the Exp 3019 held-out partition and SHALL exclude row IDs that appear in the
  Exp 3020 controller update transcript.
- REQ-LEARN-3032-3: Checker inputs SHALL include exact claim IDs, non-label
  controller features, expected exact authorities, and expected feasible versus
  infeasible labels; they SHALL exclude `candidate_role`, `certificate_status`,
  `heldout_success_label`, original controller rationale text, and update
  acceptance decisions.
- REQ-LEARN-3032-4: The replay SHALL compute AUC improvement over a no-learning
  baseline, shuffled-feedback delta, false-positive delta, false-negative
  delta, and an inspectable tautology exposure report.
- REQ-LEARN-3032-5: `fr11_heldout_replay_ready` SHALL be true only when the
  held-out replay improves separation, does not improve under shuffled feedback
  or withheld-authority controls, does not increase false positives or false
  negatives, enforces information asymmetry, clears tautology risk, and makes no
  live-inference or model-weight-learning claim.

### SCENARIO-LEARN-3032: Held-Out Replay Clears FR-11 DVI Tautology Risk

**Given** terminal Exp 3019 and Exp 3020 artifacts plus their inspectable tables
and controller files
**When** Exp 3032 evaluates the bounded controller on held-out exact traces
excluded from the update transcript
**Then** it writes the required artifact fields, reports a positive
`feasible_infeasible_auc_delta`, reports non-positive
`shuffled_feedback_delta`, does not increase false positives or false
negatives, enforces information asymmetry, and sets
`fr11_heldout_replay_ready=true` only when `tautology_risk_cleared=true`.

### SCENARIO-LEARN-3032-BOUNDED: Held-Out Replay May Complete Without Promotion

**Given** terminal Exp 3019 and Exp 3020 artifacts but the held-out replay still
exposes tautological scoring or a negative control improvement
**When** Exp 3032 evaluates the bounded controller
**Then** it SHALL still write a complete terminal artifact with
`fr11_heldout_replay_ready=false`, the measured deltas, and an
`honest_verdict` beginning with `complete_`, because the preconditions were not
blocked even though FR-11 promotion was not justified.

### SCENARIO-LEARN-3032-BLOCKED: Missing Replay Evidence Fails Closed

**Given** missing or malformed Exp 3019 or Exp 3020 evidence
**When** Exp 3032 runs
**Then** it writes the required artifact fields with
`fr11_heldout_replay_ready=false`, `heldout_trace_count=0`, zeroed deltas,
`tautology_risk_cleared=false`, `information_asymmetry_enforced=false`, and an
`honest_verdict` beginning with `blocked_`.

## Implementation Status (REQ-LEARN-3032)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3032 | Implemented (`python/carnot/eval/fr11_heldout_dvi_replay_v2.py`) | Implemented (`tests/python/test_experiment_3032_fr11_heldout_dvi_replay_v2.py`) |

## REQ-LEARN-3033: FR-11 Nonforgetting And Negative-Control Stress

**Given** Exp 3032 reports held-out DVI replay readiness and Exp 3020 provides
the bounded verifier-feedback controller state, replay transcript, and exact
prior feedback traces
**When** Exp 3033 applies the held-out feedback as a cached controller update
**Then** it SHALL measure whether prior exact traces are retained under a
declared tolerance while held-out behavior improves.
**And** it SHALL run shuffled-feedback, adversarially irrelevant feedback, and
no-feedback replay controls so gains cannot be attributed to label leakage or
replay noise.
**And** it SHALL report whether any KAN-style locality probe is actually
available before making locality claims; if no such probe is present, it SHALL
set `kan_locality_probe_available=false`.
**And** it SHALL not claim live LLM inference or model-weight training because
this is a cached-feedback controller stress.

The terminal artifact SHALL be
`results/experiment_3033_fr11_nonforgetting_negative_control_stress_v1.json`
and SHALL include `fr11_nonforgetting_stress_ready`,
`fr11_self_learning_promotable`, `prior_retention_delta`,
`heldout_delta_after_update`, `shuffled_control_delta`,
`no_feedback_delta`, `drift_failures`, `kan_locality_probe_available`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3033 Sub-requirements

- REQ-LEARN-3033-1: The evaluator SHALL fail closed when Exp 3032, Exp 3020,
  the held-out replay rows, controller config, or replay transcript are missing
  or malformed.
- REQ-LEARN-3033-2: Prior exact traces SHALL be drawn from machine-checked
  Exp 3020 controller replay rows and held-out traces SHALL be drawn from the
  Exp 3032 held-out replay rows.
- REQ-LEARN-3033-3: The held-out feedback update SHALL mutate only bounded
  inspectable controller weights and SHALL report
  `inference_substrate.live_llm_inference=false` and
  `inference_substrate.model_weight_training=false`.
- REQ-LEARN-3033-4: The stress SHALL compute prior retention delta,
  held-out delta after update, shuffled-feedback control delta,
  adversarially irrelevant control delta, no-feedback delta, and visible
  `drift_failures`.
- REQ-LEARN-3033-5: `fr11_self_learning_promotable` SHALL be true only when
  `heldout_delta_after_update > 0`, shuffled and irrelevant controls do not
  improve, no-feedback replay has zero delta, and `prior_retention_delta`
  remains within the declared retention tolerance.

### SCENARIO-LEARN-3033: Held-Out Feedback Improves Without Forgetting

**Given** terminal Exp 3032 and Exp 3020 controller artifacts plus replay rows
**When** Exp 3033 applies the held-out feedback update to the bounded
controller
**Then** it writes the required artifact fields, reports positive held-out
delta, reports prior retention within tolerance, rejects shuffled and
irrelevant controls, reports zero no-feedback delta, records no drift failures,
and sets `fr11_nonforgetting_stress_ready=true`.

### SCENARIO-LEARN-3033-BOUNDED: Stress Completes Without Promotion

**Given** the source artifacts are present but the stress exposes forgetting,
drift, control improvement, or no positive held-out update effect
**When** Exp 3033 runs
**Then** it SHALL still write a complete terminal artifact with measured
deltas, `fr11_nonforgetting_stress_ready=true`,
`fr11_self_learning_promotable=false`, visible `drift_failures`, and an
`honest_verdict` beginning with `complete_`.

### SCENARIO-LEARN-3033-BLOCKED: Missing Stress Evidence Fails Closed

**Given** missing or malformed Exp 3032 or Exp 3020 evidence
**When** Exp 3033 runs
**Then** it writes the required artifact fields with
`fr11_nonforgetting_stress_ready=false`, `fr11_self_learning_promotable=false`,
zeroed deltas, `drift_failures` containing the blocker, and an
`honest_verdict` beginning with `blocked_`.

## Implementation Status (REQ-LEARN-3033)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3033 | Implemented (`python/carnot/eval/fr11_nonforgetting_negative_control_stress_v1.py`) | Implemented (`tests/python/test_experiment_3033_fr11_nonforgetting_negative_control_stress_v1.py`) |

## REQ-LEARN-3045: FR-11 Governed Self-Learning Claim Boundary

**Given** Exp 3032 reports held-out DVI replay evidence and Exp 3033 reports
nonforgetting negative-control stress evidence for a bounded FR-11 controller
update
**When** Exp 3045 prepares the `.285` governed self-learning boundary artifact
**Then** it SHALL summarize the controller-only evidence, separate
controller-side edit rights from model-weight learning, predeclare the metrics
that Exp 3046 must measure, and predeclare automatic non-promotion criteria.
**And** it SHALL mark model weights out of scope unless a later experiment
actually trains model weights.
**And** it SHALL not claim live LLM inference, native model retraining, or broad
continuous self-learning beyond the cached controller/trace governance boundary.

The terminal artifact SHALL be
`results/experiment_3045_fr11_governed_self_learning_boundary_v1.json` and
SHALL include `fr11_governance_ready`, `allowed_edit_targets`,
`forbidden_claims`, `required_metrics`, `non_promotion_criteria`,
`prior_evidence_summary`, `continuous_self_learning_scope`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3045 Sub-requirements

- REQ-LEARN-3045-1: The evaluator SHALL fail closed when Exp 3032 or Exp 3033
  artifacts are missing, malformed, or not terminal.
- REQ-LEARN-3045-2: `prior_evidence_summary` SHALL identify Exp 3032 as
  held-out cached exact-trace replay and Exp 3033 as controller-only
  nonforgetting negative-control stress, including the measured deltas and
  their limits.
- REQ-LEARN-3045-3: `allowed_edit_targets` SHALL enumerate controller weights,
  trace memory, validator thresholds, KAN/locality anchors, and model weights,
  with model weights marked out of scope unless an experiment actually trains
  them.
- REQ-LEARN-3045-4: `required_metrics` SHALL require Exp 3046 to measure
  family holdout delta, prior retention delta, no-feedback delta,
  shuffled-control delta, contradiction graph size/rate, rollback count,
  delayed-regression delta, and source-trace completeness.
- REQ-LEARN-3045-5: `non_promotion_criteria` SHALL automatically deny
  promotion for tautology, self-confirming labels, family leakage, or missing
  negative controls.
- REQ-LEARN-3045-6: `fr11_governance_ready` SHALL be true only when the
  artifact contains all required edit-target, metric, forbidden-claim,
  non-promotion, source-trace, and inference-substrate declarations needed for
  a gated Exp 3046 self-learning experiment.

### SCENARIO-LEARN-3045: Governed Boundary Is Complete Enough For Exp 3046

**Given** terminal Exp 3032 and Exp 3033 artifacts that report cached
controller-only evidence with rejected negative controls
**When** Exp 3045 builds the governed boundary artifact
**Then** it writes all required artifact fields, sets
`fr11_governance_ready=true`, marks model weights out of scope, lists the Exp
3046 metrics, lists the automatic non-promotion criteria, reports no live model
inference in `inference_substrate`, and emits an `honest_verdict` beginning
with `complete_`.

### SCENARIO-LEARN-3045-BLOCKED: Missing Governance Evidence Fails Closed

**Given** missing or malformed Exp 3032 or Exp 3033 evidence
**When** Exp 3045 runs
**Then** it writes the required artifact fields with
`fr11_governance_ready=false`, empty or blocked evidence summaries, no model
weight-learning claim, and an `honest_verdict` beginning with `blocked_`.

## Implementation Status (REQ-LEARN-3045)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3045 | Implemented (`python/carnot/eval/fr11_governed_self_learning_boundary_v1.py`) | Implemented (`tests/python/test_experiment_3045_fr11_governed_self_learning_boundary_v1.py`) |

## REQ-LEARN-3046: FR-11 Solver-Feedback Self-Learning Loop

**Given** Exp 3044 exposes exact SMT/SAT correction-set feedback and Exp 3045
exposes governed controller-only self-learning edit rights
**When** Exp 3046 runs the `.285` governed controller-side self-learning loop
**Then** it SHALL verify both source artifacts before any update, build a small
SAT/SMT-style family split containing train/update cases, related held-out
cases, prior exact cases, no-feedback controls, and shuffled-feedback controls,
and apply only governance-approved controller-side updates.
**And** it SHALL not run live LLM inference, not mutate base model weights, and
not promote shuffled or no-feedback control behavior as learning.
**And** it SHALL compute family held-out improvement, prior retention,
no-feedback delta, shuffled-control delta, contradiction-rate delta, rollback
count, delayed-regression delta, and source-trace coverage.

The terminal artifact SHALL be
`results/experiment_3046_fr11_solver_feedback_self_learning_loop_v1.json` and
SHALL include `fr11_solver_feedback_ready`,
`continuous_self_learning_task`, `promotion_decision`, `edit_targets_used`,
`family_holdout_delta`, `prior_retention_delta`, `no_feedback_delta`,
`shuffled_control_delta`, `contradiction_rate_delta`, `rollback_count`,
`delayed_regression_delta`, `source_trace_counts`, `inference_substrate`, and
`honest_verdict`.

### REQ-LEARN-3046 Sub-requirements

- REQ-LEARN-3046-1: The evaluator SHALL fail closed with
  `honest_verdict="blocked_missing_governance_or_exact_feedback"` when Exp 3044
  or Exp 3045 is missing, malformed, not ready, lacks correction sets, claims
  live LLM inference, or claims model-weight training or mutation.
- REQ-LEARN-3046-2: The family split SHALL contain disjoint train/update and
  held-out case identifiers, at least one prior exact case, at least one
  no-feedback control case, and at least one shuffled-feedback control case.
- REQ-LEARN-3046-3: The update SHALL consume Exp 3044 correction-set feedback
  and mutate only governed controller-side targets such as `controller_weights`
  or `trace_memory`; it SHALL record `edit_targets_used` and source traces.
- REQ-LEARN-3046-4: The loop SHALL compute
  `family_holdout_delta`, `prior_retention_delta`, `no_feedback_delta`,
  `shuffled_control_delta`, `contradiction_rate_delta`, `rollback_count`, and
  `delayed_regression_delta` from deterministic before/after controller
  evaluations.
- REQ-LEARN-3046-5: `fr11_solver_feedback_ready` SHALL be true only when the
  governed loop completes with non-vacuous controls, no train/held-out leakage,
  positive held-out improvement, no prior or delayed regression, zero
  no-feedback improvement, no shuffled-control improvement, reduced
  contradiction rate, source-trace coverage, no live LLM inference, and no
  model-weight mutation.

### SCENARIO-LEARN-3046: Exact Solver Feedback Improves Held-Out Family

**Given** ready Exp 3044 and Exp 3045 artifacts
**When** Exp 3046 runs
**Then** it writes the required artifact fields, records controller-only edit
targets, reports positive `family_holdout_delta`, zero
`prior_retention_delta`, zero `no_feedback_delta`, non-positive
`shuffled_control_delta`, negative `contradiction_rate_delta`, a visible
rollback count, non-negative `delayed_regression_delta`, non-empty source trace
counts, `fr11_solver_feedback_ready=true`, and an `honest_verdict` beginning
with `complete_`.

### SCENARIO-LEARN-3046-BLOCKED: Missing Governance Or Exact Feedback Fails Closed

**Given** missing or malformed Exp 3044 or Exp 3045 source evidence
**When** Exp 3046 runs
**Then** it writes the required artifact fields with
`fr11_solver_feedback_ready=false`, zeroed metric deltas, empty edit targets,
source trace counts showing the missing evidence, no live LLM inference, no
model-weight mutation, and
`honest_verdict="blocked_missing_governance_or_exact_feedback"`.

## Implementation Status (REQ-LEARN-3046)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3046 | Implemented (`python/carnot/eval/fr11_solver_feedback_self_learning_loop_v1.py`) | Implemented (`tests/python/test_experiment_3046_fr11_solver_feedback_self_learning_loop_v1.py`) |

## REQ-LEARN-3047: KAN-Style Locality Probe For Solver-Feedback Controller Updates

**Given** Exp 3046 has produced a governed controller-only solver-feedback
self-learning loop with a machine-readable family split and source-traced
feedback updates
**When** Exp 3047 probes the smallest existing locality structure
**Then** it SHALL reuse Exp 3046's family split and feedback traces, compare
the baseline controller weights against the governed updated controller
weights, and report which controller anchors/weights changed versus remained
anchored.
**And** it SHALL compute a numeric locality metric, changed-anchor count,
anchored-prior count, related held-out delta, prior-retention delta,
irrelevant-control delta, and a comparator summary from a shuffled-control or
explicitly unavailable non-local comparator.
**And** it SHALL keep the claim bounded to controller/locality evidence, not
base-model self-learning, KAN model-weight training, live LLM inference, or
general KAN continual learning.

The terminal artifact SHALL be
`results/experiment_3047_kan_locality_nonforgetting_probe_v2.json` and SHALL
include `kan_locality_probe_ready`, `locality_metric`,
`changed_anchor_count`, `anchored_prior_count`, `heldout_delta`,
`prior_retention_delta`, `irrelevant_control_delta`, `comparator_summary`,
`promotion_decision`, `inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3047 Sub-requirements

- REQ-LEARN-3047-1: The probe SHALL fail closed when Exp 3046 is missing,
  malformed, not ready, claims live LLM inference, or claims model-weight
  training or mutation.
- REQ-LEARN-3047-2: The probe SHALL rebuild Exp 3046's family split and
  governed controller update from Exp 3046's cited source artifacts, rather
  than inventing a new split.
- REQ-LEARN-3047-3: `changed_anchor_count` SHALL count controller anchor/weight
  keys whose post-update value differs from the baseline, while
  `anchored_prior_count` SHALL count prior exact anchors that remain unchanged
  and correctly retained.
- REQ-LEARN-3047-4: `locality_metric` SHALL be numeric and non-vacuous, derived
  from the fraction of controller anchors that remain unchanged after the
  governed update.
- REQ-LEARN-3047-5: `kan_locality_probe_ready` SHALL be true only when locality
  and nonforgetting fields are present and non-vacuous, related held-out
  performance improves, prior retention does not regress, irrelevant controls
  are stable, a comparator is either measured or explicitly unavailable, and no
  model-weight or live-inference claim is made.

### SCENARIO-LEARN-3047: Governed Controller Update Is Local And Retains Priors

**Given** ready Exp 3046, Exp 3044, and Exp 3045 artifacts
**When** Exp 3047 runs
**Then** it writes the required artifact fields, records positive
`heldout_delta`, non-negative `prior_retention_delta`, stable
`irrelevant_control_delta`, positive `changed_anchor_count`, positive
`anchored_prior_count`, a numeric `locality_metric`, a measured comparator
summary, `kan_locality_probe_ready=true`, and an `honest_verdict` beginning
with `complete_`.

### SCENARIO-LEARN-3047-BLOCKED: Missing Solver-Feedback Evidence Fails Closed

**Given** missing or malformed Exp 3046 source evidence
**When** Exp 3047 runs
**Then** it writes the required artifact fields with
`kan_locality_probe_ready=false`, zeroed locality and nonforgetting metrics, an
explicit unavailable comparator summary, no live LLM inference, no model-weight
mutation, and `honest_verdict="blocked_missing_solver_feedback_locality_source"`.

## Implementation Status (REQ-LEARN-3047)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3047 | Implemented (`python/carnot/eval/fr11_kan_locality_nonforgetting_probe_v2.py`) | Implemented (`tests/python/test_experiment_3047_kan_locality_nonforgetting_probe_v2.py`) |

## REQ-LEARN-3060: Solver Self-Model Trace Schema For Governed FR-11 Learning

**Given** Exp 3045 defines the governed controller-only edit boundary, Exp 3046
reports a solver-feedback controller update, and Exp 3047 reports locality and
nonforgetting evidence over that update
**When** Exp 3060 defines the solver self-model trace schema for Exp 3061
**Then** it SHALL load and summarize the current controller-only evidence from
Exp 3045, Exp 3046, and Exp 3047, including any adversarial flags or claim
limits.
**And** it SHALL define machine-readable trace fields for solver prompt/input,
exact constraint family, correction set, contradiction graph update, controller
edit, rollback decision, delayed-regression window, and source artifact.
**And** it SHALL define controller-side edit rights, forbidden claims, validation
rules, delayed-regression requirements, source artifacts, and an inference
substrate that explicitly excludes live model inference and model-weight
learning.

The terminal artifact SHALL be
`results/experiment_3060_fr11_solver_self_model_trace_schema_v1.json` and SHALL
include `solver_self_model_trace_ready`, `trace_schema`,
`allowed_edit_targets`, `forbidden_claims`, `validation_rules`,
`delayed_regression_window`, `source_artifacts`,
`continuous_self_learning_scope`, `inference_substrate`, and
`honest_verdict`.

### REQ-LEARN-3060 Sub-requirements

- REQ-LEARN-3060-1: The evaluator SHALL fail closed when Exp 3045, Exp 3046, or
  Exp 3047 is missing, malformed, non-terminal, not ready, claims live LLM
  inference, or claims model-weight training or mutation.
- REQ-LEARN-3060-2: `trace_schema` SHALL be directly consumable by Exp 3061 and
  SHALL require `solver_prompt_input`, `exact_constraint_family`,
  `correction_set`, `contradiction_graph_update`, `controller_edit`,
  `rollback_decision`, `delayed_regression_window`, and `source_artifact`.
- REQ-LEARN-3060-3: `allowed_edit_targets` SHALL include only controller-side
  targets such as controller weights, trace memory, contradiction graph,
  rollback policy, validation thresholds, and delayed-regression manifest.
- REQ-LEARN-3060-4: `forbidden_claims` SHALL explicitly forbid model-weight
  learning, live LLM inference, native solver/model retraining, and broad
  autonomous self-learning beyond controller-side protocol work.
- REQ-LEARN-3060-5: `validation_rules` SHALL include automatic detectors named
  `self_confirming_labels`, `family_leakage`, `missing_exact_authority`, and
  `missing_delayed_regression_evaluation`.
- REQ-LEARN-3060-6: `solver_self_model_trace_ready` SHALL be true only when all
  source artifacts are ready, the schema has every required trace field, the
  validation rules cover the required failure modes, delayed regression is
  mandatory and measurable, no allowed edit target mutates model weights, and
  the inference substrate declares no live model inference or model-weight
  learning.

### SCENARIO-LEARN-3060: Schema Is Ready For Exp 3061 Consumption

**Given** ready Exp 3045, Exp 3046, and Exp 3047 artifacts
**When** Exp 3060 runs
**Then** it writes all required artifact fields, summarizes controller-only
evidence and limits, marks `solver_self_model_trace_ready=true`, exposes every
required trace field in `trace_schema`, predeclares all required validation
rules, makes delayed-regression evaluation mandatory, excludes model weights
from allowed edit targets, reports no live model inference in
`inference_substrate`, and emits an `honest_verdict` beginning with
`complete_`.

### SCENARIO-LEARN-3060-BLOCKED: Missing Source Evidence Fails Closed

**Given** missing or malformed Exp 3045, Exp 3046, or Exp 3047 source evidence
**When** Exp 3060 runs
**Then** it writes the required artifact fields with
`solver_self_model_trace_ready=false`, an explicit blocked reason, no model
weight-learning claim, no live inference claim, and an `honest_verdict`
beginning with `blocked_`.

## Implementation Status (REQ-LEARN-3060)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3060 | Implemented (`python/carnot/eval/fr11_solver_self_model_trace_schema_v1.py`) | Implemented (`tests/python/test_experiment_3060_fr11_solver_self_model_trace_schema_v1.py`) |

## REQ-LEARN-3061: Delayed-Regression Solver Self-Model Pilot

**Given** Exp 3060 exposes a directly consumable solver self-model trace schema
and Exp 3058 preserves an exact SMT fallback
**When** Exp 3061 runs the mandatory continuous self-learning pilot
**Then** it SHALL verify both source artifacts before any update, build a small
exact SAT/SMT-style split containing train/update cases, related held-out
cases, prior cases, delayed-regression cases, no-feedback controls, and
shuffled-feedback controls, and apply only governance-approved controller-side
or trace-memory updates.
**And** it SHALL store one self-model trace per attempted update using the Exp
3060 trace schema, including exact correction set, controller edit, rollback
decision, delayed-regression window, and source artifact provenance.
**And** it SHALL evaluate family holdout, prior retention, no-feedback control,
shuffled-feedback control, contradiction rate, rollback burden, and delayed
regression deltas without live LLM inference or base model-weight mutation.

The terminal artifact SHALL be
`results/experiment_3061_fr11_delayed_regression_solver_self_model_pilot_v1.json`
and SHALL include `fr11_delayed_regression_ready`,
`continuous_self_learning_task`, `promotion_decision`, `edit_targets_used`,
`self_model_trace_count`, `family_holdout_delta`, `prior_retention_delta`,
`no_feedback_delta`, `shuffled_control_delta`, `contradiction_rate_delta`,
`rollback_count`, `delayed_regression_delta`, `source_trace_counts`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3061 Sub-requirements

- REQ-LEARN-3061-1: The evaluator SHALL fail closed with
  `honest_verdict="blocked_missing_trace_schema_or_formal_fallback"` when Exp
  3060 or Exp 3058 is missing, malformed, non-terminal, not ready, lacks the
  required trace or exact-fallback fields, claims model-weight training or
  mutation, or lacks an exact solver/fallback authority.
- REQ-LEARN-3061-2: The split SHALL contain disjoint train/update, related
  held-out, prior, delayed-regression, no-feedback control, and
  shuffled-feedback control identifiers; both control groups SHALL be
  non-vacuous.
- REQ-LEARN-3061-3: Every attempted update SHALL produce exactly one
  self-model trace row whose fields match the Exp 3060 `trace_schema`, whose
  `controller_edit.target` is an allowed controller-side target, whose
  `controller_edit.model_weight_mutation` is false, whose `correction_set`
  names an independent exact authority, whose `rollback_decision` records
  whether the update was applied or rolled back, and whose `source_artifact`
  points to Exp 3058 or Exp 3060 evidence.
- REQ-LEARN-3061-4: The pilot SHALL compute
  `family_holdout_delta`, `prior_retention_delta`, `no_feedback_delta`,
  `shuffled_control_delta`, `contradiction_rate_delta`, `rollback_count`, and
  `delayed_regression_delta` from deterministic before/after controller
  evaluations.
- REQ-LEARN-3061-5: `fr11_delayed_regression_ready` SHALL be true only when the
  trace schema is populated, self-model trace count is positive, source trace
  counts are positive, controls are non-vacuous, family holdout improves,
  prior and delayed regression do not regress, no-feedback and shuffled
  controls do not improve, contradiction rate decreases, no live LLM inference
  or model-weight mutation is claimed, and the honest verdict uses a terminal
  success prefix.

### SCENARIO-LEARN-3061: Self-Model Traces Support Delayed Regression Evaluation

**Given** ready Exp 3060 and Exp 3058 artifacts
**When** Exp 3061 runs
**Then** it writes the required artifact fields, stores one self-model trace per
update, records controller-only edit targets, reports positive
`family_holdout_delta`, non-negative `prior_retention_delta`, non-positive
`no_feedback_delta`, non-positive `shuffled_control_delta`, negative
`contradiction_rate_delta`, visible rollback burden, non-negative
`delayed_regression_delta`, non-empty source trace counts,
`fr11_delayed_regression_ready=true`, and an `honest_verdict` beginning with
`complete_`.

### SCENARIO-LEARN-3061-BLOCKED: Missing Trace Schema Or Formal Fallback Fails Closed

**Given** missing or malformed Exp 3060 or Exp 3058 source evidence
**When** Exp 3061 runs
**Then** it writes the required artifact fields with
`fr11_delayed_regression_ready=false`, zeroed metric deltas, empty edit targets
and self-model traces, no live LLM inference claim, no model-weight mutation,
and `honest_verdict="blocked_missing_trace_schema_or_formal_fallback"`.

## Implementation Status (REQ-LEARN-3061)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3061 | Implemented (`python/carnot/eval/fr11_delayed_regression_solver_self_model_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3061_fr11_delayed_regression_solver_self_model_pilot_v1.py`) |

## REQ-LEARN-3062: KAN/PWA Locality Verification Audit

**Given** Exp 3047 reports controller-side KAN-style locality/nonforgetting
evidence and Exp 3061 reports a delayed-regression solver self-model pilot
**When** Exp 3062 audits whether that evidence can support a stronger KAN
verification claim
**Then** it SHALL load both source artifacts, identify the smallest locality
object available for bounding, and report numeric `locality_bound`,
`prior_retention_bound`, and `approximation_error_bound` fields.
**And** it SHALL set `kan_pwa_verification_ready=true` only when an exact or
bounded PWA/MILP-style check is actually available for the audited locality
object.
**And** it SHALL keep `promotion_decision="controller_locality_evidence_only"`
when the available evidence is limited to controller anchors or process traces
rather than trained KAN model weights.

The terminal artifact SHALL be
`results/experiment_3062_kan_pwa_locality_verification_audit_v1.json` and
SHALL include `kan_pwa_verification_ready`, `locality_bound`,
`approximation_error_bound`, `prior_retention_bound`, `verification_path`,
`tests_or_checks_run`, `promotion_decision`, `spec_updates`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3062 Sub-requirements

- REQ-LEARN-3062-1: The audit SHALL fail closed when Exp 3047 or Exp 3061 is
  missing, malformed, non-terminal, not ready, or claims live inference or
  model-weight training/mutation.
- REQ-LEARN-3062-2: `locality_bound` SHALL be derived from Exp 3047's numeric
  locality evidence, and `prior_retention_bound` SHALL be the tighter
  nonnegative prior-retention evidence shared by Exp 3047 and Exp 3061.
- REQ-LEARN-3062-3: `approximation_error_bound` SHALL be finite and explicit;
  it MAY be `0.0` only when the audit path is an exact controller-anchor audit
  rather than a PWA abstraction of trained KAN model weights.
- REQ-LEARN-3062-4: `verification_path` SHALL distinguish
  `exact_controller_anchor_audit` from a true PWA/MILP KAN model-weight path.
- REQ-LEARN-3062-5: `promotion_decision` SHALL NOT promote controller-only
  evidence into a model-weight learning or general KAN verification claim.

### SCENARIO-LEARN-3062: Controller Locality Evidence Is Bounded But Not Promoted

**Given** ready Exp 3047 and Exp 3061 artifacts
**When** Exp 3062 runs
**Then** it writes all required artifact fields, records positive numeric
`locality_bound`, nonnegative `prior_retention_bound`, finite
`approximation_error_bound`, explicit `verification_path`, no live LLM
inference, no model-weight mutation, `promotion_decision` bounded to
controller-locality evidence, `kan_pwa_verification_ready=false`, and an
`honest_verdict` beginning with `complete_`.

### SCENARIO-LEARN-3062-BLOCKED: Missing Locality Or Delayed-Regression Evidence Fails Closed

**Given** missing or malformed Exp 3047 or Exp 3061 source evidence
**When** Exp 3062 runs
**Then** it writes the required artifact fields with zeroed bounds,
`kan_pwa_verification_ready=false`, an explicit blocked reason, no live LLM
inference, no model-weight mutation, and
`honest_verdict="blocked_missing_kan_locality_or_delayed_regression_source"`.

## Implementation Status (REQ-LEARN-3062)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3062 | Implemented (`python/carnot/eval/fr11_kan_pwa_locality_verification_audit_v1.py`) | Implemented (`tests/python/test_experiment_3062_kan_pwa_locality_verification_audit_v1.py`) |

## REQ-LEARN-3076: FR-11 Online Soundness/Completeness Mistake-Budget Protocol

**Given** Exp 3046 reports a controller-only solver-feedback loop, Exp 3060
defines a solver self-model trace schema, and Exp 3061 reports a delayed-
regression pilot with controls
**When** Exp 3076 defines the FR-11 online controller-side learning budget
protocol for Exp 3077
**Then** it SHALL load and classify the prior FR-11 artifacts as
controller-only, adversarially flagged, or out of scope.
**And** it SHALL define soundness mistakes, completeness mistakes,
delayed-regression mistakes, contradiction mistakes, and rollback triggers.
**And** it SHALL define numeric tiny-pilot budget fields for maximum allowed
soundness mistakes, maximum delayed regressions, no-feedback control,
shuffled-feedback control, and prior-retention floor.
**And** it SHALL forbid claims of model-weight self-learning, autonomous
production self-modification, native KAN integration, native EBT integration,
live model inference, or production promotion.

The terminal artifact SHALL be
`results/experiment_3076_fr11_online_soundness_completeness_budget_v1.json`
and SHALL include `soundness_completeness_budget_ready`,
`soundness_mistake_definition`, `completeness_mistake_definition`,
`delayed_regression_window`, `mistake_budget`, `required_controls`,
`forbidden_claims`, `source_artifacts`, `continuous_self_learning_task`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3076 Sub-requirements

- REQ-LEARN-3076-1: The protocol SHALL fail closed when Exp 3046, Exp 3060, or
  Exp 3061 is missing, malformed, non-terminal, not ready, or claims live model
  inference, model-weight training, or model-weight mutation.
- REQ-LEARN-3076-2: `soundness_mistake_definition` SHALL define unsafe accepts
  as controller accept decisions where an independent exact authority rejects
  the candidate, and SHALL include count key, rate key, severity, measurement
  source, and rollback trigger fields.
- REQ-LEARN-3076-3: `completeness_mistake_definition` SHALL define unsafe
  rejects/abstentions as controller reject or abstain decisions where an
  independent exact authority accepts the candidate, and SHALL include count
  key, rate key, severity, measurement source, and rollback trigger fields.
- REQ-LEARN-3076-4: The protocol SHALL define delayed-regression,
  contradiction, and rollback-trigger accounting in machine-readable form.
- REQ-LEARN-3076-5: `mistake_budget` SHALL include numeric fields named
  `max_soundness_mistakes`, `max_delayed_regressions`,
  `no_feedback_max_delta`, `shuffled_feedback_max_delta`, and
  `prior_retention_floor`.
- REQ-LEARN-3076-6: `required_controls` SHALL require both no-feedback and
  shuffled-feedback controls, and SHALL require the learning condition to beat
  both controls before any promotion.
- REQ-LEARN-3076-7: `soundness_completeness_budget_ready` SHALL be true only
  when all required definitions, controls, source traces, forbidden claims, and
  no-live-inference/no-model-weight boundaries are directly consumable by Exp
  3077.

### SCENARIO-LEARN-3076: Mistake Budget Is Ready For Exp 3077

**Given** ready Exp 3046, Exp 3060, and Exp 3061 artifacts
**When** Exp 3076 runs
**Then** it writes all required artifact fields, classifies Exp 3046 and Exp
3061 as adversarially flagged but controller-only, marks model-weight learning
and native KAN/EBT integration out of scope, defines soundness and completeness
mistake accounting, defines numeric budget gates, requires no-feedback and
shuffled-feedback controls, reports no live model inference in
`inference_substrate`, sets `soundness_completeness_budget_ready=true`, and
emits an `honest_verdict` beginning with `complete_`.

### SCENARIO-LEARN-3076-BLOCKED: Missing Prior Evidence Fails Closed

**Given** missing or malformed Exp 3046, Exp 3060, or Exp 3061 source evidence
**When** Exp 3076 runs
**Then** it writes the required artifact fields with
`soundness_completeness_budget_ready=false`, explicit blocked source
classification, no live model inference claim, no model-weight mutation claim,
and `honest_verdict="blocked_missing_soundness_completeness_budget_sources"`.

## Implementation Status (REQ-LEARN-3076)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3076 | Implemented (`python/carnot/eval/fr11_online_soundness_completeness_budget_v1.py`) | Implemented (`tests/python/test_experiment_3076_fr11_online_soundness_completeness_budget_v1.py`) |

## REQ-LEARN-3077: FR-11 Soundness-Bounded Online Self-Learning Pilot

**Given** Exp 3076 exposes a complete FR-11 online soundness/completeness
mistake budget for a tiny controller-side pilot
**When** Exp 3077 runs a governed online self-learning update
**Then** it SHALL verify the Exp 3076 budget before any update, run a tiny
exact SAT/SMT-style family split with update cases, related holdout cases,
prior cases, delayed-regression cases, no-feedback controls, and shuffled-
feedback controls, and write
`results/experiment_3077_fr11_soundness_bounded_online_self_learning_pilot_v1.json`.
**And** it SHALL update only governance-approved controller state or trace
memory, never base model weights.
**And** it SHALL store Exp 3060-shaped self-model traces for online decisions
and label decisions as correct, soundness mistake, completeness mistake,
abstention, rollback, or delayed regression.
**And** it SHALL measure family holdout, prior retention, no-feedback control,
shuffled-feedback control, contradiction-rate, rollback burden, delayed-
regression, soundness-mistake, and completeness-mistake deltas.
**And** it SHALL refuse stronger promotion whenever any Exp 3076 budget gate is
exceeded or controls are vacuous.

The terminal artifact SHALL include `fr11_soundness_bounded_ready`,
`continuous_self_learning_task`, `promotion_decision`, `edit_targets_used`,
`self_model_trace_count`, `soundness_mistakes`, `completeness_mistakes`,
`mistake_budget_delta`, `family_holdout_delta`, `prior_retention_delta`,
`no_feedback_delta`, `shuffled_control_delta`,
`contradiction_rate_delta`, `rollback_count`, `delayed_regression_delta`,
`source_trace_counts`, `inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3077 Sub-requirements

- REQ-LEARN-3077-1: The pilot SHALL fail closed with
  `honest_verdict="blocked_missing_soundness_budget"` when Exp 3076 is missing,
  malformed, non-terminal, not ready, missing a numeric mistake budget, missing
  no-feedback or shuffled-feedback controls, or claims live model inference,
  model-weight training, or model-weight mutation.
- REQ-LEARN-3077-2: The split SHALL contain non-empty disjoint update, related
  holdout, prior, delayed-regression, no-feedback control, and shuffled-feedback
  control partitions, all labeled by independent exact authority.
- REQ-LEARN-3077-3: Every online decision SHALL produce an Exp 3060-shaped
  self-model trace with `controller_edit.model_weight_mutation=false` and a
  decision label in the set `{correct, soundness_mistake,
  completeness_mistake, abstention, rollback, delayed_regression}`.
- REQ-LEARN-3077-4: The artifact SHALL count `soundness_mistakes` as exact-
  rejected cases accepted by the controller and `completeness_mistakes` as
  exact-accepted cases rejected or abstained by the controller.
- REQ-LEARN-3077-5: `mistake_budget_delta` SHALL compare observed mistakes,
  delayed regressions, controls, and prior retention against the Exp 3076
  budget gates in machine-readable form.
- REQ-LEARN-3077-6: `promotion_decision` SHALL remain controller-only and SHALL
  refuse stronger promotion unless all budget gates pass and no-feedback plus
  shuffled-feedback controls are non-vacuous.
- REQ-LEARN-3077-7: `fr11_soundness_bounded_ready` SHALL be true for a complete
  bounded pilot result even when stronger promotion is refused, and false only
  for precondition-blocked artifacts.

### SCENARIO-LEARN-3077: Budgeted Pilot Refuses Stronger Promotion

**Given** a complete Exp 3076 budget with zero allowed soundness and
completeness mistakes
**When** Exp 3077 applies a tiny exact controller update that improves related
holdout behavior but abstains on one exact-valid online case
**Then** it records zero soundness mistakes, one completeness mistake, a
positive family-holdout delta, non-vacuous no-feedback and shuffled-feedback
controls, one rolled-back shuffled-control candidate, no model-weight update,
`fr11_soundness_bounded_ready=true`, and a bounded controller-only
`promotion_decision` that refuses stronger promotion because the completeness
budget is exceeded.

### SCENARIO-LEARN-3077-BLOCKED: Missing Soundness Budget Fails Closed

**Given** missing or malformed Exp 3076 source evidence
**When** Exp 3077 runs
**Then** it writes the required artifact fields with zeroed metrics,
`fr11_soundness_bounded_ready=false`, no live model inference claim, no
model-weight mutation claim, and
`honest_verdict="blocked_missing_soundness_budget"`.

## Implementation Status (REQ-LEARN-3077)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3077 | Implemented (`python/carnot/eval/fr11_soundness_bounded_online_self_learning_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3077_fr11_soundness_bounded_online_self_learning_pilot_v1.py`) |

## REQ-LEARN-3090: FR-11 ReSyn KAN-CL Completeness Repair Pilot

**Given** Exp 3084 has produced a ready ReSyn exact fixture bank with
deterministic SMT, arithmetic, JSON, and Python-runtime labels
**When** Exp 3090 runs a bounded FR-11 online self-learning pilot
**Then** it SHALL fail closed with
`honest_verdict="blocked_fixture_precondition_failed"` unless the Exp 3084
artifact exists, reports `resyn_fixture_bank_ready=true`, the manifest exists,
all loaded fixture rows expose exact labels, and delayed-regression rows have
exact labels available.
**And** it SHALL update only bounded controller-side state, never model weights.
**And** it SHALL use concrete KAN-CL-inspired retention anchors, such as
per-family knots plus constraint-local basis weights, to retain prior exact
negative capability while learning from online fixture feedback.
**And** it SHALL run baseline, online update, no-feedback control,
shuffled-feedback control, perturbation-family holdout, prior-retention, and
delayed-regression checks.
**And** it SHALL count soundness mistakes, completeness mistakes, family-holdout
delta, prior-retention delta, no-feedback control delta, shuffled-feedback
control delta, KAN-CL anchor count, rollback count, delayed-regression delta,
and contradiction-rate delta.
**And** it SHALL keep promotion bounded to controller-only claims unless exact
soundness and completeness budgets both pass, and SHALL always declare no
model-weight mutation.

The terminal artifact SHALL be
`results/experiment_3090_fr11_resyn_kancl_completeness_repair_v1.json` and
SHALL include `fr11_resyn_kancl_ready`, `continuous_self_learning_task`,
`promotion_decision`, `soundness_mistakes`, `completeness_mistakes`,
`family_holdout_delta`, `prior_retention_delta`,
`no_feedback_control_delta`, `shuffled_feedback_control_delta`,
`kancl_anchor_count`, `rollback_count`, `delayed_regression_delta`,
`preconditions_checked`, `inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3090 Sub-requirements

- REQ-LEARN-3090-1: The pilot SHALL check Exp 3084 readiness and exact delayed-
  regression labels before any controller update, and SHALL write a blocked
  artifact rather than running a partial pilot when fixture preconditions fail.
- REQ-LEARN-3090-2: The online update policy SHALL mutate only bounded
  controller anchors/weights and trace memory, SHALL cap anchor weights, and
  SHALL rollback shuffled or unsafe candidate updates that introduce exact
  soundness mistakes or fail control gates.
- REQ-LEARN-3090-3: The KAN-CL-inspired anchor report SHALL include a positive
  `kancl_anchor_count` and concrete per-anchor fields for family knots and
  constraint-local basis weights.
- REQ-LEARN-3090-4: The artifact SHALL report online soundness mistakes as
  exact-rejected fixtures accepted by the controller and completeness mistakes
  as exact-accepted fixtures rejected or abstained by the online controller.
- REQ-LEARN-3090-5: The controls SHALL be non-vacuous: the no-feedback and
  shuffled-feedback control deltas must not explain the online holdout gain,
  and the shuffled candidate SHALL be reversible through an explicit rollback
  count when it violates the policy.
- REQ-LEARN-3090-6: `fr11_resyn_kancl_ready` SHALL be true only for a complete
  controller-only pilot with exact fixture preconditions checked, no live model
  inference, no model-weight training, no model-weight mutation, positive
  holdout improvement, nonnegative prior retention, zero online soundness
  mistakes, and zero online completeness mistakes.

### SCENARIO-LEARN-3090: KAN-CL Anchors Repair Completeness Without Soundness Regression

**Given** the ready Exp 3084 fixture bank
**When** Exp 3090 applies bounded online exact feedback to controller anchors
**Then** the online controller records zero soundness mistakes, zero completeness
mistakes, a positive perturbation-family holdout delta, nonnegative prior-
retention delta, no-feedback and shuffled-feedback controls at zero or worse
delta, at least one concrete KAN-CL anchor, at least one rolled-back shuffled
candidate, no model-weight mutation, `fr11_resyn_kancl_ready=true`, and an
`honest_verdict` beginning with `complete_`.

### SCENARIO-LEARN-3090-BLOCKED: Missing Fixture Preconditions Fail Closed

**Given** missing, malformed, or not-ready Exp 3084 fixture evidence
**When** Exp 3090 runs
**Then** it writes the required artifact fields with zeroed metrics,
`fr11_resyn_kancl_ready=false`, no live model inference claim, no model-weight
mutation claim, explicit failed precondition diagnostics, and
`honest_verdict="blocked_fixture_precondition_failed"`.

## Implementation Status (REQ-LEARN-3090)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3090 | Implemented (`python/carnot/eval/fr11_resyn_kancl_completeness_repair_v1.py`) | Implemented (`tests/python/test_experiment_3090_fr11_resyn_kancl_completeness_repair_v1.py`) |

## REQ-LEARN-3103: FR-11 ReSyn/KAN-CL Stress Promotion Boundary

**Given** Exp 3090 recovered FR-11 under a controller-only ReSyn/KAN-CL
budget and Exp 3097 produced a ready exact fixture evaluation protocol
**When** Exp 3103 stress-tests that controller-only loop
**Then** it SHALL load the Exp 3090 controller evidence and the Exp 3097
protocol manifest before any stress update.
**And** it SHALL define harder family holdouts, prior-retention cases,
shuffled-label negative controls, no-feedback controls, and delayed-regression
probes.
**And** it SHALL replay only bounded controller-side updates and SHALL NOT
mutate base model weights, KAN model weights, or invoke live model inference.
**And** it SHALL measure soundness mistakes, completeness mistakes,
family-holdout delta, prior-retention delta, delayed-regression delta,
rollback count, and negative-control outcomes.
**And** it SHALL set `promotion_decision` to one of `controller_only`,
`blocked`, or `broader_promotion_candidate` based only on measured stress
boundaries.

The terminal artifact SHALL be
`results/experiment_3103_fr11_resyn_kancl_stress_promotion_boundary_v2.json`
and SHALL include `fr11_stress_ready`, `continuous_self_learning_task`,
`promotion_decision`, `soundness_mistakes`, `completeness_mistakes`,
`family_holdout_delta`, `prior_retention_delta`, `delayed_regression_delta`,
`rollback_count`, `negative_control_results`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3103 Sub-requirements

- REQ-LEARN-3103-1: The stress runner SHALL fail closed with
  `promotion_decision="blocked"` when the Exp 3090 prior artifact, Exp 3097
  protocol artifact, or Exp 3097 stratified manifest is missing, malformed, or
  not ready.
- REQ-LEARN-3103-2: The stress split SHALL contain non-empty train-update,
  harder family-holdout, prior-retention, delayed-regression, no-feedback, and
  shuffled-label negative-control partitions.
- REQ-LEARN-3103-3: `negative_control_results` SHALL report no-feedback and
  shuffled-label outcomes, including case counts, deltas, mistake counts, and
  rollback status for unsafe shuffled candidates.
- REQ-LEARN-3103-4: `soundness_mistakes` SHALL count stress rows whose exact
  protocol target is reject but the controller accepts; `completeness_mistakes`
  SHALL count stress rows whose exact protocol target is accept but the
  controller rejects or abstains.
- REQ-LEARN-3103-5: `prior_retention_delta` SHALL compare the stressed
  candidate against the Exp 3090 prior controller target semantics so protocol
  gains cannot silently erase prior controller behavior.
- REQ-LEARN-3103-6: `promotion_decision` SHALL be `blocked` whenever the
  stressed candidate has nonzero soundness mistakes, nonzero completeness
  mistakes, negative prior retention, positive negative-control delta, or any
  forbidden model-weight mutation claim.

### SCENARIO-LEARN-3103: Stress Boundary Blocks Broader FR-11 Promotion

**Given** the ready Exp 3090 artifact and ready Exp 3097 exact protocol
**When** Exp 3103 replays the controller-only update on protocol feedback
**Then** it writes a complete terminal artifact with
`fr11_stress_ready=true`, `continuous_self_learning_task=true`, measured
stress deltas, visible rollback behavior, `promotion_decision="blocked"` when
the stress budget fails, no base model weight mutation, and an `honest_verdict`
beginning with `complete_`.

### SCENARIO-LEARN-3103-BLOCKED: Missing Stress Preconditions Fail Closed

**Given** missing Exp 3090 or Exp 3097 source evidence
**When** Exp 3103 runs
**Then** it writes the required artifact fields with zeroed metrics,
`fr11_stress_ready=false`, `promotion_decision="blocked"`, empty negative
controls, no model-weight mutation claim, explicit failed precondition
diagnostics, and `honest_verdict="blocked_precondition_failed"`.

## Implementation Status (REQ-LEARN-3103)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3103 | Planned (`python/carnot/eval/fr11_resyn_kancl_stress_promotion_boundary_v2.py`) | Planned (`tests/python/test_experiment_3103_fr11_resyn_kancl_stress_promotion_boundary_v2.py`) |

---

## REQ-LEARN-3116: FR-11 Unsolvable/Hard-Family Curriculum Retention Guard

**Given** Exp 3103 blocked broader FR-11 promotion because the controller-only
stress replay retained zero soundness mistakes but still had completeness
mistakes and negative retention
**When** Exp 3116 builds a controller-only curriculum over the hard or zero-pass
families identified from Exp 3103 and the Exp 3097 exact fixture protocol
**Then** it SHALL generate only abstract, solver-derived hints from checked
certificate evidence and SHALL NOT reveal final answers, mutate base model
weights, mutate KAN model weights, or invoke live model inference.
**And** it SHALL apply only controller-memory updates with rollback gates over
no-feedback, shuffled-hint, stale-hint, and contradictory-hint controls when
those controls are feasible from the available hard-family evidence.
**And** it SHALL measure soundness mistakes, completeness mistakes,
family-holdout delta, prior-retention delta, delayed-regression delta, rollback
count, and hint usefulness against the Exp 3103 promotion-boundary baseline.
**And** it SHALL mechanically promote only when soundness mistakes are zero,
prior retention is nonnegative, completeness mistakes are reduced versus Exp
3103, delayed regression is nonnegative, and negative controls fail safely.

The terminal artifact SHALL be
`results/experiment_3116_fr11_unsolvable_curriculum_retention_guard_v1.json`
and SHALL include `fr11_unsolvable_curriculum_ready`,
`continuous_self_learning_task`, `controller_only`, `no_weight_update_claim`,
`model_specs`, `hard_family_count`, `unsolvable_detection_summary`,
`hint_policy_summary`, `soundness_mistakes`, `completeness_mistakes`,
`prior_retention_delta`, `delayed_regression_delta`, `rollback_count`,
`promotion_decision`, `negative_controls`, `tests_run`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3116 Sub-requirements

- REQ-LEARN-3116-1: The runner SHALL fail closed with
  `promotion_decision="blocked"` when Exp 3090, Exp 3097, Exp 3103, or the
  Exp 3097 stratified manifest is missing, malformed, or not ready.
- REQ-LEARN-3116-2: Hard-family detection SHALL be auditable from prior stress
  completeness mistakes, prior-retention regressions, exact fixture rows, or
  generated controller-only curriculum rows, and SHALL report the hard-family
  count and zero-pass family count.
- REQ-LEARN-3116-3: Hints SHALL be abstract and solver-derived: each hint SHALL
  cite only certificate class, task axis, family, perturbation type, authority,
  and target context, and SHALL record that no final answer or model-weight
  update was used.
- REQ-LEARN-3116-4: Negative controls SHALL include no-feedback, shuffled-hint,
  stale-hint, and contradictory-hint entries when feasible, each reporting case
  count, mistake counts, deltas, rollback status, and whether the promotion gate
  failed safely.
- REQ-LEARN-3116-5: `soundness_mistakes` SHALL count exact-rejected protocol
  rows accepted by the guarded controller; `completeness_mistakes` SHALL count
  exact-accepted protocol rows rejected or abstained by the guarded controller.
- REQ-LEARN-3116-6: `promotion_decision` SHALL be `controller_only` only when
  there are zero soundness mistakes, fewer completeness mistakes than Exp 3103,
  nonnegative prior retention, nonnegative delayed regression, a positive
  family-holdout delta, and every negative control fails safely without model
  weight mutation.

### SCENARIO-LEARN-3116: Abstract Hints Repair Hard Completeness and Retention

**Given** ready Exp 3090, Exp 3097, and Exp 3103 source artifacts
**When** Exp 3116 replays solver-derived abstract hints over the identified
hard/zero-pass families
**Then** it writes a complete terminal artifact with controller-only learning,
no weight updates, zero protocol soundness mistakes, fewer completeness mistakes
than Exp 3103, nonnegative prior retention, nonnegative delayed regression,
negative controls that fail safely, `promotion_decision="controller_only"`, and
an `honest_verdict` beginning with `complete_`.

### SCENARIO-LEARN-3116-BLOCKED: Missing Curriculum Sources Fail Closed

**Given** missing Exp 3090, Exp 3097, Exp 3103, or stratified fixture evidence
**When** Exp 3116 runs
**Then** it writes the required artifact fields with zeroed metrics,
`fr11_unsolvable_curriculum_ready=false`, `promotion_decision="blocked"`,
empty negative controls, no model-weight mutation claim, explicit failed
precondition diagnostics, and `honest_verdict="blocked_precondition_failed"`.

## Implementation Status (REQ-LEARN-3116)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3116 | Implemented (`python/carnot/eval/fr11_unsolvable_curriculum_retention_guard_v1.py`) | Implemented (`tests/python/test_experiment_3116_fr11_unsolvable_curriculum_retention_guard_v1.py`) |

## REQ-LEARN-3128: FR-11 EvoEnv Verifiable Environment Synthesis Pilot

**Given** Exp 3123 records the mandated local SOTA model cache status and Exp
3116 records the prior hard-family retention guard
**When** Exp 3128 runs the bounded EvoEnv-style FR-11 pilot
**Then** it SHALL define executable constraint-environment schemas that can
sample deterministic instances, compute exact reference solutions, and score
candidate responses with a verifier cheaper than the solver.
**And** it SHALL admit a bounded set of candidate environments only when each
environment passes executable determinism, solve-verify asymmetry, novelty
against prior fixture evidence, difficulty calibration, no-answer-leakage, and
solver-authority checks.
**And** it SHALL separate live model environment synthesis from solver-only
environment admission by reporting `selected_model_ids`, `live_call_count`, and
`live_model_environment_synthesis` without using legacy small models for
headline FR-11 evidence.
**And** it SHALL replay the prior FR-11 retention family evidence so new
environment curricula do not regress prior exact constraints, while explicitly
declaring that no model-weight update happened unless separately tested.

The terminal artifact SHALL be
`results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json`
and SHALL include `fr11_evoenv_pilot_v1_ready`,
`continuous_self_learning_targeted`, `model_specs`, `selected_model_ids`,
`live_call_count`, `live_model_environment_synthesis`,
`candidate_environment_count`, `admitted_environment_count`,
`solve_verify_asymmetry_pass_rate`, `novelty_pass_rate`,
`no_answer_leakage_pass_rate`, `retention_delta`, `soundness_errors`,
`completeness_errors`, `no_weight_update_claim`, `tests_run`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3128 Sub-requirements

- REQ-LEARN-3128-1: The schema SHALL include a constraint-environment object
  with deterministic sampling, exact reference computation, and response
  scoring methods.
- REQ-LEARN-3128-2: Admission SHALL reject any candidate that is nondeterministic,
  duplicate of prior fixture evidence, too easy or too hard for bounded replay,
  leaks its canonical answer in the prompt, lacks exact solver authority, or
  fails the solve-verify asymmetry threshold.
- REQ-LEARN-3128-3: Solver authority SHALL count false accepts as
  `soundness_errors` and false rejects of canonical references as
  `completeness_errors`; `fr11_evoenv_pilot_v1_ready` SHALL require both counts
  to be zero.
- REQ-LEARN-3128-4: The pilot SHALL read the Exp 3123 precondition manifest and
  report mandated model availability, but SHALL set
  `live_model_environment_synthesis=false` and `live_call_count=0` whenever no
  live model call was actually made.
- REQ-LEARN-3128-5: The retention replay SHALL compare prior FR-11 exact-family
  decisions before and after adding the admitted environments and SHALL require
  `retention_delta >= 0.0` for readiness.
- REQ-LEARN-3128-6: The artifact SHALL set `no_weight_update_claim=true` and
  SHALL NOT claim base-model, KAN-model, or LLM-weight learning for controller
  or environment-schema admission.

### SCENARIO-LEARN-3128: Bounded EvoEnv Admission Produces Reusable Environments

**Given** ready Exp 3123 and Exp 3116 source artifacts
**When** Exp 3128 samples bounded constraint-environment candidates and runs the
admission gates
**Then** it writes a complete terminal artifact with at least one admitted
environment, zero soundness errors, zero completeness errors, nonnegative
retention delta, no live-model synthesis claim unless a live call occurred,
`no_weight_update_claim=true`, and an `honest_verdict` beginning with
`complete:`.

### SCENARIO-LEARN-3128-BLOCKED: Missing Source Evidence Fails Closed

**Given** missing Exp 3123 or Exp 3116 evidence
**When** Exp 3128 runs
**Then** it writes the required artifact fields with zero admitted
environments, zero live calls, `live_model_environment_synthesis=false`,
`fr11_evoenv_pilot_v1_ready=false`, explicit failed-precondition diagnostics,
and a blocked `honest_verdict`.

## Implementation Status (REQ-LEARN-3128)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3128 | Implemented (`python/carnot/eval/fr11_evoenv_verifiable_environment_synthesis_v1.py`) | Implemented (`tests/python/test_experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.py`) |

## REQ-LEARN-3129: FR-11 Constraint Memory Retention And Drift Audit

**Given** Exp 3128 admitted EvoEnv-style constraint environments and Exp 3116
records the prior FR-11 hard-family retention guard
**When** Exp 3129 audits the admitted environments after synthesis
**Then** it SHALL reload the admitted environments from the Exp 3128 artifact,
reconstruct their executable validators, and replay every admitted environment
by exact enumeration rather than trusting stored pass/fail summaries.
**And** it SHALL replay the prior FR-11 guarded decision families from Exp 3116
against their exact target actions so the audit can detect forgetting after
environment-memory admission.
**And** it SHALL replay the Exp 3126 fragment-time monitor ledger so
satisfiable drift remains distinct from contradiction, format failure, and
data-prior mismatch.
**And** it SHALL separate controller/environment memory from model-weight
learning by reporting the inference substrate and by keeping
`no_weight_update_claim=true`.

The terminal artifact SHALL be
`results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json`
and SHALL include `fr11_constraint_memory_audit_v1_ready`,
`admitted_environment_count`, `replay_family_count`,
`prior_retention_delta`, `novelty_retention_delta`, `soundness_errors`,
`completeness_errors`, `satisfiable_drift_count`,
`ledger_consistency_rate`, `promotion_recommendation`,
`no_weight_update_claim`, `tests_run`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3129 Sub-requirements

- REQ-LEARN-3129-1: The audit denominator SHALL match Exp 3128's
  `admitted_environment_count` and SHALL fail closed when that count diverges
  from the number of replayed admitted-environment rows.
- REQ-LEARN-3129-2: Environment replay SHALL count soundness errors as exact
  invalid assignments accepted by the environment verifier and completeness
  errors as exact valid assignments rejected by the environment verifier.
- REQ-LEARN-3129-3: Prior retention replay SHALL compute
  `prior_retention_delta` from Exp 3116 guarded decisions before and after the
  audit replay; negative deltas SHALL block FR-11 promotion.
- REQ-LEARN-3129-4: Novelty retention replay SHALL compute
  `novelty_retention_delta` by comparing Exp 3128 admission-time environment
  success against the post-synthesis exact replay success rate.
- REQ-LEARN-3129-5: Satisfiable drift SHALL be recomputed from Exp 3126 monitor
  events and SHALL NOT be merged into contradiction or generic soundness
  counters.
- REQ-LEARN-3129-6: `promotion_recommendation` SHALL explicitly distinguish
  controller/environment-memory reuse from model-weight learning, and the audit
  SHALL set `no_weight_update_claim=true`.

### SCENARIO-LEARN-3129: EvoEnv Memory Retains Prior And Novel Families

**Given** ready Exp 3116, Exp 3126, and Exp 3128 source artifacts
**When** Exp 3129 replays prior FR-11 families and admitted EvoEnv families
through exact validators
**Then** it writes a complete terminal artifact whose admitted-environment
denominator matches Exp 3128, whose soundness and completeness errors are zero,
whose prior and novelty retention deltas are nonnegative, whose satisfiable
drift count is recomputed separately, whose no-weight-update claim is true, and
whose `honest_verdict` begins with `complete:`.

### SCENARIO-LEARN-3129-BLOCKED: Missing Audit Sources Fail Closed

**Given** missing Exp 3116, Exp 3126, or Exp 3128 source evidence
**When** Exp 3129 runs
**Then** it writes the required artifact fields with zeroed replay metrics,
`fr11_constraint_memory_audit_v1_ready=false`, a blocking promotion
recommendation, no model-weight mutation claim, explicit failed-precondition
diagnostics, and a blocked `honest_verdict`.

## Implementation Status (REQ-LEARN-3129)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3129 | Implemented (`python/carnot/eval/fr11_constraint_memory_retention_drift_audit_v1.py`) | Implemented (`tests/python/test_experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.py`) |

## REQ-LEARN-3142: FR-11 VeRA EvoEnv Hardening v2

**Given** Exp 3128 admitted executable EvoEnv-style environments, Exp 3129
records the constraint-memory audit and ledger blocker, and Exp 3123 records
the mandated local SOTA model cache policy
**When** Exp 3142 runs the VeRA-style FR-11 hardening pass
**Then** it SHALL load every admitted environment, generate executable
equivalent variants and hardened variants with deterministic solver templates,
and replay both prior and new variants through exact validators and a
constraint-memory ledger.
**And** it SHALL report live-model variant generation separately from
solver-only variant generation by recording `model_specs`,
`selected_model_ids`, `live_call_count`, and
`live_model_variant_generation`.
**And** it SHALL measure determinism, solve-verify asymmetry, answer leakage,
novelty, difficulty calibration, ledger consistency, soundness, and
completeness without claiming model-weight learning.

The terminal artifact SHALL be
`results/experiment_3142_fr11_vera_evoenv_hardening_v2.json` and SHALL include
`fr11_vera_evoenv_v2_ready`, `continuous_self_learning_targeted`,
`model_specs`, `selected_model_ids`, `live_call_count`,
`live_model_variant_generation`, `admitted_environment_count`,
`equivalent_variant_count`, `hardened_variant_count`,
`solve_verify_asymmetry_pass_rate`, `no_answer_leakage_pass_rate`,
`ledger_consistency_rate`, `soundness_errors`, `completeness_errors`,
`no_weight_update_claim`, `promotion_recommendation`, `tests_run`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3142 Sub-requirements

- REQ-LEARN-3142-1: The variant generator SHALL produce at least one
  equivalent variant and at least one hardened variant per admitted Exp 3128
  environment, using only executable templates and deterministic solver
  parameters unless live model calls are explicitly recorded.
- REQ-LEARN-3142-2: Equivalent variants SHALL preserve the admitted
  environment's exact solution count while changing prompt or variable surface
  form enough to yield a new novelty signature.
- REQ-LEARN-3142-3: Hardened variants SHALL preserve exact solver authority
  and no-answer-leakage while increasing verifier hardness or reducing
  solution density relative to the admitted source environment.
- REQ-LEARN-3142-4: The hardening replay SHALL count soundness errors as exact
  invalid assignments accepted by a variant verifier and completeness errors as
  exact valid assignments rejected by a variant verifier; readiness requires
  both counts to be zero.
- REQ-LEARN-3142-5: The memory-ledger replay SHALL include the prior Exp 3129
  ledger consistency measurement and the new exact variant ledger measurement
  so the Exp 3129 blocker remains directly auditable.
- REQ-LEARN-3142-6: The artifact SHALL set `no_weight_update_claim=true` and
  SHALL NOT claim base-model, KAN-model, or LLM-weight learning when only
  controller/environment variants are generated.

### SCENARIO-LEARN-3142: VeRA Variants Harden EvoEnv Without Weight Claims

**Given** ready Exp 3123, Exp 3128, and Exp 3129 source artifacts
**When** Exp 3142 generates solver-only VeRA-E and VeRA-H variants and replays
them through exact validators and memory ledgers
**Then** it writes a complete terminal artifact with the admitted-environment
denominator preserved, nonzero equivalent and hardened variant counts, zero
soundness errors, zero completeness errors, no live-model generation claim when
no live call occurred, `no_weight_update_claim=true`, an explicit promotion
recommendation, and an `honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-3142-BLOCKED: Missing VeRA Source Evidence Fails Closed

**Given** missing Exp 3123, Exp 3128, or Exp 3129 source evidence
**When** Exp 3142 runs
**Then** it writes the required artifact fields with zeroed variant metrics,
`fr11_vera_evoenv_v2_ready=false`, zero live calls,
`live_model_variant_generation=false`, no model-weight mutation claim, explicit
failed-precondition diagnostics, and a blocked `honest_verdict`.

## Implementation Status (REQ-LEARN-3142)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3142 | Implemented (`python/carnot/eval/fr11_vera_evoenv_hardening_v2.py`) | Implemented (`tests/python/test_experiment_3142_fr11_vera_evoenv_hardening_v2.py`) |

## REQ-LEARN-3143: FR-11 Experience-Driven Verifier Memory

**Given** Exp 3136 records false-accept mechanisms and prior verifier rows,
Exp 3142 records FR-11 VeRA/EvoEnv variant replay outcomes, and Exp 3129
records the prior constraint-memory ledger blocker
**When** Exp 3143 builds an experience-driven verifier memory policy
**Then** it SHALL load those checked-in artifacts, derive replayable memory keys
from fixture family, difficulty, answer format, failure mechanism, and contract
decision, and simulate a deterministic routing policy over the historical rows.
**And** it SHALL suppress checks only for exact low-risk memory keys with no
false-accept, false-reject, soundness, completeness, or ledger inconsistency
history while escalating families with historical false accepts or replay
errors.
**And** it SHALL report efficiency and safety metrics including suppressed
checks, escalated checks, check savings, residual false-accept risk, residual
false-reject risk, and ledger consistency without claiming model-weight
learning.

The terminal artifact SHALL be
`results/experiment_3143_fr11_experience_driven_verifier_memory_v1.json` and
SHALL include `fr11_experience_verifier_memory_v1_ready`,
`continuous_self_learning_targeted`, `memory_key_schema`, `replay_row_count`,
`suppressed_check_count`, `escalated_check_count`,
`estimated_check_savings_rate`, `residual_false_accept_risk`,
`residual_false_reject_risk`, `ledger_consistency_rate`,
`no_weight_update_claim`, `promotion_recommendation`, `tests_run`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3143 Sub-requirements

- REQ-LEARN-3143-1: The memory key schema SHALL be machine-readable and SHALL
  include fixture family, difficulty, answer format, failure mechanism, and
  contract decision fields so learned experience can be replayed exactly.
- REQ-LEARN-3143-2: The replay denominator SHALL combine Exp 3136 verifier
  rows and Exp 3142 variant replay rows, with every replay row assigned exactly
  one routing decision: `suppress`, `normal`, or `escalate`.
- REQ-LEARN-3143-3: A check SHALL be suppressed only when exact memory history
  for that key has no false accepts, no false rejects, no replay errors, and a
  ledger consistency rate of 1.0.
- REQ-LEARN-3143-4: A check SHALL be escalated when its exact key or broader
  family has false-accept history, ledger inconsistency, soundness errors, or
  completeness errors.
- REQ-LEARN-3143-5: Residual false-accept and false-reject risk SHALL be
  computed from simulated routing outcomes and SHALL remain visible even when
  suppression creates estimated check savings.
- REQ-LEARN-3143-6: The artifact SHALL set `no_weight_update_claim=true` and
  SHALL distinguish controller/routing memory from base-model, KAN-model, or
  LLM-weight learning.

### SCENARIO-LEARN-3143: Experience Memory Suppresses Safe Keys And Escalates False-Accept Families

**Given** ready Exp 3136, Exp 3142, and Exp 3129 source artifacts
**When** Exp 3143 simulates the experience-driven verifier memory policy
**Then** it writes a complete terminal artifact with a nonzero replay
denominator, at least one suppressed check, at least one escalated check,
nonzero estimated check savings, zero residual false-accept risk for
suppressed rows, explicit false-reject risk, the inherited ledger consistency
rate, `no_weight_update_claim=true`, an explicit promotion recommendation, and
an `honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-3143-BLOCKED: Missing Experience Sources Fail Closed

**Given** missing Exp 3136, Exp 3142, or Exp 3129 source evidence
**When** Exp 3143 runs
**Then** it writes the required artifact fields with zeroed routing metrics,
`fr11_experience_verifier_memory_v1_ready=false`, no model-weight mutation
claim, explicit failed-precondition diagnostics, and a blocked
`honest_verdict`.

## Implementation Status (REQ-LEARN-3143)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3143 | Implemented (`python/carnot/eval/fr11_experience_driven_verifier_memory_v1.py`) | Implemented (`tests/python/test_experiment_3143_fr11_experience_driven_verifier_memory_v1.py`) |

## REQ-LEARN-3156: FR-11 Ledger Consistency Closure Replay

**Given** Exp 3128 records admitted executable EvoEnv environments, Exp 3129
records the prior FR-11 constraint-memory ledger blocker, Exp 3142 records
VeRA/EvoEnv equivalent and hardened variants, Exp 3143 records
experience-driven verifier memory, and Exp 3136 records historical
false-accept families
**When** Exp 3156 runs the FR-11 ledger-consistency closure replay
**Then** it SHALL build a replay panel that covers admitted environments,
equivalent variants, hardened variants, historical false-accept families, and
any residual inconsistent monitor rows without using tautological stored
consistency booleans as the measured ledger outcome.
**And** it SHALL recompute the ledger consistency numerator and denominator
from exact environment replay, exact variant replay, monitor ledger actions,
exact expected actions, and observed verifier decisions.
**And** it SHALL classify every residual mismatch as exactly one of
`missing_label`, `stale_memory`, `contradictory_memory`,
`variant_generation_error`, or `monitor_replay_error`.
**And** it SHALL promote controller/environment memory only when
`ledger_consistency_rate == 1.0`, `soundness_errors == 0`, and
`completeness_errors == 0`; otherwise it SHALL block promotion while preserving
`no_weight_update_claim=true`.

The terminal artifact SHALL be
`results/experiment_3156_fr11_ledger_consistency_closure_v1.json` and SHALL
include `fr11_ledger_consistency_closure_v1_ready`,
`continuous_self_learning_targeted`, `replay_panel_count`,
`ledger_consistency_rate`, `soundness_errors`, `completeness_errors`,
`residual_mismatch_rows`, `promotion_recommendation`,
`no_weight_update_claim`, `methodology_complete`, `tests_run`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3156 Sub-requirements

- REQ-LEARN-3156-1: The replay panel SHALL include one row per admitted
  Exp 3128 environment, one row per Exp 3142 equivalent variant, one row per
  Exp 3142 hardened variant, every Exp 3136 verifier row from a historical
  false-accept source family, and every additional observed Exp 3126 monitor
  row that remains inconsistent and is not already represented by Exp 3136.
- REQ-LEARN-3156-2: Environment and variant rows SHALL be replayed from their
  executable constraint systems; stored pass/fail summaries SHALL NOT be the
  sole source of truth for ledger consistency.
- REQ-LEARN-3156-3: Historical verifier rows SHALL recompute consistency by
  comparing monitor `constraint_ledger` actions, exact expected actions, and
  observed verifier decisions; stored fields such as
  `final_answer_consistent_with_ledger` and `ledger_consistent` SHALL NOT be
  used as the measured outcome.
- REQ-LEARN-3156-4: `residual_mismatch_rows` SHALL include enough row id,
  source artifact, fixture family, expected action, observed action, ledger
  action, and mismatch-class evidence to replay each counterexample.
- REQ-LEARN-3156-5: `promotion_recommendation` SHALL block all FR-11 promotion
  when ledger consistency is below 1.0 or soundness/completeness checks fail;
  only a perfect ledger with zero soundness and completeness errors may promote
  controller/environment memory, and no artifact may claim model-weight
  learning.

### SCENARIO-LEARN-3156: Closure Replay Preserves A Residual Ledger Gap

**Given** ready Exp 3128, 3129, 3136, 3142, and 3143 source artifacts
**When** Exp 3156 replays environment memory and experience memory against
their exact variants
**Then** it writes a complete terminal artifact with an explicit replay-panel
denominator, a recomputed ledger consistency rate, zero soundness and
completeness errors for safe controller memory, residual mismatch
counterexamples for any non-perfect ledger rows, an explicit blocked promotion
recommendation when the rate is below 1.0, `no_weight_update_claim=true`, and
an `honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-3156-BLOCKED: Missing Closure Sources Fail Closed

**Given** missing Exp 3128, 3129, 3136, 3142, or 3143 source evidence
**When** Exp 3156 runs
**Then** it writes the required artifact fields with a zero replay denominator,
`fr11_ledger_consistency_closure_v1_ready=false`, a blocking promotion
recommendation, no model-weight mutation claim, explicit failed-precondition
diagnostics, and a blocked `honest_verdict`.

## Implementation Status (REQ-LEARN-3156)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3156 | Implemented (`python/carnot/eval/fr11_ledger_consistency_closure_v1.py`) | Implemented (`tests/python/test_experiment_3156_fr11_ledger_consistency_closure_v1.py`) |

## REQ-LEARN-3157: FR-11 Attractor Residual Memory Audit

**Given** Exp 3156 records a bounded FR-11 ledger closure panel, Exp 3143
records experience-driven verifier memory, and Exp 3136 records historical
false-accept-prone verifier families
**When** Exp 3157 audits attractor/residual-style controller memory
**Then** it SHALL define measurable residual-like signals available from exact
replay, including repeated mismatch count, convergence to a stable verdict,
contradiction-core stability, and memory routing entropy.
**And** it SHALL replay those signals over the FR-11 ledger closure panel and
historical false-accept-prone families without running live LLM inference or
mutating model weights.
**And** it SHALL measure whether residual memory suppresses redundant checks,
escalates risky families, or introduces unsafe skips.
**And** it SHALL block promotion when any unsafe skip appears or when Exp 3156
ledger consistency is below 1.0, preserving `no_weight_update_claim=true`.

The terminal artifact SHALL be
`results/experiment_3157_fr11_attractor_residual_memory_audit_v1.json` and
SHALL include `fr11_attractor_residual_memory_audit_v1_ready`,
`continuous_self_learning_targeted`, `residual_signal_definitions`,
`replay_panel_count`, `risky_family_escalation_rate`,
`redundant_check_suppression_rate`, `unsafe_skip_count`,
`promotion_recommendation`, `no_weight_update_claim`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3157 Sub-requirements

- REQ-LEARN-3157-1: `residual_signal_definitions` SHALL be a machine-readable
  list defining repeated mismatch count, stable verdict convergence,
  contradiction-core stability, and memory routing entropy, including the
  exact replay fields used to compute each signal.
- REQ-LEARN-3157-2: The replay denominator SHALL be the Exp 3156 closure panel,
  augmented with Exp 3143 routing memory when row ids overlap; the artifact
  SHALL report the explicit `replay_panel_count`.
- REQ-LEARN-3157-3: Risky-family escalation SHALL count rows whose family or
  exact row has prior false-accept, ledger-inconsistency, or repeated-mismatch
  evidence and whose residual policy routes to `escalate`.
- REQ-LEARN-3157-4: Redundant-check suppression SHALL be allowed only for rows
  with stable exact verdict convergence, no repeated mismatch evidence, no
  contradiction-core instability, and a low-risk Exp 3143 routing decision.
- REQ-LEARN-3157-5: `unsafe_skip_count` SHALL count any suppressed row whose
  exact replay expects rejection, whose observed decision is not acceptably
  consistent, or whose family is risky; any unsafe skip SHALL block promotion.
- REQ-LEARN-3157-6: The artifact SHALL set `no_weight_update_claim=true` and
  SHALL distinguish controller residual/routing memory updates from base-model,
  KAN-model, or LLM-weight learning.

### SCENARIO-LEARN-3157: Residual Memory Escalates Risk And Blocks Imperfect Ledger Promotion

**Given** ready Exp 3156, Exp 3143, and Exp 3136 source artifacts
**When** Exp 3157 computes residual-like signals and simulates the bounded
controller memory policy
**Then** it writes a complete terminal artifact with a nonzero replay
denominator, all required residual signal definitions, positive risky-family
escalation, measured redundant-check suppression, zero unsafe skips, an
explicit blocked promotion recommendation when Exp 3156 ledger consistency is
below 1.0, `no_weight_update_claim=true`, and an `honest_verdict` beginning
with `complete:`.

### SCENARIO-LEARN-3157-BLOCKED: Missing Residual Audit Sources Fail Closed

**Given** missing Exp 3156, Exp 3143, or Exp 3136 source evidence
**When** Exp 3157 runs
**Then** it writes the required artifact fields with zeroed routing metrics,
`fr11_attractor_residual_memory_audit_v1_ready=false`, no model-weight
mutation claim, explicit failed-precondition diagnostics, and a blocked
`honest_verdict`.

## Implementation Status (REQ-LEARN-3157)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3157 | Implemented (`python/carnot/eval/fr11_attractor_residual_memory_audit_v1.py`) | Implemented (`tests/python/test_experiment_3157_fr11_attractor_residual_memory_audit_v1.py`) |

## REQ-LEARN-3171: FR-11 Ledger Counterexample Isolation

**Given** Exp 3156 reports a non-perfect FR-11 ledger closure rate and Exp 3157
blocks promotion while preserving controller-only residual memory evidence
**When** Exp 3171 isolates the ledger blocker
**Then** it SHALL reconstruct the Exp 3156 replay panel from checked-in source
artifacts, separate passing rows from failing rows, and identify concrete
counterexample families rather than reporting only an aggregate rate.
**And** it SHALL define a controller-only environment/variant split with
training-update rows, held-out replay rows, and negative-control rows so the
next FR-11 pilot can evaluate nonforgetting separately from memory updates.
**And** it SHALL classify whether the observed inconsistency is explained by
stale memory, environment mismatch, threshold drift, missing exact labels,
aggregation schema mismatch, or a direct controller-observation contradiction.
**And** it SHALL block promotion whenever the prior ledger consistency rate is
below 1.0 and SHALL preserve `no_model_weight_update_claimed=true`.

The terminal artifact SHALL be
`results/experiment_3171_fr11_ledger_counterexample_isolation_v1.json` and
SHALL include `fr11_ledger_counterexample_isolation_ready`,
`continuous_self_learning_task`, `prior_ledger_consistency_rate`,
`isolated_counterexample_families`, `passing_families`,
`environment_variant_split`, `negative_control_rows`,
`suspected_failure_modes`, `promotion_allowed`,
`no_model_weight_update_claimed`, `source_artifacts`, `inference_substrate`,
and `honest_verdict`.

### REQ-LEARN-3171 Sub-requirements

- REQ-LEARN-3171-1: The isolation denominator SHALL equal Exp 3156's
  `replay_panel_rows` count, and the carried-forward consistency rate SHALL
  equal Exp 3156's `ledger_consistency_rate`.
- REQ-LEARN-3171-2: `isolated_counterexample_families` SHALL include every
  non-consistent Exp 3156 row grouped by fixture family, including row id,
  exact expected action, ledger action, observed action, mismatch class,
  panel category, routing decision, and source artifact.
- REQ-LEARN-3171-3: `passing_families` SHALL preserve working behavior by
  grouping consistent rows by fixture family and panel category.
- REQ-LEARN-3171-4: `environment_variant_split.training_update_rows` SHALL
  include only failing counterexample rows, while held-out replay and negative
  controls SHALL include non-failing rows and SHALL NOT duplicate training
  update row ids.
- REQ-LEARN-3171-5: Failure-mode classification SHALL explicitly mark stale
  memory, environment mismatch, threshold drift, missing exact label, and
  aggregation schema mismatch as supported or rejected by row-level evidence.
- REQ-LEARN-3171-6: The artifact SHALL declare zero live inference calls, no
  base-model/KAN/model-weight mutation, and controller replay only.

### SCENARIO-LEARN-3171: Isolated Counterexamples Feed A Nonforgetting Pilot

**Given** ready Exp 3128, 3129, 3142, 3143, 3156, and 3157 artifacts
**When** Exp 3171 reconstructs the ledger blocker
**Then** it writes a complete artifact where the failing rows are isolated as
controller-observation contradictions, passing environment and variant families
remain visible, training rows are separated from held-out replay rows and
negative controls, promotion remains blocked, and `honest_verdict` begins with
`complete:`.

### SCENARIO-LEARN-3171-BLOCKED: Missing Isolation Sources Fail Closed

**Given** missing Exp 3156 or missing replay panel evidence
**When** Exp 3171 runs
**Then** it writes schema-complete fields with no training-update rows,
`fr11_ledger_counterexample_isolation_ready=false`, promotion blocked, no
model-weight mutation claim, explicit failed-precondition diagnostics, and a
blocked `honest_verdict`.

## Implementation Status (REQ-LEARN-3171)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3171 | Planned (`python/carnot/eval/fr11_ledger_counterexample_isolation_v1.py`) | Planned (`tests/python/test_experiment_3171_fr11_ledger_counterexample_isolation_v1.py`) |

## REQ-LEARN-3172: FR-11 Nonforgetting Self-Learning Pilot v2

**Given** Exp 3171 records an isolated FR-11 environment/variant split with
training-update rows, held-out replay rows, and negative-control rows
**When** Exp 3172 runs the bounded nonforgetting self-learning pilot
**Then** it SHALL load the checked-in Exp 3171 artifact and verify that the
environment/variant split is present before applying any update.
**And** it SHALL apply only a bounded controller-memory update or threshold
adjustment to the `training_update_rows`; held-out rows and negative-control
rows SHALL NOT contribute to the update.
**And** it SHALL replay training, held-out, and negative-control rows after the
update, measuring before/after ledger consistency, held-out consistency, and
negative-control regressions.
**And** it SHALL preserve the model-weight boundary: no finetuning, no hidden
state mutation claim, no base-model/KAN/model-weight update, and zero live
model inference calls.
**And** it SHALL allow promotion only when after-update ledger consistency is
1.0, held-out consistency is 1.0, no negative controls regress, and
nonforgetting passes.

The terminal artifact SHALL be
`results/experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.json` and
SHALL include `fr11_nonforgetting_self_learning_pilot_v2_ready`,
`continuous_self_learning_task`, `controller_memory_update_applied`,
`model_weight_update_claimed`, `before_ledger_consistency_rate`,
`after_ledger_consistency_rate`, `heldout_consistency_rate`,
`negative_control_regression_count`, `nonforgetting_passed`,
`promotion_allowed`, `promotion_recommendation`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3172 Sub-requirements

- REQ-LEARN-3172-1: The pilot SHALL fail closed with a schema-complete blocked
  artifact and `promotion_allowed=false` when Exp 3171 is missing, not ready, or
  lacks `training_update_rows`, `held_out_replay_rows`, or negative-control
  rows.
- REQ-LEARN-3172-2: The controller-memory update SHALL be bounded to exact
  training row ids and SHALL NOT read held-out or negative-control rows when
  constructing update memory.
- REQ-LEARN-3172-3: Before/after ledger consistency SHALL be measured over the
  training-update plus held-out replay denominator; held-out and
  negative-control rows SHALL be replayed after the update without contributing
  to it.
- REQ-LEARN-3172-4: `negative_control_regression_count` SHALL count any
  negative-control row whose observed action changes or whose consistency
  regresses after the controller-memory update.
- REQ-LEARN-3172-5: `promotion_allowed` SHALL be true only when after-update
  ledger consistency is 1.0, held-out consistency is 1.0, nonforgetting passes,
  `negative_control_regression_count == 0`, and
  `model_weight_update_claimed=false`.
- REQ-LEARN-3172-6: The artifact SHALL declare zero live inference calls and no
  base-model, KAN-model, hidden-state, or model-weight learning.

### SCENARIO-LEARN-3172: Controller-Only Update Repairs Isolated Failures

**Given** a ready Exp 3171 artifact whose training-update rows contain isolated
reject-vs-accept ledger contradictions and whose held-out and negative-control
rows are already consistent
**When** Exp 3172 applies row-bounded controller memory only to the
training-update rows
**Then** the after-update ledger consistency reaches 1.0, held-out consistency
remains 1.0, negative-control regression count is zero, promotion is allowed
for controller memory only, `model_weight_update_claimed=false`, and
`honest_verdict` begins with `complete:`.

### SCENARIO-LEARN-3172-BLOCKED: Missing Split Fails Closed

**Given** Exp 3171 is missing, not ready, or lacks the environment/variant split
needed for nonforgetting replay
**When** Exp 3172 runs
**Then** it writes all required fields with zeroed rates,
`controller_memory_update_applied=false`, `promotion_allowed=false`, no
model-weight update claim, explicit precondition diagnostics, and a blocked
`honest_verdict`.

## Implementation Status (REQ-LEARN-3172)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3172 | Implemented (`python/carnot/eval/fr11_nonforgetting_self_learning_pilot_v2.py`) | Implemented (`tests/python/test_experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.py`) |

## REQ-LEARN-3186: FR-11 Controller-Memory Promotion Pack v1

**Given** Exp 3172 produced a promotable FR-11 controller-memory update with
before/after ledger consistency, held-out replay, and negative-control replay
evidence
**When** Exp 3186 packages the update for promotion
**Then** it SHALL write an auditable promotion pack that traces the exact
controller-memory rule to Exp 3172 and SHALL NOT claim model-weight learning,
model-weight mutation, hidden-state mutation, KAN training, or live model
inference.
**And** it SHALL classify the learning tier using `research-program.md` as
controller-memory learning, not online model-weight learning.
**And** it SHALL include a promotion manifest with update ID, source
counterexample families, activation predicate, rollback predicate, replay
requirements, and owner artifact paths for the source update, pack, tests,
module, Exp 3187 drift replay, and ops reconciliation.
**And** it SHALL include a rollback and monitoring policy for regressions,
stale ledger evidence, drift failure, and exact-authority conflict.

The terminal artifact SHALL be
`results/experiment_3186_fr11_controller_memory_promotion_pack_v1.json` and
SHALL include `fr11_controller_memory_promotion_pack_v1_ready`,
`continuous_self_learning_task`, `learning_tier`,
`no_model_weight_update_claimed`, `source_update_artifact`,
`promotion_manifest`, `replay_requirements`, `rollback_policy`,
`promotion_allowed`, `inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3186 Sub-requirements

- REQ-LEARN-3186-1: The pack SHALL fail closed with schema-complete fields
  when Exp 3172 is missing, not ready, not promotion-allowed, or claims a
  model-weight update.
- REQ-LEARN-3186-2: The promoted update SHALL be extracted from
  `controller_memory_update.updated_rows` and `row_action_overrides`, including
  before/after ledger consistency, held-out consistency, and negative-control
  regression results.
- REQ-LEARN-3186-3: The promotion manifest SHALL expose a deterministic update
  ID, source counterexample families, exact-row activation predicate, rollback
  predicate, replay requirements, and owner artifact paths.
- REQ-LEARN-3186-4: The learning tier SHALL be derived from
  `research-program.md` and SHALL identify Tier 2 controller-memory /
  constraint-memory learning, not online weight learning.
- REQ-LEARN-3186-5: The rollback policy SHALL make promotion reversible on
  negative-control regression, stale ledger evidence, drift replay failure, or
  exact-authority conflict.
- REQ-LEARN-3186-6: The artifact SHALL declare zero live inference calls and no
  base-model, KAN-model, hidden-state, or model-weight learning.

### SCENARIO-LEARN-3186: Promotable Controller-Memory Update Is Packaged

**Given** a ready Exp 3172 artifact whose controller update is promotion
allowed, reaches after-update ledger consistency 1.0, preserves held-out
consistency 1.0, and reports zero negative-control regressions
**When** Exp 3186 builds the promotion pack
**Then** the pack is ready, promotion remains allowed for controller memory
only, the manifest lists the exact row-id override rule and rollback metadata,
the replay requirements are ready for Exp 3187 drift replay, no model-weight
update is claimed, and `honest_verdict` begins with `complete:`.

### SCENARIO-LEARN-3186-BLOCKED: Missing Or Unsafe Source Fails Closed

**Given** Exp 3172 is missing, not ready, not promotion-allowed, regresses a
negative control, or claims model-weight learning
**When** Exp 3186 runs
**Then** it writes all required fields with `promotion_allowed=false`, a
blocked promotion manifest, controller-only/no-live-inference substrate
metadata, explicit failed-precondition diagnostics, and a blocked
`honest_verdict`.

## Implementation Status (REQ-LEARN-3186)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3186 | Implemented (`python/carnot/eval/fr11_controller_memory_promotion_pack_v1.py`) | Implemented (`tests/python/test_experiment_3186_fr11_controller_memory_promotion_pack_v1.py`) |

## REQ-LEARN-3187: FR-11 Cross-Environment Drift Replay v1

**Given** Exp 3186 packaged a promotion-allowed FR-11 controller-memory update
with an exact-row activation predicate and rollback triggers
**When** Exp 3187 replays that update for matrix v29
**Then** it SHALL load the checked-in Exp 3186 promotion manifest and fail
closed without replay mutation when the manifest is missing, not ready, or not
promotion-allowed.
**And** it SHALL select held-out, negative-control, and cross-environment rows
from checked-in exact/replay artifacts, preferring environments whose fixture
families differ from the original counterexample families while preserving
exact authority.
**And** it SHALL apply the controller-memory routing/update in replay mode only
and compare before/after consistency, held-out consistency, cross-environment
consistency, and negative-control regressions without making live model calls
or updating model weights.
**And** it SHALL report drift cases and rollback triggers, allowing promotion
only when no exact-authority or negative-control regression appears.

The terminal artifact SHALL be
`results/experiment_3187_fr11_cross_environment_drift_replay_v1.json` and
SHALL include `fr11_cross_environment_drift_replay_v1_ready`,
`continuous_self_learning_task`, `replay_mode_only`,
`no_model_weight_update_claimed`, `heldout_row_count`,
`cross_environment_row_count`, `before_after_consistency`,
`negative_control_regression_count`, `drift_cases`, `rollback_triggered`,
`promotion_allowed`, `inference_substrate`, and `honest_verdict`.

### REQ-LEARN-3187 Sub-requirements

- REQ-LEARN-3187-1: The replay SHALL write a schema-complete blocked artifact
  with `promotion_allowed=false`, `replay_mode_only=true`, and no replay
  mutation when Exp 3186 is missing, not ready, not promotion-allowed, or lacks
  a usable promotion manifest.
- REQ-LEARN-3187-2: Cross-environment row selection SHALL exclude the source
  counterexample families listed in the Exp 3186 manifest and SHALL use rows
  whose expected action, ledger action, and observed action provide exact replay
  authority.
- REQ-LEARN-3187-3: The controller update SHALL be applied only as an in-memory
  exact row-id override during replay, and the artifact SHALL declare zero live
  inference calls and no base-model, KAN-model, hidden-state, or model-weight
  learning.
- REQ-LEARN-3187-4: `before_after_consistency` SHALL expose denominators and
  rates for the source before/after panel, held-out rows, cross-environment
  rows, and negative-control rows.
- REQ-LEARN-3187-5: `drift_cases` SHALL list every held-out or
  cross-environment row whose consistency regresses, whose observed action
  changes unexpectedly, or whose exact authority conflicts with the replayed
  controller-memory update.
- REQ-LEARN-3187-6: `rollback_triggered` SHALL be true and
  `promotion_allowed` SHALL be false whenever any drift case or negative-control
  regression is present.

### SCENARIO-LEARN-3187: Cross-Environment Replay Preserves Promotion

**Given** a ready Exp 3186 promotion pack with exact row-id overrides and a
ready Exp 3171/3172 replay split containing held-out, negative-control, and
cross-environment rows from non-counterexample families
**When** Exp 3187 replays the controller-memory update without mutating
production state
**Then** the artifact reports nonzero held-out and cross-environment
denominators, no drift cases, zero negative-control regressions, promotion
allowed, no model-weight update claim, and an `honest_verdict` beginning with
`complete:`.

### SCENARIO-LEARN-3187-BLOCKED: Missing Or Unsafe Promotion Manifest Fails Closed

**Given** Exp 3186 is missing, not ready, not promotion-allowed, or lacks a
usable promotion manifest
**When** Exp 3187 runs
**Then** it writes all required fields with zeroed replay denominators,
`rollback_triggered=true`, `promotion_allowed=false`, controller-only/no-live
inference substrate metadata, explicit failed-precondition diagnostics, and a
blocked `honest_verdict`.

## Implementation Status (REQ-LEARN-3187)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3187 | Planned (`python/carnot/eval/fr11_cross_environment_drift_replay_v1.py`) | Planned (`tests/python/test_experiment_3187_fr11_cross_environment_drift_replay_v1.py`) |

## REQ-LEARN-3200: FR-11 VeriFY-Style Trace-Memory Controller

**Given** Exp 3186 promoted a controller-memory update and Exp 3187 replayed
that update across held-out, cross-environment drift, and negative-control rows
without model-weight mutation
**When** Exp 3200 builds the `.296` continuous self-learning artifact
**Then** it SHALL load the checked-in Exp 3186 and Exp 3187 artifacts, define a
VeriFY-style trace record containing initial answer, verification query,
consistency judgment, answer/abstain decision, exact label, and routing outcome,
and materialize trace-memory records for the held-out, drift, and
negative-control replay rows.
**And** it SHALL add an experience-pool rule that suppresses redundant rechecks
only when prior exact evidence for the same row/action label has remained
consistent and unchanged.
**And** it SHALL require zero negative-control regressions before promotion and
SHALL NOT claim any base-model, verifier-model, KAN-model, hidden-state, or LLM
model-weight update.

The terminal artifact SHALL be
`results/experiment_3200_fr11_verify_trace_memory_controller_v1.json` and SHALL
include `schema_version`, `experiment_id`, `continuous_self_learning_task`,
`source_artifacts`, `trace_schema`, `trace_count`, `heldout_row_count`,
`drift_row_count`, `negative_control_regression_count`,
`redundant_check_suppression_count`, `routing_accuracy_delta`,
`model_weight_update_performed`, `promotion_allowed`, and `honest_verdict`.

### REQ-LEARN-3200 Sub-requirements

- REQ-LEARN-3200-1: The trace schema SHALL be machine-readable and SHALL include
  the fields `initial_answer`, `verification_query`, `consistency_judgment`,
  `answer_abstain_decision`, `exact_label`, and `routing_outcome`.
- REQ-LEARN-3200-2: The replay denominator SHALL include Exp 3187 held-out
  rows, Exp 3187 cross-environment drift rows, and Exp 3187 negative-control
  rows, with `drift_row_count` mapped to the cross-environment replay
  denominator.
- REQ-LEARN-3200-3: A redundant check SHALL be suppressed only when the
  experience pool already contains exact evidence for the same row/action label
  with no inconsistent judgment and no observed-action change.
- REQ-LEARN-3200-4: `promotion_allowed` SHALL be true only when Exp 3186 and
  Exp 3187 allow promotion, held-out and drift replay denominators are nonzero,
  `negative_control_regression_count == 0`, and
  `model_weight_update_performed=false`.
- REQ-LEARN-3200-5: `routing_accuracy_delta` SHALL be numeric when inherited
  Exp 3187 source before/after routing lift is available, otherwise null, and
  the artifact SHALL keep the held-out/drift replay delta visible separately.
- REQ-LEARN-3200-6: The artifact SHALL declare controller-memory replay only:
  no live inference calls, no hidden fine-tune claim, and no model-weight
  mutation.

### SCENARIO-LEARN-3200: Trace Memory Suppresses Redundant Replay Without Regressions

**Given** ready Exp 3186 and Exp 3187 artifacts with nonzero held-out and
cross-environment drift replay rows and zero negative-control regressions
**When** Exp 3200 materializes the VeriFY-style trace-memory controller
**Then** the artifact reports nonzero trace count, nonzero held-out and drift
row counts, at least one experience-driven redundant-check suppression for a
previously seen exact row, zero negative-control regressions,
`model_weight_update_performed=false`, `promotion_allowed=true`, and an
`honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-3200-BLOCKED: Unsafe Source Or Negative Control Blocks Promotion

**Given** Exp 3186 or Exp 3187 is missing, not promotion-allowed, or Exp 3187
reports any negative-control regression
**When** Exp 3200 runs
**Then** it writes all required fields with controller-memory/no-live-inference
metadata, does not claim model-weight updates, sets `promotion_allowed=false`,
keeps the negative-control regression count visible, and emits an
`honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-LEARN-3200)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3200 | Implemented (`python/carnot/eval/fr11_verify_trace_memory_controller_v1.py`) | Implemented (`tests/python/test_experiment_3200_fr11_verify_trace_memory_controller_v1.py`) |

## REQ-LEARN-3201: FR-11 KAN-CL Nonforgetting Sidecar Audit

**Given** Exp 3200 materialized a VeriFY-style trace-memory controller and
Exp 3187 replayed the same controller-memory update across held-out,
cross-environment drift, and negative-control rows
**When** Exp 3201 audits the `.296` trace-memory controller with
KAN-CL-style locality and nonforgetting metrics
**Then** it SHALL load the checked-in Exp 3200 and Exp 3187 artifacts, replay
held-out, drift, and negative-control trace rows using existing exact labels,
and report regressions, locality violations, and rollback status.
**And** it SHALL define an interpretable `audit_metric_schema` over trace
features and routing bins, including exact-label consistency, routing-bin
retention, negative-control regression, and locality-boundary checks.
**And** it SHALL explicitly deny model-weight updates and sidecar verifier
promotion authority.

The terminal artifact SHALL be
`results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json` and SHALL
include `schema_version`, `experiment_id`, `source_artifacts`,
`audit_metric_schema`, `heldout_replay_count`, `drift_replay_count`,
`negative_control_regression_count`, `locality_violation_count`,
`rollback_triggered`, `model_weight_update_performed`,
`sidecar_promotion_allowed`, and `honest_verdict`.

### REQ-LEARN-3201 Sub-requirements

- REQ-LEARN-3201-1: The audit SHALL fail closed with schema-complete fields
  when Exp 3200 or Exp 3187 is missing, not ready, not terminal, or claims live
  inference or model-weight mutation.
- REQ-LEARN-3201-2: Held-out, drift, and negative-control replay counts SHALL
  be computed from Exp 3200 trace records, with drift mapped to Exp 3200's
  cross-environment trace role.
- REQ-LEARN-3201-3: Exact-label replay SHALL count a regression whenever an
  exact accepted row does not route to answer/verify, an exact rejected row does
  not route to abstain/escalate, or a trace is inconsistent or changed.
- REQ-LEARN-3201-4: Locality violations SHALL count any routing-bin boundary
  where the same exact row/evidence key changes expected action, exact label,
  answer/abstain decision, or non-control routing across held-out, drift, and
  negative-control roles.
- REQ-LEARN-3201-5: `rollback_triggered` SHALL be true whenever drift
  regressions, held-out regressions, negative-control regressions, source drift
  cases, or locality violations are present.
- REQ-LEARN-3201-6: The artifact SHALL set
  `model_weight_update_performed=false` and
  `sidecar_promotion_allowed=false` even when the sidecar audit finds zero
  regressions, because this audit is diagnostic and not verifier authority.

### SCENARIO-LEARN-3201: Sidecar Audit Preserves Trace-Memory Boundaries

**Given** ready Exp 3200 and Exp 3187 artifacts with exact held-out, drift, and
negative-control replay rows and zero source regressions
**When** Exp 3201 computes the KAN-CL-style sidecar audit
**Then** the artifact reports nonzero held-out and drift replay counts, zero
negative-control regressions, zero locality violations, rollback not triggered,
no model-weight update, no sidecar promotion authority, and an
`honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-3201-BLOCKED: Unsafe Source Or Boundary Violation Fails Closed

**Given** Exp 3200 or Exp 3187 is missing, unsafe, or the replay rows create a
negative-control or locality-boundary violation
**When** Exp 3201 runs
**Then** it writes all required fields with controller-memory/no-live-inference
metadata, keeps the regression and locality counts visible, sets
`rollback_triggered=true`, keeps `sidecar_promotion_allowed=false`, and emits
an `honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-LEARN-3201)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3201 | Implemented (`python/carnot/eval/fr11_kan_cl_nonforgetting_sidecar_audit_v1.py`) | Implemented (`tests/python/test_experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.py`) |

## REQ-LEARN-3215: FR-11 Evidence-Gated Trace Replay Controller Labels

**Given** Exp 3200 materialized the `.296` FR-11 VeriFY-style trace-memory
controller and Exp 3201 audited that trace memory without sidecar promotion
**When** Exp 3215 extends trace replay with utility labels
**Then** it SHALL load the checked-in Exp 3200 trace-memory artifact, reuse the
Exp 3200 trace schema fields where possible, and label replay candidates only
from verifier-backed trace outcomes rather than planned trajectories.
**And** each replay utility label SHALL record the exact verifier outcome,
controller utility of the prior route, redundant-check suppression status, and
rollback/retraction status.
**And** reward-weighted verifier logic SHALL be represented only as controller
utility label metadata; it SHALL NOT fine-tune, train, mutate, or claim to
improve any model weights.
**And** held-out and drift rows SHALL be evaluated for routing improvement and
redundant-check suppression while negative-control regressions and rollback
events gate promotion.

The terminal artifact SHALL be
`results/experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.json`
and SHALL include `schema_version`, `experiment_id`, `milestone`,
`continuous_self_learning_task`, `prior_trace_memory_artifact`, `trace_count`,
`evidence_backed_trace_count`, `replay_utility_label_count`,
`redundant_check_suppression_count`, `heldout_row_count`, `drift_row_count`,
`routing_improvement_count`, `negative_control_regression_count`,
`rollback_event_count`, `model_weight_update_claimed`, `promotion_allowed`,
`conductor_file_modified`, `active_roadmap_modified`, and `honest_verdict`.

### REQ-LEARN-3215 Sub-requirements

- REQ-LEARN-3215-1: The label schema SHALL preserve Exp 3200's trace fields
  `trace_id`, `row_id`, `replay_role`, `verification_query`,
  `consistency_judgment`, `answer_abstain_decision`, `exact_label`, and
  `routing_outcome`.
- REQ-LEARN-3215-2: A trace SHALL be evidence-backed only when it has an exact
  verifier outcome derived from checked-in trace evidence; labels SHALL NOT be
  emitted for missing, planned-only, or unverifiable trajectories.
- REQ-LEARN-3215-3: Reward weights SHALL be controller utility labels only:
  positive for verifier-backed useful routes, positive with redundant-check
  suppression metadata for repeated exact evidence, and negative/blocking for
  rollback or retraction cases.
- REQ-LEARN-3215-4: `routing_improvement_count` SHALL count held-out and drift
  labels whose verifier-backed route utility is positive and not rolled back or
  retracted.
- REQ-LEARN-3215-5: `promotion_allowed` SHALL be true only when held-out and
  drift denominators are nonzero, every trace has an evidence-backed utility
  label, `negative_control_regression_count == 0`, `rollback_event_count == 0`,
  `model_weight_update_claimed=false`, `conductor_file_modified=false`, and
  `active_roadmap_modified=false`.

### SCENARIO-LEARN-3215: Evidence-Gated Labels Promote Controller Replay Only

**Given** ready Exp 3200 and Exp 3201 artifacts with verifier-backed trace
records, nonzero held-out and drift rows, zero negative-control regressions, and
no rollback events
**When** Exp 3215 labels replay candidates
**Then** the artifact reports evidence-backed trace and utility-label counts
equal to the trace count, nonzero routing improvement over held-out/drift rows,
the inherited redundant-check suppression count, no model-weight update claim,
`promotion_allowed=true`, and an `honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-3215-BLOCKED: Planned-Only Or Unsafe Replay Blocks Promotion

**Given** Exp 3200 is missing, unsafe, contains traces without exact verifier
outcomes, or the source artifacts report negative-control regressions or
rollback events
**When** Exp 3215 runs
**Then** it writes all required fields with controller-memory/no-live-inference
metadata, emits no utility label for planned-only rows, keeps the blocker counts
visible, sets `promotion_allowed=false`, and emits an `honest_verdict` beginning
with `complete:`.

## Implementation Status (REQ-LEARN-3215)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3215 | Implemented (`python/carnot/eval/fr11_evidence_gated_trace_replay_controller_v2.py`) | Implemented (`tests/python/test_experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.py`) |

## REQ-LEARN-3216: FR-11 Grounded Continuation Trace Graph Nonforgetting Queue

**Given** Exp 3215 materialized evidence-gated FR-11 trace replay labels, or
Exp 3200 materialized verifier-backed trace records when Exp 3215 is
unavailable
**When** Exp 3216 builds a small grounded-continuation trace graph
**Then** it SHALL load Exp 3215 first, fall back to Exp 3200 only when Exp 3215
is missing or not terminal, and record the chosen source artifact and fallback
status.
**And** it SHALL convert a bounded trace set into graph records for claims,
evidence, retractions, and repairs with dependency edges that make stale
premise propagation auditable.
**And** stale-premise invalidations SHALL propagate from retraction nodes to
dependent claim and repair nodes, and the artifact SHALL report the affected
routes.
**And** it SHALL define a virtual nonforgetting queue for held-out and drift
regression pressure, evaluate proposed controller-memory changes against that
budget, and keep the run controller-memory only.
**And** it SHALL NOT update model weights, SHALL NOT promote KAN sidecars, and
SHALL NOT modify `scripts/research_conductor.py` or the active roadmap.

The terminal artifact SHALL be
`results/experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.json`
and SHALL include `schema_version`, `experiment_id`, `milestone`,
`continuous_self_learning_task`, `source_trace_artifact`,
`trace_graph_node_count`, `trace_graph_edge_count`,
`stale_premise_invalidations`, `affected_route_count`,
`nonforgetting_queue_defined`, `nonforgetting_queue_value`,
`nonforgetting_budget_exceeded`, `model_weight_update_claimed`,
`controller_memory_promotion_allowed`, `conductor_file_modified`,
`active_roadmap_modified`, and `honest_verdict`.

### REQ-LEARN-3216 Sub-requirements

- REQ-LEARN-3216-1: Source selection SHALL prefer
  `results/experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.json`
  and SHALL fall back to
  `results/experiment_3200_fr11_verify_trace_memory_controller_v1.json` only
  when Exp 3215 is missing or not terminal.
- REQ-LEARN-3216-2: Each selected trace SHALL emit graph records for at least a
  claim node and an evidence node, and traces with rollback, retraction, or
  failed exact replay SHALL emit a retraction node and a repair node.
- REQ-LEARN-3216-3: Graph edges SHALL connect claims to evidence, claims to
  route nodes, retractions to stale premises, and repairs to retractions so
  stale-premise invalidations can be propagated without replaying the full
  transcript.
- REQ-LEARN-3216-4: The virtual nonforgetting queue SHALL accumulate held-out,
  drift, negative-control, and stale-route regression pressure, SHALL expose a
  numeric queue value, and SHALL set `nonforgetting_budget_exceeded=true` only
  when that queue exceeds the configured budget.
- REQ-LEARN-3216-5: The artifact SHALL set
  `model_weight_update_claimed=false`,
  `controller_memory_promotion_allowed=false`,
  `conductor_file_modified=false`, and `active_roadmap_modified=false`
  regardless of queue outcome because Exp 3216 is an audit/reporting artifact.

### SCENARIO-LEARN-3216: Grounded Graph And Queue Stay Inside Budget

**Given** a terminal Exp 3215 artifact with held-out and drift replay labels,
including one explicit retraction or failed exact replay
**When** Exp 3216 builds the trace graph and evaluates the virtual
nonforgetting queue
**Then** the artifact reports nonzero graph node and edge counts, propagates at
least one stale-premise invalidation to an affected route, defines a numeric
queue value, keeps the nonforgetting budget unexceeded when regression pressure
is within budget, denies controller-memory promotion, and emits an
`honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-3216-FALLBACK: Missing Exp 3215 Falls Back To Exp 3200

**Given** Exp 3215 is unavailable or not terminal but Exp 3200 is terminal
**When** Exp 3216 runs
**Then** it records Exp 3200 as `source_trace_artifact`, records that the
fallback was used, builds graph records from Exp 3200 trace records, defines
the queue, keeps model-weight update claims false, and emits an
`honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-LEARN-3216)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3216 | Pending (`python/carnot/eval/fr11_grounded_continuation_nonforgetting_queue_v1.py`) | Pending (`tests/python/test_experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.py`) |

## REQ-LEARN-3229: FR-11 Nonforgetting-Aware Controller Promotion Governance

**Given** Exp 3215 materialized evidence-gated FR-11 controller replay labels
and Exp 3216 materialized a grounded-continuation nonforgetting queue
**When** Exp 3229 simulates controller-memory promotion
**Then** it SHALL load both checked-in artifacts, identify all candidate replay
traces from Exp 3215, identify stale-premise invalidations and nonforgetting
queue pressure from Exp 3216, and run the replay simulation from those existing
artifacts only.
**And** it SHALL admit only evidence-backed traces that pass held-out/drift
coverage checks, negative-control checks, stale-premise checks, and the
nonforgetting queue budget.
**And** it SHALL reject stale-premise regressions and negative-control
regressions, defer otherwise useful traces when the nonforgetting queue budget
is exceeded, and report accepted, rejected, and deferred trace records.
**And** it SHALL define rollback triggers for negative-control regressions,
stale-premise failures, and contradiction graph updates.
**And** it SHALL keep controller-memory promotion separate from model training:
controller-memory updates may be allowed for admitted traces, but
`model_weight_update_claimed=false` and no training, fine-tuning, hidden weight
mutation, or conductor/roadmap mutation may be claimed.

The terminal artifact SHALL be
`results/experiment_3229_fr11_nonforgetting_promotion_controller_v3.json`
and SHALL include `schema_version`, `experiment_id`, `milestone`,
`continuous_self_learning_task`, `source_trace_artifacts`,
`candidate_trace_count`, `accepted_trace_count`, `rejected_trace_count`,
`deferred_trace_count`, `promotion_allowed`,
`controller_memory_promotion_allowed`, `nonforgetting_budget_exceeded`,
`rollback_policy_defined`, `rollback_trigger_count`,
`negative_control_regression_count`, `stale_premise_rejection_count`,
`model_weight_update_claimed`, `inference_substrate`,
`conductor_file_modified`, `active_roadmap_modified`, and `honest_verdict`.

### REQ-LEARN-3229 Sub-requirements

- REQ-LEARN-3229-1: Source selection SHALL require terminal, safe Exp 3215 and
  Exp 3216 artifacts and SHALL fail closed when either source is missing,
  non-terminal, or claims live inference or model-weight mutation.
- REQ-LEARN-3229-2: Candidate traces SHALL be the Exp 3215
  `replay_utility_labels`; each accepted trace SHALL preserve evidence label,
  replay role, reward weight, and route utility metadata.
- REQ-LEARN-3229-3: A trace SHALL be rejected when its route is named by Exp
  3216's affected routes, when it carries rollback/retraction status, when its
  reward weight is negative, or when negative-control regression pressure is
  nonzero.
- REQ-LEARN-3229-4: A trace that otherwise passes admission SHALL be deferred
  when the Exp 3216 nonforgetting queue exceeds budget; queue deferral SHALL
  NOT be reported as accepted promotion.
- REQ-LEARN-3229-5: `promotion_allowed` and
  `controller_memory_promotion_allowed` SHALL be true exactly when at least one
  trace is accepted, no negative-control regression is present, the
  nonforgetting budget is not exceeded, rollback policy is defined, and no
  model-weight update, conductor-file mutation, or active-roadmap mutation is
  claimed.
- REQ-LEARN-3229-6: The artifact SHALL expose rollback triggers named
  `negative_control_regression`, `stale_premise_failure`, and
  `contradiction_graph_update`, with `rollback_policy_defined=true` and
  `rollback_trigger_count=3`.

### SCENARIO-LEARN-3229: Accepted Promotion Excludes Stale Premises

**Given** terminal Exp 3215 and Exp 3216 artifacts where Exp 3215 contains
evidence-backed replay labels and Exp 3216 names stale affected routes while
keeping the nonforgetting queue within budget
**When** Exp 3229 runs the replay simulation
**Then** stale affected routes are rejected, the remaining admissible
evidence-backed traces are accepted, no trace is deferred, controller-memory
promotion is allowed for the accepted subset, model-weight update claims remain
false, and the artifact emits an `honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-3229-DEFERRED: Queue Budget Blocks Promotion Without Training

**Given** terminal Exp 3215 and Exp 3216 artifacts where admissible
evidence-backed traces exist but the Exp 3216 nonforgetting queue exceeds its
budget
**When** Exp 3229 runs the replay simulation
**Then** otherwise admissible traces are deferred, promotion is not allowed,
controller-memory promotion is not allowed, model-weight update claims remain
false, and the artifact emits an `honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-LEARN-3229)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3229 | Pending (`python/carnot/eval/fr11_nonforgetting_promotion_controller_v3.py`) | Pending (`tests/python/test_experiment_3229_fr11_nonforgetting_promotion_controller_v3.py`) |

## REQ-LEARN-3230: FR-11 KAN-CL Certificate Boundary Audit V2

**Given** Exp 3201 audited a KAN-CL-style FR-11 sidecar without granting
promotion authority and Exp 3216 materialized a grounded-continuation
nonforgetting queue
**When** Exp 3230 defines certificate boundaries for any future KAN-CL sidecar
promotion
**Then** it SHALL load both checked-in artifacts, enumerate the certificate
requirements needed before promotion, and map each requirement to current
artifact evidence or explicit missing evidence.
**And** the certificate requirements SHALL include bounded sidecar input
domains, per-knot nonforgetting budgets, per-knot monotonicity or Lipschitz
evidence, PWA/MILP abstraction readiness, nonforgetting budget checks, and
model-weight immutability.
**And** `kan_sidecar_promotion_allowed` SHALL be false unless every certificate
requirement is present, `per_knot_budget_defined=true`,
`pwa_milp_abstraction_ready=true`, `certificate_boundary_ready=true`, and no
model-weight, conductor-file, or active-roadmap mutation is claimed.
**And** the audit SHALL NOT train, fine-tune, update KAN weights, update base
model weights, or modify `scripts/research_conductor.py` or the active
roadmap.

The terminal artifact SHALL be
`results/experiment_3230_kan_cl_certificate_boundary_audit_v2.json` and SHALL
include `schema_version`, `experiment_id`, `milestone`,
`continuous_self_learning_task`, `source_artifacts`,
`certificate_requirements`, `requirement_evidence_matrix`,
`missing_certificate_count`, `per_knot_budget_defined`,
`pwa_milp_abstraction_ready`, `certificate_boundary_ready`,
`kan_sidecar_promotion_allowed`, `model_weight_update_claimed`,
`inference_substrate`, `conductor_file_modified`, `active_roadmap_modified`,
and `honest_verdict`.

### REQ-LEARN-3230 Sub-requirements

- REQ-LEARN-3230-1: Source selection SHALL require terminal, checked-in Exp
  3201 and Exp 3216 artifacts and SHALL fail closed when either artifact is
  missing, non-terminal, or claims live inference, model-weight mutation,
  conductor-file mutation, or active-roadmap mutation.
- REQ-LEARN-3230-2: `certificate_requirements` SHALL be a deterministic list
  of sidecar-promotion requirements with stable requirement IDs, descriptions,
  evidence keys, and promotion-critical flags.
- REQ-LEARN-3230-3: `requirement_evidence_matrix` SHALL include one row per
  certificate requirement with `evidence_status` equal to `present` or
  `missing`, concrete source artifact references, and an `evidence_summary`
  explaining the current artifact evidence.
- REQ-LEARN-3230-4: `missing_certificate_count` SHALL equal the number of
  evidence-matrix rows whose `evidence_status` is `missing`.
- REQ-LEARN-3230-5: `per_knot_budget_defined` SHALL be true only when the
  current artifacts expose a per-knot or per-template budget, not merely an
  aggregate nonforgetting queue.
- REQ-LEARN-3230-6: `pwa_milp_abstraction_ready` SHALL be true only when the
  current KAN sidecar artifact links bounded input domains, spline/PWA
  segments, error bounds, and MILP-compatible property checks for the sidecar.
- REQ-LEARN-3230-7: `certificate_boundary_ready` SHALL be true only when there
  are zero missing promotion-critical requirements and both
  `per_knot_budget_defined` and `pwa_milp_abstraction_ready` are true.
- REQ-LEARN-3230-8: The artifact SHALL set
  `model_weight_update_claimed=false`, `conductor_file_modified=false`,
  `active_roadmap_modified=false`, and
  `kan_sidecar_promotion_allowed=false` when certificate evidence is missing.

### SCENARIO-LEARN-3230: Missing Certificates Block KAN Sidecar Promotion

**Given** terminal Exp 3201 and Exp 3216 artifacts where aggregate
nonforgetting checks exist but bounded KAN input domains, per-knot budgets,
monotonicity/Lipschitz evidence, and sidecar-specific PWA/MILP abstractions are
not present
**When** Exp 3230 builds the certificate boundary audit
**Then** the artifact records the missing certificate rows, reports
`per_knot_budget_defined=false`, `pwa_milp_abstraction_ready=false`,
`certificate_boundary_ready=false`, keeps
`kan_sidecar_promotion_allowed=false`, and emits an `honest_verdict` beginning
with `complete:`.

### SCENARIO-LEARN-3230-BLOCKED: Unsafe Or Missing Sources Fail Closed

**Given** Exp 3201 or Exp 3216 is missing, non-terminal, or claims unsafe live
inference or mutation
**When** Exp 3230 runs
**Then** it writes all required fields, treats every promotion-critical
certificate as missing, keeps model-weight update claims false, blocks KAN
sidecar promotion, and emits an `honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-LEARN-3230)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3230 | Pending (`python/carnot/eval/fr11_kan_cl_certificate_boundary_audit_v2.py`) | Pending (`tests/python/test_experiment_3230_kan_cl_certificate_boundary_audit_v2.py`) |

## REQ-LEARN-3243: FR-11 Failure-Memory Controller Update

**Given** recent conductor-log entries and result artifacts expose missing
artifacts, repeated gate blocks, stale premises, certificate misses, and backend
failures
**When** Exp 3243 builds the FR-11 failure-memory controller update
**Then** it SHALL aggregate evidence from checked-in upstream artifacts and
`ops/conductor-log.md`, define a failure-memory schema with keys for
prerequisite, failure signature, stale premise, accepted next action, and
retirement risk, and score at least one heldout replay slice where the
controller avoids a doomed rerun or forces a gate.
**And** it SHALL record nonforgetting checks against prior FR-11 accepted and
rejected traces, reject stale premises, avoid repeated doomed reruns, and keep
controller-memory updates separate from model training.
**And** it SHALL set `model_weight_update_claimed=false` and
`controller_memory_updates_are_not_training=true`; no foundation-model,
KAN-sidecar, hidden-state, conductor-file, or active-roadmap mutation may be
claimed.

The terminal artifact SHALL be
`results/experiment_3243_fr11_failure_memory_controller_v1.json` and SHALL
include `experiment_id="exp3243"`,
`task_id="exp3243-fr11-failure-memory-controller-v1"`,
`milestone="2026.05.300"`,
`inference_substrate="aggregation_from_upstream_artifacts"`,
`principle_annotations`, `continuous_self_learning_task`,
`failure_memory_schema_ready`, `failure_trace_count`,
`heldout_replay_count`, `heldout_replay_delta`, `nonforgetting_delta`,
`stale_premise_rejection_count`, `doomed_rerun_avoidance_count`,
`model_weight_update_claimed`,
`controller_memory_updates_are_not_training`,
`fr11_controller_update_ready`, and `honest_verdict`.

### REQ-LEARN-3243 Sub-requirements

- REQ-LEARN-3243-1: The failure-memory schema SHALL expose the stable keys
  `prerequisite`, `failure_signature`, `stale_premise`,
  `accepted_next_action`, and `retirement_risk`.
- REQ-LEARN-3243-2: Failure traces SHALL include at least one missing-artifact
  trace, one repeated-gate-block trace, one stale-premise trace, and one backend
  failure trace extracted from the named upstream artifacts or conductor log.
- REQ-LEARN-3243-3: Heldout replay scoring SHALL count controller decisions
  that avoid a doomed rerun or force a prerequisite gate as positive deltas and
  SHALL expose `heldout_replay_delta > 0` when any doomed rerun is avoided.
- REQ-LEARN-3243-4: Nonforgetting checks SHALL cite prior FR-11 accepted and
  rejected traces and SHALL keep accepted-route retention nonnegative while
  preserving stale-premise rejections.
- REQ-LEARN-3243-5: `fr11_controller_update_ready` SHALL be true only when the
  schema is ready, failure traces and heldout replays are nonzero, stale
  premises were rejected, at least one doomed rerun was avoided, nonforgetting
  delta is nonnegative, and both no-training flags are set.
- REQ-LEARN-3243-6: `honest_verdict` SHALL begin with `complete:` and SHALL
  explicitly state that no model weights were updated.

### SCENARIO-LEARN-3243: Failure Memory Avoids Doomed Reruns

**Given** recent milestones contain repeated pre-gate failures for missing SOTA
receipt artifacts, repair-gate blockers, and CUDA backend smoke failures
**When** Exp 3243 scores a heldout replay slice for the controller memory
**Then** it records a positive heldout replay delta, avoids at least one doomed
rerun, forces a gate or accepted repair action for unsafe prerequisites,
preserves stale-premise rejections, keeps `model_weight_update_claimed=false`,
keeps `controller_memory_updates_are_not_training=true`, and emits an
`honest_verdict` beginning with `complete:` that says no model weights were
updated.

### SCENARIO-LEARN-3243-BLOCKED: Missing Evidence Fails Closed

**Given** the conductor log and upstream artifacts are missing or unreadable
**When** Exp 3243 builds the failure-memory update
**Then** it still writes the required schema fields, reports zero failure traces
and heldout replays, keeps readiness false, makes no training or model-weight
claim, and emits an `honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-LEARN-3243)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3243 | Pending (`python/carnot/eval/fr11_failure_memory_controller_v1.py`) | Pending (`tests/python/test_experiment_3243_fr11_failure_memory_controller_v1.py`) |

## REQ-LEARN-3255: FR-11 Lifelong Failure-Memory Retention Audit

**Given** Exp 3243 materialized a controller-only FR-11 failure-memory update,
Exp 3229 materialized nonforgetting-aware controller promotion governance, Exp
3215 materialized evidence-gated replay labels, and recent conductor-log rows
contain `.300` and `.301` gate outcomes
**When** Exp 3255 runs a LifelongAgentBench-style audit over those existing
artifacts and log rows
**Then** it SHALL map retention to preserved avoidance of previously doomed
reruns, adaptation to correct controller handling of new `.300` and `.301`
failure signatures, and forgetting to regressions on prior accepted FR-11
controller traces.
**And** it SHALL split the evidence into remembered, adapted, and held-out
negative-control slices with auditable trace IDs and source paths.
**And** it SHALL aggregate only checked-in artifact/log evidence, SHALL NOT call
a live LLM, and SHALL NOT train, fine-tune, mutate, or claim any
foundation-model, KAN-sidecar, hidden-state, conductor-file, or active-roadmap
update.

The terminal artifact SHALL be
`results/experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.json`
and SHALL include `experiment_id="exp3255"`,
`task_id="exp3255-fr11-lifelong-failure-memory-retention-audit-v1"`,
`milestone="2026.05.301"`,
`inference_substrate="aggregation_from_upstream_artifacts"`,
`principle_annotations`, `continuous_self_learning_task`,
`fr11_controller_update_ready`, `lifelong_eval_ready`,
`failure_trace_count`, `heldout_replay_count`, `retention_score`,
`adaptation_score`, `forgetting_score`,
`negative_control_regression_count`, `doomed_rerun_avoidance_count`,
`model_weight_update_claimed`, `no_new_llm_invoked`,
`reproducibility_checksum`, and `honest_verdict`.

### REQ-LEARN-3255 Sub-requirements

- REQ-LEARN-3255-1: The audit SHALL cite the LifelongAgentBench planning note
  and expose a mapping from retention, adaptation, and forgetting to Carnot
  artifact/gate traces.
- REQ-LEARN-3255-2: Failure traces SHALL include Exp 3243 controller-memory
  traces plus new `.300` selected-Python CUDA gate blocks and `.301`
  selected-Python/SOTA/prompt-injection gate signatures from checked-in
  artifacts or `ops/conductor-log.md`.
- REQ-LEARN-3255-3: The remembered slice SHALL be scored by preserved positive
  controller replay deltas on previously doomed reruns, while the adapted slice
  SHALL be scored by correct force-gate, repair-backend, or reject-stale actions
  for new `.300`/`.301` signatures.
- REQ-LEARN-3255-4: The held-out negative-control slice SHALL be sourced from
  prior FR-11 replay labels and SHALL contribute to
  `negative_control_regression_count`; regressions SHALL block
  `lifelong_eval_ready`.
- REQ-LEARN-3255-5: `forgetting_score` SHALL be one minus prior accepted-trace
  regression rate, using Exp 3229 accepted traces as the prior accepted FR-11
  slice; missing accepted traces SHALL fail closed with score `0.0`.
- REQ-LEARN-3255-6: `lifelong_eval_ready` SHALL be true only when the FR-11
  controller update is ready, remembered/adapted/negative-control slices are
  nonempty, retention/adaptation/forgetting scores are all `1.0`, no negative
  control regression exists, `model_weight_update_claimed=false`, and
  `no_new_llm_invoked=true`.
- REQ-LEARN-3255-7: `honest_verdict` SHALL begin with `complete:` and SHALL
  distinguish controller-memory updates from foundation-model learning.

### SCENARIO-LEARN-3255: Lifelong Audit Retains And Adapts Without Weight Updates

**Given** Exp 3243 reports ready failure-memory replay, Exp 3229 reports zero
negative-control regressions, Exp 3215 contains held-out negative-control
labels, and conductor-log rows expose `.300` and `.301` gate blocks
**When** Exp 3255 builds the lifelong audit
**Then** remembered, adapted, and held-out negative-control slices are nonempty,
retention, adaptation, and forgetting scores equal `1.0`,
`negative_control_regression_count=0`, `model_weight_update_claimed=false`,
`no_new_llm_invoked=true`, `lifelong_eval_ready=true`, and `honest_verdict`
begins with `complete:`.

### SCENARIO-LEARN-3255-BLOCKED: Missing Or Regressive Evidence Fails Closed

**Given** the upstream artifacts are missing, nonterminal, or contain negative
control regressions
**When** Exp 3255 builds the lifelong audit
**Then** it still writes all required schema fields, keeps
`model_weight_update_claimed=false`, keeps `no_new_llm_invoked=true`, sets
`lifelong_eval_ready=false`, and emits an `honest_verdict` beginning with
`complete:`.

## Implementation Status (REQ-LEARN-3255)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3255 | Implemented (`python/carnot/eval/fr11_lifelong_failure_memory_retention_audit_v1.py`) | Implemented (`tests/python/test_experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.py`) |

## REQ-LEARN-3278: FR-11 Full-Corpus Continual Self-Learning Audit

**Given** Exp 3272 has assembled the prompt-injection v4 full 15k corpus,
Garak/adaptive rows may be available in frozen splits, and legacy FR-11
failure-memory artifacts contain repair/SOTA gate-block traces
**When** Exp 3278 audits continual self-learning over those sources
**Then** it SHALL first check `full_15k_corpus_ready`; if that precondition is
false or absent, it SHALL write a complete gated-skip artifact without claiming
learning readiness.
**And** when ready, it SHALL build a controller-memory stream from prompt-
injection failure rows, Garak/adaptive rows when present, and legacy gate-block
traces; evaluate before/after controller-memory decisions on held-out traces;
and report retention, adaptation, forgetting, and negative transfer without
updating or claiming updates to foundation-model weights.

The terminal artifact SHALL be
`results/experiment_3278_fr11_full_corpus_continual_self_learning_audit_v1.json`
and SHALL include `continuous_self_learning_task`,
`fr11_full_corpus_audit_ready`, `controller_memory_only`,
`foundation_weight_updates_performed`, `retention_score`,
`adaptation_score`, `forgetting_rate`, `negative_transfer_rate`,
`heldout_trace_count`, `rollback_policy`, `output_paths`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `honest_verdict`.

### REQ-LEARN-3278 Sub-requirements

- REQ-LEARN-3278-1: The audit SHALL treat
  `results/experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1.json`
  as the readiness gate and SHALL fail closed when `full_15k_corpus_ready` is
  not true.
- REQ-LEARN-3278-2: The failure stream SHALL include prompt-injection rows from
  frozen v4 train/eval data, Garak/adaptive rows when available, and legacy
  FR-11 repair/SOTA gate-block traces from Exp 3243 and/or Exp 3255 artifacts.
- REQ-LEARN-3278-3: The before/after evaluation SHALL use held-out prompt-
  injection rows and held-out legacy controller-memory traces, not training-
  eligible rows only.
- REQ-LEARN-3278-4: `retention_score` SHALL measure preservation of older
  gate-block controller behavior, `adaptation_score` SHALL measure improved
  handling of new prompt-injection failures, `forgetting_rate` SHALL equal
  `1.0 - retention_score`, and `negative_transfer_rate` SHALL measure held-out
  benign prompt rows newly overblocked by the controller-memory update.
- REQ-LEARN-3278-5: The artifact SHALL include a `rollback_policy` with metric
  thresholds and a `rollback_required` boolean, and unsafe updates SHALL be
  rollback-required if retention drops, adaptation is too low, or negative
  transfer exceeds the threshold.
- REQ-LEARN-3278-6: `controller_memory_only` SHALL be true,
  `foundation_weight_updates_performed` SHALL be false, and
  `honest_verdict` SHALL begin with `complete:`, `success:`, `passed:`, or
  `shipped:` while attesting that no foundation-model weights were updated.

### SCENARIO-LEARN-3278: Full-Corpus Controller Memory Retains And Adapts

**Given** Exp 3272 is ready, frozen v4 splits are present, Garak/adaptive rows
are available, and legacy FR-11 memory artifacts contain held-out gate traces
**When** Exp 3278 builds the continual self-learning audit
**Then** it writes the required schema fields, sets
`continuous_self_learning_task=true`, sets `controller_memory_only=true`, keeps
`foundation_weight_updates_performed=false`, reports nonzero
`heldout_trace_count`, computes retention/adaptation/forgetting/negative-
transfer metrics, emits rollback criteria, and writes an `honest_verdict`
beginning with `complete:`.

### SCENARIO-LEARN-3278-BLOCKED: Missing Full Corpus Writes Complete Gated Skip

**Given** Exp 3272 is absent or `full_15k_corpus_ready` is not true
**When** Exp 3278 runs
**Then** it writes a schema-complete gated-skip artifact with
`fr11_full_corpus_audit_ready=false`, `controller_memory_only=true`,
`foundation_weight_updates_performed=false`, zero held-out traces, a rollback
policy, and an `honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-LEARN-3278)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3278 | Implemented (`python/carnot/eval/fr11_full_corpus_continual_self_learning_audit_v1.py`) | Implemented (`tests/python/test_experiment_3278_fr11_full_corpus_continual_self_learning_audit_v1.py`) |

## REQ-LEARN-3291: FR-11 Garak And Abstention Memory Replay

**Given** Exp 3278 produced a controller-memory-only FR-11 full-corpus
baseline, and milestone .304 may contain Garak/toolchain blockers, Garak
red-team evidence, clean-verifier abstention root-cause evidence, KAN boundary
decisions, and repair-gate outcomes
**When** Exp 3291 replays controller routing over prior held-out routes and the
new .304 blocker/evidence traces
**Then** it SHALL preserve raw learning episodes before any summary memory,
SHALL evaluate retention, adaptation, forgetting, and negative transfer, and
SHALL NOT update or claim updates to foundation-model weights.
**And** missing or gate-blocked .304 artifacts SHALL be converted into raw
learning episodes instead of silently dropping the category.

The terminal artifact SHALL be
`results/experiment_3291_fr11_garak_abstention_memory_replay_v1.json` and SHALL
include `continuous_self_learning_task`,
`fr11_garak_abstention_memory_replay_ready`, `controller_memory_only`,
`foundation_weight_updates_performed`, `raw_episodes_preserved`,
`new_episode_count`, `heldout_trace_count`, `retention_score`,
`adaptation_score`, `forgetting_rate`, `negative_transfer_rate`,
`memory_update_policy`, `blocked_trace_categories`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `honest_verdict`.

### REQ-LEARN-3291 Sub-requirements

- REQ-LEARN-3291-1: The replay SHALL load
  `results/experiment_3278_fr11_full_corpus_continual_self_learning_audit_v1.json`
  as the baseline and SHALL fail closed when it is missing, not
  controller-memory-only, or claims foundation-weight updates.
- REQ-LEARN-3291-2: The raw episode set SHALL include at least the .304
  categories `garak_toolchain`, `garak_redteam`, `clean_verifier_abstention`,
  `kan_boundary`, and `repair_gate` whenever their source artifacts or
  gate-block records are present; absent categories SHALL produce
  `missing_artifact` episodes.
- REQ-LEARN-3291-3: `raw_episodes_preserved` SHALL be true only when the
  artifact includes one JSON-safe raw episode per learned .304 blocker or
  evidence row before any consolidated summary memory is admitted.
- REQ-LEARN-3291-4: `memory_update_policy` SHALL explicitly allow only
  controller-route keys derived from raw artifacts, require replay gates before
  consolidation, and forbid foundation-model, hidden-state, and KAN sidecar
  weight updates.
- REQ-LEARN-3291-5: `retention_score` SHALL measure replay preservation for the
  Exp 3278 prior held-out denominator, `adaptation_score` SHALL measure the
  fraction of new .304 episodes routed to their expected controller action,
  `forgetting_rate` SHALL equal `1.0 - retention_score`, and
  `negative_transfer_rate` SHALL remain zero unless new memory would alter an
  unrelated prior route.
- REQ-LEARN-3291-6: `fr11_garak_abstention_memory_replay_ready` SHALL be true
  only when raw episodes are preserved, at least one new episode and one prior
  held-out trace are replayed, the update remains controller-memory-only,
  adaptation is nonzero, and negative transfer remains within policy.

### SCENARIO-LEARN-3291: .304 Blockers Become Safe Controller Memory

**Given** Exp 3278 is a valid controller-memory-only baseline and .304 Garak,
abstention, KAN boundary, and repair-gate artifacts are available
**When** Exp 3291 builds the replay artifact
**Then** it writes the required schema fields, preserves raw episodes for every
new category, keeps `foundation_weight_updates_performed=false`, reports
nonzero `new_episode_count` and `heldout_trace_count`, computes retention,
adaptation, forgetting, and negative-transfer metrics, and emits an
`honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-3291-BLOCKED: Missing Baseline Fails Closed

**Given** the Exp 3278 baseline is absent or unsafe
**When** Exp 3291 runs
**Then** it writes a schema-complete artifact with
`fr11_garak_abstention_memory_replay_ready=false`, preserves any raw .304
episodes that can be loaded, keeps `controller_memory_only=true`,
`foundation_weight_updates_performed=false`, zero prior held-out traces, and an
`honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-LEARN-3291)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3291 | Implemented (`python/carnot/eval/fr11_garak_abstention_memory_replay_v1.py`) | Implemented (`tests/python/test_experiment_3291_fr11_garak_abstention_memory_replay_v1.py`) |

## REQ-LEARN-3304: FR-11 Red-Team Defense Repair Memory Replay

**Given** Exp 3291 produced a controller-memory-only FR-11 baseline, milestone
.305 contains Garak red-team/defense evidence from Exp 3299 and Exp 3300, and
the Exp 3302 repair panel may exist
**When** Exp 3304 replays those red-team, defense, and repair episodes through
the FR-11 controller-memory loop
**Then** it SHALL preserve raw episodes, gate consolidation on held-out replay
and negative-transfer checks, report retention, adaptation, forgetting, and
negative transfer, and SHALL NOT update or claim updates to foundation-model
weights.

The terminal artifact SHALL be
`results/experiment_3304_fr11_redteam_repair_memory_replay_v2.json` and SHALL
include `fr11_redteam_repair_memory_replay_ready`,
`continuous_self_learning_task`, `new_episode_count`,
`raw_episode_preservation_path`, `heldout_trace_count`, `retention_score`,
`adaptation_score`, `forgetting_rate`, `negative_transfer_rate`,
`consolidation_gate_passed`, `controller_memory_only`,
`foundation_weight_updates_performed`, `learned_policy_updates`,
`inference_substrate`, `random_seed`, `reproducibility_checksum`,
`duration_s`, and `honest_verdict`.

### REQ-LEARN-3304 Sub-requirements

- REQ-LEARN-3304-1: The replay SHALL load
  `results/experiment_3291_fr11_garak_abstention_memory_replay_v1.json` as the
  baseline and SHALL fail closed when it is missing, not controller-memory-only,
  or claims foundation-weight updates.
- REQ-LEARN-3304-2: The raw episode set SHALL include Garak family pass/fail
  rows from Exp 3300, the selected defense policy from Exp 3299, the Exp 3302
  repair manifest when present, and Exp 3302 repair outcome rows when present.
- REQ-LEARN-3304-3: `raw_episode_preservation_path` SHALL identify the artifact
  path that embeds the complete raw episode list before summary memory is
  consolidated.
- REQ-LEARN-3304-4: `learned_policy_updates` SHALL be derived only from raw
  episodes, SHALL be inspectable as controller-policy updates, and SHALL forbid
  foundation-model, hidden-state, and KAN sidecar weight updates.
- REQ-LEARN-3304-5: `retention_score` SHALL preserve the Exp 3291 held-out
  replay denominator, `adaptation_score` SHALL measure routing coverage over
  new .305 episodes, `forgetting_rate` SHALL equal `1.0 - retention_score`, and
  `negative_transfer_rate` SHALL reflect unrelated aligned-benign false
  positives introduced by the new memory.
- REQ-LEARN-3304-6: `consolidation_gate_passed` and
  `fr11_redteam_repair_memory_replay_ready` SHALL be true only when raw
  episodes are preserved, held-out and new replay are non-empty,
  `retention_score >= 0.95`, `adaptation_score > 0`, negative transfer is at
  most `0.05`, and `foundation_weight_updates_performed=false`.

### SCENARIO-LEARN-3304: .305 Evidence Becomes Controller Memory

**Given** Exp 3291 is a valid controller-memory-only baseline and Exp 3299,
Exp 3300, and Exp 3302 artifacts are available
**When** Exp 3304 builds the replay artifact
**Then** it writes the required schema fields, preserves raw Garak family,
selected defense, repair manifest, and repair outcome episodes, keeps
`foundation_weight_updates_performed=false`, reports nonzero
`new_episode_count` and `heldout_trace_count`, computes retention, adaptation,
forgetting, and negative-transfer metrics, and emits an `honest_verdict`
beginning with `complete:`.

### SCENARIO-LEARN-3304-BLOCKED: Missing Baseline Fails Closed

**Given** the Exp 3291 baseline is absent or unsafe
**When** Exp 3304 runs
**Then** it writes a schema-complete artifact with
`fr11_redteam_repair_memory_replay_ready=false`, preserves any .305 raw
episodes that can be loaded, keeps `controller_memory_only=true`,
`foundation_weight_updates_performed=false`, zero prior held-out traces, and an
`honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-LEARN-3304)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3304 | Implemented (`python/carnot/eval/fr11_redteam_repair_memory_replay_v2.py`) | Implemented (`tests/python/test_experiment_3304_fr11_redteam_repair_memory_replay_v2.py`) |

## REQ-LEARN-3334: FR-11 Online Verifier Memory Nonforgetting

### REQ-LEARN-3334 Sub-requirements
- REQ-LEARN-3334-1: The experiment SHALL load verified cases from clean .308 artifacts when available, with fallback to stable .305-.306 cached verifier evidence.
- REQ-LEARN-3334-2: The data SHALL be split into update stream, new-task evaluation, and old-task holdout.
- REQ-LEARN-3334-3: The artifact SHALL record every accepted update, rejected update, and rollback trigger.
- REQ-LEARN-3334-4: The artifact SHALL measure `new_task_delta`, `old_task_delta`, `false_positive_delta`, and `rollback_count`.
- REQ-LEARN-3334-5: `fr11_nonforgetting_ready` SHALL be true only if updates improve or preserve new-task performance while old-task degradation stays within tolerance.
- REQ-LEARN-3334-6: The artifact SHALL write fields: `honest_verdict`, `inference_substrate`, `random_seed`, `reproducibility_checksum`, `duration_s`, `n_update_cases`, `n_new_eval_cases`, `n_old_holdout_cases`, `new_task_delta`, `old_task_delta`, `false_positive_delta`, `rollback_count`, `fr11_nonforgetting_ready`, `blocked_reasons`.

## Implementation Status (REQ-LEARN-3334)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3334 | Implemented (`python/carnot/eval/fr11_online_verifier_memory_nonforgetting_v4.py`) | Implemented (`tests/python/test_experiment_3334_fr11_online_verifier_memory_nonforgetting_v4.py`) |

## REQ-LEARN-080: FR-11 Online Verifier-Memory Nonforgetting v5
**Given** a continuous self-learning stream with verifier outcomes
**When** memory controller applies online updates from exact verifier outcomes
**Then** the experiment must measure new_task_delta, old_task_delta, false_positive_delta, soundness_error_delta, completeness_error_delta, rollback_count, and rollback_recovered_count
**And** it must rollback harmful updates preserving old-task retention
**And** fr11_nonforgetting_ready must only be true if new-task improves or preserves, old-task degradation stays in bounds, false positives don't increase, and rollback works on harmful updates.

### SCENARIO-LEARN-120: Rollback Prevents Old-Task Catastrophic Forgetting
**Given** an online update that degrades old-task performance beyond planned bounds
**When** the controller detects the harmful update
**Then** it reverts the memory state
**And** increments rollback_count and rollback_recovered_count.

## REQ-LEARN-3357: FR-11 LogicVault Z3 Integration
**Given** an incoming fact and a set of historically verified beliefs
**When** the vault processes the incoming fact
**Then** it MUST use Z3 to check if the new fact is consistent with historical beliefs
**And** track `ledger_consistency_rate`.

### SCENARIO-LEARN-3357: Consistent Facts Admitted
**Given** a LogicVault with axiom `x > 0`
**When** we check and admit `x > 5`
**Then** it is consistent (SAT), it is admitted, and `ledger_consistency_rate` is updated.

### SCENARIO-LEARN-3357-REJECT: Contradictory Facts Rejected
**Given** a LogicVault with axiom `x > 5`
**When** we check and admit `x < 0`
**Then** it is inconsistent (UNSAT), it is rejected, and `ledger_consistency_rate` is updated.

## Implementation Status (REQ-LEARN-3357)
| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3357 | Implemented (`python/carnot/pipeline/session_memory.py`, `scripts/experiment_3357_fr11_logicvault.py`) | Implemented (`tests/python/test_experiment_3357_fr11_logicvault.py`) |

## REQ-LEARN-3373: FR-11 LogicVault Concurrent Agent Beliefs
**Given** multiple agents processing facts simultaneously
**When** the vault processes incoming facts from different agents
**Then** it MUST maintain separate Z3 solvers and belief states for each agent via an `agent_id` parameter
**And** track `ledger_consistency_rate` per agent independently.

### SCENARIO-LEARN-3373: Concurrent Independent Beliefs
**Given** two agents, "agent_A" and "agent_B"
**When** "agent_A" adds `x > 0` and checks `x > 5`
**And** "agent_B" adds `y < 0` and checks `y < -5`
**Then** both facts are admitted in their respective vaults without interfering
**And** `ledger_consistency_rate` is tracked correctly per agent.

## Implementation Status (REQ-LEARN-3373)
| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3373 | Pending | Pending |


## REQ-LEARN-3395: FR-11 Energy-Guided Sample Selection
**Given** a memory buffer of constraint violations
**When** the FR-11 continual update loop selects samples for replay
**Then** it MUST use the Ising energy difference to prioritize the most critical samples
**And** it MUST measure a nonforgetting metric comparing the energy-guided selection against random replay.

### SCENARIO-LEARN-3395: Energy Difference Guides Critical Sample Selection
**Given** multiple constraint violations in the memory buffer
**When** the Ising energy difference is computed for each sample
**Then** the samples with the largest energy difference are selected for replay
**And** the nonforgetting metric shows improvement or parity compared to random selection.

## Implementation Status (REQ-LEARN-3395)
| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3395 | Implemented (`scripts/experiment_3395_energy_based_replay.py`) | Implemented (`tests/python/test_experiment_3395_energy_based_replay.py`) |

## REQ-LEARN-3399: LogicVault CDCL Long Context Verification
**Given** a long multi-turn context (e.g. 16k tokens) generated via `cached_sota_pair`
**When** new facts and axioms are checked for consistency
**Then** the LogicVault CDCL logic MUST identify contradictions and track learned clauses
**And** it MUST emit a verdict of "contradiction_caught" when an inconsistent statement is parsed.

### SCENARIO-LEARN-3399-A: CDCL Identifies Contradictions
**Given** accumulated logical axioms from a multi-turn chat
**When** the agent issues a contradicting claim
**Then** LogicVault CDCL catches the contradiction
**And** the check returns False.

## Implementation Status (REQ-LEARN-3399)
| Requirement | Implementation | Tests |
|---|---|---|
| REQ-LEARN-3399 | Implemented (`scripts/experiment_3399_logicvault_long_context.py`) | Implemented (`tests/python/test_experiment_3399_logicvault_long_context.py`) |

## REQ-LEARN-030: Latent Energy Spills as Reward Signal

**Given** the FR-11 continuous self-learning loop
**When** inference is run using `MODEL_SPECS = ["unsloth/Qwen3.6-35B-A3B-GGUF"]`
**Then** the spill detection algorithm identifies when the model relies on language priors
**And** failed examples are replayed and constraint templates updated using Latent Energy Spill values as priority weights
**And** the script `scripts/experiment_3410_fr11_updates_spills.py` outputs a retention and adaptation score to `results/experiment_3410_fr11_updates_spills.json`.

### SCENARIO-LEARN-030: Spills Driven Constraint Update

**Given** a set of model outputs
**When** latent energy spills are detected
**Then** the constraint templates are updated weighted by the spill values, correctly computing adaptation and retention scores.

## REQ-FR11-BML-001: Beta-Min Lambda-Min Decision-Covariance Measurement

The system MUST compute the k×k binary decision covariance matrix Sigma from at least
3 grounding configurations (varying ACTIVE_WEIGHT) on the cached trace corpus,
yielding lambda_min(Sigma), effective_k (participation ratio), and pairwise_max_correlation
per configuration.

**Implementation status:** In progress (exp3498).
**Spec traces:** REQ-FR11-BML-001

## REQ-FR11-BML-002: Beta-Min Sweep per Grounding Configuration

For each grounding configuration defined by REQ-FR11-BML-001, the system MUST sweep
entropy beta in {0, 0.1, 0.25, 0.5} on the FR-11 self-improvement loop at N>=200
iterations, identifying the minimal-sufficient entropy beta that prevents mode-collapse.

**Implementation status:** In progress (exp3498).
**Spec traces:** REQ-FR11-BML-002

## REQ-FR11-BML-003: Predictive Law beta_min ~ f(lambda_min)

The system MUST fit a predictive law beta_min ~ f(lambda_min) across >=3 grounding
configurations and validate it via leave-one-config-out cross-validation. A valid
law enables the Phase-5 deployment formula: measure lambda_min(Sigma), set
beta >= f(lambda_min) with safety margin.

**Implementation status:** In progress (exp3498).
**Spec traces:** REQ-FR11-BML-003

### SCENARIO-FR11-BML-001: At-Risk Config Requires Higher Beta

**Given** a grounding configuration with low lambda_min (high null-space dominance)
**When** the beta sweep is run at N=200
**Then** the minimal sufficient beta is higher than for a well-grounded configuration

### SCENARIO-FR11-BML-002: Law Holds Out-of-Sample

**Given** a predictive law fitted on 3 grounding configurations
**When** the law is applied to a held-out 4th configuration
**Then** the predicted beta_min is within 0.15 of the actual beta_min

### SCENARIO-FR11-BML-003: Gaming Signal Confirmed at Beta=0

**Given** any grounding configuration at beta=0
**When** the FR-11 self-improvement loop runs for N=200 iterations
**Then** verifier pass_rate >> true_accuracy (gaming confirmed from distinct sources)

---

## REQ-FR11-CLD-001: Lambda_min Measured and Beta Deployed from exp3498 Formula

**Given** a fresh grounding configuration (ACTIVE_WEIGHT not in the exp3498 fit set)
**When** the closed-loop deployment experiment runs
**Then** the system measures lambda_min(Sigma) from the k×k verifier-decision covariance
**And** computes beta_deployed = max(0, slope * lambda_min + intercept) using the
        fitted law coefficients from exp3498 (slope=1.8461, intercept=-0.3001)

**Implementation status:** In progress (exp3509).
**Spec traces:** REQ-FR11-CLD-001

## REQ-FR11-CLD-002: Three-Arm Closed Loop at N>=200 with Distinct Metrics

**Given** a fresh grounding configuration with measured lambda_min and beta_deployed
**When** the FR-11 self-improvement loop runs to N>=200
**Then** three arms are executed: Arm A (beta=f(lambda_min)), Arm B (beta=0), Arm C (beta=0.5)
**And** pass_rate and true_accuracy are verified element-wise distinct before each arm

**Implementation status:** In progress (exp3509).
**Spec traces:** REQ-FR11-CLD-002

## REQ-FR11-CLD-003: Deployment Validated: Arm A Prevents Collapse, Arm B Collapses

**Given** three-arm results across >=2 fresh grounding configurations
**When** collapse is evaluated at N=200 depth (depth-aware criterion)
**Then** Arm A (deployed law) shows collapse_detected=False at ALL fresh configs
**And** Arm B (beta=0 control) shows collapse_detected=True at >=1 fresh config

**Implementation status:** In progress (exp3509).
**Spec traces:** REQ-FR11-CLD-003

### SCENARIO-FR11-CLD-001: Deployed Law Prevents Collapse at Fresh Configs

**Given** fresh grounding configs with AW not in {0.05, 0.10, 0.146, 0.30}
**When** Arm A runs with beta = f(lambda_min) from exp3498 law
**Then** collapse_detected is False at all fresh configs (law generalizes to deployment)

### SCENARIO-FR11-CLD-002: Beta=0 Control Collapses

**Given** the same fresh grounding configs as SCENARIO-FR11-CLD-001
**When** Arm B runs with beta=0
**Then** collapse_detected is True at >=1 config (proves the loop CAN collapse)

### SCENARIO-FR11-CLD-003: Arm A Accuracy Not Materially Worse Than Arm C

**Given** three-arm results including fixed-conservative Arm C (beta=0.5)
**When** true_accuracy is compared between Arm A and Arm C
**Then** mean(accuracy_A - accuracy_C) >= -0.001 (law not over-regularizing)


## REQ-FR11-AOB-001: Adaptive Online Beta Measurement

**Given** a running FR-11 self-improvement loop
**When** evaluated at each step
**Then** lambda_min MUST be measured online using the current trace probability distribution to compute the weighted decision covariance.

**Implementation status:** Implemented (exp3521).
**Spec traces:** REQ-FR11-AOB-001

## REQ-FR11-AOB-002: Four-Arm Adaptive Deployment Loop

**Given** a verifier ensemble and cached traces
**When** running the self-learning loop for >=200 iterations
**Then** it MUST run four arms: Arm A (Adaptive Online), Arm B (beta=0), Arm C (conservative beta=0.5), and Arm D (static offline law).

**Implementation status:** Implemented (exp3521).
**Spec traces:** REQ-FR11-AOB-002

## REQ-FR11-AOB-003: Adaptive Rule Validated Against Conservative

**Given** the four-arm results
**When** checked across >=4 fresh configs
**Then** either the Adaptive Online rule OR the Conservative Default MUST prevent collapse at all configs while Arm B collapses at >=1 config.

**Implementation status:** Implemented (exp3521).
**Spec traces:** REQ-FR11-AOB-003

### SCENARIO-FR11-AOB-001: Adaptive Rule Robustly Prevents Collapse

**Given** 5 fresh grounding configs
**When** Arm A runs with beta = clamp(1.8461 * lambda_min_online - 0.3001, 0.0, 0.5)
**Then** collapse_detected is False at all configs.

## REQ-LEARN-3580: FR-11 Continuous Self Learning v4 Forward Difference

**Given** a FR-11 module with cached traces present
**When** the continuous self-learning experiment (Exp 3580) runs on a fresh nondegenerate corpus
**Then** the deploy arm MUST prevent collapse (collapse_detected_deploy_arm=False)
**And** the control arm MUST collapse (collapse_detected_control_beta0=True)
**And** pass_rate_vs_true_accuracy_distinct_assert MUST be True
**And** quality_maintained MUST be True
**And** honest_verdict MUST be "complete: fr11_conservative_default_holds_on_fresh_nondegenerate_corpus_quality_maintained"

### REQ-LEARN-3580 Sub-requirements

- REQ-LEARN-3580-1: The experiment SHALL log inference_substrate, n_grounding_configs, random_seed, reproducibility_checksum, duration_s.

---

## REQ-LEARN-3590: FR-11 Continuous Self Learning v5 Grounding Calibration

**Given** the FR-11 module with a conservative-default beta and the factual-grounding verifier
**When** the self-learning loop is run on a fresh non-degenerate corpus
**Then** the deployed conservative-default beta MUST prevent self-distillation collapse
**And** the positive control (beta=0) MUST collapse
**And** the pass_rate and true_accuracy metrics MUST be distinct
**And** the factual verifier's calibration MUST improve
**And** overall detector quality MUST be maintained.

### REQ-LEARN-3590 Sub-requirements

- REQ-LEARN-3590-1: `evaluate_continuous_self_learning_v5` SHALL enforce distinct arrays for pass_rate and true_accuracy.
- REQ-LEARN-3590-2: The acceptance gate SHALL pass only if deploy arm holds, control arm collapses, and arrays are distinct.

## SCENARIO-LEARN-3590: Grounding Verifier Calibration Validation
**Given** the evaluation function with deploy arm collapse = False and control collapse = True
**When** the results are computed
**Then** `honest_verdict` MUST indicate the calibration holds and quality is maintained.


## REQ-LEARN-3604: FR-11 Continuous Self Learning v6 with Fallback Verifier

**Given** the FR-11 continuous self-learning loop and a fresh non-degenerate corpus
**When** the deployed conservative-default beta is used to calibrate the grounding verifier's decision threshold online
**Then** the deploy arm MUST prevent self-distillation collapse (collapse_detected_deploy_arm == False)
**And** the control arm (beta=0) MUST collapse (collapse_detected_control_beta0 == True)
**And** the pass_rate and true_accuracy arrays MUST be genuinely distinct (pass_rate_vs_true_accuracy_distinct_assert == True)
**And** detector quality MUST be maintained

### REQ-LEARN-3604 Sub-requirements

- REQ-LEARN-3604-1: If exp3600's real NLI grounding verifier landed, use it; else fall back to the existing factual verifier.
- REQ-LEARN-3604-2: honest_verdict MUST be "complete: fr11_conservative_default_calibrates_real_grounding_verifier_holds_quality_maintained" if the acceptance gate passes.
- REQ-LEARN-3604-3: The evaluation MUST measure detector calibration (e.g. factual_verifier_calibration_improved=True).

### SCENARIO-LEARN-3604: Successful Calibration Prevents Collapse

**Given** deploy arm without collapse, control arm with collapse, and distinct arrays for pass_rate vs true_accuracy
**When** evaluate_continuous_self_learning_v6 is called
**Then** honest_verdict == "complete: fr11_conservative_default_calibrates_real_grounding_verifier_holds_quality_maintained"

---

## Implementation Status (REQ-LEARN-3604)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3604 | Implemented (python/carnot/fr11/continuous_self_learning_v6.py) | Implemented (tests/python/test_experiment_3604_fr11_csl_v6.py) |

---

## REQ-LEARN-3647: FR-11 Continuous Self Learning v8 Online Correlation-Aware Weighting

**Given** the FR-11 module and cached verifier/candidate traces are present
**When** Exp 3647 runs a closed FR-11 continuous self-learning loop over at
least 200 cached examples from a fresh non-degenerate corpus
**Then** the deploy arm SHALL apply a conservative-default online update with
an uncertainty-gated redundancy penalty that down-weights highly conditionally
correlated verifiers
**And** the control arm SHALL run a naive beta=0 online update without the
redundancy penalty
**And** the artifact SHALL report deploy/control collapse status, before/after
AUROC, Brier/ECE calibration, the correlation-aware AUROC gain over
noise-only weighting, and a pass-rate-vs-true-accuracy distinctness assertion.

### REQ-LEARN-3647 Sub-requirements

- REQ-LEARN-3647-1: If Exp 3644's label-conditional verifier correlation
  matrix exists, the redundancy penalty SHALL seed from that matrix; otherwise
  the workflow SHALL compute the matrix inline from the cached scores.
- REQ-LEARN-3647-2: `n_online_updates` SHALL be at least 200 for runnable
  artifacts, and blocked artifacts SHALL use the terminal verdict
  `complete: blocked_fr11_module_or_traces_unavailable`.
- REQ-LEARN-3647-3: The acceptance gate SHALL pass only when
  `collapse_detected_deploy_arm == false`,
  `collapse_detected_control == true`, and
  `pass_rate_vs_true_accuracy_distinct_assert == true`.
- REQ-LEARN-3647-4: Runnable artifacts SHALL include principle annotations for
  `honest_verdict`, `inference_substrate`, `n_online_updates`,
  `collapse_detected_deploy_arm`, `collapse_detected_control`,
  `correlation_aware_auroc_gain`, `calibration_improved`,
  `pass_rate_vs_true_accuracy_distinct_assert`, `quality_maintained`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- REQ-LEARN-3647-5: The terminal verdict SHALL be
  `complete: fr11_v8_online_correlation_aware_weighting_holds_no_collapse_quality_maintained`
  when the acceptance gate passes and the redundancy penalty improves AUROC
  over noise-only weighting, otherwise
  `complete: fr11_v8_correlation_penalty_no_auroc_gain_noise_only_weighting_sufficient`
  when the gate passes without a positive correlation-aware AUROC gain.

### SCENARIO-LEARN-3647: Guarded Online Weighting Prevents Collapse

**Given** cached verifier scores with binary correctness labels and at least
one redundant verifier pair
**When** the v8 online self-learning evaluator compares the deploy arm against
the beta=0 control arm
**Then** the deploy arm does not collapse, the control arm collapses, the
pass-rate and true-accuracy trajectories are not element-wise identical, and
quality is maintained.

## Implementation Status (REQ-LEARN-3647)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3647 | Implemented (python/carnot/fr11/continuous_self_learning_v8.py; scripts/experiment_3647_fr11_continuous_self_learning_v8.py) | Implemented (tests/python/test_experiment_3647_fr11_csl_v8.py) |

---

## REQ-LEARN-3660: FR-11 Continuous Self Learning v9 Online Fusion Weighting

**Given** the FR-11 module and cached detector traces with both verifier
ensemble scores and confidence-derived error scores are present
**When** Exp 3660 runs a closed FR-11 continuous self-learning loop over at
least 200 non-degenerate cached examples
**Then** the deploy arm SHALL learn the fused-detector ensemble-vs-confidence
weight online per domain from observed catch-rate evidence
**And** the deploy arm SHALL apply a conservative default and uncertainty gate
that clips the learned fusion weight away from 0 and 1
**And** the control arm SHALL run a naive online fusion-weight update without
that guard so collapse can be detected as a positive control
**And** the artifact SHALL report fused-detector AUROC, Brier, and ECE before
and after online adaptation, online fusion AUROC gain over fixed 50/50 fusion,
deploy/control collapse status, quality maintenance, and a
pass-rate-vs-true-accuracy distinctness assertion.

### REQ-LEARN-3660 Sub-requirements

- REQ-LEARN-3660-1: Runnable artifacts SHALL use cached traces only, with
  `inference_substrate` equal to
  `verifier_ensemble_against_cached_candidates (principle: scores cached traces; no LLM load).`.
- REQ-LEARN-3660-2: If the FR-11 module or cached ensemble-plus-confidence
  traces are unavailable, the workflow SHALL write the terminal verdict
  `complete: blocked_fr11_module_or_traces_unavailable`.
- REQ-LEARN-3660-3: `n_online_updates` SHALL be at least 200 for runnable
  artifacts.
- REQ-LEARN-3660-4: The acceptance gate SHALL pass only when
  `collapse_detected_deploy_arm == false`,
  `collapse_detected_control == true`, and
  `pass_rate_vs_true_accuracy_distinct_assert == true`.
- REQ-LEARN-3660-5: Runnable artifacts SHALL include principle annotations for
  `honest_verdict`, `inference_substrate`, `n_online_updates`,
  `collapse_detected_deploy_arm`, `collapse_detected_control`,
  `online_fusion_auroc_gain`, `calibration_improved`,
  `pass_rate_vs_true_accuracy_distinct_assert`, `quality_maintained`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- REQ-LEARN-3660-6: The terminal verdict SHALL be
  `complete: fr11_v9_online_fusion_weighting_holds_no_collapse_quality_maintained`
  when the acceptance gate passes and online fusion improves AUROC over fixed
  50/50 fusion, otherwise
  `complete: fr11_v9_online_fusion_no_gain_fixed_fusion_sufficient`.

### SCENARIO-LEARN-3660: Guarded Online Fusion Weighting Prevents Collapse

**Given** cached ensemble and confidence scores with binary correctness labels
and at least one domain where one signal has higher observed catch utility
**When** the v9 online self-learning evaluator compares the guarded deploy arm
against the naive control arm
**Then** the deploy arm keeps every fusion weight bounded away from 0 and 1,
the control arm collapses to a boundary weight, pass-rate and true-accuracy
trajectories are not element-wise identical, and quality is maintained.

## Implementation Status (REQ-LEARN-3660)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3660 | Proposed (python/carnot/fr11/continuous_self_learning_v9.py; scripts/experiment_3660_fr11_continuous_self_learning_v9.py) | Proposed (tests/python/test_experiment_3660_fr11_csl_v9.py) |

---

## REQ-LEARN-3673: FR-11 Continuous Self Learning v10 Online Dependency-Aware Weighting

**Given** the FR-11 module and cached verifier traces with per-verifier scores
and binary labels are present
**When** Exp 3673 runs a closed FR-11 continuous self-learning loop over at
least 200 non-degenerate cached examples
**Then** the deploy arm SHALL learn the label-conditional dependency structure
and verifier weights online from observed catch-rate evidence
**And** the deploy arm SHALL begin from a conservative default, update only
after an uncertainty gate clears, and apply a collapse guard that prevents any
single verifier weight from concentrating at 0 or 1
**And** the control arm SHALL run a naive online update without that guard so
single-verifier collapse can be detected as a positive control
**And** the artifact SHALL report ensemble AUROC before and after online
adaptation, online dependency-aware AUROC gain over both fixed dependency-aware
weighting and Carnot static weighting, deploy/control collapse status, quality
maintenance, and a pass-rate-vs-true-accuracy distinctness assertion.

### REQ-LEARN-3673 Sub-requirements

- REQ-LEARN-3673-1: Runnable artifacts SHALL use cached traces only, with
  `inference_substrate` equal to
  `verifier_ensemble_against_cached_candidates (principle: scores cached traces; no LLM load).`.
- REQ-LEARN-3673-2: If the FR-11 module or cached per-verifier trace scores are
  unavailable, the workflow SHALL write the terminal verdict
  `complete: blocked_fr11_module_or_traces_unavailable`.
- REQ-LEARN-3673-3: `n_online_updates` SHALL be at least 200 for runnable
  artifacts.
- REQ-LEARN-3673-4: The acceptance gate SHALL pass only when
  `collapse_detected_deploy_arm == false`,
  `collapse_detected_control == true`, and
  `pass_rate_vs_true_accuracy_distinct_assert == true`.
- REQ-LEARN-3673-5: Runnable artifacts SHALL include principle annotations for
  `honest_verdict`, `inference_substrate`, `n_online_updates`,
  `collapse_detected_deploy_arm`, `collapse_detected_control`,
  `online_dependency_aware_auroc_gain`,
  `pass_rate_vs_true_accuracy_distinct_assert`, `quality_maintained`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- REQ-LEARN-3673-6: The terminal verdict SHALL be
  `complete: fr11_v10_online_dependency_aware_weighting_holds_no_collapse_quality_maintained`
  when the acceptance gate passes and the online dependency-aware ensemble
  improves AUROC over fixed dependency-aware and Carnot static weighting,
  otherwise
  `complete: fr11_v10_online_no_gain_fixed_weighting_sufficient`.

### SCENARIO-LEARN-3673: Online Dependency-Aware Weighting Prevents Collapse

**Given** cached verifier scores with binary correctness labels and at least one
verifier whose observed catch utility differs from the others
**When** the v10 online self-learning evaluator compares the guarded
dependency-aware deploy arm against the naive single-best-verifier control arm
**Then** the deploy arm keeps verifier weights bounded away from single-verifier
collapse, the control arm collapses to a boundary weight, pass-rate and
true-accuracy trajectories are not element-wise identical, and quality is
maintained.

## Implementation Status (REQ-LEARN-3673)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3673 | Proposed (python/carnot/fr11/continuous_self_learning_v10.py; scripts/experiment_3673_fr11_continuous_self_learning_v10.py) | Proposed (tests/python/test_experiment_3673_fr11_csl_v10.py) |

---

## REQ-LEARN-3685: FR-11 Continuous Self Learning v11 Drift-Aware Online Dependency Re-estimation

**Given** the FR-11 module and cached per-verifier trace scores with binary
labels are present, and the cached score stream contains two
distributionally-distinct verifier-vote slices
**When** Exp 3685 runs a closed FR-11 continuous self-learning loop over at
least 200 cached examples with an injected mid-stream verifier-vote
distribution shift
**Then** the deploy arm SHALL explicitly detect verifier-vote distribution
drift and re-estimate the dependency structure online after the drift point
**And** the deploy arm SHALL keep the v10 conservative default,
uncertainty-gated update rule, and collapse guard so no single verifier weight
concentrates at 0 or 1
**And** the control arm SHALL run a naive online update without drift detection
or the guard, so single-verifier collapse or adaptation failure is visible as a
positive control
**And** the artifact SHALL report pre-drift and post-drift AUROC for the deploy
arm, control arm, v10 fixed-structure online baseline, and static Carnot
baseline, plus the post-drift AUROC gain over v10, quality maintenance,
collapse status, drift-detection status, and pass-rate-vs-true-accuracy
distinctness.

### REQ-LEARN-3685 Sub-requirements

- REQ-LEARN-3685-1: Runnable artifacts SHALL use cached traces only, with
  `inference_substrate` equal to
  `verifier_ensemble_against_cached_candidates (principle: scores cached traces; no LLM load).`.
- REQ-LEARN-3685-2: If the FR-11 module, cached per-verifier trace scores, or
  two distributionally-distinct verifier-vote slices are unavailable, the
  workflow SHALL write the terminal verdict
  `complete: blocked_fr11_module_or_traces_unavailable`.
- REQ-LEARN-3685-3: `n_online_updates` SHALL be at least 200 for runnable
  artifacts.
- REQ-LEARN-3685-4: The acceptance gate SHALL pass only when
  `drift_detected_deploy_arm == true`,
  `collapse_detected_deploy_arm == false`,
  `collapse_detected_control == true`, and
  `pass_rate_vs_true_accuracy_distinct_assert == true`.
- REQ-LEARN-3685-5: Runnable artifacts SHALL include principle annotations for
  `honest_verdict`, `inference_substrate`, `n_online_updates`,
  `drift_detected_deploy_arm`, `collapse_detected_deploy_arm`,
  `collapse_detected_control`, `post_drift_auroc_gain_over_v10`,
  `pass_rate_vs_true_accuracy_distinct_assert`, `quality_maintained`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- REQ-LEARN-3685-6: The terminal verdict SHALL be
  `complete: fr11_v11_drift_aware_online_dependency_aware_recovers_no_collapse_quality_maintained`
  when the acceptance gate passes, quality is maintained, and the drift-aware
  deploy arm improves post-drift AUROC over the v10 fixed-structure online
  baseline, otherwise
  `complete: fr11_v11_no_gain_over_v10_fixed_structure_online_sufficient`.

### SCENARIO-LEARN-3685: Drift-Aware Re-estimation Recovers Without Collapse

**Given** cached verifier scores whose first slice and second slice have
different verifier-vote distributions
**When** the v11 evaluator streams the first slice before the second slice
**Then** the deploy arm detects the mid-stream drift, re-learns a dependency
structure using post-drift evidence, keeps verifier weights bounded away from
single-verifier collapse, and reports pass-rate and true-accuracy trajectories
that are not element-wise identical
**And** the naive control arm collapses to one verifier while the deploy arm's
post-drift AUROC is compared against v10 fixed-structure online weighting and
static Carnot weighting.

## Implementation Status (REQ-LEARN-3685)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3685 | Proposed (python/carnot/fr11/continuous_self_learning_v11.py; scripts/experiment_3685_fr11_continuous_self_learning_v11.py) | Proposed (tests/python/test_experiment_3685_fr11_csl_v11.py) |

---

## REQ-LEARN-3708: FR-11 Continuous Self Learning v13 Multi-Session Template Consolidation

**Given** the FR-11 module and cached per-verifier trace scores with binary
labels are present, and the cached score stream contains at least three
distributionally distinct verifier-vote slices
**When** Exp 3708 runs a closed FR-11 continuous self-learning loop across at
least three simulated session boundaries
**Then** the deploy arm SHALL learn one dependency-aware verifier structure per
session and consolidate those structures into a bounded Tier-2 reusable
template library
**And** the library SHALL persist and restore by SHA256 after each session
boundary
**And** a consolidated template applied to a fresh held-out session SHALL be
compared against a cold-start no-template arm on ensemble AUROC
**And** the artifact SHALL report transfer gain over cold-start, library
boundedness, persistence round-trip status, collapse status, quality
maintenance, and pass-rate-vs-true-accuracy distinctness.

### REQ-LEARN-3708 Sub-requirements

- REQ-LEARN-3708-1: Runnable artifacts SHALL use cached traces only, with
  `inference_substrate` equal to
  `verifier_ensemble_against_cached_candidates (principle: scores cached traces; no LLM load; no compute-bound marker).`.
- REQ-LEARN-3708-2: If the FR-11 module, cached per-verifier trace scores, or
  at least three distributionally distinct session slices are unavailable, the
  workflow SHALL write the terminal verdict
  `complete: blocked_fr11_module_or_traces_unavailable`.
- REQ-LEARN-3708-3: Runnable artifacts SHALL report `n_sessions >= 3` and
  `n_online_updates >= 200`.
- REQ-LEARN-3708-4: The acceptance gate SHALL pass only when
  `template_library_bounded == true`,
  `structure_persisted_and_restored == true`,
  `collapse_detected_deploy_arm == false`, and
  `pass_rate_vs_true_accuracy_distinct_assert == true`.
- REQ-LEARN-3708-5: Runnable artifacts SHALL include principle annotations for
  `honest_verdict`, `inference_substrate`, `n_sessions`,
  `n_online_updates`, `template_library_bounded`,
  `consolidated_template_transfer_gain_over_cold_start`,
  `structure_persisted_and_restored`, `collapse_detected_deploy_arm`,
  `pass_rate_vs_true_accuracy_distinct_assert`, `quality_maintained`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- REQ-LEARN-3708-6: The terminal verdict SHALL be
  `complete: fr11_v13_multi_session_consolidation_transfers_no_collapse_quality_maintained`
  when the acceptance gate passes, the consolidated template improves held-out
  AUROC over cold-start, and quality is maintained; otherwise it SHALL be
  `complete: fr11_v13_consolidation_no_gain_over_cold_start_single_session_sufficient`
  when the acceptance gate passes without positive transfer gain.

### SCENARIO-LEARN-3708: Consolidated Template Transfers Without Collapse

**Given** three distributionally distinct cached verifier sessions and one
fresh held-out session with binary labels
**When** the v13 evaluator learns a dependency structure per session and
consolidates those structures into a capped template library
**Then** the library remains at or below its cap, persists and restores with
matching SHA256 values across the session boundaries, and the consolidated
deploy template does not collapse onto one verifier
**And** the deploy arm's pass-rate trajectory is not element-wise identical to
its true-accuracy trajectory
**And** the deploy arm's fresh-session AUROC is compared against the cold-start
no-template AUROC to decide the honest terminal verdict.

## Implementation Status (REQ-LEARN-3708)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3708 | Proposed (python/carnot/fr11/continuous_self_learning_v13.py; scripts/experiment_3708_fr11_continuous_self_learning_v13.py) | Proposed (tests/python/test_experiment_3708_fr11_csl_v13.py) |

---

## REQ-LEARN-3720: FR-11 Continuous Self Learning v14 Template Robustness Under Distribution Shift

**Given** the FR-11 module, cached per-verifier trace scores with binary labels,
and the Exp 3708 v13 bounded template library are present
**And** the cached score stream contains at least one distributionally shifted
fresh session slice relative to the v13 template-source slices
**When** Exp 3720 applies the v13 consolidated template to that shifted fresh
session
**Then** the deploy arm SHALL compare the consolidated-template deployment policy
against a cold-start no-template control on ensemble AUROC
**And** the conservative-default rule SHALL detect shifted sessions and fall back
to cold-start when the consolidated template does not beat cold-start with a
positive paired-bootstrap AUROC delta confidence interval
**And** the artifact SHALL report the shifted slice source, deploy and cold-start
AUROCs under shift, fallback status, collapse status, bounded-library status,
pass-rate-vs-true-accuracy distinctness, and an honest terminal verdict.

### REQ-LEARN-3720 Sub-requirements

- REQ-LEARN-3720-1: Runnable artifacts SHALL use cached traces only, with
  `inference_substrate` equal to
  `verifier_ensemble_against_cached_candidates (principle: scores cached traces; no LLM load; no compute-bound marker).`.
- REQ-LEARN-3720-2: If the FR-11 module, v13 template library, cached
  per-verifier trace scores, or at least one shifted session slice are
  unavailable, the workflow SHALL write the terminal verdict
  `complete: blocked_fr11_module_or_shifted_slice_unavailable`.
- REQ-LEARN-3720-3: Runnable artifacts SHALL report `n_online_updates >= 200`,
  `template_library_bounded == true`, and a bare boolean
  `pass_rate_vs_true_accuracy_distinct_assert`.
- REQ-LEARN-3720-4: A positive robustness result SHALL require the raw
  consolidated-template AUROC to exceed cold-start with a paired-bootstrap delta
  confidence interval whose lower bound is greater than zero.
- REQ-LEARN-3720-5: A conservative graceful-fallback result SHALL require shift
  detection, fallback to the cold-start arm, effective deploy AUROC no worse than
  cold-start AUROC, and no deploy-arm weight collapse.
- REQ-LEARN-3720-6: The acceptance gate SHALL pass only when
  `template_robust_or_graceful_fallback == true`,
  `collapse_detected_deploy_arm == false`,
  `template_library_bounded == true`, and
  `pass_rate_vs_true_accuracy_distinct_assert == true`.
- REQ-LEARN-3720-7: Runnable artifacts SHALL include principle annotations for
  `honest_verdict`, `inference_substrate`, `shifted_session_source`,
  `deploy_arm_auroc_under_shift`, `cold_start_auroc_under_shift`,
  `template_robust_or_graceful_fallback`, `conservative_fallback_triggered`,
  `collapse_detected_deploy_arm`, `template_library_bounded`,
  `pass_rate_vs_true_accuracy_distinct_assert`, `n_online_updates`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- REQ-LEARN-3720-8: The terminal verdict SHALL be
  `complete: fr11_v14_template_robust_under_distribution_shift_no_collapse`
  when the raw consolidated template beats cold-start with the positive delta
  confidence interval and the acceptance gate passes; SHALL be
  `complete: fr11_v14_template_falls_back_gracefully_under_shift_no_collapse`
  when the acceptance gate passes through conservative fallback; SHALL be
  `complete: fr11_v14_template_hurts_under_shift_deploy_policy_narrowed_honest`
  when the template hurts under shift and the fallback policy does not prevent
  the loss; and SHALL be
  `complete: blocked_fr11_module_or_shifted_slice_unavailable` when required
  preconditions are absent.

### SCENARIO-LEARN-3720: Shifted Template Deployment Either Helps Or Falls Back

**Given** a v13 consolidated bounded template library and a fresh cached verifier
slice whose vote distribution differs from the template-source slices
**When** the v14 evaluator scores the shifted slice with the consolidated
template and with cold-start uniform weighting
**Then** the evaluator records raw template, effective deploy, and cold-start
AUROCs under shift
**And** a raw template improvement counts as robust only when the paired
bootstrap AUROC delta confidence interval excludes zero positively
**And** a non-improving shifted deployment falls back to cold-start when the
conservative-default shift detector fires
**And** the artifact acceptance gate passes only if the deployment either helps
or falls back gracefully without collapse, the template library remains bounded,
and pass-rate and true-accuracy trajectories are not element-wise identical.

## Implementation Status (REQ-LEARN-3720)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3720 | Implemented (python/carnot/fr11/continuous_self_learning_v14.py; scripts/experiment_3720_fr11_continuous_self_learning_v14.py) | Implemented (tests/python/test_experiment_3720_fr11_csl_v14.py) |

---

## REQ-LEARN-3740: FR-11 Continuous Self Learning v15 Tier-1 EBT Stabilizer Counter Tracker

**Given** checked-in EBT training diagnostics that report per-chunk
`stabilizers_applied`, `nan_or_divergence_events`, and `ebt_converged` tuples
**When** Exp 3740 runs the FR-11 v15 self-learning tracker
**Then** the deploy arm SHALL parse Exp 3734, Exp 3735, and any earlier EBT
training artifacts that expose those fields
**And** for each observed stabilizer it SHALL update raw CPU counters for
`P(no-divergence | stabilizer enabled)` and
`P(no-divergence | stabilizer disabled)`
**And** it SHALL persist the raw tracker state so a future EBT-training
milestone resumes from the accumulated counters
**And** it SHALL emit the stabilizer set with the best observed no-divergence
correlation as a preliminary recipe for the next bounded run.

### REQ-LEARN-3740 Sub-requirements

- REQ-LEARN-3740-1: Runnable artifacts SHALL use upstream checked-in training
  diagnostics only, with `inference_substrate` equal to
  `aggregation_from_upstream_artifacts (principle: reads training diagnostics, no live model).`.
- REQ-LEARN-3740-2: If no Exp 3734/3735 or earlier EBT training diagnostics
  expose the required tuple fields, the workflow SHALL persist an empty tracker
  state and write an `honest_verdict` that includes
  `no training diagnostics to learn from -- tracker initialized empty`.
- REQ-LEARN-3740-3: For every observed stabilizer, the tracker state SHALL store
  raw enabled/disabled totals and enabled/disabled no-divergence counts; rates
  and deltas SHALL be derived from those counters at artifact-build time.
- REQ-LEARN-3740-4: The recommended recipe SHALL contain every stabilizer tied
  for the maximum enabled-minus-disabled no-divergence rate delta, with ties
  retained rather than fabricated into a strict ranking.
- REQ-LEARN-3740-5: Runnable artifacts SHALL include bare values for
  `honest_verdict`, `inference_substrate`, `stabilizer_efficacy_table`,
  `recommended_recipe`, `n_chunks_observed`, `tier1_counter_update`,
  `tracker_state_persisted`, `is_preliminary_heuristic`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`, plus separate field-principle
  annotations for those fields.
- REQ-LEARN-3740-6: The artifact SHALL avoid live-model, CUDA, and GGUF
  substrate markers and SHALL label the table as a small-sample preliminary
  heuristic rather than a statistical efficacy claim.

### SCENARIO-LEARN-3740: Stabilizer Counters Recommend a Preliminary Recipe

**Given** one EBT chunk that enables `grad_clip` and `kl_cd_fix` without
divergence, one chunk that disables them and diverges, and one chunk that
enables only `grad_clip` without divergence
**When** the v15 tracker updates counters from those chunks
**Then** `grad_clip` has two enabled no-divergence observations, `kl_cd_fix`
has one enabled no-divergence observation, both have disabled observations,
and the recommended recipe contains the stabilizer or stabilizers tied for the
largest enabled-minus-disabled no-divergence rate delta.

## Implementation Status (REQ-LEARN-3740)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3740 | Implemented (python/carnot/fr11/continuous_self_learning_v15.py; scripts/experiment_3740_fr11_self_learning_v15_stabilizer_tracker.py) | Implemented (tests/python/test_experiment_3740_fr11_csl_v15.py) |

---

## REQ-LEARN-3772: FR-11 v17 Tier-1 Live Verifier Precision Tracker

**Given** the FoVer corpus can be scored by the four live headline verifiers:
`fr11_session_memory`, `tier0r_curry_howard`, `tier0s_arithmetic_gap`, and
`tier0u_logical_consistency`
**When** Exp 3772 runs the FR-11 v17 self-learning tracker
**Then** it SHALL pivot off the bounded EBT stabilizer lineage and update
per-verifier raw CPU counters online across the FoVer labels and cached
verifier scores
**And** it SHALL derive per-verifier precision and recall from true-positive,
false-positive, false-negative, and true-negative counters
**And** it SHALL upweight high-precision verifiers while retaining a positive
`fr11_session_memory` weight so the +0.0185 FR-11 memory contribution remains
load-bearing
**And** it SHALL persist the raw tracker state so a future milestone resumes
the counter history.

### REQ-LEARN-3772 Sub-requirements

- REQ-LEARN-3772-1: The runnable artifact SHALL use
  `inference_substrate` equal to
  `verifier_ensemble_against_cached_candidates (principle: reads cached verifier scores, no live model).`
  and SHALL avoid live-model, CUDA, GGUF, or retraining claims.
- REQ-LEARN-3772-2: Each online update SHALL use only one row label and one
  score per verifier, threshold verifier scores into positive predictions, and
  update raw TP/FP/FN/TN counters without fitting model weights.
- REQ-LEARN-3772-3: The per-verifier precision table SHALL include all four
  scoring verifiers, `n_examples=1000` for the full run, raw counters,
  precision, recall, and a measured-table note rather than a small-sample
  heuristic label.
- REQ-LEARN-3772-4: The learned weighting SHALL be derived from measured
  precision, normalized, and SHALL include a strictly positive
  `fr11_session_memory` weight whenever that verifier is present.
- REQ-LEARN-3772-5: The artifact SHALL recompute ensemble AUROC under the
  learned weighting and SHALL set `memory_contribution_preserved` to a bare
  boolean that is true exactly when the learned weighting retains
  `fr11_session_memory` and the reference Exp 2837 learning contribution is at
  least +0.0185 when rounded to four decimals.
- REQ-LEARN-3772-6: If FoVer scores or labels are absent, the artifact SHALL
  persist an empty tracker state and record an `honest_verdict` containing
  `no corpus to learn from -- tracker not updated` rather than fabricating a
  precision table.
- REQ-LEARN-3772-7: Runnable artifacts SHALL include bare values for
  `honest_verdict`, `inference_substrate`, `per_verifier_precision_table`,
  `learned_weighting`, `ensemble_auroc_under_learned_weighting`,
  `memory_contribution_preserved`, `pivoted_off_dead_ebt_lineage`,
  `tier1_counter_update`, `tracker_state_persisted`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`, plus separate field-principle
  annotations for those fields.

### SCENARIO-LEARN-3772: Live Verifier Tracker Resists Poisoned Weighting

**Given** labeled FoVer-style rows where a non-memory verifier has the highest
precision but `fr11_session_memory` has real true positives
**When** the v17 tracker processes rows online and derives learned weights
**Then** every row increments raw counters exactly once per verifier
**And** the highest-precision verifier receives the largest learned weight
**And** `fr11_session_memory` keeps a positive learned weight
**And** the artifact reports `memory_contribution_preserved=true`, persists
tracker state, labels the substrate as verifier-scoring only, and records that
the v17 lineage is the live verifier precision tracker rather than the bounded
EBT stabilizer tracker.

## Implementation Status (REQ-LEARN-3772)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3772 | Implemented (python/carnot/fr11/continuous_self_learning_v17.py; scripts/experiment_3772_fr11_self_learning_v17_verifier_precision_tracker.py) | Implemented (tests/python/test_experiment_3772_fr11_csl_v17.py) |

---

## REQ-LEARN-3778: FR-11 v18 Tier-2 Constraint-Memory Threshold Consolidation

**Given** the FoVer cached corpus and the four headline verifier scoring paths
are present
**When** Exp 3778 runs the FR-11 v18 self-learning consolidator
**Then** it SHALL reuse `Tier2ThresholdMemory` to consolidate one persisted
score-threshold delta per headline verifier domain across the FoVer labels and
cached verifier scores
**And** it SHALL recompute ensemble AUROC after applying the persisted deltas
through `Tier2ThresholdMemory.apply_delta`
**And** the consolidated AUROC SHALL remain within or above the frozen Exp 2837
condition-A CI95 `[0.9027, 0.9235]`
**And** the +0.0185 `fr11_session_memory` contribution from Exp 2837 SHALL
remain preserved
**And** the artifact SHALL identify v18 as Tier-2 CPU + system-memory lookup
state, not v17 Tier-1 precision counters or a model retrain.

### REQ-LEARN-3778 Sub-requirements

- REQ-LEARN-3778-1: The runnable artifact SHALL use
  `inference_substrate` equal to
  `verifier_ensemble_against_cached_candidates (principle: reads cached verifier scores, no live model).`
  and SHALL avoid live-model, CUDA, GGUF, or retraining claims.
- REQ-LEARN-3778-2: Preconditions SHALL run under `.venv/bin/python`, require
  importable `pyyaml`, `numpy`, `sklearn`, and `Tier2ThresholdMemory`, and
  require the absolute FoVer corpus path to exist before consolidation.
- REQ-LEARN-3778-3: If the FoVer corpus or verifier scoring path is missing,
  the artifact SHALL use a `blocked_<resource>` honest verdict and SHALL record
  `no corpus to consolidate from` without fabricating
  `per_domain_threshold_deltas`.
- REQ-LEARN-3778-4: Each domain delta SHALL be produced by calling
  `Tier2ThresholdMemory.update_domain_delta(domain_key, examples, labels)` over
  the real score column for that domain, and the reported delta SHALL equal the
  value read back through `get_domain_delta(domain_key)`.
- REQ-LEARN-3778-5: Consolidated ensemble scores SHALL be computed only by
  applying `Tier2ThresholdMemory.apply_delta(domain_key, raw_score)` before the
  existing headline ensemble weighting and AUROC calculation.
- REQ-LEARN-3778-6: Runnable artifacts SHALL include bare values for
  `honest_verdict`, `inference_substrate`, `per_domain_threshold_deltas`,
  `ensemble_auroc_under_consolidation`, `auroc_within_frozen_ci`,
  `memory_contribution_preserved`, `is_tier2_not_tier1`,
  `tier2_lookup_is_cpu_memory`, `tracker_state_persisted`, `model_specs`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`, plus separate
  field-principle annotations for those fields.
- REQ-LEARN-3778-7: The terminal verdict SHALL be
  `complete: fr11_v18_tier2_constraint_memory_consolidated_auroc_within_frozen_ci_memory_contribution_preserved_state_persisted`
  when the consolidated AUROC no-regression gate passes, the memory
  contribution is preserved, and the Tier-2 memory database is persisted.

### SCENARIO-LEARN-3778: Constraint Memory Applies Real Persisted Deltas

**Given** labeled FoVer-style scores where each verifier domain has at least 32
examples and `fr11_session_memory` remains part of the ensemble
**When** the v18 consolidator updates `Tier2ThresholdMemory` for each domain
and recomputes AUROC through `apply_delta`
**Then** each reported delta round-trips from the persisted database
**And** applying a poisoned persisted delta after consolidation changes the
adjusted score returned by `apply_delta`, proving the result uses the
tracker's real persisted behavior
**And** the artifact reports `memory_contribution_preserved=true`,
`is_tier2_not_tier1=true`, `tracker_state_persisted=true`, and verifier-scoring
substrate only.

## Implementation Status (REQ-LEARN-3778)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3778 | Proposed (python/carnot/fr11/continuous_self_learning_v18.py; scripts/experiment_3778_fr11_self_learning_v18_tier2_constraint_memory.py) | Proposed (tests/python/test_experiment_3778_fr11_csl_v18.py) |

---

## REQ-LEARN-3788: FR-11 v19 Tier-3 Predictive Verification Head

**Given** the FoVer cached corpus and the four frozen headline verifier scoring
paths are present
**When** Exp 3788 runs the FR-11 v19 self-learning experiment
**Then** it SHALL train `FR11ExtendedJEPA` as an additive Tier-3 predictive
head over the cached verifier feature signal and binary step-error labels
**And** it SHALL report held-out predictive AUROC from a deterministic
train/test split
**And** it SHALL persist the trained predictor state so a later milestone can
resume from the learned predictive head
**And** it SHALL leave the frozen 0.9131 FoVer scoring ensemble unchanged and
preserve the +0.0185 `fr11_session_memory` contribution from Exp 2837
**And** it SHALL identify v19 as Tier-3 predictive verification, not v17
Tier-1 precision tracking or v18 Tier-2 constraint-memory consolidation.

### REQ-LEARN-3788 Sub-requirements

- REQ-LEARN-3788-1: The runnable artifact SHALL use
  `inference_substrate` equal to
  `verifier_ensemble_against_cached_candidates (principle: reads cached verifier scores, no live model).`
  and SHALL avoid live-model, CUDA, GGUF, or full-ensemble replacement claims.
- REQ-LEARN-3788-2: Preconditions SHALL run under `.venv/bin/python`, require
  importable `jax`, `numpy`, `sklearn`, and `FR11ExtendedJEPA`, and require the
  absolute FoVer corpus path to exist before training.
- REQ-LEARN-3788-3: If the FoVer corpus or verifier scoring path is missing,
  the artifact SHALL use a `blocked_<resource>` honest verdict and SHALL record
  `no corpus to train on` without fabricating predictive AUROC or predictor
  state.
- REQ-LEARN-3788-4: The predictor training set SHALL be built by scoring the
  cached FoVer rows through the four headline verifiers, converting those
  verifier scores into a 258-dimensional `FR11ExtendedJEPA` embedding, and
  mapping the binary step-error label into the predictor's violation targets.
- REQ-LEARN-3788-5: Held-out predictive AUROC SHALL be computed on rows excluded
  from predictor training; the artifact SHALL include train/test split sizes so
  the AUROC is auditable.
- REQ-LEARN-3788-6: Runnable artifacts SHALL include bare values for
  `honest_verdict`, `inference_substrate`, `predictive_auroc`,
  `train_test_split_sizes`, `headline_ensemble_unchanged`,
  `memory_contribution_preserved`, `is_tier3_not_tier1_or_tier2`,
  `tracker_state_persisted`, `model_specs`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`, plus separate field-principle
  annotations for those fields.
- REQ-LEARN-3788-7: The terminal verdict SHALL start with
  `complete: fr11_v19_tier3_predictive_verifier_trained_predictive_auroc_`
  and end with
  `_headline_ensemble_unchanged_memory_contribution_preserved_state_persisted`
  when the predictor trains, the frozen ensemble reference is unchanged, the
  memory contribution is preserved, and predictor state is persisted.

### SCENARIO-LEARN-3788: Predictive Head Persists Real Learned Behavior

**Given** labeled FoVer-style verifier scores with both positive and negative
step-error labels
**When** the v19 trainer builds JEPA embeddings, trains `FR11ExtendedJEPA`, and
persists the predictor state
**Then** the reloaded predictor produces the same predictions as the trained
predictor for the same embedding
**And** poisoning the persisted predictor parameters changes that prediction,
proving the artifact depends on the real predictive head state
**And** the artifact reports `headline_ensemble_unchanged=true`,
`memory_contribution_preserved=true`, `is_tier3_not_tier1_or_tier2=true`,
`tracker_state_persisted=true`, and verifier-scoring substrate only.

## Implementation Status (REQ-LEARN-3788)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3788 | Proposed (python/carnot/fr11/continuous_self_learning_v19.py; scripts/experiment_3788_fr11_self_learning_v19_tier3_predictive.py) | Proposed (tests/python/test_experiment_3788_fr11_csl_v19.py) |

---

## REQ-LEARN-3803: FR-11 v20 Tier-3 Fast-Path Cascade Gate

**Given** Exp 3788 persisted a trained `FR11ExtendedJEPA` Tier-3 predictor and
the FoVer cached corpus plus the four frozen headline verifier scoring paths
are present
**When** Exp 3803 runs the FR-11 v20 self-learning application experiment
**Then** it SHALL load the persisted v19 predictor state and SHALL NOT retrain
the predictor
**And** it SHALL evaluate the fast-path gate on FoVer rows excluded from the
v19 predictor training split
**And** it SHALL sweep confidence thresholds where confident predictor scores
short-circuit full four-verifier scoring and uncertain rows fall through to the
unchanged frozen ensemble scorer
**And** it SHALL choose an operating point whose combined fast-path plus
fall-through AUROC remains inside the frozen Exp 2837 condition-A CI95
`[0.9027, 0.9235]`
**And** it SHALL persist the selected operating point so the next milestone can
resume the Tier-3 fast-path policy.

### REQ-LEARN-3803 Sub-requirements

- REQ-LEARN-3803-1: The runnable artifact SHALL use `inference_substrate`
  equal to
  `verifier_ensemble_against_cached_candidates (principle: reads cached scores + the persisted predictor, no live model).`
  and SHALL avoid live-model, CUDA, GGUF, or predictor-retraining claims.
- REQ-LEARN-3803-2: Preconditions SHALL run under `.venv/bin/python`, require
  importable `jax`, `numpy`, `sklearn`, and `FR11ExtendedJEPA`, require the
  v19 predictor state file to load, and require the absolute FoVer corpus path
  to exist before gate evaluation.
- REQ-LEARN-3803-3: If the interpreter, v19 predictor state, FoVer corpus, or
  cached verifier scoring path is missing, the artifact SHALL use a
  `blocked_<resource>` honest verdict and SHALL NOT fabricate skip rate,
  effective AUROC, or operating-point persistence.
- REQ-LEARN-3803-4: The held-out split SHALL be disjoint from the v19 predictor
  training rows by reconstructing the v19 deterministic split from the same
  labels and random seed, using only v19 test rows for gate measurement.
- REQ-LEARN-3803-5: The gate SHALL compute predictor confidence as
  `abs(predictor_probability - 0.5)`, short-circuit rows whose confidence is at
  least the candidate threshold, and use frozen full-ensemble scores for all
  rows below that threshold.
- REQ-LEARN-3803-6: The chosen operating point SHALL maximize skip rate among
  swept thresholds whose combined effective AUROC is inside the frozen CI; if
  no threshold satisfies the CI, the artifact SHALL report a no-saving finding
  rather than claiming a regressing compute saving.
- REQ-LEARN-3803-7: Runnable artifacts SHALL include bare values for
  `honest_verdict`, `inference_substrate`, `skip_rate_at_no_regression`,
  `effective_auroc_at_operating_point`, `compute_saving_vs_accuracy_curve`,
  `accuracy_regression`, `held_out_split_sizes`,
  `headline_ensemble_unchanged`, `is_tier3_application_not_retrain`,
  `operating_point_persisted`, `model_specs`, `random_seed`,
  `reproducibility_checksum`, and `duration_s`, plus separate field-principle
  annotations for those fields.
- REQ-LEARN-3803-8: The terminal verdict SHALL be
  `complete: fr11_v20_tier3_fast_path_gate_skip_rate_<x>_effective_auroc_<y>_in_frozen_ci_no_accuracy_regression_headline_ensemble_unchanged_operating_point_persisted`
  when the operating point is persisted, the effective AUROC is inside the
  frozen CI, no accuracy regression is reported, and the frozen headline
  ensemble remains the fall-through path.

### SCENARIO-LEARN-3803: Fast-Path Gate Uses Persisted Predictor Without Replacing Fall-Through

**Given** labeled FoVer-style verifier scores, a persisted `FR11ExtendedJEPA`
state, and a deterministic v19 split with both positive and negative held-out
rows
**When** the v20 gate evaluates predictor probabilities on the held-out rows
and sweeps confidence thresholds
**Then** confident rows take predictor scores and increment the skip-rate
counter
**And** uncertain rows keep the frozen full-ensemble scores exactly
**And** poisoning the persisted predictor state changes the chosen gate curve,
proving the result uses the real persisted Tier-3 predictor
**And** the artifact reports `headline_ensemble_unchanged=true`,
`is_tier3_application_not_retrain=true`, `operating_point_persisted=true`, and
verifier-scoring substrate only.

## Implementation Status (REQ-LEARN-3803)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3803 | Implemented (python/carnot/fr11/continuous_self_learning_v20.py; scripts/experiment_3803_fr11_v20_tier3_fast_path_gate.py, Exp 3803) | Implemented (tests/python/test_experiment_3803_fr11_v20_fast_path_gate.py) |

---

## REQ-LEARN-3813: FR-11 v21 Fast-Path Operating-Point Robustness

**Given** Exp 3788 persisted a trained `FR11ExtendedJEPA` Tier-3 predictor and
Exp 3803 persisted a fast-path operating point with threshold `T` and skip rate
0.56
**When** Exp 3813 runs the FR-11 v21 robustness measurement
**Then** it SHALL load the persisted predictor state and the persisted v20
operating point and SHALL NOT retrain the predictor or re-tune the threshold
**And** it SHALL construct a second held-out FoVer split with a different seed
that is disjoint from both the v19 predictor-training rows and the v20
measurement rows
**And** it SHALL apply the persisted threshold unchanged: confident rows use the
Tier-3 predictor score, and uncertain rows fall through to the frozen
four-verifier ensemble
**And** it SHALL report the second-split skip rate and effective AUROC honestly,
including the valid finding that the v20 operating point may require
per-split recalibration.

### REQ-LEARN-3813 Sub-requirements

- REQ-LEARN-3813-1: Preconditions SHALL run under `.venv/bin/python`, require
  importable `jax`, `numpy`, `sklearn`, and `FR11ExtendedJEPA`, and require both
  the v19 predictor state and v20 operating-point state to load before any
  second-split measurement is reported.
- REQ-LEARN-3813-2: If the interpreter, persisted predictor state, persisted
  operating-point state, FoVer corpus, or cached verifier scoring path is
  missing, the artifact SHALL use a `blocked_<resource>` honest verdict and
  SHALL NOT fabricate skip rate, effective AUROC, or generalization success.
- REQ-LEARN-3813-3: The second split SHALL preserve original FoVer row IDs and
  SHALL prove that the v19 predictor-training split, the v20 measurement split,
  and the v21 second held-out split are mutually disjoint.
- REQ-LEARN-3813-4: The v20 persisted operating-point threshold SHALL be reused
  unchanged; Exp 3813 SHALL NOT sweep thresholds or select a new operating
  point to match the v20 skip rate.
- REQ-LEARN-3813-5: `operating_point_generalizes` SHALL be true only when the
  second-split skip rate is within the configured robustness band around 0.56,
  the second-split effective AUROC is inside frozen CI95 `[0.9027, 0.9235]`,
  `accuracy_regression=false`, and the frozen headline ensemble remains
  unchanged.
- REQ-LEARN-3813-6: Runnable artifacts SHALL include bare values for
  `honest_verdict`, `inference_substrate`, `v20_operating_point_threshold`,
  `skip_rate_second_split`, `effective_auroc_second_split`,
  `operating_point_generalizes`, `three_split_sizes`, `accuracy_regression`,
  `headline_ensemble_unchanged`, `is_measurement_not_retrain`, `model_specs`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`, plus separate
  field-principle annotations for those fields.
- REQ-LEARN-3813-7: The terminal verdict SHALL be
  `complete: fr11_v21_fast_path_robustness_skip_<x>_effective_auroc_<y>_operating_point_generalizes_<true|false>_headline_ensemble_unchanged_measurement_not_retrain`
  when the persisted operating point is measured on the second split, whether
  or not it generalizes.

### SCENARIO-LEARN-3813: Persisted Fast-Path Gate Is Measured Without Re-Tuning

**Given** labeled FoVer-style verifier scores, original row IDs, a persisted
`FR11ExtendedJEPA` state, and a persisted v20 operating-point threshold
**When** the v21 robustness measurement excludes the reconstructed v19
training rows and v20 measurement rows
**Then** the second split is disjoint from both earlier splits
**And** the persisted threshold is applied unchanged to compute skip rate and
effective AUROC
**And** poisoning the persisted predictor state changes the second-split gate
behavior, proving the result uses the real persisted Tier-3 predictor
**And** the artifact reports `is_measurement_not_retrain=true`,
`headline_ensemble_unchanged=true`, and verifier-scoring substrate only.

## Implementation Status (REQ-LEARN-3813)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3813 | Proposed (python/carnot/fr11/continuous_self_learning_v21.py; scripts/experiment_3813_fr11_v21_fast_path_robustness.py, Exp 3813) | Proposed (tests/python/test_experiment_3813_fr11_v21_fast_path_robustness.py) |

## REQ-LEARN-061: Tier 4 Adaptive Structure Prototype
**Given** the active verifier set and their scores on the corpus
**When** Tier 4 evaluates marginal contributions
**Then** verifiers with marginal contribution < 0.002 are pruned
**And** the pruned structure is evaluated to ensure it holds the 0.9131 CI
**And** residual gap regions (gold-incorrect but missed by all retained verifiers) are flagged.

## SCENARIO-LEARN-105: Tier 4 Prunes Redundant Verifiers
**Given** an ensemble with tier0r_curry_howard, tier0u_logical_consistency, fr11_session_memory, tier0s_arithmetic_gap
**When** the marginal contribution is evaluated
**Then** tier0u and tier0s are pruned (marginal < 0.002)
**And** the remaining structure holds AUROC > 0.903
**And** compute_saving_fraction = 0.5

---

## REQ-LEARN-3864: FR-11 v23 Online Independence-Reweighting Tracker

**Given** importable `carnot.verify`, cached FoVer-style rows, and per-verifier
scores for the four frozen Exp 2837 verifier columns
**When** Exp 3864 runs online over the corpus
**Then** it SHALL maintain per-verifier precision counters and an independence
counter for errors caught by exactly one verifier
**And** it SHALL derive a deterministic learned weighting from the frozen prior,
online precision, and unique-catch independence
**And** it SHALL evaluate the reweighted ensemble AUROC against the frozen
0.9131 CI without moving the frozen headline number
**And** it SHALL persist the learned weighting state for the next FR-11
milestone.

### REQ-LEARN-3864 Sub-requirements

- REQ-LEARN-3864-1: Preconditions SHALL require `import carnot.verify` to
  succeed and SHALL require a cached FoVer-style corpus source before scoring;
  missing resources SHALL produce an `honest_verdict` beginning with
  `blocked_<resource>`.
- REQ-LEARN-3864-2: The online tracker SHALL update precision counters and
  unique-error-catch independence counters using only CPU counter updates over
  cached verifier scores; it SHALL NOT load an LLM, GPU model, or mutate model
  weights.
- REQ-LEARN-3864-3: `independence_weight_per_verifier` SHALL include all four
  frozen verifier names and SHALL upweight high-precision, high-independence
  verifiers while retaining the frozen prior as a no-regression guard.
- REQ-LEARN-3864-4: `auroc_in_frozen_ci` SHALL be emitted as a bare boolean and
  SHALL be true exactly when `reweighted_ensemble_auroc` lies inside the frozen
  CI95 range `[0.9027316334533082, 0.9235355665466916]`.
- REQ-LEARN-3864-5: `memory_ablation_contribution_preserved` SHALL be emitted
  as a bare boolean and SHALL require the Exp 2837 learning contribution to
  round to at least `+0.0185` and the learned `fr11_session_memory` weight to
  remain positive.
- REQ-LEARN-3864-6: Runnable artifacts SHALL include bare values for
  `reweighted_ensemble_auroc`, `auroc_in_frozen_ci`,
  `independence_weight_per_verifier`,
  `memory_ablation_contribution_preserved`, `update_cost_note`,
  `state_persisted_path`, `preconditions_checked`, `random_seed`,
  `reproducibility_checksum`, `inference_substrate`, and `duration_s`, plus
  separate field-principle annotations for those fields.
- REQ-LEARN-3864-7: The terminal verdict SHALL be
  `complete: fr11_v23_independence_reweighting_learned_auroc<A>_in_frozen_ci_memory_contribution_preserved_state_persisted`
  when `auroc_in_frozen_ci=true` and
  `memory_ablation_contribution_preserved=true`; otherwise, if the learned
  reweighting regresses below the frozen CI, the verdict SHALL be
  `complete: fr11_v23_independence_reweighting_REGRESSED_below_frozen_ci_reverted_to_static_weights`.

### SCENARIO-LEARN-3864: Unique Error Catches Increase Independence Weight

**Given** labeled verifier scores where one verifier catches positive rows that
all other verifiers miss while preserving precision
**When** the v23 tracker processes rows online and computes weights
**Then** the unique catches increase that verifier's independence score and
learned weight relative to the frozen prior
**And** the artifact reports `auroc_in_frozen_ci` and
`memory_ablation_contribution_preserved` as bare booleans
**And** the persisted state contains the learned weights and raw counter table
used by the next FR-11 milestone.

## Implementation Status (REQ-LEARN-3864)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3864 | Proposed (python/carnot/fr11/continuous_self_learning_v23.py; scripts/experiments/experiment_3864_fr11_self_learning_v23_independence_reweighting.py, Exp 3864) | Proposed (tests/python/test_experiment_3864_fr11_v23_independence_reweighting.py) |

---

## REQ-LEARN-3888: FR-11 v24 Continues Persisted Independence-Reweighting State

**Given** the persisted Exp 3864 v23 independence-reweighting state,
importable `carnot.verify`, and a fresh FoVer verification-corpus slice from
`data/fover_corpus_v4.json` with `data/fover_corpus_v3.json` as fallback
**When** Exp 3888 runs Tier-1 online independence-reweighting
**Then** it SHALL load the v23 state, update the existing per-verifier counters
with fresh CPU-only cached verifier scores, remeasure the learned ensemble on
the frozen FoVer headline scoring path, and persist a v24 state for the next
milestone
**And** it SHALL report whether the frozen CI-band AUROC, the FR-11 memory
ablation contribution, and the frozen `0.9131` headline invariant all hold.

### REQ-LEARN-3888 Sub-requirements

- REQ-LEARN-3888-1: Preconditions SHALL require `import carnot.verify` to
  succeed, the v23 state JSON to load, and at least one fresh verification
  corpus (`fover_corpus_v4.json` preferred, `fover_corpus_v3.json` fallback) to
  contain both FoVer label classes; missing resources SHALL produce an
  `honest_verdict` beginning with `blocked_<resource>`.
- REQ-LEARN-3888-2: The online update SHALL continue the v23 counters rather
  than reinitializing them, and SHALL upweight verifiers that uniquely catch
  gold-incorrect rows missed by all peer verifiers.
- REQ-LEARN-3888-3: The artifact SHALL emit bare values, not `{value,
  principle}` wrappers, for `learned_ensemble_auroc`,
  `memory_ablation_contribution`, `frozen_headline_unchanged`,
  `invariant_held`, `state_persisted_path`, `n_updates`,
  `preconditions_checked`, `random_seed`, `reproducibility_checksum`,
  `duration_s`, and `inference_substrate`, with separate field-principle text.
- REQ-LEARN-3888-4: `invariant_held` SHALL be a bare boolean that is true
  exactly when `learned_ensemble_auroc` lies inside the frozen CI
  `[0.9027, 0.9235]`, `memory_ablation_contribution >= 0.012`, and
  `frozen_headline_unchanged=true`.
- REQ-LEARN-3888-5: The terminal verdict SHALL be
  `complete: fr11_v24_INVARIANT_HELD_auroc<A>_memcontrib<M>_state_persisted`
  when the invariant holds; otherwise it SHALL be
  `complete: fr11_v24_INVARIANT_BROKEN_auroc<A>_memcontrib<M>_online_reweighting_drifts`.

### SCENARIO-LEARN-3888: Fresh Online Update Preserves Frozen Invariant

**Given** a v23 persisted state with non-zero FR-11 session-memory weight and a
fresh balanced FoVer v4 slice
**When** v24 loads the state, applies online independence updates, and evaluates
the learned weights against the frozen headline corpus
**Then** the resulting artifact reports a bare `invariant_held` value
**And** the v24 state contains cumulative counters whose observed-example count
exceeds the v23 state by `n_updates`
**And** the honest verdict names either `INVARIANT_HELD` or
`INVARIANT_BROKEN` rather than silently substituting the frozen `0.9131`
headline.

## Implementation Status (REQ-LEARN-3888)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3888 | Proposed (python/carnot/fr11/continuous_self_learning_v24.py; scripts/experiments/experiment_3888_fr11_v24_independence_reweighting.py, Exp 3888) | Proposed (tests/python/test_experiment_3888_fr11_v24_independence_reweighting.py) |

---

## REQ-LEARN-3921: FR-11 v25 Continues Persisted v24 Independence-Reweighting State

**Given** the persisted Exp 3888 v24 independence-reweighting state,
importable `carnot.verify`, and a fresh verification-corpus slice from
`data/fover_corpus_v4.json` with `data/fover_corpus_v3.json` as fallback
**When** Exp 3921 runs Tier-1 online independence-reweighting
**Then** it SHALL load the v24 state, continue the cumulative per-verifier
counters with CPU-only cached verifier-score updates, remeasure the learned
ensemble AUROC on the frozen FoVer headline scoring path, and persist a v25
state for the next milestone
**And** it SHALL report whether the frozen CI-band AUROC, FR-11 memory
ablation contribution, and frozen `0.9131` headline invariant all hold.

### REQ-LEARN-3921 Sub-requirements

- REQ-LEARN-3921-1: Preconditions SHALL require `import carnot.verify` to
  succeed, the v24 state JSON to load, and at least one fresh verification
  corpus (`fover_corpus_v4.json` preferred, `fover_corpus_v3.json` fallback) to
  contain both FoVer label classes; missing resources SHALL produce an
  `honest_verdict` beginning with `blocked_<resource>`.
- REQ-LEARN-3921-2: The online update SHALL continue the v24 counters rather
  than reinitializing them, and SHALL upweight verifiers that uniquely catch
  gold-incorrect rows missed by all peer verifiers.
- REQ-LEARN-3921-3: The artifact SHALL emit bare values, not `{value,
  principle}` wrappers, for `learned_ensemble_auroc`,
  `memory_ablation_contribution`, `frozen_headline_unchanged`,
  `invariant_held`, `state_persisted_path`, `n_updates`,
  `preconditions_checked`, `random_seed`, `reproducibility_checksum`,
  `duration_s`, and `inference_substrate`, with separate field-principle text.
- REQ-LEARN-3921-4: `invariant_held` SHALL be a bare boolean that is true
  exactly when `learned_ensemble_auroc` lies inside the frozen CI
  `[0.9027, 0.9235]`, `memory_ablation_contribution >= 0.012`, and
  `frozen_headline_unchanged=true`.
- REQ-LEARN-3921-5: The terminal verdict SHALL be
  `complete: fr11_v25_INVARIANT_HELD_auroc<A>_memcontrib<M>_state_persisted`
  when the invariant holds; otherwise it SHALL be
  `complete: fr11_v25_INVARIANT_BROKEN_auroc<A>_memcontrib<M>_online_reweighting_drifts`.

### SCENARIO-LEARN-3921: Fresh v25 Update Preserves Frozen Invariant

**Given** a v24 persisted state with non-zero FR-11 session-memory weight and a
fresh balanced FoVer verification-corpus slice
**When** v25 loads the state, applies online independence updates, and
evaluates the learned weights against the frozen headline corpus
**Then** the resulting artifact reports a bare `invariant_held` value
**And** the v25 state contains cumulative counters whose observed-example count
exceeds the v24 state by `n_updates`
**And** the honest verdict names either `INVARIANT_HELD` or
`INVARIANT_BROKEN` rather than silently substituting the frozen `0.9131`
headline.

## Implementation Status (REQ-LEARN-3921)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3921 | Implemented (python/carnot/fr11/continuous_self_learning_v25.py; scripts/experiments/experiment_3921_fr11_v25_independence_reweighting.py, Exp 3921) | Implemented (tests/python/test_experiment_3921_fr11_v25_independence_reweighting.py) |

---

## REQ-LEARN-3930: FR-11 v26 Continues Persisted v25 State and Learns Cascade Band Online

**Given** the persisted Exp 3921 v25 independence-reweighting state,
importable `carnot.verify`, a fresh verification-corpus slice from
`data/fover_corpus_v4.json` with `data/fover_corpus_v3.json` as fallback, and
an optional Exp 3927 cascade-router artifact on disk
**When** Exp 3930 runs Tier-1 online independence-reweighting
**Then** it SHALL load the v25 state, continue the cumulative per-verifier
counters with CPU-only cached verifier-score updates, remeasure learned
ensemble AUROC on the frozen FoVer headline scoring path, persist a v26 state
for the next milestone, and, when Exp 3927 is present, update the cascade
escalation band from fresh-slice margins while recording `band_before` and
`band_after`.

### REQ-LEARN-3930 Sub-requirements

- REQ-LEARN-3930-1: Preconditions SHALL require `import carnot.verify` to
  succeed and the v25 state JSON to load before any online update; missing
  resources SHALL produce an `honest_verdict` beginning with
  `blocked_<resource>`.
- REQ-LEARN-3930-2: The online update SHALL continue the v25 counters rather
  than reinitializing them, and SHALL upweight verifiers that uniquely catch
  gold-incorrect rows missed by all peer verifiers on the fresh slice.
- REQ-LEARN-3930-3: If
  `results/experiment_3927_non_degenerate_cascade_router.json` exists, the
  runner SHALL disk-read it, derive fresh learned-ensemble margins around the
  cascade energy threshold, and record `band_before`, `band_after`, and whether
  `cascade_band_learned` is a bare boolean true. If the artifact is absent,
  `cascade_band_learned` SHALL be false.
- REQ-LEARN-3930-4: The artifact SHALL emit bare values, not `{value,
  principle}` wrappers, for `learned_ensemble_auroc`,
  `memory_ablation_contribution`, `frozen_headline_unchanged`,
  `cascade_band_learned`, `invariant_held`, `state_persisted_path`,
  `n_updates`, `preconditions_checked`, `random_seed`,
  `reproducibility_checksum`, `duration_s`, and `inference_substrate`, with
  separate field-principle text.
- REQ-LEARN-3930-5: `invariant_held` SHALL be a bare boolean that is true
  exactly when `learned_ensemble_auroc` lies inside the frozen CI
  `[0.9027, 0.9235]`, `memory_ablation_contribution >= 0.012`, and
  `frozen_headline_unchanged=true`.
- REQ-LEARN-3930-6: The terminal verdict SHALL be
  `complete: fr11_v26_INVARIANT_HELD_auroc<A>_memcontrib<M>_cascade_band_learned<B>_state_persisted`
  when the invariant holds; otherwise it SHALL be
  `complete: fr11_v26_INVARIANT_BROKEN_auroc<A>_memcontrib<M>_online_reweighting_drifts`.

### SCENARIO-LEARN-3930: Fresh v26 Update Preserves Frozen Invariant and Updates Cascade Band

**Given** a v25 persisted state with non-zero FR-11 session-memory weight, a
fresh balanced FoVer verification-corpus slice, and an Exp 3927 cascade artifact
on disk
**When** v26 loads the state, applies online independence updates, evaluates the
learned weights against the frozen headline corpus, and learns the cascade band
from fresh-slice margins
**Then** the resulting artifact reports bare `invariant_held` and
`cascade_band_learned` values
**And** the v26 state contains cumulative counters whose observed-example count
exceeds the v25 state by `n_updates`
**And** the honest verdict names either `INVARIANT_HELD` or
`INVARIANT_BROKEN` rather than silently substituting the frozen `0.9131`
headline.

## Implementation Status (REQ-LEARN-3930)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-3930 | Proposed (python/carnot/fr11/continuous_self_learning_v26.py; scripts/experiments/experiment_3930_fr11_v26_cascade_band_online_learning.py, Exp 3930) | Proposed (tests/python/test_experiment_3930_fr11_v26_cascade_band_online_learning.py) |
\n## REQ-LEARN-3958: Cross-game DSL fragment transfer\n\n**Given** an ordered sequence of non-spatial games\n**When** models are induced with a growing library of extracted helper functions\n**Then** later games require fewer codex calls or achieve lower consistency energy.\n\n## SCENARIO-LEARN-3958: Extracts only helper functions, not predict\n**Given** a python module with predict and a helper\n**When** extract_library_fragments is called\n**Then** only the helper function is returned.\n\n## Implementation Status\n| Requirement | Python | Tests |\n|---|---|---|\n| REQ-LEARN-3958 | Implemented (scripts/experiments/experiment_3958_cross_game_dsl_transfer.py) | Implemented (tests/python/test_experiment_3958_cross_game_dsl_transfer.py) |\n

---

## REQ-LEARN-4039: ArcMemo v6 Compressed Concept Library Transfer

**Given** accumulated ArcMemo concept records from prior real-env-confirmed
ARC-AGI-3 solve-transfer artifacts and the `.373` candidate artifacts
`results/experiment_4035_hierarchical_search_over_vc33_wm.json` and
`results/experiment_4038_seventh_game_explore_first.json`
**When** Exp 4039 builds the ArcMemo v6 library
**Then** it SHALL cluster recurring induced-program fragments by family,
operation tokens, and observed effects into named, documented abstractions
rather than storing only raw concepts
**And** it SHALL measure transfer cost on `.373` content against cold start and
the v5 raw-memory arm, counting only real-env-confirmed solved content as
eligible transfer evidence
**And** it SHALL write
`results/experiment_4039_arcmemo_concept_library_v6.json`.

### REQ-LEARN-4039 Sub-requirements

- REQ-LEARN-4039-1: The compressed library SHALL expose at least one named
  abstraction with `name`, `signature`, `documentation`,
  `source_concepts`, and `lambda_abstraction` fields.
- REQ-LEARN-4039-2: Exp 4039 SHALL report bare top-level fields
  `honest_verdict`, `solve_transfer_win`, `actions_cold`, `actions_v5`,
  `actions_v6`, `n_named_abstractions`, and `inference_substrate`.
- REQ-LEARN-4039-3: `inference_substrate` SHALL equal
  `offline_arc_agi3_arcmemo_concept_transfer`.
- REQ-LEARN-4039-4: `solve_transfer_win` SHALL be true only when
  `actions_v6 < actions_cold` and `actions_v6 < actions_v5`; otherwise the
  honest verdict SHALL begin with `complete: arcmemo_v6_no_transfer_`.
- REQ-LEARN-4039-5: `.373` artifacts that are not real-env-confirmed solves
  SHALL be preserved as excluded evidence and SHALL NOT contribute to aggregate
  action totals.

### SCENARIO-LEARN-4039: v6 Compression Must Beat Cold and v5

**Given** one confirmed `.373` solve with a cold baseline, a v5 action cost, and
a matching compressed abstraction
**When** Exp 4039 builds the compressed library and evaluates that content
**Then** it reports `solve_transfer_win=true` only if the v6 action count is
strictly below both baselines
**And** the honest verdict uses
`success: arcmemo_v6_library_transfer_<cold>to<v6>_actions`.

**Given** a `.373` artifact whose real environment confirmation is false
**When** Exp 4039 evaluates transfer content
**Then** that artifact appears in excluded evidence and does not lower or raise
`actions_cold`, `actions_v5`, or `actions_v6`.

## Implementation Status (REQ-LEARN-4039)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4039 | Proposed (python/carnot/agentic/arc_arcmemo_concept_library_v6.py; scripts/experiments/exp4039_arcmemo_concept_library_v6.py, Exp 4039) | Proposed (tests/python/test_exp4039_arcmemo_concept_library_v6.py) |

---

## REQ-LEARN-4050: ArcMemo v7 Cross-Game Concept Library Transfer

**Given** the seven prior real-env-confirmed ARC-AGI-3 solved games
`r11l`, `lp85`, `sc25`, `su15`, `tn36`, `cd82`, and `dc22`
and the eighth-game artifact
`results/experiment_4049_eighth_game_explore_first.json`
**When** Exp 4050 builds the ArcMemo v7 library
**Then** it SHALL compress only prior-game induced-program fragments into
named, documented abstractions
**And** it SHALL seed the eighth-game explore-first accounting from that
prior-game library without importing the eighth-game induced mechanic into the
library
**And** it SHALL write
`results/experiment_4050_arcmemo_cross_game_transfer_v7.json`.

### REQ-LEARN-4050 Sub-requirements

- REQ-LEARN-4050-1: The v7 library SHALL expose named abstractions with
  `name`, `signature`, `documentation`, `source_games`, `source_fragments`,
  and `lambda_abstraction` fields.
- REQ-LEARN-4050-2: Exp 4050 SHALL report bare top-level fields
  `honest_verdict`, `cross_game_transfer_win`, `actions_cold`,
  `actions_within_game_v6`, `actions_cross_game_v7`,
  `n_reused_abstractions`, and `inference_substrate`.
- REQ-LEARN-4050-3: `inference_substrate` SHALL equal
  `offline_arc_agi3_arcmemo_cross_game_transfer`.
- REQ-LEARN-4050-4: `cross_game_transfer_win` SHALL be true only when
  `actions_cross_game_v7 < actions_cold` and
  `actions_cross_game_v7 < actions_within_game_v6`; otherwise the honest
  verdict SHALL begin with `complete: arcmemo_v7_no_cross_game_transfer_`.
- REQ-LEARN-4050-5: If Exp 4049 did not solve but exposes a usable attempt
  trace, Exp 4050 SHALL measure transfer on that attempt depth and SHALL NOT
  fabricate a solved-game transfer claim.

### SCENARIO-LEARN-4050: v7 Must Beat Cold and Within-Game v6 on a New Game

**Given** an Exp 4049 eighth-game trace and a v7 library built from prior-game
fragments only
**When** a prior-game abstraction fires on the new game
**Then** Exp 4050 records the fired abstraction count and induction-call cost
**And** it reports `cross_game_transfer_win=true` only if the v7 action count is
strictly below both the cold baseline and the within-game-v6 baseline.

**Given** Exp 4049 lacks a real-env-confirmed solve but still contains a usable
attempt trace
**When** Exp 4050 evaluates transfer
**Then** it measures the same attempt depth and emits a `complete:` verdict
rather than claiming a solved-game transfer win.

## Implementation Status (REQ-LEARN-4050)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4050 | Proposed (python/carnot/agentic/arc_arcmemo_cross_game_transfer_v7.py; scripts/experiments/exp4050_arcmemo_cross_game_transfer_v7.py, Exp 4050) | Proposed (tests/python/test_exp4050_arcmemo_cross_game_transfer_v7.py) |

---

## REQ-LEARN-4062: ArcMemo v8 Richer Cross-Game Concept Library Transfer

**Given** the eight prior real-env-confirmed ARC-AGI-3 solved games
`r11l`, `lp85`, `sc25`, `su15`, `tn36`, `cd82`, `dc22`, and `sb26`
and the ninth-game artifact
`results/experiment_4060_ninth_game_explore_first.json`
**When** Exp 4062 builds the ArcMemo v8 library
**Then** it SHALL compress only prior-game induced-program fragments from all
eight solves into named, documented abstractions
**And** it SHALL seed the ninth-game explore-first accounting from that richer
prior-game library without importing any ninth-game induced mechanic into the
library
**And** it SHALL measure the ninth game against both cold start and
within-game-only memory before claiming cross-game transfer
**And** it SHALL write
`results/experiment_4062_arcmemo_cross_game_transfer_v8.json`.

### REQ-LEARN-4062 Sub-requirements

- REQ-LEARN-4062-1: The v8 library SHALL expose named abstractions with
  `name`, `signature`, `documentation`, `source_games`, `source_fragments`,
  and `lambda_abstraction` fields, and SHALL include the eighth prior solve
  when it is real-env-confirmed.
- REQ-LEARN-4062-2: Exp 4062 SHALL report bare top-level fields
  `honest_verdict`, `cross_game_transfer_win`, `actions_cold`,
  `actions_within_game`, `actions_cross_game_v8`, `n_reused_abstractions`,
  and `inference_substrate`.
- REQ-LEARN-4062-3: `inference_substrate` SHALL equal
  `offline_arc_agi3_arcmemo_cross_game_transfer`.
- REQ-LEARN-4062-4: `cross_game_transfer_win` SHALL be true only when
  `actions_cross_game_v8 < actions_cold` and
  `actions_cross_game_v8 < actions_within_game`; otherwise the honest verdict
  SHALL begin with `complete: arcmemo_v8_no_cross_game_transfer_`.
- REQ-LEARN-4062-5: If Exp 4060 did not solve but exposes a usable attempt
  trace, Exp 4062 SHALL measure transfer on that same attempt depth and SHALL
  NOT fabricate a solved-game transfer claim.
- REQ-LEARN-4062-6: If Exp 4060 is absent or lacks any usable trace, Exp 4062
  SHALL emit a complete no-transfer artifact with zero action counts and
  explicit target evidence rather than substituting another game.

### SCENARIO-LEARN-4062: v8 Must Beat Cold and Within-Game Memory on a New Ninth Game

**Given** an Exp 4060 ninth-game trace and a v8 library built from all eight
prior solved-game fragments
**When** one or more prior-game abstractions fire on the new game
**Then** Exp 4062 records the fired abstraction count and induction-call cost
**And** it reports `cross_game_transfer_win=true` only if the v8 action count
is strictly below both the cold baseline and the within-game-only baseline.

**Given** Exp 4060 lacks a real-env-confirmed solve but still contains a usable
attempt trace
**When** Exp 4062 evaluates transfer
**Then** it measures the same attempt depth and emits a `complete:` verdict
rather than claiming a solved-game transfer win.

**Given** Exp 4060 is missing or exposes no usable trace
**When** Exp 4062 evaluates transfer
**Then** it writes a valid `complete:` artifact that preserves the eight-game
library evidence and reports no ninth-game transfer measurement.

## Implementation Status (REQ-LEARN-4062)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4062 | Proposed (python/carnot/agentic/arc_arcmemo_cross_game_transfer_v8.py; scripts/experiments/exp4062_arcmemo_cross_game_transfer_v8.py, Exp 4062) | Proposed (tests/python/test_exp4062_arcmemo_cross_game_transfer_v8.py) |

---

## REQ-LEARN-4072: ArcMemo v9 Cross-Game Transfer on Fresh Ninth-Game Work

**Given** the eight prior real-env-confirmed ARC-AGI-3 solved games
`r11l`, `lp85`, `sc25`, `su15`, `tn36`, `cd82`, `dc22`, and `sb26`
and the fresh ninth-game artifact
`results/experiment_4070_ninth_game_explore_first.json`
**When** Exp 4072 builds the ArcMemo v9 library
**Then** it SHALL compress only prior solved-game induced-program fragments into
named, documented abstractions richer than the v8 library
**And** it SHALL evaluate the Exp 4070 ninth-game trace honestly as either a
confirmed solve or a bounded attempt trace
**And** it SHALL measure the ninth game against cold start and within-game-only
memory before claiming cross-game transfer
**And** it SHALL write
`results/experiment_4072_arcmemo_cross_game_transfer_v9.json`.

### REQ-LEARN-4072 Sub-requirements

- REQ-LEARN-4072-1: The v9 library SHALL expose named abstractions with
  `name`, `signature`, `documentation`, `source_games`, `source_fragments`,
  `lambda_abstraction`, and `match_tokens` fields, and SHALL include all eight
  prior solved-game fragments when they are real-env-confirmed.
- REQ-LEARN-4072-2: Exp 4072 SHALL report bare top-level fields
  `honest_verdict`, `cross_game_transfer_win`, `actions_cold`,
  `actions_within_game`, `actions_cross_game_v9`, `n_reused_abstractions`,
  and `inference_substrate`.
- REQ-LEARN-4072-3: `inference_substrate` SHALL equal
  `offline_arc_agi3_arcmemo_cross_game_transfer`.
- REQ-LEARN-4072-4: `cross_game_transfer_win` SHALL be true only when
  `actions_cross_game_v9 < actions_cold` and
  `actions_cross_game_v9 < actions_within_game`; otherwise the honest verdict
  SHALL begin with `complete: arcmemo_v9_no_cross_game_transfer_`.
- REQ-LEARN-4072-5: If Exp 4070 did not solve but exposes a usable attempt
  trace, Exp 4072 SHALL measure transfer on that same attempt depth and SHALL
  NOT fabricate a solved-game transfer claim.
- REQ-LEARN-4072-6: If Exp 4070 is absent or lacks any usable trace, Exp 4072
  SHALL emit a complete no-transfer artifact with zero action counts and
  explicit target evidence rather than substituting another game.
- REQ-LEARN-4072-7: If Exp 4070 is solved, Exp 4072 SHALL preserve the
  ninth-game induced fragment as future-library evidence, but that held-out
  target fragment SHALL NOT be counted as a prior-game reused abstraction for
  the Exp 4072 cross-game win claim.

### SCENARIO-LEARN-4072: v9 Must Beat Cold and Within-Game Memory on Exp 4070

**Given** the Exp 4070 ninth-game trace and a v9 library built from all eight
prior solved-game fragments
**When** prior-game abstractions fire on the new game
**Then** Exp 4072 records the fired abstraction count and induction-call cost
**And** it reports `cross_game_transfer_win=true` only if the v9 action count
is strictly below both the cold baseline and the within-game-only baseline.

**Given** Exp 4070 lacks a real-env-confirmed solve but still contains a usable
attempt trace
**When** Exp 4072 evaluates transfer
**Then** it measures the same attempt depth and emits a `complete:` verdict
rather than claiming a solved-game transfer win.

**Given** Exp 4070 is solved
**When** Exp 4072 builds its audit payload
**Then** it records the ninth-game fragment as target-derived future-library
evidence without letting that target-derived fragment satisfy
`n_reused_abstractions`.

## Implementation Status (REQ-LEARN-4072)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4072 | Proposed (python/carnot/agentic/arc_arcmemo_cross_game_transfer_v9.py; scripts/experiments/exp4072_arcmemo_cross_game_transfer_v9.py, Exp 4072) | Proposed (tests/python/test_exp4072_arcmemo_cross_game_transfer_v9.py) |

---

## REQ-LEARN-4077: ARC Execution-Verifier Reward RFT Corpus Build Gate

**Given** cached ARC program-generation outputs with at least `k=8` generated
transform programs per task, HF safetensors LADDER bases
`Qwen/Qwen3.5-0.8B` and `openbmb/MiniCPM5-1B`, importable `trl`/`peft`
trainers, visible CUDA, and loadable ARC-1 plus ARC-2 task pools
**When** Exp 4077 prepares the verifier-as-self-improvement reward beachhead
**Then** it SHALL check every precondition before corpus construction or
training, compute the Phase-0 certification precision and recall of
`demo-perfect => test-gold` on a labeled held-out split, and stop with
`honest_verdict=blocked_precision_gate_unmet_<p>_<r>` when precision is below
`0.85` or recall is below `0.20`.

If and only if the precision gate passes, the experiment SHALL build three
N-matched corpora from the same generated program pool: `RFT-CORRECT`
containing demo-perfect programs, `RFT-ABLATION` containing not-demo-perfect
programs matched to the RFT-correct task distribution, and `gold-SFT`
containing test-gold-passing programs matched to the RFT-correct size where
possible. The disjoint held-out eval split SHALL never be included in any
training corpus.

The terminal artifact SHALL write
`results/experiment_4077_verifier_reward_rft_corpus_build.json` with bare
top-level fields `honest_verdict`, `certification_precision`,
`certification_recall`, `n_rft_correct`, `n_rft_ablation`, `n_gold_sft`,
`n_heldout_tasks`, `runner_ready`, `trainer_smoke_passed`,
`preconditions_checked`, and `inference_substrate`.

### REQ-LEARN-4077 Sub-requirements

- REQ-LEARN-4077-1: The runner SHALL reuse
  `carnot.agentic.arc_gap4_execution_verifier` and
  `carnot.verify.sandbox.sandboxed_exec_function` to score generated transform
  programs against public demos and held-out candidate gold labels.
- REQ-LEARN-4077-2: The precision gate SHALL pass only when
  `certification_precision >= 0.85` and `certification_recall >= 0.20`.
- REQ-LEARN-4077-3: When the precision gate fails, corpus counts SHALL be zero
  and the runner SHALL NOT build RFT/SFT corpora or run a smoke train.
- REQ-LEARN-4077-4: When the precision gate passes, the three corpora SHALL be
  generated from the same held-in program pool and SHALL have equal item counts.
- REQ-LEARN-4077-5: The LoRA runner SHALL expose three arms
  `rft_correct`, `rft_ablation`, and `gold_sft`, plus a held-out eval manifest,
  and SHALL support a two-task smoke train without launching the full Exp 4078
  training run.

### SCENARIO-LEARN-4077: Precision Gate Blocks Poisoned Demo-Perfect Labels

**Given** a labeled held-out ARC program pool where demo-perfect programs are
gold-correct below 85% precision
**When** Exp 4077 computes the certification gate
**Then** the artifact reports `blocked_precision_gate_unmet_<p>_<r>`, leaves
all three corpus counts at zero, and records `trainer_smoke_passed=false`.

### SCENARIO-LEARN-4077-NMATCH: Passing Gate Builds Three Matched Arms

**Given** a generated held-in program pool whose demo-perfect certification
passes the precision gate and contains enough not-demo-perfect and test-gold
programs
**When** Exp 4077 builds the training corpora
**Then** `n_rft_correct == n_rft_ablation == n_gold_sft`, every ablation item
comes from the same generator but is not demo-perfect, and the held-out eval
manifest contains only disjoint task ids.

## Implementation Status (REQ-LEARN-4077)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4077 | Proposed (python/carnot/agentic/arc_exp4077_verifier_reward_rft_corpus_build.py; scripts/experiments/exp4077_verifier_reward_rft_corpus_build.py, Exp 4077) | Proposed (tests/python/test_experiment_4077_verifier_reward_rft_corpus_build.py) |

---

## REQ-LEARN-4078: Verifier-Reward RFT Detached LoRA Train Launch

**Given** the Exp 4077 three-arm corpora are present and non-empty, HF
safetensors ladder bases `Qwen/Qwen3.5-0.8B` and `openbmb/MiniCPM5-1B`
are cached, `trl`/`peft` expose `SFTTrainer` and `GRPOTrainer`, and CUDA is
available
**When** the Exp 4078 launcher runs
**Then** it SHALL launch detached LoRA training workers for the three arms
`rft_correct`, `rft_ablation`, and `gold_sft` with the same rank, learning
rate, epoch/step budget, and random seed for each base, differing only by the
arm corpus and label.

The launcher SHALL write stable checkpoint paths keyed by base and arm under
`results/checkpoints/experiment_4078_verifier_reward_rft_train/<base>/<arm>`
so later conductor windows resume accumulated progress rather than restart from
scratch. It SHALL record `epochs_completed` for every base+arm key from any
existing trainer checkpoints before launch.

The terminal artifact SHALL write
`results/experiment_4078_verifier_reward_rft_train_launch.json` with bare
top-level fields `honest_verdict`, `train_launched`, `checkpoint_paths`,
`epochs_completed`, and `inference_substrate`. It SHALL also include
`preconditions_checked`, `launched_workers`, `training_config`, and
`field_principles` so auditors can distinguish a real detached launch from a
fabricated training claim.

### REQ-LEARN-4078 Sub-requirements

- REQ-LEARN-4078-1: The launcher SHALL fail closed with
  `honest_verdict="blocked_<resource>"` and `train_launched=false` when any
  required base, trainer import, CUDA, or Exp 4077 corpus precondition is
  missing.
- REQ-LEARN-4078-2: The launcher SHALL use POSIX session-detached subprocess
  launch semantics for training workers so training can survive agent exit and
  conductor iteration boundaries.
- REQ-LEARN-4078-3: The checkpoint path map SHALL contain one stable
  base+arm-keyed path per planned training arm, whether this run is blocked,
  newly launched, or resumed.
- REQ-LEARN-4078-4: The worker SHALL configure TRL/PEFT LoRA training with
  `save_strategy="epoch"` and `resume_from_checkpoint` when a stable checkpoint
  already exists for the base+arm path.
- REQ-LEARN-4078-5: Arm launch order SHALL prioritize `rft_correct` and
  `rft_ablation` before `gold_sft`, preserving the ablation contrast when a
  future launch window cannot fit all arms.

### SCENARIO-LEARN-4078-BLOCKED: Missing Exp 4077 Corpora Block Training

**Given** the Exp 4077 terminal artifact is blocked or any of the three corpus
JSONL sidecars is absent or empty
**When** Exp 4078 runs
**Then** it writes `train_launched=false`, zero launch workers, stable
checkpoint paths for the planned arms, and
`honest_verdict="blocked_exp4077_corpora_missing"`.

### SCENARIO-LEARN-4078-LAUNCH: Detached Workers Preserve Stable Checkpoints

**Given** all preconditions pass
**When** Exp 4078 launches training
**Then** every launched worker command targets exactly one base+arm stable
checkpoint path, the launch uses session-detached semantics, and the artifact
reports `honest_verdict` starting with
`complete: rft_3arm_train_launched_detached`.

## Implementation Status (REQ-LEARN-4078)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4078 | Proposed (python/carnot/agentic/arc_exp4078_verifier_reward_rft_train_launch.py; scripts/experiments/exp4078_verifier_reward_rft_train_launch.py, Exp 4078) | Proposed (tests/python/test_experiment_4078_verifier_reward_rft_train_launch.py) |

---

## REQ-LEARN-4080: Sudoku Verifier-RFT Positive Control For Pipeline Sanity

**Given** the original Sudoku verifier-as-teacher beachhead artifact with
held-out per-seed rates for verifier-certified RFT
(`energy_distilled_greedy`) and gold-SFT (`gold_distilled_greedy`), visible
CUDA, and importable TRL/PEFT trainers matching the Exp 4078 precondition
surface
**When** Exp 4080 checks the verifier-as-reward pipeline sanity before trusting
the ARC RFT headline
**Then** it SHALL aggregate at least three held-out seeds, emit
`results/experiment_4080_sudoku_rft_positive_control.json`, and set
`reproduces_beachhead=true` only when verifier-certified RFT is at least
gold-SFT on the aggregate rate and on every included seed.

The terminal artifact SHALL include bare top-level fields `honest_verdict`,
`rft_rate`, `sft_rate`, `n_seeds`, `reproduces_beachhead`, and
`inference_substrate`. When the positive control reproduces, the verdict SHALL
be `complete: sudoku_positive_control_rft_ge_sft_reproduced`. When RFT falls
below SFT, the verdict SHALL be
`complete: sudoku_positive_control_FAILED_pipeline_suspect` because the ARC
result is not interpretable until the pipeline is fixed.

### REQ-LEARN-4080 Sub-requirements

- REQ-LEARN-4080-1: The runner SHALL reuse the Exp 4078 GPU and trainer
  precondition checks before issuing a positive-control verdict.
- REQ-LEARN-4080-2: `rft_rate` SHALL be the float mean of
  `energy_distilled_greedy` over the included Sudoku seeds.
- REQ-LEARN-4080-3: `sft_rate` SHALL be the float mean of
  `gold_distilled_greedy` over the same included Sudoku seeds.
- REQ-LEARN-4080-4: `n_seeds` SHALL be a bare int and SHALL be at least `3`
  for any completed positive-control verdict.
- REQ-LEARN-4080-5: `reproduces_beachhead` SHALL be a bare bool whose field
  principle records that this is the pipeline sanity check that makes the ARC
  headline interpretable.

### SCENARIO-LEARN-4080: Sudoku Positive Control Reproduces RFT >= SFT

**Given** three held-out Sudoku seeds where verifier-certified RFT is at least
gold-SFT for every seed
**When** Exp 4080 aggregates the positive control
**Then** the artifact reports the reproduced verdict, `rft_rate >= sft_rate`,
`n_seeds >= 3`, and `reproduces_beachhead=true`.

### SCENARIO-LEARN-4080-FAIL: Sudoku Positive Control Flags A Suspect Pipeline

**Given** held-out Sudoku seeds where verifier-certified RFT falls below
gold-SFT
**When** Exp 4080 aggregates the positive control
**Then** the artifact reports
`complete: sudoku_positive_control_FAILED_pipeline_suspect` and
`reproduces_beachhead=false`.

## Implementation Status (REQ-LEARN-4080)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4080 | Proposed (python/carnot/agentic/sudoku_exp4080_rft_positive_control.py; scripts/experiments/exp4080_sudoku_rft_positive_control.py, Exp 4080) | Proposed (tests/python/test_experiment_4080_sudoku_rft_positive_control.py) |

---

## REQ-LEARN-4087: GAP-5 Certification Precision Rescue Sweep

**Given** cached GAP-4 ARC program pools
`results/arc3_gap4_induced_programs.json`,
`results/arc3_gap4_arc2_induced_programs.json`, and any cached ARC-2
consistency/chain ensemble arms, with no live Codex generation and no GPU
dependency
**When** Exp 4087 evaluates cheap deterministic certification filters over the
saved `def transform(grid)` programs
**Then** it SHALL replay each candidate against the public demo pairs and
held-out gold labels, sweep the precision-recall frontier for demo-perfect,
augmentation-invariance, k-of-n agreement, graded min-Hamming, and stacked
filter variants, and set `precision_rescue_succeeded=true` only when some
operating point reaches `precision >= 0.85` and `recall >= 0.20`.

The terminal artifact SHALL write
`results/experiment_4087_certification_precision_rescue.json` with bare
top-level fields `honest_verdict`, `precision_rescue_succeeded`,
`best_certified_precision`, `best_op_point_recall`, `frontier`,
`n_tasks_scored`, `n_codex_calls`, `random_seed`,
`reproducibility_checksum`, and `inference_substrate`. The
`inference_substrate` SHALL equal
`offline_saved_gap4_program_replay_precision_rescue`, and `n_codex_calls`
SHALL remain `0`.

### REQ-LEARN-4087 Sub-requirements

- REQ-LEARN-4087-1: Preconditions SHALL verify the cached ARC-1 and ARC-2
  induced-program pools are readable and non-empty and that
  `gap5_cross_example_selector.safe_transform_from_code` is importable before
  replaying any candidate.
- REQ-LEARN-4087-2: Demo-perfect certification SHALL execute saved programs
  only against public demo inputs and SHALL score held-out gold only after
  certification decisions are made.
- REQ-LEARN-4087-3: Augmentation-invariance certification SHALL apply
  deterministic color-permutation and D4 grid symmetries to demo pairs and
  require the program to stay demo-perfect on every augmented demo.
- REQ-LEARN-4087-4: Agreement certification SHALL certify a task prediction
  only when at least `k` distinct demo-perfect programs for that task produce
  the same held-out prediction.
- REQ-LEARN-4087-5: Graded min-Hamming certification SHALL sweep `tau`
  thresholds over the minimum demo-set cell-disagreement energy observed for
  each candidate.
- REQ-LEARN-4087-6: The frontier SHALL include entries of the form
  `{filter_stack, threshold, precision, recall, n_certified}` for every
  deterministic operating point considered.

### SCENARIO-LEARN-4087: Precision Rescue Gate Succeeds

**Given** an offline replay pool where a stacked filter operating point reaches
at least 85% certified precision while retaining at least 20% recall
**When** Exp 4087 builds the frontier
**Then** the artifact reports `precision_rescue_succeeded=true` and an
`honest_verdict` beginning with
`complete: precision_rescue_succeeded_best_`.

### SCENARIO-LEARN-4087-FAIL: Precision Rescue Gate Fails Closed

**Given** an offline replay pool where no filter operating point reaches the
precision and recall thresholds simultaneously
**When** Exp 4087 builds the frontier
**Then** the artifact reports `precision_rescue_succeeded=false`,
`n_codex_calls=0`, and an `honest_verdict` beginning with
`complete: precision_rescue_FAILED_`.

## Implementation Status (REQ-LEARN-4087)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4087 | Proposed (python/carnot/agentic/arc_exp4087_certification_precision_rescue.py; scripts/experiments/exp4087_certification_precision_rescue.py, Exp 4087) | Proposed (tests/python/test_experiment_4087_certification_precision_rescue.py) |

---

## REQ-LEARN-4099: TRM Pool Verifier Discrimination Probe

**Given** cached TRM ARC pass@K candidate-grid outputs, Exp 4099 SHALL run as
a deterministic offline rerank over saved TRM candidate grids with zero Codex
generation during selection. It SHALL first check that
`results/trm_verifier_rerank_opportunity.json` and
`results/arc3_trm_verifier_rerank_gap1.json` are present. If no saved
candidate-grid pool can be loaded or regenerated, it SHALL write an honest
blocked verdict beginning with `blocked_trm_pool_missing`.

**When** Exp 4099 scores each TRM candidate grid, it SHALL compute model-free
signals derived from the Exp 4087 precision-rescue primitives: demo-fit or
demo-consensus consistency, deterministic color-permutation plus D4
augmentation-invariance, k-of-n agreement among TRM samples producing the same
grid, and graded min-Hamming energy to the task consensus grid. It SHALL build
single-signal and stacked rerankers, compare them to `TRM_VOTE`, and include an
oracle pass@2 ceiling from whether any candidate in the task pool is gold.

**Then** the terminal artifact SHALL write
`results/experiment_4099_trm_pool_verifier_discrimination_probe.json` with
bare top-level fields `honest_verdict`, `verifier_beats_trm_vote`,
`captured_pp_directional`, `pool_n_tasks`, `underpowered`, `per_reranker`,
`oracle_ceiling`, `n_tasks_scored`, `random_seed`, and
`reproducibility_checksum`. The strict `verifier_beats_trm_vote` gate SHALL be
`true` only when the best non-TRM reranker's pass@2 exceeds `TRM_VOTE` and the
bootstrap 95% confidence interval for `(reranker_pass@2 - TRM_VOTE_pass@2)`
excludes zero. If the reranked pool contains fewer than 100 tasks, the artifact
SHALL set `underpowered=true` while still reporting the directional
`captured_pp_directional` point estimate.

### SCENARIO-LEARN-4099: Strict Reranker Win Requires Bootstrap Separation

**Given** a deterministic TRM candidate pool where a verifier reranker selects
more pass@2-correct candidates than `TRM_VOTE`
**When** the paired bootstrap confidence interval for captured pass@2 excludes
zero
**Then** Exp 4099 reports `verifier_beats_trm_vote=true`, records the winning
reranker under `per_reranker`, and emits an `honest_verdict` beginning with a
terminal complete/success/passed/shipped prefix.

### SCENARIO-LEARN-4099-UNDERPOWERED: Directional Signal Without Power

**Given** fewer than 100 TRM-pool tasks can be reranked offline
**When** no reranker has a bootstrap CI95 excluding zero
**Then** Exp 4099 reports `verifier_beats_trm_vote=false`,
`underpowered=true`, preserves the best point estimate in
`captured_pp_directional`, and keeps `honest_verdict` terminal-prefixed rather
than treating an honest no-win verdict as a harness failure.

## Implementation Status (REQ-LEARN-4099)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4099 | Implemented (python/carnot/agentic/arc_exp4099_trm_pool_verifier_discrimination_probe.py; scripts/experiments/exp4099_trm_pool_verifier_discrimination_probe.py, Exp 4099) | Implemented (tests/python/test_experiment_4099_trm_pool_verifier_discrimination_probe.py) |

---

## REQ-LEARN-4100: Conditional TRM Verifier-RFT or Native-Trainer Smoke

**Given** Exp 4099 has written
`results/experiment_4099_trm_pool_verifier_discrimination_probe.json`, Exp
4100 SHALL first check all required training resources before branching:
cached Hugging Face TRM weights under
`~/.cache/huggingface/hub/models--arcprize--trm_arc_prize_verification/`,
the native `nano-trm` substrate files `nano-trm/src/arc_evaluator.py` and
`nano-trm/src/baseline.py`, CUDA availability through torch, and the Exp 4099
probe artifact. If any precondition is missing, Exp 4100 SHALL write an honest
blocked artifact with `branch_taken="blocked"` and a corresponding
`blocked_trm_weights_not_cached`, `blocked_trm_substrate_missing`,
`blocked_cuda_unavailable`, or `blocked_exp4099_probe_missing` verdict.

**When** Exp 4099 reports `verifier_beats_trm_vote=true`, Exp 4100 SHALL take
`branch_taken="rft"`, build a verifier-certified training corpus from the TRM
candidate pool, build an N-matched vote-certified self-SFT ablation corpus from
the same TRM samples, reserve a disjoint held-out split, and fine-tune the TRM
with the native `nano-trm` trainer rather than trl, peft, or LoRA. It SHALL
measure held-out ARC pass@1 and pass@2 for the verifier-certified arm, the
vote-certified ablation arm, and the unmodified cold TRM. The load-bearing
contrast SHALL be the A-vs-B `rft_vs_ablation_delta` with a bootstrap 95%
confidence interval.

**When** Exp 4099 reports `verifier_beats_trm_vote=false`, Exp 4100 SHALL take
`branch_taken="smoke"` and SHALL NOT run a full RFT. Instead it SHALL execute a
minimal real native `nano-trm` training run synchronously, print progress from
that training process, require that a checkpoint is saved and torch-reloaded,
and report the verifier-grid-discrimination bottleneck from Exp 4099. The
measured `duration_s` SHALL be wall-clock train plus checkpoint reload time; if
that real smoke finishes in under 60 seconds, the artifact SHALL omit live-model
success markers while still honestly reporting the checkpoint result.

**Then** the terminal artifact SHALL write
`results/experiment_4100_trm_verifier_rft_conditional.json` with bare
top-level fields `honest_verdict`, `branch_taken`,
`rft_vs_ablation_delta`, `trm_native_trainer_checkpoint_ok`,
`preconditions_checked`, `random_seed`, and `reproducibility_checksum`.
Complete verdicts SHALL begin with `complete:`, `success:`, `passed:`, or
`shipped:`; honest null outcomes such as A~=B or smoke-only because Exp 4099
found no grid discrimination SHALL still be complete decision-grade outcomes.
The `reproducibility_checksum` SHALL hash the train/held-out split plus the
certified corpus or smoke substrate inputs, so corpus drift is visible.

### SCENARIO-LEARN-4100-SMOKE: No Grid Discrimination Runs Native Checkpoint Smoke

**Given** Exp 4099 reports `verifier_beats_trm_vote=false`
**When** the native nano-trm trainer produces a checkpoint and torch reloads it
**Then** Exp 4100 reports `branch_taken="smoke"`,
`trm_native_trainer_checkpoint_ok=true`, keeps
`rft_vs_ablation_delta.status="not_run_no_verifier_signal"`, and writes an
`honest_verdict` beginning with `complete:`.

### SCENARIO-LEARN-4100-RFT: Verifier Label Must Beat Vote Ablation

**Given** Exp 4099 reports `verifier_beats_trm_vote=true`
**When** the verifier-certified arm beats the vote-certified self-SFT ablation
on held-out pass@2 and the bootstrap CI95 excludes zero
**Then** Exp 4100 reports `branch_taken="rft"`, records the A-vs-B delta under
`rft_vs_ablation_delta`, and uses a terminal complete/success/passed/shipped
verdict rather than a gate wrapper.

## Implementation Status (REQ-LEARN-4100)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4100 | Proposed (python/carnot/agentic/arc_exp4100_trm_verifier_rft_conditional.py; scripts/experiments/exp4100_trm_verifier_rft_conditional.py, Exp 4100) | Proposed (tests/python/test_experiment_4100_trm_verifier_rft_conditional.py) |

---

## REQ-LEARN-4107: Native nano-trm Sudoku Mechanism Smoke

**Given** the `.380` verifier-as-reward pivot depends on the native
`olivkoch/nano-trm` training mechanism rather than LoRA, trl, or a synthetic
checkpoint, Exp 4107 SHALL first check that `uv` is on PATH, that
`nano-trm/src/nn/train.py` exists, that `torch.cuda.is_available()` is true,
and that `results/trm_runs/` exists as a persistent writable save parent under
the repository. If `uv` or the native trainer is missing, it SHALL stop with
`honest_verdict="blocked_nanotrm_or_uv_missing"`. If CUDA is unavailable, it
SHALL stop with `honest_verdict="blocked_cuda_unavailable"`. A missing
transient `/tmp` save path SHALL NOT be accepted as a mechanism smoke.

**When** preconditions pass, Exp 4107 SHALL run the native command from the
`nano-trm/` directory using Hydra experiment `trm_sudoku_4x4` and a persistent
`save_dir` under `results/trm_runs/`. If the configured
`nano-trm/data/sudoku_4x4_small` dataset is absent, the documented Sudoku data
builder SHALL be used before training. The trainer run SHALL print progress
while it runs and SHALL produce a real `.ckpt` checkpoint that can be reloaded
with torch.

**Then** the terminal artifact SHALL be written to
`results/experiment_4107_nanotrm_mechanism_smoke.json` and SHALL contain the
required fields `honest_verdict`, `nanotrm_trainer_checkpoint_ok`,
`exact_accuracy`, `checkpoint_path`, `duration_s`, `preconditions_checked`,
and `random_seed`. `nanotrm_trainer_checkpoint_ok` SHALL be a bare bool that is
true only when nano-trm produced a checkpoint and torch reloaded it.
`exact_accuracy` SHALL come from a real solve metric such as
`val/exact_accuracy` or `train/exact_accuracy`; `val/q_halt_accuracy alone SHALL NOT satisfy`
the contract. The acceptance gate is passed only when
`nanotrm_trainer_checkpoint_ok == true`, a real exact accuracy is reported, and
`duration_s > 60`.

### SCENARIO-LEARN-4107: Reloadable Checkpoint And Real Exact Metric Clear The Gate

**Given** the native nano-trm Sudoku trainer writes `checkpoints/last.ckpt` and
CSV metrics containing `val/exact_accuracy`
**When** Exp 4107 reloads the checkpoint and builds the artifact
**Then** it reports `nanotrm_trainer_checkpoint_ok=true`, records the real
exact accuracy rather than `q_halt_accuracy`, preserves the persistent
checkpoint path, and marks the acceptance gate as passed only if the wall-clock
duration is greater than 60 seconds.

### SCENARIO-LEARN-4107-BLOCKED: Missing Runtime Resources Stop Honestly

**Given** `uv`, `nano-trm/src/nn/train.py`, CUDA, or the persistent save parent
is unavailable
**When** Exp 4107 starts
**Then** it writes the corresponding `blocked_<resource>` honest verdict,
records every precondition already checked, leaves
`nanotrm_trainer_checkpoint_ok=false`, and does not fabricate a checkpoint or
exact-accuracy metric.

## Implementation Status (REQ-LEARN-4107)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4107 | Implemented (python/carnot/experiment_4107_nanotrm_mechanism_smoke.py, Exp 4107) | Implemented (tests/python/test_experiment_4107_nanotrm_mechanism_smoke.py) |

---

## REQ-LEARN-4108: nano-trm Sudoku Extreme Published Baseline

**Given** Exp 4107 has written
`results/experiment_4107_nanotrm_mechanism_smoke.json`, Exp 4108 SHALL first
read that artifact. If `nanotrm_trainer_checkpoint_ok` is not true, the native
trainer mechanism is unproven and Exp 4108 SHALL run only a shorter honest
Sudoku Extreme reproduction attempt, not hard-blocking and not claiming the
published baseline was reproduced.

**When** the Sudoku Extreme dataset is absent, Exp 4108 SHALL invoke the native
nano-trm builder from `nano-trm/` with
`uv run python scripts/data/build_sudoku_extreme_dataset.py --output-dir
./data/sudoku_extreme_1k_aug_1k --subsample-size 1000 --num-aug 1000
--eval-ratio 0.01`. It SHALL then run the native nano-trm trainer from
`nano-trm/` with Hydra experiment `trm_sudoku_extreme_1k_aug_1k`, CUDA, CSV
logging, progress printed at validation intervals, and a persistent save
directory under `results/trm_runs/`. Progress printing SHALL be best-effort:
terminal pipe closure MUST NOT abort training or prevent the checkpoint from
being written.

**Then** the terminal artifact SHALL be written to
`results/experiment_4108_nanotrm_sudoku_extreme_baseline.json` and SHALL
contain top-level required fields `honest_verdict`,
`reproduced_exact_accuracy`, `matches_published_087`, `checkpoint_path`,
`duration_s`, `random_seed`, and `reproducibility_checksum`.
`reproduced_exact_accuracy` SHALL be the measured `val/exact_accuracy` when
available, never `val/q_halt_accuracy`. `matches_published_087` SHALL be a bare
bool indicating whether the measured exact accuracy is within tolerance of the
published approximate 0.87 target. The acceptance gate is complete when a
persisted reloadable checkpoint exists and `reproduced_exact_accuracy` is
reported; a below-target exact accuracy is still a complete honest verdict.

### SCENARIO-LEARN-4108: Reloadable Extreme Baseline Reports Real Accuracy

**Given** the native nano-trm Sudoku Extreme trainer writes a checkpoint and CSV
metrics containing `val/exact_accuracy`
**When** Exp 4108 builds the baseline artifact
**Then** it reports the real validation exact accuracy, a bare
`matches_published_087` boolean, the persistent checkpoint path, wall-clock
duration, random seed, and a reproducibility checksum over the dataset and
native configuration inputs.

### SCENARIO-LEARN-4108-SHORT: Unproven Mechanism Uses Short Honest Attempt

**Given** Exp 4107 does not report `nanotrm_trainer_checkpoint_ok=true`
**When** Exp 4108 runs
**Then** it uses the shorter reproduction mode, preserves the measured exact
accuracy if one exists, and writes a complete honest verdict without claiming
the published 0.87 baseline was reproduced.

## Implementation Status (REQ-LEARN-4108)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4108 | Implemented (python/carnot/experiment_4108_nanotrm_sudoku_extreme_baseline.py, Exp 4108) | Implemented (tests/python/test_experiment_4108_nanotrm_sudoku_extreme_baseline.py) |

---

## REQ-LEARN-4116: Resumable nano-trm Sudoku Extreme Pass 1

**Given** Exp 4108 produced only a partial Sudoku Extreme baseline because the
conductor wall-clock cap interrupted a long native nano-trm run, Exp 4116 SHALL
use the stable shared checkpoint lineage
`results/trm_runs/sudoku_extreme_baseline/last.ckpt` rather than a per-run Hydra
directory. Before launch it SHALL check that `uv` is on PATH, that
`nano-trm/src/nn/train.py` exists, that `torch.cuda.is_available()` is true,
that the `nano-trm/data/sudoku_extreme_1k_aug_1k` dataset is complete or can be
generated by the native builder, and that the stable checkpoint directory
exists. Missing `uv` or trainer resources SHALL yield
`honest_verdict="blocked_nanotrm_or_uv_missing"`; missing CUDA SHALL yield
`honest_verdict="blocked_cuda_unavailable"`.

**When** a loadable Exp 4108 checkpoint exists, Exp 4116 SHALL seed the stable
checkpoint path by copying that checkpoint before training. It SHALL then run
from `nano-trm/` with Hydra experiment `trm_sudoku_extreme_1k_aug_1k`, CSV
logging, `ckpt_path=<stable>/last.ckpt`, `trainer.max_time="00:01:00:00"`, and
Hydra/model-checkpoint overrides that save `last.ckpt` back into the same stable
directory. The run SHALL print progress at validation/checkpoint intervals and
SHALL stop cleanly before the conductor's 80-minute cap.

**Then** the terminal artifact SHALL be written to
`results/experiment_4116_sudoku_extreme_resume_pass1.json` and SHALL contain
top-level required fields `honest_verdict`, `val_exact_accuracy`,
`cumulative_epochs`, `stable_checkpoint_path`, `checkpoint_reload_ok`,
`duration_s`, and `random_seed`. `val_exact_accuracy` SHALL come from a real
`val/exact_accuracy` CSV metric, never `q_halt_accuracy`. `checkpoint_reload_ok`
SHALL be a bare bool that is true only when torch reloads the saved stable
checkpoint. The acceptance gate passes when the saved stable checkpoint reloads,
`duration_s < 4800`, and `val_exact_accuracy` is reported; reaching the
published 0.87 target is not required for this pass.

### SCENARIO-LEARN-4116: Stable Resume Writes Reloadable Checkpoint

**Given** a loadable prior Exp 4108 checkpoint and CSV metrics containing
`val/exact_accuracy`
**When** Exp 4116 runs the bounded native trainer with `ckpt_path` pointed at
the stable shared checkpoint
**Then** it reports the real validation exact accuracy, cumulative epochs across
the resume lineage, a stable checkpoint path ending in
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`, reload success, bounded
duration, and a terminal honest verdict even when validation accuracy remains
below 0.87.

### SCENARIO-LEARN-4116-BLOCKED: Missing Runtime Resources Stop Honestly

**Given** `uv`, the native nano-trm trainer, or CUDA is unavailable
**When** Exp 4116 starts
**Then** it writes the corresponding `blocked_<resource>` artifact, records the
precondition evidence, leaves `checkpoint_reload_ok=false`, and does not
fabricate a validation exact-accuracy metric or checkpoint.

## Implementation Status (REQ-LEARN-4116)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4116 | Implemented (python/carnot/experiment_4116_sudoku_extreme_resume_pass1.py, Exp 4116) | Implemented (tests/python/test_experiment_4116_sudoku_extreme_resume_pass1.py) |

---

## REQ-LEARN-4117: Resumable nano-trm Sudoku Extreme Pass 2 Accumulation Floor

**Given** Exp 4116 established the shared Sudoku Extreme resume lineage, Exp
4117 SHALL read `results/experiment_4116_*.json` for the pass1
`stable_checkpoint_path` and `val_exact_accuracy`. If the pass1 JSON lacks the
validation exact-accuracy but names a Hydra run directory with
`val/exact_accuracy` CSV metrics, Exp 4117 MAY recover the pass1 baseline from
those metrics while recording that source. Before launch it SHALL check `uv`,
the native nano-trm trainer, CUDA, the Sudoku Extreme dataset, and that the
shared stable checkpoint exists and reloads with trusted full-checkpoint
loading. A missing or unloadable stable checkpoint SHALL write
`honest_verdict="blocked_stable_checkpoint_missing"` and stop without
fabricating validation metrics.

**When** the stable checkpoint is loadable, Exp 4117 SHALL resume from
`<stable>/last.ckpt` using Hydra experiment `trm_sudoku_extreme_1k_aug_1k`, CSV
logging, `trainer.max_time="00:01:00:00"`, and model-checkpoint overrides that
save `last.ckpt` back into the same stable directory. The run SHALL print
progress during training and checkpointing.

**Then** the terminal artifact SHALL be written to
`results/experiment_4117_sudoku_extreme_resume_pass2.json` and SHALL include
top-level fields `honest_verdict`, `val_exact_accuracy`,
`val_delta_vs_pass1`, `accumulation_stalled`, `stable_checkpoint_path`,
`duration_s`, and `cumulative_epochs`. `val_delta_vs_pass1` SHALL be computed
as pass2 `val_exact_accuracy` minus pass1 `val_exact_accuracy`.
`accumulation_stalled` SHALL be the bare bool `true` exactly when the delta is
less than or equal to zero. The acceptance gate passes when
`val_exact_accuracy` is reported and either `val_delta_vs_pass1 > 0` or
`accumulation_stalled` is flagged, making both improvement and honest stall
decision-grade outcomes.

### SCENARIO-LEARN-4117: Pass2 Reports Improvement Or Stalled Accumulation

**Given** a loadable Exp 4116 stable checkpoint and a pass1 validation exact
accuracy
**When** Exp 4117 runs the bounded native trainer with `ckpt_path` pointed at
the stable shared checkpoint
**Then** it reports pass2 validation exact accuracy, cumulative epochs, delta
versus pass1, a bare accumulation-stalled bool, the same stable checkpoint
path, bounded duration, and a terminal honest verdict.

### SCENARIO-LEARN-4117-BLOCKED: Broken Stable Checkpoint Stops The Lineage

**Given** the Exp 4116 stable checkpoint path is missing or unloadable
**When** Exp 4117 starts
**Then** it writes `honest_verdict="blocked_stable_checkpoint_missing"`,
records checkpoint-load evidence, leaves `val_exact_accuracy` and
`val_delta_vs_pass1` null, and does not run blind training.

## Implementation Status (REQ-LEARN-4117)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4117 | Implemented (python/carnot/experiment_4117_sudoku_extreme_resume_pass2.py, Exp 4117) | Implemented (tests/python/test_experiment_4117_sudoku_extreme_resume_pass2.py) |

---

## REQ-LEARN-4118: Resumable nano-trm Sudoku Extreme Pass 3 Convergence Verdict

**Given** Exp 4117 has written
`results/experiment_4117_sudoku_extreme_resume_pass2.json`, Exp 4118 SHALL
read that pass2 artifact before deciding whether to train. If pass2 reports
`val_exact_accuracy >= 0.85`, Exp 4118 SHALL take the
`early-converged-confirm` branch and write the convergence verdict without
launching another native trainer. If pass2 reports `accumulation_stalled=true`,
Exp 4118 SHALL take the `config-audit` branch, inspect the lr schedule, batch
size, nano-trm `trm_sudoku_extreme` configuration, and README recipe evidence,
and report a likely root cause rather than burning another blind GPU pass.
Otherwise Exp 4118 SHALL verify `uv`, the native nano-trm trainer, CUDA, the
Sudoku Extreme dataset, and the shared stable checkpoint before launching the
training branch.

**When** the training branch runs, Exp 4118 SHALL resume from the same shared
`results/trm_runs/sudoku_extreme_baseline/last.ckpt` checkpoint, use Hydra
experiment `trm_sudoku_extreme_1k_aug_1k`, CSV logging,
`trainer.max_time="00:01:00:00"`, and model-checkpoint overrides that save
`last.ckpt` back into the same stable directory. The run SHALL print progress
during training and checkpointing.

**Then** the terminal artifact SHALL be written to
`results/experiment_4118_sudoku_extreme_resume_pass3.json` and SHALL include
top-level fields `honest_verdict`, `val_exact_accuracy`,
`matches_published_087`, `total_cumulative_epochs`,
`stable_checkpoint_path`, and `branch_taken`. `matches_published_087` SHALL be
a bare bool computed as `abs(val_exact_accuracy - 0.87) <= 0.02` when a final
validation exact accuracy is known. The acceptance gate passes when a final
validation exact accuracy and `matches_published_087` are reported, or when a
config-audit branch reports a likely root cause for a stalled pass2 lineage.

### SCENARIO-LEARN-4118: Pass3 Trains Or Confirms Published Baseline

**Given** pass2 did not reach 0.85 and did not flag stalled accumulation
**When** Exp 4118 runs the bounded native trainer with `ckpt_path` pointed at
the shared stable checkpoint
**Then** it reports final validation exact accuracy, total cumulative epochs,
the same stable checkpoint path, branch_taken=`train`, and whether the result
matches published 0.87 within 0.02.

### SCENARIO-LEARN-4118-AUDIT: Stalled Pass2 Triggers Config Audit

**Given** Exp 4117 reports `accumulation_stalled=true`
**When** Exp 4118 starts
**Then** it does not launch training, records branch_taken=`config-audit`,
preserves the latest pass2 validation exact accuracy, and reports a likely
root cause using configuration evidence.

### SCENARIO-LEARN-4118-CONFIRM: Early Convergence Skips More Training

**Given** Exp 4117 already reports `val_exact_accuracy >= 0.85`
**When** Exp 4118 starts
**Then** it does not launch training, records branch_taken=`early-converged-confirm`,
preserves the shared stable checkpoint path, and computes
`matches_published_087` from the pass2 validation exact accuracy.

## Implementation Status (REQ-LEARN-4118)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4118 | Implemented (python/carnot/experiment_4118_sudoku_extreme_resume_pass3.py, Exp 4118) | Implemented (tests/python/test_experiment_4118_sudoku_extreme_resume_pass3.py) |

---

## REQ-LEARN-4126: nano-trm LR Resume Correctness Fix

**Given** the `.381` Sudoku Extreme resume lineage restored checkpoint epoch
and optimizer state but restarted the nano-trm manual learning-rate warmup each
bounded pass, Exp 4126 SHALL diagnose the trainer and model LR path before
running more resume passes. The diagnosis SHALL read
`nano-trm/src/nn/train.py`, `nano-trm/src/nn/models/trm.py`,
`nano-trm/src/nn/configs/model/trm.yaml`, and
`nano-trm/src/nn/configs/experiment/trm_sudoku_extreme_1k_aug_1k.yaml`, and
SHALL report whether the reset came from scheduler-not-restored,
local-step-warmup, missing manual scheduler checkpoint state, or another
identified cause.

**When** a resume checkpoint is loaded, nano-trm SHALL continue its manual LR
schedule from a persisted manual LR step. For legacy checkpoints without that
field, it SHALL infer the step from Lightning loop batch progress, then
`global_step`, then restored epoch evidence. The training step SHALL compute
the configured warmup/cosine LR from this resumed step for every optimizer
step, including post-warmup decay, rather than restarting warmup from zero or
pinning LR to a constant base rate after warmup.

**Then** Exp 4126 SHALL run one bounded native resume validation pass from
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`, inspect the new CSV
metrics, and write `results/experiment_4126_lr_resume_correctness_fix.json`
with required fields `honest_verdict`, `lr_rewarm_root_cause`,
`lr_continuous_across_resume`, `validation_first_lr`, `val_exact_accuracy`,
`stable_checkpoint_path`, and `duration_s`. The acceptance gate passes only
when `lr_continuous_across_resume=true` and the validation pass start LR is not
the fresh-warmup `2.45e-6` reset value, or when the artifact honestly reports a
diagnosed bounded-resume blocker and recommends a contiguous run alternative.

### SCENARIO-LEARN-4126: Resume Starts From Restored LR Step

**Given** a legacy Sudoku Extreme checkpoint whose Lightning `global_step` is
zero but whose fit-loop batch progress records completed training batches
**When** nano-trm loads that checkpoint and runs another bounded pass
**Then** the manual LR step is initialized from the completed batch progress,
the first logged `train/lr` is not the fresh-warmup `2.45e-6` value, and the
artifact records `lr_continuous_across_resume=true` with the observed first LR
and validation exact accuracy.

## Implementation Status (REQ-LEARN-4126)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4126 | Implemented (nano-trm/src/nn/models/trm.py, python/carnot/experiment_4126_lr_resume_correctness_fix.py, Exp 4126) | Implemented (tests/python/test_experiment_4126_lr_resume_correctness_fix.py) |

---

## REQ-LEARN-4127: Sudoku Extreme Accumulation With Fixed LR Resume

**Given** Exp 4126 has written
`results/experiment_4126_lr_resume_correctness_fix.json`, Exp 4127 SHALL read
that artifact before launching native training. If
`lr_continuous_across_resume` is not the bare bool `true`, Exp 4127 SHALL write
`results/experiment_4127_sudoku_extreme_accumulate_fixed.json` with
`honest_verdict="blocked_lr_fix_not_landed"`, SHALL restate the contiguous-run
recommendation, and SHALL NOT launch any more bounded resume passes that would
restart LR warmup.

**When** the LR-fix precondition passes and `uv`, nano-trm, CUDA, the Sudoku
Extreme dataset, and the shared stable checkpoint are available, Exp 4127
SHALL run up to two native bounded resume passes from
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`. Each pass SHALL use the
corrected manual LR schedule, a Lightning `trainer.max_time` bound of one hour
or less, CSV logging, terminal progress printing, and checkpoint saving back to
the same stable checkpoint path.

**Then** the terminal artifact SHALL contain top-level required fields
`honest_verdict`, `val_trajectory`, `matches_published_087`,
`per_pass_delta_vs_v381`, `stable_checkpoint_path`, and `duration_s`.
`val_trajectory` SHALL record the starting baseline and each measured pass
validation exact accuracy with deltas versus the previous known value.
`matches_published_087` SHALL be a bare bool computed as
`abs(final_val_exact_accuracy - 0.87) <= 0.02` when a final validation exact
accuracy is available. `per_pass_delta_vs_v381` SHALL compare the measured
delta(s) against the `.381` reference of roughly +0.01 validation exact
accuracy per pass. The acceptance gate passes when the trajectory is reported
and `matches_published_087` is set, or when the LR-fix precondition blocks
training with the contiguous-run recommendation.

### SCENARIO-LEARN-4127-BLOCKED: Missing LR Fix Stops Resume Training

**Given** the latest Exp 4126 artifact does not report
`lr_continuous_across_resume=true`
**When** Exp 4127 starts
**Then** it writes `honest_verdict="blocked_lr_fix_not_landed"`, preserves the
stable checkpoint path, includes the contiguous-run recommendation, and does
not call the native trainer.

### SCENARIO-LEARN-4127: Fixed Resume Reports Accumulation Trajectory

**Given** Exp 4126 reports `lr_continuous_across_resume=true` and the stable
checkpoint is loadable
**When** Exp 4127 runs one or two bounded native resume passes
**Then** it saves checkpoints back to
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`, reports a validation
trajectory for the measured passes, computes `matches_published_087`, and
reports whether the mean per-pass validation delta beats the `.381` +0.01/pass
reference.

## Implementation Status (REQ-LEARN-4127)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4127 | Implemented (python/carnot/experiment_4127_sudoku_extreme_accumulate_fixed.py, Exp 4127) | Implemented (tests/python/test_experiment_4127_sudoku_extreme_accumulate_fixed.py) |

---

## REQ-LEARN-4135: Sudoku Extreme Fixed-LR Accumulation Pass 1

**Given** Exp 4126 has proven the nano-trm resume schedule continues rather
than restarting fresh warmup, and Exp 4127 has established
`results/trm_runs/sudoku_extreme_baseline/last.ckpt` as the stable checkpoint
with validation exact accuracy `0.278`, Exp 4135 SHALL run one bounded native
resume pass from that stable checkpoint using
`experiment=trm_sudoku_extreme_1k_aug_1k`,
`data.data_dir=./data/sudoku_extreme_1k_aug_1k`, `seed=4108`,
`timekeeping.batch_size=128`, `ckpt_path=<stable>/last.ckpt`, and
`trainer.max_time` of no more than one hour.
If the stable checkpoint carries an exhausted Lightning `Timer` callback from a
prior bounded pass, Exp 4135 SHALL reset only that timer elapsed-state before
resume; it SHALL preserve model weights, optimizer state, and the
`nano_trm_manual_lr_step` schedule state so this timer cleanup cannot re-warm
the LR schedule.

**When** Exp 4135 starts, it SHALL check `uv`, `nano-trm/src/nn/train.py`,
CUDA availability, and that the stable checkpoint exists and is loadable before
launching training. If any precondition fails, Exp 4135 SHALL write
`results/experiment_4135_sudoku_accumulate_pass1_fixed_lr.json` with a
`blocked_<resource>` verdict, SHALL NOT launch the native trainer, and SHALL
still emit the required scalar artifact fields.

**Then** Exp 4135 SHALL save the resumed model back to
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`, read the new pass
Lightning CSV metrics, and write
`results/experiment_4135_sudoku_accumulate_pass1_fixed_lr.json` with top-level
fields `honest_verdict`, `val_exact_accuracy`, `delta_vs_previous`,
`lr_continued_not_rewarmed`, `matches_published_087`,
`stable_checkpoint_path`, `random_seed`, `reproducibility_checksum`,
`model_specs`, and scalar `duration_s`. `delta_vs_previous` SHALL compare the
measured validation exact accuracy against `0.278`.
`lr_continued_not_rewarmed` SHALL be a bare bool computed from the first
non-empty `train/lr` metric and SHALL be false when that first LR equals the
fresh warmup value `2.4500000108673703e-06`.
`reproducibility_checksum` SHALL hash the actual native config inputs, the
stable checkpoint, and the Sudoku Extreme dataset directory.

### SCENARIO-LEARN-4135-BLOCKED: Runtime Preconditions Stop Training

**Given** any of `uv`, the native nano-trm trainer, CUDA, or the loadable stable
checkpoint is missing
**When** Exp 4135 starts
**Then** it writes a blocked artifact with all required fields, preserves the
stable checkpoint path, and does not call the native trainer.

### SCENARIO-LEARN-4135: One Pass Reports Accuracy, Delta, and LR Continuity

**Given** all runtime preconditions pass and Exp 4126 reports
`lr_continuous_across_resume=true`
**When** Exp 4135 runs one bounded native resume pass
**Then** the artifact reports the pass validation exact accuracy, the delta
versus `0.278`, whether the LR continued instead of rewarmed, whether the pass
matches the published `0.87` target within `0.02`, the stable checkpoint path
that the next pass resumes from, seed `4108`, model specs naming nano-trm and
`trm_sudoku_extreme_1k_aug_1k`, and a scalar runtime below 4800 seconds.

## Implementation Status (REQ-LEARN-4135)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4135 | Planned (python/carnot/experiment_4135_sudoku_accumulate_pass1_fixed_lr.py, Exp 4135) | Planned (tests/python/test_experiment_4135_sudoku_accumulate_pass1_fixed_lr.py) |

---

## REQ-LEARN-4136: Sudoku Extreme Fixed-LR Accumulation Pass 2

**Given** Exp 4135 wrote
`results/experiment_4135_sudoku_accumulate_pass1_fixed_lr.json`, Exp 4136 SHALL
defensively read that pass1 artifact before any native training. If pass1 did
not report a numeric `val_exact_accuracy` and a positive numeric
`delta_vs_previous`, Exp 4136 SHALL NOT launch another bounded resume pass.
Instead, it SHALL run a short configuration audit covering scheduler horizon,
peak LR, data path, batch size versus the published recipe, and the pass1
runtime/metric evidence, then write
`results/experiment_4136_sudoku_accumulate_pass2_fixed_lr.json` with
`plateau_audit_done=true`, a suspected cause, and a corrected configuration
recommendation.

**When** pass1 did report positive accumulation, Exp 4136 SHALL check `uv`,
`nano-trm/src/nn/train.py`, CUDA availability, and that
`results/trm_runs/sudoku_extreme_baseline/last.ckpt` exists and is loadable.
It SHALL then run exactly one bounded native resume pass from that checkpoint
using `experiment=trm_sudoku_extreme_1k_aug_1k`,
`data.data_dir=./data/sudoku_extreme_1k_aug_1k`, `seed=4108`,
`timekeeping.batch_size=128`, `ckpt_path=<stable>/last.ckpt`, and
`trainer.max_time` of no more than one hour, saving back to the same stable
checkpoint.

**Then** Exp 4136 SHALL write top-level scalar fields `honest_verdict`,
`val_exact_accuracy`, `delta_vs_previous`, `plateau_audit_done`,
`matches_published_087`, `stable_checkpoint_path`, `random_seed`,
`reproducibility_checksum`, and `duration_s`. `delta_vs_previous` SHALL compare
the pass2 validation exact accuracy against pass1's measured
`val_exact_accuracy` when training runs. `matches_published_087` SHALL be true
only when the pass2 validation exact accuracy is within `0.02` of `0.87`.
`reproducibility_checksum` SHALL hash the native config inputs, the resumed
stable checkpoint, and the Sudoku Extreme dataset directory.

### SCENARIO-LEARN-4136-AUDIT: Pass1 Without Positive Delta Stops Training

**Given** pass1 has missing, zero, or negative accumulation evidence
**When** Exp 4136 starts
**Then** it writes a short audit artifact with `plateau_audit_done=true`, does
not call the native trainer, reports the suspected cause and corrected config
recommendation, and keeps runtime below 600 seconds.

### SCENARIO-LEARN-4136: One Pass Reports Accuracy Delta From Pass1

**Given** pass1 reported a numeric validation exact accuracy and positive
delta, all runtime preconditions pass, and Exp 4126 reports
`lr_continuous_across_resume=true`
**When** Exp 4136 runs one bounded native resume pass
**Then** the artifact reports pass2 validation exact accuracy, the delta versus
pass1, whether the result matches the published `0.87` target within `0.02`,
the stable checkpoint path that Exp 4137 resumes from, seed `4108`, a
reproducibility checksum, and scalar runtime below 4800 seconds.

## Implementation Status (REQ-LEARN-4136)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4136 | Planned (python/carnot/experiment_4136_sudoku_accumulate_pass2_fixed_lr.py, Exp 4136) | Planned (tests/python/test_experiment_4136_sudoku_accumulate_pass2_fixed_lr.py) |

---

## REQ-LEARN-4137: Sudoku Extreme Fixed-LR Accumulation Pass 3

**Given** Exp 4136 wrote
`results/experiment_4136_sudoku_accumulate_pass2_fixed_lr.json`, Exp 4137 SHALL
defensively read that pass2 artifact before any native training. If pass2 did
not report a numeric `val_exact_accuracy`, a positive numeric
`delta_vs_previous`, or if pass2 already set `plateau_audit_done=true`, Exp
4137 SHALL NOT launch another bounded resume pass. Instead, it SHALL report
that the baseline is config-blocked before the `.384` pass, preserve the
corrected configuration recommendation, and write
`results/experiment_4137_sudoku_accumulate_pass3_fixed_lr.json` with
`plateau_audit_done=true`.

**When** pass2 did report positive accumulation, Exp 4137 SHALL check `uv`,
`nano-trm/src/nn/train.py`, CUDA availability, and that
`results/trm_runs/sudoku_extreme_baseline/last.ckpt` exists and is loadable.
It SHALL then run exactly one bounded native resume pass from that checkpoint
using `experiment=trm_sudoku_extreme_1k_aug_1k`,
`data.data_dir=./data/sudoku_extreme_1k_aug_1k`, `seed=4108`,
`timekeeping.batch_size=128`, `ckpt_path=<stable>/last.ckpt`, and
`trainer.max_time` of no more than one hour, saving back to the same stable
checkpoint.

**Then** Exp 4137 SHALL write top-level scalar fields `honest_verdict`,
`val_exact_accuracy`, `delta_vs_previous`, `plateau_audit_done`,
`matches_published_087`, `stable_checkpoint_path`, `random_seed`,
`reproducibility_checksum`, and `duration_s`. `delta_vs_previous` SHALL compare
the pass3 validation exact accuracy against pass2's measured
`val_exact_accuracy` when training runs. `matches_published_087` SHALL be true
only when the pass3 validation exact accuracy is within `0.02` of `0.87`.
`reproducibility_checksum` SHALL hash the native config inputs, the resumed
stable checkpoint, and the Sudoku Extreme dataset directory.

### SCENARIO-LEARN-4137-AUDIT: Pass2 Without Positive Delta Stops Training

**Given** pass2 has missing, zero, or negative accumulation evidence, or pass2
already performed a plateau audit
**When** Exp 4137 starts
**Then** it writes a short audit artifact with `plateau_audit_done=true`, does
not call the native trainer, reports the baseline as config-blocked, preserves
the corrected config recommendation, and keeps runtime below 600 seconds.

### SCENARIO-LEARN-4137: One Pass Reports Accuracy Delta From Pass2

**Given** pass2 reported a numeric validation exact accuracy and positive
delta, all runtime preconditions pass, and Exp 4126 reports
`lr_continuous_across_resume=true`
**When** Exp 4137 runs one bounded native resume pass
**Then** the artifact reports pass3 validation exact accuracy, the delta versus
pass2, whether the result matches the published `0.87` target within `0.02`,
the stable checkpoint path that Exp 4138 resumes from, seed `4108`, a
reproducibility checksum, and scalar runtime below 4800 seconds.

## Implementation Status (REQ-LEARN-4137)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4137 | Planned (python/carnot/experiment_4137_sudoku_accumulate_pass3_fixed_lr.py, Exp 4137) | Planned (tests/python/test_experiment_4137_sudoku_accumulate_pass3_fixed_lr.py) |

---

## REQ-LEARN-4138: Sudoku Extreme Fixed-LR Accumulation Pass 4

**Given** Exp 4137 wrote
`results/experiment_4137_sudoku_accumulate_pass3_fixed_lr.json`, Exp 4138 SHALL
defensively read that pass3 artifact before any native training. If pass3 did
not report a numeric `val_exact_accuracy`, a positive numeric
`delta_vs_previous`, or if pass3 already set `plateau_audit_done=true` or
`baseline_status=config-blocked`, Exp 4138 SHALL NOT launch another bounded
resume pass. Instead, it SHALL write
`results/experiment_4138_sudoku_accumulate_pass4_convergence_check.json` with
an honest config-blocked verdict and preserve the corrected configuration
recommendation for the `.384` baseline.

**When** pass3 did report positive accumulation, Exp 4138 SHALL check `uv`,
`nano-trm/src/nn/train.py`, CUDA availability, and that
`results/trm_runs/sudoku_extreme_baseline/last.ckpt` exists and is loadable.
It SHALL then run exactly one bounded native resume pass from that checkpoint
using `experiment=trm_sudoku_extreme_1k_aug_1k`,
`data.data_dir=./data/sudoku_extreme_1k_aug_1k`, `seed=4108`,
`timekeeping.batch_size=128`, `ckpt_path=<stable>/last.ckpt`, and
`trainer.max_time` of no more than one hour, saving back to the same stable
checkpoint.

**Then** Exp 4138 SHALL write top-level scalar fields `honest_verdict`,
`val_exact_accuracy`, `val_trajectory_383`, `matches_published_087`,
`near_faithful_080`, `estimated_passes_to_converge`,
`stable_checkpoint_path`, `random_seed`, `reproducibility_checksum`, and
`duration_s`. `val_trajectory_383` SHALL report the `.382` anchor and each
`.383` accumulation pass through pass4. `matches_published_087` SHALL be a bare
bool that is true only when validation exact accuracy is within `0.02` of
`0.87`. `near_faithful_080` SHALL be a bare bool that is true only when
validation exact accuracy is at least `0.80`. If the pass4 validation exact
accuracy is still below `0.80`, `estimated_passes_to_converge` SHALL estimate
the remaining passes to `0.87` from the observed positive per-pass validation
deltas when such an estimate is possible.

### SCENARIO-LEARN-4138-AUDIT: Pass3 Config-Blocked Stops Training

**Given** pass3 has missing, zero, or negative accumulation evidence, already
performed a plateau audit, or reports `baseline_status=config-blocked`
**When** Exp 4138 starts
**Then** it writes a config-blocked artifact, does not call the native trainer,
reports `matches_published_087=false` and `near_faithful_080=false`, preserves
the `.384` corrected config recommendation, reports the `.382+.383`
trajectory available so far, and keeps runtime below 600 seconds.

### SCENARIO-LEARN-4138: Pass4 Gates Faithful and Near-Faithful Baselines

**Given** pass3 reported a numeric validation exact accuracy and positive
delta, all runtime preconditions pass, and Exp 4126 reports
`lr_continuous_across_resume=true`
**When** Exp 4138 runs one bounded native resume pass
**Then** the artifact reports pass4 validation exact accuracy, the full
`.382+.383` validation trajectory through pass4, whether the result matches
the published `0.87` target within `0.02`, whether it is near-faithful at
`>=0.80`, the stable checkpoint path that Exp 4139 grafts onto, seed `4108`, a
reproducibility checksum, and scalar runtime below 4800 seconds.

## Implementation Status (REQ-LEARN-4138)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4138 | Planned (python/carnot/experiment_4138_sudoku_accumulate_pass4_convergence_check.py, Exp 4138) | Planned (tests/python/test_experiment_4138_sudoku_accumulate_pass4_convergence_check.py) |

---

## REQ-LEARN-4146: Sudoku Extreme Pass 1 Epoch-Ceiling No-Op Guard

**Given** the `.384` pass resumes from
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`, Exp 4146 SHALL verify
`uv`, `nano-trm/src/nn/train.py`, CUDA availability, and that the stable
checkpoint exists and is loadable before deciding whether native training is
allowed. It SHALL read the checkpoint seed epoch, global step, and Lightning
timer elapsed state from that same checkpoint. It SHALL also read the
`trm_sudoku_extreme_1k_aug_1k` experiment config and resolve the current
`trainer.max_epochs` ceiling from `timekeeping.max_epochs`.

**When** the checkpoint seed epoch is greater than or equal to the resolved
max-epochs ceiling, Exp 4146 SHALL set `max_epochs_cap_confirmed=true` and may
launch the native trainer only with an epoch ceiling raised above the checkpoint
epoch by approximately 3000 epochs, a one-day `trainer.max_time` bound, CSV
logging, progress printing, and ModelCheckpoint writing `last.ckpt` back into
`results/trm_runs/sudoku_extreme_baseline`. The command SHALL resume
`ckpt_path=<repo>/results/trm_runs/sudoku_extreme_baseline/last.ckpt`, use
`experiment=trm_sudoku_extreme_1k_aug_1k`, `data.data_dir=./data/sudoku_extreme_1k_aug_1k`,
`seed=4108`, and preserve the fixed-LR resume environment from Exp 4127.

**Then** Exp 4146 SHALL write
`results/experiment_4146_sudoku_accumulate_pass1_epochfix.json` with the
top-level fields `honest_verdict`, `max_epochs_cap_confirmed`, `seed_epoch`,
`post_epoch`, `val_exact_accuracy`, `stable_checkpoint_path`, `duration_s`, and
`random_seed`. A `complete:` verdict is valid only when `duration_s > 120`,
`post_epoch > seed_epoch`, and `val_exact_accuracy` is a real numeric
validation exact accuracy parsed from the CSV metrics. If any anti-no-op proof
fails, the artifact SHALL use an honest `blocked_noop_*` verdict that names the
diagnosed cause instead of claiming progress. If runtime preconditions fail,
the artifact SHALL use the relevant `blocked_<resource>` verdict and SHALL NOT
launch training.

### SCENARIO-LEARN-4146-BLOCKED-NOOP: Diagnosis Blocks A False Epoch-Cap Fix

**Given** the stable checkpoint loads but its seed epoch is below the resolved
max-epochs ceiling, or another checkpointed stop state already explains an
instant return
**When** Exp 4146 starts
**Then** it writes a `blocked_noop_*` artifact, reports
`max_epochs_cap_confirmed=false`, preserves the measured seed epoch, keeps
`post_epoch` equal to the seed epoch, reports `val_exact_accuracy=null`, records
the diagnosed cause, and does not call the native trainer.

### SCENARIO-LEARN-4146: Bounded Epoch-Fixed Pass Proves Real Training

**Given** the checkpoint seed epoch is at the resolved max-epochs ceiling and
all runtime preconditions pass
**When** Exp 4146 runs the bounded native epoch-fixed resume pass
**Then** the artifact reports `max_epochs_cap_confirmed=true`, the original
seed epoch, a post-pass epoch greater than that seed epoch, a real
`val_exact_accuracy`, the shared stable checkpoint path, seed `4108`, and a
duration above 120 seconds.

## Implementation Status (REQ-LEARN-4146)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4146 | Implemented (python/carnot/experiment_4146_sudoku_accumulate_pass1_epochfix.py, Exp 4146) | Implemented (tests/python/test_experiment_4146_sudoku_accumulate_pass1_epochfix.py) |

---

## REQ-LEARN-4147: Sudoku Extreme Pass 2 Epoch-Fixed No-Op Continuation

**Given** Exp 4147 is asked to continue the Exp 4146 Sudoku Extreme
epoch-fixed lineage from
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`, it SHALL first verify
`uv`, `nano-trm/src/nn/train.py`, CUDA availability, and that the stable
checkpoint exists and is loadable. It SHALL then read
`results/experiment_4146_sudoku_accumulate_pass1_epochfix.json` before any
native trainer launch.

**When** the Exp 4146 pass1 artifact reports an honest `blocked_noop_*`
verdict or otherwise lacks proof of real pass1 training, Exp 4147 SHALL NOT
blind-retrain. It SHALL write
`results/experiment_4147_sudoku_accumulate_pass2.json` with
`honest_verdict="blocked_pass1_noop_unresolved"`, `val_exact_accuracy=null`,
`delta_vs_pass1=null`, `post_epoch` copied from the pass1 artifact when
available, a bounded `duration_s`, and a cause summary that restates the pass1
configuration diagnosis.

**When** pass1 is decision-grade and runtime preconditions pass, Exp 4147 SHALL
launch a single bounded resume pass using
`ckpt_path=<repo>/results/trm_runs/sudoku_extreme_baseline/last.ckpt`,
`+trainer.max_epochs=<current_epoch+3000>`, `+trainer.max_time="00:01:00:00"`,
CSV logging, progress printing, and ModelCheckpoint writing `last.ckpt` back
to `results/trm_runs/sudoku_extreme_baseline`.

**Then** a non-blocked Exp 4147 artifact SHALL report `honest_verdict`,
`val_exact_accuracy`, `delta_vs_pass1`, `post_epoch`, and `duration_s` with
field principles matching the operator schema. A successful or plateau verdict
is valid only when `duration_s > 120`, `post_epoch` exceeds pass1's epoch, and
`val_exact_accuracy` is numeric. Positive `delta_vs_pass1` reports continued
convergence; non-positive delta is valid only with an explicit honest plateau
flag. The blocked pass1 verdict is also decision-grade because it stops before
a fake pass2 claim.

### SCENARIO-LEARN-4147-BLOCKED-PASS1: Upstream No-Op Stops Pass 2

**Given** Exp 4146 reports `blocked_noop_cap_not_confirmed_timer_elapsed`,
`post_epoch` equals the seed epoch, `duration_s < 120`, and
`val_exact_accuracy=null`
**When** Exp 4147 starts
**Then** it writes `blocked_pass1_noop_unresolved`, records the pass1
configuration cause, preserves null solve metrics, and never calls the native
trainer.

### SCENARIO-LEARN-4147: Bounded Pass 2 Reports Delta Or Plateau

**Given** Exp 4146 reports a decision-grade numeric validation accuracy and
the stable checkpoint loads
**When** Exp 4147 runs one bounded native pass2 resume
**Then** the artifact reports a terminal-prefixed honest verdict, the pass2
validation exact accuracy, `delta_vs_pass1`, a post-pass epoch greater than
pass1's epoch, and either positive improvement or an honest plateau flag.

## Implementation Status (REQ-LEARN-4147)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4147 | Implemented (python/carnot/experiment_4147_sudoku_accumulate_pass2.py, Exp 4147) | Implemented (tests/python/test_experiment_4147_sudoku_accumulate_pass2.py) |

---

## REQ-LEARN-4148: Sudoku Extreme Pass 3 From Pass 2 Stable Checkpoint

**Given** Exp 4148 is asked to continue the Exp 4147 Sudoku Extreme
epoch-fixed lineage from
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`, it SHALL first verify
`uv`, `nano-trm/src/nn/train.py`, CUDA availability, and that the stable
checkpoint exists and is loadable. It SHALL then read
`results/experiment_4147_sudoku_accumulate_pass2.json` before any native
trainer launch.

**When** the Exp 4147 pass2 artifact reports `val_exact_accuracy >= 0.87`,
Exp 4148 SHALL skip native training and write
`results/experiment_4148_sudoku_accumulate_pass3.json` as an early-converged
confirmation with the pass2 validation accuracy, `delta_vs_pass2=0.0`,
`post_epoch` copied from pass2, and `native_trainer_launched=false`.

**When** the Exp 4147 pass2 artifact reports an honest `blocked_*` or
`blocked_noop_*` verdict or otherwise lacks proof of real pass2 training, Exp
4148 SHALL NOT blind-retrain. It SHALL write
`results/experiment_4148_sudoku_accumulate_pass3.json` with an honest upstream
blocked verdict, `val_exact_accuracy=null`, `delta_vs_pass2=null`,
`post_epoch` copied from pass2 when available, a bounded `duration_s`, and a
cause summary that restates the pass2 no-op lineage.

**When** pass2 is decision-grade, below the early-convergence threshold, and
runtime preconditions pass, Exp 4148 SHALL launch a single bounded resume pass
using `ckpt_path=<repo>/results/trm_runs/sudoku_extreme_baseline/last.ckpt`,
`+trainer.max_epochs=<current_epoch+3000>`, `+trainer.max_time="00:01:00:00"`,
CSV logging, progress printing, and ModelCheckpoint writing `last.ckpt` back
to `results/trm_runs/sudoku_extreme_baseline`.

**Then** the Exp 4148 artifact SHALL include top-level fields
`honest_verdict`, `val_exact_accuracy`, `delta_vs_pass2`, `post_epoch`, and
`duration_s`, with field principles stating terminal-prefixed honesty, solve
metric after pass3, per-pass convergence delta, epoch-advance anti-no-op, and
real bounded runtime. The acceptance gate SHALL pass only for a real trained
pass with `duration_s > 120`, epoch advance, and a real validation accuracy; an
early-converged confirmation; or an honest plateau/upstream no-op.

### SCENARIO-LEARN-4148-BLOCKED-PASS2: Upstream No-Op Stops Pass 3

**Given** Exp 4147 reports `blocked_pass1_noop_unresolved`, `duration_s < 120`,
and `val_exact_accuracy=null`
**When** Exp 4148 starts
**Then** it writes an upstream blocked artifact, preserves null solve metrics,
copies the pass2 epoch, records the pass2 blocker, and never calls the native
trainer.

### SCENARIO-LEARN-4148-EARLY-CONVERGED: Pass 2 Already Meets Target

**Given** Exp 4147 reports `val_exact_accuracy >= 0.87`
**When** Exp 4148 starts
**Then** it writes a terminal early-converged confirmation, reports
`delta_vs_pass2=0.0`, preserves the pass2 checkpoint epoch, and skips native
training.

### SCENARIO-LEARN-4148: Bounded Pass 3 Reports Delta Or Plateau

**Given** Exp 4147 reports a decision-grade numeric validation accuracy below
`0.87` and the stable checkpoint loads
**When** Exp 4148 runs one bounded native pass3 resume
**Then** the artifact reports a terminal-prefixed honest verdict, the pass3
validation exact accuracy, `delta_vs_pass2`, a post-pass epoch greater than
pass2's epoch, and either positive improvement or an honest plateau flag.

## Implementation Status (REQ-LEARN-4148)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4148 | Implemented (python/carnot/experiment_4148_sudoku_accumulate_pass3.py, Exp 4148) | Implemented (tests/python/test_experiment_4148_sudoku_accumulate_pass3.py) |

---

## REQ-LEARN-4149: Sudoku Extreme Pass 4 Final Accumulation Decision

**Given** Exp 4149 is the final `.384` accumulation pass before Exp 4150 and
`.385` decide whether a faithful Sudoku Extreme baseline exists, it SHALL read
`results/experiment_4148_sudoku_accumulate_pass3.json` before any native
trainer launch. It SHALL also recover the `.384` validation trajectory from
the activated `0.278` baseline plus Exp 4146, Exp 4147, and Exp 4148, preserving
null pass metrics when an upstream pass stopped as a no-op.

**When** Exp 4148 already reports `val_exact_accuracy >= 0.87`, Exp 4149
SHALL skip native training and write
`results/experiment_4149_sudoku_accumulate_pass4_convergence.json` as an
early-converged confirmation. The artifact SHALL report the final validation
exact accuracy, set `matches_published_087` as a bare bool using the
`abs(val_exact_accuracy - 0.87) <= 0.02` rule, keep the stable checkpoint path,
and include the full `.384` validation trajectory from `0.278`.

**When** Exp 4148 reports an upstream `blocked_*` or `blocked_noop_*` verdict,
or otherwise lacks proof of real pass3 training, Exp 4149 SHALL surface the
upstream blocker and STOP before training. The blocked artifact SHALL still
report the effective final accumulated validation accuracy inherited from the
stable `0.278` baseline when no later pass reports a real validation metric,
set `matches_published_087=false`, preserve the stable checkpoint path, and
include the `.384` trajectory showing the blocked passes.

**When** Exp 4148 is decision-grade, below the early-convergence threshold, and
runtime preconditions pass, Exp 4149 SHALL launch one bounded native resume pass
using `ckpt_path=<repo>/results/trm_runs/sudoku_extreme_baseline/last.ckpt`,
`+trainer.max_epochs=<current_epoch+3000>`, `+trainer.max_time="00:01:00:00"`,
CSV logging, progress printing, and ModelCheckpoint writing `last.ckpt` back to
`results/trm_runs/sudoku_extreme_baseline`.

**Then** Exp 4149 SHALL write the terminal artifact
`results/experiment_4149_sudoku_accumulate_pass4_convergence.json` with
top-level fields `honest_verdict`, `val_exact_accuracy`,
`matches_published_087`, `val_trajectory_v384`, `stable_checkpoint_path`, and
`duration_s`, each carrying the required operator principle text. A complete
trained verdict is valid only when `duration_s > 120`, the checkpoint epoch
advances beyond Exp 4148, and a real validation exact accuracy is reported. An
early-converged confirmation or honest upstream no-op is also acceptance-grade
when the final validation value and `matches_published_087` bool are present.

### SCENARIO-LEARN-4149-BLOCKED-PASS3: Upstream No-Op Stops Pass 4

**Given** Exp 4148 reports `blocked_pass2_noop_unresolved`,
`val_exact_accuracy=null`, and a stable checkpoint path
**When** Exp 4149 starts
**Then** it writes `blocked_pass3_noop_unresolved`, never calls the native
trainer, reports the effective final validation accuracy from the `.384`
baseline, sets `matches_published_087=false`, and includes a trajectory from
`0.278` through the blocked pass1, pass2, and pass3 artifacts.

### SCENARIO-LEARN-4149-EARLY-CONVERGED: Pass 3 Already Meets Target

**Given** Exp 4148 reports `val_exact_accuracy >= 0.87`
**When** Exp 4149 starts
**Then** it writes a terminal early-converged confirmation, reports the pass3
validation exact accuracy as the final accumulated value, sets
`matches_published_087` from the `0.02` tolerance rule, preserves the stable
checkpoint path, and skips native training.

### SCENARIO-LEARN-4149: Bounded Pass 4 Reports Final Accumulated Metric

**Given** Exp 4148 reports a decision-grade numeric validation accuracy below
`0.87` and the stable checkpoint loads
**When** Exp 4149 runs one bounded native pass4 resume
**Then** the artifact reports a terminal-prefixed honest verdict, the final
pass4 validation exact accuracy, a bare `matches_published_087` bool, the full
`.384` validation trajectory, and anti-no-op proof through runtime and epoch
advance.

## Implementation Status (REQ-LEARN-4149)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4149 | Proposed (python/carnot/experiment_4149_sudoku_accumulate_pass4_convergence.py, Exp 4149) | Proposed (tests/python/test_experiment_4149_sudoku_accumulate_pass4_convergence.py) |

---

## REQ-LEARN-4150: Decision-Grade Sudoku Verifier Graft Gate

**Given** Exp 4149 has written
`results/experiment_4149_sudoku_accumulate_pass4_convergence.json`, Exp 4150
SHALL read `val_exact_accuracy`, `stable_checkpoint_path`, and the recorded
runtime preconditions before attempting any verifier-as-reward graft. It SHALL
also verify CUDA visibility and that the stable checkpoint path is present and
loadable, recording all checks in `preconditions_checked`.

**When** Exp 4149 reports `val_exact_accuracy < 0.85`, Exp 4150 SHALL stop
before reranking or RFT training, set `graft_deferred=true`, report the current
baseline validation accuracy, and estimate additional `.385` passes needed to
reach `0.85`. This honest deferral is acceptance-grade because a graft on an
unfaithful baseline repeats the `.383` false-negative anti-pattern.

**When** Exp 4149 reports `val_exact_accuracy >= 0.85` and preconditions pass,
Exp 4150 SHALL run the real graft branch: sample K candidate Sudoku solutions
per held-out puzzle, rerank with the executable Sudoku verifier, report
`rerank_lift_vs_vote` against TRM-vote and oracle with bootstrap CI, and run an
N-matched RFT label contrast where A is verifier-certified labels and B is
vote-certified labels from the same TRM.

**Then** Exp 4150 SHALL write
`results/experiment_4150_decisive_verifier_graft_sudoku.json` with top-level
fields `honest_verdict`, `graft_deferred`, `rerank_lift_vs_vote`,
`rft_vs_ablation_delta`, `verifier_value_added`, and `preconditions_checked`,
each with principle text. The acceptance gate passes when either
`graft_deferred=true` with baseline status for `val<0.85`, or
`graft_deferred=false` with `rerank_lift_vs_vote`,
`rft_vs_ablation_delta`, and `verifier_value_added` reported with CI95.

### SCENARIO-LEARN-4150-DEFER: Non-Faithful Baseline Stops The Graft

**Given** Exp 4149 reports validation accuracy below `0.85`
**When** Exp 4150 starts
**Then** it writes a terminal-prefixed honest deferral, sets
`graft_deferred=true`, records the baseline checkpoint and CUDA preconditions,
reports current validation accuracy, estimates `.385` passes to converge, and
does not sample candidates or launch RFT.

### SCENARIO-LEARN-4150-GRAFT: Faithful Baseline Runs The Measurement

**Given** Exp 4149 reports validation accuracy at least `0.85`
**When** Exp 4150 has candidate pools from the same TRM
**Then** it reports `rerank_lift_vs_vote` with CI95 and an oracle comparison,
reports `rft_vs_ablation_delta` with CI95 from the N-matched label contrast,
and sets `verifier_value_added` from the A-vs-B delta rather than from an
uninformative non-faithful graft.

## Implementation Status (REQ-LEARN-4150)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4150 | Planned (python/carnot/experiment_4150_decisive_verifier_graft_sudoku.py, Exp 4150) | Planned (tests/python/test_experiment_4150_decisive_verifier_graft_sudoku.py) |

---

## REQ-LEARN-4157: Contiguous Sudoku Baseline Harvest And Continue

**Given** `.384` bounded passes no-op'd on stale Timer state and the `.385`
baseline decision depends on the operator's single contiguous nano-trm Sudoku
Extreme run, Exp 4157 SHALL read
`results/trm_runs/contiguous_run_hydra/csv/version_*/metrics.csv` directly
without loading a model or touching CUDA. It SHALL parse `val/exact_accuracy`
rows across all CSV versions, report the latest validation exact accuracy as
`current_val`, report the best validation exact accuracy as `max_val`, and
preserve the full `val_trajectory`.

**When** Exp 4157 reads the shared stable checkpoint
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`, it SHALL use
`torch.load(..., map_location="cpu", weights_only=False)` only to extract the
top-level `epoch` and `nano_trm_manual_lr_step` scalars. It SHALL never
instantiate the nano-trm model, move tensors to GPU, or use epoch advance as
the launch/no-op guard for this milestone.

**When** `results/trm_runs/contiguous_run.pid` names a live process and the
newest contiguous-run CSV validation epoch is advancing, Exp 4157 SHALL record
only. It SHALL write
`results/experiment_4157_baseline_harvest_contiguous_continue.json` with
`run_alive=true`, `baseline_faithful` set from `current_val >= 0.85`, the
checkpoint `manual_lr_step`, and an honest terminal verdict such as
`complete: baseline_advancing_contiguous_run_live_val_0.NN`. It SHALL NOT
launch a competing trainer while the live contiguous run owns the GPU and
checkpoint write-back.

**When** the contiguous run is dead and `current_val >= 0.85`, Exp 4157 SHALL
confirm the faithful baseline without training and write a terminal verdict
`complete: baseline_faithful_val_0.NN`.

**When** the contiguous run is dead and `current_val < 0.85`, Exp 4157 SHALL
resume one contiguous native nano-trm run from
`results/trm_runs/sudoku_extreme_baseline/last.ckpt` using the same environment
mechanics as Exp 4127 (`DISABLE_COMPILE=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and W&B disabled) and the
single long Hydra command rooted at
`results/trm_runs/contiguous_run_hydra` with `+trainer.max_time=00:11:30:00`.
After a launch by this task, Exp 4157 SHALL use a step-based anti-no-op guard:
`nano_trm_manual_lr_step` must advance and a new contiguous-run CSV validation
row must be written. A stalled manual LR step SHALL produce the honest verdict
`blocked_noop_step_unchanged` with the diagnosed cause.

**Then** the Exp 4157 artifact SHALL include top-level fields
`honest_verdict`, `current_val`, `max_val`, `baseline_faithful`, `run_alive`,
`manual_lr_step`, `val_trajectory`, `stable_checkpoint_path`, and
`estimated_passes_to_085`, each carrying the required operator principle text
where applicable. The acceptance gate passes when `current_val` is read,
`run_alive` is recorded as a bare bool, `baseline_faithful` is recorded as a
bare bool, and either the run is record-only/live, the baseline is already
faithful, or a task-launched run proves both manual-step advance and a real new
validation row.

### SCENARIO-LEARN-4157-LIVE: Live Contiguous Run Is Record-Only

**Given** the PID file resolves to a live process and the newest contiguous CSV
validation epoch advances
**When** Exp 4157 starts
**Then** it writes the harvest artifact with `run_alive=true`, records latest
and max validation accuracy from the CSV, records the CPU-only checkpoint
manual LR step, sets `baseline_faithful` from the `0.85` threshold, and never
launches a native trainer.

### SCENARIO-LEARN-4157-FAITHFUL: Dead Run Already Reached The Gate

**Given** the PID file is stale or absent and the latest contiguous CSV
validation exact accuracy is at least `0.85`
**When** Exp 4157 starts
**Then** it writes a faithful complete verdict, sets `baseline_faithful=true`,
and skips native training.

### SCENARIO-LEARN-4157-CONTINUE: Dead Run Below Gate Relaunches Once

**Given** the PID file is stale or absent and the latest contiguous CSV
validation exact accuracy is below `0.85`
**When** Exp 4157 launches the native contiguous resume
**Then** the artifact is acceptance-grade only if `nano_trm_manual_lr_step`
advances and a new `val/exact_accuracy` row appears in the contiguous CSV;
otherwise it reports `blocked_noop_step_unchanged` instead of a fake pass.

## Implementation Status (REQ-LEARN-4157)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4157 | Implemented (python/carnot/experiment_4157_baseline_harvest_contiguous_continue.py, Exp 4157) | Implemented (tests/python/test_experiment_4157_baseline_harvest_contiguous_continue.py) |

---

## REQ-LEARN-4167: Outer-Loop TRM Training Monitor Is Read-Only

**Given** the conductor ceded contiguous nano-TRM Sudoku Extreme training to the
outer loop after `.385` collisions, Exp 4167 SHALL only report status for the
outer-loop-owned run. It SHALL NOT launch native training, SHALL NOT terminate a
`train.py` process, and SHALL NOT write anything under
`results/trm_runs/sudoku_extreme_baseline/`.

**When** Exp 4167 reads the stable checkpoint
`results/trm_runs/sudoku_extreme_baseline/last.ckpt`, it SHALL record the
checkpoint mtime and use `torch.load(..., map_location="cpu",
weights_only=False)` only to extract the top-level `epoch` scalar.

**When** Exp 4167 reads `results/trm_runs/contiguous_run.pid`, it SHALL check
that PID with `ps -o etime= -p <pid>` and report the result as the bare bool
`outerloop_train_alive`.

**When** Exp 4167 reads contiguous-run metrics, it SHALL find
`results/trm_runs/contiguous_run_hydra/**/metrics.csv`, parse
`val/exact_accuracy` or `val_exact_accuracy` rows, report the latest value as
`current_val_exact_accuracy`, preserve the parsed `val_trajectory`, and report
whether the latest value crossed the `0.85` faithful threshold.

**Then** the Exp 4167 artifact SHALL be written to
`results/experiment_4167_outerloop_training_monitor.json` with required
top-level fields `honest_verdict`, `outerloop_train_alive`,
`current_val_exact_accuracy`, `baseline_faithful`, and `checkpoint_mtime`.
`baseline_faithful` SHALL be a bare bool that is true only when
`current_val_exact_accuracy >= 0.85` and `outerloop_train_alive == false`, so
the graft only runs against a faithful and stable checkpoint.

### SCENARIO-LEARN-4167-READONLY-MONITOR: Live Outer-Loop Run Blocks Graft Stability

**Given** the PID file resolves to a live process and the latest contiguous CSV
validation row is below `0.85`
**When** Exp 4167 runs
**Then** it writes a complete status report with `outerloop_train_alive=true`,
the latest validation value, checkpoint mtime, checkpoint epoch, full
trajectory, and `baseline_faithful=false`, without launching or stopping any
training.

### SCENARIO-LEARN-4167-FAITHFUL-STABLE: Stopped Run Above Threshold Unlocks Graft

**Given** the PID file is stale or absent and the latest contiguous CSV
validation exact accuracy is at least `0.85`
**When** Exp 4167 runs
**Then** it writes a complete status report with `outerloop_train_alive=false`,
`current_val_exact_accuracy >= 0.85`, and `baseline_faithful=true`.

## Implementation Status (REQ-LEARN-4167)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4167 | Planned (python/carnot/experiment_4167_outerloop_training_monitor.py, Exp 4167) | Planned (tests/python/test_experiment_4167_outerloop_training_monitor.py) |

---

## REQ-LEARN-4158: Decision-Grade Sudoku Verifier Rerank Recovery Moat

**Given** Exp 4157 has written
`results/experiment_4157_baseline_harvest_contiguous_continue.json`, Exp 4158
SHALL read `current_val`, `max_val`, and `stable_checkpoint_path`, verify CUDA
visibility, and verify that the stable checkpoint exists before any model load.
It SHALL copy `last.ckpt` to a frozen sibling snapshot path such as
`last-rerank-snapshot.ckpt` before sampling so a live checkpoint writer cannot
produce a half-read verifier-rerank result. Missing CUDA, missing Exp 4157
evidence, or missing/un-snapshot-able checkpoint SHALL produce a terminal
`blocked_<resource>` artifact rather than fabricated rerank metrics.

**When** runtime preconditions pass, Exp 4158 SHALL sample at least `K=8`
candidate Sudoku solutions for at least `64` held-out puzzles from the frozen
checkpoint snapshot using `sample_checkpoint_candidate_pools()`. It SHALL
compute `vote_at_1` from the TRM majority-vote selector and `oracle_at_k` as
the fraction of puzzles with at least one exact-valid candidate in the sampled
pool.

**When** `oracle_at_k <= vote_at_1`, Exp 4158 SHALL set
`headroom_present=false`, report the honest verdict
`complete: no_headroom_rerank_uninformative_at_val_0.NN`, and SHALL NOT claim a
verifier moat. This is the false-negative-risk guard: a null is informative only
when oracle best-of-K proves recoverable headroom beyond the self-consistency
vote.

**When** `oracle_at_k > vote_at_1`, Exp 4158 SHALL rerank each candidate pool
with the executable Sudoku verifier, report `rerank_lift_vs_vote` as
`pass@1(verifier_rerank) - vote_at_1` with a paired bootstrap CI95, and count
`verifier_recovers_outvoted` as puzzles where an exact-valid candidate was
present, the majority vote missed it, and the executable verifier selected it.
It SHALL also report `cost_ratio_vs_llm_judge` from the measured local verifier
mean per-candidate wall cost in microseconds and an explicit one-line LLM judge
cost estimate.

**Then** Exp 4158 SHALL write
`results/experiment_4158_verifier_rerank_recovery_moat.json` with top-level
fields `honest_verdict`, `headroom_present`, `oracle_at_k`, `vote_at_1`,
`rerank_lift_vs_vote`, `verifier_recovers_outvoted`,
`cost_ratio_vs_llm_judge`, and `random_seed`, each carrying the required
principle text. The acceptance gate passes when `headroom_present` is reported;
if true, the artifact includes rerank CI95, outvoted recovery count, and cost
ratio, and if false, the artifact includes the no-headroom verdict with
`oracle_at_k` and `vote_at_1`.

### SCENARIO-LEARN-4158-NO-HEADROOM: Oracle Does Not Beat Vote

**Given** sampled candidate pools where oracle best-of-K exact accuracy is less
than or equal to majority-vote exact accuracy
**When** Exp 4158 evaluates the rerank moat gate
**Then** it reports `headroom_present=false`, preserves `oracle_at_k` and
`vote_at_1`, writes a terminal no-headroom verdict, and does not claim a
verifier moat.

### SCENARIO-LEARN-4158-RERANK-RECOVERY: Verifier Recovers Present-But-Out-Voted Answers

**Given** sampled candidate pools where at least one exact-valid candidate is
present but out-voted by duplicate invalid candidates
**When** Exp 4158 reranks with the executable Sudoku verifier
**Then** it reports `headroom_present=true`, positive
`rerank_lift_vs_vote.delta` with CI95, and increments
`verifier_recovers_outvoted` for the recovered puzzles.

## Implementation Status (REQ-LEARN-4158)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4158 | Planned (python/carnot/experiment_4158_verifier_rerank_recovery_moat.py, Exp 4158) | Planned (tests/python/test_experiment_4158_verifier_rerank_recovery_moat.py) |

---

## REQ-LEARN-4159: Decisive Verifier-As-Reward Graft With Phase-0 Precision

**Given** Exp 4157 has written
`results/experiment_4157_baseline_harvest_contiguous_continue.json`, Exp 4159
SHALL read `current_val`, `stable_checkpoint_path`, and the recorded runtime
evidence before attempting any verifier-as-reward training graft. It SHALL
verify CUDA visibility and the stable checkpoint path, and SHALL snapshot the
checkpoint before any faithful-branch model load so a live writer cannot taint
the reward-graft comparison.

**When** Exp 4157 reports `current_val < 0.85`, Exp 4159 SHALL stop before
candidate sampling, phase-0 certification, or RFT training. It SHALL write
`graft_deferred=true`, report `current_val`, and estimate additional `.386`
validation intervals needed to reach `0.85`. This deferral is complete because
training a verifier reward on a non-faithful baseline would repeat the
`.383/.384` uninformative false-negative pattern.

**When** Exp 4157 reports `current_val >= 0.85`, Exp 4159 SHALL build
N-matched verifier-certified and vote-certified label corpora from the same TRM
checkpoint. Before any RFT delta is attributed, it SHALL estimate
`phase0_precision = P(test-gold | demo-perfect)` on the verifier-certified
corpus and require precision at least `0.85`; if the precision gate fails, it
SHALL defer the graft with the measured `phase0_precision` instead of
attributing label-noise effects to a reward-training delta.

**When** the phase-0 precision gate passes, Exp 4159 SHALL run the
de-confounded reward-training contrast: arm A is verifier-certified labels, arm
B is vote-certified labels, both are N-matched, both resume from the same TRM
initial checkpoint, both use bounded contiguous-style training rather than the
bounded-pass chain, and both report held-out exact accuracy. The load-bearing
`rft_vs_ablation_delta` SHALL be `A - B` with CI95, and
`verifier_value_added` SHALL be true only when the delta is positive and the
CI95 lower bound excludes zero.

**Then** Exp 4159 SHALL write
`results/experiment_4159_decisive_verifier_reward_graft.json` with top-level
fields `honest_verdict`, `graft_deferred`, `phase0_precision`,
`rft_vs_ablation_delta`, `verifier_value_added`, and `preconditions_checked`,
each with principle text. The acceptance gate passes when either
`graft_deferred=true` with `current_val < 0.85`, or a faithful real graft
reports phase-0 precision, RFT CI95, and the bare `verifier_value_added` bool.

### SCENARIO-LEARN-4159-DEFER: Non-Faithful Baseline Stops The Reward Graft

**Given** Exp 4157 reports `current_val` below `0.85`
**When** Exp 4159 starts
**Then** it writes a terminal-prefixed honest deferral, sets
`graft_deferred=true`, records the baseline checkpoint and CUDA preconditions,
reports current validation accuracy, estimates `.386` convergence intervals,
and does not snapshot, sample candidates, or launch RFT.

### SCENARIO-LEARN-4159-PHASE0: Demo-Perfect Labels Must Be Test-Gold

**Given** Exp 4157 reports a faithful baseline and candidate pools produce a
verifier-certified corpus
**When** Exp 4159 estimates `P(test-gold | demo-perfect)`
**Then** it reports `phase0_precision` with numerator, denominator, threshold,
and status; precision below `0.85` defers the graft before RFT attribution.

### SCENARIO-LEARN-4159-RFT: Verifier Reward Beats Vote Reward Only With CI Separation

**Given** Exp 4157 reports a faithful baseline and phase-0 precision is at
least `0.85`
**When** Exp 4159 runs the A-vs-B reward-training contrast
**Then** both arms use the same TRM initialization and N-matched corpora, the
artifact reports `rft_vs_ablation_delta` with CI95, and
`verifier_value_added` is true only when the positive CI95 excludes zero.

## Implementation Status (REQ-LEARN-4159)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4159 | Planned (python/carnot/experiment_4159_decisive_verifier_reward_graft.py, Exp 4159) | Planned (tests/python/test_experiment_4159_decisive_verifier_reward_graft.py) |

---

## REQ-LEARN-4168: Defensive Decisive Verifier Graft Uses Only A Faithful Stable Copy

**Given** Exp 4167 reports the outer-loop-owned Sudoku Extreme baseline status,
Exp 4168 SHALL read or recompute the Exp 4167 monitor before attempting any
verifier-graft work. The baseline is eligible only when
`baseline_faithful=true`, `current_val_exact_accuracy >= 0.85`, and
`outerloop_train_alive=false`.

**When** the baseline is not faithful and stable, Exp 4168 SHALL write
`results/experiment_4168_decisive_verifier_graft_defensive.json` with
`graft_deferred=true`, an honest terminal-prefixed verdict containing
`graft_deferred_outerloop_training` and the current validation accuracy, and
deferred metric objects for `rerank_lift_vs_vote` and
`rft_vs_ablation_delta`. This branch SHALL NOT train, SHALL NOT terminate any
outer-loop process, SHALL NOT copy over or write under the stable checkpoint
directory, and SHALL NOT sample held-out candidates.

**When** the baseline is faithful and stable, Exp 4168 SHALL copy the stable
checkpoint to a task-local checkpoint path before any model load or training.
All candidate sampling, verifier reranking, and reward fine-tuning SHALL use
that copy, never `results/trm_runs/sudoku_extreme_baseline/last.ckpt`.

**When** the copied-checkpoint branch runs, Exp 4168 SHALL sample `K` candidate
solutions per held-out Sudoku puzzle, compare TRM majority vote, executable
Sudoku verifier rerank, and oracle best-of-K, and report
`rerank_lift_vs_vote` as verifier pass@1 minus vote pass@1 with CI95 plus the
verifier-vs-oracle gap.

**When** the reward fine-tuning arm runs, Exp 4168 SHALL build N-matched
verifier-certified and vote-certified corpora from the copied-checkpoint
candidates, train arm A from verifier-certified labels and arm B from
vote-certified labels from the same copied initial checkpoint with cumulative
time plus budget, and report `rft_vs_ablation_delta` as held-out exact accuracy
`A - B` with CI95. `verifier_value_added` SHALL be a bare bool true only when
the A-vs-B delta is positive and the CI95 lower bound excludes zero.

**Then** the Exp 4168 artifact SHALL include top-level fields
`honest_verdict`, `graft_deferred`, `rerank_lift_vs_vote`,
`rft_vs_ablation_delta`, and `verifier_value_added`, each carrying these exact
principles:

- `honest_verdict`: "Terminal-prefixed. An honest deferral, an A>B win, or an A~=B null are all COMPLETE."
- `graft_deferred`: "Bare bool: True if the baseline was not faithful+stable -> deferred. Prevents an uninformative graft + a collision with the outer-loop run."
- `rerank_lift_vs_vote`: "pass@1 lift from verifier-reranking (if grafted); the executable-verifier discrimination signal."
- `rft_vs_ablation_delta`: "The de-confounded A-vs-B held-out delta with CI -- THE moat measurement."
- `verifier_value_added`: "Bare bool: did the graft beat the vote ablation? Resolves the moat question + the DiffusionGemma gate."

The acceptance gate passes when either `graft_deferred=true` because the
baseline was not faithful and stable, or a copied-checkpoint graft reports
rerank lift, RFT A-vs-B CI95, and the bare `verifier_value_added` bool.

### SCENARIO-LEARN-4168-DEFER: Baseline Not Faithful Stable Stops Before Copy Or Training

**Given** Exp 4167 reports `baseline_faithful=false`, validation below `0.85`,
or a live outer-loop training PID
**When** Exp 4168 starts
**Then** it writes a terminal-prefixed `graft_deferred_outerloop_training`
artifact with the current validation accuracy, sets `graft_deferred=true`,
does not copy the stable checkpoint, does not sample candidates, does not train,
and does not terminate any outer-loop process.

### SCENARIO-LEARN-4168-COPY-GRAFT: Faithful Stable Baseline Uses Only The Task-Local Copy

**Given** Exp 4167 reports `baseline_faithful=true`,
`current_val_exact_accuracy >= 0.85`, and `outerloop_train_alive=false`
**When** Exp 4168 runs the rerank and reward fine-tuning arms
**Then** it copies the stable checkpoint to a task-local path, samples
candidates from that copy, trains both RFT arms from that copy, reports
`rerank_lift_vs_vote` with CI95 and verifier-vs-oracle gap, reports
`rft_vs_ablation_delta` with CI95, and sets `verifier_value_added` from the
positive CI95-separated A-vs-B delta.

## Implementation Status (REQ-LEARN-4168)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4168 | Planned (python/carnot/experiment_4168_decisive_verifier_graft_defensive.py, Exp 4168) | Planned (tests/python/test_experiment_4168_decisive_verifier_graft_defensive.py) |

---

## REQ-LEARN-4168-V2: Gate-082 Defensive Verifier Graft Recomputes Preconditions From Stable Files

**Given** the operator lowered the faithful-baseline gate from `0.85` to
`0.82` on 2026-06-13, Exp 4168 v2 SHALL ignore stale Exp 4167 monitor
artifacts and recompute preconditions from the stable training files before
any copy, candidate sampling, verifier rerank, or reward-graft work. The
baseline is eligible only when
`results/trm_runs/sudoku_extreme_baseline/last.ckpt.bestval` parses to a
finite value greater than or equal to `0.82`, the PID in
`results/trm_runs/contiguous_run.pid` is not alive according to `ps -p`, and
`nvidia-smi --query-compute-apps=pid` has no associated `train.py` process.

**When** any gate-082 precondition fails, Exp 4168 v2 SHALL write
`results/experiment_4168_decisive_verifier_graft_v2_gate082.json` with
`graft_deferred=true`, an honest terminal-prefixed verdict containing
`graft_deferred_outerloop_training` and the current `bestval`, deferred
metric objects for `rerank_lift_vs_vote` and `rft_vs_ablation_delta`, and
read-only action flags proving it did not train, did not terminate any
process, did not copy the stable checkpoint, and did not sample candidates.

**When** all gate-082 preconditions pass, Exp 4168 v2 SHALL copy
`results/trm_runs/sudoku_extreme_baseline/last.ckpt` into a task-local path
under `results/trm_runs/experiment_4168_decisive_verifier_graft_v2_gate082/`
before any model load or candidate sampling. All rerank and A-vs-B
deconfounding work SHALL use that copy and SHALL NOT write to the stable
checkpoint directory.

**When** the copied-checkpoint branch runs, Exp 4168 v2 SHALL sample `K`
candidate solutions per held-out Sudoku puzzle, compare TRM vote, executable
Sudoku verifier rerank, and oracle best-of-K, and report
`rerank_lift_vs_vote` as verifier pass@1 minus vote pass@1 with CI95 plus the
verifier-vs-oracle gap. The A-vs-B arm SHALL use N-matched verifier-certified
and vote-certified corpora from the copied-checkpoint candidates. A supplied
native RFT runner SHALL train both arms from the copied checkpoint; the
bounded default path SHALL report the matched-label deconfound with
`rft_native_training_launched=false` rather than claiming native training ran.

**Then** the Exp 4168 v2 artifact SHALL include top-level fields
`honest_verdict`, `graft_deferred`, `rerank_lift_vs_vote`,
`rft_vs_ablation_delta`, and `verifier_value_added`, each carrying these exact
principles:

- `honest_verdict`: "Terminal-prefixed. An honest deferral, an A>B win, or an A~=B null are all COMPLETE."
- `graft_deferred`: "Bare bool: True if the baseline was not faithful+stable -> deferred. Prevents an uninformative graft + a collision with the outer-loop run."
- `rerank_lift_vs_vote`: "pass@1 lift from verifier-reranking (if grafted); the executable-verifier discrimination signal."
- `rft_vs_ablation_delta`: "The de-confounded A-vs-B held-out delta with CI -- THE moat measurement."
- `verifier_value_added`: "Bare bool: did the graft beat the vote ablation? Resolves the moat question + the DiffusionGemma gate."

The acceptance gate passes when either `graft_deferred=true` because the
fresh gate-082 baseline was not faithful and stable, or a copied-checkpoint
graft reports rerank lift, A-vs-B CI95, and the bare `verifier_value_added`
bool.

### SCENARIO-LEARN-4168-V2-DEFER: Fresh Gate-082 Failure Stops Before Copy Or Training

**Given** `last.ckpt.bestval` is missing, unparsable, or below `0.82`, the PID
file still names a live process, or a GPU compute PID resolves to a `train.py`
process
**When** Exp 4168 v2 starts
**Then** it writes the gate-082 artifact with `graft_deferred=true`, the
current bestval evidence, deferred metric objects, no checkpoint copy, no
candidate sampling, no training, and no process termination attempt.

### SCENARIO-LEARN-4168-V2-COPY-GRAFT: Fresh Gate-082 Pass Uses Only The Task-Local Copy

**Given** `last.ckpt.bestval >= 0.82`, the recorded contiguous-run PID is not
alive, and no GPU compute PID resolves to `train.py`
**When** Exp 4168 v2 runs the rerank and A-vs-B arms
**Then** it copies the stable checkpoint to the task-local v2 path, samples
candidates from that copy, reports `rerank_lift_vs_vote` with CI95 and
verifier-vs-oracle gap, reports `rft_vs_ablation_delta` with CI95, and sets
`verifier_value_added` from the positive CI95-separated A-vs-B delta.

## Implementation Status (REQ-LEARN-4168-V2)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4168-V2 | Implemented (python/carnot/experiment_4168_decisive_verifier_graft_v2_gate082.py, Exp 4168 v2) | Implemented (tests/python/test_experiment_4168_decisive_verifier_graft_v2_gate082.py) |

---

## REQ-LEARN-4139: Decisive Sudoku Verifier Graft With Oracle Separation

**Given** Exp 4138 has written
`results/experiment_4138_sudoku_accumulate_pass4_convergence_check.json`, Exp
4139 SHALL defensively read `matches_published_087`, `near_faithful_080`,
`val_exact_accuracy`, `stable_checkpoint_path`, and the `.384` convergence
estimate before attempting any verifier graft. It SHALL verify CUDA visibility
and that the stable checkpoint exists and is loadable. If the checkpoint is
missing or unloadable, it SHALL write an artifact with
`honest_verdict="blocked_baseline_checkpoint_missing"` and stop without
sampling candidates or inventing verifier measurements.

**When** runtime preconditions pass, Exp 4139 SHALL sample K held-out candidate
Sudoku solutions from the stable TRM checkpoint and first compute the positive
control `headroom_present` as oracle best-of-K exact accuracy greater than TRM
vote pass@1. If the oracle best-of-K does not exceed vote pass@1, Exp 4139
SHALL report `headroom_present=false`, `false_negative_risk=true`, and SHALL
not conclude that the transferable verifier failed.

**When** rerank metrics are reported, Exp 4139 SHALL always keep two measures
separate. `executable_oracle_upper_bound` SHALL be the executable Sudoku
validity rerank lift versus vote, explicitly labeled with
`executable_verifier_is_oracle=true` because unique-solution Sudoku makes this
lift equal to `oracle_vs_vote_gap` by construction. It SHALL NOT contribute to
`verifier_value_added`. `ensemble_rerank_lift_vs_vote` SHALL be the headline
non-oracle Carnot energy/text-stat ensemble rerank lift versus vote, with a
paired bootstrap CI and no exact-validity check in the ensemble score.

**When** Exp 4138 reports `matches_published_087=true` or
`near_faithful_080=true`, Exp 4139 SHALL run the de-confounded RFT label arm:
A is verifier-certified execution-valid labels, B is vote-certified labels,
both N-matched from the same TRM candidate pool and evaluated as held-out exact
accuracy with a CI95 A-B delta. When Exp 4138 reports validation accuracy below
`0.80`, Exp 4139 SHALL set `graft_deferred=true`, report the current
validation accuracy and `.384` estimate, and still preserve the headroom
control plus ensemble rerank measurement when candidate sampling was possible.

**Then** the terminal artifact SHALL be written to
`results/experiment_4139_decisive_verifier_graft_sudoku.json` and SHALL
contain top-level fields `honest_verdict`, `headroom_present`,
`oracle_vs_vote_gap`, `executable_verifier_is_oracle`,
`executable_oracle_upper_bound`, `ensemble_rerank_lift_vs_vote`,
`rft_vs_ablation_delta`, `graft_deferred`, `verifier_value_added`, and
`preconditions_checked`, each with principle text. `verifier_value_added` SHALL
be true only when headroom is present and either the non-oracle ensemble rerank
or the RFT A-vs-B label contrast beats vote with CI support; it SHALL never be
computed from the executable-oracle rerank.

### SCENARIO-LEARN-4139-NO-HEADROOM: Positive Control Is Uninformative

**Given** the sampled candidate pools have oracle best-of-K exact accuracy less
than or equal to TRM vote pass@1
**When** Exp 4139 evaluates the rerank arms
**Then** it reports `headroom_present=false`, `false_negative_risk=true`,
keeps `executable_verifier_is_oracle=true`, preserves the oracle upper bound
and ensemble metrics for context, and sets `verifier_value_added=false`
without claiming that the transferable verifier failed.

### SCENARIO-LEARN-4139-RERANK: Transferable Ensemble Is Headline Rerank

**Given** candidate pools with positive oracle headroom
**When** Exp 4139 reranks candidates
**Then** `executable_oracle_upper_bound.delta` equals `oracle_vs_vote_gap`,
`ensemble_rerank_lift_vs_vote` reports the non-oracle weighted
energy/text-stat ensemble lift with CI95, and only the ensemble result can
contribute to the rerank side of `verifier_value_added`.

### SCENARIO-LEARN-4139-RFT-OR-DEFER: Label Contrast Runs Only On Near-Faithful Baselines

**Given** Exp 4138 reports a faithful or near-faithful baseline
**When** Exp 4139 has candidate pools with verifier-certified labels
**Then** it reports `rft_vs_ablation_delta` with CI95 from the N-matched A-vs-B
label contrast.

**Given** Exp 4138 reports validation accuracy below `0.80`
**When** Exp 4139 completes the rerank measurements
**Then** it reports `graft_deferred=true`, current validation accuracy, and the
`.384` estimate instead of dressing the missing RFT arm as a result.

## Implementation Status (REQ-LEARN-4139)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4139 | Planned (python/carnot/experiment_4139_decisive_verifier_graft_sudoku.py, Exp 4139) | Planned (tests/python/test_experiment_4139_decisive_verifier_graft_sudoku.py) |

---

## REQ-LEARN-4128: Conditional Sudoku Verifier Graft After Fixed-LR Baseline

**Given** Exp 4127 has written
`results/experiment_4127_sudoku_extreme_accumulate_fixed.json`, Exp 4128 SHALL
read that artifact before attempting any verifier-as-reward graft. It SHALL
extract `matches_published_087`, the current validation exact accuracy from
the final trajectory entry or artifact summary, and `stable_checkpoint_path`.
It SHALL also record CUDA availability and stable-checkpoint evidence in
`preconditions_checked`.

**When** the Exp 4127 baseline is not faithful enough
(`matches_published_087` is not true and validation exact accuracy is below
`0.85`), Exp 4128 SHALL set `graft_deferred=true`, report the current baseline
status and estimated additional fixed-LR passes to reach `0.85`, and SHALL NOT
run reranking or RFT training arms. This deferral is complete and acceptance
grade because a graft on an unfaithful TRM baseline would not answer the
verifier-value question.

**When** the Exp 4127 baseline is faithful (`matches_published_087=true` or
validation exact accuracy `>= 0.85`), Exp 4128 SHALL run the graft branch:
sample K TRM candidate solutions per held-out Sudoku puzzle, rerank by the
executable Sudoku verifier, compare pass@1 lift against TRM-vote and oracle
with bootstrap CI, and run a de-confounded verifier-label RFT arm A versus
vote-label ablation B with N-matched corpora, the same TRM, bounded fixed
schedule, held-out exact accuracy, and CI95 on A-B.

**Then** the terminal artifact SHALL be written to
`results/experiment_4128_carnot_verifier_graft_sudoku.json` and SHALL contain
top-level fields `honest_verdict`, `graft_deferred`,
`rerank_lift_vs_vote`, `rft_vs_ablation_delta`,
`verifier_value_added`, and `preconditions_checked` with principle text. The
acceptance gate passes when either `graft_deferred=true` with baseline status
reported, or `graft_deferred=false` with both rerank lift and RFT-vs-ablation
delta reported with CI.

### SCENARIO-LEARN-4128-DEFER: Fixed-LR Baseline Still Not Faithful

**Given** Exp 4127 reports `matches_published_087=false` and validation exact
accuracy below `0.85`
**When** Exp 4128 runs
**Then** it writes an honest `graft_deferred=true` artifact with current
baseline validation accuracy, stable checkpoint path, CUDA/checkpoint
precondition evidence, and estimated additional passes rather than running
rerank or RFT arms.

### SCENARIO-LEARN-4128-GRAFT: Faithful Baseline Reports Rerank And RFT CIs

**Given** Exp 4127 reports a faithful Sudoku Extreme baseline
**When** Exp 4128 runs the graft branch
**Then** it writes `graft_deferred=false`, reports `rerank_lift_vs_vote` and
`rft_vs_ablation_delta` with CI95, and sets the bare
`verifier_value_added` bool only from the A-vs-B CI separation.

## Implementation Status (REQ-LEARN-4128)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4128 | Proposed (python/carnot/experiment_4128_carnot_verifier_graft_sudoku.py, Exp 4128) | Proposed (tests/python/test_experiment_4128_carnot_verifier_graft_sudoku.py) |

---

## REQ-LEARN-4119: Conditional Sudoku Verifier Graft After Faithful TRM Baseline

**Given** Exp 4118 has written `results/experiment_4118_*.json`, Exp 4119
SHALL read the latest Exp 4118 artifact and extract `matches_published_087`,
`val_exact_accuracy`, `stable_checkpoint_path`, and cumulative training status
before deciding whether a verifier graft is meaningful. Exp 4119 SHALL verify
CUDA visibility and the stable checkpoint path before writing a verdict.

**When** the Exp 4118 baseline is not faithful enough
(`matches_published_087` is not true and `val_exact_accuracy < 0.85`), Exp
4119 SHALL defer the graft instead of manufacturing a verifier-as-reward
measurement. The deferral artifact SHALL set `graft_deferred=true`, report the
current baseline validation exact accuracy, preserve the stable checkpoint
path, and include an estimated additional pass count to reach the faithful
baseline threshold when a positive pass-to-pass improvement can be inferred.

**When** the Exp 4118 baseline is faithful
(`matches_published_087=true` or `val_exact_accuracy >= 0.85`), Exp 4119 SHALL
run the graft branch: sample K TRM candidate solutions per held-out Sudoku
puzzle, rerank by the executable Sudoku verifier, compare pass@1 lift against
TRM-vote and oracle with bootstrap CI, and run the de-confounded verifier-label
RFT arm A versus vote-label ablation B with N-matched corpora, the same TRM,
bounded native resume training, held-out exact accuracy, and CI95 on A-B.

**Then** the terminal artifact SHALL be written to
`results/experiment_4119_carnot_verifier_graft_sudoku.json` and SHALL contain
top-level fields `honest_verdict`, `graft_deferred`,
`rerank_lift_vs_vote`, `rft_vs_ablation_delta`,
`verifier_value_added`, and `preconditions_checked` with principle text. The
acceptance gate passes when either `graft_deferred=true` with baseline status
reported, or `graft_deferred=false` with both rerank lift and RFT-vs-ablation
delta reported with CI.

### SCENARIO-LEARN-4119-DEFER: Non-Faithful Baseline Defers The Graft

**Given** Exp 4118 reports `matches_published_087=false` and validation exact
accuracy below 0.85
**When** Exp 4119 runs
**Then** it writes an honest `graft_deferred=true` verdict with the current
baseline validation accuracy, stable checkpoint path, precondition evidence,
and estimated additional passes rather than running rerank or RFT arms.

### SCENARIO-LEARN-4119-GRAFT: Faithful Baseline Reports Rerank And RFT CIs

**Given** Exp 4118 reports a faithful Sudoku Extreme baseline
**When** Exp 4119 runs the graft branch
**Then** it writes `graft_deferred=false`, reports `rerank_lift_vs_vote` and
`rft_vs_ablation_delta` with CI95, and sets the bare
`verifier_value_added` bool only from the A-vs-B CI separation.

## Implementation Status (REQ-LEARN-4119)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4119 | Proposed (python/carnot/experiment_4119_carnot_verifier_graft_sudoku.py, Exp 4119) | Proposed (tests/python/test_experiment_4119_carnot_verifier_graft_sudoku.py) |

---

## REQ-LEARN-4109: Sudoku Verifier Graft Over nano-trm

**Given** Exp 4108 has written a nano-trm Sudoku baseline artifact, Exp 4109
SHALL first load the Exp 4108 checkpoint path. If no Exp 4108 checkpoint is
present, it SHALL fall back to the Exp 4107 smoke checkpoint and record that
limitation honestly instead of hard-blocking. It SHALL also record CUDA
availability, the chosen checkpoint, random seed, and whether the chosen
checkpoint came from the partial Exp 4108 baseline or the Exp 4107 smoke
fallback.

**When** Exp 4109 scores Sudoku candidates, it SHALL use an executable Sudoku
verifier over decoded nano-trm grids: every candidate gets a constraint
satisfaction count, a normalized verifier score, and an exact-validity boolean
based on range, row, column, box, and clue constraints. The rerank arm SHALL
compare verifier-reranked pass@1 against TRM-vote pass@1 on paired held-out
puzzles, report the oracle ceiling from any exact candidate in each pool, and
include a paired bootstrap confidence interval for the verifier rerank lift.

**When** Exp 4109 builds the de-confounded self-training comparison, it SHALL
form A from verifier-certified candidate labels and B from TRM-vote-majority
candidate labels from the same TRM candidate pools, keep A and B N-matched, and
report the held-out exact-accuracy delta A minus B with a paired bootstrap
CI95. When native nano-trm full fine-tuning is not launched, the artifact SHALL
state that limitation explicitly while still preserving the verifier-label
versus vote-label measurement. `verifier_value_added` SHALL be a bare bool that
is true only when A beats B and the lower CI bound is greater than zero.

**Then** the terminal artifact SHALL be written to
`results/experiment_4109_carnot_verifier_graft_sudoku.json` and SHALL contain
the required top-level fields `honest_verdict`, `rerank_lift_vs_vote`,
`rft_vs_ablation_delta`, `verifier_value_added`, `preconditions_checked`,
`random_seed`, and `reproducibility_checksum`. Each required field SHALL carry
the Exp 4109 field-principle text, and the checksum SHALL hash the corpus and
held-out split so silent drift is visible.

### SCENARIO-LEARN-4109-RERANK: Executable Verifier Discriminates Sudoku Candidates

**Given** paired TRM candidate pools contain at least one invalid TRM-vote
winner and one verifier-valid alternative
**When** Exp 4109 reranks by executable Sudoku constraint satisfaction
**Then** `rerank_lift_vs_vote.delta` is positive, its CI is reported, and
`oracle_ceiling_pass_at_1` records the best possible pass@1 from the same
candidate pools.

### SCENARIO-LEARN-4109-RFT: Verifier Label Beats Vote Label Only With CI Separation

**Given** N-matched A and B corpora are built from the same TRM candidates
**When** A's held-out exact accuracy beats B's held-out exact accuracy and the
paired bootstrap CI95 excludes zero
**Then** Exp 4109 reports `verifier_value_added=true` and records the A-vs-B
delta under `rft_vs_ablation_delta`; otherwise it reports a complete honest
null with `verifier_value_added=false`.

## Implementation Status (REQ-LEARN-4109)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4109 | Proposed (python/carnot/experiment_4109_carnot_verifier_graft_sudoku.py, Exp 4109) | Proposed (tests/python/test_experiment_4109_carnot_verifier_graft_sudoku.py) |

---

## REQ-LEARN-4088: Trustworthy ARC Verifier-Reward RFT Corpus Build

**Given** Exp 4087 has already produced
`precision_rescue_succeeded=true` with an operating point whose certification
precision is at least `0.85`, the local `Qwen/Qwen3.5-0.8B` HF safetensors
base loads as trainable Transformers weights, `trl`/`peft` trainers import,
CUDA is visible, and ARC task pools are readable
**When** Exp 4088 prepares the verifier-as-self-improvement RFT corpus
**Then** it SHALL fail closed with `honest_verdict="blocked_<resource>"`
before live generation whenever any mandatory resource is missing, and it SHALL
record every checked resource in `preconditions_checked`.

When preconditions pass, Exp 4088 SHALL reserve a disjoint held-out ARC eval
split, run live Codex generation for at least `k=8` transform programs per
held-in task, and certify each generated program with the Exp 4087 selected
operating point instead of the raw demo-perfect rule. The corpus build SHALL
produce three N-matched training arms from the same generator distribution:
`RFT-CORRECT` for certified-correct programs, `RFT-ABLATION` for
certified-not-correct or demo-failing programs matched to the RFT-correct task
distribution, and `gold-SFT` for test-gold programs matched to the
RFT-correct size where possible. The held-out eval task ids SHALL be recorded
and SHALL NOT appear in any training corpus.

The terminal artifact SHALL write
`results/experiment_4088_verifier_reward_rft_corpus_build.json` with bare
top-level fields `honest_verdict`, `operating_point_used`,
`n_rft_correct`, `n_rft_ablation`, `n_gold_sft`, `n_heldout_tasks`,
`runner_ready`, `trainer_smoke_passed`, `preconditions_checked`,
`model_specs`, `random_seed`, `reproducibility_checksum`, and
`inference_substrate`. Complete artifacts SHALL use
`inference_substrate="live_llm_inference"` and SHALL include methodology
metadata proving live Codex generation and two-task LoRA smoke checkpoint
creation rather than cached replay or fabricated timing.

### REQ-LEARN-4088 Sub-requirements

- REQ-LEARN-4088-1: The precondition checker SHALL verify the actual
  trainable Qwen HF model load, `trl`/`peft` trainer imports, CUDA visibility,
  and the Exp 4087 precision-rescue operating point before live generation.
- REQ-LEARN-4088-2: The runner SHALL invoke Codex live for at least `k=8`
  transform programs per held-in ARC task, print progress per task, and record
  per-task generation latency and call counts in methodology fields.
- REQ-LEARN-4088-3: Certification SHALL inherit Exp 4087's selected operating
  point. For `k_of_n_agreement` with `threshold="k=1"`, a program is
  certified-correct only when it is demo-perfect and belongs to a prediction
  agreement bucket whose size is at least one.
- REQ-LEARN-4088-4: The three corpora SHALL be N-matched across arms and
  matched by held-in task distribution, so a downstream delta is attributable
  to the label rather than corpus size or task composition.
- REQ-LEARN-4088-5: The LoRA runner SHALL expose the same three arms with
  identical training configuration per arm, smoke-train two tasks on
  `Qwen/Qwen3.5-0.8B`, and assert checkpoint directories are written to disk.
- REQ-LEARN-4088-6: The held-out eval harness SHALL write a manifest containing
  only disjoint held-out task ids and enough model/checkpoint metadata for Exp
  4089 to evaluate the three arms without rebuilding the corpus.

### SCENARIO-LEARN-4088-BLOCKED: Missing Resource Prevents Fabrication

**Given** any mandatory precondition is unavailable
**When** Exp 4088 runs
**Then** the artifact reports `honest_verdict="blocked_<resource>"`,
`runner_ready=false`, zero corpus counts, `trainer_smoke_passed=false`, and a
`preconditions_checked` entry for the missing resource.

### SCENARIO-LEARN-4088-NMATCH: Rescue Operating Point Builds Three Matched Arms

**Given** Exp 4087's operating point passes the precision rescue gate and the
held-in live Codex pool contains certified-correct, certified-not-correct, and
gold programs for the same task distribution
**When** Exp 4088 builds corpora
**Then** `n_rft_correct == n_rft_ablation == n_gold_sft`, every RFT-correct
item is certified by the Exp 4087 operating point, every ablation item is not
certified-correct, and held-out task ids are absent from all three training
arms.

### SCENARIO-LEARN-4088-SMOKE: Two-Task LoRA Smoke Writes Checkpoints

**Given** non-empty N-matched corpora and available GPU/trainer/model
preconditions
**When** the Exp 4088 LoRA smoke path runs
**Then** each arm uses the same training config, writes a checkpoint directory
under `results/checkpoints/experiment_4088_verifier_reward_rft_corpus_build`,
and reports `trainer_smoke_passed=true` only when the checkpoint assertion
passes.

## Implementation Status (REQ-LEARN-4088)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4088 | Proposed (python/carnot/agentic/arc_exp4088_verifier_reward_rft_corpus_build.py; scripts/experiments/exp4088_verifier_reward_rft_corpus_build.py, Exp 4088) | Proposed (tests/python/test_experiment_4088_verifier_reward_rft_corpus_build.py) |

---

## REQ-LEARN-4263: FR-11 Verifier-As-Reward Out-Of-Band Package Or Retirement

**Given** the verifier-as-reward in-window LoRA axis has already failed seven
times, Exp 4247 reports `cannot_run_in_window`, Exp 4248 never ran its
A-vs-B-vs-C gate, and the intact A/B/C corpora from the stable offline harness
are available
**When** Exp 4263 runs
**Then** it SHALL perform only preparation work: load the intact verifier-
certified (A), same-generator random-label (B), and hidden-gold (C) corpora;
precompute a deterministic reward-weighted JSONL training artifact; write a
single-command out-of-band runner plus validation harness; and write
`results/experiment_4263_verifier_as_reward_out_of_band_or_retire.json`.

The experiment SHALL NOT run training, SHALL NOT touch `results/trm_runs/`,
SHALL NOT use Qwen as the trained base, and SHALL set `verifier_is_oracle=true`
because the reward label is the execution verifier. The terminal artifact SHALL
include bare top-level fields `ready_for_out_of_band`,
`verifier_as_reward_retired`, `out_of_band_runner_path`, `random_seed`,
`reproducibility_checksum`, `model_specs`, and `honest_verdict` with the field
principles required by the milestone prompt.

If the A/B/C corpora are missing or unreadable, Exp 4263 SHALL stop before
runner generation with `honest_verdict="blocked_abc_corpora_missing"`,
`ready_for_out_of_band=false`, and no fake training artifact. If the corpora
load but cannot support a clean A-vs-B reward contrast, Exp 4263 SHALL retire
the in-loop verifier-as-reward axis by setting `verifier_as_reward_retired=true`
and recording that the FoVer +0.0185 memory-ablation evidence from Exp 2837
stands as the self-learning evidence.

### SCENARIO-LEARN-4263-READY: Out-Of-Band Package Prepared Without Training

**Given** all three stable A/B/C corpora are present and non-empty
**When** Exp 4263 prepares the package
**Then** the reward-weighted corpus is written, the runner script is written,
`ready_for_out_of_band=true`, `verifier_as_reward_retired=false`,
`verifier_is_oracle=true`, and `out_of_band_runner_path` names the written
single-command runner whose validation requires LoRA attachment, more than zero
trainable parameters, at least 20 optimizer steps, `loss_final < loss_initial`,
and plausible runtime on a small non-Qwen base loaded through
`AutoModelForCausalLM`.

### SCENARIO-LEARN-4263-BLOCKED: Missing A/B/C Corpora Stop The Prep

**Given** one or more of the stable A/B/C corpus files is missing or empty
**When** Exp 4263 runs
**Then** it writes `honest_verdict="blocked_abc_corpora_missing"`,
`ready_for_out_of_band=false`, `verifier_as_reward_retired=false`, and does not
write a fake training artifact.

## Implementation Status (REQ-LEARN-4263)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4263 | Proposed (python/carnot/experiment_4263_verifier_as_reward_out_of_band_or_retire.py; results/experiment_4263_verifier_as_reward_out_of_band_or_retire.py, Exp 4263) | Proposed (tests/python/test_experiment_4263_verifier_as_reward_out_of_band_or_retire.py) |

---

## REQ-LEARN-4318: ARC Cross-Game Learned Value-Head Transfer

**Given** at least three reproducibly solved ARC-AGI-3 games with replayable solve
traces
**When** Exp 4318 performs leave-one-game-out training of
`arc_value_learner.LearnedVerifier` on the other games' traces
**Then** the held-out game SHALL be solved with `arc_solver_kit.OfflineSolver`
twice: once with a uniform untrained heuristic and once with the transferred
value-head
**And** the artifact SHALL report per-level search states, per-held-out-game
state reduction, an aggregate `cross_game_state_reduction`, a bootstrap
`cross_game_state_reduction_ci95` using at least 2000 resamples, and a bare
`cross_game_transfer_helps` bool that is true only when the reduction and CI
lower bound both exceed 1.0.
The required top-level artifact fields are `honest_verdict`,
`cross_game_transfer_helps`, `cross_game_state_reduction`,
`cross_game_state_reduction_ci95`, `per_held_out_game_reduction`,
`baseline_solves_held_out`, `verifier_is_oracle`, `random_seed`,
`reproducibility_checksum`, and `model_specs`.

### REQ-LEARN-4318 Sub-requirements

- REQ-LEARN-4318-1: Preconditions SHALL verify that `arc_solver_kit`,
  `arc_value_learner`, and the solved-game traces for r11l, ls20, wa30, and lp85
  are loadable before running the experiment.
- REQ-LEARN-4318-2: If fewer than three games have usable traces with at least
  one reproduced level, Exp 4318 SHALL emit `honest_verdict` containing
  `blocked_insufficient_solve_traces` and SHALL NOT fabricate transfer metrics.
- REQ-LEARN-4318-3: For every held-out game, the training split SHALL exclude all
  levels from that game and the artifact SHALL record the train/held-out split in
  `model_specs`.
- REQ-LEARN-4318-4: The uniform baseline SHALL solve every held-out level before
  a null is considered informative, and the bare `baseline_solves_held_out`
  field SHALL report that positive control.
- REQ-LEARN-4318-5: The artifact SHALL set `verifier_is_oracle=false`,
  `random_seed`, `reproducibility_checksum`, and `model_specs`; the value-head is
  a CPU linear heuristic over solver state and SHALL NOT mutate LLM weights.
- REQ-LEARN-4318-6: If `cross_game_transfer_helps=false` after the positive
  control passes, Exp 4318 SHALL log a game-invariant ARC value-representation
  missing-verifier gap.

### SCENARIO-LEARN-4318: Leave-One-Game-Out Reports Decision-Grade Transfer Or Null

**Given** r11l, ls20, wa30, and lp85 solved traces are present
**When** Exp 4318 trains a value-head on all games except the held-out game
**Then** the held-out game result records `states_uniform`,
`states_transferred`, and `state_reduction`
**And** the aggregate artifact reports `cross_game_transfer_helps` as a bare
bool, `cross_game_state_reduction` as a bare float, and
`cross_game_state_reduction_ci95` as a two-float list.

### SCENARIO-LEARN-4318-BLOCKED: Insufficient Solve Traces Stop Honestly

**Given** fewer than three solved games have usable replay traces
**When** Exp 4318 runs
**Then** it SHALL write a terminal artifact with
`honest_verdict=blocked_insufficient_solve_traces`
**And** `cross_game_transfer_helps=false` and
`baseline_solves_held_out=false`.

## Implementation Status (Exp 4318)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4318 | Implemented (`python/carnot/experiment_4318_arc_cross_game_learned_verifier_transfer.py`; `results/experiment_4318_arc_cross_game_learned_verifier_transfer.py`; `results/experiment_4318_arc_cross_game_learned_verifier_transfer.json`) | Implemented (`tests/python/test_experiment_4318_arc_cross_game_learned_verifier_transfer.py`) |

---

## REQ-LEARN-4331: ARC Cross-Game Learned Frame Encoder Transfer

**Given** at least three reproducibly solved ARC-AGI-3 games with replayable
solve traces and the existing `arc_solver_kit`, `arc_value_learner`, and
cross-game trainer imports available
**When** Exp 4331 performs leave-one-game-out training of a CPU-only learned
frame encoder on the other solved games' raw frames
**Then** the held-out game SHALL be solved with `arc_solver_kit.OfflineSolver`
twice: once with a uniform untrained heuristic positive control and once with a
learned-encoder value-head whose linear head consumes the encoder embedding
**And** the artifact SHALL report `learned_encoder_transfer_helps` as a bare
bool that is true only when aggregate held-out `cross_game_state_reduction` is
greater than 1.0 and the bootstrap `cross_game_state_reduction_ci95` lower bound
is greater than 1.0.
The required top-level artifact fields are `honest_verdict`,
`learned_encoder_transfer_helps`, `cross_game_state_reduction`,
`cross_game_state_reduction_ci95`, `per_held_out_game_reduction`,
`baseline_solves_held_out`, `verifier_is_oracle`, `random_seed`,
`reproducibility_checksum`, and `model_specs`.

### REQ-LEARN-4331 Sub-requirements

- REQ-LEARN-4331-1: Preconditions SHALL verify that `arc_solver_kit`,
  `arc_value_learner`, the cross-game trainer, and solved-game trace files are
  importable/loadable before running the learned-encoder experiment.
- REQ-LEARN-4331-2: If fewer than three games have usable replay traces with at
  least one reproduced level, Exp 4331 SHALL emit
  `honest_verdict=blocked_insufficient_solve_traces` and SHALL NOT fabricate
  transfer metrics.
- REQ-LEARN-4331-3: For every held-out game, the training split SHALL exclude
  all frames from the held-out game and SHALL record the train/held-out split in
  `model_specs`.
- REQ-LEARN-4331-4: The learned frame encoder SHALL be a tiny CPU-trained
  learned convolutional feature map over raw ARC frames, with no LLM weight
  mutation, and the value-head SHALL be the existing CPU linear
  `LearnedVerifier` over the encoder embedding.
- REQ-LEARN-4331-5: The uniform baseline SHALL solve every held-out level before
  a null is considered informative, and the bare `baseline_solves_held_out`
  field SHALL report that positive control.
- REQ-LEARN-4331-6: The artifact SHALL set `verifier_is_oracle=false`,
  `random_seed`, `reproducibility_checksum`, `model_specs`, and run
  `scripts/adversarial_verify.py` cleanly.
- REQ-LEARN-4331-7: If `learned_encoder_transfer_helps=false` after the positive
  control passes, Exp 4331 SHALL sharpen the game-invariant ARC value
  representation gap: a small learned frame encoder over the current solved set
  is insufficient.

### REQ-LEARN-4331 Field Principles

- `honest_verdict`: Terminal-prefixed. A learned-encoder transfer win (reduces
  held-out search states where generic features failed -- the gap closes), a
  powered null (the encoder is still insufficient -> sharper gap, retire the
  small-encoder ask), and an honest blocked_insufficient_solve_traces are ALL
  COMPLETE and decision-grade.
- `learned_encoder_transfer_helps`: BARE bool: the capstone reads this; true iff
  the LEARNED-encoder value-head trained on OTHER games reduces held-out-game
  search states (reduction > 1.0 AND CI95 lower bound > 1.0) -- cross-game
  self-learning via a learned game-invariant representation.
- `cross_game_state_reduction`: BARE float: held-out states_uniform /
  states_transferred (>1.0 = transfer helps; ~1.0 = still per-game-bound) --
  compare to exp4318's generic-feature 1.0; the north-star EFFICIENCY metric.
- `cross_game_state_reduction_ci95`: Bootstrap CI95 (>=2000 resamples) over
  held-out levels -- a lower bound > 1.0 makes 'transfer helps' decision-grade.
- `per_held_out_game_reduction`: State-reduction reported SEPARATELY per
  held-out game -- guards the one-game-dominates failure mode + shows which
  games transfer.
- `baseline_solves_held_out`: BARE bool: the uniform-heuristic solver actually
  solves the held-out levels (the positive control) -- a no-reduction is only
  informative if the baseline succeeds (FALSE_NEGATIVE_RISK guard).
- `verifier_is_oracle`: BARE bool=false -- a learned encoder + value-head over
  solver state (oracle-distinct heuristic), NOT the executable env oracle; NO
  LLM weight mutation.
- `random_seed`: Determinism precondition for the encoder training + the solver.
- `reproducibility_checksum`: Hash of the solve traces + the encoder + the
  leave-one-game-out split + the state counts; lets a third party re-run.
- `model_specs`: The learned-encoder architecture + the value-head + the
  train/held-out game split + the OfflineSolver state-count instrumentation;
  required methodology.

### SCENARIO-LEARN-4331: Leave-One-Game-Out Learned Encoder Reports Decision-Grade Transfer Or Null

**Given** solved traces for the reproduced ARC games are present
**When** Exp 4331 trains the learned frame encoder and value-head on all games
except the held-out game
**Then** each held-out level records `states_uniform`, `states_transferred`, and
`state_reduction`
**And** the aggregate artifact reports `learned_encoder_transfer_helps` as a
bare bool, `cross_game_state_reduction` as a bare float,
`cross_game_state_reduction_ci95` as a two-float list, and
`verifier_is_oracle=false`.

### SCENARIO-LEARN-4331-BLOCKED: Insufficient Solve Traces Stop Honestly

**Given** fewer than three solved games have usable replay traces
**When** Exp 4331 runs
**Then** it SHALL write a terminal artifact with
`honest_verdict=blocked_insufficient_solve_traces`
**And** `learned_encoder_transfer_helps=false` and
`baseline_solves_held_out=false`.

## Implementation Status (Exp 4331)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4331 | Planned (`python/carnot/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.py`; `results/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.py`; `results/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.json`) | Planned (`tests/python/test_experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.py`) |

---

## REQ-LEARN-4342: ARC Cross-Game Action-Role Interaction Encoder Transfer

**Given** at least three reproducibly solved ARC-AGI-3 games with replayable
solve traces and the existing `arc_solver_kit`, `arc_value_learner`, and
cross-game trace-frontier solver imports available
**When** Exp 4342 performs leave-one-game-out training of a CPU-only
game-agnostic action-role / object-interaction encoder on the other solved
games' traces
**Then** the held-out game SHALL be solved with a uniform positive-control
search and with a transferred value head over action-role interaction features
**And** the artifact SHALL report `learned_encoder_transfer_helps` as a bare
bool that is true only when aggregate held-out
`cross_game_state_reduction > 1.0` and the bootstrap
`cross_game_state_reduction_ci95` lower bound is greater than 1.0.
The required top-level artifact fields are `honest_verdict`,
`learned_encoder_transfer_helps`, `cross_game_state_reduction`,
`cross_game_state_reduction_ci95`, `positive_control_passed`,
`n_held_out_games`, `verifier_is_oracle`, `preconditions_checked`,
`random_seed`, `reproducibility_checksum`, and `model_specs`.

### REQ-LEARN-4342 Sub-requirements

- REQ-LEARN-4342-1: Preconditions SHALL verify that
  `ops/arc_solve_registry.yaml`, the cached solved-game trace files, the
  trace-frontier search substrate, and the TRM stand-down condition are present
  before running the action-role encoder experiment.
- REQ-LEARN-4342-2: If fewer than three games have usable replay traces with at
  least one reproduced level, Exp 4342 SHALL emit
  `honest_verdict=blocked_insufficient_game_traces` and SHALL NOT fabricate
  transfer metrics.
- REQ-LEARN-4342-3: For every held-out game, the training split SHALL exclude
  all feature rows from that game and SHALL record the train/held-out split in
  `model_specs`.
- REQ-LEARN-4342-4: The action-role encoder SHALL expose game-agnostic features
  derived from transition effects rather than per-game pixels: action role,
  movement/select/toggle/commit/fire-style effects, changed-object counts,
  centroid/bounding-box motion, component merge/split collision proxies, and
  terminal-interaction alignment deltas.
- REQ-LEARN-4342-5: The uniform baseline SHALL solve every held-out game before
  a null is considered informative, and the bare `positive_control_passed`
  field SHALL report that FALSE_NEGATIVE_RISK guard.
- REQ-LEARN-4342-6: The artifact SHALL set `verifier_is_oracle=false`,
  `random_seed`, `reproducibility_checksum`, `model_specs`, and SHALL run
  `scripts/adversarial_verify.py` cleanly.
- REQ-LEARN-4342-7: The artifact SHALL report `n_held_out_games` as the number
  of leave-one-game-out folds and SHALL compute `cross_game_state_reduction_ci95`
  across held-out games, not across repeated rows from one game.

### REQ-LEARN-4342 Field Principles

- `honest_verdict`: Terminal-prefixed. A transfer WIN (reduction>1.0, CI
  lower>1.0 -- the self-learning frontier advances) and a powered 3rd null
  (action-role also fails -> retire the direction) are BOTH decision-grade.
- `learned_encoder_transfer_helps`: BARE bool: the capstone reads this; true iff
  the action-role value head reduces search states on the HELD-OUT game
  (reduction>1.0 AND CI95 lower bound>1.0) -- the cross-game transfer the
  raw-frame encoder failed to deliver.
- `cross_game_state_reduction`: BARE float: baseline_states / guided_states on
  the held-out games (>1.0 = the action-role value head helps; compare to
  exp4331's 1.008 null).
- `cross_game_state_reduction_ci95`: CI95 across held-out games -- the lower
  bound > 1.0 is the decision-grade transfer (a 3rd null retires the direction).
- `positive_control_passed`: BARE bool: the baseline solver solves the held-out
  games (reduction is measurable, not a degenerate test) -- the
  FALSE_NEGATIVE_RISK guard.
- `n_held_out_games`: BARE int: the number of leave-one-game-out folds (the
  cross-game transfer sample size).
- `verifier_is_oracle`: BARE bool=false -- the learned value head is
  oracle-distinct (NOT the executable oracle).
- `preconditions_checked`: Records the game-traces availability +
  TRM-stand-down; pre-empts the silent-missing-resource fabrication mode.
- `random_seed`: Determinism precondition for the encoder training +
  leave-one-game-out.
- `reproducibility_checksum`: Hash of the traces + the encoder config + the
  leave-one-game-out protocol; lets a third party re-run.
- `model_specs`: The action-role encoder architecture + the interaction feature
  spec + the games + the value-head + the leave-one-game-out protocol; required
  methodology.

### SCENARIO-LEARN-4342: Leave-One-Game-Out Action-Role Encoder Reports Decision-Grade Transfer Or Null

**Given** solved traces for the reproduced ARC games are present
**When** Exp 4342 trains the action-role interaction encoder and value head on
all games except the held-out game
**Then** each held-out game records baseline and guided state counts
**And** the aggregate artifact reports `learned_encoder_transfer_helps` as a
bare bool, `cross_game_state_reduction` as a bare float,
`cross_game_state_reduction_ci95` as a two-float list,
`positive_control_passed=true`, `n_held_out_games`, and
`verifier_is_oracle=false`.

### SCENARIO-LEARN-4342-BLOCKED: Insufficient Game Traces Stop Honestly

**Given** fewer than three solved games have usable replay traces
**When** Exp 4342 runs
**Then** it SHALL write a terminal artifact with
`honest_verdict=blocked_insufficient_game_traces`
**And** `learned_encoder_transfer_helps=false`,
`positive_control_passed=false`, and `n_held_out_games=0`.

## Implementation Status (Exp 4342)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4342 | Planned (`python/carnot/experiment_4342_self_learning_action_role_cross_game_encoder.py`; `results/experiment_4342_self_learning_action_role_cross_game_encoder.py`; `results/experiment_4342_self_learning_action_role_cross_game_encoder.json`) | Planned (`tests/python/test_experiment_4342_self_learning_action_role_cross_game_encoder.py`) |

---

## REQ-LEARN-4476: ARC Cross-Game Verifier Features V3 LOO Gate

The cross-game verifier trainer SHALL expose `cross_game_features_v3` as a
game-agnostic feature vector that extends v2's frame-only scalar and occupancy
features with object-relational, frame-delta, action-conditioned, and
predicate-distance feature classes. V3 feature extraction SHALL accept optional
`previous_frame`, `action_id`, and `goal_frame` context while remaining callable
with only the current frame for existing live routes.

The Exp 4476 workflow SHALL run the existing discriminative
leave-one-game-out gate with v3 features, compare it to the v2 baseline, and
write `results/experiment_4476_verifier_features_v3_loo_gate.json` with
top-level fields `honest_verdict`, `inference_substrate`,
`offline_reproduced`, `reproduced_levels`, `preconditions_checked`,
`v2_baseline_loo_auroc`, `v3_loo_auroc`, `target_loo_auroc`,
`loo_gate_passed`, `feature_class_loo_auroc`, `feature_class_deltas`,
`value_head_routing_measure`, `tests_pass`, `field_principles`, `spec_refs`,
and `reproducibility_checksum`.

### REQ-LEARN-4476 Sub-requirements

- REQ-LEARN-4476-1: Object-relational features SHALL include within-frame
  pairwise object Manhattan-distance summaries and cross-frame object
  correspondence displacement summaries.
- REQ-LEARN-4476-2: Frame-delta features SHALL measure current-vs-previous
  changed cells, nonzero-count movement, color-histogram movement, object-count
  movement, centroid movement, level movement, and shape changes.
- REQ-LEARN-4476-3: Action-conditioned features SHALL encode the action that
  produced the current state without making the feature vector game-specific.
- REQ-LEARN-4476-4: Predicate-distance features SHALL compare the current frame
  against the next banked level-up or supplied goal frame by grid, color,
  object-count, centroid, and pairwise-distance distance.
- REQ-LEARN-4476-5: The trainer SHALL report feature-class leave-one-game-out
  AUROCs so an honest null can identify which v3 class moved transfer and which
  did not.

### SCENARIO-LEARN-4476-FEATURES: V3 Uses Relational, Delta, Action, And Predicate Context

**Given** a previous frame, a current frame, an action id, and a goal frame
**When** `cross_game_features_v3` featurizes the current frame
**Then** the returned vector is longer than v2, has stable named slices for the
object-relational, frame-delta, action-conditioned, and predicate-distance
classes, and each class changes when its corresponding context changes.

### SCENARIO-LEARN-4476-GATE: V3 LOO Artifact Reports Transfer Or Honest Null

**Given** the banked ARC trajectories are replayable offline
**When** the Exp 4476 discriminative gate trains v3 against held-out games
**Then** the artifact records v2 and v3 leave-one-game-out AUROC, the target
threshold `0.6`, whether the gate passed, the feature-class AUROC deltas, the
existing value-head routing measure, and required field principles with a
terminal-prefix `honest_verdict`.

## Implementation Status (REQ-LEARN-4476)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4476 | Planned (`python/carnot/agentic/arc_value_learner.py`, `scripts/arc_cross_game_verifier_train.py`, `results/experiment_4476_verifier_features_v3_loo_gate.json`) | Planned (`tests/python/test_arc_verifier_variant_augmentation.py`) |

---

## REQ-LEARN-4652: ARC Value-Routing Cost Fix Live Productionization

The live ARC value-routing path SHALL replace pure-Python connected-component
flood fill in `arc_agi3_world_model.objects()` and
`arc_value_learner._component_stats_from_grid()` with a
`scipy.ndimage.label` fast path while preserving a pure-Python fallback for
offline runtimes where scipy is unavailable. The fast path and fallback SHALL
produce identical component targets and component statistics over at least 40
deterministic random grids.

The submitted `E3AgentPolicy` SHALL route the value head with the cheap
`v2 + frame_delta` feature subset, excluding the previously measured
dead-weight `action_conditioned` and `predicate_distance` classes. The
submitted value weight SHALL be raised above `0.0` only in the cost-fixed,
frame-hash-cached live path, and Exp 4652 SHALL compare that routed agent to a
matched `value_weight=0` baseline on the same public-game variants with
first-win rate, multi-level solve rate, per-node feature cost, timeout status,
bootstrap confidence interval, and offline reproduction for newly solved
variants.

### REQ-LEARN-4652 Sub-requirements

- REQ-LEARN-4652-1: Component labeling SHALL prefer `scipy.ndimage.label`
  with 4-connectivity and SHALL fall back to pure Python without changing
  component outputs.
- REQ-LEARN-4652-2: Value-routing features SHALL expose a stable
  `v2 + frame_delta` subset and SHALL ignore action-conditioned and
  predicate-distance contexts.
- REQ-LEARN-4652-3: The submitted `E3AgentPolicy` config SHALL set
  `value_weight > 0`, `verifier_is_oracle=false`, and a feature-subset label
  documenting the cheap cost-fixed route.
- REQ-LEARN-4652-4: `experiment_4652_value_routing_cost_fix_live` SHALL write
  the required principle-annotated artifact fields, including the matched
  `value_weight=0` baseline, per-node feature cost under 1 ms, timeout status,
  bootstrap CI, and residual-cause hypothesis for honest nulls.

### SCENARIO-LEARN-4652-COMPONENTS: Fast Labeling Is Output-Identical

**Given** at least 40 deterministic random ARC-style grids
**When** the world-model object extractor and value-learner component stats run
through both the scipy fast path and the pure-Python fallback
**Then** the outputs are identical after deterministic ordering.

### SCENARIO-LEARN-4652-VALUE-ROUTE: Submitted Path Uses Cheap Frame-Delta Value

**Given** a previous frame, current frame, action id, and goal frame
**When** the live value-routing featurizer builds its feature vector
**Then** the vector contains only the v2 and frame-delta slices, changes with
frame-delta context, and is unchanged by action-conditioned or
predicate-distance context.

### SCENARIO-LEARN-4652-LIVE-ARTIFACT: Matched Controls Decide Lift Or Null

**Given** the offline arcade public variants are runnable
**When** Exp 4652 compares `value_weight>0` against `value_weight=0`
**Then** the artifact reports live first-win and multi-level solve rates,
matched deltas, bootstrap CI, no-timeout status, per-node feature cost,
live-path reachability, parity test status, and an honest residual hypothesis
when the live lift is null.

### REQ-LEARN-4652 Field Principles

- `honest_verdict`: terminal prefix; success: value_routing_cost_fixed_live_<firstwin|solverate>_up_<n> OR complete: value_routing_cost_fixed_no_live_lift_residual_dist_shift_or_calibration.
- `inference_substrate`: verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over cached variants (1s floor); the value head is a small CPU computation, no live_llm_inference.
- `verifier_is_oracle`: MUST be false -- cross_game_features_v3 is a learned discriminator (LOO-AUROC 0.725), oracle-DISTINCT from the executable win-check.
- `solve_provenance`: live_agent_self_discovery -- this improves the SCORED live agent's OWN search guidance (E3AgentPolicy value-routing); NOT a parallel solver, NOT outer_loop_re.
- `live_path_reachable`: HARD gate -- the changed modules are in the live import closure (arc_agi3_world_model/arc_value_learner/arc_competition_agent <- E3AgentPolicy); arc_orphan_solver_lint passes.
- `feature_output_identical_verified`: the scipy.ndimage.label fix MUST produce IDENTICAL component stats to the pure-python flood fill (asserted over >=40 random grids) -- a pure speedup, not a behavior change.
- `per_node_feature_cost_ms`: the productionized per-node cost (must be < 1ms; was 13ms) -- the load-bearing number that removes the 25-game-sim timeout.
- `sim_timed_out`: MUST be false with the cost fix + value_weight>0 -- the timeout was the named root cause of the prior value-head reversion.
- `value_weight_set`: the value_weight raised off the 0.0 floor (the prescribed sequel to the cost fix); records the chosen dense-bias weight that does not time out.
- `live_first_win_rate_value_routed`: the HEADLINE -- LIVE first-win-rate WITH the cost-fixed value head at value_weight>0 on the SCORED agent.
- `live_solve_rate_value_routed`: LIVE multi-level (>=2) solve-rate WITH value-routing (the deeper wall -- does affordable guidance chain a 2nd level-up).
- `live_baseline_value_weight_zero`: the matched value_weight=0 baseline first-win + solve-rate on the SAME variants (the no-regression control).
- `first_win_rate_delta`: value_routed - baseline first-win-rate (positive = affordable guidance crossed the bridge), emitted explicitly so a null (0) is annotated.
- `solve_rate_delta`: value_routed - baseline multi-level solve-rate; emitted explicitly so a null is annotated.
- `live_lift_ci`: bootstrap CI on the chosen live-lift metric; a claim above baseline requires the CI to exclude it.
- `bare_control_passed`: the POSITIVE CONTROL -- the value_weight=0 baseline ran on a corpus with reachable headroom; a no-lift null is valid only then.
- `false_negative_risk_checked`: true with the no-timeout cost control + baseline + reachable-headroom confirmed -- a 'no lift' null is valid only then.
- `residual_cause_hypothesis`: if the affordable value head still nulls, names the residual cause (distribution_shift | calibration) -- the B1/.430 diagnostic target; 'none' if it lifted.
- `null_delta_methodology_note`: present when a delta==0 -- states the equality is an honest no-value null, not a measurement bug.
- `chosen_submitted_config`: the recommended SUBMITTED_AGENT_CONFIG change (cost-fix on, value_weight value, feature subset) -- the A6 input; 'unchanged' if null.
- `parity_test_green`: HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config stays the single source of truth.
- `offline_reproduced`: any newly-solved variant must offline-reproduce to count.
- `random_seed`: determinism precondition for reproducibility.
- `reproducibility_checksum`: content-addressed hash catches silent harness/corpus drift on replay.
- `preconditions_checked`: records resources verified (offline arcade, E3AgentPolicy + world-model + value-learner importable, scipy present); pre-empts missing-resource fabrication.

## Implementation Status (REQ-LEARN-4652)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4652 | Planned (`python/carnot/agentic/arc_agi3_world_model.py`, `python/carnot/agentic/arc_value_learner.py`, `python/carnot/agentic/arc_competition_agent.py`, `python/carnot/experiment_4652_value_routing_cost_fix_live.py`, `results/experiment_4652_value_routing_cost_fix_live.json`) | Planned (`tests/python/test_experiment_4652_value_routing_cost_fix_live.py`, `tests/python/test_arc_verifier_variant_augmentation.py`, `tests/python/test_arc_submitted_agent_parity.py`) |

---

## REQ-LEARN-4658: ARC Value-Routing CI Gate And Residual Diagnostic

The ARC value-routing infrastructure SHALL provide a deterministic CI gate for
the cost-fixed Exp 4652 value-routed 25-game live simulation. The gate SHALL fail
when the cost-fixed value-routed simulation times out, when any live attempt in
the value-routed or matched zero-weight baseline measurement reports a timeout,
when the value-routed first-win rate drops below the Exp 4652 floor, or when the
value-routed multi-level solve-rate drops below the Exp 4652 floor. The floor
SHALL default to the Exp 4652 artifact values so future changes cannot silently
revert the affordable value-weight integration.

If Exp 4652 remains an honest no-lift/null value-routing result, the same
workflow SHALL run an oracle-distinct residual diagnostic that distinguishes
distribution shift from calibration. The distribution-shift probe SHALL compare
value-head scores on winning-path states with off-path search-distribution states
and emit a high-is-shift score. The calibration probe SHALL fit a monotonic
isotonic/Platt-style score-to-cost mapping over cached candidate scores and
report whether that calibrated cost changes the live routing decision. The
workflow SHALL write
`results/experiment_4658_value_routing_cigate_diagnostic.json` and SHALL NOT
perform live LLM inference or leaderboard submission.

### SCENARIO-LEARN-4658-CIGATE: Cost-Fixed Value Routing Cannot Regress Quietly

**Given** the Exp 4652 value-routing live artifact is present
**When** the CI gate evaluates timeout status, matched live attempts, first-win
rate, and multi-level solve-rate
**Then** the gate passes only when no timeout occurred, the 25-game measurement
is present, first-win rate is at least the Exp 4652 floor, and multi-level
solve-rate is at least the Exp 4652 floor.

### SCENARIO-LEARN-4658-DIAGNOSTIC: Residual Cause Is Localized

**Given** the Exp 4652 artifact reports
`residual_cause_hypothesis=distribution_shift_or_calibration`
**When** the diagnostic compares off-path search scores to winning-path scores
and calibrates cached ranking scores into costs
**Then** the artifact reports `distribution_shift_score`,
`calibration_changes_routing`, and `dominant_residual_cause` as one of
`distribution_shift`, `calibration`, or `none_a1_lifted`.

### REQ-LEARN-4658 Field Principles

- `honest_verdict`: terminal prefix; success: value_routing_cigate_plus_diagnostic_shipped_tests_green.
- `inference_substrate`: verifier_ensemble_against_cached_candidates -- offline value-head re-scoring + a small sim re-run (1s floor); no live_llm_inference.
- `verifier_is_oracle`: MUST be false -- the value head/diagnostic are oracle-distinct from the executable win-check.
- `cigate_added`: the CI-gate that fails on a value-routing timeout-regression or live first-win/solve-rate floor breach (guards the A1 win).
- `distribution_shift_score`: the DAgger-lite off-path-vs-winning-path value-head score gap (high = distribution-shift is the residual cause).
- `calibration_changes_routing`: whether isotonic/Platt calibration of the 0.725 ranking changes the live routing decision (true = calibration is a residual cause).
- `dominant_residual_cause`: the localized .430 target (distribution_shift | calibration | none_a1_lifted) -- converts A1's null into a sharp next-milestone attack.
- `tests_added`: the unit tests added for the CI-gate + diagnostic (Tests Must Run and Assert).
- `random_seed`: determinism precondition for reproducibility.
- `reproducibility_checksum`: content-addressed hash catches silent drift on replay.
- `preconditions_checked`: records resources verified; pre-empts missing-resource fabrication.

## Implementation Status (REQ-LEARN-4658)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4658 | Planned (`python/carnot/experiment_4658_value_routing_cigate_diagnostic.py`, `results/experiment_4658_value_routing_cigate_diagnostic.json`) | Planned (`tests/python/test_experiment_4658_value_routing_cigate_diagnostic.py`) |

---

## REQ-LEARN-4665: ARC DAgger-Lite Distribution-Shift Value Routing

The ARC value-routing workflow SHALL aggregate win-reachability value-head
training data from the search distribution induced by the live `E3AgentPolicy`
explorer, not only from replayed winning-path states. The DAgger-lite collector
SHALL roll out the live explorer on offline-arcade variants, log frontier states
from the live import path, relabel those rows with executable reproduction
evidence for training only, aggregate them with the existing winning-path
dataset, and train a cheap `v2 + frame_delta` win-reachability value head whose
inference remains oracle-distinct.

The submitted live value route SHALL prefer the DAgger-corrected cheap-feature
head when its checkpoint is present, while retaining the .429 winning-path
trained head as a matched baseline control. Exp 4665 SHALL compare corrected
live first-win and multi-level solve-rate against the .429 value-routed baseline
on the same cached public-game variants, re-run the distribution-shift probe
before and after aggregation, report bootstrap confidence intervals, and write
`results/experiment_4665_dagger_distribution_shift_value_routing.json`.

### SCENARIO-LEARN-4665-DAGGER-DATA: Live Frontier Rows Enter Training

**Given** the live `StepwiseExplorer` visits frontier states during an offline
arcade rollout
**When** Exp 4665 collects DAgger-lite training rows
**Then** the rows include source, feature vector, node path, and relabeled
win-reachability label, and the aggregate dataset contains both winning-path
positives and search-distribution off-path negatives.

### SCENARIO-LEARN-4665-LIVE-ROUTE: Corrected Head Is Live-Reachable

**Given** a DAgger-corrected cheap-feature checkpoint is present
**When** `load_cross_game_value_head()` builds the submitted E3 value route
**Then** it loads the distribution-corrected head before the .429
winning-path-trained baseline, keeps `verifier_is_oracle=false`, and preserves
the `v2 + frame_delta` feature subset.

### SCENARIO-LEARN-4665-ARTIFACT: Lift Or Null Is Reported Honestly

**Given** the B1 localization artifact and the .429 A1 baseline artifact are
present
**When** Exp 4665 measures the corrected value route against the same variants
**Then** the artifact reports before/after distribution-shift score, first-win
and solve-rate deltas, bootstrap CI, live-path lint, parity test status,
offline reproduction for any newly solved variant, and a residual bridge gap
when no live lift crosses the confidence gate.

### REQ-LEARN-4665 Field Principles

- `honest_verdict`: terminal prefix; success: dagger_distribution_shift_value_routing_live_<firstwin|solverate>_up_<n> OR complete: dagger_distribution_corrected_no_live_lift_residual_logged.
- `inference_substrate`: verifier_ensemble_against_cached_candidates -- offline-arcade DAgger data collection + value-head re-train + live-search measurement over cached variants (1s floor); the value head is CPU, no live_llm_inference.
- `verifier_is_oracle`: MUST be false -- the win-reachability value head is a learned discriminator, oracle-DISTINCT from the executable win-check (the oracle is used only to LABEL training data, not at inference).
- `solve_provenance`: live_agent_self_discovery -- this improves the SCORED live agent's OWN search guidance (E3AgentPolicy value-routing); NOT a parallel solver, NOT outer_loop_re.
- `live_path_reachable`: HARD gate -- the changed modules (arc_value_learner/arc_competition_agent) are in the E3AgentPolicy import closure; arc_orphan_solver_lint passes.
- `distribution_shift_score_before`: B1's measured 0.699 -- the localized residual cause this task attacks.
- `distribution_shift_score_after`: the post-DAgger shift score -- a DROP is the evidence the correction took (the mechanism worked) independent of the live lift.
- `shift_score_delta`: after - before (negative = shift reduced), emitted explicitly so a null is annotated.
- `live_first_win_rate_corrected`: the live first-win-rate WITH the distribution-corrected value head on the SCORED agent.
- `live_solve_rate_corrected`: the live multi-level (>=2) solve-rate WITH the distribution-corrected head (the deeper wall).
- `live_baseline_winning_path_trained`: the matched .429 winning-path-trained value head first-win + solve-rate on the SAME variants (the no-regression control).
- `first_win_rate_delta`: corrected - baseline first-win-rate (positive = distribution correction crossed the bridge), emitted explicitly so a null (0) is annotated.
- `solve_rate_delta`: corrected - baseline multi-level solve-rate; emitted explicitly so a null is annotated.
- `live_lift_ci`: bootstrap CI on the chosen live-lift metric; a claim above baseline requires the CI to exclude it.
- `bare_control_passed`: the POSITIVE CONTROL -- the baseline ran on a corpus with reachable headroom; a no-lift null is valid only then.
- `false_negative_risk_checked`: true with the matched baseline + reachable-headroom confirmed -- a 'no lift' null is valid only then.
- `null_methodology_note`: present when a delta==0 -- states the equality is an honest no-value null, not a measurement bug.
- `chosen_submitted_config`: the recommended SUBMITTED_AGENT_CONFIG change (distribution-corrected head on, value_weight, feature subset) -- the A6 input; 'unchanged' if null.
- `parity_test_green`: HARD gate -- test_arc_submitted_agent_parity.py passes.
- `offline_reproduced`: any newly-solved variant must offline-reproduce to count.
- `residual_bridge_gap`: the Missing-Verifier / bridge gap logged if the corrected head still nulls -- the .431 next-attack record.
- `random_seed`: determinism precondition for reproducibility (DAgger sampling RNG seeded).
- `reproducibility_checksum`: content-addressed hash catches silent harness/corpus drift on replay.
- `preconditions_checked`: records resources verified (offline arcade, value-learner + agent importable, B1 artifact present); pre-empts missing-resource fabrication.

## Implementation Status (REQ-LEARN-4665)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4665 | Planned (`python/carnot/agentic/arc_value_learner.py`, `python/carnot/agentic/arc_competition_agent.py`, `python/carnot/experiment_4665_dagger_distribution_shift_value_routing.py`, `results/experiment_4665_dagger_distribution_shift_value_routing.json`) | Planned (`tests/python/test_experiment_4665_dagger_distribution_shift_value_routing.py`, `tests/python/test_arc_submitted_agent_parity.py`) |

---

## REQ-LEARN-4353: ARC Learned Action-Cost Heuristic Improves Held-Out Actions-To-Solve

**Given** at least five reproduced ARC-AGI-3 solved levels with loadable action
traces from `ops/arc_solve_registry.yaml`, `results/arc_explore_trajectory_*.json`,
and per-game solver artifacts
**When** Exp 4353 trains a CPU-only learned A* action-cost heuristic on
solve-trace states and leaves some reproduced levels out
**Then** the held-out levels SHALL be planned with both the baseline planner and
the learned action-cost heuristic
**And** every counted plan SHALL pass `arc_solver_kit.reproduce`
**And** the artifact SHALL report the north-star action-efficiency fields as
bare values: `action_efficiency_improves`, `held_out_actions_baseline`,
`held_out_actions_learned`, `positive_control_passed`, `reproduction_gated`,
`verifier_is_oracle=false`, `preconditions_checked`, `random_seed`, and
`reproducibility_checksum`.

### REQ-LEARN-4353 Sub-requirements

- REQ-LEARN-4353-1: Preconditions SHALL verify the solved-level registry, local
  trace files, offline reproduction substrate, and TRM stand-down condition. If
  fewer than five reproduced levels with action traces are loadable, Exp 4353
  SHALL emit `honest_verdict=blocked_insufficient_solve_traces` and stop without
  fabricating action-efficiency metrics.
- REQ-LEARN-4353-2: The training corpus SHALL contain `(state_features,
  minimal_actions_to_win)` rows extracted from reproduced solve traces, where
  each row's target is the remaining env-action count in that level.
- REQ-LEARN-4353-3: The held-out split SHALL exclude the held-out level rows
  from heuristic training and SHALL record the split plus heuristic config in
  `model_specs`.
- REQ-LEARN-4353-4: The learned heuristic SHALL be a lightweight CPU regression
  over generic grid/action features and SHALL be oracle-distinct from the
  executable environment/reproduction oracle.
- REQ-LEARN-4353-5: `action_efficiency_improves` SHALL be true exactly when
  total held-out env-actions-to-solve with the learned heuristic is strictly
  lower than the baseline total and the positive-control headroom check passes.
- REQ-LEARN-4353-6: `reproduction_gated` SHALL be true only when every counted
  baseline and learned plan passes `arc_solver_kit.reproduce`; a non-reproduced
  shorter path SHALL NOT count.
- REQ-LEARN-4353-7: Any held-out level where the learned heuristic cannot reduce
  actions SHALL be represented in `missing_verifier_gaps` and appended to
  `ops/verifier_gaps.md` when the experiment writes the artifact.
- REQ-LEARN-4353-8: The written artifact SHALL run cleanly through
  `scripts/adversarial_verify.py` and SHALL include the verifier-oracle,
  random-seed, methodology, and reproducibility-checksum fields required by the
  adversarial artifact discipline.

### REQ-LEARN-4353 Field Principles

- `honest_verdict`: Terminal-prefixed. An action-efficiency improvement (the
  learned heuristic lowers env-actions-to-solve on held-out levels) and an
  honest null with the positive control passing (the value head already captures
  it) are BOTH decision-grade.
- `action_efficiency_improves`: BARE bool: the capstone reads this; true iff
  held-out actions-to-solve is REDUCED with the learned A* path-cost heuristic
  vs the baseline planner AND the positive control confirms headroom existed
  (not a degenerate no-headroom null).
- `held_out_actions_baseline`: BARE int: total env-actions-to-solve across
  held-out levels with the baseline planner -- the efficiency baseline.
- `held_out_actions_learned`: BARE int: total env-actions-to-solve across
  held-out levels with the learned A* heuristic -- the efficiency result (lower
  is better; the north-star action-efficiency metric).
- `positive_control_passed`: BARE bool: a level with known optimal action-count
  confirms headroom existed, so a null is "the value head already captures it",
  not "no headroom" (FALSE_NEGATIVE_RISK guard).
- `reproduction_gated`: BARE bool: true iff every counted plan still passes
  `arc_solver_kit.reproduce` -- an action-minimal plan that does not reproduce
  does NOT count.
- `verifier_is_oracle`: BARE bool=false -- the learned action-cost heuristic is
  not the executable oracle.
- `preconditions_checked`: Records the solve-trace availability +
  TRM-stand-down; pre-empts the silent-missing-resource fabrication mode.
- `random_seed`: Determinism precondition for the heuristic training + the
  held-out split + the planning.
- `reproducibility_checksum`: Hash of the training corpus + the held-out split +
  the heuristic config; lets a third party re-run.

### SCENARIO-LEARN-4353: Learned Action-Cost Artifact Is Reproduction-Gated

**Given** at least five reproduced ARC solved-level traces are loadable
**When** Exp 4353 trains the action-cost heuristic and evaluates held-out levels
**Then** the artifact SHALL include bare `action_efficiency_improves`,
`held_out_actions_baseline`, `held_out_actions_learned`,
`positive_control_passed`, `reproduction_gated`, `verifier_is_oracle=false`,
`preconditions_checked`, `random_seed`, and `reproducibility_checksum`
**And** `action_efficiency_improves=true` only if
`held_out_actions_learned < held_out_actions_baseline`,
`positive_control_passed=true`, and `reproduction_gated=true`.

### SCENARIO-LEARN-4353-BLOCKED: Insufficient Solved Traces Stop Honestly

**Given** fewer than five reproduced levels with action traces are loadable
**When** Exp 4353 runs
**Then** it SHALL write a terminal artifact with
`honest_verdict=blocked_insufficient_solve_traces`
**And** `action_efficiency_improves=false`,
`held_out_actions_baseline=0`, `held_out_actions_learned=0`,
`positive_control_passed=false`, and `reproduction_gated=false`.

## Implementation Status (Exp 4353)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4353 | Planned (`python/carnot/experiment_4353_learned_action_cost_heuristic_efficiency.py`; `results/experiment_4353_learned_action_cost_heuristic_efficiency.py`; `results/experiment_4353_learned_action_cost_heuristic_efficiency.json`) | Planned (`tests/python/test_experiment_4353_learned_action_cost_heuristic_efficiency.py`) |

---

## REQ-LEARN-4364: ARC Action-Cost Heuristic Deployment and Compounding Curve

**Given** at least eight reproduced ARC-AGI-3 solved levels with loadable action
traces from `ops/arc_solve_registry.yaml` and the local solve-trace artifacts
**When** Exp 4364 deploys the Exp 4353 learned action-cost heuristic mechanism
into `arc_solver_kit` and evaluates fixed held-out levels across increasing
solved-trace corpus prefixes
**Then** `OfflineSolver` SHALL use an additive A* planner cost (`g + h`) by
default for ARC planning while preserving an explicit zero path-cost baseline
for BFS/value-head comparisons
**And** the experiment artifact SHALL report the compounding learning curve as
bare fields: `honest_verdict`, `action_efficiency_compounds`,
`compounding_curve`, `deployed_into_solver_kit`, `positive_control_passed`,
`reproduction_gated`, `llm_heuristic_arm`, `verifier_is_oracle=false`,
`preconditions_checked`, `random_seed`, and `reproducibility_checksum`.

### REQ-LEARN-4364 Sub-requirements

- REQ-LEARN-4364-1: Preconditions SHALL verify the solved-level registry, local
  trace files, offline reproduction substrate, and TRM stand-down condition. If
  fewer than eight reproduced levels with action traces are loadable, Exp 4364
  SHALL emit `honest_verdict=blocked_insufficient_solve_traces` and stop without
  fabricating a compounding curve.
- REQ-LEARN-4364-2: `OfflineSolver` SHALL default to the standing ARC A* path
  cost while accepting `path_cost_weight=0.0` for the legacy BFS/value-head
  baseline.
- REQ-LEARN-4364-3: The compounding curve SHALL use a fixed held-out level set
  and increasing solved-trace corpus prefixes, and every counted held-out plan
  SHALL pass `arc_solver_kit.reproduce`.
- REQ-LEARN-4364-4: `action_efficiency_compounds` SHALL be true exactly when
  held-out actions-to-solve decrease as corpus size grows, the positive-control
  headroom check passes, reproduction gating passes, and the solver-kit
  deployment check passes.
- REQ-LEARN-4364-5: The optional LLM-generated heuristic-program arm SHALL be
  reported as `llm_heuristic_arm={ran:false, beats_linear:false,
  static_analysis_clean:true}` when skipped for offline/zero-quota execution.
- REQ-LEARN-4364-6: The learned action-cost heuristic SHALL remain
  oracle-distinct from the executable environment/reproduction oracle and SHALL
  report `verifier_is_oracle=false`.
- REQ-LEARN-4364-7: Any held-out level where the learned heuristic cannot reduce
  actions after reproduced evaluation SHALL be represented in
  `missing_verifier_gaps` and appended to `ops/verifier_gaps.md` when the
  experiment writes the artifact.
- REQ-LEARN-4364-8: The written artifact SHALL run cleanly through
  `scripts/adversarial_verify.py` and SHALL include the random seed,
  methodology note, and reproducibility checksum over the corpus, held-out
  split, heuristic config, and curve.

### REQ-LEARN-4364 Field Principles

- `honest_verdict`: Terminal-prefixed. A compounding result (held-out
  env-actions fall as the corpus grows) and an honest null with the positive
  control passing (a plateau the value head already captures) are BOTH
  decision-grade.
- `action_efficiency_compounds`: BARE bool: the capstone reads this; true iff
  held-out env-actions-to-solve DECREASES as the solved-trace corpus grows
  (learning that compounds) AND the positive control confirms headroom existed
  (not a degenerate no-headroom null).
- `compounding_curve`: list of `{corpus_size_k, held_out_actions_to_solve}` --
  the learning curve; a monotone-ish decrease is the "gets smarter over time"
  signal (PRD core).
- `deployed_into_solver_kit`: BARE bool: true iff the learned action-cost
  heuristic is wired into `arc_solver_kit` as the standing A* cost (every future
  ARC solve action-minimal by default).
- `positive_control_passed`: BARE bool: a held-out level with known optimal
  action-count confirms headroom existed, so a null is "plateau", not "no
  headroom" (FALSE_NEGATIVE_RISK guard).
- `reproduction_gated`: BARE bool: true iff every counted plan still passes
  `arc_solver_kit.reproduce` -- an action-minimal plan that does not reproduce
  does NOT count.
- `llm_heuristic_arm`: optional `{ran: bool, beats_linear: bool,
  static_analysis_clean: bool}` -- the stronger-function-class probe
  (2503.18809), if quota allowed.
- `verifier_is_oracle`: BARE bool=false -- the learned action-cost heuristic is
  not the executable oracle.
- `preconditions_checked`: Records the solve-trace availability +
  TRM-stand-down; pre-empts the silent-missing-resource fabrication mode.
- `random_seed`: Determinism precondition for the heuristic training + the
  held-out split + the corpus-prefix curve.
- `reproducibility_checksum`: Hash of the training corpus + the held-out split +
  the heuristic config + the curve; lets a third party re-run.

### SCENARIO-LEARN-4364: Deployed Action-Cost Curve Compounds

**Given** at least eight reproduced ARC solved-level traces are loadable
**When** Exp 4364 evaluates the standing A* action-cost heuristic on the fixed
held-out level set across increasing corpus prefixes
**Then** the artifact SHALL include bare `action_efficiency_compounds`,
`compounding_curve`, `deployed_into_solver_kit`, `positive_control_passed`,
`reproduction_gated`, `verifier_is_oracle=false`, `preconditions_checked`,
`random_seed`, and `reproducibility_checksum`
**And** `action_efficiency_compounds=true` only if the curve decreases,
`deployed_into_solver_kit=true`, `positive_control_passed=true`, and
`reproduction_gated=true`.

### SCENARIO-LEARN-4364-BLOCKED: Insufficient Solved Traces Stop Honestly

**Given** fewer than eight reproduced levels with action traces are loadable
**When** Exp 4364 runs
**Then** it SHALL write a terminal artifact with
`honest_verdict=blocked_insufficient_solve_traces`
**And** `action_efficiency_compounds=false`, `compounding_curve=[]`,
`deployed_into_solver_kit=false`, `positive_control_passed=false`, and
`reproduction_gated=false`.

## Implementation Status (Exp 4364)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4364 | Planned (`python/carnot/experiment_4364_self_learning_action_cost_compounds.py`; `results/experiment_4364_self_learning_action_cost_compounds.py`; `results/experiment_4364_self_learning_action_cost_compounds.json`) | Planned (`tests/python/test_experiment_4364_self_learning_action_cost_compounds.py`) |

---

## REQ-LEARN-4370: ARC LLM-Generated Domain Heuristic Programs

**Given** at least eight reproduced ARC-AGI-3 solved levels with loadable action
traces and an importable `arc_solver_kit.reproduce` gate
**When** Exp 4370 generates domain-dependent Python heuristic programs for ARC
games with at least two reproduced levels
**Then** each generated program SHALL pass static leakage analysis before
selection
**And** the selected clean heuristics SHALL be evaluated on a fresh held-out
split against the deployed Exp 4364 linear action-cost heuristic and the
BFS/region-count baseline
**And** every counted plan SHALL be reproduction-gated
**And** the artifact SHALL report bare gate fields: `llm_heuristic_beats_linear`,
`held_out_actions_by_heuristic`, `per_game_scorecard`,
`static_leakage_clean`, `reproduction_gated`, `n_held_out_levels`,
`verifier_is_oracle=false`, `preconditions_checked`, `random_seed`,
`reproducibility_checksum`, and `model_specs`.

### REQ-LEARN-4370 Sub-requirements

- REQ-LEARN-4370-1: Preconditions SHALL verify the solved-level registry, local
  trace files, deployed Exp 4364 linear heuristic artifact, `arc_solver_kit`
  importability, `reproduce` availability, and TRM stand-down. If fewer than
  eight reproduced levels with action traces are loadable, Exp 4370 SHALL emit
  `honest_verdict=blocked_insufficient_solve_traces`. If the solver kit is not
  importable or `reproduce` is unavailable, Exp 4370 SHALL emit
  `honest_verdict=blocked_solver_kit_unavailable`.
- REQ-LEARN-4370-2: For every game with at least two reproduced levels, the
  generator SHALL write at least three candidate Python `h(state) -> cost`
  functions using observable grid features only.
- REQ-LEARN-4370-3: Static leakage analysis SHALL reject generated programs that
  read answer cells, environment internals, solver goal predicates, or
  hard-coded level layouts before any selection score is computed.
- REQ-LEARN-4370-4: Selection SHALL score only clean heuristic programs on the
  training split and keep the strongest per game by reproduced action count,
  then expansions, then deterministic candidate name.
- REQ-LEARN-4370-5: Held-out evaluation SHALL use at least eight fresh held-out
  levels excluded from selection and report action totals for
  `linear`, `llm_generated`, and `bfs_baseline`.
- REQ-LEARN-4370-6: `llm_heuristic_beats_linear` SHALL be true exactly when the
  selected clean LLM-generated heuristics have strictly fewer held-out
  env-actions-to-solve than the deployed linear heuristic, static leakage is
  clean, all counted plans reproduce, and `n_held_out_levels >= 8`.
- REQ-LEARN-4370-7: Any held-out level where no clean selected heuristic reduces
  actions below the deployed linear heuristic SHALL be represented in
  `missing_verifier_gaps` and appended to `ops/verifier_gaps.md` when the
  experiment writes the artifact.
- REQ-LEARN-4370-8: The written artifact SHALL run cleanly through
  `scripts/adversarial_verify.py` and SHALL include `random_seed`,
  `reproducibility_checksum`, `model_specs`, a linear-baseline positive control,
  and `verifier_is_oracle=false`.

### REQ-LEARN-4370 Field Principles

- `honest_verdict`: Terminal-prefixed. A win (the stronger function class
  reduces held-out actions, leakage-clean and reproduction-gated) and a clean
  null (the linear cost is already near-optimal on the solved games) are BOTH
  decision-grade.
- `llm_heuristic_beats_linear`: BARE bool: true iff the best clean
  LLM-generated heuristic reduces held-out actions-to-solve below the deployed
  linear heuristic AND static leakage is clean AND every counted plan
  reproduces.
- `held_out_actions_by_heuristic`: dict `{linear, llm_generated,
  bfs_baseline}` -> held-out env-actions-to-solve.
- `per_game_scorecard`: list of `{game, best_heuristic_src,
  held_out_actions_llm, held_out_actions_linear, static_leakage_clean,
  reproduced}`.
- `static_leakage_clean`: BARE bool: true iff every selected heuristic passed
  static leakage analysis.
- `reproduction_gated`: BARE bool: true iff every counted plan still passes
  `arc_solver_kit.reproduce`.
- `n_held_out_levels`: BARE int: held-out level count, minimum eight for the
  action-delta claim.
- `verifier_is_oracle`: BARE bool=false -- the heuristic estimates
  cost-to-go; the executable env defines the win.
- `preconditions_checked`: Records solve-trace, solver-kit, Exp 4364 baseline,
  and TRM-stand-down checks.
- `random_seed`: Determinism precondition for heuristic synthesis, held-out
  split, and selection.
- `reproducibility_checksum`: Hash of the training corpus, held-out split,
  selected heuristic programs, and reproduce results.
- `model_specs`: Generator, declared reproducible alternative GGUF, deployed
  linear baseline, held-out split, and sample size.

### SCENARIO-LEARN-4370: Clean LLM-Heuristic Artifact Is Reproduction-Gated

**Given** at least eight reproduced ARC solved-level traces are loadable
**When** Exp 4370 generates clean domain-dependent heuristic programs, selects
the strongest per game on training levels, and evaluates held-out levels
**Then** the artifact SHALL include bare `llm_heuristic_beats_linear`,
`held_out_actions_by_heuristic`, `per_game_scorecard`,
`static_leakage_clean`, `reproduction_gated`, `n_held_out_levels`,
`verifier_is_oracle=false`, `preconditions_checked`, `random_seed`,
`reproducibility_checksum`, and `model_specs`
**And** `llm_heuristic_beats_linear=true` only if held-out
`llm_generated < linear`, `static_leakage_clean=true`,
`reproduction_gated=true`, and `n_held_out_levels >= 8`.

### SCENARIO-LEARN-4370-BLOCKED: Missing Trace Or Solver Resources Stop Honestly

**Given** fewer than eight reproduced levels with action traces are loadable or
the solver-kit reproduction gate is unavailable
**When** Exp 4370 runs
**Then** it SHALL write a terminal artifact with `honest_verdict` equal to
`blocked_insufficient_solve_traces` or `blocked_solver_kit_unavailable`
**And** `llm_heuristic_beats_linear=false`, `held_out_actions_by_heuristic`
contains zero totals, `static_leakage_clean=false`,
`reproduction_gated=false`, and `n_held_out_levels=0`.

## Implementation Status (Exp 4370)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4370 | Planned (`python/carnot/experiment_4370_llm_generated_action_cost_heuristics.py`; `results/experiment_4370_llm_generated_action_cost_heuristics.py`; `results/experiment_4370_llm_generated_action_cost_heuristics.json`) | Planned (`tests/python/test_experiment_4370_llm_generated_action_cost_heuristics.py`) |

---

## REQ-LEARN-4418: Config-Rule Vocabulary Transfer Measurement

**Given** at least two ARC config games with grounded win-rules from the solved
registry or grounded Layer-B artifacts
**And** the local Gemma 4 12B Q4 scaffolded inducer is serving on the iGPU
**And** TRM training is stood down
**When** Exp 4418 derives relational primitives from solved config-game
win-rules and evaluates leave-one-game-out cold-start vs vocabulary-seeded
induction on held-out config games
**Then** the artifact SHALL report `config_rule_vocabulary`,
`transfer_learning_curve`, `config_rule_vocabulary_transfers`,
`verifier_is_oracle=false`, `preconditions_checked`, `random_seed`,
`reproducibility_checksum`, and `model_specs`
**And** `config_rule_vocabulary_transfers` SHALL be true only when the
vocabulary-seeded grounding rate exceeds cold-start on held-out games and the
paired bootstrap CI95 for the delta excludes zero.

### REQ-LEARN-4418 Sub-requirements

- REQ-LEARN-4418-1: Preconditions SHALL verify at least two grounded
  config-rule source games, the local iGPU Gemma 4 12B Q4 inducer, and TRM
  stand-down before any induction measurement.
- REQ-LEARN-4418-2: If fewer than two grounded rules are available, Exp 4418
  SHALL write `honest_verdict=blocked_insufficient_grounded_rules` and SHALL
  not claim transfer.
- REQ-LEARN-4418-3: If the local iGPU inducer is unavailable, Exp 4418 SHALL
  write `honest_verdict=blocked_local_model_unavailable` and SHALL not claim
  transfer.
- REQ-LEARN-4418-4: The vocabulary SHALL be extracted as reusable relational
  primitives, including count equality, editable/reference relations,
  position/region matches, glyph maps, sequence rewrites, and shape-pattern
  matches when those primitives are present in grounded source rules.
- REQ-LEARN-4418-5: The transfer curve SHALL contain per-held-out-game cold and
  vocabulary-seeded grounding rates, delta, and bootstrap CI95.
- REQ-LEARN-4418-6: `verifier_is_oracle` SHALL be false because grounding is
  execution-checked but the transfer claim concerns a learned vocabulary, not
  the executable oracle itself.

### SCENARIO-LEARN-4418: Vocabulary-Seeded Transfer Artifact Is Decision-Grade

**Given** grounded source rules exist for at least two solved config games
**And** the local iGPU inducer is available
**When** Exp 4418 evaluates cold-start and vocabulary-seeded arms on held-out
config games
**Then** it SHALL write a transfer curve with one record per held-out game
**And** `config_rule_vocabulary_transfers=true` only when the seeded arm's
grounding-rate delta is positive and its CI95 excludes zero.

### SCENARIO-LEARN-4418-BLOCKED: Missing Local Inducer Stops Honestly

**Given** at least two grounded source rules exist
**And** the local Gemma 4 12B Q4 iGPU inducer is not serving
**When** Exp 4418 runs
**Then** it SHALL write `honest_verdict=blocked_local_model_unavailable`
**And** `config_rule_vocabulary_transfers=false`,
`transfer_learning_curve=[]`, `verifier_is_oracle=false`, and populated
`preconditions_checked`.

## Implementation Status (Exp 4418)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4418 | Planned (`python/carnot/experiment_4418_config_rule_vocabulary_transfer.py`; `results/experiment_4418_config_rule_vocabulary_transfer.py`; `results/experiment_4418_config_rule_vocabulary_transfer.json`) | Planned (`tests/python/test_experiment_4418_config_rule_vocabulary_transfer.py`) |

---

## REQ-LEARN-4425: Qwen Config-Rule Vocabulary Transfer Retest

**Given** solved ARC config games with grounded win-rule predicates or
reproduced registry entries
**And** the Qwen3.5-9B-MTP local iGPU `/no_think` Layer-B repeat-bench output
is available for cold-start comparison
**When** Exp 4425 builds a reusable vocabulary of relational win-rule
primitives and seeds the scaffolded inducer prompt with that vocabulary
**Then** it SHALL write
`results/experiment_4425_config_rule_vocabulary_transfer.json` with
`config_rule_vocabulary`, `vocabulary_seeded_prompt`, `transfer_learning_curve`,
`config_rule_vocabulary_transfers`, `verifier_is_oracle=false`,
`random_seed`, `honest_verdict`, `logged_gaps`, `model_specs`, and
`reproducibility_checksum`
**And** `config_rule_vocabulary_transfers` SHALL be a bare boolean that is true
only when the held-out vocabulary-seeded grounding-rate lift over cold-start is
positive and the 95% confidence interval excludes zero.

### REQ-LEARN-4425 Sub-requirements

- REQ-LEARN-4425-1: The vocabulary SHALL include reusable relational
  primitives such as editable/reference count equality, match-reference,
  progress-fill, glyph-rewrite, marker-coverage, and shape-pattern match when
  those primitives are present in solved config-game sources.
- REQ-LEARN-4425-2: The seeded prompt SHALL include the vocabulary before the
  scaffolded inducer task text, and the generator metadata SHALL declare
  Qwen3.5-9B-MTP on iGPU with `/no_think`, MTP enabled, and four seeds.
- REQ-LEARN-4425-3: The transfer curve SHALL report cold-start and
  vocabulary-seeded grounded counts, grounding rates, lift, and CI95 for each
  held-out game and overall.
- REQ-LEARN-4425-4: If the vocabulary-seeded arm cannot be measured, Exp 4425
  SHALL still write a terminal artifact, set
  `config_rule_vocabulary_transfers=false`, and log the null measurement as a
  gap instead of gating or fabricating a lift.
- REQ-LEARN-4425-5: `verifier_is_oracle` SHALL be false because the verifier
  grounds proposed predicates but does not define correctness for the transfer
  claim.

### SCENARIO-LEARN-4425: Seeded Prompt And Transfer Artifact Are Stable

**Given** grounded source rules and four-seed cold/seeded arm observations
**When** Exp 4425 builds the vocabulary-seeded prompt and transfer artifact
**Then** the prompt SHALL name the relational primitives and the artifact SHALL
validate the required fields
**And** `config_rule_vocabulary_transfers=true` only when the held-out
grounding-rate lift CI95 excludes zero.

### SCENARIO-LEARN-4425-NULL: Missing Seeded Arm Logs A Gap

**Given** the four-seed Qwen cold-start repeat-bench is available
**And** no vocabulary-seeded repeat-bench artifact is available
**When** Exp 4425 runs
**Then** it SHALL write a terminal `complete:` verdict, set
`config_rule_vocabulary_transfers=false`, keep `verifier_is_oracle=false`, and
record the missing seeded-arm measurement in `logged_gaps`.

---

## REQ-LEARN-4927: Rotated ARC Continuous Self-Play Verifier Checkpoint Refresh

Experiment 4927 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, selecting a target that is
not `.453`'s `bp35`, `.452`'s `vc33`, `.451`'s `sk48`, `.450`'s `ls20`,
`.449`'s `re86`, A1's `sp80`, or A2's `su15`. The loop SHALL warm-start from
`models/arc_verifier_<game>.json`, reproduction-gate at least one offline
level, train on this run's self-play traces, refresh the checkpoint, and write
`results/experiment_4927_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4927 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`, and
`reproduced_levels >= 1`. A failed loop, missing checkpoint, stale mtime,
failed reproduction gate, or rotation back to `bp35`, `vc33`, `sk48`, `ls20`,
`re86`, `sp80`, or `su15` SHALL write a terminal `complete_*` or `blocked_*`
artifact with the residual instead of fabricating checkpoint progress.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86 AND from A1/A2)."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-4927-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `lf52` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_lf52.json` exists before
the standing loop runs
**When** `scripts/arc_loop_solve.py --game lf52` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 4927 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_lf52.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=lf52`,
`solve_provenance=live_agent_self_discovery`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-4927-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, or the checkpoint mtime does not advance
**When** Experiment 4927 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false`, preserves the
residual cause, keeps `offline_reproduced` and `reproduced_levels` grounded in
the loop result, and does not emit a success verdict.

### SCENARIO-LEARN-4927-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent or conflicting targets `bp35`, `vc33`,
`sk48`, `ls20`, `re86`, `sp80`, or `su15`
**When** Experiment 4927 checks preconditions
**Then** it writes `results/experiment_4927_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 4927)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4927 | Implemented (`python/carnot/experiment_4927_self_play_verifier_checkpoint.py`) | Implemented (`tests/python/test_experiment_4927_self_play_verifier_checkpoint.py`) |

---

## REQ-LEARN-4938: Rotated ARC Continuous Self-Play Verifier Checkpoint Refresh

Experiment 4938 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, selecting `ar25` because
it is banked, has `models/arc_verifier_ar25.json`, is not `.453`'s `bp35`,
`.452`'s `vc33`, `.451`'s `sk48`, `.450`'s `ls20`, `.449`'s `re86`, A1's
`lf52`, or A2's `sb26`. The loop SHALL warm-start from
`models/arc_verifier_ar25.json`, reproduction-gate at least one offline level,
train on this run's self-play traces, refresh the checkpoint, and write
`results/experiment_4938_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4938 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`, and
`reproduced_levels >= 1`. A failed loop, missing checkpoint, stale mtime,
failed reproduction gate, or rotation back to `bp35`, `vc33`, `sk48`, `ls20`,
`re86`, `lf52`, or `sb26` SHALL write a terminal `complete_*` or `blocked_*`
artifact with the residual instead of fabricating checkpoint progress.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86 AND from A1/A2)."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-4938-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `ar25` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_ar25.json` exists before
the standing loop runs
**When** `scripts/arc_loop_solve.py --game ar25` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 4938 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_ar25.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=ar25`,
`solve_provenance=live_agent_self_discovery`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-4938-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, or the checkpoint mtime does not advance
**When** Experiment 4938 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false`, preserves the
residual cause, keeps `offline_reproduced` and `reproduced_levels` grounded in
the loop result, and does not emit a success verdict.

### SCENARIO-LEARN-4938-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent or conflicting targets `bp35`, `vc33`,
`sk48`, `ls20`, `re86`, `lf52`, or `sb26`
**When** Experiment 4938 checks preconditions
**Then** it writes `results/experiment_4938_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 4938)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4938 | Implemented (`python/carnot/experiment_4938_self_play_verifier_checkpoint.py`) | Implemented (`tests/python/test_experiment_4938_self_play_verifier_checkpoint.py`) |

---

## REQ-LEARN-4949: Honest-Substrate ARC Self-Play Verifier Checkpoint Refresh

Experiment 4949 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, selecting `lp85` because
it is banked, has `models/arc_verifier_lp85.json`, is not `.455`'s `ar25`,
`.453`'s `bp35`, `.452`'s `vc33`, `.451`'s `sk48`, `.450`'s `ls20`,
`.449`'s `re86`, A1's `ar25`, or A2's `vc33`. The loop SHALL warm-start from
`models/arc_verifier_lp85.json`, reproduction-gate at least one offline level,
train on this run's self-play traces, refresh the checkpoint, and write
`results/experiment_4949_self_play_verifier_checkpoint.json`.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`,
`reproduced_levels >= 1`, and the fresh artifact is not live-rechecked as
critical by the adversarial verifier. A failed loop, missing checkpoint, stale
mtime, failed reproduction gate, or rotation back to `ar25`, `bp35`, `vc33`,
`sk48`, `ls20`, or `re86` SHALL write a terminal `complete_*` or `blocked_*`
artifact with the residual instead of fabricating checkpoint progress.

Experiment 4949 SHALL declare `inference_substrate` honestly. When the standing
loop uses the offline reproduction gate and learned checkpoint training without
invoking an LLM generator, the artifact SHALL declare
`verifier_ensemble_against_cached_candidates` and SHALL record a measured
`duration_s >= 1.0`. It SHALL declare `live_llm_inference` only for a measured
LLM-generator run with `duration_s >= 60.0`. This requirement fixes the Exp
4938 `DURATION_TOO_SHORT` substrate mismatch.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .455 ar25 / .453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86 AND from A1/A2)."
- `inference_substrate`: principle "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) for an offline gate run; live_llm_inference (60s floor) ONLY if the generator ran >=60s. This fixes the .455 exp4938 DURATION_TOO_SHORT flag."
- `duration_s`: principle "the measured wall-clock; must be consistent with inference_substrate (>=1s for verifier-ensemble, >=60s for live_llm_inference)."
- `flag_resolved`: principle "true iff the fresh artifact is NOT flagged true_live_recheck=critical (the .455 DURATION_TOO_SHORT substrate mismatch is fixed)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-4949-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `lp85` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_lp85.json` exists before
the standing loop runs
**When** `scripts/arc_loop_solve.py --game lp85` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 4949 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_lp85.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=lp85`,
`solve_provenance=live_agent_self_discovery`, `flag_resolved=true`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-4949-SUBSTRATE-FIX

**Given** the loop uses the offline reproduction gate and checkpoint training
without an LLM generator
**When** Experiment 4949 builds its artifact
**Then** it declares
`inference_substrate=verifier_ensemble_against_cached_candidates`, records
`duration_s >= 1.0`, and does not trigger a critical `DURATION_TOO_SHORT`
live recheck.

### SCENARIO-LEARN-4949-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, the checkpoint mtime does not advance, or the fresh
artifact is critically flagged
**When** Experiment 4949 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false` or
`flag_resolved=false` as applicable, preserves the residual cause, keeps
`offline_reproduced` and `reproduced_levels` grounded in the loop result, and
does not emit a success verdict.

### SCENARIO-LEARN-4949-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent or conflicting targets `ar25`, `bp35`,
`vc33`, `sk48`, `ls20`, or `re86`
**When** Experiment 4949 checks preconditions
**Then** it writes `results/experiment_4949_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 4949)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4949 | Implemented (`python/carnot/experiment_4949_self_play_verifier_checkpoint.py`) | Implemented (`tests/python/test_experiment_4949_self_play_verifier_checkpoint.py`) |

---

## REQ-LEARN-4960: Continued Honest-Substrate ARC Self-Play Verifier Checkpoint Refresh

Experiment 4960 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, selecting `dc22` because
it is banked, has `models/arc_verifier_dc22.json`, is not `.456`'s `lp85`,
`.455`'s `ar25`, `.453`'s `bp35`, `.452`'s `vc33`, `.451`'s `sk48`,
`.450`'s `ls20`, `.449`'s `re86`, or A1/A2's targets. The loop SHALL
warm-start from `models/arc_verifier_dc22.json`, reproduction-gate at least one
offline level, train on this run's self-play traces, refresh the checkpoint,
and write `results/experiment_4960_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4960 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`,
`reproduced_levels >= 1`, and the fresh artifact is not live-rechecked as
critical by the adversarial verifier. A failed loop, missing checkpoint, stale
mtime, failed reproduction gate, critical live recheck, or rotation back to
`lp85`, `ar25`, `bp35`, `vc33`, `sk48`, `ls20`, or `re86` SHALL write a
terminal `complete_*` or `blocked_*` artifact with the residual instead of
fabricating checkpoint progress.

Experiment 4960 SHALL maintain the Exp 4949 substrate fix. When the standing
loop uses the offline reproduction gate and checkpoint training without
invoking an LLM generator, the artifact SHALL declare
`verifier_ensemble_against_cached_candidates` and SHALL record a measured
`duration_s >= 1.0`. It SHALL declare `live_llm_inference` only for a measured
LLM-generator run with `duration_s >= 60.0`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .456 lp85 / .455 ar25 / .453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86 AND from A1/A2)."
- `inference_substrate`: principle "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) for an offline gate run; live_llm_inference (60s floor) ONLY if the generator ran >=60s. This MAINTAINS the .456 exp4949 substrate fix."
- `duration_s`: principle "the measured wall-clock; must be consistent with inference_substrate (>=1s for verifier-ensemble, >=60s for live_llm_inference)."
- `flag_resolved`: principle "true iff the fresh artifact is NOT flagged true_live_recheck=critical (the substrate declaration holds)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-4960-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `dc22` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_dc22.json` exists before
the standing loop runs
**When** `scripts/arc_loop_solve.py --game dc22` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 4960 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_dc22.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=dc22`,
`solve_provenance=live_agent_self_discovery`, `flag_resolved=true`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-4960-SUBSTRATE-FIX

**Given** the loop uses the offline reproduction gate and checkpoint training
without an LLM generator
**When** Experiment 4960 builds its artifact
**Then** it declares
`inference_substrate=verifier_ensemble_against_cached_candidates`, records
`duration_s >= 1.0`, and does not trigger a critical `DURATION_TOO_SHORT`
live recheck.

### SCENARIO-LEARN-4960-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, the checkpoint mtime does not advance, or the fresh
artifact is critically flagged
**When** Experiment 4960 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false` or
`flag_resolved=false` as applicable, preserves the residual cause, keeps
`offline_reproduced` and `reproduced_levels` grounded in the loop result, and
does not emit a success verdict.

### SCENARIO-LEARN-4960-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent or conflicting targets `lp85`, `ar25`,
`bp35`, `vc33`, `sk48`, `ls20`, or `re86`
**When** Experiment 4960 checks preconditions
**Then** it writes `results/experiment_4960_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 4960)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4960 | Implemented (`python/carnot/experiment_4960_self_play_verifier_checkpoint.py`) | Implemented (`tests/python/test_experiment_4960_self_play_verifier_checkpoint.py`) |

---

## REQ-LEARN-4971: Continued Rotated ARC Self-Play Verifier Checkpoint Refresh

Experiment 4971 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, selecting `ft09` because
the preferred `sp80` candidate reproduced but did not refresh a checkpoint under
the current adapter, while `ft09` is banked, has `models/arc_verifier_ft09.json`,
is not `.457`'s `dc22`, `.456`'s `lp85`, `.455`'s `ar25`, `.453`'s `bp35`,
`.452`'s `vc33`, `.451`'s `sk48`, `.450`'s `ls20`, `.449`'s `re86`, or A1/A2's
targets. The loop SHALL warm-start from `models/arc_verifier_ft09.json`,
reproduction-gate at least one offline level, train on this run's self-play
traces, refresh the checkpoint, and write
`results/experiment_4971_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4971 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`,
`reproduced_levels >= 1`, and the fresh artifact is not live-rechecked as
critical by the adversarial verifier. A failed loop, missing checkpoint, stale
mtime, failed reproduction gate, critical live recheck, or rotation back to
`dc22`, `lp85`, `ar25`, `bp35`, `vc33`, `sk48`, `ls20`, or `re86` SHALL write
a terminal `complete_*` or `blocked_*` artifact with the residual instead of
fabricating checkpoint progress.

Experiment 4971 SHALL maintain the Exp 4949/4960 substrate fix. When the
standing loop uses the offline reproduction gate and checkpoint training
without invoking an LLM generator, the artifact SHALL declare
`verifier_ensemble_against_cached_candidates` and SHALL record a measured
`duration_s >= 1.0`. It SHALL declare `live_llm_inference` only for a measured
LLM-generator run with `duration_s >= 60.0`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .457 dc22 / .456 lp85 / .455 ar25 / .453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86 AND from A1/A2)."
- `inference_substrate`: principle "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) for an offline gate run; live_llm_inference (60s floor) ONLY if the generator ran >=60s. This MAINTAINS the .456 exp4949 substrate fix."
- `duration_s`: principle "the measured wall-clock; must be consistent with inference_substrate (>=1s for verifier-ensemble, >=60s for live_llm_inference)."
- `flag_resolved`: principle "true iff the fresh artifact is NOT flagged true_live_recheck=critical (the substrate declaration holds)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-4971-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `ft09` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_ft09.json` exists before
the standing loop runs
**When** `scripts/arc_loop_solve.py --game ft09` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 4971 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_ft09.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=ft09`,
`solve_provenance=live_agent_self_discovery`, `flag_resolved=true`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-4971-SUBSTRATE-FIX

**Given** the loop uses the offline reproduction gate and checkpoint training
without an LLM generator
**When** Experiment 4971 builds its artifact
**Then** it declares
`inference_substrate=verifier_ensemble_against_cached_candidates`, records
`duration_s >= 1.0`, and does not trigger a critical `DURATION_TOO_SHORT`
live recheck.

### SCENARIO-LEARN-4971-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, the checkpoint mtime does not advance, or the fresh
artifact is critically flagged
**When** Experiment 4971 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false` or
`flag_resolved=false` as applicable, preserves the residual cause, keeps
`offline_reproduced` and `reproduced_levels` grounded in the loop result, and
does not emit a success verdict.

### SCENARIO-LEARN-4971-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent or conflicting targets `dc22`, `lp85`,
`ar25`, `bp35`, `vc33`, `sk48`, `ls20`, or `re86`
**When** Experiment 4971 checks preconditions
**Then** it writes `results/experiment_4971_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 4971)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4971 | Implemented (`python/carnot/experiment_4971_self_play_verifier_checkpoint.py`) | Implemented (`tests/python/test_experiment_4971_self_play_verifier_checkpoint.py`) |

---

## REQ-LEARN-4982: Continuous Rotated ARC Self-Play Verifier Checkpoint Refresh

Experiment 4982 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, preferring `ft09` because
it has `models/arc_verifier_ft09.json`, is banked, and differs from the recent
self-play rotation targets `.458 sp80`, `.457 dc22`, `.456 lp85`, `.455 ar25`,
`.453 bp35`, `.452 vc33`, `.451 sk48`, `.450 ls20`, and `.449 re86`, and also
differs from A1's `tn36` and A2's `cd82`. The loop SHALL warm-start from
`models/arc_verifier_ft09.json`, reproduction-gate at least one offline level,
train on this run's self-play traces, refresh the checkpoint, and write
`results/experiment_4982_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4982 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`,
`reproduced_levels >= 1`, and the fresh artifact is not live-rechecked as
critical by the adversarial verifier. A failed loop, missing checkpoint, stale
mtime, failed reproduction gate, critical live recheck, or rotation back to
`sp80`, `dc22`, `lp85`, `ar25`, `bp35`, `vc33`, `sk48`, `ls20`, `re86`, `tn36`,
or `cd82` SHALL write a terminal `complete_*` or `blocked_*` artifact with the
residual instead of fabricating checkpoint progress.

Experiment 4982 SHALL maintain the Exp 4949/4960 substrate fix. When the
standing loop uses the offline reproduction gate and checkpoint training
without invoking an LLM generator, the artifact SHALL declare
`verifier_ensemble_against_cached_candidates` and SHALL record a measured
`duration_s >= 1.0`. It SHALL declare `live_llm_inference` only for a measured
LLM-generator run with `duration_s >= 60.0`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .458 sp80 / .457 dc22 / .456 lp85 / .455 ar25 / .453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86 AND from A1/A2)."
- `inference_substrate`: principle "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) for an offline gate run; live_llm_inference (60s floor) ONLY if the generator ran >=60s. This MAINTAINS the .456 exp4949 substrate fix."
- `duration_s`: principle "the measured wall-clock; must be consistent with inference_substrate (>=1s for verifier-ensemble, >=60s for live_llm_inference)."
- `flag_resolved`: principle "true iff the fresh artifact is NOT flagged true_live_recheck=critical (the substrate declaration holds)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-4982-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `ft09` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_ft09.json` exists before
the standing loop runs
**When** `scripts/arc_loop_solve.py --game ft09` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 4982 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_ft09.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=ft09`,
`solve_provenance=live_agent_self_discovery`, `flag_resolved=true`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-4982-SUBSTRATE-FIX

**Given** the loop uses the offline reproduction gate and checkpoint training
without an LLM generator
**When** Experiment 4982 builds its artifact
**Then** it declares
`inference_substrate=verifier_ensemble_against_cached_candidates`, records
`duration_s >= 1.0`, and does not trigger a critical `DURATION_TOO_SHORT`
live recheck.

### SCENARIO-LEARN-4982-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, the checkpoint mtime does not advance, or the fresh
artifact is critically flagged
**When** Experiment 4982 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false` or
`flag_resolved=false` as applicable, preserves the residual cause, keeps
`offline_reproduced` and `reproduced_levels` grounded in the loop result, and
does not emit a success verdict.

### SCENARIO-LEARN-4982-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent or conflicting targets `sp80`, `dc22`,
`lp85`, `ar25`, `bp35`, `vc33`, `sk48`, `ls20`, `re86`, `tn36`, or `cd82`
**When** Experiment 4982 checks preconditions
**Then** it writes `results/experiment_4982_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 4982)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4982 | Implemented (`python/carnot/experiment_4982_self_play_verifier_checkpoint.py`) | Implemented (`tests/python/test_experiment_4982_self_play_verifier_checkpoint.py`) |

---

## REQ-LEARN-4993: Continuous Rotated ARC Self-Play Verifier Checkpoint Refresh

Experiment 4993 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, preferring `su15` first
because it has `models/arc_verifier_su15.json`, is banked, and differs from the
recent self-play rotation targets `.459 ft09`, `.458 sp80`, `.457 dc22`, `.456
lp85`, `.455 ar25`, `.453 bp35`, `.452 vc33`, `.451 sk48`, `.450 ls20`, and
`.449 re86`, and also differs from A1's `sc25` and A2's `cd82`. If `su15`
reproduces but does not refresh a learned verifier checkpoint, the experiment
SHALL record that residual and may use checkpointed alternate `lf52`, which also
differs from the recent rotation and A1/A2 targets. The successful loop SHALL
warm-start from `models/arc_verifier_lf52.json`, reproduction-gate at least one
offline level, train on this run's self-play traces, refresh the checkpoint,
and write `results/experiment_4993_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4993 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`,
`reproduced_levels >= 1`, and the fresh artifact is not live-rechecked as
critical by the adversarial verifier. A failed loop, missing checkpoint, stale
mtime, failed reproduction gate, critical live recheck, or rotation back to
`ft09`, `sp80`, `dc22`, `lp85`, `ar25`, `bp35`, `vc33`, `sk48`, `ls20`, `re86`,
`sc25`, or `cd82` SHALL write a terminal `complete_*` or `blocked_*` artifact
with the residual instead of fabricating checkpoint progress.

Experiment 4993 SHALL maintain the Exp 4949/4960/4971/4982 substrate fix. When
the standing loop uses the offline reproduction gate and checkpoint training
without invoking an LLM generator, the artifact SHALL declare
`verifier_ensemble_against_cached_candidates` and SHALL record a measured
`duration_s >= 1.0`. It SHALL declare `live_llm_inference` only for a measured
LLM-generator run with `duration_s >= 60.0`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .459 ft09 / .458 sp80 / .457 dc22 / .456 lp85 / .455 ar25 / .453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86 AND from A1/A2)."
- `inference_substrate`: principle "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) for an offline gate run; live_llm_inference (60s floor) ONLY if the generator ran >=60s. This MAINTAINS the .456 exp4949 substrate fix."
- `duration_s`: principle "the measured wall-clock; must be consistent with inference_substrate (>=1s for verifier-ensemble, >=60s for live_llm_inference)."
- `flag_resolved`: principle "true iff the fresh artifact is NOT flagged true_live_recheck=critical (the substrate declaration holds)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-4993-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `lf52` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_lf52.json` exists before
the standing loop runs after the preferred `su15` attempt failed to refresh a
checkpoint
**When** `scripts/arc_loop_solve.py --game lf52` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 4993 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_lf52.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=lf52`,
`solve_provenance=live_agent_self_discovery`, `flag_resolved=true`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-4993-SUBSTRATE-FIX

**Given** the loop uses the offline reproduction gate and checkpoint training
without an LLM generator
**When** Experiment 4993 builds its artifact
**Then** it declares
`inference_substrate=verifier_ensemble_against_cached_candidates`, records
`duration_s >= 1.0`, and does not trigger a critical `DURATION_TOO_SHORT`
live recheck.

### SCENARIO-LEARN-4993-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, the checkpoint mtime does not advance, or the fresh
artifact is critically flagged
**When** Experiment 4993 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false` or
`flag_resolved=false` as applicable, preserves the residual cause, keeps
`offline_reproduced` and `reproduced_levels` grounded in the loop result, and
does not emit a success verdict.

### SCENARIO-LEARN-4993-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent or conflicting targets `ft09`, `sp80`,
`dc22`, `lp85`, `ar25`, `bp35`, `vc33`, `sk48`, `ls20`, `re86`, `sc25`, or
`cd82`
**When** Experiment 4993 checks preconditions
**Then** it writes `results/experiment_4993_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 4993)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-4993 | Implemented (`python/carnot/experiment_4993_self_play_verifier_checkpoint.py`) | Implemented (`tests/python/test_experiment_4993_self_play_verifier_checkpoint.py`) |

---

## REQ-LEARN-5011: Preferred r11l ARC Self-Play Verifier Checkpoint Refresh

Experiment 5011 SHALL run the standing ARC self-play loop on the preferred
banked target `r11l` with the existing `models/arc_verifier_r11l.json`
checkpoint. The loop SHALL warm-start from the saved checkpoint when its feature
arity matches the adapter, reproduction-gate at least one offline level, train
on this run's self-play traces, refresh the learned verifier checkpoint, and
write `results/experiment_5011_self_play_verifier_checkpoint.json`. If `r11l`
cannot satisfy the checkpoint-refresh precondition, the experiment SHALL record
the blocked resource or residual honestly and SHALL NOT fabricate refresh
evidence.

`r11l` SHALL expose a deterministic learned-verifier feature vector compatible
with its 4-weight adapter-free first-solve checkpoint: nonzero cell count,
distinct color count, and cell count, plus the checkpoint's stored bias term.
Loading the existing checkpoint with this feature vector SHALL not require an
LLM generator; failure to load SHALL fall back to the hand verifier without
claiming a warm-started checkpoint.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that `r11l` has a banked level,
and that `models/arc_verifier_r11l.json` exists before the run. If any required
resource is missing, Experiment 5011 SHALL write a terminal `blocked_*` artifact
that records the failed precondition and SHALL NOT claim a refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`,
`reproduced_levels >= 1`, and the fresh artifact is not live-rechecked as
critical by the adversarial verifier. A failed loop, missing checkpoint, stale
mtime, failed reproduction gate, critical live recheck, or rotation back to
`su15`, `ft09`, `sp80`, `dc22`, `lp85`, `ar25`, `bp35`, `vc33`, `sk48`,
`ls20`, or `re86` SHALL write a terminal `complete_*` or `blocked_*` artifact
with the residual instead of fabricating checkpoint progress.

Experiment 5011 SHALL maintain the Exp 4949/4960/4971/4982/4993 substrate fix.
When the standing loop uses the offline reproduction gate and checkpoint
training without invoking an LLM generator, the artifact SHALL declare
`verifier_ensemble_against_cached_candidates` and SHALL record a measured
`duration_s >= 1.0`. It SHALL declare `live_llm_inference` only for a measured
LLM-generator run with `duration_s >= 60.0`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11 / continuous self-learning)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .460 su15 / .459 ft09 / .458 sp80 / .457 dc22 / .456 lp85 / .455 ar25 / .453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts (NOT outer_loop_re)."
- `inference_substrate`: principle "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) for an offline gate run; live_llm_inference (60s floor) ONLY if the generator ran >=60s (maintains the .456 substrate fix)."
- `duration_s`: principle "measured wall-clock; consistent with inference_substrate."
- `flag_resolved`: principle "true iff NOT flagged true_live_recheck=critical."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-5011-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `r11l` is banked in
`ops/arc_solve_registry.yaml`, `models/arc_verifier_r11l.json` exists before
the standing loop runs, and the `r11l` adapter exposes a compatible 3-feature
learned-verifier vector
**When** `scripts/arc_loop_solve.py --game r11l` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 5011 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_r11l.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=r11l`,
`solve_provenance=live_agent_self_discovery`, `flag_resolved=true`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-5011-SUBSTRATE-FIX

**Given** the loop uses the offline reproduction gate and checkpoint training
without an LLM generator
**When** Experiment 5011 builds its artifact
**Then** it declares
`inference_substrate=verifier_ensemble_against_cached_candidates`, records
`duration_s >= 1.0`, and does not trigger a critical `DURATION_TOO_SHORT`
live recheck.

### SCENARIO-LEARN-5011-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, the checkpoint mtime does not advance, or the fresh
artifact is critically flagged
**When** Experiment 5011 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false` or
`flag_resolved=false` as applicable, preserves the residual cause, keeps
`offline_reproduced` and `reproduced_levels` grounded in the loop result, and
does not emit a success verdict.

### SCENARIO-LEARN-5011-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent targets `su15`, `ft09`, `sp80`, `dc22`,
`lp85`, `ar25`, `bp35`, `vc33`, `sk48`, `ls20`, or `re86`
**When** Experiment 5011 checks preconditions
**Then** it writes `results/experiment_5011_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 5011)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-5011 | Implemented (`python/carnot/experiment_5011_self_play_verifier_checkpoint.py`) | Implemented (`tests/python/test_experiment_5011_self_play_verifier_checkpoint.py`) |

---

## REQ-LEARN-5025: Preferred lf52 ARC Self-Play Verifier Checkpoint Refresh

Experiment 5025 SHALL run the standing ARC self-play loop on the preferred
banked target `lf52` with the existing `models/arc_verifier_lf52.json`
checkpoint, because `lf52` is banked, checkpointed, and is not in the recent
self-play rotation. The loop SHALL warm-start from the saved checkpoint when
available, reproduction-gate at least one offline level, train on this run's
self-play traces, refresh the learned verifier checkpoint, and write
`results/experiment_5025_self_play_verifier_checkpoint.json`. If `lf52` cannot
satisfy the checkpoint-refresh precondition, `ls20` MAY be used as a
checkpointed alternate only if it is explicitly selected; otherwise the
experiment SHALL record the blocked resource or residual honestly and SHALL NOT
fabricate refresh evidence.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 5025 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`,
`reproduced_levels >= 1`, and the fresh artifact is not live-rechecked as
critical by the adversarial verifier. A failed loop, missing checkpoint, stale
mtime, failed reproduction gate, critical live recheck, or rotation back to
`r11l`, `su15`, `ft09`, `sp80`, `dc22`, `lp85`, `ar25`, `bp35`, `vc33`,
`sk48`, `ls20`, or `re86` SHALL write a terminal `complete_*` or `blocked_*`
artifact with the residual instead of fabricating checkpoint progress.

Experiment 5025 SHALL maintain the Exp 4949/4960/4971/4982/4993/5011 substrate
fix. When the standing loop uses the offline reproduction gate and checkpoint
training without invoking an LLM generator, the artifact SHALL declare
`verifier_ensemble_against_cached_candidates` and SHALL record a measured
`duration_s >= 1.0`. It SHALL declare `live_llm_inference` only for a measured
LLM-generator run with `duration_s >= 60.0`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11 / continuous self-learning)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .461 r11l / .460 su15 / .459 ft09 / .458 sp80 / .457 dc22 / .456 lp85 / .455 ar25 / .453 bp35 / .452 vc33 / .451 sk48 / .450 ls20 / .449 re86)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts (NOT outer_loop_re)."
- `inference_substrate`: principle "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) for an offline gate run; live_llm_inference (60s floor) ONLY if the generator actually ran >=60s (maintains the .456 substrate fix)."
- `duration_s`: principle "measured wall-clock; consistent with inference_substrate."
- `flag_resolved`: principle "true iff NOT flagged true_live_recheck=critical."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-5025-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `lf52` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_lf52.json` exists before
the standing loop runs
**When** `scripts/arc_loop_solve.py --game lf52` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 5025 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_lf52.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=lf52`,
`solve_provenance=live_agent_self_discovery`, `flag_resolved=true`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-5025-SUBSTRATE-FIX

**Given** the loop uses the offline reproduction gate and checkpoint training
without an LLM generator
**When** Experiment 5025 builds its artifact
**Then** it declares
`inference_substrate=verifier_ensemble_against_cached_candidates`, records
`duration_s >= 1.0`, and does not trigger a critical `DURATION_TOO_SHORT`
live recheck.

### SCENARIO-LEARN-5025-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, the checkpoint mtime does not advance, or the fresh
artifact is critically flagged
**When** Experiment 5025 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false` or
`flag_resolved=false` as applicable, preserves the residual cause, keeps
`offline_reproduced` and `reproduced_levels` grounded in the loop result, and
does not emit a success verdict.

### SCENARIO-LEARN-5025-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent targets `r11l`, `su15`, `ft09`, `sp80`,
`dc22`, `lp85`, `ar25`, `bp35`, `vc33`, `sk48`, `ls20`, or `re86`
**When** Experiment 5025 checks preconditions
**Then** it writes `results/experiment_5025_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 5025)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-5025 | Implemented (`python/carnot/experiment_5025_self_play_verifier_checkpoint.py`) | Implemented (`tests/python/test_experiment_5025_self_play_verifier_checkpoint.py`) |

---

## REQ-LEARN-5039: Preferred ls20 ARC Self-Play Verifier Checkpoint Refresh

Experiment 5039 SHALL run the standing ARC self-play loop on the preferred
banked target `ls20` with the existing `models/arc_verifier_ls20.json`
checkpoint, because `ls20` is banked, checkpointed, and explicitly selected for
the current continuous self-learning rotation. The loop SHALL warm-start from
the saved checkpoint when available, reproduction-gate at least one offline
level, train on this run's self-play traces, refresh the learned verifier
checkpoint, and write
`results/experiment_5039_self_play_verifier_checkpoint.json`. If `ls20` cannot
satisfy the checkpoint-refresh precondition, checkpointed alternate `sk48` MAY
be used only when explicitly selected; otherwise the experiment SHALL record the
blocked resource or residual honestly and SHALL NOT fabricate refresh evidence.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 5039 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`,
`reproduced_levels >= 1`, and the fresh artifact is not live-rechecked as
critical by the adversarial verifier. A failed loop, missing checkpoint, stale
mtime, failed reproduction gate, critical live recheck, or rotation back to
`lf52`, `r11l`, `su15`, or `ft09` SHALL write a terminal `complete_*` or
`blocked_*` artifact with the residual instead of fabricating checkpoint
progress.

Experiment 5039 SHALL maintain the Exp 4949/4960/4971/4982/4993/5011/5025
substrate fix. When the standing loop uses the offline reproduction gate and
checkpoint training without invoking an LLM generator, the artifact SHALL
declare `verifier_ensemble_against_cached_candidates` and SHALL record a
measured `duration_s >= 1.0`. It SHALL declare `live_llm_inference` only for a
measured LLM-generator run with `duration_s >= 60.0`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11 / continuous self-learning)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .462 lf52 / .461 r11l / .460 su15 / .459 ft09)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts (NOT outer_loop_re)."
- `inference_substrate`: principle "the HONEST substrate -- verifier_ensemble_against_cached_candidates (1s floor) for an offline gate run; live_llm_inference (60s floor) ONLY if the generator actually ran >=60s (maintains the .456 substrate fix)."
- `duration_s`: principle "measured wall-clock; consistent with inference_substrate."
- `flag_resolved`: principle "true iff NOT flagged true_live_recheck=critical."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

### SCENARIO-LEARN-5039-CHECKPOINT-REFRESHED

**Given** `arc_solver_kit.offline_arcade()` is available, `ls20` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_ls20.json` exists before
the standing loop runs
**When** `scripts/arc_loop_solve.py --game ls20` reports an offline-reproduced
level and writes the learned verifier checkpoint
**Then** Experiment 5039 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_ls20.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=ls20`,
`solve_provenance=live_agent_self_discovery`, `flag_resolved=true`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

### SCENARIO-LEARN-5039-SUBSTRATE-FIX

**Given** the loop uses the offline reproduction gate and checkpoint training
without an LLM generator
**When** Experiment 5039 builds its artifact
**Then** it declares
`inference_substrate=verifier_ensemble_against_cached_candidates`, records
`duration_s >= 1.0`, and does not trigger a critical `DURATION_TOO_SHORT`
live recheck.

### SCENARIO-LEARN-5039-RESIDUAL-NO-FABRICATION

**Given** the loop result is missing, the reproduction gate fails, the
checkpoint path is absent, the checkpoint mtime does not advance, or the fresh
artifact is critically flagged
**When** Experiment 5039 builds its artifact
**Then** it records `verifier_checkpoint_refreshed=false` or
`flag_resolved=false` as applicable, preserves the residual cause, keeps
`offline_reproduced` and `reproduced_levels` grounded in the loop result, and
does not emit a success verdict.

### SCENARIO-LEARN-5039-BLOCKED-PRECONDITION

**Given** the offline arcade is unavailable, the registry is missing, the
selected target is not banked, the checkpoint file is absent before the run, or
the selection is one of the recent targets `lf52`, `r11l`, `su15`, or `ft09`
**When** Experiment 5039 checks preconditions
**Then** it writes `results/experiment_5039_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

## Implementation Status (Exp 5039)

| Requirement | Python | Tests |
|---|---|---|
| REQ-LEARN-5039 | Planned (`python/carnot/experiment_5039_self_play_verifier_checkpoint.py`) | Planned (`tests/python/test_experiment_5039_self_play_verifier_checkpoint.py`) |
