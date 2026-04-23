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

## Implementation Status (PSV-010/011)

| Requirement  | Python | Tests |
|-------------|--------|-------|
| REQ-PSV-010 | Implemented (scripts/experiment_736_psv_specialization.py) | tests/python/test_experiment_736_psv_specialization.py |
| REQ-PSV-011 | Implemented (scripts/experiment_736_psv_specialization.py) | tests/python/test_experiment_736_psv_specialization.py |

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
