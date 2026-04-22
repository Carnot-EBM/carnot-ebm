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
