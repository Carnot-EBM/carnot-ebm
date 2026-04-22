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
