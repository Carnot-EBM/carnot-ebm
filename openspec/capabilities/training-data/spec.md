# Training Data Capability Specification

**Capability:** training-data
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-14

## Overview

Defines how Carnot generates labeled training corpora for the FoVer (Formal Verification)
JEPA predictor.  The predictor needs step-level (step, correct/incorrect) pairs; this
capability covers the labeling pipelines that produce them at scale without human annotation.

Two complementary labeling methods are specified:

- **Z3 / SMT labeling** (FoVer v1, arXiv 2505.15960): checks symbolic arithmetic entailment.
- **PDDL transition labeling** (FoVer v2, arXiv 2604.17957): checks state-update correctness
  for word problems by encoding each CoT step as a PDDL state-action-state transition.

## Requirements

### REQ-DATA-001: Labeled Pair Schema

Every labeled pair shall contain at minimum:
- `question`: the source problem text
- `step_text`: the reasoning step being labeled
- `step_index`: integer position within the CoT chain
- `step_correct`: boolean ground-truth label
- `labeler`: string identifying the labeling method used (e.g. `"z3"`, `"pddl"`)

### REQ-DATA-002: Z3 Labeling Pipeline

The system shall implement `FoVerZ3Labeler` (in `carnot.training.fover_z3_labeler`)
that uses Z3 SMT solving to assign step-level labels to arithmetic reasoning chains.

### REQ-DATA-003: FoVer v1 Corpus

The system shall produce a FoVer v1 corpus of at least 200 Z3-labeled step pairs,
stored at `results/fover_labeled_formal_v1.json`.

**Implementation Status:** Complete (Exp 686 — 200 pairs)

### REQ-DATA-004: Corpus File Format

Corpus files shall be JSON with top-level keys matching `REQUIRED_RESULT_FIELDS`
from `ExperimentTemplate` plus a `pairs` list conforming to REQ-DATA-001.

### REQ-DATA-005: FoVer v2 Combined Corpus

The system shall produce a FoVer v2 corpus combining Z3 labels (from v1) and
PDDL transition labels, containing a total of at least 1000 step-level labeled pairs.
The corpus shall be stored at `results/fover_v2_combined.json`.

**Acceptance criteria:**
- `n_total_pairs >= 1000` → `honest_verdict = "fover_v2_target_met"`
- `500 <= n_total_pairs < 1000` → `honest_verdict = "fover_v2_partial"`
- `n_total_pairs < 500` → `honest_verdict = "fover_v2_insufficient"`

### REQ-DATA-006: PDDL State Encoder

The system shall implement `extract_quantities(problem_text)` in
`carnot.training.pddl_labeler` that:
- Parses named numeric quantities from GSM8K-style word problems using regex heuristics
- Returns a `dict[str, float]` mapping quantity names to their numeric values
- Handles integer and decimal values; ignores non-numeric tokens

**Why PDDL state encoding?**  Z3 verifies symbolic entailment but misses
state-update errors (e.g., "multiply when you should add") when the formula
is syntactically valid.  Encoding each step as a PDDL state-action-state
transition catches these semantic errors by checking whether the resulting
state matches the stated arithmetic outcome.

### REQ-DATA-007: PDDL Transition Verifier

The system shall implement `verify_transition(step_text, prev_state, next_state)`
in `carnot.training.pddl_labeler` that:
- Extracts arithmetic expressions from `step_text` using regex
- Evaluates each candidate expression via `eval()` in a restricted namespace
- Returns `True` if any evaluated result matches a value in `next_state` that
  differs from `prev_state` (i.e., the step's arithmetic produced the right update)
- Returns `False` when no expression can be extracted or none matches

**Why eval()?**  Mirrors the SymCodeVerifier pattern (Exp 619/686).  No GPU
required; ~1 µs per step vs ~200 ms for Z3.  The restricted namespace prevents
arbitrary code execution: only numeric literals and the four arithmetic operators
are passed through.

## Scenarios

### SCENARIO-DATA-005: FoVer v2 target met

**Given** 400 GSM8K questions each with 2-3 synthetic CoT steps  
**When** the PDDL labeler runs and results are combined with 200 Z3 v1 pairs  
**Then** `n_total_pairs >= 1000` and `honest_verdict = "fover_v2_target_met"`

### SCENARIO-DATA-006: Quantity extraction from word problem

**Given** the text "Alice has 5 apples and 3 oranges"  
**When** `extract_quantities` is called  
**Then** the returned dict contains `{"apples": 5.0, "oranges": 3.0}` (and possibly `{"alice": ...}` depending on heuristics)

### SCENARIO-DATA-007: Transition verification — correct step

**Given** `prev_state = {"apples": 5.0}`, `next_state = {"apples": 8.0}`, `step_text = "5 + 3 = 8"`  
**When** `verify_transition` is called  
**Then** the function returns `True`

### SCENARIO-DATA-007b: Transition verification — incorrect step

**Given** `prev_state = {"apples": 5.0}`, `next_state = {"apples": 8.0}`, `step_text = "5 * 3 = 15"`  
**When** `verify_transition` is called  
**Then** the function returns `False`

## Implementation Status

| Requirement | Status |
|-------------|--------|
| REQ-DATA-001 | Complete (Exp 686) |
| REQ-DATA-002 | Complete (Exp 686) |
| REQ-DATA-003 | Complete (Exp 686) |
| REQ-DATA-004 | Complete (Exp 686) |
| REQ-DATA-005 | Implemented (Exp 712) |
| REQ-DATA-006 | Implemented (Exp 712) |
| REQ-DATA-007 | Implemented (Exp 712) |
