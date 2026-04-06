# LLM-EBM Inference Capability Specification

**Capability:** llm-ebm-inference
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-11, FR-12

## Overview

Defines the first concrete "anti-hallucination" pipeline: an LLM proposes a solution to a constraint satisfaction problem (SAT, graph coloring), and the EBM verifies it, repairs violations via gradient descent, and issues a verification certificate. This bridges Carnot's verifiable reasoning (FR-12) with LLM output, demonstrating that energy-based verification can catch and fix hallucinated reasoning.

The pipeline reuses existing primitives (BaseConstraint, ComposedEnergy, repair(), VerificationResult) and adds domain-specific constraint encoders plus an LLM output parser.

## Requirements

### REQ-INFER-001: SAT Constraint Encoding

The system shall encode CNF SAT clauses as differentiable energy terms:
- Variables x_i represented as continuous values in [0, 1] (0 = False, 1 = True)
- Each clause uses product relaxation: E = prod(1 - literal_value) for each literal
  - Energy = 0 when any literal is true (clause satisfied)
  - Energy = 1 when all literals are false (clause violated)
- A binary penalty constraint pushes variables toward {0, 1}: E = sum(x_i * (1 - x_i))
- The system shall support DIMACS CNF format parsing
- `build_sat_energy()` shall produce a ComposedEnergy with one constraint per clause plus binary penalty

### REQ-INFER-002: Graph Coloring Constraint Encoding

The system shall encode graph coloring as differentiable energy terms:
- Each node's color is a continuous value in [0, n_colors - 1]
- Adjacent nodes must have different colors: E = max(0, 1 - |x_a - x_b|)^2
- A range penalty keeps values in bounds: E = sum(max(0, -x_i)^2 + max(0, x_i - (n_colors-1))^2)
- `build_coloring_energy()` shall produce a ComposedEnergy with one constraint per edge plus range penalty

### REQ-INFER-003: LLM Output Parsing

The system shall parse LLM text output into JAX arrays for verification:
- SAT: supports "1 0 1 0", "x1=True, x2=False", "T F T F" formats
- Graph coloring: supports "0 1 2 0 1" and "node 0: red, node 1: blue" formats
- Raises ValueError on unparseable input or wrong variable count

### REQ-INFER-004: Verify-and-Repair Pipeline

The system shall provide a complete verify-and-repair pipeline:
1. Score the LLM's assignment against the energy function
2. If constraints are violated, run gradient repair (reusing existing `repair()`)
3. Round the repaired continuous assignment to discrete values
4. Re-verify the rounded result
5. Return a VerifyRepairResult containing initial, repaired, and rounded verifications plus the repair trajectory

### REQ-INFER-005: Benchmark Harness

The system shall provide a benchmark harness that:
- Generates random SAT instances (configurable n_vars, n_clauses, clause_size)
- Generates random graph coloring instances (configurable n_nodes, n_colors, edge probability)
- Generates random assignments (simulating LLM output)
- Runs the verify-and-repair pipeline on each instance
- Reports aggregated statistics (initial violations, repaired violations, success rate, energy reduction)

## Scenarios

### SCENARIO-INFER-001: SAT Clause Verification

**Given** a 3-SAT clause (x1 OR NOT x2 OR x3)
**And** assignment x1=True, x2=True, x3=False
**When** the clause constraint energy is computed
**Then** energy is approximately 0 (clause satisfied by x1=True)

**Given** the same clause with assignment x1=False, x2=False, x3=False
**When** energy is computed
**Then** energy is approximately 1 (clause violated: x1=F, NOT x2=T but x3=F... wait, NOT x2=True so clause IS satisfied)
**Corrected**: assignment x1=False, x2=True, x3=False → NOT x2=False, all literals false → energy ≈ 1

### SCENARIO-INFER-002: DIMACS Parsing

**Given** a DIMACS CNF string with header "p cnf 3 2" and clauses "1 -2 3 0" and "-1 2 0"
**When** parsed
**Then** produces 2 SATClause objects with correct variable indices and negation flags
**And** n_vars = 3

### SCENARIO-INFER-003: Graph Coloring Verification

**Given** a triangle graph (3 nodes, all pairs connected)
**And** coloring [0, 1, 2] (all different)
**When** coloring energy is computed
**Then** energy is approximately 0 (no adjacent same-color)

**Given** coloring [0, 0, 1] (nodes 0 and 1 share color)
**When** energy is computed
**Then** energy > 0 (edge 0-1 violated)

### SCENARIO-INFER-004: Gradient Repair Reduces Violations

**Given** a random 3-SAT instance with 20 variables and 60 clauses
**And** a random assignment violating multiple clauses
**When** gradient repair runs for up to 200 steps
**Then** the number of violated clauses decreases
**And** total energy decreases monotonically (or at least overall)

### SCENARIO-INFER-005: Full Pipeline

**Given** a SAT instance with a known satisfying assignment
**And** a deliberately wrong assignment (simulating LLM hallucination)
**When** verify_and_repair() is called
**Then** initial_verification shows violations
**And** repaired_verification shows fewer violations
**And** the result contains a repair trajectory

### SCENARIO-INFER-006: Binary Penalty

**Given** SAT variables at x_i = 0.5 (maximally ambiguous)
**When** binary penalty energy is computed
**Then** energy is maximal
**Given** variables at x_i ∈ {0, 1}
**Then** binary penalty energy is 0

### REQ-INFER-006: LLM Solver Integration

The system shall provide functions to send constraint problems to an LLM and parse the responses:
- `solve_sat_with_llm()`: builds SAT prompt, calls OpenAI-compatible API, returns raw text
- `solve_coloring_with_llm()`: builds coloring prompt, calls API
- `run_llm_sat_experiment()`: full pipeline — LLM solve → parse → verify → repair → certify
- `run_llm_coloring_experiment()`: full pipeline for coloring
- Graceful degradation when openai package is not installed or API fails

### SCENARIO-INFER-007: Full LLM Pipeline

**Given** a SAT instance sent to an LLM via the solver
**When** the LLM returns a (possibly incorrect) assignment
**Then** the assignment is parsed and verified against the energy function
**And** if violated, gradient repair improves the assignment
**And** the full VerifyRepairResult contains initial, repaired, and rounded verifications

### REQ-INFER-007: Learned Energy Functions

The system shall support training EBMs to learn verification criteria from (correct, incorrect) example pairs:
- Generate training data via rejection sampling (satisfying assignments = data, random = noise)
- Train using NCE loss from `carnot.training.nce`
- Wrap the trained model as a `BaseConstraint` for use in the verify-and-repair pipeline
- Provide comparison tools to measure learned verifier accuracy against hand-coded ground truth

### SCENARIO-INFER-008: Learned SAT Verifier

**Given** a random 3-SAT instance with 5 variables and 15 clauses
**And** a Gibbs model trained on satisfying vs random assignments via NCE
**When** the trained model scores a new satisfying assignment
**Then** its energy is lower than for a violating assignment
**And** the learned verifier classifies at least 70% of test assignments correctly
**And** verify-and-repair using the learned energy improves random assignments

### REQ-INFER-014: Hallucination Direction Detection

The system shall detect hallucination by finding the principal direction in activation space that separates correct from hallucinated LLM outputs:
- Compute mean-difference direction between correct and hallucinated activation clusters
- Optional SVD refinement for top-k distinguishing directions
- Hallucination energy: projection of activation onto the direction (high = likely hallucination)
- HallucinationDirectionConstraint wraps the energy for ComposedEnergy integration
- One-sided (ReLU) penalty: only penalizes projections toward hallucination, not away

### SCENARIO-INFER-014-001: Hallucination Direction Discovery

**Given** per-layer activations from 50 correct and 50 hallucinated outputs
**When** `find_hallucination_direction()` is called
**Then** the discovered direction aligns (cosine similarity > 0.8) with the true separation axis
**And** hallucinated activations get higher energy than correct ones
**And** the direction is unit-normalized by default

### REQ-INFER-015: EBM-Guided Rejection Sampling

The system shall combine per-token activation energy (from a trained Gibbs EBM) with logprob scores for candidate selection:
- Generate N candidates with sampling
- Extract per-token hidden states from a configurable transformer layer
- Score each token through the trained EBM (low energy = likely correct)
- Combine as composite: ebm_weight * mean_ebm_energy - logprob_weight * mean_logprob
- Select candidate with lowest composite energy
- Configurable weights allow pure EBM, pure logprob, or weighted combination

### SCENARIO-INFER-015-001: EBM Scoring Correctness

**Given** a Gibbs EBM trained on correct vs wrong activations
**And** per-token activations from a generated response
**When** `score_activations_with_ebm()` is called
**Then** the mean energy is finite and reflects the EBM's learned distinction
**And** activations from the trained distribution get lower energy than out-of-distribution

### REQ-INFER-016: Multi-Layer Hallucination Probing

The system shall probe each transformer layer independently to find where hallucination signal is strongest:
- Train a small Gibbs EBM probe at each layer using NCE loss
- Report per-layer train/test accuracy and energy gap
- Identify the layer with highest test accuracy as the best probe layer
- Support probing a subset of layers for efficiency

### SCENARIO-INFER-016-001: Best Layer Identification

**Given** per-layer activations from correct and wrong model outputs with varying separability
**When** `probe_all_layers()` is called
**Then** the layer with highest separation is identified as best_layer
**And** the best layer's test accuracy exceeds layers with lower separation
**And** results include per-layer accuracy, gap, and train/test counts

## Implementation Status

| Requirement | Python | Tests |
|------------|--------|-------|
| REQ-INFER-001 | Implemented | 15+ Python |
| REQ-INFER-002 | Implemented | 8+ Python |
| REQ-INFER-003 | Implemented | 8+ Python |
| REQ-INFER-004 | Implemented | 6+ Python |
| REQ-INFER-005 | Implemented | 7+ Python |
| REQ-INFER-006 | Implemented | 14+ Python |
| REQ-INFER-007 | Implemented | 18+ Python |
| REQ-INFER-014 | Implemented | 35 Python |
| REQ-INFER-015 | Implemented | 14 Python |
| REQ-INFER-016 | Implemented | 10 Python |
