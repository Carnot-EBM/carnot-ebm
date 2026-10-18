# Semantic Pruning Capability Specification

**Capability:** semantic-pruning
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-11

## REQ-LEARN-1761: Semantic Pruning
**Statement:** SemanticPruner MUST discard redundant constraints from the continual memory buffer based on semantic similarity.
**Acceptance criteria:**
- `SemanticPruner.prune(memory_states)` returns a list of unique states.
- States with cosine similarity >= threshold to an already kept state are discarded.

## SCENARIO-LEARN-1761: Redundant states are discarded
**Given** two states with identical or highly similar vectors
**When** `SemanticPruner.prune` is called
**Then** only the first state is kept and the redundant state is discarded.

## REQ-LEARN-1762: EBFT Stability Testing on LTLZinc
**Statement:** The pruned, EBFT-trained system MUST be validated on the LTLZinc spatial dataset to measure forgetting rates and reasoning stability.
**Acceptance criteria:**
- Exp 1762 runs the validation and produces `results/experiment_1762_stability.json`.
- The artifact reports `forgetting_rate` and `reasoning_stability_score`.

## SCENARIO-LEARN-1762: Stability Metrics Recorded
**Given** an EBFT-trained model and LTLZinc spatial dataset
**When** Exp 1762 is run
**Then** forgetting rates and reasoning stability are measured and written to the artifact.

## REQ-LEARN-1868: LTLZinc Memory Retention on Pruned Traces
**Statement:** The semantic pruning mechanism MUST guarantee zero catastrophic forgetting, verified using LTLZinc constraint evaluations.
**Acceptance criteria:**
- LTLZinc temporal evaluations are run on the semantically pruned memory.
- The nonforgetting rate MUST be 1.0.
- Exp 1868 produces `results/experiment_1868_ltlzinc.json`.

## SCENARIO-LEARN-1868: Pruned Traces Retain Temporal Constraints
**Given** an updated FR-11 memory trace that has been semantically pruned
**When** evaluated with LTLZinc temporal constraints
**Then** the `cerce_nonforgetting_rate` must be 1.0.

