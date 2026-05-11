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
