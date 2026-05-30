# Constraint Store Capability Spec

## Overview

The constraint store capability provides persistent, retrieval-efficient storage
for EBM constraint embeddings accumulated across verification sessions.

## Requirements

### REQ-STORE-010 — Add constraint

EmbeddingConstraintStore MUST accept a float32 embedding vector and a boolean
label, normalise the vector to unit length, and store both for later retrieval.

### REQ-STORE-011 — Retrieve by cosine similarity

EmbeddingConstraintStore MUST return the top-k most similar stored constraints
for a given query embedding, ranked by cosine similarity (dot product when
stored vectors are unit-length).

### REQ-STORE-020 — K=32 centroid compression

EmbeddingConstraintStore MUST support K=32 centroid compression via
MemoryBankCompressor after 10+ sessions of accumulation, preserving retrieval
AUROC > 0.75 on synthetic benchmarks.

## Scenarios

### SCENARIO-STORE-010 — Single constraint round-trip

Given an embedding and label, add_constraint() stores the embedding and
retrieve() returns a result including the stored label.

### SCENARIO-STORE-020 — Similarity ranking is correct

Given two clusters of embeddings with different labels, retrieve() ranks
same-cluster embeddings higher than cross-cluster embeddings.

### SCENARIO-STORE-030 — Memory compression AUROC

Given a 1000-embedding store (10 sessions, 5 semantic clusters), compressing
to K=32 centroids preserves retrieval AUROC > 0.75 on 50 held-out queries.

## Implementation Status

| REQ | Status | Experiment |
|-----|--------|------------|
| REQ-STORE-010 | IMPLEMENTED | python/carnot/stores/embedding_constraint_store.py |
| REQ-STORE-011 | IMPLEMENTED | python/carnot/stores/embedding_constraint_store.py |
| REQ-STORE-020 | IMPLEMENTED | Exp 865 |

### REQ-CAS-001 — CAS bounded memory update for constraint templates

CASConstraintUpdater MUST apply Compress-Add-Smooth updates to a
ConstraintTemplateLibrary such that no observation count exceeds max_count
after any finite sequence of CAS update steps.

Sub-requirements:
- REQ-CAS-001-1: compress() SHALL multiply every existing observation count by
  compress_factor ∈ (0, 1), producing values in [0, original_count).
- REQ-CAS-001-2: add() SHALL incorporate new (pattern_key, model_id) → count
  observations into the library by calling observe_pattern().
- REQ-CAS-001-3: smooth() SHALL blend each count toward smooth_target and cap
  at max_count, ensuring all counts remain in [0, max_count].
- REQ-CAS-001-4: cas_update() SHALL apply compress → add → smooth in that order
  and return the updated observation mapping.

### SCENARIO-CAS-001 — Decay under repeated compress steps

Given a ConstraintTemplateLibrary with one (pattern_key, model_id) count=100
and a CASConstraintUpdater with compress_factor=0.9 and max_count=200,
after 10 compress() calls the count is below 40 (100 × 0.9^10 ≈ 34.9).

### SCENARIO-CAS-002 — New observation survives CAS update above threshold

Given a ConstraintTemplateLibrary with min_frequency=5 templates and a new
observation batch {("carry_check", "m1"): 10}, after one cas_update() the
carry_check template is active for "m1" because the post-smooth count ≥ 5.

### REQ-CONSTRAIN-001 — Natural language constraint compilation
The system MUST be able to compile natural language rules into executable Python validators using an LLM.

### SCENARIO-CONSTRAIN-001 — Compile and validate
Given a natural language rule, the compiler returns a callable Python validator that correctly evaluates boolean assignments.
