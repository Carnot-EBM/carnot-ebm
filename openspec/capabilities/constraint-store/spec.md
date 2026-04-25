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
