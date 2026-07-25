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

### REQ-STORE-5432 — Ontology-backed constraint-memory fixture V494

Carnot SHALL provide Exp5432 at
`python/carnot/experiment_5432_ontology_softlogic_constraint_memory_v494.py`
and write
`results/experiment_5432_ontology_softlogic_constraint_memory_v494.json`
without modifying `scripts/research_conductor.py`. The fixture SHALL be a
bounded finite-domain workflow with RDF-like triples, typed entities, relation
constraints, deterministic tool-output evidence, and explicit memory-update
records.

The verifier SHALL implement local SHACL-style validation for predicate
domain/range, known entity/type membership, unsupported predicate/type
abstention, and stale relation checks. It SHALL also run deterministic planner/solver checks over the workflow order, prerequisite coverage, evidence
availability, and finite-domain reachability. SHACL and deterministic solver
checks SHALL be the final authority for accept, reject, or abstain decisions.

The fixture rows SHALL include valid triple updates, false triple updates,
stale relation updates, unsupported memory writes, and semantically plausible but infeasible retrievals. Soft-logic residual scores MAY be recorded as
advisory conflict scores and MAY route rows to exact verification, but they
MUST NOT override the deterministic verifier or create a learned logical truth
claim.

The terminal result artifact MUST include bare JSON fields:
`ontology_fixture_count`, `triple_count`, `shacl_validation_pass_rate`,
`deterministic_solver_authority`, `false_triple_rejection_rate`,
`valid_update_preservation_rate`, `unsupported_update_abstention_rate`,
`soft_logic_residuals_recorded`, `soft_logic_overrode_solver`,
`ontology_constraint_memory_ready`, `inference_substrate`, and
`honest_verdict`. `inference_substrate` MUST be
`deterministic_ontology_verifier`; `soft_logic_overrode_solver` MUST be false;
and `honest_verdict` MUST start with `complete:` or `blocked:`.

Field principles:
- `ontology_fixture_count`: coverage.
- `triple_count`: graph scale.
- `shacl_validation_pass_rate`: structural validity.
- `deterministic_solver_authority`: no learned oracle.
- `false_triple_rejection_rate`: safety guard.
- `valid_update_preservation_rate`: no over-rejection.
- `unsupported_update_abstention_rate`: missing-evidence guard.
- `soft_logic_residuals_recorded`: advisory conflict signal.
- `soft_logic_overrode_solver`: final authority boundary.
- `ontology_constraint_memory_ready`: downstream gate.
- `inference_substrate`: no hidden live model inference.
- `honest_verdict`: terminal status; starts with complete: or blocked:.

### SCENARIO-STORE-5432 — Deterministic ontology verifier owns memory decisions

Given the V494 finite-domain workflow fixture with typed entities, RDF-like
triples, tool-output evidence, valid updates, false updates, stale updates,
unsupported writes, and infeasible retrievals,

When Exp5432 evaluates the rows,

Then valid triple updates are preserved, false triples are rejected,
unsupported writes abstain, stale relation updates fail the deterministic
authority check, infeasible retrievals are rejected by the planner/solver, and
soft-logic residuals are recorded only as advisory routing scores.

And if a soft residual is low for an invalid row or high for a valid row,

Then the final decision remains the deterministic verifier decision and the
artifact records `soft_logic_overrode_solver=false`.

### REQ-STORE-5761 — Hash-sealed typed acquisition model variants

Carnot SHALL store Exp5761 typed constraint-acquisition benchmark evidence as
content-addressed model variants derived from sealed Exp5746 receipts. For each
selected source case, the store-level artifact SHALL preserve domain artifact
hashes, faithful model hashes, incomplete model hashes, over-fit model hashes,
mixed model hashes, hard/soft role receipts, positive and negative assignment
hashes, distinguishing query hashes, and expected repair receipt hashes.

The stored variant records SHALL distinguish model AST from model text and
SHALL preserve hard constraints, soft preferences, and soft objective roles
without converting soft preferences into hard rules or treating objectives as
membership authority. Incomplete, over-fit, and mixed mutations SHALL be
accepted into the benchmark store only when exact semantic comparison proves
they are not no-op equivalents of the faithful model.

### SCENARIO-STORE-5761 — Acquisition variant hashes replay

Given Exp5761 has selected a sealed Exp5746 source case,

When the typed acquisition variants are materialized,

Then each faithful, incomplete, over-fit, and mixed variant has a stable model
hash, domain artifact hash, hard/soft role receipt, assignment receipt, query
receipt, and expected repair receipt, and replaying those fields reproduces the
stored hashes exactly.

### REQ-STORE-5762 — Query lifecycle receipt store

Carnot SHALL store Exp5762 query-driven lifecycle evidence as content-addressed
receipts derived from the sealed Exp5761 manifest. The store-level artifact
SHALL preserve benchmark manifest hashes, science split hashes, template
library hashes, membership query hashes, lifecycle state hashes, birth,
refinement, quarantine, supersession, rollback, and restart receipts for every
episode without storing or exposing a faithful target model AST/text to the
learner path.

Stored lifecycle receipts SHALL distinguish observed assignments from oracle
membership answers, candidate constraints from promoted active constraints,
rejected or quarantined candidates from propagated constraints, and protected
prefix replay hashes from science recovery metrics. Producer gate fields for
`constraint_recovery_gain_lcb`, `prefix_retention_pass_score`,
`unsafe_update_count`, and `rollback_hash_mismatch_count` SHALL remain bare
top-level scalars in the result artifact, with principles recorded only in
`field_principles`.

### SCENARIO-STORE-5762 — Lifecycle receipt hashes replay

Given Exp5762 has consumed the sealed Exp5761 acquisition benchmark,

When its query, update, rollback, and restart receipts are replayed,

Then every lifecycle ledger row hash, membership query hash, promoted
constraint hash, quarantined candidate hash, supersession hash, rollback hash,
restart hash, science split hash, and template library hash reproduces exactly
and rejected updates have zero propagation.

### REQ-STORE-5763 — Dependent task lifecycle receipt store

Carnot SHALL store Exp5763 dependent chronological task evidence as
content-addressed receipts derived from the qualified Exp5762 result. The
store-level artifact SHALL preserve upstream artifact hashes, generator
version, dependency DAG hash, stream root hash, row/query label hashes,
operation-order hashes, protected-prefix hashes, held-out composition hashes,
shift, conflict, supersession, delayed-counterexample, crash, corruption,
non-forgetting, restart, and rollback receipts without exposing faithful target
model AST/text to the learner path.

Stored dependent-task receipts SHALL distinguish exact membership-query answers
from observed examples, dependencies from supersessions, conflicts from
quarantines, delayed counterexamples from initial evidence, and rejected or
corrupted updates from propagated active constraints. Restart and rollback
receipts SHALL reproduce exact state hashes, and rejected updates SHALL have
zero propagation.

### SCENARIO-STORE-5763 — Dependent task receipt hashes replay

Given Exp5763 has consumed the qualified Exp5762 query-driven lifecycle result,

When its dependency graph, stream rows, query/update labels, recovery receipts,
rollback receipts, restart receipts, held-out composition suffix, and
non-forgetting certificates are replayed,

Then every recorded hash reproduces exactly, protected prefixes retain exact
membership labels, crash/corruption recovery restores the committed state hash,
and rejected updates have zero propagation.

### REQ-STORE-5924 — Transactional constraint-memory receipt store

Carnot SHALL store Exp5924 transactional constraint-memory V2 evidence as a
versioned operation ledger derived from the admitted Exp5920 event stream. The
store-level artifact SHALL preserve the Exp5920 stream path, stream hash, row
count, prefix-chain receipt, transaction schema version, operation ledger hash,
per-operation previous and resulting state hashes, exact-validator receipt
hashes, active-state hashes, quarantine hashes, supersession receipts,
capacity receipts, rollback receipts, restart receipts, rejected-update
non-propagation receipts, and immutable model-weight hashes.

Stored transaction receipts SHALL distinguish pre-event snapshots from
post-validation writes, proposals from committed proposals, exact validation
from memory similarity, promoted exact facts from quarantined or rejected model
updates, superseded active entries from evicted capacity entries, and rollback
targets from restart checkpoints. Replaying the stored ledger SHALL reproduce
the final state hash exactly; tampering with operation order, validator
authority, prior-state hash, resulting-state hash, stream prefix, or rejected
update propagation SHALL reject without partial state writes.

### SCENARIO-STORE-5924 — Transaction ledger hashes replay

Given Exp5924 has consumed the admitted Exp5920 stream,

When its operation ledger, exact-validator receipts, quarantine records,
supersession records, capacity records, rollback checkpoint, restart checkpoint,
and rejected-update receipts are replayed,

Then every previous-state hash, resulting-state hash, ledger hash, rollback
hash, restart hash, active-memory hash, and quarantine hash reproduces exactly,
protected-prefix labels are retained, rejected updates have zero propagation,
and validator-substitution tampering fails closed.
