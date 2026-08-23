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

### REQ-STORE-5926 — Adaptive-state ABI v2 parity receipt store

Carnot SHALL store Exp5926 adaptive-state ABI v2 evidence as a versioned
Python/Rust/PyO3 conformance ledger derived from the executed Exp5924
transaction ledger. The store-level artifact SHALL preserve Exp5924 gate replay
hashes, Exp5859 preservation hashes, ABI v2 schema and operation receipts,
per-operation previous and resulting state hashes, payload hashes, proposal
identities, validator receipt hashes, ownership/lifetime receipts,
serialization bytes hashes, crash-prefix recovery receipts, rollback receipts,
fresh-process recovery receipts, tamper rejection receipts, protected-file
hashes, task-owned command receipts, and global failure-delta receipts.

Stored ABI v2 receipts SHALL distinguish snapshot reads from committed writes,
proposals from commits, exact validation from promotion authority, active facts
from quarantined/rejected proposals, superseded entries from evicted entries,
rollback targets from crash-recovery checkpoints, and Python/Rust/PyO3 parity
from historical Exp5859 ABI v1 evidence. Replaying the stored traces SHALL
reproduce byte, state, status, and error parity exactly; tampering with
operation order, expected prior state, resulting state, schema version,
checkpoint bytes, validator receipt, replayed commit state, stale snapshot
state, or released-core lifetime SHALL reject without partial state writes.

### SCENARIO-STORE-5926 — ABI v2 parity hashes replay

Given Exp5926 has consumed Exp5924's executed transaction ledger without
mutating Exp5859,

When its ABI schema, conformance trace manifest, operation receipts,
serialization receipts, rollback receipts, crash-prefix recovery receipts,
fresh-process receipts, and tamper rejection receipts are replayed,

Then Python, Rust, and PyO3 produce identical canonical bytes, state hashes,
status values, and error codes; rejected updates have zero propagation;
released cores reject use-after-release and double release; and stale,
replayed, corrupt, or tampered operations fail closed.

### REQ-STORE-5967 — Delayed-commit memory fixture receipt store

Carnot SHALL store Exp5967 delayed-commit memory evidence as a versioned
operation receipt store derived only from ready Exp5920, Exp5924, and Exp5926
receipts. The store-level artifact SHALL preserve upstream replay hashes and
readiness, delayed-commit state-machine schema, frozen base-version snapshot
receipts, proposal sealing receipts, label-reveal timing receipts,
post-event exact-validation receipts, delayed commit receipts, matched
write-through control receipts, quarantine, supersede, rollback, crash replay,
bounded-state, rejected-update non-propagation, Python/Rust/PyO3 trace parity,
fixed-width operation trace hash, immutable model-weight hash, protected-file
hash, task-command receipts, and reproducibility checksum.

Stored delayed-commit receipts SHALL distinguish immutable read snapshots from
pending proposals, sealed proposals from exact validation, exact validation
from readable commit, production delayed writes from same-event write-through
control writes, quarantined or rejected proposals from active state,
superseded entries from capacity evictions, rollback targets from crash replay
checkpoints, and backend-neutral trace portability from hardware execution.
Replaying the stored trace SHALL reproduce final state hashes exactly; tamper,
stale-base, duplicate-event, interrupted, reordered, corrupted, or
write-through-as-production transitions SHALL reject without partial mutation.

### SCENARIO-STORE-5967 — Delayed-commit receipt hashes replay

Given Exp5967 has consumed ready Exp5920, Exp5924, and Exp5926 receipts,

When its delayed-commit ledger, frozen snapshots, proposal receipts,
validation receipts, commit receipts, write-through control trace,
quarantine/supersede/rollback receipts, crash replay checkpoints, bounded
state receipts, fixed-width trace, and backend parity receipts are replayed,

Then every base-version hash, proposal hash, validation receipt hash,
commit-result hash, rollback hash, crash-replay hash, final state hash,
fixed-width trace hash, and Python/Rust/PyO3 return value reproduces exactly;
rejected updates have bare zero propagation; and stale, duplicate, tampered,
or corrupted transitions fail closed.

### REQ-STORE-6521 — Transactional exact-conflict memory

Carnot SHALL provide Exp6521 at
`python/carnot/experiment_6521_transactional_refinement_conflict_memory.py`
and write
`results/experiment_6521_transactional_refinement_conflict_memory.json`.
The mechanism SHALL persist exact conflict records only when the source query,
target query, solver identity, refinement witness, and exact replay receipt all
verify locally. The durable-write gate SHALL reject unrelated, relaxed,
schema-mismatched, solver-mismatched, stale, malformed, duplicate-conflicting,
and invalid-replay records before insertion or use.

Each conflict record SHALL include a source query hash, clause or constraint
payload, solver and solver-version hash, refinement witness, exact replay
receipt, lifecycle state, use count, benefit fields, and content hash. The
memory controller SHALL implement prepare, validate, commit, abort, load, use,
evict, checkpoint, restart, rollback, and quarantine. Commits SHALL use the
Exp6514 atomic shard transaction where durable writes leave the in-memory
process. Capacity SHALL be bounded. Eviction SHALL be deterministic by lowest
benefit, lowest use count, oldest committed version, and content hash.

The supported refinement relation SHALL be local and exact. A target query is
a safe refinement only when it preserves the source clauses, adds clauses, and
keeps the same variable domain, schema version, and solver hash. Relaxed
queries and unrelated queries are not refinements. Every use SHALL replay the
stored conflict against the target query with the exact verifier before the
conflict can prune or block work. If the memory file is missing, unreadable, or
corrupt, the controller SHALL quarantine bad bytes when possible and fall back
to native exact solving without using memory.

The artifact SHALL report pilot-audit gate path and hash, solver capabilities,
available refinement relations, resources, protected-file hashes, lifecycle
rows, valid reuse rows, invalid-reuse veto rows, capacity and eviction rows,
restart and rollback rows, corruption quarantine rows, native fallback rows,
fixed-width CPU mapping rows, gate checks, per-unit rows, aggregate
recomputation, tests run, and a reproducibility checksum. Fixed-width CPU
mapping rows SHALL report logical bytes, mapped bytes, topology expansion,
mapping time, and unsupported fields. They SHALL make no hardware execution or
acceleration claim.

Required artifact fields and principles:

- `status`: Records the terminal exact-conflict memory state.
- `honest_verdict`: States exact safety readiness without claiming routing speed or learning benefit.
- `verdict_class`: Uses circular_positive only for exact mechanism readiness.
- `upstream_gate_receipt`: Binds the run to the independent pilot-audit gate path and hash.
- `conflict_record_schema`: Defines the durable conflict fields and content hash.
- `refinement_relation_contract`: Defines the only supported local safe-reuse relation.
- `lifecycle_rows`: Shows prepare, validate, commit, abort, load, use, checkpoint, and rollback.
- `valid_reuse_rows`: Shows exact replay before each accepted reuse.
- `invalid_reuse_veto_rows`: Shows unsafe candidates were rejected before write or use.
- `capacity_and_eviction_rows`: Shows bounded capacity and deterministic eviction order.
- `restart_rollback_rows`: Shows restart parity and rollback hash restoration.
- `corruption_quarantine_rows`: Shows corrupt durable bytes are moved out of the active path.
- `native_fallback_rows`: Shows exact native solving continues when memory is unavailable.
- `fixed_width_mapping_rows`: Reports CPU mapping cost without a hardware claim.
- `conflict_memory_controller_ready_score`: A conjunctive score opens only with zero unsafe admission and zero unsafe use.
- `gate_check_summary`: Names each gate, expected value, observed value, and failure.
- `per_unit_rows`: Combines lifecycle, safety, mapping, fallback, and attack rows.
- `aggregate_row_recomputation`: Recomputes readiness from rows rather than summary text.
- `preconditions_checked`: Records solver capability, resources, relation, and protected hashes.
- `protected_files_unchanged`: Proves protected upstream files did not change during the run.
- `inference_substrate`: Declares exact local memory and CPU mapping with no LLM.
- `verifier_is_oracle`: Exact replay is authoritative only inside the declared finite domain.
- `field_principles`: Preserves why each artifact field exists.
- `field_provenance`: Maps each field to gates, rows, exact replay, transactions, or tests.
- `random_seed`: Fixes the deterministic scenario order.
- `duration_s`: Records measured wall-clock duration.
- `tests_run`: Records validation commands and exit codes.
- `reproducibility_checksum`: Detects later drift in rows, gates, code, or hashes.

`conflict_memory_controller_ready_score` SHALL be bare `1.0` only when all
lifecycle and safety attack rows pass with zero unsafe admission and zero unsafe
use. `verdict_class` SHALL be `circular_positive` at most for exact safety and
mechanism readiness, `partial` for a bounded mechanism, `blocked` for a missing
gate or solver capability, or `disqualified` for unsafe reuse. The artifact
SHALL set `inference_substrate` to
`transactional_exact_conflict_memory_and_cpu_mapping_no_llm` and
`verifier_is_oracle` to bare `true` for exact conflict validity.

### SCENARIO-STORE-6521-VALID-REUSE — Refinement witness admits exact reuse

Given a committed conflict record for an unsatisfiable source query,

When the target query preserves all source clauses, adds only stronger clauses,
keeps the same schema, variable domain, solver hash, and replay receipt, and
the exact verifier confirms the conflict remains valid,

Then prepare, validate, commit, load, and use succeed; the use count increases;
the replay receipt hash reproduces; and the artifact records zero unsafe use.

### SCENARIO-STORE-6521-INVALID-VETO — Unsafe records never write or use

Given candidate conflict records with unrelated clauses, relaxed clauses,
schema mismatch, solver mismatch, stale source hash, malformed content,
duplicate conflicting payloads, or invalid exact replay,

When prepare, validate, commit, load, or use evaluates those records,

Then each unsafe record is rejected or quarantined before durable insertion or
memory-assisted use, and native exact solving remains available.

### SCENARIO-STORE-6521-LIFECYCLE — Transaction, restart, rollback, corruption

Given a bounded exact-conflict memory with a committed checkpoint,

When commits are interrupted, records exceed capacity, ties require eviction,
the process restarts, rollback targets a prior checkpoint, or the memory file is
corrupted,

Then committed state is deterministic, interrupted writes do not publish partial
state, restart reproduces the prior hash, rollback restores the target hash,
corrupt bytes are quarantined, and fallback solving does not read quarantined
memory.

### SCENARIO-STORE-6521-FIXED-WIDTH-MAPPING — CPU mapping is accounting only

Given committed conflict records with supported fixed-width fields,

When the CPU reference mapper encodes them,

Then logical bytes, mapped bytes, topology expansion, mapping time, unsupported
fields, and mapping hashes are reported deterministically, with no hardware
execution, speed, power, or acceleration claim.

### REQ-STORE-6522 — Chronological exact-conflict self-learning

Carnot SHALL provide Exp6522 at
`python/carnot/experiment_6522_chronological_conflict_self_learning.py`
and write
`results/experiment_6522_chronological_conflict_self_learning.json`.
The experiment SHALL compare scratch solving, frozen empty memory, valid
unbounded reuse, valid bounded reuse with deterministic eviction, restart,
rollback, and invalid-reuse attack arms on one sealed chronological stream.
The stream SHALL contain refinement chains, unrelated queries, recurrence after
gaps, distribution shifts, corruption injections, and held-future suffixes.
All stream rows and thresholds SHALL be committed before any held-future
outcome is scored.

Each arm SHALL receive the same solver, query order, reuse opportunity, and
charged lookup plus mapping budget. Learning arms SHALL record propose,
validate, commit, use, abstain, evict, rollback, quarantine, and fallback
actions. Every event SHALL record the store hash before and after the action.
Every accepted use SHALL re-run the Exp6521 refinement witness and exact replay
before it can affect effort. Invalid reuse SHALL be vetoed before durable write
or use.

The artifact SHALL report controller gate path and hash, expected and observed
gate values, solver versions, resources, stream commitment, protected-file
hashes, prior failure receipts, per-game results, lifecycle action rows, store
hash rows, exact answer equality rows, immediate metrics, old-prefix retention,
held-future support, interference, capacity, restart, rollback, invalid-reuse
attacks, sequential evidence, per-unit rows, aggregate recomputation, tests run,
and a reproducibility checksum.

Required artifact fields and principles:

- `status`: Records whether the chronological conflict-learning comparison is positive, null, partial, blocked, or disqualified.
- `honest_verdict`: States the measured learning result and the exact-safety limits.
- `verdict_class`: Uses positive only for charged held-future benefit with exact safety.
- `upstream_gate_receipt`: Binds the run to the Exp6521 controller gate path, hash, and expected value.
- `prior_failure_receipts`: Records why Exp6496 and Exp6498 did not open a held-future learning claim.
- `chronological_stream_commitment`: Freezes the stream, thresholds, and held-future boundary before scoring.
- `arm_and_dose_contract`: Shows each arm got matched solver, query, opportunity, and charged budget.
- `per_game_results`: Reports one row per chronological unit and arm.
- `lifecycle_action_rows`: Records propose, validate, commit, use, abstain, evict, rollback, quarantine, and fallback.
- `store_hash_rows`: Records before and after store hashes for every event.
- `exact_answer_equality_rows`: Proves every arm matches the exact release solver.
- `immediate_metric_rows`: Measures current-query utility after charged lookup and mapping cost.
- `prefix_retention_rows`: Measures old-prefix support after the full stream.
- `held_future_support_rows`: Measures charged held-future support and benefit by chain.
- `interference_rows`: Measures unrelated-query abstention, safety, and extra charged cost.
- `capacity_restart_rollback_rows`: Records eviction, restart parity, and rollback parity.
- `invalid_reuse_attack_rows`: Shows unsafe reuse, leakage, and hidden-validation attacks were vetoed.
- `sequential_evidence`: Proves decisions use only prior store state and sealed thresholds.
- `csl_execution_complete_score`: Bare scalar that is one only when all planned rows are terminal.
- `continuous_self_learning_candidate_score`: Bare scalar that is one only for exact safe positive held-future benefit.
- `gate_check_summary`: Names expected and observed gate values plus failed checks.
- `per_unit_rows`: Combines all event and metric rows with source groups.
- `aggregate_row_recomputation`: Recomputes all scores from rows rather than prose.
- `preconditions_checked`: Records solver versions, resources, stream commitment, and protected hashes.
- `protected_files_unchanged`: Proves protected upstream files did not change.
- `inference_substrate`: Declares chronological exact conflict memory with no LLM.
- `verifier_is_oracle`: Bare false because the learning-benefit metric is not an oracle.
- `exact_solver_is_release_authority`: Bare true because exact answer equality is judged by the release solver.
- `field_principles`: Preserves why each artifact field exists.
- `field_provenance`: Maps each field to gates, rows, exact replay, or tests.
- `random_seed`: Fixes stream and arm order.
- `duration_s`: Records measured wall-clock duration.
- `tests_run`: Records validation commands and exit codes.
- `reproducibility_checksum`: Detects later drift in rows, gates, code, or hashes.

`csl_execution_complete_score` SHALL be bare `1.0` only when every planned row
is terminal. `continuous_self_learning_candidate_score` SHALL be bare `1.0`
only when there are zero unsafe writes, zero unsafe uses, exact answer equality,
positive charged held-future benefit, old-prefix retention within margin,
preserved support, and benefit beyond scratch and frozen controls.
`verdict_class` SHALL be `positive` only for oracle-distinct charged
held-future benefit with exact safety, `null` for a complete no-benefit result,
`partial` for incomplete usable evidence, `blocked` for gate or precondition
failure, or `disqualified` for unsafe reuse, leakage, or correctness drift. The
artifact SHALL set `inference_substrate` to
`chronological_exact_conflict_memory_self_learning_no_llm`,
`verifier_is_oracle` to bare `false`, and SHALL record exact solver release
authority as true.

### SCENARIO-STORE-6522-SEALING — Stream and thresholds are frozen first

Given a planned chronological stream with prefix and held-future rows,

When Exp6522 starts,

Then the stream commitment, held-future boundary, opportunity budget, and
thresholds are hashed before scoring, and no event row reads a later outcome.

### SCENARIO-STORE-6522-MATCHED-DOSE — Arms get equal opportunity and cost

Given scratch, frozen, unbounded, bounded, restart, rollback, and attack arms,

When the comparison runs,

Then each arm uses the same query order, solver identity, opportunity count,
lookup charge, mapping charge, and exact-release answer check.

### SCENARIO-STORE-6522-LEARNING-ACTIONS — Online memory changes are visible

Given a source query whose conflict is exact and a later safe refinement,

When a learning arm reaches those events,

Then propose, validate, commit, use, abstain, evict, rollback, quarantine, and
fallback actions are recorded with before and after store hashes.

### SCENARIO-STORE-6522-FUTURE-SUPPORT — Held future benefit is charged

Given held-future refinement targets after chronological gaps,

When valid conflict memory is available,

Then charged held-future effort is lower than scratch and frozen controls on
more than one chain, while exact answers remain equal.

### SCENARIO-STORE-6522-PREFIX-RETENTION — Old prefix support remains

Given bounded memory after the full stream,

When protected prefix targets are replayed,

Then support stays within the frozen margin and retained rows use exact replay.

### SCENARIO-STORE-6522-SAFETY — Invalid reuse is vetoed

Given relaxed, unrelated, schema-mismatched, corrupted, replay-leaky,
future-aware-eviction, hidden-validation, unsafe-unrelated, restart-drift,
rollback-drift, support-collapse, one-chain, and aggregate-only attacks,

When the attack arm evaluates them,

Then each attack fails closed with zero durable unsafe writes, zero unsafe uses,
and native exact fallback still available.

### SCENARIO-STORE-6522-RESTART-ROLLBACK-CAPACITY — State changes are deterministic

Given bounded capacity, a checkpoint, a process restart, and a rollback target,

When Exp6522 exceeds capacity, reloads state, and rolls back,

Then eviction order is deterministic, restart state hash matches, and rollback
restores the checkpoint hash.

### SCENARIO-STORE-6522-SEQUENTIAL-EVIDENCE — Summary claims replay from rows

Given lifecycle, metric, safety, support, and equality rows,

When aggregate scores are recomputed,

Then terminal scores and verdict class match the row-derived checks and cannot
come only from aggregate prose.
