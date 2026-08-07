# Continuous Learning Capability Specification

**Capability:** continuous-learning
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-11, research-program.md continuous self-learning mandate

## Overview

This capability covers Carnot experiments that learn persistent, bounded,
externally certified strategy state from prior verified outcomes while keeping
model weights immutable.  The 20260806 Exp6164 task is intentionally
fail-closed: it must always write a terminal artifact, but it may load models
only after the Exp6162 decision-admission and Exp6163 strategy-store
prerequisites recompute ready inside the experiment.

## REQ-CL-6164-MANDATORY-ARTIFACT: Exp6164 Always Writes the Terminal Artifact

**Given** the mandatory continuous self-learning task for run date 20260806
**When** Exp6164 starts
**Then** it SHALL always write
`results/experiment_6164_continuous_strategy_learning_ab.json`
**And** the artifact SHALL contain bare `true` values for
`continuous_self_learning_task` and `mandatory_artifact_written`.

## REQ-CL-6164-PREREQUISITE-RECOMPUTE: Exp6162 and Exp6163 Gates Are Internal

**Given** cached Exp6162 and Exp6163 artifacts
**When** Exp6164 evaluates readiness
**Then** it SHALL recompute the Exp6162 policy/verdict gate and the Exp6163
schema/ABI/verdict gate before acquiring GPU leases, resolving model paths, or
loading any tokenizer or model.

## REQ-CL-6164-BLOCKED-MODEL-LOAD: Failed Prerequisites Block Before Runtime

**Given** either prerequisite fails or is absent
**When** Exp6164 writes its terminal artifact
**Then** `honest_verdict` SHALL start with `blocked:`
**And** `blocked_before_model_load_receipt` SHALL prove that model, tokenizer,
loader, native-chat, CUDA, GPU-worker, and generated-token invocation counts are
all zero.

## REQ-CL-6164-MANDATED-MODEL: Frozen SOTA GGUF Pair

**Given** both prerequisites pass
**When** Exp6164 runs model-backed arms
**Then** top-level `MODEL_SPECS` SHALL contain exactly
`unsloth/Qwen3.6-35B-A3B-GGUF` as the primary model and
`unsloth/gemma-4-26B-A4B-it-GGUF` as the confirmation model
**And** each record SHALL include resolved path, revision, quantization, hash,
loader, GPU assignment, embedded tokenizer, native-chat, CUDA PID/lifecycle, and
before/after immutable-weight receipts.

## REQ-CL-6164-FOUR-ARM-MATCHING: Resource-Matched Chronological A/B

**Given** the mandated model pair
**When** prerequisites pass
**Then** Exp6164 SHALL compare four arms: no memory, Exp6120 utility-only
memory, certificate-only certified strategy memory, and Exp6162
decision-calibrated certified strategy memory
**And** prompts, chronological event order, seeds, token budgets, invocation
counts, wall caps, and resource caps SHALL match across arms.

## REQ-CL-6164-CHRONOLOGICAL-ISOLATION: No Future Leakage

**Given** an event at chronological index N
**When** an arm makes its decision
**Then** the arm SHALL read a frozen decision snapshot, retrieve only
certificates from events `< N`, and forbid current-label visibility,
same-decision writes, label-conditioned retries, prompt adaptation, and weight
mutation.

## REQ-CL-6164-READ-ONLY-SNAPSHOT: Decisions Cannot Mutate Pre-Outcome State

**Given** a decision snapshot
**When** the model produces a decision
**Then** the snapshot SHALL remain read-only until exact post-outcome
validation decides whether a strategy update can commit.

## REQ-CL-6164-POST-OUTCOME-COMMIT: Exact Transaction Outcomes

**Given** a candidate strategy update
**When** exact outcome validation succeeds
**Then** the update SHALL commit after the decision
**And** failed or unsafe validations SHALL abort or quarantine rather than enter
the certified store.

## REQ-CL-6164-CERTIFICATE: Certified Strategy State Is Bounded and Conservative

**Given** the certificate-only strategy arm
**When** a strategy is admitted
**Then** the update SHALL carry a certificate-only conservative admission
receipt and SHALL respect bounded-state eviction limits.

## REQ-CL-6164-DECISION-ADMISSION: Exp6162 Policy Controls the Calibrated Arm

**Given** the decision-calibrated strategy arm
**When** a strategy is admitted
**Then** the arm SHALL use the frozen Exp6162 decision-calibrated admission
policy and SHALL NOT refit thresholds, selectors, prompts, scores, or labels.

## REQ-CL-6164-UTILITY: Future Utility Is Per-Model Before Pooling

**Given** the four arms complete
**When** Exp6164 reports utility
**Then** it SHALL report future utility, accuracy, regret, grouped intervals,
chronological learning curves, and time-to-benefit per model, family, and
partition before any pooled summary.

## REQ-CL-6164-RETENTION: Protected Retention Cannot Regress

**Given** the decision-calibrated strategy arm improves utility
**When** readiness is computed
**Then** protected retention and forgetting metrics SHALL show no protected
regression for either mandated model.

## REQ-CL-6164-POISON: Unsafe and Poison Updates Cannot Propagate

**Given** unsafe, poison, duplicate, or reordered delivery events
**When** the strategy store processes them
**Then** unsafe admission, poison propagation, and abstention regressions SHALL
remain zero for readiness.

## REQ-CL-6164-ROLLBACK: Duplicate, Reorder, Restart, and Rollback Are Idempotent

**Given** duplicate delivery, reordered delivery, rollback, or restart
**When** Exp6164 replays state transitions
**Then** the resulting state and receipts SHALL be idempotent.

## REQ-CL-6164-BOUNDED-STATE: Certified State Bytes Are Capped

**Given** accepted strategy updates accumulate
**When** state size reaches the configured bound
**Then** eviction SHALL keep `max_state_bytes <= state_byte_bound`.

## REQ-CL-6164-LIFECYCLE: Runtime Durations and Cleanup Are Separated

**Given** a qualified model-backed run
**When** Exp6164 reports lifecycle receipts
**Then** GPU acquisition, live inference, cached analysis, and cleanup durations
SHALL be reported separately, and all task-owned workers/CUDA contexts SHALL be
released.

## REQ-CL-6164-IMMUTABLE-WEIGHT: Model Weights Are Never Updated

**Given** certified external strategy state changes
**When** Exp6164 verifies model files and runtime fingerprints before and after
the run
**Then** all model-weight fingerprints SHALL remain unchanged and
`weight_update_count` SHALL be zero.

## REQ-CL-6164-READY-SCORE: Positive Readiness Is Fully Conjunctive

**Given** Exp6164 computes `continuous_strategy_learning_ready_score`
**When** the score is one
**Then** the decision-calibrated arm SHALL beat no-memory and utility-only arms
for both mandated models with positive lower confidence intervals, no protected
or safety regression, bounded certified state, lifecycle cleanup, and immutable
weights
**And** pooled success SHALL NOT mask either model.

## SCENARIO-CL-6164-BLOCKED: Missing Exp6163 Blocks Before Model Load

**Given** Exp6162 is ready and
`results/experiment_6163_certified_strategy_store_scaleup.json` is missing
**When** Exp6164 runs
**Then** it writes a `blocked:` artifact with zero model, tokenizer, CUDA, GPU,
native-chat, and generated-token invocation counts.

## SCENARIO-CL-6164-MATCHED: Qualified Run Uses Four Matched Arms

**Given** Exp6162 and Exp6163 both recompute ready
**When** Exp6164 runs a model-backed A/B
**Then** every model receives the same prompts, events, seeds, token budgets,
invocation counts, and wall/resource caps for all four arms.

## SCENARIO-CL-6164-TRANSACTION: Strategy Learning Is Outcome Committed

**Given** a decision at chronological event N
**When** a strategy update is proposed
**Then** only certificates from events `< N` are visible before the decision,
and the update commits only after exact outcome validation.

## SCENARIO-CL-6164-READY: Per-Model Gates Control the Positive Verdict

**Given** pooled utility improves but either mandated model has a non-positive
lower confidence bound, safety regression, unbounded state, failed cleanup, or
mutable-weight receipt
**When** readiness is computed
**Then** `continuous_strategy_learning_ready_score` SHALL be `0.0` and
`honest_verdict` SHALL start with `complete_null:` or `retired:`.

## Required Artifact Fields and Principles

- `status`: Terminal status distinguishes blocked, positive, null, and retired strategy-learning evidence.
- `preconditions_checked`: Hash Exp6160 rows, Exp6162 policy/verdict, Exp6163 schema/ABI/verdict, event order, prompts, models, validators, exclusions, outputs, and protected files before GPU acquisition.
- `continuous_self_learning_task`: This field is bare true because the task is the mandatory FR-11 continuous self-learning run.
- `mandatory_artifact_written`: This field is bare true because every terminal path must write the Exp6164 artifact.
- `prerequisite_gate_receipts`: Exp6162 and Exp6163 readiness are recomputed internally and conjunctively.
- `blocked_before_model_load_receipt`: A failed prerequisite must prove all model, tokenizer, CUDA, and GPU invocation counts are zero.
- `MODEL_SPECS`: The top-level model list contains only the two mandated frozen SOTA GGUF hub ids.
- `model_specs`: The lowercase model list mirrors MODEL_SPECS for downstream schema consumers.
- `resolved_paths_revisions_quantizations_hashes_and_loader_receipts`: Model paths, revisions, quantizations, hashes, loaders, and GPU assignments are resolved only after prerequisites pass.
- `embedded_tokenizer_chat_template_cuda_pid_and_lifecycle_receipts`: Tokenizer, chat-template, CUDA PID, native-chat, worker, and lifecycle receipts distinguish cached setup from live inference.
- `arm_definitions_and_resource_matching`: No-memory, Exp6120 utility-only, certificate-only, and decision-calibrated arms are matched on events, prompts, seeds, token budgets, invocations, and resource caps.
- `chronological_event_order_and_decision_snapshot_receipts`: Every decision reads a frozen read-only snapshot with only prior certificates.
- `exact_post_outcome_commit_abort_quarantine_receipts`: Strategy writes commit only after exact post-outcome validation, otherwise abort or quarantine.
- `per_model_family_partition_future_utility_accuracy_regret_and_grouped_intervals`: Future utility, accuracy, regret, and grouped intervals are reported per model, family, and partition before pooling.
- `learning_speed_and_time_to_benefit`: Chronological learning curves and time-to-benefit are separated from final utility.
- `protected_retention_forgetting_safety_abstention_and_poison_metrics`: Utility cannot buy protected forgetting, unsafe admission, abstention, or poison regressions.
- `duplicate_reordered_rollback_restart_eviction_and_state_bytes`: Duplicate, reordered, rollback, restart, eviction, and bounded-state bytes are explicit lifecycle checks.
- `model_weight_immutability_receipt`: This experiment may update certified external strategy state but never model weights.
- `acquisition_analysis_duration_and_cleanup_receipts`: GPU acquisition, live inference, cached analysis, and cleanup durations are reported separately.
- `continuous_strategy_learning_ready_score`: Readiness is one only when the decision-calibrated strategy beats both baselines for both models with positive lower intervals, no regressions, bounded state, cleanup, and immutable weights.
- `retirement_triggered`: Repeated non-positive strategy-learning evidence can retire the construction instead of hiding a null.
- `protected_files_unchanged`: Conductor, ops, and traceability files remain outside this experiment's mutable surface.
- `duration_s`: Measured wall time is reported without classifying cached analysis as live inference.
- `inference_substrate`: The substrate states whether the run blocked before load or used live local SOTA GGUF CUDA.
- `verifier_is_oracle`: Exact validators score post-outcome commits, but the decision policy is not an oracle.
- `missing_verifier_gaps`: Any missing prerequisite, model, lifecycle, safety, or validation gap is made explicit.
- `field_provenance`: Every field traces to spec, upstream artifacts, model receipts, transaction receipts, tests, commands, or protected-file hashes.
- `test_commands`: Commands document focused unit, coverage, prerequisite, artifact, model/cache/tokenizer/CUDA, arm matching, chronological isolation, transaction, metrics, immutability, lifecycle, schema, adversarial, protected-file, E2E, global pytest, and root-clutter checks.
- `test_exit_codes`: Non-zero verification commands prevent readiness.
- `reproducibility_checksum`: A checksum detects drift in inputs, model specs, receipts, metrics, commands, protected files, and output paths.
- `honest_verdict`: Use `complete_positive:`, `complete_null:`, `retired:`, or `blocked:` and state whether self-learning actually executed.

## Implementation Status

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6164-MANDATORY-ARTIFACT | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-PREREQUISITE-RECOMPUTE | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-BLOCKED-MODEL-LOAD | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-MANDATED-MODEL | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-FOUR-ARM-MATCHING | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-CHRONOLOGICAL-ISOLATION | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-READ-ONLY-SNAPSHOT | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-POST-OUTCOME-COMMIT | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-CERTIFICATE | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-DECISION-ADMISSION | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-UTILITY | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-RETENTION | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-POISON | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-ROLLBACK | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-BOUNDED-STATE | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-LIFECYCLE | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-IMMUTABLE-WEIGHT | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |
| REQ-CL-6164-READY-SCORE | Implemented | tests/python/test_experiment_6164_continuous_strategy_learning_ab.py |

## REQ-CL-6179-MANDATORY-EXECUTION: Retention-Safe Continuous Strategy Learning A/B

**Given** the mandatory ungated continuous strategy-learning task for run date
20260807
**When** Exp6179 starts
**Then** it SHALL always write
`results/experiment_6179_retention_safe_continuous_strategy_learning_ab.json`
**And** the artifact SHALL contain bare `true` values for
`continuous_self_learning_task` and `mandatory_artifact_written`.

## REQ-CL-6179-LOCAL-GGUF: Frozen Local GGUF Identity

**Given** the mandated public model pair
**When** Exp6179 snapshots preconditions
**Then** it SHALL record local cache paths, revisions, quantizations, sizes,
and cache checksums for exactly `unsloth/Qwen3.6-35B-A3B-GGUF` and
`unsloth/gemma-4-26B-A4B-it-GGUF`
**And** tiny legacy models SHALL NOT satisfy the model identity receipt.

## REQ-CL-6179-IMMUTABLE-WEIGHTS: Strategy Learning Cannot Mutate Weights

**Given** strategy state changes during the A/B
**When** Exp6179 compares before and after model receipts
**Then** all model-weight fingerprints SHALL remain unchanged and
`weight_update_count` SHALL be zero.

## REQ-CL-6179-EXTERNAL-MEMORY: Task-Owned Memory Boundary

**Given** model weights are immutable
**When** Exp6179 stores learned strategy information
**Then** all mutable state SHALL live only in task-owned external memory paths
declared by the artifact for the sealed stream, bounded store, replay ledger,
rollback ledger, and quarantine ledger.

## REQ-CL-6179-POST-OUTCOME-WRITE: Exact Outcomes Gate Every Commit

**Given** a model decision at chronological event N
**When** an update is considered for admission
**Then** no same-decision write SHALL be visible before the decision
**And** every committed strategy record SHALL reference an exact post-outcome
receipt from event N before it becomes retrievable for event N+1.

## REQ-CL-6179-BOUNDED-REPLAY: Replay Is Chronological and State-Bounded

**Given** accepted strategy records accumulate
**When** the replay arm retrieves memory
**Then** it SHALL retrieve only prior chronological records, respect the
configured replay window and state byte bound, and evict only according to a
deterministic protected-family-preserving policy.

## REQ-CL-6179-RETENTION: Prior-Family Retention Is Measured After Every Update

**Given** an update is admitted into any mutable memory arm
**When** the commit completes
**Then** Exp6179 SHALL immediately measure protected and prior-family
retention against all families present before the update
**And** positive utility SHALL NOT produce readiness if the selected replay arm
forgets a prior or protected family.

## REQ-CL-6179-POISON-QUARANTINE: Poisoned Updates Fail Closed

**Given** poisoned, malformed, contradictory, duplicate, or reordered update
events
**When** the bounded strategy store processes them
**Then** unsafe updates SHALL be rejected or quarantined, poison propagation
SHALL be zero, and quarantine precision and recall SHALL be reported.

## REQ-CL-6179-ROLLBACK: Rollback Restores Exact State Hashes

**Given** a poisoned update or an explicit rollback request
**When** rollback executes
**Then** the restored store hash SHALL exactly match the referenced prior
snapshot hash and rollback attempts past the sealed root SHALL fail closed.

## REQ-CL-6179-PROTECTED-FILES: Experiment Scope Is Narrow

**Given** Exp6179 mutates only task-owned result artifacts
**When** it finishes
**Then** `scripts/research_conductor.py`, `ops/changelog.md`,
`ops/status.md`, and `_bmad/traceability.md` SHALL remain byte-identical.

## REQ-CL-6179-ARMS: Five Matched Memory Arms

**Given** the sealed chronological stream
**When** Exp6179 runs the A/B
**Then** it SHALL compare exactly five resource-matched arms:
`no_memory`, `fixed_memory`, `write_through`, `replay`, and
`shuffled_retrieval`
**And** each arm SHALL share the same model IDs, event order, prompts, seeds,
token budgets, and external-memory byte budget.

## REQ-CL-6179-RECEIPTS: Required Artifact Fields and Principles

- `status`: Terminal state follows sealed stream, model-cache, utility, retention, quarantine, rollback, protected-file, and test receipts.
- `preconditions_checked`: Snapshots model caches, stream/store paths, retention families, poisoning controls, protected files, root clutter, and git status before mutation.
- `continuous_self_learning_task`: Bare true marks this as the mandatory continuous self-learning task.
- `mandatory_artifact_written`: Bare true records that the terminal artifact was written.
- `MODEL_SPECS`: The top-level model list contains exactly the two mandated frozen local GGUF hub ids.
- `model_specs`: The lowercase model list mirrors `MODEL_SPECS` for downstream consumers.
- `sealed_chronological_stream_receipt`: Event order, stream hash chain, and no-future-label controls seal the stream.
- `task_owned_external_memory_receipt`: All mutable strategy state is confined to task-owned external-memory paths.
- `arm_definitions_and_resource_matching`: The five arms share event order, prompts, seeds, model IDs, token budgets, and memory bounds.
- `exact_post_outcome_write_receipts`: Commits occur only after exact outcomes and no same-decision write is visible.
- `utility_by_arm_family_and_model`: Utility, accuracy, regret, and intervals are reported by model, arm, and family before pooling.
- `prior_family_retention_after_every_update`: Protected and prior-family retention are measured immediately after every admitted update.
- `bounded_strategy_store_receipt`: State size, replay window, protected prefix, eviction, and checksum receipts bound the store.
- `rollback_and_quarantine_receipts`: Rollback exactness, fail-closed rollback, poison quarantine, and duplicate/reorder controls are auditable.
- `state_bound_receipt`: Runtime state remains within the configured byte and record bounds.
- `model_weight_immutability_receipt`: Weight fingerprints remain unchanged and weight update count is zero.
- `provenance_receipts`: Decisions, updates, outcomes, and quarantines trace to sealed event IDs and hashes.
- `protected_files_unchanged`: Protected repository files remain byte-identical.
- `duration_s`: Wall-clock experiment duration is recorded.
- `inference_substrate`: The substrate states frozen local GGUF identity plus task-owned external memory.
- `retention_safe_continuous_strategy_learning_ready_score`: Readiness is one only when replay beats all controls without prior-family forgetting, poison propagation, rollback failure, state overflow, weight mutation, protected-file mutation, or test failure.
- `missing_verifier_gaps`: Any model-cache, utility, retention, safety, rollback, state, protected-file, or test gap is explicit.
- `field_provenance`: Every required field traces to a requirement, receipt, checksum, test, or protected-file hash.
- `test_commands`: Focused, coverage, schema, spec-coverage, adversarial, protected-file, root-clutter, and full-suite commands are listed.
- `test_exit_codes`: Exit codes prevent failed checks from being reported as success.
- `checksum_receipts`: Stream, store, model-cache, protected-file, sidecar, and artifact checksum inputs are recorded.
- `reproducibility_checksum`: The artifact checksum detects drift excluding the checksum field itself.
- `honest_verdict`: The verdict starts with `complete:`, `complete_null:`, or `blocked:` and states whether live model generation occurred.

## SCENARIO-CL-6179-SEALED-ARMS: Stream and Arms Are Matched

**Given** Exp6179 builds its chronological stream
**When** five arms execute
**Then** every arm reads the same sealed event order and resource signature,
and only task-owned external memory can change.

## SCENARIO-CL-6179-RETENTION-AFTER-UPDATE: Forgetting Cannot Hide Behind Utility

**Given** the replay arm has positive utility
**When** any admitted update lowers prior-family retention below the configured
floor
**Then** readiness is zero and `missing_verifier_gaps` includes a retention
failure.

## SCENARIO-CL-6179-POISON-ROLLBACK: Poison Quarantine and Rollback Are Exact

**Given** poisoned and invalid update events are present in the stream
**When** the bounded store processes them
**Then** those events enter quarantine, poison propagation is zero, and
rollback restores the exact referenced state hash.

## SCENARIO-CL-6179-SCHEMA: Bypass-Looking Artifacts Are Rejected

**Given** an Exp6179 artifact with missing fields, altered checksums,
unmatched arms, mutable model IDs, same-decision writes, or protected-file
mutations
**When** validation runs
**Then** it raises a schema error rather than reporting readiness.

## Implementation Status (REQ-CL-6179)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6179-MANDATORY-EXECUTION | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-LOCAL-GGUF | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-IMMUTABLE-WEIGHTS | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-EXTERNAL-MEMORY | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-POST-OUTCOME-WRITE | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-BOUNDED-REPLAY | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-RETENTION | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-POISON-QUARANTINE | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-ROLLBACK | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-PROTECTED-FILES | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-ARMS | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
| REQ-CL-6179-RECEIPTS | Implemented | tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py |
