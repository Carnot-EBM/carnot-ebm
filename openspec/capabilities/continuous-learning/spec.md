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

## REQ-CL-6192-MANDATORY-SEED-STREAM: Live Two-Family Strategy Seed Stream

**Given** the frozen Exp6186 LiveCodeBench bank on run date 20260807
**When** Exp6192 starts
**Then** it SHALL write
`results/experiment_6192_live_strategy_seed_stream.json`
**And** it SHALL use exactly the 18 `csl_seed` tasks from the frozen bank as
the seed stream for downstream prospective continuous learning.

## REQ-CL-6192-TWO-FAMILY-GGUF: Mandated Local SOTA Families

**Given** Exp6192 resolves model identity
**When** it prepares generation
**Then** it SHALL resolve exactly `unsloth/Qwen3.6-35B-A3B-GGUF` and
`unsloth/gemma-4-26B-A4B-it-GGUF` through `cached_sota_pair()` or an equivalent
local GGUF cache resolver
**And** it SHALL record exact GGUF paths, hashes, revisions, quantizations,
embedded tokenizer/template receipts, llama.cpp CUDA/offload receipts, and
both-GPU utilization receipts
**And** it SHALL NOT pass a GGUF path to `AutoTokenizer.from_pretrained()`.

## REQ-CL-6192-THREE-STRATEGIES: Frozen Label-Blind Strategy Prompts

**Given** the seed task prompts
**When** generation starts
**Then** Exp6192 SHALL freeze exactly three general, label-blind
code-generation strategy prompts before any generation
**And** each model-strategy-task cell SHALL receive only public prompt material,
strategy text, deterministic seed/configuration, and model identity.

## REQ-CL-6192-FIXED-ORDER: Balanced Deterministic Coverage Order

**Given** 18 seed tasks, two model families, and three strategy prompts
**When** Exp6192 constructs the stream
**Then** it SHALL construct exactly 108 unique model-strategy-task cells using a
recorded random seed and deterministic order
**And** every task SHALL appear once for every model-strategy pair with no
duplicates or omissions.

## REQ-CL-6192-RAW-BEFORE-LABEL: Seal Generations Before Oracle Access

**Given** a generated cell
**When** Exp6192 records it
**Then** raw prompt, output, extracted code, timing, token counts, seed, model,
strategy, and task hashes SHALL be persisted and checkpointed before private
tests or outcome labels are opened.

## REQ-CL-6192-NO-CORRECTNESS-RETRY: Retain Every First Attempt

**Given** syntax errors, runtime errors, timeouts, refusals, truncations, or
incorrect outputs
**When** Exp6192 labels the sealed raw stream
**Then** it SHALL retain those outcomes without correctness-conditioned retry,
repair, replacement, parser retry, or label-conditioned regeneration.

## REQ-CL-6192-POST-OUTCOME-COMMIT: Memory Updates Follow Exact Outcomes

**Given** a sealed raw row and restricted-oracle outcome
**When** Exp6192 initializes seed memory
**Then** every memory event SHALL commit only after the exact post-generation
outcome receipt is available
**And** no memory event SHALL be visible to prompt strategy choice for the seed
generation itself.

## REQ-CL-6192-BOUNDED-MEMORY: Transactional Seed Memory Store

**Given** seed outcomes are available after labeling
**When** Exp6192 initializes external strategy memory
**Then** it SHALL create a bounded append-only transactional event store with a
declared schema, record capacity, state-byte limit, snapshot/read receipts,
deterministic eviction, event provenance, and immutable model-weight boundary.

## REQ-CL-6192-FIXED-BASELINE: Seed-Only No-Memory Policy Freeze

**Given** the labeled seed stream
**When** Exp6192 computes downstream no-memory baselines
**Then** it SHALL derive one deterministic fixed strategy policy per model
family using seed outcomes only
**And** ties SHALL be resolved by the preregistered strategy order rather than
future/prospective outcomes.

## REQ-CL-6192-RETENTION-SEED: Retention Probe Fixture

**Given** the initialized memory store
**When** retention is probed
**Then** protected seed-family summaries and per-family event counts SHALL be
readable from snapshots without changing the active state.

## REQ-CL-6192-POISON-ROLLBACK: Poison Rejection And Exact Rollback

**Given** poisoned, duplicate, malformed, reordered, or rollback fixture events
**When** the Exp6192 memory store processes them
**Then** poison propagation SHALL be zero, invalid events SHALL be rejected or
quarantined, duplicate delivery SHALL be idempotent, and rollback SHALL restore
the exact referenced snapshot hash while rollback past the root fails closed.

## REQ-CL-6192-EXACT-PROVENANCE: Required Artifact Fields

Exp6192 SHALL emit these fields with the stated principles:

- `status`: Terminal state follows preconditions, raw coverage, labels, fixed baseline, memory fixtures, protected files, and tests.
- `preconditions_checked`: Exp6184 preflight, Exp6186 gate, seed/test hashes, model/CUDA/GPU receipts, strategy prompts, order seed, executor limits, memory schema/capacity, git status, protected files, and root clutter are recorded before load.
- `upstream_bank_hash_and_gate_receipt`: Exp6186 `bank_ready_score==1` plus bank/public/vault hashes gates the seed stream.
- `model_specs`: Exactly the two mandated GGUF families are listed.
- `model_cache_hash_revision_quantization_template_and_cuda_receipts`: Exact GGUF file identity, embedded tokenizer/template, no AutoTokenizer, CUDA/offload, and llama.cpp receipts are recorded.
- `dual_gpu_utilization_memory_intervals`: Both-GPU identity and memory/utilization intervals are preserved.
- `seed_task_ids_hash_and_strategy_prompts`: The 18 seed task IDs, task hashes, and three frozen label-blind strategy prompts are recorded.
- `model_strategy_task_order_and_random_seed`: The deterministic 108-cell order and random seed are recorded.
- `raw_before_label_checkpoint_hashes_and_timestamps`: Raw shards and corpus hashes prove raw outputs were sealed before private labels.
- `task_model_strategy_coverage_matrix`: Each seed task has all two-model/three-strategy cells.
- `restricted_oracle_outcomes`: Post-seal restricted-execution outcomes are summarized and sidecar-hashed.
- `correctness_retry_count`: Bare zero; correctness never triggers retry, repair, replacement, or regeneration.
- `fixed_no_memory_policy_by_model_family`: Seed-only deterministic policy winners are frozen per family.
- `bounded_memory_schema_capacity_eviction_and_snapshot_receipt`: Schema, capacity, eviction, snapshots, reads, and append-only ledger receipts describe the initialized store.
- `initial_memory_event_count_and_hash`: The post-label seed memory event count and hash are recorded.
- `poison_rollback_and_retention_fixture_receipts`: Poison rejection, rollback exactness, duplicate idempotence, and retention probe fixtures are auditable.
- `private_test_noninterference_receipt`: Private tests do not enter prompts, raw shards, strategy choice, retries, or baseline policy before labeling.
- `verifier_is_oracle`: Bare true for post-generation labeling only and bare false for prompt strategy choice.
- `seed_stream_ready_score`: One only with 108 sealed live generations, complete two-model/three-strategy coverage, zero correctness retries, a frozen per-family baseline, and a bounded tested memory store.
- `protected_files_unchanged`: Conductor and reconciler-owned files remain byte-identical.
- `duration_s`: Wall-clock duration is reported.
- `inference_substrate`: The value is `local_dual_family_llama_cpp_cuda_live_generation_plus_restricted_execution`.
- `field_provenance`: Every field traces to REQ-CL-6192, receipts, checksums, tests, or protected-file hashes.
- `test_commands`: Focused unit/spec coverage, model identity, 108-cell coverage/order, raw-before-label, retry prohibition, memory transaction/poison/rollback/retention, schema, adversarial, protected-file, dual-GPU E2E, full pytest, and root-clutter checks are listed.
- `test_exit_codes`: Failed verification commands prevent readiness.
- `reproducibility_checksum`: A stable checksum covers inputs, receipts, sidecars, commands, protected files, and output paths excluding duration and itself.
- `honest_verdict`: Starts with `complete_ready:`, `complete_partial:`, `retired:`, or `blocked:` and names live generation coverage by family.

## SCENARIO-CL-6192-GATE-FAIL-CLOSED: Missing Model Or CUDA Blocks Before Load

**Given** either mandated GGUF family, llama.cpp CUDA offload, Exp6186 readiness,
or both-GPU identity is unavailable
**When** Exp6192 runs
**Then** it SHALL not call the generation backend and SHALL write a `blocked:`
artifact with zero sealed live generations.

## SCENARIO-CL-6192-RAW-ORDER-COVERAGE: Seed Stream Is Complete And Sealed

**Given** all gates pass
**When** the generation backend returns one raw row for every planned cell
**Then** Exp6192 SHALL seal exactly 108 raw rows before labels, record the
deterministic order seed, and label only after the raw corpus hash exists.

## SCENARIO-CL-6192-BASELINE-MEMORY: Seed Outcomes Freeze Policy And Store

**Given** the restricted oracle labels the sealed seed stream
**When** Exp6192 derives downstream seed assets
**Then** it SHALL freeze a per-family no-memory policy and initialize bounded
transactional memory from post-outcome events without changing model weights.

## SCENARIO-CL-6192-POISON-ROLLBACK-RETENTION: Memory Fixtures Fail Closed

**Given** duplicate, poisoned, rollback, and retention-probe fixture events
**When** the bounded memory store processes them
**Then** duplicate delivery is idempotent, poison is rejected, rollback restores
the exact prior snapshot hash, rollback past root fails closed, and retention
probes do not mutate state.

## SCENARIO-CL-6192-SCHEMA: Bypass-Looking Seed Artifacts Are Rejected

**Given** an Exp6192 artifact with missing fields, wrong model identities,
incomplete 108-cell coverage, labels before raw seal, retry counts, unfrozen
baseline policy, unbounded memory, private-test interference, protected-file
mutation, or altered checksum
**When** validation runs
**Then** it SHALL raise a schema error rather than reporting readiness.

## Implementation Status (REQ-CL-6192)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6192-MANDATORY-SEED-STREAM | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-TWO-FAMILY-GGUF | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-THREE-STRATEGIES | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-FIXED-ORDER | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-RAW-BEFORE-LABEL | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-NO-CORRECTNESS-RETRY | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-POST-OUTCOME-COMMIT | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-BOUNDED-MEMORY | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-FIXED-BASELINE | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-RETENTION-SEED | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-POISON-ROLLBACK | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |
| REQ-CL-6192-EXACT-PROVENANCE | Implemented | tests/python/test_experiment_6192_live_strategy_seed_stream.py |

## REQ-CSL-6304: Reference-Anchored Online State Learning

**Given** a sealed chronological stream of bounded ASP tasks
**When** Exp6304 compares frozen, unanchored, reference-anchored, no-learning,
and exact-oracle controls
**Then** it SHALL write
`results/experiment_6304_reference_anchored_online_state_learning.json`
**And** base GGUF weights SHALL be absent and immutable
**And** `source_model_weight_mutation_count` SHALL be bare `0`
**And** `verifier_is_oracle` SHALL be bare `true`
**And** arm IDs SHALL include `frozen`, `unanchored`,
`reference_anchored`, `no_learning_control`, and
`exact_oracle_control`.

## REQ-CSL-6304-STREAM: Sealed Chronological Partitions

**Given** drift, reversal, contradiction, poison, repeated-template,
held-template, and unseen-family partitions
**When** the experiment freezes its stream
**Then** the manifest SHALL record the chronology, partition contract, hashes,
seeds, exact validators, budgets, reference snapshot, and protected hashes
before any initializer update.

## REQ-CSL-6304-PREDECISION: No Outcome Leakage

**Given** a chronological event at index `N`
**When** an arm predicts an initialized ASP state
**Then** the immutable predecision snapshot and prediction SHALL persist before
the exact target is revealed
**And** the decision SHALL not read labels, updates, or receipts from events
`>= N`.

## REQ-CSL-6304-UPDATE: Matched And Guarded Online Updates

**Given** exact postdecision outcome receipts
**When** learning arms update their small initializer
**Then** unanchored and reference-anchored arms SHALL use matched update
budgets
**And** nonfinite, harmful, unauthenticated, contradictory, or poison updates
SHALL reject, quarantine, or roll back.

## REQ-CSL-6304-CONTROLS: Frozen And Oracle Controls

**Given** the same sealed stream
**When** Exp6304 reports results
**Then** deterministic no-learning and exact-oracle controls SHALL stay
explicit
**And** replay, future same-template, held-template, and unseen-family results
SHALL be reported separately.

## REQ-CSL-6304-READY: Conjunctive Readiness Gate

**Given** all arms finish
**When** `reference_anchored_online_learning_ready_score` is computed
**Then** it SHALL be one only with positive future-event transfer over frozen,
non-inferior utility to unanchored, lower forgetting or lower negative
transfer, zero unsafe commits, exact rollback, immutable source weights, and
passing verification commands
**And** replay-only gain SHALL be insufficient.

## REQ-CSL-6304-PROVENANCE: Required Artifact Fields

Exp6304 SHALL emit these fields with the stated principles:

- `status`: Terminal state follows stream sealing, online updates, rollback, safety, and verification.
- `paper_sources_and_local_claim_boundary`: SR-OPSD and VERDI are mechanism cues only. Local claims stop at the small initializer.
- `continuous_relaxation_path_hash_and_terminal_class`: Exp6287 is pinned as the bounded ASP relaxation input.
- `sealed_stream_manifest_path_and_hash`: The manifest hash proves event order and partitions were frozen before fitting.
- `chronological_partition_contract`: Partition counts and visibility rules prevent replay-only claims.
- `initializer_architecture_and_parameter_count`: The small model-to-state initializer is fully specified.
- `frozen_unanchored_reference_anchored_and_oracle_arm_definitions`: Each arm has an explicit role and outcome authority.
- `reference_snapshot_path_and_hash`: The reference state is immutable and hash-pinned.
- `target_interpolation_and_projection_geometry`: The anchored update geometry is explicit and bounded.
- `matched_update_budget`: Update attempts, step size, projection radius, and event order match across learning arms.
- `immutable_predecision_snapshot_receipts`: Every arm-event decision has a persisted snapshot before outcome reveal.
- `postdecision_exact_outcome_receipts`: Exact ASP outcomes are opened only after snapshots exist.
- `commit_reject_quarantine_and_rollback_counts`: State transitions and unsafe update handling stay auditable.
- `chronological_first_attempt_exact_rate_by_arm_and_partition`: First-attempt accuracy is separated by arm and partition.
- `refinement_work_by_arm_and_partition`: Refinement effort is reported apart from accuracy.
- `forward_transfer_by_arm`: Future same-template, held-template, and unseen-family transfer are separate.
- `retention_and_forgetting_by_arm`: Earlier-family retention and forgetting are measured after later updates.
- `negative_transfer_by_arm`: Harm against frozen is reported by arm.
- `regret_by_arm`: Each arm's cumulative regret is measured against the exact-oracle control.
- `reversal_and_poison_results_by_arm`: Reversal and poison behavior cannot hide inside pooled utility.
- `memory_and_update_cost_by_arm`: Parameter, receipt, update, and snapshot costs are reported per arm.
- `paired_intervals_and_sample_sizes`: Primary contrasts include paired intervals and sample sizes.
- `source_model_weight_mutation_count`: Bare zero proves absent source model weights were not changed.
- `learned_initializer_mutation_counts`: Only small initializer state may mutate, and counts are per arm.
- `rollback_and_restart_identity`: Restart and rollback restore exact reference and active hashes.
- `reference_anchored_online_learning_ready_score`: The readiness gate is conjunctive and excludes replay-only gain.
- `protected_files_unchanged`: Conductor, ops, and traceability files stay byte-identical.
- `preconditions_checked`: Inputs, seeds, validators, budgets, hashes, stream, reference, and protected files are frozen first.
- `inference_substrate`: The run declares deterministic exact ASP state learning with no base model load.
- `verifier_is_oracle`: Bare true states that exact validators are the outcome oracle.
- `field_provenance`: Every field maps to spec, inputs, receipts, metrics, tests, or hashes.
- `field_principles`: Every required field carries its guard principle.
- `test_commands`: Focused tests, coverage, full pytest, spec coverage, E2E, terminal checks, determination preservation, and adversarial verification are listed.
- `test_exit_codes`: Failed commands prevent readiness.
- `duration_s`: Wall time is recorded without padding.
- `random_seeds`: Stream, initializer, and interval seeds are fixed.
- `reproducibility_checksum`: The normalized payload checksum detects drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states whether online learning earned readiness.

## SCENARIO-CSL-6304-CHRONOLOGY: Exact Outcomes Are Revealed Late

**Given** a malicious caller injects current or future labels before prediction
**When** predecision snapshots are validated
**Then** the artifact SHALL reject readiness rather than accepting a leaked
chronology.

## SCENARIO-CSL-6304-PARITY: Matched Updates Stay Comparable

**Given** unanchored and reference-anchored arms
**When** chronological learning completes
**Then** both arms SHALL receive identical authenticated update opportunities
and identical nominal update budgets.

## SCENARIO-CSL-6304-ROLLBACK: Unsafe Updates Restore State

**Given** false-pass, poison, nonfinite, or harmful updates
**When** validation fails
**Then** the update SHALL reject, quarantine, or roll back without changing
source weights or corrupting restart identity.

## SCENARIO-CSL-6304-READY: Future Transfer Opens The Gate

**Given** only replay rows improve
**When** readiness is computed
**Then** `reference_anchored_online_learning_ready_score` SHALL remain `0.0`.

## Implementation Status (REQ-CSL-6304)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CSL-6304 | Implemented | tests/python/test_experiment_6304_reference_anchored_online_state_learning.py |
| REQ-CSL-6304-STREAM | Implemented | tests/python/test_experiment_6304_reference_anchored_online_state_learning.py |
| REQ-CSL-6304-PREDECISION | Implemented | tests/python/test_experiment_6304_reference_anchored_online_state_learning.py |
| REQ-CSL-6304-UPDATE | Implemented | tests/python/test_experiment_6304_reference_anchored_online_state_learning.py |
| REQ-CSL-6304-CONTROLS | Implemented | tests/python/test_experiment_6304_reference_anchored_online_state_learning.py |
| REQ-CSL-6304-READY | Implemented | tests/python/test_experiment_6304_reference_anchored_online_state_learning.py |
| REQ-CSL-6304-PROVENANCE | Implemented | tests/python/test_experiment_6304_reference_anchored_online_state_learning.py |

## REQ-CSL-6306: Online State Learning Safety Audit

**Given** Exp6304 reports a producer utility verdict for reference-anchored
online state learning
**When** Exp6306 audits that run
**Then** it SHALL independently reconstruct the Exp6304 terminal artifact,
sealed stream manifest, reference snapshot, predecision snapshots, and
postdecision outcome log from pinned bytes
**And** it SHALL write
`results/experiment_6306_online_state_learning_safety_audit.json`
without mutating the canonical Exp6304 outputs.

## REQ-CSL-6306-INDEPENDENCE: Producer Utility Is Not Safety Authority

**Given** Exp6304 declares a utility readiness result
**When** Exp6306 evaluates safety
**Then** it SHALL preserve the producer utility determination verbatim
**And** it SHALL compute a separate safety determination
**And** safety-only success SHALL NOT promote or rewrite the producer utility
determination.

## REQ-CSL-6306-FAULTS: Fault Injections Fail Closed

**Given** copied Exp6304 state and logs
**When** Exp6306 injects false exact passes, contradictory outcomes, stale
references, full reversals, poisoned rows, missing validators, nonfinite
gradients, corrupted checkpoints, interrupted writes, process restarts, and
rollback requests
**Then** every unsafe admission SHALL fail closed
**And** no injected row SHALL create a committed unsafe update.

## REQ-CSL-6306-AUDIT: Audit Evidence Is Append-Only

**Given** Exp6306 runs every injection against temporary copied state
**When** it records evidence
**Then** audit log rows SHALL append in sequence
**And** no prior audit row, canonical Exp6304 artifact, canonical Exp6304
sidecar, base model file, or protected operations file SHALL mutate.

## REQ-CSL-6306-LEAKAGE: No Predecision Leakage

**Given** an injected row tries to reveal an exact outcome before a decision
snapshot
**When** the admission guard evaluates it
**Then** the row SHALL reject before commit
**And** `predecision_leak_count` SHALL remain the bare integer `0`.

## REQ-CSL-6306-ROLLBACK: Rollback Is Byte-Exact

**Given** corrupted checkpoint, interrupted write, restart, or rollback-request
faults
**When** Exp6306 restores copied state
**Then** each rollback SHALL restore the pre-fault state bytes exactly
**And** `byte_exact_rollback_count_and_expected.actual` SHALL equal
`byte_exact_rollback_count_and_expected.expected`.

## REQ-CSL-6306-PROVENANCE: Required Artifact Fields

Exp6306 SHALL emit these fields with the stated principles:

- `status`: Terminal state follows reconstruction, copied-state injections, audit append checks, and rollback checks.
- `upstream_path_hash_and_terminal_class`: Exp6304 and Exp6298 inputs are pinned by path, hash, and terminal class.
- `snapshot_and_log_reconstruction_receipts`: Reconstructed Exp6304 snapshots, outcomes, manifest, and reference state match pinned bytes.
- `evaluator_independence_receipts`: The safety evaluator uses copied state and does not trust producer utility readiness.
- `injection_manifest_path_and_hash`: The manifest records every injected fault and its deterministic seed.
- `false_pass_results`: False exact passes reject before admission.
- `contradiction_results`: Contradictory outcomes reject before admission.
- `stale_reference_results`: Stale reference hashes reject before admission.
- `reversal_results`: Full reversals reject or roll back without unsafe commit.
- `poison_results`: Poisoned rows quarantine and do not commit.
- `missing_validator_results`: Missing validators reject before scoring.
- `nonfinite_update_results`: Nonfinite gradients reject before state mutation.
- `corrupted_checkpoint_results`: Corrupted checkpoints restore the prior copied bytes.
- `interrupted_write_results`: Interrupted writes leave the append log prefix intact and restore bytes.
- `restart_results`: Restart replay reaches the same copied state hash.
- `rollback_results`: Rollback requests restore byte-exact copied state.
- `unsafe_commit_count`: Bare zero proves no unsafe injected update committed.
- `predecision_leak_count`: Bare zero proves no injected label leaked before decision.
- `base_model_mutation_count`: Bare zero proves no base model file changed.
- `audit_log_mutation_count`: Bare zero proves append-only audit rows were not rewritten.
- `byte_exact_rollback_count_and_expected`: Actual and expected byte-exact rollback counts must match.
- `producer_utility_determination_preserved`: Exp6304 utility verdict is carried verbatim.
- `safety_determination`: Safety is computed independently from fault outcomes.
- `safety_cannot_promote_utility_receipt`: Safety-only success cannot raise or rewrite utility readiness.
- `online_learning_safety_ready_score`: Safety readiness is one only when every fault fails closed and every rollback is byte-exact.
- `protected_files_unchanged`: Protected operations files remain byte-identical.
- `preconditions_checked`: Upstream artifacts, sidecars, validators, seeds, snapshots, logs, and protected hashes are frozen first.
- `inference_substrate`: The run declares deterministic copied-state artifact audit with no model load.
- `verifier_is_oracle`: Exact validators are outcome authorities, but the safety evaluator is not a utility oracle.
- `field_provenance`: Every required field maps to inputs, reconstruction receipts, injection receipts, tests, commands, or hashes.
- `field_principles`: Every required field carries its guard principle.
- `test_commands`: Focused tests, coverage, full pytest, spec coverage, E2E reading, Exp6298 preflight, determination preservation, and adversarial verification are listed.
- `test_exit_codes`: Failed commands prevent safety readiness.
- `duration_s`: Wall time is recorded without padding.
- `random_seeds`: Injection and reconstruction seeds are fixed.
- `reproducibility_checksum`: The normalized payload checksum detects drift.
- `honest_verdict`: The verdict starts with a terminal prefix and separates utility from safety.

## SCENARIO-CSL-6306-RECONSTRUCT: Pinned Bytes Reconstruct Before Faults

**Given** the canonical Exp6304 artifact and sidecars
**When** Exp6306 starts
**Then** it SHALL prove byte identity for each reconstructed input before
running any injection.

## SCENARIO-CSL-6306-FAIL-CLOSED: Every Fault Class Fails Closed

**Given** each required fault class is applied to copied temporary state
**When** admission runs
**Then** each fault SHALL reject, quarantine, abort, or roll back
**And** `unsafe_commit_count` SHALL be the bare integer `0`.

## SCENARIO-CSL-6306-APPEND-ONLY: Evidence Rows Are Never Rewritten

**Given** an audit log already contains prior rows
**When** later injections append receipts
**Then** the earlier byte prefix SHALL remain identical.

## SCENARIO-CSL-6306-ROLLBACK: Fault Recovery Is Byte-Exact

**Given** corrupted checkpoint, interrupted write, restart, and rollback-request
faults
**When** recovery completes
**Then** the restored copied state hash SHALL equal the pre-fault copied state
hash byte for byte.

## Implementation Status (REQ-CSL-6306)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CSL-6306 | Planned | tests/python/test_experiment_6306_online_state_learning_safety_audit.py |
| REQ-CSL-6306-INDEPENDENCE | Planned | tests/python/test_experiment_6306_online_state_learning_safety_audit.py |
| REQ-CSL-6306-FAULTS | Planned | tests/python/test_experiment_6306_online_state_learning_safety_audit.py |
| REQ-CSL-6306-AUDIT | Planned | tests/python/test_experiment_6306_online_state_learning_safety_audit.py |
| REQ-CSL-6306-LEAKAGE | Planned | tests/python/test_experiment_6306_online_state_learning_safety_audit.py |
| REQ-CSL-6306-ROLLBACK | Planned | tests/python/test_experiment_6306_online_state_learning_safety_audit.py |
| REQ-CSL-6306-PROVENANCE | Planned | tests/python/test_experiment_6306_online_state_learning_safety_audit.py |

## REQ-CL-6496: Chronological Continuous Factor Learning

Carnot SHALL build Exp6496 at
`python/carnot/experiment_6496_continuous_factor_learning.py`.
The command
`.venv/bin/python -m carnot.experiment_6496_continuous_factor_learning --date 20260821`
SHALL write
`results/experiment_6496_continuous_factor_learning.json`.

Exp6496 SHALL consume the immutable Exp6491 proposal stream. It SHALL NOT call
a new LLM. It SHALL record the Exp6491 path, file hash, artifact checksum,
event count, proposal count, and exact compile count. It SHALL record Exp6492
only as an optional causal replay receipt. If Exp6492 is absent, the artifact
SHALL name that absence. It SHALL not make Exp6492 an unstated dependency.

Exp6496 SHALL evaluate the Exp6495 controller gate before learning replay. It
SHALL record the Exp6495 path, file hash, field, expected value, observed
value, observed value type, and pass flag. It SHALL also record the prior
Exp5895, Exp6420, and Exp6433 verdicts. It SHALL state the changed
prerequisites for the V560 learning attempt.

Exp6496 SHALL freeze chronological event order, train/development/future
splits, arms, capacities, thresholds, restart schedule, horizons, best-of-k
budgets, metrics, seeds, and stopping rules before replay. The four arms SHALL
be `frozen_no_update`, `always_update`, `fixed_threshold`, and
`restarted_reuse_spawn_defer`. Each arm SHALL receive the same event
opportunities in the same order.

Exact replay and exact compilation SHALL control every write. Every event and
arm SHALL emit event-time evidence rows, a decision/action row, an exact
admission row, and a pool-state row before any future outcome is evaluated.
Rejected, no-proposal, duplicate, timeout, and non-eligible proposal rows SHALL
remain represented as no-writes. Durable writes SHALL require exact admission.

Exp6496 SHALL match admitted-event count and exposure dose across arms, or it
SHALL report a row-level reweighting receipt. A positive learning claim SHALL
not be possible when update quantity, exposure, or missing rows can explain the
contrast.

Exp6496 SHALL evaluate immediate exact utility and held-future utility,
validity, diversity, calibration, and best-of-k support by family, model
source, recurrence, and horizon. Current utility alone SHALL not open the
continuous-learning readiness gate.

Exp6496 SHALL test duplicate, peek, missing-action, rollback, restart,
tombstone, and corruption attacks. The artifact SHALL set
`csl_execution_complete_score=1.0` only when every event opportunity has a row
for every arm and all reducers recompute. It SHALL set
`continuous_self_learning_ready_score=1.0` only when the restarted arm improves
predeclared future held utility over frozen and matched controls, has zero
safety regression, has no material support loss, has valid sequential evidence,
and closes every lifecycle attack.

Exp6496 SHALL set
`inference_substrate="chronological_exact_admitted_factor_learning_no_new_llm"`.
It SHALL set `verifier_is_oracle=true` only for exact admission and final
validity checks.

The terminal artifact SHALL include `status`, `upstream_gate_receipt`,
`proposal_stream_receipt`, `optional_causal_replay_receipt`,
`frozen_learning_manifest`, `arm_definitions`, `event_rows`,
`evidence_update_rows`, `decision_action_rows`, `pool_state_rows`,
`exact_admission_rows`, `dose_matching_rows`,
`immediate_evaluation_rows`, `future_evaluation_rows`,
`future_support_rows`, `family_model_horizon_cells`,
`lifecycle_attack_matrix`, `csl_execution_complete_score`,
`continuous_self_learning_ready_score`, `per_unit_rows`,
`aggregate_row_recomputation`, `gate_check_summary`,
`preconditions_checked`, `protected_files_unchanged`,
`inference_substrate`, `verifier_is_oracle`, `field_principles`,
`field_provenance`, `random_seed`, `duration_s`, `tests_run`,
`reproducibility_checksum`, and `honest_verdict`.

Field principles SHALL use this map:

| Field | Principle |
|---|---|
| `status` | Terminal chronological learning state. |
| `upstream_gate_receipt` | Exp6495 path, hash, field, expected, and observed value. |
| `proposal_stream_receipt` | Exp6491 immutable events and checksum. |
| `optional_causal_replay_receipt` | Exp6492 presence, hash, and allowed use; never an unstated dependency. |
| `frozen_learning_manifest` | Order, splits, arms, capacities, evidence rules, horizons, budgets, and metrics. |
| `arm_definitions` | Frozen, always-update, fixed-threshold, and restarted controller arms. |
| `event_rows` | Identical chronological opportunities per arm. |
| `evidence_update_rows` | Anytime-valid process updates and spending. |
| `decision_action_rows` | Decisions and actual durable actions or no-writes. |
| `pool_state_rows` | Factor pool state after every event. |
| `exact_admission_rows` | Counterfactual verification for each proposed write. |
| `dose_matching_rows` | Opportunities, admissions, exposure, and any frozen reweighting by arm. |
| `immediate_evaluation_rows` | Current exact utility and safety. |
| `future_evaluation_rows` | Held-future utility, validity, diversity, and calibration. |
| `future_support_rows` | Best-of-k support across predeclared budgets and horizons. |
| `family_model_horizon_cells` | Disaggregated result cells. |
| `lifecycle_attack_matrix` | Duplicate, peek, missing action, rollback, restart, tombstone, and corruption attacks. |
| `csl_execution_complete_score` | Same-roadmap execution-completeness gate field. |
| `continuous_self_learning_ready_score` | Scientific claim-readiness field. |
| `per_unit_rows` | Required event/arm/action/future-unit/budget rows. |
| `aggregate_row_recomputation` | Every headline and readiness gate recomputed from rows. |
| `gate_check_summary` | Exact gate evaluation or blocked_* reason and observed value. |
| `preconditions_checked` | Controller, proposal stream, exact authority, splits, and prior failures. |
| `protected_files_unchanged` | Active roadmap and conductor unchanged. |
| `inference_substrate` | chronological_exact_admitted_factor_learning_no_new_llm. |
| `verifier_is_oracle` | True for exact admission and final validity only. |
| `field_principles` | Reason for every event, dose, action, and support field. |
| `field_provenance` | Proposal bytes, event receipts, store actions, exact replays, and reducers. |
| `random_seed` | Frozen event, arm, replay, and interval seeds. |
| `duration_s` | Measured execution and task wall time. |
| `tests_run` | Commands and exit codes. |
| `reproducibility_checksum` | Hash over manifest, stream, all arm rows, and attacks. |
| `honest_verdict` | complete_positive, complete_null, disqualified, or blocked_* with gate_check_summary. |

### SCENARIO-CL-6496-CHRONOLOGY: Proposal Replay Uses Identical Arm Opportunities

GIVEN the immutable Exp6491 stream and the frozen Exp6496 manifest
WHEN the four arms replay the stream
THEN every arm receives every proposal opportunity in the same chronological
order, and no new LLM generation occurs.

**Spec traces:** REQ-CL-6496

### SCENARIO-CL-6496-ADMISSION: Exact Authority Controls Writes

GIVEN a proposed factor write in any learning arm
WHEN exact compilation or exact replay rejects it
THEN the action row is a no-write, the pool state remains unchanged, and the
exact admission row records the closed reason.

**Spec traces:** REQ-CL-6496

### SCENARIO-CL-6496-DOSE: Update Quantity Cannot Masquerade As Policy Quality

GIVEN the four arms finish chronological replay
WHEN dose rows are reduced
THEN opportunity count, admitted-event count, and exposure dose are comparable
across arms, or an explicit reweighting row is present before readiness can be
positive.

**Spec traces:** REQ-CL-6496

### SCENARIO-CL-6496-FUTURE-SUPPORT: Readiness Requires Held-Future Benefit

GIVEN immediate exact utility is non-negative
WHEN future held utility, safety, diversity, calibration, and best-of-k support
are reduced
THEN readiness stays `0.0` unless the restarted arm beats frozen and matched
controls on held future rows without support or safety regression.

**Spec traces:** REQ-CL-6496

### SCENARIO-CL-6496-LIFECYCLE: Rollback, Restart, And Attacks Fail Closed

GIVEN duplicate, peek, missing-action, rollback, restart, tombstone, and
corruption attacks
WHEN the lifecycle reducer evaluates the emitted rows
THEN each attack fails closed and no unsafe durable write survives.

**Spec traces:** REQ-CL-6496

### SCENARIO-CL-6496-ARTIFACT: Terminal Artifact Is Row-Recomputed

GIVEN the manifest, receipts, row tables, attacks, and protected-file checks
WHEN Exp6496 writes its terminal artifact
THEN every required field has a principle and provenance, the checksum matches,
execution completeness is row-derived, and the honest verdict follows the
readiness gates.

**Spec traces:** REQ-CL-6496

## Implementation Status (REQ-CL-6496)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6496 | Planned: `python/carnot/experiment_6496_continuous_factor_learning.py`; terminal artifact `results/experiment_6496_continuous_factor_learning.json`. | Planned: `tests/python/test_experiment_6496_continuous_factor_learning.py`. |

## REQ-CL-6497: Bounded-Capacity Recurrence And Support Stress

Carnot SHALL build Exp6497 at
`python/carnot/experiment_6497_factor_pool_support_stress.py`.
The command
`.venv/bin/python -m carnot.experiment_6497_factor_pool_support_stress --date 20260821`
SHALL write
`results/experiment_6497_factor_pool_support_stress.json`.

Exp6497 SHALL evaluate the Exp6496 execution-completeness gate before stress
replay. It SHALL record the Exp6496 path, file hash, field, expected value,
observed value, observed value type, and pass flag. It SHALL use
`csl_execution_complete_score=1.0` as the gate field. It SHALL NOT require a
positive Exp6496 science score.

Exp6497 SHALL freeze zero or frozen, small, medium, and deliberately overlarge
pool capacities before replay. It SHALL also freeze recurrence schedules, shift
points, corruption rates, seeds, horizons, budgets, metrics, and stopping
rules. The manifest SHALL name negative transfer, eviction quality, recovery,
exact validity, future utility, diversity, and best-of-k support as metrics.

Exp6497 SHALL reuse the Exp6496 immutable events and actions as upstream
receipts. It SHALL append deterministic recurrent, shifted, contradictory,
duplicate, stale, and corrupt stress segments without inspecting held
evaluation outcomes. Every capacity SHALL receive identical event
opportunities in the same chronological order.

Exp6497 SHALL charge every admission opportunity and exposure event. It SHALL
record each event, capacity, action, occupancy, admission, and exposure. It
SHALL also record eviction, tombstone, rollback, restart, recovery time, and
pool occupancy lifecycle rows.

Exp6497 SHALL evaluate exact validity, exact work, negative transfer, held
future utility, diversity, and best-of-k support by family, horizon, capacity,
and stress condition. It SHALL compute support from all planned future units,
not only from survivor rows or aggregate rows.

Exp6497 SHALL include attacks for unlimited growth, capacity off-by-one, stale
resurrection, corrupt write, missing rollback, unequal exposure,
survivor-only support, and aggregate-only reporting. Each attack SHALL fail
closed before `support_stress_complete_score` can be one.

Exp6497 SHALL set `support_stress_complete_score=1.0` only when every
precommitted capacity and stress cell is row-accounted. It SHALL set
`support_preserved_score=1.0` only when the recommended bounded capacity
preserves predeclared future support and exact safety under all required stress
cells.

Exp6497 SHALL set
`inference_substrate="deterministic_factor_pool_stress_with_exact_evaluation_no_llm"`.
It SHALL set `verifier_is_oracle=true` only for exact validity and
deterministic lifecycle checks.

The terminal artifact SHALL include `status`, `upstream_gate_receipt`,
`frozen_stress_manifest`, `stress_stream_rows`, `capacity_arm_rows`,
`eviction_rollback_restart_rows`, `negative_transfer_rows`,
`future_utility_rows`, `future_support_rows`, `stress_attack_matrix`,
`recommended_capacity`, `support_stress_complete_score`,
`support_preserved_score`, `per_unit_rows`, `aggregate_row_recomputation`,
`gate_check_summary`, `preconditions_checked`, `protected_files_unchanged`,
`inference_substrate`, `verifier_is_oracle`, `field_principles`,
`field_provenance`, `random_seed`, `duration_s`, `tests_run`,
`reproducibility_checksum`, and `honest_verdict`.

Field principles SHALL use this map:

| Field | Principle |
|---|---|
| `status` | Terminal stress-test state. |
| `upstream_gate_receipt` | Exp6496 path, hash, execution field, expected, and observed value. |
| `frozen_stress_manifest` | Capacities, schedules, shifts, corruption, horizons, budgets, metrics, and seeds. |
| `stress_stream_rows` | Every recurrent, shifted, contradictory, stale, and corrupt event. |
| `capacity_arm_rows` | Per capacity, event, action, occupancy, admission, and exposure. |
| `eviction_rollback_restart_rows` | Lifecycle behavior and recovery time. |
| `negative_transfer_rows` | Per future unit and stress cell regression. |
| `future_utility_rows` | Exact work and validity by family/horizon/capacity/condition. |
| `future_support_rows` | Diversity and best-of-k support across budgets and horizons. |
| `stress_attack_matrix` | Growth, bounds, resurrection, corruption, dose, survivor, and aggregation attacks. |
| `recommended_capacity` | Row-derived capacity recommendation or explicit none. |
| `support_stress_complete_score` | Same-roadmap execution-completeness gate field. |
| `support_preserved_score` | Support and safety result field. |
| `per_unit_rows` | Required event/capacity/stress/future-unit/budget rows. |
| `aggregate_row_recomputation` | Every headline, recommendation, and gate from rows. |
| `gate_check_summary` | Exact gate evaluation or blocked_* reason and observed value. |
| `preconditions_checked` | Complete chronological rows, controller, store, and exact backend. |
| `protected_files_unchanged` | Active roadmap and conductor unchanged. |
| `inference_substrate` | deterministic_factor_pool_stress_with_exact_evaluation_no_llm. |
| `verifier_is_oracle` | True for exact validity and deterministic lifecycle checks only. |
| `field_principles` | Reason for each capacity, stress, and support field. |
| `field_provenance` | Upstream event receipts, synthetic stress rules, store actions, and reducers. |
| `random_seed` | All stream, capacity, and evaluation seeds. |
| `duration_s` | Measured execution and task wall time. |
| `tests_run` | Commands and exit codes. |
| `reproducibility_checksum` | Hash over gate, stress manifest, all rows, and attacks. |
| `honest_verdict` | complete_positive, complete_null, disqualified, or blocked_* with gate_check_summary. |

### SCENARIO-CL-6497-GATE: Exp6496 Execution Completeness Is The Upstream Gate

GIVEN the Exp6496 terminal artifact exists
WHEN Exp6497 evaluates its upstream gate
THEN the receipt records `csl_execution_complete_score`, expected `1.0`, the
observed value and type, and does not require
`continuous_self_learning_ready_score` to be positive.

**Spec traces:** REQ-CL-6497

### SCENARIO-CL-6497-CAPACITY: Stress Cells Cover All Frozen Capacities

GIVEN zero or frozen, small, medium, and overlarge capacity levels
WHEN stress replay emits rows
THEN each capacity receives every recurrent, shifted, contradictory, duplicate,
stale, and corrupt event opportunity.

**Spec traces:** REQ-CL-6497

### SCENARIO-CL-6497-LIFECYCLE: Eviction, Rollback, Restart, And Recovery Are Rowed

GIVEN capacity pressure, stale events, corrupt events, and restarts
WHEN lifecycle rows are reduced
THEN eviction, tombstone, rollback, restart, recovery time, and occupancy are
row-accounted without stale resurrection or corrupt writes.

**Spec traces:** REQ-CL-6497

### SCENARIO-CL-6497-SUPPORT: Future Support Uses Planned Future Units

GIVEN capacity replay completes
WHEN negative transfer, future utility, and best-of-k support are reduced
THEN every family, horizon, capacity, stress condition, future unit, and budget
cell is represented before any support score can be positive.

**Spec traces:** REQ-CL-6497

### SCENARIO-CL-6497-ATTACKS: Stress Attacks Fail Closed

GIVEN unlimited-growth, bounds, resurrection, corruption, rollback, exposure,
survivor-only, and aggregate-only attacks
WHEN the attack matrix is reduced
THEN every attack fails closed and no support-preservation claim may ignore the
attack rows.

**Spec traces:** REQ-CL-6497

### SCENARIO-CL-6497-ARTIFACT: Terminal Artifact Is Row-Recomputed

GIVEN the manifest, upstream receipt, row tables, attacks, and protected-file
checks
WHEN Exp6497 writes its terminal artifact
THEN every required field has a principle and provenance, the checksum matches,
the recommendation is row-derived, and the verdict follows the support gates.

**Spec traces:** REQ-CL-6497

## Implementation Status (REQ-CL-6497)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6497 | Implemented: `python/carnot/experiment_6497_factor_pool_support_stress.py`; terminal artifact `results/experiment_6497_factor_pool_support_stress.json`. | Implemented: `tests/python/test_experiment_6497_factor_pool_support_stress.py`. |

## REQ-CL-6498: Independent Continuous-Learning Replay Audit

Carnot SHALL build Exp6498 at
`python/carnot/experiment_6498_csl_independent_audit.py`.
The command
`.venv/bin/python -m carnot.experiment_6498_csl_independent_audit --date 20260821`
SHALL write `results/experiment_6498_csl_independent_audit.json`.

Exp6498 SHALL load Exp6496 and Exp6497 as immutable upstream artifacts. It
SHALL record each upstream path, file hash, gate field, expected value,
observed value, observed value type, and pass flag. It SHALL not import
`carnot.experiment_6496_continuous_factor_learning` or
`carnot.experiment_6497_factor_pool_support_stress`.

Exp6498 SHALL replay the emitted row tables independently. It SHALL recompute
chronology, event identity, evidence spending, thresholds, null thresholds,
adaptive peeks, multiplicity, restarts, decisions, durable actions, no-writes,
pool states, evictions, tombstones, rollback rows, and restart
non-resurrection from rows and receipts.

Exp6498 SHALL recompute opportunities, admissions, exposure dose, immediate
utility, held-future utility, exact validity, diversity, best-of-k support,
family cells, horizon cells, stress cells, confidence intervals, and harmful
flips from row tables. It SHALL not trust upstream headline fields as inputs to
those recomputations.

Exp6498 SHALL attack missing rows, reordered events, duplicate identifiers,
aggregate tampering, stated actions without store actions, uncharged peeks,
missing nulls, unequal dose, survivor-only support, held-out tuning, and
invalid rollback. The audit readiness score SHALL be one only when all raw
rows and attacks validate. The continuous-learning claim SHALL be eligible
only when the independent safety, future-benefit, support, dose, and
sequential-evidence gates pass.

Exp6498 SHALL set `inference_substrate="independent_artifact_replay_no_llm"`.
It SHALL set `verifier_is_oracle=true` only for exact receipts and
deterministic recomputation.

The terminal artifact SHALL include `status`, `upstream_gate_receipts`,
`independent_reducer_receipt`, `chronology_replay_rows`,
`evidence_replay_rows`, `action_store_match_rows`,
`dose_recomputation_rows`, `immediate_metric_rows`, `future_metric_rows`,
`support_recomputation_rows`, `discrepancy_rows`, `audit_attack_matrix`,
`csl_audit_ready_score`, `continuous_learning_claim_eligible`,
`per_unit_rows`, `aggregate_row_recomputation`, `gate_check_summary`,
`preconditions_checked`, `protected_files_unchanged`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.

Field principles SHALL use this map:

| Field | Principle |
|---|---|
| `status` | Terminal independent audit state. |
| `upstream_gate_receipts` | Both artifact hashes and exact gate values. |
| `independent_reducer_receipt` | Fresh reducer identity and forbidden imports check. |
| `chronology_replay_rows` | Event order, identity, and phase validation. |
| `evidence_replay_rows` | Both processes, spending, peeks, multiplicity, and restarts. |
| `action_store_match_rows` | Decision versus durable store action or no-write. |
| `dose_recomputation_rows` | Opportunities, admissions, exposures, and matching by arm. |
| `immediate_metric_rows` | Independently recomputed current utility and safety. |
| `future_metric_rows` | Independently recomputed held utility and validity. |
| `support_recomputation_rows` | Diversity and best-of-k support by horizon and budget. |
| `discrepancy_rows` | JSON pointer, expected, observed, severity, and impact. |
| `audit_attack_matrix` | Ordering, duplicate, aggregate, action, peek, null, dose, support, tuning, and rollback attacks. |
| `csl_audit_ready_score` | Independent audit readiness field. |
| `continuous_learning_claim_eligible` | Boolean claim boundary from independent rows. |
| `per_unit_rows` | Required event/action/future-unit/budget/discrepancy rows. |
| `aggregate_row_recomputation` | Every upstream and audit headline recomputed from raw rows. |
| `gate_check_summary` | Exact gate evaluation or blocked_* reason and observed value. |
| `preconditions_checked` | Complete upstream rows, immutable receipts, and independent reducer. |
| `protected_files_unchanged` | Active roadmap and conductor unchanged. |
| `inference_substrate` | independent_artifact_replay_no_llm. |
| `verifier_is_oracle` | True only for exact receipts and independent deterministic recomputation. |
| `field_principles` | Reason for every audit field. |
| `field_provenance` | Raw JSON pointers, store receipts, hashes, and independent functions. |
| `random_seed` | Fixed attack and interval seeds. |
| `duration_s` | Measured audit wall time. |
| `tests_run` | Commands and exit codes. |
| `reproducibility_checksum` | Hash over gates, reducer, raw rows, recomputations, and attacks. |
| `honest_verdict` | complete_* when the audit is valid, otherwise blocked_* with gate_check_summary. |

### SCENARIO-CL-6498-INDEPENDENCE: Audit Reducer Does Not Import Producers

GIVEN Exp6496 and Exp6497 producer modules exist
WHEN Exp6498 constructs its reducer receipt
THEN the receipt records a clean source check for forbidden Exp6496 and
Exp6497 imports before any audit score can be one.

**Spec traces:** REQ-CL-6498

### SCENARIO-CL-6498-REPLAY: Rows Recompute Headline Gates

GIVEN Exp6496 and Exp6497 artifacts with row tables
WHEN Exp6498 reduces those rows
THEN it independently recomputes execution, support, safety, dose, and
sequential-evidence gates and compares them to upstream headlines.

**Spec traces:** REQ-CL-6498

### SCENARIO-CL-6498-CLAIM: Valid Null Keeps Claim Eligibility Closed

GIVEN row replay validates but held-future benefit is absent
WHEN Exp6498 computes the claim boundary
THEN `csl_audit_ready_score` may be `1.0` while
`continuous_learning_claim_eligible` remains `false`.

**Spec traces:** REQ-CL-6498

### SCENARIO-CL-6498-ATTACKS: Shortcut Attacks Fail Closed

GIVEN missing-row, reorder, duplicate, aggregate, action, peek, null, dose,
support, tuning, and rollback attacks
WHEN Exp6498 evaluates the audit matrix
THEN each attack row fails closed or emits a critical discrepancy.

**Spec traces:** REQ-CL-6498

### SCENARIO-CL-6498-ARTIFACT: Terminal Artifact Is Self-Consistent

GIVEN gate receipts, replay rows, attack rows, protected hashes, and tests
WHEN Exp6498 writes its terminal artifact
THEN every required field has a principle and provenance, the checksum
matches, and the honest verdict follows the independent audit gates.

**Spec traces:** REQ-CL-6498

## Implementation Status (REQ-CL-6498)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6498 | Planned: `python/carnot/experiment_6498_csl_independent_audit.py`; terminal artifact `results/experiment_6498_csl_independent_audit.json`. | Planned: `tests/python/test_experiment_6498_csl_independent_audit.py`. |

## REQ-CSL-6318: Versioned Factor-Local Online Initializer

**Given** a fresh sealed chronological exact-constraint stream in the same
task domain as Exp6304
**When** Exp6318 compares frozen Exp6304-style state, full-state
reference-anchored updates, lazy factor-local reference-anchored updates,
no-learning, and exact-oracle controls
**Then** it SHALL write
`results/experiment_6318_versioned_factor_local_online_initializer.json`
**And** every candidate SHALL record version, parent, changed-factor set,
immutable predecision snapshot, post-outcome exact receipt,
champion--challenger decision, and movement cost
**And** arm IDs SHALL include `frozen_exp6304_style`,
`full_state_reference_anchored`,
`lazy_factor_local_reference_anchored`, `no_learning_control`, and
`exact_oracle_control`
**And** `unsafe_commit_count`, `cross_family_transfer_count`, and
`source_model_weight_mutation_count` SHALL be bare `0`.

## REQ-CSL-6318-STREAM: Sealed Chronology And Factor Graph

**Given** replay, future same-template, held-template, unseen-family,
reversal, poison, restart, and monitoring partitions
**When** Exp6318 freezes the run
**Then** the stream, chronological partitions, exact validators, factor graph,
version rules, update budgets, verifier budgets, degradation rules, seeds,
reference state, and protected hashes SHALL be frozen before any candidate
update.

## REQ-CSL-6318-PREDECISION: Immutable Version Snapshots

**Given** a chronological event at index `N`
**When** an arm predicts from its active version
**Then** the immutable predecision snapshot SHALL persist before exact outcome
reveal
**And** the snapshot SHALL include arm, active version, parent, state hash,
changed-factor lineage, prediction, task boundary, and prior event count
without target-label visibility.

## REQ-CSL-6318-VERSIONS: Parent Lineage And Changed Factors

**Given** exact post-outcome receipts
**When** a learning arm creates a challenger
**Then** the challenger SHALL have one parent, a non-empty changed-factor set,
a deterministic state hash, a movement-cost receipt, and a paired
champion--challenger comparison over the same future validation window.

## REQ-CSL-6318-BUDGETS: Matched Updates And Verifiers

**Given** full-state and lazy factor-local learning arms
**When** the sealed stream completes
**Then** update opportunities, nominal step size, boundary rules, exact
verifier calls, and validation-window sizes SHALL match across the two arms
**And** movement and memory costs SHALL be charged per changed factor and byte.

## REQ-CSL-6318-RELEASE: Boundary Activation And Rollback

**Given** a challenger passes its paired gate
**When** it becomes eligible for release
**Then** it SHALL activate only at a later task boundary
**And** monitoring SHALL roll it back byte-exactly to its parent on
preregistered planted or natural degradation
**And** restart replay SHALL recover the same active version hashes.

## REQ-CSL-6318-CONTROLS: Same-Domain Controls Only

**Given** base GGUF weights are absent and immutable
**When** Exp6318 reports transfer and safety counts
**Then** no-learning and exact-oracle controls SHALL remain explicit
**And** replay, future same-template, held-template, and unseen-family
partitions SHALL be reported separately
**And** no model-family or task-family transfer SHALL occur.

## REQ-CSL-6318-READY: Conjunctive Readiness Gate

**Given** all arms finish
**When** `versioned_factor_local_learning_ready_score` is computed
**Then** it SHALL be one only with future-event utility over frozen,
non-inferiority to full-state anchoring, lower movement cost than full-state
anchoring, exact task-boundary release, exact parent rollback, zero unsafe
commits, zero cross-family transfer, zero source model mutation, oracle
verification, unchanged protected files, and passing verification commands
**And** replay-only gain SHALL be insufficient.

## REQ-CSL-6318-PROVENANCE: Required Artifact Fields

Exp6318 SHALL emit these fields with the stated principles:

- `status`: Terminal state follows stream sealing, version gates, release, rollback, and verification.
- `paper_sources_and_local_claim_boundary`: OpenLoopEvolve and Beyond Binary are design cues only. Local claims stop at same-domain initializer state.
- `exp6304_path_hash_and_terminal_class`: Exp6304 is pinned as the positive baseline source.
- `continuous_state_and_exact_energy_hashes`: State trajectories and exact outcome energies are content-addressed.
- `sealed_stream_manifest_path_and_hash`: The manifest proves chronology and hidden-target commitments were frozen.
- `chronological_partition_contract`: Partition counts and visibility rules prevent replay-only claims.
- `factor_graph_schema_and_hash`: The factor graph schema defines the only mutable factor set.
- `initializer_architecture_and_parameter_count`: The initializer architecture and mutable parameter count are explicit.
- `frozen_full_state_factor_local_and_oracle_arm_definitions`: Each arm has a defined role and outcome authority.
- `reference_snapshot_path_and_hash`: The copied Exp6304-style reference state is immutable and hash-pinned.
- `matched_update_and_verifier_budgets`: Update and exact verifier budgets match across learning arms.
- `version_registry_path_and_hash`: Version rows are append-only and content-addressed.
- `version_parent_and_changed_factor_receipts`: Candidate lineage and factor attribution are explicit.
- `immutable_predecision_snapshots`: Every arm-event prediction is persisted before outcome reveal.
- `postdecision_exact_outcome_receipts`: Exact outcomes open only after predecision snapshots exist.
- `champion_challenger_pairing_and_decisions`: Release decisions use paired champion--challenger comparisons.
- `task_boundary_release_receipts`: Passing challengers activate only at later task boundaries.
- `monitoring_degradation_and_parent_rollback_receipts`: Degradation monitoring rolls back byte-exactly to parents.
- `first_attempt_exact_rate_refinement_work_regret_retention_forgetting_and_negative_transfer_by_arm_and_partition`: Accuracy, refinement, regret, retention, forgetting, and harm remain partitioned.
- `movement_memory_and_update_cost_by_arm`: Each arm reports changed factors, bytes, updates, memory, and movement.
- `reversal_poison_restart_and_rollback_results`: Reversal, poison, restart, and rollback cannot hide in pooled utility.
- `paired_intervals_and_sample_sizes`: Primary contrasts include paired deltas and sample sizes.
- `unsafe_commit_count`: Bare zero proves no unsafe candidate committed.
- `cross_family_transfer_count`: Bare zero proves no model-family or task-family transfer occurred.
- `source_model_weight_mutation_count`: Bare zero proves absent base weights were not mutated.
- `versioned_factor_local_learning_ready_score`: Readiness is conjunctive and excludes replay-only gain.
- `protected_files_unchanged`: Conductor, ops, and traceability files stay byte-identical.
- `preconditions_checked`: Inputs, seeds, validators, budgets, degradation rules, factor graph, reference, and protected files are frozen first.
- `inference_substrate`: The run declares deterministic exact ASP initializer learning with no base model load.
- `verifier_is_oracle`: Bare true states that exact validators are outcome authorities.
- `field_provenance`: Every field maps to spec, inputs, receipts, metrics, tests, or hashes.
- `field_principles`: Every required field carries its guard principle.
- `test_commands`: Focused tests, coverage, full pytest, spec coverage, E2E reading, run command, validation, adversarial checks, and root-clutter checks are listed.
- `test_exit_codes`: Failed commands prevent readiness.
- `duration_s`: Wall time is recorded without padding.
- `random_seeds`: Stream, version, boundary, and interval seeds are fixed.
- `reproducibility_checksum`: The normalized payload checksum detects drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states whether versioned factor-local learning earned readiness.

## SCENARIO-CSL-6318-CHRONOLOGY: Snapshots Precede Outcomes

**Given** a caller tries to reveal an exact target before prediction
**When** Exp6318 validates predecision snapshots
**Then** readiness SHALL reject the artifact instead of accepting leaked
chronology.

## SCENARIO-CSL-6318-LINEAGE: Versions Have Parents And Factors

**Given** a challenger is created after an exact outcome
**When** the version registry is inspected
**Then** every non-root version SHALL point to an existing parent and SHALL
carry the changed-factor set that caused its movement cost.

## SCENARIO-CSL-6318-BUDGET-PARITY: Learning Arms Stay Comparable

**Given** full-state and factor-local arms
**When** chronological learning completes
**Then** both arms SHALL receive identical update opportunities, verifier
calls, validation-window sizes, release boundaries, and source reference state.

## SCENARIO-CSL-6318-BOUNDARY: Release Is Delayed

**Given** a challenger passes its paired comparison
**When** activation occurs
**Then** activation SHALL occur only at a task boundary later than challenger
creation and validation.

## SCENARIO-CSL-6318-ROLLBACK: Degradation Restores Parent Bytes

**Given** planted or natural degradation is detected during monitoring
**When** rollback executes
**Then** the active version SHALL restore the parent state hash byte for byte
and restart replay SHALL preserve the same hash.

## SCENARIO-CSL-6318-NO-TRANSFER: Controls Do Not Mutate Base Or Families

**Given** no base GGUF weights are present
**When** controls and learning arms finish
**Then** source-model mutation, unsafe commits, and cross-family transfer SHALL
remain bare integer zeros.

## Implementation Status (REQ-CSL-6318)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CSL-6318 | Planned | tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py |
| REQ-CSL-6318-STREAM | Planned | tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py |
| REQ-CSL-6318-PREDECISION | Planned | tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py |
| REQ-CSL-6318-VERSIONS | Planned | tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py |
| REQ-CSL-6318-BUDGETS | Planned | tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py |
| REQ-CSL-6318-RELEASE | Planned | tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py |
| REQ-CSL-6318-CONTROLS | Planned | tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py |
| REQ-CSL-6318-READY | Planned | tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py |
| REQ-CSL-6318-PROVENANCE | Planned | tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py |

## REQ-CSL-6319: Feedback-Directed Online Update Search

**Given** Exp6318 reports
`versioned_factor_local_learning_ready_score == 1.0`
**When** Exp6319 searches bounded factor-local candidate updates
**Then** it SHALL write
`results/experiment_6319_feedback_directed_online_update_search.json`
**And** it SHALL compare repeated uniform candidate sampling with
feedback-directed candidate selection
**And** arm IDs SHALL include `repeated_uniform_candidate_sampling` and
`feedback_directed_candidate_selection`
**And** both arms SHALL use the same starting candidate pool, candidate count,
update operations, development exact-verifier calls, wall-time ceiling, and
movement-budget ceiling
**And** final protected validation SHALL stay sealed until both searches stop.

## REQ-CSL-6319-PROTECTED-SEAL: Protected Validation Opens Once

**Given** the final validation partition is sealed before search
**When** either arm selects and evaluates candidates on development evidence
**Then** the arm SHALL NOT read protected targets, protected exact outcomes, or
protected-derived scores
**And** protected validation SHALL open exactly once after both arms terminate
**And** `protected_validation_reuse_count` SHALL be bare `0`.

## REQ-CSL-6319-DENSE-SIGNAL: Development-Only Progress Signal

**Given** a candidate update has development-stream predictions, exact
development outcomes, and movement receipts
**When** Exp6319 computes dense progress
**Then** the signal SHALL use only development-stream evidence available before
protected validation
**And** the signal MAY rank the next candidate
**And** the signal SHALL NOT authorize release
**And** `progress_signal_release_authority_count` SHALL be bare `0`.

## REQ-CSL-6319-MATCHED-ARMS: Candidate And Budget Parity

**Given** repeated sampling and feedback-directed search arms
**When** the searches terminate
**Then** candidate count, update-operation count, exact development-verifier
call count, wall-time ceiling, movement-budget ceiling, and source candidate
pool hash SHALL match across arms
**And** source model weights SHALL remain immutable
**And** `source_model_weight_mutation_count` SHALL be bare `0`.

## REQ-CSL-6319-READY: Protected Improvement Gate

**Given** development search is complete and protected validation has opened
once
**When** `feedback_directed_search_ready_score` is computed
**Then** it SHALL be one only when development dense progress has positive
protected signal predictiveness, feedback-directed search yields more
validated improvements per matched cost than repeated sampling, protected
regression count is no higher than repeated sampling, protected false
discovery count is no higher than repeated sampling, protected validation is
not reused, the dense signal has no release authority, model weights are
unchanged, protected files are unchanged, and verification commands pass.

## REQ-CSL-6319-PROVENANCE: Required Artifact Fields

Exp6319 SHALL emit these fields with the stated principles:

- `status`: Terminal state follows the upstream gate, sealed search, protected evaluation, and verification.
- `paper_source_and_local_claim_boundary`: The fuzz-testing paper is a design cue only. Local claims stop at bounded deterministic candidate updates.
- `upstream_path_hash_and_terminal_class`: Exp6318 is hash-pinned and must be positive before this run executes.
- `structured_gate_receipt`: The upstream gate and local schema gate are replayed before search.
- `candidate_space_schema_and_hash`: The bounded candidate pool is frozen and content-addressed.
- `development_stream_manifest_path_and_hash`: Development evidence is frozen before adaptive selection.
- `protected_validation_manifest_path_and_hash`: Protected rows are committed before search and hide targets.
- `protected_partition_seal_and_access_log`: Protected validation opens once after both arms stop.
- `repeated_sampling_and_feedback_directed_arm_definitions`: Arm roles and selection authority are explicit.
- `dense_progress_signal_definition_and_cost`: The progress score is cheap and development-only.
- `matched_candidate_update_verifier_time_and_movement_budgets`: Candidate count, update work, verifier calls, wall cap, and movement cap match across arms.
- `candidate_lineage_and_intervention_receipts`: Each selected intervention records parent, mutation, arm, and pre-execution reason.
- `development_progress_by_candidate_and_arm`: Development signal rows show the evidence used for ranking.
- `protected_exact_outcomes_by_candidate_and_arm`: Protected exact outcomes open only after search.
- `signal_predictiveness_intervals_and_sample_sizes`: Signal-to-protected-improvement estimates include sample sizes.
- `validated_improvements_false_discoveries_and_regressions_by_arm`: Protected improvements, false discoveries, and regressions stay separated.
- `validated_improvements_per_cost_by_arm`: Protected improvement yield is divided by matched cost.
- `movement_memory_and_wall_time_by_arm`: Movement, memory, and wall time are charged per arm.
- `protected_validation_reuse_count`: Bare zero proves no adaptive reuse of protected validation.
- `progress_signal_release_authority_count`: Bare zero proves the dense signal cannot release candidates.
- `source_model_weight_mutation_count`: Bare zero proves no source model weights changed.
- `feedback_directed_search_ready_score`: Readiness is conjunctive and uses protected exact validation.
- `protected_files_unchanged`: Conductor, ops, traceability, and forbidden files remain byte-identical during the run.
- `preconditions_checked`: Inputs, hashes, seals, budgets, thresholds, seeds, and protected files are frozen first.
- `inference_substrate`: The run declares deterministic exact ASP candidate search with no LLM and no base model load.
- `verifier_is_oracle`: Exact validators are outcome authorities, but the progress signal is not.
- `field_provenance`: Every field maps to spec, inputs, receipts, metrics, tests, commands, or hashes.
- `field_principles`: Every required field carries its guard principle.
- `test_commands`: Focused tests, coverage, full pytest, E2E reading, run command, validation, adversarial checks, and root-clutter checks are listed.
- `test_exit_codes`: Failed verification commands prevent readiness.
- `duration_s`: Wall time is measured without padding.
- `random_seeds`: Candidate, arm, interval, and seal seeds are fixed.
- `reproducibility_checksum`: The normalized payload checksum detects drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states whether feedback direction earned readiness.

## SCENARIO-CSL-6319-PROTECTED-LEAKAGE: Search Cannot See Protected Targets

**Given** protected rows are sealed before search
**When** candidate selection receipts and development progress rows are
inspected
**Then** they SHALL contain no protected target states or protected exact
outcomes.

## SCENARIO-CSL-6319-BUDGET-PARITY: Arms Are Matched

**Given** both arms use the same candidate pool
**When** search completes
**Then** candidate count, update-operation count, development exact-verifier
calls, wall-time ceiling, movement-budget ceiling, and candidate-pool hash
SHALL match.

## SCENARIO-CSL-6319-ONE-TIME-OPEN: Protected Validation Is Not Reused

**Given** both searches have stopped
**When** protected validation opens
**Then** the access log SHALL show one open, zero adaptive reuse, and no
feedback from protected outcomes into later selection.

## SCENARIO-CSL-6319-SIGNAL-TAMPERING: Readiness Fails On Signal Abuse

**Given** an artifact gives release authority to dense progress or feeds
protected validation into progress
**When** readiness is recomputed
**Then** `feedback_directed_search_ready_score` SHALL be `0.0`.

## SCENARIO-CSL-6319-DETERMINISTIC-REPLAY: Same Inputs Reproduce

**Given** the same date, seeds, upstream gate, candidate pool, manifests, and
budgets
**When** Exp6319 reruns
**Then** the normalized reproducibility checksum SHALL match.

## Implementation Status (REQ-CSL-6319)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CSL-6319 | Planned | tests/python/test_experiment_6319_feedback_directed_online_update_search.py |
| REQ-CSL-6319-PROTECTED-SEAL | Planned | tests/python/test_experiment_6319_feedback_directed_online_update_search.py |
| REQ-CSL-6319-DENSE-SIGNAL | Planned | tests/python/test_experiment_6319_feedback_directed_online_update_search.py |
| REQ-CSL-6319-MATCHED-ARMS | Planned | tests/python/test_experiment_6319_feedback_directed_online_update_search.py |
| REQ-CSL-6319-READY | Planned | tests/python/test_experiment_6319_feedback_directed_online_update_search.py |
| REQ-CSL-6319-PROVENANCE | Planned | tests/python/test_experiment_6319_feedback_directed_online_update_search.py |

## REQ-CSL-6320: Online Self-Evolution Safety Audit

**Given** Exp6318 reports versioned factor-local learning evidence and Exp6319
may report positive, null, skipped, or missing feedback-search evidence
**When** Exp6320 audits the V544 learner
**Then** it SHALL independently reconstruct the Exp6318 version registry,
release rows, factor changes, budgets, protected access receipts, and rollback
targets from pinned bytes
**And** it SHALL always audit Exp6318 even when Exp6319 is skipped, null, or
missing
**And** it SHALL write
`results/experiment_6320_online_self_evolution_safety_audit.json`
**And** `undetected_harmful_attack_count`, `unsafe_commit_count`, and
`protected_validation_leak_count` SHALL be bare integer `0`
**And** `utility_claim_allowed` SHALL be bare `false`.

## REQ-CSL-6320-MANIFEST: Frozen Attack Manifest Before Outcome Reads

**Given** audited artifacts, sidecars, protected files, expected attack
decisions, and seeds
**When** Exp6320 starts
**Then** it SHALL hash those inputs and freeze a deterministic attack manifest
before evaluating candidate outcomes or protected validation rows.

## REQ-CSL-6320-GRAPH: Independent Version Registry Reconstruction

**Given** Exp6318 version registry bytes
**When** Exp6320 reconstructs the graph
**Then** it SHALL verify one parent per non-root version, no cycles, no
orphans, deterministic state hashes, changed-factor attribution, task-boundary
activation, matched challenger budgets, and byte-exact rollback targets.

## REQ-CSL-6320-ATTACKS: Safety Attacks Fail Closed

**Given** copied Exp6318 and Exp6319 state
**When** Exp6320 injects false exact passes, pre-outcome leakage, parent
cycles, orphan versions, changed-factor misattribution, version hash swaps,
early activation, task-boundary drift, challenger budget asymmetry,
dense-signal inversion, protected-validation reads, validation reuse, poison,
reversal, forgetting, negative transfer, corrupted snapshots, restart faults,
and rollback failure
**Then** every harmful candidate SHALL reject, quarantine, abort, or roll back
**And** no attacked candidate SHALL become active.

## REQ-CSL-6320-PROTECTED: Protected Validation Remains Sealed

**Given** Exp6319 is positive, null, skipped, or missing
**When** Exp6320 audits protected validation
**Then** missing evidence, protected reads before search stop, protected reuse,
and protected-derived dense progress SHALL fail closed
**And** missing Exp6319 evidence SHALL NOT count as safety success.

## REQ-CSL-6320-ROLLBACK: Parent Rollback Is Byte Exact After Restart

**Given** attacked snapshots, restarts, corrupted state bytes, or rollback
faults
**When** Exp6320 restores an active version
**Then** the restored bytes and hash SHALL match the exact parent bytes after
restart.

## REQ-CSL-6320-BOUNDARY: Safety Cannot Promote Utility

**Given** every safety attack fails closed
**When** Exp6320 computes readiness
**Then** safety success SHALL produce only a safety readiness score
**And** it SHALL NOT promote Exp6318 or Exp6319 utility claims.

## REQ-CSL-6320-PROVENANCE: Required Artifact Fields

Exp6320 SHALL emit these fields with the stated principles:

- `status`: Terminal state follows frozen inputs, graph reconstruction, attack outcomes, protected validation, and rollback identity.
- `audited_paths_hashes_and_terminal_classes`: Exp6306, Exp6318, Exp6319, sidecars, source, verifier, spec, E2E plan, and protected files are hash-pinned.
- `exp6306_safety_baseline_receipt`: Exp6306 copied-state safety evidence is carried as a baseline, not as proof for V544.
- `independent_version_registry_reconstruction`: Exp6318 version graph, parents, factors, releases, budgets, protected access receipts, and rollback targets are reconstructed from bytes.
- `injection_manifest_path_and_hash`: The attack manifest records expected fail-closed decisions and seeds before outcome reads.
- `false_pass_preoutcome_leakage_parent_cycle_orphan_and_hash_swap_results`: False passes, pre-outcome leakage, parent cycles, orphan versions, and hash swaps reject or roll back.
- `factor_attribution_and_version_lineage_results`: Changed-factor misattribution and lineage tampering reject without activating a candidate.
- `early_activation_boundary_drift_and_budget_asymmetry_results`: Early activation, boundary drift, and challenger budget asymmetry fail closed.
- `dense_signal_inversion_and_release_authority_results`: Dense-signal inversion and release-authority abuse reject because dense progress is not release authority.
- `protected_validation_access_and_reuse_results`: Protected reads, validation reuse, and missing protected evidence fail closed.
- `poison_reversal_forgetting_and_negative_transfer_results`: Poison, reversal, forgetting, and negative-transfer injections do not commit harmful updates.
- `snapshot_corruption_restart_and_parent_rollback_results`: Corrupted snapshots, restart faults, and rollback failures restore exact parent bytes.
- `fail_closed_decisions_by_attack`: Each preregistered attack records reject, quarantine, abort, or rollback.
- `undetected_harmful_attack_count`: Bare zero proves no harmful injected attack escaped detection.
- `unsafe_commit_count`: Bare zero proves no unsafe attacked candidate committed.
- `protected_validation_leak_count`: Bare zero proves protected validation did not leak into adaptive selection.
- `rollback_byte_identity`: Parent rollback receipts prove restored bytes and hashes match after restart.
- `utility_claim_allowed`: Bare false proves safety success cannot promote utility.
- `online_self_evolution_safety_ready_score`: Safety readiness is one only when reconstruction, fail-closed attacks, protected seals, rollback identity, protected files, and tests pass.
- `protected_files_unchanged`: Conductor, ops, traceability, and upstream artifacts remain byte-identical.
- `preconditions_checked`: Inputs, hashes, manifests, expected decisions, seeds, protected files, and protected evidence policy are frozen first.
- `inference_substrate`: The run declares deterministic replay and artifact audit with no LLM and no base model load.
- `verifier_is_oracle`: Exact validators are outcome authorities, but this audit is not a utility oracle.
- `field_provenance`: Every field maps to spec, upstream bytes, reconstruction receipts, attack receipts, tests, commands, or hashes.
- `field_principles`: Every required field carries its guard principle.
- `test_commands`: Focused tests, coverage, global pytest, spec coverage, run command, validation, adversarial verification, E2E reading, and root-clutter checks are listed.
- `test_exit_codes`: Failed commands prevent safety readiness.
- `duration_s`: Wall time is measured without padding.
- `random_seeds`: Reconstruction, manifest, and attack seeds are fixed.
- `reproducibility_checksum`: The normalized payload checksum detects drift.
- `honest_verdict`: The verdict starts with a terminal prefix and separates safety closure from utility promotion.

## SCENARIO-CSL-6320-MANIFEST: Attacks Are Preregistered

**Given** Exp6320 starts
**When** the manifest is written
**Then** every attack class, expected terminal decision, protected-file hash,
seed, and audited input hash SHALL be frozen before protected outcomes are
inspected.

## SCENARIO-CSL-6320-GRAPH: Version Tampering Rejects

**Given** a copied registry row has a parent cycle, orphan parent, hash swap,
or changed-factor mismatch
**When** Exp6320 validates lineage
**Then** the candidate SHALL fail closed and SHALL NOT become active.

## SCENARIO-CSL-6320-PROTECTED: Missing Or Reused Protected Evidence Fails

**Given** Exp6319 is skipped, null, missing, leaked, or reused
**When** Exp6320 audits the protected partition
**Then** it SHALL record safe closure and SHALL NOT count missing protected
evidence as readiness evidence.

## SCENARIO-CSL-6320-ROLLBACK: Restart Restores Parent Bytes

**Given** rollback targets are reconstructed from registry rows
**When** snapshots or active bytes are corrupted before restart
**Then** rollback SHALL restore the exact parent bytes and hash.

## SCENARIO-CSL-6320-UTILITY: Safety Does Not Promote Utility

**Given** every attack is safely closed
**When** Exp6320 reports readiness
**Then** `utility_claim_allowed` SHALL remain `false` and no utility readiness
field SHALL be promoted by safety evidence.

## Implementation Status (REQ-CSL-6320)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CSL-6320 | Planned | tests/python/test_experiment_6320_online_self_evolution_safety_audit.py |
| REQ-CSL-6320-MANIFEST | Planned | tests/python/test_experiment_6320_online_self_evolution_safety_audit.py |
| REQ-CSL-6320-GRAPH | Planned | tests/python/test_experiment_6320_online_self_evolution_safety_audit.py |
| REQ-CSL-6320-ATTACKS | Planned | tests/python/test_experiment_6320_online_self_evolution_safety_audit.py |
| REQ-CSL-6320-PROTECTED | Planned | tests/python/test_experiment_6320_online_self_evolution_safety_audit.py |
| REQ-CSL-6320-ROLLBACK | Planned | tests/python/test_experiment_6320_online_self_evolution_safety_audit.py |
| REQ-CSL-6320-BOUNDARY | Planned | tests/python/test_experiment_6320_online_self_evolution_safety_audit.py |
| REQ-CSL-6320-PROVENANCE | Planned | tests/python/test_experiment_6320_online_self_evolution_safety_audit.py |

## REQ-LEARN-6342: Anytime E-Value Release Ledger

**Given** Exp6318 and Exp6319 report positive factor-local online-learning
evidence
**When** Exp6342 evaluates factor-local release hypotheses under repeated
looks, optional stopping, restarts, and many proposed factors
**Then** it SHALL write
`results/experiment_6342_anytime_evalue_release_ledger.json`
**And** it SHALL freeze the null family, alternatives, evidence identity,
filtration, betting rule, alpha budget, multiplicity policy, release
threshold, stream sizes, seeds, resource limits, exact safety guard, and
protected hashes before outcome processing
**And** generated labels, LLM calls, duplicate evidence, cross-factor evidence
reuse, unsafe statistical releases, and source model mutation SHALL remain bare
integer `0`.

## REQ-LEARN-6342-LEDGER: Append-Only Canonical E-Process Events

**Given** a factor-local hypothesis has a frozen predecision hash
**When** evidence events arrive
**Then** each accepted row SHALL carry a sequence number, previous row hash,
canonical event hash, evidence identity, filtration time, hypothesis id,
factor id, nonnegative e-value increment, cumulative e-value, exact-safety
receipt, and release decision
**And** replay from the JSONL bytes SHALL reconstruct byte-identical state.

## REQ-LEARN-6342-VALIDITY: Null Error Is Anytime Valid

**Given** deterministic synthetic null streams with outcome probability
bounded by the frozen null family
**When** optional stopping or repeated looks stop on the first threshold
crossing
**Then** the empirical type-I interval SHALL stay inside the preregistered
alpha bound after the multiplicity policy is applied.

## REQ-LEARN-6342-POWER: Alternative Streams Clear The Gate

**Given** deterministic synthetic alternative streams with a frozen effect
size above the null boundary
**When** the same e-process, threshold, and exact safety guard are applied
**Then** power SHALL clear the preregistered lower threshold
**And** the release-delay distribution SHALL report first-crossing look
counts.

## REQ-LEARN-6342-ATTACKS: Adaptive Evidence Attacks Fail Closed

**Given** duplicated rows, cross-factor evidence reuse, selected hypotheses,
reordered events, reset attempts, truncation, row mutation, previous-hash
breaks, and restart corruption
**When** Exp6342 replays or appends the attacked evidence
**Then** each attack SHALL reject, quarantine, abort, or refuse release
**And** no attack SHALL produce a released factor.

## REQ-LEARN-6342-GUARD: E-Values Cannot Bypass Exact Safety

**Given** a hypothesis crosses the statistical e-value threshold
**When** the exact safety guard rejects its evidence or source contract
**Then** the release SHALL fail closed
**And** readiness SHALL remain `0.0`.

## REQ-LEARN-6342-PROVENANCE: Required Artifact Fields

Exp6342 SHALL emit these fields with the stated principles:

- `status`: Terminal state follows preregistration, e-process validity, attack closure, restart identity, exact guard, and tests.
- `source_claim_boundary`: NxN E-valuation is a design cue only. Local claims stop at deterministic factor-local release certification.
- `evalue_ledger_path_and_hash`: The append-only JSONL ledger is content-addressed so replay starts from bytes, not memory.
- `ledger_schema_path_and_hash`: The frozen schema fixes row identity, hash chaining, and replay validation.
- `null_family_and_assumptions`: The null family states the composite boundary that makes the e-process a supermartingale.
- `filtration_and_evidence_identity_contract`: Evidence IDs, factor scope, and filtration time prevent optional-stopping leakage and duplicate reuse.
- `betting_rule_and_predecision_hash`: The betting rule and predecision hash are frozen before outcomes so the test is not fitted after seeing labels.
- `alpha_multiplicity_and_release_policy`: Alpha spending, multiplicity, and the release threshold are explicit and data-independent.
- `exact_safety_guard_contract`: Statistical evidence cannot release a factor unless the exact oracle safety guard also passes.
- `synthetic_stream_manifest_path_and_hash`: Null and alternative stream seeds, sizes, probabilities, and resource limits are frozen.
- `null_stream_results`: Null streams report threshold crossings and the empirical error used for readiness.
- `alternative_stream_results`: Alternative streams report power under the same frozen ledger and guard.
- `optional_stopping_results`: First-crossing stops prove repeated looks do not inflate release beyond the bound.
- `repeated_look_results`: Fixed and repeated-look summaries stay separated for audit.
- `duplicate_cross_factor_reorder_and_selection_attack_results`: Duplicate rows, cross-factor reuse, event reorder, and selected hypotheses fail closed.
- `restart_reconstruction_results`: Restart replay reproduces the same state, hashes, and release decisions.
- `append_only_tamper_results`: Truncation, row mutation, previous-hash breaks, reset, and restart corruption are detected.
- `type_i_error_interval_and_sample_size`: The type-I interval and sample size justify the null-error claim.
- `power_interval_and_sample_size`: The power interval and sample size justify the alternative claim.
- `release_delay_distribution`: First-release look counts show how long valid evidence took to cross the gate.
- `eprocess_state_examples`: Example states make the nonnegative e-value ledger auditable without replaying all streams.
- `exact_oracle_claim_boundary`: The exact checker is the outcome oracle, so the result is execution-grounded and not oracle-distinct.
- `generated_label_count`: Bare zero proves no generated labels were used.
- `llm_call_count`: Bare zero proves no LLM call was made.
- `anytime_release_certificate_ready_score`: Readiness is one only when null error, power, attacks, restart identity, exact guard, protected files, and tests pass.
- `protected_files_unchanged`: Conductor, ops, traceability, and upstream evidence files remain byte-identical.
- `preconditions_checked`: Inputs, source hashes, protected hashes, nulls, alternatives, evidence contract, betting rule, alpha, threshold, guard, seeds, stream sizes, and resource limits freeze first.
- `inference_substrate`: The run declares deterministic synthetic replay plus exact oracle checks with no LLM or base model load.
- `verifier_is_oracle`: Bare true states that exact safety and outcome checks are the oracle.
- `field_provenance`: Every field maps to spec, source artifacts, sidecars, streams, attacks, tests, or hashes.
- `field_principles`: Every required field carries its guard principle.
- `test_commands`: Focused tests, coverage, full pytest, spec coverage, run command, validation, adversarial verification, E2E reading, and root-clutter checks are listed.
- `test_exit_codes`: Failed commands prevent readiness.
- `duration_s`: Wall time is measured without padding.
- `random_seeds`: Null, alternative, ledger, attack, and interval seeds are fixed.
- `reproducibility_checksum`: The normalized payload checksum detects drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states whether the anytime release certificate is ready.

## SCENARIO-LEARN-6342-OPTIONAL-STOPPING: Null Peeking Stays Bounded

**Given** null streams are checked after every event
**When** the run stops at the first crossing
**Then** empirical type-I error SHALL remain inside the preregistered bound.

## SCENARIO-LEARN-6342-REPLAY: Ledger Replay Is Byte-Identical

**Given** the append-only ledger JSONL bytes
**When** restart reconstruction replays every row
**Then** the final state hash, release rows, and ledger digest SHALL match the
original run.

## SCENARIO-LEARN-6342-ATTACKS: Evidence Abuse Fails Closed

**Given** duplicate, cross-factor, selected-hypothesis, reordered, reset, and
tampered evidence
**When** the ledger append or replay path evaluates it
**Then** the attack SHALL fail closed with no release.

## SCENARIO-LEARN-6342-EXACT-GUARD: Statistical Evidence Is Not Sufficient

**Given** a statistical e-value crosses the release threshold
**When** the exact safety guard rejects the candidate
**Then** the ledger SHALL refuse release.

## Implementation Status (REQ-LEARN-6342)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6342 | Implemented | tests/python/test_experiment_6342_anytime_evalue_release_ledger.py |
| REQ-LEARN-6342-LEDGER | Implemented | tests/python/test_experiment_6342_anytime_evalue_release_ledger.py |
| REQ-LEARN-6342-VALIDITY | Implemented | tests/python/test_experiment_6342_anytime_evalue_release_ledger.py |
| REQ-LEARN-6342-POWER | Implemented | tests/python/test_experiment_6342_anytime_evalue_release_ledger.py |
| REQ-LEARN-6342-ATTACKS | Implemented | tests/python/test_experiment_6342_anytime_evalue_release_ledger.py |
| REQ-LEARN-6342-GUARD | Implemented | tests/python/test_experiment_6342_anytime_evalue_release_ledger.py |
| REQ-LEARN-6342-PROVENANCE | Implemented | tests/python/test_experiment_6342_anytime_evalue_release_ledger.py |

## REQ-LEARN-6343: Evidence-Carrying Factor Lifecycle

**Given** Exp6342 reports an anytime e-value release ledger with readiness
`1.0`
**When** Exp6343 processes learned factor lifecycle events
**Then** it SHALL write
`results/experiment_6343_evidence_carrying_factor_lifecycle.json`
**And** every learned factor SHALL carry a rationale, minimized exact
counterexample, replay witness, parent version, affected variables, release
certificate, retention set, and rollback target
**And** source model weight mutation, generated label count, and LLM call count
SHALL be bare integer `0`.

## REQ-LEARN-6343-EVIDENCE: Canonical Evidence Bundles

**Given** a factor lifecycle event proposes retain, merge, quarantine, delete,
or restore
**When** the event is validated
**Then** its canonical evidence bundle SHALL bind rationale, counterexample,
replay witness, lineage, affected variables, release certificate, retention
set, and rollback target into one hash
**And** rationale-only evidence SHALL reject before state mutation.

## REQ-LEARN-6343-LIFECYCLE: Deterministic Operations

**Given** active and quarantined factors are stored in versioned state
**When** retain, merge, quarantine, delete, restore, or capacity eviction runs
**Then** the operation SHALL be deterministic, append-only, hash-chained, and
replayable from the version registry bytes.

## REQ-LEARN-6343-GATES: Replay, Retention, And Rollback Gate Merge And Delete

**Given** a merge or delete event has a valid release certificate
**When** Exp6343 considers the event
**Then** exact historical replay, protected retention, and byte-identical
rollback checks SHALL all pass before the state changes
**And** stale, circular, cross-family, duplicate, witness-swapped, harmful, or
rationale-laundered evidence SHALL fail closed.

## REQ-LEARN-6343-BOUNDS: Active And Quarantine Counts Are Capped

**Given** lifecycle events exceed the active or quarantine capacity
**When** compaction runs
**Then** the oldest unprotected factor SHALL move first by deterministic key
order
**And** active and quarantined counts SHALL never exceed the frozen bounds.

## REQ-LEARN-6343-RESTART: Restart And Rollback Are Byte-Exact

**Given** the version registry, lifecycle schemas, stream manifest, and rollback
targets
**When** Exp6343 restarts from disk and rolls back each permitted change
**Then** final state bytes, state hash, registry hash, and rollback target bytes
SHALL match byte-for-byte.

## REQ-LEARN-6343-PROVENANCE: Required Artifact Fields

Exp6343 SHALL emit these fields with the stated principles:

- `status`: Terminal state follows replay, retention, rollback, bounds, attacks, protected files, and tests.
- `upstream_release_ledger_path_hash_and_ready_score`: Exp6342 readiness and ledger bytes are replayed before lifecycle credit.
- `factor_lifecycle_schema_path_and_hash`: The frozen lifecycle schema fixes state and registry row identity.
- `evidence_bundle_schema_path_and_hash`: The evidence schema keeps rationale tied to exact removable evidence.
- `rationale_counterexample_replay_lineage_and_retention_contract`: Learned factors are removable only because their rationale, counterexample, witness, lineage, retention, and rollback evidence stay linked.
- `retain_merge_quarantine_delete_and_restore_rules`: Operation rules state the deterministic lifecycle semantics.
- `active_and_quarantine_capacity_bounds`: Bounded counts prevent unbounded remembering.
- `version_registry_path_and_hash`: The append-only registry is the replay source of truth.
- `synthetic_lifecycle_stream_manifest_path_and_hash`: The deterministic stream manifest freezes operations, attacks, seeds, and limits.
- `factor_add_merge_delete_quarantine_and_restore_results`: Lifecycle results prove every required operation executed.
- `exact_historical_replay_results`: Historical replay gates each state change.
- `protected_retention_results`: Protected factors and cases cannot regress.
- `bounded_memory_growth_results`: Active and quarantine counts stay within capacity under compaction.
- `stale_circular_cross_family_duplicate_and_rationale_laundering_attack_results`: Invalid evidence classes fail closed before mutation.
- `restart_and_byte_exact_rollback_results`: Restart and rollback compare canonical bytes, not summaries.
- `catastrophic_remembering_event_definition_and_counts`: The event definition counts persistent stale or harmful factors that survive removal evidence.
- `source_model_weight_mutation_count`: Bare zero proves no base model weight changed.
- `generated_label_count`: Bare zero proves no generated labels were used.
- `llm_call_count`: Bare zero proves no LLM call was made.
- `exact_oracle_claim_boundary`: The exact checker is the outcome oracle, so the result is execution-grounded.
- `evidence_factor_lifecycle_ready_score`: Readiness is one only when lifecycle, replay, retention, bounds, attacks, rollback, protected files, and tests pass.
- `protected_files_unchanged`: Conductor, ops, traceability, and upstream evidence remain byte-identical.
- `preconditions_checked`: Upstream readiness, schemas, operations, bounds, replay sets, retention sets, attacks, seeds, limits, and protected hashes freeze first.
- `inference_substrate`: The substrate declares deterministic lifecycle replay with exact oracle checks and no LLM.
- `verifier_is_oracle`: Bare true states that exact replay and retention checks are the oracle.
- `field_provenance`: Every field maps to spec, upstream artifacts, sidecars, registry rows, attacks, tests, or hashes.
- `field_principles`: Every required field carries its guard principle.
- `test_commands`: Focused tests, coverage, full pytest, spec coverage, run command, validation, adversarial verification, E2E reading, and root-clutter checks are listed.
- `test_exit_codes`: Failed commands prevent readiness.
- `duration_s`: Wall time is measured without padding.
- `random_seeds`: Lifecycle, attack, rollback, and capacity seeds are fixed.
- `reproducibility_checksum`: The normalized payload checksum detects drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states whether evidence-carrying lifecycle is ready.

## SCENARIO-LEARN-6343-LIFECYCLE: All Operations Replay

**Given** the synthetic lifecycle stream contains add, retain, merge,
quarantine, delete, restore, and capacity eviction
**When** the version registry is replayed
**Then** the final active and quarantined factors SHALL match the original
state bytes exactly.

## SCENARIO-LEARN-6343-GATED-MERGE-DELETE: Destructive Changes Need Three Gates

**Given** merge and delete events carry release certificates
**When** exact replay, protected retention, or byte rollback fails
**Then** the operation SHALL reject without mutating lifecycle state.

## SCENARIO-LEARN-6343-ATTACKS: Evidence Laundering Fails Closed

**Given** stale certificates, parent cycles, cross-family bundles, duplicate
evidence rows, witness swaps, rationale-only bundles, harmful merge, and
harmful deletion
**When** the lifecycle engine validates them
**Then** every attack SHALL fail closed and no attack SHALL change state.

## SCENARIO-LEARN-6343-BOUNDED: Compaction Is Deterministic

**Given** active or quarantine counts exceed capacity
**When** compaction chooses a candidate
**Then** it SHALL choose the oldest unprotected factor, tie-broken by factor id,
and SHALL preserve protected retention.

## SCENARIO-LEARN-6343-RESTART-ROLLBACK: Bytes Match After Restart

**Given** committed lifecycle registry bytes and rollback targets
**When** Exp6343 restores from disk and rolls back each destructive event
**Then** restart bytes and rollback bytes SHALL match the original canonical
bytes.

## Implementation Status (REQ-LEARN-6343)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6343 | Planned | tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py |
| REQ-LEARN-6343-EVIDENCE | Planned | tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py |
| REQ-LEARN-6343-LIFECYCLE | Planned | tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py |
| REQ-LEARN-6343-GATES | Planned | tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py |
| REQ-LEARN-6343-BOUNDS | Planned | tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py |
| REQ-LEARN-6343-RESTART | Planned | tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py |
| REQ-LEARN-6343-PROVENANCE | Planned | tests/python/test_experiment_6343_evidence_carrying_factor_lifecycle.py |

## REQ-LEARN-6344: Counterexample Factor Proposal Calibration

**Given** Exp6319 reports protected improvement from feedback-directed search
and Exp6342 and Exp6343 report ready exact release and lifecycle gates
**When** Exp6344 compares bounded factor-edit proposal arms
**Then** it SHALL write
`results/experiment_6344_counterexample_factor_proposal_calibration.json`
**And** it SHALL build `MODEL_SPECS` from
`cached_sota_pair(gpu_indices=(0, 1))` plus the dense Gemma pair
**And** source model weight mutation, generated label count, protected
validation leak count, and hidden-state access count SHALL be bare integer
`0`
**And** `verifier_is_oracle` SHALL be bare `true`.

## REQ-LEARN-6344-SCHEMA: Bounded Factor Edits Are Frozen

**Given** the factor-edit schema, event manifest, minimized counterexamples,
allowed variables, edit bounds, arms, budgets, and primary endpoint
**When** proposals are generated or replayed
**Then** every proposal SHALL validate against the frozen schema
**And** a proposal that touches another factor, a forbidden variable, or an
out-of-bound step SHALL reject before selection.

## REQ-LEARN-6344-ISOLATION: Only Counterexample Feedback Is Visible

**Given** a development event has an exact violated factor
**When** a model-family arm proposes candidate edits
**Then** the exposed prompt payload SHALL include only the changed factor,
minimized exact counterexample, allowed variables, and edit bounds
**And** protected exact outcomes SHALL stay sealed until after selection.

## REQ-LEARN-6344-MATCHING: Proposal Arms Use Matched Budgets

**Given** random valid edits, repeated temperature sampling,
stability-regularized proposals, and counterexample-directed proposals
**When** Exp6344 compares arms
**Then** calls, token budgets, candidate counts, time budgets, and exact-check
cost budgets SHALL match across arms for every model family.

## REQ-LEARN-6344-SINGLE-OPEN: Protected Outcomes Open Once

**Given** selected candidates are chosen only from development-visible
information
**When** Exp6344 evaluates protected outcomes
**Then** it SHALL open the protected exact outcome seal once
**And** every model-family arm SHALL report protected exact success, movement,
and exact-check cost from that single open.

## REQ-LEARN-6344-ORACLE-BOUNDARY: Exact Checkers Keep Authority

**Given** local SOTA models propose bounded edits
**When** proposal quality is scored
**Then** exact checkers SHALL supply all labels and release authority
**And** model outputs, generated labels, hidden states, and model-weight
updates SHALL NOT become an oracle.

## REQ-LEARN-6344-PROVENANCE: Required Artifact Fields

Exp6344 SHALL emit these fields with the stated principles:

- `status`: Terminal state follows proposal success, locality, single-open, protected files, tests, and exact cost checks.
- `upstream_paths_hashes_terminal_classes_and_ready_scores`: Upstream Exp6319, Exp6342, and Exp6343 bytes and ready scores are replayed first.
- `MODEL_SPECS`: The three mandated GGUF model rows are resolved through cached SOTA helper calls.
- `models_used`: Names the model ids that supplied bounded proposal rows.
- `model_file_hashes_revisions_quantizations_and_tokenizers`: Pins model files, snapshot revisions, quantizations, tokenizer method, and file hashes.
- `llama_cpp_embedded_tokenizer_receipts`: Proves tokenizer checks used embedded GGUF metadata through llama.cpp.
- `cuda_gpu_offload_and_memory_release_receipts_by_model`: Records GPU offload and per-model release receipts before and after generation.
- `factor_edit_schema_path_and_hash`: Freezes the bounded factor-edit schema.
- `development_event_manifest_path_and_hash`: Freezes development events, split hashes, seeds, budgets, and protected seal hashes.
- `counterexample_minimizer_path_hash_and_exactness`: Pins the minimizer and proves each counterexample is exact and minimal.
- `information_exposure_contract`: Defines the only fields visible to the proposer.
- `arm_definitions`: Defines random, repeated sampling, stability, and counterexample-directed proposal arms.
- `matched_call_token_candidate_time_and_checker_budgets`: Proves budget parity across all arms.
- `raw_proposal_paths_hashes_and_counts`: Pins raw proposal rows and counts before exact scoring.
- `schema_validity_and_factor_locality_results`: Reports schema validity, factor locality, variable locality, and edit-bound failures.
- `exact_proposal_success_cost_and_movement_by_model_family_arm`: Reports exact success, checker cost, and movement per model family and arm.
- `protected_outcome_seal_and_single_open_receipt`: Shows protected outcomes opened once after selection.
- `paired_deltas_intervals_and_sample_sizes`: Reports preregistered paired deltas against repeated sampling.
- `verification_calls_time_cost_and_error_table`: Reports checker calls, checker time, cost, and errors.
- `harm_underpowered_missing_and_flagged_cells`: Keeps missing, underpowered, harmful, or flagged cells visible.
- `protected_validation_leak_count`: Bare zero proves no protected outcome leaked before selection.
- `source_model_weight_mutation_count`: Bare zero proves source model weights were not updated.
- `generated_label_count`: Bare zero proves generated labels did not enter scoring.
- `hidden_state_access_count`: Bare zero proves hidden activations did not enter scoring.
- `exact_oracle_claim_boundary`: States that exact checkers are the oracle and release authority.
- `counterexample_proposal_ready_score`: Readiness is one only when counterexample-directed proposals beat repeated sampling per matched cost in every required family and all checks pass.
- `protected_files_unchanged`: Shows conductor, ops, traceability, and upstream files stayed byte-identical.
- `preconditions_checked`: Freezes upstream readiness, GGUF files, embedded tokenizers, GPUs, VRAM, RAM, disk, timeouts, seeds, event hashes, budgets, and protected hashes.
- `inference_substrate`: Declares local GGUF llama.cpp proposal generation with exact checking.
- `verifier_is_oracle`: Bare true preserves the exact checker as authority.
- `field_provenance`: Maps every field to specs, inputs, sidecars, model receipts, tests, or exact checks.
- `field_principles`: Explains why every required field exists.
- `test_commands`: Lists run, focused, coverage, global, spec, E2E, and adversarial commands.
- `test_exit_codes`: Prevents failed commands from becoming readiness.
- `duration_s`: Reports measured wall time without padding.
- `random_seeds`: Pins deterministic proposal and split schedules.
- `reproducibility_checksum`: Detects artifact drift.
- `honest_verdict`: States the terminal claim boundary with a terminal prefix.

## SCENARIO-LEARN-6344-LOCALITY: Invalid Factor Edits Reject

**Given** a proposal changes the wrong factor, an unlisted variable, or an
out-of-bound edit
**When** the schema validator checks the proposal
**Then** the proposal SHALL be invalid and SHALL NOT be selected.

## SCENARIO-LEARN-6344-ISOLATION: Protected Outcomes Stay Sealed

**Given** a development event is rendered for a proposer
**When** the information exposure contract is applied
**Then** only changed factor, minimized counterexample, allowed variables, and
edit bounds are present.

## SCENARIO-LEARN-6344-MATCHED-BUDGETS: Arms Are Budget Matched

**Given** every model family runs every arm
**When** budget receipts are computed
**Then** call, token, candidate, time, and checker budgets SHALL be identical.

## SCENARIO-LEARN-6344-SINGLE-OPEN: Selection Precedes Protected Validation

**Given** selected proposals are fixed
**When** protected outcomes open
**Then** the open count SHALL be one and no selected row SHALL cite protected
outcomes before selection.

## SCENARIO-LEARN-6344-READY: Counterexamples Beat Repeated Sampling

**Given** all checks pass
**When** counterexample-directed proposals improve exact protected success per
matched cost over repeated sampling for every required model family
**Then** `counterexample_proposal_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6344)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6344 | Planned | tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py |
| REQ-LEARN-6344-SCHEMA | Planned | tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py |
| REQ-LEARN-6344-ISOLATION | Planned | tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py |
| REQ-LEARN-6344-MATCHING | Planned | tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py |
| REQ-LEARN-6344-SINGLE-OPEN | Planned | tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py |
| REQ-LEARN-6344-ORACLE-BOUNDARY | Planned | tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py |
| REQ-LEARN-6344-PROVENANCE | Planned | tests/python/test_experiment_6344_counterexample_factor_proposal_calibration.py |

## REQ-LEARN-6380: Three-Family Canonical Factor Transport Canary

**Given** Exp6379 reports the deterministic canonical transport contract ready
for run date 20260813
**When** Exp6380 starts
**Then** it SHALL write
`results/experiment_6380_three_family_canonical_factor_transport_canary.json`
**And** it SHALL revalidate the Exp6379 artifact, both RTX 3090 GPUs, model
hashes, embedded GGUF tokenizers, llama.cpp GPU offload, disk, source hashes,
and event-manifest hashes before live generation.

Exp6380 SHALL seal at least 12 fresh licensed executable events across at least
three constraint families. The event set SHALL balance executable structure and
surface relabeling. It SHALL use exactly these local GGUF model ids from
`cached_sota_pair()`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. `AutoTokenizer` SHALL not be called.

Exp6380 SHALL preregister three matched arms: frozen Exp6366 prompt at 192
tokens, canonical prompt at 192 tokens, and canonical prompt at the computed
per-model allowance with the fixed repetition policy. Apart from prompt,
completion allowance, and the preregistered repetition policy, sampling inputs
SHALL stay fixed.

Exp6380 SHALL run each required model sequentially through the Exp6365
observable child-process contract. It SHALL preserve raw stdout and stderr
before parsing. It SHALL not use grammar decoding, parser retry, JSON repair,
hidden states, generated labels, model-weight mutation, protected validation
reads, or an external scorer.

Exp6380 SHALL parse each raw output once with the canonical validator. It SHALL
freeze source-span alignment for parse-valid objects and only then call exact
task checkers. Exact checkers are the only correctness oracle. Output
transport, parsing, and model proposals are not oracles.

Exp6380 SHALL set `three_family_factor_transport_ready_score=1.0` only when
each constraint family has at least one nonempty source-bound parse-valid
canonical-capacity-arm object, each family has at least one exact-checker call,
protected leakage is zero, same-step writes are zero, runtime receipts are
complete, protected files are unchanged, and all recorded tests pass. If the
same all-invalid verdict recurs, the retry scope SHALL retire.

Exp6380 SHALL emit these fields with explicit principles:

- `status`: Terminal status separates blocked, positive, null, and retired transport evidence.
- `exp6379_gate_receipt`: The deterministic transport contract is revalidated before live calls.
- `MODEL_SPECS`: The three mandated GGUF model rows come from cached SOTA helper calls.
- `models_used`: Only models with authenticated runtime receipts count as used.
- `cached_sota_pair_receipts`: Helper-call receipts prevent manual model substitution.
- `model_file_hashes_revisions_quantizations_and_tokenizers`: Model file identity and tokenizer method are pinned.
- `embedded_gguf_tokenizer_receipts`: Tokenizer receipts use only embedded GGUF tokenizers.
- `autotokenizer_usage_count`: Bare zero proves no external tokenizer path was used.
- `cuda_offload_and_runtime_receipts_by_model`: CUDA offload, timing, token usage, return, raw streams, and cleanup are reported.
- `sealed_event_manifest_path_hash_license_and_balance`: Fresh licensed events are sealed before prompting.
- `canonical_schema_path_hash_and_drift_receipt`: The canonical schema source is hash-bound and checked for drift.
- `preregistered_arm_contract`: The three arms and fixed sampling differences are frozen before generation.
- `per_arm_prompt_output_and_context_capacity_receipts`: Prompt tokens, output allowance, and context margin are recorded per call.
- `raw_output_before_parse_paths_hashes_and_counts`: Raw outputs are frozen before classification or parsing.
- `failure_taxonomy_counts_by_model_and_arm`: Failure labels distinguish thinking, repetition, truncation, syntax, structure, source, semantic, timeout, and abstention.
- `parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm`: Parse outcomes stay separate from exact correctness.
- `source_span_alignment_and_conflict_counts`: Source-bound spans and conflicts are counted before exact checking.
- `exact_checker_paths_versions_calls_costs_and_errors`: Exact checker identity, calls, costs, and errors are recorded.
- `exact_pass_fail_counts_by_model_and_arm`: Exact pass and fail counts stay separate from transport readiness.
- `same_step_read_write_isolation_results`: Same-step writes and protected reads remain invisible.
- `retired_decoding_mechanism_usage_count`: Bare zero proves retired decode helpers were not used.
- `three_family_factor_transport_ready_score`: Readiness is a conjunctive transport gate.
- `semantic_utility_not_implied_by_transport`: The artifact states that transport readiness is not semantic utility.
- `harm_underpowered_missing_and_flagged_cells`: Missing, invalid, timeout, abstain, underpowered, and retired cells stay visible.
- `protected_files_unchanged`: Protected files remain byte-identical.
- `preconditions_checked`: Preconditions freeze upstream, model, tokenizer, GPU, disk, schema, event, source, and protected hashes.
- `inference_substrate`: The substrate declares local llama.cpp GGUF child-process generation.
- `verifier_is_oracle`: Bare true applies only to exact task checkers.
- `field_principles`: Every required field states its guard.
- `field_provenance`: Every required field maps to specs, inputs, sidecars, model receipts, tests, or exact checks.
- `random_seed`: Fixed seeds pin schedule and prompt construction.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states the transport boundary.

## SCENARIO-LEARN-6380-GATE: Exp6379 Gates Live Work

**Given** the Exp6379 artifact is missing, not ready, or hash-drifted
**When** Exp6380 computes preconditions
**Then** live generation SHALL not qualify readiness
**And** the terminal verdict SHALL start with `blocked:` or `retired:`.

## SCENARIO-LEARN-6380-ARMS: Three Arms Are Preregistered

**Given** the selected source events and model rows
**When** Exp6380 builds prompts
**Then** it SHALL produce the frozen Exp6366 192-token control, canonical
192-token control, and canonical computed-allowance arm with fixed sampling
inputs except for the preregistered differences.

## SCENARIO-LEARN-6380-RAW: Raw Output Freezes Before Parsing

**Given** a child process returns stdout and stderr
**When** Exp6380 classifies and parses the row
**Then** raw path, byte count, hash, and parse-start time SHALL prove the raw
bytes existed before parsing.

## SCENARIO-LEARN-6380-ORACLE: Exact Checkers Alone Are Oracles

**Given** a raw output is parse-valid and source-bound
**When** Exp6380 evaluates correctness
**Then** it SHALL call the exact task checker after raw freeze
**And** it SHALL state that transport, parsing, and model proposals are not
oracles.

## SCENARIO-LEARN-6380-READY: Each Family Must Produce Source-Bound Transport

**Given** all required models finish their assigned arms
**When** readiness is computed
**Then** `three_family_factor_transport_ready_score` SHALL be `1.0` only if
each constraint family has a nonempty source-bound parse-valid canonical
capacity-arm object and an exact-checker call with zero protected leakage,
zero same-step writes, complete runtime receipts, unchanged protected files,
and passing tests.

## Implementation Status (REQ-LEARN-6380)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6380 | Implemented: `python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py`; terminal artifact `results/experiment_6380_three_family_canonical_factor_transport_canary.json`. | Implemented: `tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py`. |
| SCENARIO-LEARN-6380-GATE | Implemented: `python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py`. | Implemented: `tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py`. |
| SCENARIO-LEARN-6380-ARMS | Implemented: `python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py`. | Implemented: `tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py`. |
| SCENARIO-LEARN-6380-RAW | Implemented: `python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py`. | Implemented: `tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py`. |
| SCENARIO-LEARN-6380-ORACLE | Implemented: `python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py`. | Implemented: `tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py`. |
| SCENARIO-LEARN-6380-READY | Implemented: `python/carnot/experiment_6380_three_family_canonical_factor_transport_canary.py`. | Implemented: `tests/python/test_experiment_6380_three_family_canonical_factor_transport_canary.py`. |

## REQ-LEARN-6394: Model-Family Factor Harness Freeze

**Given** Exp6379 is contract-ready and Exp6380 exposes family-specific
development evidence
**When** Exp6394 starts on planning date 20260813
**Then** it SHALL write
`results/experiment_6394_model_family_factor_harness_freeze.json`
**And** it SHALL use exactly these local GGUF model ids from
`cached_sota_pair()`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`.

Exp6394 SHALL revalidate both RTX 3090 GPUs, model file hashes, revisions,
quantizations, embedded GGUF tokenizers, llama.cpp GPU offload, free disk,
the Exp6379 schema hash, and the Exp6380 raw-output receipts before it marks
the freeze ready. Token counts SHALL use embedded GGUF tokenizers only.
The experiment SHALL not call a Hugging Face tokenizer loader.

Exp6394 SHALL seal disjoint development and held manifests before any freeze
selection. The development manifest SHALL contain at least 18 licensed
development events across at least three executable constraint families. It
SHALL balance executable structure and surface labels. The held manifest SHALL
contain only redacted event identifiers, family labels, and hashes during
selection. Held content and held outcomes SHALL not be read.

Exp6394 SHALL preregister at most four bounded harness variants. Variants may
change prompt-role placement, response prefix, a bounded isolated packaging
step, or deterministic field routing. They SHALL not use grammar decoding,
parser or JIT repair, post-hoc JSON repair, hidden states, external scorers,
fine-tuning, or a token increase as the only selected change.

Exp6394 SHALL measure nonempty output, thinking leakage, repetition,
truncation, parse validity, source binding, exact checker calls, exact pass or
fail, abstention, latency, and verification cost by model family and variant.
It SHALL preserve raw outputs before one parse attempt. Exact task checkers are
the only oracles. The builder, harness selector, parser, and model text are
not oracles.

Exp6394 SHALL freeze one selected harness per model family with code hash,
prompt hash, response prefix, capacity, call count, seed, schema hash, and
selection reason. If a family has no passing development transport cell, the
selected harness SHALL be an explicit abstention policy for that family.
Readiness SHALL not imply a held license.

Exp6394 SHALL set `model_family_harness_freeze_ready_score=1.0` only when all
three family selections are frozen before held access, development work is
matched, raw receipts are complete, held access during selection is zero,
protected leakage and same-step writes are zero, model weight changes are zero,
all prohibited mechanism counts are zero, protected files are unchanged, and
all recorded verification commands pass. This field is the Exp6395 gate only.

Exp6394 SHALL emit these fields with explicit principles:

- `status`: Terminal status separates positive freeze, null, and blocked evidence.
- `MODEL_SPECS`: The three mandated GGUF model rows come from cached SOTA helper calls.
- `models_used`: Only authenticated Exp6380 development rows count as used models.
- `cached_sota_pair_receipts`: Helper-call receipts prevent manual model substitution.
- `model_file_hashes_revisions_quantizations_and_tokenizers`: Model file identity and tokenizer method are pinned.
- `embedded_gguf_tokenizer_receipts`: Tokenizer receipts use only embedded GGUF tokenizers.
- `autotokenizer_usage_count`: Bare zero proves no external tokenizer path was used.
- `cuda_offload_and_runtime_receipts_by_model`: CUDA offload, timing, token usage, return, raw streams, and cleanup are reported from development evidence.
- `development_and_held_manifest_paths_hashes_licenses_and_disjointness`: Development and held manifests are sealed, licensed, hash-bound, and disjoint.
- `development_balance_receipt`: Development events meet family, structure, and surface balance.
- `preregistered_harness_variants`: The bounded variants are frozen before selection.
- `builder_model_role_and_non_oracle_boundary`: The builder may propose surfaces but is not an oracle.
- `matched_development_work_receipts`: Event order, seeds, sampling controls, call counts, output capacity, and exact-check budget are matched within each family.
- `raw_output_before_parse_paths_hashes_and_counts`: Raw bytes are frozen before classification or parsing.
- `per_family_variant_transport_source_binding_exact_and_cost_results`: Transport, source binding, exact checks, and costs stay grouped by family and variant.
- `selected_harness_by_model_family`: One frozen harness or explicit abstention is selected for each family.
- `frozen_harness_paths_hashes_and_controls`: Code, prompt, prefix, capacity, call count, seed, and schema hash are frozen.
- `explicit_abstention_policy`: Failed cells abstain instead of inheriting another family result.
- `held_access_during_selection_count`: Bare zero proves held content and outcomes did not affect selection.
- `protected_leakage_and_same_step_write_counts`: Protected replay rows, generated labels, and same-step writes remain invisible.
- `model_weight_change_count`: Bare zero proves no model weights changed.
- `grammar_parser_jit_json_repair_hidden_state_and_external_scorer_usage_counts`: Bare zero counts prove prohibited mechanisms were absent.
- `model_family_harness_freeze_ready_score`: This bare scalar opens only the Exp6395 held-license gate.
- `held_license_not_implied`: A freeze does not license any held cell.
- `harm_underpowered_missing_and_flagged_cells`: Missing, invalid, underpowered, abstention, and flagged cells stay visible.
- `protected_files_unchanged`: Protected files remain byte-identical.
- `preconditions_checked`: Preconditions bind upstream, model, tokenizer, GPU, disk, schema, raw, source, and protected hashes.
- `inference_substrate`: The substrate declares local llama.cpp GGUF development evidence and deterministic freeze construction.
- `verifier_is_oracle`: Bare true applies only to exact task checkers.
- `field_principles`: Every required field states its guard.
- `field_provenance`: Every required field maps to specs, upstream artifacts, sidecars, model receipts, tests, or exact checks.
- `random_seed`: Fixed seeds pin manifest, variant, and selector order.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states the freeze boundary.

## SCENARIO-LEARN-6394-MANIFESTS: Development And Held Splits Are Sealed

**Given** the Exp6366 generated event matrix
**When** Exp6394 builds its split manifests
**Then** the development manifest contains at least 18 licensed events across
three families
**And** the held manifest is redacted and disjoint before selection starts.

## SCENARIO-LEARN-6394-SELECTION: Family Selections Freeze Or Abstain

**Given** Exp6380 has two Gemma source-bound capacity passes and a Qwen invalid
capacity cell
**When** Exp6394 applies its preregistered exact development rule
**Then** each Gemma family selects the canonical capacity harness
**And** the Qwen family freezes an explicit abstention harness.

## SCENARIO-LEARN-6394-NON-ORACLE: Exact Checkers Alone Are Oracles

**Given** builder text, parser output, selector rules, and model text exist
**When** Exp6394 writes its oracle boundary
**Then** only exact task checkers are marked as oracles
**And** builder, selector, parser, and model text are marked non-oracle.

## SCENARIO-LEARN-6394-READY: Freeze Readiness Does Not License Held Cells

**Given** all three family selections are frozen before held access
**When** raw receipts, matched work, prohibited counts, protected hashes, and
tests all pass
**Then** `model_family_harness_freeze_ready_score` SHALL be `1.0`
**And** `held_license_not_implied` SHALL be true.

## Implementation Status (REQ-LEARN-6394)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6394 | Implemented: `python/carnot/experiment_6394_model_family_factor_harness_freeze.py`; terminal artifact `results/experiment_6394_model_family_factor_harness_freeze.json`. | Implemented: `tests/python/test_experiment_6394_model_family_factor_harness_freeze.py`. |
| SCENARIO-LEARN-6394-MANIFESTS | Implemented: `python/carnot/experiment_6394_model_family_factor_harness_freeze.py`. | Implemented: `tests/python/test_experiment_6394_model_family_factor_harness_freeze.py`. |
| SCENARIO-LEARN-6394-SELECTION | Implemented: `python/carnot/experiment_6394_model_family_factor_harness_freeze.py`. | Implemented: `tests/python/test_experiment_6394_model_family_factor_harness_freeze.py`. |
| SCENARIO-LEARN-6394-NON-ORACLE | Implemented: `python/carnot/experiment_6394_model_family_factor_harness_freeze.py`. | Implemented: `tests/python/test_experiment_6394_model_family_factor_harness_freeze.py`. |
| SCENARIO-LEARN-6394-READY | Implemented: `python/carnot/experiment_6394_model_family_factor_harness_freeze.py`. | Implemented: `tests/python/test_experiment_6394_model_family_factor_harness_freeze.py`. |

## REQ-LEARN-6395: Held Factor Transport License Matrix

**Given** Exp6394 reports
`model_family_harness_freeze_ready_score=1.0`
**When** Exp6395 starts on planning date 20260813
**Then** it SHALL write
`results/experiment_6395_held_factor_transport_license_matrix.json`
**And** it SHALL replace the universal V549 gate with a capability matrix by
exact model id and executable constraint family.

Exp6395 SHALL use exactly these local GGUF model ids from
`cached_sota_pair()`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. The experiment SHALL not call `AutoTokenizer`. A missing
mandated model SHALL block only that model's cells. Legacy models SHALL not
populate the matrix.

Exp6395 SHALL revalidate the Exp6394 gate, frozen harness sidecar hashes,
held manifest seal, model file hashes, revisions, quantizations, embedded
tokenizer hashes, CUDA offload receipts, canonical schema hash, exact checker
identity, and zero held-content access before the Exp6394 freeze. It SHALL use
at least 18 held events. At least six held events SHALL come from each of
three executable constraint families.

Exp6395 SHALL run every mandated model on every applicable held event with the
model family's frozen harness and controls. It SHALL preserve raw output
before one parse. It SHALL not retry, repair, select another harness, tune on
held rows, substitute another family, or silently fall back to another model.

Exp6395 SHALL report nonempty output, syntax, structure, source binding,
exact-check calls, exact pass or fail, abstention, timeout, latency, and
verification cost for each model-family cell. Exact task checkers are the only
correctness oracle. Transport, parser output, harness selection, and model
text are not oracles.

Exp6395 SHALL preregister the license rule before evaluating held outcomes. A
cell earns a license only when it has at least six held trials, at least four
source-bound exact-evaluable proposals, zero false accepts, zero protected
leakage, complete runtime receipts, and no prohibited mechanism use.
Otherwise the cell SHALL end as rejected or abstained with a reason.

Each license SHALL bind to one model file hash, one quantization, one embedded
tokenizer hash, one frozen harness hash, one canonical schema hash, one
constraint family, one held event-manifest hash, and one expiration rule. An
unlicensed cell SHALL abstain. It SHALL not inherit another family result.

Exp6395 SHALL attack model-row swaps, family-label swaps, harness drift, stale
schema, source substitution, missing rows, fallback laundering, abstention
suppression, repeated output, and exact-fail promotion. Each attack SHALL fail
closed before a license can be promoted.

Exp6395 SHALL emit `licensed_model_count` and
`licensed_constraint_family_count` as bare integers. It SHALL set
`held_factor_transport_license_ready_score=1.0` only when at least two
mandated models and at least two constraint families have at least one valid
license, every other cell has an explicit terminal disposition, and universal
support is not claimed.

Exp6395 SHALL emit these fields with explicit principles:

- `status`: Terminal status separates positive, null, blocked, and retired held-license evidence.
- `exp6394_gate_receipt`: The Exp6394 freeze gate and frozen sidecars are revalidated before held evaluation.
- `MODEL_SPECS`: The three mandated GGUF model rows come from cached SOTA helper calls.
- `models_used`: Only mandated models with authenticated runtime receipts count as used.
- `cached_sota_pair_receipts`: Helper-call receipts prevent manual model substitution.
- `model_file_hashes_revisions_quantizations_and_tokenizers`: Model file identity and tokenizer method are pinned.
- `embedded_gguf_tokenizer_receipts`: Tokenizer receipts use only embedded GGUF tokenizers.
- `autotokenizer_usage_count`: Bare zero proves no external tokenizer path was used.
- `cuda_offload_and_runtime_receipts_by_model`: CUDA offload, timing, raw streams, and cleanup are reported per model.
- `frozen_harness_and_schema_hashes`: Harness sidecars and canonical schema are hash-bound.
- `held_manifest_path_hash_license_balance_and_prior_access_receipt`: Held events are licensed, balanced, sealed, and not read before freeze.
- `preregistered_license_rule`: The exact licensing thresholds are frozen before held outcomes are scored.
- `raw_output_before_parse_paths_hashes_and_counts`: Raw bytes are frozen before one parse.
- `per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix`: Every model-family cell reports transport, source binding, exact calls, abstention, timeout, latency, and cost.
- `capability_license_records`: Accepted licenses bind model, harness, schema, tokenizer, family, manifest, and expiration.
- `rejected_and_abstained_cell_records`: Every unlicensed cell has a terminal reason.
- `license_binding_and_expiration_fields`: License identity and expiry fields are explicit and narrow.
- `model_row_family_label_harness_schema_source_fallback_abstention_and_promotion_attack_matrix`: Swap, drift, fallback, abstention, repetition, and promotion attacks fail closed.
- `licensed_cell_count`: Bare count of valid model-family licenses.
- `licensed_model_count`: Bare count of mandated models with at least one valid license.
- `licensed_constraint_family_count`: Bare count of families with at least one valid license.
- `held_factor_transport_license_ready_score`: Readiness is a conjunctive matrix gate and never a universal-support claim.
- `universal_support_claimed`: Bare false prevents a universal gate from reappearing under another name.
- `protected_leakage_count`: Protected leakage must be zero for any license.
- `model_weight_change_count`: Bare zero proves no model weights changed.
- `prohibited_mechanism_usage_counts`: Retry, repair, reselection, tuning, family substitution, fallback, and external-tokenizer counts stay zero.
- `harm_underpowered_missing_and_flagged_cells`: Missing, underpowered, abstained, rejected, and attacked cells stay visible.
- `protected_files_unchanged`: Protected files remain byte-identical.
- `preconditions_checked`: Preconditions bind upstream, models, tokenizers, GPU, schema, manifests, sources, and protected files.
- `inference_substrate`: The substrate declares deterministic verifier replay over local GGUF identity receipts.
- `verifier_is_oracle`: Bare true applies only to exact task checkers.
- `field_principles`: Every required field states its guard and scientific purpose.
- `field_provenance`: Every required field maps to specs, upstream artifacts, sidecars, model receipts, tests, or exact checks.
- `random_seed`: Fixed seeds pin held schedule and matrix order.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states the license boundary.
- `model_family_harness_freeze_ready_score`: The Exp6394 gate proves only that harnesses froze before held access.

## SCENARIO-LEARN-6395-MATRIX: Held Trials Stay Cell-Local

**Given** three mandated model rows and three executable constraint families
**When** Exp6395 builds the held matrix
**Then** it SHALL create nine model-family cells
**And** each cell SHALL use only its exact model row, frozen harness hash,
schema hash, held event-manifest hash, and family label.

## SCENARIO-LEARN-6395-LICENSE: Narrow Licenses Require Exact Held Evidence

**Given** a model-family cell has at least six held trials
**When** at least four proposals are source-bound and exact-evaluable with zero
false accepts, zero protected leakage, complete runtime receipts, and no
prohibited mechanism
**Then** Exp6395 SHALL issue a license bound to that exact model file,
tokenizer, harness, schema, manifest, family, and expiration rule.

## SCENARIO-LEARN-6395-ABSTAIN: Missing Or Unlicensed Cells Abstain

**Given** a mandated model file is missing or a cell fails the license rule
**When** Exp6395 writes terminal dispositions
**Then** the affected cells SHALL be rejected or abstained with reasons
**And** no legacy model, alternate model row, or other family result SHALL
populate those cells.

## SCENARIO-LEARN-6395-ATTACKS: Capability Matrix Attacks Fail Closed

**Given** a row-swap, family-swap, harness-drift, stale-schema,
source-substitution, missing-row, fallback-laundering, abstention-suppression,
repeated-output, or exact-fail-promotion attack
**When** Exp6395 evaluates the attack matrix
**Then** no attack SHALL promote a license.

## SCENARIO-LEARN-6395-READY: Matrix Readiness Is Not Universal Support

**Given** at least two mandated models and at least two constraint families
have a valid license
**When** every remaining cell has an explicit terminal disposition and
`universal_support_claimed` is false
**Then** `held_factor_transport_license_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6395)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6395 | Planned: `python/carnot/experiment_6395_held_factor_transport_license_matrix.py`; terminal artifact `results/experiment_6395_held_factor_transport_license_matrix.json`. | Planned: `tests/python/test_experiment_6395_held_factor_transport_license_matrix.py`. |
| SCENARIO-LEARN-6395-MATRIX | Planned: `python/carnot/experiment_6395_held_factor_transport_license_matrix.py`. | Planned: `tests/python/test_experiment_6395_held_factor_transport_license_matrix.py`. |
| SCENARIO-LEARN-6395-LICENSE | Planned: `python/carnot/experiment_6395_held_factor_transport_license_matrix.py`. | Planned: `tests/python/test_experiment_6395_held_factor_transport_license_matrix.py`. |
| SCENARIO-LEARN-6395-ABSTAIN | Planned: `python/carnot/experiment_6395_held_factor_transport_license_matrix.py`. | Planned: `tests/python/test_experiment_6395_held_factor_transport_license_matrix.py`. |
| SCENARIO-LEARN-6395-ATTACKS | Planned: `python/carnot/experiment_6395_held_factor_transport_license_matrix.py`. | Planned: `tests/python/test_experiment_6395_held_factor_transport_license_matrix.py`. |
| SCENARIO-LEARN-6395-READY | Planned: `python/carnot/experiment_6395_held_factor_transport_license_matrix.py`. | Planned: `tests/python/test_experiment_6395_held_factor_transport_license_matrix.py`. |

## REQ-LEARN-6396: Capability-Qualified Verified Frontier A/B

**Given** Exp6395 issued held factor transport licenses
**When** Exp6396 starts on planning date 20260813
**Then** it SHALL write
`results/experiment_6396_capability_qualified_verified_frontier_ab.json`
**And** it SHALL compare independent proposals with a verified-incumbent
frontier only inside licensed model-family cells.

Exp6396 SHALL use the three mandated local GGUF model ids from
`cached_sota_pair()`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use the embedded GGUF
tokenizer named by each license. The experiment SHALL not call
`AutoTokenizer`.

Exp6396 SHALL revalidate every structured gate, license binding, model hash,
harness hash, schema hash, CUDA offload receipt, exact checker, and protected
future partition before an arm runs. Unlicensed cells SHALL emit a frozen
abstention. They SHALL not call a model and SHALL not receive a substitute
model, harness, family, or event result.

Exp6396 SHALL seal at least 24 train-counterexample events and at least 24
untouched future events across the licensed families. It SHALL balance
executable structure, source labels, and solver difficulty. Solver effort
SHALL not be used as model difficulty.

Exp6396 SHALL preregister two matched arms: independent restart and
verified frontier. The arms SHALL match licensed cells, model order, seeds,
event order, call count, harness capacity, candidate count, exact-check
budget, and wall-clock cap.

In the verified-frontier arm, Exp6396 SHALL retain only the strongest exactly
verified incumbent. Later rounds SHALL receive immutable residual failures
only. The active registry SHALL stay read-only.

Exp6396 SHALL record transport validity, source binding, exact pass rate,
incumbent changes, residual changes, effective proposal diversity, marginal
verified gain, stop reason, latency, and exact-check cost for every licensed
cell. It SHALL freeze one selected factor per arm before future access. It
SHALL evaluate untouched future exact outcomes once.

Exp6396 SHALL report proposal learnability, exact alignment, future utility,
confidence intervals, and model-family effects as separate fields. It SHALL
also run placebo labels, event-order perturbation, identity-blind joins,
license swaps, equal-work checks, no-gain stopping attacks, and protected
future leakage checks.

Exp6396 SHALL emit `delta_verified_future_exact_yield` as a finite bare
number. It SHALL set `capability_qualified_frontier_ready_score=1.0` only when
the verified-frontier treatment fired in every licensed model, work matched,
no protected leak occurred, all unlicensed cells abstained, and future
outcomes were read once. Readiness SHALL not require a positive delta.

Exp6396 SHALL emit these fields with explicit principles:

- `status`: Terminal status separates positive, null, blocked, and retired frontier evidence.
- `exp6395_gate_receipts`: Exp6395 readiness, licenses, and cell abstentions gate this experiment.
- `MODEL_SPECS`: The three mandated GGUF model rows come from cached SOTA helper calls.
- `models_used`: Only licensed mandated models with matched frontier work count as used.
- `cached_sota_pair_receipts`: Helper-call receipts prevent manual model substitution.
- `embedded_gguf_tokenizer_receipts`: Tokenizer receipts use only embedded GGUF tokenizers.
- `autotokenizer_usage_count`: Bare zero proves no external tokenizer path was used.
- `license_records_used_and_hashes`: Licenses bind model, tokenizer, harness, schema, family, manifest, and expiry.
- `unlicensed_cell_abstention_records`: Unlicensed cells remain visible and abstain without substitution.
- `model_harness_schema_and_checker_bindings`: Model files, harnesses, schemas, and exact checkers are bound before arms run.
- `cuda_offload_and_runtime_receipts_by_model`: CUDA offload and cleanup are reported for mandated models.
- `train_and_future_manifest_paths_hashes_licenses_balance_and_disjointness`: Train and future manifests are sealed, balanced, licensed, and disjoint.
- `preregistered_arm_contract`: The independent and frontier arms are frozen before scoring.
- `matched_work_receipts`: Calls, candidates, event order, exact checks, and caps match across arms.
- `raw_output_before_parse_paths_hashes_and_counts`: Raw proposal bytes are frozen before parse.
- `per_cell_transport_source_binding_exact_and_cost_results`: Licensed cells report transport, source binding, exact outcomes, latency, and cost.
- `incumbent_and_residual_histories`: Frontier state stores only verified incumbents and immutable residual failures.
- `proposal_learnability_results`: Training counterexample response is separate from future utility.
- `exact_alignment_results`: Exact checker agreement is separate from proposal learnability and future utility.
- `frozen_selected_factors_by_arm`: One factor per arm is frozen before future access.
- `untouched_future_evaluation_receipts`: Protected future outcomes open once after factor freeze.
- `future_exact_yield_by_arm_and_model`: Future exact utility is reported per arm and model before pooling.
- `delta_verified_future_exact_yield`: The paired future yield delta is a finite bare number.
- `confidence_intervals_and_effective_sample_sizes`: Intervals and effective sample sizes are reported separately from point estimates.
- `identity_license_order_placebo_work_stopping_and_leakage_attack_matrix`: Identity, license, order, placebo, work, stopping, and leakage attacks fail closed.
- `capability_qualified_frontier_ready_score`: Readiness checks treatment firing, work parity, abstention, leak-free future access, and single future open.
- `registry_write_count`: Bare zero proves the active registry stayed read-only.
- `protected_leakage_count`: Bare zero proves protected future labels did not leak.
- `model_weight_change_count`: Bare zero proves no model weights changed.
- `harm_underpowered_missing_and_flagged_cells`: Missing, unlicensed, underpowered, and attacked cells stay visible.
- `protected_files_unchanged`: Protected files remain byte-identical.
- `preconditions_checked`: Preconditions bind gates, licenses, models, tokenizers, GPUs, schema, manifests, sources, and protected files.
- `inference_substrate`: The substrate declares deterministic replay over licensed local GGUF receipts.
- `verifier_is_oracle`: Bare true applies only to exact task checkers.
- `field_principles`: Every required field states its guard and scientific purpose.
- `field_provenance`: Every required field maps to specs, upstream artifacts, manifests, tests, or exact checks.
- `random_seed`: Fixed seeds pin split, arm, and event order.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states the frontier boundary.

## SCENARIO-LEARN-6396-LICENSED-CELLS: Frontier Runs Only On Licensed Cells

**Given** Exp6395 licenses a subset of model-family cells
**When** Exp6396 runs the two arms
**Then** only those licensed cells SHALL receive model calls
**And** every other cell SHALL emit a frozen abstention record.

## SCENARIO-LEARN-6396-FRONTIER: Residuals Follow The Verified Incumbent

**Given** the verified-frontier arm has a current exact incumbent
**When** the next round starts
**Then** it SHALL expose only immutable residual failures
**And** it SHALL not write to the active registry.

## SCENARIO-LEARN-6396-FUTURE: Future Outcomes Open Once

**Given** both arms freeze one selected factor per licensed cell
**When** Exp6396 evaluates future utility
**Then** the untouched future exact outcomes SHALL be read once
**And** proposal learnability, exact alignment, and future utility SHALL be
reported separately.

## SCENARIO-LEARN-6396-ATTACKS: Qualification Attacks Fail Closed

**Given** placebo labels, event-order perturbation, identity-blind joins,
license swaps, unequal work, no-gain stopping, or protected-future leakage
**When** Exp6396 evaluates the attack matrix
**Then** no attack SHALL promote readiness.

## SCENARIO-LEARN-6396-READY: Readiness Does Not Require Positive Utility

**Given** the frontier treatment fires in every licensed model, work matches,
protected leakage is zero, all unlicensed cells abstain, and future outcomes
open once
**When** `delta_verified_future_exact_yield` is finite
**Then** `capability_qualified_frontier_ready_score` SHALL be `1.0` even if
the delta is not positive.

## Implementation Status (REQ-LEARN-6396)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6396 | Planned: `python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py`; terminal artifact `results/experiment_6396_capability_qualified_verified_frontier_ab.json`. | Planned: `tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py`. |
| SCENARIO-LEARN-6396-LICENSED-CELLS | Planned: `python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py`. | Planned: `tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py`. |
| SCENARIO-LEARN-6396-FRONTIER | Planned: `python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py`. | Planned: `tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py`. |
| SCENARIO-LEARN-6396-FUTURE | Planned: `python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py`. | Planned: `tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py`. |
| SCENARIO-LEARN-6396-ATTACKS | Planned: `python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py`. | Planned: `tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py`. |
| SCENARIO-LEARN-6396-READY | Planned: `python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py`. | Planned: `tests/python/test_experiment_6396_capability_qualified_verified_frontier_ab.py`. |

## REQ-LEARN-6397: Transactional Continuous Factor Learning

**Given** Exp6396 has a qualified verified-frontier result and Exp6383 has a
positive selective rollback control
**When** Exp6397 starts on planning date 20260813
**Then** it SHALL write
`results/experiment_6397_transactional_continuous_factor_learning.json`
**And** it SHALL activate candidate factors only through exact predecessor-bound
transactions on Exp6395 licensed cells.

Exp6397 SHALL revalidate both Exp6396 gates, license records, frozen harnesses,
model files, GPU offload receipts, exact checker hashes, the e-value release
ledger, the Exp6383 rollback receipt, and protected partitions before opening
the chronological stream. Unlicensed cells SHALL abstain without model calls,
substitution, or inherited evidence.

Exp6397 SHALL use exactly these local GGUF model ids from `cached_sota_pair()`:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. The experiment SHALL not call `AutoTokenizer`.

Exp6397 SHALL seal a chronological stream with at least 48 events. The stream
SHALL include acquisition, release, retention, and untouched future segments,
at least three update opportunities, and at least two restart boundaries. The
licensed constraint families SHALL be balanced. Protected future events SHALL
be evaluated once after all factor heads are frozen.

Exp6397 SHALL compare three matched arms: frozen baseline, V546
replay-certified factor control, and capability-qualified live learner. Event
order, exact checks, consumer budget, and protected partitions SHALL match
across arms.

The active factor head SHALL be read-only during proposal. Each typed candidate
SHALL be evaluated off-commit and bound to `predecessor_head_hash`,
`candidate_hash`, evidence hashes, exact release receipt, e-process state, and
proposed effects. Each proposal SHALL atomically record exactly one
disposition: Commit, Reject, Quarantine, or Defer.

A Commit SHALL revalidate ownership, predecessor freshness, exact support,
effect uniqueness, retention, and protected replay before the head advances.
Reject, Quarantine, and Defer dispositions SHALL never advance the head.
Stale predecessors, duplicate effects, replayed evidence, self-approval,
concurrent proposals, interrupted writes, and restart recovery attacks SHALL
fail closed. No failed transaction may change the active head.

Exp6397 SHALL report proposal learnability, exact alignment, forward transfer,
backward retention, negative transfer, forgetting, abstention, factor growth,
verification cost, restart recovery, and selective rollback. It SHALL carry
`selective_rollback_control_ready_score` exactly from Exp6383.

Exp6397 SHALL set
`transactional_continuous_self_learning_ready_score=1.0` only when at least one
factor commits through the full transaction, untouched future exact yield beats
the frozen baseline, retention has no harmful regression, factor growth stays
bounded, stale or duplicate attacks fail closed, protected leakage is zero,
model weight changes are zero, and all recorded tests pass.

Exp6397 SHALL emit these fields with explicit principles:

- `status`: Terminal status follows transactional activation gates and protected replay.
- `exp6396_gate_receipts`: Exp6396 readiness, licenses, future yield, and protected partitions gate this run.
- `MODEL_SPECS`: The three mandated GGUF rows come from cached SOTA helper calls.
- `models_used`: Only licensed mandated models with transactional work count as used.
- `cached_sota_pair_receipts`: Helper-call receipts prevent manual model substitution.
- `embedded_gguf_tokenizer_receipts`: Tokenizer receipts use only embedded GGUF tokenizers.
- `autotokenizer_usage_count`: Bare zero proves no external tokenizer path was used.
- `license_and_frozen_harness_bindings`: Licenses, harnesses, schemas, models, and exact checkers are bound before events run.
- `unlicensed_cell_abstention_records`: Unlicensed cells abstain without substitution.
- `cuda_offload_and_runtime_receipts_by_model`: CUDA offload and cleanup are reported for mandated models.
- `chronological_manifest_path_hash_license_balance_and_partition_seals`: Chronology, licenses, balance, restart boundaries, and partitions are sealed.
- `preregistered_arm_contract`: Frozen, V546 control, and live learner arms are matched.
- `factor_head_initial_hash`: The initial read-only factor head is frozen.
- `typed_candidate_records`: Typed candidates are evaluated off-commit.
- `predecessor_candidate_evidence_checker_eprocess_and_effect_bindings`: Candidate activation inputs are hash-bound.
- `atomic_disposition_records`: Each candidate has exactly one terminal disposition.
- `factor_head_transition_history`: Only successful commits advance the head.
- `commit_reject_quarantine_and_defer_counts`: Disposition counts stay explicit.
- `stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix`: Transaction attacks fail closed.
- `proposal_learnability_results`: Learnability is reported separately from utility.
- `exact_alignment_results`: Exact checker alignment is reported separately from learnability and utility.
- `forward_transfer_results`: Future exact transfer is measured per arm.
- `backward_retention_and_forgetting_results`: Prior retained behavior cannot regress.
- `negative_transfer_and_harm_results`: Harmful transfer, abstention, and leakage stay visible.
- `factor_growth_and_capacity_results`: Factor growth stays bounded.
- `verification_cost_results`: Exact checker calls, latency, and cost are charged.
- `untouched_future_evaluation_receipts`: Protected future outcomes open once after head freeze.
- `future_exact_yield_by_arm`: Future exact utility is reported by arm.
- `delta_future_exact_yield_over_frozen`: Live learner future yield is compared with frozen.
- `selective_rollback_control_path_hash_and_terminal_class`: Exp6383 is carried as a rollback control.
- `selective_rollback_control_ready_score`: The exact Exp6383 ready score is carried.
- `transactional_continuous_self_learning_ready_score`: Readiness is conjunctive over commit, utility, retention, growth, attacks, leaks, weights, and tests.
- `protected_leakage_count`: Bare zero proves protected partitions did not leak.
- `same_step_write_count`: Bare zero proves proposal-time writes stayed invisible.
- `model_weight_change_count`: Bare zero proves no model weights changed.
- `harm_underpowered_missing_and_flagged_cells`: Missing, underpowered, unlicensed, rejected, and attacked cells stay visible.
- `protected_files_unchanged`: Protected files remain byte-identical.
- `preconditions_checked`: Preconditions bind upstream gates, models, tokenizers, GPUs, exact checkers, manifests, seeds, and protected files.
- `inference_substrate`: The substrate declares deterministic transactional replay over licensed local GGUF receipts.
- `verifier_is_oracle`: Bare true applies only to exact task checkers and exact release tests.
- `field_principles`: Every required field states its guard and purpose.
- `field_provenance`: Every required field maps to specs, upstream artifacts, transactions, attacks, tests, or exact checks.
- `random_seed`: Fixed seeds pin chronology, proposals, attacks, and future opens.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states the transaction boundary.

## SCENARIO-LEARN-6397-CHRONOLOGY: Stream And Arms Are Sealed

**Given** licensed Exp6395 cells and Exp6396 qualification receipts
**When** Exp6397 seals the chronological stream
**Then** it SHALL contain at least 48 events, acquisition, release, retention,
and untouched future partitions, at least three update opportunities, and at
least two restart boundaries
**And** all three arms SHALL share event order, exact checks, and consumer
budget.

## SCENARIO-LEARN-6397-TRANSACTION: Commits Are Predecessor-Bound

**Given** a typed candidate proposed against a read-only active factor head
**When** Exp6397 evaluates the candidate off-commit
**Then** the record SHALL bind predecessor head, candidate, evidence, exact
checker, e-process state, release receipt, and proposed effects
**And** exactly one disposition SHALL be recorded.

## SCENARIO-LEARN-6397-ATTACKS: Failed Transactions Do Not Advance The Head

**Given** stale predecessor, duplicate effect, replayed evidence,
self-approval, concurrent proposal, interrupted write, or restart attack
**When** Exp6397 replays the transaction journal
**Then** every attack SHALL fail closed
**And** the active head hash SHALL remain unchanged for each failed
transaction.

## SCENARIO-LEARN-6397-FUTURE: Utility Opens Once After Head Freeze

**Given** at least one factor committed through the full transaction
**When** Exp6397 evaluates untouched future events
**Then** future exact outcomes SHALL open once after the head is frozen
**And** live-learner future yield SHALL be reported against the frozen
baseline.

## SCENARIO-LEARN-6397-READY: Readiness Is Fully Conjunctive

**Given** no committed factor, non-positive future delta, harmful retention
regression, unbounded factor growth, stale or duplicate attack survivor,
protected leakage, model weight mutation, protected-file mutation, or failed
test
**When** readiness is computed
**Then** `transactional_continuous_self_learning_ready_score` SHALL be `0.0`.

## Implementation Status (REQ-LEARN-6397)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6397 | Planned: `python/carnot/experiment_6397_transactional_continuous_factor_learning.py`; terminal artifact `results/experiment_6397_transactional_continuous_factor_learning.json`. | Planned: `tests/python/test_experiment_6397_transactional_continuous_factor_learning.py`. |
| SCENARIO-LEARN-6397-CHRONOLOGY | Planned: `python/carnot/experiment_6397_transactional_continuous_factor_learning.py`. | Planned: `tests/python/test_experiment_6397_transactional_continuous_factor_learning.py`. |
| SCENARIO-LEARN-6397-TRANSACTION | Planned: `python/carnot/experiment_6397_transactional_continuous_factor_learning.py`. | Planned: `tests/python/test_experiment_6397_transactional_continuous_factor_learning.py`. |
| SCENARIO-LEARN-6397-ATTACKS | Planned: `python/carnot/experiment_6397_transactional_continuous_factor_learning.py`. | Planned: `tests/python/test_experiment_6397_transactional_continuous_factor_learning.py`. |
| SCENARIO-LEARN-6397-FUTURE | Planned: `python/carnot/experiment_6397_transactional_continuous_factor_learning.py`. | Planned: `tests/python/test_experiment_6397_transactional_continuous_factor_learning.py`. |
| SCENARIO-LEARN-6397-READY | Planned: `python/carnot/experiment_6397_transactional_continuous_factor_learning.py`. | Planned: `tests/python/test_experiment_6397_transactional_continuous_factor_learning.py`. |

## REQ-LEARN-6398: Default-Off Transactional Factor Consumer

**Given** Exp6397 has a retained predecessor-bound factor head and Exp6383 has
a positive selective rollback control
**When** Exp6398 runs on planning date 20260813
**Then** it SHALL write
`results/experiment_6398_default_off_transactional_factor_consumer.json`
**And** it SHALL evaluate a default-off future consumer without writing factors,
advancing heads, renewing licenses, approving fallbacks, reading protected
outcomes early, or enabling the production path.

Exp6398 SHALL revalidate both Exp6397 activation gates, factor-head hash,
transaction log, license bindings, rollback receipt, model files, harnesses,
GPU offload receipts, exact checker hashes, and the untouched consumer-event
seal before any consumer decision. The active Exp6397 head SHALL be retained
and predecessor-bound. It SHALL be read-only for the whole run.

Exp6398 SHALL use exactly these local GGUF model ids from `cached_sota_pair()`:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. The experiment SHALL not call `AutoTokenizer`.

Exp6398 SHALL freeze all factor and license writes. It SHALL use at least 24
untouched future consumer events across licensed families. It SHALL compare
three matched arms: frozen baseline, V546 replay-certified registry, and V550
transactional registry. Event order, model work, exact checker calls, token
budgets, and protected access rules SHALL match across arms.

Exp6398 SHALL call only licensed model-family cells. Unlicensed, rejected,
expired, stale, revoked, missing, or family-mismatched cells SHALL preserve an
explicit abstention. Retry, switch, and abstain outcomes SHALL remain distinct.
A failed or revoked model-family cell SHALL NOT silently switch to another
family or inherit another family's license.

Exp6398 SHALL record source-bound proposals, factor retrievals, license
checks, abstentions, exact checker calls, exact yield, false accepts, false
rejects, latency, verification cost, and consumer decisions by model and
family. It SHALL report family-specific and pooled confidence intervals,
effective sample sizes, negative transfer, harm, underpowered cells, and
missing cells. It SHALL NOT pool abstentions as successes.

Exp6398 SHALL inject stale head, revoked descendant, expired license,
model-row swap, family switch request, absent licensed model, duplicated
evidence, incomplete rollback, and suppressed abstention attacks. Every attack
SHALL fail closed. Exp6398 SHALL apply the Exp6383 selective rollback control
to harmful descendants on the injected cells only, and compare it with full
reset and no rollback on those same injected cells. It SHALL NOT claim a new
rollback method.

Exp6398 SHALL set
`default_off_transactional_consumer_ready_score=1.0` only when V550 improves
exact yield over frozen, false accepts do not increase, every attack fails
closed, selective rollback removes harmful descendants, the production enable
count stays zero, and all recorded tests pass.

Exp6398 SHALL emit these fields with explicit principles:

- `status`: Terminal status follows read-only consumer safety, arm utility, rollback, protected access, and tests.
- `exp6397_gate_receipts`: Exp6397 gates, factor head, transaction log, licenses, rollback carry, and protected seals gate this run.
- `MODEL_SPECS`: The three mandated GGUF rows come from cached SOTA helper calls.
- `models_used`: Only licensed mandated models with default-off consumer work count as used.
- `cached_sota_pair_receipts`: Helper-call receipts prevent manual model substitution.
- `embedded_gguf_tokenizer_receipts`: Tokenizer receipts use only embedded GGUF tokenizers.
- `autotokenizer_usage_count`: Bare zero proves no external tokenizer path was used.
- `frozen_factor_head_and_transaction_log_hashes`: Exp6397 retained head and transaction log are hash-bound before consumer reads.
- `license_and_harness_bindings`: Licenses, harnesses, exact checkers, and release ledger are bound before decisions.
- `cuda_offload_and_runtime_receipts_by_model`: CUDA offload and cleanup are reported for mandated models.
- `untouched_consumer_manifest_path_hash_license_balance_and_prior_access_receipt`: Future consumer events, license balance, and no-prior-access seal are frozen.
- `preregistered_arm_contract`: Frozen, V546, and V550 consumer arms are matched before scoring.
- `matched_work_receipts`: Event counts, model calls, exact checks, token budgets, latency rules, and work caps match across arms.
- `per_model_family_retrieval_license_abstention_checker_yield_and_cost_results`: Retrievals, license checks, abstentions, checker calls, yield, latency, cost, and decisions are reported by model and family.
- `exact_yield_by_arm`: Consumer exact yield is reported by arm.
- `delta_exact_yield_over_frozen`: V550 utility is compared with frozen baseline.
- `false_accept_false_reject_negative_transfer_and_harm_results`: False accepts, false rejects, negative transfer, and harm stay visible.
- `confidence_intervals_and_effective_sample_sizes`: Per-family and pooled intervals, effective sample sizes, and abstention exclusions are explicit.
- `stale_head_revoked_descendant_expired_license_model_swap_family_switch_missing_model_duplicate_evidence_rollback_and_abstention_attack_matrix`: Every preregistered consumer attack fails closed.
- `selective_rollback_full_reset_and_no_rollback_injected_cell_results`: Exp6383 selective rollback, full reset, and no rollback are compared only on injected cells.
- `consumer_factor_write_count`: Bare zero proves the consumer wrote no factors.
- `factor_head_advance_count`: Bare zero proves no head advanced.
- `license_renewal_count`: Bare zero proves licenses were not renewed.
- `silent_fallback_count`: Bare zero proves no fallback was approved silently.
- `production_enable_count`: Bare zero proves the default-off path stayed off.
- `protected_leakage_count`: Bare zero proves protected outcomes were not read early.
- `default_off_transactional_consumer_ready_score`: Readiness is conjunctive over utility, false accepts, attacks, rollback, production enablement, and tests.
- `harm_underpowered_missing_and_flagged_cells`: Missing, underpowered, unlicensed, rejected, expired, stale, revoked, and attacked cells stay visible.
- `protected_files_unchanged`: Protected files remain byte-identical.
- `preconditions_checked`: Preconditions bind date, upstream gates, models, tokenizers, GPUs, exact checkers, manifests, seeds, and protected files.
- `inference_substrate`: The substrate declares deterministic default-off consumer replay over licensed local GGUF receipts.
- `verifier_is_oracle`: Bare true applies only to exact task checkers.
- `field_principles`: Every required field states its guard and purpose.
- `field_provenance`: Every required field maps to specs, upstream artifacts, consumer events, attacks, tests, or exact checks.
- `random_seed`: Fixed seed pins consumer events, arm order, attacks, and future opens.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states the default-off consumer boundary.

## SCENARIO-LEARN-6398-READONLY: Consumer Cannot Mutate State

**Given** the Exp6397 terminal factor head and transaction log
**When** Exp6398 evaluates default-off future consumer events
**Then** factor writes, head advances, license renewals, silent fallbacks,
protected leakage, and production enables SHALL all remain zero.

## SCENARIO-LEARN-6398-LICENSED: License Cells Cannot Switch Families

**Given** licensed and unlicensed model-family cells
**When** a cell is rejected, expired, stale, revoked, missing, or not licensed
**Then** the consumer SHALL abstain explicitly
**And** it SHALL NOT switch to another model family or inherit another
family's license.

## SCENARIO-LEARN-6398-MATCHED: Arms Share Future Consumer Work

**Given** at least 24 untouched future consumer events across licensed families
**When** frozen, V546, and V550 arms run
**Then** event order, model work, exact checker calls, token budgets, and
protected access rules SHALL match across arms
**And** V550 exact yield SHALL be compared with frozen baseline.

## SCENARIO-LEARN-6398-ATTACKS: Consumer Attacks Fail Closed

**Given** stale head, revoked descendant, expired license, model-row swap,
family switch, missing model, duplicate evidence, incomplete rollback, or
suppressed abstention attack
**When** Exp6398 evaluates the attacked cell
**Then** the decision SHALL fail closed as abstain, reject, quarantine, or
rollback
**And** no failed cell SHALL write factors, advance a head, renew a license, or
enable production.

## SCENARIO-LEARN-6398-ROLLBACK: Exp6383 Control Applies Only To Injected Cells

**Given** harmful descendants on injected cells
**When** Exp6398 compares selective rollback, full reset, and no rollback
**Then** the Exp6383 selective rollback control SHALL remove harmful
descendants, full reset SHALL over-remove valid injected state, and no rollback
SHALL leave unsafe survivors
**And** the comparison SHALL NOT rerun or rename the original rollback method.

## SCENARIO-LEARN-6398-READY: Readiness Is Fully Conjunctive

**Given** no V550 exact-yield gain, increased false accepts, attack survivor,
rollback failure, production enablement, protected leakage, protected-file
mutation, or failed test
**When** readiness is computed
**Then** `default_off_transactional_consumer_ready_score` SHALL be `0.0`.

## Implementation Status (REQ-LEARN-6398)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6398 | Implemented: `python/carnot/experiment_6398_default_off_transactional_factor_consumer.py`; terminal artifact `results/experiment_6398_default_off_transactional_factor_consumer.json`. | Implemented: `tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py`. |
| SCENARIO-LEARN-6398-READONLY | Implemented: `python/carnot/experiment_6398_default_off_transactional_factor_consumer.py`. | Implemented: `tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py`. |
| SCENARIO-LEARN-6398-LICENSED | Implemented: `python/carnot/experiment_6398_default_off_transactional_factor_consumer.py`. | Implemented: `tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py`. |
| SCENARIO-LEARN-6398-MATCHED | Implemented: `python/carnot/experiment_6398_default_off_transactional_factor_consumer.py`. | Implemented: `tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py`. |
| SCENARIO-LEARN-6398-ATTACKS | Implemented: `python/carnot/experiment_6398_default_off_transactional_factor_consumer.py`. | Implemented: `tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py`. |
| SCENARIO-LEARN-6398-ROLLBACK | Implemented: `python/carnot/experiment_6398_default_off_transactional_factor_consumer.py`. | Implemented: `tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py`. |
| SCENARIO-LEARN-6398-READY | Implemented: `python/carnot/experiment_6398_default_off_transactional_factor_consumer.py`. | Implemented: `tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py`. |

## REQ-LEARN-6399: V550 Capability Learning Safety Audit

**Given** V550 factor-chain artifacts may be present, absent, blocked,
skipped, null, flagged, retired, or malformed
**When** Exp6399 runs on planning date 20260813
**Then** it SHALL write
`results/experiment_6399_capability_learning_safety_audit.json`
**And** it SHALL audit Exp6394 through Exp6398 independently of conductor
success, without invoking an LLM, rerunning upstream experiments, or repairing
upstream evidence.

Exp6399 SHALL register expected artifacts, sidecars, source files, task IDs,
model IDs, schemas, harnesses, license records, factor heads, transaction logs,
and exact checker versions before reading upstream conclusion fields. Missing,
blocked, skipped, null, flagged, and retired rows SHALL keep their own classes.
The audit SHALL keep artifact verdicts separate from conductor outcomes.

Exp6399 SHALL treat exact checker receipts as oracle evidence only within their
declared event and license scope. Harnesses, licenses, transaction records, and
model text SHALL be non-oracles. The top-level `verifier_is_oracle` SHALL be
bare `false`.

Exp6399 SHALL attack development-held leakage, future leakage, source
substitution, family identity drift, model hash drift, harness drift, schema
drift, license overreach, inherited licenses, silent fallback, abstention
suppression, and partial-cell pooling. It SHALL also attack self-activation,
stale predecessor, duplicate effect, replayed evidence, optional-stopping
reset, interrupted atomic write, concurrent head advance, restart corruption,
unauthorized license renewal, exact-check omission, verifier-version drift,
rollback underreach, rollback overreach, revoked-descendant survival, consumer
write, and production enablement.

Exp6399 SHALL recompute readiness and downstream gates from bare terminal
fields. Missing, nested, wrong-type, boolean, NaN, and infinity values SHALL
fail closed. Safety success SHALL not promote utility. A narrow model-family
license SHALL never become a general factor utility claim. A predecessor-bound
transaction SHALL never launder a partial or failed model-family cell into a
public claim.

Exp6399 SHALL verify the mandated `MODEL_SPECS`, cached SOTA receipts, embedded
GGUF tokenizer receipts, no `AutoTokenizer` use, no legacy headline result,
accurate `inference_substrate`, and task-linked GPU evidence where applicable.
It SHALL set `public_factor_claim_eligibility=false` unless every required clean
scientific gate and safety gate passes for the full declared public scope.

Exp6399 SHALL emit these fields with explicit fail-closed principles:

- `status`: Terminal status follows the independent audit and public-claim gate.
- `audit_registration_path_hash_and_expected_scope`: Registration binds paths, scopes, versions, and read order before conclusions.
- `present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix`: Every expected input keeps its terminal evidence class.
- `artifact_verdict_and_conductor_outcome_reconciliation`: Artifact verdicts and conductor outcomes stay separate.
- `model_schema_harness_license_factor_head_transaction_and_checker_hash_matrix`: Model, schema, harness, license, head, transaction, and checker hashes are bound together.
- `development_held_future_and_source_leakage_attack_results`: Leakage and source attacks cannot promote readiness.
- `family_model_harness_schema_license_fallback_abstention_and_pooling_attack_results`: License and pooling attacks cannot broaden scope.
- `predecessor_effect_evidence_optional_stopping_atomicity_concurrency_restart_and_renewal_attack_results`: Transaction attacks cannot advance a head or renew a license.
- `exact_checker_rollback_revocation_consumer_write_and_enablement_attack_results`: Checker, rollback, consumer-write, and enablement attacks fail closed.
- `recomputed_readiness_scores_and_gates`: Bare terminal fields recompute all readiness and claim gates.
- `model_policy_and_inference_substrate_checks`: Model, tokenizer, GPU, substrate, and legacy-claim checks stay explicit.
- `duration_receipt_source`: Wall-clock duration is measured by the audit only.
- `critical_major_and_minor_findings`: Findings are severity-separated without synthesis.
- `utility_promotion_count`: Safety evidence cannot become utility evidence.
- `public_factor_claim_eligibility`: The public claim is false unless the full clean scope passes.
- `upstream_artifacts_modified`: Upstream artifacts must remain unchanged.
- `protected_files_unchanged`: Protected repo files must remain unchanged.
- `preconditions_checked`: Preconditions bind date, registration, classes, hashes, sources, protected files, and commands.
- `inference_substrate`: The substrate declares deterministic artifact audit without LLM or upstream rerun.
- `verifier_is_oracle`: Bare false states that the audit is not an oracle.
- `field_principles`: Required fields and recomputed claim fields state their fail-closed purpose.
- `field_provenance`: Required fields trace to specs, inputs, attacks, checks, tests, or hashes.
- `random_seed`: Fixed seed pins registration and attack order.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states the public-claim boundary.

## SCENARIO-LEARN-6399-REGISTRATION: Scope Freezes Before Conclusions

**Given** the expected V550 artifact chain and sidecars
**When** Exp6399 starts
**Then** it SHALL write a registration sidecar that hashes expected paths,
source files, task IDs, model IDs, schemas, harnesses, licenses, factor heads,
transaction logs, and checker versions before reading readiness or utility
fields.

## SCENARIO-LEARN-6399-CLASS-PRESERVATION: Blocked And Missing Stay Visible

**Given** any upstream artifact is missing, blocked, skipped, null, flagged,
retired, or malformed
**When** Exp6399 builds the artifact matrix
**Then** the row SHALL keep that class
**And** no downstream positive row SHALL relabel it clean.

## SCENARIO-LEARN-6399-LICENSE-BOUNDARY: Narrow Licenses Do Not Become Public Claims

**Given** only a subset of model-family cells carries narrow licenses
**When** Exp6399 recomputes downstream utility and claim gates
**Then** `public_factor_claim_eligibility` SHALL remain `false`
**And** unlicensed, rejected, underpowered, missing, blocked, null, flagged,
and retired cells SHALL not be pooled as successes.

## SCENARIO-LEARN-6399-TRANSACTION-BOUNDARY: Failed Transactions Do Not Launder Claims

**Given** stale predecessor, duplicate effect, replayed evidence, self-activation,
optional-stopping reset, interrupted write, concurrent head advance, restart
corruption, or unauthorized license renewal
**When** Exp6399 evaluates transaction attacks
**Then** no attack SHALL advance the retained head, renew a license, promote
readiness, write a consumer factor, or enable production.

## SCENARIO-LEARN-6399-READY: Claim Readiness Fails Closed

**Given** any recomputed readiness field is missing, nested, wrong-type,
boolean, NaN, infinity, non-positive where positive is required, or outside the
declared license scope
**When** Exp6399 refreshes terminal fields
**Then** the claim gate SHALL fail closed
**And** `public_factor_claim_eligibility` SHALL be `false`.

## Implementation Status (REQ-LEARN-6399)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6399 | Implemented: `python/carnot/experiment_6399_capability_learning_safety_audit.py`; terminal artifact `results/experiment_6399_capability_learning_safety_audit.json`. | Implemented: `tests/python/test_experiment_6399_capability_learning_safety_audit.py`. |
| SCENARIO-LEARN-6399-REGISTRATION | Implemented: `python/carnot/experiment_6399_capability_learning_safety_audit.py`. | Implemented: `tests/python/test_experiment_6399_capability_learning_safety_audit.py`. |
| SCENARIO-LEARN-6399-CLASS-PRESERVATION | Implemented: `python/carnot/experiment_6399_capability_learning_safety_audit.py`. | Implemented: `tests/python/test_experiment_6399_capability_learning_safety_audit.py`. |
| SCENARIO-LEARN-6399-LICENSE-BOUNDARY | Implemented: `python/carnot/experiment_6399_capability_learning_safety_audit.py`. | Implemented: `tests/python/test_experiment_6399_capability_learning_safety_audit.py`. |
| SCENARIO-LEARN-6399-TRANSACTION-BOUNDARY | Implemented: `python/carnot/experiment_6399_capability_learning_safety_audit.py`. | Implemented: `tests/python/test_experiment_6399_capability_learning_safety_audit.py`. |
| SCENARIO-LEARN-6399-READY | Implemented: `python/carnot/experiment_6399_capability_learning_safety_audit.py`. | Implemented: `tests/python/test_experiment_6399_capability_learning_safety_audit.py`. |

## REQ-LEARN-6406: Clean V550 Factor Evidence Boundary

**Given** Exp6394 through Exp6398 produced narrow internal V550 factor
evidence, Exp6399 preserved public ineligibility as a null audit, and Exp6385
remains quarantined
**When** Exp6406 runs on planning date 20260813
**Then** it SHALL write
`results/experiment_6406_clean_v550_factor_evidence_boundary.json`
**And** it SHALL build an immutable V550-only clean evidence boundary without
invoking an LLM, regenerating upstream evidence, modifying upstream artifacts,
repairing Exp6385, rerunning Exp6399, or creating a public claim.

Exp6406 SHALL register expected V550 artifacts, sidecars, source files, model
IDs, license records, exact checker hashes, transaction heads, and conductor
outcomes before reading conclusion fields. It SHALL hash and summarize Exp6394
through Exp6399, Exp6403, and Exp6385. It SHALL keep artifact verdicts,
conductor outcomes, duration receipts, and adversarial flags as separate facts.

Exp6406 SHALL include a row only when it is V550-produced, task-linked,
terminal, unflagged, hash-complete, source-bound, and inside its declared
model-family and constraint-family license. It SHALL exclude Exp6385, Exp6399,
blocked, null, absent, unlicensed, rejected, flagged, missing-sidecar, and
unproven-duration rows. Excluded rows SHALL remain visible as excluded facts.

Exp6406 SHALL recompute only narrow internal harness, license, frontier,
transactional-learning, consumer, and safety states. It SHALL NOT recompute
universal support or public eligibility from partial cells.

Exp6406 SHALL attack artifact substitution, lineage laundering, date relabeling,
model and family swaps, license overreach, missing sidecars, conductor-result
suppression, and flagged-input omission. Every attack SHALL fail closed.

Exp6406 SHALL write an append-only claim ledger with included and excluded
artifact hashes, reasons, allowed internal claims, forbidden claims, and the
exact evidence boundary hash. It SHALL set
`clean_factor_evidence_boundary_ready_score=1.0` only when every included row is
clean, every excluded row remains excluded, only narrow internal V550 claims
reproduce, and `public_factor_claim_eligibility=false`.

Exp6406 SHALL emit these fields:

- `status`
- `audit_registration_path_hash_and_expected_scope`
- `v550_artifact_hash_verdict_conductor_duration_and_flag_matrix`
- `clean_inclusion_rule`
- `explicit_exclusion_rule`
- `included_clean_artifact_records`
- `excluded_nonclean_blocked_null_absent_unlicensed_rejected_and_flagged_records`
- `exp6385_preservation_receipt`
- `exp6399_preservation_receipt`
- `recomputed_narrow_harness_license_frontier_learning_consumer_and_safety_states`
- `universal_support_claimed`
- `public_factor_claim_eligibility`
- `allowed_internal_claims`
- `forbidden_claims`
- `claim_ledger_path_hash_and_rows`
- `substitution_laundering_date_model_family_license_sidecar_conductor_and_flag_attack_matrix`
- `clean_factor_evidence_boundary_ready_score`
- `upstream_artifacts_modified`
- `protected_files_unchanged`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

## SCENARIO-LEARN-6406-REGISTRATION: Scope Freezes Before Conclusions

**Given** the expected V550 factor artifacts, sidecars, source files, licenses,
checker hashes, transaction heads, and conductor rows
**When** Exp6406 starts
**Then** it SHALL hash and register that scope before it reads upstream
readiness, utility, public-claim, or safety conclusion fields.

## SCENARIO-LEARN-6406-INCLUSION: Only Clean V550 Rows Enter The Boundary

**Given** positive V550 artifacts, Exp6385, Exp6399, Exp6403, unlicensed cells,
rejected cells, missing sidecars, and flagged inputs
**When** Exp6406 applies the inclusion and exclusion rules
**Then** only clean V550 factor evidence rows SHALL be included
**And** all nonclean, null, context-only, unlicensed, rejected, missing, and
flagged rows SHALL remain excluded with reasons and hashes.

## SCENARIO-LEARN-6406-RECOMPUTE: Claims Stay Narrow And Internal

**Given** Exp6394 through Exp6398 provide positive internal factor evidence
inside four licensed cells
**When** Exp6406 recomputes harness, license, frontier, learning, consumer, and
safety states
**Then** it SHALL reproduce only narrow internal V550 claims
**And** `universal_support_claimed` and `public_factor_claim_eligibility` SHALL
remain `false`.

## SCENARIO-LEARN-6406-ATTACKS: Boundary Attacks Fail Closed

**Given** substitution, lineage laundering, date relabeling, model swap, family
swap, license overreach, missing sidecar, conductor suppression, or flagged
input omission
**When** Exp6406 evaluates the attack matrix
**Then** each attack SHALL fail closed
**And** no attack SHALL add an included row, suppress an excluded row, or enable
a public claim.

## SCENARIO-LEARN-6406-LEDGER: Claim Ledger Binds The Boundary

**Given** included and excluded rows have hashes and reasons
**When** Exp6406 writes the claim ledger
**Then** the ledger SHALL record allowed internal claims, forbidden claims, and
the exact evidence boundary hash
**And** the terminal artifact SHALL record the ledger path, hash, row count, and
rows.

## SCENARIO-LEARN-6406-READY: Readiness Is Conjunctive

**Given** any included row is nonclean, any excluded row becomes included, any
required sidecar or provenance is missing, any attack succeeds, universal
support is claimed, public eligibility is true, an upstream artifact changed, a
protected file changed, or a recorded test fails
**When** Exp6406 recomputes readiness
**Then** `clean_factor_evidence_boundary_ready_score` SHALL be `0.0`.

## Implementation Status (REQ-LEARN-6406)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6406 | Planned: `python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py`; terminal artifact `results/experiment_6406_clean_v550_factor_evidence_boundary.json`. | Planned: `tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py`. |
| SCENARIO-LEARN-6406-REGISTRATION | Planned: `python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py`. | Planned: `tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py`. |
| SCENARIO-LEARN-6406-INCLUSION | Planned: `python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py`. | Planned: `tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py`. |
| SCENARIO-LEARN-6406-RECOMPUTE | Planned: `python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py`. | Planned: `tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py`. |
| SCENARIO-LEARN-6406-ATTACKS | Planned: `python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py`. | Planned: `tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py`. |
| SCENARIO-LEARN-6406-LEDGER | Planned: `python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py`. | Planned: `tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py`. |
| SCENARIO-LEARN-6406-READY | Planned: `python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py`. | Planned: `tests/python/test_experiment_6406_clean_v550_factor_evidence_boundary.py`. |

## REQ-LEARN-6407: Provenance-Linked Tiered Factor Memory Protocol

**Given** the V550 factor head, release ledger, lifecycle code, exact checkers,
licenses, and protected artifacts may drift
**When** Exp6407 runs on planning date 20260813
**Then** it SHALL write
`results/experiment_6407_provenance_tiered_factor_memory_protocol.json`
**And** it SHALL freeze a research-only two-tier factor memory protocol without
invoking an LLM, measuring learning utility, or giving authority to the
compiled cache.

Exp6407 SHALL hash the V550 factor-head schema, release ledger, lifecycle code,
exact checkers, license records, upstream artifacts, sidecars, and protected
files before building any memory row. Missing hashes SHALL fail closed.

Exp6407 SHALL define an append-only raw record schema with event hash, source
spans, model and harness identity, license key, exact checker version, release
outcome, predecessor, disposition, timestamps, expiry, supersession, and
transaction receipt. Each raw row SHALL have a canonical row hash.

Exp6407 SHALL define a compiled typed graph with factor, evidence, model,
constraint-family, checker, license, predecessor, expiry, and supersession
node or edge types. Each compiled row SHALL include at least one raw row hash.
The compiled graph is a cache. It SHALL fail closed to the raw tier on any
missing or conflicting provenance.

Exp6407 SHALL compute exact affected-neighborhood receipts for additions,
revocations, expiry, and supersession. Local replay over the affected
neighborhood SHALL match full replay on deterministic fixtures.

Exp6407 SHALL define raw-tier escalation for missing provenance, implicit
support, graph/cache disagreement, stale summaries, expired licenses,
unresolved supersession, and checker drift. Each escalation condition SHALL
have a deterministic test receipt.

Exp6407 SHALL freeze at least 48 controlled candidate events across supported,
contradicted, implicit, stale, duplicated, replayed, superseded, poisoned, and
clean-negative classes. Calibration, acquisition, retention, and future
partitions SHALL be sealed.

Exp6407 SHALL expose diagnostic admission features for utility, exact
confidence, novelty, recency, and content type. These features are
interpretable diagnostics only. No weighted diagnostic score SHALL override an
exact veto.

Exp6407 SHALL attack orphan summaries, forged raw links, cycle creation,
neighborhood underreach, neighborhood overreach, stale heads, partial atomic
writes, duplicate effects, expiry removal, and cache resurrection after
restart. Each attack SHALL fail closed to raw-tier escalation, quarantine, or
rejection.

Exp6407 SHALL set
`provenance_tiered_memory_protocol_ready_score=1.0` only when all compiled rows
trace to immutable raw evidence, local and full replay agree, every attack
fails closed, partitions are sealed, protected files stay unchanged, all tests
pass, `compiled_cache_authority_claimed=false`, and
`learning_utility_claimed=false`.

Exp6407 SHALL emit these fields:

- `status`
- `upstream_factor_head_release_ledger_lifecycle_checker_and_license_hashes`
- `raw_record_schema_path_hash_and_required_fields`
- `compiled_typed_graph_schema_path_hash_node_and_edge_types`
- `raw_to_compiled_provenance_link_receipts`
- `affected_neighborhood_equations_and_receipts`
- `local_vs_full_replay_equivalence_results`
- `raw_tier_escalation_rules_and_tests`
- `contamination_manifest_path_hash_counts_classes_and_partition_seals`
- `diagnostic_admission_feature_contract`
- `exact_veto_override_count`
- `supported_contradicted_implicit_stale_duplicate_replay_supersession_poison_and_negative_fixture_results`
- `orphan_forgery_cycle_neighborhood_head_atomic_duplicate_expiry_and_restart_attack_matrix`
- `compiled_cache_authority_claimed`
- `learning_utility_claimed`
- `provenance_tiered_memory_protocol_ready_score`
- `protected_files_unchanged`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

## SCENARIO-LEARN-6407-RAW-COMPILED: Every Cache Row Links To Raw Evidence

**Given** append-only raw rows and a compiled graph cache
**When** Exp6407 compiles factors, evidence, models, checkers, licenses,
predecessors, expiry edges, and supersession edges
**Then** each compiled row SHALL include raw row hashes that exist in the raw
ledger
**And** any missing or forged raw hash SHALL escalate to the raw tier.

## SCENARIO-LEARN-6407-REPLAY: Local Replay Matches Full Replay

**Given** additions, revocations, expiry, and supersession events
**When** Exp6407 replays the affected typed neighborhood
**Then** the local replay receipt SHALL match full replay exactly
**And** underreach or overreach SHALL fail closed.

## SCENARIO-LEARN-6407-ESCALATION: Raw Tier Handles Ambiguity

**Given** missing provenance, implicit support, graph disagreement, stale
summary, expired license, unresolved supersession, or checker drift
**When** Exp6407 checks the compiled cache
**Then** the decision SHALL escalate to raw evidence
**And** no compiled summary SHALL authorize a commit.

## SCENARIO-LEARN-6407-CONTAMINATION: Controlled Fixtures Are Sealed

**Given** at least 48 candidate events across supported, contradicted,
implicit, stale, duplicate, replay, supersession, poison, and clean-negative
classes
**When** Exp6407 builds the contamination protocol
**Then** the class counts and calibration, acquisition, retention, and future
partition seals SHALL be recorded
**And** diagnostic admission features SHALL not override an exact veto.

## SCENARIO-LEARN-6407-ATTACKS: Cache Attacks Fail Closed

**Given** orphan summary, forged raw link, cycle creation, neighborhood
underreach, neighborhood overreach, stale head, partial atomic write, duplicate
effect, expiry removal, or restart resurrection attack
**When** Exp6407 evaluates the attack matrix
**Then** every attack SHALL fail closed
**And** no attack SHALL claim cache authority or learning utility.

## SCENARIO-LEARN-6407-READY: Readiness Is Fully Conjunctive

**Given** any compiled row lacks raw evidence, local and full replay differ,
an escalation condition is missing, a partition seal is absent, a diagnostic
overrides an exact veto, an attack succeeds, a protected file changes, a test
fails, compiled cache authority is claimed, or learning utility is claimed
**When** Exp6407 refreshes readiness
**Then** `provenance_tiered_memory_protocol_ready_score` SHALL be `0.0`.

## Implementation Status (REQ-LEARN-6407)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6407 | Implemented: `python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py`; terminal artifact `results/experiment_6407_provenance_tiered_factor_memory_protocol.json`. | Implemented: `tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py`. |
| SCENARIO-LEARN-6407-RAW-COMPILED | Implemented: `python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py`. | Implemented: `tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py`. |
| SCENARIO-LEARN-6407-REPLAY | Implemented: `python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py`. | Implemented: `tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py`. |
| SCENARIO-LEARN-6407-ESCALATION | Implemented: `python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py`. | Implemented: `tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py`. |
| SCENARIO-LEARN-6407-CONTAMINATION | Implemented: `python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py`. | Implemented: `tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py`. |
| SCENARIO-LEARN-6407-ATTACKS | Implemented: `python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py`. | Implemented: `tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py`. |
| SCENARIO-LEARN-6407-READY | Implemented: `python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py`. | Implemented: `tests/python/test_experiment_6407_provenance_tiered_factor_memory_protocol.py`. |

## REQ-LEARN-6408: Powered Write-Time Factor Admission A/B

**Given** Exp6406 defines a clean V550-only evidence boundary, Exp6407 freezes
the raw and compiled memory protocol, and Exp6395 licenses exactly four
model-family cells
**When** Exp6408 starts on planning date 20260813
**Then** it SHALL write
`results/experiment_6408_powered_write_time_factor_admission_ab.json`
**And** it SHALL compare frozen baseline, write-everything, and
provenance-plus-exact admission arms only inside the four licensed cells.

Exp6408 SHALL use the three mandated local GGUF model ids from
`cached_sota_pair()`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. Exp6408 SHALL not call `AutoTokenizer`. Qwen cells and the
two unlicensed Gemma cells SHALL record abstention without fallback.

Exp6408 SHALL revalidate the Exp6406 clean-boundary gate, the Exp6407
protocol gate, Exp6395 licenses, frozen harnesses, schemas, model files,
embedded tokenizers, RTX 3090 CUDA offload, and exact checker hashes before
any arm runs.

Exp6408 SHALL seal at least 36 fresh held events. Events SHALL be balanced
across the four licensed cells and contamination classes. The manifest SHALL
prove disjointness from V550 and Exp6407 development fixtures before
generation and before scoring.

Exp6408 SHALL freeze raw model bytes, parser-independent source spans,
proposed typed effects, diagnostic features, exact support receipts,
admission dispositions, and the memory head hash before future outcomes are
visible.

The provenance-plus-exact arm SHALL admit only exact-supported, source-bound,
license-valid, predecessor-fresh proposals. Contradicted, implicit, stale,
duplicate, replayed, superseded, poisoned, malformed, and unlicensed rows
SHALL reject, quarantine, defer, or abstain.

Exp6408 SHALL report proposal transport, exact evaluability, admission
precision and recall, contamination propagation, future exact yield, false
accepts and false rejects, raw escalation, abstention, latency, verification
cost, and GPU memory by arm and licensed cell. It SHALL emit
`delta_future_exact_yield` and `delta_contamination_propagation_rate` as
finite bare numbers.

Exp6408 SHALL attack model and family swaps, license inheritance, harness
drift, source substitution, exact-check omission, diagnostic veto override,
stale heads, duplicate evidence, pooled abstention, and future-label leakage.
Every attack SHALL fail closed.

Exp6408 SHALL set `powered_write_time_admission_ready_score=1.0` only when the
powered arms run, provenance admission beats write-everything on future exact
yield, contamination propagation does not increase over frozen and is lower
than write-everything, false accepts do not increase, every unlicensed cell
abstains, protected leakage is zero, tests pass, and no model weights change.

Exp6408 SHALL emit these fields:

- `status`
- `exp6406_and_exp6407_gate_receipts`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `model_file_hashes_revisions_quantizations_and_tokenizers`
- `embedded_gguf_tokenizer_receipts`
- `autotokenizer_usage_count`
- `license_and_frozen_harness_bindings`
- `unlicensed_and_rejected_cell_abstention_records`
- `cuda_offload_runtime_peak_memory_and_duration_receipts_by_model`
- `held_manifest_path_hash_counts_balance_partition_seals_and_disjointness`
- `preregistered_frozen_write_everything_and_exact_admission_arm_contract`
- `matched_work_receipts`
- `raw_bytes_source_effect_diagnostic_checker_disposition_and_head_freeze_records`
- `per_arm_model_family_contamination_admission_yield_harm_escalation_abstention_and_cost_results`
- `exact_future_yield_by_arm`
- `contamination_propagation_rate_by_arm`
- `delta_future_exact_yield`
- `delta_contamination_propagation_rate`
- `false_accept_false_reject_and_negative_transfer_results`
- `confidence_intervals_and_effective_sample_sizes`
- `model_license_harness_source_checker_diagnostic_head_duplicate_pooling_and_leakage_attack_matrix`
- `silent_fallback_count`
- `exact_veto_override_count`
- `protected_leakage_count`
- `model_weight_change_count`
- `powered_write_time_admission_ready_score`
- `universal_support_claimed`
- `public_factor_claim_eligibility`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map the Exp6406 gate, the Exp6407 gate,
`delta_future_exact_yield`, `delta_contamination_propagation_rate`, and
`powered_write_time_admission_ready_score` to their purposes. `verifier_is_oracle`
SHALL be true only for exact event checkers.

## SCENARIO-LEARN-6408-LICENSED-CELLS: Write-Time Arms Stay Licensed

**Given** Exp6395 licenses four cells and rejects or abstains from the rest
**When** Exp6408 runs the powered A/B
**Then** only licensed cells SHALL receive arm work
**And** every unlicensed or rejected cell SHALL abstain with zero fallback.

## SCENARIO-LEARN-6408-FRESH-MANIFEST: Held Events Are Fresh And Balanced

**Given** V550 and Exp6407 development fixtures already exist
**When** Exp6408 seals its held manifest
**Then** it SHALL create at least 36 fresh events balanced across licensed
cells and contamination classes
**And** it SHALL prove disjointness before generation and before scoring.

## SCENARIO-LEARN-6408-ADMISSION: Exact Support Owns Admission

**Given** raw bytes, source spans, typed effects, diagnostics, checker
receipts, and a head hash are frozen before future outcomes
**When** Exp6408 admits a proposal
**Then** exact-supported, source-bound, license-valid, predecessor-fresh rows
SHALL admit
**And** contradicted, implicit, stale, duplicate, replayed, superseded,
poisoned, malformed, and unlicensed rows SHALL not admit.

## SCENARIO-LEARN-6408-MATCHED-ARMS: Powered Arms Use Equal Work

**Given** frozen baseline, write-everything, and exact-admission arms
**When** Exp6408 executes them
**Then** prompts, event order, token budgets, checker calls, consumer budget,
models, and cells SHALL match.

## SCENARIO-LEARN-6408-ATTACKS: Admission Attacks Fail Closed

**Given** model swap, family swap, license inheritance, harness drift, source
substitution, exact-check omission, diagnostic veto override, stale head,
duplicate evidence, pooled abstention, or future-label leakage
**When** Exp6408 evaluates the attack matrix
**Then** no attack SHALL promote readiness.

## SCENARIO-LEARN-6408-READY: Readiness Requires Better Utility And Lower Harm

**Given** the powered arms ran, all unlicensed cells abstained, false accepts
did not increase, protected leakage is zero, and tests passed
**When** provenance admission beats write-everything on future exact yield and
has lower contamination propagation than write-everything without increasing
over frozen
**Then** `powered_write_time_admission_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6408)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6408 | Planned: `python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py`; terminal artifact `results/experiment_6408_powered_write_time_factor_admission_ab.json`. | Planned: `tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6408-LICENSED-CELLS | Planned: `python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py`. | Planned: `tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6408-FRESH-MANIFEST | Planned: `python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py`. | Planned: `tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6408-ADMISSION | Planned: `python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py`. | Planned: `tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6408-MATCHED-ARMS | Planned: `python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py`. | Planned: `tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6408-ATTACKS | Planned: `python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py`. | Planned: `tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6408-READY | Planned: `python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py`. | Planned: `tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py`. |

## REQ-LEARN-6417: Authentic Write-Time Factor Admission A/B

**Given** Exp6412 quarantines the old Exp6408 powered claim, Exp6414 provides
fresh authenticated model events, and Exp6416 proves safe exact refinement
**When** Exp6417 runs on planning date 20260814
**Then** it SHALL write
`results/experiment_6417_authentic_write_time_factor_admission_ab.json`
**And** it SHALL perform no new model generation.

Exp6417 SHALL revalidate the Exp6412, Exp6414, Exp6416, and old Exp6408
receipts before any arm runs. It SHALL recheck corpus hashes, process receipt
hashes, raw output bytes, event order, partition seals, exact checker hashes,
selective-refinement contract, licenses, and the initial factor head.

Exp6417 SHALL freeze acquisition, protected retention, and untouched future
partitions. No proposal or disposition SHALL read future labels before the
future evaluation opens once after all write-time heads freeze.

Exp6417 SHALL compare three matched arms: `frozen`, `write_everything`, and
`provenance_plus_exact`. The arms SHALL use the same chronological event order,
raw source evidence, exact checker calls, consumer budget, and initial head.

Every proposal SHALL bind raw event hashes, raw output hashes, source spans,
model identity, harness identity, license identity, exact support, predecessor
head, refinement receipt, expiry, and supersession state. Each row SHALL receive
exactly one atomic disposition: `Commit`, `Reject`, `Quarantine`, or `Defer`.
Contradicted, implicit, stale, duplicate, replayed, superseded, poisoned,
malformed, unlicensed, stale-head, and missing-exact rows SHALL fail closed.

Exp6417 SHALL evaluate the untouched future partition once. It SHALL report
exact yield, contamination propagation, false accepts, false rejects, protected
retention, abstention, factor growth, escalation, and exact work by arm and by
cell.

Exp6417 SHALL attack receipt substitution, source replacement, model-family
swap, license inheritance, exact-check omission, stale head, duplicate effect,
future-label leakage, and diagnostic veto override. Every attack SHALL fail
closed.

Exp6417 SHALL emit `delta_future_exact_yield`,
`delta_contamination_propagation_rate`, and `protected_retention_delta` as bare
finite numbers.

Exp6417 SHALL set `authentic_write_time_admission_ready_score=1.0` only when
`provenance_plus_exact` beats `frozen` on untouched future exact yield,
contamination does not increase over `frozen`, contamination is below
`write_everything`, protected retention does not regress, every attack fails
closed, and all verification commands pass.

Exp6417 SHALL emit these fields:

- `status`
- `exp6412_exp6414_and_exp6416_gate_receipts`
- `upstream_MODEL_SPECS_and_models_used`
- `upstream_process_receipt_and_raw_output_hashes`
- `corpus_event_order_partition_checker_license_and_head_hashes`
- `preregistered_frozen_write_everything_and_exact_admission_arm_contract`
- `matched_work_receipts`
- `per_proposal_raw_source_model_license_checker_predecessor_refinement_expiry_and_supersession_bindings`
- `atomic_disposition_records`
- `per_arm_cell_exact_yield_contamination_false_accept_false_reject_retention_abstention_growth_escalation_and_work_results`
- `untouched_future_evaluation_receipts`
- `delta_future_exact_yield`
- `delta_contamination_propagation_rate`
- `protected_retention_delta`
- `silent_fallback_count`
- `exact_veto_override_count`
- `protected_leakage_count`
- `runtime_field_synthesis_count`
- `attack_matrix`
- `authentic_write_time_admission_ready_score`
- `public_factor_claim_eligibility`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field and these gate keys:
`gate:exp6412`, `gate:exp6414`, `gate:exp6416`,
`gate:exp6408_quarantine`, `gate:raw_outputs`, `gate:event_order`,
`gate:licenses`, and `gate:initial_factor_head`. It SHALL map both deltas,
`protected_retention_delta`, and
`authentic_write_time_admission_ready_score`. `verifier_is_oracle` SHALL be
true only for exact event and retention checkers. Upstream model output,
admission, memory, and diagnostics SHALL NOT be oracles.

Required field principles:

- `status`: Names the terminal safety state for the authentic write-time replay.
- `exp6412_exp6414_and_exp6416_gate_receipts`: Pins the claim audit, fresh corpus, and exact-refinement gates.
- `upstream_MODEL_SPECS_and_models_used`: Carries only upstream model identities and marks no new generation.
- `upstream_process_receipt_and_raw_output_hashes`: Binds process receipts and raw bytes before any parser can act.
- `corpus_event_order_partition_checker_license_and_head_hashes`: Seals order, partitions, checkers, licenses, and the initial head.
- `preregistered_frozen_write_everything_and_exact_admission_arm_contract`: Defines the three matched arms before future labels open.
- `matched_work_receipts`: Shows equal row order, checker calls, consumer budget, and initial head.
- `per_proposal_raw_source_model_license_checker_predecessor_refinement_expiry_and_supersession_bindings`: Binds each proposal to raw source, model, license, checker, head, refinement, expiry, and supersession data.
- `atomic_disposition_records`: Records one Commit, Reject, Quarantine, or Defer decision for each proposal.
- `per_arm_cell_exact_yield_contamination_false_accept_false_reject_retention_abstention_growth_escalation_and_work_results`: Reports arm and cell metrics without pooled masking.
- `untouched_future_evaluation_receipts`: Proves future labels open once after write-time heads freeze.
- `delta_future_exact_yield`: Bare future exact-yield lift for exact admission over frozen.
- `delta_contamination_propagation_rate`: Bare contamination-rate change for exact admission over frozen.
- `protected_retention_delta`: Bare protected-retention change for exact admission over frozen.
- `silent_fallback_count`: Must be zero because unlicensed work cannot use substitute paths.
- `exact_veto_override_count`: Must be zero because exact rejections cannot be overridden.
- `protected_leakage_count`: Must be zero because future and protected labels cannot route writes.
- `runtime_field_synthesis_count`: Must be zero because runtime fields come from receipts, not invention.
- `attack_matrix`: Shows substitution, source, model, license, checker, head, duplicate, leakage, and diagnostic attacks fail closed.
- `authentic_write_time_admission_ready_score`: Conjunctive score for future gain without contamination or retention harm.
- `public_factor_claim_eligibility`: Limits public eligibility to this authenticated replay and excludes Exp6408.
- `harm_underpowered_missing_and_flagged_cells`: Keeps quarantined, unlicensed, unsupported, underpowered, and attacked cells visible.
- `protected_files_unchanged`: Shows protected upstream and ops files stayed byte-identical.
- `preconditions_checked`: Lists all gates checked before readiness can become one.
- `inference_substrate`: Declares deterministic replay over upstream receipts with no new model generation.
- `verifier_is_oracle`: Marks only exact event and retention checkers as oracles.
- `field_principles`: Documents why each field exists.
- `field_provenance`: Maps each field to upstream receipts, replay, exact checks, attacks, or tests.
- `random_seed`: Pins the replay constants.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records verification commands and exit codes.
- `reproducibility_checksum`: Content-addresses the payload with volatile fields normalized.
- `honest_verdict`: Uses a terminal prefix and states the authentic replay boundary.
- `gate:exp6412`: Exp6412 must quarantine the old powered claim before Exp6417 can run.
- `gate:exp6414`: Exp6414 is the only fresh model-event corpus used here.
- `gate:exp6416`: Exp6416 supplies the exact-refinement contract and not model authority.
- `gate:exp6408_quarantine`: Exp6408 is audited as old unauthentic evidence, not reused as proof.
- `gate:raw_outputs`: Raw output files and receipt hashes must match before proposals bind.
- `gate:event_order`: Chronological order and partitions must stay sealed.
- `gate:licenses`: License validity controls commits and blocks inheritance.
- `gate:initial_factor_head`: All arms start from the same read-only head.

## SCENARIO-LEARN-6417-GATES: Old Powered Claim Is Quarantined

**Given** Exp6408 exists only as an unauthentic powered claim after Exp6412
**When** Exp6417 checks upstream evidence
**Then** Exp6412 SHALL mark powered and public factor claims ineligible
**And** Exp6414 and Exp6416 SHALL provide the only usable model-event evidence.

## SCENARIO-LEARN-6417-MATCHED-ARMS: Arms Use The Same Work Surface

**Given** the frozen Exp6414 acquisition and retention rows
**When** Exp6417 runs its three arms
**Then** each arm SHALL use the same row order, raw hashes, source spans,
checker calls, consumer budget, and initial factor head.

## SCENARIO-LEARN-6417-ADMISSION: Exact Support Owns Commits

**Given** a write-time proposal is source-bound and license-valid
**When** it lacks exact support, has a stale predecessor, is implicit, is
contradicted, is stale, is duplicate, is replayed, is superseded, is poisoned,
is malformed, or is unlicensed
**Then** `provenance_plus_exact` SHALL not commit it.

## SCENARIO-LEARN-6417-FUTURE: Future Labels Open Once

**Given** all write-time dispositions are recorded and heads are frozen
**When** Exp6417 evaluates the untouched future partition
**Then** future labels SHALL open exactly once and SHALL not appear in any
proposal-time binding.

## SCENARIO-LEARN-6417-ATTACKS: Admission Attacks Fail Closed

**Given** receipt substitution, source replacement, model-family swap, license
inheritance, exact-check omission, stale head, duplicate effect, future-label
leakage, or diagnostic veto override
**When** Exp6417 validates the attack matrix
**Then** no attack SHALL commit a factor or promote readiness.

## SCENARIO-LEARN-6417-READY: Readiness Requires Future Gain Without Harm

**Given** all verification commands pass and every attack fails closed
**When** exact admission improves future exact yield over frozen, contamination
does not increase over frozen, contamination is below write-everything, and
protected retention does not regress
**Then** `authentic_write_time_admission_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6417)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6417 | Implemented: `python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py`; terminal artifact `results/experiment_6417_authentic_write_time_factor_admission_ab.json`. | Implemented: `tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6417-GATES | Implemented: `python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6417-MATCHED-ARMS | Implemented: `python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6417-ADMISSION | Implemented: `python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6417-FUTURE | Implemented: `python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6417-ATTACKS | Implemented: `python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6417-READY | Implemented: `python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6417_authentic_write_time_factor_admission_ab.py`. |

## REQ-LEARN-6428: Clean Write-Time Factor Admission A/B

**Given** Exp6427 provides a clean row-recomputable factor corpus and Exp6417 is
adversarial-flagged for implausible deterministic replay duration
**When** Exp6428 runs on planning date 20260814
**Then** it SHALL write
`results/experiment_6428_clean_write_time_factor_admission_ab.json`
**And** it SHALL perform no new model generation.

Exp6428 SHALL revalidate Exp6427 structured gates, row hashes, task receipts,
event order, partitions, exact checkers, licenses, initial heads, disk, CPU,
RAM, and untouched future seal before any arm runs. Exp6427 `calibration` rows
SHALL serve as the protected retention control partition for this replay.

Exp6428 SHALL freeze acquisition, protected retention, and future partitions.
It SHALL prove future outcomes were unavailable when every proposal and
disposition was recorded. It SHALL compare `frozen`, `write_everything`, and
`exact_admission` arms with matched row order, evidence, checker calls,
consumer budget, and initial head.

Exp6428 SHALL bind every proposal to raw event, model, prompt, source, license,
checker, predecessor, expiry, supersession, and refinement hashes. Each
proposal SHALL receive exactly one atomic disposition: `Commit`, `Reject`,
`Quarantine`, or `Defer`. Contradicted, implicit, stale, duplicate, replayed,
superseded, poisoned, malformed, unlicensed, and stale-head rows SHALL fail
closed.

Exp6428 SHALL write one per-unit future row for every arm and cell before any
aggregate. It SHALL derive exact yield, contamination, false accepts, false
rejects, protected retention, abstention, growth, escalation, and work only in
an independent reduction pass.

Exp6428 SHALL attack receipt substitution, source replacement, model swap,
license inheritance, checker omission, stale head, duplicate effect, future
leakage, exact-veto override, row deletion, and duration synthesis. Every
attack SHALL fail closed.

Exp6428 SHALL emit `delta_future_exact_yield`,
`delta_contamination_propagation_rate`, and `protected_retention_delta` as bare
finite numbers. It SHALL set `clean_write_time_admission_ready_score=1.0` only
when exact admission beats frozen future exact yield, contamination does not
increase and remains below write-everything, protected retention does not
regress, aggregates recompute exactly, all attacks fail closed, and
`current_adversarial_flag_count` is zero.

Exp6428 SHALL emit these fields:

- `status`
- `exp6427_gate_receipts`
- `upstream_model_process_raw_output_and_row_hashes`
- `corpus_event_order_partition_checker_license_and_head_hashes`
- `preregistered_frozen_write_everything_and_exact_admission_arm_contract`
- `matched_work_receipts`
- `per_unit_rows`
- `per_proposal_source_model_license_checker_predecessor_expiry_and_supersession_bindings`
- `atomic_disposition_records`
- `untouched_future_evaluation_receipts`
- `aggregate_recomputation_receipts`
- `reported_vs_recomputed_deltas`
- `delta_future_exact_yield`
- `delta_contamination_propagation_rate`
- `protected_retention_delta`
- `false_accept_delta`
- `false_reject_delta`
- `factor_growth_by_arm`
- `exact_work_by_arm`
- `exact_veto_override_count`
- `protected_leakage_count`
- `runtime_field_synthesis_count`
- `task_phase_duration_receipts`
- `attack_matrix`
- `clean_write_time_admission_ready_score`
- `current_adversarial_flag_count`
- `public_factor_claim_eligibility`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `blocked_reason`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map both gates, all arms, each delta, the flag count,
and the readiness score. `verifier_is_oracle` SHALL be true only for exact event
and protected-retention checks. Admission and memory SHALL NOT be oracles.

Required field principles:

- `status`: Names the terminal safety state for the clean Exp6427 replay.
- `exp6427_gate_receipts`: Pins the clean corpus gate and the Exp6417 duration quarantine context.
- `upstream_model_process_raw_output_and_row_hashes`: Binds model, prompt, process, raw output, and row hashes before parsing can act.
- `corpus_event_order_partition_checker_license_and_head_hashes`: Seals event order, partitions, checkers, licenses, disk, CPU, RAM, and the initial head.
- `preregistered_frozen_write_everything_and_exact_admission_arm_contract`: Defines all three arms before future outcomes open.
- `matched_work_receipts`: Shows equal row order, evidence, checker calls, consumer budget, and initial head for all arms.
- `per_unit_rows`: Records one future outcome row for every arm and cell before aggregate calculation.
- `per_proposal_source_model_license_checker_predecessor_expiry_and_supersession_bindings`: Binds every proposal to source, model, license, checker, predecessor, expiry, supersession, and refinement hashes.
- `atomic_disposition_records`: Records exactly one Commit, Reject, Quarantine, or Defer decision for every proposal.
- `untouched_future_evaluation_receipts`: Proves future outcomes open once after proposal dispositions and head freeze.
- `aggregate_recomputation_receipts`: Recomputes every comparative aggregate from per-unit rows in an independent pass.
- `reported_vs_recomputed_deltas`: Shows reported deltas and reductions match the independent recomputation.
- `delta_future_exact_yield`: Bare future exact-yield lift for exact admission over frozen.
- `delta_contamination_propagation_rate`: Bare contamination-rate change for exact admission over frozen.
- `protected_retention_delta`: Bare protected-retention change for exact admission over frozen.
- `false_accept_delta`: Bare false-accept rate change for exact admission over frozen.
- `false_reject_delta`: Bare false-reject rate change for exact admission over frozen.
- `factor_growth_by_arm`: Reports committed factor growth per arm.
- `exact_work_by_arm`: Reports exact checker work per arm under the matched budget.
- `exact_veto_override_count`: Must be zero because exact rejections cannot be overridden.
- `protected_leakage_count`: Must be zero because protected and future labels cannot route writes.
- `runtime_field_synthesis_count`: Must be zero because runtime fields come from receipts.
- `task_phase_duration_receipts`: Records monotonic phase timing without synthetic duration fields.
- `attack_matrix`: Shows substitution, source, model, license, checker, head, duplicate, leakage, veto, deletion, and duration attacks fail closed.
- `clean_write_time_admission_ready_score`: Conjunctive score for future gain without contamination, retention harm, aggregate drift, or adversarial flags.
- `current_adversarial_flag_count`: Must stay zero for the clean Exp6427 replay.
- `public_factor_claim_eligibility`: Limits public eligibility to this clean replay and excludes the flagged Exp6417 timing claim.
- `harm_underpowered_missing_and_flagged_cells`: Keeps unlicensed, underpowered, missing, blocked, and flagged cells visible.
- `protected_files_unchanged`: Shows protected upstream and ops files stayed byte-identical.
- `blocked_reason`: Explains why readiness is blocked when any precondition fails.
- `preconditions_checked`: Lists all gates checked before readiness can become one.
- `inference_substrate`: Declares cached Exp6427 deterministic replay with no new model generation.
- `verifier_is_oracle`: Marks only exact event and protected-retention checkers as oracles.
- `field_principles`: Documents why each field exists.
- `field_provenance`: Maps each field to specs, inputs, replay, reductions, attacks, or tests.
- `random_seed`: Pins the replay constants.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records verification commands and exit codes.
- `reproducibility_checksum`: Content-addresses the payload with volatile fields normalized.
- `honest_verdict`: Uses a terminal prefix and states the clean replay boundary.
- `gate:exp6427`: Exp6427 must be complete, clean, row-recomputable, and adversarial-clean before Exp6428 can promote readiness.
- `gate:exp6417_duration_quarantine`: Exp6417 is context only because its deterministic replay duration is adversarial-flagged.
- `gate:raw_outputs`: Raw output files and stored hashes must match before proposals bind.
- `gate:event_order`: Chronological order and partitions must stay sealed.
- `gate:licenses`: License validity controls commits and blocks inheritance.
- `gate:initial_factor_head`: All arms start from the same read-only head.
- `arm:frozen`: Frozen reads the future with no write-time state.
- `arm:write_everything`: Write-everything commits every licensed proposal and acts as the contamination control.
- `arm:exact_admission`: Exact admission commits only licensed joint-exact proposal rows.

## SCENARIO-LEARN-6428-GATES: Clean Exp6427 Corpus Gates The Replay

**Given** Exp6427 is the only corpus source
**When** Exp6428 checks preconditions
**Then** Exp6427 readiness, row hashes, task receipts, partitions, licenses,
and current adversarial flag count SHALL pass before readiness can become one.

## SCENARIO-LEARN-6428-MATCHED-ARMS: Arms Use The Same Clean Rows

**Given** the sealed Exp6427 acquisition and calibration rows
**When** Exp6428 records proposal dispositions
**Then** all three arms SHALL use the same row order, evidence hashes, checker
calls, consumer budget, and initial head.

## SCENARIO-LEARN-6428-ADMISSION: Exact Support Owns Clean Commits

**Given** a clean Exp6427 proposal row
**When** it is unlicensed, non-exact, stale, duplicate, replayed, superseded,
poisoned, malformed, source-mismatched, or stale-head
**Then** `exact_admission` SHALL not commit it.

## SCENARIO-LEARN-6428-FUTURE: Per-Unit Future Rows Precede Aggregates

**Given** all proposal dispositions are recorded
**When** Exp6428 opens the untouched future partition once
**Then** it SHALL record one per-unit row per arm and future cell before
aggregate calculation.

## SCENARIO-LEARN-6428-ATTACKS: Substitution And Leakage Attacks Fail Closed

**Given** receipt substitution, source replacement, model swap, license
inheritance, checker omission, stale head, duplicate effect, future leakage,
exact-veto override, row deletion, or duration synthesis
**When** Exp6428 validates the attack matrix
**Then** no attack SHALL commit a factor or promote readiness.

## SCENARIO-LEARN-6428-READY: Readiness Requires Clean Future Gain

**Given** all verification commands pass and every attack fails closed
**When** exact admission improves future exact yield, avoids contamination
regression, stays below write-everything contamination, preserves protected
retention, and has zero adversarial flags
**Then** `clean_write_time_admission_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6428)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6428 | Implemented: `python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py`; terminal artifact `results/experiment_6428_clean_write_time_factor_admission_ab.json`. | Implemented: `tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6428-GATES | Implemented: `python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6428-MATCHED-ARMS | Implemented: `python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6428-ADMISSION | Implemented: `python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6428-FUTURE | Implemented: `python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6428-ATTACKS | Implemented: `python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py`. |
| SCENARIO-LEARN-6428-READY | Implemented: `python/carnot/experiment_6428_clean_write_time_factor_admission_ab.py`. | Implemented: `tests/python/test_experiment_6428_clean_write_time_factor_admission_ab.py`. |

## REQ-LEARN-6418: Execution-Grounded Dual-Path Continuous Self-Learning

**Given** Exp6417 has an authentic exact-admission replay, Exp6413 provides
authenticated local GGUF execution receipts, Exp6407 defines raw and compiled
memory schemas, and Exp6397 defines predecessor-bound transactions
**When** Exp6418 runs on planning date 20260814
**Then** it SHALL write
`results/experiment_6418_execution_grounded_dual_path_csl.json`
**And** it SHALL compare frozen, single-path exact transactional, and
dual-path learners over at least four sessions.

Exp6418 SHALL revalidate Exp6417 gates, the authenticated receipt layer, model
files, GPUs, raw and compiled memory schemas, licenses, exact checkers, initial
heads, rollback receipts, protected partitions, and Exp6413 receipt bindings
before any new session event is generated.

Exp6418 SHALL use exactly these local GGUF model ids from `cached_sota_pair()`:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. Exp6418 SHALL not call `AutoTokenizer`.

Exp6418 SHALL preregister at least 96 chronological events across four
sessions, three drift regimes, six update opportunities, four process
restarts, two expiry boundaries, and two supersession boundaries. Future rows
SHALL be sealed before generation. Frozen, single-path exact transactional, and
dual-path arms SHALL match event order, model calls, prompts, tokens, checker
calls, consumer work, and initial heads.

Exp6418 SHALL generate new session events through Exp6413's receipt layer. It
SHALL freeze raw bytes and proposals before exact outcomes are exposed. Exact
feasibility labels and exact consequence labels SHALL open only in causal
chronological order.

The proposal-memory path SHALL update only from exact feasible-action evidence.
The selection-memory path SHALL update only from exact observed consequences.
Typed heads and predecessor hashes SHALL remain separate. Every write SHALL be
evaluated off-commit before atomic disposition.

Each memory write SHALL atomically Commit, Reject, Quarantine, or Defer after
exact support, protected retention, unique effect, license, expiry,
supersession, and predecessor checks. Exact release, exact retention, exact
feasibility, and exact consequence checks SHALL control activation. No learned
path or model output SHALL override an exact veto.

Exp6418 SHALL attack contamination, stale heads, duplicate effects, concurrent
proposals, interrupted writes, expired licenses, superseded evidence, cache
resurrection, model swaps, delayed outcomes, and restart corruption. Harmful
descendants SHALL roll back. Every attack SHALL fail closed.

Exp6418 SHALL report proposal coverage, top-1 exact success, prequential
future exact yield, forward transfer, backward retention, forgetting, negative
transfer, contamination, growth, escalation, restart recovery, and cost by arm,
session, model, and family.

Exp6418 SHALL emit `delta_proposal_coverage_over_frozen`,
`delta_selection_success_over_frozen`, and
`delta_future_exact_yield_over_frozen` as bare finite numbers.

Exp6418 SHALL set `execution_grounded_dual_path_csl_ready_score=1.0` only when
both learning paths receive causal exact outcomes, future yield improves,
contamination is zero, protected retention survives rollback, growth is
bounded, and every attack fails closed.

Exp6418 SHALL emit these fields:

- `status`
- `exp6417_gate_receipts`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `model_file_and_embedded_tokenizer_hashes`
- `autotokenizer_usage_count`
- `cuda_offload_and_authenticated_process_receipts_by_model`
- `chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals`
- `preregistered_frozen_single_path_and_dual_path_arm_contract`
- `matched_work_receipts`
- `raw_event_and_pre_outcome_proposal_freeze_records`
- `exact_feasibility_and_consequence_outcome_receipts`
- `proposal_memory_schema_head_and_transition_history`
- `selection_memory_schema_head_and_transition_history`
- `predecessor_license_checker_expiry_and_supersession_bindings`
- `atomic_disposition_records`
- `commit_reject_quarantine_and_defer_counts_by_path_and_session`
- `per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results`
- `delta_proposal_coverage_over_frozen`
- `delta_selection_success_over_frozen`
- `delta_future_exact_yield_over_frozen`
- `contamination_propagation_rate`
- `forgetting_delta`
- `protected_leakage_count`
- `same_step_write_count`
- `exact_veto_override_count`
- `model_weight_change_count`
- `attack_matrix`
- `execution_grounded_dual_path_csl_ready_score`
- `public_factor_claim_eligibility`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field and these gate keys:
`gate:exp6417`, `gate:exp6413`, `gate:exp6407`, `gate:exp6397`,
`gate:model_files`, `gate:gpu_receipts`, `gate:schemas`, `gate:licenses`,
`gate:exact_checkers`, `gate:initial_heads`, `gate:rollback`, and
`gate:protected_partitions`. It SHALL map `learning_path:proposal`,
`learning_path:selection`, all three deltas, `contamination_propagation_rate`,
`forgetting_delta`, and `execution_grounded_dual_path_csl_ready_score`.
`verifier_is_oracle` SHALL be true only for exact feasibility, consequence,
release, and retention checks. Learned proposal memory, learned selection
memory, and model outputs SHALL NOT be oracles.

Required field principles:

- `status`: Names the terminal state for the exact-governed dual-path CSL run.
- `exp6417_gate_receipts`: Pins the authentic write-time admission gate.
- `MODEL_SPECS`: Carries the three mandated GGUF model identities from cached SOTA receipts.
- `models_used`: Lists only mandated models with authenticated local receipt support.
- `cached_sota_pair_receipts`: Records helper calls so manual model substitution is detectable.
- `model_file_and_embedded_tokenizer_hashes`: Binds model files and embedded GGUF tokenizer hashes.
- `autotokenizer_usage_count`: Must be zero because external tokenizer paths are forbidden.
- `cuda_offload_and_authenticated_process_receipts_by_model`: Binds CUDA, process, command, raw output, and cleanup receipts.
- `chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals`: Seals sessions, drift, updates, restarts, expiry, supersession, and future partitions.
- `preregistered_frozen_single_path_and_dual_path_arm_contract`: Defines all three arms before labels open.
- `matched_work_receipts`: Shows event order, model calls, prompts, tokens, checker calls, consumer work, and initial heads match.
- `raw_event_and_pre_outcome_proposal_freeze_records`: Proves raw bytes and proposals froze before outcomes.
- `exact_feasibility_and_consequence_outcome_receipts`: Opens exact labels only after freeze and in causal order.
- `proposal_memory_schema_head_and_transition_history`: Records proposal-memory updates from exact feasible-action evidence only.
- `selection_memory_schema_head_and_transition_history`: Records selection-memory updates from exact observed consequences only.
- `predecessor_license_checker_expiry_and_supersession_bindings`: Binds every write to predecessor, license, checker, expiry, and supersession receipts.
- `atomic_disposition_records`: Records one Commit, Reject, Quarantine, or Defer for every write.
- `commit_reject_quarantine_and_defer_counts_by_path_and_session`: Keeps disposition counts visible per path and session.
- `per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results`: Reports all metrics without pooled masking.
- `delta_proposal_coverage_over_frozen`: Bare proposal-coverage lift for dual path over frozen.
- `delta_selection_success_over_frozen`: Bare selection-success lift for dual path over frozen.
- `delta_future_exact_yield_over_frozen`: Bare future exact-yield lift for dual path over frozen.
- `contamination_propagation_rate`: Must remain zero for readiness.
- `forgetting_delta`: Must show no protected forgetting.
- `protected_leakage_count`: Must be zero because protected partitions cannot route writes.
- `same_step_write_count`: Must be zero because current outcomes cannot influence same-step decisions.
- `exact_veto_override_count`: Must be zero because exact vetoes cannot be overridden.
- `model_weight_change_count`: Must be zero because CSL changes external memory only.
- `attack_matrix`: Shows every contamination, head, duplicate, concurrency, interruption, license, supersession, cache, model, delay, and restart attack fails closed.
- `execution_grounded_dual_path_csl_ready_score`: Conjunctive readiness score for exact-governed dual-path learning.
- `public_factor_claim_eligibility`: Allows public claim only for this exact-governed run and not for learned scores as authority.
- `harm_underpowered_missing_and_flagged_cells`: Keeps missing, underpowered, flagged, and attacked cells visible.
- `protected_files_unchanged`: Shows protected files stayed byte-identical.
- `preconditions_checked`: Lists every gate checked before readiness can become one.
- `inference_substrate`: Declares authenticated local GGUF receipt replay with exact-governed memory updates.
- `verifier_is_oracle`: Marks only exact feasibility, consequence, release, and retention checkers as oracles.
- `field_principles`: Documents why each field exists.
- `field_provenance`: Maps each field to upstream receipts, exact checks, manifests, attacks, tests, or code.
- `random_seed`: Pins session order, arm work, updates, attacks, and metric fixtures.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records verification commands and exit codes.
- `reproducibility_checksum`: Content-addresses the artifact with volatile fields normalized.
- `honest_verdict`: Uses a terminal prefix and states the exact-governed dual-path boundary.
- `gate:exp6417`: Exp6417 must be authentic and ready before Exp6418 can run.
- `gate:exp6413`: Exp6413 supplies authenticated GGUF process and raw-output receipts.
- `gate:exp6407`: Exp6407 supplies raw and compiled memory schema receipts.
- `gate:exp6397`: Exp6397 supplies predecessor-bound transaction and rollback discipline.
- `gate:model_files`: Model files must exist and match recorded hashes.
- `gate:gpu_receipts`: CUDA and process receipts must be bound to each model.
- `gate:schemas`: Raw and compiled memory schemas must be present and hash-bound.
- `gate:licenses`: License validity controls commits and blocks inheritance.
- `gate:exact_checkers`: Exact checkers own feasibility, consequence, release, and retention labels.
- `gate:initial_heads`: Proposal and selection heads start from separate read-only hashes.
- `gate:rollback`: Harmful descendants must roll back to prior exact heads.
- `gate:protected_partitions`: Future and protected partitions stay sealed before outcome open.
- `learning_path:proposal`: Proposal memory updates only from exact feasible-action outcomes.
- `learning_path:selection`: Selection memory updates only from exact observed consequences.

## SCENARIO-LEARN-6418-GATES: Authenticated Preconditions Are Revalidated

**Given** Exp6417, Exp6413, Exp6407, and Exp6397 artifacts exist
**When** Exp6418 checks its preconditions
**Then** it SHALL revalidate their ready scores, model files, GPUs, schemas,
licenses, exact checkers, initial heads, rollback receipts, and protected
partitions before any session event is generated.

## SCENARIO-LEARN-6418-CHRONOLOGY: Four Sessions Are Sealed

**Given** authenticated upstream event receipts
**When** Exp6418 preregisters its chronological manifest
**Then** it SHALL seal at least 96 events across four sessions, three drift
regimes, six update opportunities, four restarts, two expiry boundaries, and
two supersession boundaries before generation opens.

## SCENARIO-LEARN-6418-CAUSAL-PATHS: Learned Paths Use Separate Exact Labels

**Given** raw events and proposals are frozen before outcomes
**When** exact labels open in causal order
**Then** proposal memory SHALL update only from exact feasibility outcomes
**And** selection memory SHALL update only from exact consequence outcomes.

## SCENARIO-LEARN-6418-MATCHED-ARMS: Work Surfaces Match

**Given** frozen, single-path exact transactional, and dual-path arms
**When** the three arms run
**Then** event order, model calls, prompts, tokens, checker calls, consumer
work, and initial heads SHALL match across arms.

## SCENARIO-LEARN-6418-ATTACKS: Dual-Path Attacks Fail Closed

**Given** contamination, stale head, duplicate effect, concurrent proposal,
interrupted write, expired license, superseded evidence, cache resurrection,
model swap, delayed outcome, or restart corruption attacks
**When** Exp6418 validates the attack matrix
**Then** no attack SHALL commit unsafe memory, override an exact veto, or
promote readiness.

## SCENARIO-LEARN-6418-READY: Readiness Requires Exact-Governed Future Gain

**Given** both learning paths receive causal exact outcomes and every attack
fails closed
**When** dual-path learning improves future exact yield over frozen with zero
contamination, no protected forgetting, bounded growth, and passing tests
**Then** `execution_grounded_dual_path_csl_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6418)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6418 | Implemented: `python/carnot/experiment_6418_execution_grounded_dual_path_csl.py`; terminal artifact `results/experiment_6418_execution_grounded_dual_path_csl.json`. | Implemented: `tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py`. |
| SCENARIO-LEARN-6418-GATES | Implemented: `python/carnot/experiment_6418_execution_grounded_dual_path_csl.py`. | Implemented: `tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py`. |
| SCENARIO-LEARN-6418-CHRONOLOGY | Implemented: `python/carnot/experiment_6418_execution_grounded_dual_path_csl.py`. | Implemented: `tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py`. |
| SCENARIO-LEARN-6418-CAUSAL-PATHS | Implemented: `python/carnot/experiment_6418_execution_grounded_dual_path_csl.py`. | Implemented: `tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py`. |
| SCENARIO-LEARN-6418-MATCHED-ARMS | Implemented: `python/carnot/experiment_6418_execution_grounded_dual_path_csl.py`. | Implemented: `tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py`. |
| SCENARIO-LEARN-6418-ATTACKS | Implemented: `python/carnot/experiment_6418_execution_grounded_dual_path_csl.py`. | Implemented: `tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py`. |
| SCENARIO-LEARN-6418-READY | Implemented: `python/carnot/experiment_6418_execution_grounded_dual_path_csl.py`. | Implemented: `tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py`. |

## REQ-LEARN-6419: Held-Shift Restart CSL Replication

**Given** Exp6418 first showed prospective execution-grounded improvement
**When** Exp6419 runs on planning date 20260814
**Then** it SHALL write
`results/experiment_6419_held_shift_restart_csl_replication.json`
**And** it SHALL freeze the Exp6418 learner before opening held outcomes.

Exp6419 SHALL revalidate Exp6418 gates, freeze mechanism, config, checker,
model, and prompt hashes, preflight authenticated GPU receipts, and prove the
held manifest was absent from Exp6418 mechanism selection.

Exp6419 SHALL use exactly these local GGUF model ids from `cached_sota_pair()`:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. Exp6419 SHALL not call `AutoTokenizer`.

Exp6419 SHALL build at least 72 held chronological events. The stream SHALL
span model-family, constraint-family, surface-form, and temporal shifts. It
SHALL include at least three restart boundaries, expiry and supersession
boundaries, and an untouched future partition.

Exp6419 SHALL run frozen, single-path, and frozen dual-path arms at matched
work. Every row SHALL bind to authenticated process and raw-output receipts.
Raw bytes SHALL be frozen before exact held outcomes open.

Exp6419 SHALL not tune triggers, learning rates, schemas, prompts, or gates
after held outcomes. Incompatible cells SHALL count as harm or abstention.

Exp6419 SHALL report proposal coverage, selection success, future exact yield,
retention, forgetting, contamination, growth, escalation, restart recovery,
latency, and GPU cost by arm, shift, model family, model, and session.

Exp6419 SHALL attack checkpoint substitution, partial restart, stale cache
resurrection, held-label access, model swap, prompt drift, license inheritance,
and silent fallback. Every attack SHALL fail closed.

Exp6419 SHALL emit `held_delta_future_exact_yield_over_frozen` as a bare
finite number. It SHALL set `held_shift_csl_replication_ready_score=1.0` only
when the frozen dual-path learner improves held future yield, has zero
surviving contamination, no protected retention regression, bounded growth,
successful restart recovery, and no post-outcome retuning.

Exp6419 SHALL emit these fields:

- `status`
- `exp6418_gate_receipts`
- `frozen_mechanism_config_checker_model_and_prompt_hashes`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `embedded_gguf_tokenizer_receipts`
- `autotokenizer_usage_count`
- `held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals`
- `held_manifest_absence_before_freeze_receipt`
- `authenticated_process_and_raw_output_receipts_by_model`
- `matched_arm_work_receipts`
- `no_post_outcome_retuning_receipts`
- `per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results`
- `held_delta_future_exact_yield_over_frozen`
- `held_contamination_propagation_rate`
- `held_forgetting_delta`
- `protected_leakage_count`
- `silent_fallback_count`
- `attack_matrix`
- `held_shift_csl_replication_ready_score`
- `public_factor_claim_eligibility`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field. It SHALL map both gates:
`gate:exp6418_prospective_improvement` and `gate:held_manifest_absence`. It
SHALL map each shift: `shift:model_family`, `shift:constraint_family`,
`shift:surface_form`, and `shift:temporal`. It SHALL map
`held_delta_future_exact_yield_over_frozen`,
`held_contamination_propagation_rate`, `held_forgetting_delta`, and
`held_shift_csl_replication_ready_score`.

`verifier_is_oracle` SHALL be true only for exact outcome and retention
checkers. Model output, proposal memory, and selection memory SHALL NOT be
oracles. `honest_verdict` SHALL start with a terminal success prefix.

Required field principles:

- `status`: Names the terminal state for the held-shift restart replication.
- `exp6418_gate_receipts`: Pins the upstream prospective improvement gate.
- `frozen_mechanism_config_checker_model_and_prompt_hashes`: Freezes the learner, checker, model, config, and prompt identity before held outcomes.
- `MODEL_SPECS`: Carries the three mandated GGUF model identities from cached SOTA receipts.
- `models_used`: Lists only the three mandated GGUF models.
- `cached_sota_pair_receipts`: Records cached SOTA helper evidence.
- `embedded_gguf_tokenizer_receipts`: Proves embedded GGUF tokenizer use.
- `autotokenizer_usage_count`: Must remain zero because external tokenizer paths are forbidden.
- `held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals`: Seals held events, shifts, restarts, expiry, supersession, and future rows.
- `held_manifest_absence_before_freeze_receipt`: Proves Exp6418 mechanism selection did not include the held manifest.
- `authenticated_process_and_raw_output_receipts_by_model`: Binds model processes and raw bytes before outcomes.
- `matched_arm_work_receipts`: Shows frozen, single-path, and frozen dual-path arms used equal work.
- `no_post_outcome_retuning_receipts`: Proves held outcomes did not change triggers, schemas, prompts, gates, or checkers.
- `per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results`: Reports held metrics without pooled masking.
- `held_delta_future_exact_yield_over_frozen`: Bare held future-yield lift for frozen dual path over frozen.
- `held_contamination_propagation_rate`: Must remain zero for readiness.
- `held_forgetting_delta`: Must show no protected forgetting.
- `protected_leakage_count`: Must be zero because protected partitions cannot route writes.
- `silent_fallback_count`: Must be zero because fallback would break model identity.
- `attack_matrix`: Shows every held restart and substitution attack fails closed.
- `held_shift_csl_replication_ready_score`: Conjunctive readiness score for the held-shift restart replication.
- `public_factor_claim_eligibility`: Limits public claims to this exact held replication.
- `harm_underpowered_missing_and_flagged_cells`: Keeps missing, underpowered, flagged, and harmful cells visible.
- `protected_files_unchanged`: Shows protected files stayed byte-identical.
- `preconditions_checked`: Lists every gate checked before readiness can become one.
- `inference_substrate`: Declares authenticated GGUF receipt replay on a sealed held stream.
- `verifier_is_oracle`: Marks only exact outcome and retention checkers as oracles.
- `field_principles`: Documents why each field exists.
- `field_provenance`: Maps each field to upstream receipts, manifest seals, attacks, tests, or code.
- `random_seed`: Pins held order, shifts, arms, attacks, and metrics.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records verification commands and exit codes.
- `reproducibility_checksum`: Content-addresses the artifact with volatile fields normalized.
- `honest_verdict`: Uses a terminal prefix and states the held replication result.
- `gate:exp6418_prospective_improvement`: Exp6418 must be ready before the held replication can run.
- `gate:held_manifest_absence`: The held manifest must be absent from Exp6418 mechanism selection.
- `shift:model_family`: Model-family shift metrics must stay visible.
- `shift:constraint_family`: Constraint-family shift metrics must stay visible.
- `shift:surface_form`: Surface-form shift metrics must stay visible.
- `shift:temporal`: Temporal shift metrics must stay visible.

## SCENARIO-LEARN-6419-FREEZE: Held Stream Is Sealed After Mechanism Freeze

**Given** Exp6418 has a ready artifact
**When** Exp6419 freezes its mechanism and held manifest
**Then** Exp6418 hashes SHALL be recorded before held outcomes open
**And** the held manifest absence receipt SHALL prove Exp6418 mechanism
selection did not include the held manifest.

## SCENARIO-LEARN-6419-SHIFTS: Held Events Cover Declared Shifts

**Given** authenticated held source rows
**When** Exp6419 builds the held stream
**Then** at least 72 chronological events SHALL cover model-family,
constraint-family, surface-form, temporal, restart, expiry, supersession, and
future-partition shifts.

## SCENARIO-LEARN-6419-MATCHED-ARMS: Frozen Arms Use Equal Work

**Given** frozen, single-path, and frozen dual-path arms
**When** Exp6419 evaluates the held stream
**Then** event order, model calls, prompt tokens, checker calls, raw-output
receipts, latency surfaces, and GPU cost surfaces SHALL match across arms.

## SCENARIO-LEARN-6419-NO-RETUNE: Held Outcomes Cannot Change the Mechanism

**Given** held outcomes have opened once
**When** Exp6419 records terminal receipts
**Then** trigger, learning-rate, schema, prompt, gate, and checker hashes SHALL
match their frozen hashes
**And** post-outcome retuning counts SHALL be zero.

## SCENARIO-LEARN-6419-ATTACKS: Restart and Substitution Attacks Fail Closed

**Given** checkpoint substitution, partial restart, stale cache resurrection,
held-label access, model swap, prompt drift, license inheritance, or silent
fallback attacks
**When** Exp6419 validates its attack matrix
**Then** no attack SHALL commit unsafe memory, leak labels, switch models,
inherit licenses, or promote readiness.

## SCENARIO-LEARN-6419-READY: Replication Requires Held Future Gain

**Given** matched held arms and no post-outcome retuning
**When** frozen dual-path learning improves future exact yield over frozen with
zero contamination, no protected forgetting, bounded growth, and successful
restart recovery
**Then** `held_shift_csl_replication_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6419)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6419 | Implemented: `python/carnot/experiment_6419_held_shift_restart_csl_replication.py`; terminal artifact `results/experiment_6419_held_shift_restart_csl_replication.json`. | Implemented: `tests/python/test_experiment_6419_held_shift_restart_csl_replication.py`. |
| SCENARIO-LEARN-6419-FREEZE | Implemented: `python/carnot/experiment_6419_held_shift_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6419_held_shift_restart_csl_replication.py`. |
| SCENARIO-LEARN-6419-SHIFTS | Implemented: `python/carnot/experiment_6419_held_shift_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6419_held_shift_restart_csl_replication.py`. |
| SCENARIO-LEARN-6419-MATCHED-ARMS | Implemented: `python/carnot/experiment_6419_held_shift_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6419_held_shift_restart_csl_replication.py`. |
| SCENARIO-LEARN-6419-NO-RETUNE | Implemented: `python/carnot/experiment_6419_held_shift_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6419_held_shift_restart_csl_replication.py`. |
| SCENARIO-LEARN-6419-ATTACKS | Implemented: `python/carnot/experiment_6419_held_shift_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6419_held_shift_restart_csl_replication.py`. |
| SCENARIO-LEARN-6419-READY | Implemented: `python/carnot/experiment_6419_held_shift_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6419_held_shift_restart_csl_replication.py`. |

## REQ-LEARN-6420: CSL Authenticity and Safety Audit

**Given** Exp6412 records the V551 claim-boundary audit, Exp6418 records the
development dual-path CSL stream, and Exp6419 records the held restart stream
**When** Exp6420 runs on planning date 20260814
**Then** it SHALL write
`results/experiment_6420_csl_authenticity_safety_audit.json`
**And** it SHALL not repair or rewrite upstream artifacts.

Exp6420 SHALL hash every available upstream artifact, sidecar, source,
checkpoint, model byte file, checker, and determination record that is expected
by the V552 CSL chain. It SHALL record each missing expected input as evidence.

Exp6420 SHALL reconstruct event order from monotonic order fields and immutable
hashes. It SHALL prove proposals predate exact outcomes, memory updates follow
exact feedback, and held future rows remain untouched before evaluation.

Exp6420 SHALL verify proposal-memory updates against exact feasibility evidence
and selection-memory updates against exact consequence evidence. It SHALL count
exact veto overrides, protected leakage, hidden retuning, stale cache evidence,
raw-output reuse, and metric recompute mismatches.

Exp6420 SHALL recompute development and held proposal coverage, selection
success, future yield, retention, forgetting, contamination, growth, restart
recovery, and costs from published rows. Reported metric deltas SHALL match the
row recomputation before readiness can become one.

Exp6420 SHALL attack forged PIDs, substituted model bytes, raw-output reuse,
event reordering, future-label leakage, same-step writes, model identity swaps,
stale heads, duplicates, partial commits, rollback omission, cache
resurrection, poisoned evidence, and hidden retuning. Each critical attack
SHALL fail closed before readiness can become one.

The attack ids are `forged_pid`, `substituted_model_bytes`,
`raw_output_reuse`, `event_reordering`, `future_label_leakage`,
`same_step_writes`, `model_identity_swap`, `stale_heads`, `duplicates`,
`partial_commits`, `rollback_omission`, `cache_resurrection`,
`poisoned_evidence`, and `hidden_retuning`.

Exp6420 SHALL preserve historical Exp6412 claim-boundary findings, current
adversarial verification findings, and determination-preservation findings as
separate evidence. It SHALL not clear additive corrigenda.

Exp6420 SHALL compare development and held effects with uncertainty and
effective sample size. Underpowered or heterogeneous cells SHALL remain visible
instead of being pooled away.

Exp6420 SHALL set `csl_authenticity_safety_audit_ready_score=1.0` only when
both streams exist, powered receipts are authentic, causal order holds,
recomputed metrics match reported metrics, no exact veto is overridden,
contamination is zero after rollback, protected retention is non-negative, and
every critical attack fails closed.

Exp6420 SHALL emit these fields:

- `status`
- `expected_and_available_upstream_inputs`
- `upstream_artifact_sidecar_source_checkpoint_model_and_checker_hashes`
- `missing_input_findings`
- `process_and_raw_output_authenticity_rechecks`
- `reconstructed_event_time_order`
- `proposal_precedes_outcome_checks`
- `update_follows_exact_feedback_checks`
- `untouched_future_partition_checks`
- `proposal_memory_exact_feasibility_bindings`
- `selection_memory_exact_consequence_bindings`
- `recomputed_development_and_held_metrics`
- `reported_vs_recomputed_deltas`
- `retention_forgetting_contamination_growth_restart_and_cost_rechecks`
- `uncertainty_and_effective_sample_sizes`
- `exact_veto_override_count`
- `protected_leakage_count`
- `hidden_retuning_count`
- `attack_matrix`
- `adversarial_and_determination_preservation_findings`
- `prospective_csl_claim_eligibility`
- `public_factor_claim_eligibility`
- `csl_authenticity_safety_audit_ready_score`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field. It SHALL map every
missing-input rule, every attack id, both eligibility fields, and
`csl_authenticity_safety_audit_ready_score`. `verifier_is_oracle` SHALL be
false for the audit as a whole. Exact upstream checkers remain semantic
oracles that this audit inspects.

## SCENARIO-LEARN-6420-MISSING: Missing Inputs Lower Eligibility

**Given** an expected V552 upstream artifact, sidecar, source, checker, model
file, or determination record is absent
**When** Exp6420 computes eligibility
**Then** the missing input SHALL be listed in `missing_input_findings`
**And** `csl_authenticity_safety_audit_ready_score` SHALL be `0.0`.

## SCENARIO-LEARN-6420-CAUSAL: Temporal Order Is Replayed From Rows

**Given** Exp6418 and Exp6419 event rows
**When** Exp6420 reconstructs event time
**Then** proposal freeze order SHALL precede exact outcome order
**And** update event indices SHALL have exact feedback before update binding.

## SCENARIO-LEARN-6420-METRICS: Reported Metrics Must Recompute

**Given** published raw outcome, manifest, transition, and receipt rows
**When** Exp6420 recomputes development and held effects
**Then** reported deltas SHALL be compared with recomputed deltas
**And** any mismatch SHALL lower claim eligibility.

## SCENARIO-LEARN-6420-ATTACKS: Critical Attacks Fail Closed

**Given** forged PID, model substitution, raw-output reuse, event reordering,
future leakage, same-step write, identity swap, stale head, duplicate, partial
commit, rollback, cache, poison, or hidden-retuning evidence
**When** Exp6420 builds the attack matrix
**Then** each critical attack SHALL fail closed before readiness can be one.

## SCENARIO-LEARN-6420-ORACLE: The Audit Is Not The Oracle

**Given** exact feasibility, consequence, outcome, release, and retention
checkers are upstream semantic oracles
**When** Exp6420 reports `verifier_is_oracle`
**Then** the audit SHALL set the value to `false`
**And** it SHALL identify those upstream exact checkers as audited oracles.

## Implementation Status (REQ-LEARN-6420)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6420 | Planned: `python/carnot/experiment_6420_csl_authenticity_safety_audit.py`; terminal artifact `results/experiment_6420_csl_authenticity_safety_audit.json`. | Planned: `tests/python/test_experiment_6420_csl_authenticity_safety_audit.py`. |
| SCENARIO-LEARN-6420-MISSING | Planned: `python/carnot/experiment_6420_csl_authenticity_safety_audit.py`. | Planned: `tests/python/test_experiment_6420_csl_authenticity_safety_audit.py`. |
| SCENARIO-LEARN-6420-CAUSAL | Planned: `python/carnot/experiment_6420_csl_authenticity_safety_audit.py`. | Planned: `tests/python/test_experiment_6420_csl_authenticity_safety_audit.py`. |
| SCENARIO-LEARN-6420-METRICS | Planned: `python/carnot/experiment_6420_csl_authenticity_safety_audit.py`. | Planned: `tests/python/test_experiment_6420_csl_authenticity_safety_audit.py`. |
| SCENARIO-LEARN-6420-ATTACKS | Planned: `python/carnot/experiment_6420_csl_authenticity_safety_audit.py`. | Planned: `tests/python/test_experiment_6420_csl_authenticity_safety_audit.py`. |
| SCENARIO-LEARN-6420-ORACLE | Planned: `python/carnot/experiment_6420_csl_authenticity_safety_audit.py`. | Planned: `tests/python/test_experiment_6420_csl_authenticity_safety_audit.py`. |

## REQ-LEARN-6430: Prospective Write-Once Memory Capacity Frontier

**Given** Exp6428 has a clean write-time admission result, Exp6426 supplies the
task-scoped runtime receipt contract, and Exp6420 found invalid V552 CSL
metrics, raw-output reuse, cache resurrection, and underpowered cells
**When** Exp6430 runs on planning date 20260814
**Then** it SHALL write
`results/experiment_6430_prospective_write_once_memory_capacity_frontier.json`
**And** it SHALL run a fresh prospective write-once factor-memory stream.

Exp6430 SHALL revalidate Exp6428 gates, Exp6426 runtime receipts, GPUs, VRAM,
model bytes, embedded GGUF tokenizers, runner identity, memory schemas, exact
checkers, licenses, protected partitions, disk, and initial heads. It SHALL
prove the new stream manifest and final artifact paths are absent before
generation.

Exp6430 SHALL use exactly these local GGUF model ids from `cached_sota_pair()`:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. Exp6430 SHALL not call `AutoTokenizer`.

Exp6430 SHALL preregister at least 120 unique chronological events across five
sessions, three drift regimes, three model families, real process restarts,
expiry boundaries, supersession boundaries, and an untouched future partition.
It SHALL freeze capacities 0, 4, 8, 16, and 32 before outcomes open.

Exp6430 SHALL generate one fresh raw output for every event through the
task-scoped receipt helper. It SHALL freeze event rows and proposals before
exact outcomes. One event may be replayed across matched arms, but one raw
output SHALL NOT represent two event ids.

Exp6430 SHALL compare frozen and capacity-limited exact-governed memories at
matched events, model calls, prompts, tokens, checker calls, consumer work, and
initial heads. It SHALL update memory only after exact feedback. Exact release
and protected-retention checks SHALL control every activation.

Exp6430 SHALL atomically Commit, Reject, Quarantine, Defer, Evict, Expire, or
Supersede only after exact support, protected retention, unique effect, license,
predecessor, and capacity checks. It SHALL record every head transition.

Exp6430 SHALL evaluate future rows once in causal order. It SHALL write
per-unit rows before deriving proposal coverage, write precision, selection
success, future exact yield, transfer, retention, forgetting, contamination,
memory growth, eviction, restart recovery, and cost.

Exp6430 SHALL estimate the capacity-utility frontier with counts, confidence
intervals, and effective sample sizes. It SHALL not choose capacity after held
outcomes are read.

Exp6430 SHALL attack raw-output reuse, cache resurrection, stale heads,
duplicate effects, concurrent writes, interrupted commits, expired licenses,
superseded evidence, model swaps, delayed outcomes, same-step writes, hidden
retuning, and future leakage. Every critical attack SHALL fail closed.

Exp6430 SHALL set `prospective_write_once_csl_ready_score=1.0` only when at
least one nonzero capacity improves row-recomputed future exact yield over
frozen, write precision and retention meet their frozen controls,
contamination and exact-veto overrides are zero, growth is bounded, all
critical attacks fail closed, and `current_adversarial_flag_count` is zero.

Exp6430 SHALL emit these fields:

- `status`
- `exp6428_gate_receipts`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `model_file_and_embedded_tokenizer_hashes`
- `autotokenizer_usage_count`
- `task_scoped_process_gpu_runner_and_raw_output_receipts`
- `manifest_absence_before_run_receipt`
- `chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals`
- `preregistered_capacity_and_arm_contract`
- `per_unit_rows`
- `per_event_unique_raw_output_and_pre_outcome_freeze_records`
- `exact_feedback_receipts`
- `memory_schema_head_and_transition_history`
- `commit_reject_quarantine_defer_evict_expire_and_supersede_counts`
- `per_capacity_coverage_precision_selection_future_yield_transfer_retention_forgetting_contamination_growth_eviction_restart_and_cost_results`
- `capacity_utility_frontier`
- `effective_sample_sizes_and_uncertainty`
- `best_capacity_selected_without_held_tuning`
- `aggregate_recomputation_receipts`
- `reported_vs_recomputed_deltas`
- `raw_output_reuse_count`
- `cache_resurrection_count`
- `same_step_write_count`
- `contamination_propagation_rate`
- `exact_veto_override_count`
- `protected_leakage_count`
- `attack_matrix`
- `prospective_write_once_csl_ready_score`
- `current_adversarial_flag_count`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `blocked_reason`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field. It SHALL map all gates,
capacities, causal writes, frontier metrics, attacks, `current_adversarial_flag_count`,
and `prospective_write_once_csl_ready_score`. `per_unit_rows` SHALL be present
because Exp6430 makes comparative claims. `verifier_is_oracle` SHALL be true
only for exact feedback, release, and protected-retention checks. Model output
and memory SHALL NOT be oracles. `honest_verdict` SHALL start with a terminal
success prefix.

Required field principles:

- `status`: Names the terminal state for the prospective write-once capacity frontier.
- `exp6428_gate_receipts`: Pins the clean write-time admission gate and the V552 null context.
- `MODEL_SPECS`: Carries the three mandated GGUF model identities from cached SOTA receipts.
- `models_used`: Lists only the three mandated GGUF models.
- `cached_sota_pair_receipts`: Records the helper calls that supplied all mandated model ids.
- `model_file_and_embedded_tokenizer_hashes`: Binds model bytes and embedded tokenizer metadata.
- `autotokenizer_usage_count`: Must remain zero because GGUF tokenizers are embedded.
- `task_scoped_process_gpu_runner_and_raw_output_receipts`: Binds fresh event generation to task-scoped process receipts.
- `manifest_absence_before_run_receipt`: Proves the new manifest and artifact paths did not exist before generation.
- `chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals`: Seals events, sessions, drift regimes, restarts, expiry, supersession, and future rows.
- `preregistered_capacity_and_arm_contract`: Freezes capacities, arms, work, prompts, tokens, checkers, and initial heads before outcomes.
- `per_unit_rows`: Records one comparative future row before aggregate calculation.
- `per_event_unique_raw_output_and_pre_outcome_freeze_records`: Proves each event id has one fresh raw output and frozen proposal before outcome release.
- `exact_feedback_receipts`: Records exact feedback, release, and protected-retention checks.
- `memory_schema_head_and_transition_history`: Records every schema, head, and transition.
- `commit_reject_quarantine_defer_evict_expire_and_supersede_counts`: Counts each atomic memory disposition.
- `per_capacity_coverage_precision_selection_future_yield_transfer_retention_forgetting_contamination_growth_eviction_restart_and_cost_results`: Reports separated capacity metrics without pooled masking.
- `capacity_utility_frontier`: Separates capacity, coverage, write precision, and future utility.
- `effective_sample_sizes_and_uncertainty`: Reports counts, confidence intervals, and effective sample sizes.
- `best_capacity_selected_without_held_tuning`: Proves capacity was selected from the preregistered frontier rule.
- `aggregate_recomputation_receipts`: Recomputes metrics from per-unit rows.
- `reported_vs_recomputed_deltas`: Shows reported aggregates match row recomputation.
- `raw_output_reuse_count`: Must be zero because one raw output cannot represent two event ids.
- `cache_resurrection_count`: Must be zero because stale caches cannot revive writes.
- `same_step_write_count`: Must be zero because writes follow exact outcomes.
- `contamination_propagation_rate`: Must be zero for readiness.
- `exact_veto_override_count`: Must be zero because exact rejections cannot be overridden.
- `protected_leakage_count`: Must be zero because protected and future rows cannot route writes.
- `attack_matrix`: Shows all critical attacks fail closed.
- `prospective_write_once_csl_ready_score`: Conjunctive readiness score for exact-governed capacity utility.
- `current_adversarial_flag_count`: Must be zero for readiness.
- `harm_underpowered_missing_and_flagged_cells`: Keeps V552 defects and any weak cells visible.
- `protected_files_unchanged`: Shows protected upstream and ops files stayed byte-identical.
- `blocked_reason`: Explains failed preconditions.
- `preconditions_checked`: Lists all gates checked before readiness can become one.
- `inference_substrate`: Declares task-scoped local GGUF receipt replay with exact-governed memory.
- `verifier_is_oracle`: Marks only exact feedback, release, and protected-retention checks as oracles.
- `field_principles`: Documents why each field exists.
- `field_provenance`: Maps each field to specs, inputs, stream rows, reductions, attacks, or tests.
- `random_seed`: Pins event generation, capacities, arms, attacks, and metrics.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records verification commands and exit codes.
- `reproducibility_checksum`: Content-addresses the payload with volatile fields normalized.
- `honest_verdict`: Uses a terminal success prefix and states the capacity-frontier result.
- `gate:exp6428_clean_write_time_admission`: Exp6428 must be complete, clean, and ready.
- `gate:exp6426_task_scoped_receipts`: Exp6426 runtime receipt contract must pass.
- `gate:exp6420_safety_null_context`: V552 safety defects must remain visible and not be reused as evidence.
- `gate:manifest_absence`: The Exp6430 manifest and artifact must be absent before generation.
- `gate:embedded_tokenizers`: All token counts must come from embedded GGUF tokenizers.
- `capacity:0`: Frozen memory is the no-write control.
- `capacity:4`: Capacity four tests severe memory pressure.
- `capacity:8`: Capacity eight tests moderate memory pressure.
- `capacity:16`: Capacity sixteen tests the middle frontier.
- `capacity:32`: Capacity thirty-two tests the high-capacity frontier.
- `write:Commit`: Commits require exact support, valid license, protected retention, unique effect, predecessor freshness, and capacity room.
- `write:Reject`: Rejects record exact, license, or predecessor failure.
- `write:Quarantine`: Quarantine contains malformed or unsafe evidence.
- `write:Defer`: Defers rows before exact support or under frozen authority.
- `write:Evict`: Eviction keeps capacity bounded after exact lower-priority selection.
- `write:Expire`: Expiry removes records after temporal or license validity ends.
- `write:Supersede`: Supersession replaces an older exact effect with newer exact support.
- `frontier:coverage`: Coverage measures proposal reach separately from precision.
- `frontier:precision`: Write precision measures accepted exact support.
- `frontier:future_yield`: Future exact yield measures held utility.
- `frontier:retention`: Protected retention guards prior exact behavior.

## SCENARIO-LEARN-6430-GATES: Receipts And Manifest Absence Gate The Run

**Given** Exp6428 and Exp6426 artifacts are available
**When** Exp6430 checks preconditions
**Then** their gates, model files, tokenizers, runner, protected partitions,
memory schemas, exact checkers, licenses, disk, initial heads, and path absence
SHALL pass before readiness can become one.

## SCENARIO-LEARN-6430-STREAM: Fresh Chronological Events Are Frozen

**Given** the preregistered stream
**When** Exp6430 generates events
**Then** at least 120 unique events SHALL cover five sessions, three drift
regimes, three model families, restarts, expiry, supersession, and an
untouched future partition
**And** every event SHALL have a unique raw-output hash and a pre-outcome
proposal freeze record.

## SCENARIO-LEARN-6430-CAPACITY: Exact Feedback Controls Memory Writes

**Given** capacities 0, 4, 8, 16, and 32
**When** Exp6430 processes chronological feedback
**Then** every Commit, Reject, Quarantine, Defer, Evict, Expire, or Supersede
transition SHALL follow exact support, release, retention, license,
predecessor, unique-effect, and capacity checks.

## SCENARIO-LEARN-6430-FRONTIER: Per-Unit Rows Precede The Frontier

**Given** all proposal and feedback rows are frozen
**When** Exp6430 evaluates future rows once in causal order
**Then** it SHALL write per-unit rows before deriving coverage, precision,
selection, future yield, transfer, retention, forgetting, contamination,
growth, eviction, restart, cost, and frontier metrics.

## SCENARIO-LEARN-6430-ATTACKS: Critical Attacks Fail Closed

**Given** reuse, cache, head, duplicate, concurrency, interruption, license,
supersession, model, delayed-outcome, same-step-write, hidden-retuning, and
future-leakage attacks
**When** Exp6430 validates the attack matrix
**Then** no attack SHALL commit unsafe memory, leak labels, switch models,
inherit licenses, revive stale cache, or promote readiness.

## SCENARIO-LEARN-6430-READY: Readiness Requires A Clean Nonzero Capacity Gain

**Given** matched work and row-recomputed aggregates
**When** at least one nonzero capacity improves future exact yield over
frozen, write precision and retention meet controls, contamination and exact
veto overrides are zero, growth is bounded, and all attacks fail closed
**Then** `prospective_write_once_csl_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6430)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6430 | Implemented: `python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py`; terminal artifact `results/experiment_6430_prospective_write_once_memory_capacity_frontier.json`. | Implemented: `tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py`. |
| SCENARIO-LEARN-6430-GATES | Implemented: `python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py`. | Implemented: `tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py`. |
| SCENARIO-LEARN-6430-STREAM | Implemented: `python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py`. | Implemented: `tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py`. |
| SCENARIO-LEARN-6430-CAPACITY | Implemented: `python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py`. | Implemented: `tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py`. |
| SCENARIO-LEARN-6430-FRONTIER | Implemented: `python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py`. | Implemented: `tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py`. |
| SCENARIO-LEARN-6430-ATTACKS | Implemented: `python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py`. | Implemented: `tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py`. |
| SCENARIO-LEARN-6430-READY | Implemented: `python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py`. | Implemented: `tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py`. |

## REQ-LEARN-6431: Controlled Memory Interference A/B

**Given** Exp6430 has sealed prospective write-once rows, Exp6420 keeps the V552
memory safety null context visible, and V553 cites arXiv:2608.07622
**When** Exp6431 runs on planning date 20260814
**Then** it SHALL write
`results/experiment_6431_controlled_memory_interference_ab.json`
**And** it SHALL compare capacity-matched baseline memory with
authority-aware retrieval and write controls.

Exp6431 SHALL revalidate Exp6430 gates, row hashes, manifest hashes, frozen
capacity contracts, memory policy hashes, exact support checkers, authority
schema, CPU, RAM, disk, and protected future seal before readiness can become
one. It SHALL not invoke a new LLM. It SHALL not tune on the held partition.

Exp6431 SHALL freeze the interference matrix and matched arms before downstream
outcomes are used. The matrix SHALL include benign accumulation, reinforcing
evidence, contradiction, source-authority conflict, supersession, temporal
invalidity, lexical collision, structural collision, poisoned evidence, and
target occlusion. Event order, evidence, capacity, query work, and initial head
SHALL match across arms.

Exp6431 SHALL apply each relationship through the local transactional memory
path. It SHALL use write, retrieval, expiry, supersession, rollback, and exact
retention receipts. It SHALL not fabricate success labels. It SHALL not bypass
exact support.

Exp6431 SHALL record target exposure, target retrieval, downstream use, proposal
coverage, write precision, plasticity, protected stability, contamination,
rollback, future exact yield, latency, and work for every unit and arm.

Exp6431 SHALL separate target-exposure failure from downstream-use failure. It
SHALL report results by relationship, capacity, model family, and factor family.
It SHALL flag empty or underpowered cells rather than pool them into stronger
cells.

Exp6431 SHALL require lower-authority contradiction, expired evidence,
superseded evidence, and poison to fail closed. Higher-authority valid updates
MAY replace older state only through an auditable transition.

Exp6431 SHALL attack authority spoofing, recency-only override, source pooling,
lexical collision, structural collision, target hiding, cache resurrection,
rollback omission, head substitution, and post-outcome relation labels. Every
critical invalid-memory attack SHALL fail closed.

Exp6431 SHALL recompute every aggregate from per-unit rows. It SHALL set
`memory_interference_safety_ready_score=1.0` only when every critical invalid
memory attack fails closed, protected stability does not regress,
contamination is zero after rollback, valid higher-authority plasticity remains
nonzero, and all aggregates recompute.

Exp6431 SHALL emit these fields:

- `status`
- `exp6430_gate_receipts`
- `upstream_row_manifest_policy_checker_and_head_hashes`
- `preregistered_interference_matrix`
- `preregistered_capacity_matched_arm_contract`
- `per_unit_rows`
- `per_relationship_capacity_model_and_family_exposure_retrieval_use_coverage_precision_plasticity_stability_contamination_rollback_yield_latency_and_work_results`
- `exposure_failure_count`
- `downstream_use_failure_count`
- `authority_spoof_accept_count`
- `expired_or_superseded_accept_count`
- `poisoned_evidence_accept_count`
- `valid_higher_authority_update_count`
- `protected_stability_delta`
- `contamination_after_rollback`
- `aggregate_recomputation_receipts`
- `reported_vs_recomputed_deltas`
- `attack_matrix`
- `memory_interference_safety_ready_score`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `blocked_reason`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map both gates, every relationship class, all safety
counters, and `memory_interference_safety_ready_score`. `per_unit_rows` SHALL
be present because Exp6431 makes comparative claims. `verifier_is_oracle` SHALL
be true only for exact support, authority, expiry, supersession, release, and
retention checks. Retrieval and memory scores SHALL NOT be oracles.
`honest_verdict` SHALL start with a terminal success prefix.

Required field principles:

- `status`: Names the terminal state for the controlled memory-interference A/B.
- `exp6430_gate_receipts`: Pins Exp6430 readiness, Exp6420 null context, and sealed-row eligibility.
- `upstream_row_manifest_policy_checker_and_head_hashes`: Binds rows, manifests, memory policy, exact checkers, heads, and machine resources.
- `preregistered_interference_matrix`: Freezes relationship labels before downstream outcomes can steer them.
- `preregistered_capacity_matched_arm_contract`: Proves baseline and authority-aware arms share capacity, work, evidence, order, and initial heads.
- `per_unit_rows`: Records one arm-level unit before aggregate calculation.
- `per_relationship_capacity_model_and_family_exposure_retrieval_use_coverage_precision_plasticity_stability_contamination_rollback_yield_latency_and_work_results`: Reports separated cells without masking weak strata.
- `exposure_failure_count`: Counts target records blocked before retrieval.
- `downstream_use_failure_count`: Counts exposed records that were not used correctly downstream.
- `authority_spoof_accept_count`: Must be zero because spoofed authority cannot release memory.
- `expired_or_superseded_accept_count`: Must be zero because invalid temporal state cannot release memory.
- `poisoned_evidence_accept_count`: Must be zero because poison cannot release memory.
- `valid_higher_authority_update_count`: Must stay nonzero to show safe plasticity.
- `protected_stability_delta`: Must not regress protected behavior.
- `contamination_after_rollback`: Must be zero after rollback.
- `aggregate_recomputation_receipts`: Recomputes metrics from per-unit rows.
- `reported_vs_recomputed_deltas`: Shows reported aggregates match row recomputation.
- `attack_matrix`: Shows every critical invalid-memory attack fails closed.
- `memory_interference_safety_ready_score`: Conjunctive readiness for authority-aware memory interference safety.
- `harm_underpowered_missing_and_flagged_cells`: Keeps weak, missing, null, and flagged cells visible.
- `protected_files_unchanged`: Shows protected upstream and ops files stayed byte-identical.
- `blocked_reason`: Explains failed preconditions.
- `preconditions_checked`: Lists gates, hashes, schemas, resources, and seals checked before readiness.
- `inference_substrate`: Declares deterministic replay over sealed Exp6430 rows with no new LLM.
- `verifier_is_oracle`: Marks only exact support and authority lifecycle checks as oracles.
- `field_principles`: Documents why each field exists.
- `field_provenance`: Maps each field to specs, inputs, rows, reductions, attacks, or tests.
- `random_seed`: Pins relation assignment, arms, attacks, and reductions.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records verification commands and exit codes.
- `reproducibility_checksum`: Content-addresses the artifact with volatile fields normalized.
- `honest_verdict`: Uses a terminal success prefix and states the interference result.
- `gate:exp6430_complete_ready`: Exp6430 must be complete, ready, and row-sealed.
- `gate:exp6420_safety_null_context`: Exp6420 null findings must remain visible.
- `relationship:benign_accumulation`: Valid same-authority evidence should remain usable.
- `relationship:reinforcing_evidence`: Valid reinforcing evidence should improve coverage without instability.
- `relationship:contradiction`: Lower-authority contradiction must fail closed.
- `relationship:source_authority_conflict`: Lower-authority source conflict must fail closed.
- `relationship:supersession`: Valid higher-authority updates may replace older state through audit.
- `relationship:temporal_invalidity`: Expired evidence must fail closed.
- `relationship:lexical_collision`: Similar words must not hide the target record.
- `relationship:structural_collision`: Similar structure must not hide the target record.
- `relationship:poisoned_evidence`: Poisoned evidence must fail closed.
- `relationship:target_occlusion`: Target exposure must be measured apart from downstream use.
- `attack:authority_spoofing`: Spoofed source rank must not release memory.
- `attack:recency_only_override`: Recency alone must not override authority.
- `attack:source_pooling`: Pooled sources must not launder lower authority.
- `attack:lexical_collision`: Lexical collision must not select the wrong record.
- `attack:structural_collision`: Structural collision must not select the wrong record.
- `attack:target_hiding`: Hidden targets must stay visible to the exposure metric.
- `attack:cache_resurrection`: Stale cache must not revive invalid memory.
- `attack:rollback_omission`: Rollback omission must not leave contamination.
- `attack:head_substitution`: Head swaps must not alter the matched initial state.
- `attack:post_outcome_relation_labels`: Relation labels must not be assigned after outcome use.

## SCENARIO-LEARN-6431-GATES: Sealed Inputs Gate The Run

**Given** Exp6430 and Exp6420 artifacts are available
**When** Exp6431 checks preconditions
**Then** gates, row hashes, manifest hashes, memory policy, exact checkers,
authority schema, CPU, RAM, disk, and protected future seal SHALL pass before
readiness can become one.

## SCENARIO-LEARN-6431-FREEZE: Matrix And Arms Freeze First

**Given** the sealed Exp6430 stream
**When** Exp6431 builds the interference matrix
**Then** relationship labels, capacities, arms, evidence hashes, query work, and
initial heads SHALL be frozen before downstream outcomes can steer them.

## SCENARIO-LEARN-6431-PATHS: Lifecycle Controls Use Real Memory Paths

**Given** each relationship class
**When** Exp6431 replays the matched arms
**Then** write, retrieval, expiry, supersession, rollback, and exact-retention
receipts SHALL decide exposure, retrieval, use, and contamination.

## SCENARIO-LEARN-6431-METRICS: Exposure And Use Are Separate

**Given** per-unit rows exist for every arm
**When** Exp6431 reduces results
**Then** target exposure failure and downstream-use failure SHALL be counted
separately by relationship, capacity, model family, and factor family.

## SCENARIO-LEARN-6431-ATTACKS: Critical Invalid Memory Fails Closed

**Given** authority spoofing, recency override, source pooling, collisions,
target hiding, cache resurrection, rollback omission, head substitution, and
post-outcome labels
**When** Exp6431 builds the attack matrix
**Then** every critical attack SHALL fail closed in the authority-aware arm.

## SCENARIO-LEARN-6431-READY: Readiness Is Fully Conjunctive

**Given** row-recomputed aggregates and attack receipts
**When** attacks fail closed, protected stability does not regress,
contamination is zero after rollback, valid higher-authority plasticity is
nonzero, and reported metrics recompute
**Then** `memory_interference_safety_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6431)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6431 | Implemented: `python/carnot/experiment_6431_controlled_memory_interference_ab.py`; terminal artifact `results/experiment_6431_controlled_memory_interference_ab.json`. | Implemented: `tests/python/test_experiment_6431_controlled_memory_interference_ab.py`. |
| SCENARIO-LEARN-6431-GATES | Implemented: `python/carnot/experiment_6431_controlled_memory_interference_ab.py`. | Implemented: `tests/python/test_experiment_6431_controlled_memory_interference_ab.py`. |
| SCENARIO-LEARN-6431-FREEZE | Implemented: `python/carnot/experiment_6431_controlled_memory_interference_ab.py`. | Implemented: `tests/python/test_experiment_6431_controlled_memory_interference_ab.py`. |
| SCENARIO-LEARN-6431-PATHS | Implemented: `python/carnot/experiment_6431_controlled_memory_interference_ab.py`. | Implemented: `tests/python/test_experiment_6431_controlled_memory_interference_ab.py`. |
| SCENARIO-LEARN-6431-METRICS | Implemented: `python/carnot/experiment_6431_controlled_memory_interference_ab.py`. | Implemented: `tests/python/test_experiment_6431_controlled_memory_interference_ab.py`. |
| SCENARIO-LEARN-6431-ATTACKS | Implemented: `python/carnot/experiment_6431_controlled_memory_interference_ab.py`. | Implemented: `tests/python/test_experiment_6431_controlled_memory_interference_ab.py`. |
| SCENARIO-LEARN-6431-READY | Implemented: `python/carnot/experiment_6431_controlled_memory_interference_ab.py`. | Implemented: `tests/python/test_experiment_6431_controlled_memory_interference_ab.py`. |

## REQ-LEARN-6432: Held-Shift Process-Restart CSL Replication

**Given** Exp6430 selected a frozen write-once memory capacity policy, Exp6431
passed the controlled interference gate, and Exp6420 keeps the Exp6419 failure
mode visible
**When** Exp6432 runs on planning date 20260814
**Then** it SHALL write
`results/experiment_6432_held_shift_process_restart_csl_replication.json`
**And** it SHALL replicate the frozen Exp6430 policy on a fresh held
factor-family shift after real process restarts.

Exp6432 SHALL revalidate Exp6430 and Exp6431 gates before held generation. It
SHALL check GPUs, VRAM, model bytes, embedded GGUF tokenizers, the local
runner, task-scoped receipt helpers, the frozen memory policy, selected
capacity, exact checkers, licenses, disk, and protected development rows. It
SHALL prove the held manifest path and raw-output paths were absent before new
bytes are generated.

Exp6432 SHALL use the three mandated local GGUF model ids returned by
`cached_sota_pair()`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. Exp6432 SHALL not use a transformers tokenizer path.

Exp6432 SHALL preregister the held factor-family shift, model balance,
sessions, seeds, prompts, budgets, process restart points, expiry boundaries,
supersession boundaries, and untouched evaluation partition. It SHALL not tune
the memory policy after held exposure.

Exp6432 SHALL start from the persisted Exp6430 selected-capacity memory head.
It SHALL recover that head in a new process from disk before each held session.
It SHALL prove the recovered head hash matches the sealed Exp6430 head. It
SHALL preserve exact authority. It SHALL not keep any parent in-memory state
except the hash of the persisted schema.

Exp6432 SHALL generate one new raw output for each held event id. It SHALL bind
each raw output to the event id, prompt hash, model id, model bytes, embedded
tokenizer hash, task receipt, and child process id. It SHALL freeze proposals
before exact outcomes become visible.

Exp6432 SHALL compare frozen memory and selected-capacity memory at matched
work. It SHALL write one per-unit row before it computes coverage, precision,
selection, future exact yield, transfer, retention, forgetting, negative
transfer, contamination, restart recovery, latency, and GPU cost.

Exp6432 SHALL attack raw-output reuse, cache resurrection, stale or substituted
heads, model swaps, hidden retuning, future leakage, same-step writes, expired
licenses, superseded evidence, interrupted persistence, rollback omission, and
protected leakage. Every attack SHALL fail closed.

Exp6432 SHALL recompute every aggregate from held per-unit rows. It SHALL
report confidence intervals, effective sample sizes, underpowered strata, and
exact null cells without development pooling.

Exp6432 SHALL set `held_shift_restart_csl_ready_score=1.0` only when the frozen
selected-capacity policy improves row-recomputed held future exact yield over
frozen, protected retention does not regress, contamination and negative
transfer stay within preregistered bounds, restarts recover exactly, every
attack fails closed, and `current_adversarial_flag_count=0`.

Exp6432 SHALL emit these fields:

- `status`
- `exp6430_and_exp6431_gate_receipts`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `model_file_and_embedded_tokenizer_hashes`
- `autotokenizer_usage_count`
- `held_manifest_and_raw_output_path_absence_receipts`
- `held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals`
- `frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes`
- `task_scoped_process_gpu_runner_and_raw_output_receipts`
- `per_unit_rows`
- `per_event_unique_raw_output_and_pre_outcome_freeze_records`
- `process_restart_and_persisted_head_recovery_receipts`
- `per_arm_model_family_session_coverage_precision_selection_future_yield_transfer_retention_forgetting_negative_transfer_contamination_restart_latency_and_gpu_cost_results`
- `held_future_exact_yield_delta`
- `protected_retention_delta`
- `negative_transfer_delta`
- `contamination_propagation_rate`
- `effective_sample_sizes_and_uncertainty`
- `aggregate_recomputation_receipts`
- `reported_vs_recomputed_deltas`
- `raw_output_reuse_count`
- `cache_resurrection_count`
- `hidden_retuning_count`
- `protected_leakage_count`
- `attack_matrix`
- `held_shift_restart_csl_ready_score`
- `current_adversarial_flag_count`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `blocked_reason`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map both upstream gates, held freshness, all deltas,
all attacks, the flag count, and the readiness score. `per_unit_rows` SHALL be
present because Exp6432 makes comparative claims. `verifier_is_oracle` SHALL be
true only for exact feedback, persistence integrity, release, and
protected-retention checks. Model output and memory SHALL NOT be oracles.
`honest_verdict` SHALL start with a terminal success prefix.

Required field principles:

- `status`: Names the terminal state for the held-shift process-restart replication.
- `exp6430_and_exp6431_gate_receipts`: Pins the clean stream gate, the interference safety gate, and the Exp6420 failure context.
- `MODEL_SPECS`: Carries the three mandated GGUF model identities from cached SOTA receipts.
- `models_used`: Lists only the three mandated GGUF models used for held rows.
- `cached_sota_pair_receipts`: Records the helper calls that supplied all mandated model ids.
- `model_file_and_embedded_tokenizer_hashes`: Binds model bytes, bytes-in-use counts, and embedded tokenizer metadata.
- `autotokenizer_usage_count`: Must remain zero because GGUF tokenizer metadata is embedded.
- `held_manifest_and_raw_output_path_absence_receipts`: Proves held manifest, artifact, and raw-output paths were absent before generation.
- `held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals`: Seals held event order, balance, shift, restarts, expiry, supersession, and untouched evaluation rows.
- `frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes`: Freezes the Exp6430 policy, selected capacity, exact checkers, model bytes, prompts, and persisted head before held outcomes.
- `task_scoped_process_gpu_runner_and_raw_output_receipts`: Binds fresh held generation to task-scoped process, GPU, runner, and raw-output receipts.
- `per_unit_rows`: Records one matched frozen or selected-capacity row before aggregate calculation.
- `per_event_unique_raw_output_and_pre_outcome_freeze_records`: Proves each held event has one raw output and a proposal frozen before outcome release.
- `process_restart_and_persisted_head_recovery_receipts`: Proves each held session recovered the persisted Exp6430 head from disk in a new process.
- `per_arm_model_family_session_coverage_precision_selection_future_yield_transfer_retention_forgetting_negative_transfer_contamination_restart_latency_and_gpu_cost_results`: Reports separated arm, model-family, and session cells without development pooling.
- `held_future_exact_yield_delta`: Must be positive for readiness.
- `protected_retention_delta`: Must be nonnegative for readiness.
- `negative_transfer_delta`: Must stay at or below the preregistered harm bound.
- `contamination_propagation_rate`: Must be zero for readiness.
- `effective_sample_sizes_and_uncertainty`: Reports counts, confidence intervals, nulls, and underpowered strata.
- `aggregate_recomputation_receipts`: Recomputes metrics from per-unit rows.
- `reported_vs_recomputed_deltas`: Shows reported aggregates match row recomputation.
- `raw_output_reuse_count`: Must be zero because one raw output cannot represent two held event ids.
- `cache_resurrection_count`: Must be zero because stale caches cannot revive memory.
- `hidden_retuning_count`: Must be zero because the policy is frozen before held exposure.
- `protected_leakage_count`: Must be zero because protected and future labels cannot route writes.
- `attack_matrix`: Shows all critical attacks fail closed.
- `held_shift_restart_csl_ready_score`: Conjunctive readiness score for held-shift restart replication.
- `current_adversarial_flag_count`: Must be zero for readiness.
- `harm_underpowered_missing_and_flagged_cells`: Keeps weak, missing, null, and flagged cells visible.
- `protected_files_unchanged`: Shows protected upstream and ops files stayed byte-identical.
- `blocked_reason`: Explains failed preconditions.
- `preconditions_checked`: Lists gates, GPUs, VRAM, model bytes, tokenizers, runner, helpers, policy, checkers, licenses, disk, path absence, and protected rows.
- `inference_substrate`: Declares task-scoped local GGUF held generation with exact-governed persisted memory.
- `verifier_is_oracle`: Marks only exact feedback, persistence integrity, release, and protected-retention checks as oracles.
- `field_principles`: Documents why each artifact field exists.
- `field_provenance`: Maps each field to sources, rows, reductions, checks, attacks, or tests.
- `random_seed`: Pins held events, sessions, prompts, restarts, attacks, and reductions.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records verification commands and exit codes.
- `reproducibility_checksum`: Content-addresses the artifact with volatile fields normalized.
- `honest_verdict`: Uses a terminal success prefix and states the held-shift result.
- `gate:exp6430_clean_stream`: Exp6430 must be complete, ready, row-recomputed, and cache-clean.
- `gate:exp6431_interference_safety`: Exp6431 must be complete, ready, and contamination-clean.
- `gate:exp6420_failure_context`: Exp6420 must keep raw-output reuse and cache resurrection defects visible.
- `held:fresh_manifest`: Held manifest and raw-output paths must be absent before generation.
- `held:new_prompts`: Held prompts must be new and bound to the planning date.
- `held:unique_raw_outputs`: Held raw-output hashes must be unique and absent from Exp6430 raw hashes.
- `delta:future_exact_yield`: Selected-capacity future exact yield must exceed frozen.
- `delta:protected_retention`: Protected retention must not regress.
- `delta:negative_transfer`: Negative transfer must stay within the preregistered bound.
- `delta:contamination`: Contamination propagation must remain zero.
- `attack:raw_output_reuse`: Raw-output reuse must not release or promote memory.
- `attack:cache_resurrection`: Stale cache state must not revive writes.
- `attack:stale_or_substituted_heads`: Head substitution must fail persisted-head verification.
- `attack:model_swaps`: Model ids and bytes must match sealed receipts.
- `attack:hidden_retuning`: Held outcomes must not change the capacity or policy.
- `attack:future_leakage`: Future labels must not affect proposals or writes.
- `attack:same_step_writes`: Writes must not occur in the same step as proposal generation.
- `attack:expired_licenses`: Expired licenses must fail release.
- `attack:superseded_evidence`: Superseded evidence must fail unless exact newer support exists.
- `attack:interrupted_persistence`: Interrupted persistence must not promote a new head.
- `attack:rollback_omission`: Rollback omission must not leave contamination.
- `attack:protected_leakage`: Protected rows must not leak into held selection.

## SCENARIO-LEARN-6432-GATES: Clean Stream And Safety Gates Hold

**Given** Exp6430, Exp6431, Exp6426, and Exp6420 artifacts are available
**When** Exp6432 checks preconditions
**Then** both readiness gates, model bytes, embedded tokenizers, runner,
helpers, policy, selected capacity, exact checkers, resources, path absence,
and protected rows SHALL pass before readiness can become one.

## SCENARIO-LEARN-6432-PREREGISTRATION: Held Plan Freezes First

**Given** a new held factor-family shift
**When** Exp6432 builds the held manifest
**Then** model balance, sessions, seeds, prompts, budgets, restart points,
expiry, supersession, and partitions SHALL be frozen before exact outcomes
can steer the policy.

## SCENARIO-LEARN-6432-RESTARTS: Persisted Head Recovers In New Processes

**Given** the sealed Exp6430 selected-capacity head
**When** Exp6432 starts each held session
**Then** a new process SHALL recover the same head hash from disk and no
unhashed parent memory state SHALL be accepted.

## SCENARIO-LEARN-6432-ROWS: Matched Per-Unit Rows Precede Aggregates

**Given** fresh held raw outputs and frozen proposals
**When** Exp6432 evaluates frozen and selected-capacity arms
**Then** it SHALL write matched per-unit rows before it reduces future exact
yield, transfer, retention, forgetting, negative transfer, contamination,
restart, latency, and GPU cost metrics.

## SCENARIO-LEARN-6432-ATTACKS: Critical Held Attacks Fail Closed

**Given** raw-output reuse, cache resurrection, head substitution, model swap,
hidden retuning, future leakage, same-step write, expired license, superseded
evidence, interrupted persistence, rollback omission, and protected leakage
attacks
**When** Exp6432 validates the attack matrix
**Then** every attack SHALL fail closed and no attack SHALL promote readiness.

## SCENARIO-LEARN-6432-READY: Readiness Requires Positive Held Yield

**Given** row-recomputed held aggregates
**When** selected-capacity future exact yield beats frozen, retention does not
regress, negative transfer and contamination stay within bounds, restarts
recover exactly, all attacks fail closed, and no adversarial flag remains
**Then** `held_shift_restart_csl_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6432)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6432 | Implemented: `python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py`; terminal artifact `results/experiment_6432_held_shift_process_restart_csl_replication.json`. | Implemented: `tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py`. |
| SCENARIO-LEARN-6432-GATES | Implemented: `python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py`. |
| SCENARIO-LEARN-6432-PREREGISTRATION | Implemented: `python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py`. |
| SCENARIO-LEARN-6432-RESTARTS | Implemented: `python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py`. |
| SCENARIO-LEARN-6432-ROWS | Implemented: `python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py`. |
| SCENARIO-LEARN-6432-ATTACKS | Implemented: `python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py`. |
| SCENARIO-LEARN-6432-READY | Implemented: `python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py`. | Implemented: `tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py`. |

## REQ-LEARN-6433: CSL Row Recomputation Safety Audit

**Given** Exp6420 nullified the V552 CSL claim and Exp6430, Exp6431, and
Exp6432 report new positive V553 summaries
**When** Exp6433 runs on planning date 20260814
**Then** it SHALL write
`results/experiment_6433_csl_row_recomputation_safety_audit.json`
**And** it SHALL recompute V553 metrics from immutable row evidence without
importing Exp6430, Exp6431, or Exp6432 aggregate or readiness functions.

Exp6433 SHALL hash every expected upstream artifact, embedded row block,
manifest, raw output, source, test, checker, receipt helper, memory head, and
determination record. It SHALL record CPU, RAM, disk, missing inputs, and the
current repository state.

Exp6433 SHALL classify every expected input as present, missing, skipped,
blocked, null, flagged, underpowered, or eligible. Missing rows SHALL remain
visible. Missing rows SHALL never become zeros and SHALL never be dropped from
denominators.

Exp6433 SHALL recheck manifest pre-absence receipts, unique event ids, unique
raw-output hashes across event ids, prompt and model bindings, proposal-before-
outcome order, exact feedback order, capacity enforcement, memory authority,
head transitions, held-policy freeze, and restart persistence.

Exp6433 SHALL derive development capacity, interference, and held metrics from
per-unit rows and immutable sidecar rows. It SHALL compute counts, rates,
deltas, confidence intervals, capacity frontier points, interference results,
held effects, retention, forgetting, contamination, restart recovery, growth,
and costs independently of upstream aggregate code.

Exp6433 SHALL compare every reported value it audits with its independent
value. Each comparison SHALL record absolute delta, tolerance, row population,
filter, numerator, denominator, and mismatch reason.

Exp6433 SHALL replay raw-output reuse, cache resurrection, row deletion,
duplicate event, event reorder, same-step write, stale head, authority spoof,
supersession bypass, hidden retuning, future leakage, restart corruption,
rollback omission, and exact-veto override attacks. Critical attacks SHALL fail
closed before claim eligibility can become true.

Exp6433 SHALL preserve stamped adversarial findings separately from current
adversarial findings. It SHALL also record determination-preservation and
artifact-convention audit findings.

Exp6433 SHALL set `prospective_csl_claim_eligibility=true` only when all
required rows exist, headline values recompute within frozen tolerance,
development and held future effects are positive with adequate effective sample
size, protected retention holds, contamination is zero, every critical attack
fails closed, and no current flag remains. Otherwise it SHALL use a
`complete_null` or `complete_blocked` verdict with a `blocked_reason`.

Exp6433 SHALL emit these fields:

- `status`
- `expected_and_available_upstream_inputs`
- `upstream_artifact_row_manifest_raw_source_test_checker_receipt_head_and_determination_hashes`
- `missing_input_findings`
- `upstream_state_by_task`
- `per_unit_rows`
- `event_and_raw_output_uniqueness_rechecks`
- `causal_order_and_exact_feedback_rechecks`
- `capacity_and_head_transition_rechecks`
- `held_freeze_and_restart_rechecks`
- `independently_recomputed_development_capacity_interference_and_held_metrics`
- `reported_vs_recomputed_deltas`
- `mismatch_count`
- `effective_sample_sizes_and_uncertainty_rechecks`
- `retention_forgetting_contamination_growth_restart_and_cost_rechecks`
- `attack_matrix`
- `open_critical_attack_ids`
- `current_and_stamped_adversarial_findings`
- `determination_preservation_findings`
- `artifact_convention_findings`
- `public_factor_claim_eligibility`
- `prospective_csl_claim_eligibility`
- `csl_row_recomputation_audit_ready_score`
- `same_verdict_retirement_decision`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `blocked_reason`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field. It SHALL map every
missing-input rule, recomputation family, attack id, eligibility decision, and
retirement decision. `per_unit_rows` SHALL contain one audit row for every
source unit and every reported-vs-recomputed comparison. `verifier_is_oracle`
SHALL be false for the audit as a whole. Exact validators remain semantic
oracles being audited.

## SCENARIO-LEARN-6433-HASHES: Expected Inputs Are Classified

**Given** expected V553 artifacts, rows, sidecars, raw outputs, sources, tests,
checkers, receipt helpers, heads, and determination records
**When** Exp6433 builds the input ledger
**Then** each input SHALL carry a hash or missing receipt
**And** each input SHALL receive a visible state classification.

## SCENARIO-LEARN-6433-ROWS: Metrics Recompute From Rows

**Given** Exp6430, Exp6431, and Exp6432 per-unit rows
**When** Exp6433 reduces the row sets
**Then** development capacity, interference, and held metrics SHALL be derived
without importing upstream aggregate or readiness functions.

## SCENARIO-LEARN-6433-DELTAS: Reported Values Are Compared

**Given** reported headline values and independent row reductions
**When** Exp6433 compares values
**Then** each comparison SHALL record absolute delta, tolerance, row
population, filter, numerator, denominator, and mismatch reason.

## SCENARIO-LEARN-6433-ATTACKS: Critical Attacks Fail Closed

**Given** reuse, cache, deletion, duplicate, reorder, same-step, stale-head,
authority, supersession, retuning, leakage, restart, rollback, and veto attacks
**When** Exp6433 builds the attack matrix
**Then** open critical attacks SHALL force claim eligibility false.

## SCENARIO-LEARN-6433-ELIGIBILITY: Current Flags Block Claims

**Given** row recomputation matches but a current critical adversarial flag or
stamped determination remains
**When** Exp6433 computes claim eligibility
**Then** `prospective_csl_claim_eligibility` SHALL be false
**And** `honest_verdict` SHALL start with `complete_null:`.

## Implementation Status (REQ-LEARN-6433)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6433 | Implemented: `python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py`; terminal artifact `results/experiment_6433_csl_row_recomputation_safety_audit.json`. | Implemented: `tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py`. |
| SCENARIO-LEARN-6433-HASHES | Implemented: `python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py`. | Implemented: `tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py`. |
| SCENARIO-LEARN-6433-ROWS | Implemented: `python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py`. | Implemented: `tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py`. |
| SCENARIO-LEARN-6433-DELTAS | Implemented: `python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py`. | Implemented: `tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py`. |
| SCENARIO-LEARN-6433-ATTACKS | Implemented: `python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py`. | Implemented: `tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py`. |
| SCENARIO-LEARN-6433-ELIGIBILITY | Implemented: `python/carnot/experiment_6433_csl_row_recomputation_safety_audit.py`. | Implemented: `tests/python/test_experiment_6433_csl_row_recomputation_safety_audit.py`. |

## REQ-LEARN-6455: Prospective Verifier-Bounded Factor-Weight CSL

**Given** FR-11 requires closed learning with immutable validation, rollback,
and bounded forgetting
**When** Exp6455 runs on planning date 20260815
**Then** it SHALL write
`results/experiment_6455_prospective_verifier_bounded_factor_weight_csl.json`
**And** it SHALL compare frozen factor weights, self-teacher-signed updates,
and verifier-bounded updates on one fresh chronological stream per mandated
GGUF model.

Exp6455 SHALL require both RTX 3090 GPUs, all three mandated GGUF files,
embedded GGUF tokenizer metadata, exact local policy checkers, a monotonic
clock, atomic event storage, fresh raw-output and ledger paths, enough disk,
and a sealed stream, arm, and analysis manifest before readiness can become
one.

Exp6455 SHALL define `MODEL_SPECS` through `cached_sota_pair()` or the same
local resolver. It SHALL include exactly `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF`. It
SHALL use embedded GGUF tokenizers only. It SHALL not call `AutoTokenizer`.

Exp6455 SHALL seal at least 24 chronological units per model before update
state changes. Each unit SHALL include new facts, bindings, candidate actions,
protected cases, and a deterministic exact checker. Frozen, self-teacher, and
verifier-bounded arms SHALL select from the same candidate bytes for a given
model and unit.

Exp6455 SHALL maintain three independent state ledgers. The verifier-bounded arm SHALL derive update sign only from the exact checker result. Model evidence
may supply only a bounded nonnegative update magnitude. Every update SHALL be
clamped, logged, and applied only after selection and exact checking, so it can
affect later units only.

Exp6455 SHALL emit per-unit rows for every chronological unit, model, and arm.
Rows SHALL include chronological index, model, arm, candidate hashes, selected
candidate, pre-update weights, exact result, teacher signal, exact sign,
magnitude, post-update weights, head hashes, future exact outcome, protected
outcome, checker work, and real timing.

Exp6455 SHALL recompute future exact yield, online learning curves, negative
transfer, forgetting, protected retention, false accepts, abstentions, weight
growth, update sparsity, and cost from rows. Uncertainty SHALL use distinct
future units, not duplicate arm rows.

Exp6455 SHALL attack future-label leakage, same-unit update use, teacher sign
override, exact-result transport corruption, unbounded weights, state sharing
across arms, output reuse, fake model receipts, CPU fallback, timing synthesis,
and aggregate-row mismatch. Every critical attack SHALL fail closed.

Exp6455 SHALL set `verifier_bounded_csl_ready_score=1.0` only when the
verifier-bounded arm improves future exact yield over frozen weights on
distinct later units, outperforms or is safer than teacher-signed updates, has
no protected-retention or false-accept regression, respects chronology, keeps
weight growth bounded, has eligible rows for all three models, passes duration
checks, and has zero critical findings.

Exp6455 SHALL emit these fields:

- `status`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `model_and_embedded_tokenizer_hashes`
- `autotokenizer_usage_count`
- `device_and_runner_receipts`
- `sealed_stream_arm_and_analysis_manifest`
- `path_nonexistence_and_freshness_receipts`
- `exact_checker_and_update_rule_hashes`
- `event_store_and_initial_head_hashes`
- `per_unit_rows`
- `chronology_and_future_only_checks`
- `frozen_teacher_and_verifier_bounded_outcomes_by_model`
- `future_exact_yield_delta`
- `online_learning_curves`
- `negative_transfer_and_forgetting`
- `protected_retention`
- `contamination_false_accepts_and_abstentions`
- `weight_growth_and_update_sparsity`
- `transaction_head_ancestry`
- `checker_calls_tokens_and_timing`
- `effects_and_uncertainty_over_distinct_future_units`
- `raw_output_uniqueness_and_reuse_count`
- `aggregate_row_recomputation`
- `attack_matrix`
- `current_adversarial_findings`
- `verifier_bounded_csl_ready_score`
- `protected_files_unchanged`
- `blocked_reason`
- `gate_check_summary`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field and every
`verifier_bounded_csl_ready_score` condition. `verifier_is_oracle` SHALL be
true only for deterministic exact outcome checkers and row arithmetic. The
self-teacher and factor-energy ranker SHALL NOT be oracles. `honest_verdict`
SHALL start with `success:`, `complete:`, or `blocked:`.

Required field principles:

- `status`: Names the terminal state for the verifier-bounded CSL run.
- `MODEL_SPECS`: Carries the three mandated GGUF model identities from cached SOTA receipts.
- `models_used`: Lists only mandated models with eligible unit rows.
- `cached_sota_pair_receipts`: Shows the helper calls used to resolve all mandated models.
- `model_and_embedded_tokenizer_hashes`: Binds model bytes and embedded tokenizer metadata.
- `autotokenizer_usage_count`: Must remain zero because GGUF tokenizers are embedded.
- `device_and_runner_receipts`: Binds GPUs, CUDA receipts, runner mode, raw outputs, and CPU-fallback checks.
- `sealed_stream_arm_and_analysis_manifest`: Freezes units, arms, candidates, seeds, budgets, and analysis before updates.
- `path_nonexistence_and_freshness_receipts`: Proves raw-output, ledger, and result paths were fresh before the run.
- `exact_checker_and_update_rule_hashes`: Pins deterministic checker and update-rule code.
- `event_store_and_initial_head_hashes`: Records atomic event storage and independent initial heads.
- `per_unit_rows`: Contains every model, chronological unit, and arm row before aggregate calculation.
- `chronology_and_future_only_checks`: Proves decisions read only prior state and writes affect later units only.
- `frozen_teacher_and_verifier_bounded_outcomes_by_model`: Reports exact outcomes by model and arm.
- `future_exact_yield_delta`: Reports verifier-bounded future yield lift over frozen and teacher.
- `online_learning_curves`: Shows chronological improvement from row data.
- `negative_transfer_and_forgetting`: Reports harmful transfer and retained prior behavior.
- `protected_retention`: Protects protected cases from learned-weight regressions.
- `contamination_false_accepts_and_abstentions`: Counts leakage, false accepts, and abstentions.
- `weight_growth_and_update_sparsity`: Shows weight caps, clamp counts, and sparse updates.
- `transaction_head_ancestry`: Proves each arm has a separate head chain.
- `checker_calls_tokens_and_timing`: Charges exact checks, model-evidence bytes, and measured timing.
- `effects_and_uncertainty_over_distinct_future_units`: Computes uncertainty over later units.
- `raw_output_uniqueness_and_reuse_count`: Proves fresh candidate bytes were not reused.
- `aggregate_row_recomputation`: Recomputes reported metrics from rows.
- `attack_matrix`: Shows critical leakage, authority, state, receipt, and timing attacks fail closed.
- `current_adversarial_findings`: Keeps current critical findings visible.
- `verifier_bounded_csl_ready_score`: Conjunctive readiness for exact-signed bounded CSL.
- `protected_files_unchanged`: Shows protected files stayed byte-identical.
- `blocked_reason`: Explains failed preconditions for blocked artifacts.
- `gate_check_summary`: Summarizes readiness gates and blocker count.
- `preconditions_checked`: Records required hardware, cache, tokenizer, checker, path, clock, and disk checks.
- `inference_substrate`: Declares local SOTA GGUF CUDA receipts with exact checker governed external weights.
- `verifier_is_oracle`: Marks only exact checker and row arithmetic as oracle boundaries.
- `field_principles`: Documents why each field and readiness condition exists.
- `field_provenance`: Maps each field to specs, manifests, rows, receipts, attacks, or tests.
- `random_seed`: Pins streams, candidates, updates, and attacks.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records focused, coverage, full pytest, spec, E2E, adversarial, row, determination, and clutter checks.
- `reproducibility_checksum`: Content-addresses the artifact with volatile fields normalized.
- `honest_verdict`: Uses a terminal prefix and states the exact-signed boundary.

## SCENARIO-LEARN-6455-SPEC: Spec Owns The Artifact Contract

**Given** Exp6455 is an FR-11 continuous-learning experiment
**When** the OpenSpec is read
**Then** all required artifact fields, scenarios, and readiness conditions
SHALL be declared before implementation.

## SCENARIO-LEARN-6455-MODELS: Cached GGUFs And Embedded Tokenizers Are Used

**Given** the three mandated GGUFs are cached
**When** Exp6455 builds model specs
**Then** model rows SHALL come from cached SOTA helper calls, embedded
tokenizer receipts SHALL be present, and `autotokenizer_usage_count` SHALL be
zero.

## SCENARIO-LEARN-6455-CHRONOLOGY: Updates Affect Only Later Units

**Given** a chronological unit and an arm state
**When** a candidate is selected and checked
**Then** the update SHALL commit only after the exact result and SHALL NOT
change the same unit's selection.

## SCENARIO-LEARN-6455-VERIFIER-SIGN: Exact Results Own The Update Direction

**Given** model evidence and exact checker feedback disagree
**When** the verifier-bounded arm updates weights
**Then** the update sign SHALL equal the exact result sign, and the model
evidence SHALL affect only the bounded nonnegative magnitude.

## SCENARIO-LEARN-6455-ROWS: Aggregates Recompute From Per-Unit Rows

**Given** all model, arm, and unit rows are present
**When** Exp6455 reports future yield, retention, growth, and cost
**Then** those metrics SHALL recompute from row data without aggregate-only
state.

## SCENARIO-LEARN-6455-ATTACKS: Critical Attacks Fail Closed

**Given** leakage, authority, state-sharing, output-reuse, fake-receipt,
CPU-fallback, timing, or aggregate-mismatch attacks
**When** Exp6455 validates its attack matrix
**Then** no attack SHALL promote readiness or override an exact checker.

## SCENARIO-LEARN-6455-READY: Readiness Requires Future Exact Gain

**Given** all preconditions, tests, attacks, duration checks, and protected
retention gates pass
**When** verifier-bounded updates improve future exact yield over frozen and
teacher-signed updates on distinct later units
**Then** `verifier_bounded_csl_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6455)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6455 | Planned: `python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`; terminal artifact `results/experiment_6455_prospective_verifier_bounded_factor_weight_csl.json`. | Planned: `tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. |
| SCENARIO-LEARN-6455-SPEC | Planned: `python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. | Planned: `tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. |
| SCENARIO-LEARN-6455-MODELS | Planned: `python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. | Planned: `tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. |
| SCENARIO-LEARN-6455-CHRONOLOGY | Planned: `python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. | Planned: `tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. |
| SCENARIO-LEARN-6455-VERIFIER-SIGN | Planned: `python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. | Planned: `tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. |
| SCENARIO-LEARN-6455-ROWS | Planned: `python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. | Planned: `tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. |
| SCENARIO-LEARN-6455-ATTACKS | Planned: `python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. | Planned: `tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. |
| SCENARIO-LEARN-6455-READY | Planned: `python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. | Planned: `tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py`. |

## REQ-LEARN-6456: Corrupt-Feedback Held-Restart CSL Replication

**Given** Exp6455 reports eligible verifier-bounded factor-weight CSL evidence
and Exp6432 reported held restart evidence that is duration-flagged
**When** Exp6456 runs on planning date 20260815
**Then** it SHALL write
`results/experiment_6456_corrupt_feedback_held_restart_csl_replication.json`
**And** it SHALL compare frozen weights, clean verifier-bounded updates, and
governed verifier-bounded updates on a new held binding-shift stream.

Exp6456 SHALL require `verifier_bounded_csl_ready_score=1.0`, authenticated
Exp6455 upstream state and receipts, both RTX 3090 GPUs, all three mandated
GGUF files, embedded GGUF tokenizers, exact deterministic checkers, new held
paths, enough disk and wall time, a sealed held stream, and a sealed corruption
schedule before readiness can become one.

Exp6456 SHALL define `MODEL_SPECS` through `cached_sota_pair()` or the same
resolver. It SHALL include exactly `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF`. It
SHALL use embedded GGUF tokenizers only and SHALL NOT call `AutoTokenizer`.

Exp6456 SHALL seal at least 24 held units per model. The held stream SHALL use
new bindings, new clause combinations, protected cases, and zero problem or
raw-hash overlap with Exp6455 and Exp6432. The frozen, clean, and governed
arms SHALL receive the same candidate bytes for each model and held unit.

Exp6456 SHALL freeze the Exp6455 update rule and initial eligible head. It
SHALL run a frozen-weight arm, a clean verifier-bounded update arm, and a
governed verifier-bounded update arm. The governed arm SHALL inject exactly
one predeclared checker-transport corruption per model-session panel.

Exp6456 SHALL start a new process at each session boundary. Each child process
SHALL reload state from disk, validate transaction ancestry and head hash, and
record parent PID, child PID, start time, exit code, and whether inherited
in-memory state was visible. No unhashed in-memory state SHALL be accepted.

Exp6456 SHALL bind raw output, parse output, factor input, checker request,
checker transport, checker response, update, and head transition with path
receipts. A corrupt checker-transport event SHALL break the expected path hash
before update admission. A transport-corrupted checker response SHALL NOT be
authoritative.

Exp6456 SHALL quarantine every corrupt event, write a tombstone, roll back to
the last good head, restart, and prove that no corrupt update can resurrect in
later release state.

Exp6456 SHALL emit per-unit rows for every held unit, model, arm, session, and
process. Rows SHALL include candidate hashes, exact outcome, update, corrupt
event state, quarantine state, rollback state, head hash, protected outcome,
checker work, and timing.

Exp6456 SHALL recompute held future exact yield, negative transfer,
forgetting, protected retention, false accepts, abstentions, quarantine
precision and recall, rollback success, resurrection count, restart recovery,
and cost from rows. Uncertainty SHALL use distinct held units, not duplicated
arm rows.

Exp6456 SHALL attack cached development bytes, fake restart, inherited state,
stale or forged head, missed corruption, quarantine false positive,
rollback-to-bad-head, tombstone deletion, corrupt update resurrection, CPU
fallback, timing synthesis, and aggregate-row mismatch. Every critical attack
SHALL fail closed.

Exp6456 SHALL set `csl_safety_replication_ready_score=1.0` only when the clean
learner retains a positive held future exact effect, governed learning
contains every corrupt event with zero protected release and zero resurrection,
benign utility stays within tolerance, all restart and freshness checks pass,
all three models have eligible rows, duration is eligible, aggregates
recompute, and critical findings are zero.

Exp6456 SHALL emit these fields:

- `status`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `model_and_embedded_tokenizer_hashes`
- `autotokenizer_usage_count`
- `device_and_runner_receipts`
- `upstream_gate_value_policy_and_head_hashes`
- `sealed_held_stream_corruption_and_analysis_manifest`
- `path_nonexistence_freshness_and_disjointness_receipts`
- `process_restart_and_pid_receipts`
- `per_unit_rows`
- `frozen_clean_and_governed_outcomes_by_model`
- `future_exact_yield_delta`
- `negative_transfer_and_forgetting`
- `protected_retention`
- `false_accepts_and_abstentions`
- `corruption_detection_and_path_receipts`
- `quarantine_precision_and_recall`
- `tombstone_rollback_and_resurrection_results`
- `transaction_ancestry_and_restart_recovery`
- `checker_calls_tokens_and_timing`
- `effects_and_uncertainty_over_distinct_held_units`
- `aggregate_row_recomputation`
- `attack_matrix`
- `current_adversarial_findings`
- `csl_safety_replication_ready_score`
- `protected_files_unchanged`
- `blocked_reason`
- `gate_check_summary`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field and every
`csl_safety_replication_ready_score` condition. `verifier_is_oracle` SHALL be
true only for deterministic exact checkers and row arithmetic. A
transport-corrupted checker response SHALL be recorded as non-authoritative.
`honest_verdict` SHALL start with `success:`, `complete:`, or `blocked:`.

Required field principles:

- `status`: Names the terminal state for the corrupt-feedback held restart replication.
- `MODEL_SPECS`: Carries the three mandated GGUF model identities from cached SOTA receipts.
- `models_used`: Lists only mandated models with eligible unit rows.
- `cached_sota_pair_receipts`: Shows the helper calls used to resolve all mandated models.
- `model_and_embedded_tokenizer_hashes`: Binds model bytes and embedded tokenizer metadata.
- `autotokenizer_usage_count`: Must remain zero because GGUF tokenizers are embedded.
- `device_and_runner_receipts`: Binds GPUs, CUDA receipts, runner mode, raw outputs, and CPU-fallback checks.
- `upstream_gate_value_policy_and_head_hashes`: Freezes Exp6455 readiness, update rule, model policy, and initial heads.
- `sealed_held_stream_corruption_and_analysis_manifest`: Freezes held units, arms, sessions, candidates, corruption schedule, seeds, budgets, and analysis.
- `path_nonexistence_freshness_and_disjointness_receipts`: Proves result, raw-output, ledger, quarantine, and tombstone paths are fresh and disjoint from Exp6455 and Exp6432.
- `process_restart_and_pid_receipts`: Proves session children reload disk state with new PIDs and no inherited in-memory state.
- `per_unit_rows`: Contains every model, held unit, arm, session, process, receipt, update, quarantine, rollback, and timing row before aggregate calculation.
- `frozen_clean_and_governed_outcomes_by_model`: Reports exact outcomes by model and arm.
- `future_exact_yield_delta`: Reports clean and governed future yield lift over frozen weights.
- `negative_transfer_and_forgetting`: Reports harmful transfer and retained prior behavior.
- `protected_retention`: Protects held protected cases from learned-weight regressions.
- `false_accepts_and_abstentions`: Counts false accepts and abstentions from row data.
- `corruption_detection_and_path_receipts`: Shows every injected corrupt transport event broke the expected path hash before update admission.
- `quarantine_precision_and_recall`: Requires all and only corrupt events to enter quarantine.
- `tombstone_rollback_and_resurrection_results`: Proves tombstones persist, rollback restores last good heads, and corrupt updates do not resurrect.
- `transaction_ancestry_and_restart_recovery`: Proves clean and governed head chains recover after process restarts.
- `checker_calls_tokens_and_timing`: Charges exact checks, model-evidence bytes, receipt work, and measured timing.
- `effects_and_uncertainty_over_distinct_held_units`: Computes uncertainty over distinct held units.
- `aggregate_row_recomputation`: Recomputes reported metrics from rows.
- `attack_matrix`: Shows critical restart, state, quarantine, rollback, receipt, and timing attacks fail closed.
- `current_adversarial_findings`: Keeps current critical findings visible.
- `csl_safety_replication_ready_score`: Conjunctive readiness for corrupt-feedback held restart safety.
- `protected_files_unchanged`: Shows protected files stayed byte-identical.
- `blocked_reason`: Explains failed preconditions for blocked artifacts.
- `gate_check_summary`: Summarizes readiness gates and blocker count.
- `preconditions_checked`: Records hardware, cache, tokenizer, upstream, checker, path, clock, disk, and schedule checks.
- `inference_substrate`: Declares local SOTA GGUF CUDA receipts with deterministic exact checker governed external weights.
- `verifier_is_oracle`: Marks only exact checker and row arithmetic as oracle boundaries.
- `field_principles`: Documents why each field and readiness condition exists.
- `field_provenance`: Maps each field to specs, manifests, rows, receipts, attacks, or tests.
- `random_seed`: Pins streams, candidates, corrupt schedule, restarts, and attacks.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records focused, coverage, full pytest, spec, E2E, adversarial, row, determination, and clutter checks.
- `reproducibility_checksum`: Content-addresses the artifact with volatile fields normalized.
- `honest_verdict`: Uses a terminal prefix and states the corrupt-feedback safety result.
- `csl_safety_replication_ready_score:clean_positive_held_effect`: Clean verifier-bounded learning must retain a positive held exact effect.
- `csl_safety_replication_ready_score:governed_contains_corruption`: Governed learning must detect and contain every scheduled corrupt transport event.
- `csl_safety_replication_ready_score:zero_protected_release`: Corrupt events must not release protected or non-authoritative updates.
- `csl_safety_replication_ready_score:zero_resurrection`: Tombstoned corrupt updates must not resurrect.
- `csl_safety_replication_ready_score:benign_utility_within_tolerance`: Governed benign utility must remain within the preregistered tolerance.
- `csl_safety_replication_ready_score:restart_recovery`: Session restarts must recover the expected disk head.
- `csl_safety_replication_ready_score:fresh_disjoint_paths`: Held paths and raw hashes must be fresh and disjoint.
- `csl_safety_replication_ready_score:all_models_have_rows`: All three mandated models must have eligible rows.
- `csl_safety_replication_ready_score:duration_eligible`: Duration must be eligible for the declared substrate.
- `csl_safety_replication_ready_score:aggregates_recompute`: Aggregates must recompute from per-unit rows.
- `csl_safety_replication_ready_score:zero_critical_findings`: Current critical findings must be zero.

## SCENARIO-LEARN-6456-SPEC: Spec Owns The Artifact Contract

**Given** Exp6456 is a held CSL safety replication
**When** the OpenSpec is read
**Then** all required artifact fields, scenarios, and readiness conditions
SHALL be declared before implementation.

## SCENARIO-LEARN-6456-MODELS: Cached GGUFs And Embedded Tokenizers Are Used

**Given** the three mandated GGUFs are cached
**When** Exp6456 builds model specs
**Then** model rows SHALL come from cached SOTA helper calls, embedded
tokenizer receipts SHALL be present, and `autotokenizer_usage_count` SHALL be
zero.

## SCENARIO-LEARN-6456-HELD-STREAM: New Held Bindings Are Sealed

**Given** Exp6455 and Exp6432 have existing problem and raw hashes
**When** Exp6456 seals its held stream
**Then** every held unit SHALL have a new binding, new clause combination,
protected case, matched candidates, and zero overlap with the upstream hashes.

## SCENARIO-LEARN-6456-RESTARTS: Session Boundaries Use New Processes

**Given** clean and governed heads are persisted at session boundaries
**When** each session starts
**Then** a child process SHALL reload the expected head from disk, report a PID
different from the parent, and reject inherited in-memory state.

## SCENARIO-LEARN-6456-PATH-CORRUPTION: Corrupt Transport Breaks Before Update

**Given** a predeclared corrupt checker-transport event
**When** the governed arm validates path receipts
**Then** the expected path hash SHALL fail before update admission and the
transport-corrupted checker response SHALL be non-authoritative.

## SCENARIO-LEARN-6456-QUARANTINE-ROLLBACK: Tombstones Prevent Resurrection

**Given** a corrupt governed event has been detected
**When** quarantine, tombstone, rollback, and restart run
**Then** the last good head SHALL be restored and the corrupt update hash SHALL
not appear in any later active or release head.

## SCENARIO-LEARN-6456-ROWS: Aggregates Recompute From Per-Unit Rows

**Given** all model, arm, held unit, session, and process rows are present
**When** Exp6456 reports yield, retention, quarantine, rollback, restart, and
cost metrics
**Then** those metrics SHALL recompute from row data without aggregate-only
state.

## SCENARIO-LEARN-6456-ATTACKS: Critical Safety Attacks Fail Closed

**Given** cached-byte, fake-restart, inherited-state, forged-head,
missed-corruption, false-quarantine, bad-rollback, tombstone-deletion,
resurrection, CPU-fallback, timing, and aggregate attacks
**When** Exp6456 validates its attack matrix
**Then** no attack SHALL promote readiness, release a corrupt update, or
override an exact checker.

## SCENARIO-LEARN-6456-READY: Readiness Requires Utility And Containment

**Given** all preconditions, tests, attacks, duration checks, restart checks,
freshness checks, and protected-retention gates pass
**When** clean learning improves held future exact yield and governed learning
contains every corrupt event with no resurrection and acceptable benign utility
**Then** `csl_safety_replication_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6456)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6456 | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`; terminal artifact `results/experiment_6456_corrupt_feedback_held_restart_csl_replication.json`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |
| SCENARIO-LEARN-6456-SPEC | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |
| SCENARIO-LEARN-6456-MODELS | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |
| SCENARIO-LEARN-6456-HELD-STREAM | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |
| SCENARIO-LEARN-6456-RESTARTS | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |
| SCENARIO-LEARN-6456-PATH-CORRUPTION | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |
| SCENARIO-LEARN-6456-QUARANTINE-ROLLBACK | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |
| SCENARIO-LEARN-6456-ROWS | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |
| SCENARIO-LEARN-6456-ATTACKS | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |
| SCENARIO-LEARN-6456-READY | Planned: `python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. | Planned: `tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py`. |

## REQ-LEARN-6457: Independent Verifier-Bounded CSL Audit

**Given** Exp6455 and Exp6456 may report positive verifier-bounded CSL
evidence
**When** Exp6457 runs on planning date 20260815
**Then** it SHALL write
`results/experiment_6457_independent_verifier_bounded_csl_audit.json`
**And** it SHALL recompute the prospective and held CSL effects from immutable
row and event evidence without importing upstream aggregate, readiness, gate,
update, or verdict functions.

Exp6457 SHALL inventory Exp6433, Exp6444, Exp6455, Exp6456, their source
files, tests, immutable rows, raw outputs, event stores, transactions, heads,
receipts, and checker files before importing any experiment module. Missing,
zero-byte, malformed, blocked, skipped, flagged, or underpowered upstream
artifacts SHALL remain visible audit inputs.

Exp6457 SHALL freeze upstream paths, sizes, hashes, status, honest verdict,
readiness fields, adversarial findings, row counts, duration, substrate, and
model receipts. The audit SHALL not mutate upstream evidence.

Exp6457 SHALL implement independent reducers from documented row schemas only.
It SHALL run without importing upstream aggregate, readiness, gate, update, or
verdict functions.
The reducers SHALL recompute future exact yield, learning curves, negative
transfer, forgetting, protected retention, false accepts, abstentions, weight
growth, update sparsity, cost, held replication, corruption detection,
quarantine precision and recall, rollback success, resurrection count, restart
recovery, and protected releases.

Exp6457 SHALL verify exact checker authority, teacher-signal nonnegative
magnitude, future-only updates, separate arm state, transaction ancestry,
raw-output uniqueness, development-held disjointness, real process boundaries,
path continuity, task-scoped CUDA receipts, duration floors, and exact-veto
preservation.

Exp6457 SHALL emit per-unit rows for every audited row or a stable row
reference. Each audit row SHALL include upstream task, row id, upstream values,
independently recomputed values, mismatch state, inclusion decision, and
evidence path. Missing or excluded rows SHALL remain visible.

Exp6457 SHALL replay exact-veto, corrupt-feedback containment, tombstone
persistence, and aggregate-row consistency attacks independently. Critical
findings SHALL fail closed.

Exp6457 SHALL set `csl_audit_ready_score=1.0` only when all required upstream
evidence exists, all positive effects recompute, no material mismatch or
critical finding remains, safety and restart gates pass, duration and substrate
are eligible, protected files are unchanged, and verification commands pass.
Otherwise it SHALL use `complete_null` or `complete_blocked`, state every
reason, and populate `gate_check_summary` for any blocked verdict even though
the audit is ungated.

Exp6457 SHALL emit these fields:

- `status`
- `upstream_inventory_and_hashes`
- `upstream_status_verdict_readiness_duration_substrate_and_findings`
- `independent_reducer_source_and_test_hashes`
- `per_unit_rows`
- `prospective_metric_recomputation`
- `held_metric_recomputation`
- `update_direction_and_chronology_checks`
- `weight_growth_forgetting_and_protected_retention_checks`
- `corruption_quarantine_rollback_and_resurrection_checks`
- `raw_output_uniqueness_and_partition_intersections`
- `transaction_head_and_restart_checks`
- `path_receipt_and_exact_veto_checks`
- `upstream_vs_recomputed_mismatches`
- `mismatch_count_and_materiality`
- `independent_attack_replay`
- `duration_and_substrate_eligibility`
- `prospective_csl_eligibility`
- `csl_ineligibility_reasons`
- `csl_audit_ready_score`
- `current_adversarial_findings`
- `protected_files_unchanged`
- `blocked_reason`
- `gate_check_summary`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field and every readiness
condition. `verifier_is_oracle` SHALL be true only for deterministic exact
checkers and independently recomputed row arithmetic. `honest_verdict` SHALL
start with `success:`, `complete:`, or `blocked:`.

Required readiness principles:

- `csl_audit_ready_score:required_upstream_evidence_exists`: Required upstream artifacts and named evidence files must exist and be readable.
- `csl_audit_ready_score:prospective_positive_effect_recomputes`: Exp6455 prospective yield and safety metrics must recompute from rows.
- `csl_audit_ready_score:held_positive_effect_recomputes`: Exp6456 held yield and containment metrics must recompute from rows.
- `csl_audit_ready_score:zero_material_mismatch`: Material upstream and recomputed values must match within exact tolerances.
- `csl_audit_ready_score:update_direction_authority`: Exact checker signs alone must determine verifier-bounded update direction.
- `csl_audit_ready_score:safety_and_restart_gates_pass`: Corruption, quarantine, rollback, tombstone, resurrection, and restart checks must pass.
- `csl_audit_ready_score:raw_outputs_unique_and_partitions_disjoint`: Raw hashes must be unique, and development and held partitions must not overlap.
- `csl_audit_ready_score:duration_and_substrate_eligible`: Upstream and audit durations must satisfy the declared substrate floors.
- `csl_audit_ready_score:zero_current_critical_findings`: Current adversarial, row, artifact, and determination checks must report no critical finding.
- `csl_audit_ready_score:verification_commands_pass`: Focused tests, coverage, full pytest, spec coverage, row, adversarial, determination, artifact, and clutter checks must pass.

## SCENARIO-LEARN-6457-SPEC: Spec Owns The Audit Contract

**Given** Exp6457 is an independent CSL audit
**When** the OpenSpec is read
**Then** all required artifact fields, scenarios, and readiness principles
SHALL be declared before implementation.

## SCENARIO-LEARN-6457-INVENTORY: Upstream Evidence Is Frozen First

**Given** Exp6457 starts
**When** it inventories Exp6433, Exp6444, Exp6455, and Exp6456
**Then** every present artifact and referenced evidence path SHALL record
existence, size, hash, status, duration, substrate, readiness fields, and
findings before any experiment module is imported.

## SCENARIO-LEARN-6457-REDUCERS: Aggregates Recompute From Rows

**Given** Exp6455 and Exp6456 per-unit rows are present
**When** Exp6457 recomputes metrics
**Then** prospective yield, held yield, retention, forgetting, growth,
quarantine, rollback, restart, and cost metrics SHALL come from row fields and
receipt files, not from upstream aggregate functions.

## SCENARIO-LEARN-6457-AUTHORITY: Exact Checker Direction Is Preserved

**Given** teacher evidence and exact checker outcomes disagree
**When** Exp6457 audits verifier-bounded updates
**Then** exact checker signs SHALL be the only update direction authority, and
teacher evidence SHALL remain a nonnegative magnitude signal.

## SCENARIO-LEARN-6457-SAFETY: Corrupt Feedback Cannot Resurrect

**Given** Exp6456 scheduled corrupt transport events
**When** Exp6457 recomputes quarantine, tombstone, rollback, and restart rows
**Then** every corrupt event SHALL be detected and quarantined, no corrupt
update SHALL commit, and resurrection count SHALL remain zero.

## SCENARIO-LEARN-6457-READY: Blocked Verdicts Still Explain Gates

**Given** required upstream evidence is missing, malformed, duration-ineligible,
or materially mismatched
**When** Exp6457 writes a terminal artifact
**Then** `csl_audit_ready_score` SHALL be `0.0`, status SHALL be
`complete_blocked` or `complete_null`, all reasons SHALL appear in
`csl_ineligibility_reasons`, and `gate_check_summary` SHALL be populated.

## Implementation Status (REQ-LEARN-6457)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6457 | Planned: `python/carnot/experiment_6457_independent_verifier_bounded_csl_audit.py`; terminal artifact `results/experiment_6457_independent_verifier_bounded_csl_audit.json`. | Planned: `tests/python/test_experiment_6457_independent_verifier_bounded_csl_audit.py`. |
| SCENARIO-LEARN-6457-SPEC | Planned: `python/carnot/experiment_6457_independent_verifier_bounded_csl_audit.py`. | Planned: `tests/python/test_experiment_6457_independent_verifier_bounded_csl_audit.py`. |
| SCENARIO-LEARN-6457-INVENTORY | Planned: `python/carnot/experiment_6457_independent_verifier_bounded_csl_audit.py`. | Planned: `tests/python/test_experiment_6457_independent_verifier_bounded_csl_audit.py`. |
| SCENARIO-LEARN-6457-REDUCERS | Planned: `python/carnot/experiment_6457_independent_verifier_bounded_csl_audit.py`. | Planned: `tests/python/test_experiment_6457_independent_verifier_bounded_csl_audit.py`. |
| SCENARIO-LEARN-6457-AUTHORITY | Planned: `python/carnot/experiment_6457_independent_verifier_bounded_csl_audit.py`. | Planned: `tests/python/test_experiment_6457_independent_verifier_bounded_csl_audit.py`. |
| SCENARIO-LEARN-6457-SAFETY | Planned: `python/carnot/experiment_6457_independent_verifier_bounded_csl_audit.py`. | Planned: `tests/python/test_experiment_6457_independent_verifier_bounded_csl_audit.py`. |
| SCENARIO-LEARN-6457-READY | Planned: `python/carnot/experiment_6457_independent_verifier_bounded_csl_audit.py`. | Planned: `tests/python/test_experiment_6457_independent_verifier_bounded_csl_audit.py`. |

## REQ-LEARN-6468: Unique-Event Verifier-Bounded CSL

**Given** Exp6455 reported a positive verifier-bounded factor-weight effect
and Exp6457 found cloned raw output evidence and an exact-veto weakness
**When** Exp6468 runs on planning date 20260819
**Then** it SHALL write
`results/experiment_6468_unique_event_verifier_bounded_csl.json`
**And** it SHALL compare frozen factor weights, self-signed updates, and
verifier-bounded exact-sign updates on a fresh sealed chronological stream.

Exp6468 SHALL require both RTX 3090 GPUs, cached mandatory GGUF models,
embedded GGUF tokenizer metadata, new raw-output paths, an empty event-id
registry, and a sealed development, prospective-update, and future-held split
before readiness can become one.

Exp6468 SHALL define `MODEL_SPECS` through cached local resolution. It SHALL
include exactly `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL use embedded GGUF tokenizers only.
It SHALL not call `AutoTokenizer`. Legacy models may appear only as
smoke-test policy entries and not as headline rows.

Exp6468 SHALL seal development, prospective-update, and future-held units
before inference. It SHALL write an exposure ledger before inference. The
ledger SHALL prove that future-held outcomes are not exposed to model prompts,
state snapshots, update admission, or parsing.

Exp6468 SHALL generate exactly one fresh raw model output for each event. It
SHALL persist and hash raw bytes before parsing. Each event id SHALL be
non-empty and unique. No candidate row, arm row, event row, or per-unit row
SHALL reuse another event's raw hash.

Exp6468 SHALL run the deterministic exact checker before any write admission.
An admitted write SHALL include the checker receipt, pre-state head, write
decision, post-state head, and rollback pointer. A failed checker-authority receipt SHALL leave the persistent head unchanged.

Exp6468 SHALL apply bounded external factor-weight updates only after exact
checker authority is present. The exact outcome SHALL choose update direction.
Model confidence MAY scale only the nonnegative magnitude. The base GGUF files
SHALL remain frozen.

Exp6468 SHALL emit per-unit rows and event rows. Rows SHALL include chronology,
raw hash, arm, pre-state, checker result, write decision, post-state, future
exact outcome, and rollback pointer. Aggregates SHALL recompute from these
rows.

Exp6468 SHALL attack cloned raw output, duplicate event id, held exposure,
self-signed false pass, exact-veto bypass, future leakage, protected-case
regression, and aggregate mismatch. Every critical attack SHALL fail closed.

Exp6468 SHALL set `unique_event_csl_ready_score=1.0` only when one-to-one
event provenance holds, exact veto precedes every write, the verifier-bounded
arm improves future exact yield over both frozen and self-signed arms,
protected cases do not regress, model files are immutable, CPU fallback is
zero, aggregates recompute from rows, and critical attacks fail closed.

Exp6468 SHALL emit these fields:

- `status`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `model_file_and_embedded_tokenizer_hashes`
- `autotokenizer_usage_count`
- `device_and_runner_receipts`
- `sealed_chronological_manifest`
- `exposure_ledger`
- `update_rule_and_bounds`
- `raw_output_manifest`
- `event_identity_manifest`
- `exact_veto_before_write_receipts`
- `per_unit_rows`
- `event_rows`
- `effect_by_arm_and_interval`
- `protected_case_retention`
- `write_and_rollback_counts`
- `one_event_one_raw_hash_check`
- `cpu_fallback_count`
- `aggregate_row_recomputation`
- `attack_matrix`
- `current_adversarial_findings`
- `unique_event_csl_ready_score`
- `protected_files_unchanged`
- `blocked_reason`
- `gate_check_summary`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field and every
`unique_event_csl_ready_score` condition. `verifier_is_oracle` SHALL be true
only for deterministic checker authority, chronology checks, and row
arithmetic. The self-signed arm, factor ranker, parser, and model confidence
SHALL NOT be oracles. `honest_verdict` SHALL start with `success:`,
`complete:`, or `blocked:`.

Required field principles:

- `status`: Names the terminal state for the unique-event CSL run.
- `MODEL_SPECS`: Carries the three mandated cached GGUF model identities.
- `models_used`: Lists only mandated models with eligible live event rows.
- `cached_sota_pair_receipts`: Shows the cached local resolver calls.
- `model_file_and_embedded_tokenizer_hashes`: Binds model bytes and embedded tokenizer metadata.
- `autotokenizer_usage_count`: Must remain zero because GGUF tokenizers are embedded.
- `device_and_runner_receipts`: Binds GPUs, CUDA, llama.cpp, generation calls, and CPU fallback checks.
- `sealed_chronological_manifest`: Freezes units, intervals, arms, seeds, and budgets before inference.
- `exposure_ledger`: Proves held outcomes are not visible before inference or update admission.
- `update_rule_and_bounds`: Pins exact-sign authority, confidence magnitude use, and bounds.
- `raw_output_manifest`: Proves raw bytes were persisted and validated before parse.
- `event_identity_manifest`: Proves event ids are non-empty and unique.
- `exact_veto_before_write_receipts`: Proves checker authority precedes each admitted write.
- `per_unit_rows`: Contains row data before aggregate calculation.
- `event_rows`: Contains one generation event for each per-unit row.
- `effect_by_arm_and_interval`: Reports exact yield by arm and chronological interval.
- `protected_case_retention`: Blocks utility that harms protected cases.
- `write_and_rollback_counts`: Counts admitted writes, vetoes, and rollback pointers.
- `one_event_one_raw_hash_check`: Proves no raw hash is cloned across rows.
- `cpu_fallback_count`: Must be zero for ready live local GGUF evidence.
- `aggregate_row_recomputation`: Recomputes reported metrics from rows.
- `attack_matrix`: Shows critical event, veto, leakage, and aggregate attacks fail closed.
- `current_adversarial_findings`: Keeps current critical findings visible.
- `unique_event_csl_ready_score`: Conjunctive readiness for unique-event exact-veto CSL.
- `protected_files_unchanged`: Shows conductor, ops, traceability, and upstream evidence stayed byte-identical.
- `blocked_reason`: Explains failed preconditions for blocked artifacts.
- `gate_check_summary`: Summarizes readiness gates and blockers.
- `preconditions_checked`: Records hardware, cache, tokenizer, path, event-id, split, and checker checks.
- `inference_substrate`: Declares local SOTA GGUF live inference with exact-checker-governed external weights.
- `verifier_is_oracle`: Marks only deterministic checker, chronology, and row arithmetic as oracle boundaries.
- `field_principles`: Documents why each field and readiness condition exists.
- `field_provenance`: Maps fields to specs, manifests, rows, receipts, attacks, or tests.
- `random_seed`: Pins streams, events, prompts, updates, and attacks.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records focused, coverage, full pytest, spec, row, adversarial, and E2E checks.
- `reproducibility_checksum`: Content-addresses the artifact with volatile fields normalized.
- `honest_verdict`: Uses a terminal prefix and states the exact-veto boundary.

## SCENARIO-LEARN-6468-SPEC: Spec Owns The Unique-Event Contract

**Given** Exp6468 is an FR-11 continuous-learning experiment
**When** the OpenSpec is read
**Then** all required artifact fields, scenarios, attacks, and readiness
conditions SHALL be declared before implementation.

## SCENARIO-LEARN-6468-MODELS: Cached GGUFs And Embedded Tokenizers Are Used

**Given** the three mandated GGUFs are cached
**When** Exp6468 builds model specs
**Then** model rows SHALL come from cached local SOTA helper calls, embedded
tokenizer receipts SHALL be present, and `autotokenizer_usage_count` SHALL be
zero.

## SCENARIO-LEARN-6468-SEALED-SPLIT: Exposure Is Recorded Before Inference

**Given** development, prospective-update, and future-held units are sealed
**When** Exp6468 starts inference
**Then** the exposure ledger SHALL already exist and SHALL show zero held
outcome exposure.

## SCENARIO-LEARN-6468-UNIQUE-EVENTS: One Event Has One Raw Hash

**Given** event rows for all model, interval, and arm combinations
**When** Exp6468 validates provenance
**Then** event ids SHALL be unique, raw hashes SHALL be unique, and each
per-unit row SHALL reference exactly one event row.

## SCENARIO-LEARN-6468-EXACT-VETO: Checker Authority Precedes Writes

**Given** a candidate update and a pre-state head
**When** checker authority is absent or failed
**Then** the write SHALL be rejected and the post-state head SHALL equal the
pre-state head.

## SCENARIO-LEARN-6468-UPDATE-RULE: Exact Outcome Owns Direction

**Given** model confidence and exact outcome disagree
**When** the verifier-bounded arm updates weights
**Then** the update sign SHALL equal the exact outcome sign, and model
confidence SHALL only scale nonnegative magnitude.

## SCENARIO-LEARN-6468-AGGREGATES: Rows Own The Effect

**Given** per-unit rows and event rows are present
**When** Exp6468 reports arm effects, retention, writes, and raw provenance
**Then** those metrics SHALL recompute from rows without aggregate-only state.

## SCENARIO-LEARN-6468-ATTACKS: Critical Event Attacks Fail Closed

**Given** cloned raw output, duplicate event id, held exposure, self-signed
false pass, exact-veto bypass, future leakage, protected regression, or
aggregate mismatch attacks
**When** Exp6468 validates its attack matrix
**Then** no attack SHALL promote readiness or admit an unchecked write.

## SCENARIO-LEARN-6468-READY: Readiness Requires Future Exact Gain

**Given** one-to-one event provenance, exact-veto-before-write receipts,
protected retention, immutable models, zero CPU fallback, row recomputation,
and fail-closed attacks
**When** verifier-bounded updates improve future exact yield over frozen and
self-signed updates
**Then** `unique_event_csl_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6468)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6468 | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`; terminal artifact `results/experiment_6468_unique_event_verifier_bounded_csl.json`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |
| SCENARIO-LEARN-6468-SPEC | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |
| SCENARIO-LEARN-6468-MODELS | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |
| SCENARIO-LEARN-6468-SEALED-SPLIT | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |
| SCENARIO-LEARN-6468-UNIQUE-EVENTS | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |
| SCENARIO-LEARN-6468-EXACT-VETO | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |
| SCENARIO-LEARN-6468-UPDATE-RULE | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |
| SCENARIO-LEARN-6468-AGGREGATES | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |
| SCENARIO-LEARN-6468-ATTACKS | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |
| SCENARIO-LEARN-6468-READY | Implemented: `python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py`. | Implemented: `tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py`. |

## REQ-LEARN-6469: Unique-Event CSL Corruption Restart

**Given** Exp6468 reports `unique_event_csl_ready_score == 1.0`
**When** Exp6469 runs on planning date 20260819
**Then** it SHALL write
`results/experiment_6469_unique_event_csl_corruption_restart.json`
**And** it SHALL test new held binding-shift events after a real process
restart.

Exp6469 SHALL stop before generation if the Exp6468 gate is absent or below
one. A stopped run SHALL emit `gate_check_summary`, `blocked_reason`, and
`honest_verdict` with a `blocked:` prefix.

Exp6469 SHALL use exactly the three mandated cached GGUF model ids:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL resolve paths through the cached
helper. It SHALL use embedded GGUF tokenizer receipts only.

Exp6469 SHALL seal a new held manifest before event generation. The manifest
SHALL use new unit ids and raw-event identities. Its unit ids and raw hashes
SHALL have zero overlap with Exp6468 exposure, event, and raw-output ledgers.

Exp6469 SHALL start a new process after the manifest seal. The child SHALL
load only the committed store head and receipt chain. The artifact SHALL record
parent PID, child PID, start times, head hash, model receipts, and device
receipts.

Exp6469 SHALL generate new raw events for clean and governed arms. The exact
checker SHALL run before every write. A missing, corrupt, forged, or bypassed
checker receipt SHALL veto the write before state release.

Exp6469 SHALL inject five corruptions at named boundaries: forged pass,
replayed raw output, wrong-unit binding, corrupt checker response, and
interrupted write. Each corrupt event SHALL be quarantined and tombstoned
before rollback. Rollback SHALL restore the last valid head.

Exp6469 SHALL restart again after rollback. No tombstoned head, forged pass,
wrong binding, replayed raw output, corrupt checker response, or partial write
SHALL appear in active state or future exact outcomes after restart.

Exp6469 SHALL attack stale head, forged tombstone, wrong event binding,
replay, partial atomic write, exact-veto bypass, held contamination, and
aggregate mismatch. Every attack SHALL fail closed.

Exp6469 SHALL set `corruption_restart_ready_score=1.0` only when clean learning
retains a future exact effect, every corrupt event is blocked before release,
rollback restores the last valid head, restart cannot resurrect corruption,
and all held events are unique.

Exp6469 SHALL emit these fields:

- `status`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `model_file_and_embedded_tokenizer_hashes`
- `device_and_runner_receipts`
- `upstream_csl_hash`
- `sealed_new_held_manifest`
- `exposure_disjointness_receipts`
- `process_restart_receipts`
- `raw_output_manifest`
- `event_identity_manifest`
- `corruption_precommitment`
- `exact_veto_before_write_receipts`
- `per_unit_rows`
- `lifecycle_rows`
- `quarantine_tombstone_and_rollback_receipts`
- `non_resurrection_check`
- `clean_and_corrupt_effects`
- `protected_case_retention`
- `aggregate_row_recomputation`
- `attack_matrix`
- `current_adversarial_findings`
- `corruption_restart_ready_score`
- `protected_files_unchanged`
- `blocked_reason`
- `gate_check_summary`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field and every
`corruption_restart_ready_score` condition. `verifier_is_oracle` SHALL be true
only for deterministic checker output, hash-chain checks, lifecycle checks,
and row arithmetic. Model raw text, learned weights, and corruption payloads
SHALL NOT be oracles.

## SCENARIO-LEARN-6469-GATE: Exp6468 Gate Blocks The Run

**Given** Exp6468 is missing, malformed, or below readiness one
**When** Exp6469 checks the upstream gate
**Then** it SHALL stop before held event generation and write a blocked
artifact with `gate_check_summary`.

## SCENARIO-LEARN-6469-MANIFEST: New Held Events Are Disjoint

**Given** Exp6468 exposure and event ledgers are present
**When** Exp6469 seals its held manifest and raw events
**Then** new unit ids, event ids, and raw hashes SHALL have zero overlap with
Exp6468 evidence.

## SCENARIO-LEARN-6469-RESTART: Child Loads Only The Committed Head

**Given** the sealed manifest and committed store head
**When** Exp6469 starts the child process
**Then** the child PID SHALL differ from the parent PID and the recovered head
SHALL match the committed head from disk.

## SCENARIO-LEARN-6469-CORRUPTION: Exact Veto Runs Before Write

**Given** forged pass, replay, wrong binding, corrupt checker response, and
interrupted write events
**When** admission runs
**Then** each corrupt event SHALL be rejected before release.

## SCENARIO-LEARN-6469-ROLLBACK: Tombstone Precedes Rollback

**Given** a corrupt event computes a rejected child head
**When** quarantine, tombstone, and rollback run
**Then** the tombstone SHALL be written before rollback and rollback SHALL
restore the last valid head.

## SCENARIO-LEARN-6469-NON-RESURRECTION: Restart Cannot Revive Corruption

**Given** tombstoned corrupt heads and a post-rollback restart
**When** Exp6469 reloads state
**Then** no corrupt or tombstoned head SHALL appear in active state or future
exact outcomes.

## SCENARIO-LEARN-6469-ATTACKS: Lifecycle Attacks Fail Closed

**Given** stale head, forged tombstone, wrong binding, replay, partial write,
exact-veto bypass, held contamination, and aggregate mismatch attacks
**When** Exp6469 validates readiness
**Then** no attack SHALL promote readiness or release corrupt state.

## SCENARIO-LEARN-6469-READY: Readiness Is Conjunctive

**Given** clean learning has a positive future exact effect, corrupt events are
contained, rollback is exact, restart is clean, events are unique, protected
files are unchanged, and tests pass
**When** Exp6469 computes readiness
**Then** `corruption_restart_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6469)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6469 | Planned: `python/carnot/experiment_6469_unique_event_csl_corruption_restart.py`; terminal artifact `results/experiment_6469_unique_event_csl_corruption_restart.json`. | Planned: `tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py`. |
| SCENARIO-LEARN-6469-GATE | Planned: `python/carnot/experiment_6469_unique_event_csl_corruption_restart.py`. | Planned: `tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py`. |
| SCENARIO-LEARN-6469-MANIFEST | Planned: `python/carnot/experiment_6469_unique_event_csl_corruption_restart.py`. | Planned: `tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py`. |
| SCENARIO-LEARN-6469-RESTART | Planned: `python/carnot/experiment_6469_unique_event_csl_corruption_restart.py`. | Planned: `tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py`. |
| SCENARIO-LEARN-6469-CORRUPTION | Planned: `python/carnot/experiment_6469_unique_event_csl_corruption_restart.py`. | Planned: `tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py`. |
| SCENARIO-LEARN-6469-ROLLBACK | Planned: `python/carnot/experiment_6469_unique_event_csl_corruption_restart.py`. | Planned: `tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py`. |
| SCENARIO-LEARN-6469-NON-RESURRECTION | Planned: `python/carnot/experiment_6469_unique_event_csl_corruption_restart.py`. | Planned: `tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py`. |
| SCENARIO-LEARN-6469-ATTACKS | Planned: `python/carnot/experiment_6469_unique_event_csl_corruption_restart.py`. | Planned: `tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py`. |
| SCENARIO-LEARN-6469-READY | Planned: `python/carnot/experiment_6469_unique_event_csl_corruption_restart.py`. | Planned: `tests/python/test_experiment_6469_unique_event_csl_corruption_restart.py`. |

## REQ-LEARN-6470: Independent Unique-Event CSL Audit

**Given** Exp6457 denied the prior CSL claim and Exp6468 plus Exp6469 update
the raw-output and lifecycle contracts
**When** Exp6470 runs on planning date 20260819
**Then** it SHALL write
`results/experiment_6470_independent_unique_event_csl_audit.json`
**And** it SHALL independently recompute every V556 CSL claim from checked-in
raw files, immutable event rows, and lifecycle rows.

Exp6470 SHALL inventory Exp6457, Exp6468, Exp6469, all referenced raw files,
and lifecycle sidecars before it grants eligibility. Missing, zero-byte,
blocked, malformed, null, or positive evidence SHALL remain visible in the
audit artifact.

Exp6470 SHALL recompute raw hashes, event ids, unit bindings, partition
membership, held disjointness, exposure chronology, exact-veto ordering, write
effects, protected retention, rollback, restart, non-resurrection, duration,
and row aggregates from disk. It SHALL not import Exp6468 or Exp6469 reducer
functions for these claims.

Exp6470 SHALL require exactly one raw path and hash per credited event. Equal
or replayed bytes SHALL count as raw reuse. Raw reuse SHALL remove those events
from credited acquisition counts and SHALL block CSL eligibility.

Exp6470 SHALL set `csl_audit_eligible_score=1.0` only when all raw evidence
exists, every credited event is unique, exact veto precedes every write, held
exposure is zero, effects recompute, lifecycle attacks fail closed, protected
cases do not regress, duration is plausible for the declared substrates, and
critical discrepancies are zero.

Exp6470 SHALL emit these fields:

- `status`
- `upstream_artifact_inventory`
- `raw_file_inventory_and_hashes`
- `independent_event_identity_recomputation`
- `independent_exposure_ledger`
- `exact_veto_order_recomputation`
- `per_unit_rows`
- `audit_rows`
- `independent_effect_recomputation`
- `protected_case_recomputation`
- `rollback_restart_and_non_resurrection_replay`
- `duration_recomputation`
- `upstream_vs_independent_field_comparison`
- `aggregate_row_recomputation`
- `attack_matrix`
- `current_adversarial_findings`
- `critical_discrepancies`
- `csl_audit_eligible_score`
- `protected_files_unchanged`
- `blocked_reason`
- `gate_check_summary`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field and every
`csl_audit_eligible_score` condition. `verifier_is_oracle` SHALL be true only
for independent exact-checker, hash, chronology, and arithmetic recomputation.
Upstream summaries, model raw text, learned weights, and claimed gates SHALL
NOT be oracles.

## SCENARIO-LEARN-6470-INVENTORY: Evidence Is Frozen Before Audit

**Given** Exp6457, Exp6468, and Exp6469 artifacts cite raw or lifecycle files
**When** Exp6470 inventories evidence
**Then** each path SHALL report presence, byte length, SHA-256, and malformed
or zero-byte status before any eligibility score is computed.

## SCENARIO-LEARN-6470-IDENTITY: One Credited Event Has One Raw

**Given** V556 event and raw-output rows
**When** Exp6470 recomputes event identity from disk
**Then** each credited event SHALL bind one event id, one unit id, one raw
path, and one raw hash. Duplicate ids, duplicate hashes, missing paths, or
path/hash mismatches SHALL create audit rows and block eligibility.

## SCENARIO-LEARN-6470-CHRONOLOGY: Held Evidence Stays Sealed

**Given** Exp6468 and Exp6469 exposure ledgers
**When** Exp6470 replays chronology
**Then** future-held outcome exposure and held contamination counters SHALL be
zero before generation, parsing, update admission, and restart.

## SCENARIO-LEARN-6470-VETO: Exact Veto Precedes Writes

**Given** per-unit rows with checker and write receipts
**When** Exp6470 recomputes write ordering
**Then** every admitted write SHALL have a successful exact-checker receipt
that ran before the write. A failed or absent checker receipt SHALL leave the
post-state head equal to the pre-state head.

## SCENARIO-LEARN-6470-EFFECTS: Rows Own Effects And Retention

**Given** V556 per-unit rows
**When** Exp6470 recomputes effects, protected retention, and aggregate rows
**Then** independent values SHALL match upstream aggregate fields exactly or
the mismatch SHALL appear in `upstream_vs_independent_field_comparison`.

## SCENARIO-LEARN-6470-LIFECYCLE: Corruption Cannot Resurrect

**Given** Exp6469 lifecycle rows and tombstone receipts
**When** Exp6470 replays corrupt feedback, exact-veto bypass, wrong binding,
rollback, restart, and non-resurrection attacks
**Then** each corrupt event SHALL be quarantined, tombstoned, rolled back
before release, and absent from active heads after restart.

## SCENARIO-LEARN-6470-READY: Eligibility Is Conjunctive

**Given** raw evidence, unique event identity, zero held exposure, exact-veto
ordering, recomputed effects, protected retention, lifecycle safety, duration,
and attack replay all pass
**When** Exp6470 computes its final gate
**Then** `csl_audit_eligible_score` SHALL be `1.0`; otherwise it SHALL be
`0.0` with `critical_discrepancies` and `gate_check_summary` populated.

## Implementation Status (REQ-LEARN-6470)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6470 | Planned: `python/carnot/experiment_6470_independent_unique_event_csl_audit.py`; terminal artifact `results/experiment_6470_independent_unique_event_csl_audit.json`. | Planned: `tests/python/test_experiment_6470_independent_unique_event_csl_audit.py`. |
| SCENARIO-LEARN-6470-INVENTORY | Planned: `python/carnot/experiment_6470_independent_unique_event_csl_audit.py`. | Planned: `tests/python/test_experiment_6470_independent_unique_event_csl_audit.py`. |
| SCENARIO-LEARN-6470-IDENTITY | Planned: `python/carnot/experiment_6470_independent_unique_event_csl_audit.py`. | Planned: `tests/python/test_experiment_6470_independent_unique_event_csl_audit.py`. |
| SCENARIO-LEARN-6470-CHRONOLOGY | Planned: `python/carnot/experiment_6470_independent_unique_event_csl_audit.py`. | Planned: `tests/python/test_experiment_6470_independent_unique_event_csl_audit.py`. |
| SCENARIO-LEARN-6470-VETO | Planned: `python/carnot/experiment_6470_independent_unique_event_csl_audit.py`. | Planned: `tests/python/test_experiment_6470_independent_unique_event_csl_audit.py`. |
| SCENARIO-LEARN-6470-EFFECTS | Planned: `python/carnot/experiment_6470_independent_unique_event_csl_audit.py`. | Planned: `tests/python/test_experiment_6470_independent_unique_event_csl_audit.py`. |
| SCENARIO-LEARN-6470-LIFECYCLE | Planned: `python/carnot/experiment_6470_independent_unique_event_csl_audit.py`. | Planned: `tests/python/test_experiment_6470_independent_unique_event_csl_audit.py`. |
| SCENARIO-LEARN-6470-READY | Planned: `python/carnot/experiment_6470_independent_unique_event_csl_audit.py`. | Planned: `tests/python/test_experiment_6470_independent_unique_event_csl_audit.py`. |

## REQ-LEARN-6444: CSL Lifecycle Recomputation Audit

**Given** Exp6433 left the prospective CSL claim ineligible, Exp6441 through
Exp6443 are the expected V554 upstream tasks, and blocked or missing upstream
evidence must stay visible
**When** Exp6444 runs on planning date 20260815
**Then** it SHALL write
`results/experiment_6444_csl_lifecycle_recomputation_audit.json`
**And** it SHALL independently recompute development, lifecycle-safety, and
held CSL metrics from immutable per-unit rows without importing upstream
aggregate, uncertainty, gating, or verdict functions.

Exp6444 SHALL inventory Exp6441, Exp6442, and Exp6443 artifact paths and every
referenced row path before importing any experiment module. Missing, zero-byte,
malformed, blocked, skipped, flagged, null, and underpowered inputs SHALL remain
visible audit evidence.

Exp6444 SHALL freeze upstream paths, sizes, hashes, statuses, honest verdicts,
readiness fields, adversarial findings, row counts, and gate summaries. It SHALL
not mutate upstream artifacts.

Exp6444 SHALL derive proposal coverage, admission precision, development and
held future exact yield, paired deltas, uncertainty, contamination, protected
retention, forgetting, memory growth, restart recovery, lifecycle unsafe
authoring, unsafe retrieval, fresh-session harm, benign utility, quarantine
precision and recall, rollback success, protected releases, resurrection, and
online cost from documented row schemas.

Exp6444 SHALL verify raw-output uniqueness, cross-task raw-output intersections,
development-held hash disjointness, sealed future timing, event chronology,
capacity matching, transaction ancestry, memory-head recovery, process
boundaries, command-path continuity, exact-veto preservation, duration rules,
and substrate rules.

Exp6444 SHALL replay each critical attack with independent code. Critical
attacks include raw-output reuse, row deletion, duplicate event, event reorder,
future leakage, same-step write, stale head, authority spoof, supersession
bypass, rollback omission, cache resurrection, restart corruption, exact-veto
override, unsafe authoring, unsafe retrieval, protected release, and
resurrection.

Exp6444 SHALL emit one per-unit audit row or stable row reference for every
audited upstream row and every reported-vs-recomputed comparison. Each
comparison SHALL include upstream value, recomputed value, absolute delta,
tolerance, mismatch state, inclusion decision, and evidence path.

Exp6444 SHALL set `prospective_csl_eligibility=true` only when development and
held exact effects are positive, no safety regression exists, protected release
is zero, growth is bounded, timing and substrate are eligible, no critical
attack is open, every required upstream evidence item exists and passes, and no
material row mismatch exists. Otherwise it SHALL use a terminal
`complete_null:` or `complete_blocked:` verdict and name the failed evidence.

Exp6444 SHALL emit these fields:

- `status`
- `upstream_inventory_and_hashes`
- `upstream_status_verdict_readiness_and_adversarial_findings`
- `independent_reducer_source_and_test_hashes`
- `per_unit_rows`
- `development_metric_recomputation`
- `held_metric_recomputation`
- `lifecycle_safety_metric_recomputation`
- `upstream_vs_recomputed_mismatches`
- `mismatch_count_and_materiality`
- `raw_output_uniqueness_and_cross_task_intersections`
- `chronology_future_seal_and_capacity_checks`
- `memory_head_transaction_and_restart_checks`
- `command_path_chain_checks`
- `exact_veto_checks`
- `independent_attack_replay`
- `duration_and_substrate_eligibility`
- `prospective_csl_eligibility`
- `csl_ineligibility_reasons`
- `csl_audit_ready_score`
- `current_adversarial_findings`
- `protected_files_unchanged`
- `blocked_reason`
- `gate_check_summary`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every required field. It SHALL also map each
`csl_audit_ready_score` condition. `verifier_is_oracle` SHALL be false for the
mixed audit as a whole. Deterministic exact checkers and row arithmetic SHALL be
identified inside evidence records.

## SCENARIO-LEARN-6444-INVENTORY: Missing V554 Evidence Blocks Readiness

**Given** Exp6441 through Exp6443 are required upstream tasks
**When** Exp6444 inventories their artifacts and rows
**Then** missing Exp6441 or Exp6443 evidence and blocked Exp6442 evidence SHALL
be reported as audit inputs
**And** `csl_audit_ready_score` SHALL remain `0.0`.

## SCENARIO-LEARN-6444-REDUCERS: Rows Drive Development And Held Metrics

**Given** immutable Exp6430, Exp6431, and Exp6432 per-unit rows
**When** Exp6444 reduces them
**Then** development, lifecycle-safety, and held metrics SHALL be recomputed
without importing upstream aggregate or readiness functions.

## SCENARIO-LEARN-6444-CHAINS: Paths And Chronology Are Rechecked

**Given** raw-output, event, memory-head, transaction, restart, and receipt
evidence
**When** Exp6444 audits command paths
**Then** uniqueness, chronology, future seals, capacity bounds, transaction
ancestry, process boundaries, and generation-to-verdict chains SHALL be visible.

## SCENARIO-LEARN-6444-ATTACKS: Critical Attacks Block Eligibility

**Given** critical lifecycle and CSL attacks
**When** Exp6444 replays each attack independently
**Then** any open critical attack, current critical adversarial flag, missing
required evidence, or blocked upstream gate SHALL force
`prospective_csl_eligibility=false`.

## SCENARIO-LEARN-6444-DELIVERABLE: Artifact Is Terminal And Self-Checking

**Given** Exp6444 writes its terminal artifact
**When** validation runs
**Then** every required field, field principle, provenance entry, gate summary,
checksum, and terminal-prefix verdict SHALL validate.

## Implementation Status (REQ-LEARN-6444)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6444 | Implemented: `python/carnot/experiment_6444_csl_lifecycle_recomputation_audit.py`; terminal artifact `results/experiment_6444_csl_lifecycle_recomputation_audit.json`. | Implemented: `tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py`. |
| SCENARIO-LEARN-6444-INVENTORY | Implemented: `python/carnot/experiment_6444_csl_lifecycle_recomputation_audit.py`. | Implemented: `tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py`. |
| SCENARIO-LEARN-6444-REDUCERS | Implemented: `python/carnot/experiment_6444_csl_lifecycle_recomputation_audit.py`. | Implemented: `tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py`. |
| SCENARIO-LEARN-6444-CHAINS | Implemented: `python/carnot/experiment_6444_csl_lifecycle_recomputation_audit.py`. | Implemented: `tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py`. |
| SCENARIO-LEARN-6444-ATTACKS | Implemented: `python/carnot/experiment_6444_csl_lifecycle_recomputation_audit.py`. | Implemented: `tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py`. |
| SCENARIO-LEARN-6444-DELIVERABLE | Implemented: `python/carnot/experiment_6444_csl_lifecycle_recomputation_audit.py`. | Implemented: `tests/python/test_experiment_6444_csl_lifecycle_recomputation_audit.py`. |

## REQ-LEARN-6409: Graph-Local Multisession Continuous Learning

**Given** Exp6408 first shows positive future exact yield with non-increased
contamination, Exp6407 defines a raw and compiled memory protocol, and Exp6383
defines selective rollback
**When** Exp6409 runs on planning date 20260813
**Then** it SHALL write
`results/experiment_6409_graph_local_multisession_continuous_learning.json`
**And** it SHALL compare frozen, flat predecessor-bound transactional, and
graph-local two-tier learners across multiple sessions and drift regimes.

Exp6409 SHALL revalidate Exp6408 gates, licenses, frozen harnesses, raw and
compiled schemas, factor-head hashes, exact checkers, rollback receipt, model
files, GPU offload, and protected partitions before any event is sealed. It
SHALL invoke only Exp6395 licensed cells. Every other cell SHALL abstain.

Exp6409 SHALL use the three mandated local GGUF model ids from
`cached_sota_pair()`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token counts SHALL use embedded GGUF
tokenizers only. Exp6409 SHALL not call `AutoTokenizer`.

Exp6409 SHALL seal at least 72 chronological events across four sessions,
three declared drift regimes, at least six update opportunities, four process
restarts, two license-expiry boundaries, and two source-supersession
boundaries. Events SHALL be balanced across licensed cells. Protected future
events SHALL open once after heads are frozen.

Exp6409 SHALL match event order, LLM calls, token budgets, exact checks, and
consumer work for all three arms. Candidate proposals SHALL evaluate off-commit
and bind raw event hashes, source spans, model, harness, license, exact support,
predecessor head, affected neighborhood, diagnostics, expiry, and supersession.

Exp6409 SHALL atomically record Commit, Reject, Quarantine, or Defer for each
proposal. Commits SHALL pass exact support, local/full replay equivalence,
protected retention, unique effects, predecessor freshness, and license
validity. Raw escalation SHALL trigger on implicit support, graph/raw
disagreement, checker drift, stale cache, unresolved supersession, and missing
provenance.

Exp6409 SHALL inject contamination, stale heads, duplicate effects, concurrent
proposals, interrupted writes, expired licenses, superseded evidence, cache
resurrection, model-row swaps, and restart corruption. Every attack SHALL fail
closed. Selective rollback SHALL remove harmful descendants on affected
neighborhoods only.

Exp6409 SHALL report prequential exact yield, forward transfer, backward
retention, forgetting, negative transfer, contamination propagation, factor
growth, raw escalation, verification cost, restart recovery, and local-vs-full
replay work. It SHALL emit `delta_future_exact_yield_over_frozen`,
`contamination_propagation_rate`, and `forgetting_delta` as finite bare numbers.

Exp6409 SHALL set `graph_local_multisession_csl_ready_score=1.0` only when at
least two sessions commit, graph-local future exact yield beats frozen,
contamination propagation is zero, no harmful retention regression survives
rollback, growth is bounded, local/full replay decisions agree, every attack
fails closed, protected leakage is zero, tests pass, and no model weights
change.

Exp6409 SHALL emit these fields:

- `status`
- `exp6408_gate_receipts`
- `MODEL_SPECS`
- `models_used`
- `cached_sota_pair_receipts`
- `embedded_gguf_tokenizer_receipts`
- `autotokenizer_usage_count`
- `license_and_harness_bindings`
- `unlicensed_cell_abstention_records`
- `cuda_offload_runtime_peak_memory_and_duration_receipts_by_model`
- `chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_balance_and_partition_seals`
- `preregistered_frozen_flat_and_graph_local_arm_contract`
- `matched_work_receipts`
- `initial_raw_ledger_graph_and_factor_head_hashes`
- `typed_candidate_and_raw_evidence_records`
- `predecessor_license_checker_neighborhood_expiry_and_supersession_bindings`
- `atomic_disposition_records`
- `factor_head_and_graph_transition_history`
- `commit_reject_quarantine_and_defer_counts_by_session`
- `raw_escalation_trigger_accuracy_and_cost_results`
- `local_vs_full_replay_decision_and_work_results`
- `stale_duplicate_concurrency_interrupt_expiry_supersession_cache_model_and_restart_attack_matrix`
- `prequential_exact_yield_by_arm_and_session`
- `forward_transfer_results`
- `backward_retention_forgetting_and_negative_transfer_results`
- `contamination_propagation_rate`
- `factor_growth_and_capacity_results`
- `restart_recovery_results`
- `selective_rollback_results`
- `untouched_future_evaluation_receipts`
- `delta_future_exact_yield_over_frozen`
- `forgetting_delta`
- `graph_local_multisession_csl_ready_score`
- `protected_leakage_count`
- `same_step_write_count`
- `model_weight_change_count`
- `universal_support_claimed`
- `public_factor_claim_eligibility`
- `harm_underpowered_missing_and_flagged_cells`
- `protected_files_unchanged`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`
- `honest_verdict`

`field_principles` SHALL map every Exp6408 gate,
`delta_future_exact_yield_over_frozen`, `contamination_propagation_rate`,
`forgetting_delta`, and `graph_local_multisession_csl_ready_score` to their
purposes. `verifier_is_oracle` SHALL be true only for exact task checkers and
deterministic replay or retention tests.

## SCENARIO-LEARN-6409-MULTISESSION: Manifest Covers Sessions And Drift

**Given** Exp6408 has a positive gate and licensed cells
**When** Exp6409 seals its chronological manifest
**Then** the manifest SHALL contain at least 72 events, four sessions, three
drift regimes, six update opportunities, four restarts, two expiry boundaries,
and two supersession boundaries
**And** licensed cells SHALL be balanced.

## SCENARIO-LEARN-6409-GRAPH-COMMIT: Commits Are Graph-Local And Provenance-Bound

**Given** a graph-local proposal with raw evidence
**When** Exp6409 evaluates it off-commit
**Then** the proposal SHALL bind predecessor, license, checker, affected
neighborhood, expiry, supersession, diagnostics, and raw hashes
**And** Commit SHALL occur only when exact support and local/full replay agree.

## SCENARIO-LEARN-6409-ESCALATION: Raw Tier Resolves Ambiguity

**Given** implicit support, graph/raw disagreement, checker drift, stale cache,
unresolved supersession, or missing provenance
**When** Exp6409 evaluates the proposal
**Then** it SHALL escalate to raw evidence
**And** the compiled graph SHALL not authorize the commit.

## SCENARIO-LEARN-6409-ATTACKS: Multi-Session Attacks Fail Closed

**Given** contamination, stale head, duplicate effect, concurrent proposal,
interrupted write, expired license, superseded evidence, cache resurrection,
model-row swap, or restart corruption
**When** Exp6409 evaluates the attack matrix
**Then** every attack SHALL fail closed
**And** harmful descendants SHALL be removed by selective rollback.

## SCENARIO-LEARN-6409-READY: Readiness Requires Transfer Without Contamination

**Given** at least two sessions commit and all tests pass
**When** graph-local future exact yield beats frozen, contamination propagation
is zero, forgetting does not increase, growth is bounded, replay decisions
agree, attacks fail closed, and protected files stay unchanged
**Then** `graph_local_multisession_csl_ready_score` SHALL be `1.0`.

## Implementation Status (REQ-LEARN-6409)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6409 | Implemented: `python/carnot/experiment_6409_graph_local_multisession_continuous_learning.py`; terminal artifact `results/experiment_6409_graph_local_multisession_continuous_learning.json`. | Implemented: `tests/python/test_experiment_6409_graph_local_multisession_continuous_learning.py`. |
| SCENARIO-LEARN-6409-MULTISESSION | Implemented: `python/carnot/experiment_6409_graph_local_multisession_continuous_learning.py`. | Implemented: `tests/python/test_experiment_6409_graph_local_multisession_continuous_learning.py`. |
| SCENARIO-LEARN-6409-GRAPH-COMMIT | Implemented: `python/carnot/experiment_6409_graph_local_multisession_continuous_learning.py`. | Implemented: `tests/python/test_experiment_6409_graph_local_multisession_continuous_learning.py`. |
| SCENARIO-LEARN-6409-ESCALATION | Implemented: `python/carnot/experiment_6409_graph_local_multisession_continuous_learning.py`. | Implemented: `tests/python/test_experiment_6409_graph_local_multisession_continuous_learning.py`. |
| SCENARIO-LEARN-6409-ATTACKS | Implemented: `python/carnot/experiment_6409_graph_local_multisession_continuous_learning.py`. | Implemented: `tests/python/test_experiment_6409_graph_local_multisession_continuous_learning.py`. |
| SCENARIO-LEARN-6409-READY | Implemented: `python/carnot/experiment_6409_graph_local_multisession_continuous_learning.py`. | Implemented: `tests/python/test_experiment_6409_graph_local_multisession_continuous_learning.py`. |

## REQ-LEARN-6383: Dependency-Guided Factor Rollback Stress

**Given** versioned factor release and whole-version rollback already exist
**When** Exp6383 runs on planning date 20260813
**Then** it SHALL write
`results/experiment_6383_dependency_guided_factor_rollback_stress.json`
**And** it SHALL compare selective descendant rollback, full registry reset,
and no rollback on the same frozen exact fixtures, injection order, replay
work, and initial lineage graph.

Exp6383 SHALL define typed dependency nodes for source events, obligations,
exact evidence, factor versions, factors, consumer decisions, and rollback
actions. It SHALL define allowed edge types and acyclicity rules. Cycles,
missing evidence, corrupted lineage, edge tampering, orphan nodes, incomplete
invalidation, root mismatch, and interrupted journals SHALL fail closed.

Exp6383 SHALL build deterministic clean, stale, poisoned, duplicated,
misattributed, partially supported, shared-support, cyclic, and
missing-evidence fixtures. It SHALL diagnose the bad source node, invalidate
only unsupported descendants, and preserve state with an independent
exact-valid support path. Consumer decisions that depend on unsupported active
factors SHALL roll back even when they also cite valid factors.

Exp6383 SHALL set `dependency_guided_rollback_ready_score=1.0` only when
selective rollback removes every harmful descendant, preserves every
independently supported node, beats full reset on preservation, leaves zero
unsafe survivors, and all recorded tests pass. It SHALL not invoke an LLM. It
SHALL not claim live learning utility.

Exp6383 SHALL emit these fields with explicit principles:

- `status`: Terminal status follows rollback safety, preservation, corruption checks, protected files, and tests.
- `upstream_learning_context_class`: Exp6382 absence, blocked state, or terminal class is context only, not a readiness gate.
- `registry_release_ledger_and_checker_hashes`: Factor registry, release ledger, exact checker sources, and Exp6382 when present are hashed before fixture replay.
- `typed_dependency_schema_path_hash_and_version`: The node, edge, and acyclicity schema is frozen as a sidecar.
- `allowed_node_and_edge_types`: Typed nodes and allowed edge pairs define the only legal lineage surface.
- `preregistered_injection_and_arm_contract`: Bad-source injection order and the selective, full reset, and no-rollback controls are fixed before replay.
- `deterministic_fixture_manifest`: Clean, stale, poisoned, duplicated, misattributed, partially supported, shared-support, cyclic, and missing-evidence fixtures are named and seeded.
- `lineage_graphs_before_and_after_injection`: Graph roots, node counts, edge counts, and state roots are recorded before and after injection.
- `diagnosis_receipts`: The diagnosed bad source and exact replay evidence explain the invalidation frontier.
- `selective_full_reset_and_no_rollback_results`: All three controls report the same metrics on the same replay work.
- `harmful_descendants_removed`: Selective rollback must remove all unsupported harmful descendants.
- `independently_supported_state_preserved`: Exact-valid independent support paths must survive selective rollback.
- `overrollback_underrollback_and_unsafe_survivor_counts`: Over-removal, missed rollback, and unsafe survivors stay visible.
- `exact_replay_cost_latency_and_memory`: Checker calls, deterministic cost, latency, and graph memory bytes are measured.
- `cycle_missing_edge_corruption_and_interruption_results`: Cycles, missing evidence, corruption, incomplete invalidation, and interruption fail closed.
- `journal_restart_and_idempotence_receipts`: Restart, double rollback, root mismatch, edge tampering, orphan nodes, and active decision rollback are recorded.
- `terminal_registry_roots`: Terminal roots prove selective rollback is stable and exact-valid.
- `dependency_guided_rollback_ready_score`: Readiness is a conjunctive safety and preservation gate.
- `no_live_utility_claim`: Bare true states that this stress test does not promote live learning utility.
- `protected_files_unchanged`: Conductor, ops, traceability, prior factor code, and upstream artifacts remain byte-identical.
- `preconditions_checked`: Date, source hashes, protected hashes, schema, fixtures, controls, seeds, and upstream context freeze before replay.
- `inference_substrate`: The substrate declares deterministic exact replay and typed lineage analysis with no LLM.
- `verifier_is_oracle`: Bare true applies only to deterministic exact replay checkers, not lineage or rollback policy.
- `field_principles`: Every required field states its guard.
- `field_provenance`: Every required field maps to specs, source hashes, fixtures, exact checks, rollback receipts, tests, or roots.
- `random_seed`: Fixed seed pins fixture order.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states that no live utility was claimed.

## SCENARIO-LEARN-6383-SCHEMA: Typed Lineage Fails Closed

**Given** a graph with an illegal edge, cycle, missing evidence, orphan node,
or corrupted lineage hash
**When** Exp6383 validates it
**Then** the graph SHALL reject before rollback state can be promoted.

## SCENARIO-LEARN-6383-SELECTIVE: Unsupported Descendants Roll Back

**Given** a bad source has stale, poisoned, and partially supported
descendants
**When** selective rollback runs
**Then** every harmful factor version, factor, and consumer decision SHALL be
inactive
**And** nodes with independent exact-valid support SHALL remain active.

## SCENARIO-LEARN-6383-CONTROLS: Control Arms Share Replay Work

**Given** selective rollback, full reset, and no rollback controls
**When** the arms run
**Then** they SHALL share the same initial graph, injection order, exact
checks, and replay work
**And** selective rollback SHALL preserve more valid state than full reset
while no rollback leaves unsafe survivors.

## SCENARIO-LEARN-6383-JOURNAL: Restart And Idempotence Are Exact

**Given** a rollback journal is interrupted, replayed, replayed again, or
started from the wrong root
**When** Exp6383 restarts it
**Then** the valid journal SHALL converge to one exact-valid terminal root
**And** root mismatch SHALL fail closed.

## SCENARIO-LEARN-6383-READY: Readiness Requires Zero Unsafe Survivors

**Given** any harmful descendant survives, independently supported state is
lost, protected files change, tests fail, or live utility is claimed
**When** readiness is computed
**Then** `dependency_guided_rollback_ready_score` SHALL be `0.0`.

## Implementation Status (REQ-LEARN-6383)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6383 | Planned: `python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py`; terminal artifact `results/experiment_6383_dependency_guided_factor_rollback_stress.json`. | Planned: `tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py`. |
| SCENARIO-LEARN-6383-SCHEMA | Planned: `python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py`. | Planned: `tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py`. |
| SCENARIO-LEARN-6383-SELECTIVE | Planned: `python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py`. | Planned: `tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py`. |
| SCENARIO-LEARN-6383-CONTROLS | Planned: `python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py`. | Planned: `tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py`. |
| SCENARIO-LEARN-6383-JOURNAL | Planned: `python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py`. | Planned: `tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py`. |
| SCENARIO-LEARN-6383-READY | Planned: `python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py`. | Planned: `tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py`. |

## REQ-LEARN-6385: Live Factor Learning And Rollback Safety Audit

**Given** V549 factor transport, proposal search, chronological learning,
selective rollback, and default-off consumer artifacts may be positive, null,
blocked, flagged, malformed, or absent
**When** Exp6385 runs on planning date 20260813
**Then** it SHALL write
`results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json`
**And** it SHALL hash all present upstream artifacts, sidecars, source files,
exact checker files, protected manifests, and exclusion state before semantic
reads
**And** it SHALL classify absent, blocked, null, clean, and flagged inputs
before outcome-sensitive reads.

Exp6385 SHALL freeze audit registration and an attack manifest before reading
readiness or utility fields. The manifest SHALL cover transport attacks,
proposal-frontier attacks, chronological-learning attacks, dependency-rollback
attacks, and consumer attacks. Missing, blocked, null, flagged, malformed, and
underpowered cells SHALL remain visible. They SHALL not be relabeled clean.

Exp6385 SHALL not rerun upstream experiments. It SHALL replay applicable
attacks against immutable copies only. It SHALL report detected, fail-closed,
unsafe-survivor, false-alarm, and inapplicable counts by upstream terminal
class. It SHALL not create correctness labels. It SHALL state verifier-oracle
scope only for immutable exact-checker replay outputs.

Exp6385 SHALL recompute readiness fields from primary bytes. Safety success
SHALL not promote utility. `utility_promotion_count` SHALL remain zero unless a
separate clean utility artifact qualifies. Clean attacks, blocked tasks, null
tasks, absent artifacts, and protected safety evidence SHALL not count as
utility evidence.

Exp6385 SHALL set `factor_learning_rollback_safety_ready_score=1.0` only when
every applicable harmful attack fails closed, protected validation leaks are
zero, source model weight mutations are zero, registry writes during consumer
evaluation are zero, unsafe commits are zero, protected files are unchanged,
tests pass, and no missing or blocked input is relabeled clean.

Exp6385 SHALL emit these fields with explicit principles:

- `status`: Terminal status follows safety audit gates, not utility promotion.
- `upstream_artifact_and_sidecar_hashes`: Upstream artifacts, sidecars, source files, exact checkers, protected manifests, and exclusion state are frozen before semantic reads.
- `upstream_terminal_classification`: Absent, blocked, null, clean, flagged, and malformed evidence classes are explicit.
- `audit_registration_path_hash_and_preoutcome_receipt`: Registration proves read order, copies, hashes, seeds, commands, and pre-read guards.
- `attack_manifest_path_hash`: The attack manifest is hash-bound before outcome-sensitive reads.
- `transport_attack_results`: Transport attacks cover process substitution, schema drift, capacity undercount, thinking prefixes, repeated tokens, truncation laundering, parser retry, post-hoc repair, source substitution, and exact-check bypass.
- `proposal_frontier_attack_results`: Proposal attacks cover residual mutation, incumbent laundering, optional-stopping reset, family shortcuts, and unequal work.
- `chronological_learning_attack_results`: Chronology attacks cover same-step writes, future leakage, duplicate evidence, and event reorder.
- `dependency_rollback_attack_results`: Rollback attacks cover false lineage, missing edges, cycles, shared-support deletion, incomplete invalidation, journal interruption, root mismatch, and stale consumer decisions.
- `consumer_attack_results`: Consumer attacks cover registry writes, version swaps, quarantine bypass, capacity overflow, weight changes, and unsafe feature enablement.
- `detected_fail_closed_unsafe_survivor_false_alarm_and_inapplicable_counts`: Counts are grouped by upstream terminal class.
- `readiness_recomputation`: Readiness fields are recomputed from primary data and separated from safety readiness.
- `protected_validation_leak_count`: Bare zero proves protected validation data did not leak.
- `source_model_weight_mutation_count`: Bare zero proves source weights did not change.
- `registry_write_during_consumer_count`: Bare zero proves consumer evaluation stayed read-only.
- `unsafe_commit_count`: Bare zero proves attacked state did not commit.
- `utility_promotion_count`: Bare zero proves safety did not become utility evidence.
- `factor_learning_rollback_safety_ready_score`: Readiness is conjunctive over attack closure, zero counters, protected files, tests, and class preservation.
- `harm_underpowered_missing_and_flagged_cells`: Harm, missing, underpowered, blocked, and flagged cells stay visible.
- `protected_files_unchanged`: Protected repo files and upstream artifacts stay byte-identical.
- `preconditions_checked`: Preconditions bind date, hashes, copies, terminal classes, exclusions, protected files, seeds, and commands.
- `inference_substrate`: The substrate declares deterministic artifact audit with no new upstream run.
- `verifier_is_oracle`: Oracle scope is limited to immutable exact-checker replay outputs.
- `field_principles`: Every required field states its guard.
- `field_provenance`: Every required field maps to specs, inputs, attacks, checks, tests, or hashes.
- `random_seed`: Fixed seed pins manifest order.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict starts with a terminal prefix and states that safety does not promote utility.

## SCENARIO-LEARN-6385-REGISTRATION: Hashes Freeze Before Semantic Reads

**Given** V549 upstream artifact paths and sidecars
**When** Exp6385 starts
**Then** it SHALL write registration and attack-manifest sidecars before
reading readiness, utility, harm, or consumer fields.

## SCENARIO-LEARN-6385-ATTACKS: Applicable Attacks Fail Closed

**Given** the frozen attack manifest
**When** Exp6385 replays transport, proposal, chronology, rollback, and
consumer attacks against immutable copies
**Then** every applicable harmful attack SHALL detect and fail closed
**And** unsafe survivors, unsafe commits, false alarms, protected leaks,
registry writes, and model-weight mutations SHALL remain zero.

## SCENARIO-LEARN-6385-TERMINAL-CLASSES: Blocked And Missing Stay Visible

**Given** a blocked, null, flagged, malformed, or absent upstream artifact
**When** readiness is recomputed
**Then** the class SHALL remain visible in
`upstream_terminal_classification`
**And** the artifact SHALL not relabel it as clean.

## SCENARIO-LEARN-6385-UTILITY-BOUNDARY: Safety Does Not Promote Utility

**Given** safety attacks all fail closed but utility artifacts are null,
blocked, or absent
**When** Exp6385 computes `readiness_recomputation`
**Then** utility promotion SHALL remain zero
**And** clean safety results SHALL not become utility evidence.

## SCENARIO-LEARN-6385-READY: Readiness Is Conjunctive

**Given** an unsafe survivor, protected leak, consumer registry write, source
weight mutation, unsafe commit, protected-file mutation, failed test, or
relabeling of missing or blocked evidence as clean
**When** Exp6385 refreshes terminal fields
**Then** `factor_learning_rollback_safety_ready_score` SHALL be `0.0`.

## Implementation Status (REQ-LEARN-6385)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6385 | Planned: `python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py`; terminal artifact `results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json`. | Planned: `tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. |
| SCENARIO-LEARN-6385-REGISTRATION | Planned: `python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. | Planned: `tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. |
| SCENARIO-LEARN-6385-ATTACKS | Planned: `python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. | Planned: `tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. |
| SCENARIO-LEARN-6385-TERMINAL-CLASSES | Planned: `python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. | Planned: `tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. |
| SCENARIO-LEARN-6385-UTILITY-BOUNDARY | Planned: `python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. | Planned: `tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. |
| SCENARIO-LEARN-6385-READY | Planned: `python/carnot/experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. | Planned: `tests/python/test_experiment_6385_live_factor_learning_and_rollback_safety_audit.py`. |

## REQ-LEARN-6479: Verify-Repair Factor Cache Shadow Adapter

**Given** V556 unique-event verifier-bounded factor learning evidence
**When** FR-11 learning is added to the production verify-repair surface
**Then** Carnot SHALL provide a default-off factor-cache shadow adapter for
`VerifyRepairPipeline`
**And** the adapter SHALL preserve disabled behavior exactly.

The adapter SHALL require a unique event id, raw hash, unit binding, checker
hash, exact outcome, and chronological index before it admits any cache write.
It SHALL reject self-signed receipts, duplicate event ids, duplicate raw
events, wrong unit bindings, forged exact outcomes, stale cache heads, and
write-before-check attempts.

The adapter SHALL persist cache, tombstone, quarantine, and rollback state
through atomic checkpoint writes. A tombstoned factor or event SHALL NOT
resurrect after `load()` or `close()`.

Exp6479 SHALL write
`results/experiment_6479_verify_repair_factor_cache_shadow_adapter.json`.
The artifact SHALL report baseline import and output receipts, default-off
compatibility rows, shadow decision rows, exact write-admission rows,
persistence, rollback, tombstone receipts, attacks, protected file hashes,
test receipts, and a conjunctive `factor_cache_shadow_adapter_ready_score`.
The artifact SHALL declare
`inference_substrate="deterministic_pipeline_integration_no_llm"`, and
`scripts/adversarial_verify.py` SHALL classify that substrate with a nonzero
deterministic duration floor instead of treating it as live model inference or
an unknown substrate.

### SCENARIO-LEARN-6479-EXACT-ADMIT: Exact Checker Owns Writes

**Given** a proposed factor-cache write
**When** the receipt is missing prior exact validation, has a forged pass, uses
the wrong unit, replays a raw event, or reuses an event id
**Then** the adapter SHALL abstain or quarantine the proposal
**And** no cache write SHALL be admitted.

### SCENARIO-LEARN-6479-RESTART: Tombstones Do Not Resurrect

**Given** an admitted factor is tombstoned and rolled back
**When** the adapter saves, closes, and loads from disk
**Then** the tombstone and rollback state SHALL persist
**And** the tombstoned event or factor SHALL remain absent from active cache
state.

### SCENARIO-LEARN-6479-ARTIFACT: Exp6479 Gates Are Conjunctive

**Given** default-off compatibility, exact write admission, lifecycle
persistence, attack closure, protected files, and tests
**When** Exp6479 computes readiness
**Then** `factor_cache_shadow_adapter_ready_score` SHALL be `1.0` only when all
gates pass.

## Implementation Status (REQ-LEARN-6479)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-LEARN-6479 | Planned: `python/carnot/pipeline/factor_cache_shadow_adapter.py`; `python/carnot/experiment_6479_verify_repair_factor_cache_shadow_adapter.py`. | Planned: `tests/python/test_factor_cache_shadow_adapter.py`; `tests/python/test_experiment_6479_verify_repair_factor_cache_shadow_adapter.py`. |
| SCENARIO-LEARN-6479-EXACT-ADMIT | Planned: `python/carnot/pipeline/factor_cache_shadow_adapter.py`. | Planned: `tests/python/test_factor_cache_shadow_adapter.py`. |
| SCENARIO-LEARN-6479-RESTART | Planned: `python/carnot/pipeline/factor_cache_shadow_adapter.py`. | Planned: `tests/python/test_factor_cache_shadow_adapter.py`. |
| SCENARIO-LEARN-6479-ARTIFACT | Planned: `python/carnot/experiment_6479_verify_repair_factor_cache_shadow_adapter.py`. | Planned: `tests/python/test_experiment_6479_verify_repair_factor_cache_shadow_adapter.py`. |

## REQ-CL-6553: Prospective SOTA Chronological Continuous Self-Learning

Given Exp6552 supplies a reversible exact-conflict memory controller and
Exp6548 supplies a clean V567 external-evidence gate
When Exp6553 runs for planning date 20260823
Then it SHALL write
`results/experiment_6553_prospective_sota_continuous_self_learning.json`
as one terminal artifact
And it SHALL evaluate both structured gates before model runtime preconditions
can support a completed live comparison.

Exp6553 SHALL declare `MODEL_SPECS` with exactly
`unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL resolve local `.gguf` files with
llama.cpp-compatible paths only and SHALL NOT call Hugging Face
`AutoTokenizer.from_pretrained()` on a GGUF repository id. It SHALL first
evaluate dual RTX 3090 availability, driver and VRAM receipts, llama.cpp
binary and CUDA support, required GGUF hashes, writable checkpoint space, Z3,
input hashes, fixed seeds, and protected-file hashes. Failed required gates or
resources SHALL write a `blocked` artifact with the observed failed check and
SHALL NOT substitute legacy models for headline rows.

For a qualified run, Exp6553 SHALL freeze at least 36 evaluable chronological
query boundaries per model, all three domains, at least three regime
transitions, split, order, arms, update dose, replay capacity, thresholds,
seeds, budgets, timeouts, censoring rules, and support probes before held
outcomes. It SHALL compare frozen, current-only, transactional replay,
matched-dose co-observation, one-threshold, hysteretic, and same-query-mutation
arms on identical model and query cells. The same-query arm is an unsafe
diagnostic and SHALL NOT support adoption.

Each query SHALL read a frozen pre-query memory hash. Proposed memory writes
SHALL commit only after exact validation with a witness. Each row SHALL record
request hash, response hash, exact result, proposed write, witness, commit
decision, post-query memory hash, route, fallback, tokens, solver calls, GPU
samples, monotonic clocks, exit status, and censoring. Current exact success
and charged cost, retained-family exact success, future exact-satisfying
support, proposal coverage, unsafe writes and uses, state churn, capacity,
harmful interventions, restart equality, and rollback equality SHALL derive
from emitted rows.

Exp6553 SHALL set `prospective_csl_ready_score` to bare `1.0` only when a safe
arm has positive charged prospective value, zero unsafe writes and uses,
exact-output equality, retained-family and future-support non-inferiority,
restart and rollback equality, multi-model support, clean receipts, protected
files unchanged, and passing validation commands. Otherwise it SHALL close as
`null`, `partial`, `blocked`, or `disqualified` according to the observed
failure. It SHALL set
`inference_substrate="authenticated_local_llama_cpp_sota_gguf_chronological_csl_plus_exact_z3"`
and `verifier_is_oracle=false`.

Required artifact fields and principles:

- `status`: A terminal state distinguishes a completed prospective stream from cached setup output.
- `honest_verdict`: The verdict must name current, retention, future-support, safety, and receipt outcomes with a terminal prefix.
- `verdict_class`: A closed class prevents circular, unsafe, blocked, or partial learning from becoming positive.
- `upstream_gate_receipts`: Both reversible-controller and external-evidence gates must be independently recheckable.
- `MODEL_SPECS`: Exact mandated model identities prevent legacy smoke models from supporting headline claims.
- `live_model_and_gpu_receipts`: Process, model-file, GPU, timing, and output receipts prove fresh local inference occurred.
- `sample_size_and_power_contract`: Per-model query, domain, regime, and seed floors bound the strength of comparative claims.
- `frozen_chronology_and_arm_contract`: Freezing order, arms, dose, budgets, and support probes prevents outcome-driven design.
- `per_unit_rows`: Every model, query, seed, arm, and condition needs a row for recomputation.
- `memory_transition_rows`: Each proposed write and committed state change must carry its exact witness and hashes.
- `current_cost_and_success_rows`: Immediate benefit must charge model, solver, routing, and memory work.
- `retained_family_rows`: Current gains may not hide regression on earlier constraint families.
- `future_support_rows`: Endpoint gains are ineligible if future exact-satisfying behavior becomes less reachable.
- `coobservation_and_dose_receipt`: Replay benefit must be separated from extra update exposure.
- `unsafe_write_and_use_ledger`: One invalid admission or reuse is load-bearing safety evidence.
- `restart_and_rollback_receipts`: Continuous learning must persist and recover exactly across process and state failures.
- `charged_cost_recomputation`: All live inference, exact checks, persistence, and intervention costs must derive from raw receipts.
- `prospective_csl_ready_score`: A binary headline is allowed only when benefit, safety, support, and receipt gates all pass.
- `aggregate_row_recomputation`: Every headline must derive from per-unit and transition rows.
- `gate_check_summary`: A blocked run must name the failed gate or live resource and observed value.
- `preconditions_checked`: GPU, model, runner, solver, and storage checks distinguish blocked execution from null learning.
- `protected_files_unchanged`: The experiment must preserve protected orchestration files.
- `inference_substrate`: The artifact must declare authenticated local llama.cpp GGUF inference plus exact Z3 evaluation.
- `verifier_is_oracle`: The compared memory policy is not ground truth; exact Z3 outcomes remain separate authority.
- `field_provenance`: Each headline must identify model receipts, exact rows, transitions, and reducer code.
- `random_seed`: Fixed generation, order, and tie seeds make the prospective comparison repeatable.
- `duration_s`: Real flagship GGUF inference requires plausible monotonic wall time.
- `tests_run`: Named unit, lint, verifier, and E2E receipts show all paths were checked.
- `reproducibility_checksum`: A final hash detects mutation of the terminal prospective record.

### SCENARIO-CL-6553-FAIL-CLOSED-PRECONDITIONS: Block Before Headline Rows

Given an upstream gate, GPU, llama.cpp, GGUF path, tokenizer, solver, storage,
seed, or protected-file precondition fails
When Exp6553 builds its terminal artifact
Then `verdict_class` SHALL be `blocked`
And `gate_check_summary` SHALL name the failed check and observed value
And no legacy model row SHALL support the headline.

### SCENARIO-CL-6553-CHRONOLOGY-FREEZE: Query Memory Is Frozen

Given a chronological query boundary
When any safe arm evaluates the query
Then it SHALL read only the pre-query memory hash, SHALL NOT see its own label
or future turns, and SHALL commit a proposed write only after exact validation.

### SCENARIO-CL-6553-MATCHED-ARMS: Dose And Query Cells Are Shared

Given the mandated model and query cells
When the seven arms run
Then frozen, current-only, transactional replay, matched-dose co-observation,
one-threshold, hysteretic, and same-query-mutation rows SHALL share model,
query, seed, budget, timeout, and update-dose contracts.

### SCENARIO-CL-6553-SUPPORT-RETENTION: Positive Requires No Support Loss

Given a safe arm improves charged current value
When readiness is computed
Then retained-family exact success and future exact-satisfying support SHALL be
non-inferior before `prospective_csl_ready_score` can be `1.0`.

### SCENARIO-CL-6553-RESTART-ROLLBACK-SAFETY: Unsafe Controls Cannot Adopt

Given restart, rollback, corrupt-write, same-query leakage, future-turn access,
held-threshold tuning, unequal dose, stale-output, fake-CUDA, legacy-model,
unsupported-fallback, or aggregate-only attacks
When Exp6553 audits rows and receipts
Then safe arms SHALL preserve exact output, restart equality, rollback
equality, zero unsafe writes and uses, and the same-query arm SHALL remain
diagnostic only.

## Implementation Status (REQ-CL-6553)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6553 | Planned: `python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py`; terminal artifact `results/experiment_6553_prospective_sota_continuous_self_learning.json`. | Planned: `tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py`. |
| SCENARIO-CL-6553-FAIL-CLOSED-PRECONDITIONS | Planned: `python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py`. | Planned: `tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py`. |
| SCENARIO-CL-6553-CHRONOLOGY-FREEZE | Planned: `python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py`. | Planned: `tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py`. |
| SCENARIO-CL-6553-MATCHED-ARMS | Planned: `python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py`. | Planned: `tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py`. |
| SCENARIO-CL-6553-SUPPORT-RETENTION | Planned: `python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py`. | Planned: `tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py`. |
| SCENARIO-CL-6553-RESTART-ROLLBACK-SAFETY | Planned: `python/carnot/experiment_6553_prospective_sota_continuous_self_learning.py`. | Planned: `tests/python/test_experiment_6553_prospective_sota_continuous_self_learning.py`. |

## REQ-CL-6554: Independent Prospective CSL Audit

Given Exp6553 attempts a prospective SOTA continuous self-learning result
When Exp6554 runs for planning date 20260823
Then it SHALL always write
`results/experiment_6554_continuous_self_learning_independent_audit.json`
as one terminal artifact.

Exp6554 SHALL not run new GGUF generation. It SHALL audit stored Exp6553
receipts, rows, journals, checkpoints, and memory transitions. It SHALL record
input paths, hashes, raw-receipt existence, checkpoint existence, model-file
hashes computed without loading weights, Z3 and Python versions, CPU, RAM,
disk, audit seed, and protected-file hashes.

Exp6554 SHALL validate process ids, commands, model identities, model-file
hashes, output hashes, monotonic clocks, GPU samples, exit status, durations,
and stale-output evidence. Missing model identity, legacy substitution,
impossible live timing, missing rows, missing raw receipts, or non-terminal
upstream evidence SHALL not become a scientific null.

Exp6554 SHALL replay every exact outcome and memory transition from the prior
state, witness, event, and checker. It SHALL recompute memory hashes at each
boundary. It SHALL recompute current success and charged cost, retained-family
effects, future-support effects, co-observation dose, unsafe writes and uses,
churn, capacity, restarts, rollbacks, timeouts, and censoring from row data
only.

Exp6554 SHALL attack missing rows, duplicate rows, aggregate tampering,
query-boundary leakage, future access, held tuning, unequal dose, model aliases,
zero-headroom wins, all-null metric cells, and circular exact authority. It
SHALL set `continuous_self_learning_audited_ready_score` to bare `1.0` only
when receipt authenticity, row closure, exact replay, transition replay, dose,
safety, retention, support, restart, rollback, and verdict recomputation all
pass. It SHALL set
`inference_substrate="independent_stored_sota_receipt_and_exact_transition_replay_no_new_llm"`
and `verifier_is_oracle=false`.

Required artifact fields and principles:

- `status`: An always-run audit needs a terminal state for missing, invalid, null, and positive inputs.
- `honest_verdict`: The verdict must state receipt, safety, retention, support, and scientific disposition with a terminal prefix.
- `verdict_class`: A closed class keeps audit status and scientific status structurally bounded.
- `input_existence_and_hash_receipts`: The audit must identify the exact artifact, raw receipts, journals, and checkpoints it used.
- `independent_live_receipt_audit_rows`: Fresh receipt checks prevent cached or legacy execution from posing as flagship inference.
- `independent_exact_replay_rows`: Z3 replay must confirm every credited current, retained, and future outcome.
- `independent_transition_replay_rows`: Memory effects are eligible only when every state hash and witness recomputes.
- `independent_current_effect_rows`: Immediate claims must be recomputed from matched units and charged costs.
- `independent_retention_and_support_rows`: The audit must expose any older-family or future-support regression.
- `dose_and_coobservation_audit`: A replay benefit cannot be credited to more update exposure.
- `unsafe_write_and_use_audit`: No aggregate gain can hide one invalid memory action.
- `restart_rollback_and_persistence_audit`: Reusable learning must reproduce and recover exactly across state boundaries.
- `missing_input_disposition`: Missing live evidence must close blocked rather than produce a scientific null.
- `attack_matrix`: Receipt, leakage, dose, row, headroom, and circularity attacks stress the full claim.
- `continuous_self_learning_audited_ready_score`: A binary audit score defines whether the prospective result may enter the capstone.
- `per_unit_rows`: Every independent comparative conclusion needs unit-level recomputation rows.
- `aggregate_row_recomputation`: The audit verdict must derive only from independent rows.
- `gate_check_summary`: A blocked audit must list each missing or failed check and observed value.
- `preconditions_checked`: Input and replay checks distinguish a blocked audit from null science.
- `protected_files_unchanged`: The audit must not repair upstream evidence or mutate protected files.
- `inference_substrate`: The audit replays stored receipts and exact checks; it does not claim new GGUF generation.
- `verifier_is_oracle`: The learned memory policy is not authority; the audit uses separate exact evaluation.
- `field_provenance`: Each disposition field must point to immutable rows, receipts, and reducers.
- `random_seed`: A fixed audit sample and attack order make the audit reproducible.
- `duration_s`: Monotonic time exposes an audit that skipped receipt or replay work.
- `tests_run`: Named tests and E2E commands show independent checks executed.
- `reproducibility_checksum`: A final hash protects the independent determination trail.

### SCENARIO-CL-6554-MISSING-INPUT: Missing Live Rows Block The Audit

Given Exp6553 has no raw live receipt rows, no per-unit rows, no checkpoint, or
no journal
When Exp6554 audits the stored inputs
Then `verdict_class` SHALL be `blocked`
And `continuous_self_learning_audited_ready_score` SHALL be `0.0`
And `missing_input_disposition` SHALL name each missing input.

### SCENARIO-CL-6554-RECEIPTS: Live Receipts Are Independently Checked

Given stored Exp6553 model and output receipts
When Exp6554 validates them
Then every credited row SHALL have a mandated model id, matching model hash,
valid process id, terminal exit status, monotonic clock interval, GPU sample,
and output hash.

### SCENARIO-CL-6554-REPLAY: Exact And Transition Replay Close

Given per-unit and memory-transition rows
When Exp6554 replays the row stream
Then exact result hashes, witness hashes, pre-state hashes, post-state hashes,
and commit decisions SHALL recompute for every credited event.

### SCENARIO-CL-6554-ROWS: Audit Metrics Derive From Rows

Given the row stream is closed
When Exp6554 recomputes current value, retention, support, dose, safety,
restart, rollback, timeout, and censoring metrics
Then aggregate readiness SHALL derive only from independent row reductions.

### SCENARIO-CL-6554-ATTACKS: Shortcut Attacks Fail Closed

Given duplicated rows, aggregate tampering, future access, held tuning, unequal
dose, model aliases, zero-headroom wins, all-null cells, or circular authority
When Exp6554 evaluates attacks
Then no attack may leave `continuous_self_learning_audited_ready_score` at
`1.0`.

### SCENARIO-CL-6554-ATOMIC: Output Is Terminal And Atomic

Given Exp6554 finishes blocked, partial, disqualified, or clean
When it writes the artifact
Then the output SHALL be atomically replaced, checksum-protected, and validated
against the required field set.

## Implementation Status (REQ-CL-6554)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6554 | Implemented: `python/carnot/experiment_6554_continuous_self_learning_independent_audit.py`; terminal artifact `results/experiment_6554_continuous_self_learning_independent_audit.json`. | Implemented: `tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py`. |
| SCENARIO-CL-6554-MISSING-INPUT | Implemented: `python/carnot/experiment_6554_continuous_self_learning_independent_audit.py`. | Implemented: `tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py`. |
| SCENARIO-CL-6554-RECEIPTS | Implemented: `python/carnot/experiment_6554_continuous_self_learning_independent_audit.py`. | Implemented: `tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py`. |
| SCENARIO-CL-6554-REPLAY | Implemented: `python/carnot/experiment_6554_continuous_self_learning_independent_audit.py`. | Implemented: `tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py`. |
| SCENARIO-CL-6554-ROWS | Implemented: `python/carnot/experiment_6554_continuous_self_learning_independent_audit.py`. | Implemented: `tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py`. |
| SCENARIO-CL-6554-ATTACKS | Implemented: `python/carnot/experiment_6554_continuous_self_learning_independent_audit.py`. | Implemented: `tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py`. |
| SCENARIO-CL-6554-ATOMIC | Implemented: `python/carnot/experiment_6554_continuous_self_learning_independent_audit.py`. | Implemented: `tests/python/test_experiment_6554_continuous_self_learning_independent_audit.py`. |

## REQ-CL-6748: Read-Only Episode Transactional Constraint Memory

Given FR11 needs reusable external memory without same-episode self-rewrite
When Exp6748 runs for planning date 20260829
Then it SHALL freeze a deterministic controlled constraint stream before policy evaluation
And every active episode SHALL read one immutable parent snapshot
And all writes during an active episode SHALL fail closed and emit attack rows
And exact-certified records SHALL commit atomically only between episodes.

The stream SHALL contain reusable repair structure, naive distractors, held-out
families, retention anchors, and poison, stale, and conflict events. It SHALL
publish six preregistered chronological orders and immutable stream, order, and
attack seeds. Each proposed active record SHALL pass the exact checker, scope,
provenance, future-use eligibility, TTL, conflict, and duplicate checks. Each
commit receipt SHALL contain parent, evidence, and new-state hashes, a reason,
and an inverse patch.

Exp6748 SHALL use only a task-owned temporary state directory. It SHALL test
normal commits, duplicates, contradictions, stale evidence, provenance loss,
delayed-copy poison, crash before rename, crash after rename, restart from
every event boundary, quarantine, and byte-exact rollback. It SHALL not read or
write shared live memory.

The artifact SHALL contain `field_principles`, `inference_substrate`,
`duration_s`, `random_seed`, `reproducibility_checksum`, `rows`,
`stream_manifest`, `commit_receipts`, `read_only_violations`,
`unsafe_admission_count`, `unsafe_use_count`, `restart_receipts`,
`rollback_byte_identity`, `transaction_memory_ready`, `gate_check_summary`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL cover every
artifact field and every readiness gate. `inference_substrate` SHALL equal
`deterministic CPU exact-checker transactional fixture`.
The adversarial verifier SHALL recognize that exact value as a deterministic
no-LLM substrate and apply its nonzero deterministic-verifier duration floor.

`transaction_memory_ready` SHALL be bare true only when every mandatory row
passes, no unsafe record is admitted or used, every restart reproduces exact
state bytes, and rollback bytes equal the parent snapshot. A failed owned
precondition SHALL emit `complete_blocked_transaction_fixture` and a
`gate_check_summary` that names the failed check and observed value. The closed
`verdict_class` SHALL be one of `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`.

### SCENARIO-CL-6748-READ-ONLY: Active Episodes Reject Writes

Given an episode has an immutable snapshot
When code attempts to commit before that episode closes
Then the write SHALL raise a fail-closed error
And the parent bytes SHALL remain unchanged
And `read_only_violations` SHALL record the rejected attempt.

### SCENARIO-CL-6748-DELAYED-COMMIT: Exact Updates Commit Between Episodes

Given an episode closes with a proposed record
When all seven admission checks pass
Then one atomic rename SHALL publish the next state
And its receipt SHALL bind the parent, evidence, next state, reason, and inverse patch.

### SCENARIO-CL-6748-ATTACKS: Unsafe Updates Never Enter Active State

Given duplicate, contradiction, stale, missing-provenance, delayed-copy poison,
or crash injection events
When the fixture evaluates them
Then each event SHALL reject or recover at an atomic boundary
And unsafe admission and use counts SHALL remain zero
And rejected records SHALL enter only the task-owned quarantine.

### SCENARIO-CL-6748-RESTART: Every Boundary Replays Exact Bytes

Given any event or atomic-rename boundary
When a new memory instance starts from the task-owned state file
Then its bytes and state hash SHALL match the expected committed boundary.

### SCENARIO-CL-6748-ROLLBACK: Inverse Patches Restore Parent Bytes

Given a committed exact-certified update and its inverse patch
When rollback runs after restart
Then the restored state bytes SHALL match the parent snapshot byte for byte.

### SCENARIO-CL-6748-ARTIFACT: Readiness Is Row-Derived

Given the frozen stream, transaction rows, attack rows, restart receipts, and
rollback rows
When Exp6748 builds the terminal artifact
Then every readiness gate SHALL derive from those rows
And the artifact SHALL validate before atomic publication to
`results/experiment_6748_transactional_constraint_memory_fixture.json`.

## Implementation Status (REQ-CL-6748)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6748 | Implemented: `python/carnot/memory/transactional_constraint_memory.py`; `scripts/experiments/experiment_6748_transactional_constraint_memory_fixture.py`. | Implemented: `tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py`. |
| SCENARIO-CL-6748-READ-ONLY | Implemented: `python/carnot/memory/transactional_constraint_memory.py`. | Implemented: `tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py`. |
| SCENARIO-CL-6748-DELAYED-COMMIT | Implemented: `python/carnot/memory/transactional_constraint_memory.py`. | Implemented: `tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py`. |
| SCENARIO-CL-6748-ATTACKS | Implemented: `python/carnot/memory/transactional_constraint_memory.py`. | Implemented: `tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py`. |
| SCENARIO-CL-6748-RESTART | Implemented: `python/carnot/memory/transactional_constraint_memory.py`. | Implemented: `tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py`. |
| SCENARIO-CL-6748-ROLLBACK | Implemented: `python/carnot/memory/transactional_constraint_memory.py`. | Implemented: `tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py`. |
| SCENARIO-CL-6748-ARTIFACT | Implemented: `python/carnot/memory/transactional_constraint_memory.py`. | Implemented: `tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py`. |
