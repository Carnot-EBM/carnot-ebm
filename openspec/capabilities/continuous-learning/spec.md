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
