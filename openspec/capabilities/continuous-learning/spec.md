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

## REQ-CL-6749: Prospective Support-Preserving Transactional Memory A/B

Given Exp6748 has a ready transaction fixture and six frozen orders
When Exp6749 runs for planning date 20260829
Then it SHALL compare frozen no-memory and read-only transactional-memory arms
without changing model weights
And it SHALL write
`results/experiment_6749_prospective_support_preserving_csl_ab.json`.

Exp6749 SHALL use exactly `unsloth/Qwen3.6-35B-A3B-GGUF` for acquisition and
same-family evaluation. It SHALL use exactly
`unsloth/gemma-4-31B-it-GGUF` for held dense-family transfer. Both models SHALL
load from exact cached GGUF paths through CUDA-enabled llama.cpp. A failed
fixture, order, model, CUDA-offload, or sequential-VRAM check SHALL emit
`complete_blocked_prospective_csl` with the observed failure. It SHALL not use
a substitute model, order, or retrospective replay.

Before inference, Exp6749 SHALL freeze prompts, two candidates per cell, token
and verifier budgets, all six orders, retention anchors, support definitions,
rollback rules, and model, order, and candidate seeds. Each order SHALL start
both arms from clean state. An active episode may read one immutable starting
snapshot. It SHALL not write during inference. Qwen proposals may commit only
after the exact result is known and all Exp6748 admission checks pass. Gemma
SHALL read the Qwen pre-event snapshot and SHALL not add target-family evidence.

Exp6749 SHALL keep every failed, incorrect, and abstaining candidate. Rows SHALL
cover every order, model, family, event, and arm cell. Row-derived metrics SHALL
include prequential exact yield, pass at one, best at two, effective rewardable
support, joint correct-and-constraint-following support, anchor retention,
cross-family transfer, negative transfer, tokens, latency, commits, rejects,
quarantine, and rollback.

`prospective_csl_completed` SHALL be bare true only when all planned rows exist,
chronology and arm isolation hold, snapshots remain immutable, exact authority
owns at least one commit and one reject, rollback closes, and model weights
remain unchanged. This field
is a completion gate. It SHALL not encode whether the scientific result is
positive. The artifact SHALL contain every field named in the task contract,
including a field principle for every field and gate.

### SCENARIO-CL-6749-PROSPECTIVE-ORDER: Every Cell Uses Frozen Chronology

Given the six Exp6748 order manifests and frozen candidate seeds
When either arm evaluates an event
Then its row SHALL bind the preregistered order position and pre-event snapshot
And no row may use current or future outcome evidence.

### SCENARIO-CL-6749-SNAPSHOT: Active Episodes Are Immutable

Given a transactional episode has started
When all candidates for that event are generated and checked
Then the durable state hash SHALL remain equal to the starting snapshot hash
And any attempted active-episode write SHALL fail closed.

### SCENARIO-CL-6749-EXACT-ADMISSION: Exact Authority Owns Commits

Given Qwen has completed an acquisition episode
When Exp6749 evaluates a proposed reusable record
Then it SHALL commit only after the exact result is known
And every Exp6748 admission check SHALL pass before publication.

### SCENARIO-CL-6749-ARM-ISOLATION: Baseline Never Reads Or Writes Memory

Given matched no-memory and transactional-memory cells
When their candidates are generated
Then model, event, order, candidate count, budgets, and seeds SHALL match
And the no-memory arm SHALL have no memory reads, writes, commits, or rollback.

### SCENARIO-CL-6749-SUPPORT: Support Metrics Derive From Candidates

Given all candidates, including failures and abstentions, are retained
When Exp6749 reduces support metrics
Then pass at one, best at two, effective rewardable support, and joint correct
constraint support SHALL recompute from candidate rows only.

### SCENARIO-CL-6749-NO-WEIGHT-WRITES: Model Files Stay Read-Only

Given exact cached GGUF files are loaded for inference
When both model phases finish
Then their size and modification-time receipts SHALL remain unchanged
And `model_weights_mutated` SHALL be bare false.

## Implementation Status (REQ-CL-6749)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6749 | Planned: `python/carnot/experiment_6749_prospective_support_preserving_csl_ab.py`; `scripts/experiments/experiment_6749_prospective_support_preserving_csl_ab.py`. | Planned: `tests/python/test_experiment_6749_prospective_support_preserving_csl_ab.py`. |
| SCENARIO-CL-6749-PROSPECTIVE-ORDER | Planned. | Planned. |
| SCENARIO-CL-6749-SNAPSHOT | Planned. | Planned. |
| SCENARIO-CL-6749-EXACT-ADMISSION | Planned. | Planned. |
| SCENARIO-CL-6749-ARM-ISOLATION | Planned. | Planned. |
| SCENARIO-CL-6749-SUPPORT | Planned. | Planned. |
| SCENARIO-CL-6749-NO-WEIGHT-WRITES | Planned. | Planned. |

## REQ-CL-6750: Cold Durability, Support, And Poison Audit

Given Exp6749 is the prospective support-preserving CSL A/B for planning date
20260829
When Exp6750 starts in a fresh CPU process
Then it SHALL write
`results/experiment_6750_csl_durability_support_poison_audit.json`
And it SHALL load only immutable Exp6749 rows, Exp6749 state snapshot fields,
Exp6748 stream manifests, Exp6748 state bytes, and Exp6748 commit receipts.
The required wrapper command SHALL live at
`scripts/experiments/experiment_6750_csl_durability_support_poison_audit.py`.
The top-level `inference_substrate` SHALL equal
`fresh_process_no_llm_transaction_audit`.

Exp6750 SHALL not rerun live inference, fabricate rows, repair missing rows,
replace missing rows, or reinterpret rows from another experiment as Exp6749
rows. If Exp6749 is absent, `prospective_csl_completed` is not bare true, raw
rows are absent, six frozen orders are absent, state snapshots are absent, or
commit receipts are absent, it SHALL emit `complete_blocked_csl_audit` with a
`gate_check_summary` that names each failed precondition and its observed
value.

Exp6750 SHALL recompute prequential yield deltas by order from raw rows. It
SHALL compute the preregistered order-level 95 percent interval, support
contraction by support metric, retention failures, negative transfer, tokens,
commit counts, reject counts, and rollback counts from rows and receipts. A
positive CSL audit SHALL require the order-level lower confidence bound to be
above zero, no best-at-k or effective-support contraction beyond its bound, no
anchor forgetting, no harmful cross-family transfer, zero admitted poison, and
byte-exact rollback.

Exp6750 SHALL verify that each episode snapshot predates its event. It SHALL
reject any snapshot with future evidence, held-family evidence, or opposite-arm
evidence. It SHALL recompute each commit receipt's parent hash, evidence hash,
new-state hash, and inverse rollback identity from the receipt bytes.

Exp6750 SHALL replay duplicate, stale, contradiction, delayed-copy, relation
poison, provenance-loss, and tombstone-reappearance attacks with independent
CPU state. No replayed attack may admit a record, make an unsafe record
readable, or corrupt later restart state.

### SCENARIO-CL-6750-COLD-RECOMPUTE: Rows Drive The Audit

Given the immutable Exp6749 artifact has raw candidate rows
When Exp6750 recomputes metrics
Then all prequential deltas, support values, retention rows, negative-transfer
rows, token counts, and lifecycle counts SHALL derive from rows and receipts
rather than copied aggregate fields.

### SCENARIO-CL-6750-CHRONOLOGY: Future Evidence Is Denied

Given an Exp6749 episode snapshot for one order position
When Exp6750 validates it
Then the snapshot SHALL predate the event and SHALL contain no future,
held-family, or opposite-arm evidence.

### SCENARIO-CL-6750-POISON: Poison Variants Fail Closed

Given duplicate, stale, contradiction, delayed-copy, relation-poison,
provenance-loss, and tombstone-reappearance attacks
When Exp6750 replays them against copied CPU state
Then every variant SHALL have zero admission and zero unsafe use.

### SCENARIO-CL-6750-RESTART: Boundaries Replay Deterministically

Given each Exp6748 commit boundary and restart receipt
When Exp6750 replays the remaining stream from that boundary
Then each boundary SHALL reproduce the expected bytes and state hash.

### SCENARIO-CL-6750-ROLLBACK: Inverse Patches Restore Parents

Given each Exp6748 commit receipt
When Exp6750 applies its inverse patch after a restart
Then the restored state bytes SHALL equal the parent snapshot bytes exactly.

## Implementation Status (REQ-CL-6750)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6750 | Planned: `python/carnot/experiment_6750_csl_durability_support_poison_audit.py`; terminal artifact `results/experiment_6750_csl_durability_support_poison_audit.json`. | Planned: `tests/python/test_experiment_6750_csl_durability_support_poison_audit.py`. |
| SCENARIO-CL-6750-COLD-RECOMPUTE | Planned. | Planned. |
| SCENARIO-CL-6750-CHRONOLOGY | Planned. | Planned. |
| SCENARIO-CL-6750-POISON | Planned. | Planned. |
| SCENARIO-CL-6750-RESTART | Planned. | Planned. |
| SCENARIO-CL-6750-ROLLBACK | Planned. | Planned. |

## REQ-CL-6761: Capacity-Controlled Procedural Memory Stream

Given Exp6748 provides exact transactional memory but Exp6749 and Exp6750
recorded no prospective commits or rejects
When Exp6761 runs for planning date 20260829
Then it SHALL build at least six preregistered chronological orders
And every order SHALL expose at least 12 exact-eligible accepts and 12
exact-eligible rejects before a later memory comparison starts.

The stream SHALL include reusable procedure families, naive distractors, held
families, hard cases, retention anchors, contradictions, stale lessons,
duplicates, poison candidates, and provenance-loss candidates. All orders SHALL
be fixed before dry replay. An active episode SHALL read only its pre-event
snapshot. It SHALL not read its current exact result or any later event.

Each reusable event SHALL have a detailed trajectory and an abstract procedural
lesson. The procedural lesson SHALL state an abstract constraint, an
applicability scope, and a repair procedure. Neither representation SHALL
contain current or future answer content. Both representations SHALL bind to
the same evidence hash.

Detailed and procedural memory arms SHALL have equal storage bytes, record
slot bytes, top-k retrieval, context tokens, TTL policy, update opportunities,
and exact authority. The maximum committed bytes SHALL stay below the frozen
ceiling in every order and arm. Counts and capacity use SHALL derive from one
row per order and event.

Transactions SHALL occur only after an exact result closes the active episode.
Each accept or reject transaction SHALL include its parent hash, evidence hash,
representation type, scope, TTL, admission reason, inverse patch, and atomic
restart receipt. Restart from each transaction boundary SHALL reproduce exact
bytes. Each accepted transaction SHALL also have a byte-exact rollback receipt.
Poison, contradiction, duplicate, stale, delayed-evidence, and provenance-loss
candidates SHALL reject for their preregistered reason.

`procedural_memory_stream_ready` SHALL be true only when chronology,
non-saturation, nonzero accept and reject opportunity, capacity equality,
restart, rollback, and poison gates all pass. A missing Exp6748 fixture, exact
authority, atomic storage, restart helper, or rollback helper SHALL produce
`complete_blocked_procedural_stream`. `gate_check_summary` SHALL name each
failed check and its observed value. The experiment SHALL use
`deterministic_verifier_plus_replay: exact-labeled chronological stream, no LLM`
and SHALL set
`verifier_is_oracle=false` because exact authority labels admissions but is not
a learned verifier claim.

### SCENARIO-CL-6761-CHRONOLOGY: Active Episodes Cannot See Current Or Future Evidence

Given one event in a preregistered order
When its active episode opens
Then its snapshot SHALL contain only evidence from earlier closed events
And active writes, current evidence, and future evidence SHALL be unavailable.

### SCENARIO-CL-6761-CAPACITY: Representation Arms Have Equal Unsaturated Budgets

Given a detailed trajectory and procedural lesson for each reusable event
When the stream dry replay commits exact-eligible records
Then both arms SHALL charge the same fixed record slot
And storage, retrieval, context, TTL, update, and authority contracts SHALL
match
And committed bytes SHALL stay below the common ceiling.

### SCENARIO-CL-6761-TRANSACTIONS: Accepts And Rejects Are Durable And Reversible

Given an event closes with an exact admission result
When both representation arms transact between episodes
Then each receipt SHALL contain the complete transaction schema
And restart SHALL reproduce the expected state bytes
And an accepted update's inverse patch SHALL restore its parent bytes.

### SCENARIO-CL-6761-POISON: Unsafe Candidates Reject For The Intended Reason

Given poison, contradiction, duplicate, stale, delayed-evidence, and
provenance-loss candidates
When exact dry replay evaluates them
Then no candidate SHALL enter active memory
And each receipt SHALL report its preregistered rejection reason.

### SCENARIO-CL-6761-ROWS: All Readiness Counts Recompute From Rows

Given all order and event rows
When Exp6761 reduces the terminal artifact
Then each order SHALL have at least 12 eligible accepts and 12 eligible rejects
And hard-case counts, future-evidence violations, and capacity use SHALL
recompute without copied summary counts.

## Implementation Status (REQ-CL-6761)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6761 and SCENARIO-CL-6761-* | Planned: `python/carnot/experiment_6761_procedural_memory_stream.py`; `scripts/experiments/experiment_6761_procedural_memory_stream.py`; terminal artifact `results/experiment_6761_procedural_memory_stream.json`. | Planned: `tests/python/test_experiment_6761_procedural_memory_stream.py`. |

## REQ-CL-6762: Prospective Procedural Versus Trajectory Memory Comparison

Given Exp6761 provides a ready non-saturating stream with six fixed orders
When Exp6762 runs for planning date 20260829
Then it SHALL compare `no_memory`, `detailed_trajectory`, and
`procedural_constraint` on every order and event
And it SHALL use `unsloth/Qwen3.6-35B-A3B-GGUF` for acquisition and
within-family evaluation
And it SHALL use `unsloth/gemma-4-31B-it-GGUF` for dense held-out transfer.

The experiment SHALL freeze model roles, model revisions, GGUF hashes, embedded
tokenizers, six order hashes, arm order rotations, prompts, seeds, candidate
count, decode budget, context budget, storage bytes, record slots, top-k, TTL,
update opportunities, exact-check budget, and retention anchors before the
first model loads. Models SHALL run in sequential local llama.cpp CUDA sessions.
Legacy or remote models SHALL not satisfy a gate.

Every active episode SHALL read one immutable pre-event snapshot. It SHALL not
write memory or see its current exact result. Exact-approved transactions SHALL
occur only after the episode closes. Each model and arm SHALL use isolated
mutable state. Gemma SHALL not receive Qwen answer traces.

Exp6762 SHALL emit one row per order, model, arm, and event. Each row SHALL
record retrieved IDs and scores, context bytes and tokens, citation and
operational-use signals, before and after action fingerprints, exact result,
difficulty, family, latency, token counts, commit or reject reason, state
hashes, read-only evidence, restart evidence, and rollback evidence. All
headline metrics SHALL recompute from rows.

`prospective_csl_completed` SHALL be true only when all 1,080 planned rows,
state boundaries, transactions, model teardown receipts, and cold row
recomputation pass. Completion SHALL not imply a positive result.

Positive credit SHALL require procedural memory to beat both other arms with
positive order-level lower confidence bounds. It SHALL also require nonzero
commits and rejects, no retention-anchor forgetting, no hard-case regression
beyond the frozen margin, no effective-support contraction, and zero admitted
poison.

If Exp6761 is not ready, either exact cached GGUF or embedded tokenizer is
missing, llama.cpp lacks CUDA, one-model VRAM is insufficient, task ownership
or ports are unavailable, exact authority or atomic storage is unavailable, or
RAM or disk is insufficient, Exp6762 SHALL write
`complete_blocked_procedural_csl_ab`. The blocked artifact SHALL contain the
complete schema, no headline fallback rows, and a `gate_check_summary` with the
failed check and observed value.

### SCENARIO-CL-6762-CHRONOLOGY: Snapshots Contain Only Earlier Events

Given one event in one frozen order
When any arm opens its active episode
Then the visible event IDs SHALL equal the strict order prefix
And current and future evidence SHALL be unavailable.

### SCENARIO-CL-6762-READ-ONLY: Active Episodes Cannot Write

Given a detailed or procedural memory snapshot
When inference is active
Then state bytes and the state hash SHALL remain unchanged
And any update SHALL wait until the exact result closes the episode.

### SCENARIO-CL-6762-CAPACITY: Memory Arms Have Equal Resource Contracts

Given matched detailed and procedural representations
When either memory arm stores, retrieves, or renders context
Then storage, slot, top-k, TTL, context, update, and exact-check budgets SHALL
match
And the no-memory arm SHALL never read or write mutable memory.

### SCENARIO-CL-6762-RETRIEVAL-ACTION: Rows Prove Behavioral Memory Use

Given a memory-enabled event row
When retrieval supplies prior records
Then the row SHALL record IDs, scores, rendered context cost, citation, the
memory-free action fingerprint, the memory-conditioned fingerprint, and whether
the retrieved record changed or supported the selected action.

### SCENARIO-CL-6762-TRANSACTIONS: Exact Authority Produces Activity

Given an exact-eligible accept or reject event
When the active episode closes
Then each memory arm SHALL record a commit or reject receipt with its reason
And the complete run SHALL contain nonzero commits and rejects.

### SCENARIO-CL-6762-REDUCERS: Headline Metrics Come From Rows

Given all row units
When Exp6762 reduces the result
Then prequential yield, hard-case yield, best-at-k, effective support, joint
support, retention, forgetting, negative transfer, commits, rejects, actual
retrieval, action influence, tokens, restarts, and rollbacks SHALL recompute
without copied headline values.

### SCENARIO-CL-6762-RESTART: Every Boundary Reopens Exact State

Given any post-event transaction boundary
When the arm state reopens
Then its bytes and hash SHALL match the expected committed or unchanged state
And each accepted update SHALL restore its parent bytes during rollback proof.

### SCENARIO-CL-6762-BLOCKED: Failed Preconditions Do Not Produce Fallback Rows

Given one failed owned precondition
When Exp6762 writes its terminal artifact
Then `verdict_class` SHALL be `blocked`
And `rows` SHALL be empty
And `gate_check_summary` SHALL name the failed check and observed value.

## Implementation Status (REQ-CL-6762)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6762 and SCENARIO-CL-6762-* | Implemented: `python/carnot/experiment_6762_procedural_vs_trace_csl_ab.py`; `scripts/experiments/experiment_6762_procedural_vs_trace_csl_ab.py`; terminal artifact `results/experiment_6762_procedural_vs_trace_csl_ab.json`. | Implemented: `tests/python/test_experiment_6762_procedural_vs_trace_csl_ab.py`. |

## REQ-CL-6773: Owned-Lease Memory Branch Admission Contract

Given the frozen Exp6761 stream is ready with six orders
When Exp6773 runs for planning date 20260830
Then it SHALL revalidate the checked-in stream without regenerating it
And it SHALL prove one bounded first-token inference for each mandated memory model.

The mandated models SHALL be `unsloth/Qwen3.6-35B-A3B-GGUF` for acquisition
and within-family work and `unsloth/gemma-4-31B-it-GGUF` for dense held-family
transfer. Each model SHALL use its exact cached Q4_K_M GGUF and embedded
tokenizer. Legacy, CPU, remote, and substituted models SHALL not satisfy
readiness.

One typed model record SHALL contain `model_id`, `role`, `family`,
`quantization`, `revision`, `filename`, `model_path`, `model_sha256`,
`model_size_bytes`, and `tokenizer`. `model_specs` SHALL contain the two
planned resolved records before any live load. `models_used` SHALL contain the
same records, in the same order, only after both records produce live canaries.
The validator SHALL reject a missing, renamed, extra, or unequal identity
field.

The stream checks SHALL cover the source artifact hash, stream hash, order
count, equal unsaturated capacity, read-only episodes, transaction schema,
restart receipts, rollback receipts, and poison receipts. One row SHALL exist
for each stream check. One row SHALL also exist for each phase in every live
model lease journal.

`csl_live_preflight_ready` SHALL be true only when the two typed model lists
are equal, both models have complete live phase rows, both first-token canaries
pass, both teardown and VRAM recovery checks pass, and all stream checks pass.
This admission field SHALL not make a continuous-learning result claim.

### SCENARIO-CL-6773-STREAM: The Frozen Stream Revalidates Without Regeneration

Given the checked-in Exp6761 artifact
When Exp6773 validates its public fixture and memory contract
Then the source hash, six order hashes, stream hash, capacity, read-only,
transaction, restart, rollback, and poison evidence SHALL match the frozen
artifact.

### SCENARIO-CL-6773-MODEL-CONTRACT: Planned And Used Records Are Identical

Given two resolved typed model records
When live canary receipts are reduced
Then `model_specs` and `models_used` SHALL be byte-equivalent JSON values
And any identity mismatch SHALL close readiness.

### SCENARIO-CL-6773-BLOCKED: A Failed Admission Gate Produces No Live Claim

Given any failed source, model, CUDA, device, lease, port, RAM, or disk gate
When Exp6773 stops before loading
Then it SHALL emit `complete_blocked_csl_owned_lease_contract`
And `csl_live_preflight_ready` and `live_model_invoked` SHALL be false
And `gate_check_summary` SHALL retain the failed check and observed value.

## Implementation Status (REQ-CL-6773)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6773 and SCENARIO-CL-6773-* | Planned: `python/carnot/experiment_6773_csl_owned_lease_contract.py`; `scripts/experiments/experiment_6773_csl_owned_lease_contract.py`; terminal artifact `results/experiment_6773_csl_owned_lease_contract.json`. | Planned: `tests/python/test_experiment_6773_csl_owned_lease_contract.py`. |

## REQ-CL-6790: Frozen Chronological Constraint-Routing Opportunity Stream

Given Exp6786 provides a ready exact constraint-group fixture
When Exp6790 runs for planning date 20260830
Then it SHALL build at least 240 bounded route-decision events
And it SHALL freeze exactly five chronological order replicates before replay
And it SHALL write
`results/experiment_6790_chronological_constraint_routing_stream.json`.

Each event SHALL present one candidate graph and at least two named live routes.
Each live route SHALL inspect a fixed number of factors. That budget SHALL be
smaller than exhaustive checking for every event. Exact labels, exact receipts,
future outcomes, order positions, and future family statistics SHALL not enter
the action feature allowlist. The exact exhaustive route SHALL be diagnostic
only. It SHALL not appear in any event's live actions or baseline actions.

The stream SHALL include repeated visible motifs, whole-family held-future
topology, difficulty drift, local and cross-dependency failures, clean valid
cases, hard cases, and poison candidates. A poison candidate SHALL preserve its
apparent route reward. Its post-action receipt SHALL deny credit when provenance
or retention rules conflict. A held-future family SHALL not appear in factor
statistics, retrieval memory, or tuning data before its first event.

The post-action exact receipt SHALL state whether the chosen route found the
planted failure. It SHALL also state route cost, checked factors, missed
dependencies, safe factors, poison disposition, and exhaustive diagnostic
outcome. The receipt SHALL remain separate from legal observations. A hash of
the complete receipt SHALL bind the hidden pre-action receipt to the revealed
post-action value.

Exp6790 SHALL publish frozen-policy, random-route, and exhaustive-oracle
diagnostic metrics for every order. The exhaustive route SHALL measure headroom
only. In every order, frozen decision accuracy SHALL be above random-route
accuracy and below exhaustive diagnostic accuracy. Every order SHALL contain
actual route alternatives, repeated motif signal, held-future events, poison
rows, and positive diagnostic headroom.

Preconditions SHALL require `constraint_group_fixture_ready=true`, the frozen
Exp6786 artifact and module hashes, at least three topology families, nonzero
local and cross-dependency failures, and a bounded route cost model. A failed
precondition SHALL write `complete_blocked_constraint_routing_stream`. The
blocked artifact SHALL contain no event rows. Its `gate_check_summary` SHALL
name each failed check with its expected and observed values.

The terminal artifact SHALL include `field_principles`,
`inference_substrate`, `duration_s`, `random_seed`,
`reproducibility_checksum`, `source_artifact_hash`, `frozen_manifest`,
`order_definitions`, `feature_allowlist`, `feature_denylist`,
`route_definitions`, `rows`, `event_count_by_order`,
`topology_count_by_order`, `held_future_counts`, `poison_counts`,
`reusable_motif_counts`, `future_feature_violations`,
`frozen_policy_metrics_by_order`, `random_route_metrics_by_order`,
`diagnostic_headroom_by_order`, `cold_replay_hashes`,
`constraint_routing_stream_ready`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.
`inference_substrate` SHALL equal
`CPU exact chronological decision fixture, no LLM`.
`verifier_is_oracle` SHALL be bare false. `verdict_class` SHALL use only
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL start with an approved terminal prefix.

### SCENARIO-CL-6790-CHRONOLOGY: Action Features Precede Exact Receipts

Given one event in a frozen order
When the frozen and random policies choose their routes
Then both choices SHALL use only the current legal observation and earlier
closed-event memory
And the exact receipt SHALL become readable only after both actions are fixed.

### SCENARIO-CL-6790-ROUTES: Live Checks Are Bounded Alternatives

Given one candidate graph
When its available routes are materialized
Then at least two distinct live routes SHALL fit the fixed budget
And every live route SHALL cost less than exhaustive checking
And the exhaustive diagnostic route SHALL not be a live action.

### SCENARIO-CL-6790-HELD-FUTURE: Whole Families Stay Outside Prior State

Given a topology family is designated held future
When an order reaches that family's first event
Then prior factor statistics, retrieval records, and tuning event records SHALL
contain no member of that family
And later rows MAY use only receipts revealed after that first event.

### SCENARIO-CL-6790-POISON: Apparent Reward Cannot Override Safety

Given a route appears to find a failure on a poison candidate
When the post-action receipt evaluates provenance and retention rules
Then apparent reward SHALL remain observable
And credited reward SHALL be zero
And the named poison disposition SHALL remain in the row.

### SCENARIO-CL-6790-HEADROOM: Every Order Has A Floor And Ceiling

Given all 240 events appear once in an order
When baseline metrics derive from rows
Then random-route accuracy SHALL be below frozen-policy accuracy
And frozen-policy accuracy SHALL be below exhaustive diagnostic accuracy
And diagnostic headroom SHALL be strictly positive.

### SCENARIO-CL-6790-REPLAY: Frozen Rows Replay Byte-Stably

Given the frozen events, routes, and five orders
When a fresh CPU process rebuilds the stream
Then each order row hash and the aggregate row hash SHALL match
And no future-feature violation SHALL appear.

### SCENARIO-CL-6790-BLOCKED: Failed Authority Produces A Complete Block

Given one required source, hash, topology, failure class, or budget check fails
When Exp6790 builds its terminal artifact
Then readiness SHALL be false
And rows SHALL be empty
And `gate_check_summary` SHALL preserve the failed check and observed value.

## Implementation Status (REQ-CL-6790)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6790 and SCENARIO-CL-6790-* | Planned: `python/carnot/experiment_6790_chronological_constraint_routing_stream.py`; `scripts/experiments/experiment_6790_chronological_constraint_routing_stream.py`; terminal artifact `results/experiment_6790_chronological_constraint_routing_stream.json`. | Planned: `tests/python/test_experiment_6790_chronological_constraint_routing_stream.py`. |

## REQ-CL-6791: Prospective Compositional Online Constraint-Routing A/B

Given Exp6790 provides a ready five-order constraint-routing stream
When Exp6791 runs for planning date 20260830
Then it SHALL compare `frozen_controller`, `compositional_online`,
`random_update_placebo`, and `retrieval_disabled_online`
And it SHALL write
`results/experiment_6791_compositional_online_constraint_routing_ab.json`.

Exp6791 SHALL freeze the four arms, component algorithms, capacity, thresholds,
update cadence, seeds, route budget, order hashes, and stopping rule before the
first action. The online arms SHALL log separate factor-admission, retrieval,
and route-selection state. The placebo SHALL match the online write count and
logical storage bytes. It SHALL assign update targets from earlier events in
the same frozen stratum. The retrieval-disabled arm SHALL commit the same exact
factors as the compositional arm, but SHALL return no retrieved factor.

For each order, event, and arm, Exp6791 SHALL snapshot isolated state before
route selection. It SHALL record retrieved factor IDs and controller state. It
SHALL freeze one legal route before it reveals the exact Exp6790 receipt. No
arm SHALL write while the current action is active. An accepted factor SHALL
commit only after the exact receipt. It SHALL include a type, source event,
source position, evidence hash, target route, and exact provenance. Every
commit SHALL have restart and byte-exact rollback evidence.

Exp6791 SHALL record later factor reads. For each retrieved factor, it SHALL
select a route again with that factor disabled against the same snapshot. It
SHALL record whether this byte-identical counterfactual changes the action.
Component attribution SHALL separate admission, retrieval, and route selection.

All headline metrics SHALL derive from the stored rows and transaction
receipts. Metrics SHALL include held-future utility net of route cost, route
success, missed hard dependencies, hard-case harm, old-family retention,
unique-action support, writes, later reads, action changes, and rollback.
Paired effects SHALL use order-event keys. The lower confidence bound SHALL use
a seeded bootstrap over the five order-level online-minus-frozen effects.

Preconditions SHALL require `constraint_routing_stream_ready=true`, all five
frozen order hashes, positive headroom in every order, only legal live route
actions, hidden-receipt separation, and writable isolated transaction stores.
A failed check SHALL write
`complete_blocked_online_constraint_routing_ab`. It SHALL name the failed check
and observed value in `gate_check_summary`. It SHALL contain no event rows and
SHALL not run a reduced-order substitute.

`compositional_csl_completed` SHALL be true when every planned row,
transaction, aggregate, and component attribution is complete. Effect direction
SHALL not control completion. A positive verdict SHALL require nonzero online
writes, later reads, and action changes in every order. It SHALL also require an
online-minus-frozen held-future utility lower bound above zero, an online win
over the placebo in every order, and no preregistered hard-case, retention, or
support harm. A complete result without that evidence SHALL use
`verdict_class=null`. Failed preconditions SHALL use `blocked`. Leakage,
cross-arm state, missing rows, or an unplanned substitute SHALL use
`disqualified`.

The terminal artifact SHALL include `field_principles`,
`inference_substrate`, `duration_s`, `random_seed`,
`reproducibility_checksum`, `source_artifact_hash`, `frozen_manifest`,
`arm_definitions`, `component_definitions`, `transaction_capacity`, `rows`,
`transaction_receipts`, `writes_by_arm_order`, `later_reads_by_arm_order`,
`action_changes_by_arm_order`, `component_action_attribution`,
`held_future_utility_by_arm_order`, `online_minus_frozen_order_effects`,
`online_minus_frozen_lcb`, `online_minus_placebo_order_effects`,
`hard_case_harm_by_arm_order`, `retention_by_arm_order`,
`action_support_by_arm_order`, `future_feature_violations`,
`active_event_write_violations`, `compositional_csl_completed`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `inference_substrate` SHALL equal
`CPU prospective Tier-2 constraint-memory controller, no LLM`.
`verifier_is_oracle` SHALL be false because the exact receipt arrives only
after the route is frozen. `verdict_class` SHALL use only `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
`honest_verdict` SHALL start with an approved terminal prefix.

### SCENARIO-CL-6791-ARM-ISOLATION: Every Arm Owns Distinct State

Given one frozen order
When all four arms run
Then each arm SHALL use a distinct transaction path and state hash lineage
And the frozen arm SHALL have zero reads and writes.

### SCENARIO-CL-6791-READ-ONLY: Active Events Cannot Write

Given an arm has started the current event
When route selection reads its snapshot
Then a write attempt SHALL fail
And the state bytes SHALL remain unchanged until the exact receipt arrives.

### SCENARIO-CL-6791-BETWEEN-EVENT-COMMIT: Updates Follow Receipts

Given a factor passes admission after one exact receipt
When the next event starts
Then the prior commit MAY appear in the next snapshot
And it SHALL never appear in its own action snapshot.

### SCENARIO-CL-6791-COMPONENT-ATTRIBUTION: Components Stay Separately Measurable

Given the compositional arm changes an action
When admission, retrieval, or route selection is disabled in turn
Then Exp6791 SHALL record each counterfactual action from the same snapshot.

### SCENARIO-CL-6791-PLACEBO: Random Updates Match Activity

Given the online arm admits one factor
When the placebo updates at the same boundary
Then both arms SHALL write one fixed-size slot
And the placebo target SHALL name an earlier event in the same stratum.

### SCENARIO-CL-6791-RETRIEVAL-DISABLE: Retrieval Ablation Is Exact

Given the online and retrieval-disabled arms receive the same exact factors
When they select the next route
Then their committed factor IDs SHALL match
And only the retrieval-disabled arm SHALL report an empty retrieval result.

### SCENARIO-CL-6791-FUTURE-LEAKAGE: Future Receipts Stay Hidden

Given an order has not reached one event or held-future family
When any arm snapshots state
Then every visible factor SHALL come from a smaller order position
And no held-future factor SHALL appear before that family's first event.

### SCENARIO-CL-6791-PAIRED-KEYS: Every Cell Has One Paired Key

Given five orders, 240 events, and four arms
When Exp6791 finishes
Then it SHALL emit exactly 4,800 unique row keys
And each order-event pair SHALL contain all four arms.

### SCENARIO-CL-6791-RESTART-ROLLBACK: Commits Are Recoverable

Given any accepted transaction
When a fresh store opens the state and applies the inverse patch
Then restart bytes SHALL match committed bytes
And rollback bytes SHALL match the exact parent bytes.

### SCENARIO-CL-6791-ROW-VERDICT: Rows Own Aggregates And Verdicts

Given a complete artifact
When a cold reducer recomputes metrics and positive-credit gates
Then every stored aggregate and verdict SHALL match the recomputed value.

### SCENARIO-CL-6791-BLOCKED: Failed Preconditions Have No Fallback

Given one owned precondition fails
When Exp6791 builds its terminal artifact
Then `rows` and `transaction_receipts` SHALL be empty
And `gate_check_summary` SHALL retain the failed check and observed value.

## Implementation Status (REQ-CL-6791)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6791 and SCENARIO-CL-6791-* | Implemented: `python/carnot/experiment_6791_compositional_online_constraint_routing_ab.py`; `scripts/experiments/experiment_6791_compositional_online_constraint_routing_ab.py`; terminal artifact `results/experiment_6791_compositional_online_constraint_routing_ab.json`. | Implemented: `tests/python/test_experiment_6791_compositional_online_constraint_routing_ab.py`. |

## REQ-CL-6792: Independent CSL Causal And Safety Cold Audit

Given Exp6791 claims a complete compositional self-learning comparison,
When Exp6792 runs for planning date 20260830,
Then it SHALL read checked-in evidence in a fresh CPU process without an LLM.
It SHALL not import Exp6791 producer code or trust its headline aggregates.
It SHALL write
`results/experiment_6792_csl_causal_safety_cold_audit.json`.

Before replay, Exp6792 SHALL require `compositional_csl_completed=true`, one
complete row for each event, arm, and all five orders, exact source hashes,
raw parent and new-state transaction byte snapshots, and exact receipt hashes.
One missing prerequisite SHALL stop the audit before causal or safety replay.
The blocked artifact SHALL use status `complete_blocked_csl_causal_audit`.
Its `gate_check_summary` SHALL name each failed check and its observed value.

A complete audit SHALL recompute writes, later reads, action changes, component
attribution, utility, route cost, retention, hard-case harm, support, and
order-level confidence bounds from raw rows. For each credited factor, it SHALL
replay the later event with only that factor disabled. Causal credit SHALL
require a changed selected route and an exact-receipt utility difference. It
SHALL also replay every event with retrieval disabled.

A complete audit SHALL reject outcome-false poison, relation poison, stale
receipts, duplicate IDs, and tombstone reappearance. No attack SHALL be
admitted or change an action. It SHALL restart at frozen transaction
boundaries, require exact state bytes and next actions, force capacity
eviction, and recheck old-family retention and hard cases after each phase.
It SHALL trigger rollback with one controlled harmful update. Rollback SHALL
restore prior bytes, retrievals, route action, and metrics. Attack evidence
SHALL remain in the artifact.

The artifact SHALL include `schema`, `experiment_id`, `run_date`, `status`,
`field_principles`, `inference_substrate`, `duration_s`, `random_seed`,
`reproducibility_checksum`,
`source_artifact_hashes`, `rows`, `cold_recomputed_metrics`,
`headline_differences`, `credited_factor_count`,
`factors_with_changed_action_witness`, `retrieval_disable_effects`,
`poison_attack_results`, `admitted_poison_count`, `influenced_poison_count`,
`restart_byte_identity`, `restart_action_identity`,
`capacity_eviction_receipts`, `retention_after_phase`,
`hard_case_harm_after_phase`, `rollback_byte_identity`,
`rollback_action_identity`, `source_verdict_supported`,
`csl_causal_audit_completed`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `inference_substrate` SHALL equal
`fresh-process CPU causal and safety replay, no LLM`.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL use only `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
`honest_verdict` SHALL start with an approved terminal prefix.

### SCENARIO-CL-6792-PRECONDITIONS: Missing Raw Bytes Stop The Audit

Given a source omits parent or new-state transaction byte snapshots,
When Exp6792 checks its prerequisites,
Then it SHALL record the missing byte fields and stop before all replay work.

### SCENARIO-CL-6792-COLD-RECOMPUTE: Raw Rows Own Every Metric

Given all prerequisites pass,
When Exp6792 recomputes the comparison,
Then raw event rows and exact receipts SHALL own every metric and order effect.

### SCENARIO-CL-6792-CAUSAL: Credit Requires An Action And Utility Witness

Given one factor receives causal credit,
When only that factor is disabled against the same bytes,
Then the selected route and exact-receipt utility SHALL both change.

### SCENARIO-CL-6792-SAFETY: Attacks Cannot Enter Or Influence Memory

Given each preregistered poison, stale, duplicate, and tombstone attack,
When admission and later retrieval run,
Then admitted and influenced poison counts SHALL both remain zero.

### SCENARIO-CL-6792-DURABILITY: Restart, Eviction, And Rollback Preserve State

Given frozen boundaries, capacity pressure, and one harmful update,
When Exp6792 restarts and rolls back,
Then bytes, next actions, retrievals, retention, hard cases, and metrics SHALL
match their required prior values.

### SCENARIO-CL-6792-TERMINAL: Completion Does Not Depend On Effect Sign

Given all audit rows and controls finish,
When the source effect is positive, null, or rejected,
Then `csl_causal_audit_completed` SHALL be true. Positive support SHALL require
every source, causal, durability, and safety gate to pass.

## Implementation Status (REQ-CL-6792)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6792 and SCENARIO-CL-6792-* | Planned: `python/carnot/experiment_6792_csl_causal_safety_cold_audit.py`; `scripts/experiments/experiment_6792_csl_causal_safety_cold_audit.py`; terminal artifact `results/experiment_6792_csl_causal_safety_cold_audit.json`. | Planned: `tests/python/test_experiment_6792_csl_causal_safety_cold_audit.py`. |

## REQ-CL-6797: Canonical Transaction Byte Replay

Given Exp6791 completed 4,800 frozen comparison cells but omitted state bytes,
When Exp6797 runs for execution date 20260831,
Then it SHALL re-execute the exact Exp6791 mechanism on a deterministic CPU
And it SHALL write
`results/experiment_6797_canonical_transaction_byte_replay.json`.

Exp6797 SHALL verify the checked-in Exp6790 and Exp6791 file hashes before it
starts replay. It SHALL also verify 4,800 unique cells, five frozen order
hashes, four exact arm definitions, 3,189 source commits, one canonical JSON
serializer, and sufficient disk and memory. One failed check SHALL stop all
replay. The blocked artifact SHALL use
`complete_blocked_transaction_byte_replay`. Its `gate_check_summary` SHALL keep
the failed check and observed value. A reduced replay is not permitted.

Exp6797 SHALL use the Exp6791 arms, seeds, orders, thresholds, capacity,
opportunity stream, update cadence, action policy, and controls without a new
learner. Each arm-order store SHALL remain isolated. The active event SHALL be
read-only. A write SHALL occur only after the exact receipt for that event.

Each committed receipt SHALL carry a reversible base64 encoding of the exact
canonical parent state bytes and new state bytes. It SHALL carry matching
SHA-256 hashes, order, arm, event, position, transaction identity, chain index,
chain predecessor, and receipt hash. Exp6797 SHALL decode both snapshots and
verify both hashes at the commit boundary. It SHALL never infer state bytes
from a hash.

Exp6797 SHALL use a parent-owned atomic checkpoint outside worker temporary
directories. The parent SHALL stop the first worker after one fixed complete
order prefix. A fresh process SHALL resume only pending orders. The resume
SHALL skip each complete prefix cell exactly once. A changed manifest,
conflicting row, incomplete row, or duplicate commit SHALL not change accepted
checkpoint bytes.

Exp6797 SHALL pair all replay rows with Exp6791. Routes, rewards, utilities,
reads, writes, and transaction decisions SHALL match. Row reduction SHALL
reproduce 1,063 compositional online writes, 3,132 compositional online later
reads, 721 compositional online action changes, 3,189 total commits, and all
stored order summaries. These values are identity checks, not new scientific
claims.

The artifact SHALL include `schema`, `experiment_id`, `run_date`, `status`,
`field_principles`, `inference_substrate`, `duration_s`, `random_seed`,
`reproducibility_checksum`,
`source_artifact_hashes`, `frozen_manifest`, `arm_definitions`, `order_hashes`,
`checkpoint_receipts`, `transaction_schema`, `transaction_receipts`,
`committed_transaction_count`, `parent_byte_snapshot_count`,
`new_state_byte_snapshot_count`, `byte_hash_match_count`,
`replay_identity_checks`, `attack_results`, `rows`,
`transaction_byte_snapshot_fixture_ready`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.
`inference_substrate` SHALL equal
`deterministic CPU transactional replay, no LLM`.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL use only `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
`honest_verdict` SHALL start with an approved terminal prefix.

### SCENARIO-CL-6797-CANONICAL-BYTES: Encoding Round-Trips Exact State

Given one committed transaction boundary,
When its parent and new state snapshots are encoded and decoded,
Then both decoded byte strings SHALL equal the originals
And both stored hashes SHALL equal hashes recomputed from decoded bytes.

### SCENARIO-CL-6797-BYTE-CHAIN: Every Commit Extends Its Owned Store

Given committed receipts for one arm and order,
When Exp6797 verifies them in commit order,
Then each parent snapshot SHALL equal the prior new snapshot
And each predecessor SHALL equal the prior receipt hash.

### SCENARIO-CL-6797-FRESH-RESUME: A Fixed Prefix Survives Interruption

Given the first worker has published one complete order,
When the parent interrupts it and starts a fresh process,
Then 960 complete cells SHALL be skipped exactly once
And the new process SHALL publish only the four pending orders.

### SCENARIO-CL-6797-IDENTITY: Replay Preserves Exp6791 Evidence

Given all five replay orders are complete,
When rows and transactions are paired with Exp6791,
Then all 4,800 cells and all order summaries SHALL be identical
And the row reducer SHALL reproduce the frozen activity counts.

### SCENARIO-CL-6797-ATTACKS: Corruption Cannot Change Accepted Bytes

Given byte-flip, receipt-reorder, stale-parent, wrong-arm, duplicate-commit,
cross-arm-access, interrupted-write, and manifest-mismatch attacks,
When each attack reaches its validation boundary,
Then each attack SHALL fail closed
And accepted committed bytes SHALL remain unchanged.

### SCENARIO-CL-6797-BLOCKED: A Failed Precondition Has No Replay Rows

Given one authority, identity, serializer, or resource check fails,
When Exp6797 builds its terminal artifact,
Then rows and transaction receipts SHALL be empty
And `gate_check_summary` SHALL name the failed check and observed value.

## Implementation Status (REQ-CL-6797)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6797 and SCENARIO-CL-6797-* | Planned: `python/carnot/experiment_6797_canonical_transaction_byte_replay.py`; `scripts/experiments/experiment_6797_canonical_transaction_byte_replay.py`; terminal artifact `results/experiment_6797_canonical_transaction_byte_replay.json`. | Planned: `tests/python/test_experiment_6797_canonical_transaction_byte_replay.py`. |

## REQ-CL-6798: Byte-Grounded CSL Causal And Safety Cold Audit

Given Exp6797 supplies canonical parent and new state bytes for each committed
transaction,
When Exp6798 runs for execution date 20260831,
Then it SHALL audit the complete replay in a fresh deterministic CPU process
And it SHALL not import Exp6797 or Exp6791 producer code
And it SHALL write
`results/experiment_6798_csl_causal_safety_byte_audit.json`.

Before replay, Exp6798 SHALL require
`transaction_byte_snapshot_fixture_ready=true`, the exact Exp6797 artifact
hash, its exact Exp6790 and Exp6791 source hashes, 4,800 unique rows, five
order hashes, 3,189 committed transactions, 3,189 parent snapshots, 3,189 new
snapshots, and 3,189 matching byte hashes. One failed check SHALL stop the
audit. The artifact SHALL use a `complete_blocked_*` status and verdict. Its
`rows` and `chain_replay_receipts` SHALL be empty. Its `gate_check_summary`
SHALL keep every failed check with expected and observed values. A reduced
audit is not permitted.

Exp6798 SHALL implement its own canonical serializer, hash functions, state
decoder, route evaluator, factor retrieval, route selector, metric reducer,
and confidence calculation. It SHALL decode each committed parent and new
snapshot. It SHALL verify both hashes. It SHALL verify each isolated chain.
It SHALL replay all 4,800 actions from decoded state and legal observations.
It SHALL recompute exact route success, route cost, utility, writes, later
reads, action changes, held-future effects, retention, hard-case harm, action
support, and the five-order confidence lower bound from rows and exact
receipts. Producer headline values SHALL be comparison inputs only.

For each credited factor, Exp6798 SHALL replay a later event from its parent
state bytes with only that factor removed. It SHALL also replay that event
with retrieval disabled. Credit SHALL require a changed route and a nonzero
exact-receipt utility difference. Writes without both witnesses SHALL remain
uncredited and SHALL be reported separately.

Exp6798 SHALL inject future receipts, poisoned factors, stale parents,
valid bytes owned by the wrong arm, capacity pressure, eviction reorder, byte
corruption, and duplicate commits. No poisoned or invalid factor SHALL enter
active state or change an action. Capacity pressure SHALL emit eviction
receipts. Exp6798 SHALL restart at every preregistered transaction boundary
and compare state bytes and next actions. Retention or hard-case harm SHALL
trigger rollback. Rollback SHALL restore the declared parent bytes and action.

The artifact SHALL include `schema`, `experiment_id`, `run_date`, `status`,
`field_principles`, `inference_substrate`, `duration_s`, `random_seed`,
`reproducibility_checksum`, `source_artifact_hashes`,
`transaction_byte_counts`,
`chain_replay_receipts`, `cold_recomputed_metrics`,
`headline_differences`, `factors_with_changed_action_witness`,
`credited_factor_count`, `retrieval_disable_effects`,
`poison_attack_results`, `admitted_poison_count`,
`influenced_poison_count`, `capacity_eviction_receipts`,
`restart_byte_identity`, `restart_action_identity`,
`rollback_byte_identity`, `rollback_action_identity`,
`retention_after_phase`, `hard_case_harm_after_phase`, `rows`,
`source_verdict_supported`, `csl_causal_audit_completed`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `inference_substrate` SHALL equal
`independent deterministic CPU byte replay, no LLM`.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL use only
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL start with an approved terminal prefix.

`csl_causal_audit_completed` SHALL be true after the full audit even when the
effect is null. A positive result SHALL require every source, chain, replay,
causal, safety, capacity, restart, rollback, retention, and hard-case gate to
pass. A complete replay with no credited effect SHALL use `null`. A source
contradiction, future leakage, admitted poison, influenced poison, byte
corruption, or chain corruption SHALL use `disqualified`.

### SCENARIO-CL-6798-PRECONDITIONS: Missing Snapshot Evidence Stops Replay

Given one fixture flag, hash, row, order, commit, snapshot, or byte hash is
missing,
When Exp6798 evaluates preconditions,
Then it SHALL write a complete blocked artifact with no replay rows
And every failed check SHALL keep its expected and observed values.

### SCENARIO-CL-6798-SERIALIZER: Independent Bytes Round-Trip Exactly

Given a canonical state snapshot,
When Exp6798 decodes and serializes it without producer code,
Then the new bytes SHALL equal the stored bytes
And its SHA-256 hash SHALL equal the stored hash.

### SCENARIO-CL-6798-CHAIN-REPLAY: Every State And Action Replays

Given all committed transactions and legal event observations,
When Exp6798 reconstructs each arm-order chain,
Then every parent SHALL extend the prior new state
And every replayed action and exact utility SHALL match its row.

### SCENARIO-CL-6798-CAUSAL-CREDIT: Credit Needs Two Exact Witnesses

Given one stored factor is read by a later event,
When Exp6798 removes only that factor from the later parent bytes,
Then it SHALL credit the factor only if the action changes
And the exact receipt gives a nonzero utility difference.

### SCENARIO-CL-6798-ATTACKS: Invalid State Cannot Enter Or Influence

Given each preregistered leakage, poison, stale, ownership, capacity, reorder,
corruption, and duplicate attack,
When Exp6798 applies its independent admission and chain checks,
Then invalid state SHALL be rejected without changing accepted bytes or action.

### SCENARIO-CL-6798-RESTART-ROLLBACK: Recovery Restores Bytes And Actions

Given every preregistered restart boundary and one harm-triggering capacity
phase,
When Exp6798 restarts and rolls back,
Then restart and rollback bytes SHALL be exact
And the next selected actions SHALL be exact.

### SCENARIO-CL-6798-TERMINAL: Completion Is Independent Of Effect Sign

Given all replay and attack rows complete,
When Exp6798 classifies the result,
Then `csl_causal_audit_completed` SHALL be true for positive or null effects
And positive SHALL require all causal and safety gates.

## Implementation Status (REQ-CL-6798)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-CL-6798 and SCENARIO-CL-6798-* | Implemented: `python/carnot/experiment_6798_csl_causal_safety_byte_audit.py`; `scripts/experiments/experiment_6798_csl_causal_safety_byte_audit.py`; terminal artifact `results/experiment_6798_csl_causal_safety_byte_audit.json`. | Implemented: `tests/python/test_experiment_6798_csl_causal_safety_byte_audit.py`; 658/658 statements covered. |

## REQ-CL-6810: V595 Transactional Verified Memory Ownership

The continuous-learning capability SHALL own the V595
`transactional verified memory` contract. The implementing task is
`exp6816-residual-pressure-route-learning-ab`. It SHALL write
`results/experiment_6816_residual_pressure_route_learning_ab.json`.
The downstream completion gate is `residual_route_learning_completed`.

An active episode SHALL read one immutable memory snapshot. Typed add, revise,
soft-delete, retrieve, filter, and restore operations SHALL commit only between
episodes. The exact local receipt remains authoritative. Each commit SHALL bind
canonical parent bytes, new bytes, hashes, operation identity, predecessor,
inverse or rollback target, and the later read that could influence an action.
Model weights SHALL remain unchanged.

The memory SHALL fail closed on an active-episode write, invalid operation,
stale parent, future evidence, family label, poison, duplicate operation,
capacity breach, support contraction, retention harm, hard-case harm, restart
mismatch, or rollback mismatch. A failed owned precondition SHALL produce
`complete_blocked_residual_pressure_route_learning_ab`. The artifact SHALL name
the failed check, expected value, and observed value in `gate_check_summary`.
The completion field SHALL remain false and the parent bytes SHALL remain
active.

### SCENARIO-CL-6810-RECEIPT-BEFORE-COMMIT

**Given** an immutable active-episode snapshot and a proposed typed operation
**When** the episode closes
**Then** no durable bytes change until the exact local receipt passes every
admission and safety check.

### SCENARIO-CL-6810-ROLLBACK-IDENTITY

**Given** a committed operation causes support, retention, or hard-case harm
**When** the rollback boundary runs
**Then** the restored memory and next action match the canonical parent bytes
and action exactly.

## Implementation Status (REQ-CL-6810)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6810 and SCENARIO-CL-6810-* | Planned: Exp6816 and `results/experiment_6816_residual_pressure_route_learning_ab.json`. | Planned after Exp6810 contract preflight. |

## REQ-CL-6827: Frozen Chronological Causal-Edge Memory Stream

Exp6827 SHALL transform the complete Exp6812 authentic operational-handoff
corpus and the terminal Exp6826 sealed-arbiter receipt on CPU. It SHALL not
invoke an LLM or learn parameters. It SHALL write
`results/experiment_6827_chronological_causal_edge_memory_stream.json` through
`scripts/experiments/experiment_6827_chronological_causal_edge_memory_stream.py`.
The implementation SHALL live in
`python/carnot/experiment_6827_chronological_causal_edge_memory_stream.py`.

Before stream construction, Exp6827 SHALL require
`selective_arbiter_audit_complete=true`, all three mandated GGUF families, all
288 raw-output hashes, terminal component decisions, unique chronological
keys, at least one legal alternative per event, and decision snapshots with no
future field. It SHALL bind both source artifacts by their raw file hashes. A
failed check SHALL stop construction. The terminal artifact SHALL use status
`complete_blocked_chronological_causal_edge_memory_stream`, contain no rows,
set `verified_memory_stream_ready=false`, and report each failed check with its
expected and observed values in `gate_check_summary`.

The public operation grammar SHALL contain typed `add`, `revise`,
`soft_delete`, `retrieve`, `filter`, and `restore` operations. Each operation
SHALL declare an exact precondition, deterministic effect, inverse or rollback
target, and canonical bytes. Memory SHALL have equal finite capacity for every
family and order. Active decision snapshots SHALL be deeply read-only. A
soft-deleted record SHALL be hidden from normal retrieval. Only the sealed
harness authority SHALL restore it. Safe missing retrievals and empty filters
SHALL be accepted no-ops. Conflicts, stale revisions, unauthorized restores,
and capacity breaches SHALL be rejected without changing state bytes.

Exp6827 SHALL freeze exactly five deterministic chronological orders before
stream evaluation. It SHALL keep memory state isolated by source family and
SHALL not pool family observations. It SHALL freeze development and held-future
partitions, a hard-case manifest, and three leave-one-family-out rotations.
Each decision snapshot SHALL contain only the feature allowlist and past memory
state. Outcome, utility, support, retention, hard-case, later-read, audit,
family-oracle, and final-acceptance fields SHALL remain sealed until after the
decision. Final acceptance values SHALL remain in a hidden harness view.

The stream SHALL generate typed write-read-action-outcome candidate edges. For
each event, order, source family, and counterfactual kind, it SHALL emit one
public row. The counterfactual kinds SHALL be exactly `remove`, `substitute`,
and `reorder`. Inapplicable units SHALL be explicit safe no-ops. Applicable
units SHALL preserve the baseline edge identity and report the public
counterfactual transformation without exposing hidden final acceptance.

The complete stream SHALL contain nonzero legal alternatives, safe no-op
events, conflicts, admissible writes, rejected writes, later retrieval
opportunities, capacity pressure, and stale-pressure recovery. Readiness SHALL
depend only on complete chronology, positive finite headroom, sealed-field
isolation, and canonical-byte validation. It SHALL not depend on the Exp6826
deployment adoption outcome or on a favorable causal effect.

The artifact SHALL include `schema`, `experiment_id`, `title`, `run_date`,
`status`, `openspec_requirement_ids`, `replay_commands`, `field_principles`,
`inference_substrate`, `duration_s`, `random_seed`,
`reproducibility_checksum`, `source_artifact_hashes`, `operation_schema`,
`local_receipt_schema`, `capacity_contract`, `split_manifest`,
`sealed_field_manifest`, `order_hashes`, `feature_allowlist`,
`feature_denylist`, `causal_edge_schema`, `counterfactual_manifest`, `rows`,
`headroom_metrics`, `admissible_operation_count`,
`rejected_operation_count`, `later_read_opportunity_count`,
`verified_memory_stream_ready`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL contain one
plain-language principle for every top-level field. `inference_substrate` SHALL
equal `CPU transformation of frozen authentic outputs, no LLM`.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL use only `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
`honest_verdict` SHALL start with an approved terminal prefix.

### SCENARIO-CL-6827-PRECONDITIONS: Missing Frozen Evidence Stops Construction

Given one audit, family, raw hash, terminal decision, chronological key, legal
alternative, or no-future-field check fails,
When Exp6827 checks its two frozen inputs,
Then it SHALL emit the complete blocked artifact with no rows
And the gate summary SHALL record the failed check, expected value, and
observed value.

### SCENARIO-CL-6827-OPERATIONS: Typed Effects And Receipts Are Exact

Given a family-isolated fixed-capacity memory and one public typed operation,
When Exp6827 evaluates the operation,
Then its exact preconditions SHALL determine admission before mutation
And its receipt SHALL bind canonical parent bytes, operation bytes, new bytes,
effect, and inverse or rollback target.

### SCENARIO-CL-6827-VISIBILITY: Delete And Restore Preserve Authority

Given an active record is soft-deleted,
When normal retrieval or restoration is requested,
Then normal retrieval SHALL hide the record
And only sealed harness authority SHALL restore it without a capacity breach.

### SCENARIO-CL-6827-SNAPSHOTS: Decisions Read Past State Only

Given one chronological event boundary,
When Exp6827 freezes its decision snapshot,
Then the snapshot SHALL reject nested mutation
And its public feature keys SHALL be disjoint from the feature denylist.

### SCENARIO-CL-6827-ORDERS: Family-Isolated Chronology Is Complete

Given all 288 authentic events,
When Exp6827 freezes five orders and all partitions,
Then each order SHALL contain every event exactly once within its source family
And the held-future and leave-one-family-out manifests SHALL remain unchanged.

### SCENARIO-CL-6827-COUNTERFACTUALS: Three Causal Edge Tests Are Frozen

Given an accepted write with a later read-action witness,
When Exp6827 emits counterfactual units,
Then remove SHALL omit the write, substitute SHALL use its legal alternative,
And reorder SHALL move the read before the write without changing source-family
ownership.

### SCENARIO-CL-6827-SERIALIZATION: Canonical Bytes Are Stable

Given identical operation, state, row, and command inputs,
When Exp6827 serializes or hashes them twice,
Then canonical bytes and SHA-256 values SHALL match exactly
And the reproducibility checksum SHALL exclude measured wall time.

### SCENARIO-CL-6827-READINESS: Completion Is Independent Of Adoption

Given complete rows, positive headroom, sealed fields, and canonical bytes,
When Exp6827 computes `verified_memory_stream_ready`,
Then readiness SHALL be true for every terminal Exp6826 adoption decision
And SHALL not imply that the stream verifier is an oracle.

## Implementation Status (REQ-CL-6827)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6827 and SCENARIO-CL-6827-* | Implemented: `python/carnot/experiment_6827_chronological_causal_edge_memory_stream.py`, task-owned wrapper, and terminal artifact. | Implemented: 107-test conductor-equivalent shard passes; Exp6827 module and wrapper have 514/514 statements covered; focused spec coverage, artifact validation, adversarial verification, verdict-row lint, Ruff, format, mypy, and root-clutter checks pass. |

## REQ-CL-6828: Residual-Pressure Verified-Memory A/B

Exp6828 SHALL compare four equal-capacity external-memory arms over the full
Exp6827 stream. The arms SHALL be frozen memory, finite-gain residual pressure,
raw signed-violation accumulation, and seeded random exact-valid updates. The
experiment SHALL use all five frozen orders and all frozen development and
held-future events. It SHALL not invoke an LLM or change a GGUF weight. It SHALL
write `results/experiment_6828_residual_pressure_verified_memory_ab.json`
through
`scripts/experiments/experiment_6828_residual_pressure_verified_memory_ab.py`.
The implementation SHALL live in
`python/carnot/experiment_6828_residual_pressure_verified_memory_ab.py`.

Before evaluation, Exp6828 SHALL require
`verified_memory_stream_ready=true`, complete order and split manifests,
positive finite headroom, admissible and rejected operations, later reads, one
canonical serializer, and sufficient disk and RAM. It SHALL also require the
Exp6827 source hash. A failed check SHALL stop the complete stream. The blocked
artifact SHALL use `complete_blocked_residual_pressure_verified_memory_ab`,
contain no rows or transactions, set `route_learning_ab_complete=false`, and
keep each failed check, expected value, and observed value in
`gate_check_summary`. It SHALL not shrink or replace the stream.

Exp6828 SHALL freeze capacity, gains, decay, update cadence, proposals,
eviction, five order hashes, seeds, and all acceptance thresholds before the
first event. Public constants SHALL use development rows only. Each active
episode SHALL select an action from one immutable snapshot. The exact local
receipt SHALL become visible only after the action. A memory proposal SHALL
occur only after the episode ends. A commit SHALL occur only after the proposal
passes its exact receipt and the external sealed support harness.

The residual arm SHALL update projected pressure with a finite gain and decay.
The raw arm SHALL accumulate the frozen signed violation signal without a
decay term. The random arm SHALL use only seeded exact-valid proposals. The
frozen arm SHALL not write. Every arm SHALL declare the same finite capacity,
cadence, proposal budget, eviction rule, and action threshold. The active
episode SHALL reject every write attempt.

Before each commit, the sealed harness SHALL check current utility, protected
retention anchors, hard cases, and held-future reachable support. It SHALL
return only accept or reject to the learner. A rejected update SHALL preserve
the parent bytes. Threshold harm after a tentative commit SHALL restore the
exact parent bytes and parent action. Held-future fields, exact future
receipts, family-oracle fields, and final acceptance SHALL never enter a public
decision snapshot.

Every transaction SHALL preserve canonical parent and new bytes, their hashes,
the predecessor receipt, arm, order, event, operation diff, pressure state,
residual state, and exact admission receipt. Task-owned checkpoints SHALL bind
complete frozen boundaries. A restarted store SHALL recover the same bytes,
next action, and predecessor. Rollback SHALL restore the exact parent bytes.

Exp6828 SHALL test credited write-read-action-outcome edges by removing,
substituting, and reordering them. Causal credit SHALL require an admitted
write, an actual later retrieval, a changed action, and a nonzero exact outcome
effect. An operation without this full chain SHALL remain uncredited.

The positive gate SHALL require a paired held-future lower confidence bound
above zero, no support contraction, no protected-retention harm, no hard-case
harm, nonzero causal-factor witnesses, stale-pressure release, and complete
transaction bytes. `route_learning_ab_complete` SHALL depend only on complete
rows, transactions, checkpoints, and teardown. It SHALL not depend on effect
sign. This field is the exact completion input for Exp6829 and Exp6830.

The artifact SHALL include `schema`, `experiment_id`, `title`, `run_date`,
`status`, `openspec_requirement_ids`, `replay_commands`, `field_principles`,
`inference_substrate`, `duration_s`, `random_seed`,
`reproducibility_checksum`, `continuous_self_learning_task`, `MODEL_SPECS`,
`model_weight_immutability_receipt`, `source_artifact_hashes`,
`frozen_manifest`, `arm_definitions`, `residual_update_contract`,
`checkpoint_receipts`, `rows`, `transaction_schema`,
`transaction_receipts`, `commits_by_arm`, `rejects_by_arm`,
`later_reads_by_arm`, `action_influence_by_arm`, `exact_utility_by_arm`,
`held_future_support_by_arm`, `retention_by_arm`,
`hard_case_effect_by_arm`, `stale_pressure_release_by_arm`,
`causal_edge_counterfactuals`, `paired_residual_deltas`,
`causal_factor_witnesses`, `rollback_receipts`,
`acceptance_gate_positive`, `route_learning_ab_complete`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL contain one principle for every
top-level field. `inference_substrate` SHALL equal
`CPU external program learning over frozen authentic outputs`. `MODEL_SPECS`
SHALL contain the three source GGUF families. `verifier_is_oracle` SHALL be
false. `verdict_class` SHALL use only `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL use an approved
terminal prefix and SHALL agree with row-derived gates.

### SCENARIO-CL-6828-PRECONDITIONS: An Owned Failure Stops The Full Stream

Given one readiness, order, split, headroom, operation, read, serializer,
source-hash, disk, or RAM check fails,
When Exp6828 evaluates preconditions,
Then it SHALL write the complete blocked artifact without rows or transactions
And the gate summary SHALL preserve the failed check and its observed value.

### SCENARIO-CL-6828-UPDATES: Residual And Raw Pressure Stay Distinct

Given the same frozen signed receipt sequence,
When the two learning arms update pressure,
Then residual pressure SHALL use the finite-gain projection and decay
And raw pressure SHALL use cumulative signed violations without decay.

### SCENARIO-CL-6828-EPISODES: Active Episodes Are Read-Only

Given one arm has frozen its episode snapshot,
When any code attempts a memory write before the exact outcome is revealed,
Then the store SHALL reject that write
And its canonical state bytes SHALL remain unchanged.

### SCENARIO-CL-6828-ADMISSION: Commits Follow Exact And Sealed Receipts

Given an episode has ended and its exact local receipt is available,
When an arm proposes a memory operation,
Then commit SHALL require exact-valid operation bytes and sealed support
acceptance
And a rejected or harmful proposal SHALL preserve or restore its parent bytes.

### SCENARIO-CL-6828-CAPACITY: All Arms Have One Frozen Budget

Given the four arm definitions,
When Exp6828 creates isolated order stores,
Then every store SHALL use the same capacity, cadence, proposal limit, and
eviction rule
And the random arm SHALL never bypass exact admission.

### SCENARIO-CL-6828-TRANSACTIONS: Canonical Lineage Survives Recovery

Given one admitted operation and one frozen checkpoint boundary,
When the store restarts or rolls back,
Then parent and new bytes SHALL match their hashes and predecessor
And rollback SHALL restore the exact parent bytes and next action.

### SCENARIO-CL-6828-FUTURE-SEAL: Future Fields Cannot Guide An Action

Given a public event snapshot,
When the action policy reads its keys,
Then no denied future, outcome, support, audit, or acceptance key SHALL exist
And the sealed harness SHALL return only one acceptance bit to the learner.

### SCENARIO-CL-6828-CAUSAL-CREDIT: Credit Needs The Complete Later Chain

Given an admitted write is read by a later event,
When remove, substitute, and reorder counterfactuals run,
Then credit SHALL require a changed action and a nonzero exact outcome effect
And every incomplete chain SHALL remain uncredited.

### SCENARIO-CL-6828-VERDICT: Completion Does Not Depend On Effect Sign

Given all order-arm-event rows, transactions, checkpoints, and teardown
receipts are complete,
When Exp6828 computes its terminal fields,
Then `route_learning_ab_complete` SHALL be true for positive or null effects
And positive SHALL require every predeclared gain, support, safety, causal,
release, and byte-completeness gate.

## Implementation Status (REQ-CL-6828)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6828 and SCENARIO-CL-6828-* | Planned: Exp6828 module, task-owned wrapper, and terminal artifact. | Planned: focused tests and full Python suite. |

## REQ-CL-6831: V597 Causal-Edge Stream Input Admissibility

The system SHALL validate Exp6827 as an immutable input stream without running
learning. Validation SHALL cover all 4,320 row identities and hash fields, order
hashes, family split membership, canonical three-row counterfactual
transactions, family rotation, the sealed-field manifest, and recomputed
nonzero headroom counts. Parent/new hashes SHALL preserve rejected and read-only
transactions and SHALL change only for admitted mutating operations.

The system SHALL separately confirm that the upstream artifact declares no
learning ran and exposes no weight-update receipt. Exp6831 SHALL emit one row
for every stream-readiness criterion and derive `csl_inputs_admissible` and
`v597_contract_ready` from complete admissible inputs, not from any learned
effect. These exact booleans SHALL be suitable for downstream Exp6835 and
Exp6832 consumption respectively.

### SCENARIO-CL-6831-ROW-HASHES: Every Stream Row Has Stable Identity

Given the frozen Exp6827 rows,
When Exp6831 validates row and state hashes,
Then all rows SHALL have unique canonical identities and every declared digest
SHALL have the required SHA-256 form.

### SCENARIO-CL-6831-ORDERS-AND-SPLITS: Order And Split Identity Recompute

Given the five frozen order manifests and three family splits,
When Exp6831 reconstructs their identities from rows,
Then every order hash, event position, split label, and split count SHALL agree.

### SCENARIO-CL-6831-TRANSACTIONS: Counterfactual Rows Form Canonical Triples

Given one order-family-operation source unit,
When Exp6831 validates its transactions,
Then exactly observed, removal, and alternative rows SHALL agree on receipt and
state identity while preserving the canonical parent-to-new chain.

### SCENARIO-CL-6831-ROTATIONS-AND-SEALS: Held-Out Families And Fields Stay Sealed

Given the three leave-one-family-out rotations,
When Exp6831 validates them,
Then every family SHALL be held out once and no denied future or outcome field
SHALL appear in the decision feature manifest or stream rows.

### SCENARIO-CL-6831-HEADROOM: Readiness Requires Recomputed Nonzero Opportunity

Given the canonical operation stream,
When Exp6831 recomputes capacity, conflict, later-read, legal-alternative,
safe-no-op, and stale-recovery opportunities,
Then every required count SHALL be nonzero and SHALL equal the source summary.

### SCENARIO-CL-6831-NO-LEARNING: Validation Cannot Become A Learning Claim

Given immutable Exp6827 source bytes,
When Exp6831 validates learning preconditions,
Then it SHALL confirm immutable weights and the upstream no-learning statement
and SHALL not claim that training or weight updates occurred.

## Implementation Status (REQ-CL-6831)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6831 and SCENARIO-CL-6831-* | Implemented: immutable stream validator within Exp6831. | Implemented: focused row, order, split, transaction, seal, headroom, and no-learning mutation tests pass. |

## REQ-CL-6836: Typed Obligation Memory Admission Guard

The system SHALL expose the Exp6836 typed obligation program as a deterministic
external-memory admission guard. The guard SHALL compile from the same
immutable operational-obligation atoms as the constraint energy and predicate.
It SHALL not call an LLM and SHALL not read or update GGUF model weights.

The memory admission guard SHALL accept only candidates with zero exact atom
energy, complete satisfaction, and no diagnostic failures. It SHALL reject
parse failures, omitted atoms, contradictions, impossible sets, unknown
actions, and any candidate whose atom identities do not match the compiled
program. Rejection SHALL be fail closed. The guard SHALL return explicit
per-atom diagnostics so a later learner can store a cause without storing a
model score.

The guard SHALL share candidate identity, label-swap, token-length,
prompt-length, row-order, and surface-form controls with the Exp6836
candidate-pair fixture. These controls SHALL make later continuous-learning
measurements distinguish exact admission from shortcut features. The guard
SHALL remain a verifier input. It SHALL not be an oracle and SHALL not convert
readiness into a positive learning result.

### SCENARIO-CL-6836-MEMORY-GUARD: Admission Uses Exact Atom Energy

Given a compatible fixed-sequence candidate and a one-atom-violating candidate,
When the typed program evaluates memory admission,
Then the compatible candidate SHALL be admitted, the violation SHALL be
rejected, and both outcomes SHALL share the same atom identities as the
constraint predicate.

### SCENARIO-CL-6836-FAIL-CLOSED: Unsafe Or Drifted Candidates Do Not Commit

Given parse failure, omitted atom, contradiction, impossible set, unknown
action, or atom-identity drift,
When a later learner asks the guard for admission,
Then the guard SHALL reject the candidate and return a diagnostic cause without
producing or consuming a model score.

### SCENARIO-CL-6836-CONTROLS: Learning Inputs Keep Shortcut Controls

Given Exp6836 candidate rows,
When the memory surface prepares later scoring inputs,
Then candidate id, label swap, row order, token length, prompt length, and
surface form SHALL remain explicit and hash-bound.

## Implementation Status (REQ-CL-6836)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6836 and SCENARIO-CL-6836-* | Planned: deterministic guard compiled from the Exp6836 typed obligation program. | Planned: focused guard, failure, and control tests. |

## REQ-CL-6839: Bounded Residual-Memory Kernel Canary

The system SHALL implement a deterministic CPU residual-memory kernel canary
over the frozen Exp6827 chronological causal-edge stream. The canary SHALL
invoke no LLM, mutate no model weights, and run only a bounded 96-event fixed
slice before any full-stream comparison. It SHALL compare exactly four arms:
`no_memory`, `read_only_memory`, `random_admission`, and
`verified_residual_admission`.

Before any canary transition, Exp6839 SHALL require
`v598_evidence_root_ready_score=1`, the complete 4,320-row Exp6827 stream,
stable order and split hashes, exact later outcome identities for the 96-event
slice, and nonzero decision headroom. A failed check SHALL write a terminal
blocked artifact with `complete_blocked_residual_memory_kernel` in
`honest_verdict`, no transition rows, zero readiness scores, and failed checks
with observed values in `gate_check_summary`.

Each arm-event transition SHALL freeze the proposed action and action dose
before revealing the later outcome. The row SHALL store the event identity,
arm, frozen decision hash, proposed action, exact later outcome identity, signed
outcome direction, action dose, admission decision, and post-transition state
hash. Memory admission SHALL occur only after the exact later outcome is
available. Action-level credit SHALL be assigned only from the external later
outcome direction, the pre-reveal action dose, and the arm-local admitted
memory state.

The kernel SHALL keep state bounded, deterministic, and restartable. It SHALL
persist canonical state bytes, force one restart, force one stale expiry, force
one invalid update rejection, force one rollback, reject duplicate events
without state mutation, and recover from a simulated partial checkpoint by
loading the last complete checkpoint. The final state hash SHALL match a clean
replay hash. `residual_memory_kernel_ready_score` SHALL depend only on
state-machine, persistence, restart, expiry, rollback, duplicate, and replay
correctness. It SHALL not claim held-future benefit from the canary.

The artifact SHALL include `schema`, `experiment_id`, `title`, `run_date`,
`status`, `openspec_requirement_ids`, `replay_commands`, `field_principles`,
`preconditions_checked`, `inference_substrate`, `duration_s`,
`continuous_self_learning_task`, `source_artifact_hashes`, `random_seed`,
`reproducibility_checksum`, `kernel_state_schema`, `rows`,
`exact_outcome_credit_rows`, `admission_rows`,
`capacity_and_eviction_receipts`, `stale_decay_receipts`, `restart_receipt`,
`rollback_receipt`, `clean_replay_state_hash`,
`residual_memory_kernel_ready_score`,
`csl_kernel_execution_complete_score`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.
`field_principles` SHALL contain one principle for every top-level field.
`inference_substrate` SHALL equal
`CPU prospective Tier-2 constraint-memory controller, no LLM`.
`continuous_self_learning_task` SHALL be true. `verifier_is_oracle` SHALL be
false. `verdict_class` SHALL use only `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL be terminal and
start with `complete_`.

### SCENARIO-CL-6839-PRECONDITIONS: Missing Frozen Inputs Block The Canary

Given missing V598 readiness, an incomplete Exp6827 stream, drifted order or
split hashes, missing later outcomes, or zero decision headroom,
When Exp6839 checks its frozen sources,
Then it SHALL emit `complete_blocked_residual_memory_kernel` with no transition
rows
And `gate_check_summary` SHALL report each failed check and observed value.

### SCENARIO-CL-6839-CHRONOLOGY: Decisions Freeze Before Later Outcomes

Given one fixed canary event and one arm-local state snapshot,
When Exp6839 proposes an action,
Then the decision hash and dose SHALL be frozen before the later outcome is
revealed
And the decision inputs SHALL exclude exact outcome and admission fields.

### SCENARIO-CL-6839-PROPOSAL-IDENTITY: Public Inputs Own Action Identity

Given identical public event, arm, seed, and parent state hash,
When Exp6839 proposes twice,
Then the proposed action, action dose, and proposal hash SHALL match exactly
And a changed arm or parent state SHALL change the proposal hash.

### SCENARIO-CL-6839-CREDIT: Exact Later Outcome Direction Owns Credit

Given a frozen pre-reveal action and an external later outcome identity,
When Exp6839 evaluates credit,
Then signed credit SHALL equal signed direction times bounded action dose
And zero direction or zero dose SHALL produce zero credited effect.

### SCENARIO-CL-6839-BOUNDS: Admission, Dose, Capacity, And Eviction Are Bounded

Given a memory arm reaches capacity or proposes an out-of-bound dose,
When Exp6839 evaluates admission,
Then invalid proposals SHALL be rejected without state mutation
And accepted entries SHALL remain within dose bounds while deterministic
eviction keeps active entries at or below capacity.

### SCENARIO-CL-6839-STALE: Stale Decay Expires Old Entries

Given admitted entries older than the stale horizon,
When Exp6839 advances chronology,
Then stale entries SHALL expire with a receipt
And the post-expiry state hash SHALL bind the reduced state.

### SCENARIO-CL-6839-RECOVERY: Persistence, Restart, Rollback, And Crash Recovery Are Exact

Given persisted canonical state bytes and a simulated partial checkpoint,
When Exp6839 restarts or rolls back,
Then it SHALL load the last complete checkpoint, ignore the partial file,
restore exact parent bytes on rollback, and match the clean replay final hash.

### SCENARIO-CL-6839-DUPLICATES: Duplicate Event Delivery Is Idempotent

Given an already processed arm-event pair,
When Exp6839 receives it again,
Then the duplicate SHALL be rejected without changing state bytes.

### SCENARIO-CL-6839-VERDICT: Readiness Does Not Claim Held-Future Benefit

Given all canary rows, admission receipts, stale receipts, restart receipts,
rollback receipts, duplicate checks, and clean replay hashes are complete,
When Exp6839 computes terminal fields,
Then `residual_memory_kernel_ready_score` SHALL be one from state-machine
correctness only
And `verdict_class` SHALL remain `null` unless a later full comparison proves
held-future benefit.

## Implementation Status (REQ-CL-6839)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6839 and SCENARIO-CL-6839-* | Planned: `python/carnot/experiment_6839_bounded_residual_memory_kernel.py`; `scripts/experiments/experiment_6839_bounded_residual_memory_kernel.py`; `results/experiment_6839_bounded_residual_memory_kernel.json`. | Planned: focused state-machine, artifact, coverage, lint, and deterministic audit checks. |

## REQ-CL-6840: Residual-Memory Chronological Shard A

Exp6840 SHALL run the first deterministic chronological comparison shard over
the frozen Exp6827 event orders 0, 1, and 2. These zero-based order IDs map to
`order_1`, `order_2`, and `order_3`. The shard SHALL invoke no LLM, mutate no
model weights, and SHALL not pool with the later second shard. It SHALL compare
exactly four memory routes: `no_memory`, `read_only_memory`,
`random_admission`, and `verified_residual_memory`.

Before any decision, Exp6840 SHALL require
`residual_memory_kernel_ready_score=1`, complete assigned orders, stable source
hashes, exact later outcomes, nonzero headroom, and disjoint held-future
identities. A failed check SHALL write status
`complete_blocked_residual_memory_shard_a`, emit no comparison rows, set
`csl_shard_a_complete_score=0.0`, and record each failed check and observed
value in `gate_check_summary`.

Each event, arm, and seed row SHALL freeze proposal inputs and memory reads
before later outcomes are revealed. Compute budget, nominal capacity budget,
event order, and seed schedule SHALL be equal across all arms. Outcome
direction and action-level credit SHALL be assigned only after reveal. Any
memory update without a source event, action identity, and later exact outcome
SHALL reject without mutation.

The shard SHALL report every row by family, zero-based order index, arm, seed,
available headroom, memory dose, action, exact outcome, held-future metrics,
capacity use, and negative transfer. It SHALL report wins, ties, losses, and
no-headroom rows. It SHALL keep effect estimates separate from completion and
acceptance gates. `csl_shard_a_complete_score` SHALL depend only on planned row
count and receipt completeness.

The artifact SHALL include `schema`, `experiment_id`, `title`, `run_date`,
`status`, `openspec_requirement_ids`, `replay_commands`, `field_principles`,
`preconditions_checked`, `inference_substrate`, `duration_s`,
`continuous_self_learning_task`, `source_artifact_hashes`, `random_seeds`,
`reproducibility_checksum`, `split_manifest`, `arm_contracts`, `rows`,
`held_future_results`, `regret_results`, `abstention_results`,
`calibration_results`, `memory_dose_results`, `negative_transfer_results`,
`headroom_summary`, `checkpoint_manifest`, `csl_shard_a_complete_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL contain one principle for every
top-level field. `inference_substrate` SHALL equal
`deterministic CPU chronological comparison`. `continuous_self_learning_task`
SHALL be true. `verifier_is_oracle` SHALL be false. `verdict_class` SHALL use
only `positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL be terminal, row-supported, and start with
`complete_`.

### SCENARIO-CL-6840-PRECONDITIONS: Shard A Fails Closed

Given the residual kernel gate, assigned order coverage, source hash, exact
outcome, headroom, or held-future disjointness check fails,
When Exp6840 checks its frozen inputs,
Then it SHALL write `complete_blocked_residual_memory_shard_a`
And `gate_check_summary` SHALL include the failed check and observed value.

### SCENARIO-CL-6840-ISOLATION: Decisions Cannot Read Later Outcomes

Given one event, arm, seed, and parent memory state,
When Exp6840 freezes a decision row,
Then proposal material and memory reads SHALL contain only public event fields
and prior memory records
And exact outcome fields SHALL be absent until after reveal.

### SCENARIO-CL-6840-PARITY: Arms And Seeds Share Work Budgets

Given shard A starts,
When rows are built,
Then every assigned event SHALL appear once for every arm and seed
And every arm SHALL expose the same compute budget, nominal capacity budget,
and seed schedule.

### SCENARIO-CL-6840-CREDIT: Exact Later Outcomes Own Updates

Given a frozen pre-reveal action,
When Exp6840 reveals the exact later outcome,
Then action-level credit SHALL use only that outcome direction and frozen dose
And invalid updates missing source event, action identity, or exact outcome
SHALL reject without mutation.

### SCENARIO-CL-6840-RESTART: Checkpoint Replay Is Exact

Given persisted shard state bytes,
When Exp6840 restarts at the fixed checkpoint boundary,
Then loaded state hashes SHALL match persisted hashes
And final clean replay hashes SHALL match the live run.

### SCENARIO-CL-6840-METRICS: Row Completeness Drives Completion

Given all planned rows and receipts exist,
When Exp6840 computes terminal metrics,
Then held-future accuracy, regret, abstention, calibration, memory dose,
capacity, and negative-transfer summaries SHALL be row-derived
And `csl_shard_a_complete_score` SHALL remain separate from effect estimates.

## Implementation Status (REQ-CL-6840)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6840 and SCENARIO-CL-6840-* | Planned: `python/carnot/experiment_6840_residual_memory_chronological_shard_a.py`; `scripts/experiments/experiment_6840_residual_memory_chronological_shard_a.py`; `results/experiment_6840_residual_memory_chronological_shard_a.json`. | Planned: focused shard, coverage, lint, OpenSpec, adversarial, artifact, verdict-row, and root-clutter checks. |

## REQ-CL-6841: Residual-Memory Delayed-Correction Shard B

Exp6841 SHALL run the independent deterministic chronological comparison
shard over the frozen Exp6827 event orders 3 and 4. These zero-based order IDs
map to `order_4` and `order_5`. The shard SHALL invoke no LLM, mutate no model
weights, and SHALL not import Exp6840 aggregates. It SHALL compare exactly four
memory arms: `no_memory`, `read_only_memory`, `random_admission`, and
`verified_residual_memory`.

Before any decision, Exp6841 SHALL require
`residual_memory_kernel_ready_score=1`, complete assigned orders, stable source
hashes, exact later outcomes, delayed-correction headroom, and no overlap with
Exp6840 identities. A failed check SHALL write status
`complete_blocked_residual_memory_shard_b`, emit no comparison rows, set
`csl_shard_b_complete_score=0.0`, and record each failed check and observed
value in `gate_check_summary`.

Each event, arm, and seed row SHALL freeze proposal inputs, memory reads, and
the action dose before later outcomes are revealed. Outcome direction SHALL
apply only after the exact later receipt. Credit SHALL be attributed to each
action from that revealed direction, the frozen dose, and the arm-local prior
state. The shard SHALL record rejected, revised, expired, committed, and
rolled-back memory entries.

The delayed-correction cells SHALL report whether exact later evidence revises
stale memory and jointly wrong memory safely. Stale replacement SHALL tombstone
the stale entry before committing its exact correction. Joint-error correction
SHALL revise both wrong component memories from the same exact later receipt.
Rollback SHALL restore the exact pre-correction state bytes and checkpoint
restart SHALL match a clean replay.

The artifact SHALL include `schema`, `experiment_id`, `title`, `run_date`,
`status`, `openspec_requirement_ids`, `replay_commands`, `field_principles`,
`preconditions_checked`, `inference_substrate`, `duration_s`,
`continuous_self_learning_task`, `source_artifact_hashes`, `random_seeds`,
`reproducibility_checksum`, `split_manifest`, `arm_contracts`, `rows`,
`held_future_results`, `delayed_correction_results`,
`correction_latency_results`, `residual_error_results`,
`memory_state_transitions`, `negative_transfer_results`, `headroom_summary`,
`checkpoint_manifest`, `csl_shard_b_complete_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL contain one principle for every
top-level field. `inference_substrate` SHALL equal
`deterministic CPU chronological comparison`. `continuous_self_learning_task`
SHALL be true. `verifier_is_oracle` SHALL be false. `verdict_class` SHALL use
only `positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL be terminal, row-supported, and start with
`complete_`.

### SCENARIO-CL-6841-PRECONDITIONS: Shard B Fails Closed

Given the residual kernel gate, assigned order coverage, source hash, exact
outcome, delayed headroom, or Exp6840 overlap check fails,
When Exp6841 checks its frozen inputs,
Then it SHALL write `complete_blocked_residual_memory_shard_b`
And `gate_check_summary` SHALL include the failed check and observed value.

### SCENARIO-CL-6841-SHARD-DISJOINTNESS: Shard B Does Not Reuse Shard A

Given Exp6827 contains the first shard orders and the second shard orders,
When Exp6841 selects rows,
Then selected identities SHALL come only from `order_4` and `order_5`
And they SHALL be disjoint from the `order_1`, `order_2`, and `order_3`
identities used by Exp6840.

### SCENARIO-CL-6841-REVEAL-TIMING: Exact Outcomes Stay Hidden Until Receipt

Given one event, arm, seed, and parent memory state,
When Exp6841 freezes a decision row,
Then proposal material and memory reads SHALL contain only public event fields
and prior memory records
And exact outcome fields SHALL be absent until after reveal.

### SCENARIO-CL-6841-DELAYED-CREDIT: Exact Later Receipts Own Direction

Given a frozen pre-reveal action in a delayed-correction cell,
When Exp6841 reveals the exact later outcome,
Then action-level credit SHALL use only that outcome direction and frozen dose
And the row SHALL record the reveal delay and correction latency.

### SCENARIO-CL-6841-STALE-REPLACEMENT: Stale Memory Revises Safely

Given a stale-prerequisite cell and an arm-local stale memory entry,
When exact later evidence arrives,
Then the stale entry SHALL expire or revise before the correction commits
And the transition receipt SHALL record `expired` or `revised`.

### SCENARIO-CL-6841-JOINT-ERROR-CORRECTION: Jointly Wrong Memory Revises Safely

Given a competing-authorities or soft-conflict cell with two wrong memory
components,
When exact later evidence arrives,
Then both wrong components SHALL be revised or tombstoned from the same receipt
And no correction SHALL read another arm's state.

### SCENARIO-CL-6841-ROLLBACK: Correction Rollback Restores Bytes

Given a committed correction and its pre-correction checkpoint,
When Exp6841 forces rollback,
Then the restored bytes SHALL match the checkpoint hash
And the rolled-back transition SHALL be recorded.

### SCENARIO-CL-6841-CHECKPOINT-RESTART: Restart Matches Clean Replay

Given persisted shard state bytes,
When Exp6841 restarts at the fixed checkpoint boundary,
Then loaded state hashes SHALL match persisted hashes
And final clean replay hashes SHALL match the live run.

### SCENARIO-CL-6841-METRICS: Receipts Drive Completion

Given all planned rows and receipts exist,
When Exp6841 computes terminal metrics,
Then held-future performance, correction latency, residual error, negative
transfer, and no-headroom summaries SHALL be row-derived by family and order
And `csl_shard_b_complete_score` SHALL depend only on planned cells and
receipts.

## Implementation Status (REQ-CL-6841)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6841 and SCENARIO-CL-6841-* | Planned: `python/carnot/experiment_6841_residual_memory_delayed_correction_shard_b.py`; `scripts/experiments/experiment_6841_residual_memory_delayed_correction_shard_b.py`; `results/experiment_6841_residual_memory_delayed_correction_shard_b.json`. | Planned: focused shard, delayed-correction, rollback, coverage, lint, OpenSpec, adversarial, artifact, verdict-row, and root-clutter checks. |

## REQ-CL-6842: Sealed Memory Pathway Portability Audit

Exp6842 SHALL run a deterministic CPU sealed audit over the terminal Exp6840
and Exp6841 shard artifacts. It SHALL invoke no LLM, import no producer
aggregate, mutate no model weights, and recompute source-row effects from the
checked-in shard rows.

Before any reduction, Exp6842 SHALL require `csl_shard_a_complete_score=1`,
`csl_shard_b_complete_score=1`, stable source hashes, disjoint source
identities, complete arm rows, and exact outcome receipts. A failed check SHALL
write status `complete_blocked_sealed_memory_audit`, emit no audit rows, set
`sealed_csl_audit_complete_score=0.0`, set
`continuous_self_learning_ready_score=0.0`, and record each failed check and
observed value in `gate_check_summary`.

For every source row, Exp6842 SHALL recompute paired effects by family, order,
arm, and seed against the matched no-memory row. The audit SHALL separate wins,
ties, losses, no-headroom rows, and uncertainty before any pooled summary. A
losing family or order SHALL remain visible in `fresh_reduction_results`,
`negative_transfer_results`, and `leave_one_family_out_results`.

Exp6842 SHALL run deletion, substitution, reorder, poison, stale-credit, latent
error, restart, rollback, capacity, and leave-one-family-out checks. Memory
deletion SHALL remove action dose. Substitution SHALL invert the action
direction. Reorder SHALL prove order identity changes while aggregate counts
stay stable. Poison and stale-credit injections SHALL be bounded and SHALL NOT
improve readiness. Latent-error rows SHALL record whether joint errors persist,
repair, expire, or roll back. Restarted persisted state SHALL match clean replay
bytes, and rollback SHALL restore the parent state hash. Leave-one-family-out
rows SHALL run for every eligible family.

The audit SHALL set `sealed_csl_audit_complete_score=1.0` only when every
precondition, attack, durability check, capacity check, and leave-one-family-out
split is complete. It SHALL set
`continuous_self_learning_ready_score=1.0` only when held-future benefit,
durability, portability, calibrated dose, and leakage gates all pass. Null or
harmful memory evidence SHALL produce `verdict_class="null"` with a terminal
`complete_` honest verdict instead of a positive readiness score.

The artifact SHALL include `schema`, `experiment_id`, `title`, `run_date`,
`status`, `openspec_requirement_ids`, `replay_commands`, `field_principles`,
`preconditions_checked`, `inference_substrate`, `duration_s`,
`continuous_self_learning_task`, `source_artifact_hashes`, `random_seeds`,
`reproducibility_checksum`, `source_identity_summary`, `rows`,
`fresh_reduction_results`, `deletion_results`, `substitution_results`, `reorder_results`,
`poison_results`, `stale_credit_results`, `latent_error_pathways`,
`restart_durability_results`, `rollback_results`,
`leave_one_family_out_results`, `negative_transfer_results`,
`action_dose_calibration`, `capacity_results`,
`sealed_csl_audit_complete_score`, `continuous_self_learning_ready_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL contain one principle for every
top-level field. `inference_substrate` SHALL equal
`deterministic CPU sealed audit`. `continuous_self_learning_task` SHALL be
true. `verifier_is_oracle` SHALL be false. `verdict_class` SHALL use only
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL be terminal, row-supported, and start with
`complete_`.

### SCENARIO-CL-6842-PRECONDITIONS: Incomplete Shards Block The Audit

Given either shard score, source hash, identity-disjointness check, arm-row
coverage, or exact outcome receipt check fails,
When Exp6842 checks its sealed inputs,
Then it SHALL write `complete_blocked_sealed_memory_audit`
And `gate_check_summary` SHALL include the failed check and observed value.

### SCENARIO-CL-6842-FRESH-REDUCTION: Effects Recompute From Rows

Given complete Exp6840 and Exp6841 source rows,
When Exp6842 recomputes paired effects,
Then every source row SHALL have a matched no-memory baseline by shard, source
event, and seed
And results SHALL be grouped by family, order, arm, and seed with wins, ties,
losses, no-headroom rows, and uncertainty.

### SCENARIO-CL-6842-SHARD-IDENTITY: Source Shards Stay Independent

Given Exp6840 and Exp6841 were produced as independent chronological shards,
When Exp6842 reads their source identities,
Then their source event identities SHALL be disjoint
And no reducer row SHALL import a pooled producer aggregate.

### SCENARIO-CL-6842-ATTACKS: Memory Edits Do Not Create Readiness

Given baseline audit rows,
When deletion, substitution, reorder, poison, stale-credit, latent-error,
capacity, and rollback attacks run,
Then every attack SHALL emit row-supported results
And bounded poison or stale credit SHALL fail the readiness gate.

### SCENARIO-CL-6842-DURABILITY: Restart And Rollback Preserve Bytes

Given a persisted sealed audit state,
When Exp6842 restarts from that state and compares with clean replay,
Then the state hash SHALL match clean replay
And rollback SHALL restore the recorded parent hash.

### SCENARIO-CL-6842-PORTABILITY: Leave-One-Family-Out Does Not Hide Harm

Given every eligible family in both shards,
When Exp6842 leaves one family out,
Then each split SHALL report held-future benefit, negative transfer, and sample
counts separately
And a harmful or nonportable split SHALL force
`continuous_self_learning_ready_score=0.0`.

### SCENARIO-CL-6842-READY: Readiness Is Fully Conjunctive

Given all audit attacks complete but any held-future, durability, portability,
dose-calibration, or leakage gate fails,
When Exp6842 computes the terminal verdict,
Then `sealed_csl_audit_complete_score` SHALL be `1.0`
And `continuous_self_learning_ready_score` SHALL be `0.0`
And `honest_verdict` SHALL start with `complete_`.

## Implementation Status (REQ-CL-6842)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6842 and SCENARIO-CL-6842-* | Planned: `python/carnot/experiment_6842_sealed_memory_pathway_portability_audit.py`; `scripts/experiments/experiment_6842_sealed_memory_pathway_portability_audit.py`; `results/experiment_6842_sealed_memory_pathway_portability_audit.json`. | Planned: focused sealed audit, coverage, lint, OpenSpec, adversarial, artifact, verdict-row, and root-clutter checks. |

## REQ-CL-6853: Risk-Sensitive Memory Opportunity Fixture

The system SHALL build a deterministic chronological fixture from Exp6827
decision rows and the exact later outcomes in Exp6840 and Exp6841. The builder
SHALL NOT import Exp6842 code, Exp6842 rows, residual doses, residual updates,
predicted directions, admission decisions, or learner decisions. Each fixture
row SHALL contain one stable decision identity, one observed action, the three
available actions, a fixed pre-outcome context, and one exact later outcome.

The fixed actions SHALL be `verified_memory`, `no_memory`, and `abstain`.
`random_admission` and `always_memory` SHALL be comparison baselines only. The
fixture SHALL record action availability and observed potential-outcome support.
It SHALL NOT create an unobserved action outcome.

The pre-outcome context SHALL contain features for relevance, uncertainty,
exact compatibility, age, correction status, family, capacity, false-positive
risk, and cost. A provenance manifest SHALL identify the source field or fixed
rule for each feature. Outcome identities, outcome hashes, signed outcomes,
effects, headroom labels, and learner decisions SHALL not occur in the decision
context.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `random_seed`,
`reproducibility_checksum`, `rows`, `decision_context_schema`,
`feature_provenance_manifest`, `action_manifest`, `outcome_authority_manifest`,
`chronological_split_manifest`, `leakage_attack_results`,
`decision_headroom_rows`, `helpful_memory_count`, `harmful_memory_count`,
`abstention_opportunity_count`, `memory_headroom_nonzero_score`,
`risk_sensitive_stream_ready_score`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.
`inference_substrate` SHALL equal
`deterministic CPU chronological fixture construction`.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL use only `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
`honest_verdict` SHALL start with `complete_`.

### SCENARIO-CL-6853-PRECONDITIONS: Invalid Evidence Blocks The Fixture

Given `v599_evidence_contract_ready_score` is not one, a chronological source
is unreadable, a decision identity is unstable, or a later outcome is missing,
When Exp6853 checks its source evidence,
Then it SHALL write
`complete_blocked_risk_sensitive_memory_opportunity_fixture`
And `gate_check_summary` SHALL name each failed check and observed value.

### SCENARIO-CL-6853-LEAKAGE: Decision Context Excludes Outcomes

Given a decision context schema or row contains a denied outcome field,
When Exp6853 runs leakage attacks,
Then the fixture SHALL fail closed
And `risk_sensitive_stream_ready_score` SHALL equal zero.

### SCENARIO-CL-6853-DUPLICATES: Decisions Have Stable Unique Identities

Given two rows have the same decision identity,
When Exp6853 validates the chronological fixture,
Then the duplicate SHALL block readiness.

### SCENARIO-CL-6853-ACTIONS: All First-Class Actions Are Available

Given an opportunity row omits `verified_memory`, `no_memory`, or `abstain`,
When Exp6853 validates action availability,
Then the row SHALL be invalid
And no potential outcome SHALL be fabricated for an unsupported action.

### SCENARIO-CL-6853-HEADROOM: Safe Selection Has Positive Controls

Given exact later outcomes include helpful, harmful, ambiguous, delayed
correction, and zero-headroom opportunities,
When Exp6853 measures each decision and stratum,
Then it SHALL record both zero and nonzero safe-selection headroom
And readiness SHALL require helpful and harmful memory counts above zero.

### SCENARIO-CL-6853-FAMILY-BALANCE: No Family Dominates The Fixture

Given the source includes the three required model families,
When Exp6853 measures family counts,
Then each family SHALL contribute the same decision count
And any imbalance SHALL block readiness.

### SCENARIO-CL-6853-READY: Readiness Is Fully Conjunctive

Given rows are chronological, outcome-complete, nonleaking, action-complete,
identity-unique, family-balanced, and contain nonzero safe-selection headroom,
When Exp6853 computes its terminal result,
Then `memory_headroom_nonzero_score` SHALL equal one
And `risk_sensitive_stream_ready_score` SHALL equal one.

## REQ-CL-6854: Risk-Sensitive Abstention Memory Controller

The system SHALL replay Exp6853 decisions in chronological order with a small
fixed-feature contextual bandit. The selectable actions SHALL be
`verified_memory`, `no_memory`, and `abstain`. The controller SHALL use a
pessimistic confidence score and deterministic tie-breaking. It SHALL update
only its bounded external policy state. It SHALL NOT use residual pressure,
post-outcome features, a learned LLM judge, or model-weight updates.

Before replay, Exp6854 SHALL require
`risk_sensitive_stream_ready_score=1`, stable row hashes, complete exact later
outcomes, at least two actions with observed headroom, and a clean controller
state. A failed check SHALL write
`complete_blocked_risk_sensitive_abstention_memory_controller`. The blocked
artifact SHALL set `risk_sensitive_controller_complete_score=0` and SHALL name
the failed check and observed value in `gate_check_summary`.

For each decision, the controller SHALL receive only its fixed pre-outcome
context. It SHALL serialize the chosen action, context hash, and policy state
hash before the exact later outcome becomes available. Delayed feedback SHALL
reference the frozen action receipt. Each update SHALL clamp loss and state to
the declared bounds.

The risk matrix SHALL penalize harmful memory injection more than missed reuse
and more than abstention when memory would have helped. Exp6854 SHALL compare
the learned policy with no-memory, always-memory, abstain-only,
random-admission, and read-only fixed controls on the same decision identities.
It SHALL report sensitivity to each declared fixed risk ratio.

The controller SHALL checkpoint at bounded intervals. Checkpoints SHALL use
stable bytes and a content hash. Restart from a checkpoint SHALL produce the
same final controller bytes as uninterrupted replay. Corrupt checkpoints SHALL
fail closed. A poison update SHALL remain bounded, and rollback SHALL restore
the exact parent bytes. Controller state size, checkpoint storage growth, and
update latency SHALL be recorded.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `random_seed`,
`reproducibility_checksum`, `rows`, `controller_schema`, `risk_matrix`,
`pre_outcome_action_receipts`, `exact_feedback_receipts`, `update_rows`,
`checkpoint_manifest`, `restart_equivalence_results`, `rollback_results`,
`state_size_rows`, `latency_rows`, `per_arm_summary`, `held_future_effect`,
`false_positive_injection_rate`, `abstention_rate`, `risk_sensitivity_rows`,
`risk_sensitive_controller_complete_score`, `controller_benefit_gate_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `inference_substrate` SHALL equal
`deterministic CPU online contextual bandit replay`. `verifier_is_oracle`
SHALL be false. `verdict_class` SHALL use only `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
`honest_verdict` SHALL start with `complete_` and SHALL be supported by rows.

`risk_sensitive_controller_complete_score` SHALL depend only on execution and
receipt completeness. `controller_benefit_gate_score` SHALL remain separate.
The benefit gate SHALL require positive held-future effect against no-memory
and a bounded false-positive injection rate.

### SCENARIO-CL-6854-PRECONDITIONS: Invalid Inputs Block Replay

Given fixture readiness, a row hash, exact later authority, action headroom, or
clean initial state check fails,
When Exp6854 checks its inputs,
Then it SHALL emit the blocked terminal artifact
And `gate_check_summary` SHALL name the failed check and observed value.

### SCENARIO-CL-6854-ACTION-FREEZE: Choice Precedes Outcome Reveal

Given one chronological decision and its sealed later outcome,
When the controller chooses an action,
Then it SHALL serialize the action receipt and policy state hash first
And no outcome or learner field SHALL occur in the decision context.

### SCENARIO-CL-6854-DELAYED-FEEDBACK: Updates Reference Frozen Choices

Given feedback becomes available after a later sequence position,
When the controller applies that feedback,
Then the update SHALL reference the frozen action receipt
And the controller SHALL reject feedback without a pending frozen decision.

### SCENARIO-CL-6854-ASYMMETRIC-LOSS: Harmful Injection Costs More

Given a harmful-memory outcome and a helpful-memory outcome,
When the fixed risk matrix computes losses,
Then harmful verified-memory loss SHALL exceed missed-reuse loss
And it SHALL exceed helpful-case abstention loss for every sensitivity ratio.

### SCENARIO-CL-6854-ABSTENTION: Weak Evidence Uses The Safety Arm

Given an unknown or unsupported context,
When the controller computes pessimistic scores,
Then it SHALL select `abstain`
And the receipt SHALL state the conservative fallback reason.

### SCENARIO-CL-6854-UNSEEN-CONTEXT: Unknown Features Do Not Leak

Given a family or correction status outside the fixed feature schema,
When the controller encodes the context,
Then it SHALL mark the context unseen
And it SHALL NOT extend the feature vector or read an outcome field.

### SCENARIO-CL-6854-CAPACITY: Policy State Stays Bounded

Given updates exceed the declared per-action capacity,
When more feedback arrives,
Then counts and sufficient statistics SHALL remain within fixed bounds
And pending feedback SHALL not exceed its fixed capacity.

### SCENARIO-CL-6854-CHECKPOINT-CORRUPTION: Invalid Bytes Fail Closed

Given a checkpoint byte or payload hash changes,
When the controller loads the checkpoint,
Then loading SHALL raise a corruption error
And no partial state SHALL become active.

### SCENARIO-CL-6854-RESTART-EQUIVALENCE: Restart Is Byte-Identical

Given a checkpoint at a fixed chronological boundary,
When replay resumes from that checkpoint,
Then the final controller bytes SHALL equal uninterrupted replay bytes.

### SCENARIO-CL-6854-ROLLBACK: Poison Restores Parent State

Given a bounded poison update after a saved checkpoint,
When rollback loads the saved checkpoint,
Then the restored controller bytes SHALL equal the parent bytes
And no foundation-model weight file SHALL change.

### SCENARIO-CL-6854-TIE-BREAKING: Equal Scores Are Deterministic

Given two or more actions have equal pessimistic scores,
When the controller selects an action,
Then it SHALL prefer `abstain`, then `no_memory`, then `verified_memory`.

### SCENARIO-CL-6854-CONTROLS: Policies Share Decision Identities

Given a complete replay row,
When Exp6854 evaluates the learned policy and fixed controls,
Then every policy SHALL receive one metric for the same decision identity
And no control SHALL change the learned controller state.

### SCENARIO-CL-6854-GATES: Completion And Benefit Stay Separate

Given all decisions and receipts are complete but held-future benefit is not
positive or false-positive injection exceeds its bound,
When Exp6854 computes terminal gates,
Then `risk_sensitive_controller_complete_score` SHALL equal one
And `controller_benefit_gate_score` SHALL equal zero.

## Implementation Status (REQ-CL-6854)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6854 and SCENARIO-CL-6854-* | Planned: `python/carnot/continuous_learning.py`; `python/carnot/experiment_6854_risk_sensitive_abstention_memory_controller.py`; `scripts/experiments/experiment_6854_risk_sensitive_abstention_memory_controller.py`; `results/experiment_6854_risk_sensitive_abstention_memory_controller.json`. | Planned: focused controller tests, new-code coverage, full Python tests, Ruff, OpenSpec coverage, adversarial verification, artifact convention, verdict-row consistency, leakage, and root-clutter checks. |

## REQ-CL-6855: Counterfactual Memory Credit Audit

The system SHALL independently replay the bounded external-policy writes from
Exp6854. A write SHALL mean one observed update receipt that changes only the
controller's sufficient statistics. The audit SHALL use a fresh reducer and
replay engine. It SHALL NOT import Exp6854's aggregate calculator.

Before replay, Exp6855 SHALL require
`risk_sensitive_controller_complete_score=1`, stable decision and state
hashes, complete exact later outcomes, and a declared valid-counterfactual
contract. A failed check SHALL write
`complete_blocked_counterfactual_memory_credit_audit`. The blocked artifact
SHALL set `counterfactual_memory_audit_complete_score=0` and SHALL name each
failed check and observed value in `gate_check_summary`.

The valid-counterfactual contract SHALL permit deletion only when the fresh
reducer reconstructs every later selection state with the observed exogenous
decision order, contexts, action availability, and exact outcomes unchanged.
Substitution SHALL use an exact observed donor write with compatible features
and action. Order replay SHALL change only exchangeable writes at the same
reveal boundary. The audit SHALL reject missing support, duplicate writes,
invalid coalitions, and a divergent state path. It SHALL not infer an
unobserved environmental outcome.

Each target decision SHALL use a bounded window of eligible prior writes. A
window with at most eight writes SHALL use exact coalition enumeration. A
larger window SHALL use seeded bounded permutations and SHALL report a
convergence interval. Every row SHALL name its method. An approximation SHALL
never be labeled exact.

The audit SHALL compare learned selection with `no_memory`, `always_memory`,
`random_admission`, `abstain`, and `placebo_context` policies on the same
chronological decisions. Every control SHALL preserve action availability.
The placebo policy SHALL permute only a declared pre-outcome feature. The
audit SHALL report controller selection skill separately from stream
composition and environmental luck.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `random_seed`,
`reproducibility_checksum`, `rows`, `valid_counterfactual_contract`,
`deletion_rows`, `substitution_rows`, `coalition_rows`,
`approximation_receipts`, `unsupported_counterfactual_rows`,
`placebo_control_rows`, `per_write_credit_summary`, `harmful_write_count`,
`redundant_write_count`, `interaction_witnesses`, `selection_skill_effect`,
`stream_luck_effect`, `counterfactual_memory_audit_complete_score`,
`causal_memory_credit_eligible_score`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.
`inference_substrate` SHALL equal
`deterministic CPU counterfactual replay`. `verifier_is_oracle` SHALL be false.
`verdict_class` SHALL use only `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL start with
`complete_` and SHALL be supported by terminal rows.

The audit SHALL classify each supported write as `helpful`, `harmful`,
`redundant`, `interaction_only`, or `zero_headroom`. Unsupported writes SHALL
remain separate. A positive aggregate SHALL not remove a harmful write. Causal
credit eligibility SHALL depend on replay and support. Benefit eligibility and
benefit values SHALL remain separate from causal eligibility.

A complete audit with any supported harmful-write evidence SHALL use the
terminal `null` verdict class and SHALL name the harmful evidence in
`honest_verdict`. It SHALL NOT use `partial`, because replay completeness is
already established. It SHALL NOT use `positive`, because an aggregate gain
cannot override per-write harm.

### SCENARIO-CL-6855-PRECONDITIONS: Invalid Evidence Blocks The Audit

Given the controller score, a decision hash, a state hash, an exact outcome,
or the valid-counterfactual contract fails,
When Exp6855 checks its source evidence,
Then it SHALL emit the blocked terminal artifact
And `gate_check_summary` SHALL name the failed check and observed value.

### SCENARIO-CL-6855-INVALID-COALITION: Coalitions Use Eligible Writes Once

Given a coalition contains an ineligible write or repeats a write,
When the fresh replay engine validates the coalition,
Then it SHALL reject the coalition
And SHALL not emit causal credit for it.

### SCENARIO-CL-6855-MISSING-SUPPORT: Unobserved Transitions Stay Unsupported

Given deletion, substitution, or order replay lacks an exact supported
transition,
When Exp6855 evaluates the counterfactual,
Then it SHALL add an `unsupported_counterfactual_rows` entry
And its causal credit and benefit fields SHALL remain ineligible.

### SCENARIO-CL-6855-STATE-PATH-DIVERGENCE: Reconstruction Fails Closed

Given a counterfactual state becomes nonfinite, exceeds bounds, or does not
reach the declared target boundary,
When Exp6855 reconstructs the later state,
Then the counterfactual SHALL be unsupported
And no later outcome SHALL be invented.

### SCENARIO-CL-6855-DUPLICATE-WRITE: Write Identities Are Unique

Given two source updates have the same write identity,
When Exp6855 validates the chronological write ledger,
Then the audit SHALL block before counterfactual replay.

### SCENARIO-CL-6855-INTERACTION: Joint Credit Does Not Hide Interactions

Given a write has zero leave-one-out effect but a nonzero coalition marginal,
When Exp6855 reduces per-write evidence,
Then it SHALL classify the write as `interaction_only`
And SHALL emit an interaction witness with the supporting coalition.

### SCENARIO-CL-6855-PLACEBO: Placebo Context Cannot Earn Causal Credit

Given a seeded permutation changes only the declared placebo feature,
When Exp6855 replays the placebo policy,
Then decision identities, action availability, chronology, and exact outcomes
SHALL remain unchanged
And placebo rows SHALL not enter per-write causal credit.

### SCENARIO-CL-6855-ZERO-HEADROOM: Exact Zero Headroom Stays Separate

Given the exact source row has `safe_selection_headroom=0`,
When Exp6855 assigns per-write evidence for that target,
Then the row SHALL be marked `zero_headroom`
And causal replay eligibility SHALL remain separate from benefit eligibility.

### SCENARIO-CL-6855-METHODS: Exact And Approximate Methods Stay Distinct

Given an eligible window contains at most eight writes,
When Exp6855 computes coalition credit,
Then it SHALL enumerate every coalition exactly.
Given a larger eligible window,
When Exp6855 computes bounded permutation credit,
Then it SHALL report the seed, permutation count, and convergence interval
And SHALL label the method as approximate.

### SCENARIO-CL-6855-CONTROLS: Policy Effects Preserve Matched Opportunities

Given the learned and control policies receive the same decision stream,
When Exp6855 reduces their row metrics,
Then each policy SHALL contain one metric per decision in chronological order
And the audit SHALL separate learned-minus-placebo selection skill from the
no-memory stream baseline.

### SCENARIO-CL-6855-GATES: Completeness And Credit Stay Separate

Given all supported replays finish but no write has benefit-eligible causal
credit,
When Exp6855 computes terminal gates,
Then `counterfactual_memory_audit_complete_score` SHALL equal one
And `causal_memory_credit_eligible_score` SHALL equal zero.

## Implementation Status (REQ-CL-6855)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6855 and SCENARIO-CL-6855-* | Planned: `python/carnot/experiment_6855_counterfactual_memory_credit_audit.py`; `scripts/experiments/experiment_6855_counterfactual_memory_credit_audit.py`; `results/experiment_6855_counterfactual_memory_credit_audit.json`. | Planned: focused counterfactual replay, new-code coverage, full Python tests, Ruff, OpenSpec coverage, adversarial verification, artifact convention, verdict-row consistency, counterfactual validity, and root-clutter checks. |

## REQ-CL-6856: Sealed Risk-Sensitive Learning Audit

The system SHALL run a fresh deterministic reduction over the chronological
controller rows from Exp6854 and the per-write credit rows from Exp6855. The
reducer SHALL NOT import producer code or producer aggregate calculators. It
SHALL recompute each headline from decision rows and per-write metric rows.

Before reduction, Exp6856 SHALL require both structured completion scores to
equal one. It SHALL also require stable sealed row digests, a declared
chronological train and held-future split, complete comparison arms, and no
reducer-source overlap. A failed check SHALL write
`complete_blocked_sealed_risk_sensitive_learning_audit`. The artifact SHALL
name each failed check and observed value in `gate_check_summary`.

The audit SHALL compare the controller with `no_memory`, `always_memory`,
`abstain_only`, `random_admission`, and `read_only_fixed`. Each arm SHALL use
the same decision identities. The audit SHALL report wins, losses, ties, and
no-headroom decisions separately. It SHALL not map missing headroom to zero
effect.

The audit SHALL test false-positive injection, abstention calibration,
byte-identical restart, bounded state, rollback after poison, delayed
correction, poison rejection, capacity eviction, and tombstone behavior. It
SHALL run leave-one-family-out and leave-one-order-out reductions. Each
held-out family and order SHALL have its own row.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `random_seed`,
`reproducibility_checksum`, `rows`, `fresh_reducer_manifest`,
`per_arm_summary`, `held_future_effect`, `win_loss_tie_counts`,
`no_headroom_count`, `false_positive_injection_rate`,
`abstention_calibration`, `durability_results`, `restart_results`,
`rollback_results`, `delayed_correction_results`, `poison_results`,
`capacity_results`, `leave_one_family_out_rows`,
`leave_one_order_out_rows`, `leakage_results`,
`counterfactual_support_results`, `continuous_self_learning_ready_score`,
`retirement_recommendation`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`.

`inference_substrate` SHALL equal
`deterministic CPU sealed independent reduction`. `verifier_is_oracle` SHALL
be false. `verdict_class` SHALL use only `positive`, `circular_positive`,
`null`, `blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL start
with `complete_` and SHALL be supported by terminal rows.

Readiness SHALL require nonzero held-future benefit, bounded false-positive
injection, calibrated nondegenerate abstention, durability, valid
counterfactual support, no leakage, and portable nonnegative held-out effects.
Persistence SHALL not substitute for benefit. A complete audit SHALL preserve
an exact negative, null, or partial disposition when a benefit gate fails.

### SCENARIO-CL-6856-PRECONDITIONS: Invalid Evidence Blocks Reduction

Given a completion score is not one, a sealed row digest changes, a split
changes, an arm is missing, or producer code overlaps the reducer,
When Exp6856 validates its sources,
Then it SHALL emit the blocked terminal artifact
And `gate_check_summary` SHALL include the failed check and observed value.

### SCENARIO-CL-6856-ALL-NULL: Null Rows Cannot Support A Claim

Given every measured metric in a decision row is null,
When the fresh reducer validates the row,
Then the audit SHALL fail closed
And it SHALL not claim positive or continuous-learning readiness.

### SCENARIO-CL-6856-AGGREGATE-CONTRADICTION: Rows Override Aggregates

Given a producer aggregate contradicts the fresh row reduction,
When Exp6856 computes a headline,
Then the fresh row value SHALL control the artifact disposition
And the contradiction SHALL be reported as leakage or integrity failure.

### SCENARIO-CL-6856-POISON: Poison Cannot Become Durable Evidence

Given one nonfinite or over-bound credit update,
When the reducer applies the update,
Then it SHALL reject the update or clamp it within the declared bound
And rollback SHALL restore the byte-identical parent state.

### SCENARIO-CL-6856-STALE-CORRECTION: Delayed Corrections Need Live Support

Given a correction arrives after its support expires or after a tombstone,
When the reducer processes the correction,
Then it SHALL reject the correction as stale
And it SHALL not restore the removed credit.

### SCENARIO-CL-6856-ROLLBACK: Rollback Restores Exact Parent Bytes

Given a supported update is followed by a poison update,
When the reducer rolls back the poison branch,
Then the serialized state SHALL equal the parent bytes.

### SCENARIO-CL-6856-CAPACITY: Eviction Leaves A Tombstone

Given supported writes exceed the fixed state capacity,
When Exp6856 admits the next write,
Then it SHALL evict the oldest active write
And it SHALL retain a bounded tombstone that blocks stale correction.

### SCENARIO-CL-6856-FAMILY-REMOVAL: Portability Is Reported Per Family

Given one family is removed from the training partition,
When the reducer evaluates its held-out decisions,
Then it SHALL emit one row for that held-out family
And missing headroom SHALL remain null rather than become zero effect.

### SCENARIO-CL-6856-ORDER-REMOVAL: Portability Is Reported Per Order

Given one chronological order is removed from the training partition,
When the reducer evaluates its held-out decisions,
Then it SHALL emit one row for that held-out order
And it SHALL not pool the row with another order.

### SCENARIO-CL-6856-GATES: Readiness Is Fully Conjunctive

Given the sealed reduction completes but any benefit, calibration, durability,
support, leakage, or portability gate fails,
When Exp6856 computes its terminal disposition,
Then `continuous_self_learning_ready_score` SHALL equal zero
And persistence or restart success SHALL not change that result.

## Implementation Status (REQ-CL-6856)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-6856 and SCENARIO-CL-6856-* | Planned: `python/carnot/experiment_6856_sealed_risk_sensitive_learning_audit.py`; `scripts/experiments/experiment_6856_sealed_risk_sensitive_learning_audit.py`; `results/experiment_6856_sealed_risk_sensitive_learning_audit.json`. | Planned: focused sealed reducer tests, new-code coverage, full Python tests, Ruff, OpenSpec coverage, adversarial verification, artifact convention, verdict-row consistency, leakage, and root-clutter checks. |

## REQ-LEARN-6871: Observable Reliability Opportunity Stream

The system SHALL rebuild each opportunity from the row-level transaction
receipts in Exp6827 and the row-level exact outcomes in Exp6840 or Exp6841.
Exp6853 through Exp6856 SHALL NOT provide decision or outcome authority.
Their current adversarial status SHALL remain visible as precondition evidence.

Each accepted row SHALL bind one unique event identity to its primary source,
candidate action, pre-action state, later exact outcome, and immutable content
identity. A row that resolves only to aggregate prose SHALL be rejected.
Receipt hashes and exact-outcome hashes SHALL be nonempty SHA-256 identities.

Every row field SHALL be in either `decision_time_observable` or
`offline_supervision_only`. A decision feature SHALL NOT depend on a later
outcome, delayed correction, audit label, held split, task order, or any other
future field. The stream SHALL fail closed when this boundary is violated.

The action manifest SHALL freeze `no_memory`, `read_only_retrieval`,
`bounded_update`, `quarantine`, and `v599_unsafe_reference`. Each accepted
event SHALL preserve at least one valid pre-action counterfactual. Exp6871
SHALL freeze these actions and SHALL NOT apply a reliability update.

The reliability state SHALL use a small fixed node set for evidence sources
and memory actions. Its initial matrix SHALL be finite, symmetric, and zero.
The contract SHALL freeze the maximum entry change, maximum spectral change,
and post-outcome update time before any controller runs.

The base order SHALL be chronological and deterministic. At least five seeded
order replicates SHALL preserve family-held-out anchors and old-family anchors.
The transition attack manifest SHALL include omission, corruption,
unsupported insertion, provenance loss, delayed invalidation, tombstone
reappearance, restart loss, and rollback mismatch. Every attack SHALL be
detected and routed to quarantine.

Preconditions SHALL require `v601_evidence_contract_ready_score=1`, readable
primary transaction and exact-outcome receipts, and a current live adversarial
status for each V599 artifact. Any failed precondition SHALL emit
`complete_blocked_observable_reliability_opportunity_stream` with an exact
failed check, expected value, and observed value in `gate_check_summary`.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`,
`primary_receipt_manifest`, `rows`, `rejected_opportunity_rows`,
`observable_feature_manifest`, `offline_supervision_manifest`,
`leakage_witnesses`, `action_manifest`, `counterfactual_support_rows`,
`reliability_state_schema`, `bounded_update_contract`,
`chronological_order_manifest`, `order_replicate_manifest`,
`old_family_anchor_rows`, `transition_attack_manifest`, `random_seed`,
`reproducibility_checksum`, `observable_reliability_stream_ready_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`.

`inference_substrate` SHALL equal
`deterministic CPU primary-receipt opportunity reconstruction`.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL use only
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL start with `complete_`.

`observable_reliability_stream_ready_score` SHALL equal one only when every
accepted event has primary provenance, a valid pre-action counterfactual, a
later exact outcome, no leakage, all required anchors, and all required attack
rows. Otherwise, the score SHALL equal zero.

### SCENARIO-LEARN-6871-MISSING-PRIMARY: Missing Primary Receipt Blocks The Stream

Given a transaction source or exact-outcome source is absent or unreadable,
When Exp6871 checks its preconditions,
Then it SHALL emit the complete blocked verdict
And `gate_check_summary` SHALL name the missing source.

### SCENARIO-LEARN-6871-DUPLICATE: Event Identities Are Unique

Given two accepted rows have the same event identity,
When Exp6871 validates the stream,
Then readiness SHALL equal zero
And the duplicate identities SHALL be reported.

### SCENARIO-LEARN-6871-LEAKAGE: Outcomes Cannot Select Their Own Actions

Given a decision feature depends on a later outcome, correction, audit label,
held split, or task order,
When Exp6871 checks the observable boundary,
Then it SHALL record a leakage witness
And readiness SHALL equal zero.

### SCENARIO-LEARN-6871-ORDER: Reordered Events Fail Closed

Given the base event sequence is not in frozen chronological order,
When Exp6871 validates its base order,
Then readiness SHALL equal zero
And the first order mismatch SHALL remain visible.

### SCENARIO-LEARN-6871-COUNTERFACTUAL: Every Event Has A Valid Alternative

Given an accepted event has no legal pre-action counterfactual,
When Exp6871 validates counterfactual support,
Then the event SHALL be rejected
And no unsupported outcome SHALL be invented.

### SCENARIO-LEARN-6871-STALE: Stale Evidence Routes To Quarantine

Given evidence is invalid at decision time or has lost primary provenance,
When Exp6871 freezes action support,
Then `quarantine` SHALL remain available
And the stale row SHALL not authorize a bounded update.

### SCENARIO-LEARN-6871-DELAYED-CORRECTION: Corrections Stay Offline

Given a later exact correction invalidates prior evidence,
When Exp6871 freezes the event,
Then the correction SHALL be offline supervision only
And the delayed invalidation attack SHALL route to quarantine.

### SCENARIO-LEARN-6871-POISON: Nonfinite Or Over-Bound Feedback Is Rejected

Given a nonfinite or over-bound proposed reliability change,
When Exp6871 evaluates the poison case,
Then the proposal SHALL be rejected before update
And the frozen zero state SHALL remain unchanged.

### SCENARIO-LEARN-6871-RESTART: Restart Must Preserve Frozen State

Given a restart drops a reliability node or changes its initial value,
When Exp6871 evaluates the restart attack,
Then it SHALL detect restart loss
And route the transition to quarantine.

### SCENARIO-LEARN-6871-ROLLBACK: Rollback Must Restore Parent Bytes

Given rollback bytes do not equal the frozen parent bytes,
When Exp6871 evaluates the rollback attack,
Then it SHALL detect rollback mismatch
And route the transition to quarantine.

## Implementation Status (REQ-LEARN-6871)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-LEARN-6871 and SCENARIO-LEARN-6871-* | Implemented: `python/carnot/experiment_6871_observable_reliability_opportunity_stream.py`; `scripts/experiments/experiment_6871_observable_reliability_opportunity_stream.py`; `results/experiment_6871_observable_reliability_opportunity_stream.json`. | `tests/python/test_experiment_6871_observable_reliability_opportunity_stream.py` covers receipt rejection, leakage, order, counterfactual, stale evidence, delayed correction, poison, restart, rollback, and the real reconstruction. |

## REQ-LEARN-6872: Bounded Reliability Controller With Exact Quarantine

The system SHALL consume only a complete Exp6871 opportunity stream. The
preconditions SHALL require `observable_reliability_stream_ready_score=1`, no
clean-stream leakage witness, valid counterfactual support, and frozen action,
order, and bounded-update contracts. A failed precondition SHALL emit
`complete_blocked_bounded_reliability_controller_quarantine`. The
`gate_check_summary` SHALL name the failed check, expected value, and observed
value.

The controller SHALL initialize a finite symmetric reliability matrix over the
frozen evidence-source and memory-action nodes. A bounded arm SHALL update the
matrix only after its action is frozen and its later exact outcome is revealed.
Each accepted update SHALL keep the matrix symmetric. Its largest absolute
entry change and spectral norm change SHALL stay within the Exp6871 bounds.
Every state SHALL have a deterministic checksum. A rejected update SHALL keep
the prior checksum.

Each event decision SHALL use only the prior reliability state and
decision-time features. The current event outcome SHALL not select its action.
The action set SHALL be `no_memory`, `read_only`, `write`, and `abstain`.
The comparison SHALL include `frozen_no_memory`, `read_only`,
`bounded_update`, `exact_quarantine`, and `v599_unsafe_reference`. All arms
SHALL receive identical events and the same frozen order seed.

A proposed write SHALL remain provisional until exact checks cover transition
coverage, preservation, source faithfulness, provenance, old-family retention,
delayed invalidation, replay, restart, and byte-exact rollback. Failure of any
check SHALL quarantine the write. A negative later exact outcome SHALL count as
a harmful write and SHALL not be admitted by a bounded or quarantine arm.

The result SHALL record one prospective row per event and arm. Each row SHALL
bind the event, arm, proposed action, admission decision, later exact outcome,
state before, state after, and transition check. The result SHALL also report
action entropy and rates for abstention, retrieval, write proposals, admitted
writes, useful writes, harmful writes, false injection, old-family retention,
held-future utility, and rollback.

`bounded_reliability_controller_ready_score` SHALL equal one only when the
bounded controller uses more than one action, admits at least one exact-
supported write, admits no harmful transition, respects every state bound, and
records every prospective row. This score SHALL not claim a sealed utility
verdict.

The artifact SHALL set `continuous_self_learning_task=true`,
`no_model_weight_mutation=true`, and `verifier_is_oracle=false`. It SHALL bind
the no-model baseline, source artifact, module, wrapper, tests, and spec to
SHA-256 identities. The before and after no-model baseline identities SHALL be
equal.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`,
`continuous_self_learning_task`, `no_model_weight_mutation`, `rows`,
`state_transition_rows`, `spectral_bound_rows`,
`action_distribution_by_arm`, `abstention_rate_by_arm`,
`admitted_update_rows`, `rejected_update_rows`, `harmful_write_rows`,
`exact_transition_check_rows`, `held_future_utility_by_arm`,
`old_family_retention_by_arm`, `delayed_correction_rows`, `restart_rows`,
`rollback_rows`, `counterfactual_support_rows`, `random_seed`,
`reproducibility_checksum`, `bounded_reliability_controller_ready_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`.

`inference_substrate` SHALL equal
`deterministic CPU bounded reliability-state online update simulation`.
`verdict_class` SHALL use only `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL start with
`complete_`.

### SCENARIO-LEARN-6872-STATE: Invalid State Changes Fail Closed

Given an asymmetric state, nonfinite feedback, or an excessive spectral change,
When the controller evaluates an update,
Then it SHALL reject the update
And the state checksum SHALL not change.

### SCENARIO-LEARN-6872-TIMING: Same-Event Outcomes Cannot Select Actions

Given a decision context contains the current later exact outcome,
When the controller selects an action,
Then it SHALL detect the leakage
And it SHALL not update the state from that decision.

### SCENARIO-LEARN-6872-COLLAPSE: Readiness Requires Multiple Actions

Given a controller always abstains or always writes,
When readiness is computed,
Then `bounded_reliability_controller_ready_score` SHALL equal zero.

### SCENARIO-LEARN-6872-QUARANTINE: Unsafe Writes Are Not Admitted

Given a harmful write, unsupported insertion, lost anchor, delayed
invalidation, replay failure, restart drift, or rollback mismatch,
When exact admission checks run,
Then the write SHALL be quarantined
And durable memory SHALL keep its parent bytes.

### SCENARIO-LEARN-6872-WEIGHTS: Model Mutation Disqualifies Readiness

Given the no-model or model-file hash differs after the run,
When the result is validated,
Then `no_model_weight_mutation` SHALL be false
And readiness SHALL equal zero.

### SCENARIO-LEARN-6872-ROWS: Arm Comparisons Stay Prospective

Given a ready observable stream and a frozen order seed,
When all five arms run,
Then every event SHALL have one row for every arm
And every row SHALL bind action timing, exact outcome, state transition, and
admission evidence.

## Implementation Status (REQ-LEARN-6872)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-LEARN-6872 and SCENARIO-LEARN-6872-* | Implemented: `python/carnot/experiment_6872_bounded_reliability_controller_quarantine.py`; `scripts/experiments/experiment_6872_bounded_reliability_controller_quarantine.py`; `results/experiment_6872_bounded_reliability_controller_quarantine.json`. | `tests/python/test_experiment_6872_bounded_reliability_controller_quarantine.py` covers state asymmetry, spectral bounds, timing leakage, action collapse, exact quarantine attacks, restart, rollback, weight immutability, blocked gates, the CLI, and the checked-in stream. |

## REQ-LEARN-6873: Prospective Sealed Self-Learning Utility And Safety Audit

The system SHALL consume Exp6872 only when
`bounded_reliability_controller_ready_score=1`. It SHALL verify the frozen
Exp6871 stream hash and the frozen Exp6872 artifact, module, wrapper, and test
hashes. It SHALL require at least five unique frozen order seeds. Every order
SHALL preserve all events and complete pre-action counterfactual support. A
failed precondition SHALL emit
`complete_blocked_prospective_sealed_self_learning_audit`. The
`gate_check_summary` SHALL name the failed check, expected value, and observed
value.

The audit SHALL run `frozen_no_memory`, `read_only`, `bounded_update`,
`exact_quarantine`, and `v599_unsafe_reference` from the same empty initial
state for every frozen order. Each arm and order pair SHALL run in a fresh
process with a private checkpoint and empty cache. A separate fresh process
SHALL restore each checkpoint. No state, memory, tombstone, cache, or temporary
path SHALL cross an arm or order boundary.

The artifact SHALL preserve one row for every event, arm, and order replicate.
Each row SHALL bind the action, exact later outcome, state before and after,
admission checks, utility, old-family anchor retention, delayed-correction
handling, tombstone handling, and rollback evidence. All headline metrics SHALL
be recomputed from these rows. Producer aggregates SHALL not provide authority.

The audit SHALL recompute per-order and pooled action entropy, abstention,
admitted useful updates, harmful writes, false injections, held-future utility,
and old-family retention. It SHALL recompute every spectral update bound and
state-symmetry check. It SHALL verify exact persistence, fresh-process restart
restoration, delayed correction, tombstone non-resurrection, and byte-exact
active-memory rollback.

The audit SHALL compare `bounded_update` and `exact_quarantine` with
`frozen_no_memory`, `read_only`, and `v599_unsafe_reference` using paired order
effects. Each comparison SHALL preserve wins, ties, losses, missing orders,
available headroom, mean effect, and the exact interval across order effects.
A positive claim SHALL require a positive effect against both
`frozen_no_memory` and `read_only` in every declared order. A pooled gain from
only one order SHALL not satisfy this rule.

`verdict_class=positive` SHALL require all declared order comparisons, positive
headroom, more than one quarantine action, at least one admitted useful
quarantine write, zero admitted harmful quarantine writes, complete old-family
retention, zero leakage witnesses, and every spectral, delayed-correction,
persistence, restart, tombstone, and rollback gate. Safety failure SHALL
disqualify the claim. No headroom, action collapse, or zero admitted useful
writes SHALL produce a non-positive verdict. Aggregate values that contradict
the rows SHALL disqualify the artifact.

The artifact SHALL set `continuous_self_learning_task=true`,
`no_model_weight_mutation=true`, and `verifier_is_oracle=false`. It SHALL
contain `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `continuous_self_learning_task`,
`no_model_weight_mutation`, `rows`, `per_order_results`,
`action_distribution_by_arm`, `admitted_useful_updates_by_arm`,
`harmful_writes_by_arm`, `false_injection_rate_by_arm`,
`held_future_utility_by_arm`, `paired_order_effects`,
`old_family_retention_by_arm`, `spectral_bound_audit_rows`,
`delayed_correction_rows`, `persistence_rows`, `restart_rows`, `rollback_rows`,
`leakage_witnesses`, `scientific_claim_eligible`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`.

`inference_substrate` SHALL equal
`fresh-process deterministic CPU prospective CSL audit`. `verdict_class` SHALL
use only `positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL start with `complete_`.

### SCENARIO-LEARN-6873-PRECONDITIONS: Frozen Inputs Fail Closed

Given readiness, a frozen hash, an order seed, or counterfactual support is
missing,
When Exp6873 checks its inputs,
Then it SHALL emit the complete blocked artifact
And the gate summary SHALL preserve expected and observed values.

### SCENARIO-LEARN-6873-FRESH-PROCESS: State Never Crosses Boundaries

Given five arms and at least five frozen orders,
When Exp6873 runs the audit,
Then every arm-order pair SHALL start from the same empty state in a fresh
process
And each restart SHALL restore from its private checkpoint in another process.

### SCENARIO-LEARN-6873-ONE-ORDER: One Order Cannot Authorize A Pooled Win

Given one order wins but another order ties or loses,
When paired effects are reduced,
Then the declared replication rule SHALL fail
And the verdict SHALL not be positive.

### SCENARIO-LEARN-6873-NO-HEADROOM: Saturation Cannot Prove Learning

Given a comparator has no available held-future headroom,
When Exp6873 evaluates utility,
Then the comparison SHALL remain visible
And the verdict SHALL not be positive.

### SCENARIO-LEARN-6873-ACTION-COLLAPSE: Always-Abstain Fails Utility

Given the quarantine arm selects only one action,
When action entropy is recomputed from rows,
Then action collapse SHALL be true
And the verdict SHALL not be positive.

### SCENARIO-LEARN-6873-ZERO-WRITES: Useful Learning Requires A Write

Given the quarantine arm admits no useful write,
When the utility gate runs,
Then persistence success SHALL not substitute for useful learning
And the verdict SHALL not be positive.

### SCENARIO-LEARN-6873-HARMFUL-WRITE: Harm Disqualifies The Claim

Given the quarantine arm admits one harmful write,
When safety is recomputed from rows,
Then the claim SHALL be disqualified.

### SCENARIO-LEARN-6873-FORGETTING: Old Families Must Be Retained

Given an old-family anchor is lost,
When retention is recomputed,
Then the claim SHALL be disqualified.

### SCENARIO-LEARN-6873-STATE-BOUND: Invalid State Fails Closed

Given a state is asymmetric or an update exceeds an entry or spectral bound,
When the state audit runs,
Then the claim SHALL be disqualified.

### SCENARIO-LEARN-6873-DELAYED-CORRECTION: Corrected Writes Stay Tombstoned

Given a delayed correction invalidates a proposed write,
When the exact outcome is applied,
Then the write SHALL not remain active
And its tombstone SHALL survive persistence and restart.

### SCENARIO-LEARN-6873-RESTART: Restart Drift Disqualifies The Claim

Given a separate process restores different state, memory, or tombstone bytes,
When the restart audit runs,
Then the claim SHALL be disqualified.

### SCENARIO-LEARN-6873-ROLLBACK: Active Memory Rollback Is Byte Exact

Given a rejected write restores bytes that differ from its parent active
memory,
When the rollback audit runs,
Then the claim SHALL be disqualified.

### SCENARIO-LEARN-6873-LEAKAGE: Future Outcome Cannot Select Its Action

Given a decision uses a same-event outcome, delayed correction, held label, or
order label,
When leakage is audited,
Then the witness SHALL remain visible
And the claim SHALL be disqualified.

### SCENARIO-LEARN-6873-AGGREGATE-CONTRADICTION: Rows Remain Authoritative

Given a stored headline disagrees with the row-level recomputation,
When the artifact validator runs,
Then it SHALL report the contradiction
And the claim SHALL be disqualified.

## Implementation Status (REQ-LEARN-6873)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-LEARN-6873 and SCENARIO-LEARN-6873-* | Implemented: `python/carnot/experiment_6873_prospective_sealed_self_learning_audit.py`; `scripts/experiments/experiment_6873_prospective_sealed_self_learning_audit.py`; `results/experiment_6873_prospective_sealed_self_learning_audit.json`. | `tests/python/test_experiment_6873_prospective_sealed_self_learning_audit.py` covers frozen preconditions, one-order gains, headroom, action collapse, zero writes, harmful writes, forgetting, state bounds, delayed correction, fresh restart, rollback, leakage, aggregate contradictions, the CLI, and the checked-in artifact. |

## REQ-LEARN-6961: Certified Prospective Event Sequence

Carnot SHALL build a deterministic, sealed event sequence from prior Exp6957
certificates. Seed admission SHALL require an exact mapping result from both
registered authorities. Confidence, rationale, and learned scores SHALL have no
admission authority. The producer SHALL also verify the matching Exp6955 fixture
row before it uses a seed.

The producer SHALL freeze at least 24 chronological events for each headline
model family. It SHALL freeze six training events before the evaluation boundary.
It SHALL freeze event order, family identifiers, retrieval keys, token budgets,
exact outcomes, and split membership before later model inference. The producer
SHALL not run a model or report a model-quality result.

Each later formulation SHALL be related to a seed but SHALL not be identical to
it. The deterministic generator SHALL vary names, bounds, coefficients,
objective direction, and irrelevant surface text. It SHALL preserve a reusable
mapping pattern for some events. It SHALL deliberately break the pattern for
conflict events. Two exact authorities SHALL certify every frozen outcome.

For each event, the producer SHALL record eligible prior certificate identifiers,
prohibited current and future identifiers, similarity, reusable factors, conflict
class, and whether the visible formulations permit a correct proposal without an
answer copy. A prompt SHALL contain only visible formulations and bounded factor
summaries from prior events. It SHALL not contain its own outcome, a current or
future outcome, or a full mapping that is isomorphic to its answer.

The producer SHALL compute matched structural opportunity for `no_memory`,
`fifo`, and `queue` arms without running those arms. The sequence SHALL contain
relevant retrieval, irrelevant distractors, contradictions, delayed corrections,
retention probes, and a no-op retrieval control. Readiness SHALL require positive
structural headroom and at least 12 later opportunity events for each headline
model family. Insufficient opportunity SHALL produce a complete `null` result. It
SHALL not fabricate model outcomes.

The producer SHALL write a sealed checkpoint. A fresh process SHALL regenerate
the sequence from the checkpoint inputs and match every event, prompt, retrieval,
and sequence hash. `certified_event_sequence_ready_score` SHALL equal one only
when all planned rows, safety cases, split rules, opportunity floors, source
hashes, and fresh-process replay checks pass.

A failed precondition SHALL emit `blocked_certified_event_sequence`. Its
`gate_check_summary` SHALL name the failed check, expected value, and observed
value. A conforming sequence SHALL use `verdict_class=circular_positive` because
the exact verifier is the oracle for sequence conformance only. This class SHALL
not become evidence of memory benefit.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `rows`,
`seed_certificate_rows`, `event_rows`, `chronology_rows`, `family_rows`,
`transformation_rows`, `retrieval_key_rows`, `eligible_prior_rows`,
`prohibited_future_rows`, `similarity_rows`, `reusable_factor_rows`,
`conflict_rows`, `distractor_rows`, `correction_rows`, `retention_probe_rows`,
`opportunity_rows`, `headroom_rows`, `leakage_rows`, `split_rows`,
`sealed_checkpoint_path`, `fresh_process_replay_rows`, `random_seed`,
`reproducibility_checksum`, `certified_event_sequence_ready_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL give one scientific principle for
every required field.

`inference_substrate` SHALL equal
`deterministic_sealed_exact_certificate_sequence_no_llm`.
`verdict_class` SHALL use only `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL use a terminal
prefix that agrees with the verdict class.

### SCENARIO-LEARN-6961-PRECONDITIONS: Missing Exact Inputs Block

Given an incomplete Exp6957 run, missing exact rows, too few certified successes,
a nondeterministic generator, a fixture mismatch, hash drift, or an unwritable
checkpoint,
When the producer checks its inputs,
Then it SHALL write the complete blocked artifact
And the gate summary SHALL preserve the failed check and both compared values.

### SCENARIO-LEARN-6961-CHRONOLOGY: Time Reversal Fails Closed

Given an event refers to a certificate that becomes available at its own or a
later ordinal,
When sequence conformance is checked,
Then readiness SHALL remain zero
And the violation SHALL be named `time_reversal`.

### SCENARIO-LEARN-6961-LEAKAGE: Future Labels And Answers Stay Sealed

Given a prompt contains a current or future outcome, a prohibited certificate,
or a mapping isomorphic to its exact answer,
When leakage checks run,
Then the prompt SHALL fail conformance
And no readiness claim SHALL be allowed.

### SCENARIO-LEARN-6961-IDENTITY: Duplicates And Family Collisions Reject

Given duplicate event content, duplicate certificate identity, or a retrieval key
that treats a different problem family as relevant,
When identity and family checks run,
Then the sequence SHALL fail conformance with a stable reason.

### SCENARIO-LEARN-6961-RETRIEVAL: No-Op And Distractors Stay Explicit

Given a no-op control or a FIFO selection with irrelevant prior certificates,
When arm opportunity is computed,
Then the no-op control SHALL select nothing
And each irrelevant selection SHALL remain a named distractor, not relevant memory.

### SCENARIO-LEARN-6961-HEADROOM: Opportunity Is Required Per Family

Given any headline model family has fewer than 12 later events with relevant
non-answer memory or has zero structural headroom,
When the terminal gate is reduced,
Then readiness SHALL equal zero
And the terminal result SHALL be a complete null.

### SCENARIO-LEARN-6961-COPIES: Seed And Answer Copies Reject

Given two seed rows copy one source certificate or retrieved memory contains a
complete source or answer mapping,
When copy checks run,
Then the sequence SHALL fail conformance
And the copied material SHALL not count as opportunity.

### SCENARIO-LEARN-6961-REPLAY: Fresh Process Detects Hash Drift

Given the sealed generator inputs and all parent event hashes,
When a fresh process regenerates the sequence,
Then all event, prompt, retrieval, and sequence hashes SHALL match
And any mismatch SHALL keep readiness at zero.

## Implementation Status (REQ-LEARN-6961)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-LEARN-6961 and SCENARIO-LEARN-6961-* | Implemented: `python/carnot/experiment_6961_certified_event_sequence.py`; `scripts/experiments/experiment_6961_certified_event_sequence.py`; terminal evidence in `results/experiment_6961_certified_event_sequence.json`. | Verified: `tests/python/test_experiment_6961_certified_event_sequence.py`. |

## REQ-LEARN-6962: Queue-Regulated Exact-Certificate Self-Learning

Carnot SHALL run a prospective comparison on the sealed Exp6961 event sequence.
The comparison SHALL use `no_memory`, fixed-capacity `fifo`, and
`debt_queue` arms. It SHALL run every model-arm pair in a fresh process. It
SHALL match event order, prompt budget, retrieval count, memory capacity,
decoding parameters, random seeds, and attempted calls across arms.

The headline models SHALL be `unsloth/Qwen3.6-35B-A3B-GGUF` and
`unsloth/gemma-4-26B-A4B-it-GGUF`. The producer SHALL resolve local GGUF files
through the current cache resolver. It SHALL use llama.cpp for embedded
tokenizer probes and inference. It SHALL never call `AutoTokenizer` on a GGUF
repository identifier. Legacy small models MAY run CPU smoke tests. Their rows
SHALL not enter headline metrics.

Before inference, the producer SHALL require the Exp6961 readiness score, the
sealed checkpoint and hash, both model files, successful `vocab_only` probes,
authenticated CUDA offload, transactional stores, hard-reset support, exact
certifier access, and per-event recovery. A failed check SHALL write a complete
blocked artifact. Its `gate_check_summary` SHALL name the failed check, expected
value, and observed value.

Each arm SHALL keep a private transactional store. Raw model output SHALL become
durable before the current exact outcome becomes visible to the arm. Retrieval
SHALL use only earlier active exact-success certificates. A write SHALL occur
only after exact certification. A certificate SHALL bind its event, model,
prompt, raw output, exact outcome, and parent store hash. Model confidence,
rationale, and self-reported correctness SHALL have no write authority.

FIFO SHALL admit each valid exact-success certificate. It SHALL evict the oldest
active certificate when the frozen capacity is exceeded. The eviction SHALL
leave a durable tombstone. The debt arm SHALL use only past queue state and
certified metadata for admission and retrieval. Its frozen recurrence SHALL be
`Q_next = max(0, Q_prev + arrivals - service)`. One arrival SHALL be charged for
each distinct causal retrieved certificate that contributes to an exact failure,
contradiction, stale use, or retention regression. A certificate-event pair
SHALL be charged at most once. Service SHALL not exceed available debt. The
producer SHALL freeze capacity, retrieval count, arrival weights, service rule,
debt threshold, and drift-plus-penalty coefficient before inference.

Every event SHALL atomically checkpoint the raw output, retrieved identifiers,
prompt hash, exact outcome, write decision, queue state, store hash, token
counts, latency, and runtime receipt. Recovery SHALL resume at the next missing
model-arm-event key. A duplicate key SHALL not repeat inference, write, service,
or debt arrival. Model hashes SHALL match before and after all calls.

After the chronological stream, each arm SHALL run frozen irrelevant-retrieval,
delayed-copy-poison, contradiction, old-family-retention, restart, tombstone,
and rollback checks. Restart SHALL reproduce active records, tombstones, debt,
and the store hash in a fresh process. Rollback SHALL restore the exact parent
bytes. Forged, future, poisoned, contradicted, stale, or tombstoned certificates
SHALL not become active or retrievable.

Headline metrics SHALL be recomputed from event rows. The queue arm SHALL beat
both comparators on paired later-event exact accuracy. Each paired 95 percent
confidence interval SHALL have a lower bound above zero. The queue arm SHALL
not reduce old-family retention. It SHALL admit no forged or future certificate.
It SHALL pass restart and rollback. Lower debt without an accuracy gain SHALL be
a null result. An aggregate that differs from row-level recomputation SHALL
disqualify the result.

`queue_learning_run_complete_score` SHALL equal one only when every planned
model, arm, event, and safety row is complete. `queue_learning_positive_score`
SHALL equal one only when every headline utility and safety gate passes. The
artifact SHALL set `continuous_self_learning_task=true`, `learning_tier=2`,
`no_model_weight_mutation=true`, and `verifier_is_oracle=false`.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `rows`,
`model_specs`, `model_rows`, `arm_rows`, `event_rows`, `chronology_rows`,
`prompt_rows`, `raw_output_rows`, `exact_outcome_rows`, `retrieval_rows`,
`write_rows`, `admission_rows`, `fifo_rows`, `debt_arrival_rows`,
`debt_service_rows`, `debt_balance_rows`, `memory_hash_rows`,
`token_budget_rows`, `latency_rows`, `poison_rows`, `contradiction_rows`,
`retention_rows`, `restart_rows`, `tombstone_rows`, `rollback_rows`,
`future_label_isolation_rows`, `paired_metric_rows`,
`confidence_interval_rows`, `checkpoint_rows`, `model_lifecycle_rows`,
`task_runtime_receipt`, `continuous_self_learning_task`, `learning_tier`,
`no_model_weight_mutation`, `model_hashes_before`, `model_hashes_after`,
`random_seed`, `reproducibility_checksum`,
`queue_learning_run_complete_score`, `queue_learning_positive_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL give one scientific principle for
every required field.

`inference_substrate` SHALL equal
`prospective_local_gguf_exact_certificate_debt_queue_memory`.
`verdict_class` SHALL use only `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL use a terminal
prefix that agrees with `verdict_class`.

### SCENARIO-LEARN-6962-ARM-ISOLATION: Each Arm Starts Fresh

Given a model and the three frozen policies,
When the comparison starts each model-arm run,
Then it SHALL use a new process and a private empty store
And no state SHALL cross an arm boundary.

### SCENARIO-LEARN-6962-FUTURE-LABEL: Future Outcomes Stay Sealed

Given the current or a future exact outcome,
When retrieval, prompting, or admission is decided,
Then unavailable labels SHALL have no influence
And a future certificate SHALL be rejected.

### SCENARIO-LEARN-6962-WRITE-ORDER: Output Precedes Outcome And Write

Given one event has produced raw model bytes,
When the exact certifier evaluates the output,
Then the raw bytes SHALL already be durable
And no memory write SHALL predate the exact outcome.

### SCENARIO-LEARN-6962-FORGED-CERTIFICATE: Exact Binding Is Required

Given a certificate has an invalid authority or content hash,
When admission is attempted,
Then the certificate SHALL be rejected
And it SHALL never become retrievable.

### SCENARIO-LEARN-6962-CONFIDENCE: Self-Reports Have No Authority

Given two outputs differ only in confidence or rationale,
When admission is decided,
Then the decision SHALL be unchanged
And only external exact certification MAY authorize a write.

### SCENARIO-LEARN-6962-QUEUE-UNDERFLOW: Service Is Bounded

Given service exceeds the available virtual debt,
When the recurrence is applied,
Then the charged service SHALL be capped at available debt
And the new debt SHALL be nonnegative.

### SCENARIO-LEARN-6962-DUPLICATE-DEBT: Causal Debt Is Idempotent

Given one certificate causes two reports for the same event,
When debt arrivals are recorded,
Then the certificate-event pair SHALL be charged once.

### SCENARIO-LEARN-6962-POISON: Delayed Copies Stay Quarantined

Given a delayed copy or poisoned certificate,
When safety cases run,
Then the record SHALL not become active
And no active memory hash SHALL include it.

### SCENARIO-LEARN-6962-CONTRADICTION: Conflicts Charge Causal Debt

Given a retrieved certificate contributes to an exact contradiction,
When the outcome is certified,
Then the causal certificate SHALL receive one frozen debt charge
And the contradiction SHALL remain visible.

### SCENARIO-LEARN-6962-RESTART: Fresh Restore Is Exact

Given an atomically checkpointed private store,
When a fresh process restores it,
Then active records, tombstones, debt, and the store hash SHALL match.

### SCENARIO-LEARN-6962-TOMBSTONE: Removed Records Do Not Return

Given FIFO eviction or exact invalidation creates a tombstone,
When retrieval and restart run,
Then the record SHALL remain inactive and unavailable.

### SCENARIO-LEARN-6962-ROLLBACK: Parent Bytes Are Restored

Given an interrupted or rejected transaction,
When rollback runs,
Then the store SHALL equal the exact parent bytes and hash.

### SCENARIO-LEARN-6962-MODEL-HASH: Weights Stay Frozen

Given all model-arm processes completed,
When lifecycle hashes are compared,
Then each before hash SHALL equal its after hash
And no headline row SHALL claim a model-weight update.

### SCENARIO-LEARN-6962-AGGREGATE: Rows Override Stored Headlines

Given a stored arm metric differs from event-row recomputation,
When the artifact is validated,
Then the mismatch SHALL be named
And the positive score SHALL remain zero.

## Implementation Status (REQ-LEARN-6962)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-LEARN-6962 and SCENARIO-LEARN-6962-* | Planned: `python/carnot/experiment_6962_queue_regulated_self_learning.py`; `scripts/experiments/experiment_6962_queue_regulated_self_learning.py`; terminal evidence in `results/experiment_6962_queue_regulated_self_learning.json`. | Planned: focused policy, safety, lifecycle, artifact, and command tests with new-code coverage. |

## REQ-LEARN-6978: Transactional Verifier-Grounded Constraint Learning

Carnot SHALL run Exp6978 over the exact 24-event chronological stream from
Exp6967. It SHALL compare `frozen`, `read_only`, and `transactional_write`
arms. All arms SHALL use the selected Exp6976 `direct` schedule and the exact
Qwen model `unsloth/Qwen3.6-35B-A3B-GGUF`. Each arm-event call SHALL have a
fresh inference context. Event order, prompts, a 128-token output cap, seeds,
arm rotation, the exact evaluator, memory schema, memory byte limit, and
rollback threshold SHALL be frozen before the first live call.

Before inference, Exp6978 SHALL require the bare integer upstream scores
`lease_aware_runtime_ready_score=1`, `fixture_admissibility_ready_score=1`, and
`selected_policy_ready_score=1`. It SHALL require the exact Exp6967 artifact
hash, all 24 event and prompt hashes, the local Qwen GGUF, Z3, a writable
transaction directory, and a clean time-zero visibility surface. Any failed
check SHALL write `blocked_transactional_constraint_self_learning`. Its
`gate_check_summary` SHALL name the failed check, expected value, and observed
value. External absence SHALL be blocked, not partial.

At event position `t`, each arm SHALL see only event records from positions
zero through `t-1`. The current prompt and raw completion SHALL become durable
before the independent exact executor opens the current outcome. The writer
SHALL receive only the atomic error class, exact certificate digest, frozen
schedule metadata, and exact outcome. It SHALL not receive model confidence,
rationale, future labels, later outcomes, or another arm's state.

Each arm SHALL use a private store and journal. `frozen` SHALL not read or
write memory. `read_only` SHALL read the frozen initial memory and SHALL not
write. `transactional_write` SHALL start from byte-identical initial memory.
It MAY propose an update only after the exact outcome. Every proposal SHALL be
durably journaled before commit. A commit SHALL be atomic and bounded by the
frozen memory byte limit. A sealed replay window SHALL run after each commit.
If its safety score worsens beyond the frozen threshold, the update SHALL roll
back to the exact parent bytes. A forced interruption SHALL prove journal
recovery in a fresh store instance.

Every arm-event row SHALL report terminal state, event and arm identities,
exact success, parse success, selected schedule, memory hit, write, commit,
rollback, latency, token budget, token use, and state bytes. Exp6978 SHALL
derive held-future gain over read-only, plasticity on newly recurring error
classes, stability on prior successes, maximum forgetting, and final memory
bytes from row evidence. The positive score SHALL not depend on write count.

`self_learning_run_complete_score` SHALL be the bare integer one only when all
72 arm-event rows are terminal, budgets match, journal replay succeeds, and
leakage checks pass. `transactional_learning_positive_score` SHALL be the bare
integer one only when the write arm gains at least two held-future exact
successes over read-only, loses none of the read-only successes, passes the
rollback fixture, and stays within the memory byte limit. A complete run that
does not meet this effect gate SHALL have `verdict_class="null"`.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `live_duration_s`,
`source_artifact_hashes`, `MODEL_SPECS`, `models_used`, `model_file_hashes`,
`stream_hash`, `event_order_hash`, `arm_config_rows`, `budget_rows`, `rows`,
`per_event_results`, `prompt_visibility_rows`, `exact_outcome_rows`,
`memory_lookup_rows`, `update_proposal_rows`, `transaction_journal_rows`,
`commit_rows`, `rollback_rows`, `restart_recovery_rows`, `held_future_rows`,
`chronological_gain_over_readonly`, `plasticity_score`, `stability_score`,
`max_forgetting`, `memory_state_bytes`, `leakage_check_rows`,
`self_learning_run_complete_score`, `transactional_learning_positive_score`,
`checkpoint_rows`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL give one scientific principle for
every required field and both scores. `inference_substrate` SHALL equal
`live_local_qwen36_transactional_chronological_constraint_learning`.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL be one of
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL use a terminal prefix consistent with the
class.

### SCENARIO-LEARN-6978-CHRONOLOGY: Visibility Advances One Event At A Time

Given the frozen 24-event order,
When an arm starts event `t`,
Then its visible predecessor ordinals SHALL equal `range(t)`
And no later event field SHALL enter its prompt or memory lookup.

### SCENARIO-LEARN-6978-ARM-ISOLATION: Arm State Is Private

Given three arms start from their frozen initial state,
When one arm commits or rolls back an update,
Then the other arm state hashes SHALL remain unchanged
And each arm SHALL use a distinct store and inference-context identity.

### SCENARIO-LEARN-6978-POST-OUTCOME: Commit Follows Exact Outcome

Given one durable prompt and raw completion,
When the independent exact executor emits its terminal outcome,
Then the writer MAY create and journal a proposal
And no proposal, journal prepare, or commit SHALL predate that outcome.

### SCENARIO-LEARN-6978-JOURNAL: Prepare Is Durable Before Commit

Given an eligible post-outcome proposal,
When the transaction starts,
Then a file-fsynced prepare record SHALL exist before state replacement
And a directory-fsynced commit record SHALL bind parent and new state hashes.

### SCENARIO-LEARN-6978-ROLLBACK: Harmful Update Restores Parent Bytes

Given a proposed update worsens the sealed replay safety score,
When the frozen rollback rule fires,
Then the exact parent bytes and hash SHALL be restored
And the harmful record SHALL not remain active.

### SCENARIO-LEARN-6978-RESTART: Interrupted Prepare Recovers Safely

Given a forced interruption after durable prepare and before commit,
When a fresh store instance replays the journal,
Then it SHALL preserve the last committed state bytes
And it SHALL record the incomplete transaction as recovered without applying it.

### SCENARIO-LEARN-6978-BUDGETS: Live Budgets Match Across Arms

Given one event and its three rotated arms,
When all calls finish,
Then model identity, event identity, token cap, and seed SHALL match
And attempted-call and terminal-row counts SHALL be equal across arms.

### SCENARIO-LEARN-6978-NO-FUTURE: Future Labels Have No Authority

Given a current prompt, memory lookup, and writer input,
When their nested fields are audited,
Then confidence, rationale, future labels, and later outcomes SHALL be absent
And any injected denied field SHALL fail the leakage gate.

### SCENARIO-LEARN-6978-BARE: Downstream Scores Are Bare Integers

Given a blocked, complete-null, or complete-positive artifact,
When its downstream gate fields are read,
Then both completion and positive scores SHALL be bare zero-or-one integers
And neither score SHALL be a wrapped value or a write-count proxy.

## Implementation Status (REQ-LEARN-6978)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-LEARN-6978 and SCENARIO-LEARN-6978-* | Implemented: transactional constraint memory, durable bounded policy store, Exp6978 module, command wrapper, and terminal live artifact. | Verified: focused chronology, isolation, post-outcome transaction, recovery, rollback, matched-budget, leakage, artifact, and command tests pass with 100% new-code coverage. |

## REQ-LEARN-6979: Read-Only Cold Audit Of Transactional Self-Learning

Carnot SHALL audit Exp6978 in a fresh Python process. The audit process SHALL
disable network access, GPU visibility, LLM loading, and writes to the learned
store. It SHALL hash all input bytes before it reads source metrics. It SHALL
use only an in-memory state machine to replay transaction records.

The audit SHALL require `self_learning_run_complete_score=1`, exactly 72
terminal arm-event rows, the pinned Exp6978 artifact hash, the pinned Exp6967
fixture hash, the pinned chronological stream hash, all durable prompt and
completion files, a complete journal for each store, and readable final store
bytes. A missing or changed prerequisite SHALL produce
`blocked_self_learning_cold_audit`. The blocked artifact SHALL name the first
failed check with its expected and observed values. Missing upstream evidence
SHALL not produce a partial verdict.

For every event and arm, the audit SHALL rebuild the predecessor frontier from
the Exp6967 event chain. It SHALL parse each durable prompt surface. It SHALL
confirm that prompt predecessors and memory records precede the active event.
It SHALL reject denied future or confidence fields. It SHALL confirm the
sequence `prompt_durable < raw_completion_durable < exact_outcome`. Any writer
proposal and commit SHALL follow the exact outcome.

The audit SHALL replay each `prepare`, `commit`, `abort_recovered`, and
`rollback` row. It SHALL validate the row hash chain, embedded state bytes,
bounded candidate-state construction, parent state, restored bytes, and final
state hash. It SHALL compare every arm-event state transition with the replay.
It SHALL not repair a disagreement.

The audit SHALL derive arm budget rows, exact-success rows, held-future paired
gain, plasticity, stability, maximum forgetting, and state bytes from source
rows. It SHALL report Wilson 95 percent intervals for binomial rates. It SHALL
report a seeded paired-bootstrap 95 percent interval for the held-future mean
delta. Stored headline values SHALL not enter these calculations.

The positive gate SHALL require at least two held-future gains over read-only,
zero lost read-only successes, a passing rollback replay, no leakage, and a
passing state-byte budget. The artifact SHALL report each term. A source
headline disagreement SHALL force the positive gate to zero and SHALL produce
a disqualified verdict. A reproduced source null SHALL remain null.

`self_learning_audit_complete_score` SHALL be the bare integer one only when
all 72 rows and all states replay. `learning_safety_confirmed_score` SHALL be
the bare integer one only when source hashes, chronology, leakage, rollback,
budgets, state bytes, and headline arithmetic agree. The safety score MAY be
one for a reproduced null because it confirms audit integrity, not utility.

The artifact SHALL contain `schema`, `experiment_id`, `run_date`,
`field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `rows`, `per_event_results`,
`visibility_replay_rows`, `journal_replay_rows`, `state_hash_rows`,
`budget_recomputation_rows`, `metric_recomputation_rows`,
`positive_gate_recomputation`, `source_disagreement_rows`,
`leakage_audit_rows`, `rollback_audit_rows`,
`read_only_enforcement_receipt`, `self_learning_audit_complete_score`,
`learning_safety_confirmed_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL contain one
scientific principle for every required field. `inference_substrate` SHALL be
`fresh_process_readonly_transaction_replay`. `verifier_is_oracle` SHALL be
false. `verdict_class` SHALL be `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. The honest verdict prefix SHALL agree
with the class.

### SCENARIO-LEARN-6979-READ-ONLY: Learned Bytes Cannot Change

Given the source artifact and learned store snapshot,
When the fresh audit process runs,
Then a store-write probe and a network probe SHALL fail closed
And every protected input hash SHALL be unchanged after the audit.

### SCENARIO-LEARN-6979-JOURNAL: Transactions Replay In Memory

Given the complete hash-linked journals,
When the audit replays every supported phase,
Then every parent, candidate, restored, and final state hash SHALL agree
And changed journal bytes SHALL fail without repair.

### SCENARIO-LEARN-6979-ORDER: Event Visibility Is Chronological

Given the frozen Exp6967 chain and 72 durable prompts,
When each visibility frontier is rebuilt,
Then each prompt and memory lookup SHALL contain only allowed predecessors
And each exact outcome SHALL follow its durable prompt and completion.

### SCENARIO-LEARN-6979-NO-FUTURE: Future Evidence Is Rejected

Given one prompt surface or memory record contains a later event or denied key,
When the audit checks authority timing,
Then the leakage check SHALL fail
And the learning safety score SHALL remain zero.

### SCENARIO-LEARN-6979-METRICS: Rows Override Stored Headlines

Given all 72 arm-event rows,
When budgets, exact outcomes, utility, forgetting, state bytes, and intervals
are recomputed,
Then the audit SHALL use row values only
And it SHALL record each stored-value disagreement.

### SCENARIO-LEARN-6979-CLAIM: A Source Disagreement Cannot Become Positive

Given a source positive score or headline differs from row recomputation,
When the positive gate is evaluated,
Then its score SHALL be zero
And the verdict SHALL be disqualified rather than repaired or promoted.

### SCENARIO-LEARN-6979-NULL: A Reproduced Null Stays Null

Given all integrity and safety checks pass but the utility gate fails,
When the terminal artifact is written,
Then the audit and safety scores SHALL both be one
And `verdict_class` SHALL equal `null`.

## Implementation Status (REQ-LEARN-6979)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-LEARN-6979 and SCENARIO-LEARN-6979-* | Implemented: cold-audit module, command wrapper, and dated terminal artifact. | Verified: requirement-linked read-only, journal, order, leakage, metric, disagreement, artifact, command, and 100-percent new-code coverage tests. |

## REQ-LEARN-6985: Sealed Chronological Constraint-Shift Stream

Carnot SHALL provide Exp6985 at
`python/carnot/experiment_6985_chronological_constraint_stream.py`. The command
`.venv/bin/python scripts/experiments/experiment_6985_chronological_constraint_stream.py --date 20260904`
SHALL write `results/experiment_6985_chronological_constraint_stream.json`.
This experiment SHALL create evaluation evidence only. It SHALL not perform an
online update or claim an oracle-distinct learning result.

Before construction, Exp6985 SHALL require the bare integer
`contrast_fixture_complete_score=1` from the pinned Exp6984 artifact. It SHALL
also require the frozen Exp6984 fault operators, Z3, bounded enumeration, and
at least 24 source groups that Exp6984 did not use. A failed precondition SHALL
write `blocked_chronological_constraint_stream`. Each failed row in
`gate_check_summary` SHALL name the check, expected value, and observed value.

The experiment SHALL freeze exactly 24 ordered events. The events SHALL use 24
distinct source groups that are absent from Exp6984. Four ordered shift blocks
SHALL contain six events each. Each block SHALL have a different frozen fault
mix. The last block SHALL revisit the first block's fault families on new
source groups. Frozen recurrence rows SHALL link the revisit events to their
earlier anchors.

Exactly 16 events SHALL contain one exact-positive and one exact-negative
candidate. Four events SHALL contain two exact-valid candidates. Four events
SHALL contain two exact-invalid candidates. The two tied event classes SHALL
have exact group advantage zero and SHALL require no update in a later online
experiment. Candidate order SHALL be deterministic and label-blind.

Z3 and bounded enumeration SHALL certify every candidate. Both authorities
SHALL agree on every required exact obligation. Unknown, rejected, and
nonterminal construction attempts SHALL remain in `rows` and the matching
authority tables. A rejected frozen event SHALL not be replaced after the
stream hash becomes visible to a downstream task.

The experiment SHALL write an immutable JSONL stream. It SHALL freeze event
order, candidate order, shift IDs, recurrence links, retention anchors,
held-future windows, support metrics, and exact-label commitment hashes. It
SHALL store the exact labels only in a separate immutable sealed-label JSONL
file. `stream_hash` and `sealed_label_hash` SHALL hash the exact file bytes.

For event ordinal `t`, the visibility manifest SHALL contain only public event
data from ordinals zero through `t-1`. It SHALL contain no exact label,
authority result, current event, or future event. Every manifest SHALL replay
from the immutable stream. A mutation that inserts a current or future label
SHALL fail the leakage check.

`chronological_stream_ready_score` SHALL be the bare integer one only when all
24 events are terminal, every candidate authority agrees, all counts match,
source overlap with Exp6984 is zero, every immutable file hash replays, every
visibility manifest replays, all tied groups have zero advantage, and the
last block supplies recurrence. Otherwise it SHALL equal the bare integer
zero.

The artifact SHALL contain `schema`, `experiment_id`, `run_date`,
`field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `source_disjointness_rows`,
`stream_manifest`, `stream_hash`, `sealed_label_path`, `sealed_label_hash`,
`rows`, `per_event_results`, `per_candidate_rows`, `shift_block_rows`,
`fault_family_rows`, `tie_group_rows`, `recurrence_rows`,
`retention_anchor_rows`, `held_future_window_rows`,
`visibility_manifest_rows`, `future_label_leakage_rows`,
`authority_agreement_rows`, `expected_event_count`, `observed_event_count`,
`headroom_event_count`, `all_valid_event_count`, `all_invalid_event_count`,
`chronological_stream_ready_score`, `continuous_self_learning_fixture`,
`random_seed`, `reproducibility_checksum`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.
`field_principles` SHALL give one scientific principle for every required
field, including `chronological_stream_ready_score`.

`inference_substrate` SHALL equal
`deterministic_z3_chronological_stream_no_llm`.
`continuous_self_learning_fixture` SHALL be true. `verifier_is_oracle` SHALL
be true. `verdict_class` SHALL be one of `positive`, `circular_positive`,
`null`, `blocked`, `disqualified`, or `partial`. A ready fixture SHALL use
`circular_positive`. Its `honest_verdict` SHALL start with `complete_`. A
precondition block SHALL use `verdict_class="blocked"` and the required blocked
prefix.

### SCENARIO-LEARN-6985-ORDER: Event And Candidate Order Are Frozen

Given 24 selected unused source groups,
When Exp6985 writes the stream,
Then event ordinals SHALL equal zero through 23
And each six-event block and candidate order SHALL replay from its hash.

### SCENARIO-LEARN-6985-DISJOINT: Exp6984 Sources Cannot Recur

Given the Exp6984 source manifest and the Exp6985 event sources,
When source identities are compared,
Then the overlap count SHALL be zero
And one reused source SHALL keep readiness at zero.

### SCENARIO-LEARN-6985-SHIFTS: Fault Mix Changes By Block

Given four ordered blocks of six events,
When fault-family counts are reduced by block,
Then adjacent blocks SHALL have different fault mixes
And block four SHALL revisit block one's families on new sources.

### SCENARIO-LEARN-6985-TIES: Zero Advantage Requires No Update

Given an all-valid or all-invalid candidate group,
When exact group advantage is computed,
Then the advantage SHALL equal zero
And the frozen later-action value SHALL equal `no_update`.

### SCENARIO-LEARN-6985-AUTHORITY: Every Candidate Has Two Decisions

Given any candidate in the frozen stream,
When Z3 and bounded enumeration certify it,
Then both authorities SHALL be terminal and agree
And an unknown or disagreement SHALL keep readiness at zero.

### SCENARIO-LEARN-6985-NO-FUTURE: Labels Stay Outside Earlier Views

Given event ordinal `t` and its visibility manifest,
When the manifest is replayed,
Then visible ordinals SHALL equal `range(t)`
And current or future exact labels SHALL be absent.

### SCENARIO-LEARN-6985-MUTATION: Label Injection Fails Closed

Given a valid predecessor-only visibility manifest,
When a current or future label is inserted,
Then the leakage check SHALL reject the manifest
And readiness SHALL not remain one.

### SCENARIO-LEARN-6985-RECURRENCE: The Final Block Replays Early Families

Given the first and final shift blocks,
When recurrence links are checked,
Then each final-block event SHALL link to a compatible first-block anchor
And every linked event SHALL use a new source group.

### SCENARIO-LEARN-6985-BARE: Readiness Is A Bare Integer

Given a ready, blocked, or disqualified artifact,
When its downstream readiness field is read,
Then the field SHALL be the bare integer zero or one
And a Boolean or wrapped value SHALL fail validation.

## Implementation Status (REQ-LEARN-6985)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-LEARN-6985 and SCENARIO-LEARN-6985-* | Planned: chronological stream module, command wrapper, immutable JSONL stream, sealed labels, and terminal artifact. | Planned: focused order, disjointness, shift, tie, authority, label-sealing, mutation, recurrence, artifact, command, and 100-percent new-code coverage tests. |

## REQ-CSL-7020: Transactional Counterexample Belief Ledger

Carnot SHALL provide a small explicit belief ledger in
`python/carnot/agentic/arc_belief_ledger.py`. The ledger SHALL accept only
typed, game-blind mechanic keys and immediate observation records. It SHALL
not train model weights. It SHALL reject game identity, source paths, hidden
rules, solve-registry labels, adapters, and later outcomes at every write.

Each accepted observation SHALL create a prepared transaction before it
publishes state. The journal SHALL be append-only and hash-chained. A commit
SHALL bind the exact parent and new state bytes. A failed write SHALL leave
the published state bytes unchanged. A restart SHALL abort a prepared but
unpublished transaction. It SHALL recover a published transaction that lacks
its final commit row. A rollback SHALL restore caller-held parent bytes
exactly and append a rollback receipt.

The ledger SHALL expose `snapshot`, `restart`, `rollback`, `query`, and
`audit_state` behavior. A fact SHALL start as `possible`. It SHALL become
`known` only after the configured minimum support. A conflicting immediate
observation SHALL tombstone the active refuted fact before a later query can
use it. The replacement SHALL remain `uncertain` until it gets minimum
support. A tombstoned fact SHALL remain queryable as `contradicted`. New
support for a tombstoned outcome SHALL create a new generation and record
supersession instead of changing the tombstone.

Counterexamples SHALL cluster by typed mechanic key. Capacity SHALL count
stored belief facts. Eviction SHALL be deterministic. It SHALL prefer old
tombstones and weak unsupported facts. It SHALL protect supported,
non-conflicting known facts from an unrelated poison cluster. A candidate
that cannot enter without evicting protected facts SHALL be rejected with no
published state change.

The Exp7020 command SHALL replay only the construction portion of Exp7019.
It SHALL report per-event queries, writes, rejections, support changes,
contradictions, tombstones, retention, and transaction evidence. It SHALL not
open or use sealed future rows to create a belief or answer a query. This
experiment proves the storage substrate. It SHALL make no future-utility or
ARC solve claim.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `cited_upstream_artifacts`,
`source_artifact_hashes`, `rows`, `per_event_results`, `belief_state_rows`,
`update_rows`, `rejection_rows`, `contradiction_cluster_rows`,
`tombstone_rows`, `capacity_rows`, `authority_conflict_rows`,
`supersession_rows`, `poison_rows`, `retention_rows`, `transaction_rows`,
`restart_rows`, `rollback_rows`, `leakage_check_rows`, `snapshot_hashes`,
`ledger_state_bytes`, `belief_ledger_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL give one
scientific principle for every listed field.

`inference_substrate` SHALL equal
`deterministic_arc_belief_ledger_replay_no_llm`. `verifier_is_oracle` SHALL
be false. `belief_ledger_ready_score` SHALL be the bare integer one only when
all safety fixtures pass, fresh-process replay is deterministic, the state
and journal audit passes, and zero future fields enter any write or query.
A failed precondition SHALL write
`blocked_counterexample_belief_ledger`. Its `gate_check_summary` SHALL name
the failed check, expected value, and observed value.

### SCENARIO-CSL-7020-STATES-AND-SUPPORT

- GIVEN repeated immediate observations for one typed mechanic key
- WHEN support reaches the configured minimum
- THEN the active fact moves from possible to known
- AND an unseen or conflicted query reports uncertain
- AND a tombstoned fact reports contradicted

### SCENARIO-CSL-7020-CONTRADICTION-TOMBSTONE-SUPERSESSION

- GIVEN a known active fact and a conflicting next observation
- WHEN the conflict commits
- THEN the old fact is tombstoned before the query returns
- AND the grouped counterexample records both outcomes
- AND later renewed support creates a new generation with a supersession link

### SCENARIO-CSL-7020-TRANSACTION-RESTART-ROLLBACK

- GIVEN a commit, an interruption after prepare, and a saved parent snapshot
- WHEN the store restarts and then rolls back
- THEN committed bytes survive restart
- AND the interrupted candidate is absent
- AND rollback bytes equal the saved parent bytes
- AND every journal row has a valid previous-row hash

### SCENARIO-CSL-7020-CAPACITY-AND-POISON

- GIVEN a full ledger of supported non-conflicting known facts
- WHEN an unrelated weak poison fact requests admission
- THEN the poison fact is rejected
- AND every protected fact and its support remain byte-identical
- AND ordinary weak facts use the documented deterministic eviction order

### SCENARIO-CSL-7020-AUTHORITY-AND-FAILED-WRITE

- GIVEN a repeated event ID with different observation bytes
- WHEN the second record requests a write
- THEN the ledger records an authority conflict
- AND the published state bytes remain unchanged

### SCENARIO-CSL-7020-ARTIFACT

- GIVEN the ready Exp7019 construction stream and all safety fixtures
- WHEN the Exp7020 command completes in two fresh processes
- THEN replay hashes agree and every required evidence table is terminal
- AND no held-future field appears in a write or query
- AND the ready score is one without a future-utility claim

## Implementation Status (REQ-CSL-7020)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CSL-7020 and SCENARIO-CSL-7020-* | Implemented 2026-09-05 in `arc_belief_ledger.py` and the Exp7020 runner. | RED-first focused tests cover ledger semantics, safety gates, artifact schema, and 100% of new executable code. |

## REQ-CSL-7021: Prospective ARC Belief Utility Comparison

Carnot SHALL compare three policies on later ARC decisions. The policies SHALL
be a frozen no-memory policy, a recency-only policy, and the Exp7020
counterexample-belief policy. All policies SHALL declare the same memory-item
capacity. The frozen policy SHALL use fixed candidate scores. The recency
policy SHALL keep only its newest observation rows. The belief policy SHALL
use the bounded Exp7020 fact store.

The protocol SHALL freeze its split, unit IDs, candidate rule, arm definitions,
capacity, metrics, seeds, bootstrap plan, and positive gate before it opens the
held-future sidecar. The first source attempt SHALL supply construction evidence.
Later source attempts SHALL supply held decisions. A candidate SHALL be a
distinct logged action under the decision's observable hypothesis key. A policy
may use only observations with a lower `stream_index` than the decision. It
SHALL rank before the current observation updates memory.

The target SHALL classify a logged candidate as progress-producing when its
observation changes more than one cell, touches the action coordinate, or
crosses a level boundary. A non-progress candidate SHALL be contradicted only
when its observation-time contradiction flag is true. Other changed actions
SHALL be valid. A scored unit SHALL contain at least one valid or
progress-producing candidate and at least one contradicted candidate. A unit
with fewer than two distinct candidates SHALL be unsupported. A unit without
both target classes SHALL be a no-headroom unit. Tied useful-versus-contradicted
scores SHALL receive one-half ranking credit.

The primary metric SHALL be mean held action-ranking accuracy over distinct
future decision units. Secondary metrics SHALL include contradicted-action
rate, useful-query coverage, protected-case retention, memory bytes, and query
time per unit. Aggregates SHALL reduce from per-decision rows. Confidence
intervals SHALL use a fixed-seed cluster bootstrap over decision unit IDs, not
candidate pairs or rows. Paired deltas SHALL subtract each control from the
belief policy on matched decision units.

The Exp7021 command SHALL require the bare integer
`belief_ledger_ready_score=1`. It SHALL verify the exact Exp7019 fixture hash,
the exact Exp7020 artifact hash, at least three held mechanic groups, and a
writable artifact path. A failed precondition SHALL write
`blocked_prospective_belief_utility`. Its `gate_check_summary` SHALL name the
first failed check, expected value, and observed value.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `cited_upstream_artifacts`,
`source_artifact_hashes`, `frozen_split`, `arm_definition_rows`,
`capacity_match_rows`, `rows`, `per_decision_results`,
`per_mechanic_results`, `no_headroom_rows`, `unsupported_rows`,
`action_ranking_rows`, `contradiction_avoidance_rows`, `retention_rows`,
`query_cost_rows`, `paired_delta_rows`, `confidence_intervals`,
`aggregate_row_recomputation`, `leakage_check_rows`,
`belief_utility_comparison_complete_score`,
`belief_future_utility_positive_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL give one
scientific principle for every listed field.

`inference_substrate` SHALL equal
`deterministic_prospective_arc_belief_comparison_no_llm`.
`verifier_is_oracle` SHALL be false. The comparison-complete score SHALL equal
one when every planned held-unit and arm row is present, the capacity declarations
match, aggregates recompute, and leakage checks pass. The future-utility score
SHALL equal one only when the belief arm strictly beats both controls on the
primary metric, both paired interval lower bounds are at least zero, at least
two mechanic groups have a non-negative belief-minus-recency delta, and final
protected retention does not regress. A failed value gate SHALL produce a
terminal `null` verdict. Storage safety, accepted writes, and replay fit SHALL
not satisfy the value gate.

### SCENARIO-CSL-7021-FROZEN-CHRONOLOGY

- GIVEN a frozen construction attempt and later held attempts
- WHEN each arm ranks one held decision
- THEN every memory evidence index is lower than the decision index
- AND changing the current observation does not change the ranking
- AND the current observation updates writable arms only after scoring.

### SCENARIO-CSL-7021-CAPACITY-AND-RETENTION

- GIVEN equal memory-item capacity for all three arms
- WHEN later observations exceed the recency window
- THEN the recency arm evicts its oldest rows
- AND the belief arm reports whether each protected construction fact remains
- AND bytes and query time remain explicit for every arm and unit.

### SCENARIO-CSL-7021-SUPPORT-HEADROOM-AND-TIES

- GIVEN missing candidates, no target contrast, and equal policy scores
- WHEN decision rows are classified and scored
- THEN missing candidates enter `unsupported_rows`
- AND absent target contrast enters `no_headroom_rows`
- AND a useful-versus-contradicted tie receives one-half credit.

### SCENARIO-CSL-7021-CLUSTER-BOOTSTRAP

- GIVEN scored candidate pairs nested in future decision units
- WHEN intervals and paired deltas are computed
- THEN each bootstrap draw samples distinct decision groups
- AND adding duplicate candidate rows cannot change the cluster input count.

### SCENARIO-CSL-7021-LEAKAGE-AND-RECOMPUTATION

- GIVEN held outcomes and per-decision sufficient statistics
- WHEN the artifact is reduced
- THEN sidecar and current-outcome fields are absent from policy inputs
- AND all aggregates recompute from rows
- AND aggregate drift prevents a complete or positive score.

### SCENARIO-CSL-7021-TERMINAL-GATES

- GIVEN passing preconditions and a complete comparison
- WHEN the belief policy does not pass every value condition
- THEN the comparison-complete score remains one
- AND the future-utility score is zero
- AND the verdict is terminal null without a storage-utility claim.

## Implementation Status (REQ-CSL-7021)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CSL-7021 and SCENARIO-CSL-7021-* | Implemented 2026-09-05 in `arc_prospective_belief_utility.py` and the Exp7021 runner. | RED-first tests cover chronology, equal capacity, support classes, ties, decision-cluster bootstrap, leakage, recomputation, terminal gates, and 100% of the new module. |

## REQ-CSL-7022: Independent Belief Ledger Cold Audit

Carnot SHALL audit the frozen Exp7020 belief ledger in a fresh process. The
process SHALL hide CUDA devices and use a separate network namespace with no
external route. The source tree SHALL be read-only during the child run. The
audit SHALL not call an LLM, an ARC service, a game adapter, game source, or an
online lookup. Its `inference_substrate` SHALL equal
`fresh_process_arc_belief_audit_no_llm`.

The audit SHALL require the bare integer Exp7020 ledger-ready gate and the bare
integer Exp7021 comparison-complete gate. It SHALL also require exact pinned
hashes for Exp7019, Exp7020, Exp7021, the construction fixture, the sealed
sidecar, both producer modules, both command wrappers, and the solve registry.
The producer code SHALL be importable. Bubblewrap SHALL provide the fresh
process launcher. The requested artifact path SHALL be writable. The audit
SHALL hash all sources before it parses an upstream headline. A failed
precondition SHALL write `blocked_belief_ledger_cold_audit` and preserve the
first failed check, expected value, and observed value in
`gate_check_summary`.

The child SHALL replay the 27 construction rows twice into new ledger roots.
It SHALL independently reduce state, fact, cluster, query, and transaction
evidence into a path-independent digest. The two digests SHALL match each
other and the digest recorded by Exp7020. The audit SHALL reject changed event
order, changed event bytes without a new row hash, authority conflicts, and
forbidden construction fields. A semantically identical JSON serialization
SHALL keep the replay result unchanged.

The mutation matrix SHALL cover every future or identity field denied by the
updater. It SHALL include future labels, held-future data, later outcomes,
sealed-future data, game identity, source paths, hidden rules, adapters, and
solve-registry labels. Each mutation SHALL be rejected before publication and
before query. The state bytes SHALL remain unchanged. The RED tests SHALL also
bypass the field auditor. Each bypass SHALL make the mutation row fail, so a
disabled check cannot leave the test green.

The audit SHALL independently exercise direct contradiction, supersession,
retrieval-key collision, deterministic eviction, repeated poison admission,
protected-fact retention, restart, interruption before publish, interruption
after publish, changed snapshot bytes, exact rollback, tampered journal tails,
and truncated commit tails. A clean restart SHALL recover the correct durable
boundary. A valid rollback SHALL reproduce the caller-held parent bytes
exactly. Capacity SHALL remain within both the item and byte limits. Poison
traffic SHALL not change a protected case.

The audit SHALL recompute every Exp7021 arm aggregate and every paired delta
from per-decision rows. It SHALL independently recompute the positive gate,
including strict control improvement, paired interval lower bounds, mechanic
group breadth, and protected retention. A headline or paired-row mismatch
SHALL set promotion readiness to zero and SHALL produce a disqualified
verdict. The audit SHALL not import the Exp7021 aggregation or gate helpers.

`belief_shadow_safe_score` SHALL be the bare integer one only when leakage is
zero, fresh replay is deterministic, rollback is byte-exact, capacity is
bounded, protected cases are unharmed, and every required control terminates.
`belief_promotion_ready_score` SHALL be the bare integer one only when shadow
safety is one, row recomputation agrees with every headline, and the
row-recomputed Exp7021 positive gate passes. A safe ledger without demonstrated
value SHALL use `verdict_class=null`. A leaky or non-recomputable audit SHALL
use `verdict_class=disqualified`. The verifier SHALL declare
`verifier_is_oracle=false`.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `cited_upstream_artifacts`,
`source_artifact_hashes`, `rows`, `fresh_process_rows`,
`network_isolation_rows`, `gpu_isolation_rows`,
`future_leakage_mutation_rows`, `game_identity_mutation_rows`,
`serialization_mutation_rows`, `order_mutation_rows`,
`authority_conflict_rows`, `supersession_rows`,
`retrieval_collision_rows`, `poison_rows`, `capacity_rows`, `retention_rows`,
`restart_rows`, `interruption_rows`, `rollback_rows`,
`aggregate_recomputation_rows`, `shadow_safety_gate_rows`,
`promotion_gate_rows`, `belief_shadow_safe_score`,
`belief_promotion_ready_score`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL give one scientific principle for
every listed field.

### SCENARIO-CSL-7022-PRECONDITIONS

- GIVEN changed upstream bytes, a false or wrapped upstream gate, missing code,
  no fresh-process launcher, or an unwritable result path
- WHEN Exp7022 checks inputs before parsing headlines or running mutations
- THEN it writes one blocked artifact
- AND the first failed check retains its exact expected and observed values

### SCENARIO-CSL-7022-ISOLATED-REPLAY

- GIVEN the pinned construction fixture and a clean temporary ledger root
- WHEN a network-disabled and GPU-hidden child replays it twice
- THEN both path-independent digests equal the Exp7020 construction digest
- AND game source, adapters, registry labels, online lookups, and future fields
  remain outside construction and query state

### SCENARIO-CSL-7022-MUTATION-SENSITIVITY

- GIVEN each denied future, game, source, adapter, hidden-rule, or registry field
- WHEN the field is inserted into an otherwise valid event
- THEN the write and query are rejected without changing state bytes
- AND bypassing the field auditor makes that RED mutation test fail

### SCENARIO-CSL-7022-ORDER-AUTHORITY-AND-RETRIEVAL

- GIVEN reordered rows, changed serialization bytes, a reused event authority,
  and keys that differ in one typed mechanic field
- WHEN the audit replays writes and queries
- THEN order and authority violations fail closed
- AND semantic serialization is invariant
- AND one mechanic cannot retrieve another mechanic's fact

### SCENARIO-CSL-7022-CONFLICT-CAPACITY-AND-POISON

- GIVEN a known fact, a contradictory outcome, renewed support, a full store,
  and an unrelated poison cluster
- WHEN the audit commits the clean sequence
- THEN contradiction and supersession remain visible
- AND deterministic capacity stays bounded
- AND poison cannot evict or harm protected facts

### SCENARIO-CSL-7022-RESTART-INTERRUPTION-ROLLBACK

- GIVEN committed state, prepare and publish interruptions, saved snapshots,
  and changed or truncated journal tails
- WHEN a clean process restarts and rolls back
- THEN each durable boundary recovers deterministically
- AND invalid snapshot or journal bytes are rejected
- AND valid rollback bytes equal the saved parent bytes

### SCENARIO-CSL-7022-ROW-RECOMPUTATION

- GIVEN Exp7021 per-decision, action, retention, query-cost, and paired rows
- WHEN the audit independently reduces arm metrics and clustered paired deltas
- THEN every published headline has an exact row-derived witness
- AND any headline or paired-row drift disqualifies value promotion

### SCENARIO-CSL-7022-SEPARATE-DECISIONS

- GIVEN a safe replay whose row-recomputed value gate is not positive
- WHEN Exp7022 reduces the two decisions
- THEN `belief_shadow_safe_score` equals one
- AND `belief_promotion_ready_score` equals zero
- AND `verdict_class` is `null`, not `positive`

## Implementation Status (REQ-CSL-7022)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CSL-7022 and SCENARIO-CSL-7022-* | Implemented 2026-09-05 in `arc_belief_ledger_cold_audit.py` and the Exp7022 runner. | RED-first tests cover every denied field, restricted replay, ledger controls, independent row reduction, terminal decisions, and 100% of the new module. |

## REQ-CL-7105: Independent Exact Chronological Constraint Stream

Carnot SHALL construct and seal exactly 144 unique chronological events. The
stream SHALL contain at least 12 independent groups from SAT, graph coloring,
integer arithmetic, and finite-trace temporal constraints. It SHALL use local
deterministic fixture builders and exact solvers. It SHALL not read an entrance
bank or any Exp7093 artifact.

Each event SHALL contain a decision-visible input, a candidate set, one hidden
exact label, an exact witness or counterexample, a family ID, a group ID, a
chronology index, a reuse-or-decoy status, a hardness stratum, a feedback
release point, and a canonical content hash. Each group SHALL contain reusable
structure, matched decoys, a predeclared hard slice, and protected retention
probes. Group boundaries and difficulty SHALL derive only from visible
structure before labels become available.

The producer SHALL freeze group IDs, family IDs, protected groups, memory
capacity, early-middle-late slices, feedback order, and primary Exp7106
comparisons before it opens labels. The decision view SHALL contain no label,
witness, counterexample, validity, solver receipt, or label-binding hash. The
label view SHALL be a separate immutable sidecar. Its storage order SHALL be a
label-blind permutation. Consumers MAY open a label only at or after its
declared feedback release point.

Every event, decision row, and label row SHALL use canonical JSON and a SHA-256
content seal. The full stream, decision view, and label view SHALL have separate
file hashes. An existing sealed path MAY be reused only when its bytes are
identical. Any post-seal byte change SHALL fail validation.

Before construction, the producer SHALL require 12 local groups, four families,
deterministic solver replay, readable source modules, and writable stream,
sidecar, and artifact paths. A failed check SHALL produce a schema-complete
artifact with `verdict_class=blocked` and
`inference_substrate_class=blocked_no_run`. Its `gate_check_summary` SHALL name
the failed check, expected value, and observed value.

A fresh Python process SHALL regenerate all 144 events and replay every exact
label and witness. It SHALL match the parent stream, decision, label, and
witness hashes. The producer SHALL attack one byte, one label, one chronology
index, and one group assignment. Every attack SHALL invalidate the seal.

`exact_constraint_stream_ready_score` SHALL be the bare integer one only when
the stream has exactly 144 unique events, at least 12 groups, all four families,
all hardness strata, enough hard and decoy rows, no conflicts, no future-label
leakage, exact fresh-process witness parity, and four detected mutations. A
ready stream SHALL use `verdict_class=circular_positive` because exact solvers
construct and replay the labels. This result SHALL not imply model quality.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `inference_substrate_class`, `execution_venue`,
`duration_s`, `source_artifact_hashes`, `stream_path`, `decision_view_path`,
`label_view_path`, `stream_hash`, `decision_view_hash`, `label_view_hash`,
`event_count`, `group_count`, `family_count`, `frozen_group_ids`,
`frozen_family_ids`, `frozen_protected_group_ids`,
`frozen_capacity_schedule`, `frozen_slice_definitions`,
`frozen_primary_comparisons`, `rows`, `event_rows`, `group_rows`,
`family_rows`, `hardness_rows`, `reuse_rows`, `decoy_rows`,
`retention_probe_rows`, `chronology_rows`, `witness_replay_rows`,
`uniqueness_rows`, `conflict_rows`, `leakage_rows`, `mutation_attack_rows`,
`exact_constraint_stream_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL give one
scientific principle for every required field. `inference_substrate` SHALL
describe deterministic exact local constraint generation and fresh-process
witness replay. A successful run SHALL use
`inference_substrate_class=no_model_load` and `execution_venue=host`.

### SCENARIO-CL-7105-PRECONDITIONS: Missing Independent Support Blocks

- GIVEN fewer than 12 groups, fewer than four families, unstable local solvers,
  an unreadable source, or an unwritable destination
- WHEN Exp7105 evaluates preconditions
- THEN it writes a schema-complete blocked artifact
- AND the first failed check preserves exact expected and observed values.

### SCENARIO-CL-7105-IDENTITY: Duplicates Collisions And Conflicts Reject

- GIVEN a duplicate event, one group assigned to two families, or one decision
  assigned contradictory labels
- WHEN stream conformance runs
- THEN readiness remains zero
- AND a stable identity, group-collision, or label-conflict reason is present.

### SCENARIO-CL-7105-CHRONOLOGY: Order And Feedback Stay Prospective

- GIVEN an unstable chronology index, an early feedback point, or a decision
  row that contains a current or future label field
- WHEN chronology and leakage checks run
- THEN the stream fails closed
- AND no future label can enter the decision view.

### SCENARIO-CL-7105-COVERAGE: Hard Decoy And Retention Rows Are Required

- GIVEN fewer than 144 events, fewer than 12 groups, an absent family or
  hardness stratum, or insufficient hard, decoy, reuse, or retention rows
- WHEN the readiness gate reduces row evidence
- THEN the score is zero
- AND the shortfall is named.

### SCENARIO-CL-7105-WITNESS: Every Hidden Label Replays Exactly

- GIVEN a label or witness that differs from the local exact solver
- WHEN parent or fresh-process replay runs
- THEN witness parity fails
- AND the stream cannot be ready.

### SCENARIO-CL-7105-SEAL: Post-Seal Mutations Invalidate Content

- GIVEN one changed byte, label, chronology index, or group assignment
- WHEN the sealed stream is validated
- THEN each mutation fails independently
- AND an existing path with changed bytes cannot be overwritten.

### SCENARIO-CL-7105-ARTIFACT: Rows Recompute Readiness And Verdict

- GIVEN a ready, blocked, or mutated Exp7105 artifact
- WHEN an independent validator recomputes hashes, counts, coverage, replay,
  readiness, verdict, and checksum
- THEN a consistent artifact passes
- AND a forged score, hash, row, verdict, or checksum fails.

## Implementation Status (REQ-CL-7105)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7105 and SCENARIO-CL-7105-* | Planned: `python/carnot/experiment_7105_v623_exact_constraint_stream.py`; command wrapper; three immutable stream views; terminal artifact. | Planned: RED-first identity, chronology, leakage, coverage, witness, mutation, artifact, command, and new-code coverage tests. |

## REQ-CL-7106: Delayed Procedural Memory On The Sealed Exact Stream

Carnot SHALL compare five frozen arms on every Exp7105 decision event in its
sealed chronological order: `delayed_procedural`, `raw_trace`,
`equal_context_replay`, `write_while_deciding`, and `no_memory`. One fixed
deterministic candidate policy SHALL receive the identical candidate set in
all arms. Model weights SHALL remain frozen and no model SHALL load.

The comparison SHALL require Exp7105 readiness, exactly 144 decision rows, at
least 12 groups, separate decision and label seals, exact witnesses, and the
frozen capacity and time slices. A failed precondition SHALL produce a
schema-complete blocked artifact with no measurement rows,
`inference_substrate_class=blocked_no_run`, and the exact failed check,
expected value, and observed value.

Every arm-event row SHALL bind the visible-input and candidate-set hashes. It
SHALL record the common context byte budget, one retrieval slot, the common
item and byte capacity, one decision, one exact validation, and the common FIFO
eviction rule. The persistent arms SHALL charge one common record slot even
when their serialized representations differ. The stricter item or byte bound
SHALL control admission. Replay and no-memory arms SHALL preserve the empty
persistent-state hash for the full run.

The delayed arm SHALL read an immutable pre-decision snapshot. It SHALL seal
its decision before exact feedback opens. It SHALL then derive a scoped
abstract procedure from the signed outcome, retain source provenance, validate
the proposal, and atomically replace its state between events. Its current
event SHALL not retrieve or use its own update. The write-while-deciding arm
MAY install its declared provisional self-generated record before the decision,
but it SHALL not see current exact feedback. All other early writes SHALL fail.
Every committed transition SHALL bind parent and child hashes and remain within
the frozen common capacity.

The artifact SHALL retain paired rows for every event and arm. It SHALL report
all groups, families, early-middle-late slices, hard versus ordinary events,
reuse versus decoy events, every capacity slice, retrieval, decisions,
feedback, updates, transactions, hashes, evictions, paired later-event deltas,
predeclared confidence intervals, negative transfer, and protected retention.
Reducers SHALL keep adverse or failed groups and SHALL recompute all headline
scores from rows.

`procedural_memory_comparison_complete_score` SHALL equal one only for all 720
arm-event rows, all expected aggregate cells, matched budgets, sealed feedback
order, atomic transactions, bounded state, and complete receipts.
`procedural_memory_value_ready_score` SHALL equal one only when the delayed arm
has a positive predeclared 95% lower confidence bound over each of the four
controls on the 96 middle-and-late events, no negative hard-group delta, no
capacity violation, and no protected-retention failure. A complete comparison
that misses any value condition SHALL use `verdict_class=null`, never partial.

The artifact SHALL include all fields required by the Exp7106 task contract.
`field_principles` SHALL explain every field. The substrate SHALL be
`deterministic prospective candidate policy with verifier-signed bounded external memory`.
A completed run SHALL use `inference_substrate_class=no_model_load`,
`execution_venue=host`, `model_weights_changed=false`, and
`verifier_is_oracle=false` because the exact verifier signs feedback but does
not select the current candidate.

### SCENARIO-CL-7106-PRECONDITIONS: A Broken Upstream Seal Blocks

- GIVEN a missing readiness score, row, group, label sidecar, witness, seal,
  capacity schedule, or slice definition
- WHEN Exp7106 checks Exp7105 before its first decision
- THEN it writes a row-free blocked artifact
- AND it preserves the exact first failed check, expected value, and observed value.

### SCENARIO-CL-7106-ISOLATION: Decisions Cannot Read Current Or Future Labels

- GIVEN any arm-event decision
- WHEN its visible evidence and causal sequence are checked
- THEN current and future exact feedback are absent at decision time
- AND the sealed decision precedes the exact label and witness release.

### SCENARIO-CL-7106-MATCHED: Resource And State Contracts Stay Equal

- GIVEN the five rows for one event
- WHEN comparison controls are checked
- THEN context bytes, retrieval slots, capacity, decision budget, validation,
  eviction, chronology, visible input, and candidates match
- AND replay and no-memory arms expose no hidden persistent state.

### SCENARIO-CL-7106-TRANSACTION: Delayed Commits Are Atomic And Between Events

- GIVEN a validated delayed procedural update
- WHEN the memory transition commits
- THEN its parent is the read-only decision snapshot
- AND the atomic child hash becomes visible only after feedback and before the next event.

### SCENARIO-CL-7106-CHRONOLOGY: Reorders And Dropped Rows Reject

- GIVEN reordered events, a missing arm-event row, or a duplicate identity
- WHEN the artifact validator reconstructs the paired panel
- THEN comparison completeness remains zero
- AND the exact chronology or coverage error is reported.

### SCENARIO-CL-7106-AGGREGATES: Hard Cases And Protected Retention Stay Row Derived

- GIVEN changed group, hard-case, protected-retention, or capacity evidence
- WHEN cold reducers recompute the artifact
- THEN stale aggregates reject
- AND a delayed-arm hard regression or protected forgetting forces the value score to zero.

### SCENARIO-CL-7106-VERDICT: Headline Scores And Terminal Class Agree

- GIVEN a complete positive, complete null, or blocked artifact
- WHEN its rows, scores, verdict class, and honest verdict are checked
- THEN only the row-derived terminal combination passes
- AND a positive headline with a null class, or a null headline with a positive class, rejects.

## Implementation Status (REQ-CL-7106)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7106 and SCENARIO-CL-7106-* | Implemented 2026-09-07: `python/carnot/experiment_7106_v623_procedural_memory_csl.py`; command wrapper; terminal artifact. | Verified: RED-first isolation, matched-budget, state, chronology, transaction, aggregate, retention, verdict, command, and 100% new-code coverage tests. |

## REQ-CL-7107: Fresh-Process Continual-Memory Cold Audit

Carnot SHALL audit Exp7106 in a fresh Python process without loading an LLM.
The audit SHALL capture and hash all input bytes before it decodes JSON. It
SHALL require Exp7106 `procedural_memory_comparison_complete_score=1`, the
sealed Exp7105 stream, exactly 144 decision and feedback records, exactly 720
arm-event rows, and complete decision, update, transaction, snapshot-hash, and
content-hash records. A missing or changed prerequisite SHALL produce a
schema-complete blocked artifact. The artifact SHALL use
`inference_substrate_class=blocked_no_run`. Its `gate_check_summary` SHALL name
the first failed check, expected value, and observed value.

The audit SHALL start from empty private state. It SHALL rebuild each arm in
chronological order from sealed decisions, exact feedback, validation records,
and commit records. It SHALL independently derive all proposed records,
retrievals, decisions, signatures, FIFO evictions, parent hashes, provisional
writes, and post-commit hashes. It SHALL compare each reconstructed
pre-decision and post-commit hash with Exp7106. The current event SHALL not read
its own feedback. Every accepted update SHALL carry a valid exact-verifier
signature and complete source provenance. Model weights SHALL stay frozen.

The audit SHALL recompute per-event results and every early, middle, late,
group, family, hard, ordinary, reuse, decoy, capacity, negative-transfer, and
protected-retention metric from the 720 immutable rows. It SHALL recompute each
paired test from aligned event rows. It SHALL not use pooled producer metrics
as inputs. Producer aggregates SHALL serve only as parity targets.

The audit SHALL attack missing and duplicate events, event reorder, snapshot
substitution, hash mutation, unsigned feedback, a valid-looking poisoned
outcome, an invalid witness, capacity overflow, a stale parent, partial
prepare, partial commit, crashes before and after commit, duplicate commit,
rollback drift, and aggregate or verdict mismatch. Each unsafe mutation SHALL
fail closed or restore the exact prior hash. A committed adverse update SHALL
roll back to byte-identical parent state. A changed event order SHALL reject
before it can become an alternate result.

`continual_memory_cold_audit_ready_score` SHALL equal the bare integer one only
when reconstruction, metric parity, atomicity, rollback, capacity, poison
rejection, and protected retention all pass. Audit readiness SHALL remain
separate from Exp7106's procedural-memory value score. A safe audit MAY be
positive when that upstream scientific score is null. A completed adverse
audit SHALL use `verdict_class=null`. Missing upstream state SHALL use
`verdict_class=blocked`. Only recoverable incomplete work owned by Exp7107 MAY
use `verdict_class=partial`.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `inference_substrate_class`, `execution_venue`,
`duration_s`, `source_artifact_hashes`, `upstream_gate_receipt`, `stream_hash`,
`transaction_log_hash`, `snapshot_hash_rows`, `reconstruction_rows`,
`event_replay_rows`, `metric_recomputation_rows`, `paired_test_rows`,
`protected_retention_rows`, `capacity_rows`, `eviction_rows`,
`future_label_isolation_rows`, `signature_rows`, `poison_attack_rows`,
`reorder_attack_rows`, `stale_parent_rows`, `partial_write_rows`,
`crash_recovery_rows`, `rollback_rows`, `mutation_attack_rows`, `rows`,
`producer_auditor_parity_rows`, `model_weights_changed`,
`continual_memory_cold_audit_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. It SHALL also contain normal schema,
identity, date, and runtime-isolation fields. `field_principles` SHALL give one
scientific principle for every top-level field. The successful substrate SHALL
be `fresh-process deterministic memory and transaction replay`. Its class SHALL
be `aggregation`, its venue SHALL be `host`, `model_weights_changed` SHALL be
false, and `verifier_is_oracle` SHALL be false.

### SCENARIO-CL-7107-PRECONDITIONS: Missing Or Changed Upstream State Blocks

- GIVEN a missing Exp7106 completion score, sealed stream, immutable record,
  snapshot hash, or content hash
- WHEN Exp7107 captures and checks its inputs
- THEN it emits a row-free blocked artifact
- AND the first failed check preserves exact expected and observed values.

### SCENARIO-CL-7107-RECONSTRUCTION: Empty-State Replay Matches Every Hash

- GIVEN all sealed decisions, feedback, validation, and commit records
- WHEN the fresh process replays them from empty private state
- THEN each derived decision, pre-decision hash, and post-commit hash matches
- AND a missing event, duplicate event, substituted snapshot, or changed hash rejects.

### SCENARIO-CL-7107-METRICS: Rows Own Every Aggregate And Paired Test

- GIVEN the complete immutable arm-event panel
- WHEN the auditor reduces all strata and aligned comparisons
- THEN every metric and paired test comes from rows
- AND an aggregate, paired result, completion score, or verdict mismatch rejects.

### SCENARIO-CL-7107-POISON: Unsigned Or False Feedback Cannot Mutate State

- GIVEN unsigned feedback, a valid-looking poisoned label, or an invalid witness
- WHEN the auditor validates the proposed update
- THEN the update rejects before publication
- AND the exact parent hash remains active.

### SCENARIO-CL-7107-ATOMICITY: Partial Writes And Crashes Recover Exactly

- GIVEN a stale parent, overflow, truncated prepare, truncated commit, crash
  before commit, crash after commit, or duplicate commit
- WHEN recovery opens the transaction
- THEN it selects only the exact parent or committed child boundary
- AND any ambiguous bytes fail closed to the exact prior hash.

### SCENARIO-CL-7107-ROLLBACK: Adverse Commits Have Deterministic Inverses

- GIVEN a verifier-signed update that commits and is later judged adverse
- WHEN rollback applies the recorded inverse
- THEN the restored bytes and hash equal the exact parent
- AND drifted rollback evidence rejects without changing that parent.

### SCENARIO-CL-7107-CAPACITY: Frozen Eviction Preserves Protected Records

- GIVEN each frozen capacity slice and FIFO policy
- WHEN replay reaches or exceeds the item bound
- THEN overflow never becomes active, declared evictions match reconstruction,
  and protected probes survive within capacity.

### SCENARIO-CL-7107-VERDICT: Audit Readiness And Scientific Value Stay Separate

- GIVEN a complete safe replay with a positive or null Exp7106 value score
- WHEN Exp7107 derives its terminal class
- THEN audit readiness depends only on cold-audit safety evidence
- AND blocked, null, partial, or positive text agrees with the closed class.

## Implementation Status (REQ-CL-7107)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7107 and SCENARIO-CL-7107-* | Implemented: fresh-process replay module, command wrapper, and terminal artifact. | Verified: RED-first replay, metric, poison, atomicity, rollback, capacity, verdict, command, and 100% new-module line-coverage tests. |

## REQ-CL-7183: Immutable Delayed-Feedback Supersession Stream

Carnot SHALL construct exactly 240 chronological events in four frozen
60-event regimes: `stable`, `superseded_rule`, `recurrence`, and
`conflicting_poisoned_feedback`. The stream SHALL contain six constraint
families. Four whole families SHALL support adaptation. Two whole families
SHALL support transfer evaluation only. The seed, regime order, family split,
entities, numeric values, and source-version transitions SHALL be fixed before
the first event is built.

The producer SHALL write separate immutable decision, released-feedback, and
evaluator-truth views. An ordinary feedback record SHALL have release index
`decision_index + 3`. Decision inputs SHALL exclude current labels, regime IDs,
future feedback, supersession flags, evaluator truth, corruption status, and
revocation receipts. Source supersession SHALL use a new source version and a
separate explicit revocation receipt. Exactly 24 feedback records SHALL be
corrupted by a fixed-seed selection. Evaluator truth SHALL retain an exact
contradictory witness for every corrupted record.

Each adaptation family SHALL have a frozen rolling validation partition that
is disjoint from commit-support events. A decision MAY name only validation
records whose feedback was released before that decision. Transfer-family
labels and all labels from the final 60-event audit segment SHALL be unavailable
for commit selection. A frozen recurrence subset SHALL remain visible to the
evaluator for later forgetting measurement.

The producer SHALL materialize the same event order and view hashes for
`no_memory`, `static_rule`, `fifo_replay`, and `revocable_template`. Each arm
SHALL receive the same decision-context budget, 4 KiB memory charge, two record
inspection slots, and eight-template limit. A no-memory arm SHALL retain the
same charged budget even when it does not use it. The producer SHALL seal the
family, regime, held-out, operation, arm, budget, feedback, and recurrence
manifests. It SHALL also seal a chronological availability matrix.

Before construction, Exp7183 SHALL check the driving specification, required
source bytes and hashes, V633 task identity, absence of same-milestone upstream
gates, deterministic exact evaluators, required local tools, and writable
destinations. A failed external prerequisite SHALL produce a terminal blocked
artifact. Its `gate_check_summary` SHALL retain the exact check, upstream,
field, expected value, and observed value.

`stream_ready_score` SHALL be the bare integer one only when all 240 events,
960 arm rows, four regimes, six families, recurrence cells, transfer cells,
three-event releases, held-out restrictions, 24 corruption witnesses,
revocation receipts, exact independent label agreements, immutable hashes, and
leakage checks pass. Readiness certifies only the controlled stream. Exp7183
SHALL perform no learning and SHALL make no accuracy claim. A ready artifact
SHALL use `inference_substrate_class=cpu_exact_solver_or_simulator`,
`verifier_is_oracle=true`, and `verdict_class=circular_positive`.

The artifact SHALL contain the task's principle-annotated fields:
`field_principles`, `status`, `preconditions_checked`, `run_date`,
`inference_substrate`, `execution_venue`, `duration_s`,
`source_artifact_hashes`, `rows`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, `honest_verdict`, `inference_substrate_class`,
`stream_ready_score`, `decision_view_path`, `feedback_schedule`,
`heldout_manifest`, `regime_rows`, and `template_operation_contract`.
It SHALL also retain all sealed view paths and hashes, immutable manifests,
byte budgets, availability rows, event rows, exact agreement rows, corruption
witness rows, revocation rows, recurrence rows, transfer rows, leakage rows,
and arm materialization rows needed to recompute readiness.

### SCENARIO-CL-7183-PRECONDITIONS: Missing Or Changed Inputs Block

- GIVEN a missing requirement, required source, source hash, local tool,
  V633 contract field, exact evaluator, or writable destination
- WHEN Exp7183 checks preconditions before stream construction
- THEN it writes a schema-complete terminal blocked artifact when possible
- AND its failed gate records upstream, field, expected, and observed values.

### SCENARIO-CL-7183-CHRONOLOGY: Three-Event Delay Prevents Leakage

- GIVEN any chronological decision
- WHEN its decision view and availability row are inspected
- THEN only feedback with release index at or before that decision is available
- AND the current label, regime, future feedback, supersession, and audit fields
  are absent from the decision input.

### SCENARIO-CL-7183-SUPERSESSION: Versions Revoke Explicitly

- GIVEN the first event for a family after a rule change or recurrence
- WHEN its delayed feedback becomes available
- THEN a signed receipt revokes the preceding source version
- AND evaluator truth preserves the old, new, and recurrent rule identities.

### SCENARIO-CL-7183-POISON: Corruption Has Exact Contradictory Witnesses

- GIVEN the fixed poison seed
- WHEN feedback and evaluator truth are compared
- THEN exactly 24 feedback labels contradict evaluator truth
- AND every contradiction has an independently replayable exact witness.

### SCENARIO-CL-7183-HELDOUT: Commit Selection Cannot Read Evaluation Labels

- GIVEN transfer events or final audit-segment events
- WHEN commit-selection availability is reduced
- THEN their labels remain sealed from commit selection
- AND adaptation validation buffers include only disjoint, previously released
  validation events from the same family.

### SCENARIO-CL-7183-MATCHED-ARMS: Four Streams Are Byte-Identical

- GIVEN one event materialized for all four arms
- WHEN arm receipts are compared
- THEN event order, decision hash, feedback schedule, and all resource charges
  match exactly
- AND the arm name is the only treatment assignment difference.

### SCENARIO-CL-7183-READINESS: Rows And Seals Own The Verdict

- GIVEN a complete, blocked, or mutated artifact
- WHEN validation recomputes counts, hashes, exact labels, held-out cells,
  recurrence cells, corruption witnesses, availability, and arm matching
- THEN only a complete sealed stream receives `stream_ready_score=1`
- AND readiness does not imply learning or predictive accuracy.

## Implementation Status (REQ-CL-7183)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7183 and SCENARIO-CL-7183-* | Implemented 2026-09-10: Exp7183 module, command wrapper, immutable views and manifests, and terminal artifact. | Verified: RED-first chronology, supersession, poison, held-out, arm matching, artifact, command, and 100% new-module line coverage tests. |

## REQ-CL-7184: Revocable Constraint Template Continuous Learning

Carnot SHALL compare `no_memory`, `static_rule`, `fifo_replay`, and
`revocable_template` on every Exp7183 event. It SHALL run three predeclared
inspection-order seeds. Each seed SHALL retain the sealed chronological event
order. One deterministic CPU candidate policy SHALL make every decision. No
model SHALL load or mutate.

The comparison SHALL require the exact same-milestone Exp7183 gate
`stream_ready_score == 1`. Before measurement, it SHALL check the driving
requirement, required source bytes and hashes, task identity, gate fields,
sealed view hashes, local tools, and output paths. An external precondition
failure SHALL write a schema-complete terminal blocked artifact. Its
`gate_check_summary` SHALL name the check, upstream, field, expected value, and
observed value.

Every arm SHALL receive the same 4 KiB memory charge and two record inspection
slots per event. The three stateful arms SHALL remain within 4 KiB of actual
serialized state. The no-memory arm SHALL report the same charged resources
and zero used resources. Revocable memory SHALL hold at most eight active
templates. A candidate template SHALL use only the frozen
`revocable_constraint_template.v1` operation grammar and previously released
adaptation-family feedback.

A template addition SHALL add executable constraint structure. It SHALL not
change a constraint weight. Admission SHALL require at least three distinct
released, non-corrupted exact witnesses for one family and source version.
Family credit SHALL equal prior verified catches minus prior harmful
rejections. Admission SHALL also require no regression on the disjoint,
previously released validation buffer. Unsupported, corrupted, held-out,
transfer-family, current, and future feedback SHALL reject before mutation.
Rejections SHALL remain in an append-only rejection ledger.

Every action and pre-decision memory hash SHALL seal before feedback release.
Counters and memory SHALL update only between events. When newly released exact
evidence invalidates an active template's source version, the controller SHALL
revoke it. Revocation SHALL append a lineage-linked ledger row and retain the
old template as inactive history. Commit selection SHALL not read evaluator
regime names, held-out labels, or future-support scores.

The artifact SHALL contain one complete row for every event, seed, and arm. A
row SHALL include its sealed decision, one exact post-decision verification
call, memory delta, lineage reference, feedback access receipt, charged and
used resources, lookup latency, and update latency. Reducers SHALL report
future-segment error, false acceptance, transfer, recurrence retention, and
actual measured costs. Confidence intervals SHALL use paired chronological
event-block bootstrap draws. They SHALL not treat arm rows as independent.

`memory_run_complete_score` SHALL be the bare integer one only when all 2,880
rows, equal resource charges, inspection bounds, state bounds, action seals,
feedback timing, ledger rows, lineage rows, and row-derived aggregates pass.
`memory_value_score` SHALL be one only when the revocable arm's future-error
delta has a 95 percent upper bound below zero against both `static_rule` and
`fifo_replay`, false accepts do not increase, and recurrence retention does not
decrease. A complete comparison that misses any value condition SHALL finish
with `verdict_class=null`.

The artifact SHALL include `field_principles`, `status`,
`preconditions_checked`, `run_date`, `inference_substrate`, `execution_venue`,
`duration_s`, `source_artifact_hashes`, `rows`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, `honest_verdict`, `inference_substrate_class`,
`memory_run_complete_score`, `memory_value_score`,
`continuous_self_learning_task`, `memory_transition_rows`,
`template_lineage_rows`, `feedback_access_rows`,
`no_model_weight_mutation`, and `cost_rows`. `field_principles` SHALL state the
evidence reason for every required field. A completed run SHALL use
`inference_substrate_class=cpu_exact_solver_or_simulator`. It SHALL report a
CPU prototype scope and list SIMD or FPGA template matching only as future
work. It SHALL not invent hardware timing or enable the production pipeline.

### SCENARIO-CL-7184-PRECONDITIONS: Missing Or Changed Evidence Blocks

- GIVEN a missing requirement, source, hash, same-milestone gate, sealed view,
  local tool, or writable destination
- WHEN Exp7184 checks preconditions before its first policy decision
- THEN it writes a row-free terminal blocked artifact when possible
- AND the failed gate preserves its upstream, field, expected, and observed values.

### SCENARIO-CL-7184-STRUCTURAL-ADDITION: Released Errors Add A Template

- GIVEN three distinct released exact error witnesses for one adaptation family
  and source version
- WHEN family credit is positive and prior released validation does not regress
- THEN the revocable arm can commit one executable constraint template
- AND the active structure and memory hash change without a weight update.

### SCENARIO-CL-7184-REJECT: Unsupported Feedback Cannot Commit

- GIVEN corrupted, held-out, transfer-family, current, future, or fewer than
  three distinct supporting witnesses
- WHEN the controller evaluates a template proposal
- THEN the proposal enters the append-only rejection ledger
- AND active template bytes remain unchanged.

### SCENARIO-CL-7184-REVOCATION: Exact Version Evidence Retires Stale Structure

- GIVEN an active template and newly released exact evidence for a different
  source version in the same family
- WHEN the evidence invalidates the active source version
- THEN the old template becomes inactive before the next decision
- AND an append-only revocation and lineage row bind the source and replacement.

### SCENARIO-CL-7184-CHRONOLOGY-AND-MATCHING: Decisions Stay Sealed And Matched

- GIVEN one event across all arms and inspection-order seeds
- WHEN its decision is made
- THEN the action and pre-decision memory hash seal before feedback release
- AND each arm has equal charged bytes, inspection slots, candidates, and order
- AND no commit input contains a regime name, held-out label, or future score.

### SCENARIO-CL-7184-BLOCK-BOOTSTRAP: Paired Chronological Blocks Own Intervals

- GIVEN matched future-segment decisions nested in chronological blocks
- WHEN error deltas and 95 percent intervals are reduced
- THEN each draw resamples paired blocks and keeps arm outcomes aligned
- AND duplicating an arm row cannot create an independent observation.

### SCENARIO-CL-7184-TERMINAL: Completion And Value Stay Separate

- GIVEN all rows and causal receipts pass but either future comparison,
  false-accept safety, or recurrence retention misses its gate
- WHEN Exp7184 computes the terminal result
- THEN `memory_run_complete_score` remains one
- AND `memory_value_score` is zero
- AND the verdict is terminal null rather than partial or positive.

## Implementation Status (REQ-CL-7184)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7184 and SCENARIO-CL-7184-* | Planned: deterministic controller module, command wrapper, and terminal artifact. | Planned: RED-first preflight, bounds, addition, rejection, revocation, chronology, bootstrap, artifact, command, and 100-percent new-code coverage tests. |

## REQ-CL-7185: Fresh-Process Revocable-Memory Cold Audit

Carnot SHALL audit Exp7184 in a fresh Python process without loading an LLM.
The audit SHALL capture and hash all required source and artifact bytes before
it decodes JSON. It SHALL require the exact same-milestone Exp7184 gate
`memory_run_complete_score == 1` and milestone `2026.09.633`. It SHALL check
the V633 task identity, local tools, writable result and checkpoint paths, raw
row counts, and authority-sidecar availability before measurement. An external
precondition failure SHALL produce a row-free terminal blocked artifact. The
artifact SHALL use `inference_substrate_class=blocked_no_run`. Its
`gate_check_summary` SHALL name the failed check, upstream, field, expected
value, and observed value.

The audit SHALL recompute each decision's exact label, correctness, error, and
false acceptance from the raw Exp7184 decision and the sealed Exp7183 truth
sidecar. It SHALL recompute future, false-acceptance, transfer, and recurrence
metrics from these per-event rows. Producer summaries SHALL be parity targets
only. It SHALL count actual `add_template` controller lineage operations. It
SHALL not count rejected or stored proposals as additions.

The audit SHALL reconstruct revocable state from an empty state. It SHALL
verify action seals, before and after hashes, addition support, family credit,
source versions, revocation timing, and causal feedback availability. A commit
SHALL use only feedback released before its commit index. It SHALL not use the
current evaluation label, held-out label, evaluator regime name, future score,
or corrupted feedback. The audit SHALL save the last valid state before the
recurrence segment. A new isolated Python process SHALL reload those exact
bytes before it audits recurrence decisions.

The audit SHALL run six isolated mutations: early feedback exposure, stale
source kept active, missing revocation, poisoned feedback accepted, an instance
ID retained in a template, and a forged before or after hash. Each mutation
SHALL fail only its named assertion. It SHALL measure rollback to the last
valid checkpoint and require byte equality and hash equality.

The audit SHALL rerun the bounded CPU controller with real family credit and
seeded shuffled family credit on the same stream, event order, and capacity.
It SHALL retain one row per event and control. The audit SHALL delete committed
templates from the reconstructed real-credit state. If no previously improved
future decision changes, the mechanism SHALL be classified as decorative.
Prediction code, memory capacity, event order, and decision inputs SHALL stay
fixed across both credit arms and the deletion control.

`memory_audit_complete_score` SHALL be the bare integer one only when all 2,880
original rows, raw-label recomputations, causal receipts, cold-reload rows,
six mutation rows, rollback rows, credit-control rows, and deletion controls
pass their completeness checks. `memory_promotion_score` SHALL be one only when
Exp7184 also has `memory_value_score == 1`, causal timestamps are valid, cold
retention is exact, rollback is byte-identical, shuffled credit does not
replicate the real mechanism, and deletion changes a previously improved
future decision. A complete safe audit of non-improving memory SHALL use
`verdict_class=null`. External absence SHALL use `verdict_class=blocked`.

The artifact SHALL contain `field_principles`, `status`,
`preconditions_checked`, `run_date`, `inference_substrate`, `execution_venue`,
`duration_s`, `source_artifact_hashes`, `rows`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, `honest_verdict`, `inference_substrate_class`,
`memory_audit_complete_score`, `memory_promotion_score`, `mutation_rows`,
`rollback_rows`, and `credit_control_rows`. It SHALL also retain raw metric,
causal, reconstruction, cold-reload, addition, revocation, deletion, process
isolation, and upstream gate receipts. `field_principles` SHALL explain the
evidence reason for every required field. A completed audit SHALL use
`inference_substrate_class=no_model_load`. It SHALL claim no live LLM or
hardware result. The verifier SHALL declare that it reads the same authority
sidecar that supplies exact labels.

### SCENARIO-CL-7185-PRECONDITIONS: Missing Or Changed Evidence Blocks

- GIVEN a missing requirement, source, hash, task identity, same-milestone
  gate, raw view, local tool, or writable destination
- WHEN Exp7185 checks preconditions before measurement
- THEN it writes a schema-complete row-free blocked artifact when possible
- AND the failed gate retains its upstream, field, expected, and observed values.

### SCENARIO-CL-7185-RAW-REDUCTION: Sidecar Labels Own Metrics

- GIVEN the complete Exp7184 decision panel and sealed Exp7183 truth sidecar
- WHEN the cold parser joins them by event identity
- THEN each exact label, correctness, error, and false acceptance is rebuilt
- AND every metric uses rebuilt rows while producer aggregates serve only as parity targets.

### SCENARIO-CL-7185-CAUSALITY: Commits Use Released Evidence Only

- GIVEN a template addition, rejection, replacement, or revocation
- WHEN its evidence and action timestamps are checked
- THEN each commit input was released before the commit index
- AND current, future, held-out, regime, and corrupted evidence did not select it
- AND only committed `add_template` lineage rows count as controller additions.

### SCENARIO-CL-7185-COLD-RETENTION: Recurrence Reload Uses A Fresh Process

- GIVEN the last valid pre-recurrence memory checkpoint
- WHEN a new isolated Python process reloads its canonical bytes
- THEN its first state hash equals the saved hash
- AND its recurrence decisions and memory hashes equal the original cold targets.

### SCENARIO-CL-7185-MUTATIONS: Six Isolating Attacks Fail Closed

- GIVEN one mutation for early feedback, stale active source, missing
  revocation, accepted poison, retained instance ID, or a forged state hash
- WHEN the named assertion evaluates that otherwise valid record
- THEN the mutation fails its named assertion
- AND the other isolation assertions remain unchanged.

### SCENARIO-CL-7185-ROLLBACK: Last Valid Bytes Restore Exactly

- GIVEN a valid checkpoint followed by an adverse or forged transition
- WHEN rollback restores the checkpoint
- THEN the recovered bytes equal the checkpoint bytes
- AND their SHA-256 identities match exactly.

### SCENARIO-CL-7185-CREDIT-CONTROL: Shuffled History Tests Assignment

- GIVEN one frozen stream, controller, capacity, and random seed
- WHEN real family credit and seeded shuffled family credit run separately
- THEN every event-control row is retained
- AND treatment assignment is the only control difference
- AND shuffled credit cannot inherit a real-credit promotion by name.

### SCENARIO-CL-7185-DELETION: Useful Structure Must Affect Decisions

- GIVEN committed templates that improved at least one future decision
- WHEN those templates are removed from the same saved state
- THEN at least one previously improved future decision changes
- AND no changed decision classifies the mechanism as decorative.

### SCENARIO-CL-7185-TERMINAL: Completion And Promotion Stay Separate

- GIVEN a complete safe audit whose Exp7184 value gate is zero
- WHEN Exp7185 derives its terminal result
- THEN `memory_audit_complete_score` remains one
- AND `memory_promotion_score` is zero
- AND the terminal verdict is null rather than partial or positive.

## Implementation Status (REQ-CL-7185)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7185 and SCENARIO-CL-7185-* | Planned: independent cold parser, fresh-process recurrence reload, control rerun, and terminal artifact. | Planned: RED-first parser, causality, mutation, rollback, credit, deletion, verdict, command, and new-code coverage tests. |

## REQ-CL-7198: Bounded Pending-Feedback Constraint Stream

Carnot SHALL generate ten independent, fixed-seed streams before it runs any
learner or scheduler. Each stream SHALL contain 1,024 events. The disjoint
windows SHALL contain 128 warmup, 128 online-validation, 512 prospective, 128
recurrence, and 128 poison-and-rollback events. Each window SHALL contain equal
counts from four public numeric-predicate families. Every family SHALL use one
hidden parameter from the finite integer domain 0 through 32. Stable, shifted,
and recurrent parameter regimes SHALL be fixed before outcome access.

The producer SHALL write a learner-visible public stream and a separate
authority sidecar. Public rows SHALL contain the shared family grammar and the
numeric observation. They SHALL omit parameters, labels, regime transitions,
change times, poison state, feedback delays, and future release data. Authority
rows SHALL retain exact labels, independently computed labels, parameters,
regimes, poison witnesses, and all frozen delay schedules. Public input parsing,
predicate execution, and independent scoring SHALL agree for every event.

All deployable arms SHALL receive the same public events and the same requested,
released warmup evidence in each capacity-delay cell. Their initialized learner
state and empty pending queue SHALL match at the warmup boundary. The static arm
SHALL freeze there. The online admission arms MAY request at most one label per
four arriving events. They SHALL keep no event staging data after each four-event
boundary. Their staging, learner, and pending state SHALL remain within 64 KiB.

The producer SHALL cover pending capacities 1, 4, and 16 with constant feedback
delays 0, 4, and 16 plus one fixed-seed burst schedule. A request SHALL start its
delay only after all four predictions in its block commit. Random admission
SHALL sample uniformly from the block. Disagreement admission SHALL select the
largest fraction of disagreeing public hypotheses. Both SHALL use the same
seeded tie rank. A full queue SHALL discard the new request. It SHALL not evict
an existing pending record. Releases at a boundary SHALL free capacity only for
the next boundary. A lost or discarded label SHALL remain evaluator-visible but
SHALL never reach that arm.

Public-event iteration and feedback-release iteration SHALL use separate APIs.
The admission scheduler SHALL not receive a future delay or label. The online
validation window SHALL expose only labels that the arm requested and the
scheduler released. The exact full-information oracle SHALL remain an explicitly
unattainable upper bound. Headroom reports SHALL preserve Exp7184's zero-error
static slice as no-headroom evidence and SHALL not rename that ceiling.

Before computation, Exp7198 SHALL print and flush a phase boundary. It SHALL
check the driving requirement, all required source bytes and hashes, the exact
Exp7184 completion and null fields, upstream artifact quarantine metadata, the
exclusion manifest, local tools, and writable stream, checkpoint, and result
directories. It SHALL reject quarantined upstream evidence before use. It SHALL
preserve `memory_value_score == 0` as a prior null and SHALL not use that value
as a positive gate. A missing external prerequisite SHALL create a terminal,
row-free blocked artifact with a diagnostic `gate_check_summary`.

`stream_capacity_ready_score` SHALL be the bare integer one only when stream
replay is deterministic, public access is isolated, windows are disjoint,
families are balanced, request and pending budgets hold, warmup state is matched,
source grounding agrees, and every preregistered budget or leakage mutation is
detected. This readiness score SHALL certify the stream contract only. It SHALL
not certify learnability or scientific benefit.

The artifact SHALL contain `field_principles`, `status`, `run_date`,
`preconditions_checked`, `inference_substrate`, `inference_substrate_class`,
`execution_venue`, `duration_s`, `source_artifact_hashes`, `rows`,
`sample_size_budget`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`stream_capacity_ready_score`, `stream_manifest`, `public_stream_path`,
`authority_sidecar_path`, `information_budget_rows`, `headroom_rows`,
`MODEL_SPECS`, and `model_invoked`. It SHALL also retain immutable view hashes,
delay and capacity contracts, warmup state receipts, source-grounding summaries,
mutation-audit rows, and the prior-null gate receipt. A completed CPU run SHALL
use `inference_substrate_class=cpu_exact_solver_or_simulator`,
`MODEL_SPECS=[]`, `model_invoked=false`, and `verifier_is_oracle=true`.

### SCENARIO-CL-7198-PRECONDITIONS: Quarantine And Known Null Fail Closed

- GIVEN missing, changed, or quarantined upstream evidence
- WHEN file parsing and the real precondition evaluator run
- THEN construction blocks before stream computation
- AND the failed check names its upstream, field, expected, and observed values
- AND Exp7184's zero value is recorded as a null rather than promoted.

### SCENARIO-CL-7198-STREAM: Fixed Balanced Windows Hide Authority Data

- GIVEN the ten frozen seeds and five disjoint windows
- WHEN all 10,240 public and authority rows are built
- THEN every window has equal family counts for each seed
- AND public rows contain no parameter, label, regime, delay, or poison fields.

### SCENARIO-CL-7198-WARMUP: Deployable Arms Start With Equal Information

- GIVEN one seed, capacity, and delay schedule
- WHEN the shared warmup ends
- THEN all arms have equal released evidence and learner-state hashes
- AND every pending warmup request has released before the static state freezes.

### SCENARIO-CL-7198-CAPACITY: Requests Consume Pending Capacity

- GIVEN a fixed four-event block and pending capacity
- WHEN random and disagreement admissions reach the boundary
- THEN each requests at most one label without reading its delay or label
- AND a full queue drops the new request without evicting an old record
- AND a release makes capacity available only at the next boundary.

### SCENARIO-CL-7198-CHRONOLOGY: Predictions Precede Feedback Release

- GIVEN one public event and any requested feedback
- WHEN chronological replay reaches its release point
- THEN the prediction commit index precedes the label release
- AND only the separate feedback iterator exposes the released label.

### SCENARIO-CL-7198-GROUNDING: Independent Exact Scoring Agrees

- GIVEN each public family statement and numeric observation
- WHEN the public parser extracts it and both exact evaluators execute
- THEN extraction matches the public row
- AND both evaluators return the authority label.

### SCENARIO-CL-7198-MUTATIONS: Readiness Rejects Contract Violations

- GIVEN otherwise valid evidence with one changed seed, leaked field,
  overlapping window, extra request, pending eviction, or memory overflow
- WHEN the cold readiness validator runs
- THEN it rejects the named mutation
- AND an unmodified replay receives `stream_capacity_ready_score=1`.

### SCENARIO-CL-7198-HEADROOM: Oracle And Static Evidence Stay Distinct

- GIVEN the old Exp7184 zero-error static future result and the new shifted stream
- WHEN headroom diagnostics are reduced
- THEN the old slice reports zero oracle headroom
- AND the new oracle is labeled unattainable and cannot act as a deployable arm.

### SCENARIO-CL-7198-TERMINAL: Completion Does Not Claim Learnability

- GIVEN all stream, timing, budget, and mutation checks pass
- WHEN the terminal result is written atomically
- THEN readiness is one and the verdict is `circular_positive`
- AND the honest verdict states that no learning benefit was measured.

## Implementation Status (REQ-CL-7198)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7198 and SCENARIO-CL-7198-* | Implemented 2026-09-11: bounded feedback stream module, executable wrapper, immutable public and authority views, and terminal artifact. | Verified: RED-first precondition, stream, warmup, capacity, chronology, grounding, mutation, headroom, terminal, command, and 100-percent new-module line coverage tests. |

## REQ-CL-7199: Bounded Exact Version-Space Acquisition

Carnot SHALL replay every Exp7198 stream seed and capacity-delay cell with no
LLM invocation. It SHALL compare `warmup_frozen`, `fifo_admission`,
`random_admission`, `priority_admission`, and an unattainable
`all_information_oracle`. The three admission arms SHALL use one controller,
one 64 KiB memory limit, one pending-capacity rule, and one label quota. Their
only treatment difference SHALL be the event selected at each four-event
boundary.

The controller SHALL keep the finite parameter hypotheses from zero through
32 that agree with released support labels. It SHALL predict by majority vote.
Ties SHALL reject, and an empty version space SHALL abstain. Every abstention
SHALL count as an error in the full prospective and recurrence denominators.
Only support labels SHALL eliminate hypotheses. A public hash SHALL freeze each
event's support or validation role before label reveal. Each four-event block
SHALL contain two events of each role.

A singleton SHALL freeze only after at least three distinct support labels in
its current epoch. It SHALL commit as a persistent template only after eight
distinct validation labels requested and released after the freeze make zero
mistakes. Earlier labels SHALL not validate it. Validation labels SHALL not fit
its replacement. A validation failure SHALL reset the family epoch. If released
feedback makes the active version space empty, the controller SHALL revoke the
affected template, retain its superseded bytes and rollback hash, reset the
family candidate set and validation buffer, and apply the counterexample only
when its frozen role is support. Archived labels SHALL not validate a new epoch.
Evaluator-known drift times SHALL not cause a reset.

Every prediction SHALL seal before a request or feedback release. Each request,
including validation, SHALL consume the same quota and pending capacity. A full
queue SHALL drop the new request without eviction. A release SHALL free capacity
only at the next boundary. FIFO SHALL select the first event in a block. Random
SHALL use a frozen uniform seed. Priority SHALL use current hypothesis
disagreement and the same frozen tie rule. No scheduler SHALL read a label,
future delay, hidden parameter, regime, or change time.

The complete panel SHALL cover all ten stream seeds and all twelve Exp7198
capacity-delay cells. It SHALL retain decision, update, pending-queue,
validation-access, and validation-partition rows. It SHALL report prospective
and recurrence error, false acceptance, abstention, lookup latency, and update
latency. Intervals SHALL use paired bootstrap resampling by stream seed. It
SHALL not resample events or capacity cells as independent units.

The frozen primary cell SHALL be capacity four with the burst schedule.
`acquisition_value_score` SHALL be one only when priority has a strictly
negative paired 95 percent error interval against frozen and FIFO;
its false-accept increase against both frozen controls is at most zero; its
recurrence error increase is at most 0.02; and no memory or capacity violation
occurs. Priority-specific benefit SHALL separately require a strictly negative
error interval against random admission. Otherwise the related score SHALL be
zero. `acquisition_run_complete_score` SHALL be
one when every planned cell has terminal rows, even when the value result is
null. Version-space benefit and template-commit benefit SHALL remain separate.
An identical committed singleton SHALL not receive extra accuracy credit.

Before computation, Exp7199 SHALL print and flush a phase boundary. It SHALL
check source bytes and hashes, the driving requirement, exact Exp7198 gate
fields, declared quarantine flags, the exclusion manifest, local tools, and
writable checkpoint and result directories. It SHALL reject quarantined
upstream evidence before it reads stream rows. It SHALL not promote a known
failed upstream value. A missing external prerequisite SHALL produce a
terminal row-free blocked artifact. Its `gate_check_summary` SHALL name the
failed check, upstream, field, expected value, and observed value.

The artifact SHALL contain `field_principles`, `status`, `run_date`,
`preconditions_checked`, `inference_substrate`, `inference_substrate_class`,
`execution_venue`, `duration_s`, `source_artifact_hashes`, `rows`,
`sample_size_budget`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`acquisition_run_complete_score`, `acquisition_value_score`,
`continuous_self_learning_task`, `decision_rows`, `update_rows`,
`pending_queue_rows`, `acceptance_gate_learning`, `future_hardware_path`,
`no_model_weight_mutation`, `prediction_contract`, `validation_access_rows`,
`validation_partition_rows`, `MODEL_SPECS`, and `model_invoked`.
`field_principles` SHALL explain the evidence reason for every required field.
A completed run SHALL use
`inference_substrate_class=cpu_exact_solver_or_simulator`,
`MODEL_SPECS=[]`, `model_invoked=false`, and
`verifier_is_oracle=true`. The future hardware path SHALL name CPU bitset
intersections and bounded counters, then FPGA lookup and bitset logic. It SHALL
target lookup below 1 ms without claiming an unmeasured speedup.

### SCENARIO-CL-7199-PRECONDITIONS: Quarantine And Exact Gates Fail Closed

- GIVEN a missing, changed, or quarantined upstream resource
- WHEN file parsing and the real gate evaluator run before stream consumption
- THEN Exp7199 writes a terminal row-free blocked artifact when possible
- AND the failed gate records its upstream, field, expected, and observed values.

### SCENARIO-CL-7199-ROLE-SEPARATION: Validation Cannot Fit A Candidate

- GIVEN roles frozen before labels and one requested released validation label
- WHEN the controller updates its family
- THEN validation can test a post-freeze singleton but cannot remove hypotheses
- AND a failed validation resets the epoch without fitting its replacement.

### SCENARIO-CL-7199-COMMIT: Fresh Support And Later Validation Persist A Singleton

- GIVEN three distinct current-epoch support labels that freeze one singleton
- AND eight distinct requested and released later validation labels with zero mistakes
- WHEN the eighth validation label arrives
- THEN the controller commits the exact singleton predicate
- AND its prediction before and after persistence is identical.

### SCENARIO-CL-7199-REVOCATION: Empty Evidence Reopens The Family

- GIVEN a committed singleton and one newly released contradictory label
- WHEN the active version space becomes empty
- THEN the template is revoked with superseded bytes and a rollback hash
- AND the family starts a fresh epoch without old validation credit.

### SCENARIO-CL-7199-MATCHED-ADMISSION: Selection Is The Only Treatment Difference

- GIVEN one public four-event block and matched controller state
- WHEN FIFO, random, and priority select one request
- THEN all receive the same events, quota, capacity, memory, and release rule
- AND no selection input contains a label, future delay, or evaluator regime.

### SCENARIO-CL-7199-METRICS: Seed Bootstrap Owns The Primary Gate

- GIVEN all ten seeds and twelve capacity-delay cells
- WHEN outcome differences and 95 percent intervals are computed
- THEN bootstrap draws resample stream seeds and keep matched cells together
- AND every abstention stays an error in the full window denominator.

### SCENARIO-CL-7199-TERMINAL: Complete Null Remains Auditable

- GIVEN every planned cell has terminal rows but one value condition fails
- WHEN Exp7199 derives its verdict
- THEN `acquisition_run_complete_score` is one
- AND `acquisition_value_score` is zero
- AND the verdict is a complete null rather than partial.

## Implementation Status (REQ-CL-7199)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7199 and SCENARIO-CL-7199-* | Planned: exact version-space controller, matched admission panel, command wrapper, and terminal artifact. | Planned: RED-first precondition, role, commit, revocation, admission, metric, terminal, command, and 100-percent new-module line coverage tests. |

## REQ-CL-7200: Fresh-Process Acquisition Causality And Deletion Audit

Carnot SHALL audit Exp7199 from Exp7198's immutable public stream and authority
sidecar in a fresh Python process. The audit SHALL invoke no LLM. It SHALL
require the exact same-milestone Exp7199 gate
`acquisition_run_complete_score == 1`. It SHALL capture source and upstream
bytes before JSON decoding. It SHALL reject declared artifact quarantine flags
and matching entries in `ops/exclusion_manifest.yaml` before it consumes stream
rows. A missing external prerequisite SHALL produce a terminal row-free blocked
artifact whose `gate_check_summary` names the check, upstream, field, expected
value, and observed value.

The audit SHALL reconstruct each Exp7199 seed, capacity-delay cell, admission
arm, decision, pending-queue counter, and controller state hash from the sealed
Exp7198 views. It SHALL seal each action before reading its authority label. An
independent reducer SHALL recompute error, false acceptance, and abstention from
the reconstructed decisions. Producer summaries SHALL be parity targets only.
The producer's comparison reducer SHALL not supply an expected metric.

The audit SHALL write a real checkpoint under `results/checkpoints/`. A second
isolated process SHALL load exact controller bytes and reproduce recurrence
decisions. It SHALL report decision and state-hash parity. It SHALL not load
archived future feedback or pending labels after the checkpoint boundary.

The audit SHALL run four causal controls on the frozen primary capacity-delay
cell. One control SHALL delay every feedback release until after the last
decision. One control SHALL shuffle admitted-feedback identities inside each
matched capacity cell with a seed frozen before labels are read. One control
SHALL delete committed templates while retaining the learned version spaces.
One control SHALL reset version spaces, support records, validation buffers,
and committed templates to the common warmup checkpoint. The whole-state reset
SHALL reapply neither archived future feedback nor pending released labels.

Both deletion controls SHALL preserve role separation and candidate-freeze
timing. Template-only deletion SHALL diagnose commitment behavior only. Only a
whole-learning reset can establish acquisition dependence when a singleton
version space already controls predictions. If priority and random admission
receive identical evidence or make identical decisions, the audit SHALL report
no scheduling benefit even if both improve over frozen memory.

The audit SHALL exercise poison and recurrence windows without exposing their
future outcomes to commit decisions. A rejected poison intervention SHALL roll
back to the preceding checkpoint. The audit SHALL require byte equality, state
hash equality, and decision parity after rollback. A future-label read, an
unreleased-label read, a fake no-op update, a capacity violation, or a memory
violation SHALL fail validation.

`acquisition_audit_complete_score` SHALL be one when the reconstruction,
independent metric reduction, cold reload, causal controls, both deletion
controls, rollback, leakage checks, role checks, and capacity checks finish.
A complete audit can retain a null finding. `memory_promotion_score` SHALL be
one only when Exp7199 has `acquisition_value_score == 1` and all causal,
cold-reload, rollback, leakage, and deletion gates pass. Infrastructure
readiness alone SHALL not promote memory.

The artifact SHALL contain `field_principles`, `status`, `run_date`,
`preconditions_checked`, `inference_substrate`, `inference_substrate_class`,
`execution_venue`, `duration_s`, `source_artifact_hashes`, `rows`,
`sample_size_budget`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`acquisition_audit_complete_score`, `memory_promotion_score`,
`causal_control_rows`, `cold_reload_rows`, `rollback_rows`,
`whole_learning_reset_rows`, `MODEL_SPECS`, and `model_invoked`.
`field_principles` SHALL explain the evidence reason for every required field.
A completed audit SHALL use
`inference_substrate_class=cpu_exact_solver_or_simulator`,
`MODEL_SPECS=[]`, `model_invoked=false`, and `verifier_is_oracle=true`.

### SCENARIO-CL-7200-PRECONDITIONS: Quarantine And Exact Gates Fail Closed

- GIVEN missing, changed, quarantined, or known-null upstream evidence
- WHEN byte capture and the real gate evaluator run before stream consumption
- THEN the audit blocks external absence with a complete gate diagnosis
- AND it never promotes the known failed acquisition value.

### SCENARIO-CL-7200-RECONSTRUCTION: Actions Precede Independent Scoring

- GIVEN the immutable Exp7198 stream and the complete Exp7199 producer artifact
- WHEN a fresh process reconstructs the acquisition panel
- THEN predictions, budget counters, and state hashes match per unit
- AND the authority label is read only after each reconstructed action seals
- AND an independent reducer reproduces the producer metrics.

### SCENARIO-CL-7200-CONTROLS: Delayed And Shuffled Feedback Test Dependence

- GIVEN frozen primary-cell events and admission choices
- WHEN feedback is fully delayed or admitted identities are shuffled
- THEN each control records changed evidence and changed decisions separately
- AND identical priority and random evidence or decisions earns no scheduler credit.

### SCENARIO-CL-7200-DELETION: Template And Whole-State Resets Stay Distinct

- GIVEN a cold checkpoint with acquired version-space state
- WHEN committed templates alone are deleted and all learned state is reset separately
- THEN both controls preserve role and freeze-timestamp rules
- AND only the whole-state reset can establish acquisition dependence.

### SCENARIO-CL-7200-COLD-ROLLBACK: Reload And Rejection Restore Behavior

- GIVEN recurrence and poison boundaries with no future outcomes exposed
- WHEN a new process reloads state or a rejected poison change rolls back
- THEN recurrence decisions match the saved state
- AND rollback restores exact bytes, hashes, and decisions.

### SCENARIO-CL-7200-ATTACKS: Leakage And Fake Updates Fail

- GIVEN an unreleased label read, future-label read, fake no-op update, or bound violation
- WHEN the artifact validator checks causal evidence
- THEN it rejects the named violation.

### SCENARIO-CL-7200-TERMINAL: Complete Null Cannot Promote Memory

- GIVEN every audit component completes and Exp7199's value score is zero
- WHEN completion and promotion are classified
- THEN `acquisition_audit_complete_score` is one
- AND `memory_promotion_score` is zero
- AND the verdict is a terminal complete null.

## Implementation Status (REQ-CL-7200)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7200 and SCENARIO-CL-7200-* | Planned: fresh-process reconstruction, causal controls, two deletion controls, rollback, wrapper, and terminal artifact. | Planned: RED-first gate, reconstruction, control, deletion, cold reload, rollback, attack, terminal, command, and 100-percent new-module line coverage tests. |

## REQ-CL-7212: Query-Driven Refinement Fixture And Commit-Only Runtime

Carnot SHALL freeze 20 independent streams with seeds 7212001 through
7212020. Each stream SHALL contain 1,024 public events. Four consecutive
256-event phases SHALL represent stable, drift, recurrence, and poisoned
feedback conditions. The stream SHALL use the four Exp7198 finite numeric
predicate families over values 0 through 32. An evaluator-only process SHALL
generate hidden parameters, exact labels, audit labels, poison state, and
feedback release times. Learner-visible bytes SHALL contain none of those
fields before their charged release.

Each arm SHALL receive the same first 32 released warmup observations. Carnot
SHALL freeze a deployable fallback after warmup. Each arm MAY request at most
64 later labels with pending capacity four and the fixed burst-delay rule.
Exactly 48 query slots SHALL be fitting slots. Exactly 16 query slots SHALL be
promotion-validation slots. Validation values SHALL be reserved before fitting.
They SHALL never eliminate a hypothesis. A witness selector SHALL choose the
lowest value among deterministic maximally balanced disagreement splits.

The acquisition controller SHALL keep its candidate version space private.
The deployed query path SHALL use only the frozen warmup fallback or an exact
predicate that is currently committed in `TransactionalConstraintMemory`.
It SHALL NOT vote over uncommitted hypotheses. A singleton candidate SHALL
need four disjoint, charged, released validation observations for its family.
Its commit SHALL require the unchanged memory parent hash. A contradictory
released validation observation SHALL revoke the committed version and restore
fallback behavior. An uncertified candidate SHALL remain an explicit outcome.

The evaluator SHALL freeze a hidden audit panel for all 33 values, four
families, four phases, and 20 streams. Audit labels SHALL be scoring-only.
They SHALL NOT affect fitting, validation, commit, or rollback. A commit based
on partial finite-domain evidence SHALL be described as empirically validated.
It SHALL NOT be described as a universal correctness certificate.

The fixture SHALL write immutable public stream, evaluator sidecar,
split-feedback manifest, released-warmup, and controller serialization bytes
under `results/streams/experiment_7212/`. The terminal artifact SHALL bind
those bytes and all source inputs by SHA-256. It SHALL retain per-stream and
per-arm rows with `unit_id`, `arm`, `seed`, `metric`, `error`, and
`abstention`. `refinement_fixture_ready_score` SHALL equal one only when stream
separation, query accounting, disjoint validation, commit-only deployment,
transaction insertion and deletion, and poison and stale-parent rejection all
pass. This score SHALL certify fixture readiness only. It SHALL not claim
learning value.

The artifact SHALL set `MODEL_SPECS=[]`, `model_invoked=false`,
`inference_substrate_class=cpu_exact_solver_or_simulator`, and
`execution_venue=host`. It SHALL record the actual hostname separately in
`execution_host`. A completed ready fixture SHALL use
`verdict_class=circular_positive` because exact-domain labels and audit labels
share the same correctness authority. Missing or quarantined upstream evidence
SHALL produce a row-free blocked artifact. Its `gate_check_summary` SHALL name
the failed check, upstream, field, expected value, and observed value. The
known null values from Exp7199 and Exp7200 SHALL remain unpromoted.

### SCENARIO-CL-7212-PRECONDITIONS: Exact Gates And Quarantine Fail Closed

- GIVEN the V635 task identity, cited source bytes, required imports, tools,
  output paths, Exp7199 null, and Exp7200 null
- WHEN Exp7212 checks raw fields and the independent exclusion manifest
- THEN only actual principle/value wrappers are unwrapped
- AND a missing, changed, or quarantined prerequisite emits a row-free block.

### SCENARIO-CL-7212-STREAM: Evaluator Authority Stays Outside Public Bytes

- GIVEN 20 seeds and four exact predicate families
- WHEN the evaluator-only worker seals all stream views
- THEN each public stream has 1,024 balanced chronological events
- AND hidden parameters, labels, phases, poison state, and release times stay in
  the authority sidecar.

### SCENARIO-CL-7212-BUDGET: Warmup And Query Partitions Are Charged

- GIVEN the shared 32-observation warmup and 64-query later budget
- WHEN an arm acquires fitting and validation labels
- THEN fitting uses at most 48 queries and validation uses at most 16 queries
- AND capacity four and the same burst-delay protocol apply to every arm.

### SCENARIO-CL-7212-WITNESS: Balanced Disagreement Is Deterministic

- GIVEN more than one surviving parameter
- WHEN the query-driven selector chooses a non-reserved value
- THEN it maximizes the smaller prediction partition
- AND it resolves equal scores by the lowest numeric value.

### SCENARIO-CL-7212-COMMIT: Only Durable Predicates Change Deployment

- GIVEN a singleton with four disjoint released validation observations
- WHEN the expected parent hash still matches memory
- THEN the exact predicate commits through `TransactionalConstraintMemory`
- AND the deployed query changes only after that commit becomes visible.

### SCENARIO-CL-7212-ROLLBACK: Released Contradiction Restores Fallback

- GIVEN a committed predicate and a later contradictory released validation
- WHEN the controller applies the recorded inverse transaction
- THEN memory returns to the byte-identical parent
- AND the deployed query again uses the frozen fallback.

### SCENARIO-CL-7212-ATTACKS: Stale And Poisoned Transactions Reject

- GIVEN a stale expected parent hash or a false exact-label event
- WHEN either proposal reaches the commit boundary
- THEN it does not become an active predicate
- AND memory keeps its prior state hash.

### SCENARIO-CL-7212-AUDIT: Hidden Full-Domain Labels Are Scoring-Only

- GIVEN all 33 values for every family and phase
- WHEN the evaluator builds the audit panel
- THEN the learner receives no audit label
- AND no audit row appears in fitting, validation, or rollback evidence.

### SCENARIO-CL-7212-TERMINAL: Readiness Does Not Promote Prior Null Value

- GIVEN a leak-free fixture with a working commit-only seam
- WHEN the terminal result derives its readiness and verdict
- THEN readiness can equal one while learning value stays unmeasured
- AND the Exp7199 and Exp7200 zero-value gates remain explicit.

## Implementation Status (REQ-CL-7212)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7212 and SCENARIO-CL-7212-* | Implemented 2026-09-11: evaluator-only stream worker, private refinement controller, commit-only runtime, wrapper, and terminal artifact. | Verified: RED-first precondition, stream, budget, witness, commit, rollback, attack, audit, terminal, command, and 100-percent new-module line coverage tests. |

## REQ-CL-7213: Witnessed Predicate Refinement Learning

Carnot SHALL run five matched arms on all 20 immutable Exp7212 streams. The
arms SHALL be a frozen warmup fallback, passive-query committed predicates,
random-query committed predicates, witness-query committed predicates, and
the shipped online version-space-majority predictor. The version-space arm
SHALL use the same queries and released labels as the witness committed arm.

Every event prediction SHALL finish before its query decision, authority
score, or feedback release. Each learning arm SHALL use no more than 64 later
queries. At most 48 fitting queries and 16 reserved validation queries SHALL
be charged. Pending occupancy SHALL not exceed four. Validation labels SHALL
not remove hypotheses. Hidden audit labels and hidden parameters SHALL not
affect a live decision.

The experiment SHALL retain a decision row for each of 1,024 events in each
stream and arm. It SHALL also retain query, commit, revoke, support,
validation, version, cost, poison, drift, and recurrence evidence. The
top-level `rows` SHALL contain paired stream aggregates with `unit_id`, `arm`,
`seed`, `metric`, `error`, and `abstention`.

The experiment SHALL delete committed templates on a read-only state copy
while it preserves acquisition state. It SHALL also reset the full learned
state as a separate control. It SHALL record predictions that change after
actual template deletion. A checkpoint under `results/checkpoints/` SHALL
contain reloadable controller and transactional-memory state.

The primary gate SHALL use 10,000 paired stream-bootstrap draws with seed
7213001. The upper 95-percent future-error interval versus both the frozen
warmup and random committed arms SHALL be below zero. The upper false-accept
increase versus frozen warmup SHALL be at most zero. The recurrence-error
increase SHALL be at most 0.02. Capacity and validation-access violations
SHALL be zero. Template deletion SHALL cause a strictly positive prospective
error increase.

Carnot SHALL report passive acquisition and version-space comparisons even
when the primary gate fails. It SHALL not claim version-space superiority
unless measured. The secondary compiled-deployment gate SHALL require a lower
accuracy interval of at least -0.02 and measured amortized throughput of at
least two times the version-space arm. Total cost SHALL include selection,
fitting, validation, commits, and deployment. This secondary result SHALL not
change the primary value result or satisfy NFR-01.

The artifact SHALL set `continuous_self_learning_task=true`,
`no_model_weight_mutation=true`, `MODEL_SPECS=[]`, `model_invoked=false`,
`inference_substrate_class=cpu_exact_solver_or_simulator`, and
`execution_venue=host`. It SHALL record the hostname in `execution_host`.
Exact-domain success SHALL use `verdict_class=circular_positive` and
`verifier_is_oracle=true`. `refinement_run_complete_score` SHALL be one for a
complete measurement. `refinement_value_score` SHALL be one only when the
primary gate passes.

The artifact SHALL contain `field_principles`, `status`, `run_date`,
`preconditions_checked`, `inference_substrate`, `inference_substrate_class`,
`execution_venue`, `execution_host`, `duration_s`, `source_artifact_hashes`,
`rows`, `sample_size_budget`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`refinement_run_complete_score`, `refinement_value_score`,
`continuous_self_learning_task`, `no_model_weight_mutation`,
`acceptance_gate_learning`, `decision_rows`, `query_rows`,
`commit_deletion_rows`, `checkpoint_path`, `latency_summary`,
`future_hardware_path`, `MODEL_SPECS`, and `model_invoked`.
`field_principles` SHALL state the reason for every required field. A missing,
changed, unauthenticated, or quarantined Exp7212 prerequisite SHALL produce a
row-free terminal blocked artifact. The gate summary SHALL name the failed
check, upstream, field, expected value, and observed value.

### SCENARIO-CL-7213-PRECONDITIONS: Authenticated Fixture Or Terminal Block

- GIVEN the V635 task identity and the exact Exp7212 artifact and stream bytes
- WHEN the real gate evaluator checks readiness, hashes, imports, tools, paths,
  principle wrappers, and the independent exclusion manifest
- THEN a clean ready fixture can run
- AND an external failure writes a diagnostic row-free blocked result.

### SCENARIO-CL-7213-CHRONOLOGY: Decisions Precede All Authority Access

- GIVEN one frozen public chronology and its separate authority sidecar
- WHEN each arm handles an event
- THEN prediction and query selection precede scoring and feedback release
- AND no future label, audit label, or hidden parameter enters a live input.

### SCENARIO-CL-7213-ACQUISITION: Queries And Validation Are Fully Charged

- GIVEN passive, random, and deterministic witness selectors
- WHEN they request later feedback
- THEN fitting, validation, delay, and capacity costs use the fixed 48, 16, and
  four-slot limits
- AND the version-space arm receives the exact witness query schedule.

### SCENARIO-CL-7213-COMMIT: Durable Predicates Alone Change Deployment

- GIVEN released fitting labels and four disjoint reserved validation labels
- WHEN a singleton commits or later receives a contradiction
- THEN deployment changes through transactional predicate insertion or rollback
- AND uncommitted hypotheses never affect a committed-arm prediction.

### SCENARIO-CL-7213-DELETION: Template Removal Is A Causal Intervention

- GIVEN a learned checkpoint and the same next public event
- WHEN a read-only shadow removes committed templates but preserves fitting state
- THEN changed predictions and prospective errors are recorded
- AND a separate full reset does not substitute for template deletion.

### SCENARIO-CL-7213-METRICS: Stream Bootstrap Owns The Learning Gate

- GIVEN all 20 independent streams and 102,400 event decisions
- WHEN future error, false acceptance, abstention, recurrence, poison, drift,
  query, memory, and latency metrics are reduced
- THEN 10,000 paired resamples use stream seeds as the independent units
- AND every fixed gate condition retains its estimate, interval, and outcome.

### SCENARIO-CL-7213-DEPLOYMENT: Strong Inference Remains A Fair Baseline

- GIVEN compiled predicate dispatch and version-space majority inference on the
  same witness information
- WHEN accuracy and total amortized throughput are compared
- THEN noninferiority and two-times throughput are reported separately
- AND a failed primary learning gate cannot be rescued by deployment speed.

### SCENARIO-CL-7213-TERMINAL: Complete Null Remains Auditable

- GIVEN every stream, arm, intervention, checkpoint, and metric completes
- WHEN one or more primary value conditions fail
- THEN `refinement_run_complete_score` equals one
- AND `refinement_value_score` equals zero
- AND the verdict is a complete null, not partial.

## Implementation Status (REQ-CL-7213)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7213 and SCENARIO-CL-7213-* | Implemented 2026-09-11: matched refinement panel, causal deletion replay, checkpoint, wrapper, and terminal artifact. | Verified: RED-first gate, chronology, acquisition, commit, deletion, metric, deployment, terminal, command, and 100-percent new-module line coverage tests. |

## REQ-CL-7214: Independent Refinement Cold And Causal Audit

Carnot SHALL audit the complete Exp7213 refinement result from its sealed event
rows and saved state. The audit SHALL authenticate the producer artifact, the
producer checkpoint, and every declared fixture input before it reads outcome
rows. It SHALL reject an upstream quarantine independently of the structured
producer gates. It SHALL unwrap only an exact `principle` and `value` wrapper.

The audit SHALL reconstruct each producer stream aggregate from event rows. It
SHALL recompute prospective error, false acceptance, abstention, drift,
recurrence, poison, query totals, reserved validation separation, pending
occupancy, and total CPU cost. It SHALL reconstruct the fixed 10,000-draw
stream-cluster intervals without calling the producer metric builder.

For each of 20 streams, a fresh subprocess SHALL load the saved witness-arm
state. It SHALL replay 32 frozen public events without authority labels. It
SHALL compare decisions before and after reload. The audit SHALL separately
delete committed templates, reset all learned state, and remove the last
changed committed predicate. It SHALL record decision changes for each
intervention. A hash change alone SHALL not establish causality.

The audit SHALL replay matched shuffled-feedback and no-feedback controls. The
controls SHALL retain the producer query budget and event dates. A stale parent
transaction and a poisoned validation transaction SHALL cause no promotion.
Each rejected transaction SHALL preserve the prior state bytes, state hash, and
public decisions. No live decision or control SHALL read a hidden audit label.
Drift and recurrence results SHALL remain separate. Unequal information SHALL
disqualify a claimed gain.

`refinement_audit_complete_score` SHALL equal one only when all owned audit
checks complete. `memory_promotion_score` SHALL equal one only when the producer
has `refinement_value_score=1` and all cold, causal, chronology, rollback, and
information-parity checks pass. A producer value of zero SHALL produce a
complete null. It SHALL not produce a partial result or a promotion.

The terminal artifact SHALL use the fixed run date `20260911`. It SHALL set
`MODEL_SPECS=[]`, `model_invoked=false`,
`inference_substrate_class=cpu_exact_solver_or_simulator`,
`execution_venue=host`, and `verifier_is_oracle=true`. It SHALL put the actual
hostname in `execution_host`. Checkpoints SHALL stay under
`results/checkpoints/`. Missing, changed, unauthenticated, or quarantined input
SHALL produce a row-free terminal blocked artifact. Its gate summary SHALL name
the failed check, upstream, field, expected value, and observed value.

### SCENARIO-CL-7214-PRECONDITIONS: Exact Producer And Fixture Bytes Fail Closed

- GIVEN the Exp7214 task identity and all producer-declared evidence paths
- WHEN the gate evaluator checks hashes, imports, tools, paths, gates, and the
  independent exclusion manifest
- THEN only a complete authenticated producer can enter the audit
- AND an external failure produces a diagnostic row-free blocked artifact.

### SCENARIO-CL-7214-RECOMPUTATION: Event Rows Own Every Headline Metric

- GIVEN all producer decision and query rows for 20 streams
- WHEN an independent reducer reconstructs outcomes, budgets, and CPU costs
- THEN all 20 witness stream outcomes match their producer aggregates
- AND the fixed stream-cluster intervals match the producer comparisons.

### SCENARIO-CL-7214-COLD: Saved State Replays In Fresh Processes

- GIVEN one saved witness-arm boundary for each stream
- WHEN a fresh subprocess loads full state and replays 32 public events
- THEN its public decisions match the parent replay
- AND neither replay receives authority labels or hidden audit fields.

### SCENARIO-CL-7214-CAUSAL: Distinct Interventions Record Decision Effects

- GIVEN a saved learned state and the same public replay rows
- WHEN templates, all learned state, and the last changed predicate are removed
- THEN each intervention records its changed decision locations
- AND fitting state remains present in the template-only shadow.

### SCENARIO-CL-7214-CONTROLS: Feedback Information Stays Matched

- GIVEN the producer query schedule and frozen feedback dates
- WHEN shuffled-feedback and no-feedback controls replay prospectively
- THEN budgets, dates, and public inputs remain matched
- AND drift and recurrence outcomes stay separate.

### SCENARIO-CL-7214-ROLLBACK: Stale And Poisoned Transactions Cannot Promote

- GIVEN a stale parent or poisoned validation response
- WHEN either reaches the transaction boundary
- THEN no illegitimate predicate becomes active
- AND prior bytes, hashes, and public decisions remain identical.

### SCENARIO-CL-7214-TERMINAL: A Complete Producer Null Stays Null

- GIVEN every independent audit component passes and producer value equals zero
- WHEN completion and promotion scores are derived
- THEN `refinement_audit_complete_score` equals one
- AND `memory_promotion_score` equals zero
- AND the verdict is a complete null, not partial.

## Implementation Status (REQ-CL-7214)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7214 and SCENARIO-CL-7214-* | Planned: independent row reduction, fresh-process replay, causal controls, rollback probes, wrapper, and terminal artifact. | Planned: RED-first precondition, recomputation, cold replay, intervention, control, rollback, terminal, command, and new-module coverage tests. |

## REQ-CL-7226: Lossless Packed Belief Compiler

Carnot SHALL compile all surviving hypotheses for each supported finite
predicate family into a packed bit mask. It SHALL cache the accept-vote count
for every input in the finite domain. Prediction SHALL match the shipped
version-space controller. A majority tie SHALL reject. An empty state SHALL
abstain. The query choice SHALL use the shipped disagreement score and frozen
tie rank.

Only released feedback SHALL change the state. Prediction SHALL be read-only.
The compiler SHALL apply one ordered batch after a query cycle. It SHALL keep
validation feedback from eliminating hypotheses. A support contradiction SHALL
use the shipped empty-state reset and then apply the current label. No state
SHALL store a hidden parameter, target label, future label, or drift time.

Each durable state SHALL contain the survivor masks, cached vote counts,
release provenance, version, and parent hash. A commit SHALL reject a future
release, stale parent, malformed payload, or corrupt vote cache. Rollback SHALL
restore the exact parent bytes. The energy `E(label,x)` SHALL equal the fraction
of survivors that disagree with the supplied label. It SHALL return an explicit
unknown result for an unsupported label, unsupported input, or empty state.

The experiment SHALL create 20 fresh streams with seeds 7226001 through
7226020. Each stream SHALL contain 1,024 chronological events from the existing
four relation families and the existing stable, drift, recurrence, and poison
windows. The first 128 events SHALL be warmup. All later events SHALL stay in
chronological order. A separate evaluator process SHALL own hidden parameters
and labels. The public stream and manifest SHALL contain no authority fields.
The fixed contract SHALL use pending capacity four, burst delays from zero,
four, 16, and 32 events, and a total query ceiling of 128 per stream and arm.

Before readiness, Carnot SHALL compare reference and packed prediction, energy,
query, update, reset, serialization, rollback, and delayed-release behavior.
The finite-domain check SHALL cover every input for all small tested hypothesis
subsets. Fixed random replays SHALL cover longer removal and reset sequences.
Negative controls SHALL mutate a future label, release order, parent hash, and
cached votes. A changed future label SHALL have no effect before release.

`belief_compiler_ready_score` SHALL equal one only when parity, serialization,
split, grounding, and information-boundary checks all pass. This score SHALL
certify compiler readiness only. It SHALL not claim learning benefit, runtime
acceleration, or reproduction of the ECAI compact-template learner. The
terminal result SHALL use `verdict_class=circular_positive` and
`verifier_is_oracle=true` when exact finite checks pass.

The task SHALL use `MODEL_SPECS=[]`, `model_invoked=false`,
`inference_substrate=cpu_exact_solver_or_simulator`,
`inference_substrate_class=cpu_exact_solver_or_simulator`, and
`execution_venue=host`. It SHALL use `blocked_no_run` for both substrate fields
when an unchanged external gate prevents the run. It SHALL record the actual
hostname, monotonic duration, UTC start and end timestamps, source hashes, full
sample counts, and fixed seeds.

The terminal artifact SHALL contain every field required by the Exp7226 task.
It SHALL include per-seed and per-arm rows with `unit_id`, `arm`, `seed`,
`metric`, `error`, and `abstention`. It SHALL bind public, authority, release,
manifest, and compiler-state paths to hashes. A missing, changed,
unauthenticated, or quarantined prerequisite SHALL create a row-free blocked
artifact. Its gate summary SHALL name the failed check, upstream, field,
expected value, and observed value.

### SCENARIO-CL-7226-PRECONDITIONS: Exact Diagnosis Fails Closed

- GIVEN the task identity and exact Exp7213 diagnostic artifact
- WHEN code, imports, hashes, fields, quarantine, and output paths are checked
- THEN only authenticated inputs can start generation
- AND an external failure creates a diagnostic blocked artifact.

### SCENARIO-CL-7226-PARITY: Packed Decisions Match The Reference

- GIVEN each supported family and tested small survivor subset
- WHEN every finite input and frozen query block is evaluated
- THEN predictions, ties, disagreement, energy, and query choices match
- AND empty states abstain without a false correctness value.

### SCENARIO-CL-7226-UPDATE: Released Feedback Is The Only Update Input

- GIVEN an ordered batch whose release times have arrived
- WHEN the transaction applies support and validation rows
- THEN only contradicted support hypotheses are removed
- AND the shipped empty-state reset behavior is preserved.

### SCENARIO-CL-7226-TRANSACTION: State Commits And Rolls Back Exactly

- GIVEN a valid current state and matching parent hash
- WHEN a batch commits between query cycles and later rolls back
- THEN the commit serializes masks, vote counts, provenance, and parent hash
- AND rollback restores byte-identical parent state.

### SCENARIO-CL-7226-MUTATION: Invalid Boundaries Fail At Admission

- GIVEN future feedback, reordered feedback, a stale parent, or corrupt votes
- WHEN each mutation reaches the compiler boundary
- THEN future, stale, and corrupt commits are rejected without state change
- AND reordered released feedback changes its ordered provenance receipt.

### SCENARIO-CL-7226-STREAM: Fresh Splits Precede Evaluation

- GIVEN seeds 7226001 through 7226020
- WHEN the evaluator process freezes 1,024 events per seed
- THEN each stream has 128 warmup events and chronological remaining events
- AND all stream, delay, capacity, and query limits match the public manifest.

### SCENARIO-CL-7226-BOUNDARY: Public Bytes Contain No Authority

- GIVEN separate public, release, and authority views
- WHEN the extractor, compiler, executor, and independent label check run
- THEN public inputs contain no hidden parameter, label, regime, or future data
- AND future authority mutation cannot change a prediction before release.

### SCENARIO-CL-7226-TERMINAL: Readiness Is Not Learning Value

- GIVEN zero parity, serialization, split, and boundary failures
- WHEN the terminal artifact is validated and written atomically
- THEN `belief_compiler_ready_score` equals one
- AND the circular-positive verdict claims no efficacy or speed result.

## Implementation Status (REQ-CL-7226)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7226 and SCENARIO-CL-7226-* | Planned: packed finite belief state, transactional batch boundary, fresh isolated stream fixture, thin wrapper, and terminal artifact. | Planned: RED-first parity, update, transaction, mutation, stream, boundary, terminal, command, and 100-percent new-module line coverage tests. |

## REQ-CL-7227: Prospective Lossless Belief Learning

Carnot SHALL replay the clean Exp7226 twenty-stream manifest once. It SHALL
compare `frozen_warmup`, `reference_online_version_space`,
`packed_online_memory`, `original_committed_predicate`, and
`packed_feedback_withheld`. The reference and packed online arms SHALL use the
same public query decisions, released labels, release order, pending-capacity
limit of four, and query ceiling of 128. Their prediction, query, and energy
values SHALL match for every chronological event.

Every event row SHALL record the public prediction, energy, query decision,
state hash, survivor count, error, false accept, abstention, recurrence segment,
feedback delay, and pending-capacity use before feedback release. The evaluator
SHALL then release only eligible feedback. Each adaptive controller SHALL apply
the ordered release batch atomically after the decision. A future label SHALL
not affect an earlier prediction or state. No arm SHALL mutate model weights.

The primary scientific gate SHALL compare `packed_online_memory` with
`frozen_warmup`. The upper 95-percent stream-bootstrap bound for future error
delta SHALL be less than zero. The upper bound for false-accept delta SHALL be
at most zero. The recurrence error increase SHALL be at most 0.02. The packed
versus reference implementation gate SHALL require zero prediction, query, and
energy mismatches. Parity SHALL not count as statistical superiority over the
reference arm.

Carnot SHALL withhold packed feedback in predeclared future segments. It SHALL
report decision and error differences against the full packed replay. The
causality gate SHALL require at least one later decision change from released
feedback and zero effects before release. The same 20 frozen stream seeds SHALL
be the independent units. No extra seed or relaxed threshold may rescue a null.

Carnot SHALL measure selection, update, recomputation, commit, serialization,
and lookup with monotonic timers. It SHALL report allocated bytes, p50, p95,
and amortized per-event total. Runtime targets SHALL stay separate from the
scientific gate. `belief_run_complete_score` SHALL equal one only after every
scheduled event and arm row exists. `belief_learning_value_score` SHALL equal
one only when the fixed efficacy, retention, and causality criteria pass.

The task SHALL use `MODEL_SPECS=[]`, `model_invoked=false`,
`inference_substrate=cpu_exact_solver_or_simulator`,
`inference_substrate_class=cpu_exact_solver_or_simulator`, and
`execution_venue=host`. It SHALL use `blocked_no_run` for both substrate fields
when an unchanged external gate prevents replay. The artifact SHALL use the
closed verdict enum and SHALL use a complete null when replay completes but a
learning gate fails.

The terminal artifact SHALL contain all task-required fields. It SHALL bind the
Exp7226 artifact and each sealed stream file to current hashes. It SHALL retain
full per-event rows at `decision_rows_path`, per-seed arm rows, comparison rows,
deletion rows, gate outcomes, latency summaries, sample counts, and exact source
hashes. A missing, changed, quarantined, or unauthenticated prerequisite SHALL
create a row-free blocked artifact whose gate summary names the failed check,
upstream, field, expected value, and observed value.

### SCENARIO-CL-7227-PRECONDITIONS: Invalid Evidence Blocks Replay

- GIVEN the Exp7226 readiness contract or a sealed stream hash is invalid
- WHEN the learner checks identity, quarantine, imports, and output paths
- THEN it writes a diagnostic blocked artifact without running the replay
- AND the failed gate includes exact expected and observed values.

### SCENARIO-CL-7227-CHRONOLOGY: Decisions Precede Feedback

- GIVEN one chronological public event and evaluator-owned authority
- WHEN each arm makes its decision
- THEN prediction, energy, query, and state hash are frozen before release
- AND only feedback eligible at that boundary enters the atomic update.

### SCENARIO-CL-7227-MATCHED: Adaptive Information Is Identical

- GIVEN the reference and packed online arms for one stream
- WHEN a query block and release boundary complete
- THEN query identities, released labels, pending use, and ceilings match
- AND prediction, query, and energy mismatch counts remain zero.

### SCENARIO-CL-7227-CONTROLS: Frozen And Committed Controls Remain Diagnostic

- GIVEN the same public events and evaluator labels
- WHEN frozen warmup and original committed-predicate controls replay
- THEN both keep their declared state and receive full-denominator metrics
- AND the committed control is not the sole comparator for learning value.

### SCENARIO-CL-7227-CAUSAL: Withheld Feedback Changes Only Later Decisions

- GIVEN predeclared future feedback-withholding segments
- WHEN full packed and withheld packed replays are compared
- THEN all differences occur after an eligible withheld release
- AND the artifact reports every changed prediction and error.

### SCENARIO-CL-7227-GATES: Efficacy And Parity Stay Separate

- GIVEN a complete matched replay
- WHEN stream-bootstrap efficacy and exact parity are scored
- THEN learning value uses frozen-warmup error, false accepts, recurrence, and causality
- AND exact reference parity does not count as superiority.

### SCENARIO-CL-7227-COST: Every Operation Has Measured Cost

- GIVEN all scheduled event and release operations
- WHEN the host replay completes
- THEN selection, update, recomputation, commit, serialization, and lookup report p50 and p95
- AND allocated bytes and amortized per-event total retain measured values.

### SCENARIO-CL-7227-TERMINAL: Completion Does Not Imply Value

- GIVEN all twenty streams and five arms complete
- WHEN one prospective learning criterion fails
- THEN `belief_run_complete_score` equals one
- AND `belief_learning_value_score` equals zero with a complete null verdict.

## Implementation Status (REQ-CL-7227)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7227 and SCENARIO-CL-7227-* | Planned: matched chronological replay over the sealed Exp7226 streams, thin wrapper, decision-row checkpoint, and terminal artifact. | Planned: RED-first chronology, parity, causality, gate, schema, command, and 100-percent new-module coverage tests. |

## REQ-CL-7228: Independent Packed-Belief Cold And Causal Audit

Carnot SHALL audit Exp7227 when `belief_run_complete_score` equals one. It
SHALL NOT require a positive learning result. Before reading outcome rows, the
audit SHALL authenticate the producer artifact, decision rows, packed-state
checkpoint, and every immutable stream receipt. It SHALL reject missing
denominators, changed hashes, malformed principle wrappers, and quarantine
evidence. Only a dictionary with exactly `principle` and `value` keys SHALL be
unwrapped.

The audit SHALL independently reduce all 102,400 producer decision rows into
100 stream-arm rows. Each row SHALL retain its complete event denominator,
error, false-accept, abstention, recurrence, query, release, and pending counts.
The audit SHALL rebuild the three fixed stream-bootstrap comparisons and the 20
scheduled feedback-deletion rows. It SHALL NOT call the producer reducers.

For each of the 20 streams, a fresh isolated process SHALL load serialized
packed memory and a separately restored reference controller. It SHALL compare
state identity, prediction, disagreement energy, cached votes, and the next
released update. A fixed delayed correction SHALL reject before its release
boundary and apply at that boundary. The audit SHALL also replay a stale-parent
commit, corrupt survivor mask, duplicate release, and exact rollback. Each
rejection SHALL preserve the expected bytes, state hash, and later decisions.
Rollback SHALL restore the exact parent bytes and later decisions.

The audit SHALL shift hidden future labels while it keeps released history and
public input fixed. Predictions before release SHALL remain unchanged. A
process access policy SHALL deny the future authority sidecar. The artifact
SHALL report the policy, process identity, denied open observation, source
hashes, and public-only input hash. An asserted boolean alone SHALL not satisfy
this boundary.

The audit SHALL remove learned packed state on a read-only controller copy. It
SHALL record changed public decisions and prospective error effects after labels
join only for scoring. It SHALL also recompute the producer's scheduled
feedback-deletion effects from immutable rows. The audit SHALL perform no model
fit, threshold change, parameter adjustment, or model-weight update.

`belief_audit_complete_score` SHALL equal one only when all independent metric,
cold reload, parity, attack, rollback, causality, grounding, and isolation
checks complete. `belief_promotion_score` SHALL equal one only when the producer
has `belief_learning_value_score=1` and all fresh-process and causal checks
pass. A producer value of zero SHALL produce a complete null. It SHALL not
produce a partial result or promotion. The known V635 refinement and audit
nulls SHALL remain explicit historical receipts.

The terminal artifact SHALL use run date `20260912`. It SHALL set
`MODEL_SPECS=[]`, `model_invoked=false`,
`inference_substrate=cpu_exact_solver_or_simulator`,
`inference_substrate_class=cpu_exact_solver_or_simulator`, and
`execution_venue=host` after replay. A blocked run SHALL use `blocked_no_run`
for both substrate fields. The actual hostname SHALL stay in `execution_host`.
The audit SHALL record monotonic duration and actual UTC timestamps. It SHALL
leave pipeline defaults and publication unchanged.

The artifact SHALL contain `field_principles`, `status`, `run_date`,
`started_at_utc`, `completed_at_utc`, `preconditions_checked`,
`inference_substrate`, `inference_substrate_class`, `execution_venue`,
`execution_host`, `duration_s`, `source_artifact_hashes`, `rows`,
`sample_size_budget`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`MODEL_SPECS`, `model_invoked`, `belief_audit_complete_score`,
`belief_promotion_score`, `cold_reload_rows`, `rollback_rows`,
`causal_control_rows`, `metric_recomputation_rows`,
`comparison_recomputation_rows`, `deletion_recomputation_rows`,
`source_grounding_rows`, `runtime_isolation_receipt`, `checkpoint_receipt`,
`producer_gate_receipt`, `v635_history_receipt`, `audit_errors`,
`no_model_weight_mutation`, `no_fitting_or_parameter_adjustment`,
`certificate_published`, and `default_pipeline_modified`. Missing or changed
external evidence SHALL create a row-free blocked artifact. Its gate summary
SHALL name the failed check, upstream, field, expected value, and observed value.

### SCENARIO-CL-7228-PRECONDITIONS: Complete Authenticated Producer Or Block

- GIVEN the Exp7228 identity and exact Exp7227 artifact and raw receipts
- WHEN the audit checks requirements, imports, outputs, hashes, fields, and quarantine
- THEN Exp7227 completeness permits the run even when its learning value is zero
- AND any external failure creates a diagnostic row-free blocked artifact.

### SCENARIO-CL-7228-RECOMPUTATION: Immutable Event Rows Own All Metrics

- GIVEN all 102,400 per-event producer rows and full denominators
- WHEN the independent reducer rebuilds stream rows, intervals, and deletion effects
- THEN all 100 aggregates, three comparisons, and 20 deletions match
- AND stored producer outcome fields cannot hide a row-level contradiction.

### SCENARIO-CL-7228-COLD: Packed And Reference State Reload Separately

- GIVEN one serialized packed state and reference state for each stream
- WHEN a fresh process loads both and replays public probes
- THEN predictions, energies, survivor votes, and the next update match
- AND actual state hashes and process identities remain in the receipt.

### SCENARIO-CL-7228-TRANSACTIONS: Boundary Attacks Preserve Exact State

- GIVEN delayed, stale-parent, corrupt-mask, duplicate, and rollback probes
- WHEN each probe reaches the packed transaction boundary
- THEN premature or invalid changes reject without state drift
- AND rollback restores exact parent bytes, hash, and later decisions.

### SCENARIO-CL-7228-CAUSAL: Learned-State Removal Uses Later Scoring

- GIVEN the same public probes with learned packed state and empty packed state
- WHEN both controllers decide before labels join for scoring
- THEN changed decisions and prospective error effects are retained per stream
- AND scheduled feedback-deletion effects match the producer rows.

### SCENARIO-CL-7228-ISOLATION: Future Authority Cannot Enter A Decision

- GIVEN baseline and shifted hidden future-label sidecars
- WHEN isolated fresh processes replay identical public bytes
- THEN their predictions remain equal before release
- AND the recorded file policy denies both sidecars inside those processes.

### SCENARIO-CL-7228-HISTORY: Prior Nulls Remain Null Evidence

- GIVEN the V635 refinement value and cold-audit promotion scores are zero
- WHEN the V636 audit records its producer and historical receipts
- THEN both V635 nulls remain explicit and unmodified
- AND no readiness, parity, or audit-completion score promotes them.

### SCENARIO-CL-7228-TERMINAL: Complete Producer Null Stays Complete Null

- GIVEN all owned audit checks pass and Exp7227 learning value equals zero
- WHEN terminal scores and verdict are derived
- THEN `belief_audit_complete_score` equals one
- AND `belief_promotion_score` equals zero with a complete null verdict.

## Implementation Status (REQ-CL-7228)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7228 and SCENARIO-CL-7228-* | Planned: independent row reduction, packed and reference cold replay, transaction attacks, future-label isolation, learned-state deletion, wrapper, and terminal artifact. | Planned: RED-first precondition, metric, cold, transaction, causality, isolation, history, terminal, command, and new-module coverage tests. |

## REQ-CL-7230: Native Lossless Belief Semantics

Carnot SHALL preserve the Exp7226 packed controller's survivor, vote, update,
empty-reset, rollback, majority, tie, abstention, and disagreement-energy
semantics in the compiled Rust boundary. The native controller SHALL keep its
state across calls. It SHALL accept query and update inputs as contiguous
batches. It SHALL expose a deterministic semantic checkpoint that contains
survivor masks, vote counts, epochs, and version without target labels or
hidden parameters.

The parity study SHALL use Exp7226 when `belief_compiler_ready_score` equals
one. It SHALL not require positive later learning value. It SHALL authenticate
the producer and reject quarantine before it reads the gate. Small-domain
checks SHALL exhaust every input for empty, singleton, and pair survivor sets.
Twenty fixed larger update and reload sequences SHALL compare native and
Python decisions, energies, reset behavior, and semantic state bytes. At least
one checkpoint SHALL restore and continue in a second process that loads the
exact selected extension.

### SCENARIO-CL-7230-PARITY: Native State Preserves The Packed Vote

- GIVEN every supported family and each tested small survivor subset
- WHEN Python and native controllers evaluate every finite-domain input
- THEN decisions, ties, abstentions, disagreement, and both label energies match
- AND batched support updates and empty resets produce identical semantic bytes.

### SCENARIO-CL-7230-RESTORE: Native State Survives A Process Boundary

- GIVEN a checkpoint from a completed fixed update sequence
- WHEN a fresh process loads the selected compiled extension and restores it
- THEN its semantic bytes and later query results match the first process
- AND rollback restores the exact previous semantic bytes.

## Implementation Status (REQ-CL-7230)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7230 and SCENARIO-CL-7230-* | Planned: scoped Rust packed-belief module, PyO3 registration, experiment controller, and thin wrapper. | Planned: RED-first exhaustive parity, fixed sequence, process restore, artifact, and scoped coverage tests. |

## REQ-CL-7240: Feedback-Validated Bounded Archive Recurrence Fixture

Carnot SHALL provide one opt-in controller over the existing packed finite
hypothesis state. The controller SHALL keep one active state and at most four
immutable archived survivor-mask states. When released feedback contradicts
the active state, it SHALL archive the active bytes before a reset. Inactive
hypotheses SHALL not be globally deleted. Archive nomination and reactivation
SHALL use released feedback only and SHALL not read stream names, event
numbers, generator seeds, regime names, boundaries, or evaluator sidecars.

The controller SHALL validate nominated archives against a rolling window of
the most recent 16 released witnesses. It SHALL reactivate only a candidate
with at least eight applicable witnesses and zero contradictions. It SHALL
choose deterministically by validation loss and then creation order. A shuffled
nomination arm SHALL use the same archive content, nomination count, observable
feedback, query budget, and final validation gate as the validated arm.

Every prediction SHALL read a frozen state before current feedback is
released. A release commit SHALL affect only later events. Delayed
contradictions SHALL not cause same-event correction, hindsight certificates,
or replay of labels that were not accessible to the controller. Durable state
SHALL preserve exact active and archived bytes across restart. A failed or
rolled-back commit SHALL restore the exact parent bytes and hash.

The fixture SHALL seal 32 streams of 1,024 events. The first 128 events in each
stream SHALL be warmup. The pending feedback capacity SHALL be four, the query
ceiling SHALL be 128 per stream, and delays SHALL come from zero, four, 16, and
32. All adaptive arms SHALL use the existing query acquisition. The authority
SHALL vary A-B-A recurrence, A-B-C unseen drift, gradual drift, and unchanged-
input label drift while keeping regimes and boundaries outside public bytes.
At least one A-B-A recurrence SHALL preserve identical public-input moments so
moment lookup alone cannot identify the returning constraint state.

The fixture SHALL prototype six arms: `frozen_warmup`,
`destructive_packed_learner`, `reset_relearn_no_archive`,
`unvalidated_stale_archive_reuse`, `validation_selected_archive`, and
`shuffled_nomination_validated`. Rows SHALL retain one independent stream and
arm, full prospective error and abstention denominators, and intended versus
actual query counts. Completion SHALL not assert that any science gate passed
by construction.

`recurrence_fixture_ready_score` SHALL equal one only when the sealed stream
contract, controller operations, delayed-feedback chronology, archive cap,
positive controls, authority isolation, restart retention, and byte-exact
rollback all pass. The exact finite evaluator is also the correctness authority,
so a ready fixture SHALL use `verifier_is_oracle=true` and
`verdict_class=circular_positive`. It SHALL set `MODEL_SPECS=[]`,
`model_invoked=false`, `inference_substrate=cpu_exact_solver_or_simulator`,
`inference_substrate_class=cpu_exact_solver_or_simulator`, and
`execution_venue=host`. A missing external prerequisite SHALL instead produce
a row-free `blocked_no_run` artifact with the exact failed gate receipt.

The terminal artifact SHALL use run date `20260912` and bind the experiment ID,
milestone, source code, public manifest, private authority, release schedule,
raw rows, and checkpoint bytes to hashes. It SHALL retain all required V637
artifact fields, a predeclared sample-size budget and seeds, a reproducibility
checksum, every acceptance gate, the controller and arm contracts, measured
CPU state bytes, and the future Rust SIMD and FPGA BRAM operation mapping. Only
an exact two-key dictionary containing `principle` and `value` may be unwrapped.

### SCENARIO-CL-7240-PRECONDITIONS: Exact Upstream Evidence Or Block

- GIVEN the V637 roadmap identity and exact Exp7227 artifact and raw receipts
- WHEN requirements, imports, writable paths, hashes, and quarantine are checked
- THEN only authenticated evidence can start stream generation
- AND an external failure produces a row-free blocked artifact.

### SCENARIO-CL-7240-ARCHIVE: Contradiction Preserves Bounded Past State

- GIVEN an active survivor mask and released contradictory feedback
- WHEN the post-prediction transaction resets the active state
- THEN the prior active bytes enter an immutable archive before reset
- AND deterministic eviction keeps at most four archived states.

### SCENARIO-CL-7240-VALIDATION: Released Witnesses Gate Reactivation

- GIVEN archived candidates and the rolling 16-release window
- WHEN candidates are nominated for reuse
- THEN fewer than eight applicable witnesses or any contradiction rejects reuse
- AND ties use validation loss followed by creation order.

### SCENARIO-CL-7240-CHRONOLOGY: Current Feedback Is Future-Only

- GIVEN a prediction from frozen state and feedback released for that event
- WHEN a transaction archives, resets, or reactivates state
- THEN the recorded prediction and certificate remain unchanged
- AND the new state can affect only a later event.

### SCENARIO-CL-7240-STREAMS: Sealed Regimes Stay Outside Public Bytes

- GIVEN 32 frozen streams with four evaluator-owned drift patterns
- WHEN public, release, and private authority files are sealed
- THEN every count, delay, warmup, capacity, and query limit matches the contract
- AND identical-moment recurrence exposes no boundary or regime identity.

### SCENARIO-CL-7240-ARMS: Six Controls Share Observable Budgets

- GIVEN the six preregistered arms on one public stream
- WHEN adaptive query acquisition and delayed delivery run
- THEN all adaptive arms receive the same selected releases and query ceiling
- AND shuffled nomination changes order without changing the final validation gate.

### SCENARIO-CL-7240-TRANSACTION: Restart And Rollback Preserve Hashes

- GIVEN a committed archive transition and its durable checkpoint
- WHEN a fresh controller restores it and the transition is rolled back
- THEN restart retains identical active and archive bytes
- AND rollback restores the exact parent state hash and later prediction.

### SCENARIO-CL-7240-TERMINAL: Readiness Is Infrastructure Evidence

- GIVEN all sealed-stream, positive-control, isolation, and transaction checks pass
- WHEN the terminal artifact is cold-validated and atomically written
- THEN `recurrence_fixture_ready_score` equals one
- AND the circular-positive verdict claims fixture readiness, not learning efficacy.

## Implementation Status (REQ-CL-7240)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7240 and SCENARIO-CL-7240-* | Implemented: `python/carnot/experiment_7240_v637_recurrence_fixture.py` provides the bounded archived-mask controller, separated 32-stream fixture, six-arm panel, checkpoint, and atomic artifact builder; `scripts/experiments/experiment_7240_v637_recurrence_fixture.py` is the thin wrapper. | `tests/python/test_experiment_7240_v637_recurrence_fixture.py` covers preconditions, exact principle unwrapping, delayed contradiction, archive cap, validation, shuffled nomination, stream separation, all six arms, restart, rollback, cold validation, blocked output, command delegation, and 100-percent new-code line coverage. |

## REQ-CL-7241: Prospective Learning Value From Validated Archive Reuse

Carnot SHALL evaluate the authenticated Exp7240 recurrence fixture without
changing production defaults. It SHALL replay all 32 frozen streams, 1,024
events, and six arms. This produces 196,608 prospective event-and-arm rows.
Each prediction and query receipt SHALL exist before its evaluator label is
released. Evaluator labels, drift patterns, regimes, seeds, and boundaries
SHALL remain outside controller inputs.

The replay SHALL retain measured costs and source hashes for prediction,
query, delayed delivery, validation, archive changes, and state writes. It
SHALL retain complete decision rows and all final active and archived state
bytes. It SHALL summarize one row per independent stream and arm. Each row
SHALL report future error, false accepts, abstention, recurrence error,
archive-hit validity, later changed decisions, and operation costs. An
abstention SHALL count as an error in every full-denominator efficacy metric.

The validation-selected archive arm SHALL be compared with frozen warmup,
destructive learning, reset and relearn, and shuffled nomination. Stale reuse
SHALL remain a safety negative control. Paired confidence intervals SHALL use
the 32 independent stream seeds. They SHALL use 10,000 frozen bootstrap draws.
Correlated events SHALL not increase the independent sample size.

`recurrence_learning_value_score` SHALL equal one only when every frozen gate
passes. The upper 95-percent confidence bound for future-error difference
SHALL be below zero against frozen warmup and reset and relearn. The upper
bound for recurrence-error difference SHALL be below zero against destructive
learning and shuffled nomination. Recurrence-error increase against frozen
warmup SHALL be no more than 0.02. The upper bound for false-accept difference
SHALL be no more than zero against frozen warmup and reset and relearn. The
run SHALL also show a valid archive reactivation, a later changed decision,
and zero pre-release decision differences.

Every gate SHALL retain its expected value, actual value, and pass state. No
changed output or no positive-control headroom SHALL produce an inconclusive
terminal null. It SHALL not retire the family. A failed efficacy gate after a
complete replay SHALL also produce a terminal null. `recurrence_run_complete_score`
SHALL equal one only when all 196,608 scheduled outcomes are accounted for.

The terminal artifact SHALL use run date `20260912`. It SHALL bind the task
identity, Exp7240 artifact, public streams, release schedule, private evaluator,
raw rows, state manifest, operation receipts, source files, and checkpoints to
exact hashes. It SHALL reject quarantined evidence even when a numeric upstream
gate passes. Only a dictionary with exactly the keys `principle` and `value`
may be unwrapped.

The run SHALL set `MODEL_SPECS=[]`, `model_invoked=false`, and all current
invocation counts to zero. A completed CPU replay SHALL use
`cpu_exact_solver_or_simulator` for both substrate fields and `host` for the
execution venue. A missing external prerequisite SHALL use `blocked_no_run`
and produce a row-free blocked artifact. Historical evidence and synthetic
negative receipts SHALL live in hashed sidecars, not current invocation fields.

The artifact SHALL report actual lookup and update p50 and p95, serialized
state bytes, total CPU time, and the measured gap to the 100-times hardware
target. It SHALL not treat the target as an observed result. The evaluator is
also the correctness authority, so the artifact SHALL set
`verifier_is_oracle=true`.

### SCENARIO-CL-7241-PRECONDITIONS: Exact Fixture Evidence Or Block

- GIVEN the V637 roadmap identity and declared Exp7240 deliverables
- WHEN requirements, imports, writable paths, hashes, checksums, and quarantine are checked
- THEN only the exact complete upstream fixture can enter replay
- AND an external failure produces a row-free terminal blocked artifact.

### SCENARIO-CL-7241-CHRONOLOGY: Decisions Precede Authority Release

- GIVEN public events, query choices, delayed schedules, and private labels
- WHEN each arm predicts and requests feedback
- THEN prediction and query receipts are sealed before authority access
- AND a commit can change only a later decision.

### SCENARIO-CL-7241-PANEL: All Frozen Units And Arms Run

- GIVEN 32 independent stream seeds and the six-arm contract
- WHEN the CPU panel runs once under the frozen stopping rule
- THEN exactly 196,608 event-and-arm outcomes are retained
- AND stale reuse remains a separate safety negative control.

### SCENARIO-CL-7241-METRICS: Prospective Outcomes Use Full Denominators

- GIVEN one later released label for each pre-release prediction
- WHEN stream-and-arm metrics are reduced
- THEN future, recurrence, false-accept, abstention, archive, and decision-change outcomes are retained
- AND every abstention contributes to full-denominator error.

### SCENARIO-CL-7241-BOOTSTRAP: Streams Are The Independent Units

- GIVEN paired metrics for all 32 stream seeds
- WHEN each frozen comparison receives 10,000 bootstrap draws
- THEN intervals resample complete paired streams
- AND event count does not enlarge the independent sample size.

### SCENARIO-CL-7241-GATES: Every Frozen Criterion Remains Independent

- GIVEN paired confidence intervals, recurrence tolerance, and causal counts
- WHEN learning value is scored
- THEN each criterion retains its threshold, actual value, and pass state
- AND learning value equals one only when every criterion passes.

### SCENARIO-CL-7241-NULL: Missing Headroom Is Inconclusive

- GIVEN no later decision change or no effective positive-control separation
- WHEN the complete panel is classified
- THEN the verdict is an inconclusive terminal null
- AND it does not retire the constraint family.

### SCENARIO-CL-7241-STATE: Exact Memory Bytes Remain Auditable

- GIVEN active masks, archived masks, certificates, and durable checkpoints
- WHEN archive changes and state writes complete
- THEN before and after bytes and hashes remain in the state manifest
- AND rejected updates plus rollback leave the parent bytes unchanged.

### SCENARIO-CL-7241-TERMINAL: Completion And Efficacy Stay Separate

- GIVEN all scheduled outcomes and validation receipts are accounted for
- WHEN the artifact is cold-validated and atomically written
- THEN run completion can equal one even when learning value equals zero
- AND the terminal verdict cannot claim live LLM or natural-language transfer evidence.

## Implementation Status (REQ-CL-7241)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7241 and SCENARIO-CL-7241-* | Implemented: `python/carnot/experiment_7241_v637_recurrence_learning.py` authenticates and replays Exp7240, retains measured operation receipts and state bytes, computes paired bootstrap gates, and builds the terminal artifact; `scripts/experiments/experiment_7241_v637_recurrence_learning.py` is the thin wrapper. | `tests/python/test_experiment_7241_v637_recurrence_learning.py` covers preconditions, quarantine, chronology, six arms, full-denominator metrics, paired bootstrap, independent gates, E2E rejection, restore, rollback, blocked output, cold validation, command delegation, and 100-percent new-code line coverage. |

## REQ-CL-7242: Independent Cold Audit Of Recurrence Memory

Carnot SHALL audit Exp7240 and Exp7241 from their declared V637 deliverables.
It SHALL authenticate both artifacts, every declared raw input, and every
checkpoint before replay. It SHALL reject quarantined evidence even when a
numeric gate passes. It SHALL unwrap only dictionaries that contain both
`principle` and `value`.

A fresh process SHALL reduce all six arms from raw prediction-before-release
rows. The reducer SHALL join authority feedback by request index. It SHALL
reject premature labels, duplicate feedback credit, seed-hash mismatches, and
archive choices that use a hidden regime. It SHALL recompute the frozen paired
seed criteria without reading producer aggregates. It SHALL retain the
identical-input label-drift and shuffled-candidate controls.

The audit SHALL withhold selected committed updates and prove that changes
start after each commit. A fresh process SHALL restore each final checkpoint.
The restored process SHALL compare actions, energies, active hashes, and archive
hashes byte-for-byte. Stale archives, a full pending queue, out-of-order
delivery, and corrupt certificates SHALL be rejected. Each rejected replay
SHALL preserve the complete parent state.

`recurrence_audit_complete_score` SHALL equal one only after the full cold
reduction and all audit-owned checks finish. `recurrence_promotion_score` SHALL
equal one only when the independently recomputed science gates and all
causality, isolation, restore, and rollback checks pass. A failed science gate
after complete audit work SHALL produce a terminal null. A missing external
input SHALL produce a row-free terminal blocked artifact. It SHALL never
produce partial for an external absence.

The terminal artifact SHALL use run date `20260912`. It SHALL bind the task,
both upstream artifacts, public and private stream files, raw rows, checkpoints,
mutation receipts, and source code to exact hashes. Historical Exp7228 details
and synthetic negative receipts SHALL remain in a separate hashed sidecar. They
SHALL not enter current invocation provenance.

The run SHALL set `MODEL_SPECS=[]`, `model_invoked=false`, and all current
invocation counts to zero. A completed replay SHALL use
`cpu_exact_solver_or_simulator` for both substrate fields and `host` for the
execution venue. A blocked run SHALL use `blocked_no_run`. The evaluator also
defines correctness, so exact-oracle conformance SHALL use
`verifier_is_oracle=true` and a circular-positive class only when promotion
passes. The audit SHALL claim no model-weight update, cross-domain transfer,
hardware speedup, production-default change, or publication.

### SCENARIO-CL-7242-PRECONDITIONS: Both V637 Producers Or Block

- GIVEN the declared Exp7240 and Exp7241 deliverables and their raw receipts
- WHEN identity, hash, checksum, quarantine, imports, and writable paths are checked
- THEN only exact complete evidence can enter the reducer
- AND an external failure produces a row-free blocked artifact with the exact failed check.

### SCENARIO-CL-7242-REDUCTION: Raw Rows Define Every Metric

- GIVEN all 196,608 prediction-before-release rows across six arms
- WHEN a fresh process rebuilds the 192 independent seed-and-arm summaries
- THEN it joins labels by request index and ignores producer aggregate values
- AND it recomputes every frozen paired-seed criterion.

### SCENARIO-CL-7242-CHRONOLOGY: Authority Cannot Affect Its Prediction

- GIVEN delayed feedback and a frozen pre-release decision
- WHEN release chronology and update withholding are audited
- THEN no label receives premature or duplicate credit
- AND a committed update can change only a later action.

### SCENARIO-CL-7242-CONTROLS: Drift And Shuffle Remain Distinct

- GIVEN identical-input label drift and shuffled archive nomination
- WHEN the audit compares their raw outcomes
- THEN both controls retain their independent denominators
- AND no hidden regime field can select an archive.

### SCENARIO-CL-7242-RESTORE: Fresh State Is Byte-Exact

- GIVEN each final active state and bounded archive checkpoint
- WHEN a fresh process restores the checkpoint
- THEN actions, energies, active hashes, and archive hashes match
- AND the process uses no private authority input.

### SCENARIO-CL-7242-MUTATIONS: Unsafe Replay Preserves Parent State

- GIVEN stale archives, a full queue, out-of-order delivery, and corrupt certificates
- WHEN each invalid transaction is attempted
- THEN every transaction is rejected with a named reason
- AND every parent byte and hash remains unchanged.

### SCENARIO-CL-7242-PROVENANCE: Old Audit Evidence Stays Separate

- GIVEN the quarantined Exp7228 history and task-owned negative fixtures
- WHEN provenance is sealed
- THEN their paths and hashes live only in the mutation sidecar
- AND current model and runner invocation fields remain empty.

### SCENARIO-CL-7242-TERMINAL: Completion And Promotion Stay Separate

- GIVEN all independent reductions and safety checks
- WHEN terminal scores are derived and the artifact is atomically written
- THEN audit completion can equal one while promotion equals zero
- AND only reproduced science plus every safety check can promote.

## REQ-CL-7243: Native Archive Controller Conformance And Cost

Carnot SHALL measure the Exp7240 validation-selected archive controller with
the existing native packed-belief kernel. The Python and native arms SHALL use
the same 32 stream seeds, validation decisions, state transitions, archive
selection code, and delayed releases. The native arm SHALL execute the real
PyO3 `RustPackedBeliefController`; a Python substitute SHALL not count.

The study SHALL compare predictions, energies, query choices, active state,
archive state, and complete controller hashes. It SHALL restore native state in
a fresh interpreter and continue delayed updates. Exact parity uses
`verifier_is_oracle=true`, so a passing result SHALL use `circular_positive`.

The cost study SHALL run 30 paired interleaved blocks for every combination of
archive capacities 1, 2, and 4 and batch sizes 1, 16, and 128. Each row SHALL
charge Python dispatch, binding conversion, archive nomination, validation,
lookup, update, serialization, and restore. Cold import and build costs SHALL
remain separate. The artifact SHALL report their amortization break-even.

`native_archive_ready_score` SHALL equal one only after real native execution,
zero scheduled parity mismatches, exact fresh-process restore, and exact
delayed-update continuation. `native_archive_cost_value_score` SHALL equal one
only when readiness passes and the lower paired 95-percent confidence bound
for batch-one total event speedup exceeds one. NFR-01 at 10 times and the
research-program target at 100 times SHALL remain separate measured lower-bound
gates.

The executable SHALL authenticate the V637 task, Exp7240 artifact and sidecars,
the Exp7217 interpreter recipe, and the quarantined Exp7230 history before
native work. It SHALL reject quarantined evidence before reading a numeric
gate. It SHALL unwrap only dictionaries containing exactly `principle` and
`value`. A missing external prerequisite SHALL yield a row-free `blocked_no_run`
artifact with an exact failed check.

The task SHALL use `MODEL_SPECS=[]`, `model_invoked=false`, zero current
invocation counters, both substrate fields set to
`cpu_exact_solver_or_simulator` after execution, and `execution_venue=host`.
Historical and synthetic evidence SHALL remain in hashed sidecars. The task
SHALL preserve Exp7230 unchanged and SHALL not change production defaults.

### SCENARIO-CL-7243-PARITY: Both Arms Share One Archive Controller

- GIVEN the same streams, feedback schedule, controller policy, and initial state
- WHEN the Python and native packed active states process all scheduled units
- THEN predictions, energies, queries, transitions, and archive hashes match exactly
- AND the exact match is classified as circular evidence.

### SCENARIO-CL-7243-RESTORE: Fresh Native Continuation Is Exact

- GIVEN serialized active and archive state with pending delayed releases
- WHEN a fresh interpreter restores the native state and continues updates
- THEN its predictions, energies, queries, and final state hash match Python
- AND the receipt identifies the exact loaded extension bytes.

### SCENARIO-CL-7243-COST: Complete Batch-One Cost Gates Value

- GIVEN 30 paired blocks across three capacities and three batch sizes
- WHEN both arms perform the complete archive event boundary
- THEN component and total costs remain in every arm row
- AND only the batch-one paired lower confidence bound can set cost value.

### SCENARIO-CL-7243-TERMINAL: Performance And Oracle Limits Stay Visible

- GIVEN exact parity and measured full-boundary costs
- WHEN the validator derives readiness, value, 10-times, and 100-times gates
- THEN every gate retains its threshold, actual value, and pass state
- AND a failed cost gate is null while a passing oracle result is circular-positive.

## Implementation Status (REQ-CL-7243)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7243 and SCENARIO-CL-7243-* | Implemented: `python/carnot/experiment_7243_v637_native_memory.py` keeps the Exp7240 archive policy shared while executing active packed state through the real interpreter-bound PyO3 kernel; it records exact parity, fresh-process continuation, paired full-boundary costs, and a terminal artifact. | `tests/python/test_experiment_7243_v637_native_memory.py` covers authenticated preconditions, native/archive parity, fresh restore, cost gates, malformed-row validation, failed build and atomic cleanup, orchestration, and CLI paths with 754/754 scoped statements. |

## REQ-CL-7253: Bounded Coverage Archive And Effective Memory Controls

Carnot SHALL add an opt-in finite archive controller to the Exp7240 prototype.
Each snapshot SHALL have an accept, reject, or abstain signature on no more than
16 released witnesses. The controller SHALL remove duplicate snapshot
signatures. It SHALL greedily retain at most four snapshots that maximize
distinct signature coverage. Ties SHALL use the frozen distance, recency, and
snapshot identity rule. The controller SHALL preserve the eight-witness,
zero-contradiction reactivation gate and byte-exact rollback behavior.

The controller SHALL bound and measure all live memory. This includes the
witness window, four archives, pending feedback, and durable ledger. It SHALL
retain no unbounded helper collection. A failed parent, future release,
duplicate release, corrupt state, or stale rollback SHALL leave live and
durable parent bytes unchanged.

The experiment SHALL define a seeded shuffled mapping from released validation
signatures to candidate snapshot identities. It SHALL apply this mapping before
the unchanged final safety gate. Receipts SHALL include mappings and selected
identities before and after the intervention. A diagnostic fixture with
multiple distinguishable eligible candidates SHALL change at least one
selection. A changed enumeration with the same selection SHALL fail the
control.

The experiment SHALL freeze eight arms: frozen warmup, reset and relearn,
destructive update, FIFO archive with effective shuffled nomination, FIFO
aligned archive, coverage archive with effective shuffled nomination, coverage
aligned archive, and an oracle positive control. The four archive arms SHALL
share archive capacity, total byte cap, query budget, release budget, delayed
feedback, and final safety gate.

Eight development streams SHALL choose one signature distance and tie rule.
The experiment SHALL then seal 32 new independent prospective streams. Each
stream SHALL contain 1,024 events, including 128 warmup and 896 prospective
events. It SHALL preserve the prior schedule families, noisy delayed releases,
identical-input label drift, and cyclic returns. Public controller rows SHALL
not expose private labels, regime identities, boundaries, or generator seeds.

`coverage_fixture_ready_score` SHALL equal one only when the streams and all
controls are sealed, both admission modes are distinct, the shuffle changes a
candidate identity, all memory components remain within their caps, chronology
and authority isolation pass, and E2E-007 restart and rollback checks pass. This
fixture SHALL not score or claim a learning gain.

The terminal artifact SHALL use run date `20260912`. It SHALL bind the task,
three V637 upstream artifacts, source code, configuration, separated stream
views, raw rows, state and control sidecars to exact hashes. It SHALL retain
ordinary top-level values for every required artifact field and explanations in
`field_principles`. Historical evidence and injected negative fixtures SHALL
remain in a hashed sidecar outside current invocation fields.

The run SHALL set `MODEL_SPECS=[]`, `model_invoked=false`, and all current model,
load, generation, and inference counters to zero. A completed CPU run SHALL use
`cpu_exact_solver_or_simulator` for both substrate fields and `host` for the
execution venue. The exact evaluator defines correctness, so a ready fixture
SHALL use `verifier_is_oracle=true` and `verdict_class=circular_positive`. A
missing or quarantined external prerequisite SHALL produce a row-free terminal
blocked artifact with an exact `gate_check_summary`. It SHALL not produce a
partial result for external absence.

### SCENARIO-CL-7253-PRECONDITIONS: Exact V637 Evidence Or Block

- GIVEN the declared Exp7253 roadmap item and all three V637 upstream artifacts
- WHEN source bytes, hashes, quarantine state, imports, and output ownership are checked
- THEN only exact complete evidence can start stream generation
- AND an external failure produces a row-free terminal blocked artifact.

### SCENARIO-CL-7253-COVERAGE: Distinct Signatures Drive Bounded Retention

- GIVEN more than four candidate snapshots and at most 16 released witnesses
- WHEN coverage admission removes duplicates and greedily selects snapshots
- THEN at most four distinct signatures remain under the frozen tie rule
- AND a diagnostic case differs from FIFO eviction.

### SCENARIO-CL-7253-SHUFFLE: Mapping Changes Candidate Identity

- GIVEN multiple distinguishable candidates that pass the same final safety gate
- WHEN the seeded signature-to-identity mapping is applied
- THEN the before and after mapping receipts differ
- AND at least one eligible selected snapshot identity changes.

### SCENARIO-CL-7253-MEMORY: Every Mutable Collection Has A Byte Cap

- GIVEN witnesses, archives, pending feedback, and ledger receipts
- WHEN the controller reaches each declared capacity
- THEN component and total serialized bytes remain at or below their caps
- AND no hidden auxiliary collection grows with stream length.

### SCENARIO-CL-7253-STREAMS: Development And Prospective Authority Stay Separate

- GIVEN eight development streams and 32 newly seeded prospective streams
- WHEN the public, release, private authority, and manifest bytes are sealed
- THEN all fixed counts, schedules, drift families, delays, and noise rules conform
- AND no private label, regime, boundary, or seed reaches controller input.

### SCENARIO-CL-7253-ARMS: Eight Controls Share Frozen Budgets

- GIVEN the eight declared arms and four archive arms
- WHEN the fixture replays all sealed units
- THEN every arm produces a complete unit row
- AND the four archive arms share capacity, byte, query, and release limits.

### SCENARIO-CL-7253-TRANSACTION: Released Feedback Changes Only Later State

- GIVEN a prediction, delayed released feedback, and its exact parent hash
- WHEN a commit succeeds or an invalid parent is rejected
- THEN only a later decision can observe an admitted change
- AND rejection and rollback preserve exact parent bytes on disk and in memory.

### SCENARIO-CL-7253-TERMINAL: Readiness Does Not Claim Learning Gain

- GIVEN sealed streams, effective controls, bounded memory, and E2E-007 receipts
- WHEN the independent raw-row reducer and cold validator pass
- THEN `coverage_fixture_ready_score` equals one with circular-positive conformance
- AND scientific learning efficacy remains explicitly unscored.

## Implementation Status (REQ-CL-7253)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7253 and SCENARIO-CL-7253-* | Implemented: `python/carnot/experiment_7253_v638_coverage_memory.py` provides bounded coverage and FIFO archive modes, effective candidate mapping, separated stream seals, eight-arm replay, E2E-007 controls, raw reduction, and atomic artifacts; `scripts/experiments/experiment_7253_v638_coverage_memory.py` is the thin wrapper. | `tests/python/test_experiment_7253_v638_coverage_memory.py` covers admission, shuffle identity, byte caps, chronology, authority isolation, stream conformance, all arms, transaction failures, reload, rollback, raw reduction, cold validation, command dispatch, and 100-percent new-code line coverage. |

## REQ-CL-7254: Prospective Learning With Bounded Coverage Memory

Carnot SHALL replay all 32 sealed Exp7253 prospective streams. Each stream
SHALL contain 1,024 events across the eight frozen arms. All arms SHALL receive
the same public observations and release schedule. Each prediction SHALL occur
before feedback. Private labels and hidden regimes SHALL remain evaluator-only.

The run SHALL authenticate `coverage_fixture_ready_score=1`. It SHALL also
authenticate the Exp7253 controller, arm, stream, query, release, memory, and
safety contracts from exact artifact and sidecar bytes. No contract value SHALL
change after the first prospective label.

The run SHALL measure future full-denominator error, false accepts,
abstentions, recurrence error, query cost, memory, and decision changes for each
stream and arm. Raw chronological rows SHALL retain actual commit parent and
child hashes. Operation rows SHALL count constraint additions, deactivations,
valid reactivations, selection changes, and later unseen decision changes.

The coverage-aligned arm SHALL be the treatment. It SHALL use 10,000 paired
bootstrap resamples over the 32 independent streams. Its paired upper 95-percent
confidence bound for future error against frozen and reset SHALL be below zero.
Its paired upper bound for false accepts against frozen and reset SHALL be at
most zero. Its recurrence error increase against frozen SHALL be at most 0.02.
Its paired upper recurrence-error bound against destructive and coverage
shuffled SHALL be below zero. It SHALL have positive value against FIFO aligned
at matched memory. The prospective shuffle SHALL change a selected archive
identity. At least one valid reactivation SHALL change a later unseen decision.
No pre-release decision difference is permitted. The oracle arm SHALL show
control headroom only.

`coverage_run_complete_score` SHALL equal one when all 262,144 event-arm rows
are present and uncensored. `coverage_learning_value_score` SHALL equal one only
when every efficacy, safety, retention, FIFO, and shuffle gate passes. A complete
run with a failed scientific gate SHALL be a terminal null. It SHALL not change
production defaults.

The run SHALL measure lookup, update, serialization, durable commit, and memory
costs separately. It SHALL report the Tier 1 target below 1 microsecond and the
Tier 2 target below 1 millisecond as measured pass or failure. It SHALL report
the measured gap to the separate 100-times acceleration target. Hardware
suitability SHALL cite bounded bitset operations and the measured call graph.

The run SHALL set `MODEL_SPECS=[]`, `model_invoked=false`, and all current model
counters to zero. A completed run SHALL use `cpu_exact_solver_or_simulator` for
both substrate fields and `host` for execution venue. The exact evaluator SHALL
set `verifier_is_oracle=true`, so a successful learning result is
`circular_positive`, never `positive`. A missing or quarantined external input
SHALL produce a row-free terminal blocked artifact with an exact gate summary.

### SCENARIO-CL-7254-PRECONDITIONS: Exact Coverage Fixture Or Block

- GIVEN the declared Exp7254 roadmap item and sealed Exp7253 evidence
- WHEN readiness, contract, source, quarantine, import, and output checks run
- THEN only exact complete evidence can start the replay
- AND an external failure produces a row-free terminal blocked artifact.

### SCENARIO-CL-7254-PREQUENTIAL: Prediction Precedes Released Feedback

- GIVEN paired public events and delayed releases across all eight arms
- WHEN each event is processed
- THEN every prediction and query receipt precedes label access
- AND each successful commit records its actual parent and child state hash.

### SCENARIO-CL-7254-REDUCTION: Independent Streams Define Learning Value

- GIVEN all chronological event and operation rows
- WHEN the independent reducer computes stream-arm metrics
- THEN it reproduces future, safety, recurrence, cost, memory, and change totals
- AND 10,000 paired bootstrap draws use streams rather than events as units.

### SCENARIO-CL-7254-CAUSAL: Coverage Changes Later Unseen Decisions

- GIVEN aligned, shuffled, FIFO, destructive, reset, frozen, and oracle controls
- WHEN released constraints add, deactivate, or reactivate state
- THEN the run counts actual prospective selection and later-decision changes
- AND zero decision difference occurs before any released feedback.

### SCENARIO-CL-7254-COST: Tier Targets Remain Measured Gates

- GIVEN measured lookup, update, serialization, durable commit, and memory rows
- WHEN CPU costs are reduced
- THEN Tier 1 and Tier 2 targets report measured pass or failure
- AND the 100-times acceleration gap is not replaced by an assumed speedup.

### SCENARIO-CL-7254-E2E: Restart And Rejected Update Preserve State

- GIVEN the complete prequential replay and one durable controller
- WHEN a valid update, fresh reload, rejected stale-parent update, and rollback run
- THEN restart preserves decisions and the rejected update preserves exact bytes
- AND rollback restores the exact committed parent.

### SCENARIO-CL-7254-TERMINAL: Completion And Scientific Value Stay Separate

- GIVEN every scheduled row and every frozen acceptance gate
- WHEN the terminal verdict is classified
- THEN run completion stays one even if scientific value is zero
- AND a failed scientific gate yields `complete_null` without promotion.

## Implementation Status (REQ-CL-7254)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7254 and SCENARIO-CL-7254-* | Implemented: `python/carnot/experiment_7254_v638_coverage_learning.py` authenticates the sealed coverage fixture, replays eight arms, records actual commit hashes and costs, computes paired stream gates, runs E2E-007 controls, and writes an atomic terminal artifact. The script entrypoint remains thin. | `tests/python/test_experiment_7254_v638_coverage_learning.py` covers exact preconditions, blocked output, prequential replay, raw reduction, causal gates, cost targets, E2E restart and rollback, terminal validation, CLI dispatch, defensive failures, and 100-percent new-code line coverage. |

## REQ-CL-7255: Independent Coverage Learning Causality Audit

Carnot SHALL audit the complete Exp7253 fixture and the complete Exp7254
learner in a fresh process. The audit SHALL authenticate exact artifact and
sidecar bytes before reduction. A missing, malformed, or quarantined upstream
SHALL produce a row-free terminal blocked result. An upstream scientific null
SHALL remain eligible for a complete safety audit.

The audit SHALL reconstruct every stream-arm error, false-accept, abstention,
recurrence, query, release, update, reactivation, decision-change, cost, and
memory value from the hashed event rows. It SHALL rebuild the frozen 10,000-draw
paired confidence intervals from streams with bootstrap seed 7,254,951. It
SHALL recompute the unchanged recurrence increase limit of 0.02 and every
Exp7254 scientific gate without using producer aggregates.

The audit SHALL replay the finite controllers from public observations and
released feedback. It SHALL prove that predictions precede feedback and that
controller inputs exclude labels and private regime fields. It SHALL validate
each commit parent and child hash. It SHALL validate every reactivation against
its contemporaneous released witnesses. It SHALL enforce every component and
total memory bound. Cold final states SHALL match the authenticated state
manifest and durable bytes.

For FIFO and coverage admission, the audit SHALL compare aligned and shuffled
arms. Each stream and admission mode SHALL report candidate-mapping changes,
selected-archive changes, later-decision changes, and zero-headroom counts.
Zero-headroom events SHALL remain in the denominator.

Five isolated negative fixtures SHALL mutate FIFO and coverage into one
function, cancel the shuffle, expose a future label, change a private regime,
and omit a durable write. Each mutation SHALL fail its named check. These
fixtures and the preserved Exp7242 historical receipt SHALL live in one hashed
sidecar outside current invocation fields.

The E2E-007 adaptation SHALL cold-load durable state, predict one unseen query,
apply delayed feedback, accept one valid commit, reject one wrong-parent
commit, and reload in another process with decision parity. It SHALL prove
that rejected writes preserve bytes and that rollback restores exact parent
bytes.

`coverage_audit_complete_score` SHALL equal one when raw reduction, controller
replay, causality, bounds, final-state, mutation, and E2E checks complete.
`coverage_promotion_score` SHALL equal one only when the independent audit also
reproduces every Exp7254 scientific value gate. A complete audit of a null
learner SHALL use `verdict_class=null` and SHALL not promote the learner.

The run SHALL use date `20260913`, `MODEL_SPECS=[]`, `model_invoked=false`, and
zero current model counters. Completed work SHALL use
`cpu_exact_solver_or_simulator` for both substrate fields and `host` as the
execution venue. The exact evaluator SHALL set `verifier_is_oracle=true`, so a
successful promotion could only be `circular_positive`. The task SHALL not
change model weights, production defaults, or external systems.

### SCENARIO-CL-7255-PRECONDITIONS: Exact Upstreams Or Block

- GIVEN the fixed Exp7253, Exp7254, raw, stream, state, and historical receipts
- WHEN hashes, checksums, quarantine state, imports, and output ownership are checked
- THEN only exact complete upstream evidence can enter the fresh reducer
- AND an external failure produces a row-free blocked artifact with the exact failed field.

### SCENARIO-CL-7255-REDUCTION: Raw Rows Define Every Metric

- GIVEN all 262,144 hashed event-arm rows
- WHEN the fresh reducer joins public, release, and private authority rows
- THEN it reconstructs all 256 stream-arm summaries and eight paired intervals
- AND stored producer outcomes or aggregates cannot replace the reconstructed values.

### SCENARIO-CL-7255-CAUSALITY: Feedback And Regimes Stay Hidden

- GIVEN public decisions and delayed releases
- WHEN the audit replays each event and commit
- THEN no decision sees unreleased feedback or a private regime field
- AND every parent, child, reactivation witness, and memory bound passes independently.

### SCENARIO-CL-7255-CONTROLS: Shuffles And Admission Must Be Effective

- GIVEN FIFO, coverage, aligned, and shuffled archive arms
- WHEN their nomination and decision histories are compared
- THEN mapping, selected-archive, later-decision, and zero-headroom counts are retained
- AND a canceled or aliased intervention cannot pass as effective.

### SCENARIO-CL-7255-MUTATIONS: Five Attacks Fail Their Named Checks

- GIVEN isolated admission, shuffle, label, regime, and durable-write mutations
- WHEN each negative fixture is audited
- THEN each mutation fails its corresponding check
- AND the negative bytes remain in a hashed sidecar outside current compute fields.

### SCENARIO-CL-7255-E2E: Durable Learning Survives A Fresh Process

- GIVEN one cold controller and one unseen query
- WHEN delayed feedback is committed, a wrong parent is rejected, and another process reloads
- THEN accepted state and later decisions have parity after reload
- AND rejection and rollback preserve exact durable bytes.

### SCENARIO-CL-7255-TERMINAL: Completion Does Not Imply Promotion

- GIVEN a complete independent audit and recomputed Exp7254 gates
- WHEN at least one scientific gate fails
- THEN `coverage_audit_complete_score` remains one and `coverage_promotion_score` is zero
- AND the terminal verdict is `complete_null` with no production promotion.

## Implementation Status (REQ-CL-7255)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7255 and SCENARIO-CL-7255-* | Implemented: `python/carnot/experiment_7255_v638_coverage_audit.py` authenticates Exp7253/Exp7254, independently reduces raw rows, instruments shipped finite-controller replay, audits aligned/shuffled effects, runs five isolated mutations and E2E-007, and seals a no-LLM terminal artifact through a thin script entrypoint. | Verified: `tests/python/test_experiment_7255_v638_coverage_audit.py` covers exact and blocked preconditions, raw reconstruction, authority isolation, controller/state replay, controls, all mutations, E2E restart/rejection/rollback, null promotion, terminal validation, CLI dispatch, defensive failures, and 100-percent new-code statement coverage. |

## REQ-CL-7256: Persistent Native Archive Controller

Carnot SHALL move the complete Exp7240 FIFO archive controller into one
persistent native object. The object SHALL own active masks, archive entries,
the released-witness window, validation receipts, version, and hash lineage.
It SHALL preserve the Exp7240 prediction, energy, query, archive selection,
release-order, commit, and rollback semantics. It SHALL not adopt Exp7253
coverage admission.

The study SHALL replay eight fixed existing streams through the Python
reference, the Exp7243 native-active wrapper, and the new native controller.
Each event and arm SHALL retain decisions, semantic state, archive identities,
and lineage. The replay SHALL exercise all four archive slots, conflicting
delayed releases, rollback, durable snapshot restore, and a fresh-process next
decision. A mismatch SHALL fail readiness.

The native hot path SHALL accept typed events. It SHALL not parse controller
JSON or reconstruct an active object for each prediction. Snapshot parsing is
permitted only during explicit construction or restore. Conversion-count rows
SHALL show typed calls, snapshot parses, and active reconstructions for each
arm. Throughput remains unscored until Exp7257.

`native_controller_ready_score` SHALL equal one only after exact three-arm
parity, exact durable and fresh-process continuation, transaction-negative
tests, and visible removal of repeated hot-path conversion. The exact Python
reference is an oracle. A ready result SHALL therefore use
`verdict_class=circular_positive`, never `positive`.

The executable SHALL authenticate exact upstream bytes, quarantine state,
imports, output ownership, the current interpreter, and the isolated extension
before replay. It SHALL set `MODEL_SPECS=[]`, `model_invoked=false`, all current
model counters to zero, both substrate fields to
`cpu_exact_solver_or_simulator`, and `execution_venue=host`. External absence
SHALL produce a row-free terminal blocked artifact. It SHALL not publish a
success-shaped artifact before measurement and cold validation finish.

### SCENARIO-CL-7256-PARITY: Three Controllers Preserve FIFO Semantics

- GIVEN eight fixed V637 streams and identical released feedback
- WHEN Python, native-active, and persistent-native controllers process every unit
- THEN predictions, energies, queries, archive selection, commits, and semantic states match
- AND all four FIFO archive slots and conflicting delayed releases are exercised.

### SCENARIO-CL-7256-PERSISTENCE: Native Ownership Removes Repeated Conversion

- GIVEN one imported persistent native controller
- WHEN typed prediction, energy, query, and commit operations execute
- THEN active state and archive policy stay in the same native object
- AND no prediction parses JSON or reconstructs an active controller.

### SCENARIO-CL-7256-TRANSACTION: Invalid Changes Preserve Exact Parent Bytes

- GIVEN a valid controller parent and durable snapshot
- WHEN malformed snapshots, stale parents, interrupted writes, or stale rollbacks occur
- THEN the operation fails before admission
- AND live and durable parent bytes remain exact.

### SCENARIO-CL-7256-RESTORE: A Fresh Process Continues Identically

- GIVEN a validated commit and native snapshot
- WHEN a new process loads the isolated extension and restores that snapshot
- THEN its semantic state and next decisions match Python on every unit
- AND the receipt binds the exact interpreter and extension bytes.

### SCENARIO-CL-7256-TERMINAL: Readiness Is Circular And Throughput Is Deferred

- GIVEN exact replay, restore, transaction, and conversion evidence
- WHEN the independent reducer and cold validator derive the frozen gates
- THEN readiness can equal one with `circular_positive` classification
- AND no throughput value or new learning behavior is claimed.

## Implementation Status (REQ-CL-7256)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7256 and SCENARIO-CL-7256-* | Implemented: persistent native FIFO controller, typed operations, isolated build, eight-stream replay, restart, and terminal artifact. | Verified: RED-first parity, conversion, snapshot, transaction, restart, artifact, command, and 100% scoped new-module coverage tests. |

## REQ-CL-7267: Active Recognition Before Constraint-Memory Reactivation

Carnot SHALL diagnose the complete Exp7254 and Exp7255 null evidence before it
builds a replacement fixture. The diagnosis SHALL reduce the saved selection
rows into archive-size, distinguishability, eligibility, shuffled-choice, and
later-decision counts. It SHALL separate one-candidate states, merged
signatures, and safety rejection. The result SHALL not assign signature rebasing
as the cause unless an observed replay preserves a distinguishing
counterexample on a stable released-only witness basis.

The recognition controller SHALL keep at most four immutable survivor-mask
snapshots. It SHALL keep a bounded stable witness basis that retains released
examples when they discriminate stored snapshots. It SHALL not rebase stored
snapshot identity onto only the most recent examples. A separate bounded fresh
validation set SHALL decide whether a stored snapshot is safe to reactivate.
Reactivation SHALL require fresh released feedback, at least eight applicable
fresh witnesses, and zero contradictions.

When more than one archived survivor mask remains distinguishable, the active
arm SHALL select a public query that maximizes archived prediction disagreement.
The random-query arm SHALL use the same candidate set and query budget. The
shuffled-association arm SHALL permute archive identities while it preserves
snapshot contents. A query and its later feedback SHALL not change the
prediction for the event that requested it. No hidden regime ID, future label,
evaluator parameter, or supplied boundary SHALL enter nomination. Existing
released contradiction detection SHALL provide the online change signal.
Supplied boundaries MAY appear only in an oracle-assisted diagnostic row.

Eight development streams SHALL freeze all representation, query, memory, and
tie parameters before prospective labels are inspected. The fixture SHALL then
seal 24 fresh streams of 1,024 events. Each stream SHALL have 128 warmup events
and at most 128 charged feedback queries per arm. Twelve prospective streams
SHALL use separated recurring regimes. Twelve SHALL use deliberately
overlapping regimes. Public events, release schedules, and private evaluator
authority SHALL use separate sealed bytes.

The fixture SHALL replay frozen, reset, previous coverage, active recognition,
random-query recognition, shuffled archive association, full version-space
memory, and feedback-withheld arms. All arms SHALL use the same public stream.
The bounded arms SHALL include pending requests, witness basis, archives, and
ledgers within 69,632 total serialized bytes. The 32 old streams SHALL remain a
diagnostic input only and SHALL not enter prospective fixture rows.

The E2E control SHALL process a released event, request a query, receive delayed
feedback, commit atomically, make a later prediction, cold restore, and roll
back. Mutation controls SHALL reject future-label access, a shuffled identity
mapping that cannot change selection, duplicate release IDs, and stale parent
hashes. Rejected writes SHALL preserve exact in-memory and durable bytes.

`recognition_fixture_ready_score` SHALL equal one only when at least one
development selection changes under active discrimination, the stable-basis
counterexample survives, all 24 prospective streams and eight arms are
complete, and every query, memory, mutation, and E2E control passes. Readiness
SHALL not depend on favorable held-out error. A complete fixture with a failed
mechanism gate SHALL be a terminal null with a specific stop reason.

The task SHALL use date `20260913`, `MODEL_SPECS=[]`, and
`model_invoked=false`. Every current load, generation, inference, and usable
answer count SHALL be zero. Completed CPU work SHALL use
`cpu_exact_solver_or_simulator` for both substrate fields and `host` for the
execution venue. The exact evaluator SHALL set `verifier_is_oracle=true`.
Therefore a ready fixture SHALL use `verdict_class=circular_positive`, never
`positive`. A missing or quarantined external prerequisite SHALL produce a
row-free terminal blocked artifact with an exact `gate_check_summary`.

### SCENARIO-CL-7267-PRECONDITIONS: Exact Null Evidence Or Block

- GIVEN the active Exp7267 task, both complete V638 null artifacts, and their saved rows
- WHEN hashes, quarantine state, imports, requirements, and output ownership are checked
- THEN only exact complete evidence can start diagnosis or stream generation
- AND an external failure produces a row-free terminal blocked artifact.

### SCENARIO-CL-7267-DIAGNOSIS: Selection Stages Name The Observed Cause

- GIVEN saved V638 nomination summaries and an instrumented released-only replay
- WHEN archive size, distinct candidates, eligible candidates, choices, and later decisions are reduced
- THEN one-candidate, signature-merge, and safety-rejection states remain separate
- AND rebasing is named only if a stable-basis counterexample stays distinguishable.

### SCENARIO-CL-7267-BASIS: Released Discriminators Stay Stable And Bounded

- GIVEN more released examples than the witness capacity
- WHEN stable-basis retention evaluates archived survivor-mask disagreement
- THEN retained examples preserve discriminatory signatures under the fixed tie rule
- AND fresh validation witnesses remain separate from the identity basis.

### SCENARIO-CL-7267-QUERY: Maximum Disagreement Drives Charged Feedback

- GIVEN at least two distinguishable archived survivor masks and public query candidates
- WHEN active recognition selects one query
- THEN it maximizes archived prediction disagreement under the frozen tie rule
- AND neither evaluator authority nor the later query label reaches that decision.

### SCENARIO-CL-7267-STREAMS: Development And Prospective Authority Stay Separate

- GIVEN eight development streams and 24 fresh prospective streams in two equal strata
- WHEN public, release, private authority, raw, and manifest bytes are sealed
- THEN all fixed counts, delays, identities, and query limits conform
- AND no label, regime, boundary, parameter, or seed reaches controller input.

### SCENARIO-CL-7267-PANEL: Eight Arms Share Frozen Limits

- GIVEN all eight declared arms and the 24 sealed prospective streams
- WHEN the fixture replays every event without efficacy-based stopping
- THEN every stream-arm row is complete and uncensored
- AND each bounded arm stays within 128 queries, four archives, and 69,632 bytes.

### SCENARIO-CL-7267-TRANSACTION: Feedback Changes Only Later Durable State

- GIVEN a public event, query, delayed release, exact parent, and durable controller
- WHEN commit, later prediction, reload, rejection, and rollback execute
- THEN only a later prediction can observe the released change
- AND future labels, duplicate releases, stale parents, and ineffective shuffles fail closed.

### SCENARIO-CL-7267-TERMINAL: Readiness Is Mechanism Evidence

- GIVEN complete diagnosis, development intervention, prospective fixture, controls, and raw reduction
- WHEN terminal readiness is classified
- THEN one certifies a causally effective fixture without scoring held-out learning value
- AND any failed mechanism gate produces a complete null with its specific stop reason.

## Implementation Status (REQ-CL-7267)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7267 and SCENARIO-CL-7267-* | Implemented: `python/carnot/experiment_7267_v639_recognition_prototype.py` provides the stable released-witness basis, active disagreement controller, two-stratum stream sealer, eight-arm fixture, diagnosis reducer, controls, and artifact builder. The script entrypoint remains thin. | Verified: RED-first basis, query, diagnosis, isolation, panel, transaction, terminal, command, and defensive tests pass with 902 of 902 scoped statements covered. |

## REQ-CL-7268: Prospective Learning With Autonomous Change Recognition

Carnot SHALL authenticate the sealed Exp7267 artifact before measurement. The
artifact SHALL have `recognition_fixture_ready_score=1`. Its controller code,
recognition contract, stream manifest, public events, release schedule, and
private authority bytes SHALL match their declared hashes. A missing, changed,
or quarantined prerequisite SHALL produce a row-free terminal blocked artifact.
The artifact SHALL name the failed check in `gate_check_summary`.

The learning run SHALL replay all eight frozen Exp7267 arms on all 24 sealed
prospective streams. All arms SHALL receive the same event order and label
delay. Each prediction SHALL be sealed before due feedback is released. Every
queried label SHALL count against the fixed limit of 128 labels per stream-arm.
The run SHALL retain exactly 196,608 event-arm rows. It SHALL retain query,
addition and deactivation, nomination, validation, commit, and reactivation
receipts. A supplied-change diagnostic MAY exist only as separate oracle rows.

The run SHALL use both twelve-stream strata in the overall result. It SHALL not
remove the overlapping stratum. It SHALL retain frozen, reset, active
recognition, random-query recognition, shuffled association, previous coverage,
full version-space memory, and feedback-withheld results. It SHALL report the
full-memory comparator separately. It SHALL not claim superiority over full
memory unless the full-memory paired confidence interval supports that claim.

The reducer SHALL use streams as independent units. It SHALL report future
error, false accepts, coverage, recurrence error, recurrence recovery delay,
retained constraints, and event cost. Cost rows SHALL include lookup, query,
update, validation, memory bytes, durable commit, and full event p50 and p95.
The reducer SHALL bootstrap whole streams with 10,000 fixed-seed resamples.

The frozen primary value gates SHALL require all of these conditions:

- The active-minus-reset future-error CI95 upper bound is less than zero.
- The active-minus-reset false-accept CI95 upper bound is at most zero.
- The active-minus-random recurrence-error CI95 upper bound is less than zero.
- The active-minus-shuffled recurrence-error CI95 upper bound is less than zero.
- The active-minus-frozen recurrence-error degradation is at most 0.02.
- At least one prospective query selection differs and a later prediction changes.
- No prediction changes before its requesting feedback release.
- No query or memory limit is exceeded.

`recognition_run_complete_score` SHALL equal one when all stream-arm units have
a complete or censored accounting and the raw evidence passes cold reduction.
`recognition_value_score` SHALL equal one only when every frozen value gate
passes. A completed run with any failed value gate SHALL be a terminal null.
The run SHALL not expand its stream count or label budget after a null result.

The E2E control SHALL process delayed feedback through an atomic durable commit.
It SHALL make a later prediction, cold-load the durable state, and prove reload
parity. It SHALL reject premature, duplicate, and stale writes without changing
state bytes. Measurement SHALL stop within 1,800 seconds. It SHALL checkpoint
after each stream and preserve unfinished units outside the terminal artifact.

The task SHALL use date `20260913`, `MODEL_SPECS=[]`, and
`model_invoked=false`. All current model load, generation, and answer counters
SHALL be zero. CPU replay SHALL declare `cpu_exact_solver_or_simulator` for both
substrate fields and `host` for the execution venue. A read-only reducer SHALL
declare `aggregation_from_upstream_artifacts` and class `aggregation`. The exact
evaluator SHALL set `verifier_is_oracle=true`. The controller SHALL not mutate
model weights. The result SHALL set `continuous_self_learning_task=true` only
for online constraint-state updates.

### SCENARIO-CL-7268-PRECONDITIONS: The Sealed Prototype Is Exact Or Blocked

- GIVEN the active Exp7268 task and the sealed Exp7267 evidence
- WHEN code, contract, stream, readiness, quarantine, and output checks run
- THEN only exact ready evidence can start the prospective replay
- AND an external failure produces a row-free blocked artifact.

### SCENARIO-CL-7268-PREQUENTIAL: Prediction Precedes Delayed Feedback

- GIVEN identical public events and release schedules for all eight arms
- WHEN each event enters the online replay
- THEN each arm seals its prediction before any due feedback commit
- AND each queried label counts once against the 128-label ceiling.

### SCENARIO-CL-7268-ROWS: Complete Evidence Retains Lifecycle Receipts

- GIVEN 24 streams, eight arms, and 1,024 events per stream
- WHEN the run completes without censoring
- THEN exactly 196,608 event-arm rows exist
- AND query, update, nomination, validation, commit, and reactivation receipts remain auditable.

### SCENARIO-CL-7268-METRICS: Both Strata And Full Memory Stay Visible

- GIVEN separated and overlapping recurrence streams
- WHEN stream-paired metrics are reduced
- THEN the overall rows include both strata and preserve stratum rows
- AND the full-memory comparator has its own performance and paired interval.

### SCENARIO-CL-7268-BOOTSTRAP: Frozen Gates Resample Whole Streams

- GIVEN stream-paired error and safety rates
- WHEN 10,000 fixed-seed bootstrap draws form each CI95
- THEN no event is treated as an independent statistical unit
- AND every frozen primary gate retains its expected, observed, and pass value.

### SCENARIO-CL-7268-COST: Event And Durable Costs Are Measured

- GIVEN the complete CPU event loop and durable E2E commit
- WHEN costs are reduced
- THEN lookup, query, update, validation, memory, durable commit, and full event costs exist
- AND full event cost includes measured p50 and p95 values.

### SCENARIO-CL-7268-E2E: Durable Restart Preserves Online State

- GIVEN a sealed prediction and delayed released feedback
- WHEN commit, later prediction, cold restart, rejection, and rollback run
- THEN only later predictions can observe committed state
- AND durable and in-memory bytes remain consistent.

### SCENARIO-CL-7268-TERMINAL: Completion Is Separate From Value

- GIVEN a fixed complete run and all validation evidence
- WHEN the frozen primary gates are scored
- THEN completion can equal one while value equals zero
- AND a failed scientific gate produces a complete null, never a partial result.

## Implementation Status (REQ-CL-7268)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7268 and SCENARIO-CL-7268-* | Implemented: `python/carnot/experiment_7268_v639_recognition_learning.py` authenticates the sealed prototype, checkpoints each eight-arm stream replay, retains lifecycle and cost receipts, bootstraps paired streams, and keeps completion separate from value. The script entrypoint remains thin. | Verified: RED-first seal, chronology, receipt, metric, bootstrap, cost, E2E, terminal, defensive, and command tests pass with 654 of 654 scoped statements covered. |

## REQ-CL-7269: Independent Recognition Causality And Memory Safety Audit

Carnot SHALL audit Exp7268 when its `recognition_run_complete_score` is one.
The audit SHALL not require a positive `recognition_value_score`. It SHALL
authenticate the exact producer artifact, raw event rows, lifecycle receipts,
sealed Exp7267 streams, controller code, exclusions, requirements, and output
ownership before reduction. An absent, changed, malformed, or quarantined
external prerequisite SHALL produce a row-free terminal blocked artifact with
the exact failed check in `gate_check_summary`.

A cold process SHALL reconstruct all 192 stream-arm summaries from the 196,608
raw event rows. It SHALL rebuild all frozen Exp7268 comparisons, confidence
intervals, science gates, label budgets, byte caps, retained constraints,
false accepts, overlap recall, intervention joins, and recurrence measures. A
missing field or source SHALL fail validation and SHALL not become a measured
zero.

The audit SHALL replay active, random, and shuffled query choices from public
events and then-released feedback only. It SHALL verify prediction seals before
release. It SHALL require prospective active-versus-random query changes and
join changed choice identity to later changed predictions. It SHALL retain
zero-effect rows and both recurrence strata.

Five isolated mutation controls SHALL inject a future label, a private regime
ID, a duplicate release, a stale parent, and a corrupted archive. Every attack
SHALL be rejected before state publication. Rejection SHALL preserve exact
prior in-memory and durable bytes. Historical or injected model metadata SHALL
remain in a hashed sidecar outside current invocation fields.

The E2E-007 adaptation SHALL seal a prediction before release, accept delayed
feedback with authenticated changed state, reject invalid writes without byte
changes, cold-load with decision parity, and roll back to exact parent bytes.
These checks audit the new finite recognition memory. They SHALL not claim that
Exp1659 ran again.

`recognition_audit_complete_score` SHALL equal one after complete independent
reduction, replay, mutation, and lifecycle review. `recognition_promotion_score`
SHALL equal one only if every reconstructed Exp7268 science gate and every audit
safety gate passes. A complete audit of a null learner SHALL remain complete,
use `verdict_class=null`, and keep promotion zero.

The task SHALL use date `20260913`, `MODEL_SPECS=[]`, `model_invoked=false`, and
zero current load, generation, inference, and answer counters. CPU checks SHALL
use `cpu_exact_solver_or_simulator` for both substrate fields. The raw reducer
SHALL declare `aggregation_from_upstream_artifacts` and class `aggregation`.
The execution venue SHALL be `host`. The exact evaluator SHALL set
`verifier_is_oracle=true`. The task SHALL not change model weights, production
defaults, repository roadmap state, or external systems.

### SCENARIO-CL-7269-PRECONDITIONS: Complete Recognition Run Or Block

- GIVEN the active Exp7269 task and exact Exp7268 terminal and sidecar bytes
- WHEN status, completion, checksums, hashes, quarantine, imports, and output ownership are checked
- THEN `recognition_run_complete_score=1` permits the audit even when value is zero
- AND an external failure produces a row-free blocked artifact with its exact observed value.

### SCENARIO-CL-7269-REDUCTION: Raw Rows Reconstruct The Full Matrix

- GIVEN 196,608 authenticated prequential rows from 24 streams and eight arms
- WHEN a cold process reduces the evidence without producer aggregates
- THEN all 192 stream-arm rows, comparisons, intervals, and frozen gates match
- AND a missing field fails instead of becoming a zero effect.

### SCENARIO-CL-7269-REPLAY: Only Released Data Drives Recognition

- GIVEN sealed public events, delayed releases, and private authority bytes
- WHEN selected queries and archive associations are replayed
- THEN no future label or private regime field enters a decision
- AND changed prospective choices join to later changed predictions.

### SCENARIO-CL-7269-BOUNDS: Labels, Bytes, Constraints, Safety, And Overlap Stay Visible

- GIVEN every stream-arm row in both recurrence strata
- WHEN the audit recomputes resource and outcome metrics
- THEN label ceilings, byte caps, retained constraints, false accepts, and overlap recall are explicit
- AND zero-headroom or zero-effect rows stay in their original denominators.

### SCENARIO-CL-7269-MUTATIONS: Invalid Inputs Preserve Prior Bytes

- GIVEN future-label, regime-ID, duplicate-release, stale-parent, and corrupted-archive attacks
- WHEN each isolated mutation reaches the recognition transaction boundary
- THEN each named attack is rejected
- AND in-memory and durable bytes remain equal to their exact parent bytes.

### SCENARIO-CL-7269-E2E: Accepted State Survives Reload And Rollback

- GIVEN a prediction sealed before delayed feedback
- WHEN a valid commit, invalid commit, cold reload, later decision, and rollback execute
- THEN accepted state changes are authenticated and reload decisions match
- AND rejection and rollback preserve the required exact bytes.

### SCENARIO-CL-7269-TERMINAL: Audit Completion Does Not Depend On Accuracy

- GIVEN a complete independent audit and reconstructed safety and science gates
- WHEN one or more science gates fail
- THEN `recognition_audit_complete_score=1` and `recognition_promotion_score=0`
- AND the terminal result is `complete_null`, never partial or blocked.

## Implementation Status (REQ-CL-7269)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CL-7269 and SCENARIO-CL-7269-* | Implemented: `python/carnot/experiment_7269_v639_recognition_audit.py` authenticates complete Exp7268 evidence independently of its value score, reconstructs raw metrics in a cold process, replays released-only recognition, runs five byte-preserving mutations and E2E lifecycle checks, and seals terminal evidence through a thin wrapper. | Verified: RED-first precondition, reduction, replay, bound, mutation, lifecycle, null-completion, command, and defensive tests pass with 713 of 713 scoped statements covered. |
