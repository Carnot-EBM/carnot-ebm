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
