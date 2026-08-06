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
