#!/usr/bin/env python3
"""Adversarial-verify pass for experiment artifacts.

Detects patterns that indicate fabrication, tautology, methodology
gaps, or statistically-unsupported claims. Designed to run after every
experiment lands an artifact, AND as a sweep-pass over historical
results to flag earlier artifacts that slipped through.

Detection rules (each emits a discrete flag with severity):

  TAUTOLOGY
    Two distinct numerical metrics with floating-point values that
    agree to more than 5 significant figures. Example: exp1938 reported
    nrgpt_grad_norm == ce_grad_norm to 10+ digits — two different
    loss functions cannot share gradient norms by coincidence.

  IMPLAUSIBLE_PERFECT
    A metric that should be in [0, 1] reaching exactly 0.0 or 1.0,
    OR an error metric reaching exactly 0.0 with non-trivial sample
    size. Examples: exp1851 NLA probe reported TPR=1.0 with 10
    adversarial examples; this is the "fabricated" signature.

  SIGN_ANOMALY
    Optimization that claims "minimization" but final_value > initial,
    or vice versa for maximization. Example: exp1941 EBT bridge
    reported initial_energy -0.40 -> final_energy -0.10 (energy went
    up) while claiming success.

  DURATION_TOO_SHORT
    Artifact claims to invoke a live SOTA model (any GGUF in
    target_model / model_specs / responses) but duration_s < 60s.
    Loading and running a 30B+ GGUF takes minutes; if the artifact
    completed faster, the model was not actually invoked.

  SAMPLE_SIZE_BELOW_CLAIM
    Artifact reports distributional metrics (KL divergence, KS test,
    mean/variance estimates, etc.) but n_samples is below the
    statistical threshold needed for the claim type. Example:
    exp1850 reported KL=0.278 at n_spins=128 with only 100 samples;
    100 samples on a 2^128 state space cannot estimate KL meaningfully.

  GATE_PASSED_WITHOUT_DATA
    acceptance_gate_passed=true but key numerical fields referenced
    in the gate are null, missing, or zero. The agent set the gate
    flag without populating the supporting data.

  METHODOLOGY_MISSING
    Compute-bound artifact (claims model invocation, GPU, hardware)
    but lacks model_specs, random_seed, OR reproducibility_checksum.
    Methodology is unverifiable from the artifact alone.

  IMPLAUSIBLE_TIGHT_CI
    A bootstrap or empirical confidence interval whose width is far
    smaller than 1/sqrt(N) on the metric's natural scale. Real
    stochastic measurements at finite N have a minimum CI width set
    by sample variance; CIs much tighter than that signal either
    (a) the metric is deterministic-by-construction (an invariant
    rather than a measurement — should be disclosed in
    methodology_note), or (b) the bootstrap used 1 seed reshuffled
    instead of N independent draws. Example: exp1693 reported
    delta_alpha=0.15054 with bootstrap_ci_95=[0.15040, 0.15070]
    (width 3e-4) at n_seeds=30. Sample-variance-floor for a
    metric near 0.15 at N=30 is ~0.18/sqrt(30) ~ 0.03 — the
    observed CI is 100x tighter than the floor.

  DEGENERATE_SEPARATION
    A selector-vs-vote transfer artifact with a near-perfect delta
    while vote or matched control is near zero, OR a perfect
    set-encoder selector with oracle@K saturated at 1.0. Example:
    exp4282 reported cross_family_delta=1.0, vote_at_1=0.0,
    set_encoder_at_1=1.0, and oracle_at_k=1.0 on a wrong-majority
    ARC-GEN pool; that is a pool-construction artifact, not transfer.

  DEGENERATE_CONTROLS
    A condition-arm artifact whose distinct control arms report
    bit-identical accuracy/pass-rate values. Example: exp4293 reported
    rfg == unguided == entrgi in condition_accuracy, which is the
    signature of no-op or aliased controls rather than differentiated
    baselines.

Output: a JSON report listing flagged artifacts with per-flag details.
Exit code: 0 if no flags, 1 if any flags present.

Usage:

    # Single artifact
    python scripts/adversarial_verify.py results/experiment_1938_*.json

    # Sweep all artifacts from a milestone range
    python scripts/adversarial_verify.py --milestone-range 1900 2000

    # All .149-era artifacts (after the structural fixes landed)
    python scripts/adversarial_verify.py --milestone-range 1900 1950
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from carnot.terminal_artifacts import (
    TerminalClassification,
    classify_artifact_path,
    classify_artifact_payload,
    path_sha256,
)

# Floating-point agreement threshold for tautology detection. Two
# distinct metrics agreeing to 5 significant figures is suspicious;
# legitimate cases (two implementations of the same function with the
# same seed) should be deliberate and documented.
TAUTOLOGY_DIGITS = 5
NONTERMINAL_FLAG_KIND = "NONTERMINAL_DECLARED_ARTIFACT"
DECLARED_EXPERIMENT_ARTIFACT_RE = re.compile(r"experiment_\d+_.*\.json$")

DEGENERATE_DELTA_THRESHOLD = 0.95
DEGENERATE_BASELINE_THRESHOLD = 0.05
DEGENERATE_DELTA_KEYS = ("cross_generator_delta", "cross_family_delta")
DEGENERATE_BASELINE_KEYS = ("vote_at_1", "matched_control_at_1")
CONDITION_ARM_MAP_KEYS = ("condition_accuracy", "per_condition_accuracy", "arms")
CONDITION_ARM_METRIC_KEYS = (
    "condition_accuracy",
    "per_condition_accuracy",
    "accuracy",
    "pass_rate",
    "success_rate",
    "solve_rate",
    "score",
)
CONTROL_ARM_MARKERS = (
    "baseline",
    "control",
    "entrgi",
    "no_guidance",
    "no-guidance",
    "random",
    "rfg",
    "self_guidance",
    "self-guidance",
    "unguided",
    "unconditioned",
)
PLACEBO_OR_REPLICATE_MARKERS = ("placebo", "replicate", "replica", "duplicate")

# Compute-bound markers — if the artifact mentions any of these, the
# experiment was supposed to invoke real hardware/model work.
COMPUTE_BOUND_MARKERS = (
    "unsloth/",
    "Qwen3.6-",
    "Qwen3.5-",
    "Qwen1.5-",  # 0.5B fallback — also compute-bound
    "gemma-4-",
    "GGUF",
    "DualGPURunner",
    "DualGPUHarness",
    "llama.cpp",
    "torch",
    "torch.cuda",
    ".cuda(",
)

# Minimum duration (seconds) for a compute-bound artifact. Loading +
# inference on even a 0.5B GGUF takes at least this long.
COMPUTE_BOUND_MIN_DURATION_S = 60.0
LIVE_LLM_SUBSTRATE = "live_llm_inference"

# Verifier-scoring artifacts: experiments that score an ensemble of
# verifiers against already-cached (input, candidate, label) triples
# WITHOUT invoking a fresh LLM forward pass. The GGUF markers in their
# model_specs are vestigial template declarations -- the task spec
# mandated naming the SOTA model, but the actual run does not load it.
# These experiments finish in seconds (verifier scoring of a few
# hundred candidates), not minutes, so the standard
# COMPUTE_BOUND_MIN_DURATION_S threshold produces false positives.
#
# Recognition is by either:
#   (a) explicit `inference_substrate` field set to the canonical
#       sentinel below (preferred -- forward-only), OR
#   (b) one of the known verifier-scoring schema prefixes (legacy
#       -- covers artifacts that pre-date the explicit field).
VERIFIER_SCORING_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
VERIFIER_SCORING_SCHEMA_PREFIXES = (
    "carnot.fover_memory_leakage_",
    "carnot.cross_corpus_verifier_matrix",
    "carnot.mbpp_dual_condition_v",
    "carnot.halueval_fever_pilot",
    "carnot.mbpp_humaneval_generated_code_clean_row",
)
VERIFIER_SCORING_MIN_DURATION_S = 1.0
CHEAP_LEARNED_VALUE_MIN_DURATION_S = 0.0001
CHEAP_LEARNED_VALUE_MARKERS = (
    "value_head",
    "value-head",
    "value head",
    "spatialvaluenet",
    "learnedvaluenet",
    "linear forward",
    "linear_forward",
    "linear value",
    "cnn",
    "forward pass",
    "cached_candidate_linear_forward_pass",
)

# Aggregation-only artifacts: milestone capstones, archive/activate
# transitions, paper-table synthesizers, cross-corpus-matrix builders.
# These don't invoke the model OR score verifiers -- they read existing
# artifacts off disk, compute deltas / format tables / build manifests,
# and write the result. Their wall-clock measures JSON-read + arithmetic,
# legitimately a few milliseconds. The GGUF markers in their model_specs
# (when present) are forwarded from the upstream artifacts they cite.
AGGREGATION_SUBSTRATE = "aggregation_from_upstream_artifacts"
AGGREGATION_SCHEMA_PREFIXES = (
    "capstone_v",
    "carnot.milestone_capstone",
    "carnot.paper_v6_capstone",
    "carnot.archive_activation",
    "carnot.paper_v6_multicorpus_table",
)
AGGREGATION_MIN_DURATION_S = 0.0001  # 100us floor catches truly-zero/missing

# Explicit substrate classes. These are deliberately top-level declarations:
# quoted upstream receipts can contain GGUF/CUDA strings without changing what
# THIS artifact executed. Unknown or missing declarations still fall through to
# the conservative compute-marker scan.
SUBSTRATE_KIND_AGGREGATION = "aggregation"
SUBSTRATE_KIND_NO_LLM = "no_llm"
SUBSTRATE_KIND_LIVE_MODEL = "live_model"
SUBSTRATE_KIND_UNKNOWN = "unknown"

AGGREGATION_SUBSTRATE_ALIASES = (  # pragma: no cover - declarative allowlist
    AGGREGATION_SUBSTRATE,
    "aggregation_from_exact_declared_artifacts",
    "aggregation_from_upstream_artifacts_no_llm",
    "aggregation_from_external_primary_sources",
    "aggregation_from_exp5603_exp5611_artifacts",
    "artifact_reconciliation_and_validation_only",
    "artifact_reconciliation_only",
    "artifact_synthesis_and_gate_reconciliation",
    "cached_artifact_reconciliation_no_llm",
    "capstone_aggregation_from_upstream_artifacts",
    "exact_local_artifact_conductor_gate_and_document_reconciliation_no_llm",
    "local_artifact_aggregation_and_doc_reconcile",
)

# Deterministic verifier artifacts replay or aggregate checked-in candidate
# evidence. They can honestly cite upstream GGUF model names while making no
# new LLM call of their own, so they need a JSON-work duration floor rather
# than the live-model floor.
DETERMINISTIC_VERIFIER_SUBSTRATES = (
    "deterministic_verifier",
    "deterministic_verifier_plus_replay",
    "deterministic_exact_verifier_and_versioned_external_state_no_llm",
    "artifact_provenance_audit",
    "post_marker_source_ingestion_and_roadmap_scope_audit",
    "post_marker_source_ingestion_and_v539_scope_freeze",
    "cached_sota_event_energy_calibration",
    "cached_authentic_sota_rows_cpu_analysis",
    "deterministic_pipeline_integration_no_llm",
    "sealed_cached_event_evaluation",
    "transactional_exact_conflict_memory_and_cpu_mapping_no_llm",
)
DETERMINISTIC_VERIFIER_MIN_DURATION_S = 0.0001
NO_LLM_DECLARED_MIN_DURATION_S = 0.0001

# ARC live-agent runtime self-discovery artifacts: the live agent takes real
# actions against the offline arcade / a live ARC env WITHOUT invoking an LLM
# (pure Python env-stepping, world-model/verifier scoring, go-explore archive
# bookkeeping -- no GGUF load, no CUDA). Their model_specs often name the
# generator that WOULD be used if the LLM tier fired (invoked=false) purely
# for template/provenance completeness; those GGUF strings are vestigial, not
# a claim of invocation. Discovered 2026-06-30 (exp5054 tu93 go-explore
# attempt, 35 actions / 2.9s ~= 83ms/action -- plausible for pure Python env
# stepping, wrongly floored at 60s because the substrate wasn't recognized
# and fell through to the generic GGUF-marker scan). Per CLAUDE.md
# "Inference-Substrate Declaration Discipline" -- the forward-only 5th legal
# value.
ARC_LIVE_AGENT_NO_LLM_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
ARC_LIVE_AGENT_ENV_INTERACTION_SUBSTRATE = "live_agent_environment_interaction"
ARC_LIVE_E3_ADAPTER_DISABLED_RUNTIME_TRANSITIONS_SUBSTRATE = (
    "live_e3_adapter_disabled_runtime_transitions"
)
ARC_SUBMITTED_KERNEL_OFFLINE_FROZEN_POLICY_REPLAY_SUBSTRATE = (
    "submitted_live_agent_kernel_acquisition_plus_offline_frozen_policy_replay"
)
ARC_FILTER_RUNTIME_NO_LLM_SUBSTRATE = "offline_arcade_live_agent_runtime_filters_no_new_llm"
ARC_LIVE_AGENT_NO_LLM_MIN_DURATION_S = (
    0.01  # 10ms/action-scale floor; still nonzero-fabrication-proof
)

# LLM embedding-extraction artifacts: the task loads a real local GGUF model and
# extracts hidden-state / final-token embedding vectors (llama_cpp.Llama(embedding=True)
# or equivalent) for a small candidate set, WITHOUT full autoregressive generation. This
# is real compute -- a genuine model load + real forward passes -- but each forward pass
# is a single pass with no iterative token-by-token decoding loop, so it is far cheaper
# than LIVE_LLM_SUBSTRATE's generative-inference floor while still needing meaningfully
# more than VERIFIER_SCORING_SUBSTRATE's "does not load the LLM at all" floor. Discovered
# 2026-07-03 (exp5178's hidden-state verifier pilot: gemma-4-26B-A4B GGUF, 48 candidates,
# 35.28s total -- plausible for model load + 48 single-pass embeddings, implausible under
# the generic 60s live_llm_inference floor calibrated for full generation). Per CLAUDE.md
# "Inference-Substrate Declaration Discipline" -- the forward-only 6th legal value.
LLM_EMBEDDING_EXTRACTION_SUBSTRATE = "live_llm_embedding_extraction"
LOCAL_SOTA_FIXED_SEQUENCE_REPRESENTATION_SUBSTRATE = (
    "live_local_sota_gguf_fixed_sequence_representation"
)
LLM_EMBEDDING_EXTRACTION_MIN_DURATION_S = (
    2.0  # model load + >=1 single-pass embedding, even on the smallest SOTA GGUF
)

# Log-analysis-plus-bounded-local-timing artifacts: a "should we build X" gate that reads a
# handful of existing JSON logs/artifacts (no LLM call) and runs one small, LOCAL, non-model
# timing measurement (e.g. a tiny CUDA kernel dispatch to sanity-check an amortization estimate,
# clamped to a conservative floor rather than trusted raw). Genuine compute -- reads real files,
# dispatches a real (if microsecond-scale) local operation -- but far cheaper than either full
# generation or embedding extraction, and the presence of a CUDA/GPU backend string in the timing
# evidence (e.g. "torch_cuda:0") should not pull this back to the 60s live_llm_inference floor.
# Discovered 2026-07-04 (exp5215's ARC PAW amortization gate: read 9 arc_loop_solve_*.json logs +
# one local-GPU micro-timing check clamped to a 2.13s conservative floor, 4.10s total -- correctly
# concluded paw_amortization_viable=False, but flagged DURATION_TOO_SHORT under the generic 60s
# floor because no substrate category recognized this shape). Per CLAUDE.md "Inference-Substrate
# Declaration Discipline" -- the forward-only 7th legal value.
LOG_ANALYSIS_LOCAL_TIMING_SUBSTRATE = "arc_log_analysis_plus_local_timing"
LOG_ANALYSIS_LOCAL_TIMING_MIN_DURATION_S = (
    1.0  # reading several JSON logs + one bounded local timing check, no model load
)

# Small-N local-SOTA-GGUF pilots: a handful (single-digit) of short generation calls
# (structured JSON constraint extraction, not long-form prose) OR a handful of
# single-forward-pass logit/logprob probes over a bounded fixture set -- real model
# load + real compute, but far lighter than the full multi-hundred-token generation
# runs the generic 60s live_llm_inference floor is calibrated for. Discovered
# 2026-07-05: exp5262 (4 short constraint-generation fixtures, 56.3s) and exp5263 (6
# logit-extraction forward passes, no generation, 23.7s) both declared this substrate
# and were flagged DURATION_TOO_SHORT under the generic 60s floor -- both are honest,
# non-fabricated null/negative results (validity=0.25 vs baseline 0.5; signal_delta
# 0.0048), not implausibly-fast claims. Floor set below both real observations but
# well above what a genuine local-GGUF model load could plausibly complete in.
LOCAL_SOTA_GGUF_SMALL_N_SUBSTRATE = "live_llm_inference_local_gguf_sota"
LOCAL_LLAMA_CPP_GGUF_ATOMIC_FACTOR_PROPOSALS_SUBSTRATE = (
    "local_llama_cpp_gguf_atomic_factor_proposals"
)
LOCAL_LLAMA_CPP_THREE_FAMILY_FORMAL_MUTATION_SUBSTRATE = (
    "local_llama_cpp_three_family_formal_mutation_generation_plus_exact_solver_admission"
)
LOCAL_SOTA_GGUF_SMALL_N_MIN_DURATION_S = 10.0

# Deterministic solver/formal-checker validation (SMT/Z3-style hint acceptance, constraint
# checking) that explicitly declares itself LLM-free in its own substrate name. No model load,
# no GPU, just a fast deterministic check -- a near-zero floor is appropriate, same reasoning as
# aggregation_from_upstream_artifacts and artifact_qa_lint_tests. Discovered 2026-07-06 (exp5318,
# duration_s=0.03, flagged under the generic 60s live_llm_inference floor despite its own substrate
# name declaring "no_llm").
DETERMINISTIC_SMT_HINT_VALIDATION_SUBSTRATE = "deterministic_smt_hint_validation_no_llm"
DETERMINISTIC_SMT_HINT_VALIDATION_MIN_DURATION_S = 0.0001

# Native-llama.cpp-CLI backend/flag bisect: probes a handful of backend-variant + flag
# combinations against the mandated SOTA models to find one that completes load + first-token +
# bounded generation without hanging or timing out. Real GPU-backed model load(s) happen, but the
# bisect stops as soon as ONE working combination is confirmed rather than running a full
# generation benchmark across all variants -- genuinely faster than the 60s live_llm_inference
# floor, similar reasoning to the small-N SOTA GGUF pilot substrate. Discovered 2026-07-06
# (exp5323, duration_s=20.8, found the working `-st`/`--single-turn` variant this project's own
# CLAUDE.md documents -- see the native-llama-cli-hang fix -- and confirmed
# completed_load_first_token_and_8_tokens=True with authenticated GPU offload).
NATIVE_GGUF_BACKEND_BISECT_SUBSTRATE = "local_native_llama_cpp_gguf_backend_bisect"
NATIVE_GGUF_BACKEND_BISECT_MIN_DURATION_S = 5.0

# Source-delta/SOTA-ingestion artifacts (per CLAUDE.md's SOTA-Ingestion Cycle Discipline) do
# WebSearch/WebFetch + read research-references.md; no model is loaded, no GPU touched. Genuinely
# fast (sub-second) but the literature they ingest routinely discusses GGUF/CUDA/model names,
# tripping the generic compute-bound-marker detector under the 60s live_llm_inference floor.
# Discovered 2026-07-20 (exp5718/exp5732 v511/v512 source-delta-ingestion artifacts, duration_s
# 2.7e-07 and 0.05, both flagged DURATION_TOO_SHORT despite declaring
# inference_substrate=web_and_bibliographic_search_only -- a value the linter did not yet
# recognize -- while two sibling v510/v513 ingestion artifacts with the same substrate value but
# content that didn't trip the marker detector passed clean, confirming the false-positive was
# substrate-recognition-gap-driven, not artifact-specific).
WEB_BIBLIOGRAPHIC_SEARCH_ONLY_SUBSTRATE = "web_and_bibliographic_search_only"
SOURCE_RECEIPTS_METHOD_PREREGISTRATION_NO_LLM_SUBSTRATE = (
    "source_receipts_and_local_method_preregistration_no_llm"
)
LOW_CONCURRENCY_PRIMARY_SOURCE_INGESTION_NO_EXPERIMENTAL_LLM_SUBSTRATE = (
    "low_concurrency_primary_source_ingestion_no_experimental_llm"
)
WEB_BIBLIOGRAPHIC_SEARCH_ONLY_MIN_DURATION_S = 0.0001

# Artifact-QA lint-test artifacts intentionally embed verifier fixture reports,
# including negative controls whose details mention GGUF/CUDA/live-model markers.
# The artifact itself runs JSON lint tests, not model inference, so exact
# substrate recognition is needed to avoid re-flagging the calibration receipt
# for the fixture text it is required to preserve. This does not weaken real
# compute-bound artifacts; they do not declare this substrate.
ARTIFACT_QA_LINT_TESTS_SUBSTRATE = "artifact_qa_lint_tests"
ARTIFACT_QA_LINT_TESTS_MIN_DURATION_S = 0.0001
DETERMINISTIC_QA_REGRESSION_NO_LLM_SUBSTRATE = "deterministic_qa_regression_no_llm"
DETERMINISTIC_RUNTIME_RECEIPT_VALIDATION_NO_LLM_SUBSTRATE = (
    "deterministic_runtime_receipt_validation_no_llm"
)
EXACT_SOLVER_DATASET_COMMITMENT_NO_LLM_SUBSTRATE = "exact_solver_dataset_commitment_no_llm"
PROCEDURAL_FORMAL_EXACT_SOLVER_LABELS_NO_LLM_SUBSTRATE = (
    "procedural_formal_instances_and_exact_solver_labels_no_llm"
)
EXACT_SOLVER_TRAJECTORY_RECORDING_NO_LLM_SUBSTRATE = "exact_solver_trajectory_recording_no_llm"
PROCEDURAL_EXACT_SOLVER_STRUCTURAL_CONTROLS_NO_LLM_SUBSTRATE = (
    "procedural_exact_solver_structural_controls_no_llm"
)
INDEPENDENT_STRUCTURAL_CONTROL_ROW_AND_SOLVER_RECEIPT_REPLAY_NO_LLM_SUBSTRATE = (
    "independent_structural_control_row_and_solver_receipt_replay_no_llm"
)
LOCAL_COMPACT_ENERGY_HEADS_NO_LLM_SUBSTRATE = (
    "local_compact_energy_heads_on_exact_solver_features_no_llm"
)
LOCAL_COMPACT_ROUTER_EXCEPTION_TABLE_NO_LLM_SUBSTRATE = (
    "local_compact_router_plus_exact_exception_table_and_native_solver_no_llm"
)
DETERMINISTIC_REPRESENTATION_CONTRACT_NO_LLM_SUBSTRATE = (
    "deterministic_representation_contract_no_llm"
)
DETERMINISTIC_TRANSITION_CONTRACT_NO_LLM_SUBSTRATE = "deterministic_transition_contract_no_llm"
DETERMINISTIC_FACTOR_POOL_CONTROLLER_NO_LLM_SUBSTRATE = (
    "deterministic_anytime_factor_pool_controller_no_llm"
)
DETERMINISTIC_FACTOR_POOL_SUPPORT_STRESS_NO_LLM_SUBSTRATE = (
    "deterministic_factor_pool_stress_with_exact_evaluation_no_llm"
)

NO_LLM_SUBSTRATE_ALIASES = (  # pragma: no cover - declarative allowlist
    VERIFIER_SCORING_SUBSTRATE,
    *DETERMINISTIC_VERIFIER_SUBSTRATES,
    ARC_LIVE_AGENT_NO_LLM_SUBSTRATE,
    ARC_LIVE_AGENT_ENV_INTERACTION_SUBSTRATE,
    ARC_LIVE_E3_ADAPTER_DISABLED_RUNTIME_TRANSITIONS_SUBSTRATE,
    ARC_SUBMITTED_KERNEL_OFFLINE_FROZEN_POLICY_REPLAY_SUBSTRATE,
    ARC_FILTER_RUNTIME_NO_LLM_SUBSTRATE,
    LOG_ANALYSIS_LOCAL_TIMING_SUBSTRATE,
    WEB_BIBLIOGRAPHIC_SEARCH_ONLY_SUBSTRATE,
    SOURCE_RECEIPTS_METHOD_PREREGISTRATION_NO_LLM_SUBSTRATE,
    LOW_CONCURRENCY_PRIMARY_SOURCE_INGESTION_NO_EXPERIMENTAL_LLM_SUBSTRATE,
    ARTIFACT_QA_LINT_TESTS_SUBSTRATE,
    DETERMINISTIC_QA_REGRESSION_NO_LLM_SUBSTRATE,
    DETERMINISTIC_RUNTIME_RECEIPT_VALIDATION_NO_LLM_SUBSTRATE,
    EXACT_SOLVER_DATASET_COMMITMENT_NO_LLM_SUBSTRATE,
    PROCEDURAL_FORMAL_EXACT_SOLVER_LABELS_NO_LLM_SUBSTRATE,
    EXACT_SOLVER_TRAJECTORY_RECORDING_NO_LLM_SUBSTRATE,
    PROCEDURAL_EXACT_SOLVER_STRUCTURAL_CONTROLS_NO_LLM_SUBSTRATE,
    INDEPENDENT_STRUCTURAL_CONTROL_ROW_AND_SOLVER_RECEIPT_REPLAY_NO_LLM_SUBSTRATE,
    LOCAL_COMPACT_ENERGY_HEADS_NO_LLM_SUBSTRATE,
    LOCAL_COMPACT_ROUTER_EXCEPTION_TABLE_NO_LLM_SUBSTRATE,
    DETERMINISTIC_REPRESENTATION_CONTRACT_NO_LLM_SUBSTRATE,
    DETERMINISTIC_TRANSITION_CONTRACT_NO_LLM_SUBSTRATE,
    DETERMINISTIC_FACTOR_POOL_CONTROLLER_NO_LLM_SUBSTRATE,
    DETERMINISTIC_FACTOR_POOL_SUPPORT_STRESS_NO_LLM_SUBSTRATE,
    "chronological_exact_admitted_factor_learning_no_new_llm",
    DETERMINISTIC_SMT_HINT_VALIDATION_SUBSTRATE,
    "artifact_reducer_no_llm",
    "cached_exp5567_responses_no_llm",
    "cached_fixture_replay_no_llm",
    "cached_sota_event_energy_calibration",
    "deterministic_automaton_no_llm",
    "deterministic_validation_over_canonical_pool",
    "independent_raw_representation_audit_no_llm",
    "offline_exact_forensics_over_existing_real_gguf_rows_no_llm",
    "online_exact_membership_query_sidecar_no_llm",
    "primary_source_metadata_and_local_ledger_synthesis_no_experiment_llm",
    "simulation",
)

LIVE_MODEL_SUBSTRATE_ALIASES = (  # pragma: no cover - declarative allowlist
    LIVE_LLM_SUBSTRATE,
    LLM_EMBEDDING_EXTRACTION_SUBSTRATE,
    LOCAL_SOTA_FIXED_SEQUENCE_REPRESENTATION_SUBSTRATE,
    LOCAL_SOTA_GGUF_SMALL_N_SUBSTRATE,
    LOCAL_LLAMA_CPP_GGUF_ATOMIC_FACTOR_PROPOSALS_SUBSTRATE,
    LOCAL_LLAMA_CPP_THREE_FAMILY_FORMAL_MUTATION_SUBSTRATE,
    NATIVE_GGUF_BACKEND_BISECT_SUBSTRATE,
    "actual_capability_bound_adapter_disabled_e3_local_mandated_gguf_public_llama_cpp_cuda",
    "exact_validated_local_sota_gguf_panel",
    "live_llm_generation_with_deterministic_execution_guard",
    "live_llm_hidden_state_extraction",
    "live_llm_inference_and_row_reanalysis",
    "live_llm_inference_plus_z3",
    "live_llm_inference_with_frozen_policy_state",
    "live_llm_internal_telemetry_local_gguf_sota",
    "live_local_sota_gguf_cross_model_csl",
    "live_local_sota_gguf_cross_model_csl_transfer_or_gate_skip",
    "live_local_sota_gguf_exact_validated_panel",
    "live_local_sota_reset_free_exact_feedback_harness",
    "local_gguf_llamacpp_cuda",
    "local_gguf_llamacpp_cuda_evidence_fixture",
    "local_llama_cpp_python_cuda_gguf",
    "local_llama_cpp_python_cuda_gguf_diagnostic",
    "local_llama_cpp_python_cuda_gguf_exact_proposal_stream",
    "local_llama_cpp_python_cuda_gguf_finite_choice_proposals",
    "local_llama_cpp_python_cuda_gguf_plus_exact_validators",
    "local_llama_cpp_gguf_runtime",
    "local_mandated_gguf_public_llama_cpp_cuda_schema_supported_decoding",
    "local_native_llama_cpp_stability_receipts",
    "local_sota_gguf_bounded_smoke",
    "local_sota_gguf_constrained_generation_or_verified_repair",
    "local_sota_gguf_csl_memory_panel",
    "local_sota_gguf_llama_cpp_or_blocked",
    "local_sota_gguf_plus_deterministic_solver_feedback",
    "public_llama_cpp_cuda_tokenizer_bridge_smoke",
    "real_local_llama_cpp_cuda_gguf_generation_plus_exact_validation",
    "real_local_llama_cpp_cuda_gguf_generation_plus_exact_z3_validation",
    "structured_output_fixture_or_live_llm_smoke",
    "structured_output_fixture_plus_live_llm_smoke",
    "structured_output_repair_fixture_plus_live_llm_smoke",
)

# Offline ARC solve / learned-verifier artifacts do not have a model to name:
# their methodology is the solver entrypoint, reproduce gate/checksum, and
# learned-verifier checkpoint. Treat those fields as the methodology descriptor
# only when the substrate already says this is cached-candidate verifier work or
# upstream aggregation, never for live LLM inference.
OFFLINE_ARC_METHOD_DESCRIPTOR_KEYS = (
    "offline_reproduced",
    "primitive_persisted",
    "solver_module",
    "solver_modules",
    "reproduction_gate",
    "reproduce_gate",
    "reproduce_gate_checksum",
    "verifier_checkpoint",
    "verifier_checkpoints",
    "learned_verifier_checkpoint",
)
OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS = frozenset(
    {
        "field_principles",
        "required_artifact_fields",
        "tests_added_pass",
    }
)


class Flag:
    """A single detected concern on an artifact."""

    SEVERITY_RANK = {"info": 0, "warn": 1, "critical": 2}

    def __init__(self, kind: str, severity: str, detail: str) -> None:
        self.kind = kind
        self.severity = severity
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "severity": self.severity, "detail": self.detail}


def _declared_experiment_artifact_path(path: Path) -> bool:
    return bool(DECLARED_EXPERIMENT_ARTIFACT_RE.match(path.name))


def _declares_terminal_artifact_readiness(payload: Any) -> bool:
    return isinstance(payload, dict) and "status" in payload


def _should_check_terminal_artifact_readiness(
    path: Path,
    payload: Any = None,
    *,
    declared: bool | None = None,
) -> bool:
    if not _declared_experiment_artifact_path(path):
        return False
    if declared is not None:
        return declared
    return _declares_terminal_artifact_readiness(payload)


def check_terminal_artifact_readiness(
    classification: TerminalClassification,
    flags: list[Flag],
) -> None:
    """Flag declared artifacts whose exact path has no terminal state."""

    if classification.terminal:
        return
    flags.append(
        Flag(
            kind=NONTERMINAL_FLAG_KIND,
            severity="critical",
            detail=(
                f"declared artifact is nonterminal: path={classification.path}; "
                f"classification={classification.classification}; reason={classification.reason}"
            ),
        )
    )


def _flag_summary(flags: list[Flag]) -> dict[str, Any]:
    return {
        "flag_count": len(flags),
        "max_severity": max((Flag.SEVERITY_RANK[f.severity] for f in flags), default=-1),
        "flags": [f.to_dict() for f in flags],
    }


def _is_finite_number(v: Any) -> bool:
    """True if v is a real, finite numeric value."""
    if isinstance(v, bool):
        # bool is a subclass of int; explicitly reject so True/False
        # don't trigger tautology comparison.
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(float(v))
    return False


def _significant_digits_match(a: float, b: float, digits: int) -> bool:
    """Return True if a and b agree to at least `digits` significant figures.

    Handles both small and large values via relative tolerance.
    """
    if a == b:
        return True
    if a == 0.0 or b == 0.0:
        # One zero and the other not -> they disagree by definition
        return False
    rel = abs(a - b) / max(abs(a), abs(b))
    # 5 sig figs => rel < 1e-5
    return rel < 10 ** (-digits)


def _numeric_pairs(d: dict[str, Any]) -> list[tuple[str, str, float, float]]:
    """Return all distinct pairs of finite-numeric top-level keys."""
    items = [(k, float(v)) for k, v in d.items() if _is_finite_number(v)]
    pairs = []
    for i, (k1, v1) in enumerate(items):
        for k2, v2 in items[i + 1 :]:
            pairs.append((k1, k2, v1, v2))
    return pairs


def _is_count_field(name: str) -> bool:
    """Field whose name implies a combinatorial count — small-integer
    coincidence is plausible, not suspicious."""
    nl = name.lower()
    tokens = set(re.findall(r"[a-z0-9]+", nl))
    if tokens & {"wins", "losses", "ties"}:
        return True
    count_markers = (
        "count",
        "n_",
        "_n",
        "num_",
        "_num",
        "rows",
        "total",
        "size",
        "_index",
        "step",
        "iteration",
        "epoch",
    )
    return any(m in nl for m in count_markers)


def _is_integer_value(v: float) -> bool:
    """True if the float value is exactly an integer."""
    return float(v).is_integer()


_IDENTIFIER_FIELDS = frozenset(
    {
        "experiment",
        "experiment_id",
        "exp_id",
        "id",
        "run_id",
        "task_id",
        "random_seed",
        "seed",
        "rng_seed",
        "torch_seed",
        "np_seed",
        "jax_seed",
        "milestone",
        "milestone_id",
        "schema_version",
        "version",
        "pid",
        "port",
        "gpu_id",
        "device_id",
        "rank",
        "world_size",
    }
)


def _is_timestamp_field(k: str) -> bool:
    """True if the field is a wall-clock TIMESTAMP (not a measured metric).

    Two timestamps share their leading significant figures by construction --
    any two events in the same era agree to many sig figs -- so they false-
    positive the tautology check exactly like identifiers/seeds.
    Origin: exp4763 (.438 A3 self-play) -- `checkpoint_mtime_before_ns`
    (1.78244782e18) vs `checkpoint_mtime_after_ns` (1.78244974e18) was a GENUINE
    ~2s checkpoint advance (verdict success_..._checkpoint_refreshed) but flagged
    TAUTOLOGY because both nanosecond epoch timestamps start 1.78244...

    Conservative: a bare duration like `latency_ns` / `duration_ns` is NOT a
    timestamp (no time-ish token) and is left in the tautology check, so a real
    duration coincidence is still caught.
    """
    kl = k.lower()
    if "mtime" in kl or "timestamp" in kl:
        return True
    if kl.endswith(("_ts", "_epoch", "_unixtime")):
        return True
    # epoch timestamps in ns/us/ms carry a time-ish token (time/ts/clock/epoch/stamp)
    if kl.endswith(("_ns", "_us", "_ms")) and any(
        t in kl for t in ("time", "clock", "epoch", "stamp")
    ):
        return True
    return False


def _is_chance_floor_score(k: str) -> bool:
    """True if the field is an AUROC / probe / control SCORE whose chance floor
    is 0.5 -- such fields legitimately sit at exactly 0.5 (a majority-class
    control by construction, a null/origin probe driven to chance, a shuffled
    control), so two of them coinciding at 0.5 is a floor coincidence, not a
    distinct-measurement tautology. See exp4771 (S0' reopen) in check_tautology."""
    kl = k.lower()
    return any(
        t in kl
        for t in ("auroc", "auc", "probe", "control", "chance", "baseline", "majority", "shuffled")
    )


def _is_identifier_field(k: str) -> bool:
    """True if the field is an identifier / seed / timestamp / metadata field,
    not a measured metric. Identifiers and timestamps legitimately coincide
    (e.g. random_seed == experiment_id; two mtimes share leading sig figs) and
    must be excluded from tautology comparison."""
    kl = k.lower()
    if kl in _IDENTIFIER_FIELDS:
        return True
    if _is_timestamp_field(kl):
        return True
    return kl.endswith("_seed") or kl.endswith("_id") or kl.endswith("_seed_used")


# Reference fields name a KNOWN PRIOR baseline (not this experiment's fresh
# outcome); arithmetic-derived fields name a FUNCTION of other reported fields
# (a delta = treatment - baseline). Neither is an INDEPENDENT measurement, so
# their coincidence — two names for the same baseline, two related baselines that
# share a starting reference, or a delta equal to its own baseline (which happens
# whenever treatment == 2*baseline) — is structural arithmetic, not the "two
# DISTINCT measured metrics agree to >5 sig figs" signal a fabrication TAUTOLOGY
# is meant to catch. This mirrors the _is_identifier_field carve-out.
#
# SAFETY (adversarial review 2026-06-22): matching is SUFFIX-ANCHORED (not bare
# substring) so it cannot collide with measured outcomes whose names merely
# contain "reference"/"diff"/"change" (e.g. `n_referenced_artifacts`,
# `all_spins_different`, `ops_changelog_modified`). And a `*_delta` field is
# treated as derived ONLY IF its value is VERIFIED to equal the difference of two
# other present fields that SHARE its metric stem — so a fabricator cannot escape
# quarantine merely by NAMING two distinct copied outcomes `accuracy_delta` /
# `auroc_delta` (those have no backing arithmetic and stay CRITICAL).
_DELTA_SUFFIXES = ("_delta", "_diff", "_change")


def _is_reference_field(k: str) -> bool:
    """True if the field names a KNOWN PRIOR baseline/reference. Suffix-anchored
    to avoid substring collisions with measured outcomes.

    The prior-best of a metric is also a carried reference (it EQUALS the baseline until the
    metric improves), so a `prior_best_*` / `prior_submitted_*` field is a reference too --
    prefix-anchored. Origin: exp4886 (.450 A4) where `first_win_baseline` (=0.04) vs
    `prior_best_heldout_first_win_rate` (=0.04) was CRITICAL-flagged because only the `_baseline`
    side was recognized; both are references, so the pair is the expected no-improvement state."""
    kl = k.lower()
    return (
        kl in ("baseline", "reference", "ref")
        or kl.endswith(("_baseline", "_reference", "_ref"))
        or "_baseline_" in kl
        or "_reference_" in kl
        or kl.startswith(("prior_best", "prior_submitted"))
    )


def _delta_stem(k: str) -> str | None:
    """If k names a delta/diff/change, return the metric STEM it should derive
    from (text before the delta token); '' for a bare `delta`; None otherwise."""
    kl = k.lower()
    if kl in ("delta", "diff", "change"):
        return ""
    for suf in _DELTA_SUFFIXES:
        if kl.endswith(suf):
            return kl[: -len(suf)]
        marker = suf + "_"  # infix, e.g. accuracy_delta_vs_self_consistency
        if marker in kl:
            return kl.split(marker, 1)[0]
    return None


def _is_verified_arithmetic_delta(k: str, v: Any, d: dict[str, Any]) -> bool:
    """True if k is a delta/diff/change field whose value EQUALS the difference of
    two other present numeric fields that SHARE its metric stem — i.e. a genuinely
    DERIVED quantity, not an independent measurement merely NAMED like a delta.
    Null deltas (==0) are out of scope here (handled by the honest-null carve-out).
    The stem-binding + arithmetic check closes the fabrication hole where two
    distinct copied outcomes are named `*_delta` to dodge quarantine."""
    stem = _delta_stem(k)
    if stem is None or not _is_finite_number(v):
        return False
    target = abs(float(v))
    if target <= 1e-12:
        return False
    operands = [
        float(x)
        for kk, x in d.items()
        if _is_finite_number(x) and kk.lower() != k.lower() and (stem == "" or stem in kk.lower())
    ]
    for i in range(len(operands)):
        for j in range(len(operands)):
            if i == j:
                continue
            a, b = operands[i], operands[j]
            tol = max(1e-9, 1e-6 * max(abs(a), abs(b), 1.0))
            if abs(abs(a - b) - target) <= tol:
                return True
    return False


def _is_structural_nonmeasurement(k: str, v: Any, d: dict[str, Any]) -> bool:
    """True if k is a baseline/reference OR a VERIFIED arithmetic delta — a field
    that is not an independent fresh measurement, so its coincidence with another
    such field is structural arithmetic rather than a fabrication TAUTOLOGY."""
    return _is_reference_field(k) or _is_verified_arithmetic_delta(k, v, d)


_SMALL_SHARED_DENOMINATOR_MAX = 100
_RATE_METRIC_MARKERS = (
    "rate",
    "fraction",
    "first_win",
    "transfer",
    "winner_generated",
)
_DENOMINATOR_KEY_MARKERS = (
    "denominator",
    "variant_attempts_count",
    "winner_generated_attempted_count",
    "variant_count",
    "n_variants",
    "total_variants",
)


def _is_rate_metric_field(k: str) -> bool:
    """True for small-sample k/N ARC rate fields, not arbitrary floats."""
    kl = k.lower()
    return (
        kl.endswith(("_rate", "_fraction"))
        or "_rate_" in kl
        or "_fraction_" in kl
        or any(marker in kl for marker in _RATE_METRIC_MARKERS[2:])
    )


def _add_variant_denominators_from_value(
    key: str,
    value: Any,
    out: set[int],
) -> None:
    """Collect explicit small variant denominators from counts or list lengths."""
    kl = key.lower()
    if _is_finite_number(value) and any(marker in kl for marker in _DENOMINATOR_KEY_MARKERS):
        n = int(float(value))
        if 1 < n <= _SMALL_SHARED_DENOMINATOR_MAX and float(value).is_integer():
            out.add(n)
        return
    if isinstance(value, list):
        if "variant" in kl and 1 < len(value) <= _SMALL_SHARED_DENOMINATOR_MAX:
            out.add(len(value))
        for item in value:
            if isinstance(item, (dict, list)):
                _add_variant_denominators_from_value(key, item, out)
    elif isinstance(value, dict):
        for nested_key, nested_value in value.items():
            nested_path = f"{key}.{nested_key}" if key else str(nested_key)
            _add_variant_denominators_from_value(nested_path, nested_value, out)


def _variant_denominators(d: dict[str, Any]) -> set[int]:
    """Return small variant denominators explicitly evidenced in the artifact."""
    denominators: set[int] = set()
    for key, value in d.items():
        _add_variant_denominators_from_value(key, value, denominators)
    return denominators


def _is_fraction_over_denominator(v: float, denominator: int) -> bool:
    if not (0.0 <= float(v) <= 1.0):
        return False
    numerator = float(v) * denominator
    return abs(numerator - round(numerator)) <= 1.0e-8


def _is_small_shared_denominator_rate_pair(
    k1: str,
    k2: str,
    v1: float,
    v2: float,
    d: dict[str, Any],
) -> bool:
    """True for equal small k/N ARC rate metrics over the same denominator.

    This is the rate-metric analogue of the identifier carve-out: two rates
    ranging over the same 25 variants can collide because their numerators happen
    to be equal. Require explicit denominator evidence so copied unrelated
    high-precision metrics do not escape TAUTOLOGY by naming alone.
    """
    if not (_is_rate_metric_field(k1) and _is_rate_metric_field(k2)):
        return False
    if not _significant_digits_match(v1, v2, TAUTOLOGY_DIGITS):
        return False
    for denominator in _variant_denominators(d):
        if _is_fraction_over_denominator(v1, denominator) and _is_fraction_over_denominator(
            v2, denominator
        ):
            return True
    return False


_POSITIVE_CONTROL_NULL_VERDICT_MARKERS = (
    "honest_null",
    "null",
    "no_deeper",
    "no_value",
    "no_value_added",
    "no_lever",
    "no_lift",
    "no_gain",
    "no_delta",
    "unmoved",
    "unchanged",
    "positive_control_failed",
)
_POSITIVE_CONTROL_NULL_DOMAIN_MARKERS = (
    "efficiency",
    "transfer",
    "proposer",
    "reinduction",
    "llm_proposer",
    "core_efficiency",
    "generic_transfer",
)
_POSITIVE_CONTROL_NULL_DELTA_KEYS = (
    "efficiency_delta",
    "core_efficiency_delta",
    "generic_transfer_delta",
    "transfer_delta",
)


def _finite_float(d: dict[str, Any], key: str) -> float | None:
    value = d.get(key)
    return float(value) if _is_finite_number(value) else None


def _metric_pair_equal(d: dict[str, Any], left: str, right: str) -> bool:
    left_value = _finite_float(d, left)
    right_value = _finite_float(d, right)
    if left_value is None or right_value is None:
        return False
    return math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-12)


def _has_positive_control_null_metric(d: dict[str, Any]) -> bool:
    for key in _POSITIVE_CONTROL_NULL_DELTA_KEYS:
        value = _finite_float(d, key)
        if value is not None and math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return True
    return any(
        _metric_pair_equal(d, left, right)
        for left, right in (
            ("core_efficiency_baseline", "core_efficiency_best"),
            ("core_efficiency_baseline", "core_efficiency_integrated"),
            ("generic_transfer_rate_baseline", "generic_transfer_rate_with_verifier"),
            ("generic_transfer_rate_baseline", "generic_transfer_rate_integrated"),
        )
    )


def _is_positive_control_null_claim(d: dict[str, Any], verdict: str) -> bool:
    """True for efficiency/proposer/transfer nulls that require a positive control."""
    if not (
        "positive_control_passed" in d
        or "false_negative_risk_checked" in d
        or "positive_control" in verdict
    ):
        return False
    text = " ".join([verdict, " ".join(str(key).lower() for key in d)])
    if not any(marker in text for marker in _POSITIVE_CONTROL_NULL_DOMAIN_MARKERS):
        return False
    verdict_declares_null = any(
        marker in verdict for marker in _POSITIVE_CONTROL_NULL_VERDICT_MARKERS
    )
    if (
        verdict.startswith(("success:", "success_", "shipped:", "shipped_", "passed:", "passed_"))
        and not verdict_declares_null
    ):
        return False
    return verdict_declares_null or _has_positive_control_null_metric(d)


def _positive_control_failed_or_unchecked(d: dict[str, Any]) -> bool:
    """A null is informative only when the positive control and FNR check passed."""
    return (
        d.get("positive_control_passed") is not True
        or d.get("false_negative_risk_checked") is not True
    )


def check_false_negative_risk(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect NULL/negative claims that lack a valid positive control or that
    rest on a degenerate (un-exercised) method — the false-negative trap.

    Origin: exp3507 reported the process-energy reranker "does not beat
    self-consistency" (a null claim) while flip_count==0 — the reranker never
    actually changed a single selection, so the corpus had no headroom for the
    method to act on. A null claim from a method that was never exercised is
    not evidence the method fails; it's evidence the TEST was degenerate.

    Three independent risk signals (any one fires a warn; we never auto-pass a
    null claim that lacks a demonstrated positive control):

      1. NON-DEGENERACY: a `*flip*count*` / `*n_changed*` field == 0 means the
         method under test produced identical output to the baseline — the
         experiment cannot distinguish "method fails" from "no headroom."
      2. POSITIVE CONTROL / ORACLE: if an oracle/optimal upper bound is present
         and it does NOT exceed the baseline, the corpus has no selectable
         headroom; no method could win, so the null is uninformative.
      3. EXPLICIT G2-STYLE GATE: a `*non_degenerate*` / `*g2*` acceptance gate
         recorded as False is the experiment self-reporting its own degeneracy.
    """
    verdict_raw = d.get("honest_verdict") or ""
    verdict = (verdict_raw if isinstance(verdict_raw, str) else "").lower()
    null_markers = (
        "no_improvement",
        "does_not",
        "doesnt",
        "no_delta",
        "no_gain",
        "not_beat",
        "no_beat",
        "no_effect",
        "refuted",
        "null_result",
        "no_lift",
        "no_advantage",
        "no_benefit",
        "fails_to_beat",
        "not_better",
        "no_headroom",
    )
    is_null_claim = any(m in verdict for m in null_markers)
    if _is_positive_control_null_claim(d, verdict) and _positive_control_failed_or_unchecked(d):
        flags.append(
            Flag(
                kind="FALSE_NEGATIVE_RISK",
                severity="warn",
                detail=(
                    "false_negative_risk_open: null efficiency/proposer/transfer "
                    f"claim ({verdict[:64]!r}) lacks a passed positive control "
                    "and completed false-negative-risk check "
                    f"(positive_control_passed={d.get('positive_control_passed')!r}, "
                    "false_negative_risk_checked="
                    f"{d.get('false_negative_risk_checked')!r}). Treat this as a "
                    "broken-test signal, not as evidence for a clean null."
                ),
            )
        )
    if not is_null_claim:
        return

    # Signal 1: degenerate non-exercise (flip/change count == 0)
    for k, v in d.items():
        kl = k.lower()
        if not _is_finite_number(v):
            continue
        if ("flip" in kl and "count" in kl) or kl.startswith("n_changed") or ("n_flips" in kl):
            if float(v) == 0.0:
                flags.append(
                    Flag(
                        kind="FALSE_NEGATIVE_RISK",
                        severity="warn",
                        detail=(
                            f"Null claim ({verdict[:48]!r}) but {k}=0: the "
                            f"method never changed any selection. Cannot "
                            f"distinguish 'method fails' from 'no headroom'. "
                            f"Re-run on a corpus where {k}>0 (positive control) "
                            f"before treating this as evidence the method fails."
                        ),
                    )
                )

    # Signal 2: oracle/optimal upper bound does not exceed baseline
    oracle = None
    baseline = None
    for k, v in d.items():
        if not _is_finite_number(v):
            continue
        kl = k.lower()
        if (
            oracle is None
            and any(s in kl for s in ("oracle", "optimal", "upper_bound", "best_possible"))
            and any(s in kl for s in ("acc", "rate", "solve", "score"))
        ):
            oracle = float(v)
        if (
            baseline is None
            and any(s in kl for s in ("self_consistency", "baseline", "majority"))
            and any(s in kl for s in ("acc", "rate", "solve", "score"))
        ):
            baseline = float(v)
    if oracle is not None and baseline is not None and oracle <= baseline:
        flags.append(
            Flag(
                kind="FALSE_NEGATIVE_RISK",
                severity="warn",
                detail=(
                    f"Null claim but oracle/optimal upper bound ({oracle}) does "
                    f"not exceed the baseline ({baseline}): the corpus has no "
                    f"selectable headroom, so NO method could win here. The null "
                    f"is uninformative about the method. Build a difficulty-"
                    f"matched corpus with oracle>baseline before re-testing."
                ),
            )
        )

    # Signal 3: an explicit non-degeneracy / G2 gate self-reported False
    for k, v in d.items():
        kl = k.lower()
        if (
            isinstance(v, bool)
            and v is False
            and ("non_degenerate" in kl or "g2" in kl or "headroom" in kl)
        ):
            flags.append(
                Flag(
                    kind="FALSE_NEGATIVE_RISK",
                    severity="warn",
                    detail=(
                        f"Null claim and the experiment's own gate {k}=False: "
                        f"it self-reports the test was degenerate. Do not "
                        f"propagate this null to a forward-facing claim."
                    ),
                )
            )


_COMPARISON_VERDICT_MARKERS = (
    "beats",
    "_vs_",
    "generaliz",
    "outperform",
    "superior",
    "wins",
    "better_than",
    "_beat_",
    "dominates",
)
_TRIVIAL_BASELINE_MARKERS = (
    "vanilla",
    "greedy",
    "random",
    "single",
    "descent",
    "naive",
    "trivial",
    "default",
    "sequential",
    "baseline",
)


def _is_comparative_claim(d: dict[str, Any]) -> bool:
    """True if the artifact asserts one method beats another — the precondition
    for a CEILING_SATURATION false positive to matter."""
    v = str(d.get("honest_verdict", "")).lower()
    if any(m in v for m in _COMPARISON_VERDICT_MARKERS):
        return True
    # a weak baseline field strictly below the headline rate (exclude the
    # exact/oracle upper bound, which is supposed to be at the ceiling)
    headline = d.get("solve_rate") or d.get("accuracy") or d.get("pass_rate")
    if _is_finite_number(headline):
        for k, vv in d.items():
            kl = k.lower()
            if "baseline" in kl and "exact" not in kl and "oracle" not in kl:
                if _is_finite_number(vv) and float(vv) < float(headline):
                    return True
    return False


def check_ceiling_saturation(d: dict[str, Any], flags: list[Flag]) -> None:
    """Positive-claim partner to FALSE_NEGATIVE_RISK. A method-superiority claim
    is uninformative if the corpus is ceiling-saturated: every method variant
    (including a TRIVIAL baseline) and/or every difficulty tier hits the same
    maximum. Then the corpus has no headroom on the METHOD side — it cannot
    distinguish a powerful method from a trivial one, so "method X generalizes /
    beats Y" over-claims.

    Origin: exp3518 reported "energy global inference generalizes to graph
    coloring, solve_rate 1.00 vs AR 0.50", but vanilla_descent (no annealing, no
    tempering) ALSO solved 100% on the 'extreme' difficulty tier — all 5
    optimizers and all 4 difficulties tied at 1.0. The win only proves greedy-AR
    has a known ordering pathology, not that energy inference is uniquely
    capable. Only fires on comparative claims (gated) to avoid flagging
    legitimately-easy sanity checks that make no superiority claim."""
    if not _is_comparative_claim(d):
        return
    CEIL = 0.99

    # Signal 1 — a trivial baseline variant saturates alongside the method.
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        kl = k.lower()
        if not any(
            s in kl
            for s in (
                "by_optimizer",
                "by_variant",
                "by_method",
                "by_model",
                "by_approach",
                "by_sampler",
                "by_solver",
            )
        ):
            continue
        nums = {kk: float(vv) for kk, vv in v.items() if _is_finite_number(vv)}
        if len(nums) < 2 or max(nums.values()) < CEIL:
            continue
        at_ceiling = [kk for kk, vv in nums.items() if vv >= CEIL]
        trivial = [
            kk for kk in at_ceiling if any(m in kk.lower() for m in _TRIVIAL_BASELINE_MARKERS)
        ]
        if len(at_ceiling) >= 2 and trivial:
            flags.append(
                Flag(
                    kind="CEILING_SATURATION",
                    severity="warn",
                    detail=(
                        f"{k}: a trivial baseline ({trivial[0]}) also saturates "
                        f"at the ceiling (>= {CEIL}); {len(at_ceiling)} variants "
                        f"tie at {max(nums.values())}. The corpus cannot "
                        f"discriminate a powerful method from a trivial one, so "
                        f"the superiority claim is uninformative. Harden the "
                        f"corpus until the trivial baseline drops below ceiling."
                    ),
                )
            )

    # Signal 2 — every difficulty tier saturates: the difficulty axis is inert.
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        kl = k.lower()
        if not any(s in kl for s in ("by_difficulty", "by_tier", "by_hardness", "by_level")):
            continue
        nums = {kk: float(vv) for kk, vv in v.items() if _is_finite_number(vv)}
        if len(nums) >= 2 and min(nums.values()) >= CEIL:
            flags.append(
                Flag(
                    kind="CEILING_SATURATION",
                    severity="warn",
                    detail=(
                        f"{k}: every difficulty tier saturates at the ceiling "
                        f"({sorted(nums.items())}). The hardest tier is as easy "
                        f"as the easiest, so the difficulty axis is inert and a "
                        f"'solves hard instances' claim is unsupported. Add "
                        f"genuinely harder instances until the top tier drops."
                    ),
                )
            )


def _metric_from_top_or_pass_rates(d: dict[str, Any], key: str) -> float | None:
    value = d.get(key)
    if _is_finite_number(value):
        return float(value)
    pass_rates = d.get("pass_rates")
    if isinstance(pass_rates, dict) and _is_finite_number(pass_rates.get(key)):
        return float(pass_rates[key])
    return None


def _metric_items_from_top_or_pass_rates(
    d: dict[str, Any], keys: tuple[str, ...]
) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    pass_rates = d.get("pass_rates")
    for key in keys:
        value = d.get(key)
        if _is_finite_number(value):
            items.append((key, float(value)))
            continue
        if isinstance(pass_rates, dict) and _is_finite_number(pass_rates.get(key)):
            items.append((key, float(pass_rates[key])))
    return items


def check_degenerate_separation(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect synthetic selection wins where vote cannot win and oracle saturates.

    Exp 4282 exposed a false-positive pattern that the older artifact checks
    missed: a candidate pool filtered to wrong-majority rows with only a few
    candidates per task can produce `delta=1.0`, `vote@1=0.0`, and
    `oracle@K=1.0`. That proves the pool construction is separable, not that a
    learned selector generalized. The guard also catches the saturated-oracle
    variant where a perfect set-encoder selector reaches oracle@K exactly.
    """
    verdict = " ".join(
        str(d.get(key, "")).lower() for key in ("honest_verdict", "experiment", "schema")
    )
    if not any(
        marker in verdict for marker in ("arcgen", "cross_generator", "cross_family", "generaliz")
    ):
        return

    deltas = _metric_items_from_top_or_pass_rates(d, DEGENERATE_DELTA_KEYS)
    baselines = _metric_items_from_top_or_pass_rates(d, DEGENERATE_BASELINE_KEYS)
    oracle_at_k = _metric_from_top_or_pass_rates(d, "oracle_at_k")
    set_encoder_at_1 = _metric_from_top_or_pass_rates(d, "set_encoder_at_1")

    near_perfect_delta = [
        (key, value) for key, value in deltas if value >= DEGENERATE_DELTA_THRESHOLD
    ]
    trivial_baseline = [
        (key, value) for key, value in baselines if value <= DEGENERATE_BASELINE_THRESHOLD
    ]
    saturated_oracle_selector = (
        oracle_at_k is not None
        and set_encoder_at_1 is not None
        and math.isclose(oracle_at_k, 1.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(set_encoder_at_1, 1.0, rel_tol=0.0, abs_tol=1e-12)
    )

    if not ((near_perfect_delta and trivial_baseline) or saturated_oracle_selector):
        return

    delta_detail = (
        ", ".join(f"{key}={value}" for key, value in near_perfect_delta)
        if near_perfect_delta
        else "no near-perfect delta field"
    )
    baseline_detail = (
        ", ".join(f"{key}={value}" for key, value in trivial_baseline)
        if trivial_baseline
        else "no near-zero baseline field"
    )
    oracle_detail = "missing" if oracle_at_k is None else str(oracle_at_k)
    selector_detail = "missing" if set_encoder_at_1 is None else str(set_encoder_at_1)
    flags.append(
        Flag(
            kind="DEGENERATE_SEPARATION",
            severity="critical",
            detail=(
                f"Selector-vs-vote degenerate signal: {delta_detail}; "
                f"baselines {baseline_detail}; oracle_at_k={oracle_detail}; "
                f"set_encoder_at_1={selector_detail}. This matches a "
                f"wrong-majority/trivial-separation pool signature. Rebuild "
                f"the candidate pool with vote-winning tasks, realistic "
                f"candidate counts, and oracle_at_k<1 before claiming transfer."
            ),
        )
    )


def _is_control_arm_key(key: str) -> bool:
    """True when an arm name describes a baseline/control condition."""
    kl = key.lower()
    return any(marker in kl for marker in CONTROL_ARM_MARKERS)


def _is_placebo_or_replicate_key(key: str) -> bool:
    """True for intentionally duplicated placebo/replicate controls."""
    kl = key.lower()
    return any(marker in kl for marker in PLACEBO_OR_REPLICATE_MARKERS)


def _documented_identical_controls(d: dict[str, Any]) -> bool:
    """True when the artifact explicitly says identical controls are expected."""
    for key, value in d.items():
        kl = key.lower()
        if (
            isinstance(value, bool)
            and value is True
            and "identical" in kl
            and ("control" in kl or "arm" in kl or "placebo" in kl)
        ):
            return True
        if isinstance(value, str):
            vl = value.lower()
            if ("deliberately identical" in vl or "intentionally identical" in vl) and (
                "control" in vl or "arm" in vl or "placebo" in vl
            ):
                return True
    return False


def _arm_numeric_value(value: Any) -> float | None:
    """Extract the accuracy-like value from a flat or nested arm entry."""
    if _is_finite_number(value):
        return float(value)
    if not isinstance(value, dict):
        return None
    for metric_key in CONDITION_ARM_METRIC_KEYS:
        metric = value.get(metric_key)
        if _is_finite_number(metric):
            return float(metric)
    return None


def check_degenerate_controls(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect condition-arm maps where distinct controls are bit-identical.

    Exp 4293 exposed a no-op-controls failure mode: the headline arm differed,
    but every control arm reported the exact same aggregate accuracy. That is a
    harness bug signature for in-generation comparisons because model-self,
    unguided, and alternate-control arms should be independently exercised.
    """
    documented_identical = _documented_identical_controls(d)
    for map_key in CONDITION_ARM_MAP_KEYS:
        arm_map = d.get(map_key)
        if not isinstance(arm_map, dict):
            continue
        controls: list[tuple[str, float]] = []
        for arm_key, value in arm_map.items():
            if not isinstance(arm_key, str) or not _is_control_arm_key(arm_key):
                continue
            numeric_value = _arm_numeric_value(value)
            if numeric_value is None:
                continue
            controls.append((arm_key, numeric_value))
        if len(controls) < 2:
            continue

        by_bits: dict[str, list[tuple[str, float]]] = {}
        for arm_key, numeric_value in controls:
            by_bits.setdefault(float(numeric_value).hex(), []).append((arm_key, numeric_value))

        for duplicate_controls in by_bits.values():
            if len(duplicate_controls) < 2:
                continue
            if documented_identical or all(
                _is_placebo_or_replicate_key(arm_key) for arm_key, _ in duplicate_controls
            ):
                continue
            detail = ", ".join(
                f"{arm_key}={numeric_value:g}"
                for arm_key, numeric_value in sorted(duplicate_controls)
            )
            flags.append(
                Flag(
                    kind="DEGENERATE_CONTROLS",
                    severity="critical",
                    detail=(
                        f"{map_key}: distinct control arms have bit-identical "
                        f"values ({detail}). This matches a no-op/aliased-controls "
                        f"signature; rerun with independently exercised controls "
                        f"before claiming the condition-arm comparison."
                    ),
                )
            )


# Tokens an artifact's honest_verdict uses to DECLARE a no-value null result. A genuine fabrication
# would never self-label as a null, so requiring one of these is the load-bearing safety condition
# for the control-vs-treatment carve-out below.
_HONEST_NULL_VERDICT_TOKENS = (
    "honest_null",
    "no_value_added",
    "no_lever_raises",
    "no_value",
    "no_improvement",
    "no_metric_moved",
    "no_delta",
)
# Qualifier tokens that mark a metric key as one ARM of a control/treatment ablation (X_baseline vs
# X_with_verifier vs X_integrated). A pair where at least one side carries one of these, in an artifact
# that declares an honest null, is an EXPECTED ablation equality (the treatment changed nothing) — not
# a coincidence between two distinct measurements.
_CONTROL_TREATMENT_QUALIFIERS = (
    "_baseline_reference",
    "_baseline",
    "_with_verifier",
    "_integrated",
    "_random_router",
    "_control",
    "_treatment",
    "_reference",
    "_ablation",
)
_DELTA_COVERAGE_STOP_TOKENS = frozenset(
    {
        "arm",
        "baseline",
        "change",
        "control",
        "delta",
        "diff",
        "metric",
        "rate",
        "result",
        "score",
        "value",
        "with",
        "without",
    }
)


def _is_declared_honest_null(d: dict[str, Any]) -> bool:
    """True if the artifact's honest_verdict declares a no-value/no-lever null result."""
    v = str(d.get("honest_verdict", "")).lower()
    return any(tok in v for tok in _HONEST_NULL_VERDICT_TOKENS)


def _has_control_treatment_qualifier(k: str) -> bool:
    """True if the metric key carries a control/treatment ablation-arm qualifier."""
    kl = k.lower()
    return any(q in kl for q in _CONTROL_TREATMENT_QUALIFIERS)


def _is_explicit_zero(value: Any) -> bool:
    return _is_finite_number(value) and math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1e-12)


def _passing_positive_control_key(d: dict[str, Any]) -> str | None:
    """Return a top-level passing positive-control key if the artifact declares one."""
    for key, value in d.items():
        kl = key.lower()
        if (kl == "positive_control_passed" or kl.endswith("_control_passed")) and value is True:
            return key
    return None


def _delta_key_covers_pair(delta_key: str, left: str, right: str) -> bool:
    """True when a zero-delta field names the metric family shared by both arms."""
    stem = _delta_stem(delta_key)
    if stem is None:
        return False
    left_lower = left.lower()
    right_lower = right.lower()
    if stem and stem in left_lower and stem in right_lower:
        return True
    tokens = [
        tok
        for tok in re.findall(r"[a-z0-9]+", stem)
        if len(tok) >= 3 and tok not in _DELTA_COVERAGE_STOP_TOKENS
    ]
    return any(tok in left_lower and tok in right_lower for tok in tokens)


def _is_heldout_firstwin_null_delta_pair(left: str, right: str) -> bool:
    """True for the retargeted A4 first-win baseline-vs-integrated equality."""
    return {left.lower(), right.lower()} == {
        "first_win_baseline",
        "first_win_rate_integrated",
    }


def _declared_null_delta_descriptor(
    d: dict[str, Any], left: str, right: str
) -> dict[str, str] | None:
    """Recognize explicit honest-null evidence for an equal metric pair.

    The descriptor is deliberately stricter than an `honest_verdict` string:
    it needs a covering zero delta, a methodology note, and a passing control.
    """
    note = d.get("null_delta_methodology_note")
    if not isinstance(note, str) or not note.strip():
        return None
    control_key = _passing_positive_control_key(d)
    if control_key is None:
        return None
    if _is_heldout_firstwin_null_delta_pair(left, right) and _is_explicit_zero(
        d.get("first_win_delta_vs_baseline")
    ):
        return {
            "delta_key": "first_win_delta_vs_baseline",
            "methodology_key": "null_delta_methodology_note",
            "control_key": control_key,
        }
    for key, value in d.items():
        if key in (left, right):
            continue
        if _delta_stem(key) is None or not _is_explicit_zero(value):
            continue
        if _delta_key_covers_pair(key, left, right):
            return {
                "delta_key": key,
                "methodology_key": "null_delta_methodology_note",
                "control_key": control_key,
            }
    return None


def _declared_arc_nondegenerate_firstwin_null_descriptor(
    d: dict[str, Any], left: str, right: str
) -> dict[str, str] | None:
    """Recognize validated ARC generation/exploration no-lift first-win nulls."""
    if not _is_arc_generation_or_exploration_artifact(d):
        return None
    if not _real_field_has_true(d, "arms_non_degenerate"):
        return None
    note = d.get("null_delta_methodology_note")
    if not isinstance(note, str) or not note.strip():
        return None
    if d.get("positive_control_passed") is not True:
        return None
    left_lower = left.lower()
    right_lower = right.lower()
    if "first_win" not in left_lower or "first_win" not in right_lower:
        return None
    return {
        "delta_key": "validated_non_degenerate_first_win_null",
        "methodology_key": "null_delta_methodology_note",
        "control_key": "positive_control_passed",
    }


def check_tautology(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect distinct metrics agreeing to TAUTOLOGY_DIGITS sig figs.

    Tightened to avoid false positives on small-integer count
    coincidences (e.g. `completed_count == failed_count` is just
    arithmetic chance, not bug evidence). Genuine tautology signals
    are floating-point values matching to many decimal places —
    that's where two distinct loss-function gradient norms shouldn't
    coincide.
    """
    for k1, k2, v1, v2 in _numeric_pairs(d):
        if _legitimate_pair(k1, k2):
            continue
        # Skip identifier / seed / metadata fields. These are NOT metrics:
        # `experiment_id`, `experiment`, and `random_seed` legitimately all
        # equal the experiment number (seeding the RNG off the experiment ID
        # is good reproducibility practice — see exp3505/3506/3496/3481 which
        # this rule was false-flagging as TAUTOLOGY). Two identifiers agreeing
        # is structural, not a coincidence between two distinct measurements.
        if _is_identifier_field(k1) or _is_identifier_field(k2):
            continue
        # Skip two SCORE metrics both pinned at the 0.5 chance floor. An AUROC
        # majority-class control is 0.5 BY CONSTRUCTION; a successful null /
        # origin probe driven to chance is 0.5 (the DESIRED outcome); a
        # shuffled-label control is ~0.5. Two of them landing at exactly 0.5 is a
        # floor coincidence, not a distinct-measurement bug.
        # Origin: exp4771 (S0' origin-matched) -- the SUCCESS verdict
        # success_structural_energy_s0prime_reopens_s1 was TAUTOLOGY-flagged
        # because loo_auroc_majority_control=0.5 (definitional floor) ==
        # origin_probe_auroc=0.5 (the success signal that origin-matching removed
        # the S0 leak). Quarantining that would hide the headline result.
        if v1 == 0.5 and v2 == 0.5 and (_is_chance_floor_score(k1) or _is_chance_floor_score(k2)):
            continue
        # Skip small shared-denominator ARC rate metrics. A pair such as
        # `winner_generated_rate=2/25` and `generic_transfer_rate=2/25`
        # collides by arithmetic over the same variant set, not because two
        # unrelated measured floats were copied. Require rate/fraction naming
        # plus explicit small denominator evidence, mirroring the identifier
        # carve-out while keeping unrelated high-precision metrics critical.
        if _is_small_shared_denominator_rate_pair(k1, k2, v1, v2, d):
            continue
        declared_null_delta = _declared_null_delta_descriptor(d, k1, k2)
        if declared_null_delta is None:
            declared_null_delta = _declared_arc_nondegenerate_firstwin_null_descriptor(d, k1, k2)
        # Skip DECLARED control-vs-treatment HONEST NULLS. When an ablation artifact's own
        # honest_verdict declares a no-value null (verifier_router_no_value_added,
        # no_lever_raises_a_metric, ...), a control==treatment equality where one side is an
        # ablation arm (X_baseline == X_with_verifier == 0.04) is the EXPECTED, MEANINGFUL outcome —
        # the treatment changed nothing — not a coincidence between two distinct measurements. Gate
        # on BOTH the honest-null verdict (a fabrication never self-labels a null) AND a
        # control/treatment qualifier key, so a generic two-metric coincidence is still flagged.
        # Origin: exp4556 (HEADLINE generic_transfer null) + exp4560 (integration gate null) were
        # spuriously quarantined ~8x across .420/.421, excluding the project's two most important ARC
        # measurements from capstone aggregation. Mirrors the _is_identifier_field carve-out above.
        if (
            declared_null_delta is None
            and _is_declared_honest_null(d)
            and (_has_control_treatment_qualifier(k1) or _has_control_treatment_qualifier(k2))
        ):
            continue
        # Skip count-coincidence pairs: both names imply counts AND
        # both values are small integers.
        if (
            _is_count_field(k1)
            and _is_count_field(k2)
            and _is_integer_value(v1)
            and _is_integer_value(v2)
            and abs(v1) < 1000
        ):
            continue
        # Skip integer-value pairs entirely if either value is small
        # — small integer coincidence is uninformative.
        if _is_integer_value(v1) and _is_integer_value(v2) and abs(v1) < 100 and abs(v2) < 100:
            continue
        if _significant_digits_match(v1, v2, TAUTOLOGY_DIGITS):
            if declared_null_delta is not None:
                flags.append(
                    Flag(
                        kind="TAUTOLOGY",
                        severity="warn",
                        detail=(
                            f"{k1}={v1!r} and {k2}={v2!r} agree to "
                            f">{TAUTOLOGY_DIGITS} sig figs, but the artifact "
                            f"declares declared_null_delta via "
                            f"{declared_null_delta['delta_key']}==0, "
                            f"{declared_null_delta['methodology_key']}, and "
                            f"{declared_null_delta['control_key']}=true. "
                            f"Downgraded CRITICAL->WARN so an honest null is "
                            f"read instead of quarantined."
                        ),
                    )
                )
            # Baseline-identity carve-out: when BOTH sides are baseline/reference
            # or arithmetic-derived (delta) fields, the agreement is structural
            # arithmetic (the same baseline reported twice, or a delta collapsing
            # onto its baseline when treatment == 2*baseline), NOT two distinct
            # measured metrics coinciding. Downgrade CRITICAL -> annotated WARN so
            # the artifact is NOT quarantined but the coincidence is still
            # surfaced for audit. Two distinct OUTCOME metrics agreeing stays
            # CRITICAL. Origin: exp4592 (.424 generation-completeness, a GENUINE
            # winner_generated 1/25->2/25 positive) was quarantined by 11
            # TAUTOLOGY flags, ~all the 0.04 baseline/delta arithmetic cascade
            # (baseline=0.04=1/25, treatment=0.08=2/25, delta=0.08-0.04=0.04).
            if (
                _is_structural_nonmeasurement(k1, v1, d)
                and _is_structural_nonmeasurement(k2, v2, d)
                and declared_null_delta is None
            ):
                flags.append(
                    Flag(
                        kind="TAUTOLOGY",
                        severity="warn",
                        detail=(
                            f"{k1}={v1!r} and {k2}={v2!r} agree, but BOTH are "
                            f"baseline/reference or VERIFIED arithmetic-derived "
                            f"(delta) fields, not independent measurements — "
                            f"structural arithmetic, not a coincidence between two "
                            f"distinct measured metrics. Downgraded CRITICAL->WARN "
                            f"(baseline-identity carve-out)."
                        ),
                    )
                )
            elif declared_null_delta is None:
                flags.append(
                    Flag(
                        kind="TAUTOLOGY",
                        severity="critical",
                        detail=(
                            f"{k1}={v1!r} and {k2}={v2!r} agree to "
                            f">{TAUTOLOGY_DIGITS} sig figs. Two distinct "
                            f"metrics matching this precisely is more likely "
                            f"a bug than a finding."
                        ),
                    )
                )


def _legitimate_pair(k1: str, k2: str) -> bool:
    """Pairs where bit-identity is structurally legitimate."""
    legit_suffixes = ("_abs", "_pct", "_ratio", "_mean", "_var", "_std", "_min", "_max")
    if k1.startswith(k2) or k2.startswith(k1):
        for s in legit_suffixes:
            if k1.endswith(s) or k2.endswith(s):
                return True
    # Same metric measured at two timepoints (initial vs final) often
    # legitimately matches when convergence is reached. Allow.
    paired_prefixes = (("initial_", "final_"), ("baseline_", "best_"))
    for a, b in paired_prefixes:
        if (k1.startswith(a) and k2.startswith(b)) or (k1.startswith(b) and k2.startswith(a)):
            if k1[len(a) :] == k2[len(b) :] or k1[len(b) :] == k2[len(a) :]:
                return True
    # Two same-type metrics of the same family coinciding is a legitimate
    # research outcome, not a bug. The canonical case is an ablation /
    # selection-strategy sweep where two different aggregation conditions
    # produce the SAME exact-match accuracy on the SAME corpus (e.g.
    # energy-weighted-vote vs self-consistency tying when the energy never
    # flips the majority — exp3449). Accuracy is a bounded rational
    # (correct / n), so exact coincidence across strategies is expected and
    # meaningful, exactly like the initial/final convergence carve-out above.
    # Narrow guard: BOTH names must end in the same metric-family suffix.
    same_family_suffixes = ("_accuracy", "_solve_rate", "_pass_rate", "_cross_family_delta")
    for s in same_family_suffixes:
        if k1.endswith(s) and k2.endswith(s):
            return True
    # An INDEPENDENT REPRODUCTION / re-score / clean-room replication of a metric is
    # DESIGNED to reproduce the original value (the G2-reproducer pattern; CLAUDE.md
    # publication gate G2). When such a field matches the metric it reproduces, that is
    # a SUCCESSFUL reproduction — the evidence of reproducibility we WANT — NOT two
    # distinct metrics coinciding by bug. Canonical case: exp4257's
    # `independent_rescore_delta == single_seed_4245_delta` (0.4423) confirming the +44pp
    # ARC oracle-distinct win reproduces off the persisted artifact via a second code
    # path. A reproduction field carries an explicit re-score/replication marker, so this
    # cannot mask a genuine two-distinct-metric coincidence (those have no such marker).
    repro_markers = ("rescore", "re_score", "reproduc", "replicat", "cleanroom", "clean_room")
    kl1, kl2 = k1.lower(), k2.lower()
    if any(m in kl1 for m in repro_markers) or any(m in kl2 for m in repro_markers):
        return True
    # NESTED WALL CLOCKS. `duration_s` is the analyser PROCESS's total wall time;
    # `measurement_wall_s` is the sum of the per-row measurement spans INSIDE that same
    # process. One physically CONTAINS the other, so `duration_s >= measurement_wall_s`
    # always and the gap is only the analyser's own bookkeeping overhead -- which on a
    # ~30s numpy run is sub-millisecond, i.e. the two agree to 5+ significant figures BY
    # CONSTRUCTION. That is structural nesting, not a coincidence between two independent
    # measurements, which is the only thing TAUTOLOGY is meant to catch.
    #
    # Origin: 2026-07-28. exp6011/6012/6013 each report both fields deliberately (their
    # own source comments say "THE MEASUREMENT CLOCK IS NOT THE ANALYSER CLOCK ...
    # reporting one as the other is how an artifact ends up claiming a measurement it did
    # not make"). Rebuilding them made the two clocks land within 0.0002s and TAUTOLOGY
    # fired CRITICAL, which via the fabrication gate would have quarantined three honest
    # artifacts and excluded them from capstone aggregation. Raising the emitted precision
    # from 3dp to 6dp did NOT fix it and could not: the values genuinely agree to that many
    # figures. Re-running until the timings happened to differ would be dodging the
    # detector, not satisfying it. So the correct fix is here, and it is the same shape as
    # the identifier/seed carve-out above (CLAUDE.md, "TAUTOLOGY excludes identifiers/seeds").
    #
    # Deliberately NOT a generic "any two *_s duration fields" rule: two UNRELATED
    # durations coinciding to 5 sig figs would still be real evidence of a copy-paste bug,
    # so this is anchored to the specific nested pair by name.
    nested_clock_pair = {"duration_s", "measurement_wall_s"}
    if {k1, k2} == nested_clock_pair:
        return True
    return False


def _is_declared_honest_zero_delta(k: str, d: dict[str, Any]) -> bool:
    """True for explicit zero deltas documented as measured honest nulls."""
    kl = k.lower()
    if kl == "level_credit_delta":
        return (
            d.get("solve_claimed") is False
            and d.get("offline_reproduced") is False
            and int(d.get("level_credit_delta") or 0) == 0
            and "level_credit_delta" in dict(d.get("field_provenance") or {})
        )
    if "delta" not in kl:
        return False
    if not d.get("null_delta_methodology_note"):
        return False
    if not (
        d.get("bare_control_passed") is True
        or d.get("positive_control_passed") is True
        or d.get("false_negative_risk_checked") is True
    ):
        return False
    return any(
        marker in kl
        for marker in (
            "solve_rate",
            "first_win_rate",
            "state_coverage",
            "actions",
            "live_lift",
            "forward_transfer",
            "level_delta",
            "reproducible_total_levels_delta",
        )
    )


def check_implausible_perfect(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect implausibly perfect metrics (TPR/acc=1.0, error=0.0)."""
    perfect_score_fields = (
        "tpr",
        "accuracy",
        "auroc",
        "f1",
        "precision",
        "recall",
        "pass_rate",
        "success_rate",
        "agreement_rate",
    )
    perfect_error_fields = (
        "error",
        "loss",
        "delta",
        "divergence",
        "violations",
    )

    for k, v in d.items():
        if not _is_finite_number(v):
            continue
        kl = k.lower()
        vf = float(v)
        # Implausible 1.0 on a [0,1] score field
        if any(s in kl for s in perfect_score_fields) and vf == 1.0:
            n = d.get("n_samples") or d.get("n_adversarial_examples") or 0
            try:
                n = int(n)
            except (TypeError, ValueError):
                n = 0
            if n > 0 and n < 1000:
                flags.append(
                    Flag(
                        kind="IMPLAUSIBLE_PERFECT",
                        severity="warn",
                        detail=(
                            f"{k}={vf} on {n} samples is implausibly clean. "
                            f"Real classifiers exhibit non-zero error at small N."
                        ),
                    )
                )
        # Implausible 0.0 on an error/loss field
        if any(s in kl for s in perfect_error_fields) and vf == 0.0:
            if _is_count_field(k) and _is_integer_value(vf):
                continue
            if _is_declared_honest_zero_delta(k, d):
                continue
            # Allow 0.0 only if the field name is a clear baseline marker
            if "baseline" not in kl and "zero" not in kl:
                flags.append(
                    Flag(
                        kind="IMPLAUSIBLE_PERFECT",
                        severity="info",
                        detail=(f"{k}={vf} (exactly zero). Confirm this is not a stub default."),
                    )
                )


def check_sign_anomaly(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect optimization that went the wrong direction."""
    init_keys = [k for k in d if k.startswith("initial_") and _is_finite_number(d[k])]
    for ik in init_keys:
        fk = "final_" + ik[len("initial_") :]
        if fk not in d or not _is_finite_number(d[fk]):
            continue
        iv = float(d[ik])
        fv = float(d[fk])
        metric_name = ik[len("initial_") :]
        # Energy / loss / error should DECREASE during optimization.
        decrease_expected = any(
            m in metric_name.lower() for m in ("energy", "loss", "error", "violation", "regret")
        )
        # Accuracy / reward / score should INCREASE.
        increase_expected = any(
            m in metric_name.lower() for m in ("accuracy", "reward", "score", "lift")
        )

        if decrease_expected and fv > iv:
            flags.append(
                Flag(
                    kind="SIGN_ANOMALY",
                    severity="warn",
                    detail=(
                        f"initial_{metric_name}={iv} -> final_{metric_name}={fv} "
                        f"({metric_name} INCREASED). Optimization was supposed "
                        f"to minimize this. Either methodology bug, sign "
                        f"reversal, or a real finding that needs explicit "
                        f"acknowledgment."
                    ),
                )
            )
        elif increase_expected and fv < iv:
            flags.append(
                Flag(
                    kind="SIGN_ANOMALY",
                    severity="warn",
                    detail=(
                        f"initial_{metric_name}={iv} -> final_{metric_name}={fv} "
                        f"({metric_name} DECREASED). Optimization was supposed "
                        f"to maximize this."
                    ),
                )
            )


def _has_compute_bound_marker(d: dict[str, Any]) -> bool:
    """Walk dict for any compute-bound marker string."""
    text = json.dumps(d)
    return any(m in text for m in COMPUTE_BOUND_MARKERS)


def _inference_substrate_text(d: dict[str, Any]) -> str:
    """Return the top-level declared substrate as a stripped string.

    Principle-annotated artifacts may store the field as
    ``{"value": "...", "principle": "..."}``. Handle that shape here too so
    direct helper tests and caller-normalized verification use the same
    top-level declaration.
    """
    value = d.get("inference_substrate")
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    return str(value or "").strip()


def _inference_substrate_value_matches(raw: str, canonical: str) -> bool:
    """True when a raw substrate string declares a canonical substrate value.

    Newer ARC artifacts often store the canonical value followed by a human
    principle explanation (`value -- why this floor applies`). Matching the
    leading value keeps that reader annotation from changing verifier behavior.
    """
    if raw == canonical:
        return True
    # Tolerate a human principle/floor note appended after the canonical value
    # with ANY separator (space, `--`, `;`, `,`, `:`, `.`): the leading token is
    # what selects the duration floor. exp4756 (.437 B2) correctly declared
    # "aggregation_from_upstream_artifacts; 100us floor." but was
    # DURATION_TOO_SHORT false-positive-flagged because the `;` separator was not
    # recognized (only space / `--` were). The boundary-char check keeps a longer
    # different enum (e.g. `<canonical>_v2`) from matching, since `_`/alnum are
    # not separators.
    prefix = raw.split("--", 1)[0].strip()
    if prefix == canonical:
        return True
    if raw.startswith(canonical):
        return raw[len(canonical) : len(canonical) + 1] in {" ", "-", ";", ",", ":", "."}
    return False


def _inference_substrate_matches(d: dict[str, Any], canonical: str) -> bool:
    """True when `inference_substrate` declares a canonical substrate value."""
    return _inference_substrate_value_matches(_inference_substrate_text(d), canonical)


def _match_declared_substrate(raw: str, candidates: tuple[str, ...]) -> str | None:
    """Return the allowlisted substrate value matched by the top-level declaration."""
    for canonical in candidates:
        if _inference_substrate_value_matches(raw, canonical):
            return canonical
    return None


def _classify_inference_substrate(d: dict[str, Any]) -> dict[str, str]:
    """Classify substrate from explicit top-level provenance only.

    This helper intentionally ignores arbitrary nested strings. If the top-level
    declaration is missing, malformed, or unknown, callers keep using the legacy
    compute-marker scan as the conservative fallback.
    """
    raw = _inference_substrate_text(d)
    if not raw:
        return {
            "kind": SUBSTRATE_KIND_UNKNOWN,
            "declared_value": "",
            "matched_value": "",
            "source": "missing_top_level_inference_substrate",
        }

    for kind, candidates in (
        (SUBSTRATE_KIND_AGGREGATION, AGGREGATION_SUBSTRATE_ALIASES),
        (SUBSTRATE_KIND_NO_LLM, NO_LLM_SUBSTRATE_ALIASES),
        (SUBSTRATE_KIND_LIVE_MODEL, LIVE_MODEL_SUBSTRATE_ALIASES),
    ):
        matched = _match_declared_substrate(raw, candidates)
        if matched is not None:
            return {
                "kind": kind,
                "declared_value": raw,
                "matched_value": matched,
                "source": "top_level_inference_substrate",
            }

    return {
        "kind": SUBSTRATE_KIND_UNKNOWN,
        "declared_value": raw,
        "matched_value": "",
        "source": "unknown_top_level_inference_substrate",
    }


def _is_live_llm_inference(d: dict[str, Any]) -> bool:
    """True when the artifact declares a live LLM inference substrate."""
    return _inference_substrate_matches(d, LIVE_LLM_SUBSTRATE)


_VERDICT_TERMINAL_PREFIXES = (
    "complete:",
    "complete ",
    "complete_",
    "success:",
    "success ",
    "success_",
    "passed:",
    "passed ",
    "passed_",
    "shipped:",
    "shipped ",
    "shipped_",
)


def _strip_verdict_terminal_prefix(verdict: str) -> str:
    """Strip a leading terminal-prefix marker (per CLAUDE.md's Verdict Terminal-Prefix
    Discipline) plus any following whitespace, so downstream string-shape checks (like
    `blocked_` recognition) see the same content whether or not the script prefixed it.
    Mirrors `research_conductor.py:_verdict_is_untrustworthy`'s own terminal-prefix list."""
    stripped = verdict.strip()
    lowered = stripped.lower()
    for prefix in _VERDICT_TERMINAL_PREFIXES:
        if lowered.startswith(prefix):
            return stripped[len(prefix) :].lstrip()
    return stripped


def _is_precondition_check_only_blocked(d: dict[str, Any]) -> bool:
    """True when an artifact stopped before invoking the compute substrate.

    Recognizes two forms (either is sufficient): the explicit `precondition_check_only`
    substrate declaration, OR a `blocked_`-prefixed honest_verdict -- with or WITHOUT a
    leading terminal-prefix marker (`complete:`/`success:`/etc.). The latter is trusted
    directly per this project's own Pre-Launch Preconditions Discipline + Verdict
    Terminal-Prefix Discipline: a `blocked_*` verdict is a MANDATED, structured admission
    that the compute-bound work did not happen -- scripts are not required to also switch
    their declared `inference_substrate` on the blocked path (some hardcode the same
    substrate value for both the live and blocked branches of the same experiment, e.g.
    exp5274's INFERENCE_SUBSTRATE constant, discovered 2026-07-05), so requiring a
    specific substrate string here was over-narrow and produced a false DURATION_TOO_SHORT
    on an honest, zero-work blocked artifact (duration_s=0.14, extraction_results=[],
    rows_total=0 -- fully consistent with "nothing ran").

    Fixed 2026-07-14 (exp5713 incident): the ORIGINAL fix only recognized a BARE
    `blocked_` prefix, missing the far more common `complete: blocked_<resource>` form
    that CLAUDE.md's Verdict Terminal-Prefix Discipline actually MANDATES every terminal
    verdict use (a clean precondition-block IS a terminal state, not an ambiguous
    partial one). Any script following that discipline correctly for its blocked branch
    -- as exp5705's, exp5709's, and exp5713's all do (`f"complete: blocked_{miss}"`) --
    would have been false-flagged the moment its blocked path became the actual checked-
    in artifact. Now strips the terminal prefix first via `_strip_verdict_terminal_prefix`
    before checking for `blocked_`, so both forms are recognized consistently.
    """
    verdict = str(d.get("honest_verdict") or "")
    stripped = _strip_verdict_terminal_prefix(verdict).lower()
    return stripped.startswith("blocked_") or stripped.startswith("blocked:")


def _is_verifier_scoring_only(d: dict[str, Any]) -> bool:
    """True if the artifact declares it scored verifiers against
    cached candidate triples without invoking LLM inference.

    The GGUF / CUDA markers in such artifacts are vestigial template
    declarations -- the experiment's wall-clock should be measured
    against the verifier-scoring loop, not against model loading +
    inference. Two recognition modes (either is sufficient):

    1. Explicit declaration -- `inference_substrate` field equals the
       canonical sentinel `verifier_ensemble_against_cached_candidates`.
       This is the forward-only path; planner-emitted task prompts
       should require this declaration for verifier-scoring tasks.

    2. Known verifier-only schema prefix. Covers historical artifacts
       (exp2837 fover_memory_leakage_v3, cross_corpus_verifier_matrix
       family, etc.) that were authored before the explicit-field
       discipline shipped.
    """
    if _inference_substrate_matches(d, VERIFIER_SCORING_SUBSTRATE):
        return True
    schema = str(d.get("schema") or d.get("schema_version") or "")
    return any(schema.startswith(p) for p in VERIFIER_SCORING_SCHEMA_PREFIXES)


def _cheap_learned_value_marker(value: Any) -> str | None:
    """Return a cheap learned-value/CNN/linear marker from real fields."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            key_text = str(key).lower()
            for marker in CHEAP_LEARNED_VALUE_MARKERS:
                if marker in key_text:
                    return marker
            found = _cheap_learned_value_marker(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _cheap_learned_value_marker(item)
            if found is not None:
                return found
    elif isinstance(value, str):
        text = value.lower()
        for marker in CHEAP_LEARNED_VALUE_MARKERS:
            if marker in text:
                return marker
    return None


def _has_cheap_learned_value_methodology(d: dict[str, Any]) -> bool:
    """True when a fast cached value-head run carries auditable methodology."""
    return (
        bool(d.get("model_specs"))
        and (d.get("random_seed") is not None or d.get("seed") is not None)
        and bool(d.get("reproducibility_checksum"))
    )


def _cheap_learned_value_floor_descriptor(d: dict[str, Any]) -> dict[str, Any] | None:
    """Return the calibrated sub-1s floor for method-bearing cheap value scoring.

    This is intentionally narrower than the generic verifier-scoring substrate:
    it only applies below the existing 1s verifier floor, only when the artifact
    declares a cached learned value/CNN/linear forward-pass marker, and only
    when the methodology fields needed for replay are present.
    """
    duration = d.get("duration_s")
    if not (
        _is_verifier_scoring_only(d)
        and _is_finite_number(duration)
        and float(duration) < VERIFIER_SCORING_MIN_DURATION_S
    ):
        return None
    marker = _cheap_learned_value_marker(d)
    if marker is None or not _has_cheap_learned_value_methodology(d):
        return None
    return {
        "substrate": VERIFIER_SCORING_SUBSTRATE,
        "min_duration_s": CHEAP_LEARNED_VALUE_MIN_DURATION_S,
        "reason": "cheap_learned_value_scoring",
        "marker": marker,
        "methodology_fields": [
            "model_specs",
            "random_seed",
            "reproducibility_checksum",
        ],
    }


def _is_aggregation_only(d: dict[str, Any]) -> bool:
    """True if the artifact is a synthesis / aggregation over upstream
    artifacts and not itself a compute-bound experiment.

    Capstones, archive/activate transitions, and paper-table builders
    legitimately finish in milliseconds. Their model_specs / GGUF
    markers (when present) are inherited from upstream sources cited
    in the artifact body, not invoked by the artifact itself.
    """
    if _classify_inference_substrate(d)["kind"] == SUBSTRATE_KIND_AGGREGATION:
        return True
    schema = str(d.get("schema") or d.get("schema_version") or "")
    return any(schema.startswith(p) for p in AGGREGATION_SCHEMA_PREFIXES)


def _is_deterministic_verifier(d: dict[str, Any]) -> bool:
    """True when the artifact declares replay / reconciliation over checked-in evidence.

    Besides the canonical sentinels, recognise substrate names that self-declare
    deterministic replay or ledger reconciliation over checked-in artifacts and make
    NO live LLM call of their own (e.g. registry/gaps-hygiene tasks declaring
    `cached_gap4_replay_and_ledger_reconciliation`). These finish in milliseconds and
    must use the JSON-work floor, not the 60s live-model floor. Their GGUF/CUDA markers
    are inherited from the upstream artifacts they replay, not invoked here. This was a
    recurring DURATION_TOO_SHORT false-positive on .390 infra tasks (B2/D2). Live
    inference artifacts declare `live_llm_inference`, so this never masks a real
    fast-fabrication of a live-model claim. (Inference-Substrate Declaration Discipline.)
    """
    sub = _inference_substrate_text(d)
    if sub in DETERMINISTIC_VERIFIER_SUBSTRATES:
        return True
    return any(tok in sub for tok in ("replay", "reconciliation"))


def _is_arc_live_agent_no_llm(d: dict[str, Any]) -> bool:
    """True when the artifact declares live ARC-agent environment-stepping with NO LLM call.

    Recognition is the canonical `inference_substrate` sentinel only (forward-only, no legacy
    schema-prefix fallback -- this class was introduced 2026-06-30, so there is no pre-discipline
    history to backfill). See ARC_LIVE_AGENT_NO_LLM_SUBSTRATE for the exemplar incident (exp5054).
    """
    return (
        _inference_substrate_matches(d, ARC_LIVE_AGENT_NO_LLM_SUBSTRATE)
        or _inference_substrate_matches(d, ARC_LIVE_AGENT_ENV_INTERACTION_SUBSTRATE)
        or _inference_substrate_matches(
            d, ARC_SUBMITTED_KERNEL_OFFLINE_FROZEN_POLICY_REPLAY_SUBSTRATE
        )
    )


def _is_arc_filter_runtime_no_llm(d: dict[str, Any]) -> bool:
    """True when the artifact declares a no-new-LLM ARC live-agent filter A/B.

    Exp5609 measures already-wired runtime filters against the offline arcade live
    agent path. It may mention the frozen generator for parity, but the experiment
    must not invoke new LLM calls; this substrate keeps the same duration floor as
    the older no-LLM live-agent runtime while making the filter-specific scope
    auditable.
    """

    return _inference_substrate_matches(d, ARC_FILTER_RUNTIME_NO_LLM_SUBSTRATE)


def _is_llm_embedding_extraction(d: dict[str, Any]) -> bool:
    """True when the artifact declares live LLM embedding-extraction (no generation).

    Recognition is the canonical `inference_substrate` sentinel only (forward-only, no legacy
    schema-prefix fallback -- this class was introduced 2026-07-03, so there is no pre-discipline
    history to backfill). See LLM_EMBEDDING_EXTRACTION_SUBSTRATE for the exemplar incident (exp5178).
    """
    return _inference_substrate_matches(
        d, LLM_EMBEDDING_EXTRACTION_SUBSTRATE
    ) or _inference_substrate_matches(d, LOCAL_SOTA_FIXED_SEQUENCE_REPRESENTATION_SUBSTRATE)


def _is_log_analysis_local_timing(d: dict[str, Any]) -> bool:
    """True when the artifact declares log-analysis-plus-bounded-local-timing (no LLM call).

    Recognition is the canonical `inference_substrate` sentinel only (forward-only, no legacy
    schema-prefix fallback -- this class was introduced 2026-07-04, so there is no pre-discipline
    history to backfill). See LOG_ANALYSIS_LOCAL_TIMING_SUBSTRATE for the exemplar incident (exp5215).
    """
    return _inference_substrate_matches(d, LOG_ANALYSIS_LOCAL_TIMING_SUBSTRATE)


def _is_web_bibliographic_search_only(d: dict[str, Any]) -> bool:
    """True when the artifact declares pure web/bibliographic search, no model load.
    See WEB_BIBLIOGRAPHIC_SEARCH_ONLY_SUBSTRATE for the exemplar incidents (exp5718, exp5732)."""

    return _inference_substrate_matches(
        d, WEB_BIBLIOGRAPHIC_SEARCH_ONLY_SUBSTRATE
    ) or _inference_substrate_matches(
        d, SOURCE_RECEIPTS_METHOD_PREREGISTRATION_NO_LLM_SUBSTRATE
    ) or _inference_substrate_matches(
        d, LOW_CONCURRENCY_PRIMARY_SOURCE_INGESTION_NO_EXPERIMENTAL_LLM_SUBSTRATE
    )


def _is_artifact_qa_lint_tests(d: dict[str, Any]) -> bool:
    """True when the artifact declares fixture-driven artifact QA lint tests."""

    return _inference_substrate_matches(
        d, ARTIFACT_QA_LINT_TESTS_SUBSTRATE
    ) or _inference_substrate_matches(d, DETERMINISTIC_QA_REGRESSION_NO_LLM_SUBSTRATE)


def _is_local_sota_gguf_small_n(d: dict[str, Any]) -> bool:
    """True when the artifact declares a small-N local-SOTA-GGUF pilot (a handful of
    short generation calls or logit/logprob forward passes over a bounded fixture set,
    not full multi-hundred-token generation). See LOCAL_SOTA_GGUF_SMALL_N_SUBSTRATE for
    the exemplar incidents (exp5262, exp5263)."""

    return _inference_substrate_matches(
        d, LOCAL_SOTA_GGUF_SMALL_N_SUBSTRATE
    ) or _inference_substrate_matches(
        d, LOCAL_LLAMA_CPP_GGUF_ATOMIC_FACTOR_PROPOSALS_SUBSTRATE
    ) or _inference_substrate_matches(
        d, LOCAL_LLAMA_CPP_THREE_FAMILY_FORMAL_MUTATION_SUBSTRATE
    )


def _is_deterministic_smt_hint_validation(d: dict[str, Any]) -> bool:
    """True when the artifact declares deterministic, LLM-free SMT/solver hint validation.
    See DETERMINISTIC_SMT_HINT_VALIDATION_SUBSTRATE for the exemplar incident (exp5318)."""

    return _inference_substrate_matches(d, DETERMINISTIC_SMT_HINT_VALIDATION_SUBSTRATE)


def _is_native_gguf_backend_bisect(d: dict[str, Any]) -> bool:
    """True when the artifact declares a native-llama.cpp-CLI backend/flag bisect (stops at
    the first working combination, not a full generation benchmark). See
    NATIVE_GGUF_BACKEND_BISECT_SUBSTRATE for the exemplar incident (exp5323)."""

    return _inference_substrate_matches(d, NATIVE_GGUF_BACKEND_BISECT_SUBSTRATE)


def _descriptor_key_present(value: Any, wanted: str) -> bool:
    """True if a real artifact field named `wanted` appears outside metadata.

    Several artifacts explain required fields inside `field_principles`; those
    prose-only mentions must not count as methodology evidence. Nested result
    rows are different: exp4572 stores many real `reproduction_gate` objects in
    per-game rows, and those should count.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            if key == wanted and nested is not None:
                return True
            if _descriptor_key_present(nested, wanted):
                return True
    elif isinstance(value, list):
        return any(_descriptor_key_present(item, wanted) for item in value)
    return False


def offline_arc_methodology_descriptor(d: dict[str, Any]) -> dict[str, Any] | None:
    """Return the recognized offline ARC methodology descriptor, if present.

    Offline ARC artifacts are compute-bound in the sense that they may mention
    torch/CUDA/CNN artifacts, but their run did not invoke a live LLM. A model
    spec would be misleading there. The auditable methodology is instead:
    an offline solver/reproduce gate or verifier checkpoint plus a stable
    reproducibility checksum. This helper is deliberately gated by substrate so
    a live_llm_inference artifact cannot use these fields to avoid naming the
    model it claims to have run.
    """
    if not (_is_verifier_scoring_only(d) or _is_aggregation_only(d)):
        return None

    evidence_fields = [
        key for key in OFFLINE_ARC_METHOD_DESCRIPTOR_KEYS if _descriptor_key_present(d, key)
    ]
    if not evidence_fields or not d.get("reproducibility_checksum"):
        return None

    evidence_with_checksum = sorted(set(evidence_fields + ["reproducibility_checksum"]))
    return {
        "kind": "offline_arc_methodology_descriptor",
        "substrate": (
            VERIFIER_SCORING_SUBSTRATE if _is_verifier_scoring_only(d) else AGGREGATION_SUBSTRATE
        ),
        "evidence_fields": evidence_with_checksum,
        "reason": (
            "offline ARC solver/reproduce/checkpoint methodology; no live model_specs required"
        ),
    }


def duration_floor_for_artifact(d: dict[str, Any]) -> dict[str, Any] | None:
    """Return the duration floor selected from the artifact substrate.

    The return value is intentionally small and JSON-like so
    `summarize_artifact.py` can print it directly for reviewer-facing
    diagnostics. `None` means no compute-bound marker or floor-bearing
    substrate was declared.
    """
    if _is_precondition_check_only_blocked(d):
        return None
    classification = _classify_inference_substrate(d)
    if _is_verifier_scoring_only(d):
        cheap_floor = _cheap_learned_value_floor_descriptor(d)
        if cheap_floor is not None:
            return cheap_floor
        return {
            "substrate": VERIFIER_SCORING_SUBSTRATE,
            "min_duration_s": VERIFIER_SCORING_MIN_DURATION_S,
            "reason": "verifier_scoring",
        }
    if _is_aggregation_only(d):
        return {
            "substrate": classification["matched_value"] or AGGREGATION_SUBSTRATE,
            "min_duration_s": AGGREGATION_MIN_DURATION_S,
            "reason": "aggregation",
        }
    if _is_deterministic_verifier(d):
        return {
            "substrate": _inference_substrate_text(d) or "deterministic_verifier",
            "min_duration_s": DETERMINISTIC_VERIFIER_MIN_DURATION_S,
            "reason": "deterministic_verifier",
        }
    if _is_arc_live_agent_no_llm(d):
        return {
            "substrate": ARC_LIVE_AGENT_NO_LLM_SUBSTRATE,
            "min_duration_s": ARC_LIVE_AGENT_NO_LLM_MIN_DURATION_S,
            "reason": "arc_live_agent_no_llm",
        }
    if _is_arc_filter_runtime_no_llm(d):
        return {
            "substrate": ARC_FILTER_RUNTIME_NO_LLM_SUBSTRATE,
            "min_duration_s": ARC_LIVE_AGENT_NO_LLM_MIN_DURATION_S,
            "reason": "arc_live_agent_filter_runtime_no_llm",
        }
    if _is_llm_embedding_extraction(d):
        return {
            "substrate": LLM_EMBEDDING_EXTRACTION_SUBSTRATE,
            "min_duration_s": LLM_EMBEDDING_EXTRACTION_MIN_DURATION_S,
            "reason": "llm_embedding_extraction",
        }
    if _is_log_analysis_local_timing(d):
        return {
            "substrate": LOG_ANALYSIS_LOCAL_TIMING_SUBSTRATE,
            "min_duration_s": LOG_ANALYSIS_LOCAL_TIMING_MIN_DURATION_S,
            "reason": "log_analysis_local_timing",
        }
    if _is_artifact_qa_lint_tests(d):
        return {
            "substrate": classification["matched_value"] or ARTIFACT_QA_LINT_TESTS_SUBSTRATE,
            "min_duration_s": ARTIFACT_QA_LINT_TESTS_MIN_DURATION_S,
            "reason": "artifact_qa_lint_tests",
        }
    if _is_web_bibliographic_search_only(d):
        return {
            "substrate": classification["matched_value"] or WEB_BIBLIOGRAPHIC_SEARCH_ONLY_SUBSTRATE,
            "min_duration_s": WEB_BIBLIOGRAPHIC_SEARCH_ONLY_MIN_DURATION_S,
            "reason": "web_bibliographic_search_only",
        }
    if _is_local_sota_gguf_small_n(d):
        return {
            "substrate": LOCAL_SOTA_GGUF_SMALL_N_SUBSTRATE,
            "min_duration_s": LOCAL_SOTA_GGUF_SMALL_N_MIN_DURATION_S,
            "reason": "local_sota_gguf_small_n",
        }
    if _is_deterministic_smt_hint_validation(d):
        return {
            "substrate": DETERMINISTIC_SMT_HINT_VALIDATION_SUBSTRATE,
            "min_duration_s": DETERMINISTIC_SMT_HINT_VALIDATION_MIN_DURATION_S,
            "reason": "deterministic_smt_hint_validation",
        }
    if _is_native_gguf_backend_bisect(d):
        return {
            "substrate": NATIVE_GGUF_BACKEND_BISECT_SUBSTRATE,
            "min_duration_s": NATIVE_GGUF_BACKEND_BISECT_MIN_DURATION_S,
            "reason": "native_gguf_backend_bisect",
        }
    if classification["kind"] == SUBSTRATE_KIND_NO_LLM:
        return {
            "substrate": classification["matched_value"] or _inference_substrate_text(d),
            "min_duration_s": NO_LLM_DECLARED_MIN_DURATION_S,
            "reason": "no_llm_declared",
        }
    if _is_live_llm_inference(d):
        return {
            "substrate": LIVE_LLM_SUBSTRATE,
            "min_duration_s": COMPUTE_BOUND_MIN_DURATION_S,
            "reason": "live_model",
        }
    if _has_compute_bound_marker(d):
        return {
            "substrate": _inference_substrate_text(d) or "compute_bound_marker",
            "min_duration_s": COMPUTE_BOUND_MIN_DURATION_S,
            "reason": "live_model",
        }
    return None


def check_duration_vs_claim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Compute-bound artifact with implausibly short duration."""
    duration = d.get("duration_s")
    if not _is_finite_number(duration):
        return
    if _is_precondition_check_only_blocked(d):
        return
    classification = _classify_inference_substrate(d)
    if classification["kind"] == SUBSTRATE_KIND_AGGREGATION:
        floor = duration_floor_for_artifact(d)
        assert floor is not None
        min_duration = float(floor["min_duration_s"])
        if float(duration) < min_duration:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"aggregation substrate. Even loading upstream "
                        f"JSON takes microseconds; a value below "
                        f"{min_duration}s suggests the "
                        f"duration was not measured at all."
                    ),
                )
            )
        return
    # An artifact that EXPLICITLY declares the ARC live-agent no-LLM substrate must reach its own
    # floor check even though, by that substrate's definition, it invokes no model and so carries
    # no compute-bound marker. Before 2026-07-30 the dedicated `arc_live_agent_no_llm` branch below
    # sat BEHIND the marker guard on the next line, which made it reachable ONLY for artifacts that
    # happened to carry a VESTIGIAL GGUF/CUDA string -- exactly backwards, since that branch exists
    # precisely because such strings are vestigial for this substrate. Net effect: a CLEAN no-LLM
    # artifact (no vestigial marker) could declare any duration at all, including 0.0, and the 0.01s
    # floor never ran. Found by probe while verifying an artifact of this session that declared the
    # substrate at duration_s=0.002 -- below its own floor -- and was reported clean. The
    # declaration is an affirmative claim of real per-action environment stepping, so it is
    # self-sufficient grounds to check the floor; no marker should be required.
    # UNRECOGNIZED SUBSTRATE (2026-08-21). The paragraph above fixed this bug class for ONE
    # substrate. This generalizes it, because the shape recurs for every substrate string the
    # recognizer chain does not know: `duration_floor_for_artifact` walks a list of recognizers and
    # returns None when none match, and the guard below then returns before any floor is applied.
    # So an artifact can declare a substrate nobody taught the linter, carry any duration at all,
    # and be reported clean -- the guard's pattern list is narrower than the concept it protects.
    #
    # Measured over the 5,691-artifact corpus: 720 artifacts declare a substrate, are not
    # precondition-blocked, and receive NO floor; 237 of those ran in under one second. Found via
    # exp6478, whose `exact_solver_held_candidate_selection_no_llm` fell through every recognizer
    # while its stored duration_s was a dead constant of 0.0001.
    #
    # WARN, not CRITICAL, deliberately. Most of the 720 are honest work under a substrate nobody
    # has taught the linter yet, and a critical flag would retroactively quarantine them. The point
    # is to make an unchecked declaration VISIBLE so a floor can be added, rather than letting
    # silence read as approval. Artifacts declaring no substrate at all are a separate, larger
    # population governed by the compute-bound-marker path below and are not flagged here.
    _declared_substrate = _inference_substrate_text(d)
    if _declared_substrate and duration_floor_for_artifact(d) is None:
        flags.append(
            Flag(
                kind="SUBSTRATE_HAS_NO_DURATION_FLOOR",
                severity="warn",
                detail=(
                    f"inference_substrate={_declared_substrate!r} matches no duration-floor "
                    f"recognizer, so duration_s={duration} was NOT checked against any floor. "
                    "A declaration is an affirmative claim about what ran; an unrecognized one is "
                    "unverified, not verified. Add a recognizer plus floor for this substrate, or "
                    "correct the declaration to an existing one."
                ),
            )
        )

    _declares_arc_live_no_llm = _is_arc_live_agent_no_llm(d)
    if (
        not _has_compute_bound_marker(d)
        and not _is_live_llm_inference(d)
        and not _declares_arc_live_no_llm
    ):
        return
    floor = duration_floor_for_artifact(d)
    if floor is None:
        return
    # Verifier-scoring artifacts run in seconds because they score
    # cached candidates -- their GGUF markers are vestigial. Apply
    # the tighter verifier-scoring threshold instead of the full
    # model-inference threshold.
    if floor["reason"] == "cheap_learned_value_scoring":
        min_duration = float(floor["min_duration_s"])
        if float(duration) < min_duration:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"cheap learned-value/CNN/linear scoring substrate. "
                        f"Even loading cached candidates takes >= "
                        f"{min_duration}s; this duration suggests it "
                        f"was not measured at all."
                    ),
                )
            )
        return
    if floor["reason"] == "verifier_scoring":
        min_duration = float(floor["min_duration_s"])
        if float(duration) < min_duration:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"verifier-scoring substrate. Even verifier "
                        f"scoring of a few hundred candidates takes "
                        f">={min_duration}s; this "
                        f"completed too fast to have scored anything."
                    ),
                )
            )
        return
    # Aggregation-only artifacts (capstones, archive/activate,
    # paper-table builders) just read upstream JSON and arithmetic.
    # Milliseconds are honest. The GGUF markers are inherited from
    # the upstream artifacts they cite, not invoked here.
    if floor["reason"] == "aggregation":
        min_duration = float(floor["min_duration_s"])
        if float(duration) < min_duration:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"aggregation substrate. Even loading upstream "
                        f"JSON takes microseconds; a value below "
                        f"{min_duration}s suggests the "
                        f"duration was not measured at all."
                    ),
                )
            )
        return
    if floor["reason"] == "deterministic_verifier":
        min_duration = float(floor["min_duration_s"])
        if float(duration) < min_duration:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"deterministic-verifier substrate. Even loading "
                        f"checked-in JSON takes microseconds; a value below "
                        f"{min_duration}s suggests "
                        f"the duration was not measured at all."
                    ),
                )
            )
        return
    # ARC live-agent runtime self-discovery (no LLM invoked). Real per-action
    # offline-env-stepping compute, no GPU/model load. GGUF strings in a
    # nested model_specs (naming the generator that WOULD fire if the LLM
    # tier were used) are vestigial, not a live-inference claim.
    if floor["reason"] == "arc_live_agent_no_llm":
        min_duration = float(floor["min_duration_s"])
        if float(duration) < min_duration:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"ARC live-agent no-LLM substrate. Even a single "
                        f"real environment action takes non-zero time; a "
                        f"value below {min_duration}s suggests "
                        f"the duration was not measured at all."
                    ),
                )
            )
        return
    min_duration = float(floor["min_duration_s"])
    if float(duration) < min_duration:
        flags.append(
            Flag(
                kind="DURATION_TOO_SHORT",
                severity="critical",
                detail=(
                    f"duration_s={duration} but artifact references "
                    f"compute-bound markers (GGUF / CUDA / live model). "
                    f"Loading and running a real model takes "
                    f">={min_duration}s minimum; this "
                    f"completed too fast to have invoked the model."
                ),
            )
        )


def check_sample_size(d: dict[str, Any], flags: list[Flag]) -> None:
    """Distributional claims with sample size below statistical threshold."""
    n = d.get("n_samples") or d.get("n_examples")
    if n is None:
        return
    try:
        n = int(n)
    except (TypeError, ValueError):
        return

    has_kl = "kl_divergence" in d or "kl" in d
    has_ks = "ks_p_value" in d or "ks_statistic" in d
    has_dist_mean = any(k.startswith("mean_") and "delta" in k for k in d)

    n_spins = d.get("n_spins")
    if has_kl or has_ks or has_dist_mean:
        # Heuristic: for an n_spins=N substrate, you need many more
        # samples than 2^(N/k) for some k. At minimum, 10k samples
        # for any non-toy claim.
        min_required = 1000
        if isinstance(n_spins, (int, float)) and n_spins >= 32:
            min_required = max(min_required, int(10 * n_spins))
        if isinstance(n_spins, (int, float)) and n_spins >= 64:
            min_required = max(min_required, 10000)
        if n < min_required:
            flags.append(
                Flag(
                    kind="SAMPLE_SIZE_BELOW_CLAIM",
                    severity="warn",
                    detail=(
                        f"n_samples={n} below statistical threshold "
                        f"~{min_required} for distributional claims "
                        f"(KL/KS/mean delta) at n_spins={n_spins!r}. "
                        f"KL is dominated by sample-size noise; ks_p "
                        f"can pass spuriously on small N."
                    ),
                )
            )


def check_gate_passed_without_data(d: dict[str, Any], flags: list[Flag]) -> None:
    """acceptance_gate_passed=true but key metric fields are null/missing/0."""
    if d.get("acceptance_gate_passed") is not True:
        return
    # Look for keys mentioned in the gate definitions: anything with
    # "gate", "threshold", "delta", "rate", "score". If any is null/0,
    # flag.
    suspect = []
    for k, v in d.items():
        kl = k.lower()
        if any(s in kl for s in ("threshold", "delta", "rate", "score", "lift", "ratio")):
            if v is None:
                suspect.append(f"{k}=null")
            elif _is_finite_number(v) and float(v) == 0.0:
                suspect.append(f"{k}=0")
    if suspect:
        flags.append(
            Flag(
                kind="GATE_PASSED_WITHOUT_DATA",
                severity="critical",
                detail=(
                    f"acceptance_gate_passed=true but these fields are "
                    f"null/zero: {', '.join(suspect[:6])}"
                ),
            )
        )


# Substrates whose artifacts must record WHICH preconditions were verified
# before launch (REQ-CONDUCTOR-PRECOND-1). These are the substrates where the
# Pre-Launch Preconditions Discipline's fabrication mode lives: the agent
# silently lacked the model/board/credential and synthesized a result.
_PRECONDITIONS_REQUIRED_SUBSTRATES: tuple[str, ...] = (
    "live_llm_inference",
    "live_llm_embedding_extraction",
    "hardware_smoke",
)


def check_preconditions_declared(d: dict[str, Any], flags: list[Flag]) -> None:
    """WARN when a compute-bound artifact lacks `preconditions_checked`.

    The extension the Pre-Launch Preconditions Discipline's own prose
    requested ("the METHODOLOGY_MISSING detector should be extended to
    flag any compute-bound artifact that lacks a preconditions_checked
    field") and never got — Conversion 5 of
    docs/research-notes/cumulative-coherence-rule-to-check-2026-08-21.md.

    Presence-only, WARN-only: the field's CONTENT is task-specific and
    unlintable, and a backfill dry-run decides whether the severity may
    ever rise. Keyed on the three explicitly-declared compute substrates
    rather than marker inference, so aggregation / verifier-scoring /
    no-LLM artifacts never draw a category-error flag.
    """
    raw = _inference_substrate_text(d)
    if _match_declared_substrate(raw, _PRECONDITIONS_REQUIRED_SUBSTRATES) is None:
        return
    # Presence check only. Any shape counts (list of {resource, available},
    # principle-wrapped, prose) — the discipline requires the field on
    # success AND on blocked_* artifacts alike, so there is no exemption.
    if d.get("preconditions_checked") is not None:
        return
    flags.append(
        Flag(
            kind="PRECONDITIONS_UNDECLARED",
            severity="warn",
            detail=(
                f"substrate {raw!r} declares real compute but the artifact has no "
                "preconditions_checked field. The Pre-Launch Preconditions "
                "Discipline requires recording WHICH resources were verified "
                "before launch — its absence is the shape every confirmed "
                "fabrication shared (exp1851, exp1680)."
            ),
        )
    )


def check_methodology_present(d: dict[str, Any], flags: list[Flag]) -> None:
    """Compute-bound artifact missing methodology evidence."""
    if _classify_inference_substrate(d)["kind"] == SUBSTRATE_KIND_AGGREGATION:
        return
    if not _has_compute_bound_marker(d) and not _is_live_llm_inference(d):
        return
    # An honest blocked_* verdict (Pre-Launch Preconditions Discipline) means no
    # measurement happened at all -- requiring random_seed/reproducibility_checksum
    # here is the same category error as requiring them for the duration floor.
    if _is_precondition_check_only_blocked(d):
        return
    if offline_arc_methodology_descriptor(d) is not None:
        return
    # Aggregation-only artifacts inherit methodology from the upstream
    # sources they cite; they aren't themselves a measurement, so this
    # check would be a category error.
    if _is_aggregation_only(d):
        return
    # Deterministic verifier/audit artifacts cite upstream model strings while
    # reading checked-in evidence; requiring fresh model_specs/random_seed would
    # confuse provenance audit with model inference.
    if _is_deterministic_verifier(d):
        return
    # ARC live-agent no-LLM artifacts (and log-analysis-plus-local-timing gates, added
    # 2026-07-04 alongside the substrate itself) genuinely have no model to name (no LLM
    # was invoked) -- requiring model_specs/target_model here would be a category error
    # identical to the aggregation case. random_seed and reproducibility_checksum are
    # STILL required (this substrate is a real measurement, just not a model-inference one).
    if _is_artifact_qa_lint_tests(d):
        return
    no_model_spec_required = (
        _classify_inference_substrate(d)["kind"] == SUBSTRATE_KIND_NO_LLM
        or _is_arc_live_agent_no_llm(d)
        or _is_log_analysis_local_timing(d)
        or _is_deterministic_smt_hint_validation(d)
        or _inference_substrate_matches(
            d, DETERMINISTIC_RUNTIME_RECEIPT_VALIDATION_NO_LLM_SUBSTRATE
        )
        or _inference_substrate_matches(d, DETERMINISTIC_REPRESENTATION_CONTRACT_NO_LLM_SUBSTRATE)
    )
    # Some experiment scripts (24 in the corpus as of 2026-07-05, e.g. exp5262/exp5263) declare
    # the field as the uppercase Python-constant-style "MODEL_SPECS" rather than lowercase
    # "model_specs" -- both are legitimate, so recognize either casing.
    has_model_spec = no_model_spec_required or (
        d.get("model_specs")
        or d.get("MODEL_SPECS")
        or d.get("target_model")
        or d.get("models_tested")
    )
    # Recognize singular (`random_seed`, `seed`) and plural
    # (`random_seeds_used`, `seeds`) forms. Multi-seed experiments
    # legitimately use the plural list-of-seeds form.
    has_seed = (
        d.get("random_seed") is not None
        or d.get("seed") is not None
        or d.get("random_seeds_used")
        or d.get("seeds")
    )
    has_repro = d.get("reproducibility_checksum")
    # duration_s (added 2026-07-30). `check_duration_vs_claim` returns immediately when
    # duration_s is absent or non-finite, so a compute-bound artifact that simply OMITS the
    # field sails past the entire DURATION_TOO_SHORT family in silence. Verified directly on
    # this session's composite pre-flight artifact: injecting duration_s=1.0 or 0.0001 fires
    # CRITICAL DURATION_TOO_SHORT, while duration_s=null passes clean -- so "adversarial_verify
    # clean" was not evidence of duration plausibility, because the field the check is DEFINED
    # on was absent. That is strictly worse than a short duration: a fabricator who omits the
    # field is invisible to the check built to catch them, while an honest one who records 35s
    # gets flagged. CLAUDE.md's Adversarial Artifact Verification rule lists duration_s as the
    # load-bearing fabrication-detection signal precisely because real compute takes wall-clock
    # time, so its ABSENCE on a compute-bound artifact is a methodology gap in exactly the same
    # sense as a missing seed. Reported at `warn`, alongside the other three, rather than as a
    # critical: the corpus predates the requirement and quarantining historical artifacts for a
    # newly-added field would be a retroactive gate, not a fabrication finding.
    has_duration = _is_finite_number(d.get("duration_s"))
    missing = []
    if not has_model_spec:
        missing.append("model_specs/target_model")
    if not has_seed:
        missing.append("random_seed")
    if not has_repro:
        missing.append("reproducibility_checksum")
    if not has_duration:
        missing.append("duration_s")
    if missing:
        flags.append(
            Flag(
                kind="METHODOLOGY_MISSING",
                severity="warn",
                detail=(
                    f"Compute-bound artifact missing: "
                    f"{', '.join(missing)}. Methodology unverifiable."
                ),
            )
        )


def check_implausible_tight_ci(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect bootstrap/empirical CIs much tighter than sample-variance floor.

    The minimum CI half-width on a stochastic estimator at N seeds is
    approximately sigma_est / sqrt(N), where sigma_est is the metric's
    natural sample stddev (heuristically ~ 0.1-0.5 of the metric value
    for a [0,1]-scale measurement, or ~ 0.2 on dimensionless deltas).

    A reported 95% CI whose total width is below 0.1 / sqrt(N) on a
    metric near 0.1+ is at least an order of magnitude tighter than
    the variance floor — that signals either a deterministic-by-
    construction invariant (which must be disclosed) or a bootstrap
    bug (e.g., one seed reshuffled N times).

    This rule was added 2026-05-15 after exp1693 (`.171 Phase 4
    delta_alpha at n=64) reported [0.15040, 0.15070] CI at N=30
    seeds — 100x tighter than the floor. The literature finding
    (arXiv:2512.15605 AR-LM↔EBM bijection) provides a plausible
    explanation (alpha_t is bijection-invariant by construction),
    but that finding came AFTER the measurement; this linter would
    have flagged it at task-completion-time.
    """
    # Look for fields named *_bootstrap_ci_95, *_ci, *_confidence_interval
    # whose value is a 2-element list [lower, upper].
    ci_field_suffixes = (
        "_bootstrap_ci_95",
        "_bootstrap_ci_90",
        "_bootstrap_ci",
        "_ci_95",
        "_ci_90",
        "_ci",
        "_confidence_interval",
        "_credible_interval",
    )

    n_seeds_raw = d.get("n_seeds") or d.get("seeds") or 0
    try:
        n_seeds = int(n_seeds_raw)
    except (TypeError, ValueError):
        n_seeds = 0

    # If we don't know N, we can't bound the floor — fall back to a
    # conservative absolute-width gate at 1e-3 on any reported CI.
    # That catches the exp1693 pattern even if n_seeds is absent.
    if n_seeds <= 0:
        # Still flag absurdly-tight CIs even without N — width < 1e-3
        # on a metric in the [0, 1] range is implausible.
        n_seeds_for_floor = 1000  # treat as "even at N=1000 this is tight"
    else:
        n_seeds_for_floor = n_seeds

    # Heuristic variance floor: assume sigma_est >= 0.05 on the metric
    # scale (very forgiving — most real measurements have larger
    # variance than this).
    assumed_sigma = 0.05
    floor_half_width = assumed_sigma / math.sqrt(n_seeds_for_floor)
    # 95% CI half-width is ~1.96 * sigma / sqrt(N), so full width
    # floor is ~3.92 * sigma / sqrt(N).
    floor_full_width = 3.92 * floor_half_width
    # Discount the floor by 10x to avoid false positives on
    # genuinely-well-converged measurements.
    flagging_threshold = floor_full_width / 10.0

    for k, v in d.items():
        if not isinstance(v, list) or len(v) != 2:
            continue
        if not all(_is_finite_number(x) for x in v):
            continue
        if not any(k.lower().endswith(s) for s in ci_field_suffixes):
            continue
        lo, hi = float(v[0]), float(v[1])
        if hi <= lo:
            continue  # invalid CI, ignore
        ci_width = hi - lo
        # Skip CIs whose metric scale is itself tiny (avoid false
        # positives on metrics that genuinely live near zero, like
        # variance or KL on near-identical distributions).
        midpoint = abs((lo + hi) / 2.0)
        if midpoint < 0.01:
            continue
        if ci_width < flagging_threshold:
            n_seeds_msg = f"n_seeds={n_seeds}" if n_seeds > 0 else "n_seeds unknown"
            flags.append(
                Flag(
                    kind="IMPLAUSIBLE_TIGHT_CI",
                    severity="warn",
                    detail=(
                        f"{k}={v} has CI width {ci_width:.2g} on midpoint "
                        f"{midpoint:.3g} ({n_seeds_msg}). Sample-variance "
                        f"floor at sigma>=0.05 and N>={n_seeds_for_floor} "
                        f"is ~{floor_full_width:.2g}; observed CI is "
                        f"{floor_full_width / max(ci_width, 1e-12):.0f}x tighter. "
                        f"Likely deterministic-by-construction (should be "
                        f"disclosed in methodology_note) OR bootstrap bug "
                        f"(e.g., one seed reshuffled instead of N independent "
                        f"draws)."
                    ),
                )
            )


def _flatten_metrics(d: dict[str, Any]) -> dict[str, Any]:
    """Flatten common nested-metric dicts (`metrics`, `report`,
    `summary`, etc.) up to the top level so the checks see them.
    Preserves top-level keys; nested keys merged with their parent
    name stripped (e.g. `metrics.initial_energy` -> `initial_energy`).
    """
    out: dict[str, Any] = dict(d)
    nested_containers = ("metrics", "report", "summary", "case_scores", "results")
    for container in nested_containers:
        v = d.get(container)
        if isinstance(v, dict):
            for k, sub in v.items():
                if k not in out:
                    out[k] = sub
    return out


def _normalize_principle_wrapped_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Unwrap top-level fields written as ``{"value": ..., "principle": "..."}``
    (CLAUDE.md "Principle-Annotated Artifact Fields" discipline) to their bare
    ``value`` so every downstream check sees the field the way it would if the
    author had written it bare.

    Origin: 2026-07-02, exp5161's GAP-4 pilot was HARD-flagged DURATION_TOO_SHORT
    for a reason that had nothing to do with its actual duration: its
    `inference_substrate` was `{"principle": "...", "value":
    "verifier_ensemble_against_cached_candidates"}`, and `_inference_substrate_text()`
    did `str(d.get("inference_substrate"))`, which stringifies a DICT to a Python
    repr matching no canonical substrate string -- so the duration-floor check
    fell through to the generic 60s compute-bound fallback regardless of what
    substrate was actually declared. A corpus-wide scan found this is not a
    one-off: 176 artifacts wrap `inference_substrate` this way, plus smaller
    counts for `honest_verdict` (9), `duration_s` (12), `random_seed` (14), and
    `reproducibility_checksum` (14) -- every one of those fields is read via a
    bare `d.get(...)` somewhere in this file's checks, so all of them were
    exposed to the same class of silent misbehavior (a verdict that never
    matches its terminal-prefix, a duration check that silently no-ops, etc).

    Fixing every individual read site is exactly the kind of whack-a-mole this
    bug already demonstrates is unreliable (the ORIGINAL author of exp5161 did
    write a `{principle, value}` dict following an established, documented
    convention -- it just didn't occur to anyone that `_inference_substrate_text`
    wouldn't understand it). Normalizing once, immediately after `_flatten_metrics`
    and before any check runs, means every existing and future check
    automatically sees the bare value with zero further changes required.

    Deliberately conservative: only unwraps a dict that has BOTH `value` and
    `principle` keys (not just `value` alone, which could coincidentally be
    real domain data for some other field), and only at the top level (does not
    recurse into nested structures like `model_specs`, `pilot_rows`, etc. --
    those have their own, different internal shapes).
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict) and "value" in v and "principle" in v:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


_MOAT_HEADLINE_MARKERS = (
    "moat_won",
    "moat_proven",
    "moat proven",
    "moat-proven",
    "efficiency_moat_won",
    "verifier_value_added_true",
    "verifier_efficiency_win",
)
MOAT_CLAIM_RIGOR_KIND = "MOAT_CLAIM_RIGOR"
MOAT_HEADROOM_MIN_DELTA = 0.10
_MOAT_RIGOR_CLAIM_KEYS = (
    "honest_verdict",
    "headline_outcome",
    "headline",
    "paper_summary",
    "decision",
    "oracle_distinct_status",
)
_MOAT_RIGOR_RELEVANCE_MARKERS = (
    "beats_sc",
    "beats_tuned_sc",
    "beats_naive_sc",
    "beats_self_consistency",
    "beat_self_consistency",
    "does_not_beat_sc",
    "does_not_beat_self_consistency",
    "moat_realized",
    "moat_retired",
    "moat_won",
    "moat_proven",
    "success_moat",
    "verifier_moat",
    "verifier_value_added",
)
_MOAT_RIGOR_WIN_MARKERS = (
    "beats_sc",
    "beats_tuned_sc",
    "beats_naive_sc",
    "beats_self_consistency",
    "beat_self_consistency",
    "moat_realized",
    "moat_won",
    "success_moat",
    "success_verifier_moat",
    "verifier_value_added_true",
)
_MOAT_RIGOR_NULL_MARKERS = (
    "does_not_beat_sc",
    "does_not_beat_self_consistency",
    "not_beat_sc",
    "not_beat_self_consistency",
    "moat_retired",
    "retired_bounded",
    "ci_incl_0",
)


def _claims_moat(d: dict[str, Any]) -> bool:
    """True if the artifact headlines a verifier moat / superiority win."""
    if d.get("verifier_value_added") is True or d.get("verifier_efficiency_win") is True:
        return True
    for key in ("honest_verdict", "headline_outcome", "headline"):
        v = d.get(key)
        if isinstance(v, str) and any(m in v.lower() for m in _MOAT_HEADLINE_MARKERS):
            return True
    return False


_GATE_MET_RE = re.compile(r"(?<![a-z0-9])gate_met(?![a-z0-9])")
_DIFFUSIONGEMMA_MET_RE = re.compile(r"diffusiongemma_met(?![a-z0-9])")


def _flips_gate(d: dict[str, Any]) -> bool:
    """True if the artifact asserts a (DiffusionGemma) gate is MET / flipped.

    2026-07-02: exp5173's honest_verdict `blocked_diffusiongemma_meta_tensor_bug_
    unresolved` false-triggered CIRCULAR_MOAT_OVERCLAIM -- a plain `"diffusiongemma_met"
    in hv.lower()` substring check matched inside "diffusiongemma_META_tensor" (a
    PyTorch technical term for placeholder tensors, unrelated to a gate being MET). The
    artifact was honestly BLOCKED (nothing ran: empty arm_rows, all-zero pass@1s) yet
    got flagged as if it had flipped a circular gate. Fixed with negative-lookahead
    word-boundary regexes so "met" must not be immediately followed by another
    lowercase letter/digit (excludes meta/method/metric/metadata/etc colliding with
    "met"), while still matching the genuine word "met" followed by "_" or end-of-string.
    """
    v = d.get("diffusiongemma_gate_status")
    if isinstance(v, str) and v.strip().upper() == "MET":
        return True
    g = d.get("diffusiongemma_gate")
    if isinstance(g, dict) and (
        g.get("met") is True or str(g.get("status", "")).strip().upper() == "MET"
    ):
        return True
    hv = d.get("honest_verdict")
    if isinstance(hv, str):
        hv_lower = hv.lower()
        if _GATE_MET_RE.search(hv_lower) or _DIFFUSIONGEMMA_MET_RE.search(hv_lower):
            return True
    return False


def _moat_rigor_claim_text(d: dict[str, Any]) -> str:
    return " ".join(str(d.get(key, "")) for key in _MOAT_RIGOR_CLAIM_KEYS).lower()


def _moat_rigor_norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _moat_rigor_real_fields(d: dict[str, Any]) -> list[tuple[tuple[str, ...], Any]]:
    return _iter_real_fields(d)


def _moat_rigor_numeric_items(d: dict[str, Any]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for path, value in _moat_rigor_real_fields(d):
        if _is_finite_number(value):
            out.append((_path_text(path).lower(), float(value)))
    return out


def _moat_rigor_positive_delta_items(d: dict[str, Any]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for path_text, value in _moat_rigor_numeric_items(d):
        leaf = path_text.rsplit(".", 1)[-1]
        if "delta" not in leaf:
            continue
        if "sc" not in leaf and "self_consistency" not in leaf:
            continue
        if value > 0.0:
            out.append((path_text, value))
    return out


def _moat_rigor_headroom_state(d: dict[str, Any]) -> tuple[bool, str]:
    headroom_declared = False
    for path, value in _moat_rigor_real_fields(d):
        if path and path[-1].lower() == "headroom_present" and value is True:
            headroom_declared = True
            break
    if not headroom_declared:
        return False, "headroom_present is not declared true"

    numeric = _moat_rigor_numeric_items(d)
    oracle_values = [
        value
        for path_text, value in numeric
        if "oracle_at_k" in path_text or "oracle@k" in path_text
    ]
    tuned_sc_values = [
        value
        for path_text, value in numeric
        if ("tuned_sc" in path_text or "tuned_self_consistency" in path_text)
        and "delta" not in path_text
    ]
    if oracle_values and tuned_sc_values:
        best_headroom = max(o - s for o in oracle_values for s in tuned_sc_values)
        if best_headroom < MOAT_HEADROOM_MIN_DELTA:
            return (
                False,
                f"oracle@K - tuned_sc headroom is {best_headroom:.3g}, "
                f"below {MOAT_HEADROOM_MIN_DELTA:.2f}",
            )

    flip_values = [value for path_text, value in numeric if "flip" in path_text and value >= 0.0]
    if flip_values and max(flip_values) <= 0.0:
        return False, "flips evidence is zero"

    return True, "headroom_present=true with no contradictory headroom evidence"


def _moat_rigor_claims_relevant(d: dict[str, Any]) -> bool:
    if (
        d.get("verifier_value_added") is True
        or d.get("verifier_efficiency_win") is True
        or d.get("moat_realized") is True
        or d.get("moat_retired_bounded") is True
    ):
        return True
    if _claims_moat(d) or _flips_gate(d):
        return True
    norm = _moat_rigor_norm(_moat_rigor_claim_text(d))
    return any(marker in norm for marker in _MOAT_RIGOR_RELEVANCE_MARKERS)


def _moat_rigor_claims_win(d: dict[str, Any]) -> bool:
    if not _moat_rigor_positive_delta_items(d):
        return False
    norm = _moat_rigor_norm(_moat_rigor_claim_text(d))
    if d.get("moat_realized") is True or d.get("verifier_value_added") is True:
        return True
    return any(marker in norm for marker in _MOAT_RIGOR_WIN_MARKERS)


def _moat_rigor_claims_null(d: dict[str, Any]) -> bool:
    if d.get("moat_retired_bounded") is True:
        return True
    norm = _moat_rigor_norm(_moat_rigor_claim_text(d))
    return any(marker in norm for marker in _MOAT_RIGOR_NULL_MARKERS)


def _moat_rigor_uses_naive_sc(d: dict[str, Any]) -> bool:
    field_text = _field_name_text(d)
    claim_text = _moat_rigor_claim_text(d)
    combined = f"{claim_text} {field_text}"
    norm = _moat_rigor_norm(combined)
    has_tuned = "tuned_sc" in norm or "tuned_self_consistency" in norm
    has_naive = "naive_sc" in norm or "naive_self_consistency" in norm

    for path, value in _moat_rigor_real_fields(d):
        leaf = path[-1].lower() if path else ""
        path_text = _path_text(path).lower()
        if isinstance(value, str):
            value_norm = _moat_rigor_norm(value)
            if "naive_sc" in value_norm or "naive_self_consistency" in value_norm:
                has_naive = True
            if "tuned_sc" in value_norm or "tuned_self_consistency" in value_norm:
                has_tuned = True
        if leaf in {
            "self_consistency_accuracy",
            "self_consistency_baseline",
            "delta_vs_sc",
            "sc_accuracy",
        }:
            has_naive = True
        if "naive" in path_text and ("sc" in path_text or "self_consistency" in path_text):
            has_naive = True
        if "tuned_sc" in path_text or "tuned_self_consistency" in path_text:
            has_tuned = True

    return has_naive and not has_tuned


def _moat_rigor_has_paired_significance(d: dict[str, Any]) -> bool:
    has_ci = False
    has_mcnemar = False
    for path, value in _moat_rigor_real_fields(d):
        leaf = path[-1].lower() if path else ""
        if ("paired_ci95" in leaf or "paired_ci_95" in leaf) and (
            isinstance(value, list)
            and len(value) == 2
            and all(_is_finite_number(item) for item in value)
        ):
            has_ci = True
        if "mcnemar_p" in leaf and _is_finite_number(value):
            has_mcnemar = True
    return has_ci and has_mcnemar


def check_moat_claim_rigor(d: dict[str, Any], flags: list[Flag]) -> None:
    """Enforce oracle-distinct, headroom-controlled moat claim rigor.

    This complements `check_circular_moat_overclaim` and
    `FALSE_NEGATIVE_RISK`: it guards artifacts that try to headline a
    verifier-moat / beats-SC / verifier-value-added claim without the full
    anti-trap contract of oracle distinctness, tuned-SC baseline, headroom, and
    paired significance.
    """
    if not _moat_rigor_claims_relevant(d):
        return

    verifier_is_oracle = d.get("verifier_is_oracle")
    if verifier_is_oracle is not False:
        flags.append(
            Flag(
                kind=MOAT_CLAIM_RIGOR_KIND,
                severity="critical",
                detail=(
                    f"Moat/verifier-value claim has verifier_is_oracle={verifier_is_oracle!r}. "
                    "Every verifier-moat / beats-SC / verifier_value_added claim must declare "
                    "verifier_is_oracle=False; missing or true is circular and headline-ineligible."
                ),
            )
        )

    if _moat_rigor_uses_naive_sc(d):
        flags.append(
            Flag(
                kind=MOAT_CLAIM_RIGOR_KIND,
                severity="warn",
                detail=(
                    "Moat claim compares against naive self-consistency rather than a TUNED-SC "
                    "baseline. Add tuned_sc_accuracy / tuned_self_consistency evidence before "
                    "using the result as a headroom-controlled moat claim."
                ),
            )
        )

    is_win = _moat_rigor_claims_win(d)
    headroom_ok, headroom_detail = _moat_rigor_headroom_state(d)
    if is_win and not headroom_ok:
        delta_detail = ", ".join(
            f"{path}={value:g}" for path, value in _moat_rigor_positive_delta_items(d)
        )
        flags.append(
            Flag(
                kind=MOAT_CLAIM_RIGOR_KIND,
                severity="critical",
                detail=(
                    "Positive beats-SC/moat win lacks a valid headroom_present=True anchor "
                    f"({headroom_detail}; positive deltas: {delta_detail or 'unavailable'}). "
                    "A win needs oracle@K - tuned_sc >= 0.10 and flips>0 when those fields "
                    "are present."
                ),
            )
        )

    if is_win and not _moat_rigor_has_paired_significance(d):
        flags.append(
            Flag(
                kind=MOAT_CLAIM_RIGOR_KIND,
                severity="critical",
                detail=(
                    "Positive beats-SC/moat win lacks paired significance evidence: require "
                    "paired_ci95 and mcnemar_p before a delta>0 headline can be cited."
                ),
            )
        )

    if _moat_rigor_claims_null(d) and not headroom_ok:
        flags.append(
            Flag(
                kind=MOAT_CLAIM_RIGOR_KIND,
                severity="warn",
                detail=(
                    f"Null/moat-retirement claim is uninformative because {headroom_detail}. "
                    "A no-headroom corpus cannot support 'does not beat SC' or 'moat retired' "
                    "as a moat bound."
                ),
            )
        )


def check_circular_moat_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Circularity / oracle-distinctness guard (2026-06-14, operator-directed).

    A verifier "moat" / efficiency win is a NON-CIRCULAR claim only when the
    verifier is INDEPENDENT of the executable oracle that defines correctness.
    Where the verifier IS the oracle (run the unit tests on HumanEval,
    check_sudoku_validity, etc.), beating self-consistency or an LLM-judge is
    true-but-circular: it does NOT show a learned/energy verifier adds value, and
    it must NOT headline a moat or flip a gate. Every moat/gate claim must declare
    ``verifier_is_oracle: bool``.

    Origin: the .387/.388 capstones over-claimed the code/efficiency win as
    "moat proven" and flipped the DiffusionGemma gate to MET on a circular
    (verifier==oracle) result; only the operator caught it, twice. This turns that
    manual catch into a mechanical guard. See CLAUDE.md "Circularity /
    Oracle-Distinctness Discipline".
    """
    if not (_claims_moat(d) or _flips_gate(d)):
        return
    vio = d.get("verifier_is_oracle")
    if _flips_gate(d) and vio is not False:
        # The precise over-claim: headlining a gate-flip while the verifier is
        # the oracle (or undeclared). CRITICAL -> the artifact may not flip the
        # gate; a circular result is not gate-eligible.
        flags.append(
            Flag(
                kind="CIRCULAR_MOAT_OVERCLAIM",
                severity="critical",
                detail=(
                    f"Artifact flips a gate/headlines a moat but verifier_is_oracle={vio!r} "
                    "(not False). A circular or oracle-undeclared verifier cannot flip a gate "
                    "or headline 'moat proven' -- only an oracle-DISTINCT (learned/energy) win "
                    "is headline/gate-eligible. Declare verifier_is_oracle=False and use an "
                    "oracle-distinct verifier, or do not flip the gate."
                ),
            )
        )
    elif vio is None:
        flags.append(
            Flag(
                kind="CIRCULAR_MOAT_OVERCLAIM",
                severity="warn",
                detail=(
                    "Verifier-value/moat claim does not declare verifier_is_oracle. Is the "
                    "verifier the SAME executable oracle that defines correctness (circular, "
                    "NOT headline-eligible) or oracle-distinct (learned/energy, headline-"
                    "eligible)? Declare verifier_is_oracle: bool."
                ),
            )
        )
    elif vio is True:
        flags.append(
            Flag(
                kind="CIRCULAR_MOAT_OVERCLAIM",
                severity="warn",
                detail=(
                    "verifier_is_oracle=True: EXECUTION-GROUNDED CIRCULAR win (the verifier IS "
                    "the oracle). Valid as a result, but NOT headline-eligible as 'moat proven' "
                    "and may not flip a gate -- the non-circular claim (oracle-distinct "
                    "learned/energy verifier) remains the open frontier."
                ),
            )
        )


# --- Declared verdict class, cross-checked structurally (2026-08-21) ---------------------------------
# The CLOSED enum. Closed on purpose: verdict semantics previously lived in four
# drifting substring token lists, patched at least six times ('disqualified:' —
# an honest negative — drew a critical flag in the week this shipped). An open
# vocabulary here would just be list number five.
_VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)


def check_verdict_class_consistency(d: dict[str, Any], flags: list[Flag]) -> None:
    """Cross-check a declared verdict_class against structural fields (REQ-CONDUCTOR-VERDICT-1).

    Mechanism 4 of docs/research-notes/conductor-self-improvement-2026-08-21.md.
    Declaration replaces inference: the artifact says which CLASS its verdict is,
    and this check verifies the claim against fields the linter already reads —
    no substring matching against honest_verdict, ever.

    - a value outside the closed enum is CRITICAL;
    - verifier_is_oracle=True forbids `positive`: the class must be
      `circular_positive`, so a declared-circular result cannot carry a
      research-win class into capstones/headlines (the exp6478 shape: an honest
      artifact whose `complete_positive` prefix connoted a win downstream);
    - a failed acceptance_gate_* self-report forbids `positive` and
      `circular_positive`;
    - the informative-`null` cross-check (flip_count == 0 forbids a meaningful
      null) is NOT duplicated here — check_false_negative_risk already
      implements it structurally.

    Absent verdict_class draws no flag: adoption is forward-only via the
    planner prompt, and the historical corpus carries only free-text verdicts.
    """
    vc = d.get("verdict_class")
    if vc is None:
        return
    if not isinstance(vc, str) or vc not in _VERDICT_CLASSES:
        flags.append(
            Flag(
                kind="VERDICT_CLASS_MISMATCH",
                severity="critical",
                detail=(
                    f"verdict_class={vc!r} is outside the closed enum "
                    f"{sorted(_VERDICT_CLASSES)}. The enum is closed on purpose — "
                    "extending it per-artifact recreates the drifting-token-list "
                    "problem this field replaces. Pick the nearest class and put "
                    "the nuance in honest_verdict."
                ),
            )
        )
        return
    if vc == "positive" and d.get("verifier_is_oracle") is True:
        flags.append(
            Flag(
                kind="VERDICT_CLASS_MISMATCH",
                severity="critical",
                detail=(
                    "verdict_class=positive but verifier_is_oracle=True: a circular "
                    "(verifier==oracle) result must declare circular_positive so the "
                    "circularity travels WITH the claim into capstones and headline "
                    "aggregation. See CLAUDE.md 'Circularity / Oracle-Distinctness "
                    "Discipline'."
                ),
            )
        )
    if vc in ("positive", "circular_positive"):
        failed_gates = [
            k
            for k, v in d.items()
            if isinstance(k, str) and k.startswith("acceptance_gate") and v is False
        ]
        if failed_gates:
            flags.append(
                Flag(
                    kind="VERDICT_CLASS_MISMATCH",
                    severity="critical",
                    detail=(
                        f"verdict_class={vc} but failed acceptance-gate self-report(s) "
                        f"{failed_gates[:4]}: a failed gate forbids a positive class. "
                        "Declare null/partial/blocked as appropriate."
                    ),
                )
            )


# --- ARC live-agent self-solve discipline (2026-06-22, operator-directed; 2nd recurrence) ------------
_ARC_OUTER_LOOP_INPUT_FLAGS = (
    "used_env_source",
    "read_game_source",
    "offline_ground_truth_bfs",
    "exhaustive_bfs_calibration",
    "hand_calibrated_per_game",
)
_ARC_VALID_PROVENANCE = {"live_agent_self_discovery", "development_proxy", "outer_loop_re"}


def _arc_claimed_level(d: dict[str, Any]) -> int | None:
    best: int | None = None
    for k in ("reproduced_levels", "reached_level", "levels_completed"):
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            best = max(best if best is not None else 0, int(v))
    return best


def _is_arc_solve_claim(d: dict[str, Any]) -> bool:
    """Narrowly: an artifact claiming the agent SOLVED a hidden-game level -- offline_reproduced True AND
    a numeric level >= 1 AND a game id. Verifier-eval / oracle-distinct / aggregator ARC artifacts do
    NOT have this shape, so this check leaves them untouched (no retroactive blast radius)."""
    if d.get("offline_reproduced") is not True or not isinstance(d.get("game"), str):
        return False
    lvl = _arc_claimed_level(d)
    return lvl is not None and lvl >= 1


def _arc_registry_level(game: str) -> int | None:
    """Best-effort: the levels_reproduced the registry already records for `game` (None if unknown)."""
    try:
        import yaml  # optional; if unavailable, the duplication sub-check is simply skipped

        root = Path(__file__).resolve().parents[1]
        reg = yaml.safe_load((root / "ops" / "arc_solve_registry.yaml").read_text())
    except Exception:
        return None
    entries = (
        reg if isinstance(reg, list) else (reg.get("games") if isinstance(reg, dict) else None)
    )
    if not isinstance(entries, list):
        return None
    best: int | None = None
    for e in entries:
        if isinstance(e, dict) and e.get("game") == game:
            lv = e.get("levels_reproduced")
            if isinstance(lv, (int, float)) and not isinstance(lv, bool):
                best = max(best if best is not None else 0, int(lv))
    return best


# The outer-loop OFFLINE-GROUND-TRUTH calibration signature, keyed on the experiment NAME (not prose, so an
# honest artifact whose methodology says "NO offline BFS" is not false-flagged). This is the exact pattern of
# the 2nd recurrence: experiment_hazard_l3_calibration ran an exhaustive position-keyed real-env BFS to label
# ground-truth (state, action, died) and hand-fit a model's lethal zone -- RE the live agent cannot run on a
# hidden game (no exhaustive budget, no oracle). It is NOT a live solve regardless of structural shape.
_ARC_CALIBRATION_NAME_RE = re.compile(
    r"calibrat|ground[_-]?truth|exhaustive[_-]?bfs|position[_-]?keyed[_-]?bfs|bfs[_-]?ground",
    re.IGNORECASE,
)
# `(?<![a-z])solv` NOT bare `solv` (2026-07-30). The bare form matched inside UNRELATED LONGER
# WORDS, and the words it hit were the opposite of a solve claim: "unre-solv-ed", "re-solv-e",
# "dis-solv-e". A cost artifact whose verdict said "...affordability is UNRESOLVED..." was
# CRITICAL-flagged as an ARC game-solve claim on that substring alone.
#
# A plain `\b` will not do here: this project's verdicts are underscore-joined
# (`complete_self_solve_...`), and `_` is a word character, so `\bsolv` would MISS the real
# matches it exists to catch. The negative lookbehind for a letter accepts `_solve` / `-solve` /
# start-of-string `solving` while rejecting `unresolved` / `resolve` / `dissolve`.
#
# This is the exact bug class CLAUDE.md "QA-Layer Authenticity Discipline" was written for
# (origin bug #3 was `"diffusiongemma_met" in verdict` matching inside `meta_tensor`), found the
# same way: a legitimate artifact quarantined by a substring hit with no boundary awareness.
_ARC_GAME_SOLVE_CLAIM_RE = re.compile(
    r"winpath|win[_-]?path|lethal|(?<![a-z])solv|\bl[1-9]\b|level[_-]?up", re.IGNORECASE
)


# Known ARC-AGI-3 game ids (distinctive 4-char tokens). Substring-matched (not \b-bounded) so a verdict
# like "..._on_tu93_L3" -- where the id is wedged between underscores -- is still recognized as ARC.
_ARC_GAME_IDS = (
    "tu93",
    "lp85",
    "sc25",
    "ka59",
    "ls20",
    "r11l",
    "dc22",
    "tr87",
    "vc33",
    "ar25",
    "cn04",
    "cd82",
    "ft09",
    "m0r0",
    "sp80",
    "su15",
    "wa30",
    "re86",
    "sb26",
    "bp35",
    "lf52",
    "tn36",
)


def _is_arc_artifact(d: dict[str, Any]) -> bool:
    text = f"{d.get('experiment', '')} {d.get('honest_verdict', '')} {d.get('mode', '')}".lower()
    return isinstance(d.get("game"), str) or "arc" in text or any(g in text for g in _ARC_GAME_IDS)


_ARC_LIVE_CLAIM_TEXT_KEYS = (
    "honest_verdict",
    "headline",
    "headline_outcome",
    "title",
    "claim",
    "summary",
)
_ARC_LIVE_CONTEXT_MARKERS = (
    "live",
    "live_agent",
    "live agent",
    "scored agent",
    "submitted agent",
)
_ARC_LIVE_WIN_MARKERS = (
    "first_win",
    "first-win",
    "first win",
    "efficiency",
    "actions",
    "solve",
    "solved",
    "levelup",
    "level-up",
    "search win",
)
_ARC_POSITIVE_LIVE_CLAIM_MARKERS = (
    "success:",
    "_up",
    " up",
    "lift",
    "improv",
    "wins",
    "won",
    "solved",
    "fewer actions",
    "reduced actions",
)
_ARC_NULL_LIVE_CLAIM_MARKERS = (
    "honest_null",
    "no_live_value",
    "no live value",
    "no_value",
    "no value",
    "null",
    "regressed",
    "gap_open",
    "blocked",
)
_ARC_LIVE_METRIC_KEY_MARKERS = (
    "first_win_rate",
    "actions_to_first",
    "median_actions",
    "actions_delta",
    "action_efficiency",
    "solve_rate",
)
_ARC_OFFLINE_AUROC_KEY_MARKERS = ("auroc", "auc")
_ARC_OFFLINE_AUROC_CONTEXT_MARKERS = ("offline", "loo", "leave_one", "leave-one", "detector")
INTRINSIC_REWARD_WITHOUT_DOWNSTREAM_GAIN_KIND = "intrinsic-reward-without-downstream-gain"
GOAL_ENERGY_WITHOUT_ABLATION_KIND = "goal-energy-without-ablation"
QD_WITHOUT_RANDOM_MUTATION_ABLATION_KIND = "qd-without-random-mutation-ablation"
QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND = "qd-random-mutation-ablation-omitted"
VALUE_ROUTING_WITHOUT_COST_CONTROL_KIND = "value-routing-without-cost-control"
VALUE_ROUTING_COST_CONTROL_OMITTED_KIND = "value-routing-cost-control-omitted"
L2_GOAL_INDUCTION_WITHOUT_SATISFIABILITY_CHECK_KIND = (
    "l2-goal-induction-without-satisfiability-check"
)
L2_GOAL_SATISFIABILITY_CHECK_OMITTED_KIND = "l2-goal-satisfiability-check-omitted"
MULTI_LEVEL_WITHOUT_NONDEGENERATE_METRIC_KIND = "multi-level-without-nondegenerate-metric"
MULTI_LEVEL_NONDEGENERATE_METRIC_OMITTED_KIND = "multi-level-nondegenerate-metric-omitted"
SUBGOAL_SEARCH_WITHOUT_DECOMPOSITION_EVIDENCE_KIND = "subgoal-search-without-decomposition-evidence"
SUBGOAL_DECOMPOSITION_EVIDENCE_OMITTED_KIND = "subgoal-decomposition-evidence-omitted"
GENERATION_COVERAGE_WITHOUT_BASELINE_KIND = "generation-coverage-without-baseline"
GENERATION_COVERAGE_BASELINE_OMITTED_KIND = "generation-coverage-baseline-omitted"
NOVELTY_PROPOSAL_WITHOUT_ABLATION_KIND = "novelty-proposal-without-ablation"
NOVELTY_PROPOSAL_ABLATION_OMITTED_KIND = "novelty-proposal-ablation-omitted"
PROPOSAL_FILTER_WITHOUT_HELDOUT_REJECTION_KIND = "proposal-filter-without-heldout-rejection"
PROPOSAL_FILTER_HELDOUT_REJECTION_OMITTED_KIND = "proposal-filter-heldout-rejection-omitted"
PERCEPTION_OVERCLAIM_KIND = "perception-overclaim"
PERCEPTION_OVERCLAIM_OMITTED_KIND = "perception-overclaim-omitted"
LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND = "LEVER_EXERCISE_EVIDENCE_DEGENERATE"
_INTRINSIC_REWARD_CONTEXT_MARKERS = (
    "curiosity",
    "exploration",
    "intrinsic reward",
    "intrinsic_reward",
    "intrinsic-bonus",
    "intrinsic_bonus",
    "learning-progress",
    "learning_progress",
    "learning progress",
)
_INTRINSIC_REWARD_WIN_MARKERS = (
    "success:",
    "win",
    "wins",
    "won",
    "_up",
    " up",
    "lift",
    "improv",
    "solve",
    "solved",
    "coverage",
)
_INTRINSIC_REWARD_DIAGNOSTIC_OR_NULL_MARKERS = (
    "diagnostic",
    "honest_null",
    "null",
    "no_live_lift",
    "no live lift",
    "no_win",
    "no win",
    "no_lift",
    "no lift",
    "no_gain",
    "no gain",
    "no_improvement",
    "no improvement",
    "gap_open",
    "blocked",
    "unchanged",
    "regressed",
)
_INTRINSIC_REWARD_MAGNITUDE_KEY_MARKERS = (
    "intrinsic",
    "curiosity",
    "learning_progress",
    "learning-progress",
    "bonus",
)
_INTRINSIC_REWARD_DOWNSTREAM_DELTA_KEYS = (
    "solve_rate_delta",
    "state_coverage_delta",
    "first_win_rate_delta",
)
_GOAL_ENERGY_CONTEXT_MARKERS = (
    "goal-energy",
    "goal_energy",
    "goal energy",
    "graded goal",
    "graded_goal",
    "energy-driven",
    "energy driven",
    "energy-as-fitness",
    "energy as fitness",
    "energy heuristic",
    "goal-satisfaction energy",
    "goal satisfaction energy",
)
_GOAL_ENERGY_GENERATION_MARKERS = (
    "generation",
    "generate",
    "generated",
    "generator",
    "live",
    "search",
    "solve_rate",
    "solve-rate",
    "solve rate",
    "first_win",
    "first-win",
    "first win",
)
_GOAL_ENERGY_WIN_MARKERS = (
    "success:",
    "win",
    "wins",
    "won",
    "_up",
    " up",
    "raise",
    "raised",
    "lift",
    "improv",
    "beat",
    "beats",
    "beating",
    "solved",
)
_GOAL_ENERGY_DIAGNOSTIC_OR_NULL_MARKERS = (
    "diagnostic",
    "honest_null",
    "null",
    "no_live_lift",
    "no live lift",
    "no_win",
    "no win",
    "no_lift",
    "no lift",
    "no_gain",
    "no gain",
    "no_improvement",
    "no improvement",
    "no_value",
    "no value",
    "gap_open",
    "blocked",
    "unchanged",
    "regressed",
)
_GOAL_ENERGY_ABLATION_KEY_MARKERS = (
    "uniform_energy_ablation_passed",
    "uniform_energy_ablation",
    "uniform-energy-ablation",
    "uniform_measurement",
    "uniform_energy",
    "uniform-energy",
    "random_energy_ablation",
    "random-energy-ablation",
    "random_energy",
    "random-energy",
)
_GOAL_ENERGY_ABLATION_ARM_VALUE_MARKERS = (
    "uniform_energy",
    "uniform-energy",
    "uniform energy",
    "random_energy",
    "random-energy",
    "random energy",
)
_GOAL_ENERGY_ABLATION_ARM_NAME_KEYS = (
    "name",
    "arm",
    "policy_mode",
    "mode",
    "condition",
    "label",
)
_GOAL_ENERGY_POSITIVE_DELTA_KEYS = (
    "solve_rate_delta",
    "first_win_rate_delta",
    "live_solve_rate_delta",
    "live_first_win_rate_delta",
    "energy_on_baseline_delta",
    "goal_energy_baseline_delta",
)
_GOAL_ENERGY_BASELINE_PAIRS = (
    ("live_solve_rate_goal_energy", "live_solve_rate_baseline"),
    ("goal_energy_solve_rate", "baseline_solve_rate"),
    ("solve_rate_goal_energy", "solve_rate_baseline"),
    ("live_first_win_rate_goal_energy", "live_first_win_rate_baseline"),
    ("first_win_rate_goal_energy", "first_win_rate_baseline"),
    ("goal_energy_first_win_rate", "baseline_first_win_rate"),
)
_GOAL_ENERGY_BEATS_BASELINE_KEY_MARKERS = (
    "energy_on_beats_baseline",
    "goal_energy_beats_baseline",
    "energy_beats_baseline",
)
_GOAL_ENERGY_CLAIM_TEXT_KEYS = _ARC_LIVE_CLAIM_TEXT_KEYS + (
    "goal_energy_source",
    "chosen_submitted_config",
)


def _arc_live_claim_text(d: dict[str, Any]) -> str:
    return " ".join(str(d.get(key, "")) for key in _ARC_LIVE_CLAIM_TEXT_KEYS).lower()


def _claims_arc_live_search_win(d: dict[str, Any]) -> bool:
    """True if the artifact headlines a positive live-agent ARC search win."""
    if not _is_arc_artifact(d):
        return False
    text = _arc_live_claim_text(d)
    live_context = any(marker in text for marker in _ARC_LIVE_CONTEXT_MARKERS) or (
        d.get("solve_provenance") == "live_agent_self_discovery"
    )
    if not live_context:
        return False
    if not any(marker in text for marker in _ARC_LIVE_WIN_MARKERS):
        return False
    positive = any(marker in text for marker in _ARC_POSITIVE_LIVE_CLAIM_MARKERS)
    null = any(marker in text for marker in _ARC_NULL_LIVE_CLAIM_MARKERS)
    return positive and not null


def _has_measured_arc_live_metric(d: dict[str, Any]) -> bool:
    """Return true for real metric fields, not prose-only field principles."""
    for key, value in d.items():
        kl = str(key).lower()
        if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
            continue
        if value is None:
            continue
        if kl.startswith("offline_") or "_offline" in kl:
            continue
        if kl.startswith("live_") and (
            _is_finite_number(value) or isinstance(value, (dict, list, bool))
        ):
            return True
        if any(marker in kl for marker in _ARC_LIVE_METRIC_KEY_MARKERS) and (
            _is_finite_number(value) or isinstance(value, (dict, list))
        ):
            return True
    return False


def _has_offline_auroc_metric(d: dict[str, Any]) -> bool:
    for key, value in d.items():
        kl = str(key).lower()
        if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS or not _is_finite_number(value):
            continue
        if not any(marker in kl for marker in _ARC_OFFLINE_AUROC_KEY_MARKERS):
            continue
        if any(marker in kl for marker in _ARC_OFFLINE_AUROC_CONTEXT_MARKERS):
            return True
    return False


def check_arc_offline_live_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Warn when an ARC live-search win is backed only by offline AUROC."""
    if not _claims_arc_live_search_win(d):
        return
    if _has_measured_arc_live_metric(d):
        return
    if not _has_offline_auroc_metric(d):
        return
    flags.append(
        Flag(
            kind="OFFLINE_SUBSTITUTED_FOR_LIVE",
            severity="warn",
            detail=(
                "ARC artifact claims a live search win but reports offline AUROC "
                "evidence without a measured live metric field. Add first_win_rate_*, "
                "actions_*/median_actions_*, or live_* evidence from the live agent; "
                "offline AUROC alone characterizes the detector, not a live win."
            ),
        )
    )


def _claims_intrinsic_reward_exploration_win(d: dict[str, Any]) -> bool:
    """True when an ARC headline claims an intrinsic-reward exploration win."""
    if not _is_arc_artifact(d):
        return False
    text = _arc_live_claim_text(d)
    if not any(marker in text for marker in _INTRINSIC_REWARD_CONTEXT_MARKERS):
        return False
    if any(marker in text for marker in _INTRINSIC_REWARD_DIAGNOSTIC_OR_NULL_MARKERS):
        return False
    return any(marker in text for marker in _INTRINSIC_REWARD_WIN_MARKERS)


def _is_intrinsic_reward_downstream_delta_key(key: str) -> bool:
    kl = str(key).lower()
    return any(
        kl == wanted or kl.endswith(f"_{wanted}")
        for wanted in _INTRINSIC_REWARD_DOWNSTREAM_DELTA_KEYS
    )


def _has_measured_intrinsic_reward_downstream_delta(value: Any) -> bool:
    """Find real downstream delta fields outside metadata principle prose."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            if _is_intrinsic_reward_downstream_delta_key(key) and _is_finite_number(nested):
                return True
            if _has_measured_intrinsic_reward_downstream_delta(nested):
                return True
    elif isinstance(value, list):
        return any(_has_measured_intrinsic_reward_downstream_delta(item) for item in value)
    return False


def _has_rising_intrinsic_reward_magnitude(value: Any) -> bool:
    """Find numeric intrinsic reward / curiosity bonus magnitude evidence."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            kl = str(key).lower()
            if (
                any(marker in kl for marker in _INTRINSIC_REWARD_MAGNITUDE_KEY_MARKERS)
                and _is_finite_number(nested)
                and float(nested) > 0.0
            ):
                return True
            if _has_rising_intrinsic_reward_magnitude(nested):
                return True
    elif isinstance(value, list):
        return any(_has_rising_intrinsic_reward_magnitude(item) for item in value)
    return False


def check_intrinsic_reward_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Warn when an intrinsic-reward exploration win lacks downstream deltas."""
    if not _claims_intrinsic_reward_exploration_win(d):
        return
    if _has_measured_intrinsic_reward_downstream_delta(d):
        return
    if not _has_rising_intrinsic_reward_magnitude(d):
        return
    flags.append(
        Flag(
            kind=INTRINSIC_REWARD_WITHOUT_DOWNSTREAM_GAIN_KIND,
            severity="warn",
            detail=(
                "intrinsic-reward-without-downstream-gain: ARC artifact claims "
                "a curiosity/exploration/learning-progress win but reports only "
                "intrinsic-bonus magnitude evidence. Add a measured downstream "
                "solve_rate_delta, state_coverage_delta, or first_win_rate_delta "
                "versus a control before treating the intrinsic reward as a win."
            ),
        )
    )


def _goal_energy_claim_text(d: dict[str, Any]) -> str:
    return " ".join(str(d.get(key, "")) for key in _GOAL_ENERGY_CLAIM_TEXT_KEYS).lower()


def _has_positive_goal_energy_baseline_win_evidence(d: dict[str, Any]) -> bool:
    for key, value in d.items():
        if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
            continue
        kl = str(key).lower()
        if (
            any(
                kl == wanted or kl.endswith(f"_{wanted}")
                for wanted in _GOAL_ENERGY_POSITIVE_DELTA_KEYS
            )
            and _is_finite_number(value)
            and float(value) > 0.0
        ):
            return True
        if (
            any(marker in kl for marker in _GOAL_ENERGY_BEATS_BASELINE_KEY_MARKERS)
            and value is True
        ):
            return True
    for goal_key, baseline_key in _GOAL_ENERGY_BASELINE_PAIRS:
        goal = _finite_float(d, goal_key)
        baseline = _finite_float(d, baseline_key)
        if goal is not None and baseline is not None and goal > baseline:
            return True
    return False


def _claims_goal_energy_generation_win(d: dict[str, Any]) -> bool:
    """True when an ARC headline claims goal-energy drove live generation."""
    if not _is_arc_artifact(d):
        return False
    text = _goal_energy_claim_text(d)
    if not any(marker in text for marker in _GOAL_ENERGY_CONTEXT_MARKERS):
        return False
    if any(marker in text for marker in _GOAL_ENERGY_DIAGNOSTIC_OR_NULL_MARKERS):
        return False
    if not any(marker in text for marker in _GOAL_ENERGY_GENERATION_MARKERS):
        return False
    if not any(marker in text for marker in _GOAL_ENERGY_WIN_MARKERS):
        return False
    return _has_positive_goal_energy_baseline_win_evidence(d)


def _has_uniform_energy_ablation_evidence(value: Any) -> bool:
    """Find real uniform/random-energy ablation fields outside principle prose."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            kl = str(key).lower()
            if (
                any(marker in kl for marker in _GOAL_ENERGY_ABLATION_KEY_MARKERS)
                and nested is not None
            ):
                return True
            if (
                kl in _GOAL_ENERGY_ABLATION_ARM_NAME_KEYS
                and isinstance(nested, str)
                and any(
                    marker in nested.lower() for marker in _GOAL_ENERGY_ABLATION_ARM_VALUE_MARKERS
                )
            ):
                return True
            if _has_uniform_energy_ablation_evidence(nested):
                return True
    elif isinstance(value, list):
        return any(_has_uniform_energy_ablation_evidence(item) for item in value)
    return False


def check_goal_energy_ablation_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Warn when a goal-energy generation win lacks a uniform-energy ablation."""
    if not _claims_goal_energy_generation_win(d):
        return
    if _has_uniform_energy_ablation_evidence(d):
        return
    flags.append(
        Flag(
            kind=GOAL_ENERGY_WITHOUT_ABLATION_KIND,
            severity="warn",
            detail=(
                "goal-energy-without-ablation: ARC artifact claims an energy-driven "
                "generation win but reports only energy-on beat the baseline evidence. "
                "Add uniform-energy ablation evidence such as "
                "uniform_energy_ablation_passed or a uniform/random-energy ablation "
                "arm before treating goal-energy as the driver."
            ),
        )
    )


_QD_CLAIM_TEXT_KEYS = _ARC_LIVE_CLAIM_TEXT_KEYS + (
    "experiment",
    "chosen_submitted_config",
    "solve_provenance",
)
_QD_CONTEXT_MARKERS = (
    "energy_fitness_qd",
    "energy-fitness qd",
    "energy fitness qd",
    "quality-diversity",
    "quality_diversity",
    "map-elites",
    "map_elites",
    "energy-fitness",
    "energy_fitness",
    "energy-as-fitness",
    "energy as fitness",
)
_QD_GENERATION_MARKERS = (
    "generation",
    "generate",
    "generated",
    "generator",
    "winner_generated",
    "winner generated",
    "solve_rate",
    "solve-rate",
    "solve rate",
    "live",
)
_QD_POSITIVE_DELTA_KEYS = (
    "solve_rate_delta",
    "live_solve_rate_delta",
    "first_win_rate_delta",
    "live_first_win_rate_delta",
)
_QD_BASELINE_PAIRS = (
    ("live_solve_rate_qd", "live_solve_rate_search_baseline"),
    ("live_solve_rate_energy_fitness_qd", "live_solve_rate_search_baseline"),
    ("first_win_rate_qd", "first_win_rate_search_baseline"),
    ("live_first_win_rate_qd", "live_first_win_rate_search_baseline"),
)
_QD_FIRST_PARTY_FIELD_MARKERS = (
    "winner_generated",
    "winning_trajectory_surfaced",
    "qd_generation",
    "qd_arm",
    "qd_lift",
    "energy_fitness_qd",
    "energy_qd",
    "live_solve_rate_qd",
    "live_first_win_rate_qd",
    "first_win_rate_qd",
)
_VALUE_ROUTING_CLAIM_TEXT_KEYS = _ARC_LIVE_CLAIM_TEXT_KEYS + (
    "experiment",
    "chosen_submitted_config",
    "solve_provenance",
)
_VALUE_ROUTING_CONTEXT_MARKERS = (
    "value-routing",
    "value_routing",
    "value routing",
    "value-routed",
    "value_routed",
    "value routed",
    "value head",
    "value_head",
    "value_weight",
    "cost-fixed",
    "cost_fixed",
    "cost fix",
)
_VALUE_ROUTING_POSITIVE_DELTA_KEYS = (
    "first_win_rate_delta",
    "live_first_win_rate_delta",
    "solve_rate_delta",
    "live_solve_rate_delta",
)
_VALUE_ROUTING_BASELINE_PAIRS = (
    ("live_first_win_rate_value_routed", "live_first_win_rate_baseline"),
    ("first_win_rate_value_routed", "first_win_rate_baseline"),
    ("live_solve_rate_value_routed", "live_solve_rate_baseline"),
    ("solve_rate_value_routed", "solve_rate_baseline"),
)
_L2_GOAL_INDUCTION_CLAIM_TEXT_KEYS = _ARC_LIVE_CLAIM_TEXT_KEYS + (
    "experiment",
    "solve_provenance",
    "residual_cause_hypothesis",
)
_L2_GOAL_INDUCTION_CONTEXT_MARKERS = (
    "l2_goal_induction",
    "l2 goal induction",
    "goal_induction",
    "goal induction",
    "runtime induction",
    "runtime_induction",
    "re-induction",
    "reinduction",
    "induced goal",
    "induced_goal",
)
_L2_CLAIM_MARKERS = (
    "l2",
    "level 2",
    "level_2",
    "level-2",
    "multi-level",
    "multi_level",
)
_L2_WIN_TEXT_MARKERS = (
    "success:",
    "reached l2",
    "reached_l2",
    "level 2 reached",
    "level_2_reached",
    "generic_agent_reached_l2",
    "generic_agent_reached_level",
)
_L2_NULL_TEXT_MARKERS = (
    "no_deepening",
    "no deepening",
    "no_l2",
    "no l2",
    "residual",
    "blocked",
    "null",
    "complete:",
)
_MULTI_LEVEL_RATE_KEYS = (
    "live_multi_level_solve_rate",
    "multi_level_solve_rate",
)
_SUBGOAL_SEARCH_CLAIM_TEXT_KEYS = _ARC_LIVE_CLAIM_TEXT_KEYS + (
    "experiment",
    "solve_provenance",
    "chosen_submitted_config",
)
_SUBGOAL_SEARCH_CONTEXT_MARKERS = (
    "hierarchical_subgoal",
    "hierarchical subgoal",
    "subgoal_search",
    "subgoal search",
    "subgoal-search",
    "subgoal planner",
    "runtime subgoal",
    "subgoal decomposition",
)
_SUBGOAL_SEARCH_WIN_MARKERS = (
    "success:",
    "new_level",
    "new level",
    "reached l",
    "reached_l",
    "reached level",
    "reached_level",
    "generic_agent_reached_level",
    "levelup",
    "level-up",
    "level up",
)
_SUBGOAL_SEARCH_NULL_TEXT_MARKERS = (
    "complete:",
    "no_new_level",
    "no new level",
    "no_level",
    "no level",
    "no_deepening",
    "no deepening",
    "residual",
    "null",
    "unchanged",
    "regressed",
    "blocked",
)
_SUBGOAL_SEARCH_REQUIRED_EVIDENCE_KEYS = (
    "subgoal_decomposition",
    "per_subgoal_reachable",
    "no_subgoal_ablation_reached_level",
    "random_subgoal_ablation_reached_level",
    "offline_reproduced",
)
_GENERATION_COVERAGE_CLAIM_TEXT_KEYS = _ARC_LIVE_CLAIM_TEXT_KEYS + (
    "experiment",
    "solve_provenance",
    "chosen_submitted_config",
    "residual_bridge_gap",
)
_GENERATION_COVERAGE_CONTEXT_MARKERS = (
    "candidate_generation_coverage",
    "candidate-generation coverage",
    "candidate generation coverage",
    "coverage_delta",
    "coverage delta",
    "coverage-up",
    "coverage up",
    "factored planner",
    "factored subgoal planner",
    "poe_world_factored",
    "product planner",
)
_GENERATION_COVERAGE_WIN_MARKERS = (
    "success:",
    "coverage_up",
    "coverage-up",
    "coverage up",
    "coverage rose",
    "coverage rise",
    "coverage lift",
    "lift",
    "improv",
    "_up",
    " up",
)
_GENERATION_COVERAGE_NULL_TEXT_MARKERS = (
    "complete:",
    "no_coverage_gain",
    "no coverage gain",
    "no_gain",
    "no gain",
    "null",
    "residual",
    "unchanged",
    "regressed",
    "blocked",
)
_GENERATION_COVERAGE_POSITIVE_METRIC_KEYS = (
    "coverage_delta",
    "candidate_generation_coverage_delta",
)
_GENERATION_COVERAGE_VALUE_KEYS = (
    "candidate_generation_coverage",
    "candidate_generation_coverage_factored",
)
_NOVELTY_PROPOSAL_CLAIM_TEXT_KEYS = _ARC_LIVE_CLAIM_TEXT_KEYS + (
    "experiment",
    "solve_provenance",
    "chosen_submitted_config",
    "residual_cause_hypothesis",
)
_NOVELTY_PROPOSAL_CONTEXT_MARKERS = (
    "controllable_novelty",
    "controllable novelty",
    "novelty proposal",
    "novelty_bonus",
    "novelty bonus",
    "controllability_gate",
    "controllability gate",
    "no_novelty_ablation_reached_level",
    "cosmetic_novelty_ablation_reached_level",
)
_NOVELTY_PROPOSAL_WIN_MARKERS = (
    "success:",
    "new_level",
    "new level",
    "reached l",
    "reached_l",
    "reached level",
    "reached_level",
    "generic_agent_reached_level",
    "levelup",
    "level-up",
    "level up",
)
_NOVELTY_PROPOSAL_NULL_TEXT_MARKERS = (
    "complete:",
    "no_new_level",
    "no new level",
    "no_level",
    "no level",
    "residual",
    "null",
    "unchanged",
    "regressed",
    "blocked",
)
_NOVELTY_PROPOSAL_REQUIRED_EVIDENCE_KEYS = (
    "no_novelty_ablation_reached_level",
    "cosmetic_novelty_ablation_reached_level",
    "offline_reproduced",
)
_PROPOSAL_FILTER_CLAIM_TEXT_KEYS = _ARC_LIVE_CLAIM_TEXT_KEYS + (
    "experiment",
    "solve_provenance",
    "chosen_submitted_config",
    "residual_bridge_gap",
)
_PROPOSAL_FILTER_CONTEXT_MARKERS = (
    "program_synthesis",
    "program synthesis",
    "proposal_filter",
    "proposal filter",
    "action-effect proposal filter",
    "action effect proposal filter",
    "candidate_generation_coverage_filter",
    "heldout_programs",
    "held-out",
    "heldout",
)
_PROPOSAL_FILTER_WIN_MARKERS = (
    "success:",
    "coverage_up",
    "coverage-up",
    "coverage up",
    "coverage rose",
    "coverage rise",
    "coverage lift",
    "lift",
    "improv",
    "_up",
    " up",
)
_PROPOSAL_FILTER_NULL_TEXT_MARKERS = (
    "complete:",
    "no_coverage_gain",
    "no coverage gain",
    "no_gain",
    "no gain",
    "null",
    "residual",
    "unchanged",
    "regressed",
    "blocked",
)
_PROPOSAL_FILTER_POSITIVE_METRIC_KEYS = (
    "coverage_delta",
    "candidate_generation_coverage_delta",
)
_PROPOSAL_FILTER_REQUIRED_EVIDENCE_KEYS = (
    "heldout_programs_rejected",
    "candidate_generation_coverage_blind_baseline",
)
_PERCEPTION_CLAIM_TEXT_KEYS = _ARC_LIVE_CLAIM_TEXT_KEYS + (
    "experiment",
    "solve_provenance",
    "chosen_submitted_config",
    "residual_cause_hypothesis",
)
_PERCEPTION_CONTEXT_MARKERS = (
    "object_centric",
    "object-centric",
    "object centric",
    "relational representation",
    "relational",
    "perception",
    "order1_ablation_reached_level",
    "proposal_coverage_by_representation",
)
_PERCEPTION_WIN_MARKERS = (
    "success:",
    "first_win",
    "first-win",
    "first win",
    "new_level",
    "new level",
    "reached l",
    "reached_l",
    "reached level",
    "reached_level",
    "generic_agent_reached_level",
    "levelup",
    "level-up",
    "level up",
    "lift",
    "lifted",
    "raised",
)
_PERCEPTION_NULL_TEXT_MARKERS = (
    "complete:",
    "no_new_level",
    "no new level",
    "no_level",
    "no level",
    "no_first_win",
    "no first win",
    "residual",
    "null",
    "unchanged",
    "regressed",
    "blocked",
)
_PERCEPTION_POSITIVE_METRIC_KEYS = (
    "first_win_delta",
    "first_win_rate_delta",
    "live_first_win_rate_delta",
)
_PERCEPTION_REQUIRED_EVIDENCE_KEYS = (
    "order1_ablation_reached_level",
    "offline_reproduced",
)
_LEVER_EXERCISE_CONTEXT_MARKERS = (
    "candidate_generation_coverage",
    "candidate-generation coverage",
    "candidate generation coverage",
    "go_explore",
    "go-explore",
    "archive_injections",
    "archive cells",
    "archive_cells",
    "actions_injected",
    "prefixes_injected",
    "proposal_pool",
    "proposal pool",
    "candidate_pool",
    "candidate pool",
    "online_action_learning",
    "online action learning",
    "online-driver-arms",
    "online_driver_arms",
    "online_warm_first_win",
    "online_scratch_first_win",
    "active_probe",
    "active-probe",
    "hypothesis_posterior",
    "posterior_entropy_reduction",
    "probe_actions_taken",
)
_LEVER_ZERO_DELTA_KEY_MARKERS = (
    "coverage_delta",
    "candidate_generation_coverage",
    "first_win_rate_delta",
    "first_win_delta",
    "online_warm_vs_frozen_delta",
    "best_online_delta",
)
_LEVER_POOL_KEY_MARKERS = (
    "candidate_pool",
    "proposal_pool",
    "candidate_generation_pool",
    "generated_pool",
)
_LEVER_ARCHIVE_PATH_MARKERS = ("archive", "go_explore", "go-explore")
_LEVER_ARCHIVE_ZERO_KEYS = (
    "actions_injected",
    "archive_injections",
    "prefixes_injected",
    "stored_cells",
    "archive_cells",
)
_LEVER_ONLINE_ARM_MARKERS = ("frozen", "scratch", "warm")
_LEVER_ONLINE_METRIC_MARKERS = (
    "first_win",
    "first_win_rate",
    "solve_rate",
    "live_solve_rate",
)
_LEVER_PROBE_DECLARATION_MARKERS = (
    "active_probe",
    "active-probe",
    "hypothesis_posterior",
    "probe_actions_taken",
    "hypothesis_posterior_built",
    "posterior_entropy_reduction",
)


def _claim_text(d: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " ".join(str(d.get(key, "")) for key in keys).lower()


def _field_name_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            parts.append(str(key).lower())
            parts.append(_field_name_text(nested))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_field_name_text(item) for item in value)
    return ""


def _has_marker(text: str, markers: tuple[str, ...]) -> bool:
    text = text.lower()
    for marker in markers:
        if marker == "qd":
            if re.search(r"(^|[^a-z0-9])qd([^a-z0-9]|$)", text):
                return True
            continue
        if marker in text:
            return True
    return False


def _real_field_values(value: Any, wanted_key: str) -> list[Any]:
    values: list[Any] = []
    wanted = wanted_key.lower()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            if str(key).lower() == wanted:
                values.append(nested)
                continue
            values.extend(_real_field_values(nested, wanted))
    elif isinstance(value, list):
        for item in value:
            values.extend(_real_field_values(item, wanted))
    return values


def _typed_field_values(value: Any, wanted_key: str, kind: str) -> list[Any]:
    """`_real_field_values` FILTERED TO VALUES OF THE ASKED-FOR TYPE.

    WHY THIS EXISTS (2026-07-27 review finding 6; QA-Layer Authenticity Discipline).
    `_real_field_values` collects ANY value stored under the wanted key, and several callers
    then only tested it for NON-EMPTINESS:

        cost_values = _real_field_values(d, "per_node_feature_cost_ms")
        if not cost_values: omitted.append("per_node_feature_cost_ms")

    So a PROSE SENTENCE stored under that key satisfied a check that is asking for a finite
    cost in milliseconds, and a prose sentence under `sim_timed_out` satisfied a check asking
    for a boolean. This was found the hard way: a first draft of the 2026-07-27 first-win
    artifact wrote exactly those two key names with explanatory prose as their values, and
    the WARN CLEARED -- adversarial_verify went from 1 flagged to 0 -- for no reason other
    than that a string is truthy. That is a general way to clear this class of check with
    prose, on the layer that decides whether every OTHER result counts as clean.

    A wrapped value (`{"principle": ..., "value": 3.2}`) is unwrapped before typing, per the
    Principle-Annotated Artifact Fields discipline, so honest annotated fields still satisfy
    the check -- the point is to reject the WRONG TYPE, not to reject annotation.
    """
    out: list[Any] = []
    for raw in _real_field_values(value, wanted_key):
        candidates = [raw]
        if isinstance(raw, dict) and "value" in raw:
            candidates.append(raw["value"])
        for candidate in candidates:
            # LEAF RECOVERY, not a top-level type test. Corrected 2026-07-27 after a first
            # draft demanding a BARE bool over-fired on the real per-game shape these fields
            # actually use -- `goal_predicate_satisfiable: {"lp85": true}` is a dict, and
            # `_harness_break_at_first_win` has always read these via `_bool_leaf_values`.
            # Requiring the top value to be the scalar would have flagged an HONEST artifact,
            # which is the mirror-image of the defect under repair and strictly worse (a
            # false positive here quarantines real work and teaches the operator to ignore
            # the gate). Leaf recovery still rejects the actual defect: `_bool_leaf_values`
            # and `_numeric_leaf_values` both return [] for a bare string.
            #
            # RETURN THE LEAVES, not the container. Callers type-test the returned items
            # directly -- check_value_routing_cost_control_overclaim's CRITICAL branch does
            # `any(_is_finite_number(v) for v in cost_values)` and
            # `all(v is False for v in timeout_values)`. Returning the wrapper dict would
            # make both False on an HONEST principle-annotated artifact and fire a CRITICAL,
            # which is a worse failure than the WARN this whole change is repairing.
            if kind == "number":
                leaves = _numeric_leaf_values(candidate)
                if leaves:
                    out.extend(leaves)
                    break
            if kind == "bool":
                leaves = _bool_leaf_values(candidate)
                if leaves:
                    out.extend(leaves)
                    break
            # "structured" is the weaker filter for keys whose legitimate shape is a
            # CONTAINER of sub-fields rather than a scalar -- `metric_harness_fixed` is a
            # bool in exp4669 and a {target_levels, break_at_first_win, port} dict in
            # exp4664, so demanding a bool there would over-fire on an honest artifact.
            # It still rejects the actual defect: a bare prose string.
            if kind == "structured" and candidate is not None and not isinstance(candidate, str):
                out.append(candidate)
                break
    return out


def _iter_real_fields(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    """Walk artifact fields while skipping metadata principle prose."""
    rows: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            nested_path = path + (key_text,)
            rows.append((nested_path, nested))
            rows.extend(_iter_real_fields(nested, nested_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested_path = path + (f"[{index}]",)
            rows.append((nested_path, item))
            rows.extend(_iter_real_fields(item, nested_path))
    return rows


def _path_text(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _path_has_marker(path: tuple[str, ...], markers: tuple[str, ...]) -> bool:
    text = _path_text(path).lower()
    return any(marker in text for marker in markers)


def _is_arc_generation_or_exploration_artifact(d: dict[str, Any]) -> bool:
    text = (
        f"{d.get('experiment', '')} {d.get('schema', '')} "
        f"{d.get('honest_verdict', '')} {d.get('inference_substrate', '')} "
        f"{d.get('solve_provenance', '')} {_field_name_text(d)}"
    ).lower()
    arc_context = _is_arc_artifact(d) or "arc" in text or "online_action_learning" in text
    return arc_context and any(marker in text for marker in _LEVER_EXERCISE_CONTEXT_MARKERS)


def _max_positive_real_field(d: dict[str, Any], wanted_key: str) -> float | None:
    positives = [
        float(number)
        for value in _real_field_values(d, wanted_key)
        for number in _numeric_leaf_values(value)
        if number > 0.0
    ]
    return max(positives, default=None)


def _has_nontrivial_lever_run(d: dict[str, Any]) -> bool:
    for wanted_key in ("actions", "budget", "duration_s", "iterations", "attempts", "observed"):
        if _max_positive_real_field(d, wanted_key) is not None:
            return True
    return any(value is True for value in _real_field_values(d, "attempted"))


def _archive_zero_reasons(d: dict[str, Any]) -> list[str]:
    if not _has_nontrivial_lever_run(d):
        return []
    reasons: list[str] = []
    for path, value in _iter_real_fields(d):
        leaf = path[-1].lower()
        if leaf not in _LEVER_ARCHIVE_ZERO_KEYS:
            continue
        if not _path_has_marker(path[:-1], _LEVER_ARCHIVE_PATH_MARKERS):
            continue
        if _is_finite_number(value) and float(value) == 0.0:
            reasons.append(f"{_path_text(path)}=0")
    return reasons


def _pool_degenerate_reasons(d: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for path, value in _iter_real_fields(d):
        leaf = path[-1].lower()
        if not any(marker in leaf for marker in _LEVER_POOL_KEY_MARKERS):
            continue
        if value in (None, "", [], {}):
            reasons.append(f"{_path_text(path)} is empty")
        elif isinstance(value, dict):
            keys = {str(key).lower(): nested for key, nested in value.items()}
            left = keys.get("pre") or keys.get("before") or keys.get("input")
            right = keys.get("post") or keys.get("after") or keys.get("output")
            if left not in (None, "", [], {}) and left == right:
                reasons.append(f"{_path_text(path)} is byte-identical before/after transform")
    return reasons


def _shape_dims(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(part) for part in re.findall(r"\d+", value)]
    if isinstance(value, (list, tuple)):
        dims: list[int] = []
        for item in value:
            if isinstance(item, bool):
                return []
            if isinstance(item, int):
                dims.append(item)
            elif isinstance(item, float) and item.is_integer():
                dims.append(int(item))
            else:
                return []
        return dims
    return []


def _grid_shape_degenerate_reasons(d: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for path, value in _iter_real_fields(d):
        leaf = path[-1].lower()
        if "shape" not in leaf and "grid_tensor" not in leaf:
            continue
        dims = _shape_dims(value)
        if len(dims) >= 3 and dims[0] == 1 and dims[-2] > 1 and dims[-1] > 1:
            reasons.append(f"{_path_text(path)}={tuple(dims)} has leading singleton grid axis")
    return reasons


def _scorer_diagnostics_error_reasons(d: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for path, value in _iter_real_fields(d):
        if not isinstance(value, dict) or path[-1].lower() != "scorer_diagnostics":
            continue
        errors = value.get("errors")
        if not (_is_finite_number(errors) and float(errors) > 0.0):
            continue
        observed = value.get("observed")
        fits = value.get("fits")
        detail = f"{_path_text(path)}.errors={errors}"
        if _is_finite_number(observed):
            detail += f", observed={observed}"
        if _is_finite_number(fits):
            detail += f", fits={fits}"
        reasons.append(detail)
    return reasons


def _has_positive_online_training_evidence(d: dict[str, Any]) -> bool:
    for path, value in _iter_real_fields(d):
        if not isinstance(value, dict) or path[-1].lower() != "scorer_diagnostics":
            continue
        observed = value.get("observed")
        fits = value.get("fits")
        errors = value.get("errors", 0)
        if (
            _is_finite_number(observed)
            and _is_finite_number(fits)
            and _is_finite_number(errors)
            and float(observed) > 0.0
            and float(fits) > 0.0
            and float(errors) == 0.0
        ):
            return True
    return False


def _has_distinct_arm_evidence(d: dict[str, Any]) -> bool:
    return (
        _real_field_has_true(d, "arms_non_degenerate")
        or _real_field_has_true(d, "per_arm_action_distribution_distinct")
        or _has_positive_online_training_evidence(d)
    )


def _online_arm_metric_items(d: dict[str, Any]) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for key, value in d.items():
        kl = str(key).lower()
        if not _is_finite_number(value):
            continue
        if not any(marker in kl for marker in _LEVER_ONLINE_ARM_MARKERS):
            continue
        if not any(marker in kl for marker in _LEVER_ONLINE_METRIC_MARKERS):
            continue
        items.append((str(key), float(value)))
    arms = d.get("arms")
    if isinstance(arms, list):
        for index, row in enumerate(arms):
            if not isinstance(row, dict):
                continue
            arm = str(row.get("arm", f"arm_{index}"))
            value = row.get("first_win_rate")
            if _is_finite_number(value):
                items.append((f"arms.{arm}.first_win_rate", float(value)))
    return items


def _byte_identical_online_arm_reason(d: dict[str, Any]) -> str | None:
    items = _online_arm_metric_items(d)
    labels = " ".join(label.lower() for label, _ in items)
    if len(items) < 3 or not all(marker in labels for marker in _LEVER_ONLINE_ARM_MARKERS):
        return None
    first_value = items[0][1]
    if not all(
        _significant_digits_match(first_value, value, TAUTOLOGY_DIGITS) for _, value in items[1:]
    ):
        return None
    formatted = ", ".join(f"{label}={value:.6g}" for label, value in items)
    return f"byte-identical online-driver arms to >{TAUTOLOGY_DIGITS} sig figs ({formatted})"


def _has_positive_probe_exercise_evidence(d: dict[str, Any]) -> bool:
    """True when a declared active-probe path emitted real exercise evidence."""
    probe_actions_positive = any(
        float(number) > 0.0
        for value in _real_field_values(d, "probe_actions_taken")
        for number in _numeric_leaf_values(value)
    )
    entropy_reduction_positive = any(
        float(number) > 0.0
        for value in _real_field_values(d, "posterior_entropy_reduction")
        for number in _numeric_leaf_values(value)
    )
    return probe_actions_positive and entropy_reduction_positive


def _declared_but_unrun_probe_reasons(d: dict[str, Any]) -> list[str]:
    if _has_positive_probe_exercise_evidence(d):
        return []

    declared = False
    reasons: list[str] = []
    for path, value in _iter_real_fields(d):
        path_text = _path_text(path)
        path_lower = path_text.lower()
        leaf = path[-1].lower()
        if any(marker in path_lower for marker in _LEVER_PROBE_DECLARATION_MARKERS):
            declared = True
        if leaf == "probe_actions_taken" and _is_finite_number(value) and float(value) == 0.0:
            reasons.append(f"{path_text}=0")
        elif leaf == "hypothesis_posterior_built" and value is False:
            reasons.append(f"{path_text}=False")
        elif (
            leaf == "posterior_entropy_reduction"
            and _is_finite_number(value)
            and float(value) == 0.0
        ):
            reasons.append(f"{path_text}=0.0")
    if not declared:
        return []
    return reasons


def _has_nondegenerate_lever_evidence(d: dict[str, Any]) -> bool:
    if _has_distinct_arm_evidence(d):
        return True
    if _has_positive_probe_exercise_evidence(d):
        return True
    for key in (
        "actions_injected",
        "archive_injections",
        "prefixes_injected",
        "stored_cells",
        "archive_cells",
        "heldout_programs_rejected",
        "candidate_scores",
        "observed_effects",
        "augmented_candidates",
        "candidate_group_count",
    ):
        if _max_positive_real_field(d, key) is not None:
            return True
    for path, value in _iter_real_fields(d):
        leaf = path[-1].lower()
        if any(marker in leaf for marker in _LEVER_POOL_KEY_MARKERS):
            if isinstance(value, (list, dict, str)) and len(value) > 0:
                return True
        if "shape" in leaf:
            dims = _shape_dims(value)
            if len(dims) == 2 and dims[0] > 1 and dims[1] > 1:
                return True
    return False


def _zero_lever_delta_reasons(d: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for path, value in _iter_real_fields(d):
        leaf = path[-1].lower()
        if not any(marker in leaf for marker in _LEVER_ZERO_DELTA_KEY_MARKERS):
            continue
        if _is_finite_number(value) and float(value) == 0.0:
            reasons.append(f"{_path_text(path)}=0.0 with no non-degenerate exercise evidence")
    return reasons


def _lever_exercise_severity(d: dict[str, Any]) -> str:
    verdict = str(d.get("honest_verdict", "")).lower()
    if d.get("flagged_adversarial") is True or d.get("submitted_to_leaderboard") is True:
        return "critical"
    if _flips_gate(d) or _claims_arc_live_search_win(d) or _is_arc_solve_claim(d):
        return "critical"
    if verdict.startswith(("success:", "success_", "shipped:", "shipped_", "passed:", "passed_")):
        return "critical"
    return "warn"


def check_lever_exercise_evidence(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag generation/exploration artifacts whose declared lever did not really exercise."""
    if not _is_arc_generation_or_exploration_artifact(d):
        return

    reasons: list[str] = []
    reasons.extend(_archive_zero_reasons(d))
    reasons.extend(_pool_degenerate_reasons(d))
    reasons.extend(_grid_shape_degenerate_reasons(d))
    reasons.extend(_scorer_diagnostics_error_reasons(d))
    reasons.extend(_declared_but_unrun_probe_reasons(d))

    arm_reason = _byte_identical_online_arm_reason(d)
    if arm_reason is not None and not _has_distinct_arm_evidence(d):
        reasons.append(arm_reason)

    if not reasons and not _has_nondegenerate_lever_evidence(d):
        reasons.extend(_zero_lever_delta_reasons(d))

    if not reasons:
        return

    unique_reasons = list(dict.fromkeys(reasons))
    flags.append(
        Flag(
            kind=LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND,
            severity=_lever_exercise_severity(d),
            detail=(
                "lever-exercise-evidence-degenerate: artifact declares a generation/exploration "
                "lever, but its exercise evidence is degenerate: " + "; ".join(unique_reasons[:6])
            ),
        )
    )


def _has_positive_top_level_metric(d: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key, value in d.items():
        if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
            continue
        kl = str(key).lower()
        if (
            any(kl == wanted or kl.endswith(f"_{wanted}") for wanted in keys)
            and _is_finite_number(value)
            and float(value) > 0.0
        ):
            return True
    return False


def _has_positive_metric_pair(d: dict[str, Any], pairs: tuple[tuple[str, str], ...]) -> bool:
    for left_key, right_key in pairs:
        left = _finite_float(d, left_key)
        right = _finite_float(d, right_key)
        if left is not None and right is not None and left > right:
            return True
    return False


def _top_level_field_name_text(d: dict[str, Any]) -> str:
    return " ".join(
        str(key).lower() for key in d if key not in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS
    )


def _has_first_party_qd_result_field(d: dict[str, Any]) -> bool:
    for key in d:
        if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
            continue
        kl = str(key).lower()
        if any(marker in kl for marker in _QD_FIRST_PARTY_FIELD_MARKERS):
            return True
    return False


def _qd_first_party_claim_text(d: dict[str, Any]) -> str:
    return f"{_claim_text(d, _QD_CLAIM_TEXT_KEYS)} {_top_level_field_name_text(d)}"


def _has_qd_context(d: dict[str, Any]) -> bool:
    if _is_aggregation_only(d):
        return False
    text = _qd_first_party_claim_text(d)
    arc_context = _is_arc_artifact(d) or _has_first_party_qd_result_field(d)
    return arc_context and (_has_marker(text, _QD_CONTEXT_MARKERS) or _has_marker(text, ("qd",)))


def _claims_qd_energy_fitness_claim(d: dict[str, Any]) -> bool:
    if not _has_qd_context(d):
        return False
    text = _qd_first_party_claim_text(d)
    return _has_marker(text, _QD_GENERATION_MARKERS) or _has_first_party_qd_result_field(d)


def _claims_qd_energy_fitness_generation_win(d: dict[str, Any]) -> bool:
    if not _has_qd_context(d):
        return False
    if d.get("winner_generated") is True:
        return True
    winner_count = _finite_float(d, "winner_generated_count")
    if winner_count is not None and winner_count > 0.0:
        return True
    return _has_positive_top_level_metric(d, _QD_POSITIVE_DELTA_KEYS) or _has_positive_metric_pair(
        d, _QD_BASELINE_PAIRS
    )


def check_qd_random_mutation_ablation_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag QD generation wins that do not beat random-mutation ablation."""
    if not _claims_qd_energy_fitness_claim(d):
        return
    # TYPED (2026-07-27 review finding 6): this gate asks "did the ablation PASS", a
    # boolean. Prose under the key used to satisfy it by being truthy.
    ablation_values = _typed_field_values(d, "random_mutation_ablation_passed", "bool")
    ablation_omitted = not ablation_values
    if ablation_omitted:
        flags.append(
            Flag(
                kind=QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND,
                severity="warn",
                detail=(
                    "qd-random-mutation-ablation-omitted: QD / energy-fitness "
                    "generation claim omits random_mutation_ablation_passed. "
                    "Report the random-mutation/no-energy-fitness ablation before "
                    "attributing the lift to energy fitness."
                ),
            )
        )
    if not _claims_qd_energy_fitness_generation_win(d):
        return
    if any(value is True for value in ablation_values):
        return
    flags.append(
        Flag(
            kind=QD_WITHOUT_RANDOM_MUTATION_ABLATION_KIND,
            severity="critical",
            detail=(
                "qd-without-random-mutation-ablation: artifact claims a QD / "
                "energy-fitness generation win, but random_mutation_ablation_passed "
                "is false or absent. The win must beat the "
                "random-mutation/no-energy-fitness ablation before it can be "
                "credited to energy fitness rather than search branching."
            ),
        )
    )


def _has_value_routing_context(d: dict[str, Any]) -> bool:
    text = f"{_claim_text(d, _VALUE_ROUTING_CLAIM_TEXT_KEYS)} {_field_name_text(d)}"
    return _is_arc_artifact(d) and _has_marker(text, _VALUE_ROUTING_CONTEXT_MARKERS)


def _claims_value_routing_live_claim(d: dict[str, Any]) -> bool:
    if not _has_value_routing_context(d):
        return False
    text = f"{_claim_text(d, _VALUE_ROUTING_CLAIM_TEXT_KEYS)} {_field_name_text(d)}"
    return (
        d.get("solve_provenance") == "live_agent_self_discovery"
        or _has_marker(text, _ARC_LIVE_CONTEXT_MARKERS)
        or "live_" in text
    )


def _claims_value_routing_live_win(d: dict[str, Any]) -> bool:
    if not _claims_value_routing_live_claim(d):
        return False
    return _has_positive_top_level_metric(
        d, _VALUE_ROUTING_POSITIVE_DELTA_KEYS
    ) or _has_positive_metric_pair(d, _VALUE_ROUTING_BASELINE_PAIRS)


def check_value_routing_cost_control_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag value-routing wins that do not report feature cost and no-timeout."""
    if not _claims_value_routing_live_claim(d):
        return
    # TYPED, not merely present (2026-07-27 review finding 6). A prose sentence stored under
    # either key used to clear this WARN outright -- see _typed_field_values' docstring.
    cost_values = _typed_field_values(d, "per_node_feature_cost_ms", "number")
    timeout_values = _typed_field_values(d, "sim_timed_out", "bool")
    omitted: list[str] = []
    if not cost_values:
        omitted.append("per_node_feature_cost_ms")
    if not timeout_values:
        omitted.append("sim_timed_out")
    if omitted:
        flags.append(
            Flag(
                kind=VALUE_ROUTING_COST_CONTROL_OMITTED_KIND,
                severity="warn",
                detail=(
                    "value-routing-cost-control-omitted: value-routing live "
                    f"claim omits {', '.join(omitted)}. Report finite "
                    "per_node_feature_cost_ms and sim_timed_out=false before "
                    "attributing live lift to value routing."
                ),
            )
        )
    if not _claims_value_routing_live_win(d):
        return
    cost_ok = any(_is_finite_number(value) for value in cost_values)
    timeout_ok = bool(timeout_values) and all(value is False for value in timeout_values)
    if cost_ok and timeout_ok:
        return
    flags.append(
        Flag(
            kind=VALUE_ROUTING_WITHOUT_COST_CONTROL_KIND,
            severity="critical",
            detail=(
                "value-routing-without-cost-control: artifact claims a "
                "value-routing live first-win/solve-rate lift without proving "
                "the cost fix is the controlled variable. Report finite "
                "per_node_feature_cost_ms and sim_timed_out=false; a timeout "
                "or missing cost control means the apparent win may be the "
                "baseline finally finishing rather than a real signal lift."
            ),
        )
    )


def _numeric_leaf_values(value: Any) -> list[float]:
    if _is_finite_number(value):
        return [float(value)]
    if isinstance(value, dict):
        values: list[float] = []
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            values.extend(_numeric_leaf_values(nested))
        return values
    if isinstance(value, list):
        values: list[float] = []
        for item in value:
            values.extend(_numeric_leaf_values(item))
        return values
    return []


def _bool_leaf_values(value: Any) -> list[bool]:
    if isinstance(value, bool):
        return [value]
    if isinstance(value, dict):
        values: list[bool] = []
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            values.extend(_bool_leaf_values(nested))
        return values
    if isinstance(value, list):
        values: list[bool] = []
        for item in value:
            values.extend(_bool_leaf_values(item))
        return values
    return []


def _max_real_field_number(d: dict[str, Any], wanted_key: str) -> float | None:
    values: list[float] = []
    for value in _real_field_values(d, wanted_key):
        values.extend(_numeric_leaf_values(value))
    return max(values) if values else None


def _real_field_has_true(d: dict[str, Any], wanted_key: str) -> bool:
    return any(
        leaf is True
        for value in _real_field_values(d, wanted_key)
        for leaf in _bool_leaf_values(value)
    )


def _l2_goal_induction_text(d: dict[str, Any]) -> str:
    return f"{_claim_text(d, _L2_GOAL_INDUCTION_CLAIM_TEXT_KEYS)} {_field_name_text(d)}"


def _has_l2_goal_induction_context(d: dict[str, Any]) -> bool:
    text = _l2_goal_induction_text(d)
    return _is_arc_artifact(d) and _has_marker(text, _L2_GOAL_INDUCTION_CONTEXT_MARKERS)


def _claims_l2_goal_induction_win(d: dict[str, Any]) -> bool:
    if not _has_l2_goal_induction_context(d):
        return False
    text = _l2_goal_induction_text(d)
    reached_level = _max_real_field_number(d, "generic_agent_reached_level")
    if reached_level is not None and reached_level >= 2.0:
        return True
    reproduced = _max_real_field_number(d, "reproduced_levels")
    if reproduced is None or reproduced <= 0.0:
        return False
    if not _has_marker(text, _L2_CLAIM_MARKERS):
        return False
    if _has_marker(text, _L2_NULL_TEXT_MARKERS):
        return False
    return _has_marker(text, _L2_WIN_TEXT_MARKERS)


def check_l2_goal_induction_satisfiability_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag L2 induction wins missing satisfiable-goal and reachable-plan controls."""
    if not _claims_l2_goal_induction_win(d):
        return
    # TYPED (2026-07-27 review finding 6): both are yes/no controls, so a string under
    # either key must NOT count as having reported the control.
    goal_values = _typed_field_values(d, "goal_predicate_satisfiable", "bool")
    plan_values = _typed_field_values(d, "l2_plan_reaches_goal", "bool")
    omitted = []
    if not goal_values:
        omitted.append("goal_predicate_satisfiable")
    if not plan_values:
        omitted.append("l2_plan_reaches_goal")
    if omitted:
        flags.append(
            Flag(
                kind=L2_GOAL_SATISFIABILITY_CHECK_OMITTED_KIND,
                severity="warn",
                detail=(
                    "l2-goal-satisfiability-check-omitted: L2 goal-induction "
                    f"win claim omits {', '.join(omitted)}. Report "
                    "goal_predicate_satisfiable=true and l2_plan_reaches_goal=true "
                    "before crediting an L2-via-induction win."
                ),
            )
        )
    if _real_field_has_true(d, "goal_predicate_satisfiable") and _real_field_has_true(
        d, "l2_plan_reaches_goal"
    ):
        return
    flags.append(
        Flag(
            kind=L2_GOAL_INDUCTION_WITHOUT_SATISFIABILITY_CHECK_KIND,
            severity="critical",
            detail=(
                "l2-goal-induction-without-satisfiability-check: artifact "
                "claims a generic-agent L2 win via goal induction, but does "
                "not report goal_predicate_satisfiable=true and "
                "l2_plan_reaches_goal=true. The apparent L2 win may be a "
                "constant-False degenerate-goal vacuous pass through the "
                "DYNAMICS-only held-out gate."
            ),
        )
    )


def _has_positive_multilevel_solve_rate(d: dict[str, Any]) -> bool:
    for key in _MULTI_LEVEL_RATE_KEYS:
        for value in _real_field_values(d, key):
            if any(number > 0.0 for number in _numeric_leaf_values(value)):
                return True
    return False


def _harness_target_levels(value: Any, d: dict[str, Any]) -> float | None:
    candidates = _numeric_leaf_values(value.get("target_levels")) if isinstance(value, dict) else []
    if not candidates:
        top_level = _max_real_field_number(d, "target_levels")
        return top_level
    return max(candidates)


def _harness_break_at_first_win(value: Any, d: dict[str, Any]) -> list[bool]:
    values = _bool_leaf_values(value.get("break_at_first_win")) if isinstance(value, dict) else []
    if values:
        return values
    return [
        leaf
        for candidate in _real_field_values(d, "break_at_first_win")
        for leaf in _bool_leaf_values(candidate)
    ]


def _has_fixed_multilevel_metric_harness(d: dict[str, Any]) -> bool:
    harness_values = _real_field_values(d, "metric_harness_fixed")
    candidates = harness_values or [d]
    for value in candidates:
        target_levels = _harness_target_levels(value, d)
        break_values = _harness_break_at_first_win(value, d)
        if (
            target_levels is not None
            and target_levels >= 2.0
            and any(flag is False for flag in break_values)
        ):
            return True
    return False


def check_multilevel_nondegenerate_metric_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag positive multi-level solve-rate claims missing the fixed harness."""
    if not _is_arc_artifact(d):
        return
    if not _has_positive_multilevel_solve_rate(d):
        return
    # TYPED (2026-07-27 review finding 6): metric_harness_fixed and break_at_first_win are
    # booleans and target_levels is a count; prose under any of them is not the control.
    harness_values = _typed_field_values(d, "metric_harness_fixed", "structured")
    has_equivalent_top_level = bool(
        _typed_field_values(d, "target_levels", "number")
        and _typed_field_values(d, "break_at_first_win", "bool")
    )
    if not harness_values and not has_equivalent_top_level:
        flags.append(
            Flag(
                kind=MULTI_LEVEL_NONDEGENERATE_METRIC_OMITTED_KIND,
                severity="warn",
                detail=(
                    "multi-level-nondegenerate-metric-omitted: positive "
                    "multi-level solve-rate claim omits metric_harness_fixed "
                    "or equivalent target_levels/break_at_first_win fields."
                ),
            )
        )
    if _has_fixed_multilevel_metric_harness(d):
        return
    flags.append(
        Flag(
            kind=MULTI_LEVEL_WITHOUT_NONDEGENERATE_METRIC_KIND,
            severity="critical",
            detail=(
                "multi-level-without-nondegenerate-metric: artifact reports a "
                "positive multi-level solve-rate without a fixed "
                "target_levels>=2 and break_at_first_win=false harness. The "
                "metric may be the degenerate 0.0-by-construction artifact "
                "rather than evidence that the live agent can attempt depth >=2."
            ),
        )
    )


def _subgoal_search_text(d: dict[str, Any]) -> str:
    return f"{_claim_text(d, _SUBGOAL_SEARCH_CLAIM_TEXT_KEYS)} {_field_name_text(d)}"


def _has_subgoal_search_context(d: dict[str, Any]) -> bool:
    text = _subgoal_search_text(d)
    return _is_arc_artifact(d) and _has_marker(text, _SUBGOAL_SEARCH_CONTEXT_MARKERS)


def _claims_subgoal_search_new_level_win(d: dict[str, Any]) -> bool:
    if not _has_subgoal_search_context(d):
        return False
    text = _subgoal_search_text(d)
    if _has_marker(text, _SUBGOAL_SEARCH_NULL_TEXT_MARKERS):
        return False
    reproduced = _max_real_field_number(d, "reproduced_levels")
    if reproduced is not None and reproduced >= 1.0:
        return True
    reached = _max_real_field_number(d, "generic_agent_reached_level")
    return reached is not None and reached > 0.0 and _has_marker(text, _SUBGOAL_SEARCH_WIN_MARKERS)


def _nontrivial_subgoal_decomposition(value: Any) -> bool:
    if isinstance(value, list):
        meaningful = [item for item in value if item not in (None, "")]
        if len(meaningful) >= 2:
            return True
        return any(_nontrivial_subgoal_decomposition(item) for item in meaningful)
    if isinstance(value, dict):
        nested_values = [
            nested
            for key, nested in value.items()
            if key not in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS and nested not in (None, "")
        ]
        if len(nested_values) >= 2:
            return True
        return any(_nontrivial_subgoal_decomposition(item) for item in nested_values)
    if isinstance(value, str):
        return value.count("->") >= 2
    return False


def _has_nontrivial_subgoal_decomposition(d: dict[str, Any]) -> bool:
    return any(
        _nontrivial_subgoal_decomposition(value)
        for value in _real_field_values(d, "subgoal_decomposition")
    )


def _real_field_all_true(d: dict[str, Any], wanted_key: str) -> bool:
    leaves = [
        leaf for value in _real_field_values(d, wanted_key) for leaf in _bool_leaf_values(value)
    ]
    return bool(leaves) and all(leaf is True for leaf in leaves)


def _subgoal_ablations_strictly_lower(d: dict[str, Any]) -> bool:
    reached = _max_real_field_number(d, "generic_agent_reached_level")
    no_subgoal = _max_real_field_number(d, "no_subgoal_ablation_reached_level")
    random_subgoal = _max_real_field_number(d, "random_subgoal_ablation_reached_level")
    return (
        reached is not None
        and no_subgoal is not None
        and random_subgoal is not None
        and no_subgoal < reached
        and random_subgoal < reached
    )


def check_subgoal_search_decomposition_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag subgoal-search wins missing decomposition and ablation evidence."""
    if not _claims_subgoal_search_new_level_win(d):
        return
    omitted = [
        key for key in _SUBGOAL_SEARCH_REQUIRED_EVIDENCE_KEYS if not _real_field_values(d, key)
    ]
    if omitted:
        flags.append(
            Flag(
                kind=SUBGOAL_DECOMPOSITION_EVIDENCE_OMITTED_KIND,
                severity="warn",
                detail=(
                    "subgoal-decomposition-evidence-omitted: hierarchical "
                    f"subgoal-search new-level claim omits {', '.join(omitted)}. "
                    "Report subgoal_decomposition, per_subgoal_reachable, "
                    "no_subgoal_ablation_reached_level, "
                    "random_subgoal_ablation_reached_level, and "
                    "offline_reproduced before crediting a subgoal-search win."
                ),
            )
        )
    evidence_ok = (
        _has_nontrivial_subgoal_decomposition(d)
        and _real_field_all_true(d, "per_subgoal_reachable")
        and _subgoal_ablations_strictly_lower(d)
        and _real_field_has_true(d, "offline_reproduced")
    )
    if evidence_ok:
        return
    flags.append(
        Flag(
            kind=SUBGOAL_SEARCH_WITHOUT_DECOMPOSITION_EVIDENCE_KIND,
            severity="critical",
            detail=(
                "subgoal-search-without-decomposition-evidence: artifact claims "
                "a generic-agent new-level win via hierarchical subgoal search "
                "without proving a nontrivial subgoal_decomposition, true "
                "per_subgoal_reachable evidence, no_subgoal_ablation_reached_level "
                "and random_subgoal_ablation_reached_level both strictly lower "
                "than generic_agent_reached_level, and offline_reproduced=true. "
                "The apparent win may be flat search mislabeled as subgoal "
                "search, or a 'subgoal' that is only the global goal."
            ),
        )
    )


def _generation_coverage_text(d: dict[str, Any]) -> str:
    return f"{_claim_text(d, _GENERATION_COVERAGE_CLAIM_TEXT_KEYS)} {_field_name_text(d)}"


def _has_generation_coverage_context(d: dict[str, Any]) -> bool:
    text = _generation_coverage_text(d)
    return _is_arc_artifact(d) and _has_marker(text, _GENERATION_COVERAGE_CONTEXT_MARKERS)


def _claims_generation_coverage_up(d: dict[str, Any]) -> bool:
    if not _has_generation_coverage_context(d):
        return False
    text = _generation_coverage_text(d)
    if _has_marker(text, _GENERATION_COVERAGE_NULL_TEXT_MARKERS):
        return False
    for key in _GENERATION_COVERAGE_POSITIVE_METRIC_KEYS:
        value = _max_real_field_number(d, key)
        if value is not None and value > 0.0:
            return True
    if not _has_marker(text, _GENERATION_COVERAGE_WIN_MARKERS):
        return False
    return any(
        (value := _max_real_field_number(d, key)) is not None and value > 0.0
        for key in _GENERATION_COVERAGE_VALUE_KEYS
    )


def check_generation_coverage_baseline_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag coverage-up claims that do not report the flat-search baseline."""
    if not _claims_generation_coverage_up(d):
        return
    if _real_field_values(d, "candidate_generation_coverage_flat_baseline"):
        return
    flags.append(
        Flag(
            kind=GENERATION_COVERAGE_BASELINE_OMITTED_KIND,
            severity="warn",
            detail=(
                "generation-coverage-baseline-omitted: candidate-generation "
                "coverage-up claim omits candidate_generation_coverage_flat_baseline. "
                "Report the matched flat-search baseline before crediting "
                "coverage lift to generation."
            ),
        )
    )
    flags.append(
        Flag(
            kind=GENERATION_COVERAGE_WITHOUT_BASELINE_KIND,
            severity="critical",
            detail=(
                "generation-coverage-without-baseline: artifact claims "
                "candidate-generation coverage rose but does not report "
                "candidate_generation_coverage_flat_baseline. The coverage "
                "number is unfalsifiable without the matched flat-search "
                "control because a candidate pool can always contain the winner "
                "without demonstrating a generation gain."
            ),
        )
    )


def _novelty_proposal_text(d: dict[str, Any]) -> str:
    return f"{_claim_text(d, _NOVELTY_PROPOSAL_CLAIM_TEXT_KEYS)} {_field_name_text(d)}"


def _has_novelty_proposal_context(d: dict[str, Any]) -> bool:
    text = _novelty_proposal_text(d)
    return _is_arc_artifact(d) and _has_marker(text, _NOVELTY_PROPOSAL_CONTEXT_MARKERS)


def _claims_controllable_novelty_new_level_win(d: dict[str, Any]) -> bool:
    if not _has_novelty_proposal_context(d):
        return False
    text = _novelty_proposal_text(d)
    if _has_marker(text, _NOVELTY_PROPOSAL_NULL_TEXT_MARKERS):
        return False
    reproduced = _max_real_field_number(d, "reproduced_levels")
    if reproduced is not None and reproduced >= 1.0:
        return True
    reached = _max_real_field_number(d, "generic_agent_reached_level")
    return (
        reached is not None and reached > 0.0 and _has_marker(text, _NOVELTY_PROPOSAL_WIN_MARKERS)
    )


def _novelty_ablations_strictly_lower(d: dict[str, Any]) -> bool:
    reached = _max_real_field_number(d, "generic_agent_reached_level")
    no_novelty = _max_real_field_number(d, "no_novelty_ablation_reached_level")
    cosmetic = _max_real_field_number(d, "cosmetic_novelty_ablation_reached_level")
    return (
        reached is not None
        and no_novelty is not None
        and cosmetic is not None
        and no_novelty < reached
        and cosmetic < reached
    )


def check_novelty_proposal_ablation_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag controllable-novelty wins missing lower novelty ablations."""
    if not _claims_controllable_novelty_new_level_win(d):
        return
    omitted = [
        key for key in _NOVELTY_PROPOSAL_REQUIRED_EVIDENCE_KEYS if not _real_field_values(d, key)
    ]
    if omitted:
        flags.append(
            Flag(
                kind=NOVELTY_PROPOSAL_ABLATION_OMITTED_KIND,
                severity="warn",
                detail=(
                    "novelty-proposal-ablation-omitted: controllable-novelty "
                    f"new-level claim omits {', '.join(omitted)}. Report "
                    "no_novelty_ablation_reached_level, "
                    "cosmetic_novelty_ablation_reached_level, and "
                    "offline_reproduced before crediting a controllable-novelty win."
                ),
            )
        )
    if _novelty_ablations_strictly_lower(d) and _real_field_has_true(d, "offline_reproduced"):
        return
    flags.append(
        Flag(
            kind=NOVELTY_PROPOSAL_WITHOUT_ABLATION_KIND,
            severity="critical",
            detail=(
                "novelty-proposal-without-ablation: artifact claims a "
                "generic-agent new-level win via controllable novelty without "
                "proving no_novelty_ablation_reached_level and "
                "cosmetic_novelty_ablation_reached_level are both strictly "
                "lower than generic_agent_reached_level, and "
                "offline_reproduced=true. The apparent win may be flat "
                "exploration mislabeled as controllable novelty, or the "
                "controllability gate may add nothing over cosmetic novelty."
            ),
        )
    )


def _proposal_filter_text(d: dict[str, Any]) -> str:
    return f"{_claim_text(d, _PROPOSAL_FILTER_CLAIM_TEXT_KEYS)} {_field_name_text(d)}"


def _has_proposal_filter_context(d: dict[str, Any]) -> bool:
    text = _proposal_filter_text(d)
    return _is_arc_artifact(d) and _has_marker(text, _PROPOSAL_FILTER_CONTEXT_MARKERS)


def _claims_proposal_filter_coverage_up(d: dict[str, Any]) -> bool:
    if not _has_proposal_filter_context(d):
        return False
    text = _proposal_filter_text(d)
    if _has_marker(text, _PROPOSAL_FILTER_NULL_TEXT_MARKERS):
        return False
    for key in _PROPOSAL_FILTER_POSITIVE_METRIC_KEYS:
        value = _max_real_field_number(d, key)
        if value is not None and value > 0.0:
            return True
    filter_coverage = _max_real_field_number(d, "candidate_generation_coverage_filter")
    blind_baseline = _max_real_field_number(d, "candidate_generation_coverage_blind_baseline")
    if (
        filter_coverage is not None
        and blind_baseline is not None
        and filter_coverage > blind_baseline
    ):
        return True
    return (
        filter_coverage is not None
        and filter_coverage > 0.0
        and _has_marker(text, _PROPOSAL_FILTER_WIN_MARKERS)
    )


def _has_finite_real_field_number(d: dict[str, Any], wanted_key: str) -> bool:
    return _max_real_field_number(d, wanted_key) is not None


def check_proposal_filter_heldout_rejection_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag program-synthesis coverage wins missing held-out rejection evidence."""
    if not _claims_proposal_filter_coverage_up(d):
        return
    omitted = [
        key for key in _PROPOSAL_FILTER_REQUIRED_EVIDENCE_KEYS if not _real_field_values(d, key)
    ]
    if omitted:
        flags.append(
            Flag(
                kind=PROPOSAL_FILTER_HELDOUT_REJECTION_OMITTED_KIND,
                severity="warn",
                detail=(
                    "proposal-filter-heldout-rejection-omitted: "
                    "program-synthesis coverage-up claim omits "
                    f"{', '.join(omitted)}. Report heldout_programs_rejected "
                    "and candidate_generation_coverage_blind_baseline before "
                    "crediting proposal-filter coverage lift."
                ),
            )
        )
    evidence_ok = all(
        _has_finite_real_field_number(d, key) for key in _PROPOSAL_FILTER_REQUIRED_EVIDENCE_KEYS
    )
    if evidence_ok:
        return
    flags.append(
        Flag(
            kind=PROPOSAL_FILTER_WITHOUT_HELDOUT_REJECTION_KIND,
            severity="critical",
            detail=(
                "proposal-filter-without-heldout-rejection: artifact claims "
                "program-synthesis candidate-generation coverage rose without "
                "reporting finite heldout_programs_rejected and "
                "candidate_generation_coverage_blind_baseline fields. The "
                "coverage number is unfalsifiable without proof that held-out "
                "rejection actually ran and without the matched blind-proposal "
                "baseline; it may be experts_overfit_prefix leakage."
            ),
        )
    )


def _perception_overclaim_text(d: dict[str, Any]) -> str:
    return f"{_claim_text(d, _PERCEPTION_CLAIM_TEXT_KEYS)} {_field_name_text(d)}"


def _has_perception_context(d: dict[str, Any]) -> bool:
    text = _perception_overclaim_text(d)
    return _is_arc_artifact(d) and _has_marker(text, _PERCEPTION_CONTEXT_MARKERS)


def _has_positive_perception_firstwin_metric(d: dict[str, Any]) -> bool:
    return _has_positive_top_level_metric(d, _PERCEPTION_POSITIVE_METRIC_KEYS)


def _claims_perception_attributable_win(d: dict[str, Any]) -> bool:
    if not _has_perception_context(d):
        return False
    text = _perception_overclaim_text(d)
    if _has_marker(text, _PERCEPTION_NULL_TEXT_MARKERS):
        return False
    reproduced = _max_real_field_number(d, "reproduced_levels")
    if reproduced is not None and reproduced >= 1.0:
        return True
    reached = _max_real_field_number(d, "generic_agent_reached_level")
    if reached is not None and reached > 0.0 and _has_marker(text, _PERCEPTION_WIN_MARKERS):
        return True
    return _has_marker(text, _PERCEPTION_WIN_MARKERS) and _has_positive_perception_firstwin_metric(
        d
    )


def _perception_order1_ablation_strictly_lower(d: dict[str, Any]) -> bool:
    reached = _max_real_field_number(d, "generic_agent_reached_level")
    order1 = _max_real_field_number(d, "order1_ablation_reached_level")
    return reached is not None and order1 is not None and order1 < reached


def check_perception_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag perception-attributable wins missing order-1 ablation evidence."""
    if not _claims_perception_attributable_win(d):
        return
    omitted = [key for key in _PERCEPTION_REQUIRED_EVIDENCE_KEYS if not _real_field_values(d, key)]
    if omitted:
        flags.append(
            Flag(
                kind=PERCEPTION_OVERCLAIM_OMITTED_KIND,
                severity="warn",
                detail=(
                    "perception-overclaim-omitted: object-centric/relational "
                    f"first-win or new-level claim omits {', '.join(omitted)}. "
                    "Report order1_ablation_reached_level and "
                    "offline_reproduced before crediting a perception win."
                ),
            )
        )
    if _perception_order1_ablation_strictly_lower(d) and _real_field_has_true(
        d, "offline_reproduced"
    ):
        return
    flags.append(
        Flag(
            kind=PERCEPTION_OVERCLAIM_KIND,
            severity="critical",
            detail=(
                "perception-overclaim: artifact claims an "
                "object-centric/relational representation first-win or new-level "
                "win without proving order1_ablation_reached_level is strictly "
                "lower than generic_agent_reached_level and "
                "offline_reproduced=true. The apparent win may be a search-budget "
                "win mislabeled as a perception representation win."
            ),
        )
    )


_WORLD_MODEL_TRUST_RATE_KEYS = (
    "world_model_trust_pass_rate",
    "world_model_trust_pass_rate_new",
    "world_model_trust_pass_rate_integrated",
)
_WORLD_MODEL_TRUST_VERDICT_MARKERS = (
    "world_model_trust_energy_pass_rate_up",
    "world_model_trust_pass_rate_up",
    "integrated_world_model_trust_raised",
)
_GRID_CHANGING_CORRECT_KEYS = (
    "n_correct_grid_changing_transitions",
    "correct_grid_changing_transitions",
    "grid_changing_transitions_correct",
    "n_changes_correct",
    "correct_changed_cells",
    "new_correct_changed_cells",
    "heldout_correct_changed_cells",
)
_WORLD_MODEL_METADATA_KEYS = frozenset(
    {
        "field_principles",
        "required_artifact_fields",
        "tests_added",
        "tests_added_pass",
    }
)


def _claims_world_model_trust_pass(d: dict[str, Any]) -> bool:
    """True when an ARC artifact affirmatively claims a world-model trust pass."""
    text = f"{d.get('experiment', '')} {d.get('honest_verdict', '')}".lower()
    if not (_is_arc_artifact(d) or "world_model_trust" in text):
        return False
    if any(marker in text for marker in _WORLD_MODEL_TRUST_VERDICT_MARKERS):
        return True
    numerator = d.get("trust_pass_numerator")
    if _is_finite_number(numerator) and float(numerator) > 0:
        return True
    for key in _WORLD_MODEL_TRUST_RATE_KEYS:
        value = d.get(key)
        if _is_finite_number(value) and float(value) > 0.0:
            return True
    return False


def _grid_changing_correct_evidence(value: Any) -> tuple[str, float] | None:
    """Find positive evidence for at least one correctly predicted real change."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in _WORLD_MODEL_METADATA_KEYS:
                continue
            kl = str(key).lower()
            if _is_finite_number(nested) and any(k in kl for k in _GRID_CHANGING_CORRECT_KEYS):
                if float(nested) > 0.0:
                    return str(key), float(nested)
            found = _grid_changing_correct_evidence(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _grid_changing_correct_evidence(item)
            if found is not None:
                return found
    return None


# S2-style engine-selection schema signal (scopes the check; closes FN-1 renamed-delta
# evasion + FP-1 incidental-energy_delta false positive). The diversity-spread epsilon
# requires a MEANINGFUL outcome range (closes FN-4 float-noise manufactured diversity).
_S2_SCHEMA_TOKENS = ("s2", "offpath", "off_path", "trust_gate", "engine_selection", "engine_select")
_EFFECTIVE_SPREAD_EPS = 1e-3
_NO_VALUE_VERDICT_TOKENS = (
    "no_live_trust_value",
    "no_trust",
    "no_value",
    "bounded",
    "does_not_beat",
    "below_control",
    "inconclusive",
    "negative_result",
    "ties_control",
    "no_delta",
    "null",
)
_PASS_VERDICT_TOKENS = ("authorizes_s3", "trust_gate", "_passes", "_pass_", "success_")


def _has_s2_schema_signal(d: dict[str, Any]) -> bool:
    blob = " ".join(
        str(d.get(k, ""))
        for k in ("experiment", "experiment_id", "schema", "honest_verdict", "title")
    ).lower()
    return any(t in blob for t in _S2_SCHEMA_TOKENS)


def _engine_selection_game_rows(d: dict[str, Any]) -> list[Any] | None:
    """Per-game candidate rows for an S2-style engine-SELECTION comparison, or None.
    Recognized by `game_results` rows carrying `candidate_rows` AND (an S2 schema
    signal s2/offpath/trust_gate/engine_selection OR a top-level delta/margin key).
    Scoping to the S2 schema (not an incidental field-name substring) closes the
    renamed-delta evasion (FN-1) and the incidental-`energy_delta` FP (FP-1)."""
    gr = d.get("game_results")
    if not isinstance(gr, list) or not gr:
        return None
    if not any(isinstance(g, dict) and "candidate_rows" in g for g in gr):
        return None
    has_delta = any(("delta" in k.lower() or "margin" in k.lower()) for k in d)
    if not (_has_s2_schema_signal(d) or has_delta):
        return None
    return gr


def _candidate_outcome_values(rows: list[Any]) -> list[float]:
    """The per-candidate OUTCOME metric values, field-agnostic: prefer held-out
    off-path cell_recall (the metric the S2 delta is computed from), else off-path
    structural energy, else held-out accuracy. Closes FP-2 (a diverse pool logged
    under a different valid field being mis-counted as degenerate)."""
    for field in ("heldout_cell_recall", "offpath_structural_energy", "heldout_accuracy"):
        vals = [
            float(r[field]) for r in rows if isinstance(r, dict) and _is_finite_number(r.get(field))
        ]
        if len(vals) >= 2:
            return vals
    return []


def _count_effective_selection_games(gr: list[Any]) -> tuple[int, int]:
    """(effective, total). A game is EFFECTIVE iff its candidate pool spans a
    MEANINGFUL outcome range -- max-min of the candidates' outcome metric exceeds
    _EFFECTIVE_SPREAD_EPS. Requiring a real spread (not mere distinctness) closes
    the 1e-8 float-noise manufactured-diversity evasion (FN-4); the exp4791 trap was
    different files with bit-identical off-path predictions."""
    effective = 0
    total = 0
    for g in gr:
        if not isinstance(g, dict):
            continue
        total += 1
        rows = g.get("candidate_rows")
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        vals = _candidate_outcome_values(rows)
        if vals and (max(vals) - min(vals)) > _EFFECTIVE_SPREAD_EPS:
            effective += 1
    return effective, total


def _draws_selection_conclusion(d: dict[str, Any]) -> bool:
    """True if the artifact draws a SELECTION conclusion -- a no-value/bounded null OR
    a PASS/trust win. BOTH must clear the diversity bar: a PASS is NOT exempt (closes
    the PASS-bypass, attack #3 -- a PASS off a degenerate/broken pool is also invalid).
    A zero-OR-NEGATIVE selection delta counts as a no-value read (closes the
    negative-delta evasion, FN-2: energy strictly LOSING is the clearest no-value)."""
    verdict = str(d.get("honest_verdict", "")).lower()
    if any(t in verdict for t in _NO_VALUE_VERDICT_TOKENS):
        return True
    if any(t in verdict for t in _PASS_VERDICT_TOKENS):
        return True
    for k, v in d.items():
        kl = k.lower()
        if ("delta" in kl or "margin" in kl) and (
            "energy" in kl or "select" in kl or "trust" in kl
        ):
            if _is_finite_number(v) and float(v) <= 1e-9:  # zero or negative = no value
                return True
    return False


def check_engine_selection_candidate_diversity(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag an S2-style engine-SELECTION comparison whose verdict rests on a
    behaviorally-DEGENERATE candidate pool.

    Origin: exp4791 (S2 off-path trust gate). It reported energy_minus_accuracy_delta
    == 0.0 (CI [0,0]) and the BOUNDED verdict 'no_live_trust_value' -- but 2 of 5
    games had behaviorally-IDENTICAL candidate engines (bit-identical off-path energy
    + cell_recall) and a 3rd had equal recalls, so only 2 games genuinely tested
    whether the energy beats the accuracy gate. A 0-delta from a pool with no
    behavioral diversity is a NON-TEST, not a genuine null.

    Hardened (adversarial verify wf_3c4337f4): the `required` bar is a HARD independent
    floor the producing agent cannot lower -- max(5, ceil(0.6*total), declared) -- so an
    artifact declaring `min_heldout_games: 1` can no longer dodge the check (FN-3). The
    bar also applies to PASS verdicts, not just nulls (attack #3), uses a meaningful
    spread (FN-4), treats negative deltas as no-value (FN-2), and is S2-scoped (FN-1/FP-1)."""
    gr = _engine_selection_game_rows(d)
    if gr is None:
        return
    effective, total = _count_effective_selection_games(gr)
    declared = d.get("min_heldout_games")
    declared = int(declared) if _is_finite_number(declared) else 0
    n_available = d.get("n_available_games")
    n_available = int(n_available) if _is_finite_number(n_available) else 0
    # CORPUS-COVERAGE + POWER floor (operator 2026-06-26 "require corpus coverage"):
    # S2-v2 (exp4801) satisfied the old absolute floor of 5 by testing 5 of 25 games --
    # an underpowered narrow sample (its delta CI was [-0.478, +0.004] on n=5). The bar
    # is now max(10, ceil(0.6*corpus)) where corpus = max(games tested, n_available_games
    # declared) -- so a full ~25-game corpus needs ~15 effective games, and under-declaring
    # the corpus cannot lower the absolute floor of 10. The agent's min_heldout_games may
    # only TIGHTEN (the FN-3 independent-floor property is preserved).
    corpus = max(total, n_available)
    required = max(10, math.ceil(0.6 * corpus), declared)
    if effective >= required:
        return  # enough behaviorally-diverse games across the corpus -- a genuine, powered test
    if not _draws_selection_conclusion(d):
        return
    flags.append(
        Flag(
            kind="DEGENERATE_CANDIDATE_POOL",
            severity="critical",
            detail=(
                f"Engine-selection comparison rests on a degenerate / under-covered candidate pool: "
                f"only {effective} games had behaviorally-diverse candidates (outcome spread "
                f">{_EFFECTIVE_SPREAD_EPS}) out of {total} tested / {corpus}-game corpus, below the "
                f"required {required} (max(10, 0.6*corpus); min_heldout_games may tighten but not "
                f"lower it). A selection conclusion ({d.get('honest_verdict')!r}) -- bounded-null OR "
                f"PASS -- cannot be drawn: a 0-or-negative delta from a thin/non-diverse pool is a "
                f"NON-TEST. Re-run across the FULL corpus (attempt all available games; generate "
                f"genuinely diverse engines) until >= the required effective games."
            ),
        )
    )


def check_world_model_trust_degeneracy(d: dict[str, Any], flags: list[Flag]) -> None:
    """Flag circular or degenerate ARC world-model trust-pass claims.

    A trusted world model must be oracle-distinct and must correctly predict at
    least one real grid-changing transition; otherwise an identity model can
    pass on no-op-heavy corpora without learning dynamics.
    """
    if not _claims_world_model_trust_pass(d):
        return
    problems = []
    if d.get("verifier_is_oracle") is not False:
        problems.append(f"verifier_is_oracle={d.get('verifier_is_oracle')!r}, not False")
    evidence = _grid_changing_correct_evidence(d)
    if evidence is None:
        problems.append("no positive correctly-predicted grid-changing transition evidence")
    if not problems:
        return
    flags.append(
        Flag(
            kind="WORLD_MODEL_TRUST_DEGENERACY",
            severity="critical",
            detail=(
                "World-model trust pass is degenerate or circular: "
                + "; ".join(problems)
                + ". A trust pass must declare verifier_is_oracle=false and show >=1 "
                "correctly predicted grid-changing transition so identity/no-op models "
                "cannot clear the gate."
            ),
        )
    )


def _is_arc_outer_loop_calibration_solve(d: dict[str, Any]) -> bool:
    """An ARC artifact that derives a game solve via an OFFLINE-GROUND-TRUTH BFS / per-game calibration --
    detected from the experiment NAME signature OR a declared outer-loop input flag -- AND makes a game
    solve/level claim. This catches the prose-only solve claim the structural _is_arc_solve_claim misses
    (the 2nd-recurrence incident artifact). False-positive-guarded: keyed on the experiment NAME (not the
    methodology prose) so an honest 'no offline BFS' note does not trip it."""
    if not _is_arc_artifact(d):
        return False
    name = str(d.get("experiment", ""))
    sig = bool(_ARC_CALIBRATION_NAME_RE.search(name)) or any(
        d.get(k) is True for k in _ARC_OUTER_LOOP_INPUT_FLAGS
    )
    if not sig:
        return False
    claim_text = f"{d.get('experiment', '')} {d.get('honest_verdict', '')}"
    return bool(_ARC_GAME_SOLVE_CLAIM_RE.search(claim_text))


def check_arc_outer_loop_solve(d: dict[str, Any], flags: list[Flag]) -> None:
    """ARC live-agent self-solve discipline (2026-06-22, operator-directed; 2nd recurrence).

    The ARC-AGI-3 deliverable is a LIVE agent that DISCOVERS solves to HIDDEN games on its own -- from
    its OWN attempts + runtime RE -- NOT a human/outer-loop reverse-engineering the game (reading its
    source, running an exhaustive offline ground-truth BFS, hand-calibrating a per-game model) and NOT a
    parallel solver the live agent cannot reach. Twice an outer-loop session has built an off-path solver
    and "solved" a game the live agent already solved. This is the artifact-side catch (the orphan-solver
    lint is the commit-time HARD STOP; the milestone-close arc_self_solve_audit is the AI layer). An ARC
    solve artifact must declare HOW the solve was produced; outer-loop / off-path / duplicate solves are
    flagged. CRITICAL sub-checks only fire once provenance is DECLARED, so artifacts predating the
    contract get at most a WARN (no retroactive quarantine). See CLAUDE.md "ARC Live-Path Reachability
    Discipline" + "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent".
    """
    # Catch the offline-ground-truth-BFS / per-game CALIBRATION solve FIRST -- it makes a solve claim in
    # PROSE without the structural offline_reproduced+game+level shape (the 2nd-recurrence incident artifact
    # slipped through the structural gate). This fires regardless of provenance unless explicitly labeled
    # outer_loop_re (where the outer_loop_re CRITICAL below already fires).
    if _is_arc_outer_loop_calibration_solve(d) and d.get("solve_provenance") != "outer_loop_re":
        flags.append(
            Flag(
                kind="ARC_OUTER_LOOP_SOLVE",
                severity="critical",
                detail=(
                    "ARC game-solve derived via an OFFLINE-GROUND-TRUTH BFS / per-game calibration (the "
                    "experiment name / declared inputs show it). The LIVE agent CANNOT run an exhaustive "
                    "real-env ground-truth BFS on a HIDDEN game (no oracle, no exhaustive action budget), "
                    "so this is outer-loop RE, NOT a live-agent solve. Set solve_provenance=outer_loop_re "
                    "and do not headline it as solving the game; wire a live-runnable mechanism instead."
                ),
            )
        )

    if not _is_arc_solve_claim(d):
        return
    prov = d.get("solve_provenance")
    if prov is None:
        flags.append(
            Flag(
                kind="ARC_OUTER_LOOP_SOLVE",
                severity="warn",
                detail=(
                    "ARC solve artifact does not declare solve_provenance (live_agent_self_discovery | "
                    "development_proxy | outer_loop_re). The deliverable is the LIVE agent self-discovering "
                    "hidden-game solves from its OWN attempts + runtime RE; undeclared provenance hides "
                    "outer-loop / off-path RE. Declare solve_provenance."
                ),
            )
        )
        return
    if prov not in _ARC_VALID_PROVENANCE:
        flags.append(
            Flag(
                kind="ARC_OUTER_LOOP_SOLVE",
                severity="critical",
                detail=f"Unknown solve_provenance {prov!r}; must be one of {sorted(_ARC_VALID_PROVENANCE)}.",
            )
        )
        return
    if prov == "outer_loop_re":
        flags.append(
            Flag(
                kind="ARC_OUTER_LOOP_SOLVE",
                severity="critical",
                detail=(
                    "solve_provenance=outer_loop_re: a human/outer-loop hand-RE or off-path solve is NOT "
                    "the deliverable and does NOT count as the live agent solving the game. The live agent "
                    "must generate the solve from its OWN attempts + runtime RE (reachable from "
                    "arc_competition_agent / arc_loop_solve). Wire the capability into the live path; do "
                    "not headline an outer-loop solve."
                ),
            )
        )
    bad = [k for k in _ARC_OUTER_LOOP_INPUT_FLAGS if d.get(k) is True]
    if bad and prov != "outer_loop_re":
        flags.append(
            Flag(
                kind="ARC_OUTER_LOOP_SOLVE",
                severity="critical",
                detail=(
                    f"solve_provenance={prov} but the artifact declares outer-loop-only inputs {bad}: "
                    "reading the game source / running an offline exhaustive ground-truth BFS / "
                    "hand-calibrating per game is RE the live agent CANNOT do on a HIDDEN game. Either set "
                    "solve_provenance=outer_loop_re (and do not headline) or remove the dependency."
                ),
            )
        )
    if prov != "development_proxy":
        lvl = _arc_claimed_level(d)
        reg = _arc_registry_level(str(d.get("game")))
        if lvl is not None and reg is not None and lvl <= reg:
            flags.append(
                Flag(
                    kind="ARC_OUTER_LOOP_SOLVE",
                    severity="critical",
                    detail=(
                        f"Re-solves {d.get('game')} L{lvl} but the registry already records "
                        f"levels_reproduced={reg} for it -- no NEW live capability. A milestone must "
                        "ADVANCE the live agent (a new level / a new game / a reusable method), not "
                        "re-derive an already-solved level (CLAUDE.md ARC Incremental-Progress Scoping)."
                    ),
                )
            )


def verify_artifact(path: Path, *, declared: bool | None = None) -> dict[str, Any]:
    """Run all checks on a single artifact. Return a report dict."""
    path = Path(path)
    flags: list[Flag] = []
    declared_readiness = _should_check_terminal_artifact_readiness(path, declared=declared)
    try:
        with open(path) as f:
            d_raw = json.load(f)
    except Exception as e:
        if declared_readiness:
            check_terminal_artifact_readiness(classify_artifact_path(path), flags)
        return {
            "artifact": str(path),
            "loaded": False,
            "error": str(e),
            **_flag_summary(flags),
        }

    # Skip non-dict top-level artifacts (some experiments emit a list
    # at top level — those aren't standard results files).
    if not isinstance(d_raw, dict):
        if declared_readiness:
            classification = classify_artifact_payload(
                d_raw,
                path=path,
                sha256=path_sha256(path),
            )
            check_terminal_artifact_readiness(classification, flags)
        return {
            "artifact": str(path),
            "loaded": True,
            "non_dict_top_level": True,
            **_flag_summary(flags),
        }

    if _should_check_terminal_artifact_readiness(path, d_raw, declared=declared):
        classification = classify_artifact_payload(
            d_raw,
            path=path,
            sha256=path_sha256(path),
        )
        check_terminal_artifact_readiness(classification, flags)

    schema_raw = d_raw.get("schema", "") or ""
    schema = schema_raw if isinstance(schema_raw, str) else ""

    # Skip blocked-gate artifacts — those are conductor diagnostic
    # outputs, not experimental results.
    if schema.startswith("blocked"):
        return {
            "artifact": str(path),
            "loaded": True,
            "blocked_gate_artifact": True,
            **_flag_summary(flags),
        }

    # Skip retrospective artifacts — they summarize other experiments
    # and reference compute-bound markers without invoking models
    # themselves. They have a different verification profile (their
    # rigor is "did they read the artifacts they claim to summarize"
    # which is a different check entirely).
    title_raw = d_raw.get("title") or d_raw.get("name") or ""
    title = title_raw if isinstance(title_raw, str) else ""
    is_retro = (
        "retro" in path.name.lower()
        or "retro" in schema.lower()
        or "retrospective" in title.lower()
    )
    if is_retro:
        return {
            "artifact": str(path),
            "loaded": True,
            "is_retro": True,
            "exp_id": d_raw.get("experiment") or d_raw.get("experiment_id"),
            "title": title[:80],
            **_flag_summary(flags),
        }

    # Flatten nested-metric containers (e.g. `metrics`,`report`) so
    # checks see the inner fields even when the agent emitted them
    # one level deep. Several agents emit
    # `{"metrics": {"initial_energy": ..., "final_energy": ...}}`
    # instead of top-level fields.
    d = _flatten_metrics(d_raw)
    # Unwrap {"value": ..., "principle": "..."} field annotations to their bare
    # value so every check below sees the field consistently, regardless of
    # whether the author wrote it bare or principle-annotated. See
    # _normalize_principle_wrapped_fields's docstring for the exp5161 incident
    # this fixes (176 artifacts affected corpus-wide, not a one-off).
    d = _normalize_principle_wrapped_fields(d)

    check_tautology(d, flags)
    check_implausible_perfect(d, flags)
    check_sign_anomaly(d, flags)
    check_duration_vs_claim(d, flags)
    check_sample_size(d, flags)
    check_gate_passed_without_data(d, flags)
    check_methodology_present(d, flags)
    check_preconditions_declared(d, flags)
    check_implausible_tight_ci(d, flags)
    check_false_negative_risk(d, flags)
    check_intrinsic_reward_overclaim(d, flags)
    check_goal_energy_ablation_overclaim(d, flags)
    check_qd_random_mutation_ablation_overclaim(d, flags)
    check_value_routing_cost_control_overclaim(d, flags)
    check_l2_goal_induction_satisfiability_overclaim(d, flags)
    check_multilevel_nondegenerate_metric_overclaim(d, flags)
    check_subgoal_search_decomposition_overclaim(d, flags)
    check_generation_coverage_baseline_overclaim(d, flags)
    check_lever_exercise_evidence(d, flags)
    check_novelty_proposal_ablation_overclaim(d, flags)
    check_proposal_filter_heldout_rejection_overclaim(d, flags)
    check_perception_overclaim(d, flags)
    check_ceiling_saturation(d, flags)
    check_degenerate_separation(d, flags)
    check_degenerate_controls(d, flags)
    check_circular_moat_overclaim(d, flags)
    check_verdict_class_consistency(d, flags)
    check_moat_claim_rigor(d, flags)
    check_world_model_trust_degeneracy(d, flags)
    check_engine_selection_candidate_diversity(d, flags)
    check_arc_offline_live_overclaim(d, flags)
    check_arc_outer_loop_solve(d, flags)

    verdict_raw = d_raw.get("honest_verdict") or ""
    verdict = verdict_raw if isinstance(verdict_raw, str) else ""
    return {
        "artifact": str(path),
        "loaded": True,
        "exp_id": d_raw.get("experiment") or d_raw.get("experiment_id"),
        "title": title[:80],
        "honest_verdict": verdict[:80],
        **_flag_summary(flags),
    }


def sweep_milestone_range(results_dir: Path, low: int, high: int) -> list[dict[str, Any]]:
    """Verify all experiment_NNNN_*.json files where low <= NNNN <= high."""
    reports = []
    pattern = re.compile(r"experiment_(\d+)_.*\.json$")
    for path in sorted(results_dir.glob("experiment_*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        exp = int(m.group(1))
        if low <= exp <= high:
            reports.append(verify_artifact(path))
    return reports


# High-precision fabrication-signal kinds. These are safe to backfill-stamp
# retroactively across all of history because a false positive is very unlikely:
# a duration_s far below the model-load floor while claiming a live GGUF/CUDA
# model is almost certainly fabricated, and a gate marked passed with the gated
# metric null/missing is structurally untrustworthy. TAUTOLOGY is deliberately
# NOT here — retroactively it over-flags legitimate coincidental old findings
# (e.g. abstention having no measured effect, or a tiny-corpus recall tie), so
# historical TAUTOLOGY is left to the completion-time gate + operator review.
HIGH_PRECISION_KINDS = ("DURATION_TOO_SHORT", "GATE_PASSED_WITHOUT_DATA")


def _claims_live_model(d: dict[str, Any]) -> bool:
    """True if the artifact AFFIRMATIVELY claims it ran a named live model
    (declares model_specs / target_model / models, or
    inference_substrate=live_llm_inference). DURATION_TOO_SHORT is only a real
    fabrication signal when paired with such a claim — an aggregation/audit
    artifact that merely mentions 'GGUF' in prose but declares no model didn't
    claim a live run, so a sub-floor duration is expected, not suspicious. This
    guard keeps the retroactive backfill high-precision (see exp1877/1498/1459
    aggregation false positives vs exp1851/1782 real live-claim fabrications)."""
    if _is_live_llm_inference(d):
        return True
    for key in ("model_specs", "target_model", "models", "model"):
        v = d.get(key)
        if v:
            return True
    return False


def backfill_stamps(
    paths: list[Path],
    apply: bool = False,
    kinds_filter: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Backstop for the conductor's completion-time fabrication gate.

    The gate in `research_conductor._log_experiment_completion` only fires for
    artifacts that complete inside `research_step`. Artifacts written out-of-band
    (manual reruns, batch scripts, historical pre-gate experiments) escape it.
    This sweep re-verifies a set of artifacts and stamps `flagged_adversarial`
    on any UNSTAMPED one whose critical flags intersect `kinds_filter`
    (None = any critical kind).

    NON-DESTRUCTIVE: only ADDS `flagged_adversarial: true` +
    `corrigendum_pending` (the offending flags) + a `corrigendum_note`. Never
    deletes or alters existing fields, per the never-prune discipline. Idempotent
    (skips already-stamped artifacts).

    Returns one record per artifact that HAS a qualifying critical flag, whether
    or not it was written (so dry-run reports the full scope).
    """
    out: list[dict[str, Any]] = []
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("flagged_adversarial"):
            continue
        try:
            rep = verify_artifact(p)
        except Exception:
            continue
        crit = [f for f in rep.get("flags", []) if str(f.get("severity", "")).lower() == "critical"]
        if kinds_filter is not None:
            crit = [f for f in crit if f.get("kind") in kinds_filter]
        # Precision guard: DURATION_TOO_SHORT only counts when the artifact
        # affirmatively claims a live model run. Otherwise an aggregation/audit
        # artifact that merely references compute markers in prose would be
        # mislabeled (the operator's explicit false-positive concern).
        crit = [f for f in crit if f.get("kind") != "DURATION_TOO_SHORT" or _claims_live_model(d)]
        if not crit:
            continue
        kinds = sorted({str(f.get("kind")) for f in crit})
        written = False
        if apply:
            d["flagged_adversarial"] = True
            d.setdefault("corrigendum_pending", []).extend(crit)
            d["corrigendum_note"] = (
                "backfill_stamps: flagged_adversarial added by "
                "adversarial_verify.py --backfill (completion-gate backstop). "
                "Excluded from headline / capstone aggregation."
            )
            p.write_text(json.dumps(d, indent=2))
            written = True
        out.append({"path": str(p), "kinds": kinds, "written": written})
    return out


def main(argv: list[str]) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="*", help="Specific artifact files to verify")
    parser.add_argument(
        "--milestone-range",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        help="Sweep experiment_LOW..HIGH inclusive",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument("--json", action="store_true", help="Output full JSON report")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Completion-gate backstop: scan results/ and stamp unstamped "
        "real-critical artifacts (dry-run unless --apply).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --backfill: actually write the stamps (default is dry-run).",
    )
    parser.add_argument(
        "--high-precision-only",
        action="store_true",
        help="With --backfill: restrict to %s (safe for retroactive history)."
        % ", ".join(HIGH_PRECISION_KINDS),
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=None,
        help="With --backfill: only artifacts modified within the last N hours.",
    )
    args = parser.parse_args(argv[1:])

    if args.backfill:
        import time

        paths = sorted(args.results_dir.glob("experiment_*.json"))
        if args.since_hours is not None:
            cutoff = time.time() - args.since_hours * 3600.0
            paths = [p for p in paths if p.stat().st_mtime >= cutoff]
        kinds_filter = HIGH_PRECISION_KINDS if args.high_precision_only else None
        recs = backfill_stamps(paths, apply=args.apply, kinds_filter=kinds_filter)
        mode = "APPLIED" if args.apply else "DRY-RUN"
        scope = (
            f"high-precision({','.join(HIGH_PRECISION_KINDS)})"
            if args.high_precision_only
            else "any-critical"
        )
        win = f", last {args.since_hours}h" if args.since_hours else ""
        print(
            f"[backfill {mode}] scanned {len(paths)} artifact(s){win}; scope={scope}; "
            f"{len(recs)} qualifying unstamped critical artifact(s):"
        )
        for r in recs:
            tag = "stamped" if r["written"] else "would-stamp"
            print(f"  [{tag}] {Path(r['path']).name}: {r['kinds']}")
        return 1 if recs else 0

    reports: list[dict[str, Any]] = []
    if args.milestone_range:
        reports = sweep_milestone_range(
            args.results_dir, args.milestone_range[0], args.milestone_range[1]
        )
    for a in args.artifacts:
        reports.append(verify_artifact(Path(a)))

    if not reports:
        parser.print_help()
        return 2

    flagged = [r for r in reports if r.get("flag_count", 0) > 0]

    if args.json:
        print(json.dumps({"reports": reports, "flagged_count": len(flagged)}, indent=2))
    else:
        print(f"Scanned {len(reports)} artifact(s); {len(flagged)} flagged.\n")
        for r in flagged:
            sev_label = {2: "CRITICAL", 1: "WARN", 0: "INFO"}.get(r.get("max_severity", -1), "?")
            print(
                f"[{sev_label}] exp{r.get('exp_id')} ({r.get('title')}) -- "
                f"{r.get('flag_count')} flag(s)"
            )
            for f in r.get("flags", []):
                print(f"  - {f['severity'].upper():8s} {f['kind']:30s} {f['detail']}")
            print()

    return 1 if flagged else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
