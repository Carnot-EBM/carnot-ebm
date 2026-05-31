# Verification Capability Specification

**Capability:** verification
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines formal verification helpers that bound Carnot energy-model outputs over
specified input sets. These helpers are CPU-only software proofs and do not
claim hardware correctness.

## Requirements

### REQ-VERIFY-2976: Intent-Preserving Trace-Aware DCCD Repair Protocol

The repository shall provide a deterministic Exp 2976 protocol builder that
aggregates the Exp 2963 DCCD manifest, Exp 2964 live DCCD replication, Exp
2952 taxonomy-guided repair evaluation, Exp 2953 verifier threshold policy, and
the 2026-05-24 post-.279 research sweep before writing
`results/experiment_2976_dccd_adaptrack_tracecoder_protocol_v1.json`.

The artifact shall include `honest_verdict`,
`intent_preserving_repair_protocol_ready`, `trace_execution_plan_ready`,
`downstream_min_tasks`, `required_model_specs`,
`mandatory_headline_model_ids`, `legacy_model_policy`,
`baseline_conditions`, `repair_manifest_schema`, `acceptance_gates`,
`schema_regression_guard`, `syntax_regression_guard`, `false_accept_guard`,
`prior_failure_addressed`, `inference_substrate`, and `duration_s`.

The protocol must preserve draft intent before constrained decoding, represent
AdapTrack-style backtracking, require TraceCoder-style execution traces, and
gate downstream Exp 2977 claims on deterministic schema, syntax, false-accept,
and pass-rate guards.

### SCENARIO-VERIFY-2976: Schema-Collapse DCCD Rerun Is Converted To Gates

Given the Exp 2963 protocol artifact, Exp 2964 failed live DCCD replication,
Exp 2952 taxonomy-guided repair reference, Exp 2953 threshold policy, and the
post-.279 research sweep are present,
When the Exp 2976 protocol builder runs,
Then it writes the terminal JSON artifact with a repair manifest schema
containing `draft_intent`, `constrained_patch`, `backtracking_steps`,
`execution_trace`, `verifier_result`, `schema_result`, `syntax_result`,
`false_accept_audit`, and `acceptance_reason`, requires at least 20 downstream
repair tasks across baseline, schema-only DCCD, intent-preserving DCCD, and
trace-aware repair conditions, and sets promotion gates so the Exp 2964
schema/syntax collapse and pass-rate regression cannot be accepted as progress.

### REQ-VERIFY-2977: SOTA Intent-Preserving Code-Repair Rerun

The repository shall provide an Exp 2977 rerun harness that applies the Exp
2976 intent-preserving, trace-aware repair protocol to code-repair candidates
from prior repair artifacts and writes
`results/experiment_2977_sota_intent_preserving_code_repair_v1.json`.

The harness shall call `cached_sota_pair()` before selecting any model. When a
cached SOTA GGUF pair is available, the harness shall run at least 20
code-repair tasks across baseline, schema-only DCCD, and intent-preserving
trace-aware repair conditions with local live LLM inference, store raw
candidates, runtime traces, verifier outputs, and schema/syntax diagnostics,
and compute pass@1, pass@k, schema-failure, syntax-failure, false-accept, trace
coverage, and per-model metrics. When `cached_sota_pair()` returns `None`, the
harness shall run only a bounded CPU smoke path with legacy tiny-model labels,
set `headline_result=false`, set `legacy_model_used_only_for_smoke=true`, and
mark the verdict blocked or pilot-only rather than clean.

The artifact shall include `honest_verdict`, `repair_rerun_clean`,
`headline_result`, `n_tasks`, `models_used`, `model_specs`,
`mandatory_headline_model_ids`, `legacy_model_used_only_for_smoke`,
`baseline_pass_at_1`, `schema_only_pass_at_1`,
`intent_preserving_pass_at_1`, `pass_at_1_delta`, `pass_at_k_delta`,
`schema_failure_rate_delta`, `syntax_failure_rate_delta`, `false_accept_delta`,
`runtime_trace_coverage`, `per_model_metrics`, `failures_by_category`,
`inference_substrate="live_llm_inference"`, and `duration_s`.

`repair_rerun_clean` shall be true only when `n_tasks>=20`,
`pass_at_1_delta>0`, `pass_at_k_delta>=0`,
`schema_failure_rate_delta<=0`, `syntax_failure_rate_delta<=0`,
`false_accept_delta<=0`, and `runtime_trace_coverage>=0.8`.

### SCENARIO-VERIFY-2977: Cached GGUF Gate Controls Headline Promotion

Given the Exp 2976 protocol artifact is ready and prior repair artifacts
provide at least 20 candidate repair tasks,
When the Exp 2977 rerun harness executes,
Then it first attempts `cached_sota_pair()`, preserves raw candidate and trace
evidence for every evaluated condition, reports the required metrics and
per-model breakdowns, and promotes to a clean headline result only when the
Exp 2976 pass-rate, schema, syntax, false-accept, and runtime-trace gates all
clear.

If the cached SOTA GGUF pair is unavailable, then the same harness writes the
terminal Exp 2977 JSON as a non-headline CPU-smoke artifact, reports fewer than
20 smoke tasks, keeps `repair_rerun_clean=false`, and records that legacy
models were used only for smoke coverage.

### REQ-VERIFY-2978: First-Step And Semantic-Energy Repair Telemetry Panel

The repository shall provide an Exp 2978 diagnostic telemetry panel that reads
checked-in repair and solver candidate artifacts from `.279` or `.280` and
writes `results/experiment_2978_first_step_semantic_energy_repair_telemetry_v1.json`.
The panel MUST NOT run fresh live inference unless explicitly configured, MUST
NOT modify `scripts/research_conductor.py`, MUST NOT push, and MUST NOT promote
early-prefix, first-step, logprob, confidence, or semantic-energy telemetry into
a verifier or acceptance gate.

The panel MUST require at least one candidate-level repair or solver source
artifact. It SHALL prefer Exp 2977 candidate rows when present and SHALL use Exp
2964 and Exp 2967 as fallback or additional sources. It SHALL extract one row
per candidate with final verifier outcome, schema status, syntax or solver
execution status, failure category, source artifact, model identity, and
bounded diagnostic feature values. If local token traces or logprobs are absent,
the artifact SHALL set `logprob_unavailable=true`, leave direct logprob metrics
unclaimed, and compute only artifact-derived proxy telemetry.

The terminal artifact MUST include `honest_verdict`, `telemetry_panel_ready`,
`first_step_signal_usable`, `semantic_energy_signal_usable`,
`logprob_unavailable`, `no_headline_verifier_claim=true`, `models_used`,
`mandatory_headline_model_ids`, `candidate_rows`, `calibration_metrics`,
`triage_examples`, `failure_modes_explained`,
`inference_substrate="mixed_artifact_and_optional_live_llm"`, and measured
`duration_s`. It MAY include `model_specs`, sampled or full candidate-level
rows, source checksums, run date, and methodology notes as long as every value
is derived from local artifacts or explicitly marked as optional live inference
evidence.

`telemetry_panel_ready` shall be true only when at least one candidate row is
loaded and calibration metrics are computed or honestly marked unavailable.
`first_step_signal_usable` and `semantic_energy_signal_usable` shall be true
only when the corresponding local calibration metric is defined and clears the
panel's diagnostic floor; otherwise the panel shall report the signal as
measured but not usable. `no_headline_verifier_claim` shall always remain true.

### SCENARIO-VERIFY-2978: Artifact-Only Telemetry Is Calibrated As Triage

Given Exp 2977 is present or Exp 2964 and Exp 2967 are present with
candidate-level rows,
When the Exp 2978 telemetry panel runs without live logprob traces,
Then it writes the terminal JSON artifact with the required fields, sets
`logprob_unavailable=true`, extracts candidate rows from available source
artifacts, computes AUROC or rank-correlation diagnostics only where both
failure classes and finite proxy scores exist, reports false-positive and
false-negative triage examples, explains schema, syntax, solver, and verifier
failure modes, and keeps `no_headline_verifier_claim=true`.

## Implementation Status (REQ-VERIFY-2978)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-2978 | Implemented (`python/carnot/reporting/first_step_semantic_energy_repair_telemetry_v1.py`) | Implemented (`tests/python/test_experiment_2978_first_step_semantic_energy_repair_telemetry.py`) |

### REQ-VERIFY-2979: Solver-Feedback MCS/MUS Frontier

The repository shall provide a deterministic Exp 2979 solver-feedback frontier
builder over the existing Exp 2966 exact-verifier tasks and Exp 2967 live
NL-to-Z3 formalization outcomes. The builder MUST NOT run fresh live inference,
MUST NOT modify `scripts/research_conductor.py`, and MUST write
`results/experiment_2979_solver_feedback_mcs_frontier_v1.json`.

The builder MUST check that Python can import Z3 before building the frontier.
If Z3 is unavailable, it MUST write an honest blocked artifact with exact
environment diagnostics. When Z3 is available, it MUST load Exp 2966 and Exp
2967, preserve Exp 2967 parse, execution, and solver-wrong failure categories,
define a consumable `solver_feedback` schema containing `parse_error`,
`z3_exception`, `model_counterexample`, `unsat_core_or_mus`,
`minimal_correction_hint`, `skill_label`, and
`accepted_reference_formalization`, build a small deterministic fixture with
reference formalizations and feedback examples for every Exp 2966 skill label,
and run Z3 over the fixture references.

The terminal artifact MUST include `honest_verdict`,
`mcs_feedback_schema_ready`, `frontier_upgrade_ready`,
`reference_z3_execution_rate`, `reference_solver_verified_accuracy`,
`feedback_schema`, `frontier_items`, `failure_categories_from_exp2967`,
`mcs_mus_examples`, `exp2980_input_path`,
`inference_substrate="deterministic_z3_and_artifact_generation"`, and
`duration_s`.

`mcs_feedback_schema_ready` and `frontier_upgrade_ready` shall be true only
when the schema contains all required feedback fields, the deterministic
fixture covers every Exp 2966 skill label, Z3 executes every fixture reference,
reference solver accuracy is 100%, and `exp2980_input_path` points to the
written artifact.

### SCENARIO-VERIFY-2979: Exp 2967 Failures Become Structured Feedback

Given Exp 2966 has materialized the exact-verifier frontier and Exp 2967 has
reported live NL-to-Z3 formalization failures,
When the Exp 2979 frontier builder runs with Z3 importable,
Then it writes the terminal JSON artifact, exposes a `solver_feedback` schema
that Exp 2980 can consume, maps unparseable rows to `parse_error` feedback, Z3
execution failures to `z3_exception` feedback, satisfiable wrong or
countermodel rows to `model_counterexample`, unsatisfiable diagnostic rows to
`unsat_core_or_mus`, records skill-specific `minimal_correction_hint` values,
and reports deterministic reference execution and solver accuracy from Z3.

## Implementation Status (REQ-VERIFY-2979)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-2979 | Implemented (`python/carnot/reporting/solver_feedback_mcs_frontier_v1.py`) | Implemented (`tests/python/test_experiment_2979_solver_feedback_mcs_frontier.py`) |

### REQ-VERIFY-2980: SOTA Solver Formalization Feedback V2

The repository shall provide an Exp 2980 local SOTA NL-to-Z3 formalization
runner that consumes
`results/experiment_2979_solver_feedback_mcs_frontier_v1.json`, uses
feedback-aware prompts, runs deterministic Z3 authority over every parseable
proposal, optionally applies one deterministic repair pass from the recorded
solver feedback fields, and writes
`results/experiment_2980_sota_solver_formalization_feedback_v2.json`.

The runner MUST first verify that Exp 2979 reports
`mcs_feedback_schema_ready=true` and that Python can import Z3. It MUST call
`cached_sota_pair(gpu_indices=(0, 1))` before any single-model fallback. For
headline results, `MODEL_SPECS` and `models_used` MUST include at least one of
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`. If only legacy tiny models run, the runner
MUST set `headline_result=false`, MUST NOT claim a clean solver row, and MUST
label the run as smoke-only.

For each Exp 2979 frontier item, the runner shall record the initial local-GGUF
formalization proposal, parse outcome, Z3 execution result, solver feedback
object, and final post-repair proposal/result. The deterministic repair pass
may use `parse_error`, `z3_exception`, `model_counterexample`,
`unsat_core_or_mus`, `minimal_correction_hint`, `skill_label`, and
`accepted_reference_formalization` from Exp 2979, but the artifact must expose
the resulting `feedback_repair_delta` rather than attributing the repair to raw
model capability alone.

The terminal artifact MUST include `honest_verdict`,
`formalization_feedback_clean`, `headline_result`, `n_items`, `models_used`,
`model_specs`, `mandatory_headline_model_ids`, `parseability_rate`,
`z3_execution_rate`, `solver_verified_accuracy`, `answer_accuracy`,
`tautology_flag_rate`, `per_skill_metrics`, `feedback_repair_delta`,
`failure_categories`, `solver_feedback_examples`,
`inference_substrate="live_llm_inference_plus_z3"`, and `duration_s`.

`formalization_feedback_clean` shall be true only when `headline_result=true`,
`parseability_rate>=0.50`, `z3_execution_rate>=0.50`,
`solver_verified_accuracy>=0.40`, `answer_accuracy>=0.40`, and
`tautology_flag_rate==0`.

### SCENARIO-VERIFY-2980: Solver Feedback Repairs Local Formalization Proposals

Given Exp 2979 has produced a feedback-ready solver frontier and at least one
mandated local SOTA GGUF can generate proposal text,
When the Exp 2980 runner processes each frontier item,
Then it records the initial proposal, executes parseable proposals with Z3,
attaches the explicit solver feedback object, applies at most one deterministic
feedback repair pass, computes aggregate and per-skill parseability, Z3
execution, solver accuracy, answer accuracy, tautology, failure-category, and
repair-delta metrics, and promotes a clean result only if the explicit .280
gates all clear.

## Implementation Status (REQ-VERIFY-2980)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-2980 | Implemented (`python/carnot/eval/sota_solver_formalization_feedback_v2.py`) | Implemented (`tests/python/test_experiment_2980_sota_solver_formalization_feedback.py`) |

### REQ-VERIFY-2992: SOTA Solver Formalization Provenance Reproduction

The repository shall provide an Exp 2992 local SOTA solver-feedback
reproduction runner that re-tests the Exp 2980 NL-to-Z3 result on a fixed item
set of at least 12 Exp 2966 exact-verifier items, preserves durable model and
solver provenance, and writes
`results/experiment_2992_sota_solver_formalization_provenance_reproduction_v1.json`.

The runner MUST first verify SOTA cache readiness through the repo-local SOTA
helper, CUDA/cache status, exact Z3 availability, the fixed item-set hash, and
availability of the prior Exp 2980 artifact. If those preconditions cannot
support live headline evidence, it MUST write a terminal blocked artifact rather
than promoting smoke-only or simulated evidence.

For every selected item, the runner shall record the prompt hash, raw model
output, raw output hash, solver feedback used for repair, final Z3 input, Z3
transcript hash, solver status, answer correctness, and unsat-core/MCS/MUS
evidence where applicable. Z3/runtime verification remains authoritative: LLM
self-judgment MUST NOT decide correctness, and implausibly short live-inference
durations MUST NOT set `solver_provenance_reproduced=true`.

The terminal artifact MUST include `solver_provenance_reproduced`,
`formalization_clean`, `n_items`, `parseability`, `z3_execution_rate`,
`solver_verified_accuracy`, `feedback_repair_delta`, `tautology_rate`,
`prompt_hashes_recorded`, `z3_transcript_hashes_recorded`,
`model_checksums_recorded`, `duration_seconds`, `inference_substrate`, and
`honest_verdict`.

`solver_provenance_reproduced` shall be true only when the fixed item set has at
least 12 items, at least one mandated headline GGUF produced live local output,
model checksum evidence is recorded, prompt and Z3 transcript hashes are
recorded for every item, exact Z3 executes every final formalization,
`formalization_clean=true`, and live inference duration is plausible for a
headline GGUF run.

### SCENARIO-VERIFY-2992: Larger Fixed Set Reproduces Or Falsifies Exp 2980

Given Exp 2980 previously reported a clean feedback-aware SOTA formalization
result,
When the Exp 2992 runner executes on the fixed 12+ item set with a mandated
headline GGUF and exact Z3,
Then it writes the terminal JSON artifact, compares its metrics against Exp
2980, reports any regression or non-reproduction reason, records replayable
prompt/model/Z3 provenance, and sets `honest_verdict` to reproduced, flagged, or
blocked according to the solver-backed evidence.

## Implementation Status (REQ-VERIFY-2992)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-2992 | Planned (`python/carnot/eval/sota_solver_formalization_provenance_reproduction_v1.py`) | Planned (`tests/python/test_experiment_2992_sota_solver_formalization_provenance_reproduction.py`) |

### REQ-VERIFY-2994: Prompt-To-Validator Dialogue Schema V1

The repository shall provide a bounded prompt-to-validator dialogue protocol
and deterministic harness for translating prompt-defined natural-language
requirements into executable validator trees without running live LLM
inference. The protocol document shall live under
`openspec/change-proposals/` or `ops/` and shall describe the required stages:
prompt constraints, validator tree, exact-check nodes, feedback object, and
rejection reasons.

The deterministic harness shall compile at least three fixed prompt-defined
constraint fixtures into executable validator trees. The fixture set shall
include JSON/runtime validation, Python AST/runtime validation, and exact Z3
or equivalent solver-backed validation. The harness shall evaluate known-good
and known-bad candidates for every compiled fixture and shall reject
unsupported prompt patterns instead of speculatively accepting them.

The protocol may include a STATIC-inspired sparse/static transition-table
representation for validator grammars, but it MUST present that representation
only as a schema/automaton design proposal. The artifact MUST NOT make any
local speed, latency, or acceleration claim from STATIC inspiration alone.
LLM self-judgment MUST NOT decide correctness; exact Z3/runtime validator
execution remains the acceptance authority.

The terminal artifact MUST be written to
`results/experiment_2994_prompt_validator_dialogue_schema_v1.json` and include
`prompt_validator_protocol_ready`, `protocol_doc_path`,
`deterministic_harness_path`, `n_validator_tree_fixtures`,
`exact_verifier_authority_preserved`,
`static_transition_representation_designed`, `no_speed_claim_made`,
`validation_commands`, and `honest_verdict`.

`prompt_validator_protocol_ready` shall be true only when the protocol document
exists, at least three validator-tree fixtures compile, every known-good
fixture is accepted, every known-bad fixture is rejected by exact runtime/Z3
feedback, unsupported prompt patterns reject with a named reason, the static
transition representation is present without speed claims, and the artifact's
`honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-2994: Prompt Constraints Become Exact Validator Feedback

Given the prompt-validator protocol document is present and exact local
runtime/Z3 verification is available,
When the deterministic Exp 2994 harness runs,
Then it compiles the fixed JSON, Python AST, and solver fixtures into validator
trees, evaluates known-good and known-bad candidates through exact-check nodes,
emits feedback objects with named failing node IDs and rejection reasons,
records a sparse/static transition-table design proposal without claiming
speed, and writes the required terminal JSON artifact with
`exact_verifier_authority_preserved=true`.

## Implementation Status (REQ-VERIFY-2994)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-2994 | Implemented (`python/carnot/eval/prompt_validator_dialogue_schema_v1.py`) | Implemented (`tests/python/test_experiment_2994_prompt_validator_dialogue_schema.py`) |

### REQ-VERIFY-3005: Solver-To-Validator Tree Expansion Corpus

The repository shall provide a deterministic Exp 3005 solver-to-validator tree
expansion harness that consumes the Exp 2992 solver-feedback formalization
line and the Exp 2994 prompt-validator dialogue protocol, builds a corpus of at
least 20 small solver/formalization items, and writes
`results/experiment_3005_solver_to_validator_tree_expansion_v1.json`.

Each accepted corpus item shall have an inspectable validator tree with
explicit exact-check nodes. Runtime nodes shall check JSON shape, required
candidate fields, assertion list shape, query shape, and expected solver-status
metadata. Z3 nodes shall replay the formalization and compare actual solver
status against the candidate's expected status and the item reference status.
Items without deterministic exact checks, with nondeterministic tests, or with
LLM-only judgment labels shall be rejected and reported rather than silently
included.

For every accepted item, the harness shall execute full-candidate validation
and LAVE-inspired partial-candidate viability checks. Invalid prefixes or
partial constraints shall fail early with named rejection reasons, while valid
prefixes that are still extendable to the reference formalization shall remain
accepted as partial candidates. The run shall save replayable runtime and Z3
transcripts plus hash evidence, and it shall write a JSONL validator manifest
that downstream diagnostics can inspect without relying on LLM-as-judge
semantics.

The terminal artifact MUST include `validator_tree_expanded`,
`validator_manifest_path`, `n_solver_items`, `n_validator_trees`,
`all_trees_exact_checked`, `partial_viability_checked`,
`z3_transcript_paths`, `runtime_transcript_paths`, `rejected_constraints`,
`llm_judge_used`, and `honest_verdict`.

`validator_tree_expanded` shall be true only when at least 20 solver items are
accepted, every accepted item has exactly one validator tree, all trees execute
runtime and Z3 exact checks, every accepted item has passing full validation,
valid/invalid partial viability evidence is recorded for every item, transcript
paths are present and hashable, rejected constraints are surfaced, and
`llm_judge_used=false`.

### SCENARIO-VERIFY-3005: Expanded Validator Trees Gate On Exact Checks

Given Exp 2992 has reproduced solver-feedback formalization with Z3 provenance
and Exp 2994 has produced a deterministic prompt-validator protocol,
When the Exp 3005 expansion harness runs,
Then it writes an inspectable validator manifest with at least 20 solver items
and validator trees, executes every tree through runtime and Z3 authorities,
records runtime and Z3 transcript hashes, accepts extendable valid partials,
rejects invalid partials early with named reasons, reports rejected
nondeterministic, missing-exact-check, or LLM-only constraints, and writes the
terminal artifact with `validator_tree_expanded=true` only when every exact
gate clears.

## Implementation Status (REQ-VERIFY-3005)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3005 | Implemented (`python/carnot/eval/solver_to_validator_tree_expansion_v1.py`) | Implemented (`tests/python/test_experiment_3005_solver_to_validator_tree_expansion.py`) |

### REQ-VERIFY-3017: NSVIF Instruction Validator-Tree Expansion

The repository shall provide a deterministic Exp 3017 instruction-following
validator-tree expansion harness that extends the Exp 3005 exact-check corpus
from solver/formalization items into at least 20 small instruction-following
items and writes
`results/experiment_3017_nsvif_instruction_validator_tree_expansion_v1.json`.

The accepted instruction items shall cover required output fields, forbidden
tokens, ordering constraints, numeric bounds, simple string transformations,
Z3-checkable integer relations, Python AST constraints, and executable
runtime invariants. Each accepted item shall have an inspectable validator
tree. Authoritative validator nodes MUST use executable local authority such
as runtime JSON parsing, string checks, Python AST parsing, Python runtime
execution, or Z3 solver replay. Any semantic-only node MUST be marked
non-authoritative and MUST NOT decide acceptance.

The harness shall execute every accepted validator tree against a known-good
candidate and a known-bad candidate, save replayable runtime and Z3 transcripts
with SHA-256 hashes, and write a JSONL validator manifest. Ambiguous
instructions, nondeterministic validators, and LLM-only labels shall be
rejected and reported rather than included as verifier authority. LLM
self-judgment MUST NOT be used as verifier authority.

The terminal artifact MUST include `instruction_validator_tree_ready`,
`validator_manifest_path`, `n_instruction_items`, `n_validator_trees`,
`exact_check_coverage`, `all_authoritative_nodes_exact_checked`,
`z3_transcript_paths`, `runtime_transcript_paths`, `rejected_items`,
`llm_judge_used`, and `honest_verdict`.

`instruction_validator_tree_ready` shall be true only when at least 20
instruction items are accepted, every accepted item has exactly one validator
tree, all authoritative nodes execute exact local checks, every known-good
candidate is accepted, every known-bad candidate is rejected with named
reasons, transcript paths are present and hashable, rejected invalid items are
surfaced, `llm_judge_used=false`, and `honest_verdict` starts with a terminal
success prefix.

### SCENARIO-VERIFY-3017: Instruction Constraints Become Executable Validator Evidence

Given the Exp 3005 validator-tree expansion line is present and the Exp 2994
prompt-validator protocol preserves exact verifier authority,
When the Exp 3017 harness runs,
Then it writes an inspectable manifest for at least 20 instruction-following
items spanning required fields, forbidden tokens, ordering, numeric bounds,
transformations, Z3 relations, AST checks, and runtime invariants; executes
each tree through exact authoritative nodes; records runtime and Z3 transcript
hashes; marks semantic-only labels as non-authoritative; rejects ambiguous,
nondeterministic, or LLM-only candidates; and writes the terminal artifact with
`instruction_validator_tree_ready=true` only when every exact authority gate
clears.

## Implementation Status (REQ-VERIFY-3017)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3017 | Implemented (`python/carnot/eval/nsvif_instruction_validator_tree_expansion_v1.py`) | Implemented (`tests/python/test_experiment_3017_nsvif_instruction_validator_tree_expansion.py`) |

### REQ-VERIFY-3018: BEAVER-Style Validator Frontier Certificate

The repository shall provide a deterministic Exp 3018 frontier-certificate
harness that consumes the Exp 3017 instruction validator-tree manifest, keeps
live LLM evidence and enumerator fallback provenance separate, and writes
`results/experiment_3018_beaver_style_validator_frontier_certificate_v1.json`.

The harness shall identify which authoritative validator-tree constraints are
prefix-closed, which constraints are only bounded-frontier explorable through
finite or cached candidate sets, and which rows are non-prefix-closed or
unresolved. It SHALL execute a small deterministic frontier explorer over the
cached Exp 3017 known-good and known-bad candidates, record deterministic
validator outcomes, and write a JSONL certificate manifest whose rows are
inspectable without re-running live inference.

The certificate SHALL NOT claim exact BEAVER probability bounds unless a full
token-trie/model-probability frontier is implemented. Every certificate row
shall include explicit uncertainty/probability-bound placeholders stating
whether exact probability was computed, and unresolved or non-prefix-closed
rows shall remain visible rather than being folded into pass/fail counts.

The terminal artifact MUST include `frontier_certificate_ready`,
`certificate_manifest_path`, `n_frontier_items`, `n_prefix_closed_items`,
`certified_safe_count`, `certified_violating_count`, `unresolved_count`,
`enumerator_fallback_separated`, `live_llm_evidence_used`,
`transcript_paths`, and `honest_verdict`.

`frontier_certificate_ready` shall be true only when the Exp 3017 manifest is
available and ready, at least one safe and one violating deterministic frontier
row are certified, prefix-closed assumptions are counted, unresolved and
non-prefix-closed rows are explicitly represented, certificate transcripts are
written, `live_llm_evidence_used=false`, and enumerator fallback evidence is
not mixed into live LLM evidence.

### SCENARIO-VERIFY-3018: Validator Trees Become Inspectable Frontier Certificates

Given Exp 3017 has written an exact-checked instruction validator-tree manifest,
When the Exp 3018 certificate harness runs without fresh live inference,
Then it classifies prefix-closed, bounded-frontier, non-prefix-closed, and
unresolved rows; validates cached known-good and known-bad candidates through
the deterministic validator trees; writes replayable certificate manifests and
transcripts; records probability-bound placeholders where exact probability is
not computed; keeps live LLM transcript provenance separate from enumerator
fallback provenance; and writes the terminal artifact with
`frontier_certificate_ready=true` only when the cached/exact certificate gates
clear.

## Implementation Status (REQ-VERIFY-3018)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3018 | Implemented (`python/carnot/eval/beaver_style_validator_frontier_certificate_v1.py`) | Implemented (`tests/python/test_experiment_3018_beaver_style_validator_frontier_certificate.py`) |

### REQ-VERIFY-3019: FR-11 Feasibility-Channel De-Tautology Diagnostic

The repository shall provide a deterministic Exp 3019 feasibility-channel
diagnostic that consumes exact Exp 3017 validator rows, Exp 3018 certificate
outcomes, and Exp 3007 FR-11 trace-memory evidence, then writes
`results/experiment_3019_fr11_feasibility_channel_de_tautology_diagnostic_v1.json`.

The diagnostic shall derive transparent feasibility features only from exact
evidence: exact validator failure counts, exact-authority coverage,
frontier/certificate provenance, transcript presence, unresolved source
rejections, and Exp 3007 negative-control deltas. It SHALL NOT use
`certificate_status`, `candidate_role`, held-out success labels, or self-reported
memory utility as feasibility-score features.

The diagnostic shall write an inspectable JSONL table that includes feasible,
violating, unresolved, non-prefix, and negative-control rows. It SHALL measure
feasible-vs-infeasible separation, negative-control rejection, and held-out
verifier-success correlation on a deterministic held-out item partition. The
held-out label SHALL come from source exact verifier outcomes rather than the
same scalar used as the feasibility score.

The diagnostic shall make tautology risk explicit whenever its exact evidence is
still too close to the verifier labels for promotion-grade FR-11 gating. It
SHALL NOT claim a native Differentiable Symbolic Planning implementation or any
broad self-improvement capability.

The terminal artifact MUST include `feasibility_channel_diagnostic_ready`,
`diagnostic_table_path`, `n_rows`, `feasible_infeasible_auc`,
`negative_control_rejection_rate`, `heldout_metric_correlation`,
`tautology_risk_flag`, `reused_label_as_feature`, `native_dsp_claim_made`, and
`honest_verdict`.

`feasibility_channel_diagnostic_ready` shall be true only when the exact source
rows are present, the diagnostic table is written, feasible and infeasible
certificate rows are both represented, negative controls are measured,
held-out correlation is computed without label-feature reuse,
`reused_label_as_feature=false`, `native_dsp_claim_made=false`, and
`honest_verdict` starts with a terminal completion prefix. A true
`tautology_risk_flag` is allowed only as an explicit diagnostic warning, not as
a promotion claim.

### SCENARIO-VERIFY-3019: Exact Certificate Rows Become A Feasibility Diagnostic

Given Exp 3017 exact validator rows, Exp 3018 certificate rows, and Exp 3007
trace-memory control evidence are available,
When the Exp 3019 diagnostic runs,
Then it writes an inspectable table covering feasible, violating, unresolved,
non-prefix, and negative-control rows; computes feasible-vs-infeasible AUC,
negative-control rejection rate, and held-out exact-verifier correlation; marks
tautology risk explicitly; rejects label-as-feature reuse; makes no native DSP
claim; and writes the terminal artifact with
`feasibility_channel_diagnostic_ready=true`.

### SCENARIO-VERIFY-3019-BLOCKED: Missing Exact Evidence Fails Closed

Given any required Exp 3017, Exp 3018, or Exp 3007 source evidence is missing or
not ready,
When the Exp 3019 diagnostic runs,
Then it writes the required artifact fields with
`feasibility_channel_diagnostic_ready=false`, zero sample and metric values, no
native DSP claim, no label-feature reuse, and an `honest_verdict` beginning with
`blocked_`.

## Implementation Status (REQ-VERIFY-3019)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3019 | Proposed (`python/carnot/eval/fr11_feasibility_channel_de_tautology_diagnostic_v1.py`) | Proposed (`tests/python/test_experiment_3019_fr11_feasibility_channel_de_tautology_diagnostic.py`) |

### REQ-VERIFY-3044: SMT/SAT Validator-Tree Exactness Upgrade

The repository shall provide a deterministic Exp 3044 SMT/SAT validator-tree
exactness upgrade that runs tiny local solver-backed fixtures without live LLM
inference and writes
`results/experiment_3044_smt_sat_validator_tree_exactness_upgrade_v1.json`.

The upgrade shall expose an explicit exact-validator implementation path,
write inspectable evidence rows, and distinguish verified, irrelevant,
unresolved, fallback-only, and correction-set rows. Verified rows shall come
only from exact solver satisfiability checks over executable validator-tree
constraints. Unsatisfiable exact rows shall report minimal correction sets
that localize the candidate field or assertion to change. Irrelevant rows shall
remain clipped to non-authoritative or out-of-scope validator regions.
Unresolved rows shall remain visible when a row is not exactly decidable by the
implemented SMT/SAT path. Fallback-only rows shall never be promoted as exact
authority.

The terminal artifact MUST include `validator_tree_exactness_ready`,
`exact_validator_path`, `tests_or_checks_run`, `verified_count`,
`unresolved_count`, `fallback_only_count`, `correction_sets`, `spec_updates`,
`model_specs`, `inference_substrate`, and `honest_verdict`. It shall also
record the evidence row path and enough row counts for downstream
self-learning controllers to gate on exact feedback availability.

`validator_tree_exactness_ready` shall be true only when the exact-validator
implementation path exists, the evidence row file exists, at least one
verified row is counted, at least one correction set is emitted, unresolved
and fallback-only rows remain visible, no live LLM inference was used, and
`honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-3044: Exact Rows Gate Downstream Self-Learning

Given deterministic validator-tree fixtures with satisfiable, unsatisfiable,
irrelevant, unresolved, and fallback-only cases,
When the Exp 3044 exactness upgrade runs,
Then it writes an evidence row file and terminal artifact, classifies the
satisfiable exact row as verified, converts the unsatisfiable exact row into a
minimal correction-set row, preserves irrelevant and unresolved rows as
non-promotable, keeps fallback-only rows separate from exact authority, and
sets `validator_tree_exactness_ready=true` only when the exact path and
evidence file are present.

## Implementation Status (REQ-VERIFY-3044)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3044 | Implemented (`python/carnot/eval/smt_sat_validator_tree_exactness_upgrade_v1.py`) | Implemented (`tests/python/test_experiment_3044_smt_sat_validator_tree_exactness_upgrade.py`) |

### REQ-VERIFY-3057: Local SOTA Solution-Verifier Gain Panel

The repository shall provide a tiny Exp 3057 local SOTA solution-verifier
calibration panel over deterministic SAT/SMT-style fixtures with exact solver
ground truth. The panel shall write
`results/experiment_3057_local_sota_solution_verifier_gain_panel_v1.json`.

The panel shall build 6-12 deterministic fixtures whose exact labels are
derived from an executable solver authority, use at least one mandated local
SOTA GGUF model from `python/carnot/inference/sota_models.py` for live solving
or verification when such a model can be loaded, and compare one-shot solver
accuracy against verifier-selected accuracy over a small candidate pool. If two
mandated GGUF model families can be loaded, the solver and verifier roles shall
be separated across families; otherwise the artifact shall disclose
`cross_family_used=false`. Legacy tiny models shall not satisfy calibration
readiness and may appear only as `legacy_smoke_only_used=true` smoke evidence.

The terminal artifact MUST include `solution_verifier_calibration_ready`,
`verifier_gain_delta`, `false_positive_rate`, `false_negative_rate`,
`exact_ground_truth_count`, `models_used`, `model_specs`,
`legacy_smoke_only_used`, `cross_family_used`, `prompt_hashes`,
`inference_substrate`, and `honest_verdict`. It shall also record model cache
resolution, CUDA/GPU memory provenance, seed, prompt/decode settings,
wall-clock duration, exact-solver agreement, and enough candidate-level rows to
audit false acceptance and false rejection errors without embedding full
prompts.

`solution_verifier_calibration_ready` shall be true only when exact ground
truth is present for at least six fixtures, at least one mandated local GGUF
successfully produced live solver or verifier evidence, model specifications
name the exact model IDs and runtime/cache paths, verifier-gain and false-error
metrics are non-vacuous, `legacy_smoke_only_used=false`, and `honest_verdict`
starts with a terminal success prefix. If no mandated GGUF can be loaded, the
artifact shall fail closed with `solution_verifier_calibration_ready=false`,
empty model evidence, zero exact-ground-truth promotion, and an
`honest_verdict` beginning with `blocked_sota_gguf_unavailable`.

### SCENARIO-VERIFY-3057: Exact Ground Truth Calibrates Verifier Selection

Given deterministic SAT/SMT-style fixtures with exact solver labels and a
mandated local GGUF runtime that can load,
When the Exp 3057 panel runs,
Then it obtains live local model output for solver and/or verifier roles,
hashes every prompt, evaluates one-shot solver candidates and
verifier-selected candidate-pool answers against exact solver authority,
reports verifier gain, false positive rate, false negative rate,
exact-solver agreement, cross-family usage, and legacy-smoke status, and sets
`solution_verifier_calibration_ready=true` only when the live evidence and
non-vacuous metrics are present.

### SCENARIO-VERIFY-3057-BLOCKED: Missing Mandated GGUF Fails Closed

Given the mandated local GGUF cache paths are absent or every mandated GGUF
load attempt fails,
When the Exp 3057 panel runs,
Then it writes the required artifact fields, records CUDA/GPU/cache
preconditions and decode settings, leaves model evidence empty, does not use
legacy tiny models as headline evidence, and emits an `honest_verdict`
beginning with `blocked_sota_gguf_unavailable`.

## Implementation Status (REQ-VERIFY-3057)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3057 | Implemented (`python/carnot/eval/local_sota_solution_verifier_gain_panel_v1.py`) | Implemented (`tests/python/test_experiment_3057_local_sota_solution_verifier_gain_panel.py`) |

### REQ-VERIFY-3070: First-Token Abstention SOTA Panel

The repository shall provide a tiny Exp 3070 first-token confidence and
abstention panel over deterministic SAT/SMT-style fixtures with exact solver
ground truth. The panel shall write
`results/experiment_3070_first_token_abstention_sota_panel_v1.json`.

The panel shall reuse the Exp 3057 exact-labeled fixtures or rebuild 6-12
deterministic fixtures with known valid/invalid outcomes, use at least one
mandated local SOTA GGUF model from
`python/carnot/inference/sota_models.py` to score candidate solutions, capture
first-token entropy or the closest available first-token confidence proxy, and
record any limitation when true top-k logprobs are unavailable. The panel
shall define an abstention threshold using only a calibration split and then
evaluate acceptance precision, rejection recall, abstention coverage, false
positives, false negatives, first-token AUC, and verifier selection gain on a
held-out tiny split.

The terminal artifact MUST include `first_token_panel_ready`,
`confidence_signal`, `first_token_auc`, `abstention_precision`,
`rejection_recall`, `abstention_coverage`,
`verifier_gain_delta_with_abstention`, `false_positive_rate`,
`false_negative_rate`, `exact_ground_truth_count`, `models_used`,
`model_specs`, `legacy_smoke_only_used`, `prompt_hashes`,
`inference_substrate`, and `honest_verdict`. It shall also record CUDA
availability, GPU memory, GGUF cache paths, model IDs, quantization, seed,
prompt hashes, decode settings, logprob support, calibration threshold, split
sizes, and wall-clock duration without embedding full prompts.

`first_token_panel_ready` shall be true only when exact labels are present for
at least six fixtures, at least one mandated local GGUF produced live scoring
evidence, `model_specs` name exact model IDs and cache/runtime paths,
confidence fields are non-vacuous, abstention metrics are computed on the
held-out split, `legacy_smoke_only_used=false`, and `honest_verdict` starts
with a terminal success prefix. If no mandated GGUF can be loaded or no
first-token confidence signal can be captured, the artifact shall fail closed
with `first_token_panel_ready=false`, empty or non-promoted model evidence,
zero exact-ground-truth promotion, and an `honest_verdict` beginning with
`blocked_sota_confidence_unavailable`.

### SCENARIO-VERIFY-3070: First-Token Confidence Calibrates Abstention

Given deterministic SAT/SMT-style fixtures with exact solver labels and a
mandated local GGUF runtime that can load,
When the Exp 3070 panel scores candidate solutions,
Then it hashes every prompt, records first-token entropy or an explicit proxy
for each scored row, derives the abstention threshold from calibration rows
only, evaluates held-out accepted/rejected/abstained decisions against exact
labels, reports first-token AUC, acceptance precision, rejection recall,
abstention coverage, false-positive rate, false-negative rate, and verifier
gain delta with abstention, and sets `first_token_panel_ready=true` only when
the live evidence and non-vacuous metrics are present.

### SCENARIO-VERIFY-3070-BLOCKED: Missing Confidence Fails Closed

Given the mandated local GGUF cache paths are absent, every mandated GGUF load
attempt fails, or the runtime returns no first-token logprob or confidence
proxy,
When the Exp 3070 panel runs,
Then it writes the required artifact fields, records CUDA/GPU/cache
preconditions and decode settings, leaves headline model evidence unpromoted,
does not use legacy tiny models as headline evidence, and emits an
`honest_verdict` beginning with `blocked_sota_confidence_unavailable`.

## Implementation Status (REQ-VERIFY-3070)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3070 | Implemented (`python/carnot/eval/first_token_abstention_sota_panel_v1.py`) | Implemented (`tests/python/test_experiment_3070_first_token_abstention_sota_panel.py`) |

### REQ-VERIFY-3058: AquaForte-Style LLM-Guided SMT Instantiation Pilot

The repository shall provide a tiny Exp 3058 AquaForte-style pilot in which a
mandated local SOTA GGUF proposes quantified-SMT or uninterpreted-function
instantiations and an exact solver decides whether each proposal proves the
target claim. The pilot shall write
`results/experiment_3058_aquaforte_style_llm_guided_smt_pilot_v1.json`.

The pilot shall first confirm that Exp 3057 reports
`solution_verifier_calibration_ready=true`, that the exact solver path can
import Z3 or CVC5, and that at least one mandated GGUF from
`python/carnot/inference/sota_models.py` resolves and loads through the local
runtime. If any precondition fails, the artifact shall fail closed with
`llm_guided_smt_pilot_ready=false`, empty or non-promoted model evidence, and
an `honest_verdict` beginning with the matching blocked precondition.

When preconditions pass, the pilot shall run 4-8 tiny fixtures, preferring
quantified or uninterpreted-function-inspired fixtures when Z3 UF support is
available. For each fixture the local GGUF shall propose instantiations or
variable/function mappings, every prompt shall be represented by a SHA-256
hash and deterministic decode settings, and every proposal shall be validated
by the exact solver before being counted as guided success. Invalid,
unparseable, or non-proving LLM proposals shall remain visible. The solver-only
fallback shall remain available, be counted separately, and stay authoritative.
Guidance may reduce unresolved or fallback-only cases, but the artifact shall
report no reduction when the solver-only baseline already succeeds.

The terminal artifact MUST include `llm_guided_smt_pilot_ready`,
`formal_fallback_preserved`, `guided_success_count`,
`solver_only_success_count`, `unresolved_count`,
`invalid_llm_proposal_count`, `models_used`, `model_specs`,
`legacy_smoke_only_used`, `exact_solver_path`, `prompt_hashes`,
`inference_substrate`, and `honest_verdict`. It shall also record fixture rows,
model cache/runtime paths, decode settings, solver-only comparison metrics,
and enough row-level hashes to audit the run without embedding full prompts.

`llm_guided_smt_pilot_ready` shall be true only when Exp 3057 is ready, an
exact solver validates every fixture row, at least one mandated GGUF generated
live proposal text, at least one proposal is accepted by exact validation,
solver-only fallback remains preserved and authoritative,
`legacy_smoke_only_used=false`, and `honest_verdict` starts with a terminal
success prefix.

### SCENARIO-VERIFY-3058: Local GGUF Proposals Are Exact-Solver Checked

Given Exp 3057 is ready, Z3 is importable, and a mandated local SOTA GGUF can
load,
When the Exp 3058 pilot runs over tiny quantified or
uninterpreted-function-inspired fixtures,
Then it hashes each prompt, records decode and runtime provenance, parses the
local GGUF proposal as instantiations or mappings, validates each proposal with
the exact solver, counts only solver-confirmed proposals as guided successes,
counts invalid LLM proposals separately, runs the solver-only fallback over the
same fixtures, and writes the required terminal artifact fields.

### SCENARIO-VERIFY-3058-BLOCKED: Missing Preconditions Fail Closed

Given Exp 3057 is not ready, the exact solver cannot import, or no mandated
local GGUF resolves or loads,
When the Exp 3058 pilot runs,
Then it writes a blocked terminal artifact with
`llm_guided_smt_pilot_ready=false`, preserves solver-only authority when the
solver exists, does not promote legacy tiny models as headline evidence, and
uses an `honest_verdict` beginning with the blocked precondition reason.

## Implementation Status (REQ-VERIFY-3058)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3058 | Implemented (`python/carnot/eval/aquaforte_style_llm_guided_smt_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3058_aquaforte_style_llm_guided_smt_pilot.py`) |

### REQ-VERIFY-3071: VERGE-Style MCS SMT Correction Feedback Pilot

The repository shall provide an Exp 3071 VERGE-style correction-feedback pilot
in which exact solvers produce minimal correction subsets or refinement
signals for tiny SMT-encodable candidate records, and a mandated local SOTA
GGUF proposes corrected candidates using only that correction feedback. The
pilot shall write
`results/experiment_3071_verge_mcs_smt_correction_pilot_v1.json`.

The pilot shall first confirm that Z3 or an equivalent exact solver can run
and that at least one mandated GGUF from
`python/carnot/inference/sota_models.py` resolves and loads through the local
runtime. If either precondition fails, the artifact shall fail closed with
`mcs_feedback_ready=false` and an `honest_verdict` beginning with
`blocked_solver_or_sota_unavailable`.

When preconditions pass, the pilot shall run 4-8 tiny fixtures spanning
already-valid, overconstrained, underconstrained, invalid, and repairable
candidate states. For every non-valid fixture, the exact solver shall emit a
non-empty correction subset or refinement signal before any model repair
prompt is constructed. Every model prompt shall be represented by a SHA-256
hash and deterministic decode settings. Every corrected candidate proposed by
the model shall be parsed and validated with the exact solver before being
counted as guided success. A solver-only exact repair baseline shall run on
the same fixtures and remain separate from the model path.

The terminal artifact MUST include `mcs_feedback_ready`,
`formal_fallback_preserved`, `mcs_count`, `guided_success_count`,
`solver_only_success_count`, `invalid_llm_proposal_count`,
`correction_subset_useful_count`, `exact_solver_path`, `models_used`,
`model_specs`, `legacy_smoke_only_used`, `prompt_hashes`,
`inference_substrate`, and `honest_verdict`. `mcs_feedback_ready` shall be true
only when correction feedback is non-vacuous, exact validation is non-vacuous,
at least one mandated local GGUF generated live proposal text, solver-only
fallback remains preserved and authoritative, `legacy_smoke_only_used=false`,
and `honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-3071: MCS Feedback Guides Candidate Repair

Given Z3 is importable and a mandated local SOTA GGUF can load,
When the Exp 3071 pilot runs over tiny SMT correction fixtures,
Then it records exact correction feedback for each invalid or incomplete
candidate, hashes every repair prompt, asks the local GGUF to propose a
corrected candidate using only that feedback, validates each proposal with the
exact solver, compares against a solver-only repair baseline, counts invalid
LLM proposals separately, and writes the required terminal artifact fields.

### SCENARIO-VERIFY-3071-BLOCKED: Solver Or SOTA Runtime Missing

Given the exact solver cannot run, no mandated local GGUF resolves, or every
mandated GGUF load attempt fails,
When the Exp 3071 pilot runs,
Then it writes a blocked terminal artifact with `mcs_feedback_ready=false`,
empty or non-promoted model evidence, no legacy tiny-model headline evidence,
and an `honest_verdict` beginning with
`blocked_solver_or_sota_unavailable`.

## Implementation Status (REQ-VERIFY-3071)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3071 | Implemented (`python/carnot/eval/verge_mcs_smt_correction_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3071_verge_mcs_smt_correction_pilot.py`) |

### REQ-VERIFY-3084: ReSyn Exact Fixture Bank Generator

The repository shall provide an Exp 3084 deterministic exact fixture-bank
generator that writes
`results/experiment_3084_resyn_exact_fixture_bank_generator_v1.json` and the
downstream JSONL fixture manifest
`results/resyn_exact_fixture_bank_3084/fixture_manifest.jsonl` without invoking
live LLM inference.

The generator shall first check that exact-label tooling is viable through Z3
and local Python/runtime validation. If exact labeling cannot be supported, it
shall write a terminal blocked artifact with `resyn_fixture_bank_ready=false`
and an `honest_verdict` beginning with `blocked_exact_label_tooling_missing`.

When preconditions pass, the generator shall create at least 64 exact fixtures
across at least three perturbation families that include SMT/SAT constraints,
arithmetic or code assertions, and repairable invalid candidates. Every
fixture row shall include an exact label, exact label source, perturbation
family, family name, leakage-safe prompt payload or prompt hash, and no
LLM-derived label. The terminal artifact MUST include
`resyn_fixture_bank_ready`, `exact_fixture_count`, `family_count`,
`fixture_manifest_path`, `exact_label_sources`, `perturbation_families`,
`tests_added_or_reused`, `preconditions_checked`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`.

`resyn_fixture_bank_ready` shall be true only when the manifest exists, at
least 64 rows validate against exact local authorities, at least three
families are represented, every row has leakage-safe prompt metadata, and
`inference_substrate` declares no live LLM inference.

### SCENARIO-VERIFY-3084: Exact Fixtures Feed Downstream Verifier Tasks

Given Z3 and local Python/runtime validation are available,
When the Exp 3084 generator runs,
Then it writes the terminal JSON artifact and fixture manifest, reports family
counts and exact label sources, keeps solving, verification, abstention, and
repair perturbations separable, records Exp 3070 and Exp 3083 as source
protocol artifacts, and exposes a concrete manifest path for downstream .288
verifier, abstention, and repair tasks.

## Implementation Status (REQ-VERIFY-3084)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3084 | Implemented (`python/carnot/eval/resyn_exact_fixture_bank_generator_v1.py`) | Implemented (`tests/python/test_experiment_3084_resyn_exact_fixture_bank_generator.py`) |

### REQ-VERIFY-3085: I-CALM Task-Abstention SOTA Panel V2

The repository shall provide an Exp 3085 local SOTA GGUF abstention panel that
reruns first-token abstention evidence on the Exp 3084 exact fixture bank with
baseline prompts and I-CALM/task-abstention prompts. The panel shall write
`results/experiment_3085_icalm_task_abstention_sota_panel_v2.json` and a
row-level JSONL transcript without modifying the Exp 3084 fixture manifest or
`scripts/research_conductor.py`.

The panel shall first check CUDA/GPU availability, local GGUF cache resolution,
actual loadability for at least one mandated headline model
(`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`), and readability of
`results/resyn_exact_fixture_bank_3084/fixture_manifest.jsonl`. If any
precondition fails, it shall write a terminal blocked artifact with
`abstention_panel_v2_ready=false`, `first_token_panel_ready=false`, no
headline model promotion, zero promotion metrics, explicit precondition
diagnostics, and an `honest_verdict` beginning with
`blocked_sota_or_fixture_precondition_failed`.

When preconditions pass, the panel shall sample a balanced, deterministic
subset across fixture families without including exact labels, expected
answers, or authority payloads in model prompts. It shall run each selected
fixture with a baseline answer prompt and an I-CALM/task-abstention prompt,
record prompt hashes, decode settings, exact model IDs, runtime/cache paths,
hardware substrate metadata, first-token confidence when available, verbal
confidence, parsed answer correctness, abstention, rejection, and
overacceptance. Acceptance, rejection, and abstention behavior shall be
reported separately, and Exp 3070 abstention metrics shall be loaded for
comparison.

The terminal artifact MUST include `abstention_panel_v2_ready`,
`first_token_panel_ready`, `abstention_precision`, `rejection_recall`,
`abstention_coverage`, `overacceptance_rate`, `exact_ground_truth_count`,
`fixture_manifest_path`, `models_used`, `model_specs`,
`legacy_smoke_only_used`, `preconditions_checked`, `prompt_hashes`,
`inference_substrate`, and `honest_verdict`. It shall also report baseline
and task-abstention row counts, accepted/rejected/abstained counts, verbal
confidence coverage, first-token confidence coverage, prompt hashes, row
transcript checksum, and whether `abstention_precision >= 0.7`.

`abstention_panel_v2_ready` shall be true only when the exact fixture manifest
is readable, at least one mandated local GGUF produced live output for both
prompt policies, exact ground truth is available for every selected row,
accept/reject/abstain denominators are reported separately, legacy tiny models
are not promoted, and the `honest_verdict` starts with a terminal success
prefix. `first_token_panel_ready` shall additionally require at least one
captured first-token confidence signal.

### SCENARIO-VERIFY-3085: I-CALM Prompt Separates Accept Reject And Abstain

Given the Exp 3084 exact fixture manifest is readable, CUDA/GPU is available,
and a mandated local GGUF can load,
When the Exp 3085 panel runs,
Then it samples fixtures across every manifest family, hashes every baseline
and I-CALM/task-abstention prompt, records model/cache/runtime provenance,
parses first-token and verbal confidence where available, evaluates answer
correctness against exact labels only after generation, reports accepted,
rejected, and abstained counts separately, computes abstention precision,
rejection recall, abstention coverage, overacceptance rate, compares against
Exp 3070, and writes the required artifact fields.

### SCENARIO-VERIFY-3085-BLOCKED: Missing SOTA Or Fixtures Fail Closed

Given CUDA/GPU is unavailable, no mandated headline GGUF resolves and loads,
or the Exp 3084 fixture manifest is missing or unreadable,
When the Exp 3085 panel runs,
Then it writes the required artifact fields with readiness flags false, empty
or unpromoted headline evidence, explicit precondition diagnostics, zero
promotion metrics, no legacy tiny-model headline substitution, and an
`honest_verdict` beginning with
`blocked_sota_or_fixture_precondition_failed`.

## Implementation Status (REQ-VERIFY-3085)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3085 | Implemented (`python/carnot/eval/icalm_task_abstention_sota_panel_v2.py`) | Implemented (`tests/python/test_experiment_3085_icalm_task_abstention_sota_panel.py`) |

### REQ-VERIFY-3086: Dafny/Z3 Formal-Feedback Pilot

The repository shall provide an Exp 3086 formal-feedback pilot that runs tiny
verified-code repair fixtures through Dafny when available and a Z3 plus local
execution fallback when Dafny is unavailable. The pilot shall write
`results/experiment_3086_dafny_z3_formal_feedback_pilot_v1.json` without
modifying `scripts/research_conductor.py`.

The pilot shall first check `command -v dafny`, `command -v z3`, CUDA/GPU
availability, and local cache/loadability for at least one mandated headline
GGUF (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`). If neither Dafny nor Z3 is available, it
shall write a terminal blocked artifact with `formal_feedback_ready=false`,
`dafny_available=false`, `z3_available=false`, and an `honest_verdict`
beginning with `blocked_formal_toolchain_missing`. If Dafny is absent but Z3
is present, it shall run the Z3-only path and set `dafny_available=false`.

When formal tooling and a mandated local GGUF are available, the pilot shall
evaluate 4-8 small fixtures spanning invalid candidate implementations,
repairable malformed candidates, vacuous specifications, and weak
postconditions. It shall generate formal diagnostics containing only verifier
loop evidence such as counterexamples, missing fields, unsatisfiable
preconditions, weak-postcondition witnesses, and failing constraint names. It
shall hash every model prompt, ask the mandated GGUF for corrected candidates
using only those diagnostics, validate every corrected candidate exactly, and
compare guided repairs against a solver-only validation baseline that does not
use model correction.

The terminal artifact MUST include `formal_feedback_ready`,
`formal_feedback_delta`, `dafny_available`, `z3_available`,
`vacuity_guard_passed`, `guided_success_count`, `solver_only_success_count`,
`exact_ground_truth_count`, `models_used`, `model_specs`,
`legacy_smoke_only_used`, `preconditions_checked`, `prompt_hashes`,
`inference_substrate`, and `honest_verdict`. It shall also record fixture-level
diagnostics, guided validation results, and baseline validation results.

`formal_feedback_ready` shall be true only when formal diagnostics are
non-vacuous, the vacuity/weak-postcondition guards fire on the corresponding
fixtures, at least one mandated GGUF produced live correction text, every
guided success was exactly validated, `guided_success_count >
solver_only_success_count`, `legacy_smoke_only_used=false`, and
`honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-3086: Z3 Fallback Calibrates Formal Repair Feedback

Given Dafny is absent, Z3 is available, CUDA/GPU is available, and a mandated
local GGUF resolves and loads,
When the Exp 3086 pilot runs over tiny invalid, repairable, vacuous, and
weak-postcondition fixtures,
Then it records `dafny_available=false`, `z3_available=true`, produces
counterexample and vacuity diagnostics without leaking exact repairs, hashes
every repair prompt, validates model corrections with exact Z3 or execution
authority, reports guided and solver-only success counts separately, and writes
the required terminal artifact fields.

### SCENARIO-VERIFY-3086-BLOCKED: Missing Formal Checker Fails Closed

Given neither Dafny nor Z3 is available,
When the Exp 3086 pilot runs,
Then it writes a blocked terminal artifact with formal feedback readiness
false, explicit precondition diagnostics, empty model-promotion evidence, no
legacy tiny-model substitution, and an `honest_verdict` beginning with
`blocked_formal_toolchain_missing`.

## Implementation Status (REQ-VERIFY-3086)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3086 | Implemented (`python/carnot/eval/dafny_z3_formal_feedback_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3086_dafny_z3_formal_feedback_pilot.py`) |

### REQ-VERIFY-3097: Exact-Fixture Stratified Evaluation Protocol Audit

The repository shall provide an Exp 3097 exact-fixture evaluation protocol
audit for milestone `.289` that writes
`results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json` without
modifying `scripts/research_conductor.py` or invoking live model inference.

The audit shall load the Exp 3084 terminal artifact, the checked-in Exp 3084
fixture manifest, Exp 3085 abstention-panel evidence, Exp 3086 formal-feedback
evidence, and the Exp 3094 `.288` capstone. It shall validate exact fixture
rows against local authority metadata, detect malformed rows and duplicate
fixture identities or prompt hashes, and report why Exp 3085 evaluated only a
tiny exact panel when the fixture bank exposed more usable rows.

When fixture authority is sufficient, the audit shall write a derived
stratified JSONL manifest or an equivalent manifest specification with at
least `task_family`, `expected_answer`, `solver_label`, `perturbation_type`,
`verifier_target`, and `repair_target` for each usable fixture. The terminal
artifact MUST include `eval_protocol_ready`, `usable_fixture_count`,
`rejected_fixture_count`, `stratified_eval_manifest_path`,
`minimum_live_eval_count`, `fixture_family_counts`, `downstream_usage`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.

`eval_protocol_ready` shall be true only when at least 48 usable exact fixtures
are available, the derived manifest includes all usable exact fixtures, every
required manifest field is populated, rejected fixture accounting is explicit,
and downstream `.289` SOTA tasks define honest skip gates instead of promoting
tiny panels. Abstention and verifier-calibration reruns SHALL honestly skip
headline claims when fewer than 48 unique exact fixtures are selected. Formal
feedback and repair-gating reruns SHALL honestly skip when fewer than 18 and
24 repair-target fixtures respectively are selected. FR-11 stress reruns SHALL
honestly skip when fewer than 48 exact fixtures spanning accept, reject, and
repair targets are selected.

### SCENARIO-VERIFY-3097: Stratified Manifest Replaces Tiny Abstention Panel

Given the ready Exp 3084 exact fixture bank and the checked-in Exp 3085 panel
artifact,
When the Exp 3097 audit runs,
Then it reports the full usable/rejected fixture count, explains that Exp 3085
used only 9 unique exact fixtures because its row transcript contains two
prompt-policy rows for each of 3 fixtures per family, writes the `.289`
stratified manifest with exact post-generation targets, records no live model
inference, and requires downstream tasks to skip rather than headline results
below the minimum live-evaluation counts.

## Implementation Status (REQ-VERIFY-3097)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3097 | Implemented (`python/carnot/eval/exact_fixture_eval_protocol_audit_v1.py`) | Implemented (`tests/python/test_experiment_3097_exact_fixture_eval_protocol_audit.py`) |

### REQ-VERIFY-3098: MaxSAT Abstention Routing Policy

The repository shall provide an Exp 3098 offline MaxSAT-style routing policy
for `.289` SOTA verifier and repair experiments. The policy shall write
`results/experiment_3098_maxsat_abstention_routing_policy_v1.json` and a
machine-readable routing policy file without modifying
`scripts/research_conductor.py` or invoking live model inference.

The policy shall translate Carnot accept, reject, and abstain decisions into
non-negotiable hard constraints and auditable weighted soft constraints. Hard
constraints MUST cover exact-label disagreement, mandated model-cache
availability for headline live runs, syntax/schema validity, repair intent
preservation, no-tiny-panel disqualification, and the requirement that formal
feedback lift must be nonnegative before repair promotion. Weighted soft
constraints MUST expose the tradeoffs among accepting exact-consistent
candidates, rejecting exact-inconsistent candidates, abstaining on uncertainty,
preferring formal-feedback lift, preserving repair intent, minimizing false
accepts, minimizing false rejects, and minimizing unnecessary abstention.

When a MaxSAT/MaxSMT solver package is unavailable locally, the policy shall
fail closed to a deterministic reference evaluator that enumerates the three
actions (`accept`, `reject`, `abstain`), filters any action violating a hard
constraint, scores the remaining actions by the declared soft weights, and
breaks ties in the fixed order `abstain`, `reject`, `accept`. The fallback
MUST be described in the terminal artifact so downstream tasks cannot silently
use an absent solver as permission to accept.

The terminal artifact MUST include `maxsat_policy_ready`,
`routing_policy_path`, `hard_constraints`, `soft_constraints`,
`objective_terms`, `fallback_evaluator`, `downstream_usage`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.
`maxsat_policy_ready` shall be true only when the routing policy file exists,
all required hard and soft constraint topics are represented, the fallback
evaluator is deterministic and fail-closed, downstream usage is defined for
Exp 3099, Exp 3101, and Exp 3102, prior `.288` failure artifacts are cited,
and `inference_substrate` declares no live model inference.

### SCENARIO-VERIFY-3098: Policy Routes Accept Reject And Abstain Deterministically

Given the Exp 3097 exact-fixture protocol is ready and the `.288` abstention,
formal-feedback, calibration, and capstone artifacts are available,
When Exp 3098 builds the MaxSAT-style policy,
Then it writes the terminal artifact with complete hard/soft constraints,
records the OpenReview weighted MaxSAT routing reference as design context,
writes a concrete policy JSON consumed by Exp 3099, Exp 3101, and Exp 3102,
and reports a deterministic fallback evaluator that abstains when every
accept/reject route violates a hard constraint or ties with a safer route.

## Implementation Status (REQ-VERIFY-3098)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3098 | Implemented (`python/carnot/eval/maxsat_abstention_routing_policy_v1.py`) | Implemented (`tests/python/test_experiment_3098_maxsat_abstention_routing_policy.py`) |

### REQ-VERIFY-3099: Local SOTA Confidence Abstention Panel V3

The repository shall provide an Exp 3099 local SOTA confidence and abstention
panel for milestone `.289` that consumes the Exp 3097 exact-fixture evaluation
protocol and the Exp 3098 MaxSAT-style routing policy. The panel shall write
`results/experiment_3099_local_sota_confidence_abstention_panel_v3.json` and
a row-level transcript without modifying `scripts/research_conductor.py` or
substituting legacy tiny models for headline evidence.

The panel shall first verify that Exp 3097 reports `eval_protocol_ready=true`,
that the stratified exact-fixture manifest is readable, that Exp 3098 reports
`maxsat_policy_ready=true`, that
`results/maxsat_abstention_routing_policy_3098/policy.json` is loaded, that
CUDA/GPU runtime metadata is recorded, and that mandated local GGUF cache
status is measured for `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. If cache, runtime, protocol, policy, or
minimum exact-count preconditions fail, it shall write a terminal blocked
artifact with readiness false and explicit blocked, skipped, and
cache-missing outcomes.

When preconditions pass, the panel shall run at least the Exp 3097
`minimum_live_eval_count` exact fixtures using only cached mandated local
SOTA GGUF models. Every prompt shall be represented by a SHA-256 hash and
shall use only leakage-safe prompt payloads before exact labels are consulted.
Every generated row shall be routed through the loaded MaxSAT policy and shall
record the raw parsed answer, confidence signal, policy route decision,
exact-answer match, expected action, and route audit details.

The terminal artifact MUST include `abstention_panel_v3_ready`, `model_specs`,
`exact_ground_truth_count`, `abstention_precision`, `rejection_recall`,
`abstention_coverage`, `false_accept_rate`, `false_reject_rate`,
`solve_accuracy`, `verification_accuracy`, `maxsat_policy_used`,
`thermodynamic_decode_telemetry`, `prompt_hashes`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. It shall also record route
decision counts, cache status for every mandated model, selected model IDs,
prompt-hash count, exact fixture counts, row transcript checksum, and any
optional entropy-production or free-energy-style token telemetry as
diagnostic-only evidence.

`abstention_panel_v3_ready` shall be true only when the exact protocol and
MaxSAT policy are both ready, at least `minimum_live_eval_count` exact
fixtures are evaluated, at least one cached mandated local SOTA GGUF produced
live output, legacy tiny models are not promoted, `maxsat_policy_used=true`,
and the `honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-3099: MaxSAT Routes A Minimum Exact Abstention Panel

Given the Exp 3097 exact-fixture protocol is ready, the Exp 3098 routing
policy is ready, CUDA/GPU metadata can be recorded, and a mandated local SOTA
GGUF can load,
When Exp 3099 runs the confidence and abstention panel,
Then it evaluates at least `minimum_live_eval_count` exact fixtures, hashes
every leakage-safe prompt, parses model answers and confidence, routes every
row through the loaded MaxSAT policy, computes solve accuracy, verification
accuracy, abstention precision, rejection recall, abstention coverage, false
accept rate, false reject rate, and route-decision counts, records optional
token-telemetry diagnostics without making them a gate, and writes the
required terminal artifact fields.

### SCENARIO-VERIFY-3099-BLOCKED: Missing Protocol Policy Or SOTA Runtime Fails Closed

Given the exact protocol is not ready, the MaxSAT policy is absent or not
ready, fewer than `minimum_live_eval_count` exact fixtures are available,
CUDA/GPU runtime checks fail, or no mandated local SOTA GGUF resolves and
loads,
When Exp 3099 runs,
Then it writes the required artifact fields with
`abstention_panel_v3_ready=false`, `maxsat_policy_used` reflecting whether the
policy was actually loaded, zero promoted exact-ground-truth count, empty
prompt hashes, explicit blocked/skipped/cache-missing outcomes, no legacy
tiny-model headline substitution, and an `honest_verdict` beginning with the
blocked precondition reason.

## Implementation Status (REQ-VERIFY-3099)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3099 | Implemented (`python/carnot/eval/local_sota_confidence_abstention_panel_v3.py`) | Implemented (`tests/python/test_experiment_3099_local_sota_confidence_abstention_panel_v3.py`) |

### REQ-VERIFY-3100: Z3/Test-Oracle Formal-Feedback V2 Pilot

The repository shall provide an Exp 3100 Z3/test-oracle formal-feedback v2
pilot for milestone `.289` that consumes the Exp 3097 exact-fixture protocol
and writes `results/experiment_3100_z3_oracle_feedback_v2.json` without
modifying `scripts/research_conductor.py`, without pushing, and without
claiming Dafny execution when Dafny is unavailable.

The pilot shall preflight Z3, Dafny, local mandated SOTA GGUF cache status, the
`cached_sota_pair()` readiness signal or local equivalent, Exp 3097 protocol
readiness, and the derived stratified exact-fixture manifest. If Dafny is
unavailable but Z3 and Python test-oracle checks are available, the pilot shall
continue with `dafny_available=false` and `z3_available=true`. If Z3 or the
exact protocol is unavailable, the pilot shall still write a terminal blocked
artifact with explicit precondition diagnostics and no promoted headline
evidence.

When exact repair fixtures are available, the pilot shall select a small
deterministic panel from the `.289` protocol and evaluate three conditions
where feasible: no-feedback candidate reuse, solver-only fallback repair, and
LLM-guided repair. Every candidate counted as a success must pass the exact
execution/test-oracle check for its fixture. Z3-based numeric repairs must also
pass solver checks. JSON syntax repairs and Python assertion repairs must pass
local execution/test-oracle checks. Vacuous successes, empty repairs, and
candidate outputs that avoid the required repaired field or specification must
be disqualified using guards inspired by MiniF2F-Dafny empty-proof rejection and
formal annotation work.

The terminal artifact MUST include `formal_feedback_v2_ready`, `model_specs`,
`z3_available`, `dafny_available`, `exact_ground_truth_count`,
`formal_feedback_delta`, `guided_success_count`,
`solver_only_success_count`, `vacuity_guard_passed`, `test_oracle_count`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`. It shall also
record selected fixture rows, per-condition success counts, cache status for
every mandated model ID, whether headline live LLM execution was blocked, and
which test-oracle or Z3 authority validated every accepted row.

`formal_feedback_v2_ready` shall be true only when the Exp 3097 protocol is
ready, at least one exact repair fixture is evaluated, Z3 and execution
test-oracles run, vacuity guards pass, model cache preconditions permit
headline guided execution, at least one mandated SOTA GGUF generated live
guided repair text, and `guided_success_count > solver_only_success_count`.
When the model-cache precondition blocks headline execution, the artifact shall
start `honest_verdict` with `complete_` or another terminal success prefix only
as an honest blocked-headline/pilot artifact, while keeping
`formal_feedback_v2_ready=false`.

### SCENARIO-VERIFY-3100: Z3 Oracle Feedback Runs Without Dafny

Given Exp 3097 is ready, the stratified manifest contains repair-target
fixtures, Z3 is importable, Dafny is unavailable, and local test-oracle checks
can execute,
When Exp 3100 runs,
Then it writes the terminal artifact with `dafny_available=false`,
`z3_available=true`, selected exact repair fixtures, no-feedback,
solver-only, and guided condition rows where feasible, explicit
`test_oracle_count`, vacuity disqualification evidence, source artifact
checksums, and an honest readiness signal that does not claim Dafny execution.

### SCENARIO-VERIFY-3100-BLOCKED-HEADLINE: Missing SOTA Pair Does Not Promote

Given Z3 and the exact protocol are available but `cached_sota_pair()` or the
local equivalent cannot resolve enough mandated GGUF models for headline guided
execution,
When Exp 3100 runs,
Then it still writes `results/experiment_3100_z3_oracle_feedback_v2.json`,
records per-model cache status and any solver/test-oracle pilot rows, keeps
`formal_feedback_v2_ready=false`, sets promoted guided counts only for live
model validated successes, and uses an `honest_verdict` beginning with a
terminal complete or blocked-headline prefix rather than claiming a clean
formal-feedback lift.

## Implementation Status (REQ-VERIFY-3100)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3100 | Implemented (`python/carnot/eval/z3_oracle_feedback_v2.py`) | Implemented (`tests/python/test_experiment_3100_z3_oracle_feedback_v2.py`) |

### REQ-VERIFY-3112: Logic-Regularized Verifier Pilot On Exact Fixtures

The repository shall provide an Exp 3112 deterministic pilot that writes
`results/experiment_3112_logic_regularized_verifier_pilot_v1.json` by scoring
a non-tiny `.289` exact-fixture subset with LOVER-style logic diagnostics
without promoting the result as verifier recovery.

The pilot shall consume the Exp 3097 stratified exact-fixture manifest, Exp
3099 local SOTA confidence/abstention panel rows, Exp 3110 mandated model
cache manifest, and Exp 3111 certified coherence feedback v3 certificates. It
shall select only fixtures that have clear positive/negative exact labels and
matching certified feedback fields, construct multiple deterministic
candidate/reasoning paths per fixture including contrastive true/false
assertions, and keep exact solver labels as the sole ground truth authority.

The pilot shall compute negation consistency, intra-answer-group consistency,
inter-answer-group consistency, agreement with exact labels, verifier recall
movement, and false-positive movement against the Exp 3099 route decisions.
False-positive and false-negative movement shall be reported separately. New
live LLM inference is optional; if it is not run, the artifact shall set
`live_llm_inference=false` and identify any reused cached trace source without
conflating it with a fresh live run.

The terminal artifact MUST include
`logic_regularized_verifier_pilot_ready`, `model_specs`,
`mandatory_headline_model_ids`, `selected_headline_model_ids`,
`live_llm_inference`, `exact_ground_truth_count`,
`negation_consistency_rate`, `answer_group_consistency_rate`,
`verifier_recall_delta`, `false_positive_delta`, `tests_run`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`. It shall also
record selected fixture IDs, path counts, separate false-positive and
false-negative movement summaries, the row-level diagnostic transcript path,
and whether any promotion claim was made. `logic_regularized_verifier_pilot_ready`
shall be true only when the selected exact subset is non-tiny, every selected
fixture has certified feedback v3 fields, at least two candidate paths are
scored per fixture, all required diagnostics are finite rates in `[0, 1]`, and
the `honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-3112: Logic Diagnostics Measure Movement Without Promotion

Given Exp 3097, Exp 3099, Exp 3110, and Exp 3111 source artifacts are present,
When Exp 3112 scores the exact positive/negative subset,
Then it writes the required artifact fields, records no fresh live LLM
inference unless explicitly run, computes negation and answer-group
consistency over contrastive paths, compares its pilot decisions against Exp
3099 route decisions, reports false-positive and false-negative movement
separately, and keeps promotion disabled while exact labels remain authority.

## Implementation Status (REQ-VERIFY-3112)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3112 | Implemented (`python/carnot/eval/logic_regularized_verifier_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3112_logic_regularized_verifier_pilot.py`) |

### REQ-VERIFY-3113: Diagnostic Local SOTA Verifier Calibration V5

The repository shall provide an Exp 3113 diagnostic local SOTA verifier
calibration builder that writes
`results/experiment_3113_diagnostic_local_sota_verifier_calibration_v5.json`
from the Exp 3097 exact-fixture protocol, Exp 3098 MaxSAT routing policy, Exp
3099 cached/live local SOTA abstention rows, Exp 3110 mandated model-cache
manifest, Exp 3111 certified coherence feedback v3, and Exp 3112
logic-regularized verifier pilot signals. The builder MUST NOT modify
`scripts/research_conductor.py`, push commits, substitute legacy tiny models
for headline evidence, or conflate reused cached traces with fresh live
inference.

The calibration shall select fixture rows with exact solver/test labels,
MaxSAT route decisions, certified coherence certificates, and
logic-regularized pilot decisions where available. It shall compute the raw
verifier-gain delta against the cached local SOTA MaxSAT route decisions, the
verifier-gain delta after certified-coherence adjudication, false-accept rate,
false-reject rate, calibration error, abstention precision, and rejection
recall. Calibration evidence MUST be emitted even when the measured lift is
zero or negative; repair promotion is controlled separately by
`repair_gate_state`.

The terminal artifact MUST include
`diagnostic_verifier_calibration_v5_ready`, `model_specs`,
`mandatory_headline_model_ids`, `selected_headline_model_ids`,
`live_llm_inference`, `exact_ground_truth_count`, `verifier_gain_delta`,
`verifier_gain_delta_with_certified_coherence`, `false_accept_rate`,
`false_reject_rate`, `calibration_error`, `abstention_precision`,
`rejection_recall`, `repair_gate_state`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. It shall also include a
machine-readable Exp 3115 repair-gate explanation that records the delta sign,
blocking reason, and downstream action.

`repair_gate_state` shall be one of `unblocked`, `blocked_negative_delta`,
`blocked_model_cache`, `blocked_tiny_panel`, or `blocked_missing_inputs`.
Missing required source artifacts or unreadable row evidence shall produce
`blocked_missing_inputs`. A below-floor exact calibration panel shall produce
`blocked_tiny_panel`. Missing selected mandated local SOTA cache evidence
shall produce `blocked_model_cache` while preserving any solver-only
diagnostics. Non-positive certified-coherence verifier gain shall produce
`blocked_negative_delta`. Only a non-tiny diagnostic panel with at least one
selected mandated cached headline model and positive certified-coherence gain
may set `repair_gate_state=unblocked`.

### SCENARIO-VERIFY-3113: Certified Diagnostic Calibration Gates Repair Without Positive-Lift Assumption

Given Exp 3097, Exp 3098, Exp 3099, Exp 3110, Exp 3111, and Exp 3112 source
artifacts are present,
When Exp 3113 builds the diagnostic calibration,
Then it writes the required machine-readable artifact, reuses cached/live
mandated SOTA traces without claiming fresh live inference, compares route,
logic-pilot, and certified-coherence decisions against exact labels, reports
the required safety and calibration metrics, and sets the repair gate to
`unblocked` only when the measured certified-coherence lift is positive and
model-cache and panel-size gates also pass.

## Implementation Status (REQ-VERIFY-3113)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3113 | Implemented (`python/carnot/eval/diagnostic_local_sota_verifier_calibration_v5.py`) | Implemented (`tests/python/test_experiment_3113_diagnostic_local_sota_verifier_calibration_v5.py`) |

### REQ-VERIFY-3114: Fragment-Level Code And Constraint Verification Pilot

The repository shall provide an Exp 3114 deterministic offline pilot that
writes
`results/experiment_3114_fragment_level_code_constraint_verification_pilot_v1.json`
from checked-in exact fixture artifacts only. The pilot MUST NOT invoke live
LLM inference, modify `scripts/research_conductor.py`, push commits, or claim
repair success. Its purpose is to localize exact candidate failures into
repairable fragments or constraint clauses before any broader repair run.

The pilot shall consume the Exp 3097 stratified exact-fixture manifest, Exp
3100 Z3/test-oracle feedback artifact, and Exp 3111 certified-coherence
artifact. It shall select deterministic fixtures that include repairable
invalid candidates and code/assertion-style constraints. For each selected
fixture it shall split candidates into fragments or clauses using structured
parsing where available: JSON parsing for JSON candidates, Python AST parsing
for assertion candidates, and structured constraint parsing for integer
assignment clauses. Each fragment or clause shall be checked with an exact
offline authority and labeled `pass`, `fail`, `unknown`, or `non-applicable`.

The pilot shall emit a repair target manifest at
`results/fragment_verification_pilot_3114/repair_target_manifest.jsonl` with
one row per failing fragment. Each manifest row MUST include
`fixture_id`, `fragment_id`, `failing_constraint`, `expected_direction`, and
`solver_evidence`. Solver evidence may come from exact Python AST/runtime,
JSON parser diagnostics, or deterministic integer constraint evaluation, but
the authority must be explicit and must not depend on live model output.

The terminal artifact MUST include
`fragment_verification_pilot_ready`, `exact_fixture_count`, `fragment_count`,
`failing_fragment_count`, `unknown_fragment_count`,
`repair_target_manifest_path`, `tests_run`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. `fragment_verification_pilot_ready`
shall be true only when at least one exact fixture is selected, at least one
fragment is checked, failing fragments are written to the repair target
manifest, unknown fragments are counted honestly, all required source
artifacts are present, no live inference is declared, and the `honest_verdict`
starts with a terminal success prefix.

### SCENARIO-VERIFY-3114: Fragment Checks Localize Repair Targets Offline

Given Exp 3097, Exp 3100, and Exp 3111 source artifacts are present and the
stratified exact-fixture manifest contains repairable JSON, numeric, Python
assertion, and arithmetic assertion rows,
When the Exp 3114 pilot runs,
Then it writes the required terminal JSON artifact, records no live LLM
inference, emits fragment rows labeled with exact pass/fail/unknown or
non-applicable states, writes a repair target manifest whose failing rows carry
the required localization fields, reports fixture and fragment counts, and
uses an `honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-VERIFY-3114)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3114 | Implemented (`python/carnot/eval/fragment_verification_pilot_3114.py`) | Implemented (`tests/python/test_experiment_3114_fragment_level_code_constraint_verification_pilot.py`) |

### REQ-VERIFY-3115: Explicit Repair Gate And Micro-Panel V4

The repository shall provide an Exp 3115 explicit repair boundary builder that
writes `results/experiment_3115_explicit_repair_gate_micro_panel_v4.json` in
all cases. The builder shall consume Exp 3113 diagnostic repair-gate evidence
and Exp 3114 fragment-level repair targets. It MUST NOT modify
`scripts/research_conductor.py`, push commits, use legacy tiny models for
headline repair evidence, or allow the repair micro-panel to disappear as a
missing artifact.

If Exp 3113 `repair_gate_state` is not `unblocked`, the builder MUST NOT run
repair generation. It shall still write a terminal artifact with
`repair_micro_panel_v4_artifact_ready=true`, `repair_unblocked=false`,
`repair_run_executed=false`, and an actionable `gate_block_reason` derived from
the upstream gate state and source evidence. If Exp 3113 is unblocked, the
builder shall load the Exp 3114 repair target manifest and attempt the smallest
available exact-fixture repair micro-panel using at least one selected cached
mandated local SOTA GGUF model. Any missing target manifest, missing selected
SOTA model, or runtime failure shall be recorded as an explicit blocked repair
boundary rather than a missing artifact.

Executed repair rows shall verify repaired outputs with exact offline
authority matching the target family: Python AST arithmetic assertions for
assertion repairs, Python JSON parsing for JSON repairs, and deterministic
integer constraint evaluation for numeric constraint repairs. The artifact
shall report `repair_success_delta`, `false_repair_accept_rate`, and
`intent_preservation_rate`; blocked artifacts shall set these metrics to
finite zero values and state why generation did not run. Model/cache metadata
and inference substrate details shall remain auditable whether repair ran or
remained blocked.

The terminal artifact MUST include
`repair_micro_panel_v4_artifact_ready`, `repair_unblocked`,
`repair_run_executed`, `gate_block_reason`, `model_specs`,
`selected_headline_model_ids`, `exact_ground_truth_count`,
`repair_success_delta`, `false_repair_accept_rate`,
`intent_preservation_rate`, `tests_run`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`.

### SCENARIO-VERIFY-3115: Repair Boundary Artifact Never Disappears

Given Exp 3113 and Exp 3114 source artifacts may be present, blocked, or
unblocked,
When Exp 3115 builds the explicit repair gate and micro-panel artifact,
Then it always writes
`results/experiment_3115_explicit_repair_gate_micro_panel_v4.json`, separates
`repair_unblocked` from `repair_run_executed`, records concrete blocked
reasons when generation is not authorized or cannot execute, and, when repair
does execute, verifies every accepted repair against exact offline authority
while preserving mandated SOTA model/cache provenance.

## Implementation Status (REQ-VERIFY-3115)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3115 | Implemented (`python/carnot/eval/explicit_repair_gate_micro_panel_v4.py`) | Implemented (`tests/python/test_experiment_3115_explicit_repair_gate_micro_panel_v4.py`) |

### REQ-VERIFY-3124: Difficulty-Stratified Live SOTA Verifier Panel V6

The repository shall provide an Exp 3124 difficulty-stratified live SOTA
verifier-panel builder that writes
`results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json`
from the Exp 3123 mandated-model cache/precondition manifest and the .290
exact-verifier artifacts, including Exp 3097 exact fixtures, Exp 3099 cached
local-SOTA route rows, Exp 3111 certified-coherence certificates, Exp 3112
logic-regularized verifier rows, Exp 3113 diagnostic calibration, Exp 3114
fragment/code verification rows, and Exp 3115 repair-gate evidence. The
builder MUST NOT modify `scripts/research_conductor.py`, push commits, or
substitute legacy small models for headline evidence.

The panel shall preserve exact solver/test labels as the sole authority. It
shall label rows by difficulty bucket, fixture family, generator family, and
answer-extraction format. The difficulty stratification MUST include stable
keys for `easy`, `medium`, `hard`, `contradiction`, `satisfiable_drift`, and
`fragment_code` even when a bucket has no live call. When at least one mandated
local SOTA model is usable, the builder shall run a bounded number of live
verifier calls, record prompt hashes, bounded model hashes or cache-path
evidence, raw outputs, extracted answers, exact labels, decisions, and failure
mechanisms. When no mandated model is locally usable or the live runtime cannot
load one, it shall write a complete diagnostic artifact with
`live_call_count=0`, `headline_claim_allowed=false`, preserved fixture
metadata, and an explicit blocked repair gate.

The terminal artifact MUST include
`difficulty_stratified_live_sota_panel_v6_ready`, `model_specs`,
`selected_model_ids`, `live_call_count`, `headline_claim_allowed`,
`exact_ground_truth_count`, `difficulty_buckets`, `fixture_family_metrics`,
`answer_extraction_metrics`, `failure_mechanism_counts`,
`false_accept_rate`, `false_reject_rate`, `verifier_gain_delta`,
`repair_gate_state`, `tests_run`, `source_artifacts`, `inference_substrate`,
and `honest_verdict`. It SHOULD also record live rows, panel fixture metadata,
generator-family metrics, source checksums, and validation notes so downstream
repair tasks can audit why the gate is open or blocked.

`difficulty_stratified_live_sota_panel_v6_ready` shall be true only when
required source artifacts are present, exact labels are available, at least one
mandated model produced live bounded calls, and all finite metrics were computed
from live rows. `headline_claim_allowed` shall remain false when
`live_call_count=0`, when only legacy small models are available, when the
runtime fails before live evidence is produced, or when the panel is too small
or unsafe for headline wording. `repair_gate_state` shall block repair when
inputs are missing, no mandated live model ran, the panel is tiny, any false
accept occurs, or measured verifier gain is not positive.

### SCENARIO-VERIFY-3124: Stratified Live Panel Blocks Headline Lift Without Live Mandated Evidence

Given Exp 3123 and the .290 exact-verifier artifacts are present,
When Exp 3124 builds the difficulty-stratified panel,
Then it writes the required terminal JSON artifact, preserves every selected
fixture's exact authority metadata, reports difficulty, fixture-family,
generator-family, and extraction-format metrics, records live local SOTA
outputs only from mandated models, and sets headline and repair gates from the
observed false-accept, false-reject, precision, recall, panel-size, and lift
metrics. If no mandated local model can produce live verifier output, it still
writes the diagnostic artifact with `live_call_count=0`,
`headline_claim_allowed=false`, preserved fixture metadata, and an honest
blocked verdict.

## Implementation Status (REQ-VERIFY-3124)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3124 | Implemented (`python/carnot/eval/difficulty_stratified_live_sota_verifier_panel_v6.py`) | Implemented (`tests/python/test_experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.py`) |

### REQ-VERIFY-3125: Prefix-Closed Deterministic Verifier Bound Pilot

The repository shall provide an Exp 3125 bounded local pilot that enumerates a
small deterministic token/prefix frontier over exact prefix-closed semantic
constraints and writes
`results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json`.
The pilot MUST NOT run live model inference, MUST NOT modify
`scripts/research_conductor.py`, and MUST NOT claim full LLM correctness beyond
the explored frontier.

The pilot shall use a small exact fixture subset with JSON-like answer shape,
answer-label semantics, bounded numeric invariants, and forbidden-token
constraints. The prefix abstraction shall reject prefixes that cannot be
extended into a satisfying terminal candidate, preserve monotonic rejection
once a prefix is pruned, and aggregate conservative lower and upper
satisfaction bounds from a deterministic finite token prior. Explored frontier
size, pruned-prefix count, bound width, explored probability mass, and semantic
coverage limitations shall be reported explicitly.

The terminal artifact MUST include `prefix_closed_bound_pilot_ready`,
`constraint_families`, `fixture_count`, `explored_prefix_count`,
`pruned_prefix_count`, `lower_bound`, `upper_bound`, `bound_width`,
`semantic_coverage`, `limitations`, `tests_run`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`.

`prefix_closed_bound_pilot_ready` shall be true only when at least one exact
fixture is enumerated, the deterministic frontier has both accepted or viable
mass and pruned-prefix evidence, lower and upper bounds are finite and ordered,
`bound_width == upper_bound - lower_bound`, semantic coverage separates syntax
from semantic and non-covered live-LLM claims, no live model inference is used,
and `honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-3125: Exact Prefix Frontier Produces Bounded Local Evidence

Given the Exp 3125 exact JSON-like fixtures and deterministic token prior are
available,
When the prefix-closed bound pilot enumerates the bounded frontier,
Then it accepts only terminal candidates that satisfy exact JSON parsing,
answer-label, bounded-score, and forbidden-token checks; rejects invalid
prefixes monotonically; aggregates lower and upper satisfaction bounds from the
explored prefix mass; reports frontier, pruned-prefix, semantic-coverage, and
limitation fields; and writes the terminal artifact without making a live-model
or open-world correctness claim.

## Implementation Status (REQ-VERIFY-3125)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3125 | Implemented (`python/carnot/eval/prefix_closed_deterministic_verifier_bound_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3125_prefix_closed_deterministic_verifier_bound_pilot.py`) |

### REQ-VERIFY-3126: Fragment-Time Monitor And Satisfiable-Drift Audit

The repository shall provide an Exp 3126 deterministic fragment-time monitor
that writes
`results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json`
from checked-in exact fixture and trace artifacts only. The monitor MUST
consume the .290 exact-verifier manifest exposed by Exp 3097, the Exp 3114
fragment/code verification artifact, and the Exp 3124 difficulty-stratified
live SOTA verifier panel. It MUST NOT run fresh live inference, modify
`scripts/research_conductor.py`, push commits, or claim repair success.

The monitor shall emit replayable event rows for partial trace state,
candidate final answer, constraint ledger, exact test/Z3 result, and drift
classification. Constraint-ledger events shall preserve the ledger action
derived from exact fragment status when fragments are present and from exact
solver/test labels otherwise. Candidate-answer events shall compare returned
answers against both exact labels and the maintained ledger. Drift
classification events shall separate `contradiction`, `satisfiable_drift`,
`extraction_format_failure`, `data_prior_mismatch`, and `unknown` failures;
non-failing rows may be labeled `no_failure` and missing bounded-panel returns
may be labeled `not_observed` without being counted as monitor violations.

The terminal artifact MUST include
`fragment_time_monitor_v1_ready`, `monitored_fixture_count`,
`monitor_event_schema`, `monitor_violation_count`,
`satisfiable_drift_count`, `contradiction_count`,
`ledger_consistency_rate`, `failure_mechanism_counts`,
`downstream_repair_constraints`, `tests_run`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. It SHOULD also include
`monitor_events`, `monitor_event_count`, `ledger_replay_summary`,
`self_checks`, and `event_stream_hash` so downstream repair can verify that
partial-state evidence was replayed before any repair prompt is authorized.

`fragment_time_monitor_v1_ready` shall be true only when at least one exact
fixture is monitored, the required event schema is present, all required
source artifacts are available, ledger replay passes, final-answer consistency
is checked for every observed candidate answer, monitor determinism is
self-checked, no fresh live inference is declared, and `honest_verdict` starts
with a terminal success prefix. `monitor_violation_count`,
`satisfiable_drift_count`, `contradiction_count`, and
`ledger_consistency_rate` shall be derived from replayed monitor events rather
than copied unverified from upstream artifacts.

### SCENARIO-VERIFY-3126: Monitor Evidence Gates Repair Before Any Repair Claim

Given the .290 exact fixture manifest, Exp 3114 fragment/code verification
rows, and Exp 3124 verifier-panel live rows are present,
When Exp 3126 builds the fragment-time monitor artifact,
Then it writes the required terminal JSON artifact, records no new live
inference, defines the replayable monitor event schema, emits event rows for
partial state, final answer, ledger, exact result, and drift classification,
counts contradiction separately from satisfiable drift, reports
ledger-consistency and failure-mechanism counts from event replay, records
self-checks for ledger replay, final-answer consistency, and determinism, and
exposes downstream repair constraints that require monitor evidence before any
repair generation or repair-success claim.

## Implementation Status (REQ-VERIFY-3126)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3126 | Implemented (`python/carnot/eval/fragment_time_monitor_satisfiable_drift_audit_v1.py`) | Implemented (`tests/python/test_experiment_3126_fragment_time_monitor_satisfiable_drift_audit.py`) |

### REQ-VERIFY-3137: Exact-Safe Accept Abstain Reject Contract V1

The repository shall provide an Exp 3137 exact-safe accept/abstain/reject
contract builder that writes
`results/experiment_3137_exact_safe_accept_abstain_contract_v1.json` from
checked-in artifacts only. The builder MUST consume the Exp 3136 false-accept
autopsy, Exp 3125 prefix-closed bound pilot, Exp 3126 fragment-time monitor
ledger, Exp 3098 MaxSAT abstention policy when present, and the Exp 3097 exact
fixture manifest. It MUST NOT run fresh live inference, modify
`scripts/research_conductor.py`, push commits, or open a repair gate directly.

The contract shall define ordered auditable rules for `accept`, `abstain`, and
`reject`. Accept shall require exact accept labels, parse confidence at the
declared exact floor, answer-token-family agreement, premise/answer
consistency, prefix-bound coverage for the accepted label, and monitor-ledger
agreement when replay evidence exists. Reject shall be allowed for exact
reject labels when the candidate agrees with the exact label or the exact
authority and monitor ledger both require rejection. Abstain shall take
precedence for known .291 false-accept regression rows, known false-accept
failure families, missing exact labels, low parse confidence, token-family
mismatch, unavailable accept-label prefix coverage, missing live monitor
ledger evidence, or any residual unsafe tie.

The builder shall replay the contract over the .291 live verifier rows from
the Exp 3136 autopsy and over the prior exact fixture rows exposed by the
Exp 3097 manifest. Replay metrics shall count false accepts, abstentions, and
false rejects directly from the replayed decisions. All known Exp 3136
regression rows MUST appear in `regression_row_set` and MUST NOT receive an
`accept` replay decision; abstention is permitted for those rows.

The terminal artifact MUST include `acceptance_contract_v1_ready`,
`contract_rules`, `known_false_accept_rows_blocked`,
`replay_false_accept_rate`, `replay_abstention_rate`,
`replay_false_reject_rate`, `repair_gate_prerequisites`,
`regression_row_set`, `tests_run`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. It SHOULD also include replay
rows, replay counts, self-checks, source checksums, and no-live-inference
guard fields so the next live rerun can consume the contract without implicit
policy inference.

`acceptance_contract_v1_ready` shall be true only when all required source
artifacts are present, the ordered rules include at least one accept rule, one
reject rule, and one abstain rule, the replay has zero false accepts, every
known false-accept regression row is blocked from accept, replay metrics are
finite rates in `[0, 1]`, deterministic self-checks pass, no fresh inference
is declared, and the `honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-3137: Regression Rows Are Blocked Before Live Rerun

Given the Exp 3136 autopsy has recorded .291 false-accept rows, Exp 3125 has
reported prefix-bound coverage, Exp 3126 has replayable monitor-ledger events,
and the Exp 3097 exact fixture manifest is available,
When Exp 3137 builds the exact-safe contract,
Then it writes the required terminal JSON artifact, exposes ordered
machine-readable contract rules, replays every .291 live row and prior exact
fixture row, reports zero replay false accepts, reports the abstention and
false-reject rates, lists all known false-accept regression rows, and records
repair-gate prerequisites that require the next live rerun to load this
contract before accepting any verifier row.

## Implementation Status (REQ-VERIFY-3137)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3137 | Implemented (`python/carnot/verify/exact_safe_accept_abstain_contract_v1.py`) | Implemented (`tests/python/test_experiment_3137_exact_safe_accept_abstain_contract_v1.py`) |

### REQ-VERIFY-3138: Canonical Answer And VeriCoT Grounding Pilot V1

The repository shall provide an Exp 3138 deterministic canonical-answer and
premise-grounding pilot that writes
`results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json`
from checked-in artifacts only. The pilot MUST consume the Exp 3136
false-accept autopsy, Exp 3137 exact-safe contract, Exp 3111 certified
coherence/Z3 feedback, and Exp 3126 fragment-time monitor ledger. It MUST NOT
run fresh live inference, invoke a model-backed verifier, modify
`scripts/research_conductor.py`, push commits, or claim a production VeriCoT
verifier.

The pilot shall define canonical forms for validity labels, SAT labels,
numeric answers, bounded JSON-like outputs, and code-fragment verdicts.
Validity labels and SAT labels SHALL remain different answer-token families so
that `VALID` is not answer-equivalent to `SAT` or `UNSAT`. Numeric answers
SHALL normalize equivalent integer/decimal surface forms. Bounded JSON-like
outputs SHALL normalize key order and compact separators while refusing
unbounded structures. Code-fragment verdicts SHALL normalize runtime verdict
phrases such as assertion pass/fail into a separate code-verdict family.

For every known false-accept regression row, the pilot shall map available
fragment, ledger, or explanation evidence into simple premise records. Rows
with no available step or premise evidence MUST include an explicit absent
premise-grounding record. The replay shall check candidate answer
canonicalization, premise-to-answer consistency, answer-to-premise
consistency, exact-label agreement, solver-certificate agreement, and
monitor-ledger replay agreement. Blocking reasons SHALL distinguish
canonicalization blocks, premise-grounding blocks, and ledger-replay blocks so
parser/answer extraction failures remain separate from reasoning failures.

The terminal artifact MUST include
`canonical_grounding_pilot_v1_ready`, `canonicalizer_implemented`,
`premise_grounding_rows`, `regression_rows_evaluated`,
`false_accept_rows_blocked`, `canonicalization_block_count`,
`premise_grounding_block_count`, `ledger_replay_block_count`,
`residual_false_accept_rows`, `tests_run`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. It SHOULD also include row-level
canonical forms, premise records, solver-certificate summaries, ledger replay
summaries, source checksums, and deterministic self-checks.

`canonical_grounding_pilot_v1_ready` shall be true only when the canonicalizer
is implemented in code, all required source artifacts are present, at least
one known false-accept regression row is evaluated, every evaluated
false-accept row is blocked from accept by one or more explicit categories,
source replay is deterministic, no live inference or solver execution is
declared, and `honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-3138: False-Accept Rows Are Blocked By Canonical Grounding

Given Exp 3136 has recorded .291 false-accept rows, Exp 3137 has replayed the
exact-safe accept/abstain/reject contract, Exp 3111 has solver-certificate
feedback for the same fixture IDs, and Exp 3126 has replayable monitor-ledger
events,
When Exp 3138 builds the canonical grounding pilot,
Then it writes the required terminal JSON artifact, canonicalizes the exact
and candidate answers for every known false-accept row, marks absent premise
grounding explicitly when no step evidence exists, reports row-level
premise/answer and answer/premise consistency against exact labels and solver
certificates, reports canonicalization, premise-grounding, and ledger-replay
block counts, leaves `residual_false_accept_rows` empty only when every known
false accept is blocked, and records that no live model-backed verifier was
introduced.

## Implementation Status (REQ-VERIFY-3138)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3138 | Implemented (`python/carnot/verify/canonical_answer_vericot_grounding_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3138_canonical_answer_vericot_grounding_pilot_v1.py`) |

### REQ-VERIFY-3139: Live SOTA Verifier Rerun V7 With Exact-Safe Contract

The repository shall provide an Exp 3139 live SOTA verifier rerun builder that
writes `results/experiment_3139_live_sota_verifier_rerun_v7.json` after the
Exp 3136 false-accept autopsy, Exp 3137 exact-safe accept/abstain/reject
contract, and Exp 3138 canonical grounding pilot are available. The rerun MUST
build its row set from all known .291 false-accept regression rows plus
balanced exact fixtures covering contradiction, satisfiable drift, medium or
hard difficulty, and fragment-code families. Exact solver/test labels remain
the sole scoring authority.

The rerun MUST select models only from the mandated local SOTA GGUF policy:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. The builder SHALL use `cached_sota_pair()`
or the Exp 3123 cache manifest to resolve local GGUF paths and SHALL NOT use
legacy small models for headline evidence. If no mandated model is locally
usable, or the live runtime fails before producing bounded calls, the builder
MUST write a complete blocked diagnostic artifact with `live_call_count=0`,
`headline_claim_allowed=false`, preserved row metadata, model/cache substrate
diagnostics, and an honest blocked verdict.

For every live row, the rerun SHALL record the prompt hash, selected model ID,
model path or cache-path evidence, raw output hash, extracted answer,
canonical exact and candidate answers, exact label, raw live decision, and
exact-safe contract decision. Metrics SHALL be computed from the exact-safe
contract decisions against exact labels, while also preserving raw live
baseline metrics so the false-accept reduction is auditable. `false_accept_rate`
SHALL be zero before a repair-gate candidate can be reported as ready.

The terminal artifact MUST include `live_verifier_rerun_v7_ready`,
`model_specs`, `selected_model_ids`, `live_call_count`,
`headline_claim_allowed`, `regression_rows_included`,
`exact_ground_truth_count`, `false_accept_rate`, `false_reject_rate`,
`abstention_rate`, `verifier_gain_delta`, `repair_gate_candidate_state`,
`false_accept_gate_passed`, `tests_run`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. It SHOULD also include rerun
rows, raw live metrics, exact-safe metrics, prompt/canonicalization evidence,
source checksums, and deterministic self-checks.

`live_verifier_rerun_v7_ready` shall be true only when all required source
artifacts are present, Exp 3137 and Exp 3138 are ready, every known regression
row is included, at least one mandated model produces bounded live output, the
exact-safe contract yields `false_accept_rate=0.0`, metrics are finite rates in
`[0, 1]`, the repair-gate candidate state is not blocked, no legacy model is
used for headline evidence, and `honest_verdict` starts with a terminal
success prefix.

### SCENARIO-VERIFY-3139: Exact-Safe Rerun Reduces False Accepts

Given Exp 3123 exposes mandated local SOTA cache evidence, Exp 3136 lists the
.291 false-accept regression rows, Exp 3137 defines the exact-safe contract,
and Exp 3138 reports canonical grounding readiness,
When Exp 3139 runs the bounded live verifier rerun,
Then it writes the required terminal JSON artifact, includes the known
regression rows in the live rerun set, records model and prompt provenance for
each live call, applies the exact-safe contract to each extracted answer,
reports false-accept, false-reject, abstention, and gain metrics from exact
labels, and sets `false_accept_gate_passed=true` and
`live_verifier_rerun_v7_ready=true` only when exact-safe decisions reduce
false accepts to zero without relying on legacy small-model evidence.

## Implementation Status (REQ-VERIFY-3139)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3139 | Planned (`python/carnot/verify/live_sota_verifier_rerun_v7.py`) | Planned (`tests/python/test_experiment_3139_live_sota_verifier_rerun_v7.py`) |

### REQ-VERIFY-3140: Repair-Gate Unlock Decision V1

The repository shall provide an Exp 3140 deterministic repair-gate unlock
decision builder that writes
`results/experiment_3140_repair_gate_unlock_decision_v1.json` from checked-in
artifacts only. The builder MUST consume the Exp 3139 live verifier rerun,
Exp 3137 exact-safe accept/abstain/reject contract, Exp 3126 fragment-time
monitor ledger, Exp 3125 prefix-closed deterministic bound pilot, and Exp 3115
prior repair micro-panel evidence. It MUST NOT call live models, execute repair
generation, modify `scripts/research_conductor.py`, push commits, or rely on a
headline claim when a gate precondition is missing.

The decision gate shall require `false_accept_rate <= 0.10`,
`regression_rows_included=true`, known false-accept rows blocked from accept,
exact labels present, replayable monitor-ledger evidence, a bounded live-model
rerun substrate, repair rows with explicit constraints, and no headline-claim
disqualifier. The decision state SHALL be exactly one of `unblocked`,
`blocked_false_accept`, `blocked_missing_live_model`,
`blocked_missing_exact_labels`, or `blocked_other`. If any false accepts remain
above the gate, the state MUST be `blocked_false_accept` and the builder MUST
leave `selected_repair_rows=[]`.

The terminal artifact MUST include `repair_gate_decision_v1_ready`,
`repair_gate_state`, `false_accept_rate`, `false_accept_gate_passed`,
`regression_rows_included`, `exact_authority_ready`, `monitor_ledger_ready`,
`selected_repair_rows`, `repair_blockers`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. It SHOULD also preserve the
known-false-accept, live-model, headline-disqualifier, and repair-row
availability checks that determined the state.

`repair_gate_decision_v1_ready` shall be true when the artifact reaches one of
the allowed terminal gate states with source provenance and an honest verdict.
For `repair_gate_state=unblocked`, `honest_verdict` MUST start with a terminal
success prefix and `selected_repair_rows` MUST list the explicit repair
denominator and constraints. For blocked states, `honest_verdict` may start
with the blocked state and `repair_blockers` MUST be non-empty and actionable.

### SCENARIO-VERIFY-3140: Repair Calls Stay Blocked Until The Gate Is Safe

Given Exp 3139 has rerun the live verifier with exact-safe decisions, Exp 3137
has blocked the known false-accept regression rows, Exp 3126 provides ledger
replay evidence, Exp 3125 exposes exact-label boundary coverage, and Exp 3115
identifies repair rows,
When Exp 3140 builds the repair-gate unlock decision,
Then it writes the required terminal JSON artifact without live inference,
sets `repair_gate_state=unblocked` only when every gate criterion passes,
lists selected repair rows with constraints for the unblocked case, and
otherwise preserves exact blockers while avoiding repair execution.

## Implementation Status (REQ-VERIFY-3140)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3140 | Implemented (`python/carnot/verify/repair_gate_unlock_decision_v1.py`) | Implemented (`tests/python/test_experiment_3140_repair_gate_unlock_decision_v1.py`) |

### REQ-VERIFY-3150: Adversarial Verifier-Evidence Corrigendum V1

The repository shall provide an Exp 3150 deterministic adversarial
verifier-evidence corrigendum that writes
`results/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.json`
from checked-in artifacts only. The builder MUST consume the Exp 3136
false-accept autopsy, Exp 3137 exact-safe accept/abstain/reject contract, Exp
3138 canonical grounding pilot, Exp 3139 live verifier rerun, Exp 3140 repair
gate decision, Exp 3147 cross-corpus matrix, and Exp 3148 capstone. It MUST
NOT call live models, execute repair generation, execute verifiers, modify
`scripts/research_conductor.py`, or push commits.

The corrigendum SHALL classify adversarial evidence flags separately,
including tautology, too-short duration, missing random seed or reproducibility
checksum, missing transcript evidence, inconsistent model-load evidence, and
aggregate artifacts that inherit flagged source data. It SHALL build a
sanity-check table covering field provenance, non-tautological recomputation,
adversarial regression rows, and methodology completeness, and SHALL decide
which downstream fields remain safe for gates versus which fields are blocked
pending a clean rerun.

The terminal artifact MUST include `adversarial_corrigendum_v1_ready`,
`audited_artifacts`, `flagged_artifact_count`, `adversarial_flag_counts`,
`safe_downstream_fields`, `blocked_downstream_fields`,
`known_false_accept_recovery_preserved`, `live_verifier_evidence_trusted`,
`repair_gate_implication`, `methodology_requirements_for_rerun`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`. It SHOULD
also include sanity-check rows, inherited-flag traceability, tests run,
duration, and source checksums.

`adversarial_corrigendum_v1_ready` shall be true only when all required source
artifacts are readable, at least one flagged source is classified, exact replay
success from Exp 3137 and Exp 3138 is preserved, live verifier evidence is not
trusted when authenticity evidence is missing, and
`repair_gate_implication` is one of `blocked_pending_clean_rerun`,
`exact_recovery_only_no_repair_unlock`, or `clean_rerun_required_before_unlock`.

### SCENARIO-VERIFY-3150: Exact Recovery Survives While Live Evidence Is Blocked

Given the Exp 3139 live verifier rerun is adversarially flagged for
methodology and duration problems, and Exp 3137 plus Exp 3138 report exact
false-accept regression rows blocked by deterministic replay,
When Exp 3150 builds the adversarial evidence corrigendum,
Then it writes the required terminal JSON artifact without live inference,
preserves exact-safe false-accept recovery as a safe downstream gate input,
blocks live verifier lift, live false-accept-rate, verifier gain, and repair
unlock fields pending a clean rerun, records inherited flags for the matrix and
capstone, and emits a machine-readable repair-gate implication that prevents
repair from unlocking from untrusted live-inference evidence.

## Implementation Status (REQ-VERIFY-3150)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3150 | Planned (`python/carnot/verify/adversarial_verifier_evidence_corrigendum_v1.py`) | Planned (`tests/python/test_experiment_3150_adversarial_verifier_evidence_corrigendum_v1.py`) |

### REQ-VERIFY-3151: Live Inference Authenticity Preflight V1

The repository shall provide an Exp 3151 live-inference authenticity preflight
builder that writes
`results/experiment_3151_live_inference_authenticity_preflight_v1.json`
before any clean Exp 3152 live SOTA verifier rerun can claim live model
evidence. The builder MUST read the mandated local SOTA GGUF policy, inspect
the Exp 3123 cache/precondition manifest and the current filesystem paths
without downloading models, and MUST NOT score a verifier panel, execute
repairs, modify `scripts/research_conductor.py`, or use legacy small models as
headline evidence.

The preflight SHALL define the required evidence for a clean live rerun:
selected mandated model IDs, path existence and nonzero size, the exact load
command, model-load wall time, GPU/CPU substrate, transcript hashes, prompt and
completion token counts, `random_seed`, `reproducibility_checksum`, and a
minimum plausible duration requirement. If a mandated local SOTA GGUF is
usable and the local runtime is safe to invoke, it SHALL run at most one
bounded smoke prompt on one mandated model and record only hash/provenance
evidence. If no mandated local SOTA GGUF is usable, CUDA/runtime support is
absent, the smoke call fails, or duration evidence is implausible, it SHALL
write a complete blocked artifact with `preflight_passed=false`,
`live_call_count=0` unless a real smoke transcript was produced,
`headline_claim_allowed=false`, and an actionable `blocked_reason`.

The terminal artifact MUST include
`live_inference_authenticity_preflight_ready`, `model_specs`,
`locally_usable_model_ids`, `selected_model_ids`, `preflight_passed`,
`live_call_count`, `model_load_evidence`, `transcript_hashes`,
`minimum_duration_requirement_s`, `headline_claim_allowed`, `blocked_reason`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`. It SHOULD also
include an Exp 3152 reusable preflight contract, token counts, field-principle
annotations, tests run, duration, source checksums, and the reproducibility
checksum used by the gate.

`preflight_passed` shall be true only when at least one mandated local SOTA
GGUF path exists, a selected mandated model completes the bounded smoke call,
the model-load evidence includes a nonnegative load wall time and exact load
command, transcript hashes and token counts are nonempty, the wall-clock
duration meets the explicit minimum duration requirement, and
`headline_claim_allowed=false` because a smoke preflight alone cannot create a
headline verifier result. `honest_verdict` MUST start with a terminal
`complete:`/`complete_`/`success:`/`success_`/`passed:`/`passed_`/`shipped_`
prefix only for a passed preflight, and with `blocked_` for any failed
precondition.

### SCENARIO-VERIFY-3151: Authenticity Preflight Blocks Unproven Live Claims

Given Exp 3123 exposes mandated local SOTA cache evidence and Exp 3139 was
flagged for live-call claims that lacked coherent model-load evidence,
When Exp 3151 builds the live-inference authenticity preflight,
Then it writes the required terminal JSON artifact without scoring the
verifier panel, records mandated model availability from both the manifest and
actual filesystem paths, attempts at most one bounded mandated-model smoke
call when locally safe, records load-command, load-duration, transcript-hash,
token-count, seed, checksum, and substrate evidence, sets
`headline_claim_allowed=false`, and exposes a reusable Exp 3152 contract that
requires the same authenticity evidence before any clean live SOTA rerun can
headline results.

## Implementation Status (REQ-VERIFY-3151)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3151 | Implemented (`python/carnot/verify/live_inference_authenticity_preflight_v1.py`) | Implemented (`tests/python/test_experiment_3151_live_inference_authenticity_preflight_v1.py`) |

### REQ-VERIFY-3164: Duration-Corrected Live Inference Authenticity Contract V2

The repository shall provide an Exp 3164 duration-corrected live-inference
authenticity contract builder that writes
`results/experiment_3164_duration_corrected_authenticity_contract_v2.json`
without running live model inference, verifier panels, repairs, solvers,
hardware commands, or modifying `scripts/research_conductor.py`.

The builder MUST read Exp 3151 and extract the measured model-load wall time,
generation wall time, token counts, selected model ID, selected local model
path, load command hash, worker code hash, transcript-hash availability, prompt
hashes, random seed, reproducibility checksum, subprocess return codes, and
GPU/CPU substrate evidence. It MUST trace the contract to the prior Exp 3151
duration failure, Exp 3150 adversarial methodology requirements, and Exp 3162
capstone blocker summary. Missing, malformed, or internally inconsistent source
evidence MUST remain explicit rejection criteria rather than being inferred.

The v2 contract SHALL retire the old fixed 60-second duration rule as a hard
gate. Duration plausibility MUST instead be tied to observed work: local model
path existence, successful model-load proof, controlled subprocess return
codes, transcript hashes, prompt hashes, nonzero token counts, seed/checksum
evidence, repeated smoke calls or a token-scaled work budget, and wall-clock
claims supported by command output. The old 60-second floor MAY remain only as
an optional warning for large panels, never as the sole pass/fail criterion for
a one-prompt smoke call.

The terminal artifact MUST include
`duration_corrected_authenticity_contract_v2_ready`,
`old_fixed_duration_rule_retired_as_hard_gate`,
`measured_work_requirements`, `token_scaled_duration_policy`,
`repeated_call_policy`, `required_preflight_fields`,
`fake_evidence_rejection_criteria`, `headline_claim_policy`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`. It SHOULD also
include extracted Exp 3151 measurements, reusable contract blocks for Exp 3165
and Exp 3167, field-principle annotations, tests run, source checksums,
duration, and run date. The readiness flag shall be true only when the fixed
duration rule is not load-bearing, observed-work requirements are present, fake
evidence rejection criteria are explicit, required machine-checkable preflight
fields cover transcripts, prompts, tokens, seed, checksum, model-load evidence,
and controlled return codes, and the inference substrate declares no live model
inference for Exp 3164 itself.

### SCENARIO-VERIFY-3164: Fast Smoke Is Accepted Only With Measured Work Evidence

Given Exp 3151 contains a real local GGUF model load and one smoke call that
was blocked solely by a fixed 60-second floor,
When Exp 3164 builds the duration-corrected authenticity contract,
Then it records the measured Exp 3151 work, retires the fixed duration rule as
a hard gate, defines token-scaled duration and repeated-call policies, names
fake-evidence rejection criteria for missing model loads, transcript hashes,
seed/checksum, impossible throughput, stale transcript reuse, selected
model/path mismatch, and unsupported wall-clock claims, and emits reusable
machine-checkable contract requirements for Exp 3165 and Exp 3167 without
running fresh live inference.

## Implementation Status (REQ-VERIFY-3164)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3164 | Implemented (`python/carnot/verify/duration_corrected_authenticity_contract_v2.py`) | Implemented (`tests/python/test_experiment_3164_duration_corrected_authenticity_contract_v2.py`) |

### REQ-VERIFY-3165: Live SOTA Authenticity Replay V2

The repository shall provide an Exp 3165 duration-corrected local SOTA
authenticity replay builder that writes
`results/experiment_3165_live_sota_authenticity_replay_v2.json` and decides
whether downstream live verifier reruns may proceed under the Exp 3164 v2
contract. The builder MUST read the Exp 3164 contract first, inspect local
HuggingFace cache paths for the three mandated SOTA GGUF model IDs without
downloading models, and MUST NOT score verifier panels, execute repairs,
modify `scripts/research_conductor.py`, or use legacy small models as headline
evidence.

The replay SHALL select only a locally usable mandated SOTA GGUF. If one is
available and the local CUDA/runtime substrate is safe to invoke, it SHALL run
the smallest safe repeated-call smoke required by Exp 3164: at least two
distinct smoke prompts on the selected model, or a predeclared longer
token-scaled single call. It SHALL record load command and command hash, worker
code hash, selected model path, substrate evidence, load wall time, per-call
generation wall times, prompt hashes, response hashes, transcript hashes, token
counts, `random_seed`, and `reproducibility_checksum`. If no mandated model is
usable, runtime preconditions fail, the smoke call fails, repeated-call
controls fail, or any required evidence is missing or internally inconsistent,
it SHALL write a complete blocked artifact rather than a bootstrap-only
artifact.

The terminal artifact MUST include
`live_sota_authenticity_replay_v2_ready`, `model_specs`,
`locally_usable_model_ids`, `selected_model_ids`, `unavailable_model_ids`,
`preflight_passed`, `live_call_count`, `model_load_evidence`,
`prompt_hashes`, `transcript_hashes`, `token_counts`,
`measured_work_policy_passed`, `fake_evidence_rejection_passed`,
`headline_claim_allowed`, `blocked_reason`, `random_seed`,
`reproducibility_checksum`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. `preflight_passed` shall be true only when the completed
artifact satisfies all required v2 evidence fields, the measured-work policy,
the token-scaled duration policy, and fake-evidence rejection criteria.
`headline_claim_allowed` SHALL remain false because a smoke replay only gates a
later live verifier rerun; it is not itself headline verifier evidence.

### SCENARIO-VERIFY-3165: Replay Either Passes With Two Fresh Smoke Calls Or Blocks

Given Exp 3164 exposes a v2 replay contract and the local HuggingFace cache may
contain zero or more mandated SOTA GGUFs,
When Exp 3165 builds the live SOTA authenticity replay artifact,
Then it records the mandated model policy and current cache availability,
attempts the smallest safe repeated-call smoke only for a locally usable
mandated model, records distinct prompt/transcript/response hashes and token
counts for each completed call, applies measured-work and fake-evidence
rejection checks, keeps `headline_claim_allowed=false`, and emits either
`preflight_passed=true` with `live_call_count>=2` or a complete blocked
artifact with `preflight_passed=false` and an actionable `blocked_reason`.

## Implementation Status (REQ-VERIFY-3165)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3165 | Implemented (`python/carnot/verify/live_sota_authenticity_replay_v2.py`) | Implemented (`tests/python/test_experiment_3165_live_sota_authenticity_replay_v2.py`) |

### REQ-VERIFY-3166: Verifier Invariance Token Suspicion Audit V1

The repository shall provide an Exp 3166 controlled-invariance and
token-suspicion audit builder that writes
`results/experiment_3166_verifier_invariance_token_suspicion_audit_v1.json`
from checked-in artifacts only. The builder MUST consume the Exp 3136
false-accept autopsy, Exp 3137 exact-safe accept/abstain/reject contract, Exp
3138 canonical grounding pilot, Exp 3150 adversarial verifier-evidence
corrigendum, Exp 3151 live-inference authenticity preflight, and Exp 3165 live
SOTA authenticity replay. It MUST NOT run fresh live model inference, execute
verifiers, execute repair generation, modify `scripts/research_conductor.py`,
or push commits.

The audit shall collect trusted exact rows from Exp 3136, Exp 3137, and Exp
3138, plus any transcript hashes present in Exp 3165. It shall define
controlled-invariance checks for Force, Remove, shuffled-trace, answer-only,
and trace-only controls. Each control SHALL route transformed rows back to
exact authority checks and SHALL NOT authorize acceptance by itself. The audit
shall distinguish diagnostics that are computable from existing artifacts from
diagnostics blocked on future live token/logprob telemetry.

The audit shall define token-level and first-token suspicion fields for
triage, including token counts, response/transcript hash reuse, first-token
entropy or negative logprob when available, token-family mismatch, and
answer/trace artifact-only signals. These fields may prioritize rows for exact
checking or conservatively block repair promotion when evidence is missing or
suspicious, but they SHALL NOT become acceptance authority. Exact labels,
exact-safe replay decisions, canonical equivalence, solver/test authority, and
monitor-ledger consistency SHALL remain separate acceptance-authority fields.

The terminal artifact MUST include
`verifier_invariance_token_suspicion_audit_ready`,
`controlled_invariance_checks`, `computed_checks`, `blocked_checks`,
`token_suspicion_fields`, `acceptance_authority_fields`,
`diagnostics_allowed_to_gate_repair`, `diagnostics_not_allowed_to_accept`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`. It SHOULD
also include trusted exact-row summaries, transcript-hash inventory,
downstream Exp 3167 policy, field-principle annotations, tests run, source
checksums, duration, and run date.

`verifier_invariance_token_suspicion_audit_ready` shall be true only when the
required checked-in source artifacts are readable, at least one trusted exact
row is collected, all five controlled-invariance check definitions are
present, missing token/logprob telemetry remains visible in
`blocked_checks`, token suspicion fields are explicitly marked as non-authority
triage, acceptance-authority fields are exact-only, the Exp 3167 downstream
policy requires exact checks for acceptance, and the inference substrate
declares no new live model inference.

### SCENARIO-VERIFY-3166: Suspicion Routes Exact Checks But Cannot Accept

Given Exp 3136, Exp 3137, and Exp 3138 expose exact false-accept recovery rows,
Exp 3150 blocks untrusted live verifier evidence pending a clean rerun, and
Exp 3165 may or may not contain transcript hashes,
When Exp 3166 builds the invariance and token-suspicion audit,
Then it writes the required terminal JSON artifact without live inference,
lists Force, Remove, shuffled-trace, answer-only, and trace-only controls,
separates computed exact-row and transcript-hash inventories from proposed
future logprob/token checks, records blocked first-token and token-level
telemetry when unavailable, marks token and trace diagnostics as triage-only,
and exposes an Exp 3167 policy that permits acceptance only from exact
authority fields while allowing suspicion diagnostics to route rows to exact
checks or conservatively block repair promotion.

## Implementation Status (REQ-VERIFY-3166)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3166 | Implemented (`python/carnot/verify/verifier_invariance_token_suspicion_audit_v1.py`) | Implemented (`tests/python/test_experiment_3166_verifier_invariance_token_suspicion_audit_v1.py`) |

### REQ-VERIFY-3167: Clean Live SOTA Verifier Rerun V9 Gate Artifact

The repository shall provide an Exp 3167 clean live SOTA verifier rerun v9
builder that writes `results/experiment_3167_clean_live_sota_verifier_rerun_v9.json`.
The builder MUST read Exp 3165 and Exp 3166 before any live model call. It
MUST also consume the Exp 3136 false-accept autopsy, Exp 3137 exact-safe
accept/abstain contract, Exp 3138 canonical grounding pilot, and Exp 3150
adversarial verifier-evidence corrigendum so the rerun set and gate decision
trace to exact labels rather than prior untrusted verifier metrics.

If Exp 3165 reports `preflight_passed=false`, Exp 3166 is absent or not ready,
or no mandated local SOTA GGUF can be used for the rerun, the builder SHALL NOT
call a model. It SHALL instead write a complete gated-skip artifact with
`clean_live_verifier_rerun_v9_ready=true`, `gated_skip=true`,
`live_call_count=0`, `controlled_invariance_passed=false`,
`false_accept_gate_passed=false`, `flagged_adversarial=false`, and
`headline_claim_allowed=false`. The gated-skip artifact MUST still include the
mandated model policy, selected and unavailable model IDs, upstream model-load
evidence, empty prompt/transcript hashes, zero token counts, exact-row count,
regression-row inclusion status, deterministic seed, reproducibility checksum,
source-artifact provenance, and explicit inference-substrate metadata.

When all preconditions pass, the builder SHALL build the rerun set from known
false-accept regression rows plus balanced exact fixtures across arithmetic,
SMT, satisfiable-drift, contradiction, and fragment-code families; use only
the mandated local SOTA GGUF model IDs for headline evidence; record model-load
evidence, prompt hashes, transcript hashes, token counts, random seed, and
reproducibility checksum; apply exact-safe, canonical-grounding, and
controlled-invariance diagnostics; and compute `false_accept_rate`,
`false_reject_rate`, `abstention_rate`, `verifier_gain_delta`,
`false_accept_gate_passed`, `flagged_adversarial`, and
`headline_claim_allowed` from the exact authority rows. Legacy small models MAY
be recorded only as CPU smoke tests and SHALL NOT populate headline fields.

The terminal artifact MUST include `clean_live_verifier_rerun_v9_ready`,
`gated_skip`, `gated_skip_reason`, `model_specs`, `selected_model_ids`,
`unavailable_model_ids`, `live_call_count`, `model_load_evidence`,
`prompt_hashes`, `transcript_hashes`, `token_counts`,
`exact_ground_truth_count`, `regression_rows_included`,
`controlled_invariance_passed`, `false_accept_rate`, `false_reject_rate`,
`abstention_rate`, `verifier_gain_delta`, `false_accept_gate_passed`,
`flagged_adversarial`, `headline_claim_allowed`, `random_seed`,
`reproducibility_checksum`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. It SHOULD also include planned rerun-set family counts,
precondition checks, field-principle annotations, source checksums, tests run,
duration, run date, and an explicit statement that ops/status, changelog, and
traceability reconciliation are left to the conductor when the task prompt says
so.

### SCENARIO-VERIFY-3167: Failed Preconditions Produce Complete Gated Skip

Given Exp 3165 reports `preflight_passed=false` and Exp 3166 reports a ready
invariance/token-suspicion audit,
When Exp 3167 builds the clean live SOTA verifier rerun v9 artifact,
Then it reads the exact false-accept and canonical-grounding source artifacts,
constructs a planned rerun-set summary that includes the known false-accept
regression row IDs and balanced exact-family buckets, does not call a model,
emits `live_call_count=0`, empty prompt and transcript hashes, zero token
counts, `gated_skip=true` with an actionable reason,
`false_accept_gate_passed=false`, `flagged_adversarial=false`,
`headline_claim_allowed=false`, and an `honest_verdict` that starts with
`complete:` so downstream matrix tooling can distinguish a completed gated skip
from a missing rerun artifact.

## Implementation Status (REQ-VERIFY-3167)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3167 | Implemented (`python/carnot/verify/clean_live_sota_verifier_rerun_v9.py`) | Implemented (`tests/python/test_experiment_3167_clean_live_sota_verifier_rerun_v9.py`) |

### REQ-VERIFY-3168: Conservative Repair Gate Decision V3

The repository shall provide an Exp 3168 conservative repair-gate decision
builder that writes `results/experiment_3168_repair_gate_decision_v3.json`
from checked-in artifacts only. The builder MUST consume the Exp 3167 clean
live SOTA verifier rerun v9 artifact when present, plus the Exp 3166
invariance/token-suspicion audit, Exp 3137 exact-safe accept/abstain contract,
Exp 3138 canonical grounding pilot, and the prior Exp 3153 and Exp 3140 repair
gate decisions for provenance. It MUST NOT call live models, execute repair
generation, modify `scripts/research_conductor.py`, push commits, or treat a
missing clean rerun artifact as an implicit gate result.

If the clean v9 verifier artifact is missing, the builder SHALL write a
complete blocked decision with `repair_gate_state=blocked_missing_clean_rerun`
and actionable blockers. If the clean v9 artifact is present, the gate SHALL
permit `repair_gate_state=unblocked` only when all of these criteria hold:
`clean_live_verifier_rerun_v9_ready=true`, `gated_skip=false`,
`false_accept_rate <= 0.10`, `false_accept_gate_passed=true`,
`flagged_adversarial=false`, `regression_rows_included=true`, exact labels are
present through exact-authority rows, `controlled_invariance_passed=true`, and
`headline_claim_allowed=true`. Any failed criterion SHALL block repair and
SHALL keep `selected_repair_rows=[]`.

The decision state SHALL be exactly one of `unblocked`,
`blocked_false_accept`, `blocked_flagged_verifier`,
`blocked_missing_live_model`, `blocked_missing_exact_labels`,
`blocked_invariance_failure`, `blocked_gated_skip`,
`blocked_missing_clean_rerun`, or `blocked_other`. The terminal artifact MUST
include `repair_gate_decision_v3_ready`, `repair_gate_state`,
`clean_rerun_artifact_present`, `false_accept_rate`,
`false_accept_gate_passed`, `flagged_adversarial`,
`controlled_invariance_passed`, `exact_authority_ready`,
`selected_repair_rows`, `repair_blockers`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`.

`repair_gate_decision_v3_ready` shall be true when the artifact reaches one of
the allowed terminal gate states with source provenance, no live inference, and
an honest verdict. For `repair_gate_state=unblocked`, `honest_verdict` MUST
start with a terminal success prefix and `selected_repair_rows` MUST list the
bounded repair denominator with exact-authority constraints. For blocked
states, `repair_blockers` MUST be non-empty and list the exact failed criteria.

### SCENARIO-VERIFY-3168: Tainted Or Skipped Clean Rerun Blocks Repair

Given Exp 3167 has written a clean-live-verifier v9 artifact or that artifact
is missing,
When Exp 3168 builds the repair-gate decision,
Then it writes a complete terminal JSON decision without live inference,
copies the false-accept, adversarial-flag, exact-authority, invariance, gated
skip, regression-row, and headline-claim gate results into machine-readable
fields, selects repair rows only for a fully unblocked clean rerun, and
otherwise reports exact blockers while avoiding model and repair execution.

## Implementation Status (REQ-VERIFY-3168)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3168 | Implemented (`python/carnot/verify/repair_gate_decision_v3.py`) | Implemented (`tests/python/test_experiment_3168_repair_gate_decision_v3.py`) |

### REQ-VERIFY-3169: Repair Ladder Materializer V4

The repository shall provide an Exp 3169 repair-ladder materializer that
writes `results/experiment_3169_repair_ladder_materializer_v4.json` whether
repair is allowed or blocked. The builder MUST read
`results/experiment_3168_repair_gate_decision_v3.json` before any repair
generation and MUST NOT call a model unless Exp 3168 reports
`repair_gate_state=unblocked`, selected repair rows are present, and at least
one mandated local SOTA GGUF model is usable. The mandated model policy is
exactly `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; legacy small models MAY be recorded only as
CPU smoke evidence and SHALL NOT support headline repair claims.

If Exp 3168 is missing, blocked, gated-skipped, has no selected repair rows, or
has no usable mandated model, the materializer SHALL write a complete
gated-skip artifact with `repair_ladder_materializer_v4_ready=true`,
`gated_skip=true`, `live_call_count=0`, `repair_attempt_count=0`,
`exact_authority_accept_count=0`, `repair_success_delta=0.0`,
`false_repair_accept_rate=0.0`, `intent_preservation_rate=0.0`, and
`headline_repair_claim_allowed=false`. The skip reason SHALL include the
actionable gate state or runtime blocker so downstream matrix tooling can
distinguish a completed skip from a missing repair ladder.

When Exp 3168 unblocks repair and a mandated local SOTA GGUF is usable, the
materializer SHALL run only bounded repair attempts over the selected repair
rows and SHALL score candidates with exact authority evidence, canonical
grounding, controlled-invariance checks, and monitor replay before accepting a
repair. It SHALL NOT accept a candidate based on model confidence or schema
validity alone. The terminal artifact MUST include
`repair_ladder_materializer_v4_ready`, `gated_skip`, `gated_skip_reason`,
`model_specs`, `selected_model_ids`, `live_call_count`,
`selected_repair_rows`, `repair_attempt_count`,
`exact_authority_accept_count`, `repair_success_delta`,
`false_repair_accept_rate`, `intent_preservation_rate`,
`headline_repair_claim_allowed`, `source_artifacts`, `inference_substrate`,
and `honest_verdict`.

`headline_repair_claim_allowed` shall be true only when the gate is unblocked,
repair attempts were actually run on mandated local SOTA GGUF evidence, at
least one candidate is accepted by exact authority, false repair accepts are
zero, and every accepted repair preserves intent.

### SCENARIO-VERIFY-3169: Blocked Gate Produces Full No-Call Repair Ladder

Given Exp 3168 has written a repair-gate decision or that artifact is missing,
When Exp 3169 materializes the repair ladder,
Then it first reads the gate decision, writes a complete terminal JSON artifact
in both run and skip paths, carries forward the selected repair denominator,
records zero live calls and no headline claim whenever the gate is blocked, and
executes repair generation only for an unblocked gate with usable mandated
local SOTA GGUF model evidence and exact-authority selected rows.

## Implementation Status (REQ-VERIFY-3169)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3169 | Implemented (`python/carnot/verify/repair_ladder_materializer_v4.py`) | Implemented (`tests/python/test_experiment_3169_repair_ladder_materializer_v4.py`) |

### REQ-VERIFY-3170: Counterexample-Certificate Repair Pilot V2

The repository shall provide an Exp 3170 counterexample-certificate repair
pilot that writes
`results/experiment_3170_counterexample_certificate_repair_pilot_v2.json`
from exact checked-in artifacts only. The builder MUST select a tiny explicit
set of exact false-accept, satisfiable-drift, and fragment-code rows with known
counterexamples, satisfying assignments, solver certificates, or parser
certificates. It MUST NOT invoke live LLM inference, run repair generation,
modify `scripts/research_conductor.py`, push commits, or accept any repair
candidate without exact-authority evidence.

For every selected row, the pilot SHALL construct a replayable certificate
record containing the row id, row type, exact label, violated constraint or
positive-anchor invariant, minimal failing assignment or trace when one exists,
expected corrected invariant, exact verifier to rerun, solver authority,
minimal correction set or unsat core when available, and prior repair scoring
status. Satisfiable-drift rows SHALL be represented honestly as positive
anchors and SHALL keep missing failing-assignment evidence visible rather than
fabricating a counterexample.

When Exp 3169 exposes prior repair attempts, the pilot SHALL score only those
attempts whose row id matches a certificate and SHALL count an exact accept only
when the candidate was accepted and its repaired label matches the certificate
exact label. When no prior repair candidate exists, the pilot SHALL still write
the certificate package with `prior_repair_candidates_scored=0` and
`repair_call_required_for_next_step=true`.

The pilot SHALL record BEAVER-style bounded-frontier fields from Exp 3125 when
present, including bound width, explored mass, viable prefix counts, pruned
prefix counts, and constraint families. Any unavailable bound fields, live
logprob fields, missing source artifacts, or intentionally absent
satisfiable-drift failing assignments SHALL be listed in
`unavailable_certificate_fields`.

The terminal artifact MUST include
`counterexample_certificate_repair_pilot_v2_ready`, `exact_row_count`,
`counterexample_count`, `certificate_records`, `bounded_frontier_records`,
`unavailable_certificate_fields`, `prior_repair_candidates_scored`,
`exact_accept_count`, `repair_call_required_for_next_step`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.
`honest_verdict` MUST begin with a terminal success prefix when the package is
complete and MUST report that live repair is still required unless all selected
certificates have exact accepted repairs.

### SCENARIO-VERIFY-3170: Exact Counterexamples Become A Repair Package

Given Exp 3111, Exp 3125, Exp 3136, Exp 3137, Exp 3138, Exp 3168, and Exp 3169
artifacts are available or explicitly absent,
When Exp 3170 builds the counterexample-certificate repair pilot,
Then it writes a complete terminal JSON artifact without live model inference,
selects the exact false-accept, satisfiable-drift, and fragment-code pilot
rows, records replayable certificate records and bounded-frontier summaries,
scores any available prior repair candidates against exact labels only, and
keeps missing certificate or frontier fields visible for future repair calls.

## Implementation Status (REQ-VERIFY-3170)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3170 | Planned (`python/carnot/verify/counterexample_certificate_repair_pilot_v2.py`) | Planned (`tests/python/test_experiment_3170_counterexample_certificate_repair_pilot_v2.py`) |

### REQ-VERIFY-3073: EBT/ARM-EBM Adapter Feasibility Audit

The repository shall provide an Exp 3073 deterministic architecture audit that
writes `results/experiment_3073_ebt_arm_ebm_adapter_feasibility_audit_v1.json`
without implementing, wiring, or claiming a production EBT/ARM adapter.

The audit shall inspect local architecture surfaces that could host
EBT-style latent reasoning or ARM-EBM-style recurrent/logprob energy
refinement, including the core energy protocol, the verify-repair pipeline,
ARM logprob energy extraction, EBT reasoning bridges, gradient refinement, and
reasoning-energy embedding paths. It shall compare those surfaces against
current code and specs, separate near-term adapter opportunities from
paper-context-only references, and explicitly list the future prerequisites for
data shape, energy objective, verifier interface, sampling path, evaluation
metric, and rollback/claim boundaries.

The terminal artifact MUST include `ebt_arm_adapter_feasibility_ready`,
`ebt_arm_adapter_feasible`, `adapter_surface`, `required_prerequisites`,
`blockers`, `recommended_next_experiment`, `source_refs`,
`inference_substrate`, and `honest_verdict`. It shall also record enough
bounded context to prevent unsupported implementation claims, including
whether an adapter implementation is claimed, whether the audit ran live model
inference, and which references are only theory or paper context.

`ebt_arm_adapter_feasibility_ready` shall be true only when every required
field is populated, at least four local adapter surfaces map to existing
repository files or specs, all six prerequisite categories are present,
paper-context-only references are separated from near-term adapter
opportunities, `inference_substrate` declares no live model inference, and
`honest_verdict` starts with a terminal success prefix. `ebt_arm_adapter_feasible`
may be true only for a concrete future path with explicit local prerequisites;
otherwise the audit must set a bounded theory-context-only state instead of a
feasibility claim.

### SCENARIO-VERIFY-3073: Feasibility Is Separated From Adapter Implementation

Given Carnot contains EBM energy protocols, verifier/repair surfaces,
ARM logprob telemetry code, EBT bridge prototypes, and EBT/ARM literature
references,
When the Exp 3073 audit runs,
Then it writes the terminal JSON artifact with matrix-v21-consumable fields,
maps each adapter surface to local code or specs, lists exact prerequisites
for any future implementation, marks external EBT, ARM-EBM, LoopUS, and Kona
references as bounded context unless local code and tests support the claim,
sets `adapter_implementation_claimed=false`, declares no live model inference,
and uses an `honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-VERIFY-3073)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3073 | Implemented (`python/carnot/eval/ebt_arm_ebm_adapter_feasibility_audit_v1.py`) | Implemented (`tests/python/test_experiment_3073_ebt_arm_ebm_adapter_feasibility_audit.py`) |

### REQ-VERIFY-3091: EBT/ARM Sidecar Adapter Schema Prototype

The repository shall provide an Exp 3091 deterministic sidecar-only prototype
for EBT/ARM adapter data. The prototype MUST define a canonical JSON schema
for cached sidecar rows containing a candidate, constraints, deterministic
energy terms, verifier feedback, confidence and abstention metadata, and exact
label references. The prototype MUST NOT integrate with live model inference,
train an EBT/ARM model, mutate model weights, claim benchmark speedup, or claim
hardware acceleration.

The repository shall also provide an inspectable replay scorer that consumes
cached sidecar rows and computes deterministic energy components from stored
fields only. The scorer may use explicit constraints, stored token logprobs,
stored verifier feedback, confidence metadata, and exact label references; it
MUST declare no live LLM inference, no model weight loading, no generation, and
no dependency on live model weights.

The terminal artifact MUST be written to
`results/experiment_3091_ebt_arm_sidecar_adapter_schema_prototype_v1.json` and
include `adapter_schema_ready`, `sidecar_replay_scorer_ready`, `schema_path`,
`replay_scorer_path`, `tests_added_or_reused`, `implementation_claim_boundary`,
`no_weight_update_claim`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`.

`adapter_schema_ready` may be true only when the schema path exists and covers
the required sidecar row fields. `sidecar_replay_scorer_ready` may be true only
when deterministic replay examples validate the same score across repeated
runs and the artifact records the no-live-inference boundary.

### SCENARIO-VERIFY-3091: Cached Sidecar Rows Replay Without Live Weights

Given the Exp 3073 feasibility audit found local EBT/ARM adapter surfaces but
did not claim an implementation,
When the Exp 3091 prototype validates fixture sidecar rows and replays their
energy components,
Then the terminal artifact names the concrete schema and scorer paths, records
the fixture tests, reports no live model inference or model-weight loading,
keeps EBT/ARM training and runtime integration out of scope, and uses an
`honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-VERIFY-3091)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3091 | Implemented (`python/carnot/inference/ebt_arm_sidecar_adapter.py`; `python/carnot/eval/ebt_arm_sidecar_adapter_schema_prototype_v1.py`; `python/carnot/schemas/ebt_arm_sidecar_adapter_v1.json`) | Implemented (`tests/python/test_experiment_3091_ebt_arm_sidecar_adapter_schema_prototype.py`) |

### REQ-VERIFY-3117: EBT/ARM Sidecar Score Correlation Boundary v3

The repository shall provide an Exp 3117 deterministic sidecar score
correlation diagnostic that writes
`results/experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3.json`
from checked-in exact fixture artifacts only. The diagnostic MUST NOT invoke
live model inference, integrate sidecar scores into a live model path, mutate
model weights, train an EBT/ARM model, or claim benchmark speedup.

The diagnostic shall load the Exp 3097 stratified exact-fixture manifest and
build replay-compatible sidecar records for accepted, rejected, and repairable
cases. It shall compute existing sidecar replay scores and label-blind feature
summaries, then report rank correlation, calibration bins, separability, and
failure cases against exact fixture labels. Any score that uses exact label
references shall be named as label-aware replay evidence and shall not be
promoted as a live pre-label scoring claim.

The terminal artifact MUST include
`sidecar_score_correlation_boundary_v3_ready`, `exact_fixture_count`,
`score_correlation_summary`, `calibration_summary`, `failure_cases`,
`no_live_model_integration_claim`, `no_weight_update_claim`,
`no_speedup_claim`, `tests_run`, `source_artifacts`, `inference_substrate`,
and `honest_verdict`.

`sidecar_score_correlation_boundary_v3_ready` may be true only when at least
48 exact fixture rows are loaded, accepted/rejected/repairable counts are
non-zero, finite rank-correlation and separability summaries are present,
calibration bins account for every scored row, failure-case accounting is
explicit, the no-live/no-training/no-speedup boundary is recorded, and
`honest_verdict` starts with a terminal success prefix.

### SCENARIO-VERIFY-3117: Sidecar Scores Stay Diagnostic Against Exact Fixtures

Given the Exp 3091 sidecar schema/replay scorer and the Exp 3097 exact fixture
protocol are available,
When the Exp 3117 diagnostic builds replay-compatible sidecar rows from the
checked-in exact fixture manifest,
Then it scores accepted, rejected, and repairable cases, reports rank
correlation/calibration/separability/failure diagnostics against exact labels,
records that label-aware replay is not a live pre-label score, declares no live
model integration, no model-weight update, and no speedup claim, and uses an
`honest_verdict` beginning with `complete:`.

## Implementation Status (REQ-VERIFY-3117)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3117 | Implemented (`python/carnot/eval/ebt_arm_sidecar_score_correlation_boundary_v3.py`) | Implemented (`tests/python/test_experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3.py`) |

### REQ-VERIFY-3130: ARM/EBT Energy-Budget Sidecar Diagnostic v2

The repository shall provide an Exp 3130 ARM/EBT energy-budget sidecar
diagnostic that writes
`results/experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.json`
from checked-in exact fixture outcomes, the Exp 3123 mandated-SOTA cache
precondition manifest, the Exp 3117 sidecar correlation rows, and any available
Exp 3124 local mandated-SOTA trace rows. The diagnostic MUST NOT call a live
model, integrate the sidecar into generation, train an EBT/ARM model, mutate
model weights, claim architecture improvement, or modify
`scripts/research_conductor.py`.

The diagnostic shall separate deterministic constraint penalties from
learned/proxy quality scores and uncertainty estimates. It shall summarize
prefix and final energy proxies, exact fixture verdicts, correlation with
reject/repair labels, approximation gaps between proxy energies and exact
authority, model-identity shortcuts/confounds, and fixture-family separation.
When no mandated model or usable trace source exists, it shall still write a
complete blocked diagnostic artifact with `live_call_count=0` and
`live_integration=false`.

The terminal artifact MUST include
`arm_ebt_energy_budget_sidecar_v2_ready`, `model_specs`,
`selected_model_ids`, `live_call_count`, `exact_fixture_count`,
`deterministic_constraint_penalty_summary`,
`learned_quality_proxy_summary`, `uncertainty_summary`,
`approximation_gap_summary`, `model_identity_confound_audit`,
`correlation_metrics`, `live_integration`, `integration_blockers`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`.

`arm_ebt_energy_budget_sidecar_v2_ready` may be true only when exact fixture
labels are present, proxy/penalty/uncertainty summaries are finite, correlation
metrics are computed from exact labels, model provenance is auditable, live
integration remains false, and the artifact names concrete blockers for any
future architecture integration. It MUST NOT become true by inferring live
generation from cache presence alone.

### SCENARIO-VERIFY-3130: Energy-Budget Sidecar Separates Proxies From Exact Authority

Given Exp 3123, Exp 3117, and optionally Exp 3124 are present,
When Exp 3130 builds the ARM/EBT energy-budget sidecar diagnostic,
Then it writes the terminal JSON artifact, consumes only checked-in artifacts,
reports exact fixture counts and selected mandated model IDs, separates
deterministic constraint penalties from quality proxies and uncertainty,
quantifies correlations and approximation gaps against exact reject/repair
labels, audits model-identity confounds, leaves `live_integration=false`, and
uses an `honest_verdict` beginning with `complete:` when the diagnostic is
complete or `blocked_` when mandated preconditions are missing.

## Implementation Status (REQ-VERIFY-3130)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3130 | Implemented (`python/carnot/eval/arm_ebt_energy_budget_sidecar_diagnostic_v2.py`) | Implemented (`tests/python/test_experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.py`) |

### REQ-VERIFY-3144: EBT/ARM False-Accept Calibration Boundary v3

The repository shall provide an Exp 3144 offline calibration generator that
writes
`results/experiment_3144_ebt_arm_false_accept_calibration_boundary_v3.json`
from checked-in `.291` evidence only. It MUST load the Exp 3130 ARM/EBT
energy-budget sidecar diagnostic, the Exp 3124 live verifier rows and exact
labels, the Exp 3136 false-accept row IDs and autopsy rows, and the Exp 3117
row-level sidecar diagnostics needed to reconstruct deterministic constraint
penalty, proxy quality, uncertainty, and approximation-gap fields per live
row. It MUST NOT run live model inference, train an EBT/ARM model, integrate
the sidecar into generation, mutate model weights, run verifier scoring,
generate repairs, push commits, or modify `scripts/research_conductor.py`.

The calibration MUST compare false-accept rows against non-false-accept live
rows for deterministic constraint penalties, sidecar energy proxies, proxy
quality, uncertainty, and approximation gaps to exact reject/repair authority.
It MUST quantify separation metrics, identify whether any sidecar field could
serve as an abstention feature under exact-safe contract rules, and explain
which required contract conditions are still missing before that feature can
be used online. It MUST audit model-identity confounds, including single-model
trace coverage, mandated local model policy visibility, and whether selected
model IDs or model hashes can shortcut the row classification.

The terminal artifact MUST include
`ebt_arm_false_accept_calibration_v3_ready`, `model_specs`,
`selected_model_ids`, `live_call_count`, `false_accept_rows_evaluated`,
`abstention_feature_candidates`, `false_accept_separation_metrics`,
`approximation_gap_summary`, `model_identity_confound_audit`,
`live_integration`, `integration_blockers`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. `live_integration` MUST remain
false unless real generation-path sidecar integration code and tests are
implemented in the same change.

`ebt_arm_false_accept_calibration_v3_ready` may be true only when the
checked-in source artifacts are present, at least one false-accept row is
joined to sidecar diagnostics, every false-accept row named by Exp 3136 is
evaluated, separation metrics are finite, the mandated model policy is visible
in `model_specs`, `live_call_count` is copied from upstream evidence rather
than inferred, `live_integration=false`, and concrete integration blockers are
reported.

### SCENARIO-VERIFY-3144: Sidecar Fields Are Calibrated Against .291 False Accepts

Given Exp 3124 contains live verifier rows and exact labels, Exp 3130 contains
the ARM/EBT sidecar diagnostic boundary, Exp 3136 identifies the `.291`
false-accept row IDs, and Exp 3117 contains row-level sidecar diagnostics,
When the Exp 3144 calibration generator runs,
Then it writes the terminal JSON artifact, consumes only checked-in artifacts,
evaluates the named false accepts directly, compares sidecar fields on false
accepts versus non-false-accept live rows, reports abstention feature
candidates with exact-safe contract limitations, audits model-identity
confounds and missing integration tests, leaves `live_integration=false`, and
uses an `honest_verdict` beginning with `complete:` when the offline
calibration is complete or `blocked_` when required source evidence is missing.

## Implementation Status (REQ-VERIFY-3144)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3144 | Implemented (`python/carnot/eval/ebt_arm_false_accept_calibration_boundary_v3.py`) | Implemented (`tests/python/test_experiment_3144_ebt_arm_false_accept_calibration_boundary_v3.py`) |

### REQ-VERIFY-3158: EBCN Energy Sidecar Calibration v1

The repository shall provide an Exp 3158 offline EBCN-inspired energy sidecar
calibration generator that writes
`results/experiment_3158_ebcn_energy_sidecar_calibration_v1.json` from checked-in
exact-labeled evidence only. The generator MUST consume the Exp 3144 sidecar
calibration rows, Exp 3136 known false-accept autopsy rows, Exp 3137 exact-safe
accept/abstain contract rows, Exp 3138 canonical grounding rows, and local
research/reference files. It MUST NOT run live model inference, train an EBCN
or EBT model, integrate the sidecar into generation, mutate model weights, push
commits, or modify `scripts/research_conductor.py`.

The calibration MUST select exact-labeled rows covering known false accepts,
clean accepts, contradictions, and satisfiable drift cases. It MUST define a
bounded scalar energy from label-blind sidecar fields, define per-row
violation-localization fields from monitor ledger positions or a deterministic
diagnostic fallback, and compute whether scalar energy separates known false
accepts from clean exact rows without using exact labels, false-accept flags,
or approximation-gap labels as scalar-energy inputs.

The terminal artifact MUST include `ebcn_energy_sidecar_calibration_v1_ready`,
`exact_labeled_row_count`, `known_false_accept_rows_scored`,
`scalar_energy_auc`, `violation_localization_coverage`,
`scale_compatibility_notes`, `live_integration_claim_allowed`,
`residual_blockers`, `tests_run`, `source_artifacts`, `inference_substrate`,
and `honest_verdict`. `live_integration_claim_allowed` MUST remain false unless
real generation-path sidecar integration code and tests are implemented in the
same change.

`ebcn_energy_sidecar_calibration_v1_ready` may be true only when required
checked-in source artifacts are present, at least one exact-labeled row is
scored, every Exp 3136 known false-accept row is scored, scalar-energy AUC is
finite against clean exact rows, violation localization covers every row that
requires localization, no label-aware field is used in scalar energy, the
inference substrate declares zero new live model calls, and residual blockers
keep live verifier integration disallowed.

### SCENARIO-VERIFY-3158: Bounded EBCN Sidecar Scores Exact False Accepts

Given Exp 3144 contains row-level sidecar energy fields, Exp 3136 identifies
known false accepts with exact labels and monitor events, Exp 3137 contains the
exact-safe accept/abstain contract, and Exp 3138 contains canonical grounding
for the false-accept regressions,
When the Exp 3158 calibration generator runs,
Then it writes the terminal JSON artifact, consumes only checked-in artifacts,
scores exact-labeled rows spanning false accepts, clean accepts,
contradictions, and satisfiable drift, reports a bounded scalar energy AUC
against clean exact rows, reports violation-localization coverage, records
scale-compatibility assumptions for branch-composed energies, leaves
`live_integration_claim_allowed=false`, and uses an `honest_verdict` beginning
with `complete:` when diagnostic evidence is complete or `blocked_` when
required source evidence is missing.

## Implementation Status (REQ-VERIFY-3158)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3158 | Implemented (`python/carnot/eval/ebcn_energy_sidecar_calibration_v1.py`) | Implemented (`tests/python/test_experiment_3158_ebcn_energy_sidecar_calibration_v1.py`) |

### REQ-VERIFY-3173: EBCN/KAN Bounded Diagnostic Expansion V2

The repository shall provide an Exp 3173 bounded diagnostic panel that expands
the exact-label EBCN/KAN evidence for matrix v28 without making live
integration or deployed-verifier claims. The builder MUST read only checked-in
artifacts, including the Exp 3136 known false-accept autopsy, Exp 3137
exact-safe accept/abstain contract, Exp 3138 canonical grounding pilot, Exp
3158 EBCN energy sidecar calibration, Exp 3159 KAN proof-carrying monitor
expansion, and any clean live verifier rerun artifact when present. It MUST
NOT run live model inference, train or mutate model weights, integrate EBCN or
KAN evidence into generation, push commits, or modify
`scripts/research_conductor.py`.

The panel MUST collect exact-labeled rows from known false accepts,
exact-safe contract replay rows, canonical grounding rows, EBCN sidecar rows,
KAN monitor records, and present clean-rerun rows. It MUST deduplicate rows by
stable row ID while preserving per-source provenance, score or replay EBCN
energy diagnostics and KAN proof-carrying monitor records on the expanded exact
set, report localization quality, false-accept row separation, and
monitor-record coverage, and name all blockers that prevent promotion beyond a
bounded diagnostic.

The terminal artifact MUST be written to
`results/experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.json` and
include `ebcn_kan_bounded_diagnostic_expansion_v2_ready`,
`exact_labeled_row_count`, `known_false_accept_rows_scored`,
`ebcn_localization_metrics`, `kan_monitor_record_count`,
`deployed_verifier_claim_allowed`, `live_integration_claim_allowed`,
`promotion_blockers`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. `deployed_verifier_claim_allowed` and
`live_integration_claim_allowed` MUST remain false unless a real deployed
verifier and live generation-path integration are implemented and tested in the
same change.

`ebcn_kan_bounded_diagnostic_expansion_v2_ready` may be true only when required
source artifacts are present, at least one exact-labeled row is collected, every
known false-accept row from Exp 3136 is scored, EBCN localization metrics are
finite, KAN monitor-record coverage is countable, inference substrate declares
zero new live model calls, and promotion blockers explicitly include the tiny
denominator, missing live integration, and missing deployed verifier.

### SCENARIO-VERIFY-3173: Matrix v28 Receives A Bounded EBCN/KAN Panel

Given the checked-in Exp 3136, Exp 3137, Exp 3138, Exp 3158, and Exp 3159
artifacts are present, and a clean live verifier rerun artifact may or may not
be present,
When the Exp 3173 diagnostic builder runs,
Then it writes the terminal JSON artifact, aggregates exact-labeled rows with
source provenance, includes every known false-accept row, reports EBCN
localization and false-accept separation metrics, reports KAN monitor-record
coverage, includes clean-rerun status without treating a gated skip as live
evidence, keeps `deployed_verifier_claim_allowed=false` and
`live_integration_claim_allowed=false`, and uses an `honest_verdict` beginning
with `complete:` or `complete_` when bounded diagnostic evidence is complete.

## Implementation Status (REQ-VERIFY-3173)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3173 | Implemented (`python/carnot/eval/ebcn_kan_bounded_diagnostic_expansion_v2.py`) | Implemented (`tests/python/test_experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.py`) |

### REQ-VERIFY-3178: Receipt-Backed Live Authenticity Contract V3

The repository shall provide an Exp 3178 receipt-backed live-inference
authenticity contract builder that writes
`results/experiment_3178_receipt_backed_authenticity_contract_v3.json`
without making live model calls, executing verifier panels, running repairs,
downloading models, pushing commits, or modifying `scripts/research_conductor.py`.

The builder MUST preserve valid Exp 3164 v2 measured-work requirements,
including local model path evidence, model-load proof, controlled subprocess
return codes, prompt and transcript hashes, token counts, random seed,
reproducibility checksum or equivalent command hash evidence, and wall-clock
claims supported by command output. It MUST read Exp 3165 and record the exact
`.294` replay blocker reason rather than collapsing it into a generic failed
preflight. It MUST also cite Exp 3167 and Exp 3176 so downstream clean-rerun
tasks can distinguish a completed gated skip from missing evidence.

The v3 contract SHALL define substrate classes named exactly
`model_cache_missing`, `loader_missing`, `cuda_unavailable`,
`cuda_available_unhealthy`, `cpu_fallback_receipt_only`, and
`full_local_sota_receipt`. The substrate classification policy MUST separate
HuggingFace cache availability, loader/import availability, CUDA discovery,
CUDA health, CPU fallback admissibility, receipt completeness, and headline
eligibility. CPU fallback MAY prove receipt wiring for a bounded smoke, but it
SHALL NOT unlock headline verifier benchmark claims or clean verifier reruns.

The required receipt field policy SHALL include `selected_model_id`,
`model_path`, `model_file_hash` when cheap, `loader_name`, `substrate_used`,
`prompt_hashes`, `transcript_hashes`, `token_counts`, `random_seed`,
`wall_clock_s`, `command_hash`, `subprocess_return_code`, `stderr_tail`,
`throughput_plausibility`, and `replay_count`. Fake-evidence rejection criteria
MUST explicitly cover missing model/cache proof, missing loader proof, missing
CUDA health for headline claims, CPU-only evidence promoted as headline
evidence, missing transcript or prompt hashes, stale transcript reuse, missing
token counts or seeds, unsupported wall-clock claims, uncontrolled subprocess
return codes, impossible throughput, and one-prompt smoke evidence promoted as
a benchmark.

The terminal artifact MUST include
`receipt_backed_authenticity_contract_v3_ready`,
`inherited_v2_contract_fields`, `substrate_classification_policy`,
`required_receipt_fields`, `cpu_fallback_policy`,
`fake_evidence_rejection_criteria`, `clean_rerun_unlock_requirements`,
`headline_claim_policy`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. It SHOULD also include the extracted Exp 3164 measured-work
requirements, the exact Exp 3165 blocker reason, field-principle annotations,
tests run, source checksums, duration, and run date. The readiness flag SHALL
be true only when all required policy blocks are present, the v2 evidence rules
are inherited, the six substrate classes are defined, CPU fallback remains
non-headline, clean rerun unlocks require full local SOTA receipts plus exact
authority scoring and controlled invariance, and the inference substrate
declares no live model inference for Exp 3178 itself.

### SCENARIO-VERIFY-3178: V3 Contract Separates Substrate Blockers From Headline Evidence

Given Exp 3164 contains the duration-corrected v2 measured-work contract, Exp
3165 reports a blocked local SOTA replay, Exp 3167 reports a clean verifier
gated skip, and Exp 3176 closes `.294` with paper readiness still blocked,
When Exp 3178 builds the receipt-backed authenticity contract,
Then it writes the terminal JSON artifact without live inference, preserves the
v2 measured-work rules, records the exact Exp 3165 blocker reason, defines the
six required substrate classes, names all required receipt fields, allows CPU
fallback only as receipt-wiring evidence, refuses headline claims from CPU
fallback or one-prompt smoke evidence, and requires full local SOTA receipts,
exact-authority scoring, and controlled invariance before any clean verifier
rerun can become headline-eligible.

## Implementation Status (REQ-VERIFY-3178)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3178 | Implemented (`python/carnot/verify/receipt_backed_authenticity_contract_v3.py`) | Implemented (`tests/python/test_experiment_3178_receipt_backed_authenticity_contract_v3.py`) |

### REQ-VERIFY-3179: Local SOTA Receipt Smoke V3

The repository shall provide an Exp 3179 local SOTA GGUF receipt-smoke
builder that writes `results/experiment_3179_local_sota_receipt_smoke_v3.json`
under the Exp 3178 v3 receipt contract. The builder MUST load the checked-in
Exp 3178 contract before attempting model work, MUST inspect
`cached_sota_pair()` and the HuggingFace cache state for all three mandated
GGUF IDs (`unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`), MUST check `llama_cpp` loader import
availability, MUST check torch CUDA availability when torch is importable, and
MUST NOT download models, use legacy small models in headline fields, push
commits, or modify `scripts/research_conductor.py`.

The builder SHALL choose the strongest locally available mandated GGUF in this
order: Qwen3.6-35B-A3B, Gemma4-31B-it, then Gemma4-26B-A4B-it. It SHALL prefer
a CUDA llama.cpp path when CUDA is healthy. If CUDA is absent or unhealthy but
the loader can run a bounded CPU smoke, the CPU smoke MAY prove receipt wiring
only and MUST be classified as `cpu_fallback_receipt_only`; it SHALL NOT allow
headline claims or unlock clean verifier reruns. If no mandated model can be
located or loaded, the artifact SHALL set `preflight_passed=false`,
`live_call_count=0`, empty `proof_receipts`, a precise substrate
classification, and a blocked `honest_verdict` without claiming live verifier
evidence.

When a mandated GGUF can run, the builder SHALL execute at least two tiny
deterministic prompts using a fixed seed and deterministic decoding parameters
where supported. For every completed call it SHALL capture prompt hashes,
transcript hashes, token counts, wall-clock evidence, command hash, worker code
hash, subprocess return code, and stderr tail. It SHALL reject reused or stale
transcript hashes and SHALL fail throughput plausibility when completion-token
throughput is impossible under the v3 contract. The smoke artifact remains
receipt proof only: `headline_claim_allowed` MUST be false even when the
preflight passes.

The terminal artifact MUST include `local_sota_receipt_smoke_v3_ready`,
`preflight_passed`, `live_call_count`, `mandated_model_inventory`,
`selected_model_ids`, `substrate_classification`, `cpu_fallback_used`,
`proof_receipts`, `throughput_plausibility_passed`, `headline_claim_allowed`,
`clean_rerun_allowed`, `inference_substrate`, and `honest_verdict`. It SHOULD
also include the Exp 3178 contract summary, cached-pair probe, loader and CUDA
probe evidence, prompt/transcript hash lists, source checksums, tests run,
duration, and run date. `clean_rerun_allowed` SHALL be true only for a full
local SOTA receipt with CUDA-backed mandated GGUF calls, at least two fresh
receipts, passing throughput plausibility, and no stale transcript reuse.

### SCENARIO-VERIFY-3179: Local Receipt Smoke Either Produces Fresh Receipts Or Blocks Honestly

Given the Exp 3178 v3 contract exists and the local host may or may not have
mandated SOTA GGUF files, `llama_cpp`, and CUDA,
When Exp 3179 builds the local SOTA receipt-smoke artifact,
Then it writes the terminal JSON artifact, records all three mandated model
inventory rows, records the cached-pair and runtime substrate probes, either
captures at least two fresh deterministic proof receipts from a mandated GGUF
or records `live_call_count=0` with the exact cache, loader, CUDA, or runtime
blocker, keeps `headline_claim_allowed=false`, and sets
`clean_rerun_allowed=true` only for the `full_local_sota_receipt` substrate.

## Implementation Status (REQ-VERIFY-3179)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3179 | Implemented (`python/carnot/verify/local_sota_receipt_smoke_v3.py`) | Implemented (`tests/python/test_experiment_3179_local_sota_receipt_smoke_v3.py`) |

### REQ-VERIFY-3192: Receipt And Adversarial Contract V4 Dual Thresholds

The repository shall provide an Exp 3192 receipt/adversarial contract v4
builder that writes
`results/experiment_3192_receipt_adversarial_contract_v4.json` from checked-in
artifacts only. The builder MUST compare the Exp 3178 v3 receipt contract, Exp
3179 local SOTA receipt smoke, and Exp 3189 cross-corpus matrix v29 findings
without running live model inference, verifier scoring, repairs, solver
execution, hardware commands, the conductor, pushes, or modifying
`scripts/research_conductor.py`.

The v4 contract SHALL define two separate downstream thresholds. The
`proof_execution_sufficient` threshold MAY be satisfied by
`cpu_fallback_receipt_only` evidence when receipt rows contain the required
model identity, model-file hash, loader, substrate, prompt and transcript
hashes, token counts, random seeds, wall-clock evidence, command hash,
subprocess return code, stderr tail, throughput plausibility, and replay count.
The `clean_rerun_allowed` threshold SHALL require full local SOTA substrate
evidence and SHALL NOT be satisfied by CPU fallback, missing cache, missing
loader, unavailable CUDA, unhealthy CUDA/offload, or any diagnostic-only row.

The contract SHALL define mandatory adversarial and methodology fields for
live, gated-skip, aggregate, and diagnostic-only artifacts. Aggregate artifacts
MUST declare inherited methodology and exact source checksums instead of
pretending to have run a model. Gated-skip artifacts MUST remain auditable by
recording gate reasons, upstream snapshots, preconditions, zero live calls, and
non-headline boundaries. Diagnostic-only artifacts MUST deny headline and
deployed-claim permissions explicitly. Live artifacts MUST record full
methodology and receipt provenance.

The contract SHALL define terminal verdict prefixes and blocked-precondition
verdict allowances. Terminal success verdicts MUST start with `complete:`,
`complete_`, `success:`, `success_`, `passed:`, `passed_`, `shipped:`, or
`shipped_`. Honest precondition failures MAY start with a `blocked_` prefix
when the verdict names the concrete failed resource or gate. The terminal Exp
3192 artifact itself MUST use an `honest_verdict` that starts with
`complete:`.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`contract_version`, `proof_execution_required_fields`,
`clean_rerun_required_fields`, `aggregate_required_fields`,
`gated_skip_required_fields`, `accepted_substrate_classes`,
`rejected_headline_substrate_classes`, `terminal_verdict_prefixes`,
`blocked_verdict_prefixes`, `downstream_unlock_fields`, and
`honest_verdict`. It SHOULD also include live and diagnostic-only field
requirements, source-artifact provenance, comparison findings, current evidence
assessment, field-principle annotations, tests run, source checksums, duration,
run date, and explicit protected-file non-modification flags.

### SCENARIO-VERIFY-3192: V4 Separates CPU Proof Receipts From Clean Rerun Eligibility

Given Exp 3178 v3 exists, Exp 3179 reports real `llama_cpp` proof receipts with
`substrate_classification=cpu_fallback_receipt_only`, and matrix v29 reports
flagged, gated-skip, aggregate, and diagnostic-only rows that need clearer
methodology fields,
When Exp 3192 builds the v4 receipt/adversarial contract,
Then it writes the terminal JSON artifact without live inference, marks the
current CPU fallback receipts sufficient for proof-of-execution only, keeps
`clean_rerun_allowed=false` until `full_local_sota_receipt` evidence exists,
defines mandatory methodology and adversarial fields for live, gated-skip,
aggregate, and diagnostic-only artifacts, rejects CPU fallback and other
non-full substrates from headline claims, publishes explicit terminal and
blocked verdict prefix rules, exposes machine-readable downstream unlock
fields, and uses an `honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-VERIFY-3192)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3192 | Implemented (`python/carnot/verify/receipt_adversarial_contract_v4.py`) | Implemented (`tests/python/test_experiment_3192_receipt_adversarial_contract_v4.py`) |

### REQ-VERIFY-3193: Llama.cpp CUDA Offload Health Probe V1

The repository shall provide an Exp 3193 llama.cpp CUDA/offload health probe
that writes
`results/experiment_3193_llama_cpp_cuda_offload_health_probe_v1.json`. The
probe MUST inspect local preconditions before attempting model work, including
`nvidia-smi` visibility, Python torch CUDA availability, `llama_cpp` import and
backend metadata, and the local HuggingFace cache inventory for all three
mandated GGUF model families:
`unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. The probe MUST NOT download large files,
use legacy small models for headline fields, push commits, or modify
`scripts/research_conductor.py`.

When `llama_cpp` reports GPU offload support, CUDA hardware is visible, and at
least one mandated GGUF candidate exists locally, the probe SHALL attempt one
tiny deterministic receipt call against the strongest available mandated
candidate with explicit `n_gpu_layers` intent. For that call it SHALL record
wall-clock duration, token counts, prompt hash, response hash, transcript hash,
model-file hash, stderr/backend tail, GPU memory observations before and after
the attempt, and whether GPU layers were actually evidenced by backend metadata
or GPU memory deltas. A compliant result SHALL set
`substrate_classification=full_local_sota_receipt`,
`clean_rerun_allowed=true`, and `headline_claim_allowed=true` only when the
receipt completed on a mandated GGUF with offload evidence and no CPU fallback.

When CUDA, `llama_cpp` GPU offload, the mandated cache, or the live receipt
attempt is unavailable or unhealthy, the probe SHALL write a terminal blocked
artifact instead of claiming clean verifier eligibility. CPU fallback MAY be
recorded only as non-headline diagnostic evidence and SHALL keep both
`clean_rerun_allowed=false` and `headline_claim_allowed=false`.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`model_specs`, `preconditions_checked`, `cache_inventory`, `selected_model`,
`selected_model_file_hash`, `llama_cpp_backend_metadata`,
`n_gpu_layers_requested`, `n_gpu_layers_effective`, `gpu_observations`,
`receipt_count`, `receipts`, `substrate_classification`,
`clean_rerun_allowed`, `headline_claim_allowed`, `blocker_reasons`, and
`honest_verdict`. The `honest_verdict` MUST start with `complete:` for a
compliant offload receipt or `blocked_` for a precise blocked precondition.

### SCENARIO-VERIFY-3193: CUDA Offload Receipt Or Precise Blocked Substrate

Given the Exp 3192 v4 contract allows clean reruns only for full local SOTA
receipt evidence, and the local host may have mandated GGUF cache files,
`llama_cpp`, CUDA hardware, or only CPU fallback,
When Exp 3193 builds the llama.cpp CUDA/offload health probe artifact,
Then it writes the terminal JSON artifact with all required fields, records all
three mandated model families and available quant files, records CUDA and
`llama_cpp` backend preconditions, either captures one deterministic offload
receipt with hashes, token counts, model-file hash, stderr/backend tail, and
GPU memory evidence or records zero headline receipts with exact blocker
reasons, and never sets `clean_rerun_allowed=true` or
`headline_claim_allowed=true` for CPU fallback or blocked substrate states.

## Implementation Status (REQ-VERIFY-3193)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3193 | Implemented (`python/carnot/verify/llama_cpp_cuda_offload_health_probe_v1.py`) | Implemented (`tests/python/test_experiment_3193_llama_cpp_cuda_offload_health_probe_v1.py`) |

### REQ-VERIFY-3195: Adaptive Verification Granularity Policy V1

The repository shall provide an Exp 3195 deterministic adaptive verification
granularity policy builder that writes
`results/experiment_3195_adaptive_verification_granularity_policy_v1.json`
from checked-in artifacts only. The builder MUST consume the Exp 3180
controlled-invariance executor v2, Exp 3183 counterexample-certificate
expansion v3, Exp 3189 cross-corpus matrix v29, the post-.295 research
references, and the repository workflow/spec instructions. It MUST NOT call an
LLM, score with a live verifier, execute repair generation, train or promote an
EBM/LLM sidecar, download models, push commits, or modify
`scripts/research_conductor.py`.

The policy SHALL map interpretable row/risk features to one action from the
explicit action space `final-answer-only`, `step-chunk`,
`counterexample-fragment`, `abstain/escalate`, or
`skip redundant recheck`. Policy features SHALL include row family, known
false-accept risk, certificate depth, answer ambiguity, prior verification
outcome, exact-authority completeness, and receipt-backed transcript context.
Exact solvers, exact-authority replay, canonical answers, and counterexample
certificates SHALL remain the only authority; EBM, LLM, transcript, and receipt
fields MAY route or suppress redundant checks but SHALL NOT accept an answer or
authorize promotion.

The simulation SHALL run over existing certificate rows only, without fresh
model or verifier calls. It SHALL identify available exact rows,
false-accept families, counterexample certificates, bounded frontier evidence,
and receipt-backed transcripts, then estimate verifier-call changes against a
declared fixed-granularity baseline. It SHALL report risk tradeoffs by counting
whether any known false-accept or ambiguous row is suppressed, skipped, or
routed below counterexample-fragment granularity.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`policy_version`, `source_artifacts`, `exact_rows_used`, `policy_features`,
`policy_actions`, `simulated_rows`, `estimated_verifier_call_delta`,
`false_accept_risk_increase`, `redundant_recheck_suppression_rule`,
`promotion_allowed`, and `honest_verdict`. It SHOULD also include evidence
inventory, policy schedule counts, verifier-call accounting, row-level
simulation records, false-accept family summaries, no-new-execution metadata,
source checksums, tests run, and explicit notes that ops/status, changelog, and
traceability reconciliation are delegated when the conductor prompt says so.

`promotion_allowed` SHALL always remain false for Exp 3195. The honest verdict
MUST start with `complete:` when the deterministic policy artifact is
materialized and schema-valid, even when the simulated policy keeps repair or
headline promotion blocked.

### SCENARIO-VERIFY-3195: Existing Rows Route To Adaptive Granularity Without New Authority

Given Exp 3180 reports exact rows, known false-accept regression rows,
controlled-invariance results, and receipt-backed transcripts; Exp 3183 reports
expanded certificate records and bounded frontier records; and Exp 3189 reports
the .295 matrix blocker state,
When the Exp 3195 policy builder runs,
Then it writes
`results/experiment_3195_adaptive_verification_granularity_policy_v1.json`
without live inference, routes known false-accept and ambiguous certified rows
to counterexample-fragment checks, routes repair/fragment rows to step-chunk
checks, escalates incomplete or unknown-family rows, permits redundant recheck
suppression only for exact-complete low-risk accepted rows already covered by
controlled invariance, estimates verifier-call delta against the declared
baseline, reports zero increased known false-accept risk only when no known
false-accept row is skipped or downgraded, keeps EBM/LLM/receipt fields
non-authoritative, sets `promotion_allowed=false`, and emits an
`honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-VERIFY-3195)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3195 | Implemented (`python/carnot/verify/adaptive_verification_granularity_policy_v1.py`) | Implemented (`tests/python/test_experiment_3195_adaptive_verification_granularity_policy_v1.py`) |

### REQ-VERIFY-3210: Context-CoT/CL-Bench Parametric-Shortcut Fixture Bank V1

The repository shall provide an Exp 3210 deterministic fixture-bank builder
that writes
`data/research/context_cot_clbench_parametric_shortcut_v1.jsonl` and
`results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json`
without invoking an LLM, running GPU inference, downloading models, pushing
commits, modifying `scripts/research_conductor.py`, or modifying the active
`research-roadmap.yaml`.

The fixture bank SHALL contain 20 to 40 rows spanning at least the symbolic
aliases, local arithmetic rules, and context-defined entity facts families.
Every row SHALL include a stable fixture id, family, context, question,
expected answer, prior-bait answer, exact checker type, and minimal
counterexample. The context SHALL define a local rule or fact that intentionally
contradicts a likely pretrained prior, and the expected answer SHALL follow the
local context while the prior-bait answer SHALL be rejected by the exact
checker.

The builder SHALL expose deterministic exact checking helpers that accept the
expected answer and reject each row's prior-bait answer. It MAY expose a score
helper for candidate answers, but any optional LLM smoke evaluation SHALL remain
null unless it records at least one mandated local SOTA GGUF model spec:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`; legacy small models SHALL be smoke-only and
SHALL NOT create headline context-following claims.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`milestone`, `reference_papers`, `fixture_path`, `fixture_count`,
`fixture_families`, `exact_checker_types`, `prior_bait_row_count`,
`context_following_score_available`, `optional_llm_smoke`,
`ready_for_clean_verifier`, `conductor_file_modified`,
`active_roadmap_modified`, and `honest_verdict`.

`ready_for_clean_verifier` SHALL be true only when the JSONL fixture exists,
the fixture count is between 20 and 40, all three required families are covered,
all rows have prior-bait counterexamples, every expected answer passes its
checker, and every prior-bait answer fails its checker. The honest verdict MUST
start with `complete:` when the deterministic fixture bank and schema-valid
artifact are materialized.

### SCENARIO-VERIFY-3210: Parametric Shortcuts Are Exact-Checkable

Given the post-.296 Context-CoT/CL-Bench research references identify
parametric shortcuts as a context-learning failure mode,
When the Exp 3210 fixture builder runs,
Then it writes the JSONL fixture bank and terminal result artifact, covers the
symbolic aliases, local arithmetic rules, and context-defined entity facts
families, records a prior-bait answer and minimal counterexample for every row,
confirms exact checkers accept context-following answers and reject likely
prior answers, keeps optional LLM smoke evaluation null, sets
`ready_for_clean_verifier=true`, and records that neither the conductor file nor
the active roadmap was modified.

## Implementation Status (REQ-VERIFY-3210)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3210 | Implemented (`python/carnot/verify/context_cot_clbench_parametric_shortcut_fixtures_v1.py`) | Implemented (`tests/python/test_experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.py`) |

### REQ-VERIFY-3223: Distributional Exact-Row Uncertainty Sidecar V2

The repository shall provide an Exp 3223 deterministic uncertainty sidecar
builder that writes
`results/experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.json`
from the `.297` context-shortcut and ConstraintBench-style exact fixture
artifacts:
`results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json`,
`data/research/context_cot_clbench_parametric_shortcut_v1.jsonl`,
`results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json`,
and `data/research/constraintbench_feasibility_objective_pilot_v1.jsonl`.
It MUST NOT invoke an LLM, train a model, download models, push commits,
modify `scripts/research_conductor.py`, or modify the active roadmap.

The sidecar SHALL score every loaded exact fixture row using deterministic
features already present in the source artifacts. Context rows SHALL expose
context-dependency and prior-bait shortcut features. ConstraintBench rows SHALL
expose satisfiability or feasibility, objective-gap, contradiction or
constraint-violation, and checker-backend metadata features. The sidecar SHALL
emit row-level uncertainty, abstention risk, shortcut risk, and
solver-disagreement risk fields, but those fields SHALL be triage metadata
only and SHALL NOT replace exact checker or solver authority.

The builder SHALL run a negative-control audit that checks whether sidecar risk
is dominated by model identity or artifact source rather than row difficulty.
Because these `.297` fixtures are deterministic and do not contain live model
responses, absent model identity SHALL be recorded explicitly rather than
imputed. Artifact-source effects SHALL be measured separately from difficulty
features, and any domination finding SHALL block
`uncertainty_sidecar_ready`.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`milestone`, `source_fixture_artifacts`, `exact_row_count`,
`uncertainty_sidecar_ready`, `abstention_threshold_defined`,
`shortcut_risk_rows`, `solver_disagreement_risk_rows`,
`model_identity_shortcut_audit`, `clean_verifier_consumption_plan`,
`exact_verifier_authority_preserved`, `inference_substrate`,
`conductor_file_modified`, `active_roadmap_modified`, and `honest_verdict`.
It SHOULD also include all row scores, sidecar method details, source
checksums, tests run, run date, and a plan for Exp 3225 to consume the sidecar
as triage metadata while preserving exact verifier scoring.

`uncertainty_sidecar_ready` SHALL be true only when the required source
fixture artifacts are present, at least one context row and one ConstraintBench
row are scored, `exact_row_count` equals the row-score denominator, an
abstention threshold is defined over row-level abstention risk, the negative
control audit is not model-identity-dominated or artifact-source-dominated,
`exact_verifier_authority_preserved=true`, and the inference substrate records
deterministic artifact replay with zero model calls.

### SCENARIO-VERIFY-3223: Exact Fixture Rows Get Triage Metadata Only

Given Exp 3210 and Exp 3211 have materialized deterministic `.297` exact
fixture artifacts,
When the Exp 3223 sidecar builder runs,
Then it loads both fixture banks, scores every exact row without live
inference, emits uncertainty, abstention, shortcut-risk, and
solver-disagreement-risk fields per row, reports high-priority shortcut and
solver-disagreement rows as triage lists, records a negative-control audit for
model identity and artifact-source shortcut domination, defines how Exp 3225
may consume the sidecar as routing metadata, and keeps exact verifier scoring
as the only authority for accept/reject decisions.

## Implementation Status (REQ-VERIFY-3223)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3223 | Implemented (`python/carnot/eval/distributional_ebm_exact_row_uncertainty_sidecar_v2.py`) | Implemented (`tests/python/test_experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.py`) |

### REQ-VERIFY-3224: Logitext Partial SMT Context Coverage Pilot V1

The repository shall provide an Exp 3224 deterministic partial SMT coverage
builder that writes
`results/experiment_3224_logitext_partial_smt_context_coverage_pilot_v1.json`
from the `.297` context-shortcut and ConstraintBench-style exact fixture
artifacts:
`results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json`,
`data/research/context_cot_clbench_parametric_shortcut_v1.jsonl`,
`results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json`,
and `data/research/constraintbench_feasibility_objective_pilot_v1.jsonl`.
It MUST NOT invoke an LLM, train a model, download models, push commits,
modify `scripts/research_conductor.py`, or modify the active roadmap.

The builder SHALL define a small typed constraint taxonomy covering at least
`boolean`, `string_equality`, `arithmetic_interval`, `all_different`,
`graph_relation`, `feasibility`, and `objective_bound` fragments. It SHALL inspect every loaded
fixture row and mark the row as `fully_formalizable`,
`partially_formalizable`, or `not_formalizable_without_extraction`. Context
rows SHALL preserve the boundary between exact answer-check fragments already
present in the fixture and natural-language local-rule extraction that is not
being claimed. ConstraintBench rows SHALL map their structured instances to
solver-ready fragments or to the existing exact solver backend label.

For each formalizable row, the builder SHALL record either a deterministic
solver-ready representation or a pointer to the existing exact checker/solver
label. It SHALL report coverage by fixture family and identify high-value rows
for Exp 3225 clean verifier scoring, while preserving exact solver/checker
authority and treating any Logitext-style natural-language layer as future
extraction work rather than current solver evidence.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`milestone`, `source_fixture_artifacts`, `fixture_row_count`,
`fully_formalizable_count`, `partially_formalizable_count`,
`not_formalizable_count`, `constraint_taxonomy`, `smt_rows`,
`partial_smt_coverage`, `exact_solver_row_count`, `coverage_ready`,
`inference_substrate`, `conductor_file_modified`,
`active_roadmap_modified`, and `honest_verdict`.

`coverage_ready` SHALL be true only when all four source fixture artifacts are
present, at least one context row and one ConstraintBench row are loaded, every
loaded row receives exactly one formalizability label, every fully or partially
formalizable row has a solver-ready fragment or exact solver pointer, the
coverage counts add up to the fixture-row denominator, and the inference
substrate records deterministic artifact replay with zero model calls.

### SCENARIO-VERIFY-3224: Context And Constraint Rows Receive SMT Coverage Labels

Given Exp 3210 and Exp 3211 have materialized deterministic `.297` exact
fixture artifacts,
When the Exp 3224 coverage builder runs,
Then it loads both fixture banks without live inference, emits a typed
constraint taxonomy, assigns one formalizability status to every context and
ConstraintBench row, records exact answer-check or solver-backend pointers for
formalizable rows, summarizes coverage by fixture family, lists the
highest-value rows for Exp 3225 clean verifier scoring, sets
`coverage_ready=true`, and records that neither the conductor file nor the
active roadmap was modified.

## Implementation Status (REQ-VERIFY-3224)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3224 | Implemented (`python/carnot/verify/logitext_partial_smt_context_coverage_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3224_logitext_partial_smt_context_coverage_pilot_v1.py`) |

### REQ-VERIFY-3196: GenCP Domain Preview Repair Compiler V1

The repository shall provide an Exp 3196 deterministic GenCP-style domain
preview compiler that writes
`results/experiment_3196_gencp_domain_preview_repair_compiler_v1.json` from
checked-in artifacts only. The compiler MUST consume the Exp 3183
counterexample-certificate expansion v3, Exp 3195 adaptive verification
granularity policy v1, Exp 3138 canonical-answer grounding pilot, Exp 3084
exact fixture bank, invariant/frontier certificate artifacts when available,
the post-.295 research references, and the repository workflow/spec
instructions. It MUST NOT call an LLM, invoke repair generation, run live
verifier scoring, train or promote an EBM/LLM sidecar, download models, push
commits, or modify `scripts/research_conductor.py`.

The compiler SHALL define repair-domain records from canonical answers,
known false-accept families, exact fixture authority payloads, and bounded
invariant/certificate frontiers. Each record SHALL include a row id, source
artifact, row family, canonical answer, candidate domain values, domain size,
constraint evidence, propagation trace, exact rejection test ids, and an
authority note stating that exact/canonical checks remain final. Candidate
domains SHALL be bounded by deterministic values already present in source
rows, such as canonical labels, observed candidate answers, minimal correction
set values, exact fixture labels, and frontier statuses.

The compiler SHALL define explicit propagation rules that prune candidate
domains before any later repair prompt is attempted. Rules SHALL cover
canonical-answer equality, false-accept candidate conflict elimination,
minimal-correction-set or unsat-core preservation, fixture-authority
compatibility, frontier-status pruning, and repair-gate blocking. The compiler
SHALL define exact rejection predicates that a later repair call must satisfy,
including canonical answer mismatch, known false-accept token-family confusion,
violated MCS/unsat-core invariant, fixture authority contradiction,
frontier-pruned prefix use, non-authoritative receipt or sidecar promotion,
and repair-gate blocked state.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`compiler_version`, `source_artifacts`, `domain_record_schema`,
`propagation_rules`, `exact_rejection_tests`, `preview_domain_count`,
`average_candidate_domain_size`, `repair_call_ready`, `promotion_allowed`, and
`honest_verdict`. It SHOULD also include a small preview manifest of bounded
candidate spaces, source schema observations, no-new-execution metadata,
source checksums, tests run, and explicit rationale for why the artifact does
not invoke or promote LLM repair. `repair_call_ready` and `promotion_allowed`
SHALL remain false for Exp 3196 because the artifact prepares domains only and
does not clear the blocked repair gate.

The honest verdict MUST start with `complete:` when the deterministic compiler
artifact is materialized and schema-valid, even though it keeps repair calls and
promotion blocked.

### SCENARIO-VERIFY-3196: Existing Rows Compile Into Bounded Preview Domains

Given Exp 3183 exposes certificate rows and bounded frontiers, Exp 3195 keeps
promotion blocked while routing exact rows, Exp 3138 exposes canonical
false-accept grounding, and Exp 3084 exposes exact fixture authority payloads,
When Exp 3196 builds the GenCP domain preview compiler artifact,
Then it writes
`results/experiment_3196_gencp_domain_preview_repair_compiler_v1.json`
without live inference, creates bounded domain preview records from existing
rows, applies deterministic propagation rules that keep only candidates
compatible with canonical answers, exact fixture authority, false-accept
certificates, and frontier status, lists exact rejection tests for later repair
calls, reports bounded candidate-domain accounting, keeps
`repair_call_ready=false`, keeps `promotion_allowed=false`, and emits an
`honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-VERIFY-3196)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3196 | Implemented (`python/carnot/verify/gencp_domain_preview_repair_compiler_v1.py`) | Implemented (`tests/python/test_experiment_3196_gencp_domain_preview_repair_compiler_v1.py`) |

### REQ-VERIFY-3197: ExVerus Inductive Certificate Expansion V1

The repository shall provide an Exp 3197 deterministic ExVerus-style
inductive certificate expansion builder that writes
`results/experiment_3197_exverus_inductive_certificate_expansion_v1.json`
from checked-in artifacts only. The builder MUST consume the Exp 3183
counterexample-certificate expansion v3, the Exp 3196 GenCP domain preview
repair compiler v1, the post-.295 research references, and the repository
workflow/spec instructions. It MUST NOT call an LLM, run repair generation,
execute live verifier scoring, download models, push commits, or modify
`scripts/research_conductor.py`.

The builder SHALL expand a bounded set of exact certificate rows carrying
pilot counterexample, repair-fragment, or drift-anchor evidence into inductive
invariant records. Each invariant record SHALL include an observed
counterexample or anchor, a generalized invariant, an exact guard, an
anti-overfit test, source lineage, and any linked GenCP domain preview record.
Exact guards SHALL remain authoritative and SHALL be derived from exact
checker, canonical-answer, MCS, unsat-core, parser, or solver evidence rather
than from model text or sidecar scores.

Anti-overfit tests SHALL reject repairs that merely patch the observed failing
instance while violating the generalized invariant. The tests SHALL include the
exact authority to rerun, the row family, the observed patch risk, and the
expected rejection or preservation outcome. For positive drift anchors, the
anti-overfit test SHALL guard against over-repair by requiring the already
accepted exact label to remain unchanged.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`source_artifacts`, `invariant_schema`, `invariant_record_count`,
`exact_guard_count`, `anti_overfit_test_count`,
`linked_domain_preview_count`, `repair_call_ready`, and `honest_verdict`. It
SHOULD also include invariant records, source checksums, source errors,
limitations, no-new-execution metadata, tests run, and explicit rationale for
why the artifact does not execute or unlock repair. `repair_call_ready` SHALL
remain false for Exp 3197 because invariant guards prepare later repair
evaluation only and do not clear the blocked repair gate.

The honest verdict MUST start with `complete:` when the deterministic
artifact is materialized and schema-valid, even though it keeps repair calls
blocked.

### SCENARIO-VERIFY-3197: Counterexamples Become Inductive Guards

Given Exp 3183 exposes counterexample certificate records with pilot
invariant evidence and Exp 3196 exposes matching GenCP domain preview rows,
When Exp 3197 builds the ExVerus inductive certificate expansion artifact,
Then it writes
`results/experiment_3197_exverus_inductive_certificate_expansion_v1.json`
without live inference, emits one bounded invariant record for each selected
exact pilot row, preserves observed counterexample evidence, generalizes the
record into an invariant-level guard, materializes an exact anti-overfit test
for the repair gate, links matching GenCP preview domains, records limitations
and the no-repair-execution boundary, keeps `repair_call_ready=false`, and
emits an `honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-VERIFY-3197)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3197 | Implemented (`python/carnot/verify/exverus_inductive_certificate_expansion_v1.py`) | Implemented (`tests/python/test_experiment_3197_exverus_inductive_certificate_expansion_v1.py`) |

### REQ-VERIFY-3198: Repair Gate Decision V5 Aggregation

The repository shall provide an Exp 3198 deterministic repair-gate decision v5
builder that writes `results/experiment_3198_repair_gate_decision_v5.json`
from checked-in `.296` upstream artifacts only. The builder MUST consume Exp
3192 receipt/adversarial contract v4, Exp 3193 llama.cpp CUDA/offload health
probe v1, Exp 3194 clean live SOTA verifier rerun v11, Exp 3195 adaptive
verification granularity policy v1, Exp 3196 GenCP domain preview repair
compiler v1, and Exp 3197 ExVerus inductive certificate expansion v1. It MUST
NOT run live inference, invoke repair generation, execute verifier scoring,
download models, push commits, or modify `scripts/research_conductor.py`.

The gate SHALL fail closed if any required upstream artifact is missing,
malformed, blocked, gate-skipped, CPU-fallback-only, or adversarially flagged.
It SHALL evaluate receipt eligibility, clean-verifier readiness, adversarial
flags, adaptive scheduling readiness, bounded domain-preview readiness, and
invariant-certificate coverage as separate machine-readable states. Clean
repair SHALL be unblocked only when the receipt/offload evidence is full local
SOTA and unflagged, the clean local SOTA verifier artifact is eligible and
unflagged with computed false-accept evidence, the adaptive policy has
zero known false-accept risk increase, the GenCP domain preview has bounded
candidate domains, and ExVerus invariant records provide exact guards and
anti-overfit tests linked to those domains.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`source_artifacts`, `receipt_gate_state`, `clean_verifier_state`,
`adaptive_policy_state`, `domain_preview_state`,
`invariant_certificate_state`, `repair_gate_state`, `repair_allowed_scope`,
`blocker_reasons`, `downstream_gated_skip_expected`, and `honest_verdict`.
`repair_gate_state` SHALL be either `unblocked_for_bounded_repair_ladder` or a
precise blocked state. Blocked states SHALL set `repair_allowed_scope=null`,
include machine-readable blocker reasons, and set
`downstream_gated_skip_expected=true`. The unblocked state SHALL include an
exact bounded repair scope with finite row and attempt budgets, mandated local
SOTA model policy, domain-preview and invariant-source requirements,
exact-authority acceptance, and a no-headline-claim boundary.

The `honest_verdict` MUST start with `complete:` for both blocked and
unblocked terminal artifacts, because Exp 3198 is an aggregation decision whose
truthful completion can still decide that downstream repair must be skipped.

### SCENARIO-VERIFY-3198: V5 Gate Blocks On Skipped Or Flagged Clean Verifier

Given the `.296` Exp 3192 through Exp 3197 artifacts may be present, blocked,
gate-skipped, CPU fallback only, or adversarially flagged,
When Exp 3198 builds the repair-gate decision v5 artifact,
Then it writes the terminal JSON artifact without live inference, records each
upstream artifact in `source_artifacts`, reports receipt, clean-verifier,
adaptive-policy, domain-preview, and invariant-certificate states separately,
keeps repair blocked whenever the clean local SOTA verifier is not eligible or
is adversarially flagged, carries precise machine-readable blocker reasons for
missing, blocked, gate-skipped, CPU-fallback-only, or flagged upstreams, and
sets `downstream_gated_skip_expected=true` for the downstream repair ladder.

If every upstream precondition is clean, unflagged, and ready, then the same
builder SHALL set `repair_gate_state=unblocked_for_bounded_repair_ladder` and
publish only a bounded repair scope over the intersection of Exp 3196 preview
domains and Exp 3197 invariant guards.

## Implementation Status (REQ-VERIFY-3198)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3198 | Implemented (`python/carnot/verify/repair_gate_decision_v5.py`) | Implemented (`tests/python/test_experiment_3198_repair_gate_decision_v5.py`) |

### REQ-VERIFY-3213: Repair Gate Decision V6 Artifact Aggregation

The repository shall provide an Exp 3213 deterministic repair-gate decision
v6 builder that writes
`results/experiment_3213_repair_gate_decision_v6.json` from checked-in
`.297` evidence only. The builder MUST load the listed upstream artifacts
when present, record missing paths explicitly, and MUST NOT invoke an LLM,
run live inference, execute repair generation, execute verifier scoring,
download models, push commits, modify `scripts/research_conductor.py`, or
modify the active `research-roadmap.yaml`.

The gate SHALL treat Exp 3208 receipt evidence, Exp 3209 clean-verifier
evidence, and Exp 3212 structured proposal preflight evidence as mandatory.
It SHALL set `receipt_gate_passed=true` only when Exp 3208 reports
`clean_rerun_allowed=true`. It SHALL set `clean_verifier_gate_passed=true`
only when Exp 3209 reports `clean_verifier_state=clean` and carries no
unhandled adversarial methodology flag. It SHALL set
`structured_proposal_gate_passed=true` only when Exp 3212 reports
`ready_for_repair_gate=true` while `repair_correctness_claimed=false`.

The builder SHALL incorporate Exp 3210 and Exp 3211 as auxiliary exact fixture
coverage, not as mandatory blockers when absent. Auxiliary fixture artifacts
SHALL block only when present and explicitly flagged invalid. The prior Exp
3198 repair-gate artifact and conductor log SHALL be recorded as provenance,
not as independent authorities that can override the three mandatory gates.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`milestone`, `required_artifacts`, `missing_artifacts`,
`receipt_gate_passed`, `clean_verifier_gate_passed`,
`structured_proposal_gate_passed`, `auxiliary_fixture_artifacts`,
`repair_gate_state`, `repair_ladder_allowed`, `blockers`,
`conductor_file_modified`, `active_roadmap_modified`, and `honest_verdict`.
`repair_gate_state` SHALL be `unblocked`, `blocked`, or `diagnostic_only`.
`repair_gate_state=unblocked` and `repair_ladder_allowed=true` SHALL occur
only when the receipt, clean verifier, and structured proposal gates all pass
and no present auxiliary fixture artifact is flagged invalid. Blocked and
diagnostic-only artifacts SHALL include machine-readable blocker details and
SHALL keep `repair_ladder_allowed=false`.

The `honest_verdict` MUST start with `complete:` for all terminal decisions,
because Exp 3213 is an evidence aggregator whose truthful result can be an
unblocked gate, a blocked gate, or a diagnostic-only artifact.

### SCENARIO-VERIFY-3213: V6 Gate Uses Only Artifact Evidence

Given `.297` evidence may contain a blocked Exp 3208 receipt artifact, missing
or gate-skipped Exp 3209 clean-verifier evidence, missing or gate-skipped Exp
3212 structured proposal evidence, auxiliary exact fixture artifacts, and the
prior Exp 3198 repair-gate decision,
When Exp 3213 builds the repair-gate decision v6 artifact,
Then it records every listed source path as present or missing, keeps Exp 3210
and Exp 3211 auxiliary unless they are explicitly invalid, emits separate
receipt, clean-verifier, and structured-proposal gate booleans, and sets
`repair_gate_state=unblocked` only when those three mandatory gates pass
without invalid auxiliary evidence.

If any mandatory gate is missing, blocked, malformed, adversarially unhandled,
or otherwise insufficient, then the builder SHALL keep
`repair_ladder_allowed=false`, set `repair_gate_state` to `blocked` or
`diagnostic_only`, and include stable blocker codes explaining the failed
evidence.

## Implementation Status (REQ-VERIFY-3213)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3213 | Implemented (`python/carnot/verify/repair_gate_decision_v6.py`) | Implemented (`tests/python/test_experiment_3213_repair_gate_decision_v6.py`) |

### REQ-VERIFY-3227: Repair Gate Decision V7 Artifact Aggregation

The repository shall provide an Exp 3227 deterministic repair-gate decision
v7 builder that writes
`results/experiment_3227_repair_gate_decision_v7.json` from checked-in `.298`
upstream artifacts only. The builder MUST load Exp 3222 full local SOTA
receipt v6, Exp 3225 clean live SOTA verifier rerun v13, and Exp 3226
structured repair proposal preflight v2 when present. It SHOULD also record
Exp 3213 repair gate decision v6, the repository workflow instructions, and
the conductor gate helper as provenance. It MUST NOT invoke an LLM, run live
inference, execute repair generation, execute verifier scoring, download
models, push commits, modify `scripts/research_conductor.py`, or modify the
active `research-roadmap.yaml`.

The gate SHALL set `receipt_ok=true` only when Exp 3222 reports
`clean_rerun_allowed=true`, is readable, is not a conductor pre-gate skip, and
does not record CPU fallback as receipt evidence. It SHALL set
`clean_verifier_ok=true` only when Exp 3225 reports `clean_verifier_ready=true`,
is readable, is not a conductor pre-gate skip, does not record CPU fallback,
and preserves exact verifier authority through exact labels or exact verifier
metadata. It SHALL set `structured_preflight_ok=true` only when Exp 3226
reports `ready_for_repair_gate=true`, `exact_verifier_handoff_ready=true`, and
`repair_correctness_claimed=false`, with no schema-only repair limitation.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`milestone`, `input_artifacts`, `receipt_ok`, `clean_verifier_ok`,
`structured_preflight_ok`, `blocker_list`, `blocker_count`,
`repair_gate_state`, `repair_ladder_allowed`, `inference_substrate`,
`conductor_file_modified`, `active_roadmap_modified`, and `honest_verdict`.
`repair_gate_state` SHALL be `unblocked`, `blocked`, or `diagnostic_only`.
`repair_gate_state=unblocked` and `repair_ladder_allowed=true` SHALL occur
only when the receipt, clean verifier, and structured preflight gates all pass
and `blocker_list` is empty. Missing artifacts, malformed artifacts,
conductor pre-gate skips, CPU fallback evidence, absent exact verifier
authority, and schema-only repair limitations SHALL be preserved as stable
machine-readable blockers. Blocked and diagnostic-only artifacts SHALL set
`repair_ladder_allowed=false`.

The `honest_verdict` MUST start with `complete:` for every terminal decision,
because Exp 3227 is an aggregation decision whose truthful result can still
decide that the downstream repair ladder must remain blocked.

### SCENARIO-VERIFY-3227: V7 Gate Fails Closed Until Receipt, Verifier, And Preflight Pass

Given Exp 3222, Exp 3225, and Exp 3226 evidence may be present, missing,
malformed, gate-skipped, CPU-fallback-only, exact-verifier-incomplete, or
schema-only,
When Exp 3227 builds the repair-gate decision v7 artifact,
Then it records every input source as present or missing, emits separate
receipt, clean-verifier, and structured-preflight booleans, preserves blockers
for each insufficient upstream condition, keeps `repair_ladder_allowed=false`
unless all three mandatory gates are genuinely ready, and records that neither
the conductor file nor the active roadmap was modified.

If all three upstream artifacts are readable and clean, Exp 3222 has
`clean_rerun_allowed=true` without CPU fallback, Exp 3225 has
`clean_verifier_ready=true` with exact verifier authority, and Exp 3226 has
`ready_for_repair_gate=true` plus exact-verifier handoff without claiming
repair correctness, then the builder SHALL set `repair_gate_state=unblocked`,
`repair_ladder_allowed=true`, and an empty blocker list.

## Implementation Status (REQ-VERIFY-3227)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3227 | Implemented (`python/carnot/verify/repair_gate_decision_v7.py`) | Implemented (`tests/python/test_experiment_3227_repair_gate_decision_v7.py`) |

### REQ-VERIFY-3180: Controlled-Invariance Executor V2

The repository shall provide an Exp 3180 controlled-invariance executor that
writes `results/experiment_3180_controlled_invariance_executor_v2.json` from
checked-in exact-authority evidence and receipt-backed transcripts only. The
executor MUST consume the Exp 3166 invariance/token-suspicion audit, Exp 3167
clean live SOTA verifier rerun gate artifact, Exp 3179 local SOTA receipt
smoke artifact when present, and the Exp 3136/3137/3138 exact false-accept and
canonical authority artifacts. It MUST NOT make live model calls, download
models, execute repairs, execute verifier scoring, modify
`scripts/research_conductor.py`, or push commits.

The executor SHALL load the .294 control definitions for force-answer,
remove-answer, shuffled-trace, answer-only, trace-only, transcript-hash, and
token-suspicion triage, then execute those controls against exact-authority
rows and known false-accept regression rows. Exact labels, canonical
equivalence, exact-safe replay, solver/test authority, and monitor-ledger
consistency SHALL remain the only acceptance authority. Token-suspicion,
transcript-hash, answer-only, trace-only, and shuffled-trace signals MAY route
rows to exact checks or block promotion, but SHALL NOT accept outputs.

The terminal artifact MUST include
`controlled_invariance_executor_v2_ready`, `exact_row_count`,
`receipt_backed_transcript_count`, `control_results`,
`shortcut_failure_count`, `known_false_accept_regression_count`,
`token_suspicion_used_as_triage_only`, `controlled_invariance_passed`,
`blocker_reasons`, `inference_substrate`, and `honest_verdict`. It SHOULD also
include semantic false-accept counts, executed-row summaries, source artifact
provenance, tests run, source checksums, duration, run date, and an explicit
statement that ops/status, changelog, and traceability reconciliation are left
to the conductor when the task prompt says so.

`controlled_invariance_passed` shall be true only when every load-bearing
control is executed and passes, at least one exact row is evaluated, every
known false-accept regression row is included and rejected by exact authority,
receipt-backed transcript hashes are unique when receipts are present, no
shortcut control accepts a row without exact authority, token-suspicion fields
remain triage-only, semantic false accepts are zero, and the inference
substrate declares zero new live model calls.

### SCENARIO-VERIFY-3180: Exact Authority Executes Controls Without Live Calls

Given Exp 3166 defines the controlled-invariance and token-suspicion controls,
Exp 3167 exposes exact-row and regression-row provenance, and Exp 3179 may
contain receipt-backed transcript hashes,
When Exp 3180 builds the controlled-invariance executor artifact,
Then it writes the terminal JSON artifact without live inference, executes
force-answer, remove-answer, shuffled-trace, answer-only, trace-only,
transcript-hash, and token-suspicion triage controls over the exact rows and
available receipts, reports shortcut failures separately from semantic false
accepts, keeps token suspicion as non-authority triage, includes every known
false-accept regression row as load-bearing evidence, and sets
`controlled_invariance_passed=true` only when all load-bearing controls pass.

## Implementation Status (REQ-VERIFY-3180)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3180 | Implemented (`python/carnot/verify/controlled_invariance_executor_v2.py`) | Implemented (`tests/python/test_experiment_3180_controlled_invariance_executor_v2.py`) |

### REQ-VERIFY-3181: Clean Live SOTA Verifier Rerun V10 Receipt Gate

The repository shall provide an Exp 3181 clean live SOTA verifier rerun v10
builder that writes
`results/experiment_3181_clean_live_sota_verifier_rerun_v10.json` for repair
gate v4. The builder MUST consume Exp 3178 receipt-backed authenticity
contract v3, Exp 3179 local SOTA receipt smoke v3, Exp 3180
controlled-invariance executor v2, and Exp 3167 clean live SOTA verifier rerun
v9 before attempting any live verifier work. It MUST NOT make live local SOTA
model calls unless Exp 3179 has `clean_rerun_allowed=true` and Exp 3180 has
`controlled_invariance_passed=true`. It MUST NOT use legacy small models for
headline verifier metrics, download models, push commits, or modify
`scripts/research_conductor.py`.

When either precondition is missing or false, the builder SHALL write a
complete gated-skip artifact rather than leaving repair gate v4 to infer status
from a missing file. The gated-skip artifact SHALL set
`clean_live_sota_verifier_rerun_v10_ready=true`, `gated_skip=true`,
`live_call_count=0`, empty `models_used`, explicit `gate_reasons`, and
`headline_claim_allowed=false`. It SHALL carry forward exact-row and known
false-accept regression denominators from Exp 3180 when available, include the
receipt rows inspected from Exp 3179 without treating them as verifier-score
evidence, and set `flagged_adversarial=true` unless receipt proof is valid,
controlled invariance passes, and every known false-accept regression row is
rejected by exact authority.

When both preconditions pass, the builder SHALL select mandated SOTA GGUFs
from `cached_sota_pair()` or equivalent local path discovery over
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, run only a bounded clean verifier panel
over exact-authority rows plus known false-accept regression rows, and score
the panel with canonical exact authority only. For every live row it SHALL
record model ID, model path, prompt hash, transcript hash, exact label,
decision, and whether the row is a known false-accept regression. Token
suspicion and transcript receipt fields MAY triage or block promotion, but
SHALL NOT accept verifier outputs.

The terminal artifact MUST include
`clean_live_sota_verifier_rerun_v10_ready`, `gated_skip`, `gate_reasons`,
`live_call_count`, `models_used`, `proof_receipts_used`, `exact_row_count`,
`known_false_accept_regression_count`, `false_accept_rate`,
`false_reject_rate`, `abstention_rate`, `controlled_invariance_passed`,
`flagged_adversarial`, `headline_claim_allowed`, `inference_substrate`, and
`honest_verdict`. It SHOULD also include source provenance, prompt/transcript
hash lists, token-suspicion triage status, transcript receipt validity,
per-row exact scoring summaries, field-principle annotations, tests run,
source checksums, duration, and run date.

### SCENARIO-VERIFY-3181: Receipt Preconditions Gate V10 Live Verifier Work

Given Exp 3179 may contain only CPU fallback receipt proof and Exp 3180 may or
may not have passed controlled invariance,
When Exp 3181 builds the clean live SOTA verifier rerun v10 artifact,
Then it writes a terminal JSON artifact for repair gate v4, makes zero live
model calls whenever `clean_rerun_allowed` or `controlled_invariance_passed`
is false or missing, records exact gate reasons for the skipped work, preserves
the exact-row and known false-accept denominators, keeps headline claims
blocked, and marks the verifier evidence adversarial until full local SOTA
receipts, controlled invariance, transcript receipt validity, and exact
known-regression rejection all pass.

## Implementation Status (REQ-VERIFY-3181)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3181 | Implemented (`python/carnot/verify/clean_live_sota_verifier_rerun_v10.py`) | Implemented (`tests/python/test_experiment_3181_clean_live_sota_verifier_rerun_v10.py`) |

### REQ-VERIFY-3182: Distributional EBM Exact-Row Sidecar V1

The repository shall provide an Exp 3182 distributional EBM uncertainty
sidecar builder that writes
`results/experiment_3182_distributional_ebm_exact_row_sidecar_v1.json` from
checked-in exact verifier rows, known false-accept rows, cached candidate
evidence, Exp 3173 EBCN/KAN diagnostics, Exp 3180 controlled-invariance
results, and the post-.294 reference sweep. It MUST NOT make live model calls,
train a network, download models, modify `scripts/research_conductor.py`, push
commits, or present the sidecar as deployed verifier authority.

The sidecar SHALL represent each available exact row with deterministic
structured constraint features or graph-like features where practical. The
method MAY use deterministic proxy scoring when the available false-accept
denominator is too small for training, but it MUST label the method as a
proxy diagnostic and keep exact labels, exact-safe replay, canonical grounding,
and controlled-invariance checks as the only acceptance authority.

The terminal artifact MUST include
`distributional_ebm_exact_row_sidecar_v1_ready`, `source_reference_ids`,
`exact_labeled_row_count`, `known_false_accept_rows_scored`,
`sidecar_method`, `false_accept_separation_auc`,
`uncertainty_calibration`, `abstention_policy`,
`comparison_to_ebcn_kan`, `deployed_verifier_claim_allowed`,
`inference_substrate`, and `honest_verdict`. It SHOULD also include row-level
scores, an uncertainty ranking, calibration-bin details, coverage denominators,
source checksums, tests run, duration, run date, and explicit comparison to
Exp 3180 shortcut/controlled-invariance findings.

`distributional_ebm_exact_row_sidecar_v1_ready` shall be true only when the
Exp 3173 and Exp 3180 source artifacts are present, at least one exact row is
scored, every known false-accept regression row available from those sources
is included, calibration is computed or honestly marked unavailable, the
abstention policy names its threshold and coverage denominator, the inference
substrate records zero new live model calls, and
`deployed_verifier_claim_allowed=false`.

### SCENARIO-VERIFY-3182: Deterministic Sidecar Scores Exact Rows Without Promotion

Given Exp 3173 provides exact labeled rows with EBCN/KAN diagnostic records,
Exp 3180 provides controlled-invariance row outcomes and shortcut-exposure
evidence, and the post-.294 reference sweep cites distributional EBMs and
Graph Energy Matching as diagnostic sidecar directions,
When Exp 3182 builds the distributional exact-row sidecar,
Then it writes the terminal JSON artifact without live inference, computes
row-level proxy energy and uncertainty over the exact-row denominator, reports
false-accept separation AUROC when both clean and known false-accept rows are
scored, reports ECE or an explicit calibration-unavailable reason, proposes an
abstention threshold with coverage, compares the findings to Exp 3173 EBCN/KAN
records and Exp 3180 shortcut controls, and keeps
`deployed_verifier_claim_allowed=false`.

## Implementation Status (REQ-VERIFY-3182)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3182 | Implemented (`python/carnot/eval/distributional_ebm_exact_row_sidecar_v1.py`) | Implemented (`tests/python/test_experiment_3182_distributional_ebm_exact_row_sidecar_v1.py`) |

### REQ-VERIFY-3183: Counterexample-Certificate Expansion V3

The repository shall provide an Exp 3183 counterexample-certificate expansion
builder that writes
`results/experiment_3183_counterexample_certificate_expansion_v3.json` for
repair gate v4 and downstream Exp 3184. The builder MUST consume the Exp 3170
counterexample-certificate pilot v2, Exp 3168 repair gate decision v3, Exp 3169
repair ladder materializer v4, Exp 3180 controlled-invariance executor v2, Exp
3181 clean verifier rerun v10, Exp 3182 distributional sidecar v1, and exact
authority artifacts when available. It MUST NOT call an LLM, run repair
generation, execute live verifier scoring, download models, push commits, or
modify `scripts/research_conductor.py`.

The expansion SHALL preserve every Exp 3170 certificate row and SHALL expand
coverage with the larger exact-row denominator from Exp 3180 or Exp 3181. Every
certificate record SHALL include row id, canonical answer or source, exact
label, checker result, known false-accept or regression status, counterexample
family, source artifact, and whether the record depends on flagged live
verifier evidence. Known false-accept rows and available regression rows SHALL
remain load-bearing and counted explicitly.

The builder SHALL also materialize BEAVER-inspired bounded frontier records
from Exp 3170 and Exp 3125 evidence. Each frontier record SHALL include a
prefix or state, constraint family, lower and upper bounds when bounded mass is
available or an exact status otherwise, and a deterministic stop reason.

The terminal artifact MUST include
`counterexample_certificate_expansion_v3_ready`, `exact_row_count`,
`counterexample_count`, `certificate_records`, `bounded_frontier_records`,
`known_false_accept_rows_covered`, `flagged_adversarial`,
`repair_call_ready`, `blocker_reasons`, `inference_substrate`, and
`honest_verdict`. `repair_call_ready` SHALL be true only when certificates are
broad enough, exact-authority scoring is complete, and no certificate evidence
depends on a flagged live verifier. When readiness is false, blocker reasons
MUST be actionable and `honest_verdict` MUST still start with a terminal
success prefix if the deterministic artifact itself was materialized.

### SCENARIO-VERIFY-3183: Expanded Certificates Gate Repair Calls Without Live Inference

Given Exp 3170 has materialized a certificate pilot, Exp 3180 or Exp 3181
exposes exact rows and known false-accept regressions, and Exp 3182 may provide
offline sidecar row scores,
When Exp 3183 builds the counterexample-certificate expansion,
Then it writes the terminal JSON artifact without live inference, records every
expanded exact row with canonical answer or source, checker result, and
counterexample family, converts bounded frontier rows into deterministic
prefix/state records with bounds or exact statuses, counts all covered known
false-accept rows, carries forward flagged-adversarial blockers, and sets
`repair_call_ready=true` only when exact evidence is broad, complete, and free
of flagged live-verifier dependence.

## Implementation Status (REQ-VERIFY-3183)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3183 | Implemented (`python/carnot/verify/counterexample_certificate_expansion_v3.py`) | Implemented (`tests/python/test_experiment_3183_counterexample_certificate_expansion_v3.py`) |

### REQ-VERIFY-3184: Repair Gate Decision V4 Aggregation

The repository shall provide an Exp 3184 deterministic repair-gate decision
builder that writes `results/experiment_3184_repair_gate_decision_v4.json`
from checked-in upstream artifacts and exact gate policies only. The builder
MUST consume Exp 3179 local SOTA receipt smoke v3, Exp 3180 controlled
invariance executor v2, Exp 3181 clean live SOTA verifier rerun v10, Exp 3183
counterexample-certificate expansion v3, and the prior Exp 3168 repair-gate
decision v3 for provenance. It MUST NOT make live model calls, execute repair
generation, execute verifier scoring, download models, push commits, or modify
`scripts/research_conductor.py`.

The gate SHALL evaluate these load-bearing unblocking predicates explicitly:
`exp3179.clean_rerun_allowed=true`, `exp3180.controlled_invariance_passed=true`,
`exp3181.flagged_adversarial=false`, `exp3181.headline_claim_allowed=true` for
verifier metrics, an acceptable false-accept gate from computed clean verifier
metrics, and `exp3183.repair_call_ready=true`. If any required artifact or gate
policy source is missing or malformed, the builder SHALL list it in
`missing_artifacts` and set `repair_gate_state=blocked_missing_artifact`. If any
predicate is false or flagged, the builder SHALL list every failed predicate in
`blocker_reasons` and choose the most specific blocked state rather than
collapsing failures into a generic skip.

The terminal artifact MUST include `repair_gate_decision_v4_ready`,
`repair_gate_state`, `unblocking_predicates`, `blocker_reasons`,
`missing_artifacts`, `allowed_repair_attempt_budget`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. The state SHALL be exactly one of
`blocked_missing_artifact`, `blocked_receipt_precondition`,
`blocked_controlled_invariance`, `blocked_clean_verifier_flagged`,
`blocked_headline_claim_blocked`, `blocked_false_accept_gate`,
`blocked_certificate_not_ready`, `blocked_other`, or
`unblocked_for_bounded_repair_ladder`.

`repair_gate_state=unblocked_for_bounded_repair_ladder` SHALL be allowed only
when every unblocking predicate passes, no required artifact is missing, and
the artifact specifies a strict repair attempt budget with finite limits,
mandated local-SOTA model policy, exact-authority acceptance, and a no-headline
claim boundary. Blocked states SHALL set the repair-attempt budget to disabled
with zero attempts and an actionable disabled reason. The inference substrate
SHALL declare zero new live model calls, zero repair calls, and aggregation-only
execution.

### SCENARIO-VERIFY-3184: V4 Gate Blocks Until Receipts, Verifier Metrics, And Certificates Pass

Given Exp 3179, Exp 3180, Exp 3181, Exp 3183, and Exp 3168 artifacts may be
present, missing, blocked, or flagged,
When Exp 3184 builds the repair-gate decision v4 artifact,
Then it writes the terminal JSON artifact without live inference, records each
unblocking predicate with source path, expected value, actual value, and pass
status, exposes missing artifacts separately from failed predicates, keeps
repair blocked whenever receipt smoke does not allow clean rerun, controlled
invariance fails, the clean verifier is flagged, headline verifier metrics are
not allowed, the false-accept gate is not backed by computed clean metrics, or
certificates are not repair-call ready, and opens only the bounded repair
ladder state with strict attempt limits when every predicate passes.

## Implementation Status (REQ-VERIFY-3184)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3184 | Implemented (`python/carnot/verify/repair_gate_decision_v4.py`) | Implemented (`tests/python/test_experiment_3184_repair_gate_decision_v4.py`) |

### REQ-VERIFY-3185: Multi-Turn Repair Ladder V5

The repository shall provide an Exp 3185 multi-turn repair ladder v5 builder
that writes `results/experiment_3185_multi_turn_repair_ladder_v5.json` whether
the Exp 3184 repair gate opens or remains blocked. The builder MUST load
`results/experiment_3184_repair_gate_decision_v4.json` before any repair
attempt. If `repair_gate_state` is anything other than
`unblocked_for_bounded_repair_ladder`, the builder MUST NOT call a live model,
MUST set `gated_skip=true`, `repair_attempt_count=0`, empty `models_used`,
empty `transcript_receipts`, empty `exact_check_results`,
`repair_success_delta=0.0`, `headline_claim_allowed=false`, and MUST carry all
gate blockers into `remaining_blockers`.

When Exp 3184 unblocks repair, the builder SHALL select only a bounded set of
certificate-backed repair targets from Exp 3183 counterexample certificate
records. Before every repair call it SHALL re-check mandated local SOTA GGUF
availability and receipt evidence for the exact model used. The mandated model
policy is exactly `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; legacy small models MAY be recorded only as
loud smoke fallbacks and SHALL NOT populate headline repair metrics.

For each selected repair target, the ladder SHALL run at most two bounded
turns: an initial proposal, an exact/canonical semantic check, counterexample
feedback when the first proposal fails, a second proposal, and a second exact
check. It SHALL hash every repair-call transcript and SHALL score success only
from exact/canonical semantic checks, never from JSON validity, model
self-judgment, or schema shape alone.

The terminal artifact MUST include `multi_turn_repair_ladder_v5_ready`,
`gated_skip`, `gate_state`, `repair_attempt_count`, `models_used`,
`repair_targets`, `transcript_receipts`, `exact_check_results`,
`repair_success_delta`, `remaining_blockers`, `headline_claim_allowed`,
`inference_substrate`, and `honest_verdict`. `headline_claim_allowed` may be
true only when the gate is unblocked, mandated local SOTA GGUF evidence was
used, every accepted repair has exact/canonical semantic acceptance, no legacy
small model contributes to headline metrics, and all remaining blockers are
resolved.

### SCENARIO-VERIFY-3185: Blocked Gate Produces Explicit No-Call Repair Ladder

Given Exp 3184 may be blocked and Exp 3183 may contain certificate-backed
counterexample records,
When Exp 3185 builds the repair ladder v5 artifact,
Then it first reads the Exp 3184 gate, writes a complete terminal artifact even
when repair is skipped, preserves gate blockers in `remaining_blockers`, records
zero repair attempts and no headline claim for blocked gates, and performs live
repair calls only for the exact unblocked gate state with mandated SOTA GGUF
cache and receipt evidence re-checked before each call.

## Implementation Status (REQ-VERIFY-3185)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3185 | Implemented (`python/carnot/verify/multi_turn_repair_ladder_v5.py`) | Implemented (`tests/python/test_experiment_3185_multi_turn_repair_ladder_v5.py`) |

### REQ-VERIFY-3006: EqR Fixed-Point Energy Diagnostic Over Cached Validator Trajectories

The repository shall provide a deterministic Exp 3006 fixed-point energy
diagnostic over cached Exp 3005 solver/validator trajectories. The diagnostic
MUST NOT implement or claim a native Equilibrium Reasoner, attractor model, or
learned fixed-point architecture. It SHALL derive an auditable scalar energy
from exact validator evidence: runtime node failures, Z3 execution failures,
solver-status mismatches, partial-prefix violations, contradiction injections,
and expected-status incompatibilities.

For every loaded trajectory, the diagnostic SHALL measure whether the
available iterative repair/constraint-feedback states move toward a lower
energy fixed point. At minimum, the trajectory states SHALL include an invalid
partial or contradiction-feedback state, a valid extendable partial state, and
the full exact-checked candidate state. The diagnostic SHALL report convergence
rate from first to final state, energy monotonicity across adjacent states,
basin sensitivity under small deterministic perturbations, and rejection of
negative controls generated by permuting constraints, swapping incompatible
validator references, or injecting contradiction nodes.

The terminal artifact MUST be written to
`results/experiment_3006_eqr_fixed_point_energy_diagnostic_v1.json` and include
`fixed_point_diagnostic_ready`, `n_trajectories`, `energy_definition`,
`convergence_rate`, `energy_monotonicity_rate`,
`basin_sensitivity_summary`, `negative_control_rejection_rate`,
`diagnostic_table_path`, `native_eqr_claim_made`, and `honest_verdict`.
`fixed_point_diagnostic_ready` shall be true only when the Exp 3005 manifest is
loaded, at least one trajectory is scored, the energy definition is present,
the diagnostic table exists, convergence and monotonicity rates are measured,
basin perturbation evidence is summarized, negative controls are rejected at a
nonzero rate, and `native_eqr_claim_made=false`.

### SCENARIO-VERIFY-3006: Cached Validator Feedback Is Scored As A Diagnostic Energy

Given the Exp 3005 validator manifest and transcripts exist,
When the Exp 3006 diagnostic runs,
Then it reconstructs deterministic solver/validator trajectories from cached
exact checks, assigns scalar energies to invalid-feedback, partial-feedback,
and full-candidate states, measures convergence, adjacent-step monotonicity,
basin sensitivity, and negative-control rejection, writes an inspectable table,
and emits a terminal artifact with `native_eqr_claim_made=false`.

## Implementation Status (REQ-VERIFY-3006)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3006 | Implemented (`python/carnot/eval/eqr_fixed_point_energy_diagnostic_v1.py`) | Implemented (`tests/python/test_experiment_3006_eqr_fixed_point_energy_diagnostic.py`) |

### REQ-VERIFY-2981: Interwhen Partial-Monitor Promotion Metrics V2

The repository shall provide an Exp 2981 deterministic promotion evaluator for
the Exp 2968 partial-output monitor harness. The evaluator MUST NOT run fresh
live inference, MUST NOT modify `scripts/research_conductor.py`, and MUST write
`results/experiment_2981_interwhen_partial_monitor_promotion_v2.json`.

The evaluator MUST first verify that
`results/experiment_2979_solver_feedback_mcs_frontier_v1.json` reports
`frontier_upgrade_ready=true`. It SHALL load the Exp 2968 monitor harness
artifact and the Exp 2979 feedback frontier, define partial code and solver
monitor event types for `draft_intent`, `constraint_emission`,
`parse_boundary`, `verifier_call`, `counterexample`, and `repair_step`, and
evaluate deterministic monitor fixtures that measure event coverage, prefix
failure localization, monitor latency, and false alarms.

When `results/experiment_2980_sota_solver_formalization_feedback_v2.json` is
present, the evaluator SHALL additionally derive monitor traces from the actual
Exp 2980 solver candidate rows. When Exp 2980 is absent, it SHALL report
`live_trace_count=0` and rely only on deterministic fixtures. The evaluator
SHALL keep `full_streaming_verification_claim=false` unless all required live
code and solver streams are actually monitored; checked-in partial fixtures
and solver candidate traces alone do not justify a full streaming verification
claim.

The terminal artifact MUST include `honest_verdict`,
`partial_monitor_promoted`, `full_streaming_verification_claim=false`,
`event_types`, `coverage_by_event`, `prefix_failure_localization_rate`,
`monitor_latency_ms`, `false_alarm_rate`, `fixture_count`,
`live_trace_count`, `promotion_gates`,
`inference_substrate="deterministic_monitor_harness"`, and `duration_s`.

`partial_monitor_promoted` shall be true only when Exp 2979 is ready, every
required event type has measured coverage, both code and solver streams are
covered by fixtures or live traces, prefix failure localization is at least
0.80, and false alarm rate is at most 0.20.

### SCENARIO-VERIFY-2981: Promotion Requires Coverage And Localization Metrics

Given Exp 2968 has produced a deterministic partial monitor harness artifact
and Exp 2979 reports `frontier_upgrade_ready=true`,
When the Exp 2981 promotion evaluator runs,
Then it emits the required event vocabulary, measures coverage for code and
solver partial-output events, computes prefix failure localization and
false-alarm rates from deterministic fixtures plus any available Exp 2980
candidate traces, promotes the partial monitor only when the explicit gates
clear, and leaves `full_streaming_verification_claim=false` because the
artifact does not monitor all live generation streams.

## Implementation Status (REQ-VERIFY-2981)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-2981 | Implemented (`python/carnot/eval/interwhen_partial_monitor_promotion_v2.py`) | Implemented (`tests/python/test_experiment_2981_interwhen_partial_monitor_promotion_v2.py`) |

### REQ-VERIFY-2932: Citation Hallucination Field Verifier

The repository shall provide a local citation-hallucination probe that:

- builds a fixture of at least 30 citation cases from real `research-references.md`
  references plus controlled title, author, year, venue, DOI/arXiv id, URL, and
  nonexistent-seed mutations;
- calls `cached_sota_pair(gpu_indices=(0, 1))` before selecting any local GGUF
  fallback, and records at least one of the mandated SOTA GGUF model IDs when
  live inference is available;
- asks the selected local GGUF model for short citation-bearing answers, writes
  raw responses to disk, and never uses web search as experiment truth;
- extracts citation fields with deterministic parsing before judgment;
- verifies extracted title, author, year, venue, DOI/arXiv id, and URL fields
  against the fixture truth; and
- writes `results/experiment_2932_citation_hallucination_field_verifier_v1.json`
  with the required Exp 2932 artifact fields, including taxonomy counts and a
  reproducibility checksum.

### SCENARIO-VERIFY-2932: Fixture Citations Are Classified By Field Truth

Given the Exp 2932 fixture contains real references, field mutations, and a
nonexistent-seed case,
When model raw outputs are parsed into structured citation fields,
Then the deterministic verifier classifies each citation as `real`,
`potential/ambiguous`, `hallucinated-field`, or `nonexistent-seed`, reports
extraction success, field-match accuracy, hallucination-detection accuracy, and
writes the required terminal JSON without consulting external search.

### REQ-VERIFY-2358: Synthetic EBM-CoT Consistency Calibration

The repository shall provide a deterministic, CPU-only EBM-CoT consistency
calibrator that:

- scores a chain-of-thought trace with lower energy for adjacent-step
  consistency and higher energy for adjacent contradictions;
- refines traces with discrete Langevin-style proposals that accept only
  non-increasing-energy updates;
- calibrates batches of traces by returning one energy score per trace; and
- writes `results/experiment_2358_ebm_cot.json` with AUROC, mean energy
  reduction, trace count, random seed, and an honest terminal-prefix verdict.

The experiment MUST use 50 synthetic traces and fixed random seed 42.

### SCENARIO-VERIFY-2358: EBM-CoT Synthetic Corpus Separates Consistency

Given 25 consistent synthetic chain-of-thought traces and 25 traces with one
adjacent contradiction,
When the EBM-CoT calibrator scores all traces and refines five inconsistent
traces for 50 steps,
Then the artifact reports `n_traces=50`, `random_seed=42`,
`ebm_cot_validated` exactly when AUROC is at least 0.60, and a non-negative
mean energy reduction after refinement.

### REQ-VERIFY-1673: Hybrid Verifier Pipeline

The system shall provide a unified pipeline (`python/carnot/solvers/hybrid_verifier.py`) where a neural generator predicts, PiNet projects the predictions to satisfy continuous constraints, and Z3 formally verifies the boolean logic of the projected continuous constraints. The hybrid verifier MUST expose pass rates and validation latency.

### SCENARIO-VERIFY-1673: Hybrid Verifier E2E Evaluation

Given a set of constraints,
When the hybrid verifier runs the unified pipeline (generator predicts, PiNet projects, Z3 verifies),
Then it should produce a deliverable JSON (`results/experiment_1673_hybrid_constraint.json`) with pass rates, validation latency, and an honest verdict.

### REQ-VERIFY-1372: GS-KAN PWA Energy-Bound Verification

The repository shall provide a deterministic verification path for a small
`GSKANEnergy` layer that:

- builds a piecewise-affine abstraction for each shared GS-KAN spline;
- records the maximum abstraction error across splines;
- encodes an input-box energy-bound property as an LP or MILP, using a manual
  LP fallback when optional solvers such as SciPy or PuLP are unavailable;
- compares the certified bound against interval arithmetic; and
- writes `results/experiment_1372_optimal_kan_pwa_formal_verification.json`
  with explicit fields showing whether a formal KAN energy-bound claim is
  allowed.

The artifact MUST state that the proof is CPU-only and MUST NOT claim hardware
execution or hardware correctness.

### SCENARIO-VERIFY-1372: Small GS-KAN Layer Bound Is Certified

Given a deterministic small `GSKANEnergy` layer and a FoVer-derived valid input
feature box,
When the verifier builds knot-aligned PWA abstractions and solves the separable
LP bound,
Then the artifact reports `milp_verification_result="verified"` only when the
certified upper bound is strictly below the tested energy threshold, and
`kan_formal_claim_allowed` is true only in that verified case.

### REQ-VERIFY-1381: DVI Discriminative Verifier Training

The repository shall provide a deterministic DVI training path that:

- writes `results/experiment_1381_dvi_discriminative_verifier_training_v1.json`
  with `status="in_progress"` before source loading or training;
- loads only fresh semantically verified positive cases from Exp 1374's
  primary semantic path;
- samples contrastive incorrect reasoning steps from the FoVer corpus at a
  positive:negative ratio of at least 1:3;
- initializes from the current SC-Energy or GS-KAN verifier checkpoint when
  one is available;
- runs at least ten discriminative training epochs with a binary
  cross-entropy or contrastive hinge objective;
- measures AUROC on one fixed held-out FoVer split before and after training;
- writes `python/carnot/models/dvi_checkpoint_v1.pt`; and
- records `dvi_auroc_delta`, deployment status, and an honest verdict without
  requiring any fresh LLM inference.

### SCENARIO-VERIFY-1381: DVI Checkpoint And AUROC Delta Are Auditable

Given Exp 1374 contains four primary semantic verified positive cases and the
FoVer corpus contains correct and incorrect held-out rows,
When the DVI training runner executes,
Then the final artifact contains the required DVI fields, the checkpoint path
exists when `dvi_deployed` is true, and
`discriminative_improvement_measured` is true whenever the AUROC delta was
computed on the fixed held-out split.

### REQ-VERIFY-1386: SECL Discriminative Self-Calibration

The repository shall provide a deterministic, CPU-only SECL calibration path
for the SC-Energy verifier that:

- writes `results/experiment_1386_secl_discriminative_self_calibration.json`
  with `status="in_progress"` before loading data or calibrating;
- identifies the current SC-Energy checkpoint or deterministic SC-Energy
  fallback used for scoring;
- loads only Exp 1374 promoted primary-semantic verified positive cases as the
  positive discriminative signal;
- loads FoVer contrastive negative cases for the calibration slice;
- trains a confidence head on the selected SECL slice by minimizing empirical
  Expected Calibration Error for fixed confidence bins;
- measures `ece_before` and `ece_after` on one fixed held-out FoVer split;
- computes `ece_reduction_pct`, `discriminative_signal_correlation`, and
  `calibration_cases_used`; and
- sets `secl_viable_for_dvi` to true exactly when `ece_reduction_pct > 10.0`.

### SCENARIO-VERIFY-1386: SECL Artifact Reports Held-Out ECE

Given Exp 1374 contains promoted semantic positives and the FoVer corpus
contains correct and incorrect held-out rows,
When the SECL self-calibration runner executes,
Then the final artifact is complete, records the SC-Energy verifier target,
uses the promoted positives and FoVer negatives for calibration, and reports a
held-out ECE reduction percentage with an honest verdict.

### REQ-VERIFY-1394: DVI V2 With SECL Combined Deployment

The repository shall provide a deterministic, CPU-only DVI v2 combined
deployment path that:

- writes `results/experiment_1394_dvi_v2_secl_combined.json` with
  `status="in_progress"` before loading source artifacts or training;
- loads exactly the 59 DVI-only fresh verified Exp 1382 case IDs promoted by
  Exp 1388, matching `fresh_verified_sample_count=59`;
- initializes DVI v2 from the deployed Exp 1381 verifier checkpoint;
- fine-tunes the verifier with the same SECL-style binary cross-entropy
  discriminative objective used by Exp 1381, using FoVer incorrect rows as
  contrastive negatives;
- measures `dvi_v2_baseline_auroc`, `dvi_v2_trained_auroc`, and
  `dvi_v2_auroc_delta` on a fixed held-out FoVer split;
- applies the Exp 1386 histogram SECL discriminative self-calibration recipe to
  the DVI v2 checkpoint and reports `secl_ece_before`,
  `secl_ece_after`, and `secl_ece_reduction_pct`;
- deploys the combined DVI v2 metric, bias, loss history, and SECL confidence
  head under `python/carnot/verify/`; and
- writes a complete artifact containing `status`, `fresh_cases_used`,
  `dvi_v2_baseline_auroc`, `dvi_v2_trained_auroc`,
  `dvi_v2_auroc_delta`, `secl_ece_before`, `secl_ece_after`,
  `secl_ece_reduction_pct`, `dvi_v2_deployed`, `checkpoint_path`, and
  `honest_verdict`.

### SCENARIO-VERIFY-1394: Fresh DVI V2 Checkpoint Is Calibrated And Deployed

Given Exp 1388 identifies 59 DVI-only promoted Exp 1382 cases and the Exp 1381
DVI checkpoint is deployed,
When the DVI v2 + SECL combined runner executes,
Then the final artifact is complete, `fresh_cases_used` is 59, the AUROC delta
is measured before and after DVI v2 fine-tuning, the SECL ECE values are
measured before and after calibration, and the combined checkpoint exists when
`dvi_v2_deployed` is true.

### REQ-VERIFY-1396: FoVer Semantic Validation Calibration Fix

The repository shall provide a deterministic FoVer semantic validation
calibration path that:

- preserves Exp 1382 certificate parsing and certificate-state checks;
- applies an arithmetic-aware fallback before accepting DVI SAT on labeled
  incorrect FoVer/math arithmetic rows;
- applies a configurable DVI abstention band around the incorrect-probability
  threshold for SAT certificates on labeled correct rows;
- records source family, arithmetic claim count, arithmetic verifier score,
  DVI threshold margin, fallback route, and fallback verdict in each calibrated
  semantic row; and
- writes `results/experiment_1396_semantic_validation_pass_rate_fix_v1.json`
  with before/after semantic validation pass rates measured on a sample of Exp
  1382 semantic failures.

### SCENARIO-VERIFY-1396: Calibrated FoVer Rows Recover DVI Boundary Failures

Given parsed Exp 1382 certificate rows whose certificate state already matches
the FoVer label-implied state,
When the DVI score disagrees with the label on a known arithmetic source,
Then an incorrect row that DVI would classify as SAT is escalated to the repair
path, a correct SAT row inside the DVI abstention band is accepted through the
certificate/full-verifier path, and the calibrated row records diagnostic
fallback fields for later failure analysis.

### REQ-VERIFY-1400: BiPRM R2L Retrospective FoVer Pivot Probe

The repository shall provide a deterministic, CPU-only BiPRM retrospective
verification probe that:

- writes `results/experiment_1400_biprm_retrospective_verification_probe.json`
  with `status="in_progress"` before loading the FoVer corpus;
- loads FoVer verified pairs with one verified-correct positive reasoning row
  and one rejected negative reasoning row;
- records the BiPRM right-to-left update rule used by the probe as
  `r_t^R2L = f_theta(s_t | q, s_>t)`;
- computes a forward-only pivot score for each step as the verifier-energy
  decrease caused by removing that step;
- computes a retrospective R2L pivot score for each step from the final answer
  backward, using later steps as context for the current candidate pivot;
- measures pivot-step identification precision against human important-step
  metadata when present, and otherwise against FoVer rejected-step proxy labels;
- categorizes pivotal steps into arithmetic error, logical fallacy, missing
  premise, and hallucination; and
- writes a complete artifact containing `status`, `corpus_cases_used`,
  `forward_only_pivot_precision`, `biprm_r2l_pivot_precision`,
  `pivot_precision_delta`, `retrospective_verification_viable`,
  `pivotal_step_categories`, and `honest_verdict`.

The probe MUST NOT call any LLM or require GPU hardware.

### SCENARIO-VERIFY-1400: R2L Retrospective Scores Improve Proxy Pivot Localization

Given local FoVer positive/negative pairs with rejected-step proxy pivots,
When the BiPRM retrospective probe scores each negative reasoning trace,
Then the artifact reports forward-only and R2L pivot precision, computes
`pivot_precision_delta` as R2L precision minus forward-only precision, sets
`retrospective_verification_viable` exactly when that delta is positive, and
preserves an honest verdict describing whether the result rests on proxy rather
than human pivot annotations.

### REQ-VERIFY-1415: DVI V3 Fresh 1508-Case Update

The repository shall provide a deterministic, CPU-only DVI v3 update path that:

- writes `results/experiment_1415_dvi_v3_1508_fresh_cases.json` with
  `status="in_progress"` before loading Exp 1395 fresh IDs or training;
- loads the 1508 fresh verified FoVer case IDs promoted by Exp 1395 from
  `memory_updates.promoted` and reconstructs their labeled FoVer rows without
  fresh LLM inference;
- initializes from the deployed Exp 1394 DVI v2 + SECL checkpoint;
- trains on all 1508 fresh verified labeled cases using a fixed seed and a
  deterministic train/replay split;
- measures `dvi_v3_auroc_delta` on the fixed held-out FoVer split and compares
  it with the Exp 1394 `dvi_v2_auroc_delta` baseline;
- measures `nonforgetting_rate` on replay examples from Exp 1395 demotions;
- preserves or re-measures the Exp 1394 SECL ECE reduction when the combined
  verifier checkpoint is touched;
- deploys a DVI v3 checkpoint only when the v3 AUROC delta improves on the DVI
  v2 baseline, nonforgetting is preserved, and SECL calibration is preserved;
  and
- writes a complete or blocked artifact containing `status`,
  `fresh_verified_cases_used`, `dvi_v2_auroc_delta_baseline`,
  `dvi_v3_auroc_delta`, `dvi_v3_deployed`, `dvi_v3_checkpoint_path`,
  `nonforgetting_rate`, `secl_ece_reduction_pct_preserved`, `tests_run`, and
  `honest_verdict`.

### SCENARIO-VERIFY-1415: DVI V3 Checkpoint Is Deployed Or Honestly Blocked

Given Exp 1395 reports 1508 promoted fresh FoVer IDs and Exp 1394 deployed a
DVI v2 + SECL checkpoint,
When the DVI v3 update runner executes,
Then the final artifact records all required DVI v3 fields, uses all 1508 fresh
verified cases, compares the measured v3 AUROC delta against the DVI v2
baseline delta of 0.011458, records replay nonforgetting, and either writes an
existing checkpoint path when `dvi_v3_deployed=true` or reports the blocking
reason in `honest_verdict`.

### REQ-VERIFY-1432: DVI V3 Replay-Heldout Nonforgetting Repair

The repository shall provide a deterministic, CPU-only DVI v3 repair path that:

- writes `results/experiment_1432_dvi_v3_nonforgetting_replay_balanced.json`
  with `status="in_progress"` before loading source artifacts;
- reads Exp 1415, Exp 1394, and Exp 1395 evidence and classifies the Exp 1415
  nonforgetting failure as thresholding, sampling imbalance, model update drift,
  or unresolved from measured AUROC and replay-gate evidence;
- uses the 1508 Exp 1395 fresh verified cases and a deterministic replay
  calibration/evaluation split from Exp 1395 demotions without fresh LLM
  inference;
- applies a bounded replay-heldout threshold calibration or a replay-balanced
  update before deployment;
- uses the recorded Exp 1394 `dvi_v2_auroc_delta` as the baseline when present,
  otherwise the fixed baseline `0.011458`;
- deploys DVI v3 only when `nonforgetting_rate >= 0.99` on held-out replay and
  the DVI v3 AUROC delta does not regress below the DVI v2 baseline; and
- writes a terminal artifact containing `status`, `dvi_v3_deployed`,
  `dvi_v3_auroc_delta`, `dvi_v2_auroc_delta_baseline`,
  `nonforgetting_rate`, `replay_balance_applied`,
  `threshold_calibration_applied`, `fresh_cases_used`, `tests_run`, and
  `honest_verdict`.

### SCENARIO-VERIFY-1432: Replay-Heldout Calibration Repairs Threshold Failure

Given Exp 1415 improved DVI v3 AUROC but blocked deployment because
`nonforgetting_rate < 0.99`
And Exp 1395 provides 1508 fresh verified cases and replay demotions,
When Exp 1432 calibrates the DVI v3 acceptance threshold on a replay
calibration split and audits held-out replay,
Then the final artifact records the diagnosed failure mode, the calibration
settings, and all required deployment fields
And `dvi_v3_deployed=true` only if held-out nonforgetting is at least 0.99 and
the measured DVI v3 AUROC delta remains at or above the DVI v2 baseline.

### REQ-VERIFY-1416: EBM-CoT V3 Post-Hoc Temperature Calibration

The repository shall provide a deterministic, CPU-only post-hoc temperature
calibration pass for Exp 1401 EBM-CoT hinge-only scores that:

- writes `results/experiment_1416_ebm_cot_v3_temperature_calibration.json` with
  `status="in_progress"` before loading source scores or fitting the
  temperature;
- reuses Exp 1401 scores when available or regenerates only the minimal
  deterministic FoVer split needed to recover those scores without fresh LLM
  inference;
- fits a single positive scalar temperature `T*` on a validation split, never on
  the test split;
- applies the fitted temperature to EBM-CoT energies as a post-hoc scaling
  operation;
- reports AUROC before and after temperature scaling, preserving Exp 1401's
  positive AUROC delta when ranking is unchanged within measured tolerance;
- reports paraphrase energy variance before and after scaling; and
- writes a complete artifact containing `status`, `temperature_scaling_applied`,
  `best_temperature`, `calibration_auroc_delta_before`,
  `calibration_auroc_delta_after`,
  `paraphrase_energy_variance_before_temp_scaling`,
  `paraphrase_energy_variance_after_temp_scaling`, `variance_worsened`,
  `auroc_preserved`, and `honest_verdict`.

### SCENARIO-VERIFY-1416: Temperature Scaling Reduces Variance Without Changing Ranking

Given Exp 1401 reports a positive EBM-CoT hinge-only AUROC delta and worsened
paraphrase energy variance,
When the post-hoc temperature calibration runner fits `T*` on a validation split
and applies it to held-out EBM-CoT scores,
Then the final artifact reports the before/after AUROC deltas, marks
`auroc_preserved=true` only when the positive delta is preserved within
tolerance, marks `variance_worsened=false` only when post-temperature variance
is no greater than pre-temperature variance within tolerance, and records an
honest verdict describing both gates.

### REQ-VERIFY-1423: FoVer Process Reward Model V1

The repository shall provide a deterministic, CPU-only process reward model
training path that:

- writes `results/experiment_1423_process_reward_model_v1_fover_1508.json`
  with `status="in_progress"` before loading traces or training;
- loads the 1508 Exp 1395 promoted FoVer trace IDs and reconstructs
  step-level labels from available Exp 1397 certificate, scheduler, validator,
  or repair-localization outputs without fresh LLM inference;
- trains a lightweight feature-based classifier that predicts step correctness
  without executing the full certificate path at inference time;
- uses a fixed held-out split for step-level evaluation and reports AUROC,
  precision, and recall;
- saves a checkpoint only when both positive and negative step labels are
  available and training completes; and
- writes a complete or blocked artifact containing `status`,
  `training_traces_used`, `step_labels_available`, `prmv1_trained`,
  `prmv1_auroc`, `prmv1_step_precision`, `prmv1_step_recall`,
  `checkpoint_path`, and `honest_verdict`.

### SCENARIO-VERIFY-1423: Lightweight PRM Reports Step-Level Metrics

Given Exp 1395 provides 1508 promoted FoVer trace IDs and Exp 1397 contains
local certificate, scheduler, validator, or repair-localization labels,
When the Exp 1423 PRM training runner executes,
Then the final artifact either reports held-out step-level AUROC, precision,
recall, and an existing checkpoint for a trained CPU classifier, or reports a
blocked verdict with the missing positive/negative step-label counts.

### REQ-VERIFY-1434: FoVer PRM Label Completion V2

The repository shall provide a deterministic, CPU-only PRM v2 label completion
path that:

- writes `results/experiment_1434_fover_prm_label_completion_v2.json` with
  `status="in_progress"` before loading source artifacts or replaying labels;
- reads Exp 1423 and Exp 1395 evidence to identify the promoted trace IDs that
  lacked local step labels in PRM v1;
- recovers only labels whose Exp 1395 promoted trace ID can be mapped back to a
  local FoVer row through deterministic duplicate-ID replay, certificate output,
  scheduler output, validator output, or repair-localization output;
- writes `docs/research/prm_missing_label_ledger_v2.md` with every unrecovered
  promoted trace ID and the concrete blocker for that trace;
- retrains the lightweight PRM on all available local labels and reports AUROC,
  precision, and recall on the fixed held-out split; and
- writes a terminal artifact containing `status`, `missing_labels_before`,
  `missing_labels_filled`, `missing_labels_remaining`,
  `label_blocker_ledger_path`, `training_traces_used`, `prmv2_trained`,
  `prmv2_auroc`, `prmv2_precision`, `prmv2_recall`,
  `headline_label_coverage_ready`, and `honest_verdict`.

The path MUST NOT invent labels and MUST NOT require fresh LLM inference.

### SCENARIO-VERIFY-1434: Ordinal Replay Recovers PRM V1 Missing Labels

Given Exp 1423 reports 478 missing local labels from Exp 1395's 1508 promoted
FoVer traces,
When the Exp 1434 label-completion runner replays Exp 1395's deterministic
FoVer duplicate-ID normalization,
Then every recovered label records the source case and label source, every
unrecovered label is written to the blocker ledger, and
`headline_label_coverage_ready=true` only when all 1508 promoted traces are
covered or the remaining blockers are explicitly outside local recovery scope.

### REQ-VERIFY-1469: HALT and Spilled-Energy Telemetry Diagnostic

The repository shall provide a deterministic, CPU-only telemetry diagnostic for
Exp 1469 that:

- writes `results/experiment_1469_halt_spilled_energy_telemetry_diagnostic.json`
  with `status="in_progress"` before loading Exp 1468 telemetry rows;
- reuses `results/live_sota_telemetry_manifest_1468.jsonl` and MUST NOT require
  fresh LLM inference when that manifest contains top-k logprobs;
- computes HALT-style time-series features including token logprob trend,
  top-k entropy trend, and top-k gap trend from per-token telemetry;
- computes Spilled-Energy-style logprob proxies including spilled-energy and
  marginal-energy estimates from the top-k logprob mass;
- compares feature rank signals against the bounded case labels with AUROC
  when binary labels are available, while recording small-sample caveats;
- checks whether the best rank signal is explained by response length, exact
  answer formatting, JSON-like formatting, or another superficial confound;
- writes `results/live_sota_halt_spilled_diagnostics_1469.json` with per-case
  feature rows and label provenance; and
- writes a terminal artifact containing `status`, `model_specs`,
  `telemetry_rows_loaded`, `halt_features_computed`,
  `spilled_energy_features_computed`, `telemetry_diagnostic_complete`,
  `auroc_or_rank_signal`, `best_signal_name`,
  `length_or_format_confound_checked`, `diagnostic_path`,
  `diagnostic_lineage_preserved`, `diagnostic_lineage_retired`, and
  `honest_verdict`.

### SCENARIO-VERIFY-1469: Small-N Telemetry Signal Retires When Confounded

Given Exp 1468 has live local SOTA top-k telemetry for a bounded FoVer/GSM8K
case set,
When Exp 1469 computes HALT and Spilled-Energy features over the manifest,
Then the diagnostic records per-case features, AUROC or rank-separation
evidence, and label provenance,
And the diagnostic lineage is preserved only when the best logprob feature has
a nontrivial signal that is not matched or exceeded by length or formatting
confounds.

### REQ-VERIFY-1473: Live Telemetry Adversarial Validity Audit

The repository shall provide a deterministic, CPU-only adversarial audit for
Exp 1473 that:

- writes `results/experiment_1473_live_telemetry_adversarial_validity_audit.json`
  with `status="in_progress"` before loading the Exp 1468, Exp 1469, and Exp
  1470 artifacts;
- audits whether the Exp 1468/.113 live logprob telemetry, Exp 1469
  HALT/Spilled-Energy diagnostic, and Exp 1470 BEAVER-lite smoke can satisfy
  their gates through response length, token count, JSON/schema formatting,
  prompt-family membership, or mock/live logprob labeling rather than a real
  verifier signal;
- compares the reported telemetry signal against explicit superficial
  baselines and records their oriented AUROC or gate-equivalent result;
- checks whether BEAVER-lite used mock logprobs and whether the artifact labels
  `mock_logprobs` versus `live_exp1468` unambiguously;
- writes `docs/research-notes/live_telemetry_adversarial_validity_audit.md`
  with pass/fail results for length, format, prompt-family, and mock-logprob
  confounds; and
- writes a terminal artifact containing `status`, `artifacts_audited`,
  `length_confound_checked`, `format_confound_checked`,
  `prompt_family_confound_checked`, `mock_logprob_leakage_checked`,
  `superficial_baseline_results`, `telemetry_validity_verdict`,
  `claim_allowed`, `audit_note_path`, and `honest_verdict`.

The audit MUST set `claim_allowed=false` whenever a superficial baseline
matches or exceeds the proposed diagnostic, when the source diagnostic already
retired its lineage, or when an external-bound smoke passes only by checking a
surface constraint that does not measure semantic verifier correctness.

### SCENARIO-VERIFY-1473: Confounded Telemetry Blocks Headline Claim

Given Exp 1468 reports live top-k telemetry, Exp 1469 reports small-N
HALT/Spilled-Energy rank evidence, and Exp 1470 reports a BEAVER-lite sound
bound,
When Exp 1473 audits these artifacts adversarially,
Then the audit records each confound check, compares superficial baselines
against the proposed signal, verifies BEAVER mock/live labeling, writes the
research note, and allows a headline telemetry claim only when the evidence
cannot pass for superficial or mechanical reasons.

### REQ-VERIFY-1481: Semantic Energy Feasibility Audit

The repository shall provide a deterministic, CPU-only Semantic Energy
feasibility audit for Exp 1481 that:

- writes `results/experiment_1481_semantic_energy_feasibility_audit.json`
  with `status="in_progress"` before loading Exp 1480 telemetry rows;
- reuses `results/live_sota_balanced_telemetry_manifest_1480.jsonl` and MUST
  NOT require fresh LLM inference when that manifest contains top-k/logit
  telemetry;
- computes bounded Semantic Energy-style proxy features from the recorded
  final-token top-k alternatives, including final-logit entropy, top-k
  semantic-cluster proxy, answer-choice energy gap, and per-case uncertainty
  spread;
- computes rank/accuracy metrics for the superficial baselines recorded by Exp
  1480 on the same labels;
- sets `signal_beats_superficial_baselines=true` only when the best semantic
  proxy beats every measured superficial baseline on the same labels;
- writes `results/semantic_energy_features_1481.json` with per-case semantic
  features, baseline scores, and label provenance; and
- writes a terminal artifact containing `status`, `model_specs`,
  `telemetry_rows_loaded`, `semantic_energy_features_computed`,
  `baseline_features_computed`, `semantic_energy_audit_complete`,
  `best_semantic_signal`, `best_superficial_baseline`,
  `signal_beats_superficial_baselines`, `diagnostic_path`, `claim_allowed`,
  `diagnostic_lineage_retired`, and `honest_verdict`.

The audit MUST set `claim_allowed=false` and retire headline telemetry lineage
when the semantic-energy proxy is flat, unavailable, or matched/exceeded by a
superficial baseline.

### SCENARIO-VERIFY-1481: Semantic Signal Must Beat Superficial Baselines

Given Exp 1480 contains balanced live local SOTA rows with top-k alternatives,
logit availability, labels, and recorded superficial baselines,
When Exp 1481 computes Semantic Energy proxy features and evaluates them
against the same labels used for the superficial baselines,
Then the final artifact allows a telemetry claim only when the best semantic
signal has nontrivial oriented rank evidence and strictly beats every
superficial baseline, otherwise it retires the diagnostic lineage for headline
telemetry claims.

### REQ-VERIFY-1474: T-SKM Linear Constraint Projection Smoke

The repository shall provide a deterministic, CPU-only SKM/Kaczmarz-Motzkin
style projection smoke for toy linear certificate constraints that:

- writes `results/experiment_1474_tskm_linear_constraint_projection_smoke.json`
  with `status="in_progress"` before loading or evaluating toy cases;
- supports bounded iterative projection over `Ax <= b` constraints and exact
  equality constraints by transforming each equality into two inequalities;
- evaluates a small deterministic set of toy arithmetic/certificate cases with
  known feasible points;
- compares the projected solution verdicts against existing Carnot, Z3
  arithmetic, and Ising energy checks; and
- writes a terminal artifact containing `status`, `toy_cases_evaluated`,
  `zero_violation_projection`, `max_constraint_violation`,
  `baseline_verifier_agreement`, `projection_iterations_p50`,
  `projection_iterations_p95`, `helper_path`, `tests_run`, and
  `honest_verdict`.

The smoke MUST NOT train a neural model, call an LLM, require GPU hardware, or
revive the retired HardNet++/DSP repair lineage.

### SCENARIO-VERIFY-1474: Projected Toy Linear Certificates Agree With Baselines

Given deterministic toy linear certificate systems with known feasible points,
When the SKM-style projection helper runs from infeasible starting points,
Then each projected solution has zero violation within tolerance, the maximum
constraint violation is reported, Carnot/Z3/Ising baseline verdicts agree with
the projection verdicts, and the experiment artifact records deterministic
iteration quantiles and an honest CPU-only verdict.

### REQ-VERIFY-1475: Static CSR Certificate Automaton Smoke

The repository shall provide a deterministic, CPU-only STATIC-style CSR
automaton smoke for a tiny Carnot certificate schema that:

- writes `results/experiment_1475_static_csr_certificate_automaton_smoke.json`
  with `status="in_progress"` before evaluating certificate cases;
- reuses the smallest existing Carnot certificate parser/schema path as the
  acceptance baseline;
- encodes the bounded accepted certificate strings as CSR-like sparse
  transition arrays without adding an LLM generation path, repair loop, or
  model dependency;
- evaluates exact acceptance equivalence on a deterministic mix of valid and
  invalid certificate strings, reporting false accepts and false rejects; and
- writes a terminal artifact containing `status`, `schema_cases_evaluated`,
  `csr_automaton_path`, `exact_acceptance_equivalent`, `false_accepts`,
  `false_rejects`, `existing_path_latency_ms_p50`, `csr_latency_ms_p50`,
  `speedup_ratio`, `tests_run`, and `honest_verdict`.

The smoke MUST NOT claim general JSON-schema language equivalence beyond the
bounded case set measured in the artifact.

### SCENARIO-VERIFY-1475: CSR Automaton Matches Existing Certificate Parser Cases

Given the existing minimal certificate schema and parser/regex validation path,
When the STATIC CSR automaton smoke evaluates canonical valid certificate
strings and malformed or schema-invalid strings,
Then the CSR automaton and existing path produce identical accept/reject
decisions on every measured case, latency p50 values are reported for both
paths, and the artifact states that the equivalence claim is bounded to the
measured certificate strings.

### REQ-VERIFY-1486: CCTU Executable Constraint Micro-Benchmark

The repository shall provide a deterministic CCTU-style executable constraint
micro-benchmark for Exp 1486 that:

- writes
  `results/experiment_1486_cctu_executable_constraint_microbenchmark.json`
  with `status="in_progress"` before loading source code, validators, or local
  models;
- defines exactly 20 local tool-use cases spanning arithmetic, table
  filtering, string constraints, and graph/path constraints;
- validates each model transcript with deterministic checks for tool-call
  structure, local tool-result consistency, final-answer validity, and verifier
  outcome agreement;
- writes `results/cctu_microbenchmark_manifest_1486.jsonl` with prompt, model
  output, validator result, and verifier result for every evaluated case;
- attempts live local GGUF inference with the mandated SOTA model set
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`), using at least one model for headline
  tool-use rows when runtime and cache state permit and recording blockers for
  any skipped model; and
- writes a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`,
  `executable_constraint_benchmark_ready`, `benchmark_cases`,
  `validators_path`, `manifest_path`, `tool_call_validity_rate`,
  `final_answer_validity_rate`, `verifier_catch_rate`,
  `verifier_false_accept_rate`, `models_used`, `tests_run`, and
  `honest_verdict`.

Legacy small models may be used only for CPU smoke-tests and MUST NOT be
reported as headline tool-use results.

### SCENARIO-VERIFY-1486: Executable Tool-Use Constraints Are Auditable

Given the 20 fixed local CCTU-style prompts and deterministic local tools,
When Exp 1486 evaluates live local SOTA GGUF transcripts,
Then each manifest row records the raw model output, the executable validator
decisions, and the verifier accept/reject outcome, while the final artifact
reports aggregate tool-call validity, final-answer validity, verifier catch
rate, false-accept rate, model provenance, and an honest terminal verdict.

### REQ-VERIFY-1487: V_1 Pairwise Self-Verification vs Energy

The repository shall provide a bounded V_1-style pairwise self-verification
evaluation for Exp 1487 that:

- writes
  `results/experiment_1487_v1_pairwise_self_verification_vs_energy.json`
  with `status="in_progress"` before loading Exp 1486 rows or calling local
  models;
- loads only Exp 1486 deterministic executable-constraint rows and constructs
  candidate answer pairs with one executable-validator-valid answer and one
  executable-validator-invalid answer where possible;
- asks at least one mandated local SOTA GGUF verifier
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`) to select the better answer pairwise,
  while recording blockers for unavailable mandated models;
- scores Carnot energy and BEAVER-style ranking on the same candidate pairs
  using deterministic executable constraint signals when available;
- compares pairwise accuracy against random, response-length, format-validity,
  and energy-ranking baselines; and
- writes `results/v1_pairwise_verification_1487.json` with per-pair decisions,
  scores, baseline decisions, and provenance before completing the terminal
  artifact.

The artifact MUST include `status`, `model_specs`,
`live_sota_model_inference_used`, `pairwise_verification_complete`,
`benchmark_cases_loaded`, `candidate_pairs_evaluated`, `pairwise_accuracy`,
`energy_ranking_accuracy`, `random_baseline_accuracy`,
`superficial_baseline_accuracy`, `pairwise_delta_over_energy`,
`improvement_allowed`, `diagnostic_path`, `tests_run`, and `honest_verdict`.

`improvement_allowed` MUST be true only when pairwise accuracy strictly exceeds
energy-ranking accuracy and the improvement is not matched or exceeded by the
best superficial baseline on the same pairs. Legacy small models may be used
only for CPU smoke-tests and MUST NOT be reported as headline pairwise verifier
results.

### SCENARIO-VERIFY-1487: Pairwise Selection Must Beat Superficial Baselines

Given Exp 1486 has a complete executable CCTU manifest with live local SOTA
rows,
When Exp 1487 constructs valid/invalid answer pairs, obtains pairwise choices
from at least one mandated local SOTA GGUF verifier, and scores deterministic
energy and superficial baselines on the same pair set,
Then the diagnostic records every per-pair choice and score, the terminal
artifact reports all required fields, and `improvement_allowed=true` only when
the pairwise verifier beats energy ranking and is not explained by response
length or format-validity baselines.

### REQ-VERIFY-1494: Bounded ConstrainPrompt Validator Compiler Audit

The repository shall provide a bounded ConstrainPrompt-style prompt-to-validator
compiler audit for Exp 1494 that:

- writes
  `results/experiment_1494_constrainprompt_validator_compiler_audit.json` with
  `status="in_progress"` before loading local models or compiling validators;
- builds exactly 30 fixed CCTU-style prompts from Exp 1486 cases plus new
  arithmetic, JSON-schema, simple-code, and graph/path cases;
- asks at least one mandated local SOTA GGUF model
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`) to propose constraint fields or validator
  skeletons, while recording a terminal blocker when the mandated model path
  cannot load;
- compiles validators only through a deterministic safe DSL and fixed Python
  validator functions over restricted inputs, with no arbitrary `eval`,
  `exec`, or model-generated code execution path;
- tests every compiled validator against at least one known-good and one
  known-bad output and records compile failures, manual-review markers, false
  accepts, and false rejects;
- writes `results/constrainprompt_validator_manifest_1494.jsonl` with one row
  per prompt; and
- writes a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `validator_compiler_ready`,
  `prompts_attempted`, `validator_skeletons_generated`,
  `validators_compiled`, `validator_compile_rate`, `known_good_pass_rate`,
  `known_bad_reject_rate`, `verifier_false_accept_rate`,
  `manual_review_required_count`, `validator_manifest_path`, `models_used`,
  `gpu_probe`, `blockers`, and `honest_verdict`.

`validator_compiler_ready` MUST be true only when compile metrics and
false-accept metrics are present, at least one mandated live local SOTA GGUF
contributed skeleton rows, and the compiler did not introduce an arbitrary-code
execution path. Legacy small models may be used only for CPU smoke-tests and
MUST NOT count as headline validator compiler evidence.

### SCENARIO-VERIFY-1494: Prompt-To-Validator Audit Reports Safe DSL Metrics

Given the fixed 30-prompt Exp 1494 CCTU-style prompt set and a mandated local
SOTA GGUF skeleton proposer,
When the bounded ConstrainPrompt compiler audit runs on the run date `20260507`,
Then it writes the in-progress artifact first
And it writes one manifest row per prompt with prompt text, model skeleton
provenance, compiled DSL, known-good result, known-bad result, manual-review
status, and false-accept status
And it reports compile rate, known-good pass rate, known-bad reject rate, and
false-accept rate in the terminal artifact
And it sets `validator_compiler_ready=true` only when the safe DSL metrics are
present and no arbitrary-code execution path was introduced.

### REQ-VERIFY-1495: CPU-Only Interwhen Monitor Prototype Replay

The repository shall provide a CPU-only interwhen-style monitor prototype for
Exp 1495 that replays existing CCTU trigger-certificate and validator-compiler
artifacts without generating new LLM rows.

- REQ-VERIFY-1495-1: The workflow SHALL write
  `results/experiment_1495_interwhen_monitor_prototype.json` with
  `status="in_progress"` before loading gated upstream artifacts or emitting
  monitor events.
- REQ-VERIFY-1495-2: The workflow SHALL require both
  `results/experiment_1493_trigger_token_certificate_export_v1.json` and
  `results/experiment_1494_constrainprompt_validator_compiler_audit.json` to
  be complete and ready before setting `gated_inputs_present=true`; otherwise
  it SHALL write a terminal gated artifact with concrete blockers.
- REQ-VERIFY-1495-3: Certificate rows and validator rows SHALL be converted
  into replayable trace states with deterministic token offsets or synthetic
  polling intervals and no fresh model generation.
- REQ-VERIFY-1495-4: Each poll SHALL run deterministic certificate,
  validator, and verifier checks, emit exactly one JSONL monitor event, and
  mark whether the monitor interrupts only because an error was detected.
- REQ-VERIFY-1495-5: The workflow SHALL write
  `results/interwhen_monitor_events_1495.jsonl` with one event per poll and
  SHALL report detection count, interruptions, false interruptions, false
  accepts, and verifier false-accept rate.
- REQ-VERIFY-1495-6: The terminal artifact SHALL include `status`,
  `monitor_intervention_ready`, `gated_inputs_present`, `traces_replayed`,
  `polling_interval_tokens`, `monitor_events_emitted`, `errors_detected`,
  `interruptions_triggered`, `false_interruptions`,
  `verifier_false_accept_rate`, `monitor_event_manifest_path`, `blockers`, and
  `honest_verdict`.

`monitor_intervention_ready` MUST be true only when the event manifest exists,
at least one real recorded error is detected, and the verifier false-accept
rate is zero.

### SCENARIO-VERIFY-1495: Replayed Monitor Events Gate Intervention Readiness

Given complete Exp 1493 trigger-certificate evidence and complete Exp 1494
safe-DSL validator evidence,
When the Exp 1495 CPU-only interwhen replay runs on the run date `20260507`,
Then it emits one monitor event per synthetic poll over the replayed trace
states
And each event records the case ID, poll offset, deterministic check outcomes,
error detection status, interrupt decision, and false-interruption status
And it sets `monitor_intervention_ready=true` only when at least one recorded
error is detected, no false interruptions are triggered, and verifier
false-accept rate remains zero.

### REQ-VERIFY-1496: HoVer Safe-Prefix Continuation Audit

The repository shall provide a bounded HoVer-style safe-prefix continuation
audit for Exp 1496 that:

- writes
  `results/experiment_1496_hover_safe_prefix_continuation_audit.json` with
  `status="in_progress"` before loading monitor events, validators, or local
  models;
- requires the Exp 1495 monitor event manifest to exist before attempting
  continuation, and writes a terminal blocked artifact with concrete blockers
  when the gated monitor evidence is unavailable;
- defines a deterministic last-safe-prefix selection rule from monitor events:
  for each selected CCTU trigger-certificate row, choose the earliest
  interrupting monitor event for that case and lane, then keep only the
  free-form reasoning plus trigger-token boundary before the unsafe
  certificate suffix;
- asks at least one mandated local SOTA GGUF model
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`) to continue from those selected prefixes,
  while recording a terminal blocker when no mandated model can load;
- evaluates matched no-continuation, safe-prefix continuation, and full
  regeneration rows for the same selected cases with deterministic CCTU and
  compiled safe-DSL validators; and
- writes `results/safe_prefix_continuations_1496.jsonl` with one row per
  case/baseline and a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `safe_prefix_continuation_ready`,
  `cases_attempted`, `continuations_completed`,
  `baseline_validator_pass_rate`, `safe_prefix_validator_pass_rate`,
  `full_regeneration_validator_pass_rate`, `verifier_false_accept_rate`,
  `last_safe_prefix_selection_rule`, `continuation_manifest_path`,
  `models_used`, `gpu_probe`, `blockers`, and `honest_verdict`.

`safe_prefix_continuation_ready` MUST be true only when the pass-rate and
false-accept metrics are present and at least one mandated live local SOTA GGUF
contributed safe-prefix continuation rows. Legacy small models may be used only
for CPU smoke-tests and MUST NOT count as headline continuation evidence.

### SCENARIO-VERIFY-1496: Safe-Prefix Continuation Reports Matched Validator Rates

Given Exp 1495 has emitted monitor events over CCTU trigger-certificate rows,
When the Exp 1496 audit selects the last safe prefix for interrupted cases and
continues from that prefix on the run date `20260507`,
Then it writes one manifest row for each no-continuation, safe-prefix, and
full-regeneration evaluation
And every row records the selected prefix, model provenance, deterministic
CCTU validation result, compiled-validator result, final validator decision,
and false-accept status
And the terminal artifact reports matched pass rates and verifier false-accept
rate before setting `safe_prefix_continuation_ready=true`.

### REQ-VERIFY-1499: Verifier Ensemble DRY And Conditional Orthogonality Audit

The repository shall provide a deterministic Exp 1499 audit over the active
post-.114 verifier surfaces: BEAVER-lite bounds, CCTU executable validators,
energy/localization, query-time memory-policy checks, and structured verdict
records. The audit MUST NOT count retired Semantic Energy or V_1 headline
signals as active verifiers.

The audit shall:

- write
  `results/experiment_1499_verifier_ensemble_dry_orthogonality_v2.json` with
  `status="in_progress"` before loading source artifacts;
- inventory active verifier surfaces from code and checked-in result artifacts;
- build a bounded case table with pass/fail/abstain labels from available
  verifier outputs;
- compute pairwise agreement, pairwise conditional acceptance rates,
  redundant verifier pairs, and a conservative `k_effective_estimate`;
- identify duplicated verifier wrappers or conditional duplicates that should
  be merged, retired, or demoted;
- prefer deterministic validators and energy ranking ahead of generative
  pairwise self-verification in recommendations; and
- write a terminal artifact containing `status`,
  `orthogonality_matrix_written`, `verifiers_audited`, `cases_evaluated`,
  `conditional_acceptance_matrix`, `redundant_verifier_pairs`,
  `k_effective_estimate`, `deterministic_first_recommendations`,
  `retire_or_keep_decisions`, `blockers`, and `honest_verdict`.

### SCENARIO-VERIFY-1499: Conditional Matrix Flags Duplicate Validators

Given complete Exp 1482 BEAVER-lite, Exp 1486 CCTU, Exp 1490 localization,
Exp 1484 query-time memory, and structured-verdict artifacts,
When the Exp 1499 audit builds the bounded case table,
Then it writes a terminal artifact with a conditional acceptance matrix for
the active verifier labels,
And it reports redundant verifier pairs when two labels accept exactly the
same observed cases or one label's accepts are conditionally contained in the
other at the configured redundancy threshold,
And it excludes retired Semantic Energy and V_1 headline signals from the
active verifier inventory.

### REQ-VERIFY-1500: Latent-Vs-Deterministic Discipline Gate

The repository shall provide a deterministic Exp 1500 discipline gate that
classifies post-.114 verifier and telemetry signals by the strongest claim
surface they may support: headline evidence, auxiliary ranking evidence,
triage evidence, or retired/no-claim evidence.

The gate shall:

- write
  `results/experiment_1500_latent_deterministic_discipline_gate.json` with
  `status="in_progress"` before loading gated upstream artifacts;
- require Exp 1499 to have written an orthogonality matrix before setting
  `discipline_gate_ready=true`, otherwise write a terminal gated artifact with
  concrete blockers;
- read the Exp 1481 Semantic Energy retirement, Exp 1487 V_1 pairwise
  self-verification comparison, and Exp 1499 deterministic-first
  recommendations before assigning signal roles;
- publish `ops/latent_deterministic_discipline_gate_1500.md` with policy
  tables for headline, auxiliary ranking, triage, and retired/no-claim
  signals;
- allow latent, energy-like, probabilistic, or LLM-derived signals to influence
  decisions only after deterministic validator comparison, superficial-baseline
  comparison, held-out calibration, and false-accept accounting are present;
- require deterministic validators to dominate whenever a deterministic
  validator is applicable to the same claim; and
- write a terminal artifact containing `status`, `discipline_gate_ready`,
  `gated_inputs_present`, `signal_classes_audited`,
  `headline_allowed_signals`, `auxiliary_allowed_signals`, `retired_signals`,
  `deterministic_first_rules`, `superficial_baseline_required_rules`,
  `ops_note_path`, `blockers`, and `honest_verdict`.

`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1500: Discipline Gate Demotes Confounded Latent Signals

Given Exp 1499 contains a written orthogonality matrix, Exp 1481 retires
Semantic Energy because it does not beat superficial baselines, and Exp 1487
shows V_1 pairwise self-verification does not beat deterministic energy
ranking,
When the Exp 1500 discipline gate builds the policy artifact,
Then deterministic executable validators and conservative deterministic bounds
are the only headline-allowed signals, energy/localization and calibrated
probabilistic signals are limited to auxiliary or triage roles, Semantic
Energy headline telemetry and V_1 pairwise self-verification are retired from
claims, and the terminal artifact records the required acceptance rules and an
allowed-prefix honest verdict.

### REQ-VERIFY-1501: GNNVerifier Plan-Graph Energy Adapter

The repository shall provide a deterministic CPU-only Exp 1501 adapter that
converts bounded CCTU tool-use cases into directed plan graphs and scores
injected dependency faults with graph-risk energy. The adapter MUST NOT claim
trained GNN performance unless a trained model and honest train/eval split are
implemented.

The adapter shall:

- write
  `results/experiment_1501_gnnverifier_plan_graph_energy_adapter.json` with
  `status="in_progress"` before loading CCTU rows or scoring faults;
- select a bounded deterministic subset of CCTU tool-use cases and represent
  each as directed nodes and edges with node types, edge types, tool
  dependencies, and expected outputs;
- inject deterministic dependency faults including missing edges, wrong tool
  input type, missing intermediate, wrong ordering, and dangling output;
- score each faulty graph with deterministic graph-risk energy and localize the
  highest-risk node and edge when the fault exposes a node or edge target;
- compare node and edge top-1 localization against random and length/degree
  baselines on the same fault rows;
- write `results/plan_graph_energy_manifest_1501.jsonl` with one row per
  trace/fault; and
- write a terminal artifact containing `status`, `plan_graph_energy_ready`,
  `traces_converted`, `injected_graph_faults`,
  `node_localization_top1_rate`, `edge_localization_top1_rate`,
  `random_baseline_top1_rate`, `length_baseline_top1_rate`,
  `graph_energy_beats_baselines`, `adapter_manifest_path`, `blockers`, and
  `honest_verdict`.

`plan_graph_energy_ready` MUST be true only when graph conversion rows,
fault-injection rows, and baseline comparison metrics are written.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1501: Plan-Graph Energy Localizes Injected Dependency Faults

Given the fixed Exp 1486 CCTU tool-use cases and deterministic local expected
tool outputs,
When Exp 1501 converts a bounded subset into plan graphs on the run date
`20260507`,
Then each manifest row records the trace ID, graph node and edge attributes,
injected fault type, expected risky node or edge, graph-risk ranking, random
baseline credit, length or degree baseline credit, and localization outcome
And the terminal artifact sets `plan_graph_energy_ready=true` only when at
least one trace and fault row are written, baseline rates are present, and the
deterministic graph energy beats both baselines without claiming trained GNN
performance.

### REQ-VERIFY-1507: Safe-DSL Verifier Induction Pack

The repository shall provide a bounded AutoPyVerifier-inspired verifier
induction pack for Exp 1507 that keeps generated verifier proposals outside
the Python execution trust boundary.

The pack shall:

- write
  `results/experiment_1507_autopyverifier_safe_dsl_induction_pack.json` with
  `status="in_progress"` before loading manifests, models, or candidates;
- require both `results/cctu_trigger_certificates_1493.jsonl` and
  `results/constrainprompt_validator_manifest_1494.jsonl`, and write a
  terminal artifact with concrete missing-path blockers if either is absent;
- load labeled certificate rows and compiled-validator audit rows from those
  manifests, preserving deterministic pass/fail labels for false-accept
  accounting;
- ask at least one mandated local SOTA GGUF model
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`) for safe-DSL verifier skeletons only,
  while rejecting arbitrary Python, filesystem access, imports, eval/exec,
  network calls, and non-deterministic logic;
- compile candidate skeletons only through a minimal safe-DSL compiler or the
  existing safe validator compiler, recording compile failures explicitly;
- search for a compact verifier set that maximizes labeled-row coverage while
  preserving zero false accepts on available labels;
- write `results/safe_dsl_verifier_induction_1507.jsonl` with one row per
  candidate and a final selected-set summary; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `verifier_induction_ready`,
  `labeled_rows_loaded`, `candidate_verifiers_proposed`,
  `candidate_verifiers_compiled`, `verifier_compile_rate`,
  `verifier_set_size`, `verifier_coverage_rate`,
  `verifier_false_accept_rate`, `baseline_validator_coverage_rate`,
  `induction_manifest_path`, `models_used`, `gpu_probe`, `blockers`, and
  `honest_verdict`.

`verifier_induction_ready` MUST be true only when at least one mandated live
local SOTA GGUF model proposed candidates, at least one candidate compiled, and
`verifier_false_accept_rate` is reported. Legacy small models may be used only
for CPU smoke-tests and MUST NOT count as headline verifier-induction evidence.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1507: Safe-DSL Induction Preserves Zero False Accepts

Given the Exp 1493 CCTU certificate manifest and the Exp 1494 safe validator
compiler manifest exist on the run date `20260507`,
When Exp 1507 asks a mandated local SOTA GGUF model for safe-DSL verifier
skeletons, compiles them, and searches over the labeled rows,
Then arbitrary generated Python and unsafe capabilities are rejected before
scoring
And each candidate manifest row records provenance, compile status, compile
failure reason, coverage, false accepts, and accepted labeled row IDs
And the terminal selected-set summary reports the compact zero-false-accept
set before the artifact sets `verifier_induction_ready=true`.

### REQ-VERIFY-1508: Trigger+Grammar Certificate Decoder Audit

The repository shall provide a bounded trigger-token plus grammar/GBNF
certificate decoder audit for Exp 1508 that compares runtime-constrained
certificate tails against Exp 1493 schema-only certificate parsing.

The audit shall:

- write
  `results/experiment_1508_trigger_grammar_certificate_decoder_audit.json`
  with `status="in_progress"` before loading gates, manifests, models, or
  grammar backends;
- require
  `results/experiment_1507_autopyverifier_safe_dsl_induction_pack.json` to
  report `verifier_induction_ready=true`, and write a terminal gated artifact
  when that prerequisite is absent or false;
- load Exp 1493 CCTU certificate rows and the Exp 1507 selected verifier set
  where available, preserving schema-only parse, validation, and false-accept
  accounting for comparison;
- resolve mandated local SOTA GGUF model specs through `cached_sota_pair()` or
  the established SOTA GGUF cache resolver, without using legacy small models
  for headline decoder evidence;
- run at least one mandated local SOTA GGUF model for trigger+grammar decoder
  rows when a local grammar backend can enforce a bounded certificate grammar;
- record a concrete grammar backend name and blocker when runtime grammar
  enforcement is unavailable, falling back only to parse-only diagnostic rows;
- write `results/trigger_grammar_certificates_1508.jsonl` with one row per
  case, model, and decoder mode;
- report trigger-token presence, grammar parse, schema-only parse, grammar
  validation, schema-only validation, and verifier false-accept rates; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `certificate_decoder_ready`,
  `gated_inputs_present`, `cases_attempted`, `grammar_backend`,
  `trigger_token_presence_rate`, `grammar_parse_rate`,
  `schema_only_parse_rate`, `grammar_validation_rate`,
  `schema_only_validation_rate`, `verifier_false_accept_rate`,
  `decoder_manifest_path`, `models_used`, `gpu_probe`, `blockers`, and
  `honest_verdict`.

`certificate_decoder_ready` MUST be true only when gated inputs are present,
at least one mandated live local SOTA GGUF grammar row exists, and both parse
and validation metrics are reported. Legacy small models may be used only for
CPU smoke-tests and MUST NOT count as headline decoder evidence.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1508: Trigger+Grammar Rows Compare Against Schema-Only Rows

Given Exp 1507 verifier induction is ready and Exp 1493 CCTU certificate rows
exist on the run date `20260507`,
When Exp 1508 runs a mandated local SOTA GGUF model through a trigger-token
reasoning phase followed by a grammar-bounded certificate phase,
Then each manifest row records case ID, decoder mode, model provenance,
grammar backend, trigger-token presence, parser result, deterministic
validation result, and false-accept status
And the terminal artifact compares trigger+grammar rates against Exp 1493
schema-only rates before setting `certificate_decoder_ready=true`.

### REQ-VERIFY-1509: Executable Monitor Runtime Adapter

The repository shall provide a reusable CPU-only executable monitor runtime
adapter for Exp 1509 that normalizes already-recorded monitor, safe-prefix,
safe-DSL verifier, and trigger+grammar certificate artifacts into one
replayable manifest without generating new LLM rows.

The adapter shall:

- write
  `results/experiment_1509_executable_monitor_runtime_adapter.json` with
  `status="in_progress"` before loading upstream gates or manifest rows;
- require Exp 1507 verifier induction and Exp 1508 trigger+grammar certificate
  decoder artifacts to be complete and ready before setting
  `gated_inputs_present=true`, otherwise writing a terminal gated artifact
  with concrete blockers;
- inventory the Exp 1495 monitor event manifest, Exp 1496 safe-prefix
  continuation manifest, Exp 1507 safe-DSL verifier induction manifest, and
  Exp 1508 trigger+grammar certificate manifest;
- define a small normalized event schema with event schema version, replay
  order, source experiment, source path, source line, source row ID, case ID,
  event kind, validation status, verifier false-accept status, and provenance
  fields;
- validate every normalized event against that schema before writing the
  replay manifest;
- link safe-prefix continuation rows to monitor events only when recorded
  event IDs, token offsets, or unambiguous case IDs match, and record unmatched
  rows without inventing links;
- write `results/executable_monitor_events_1509.jsonl` with one normalized
  event per source manifest row; and
- write a terminal artifact containing `status`, `monitor_runtime_ready`,
  `gated_inputs_present`, `events_loaded`, `events_normalized`,
  `event_schema_version`, `verifier_false_accept_rate`,
  `safe_prefix_events_linked`, `monitor_event_manifest_path`,
  `adapter_tests_run`, `blockers`, and `honest_verdict`.

`monitor_runtime_ready` MUST be true only when the replay manifest exists, all
loaded events normalize successfully, the Exp 1507 and Exp 1508 gates are
ready, and `verifier_false_accept_rate` remains reported. `honest_verdict`
MUST begin with one of `complete:`, `complete_`, `success:`, `success_`,
`passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1509: Runtime Adapter Replays Normalized Monitor Events

Given complete Exp 1507 verifier induction evidence, complete Exp 1508
trigger+grammar certificate evidence, and available .115/.116 monitor,
safe-prefix, verifier, and certificate manifests on the run date `20260507`,
When the Exp 1509 adapter loads and normalizes those recorded rows,
Then it writes exactly one normalized runtime event for each source manifest
row
And every event carries deterministic replay order, source provenance,
validation status, false-accept status, and a stable schema version
And safe-prefix rows carry a monitor link only when a recorded event ID, token
offset, or unambiguous case ID match exists
And the terminal artifact sets `monitor_runtime_ready=true` only when the
manifest exists, all events validate, and verifier false-accept rate is
reported.

### REQ-VERIFY-1510: Plan-Graph Structural Contract Gate

The repository shall provide a deterministic CPU-only Exp 1510 structural
contract gate that checks CCTU plan graphs before execution. The gate moves
Exp 1501's post-hoc dependency localization signal into a pre-execution
contract check and MUST NOT invoke external tools, LLMs, or learned graph
models while classifying violations.

The gate shall:

- write
  `results/experiment_1510_plan_graph_structural_contract_gate.json` with
  `status="in_progress"` before loading Exp 1501 or Exp 1509 artifacts;
- load Exp 1501 plan-graph cases and normalized Exp 1509 runtime events when
  available, preserving concrete source blockers without fabricating rows;
- define a small structural contract schema covering required prerequisites,
  acquisition paths from tool calls to final answers, tool ordering, required
  object acquisition, and incompatible API operations;
- evaluate known-good graphs and deterministic injected-violation graphs with
  exact structural checks rather than probabilistic scoring;
- classify violations by contract family and compare detection against
  deterministic random and length baselines when injected violations exist;
- write `results/plan_graph_structural_contracts_1510.jsonl` with one row per
  graph and contract result; and
- write a terminal artifact containing `status`,
  `structural_contract_gate_ready`, `plan_graphs_checked`,
  `contracts_defined`, `violations_injected`, `violations_detected`,
  `false_accept_rate`, `false_reject_rate`,
  `random_baseline_detection_rate`, `length_baseline_detection_rate`,
  `contract_manifest_path`, `blockers`, and `honest_verdict`.

`structural_contract_gate_ready` MUST be true only when the contract manifest
exists, at least one graph is checked, at least one contract is defined, and
`false_accept_rate` is reported. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1510: Contract Gate Rejects Structural Plan Violations

Given Exp 1501 plan graphs and any available Exp 1509 normalized runtime
events on the run date `20260507`,
When Exp 1510 defines structural contracts and evaluates known-good plus
injected-violation plan graphs,
Then known-good graphs pass without false rejects,
And injected missing-prerequisite, broken-acquisition-path, wrong-ordering,
missing-object-acquisition, and incompatible-operation graphs are rejected
with deterministic violation classifications,
And each manifest row records graph provenance, contract family, expected
violation status, detected violation status, classifier outcome, random
baseline outcome, length baseline outcome, and contract evidence before the
terminal artifact sets `structural_contract_gate_ready=true`.

### REQ-VERIFY-1520: Runtime-Contract E2E Harness

The repository shall provide a deterministic CPU-only runtime-contract E2E
harness for Exp 1520 that loads the Exp 1507 safe-DSL verifier induction pack,
Exp 1508 trigger+grammar certificate decoder audit, Exp 1509 executable
monitor runtime adapter, and Exp 1510 plan-graph structural contract gate into
one acceptance manifest without generating new LLM rows.

The harness shall:

- write `results/experiment_1520_runtime_contract_e2e_harness.json` with
  `status="in_progress"` before loading source artifacts or manifest rows;
- load the Exp 1507, Exp 1508, Exp 1509, Exp 1510, and Exp 1511 terminal JSON
  artifacts and the Exp 1507, Exp 1508, Exp 1509, and Exp 1510 source JSONL
  manifests, writing a terminal blocker with concrete missing paths when any
  required source cannot be resolved;
- define a normalized contract-case schema containing prompt or case ID,
  proposed output, certificate parse result, safe-DSL verifier result, monitor
  event result, structural-contract result, expected label when explicitly
  available, and final deterministic accept/reject;
- write `results/runtime_contract_e2e_manifest_1520.jsonl` with one normalized
  contract-case row per linked source row and a final summary row;
- compute false accepts and false rejects only for normalized rows with an
  explicit expected label, never by inferring success from LLM prose;
- report linked-row counts separately for safe-DSL, grammar certificate,
  monitor event, and structural-contract families; and
- write a terminal artifact containing `status`, `runtime_contract_e2e_ready`,
  `source_artifacts_loaded`, `contract_cases_total`,
  `safe_dsl_cases_linked`, `grammar_certificate_cases_linked`,
  `monitor_events_linked`, `structural_contract_cases_linked`,
  `false_accept_count`, `false_accept_rate`, `false_reject_count`,
  `runtime_contract_manifest_path`, `focused_tests_passed`, `blockers`, and
  `honest_verdict`.

`runtime_contract_e2e_ready` MUST be true only when all mandatory source
artifacts load, at least one row from each of the four .116 contract families
is linked, `false_accept_rate` is reported as `0.0`, and the focused harness
tests have passed. `honest_verdict` MUST begin with one of `complete:`,
`complete_`, `success:`, `success_`, `passed:`, `passed_`, `shipped:`, or
`shipped_`.

### SCENARIO-VERIFY-1520: Runtime Contract Ledger Combines .116 Families

Given complete Exp 1507 safe-DSL, Exp 1508 certificate, Exp 1509 monitor, and
Exp 1510 structural-contract artifacts on the run date `20260508`,
When Exp 1520 resolves source manifests, normalizes rows, and writes the E2E
manifest,
Then every contract-case row records source provenance plus certificate,
safe-DSL, monitor, and structural-contract result fields
And the false-accept ledger counts only rows whose source artifact provides an
explicit expected label
And the terminal artifact sets `runtime_contract_e2e_ready=true` only when each
.116 contract family contributes at least one linked row and the reported
false-accept rate is exactly zero.

### REQ-VERIFY-1521: Live SOTA Contract-Guided Repair

The repository shall provide an Exp 1521 live local-SOTA repair adapter that
uses the Exp 1520 runtime-contract E2E manifest as deterministic authority and
compares baseline generation, grammar-only repair, and draft-conditioned
contract-guided repair without using legacy small models for headline rows.

The adapter shall:

- write `results/experiment_1521_live_sota_contract_guided_repair_v1.json`
  with `status="in_progress"` before model loading or row generation;
- load `results/runtime_contract_e2e_manifest_1520.jsonl` and select a bounded
  subset of deterministic, explicitly labeled contract-failing or
  contract-marginal rows;
- resolve mandated local SOTA GGUFs using `cached_sota_pair()` or the same local
  cache pattern, and write a terminal blocker if no mandated SOTA GGUF completes
  live inference;
- generate one baseline, grammar-only, and draft-conditioned output for each
  selected case and model, recording a raw-output hash or path per row;
- validate every generated output by converting it into an Exp 1520
  contract-case row and computing deterministic false-accept outcomes with the
  runtime-contract ledger helpers, never by trusting LLM prose;
- write `results/live_contract_guided_repair_1521.jsonl` with one row per case,
  model, and mode containing model provenance, mode, output hash or raw path,
  deterministic validator outcome, and repair outcome;
- report `baseline_accept_rate`, `grammar_only_accept_rate`,
  `draft_conditioned_accept_rate`, `repair_accept_rate_delta`,
  `false_accept_count`, and `false_accept_rate`; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `contract_guided_repair_ready`,
  `e2e_cases_loaded`, `repair_cases_attempted`, `models_used`, `gpu_probe`,
  `repair_manifest_path`, `blockers`, and `honest_verdict`.

`contract_guided_repair_ready` MUST be true only when at least one mandated SOTA
GGUF produced live inference rows, repair metrics are reported, and
`false_accept_rate` is exactly `0.0`. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1521: Draft-Conditioned Repair Stays Contract-Grounded

Given a complete Exp 1520 runtime-contract E2E manifest with explicitly labeled
reject or marginal rows on the run date `20260508`,
When Exp 1521 runs mandated local SOTA GGUF generation in baseline,
grammar-only, and draft-conditioned modes,
Then the JSONL repair manifest records a deterministic validator outcome for
each model, case, and mode
And the aggregate false-accept ledger is computed from Exp 1520 contract-case
rows rather than from generated prose
And the terminal artifact sets `contract_guided_repair_ready=true` only when
live SOTA inference completed and the reported false-accept rate remains zero.

### REQ-VERIFY-1535: XGrammar/ABS Contract Decoder Adapter

The repository shall provide an Exp 1535 contract decoder adapter that compares
the current grammar-only/post-decode runtime-contract path against an
XGrammar-2-compatible adapter interface with ABS-style DFA token masking for
bounded regular contract constraints.

The adapter shall:

- write `results/experiment_1535_xgrammar_abs_contract_decoder_adapter.json`
  with `status="in_progress"` before loading source manifests, probing
  optional grammar packages, or resolving models;
- load the Exp 1520 runtime-contract E2E manifest and select a bounded case set
  containing certificate, safe-DSL validator, monitor-event, and structural
  contract families when those families are available;
- probe whether a local XGrammar-compatible package is importable, and when it
  is absent, provide the same adapter interface through a deterministic
  ABS-style DFA mask simulation for regular contract fields;
- resolve mandated local SOTA GGUF model specs through the established
  `cached_sota_pair()` cache pattern where runtime inference is attempted, and
  mark any legacy small model rows as smoke-test-only and excluded from
  headline metrics;
- compare baseline grammar-only/post-decode outputs with automata-guided
  outputs on parse rate, contract accept rate, latency delta, and deterministic
  false-accept rate;
- hand every parsed output to the existing Exp 1520 deterministic validators
  before treating a row as accepted; and
- write a terminal artifact containing `status`, `milestone`,
  `contract_decoder_adapter_ready`, `model_specs`,
  `live_sota_model_inference_used`, `cases_attempted`,
  `baseline_parse_rate`, `automata_parse_rate`,
  `baseline_contract_accept_rate`, `automata_contract_accept_rate`,
  `latency_delta_seconds`, `false_accept_rate`, `xgrammar_available`,
  `abs_dfa_masks_used`, `adapter_path`, `focused_tests_passed`, and
  `honest_verdict`.

`contract_decoder_adapter_ready` MUST be true only when at least one case from
each available runtime-contract family is evaluated, automata metrics are
reported, deterministic validator handoff runs, and `false_accept_rate` is
exactly `0.0`. `honest_verdict` MUST begin with one of `complete:`,
`complete_`, `success:`, `success_`, `passed:`, `passed_`, `shipped:`, or
`shipped_`.

### SCENARIO-VERIFY-1535: Automata Decoder Masks Invalid Contract Prefixes

Given the Exp 1520 runtime-contract E2E manifest contains certificate,
safe-DSL, monitor-event, and structural contract rows on the run date
`20260508`,
When Exp 1535 compiles bounded DFA masks for the selected contract cases and
compares grammar-only/post-decode decoding against automata-guided decoding,
Then malformed or case-mismatched baseline outputs may fail parsing before
validation
And automata-guided outputs preserve only valid bounded JSON contract fields
before deterministic validator handoff
And the terminal artifact reports parse-rate, contract-accept-rate, latency,
and false-accept metrics without counting legacy small-model smoke rows as
headline evidence.

### REQ-VERIFY-1580: DCCD/JSONSchemaBench SOTA Structured-Output Smoke

The repository shall provide an Exp 1580 bounded structured-output smoke test
for Carnot verifier-output schemas using a JSONSchemaBench-style schema slice
and draft-conditioned constrained decoding.

The smoke test shall:

- write
  `results/experiment_1580_dccd_jsonschemabench_sota_structured_output_smoke.json`
  with `status="in_progress"` before resolving models or running decoders;
- define `MODEL_SPECS` by calling `cached_sota_pair(gpu_indices=(0, 1))`
  where local SOTA GGUF inference is attempted, and record exact resolved
  model paths and hub IDs;
- include at least one mandated SOTA GGUF from
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF` in `models_used` when live rows complete;
- evaluate a bounded case slice that includes both JSONSchemaBench-style
  object constraints and Carnot verifier-output schemas;
- compare unconstrained draft output, standard constrained decoding, and DCCD
  output where local runtime support is available;
- report strict schema validity, semantic verifier correctness, latency,
  projection-tax proxy delta, false accepts, and whether a legacy tiny-model
  fallback path was used; and
- write a terminal artifact containing `status`, `MODEL_SPECS`, `models_used`,
  `used_mandated_sota_gguf`, `legacy_tiny_model_fallback_used`, `n_schemas`,
  `strict_schema_validity_rate`, `semantic_correctness_rate`,
  `false_accept_count`, `projection_tax_proxy_delta`,
  `dccd_jsonschema_smoke_complete`, and `honest_verdict`.

Legacy tiny-model fallback rows MUST be excluded from headline metrics and MUST
NOT set `used_mandated_sota_gguf=true`. `dccd_jsonschema_smoke_complete` MUST
be true only when at least one mandated SOTA GGUF row completes and the DCCD
mode reports strict schema validity and semantic correctness for every selected
schema without false accepts.

### SCENARIO-VERIFY-1580: DCCD Output Remains Schema-Strict And Semantically Checked

Given cached SOTA GGUFs are available through
`cached_sota_pair(gpu_indices=(0, 1))` and the selected Carnot verifier-output
schemas include deterministic semantic targets,
When Exp 1580 evaluates unconstrained draft, constrained decoding, and DCCD
outputs on the bounded JSONSchemaBench-style slice,
Then the terminal artifact records exact models used, per-mode schema and
semantic metrics, latency and projection-tax proxy deltas, and zero DCCD false
accepts before setting `dccd_jsonschema_smoke_complete=true`.

### REQ-VERIFY-1537: BEAVER-Lite Prefix-Bound Contract Audit

The repository shall provide an Exp 1537 BEAVER-lite prefix-bound audit that
uses the Exp 1535 contract decoder adapter rows and Exp 1520 runtime-contract
rows to produce deterministic routing-risk bounds for selected contract and
automata-decoder prefixes.  The audit shall never treat BEAVER bounds,
logprobs, or top-k mass as acceptance authority; deterministic runtime-contract
validators remain the final source of false-accept truth.

The audit shall:

- write `results/experiment_1537_beaver_prefix_bound_contracts_v3.json` with
  `status="in_progress"` before source manifest loading or prefix evaluation;
- select bounded runtime-contract and automata-decoder cases from Exp 1535,
  preserving contract case IDs, source families, decoder modes, and
  deterministic accept/reject labels;
- build a bounded prefix trie/frontier over canonical contract JSON targets and
  report monotone structural upper bounds for unexplored or invalid prefixes;
- use token logprob and top-k telemetry when a selected decoder row exposes it,
  otherwise set `token_logprob_available=false`, `topk_available=false`, and
  record the structural-bound simulation path explicitly;
- rank high-risk instances by BEAVER-lite structural/logprob bound and compare
  them with deterministic validator outcomes without promoting any bound to an
  accept/reject decision;
- compute `false_accept_rate` from Exp 1520-compatible validator rows rather
  than from prefix bounds; and
- write a terminal artifact containing `status`, `milestone`,
  `beaver_bound_ready`, `model_specs`, `live_sota_model_inference_used`,
  `bounded_prefixes`, `token_logprob_available`, `topk_available`,
  `bound_violations`, `high_risk_instances`,
  `deterministic_validator_final_authority`, `false_accept_rate`,
  `bound_audit_path`, `focused_tests_passed`, and `honest_verdict`.

`beaver_bound_ready` MUST be true only when at least one selected prefix is
bounded, every reported upper bound is in `[0, 1]`, deterministic validator
authority is explicitly true, false-accept metrics come from validator rows,
and focused tests have passed. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1537: Prefix Bounds Rank Risk Without Acceptance Authority

Given complete Exp 1535 decoder rows and the Exp 1520 runtime-contract manifest
on the run date `20260508`,
When Exp 1537 audits selected contract JSON prefixes with live telemetry when
available or structural simulation otherwise,
Then prefix-bound series remain structurally monotone for canonical automata
targets
And high-risk rankings record deterministic validator outcomes alongside the
bound
And the terminal artifact reports `deterministic_validator_final_authority=true`
with `false_accept_rate` computed from validator rows, not BEAVER bounds.

### REQ-VERIFY-1538: Residual-Drift Commitment Ledger

The repository shall provide an Exp 1538 deterministic commitment ledger that
replays bounded multi-turn SATQuest, product-line, and runtime-contract cases
and classifies residual failures as true contradictions, satisfiable drift, or
other blockers.

The ledger shall:

- write `results/experiment_1538_residual_drift_commitment_ledger.json` with
  `status="in_progress"` before loading source manifests;
- load bounded cases from `results/satquest_cnf_verifier_1536.jsonl`,
  `results/product_line_rescue_1523.jsonl`, and
  `results/cdg_root_cause_repair_1522.jsonl` when available, preserving
  concrete missing-source blockers without fabricating rows;
- record per-turn commitments introduced by the prompt, intermediate answer or
  staged feedback, oracle/validator replay, and final decision;
- validate final answers with the deterministic SAT, product-line, and
  runtime-contract validators rather than LLM self-evaluation;
- classify a failure as `satisfiable_drift` only when the prior commitments
  have a deterministic satisfying completion but the final answer or plan
  forgets at least one prior commitment;
- classify a failure as `true_contradiction` only when the solver or contract
  oracle proves the combined commitments are unsatisfiable for the attempted
  final decision;
- write `results/residual_drift_commitment_ledger_1538.jsonl` with one row per
  replayed case plus a final summary row; and
- write a terminal artifact containing `status`, `milestone`,
  `residual_drift_ledger_ready`, `model_specs`,
  `live_sota_model_inference_used`, `multi_turn_cases`,
  `contradiction_cases`, `satisfiable_drift_cases`, `drift_rate`,
  `repaired_drift_cases`, `solver_oracle_used`, `false_accept_rate`,
  `ledger_path`, `focused_tests_passed`, and `honest_verdict`.

`residual_drift_ledger_ready` MUST be true only when at least one case from
each available SATQuest, product-line, and runtime-contract source is replayed,
deterministic oracle replay has run, the ledger file exists, focused tests have
passed, and `false_accept_rate` is exactly `0.0`. `honest_verdict` MUST begin
with one of `complete:`, `complete_`, `success:`, `success_`, `passed:`,
`passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1538: Ledger Distinguishes Drift From Contradiction

Given complete Exp 1536 SATQuest rows, Exp 1523 product-line rescue rows, and
Exp 1522 runtime-contract CDG rows on the run date `20260508`,
When Exp 1538 builds the residual-drift commitment ledger,
Then every ledger case records ordered commitments and deterministic validation
evidence before classification
And SAT/product-line/runtime failures with a known satisfying completion are
classified as satisfiable drift rather than contradiction
And impossible SAT final decisions are classified as true contradictions
And the terminal artifact reports bounded metrics without claiming results
beyond the replayed source manifests.

### REQ-VERIFY-1522: Constraint Dependency Graph Root-Cause Repair Ordering

The repository shall provide an Exp 1522 deterministic CPU-only Constraint
Dependency Graph (CDG) analyzer that uses the Exp 1520 runtime-contract E2E
manifest as the local trust boundary and optionally loads Exp 1521 repair rows
when present.

The analyzer shall:

- write `results/experiment_1522_constraint_dependency_graph_root_cause_repair.json`
  with `status="in_progress"` before loading source rows;
- define auditable CDG nodes for parse, certificate, safe-DSL verifier, monitor
  event, structural dependency, solver oracle, and final accept contract
  categories;
- estimate directed dependencies deterministically from lifecycle ordering and
  observed co-failure evidence in the loaded runtime-contract rows;
- select failing rows from Exp 1520, compare flat validator-order localization
  against CDG-prioritized upstream localization, and report one manifest row per
  attempted failing case;
- validate every candidate repair decision through deterministic Exp 1520
  contract ledger semantics and never through LLM self-evaluation;
- write `results/cdg_root_cause_repair_1522.jsonl` with one row per case plus
  a final graph summary row containing node, edge, efficiency, and false-accept
  metrics;
- report `flat_order_fix_efficiency`, `cdg_fix_efficiency`, and
  `cdg_efficiency_delta`; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `cdg_root_cause_repair_ready`,
  `e2e_cases_loaded`, `cdg_nodes`, `cdg_edges`,
  `root_cause_cases_attempted`, `flat_order_fix_efficiency`,
  `cdg_fix_efficiency`, `cdg_efficiency_delta`, `false_accept_count`,
  `false_accept_rate`, `cdg_manifest_path`, `models_used`, `blockers`, and
  `honest_verdict`.

`cdg_root_cause_repair_ready` MUST be true when CDG metrics are computed and
the deterministic false-accept rate is exactly `0.0`, even if the CDG
efficiency delta is negative. If no LLM repair proposal is invoked, the
artifact MUST set `live_sota_model_inference_used=false` and `models_used=[]`
while still recording the mandated local SOTA GGUF `model_specs`.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1522: CDG Prioritizes Upstream Contract Failures

Given a complete Exp 1520 runtime-contract E2E manifest on the run date
`20260508` and optional Exp 1521 repair rows,
When Exp 1522 builds the runtime-contract CDG and analyzes failing contract
rows,
Then graph edges are deterministic and derived from lifecycle ordering plus
observed co-failures
And each attempted case records flat and CDG localization order, root-cause
category, repair-ready status, deterministic validation outcome, and false
accept status
And the terminal artifact sets `cdg_root_cause_repair_ready=true` exactly when
case metrics are computed and the reported false-accept rate remains zero.

### REQ-VERIFY-1525: MARCH Claim-Isolation Verifier Ablation

The repository shall provide an Exp 1525 MARCH-style ablation that compares
full-context verifier feedback against claim-isolated verifier feedback on
promoted Exp 1524 FR-11/runtime-contract cases while keeping deterministic
runtime-contract validators as the final authority.

The ablation shall:

- write
  `results/experiment_1525_march_claim_isolation_verifier_ablation.json` with
  `status="in_progress"` before loading source rows or running model checks;
- load promoted-policy evaluation rows from
  `results/fr11_live_policy_promotion_1524.jsonl` and deterministic contract
  rows from `results/runtime_contract_e2e_manifest_1520.jsonl`;
- resolve mandated local SOTA GGUFs using `cached_sota_pair()` or the same
  local cache pattern, and write a terminal blocker if no mandated SOTA GGUF can
  run rather than using legacy tiny models for headline claim-isolation rows;
- extract an atomic-claim schema from each answer or contract row and record
  stable claim IDs, source case IDs, claim text, source mode, source family, and
  deterministic accept/reject labels;
- run both full-context and claim-isolated checker modes, where the
  claim-isolated checker does not receive the original answer text;
- validate every checker accept/reject through the Exp 1520 deterministic
  runtime-contract ledger and treat checker disagreement as auxiliary evidence
  only;
- write `results/march_claim_isolation_1525.jsonl` with one row per claim/case
  plus a summary row;
- report full-context and claim-isolated accept rates, verifier-call budgets,
  `budget_delta`, false-accept count, and false-accept rate; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `claim_isolation_ablation_ready`,
  `cases_loaded`, `claims_extracted`, `full_context_accept_rate`,
  `claim_isolated_accept_rate`, `claim_isolation_delta`,
  `verifier_calls_full_context`, `verifier_calls_claim_isolated`,
  `budget_delta`, `false_accept_count`, `false_accept_rate`,
  `claim_isolation_manifest_path`, `models_used`, `blockers`, and
  `honest_verdict`.

`claim_isolation_ablation_ready` MUST be true only when both checker modes run
on at least one mandated SOTA GGUF and `false_accept_rate` is reported.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1525: Claim-Isolated Feedback Stays Validator-Grounded

Given promoted Exp 1524 FR-11 policy rows and deterministic Exp 1520
runtime-contract rows on the run date `20260508`,
When Exp 1525 extracts atomic claims and evaluates each case in full-context
and claim-isolated checker modes,
Then every claim row records whether the original answer was hidden from the
claim-isolated checker
And deterministic runtime-contract labels remain the final accept/reject
authority for false accepts
And the terminal artifact sets `claim_isolation_ablation_ready=true` only when
at least one mandated SOTA GGUF produced both full-context and claim-isolated
rows and a false-accept rate is reported.

### REQ-VERIFY-1541: Claim-Isolation Uncertainty Router

The repository shall provide an Exp 1541 claim-isolation uncertainty router
that combines Verify-When-Uncertain-style validator uncertainty with
BEAVER-lite prefix-risk signals so claim-isolated verification is applied only
to cases where it can change risk or cost.

The router shall:

- write `results/experiment_1541_claim_isolation_uncertainty_router_v2.json`
  with `status="in_progress"` before source manifest loading;
- load the Exp 1525 claim-isolation artifact/manifest and Exp 1537
  BEAVER-lite high-risk instance rankings;
- build a bounded case set from `results/runtime_contract_e2e_manifest_1520.jsonl`,
  `results/satquest_cnf_verifier_1536.jsonl`, and
  `results/product_line_rescue_1523.jsonl`, preserving source family, case ID,
  deterministic validator label, and extractable atomic claims;
- route only cases with uncertainty, prefix risk, or validator disagreement to
  claim-isolated verification while leaving low-risk full-context decisions
  unexpanded;
- compute full-context acceptance, routed claim-isolated acceptance,
  disagreements, verifier-call budget delta, and false-accept rate from
  deterministic SAT/product-line/runtime-contract validators rather than LLM
  self-evaluation;
- write `results/claim_isolation_uncertainty_router_1541.jsonl` with one row
  per routed or bypassed case plus a summary row; and
- write a terminal artifact containing `status`, `milestone`,
  `uncertainty_router_ready`, `model_specs`,
  `live_sota_model_inference_used`, `cases_loaded`, `claims_extracted`,
  `routed_cases`, `full_context_accept_rate`, `claim_isolated_accept_rate`,
  `disagreements`, `budget_delta`, `false_accept_rate`,
  `routing_policy_path`, `focused_tests_passed`, and `honest_verdict`.

`uncertainty_router_ready` MUST be true only when at least one case from each
available source family is loaded, at least one but not all cases are routed,
focused tests have passed, and `false_accept_rate` is exactly `0.0`.  The
artifact MUST claim budget improvement only when `budget_delta <= 0` and
`false_accept_rate == 0.0`.  `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1541: Uncertainty Routing Limits Claim Isolation

Given complete Exp 1525 claim-isolation rows, Exp 1537 BEAVER-lite risk
rankings, Exp 1536 SATQuest rows, Exp 1523 product-line rows, and Exp 1520
runtime-contract rows on the run date `20260508`,
When Exp 1541 extracts deterministic claims and evaluates the uncertainty
routing policy,
Then uncertain, prefix-risky, or validator-disagreement cases are routed to
claim-isolated verification
And low-risk cases remain full-context-only
And the terminal artifact reports whether routing improves verifier-call cost
or error detection without deterministic false accepts.

### REQ-VERIFY-1542: ARM/EBT Soft-Value Diagnostic

The repository shall provide an Exp 1542 ARM/EBT diagnostic that compares
autoregressive soft-value proxies with explicit Carnot energy and deterministic
validator labels without using soft values as verifier authority.

The diagnostic shall:

- write `results/experiment_1542_arm_ebm_soft_value_diagnostic.json` with
  `status="in_progress"` before source loading or metric computation;
- load deterministic-labeled cases from SATQuest CNF rows, runtime-contract
  rows, and BEAVER-lite prefix-risk outputs;
- use mandated local SOTA GGUF provenance when available and record whether
  logprob/top-k/value proxies are available; legacy small GGUFs shall not count
  as headline evidence;
- compute explicit Carnot-energy-to-label correlation whenever finite energy
  scores and both deterministic labels are present;
- compute soft-value/logprob-to-label correlation only when real soft-value
  telemetry is present, otherwise report `logprob_available=false` and an
  honest blocker while still running the Carnot-energy-only diagnostic;
- compute a bounded routing AUC when the case set supports positive and
  negative deterministic labels;
- keep deterministic SAT/runtime-contract validators as the final accept/reject
  authority and record `no_model_weight_mutation=true`; and
- write a terminal artifact containing `status`, `milestone`,
  `arm_ebm_diagnostic_ready`, `model_specs`,
  `live_sota_model_inference_used`, `diagnostic_cases`,
  `logprob_available`, `carnot_energy_available`,
  `energy_label_correlation`, `soft_value_label_correlation`, `routing_auc`,
  `deterministic_validators_final_authority`, `no_model_weight_mutation`,
  `diagnostic_report_path`, `focused_tests_passed`, and `honest_verdict`.

`arm_ebm_diagnostic_ready` MUST be true only when at least one diagnostic case
is loaded, explicit Carnot energy is available, deterministic validators remain
final authority, no model weights are mutated, and focused tests have passed.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1542: Soft Values Remain Diagnostic Only

Given complete Exp 1536 SATQuest rows, Exp 1520 runtime-contract rows, and Exp
1537 BEAVER-lite prefix-risk outputs on the run date `20260508`,
When Exp 1542 builds the ARM/EBT soft-value diagnostic,
Then the report compares explicit Carnot energy, available soft-value/logprob
proxies, prefix-risk scores, and deterministic accept/reject labels
And missing logprob telemetry blocks only the soft-value correlation field
And deterministic SAT/runtime-contract labels remain the final accept/reject
authority in every diagnostic row.

### REQ-VERIFY-1556: ARM/EBT Logprob Telemetry Repair Diagnostic

The repository shall provide an Exp 1556 ARM/EBT telemetry-repair diagnostic
that reuses the mandated local SOTA GGUF runtime path to capture token logprob
and top-k telemetry for deterministic-labeled verifier cases when the runtime
exposes that telemetry, and otherwise records exact blockers without promoting
soft signals to acceptance authority.

The diagnostic shall:

- write `results/experiment_1556_arm_ebm_logprob_telemetry_repair.json` with
  `status="in_progress"` before source loading, runtime probing, or metric
  computation;
- select a bounded case set from SATQuest, runtime-contract, or product-line
  rows where deterministic accept/reject labels and explicit energy scores are
  available;
- invoke the existing local SOTA GGUF telemetry path for at least one mandated
  model when locally available, requesting token logprobs and top-k alternatives
  without using legacy small GGUFs as headline telemetry evidence;
- parse token logprobs and top-k alternatives into per-case diagnostic rows,
  recording `logprob_available=false` or `topk_available=false` with precise
  blockers when the runtime returns text without those telemetry fields;
- compute energy-to-label correlation and routing AUC only as diagnostic
  measurements over deterministic labels;
- keep deterministic SAT/runtime-contract/product-line validators as the final
  accept/reject authority, so logprob, top-k, soft-value, model confidence, and
  model-declared acceptance can never override a validator rejection; and
- write a terminal artifact containing `status`, `milestone`,
  `arm_ebm_logprob_telemetry_ready`, `model_specs`,
  `live_sota_model_inference_used`, `logprob_available`, `topk_available`,
  `telemetry_adapter_path`, `diagnostic_cases`,
  `energy_label_correlation`, `routing_auc`,
  `deterministic_validators_final_authority`, `telemetry_blockers`,
  `focused_tests_passed`, and `honest_verdict`.

`arm_ebm_logprob_telemetry_ready` MUST be true only when at least one
diagnostic case has live mandated SOTA text, token logprob telemetry, top-k
alternatives, deterministic validator labels, explicit energy scores,
deterministic validators remain final authority, and focused tests have passed.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1556: Logprobs Stay Below Validator Authority

Given deterministic-labeled SATQuest, runtime-contract, or product-line cases
on the run date `20260508`,
When Exp 1556 captures local SOTA token logprob and top-k telemetry for those
cases,
Then the diagnostic report records available token logprobs, top-k
alternatives, energy scores, deterministic labels, energy-label correlation,
and routing AUC
And missing runtime telemetry is recorded as explicit blockers rather than
fabricated from legacy models
And every final accept/reject decision remains the deterministic validator
decision even when soft telemetry would prefer the opposite answer.

### REQ-VERIFY-1551: Automata/SAT Unified Contract Gate

The repository shall provide an Exp 1551 unified contract gate that sequences
generation-time constraints and deterministic validators for bounded SATQuest,
product-line, and runtime-contract cases.

The gate shall:

- write `results/experiment_1551_automata_sat_unified_contract_gate.json` with
  `status="in_progress"` before loading predecessor artifacts or evaluating
  cases;
- load Exp 1535 automata/ABS, Exp 1549 SATQuest oracle repair, and Exp 1540
  product-line artifacts, and refuse SATQuest acceptance authority unless Exp
  1549 reports zero repaired solver-oracle false accepts;
- expose one shared gate abstraction that routes a generated output through
  syntax or automata masks, semantic repair, the relevant SAT or product-line
  solver oracle, and runtime contracts in that order;
- keep deterministic SAT, product-line, and runtime-contract validators as the
  final accept/reject authority, so soft signals, model confidence, or prefix
  masks can never override a validator mismatch;
- run at least one mandated headline SOTA GGUF model when locally available,
  otherwise record concrete availability blockers and exclude legacy small
  GGUF smoke tests from headline metrics;
- evaluate a bounded mixed set containing SATQuest, product-line, and
  runtime-contract cases, reporting syntax acceptance, semantic repair success,
  oracle agreement, false accepts, and latency delta; and
- write a terminal artifact containing `status`, `milestone`,
  `unified_contract_gate_ready`, `model_specs`,
  `live_sota_model_inference_used`, `cases_attempted`,
  `automata_masks_used`, `semantic_repair_layer_used`, `sat_oracle_used`,
  `product_line_oracle_used`, `runtime_contracts_used`,
  `syntax_accept_rate`, `semantic_repair_success_rate`,
  `oracle_agreement_rate`, `false_accept_rate`, `latency_delta_seconds`,
  `gate_module_path`, `focused_tests_passed`, and `honest_verdict`.

`unified_contract_gate_ready` MUST be true only when each available case family
is evaluated, automata masks and semantic repair are exercised, deterministic
validator final authority is preserved, focused tests have passed, and
`false_accept_rate` is exactly `0.0`. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1551: Gate Rejects Solver Mismatches After Repair

Given complete Exp 1535, Exp 1549, and Exp 1540 artifacts on the run date
`20260508`,
When Exp 1551 evaluates bounded SATQuest, product-line, and runtime-contract
outputs through the unified gate,
Then syntax or automata masks run before semantic repair
And semantic repair runs before SAT/product-line/runtime validators
And any solver or runtime-contract mismatch is rejected even when a soft signal
or model-declared accept says the output should pass
And the terminal artifact reports zero deterministic false accepts for the
bounded mixed case set.

### REQ-VERIFY-1552: Residual-Drift Local Repair Policy

The repository shall provide an Exp 1552 residual-drift repair policy that
loads the Exp 1538 commitment ledger, separates true contradictions from
satisfiable drift cases, localizes the violated commitment or validator span,
and proposes minimal repairs instead of regenerating whole answers.

The policy shall:

- write `results/experiment_1552_residual_drift_repair_policy_v1.json` with
  `status="in_progress"` before loading the residual-drift ledger;
- load `results/residual_drift_commitment_ledger_1538.jsonl` and preserve
  concrete missing-ledger blockers without fabricating drift rows;
- attempt repair only for `satisfiable_drift` rows and leave
  `true_contradiction` rows untouched;
- localize each attempted repair to the forgotten SAT answer/assignment,
  product-line feature selection, or runtime-contract root-cause span;
- use at least one mandated local SOTA GGUF model for repair-proposal text when
  available, otherwise record concrete availability blockers and exclude
  legacy small GGUFs from headline results;
- replay deterministic SAT, product-line, and runtime-contract validators after
  every proposed repair;
- reject repairs that hide contradictions, create deterministic false accepts,
  or fail the relevant validator; and
- write a terminal artifact containing `status`, `milestone`,
  `residual_drift_repair_ready`, `model_specs`,
  `live_sota_model_inference_used`, `drift_cases_before`,
  `repair_attempts`, `localized_repairs_attempted`,
  `repaired_drift_cases`, `drift_reduction_delta`,
  `contradiction_cases_untouched`, `false_accept_rate`,
  `replay_pass_rate`, `repair_policy_path`, `focused_tests_passed`, and
  `honest_verdict`.

`residual_drift_repair_ready` MUST be true only when at least one satisfiable
drift case is attempted, all true contradictions are untouched, every accepted
repair passes deterministic replay, focused tests have passed, and
`false_accept_rate` is exactly `0.0`. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1552: Local Repairs Replay Before Acceptance

Given a complete Exp 1538 residual-drift commitment ledger on the run date
`20260508`,
When Exp 1552 evaluates the repair policy,
Then true contradiction rows are counted as untouched and receive no repair
proposal
And satisfiable drift rows receive localized edit plans for only the violated
commitment or validator span
And candidate repairs are accepted only after deterministic SAT/product-line/
runtime-contract replay passes
And candidate repairs that would create false accepts are rejected before they
can reduce the reported drift count.

### REQ-VERIFY-1553: Claim-Isolation Router Scale Behind Unified Gate

The repository shall provide an Exp 1553 claim-isolation router scale run that
loads the Exp 1541 uncertainty-router artifact and the Exp 1551 unified
contract-gate artifact before evaluating a larger mixed case set.

The scale run shall:

- write `results/experiment_1553_claim_isolation_router_scale_v3.json` with
  `status="in_progress"` before loading predecessor artifacts or source rows;
- require `unified_contract_gate_ready=true` from Exp 1551 before claiming a
  ready terminal artifact;
- reuse the Exp 1541 routing policy and evaluate at least 75 mixed cases when
  the checked-in manifests contain enough rows, including runtime-contract,
  SATQuest, product-line, and residual-drift examples;
- compare routed claim-isolation verifier calls against a full-context
  verification baseline under matched unified-gate acceptance criteria;
- keep deterministic SAT, product-line, runtime-contract, and residual-drift
  replay validators as final authority so hidden deterministic failures cannot
  be accepted by either routed or bypassed paths;
- compute `budget_delta`, `budget_reduced`, `false_accept_rate`, and
  `missed_failure_count` from deterministic labels rather than model
  self-evaluation;
- record mandated SOTA GGUF provenance from predecessor artifacts and concrete
  availability blockers whenever a mandated model is unavailable; and
- write a terminal artifact containing `status`, `milestone`,
  `claim_isolation_router_scale_ready`, `model_specs`,
  `live_sota_model_inference_used`, `cases_total`, `routed_cases`,
  `full_context_cases`, `claims_extracted`, `budget_delta`, `budget_reduced`,
  `false_accept_rate`, `missed_failure_count`, `router_policy_path`,
  `focused_tests_passed`, and `honest_verdict`.

`claim_isolation_router_scale_ready` MUST be true only when Exp 1551 is ready,
at least 75 cases are evaluated, all four required source kinds are present,
at least one but not all cases are routed, focused tests have passed,
`false_accept_rate` is exactly `0.0`, and `budget_reduced=true`.
`budget_reduced` MUST be true only when the routed verifier-call budget is
lower than the full-context baseline under the same unified-gate acceptance
criteria. `honest_verdict` MUST begin with one of `complete:`, `complete_`,
`success:`, `success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1553: Scaled Routing Preserves Gate Safety

Given complete Exp 1541 and Exp 1551 artifacts on the run date `20260508`,
When Exp 1553 evaluates the uncertainty router on a 75+ mixed case set behind
the unified contract gate,
Then threshold-risky, prefix-risky, validator-disagreement, and residual-drift
cases are routed to claim-isolated verification
And low-risk cases are excluded from the routed verifier-call budget
And every final accept is still gated by deterministic SAT/product-line/
runtime-contract or residual-drift replay authority
And the terminal artifact reports whether the routed budget is lower than the
full-context baseline with zero deterministic false accepts.

### REQ-VERIFY-1557: Weaver Verification-Compute Router

The repository shall provide an Exp 1557 verification-compute router that loads
the Exp 1550 SATQuest SOTA re-evaluation artifact and the Exp 1551 unified
contract-gate artifact before selecting a bounded mixed candidate set.

The router shall:

- write `results/experiment_1557_weaver_verification_compute_router.json` with
  `status="in_progress"` before loading predecessor artifacts or source rows;
- use weak verifier signals, including automata/format validity, BEAVER-style
  prefix risk, claim-router uncertainty, energy/logprob diagnostics, and model
  self-declared accept flags, only to choose how much verification compute to
  spend;
- never allow a weak or soft signal to become acceptance authority;
- run at least one deterministic validator before any candidate can be accepted
  and run all available deterministic validators for high-risk candidates;
- compare routed verification cost against an always-run-all-deterministic
  baseline on the same bounded candidate set;
- compute `false_accept_rate` and `missed_failure_count` from deterministic
  validator outcomes rather than model self-evaluation;
- mark `verification_compute_router_ready=false` whenever routing would hide a
  deterministic failure; and
- write a terminal artifact containing `status`, `milestone`,
  `verification_compute_router_ready`, `candidate_selection_cases`,
  `weak_verifiers_used`, `deterministic_validators_used`,
  `soft_signals_used_for_routing_only`, `verification_cost_baseline`,
  `verification_cost_router`, `verification_cost_delta`, `false_accept_rate`,
  `missed_failure_count`, `router_policy_path`, `focused_tests_passed`, and
  `honest_verdict`.

`verification_compute_router_ready` MUST be true only when Exp 1550 and Exp
1551 are loaded, at least one candidate is evaluated, focused tests have
passed, routed cost is lower than the all-deterministic baseline, every final
accept has deterministic validator support, `false_accept_rate` is exactly
`0.0`, and `missed_failure_count` is `0`. `honest_verdict` MUST begin with one
of `complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1557: Routing Saves Cost Without Acceptance Authority

Given complete Exp 1550 and Exp 1551 artifacts on the run date `20260508`,
When Exp 1557 evaluates weak verifier signals over a bounded SATQuest,
runtime-contract, product-line, and residual-drift candidate set,
Then low-risk candidates use the cheap path plus one deterministic source
validator
And high-risk candidates fall back to all available deterministic validators
And a soft accept or high confidence signal never overrides a deterministic
rejection
And the terminal artifact reports whether routed verification cost decreases
with zero deterministic false accepts and zero missed deterministic failures.

### REQ-VERIFY-1554: Product-Line Staged Scale V4 Behind Unified Gate

The repository shall provide an Exp 1554 product-line staged scale run that
loads the Exp 1540 product-line scale artifact and the Exp 1551 unified
contract gate before reporting larger solver-grounded product-line metrics.

The scale run shall:

- write `results/experiment_1554_product_line_staged_scale_v4.json` with
  `status="in_progress"` before loading predecessor artifacts or evaluating
  product-line cases;
- require `unified_contract_gate_ready=true` from Exp 1551 before claiming a
  ready terminal artifact;
- build or select a staged product-line pack up to 120 cases when the
  checked-in generators and runtime permit, covering syntax-only,
  feasibility, objective-quality, and natural-language product-line variants;
- route rows through deterministic product-line parsing and solver-oracle
  evaluation, preserving the unified contract gate as an upstream prerequisite
  and the product-line solver oracle as final accept/reject authority;
- compute parse rate, feasibility rate, oracle agreement rate, mean objective
  gap, entity hallucination rate, and false accept rate from structured row
  fields rather than model self-evaluation;
- record mandated SOTA GGUF model availability honestly, excluding legacy
  small GGUF smoke tests from headline-result model lists; and
- write a terminal artifact containing `status`, `milestone`,
  `product_line_scale_v4_ready`, `branch_retired`, `model_specs`,
  `live_sota_model_inference_used`, `cases_total`, `stages_tested`,
  `parse_rate`, `feasibility_rate`, `objective_gap_mean`,
  `oracle_agreement_rate`, `entity_hallucination_rate`, `false_accept_rate`,
  `automata_constraints_used`, `product_line_manifest_path`,
  `focused_tests_passed`, and `honest_verdict`.

`product_line_scale_v4_ready` MUST be true only when Exp 1551 is ready, at
least one product-line case is evaluated, focused tests have passed,
deterministic checks are made from structured solver-backed fields,
`false_accept_rate` is exactly `0.0`, and `branch_retired=false`.
`branch_retired` MUST be true when false accepts recur or deterministic
feasibility/oracle checks cannot be made. `honest_verdict` MUST begin with one
of `complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1554: Product-Line Scaling Reports Solver-Grounded Fields

Given complete Exp 1540 and Exp 1551 artifacts on the run date `20260508`,
When Exp 1554 evaluates a larger staged product-line pack behind the unified
contract gate,
Then syntax-only, feasibility, objective-quality, and natural-language product
line variants are represented in the manifest
And parse, feasibility, objective-gap, oracle-agreement, hallucination, and
false-accept metrics are aggregated from deterministic row fields
And any false accept or missing deterministic oracle field retires the branch
before a ready artifact can be reported.

### REQ-VERIFY-1562: BRAIN Linear-AR k-Sweep Verification

The repository shall provide a runnable Exp 1562 BRAIN-correlation verification
workflow that extends the DT-BRAIN-CORRELATIONS brute-force enumeration from
`k=4` to `k in {4, 8, 12, 15}` at `n=16`.

The workflow shall:

- enumerate all `2^16 = 65,536` binary states exactly for each `k`;
- preserve the original deterministic `k=4` seed semantics so the baseline
  factorized and Linear-AR KL values remain comparable to the 2026-05-08
  partial validation run;
- optimize and report reverse KL `KL(q || pi_beta)` for a factorized Bernoulli
  model with `n=16` parameters and a Linear-AR model with
  `n + n(n - 1) / 2 = 136` parameters;
- run the optional one-hidden-layer MADE arm with 32 hidden units only when the
  Linear-AR `k=15` KL is above `0.1`;
- write `results/experiment_1562_brain_linear_ar_k_sweep_extended.json` with
  `status`, `brain_linear_ar_rescue_validated`,
  `kl_by_k_by_parameterization`, `factorized_vs_ar_ratio_at_k15`,
  `made_required_at_k15`, `phase_3_recommendation`, and `honest_verdict`; and
- recommend `linear_ar_sufficient` only when the `k=15` ratio is at least
  `10.0` and the best `k=15` KL is at most `0.1`, `made_required` only when
  MADE is needed and reaches the `0.1` KL gate, and `brain_dropped` when the
  `k=15` factorized-vs-AR ratio is below `5.0`.

`status` MUST be `complete` for terminal artifacts. `honest_verdict` MUST begin
with `complete:` and MUST distinguish positive validation from falsification.

### SCENARIO-VERIFY-1562: k-Sweep Writes Honest BRAIN Recommendation

Given the deterministic DT-BRAIN-CORRELATIONS constraint generator with
`n=16`, `m=10`, `beta=2.0`, and `seed=42`,
When Exp 1562 runs the exact `{4, 8, 12, 15}` k-sweep,
Then the artifact reports factorized and Linear-AR reverse KL for every `k`
And preserves the `k=4` baseline comparison fields
And records `made_optional` as null unless the Linear-AR `k=15` KL exceeds
`0.1`
And maps a `k=15` ratio below `5.0` to `phase_3_recommendation="brain_dropped"`
instead of claiming the Linear-AR rescue was validated.

### REQ-VERIFY-1571: Step-Wise Baseline AR-REINFORCE

The repository shall provide a runnable Exp 1571 AR-REINFORCE variance
microbenchmark for a Linear-AR Bernoulli model on an `n=32`, `k=15`
AND-composition stress case with `3%` Gaussian energy noise.

The workflow shall:

- implement a per-token step-wise baseline for Linear-AR REINFORCE whose
  baseline for token `t` depends only on the prefix `x_<t` and model
  parameters, not on `x_t` or later sampled tokens;
- compare scalar-batch-mean REINFORCE against the step-wise baseline using
  trace variance over the Linear-AR coupling-parameter score-function
  gradient;
- report `gradient_variance_reduction_factor >= 10.0` for the step-wise
  baseline versus the scalar baseline on the `n=32`, `k=15` noisy-energy
  benchmark;
- adapt BRAIN Theorem 2's noise-resilience claim to AR stochastic
  approximation by comparing the step-wise estimator's noisy and noiseless
  gradient signal-to-noise convergence-rate proxy, and require the noisy
  `3%` run to retain at least `97%` of the noiseless proxy; and
- write `results/experiment_1571_step_wise_baseline_AR_REINFORCE.json` with
  `status`, `step_wise_baseline_implemented`,
  `gradient_variance_reduction_factor`,
  `convergence_rate_matches_theorem_2`, and `honest_verdict`.

`status` MUST be `complete` for terminal artifacts. `honest_verdict` MUST begin
with `complete:` and MUST state whether the variance and noise-resilience gates
passed.

### SCENARIO-VERIFY-1571: Step-Wise Baseline Reduces AR Coupling Variance

Given the planted `n=32`, `k=15`, ten-constraint AND-composition stress case
with `3%` Gaussian reward noise,
When Exp 1571 evaluates Linear-AR REINFORCE gradient samples with a scalar
batch-mean baseline and with the prefix-only step-wise baseline,
Then the artifact reports `step_wise_baseline_implemented=true`
And the reported AR-coupling gradient variance reduction factor is at least
`10.0`
And `convergence_rate_matches_theorem_2=true` only when the noisy step-wise
signal-to-noise convergence-rate proxy is at least `97%` of the noiseless
step-wise proxy.

### REQ-VERIFY-1578: BRAIN REINFORCE Training-Dynamics Audit at k=15

The repository shall provide a runnable Exp 1578 BRAIN REINFORCE
training-dynamics audit for the same `n=16`, `k=15`, `beta=2.0`, ten random
AND-composition regime used to interpret Exp 1562.

The workflow shall:

- initialise both the factorized Bernoulli and Linear-AR Bernoulli
  parameterizations uniformly with zero logits and zero lower-triangular AR
  weights;
- train both parameterizations with a scalar-baseline REINFORCE estimator for
  reverse KL `KL(q || pi_beta)`, using batch size `512` and at most `50,000`
  iterations;
- compute exact finite-state `KL(q || pi_beta)` against the enumerated
  `2^16`-state target distribution at iteration `0` and every `1000`
  iterations;
- track gradient L2 norm, marginal escape from `0.5`, first-1000-iteration
  gradient-active fraction, convergence iteration, and wall-clock time for
  both parameterizations;
- write `results/experiment_1578_brain_reinforce_training_dynamics_at_k15.json`
  with `status`, `factorized_gradient_active_fraction_first_1000`,
  `linear_ar_gradient_active_fraction_first_1000`, `factorized_final_kl`,
  `linear_ar_final_kl`, `factorized_converged`, `linear_ar_converged`,
  `brain_training_dynamics_verdict_ready`, `paper_v6_brain_recommendation`,
  and `honest_verdict`; and
- choose exactly one training-dynamics verdict from `factorized gradient starvation real`,
  `starvation overstated`, or `both parameterizations inadequate`, then propagate that recommendation into
  `docs/research-notes/brain-reinforce-training-dynamics-k15.md`.

`status` MUST be `complete` for terminal artifacts. `honest_verdict` MUST begin
with `complete:` and MUST state the selected training-dynamics verdict.

### SCENARIO-VERIFY-1578: k15 REINFORCE Audit Writes a Paper-v6 Recommendation

Given the deterministic `n=16`, `k=15`, ten-constraint BRAIN target with
`beta=2.0` and uniform initial q-parameters,
When Exp 1578 trains factorized Bernoulli and Linear-AR q models with
REINFORCE,
Then the artifact reports per-parameterization first-1000 gradient-active
fractions and terminal exact KL values
And `brain_training_dynamics_verdict_ready=true` only when both required
parameterizations have complete traces and one allowed verdict is selected
And `paper_v6_brain_recommendation` records whether paper v6 should cite
factorized starvation, treat starvation as overstated, or drop both
parameterizations as inadequate.

### REQ-VERIFY-1588: Bounded Instruction-To-Constraint DSL Pack

The repository shall provide a bounded instruction-to-constraint pack for Exp
1588 that converts natural-language output instructions into a small Carnot DSL
and compiles that DSL into deterministic local validators.

The pack shall:

- define the DSL schema in `python/carnot/verifiers/dsl.py`;
- parse only a fixed, auditable set of instruction patterns, including required
  text, forbidden text, JSON-object requirements, JSON key requirements, word
  count bounds, answer enumerations, and exact bullet counts;
- cap instruction length and constraint count, and fail closed rather than
  executing generated Python or accepting unsupported operators;
- compile every supported DSL constraint into a Python validator that reports
  per-constraint failures;
- emit a PySAT-compatible CNF representation of the hard conjunction whenever
  constraints compile, without requiring the optional `python-sat` package in
  the Python test environment;
- write `results/experiment_1588_nsvif_dsl.json` with `status`,
  `experiment_id`, `dsl_schema_version`, `instructions_tested`,
  `constraints_extracted`, `validators_compiled`, `pysat_cnf_compiled`,
  `python_validator_pass_rate`, `known_good_pass_rate`,
  `known_bad_reject_rate`, `false_accept_rate`,
  `arbitrary_code_execution_path_introduced`, `tests_run`, and
  `honest_verdict`; and
- set terminal `status="complete"` only when all fixture validators compile,
  known-good examples pass, known-bad examples reject, and no arbitrary-code
  execution path is introduced.

### SCENARIO-VERIFY-1588: Natural-Language Instructions Compile To Local Validators

Given bounded natural-language instructions with supported semantic constraints,
When the Exp 1588 DSL parser and compiler run,
Then the parsed pack validates against the local schema,
And the compiled Python validator accepts the known-good output while reporting
specific constraint failures for known-bad output,
And a PySAT-compatible CNF hard-conjunction view is available for the same
constraint IDs,
And `results/experiment_1588_nsvif_dsl.json` records complete validator metrics
with `false_accept_rate=0.0`.

### REQ-VERIFY-1640: NSVIF DSL Parser Emits Carnot Constraints

The repository shall provide an Exp 1640 NSVIF-style instruction-to-constraint
workflow that reuses the bounded safe DSL and exposes the parsed instructions as
Carnot `ConstraintResult` rows plus compiled local Python validators.

The workflow shall:

- define `scripts/experiment_1640_nsvif_dsl.py`;
- parse supported natural-language instruction cases through the bounded
  instruction DSL without executing generated Python or model-proposed code;
- convert each parsed DSL constraint into a serializable Carnot
  `ConstraintResult` shape with operator, field, value, source text, and DSL
  schema metadata;
- compile the same constraints into local Python validators and evaluate at
  least one known-good and one known-bad output for every parsed case;
- write `results/experiment_1640_nsvif_dsl.json` with `status`,
  `experiment_id`, `dsl_schema_version`, `parser_success`,
  `instructions_tested`, `constraints_extracted`, `carnot_constraints_emitted`,
  `validators_compiled`, `known_good_pass_rate`, `known_bad_reject_rate`,
  `false_accept_rate`, `arbitrary_code_execution_path_introduced`,
  `tests_run`, and `honest_verdict`; and
- set `parser_success=true` only when every instruction case parses into at
  least one schema-valid constraint, and set terminal `status="complete"` only
  when parser success, validator compilation, known-good acceptance, known-bad
  rejection, and zero arbitrary-code execution all hold.

### SCENARIO-VERIFY-1640: Parsed Instructions Become Carnot Constraints And Validators

Given bounded NSVIF-style natural-language instructions,
When Exp 1640 parses the instructions and compiles validators,
Then each instruction emits schema-valid DSL constraints and serializable Carnot
constraint rows,
And the compiled Python validator accepts the known-good output while rejecting
the known-bad output,
And `results/experiment_1640_nsvif_dsl.json` records `parser_success=true` with
`false_accept_rate=0.0`.

### REQ-VERIFY-1641: NSVIF SOTA GGUF Output Adapter

The repository shall provide an importable pipeline adapter that translates
bounded local SOTA GGUF generation outputs into NSVIF DSL inputs without
executing model-generated code.

The adapter shall:

- define `python/carnot/pipeline/nsvif_sota.py`;
- accept GGUF output rows containing model provenance, case IDs, raw output
  text, and optional expected-good / expected-bad validation examples;
- extract JSON objects from raw SOTA output and accept only bounded NSVIF input
  shapes: `instruction` / `nsvif_instruction`, `dsl_pack` / `constraint_pack`,
  or a supported `constraints` list;
- normalize accepted rows into the existing
  `carnot.verifiers.dsl.ConstraintPack` JSON schema and serializable Carnot
  `ConstraintResult` rows with model provenance metadata;
- compile each normalized pack through the existing NSVIF DSL compiler and
  evaluate the supplied validation examples when present;
- fail closed for unparseable output, unsupported operators, unsafe text,
  empty constraint packs, or unbounded constraint counts;
- write `results/experiment_1641_nsvif_sota.json` with `status`,
  `experiment_id`, `adapter_schema_version`, `dsl_schema_version`,
  `sota_outputs_seen`, `dsl_inputs_emitted`, `validators_compiled`,
  `known_good_pass_rate`, `known_bad_reject_rate`, `false_accept_rate`,
  `arbitrary_code_execution_path_introduced`, `tests_run`, and
  `honest_verdict`; and
- set terminal `status="complete"` only when every supplied SOTA output row
  emits a schema-valid DSL input, every emitted validator compiles, known-good
  examples pass, known-bad examples reject, and no arbitrary-code execution
  path is introduced.

### SCENARIO-VERIFY-1641: SOTA Output Becomes NSVIF DSL Input

Given bounded JSON output from a mandated local SOTA GGUF model that proposes
NSVIF instructions or safe constraints,
When the adapter normalizes the row,
Then it emits an existing NSVIF `ConstraintPack` schema payload plus Carnot
constraint rows carrying model provenance,
And the same pack compiles through the local DSL validator,
And the terminal artifact records emitted DSL inputs, validator metrics, and
`false_accept_rate=0.0`.

### REQ-VERIFY-1666: Product NSVIF Instruction-To-Constraint Parser

The repository shall provide an importable product parser that turns bounded
natural-language prompts into executable Carnot constraints and local validator
backends without executing generated code.

The parser shall:

- define `python/carnot/pipeline/nsvif_parser.py`;
- parse supported natural-language instruction prompts into the existing
  bounded Carnot instruction DSL and serializable Carnot `ConstraintResult`
  rows;
- compile each parsed prompt to local Python, PySAT-compatible CNF, and
  Z3-compatible hard-conjunction validator artifacts;
- fail closed for unsafe prompts, unsupported operators, empty constraint
  packs, or unbounded constraint counts;
- evaluate bounded GGUF model-spec rows for
  `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF` against at
  least one known-good and one known-bad output per row;
- write `results/experiment_1666_nsvif.json` with at minimum `false_accepts`
  and `compilation_rate`; and
- set terminal `status="complete"` only when every bounded model row parses,
  every validator backend compiles, every known-good output passes, every
  known-bad output rejects, and `false_accepts=0`.

### SCENARIO-VERIFY-1666: Product Prompts Compile To Executable Constraints

Given bounded natural-language prompts from the mandated GGUF model-spec rows,
When the NSVIF parser evaluates the prompt cases,
Then each prompt emits a schema-valid DSL pack and Carnot constraint rows,
And Python, PySAT-compatible CNF, and Z3-compatible validator artifacts compile
for every case,
And known-good outputs pass while known-bad outputs reject with zero false
accepts,
And `results/experiment_1666_nsvif.json` records `compilation_rate=1.0` and
`false_accepts=0`.

### REQ-VERIFY-1996: NSVIF/Z3 CoT SMT Extractor

The repository shall provide a deterministic NSVIF-style SMT extractor for Exp
1996 that formalizes supported chain-of-thought constraints into local Z3
formulas without calling an LLM or executing generated code.

The extractor shall:

- define an importable extractor under `python/carnot/pipeline/`;
- parse instruction-tuned prose arithmetic steps such as `47 plus 28 gives 76`
  into first-order arithmetic equalities and verify them with Z3;
- parse a bounded categorical-logic subset including universal class
  constraints, individual membership facts, negated membership facts, and
  `therefore` conclusions into Z3 formulas;
- fail closed on unsupported or ambiguous reasoning text by abstaining rather
  than emitting violated constraints;
- report per-step formula metadata, solver status, verdict, and zero false
  positives on the bundled correct-case fixtures; and
- write `results/experiment_1996_nsvif_smt_extractor.json` with `status`,
  `success`, `experiment_id`, `spec_refs`, `model_specs`, `cases_attempted`,
  `constraints_extracted`, `solver_checks`, `false_positives`,
  `false_accepts`, `zero_false_positives_by_design`, `tests_run`, and
  `honest_verdict`.

`success` MUST be true only when the bundled supported incorrect cases are
rejected, bundled supported correct cases are not flagged, no unsupported case
is accepted as verified, and `false_positives=0`.

### SCENARIO-VERIFY-1996: IT CoT Constraints Are Checked By Z3

Given instruction-tuned chain-of-thought prose containing supported arithmetic
or categorical-logic constraints,
When the NSVIF/Z3 SMT extractor processes the trace,
Then each supported claim is formalized into solver metadata tied to its step
index,
And arithmetic contradictions and direct logical contradictions return
`satisfied=false`,
And correct supported steps return `satisfied=true`,
And unsupported or non-entailed prose conclusions abstain instead of becoming
false positives.

### REQ-VERIFY-2353: VERGE MCS Repair Engine

The repository shall provide a deterministic VERGE-style repair engine for Exp
2353 that uses Z3 Minimal Correction Subsets to localize which violated
NSVIF-extracted arithmetic claims must be relaxed before a response can become
satisfiable.

The repair engine shall:

- define `python/carnot/repair/verge_repair.py` with an importable
  `VergeRepairEngine`;
- expose `find_mcs(constraints, violated)` returning the smallest subset of
  violated constraint indices whose removal makes the remaining Z3 constraints
  satisfiable;
- expose `suggest_repair(response, violations)` returning a concrete textual
  claim edit suggestion for supported arithmetic violations;
- evaluate exactly ten deterministic arithmetic-error scenarios with
  `random_seed=42`; and
- write `results/experiment_2353_verge_repair.json` with `honest_verdict`,
  `verge_repair_validated`, `mcs_repair_success_rate`,
  `n_repair_scenarios`, and `random_seed`.

`verge_repair_validated` MUST be true only when
`mcs_repair_success_rate >= 0.50` on the ten-scenario repair set.

### SCENARIO-VERIFY-2353: MCS Localizes Arithmetic Repair Claims

Given NSVIF-extracted Z3 constraints from responses with one arithmetic error,
When `VergeRepairEngine` checks the violated claim indices,
Then its MCS result identifies the claim index whose relaxation restores
satisfiability,
And its repair suggestion rewrites the violated arithmetic claim to the exact
computed value,
And the Exp 2353 artifact records the ten-case success rate and terminal
verdict.

### REQ-VERIFY-1878: ROCE Validator-Tree Compiler

The repository shall provide a deterministic ROCE-to-validator-tree compiler
that reuses Carnot's existing ROCE extractor and NSVIF verifier patterns before
trusting live SOTA output constraints.

The compiler shall:

- define `python/carnot/pipeline/roce_validator_tree.py`;
- accept ROCE `ConstraintResult` rows or prompt text that can be processed by
  the existing `ROCEExtractor`;
- compile supported ROCE predicates into an explicit local evaluation tree with
  Python validation leaves and PySAT-compatible plus Z3-compatible hard
  conjunction metadata;
- support at least response length, required text, forbidden text, JSON shape
  or required-key checks, arithmetic equality, and conditional guard
  constraints;
- treat conditional constraints as guarded leaves, so the guarded check is
  enforced exactly when its guard predicate is true and skipped otherwise;
- fail closed for unsupported constraint categories while recording
  unsupported categories and constraint coverage;
- evaluate a deterministic fixture set containing known-good and adversarial
  invalid outputs, proving zero false accepts on that fixture set; and
- write `results/experiment_1878_roce_validator_tree.json` with `status`,
  `honest_verdict`, `validator_tree_compiler_ready`, `zero_false_accepts`,
  `false_accept_count`, `constraint_coverage_rate`,
  `unsupported_constraint_types`, and `tests_run`.

`validator_tree_compiler_ready` MUST be true only when every supported fixture
constraint compiles, known-good fixtures pass, adversarial invalid fixtures are
rejected, false accepts are zero, and unsupported categories are recorded
without being silently accepted.

### SCENARIO-VERIFY-1878: ROCE Output Compiles To Guarded Validator Tree

Given ROCE-extracted prompt constraints for length, text inclusion/exclusion,
JSON shape, arithmetic equality, and an if-then conditional guard,
When the validator-tree compiler builds local validators and evaluates the
fixture outputs,
Then the tree exposes Python, PySAT-compatible, and Z3-compatible validation
metadata,
And guarded leaves only fail when their guard is active,
And adversarial invalid outputs are rejected with `false_accept_count=0`,
And `results/experiment_1878_roce_validator_tree.json` records complete
coverage and unsupported-category metrics.

### REQ-VERIFY-1879: BEAVER-Lite Bounds For ROCE Validator Trees

The repository shall provide a deterministic BEAVER-lite bound calculator for
compiled ROCE validator trees that reports prompt coverage without replacing
the executable validator tree as acceptance authority.

The bound calculator shall:

- define `python/carnot/verify/roce_beaver_lite_bounds.py`;
- load or reconstruct the Exp 1878 ROCE validator-tree fixture cases;
- compute one conservative bound row per compiled validator leaf, assigning
  deterministic coverage only to executable local validator leaves and
  residual risk to unsupported or uncompiled constraints;
- aggregate `deterministic_coverage_bound` and `residual_risk_bound` in the
  closed interval `[0, 1]`;
- attach the bound summary to a structured `VerdictRecord` without changing
  the `pass` or `fail` verdict returned by the executable tree validator;
- prove that BEAVER-lite bound rows cannot promote an invalid output to
  accepted; and
- write `results/experiment_1879_beaver_lite_bounds.json` with `status`,
  `honest_verdict`, `beaver_lite_bounds_ready`,
  `deterministic_coverage_bound`, `residual_risk_bound`,
  `acceptance_authority_unchanged`, and `tests_run`.

`beaver_lite_bounds_ready` MUST be true only when every emitted bound is
finite, aggregate coverage plus residual risk does not exceed one, at least one
Exp 1878 fixture tree is bounded, and `acceptance_authority_unchanged` is true.

### SCENARIO-VERIFY-1879: Bounds Sit Below Executable Validator Authority

Given the Exp 1878 ROCE fixture with required text, forbidden text, JSON shape,
arithmetic equality, and a guarded conditional leaf,
When BEAVER-lite bounds are computed for the compiled validator tree,
Then each validator leaf receives a deterministic coverage contribution,
unsupported constraints receive residual-risk contribution instead of silent
acceptance,
And an adversarial invalid output still produces a failing `VerdictRecord`,
And `results/experiment_1879_beaver_lite_bounds.json` records complete
coverage, zero residual risk for the supported fixture, and unchanged
acceptance authority.

### REQ-VERIFY-1880: Live SOTA ROCE Validator-Tree Evaluation

The repository shall provide an Exp 1880 evaluator for ROCE-compiled validator
trees on mandated local SOTA GGUF generations.

The evaluator shall:

- define `python/carnot/verify/sota_roce_validator_eval.py`;
- define `MODEL_SPECS` with `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`;
- use `cached_sota_pair()` where applicable while resolving all three mandated
  GGUF cache paths before making any headline claim;
- evaluate at least 30 prompt cases spanning format, arithmetic, lexical, and
  conditional constraints when all mandated GGUFs are available;
- validate every model output with the Exp 1878 ROCE validator tree and attach
  the Exp 1879 deterministic BEAVER-lite bound row summary;
- report zero-false-accept behavior only from executable validator-tree
  decisions, not from model self-reporting or BEAVER-lite accounting alone;
- record live/cache provenance for every attempted model; and
- write `results/experiment_1880_sota_roce_validator_eval.json` with `status`,
  `honest_verdict`, `sota_roce_eval_ready`, `inference_mode`, `MODEL_SPECS`,
  `models_used`, `zero_false_accepts`, `false_accept_count`,
  `constraint_coverage_rate`, and `tests_run`.

If any mandated model is unavailable, the evaluator MUST write a blocked
artifact with `models_used`, `sota_roce_eval_ready=false`, and an honest verdict
that does not claim headline accuracy.

### SCENARIO-VERIFY-1880: Mandated SOTA Outputs Are Gated By ROCE Validators

Given at least 30 ROCE prompt cases spanning format, arithmetic, lexical, and
conditional constraints,
When all three mandated local GGUF models are available and generate outputs,
Then the evaluator validates each output with the compiled ROCE tree,
And computes deterministic BEAVER-lite coverage from the same tree,
And reports `zero_false_accepts=true` only when no invalid generated output is
accepted,
And unavailable mandated models produce a blocked artifact rather than a
headline metric.

### REQ-VERIFY-1642: llguidance Adapter For Structured Verdict Generation

The repository shall provide an Exp 1642 llguidance adapter that exposes
Carnot structured verdict records to local GGUF/llama.cpp call sites without
making the deterministic verifier depend on optional grammar bindings.

The adapter shall:

- define `scripts/experiment_1642_llguidance.py`;
- build a bounded JSON schema for `VerdictRecord` payloads emitted by Carnot's
  structured verdict pipeline;
- compile llguidance JSON-schema grammar metadata when optional `llguidance`
  bindings are importable or injected by tests;
- expose llama.cpp request metadata containing the prompt, schema, backend
  diagnostics, and compiled grammar when available;
- preserve a post-decode deterministic fallback that validates generated JSON
  and returns abstaining `VerdictRecord` rows for malformed or schema-invalid
  output;
- round-trip a known-good `VerdictRecord` through JSON without arbitrary code
  execution; and
- write `results/experiment_1642_llguidance.json` with `status`,
  `experiment_id`, `adapter_module`, `adapter_success`,
  `llguidance_backend_available`, `fallback_backend_available`,
  `llama_cpp_adapter_ready`, `structured_verdict_roundtrip`,
  `invalid_output_abstains`, `tests_run`, and `honest_verdict`.

`adapter_success` MUST be true only when the known-good verdict round-trips,
invalid generated output abstains rather than passing, llama.cpp metadata is
available, and the deterministic fallback remains available even when
llguidance is absent.

### SCENARIO-VERIFY-1642: Adapter Emits llguidance And Fallback Verdict Rows

Given a Carnot `VerdictRecord` and a bounded structured-verdict JSON schema,
When Exp 1642 builds llguidance/llama.cpp generation metadata and parses both
valid and invalid generated JSON,
Then the valid JSON returns the original pass/fail/abstain verdict fields,
the invalid JSON returns an abstaining `VerdictRecord` with schema diagnostics,
and `results/experiment_1642_llguidance.json` records
`adapter_success=true`.

### REQ-VERIFY-1646: EBCN Reasoning-Trace Coherence Prototype

The repository shall provide an Exp 1646 deterministic EBCN prototype that
assigns structural-coherence energy scores to reasoning traces and flags direct
logical inconsistencies without calling an autoregressive generator.

The workflow shall:

- define `scripts/experiment_1646_ebcn.py`;
- reuse the existing fixed hidden-state EBCN scorer for dual support and
  contradiction heads;
- add a deterministic state-space rollout over encoded reasoning-trace steps
  before scoring, so trace order contributes to the hidden state without model
  sampling;
- evaluate bounded synthetic reasoning traces containing both consistent and
  directly contradictory claims;
- compute `coherence_score_accuracy`, where coherent traces should receive
  higher coherence scores than contradictory traces; and
- write `results/experiment_1646_ebcn.json` with `status`, `experiment_id`,
  `ebcn_prototype_ready`, `dual_head_attention_used`,
  `state_space_transition_used`, `autoregressive_generation_used`,
  `hidden_state_source`, `reasoning_trace_cases_total`, `inconsistent_cases`,
  `consistent_cases`, `inconsistent_mean_energy`, `consistent_mean_energy`,
  `energy_gap`, `coherence_score_accuracy`, `tests_run`, and
  `honest_verdict`.

`status` MUST be `complete` only when dual-head attention and state-space
rollout are used, autoregressive generation is not used, inconsistent traces
have higher mean energy than consistent traces, and `coherence_score_accuracy`
is at least `0.8`.

### SCENARIO-VERIFY-1646: EBCN Scores Reasoning-Trace Inconsistencies

Given deterministic reasoning traces with consistent claims and paired traces
that negate a prior claim,
When Exp 1646 encodes the steps, applies the state-space rollout, and scores
them with the EBCN support and contradiction heads,
Then direct logical inconsistencies receive higher energy and lower coherence
scores than consistent traces,
And `results/experiment_1646_ebcn.json` records `coherence_score_accuracy`.

### REQ-VERIFY-1591: Reusable DCCD Structured Verdict Adapter

The repository shall provide a reusable structured-verdict adapter for Exp 1591
that upgrades the Exp 1580 DCCD smoke path into a deterministic API for Carnot
verifier-output JSON schemas.

The adapter shall:

- define `DCCDStructuredVerdictAdapter` under `python/carnot/verifiers/`;
- accept a bounded JSON schema, an optional semantic-path expectation mapping,
  and raw unconstrained draft text;
- compile an llguidance JSON-schema grammar when the optional `llguidance`
  Python bindings are importable, while preserving a deterministic post-decode
  fallback when they are not installed;
- perform DCCD-style structural projection from draft payload to target payload
  without introducing arbitrary code execution;
- return a structured `VerdictRecord` whose `extras` include schema errors,
  semantic errors, false-accept status, backend diagnostics, and the parsed
  payload;
- expose prompt and grammar metadata that downstream local GGUF/llama.cpp
  call sites can use for constrained regeneration; and
- write `results/experiment_1591_dccd_adapter.json` with `status`,
  `experiment_id`, `adapter_module`, `llguidance_backend_available`,
  `fallback_backend_available`, `strict_schema_validity_rate`,
  `semantic_correctness_rate`, `false_accept_count`,
  `arbitrary_code_execution_path_introduced`, `tests_run`, and
  `honest_verdict`.

`status` MUST be `complete` only when the adapter accepts known-good structured
rows, rejects schema-valid semantic false accepts, records backend diagnostics,
and the deterministic fallback remains available without `llguidance`.

### SCENARIO-VERIFY-1591: DCCD Adapter Emits Structured Verdict Records

Given a Carnot verifier-output schema with deterministic semantic-path
expectations and a target payload,
When `DCCDStructuredVerdictAdapter` evaluates unconstrained draft text, a
DCCD-projected payload, and a schema-valid semantic false accept,
Then the draft row records schema/semantic failures without becoming a pass,
the DCCD-projected row returns a `VerdictRecord` with `verdict="pass"` and
zero false accepts,
the semantic false accept returns a `VerdictRecord` with `verdict="fail"` and
`extras["false_accept"]=true`,
and `results/experiment_1591_dccd_adapter.json` records complete metrics and
backend diagnostics for the reusable adapter.

## Implementation Status (REQ-VERIFY-1415/1416/1423/1434/1469/1473/1474/1475/1481/1486/1487/1495/1496/1499/1500/1501/1507/1508/1509/1510/1520/1521/1522/1525/1537/1538/1541/1542/1551/1552/1553/1554/1557/1562/1571/1578/1580/1588/1591/1642/1666/1878/1879/1880)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-VERIFY-1415 | Implemented (`python/carnot/reporting/dvi_v3_1508_fresh_cases.py`) | Implemented (`tests/python/test_experiment_1415_dvi_v3_1508_fresh_cases.py`) |
| REQ-VERIFY-1432 | Implemented (`python/carnot/reporting/dvi_v3_nonforgetting_replay_balanced.py`) | Implemented (`tests/python/test_experiment_1432_dvi_v3_nonforgetting_replay_balanced.py`) |
| REQ-VERIFY-1416 | Implemented (`python/carnot/models/ebm_cot_temperature_calibration.py`) | Implemented (`tests/python/test_experiment_1416_ebm_cot_temperature_calibration.py`) |
| REQ-VERIFY-1423 | Implemented (`python/carnot/reporting/process_reward_model_v1_fover_1508.py`) | Implemented (`tests/python/test_experiment_1423_process_reward_model_v1.py`) |
| REQ-VERIFY-1434 | Implemented (`python/carnot/reporting/fover_prm_label_completion_v2.py`) | Implemented (`tests/python/test_experiment_1434_fover_prm_label_completion_v2.py`) |
| REQ-VERIFY-1469 | Implemented (`python/carnot/reporting/halt_spilled_energy_telemetry_diagnostic.py`) | Implemented (`tests/python/test_experiment_1469_halt_spilled_energy_telemetry_diagnostic.py`) |
| REQ-VERIFY-1473 | Implemented (`python/carnot/reporting/live_telemetry_adversarial_validity_audit.py`) | Implemented (`tests/python/test_experiment_1473_live_telemetry_adversarial_validity_audit.py`) |
| REQ-VERIFY-1474 | Implemented (`python/carnot/verify/skm_projection.py`) | Implemented (`tests/python/test_experiment_1474_tskm_linear_constraint_projection_smoke.py`) |
| REQ-VERIFY-1475 | Implemented (`python/carnot/eval/static_csr_certificate_automaton.py`) | Implemented (`tests/python/test_experiment_1475_static_csr_certificate_automaton.py`) |
| REQ-VERIFY-1481 | Implemented (`python/carnot/reporting/semantic_energy_feasibility_audit.py`) | Implemented (`tests/python/test_experiment_1481_semantic_energy_feasibility_audit.py`) |
| REQ-VERIFY-1486 | Implemented (`python/carnot/eval/cctu_executable_constraint_microbenchmark.py`) | Implemented (`tests/python/test_experiment_1486_cctu_executable_constraint_microbenchmark.py`) |
| REQ-VERIFY-1487 | Planned (`python/carnot/eval/v1_pairwise_self_verification_vs_energy.py`) | Planned (`tests/python/test_experiment_1487_v1_pairwise_self_verification_vs_energy.py`) |
| REQ-VERIFY-1494 | Planned (`python/carnot/eval/constrainprompt_validator_compiler_audit.py`) | Planned (`tests/python/test_experiment_1494_constrainprompt_validator_compiler_audit.py`) |
| REQ-VERIFY-1495 | Implemented (`python/carnot/eval/interwhen_monitor_prototype.py`) | Implemented (`tests/python/test_experiment_1495_interwhen_monitor_prototype.py`) |
| REQ-VERIFY-1496 | Planned (`python/carnot/eval/hover_safe_prefix_continuation_audit.py`) | Planned (`tests/python/test_experiment_1496_hover_safe_prefix_continuation_audit.py`) |
| REQ-VERIFY-1499 | Implemented (`python/carnot/eval/verifier_ensemble_dry_orthogonality_v2.py`) | Implemented (`tests/python/test_experiment_1499_verifier_ensemble_dry_orthogonality_v2.py`) |
| REQ-VERIFY-1500 | Implemented (`python/carnot/verify/latent_deterministic_gate.py`) | Implemented (`tests/python/test_experiment_1500_latent_deterministic_discipline_gate.py`) |
| REQ-VERIFY-1501 | Implemented (`python/carnot/verify/plan_graph_energy_adapter.py`) | Implemented (`tests/python/test_experiment_1501_gnnverifier_plan_graph_energy_adapter.py`) |
| REQ-VERIFY-1507 | Planned (`python/carnot/verify/safe_dsl_verifier_induction.py`) | Planned (`tests/python/test_experiment_1507_autopyverifier_safe_dsl_induction_pack.py`) |
| REQ-VERIFY-1508 | Planned (`python/carnot/verify/trigger_grammar_certificate_decoder.py`) | Planned (`tests/python/test_experiment_1508_trigger_grammar_certificate_decoder_audit.py`) |
| REQ-VERIFY-1509 | Implemented (`python/carnot/verify/executable_monitor_runtime_adapter.py`) | Implemented (`tests/python/test_experiment_1509_executable_monitor_runtime_adapter.py`) |
| REQ-VERIFY-1510 | Implemented (`python/carnot/verify/plan_graph_structural_contract_gate.py`) | Implemented (`tests/python/test_experiment_1510_plan_graph_structural_contract_gate.py`) |
| REQ-VERIFY-1520 | Implemented (`python/carnot/verify/runtime_contract_e2e_harness.py`) | Implemented (`tests/python/test_experiment_1520_runtime_contract_e2e_harness.py`) |
| REQ-VERIFY-1521 | Implemented (`python/carnot/verify/live_sota_contract_guided_repair.py`) | Implemented (`tests/python/test_experiment_1521_live_sota_contract_guided_repair.py`) |
| REQ-VERIFY-1522 | Implemented (`python/carnot/verify/constraint_dependency_graph_repair.py`) | Implemented (`tests/python/test_experiment_1522_constraint_dependency_graph_repair.py`) |
| REQ-VERIFY-1525 | Implemented (`python/carnot/verify/march_claim_isolation_ablation.py`) | Implemented (`tests/python/test_experiment_1525_march_claim_isolation_ablation.py`) |
| REQ-VERIFY-1537 | Implemented (`python/carnot/verify/beaver_prefix_bound_contracts.py`) | Implemented (`tests/python/test_experiment_1537_beaver_prefix_bound_contracts.py`) |
| REQ-VERIFY-1538 | Planned (`python/carnot/verify/residual_drift_commitment_ledger.py`) | Planned (`tests/python/test_experiment_1538_residual_drift_commitment_ledger.py`) |
| REQ-VERIFY-1541 | Implemented (`python/carnot/verify/claim_isolation_uncertainty_router.py`) | Implemented (`tests/python/test_experiment_1541_claim_isolation_uncertainty_router.py`) |
| REQ-VERIFY-1542 | Implemented (`python/carnot/verify/arm_ebm_soft_value_diagnostic.py`) | Implemented (`tests/python/test_experiment_1542_arm_ebm_soft_value_diagnostic.py`) |
| REQ-VERIFY-1556 | Implemented (`python/carnot/verify/arm_ebm_logprob_telemetry_repair.py`) | Implemented (`tests/python/test_experiment_1556_arm_ebm_logprob_telemetry_repair.py`) |
| REQ-VERIFY-1551 | Implemented (`python/carnot/verify/unified_contract_gate.py`) | Implemented (`tests/python/test_experiment_1551_automata_sat_unified_contract_gate.py`) |
| REQ-VERIFY-1552 | Implemented (`python/carnot/verify/residual_drift_repair_policy.py`) | Implemented (`tests/python/test_experiment_1552_residual_drift_repair_policy.py`) |
| REQ-VERIFY-1553 | Implemented (`python/carnot/verify/claim_isolation_router_scale.py`) | Implemented (`tests/python/test_experiment_1553_claim_isolation_router_scale.py`) |
| REQ-VERIFY-1554 | Planned (`python/carnot/verify/product_line_staged_scale_v4.py`) | Planned (`tests/python/test_experiment_1554_product_line_staged_scale_v4.py`) |
| REQ-VERIFY-1557 | Implemented (`python/carnot/verify/verification_compute_router.py`) | Implemented (`tests/python/test_experiment_1557_weaver_verification_compute_router.py`) |
| REQ-VERIFY-1562 | Implemented (`python/scripts/dt_brain_correlations_verification.py`) | Implemented (`tests/python/test_experiment_1562_brain_linear_ar_k_sweep.py`) |
| REQ-VERIFY-1571 | Implemented (`python/carnot/training/ar_reinforce_stepwise_baseline.py`) | Implemented (`tests/python/test_experiment_1571_step_wise_baseline_ar_reinforce.py`) |
| REQ-VERIFY-1578 | Implemented (`python/carnot/training/brain_reinforce_training_dynamics.py`) | Implemented (`tests/python/test_experiment_1578_brain_reinforce_training_dynamics.py`) |
| REQ-VERIFY-1580 | Implemented (`python/carnot/reporting/dccd_jsonschemabench_sota_structured_output_smoke.py`) | Implemented (`tests/python/test_experiment_1580_dccd_jsonschemabench_sota_structured_output_smoke.py`) |
| REQ-VERIFY-1588 | Implemented (`python/carnot/verifiers/dsl.py`) | Implemented (`tests/python/test_experiment_1588_nsvif_dsl.py`) |
| REQ-VERIFY-1591 | Implemented (`python/carnot/verifiers/dccd_adapter.py`) | Implemented (`tests/python/test_experiment_1591_dccd_adapter.py`) |
| REQ-VERIFY-1609 | Implemented (`python/carnot/pipeline/context_induction.py`) | Implemented (`tests/python/test_experiment_1609_context_induction.py`) |
| REQ-VERIFY-1640 | Implemented (`scripts/experiment_1640_nsvif_dsl.py`) | Implemented (`tests/python/test_experiment_1640_nsvif_dsl.py`) |
| REQ-VERIFY-1641 | Implemented (`python/carnot/pipeline/nsvif_sota.py`; legacy wrapper `scripts/experiment_1641_nsvif_sota.py`) | Implemented (`tests/python/test_pipeline_nsvif_sota.py`, `tests/python/test_experiment_1641_nsvif_sota.py`) |
| REQ-VERIFY-1642 | Implemented (`scripts/experiment_1642_llguidance.py`) | Implemented (`tests/python/test_experiment_1642_llguidance.py`) |
| REQ-VERIFY-1646 | Implemented (`scripts/experiment_1646_ebcn.py`) | Implemented (`tests/python/test_experiment_1646_ebcn.py`) |
| REQ-VERIFY-1666 | Implemented (`python/carnot/pipeline/nsvif_parser.py`) | Implemented (`tests/python/test_pipeline_nsvif_parser.py`) |
| REQ-VERIFY-1878 | Implemented (`python/carnot/pipeline/roce_validator_tree.py`) | Implemented (`tests/python/test_roce_validator_tree.py`) |
| REQ-VERIFY-1879 | Implemented (`python/carnot/verify/roce_beaver_lite_bounds.py`) | Implemented (`tests/python/test_roce_beaver_lite_bounds.py`) |
| REQ-VERIFY-1880 | Implemented (`python/carnot/verify/sota_roce_validator_eval.py`) | Implemented (`tests/python/test_sota_roce_validator_eval.py`) |
| REQ-VERIFY-2353 | Implemented (`python/carnot/repair/verge_repair.py`) | Implemented (`tests/python/test_verge_repair.py`) |

### REQ-VERIFY-1593: CDG Repair Acceptance Rates
The repository shall provide a CDG repair analysis tool that builds a CDG over runtime-contract cases, compares repair localization, and contrasts repair acceptance rates between flat check and CDG ordering using the mandated MODEL_SPECS.
Given a set of runtime-contract cases and candidate repairs,
When Exp 1593 analyzes repair ordering,
Then the tool shall generate a CDG graph, compute flat vs CDG repair acceptance rates, and save the artifact to `results/experiment_1593_cdg_repair.json`.

#### SCENARIO-VERIFY-1593: CDG Repair Execution
Given a valid runtime-contract manifest and candidate repair rows,
When Exp 1593 repair analysis is executed,
Then it outputs a valid JSON artifact with schema fields `status`, `cdg_nodes`, `cdg_edges`, `flat_acceptance_rate`, `cdg_acceptance_rate`, and `honest_verdict`.

### REQ-VERIFY-1603: EBCN Hidden-State Structural Violation Energy
The repository shall provide a deterministic, CPU-only EBCN scorer prototype for
Exp 1603 that evaluates structural coherence without autoregressive generation.

The scorer shall:

- define an EBCN module under `python/carnot/verify/` that accepts fixed hidden
  states and optional proposition metadata;
- use two independently parameterized attention heads over hidden states, one
  for support/coherence evidence and one for contradiction/violation evidence;
- return a scalar non-negative structural violation energy where direct logical
  contradictions score higher than consistent synthetic traces;
- provide deterministic synthetic logical contradiction cases without calling
  an LLM, decoder, or autoregressive generator;
- write `results/experiment_1603_ebcn.json` with `status`, `experiment_id`,
  `ebcn_scorer_ready`, `dual_head_attention_used`,
  `autoregressive_generation_used`, `hidden_state_source`,
  `synthetic_cases_total`, `contradiction_cases`, `consistent_cases`,
  `contradiction_mean_energy`, `consistent_mean_energy`, `energy_gap`,
  `tests_run`, and `honest_verdict`; and
- set terminal `status="complete"` only when the dual-head scorer runs on the
  synthetic cases, contradiction energy exceeds consistent energy, and
  `autoregressive_generation_used=false`.

#### SCENARIO-VERIFY-1603: Synthetic Contradictions Have Higher EBCN Energy
Given deterministic hidden states for consistent and contradictory logical
traces,
When the EBCN scorer applies its support and contradiction attention heads,
Then contradiction traces receive higher structural violation energy than
consistent traces,
And `results/experiment_1603_ebcn.json` records complete Exp 1603 metrics
without any autoregressive generation path.

### REQ-VERIFY-1609: Residual-Drift Context Induction
The repository shall provide an Exp 1609 deterministic induction loop that
mines recent residual-drift failure logs and generates context-sensitive
constraint candidates for future gate insertion.

The induction loop shall:

- load recent residual-drift rows from `results/residual_drift_commitment_ledger_1538.jsonl`,
  `results/residual_drift_repair_policy_1552.jsonl`, and `logs/conductor.log`
  when present, preserving missing-source blockers without fabricating rows;
- select only satisfiable residual-drift failures and rejected or unrepaired
  drift repair rows as candidate evidence, while keeping true contradictions as
  exclusion evidence;
- group evidence by source domain, localized span, repair kind, validator, and
  structural contract family where available so induced constraints are
  context-sensitive rather than global string rules;
- generate at least one candidate constraint when qualifying drift evidence is
  present, with stable IDs, trigger context, predicate description, positive and
  negative evidence case IDs, support count, confidence score, and guardrails
  against true-contradiction overreach;
- write `results/experiment_1609_context_induction.json` with `status`,
  `experiment_id`, `context_induction_ready`, `failure_logs_mined`,
  `source_paths`, `candidate_constraints_generated`,
  `selected_candidate`, `constraint_candidates`, `true_contradiction_exclusions`,
  `blockers`, `focused_tests_passed`, and `honest_verdict`; and
- set terminal `status="complete"` only when at least one qualifying residual
  drift failure is mined, at least one candidate is generated, true
  contradiction rows are excluded from positive evidence, and focused tests have
  passed.

#### SCENARIO-VERIFY-1609: Context-Sensitive Constraints Come From Residual Drift
Given residual-drift ledger rows, repair-policy rows, and recent conductor
failure logs,
When Exp 1609 mines failures and induces a new constraint candidate,
Then the candidate trigger includes source-domain and localized-context fields,
positive evidence contains satisfiable drift failures,
true contradictions remain exclusion evidence,
and `results/experiment_1609_context_induction.json` records a complete
bounded artifact without modifying the research conductor.
_candidates`, `true_contradiction_exclusions`,
  `blockers`, `focused_tests_passed`, and `honest_verdict`; and
- set terminal `status="complete"` only when at least one qualifying residual
  drift failure is mined, at least one candidate is generated, true
  contradiction rows are excluded from positive evidence, and focused tests have
  passed.

#### SCENARIO-VERIFY-1609: Context-Sensitive Constraints Come From Residual Drift
Given residual-drift ledger rows, repair-policy rows, and recent conductor
failure logs,
When Exp 1609 mines failures and induces a new constraint candidate,
Then the candidate trigger includes source-domain and localized-context fields,
positive evidence contains satisfiable drift failures,
true contradictions remain exclusion evidence,
and `results/experiment_1609_context_induction.json` records a complete
bounded artifact without modifying the research conductor.


### REQ-VERIFY-1946: Ontology NN Topological Verification

The repository shall provide an Ontology NN Topological Verification layer that:
- computes Forman-Ricci curvature over a semantic dependency graph of constraints;
- applies Deep Delta Learning to project invalid constraint combinations back to the feasible manifold;
- operates without requiring any legacy small model for headline metrics;
- includes mutually exclusive logical constraint tests;
- writes `results/experiment_1946_ontology_nn_topological.json` with fields `status="complete"`, `forman_ricci_curvature_computed`, `deep_delta_projection_applied`, `mutually_exclusive_tests_passed`, and `honest_verdict`.

### SCENARIO-VERIFY-1946: Topological Verification Enforces Coherence
Given a set of mutually exclusive logical constraints,
When the topological verifier evaluates the constraint graph,
Then it identifies invalid combinations via negative Forman-Ricci curvature,
and projects them to the feasible manifold using Deep Delta Learning,
and records the results in the experiment JSON artifact.

### REQ-VERIFY-1952: GNN Benchmarking Audit against Z3/SAT baselines
The repository shall provide an audit pipeline that:
- Generates 100 hard 3-SAT constraint networks near the phase transition threshold.
- Runs the internal continuous solver vs Z3 execution time and success rate.
- Synthesizes the findings into an objective performance gap report.
- Writes `results/experiment_1952_gnn_benchmarking_audit.json` with the required fields.

### SCENARIO-VERIFY-1952: Performance Gap is Reported
Given 100 hard 3-SAT networks near the phase transition threshold,
When the internal solver and Z3 are executed,
Then execution time and success rate are compared,
And an objective performance gap report is synthesized,
And the JSON artifact is written with all fields populated.

### REQ-VERIFY-1998: Live IT Baselines with GSM8K

The repository shall establish real baselines with instruction-tuned models on GSM8K using the new SMT extractor.
- Runs 200 GSM8K questions.
- Uses `inference_mode="live_gpu"` in all results.
- Calculates TP and FP rates.
- Writes `results/experiment_1998_live_it_baselines_gsm8k.json` artifact.

### SCENARIO-VERIFY-1998: Run GSM8K Baseline
Given 200 GSM8K questions and the new SMT extractor,
When the models run inference with `live_gpu` mode,
Then the baselines TP and FP rates are calculated and saved to the JSON artifact.

### REQ-VERIFY-1694: NLA-Class 16th Verifier Prototype

The repository shall provide an NLA-class 16th verifier prototype that trains a sparse autoencoder on a small held-out corpus and predicts verifier ensemble agreement.

### SCENARIO-VERIFY-1694: SAE Predicts Ensemble Agreement

Given a 60-example held-out set, the SAE probe predicts ensemble agreement and reports Wilson 95% CIs for TPR and FPR.

### REQ-VERIFY-1716: NLA Eval-Awareness Drift Check
The repository shall test the NLA-class 16th verifier for eval-awareness drift by computing the TPR/FPR gap between an eval-style and production-style corpus of 100 examples total.


### REQ-VERIFY-1732: Behavioral Entanglement Reweighting for k=16 Ensemble

The repository shall replicate the de-entangled reweighting algorithm on Carnot's k=16 setup (arXiv:2604.07650) to measure lift on the adversarial corpus.
- Implement the reweighting algorithm to adjust weights for the 16 verifiers based on failure covariance.
- Evaluate on the adversarial test corpus to measure accuracy lift.
- Document the new weights and pass rates in `docs/research-notes/k16_reweighting.md`.
- Produce a terminal JSON artifact `results/experiment_1732_k16_reweighting.json` with status, accuracy_lift_pct, and honest_verdict.

### SCENARIO-VERIFY-1732: De-entangled Reweighting Evaluation
Given a k=16 verifier ensemble and an adversarial test corpus,
When the de-entangled reweighting algorithm adjusts weights based on failure covariance,
Then it records the new weights, evaluates accuracy lift, and writes the required artifacts.


### REQ-VERIFY-1733: NLA PWA Formal Verifier

The repository shall provide an NLA PWA Formal Verification abstraction that:
- abstracts the NLA core activation (ReLU) into Piecewise Affine (PWA) pieces;
- formulates the safety property (e.g., minimum confidence bound via max MSE for a given input radius) as a Z3 script/MILP formulation;
- computes a strict theoretical upper bound on MSE via interval arithmetic;
- runs deterministically on CPU;
- writes `results/experiment_1733.json` with `status="complete"`, `pwa_abstraction_generated=true`, the theoretical bound, and an honest verdict.

### SCENARIO-VERIFY-1733: NLA PWA Abstraction Verification
Given a trained or randomly initialized NLA MinimalSAE and a target input radius,
When the PWA abstraction and formal bounds are generated,
Then a valid Z3 script (.smt2 format) is produced with the MSE property assertion,
And a positive theoretical upper bound is calculated using interval arithmetic,
And the JSON artifact records `pwa_abstraction_generated=true` and `status="complete"`.

### REQ-VERIFY-1742: EB-SLE Reward Hacking Prevention Prototype
Carnot MUST implement an EB-SLE constraint that specifically detects reward hacking.
The verifier MUST check whether the repaired generation merely loops or exploits syntax vs solving the problem.
It MUST evaluate against known failure cases from prior milestones using MODEL_SPECS: unsloth/Qwen3.6-35B-A3B-GGUF.

### SCENARIO-VERIFY-1742: EB-SLE Hack Detection
Given an initial response and a repaired response,
When the repair generation loops or exploits syntax,
Then the EB-SLE Hack Verifier returns True,
And the JSON artifact records `ebsle_hack_detected=true` and `status="complete"`.

### REQ-VERIFY-1740: QAOD Verifier

The system shall implement QAOD orthogonal decomposition of answer representations against question context. The residual magnitude is the hallucination indicator.

### SCENARIO-VERIFY-1740: QAOD vs NLA Head-to-Head

Given the exp1716 60-example test corpus, when QAOD is evaluated against NLA-SAE, then it reports head-to-head TPR/FPR in results/experiment_1740_qaod_vs_nla_head_to_head.json.

### REQ-VERIFY-1771: SKM Projection Baseline

The system shall implement an SKM-style iterative projection baseline for Carnot's linear constraints on 50 linear equality/inequality constraints from the CCTU benchmark. The SKM randomized projection algorithm MUST verify that post-projection outputs achieve strictly zero constraint violations.
The artifact MUST be written to `results/experiment_1771_tskm_projection.json` with schema `carnot.tskm_projection.v1`, run_date `20260515`, boolean `zero_violations_achieved`, float `mean_projection_steps`, and an `honest_verdict`.

### SCENARIO-VERIFY-1771: SKM Iterative Projection Achieves Zero Violations

Given 50 CCTU linear constraints, when the SKM randomized projection algorithm executes, then post-projection outputs achieve strictly zero constraint violations and the required JSON artifact is produced.

### REQ-VERIFY-2354: Eidoku CSP Arithmetic Gate

The repository shall provide a CPU-only Eidoku CSP gate in
`python/carnot/verify/eidoku_csp.py` that:
- extracts numeric variable assignments from a response string;
- evaluates hard arithmetic constraint expressions without `eval`;
- rejects missing variables, unsafe syntax, and violated constraints as gate
  violations;
- exposes `EidokuCspGate.validate(response: str, constraints: list[str]) ->
  dict` with `gate_passed` and `violations` fields; and
- reports `results/experiment_2354_eidoku_csp.json` with
  `csp_gate_accuracy` on exactly 50 deterministic arithmetic examples.

### SCENARIO-VERIFY-2354: Eidoku CSP Gate Validates Arithmetic Corpus

Given a deterministic 50-example arithmetic constraint corpus with seed 42,
When `EidokuCspGate` evaluates each response against its hard constraints,
Then the reported `csp_gate_accuracy` is the fraction of examples correctly
classified, `n_eval_examples` is 50, and `eidoku_gate_validated` is true only
when the accuracy is at least 0.75.

### REQ-VERIFY-2548: Tier 0 Real-Corpus Hallucination Validation

The repository shall provide a deterministic, CPU-only real-corpus validation
runner for Tier 0r, Tier 0s, and Tier 0u verifier prototypes that:

- records which verifier imports and corpus preconditions were checked;
- loads a labeled FoVer corpus from local `data/` or `results/` artifacts before
  using any synthetic fallback;
- scores every selected labeled example with each importable verifier;
- computes AUROC against hallucination labels where hallucination is `1` and
  grounded/correct is `0`;
- compares each real-corpus AUROC with the verifier's prior synthetic claim;
- marks a verifier paper-citable only when AUROC was measured on a real corpus
  with at least 50 real labeled examples; and
- writes `results/experiment_2548_real_corpus_validation.json` with
  `honest_verdict`, `tier0r_real_auroc`, `tier0s_real_auroc`,
  `tier0u_real_auroc`, `corpus_type`, `n_real`, `paper_citable`,
  `preconditions_checked`, `duration_s`, and `random_seed`.

If fewer than 50 real labeled examples are available, the runner may supplement
with at least 30 diverse synthetic examples, but those AUROC values MUST be
reported as non-citable.

### SCENARIO-VERIFY-2548: FoVer Corpus Produces Citable Tier 0 AUROC

Given a local FoVer corpus with at least 50 labeled examples and both
hallucination classes,
When the Exp 2548 runner evaluates every importable Tier 0 verifier,
Then the artifact reports `corpus_type="real"`, records `n_real` from the
selected corpus, computes one AUROC field per importable verifier, and sets
each verifier's `paper_citable` flag exactly when the real-corpus sample-size
gate is satisfied.

### REQ-VERIFY-2828: FoVer FR-11 Memory Leakage Isolation

The repository shall provide an Exp 2828 FoVer leakage-isolation runner that
writes `results/experiment_2828_fover_memory_leakage_isolation.json` for a
1000-example FoVer subset and five adversarial replication seeds
`[42, 137, 271, 314, 1729]`.

Before live scoring, the runner MUST verify the required runtime resources:
`python3` can import `torch` and reports `torch.cuda.is_available()`,
`data/fover_corpus.jsonl` exists with at least 1000 rows,
`results/nexus_constraint_memory_v2.json` exists, and a
Qwen3.6-35B-A3B GGUF cache is discoverable. If any precondition is missing,
the runner MUST write a terminal `blocked_<resource>` artifact rather than
fabricating AUROC, duration, or per-verifier scores.

For successful live runs, the runner MUST compare Condition A (full FR-11
self-learning state loaded) against Condition B (FR-11 state files moved
non-destructively to `/tmp/fr11_state_backup_<seed>/`, Python restarted, and
the ensemble reloaded without NEXUS, constraint-template, or session-memory
state). It MUST restore every state file after each seed and prove SHA256
matches the original state manifest.

The artifact MUST include the dual-condition AUROC means and standard
deviations, `learning_contribution = A - B`, per-verifier learning
contributions, the FR-11 state-file manifest with SHA256 and byte size,
`state_files_restored_sha_match`, `reproducibility_checksum`, `model_specs`,
`duration_s`, `preconditions_checked`, and an honest methodology note.

### SCENARIO-VERIFY-2828: Missing CUDA Blocks Without Fabricated Metrics

Given the FoVer corpus and FR-11 state files are present but the
`python3` Torch/CUDA precondition fails,
When the Exp 2828 runner executes,
Then it writes `results/experiment_2828_fover_memory_leakage_isolation.json`
with `honest_verdict` prefixed `blocked_cuda`, null AUROC metrics, an empty
per-verifier learning contribution map, the FR-11 state-file hashes, and
`state_files_restored_sha_match=true` because no destructive reset was
attempted.

### REQ-VERIFY-2836: FoVer Dual-Condition FR-11 Isolation With Venv CUDA Gate

The repository shall provide an Exp 2836 FoVer memory-leakage isolation runner
that writes `results/experiment_2836_fover_memory_leakage_isolation.json` for
the operator-mandated 1000-example FoVer subset and five replication seeds
`[42, 137, 271, 314, 1729]`.

Before any AUROC measurement, the runner MUST check the exact CUDA command
`.venv/bin/python3 -c "import torch; assert torch.cuda.is_available() and
torch.cuda.device_count() >= 1"`, `data/fover_corpus.jsonl`,
`results/nexus_constraint_memory_v2.json`, and a real cached
Qwen3.6-35B-A3B GGUF file. A HuggingFace `.no_exist` marker is not a cached
model. If the CUDA assertion fails after the 2026-05-21 torch+cu128 fix, the
artifact MUST use `blocked_cuda_post_fix_regression` and include the full torch
version probe. If any other precondition is unavailable, the artifact MUST use
the corresponding terminal `blocked_<resource>` prefix and MUST NOT infer,
replay, cap, or synthesize AUROC values.

For a successful measurement, the runner MUST use the production verifier
ensemble v7b under two memory conditions: Condition A with all FR-11/NEXUS/
constraint-template/session-memory state visible, and Condition B after all
identified FR-11 state files are moved non-destructively under
`/tmp/fr11_state_backup_<seed>/`, Python is restarted, and logs or structured
checks confirm no NEXUS, constraint-template, or session-memory state loaded.
The same FoVer subset must be scored for both conditions for each seed, every
state file must be restored, and the restored SHA256 values must match the
initial state manifest.

The artifact MUST include `honest_verdict`, both condition AUROC means and
standard deviations, `learning_contribution = A - B`,
`per_verifier_learning_contribution`, `fr11_state_files`,
`state_files_restored_sha_match`, `n_examples`, `n_seeds`,
`random_seeds_used`, `reproducibility_checksum`, `model_specs`, real
`duration_s`, `preconditions_checked`, and an honest `methodology_note`.

### SCENARIO-VERIFY-2836: Missing Qwen Cache Blocks Without Metrics

Given `.venv/bin/python3` reports CUDA availability and the FoVer/NEXUS files
exist, but no real Qwen3.6-35B-A3B GGUF file is cached,
When the Exp 2836 FoVer isolation runner executes,
Then it writes `results/experiment_2836_fover_memory_leakage_isolation.json`
with `honest_verdict` prefixed `blocked_model_cache`, null AUROC metrics, empty
per-verifier contribution maps, populated precondition evidence, the FR-11
state-file manifest, and `state_files_restored_sha_match=true`.

### SCENARIO-VERIFY-2836-LIVE: Successful Dual-Condition Run Is Fully Auditable

Given CUDA, FoVer, NEXUS, a real Qwen3.6-35B-A3B GGUF cache, FR-11 state files,
and a live production ensemble-v7b measurement backend are available,
When the runner completes all five seeds under both memory conditions,
Then the artifact reports complete production and architecture-only AUROC
summaries, per-verifier learning contributions, per-seed condition details,
restored FR-11 state hashes, a real duration, and an `honest_verdict` prefixed
`complete:`.

### REQ-VERIFY-2837: FoVer Memory Leakage Isolation v3 Gated By Exp 2836

The repository shall provide an Exp 2837 FoVer memory-leakage runner that
writes `results/experiment_2837_fover_memory_leakage_v3.json` for a
1000-example FoVer subset and the five replication seeds
`[42, 137, 271, 314, 1729]`.

Before scoring, the runner MUST load
`results/experiment_2836_sota_runtime_preflight.json`, assert
`sota_runtime_ready=true`, use the artifact's `selected_python` for condition
scoring subprocesses, and record the usable mandated SOTA GGUF model path
reported by Exp 2836. It MUST also confirm the FoVer corpus has at least 1000
labeled rows and that FR-11/NEXUS/session memory files are present. If any
precondition fails, it MUST write a terminal `blocked_<resource>` artifact with
populated `preconditions_checked` and null AUROC metrics.

For successful runs, the runner MUST compare Condition A (production verifier
with FR-11 state visible) against Condition B (architecture-only verifier after
FR-11/NEXUS/session state files are non-destructively moved aside). Condition
A and B MUST be scored in separate Python subprocesses for each seed, Condition
B MUST verify no FR-11 state files are visible to its subprocess, and the
runner MUST restore every state file afterward with SHA256 equality to the
initial state manifest.

The artifact MUST include `honest_verdict`, `condition_a_production_auroc_mean`,
`condition_b_architecture_only_auroc_mean`, `learning_contribution`,
`per_seed_results`, `fr11_state_files`, `state_files_restored_sha_match`,
`model_specs`, `preconditions_checked`, `duration_s`, confidence intervals for
both condition means, and an honest methodology note describing the
non-destructive reset.

### SCENARIO-VERIFY-2837: Architecture-Only Condition Runs With State Absent

Given Exp 2836 reports a ready SOTA runtime, FoVer data has at least 1000
labeled rows, and FR-11 state files exist,
When the Exp 2837 runner scores the five seeds,
Then each per-seed result records a production AUROC from a state-visible
subprocess and an architecture-only AUROC from a separate subprocess where
`condition_b_state_visible_count=0`, computes `learning_contribution` as
Condition A minus Condition B, and reports
`state_files_restored_sha_match=true`.

### SCENARIO-VERIFY-2837-BLOCKED: Runtime Gate Blocks Without Metrics

Given the Exp 2836 preflight artifact is missing, not ready, or has no usable
mandated SOTA model path,
When the Exp 2837 runner executes,
Then it writes `results/experiment_2837_fover_memory_leakage_v3.json` with
`honest_verdict` prefixed `blocked_`, null condition AUROC metrics, an empty
`per_seed_results` list, populated `preconditions_checked`, and no inferred
learning contribution.

### REQ-VERIFY-2850: FoVer Dual-Condition Integrity Rerun

The repository shall provide an Exp 2850 FoVer dual-condition integrity runner
that writes `results/experiment_2850_fover_dual_condition_integrity_v4.json`
for the same 1000-example FoVer subset protocol and replication seeds
`[42, 137, 271, 314, 1729]` without claiming live-model provenance unless a
local LLM is actually invoked.

Before scoring, the runner MUST check `data/fover_corpus.jsonl`,
`results/nexus_constraint_memory_v2.json` when present, and the local
`sklearn`/`numpy` metric dependencies. If the FoVer corpus is missing or has
fewer than 1000 labeled rows, the runner MUST write
`honest_verdict="blocked_fover_dataset"` with populated
`preconditions_checked` and null AUROC fields.

For successful dataset-only scoring, the runner MUST score Condition A with
production FR-11 state visible, score Condition B after FR-11/NEXUS/session
state files are non-destructively moved aside, restore every state file, and
record whether restored SHA256 values match the initial manifest. The artifact
MUST set `live_model_invoked=false`, `compute_bound_claim=false`, omit
GGUF/CUDA/model-spec claims, include a top-level `random_seed`, compute
`reproducibility_checksum` from input hashes, FR-11 state hashes, seeds, and
score rows, and include the `scripts/adversarial_verify.py` result summary
when the script is available.

The artifact MUST include `honest_verdict`,
`condition_a_production_auroc_mean`,
`condition_a_production_auroc_std`,
`condition_b_architecture_only_auroc_mean`,
`condition_b_architecture_only_auroc_std`, `learning_contribution`,
`per_verifier_learning_contribution`, `n_examples`, `n_seeds`,
`random_seed`, `random_seeds_used`, `reproducibility_checksum`,
`fr11_state_files`, `state_files_restored_sha_match`,
`live_model_invoked`, `compute_bound_claim`, `preconditions_checked`,
`duration_s`, `adversarial_verify_passed`, `adversarial_verify_flags`, and
`run_date`.

### SCENARIO-VERIFY-2850: Dataset-Only Rerun Does Not Overclaim Runtime

Given the FoVer corpus has at least 1000 labeled rows and FR-11 state files are
present,
When the Exp 2850 runner completes all five seeds,
Then it writes `results/experiment_2850_fover_dual_condition_integrity_v4.json`
with complete dual-condition AUROC summaries, restored state hashes, a
score-row-derived reproducibility checksum, `live_model_invoked=false`,
`compute_bound_claim=false`, no GGUF/CUDA/model-spec claims, and adversarial
verification flags included.

### SCENARIO-VERIFY-2850-BLOCKED: Missing FoVer Dataset Blocks Without Metrics

Given `data/fover_corpus.jsonl` is missing or has fewer than 1000 labeled rows,
When the Exp 2850 runner executes,
Then it writes `results/experiment_2850_fover_dual_condition_integrity_v4.json`
with `honest_verdict="blocked_fover_dataset"`, null condition AUROC metrics,
empty per-verifier learning contributions, populated precondition evidence,
`live_model_invoked=false`, and `compute_bound_claim=false`.

### REQ-VERIFY-2867: Residual-Drift MUS Conflict Prioritizer Diagnostic

The repository shall provide an Exp 2867 residual-drift plus MUS conflict
prioritization diagnostic that reads
`results/experiment_2865_cross_corpus_matrix_v5.json` and writes
`results/experiment_2867_drift_mus_prioritizer_v2.json`.

Before extracting evidence, the diagnostic MUST confirm that the Exp 2865
matrix exists and has `cross_corpus_matrix_built=true`. If the matrix is not
built, the artifact MUST set `drift_mus_diagnostic_ready=false`, report zero
failure rows, and avoid inferring missing or blocked corpus metrics.

When the matrix is built, the diagnostic MUST use only rows classified `clean`
inside `verifier_corpus_dual_matrix` and MAY load the clean rows' declared
source artifacts for verifier-level evidence. It MUST identify weak or
below-random AUROC evidence as failure rows, build a small hypergraph whose
nodes are constraint families, verifier-failure signals, and failure classes,
and whose hyperedges represent available co-failure or residual-drift/MUS-proxy
relationships. If exact MUS certificates are not present, the artifact MUST say
that the edges are diagnostic proxies rather than certified MUSes.

The diagnostic MUST compare a deterministic residual-drift hypergraph
message-passing heuristic against deterministic random and degree baselines
using the same conflict-discovery metric. It MUST NOT claim a trained HGNN
policy. The artifact MUST include `honest_verdict`,
`drift_mus_diagnostic_ready`, `n_failure_rows`, `failure_class_counts`,
`hypergraph_nodes`, `hypergraph_hyperedges`,
`hgnn_inspired_heuristic_name`, `baseline_random_checks_to_conflict`,
`baseline_degree_checks_to_conflict`, `heuristic_checks_to_conflict`,
`heuristic_improvement_vs_best_baseline`, `recommended_repairs`,
`preconditions_checked`, `random_seed`, `reproducibility_checksum`,
`field_principles`, `run_date="20260522"`, and `duration_s`.

### SCENARIO-VERIFY-2867: Clean Matrix Produces Honest Drift/MUS Diagnostic

Given Exp 2865 contains clean FoVer and HaluEval/FEVER rows and declares the
cross-corpus matrix built,
When the Exp 2867 diagnostic runs,
Then it extracts failure evidence only from those clean rows and their declared
clean source artifacts, writes the required schema fields, reports a non-empty
hypergraph and baseline comparison, recommends repairs without claiming a
trained HGNN policy, and records a reproducibility checksum over the matrix,
source evidence, failure rows, hypergraph, and metric payload.

### SCENARIO-VERIFY-2867-BLOCKED: Missing Built Matrix Blocks Without Inference

Given the Exp 2865 matrix is missing or has `cross_corpus_matrix_built=false`,
When the Exp 2867 diagnostic runs,
Then it writes `results/experiment_2867_drift_mus_prioritizer_v2.json` with
`honest_verdict` prefixed `blocked_`, `drift_mus_diagnostic_ready=false`,
zero failure rows, zero hypergraph nodes and hyperedges, populated
precondition evidence, and no inferred repair-prioritization metrics.

### REQ-VERIFY-2877: HaluEval/FEVER Tiny Exact Frontier Expansion

The repository shall provide an Exp 2877 exact-frontier expansion runner that
reads the clean Exp 2866 exact arithmetic frontier evidence and the clean Exp
2864 HaluEval/FEVER local calibration artifact, then scans only the resolved
HaluEval and FEVER manifests declared by Exp 2864.

The runner MUST include HaluEval/FEVER rows in the exact frontier only when a
manually encoded, deterministic local constraint exists for the row and the
row's text anchors still match the manifest bytes. Supported constraints MAY
cover safe-prefix exact answers, direct contradiction, anchored entailment, or
arithmetic-like date/year comparisons. The runner MUST NOT use LLM
autoformalization and MUST keep every row without such a manual exact
constraint outside the exact frontier with an unsupported-row reason.

The terminal artifact MUST be written to
`results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json` and
include `honest_verdict`, `frontier_expansion_ready`, `source_artifacts`,
`selection_rule`, `n_candidate_rows`, `n_exact_supported_rows`,
`n_unsupported_rows`, `unsupported_reasons`, `exact_solver_backend`,
`certificates`, `tests_run`, `field_principles`, `run_date="20260522"`, and a
measured `duration_s`.

### SCENARIO-VERIFY-2877: Manual Exact Constraints Touch Non-FoVer Rows

Given Exp 2864 resolves local HaluEval and FEVER manifests and Exp 2866
contains clean exact Z3 arithmetic frontier evidence,
When the Exp 2877 runner scans the manifests,
Then it writes the required artifact with at least one supported HaluEval row
and at least one supported FEVER row, includes per-row certificates with solver
status, records the deterministic selection rule, and reports all remaining
manifest rows under unsupported reasons rather than inferring natural-language
coverage.

### REQ-VERIFY-2878: HaluEval/FEVER Error-Verifiability And Label-Consistency Audit

The repository shall provide an Exp 2878 audit runner that reads the clean Exp
2864 HaluEval/FEVER calibration artifact, its resolved local manifests, the Exp
2865 cross-corpus matrix, and any available `.271` HaluEval/FEVER verifier
traces such as Exp 2867 failure rows and Exp 2877 exact-frontier certificates.

The runner MUST avoid remote LLM calls and MUST NOT synthesize examples. It
MUST replay only existing manifest rows and labels, score rows with existing
local verifier outputs or deterministic injected test scorers, assign each row
to exactly one deterministic bucket from `data-grounding`, `reasoning-chain`,
`extraction/format`, `unsupported`, and `unknown`, and compute:

- the fraction of labeled error rows with an actionable violated constraint;
- agreement between dataset labels and local verifier direction;
- bucket-level AUROC where both labels and finite scores are available; and
- an honest explanation of whether weak scalar AUROC is driven by data-driven
  errors, reasoning-driven errors, or missing verifier coverage.

The terminal artifact MUST be written to
`results/experiment_2878_halueval_fever_error_verifiability_v1.json` and
include `honest_verdict`, `error_verifiability_ready`, `source_artifacts`,
`n_rows_audited`, `error_buckets`, `actionable_localization_rate`,
`label_consistency_rate`, `bucket_level_metrics`, `weak_auroc_explanation`,
`remote_llm_called=false`, `tests_run`, `field_principles`,
`run_date="20260522"`, and measured `duration_s`.

### SCENARIO-VERIFY-2878: Local Audit Separates Coverage And Direction Failures

Given Exp 2864 resolves local HaluEval and FEVER manifests and Exp 2865 marks
the HaluEval/FEVER row clean,
When the Exp 2878 audit runner executes,
Then it writes the required artifact without calling a remote LLM, reports
bucket counts that reconcile with the audited manifest row count, computes
label-consistency and actionable-localization rates from manifest labels and
local verifier directions, includes bucket AUROC values only where computable,
and explains the weak scalar AUROC without inventing missing verifier traces.

### REQ-VERIFY-2892: VeriCoT-Style Exact Frontier Expansion

The repository shall provide an Exp 2892 VeriCoT-style exact-frontier runner
that reads Exp 2877 exact HaluEval/FEVER certificates, Exp 2878
error-verifiability evidence, and the Exp 2888 TruthfulQA taxonomy manifest
when it exists. The runner MUST select only rows whose premise anchors and
logical steps can be encoded deterministically as local formal checks. It MUST
NOT call an LLM for autoformalization, and it MUST keep unsupported rows
outside the exact frontier with explicit unsupported reasons.

The terminal artifact MUST be written to
`results/experiment_2892_vericot_exact_frontier_expansion_v1.json` and include
`honest_verdict`, `vericot_frontier_ready`, `source_artifacts`,
`selection_rule`, `n_candidate_rows`, `n_vericot_supported_rows`,
`n_unsupported_rows`, `unsupported_reasons`, `solver_backend`,
`formal_checks`, `autoformalization_llm_called=false`, `tests_run`,
`field_principles`, `run_date="20260523"`, and measured `duration_s`.

### SCENARIO-VERIFY-2892: Deterministic Logical Steps Expand The Frontier

Given Exp 2877 contains bounded exact certificates, Exp 2878 contains local
error-verifiability evidence, and Exp 2888 may contain TruthfulQA taxonomy rows,
When the Exp 2892 runner executes,
Then it writes the required artifact without calling an LLM, promotes only
certificates that can be rendered as deterministic premise-grounded logical
steps, records each Z3 or local deterministic formal check, reports every
candidate row not promoted under unsupported reasons, and reconciles candidate,
supported, and unsupported row counts.

### REQ-VERIFY-2829: MBPP Dual-Memory Ensemble Evaluation

The repository shall provide an Exp 2829 MBPP dual-memory runner that writes
`results/experiment_2829_mbpp_ensemble_eval.json` for exactly 100 examples
from the MBPP sanitized test split and five adversarial replication seeds
`[42, 137, 271, 314, 1729]`.

Before any live MBPP inference or verifier scoring, the runner MUST verify
CUDA availability, HuggingFace MBPP dataset access, a discoverable
Qwen3.6-35B-A3B GGUF cache, and present FR-11 state files. If any
precondition is missing, the runner MUST write a terminal `blocked_<resource>`
artifact rather than fabricating AUROC, pass@1, duration, candidate, or
per-verifier scores.

For successful live runs, the runner MUST generate or reuse the same
problem/candidate pairs for both memory conditions: Condition A with full
FR-11 state and Condition B after non-destructively moving FR-11 state to
`/tmp`, restarting Python, and scoring architecture-only verifier behavior.
The runner MUST restore every FR-11 state file and prove SHA256 matches the
original state manifest.

The artifact MUST include `honest_verdict`, corpus identity, `n_problems`,
`n_seeds`, Condition A and B AUROC means and standard deviations,
`learning_contribution = A - B`, per-verifier AUROC lists for both
conditions, `vanilla_qwen36_pass_at_1`, random seeds, reproducibility
checksum, model specs, real `duration_s`, precondition evidence, FR-11
state-file hashes and byte sizes, `state_files_restored_sha_match`, and an
honest methodology note.

### SCENARIO-VERIFY-2829: Missing Preconditions Block MBPP Metrics

Given the Exp 2829 runner starts in an environment missing CUDA, MBPP dataset
access, the Qwen3.6 GGUF cache, or FR-11 state files,
When the runner checks preconditions before live inference,
Then it writes `results/experiment_2829_mbpp_ensemble_eval.json` with
`honest_verdict` prefixed `blocked_`, null AUROC and pass@1 metrics, empty
per-verifier AUROC maps, populated `preconditions_checked`, and no inferred
benchmark numbers.

### SCENARIO-VERIFY-2829-LIVE: Successful MBPP Run Reports Dual Conditions

Given all Exp 2829 live-resource preconditions are available and a real
Qwen3.6 MBPP generation plus ensemble-scoring backend is configured,
When the runner evaluates all five seeds under production and
architecture-only memory conditions,
Then the artifact reports complete dual-condition AUROC summaries, per-seed
results, per-verifier AUROC lists for both conditions, restored FR-11 state
hashes, `vanilla_qwen36_pass_at_1`, and an `honest_verdict` prefixed
`complete:`.

### REQ-VERIFY-MBPP-2837: MBPP Ensemble v7b Retry Artifact

The repository shall provide an Exp 2837 MBPP ensemble-v7b runner that writes
`results/experiment_2837_mbpp_ensemble_eval.json` using the same live-resource
discipline as Exp 2829: check CUDA, HuggingFace MBPP sanitized test access, a
real cached Qwen3.6-35B-A3B-GGUF file, and FR-11 state files before candidate
generation or verifier scoring. If any precondition is absent, the runner MUST
write a `blocked_<resource>` artifact with all required schema keys present,
unmeasured metric values left null, and no inferred AUROC, pass@1, candidate,
or per-verifier values.

For successful runs, the runner MUST use exactly 100 MBPP sanitized test
problems, the five seeds `[42, 137, 271, 314, 1729]`, live-GPU Qwen3.6
candidate generation, MBPP test execution for pass/fail labels, Condition A
production ensemble-v7b scoring with FR-11 state visible, and Condition B
architecture-only scoring after FR-11 state is moved aside and Python is
restarted. It MUST restore every FR-11 state file and verify SHA256 equality.

The artifact MUST include principle annotations for the required fields:
`honest_verdict`, `corpus`, `n_problems`, `n_seeds`, Condition A/B AUROC mean
and standard deviation, `learning_contribution`, per-verifier AUROC lists for
both conditions, `vanilla_qwen36_pass_at_1`, `random_seeds_used`,
`reproducibility_checksum`, `model_specs`, `duration_s`,
`preconditions_checked`, `fr11_state_files`, `state_files_restored_sha_match`,
and `methodology_note`.

### SCENARIO-VERIFY-MBPP-2837: Missing Live Resource Blocks MBPP Retry

Given the Exp 2837 MBPP runner checks all live-resource preconditions before
inference,
When the Qwen3.6 GGUF cache or another required resource is unavailable,
Then it writes `results/experiment_2837_mbpp_ensemble_eval.json` with a
specific `blocked_<resource>` verdict, populated precondition evidence,
state-file SHA metadata gathered so far, and null unmeasured MBPP metrics.

### REQ-VERIFY-2838: MBPP Dual-Condition v3 Gated By Exp 2836 SOTA Runtime

The repository shall provide an Exp 2838 MBPP dual-condition runner that writes
`results/experiment_2838_mbpp_dual_condition_v3.json` for exactly 100 MBPP
sanitized test tasks and the five replication seeds `[42, 137, 271, 314,
1729]`.

Before any candidate generation, candidate execution, or verifier scoring, the
runner MUST load `results/experiment_2836_sota_runtime_preflight.json`, assert
`sota_runtime_ready=true`, use the artifact's `selected_python`, record a
usable model path for at least one mandated SOTA GGUF
(`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`), confirm CUDA is available from the selected
Python, confirm the MBPP dataset and tests are available, confirm sandboxed
unit-test execution is available, and confirm FR-11/NEXUS/session state files
are present. If any precondition fails, it MUST write a terminal
`blocked_<resource>` artifact with populated `preconditions_checked` and no
inferred AUROC, pass@1, ranking-lift, candidate, or per-verifier scores.

For successful runs, the runner MUST generate or load deterministic MBPP
candidate sets for the same 100 tasks under all five seeds, execute the
candidate code against MBPP tests in the safe harness to obtain pass/fail
labels, and score the same candidates under Condition A (production FR-11 state
visible) and Condition B (architecture-only after FR-11/NEXUS/session state is
non-destructively moved aside and Python is restarted). At least one mandated
SOTA local GGUF recorded by Exp 2836 MUST be part of the generator or scorer
path, and legacy tiny models MUST NOT be reported as headline models.

The artifact MUST include `honest_verdict`, `n_tasks`,
`condition_a_production_auroc_mean`,
`condition_b_architecture_only_auroc_mean`, `learning_contribution`,
`per_verifier_condition_b_auroc`, `model_specs`, `preconditions_checked`,
`duration_s`, per-verifier Condition A AUROC, pass@1 and ranking-lift
summaries, per-seed results, FR-11 state-file hashes, restored-state proof, and
an honest methodology note.

### SCENARIO-VERIFY-2838-BLOCKED: Missing MBPP Resources Block Without Metrics

Given Exp 2836 reports a ready SOTA runtime but the MBPP dataset package,
dataset rows, MBPP tests, CUDA runtime, SOTA GGUF path, sandbox execution, or
FR-11 state files are unavailable,
When the Exp 2838 runner executes,
Then it writes `results/experiment_2838_mbpp_dual_condition_v3.json` with
`honest_verdict` prefixed `blocked_`, `n_tasks=100`, populated
`preconditions_checked`, null AUROC and ranking metrics, empty per-verifier
AUROC maps, and no inferred candidate labels.

### SCENARIO-VERIFY-2838-LIVE: Successful MBPP Run Reports Dual Conditions

Given Exp 2836 reports a ready SOTA runtime, a mandated SOTA GGUF is available,
the MBPP sanitized test split and sandboxed unit-test execution are available,
and FR-11 state files exist,
When the Exp 2838 runner evaluates all five seeds,
Then the artifact reports complete production and architecture-only AUROC
summaries from measured pass/fail labels, per-verifier AUROC values for both
conditions, pass@1/ranking-lift summaries, `learning_contribution = A - B`,
the mandated SOTA model path used by the scorer or generator path, and
`state_files_restored_sha_match=true`.

### REQ-VERIFY-2830: HumanEval-Full Dual-Memory Ensemble Evaluation

The repository shall provide an Exp 2830 HumanEval-full dual-memory runner
that writes `results/experiment_2830_humaneval_full_ensemble_eval.json` for
the full 164-problem `openai_humaneval` test split and five adversarial
replication seeds `[42, 137, 271, 314, 1729]`.

Before any live HumanEval inference, candidate execution, repair, or verifier
scoring, the runner MUST verify CUDA availability, HuggingFace
`openai_humaneval` dataset access, a discoverable Qwen3.6-35B-A3B GGUF cache,
and present FR-11 state files. If any precondition is missing, the runner MUST
write a terminal `blocked_<resource>` artifact rather than fabricating pass@1,
AUROC, duration, candidate, execution, or per-verifier scores.

For successful live runs, the runner MUST use clean code-execution
ground-truth for the same problem/candidate pairs under both memory
conditions: Condition A with full FR-11 production state and Condition B after
non-destructively moving FR-11 state aside, restarting Python, and scoring
architecture-only verifier behavior. The runner MUST restore every FR-11 state
file and prove SHA256 matches the original state manifest.

The artifact MUST include `honest_verdict`, `corpus="HumanEval-full"`,
`n_problems=164`, `n_seeds`, `pass_at_1_vanilla`,
`pass_at_1_after_carnot_correct_production`,
`pass_at_1_after_carnot_correct_architecture_only`, Condition A and B AUROC
means and standard deviations, `learning_contribution`, per-verifier AUROC
lists for both conditions, random seeds, reproducibility checksum, model specs,
real `duration_s`, precondition evidence, FR-11 state-file hashes and byte
sizes, `state_files_restored_sha_match`, peer-baseline comparison fields, and
an honest methodology note.

### SCENARIO-VERIFY-2830: Missing Preconditions Block HumanEval Metrics

Given the Exp 2830 runner starts in an environment missing CUDA,
`openai_humaneval` dataset access, the Qwen3.6 GGUF cache, or FR-11 state
files,
When the runner checks preconditions before live inference or code execution,
Then it writes `results/experiment_2830_humaneval_full_ensemble_eval.json`
with `honest_verdict` prefixed `blocked_`, null AUROC and pass@1 metrics,
empty per-verifier AUROC maps, populated `preconditions_checked`, preserved
FR-11 state-file hashes, and no inferred benchmark numbers.

### SCENARIO-VERIFY-2830-LIVE: Successful HumanEval Run Reports Dual Conditions

Given all Exp 2830 live-resource preconditions are available and a real
Qwen3.6 HumanEval generation, code-execution, repair, and ensemble-scoring
backend is configured,
When the runner evaluates all 164 HumanEval problems for all five seeds under
production and architecture-only memory conditions,
Then the artifact reports complete dual-condition pass@1 and AUROC summaries,
per-seed results, per-verifier AUROC lists for both conditions, restored FR-11
state hashes, peer-baseline comparisons, and an `honest_verdict` prefixed
`complete:`.

### REQ-VERIFY-HUMANEVAL-2838: HumanEval Full Ensemble v7b Retry Artifact

The repository shall provide an Exp 2838 HumanEval-full ensemble-v7b runner
that writes `results/experiment_2838_humaneval_full_ensemble_eval.json` for
the full 164-problem `openai_humaneval` test split and five replication seeds
`[42, 137, 271, 314, 1729]`.

Before any live HumanEval candidate generation, execution, FR-11 reset, or
verifier scoring, the runner MUST record the explicit
`.venv/bin/python3 -c "import torch; assert torch.cuda.is_available()"` CUDA
precondition, HuggingFace `openai_humaneval` dataset access, a mandated SOTA
GGUF model path, sandboxed HumanEval execution support, and present FR-11 state
files. If any precondition or live backend is unavailable, the runner MUST
write a terminal `blocked_<resource>` artifact with the complete requested
schema, populated precondition evidence, preserved FR-11 state-file metadata,
and null unmeasured AUROC, pass@1, candidate, ranking, and per-verifier values.

For successful runs, the runner MUST use clean HumanEval `check()` execution
ground truth for the same problem/candidate pairs under Condition A
(production FR-11 state visible) and Condition B (architecture-only after a
non-destructive FR-11 state reset and Python restart). It MUST restore every
FR-11 state file and verify SHA256 equality before reporting completion.

The artifact MUST include `honest_verdict`, `corpus="HumanEval-full"`,
`n_problems=164`, `n_seeds`, `condition_a_production_auroc_mean`,
`condition_a_production_auroc_std`,
`condition_b_architecture_only_auroc_mean`,
`condition_b_architecture_only_auroc_std`, `learning_contribution`,
per-verifier AUROC lists for both conditions, `pass_at_1_vanilla`,
`pass_at_1_after_carnot_correct_production`,
`pass_at_1_after_carnot_correct_architecture_only`, peer HumanEval verifier
baseline comparison fields, `random_seeds_used`, `reproducibility_checksum`,
`model_specs`, real `duration_s`, `preconditions_checked`, `fr11_state_files`,
`state_files_restored_sha_match`, and an honest `methodology_note`.

### SCENARIO-VERIFY-HUMANEVAL-2838-BLOCKED: Missing HumanEval Resource Blocks Metrics

Given the Exp 2838 HumanEval retry checks all live-resource preconditions
before inference, execution, FR-11 reset, or scoring,
When CUDA, HuggingFace `openai_humaneval`, a mandated SOTA GGUF path,
sandboxed execution, FR-11 state files, or the live HumanEval backend is
unavailable,
Then it writes `results/experiment_2838_humaneval_full_ensemble_eval.json`
with `honest_verdict` prefixed `blocked_`, `corpus="HumanEval-full"`,
`n_problems=164`, populated precondition evidence, null unmeasured metrics,
empty per-verifier AUROC maps, and no inferred candidate labels.

### SCENARIO-VERIFY-HUMANEVAL-2838-LIVE: Successful HumanEval Retry Reports Dual Conditions

Given all Exp 2838 HumanEval live resources are available and a real
Qwen3.6-compatible HumanEval generation, execution, and ensemble-v7b scoring
backend is configured,
When the runner evaluates all 164 HumanEval problems for all five seeds under
production and architecture-only memory conditions,
Then the artifact reports complete dual-condition pass@1 and AUROC summaries,
per-seed results, per-verifier AUROC lists for both conditions, restored FR-11
state hashes, peer-baseline comparisons, and an `honest_verdict` prefixed
`complete:`.

### REQ-VERIFY-2839: HumanEval Full Dual-Condition v3 Gated By Exp 2836 SOTA Runtime

The repository shall provide an Exp 2839 HumanEval dual-condition runner that
writes `results/experiment_2839_humaneval_dual_condition_v3.json` for all 164
`openai_humaneval` test tasks and the five replication seeds `[42, 137, 271,
314, 1729]`.

Before any candidate generation, candidate execution, or verifier scoring, the
runner MUST load `results/experiment_2836_sota_runtime_preflight.json`, assert
`sota_runtime_ready=true`, use the artifact's `selected_python`, record a usable
model path for at least one mandated SOTA GGUF
(`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`), confirm CUDA is available from the selected
Python, confirm the full HumanEval dataset and official tests are available,
confirm sandboxed unit-test execution is available, and confirm
FR-11/NEXUS/session state files are present. If any precondition fails, it MUST
write a terminal `blocked_<resource>` artifact with populated
`preconditions_checked` and no inferred AUROC, pass@1, ranking-lift, candidate,
or per-verifier scores.

For successful runs, the runner MUST generate or load deterministic HumanEval
candidate sets for the same 164 tasks under all five seeds, execute the
candidate code against the official HumanEval `check()` harness in the safe
harness to obtain pass/fail labels, and score the same candidates under
Condition A (production FR-11 state visible) and Condition B (architecture-only
after FR-11/NEXUS/session state is non-destructively moved aside and Python is
restarted). At least one mandated SOTA local GGUF recorded by Exp 2836 MUST be
part of the generator or scorer path, and legacy tiny models MUST NOT be
reported as headline models.

The artifact MUST include `honest_verdict`, `n_tasks`,
`condition_a_production_auroc_mean`,
`condition_b_architecture_only_auroc_mean`, `learning_contribution`,
`per_verifier_condition_b_auroc`, `model_specs`, `preconditions_checked`,
`duration_s`, per-verifier Condition A AUROC, pass@1 and ranking-lift
summaries, per-seed results, FR-11 state-file hashes, restored-state proof, and
an honest methodology note.

### SCENARIO-VERIFY-2839-BLOCKED: Missing HumanEval Resources Block Without Metrics

Given Exp 2836 reports a ready SOTA runtime but the `openai_humaneval` dataset
package, full 164 rows, official tests, CUDA runtime, SOTA GGUF path, sandbox
execution, or FR-11 state files are unavailable,
When the Exp 2839 runner executes,
Then it writes `results/experiment_2839_humaneval_dual_condition_v3.json` with
`honest_verdict` prefixed `blocked_`, `n_tasks=164`, populated
`preconditions_checked`, null AUROC and ranking metrics, empty per-verifier
AUROC maps, and no inferred candidate labels.

### SCENARIO-VERIFY-2839-LIVE: Successful HumanEval Run Reports Dual Conditions

Given Exp 2836 reports a ready SOTA runtime, a mandated SOTA GGUF is available,
the full HumanEval test split and sandboxed unit-test execution are available,
and FR-11 state files exist,
When the Exp 2839 runner evaluates all five seeds,
Then the artifact reports complete production and architecture-only AUROC
summaries from measured pass/fail labels, per-verifier AUROC values for both
conditions, pass@1/ranking-lift summaries, `learning_contribution = A - B`,
the mandated SOTA model path used by the scorer or generator path, and
`state_files_restored_sha_match=true`.

### REQ-VERIFY-2831: TruthfulQA-Generation Dual-Memory Ensemble Evaluation

The repository shall provide an Exp 2831 TruthfulQA-generation dual-memory
runner that writes `results/experiment_2831_truthfulqa_ensemble_eval.json` for
exactly 200 questions sampled from the TruthfulQA generation split with
test-sample seed 42 and five adversarial replication seeds
`[42, 137, 271, 314, 1729]`.

Before any live TruthfulQA inference, BLEURT labeling, or verifier scoring, the
runner MUST verify CUDA availability, HuggingFace TruthfulQA generation-split
access, a discoverable Qwen3.6-35B-A3B GGUF cache, present FR-11 state files,
and a cacheable BLEURT-base-128 scorer. If any precondition is missing, the
runner MUST write a terminal `blocked_<resource>` artifact rather than
fabricating AUROC, BLEURT thresholds, duration, candidates, labels, or
per-verifier scores.

For successful live runs, the runner MUST generate or reuse the same
question/candidate pairs under both memory conditions: Condition A with full
FR-11 production state and Condition B after non-destructively moving FR-11
state aside, restarting Python, and scoring architecture-only verifier
behavior. The ground-truth labels MUST use BLEURT-base-128 semantic similarity
against TruthfulQA `best_answer`, with the threshold tuned on a 50-question
held-out calibration set that does not overlap the 200-question test sample.
The runner MUST NOT use a closed-weight LLM judge for labels. The runner MUST
restore every FR-11 state file and prove SHA256 matches the original state
manifest.

The artifact MUST include `honest_verdict`,
`corpus="TruthfulQA-generation"`, `n_questions=200`, `n_seeds`, Condition A
and B AUROC means and standard deviations, `learning_contribution = A - B`,
per-verifier AUROC lists for both conditions, `scoring_method`, a measured
`bleurt_threshold`, random seeds, reproducibility checksum, model specs, real
`duration_s`, precondition evidence, FR-11 state-file hashes and byte sizes,
`state_files_restored_sha_match`, GPT-3 MC1 and published BLEURT-verifier
comparison fields when available, field-principle annotations, and an honest
methodology note.

### SCENARIO-VERIFY-2831: Missing Preconditions Block TruthfulQA Metrics

Given the Exp 2831 runner starts in an environment missing CUDA, TruthfulQA
dataset access, the Qwen3.6 GGUF cache, FR-11 state files, or BLEURT-base-128,
When the runner checks preconditions before live inference or BLEURT scoring,
Then it writes `results/experiment_2831_truthfulqa_ensemble_eval.json` with
`honest_verdict` prefixed `blocked_`, null AUROC and BLEURT-threshold metrics,
empty per-verifier AUROC maps, populated `preconditions_checked`, preserved
FR-11 state-file hashes, and no inferred benchmark numbers.

### SCENARIO-VERIFY-2831-LIVE: Successful TruthfulQA Run Reports Dual Conditions

Given all Exp 2831 live-resource preconditions are available and a real
Qwen3.6 TruthfulQA generation, BLEURT labeling, calibration, and
ensemble-scoring backend is configured,
When the runner evaluates all 200 sampled TruthfulQA generation questions for
all five seeds under production and architecture-only memory conditions,
Then the artifact reports complete dual-condition AUROC summaries, per-seed
results, per-verifier AUROC lists for both conditions, a non-null calibrated
BLEURT threshold, restored FR-11 state hashes, baseline comparisons, and an
`honest_verdict` prefixed `complete:`.

### REQ-VERIFY-2839-TQA: TruthfulQA Dual-Condition v7b Third Attempt

The repository shall provide an Exp 2839 TruthfulQA-generation verifier
ensemble v7b runner that writes
`results/experiment_2839_truthfulqa_ensemble_eval.json` for exactly 200
questions sampled from the TruthfulQA generation split with sample seed 42 and
replication seeds `[42, 137, 271, 314, 1729]`.

Before any live Qwen3.6 generation, BLEURT labeling, or verifier scoring, the
runner MUST execute the explicit `.venv/bin/python3` CUDA precondition, confirm
HuggingFace TruthfulQA generation access, confirm a real Qwen3.6-35B-A3B GGUF
cache entry, confirm FR-11 state files, and confirm BLEURT-base-128 is locally
usable or cacheable. If any precondition is missing, the runner MUST write a
terminal `blocked_<resource>` artifact with null AUROC, null BLEURT threshold,
empty per-verifier maps, populated precondition evidence, and no inferred
candidate, label, duration, or score values.

For successful live runs, the runner MUST use BLEURT-base-128 semantic
similarity against TruthfulQA `best_answer`, tune the threshold on a disjoint
50-question held-out calibration set, evaluate Condition A with full FR-11
production state and Condition B after a non-destructive FR-11 state reset over
the same question/candidate pairs, restore every state file, and verify SHA256
matches. It MUST NOT use a closed-weight LLM judge.

The artifact MUST include `corpus="TruthfulQA-generation"`, `n_questions=200`,
`n_seeds=5`, production and architecture-only AUROC means and standard
deviations, `learning_contribution = A - B`, per-verifier AUROC maps for both
conditions, `scoring_method="BLEURT-base-128, threshold tuned on 50-Q held-out"`,
`bleurt_threshold`, random seeds, reproducibility checksum, model specs, real
`duration_s`, preconditions checked, FR-11 state-file manifest,
`state_files_restored_sha_match`, field-principle annotations, GPT-3 MC1 /
published BLEURT-verifier comparison fields, and an honest methodology note.

### SCENARIO-VERIFY-2839-TQA-BLOCKED: Missing Exp 2839 Resources Block TruthfulQA Metrics

Given the Exp 2839 TruthfulQA runner starts without a real Qwen3.6 GGUF cache,
BLEURT-base-128 scorer, CUDA from `.venv/bin/python3`, TruthfulQA generation
access, or FR-11 state files,
When the runner checks preconditions before live inference or BLEURT scoring,
Then it writes `results/experiment_2839_truthfulqa_ensemble_eval.json` with
`honest_verdict` prefixed `blocked_`, null AUROC and BLEURT-threshold metrics,
empty per-verifier AUROC maps, populated `preconditions_checked`, preserved
FR-11 state-file hashes, and no inferred benchmark numbers.

### SCENARIO-VERIFY-2839-TQA-LIVE: Successful Exp 2839 TruthfulQA Run Reports Dual Conditions

Given all Exp 2839 live-resource preconditions are available and a real
Qwen3.6 TruthfulQA generation, BLEURT labeling, calibration, and
ensemble-scoring backend is configured,
When the runner evaluates all 200 sampled TruthfulQA generation questions for
all five seeds under production and architecture-only memory conditions,
Then the artifact reports complete dual-condition AUROC summaries, per-seed
results, per-verifier AUROC values for both conditions, a non-null calibrated
BLEURT threshold, restored FR-11 state hashes, baseline comparisons, and an
`honest_verdict` prefixed `complete:`.

### REQ-VERIFY-2840: TruthfulQA Dual-Condition v4 Local-Scorer Retry

The repository shall provide an Exp 2840 TruthfulQA dual-condition v4 runner
that writes `results/experiment_2840_truthfulqa_dual_condition_v4.json` for
exactly 200 questions from the TruthfulQA generation split and the replication
seeds `[42, 137, 271, 314, 1729]`.

Before any candidate generation, local semantic scoring, or verifier scoring,
the runner MUST load `results/experiment_2836_sota_runtime_preflight.json`,
assert `sota_runtime_ready=true`, use the artifact's `selected_python`, record
a usable model path for at least one mandated SOTA GGUF
(`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`), confirm CUDA is available from the selected
Python, confirm the TruthfulQA generation split is available, confirm a
reproducible local scorer is available (BLEURT-base-128 or an explicitly
documented local semantic scorer), confirm FR-11/NEXUS/session state files are
present, and confirm Exp 2823 is retired/fabricated rather than a usable data
source. If any precondition fails, it MUST write a terminal
`blocked_<resource>` artifact with populated `preconditions_checked` and no
inferred AUROC, calibration, candidate, label, duration, or per-verifier scores.

For successful runs, the runner MUST generate or load deterministic
TruthfulQA candidate answers under the fixed seeds, score each candidate
against `best_answer` or a recorded reference using the local scorer, record
the scorer name, version, and reproducibility metadata, and score the same
question/candidate/label rows under Condition A (production FR-11 state) and
Condition B (architecture-only after FR-11/NEXUS/session state is
non-destructively moved aside and Python is restarted). The runner MUST NOT
use a closed-weight LLM judge and MUST NOT read metrics from the retired
Exp 2823 artifact as evidence.

The artifact MUST include `honest_verdict`, `n_questions=200`, `local_scorer`,
`condition_a_production_auroc_mean`,
`condition_b_architecture_only_auroc_mean`, `learning_contribution`,
`retired_exp2823_not_used=true`, `model_specs`, `preconditions_checked`, real
`duration_s`, per-verifier AUROC for both conditions, calibration summary,
per-seed results, FR-11 state-file hashes, restored-state proof, field
principle annotations, and an honest methodology note.

### SCENARIO-VERIFY-2840-BLOCKED: Missing TruthfulQA v4 Resources Block Honestly

Given Exp 2836 may report a ready SOTA runtime but the TruthfulQA generation
split, CUDA runtime, mandated SOTA GGUF path, local scorer, FR-11 state files,
or Exp 2823 retirement evidence are unavailable,
When the Exp 2840 runner executes,
Then it writes `results/experiment_2840_truthfulqa_dual_condition_v4.json`
with `honest_verdict` prefixed `blocked_`, `n_questions=200`, populated
`preconditions_checked`, `retired_exp2823_not_used=true`, null AUROC and
calibration metrics, empty per-verifier AUROC maps, and no inferred candidate
labels.

### SCENARIO-VERIFY-2840-LIVE: Successful TruthfulQA v4 Run Reports Dual Conditions

Given Exp 2836 reports a ready SOTA runtime, a mandated SOTA GGUF is available,
the TruthfulQA generation split and a reproducible local scorer are available,
FR-11 state files exist, and the retired Exp 2823 artifact is confirmed unused,
When the Exp 2840 runner evaluates all five seeds over 200 questions,
Then the artifact reports complete production and architecture-only AUROC
summaries from locally scored labels, per-verifier AUROC values for both
conditions, calibration and local-scorer metadata,
`learning_contribution = A - B`, the mandated SOTA model path used by the
scorer or generator path, `retired_exp2823_not_used=true`, and
`state_files_restored_sha_match=true`.

### REQ-VERIFY-2841: HaluEval And FEVER Readiness Pilot

The repository shall provide an Exp 2841 factuality-readiness pilot that writes
`results/experiment_2841_halueval_fever_pilot.json` for at most 50 total
examples, sampling 25 HaluEval examples and 25 FEVER examples when both corpora
are locally available or fetchable. The pilot MUST be marked `pilot_only=true`
and MUST NOT be presented as a headline benchmark result.

Before candidate generation or verifier scoring, the runner MUST load
`results/experiment_2836_sota_runtime_preflight.json`, assert
`sota_runtime_ready=true`, use the artifact's `selected_python`, and record a
usable model path for at least one mandated SOTA GGUF
(`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`). It MUST confirm whether HaluEval and/or
FEVER rows are locally available or fetchable and write a terminal
`blocked_<resource>` artifact when no labeled corpus can be loaded or no
mandated model path is available.

For each loaded corpus, the pilot MUST generate or load candidate answers using
the mandated SOTA model provenance, score candidates with the existing verifier
ensemble against dataset labels, compute AUROC only as a pilot/readiness metric
with confidence intervals, and record caveats about sample size and label
semantics. The artifact MUST include `honest_verdict`, `pilot_only`,
`datasets_loaded`, `n_examples`, `pilot_auroc_by_dataset`, `recommendation`,
`model_specs`, `preconditions_checked`, real `duration_s`, and field principle
annotations.

### SCENARIO-VERIFY-2841-BLOCKED: Missing Pilot Resources Block Honestly

Given Exp 2836 is missing or not ready, no mandated SOTA GGUF path is available,
or neither HaluEval nor FEVER can provide labeled pilot rows,
When the Exp 2841 runner executes,
Then it writes `results/experiment_2841_halueval_fever_pilot.json` with
`honest_verdict` prefixed `blocked_`, `pilot_only=true`, populated
`preconditions_checked`, empty `datasets_loaded`, `n_examples=0`, and no
inferred AUROC values.

### SCENARIO-VERIFY-2841-PILOT: Loaded Corpora Report Readiness Metrics Only

Given Exp 2836 reports a ready SOTA runtime with at least one mandated GGUF and
HaluEval and/or FEVER labeled rows can be loaded,
When the Exp 2841 runner evaluates the pilot sample,
Then it records loaded dataset names, candidate-generation provenance,
verifier-energy summaries, AUROC confidence intervals for each measured corpus,
`pilot_only=true`, and a recommendation that scales a corpus to a future
N>=500 milestone only when the sample has both label classes, stable finite
energies, and a finite confidence interval.

### REQ-VERIFY-2832: Cross-Corpus Verifier Dual-Condition Matrix

The repository shall provide an Exp 2832 analyzer that reads only the real
Exp 2828 FoVer, Exp 2829 MBPP, Exp 2830 HumanEval-full, and Exp 2831
TruthfulQA-generation JSON artifacts and writes
`results/experiment_2832_cross_corpus_verifier_matrix_v2.json`.

The analyzer MUST construct a full verifier-by-corpus-by-condition matrix from
the upstream `per_verifier_condition_a_auroc` and
`per_verifier_condition_b_auroc` fields. Upstream scalar AUROC values and
per-seed AUROC lists MUST both be normalized to means. Missing, blocked, null,
or empty upstream measurements MUST NOT be replaced with synthetic fallback
verifiers or default AUROC values; they are recorded as absent measured data.

The artifact MUST include `honest_verdict` prefixed `complete:` or `success:`,
`verifier_corpus_dual_matrix`, `architecture_transfer_verifiers`,
`memory_augmented_verifiers`, `corpus_specific_verifiers`,
`low_signal_verifiers`, `diversity_gap_on_non_fover`, and a real
`duration_s`. Architecture-transfer verifiers are those with
architecture-only AUROC at least 0.75 on every measured corpus. Memory-augmented
verifiers are those whose production AUROC reaches at least 0.75 while
architecture-only AUROC is below 0.65 or whose production-minus-architecture
delta is at least 0.10. Low-signal verifiers have all measured AUROCs below
0.65. Remaining high-signal verifiers are corpus-specific. The diversity-gap
flag is true when fewer than three architecture-transfer verifiers cover any
non-FoVer corpus.

### SCENARIO-VERIFY-2832: Matrix Uses Real Upstream Artifacts Without Imputation

Given the four Exp 2828-2831 artifacts contain a mix of scalar and per-seed
per-verifier AUROC measurements,
When the Exp 2832 analyzer runs,
Then the emitted matrix contains one entry per measured verifier with FoVer,
MBPP, HumanEval, and TruthfulQA condition cells, each populated from measured
values or null when that verifier was absent from a corpus, and every verifier
appears in exactly one of the four classification lists.

### SCENARIO-VERIFY-2832-BLOCKED-UPSTREAM: Blocked Inputs Stay Empty

Given the four Exp 2828-2831 artifacts are terminal blocked artifacts with
empty per-verifier AUROC maps,
When the Exp 2832 analyzer runs,
Then it writes a terminal `complete:` artifact with an empty
`verifier_corpus_dual_matrix`, all four classification lists empty,
`diversity_gap_on_non_fover=true`, and methodology text explaining that no
synthetic verifier rows were inferred.

### REQ-VERIFY-MATRIX-2840: Cross-Corpus Verifier Matrix v3 From Real Dual-Condition Artifacts

The repository shall provide an Exp 2840 cross-corpus verifier-matrix analyzer
that writes `results/experiment_2840_cross_corpus_verifier_matrix_v3.json`.
The analyzer MUST read the post-torch-fix/post-codex-flip dual-condition
artifacts for FoVer, MBPP, HumanEval, and TruthfulQA. Because the .269
roadmap and completion log used two nearby filename families, the analyzer
MUST consider the canonical exp2836-2839 roadmap paths and the completed
suffixed artifact paths, select the available artifact with the most measured
per-verifier AUROC rows for each corpus, and record the selected path.

The analyzer MUST construct a verifier-by-corpus-by-condition matrix from
real upstream `per_verifier_condition_a_auroc` and
`per_verifier_condition_b_auroc` values only. Scalar AUROC values and per-seed
AUROC lists MUST be normalized to means. Missing, blocked, null, or empty
upstream measurements MUST remain null and MUST NOT be replaced with fallback
verifiers, chance AUROC values, or prior-milestone placeholders.

The artifact MUST include `honest_verdict` prefixed `complete:` or `success:`,
`verifier_corpus_dual_matrix`, `architecture_transfer_verifiers`,
`memory_augmented_verifiers`, `corpus_specific_verifiers`,
`low_signal_verifiers`, `diversity_gap_on_non_fover`, and real `duration_s`.
Architecture-transfer verifiers have architecture-only AUROC at least 0.75 on
every corpus where that verifier has any measured cell and include at least
one measured non-FoVer corpus. Memory-augmented verifiers have production
AUROC at least 0.75 in any corpus while the corresponding architecture-only
cell is missing, below 0.65, or lower by at least 0.10. Low-signal verifiers
have every measured production and architecture-only AUROC below 0.65.
Remaining high-signal verifiers are corpus-specific. The diversity-gap flag is
true when fewer than three architecture-transfer verifiers cover any non-FoVer
corpus.

### SCENARIO-VERIFY-MATRIX-2840-REAL: Matrix Uses Measured Rows Without Imputation

Given the candidate .269 upstream artifacts contain measured FoVer
per-verifier rows plus blocked or empty rows for some other corpora,
When the Exp 2840 matrix analyzer runs,
Then the emitted matrix includes every measured verifier, records null cells
for corpora where that verifier was not measured, classifies each verifier in
exactly one of the four verifier classes, records selected upstream artifact
paths, and marks the non-FoVer diversity gap open rather than inferring
missing transfer evidence.

### SCENARIO-VERIFY-MATRIX-2840-BLOCKED: Empty Upstream Rows Stay Empty

Given all available upstream artifacts have empty per-verifier AUROC maps,
When the Exp 2840 matrix analyzer runs,
Then it writes a terminal `complete:` artifact with an empty
`verifier_corpus_dual_matrix`, all four classification lists empty,
`diversity_gap_on_non_fover=true`, and methodology text explaining that no
synthetic verifier rows were inferred.

### REQ-VERIFY-2968: Deterministic Partial-Output Monitor Harness

The repository shall provide an Exp 2968 deterministic partial-output monitor
harness that replays fixture traces derived from existing code-repair and
NL-to-Z3 artifacts without invoking a live LLM and without claiming full
streaming verification.

- REQ-VERIFY-2968-1: The harness SHALL define these monitor event names:
  `partial_code_block`, `import_line`, `function_sig`,
  `assertion_or_formula_line`, `solver_query`, and `final_answer`.
- REQ-VERIFY-2968-2: The harness SHALL run deterministic checks covering parser
  prefix validity, import allow-list membership, symbol consistency, schema
  field coverage, and optional Z3 parse/execution checks.
- REQ-VERIFY-2968-3: The harness SHALL evaluate at least five fixture traces
  derived from `results/experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json`,
  `results/experiment_2959_nl_to_z3_execution_repair_mini_v2.json`, or bounded
  synthetic records that preserve the same code/Z3 schemas.
- REQ-VERIFY-2968-4: The terminal artifact SHALL be written to
  `results/experiment_2968_interwhen_partial_monitor_harness_v1.json` and SHALL
  include `honest_verdict`, `partial_monitor_harness_ready`,
  `full_streaming_verification_claim=false`, `source_artifacts`,
  `monitor_events`, `deterministic_checks`, `fixture_trace_count`,
  `fixture_checks_passed`, `coverage_by_event`, `latency_estimate_ms`,
  `escalation_policy`, `files_changed`,
  `inference_substrate="deterministic_wiring"`, and `duration_s`.
- REQ-VERIFY-2968-5: The artifact SHALL report false-positive notes and an
  escalation policy that routes parser failures, disallowed imports, symbol
  inconsistencies, schema coverage gaps, or Z3 parse/execution failures to full
  verification.

### SCENARIO-VERIFY-2968: Fixture Replay Reports Coverage And Escalation

Given the Exp 2952 and Exp 2959 source artifacts exist,
When the Exp 2968 harness builds and monitors fixture traces,
Then it emits coverage for every required monitor event, records deterministic
check results without any live model call, reports a latency estimate and
false-positive notes, and sets `partial_monitor_harness_ready=true` only when
all fixture traces pass the deterministic checks while
`full_streaming_verification_claim` remains false.

### REQ-VERIFY-3275: Clean Local SOTA Verifier Rerun V14

The repository shall provide an Exp 3275 clean local SOTA verifier rerun v14
builder that writes
`results/experiment_3275_clean_local_sota_verifier_rerun_v14.json` from the
Exp 3268 clean SOTA receipt methodology supplement and checked-in exact-row
fixtures. The builder MUST first read
`results/experiment_3268_sota_receipt_methodology_supplement_v1.json`; if
`clean_sota_receipt_eligible` is not true, it SHALL write a complete gated-skip
artifact instead of attempting verifier work. It MUST check CUDA availability,
mandated GGUF cache availability, and exact-row fixture availability before
local inference. It MUST NOT use synthetic shortcut rows, CPU fallback headline
claims, downloads, remote APIs, repair generation, `scripts/research_conductor.py`,
or active-roadmap edits.

When the gate is open, the builder SHALL use at least one locally cached
mandated SOTA GGUF from `unsloth/gemma-4-26B-A4B-it-GGUF`,
`unsloth/Qwen3.6-35B-A3B-GGUF`, or `unsloth/gemma-4-31B-it-GGUF`. It SHALL
score model verifier decisions only against exact-row authority from existing
fixtures, normalize uncertain or unparsable model responses to abstentions,
and compute conservative `false_accept_rate`, `false_reject_rate`, and
`abstention_rate` over the scored denominator. Exact fixture hashes and
per-row provenance SHALL make every evaluated row auditable.

The terminal artifact MUST include `clean_verifier_rerun_ready`,
`clean_rerun_allowed`, `false_accept_rate`, `false_reject_rate`,
`abstention_rate`, `n_eval`, `exact_row_fixture_hash`, `model_specs`,
`models_used`, `preconditions_checked`, `gpu_mem_used_mib`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `honest_verdict`. `honest_verdict`
MUST start with `complete:`, `success:`, `passed:`, or `shipped:` for terminal
complete artifacts, including gated skips.

### SCENARIO-VERIFY-3275: V14 Fails Closed Or Scores Exact Rows With Local GGUF

Given Exp 3268 may be missing, ineligible, or eligible with a cached mandated
local GGUF and the exact-row fixtures may be present or missing,
When Exp 3275 builds the clean local SOTA verifier rerun v14 artifact,
Then it emits a terminal JSON artifact in all cases, performs zero model calls
when the clean SOTA receipt gate or preconditions fail, records precondition
evidence and a compatibility `clean_rerun_allowed` signal, and only reports
repair-gate metrics from local GGUF decisions scored by exact fixture authority.

If the Exp 3268 gate is eligible, CUDA and at least one mandated GGUF are
available, and exact-row fixtures are present, then the builder SHALL run the
bounded local verifier evaluation, record actual models used and GPU memory
evidence, compute false accepts, false rejects, abstentions, sample size, and
checksum fields, and report whether the clean rerun is ready as repair-gate
input without promoting CPU fallback or synthetic rows.

## Implementation Status (REQ-VERIFY-3275)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3275 | Implemented (`python/carnot/verify/clean_local_sota_verifier_rerun_v14.py`) | Implemented (`tests/python/test_experiment_3275_clean_local_sota_verifier_rerun_v14.py`) |

### REQ-VERIFY-3286: Clean Verifier Abstention Root-Cause Audit V1

The repository shall provide an Exp 3286 deterministic root-cause audit that
reads `results/experiment_3275_clean_local_sota_verifier_rerun_v14.json`, the
Exp 3268 receipt supplement, and the checked-in exact-row fixtures, then writes
`results/experiment_3286_clean_verifier_abstention_root_cause_v1.json`.

The audit MUST NOT run live model inference, modify
`scripts/research_conductor.py`, push, or spend GPU time. It SHALL identify the
Exp 3275 evaluated rows, answerability fields, exact-answer fields, thresholds,
and abstention reasons. It SHALL separate missing canonical answers, fixture
mismatch, row extraction failure, score thresholds, model-output parsing, and
safety-policy causes instead of merging them into one abstention bucket.

The terminal artifact MUST include
`abstention_root_cause_audit_ready`, `abstention_root_cause_identified`,
`prior_abstention_rate`, `audited_exact_row_count`, `answerable_row_count`,
`malformed_or_missing_answer_count`, `threshold_or_policy_findings`,
`parser_or_extraction_findings`, `calibrated_rerun_plan`,
`target_max_abstention_rate`, `target_false_accept_rate`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `honest_verdict`.
`honest_verdict` MUST start with `complete:`, `success:`, `passed:`, or
`shipped:`.

`abstention_root_cause_identified` shall be true only when the audit has a
visible exact-row denominator, every malformed or missing-answer row is counted,
and the dominant abstention cause is assigned to data quality, extraction,
parser/model-output contract, threshold, or policy. The calibrated rerun plan
for Exp 3287 SHALL require zero false accepts, a non-trivial non-abstain
coverage target over answerable exact rows, explicit parser-output-contract
evidence, and a short smoke gate that aborts before a larger GPU rerun when the
model still does not emit parseable verifier decisions.

### SCENARIO-VERIFY-3286: Abstain-All V14 Becomes A Calibrated Rerun Plan

Given Exp 3275 scored exact fixture rows but reported `abstention_rate=1.0`,
When the Exp 3286 audit runs,
Then it emits a terminal JSON artifact showing the exact denominator,
classifies answerable, unanswerable, malformed, and unknown rows, distinguishes
strict-threshold blocking from parser/model-output contract failure, records
that missing canonical answers or row extraction were not the dominant cause
when the fixtures are present, and defines Exp 3287 acceptance criteria with
`target_false_accept_rate=0.0` and `target_max_abstention_rate<1.0`.

## Implementation Status (REQ-VERIFY-3286)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3286 | Implemented (`python/carnot/verify/clean_verifier_abstention_root_cause_v1.py`) | Implemented (`tests/python/test_experiment_3286_clean_verifier_abstention_root_cause_v1.py`) |

### REQ-VERIFY-3287: Abstention-Calibrated Clean Verifier Rerun V15

The repository shall provide an Exp 3287 abstention-calibrated clean verifier
rerun that reads the Exp 3286 calibrated rerun plan, checks CUDA and mandated
GGUF cache preconditions, evaluates exact-checkable rows from the checked-in
context fixture bank, and writes
`results/experiment_3287_abstention_calibrated_clean_verifier_v15.json`.

The builder MUST call `cached_sota_pair(gpu_indices=(0, 1))` when resolving
local SOTA GGUFs, SHALL record every missing mandated model from
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, and MAY run with a single cached mandated
GGUF only when the cached pair is unavailable. It MUST check `nvidia-smi`,
selected-Python CUDA/llama.cpp readiness, Exp 3286 root-cause readiness, and
exact-row fixture availability before inference. It MUST NOT push, modify
`scripts/research_conductor.py`, use remote APIs, fabricate fallback rows, or
open the repair gate when `abstention_rate=1.0`.

When at least 20 exact-checkable rows are available, the builder SHALL evaluate
at least 20 such rows. The calibrated abstention policy SHALL preserve the Exp
3286 strict leading-token contract (`ACCEPT`, `REJECT`, `ABSTAIN`), normalize
uncertain or unparsable outputs to abstentions, and require zero false accepts
plus non-trivial coverage before setting any repair-gate-ready flag.

The terminal artifact MUST include
`abstention_calibrated_clean_verifier_v15_ready`, `clean_verifier_rerun_ready`,
`repair_gate_input_clean_enough`, `model_specs`, `models_used`,
`missing_model_specs`, `preconditions_checked`, `n_eval`,
`exact_checkable_row_count`, `false_accept_rate`, `false_reject_rate`,
`abstention_rate`, `coverage_rate`, `abstention_reason_counts`,
`random_seed`, `reproducibility_checksum`, `duration_s`, and
`honest_verdict`. `honest_verdict` MUST start with `complete:`, `success:`,
`passed:`, or `shipped:` whenever the rerun artifact is written.

### SCENARIO-VERIFY-3287: Calibrated V15 Opens Repair Only On Zero False Accepts

Given Exp 3286 identifies parser/model-output contract mismatch as the
actionable abstention root cause and at least one mandated GGUF is locally
cached,
When Exp 3287 runs the calibrated clean verifier over exact-checkable rows,
Then the artifact records model/cache/runtime provenance, exact-row coverage,
false accepts, false rejects, abstentions, and per-class reasons, and sets
`repair_gate_input_clean_enough=true` only when false accepts remain zero and
coverage is non-trivial.

If the model still abstains on every row, if no mandated GGUF is cached, if
CUDA/llama.cpp readiness fails, or if any false accept appears, then the
artifact remains terminal and machine-readable but `clean_verifier_rerun_ready`
and `repair_gate_input_clean_enough` SHALL be false.

## Implementation Status (REQ-VERIFY-3287)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3287 | Implemented (`python/carnot/verify/abstention_calibrated_clean_verifier_v15.py`) | Implemented (`tests/python/test_experiment_3287_abstention_calibrated_clean_verifier_v15.py`) |

### REQ-VERIFY-3289: Repair Gate Decision V9 After Garak And Abstention

The repository shall provide an Exp 3289 deterministic repair-gate decision
builder that reads
`results/experiment_3285_full_garak_dataflip_redteam_eval_v2.json`,
`results/experiment_3287_abstention_calibrated_clean_verifier_v15.json`,
`results/experiment_3288_kan_sidecar_failure_autopsy_boundary_v1.json`, and
the prior Exp 3276 repair-gate decision if present, then writes
`results/experiment_3289_repair_gate_decision_v9_after_garak_abstention.json`.
The builder MUST be an aggregation-only decision and MUST NOT run repair,
rerun Garak, rerun clean verification, retrain or promote KAN, push, or modify
`scripts/research_conductor.py`.

The gate SHALL open only when all mandatory conditions pass: Exp 3285 reports
`garak_redteam_eval_ready=true`, Exp 3287 reports
`clean_verifier_rerun_ready=true`, Exp 3287 reports
`repair_gate_input_clean_enough=true`, Exp 3287 preserves a zero false-accept
rate and non-abstain coverage, and Exp 3288 reports
`kan_boundary_decision_ready=true` with downstream KAN use bounded away from
repair-gate authority or headline detector promotion. Exp 3285
`garak_gate_passed` and `blocked_reasons` SHALL remain visible in
`gate_inputs`, but the repair decision is gated on Garak evidence readiness,
not on converting target-model red-team behavior into repair success.

The terminal artifact MUST include `repair_gate_decision_v9_ready`,
`repair_gate_open`, `garak_redteam_eval_ready`, `clean_verifier_rerun_ready`,
`repair_gate_input_clean_enough`, `kan_boundary_decision_ready`,
`gate_inputs`, `blocked_reasons`, `permitted_repair_scope`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `honest_verdict`.
`blocked_reasons` SHALL contain precise source-field failures whenever the
gate is closed. `permitted_repair_scope` SHALL bound Exp 3290 when the gate is
open by naming a small sample size, mandated local SOTA GGUF model specs,
permitted case families, exact-verification requirements, false-accept and
abstention constraints, KAN-use restrictions, and the explicit no-headline
claim boundary. `honest_verdict` MUST begin with one of `complete:`,
`success:`, `passed:`, or `shipped:` whenever the decision artifact is
written.

### SCENARIO-VERIFY-3289: Conservative Aggregation Opens Only With Ready Evidence

Given the Exp 3285 full Garak/DataFlip red-team artifact, Exp 3287
abstention-calibrated clean verifier artifact, and Exp 3288 KAN boundary
artifact are present,
When Exp 3289 builds the repair-gate decision,
Then it writes the required terminal JSON artifact, traces every decision to
the upstream gate fields in `gate_inputs`, keeps Garak attack-success and KAN
failure metrics visible without promoting either as repair authority, opens
the repair gate only when the required ready/clean/bounded conditions hold,
and defines the bounded Exp 3290 micro-panel scope when `repair_gate_open=true`.

If any upstream artifact is missing or malformed, if Garak evidence is not
ready, if the clean verifier is not ready, if the clean verifier is abstain-all
or has any false accept, or if KAN downstream use is unbounded, then Exp 3289
still writes a terminal artifact with `repair_gate_open=false`, an empty or
non-executable `permitted_repair_scope`, and precise `blocked_reasons`.

## Implementation Status (REQ-VERIFY-3289)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3289 | Implemented (`python/carnot/verify/repair_gate_decision_v9.py`) | Implemented (`tests/python/test_experiment_3289_repair_gate_decision_v9.py`) |

### REQ-VERIFY-3290: Gated SOTA Repair Micro-Panel V10

The repository shall provide an Exp 3290 gated SOTA repair micro-panel runner
that reads
`results/experiment_3289_repair_gate_decision_v9_after_garak_abstention.json`,
runs only when `repair_gate_open=true`, builds a bounded panel from
`permitted_repair_scope`, generates repair candidates with at least one locally
cached mandated SOTA GGUF when available, verifies each candidate with the
Exp 3287 calibrated clean-verifier decision contract plus exact fixture checks,
and writes
`results/experiment_3290_gated_sota_repair_micro_panel_v10.json`.

The runner MUST call `cached_sota_pair(gpu_indices=(0, 1))` while resolving
mandated local SOTA GGUFs and MUST record all three mandated model ids:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. If the pair is unavailable but one
mandated GGUF is cached, the panel MAY run as a single-model diagnostic
micro-panel, provided `models_used` names that model and missing mandated
models remain explicit. If the repair gate is closed or no mandated GGUF is
available, the runner SHALL write a terminal machine-readable skipped or
blocked artifact and SHALL NOT fabricate repair rows or use legacy fallback
models.

The panel SHALL select between `permitted_repair_scope.sample_size.min_cases`
and `permitted_repair_scope.max_panel_cases` cases from permitted exact context
fixture counterexamples, SHALL preserve the failing candidate and localized
failure mode for each case, and SHALL not generalize beyond the panel. A repair
candidate SHALL count as a verified success only when the calibrated clean
verifier emits `ACCEPT` and the exact fixture check passes. A calibrated
verifier `ACCEPT` on an exact-check failure SHALL count as a false accept.
`ABSTAIN`, unparsable verifier output, or missing candidate output SHALL be
recorded as an abstention rather than a success.

The terminal artifact MUST include `sota_repair_micro_panel_v10_ready`,
`repair_panel_ran`, `model_specs`, `models_used`, `missing_model_specs`,
`preconditions_checked`, `panel_case_count`, `repair_success_rate`,
`verified_success_count`, `false_accept_count`, `abstention_count`,
`localized_failure_feedback`, `headline_claim_allowed`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `honest_verdict`.
`honest_verdict` MUST begin with one of `complete:`, `success:`, `passed:`, or
`shipped:` whenever the panel artifact is written.

### SCENARIO-VERIFY-3290: Open Repair Gate Produces Bounded Exact-Verified Panel

Given Exp 3289 reports `repair_gate_open=true` and its
`permitted_repair_scope` allows a bounded exact fixture repair micro-panel,
When Exp 3290 resolves at least one mandated local SOTA GGUF and runs the panel,
Then it writes the required terminal JSON artifact, records GPU/model/cache
preconditions, keeps `headline_claim_allowed=false`, names the exact models
used and missing mandated models, reports repair successes only for candidates
accepted by the calibrated clean verifier and passing exact checks, counts
false accepts separately, records abstentions separately, and emits localized
failure feedback for every non-successful case.

If Exp 3289 closes the gate, the exact fixture bank cannot provide the minimum
bounded panel, or no mandated GGUF is cached, then Exp 3290 still writes the
required terminal artifact with `repair_panel_ran=false`,
`sota_repair_micro_panel_v10_ready=false`, zero panel denominators, explicit
precondition diagnostics, and no headline claim.

## Implementation Status (REQ-VERIFY-3290)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3290 | Implemented (`python/carnot/verify/gated_sota_repair_micro_panel_v10.py`) | Implemented (`tests/python/test_experiment_3290_gated_sota_repair_micro_panel_v10.py`) |

### REQ-VERIFY-3297: Prefix-Closed Garak Rogue-String Text Guard V1

The repository shall provide an Exp 3297 deterministic text guard that reads
`results/experiment_3295_garak_failure_mode_autopsy_v1.json` and the cached
`.304` response previews in
`results/experiment_3285_full_garak_dataflip_redteam_eval_v2.json`, builds a
small reusable policy from the Exp 3295 `target_phrase_inventory` and
`target_prefix_inventory`, evaluates only cached previews plus deterministic
synthetic benign controls, and writes
`results/experiment_3297_prefix_closed_garak_guard_v1.json`.

The implementation MUST disclose that it is a normalized text-prefix and exact
target-phrase guard, not a true BEAVER token-probability bound. It MUST NOT run
Garak, live model inference, repair generation, KAN training, CUDA probes, the
research conductor, or modify `scripts/research_conductor.py`. The policy
SHALL normalize text with casefolding, whitespace collapse, and punctuation
trimming before matching exact target phrases or monitored target prefixes.
The guard SHALL expose a stable policy contract that downstream Exp 3299 can
consume without re-reading the source artifacts.

The terminal artifact MUST include `prefix_guard_policy_ready`, `guard_kind`,
`target_phrase_count`, `target_prefix_count`, `cached_trace_count`,
`benign_control_count`, `cached_attack_detection_rate`,
`cached_benign_false_positive_rate`, `guard_policy`,
`implementation_files_touched`, `tests_run`, `inference_substrate`,
`random_seed`, `reproducibility_checksum`, `duration_s`, and
`honest_verdict`. `guard_kind` MUST make the text-prefix limitation explicit,
`inference_substrate` MUST state cached response-preview evaluation rather than
live inference, and `honest_verdict` MUST begin with one of `complete:`,
`success:`, `passed:`, or `shipped:`.

### SCENARIO-VERIFY-3297: Cached Garak Previews Produce A Consumable Text Policy

Given Exp 3295 contains PromptInject and jailbreak/encoding target phrases plus
prefixes and Exp 3285 contains cached `.304` response previews,
When the Exp 3297 guard builder runs,
Then it writes `results/experiment_3297_prefix_closed_garak_guard_v1.json`
with a stable `guard_policy`, counts the target phrases, target prefixes,
cached previews, and benign controls, computes cached attack detection and
benign false-positive rates from local artifacts only, records every touched
implementation/test file, exposes the test commands run, and reports
`prefix_guard_policy_ready=true` without claiming live Garak benchmark
performance or true BEAVER probability bounds.

## Implementation Status (REQ-VERIFY-3297)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3297 | Implemented (`python/carnot/verify/prefix_closed_garak_guard_v1.py`) | Implemented (`tests/python/test_experiment_3297_prefix_closed_garak_guard.py`) |

### REQ-VERIFY-3301: Stratified Exact Repair Panel Manifest V11

The repository shall provide an Exp 3301 deterministic manifest builder that
preserves the Exp 3289 open repair gate and the Exp 3287 exact verifier
contract before any live repair rerun, and writes
`results/experiment_3301_exact_repair_panel_manifest_v11.json`. The builder
MUST NOT invoke an LLM, start a local model server, run live repair generation,
push, or modify `scripts/research_conductor.py`.

The builder SHALL emit a stable panel cases manifest at
`data/research/exact_repair_panel_v11.jsonl` with at least 30 repair cases
stratified across multiple exact-checkable families, including symbolic
aliases, arithmetic exact rows, context shortcuts, simple code/output checks,
and bounded logical consistency. Every case SHALL include `context`,
`question`, `failing_candidate`, `expected_answer`, `exact_checker_type`,
`localized_repair_feedback`, and `case_hash`; every known failing candidate
MUST fail its exact checker while the expected answer passes. Any case that
requires an LLM judge to decide correctness MUST be excluded.

The terminal artifact MUST include `repair_panel_manifest_ready`,
`panel_case_count`, `case_family_counts`, `exact_checker_types`,
`llm_judge_required_count`, `panel_cases_path`, `case_hashes`,
`localized_feedback_coverage`, `known_failing_candidate_count`,
`validation_commands`, `inference_substrate`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `honest_verdict`.
`repair_panel_manifest_ready` MUST be true only when the stable manifest has at
least 30 cases, at least five represented families, exact checker coverage is
complete, all case hashes are unique, localized feedback coverage is 1.0, and
`llm_judge_required_count=0`. `honest_verdict` MUST begin with one of
`complete:`, `success:`, `passed:`, or `shipped:` whenever the manifest
artifact is written.

### SCENARIO-VERIFY-3301: Fixed Exact Panel Gates The Live Repair Rerun

Given Exp 3289 has reopened the repair gate and Exp 3290 showed only a
four-case diagnostic repair smoke,
When Exp 3301 builds the fixed exact panel manifest,
Then it writes the required terminal JSON artifact and stable JSONL case
manifest without live inference, reports at least 30 case hashes across
symbolic, arithmetic, context shortcut, code/output, and bounded logic
families, records exact checker types only, reports zero LLM-judge-required
cases, confirms each case starts from a known failing candidate, and exposes
the validation commands used to parse and test the manifest.

If any case lacks localized repair feedback, has a duplicate case hash, passes
the exact checker with its failing candidate, fails the exact checker with its
expected answer, or requires an LLM judge, then Exp 3301 SHALL fail validation
instead of marking `repair_panel_manifest_ready=true`.

## Implementation Status (REQ-VERIFY-3301)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3301 | Implemented (`python/carnot/verify/exact_repair_panel_manifest_v11.py`) | Implemented (`tests/python/test_experiment_3301_exact_repair_panel_manifest_v11.py`) |

### REQ-VERIFY-3302: Headline SOTA Repair Panel V11

The repository shall provide an Exp 3302 headline-scale SOTA repair panel
runner that reads the fixed Exp 3301 exact manifest without changing the case
list after model output is observed, checks the Exp 3300 Garak gate before any
model load, checks the Exp 3287 abstention-calibrated clean verifier contract,
checks visible CUDA/GPU and mandated GGUF cache preconditions, generates one
repair per manifest case with localized verifier feedback using an available
mandated SOTA GGUF, verifies every candidate with both the calibrated clean
verifier decision contract and the deterministic exact checker, and writes
`results/experiment_3302_headline_sota_repair_panel_v11.json`.

The runner MUST record every mandated model id:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Missing mandated GGUFs MUST be explicit in
`missing_model_specs`; legacy fallback models MUST NOT be used. The panel SHALL
run only when Exp 3300 reports `garak_gate_passed=true` and Exp 3301 reports
`repair_panel_manifest_ready=true` with at least 30 exact-checkable cases. A
candidate SHALL count as a verified success only when the calibrated clean
verifier emits `ACCEPT` and the exact checker passes. A calibrated verifier
`ACCEPT` on an exact-check failure SHALL count as a false accept. `ABSTAIN`,
unparsable verifier output, verifier rejection, or missing candidate output
SHALL be recorded as a non-success with abstentions counted separately.

The terminal artifact MUST include `headline_repair_panel_ready`,
`repair_panel_ran`, `headline_claim_allowed`, `model_specs`, `models_used`,
`missing_model_specs`, `preconditions_checked`, `panel_case_count`,
`verified_success_count`, `repair_success_rate`, `repair_success_ci95`,
`false_accept_count`, `false_accept_rate_ci95`, `abstention_count`,
`per_family_metrics`, `candidate_results`, `gpu_mem_used_mib`,
`tokens_generated`, `inference_substrate`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `honest_verdict`.
`headline_claim_allowed` MUST be true only when `panel_case_count>=30`,
`false_accept_count=0`, at least one mandated model was actually used, all
preconditions passed, source provenance is clean, and the panel case hashes
match the fixed Exp 3301 manifest. `honest_verdict` MUST begin with one of
`complete:`, `success:`, `passed:`, or `shipped:` whenever the panel artifact
is written.

### SCENARIO-VERIFY-3302: Fixed Manifest Produces Auditable Headline Repair Evidence

Given Exp 3300 reports `garak_gate_passed=true`, Exp 3301 provides a ready
fixed exact repair panel manifest with at least 30 cases, Exp 3287 preserves a
zero-false-accept calibrated clean verifier contract, CUDA/GPU checks pass, and
at least one mandated GGUF is cached,
When Exp 3302 runs the repair panel,
Then it writes the required terminal JSON artifact, records the unchanged
manifest case hashes, names used and missing mandated models, records GPU
memory and generated token evidence, reports one candidate result per manifest
case, accepts repairs only when the calibrated clean verifier accepts and the
exact checker passes, counts false accepts and abstentions separately,
stratifies outcomes in `per_family_metrics`, and computes 95% confidence
intervals for repair success and false-accept risk.

If the Garak gate is not passed, the fixed manifest is not ready, the clean
verifier contract is not ready, CUDA/GPU preconditions fail, or no mandated
GGUF is cached, then Exp 3302 SHALL still write the required terminal artifact
with `repair_panel_ran=false`, `headline_repair_panel_ready=false`, zero panel
denominators, explicit precondition diagnostics, and
`headline_claim_allowed=false`.

## Implementation Status (REQ-VERIFY-3302)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3302 | Implemented (`python/carnot/verify/headline_sota_repair_panel_v11.py`) | Implemented (`tests/python/test_experiment_3302_headline_sota_repair_panel_v11.py`) |

### REQ-VERIFY-3303: Repair Headline Evidence Audit V1

The repository shall provide an Exp 3303 independent audit workflow that reads
`results/experiment_3302_headline_sota_repair_panel_v11.json` and
`results/experiment_3301_exact_repair_panel_manifest_v11.json`, runs or invokes
the available adversarial artifact checks for the Exp 3302 result when
feasible, and writes
`results/experiment_3303_repair_headline_evidence_audit_v1.json`. The workflow
MUST be aggregation only. It MUST NOT rerun repair generation, rerun any model,
alter Exp 3302, push, or modify `scripts/research_conductor.py`.

The audit SHALL confirm that every claimed repair success in Exp 3302 has a
deterministic exact checker pass and that no success depends on an LLM judge.
It SHALL verify `panel_case_count>=30`, confidence interval fields are present,
`false_accept_count=0`, manifest hashes match the Exp 3301 fixed case list, and
model declarations identify the actual invoked model while keeping missing
mandated models explicit. The audit SHALL preserve any adversarial verification
flags and SHALL fail headline promotion when duration, substrate, provenance,
or adversarial flags make the source artifact unsafe to cite as a headline
repair claim.

The terminal artifact MUST include `repair_headline_evidence_audit_ready`,
`headline_claim_allowed_after_audit`, `audited_artifact`, `panel_case_count`,
`exact_successes_audited`, `false_accept_count`,
`llm_judge_dependency_count`, `adversarial_verify_flags`,
`substrate_consistency_passed`, `confidence_interval_present`,
`claim_boundaries`, `inference_substrate`, `random_seed`,
`reproducibility_checksum`, `duration_s`, and `honest_verdict`. It SHOULD also
include source checksums, model invocation summaries, exact-check provenance,
and field-provenance annotations. `repair_headline_evidence_audit_ready` MUST
be true only when Exp 3302 and Exp 3301 are readable, the fixed manifest
denominator is checked, the exact-check and LLM-judge dependency counts are
computed, adversarial verification results are preserved, and all required
audit fields validate. `headline_claim_allowed_after_audit` MUST be true only
when the source artifact already allows a headline claim, all claimed successes
have exact checker passes, no LLM judge dependency exists, `false_accept_count`
is zero, confidence intervals are present, `panel_case_count>=30`, source
provenance is clean, model/substrate declarations are coherent, and no
critical adversarial verification flag remains. `honest_verdict` MUST begin
with one of `complete:`, `success:`, `passed:`, or `shipped:`.

### SCENARIO-VERIFY-3303: Audit Bounds Exp 3302 Headline Repair Evidence

Given Exp 3302 reports a ready 30-case repair panel over the fixed Exp 3301
manifest, 27 verified successes, `false_accept_count=0`, confidence intervals,
one actually used mandated GGUF model, two missing mandated GGUF models, and an
adversarial verification `DURATION_TOO_SHORT` flag,
When Exp 3303 runs the audit workflow,
Then it writes the required terminal JSON artifact without rerunning repair or
changing Exp 3302, records the audited source path, preserves the adversarial
flag list, confirms the claimed successes are exact-checker-backed with zero
LLM-judge dependencies, records `substrate_consistency_passed=false`, records
`headline_claim_allowed_after_audit=false`, and lists claim boundaries that
allow bounded exact-check/no-false-accept evidence while forbidding headline
repair promotion until the duration/substrate flag is independently resolved.

If Exp 3302 has any claimed success without an exact checker pass, any LLM
judge dependency, missing confidence intervals, fewer than 30 panel cases, a
nonzero false accept count, mismatched manifest hashes, missing actual model
declarations, dirty provenance, or critical adversarial flags, then Exp 3303
SHALL still write a complete audit artifact but SHALL set
`headline_claim_allowed_after_audit=false` and include the relevant restriction
in `claim_boundaries`.

## Implementation Status (REQ-VERIFY-3303)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3303 | Implemented (`python/carnot/verify/repair_headline_evidence_audit_v1.py`) | Implemented (`tests/python/test_experiment_3303_repair_headline_evidence_audit_v1.py`) |

### REQ-VERIFY-3314: Distributional EBM Repair Uncertainty Audit V1

The repository shall provide an Exp 3314 aggregation-only Distributional-EBM-inspired
repair uncertainty audit that reads
`results/experiment_3302_headline_sota_repair_panel_v11.json`,
`results/experiment_3303_repair_headline_evidence_audit_v1.json`, and
`results/experiment_3313_repair_substrate_root_cause_autopsy_v1.json`, applies
a deterministic sidecar schema to the `.305` repair panel rows when those rows
are available, and writes
`results/experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.json`.
The workflow MUST NOT invoke a model, rerun repair generation, rerun exact
verification, train a learned EBM, push, or modify `scripts/research_conductor.py`.

The audit SHALL define `distributional_energy_schema` as row-level metadata that
separates deterministic constraint penalty, learned/proxy quality,
provenance risk, model identity, uncertainty, and abstention. Deterministic
constraint penalty SHALL be derived from recorded exact-check outcomes only;
learned/proxy quality SHALL be derived from recorded clean-verifier decisions
and token/answer-shape proxies only; provenance risk SHALL preserve source
provenance, runtime-contract, adversarial-flag, and mandated-model-coverage
features; model identity SHALL record used and missing mandated model families
without imputing absent runs; uncertainty SHALL be a deterministic proxy over
clean-verifier disagreement, provenance risk, missing model coverage, and row
quality ambiguity; abstention SHALL be advisory only and SHALL NOT replace
exact checker authority.

The terminal artifact MUST include `distributional_repair_audit_ready`,
`uncertainty_abstention_policy`, `distributional_energy_schema`,
`model_identity_confound_check`, `provenance_risk_features`,
`repair_case_count`, `no_new_model_execution`, and `honest_verdict`. It SHOULD
also include row-level sidecar scores, source artifact checksums, a summary of
exact-pass/clean-verifier-disagreement rows, the Exp 3316 policy handoff, run
date, duration, random seed, and reproducibility checksum.
`distributional_repair_audit_ready` MUST be true only when the source artifacts
are readable, at least 30 repair rows are scored, every row score preserves the
exact-check authority field, the abstention policy is defined, the model-identity
confound check reports both used and missing mandated models, provenance-risk
features include runtime-contract and adversarial-flag inputs, and
`no_new_model_execution=true`.

### SCENARIO-VERIFY-3314: Repair Rows Receive Triage Metadata Only

Given Exp 3302 reports a 30-case `.305` repair panel with 30 exact-check
passes, 27 clean-verifier accepted successes, 3 clean-verifier rejections of
exact-passing rows, one used mandated Gemma GGUF model, two missing mandated
models, and Exp 3303/3313 carry a critical duration/substrate provenance flag,
When Exp 3314 builds the distributional repair uncertainty audit,
Then it writes the required terminal JSON artifact without rerunning repair or
verification, records `repair_case_count=30`, emits one row-level sidecar score
per repair row, keeps exact checker outcomes as the final correctness authority,
marks clean-verifier/exact-check disagreements as uncertainty-abstain rows,
records provenance risk features from the source audit and substrate autopsy,
reports the model-family confound caused by one used model family and missing
mandated models, and defines an Exp 3316 policy that blocks headline promotion
when row uncertainty, provenance risk, critical adversarial flags, or
model-identity coverage risk exceed the stated thresholds.

If the source panel lacks rows, has fewer than 30 repair cases, omits exact
authority fields, omits provenance-risk inputs, omits model identity coverage,
or any source artifact cannot be read, then Exp 3314 SHALL still write an
honest audit artifact when possible but SHALL set
`distributional_repair_audit_ready=false`, keep `no_new_model_execution=true`,
and require Exp 3316 headline promotion to remain blocked.

## Implementation Status (REQ-VERIFY-3314)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3314 | Planned (`python/carnot/verify/distributional_ebm_repair_uncertainty_audit_v1.py`) | Planned (`tests/python/test_experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.py`) |

### REQ-VERIFY-3315: Verifier-Guided Backtracking Repair Policy V1

The repository shall provide an Exp 3315 aggregation-only verifier-guided
backtracking repair policy that reads
`results/experiment_3313_repair_substrate_root_cause_autopsy_v1.json` and
`results/experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.json`,
then writes
`results/experiment_3315_vgb_backtracking_repair_policy_v1.json`. The
workflow MUST NOT invoke model inference, generate new repair candidates,
rerun exact verification, rerun LLM judges, push, or modify
`scripts/research_conductor.py`.

The policy SHALL define deterministic rules for `accepted`, `rejected`,
`backtracked`, and `abstained` repair candidates. Exact verifiers SHALL be the
only final acceptance authority: a candidate cannot be accepted unless the
recorded exact checker passes, no false accept is recorded, and the exact
checker type is present. LLM judges and clean/process verifiers MAY guide
proposal routing but SHALL NOT override exact verifier outcomes. Exact-check
failures and false accepts SHALL reject the candidate and trigger backtracking
only while budget remains. Exact-passing candidates with clean/process-verifier
rejection, abstention, or insufficient confidence SHALL backtrack while budget
remains and abstain when the per-case budget is exhausted. Provenance,
model-identity, critical adversarial, or uncertainty blocks from Exp 3314 SHALL
force abstention rather than headline acceptance.

The proposal budget SHALL define at least `max_attempts_per_case`,
`max_backtracks_per_case`, `max_total_attempts`, and stop conditions for
accepted candidates, exhausted per-case budget, critical false-accept evidence,
closed upstream gates, and global budget exhaustion. The policy SHALL define
verifier confidence thresholds for exact acceptance, clean/process-verifier
routing, row uncertainty, provenance risk, and model-identity coverage risk.

The terminal artifact MUST include `vgb_repair_policy_ready`,
`backtracking_policy`, `proposal_budget`, `exact_acceptance_rules`,
`verifier_confidence_thresholds`, `no_new_model_execution`, and
`honest_verdict`. It SHOULD also include source artifact checksums,
candidate-attempt logging requirements, Exp 3316 handoff fields, run date,
duration, random seed, reproducibility checksum, and
`inference_substrate="deterministic_policy_artifact_no_model_calls"`.
`vgb_repair_policy_ready` MUST be true only when the source artifacts are
readable, Exp 3314 is ready, all required action rules are present, the budget
and stop conditions are finite and deterministic, exact acceptance rules
exclude LLM judges as final authority, candidate-attempt logging includes
attempt identity, verifier confidence, exact outcomes, policy action, and
abstention reason fields, and `no_new_model_execution=true`.

### SCENARIO-VERIFY-3315: VGB Policy Backtracks Without Weak Acceptance

Given Exp 3314 reports a ready distributional repair audit with exact-row
metadata, clean-verifier disagreements, uncertainty thresholds, provenance risk
thresholds, and model-identity thresholds,
When Exp 3315 builds the verifier-guided backtracking policy,
Then it writes the required terminal JSON artifact without model execution,
defines accept/reject/backtrack/abstain rules, sets a finite proposal budget,
requires exact verifiers as final acceptance authority, logs candidate
attempts with verifier confidence and exact outcomes, and directs Exp 3316 to
abstain rather than accept whenever exact checks are missing, exact checks
fail, false accepts appear, advisory risk gates are closed, or process
verifier confidence is below the acceptance threshold after the budget is
exhausted.

If Exp 3313 or Exp 3314 cannot be read, Exp 3314 is not ready, required
thresholds are missing, or the policy cannot prove that exact verifiers remain
final authority, then Exp 3315 SHALL still write an honest policy artifact when
possible but SHALL set `vgb_repair_policy_ready=false`, keep
`no_new_model_execution=true`, and require Exp 3316 headline promotion to
remain blocked.

## Implementation Status (REQ-VERIFY-3315)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3315 | Implemented (`python/carnot/verify/vgb_backtracking_repair_policy_v1.py`) | Implemented (`tests/python/test_experiment_3315_vgb_backtracking_repair_policy_v1.py`) |

### REQ-VERIFY-3316: SOTA Repair Rerun V12 Runtime-Clean

The repository shall provide an Exp 3316 gated SOTA repair rerun that writes
`results/experiment_3316_sota_repair_rerun_v12_runtime_clean.json` only after
checking the `.306` prerequisite artifacts:
`results/experiment_3312_dataflip_garak_quality_clean_rerun_v4.json`,
`results/experiment_3309_live_runtime_provenance_contract_v1.json`,
`results/experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.json`,
`results/experiment_3315_vgb_backtracking_repair_policy_v1.json`, and the
`.305` source panel
`results/experiment_3302_headline_sota_repair_panel_v11.json`. The rerun MUST
also preserve the fixed exact manifest and calibrated clean-verifier acceptance
authority used by REQ-VERIFY-3302. The workflow MUST NOT push, MUST NOT modify
`scripts/research_conductor.py`, and MUST NOT use legacy small models as repair
evidence.

The rerun SHALL check `nvidia-smi`, selected-Python CUDA visibility, mandated
GGUF cache availability for `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF`, model
load timing, generated-token counts, runtime-provenance fields, checker-version
fields, GPU-memory evidence, manifest case hashes, and exact checker types
before any headline promotion. If the live SOTA substrate is unavailable, or if
only smoke-test/legacy models are available, the workflow SHALL write an honest
blocked artifact rather than fabricating a panel. If the panel runs, it SHALL
evaluate at least the `.305` case count, record the same or a superset of the
`.305` manifest case hashes, and count a candidate as accepted only when the
deterministic exact checker passes under the VGB policy. Process verifiers,
Distributional-EBM sidecar scores, and clean-verifier confidence MAY route
backtracking or abstention but SHALL NOT override exact acceptance authority.

The terminal artifact MUST include `repair_rerun_v12_ready`,
`repair_panel_ran`, `headline_repair_panel_ready`,
`headline_claim_allowed`, `runtime_provenance_clean`,
`duration_contract_passed`, `substrate_consistency_passed`,
`panel_case_count`, `verified_success_count`, `false_accept_count`,
`abstention_count`, `repair_success_rate`, `confidence_interval`,
`model_specs_used`, and `honest_verdict`. It SHOULD also include source
artifact hashes, runtime provenance, checker versions, model cache/load
provenance, candidate attempts, exact outcome summaries, Distributional-EBM
sidecar rows, VGB policy routing, blocked reasons, run date, duration, random
seed, and reproducibility checksum. `headline_claim_allowed` MUST be true only
when the runtime contract passes, duration contract passes, substrate
consistency passes, no legacy model is substituted, false accepts are zero,
critical adversarial flags are absent, exact acceptance authority is preserved,
and the Distributional-EBM/VGB policies do not require abstention.

### SCENARIO-VERIFY-3316: Runtime-Clean Repair Rerun Either Promotes or Blocks Honestly

Given Exp 3312 reports DataFlip and quality cleanup ready, Exp 3309 provides a
ready runtime contract, Exp 3314 provides a ready Distributional-EBM repair
uncertainty audit, Exp 3315 provides a ready verifier-guided backtracking
policy, and the `.305` repair panel provides a 30-case exact manifest baseline,
When Exp 3316 runs with visible CUDA and cached mandated GGUF models,
Then it writes the required terminal JSON artifact, records runtime provenance
and checker versions, runs at least 30 fixed-manifest cases, logs candidate
attempts and VGB routing decisions, computes exact successes, false accepts,
abstentions, repair success rate, and a confidence interval, applies the
Distributional-EBM sidecar only as uncertainty/abstention metadata, and allows
headline repair claims only when the exact-authority, runtime, substrate,
model-provenance, false-accept, and advisory-policy gates all pass.

If `nvidia-smi`, CUDA visibility, mandated GGUF cache, model load, token
generation, runtime-provenance, duration, substrate, Distributional-EBM policy,
or VGB policy checks fail, then Exp 3316 SHALL still write the required JSON
artifact with `repair_panel_ran=false` when no live panel ran, explicit
`blocked_reasons`, zero panel denominators, empty `model_specs_used` when no
mandated model was used, `headline_claim_allowed=false`, and an
`honest_verdict` that states the rerun is blocked rather than headline-ready.

## Implementation Status (REQ-VERIFY-3316)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3316 | Implemented (`python/carnot/verify/sota_repair_rerun_v12_runtime_clean.py`) | Implemented (`tests/python/test_experiment_3316_sota_repair_rerun_v12_runtime_clean.py`) |

### REQ-VERIFY-3329: Cached-Candidate Verifier Ensemble Diversity Audit V2

- REQ-VERIFY-3329-1: The workflow SHALL evaluate the verifier ensemble diversity on a cached candidate matrix.
- REQ-VERIFY-3329-2: The workflow SHALL NOT require live LLM inference.
- REQ-VERIFY-3329-3: The workflow SHALL require a cached candidate matrix of at least 1000 cases to make covariance or eigenvalue claims. If `n < 1000`, the workflow MUST explicitly mark the results as diagnostic-only.
- REQ-VERIFY-3329-4: The workflow SHALL compute pairwise agreement, covariance/correlation matrices, `lambda_min_sigma`, condition number, `effective_k`, and per-domain verifier collapse modes.
- REQ-VERIFY-3329-5: The workflow SHALL write `results/experiment_3329_verifier_ensemble_diversity_audit_v2.json` with the required fields: `honest_verdict`, `inference_substrate`, `random_seed`, `reproducibility_checksum`, `duration_s`, `n_cases`, `verifier_names`, `covariance_methodology`, `lambda_min_sigma`, `effective_k`, `diversity_gate_passed`, `verifier_diversity_audit_v2_ready`, `collapsed_pairs`, `blocked_reasons`.

## Implementation Status (REQ-VERIFY-3329)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3329 | Implemented (`scripts/experiment_3329_verifier_ensemble_diversity_audit_v2.py`) | Implemented (`tests/python/test_experiment_3329_verifier_ensemble_diversity_audit_v2.py`) |

### REQ-VERIFY-3353: Deterministic Bounds on Constraint Satisfaction

- REQ-VERIFY-3353-1: Implement a Token Trie and Frontier data structure (`PrefixClosedBoundVerifier`) to maintain prefix-closed bounds over constraint evaluations.
- REQ-VERIFY-3353-2: Evaluate a synthetic problem space where prefixes deterministically violate constraints.
- REQ-VERIFY-3353-3: Compute sound probability bounds and compare against loose sampling bounds.
- REQ-VERIFY-3353-4: Validate that the bound is monotonic and tighter than sampling.
- REQ-VERIFY-3353-5: The workflow SHALL write `results/experiment_3353_deterministic_bounds.json` with the required schema fields.

## Implementation Status (REQ-VERIFY-3353)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3353 | Implemented (`python/carnot/cascade/tier2_verifier.py`) | Implemented (`tests/python/test_tier2_verifier.py`) |

### REQ-VERIFY-3375: VGB Repair Ladder on Llama-3

- REQ-VERIFY-3375-1: The repository shall provide an Exp 3375 multi-turn repair ladder test targeting the Llama-3 model family to ensure generalizability beyond previously validated substrates.
- REQ-VERIFY-3375-2: When Exp 3375 is run, it MUST produce a valid JSON artifact `results/experiment_3375_vgb_llama3.json` containing the standard schema fields.

### SCENARIO-VERIFY-3375: Test VGB Repair Ladder on Llama-3
When Exp 3375 builds the repair ladder artifact, it produces all required schema fields on the exit path.

## Implementation Status (REQ-VERIFY-3375)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3375 | Implemented (`scripts/experiment_3375_vgb_llama3.py`) | Implemented (`tests/python/test_experiment_3375_vgb_llama3.py`) |


### REQ-VERIFY-3393: Proximal-Gradient Constraint Layer
- REQ-VERIFY-3393-1: Implement a Proximal-Gradient step for constraint satisfaction in the verifier pipeline.
- REQ-VERIFY-3393-2: Set up a subset of constraints using continuous relaxation.
- REQ-VERIFY-3393-3: Implement proximal-descent projection over cached_sota_pair() unsloth/gemma-4-31B-it-GGUF logits.
- REQ-VERIFY-3393-4: Measure constraint satisfaction improvement versus soft penalty.
- REQ-VERIFY-3393-5: The workflow SHALL write `results/experiment_3393_proximal_gradient_constraint.json` with the required schema fields.

### SCENARIO-VERIFY-3393: Proximal-Gradient Constraint Layer
When Exp 3393 runs, it produces all required schema fields on the exit path.

## Implementation Status (REQ-VERIFY-3393)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3393 | Implemented (`scripts/experiment_3393_proximal_gradient_constraint.py`, `python/carnot/verify/proximal_gradient_constraint_layer.py`) | Implemented (`tests/python/test_experiment_3393_proximal_gradient.py`) |

### REQ-VERIFY-3576: Cross-Domain Value Synthesis
- REQ-VERIFY-3576-1: The system SHALL synthesize the cross-domain value of the verifier ensemble from upstream experiments.
- REQ-VERIFY-3576-2: It MUST write `results/experiment_3576_verifier_cross_domain_synthesis.json` with required fields.

### SCENARIO-VERIFY-3576: Cross-Domain Value Synthesis
When Exp 3576 runs, it produces all required schema fields on the exit path.

## Implementation Status (REQ-VERIFY-3576)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-VERIFY-3576 | Implemented (`scripts/experiment_3576_verifier_cross_domain_synthesis.py`) | Implemented (`tests/python/test_experiment_3576_verifier_cross_domain_synthesis.py`) |
