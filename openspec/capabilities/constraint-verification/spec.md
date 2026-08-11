# Constraint Verification Capability Specification

**Capability:** constraint-verification
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines deterministic audits for executable constraint-verification corpora.
These audits decide whether a frozen candidate pool is usable for later
selector experiments before any hidden-state extraction is attempted.

## Requirements

### REQ-CONSTRAINT-6274: Bounded ASP Semantic Energy Compiler

The repository SHALL provide an executable bounded ASP semantic compiler for
Exp6274. The compiler SHALL accept only explicit ground facts, grounded normal
rules, default negation in rule bodies, integrity constraints, and standalone
bounded cardinality choice rules. It SHALL reject unsupported syntax before any
energy terms are built.

The compiler SHALL emit inspectable energy terms for facts, normal rules,
integrity constraints, bounded cardinality rules, and stable-model support. For
every enumerated state it SHALL report per-rule violation receipts that name the
local failed rule or semantic support check.

The Exp6274 harness SHALL evaluate at least 40 trusted formal fixtures across
graph coloring, scheduling, non-monotonic defaults, contradictions, and positive
or negative controls. Every fixture SHALL have a bounded finite state space that
is exactly enumerated. The zero-energy states emitted by the compiler SHALL be
compared by set equality against an independent ASP solver answer-set list.

The terminal artifact SHALL be
`results/experiment_6274_asp_energy_semantic_compiler.json`. It SHALL state the
paper source and claim boundary, supported and unsupported ASP constructs,
source and fixture hashes, independent solver version and receipts, fixture
counts, exact state counts, per-fixture answer sets, zero-energy states,
semantic parity, rule-local violation evidence, all required controls, test
commands, exit codes, reproducibility checksum, `verifier_is_oracle=true`, and
an honest verdict. The artifact SHALL NOT claim a learned verifier or an
oracle-distinct verifier moat.

### SCENARIO-CONSTRAINT-6274-SOLVER-PARITY: Energy Matches ASP Answer Sets

Given trusted bounded ASP fixtures in the supported subset,
When Exp6274 compiles each fixture and enumerates all possible atom states,
Then the zero-energy states exactly equal the independent solver answer sets
for every fixture.

### SCENARIO-CONSTRAINT-6274-FAIL-CLOSED: Unsupported Syntax Is Rejected

Given ASP text with variables, disjunction, optimization, arithmetic terms, or
unsupported aggregates,
When the bounded compiler receives that text,
Then it rejects the program before energy construction and reports the failing
syntax class.

### SCENARIO-CONSTRAINT-6274-LOCAL-RECEIPTS: Violations Name Local Causes

Given an enumerated state that violates a fact, normal rule, integrity
constraint, bounded cardinality rule, or stable-model support check,
When the compiler decomposes energy for that state,
Then the receipt names the violated rule or semantic support check and gives a
non-zero local energy contribution.

### REQ-CONSTRAINT-6275: Flagship ASP Constraint Verification Benchmark

The repository SHALL provide an Exp6275 sealed natural-language benchmark built
from the Exp6274 fixture families. The benchmark SHALL expose only natural
language task descriptions to candidate models. It SHALL NOT expose ASP program
text, formal sidecars, zero-energy states, solver answer sets, exact answers,
or verifier receipts in model prompts.

Exp6275 SHALL run the mandated local GGUF models
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF` through direct llama.cpp-backed inference
with GPU offload receipts. Each model SHALL receive at least 30 tasks across the
supported Exp6274 fixture families. The run SHALL preserve every raw prompt,
raw output, seed, token count, timeout state, and content hash in immutable
sidecars.

Exp6275 SHALL compare one-shot output, fixed-budget self-consistency, and
energy-guided repair. It SHALL report parseability and semantic validity as
separate outcomes. It SHALL score semantic validity only with exact solver
certificates from the Exp6274 sidecar. It SHALL preserve per-model,
per-family, and per-arm results, including residual rule violations,
abstentions, failed or timeout cells, paired intervals, sample sizes, latency,
and token counts.

The terminal artifact SHALL be
`results/experiment_6275_flagship_asp_constraint_verification_benchmark.json`.
It SHALL include the task-required fields for model/cache/llama.cpp/GPU
receipts, sealed benchmark hashes, formal sidecar nonexposure, seed matrix,
task counts, raw output hashes, parse success, semantic validity, exact
certificate coverage, format and semantic repair margins, residual violations,
abstentions, paired intervals, latency, failed cells, event corpus readiness,
zero weight mutation, zero external text scorer calls, protected file receipts,
preconditions, `inference_substrate="live_llm_inference"`,
`verifier_is_oracle=true`, field provenance, one field principle per required
field, test commands, exit codes, duration, seed, checksum, and an honest
verdict. The artifact SHALL NOT claim an external text scorer, a learned
verifier, or an oracle-distinct verifier moat.

### SCENARIO-CONSTRAINT-6275-SEALED-PROMPTS: Sidecars Stay Hidden

Given Exp6275 tasks derived from Exp6274 fixtures,
When prompts are built for a model,
Then no prompt contains ASP syntax, exact answer strings, solver answer-set
lists, zero-energy state lists, formal sidecar hashes as answers, or verifier
receipt text.

### SCENARIO-CONSTRAINT-6275-SEPARATE-OUTCOMES: Format And Semantics Are Separate

Given raw model output for one-shot and self-consistency arms,
When Exp6275 scores the output,
Then parse success is recorded independently from exact semantic validity, and
format repair margin is not collapsed into semantic repair margin.

### SCENARIO-CONSTRAINT-6275-EXACT-REPAIR: Energy Repair Uses Solver Certificates

Given a parseable but semantically invalid assignment,
When Exp6275 runs energy-guided repair,
Then the repaired assignment is accepted only if the Exp6274 exact oracle
certifies it, and residual rule violations remain visible when repair fails.

### SCENARIO-CONSTRAINT-6275-BLOCKED-CELL: Failed Receipts Stop A Model Cell

Given a mandated model is missing, lacks GPU offload proof, times out, or lacks
required cache/hash receipts,
When Exp6275 evaluates the model cell,
Then the cell stops with an honest terminal disposition and the artifact records
the failed or timeout cell instead of fabricating model results.

### REQ-CONSTRAINT-6288: Fail-Closed Partial Atom Evidence Adapter

The repository SHALL provide an Exp6288 adapter that replays only the
Exp6286-eligible immutable raw rows from the flagged Exp6275 benchmark. The
adapter SHALL extract explicit positive atom evidence and explicit negative
atom evidence from ordinary model text. It SHALL mark all other atoms unknown.
It SHALL NOT ask a model for finite IDs, JSON, or formal theory. It SHALL NOT
run an LLM.

The adapter SHALL use a frozen atom vocabulary for each fixture before
extraction. It SHALL reject contradictions, foreign atoms, ambiguous negation,
empty outputs, zero-token rows, and evidence that no sealed exact answer set can
support. It SHALL check support against the sealed formal sidecar only after
the lexical extraction decision has been made.

The Exp6288 workflow SHALL compare evidence-warm continuous refinement and
exact completion against blank and random starts under identical fixed budgets.
It SHALL keep positive, negative, unknown, rejected, and contradictory states
separate. The readiness gate SHALL require at least one accepted row in each
represented mandated model family, positive evidence precision above the
preregistered floor, clean leakage controls, zero unsafe evidence acceptance,
and zero source model weight mutation. A positive warm-start delta SHALL NOT be
required for readiness.

The terminal artifact SHALL be
`results/experiment_6288_partial_atom_evidence_adapter.json`. It SHALL include
`status`, `upstream_eligibility_path_hash_and_terminal_class`,
`upstream_relaxation_path_hash_and_terminal_class`,
`eligible_raw_manifest_path_and_hash`, `raw_source_paths_and_hashes`,
`models_represented`, `frozen_atom_vocabulary_by_fixture`,
`adapter_source_paths_and_hashes`,
`positive_negative_unknown_evidence_by_row`,
`contradiction_foreign_atom_and_ambiguous_negation_rejections`,
`accepted_and_rejected_row_counts`,
`evidence_precision_coverage_and_sample_sizes_by_model_family_and_fixture_family`,
`evidence_leakage_controls`, `warm_blank_and_random_start_outcomes`,
`continuous_refinement_results`, `exact_completion_results`,
`cold_exact_completion_controls`, `unsafe_evidence_acceptance_count`,
`partial_atom_evidence_adapter_ready_score`,
`source_model_weight_mutation_count`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`, `test_exit_codes`,
`duration_s`, `random_seeds`, `reproducibility_checksum`, and
`honest_verdict`. `unsafe_evidence_acceptance_count` and
`source_model_weight_mutation_count` SHALL be bare integer zero.

### SCENARIO-CONSTRAINT-6288-EXTRACT-FAIL-CLOSED: Unsafe Text Is Rejected

Given ordinary model text that contains contradictory atom statements, a
foreign atom, ambiguous negation, an empty output, or a zero generated-token
receipt,
When Exp6288 extracts partial atom evidence,
Then the row is rejected with the rejection class recorded and no unsafe atom
is accepted.

**Spec traces:** REQ-CONSTRAINT-6288

### SCENARIO-CONSTRAINT-6288-ORACLE-AFTER-EXTRACTION: Labels Stay Hidden

Given an eligible raw row, a frozen atom vocabulary, and the sealed exact
sidecar,
When Exp6288 extracts atom evidence,
Then extraction uses only raw text and the vocabulary, and sealed exact answer
sets are consulted only afterward for support, precision, and coverage.

**Spec traces:** REQ-CONSTRAINT-6288

### SCENARIO-CONSTRAINT-6288-WARM-CONTROLS: Budgets Stay Matched

Given accepted partial atom evidence for an eligible fixture,
When Exp6288 runs evidence-warm, blank, and random continuous refinement plus
exact completion controls,
Then all arms use identical fixed budgets, and readiness does not depend on a
positive evidence-warm delta.

**Spec traces:** REQ-CONSTRAINT-6288

### REQ-CONSTRAINT-6289: Flagship Exact-State Refinement Benchmark

The repository SHALL provide an Exp6289 sealed live benchmark over the three
mandated local SOTA GGUF families:
`unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. `MODEL_SPECS` SHALL include all three hub
IDs. A headline row SHALL NOT use a legacy small model as a substitute.

Exp6289 SHALL freeze tasks, sidecars, model paths, quantizations, prompt
templates, budgets, seeds, timeouts, and protected hashes before inference. It
SHALL resolve GGUF paths with `cached_sota_pair()` or exact cached paths. It
SHALL record CUDA offload receipts for each model. If a model cannot run with a
terminal receipt, its row SHALL close as terminal blocked.

Exp6289 SHALL expose only ordinary text prompts. Prompts SHALL NOT expose ASP
theories, formal atom IDs, gold assignments, exact answer sets, or solver
receipts. The exact solver SHALL see the formal sidecar and SHALL be declared as
an oracle.

Exp6289 SHALL compare one-shot ordinary text, fixed repeated generation,
partial-evidence continuous refinement, partial-evidence exact completion, cold
exact completion, and a preregistered CoBa-inspired route that chooses
generation, verification, or stop under one fixed budget. It SHALL keep model
tokens, verifier work, exact solver work, and wall time separate.

Readiness SHALL require a positive preregistered warm-start work delta or a
positive bounded-refinement delta with no exact-validity harm. Cold exact solve success SHALL NOT count as model value.
Source model weights SHALL not mutate.

The terminal artifact SHALL be
`results/experiment_6289_flagship_exact_state_refinement_benchmark.json`. It
SHALL include `status`,
`upstream_adapter_path_hash_and_terminal_class`,
`sealed_task_manifest_path_and_hash`, `formal_sidecar_path_and_hash`,
`MODEL_SPECS`, `models_used`,
`model_file_hashes_revisions_and_quantizations`,
`tokenizer_and_chat_template_hashes`,
`cuda_and_gpu_offload_receipts_by_model`, `raw_output_paths_and_hashes`,
`prompt_seed_token_timeout_and_terminal_disposition_by_row`,
`terminal_model_dispositions`, `arm_definitions_and_fixed_compute_budget`,
`one_shot_results`, `repeated_generation_results`,
`partial_evidence_continuous_refinement_results`,
`partial_evidence_exact_completion_results`, `cold_exact_completion_results`,
`compute_balanced_route_results`,
`exact_validity_by_arm_model_and_fixture_family`,
`parser_and_evidence_coverage_by_arm_model_and_fixture_family`,
`solver_nodes_or_state_evaluations_by_arm_model_and_fixture_family`,
`model_tokens_verifier_work_and_wall_time_by_arm_model_and_fixture_family`,
`paired_deltas_intervals_and_sample_sizes`, `harmful_regressions`,
`qwen_zero_token_control`, `exact_solver_oracle_receipt`,
`warm_start_value_ready_score`, `source_model_weight_mutation_count`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`, `random_seeds`,
`reproducibility_checksum`, and `honest_verdict`.
`source_model_weight_mutation_count` SHALL be bare integer `0`.
`verifier_is_oracle` SHALL be `true`. Every model disposition SHALL be terminal.
`field_principles` SHALL give one short principle for every required field.
`honest_verdict` SHALL start with a terminal prefix.

### SCENARIO-CONSTRAINT-6289-SEALED-SOTA: Sealed Prompts Use Mandated GGUFs

Given Exp6289 model specs and sealed tasks,
When the benchmark freezes prompts and model rows,
Then all three mandated GGUF hub IDs are present, no legacy model substitutes a
headline row, and no prompt exposes ASP text, formal atom IDs, exact answers, or
solver receipts.

**Spec traces:** REQ-CONSTRAINT-6289

### SCENARIO-CONSTRAINT-6289-MATCHED-BUDGETS: Arms Share Fixed Budgets

Given matched tasks and seeds for one model family,
When Exp6289 runs generation, repeated generation, evidence refinement, exact
completion, cold exact completion, and the CoBa route,
Then all arms report their fixed budgets, model tokens, verifier work, solver
state evaluations, wall time, paired deltas, and sample sizes separately.

**Spec traces:** REQ-CONSTRAINT-6289

### SCENARIO-CONSTRAINT-6289-ORACLE-VALUE: Solver Work Is Not Laundered

Given exact solver success from a cold sidecar solve,
When Exp6289 computes readiness,
Then cold exact success is not counted as model value, zero-token Qwen rows fail
closed, nonterminal rows become terminal blocked, and readiness opens only with
positive warm-start or bounded-refinement value and no exact-validity harm.

**Spec traces:** REQ-CONSTRAINT-6289

### REQ-CONSTRAINT-VERIFY-6175: CCTU Headroom Audit Fail-Closed Gate

The repository SHALL provide an Exp6175 audit over the Exp6174 CCTU K8 pool.
The audit SHALL verify the Exp6174 structured gate, upstream bank/split/
validator/preregistration hashes, raw-before-label receipts, exact validator
version, calibration and held seals, K completeness, no-retry receipt,
preregistered gates and power plan, output paths, exclusions, and protected
files before any headroom metric is trusted.

The audit SHALL revalidate every calibration and held label from immutable raw
completion text using the Exp6173 exact validator. It SHALL report all-sample
and parseable denominators, per-candidate competence against the exact floor,
partial step satisfaction, violation taxonomy, duplicate clusters,
family/constraint-count strata, and parseability. Headline denominators SHALL
retain parse failures, refusals, timeouts, truncations, duplicates, and exact
validator failures.

The audit SHALL tune only the preregistered oracle-blind consensus rule on
calibration rows using normalized action/terminal-outcome clusters. Consensus
selection SHALL NOT use held labels, hidden states, arbitrary row identifiers,
answer positions, sample indexes, or validator labels at selection time.

The audit SHALL compute oracle@8, tuned-consensus accuracy, oracle-minus-
consensus delta, case-clustered intervals, consensus-wrong/oracle-right group
count, and error-diversity metrics. Held processing SHALL emit only aggregate
qualification fields plus a sealed held row-label hash, never held row labels
or per-row held outcomes.

The audit SHALL set `phase_d_headroom_ready_score` to bare `1.0` only when all
preregistered conjuncts pass: Exp6174 gate/preconditions, parseability,
above-floor competence, below-saturation error, oracle-minus-consensus at least
0.10 with lower CI above zero, at least 30 selectable minority groups, and
family support. If any conjunct fails, the artifact SHALL set `status` to
`retired`, `future_rows_allowed_by_this_artifact` to bare `false`, and
`honest_verdict` to a `retired:` terminal prefix naming the failed conjuncts.

The terminal artifact SHALL be
`results/experiment_6175_cctu_headroom_audit.json` and SHALL include the task's
required schema fields, including
`inference_substrate="deterministic_exact_tool_trace_headroom_audit"` and
`verifier_is_oracle=true`.

### REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY: Upstream Pool Authenticity Gate

Exp6175 SHALL fail closed unless the Exp6174 structured gate passed, current
upstream bank/split/validator/preregistration bytes match the hashes declared by
Exp6174, current raw-trace and label-sidecar bytes match Exp6174 receipts, raw
rows were committed before validation, K=8 rows exist for every frozen case, no
correctness-conditioned retry/replacement occurred, calibration and held seals
remain intact, preregistered gates and power are present, output paths are
writable, exclusions are present, and protected files are byte-stable.

### REQ-CONSTRAINT-VERIFY-6175-PARSEABILITY: All-Sample Parseability Accounting

Exp6175 SHALL compute parseability over every frozen candidate row. Headline
denominators SHALL retain unparseable completions, truncations, refusals,
timeouts, duplicates, and exact-validator failures. Parseability SHALL pass only
when the preregistered minimum is met; unparseable rows SHALL retire the domain
rather than being filtered out of the headline denominator.

### REQ-CONSTRAINT-VERIFY-6175-EXACT-FLOOR: Exact Floor Provenance

Exp6175 SHALL copy the exact random executable-plan floor from the Exp6173
preregistration, report its value and provenance, and affirm that no
finite-choice or answer-position floor is used.

### REQ-CONSTRAINT-VERIFY-6175-COMPETENCE: Candidate Competence Gate

Exp6175 SHALL compute per-candidate all-sample exact-validator accuracy and a
case-clustered interval. Competence SHALL pass only when the clustered lower
bound is strictly above the exact floor.

### REQ-CONSTRAINT-VERIFY-6175-UNSATURATION: Imperfect-Pool Gate

Exp6175 SHALL measure whether the candidate pool is competent but imperfect by
checking that candidate accuracy and tuned consensus are below the
preregistered saturation limits. This gate SHALL NOT rescue a failed
parseability, competence, headroom, minority, or family-support conjunct.

### REQ-CONSTRAINT-VERIFY-6175-CONSENSUS: Oracle-Blind Consensus Freeze

Exp6175 SHALL tune only the preregistered oracle-blind consensus family on
calibration rows using normalized action and terminal-outcome clusters. The
selection rule SHALL NOT use held labels, hidden states, arbitrary identifiers,
answer positions, sample indexes, or exact-validator labels at selection time.

### REQ-CONSTRAINT-VERIFY-6175-ORACLE-K: Oracle@K Headroom Measurement

Exp6175 SHALL compute oracle@8 as the case-level accuracy achieved when any of
the frozen K candidates passes the exact terminal validator. It SHALL compute
oracle-minus-consensus and retire unless the delta is at least 0.10 with a
case-clustered lower confidence bound above zero.

### REQ-CONSTRAINT-VERIFY-6175-ERROR-DIVERSITY: Error Diversity and Shortcut Audit

Exp6175 SHALL report normalized-cluster diversity, dominant failure surfaces,
consensus-wrong/oracle-right counts, duplicate clusters, raw duplicate shares,
and shortcut audits for answer-position, arbitrary-id, sample-index, and
hidden-state channels.

### REQ-CONSTRAINT-VERIFY-6175-CLUSTERED-INFERENCE: Case-Clustered Intervals

Exp6175 SHALL compute intervals over case clusters for per-candidate accuracy,
oracle@8, consensus accuracy, and oracle-minus-consensus. The artifact SHALL
declare the deterministic resampling method or constant-cluster shortcut used
for each interval.

### REQ-CONSTRAINT-VERIFY-6175-HELD-AGGREGATE: Held Labels Stay Aggregated

Exp6175 may inspect held labels internally for aggregate qualification, but the
terminal artifact SHALL emit only held aggregate counts/rates, an aggregate
signature, and a sealed held row-label hash. It SHALL NOT emit held sample keys,
row hashes, label hashes, validator-result objects, or per-row held outcomes.

### REQ-CONSTRAINT-VERIFY-6175-NO-SELECTOR: No Selector or Hidden-State Extraction

Exp6175 SHALL be a prerequisite audit only. It SHALL NOT tune a latent selector,
extract hidden states, expose held row labels downstream, use held rows for
consensus tuning, or allow any downstream rows when readiness is zero.

### REQ-CONSTRAINT-VERIFY-6175-FAIL-CLOSED-RETIREMENT: Strict Readiness Conjunction

Exp6175 SHALL set `phase_d_headroom_ready_score` to bare `1.0` only when every
preregistered authenticity, parseability, competence, unsaturation, headroom,
minority, and family-support conjunct passes. Otherwise it SHALL set `status`
to `retired` or `blocked`, `future_rows_allowed_by_this_artifact` to bare
`false`, and `honest_verdict` to a terminal prefix naming the failed conjuncts.

### SCENARIO-CONSTRAINT-VERIFY-6175-RAW-REVALIDATION: Labels Are Replayed From Raw Text

Given immutable Exp6174 raw trace rows and calibration/held label sidecars,
When Exp6175 runs,
Then every label is recomputed from `raw_completion_text` with the Exp6173 exact
validator, raw-row hashes are checked against sidecars, and any mismatch retires
the domain.

### SCENARIO-CONSTRAINT-VERIFY-6175-NO-HELD-ROWS: Held Labels Stay Sealed

Given Exp6175 may inspect held labels internally for aggregate qualification,
When it writes the artifact,
Then the held section contains aggregate rates, counts, a sealed held row-label
hash, and an aggregate signature only, with no held row labels, sample keys, row
hashes, or per-case held outcomes.

### SCENARIO-CONSTRAINT-VERIFY-6175-RETIRE-PARSE-FAILURE: Unparseable Pools Retire

Given Exp6174 contains a complete K8 raw-before-label pool whose candidates are
all retained but unparseable,
When Exp6175 computes the preregistered gates,
Then parseability, competence, headroom, minority, and family-support conjuncts
fail closed, readiness is zero, future rows are forbidden, and the artifact
reports `retired:`.

### REQ-VERIFY-6301: Activation Bus Integrity Audit

The repository SHALL provide an Exp6301 integrity audit for the frozen Exp6300
activation bus. The audit SHALL treat Exp6300 as a representation experiment,
not as an authority for state or energy claims.

Exp6301 SHALL reconstruct every evaluation row from Exp5852 row hashes and the
frozen Exp6300 fold manifest. It SHALL load only cached Exp6300 adapter
checkpoints. It SHALL NOT refit Exp6300. It SHALL NOT train a scientific energy
head. It SHALL initialize fresh evaluation consumers for the audit controls.

Exp6301 SHALL replay claim-flip, pair-member swap, deterministic label
permutation, model-ID prediction, norm-only, length-only, token-count,
truncation, duplicate reweighting, no-information, evaluator-swap, false-pass
injection, and held-family controls. Folds SHALL remain group-aware. Decisions
SHALL stay disaggregated by fold, model, axis, family, hardness, and surface.
Pooled results SHALL NOT override a failed cell.

Exp6301 SHALL require model-ID accuracy near chance in the shared space while
also reporting matched semantic alignment. It SHALL report both signals. It
SHALL NOT trade one off against the other silently.

The terminal artifact SHALL be
`results/experiment_6301_activation_bus_integrity_audit.json`. It SHALL include
`status`, `upstream_path_hash_and_terminal_class`, `MODEL_SPECS`,
`models_covered`, `row_and_checkpoint_reconstruction_receipts`,
`evaluator_independence_receipts`, `fold_leakage_checks`,
`claim_flip_sensitivity`, `pair_swap_controls`,
`label_permutation_controls`, `model_identity_controls`,
`norm_length_token_and_truncation_controls`,
`duplicate_and_no_information_controls`, `evaluator_swap_receipts`,
`disaggregated_cell_decisions`, `failed_cells`, `surviving_shortcuts`,
`false_pass_injection_results`, `activation_bus_integrity_ready_score`,
`source_mutation_count`, `protected_files_unchanged`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`random_seeds`, `reproducibility_checksum`, and `honest_verdict`.
`source_mutation_count` SHALL be the bare integer `0`. `MODEL_SPECS` SHALL name
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`.

Readiness SHALL be bare `1.0` only when all preconditions pass, all adequately
powered cells pass, every false-pass injection is blocked, zero shortcuts
survive, every test command exits zero, and protected files remain unchanged.
Otherwise readiness SHALL be bare `0.0`, and `honest_verdict` SHALL start with
a terminal prefix that names the blocking reason.

### SCENARIO-VERIFY-6301-RECONSTRUCT: Rows And Checkpoints Are Replayed

Given Exp5852 rows, the Exp6300 fold manifest, and Exp6300 adapter
checkpoints,
When Exp6301 runs,
Then it recomputes row hashes, checks checkpoint hashes, reconstructs held rows
from group-aware folds, and records that no Exp6300 refit occurred.

**Spec traces:** REQ-VERIFY-6301

### SCENARIO-VERIFY-6301-SHORTCUTS: Controls Fail Closed

Given a cached bus whose shared vectors carry a claim-direction, pair-swap,
label, model-ID, norm, length, token-count, truncation, duplicate, null, or
held-family shortcut,
When Exp6301 evaluates disaggregated cells,
Then the matching shortcut appears in `surviving_shortcuts`, the failed cell is
listed, and readiness is `0.0`.

**Spec traces:** REQ-VERIFY-6301

### SCENARIO-VERIFY-6301-INDEPENDENCE: Consumers Are Fresh

Given an Exp6300 artifact that reports ready decisions,
When Exp6301 audits it,
Then Exp6301 ignores Exp6300 decision labels, initializes independent
evaluation consumers, runs evaluator-swap and false-pass injections, and blocks
any injected false pass.

**Spec traces:** REQ-VERIFY-6301
