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

### REQ-KONA-6326: Restricted Policy DSL and Exact Contract Compiler

The repository SHALL provide Exp6326 as a bounded typed DSL for deterministic
finite state-action policies. The DSL SHALL declare finite states, finite
actions, and exactly one action for each state. The parser SHALL reject unknown
syntax, duplicate clauses, missing state actions, unknown states or actions,
and values that exceed the frozen finite bounds.

The compiler SHALL normalize each accepted program into canonical text with a
stable semantic hash. The semantics SHALL depend only on the finite policy map,
not on source order, comments, whitespace, parser defaults, natural-language
ConstraintIR, generated labels, generated-text scoring, hidden model state,
KANs, or model-generated annotations.

The contract layer SHALL define independent finite-domain behavioral clauses.
Each accepted contract SHALL name its finite state and action domain. It SHALL
compile to named local factors. Each factor SHALL have a finite scope, a
positive integer weight, and an exact satisfaction predicate over the canonical
policy semantics. Total energy SHALL be the weighted count of unsatisfied
factors.

The Exp6326 harness SHALL ship development and held contract families plus
adversarial fixtures. It SHALL verify each family by enumerating the complete
finite policy domain or by Z3. It SHALL prove that factor energy equals exact
contract violations for every enumerated policy. It SHALL ship one verified,
hash-pinned fallback policy program for every fixture family.

The terminal artifact SHALL be
`results/experiment_6326_restricted_policy_contract_compiler.json`. It SHALL
include `status`, source and claim boundaries, grammar, schema, parser,
normalizer, semantics, factor compiler, exact checker, fixture manifest, split
rules, fallback hashes, exhaustive results by family, factor exactness results,
parser rejection and totality results, attack controls, exact oracle boundary,
bare zero counts for generated labels, hidden-state access, and external text
scorers, protected-file receipts, preconditions, `verifier_is_oracle=true`,
field provenance, field principles for every required field, test commands,
exit codes, duration, seeds, checksum, and an honest verdict. Readiness SHALL
be `1.0` only when normalization is deterministic, factor energy equals exact
violations, all fallbacks satisfy their contracts, and all attacks fail closed.

### SCENARIO-KONA-6326-CANONICAL-PARSER: Parser Is Total And Deterministic

Given bounded policy programs with the same state-action map but different
source order, whitespace, or comments,
When Exp6326 parses and normalizes them,
Then the canonical program text and semantic hash are identical.

**Spec traces:** REQ-KONA-6326

### SCENARIO-KONA-6326-FACTOR-EXACTNESS: Local Factors Equal Exact Violations

Given accepted contracts over finite state-action domains,
When Exp6326 compiles each contract and enumerates every deterministic policy,
Then the sum of unsatisfied local factors equals the independent exact violation
count for every enumerated policy.

**Spec traces:** REQ-KONA-6326

### SCENARIO-KONA-6326-FALLBACK-AND-ATTACKS: Fallbacks Are Pinned And Attacks Fail

Given each fixture family fallback and adversarial candidates for vacuity,
parser defaults, validator mutation, test deletion, fallback laundering, hash
swaps, collision probes, and nondeterministic normalization,
When Exp6326 verifies the family manifest,
Then every fallback satisfies its contract and every attack fails closed.

**Spec traces:** REQ-KONA-6326

### REQ-KONA-6327: Three-Family Guarded Policy Synthesis

The repository SHALL provide Exp6327 at
`python/carnot/experiment_6327_three_family_guarded_policy_synthesis.py`.
It SHALL write
`results/experiment_6327_three_family_guarded_policy_synthesis.json` after
generating bounded policy DSL candidates with the three mandatory local GGUF
models: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`.

`MODEL_SPECS` SHALL be resolved through the canonical `cached_sota_pair`
helper pattern with `gpu_indices=(0, 1)`. The workflow SHALL preserve exact
GGUF file paths, hashes, revisions, quantizations, GPU placements, the native
llama.cpp CLI path, and embedded-tokenizer receipts for each model. It SHALL
never call `AutoTokenizer.from_pretrained()` for a GGUF repository.

The workflow SHALL replay the structured gate before generation. The gate SHALL
check model files, embedded tokenizers, CUDA devices, VRAM, RAM, disk, timeouts,
seeds, fixture hashes, candidate budgets, fallback hashes, and protected hashes.
If any required model cell is missing, the artifact SHALL block with an honest
verdict rather than using a tiny or legacy model for headline rows.

The workflow SHALL freeze prompts, decoder settings, seeds, candidate count,
token budget, wall-time budget, and fallback accounting before model output is
read. Raw model outputs SHALL be written atomically and hashed before parsing.
Every parsed candidate SHALL be normalized by the Exp6326 restricted policy DSL.
Malformed or parser-failing output SHALL be conserved as a candidate failure.

The workflow SHALL evaluate four matched arms over identical eligible model,
family, and seed cells: one raw candidate, reject-only filtering, exact guard
plus hash-pinned fallback, and bounded exact-factor-energy-guided candidate
search plus the same fallback. Every fallback SHALL count as the fallback
utility and full fallback cost. The exact guard SHALL be the safety oracle.
The model SHALL supply only candidate programs and SHALL NOT supply labels,
hidden states, source model weight updates, or final safety authority.

The terminal artifact SHALL include every required field named in the Exp6327
task prompt. `source_model_weight_mutation_count`, `generated_label_count`, and
`hidden_state_access_count` SHALL be bare integer zero. `verifier_is_oracle`
SHALL be the bare boolean `true`. `guarded_policy_synthesis_ready_score` SHALL
be `1.0` only when all required model and contract-family cells are complete,
no accepted candidate has a contract violation, development-family utility for
bounded exact-factor-energy search improves over exact guard plus fallback under
the preregistered matched budget, protected files remain unchanged, and every
required field has a field principle. Otherwise it SHALL be `0.0`.

### SCENARIO-KONA-6327-GATE: Local GGUFs And Embedded Tokenizers Gate The Run

Given Exp6327 starts from the three mandatory local GGUF models,
When a model file, embedded tokenizer, CUDA offload receipt, native llama.cpp
path, budget, fallback hash, or protected hash is missing,
Then the workflow blocks that evidence path and does not substitute
`AutoTokenizer`, hidden states, generated labels, or tiny legacy models.

**Spec traces:** REQ-KONA-6327

### SCENARIO-KONA-6327-MATCHED-ARMS: Candidate Budgets Are Shared

Given raw rows are frozen for each model, family, and seed,
When the four arms are scored,
Then each arm uses the same candidate pool, token budget, time budget, and
fallback accounting, and malformed output remains visible as a parse failure.

**Spec traces:** REQ-KONA-6327

### SCENARIO-KONA-6327-ORACLE-BOUNDARY: Exact Guard Holds Safety Authority

Given a model candidate is accepted by an arm,
When the Exp6326 exact factor energy is computed,
Then accepted contract violations are zero, rejected candidates use the
hash-pinned fallback when that arm defines one, and readiness never depends on
model labels, hidden states, source model mutation, or model self-judgment.

**Spec traces:** REQ-KONA-6327

### REQ-KONA-6339: Incremental Prefix Enforcement Substrate

The repository SHALL provide Exp6339 at
`python/carnot/experiment_6339_incremental_prefix_enforcement_substrate.py`.
It SHALL write
`results/experiment_6339_incremental_prefix_enforcement_substrate.json`.

Exp6339 SHALL reuse the Exp6326 restricted policy DSL grammar, parser,
normalizer, exact finite semantics, fixture families, and readiness receipt.
It SHALL add a deterministic incremental lexer and parser. Completed programs
accepted by the incremental parser SHALL parse through Exp6326 and have the
same canonical source and semantic hash as Exp6326.

The incremental parser SHALL expose only observable parser-state features.
Those features SHALL include syntax phase, accepted completed line count,
declared state and action counts, rule count, missing state actions, current
expectations, prefix hash, and parser error reason when one exists. They SHALL
NOT read model hidden state, generated labels, LLM self-judgments, logits, or
natural-language repair text.

Exp6339 SHALL provide a JIT SMT prefix-feasibility checker. The checker SHALL
return `accept`, `reject`, or `timeout`. It SHALL use a bounded solver timeout.
It SHALL treat timeout as fail-closed and report timeout counts separately.

The Exp6339 harness SHALL enumerate every prefix in the bounded fixture domain.
It SHALL prove prefix soundness, feasible-completion recall, parser-state
determinism, final semantic parity with Exp6326, and adversarial fail-closed
behavior. Adversarial controls SHALL include invalid UTF-8, token splits,
whitespace aliases, prefix bombs, solver timeouts, unknown symbols, and
normalization collisions.

The terminal artifact SHALL include every required field named in the Exp6339
task prompt. `hidden_state_access_count`, `generated_label_count`, and
`llm_call_count` SHALL be bare integer zero. `verifier_is_oracle` SHALL be the
bare boolean `true`. `prefix_enforcement_substrate_ready_score` SHALL be `1.0`
only when exhaustive soundness, full feasible-completion recall, semantic
parity, deterministic states, and bounded fail-closed timeouts all pass.

### SCENARIO-KONA-6339-PREFIX-SOUNDNESS: Prefix Rejection Is Exact

Given every bounded fixture-domain prefix,
When Exp6339 checks prefix feasibility,
Then no accepted prefix is outside the feasible completion set and timeout
prefixes are reported as fail-closed.

**Spec traces:** REQ-KONA-6339

### SCENARIO-KONA-6339-FEASIBLE-RECALL: Feasible Prefixes Are Not Rejected

Given every prefix that has at least one fixture-domain completion,
When Exp6339 runs the JIT SMT feasibility checker without forcing timeout,
Then the checker does not return `reject`.

**Spec traces:** REQ-KONA-6339

### SCENARIO-KONA-6339-SEMANTIC-PARITY: Completed Programs Match Exp6326

Given every accepted completed fixture-domain program,
When Exp6339 normalizes and hashes the program,
Then the canonical source and semantic hash match Exp6326 exactly.

**Spec traces:** REQ-KONA-6339

### SCENARIO-KONA-6339-OBSERVABLE-STATE: Parser State Is Deterministic

Given repeated parses of byte-identical prefixes and whitespace aliases,
When Exp6339 emits parser-state features,
Then the features are deterministic and record zero hidden-state, generated
label, and LLM accesses.

**Spec traces:** REQ-KONA-6339

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

### REQ-CONSTRAINT-VERIFY-6415: Exact Boolean WCSP CCG Kernelization Control

Carnot SHALL provide a bounded exact Boolean WCSP-to-constraint-composite-graph
control in `python/carnot/experiment_6415_boolean_wcsp_ccg_kernelization.py`.
The run command SHALL write
`results/experiment_6415_boolean_wcsp_ccg_kernelization.json`.

The Boolean WCSP schema SHALL be canonical and reversible. It SHALL name
variables, integer constants, unary terms, pairwise terms, duplicate handling,
and deterministic ordering. The CCG schema SHALL include source, sink,
variable nodes, auxiliary term nodes, weighted edges, graph-cut capacities, and
a source-instance mapping contract.

The kernelizer SHALL use a local maxflow/min-cut construction only for exact
submodular Boolean pairwise costs. It SHALL fix variables only when residual
reachability proves the label is forced in all minimum cuts. Every fixed
variable SHALL carry a certificate that an independent exact check verifies
against the source instance.

The experiment SHALL freeze at least 48 small instances across unary,
pairwise, frustrated, sparse, dense, decomposable, degenerate, and
adversarial-weight classes. For every instance it SHALL compare the
unkernelized exact reference, CCG maxflow kernelization plus exact completion,
and a seeded energy-sampling control. The exact reference owns optimum labels.

The attack matrix SHALL cover sign inversion, zero or negative weights,
duplicate constraints, disconnected components, auxiliary-node omission,
mapping reversal, integer overflow, unsound fixed variables, and non-unique
optima. `ccg_kernelization_exact_ready_score` SHALL be bare `1.0` only when
every frozen optimum is preserved, every fixed-variable certificate verifies,
no attack creates an unsound reduction, and reduction and cost fields are
measured.

The artifact SHALL include `status`,
`source_encoder_solver_sampler_and_dependency_hashes`,
`boolean_wcsp_schema_path_hash_and_fields`,
`ccg_schema_path_hash_node_edge_and_mapping_contract`,
`kernelizer_path_and_hash`,
`frozen_manifest_path_hash_counts_classes_and_seeds`,
`exact_reference_method_and_receipts`,
`per_instance_source_and_kernelized_optima`,
`optimum_preservation_rate`,
`fixed_variable_certificates_and_independent_checks`,
`state_space_reduction_by_instance`,
`verifier_call_reduction_by_instance`,
`sampler_work_and_wall_time_by_arm`,
`sign_weight_duplicate_component_auxiliary_mapping_overflow_fixed_variable_and_nonunique_attack_matrix`,
`quantum_advantage_claimed`, `hardware_speedup_claimed`,
`ccg_kernelization_exact_ready_score`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

`verifier_is_oracle` SHALL be true only for the independent exact optimum and
certificate checks. The kernelizer and sampler SHALL NOT be oracles.
`quantum_advantage_claimed` and `hardware_speedup_claimed` SHALL be false.

Field principles SHALL be:

- `status`: Names whether the exact local control is usable or blocked.
- `source_encoder_solver_sampler_and_dependency_hashes`: Pins the code and dependency inputs used before this local control ran.
- `boolean_wcsp_schema_path_hash_and_fields`: Makes the source WCSP schema inspectable and content-addressed.
- `ccg_schema_path_hash_node_edge_and_mapping_contract`: Makes the CCG graph and reverse mapping contract inspectable.
- `kernelizer_path_and_hash`: Pins the local maxflow kernelizer implementation.
- `frozen_manifest_path_hash_counts_classes_and_seeds`: Shows the frozen panel size, classes, seeds, and manifest hash.
- `exact_reference_method_and_receipts`: Declares exhaustive enumeration as the independent optimum authority.
- `per_instance_source_and_kernelized_optima`: Compares source and kernelized exact optima for every frozen instance.
- `optimum_preservation_rate`: Measures the fraction of frozen instances whose optimum is preserved exactly.
- `fixed_variable_certificates_and_independent_checks`: Records each fixed-variable certificate and its independent exact check.
- `state_space_reduction_by_instance`: Measures how many exact states remain after certified fixes.
- `verifier_call_reduction_by_instance`: Measures exact verifier calls saved by completion after kernelization.
- `sampler_work_and_wall_time_by_arm`: Keeps exact arms and the seeded energy sampler costed separately.
- `sign_weight_duplicate_component_auxiliary_mapping_overflow_fixed_variable_and_nonunique_attack_matrix`: Proves known unsafe reductions are rejected or abstained.
- `quantum_advantage_claimed`: Must stay false because this is a local CPU control.
- `hardware_speedup_claimed`: Must stay false because no hardware path is used.
- `ccg_kernelization_exact_ready_score`: Equals 1.0 only when preservation, certificates, attacks, and measurements all pass.
- `protected_files_unchanged`: Shows conductor and reconciliation files stayed byte-stable.
- `preconditions_checked`: Lists the local gates checked before the result is trusted.
- `inference_substrate`: Declares deterministic CPU enumeration and local maxflow, not LLM inference.
- `verifier_is_oracle`: Marks only exact optimum and certificate checks as oracles.
- `field_principles`: Documents why each required artifact field exists.
- `field_provenance`: States how each required artifact field was produced.
- `random_seed`: Pins deterministic fixture and sampler replay.
- `duration_s`: Records wall time for the experiment command.
- `tests_run`: Records the verification commands run for this artifact.
- `reproducibility_checksum`: Content-addresses the payload with this field blanked.
- `honest_verdict`: Gives a terminal-prefix verdict that names the exact readiness outcome.

### SCENARIO-CONSTRAINT-VERIFY-6415-EXACT-PRESERVATION: Kernelization Preserves Optima

Given the frozen Boolean WCSP manifest,
When Exp6415 runs,
Then every CCG-kernelized exact-completion optimum equals the unkernelized
exact reference optimum, every fixed-variable certificate passes an independent
source-instance check, and `optimum_preservation_rate` is `1.0`.

**Spec traces:** REQ-CONSTRAINT-VERIFY-6415

### SCENARIO-CONSTRAINT-VERIFY-6415-ATTACKS: Unsafe Reductions Fail Closed

Given sign, weight, duplicate, component, auxiliary, mapping, overflow,
fixed-variable, and non-unique optimum attacks,
When Exp6415 audits each attack,
Then unsound or malformed reductions are rejected, abstained, or held at
readiness zero rather than credited as safe fixed-variable reductions.

**Spec traces:** REQ-CONSTRAINT-VERIFY-6415

### SCENARIO-CONSTRAINT-VERIFY-6415-NO-SPEEDUP-CLAIM: Local Control Only

Given Exp6415 compares exact reference, CCG kernelization, and energy sampling,
When it writes the terminal artifact,
Then the artifact declares a deterministic CPU local-control substrate, makes no
quantum or hardware speedup claim, and sets readiness to `1.0` only from exact
preservation and certificate checks.

**Spec traces:** REQ-CONSTRAINT-VERIFY-6415

### REQ-CONSTRAINT-VERIFY-6416: Selective Exact Refinement A/B Replay

Carnot SHALL provide Exp6416 at
`python/carnot/experiment_6416_selective_exact_refinement_ab.py`. The command
`.venv/bin/python -m carnot.experiment_6416_selective_exact_refinement_ab --date 20260814`
SHALL write
`results/experiment_6416_selective_exact_refinement_ab.json`.

Exp6416 SHALL consume only the frozen Exp6414 factor-event corpus and the
frozen Exp6415 Boolean WCSP CCG certificate artifact. It SHALL not call an LLM.
It SHALL revalidate the Exp6414 and Exp6415 gates, corpus seals, raw-output
hashes, checker versions, CCG certificates, and protected future partition
before any arm outcome is accepted.

The trigger contract SHALL be registered before acceptance outcomes are
exposed. Only these exact trigger classes may route a row to refinement:
`exact_abstention`, `missing_provenance`, `checker_disagreement`, and
`certified_ccg_reducible`. The contract SHALL separate counts and work for each
trigger class. Model confidence, score magnitude, model identity pooling, and
post-outcome class labels SHALL NOT authorize acceptance.

The A/B replay SHALL compare three matched arms over identical frozen rows:
`never_refine`, `always_refine`, and `selective_refine`. Refinement may recover
source spans from sealed Exp6414 source text, replay the deterministic Exp6414
exact checker, or apply independently verified Exp6415 CCG certificates. A row
may be accepted only when a deterministic exact checker or independent CCG
certificate-backed replay gives exact authority. Routing and confidence SHALL
not be oracles.

The artifact SHALL measure exact yield, false accepts, false rejects,
unresolved abstentions, checker calls, kernelization work, raw-tier
escalations, latency, and cost overall and by model family and trigger class.
It SHALL emit the bare finite scalar `delta_exact_yield_over_never_refine`,
the bare finite scalar `selective_vs_always_exact_accuracy_delta`, and the bare
finite scalar `selective_vs_always_work_delta`.

The attack matrix SHALL include confidence-only routing, trigger tampering,
post-outcome selection, CCG certificate substitution, source fabrication,
pooled model identities, and future-label leakage. Each attack SHALL fail
closed. `selective_refinement_safe_score` SHALL be the bare scalar `1.0` only
when selective false accepts do not exceed never-refine false accepts,
protected leakage is zero, exact yield improves over never-refine or matched
accuracy uses less exact work than always-refine, and every attack fails
closed. Otherwise it SHALL be `0.0`.

The artifact SHALL include `status`,
`exp6414_and_exp6415_gate_receipts`,
`corpus_certificate_checker_and_partition_hashes`,
`preregistered_trigger_contract`,
`preregistered_never_always_and_selective_arm_contract`,
`matched_work_contract`,
`per_arm_exact_yield_false_accept_false_reject_abstention_checker_kernel_escalation_latency_and_cost_results`,
`per_model_family_and_trigger_class_results`,
`delta_exact_yield_over_never_refine`,
`selective_vs_always_exact_accuracy_delta`,
`selective_vs_always_work_delta`, `confidence_authority_count`,
`protected_leakage_count`, `attack_matrix`,
`selective_refinement_safe_score`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

Field principles SHALL be:

- `status`: Names whether the selective exact refinement replay is safe, blocked, or null.
- `exp6414_and_exp6415_gate_receipts`: Pins both upstream gates before any arm uses their evidence.
- `corpus_certificate_checker_and_partition_hashes`: Binds raw rows, checker versions, CCG certificates, and the future partition.
- `preregistered_trigger_contract`: Shows the exact route triggers and excludes confidence authority.
- `preregistered_never_always_and_selective_arm_contract`: Defines the three matched arms before outcome selection.
- `matched_work_contract`: Keeps row sets and work units identical across comparable arms.
- `per_arm_exact_yield_false_accept_false_reject_abstention_checker_kernel_escalation_latency_and_cost_results`: Reports the required arm metrics.
- `per_model_family_and_trigger_class_results`: Disaggregates results by model family and trigger class.
- `delta_exact_yield_over_never_refine`: Bare yield lift from selective refinement over never-refine.
- `selective_vs_always_exact_accuracy_delta`: Bare matched exact-accuracy difference for selective minus always.
- `selective_vs_always_work_delta`: Bare matched work difference for selective minus always.
- `confidence_authority_count`: Must stay zero because confidence is diagnostic only.
- `protected_leakage_count`: Must stay zero because protected future labels cannot route rows.
- `attack_matrix`: Proves confidence, trigger, certificate, source, identity, and future-label attacks fail closed.
- `selective_refinement_safe_score`: Bare gate for downstream use.
- `protected_files_unchanged`: Shows protected upstream and ops files stayed byte-identical.
- `preconditions_checked`: Lists local gates checked before accepting the artifact.
- `inference_substrate`: Declares frozen deterministic replay with no new LLM calls.
- `verifier_is_oracle`: Marks only exact event checkers and independent CCG certificate checks as oracles.
- `field_principles`: Documents why each required field exists.
- `field_provenance`: States how each required field was produced.
- `random_seed`: Pins deterministic trigger and CCG certificate mapping.
- `duration_s`: Records command wall time.
- `tests_run`: Records verification commands and exit codes.
- `reproducibility_checksum`: Content-addresses the payload with volatile fields normalized.
- `honest_verdict`: Gives a terminal-prefix verdict with the exact authority boundary.
- `gate:exp6414`: Exp6414 is a gate, not a mutable data source.
- `gate:exp6415`: Exp6415 certificates are gate evidence, not routing confidence.
- `arm:never_refine`: The baseline accepts only already exact rows.
- `arm:always_refine`: The expensive control refines every frozen row.
- `arm:selective_refine`: The selective arm refines only rows allowed by the preregistered triggers.

### SCENARIO-CONSTRAINT-VERIFY-6416-TRIGGERS: Confidence Cannot Route Or Accept

Given the frozen Exp6414 rows and Exp6415 certificates,
When Exp6416 builds the trigger contract,
Then only exact abstention, missing provenance, checker disagreement, or
certified CCG reducibility may select refinement, and
`confidence_authority_count` remains `0`.

**Spec traces:** REQ-CONSTRAINT-VERIFY-6416

### SCENARIO-CONSTRAINT-VERIFY-6416-MATCHED-ARMS: Selective Matches Always With Less Work

Given never-refine, always-refine, and selective-refine run on the same frozen
rows,
When deterministic source recovery, exact checker replay, and CCG certificate
checks finish,
Then selective exact accuracy matches always-refine, selective work is less
than always-refine, and selective exact yield exceeds never-refine without
increasing false accepts.

**Spec traces:** REQ-CONSTRAINT-VERIFY-6416

### SCENARIO-CONSTRAINT-VERIFY-6416-ATTACKS: Unsafe Refinement Fails Closed

Given confidence-only routing, trigger tampering, post-outcome selection, CCG
certificate substitution, source fabrication, pooled model identity, and
future-label leakage attacks,
When Exp6416 validates the artifact,
Then each attack fails closed and `selective_refinement_safe_score` can become
`1.0` only if every attack failed closed.

**Spec traces:** REQ-CONSTRAINT-VERIFY-6416

### REQ-CONSTRAINT-VERIFY-6429: Constraint Saturation Verification-Cost A/B Replay

Carnot SHALL provide Exp6429 at
`python/carnot/experiment_6429_constraint_saturation_verification_cost_ab.py`.
The command
`.venv/bin/python -m carnot.experiment_6429_constraint_saturation_verification_cost_ab --date 20260814`
SHALL write
`results/experiment_6429_constraint_saturation_verification_cost_ab.json`.

Exp6429 SHALL consume only clean Exp6427 rows plus the frozen Exp6416 and
Exp6415 exact-refinement evidence. It SHALL not call an LLM. It SHALL
revalidate upstream gates, immutable row hashes, raw row hashes, constraint
strata, checker versions, CCG certificate versions, partition seals, CPU, RAM,
disk, and monotonic timing before any readiness score is credited.

The replay SHALL preregister `never_refine`, `always_refine`, and
`selective_refine` arms before outcome-derived cost-error counts are computed.
The preregistered contract SHALL declare fixed checker-call and wall-time
budgets. A verification-cost error SHALL mean an exact-incorrect row that the
arm did not identify as an error within its declared budget. Abstained rows
SHALL be reported as abstentions, not hidden as correct rows.

The selective arm SHALL use only `exact_abstention`, `missing_provenance`,
`checker_disagreement`, or `certified_ccg_reducible` triggers. Confidence may
be recorded as a diagnostic. Confidence, model identity pooling, post-outcome
class labels, and future partitions SHALL NOT authorize row acceptance.
Deterministic event checkers and independent CCG certificate checks are the
only oracles.

The artifact SHALL record per-unit arm outcomes with constraint count,
interaction class, model family, exact result, detected error, abstention,
checker calls, elapsed time, budget exhaustion, and cost-error status. It SHALL
derive per-constraint success, joint success, joint-success decay,
interaction penalties, verification-cost error rates, false accepts, false
rejects, exact work, time-to-verdict, uncertainty, and effective sample size
from `per_unit_rows`.

The selective-vs-always comparison SHALL test matched joint exact accuracy and
median and tail cost by constraint count, interaction class, and model family.
It SHALL report null, underpowered, or crossover cells without pooling them
away. `verification_cost_study_ready_score` SHALL be the bare scalar `1.0`
only when budgets were frozen, rows are complete, all aggregates recompute,
selective false accepts do not exceed always-refine false accepts, and every
critical attack fails closed. A null selective advantage SHALL still be a
complete result when the gates pass.

The attack matrix SHALL include confidence authority, outcome-aware budget
choice, post-outcome trigger selection, row deletion, model pooling,
certificate substitution, source fabrication, and future leakage. Each attack
SHALL fail closed.

The artifact SHALL include `status`, `exp6427_gate_receipts`,
`corpus_row_checker_certificate_and_partition_hashes`,
`preregistered_arm_and_budget_contract`,
`verification_cost_error_definition`, `per_unit_rows`,
`per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results`,
`per_constraint_success`, `joint_success`,
`joint_success_decay_by_constraint_count`, `interaction_penalty`,
`verification_cost_error_rate_by_budget`,
`false_accept_and_false_reject_deltas`,
`selective_vs_always_accuracy_delta`,
`selective_vs_always_median_and_tail_cost_deltas`,
`effective_sample_sizes_and_uncertainty`, `aggregate_recomputation_receipts`,
`reported_vs_recomputed_deltas`, `confidence_authority_count`,
`attack_matrix`, `verification_cost_study_ready_score`,
`harm_underpowered_missing_and_flagged_cells`, `protected_files_unchanged`,
`blocked_reason`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`,
`random_seed`, `duration_s`, `tests_run`, `reproducibility_checksum`, and
`honest_verdict`.

Field principles SHALL be:

- `status`: Names whether the verification-cost replay is complete, blocked, or null.
- `exp6427_gate_receipts`: Pins the clean row gate and the Exp6416 exact-refinement reference gate.
- `corpus_row_checker_certificate_and_partition_hashes`: Binds rows, raw outputs, checkers, certificates, and partitions.
- `preregistered_arm_and_budget_contract`: Freezes arms and budgets before cost-error aggregation.
- `verification_cost_error_definition`: Defines cost errors separately from exact correctness.
- `per_unit_rows`: Provides the row and arm decisions behind every comparative claim.
- `per_arm_constraint_count_interaction_model_budget_correctness_abstention_checker_time_and_cost_error_results`: Reports matched arm metrics by required strata.
- `per_constraint_success`: Reports exact per-constraint success from rows.
- `joint_success`: Reports exact joint success from rows.
- `joint_success_decay_by_constraint_count`: Shows joint collapse as simultaneous constraints accumulate.
- `interaction_penalty`: Measures interacting minus independent outcomes at each constraint count.
- `verification_cost_error_rate_by_budget`: Reports budgeted misses of exact-incorrect rows.
- `false_accept_and_false_reject_deltas`: Shows whether selective changes release harm against controls.
- `selective_vs_always_accuracy_delta`: Bare matched accuracy delta for selective minus always.
- `selective_vs_always_median_and_tail_cost_deltas`: Shows median and tail cost savings or crossover cells.
- `effective_sample_sizes_and_uncertainty`: Reports sample size and uncertainty for each stratum.
- `aggregate_recomputation_receipts`: States the formulas and row hashes used for aggregate recomputation.
- `reported_vs_recomputed_deltas`: Shows reported metrics equal row recomputation.
- `confidence_authority_count`: Must stay zero because confidence is diagnostic only.
- `attack_matrix`: Proves known authority, budget, row, source, certificate, and leakage attacks fail closed.
- `verification_cost_study_ready_score`: Bare readiness gate for downstream use.
- `harm_underpowered_missing_and_flagged_cells`: Names underpowered or missing cells instead of pooling them away.
- `protected_files_unchanged`: Shows conductor, ops, traceability, and upstream artifacts stayed byte-stable.
- `blocked_reason`: Names any precondition blocker.
- `preconditions_checked`: Lists local gates checked before accepting the artifact.
- `inference_substrate`: Declares deterministic replay over frozen rows with no new LLM call.
- `verifier_is_oracle`: Marks only deterministic event and certificate checks as oracles.
- `field_principles`: Documents why each required field exists.
- `field_provenance`: States how each required field was produced.
- `random_seed`: Pins deterministic row order, trigger mapping, and budget replay.
- `duration_s`: Records command wall time.
- `tests_run`: Records required test, coverage, spec, adversarial, and root-clutter checks.
- `reproducibility_checksum`: Content-addresses the artifact with volatile fields normalized.
- `honest_verdict`: Gives a terminal-prefix verdict with the exact authority boundary.
- `gate:exp6427`: Exp6427 is the immutable row gate, not a mutable data source.
- `gate:exp6416`: Exp6416 supplies a frozen exact-refinement reference, not new outcomes.
- `arm:never_refine`: The baseline spends no extra checker calls and exposes budgeted misses.
- `arm:always_refine`: The control spends exact checker work on every row.
- `arm:selective_refine`: The selective arm spends work only under allowed exact triggers.
- `budget:checker_calls`: Checker-call limits are frozen before cost-error aggregation.
- `budget:wall_time`: Wall-time limits are frozen before cost-error aggregation.
- `cost_error`: Cost errors measure missed exact-incorrect rows within budget.
- `readiness:verification_cost_study_ready_score`: Readiness requires frozen budgets, complete rows, recomputation, no added false accepts, and closed attacks.

### SCENARIO-CONSTRAINT-VERIFY-6429-BUDGETS: Budgets Freeze Before Cost Errors

Given clean Exp6427 rows and frozen Exp6416 evidence,
When Exp6429 builds the arm and budget contract,
Then never, always, and selective budgets are registered before
cost-error aggregation, confidence has no authority, and
`confidence_authority_count` remains `0`.

**Spec traces:** REQ-CONSTRAINT-VERIFY-6429

### SCENARIO-CONSTRAINT-VERIFY-6429-MATCHED-ARMS: Selective Matches Always With Lower Cost

Given the three arms run on identical Exp6427 rows,
When per-row arm outcomes are replayed,
Then selective and always have equal matched joint accuracy, selective has
lower median and tail time-to-verdict, and selective false accepts do not
exceed always-refine false accepts.

**Spec traces:** REQ-CONSTRAINT-VERIFY-6429

### SCENARIO-CONSTRAINT-VERIFY-6429-ROWS-AND-ATTACKS: Rows Recompute And Attacks Fail Closed

Given the Exp6429 artifact,
When aggregate recomputation and adversarial mutation checks run,
Then all reported aggregate deltas are zero, per-unit rows are present, and
confidence, outcome-aware budgets, post-outcome triggers, row deletion, model
pooling, certificate substitution, source fabrication, and future leakage fail
closed before readiness can become `1.0`.

**Spec traces:** REQ-CONSTRAINT-VERIFY-6429

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

### REQ-CONSTRAINT-6604: Exact Two-Level Plan Corpus And Compiler Contract

Carnot SHALL provide Exp6604 as a deterministic exact-executable plan corpus.
The corpus SHALL contain at least 72 bounded tasks. Exactly 36 tasks SHALL be
calibration tasks, and exactly 36 tasks SHALL be held tasks. Each task SHALL
record immutable canonical source bytes, a fixed seed, one frozen split, and a
content hash. Each task SHALL declare grounded actions, argument grammar,
initial state, action preconditions and effects, ordering obligations, goal
predicates, known feasibility, and an exact deterministic outcome. The corpus
SHALL stratify lexical, temporal, branching, and distractor features without
claiming that symbolic solver difficulty predicts model difficulty.

Carnot SHALL expose separate reusable token-syntax and action-semantics
compiler interfaces. The syntax compiler SHALL enforce canonical action names,
arguments, types, arity, delimiters, and unambiguous plan boundaries. The
semantic compiler SHALL consume structured meta-tokens. It SHALL expose state
transitions for preconditions, ordering obligations, and goals. Compiler
versions, meta-token mappings, transitions, and receipts SHALL be replayable.

An independent exact executor SHALL decide final benchmark validity. It SHALL
not call either decoding compiler and SHALL not use their acceptance result.
The fixture SHALL include a deliberate incomplete semantic encoding. At least
one candidate SHALL pass syntax and the incomplete semantic automaton, then
fail the independent executor because the compiler omitted an obligation.
Automata acceptance SHALL therefore never certify benchmark validity.

The corpus SHALL retain one fixture row per task and one row per generated
mutation. Mutations SHALL cover valid plans, syntax errors, precondition
violations, ordering violations, unmet goals, impossible tasks, parser
ambiguity, split leakage, and incomplete encodings. Attacks SHALL cover split
leakage, duplicate bytes, seed drift, nondeterminism, goal-answer leakage,
compiler-executor sharing, impossible-task mislabeling, and protected-file
mutation. Every attack SHALL fail closed.

Exp6604 SHALL write
`results/experiment_6604_exact_two_level_plan_corpus.json` by complete-temp
write, file synchronization, and atomic replacement. It SHALL set
`inference_substrate` to
`deterministic_two_level_plan_fixture_and_exact_executor_no_llm` and
`verifier_is_oracle` to `true`. A complete contract SHALL use
`verdict_class=null` and SHALL make no model-benefit claim.
`headroom_fixture_ready_score` SHALL be the bare value `1.0` only when every
task, split, hash, compiler receipt, executor receipt, mutation, attack,
protected-file check, focused test, and output check replays. Otherwise the
score SHALL be `0.0`, and `gate_check_summary` SHALL name each failed observed
value.

### SCENARIO-CONSTRAINT-6604-GENERATION-AND-SPLITS: Corpus Bytes And Splits Freeze

Given the frozen generator version and seed schedule,
When Exp6604 generates the corpus twice,
Then both runs produce the same 72 canonical task bytes and hashes, with 36
calibration rows, 36 held rows, unique bytes, and no cross-split membership.

**Spec traces:** REQ-CONSTRAINT-6604

### SCENARIO-CONSTRAINT-6604-TWO-LEVEL-COMPILATION: Syntax And Semantics Stay Separate

Given one bounded task and a candidate plan,
When the two compiler interfaces run,
Then syntax produces canonical grounded meta-tokens and semantics produces
explicit state transitions without merging either compiler with exact release
authority.

**Spec traces:** REQ-CONSTRAINT-6604

### SCENARIO-CONSTRAINT-6604-INDEPENDENT-EXECUTION: Exact Execution Owns Validity

Given valid, malformed, precondition-failing, order-failing, goal-incomplete,
ambiguous, and impossible plan cases,
When the independent executor runs twice,
Then both executions are identical and only fully executable goal-satisfying
plans are valid.

**Spec traces:** REQ-CONSTRAINT-6604

### SCENARIO-CONSTRAINT-6604-INCOMPLETE-ENCODING: Automata Do Not Certify Themselves

Given a test semantic compiler that deliberately omits the audit-before-ship
obligation,
When it checks a canonical plan without the audit action,
Then syntax and both encoded automata accept the plan while the independent
executor rejects it.

**Spec traces:** REQ-CONSTRAINT-6604

### SCENARIO-CONSTRAINT-6604-ROW-RETENTION-AND-ATOMIC-OUTPUT: Evidence Is Complete

Given all task and mutation rows,
When Exp6604 writes its terminal artifact,
Then every row remains present, all required fields have principles and
provenance, the checksum replays, and atomic replacement leaves no partial
terminal file.

**Spec traces:** REQ-CONSTRAINT-6604

### SCENARIO-CONSTRAINT-6604-ADVERSARIAL-CONTROLS: Corpus Attacks Fail Closed

Given injected leakage, duplicate bytes, seed drift, nondeterminism, answer
leakage, compiler-executor sharing, feasibility mislabeling, or protected-file
mutation,
When the corpus gate evaluates the injection,
Then the matching detector reports the observed defect and readiness stays
zero for the attacked copy.

**Spec traces:** REQ-CONSTRAINT-6604

## Implementation Status (REQ-CONSTRAINT-6604)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6604 and SCENARIO-CONSTRAINT-6604-* | Implemented (`python/carnot/experiment_6604_exact_two_level_plan_corpus.py`; terminal evidence in `results/experiment_6604_exact_two_level_plan_corpus.json`) | `tests/python/test_experiment_6604_exact_two_level_plan_corpus.py`: 9 passed; scoped module coverage: 493 statements, 0 missing (100%); conductor-equivalent smart subset: 90 passed, 1 pre-existing warning |

### REQ-CONSTRAINT-6649: Exact Proposal Checking SHALL Localize The First Failure

Exp6649 SHALL check each parsed proposal with the Exp6604 independent exact
executor. It SHALL retain one exact outcome for every parsed step. An invalid
proposal SHALL record the first failing step, its reason, and the exact-valid
prefix length. A parse failure SHALL remain distinct from an exact invalid
plan. It SHALL not become an invalid plan with prefix length zero.

`regeneration_headroom_rows` SHALL contain only invalid parsed plans with a
non-empty exact-valid prefix and at least one remaining target step. The task
SHALL count these rows without changing or regenerating a proposal.

#### SCENARIO-CONSTRAINT-6649-FIRST-FAILURE

**Given** a parsed plan with valid early actions and one later invalid action
**When** Exp6649 checks the proposal step by step
**Then** it records every step outcome, the first failing step, and the exact
valid-prefix length.

#### SCENARIO-CONSTRAINT-6649-PARSE-FAILURE

**Given** output that cannot produce canonical action lines
**When** Exp6649 parses and checks the row
**Then** it records an explicit parse failure and does not invent an exact
failure step or a zero-length invalid plan.

## Implementation Status (REQ-CONSTRAINT-6649)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6649 and SCENARIO-CONSTRAINT-6649-* | Implemented (`python/carnot/experiment_6649_exact_certificate_proposal_corpus.py`) | Implemented (`tests/python/test_experiment_6649_exact_certificate_proposal_corpus.py`; exact localization, parse-failure, mutation, and replay coverage) |

### REQ-CONSTRAINT-6650: Exact Twins SHALL Change One Localized Step Only

Exp6650 SHALL construct twins only from Exp6649 rows with a boolean exact
label and an exact-valid clean plan. Each error twin SHALL replace one complete
step with a different canonical task action. The replacement SHALL keep the
same UTF-8 byte count. The clean and error plans SHALL keep identical prefix
bytes, suffix bytes, line count, task, and formatting outside that step.

The Exp6649 independent exact executor SHALL replay both twins. It SHALL label
the clean plan valid and the error plan invalid at the changed step. A parse
failure, an already-invalid candidate, a final-step mutation, or a row without
a byte-matched semantic mutation SHALL produce an explicit rejected-pair row.
It SHALL not disappear from `per_unit_rows`.

#### SCENARIO-CONSTRAINT-6650-PAIRABLE-TWIN

**Given** an exact-valid candidate with a byte-matched canonical action that
fails at one non-final step
**When** Exp6650 constructs the twin
**Then** only that step differs and exact replay assigns clean and error labels.

#### SCENARIO-CONSTRAINT-6650-NON-PAIRABLE

**Given** a parse failure, an invalid candidate, or no safe byte-matched
replacement
**When** Exp6650 evaluates pairability
**Then** it emits a rejected-pair row with the source row ID and reason.

#### SCENARIO-CONSTRAINT-6650-EXACT-AUTHORITY

**Given** any advisory verifier score or decision
**When** Exp6650 records a twin outcome
**Then** only the frozen exact executor supplies the clean or error label and
no advisory result authorizes output.

## Implementation Status (REQ-CONSTRAINT-6650)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6650 and SCENARIO-CONSTRAINT-6650-* | Implemented (`python/carnot/experiment_6650_twin_prefix_verifier_map.py`) | Implemented (`tests/python/test_experiment_6650_twin_prefix_verifier_map.py`; byte-local semantic twins, exact authority, and rejection coverage) |

### REQ-CONSTRAINT-6661: Triggered Tails SHALL Enforce Syntax Without Answer Semantics

Exp6661 SHALL freeze at least 18 tasks across scheduling, graph constraints,
and arithmetic or logic certificates. Each task SHALL have a directly
executable exact checker. The output contract SHALL transport a certificate,
not a finite answer identifier.

Exp6661 SHALL define natural extraction, immediate JSON, and triggered-tail
arms before later model execution. The triggered arm SHALL allow free reasoning
before one fixed trigger. It SHALL apply a short syntax grammar only after that
trigger. All arms SHALL use fixed token budgets and versioned fail-closed
parsers.

The grammar MAY fix field names and primitive JSON types. It SHALL NOT contain
task IDs, labels, targets, candidate answers, answer enums, or checker logic.
Grammar-only generation SHALL not recover a correct certificate. Syntax-only
mutations SHALL not change an exact checker label.

The frozen manifest SHALL record each task, family, visible prompt, target,
checker identity, seed, and content hashes. It SHALL bind the three arm
contracts, parser versions, grammar hash, checker hashes, and token budgets
before any model run.

Each task-arm pair SHALL retain rows for answer permutation, label renaming,
task-ID removal, grammar-only generation, trigger collision, premature trigger,
missing trigger, malformed tail, unknown fields, and a semantically wrong but
syntactically valid tail. Missing, malformed, duplicate, or unknown parser data
SHALL fail closed. The exact checker SHALL remain the only semantic authority.

Exp6661 SHALL write
`results/experiment_6661_triggered_tail_fixture.json` atomically. It SHALL set
`inference_substrate` to `cpu_fixture_and_exact_checker_no_llm` and
`verifier_is_oracle` to `true`. A ready fixture SHALL use `verdict_class=null`
and SHALL make no model-quality claim.

`triggered_tail_fixture_ready` SHALL be true only when all expected task,
arm, checker-control, and attack rows exist. Every positive and negative
checker control SHALL pass. Every parser result SHALL match its frozen
expectation. No semantic leakage SHALL be present. Otherwise readiness SHALL
be false, and `gate_check_summary` SHALL name the first failed check and its
observed value.

### SCENARIO-CONSTRAINT-6661-DELAYED-SYNTAX: Grammar Starts After One Trigger

**Given** a response with non-empty free reasoning, one frozen trigger, and a
schema-valid tail
**When** the triggered-tail parser runs
**Then** only the bytes after the trigger use the syntax grammar and the exact
checker decides certificate validity.

**Spec traces:** REQ-CONSTRAINT-6661

### SCENARIO-CONSTRAINT-6661-SEMANTIC-FREE-GRAMMAR: Grammar Cannot Carry The Answer

**Given** any frozen task, target, label renaming, or answer permutation
**When** the grammar receipt and grammar-only sample are inspected
**Then** the grammar hash is unchanged, no task semantic appears, and the
grammar-only sample does not pass the exact checker.

**Spec traces:** REQ-CONSTRAINT-6661

### SCENARIO-CONSTRAINT-6661-IMMUTABLE-MANIFEST: Tasks And Arms Freeze Before Execution

**Given** the fixed task and attack seeds
**When** Exp6661 builds the fixture twice
**Then** task order, prompts, targets, checker identities, arm contracts,
budgets, parser versions, grammar bytes, and hashes are identical.

**Spec traces:** REQ-CONSTRAINT-6661

### SCENARIO-CONSTRAINT-6661-FAIL-CLOSED-PARSERS: Invalid Transport Never Becomes A Label

**Given** duplicate triggers, premature or missing triggers, malformed tails,
unknown fields, duplicate JSON fields, or wrong primitive types
**When** the applicable arm parser runs
**Then** it returns a named parse failure and does not invoke or invent a
semantic success.

**Spec traces:** REQ-CONSTRAINT-6661

### SCENARIO-CONSTRAINT-6661-ATTACK-AND-READINESS: Complete Rows Own The Gate

**Given** every frozen task, arm, checker control, and required attack
**When** readiness is recomputed only from retained rows
**Then** readiness is true only for complete expected row keys, passing
controls, matching expected parser outcomes, and zero leakage findings.

**Spec traces:** REQ-CONSTRAINT-6661

## Implementation Status (REQ-CONSTRAINT-6661)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6661 and SCENARIO-CONSTRAINT-6661-* | Implemented (`python/carnot/experiment_6661_triggered_tail_fixture.py`) | Implemented (`tests/python/test_experiment_6661_triggered_tail_fixture.py`; 32 focused tests; 100% scoped statement coverage) |

### REQ-CONSTRAINT-6676: Three Output Transports SHALL Share One Frozen Exact Corpus

Exp6676 SHALL use the qualified Exp6675 task manifest without changing task
prompts, checker inputs, parser versions, arm contracts, or grammar bytes. It
SHALL compare natural extraction, immediate JSON, and one lazy triggered-tail
generation for every model-task unit. The lazy grammar SHALL start only after
the exact frozen trigger. It SHALL contain JSON syntax only.

Each arm SHALL make one planned generation. Exp6676 SHALL not regenerate a
suffix, expose a target, use a finite answer ID, add solver output to a prompt,
or use an answer-bearing grammar alternative. A parse failure SHALL stay a
semantic failure. Only the frozen family-specific executable checker SHALL set
the exact outcome.

#### SCENARIO-CONSTRAINT-6676-ONE-GENERATION

**Given** one frozen model-task-arm identity and seed
**When** Exp6676 collects its output
**Then** the row records one request, one raw response, the frozen parser, and
one exact checker result with no repair or suffix regeneration.

#### SCENARIO-CONSTRAINT-6676-LAZY-SYNTAX

**Given** the triggered-tail arm
**When** llama.cpp emits the frozen trigger
**Then** the semantic-free GBNF activates after that trigger and the retained
raw response separates reasoning, trigger position, and JSON tail bytes.

#### SCENARIO-CONSTRAINT-6676-EXACT-AUTHORITY

**Given** any parsed certificate from any arm
**When** semantic success is reduced
**Then** only the task's frozen executable checker decides success and grammar
acceptance never becomes a semantic label.

## Implementation Status (REQ-CONSTRAINT-6676)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6676 and SCENARIO-CONSTRAINT-6676-* | Planned (`python/carnot/experiment_6676_three_family_triggered_tail_ab.py`) | Planned (`tests/python/test_experiment_6676_three_family_triggered_tail_ab.py`) |

### REQ-CONSTRAINT-6702: Exact Finite-Horizon Planning Fixture

Carnot SHALL provide Exp6702 as a deterministic finite-horizon planning
fixture. It SHALL contain inventory, battery-dispatch, job-slot, and
reservoir-control families. Each family SHALL have at least eight headline
instances and separate development instances. Horizons SHALL not exceed eight.
Action domains SHALL not exceed five actions.

Each instance SHALL retain an answer-free natural-language prompt and a typed
executable specification. The specification SHALL freeze its generator,
parameters, split, seed, transition function, hard constraints, objective,
tie policy, feasibility policy, and minimization convention. Exact dynamic
programming SHALL retain every reachable state and every action result. Each
result SHALL state legality, next state, immediate cost, exact future value,
total value, action gap, and optimum membership. Each instance SHALL also
retain its optimum action set, deterministic optimum plan, total optimum, tie
flag, and feasibility flag.

Exact labels SHALL use immutable seals. The live prompt API SHALL deny label
access until it receives a valid prompt-bound commit receipt. Exact labels MAY
authorize later evaluation. They SHALL not select a current-event action.

An independent exhaustive solver SHALL replay a frozen task-owned subset. It
SHALL compare every optimum and state-action value in that subset. Metamorphic
checks SHALL cover action renaming, constant cost shifts, equivalent state
encodings, and family-preserving prompt changes. Mutations SHALL cover bad
transitions, infeasible actions, corrupted costs, label leakage, wrong tie
metadata, and stale seals. Every mutation SHALL be detected.

`planning_fixture_ready` SHALL be true only when coverage, exactness, sealing,
split isolation, metamorphic checks, mutation detection, focused tests, scoped
coverage, specification coverage, and applicable end-to-end checks pass. The
reducer SHALL derive readiness only from retained per-unit rows. The fixture
SHALL not depend on a runtime manifest-parity artifact.

#### SCENARIO-CONSTRAINT-6702-EXACT-ROWS: Dynamic Programming Retains Every Value

**Given** one frozen instance from each planning family
**When** Exp6702 solves all reachable states
**Then** every legal action has an exact immediate, future, total, and gap value
**And** an independent exhaustive replay matches every checked value.

**Spec traces:** REQ-CONSTRAINT-6702

#### SCENARIO-CONSTRAINT-6702-SEALED-LABELS: Current Labels Need A Commit

**Given** a live prompt whose exact labels exist in the sealed evaluator
**When** a caller requests labels without a valid prompt-bound commit receipt
**Then** access fails with a stable negative result
**And** the same labels become readable only after a valid commit.

**Spec traces:** REQ-CONSTRAINT-6702

#### SCENARIO-CONSTRAINT-6702-ATTACKS: Representation Changes Hold And Mutations Fail

**Given** the four required metamorphic transforms and six required mutations
**When** Exp6702 replays its fixture checks
**Then** each invariant holds and each mutation is detected by its named check.

**Spec traces:** REQ-CONSTRAINT-6702

#### SCENARIO-CONSTRAINT-6702-ROW-REDUCTION: Raw Units Own Readiness

**Given** instance, state-action, solver, seal, metamorphic, mutation, and test
rows
**When** readiness is rebuilt
**Then** every required unit is present and passing before readiness is true
**And** removing or corrupting one required unit makes readiness false.

**Spec traces:** REQ-CONSTRAINT-6702

## Implementation Status (REQ-CONSTRAINT-6702)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6702 and SCENARIO-CONSTRAINT-6702-* | Implemented (`python/carnot/experiment_6702_exact_planning_fixture_recovery.py`) | Implemented (`tests/python/test_experiment_6702_exact_planning_fixture_recovery.py`; 11 focused tests; 100% scoped statement coverage) |

### REQ-CONSTRAINT-6703: Cold Exact Planning Fixture Audit

Exp6703 SHALL audit a deterministic blinded subset of the Exp6702 fixture.
The audit SHALL select one development unit and two headline units from each
family. It SHALL sort units by a SHA-256 score from the fixed blinding seed and
the instance identity. It SHALL freeze the selected identities, specification
hashes, prompt hashes, seal hashes, selection scores, and reveal order before
it reads any reported optimum or solver aggregate.

The audit SHALL implement its own family transitions and exhaustive path
enumerator. It SHALL not import the Exp6702 module or its dynamic-programming
solver. For every selected unit, it SHALL enumerate all action sequences. It
SHALL recompute legal actions, transitions, feasibility, plan totals, the
optimum, all optimum plans, first-action ties, state-action values, and action
gaps. It SHALL compare each available reported field without filling a missing
value or converting a blocker to zero.

#### SCENARIO-CONSTRAINT-6703-COLD-RECOMPUTATION: Independent Paths Rebuild Exact Labels

**Given** a valid frozen sample manifest and raw typed specifications
**When** the independent enumerator explores every action sequence
**Then** every selected unit has a receipt for its enumeration count, optimum,
tie set, action values, gaps, feasibility, runtime, and field comparison.

**Spec traces:** REQ-CONSTRAINT-6703

#### SCENARIO-CONSTRAINT-6703-BLINDING: Labels Open Only After Identity Freeze

**Given** the public instance identities, prompt hashes, specification hashes,
and seal hashes
**When** Exp6703 selects the blinded units
**Then** the manifest hash and ordered identities exist before any reported
optimum, action value, tie, or solver aggregate is read.

**Spec traces:** REQ-CONSTRAINT-6703

#### SCENARIO-CONSTRAINT-6703-COVERAGE: Every Selected Exact Unit Is Conserved

**Given** twelve selected units across all four families and both splits
**When** audit coverage is reduced
**Then** expected and observed instance, state, action, and solver counts match
and each selected identity occurs exactly once in the independent solver rows.

**Spec traces:** REQ-CONSTRAINT-6703

## Implementation Status (REQ-CONSTRAINT-6703)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6703 and SCENARIO-CONSTRAINT-6703-* | Implemented (`python/carnot/experiment_6703_exact_planning_fixture_audit.py`) | Implemented (`tests/python/test_experiment_6703_exact_planning_fixture_audit.py`; 9 focused tests and 100% scoped statement coverage) |

### REQ-CONSTRAINT-6715: Bounded Independent Exact Replay Audit

Exp6715 SHALL cold-recompute exactly eight Exp6702 headline instances. It
SHALL select two instances per planning family. The preregistered edge-probe
slot SHALL be `headline-01`. The contrast slot SHALL have the lowest SHA-256
selection score among the remaining headline instances. Selection SHALL use
only public identities, split and family names, prompt hashes, specification
hashes, seal hashes, and typed specifications. The audit SHALL freeze the
selection hash, reveal order, caps, versions, and protected-file hashes before
it reads any reported optimum, tie, feasibility, state-action value, or
producer aggregate. A revealed edge-probe that is neither tie-bearing nor
infeasible SHALL fail the audit. It SHALL NOT cause sample replacement.

The audit SHALL implement its own transitions and exhaustive complete-path
solver. It SHALL not import or call the Exp6702 producer or dynamic-programming
solver. For each selected typed specification, it SHALL enumerate every action
sequence. It SHALL recompute legal transitions, feasible plans, exact plan
costs, optima, optimum plans, initial tie sets, every reachable state-action
value, and every action gap.

The frozen manifest SHALL retain one deep-copied typed specification for each
selected identity. Each copy SHALL match its frozen specification hash. The
audit solver SHALL read this frozen copy. It SHALL not read the typed
specification from a row that also contains revealed labels.

The fixed caps SHALL be maximum horizon 6, maximum action count 5, maximum
reachable state count 128, maximum 15,625 complete paths per instance, maximum
50,000 complete paths across the sample, and maximum 600 seconds for audit
enumeration. The audit SHALL stop with a named cap row before it exceeds a cap.
It SHALL not reduce a cap, widen the sample, or substitute a different method.
The reachable state count SHALL include every time layer from the initial state
through the terminal layer.

#### SCENARIO-CONSTRAINT-6715-FROZEN-SAMPLE: Eight Identities Precede Label Reveal

**Given** only the public projection of the Exp6702 headline store
**When** Exp6715 freezes its sample
**Then** the manifest contains two instances from each family, the edge and
contrast selection receipts, the reveal order, fixed caps, versions, and hashes
**And** it contains one hash-matching typed specification per selected identity
**And** any input containing a reported label field is rejected before freeze.

**Spec traces:** REQ-CONSTRAINT-6715

#### SCENARIO-CONSTRAINT-6715-EXHAUSTIVE: Complete Paths Rebuild Every Exact Value

**Given** one selected typed specification within every fixed cap
**When** the independent solver enumerates all complete action sequences
**Then** it retains legal transitions, feasible plan counts and cost receipts,
optima, optimum plans, tie sets, state-action values, gaps, and a solver receipt.

**Spec traces:** REQ-CONSTRAINT-6715

#### SCENARIO-CONSTRAINT-6715-CAPS: A Bound Stops Without Sample Drift

**Given** a specification or cumulative enumeration that exceeds a frozen cap
**When** Exp6715 checks the bound
**Then** it records expected and observed values, stops before enumeration
continues, keeps the eight frozen identities, and leaves the audit gate false.
The observed state count includes the terminal layer.

**Spec traces:** REQ-CONSTRAINT-6715

## Implementation Status (REQ-CONSTRAINT-6715)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6715 and SCENARIO-CONSTRAINT-6715-* | Implemented (`python/carnot/experiment_6715_bounded_exact_replay_audit.py`) | Implemented (`tests/python/test_experiment_6715_bounded_exact_replay_audit.py`; 13 focused tests; 100% scoped statement coverage) |

### REQ-CONSTRAINT-6810: V595 Exact Priority Arbiter Ownership

The constraint-verification capability SHALL own the V595
`exact priority arbiter`. The implementing task is
`exp6813-selective-priority-arbiter-ab`. It SHALL write
`results/experiment_6813_selective_priority_arbiter_ab.json`.
The downstream completion gate is `selective_arbiter_ab_completed`.

The arbiter SHALL preserve an already-valid proposal byte for byte. Otherwise,
it SHALL compare candidates in exact lexicographic order: hard feasibility,
binding obligations in declared authority order, then soft progress. The exact
transition evaluator remains authoritative. A learned score, model identity,
future outcome, or exact utility label SHALL NOT become a proposal-time
authority.

The arbiter SHALL fail closed on an accepted hard violation, priority
inversion, stale prerequisite, authority spoof, missing fallback, weakened
consequence, unavailable legal candidate, incomplete certificate, budget
mismatch, or undeclared selection feature. A failed owned precondition SHALL
produce `complete_blocked_selective_priority_arbiter_ab`. The artifact SHALL
name the failed check, expected value, and observed value in
`gate_check_summary`. The completion field SHALL remain false.

#### SCENARIO-CONSTRAINT-6810-LEXICOGRAPHIC-AUTHORITY

**Given** frozen candidates with hard, binding, and soft obligations
**When** the V595 arbiter selects or abstains
**Then** finite soft value cannot compensate for a hard violation, safe input
bytes remain unchanged, and each rejection names its first higher-priority
conflict.

## Implementation Status (REQ-CONSTRAINT-6810)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6810 and SCENARIO-CONSTRAINT-6810-* | Planned: Exp6813 and `results/experiment_6813_selective_priority_arbiter_ab.json`. | Planned after Exp6810 contract preflight. |

### REQ-CONSTRAINT-6813: Deterministic Selective Priority Arbiter A/B

Exp6813 SHALL read the complete Exp6812 proposal corpus without invoking a
model. It SHALL stop with `complete_blocked_selective_priority_arbiter_ab` when
the corpus readiness flag, all three required model families, raw byte hashes,
paired candidate rows, or the Exp6811 compiler receipt is absent or invalid.
The blocked artifact SHALL contain no comparison rows. Its
`gate_check_summary` SHALL name each failed check with its expected and
observed value.

The experiment SHALL freeze development scenario IDs and held scenario IDs
before it reduces any outcome. It SHALL derive each fitted public constant
from development rows only. The held rows SHALL supply every headline rate and
paired interval. Each source model, scenario, seed, and handoff encoding SHALL
form one candidate set. Both comparison arms SHALL receive the same set in the
same frozen order.

The selective arm SHALL first check candidate zero as the base proposal. If
the base has valid syntax, zero hard violations, zero binding shortfall, and
exact proposal-time support, the arm SHALL preserve its action bytes. It SHALL
not replace a valid base to gain soft value. Otherwise, it SHALL reject every
hard violation and minimize the tuple of hard violations, binding shortfall in
declared authority order, negative clipped soft score, and frozen candidate
index. The last component gives a stable tie. If no hard-safe supported
candidate exists, the arm SHALL abstain to the declared fallback.

The flat arm SHALL inspect candidates in frozen order. It SHALL reject a
candidate after a syntax, hard, binding, support, or authority failure. It
SHALL accept the first candidate with no such failure. If none exists, it
SHALL abstain to the declared fallback. Both arms SHALL have equal candidate,
proposal-time exact-check, retry-cap, outcome-check, work-unit, and CPU
allowances. Realized sequential retries and measured latency remain outcomes;
they are not budget permissions.

Selection SHALL use only parsed proposal bytes, frozen order, the declared
obligation contract, observed facts, parse state, exact hard count, declared
authority-order binding vector, and proposed soft score. Selection SHALL not
use model identity, model family, scenario identity, exact answer, future
outcome, post-selection progress, harmful-selection label, or exact utility.
The exact transition evaluator SHALL run after selection and outside both arms.

Every model-scenario-seed-handoff-arm unit SHALL record selected bytes,
legality, post-selection progress, realized retries, candidate evidence,
conflict or no-op certificate, false intervention, safe-action byte identity,
abstention, work, and measured latency. A rejection certificate SHALL name the
first unsatisfied higher-priority item. False intervention means changing a
base proposal that the frozen validity test already classified as valid.

The positive gate SHALL require zero accepted hard violations, the held
false-intervention upper bound at or below its frozen limit, a positive held
paired lower bound for progress improvement or retry improvement, no required
model-family support loss, and no increase in harmful selection. A completed
null result is valid. `selective_arbiter_ab_completed` SHALL depend only on
complete rows, equal budgets, deterministic reduction, passed attacks, and the
frozen manifest. It SHALL not depend on the effect sign.

#### SCENARIO-CONSTRAINT-6813-LEXICOGRAPHIC: Hard and Binding Authority Dominate Soft Value

**Given** a hard-violating candidate with unbounded finite soft score and a
hard-safe candidate with lower soft score
**When** the selective arbiter compares the frozen set
**Then** it rejects the hard violation, respects the authority-ordered binding
vector, and uses soft score only after the higher-priority components tie.

**Spec traces:** REQ-CONSTRAINT-6813

#### SCENARIO-CONSTRAINT-6813-STABLE-TIE: Frozen Order Resolves Exact Ties

**Given** two supported candidates with identical hard, binding, and clipped
soft components
**When** the selective arbiter compares them more than once
**Then** it selects the lower frozen candidate index each time.

**Spec traces:** REQ-CONSTRAINT-6813

#### SCENARIO-CONSTRAINT-6813-NO-LEGAL: Missing Support Produces Abstention

**Given** a candidate set with no hard-safe supported candidate
**When** either arm completes its bounded checks
**Then** it selects no candidate, returns the declared fallback bytes, and
emits a complete no-legal-candidate certificate.

**Spec traces:** REQ-CONSTRAINT-6813

#### SCENARIO-CONSTRAINT-6813-NO-OP: A Valid Base Is Byte-Preserved

**Given** candidate zero is already valid
**When** the selective arm runs
**Then** its selected action bytes equal the base action bytes, false
intervention is false, and the certificate identifies a preserved no-op.

**Spec traces:** REQ-CONSTRAINT-6813

#### SCENARIO-CONSTRAINT-6813-CERTIFICATE: First Conflict Is Complete

**Given** a rejected base with more than one lower-priority defect
**When** the arbiter explains its intervention
**Then** the certificate names the first syntax, hard, or authority-ordered
binding conflict and does not substitute a later soft shortfall.

**Spec traces:** REQ-CONSTRAINT-6813

#### SCENARIO-CONSTRAINT-6813-ACCOUNTING: Safe Changes and Budgets Are Exact

**Given** paired selective and flat rows over the same frozen candidate set
**When** Exp6813 reduces false intervention and work
**Then** only changes to valid base bytes enter the false-intervention
numerator and both arms retain equal frozen work allowances and observed exact
check counts.

**Spec traces:** REQ-CONSTRAINT-6813

#### SCENARIO-CONSTRAINT-6813-COMPLETION: Effect Sign Does Not Control Readiness

**Given** all source units, budget receipts, reducers, attacks, and manifest
checks are complete
**When** the held paired effect is positive, zero, or negative
**Then** `selective_arbiter_ab_completed` is true and `verdict_class` reports
the measured effect without changing completion.

**Spec traces:** REQ-CONSTRAINT-6813

## Implementation Status (REQ-CONSTRAINT-6813)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6813 and SCENARIO-CONSTRAINT-6813-* | Planned: `python/carnot/experiment_6813_selective_priority_arbiter_ab.py`. | Planned focused tests and scoped coverage. |

### REQ-CONSTRAINT-6812: Authentic Paired Operational Handoff Corpus

The constraint-verification capability SHALL produce an authentic local-model
proposal corpus for the Exp6811 owned operational-obligation contract. The
corpus SHALL freeze at least 48 source-free scenarios before inference. The
scenarios SHALL cover stale prerequisites, competing authorities, fallback,
execution consequence, already-safe proposals, and soft conflict. Each
scenario SHALL have a direct typed handoff and a compressed prose handoff with
the same UTF-8 byte length.

The producer SHALL invoke exactly `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF` through local llama.cpp CUDA GGUF inference.
It SHALL run sequential task-owned GPU phases and SHALL NOT call a provider
API, pass a GGUF repository to `AutoTokenizer`, shrink a model, or substitute a
model. A bounded load failure SHALL write
`complete_blocked_sota_operational_handoff_corpus_v2`, name the failed gate,
and stop without pooling an incomplete model phase into the headline corpus.

Scenario, seed, token, candidate, stop, and retry budgets SHALL match between
the two arms. The producer SHALL freeze both prompt byte strings before
inference and retain their hashes and immutable bytes. It SHALL retain raw
output bytes, first-token evidence, file hashes, GPU ownership and offload
receipts, VRAM, process identifiers, separate acquisition, load, inference,
and teardown durations, and teardown evidence.

The parser SHALL apply the frozen response schema without extraction, repair,
or regeneration. The exact Exp6811 compiler and transition checker SHALL act
only after generation. Every model-scenario-seed-arm-candidate unit SHALL have
one row, including a parse-failure candidate slot. Rows SHALL retain all five
operational contract fields, parse state, hard and binding preservation,
already-safe identity, candidate diversity, legal support, and retry demand.

The producer SHALL append each model-scenario-arm cell to an atomic durable
checkpoint. Resume SHALL require a byte-identical frozen manifest. The field
`operational_handoff_corpus_ready` SHALL be true exactly when every planned row,
raw byte receipt, exact check, required model phase, and teardown receipt is
complete. Readiness SHALL be independent of the sign of an arm comparison.
The corpus is a development proxy and SHALL make no live ARC solve claim.

#### SCENARIO-CONSTRAINT-6812-FROZEN-PAIRS: Complete Length-Matched Prompt Pairs

**Given** the six required scenario families and matched generation budgets
**When** Exp6812 freezes its manifest before inference
**Then** it contains at least 48 source-free scenarios, both prompt byte
strings for every scenario, equal UTF-8 lengths within every pair, and all
planned model-scenario-seed-arm-candidate identities.

**Spec traces:** REQ-CONSTRAINT-6812

#### SCENARIO-CONSTRAINT-6812-EXACT-POSTCHECK: Raw Output Precedes Authority

**Given** one immutable local-model output for a frozen cell
**When** Exp6812 parses and evaluates its candidate slots
**Then** parsing performs no repair, the Exp6811 exact contract is applied only
after generation, and every row records parse state and operational evidence.

**Spec traces:** REQ-CONSTRAINT-6812

#### SCENARIO-CONSTRAINT-6812-RESUME-AND-READINESS: Exact Completion Is the Gate

**Given** an interrupted or completed sequential model phase
**When** Exp6812 resumes or computes its terminal readiness
**Then** resume accepts only the identical manifest and readiness requires all
planned rows, raw bytes, exact models, checks, and teardown receipts regardless
of measured effect sign.

**Spec traces:** REQ-CONSTRAINT-6812

## Implementation Status (REQ-CONSTRAINT-6812)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6812 and SCENARIO-CONSTRAINT-6812-* | Implemented (`python/carnot/experiment_6812_sota_operational_handoff_corpus_v2.py`; `results/experiment_6812_sota_operational_handoff_corpus_v2.json`). | Implemented (`tests/python/test_experiment_6812_sota_operational_handoff_corpus_v2.py`; 29 focused tests; 100% scoped statement coverage; three-model local llama.cpp CUDA E2E). |

### REQ-CONSTRAINT-6814: Independent Selective Arbiter Cold Audit

Exp6814 SHALL run in a fresh CPU process. It SHALL read the frozen Exp6811,
Exp6812, and Exp6813 artifacts without importing the Exp6813 producer. It SHALL
write `results/experiment_6814_selective_priority_arbiter_cold_audit.json`.

The audit SHALL check `selective_arbiter_ab_completed`, exact source hashes,
raw proposal bytes and hashes, complete source and comparison rows, matched
budgets, and frozen manifests before replay. A failed precondition SHALL write
`complete_blocked_selective_arbiter_cold_audit`. The blocked artifact SHALL
retain `gate_check_summary`, contain no cold rows, keep
`selective_arbiter_audit_completed` false, and stop before arbitration.

The audit SHALL decode the raw API byte streams and parse proposal bytes with
an independent parser. It SHALL rebuild the five-part obligation contracts,
authority order, prerequisite state, fallbacks, and transition outcomes from
the frozen scenario manifest. It SHALL independently replay the selective and
flat arms. It SHALL recompute all per-arm and paired headlines, confidence
intervals, false intervention, safe-action byte identity, certificates,
budgets, hard safety, and utility from cold rows. It SHALL compare each
producer headline with a frozen tight tolerance. Completion SHALL not depend
on the effect sign.

The audit SHALL inject priority inversion, stale prerequisite, wrong
authority, fallback deletion, consequence weakening, tie reorder, raw byte
mutation, model labels, exact-valid labels, utility, and future outcomes.
Proposal-time influence from a prohibited field SHALL disqualify the claim.
Every authority attack SHALL fail closed. Every rejection certificate SHALL
name the first unsatisfied higher-priority obligation. An already-valid base
proposal SHALL remain byte-identical and SHALL not count as an intervention.

The artifact SHALL contain `field_principles`, `inference_substrate`,
`duration_s`, `random_seed`, `reproducibility_checksum`,
`source_artifact_sha256s`, `independent_parser_id`, `independent_arbiter_id`,
`independent_reducer_id`, `rows`, `aggregate_recomputation`,
`headline_differences`, `budget_recomputation`, `priority_attack_results`,
`prohibited_feature_findings`, `false_intervention_recomputation`,
`certificate_findings`, `hard_safety_supported`, `utility_claim_supported`,
`source_verdict_supported`, `selective_arbiter_audit_completed`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. The verdict class SHALL be one of `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.

#### SCENARIO-CONSTRAINT-6814-PRECONDITIONS: Missing Frozen Evidence Stops Replay

**Given** a missing completion flag, raw byte receipt, source hash, comparison
row, budget match, or frozen manifest
**When** Exp6814 checks its inputs
**Then** it writes a complete blocked artifact with expected and observed
values and no cold rows.

**Spec traces:** REQ-CONSTRAINT-6814

#### SCENARIO-CONSTRAINT-6814-INDEPENDENT: Raw Bytes Own the Cold Replay

**Given** complete frozen artifacts
**When** Exp6814 runs its parser, arbiter, transition evaluator, and reducer
**Then** no Exp6813 implementation is imported and every headline derives
from one cold row per source unit and arm.

**Spec traces:** REQ-CONSTRAINT-6814

#### SCENARIO-CONSTRAINT-6814-AUTHORITY: Attacks Cannot Change Selection

**Given** an allowed proposal-time input and one injected prohibited feature
or contract mutation
**When** the independent arbiter selects or abstains
**Then** prohibited labels have no influence and each contract attack fails
closed with a local certificate.

**Spec traces:** REQ-CONSTRAINT-6814

#### SCENARIO-CONSTRAINT-6814-SAFE-NO-OP: Valid Bases Keep Exact Bytes

**Given** a base proposal that the independent evaluator classifies as valid
**When** both arms replay the frozen unit
**Then** false intervention uses the cold validity label and the selective arm
preserves the base action bytes exactly.

**Spec traces:** REQ-CONSTRAINT-6814

#### SCENARIO-CONSTRAINT-6814-AGGREGATION: Rows Own Every Claim

**Given** all cold replay and attack rows
**When** Exp6814 reduces safety, utility, budgets, intervals, and certificates
**Then** all artifact claims equal a fresh reduction and all producer
headlines are compared within the frozen tolerance.

**Spec traces:** REQ-CONSTRAINT-6814

#### SCENARIO-CONSTRAINT-6814-COMPLETION: Effect Sign Is Independent

**Given** a complete cold replay with valid authority and aggregation
**When** the paired utility effect is positive, null, or harmful
**Then** `selective_arbiter_audit_completed` remains true and
`source_verdict_supported` follows only the recomputed cold decision.

**Spec traces:** REQ-CONSTRAINT-6814

## Implementation Status (REQ-CONSTRAINT-6814)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6814 and SCENARIO-CONSTRAINT-6814-* | Planned: `python/carnot/experiment_6814_selective_priority_arbiter_cold_audit.py` and the task-owned script wrapper. | Planned focused tests and 100% scoped coverage. |

### REQ-CONSTRAINT-6824: Bounded Selective Arbiter Cold Row Replay

Exp6824 SHALL run a deterministic CPU replay without an LLM. It SHALL read
the frozen Exp6811, Exp6812, and Exp6813 artifacts. It SHALL not import the
Exp6813 producer, reducer, or verdict code. It SHALL write
`results/experiment_6824_selective_arbiter_cold_row_replay.json` through
`scripts/experiments/experiment_6824_selective_arbiter_cold_row_replay.py`.
Its independent modules SHALL be
`python/carnot/experiment_6824_cold_parser.py`,
`python/carnot/experiment_6824_cold_arbiter.py`,
`python/carnot/experiment_6824_cold_reducer.py`, and
`python/carnot/experiment_6824_selective_arbiter_cold_row_replay.py`.

The replay SHALL require readable raw-byte manifests, exactly 576 complete
source rows, valid source hashes, the frozen Exp6813 arm manifest, and
`selective_arbiter_ab_completed=true`. A failed precondition SHALL write
`complete_blocked_selective_arbiter_cold_row_replay`. The blocked artifact
SHALL contain no replay rows. Its `gate_check_summary` SHALL name each failed
check, expected value, and observed value. It SHALL then stop before replay.

A fresh parser SHALL decode each raw API byte stream and strict proposal JSON.
A fresh arbiter SHALL rebuild prerequisite, authority, fallback, consequence,
and priority checks from the frozen scenario manifest. It SHALL order
obligations by hard class, binding authority order, and soft value. It SHALL
preserve an already-valid proposal byte for byte. The two arms SHALL use the
same candidate, exact-check, outcome-check, retry, work-unit, and CPU budgets.

A fresh reducer SHALL join each pair by frozen row identity. It SHALL
recompute every Exp6813 row and every held per-arm metric. It SHALL also
recompute paired deltas, deterministic confidence intervals, false
intervention, safe-action identity, exact hard violations, legal support, and
certificate completeness. It SHALL compare all producer headlines with a
frozen numerical tolerance. Agreement SHALL not control completion.

The artifact SHALL contain one cold row for each of the 576 Exp6813 rows. It
SHALL also contain explicit deletion and duplicate-row audit cases. The audit
cases SHALL show that either fault stops aggregate reduction. The
`cold_replay_shard_complete` field SHALL depend on complete row coverage,
source hash identity, independent source modules, and complete recomputation.
It SHALL not depend on a positive, null, harmful, or contradictory result.

The artifact SHALL contain `field_principles`, `inference_substrate`,
`duration_s`, `random_seed`, `reproducibility_checksum`,
`source_artifact_hashes`, `independent_parser_id`, `independent_arbiter_id`,
`independent_reducer_id`, `rows`, `row_coverage`,
`aggregate_recomputation`, `headline_differences`, `budget_recomputation`,
`safe_action_identity_recomputation`, `hard_violation_recomputation`,
`false_intervention_recomputation`, `source_verdict_supported`,
`cold_replay_shard_complete`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. The verdict class SHALL be one of
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`.

#### SCENARIO-CONSTRAINT-6824-PRECONDITIONS: Missing Frozen Evidence Stops Replay

**Given** a missing raw manifest, source row, source hash, frozen arm manifest,
or producer completion flag
**When** Exp6824 checks its inputs
**Then** it writes a complete blocked artifact with expected and observed
values and no cold replay rows.

**Spec traces:** REQ-CONSTRAINT-6824

#### SCENARIO-CONSTRAINT-6824-PARSER: Raw Bytes Are Parsed Without Producer Code

**Given** one frozen raw API response and its proposal-byte receipt
**When** the independent parser decodes and parses the response
**Then** the exact proposal bytes and two strict candidate slots are rebuilt
without extraction, repair, coercion, or a producer import.

**Spec traces:** REQ-CONSTRAINT-6824

#### SCENARIO-CONSTRAINT-6824-ORDER-AND-IDENTITY: Authority Precedes Progress

**Given** hard, binding, and soft obligations plus an already-valid proposal
**When** the independent arbiter evaluates each arm
**Then** hard and authority-ordered binding checks precede soft value and the
selective arm preserves the safe action bytes exactly.

**Spec traces:** REQ-CONSTRAINT-6824

#### SCENARIO-CONSTRAINT-6824-BUDGETS-AND-JOINS: Paired Work Is Exact

**Given** complete frozen candidates and both comparison arms
**When** rows are joined and reduced
**Then** each pair has both arms once and all allowed work budgets match.

**Spec traces:** REQ-CONSTRAINT-6824

#### SCENARIO-CONSTRAINT-6824-INTERVALS: Paired Arithmetic Is Deterministic

**Given** held paired progress, retry, and false-intervention values
**When** the independent reducer uses the frozen audit seed
**Then** the estimates, bootstrap intervals, and Wilson upper bounds are
deterministic and are derived only from joined cold rows.

**Spec traces:** REQ-CONSTRAINT-6824

#### SCENARIO-CONSTRAINT-6824-ROW-FAULTS: Deletion And Duplication Fail Closed

**Given** one complete cold replay roster
**When** one row is deleted or one row identity is duplicated
**Then** aggregate reduction stops and the artifact retains one explicit audit
case for each fault.

**Spec traces:** REQ-CONSTRAINT-6824

#### SCENARIO-CONSTRAINT-6824-AGGREGATION: Cold Rows Own Every Headline

**Given** all 576 recomputed rows and the two row-fault audit cases
**When** Exp6824 reduces producer headlines
**Then** each metric, interval, budget receipt, and headline difference is a
fresh row-derived computation and completion remains independent of agreement.

**Spec traces:** REQ-CONSTRAINT-6824

## Implementation Status (REQ-CONSTRAINT-6824)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6824 and SCENARIO-CONSTRAINT-6824-* | Planned: independent parser, arbiter, reducer, orchestrator, and task-owned script. | Planned focused tests and 100% scoped coverage. |

### REQ-CONSTRAINT-6825: Independent Selective-Arbiter Authority Attacks

Exp6825 SHALL run a fresh-process deterministic CPU mutation audit without an
LLM. It SHALL read the frozen Exp6811 obligation schema, the Exp6812 raw source
rows, and the Exp6813 completion receipt. It SHALL not import the Exp6813
producer or reducer. It SHALL write
`results/experiment_6825_selective_arbiter_authority_attacks.json` through
`scripts/experiments/experiment_6825_selective_arbiter_authority_attacks.py`.
The independent implementation SHALL be
`python/carnot/experiment_6825_selective_arbiter_authority_attacks.py`.

The audit SHALL require the exact frozen obligation schema, matching source
artifact hashes, one valid raw representative cell for every frozen source
scenario, and `selective_arbiter_ab_completed=true`. A failed precondition
SHALL write `complete_blocked_selective_arbiter_authority_attacks`. The blocked
artifact SHALL contain no attack rows. Its `gate_check_summary` SHALL name each
failed check with the expected and observed values. It SHALL stop before the
mutation audit.

The independent harness SHALL rebuild obligation records and the hard,
binding, then soft priority order from the public contract. Source outputs
SHALL identify audit cases, but producer decisions SHALL not define expected
selections. The allowlist SHALL contain proposal-time public fields only. The
denylist SHALL contain model labels, exact-valid labels, outcomes, utility,
and future fields.

Every frozen source scenario SHALL receive priority inversion, authority
spoofing, stale prerequisite, fallback deletion, consequence weakening, tie
reorder, canonical-byte mutation, no-candidate, safe-action mutation,
model-label influence, exact-valid-label influence, future-outcome leakage,
row deletion, duplicate-row, row reorder, and fabricated-certificate attacks.
The artifact SHALL contain one row for every source-scenario and attack pair.
An attack that does not apply to one scenario SHALL retain an explicit row and
SHALL not replace the applicable positive-control rows for that attack.

Hard and binding violations SHALL fail closed. A safe proposal SHALL remain
byte-identical unless a higher-priority obligation proves intervention is
necessary. Rejection certificates SHALL name the first unsatisfied
higher-priority obligation. Deleted, duplicated, or reordered source-case
rosters SHALL fail integrity validation. Denied feature injection SHALL not
change a selection.

`hard_authority_supported` SHALL report the row-supported authority finding.
It SHALL remain separate from adoption. `authority_attack_shard_complete`
SHALL depend only on full source-by-attack row coverage, complete applicable
attack coverage, independent code identity, source seals, and a byte-identical
fresh-process replay. It SHALL not depend on whether the authority finding
passes or fails. Exp6825 SHALL make no deployment or adoption decision.

The artifact SHALL contain `field_principles`, `inference_substrate`,
`duration_s`, `random_seed`, `reproducibility_checksum`,
`source_artifact_hashes`, `independent_attack_harness_id`,
`feature_allowlist`, `feature_denylist`, `rows`,
`priority_attack_results`, `safe_action_attack_results`,
`certificate_attack_results`, `prohibited_feature_findings`,
`row_integrity_attacks`, `hard_authority_supported`,
`authority_attack_shard_complete`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`. The verdict class
SHALL be one of `positive`, `circular_positive`, `null`, `blocked`,
`disqualified`, or `partial`.

#### SCENARIO-CONSTRAINT-6825-PRECONDITIONS: Missing Frozen Evidence Stops Attacks

**Given** a missing obligation schema, source seal, representative raw row, or
producer completion flag
**When** Exp6825 checks its inputs
**Then** it writes a complete blocked artifact with expected and observed
values, no attack rows, and a false shard-completion field.

**Spec traces:** REQ-CONSTRAINT-6825

#### SCENARIO-CONSTRAINT-6825-PRIORITY: Hard And Binding Authority Fail Closed

**Given** a high-soft candidate that violates a hard or authority-ordered
binding obligation
**When** priority, authority, prerequisite, fallback, or consequence data is
mutated
**Then** the harness rejects the unsafe choice or rejects the changed contract
before selection.

**Spec traces:** REQ-CONSTRAINT-6825

#### SCENARIO-CONSTRAINT-6825-SAFE-BYTES: Safe Actions And Stable Ties Are Exact

**Given** an already-safe proposal or equal legal candidates
**When** output bytes or candidate list order is changed
**Then** the safe proposal keeps its exact canonical bytes and the frozen lower
candidate index wins the tie.

**Spec traces:** REQ-CONSTRAINT-6825

#### SCENARIO-CONSTRAINT-6825-CERTIFICATES: Local Conflict Truth Is Recomputed

**Given** a rejected candidate or a no-candidate result
**When** its certificate is generated or fabricated
**Then** the valid certificate names the first local higher-priority conflict
and the fabricated certificate fails validation.

**Spec traces:** REQ-CONSTRAINT-6825

#### SCENARIO-CONSTRAINT-6825-FEATURES: Prohibited Labels Have No Influence

**Given** the same proposal-time public fields and counterfactually swapped
model, exact-valid, or future-outcome labels
**When** the independent harness selects an action
**Then** the selected candidate and selected bytes stay unchanged.

**Spec traces:** REQ-CONSTRAINT-6825

#### SCENARIO-CONSTRAINT-6825-ROW-INTEGRITY: Roster Mutations Fail Closed

**Given** the frozen ordered source-case roster
**When** one source case is deleted, duplicated, or reordered
**Then** roster validation fails and names the integrity fault.

**Spec traces:** REQ-CONSTRAINT-6825

#### SCENARIO-CONSTRAINT-6825-COMPLETION: Coverage Is Independent Of Findings

**Given** every source-case and mutation row plus a byte-identical fresh-process
replay
**When** the harness reduces pass or fail findings
**Then** `authority_attack_shard_complete` is true for complete evidence and
`hard_authority_supported` separately reports the finding.

**Spec traces:** REQ-CONSTRAINT-6825

## Implementation Status (REQ-CONSTRAINT-6825)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6825 and SCENARIO-CONSTRAINT-6825-* | Implemented (`python/carnot/experiment_6825_selective_arbiter_authority_attacks.py`; task-owned script and terminal artifact). | Implemented (`tests/python/test_experiment_6825_selective_arbiter_authority_attacks.py`; 30 focused tests; 100% scoped statement coverage; 768 case-by-attack rows; byte-identical fresh-process replay). |

### REQ-CONSTRAINT-6826: Sealed Selective-Arbiter Adoption Receipt

Exp6826 SHALL run a deterministic CPU synthesis without an LLM. It SHALL read
the fixed Exp6813 producer artifact and the independent Exp6824 and Exp6825
audit shards. It SHALL not import Exp6813 code. It SHALL write
`results/experiment_6826_selective_arbiter_sealed_adoption.json` through
`scripts/experiments/experiment_6826_selective_arbiter_sealed_adoption.py`.
The implementation SHALL be
`python/carnot/experiment_6826_selective_arbiter_sealed_adoption.py`.

The synthesis SHALL require both shard completion fields, exact row manifests,
matching source hashes, valid independent code identities, and verdict classes
from the closed enum. A failed precondition SHALL write
`complete_blocked_selective_arbiter_sealed_adoption`. The blocked artifact
SHALL contain no decision rows. Its `gate_check_summary` SHALL name each failed
check with the expected and observed values. It SHALL then stop before decision
synthesis.

The frozen decision table SHALL preserve the most conservative input result.
Its input-result order SHALL be disqualified, blocked, partial, harmful, null,
then positive. Disqualified evidence SHALL select redesign. Blocked or partial
evidence SHALL select insufficient. Harmful utility SHALL select retire. Null
utility SHALL select keep-shadow. Only complete positive evidence can select
enable. These outcome rules SHALL apply to every ordered pair of shard-result
classes.

The synthesis SHALL recompute hard safety, safe-action preservation, utility,
and certificate truth from shard rows. Each component decision SHALL be one of
`pass`, `fail`, or `insufficient`. Hard safety SHALL not imply utility. Utility
SHALL pass only on a positive paired lower bound without harmful-selection or
legal-support regression. A confidence interval that contains zero SHALL be
insufficient. A negative upper bound or adverse utility row SHALL fail.

Deployment adoption SHALL be disqualified from enablement when any prohibited
proposal-time feature influences selection, any accepted hard violation exists,
any source row is missing or duplicated, or producer and cold arithmetic differ
beyond the frozen tolerance. A disqualifier SHALL produce `redesign`. Complete
safety and certificate evidence with null utility SHALL produce `keep_shadow`.
Complete safety and certificate evidence with harmful utility SHALL produce
`retire`. Missing component evidence SHALL produce `insufficient`. Only four
component passes with no disqualifier SHALL produce `enable`.

The artifact SHALL contain one row for each criterion and evidence source.
`selective_arbiter_audit_complete` SHALL depend on a terminal component decision
table, exact row coverage, source seals, and independent evidence. It SHALL not
depend on a favorable utility sign or deployment adoption outcome. Exp6826 SHALL
not change the live production default.

The artifact SHALL contain `field_principles`, `inference_substrate`,
`duration_s`, `random_seed`, `reproducibility_checksum`,
`source_artifact_hashes`, `rows`, `adoption_criteria`,
`hard_safety_decision`, `safe_action_preservation_decision`,
`utility_decision`, `certificate_truth_decision`,
`deployment_adoption_decision`, `selective_arbiter_audit_complete`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. Each top-level field SHALL have one entry in
`field_principles`. The deployment decision SHALL be one of `enable`,
`keep_shadow`, `redesign`, `retire`, or `insufficient`. The verdict class SHALL
be one of `positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`.

#### SCENARIO-CONSTRAINT-6826-PRECONDITIONS: An Unsealed Shard Stops Synthesis

**Given** a missing completion field, row identity, source seal, independent
code identity, or closed verdict class
**When** Exp6826 checks its fixed inputs
**Then** it writes a complete blocked artifact with expected and observed
values, no decision rows, and a false audit-completion field.

**Spec traces:** REQ-CONSTRAINT-6826

#### SCENARIO-CONSTRAINT-6826-MATRIX: Conservative Outcomes Are Total

**Given** any ordered pair of positive, null, harmful, blocked, disqualified,
or partial shard results
**When** the sealed outcome table resolves deployment authority
**Then** the most conservative result controls and all 36 pairs have a closed
deployment decision and verdict class.

**Spec traces:** REQ-CONSTRAINT-6826

#### SCENARIO-CONSTRAINT-6826-CRITERIA: Components Stay Separate

**Given** complete cold replay and authority-attack rows
**When** Exp6826 recomputes the frozen criteria
**Then** hard safety, safe-action preservation, utility, and certificate truth
each receive a separate pass, fail, or insufficient decision.

**Spec traces:** REQ-CONSTRAINT-6826

#### SCENARIO-CONSTRAINT-6826-DISQUALIFIERS: Deployment Fails Closed

**Given** prohibited feature influence, an accepted hard violation, an
incomplete source roster, or arithmetic disagreement beyond tolerance
**When** Exp6826 issues deployment authority
**Then** deployment adoption is redesign and no safety result becomes a utility
claim.

**Spec traces:** REQ-CONSTRAINT-6826

#### SCENARIO-CONSTRAINT-6826-COMPLETION: Effect Sign Does Not Control Audit Closure

**Given** a complete terminal decision table with positive, null, harmful, or
disqualified findings
**When** Exp6826 closes the sealed audit
**Then** `selective_arbiter_audit_complete` is true and the adoption outcome is
reported separately.

**Spec traces:** REQ-CONSTRAINT-6826

#### SCENARIO-CONSTRAINT-6826-ARTIFACT: Every Decision Has Row Evidence

**Given** a terminal synthesis
**When** Exp6826 writes its task-owned artifact
**Then** each criterion-source unit is present once, each top-level field has a
principle, and the checksum binds inputs, criteria, rows, commands, and output.

**Spec traces:** REQ-CONSTRAINT-6826

## Implementation Status (REQ-CONSTRAINT-6826)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6826 and SCENARIO-CONSTRAINT-6826-* | Planned: sealed row-based decision synthesis and task-owned script. | Planned focused tests and 100% scoped statement coverage. |

## REQ-CONSTRAINT-6831: V597 Selective-Arbiter Evidence Admissibility

The system SHALL issue a fresh evidence-admissibility contract from readable,
terminal Exp6813 and Exp6824 through Exp6827 artifacts whose bytes remain
stable throughout the run. The task SHALL verify complete unique source rows,
closed verdict classes, embedded source seals, and required completion fields
before reducing any evidence. A failed precondition SHALL stop the reduction
and write `complete_blocked_v597_evidence_admissibility` with an exact failed
check, expected value, and observed value in `gate_check_summary`.

The task SHALL independently reduce Exp6813, Exp6824, and Exp6825 data into
separate hard-safety, safe-action-identity, certificate-truth, utility, and
adoption decisions. The reduction SHALL use a closed conservative decision
table for positive, null, harmful, blocked, disqualified, and partial evidence.
Exp6826 SHALL remain a quarantined comparator and SHALL never supply decision
logic or deployment authority. Procedural receipt admissibility SHALL depend on
complete admissible authority evidence, not on a positive utility effect.

The terminal artifact SHALL declare only
`aggregation_from_upstream_artifacts` as its inference substrate. It SHALL
record task-owned start, read, recompute, write, and verify clocks, process and
command launch identity, accelerator samples, stable source hashes, criterion
rows, a reproducibility checksum, one principle for every declared field, and
`verifier_is_oracle=false`.

### SCENARIO-CONSTRAINT-6831-PRECONDITIONS: Missing Or Incomplete Evidence Stops

Given a required source is unreadable, nonterminal, incomplete, or missing a
required source row,
When Exp6831 checks its inputs,
Then it SHALL write the complete blocked artifact and SHALL not reduce rows.

### SCENARIO-CONSTRAINT-6831-HASH-STABILITY: Source Drift Stops

Given a source artifact changes between the initial read and final verification,
When Exp6831 compares task-owned source hashes,
Then it SHALL report the drift as the blocking gate and SHALL issue no authority.

### SCENARIO-CONSTRAINT-6831-FRESH-REDUCTION: Authority Comes From Source Data

Given complete Exp6813, Exp6824, and Exp6825 rows,
When Exp6831 recomputes the authority table,
Then safety, identity, certificate, utility, and adoption SHALL be separate
row-supported decisions and SHALL not import the Exp6826 decision procedure.

### SCENARIO-CONSTRAINT-6831-CONSERVATIVE-TABLE: Findings Propagate Fail Closed

Given any combination of positive, null, harmful, blocked, disqualified, and
partial component findings,
When Exp6831 derives adoption,
Then disqualified SHALL require redesign, harmful SHALL retire the candidate,
blocked or partial SHALL remain insufficient, null SHALL remain shadow-only,
and only all-positive evidence SHALL enable adoption.

### SCENARIO-CONSTRAINT-6831-QUARANTINE: Flagged Receipt Has No Authority

Given Exp6826 carries an adversarial duration flag,
When Exp6831 records it as a comparator,
Then its disposition SHALL remain quarantined and its decisions SHALL not be
consumed as selective-arbiter authority.

### SCENARIO-CONSTRAINT-6831-CLOCKS: Task-Owned Phases Are Ordered

Given one Exp6831 execution,
When it records start, read, recompute, write, and verify phases,
Then every phase SHALL carry an ordered task-owned clock and `duration_s` SHALL
cover the measured task wall time.

### SCENARIO-CONSTRAINT-6831-ADMISSIBILITY: Effect Sign Does Not Grant Authority

Given a complete authority audit with any utility effect sign,
When Exp6831 derives `selective_arbiter_receipt_admissible`,
Then the field SHALL reflect procedural completeness and authority admissibility
rather than the presence of a positive utility effect.

## Implementation Status (REQ-CONSTRAINT-6831)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6831 and SCENARIO-CONSTRAINT-6831-* | Implemented: fresh evidence reducer, task-owned wrapper, and terminal artifact contract. | Implemented: 43 focused tests pass and the 482-statement reducer has 100% scoped coverage. |

## REQ-CONSTRAINT-6832: Frozen Operational-Obligation Saturation Fixture

The system SHALL build a source-free deterministic fixture at simultaneous
obligation counts 1, 2, 4, 6, and 8. Each count SHALL contain exactly 30
scenarios. Fifteen scenarios SHALL use independent obligations. Fifteen
scenarios SHALL use interacting obligations. At count 1, an interacting case
means that the obligation fields interact with candidate or state data. Counts
above 1 SHALL also include interactions between obligations.

Each scenario SHALL use the Exp6811 prerequisite, authority, fallback,
execution-consequence, and priority fields. It SHALL have immutable scenario,
template, permutation, obligation, action, and exact legal-action identifiers.
Every scenario SHALL have one unique exact action set. The fixture SHALL cover
constructive actions, safe no-op actions, and intentional impossibility with a
fail-closed action.

Each semantic template SHALL have three deterministic prompt permutations.
Every permutation SHALL preserve the same source information and exact action
set. Each permutation SHALL have direct typed-handoff and compressed-prose
arms. The arms SHALL use the same output schema. Prompts SHALL contain no
answer key, hidden outcome, dependency label, checker term, or order cue.

The fixture SHALL parse exact JSON without extraction or repair. It SHALL
provide deterministic prerequisite, authority, fallback,
execution-consequence, and priority checks for each obligation. It SHALL also
provide one joint all-obligation check. The producer SHALL validate these
checks against legal, violation, omission, conflict, and reorder candidates
for every scenario.

Before generation, the producer SHALL require `v597_contract_ready=true` in
the frozen Exp6831 artifact. It SHALL require the exact frozen Exp6811 file
hash, schema, source hashes, obligation schema, automaton hash, and canonical
obligation-source hash. It SHALL require a clean worktree for the consumed
Exp6811 and Exp6831 source artifacts and the Exp6811 automaton source. A failed
check SHALL write
`complete_blocked_operational_obligation_saturation_fixture`. The blocked
artifact SHALL name the failed check, expected value, and observed value in
`gate_check_summary`, then stop before scenario generation.

The terminal artifact SHALL include `field_principles`,
`preconditions_checked`, `inference_substrate`, `duration_s`, `random_seed`,
`reproducibility_checksum`, `source_artifact_hashes`, `obligation_schema`,
`obligation_counts`, `scenario_manifest`, `prompt_arm_manifest`, `scenarios`,
`checker_manifest`, `checker_mutation_results`, `leakage_audit`,
`legal_action_headroom`, `operational_saturation_fixture_ready`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. One principle SHALL exist for every top-level field. The
reproducibility checksum SHALL bind the schema, owned code, scenarios, and
output while excluding measured duration. The inference substrate SHALL state
that fixture generation uses deterministic CPU code and no LLM.

### SCENARIO-CONSTRAINT-6832-PRECONDITIONS: Source Drift Stops Generation

Given a missing V597 gate, changed Exp6811 identity, or dirty consumed source,
When Exp6832 checks its inputs,
Then it SHALL write the complete blocked artifact with the exact failed gate
and SHALL emit no scenarios or checker results.

### SCENARIO-CONSTRAINT-6832-BALANCE: Counts And Dependency Cells Are Exact

Given obligation counts 1, 2, 4, 6, and 8,
When Exp6832 freezes its scenario manifest,
Then each count SHALL contain 30 identities with a 15/15 split between
independent and interacting obligations.

### SCENARIO-CONSTRAINT-6832-TYPED-SEMANTICS: Every Field Has Exact Meaning

Given one frozen scenario,
When its action set is derived,
Then every obligation SHALL consume all five typed contract fields and SHALL
resolve to a constructive, no-op, preempted fallback, or fail-closed action.

### SCENARIO-CONSTRAINT-6832-PERMUTATION: Order Does Not Change Truth

Given the three permutations of one semantic template,
When prompts and candidate responses are reordered,
Then both prompt arms SHALL retain equal information and all check results and
the exact action set SHALL remain unchanged.

### SCENARIO-CONSTRAINT-6832-UNIQUE-SOLUTION: Exact Sets Have No Ambiguity

Given every satisfiable, intentionally impossible, or already-safe scenario,
When the deterministic resolver evaluates its candidates,
Then exactly one action set SHALL pass the joint check and it SHALL be nonempty.

### SCENARIO-CONSTRAINT-6832-SAFE-FAIL-CLOSED: Boundary Cases Stay Useful

Given an already-safe scenario or an intentionally impossible scenario,
When the exact resolver runs,
Then the safe case SHALL select only its no-op fallbacks and the impossible
case SHALL select only its scenario fail-closed action.

### SCENARIO-CONSTRAINT-6832-CHECKERS: Field And Joint Checks Reject Mutations

Given legal, violation, omission, conflict, and reorder candidates for every
scenario,
When the field and joint checks run,
Then legal and reorder candidates SHALL pass and the other candidates SHALL
fail without repair.

### SCENARIO-CONSTRAINT-6832-LEAKAGE: Prompts Expose Only Task Information

Given both prompt arms for every permutation,
When the leakage audit scans their bytes,
Then no prompt SHALL expose an exact action set, fixture-only label, hidden
outcome, answer key, checker term, or order cue.

### SCENARIO-CONSTRAINT-6832-READINESS: Fixture Quality Controls Readiness

Given all frozen scenarios and checker mutations,
When Exp6832 computes readiness,
Then `operational_saturation_fixture_ready` SHALL depend only on source gates,
exact count balance, equal information, leakage, checker completeness,
nonzero required headroom, and stable hashes. It SHALL not depend on model
output or an expected model result.

## Implementation Status (REQ-CONSTRAINT-6832)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6832 and SCENARIO-CONSTRAINT-6832-* | Implemented: deterministic CPU fixture producer and task-owned wrapper. | Implemented: focused tests cover schema, balance, semantics, permutations, exact sets, safe and impossible cases, checks, leakage, readiness, and blocked gates. |

## REQ-CONSTRAINT-6833: Live SOTA Operational-Obligation Saturation Corpus

The system SHALL run every frozen Exp6832 scenario once for each prompt arm
and each mandated model. The mandated models SHALL be
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. A complete corpus SHALL contain exactly
900 unique model, scenario, and arm identities.

Before inference, the producer SHALL call `cached_sota_pair()`. It SHALL
require the three exact GGUF files, frozen file hashes, embedded tokenizer
metadata, embedded chat templates, a CUDA device, an exclusive task-owned GPU
lease, free task-owned ports, sufficient disk, and
`operational_saturation_fixture_ready=true`. It SHALL run a one-token live
CUDA canary for each model. Any failed check SHALL write
`complete_blocked_sota_operational_saturation_corpus`. The blocked artifact
SHALL record the exact expected and observed values in `gate_check_summary`.
The producer SHALL not use a smaller model, cached output, simulation, or an
API substitute.

The producer SHALL start one task-owned llama.cpp CUDA process at a time unless
independent GPU leases prove safe concurrency. It SHALL record the command,
PID, process start time, port, physical GPU UUID, visible devices, first token,
final token, and clean teardown for each model. It SHALL not attach to an
unrelated process. Each model phase SHALL use a separate process and lease.

Both prompt arms SHALL use the same frozen temperature, seed, stop rules,
context size, and maximum output tokens. The producer SHALL disable repair,
content-changing retries, grammar constraints, and answer feedback. It SHALL
capture the GGUF tokenizer metadata and chat template used by llama.cpp.

The producer SHALL preserve prompt bytes, raw output bytes, parsed fields,
parse status, each exact obligation result, the exact joint result, token
counts, latency, process identity, model identity, and exact checker hash for
every row. Exp6832 field checkers and joint checker SHALL be the only scoring
authority. The producer SHALL not use an LLM judge or producer aggregate as
authority.

The producer SHALL checkpoint after bounded prompt batches and after each
model. A restart SHALL validate the manifest and stored row hashes. It SHALL
continue only missing identities and SHALL never regenerate a completed row.
Rows from one model process SHALL not be attributed to another model process.

The terminal artifact SHALL include `field_principles`,
`preconditions_checked`, `inference_substrate`, `duration_s`, `phase_clocks`,
`random_seed`, `reproducibility_checksum`, `MODEL_SPECS`, `process_receipts`,
`accelerator_samples`, `checkpoint_manifest`, `per_unit_rows`, `row_coverage`,
`budget_parity`, `exact_scores`, `descriptive_aggregates`,
`operational_saturation_corpus_ready`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`. One principle
SHALL exist for every top-level field. The reproducibility checksum SHALL bind
the fixture, model files, rows, code, commands, and raw output. The artifact
SHALL set `inference_substrate` to `live_llm_inference` and
`verifier_is_oracle` to false.

`operational_saturation_corpus_ready` SHALL depend on exactly 900 complete
unique rows, authentic process receipts, equal budgets, exact scoring, valid
checkpoints, and clean teardown. Accuracy SHALL not control readiness. The
artifact SHALL report descriptive aggregates only because Exp6834 owns the
inferential audit. `verdict_class` SHALL be one of `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
`honest_verdict` SHALL be terminal and start with `complete_`.

### SCENARIO-CONSTRAINT-6833-PREFLIGHT: Failed Live Gates Stop Inference

Given any missing model, hash, tokenizer, template, CUDA, lease, port, disk,
fixture, or canary gate,
When Exp6833 runs preflight,
Then it SHALL write the complete blocked artifact with exact observed values
and SHALL start no corpus generation.

### SCENARIO-CONSTRAINT-6833-DISPATCH: Each Model Owns Its Process

Given the three mandated model specifications,
When Exp6833 runs sequential model phases,
Then every row SHALL name the exact model file and owning process and no model
SHALL reuse another model's process identity.

### SCENARIO-CONSTRAINT-6833-BUDGET: Prompt Arms Have Equal Decode Budgets

Given both Exp6832 prompt arms,
When the producer sends their requests,
Then temperature, seed, stop rules, context size, and maximum output tokens
SHALL be equal and no repair or answer feedback SHALL occur.

### SCENARIO-CONSTRAINT-6833-CHECKPOINT: Restart Preserves Completed Rows

Given a durable checkpoint with valid completed row hashes,
When a model phase restarts,
Then it SHALL skip completed identities and generate only missing identities.
The checkpoint SHALL also retain each process receipt before it publishes the
first row. It SHALL update the receipt after each bounded batch. If the task
stops between batches, restart SHALL recover only from checksummed task-owned
lease and server-log evidence. It SHALL fail closed when that evidence is
missing or does not match the retained row process identity.

### SCENARIO-CONSTRAINT-6833-RAW-BYTES: Evidence Is Byte Exact

Given one live completion,
When Exp6833 stores its row,
Then the row SHALL retain prompt and raw output bytes with matching lengths and
SHA-256 hashes.

### SCENARIO-CONSTRAINT-6833-SCORING: Exp6832 Checkers Are Authority

Given any parseable or unparseable model output,
When Exp6833 scores the row,
Then it SHALL preserve parse status and every Exp6832 obligation and joint
result without repair or LLM judgment.

### SCENARIO-CONSTRAINT-6833-TEARDOWN: Owned Servers Exit Cleanly

Given a task-owned llama.cpp process,
When its model phase ends or fails,
Then Exp6833 SHALL stop only that process, confirm its absence, release its GPU
lease, and record final-token and teardown receipts.

### SCENARIO-CONSTRAINT-6833-READINESS: Completeness Controls Readiness

Given all expected model, scenario, and arm identities,
When Exp6833 computes the terminal gate,
Then readiness SHALL require 900 valid rows and authentic execution evidence
and SHALL not depend on model accuracy.

## Implementation Status (REQ-CONSTRAINT-6833)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6833 and SCENARIO-CONSTRAINT-6833-* | Planned: live local llama.cpp CUDA corpus producer and task-owned wrapper. | Planned: focused tests and the full local CUDA run. |


## REQ-CONSTRAINT-6834: Cold Operational-Saturation Identifiability Audit

The system SHALL provide a deterministic CPU audit that reads the frozen
Exp6832 fixture and Exp6833 corpus without calling an LLM. Before reduction,
the audit SHALL require `operational_saturation_corpus_ready=true`, exactly 900
unique model, scenario, and arm rows, complete raw-output byte receipts, frozen
fixture and model hashes, equal decode budgets, authentic task-owned process
receipts, and complete teardown. A failed check SHALL write
`complete_blocked_operational_saturation_identifiability_audit`. The blocked
artifact SHALL name the first failed check, expected value, and observed value
in `gate_check_summary`, then stop before row reduction.

The audit SHALL use a fresh strict JSON parser and an independent
reimplementation of the Exp6832 prerequisite, authority, fallback,
execution-consequence, priority, and exact joint contracts. The audit module
SHALL not import Exp6833 scoring or aggregate code. It SHALL decode every raw
output receipt, check its length and SHA-256, parse without repair, and
recompute every available obligation field and joint decision. Producer scores
and aggregates SHALL not provide audit authority.

The reducer SHALL report field success by model, prompt arm, obligation count,
obligation type, and interaction class. It SHALL report joint success and
parse failure as separate outcomes. It SHALL use scenario-matched arm pairs
and scenario-level bootstrap units with a frozen seed. Headline saturation
curves SHALL remain separate for each model family. Saturation means declining
joint success as obligation count increases. Interaction penalties SHALL be
reported separately from count decay.

The audit SHALL define the finite target policy as all operational field
decisions for every model, scenario, arm, obligation, and field cell. A parsed
row observes its field cells. A transport failure does not observe latent field
preservation. The audit SHALL enumerate the compatible binary policy class
symbolically, compute a minimum identifying support certificate, and report
constructive policy collisions when the current support is not identifying.
Each collision SHALL name two policies with the same observed signature and a
different target field value. It SHALL also name the smallest added cell that
separates that pair.

The audit SHALL run deletion, duplicate, row-order, model-label, prompt-label,
and checker-mutation attacks. Deletion and duplicate attacks SHALL reject bad
coverage. Order and label permutations SHALL preserve invariant results after
canonical relabeling. Model-label masking SHALL detect loss of unpooled model
identity. Checker mutation SHALL change or invalidate at least one recomputed
decision. `operational_saturation_audit_complete` SHALL depend on complete cold
recomputation, all required attacks, and an explicit identifiability
disposition. It SHALL not depend on the sign of an arm effect or decay.

The terminal artifact SHALL include `field_principles`,
`preconditions_checked`, `inference_substrate`, `duration_s`, `random_seed`,
`reproducibility_checksum`, `source_artifact_hashes`,
`independent_parser_id`, `independent_reducer_id`, `per_unit_rows`,
`row_coverage`, `per_obligation_metrics`, `joint_success_metrics`,
`parse_failure_metrics`, `interaction_penalties`, `saturation_curves`,
`paired_arm_effects`, `identifiability_result`,
`minimum_identifying_support`, `collision_witnesses`, `attack_results`,
`operational_saturation_audit_complete`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`. One principle
SHALL exist for every top-level field. The inference substrate SHALL declare a
fresh-process deterministic CPU audit with no LLM. The reproducibility checksum
SHALL bind both source files, the independent module and wrapper, every source
and attack row, the frozen commands, and all deterministic output while
excluding measured duration. `verifier_is_oracle` SHALL be false.
`verdict_class` SHALL be one of `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL be terminal and
start with `complete_`.

### SCENARIO-CONSTRAINT-6834-PRECONDITIONS: Invalid Corpus Stops Reduction

Given a missing readiness gate, row, byte receipt, hash, budget receipt,
process receipt, or teardown receipt,
When Exp6834 checks its frozen inputs,
Then it SHALL write the complete blocked artifact with the exact failed gate
and SHALL emit no reduced source rows.

### SCENARIO-CONSTRAINT-6834-INDEPENDENT-TRUTH: Raw Bytes Control Decisions

Given all 900 raw-output receipts and the Exp6832 public contract,
When Exp6834 decodes, parses, and checks them,
Then every parse, field, and joint result SHALL come from fresh code and SHALL
not consume Exp6833 scores or aggregates as authority.

### SCENARIO-CONSTRAINT-6834-MATCHED-INFERENCE: Units Stay Paired And Unpooled

Given typed and compressed rows for each model and scenario,
When Exp6834 computes effects and intervals,
Then arm effects SHALL use matched scenario units, bootstrap resampling SHALL
resample scenarios, count order SHALL be 1, 2, 4, 6, and 8, and model-family
headline curves SHALL remain separate.

### SCENARIO-CONSTRAINT-6834-SATURATION: Decay Is Not Parse Failure

Given recomputed joint decisions and transport outcomes,
When Exp6834 reports saturation,
Then joint-success decay SHALL use all rows, parse failure SHALL have a separate
metric, and interaction penalties SHALL compare matched interaction classes at
fixed model, arm, and count.

### SCENARIO-CONSTRAINT-6834-IDENTIFIABILITY: Missing Field Support Emits Collisions

Given the finite model, scenario, arm, obligation, and field target cells,
When Exp6834 groups binary behavior policies by their observed field signature,
Then it SHALL report whether support identifies all target cells, a minimum
support certificate, every missing-cell class, constructive collisions, and a
one-cell separator for each reported witness.

### SCENARIO-CONSTRAINT-6834-ATTACKS: Audit Invariants Fail Closed

Given deletion, duplication, order, model-label, prompt-label, and checker
mutations,
When Exp6834 re-runs its coverage and reduction invariants,
Then missing or duplicate identities and masked model labels SHALL be detected,
order and semantic label permutations SHALL remain invariant after canonical
mapping, and checker mutations SHALL not preserve the complete result.

### SCENARIO-CONSTRAINT-6834-COMPLETENESS: Effect Sign Does Not Control Completion

Given a cold recomputation with any measured effect sign and either an
identifying or non-identifying support disposition,
When Exp6834 computes its terminal completion gate,
Then `operational_saturation_audit_complete` SHALL depend on procedural
completeness and the explicit disposition only.

## Implementation Status (REQ-CONSTRAINT-6834)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6834 and SCENARIO-CONSTRAINT-6834-* | Planned: independent parser, reducer, finite support audit, and task-owned wrapper. | Planned: focused tests, scoped 100% coverage, and cold artifact verification. |


## REQ-CONSTRAINT-6835: V598 Terminal Evidence Freeze

The system SHALL build Exp6835 as a deterministic CPU evidence root for V598.
It SHALL read the terminal Exp6831, Exp6832, Exp6833, and Exp6834 artifacts.
Before any row reduction, it SHALL require all four artifacts to be readable,
their SHA-256 hashes to match the frozen source hashes, Exp6833 to contain
exactly 900 unique model, scenario, and arm row identities, Exp6832 to contain
exactly 150 unique scenario identities, Exp6833 to be corpus-ready, Exp6834 to
be audit-complete, and owned source files to match their recorded hashes. A
failed check SHALL write
`complete_blocked_v598_terminal_evidence_freeze`. The blocked artifact SHALL
record the failed check and observed value in `gate_check_summary`.

The freeze SHALL reparse every Exp6833 raw output receipt with a fresh
deterministic parser. It SHALL not import the Exp6833 or Exp6834 reduction
path. It SHALL classify one row for every model, prompt arm, scenario, and
obligation atom. Each row SHALL separate protocol failure, omission,
contradiction, atom pass, and joint pass. Malformed output SHALL be a protocol
failure. A parseable output that omits the expected atom SHALL be an omission.
A parseable output that selects a conflicting action for the same resource
SHALL be a contradiction. Atom pass SHALL remain separate from joint pass.

The freeze SHALL recompute target-cell accounting from the source artifacts.
It SHALL report observed target cells, missing target cells, collision
witnesses, and a compatible-policy lower bound. It SHALL preserve the Exp6834
null when generated-answer semantic preservation remains unidentified. It
SHALL set `v598_evidence_root_ready_score` to `1` only when the evidence root
is complete and immutable. This readiness field is not a positive scientific
result.

The terminal artifact SHALL be
`results/experiment_6835_v598_terminal_evidence_freeze.json`. It SHALL include
`field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `reproducibility_checksum`, `rows`,
`failure_taxonomy`, `observed_target_cells`, `missing_target_cells`,
`collision_witnesses`, `compatible_policy_lower_bound`,
`source_null_preserved`, `obligation_failure_taxonomy_complete_score`,
`v598_evidence_root_ready_score`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. One principle SHALL exist for every
top-level field. The inference substrate SHALL declare deterministic CPU
replay. `verifier_is_oracle` SHALL be false. `verdict_class` SHALL be one of
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL be terminal, row-supported, and start with
`complete_`.

### SCENARIO-CONSTRAINT-6835-PRECONDITIONS: Invalid Sources Stop The Freeze

Given a missing row, duplicate identity, hash drift, unreadable artifact, or
owned-source hash drift,
When Exp6835 checks its frozen inputs,
Then it SHALL write the complete blocked artifact with the exact failed check
and SHALL not emit source obligation rows.

### SCENARIO-CONSTRAINT-6835-FRESH-PARSER: Raw Bytes Are Reparsed Without Reducers

Given Exp6833 raw-output byte receipts,
When Exp6835 classifies rows,
Then parser status and selected action identifiers SHALL come from Exp6835
code only, with no Exp6833 or Exp6834 reducer imports.

### SCENARIO-CONSTRAINT-6835-TAXONOMY: Obligation Outcomes Stay Separated

Given malformed output, omitted expected atoms, conflicting selected actions,
and correct expected atoms,
When Exp6835 emits obligation rows,
Then it SHALL label protocol failure, omission, contradiction, atom pass, and
joint pass as separate fields.

### SCENARIO-CONSTRAINT-6835-ACCOUNTING: Target Support Is Recomputed

Given complete source artifacts,
When Exp6835 builds its evidence root,
Then observed target cells, missing target cells, collision witnesses, and the
compatible-policy lower bound SHALL match the finite V597 support audit.

### SCENARIO-CONSTRAINT-6835-NULL-PRESERVATION: Readiness Is Not A Result

Given Exp6834 reports `not_separated`,
When Exp6835 completes the immutable evidence root,
Then it SHALL preserve the source null, set readiness only as an evidence
integrity field, and SHALL NOT convert readiness into a positive verdict.

## Implementation Status (REQ-CONSTRAINT-6835)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6835 and SCENARIO-CONSTRAINT-6835-* | Implemented: deterministic CPU evidence freeze with a fresh parser and atom taxonomy. | Implemented: focused tests, scoped 100% coverage, artifact verification, and lint checks. |


## REQ-CONSTRAINT-6836: Typed Obligation Program Fixture

The system SHALL build Exp6836 as a deterministic CPU typed obligation program
fixture for V598. It SHALL read Exp6811, Exp6832, and Exp6835 before
generation. Before compiling rows, it SHALL require
`v598_evidence_root_ready_score=1`, the Exp6811 schema, the Exp6832 fixture
schema and readiness field, stable source hashes, and no generated answer in
the new candidate fixture. A failed check SHALL write
`complete_blocked_typed_obligation_program_fixture`. The blocked artifact SHALL
name the failed check, expected value, and observed value in
`gate_check_summary`.

The typed program SHALL compile each immutable obligation atom ledger into one
shared program. The compiled views SHALL include an exact scalar energy,
satisfaction predicate, memory admission guard, ARC shadow action guard, and
per-atom diagnostic. The views SHALL share atom identities and exact semantics.
The implementation SHALL not maintain five independent hand-written policies.
Atom omission, contradiction, impossible sets, parse failure, and unknown
actions SHALL fail closed.

The fixture SHALL freeze compatible and one-atom-violating fixed-sequence
candidate pairs. It SHALL include joint violations and impossible cases. Within
each evaluated tokenizer, candidates in a pair SHALL have equal token counts.
Rows SHALL include candidate identifier, row order, label-swap, token length,
prompt length, and surface-form controls. Rows SHALL store raw text and
expected tokenization inputs. Rows SHALL store no model scores.

The exact checker SHALL validate every candidate. Readiness fields
`typed_obligation_program_ready_score` and
`obligation_pair_fixture_ready_score` SHALL depend only on compile parity and
fixture integrity. They SHALL not depend on any learned or model-derived
margin. The terminal artifact SHALL be
`results/experiment_6836_typed_obligation_program_fixture.json`. It SHALL
include `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`,
`random_seed`, `reproducibility_checksum`, `typed_program_schema`,
`compiled_view_manifest`, `atom_identity_manifest`,
`compile_parity_results`, `rows`, `candidate_pair_manifest`,
`exact_candidate_labels`, `shortcut_control_manifest`,
`checker_mutation_results`, `typed_obligation_program_ready_score`,
`obligation_pair_fixture_ready_score`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`. One principle
SHALL exist for every top-level field. The inference substrate SHALL declare
deterministic CPU compilation. `verifier_is_oracle` SHALL be false.
`verdict_class` SHALL be one of `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL be terminal
and start with `complete_`.

### SCENARIO-CONSTRAINT-6836-PRECONDITIONS: Invalid Sources Stop Compilation

Given a missing V598 evidence-root gate, schema drift, fixture readiness drift,
source-hash drift, or generated-answer field,
When Exp6836 evaluates source gates,
Then it SHALL emit the complete blocked artifact with the exact failed check
and SHALL emit no candidate rows.

### SCENARIO-CONSTRAINT-6836-COMPILE-PARITY: Views Share One Atom Ledger

Given an immutable obligation atom ledger,
When the typed program compiles its five views,
Then energy, satisfaction, memory guard, ARC shadow guard, and diagnostics
SHALL agree on every atom identity and exact pass or fail decision.

### SCENARIO-CONSTRAINT-6836-ATOM-FAILURES: Unsafe Candidates Fail Closed

Given omitted atoms, contradictory actions, impossible source sets, parse
failures, and unknown actions,
When the exact checker evaluates candidates,
Then every unsafe candidate SHALL fail closed with a positive energy and a
diagnostic that names the atom or parser failure.

### SCENARIO-CONSTRAINT-6836-CANDIDATE-PAIRS: Matched Fixed Sequences Are Frozen

Given compatible and one-atom-violating candidates,
When Exp6836 freezes pair rows,
Then each pair SHALL include raw fixed-sequence text, candidate identifiers,
exact labels, joint or impossible case coverage, and equal token counts within
each evaluated tokenizer.

### SCENARIO-CONSTRAINT-6836-CONTROLS: Shortcut Controls Are Explicit

Given identifier, row-order, label-swap, token-length, prompt-length, and
surface-form controls,
When Exp6836 validates fixture integrity,
Then each control SHALL be present, hash-bound, and independent of model
scores.

### SCENARIO-CONSTRAINT-6836-SERIALIZATION: Canonical Bytes Are Stable

Given the same program, rows, and source receipts,
When Exp6836 serializes the artifact twice or permutes input atom order,
Then hashes, labels, diagnostics, and readiness fields SHALL stay stable except
for measured duration.

## Implementation Status (REQ-CONSTRAINT-6836)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6836 and SCENARIO-CONSTRAINT-6836-* | Planned: deterministic typed obligation program and fixed-sequence candidate fixture. | Planned: focused tests, scoped 100% coverage, artifact verification, and lint checks. |

## REQ-CONSTRAINT-6837: Output-Free Compatibility Sequence Scoring

The system SHALL score the frozen Exp6836 compatible and violation candidate
pairs with local llama.cpp CUDA forced-sequence scoring. It SHALL use exactly
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL fit no probe, generate no answer,
use no LLM judge, and make no HSRM parity claim.

Before inference, the producer SHALL call `cached_sota_pair()`. It SHALL
require all three exact GGUF files and hashes, embedded GGUF tokenizer
metadata, token log-probability support, CUDA, a task-owned exclusive GPU
lease, free ports, sufficient disk, both Exp6836 readiness fields equal to 1,
and one live forced-score canary per model. Any failed check SHALL write
`complete_blocked_output_free_compatibility`. The blocked artifact SHALL
record the exact failed check, expected value, and observed value in
`gate_check_summary`.

The producer SHALL run one task-owned model process at a time. It SHALL record
the command, PID, process start time, port, GPU UUID, visible devices, model
hash, tokenizer hash, first score, final score, and clean teardown. It SHALL
observe unrelated processes but never interrupt them. Rows from one model
process SHALL not be attributed to another model.

Each fixed sequence SHALL be scored without sampling, repair, generation,
grammar constraints, or answer feedback. The prompt tokens SHALL be masked out.
Each margin SHALL store every candidate token id and token log-probability used
in the compatible and violation scores. The producer SHALL require equal-token
pairing within the model tokenizer before computing a margin. It SHALL emit raw
receipts for every scored candidate.

The producer SHALL checkpoint bounded batches. On restart it SHALL verify row
hashes and run only missing row identities. It SHALL never regenerate complete
rows.

The terminal artifact SHALL be
`results/experiment_6837_three_family_output_free_compatibility.json`. It SHALL
include `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `model_specs`, `models_used`, `model_artifact_hashes`,
`tokenizer_receipts`, `process_receipts`, `accelerator_samples`,
`random_seed`, `source_artifact_hashes`, `reproducibility_checksum`, `rows`,
`per_model_results`, `per_atom_results`, `joint_results`,
`shortcut_control_cells`, `checkpoint_manifest`, `method_parity_limits`,
`obligation_compatibility_stream_ready_score`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`. One principle
SHALL exist for every top-level field. The inference substrate SHALL be
`live_local_llama_cpp_cuda_forced_sequence_scoring`. `verifier_is_oracle` SHALL
be false. `verdict_class` SHALL be one of `positive`, `circular_positive`,
`null`, `blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL be
terminal, row-supported, and start with `complete_`.

`obligation_compatibility_stream_ready_score` SHALL depend only on authentic
receipts and row completeness. Effect estimates SHALL stay separate. Lower
compatible-sequence energy SHALL not be reported as proof of truth. Margins
SHALL be reported by model, atom family, obligation count, compatible label
position, prompt length, candidate length, and permutation. The producer SHALL
not pool models as exchangeable samples.

Field principles SHALL be:
`field_principles`: Documents why each required field exists.
`preconditions_checked`: Names every live gate and the exact observed value.
`inference_substrate`: Distinguishes forced scoring from answer generation.
`duration_s`: Makes skipped or implausibly short work visible.
`model_specs`: Freezes the mandated model identities.
`models_used`: Shows which models actually ran.
`model_artifact_hashes`: Binds model bytes to scored rows.
`tokenizer_receipts`: Proves native GGUF tokenization and lengths.
`process_receipts`: Proves task-owned process identity and teardown.
`accelerator_samples`: Records CUDA device and lease observations.
`random_seed`: Makes batch ordering and canaries reproducible.
`source_artifact_hashes`: Binds Exp6836 and implementation inputs.
`reproducibility_checksum`: Detects drift across code, source, model, and rows.
`rows`: Stores every scored model and candidate-pair margin.
`per_model_results`: Keeps model families separate.
`per_atom_results`: Reports atom-family margins without pooling models.
`joint_results`: Reports joint and impossible-case margins separately.
`shortcut_control_cells`: Audits label position, length, prompt, and permutation cues.
`checkpoint_manifest`: Proves restart skips complete rows.
`method_parity_limits`: States this is not HSRM and not truth proof.
`obligation_compatibility_stream_ready_score`: Gates only on receipts and rows.
`gate_check_summary`: Names the failed gate in blocked artifacts.
`verifier_is_oracle`: Keeps exact labels external to model scores.
`verdict_class`: Uses the closed terminal class vocabulary.
`honest_verdict`: Gives a terminal complete-prefixed outcome.

### SCENARIO-CONSTRAINT-6837-PRECONDITIONS: Failed Gates Stop Scoring

Given any missing model, hash, tokenizer metadata, log-probability support,
CUDA, lease, free port, disk budget, Exp6836 readiness field, or canary,
When Exp6837 evaluates preconditions,
Then it SHALL write `complete_blocked_output_free_compatibility` with exact
expected and observed values and SHALL emit no margin rows.

### SCENARIO-CONSTRAINT-6837-FORCED-SCORING: Candidate Scores Use Token Logprobs

Given one compatible and one violation candidate with equal tokenizer length,
When Exp6837 scores the pair,
Then the row SHALL include prompt-masked candidate token ids, token
log-probabilities, compatible score, violation score, and their margin.

### SCENARIO-CONSTRAINT-6837-TOKEN-ALIGNMENT: Pairing Is Equal-Token Only

Given a matched pair whose compatible and violation candidates tokenize to
different lengths,
When Exp6837 prepares the row,
Then the row SHALL be rejected or blocked before a margin is computed.

### SCENARIO-CONSTRAINT-6837-CHECKPOINT-RESTART: Complete Rows Are Immutable

Given a checkpoint with valid row hashes for some identities,
When Exp6837 restarts,
Then it SHALL verify those hashes, skip complete identities, and score only
missing identities.

### SCENARIO-CONSTRAINT-6837-PROCESS-OWNERSHIP: Models Are Isolated

Given three mandated model specs,
When Exp6837 runs sequential model phases,
Then each phase SHALL have one task-owned process receipt, no process identity
reuse across models, and a clean teardown receipt.

### SCENARIO-CONSTRAINT-6837-ARTIFACT: Readiness Is Receipt And Row Complete

Given all expected model-pair rows and authentic receipts,
When Exp6837 builds the terminal artifact,
Then readiness SHALL be one only when rows, raw receipts, checkpoints, model
isolation, and teardown are complete; margin direction SHALL not control
readiness.

## Implementation Status (REQ-CONSTRAINT-6837)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6837 and SCENARIO-CONSTRAINT-6837-* | Planned: output-free local llama.cpp CUDA forced-sequence scoring with blocked preflight output. | Planned: focused tests cover preconditions, token alignment, masked scoring, raw receipts, checkpoint restart, model isolation, and artifact readiness. |

## REQ-CONSTRAINT-6849: Typed Program Isomorphic Authority Audit

The system SHALL build Exp6849 as a deterministic CPU authority audit. It
SHALL require `v599_evidence_contract_ready_score=1`. It SHALL also require
readable Exp6836 raw fixture material and the Exp6847 discrepancy rows. A
failed precondition SHALL write
`complete_blocked_typed_program_isomorphic_authority_audit`. The blocked
artifact SHALL name the failed check, expected value, and observed value in
`gate_check_summary`.

Exp6849 SHALL parse the raw typed programs stored in Exp6836. It SHALL not
import the Exp6836 reducer. A fresh reducer SHALL construct unique candidate,
pair, atom, view, transform, and row identifiers. It SHALL reject duplicate
identifiers, semantic aliases, atom omissions, impossible programs, and row
collisions.

The fresh reducer SHALL compile exact energy, satisfaction, memory admission,
ARC action admission, and per-atom diagnostics from one typed source. It SHALL
recompile every view on each base row and each transformed row. All compiled
views SHALL use the same atom identities and SHALL return the same pass or fail
decision.

Exp6849 SHALL apply identifier permutations, atom renames, label swaps, row
reorderings, surface paraphrases, duplicate removal, and one-atom semantic
mutations. Isomorphic transforms SHALL preserve exact labels. One-atom semantic
mutations SHALL change exact labels. Duplicate removal SHALL preserve one
canonical semantic candidate and SHALL remove every alias.

Exp6849 SHALL freeze a sanitized candidate-pair manifest for Exp6851. The
manifest SHALL include raw sequence inputs. It SHALL store no model scores. It
SHALL not claim token equality without tokenizer receipts.

The exact checker SHALL be the authority. The terminal artifact SHALL be
`results/experiment_6849_typed_program_isomorphic_authority_audit.json`. It
SHALL include `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `random_seed`,
`reproducibility_checksum`, `rows`, `fresh_reducer_manifest`,
`candidate_identity_manifest`, `collision_witnesses`,
`compiled_view_parity_rows`, `isomorphic_transform_manifest`,
`semantic_mutation_rows`, `duplicate_removal_results`,
`sanitized_candidate_pair_manifest`, `authority_audit_complete_score`,
`typed_program_authority_ready_score`, `isomorphic_fixture_ready_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. One principle SHALL exist for every top-level field. The
inference substrate SHALL be `deterministic CPU exact compilation`.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL use the closed
terminal vocabulary. `honest_verdict` SHALL start with `complete_`.

`typed_program_authority_ready_score` and `isomorphic_fixture_ready_score`
SHALL equal one only when exact parity, identity uniqueness, transform
invariance, duplicate removal, and semantic mutation gates pass. These fields
SHALL not depend on a learned score or a model score.

### SCENARIO-CONSTRAINT-6849-PRECONDITIONS: Missing Authority Inputs Block

Given an invalid V599 contract gate, unreadable Exp6836 fixture material, or
missing Exp6847 discrepancy rows,
When Exp6849 evaluates preconditions,
Then it SHALL emit the complete blocked artifact with no audit rows and the
exact failed check in `gate_check_summary`.

### SCENARIO-CONSTRAINT-6849-DUPLICATE-IDS: Duplicate Identities Fail Closed

Given two records with the same identifier,
When the fresh reducer validates an identity namespace,
Then it SHALL reject the records and SHALL emit a collision witness.

### SCENARIO-CONSTRAINT-6849-SEMANTIC-ALIAS: Semantic Aliases Are Removed

Given different candidate identifiers with the same typed semantics,
When the fresh reducer sanitizes the fixture,
Then it SHALL keep one canonical candidate and record each removed alias.

### SCENARIO-CONSTRAINT-6849-ATOM-OMISSION: Omitted Atoms Fail Closed

Given a candidate that omits one required atom,
When the exact checker recompiles all views,
Then energy SHALL be positive and every Boolean view SHALL reject the
candidate.

### SCENARIO-CONSTRAINT-6849-IMPOSSIBLE: Impossible Programs Fail Closed

Given a typed program with no legal action set,
When the fresh reducer compiles the program,
Then it SHALL mark the program impossible and SHALL not label any candidate as
compatible.

### SCENARIO-CONSTRAINT-6849-ROW-COLLISION: Row Identities Are Unique

Given two rows with the same row identifier or semantic row identity,
When fixture integrity is checked,
Then readiness SHALL remain zero and the artifact SHALL record the collision.

### SCENARIO-CONSTRAINT-6849-ISOMORPHIC: Isomorphic Labels Stay Stable

Given a base row and an identifier permutation, atom rename, label swap, row
reordering, surface paraphrase, or duplicate-removal transform,
When the exact checker evaluates both rows,
Then the candidate labels SHALL be equal after the transform mapping is
applied.

### SCENARIO-CONSTRAINT-6849-REDUCER-MUTATION: Semantic Mutations Must Flip

Given a compatible candidate and a one-atom semantic mutation,
When the exact checker recompiles every view,
Then the candidate label SHALL change from compatible to incompatible. A
reducer that ignores the mutation SHALL fail the readiness gate.

### SCENARIO-CONSTRAINT-6849-SANITIZED-FIXTURE: Exp6851 Input Is Score-Free

Given all authority gates pass,
When Exp6849 freezes the candidate-pair manifest,
Then each candidate, pair, row, atom, view, and transform identity SHALL be
unique. The manifest SHALL store raw sequence inputs and no model scores. Token
equality SHALL remain unclaimed without tokenizer receipts.

## Implementation Status (REQ-CONSTRAINT-6849)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6849 and SCENARIO-CONSTRAINT-6849-* | Implemented: fresh deterministic reducer and sanitized isomorphic fixture. | Implemented: focused tests, scoped 100% coverage, artifact verification, mutation checks, and repository audits. |


## REQ-CONSTRAINT-6851: Three-Family Isomorphic Compatibility Stream

The system SHALL score the sanitized Exp6849 candidate pairs and every
qualified isomorphic transform with local llama.cpp CUDA forced-sequence
scoring. It SHALL use exactly `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL use two deterministic repeats for
each model, semantic pair, and transform. It SHALL fit no probe, generate no
answer, use no grammar, repair no output, use no answer feedback, and use no
LLM judge.

Before scoring, the producer SHALL require the Exp6849
`typed_program_authority_ready_score` and `isomorphic_fixture_ready_score` to
equal one. It SHALL also require the Exp6850
`three_family_scoring_admission_ready_score` to equal one. It SHALL verify the
frozen Exp6849 and Exp6850 artifact hashes, the Exp6849 fixture and transform
implementation hash, every model hash, and every embedded tokenizer hash. It
SHALL require CUDA token scoring, free task-owned ports, a task-owned exclusive
GPU lease, and a fresh one-row canary for each model. A failed gate SHALL emit
`complete_blocked_three_family_isomorphic_compatibility_stream`. The blocked
artifact SHALL name the failed check, expected value, and observed value in
`gate_check_summary`.

The producer SHALL run one owned model process at a time. Before each bounded
batch, it SHALL revalidate the lease and run a fresh unlabeled forced-score
canary. It SHALL checkpoint after each bounded batch. It SHALL stop only a
process whose ownership token, PID, start time, parent identity, command hash,
process group, and port match the recorded receipt. It SHALL record clean port
release and SHALL never stop an unrelated process.

The producer SHALL reconstruct transformed surface sequences with the exact
Exp6849 transform implementation whose hash Exp6849 recorded. Base and
transformed rows SHALL join through the Exp6849 semantic pair identity. They
SHALL not join through candidate text, identifiers, token counts, or another
surface identity. Identifier permutation, atom rename, label swap, row
reordering, surface paraphrase, and duplicate removal SHALL remain separate
transform cells.

Each row SHALL identify one model, semantic pair, transform, compatible-label
position, and repeat. It SHALL store both exact candidate labels and one raw
token-score receipt per candidate. Each receipt SHALL contain every prompt
token id, candidate token id, and candidate token log-probability used in the
margin. Prompt token ids SHALL be identical across the paired candidates and
SHALL be excluded from the candidate score. Candidate token ids and token
log-probabilities SHALL align one-to-one within each sequence. Cross-candidate
token counts need not be equal. The row SHALL report both summed conditional
log-likelihood margin and mean-token conditional log-likelihood margin. The
mean-token margin SHALL be the primary scalar compatibility margin, while the
two candidate lengths and their difference remain explicit controls.

On restart, the producer SHALL verify the checkpoint input checksum, row
hashes, and row identities. It SHALL skip only complete verified identities.
It SHALL reject duplicate, malformed, or tampered rows and SHALL run only
missing identities.

The terminal artifact SHALL be
`results/experiment_6851_three_family_isomorphic_compatibility_stream.json`.
It SHALL include `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `model_specs`, `models_used`,
`model_artifact_hashes`, `tokenizer_receipts`, `process_receipts`,
`accelerator_samples`, `random_seed`, `reproducibility_checksum`, `rows`,
`token_score_receipts`, `base_isomorphic_join_manifest`,
`per_model_margin_summary`, `per_atom_margin_summary`,
`per_transform_margin_summary`, `control_margin_summary`,
`checkpoint_manifest`, `teardown_receipts`,
`compatibility_stream_complete_score`, `positive_margin_models`,
`isomorphic_consistency_rate`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. It MAY add a separate joint-margin
summary and source-hash manifest. One principle SHALL exist for every
top-level field.

The inference substrate SHALL be
`live_local_llama_cpp_cuda_forced_sequence_scoring`.
`compatibility_stream_complete_score` SHALL depend only on complete rows,
token receipts, model receipts, batch canaries, lease receipts, checkpoints,
and clean teardown. Margin direction SHALL not affect this score.
`positive_margin_models` and `isomorphic_consistency_rate` SHALL remain effect
fields. Models SHALL stay separate in every summary and SHALL not be treated as
exchangeable samples. A terminal all-three effect SHALL be `positive` only
when all three per-model mean margins are positive; otherwise its effect class
SHALL be `null`, while `positive_margin_models` retains any mixed direction.
A lower compatible-sequence energy SHALL not be reported as exact truth.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL be one of `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
`honest_verdict` SHALL be row-supported, terminal, and start with `complete_`.

### SCENARIO-CONSTRAINT-6851-PRECONDITIONS: Structured Gates Fail Closed

Given any gate score other than one, changed fixture or admission bytes,
changed model or tokenizer bytes, unavailable CUDA scoring, unavailable lease,
occupied task port, or failed fresh canary,
When Exp6851 evaluates the gate,
Then it SHALL emit the complete blocked artifact with the exact failed check
and SHALL set `compatibility_stream_complete_score` to zero.

### SCENARIO-CONSTRAINT-6851-FORCED-SCORING: Only Candidate Tokens Form Margins

Given one exact-compatible and one matched violating fixed sequence,
When Exp6851 scores the pair,
Then it SHALL exclude prompt-token likelihoods, store both candidate token
streams and log-probabilities, and compute summed and mean-token margins
without sampling or generation.

### SCENARIO-CONSTRAINT-6851-TOKEN-ALIGNMENT: Raw Receipts Are One-To-One

Given a candidate receipt with a missing token log-probability, a non-finite
value, or prompt tokens that differ from its matched candidate,
When Exp6851 validates the receipt,
Then it SHALL reject the row before computing a margin. Unequal token counts
between the two complete candidates SHALL remain an explicit control and SHALL
not be silently truncated or padded.

### SCENARIO-CONSTRAINT-6851-ISOMORPHIC-PAIRING: Semantic Identity Joins Surfaces

Given one Exp6849 semantic pair and its six qualified transforms,
When Exp6851 creates score inputs,
Then the base and transformed surfaces SHALL share the semantic pair identity,
retain exact labels, and have distinct transform identities and surface hashes.

### SCENARIO-CONSTRAINT-6851-LABEL-POSITION: Label Swaps Stay Visible

Given the base and label-swap cells for one semantic pair,
When Exp6851 records their row identities,
Then the compatible-label position SHALL change while the semantic pair
identity and exact candidate labels remain unchanged.

### SCENARIO-CONSTRAINT-6851-CHECKPOINT-RESTART: Verified Rows Are Immutable

Given a checkpoint with complete row hashes and a matching input checksum,
When Exp6851 restarts,
Then it SHALL skip those row identities and score only missing rows. A changed
checksum, duplicate identity, or changed row hash SHALL fail closed.

### SCENARIO-CONSTRAINT-6851-PROCESS-OWNERSHIP: Cleanup Is Owner Scoped

Given sequential model phases and unrelated local processes,
When Exp6851 completes or fails,
Then each phase SHALL have a distinct owned process receipt and task lease,
and cleanup SHALL confirm process exit and port release without unrelated
signals.

### SCENARIO-CONSTRAINT-6851-RAW-RECEIPTS: Completeness Is Not Effect Direction

Given every expected row, token receipt, batch canary, lease receipt,
checkpoint hash, and clean teardown,
When Exp6851 builds the artifact,
Then `compatibility_stream_complete_score` SHALL equal one even when any or all
model margins are zero or negative. Effect fields SHALL report those outcomes
separately.

## Implementation Status (REQ-CONSTRAINT-6851)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6851 and SCENARIO-CONSTRAINT-6851-* | Implemented: owned three-family CUDA forced-sequence scoring over frozen base and isomorphic surfaces. | Implemented: focused tests and scoped 100% coverage verify gates, masking, token alignment, semantic pairing, label position, restart hashes, ownership, and receipt-only completeness. |


## REQ-CONSTRAINT-6852: Compatibility Shortcut and Authority Audit

The system SHALL build Exp6852 as an ungated, deterministic CPU audit of all
available Exp6849 authority evidence, Exp6850 admission evidence, and Exp6851
scientific score evidence. It SHALL invoke no LLM. It SHALL use a fresh reducer
that does not import Exp6849, Exp6850, or Exp6851 aggregation code.

Before reduction, Exp6852 SHALL inventory the state, path, current SHA-256 hash,
recorded source hash, and conductor records for each upstream task. It SHALL
preserve every matching skip, failure, blocked, and completion record. A missing
producer or zero scientific rows SHALL produce a complete blocked artifact. It
SHALL not cause Exp6852 to omit its artifact.

Exp6852 SHALL recompute summed and mean-token sequence margins from the raw
candidate token log-probabilities. It SHALL reject non-finite or null scores,
token and score length mismatches, different paired prompt tokens, duplicate
row or receipt identities, missing candidate receipts, and labels that disagree
with the sanitized Exp6849 semantic identity manifest. It SHALL match base and
isomorphic rows through the manifest semantic pair identity. It SHALL not match
rows through candidate text, identifier, length, token count, label position,
or row position.

For every available model and semantic pair, Exp6852 SHALL run identifier-only,
prompt-length, candidate-length, token-count, label-position, row-order,
normalization, surface-form, and model-family shortcut attacks. Each attack
SHALL produce one explicit row for each audited unit. A missing observation
SHALL remain null and SHALL never enter an average as zero.

Exp6852 SHALL test whether each base effect direction survives every available
isomorphic transform. It SHALL report whether each shortcut control explains
the same absolute margin or a greater absolute margin. Models SHALL remain
separate. Model scale and model family SHALL be controls, not exchangeable
replicates.

The terminal artifact SHALL be
`results/experiment_6852_compatibility_shortcut_authority_audit.json`. It SHALL
include `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `upstream_state_manifest`,
`conductor_skip_manifest`, `random_seed`, `reproducibility_checksum`, `rows`,
`recomputed_margin_rows`, `isomorphic_invariance_results`,
`shortcut_attack_results`, `missing_model_manifest`,
`control_explanation_results`, `authority_failure_witnesses`,
`compatibility_audit_complete_score`, `compatibility_claim_eligible_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. One principle SHALL exist for every top-level field. The
inference substrate SHALL be `deterministic CPU independent reduction`.
`verifier_is_oracle` SHALL be false.
Repository adversarial verification SHALL classify this substrate as a
deterministic verifier. Referenced upstream model names SHALL not imply a new
model invocation by Exp6852.

`compatibility_audit_complete_score` SHALL equal one when Exp6852 inventories
all upstream inputs and honestly reduces all evidence that exists. This score
MAY equal one for a blocked or partial scientific result.
`compatibility_claim_eligible_score` SHALL equal one only when every authority,
artifact-hash, model-completeness, receipt-completeness, shortcut, and
isomorphic-invariance gate passes. Otherwise it SHALL equal zero. A failed gate
SHALL name the check and its expected and observed values in
`gate_check_summary`.

`verdict_class` SHALL be one of `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. Missing scientific rows SHALL produce
`blocked`. Some but not all required model rows SHALL produce `partial` unless
an authority failure requires `disqualified`. Complete rows with an authority
or stale-hash failure SHALL produce `disqualified`. Complete authoritative rows
that fail shortcut or invariance gates SHALL produce `null`. A positive verdict
SHALL require an eligible model-level compatibility claim. `honest_verdict`
SHALL be terminal and supported by the emitted rows.

### SCENARIO-CONSTRAINT-6852-MISSING-PRODUCER: Missing Rows Stay Blocked

Given no readable Exp6851 artifact or no scientific rows,
When Exp6852 inventories and reduces the upstream evidence,
Then it SHALL emit the complete artifact, preserve the missing input as a row,
set claim eligibility to zero, and use the blocked verdict class.

### SCENARIO-CONSTRAINT-6852-PARTIAL-MODELS: Missing Models Are Not Zero

Given valid score rows for some but not all required models,
When Exp6852 reduces model effects,
Then it SHALL list each absent model in `missing_model_manifest`, keep its
margin null, exclude it from averages, and use the partial verdict class.

### SCENARIO-CONSTRAINT-6852-NULL-SCORES: Null Token Scores Fail Closed

Given a candidate receipt whose token log-probabilities are all null,
When Exp6852 recomputes its row margin,
Then it SHALL emit an authority failure witness and SHALL not convert a null
score to zero.

### SCENARIO-CONSTRAINT-6852-DUPLICATE-IDENTITY: Duplicate Rows Fail Closed

Given two producer rows with the same row identity or semantic scoring unit,
When Exp6852 validates identities,
Then it SHALL reject the duplicate and record both conflicting identities.

### SCENARIO-CONSTRAINT-6852-LABEL-INVERSION: Manifest Labels Are Authority

Given a producer row whose candidate labels invert the sanitized manifest,
When Exp6852 joins the producer row to semantic authority,
Then it SHALL reject the row and record the expected and observed labels.

### SCENARIO-CONSTRAINT-6852-ROW-REORDER: Input Order Cannot Change Results

Given the same valid producer rows in a different list order,
When Exp6852 independently reduces both inputs,
Then its scientific rows and gates SHALL be identical after canonical sorting.

### SCENARIO-CONSTRAINT-6852-STALE-HASH: Changed Inputs Are Disqualified

Given a recorded upstream source hash that differs from the current artifact
hash,
When Exp6852 checks evidence authority,
Then it SHALL record the stale hash, set claim eligibility to zero, and use the
disqualified verdict class when scientific rows exist.

### SCENARIO-CONSTRAINT-6852-SHORTCUTS: Controls Must Not Explain the Effect

Given authoritative complete model rows,
When an identifier, length, token-count, label-position, row-order,
normalization, surface-form, model-scale, or model-family control explains the
same absolute margin or a greater absolute margin,
Then Exp6852 SHALL fail the shortcut gate and set claim eligibility to zero.

### SCENARIO-CONSTRAINT-6852-ISOMORPHIC: Direction Must Survive Transforms

Given one semantic pair with a base row and available qualified transforms,
When any transform reverses or removes the base margin direction,
Then Exp6852 SHALL report the failed unit, fail the isomorphic gate, and set
claim eligibility to zero.

## Implementation Status (REQ-CONSTRAINT-6852)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6852 and SCENARIO-CONSTRAINT-6852-* | Implemented: fresh deterministic CPU reduction of semantic authority and raw token receipts; the exact substrate is registered with repository adversarial verification as deterministic-verifier work. | Implemented: focused tests cover missing, partial, malformed, reordered, stale, shortcut, invariance, and upstream-model-marker isolation with scoped 100% coverage. |


## REQ-CONSTRAINT-6862: Dual-Side Semantic Contrast Bank

The system SHALL build Exp6862 as a deterministic CPU semantic contrast bank.
It SHALL invoke no LLM and SHALL compute no model score. It SHALL require
`v600_evidence_contract_ready_score=1`, readable Exp6849 typed source
programs, and the Exp6852 shortcut and direction failure witnesses. A failed
precondition SHALL write `complete_blocked_dual_side_semantic_contrast_bank`.
The blocked artifact SHALL name the failed check and observed value in
`gate_check_summary`.

A fresh reducer SHALL build at least 96 accepted groups. The bank SHALL cover
exact energy, satisfaction predicate, memory guard, ARC guard, and diagnostic
obligation families. Each group SHALL contain one exact valid candidate and
one minimally invalid candidate. The invalid candidate SHALL differ by one
semantic atom. Every program, candidate, contrast, transform, group, and row
identifier SHALL be derived from canonical content.

A structure-side checker SHALL prove the selected variable, obligation set,
atom ledger, and one-atom mutation map. A separate solution-side checker
SHALL derive the legal action set and candidate label from the typed program.
Neither checker SHALL call or import the other. A group SHALL enter the bank
only when both authorities return the exact valid and invalid labels.

The reducer SHALL apply identifier renames, label swaps, row reorders,
normalization variants, surface paraphrases, and semantics-preserving atom
order changes. Each nuisance transform SHALL preserve both exact labels. The
reducer SHALL also apply one-atom semantic mutations. Each semantic mutation
SHALL change the exact label.

The reducer SHALL reject duplicate semantic identities, semantic aliases,
ambiguous programs, identity collisions, omitted atoms, vacuous constraints,
checker disagreements, and split mutations. A split mutation changes more
than one semantic atom. The reducer SHALL preserve every rejection witness.
Candidate names, label positions, and row order SHALL not determine a label.

Exp6862 SHALL freeze raw prompt and candidate sequence templates for Exp6863.
It SHALL not assign calibration or held groups. It SHALL not inspect a model
tokenizer. The templates SHALL contain no model score.

The terminal artifact SHALL be
`results/experiment_6862_dual_side_semantic_contrast_bank.json`. It SHALL
include `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `random_seed`,
`reproducibility_checksum`, `rows`, `fresh_reducer_manifest`,
`typed_family_manifest`, `semantic_contrast_group_manifest`,
`structure_side_check_rows`, `solution_side_check_rows`,
`authority_disagreement_witnesses`, `identity_collision_witnesses`,
`rejected_group_manifest`, `nuisance_transform_manifest`,
`semantic_mutation_rows`, `accepted_contrast_group_count`,
`dual_side_semantic_contrast_bank_ready_score`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`. It SHALL also
include the frozen template manifest. One principle SHALL exist for every
top-level field.

The inference substrate SHALL be
`deterministic CPU dual-side exact checking`. `verifier_is_oracle` SHALL be
false because two external exact checkers supply authority. `verdict_class`
SHALL be one of `positive`, `circular_positive`, `null`, `blocked`,
`disqualified`, or `partial`. `honest_verdict` SHALL be terminal and SHALL
start with `complete_`.

`dual_side_semantic_contrast_bank_ready_score` SHALL equal one only when at
least 96 groups pass both authorities, all identity namespaces are unique,
all nuisance labels stay stable, all semantic mutations change labels, and
every negative control is rejected. It SHALL otherwise equal zero.

### SCENARIO-CONSTRAINT-6862-PRECONDITIONS: Source Gates Fail Closed

Given a V600 gate other than one, unreadable Exp6849 typed sources, or missing
Exp6852 failure witnesses,
When Exp6862 evaluates preconditions,
Then it SHALL write the complete blocked artifact with no accepted groups and
the exact failed check in `gate_check_summary`.

### SCENARIO-CONSTRAINT-6862-IDENTITY-COLLISION: Identity Collisions Are Rejected

Given two records that share one content identifier but have different
canonical content,
When the reducer audits the identity namespace,
Then it SHALL reject both records and preserve the collision witness.

### SCENARIO-CONSTRAINT-6862-SEMANTIC-ALIAS: Semantic Aliases Are Rejected

Given two groups with different identifiers and the same semantic identity,
When the reducer audits group semantics,
Then it SHALL reject the alias and preserve both group identifiers.

### SCENARIO-CONSTRAINT-6862-OMITTED-ATOM: Omitted Atoms Are Rejected

Given a candidate that omits one required semantic atom,
When both authorities evaluate the candidate,
Then the group SHALL be rejected and the omitted atom SHALL appear in the
rejection witness.

### SCENARIO-CONSTRAINT-6862-VACUOUS-CONSTRAINT: Vacuous Programs Are Rejected

Given a program with no effective prerequisite, authority, fallback,
consequence, or priority atom,
When the structure-side checker validates the program,
Then the group SHALL be rejected before admission.

### SCENARIO-CONSTRAINT-6862-CHECKER-DISAGREEMENT: Authorities Must Agree

Given a candidate whose atom ledger passes structure checks but whose action
set fails independent solution checks,
When the reducer compares authority labels,
Then it SHALL reject the group and preserve the disagreement witness.

### SCENARIO-CONSTRAINT-6862-SPLIT-MUTATION: Invalid Candidates Change One Atom

Given an invalid candidate that changes two or more semantic atoms,
When the mutation map is checked,
Then the group SHALL be rejected as a split mutation.

### SCENARIO-CONSTRAINT-6862-NUISANCE-INVARIANCE: Surface Changes Preserve Labels

Given an accepted group and each required nuisance transform,
When both authorities recheck the transformed content,
Then the valid label SHALL remain true and the invalid label SHALL remain
false. Candidate names, display labels, and row positions SHALL not affect the
result.

### SCENARIO-CONSTRAINT-6862-SEMANTIC-MUTATION: One Atom Changes the Label

Given an exact valid candidate and its declared one-atom mutation,
When both authorities evaluate the pair,
Then both SHALL change the label from true to false.

### SCENARIO-CONSTRAINT-6862-FROZEN-TEMPLATES: Exp6863 Inputs Stay Unassigned

Given a ready semantic contrast bank,
When Exp6862 freezes the raw sequence templates,
Then calibration and held assignments SHALL remain null, tokenizer inspection
SHALL be false, and no template or row SHALL contain a model score.

## Implementation Status (REQ-CONSTRAINT-6862)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6862 and SCENARIO-CONSTRAINT-6862-* | Implemented: fresh deterministic dual-side semantic contrast bank with 100 accepted groups. | Implemented: focused tests, scoped 100% coverage, artifact checks, mutation controls, and repository audits. |

## REQ-VERIFY-6869: Calibration-Only Paired Semantic Rule

The system SHALL reduce only the Exp6867 calibration labels and the Exp6868
calibration token rows. It SHALL not parse the held score sidecar. It SHALL not
read a held label or held aggregate. The reducer SHALL record every file and
field access. `held_label_access_count` SHALL remain zero.

Exp6869 SHALL require `semantic_contrast_stream_v2_complete_score=1`. It SHALL
require the exact frozen Exp6867 and Exp6868 file hashes. It SHALL require a
readable calibration label manifest and a sealed held manifest. It SHALL verify
the calibration and held group sets are disjoint. It SHALL verify both score
sidecar file hashes. It SHALL verify the calibration sidecar payload hash from
its raw rows. It SHALL verify the held sidecar by file bytes only.

A failed precondition SHALL write
`complete_blocked_calibration_only_paired_semantic_rule`. The blocked artifact
SHALL include all required fields. Its `gate_check_summary` SHALL record the
first failed check, its expected value, and its observed value.

The reducer SHALL recompute each candidate mean from `token_logprobs`. It SHALL
not trust a producer aggregate. The frozen orientation SHALL be valid minus
invalid. One semantic effect SHALL equal the mean of the base and label-swap
valid-minus-invalid contrasts for one model and semantic group.

The reducer SHALL emit one row for every calibration model, semantic group, and
preregistered nuisance control. Each row SHALL preserve the base contrast, the
label-swap contrast, the semantic effect, the matched nuisance effect, and the
signed difference-in-differences value. It SHALL preserve zero-headroom,
sign-opposed, failed, duplicate, and missing cells as evidence. It SHALL not
impute a score.

The order nuisance SHALL be the mean base-to-swap change over both candidates.
The label-position nuisance SHALL be one half of the difference between the
base and label-swap semantic contrasts. Identifier, normalization, token-count,
character-length, and surface-form nuisance effects SHALL equal zero only when
their exact Exp6867 matching receipts pass. A failed receipt SHALL create a
missing-cell row. For every control, the signed difference-in-differences value
SHALL equal the semantic effect minus the matched nuisance effect. Nuisance
rejection SHALL compare the absolute nuisance effect with the semantic effect.

The interval SHALL be the preregistered 95 percent BCa cluster bootstrap. It
SHALL use 10,000 resamples, seed 6867, and semantic group identity as the
cluster. Model and pooled intervals SHALL resample within semantic family so
that each family keeps equal weight. A changed method, confidence level,
resample count, seed, cluster unit, or family weighting SHALL disqualify the
rule.

The missing-cell rule SHALL use complete nuisance-eligible cells only. It SHALL
use no imputation. The frozen missingness ceiling SHALL be zero missing required
calibration or held cells. Each required held model SHALL also retain the
preregistered floor of 20 groups.

Exp6869 SHALL report model, model-family, semantic-family, nuisance, and pooled
estimates separately. The pooled estimate SHALL give equal weight to models,
then semantic families, then semantic groups within a model-family cell. A
pooled positive SHALL not replace a model or family failure. All five semantic
families in every model SHALL have the same positive sign. All three model
families SHALL share that sign.

The held acceptance contract SHALL freeze these rules before any held label is
opened: valid-minus-invalid orientation; token-mean normalization; the BCa
interval above; a strict minimum effect of zero for every model lower bound;
an absolute nuisance ceiling equal to the largest calibration nuisance upper
bound; the family replication rule above; zero missing required cells; the
20-group held floor; and source, split, sidecar, row, interval, missingness, and
family disqualification rules. The reducer code and contract SHALL have one
stable hash for Exp6870.

`semantic_contrast_rule_ready_score` SHALL equal one only when all
preconditions pass, every model lower bound is strictly above zero, every
model semantic effect exceeds every absolute nuisance bound, all five families
replicate in all three models, no sign reversal occurs, no required cell is
missing, and the frozen reducer completes a calibration-only dry run without
held access. This readiness score SHALL describe a frozen rule. It SHALL not
state a held semantic result.

The terminal artifact SHALL be
`results/experiment_6869_calibration_only_paired_semantic_rule.json`. It SHALL
include `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `calibration_access_log`,
`held_label_access_count`, `rows`, `per_model_calibration_effects`,
`per_family_calibration_effects`, `pooled_calibration_effect`,
`nuisance_control_effects`, `missing_cell_rows`, `bootstrap_rows`,
`family_replication_result`, `frozen_held_reducer_hash`,
`held_acceptance_contract`, `random_seed`, `reproducibility_checksum`,
`semantic_contrast_rule_ready_score`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`. The inference
substrate SHALL be deterministic CPU calibration-only paired reduction.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL be one of
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL be terminal and SHALL start with `complete_`.

### SCENARIO-VERIFY-6869-HELD-LABEL-ACCESS: Held Labels Stay Closed

Given a held group identity or a held-label-shaped field,
When the calibration reducer routes an access,
Then it SHALL reject the access before loading the value, record the attempt,
and keep `held_label_access_count` at zero.

### SCENARIO-VERIFY-6869-SPLIT-COLLISION: Split Collisions Block Reduction

Given one semantic group in both calibration and held manifests,
When Exp6869 checks split authority,
Then it SHALL stop before label reduction and name the collided group.

### SCENARIO-VERIFY-6869-SIGN-REVERSAL: Opposed Transform Signs Fail

Given base and label-swap semantic contrasts with opposite nonzero signs,
When Exp6869 reduces the group,
Then it SHALL preserve the row and forbid rule readiness.

### SCENARIO-VERIFY-6869-NUISANCE-WIN: Nuisance Cannot Explain The Effect

Given an absolute nuisance effect at or above the semantic effect,
When Exp6869 applies nuisance rejection,
Then the affected model SHALL fail and rule readiness SHALL be zero.

### SCENARIO-VERIFY-6869-MISSING-CELL: Missing Cells Are Not Zero

Given a missing, duplicate, nonfinite, hash-invalid, or unmatched score cell,
When Exp6869 builds complete model-group cells,
Then it SHALL preserve a typed missing-cell row and SHALL not impute a score.

### SCENARIO-VERIFY-6869-FAMILY-ONLY-WIN: One Family Cannot Carry A Model

Given a positive pooled or model effect with one nonpositive semantic family,
When Exp6869 checks family replication,
Then the family failure SHALL remain visible and readiness SHALL be zero.

### SCENARIO-VERIFY-6869-POOLED-SIMPSON-REVERSAL: Pooled Wins Stay Disaggregated

Given a positive pooled estimate and a nonpositive model or family estimate,
When Exp6869 applies the held contract,
Then the model or family failure SHALL override the pooled result.

### SCENARIO-VERIFY-6869-BOOTSTRAP-DRIFT: Interval Drift Disqualifies

Given a bootstrap method, seed, resample count, cluster unit, confidence level,
or weighting rule that differs from Exp6867,
When Exp6869 validates the interval contract,
Then it SHALL use the disqualified verdict class and set readiness to zero.

## Implementation Status (REQ-VERIFY-6869)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-6869 and SCENARIO-VERIFY-6869-* | Planned (`python/carnot/experiment_6869_calibration_only_paired_semantic_rule.py`; `scripts/experiments/experiment_6869_calibration_only_paired_semantic_rule.py`) | Planned (`tests/python/test_experiment_6869_calibration_only_paired_semantic_rule.py`) |

### REQ-CONSTRAINT-6886: Pinned Enoki Assets And Exact Anchored Relations

Carnot SHALL provide Exp6886 as an independent science root. It SHALL NOT
inherit the failed Exp6875 manifest gate. The experiment SHALL require the
qualified Exp6274 compiler, clingo as an independent solver, sufficient local
disk space, compatible cache identities, and immutable Enoki asset revisions.
A failed precondition SHALL produce a complete blocked artifact with the failed
check, expected value, and observed value. It SHALL NOT fabricate asset or
solver evidence.

The experiment SHALL pin `s-nlp/enoki-openie-encoder` and a bounded source-only
shard from `s-nlp/EnokiQA`. It SHALL record repository URLs, immutable revision
IDs, declared licenses or an explicit missing-license declaration, local cache
paths, source file hashes, projected shard row hashes, and a deterministic shard
hash. Model or dataset bytes SHALL remain outside the repository. The experiment
SHALL NOT load the encoder, run a GGUF model, claim Enoki accuracy, or treat the
unannotated EnokiQA train split as labeled evidence.

Each accepted relation SHALL use schema `anchored_relation_v1`. A record SHALL
contain a record ID, source text hash, source span, evidence span, subject,
predicate, object, polarity, normalized tuple, ASP atom, and provenance hash.
All offsets SHALL be UTF-8 byte offsets. Subject and object text SHALL match the
exact source bytes inside the evidence span. The relation mapper SHALL use a
closed, family-typed vocabulary derived from the Exp6274 propositional subset.
It SHALL reject unknown entities, unsupported predicates, invalid polarity,
duplicate records, duplicate tuples, malformed spans, provenance drift, and
non-injective tuple-to-atom maps before compilation.

Exp6886 SHALL build at least 150 deterministic fixtures. Each of graph coloring,
scheduling, non-monotonic defaults, contradictions, and cardinality constraints
SHALL contain at least 30 fixtures. Valid, omitted, contradictory, malformed,
and abstain cases SHALL be balanced within every family. Calibration and held
groups SHALL be frozen and group-disjoint. Held labels, ASP programs, answer
sets, solver receipts, and sidecar paths SHALL remain outside every future
prompt view.

Each accepted relation set SHALL compile through the qualified Exp6274 energy
compiler. The zero-energy state set SHALL equal the clingo answer-set list by
exact set equality. Each fixture SHALL retain local rule-violation receipts.
Solver timeouts and solver disagreements SHALL fail closed.

The terminal artifact SHALL be
`results/experiment_6886_enoki_exact_relation_fixture.json`. It SHALL include
`field_principles`, `preconditions_checked`, `inference_substrate`, `duration_s`,
`source_artifact_hashes`, `enoki_asset_receipts`, `asset_revision_rows`,
`asset_license_rows`, `relation_schema_version`, `closed_vocabulary_manifest`,
`rows`, `fixture_family_counts`, `calibration_group_manifest`,
`sealed_held_group_manifest`, `split_overlap_count`, `relation_to_atom_rows`,
`atom_collision_rows`, `unsupported_rows`, `solver_parity_rows`,
`rule_violation_receipts`, `prompt_nonexposure_results`,
`independent_solver_receipts`, `random_seed`, `reproducibility_checksum`,
`relation_fixture_ready_score`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. Each required field SHALL have one entry
in `field_principles`. `inference_substrate` SHALL equal
`deterministic_enoki_asset_and_exact_asp_fixture_no_llm`.
`verifier_is_oracle` SHALL be true. `verdict_class` SHALL be one of
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`; it SHALL
never be `positive`. `honest_verdict` SHALL start with `complete_`.

`relation_fixture_ready_score` SHALL be the bare integer `1` only when immutable
asset hashes, relation schema checks, family floors, split isolation, prompt and
sidecar sealing, injective mapping, solver availability, and exact solver parity
all pass. Otherwise it SHALL be `0` and `gate_check_summary` SHALL name each
failed check with its expected and observed values.

### SCENARIO-CONSTRAINT-6886-ASSETS: Immutable Assets Stay Bounded

Given the exact Enoki repository revisions and a compatible external cache,
When Exp6886 prepares its assets,
Then only the encoder files and the bounded source-only EnokiQA projection are
cached, every local hash matches its pinned receipt, and revision or hash drift
fails closed.

### SCENARIO-CONSTRAINT-6886-ANCHORS: UTF-8 Evidence Is Exact

Given records with ASCII and multibyte source text,
When Exp6886 validates their spans,
Then valid UTF-8 byte offsets reproduce the subject and object text exactly,
and invalid, character-indexed, out-of-range, or non-boundary spans are rejected.

### SCENARIO-CONSTRAINT-6886-CLOSED-MAP: Relation Mapping Is Injective

Given positive and negative relations plus unknown entities, unsupported
predicates, duplicate tuples, and colliding atoms,
When Exp6886 maps the records,
Then each supported normalized tuple has one distinct ASP atom and every unsafe
or unsupported record fails before energy compilation.

### SCENARIO-CONSTRAINT-6886-FIXTURES: Balanced Groups Stay Disjoint

Given the frozen calibration and held manifests,
When Exp6886 counts fixture families and cases,
Then at least 150 fixtures meet all family and case floors and no group occurs
in both splits.

### SCENARIO-CONSTRAINT-6886-NONEXPOSURE: Formal Sidecars Stay Hidden

Given sealed labels, ASP programs, answer sets, solver receipts, and their cache
paths,
When Exp6886 builds a future prompt view,
Then no hidden field, hidden value, or sidecar path appears in that view.

### SCENARIO-CONSTRAINT-6886-PARITY: Atoms And Energy Match Clingo

Given accepted relation sets from every fixture family,
When Exp6886 adds their atoms to the bounded ASP programs,
Then the energy compiler zero states equal clingo answer sets exactly and each
non-zero local term retains its rule and violation reason.

### SCENARIO-CONSTRAINT-6886-FAIL-CLOSED: Timeouts And Disagreements Block Readiness

Given an unavailable asset, incompatible cache, solver timeout, or changed
independent answer set,
When Exp6886 evaluates readiness,
Then the ready score remains zero and the gate summary records the exact failed
expectation and observation.

### SCENARIO-CONSTRAINT-6886-ARTIFACT: Readiness Replays From Rows

Given all asset, fixture, mapping, nonexposure, and parity rows,
When the terminal artifact is validated,
Then every required principle exists, the checksum reproduces, the score
recomputes from row evidence, the oracle boundary is explicit, and no Enoki
accuracy or LLM-inference claim appears.

## Implementation Status (REQ-CONSTRAINT-6886)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6886 and SCENARIO-CONSTRAINT-6886-* | Planned (`python/carnot/experiment_6886_enoki_exact_relation_fixture.py`; `scripts/experiments/experiment_6886_enoki_exact_relation_fixture.py`) | Planned (`tests/python/test_experiment_6886_enoki_exact_relation_fixture.py`) |

### REQ-VERIFY-6888: Independent Held Relation Qualification

Carnot SHALL provide Exp6888 as a fresh, deterministic reducer over the frozen
Exp6887 proposal artifact. The reducer SHALL run no model inference. It SHALL
require the exact Exp6886, Exp6887, Exp6274, compiler, and sealed-sidecar hashes.
It SHALL also require the qualified compiler, an independent ASP solver, and the
complete five-arm acquisition matrix. A failed precondition SHALL emit
`complete_blocked_independent_relation_qualification`. Its gate summary SHALL
name each failed check, expected value, and observed value.

The reducer SHALL score calibration rows before it opens the held sidecar. It
SHALL choose one calibration reference arm without held data. It SHALL freeze
minimum span F1, tuple precision, tuple recall, parse coverage, family floor,
perturbation floor, and exact semantic parity from that reference. It SHALL
open the held sidecar exactly once after the thresholds are frozen. It SHALL
match labels and proposals by the frozen fixture ID and exact arm ID.

The reducer SHALL reject prompt or raw-artifact leakage of held formal labels,
ASP programs, answer sets, or solver receipts. A proposal that independently
matches a source relation is not leakage. The reducer SHALL recompute UTF-8 span
grounding from source bytes. It SHALL de-duplicate proposal tuples before it
assigns credit. Empty, explicit-abstain, malformed, timeout, false-positive,
false-negative, unsupported-atom, and no-headroom cells SHALL stay explicit.
Every cell SHALL remain in parse and abstention denominators.

Each mapped proposal set SHALL compile through the qualified Exp6274 compiler.
The reducer SHALL invoke clingo with a bounded timeout on the same program. It
SHALL compare zero-energy states and answer sets by exact set equality. It SHALL
report tuple quality separately from compiler and solver parity. It SHALL also
report contradiction detection separately from ordinary tuple scoring.

The terminal artifact SHALL be
`results/experiment_6888_independent_relation_qualification.json`. It SHALL
include `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `sealed_sidecar_hashes`,
`frozen_thresholds`, `rows`, `span_metric_rows`, `tuple_metric_rows`,
`parse_coverage_rows`, `abstention_rows`, `family_rows`, `perturbation_rows`,
`asp_compilation_rows`, `solver_parity_rows`,
`reported_vs_recomputed_metrics`, `independent_solver_receipts`,
`held_leakage_count`, `eligible_arm_rows`, `qualified_relation_event_count`,
`relation_qualification_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. Each top-level field and gate field SHALL
have one principle. `inference_substrate` SHALL equal
`sealed_reduction_of_frozen_relation_outputs_no_llm`.

`relation_qualification_ready_score` SHALL equal the bare integer `1` only when
at least one arm passes every frozen held threshold. Otherwise it SHALL equal
`0`. `qualified_relation_event_count` SHALL count frozen events for passing
arms. The count SHALL NOT impose the downstream floor of 90. Because clingo is
oracle authority, a passing artifact SHALL use `verdict_class` equal to
`circular_positive`. The artifact SHALL never use `positive`.
`honest_verdict` SHALL start with `complete_`.

#### SCENARIO-VERIFY-6888-PRECONDITIONS: Drift Or Missing Cells Block

Given a changed artifact hash, changed sidecar hash, missing raw cell, changed
arm identity, unqualified compiler, or unavailable solver,
When Exp6888 checks its inputs,
Then it emits a complete blocked artifact before it opens held labels.

#### SCENARIO-VERIFY-6888-SEAL: Held Authority Opens Once Without Leakage

Given frozen calibration thresholds and a sealed held sidecar,
When the fresh reducer starts held scoring,
Then it opens that sidecar once and rejects formal held content in prompts or
raw artifact metadata.

#### SCENARIO-VERIFY-6888-SPANS: UTF-8 Offsets Are Recomputed

Given exact and shifted UTF-8 offsets,
When source grounding is scored,
Then only offsets that reproduce the source bytes receive span credit.

#### SCENARIO-VERIFY-6888-TUPLES: Duplicate Rows Receive One Credit

Given duplicate, unsupported, missing, and spurious tuples,
When tuple precision and recall are scored,
Then a unique supported tuple receives at most one true-positive credit and all
other outcomes remain explicit.

#### SCENARIO-VERIFY-6888-DENOMINATORS: Abstentions And Failures Stay Counted

Given empty, explicit-abstain, malformed, timeout, and parsed cells,
When coverage and abstention metrics are reduced,
Then every frozen cell stays in the denominator and null precision is preserved.

#### SCENARIO-VERIFY-6888-SOLVER: Unsupported Atoms And Timeouts Fail Closed

Given a tuple outside the closed map or an independent solver timeout,
When semantic validity is checked,
Then the row records the exact failure and cannot satisfy semantic parity.

#### SCENARIO-VERIFY-6888-POOLING: Families And Perturbations Cannot Hide

Given one weak family or perturbation among stronger pooled rows,
When held eligibility is computed,
Then the minimum family and perturbation rows govern their floors.

#### SCENARIO-VERIFY-6888-IDENTITY: Arm And Record IDs Are Exact

Given a substituted arm ID, duplicate cell ID, or unmatched fixture ID,
When acquisition rows are joined to labels,
Then the reducer blocks instead of pooling the substituted rows.

#### SCENARIO-VERIFY-6888-REPLAY: Aggregates Must Match Rows

Given a terminal artifact whose reported aggregate differs from row evidence,
When artifact validation replays the metrics,
Then validation reports the disagreement and readiness cannot pass.

## Implementation Status (REQ-VERIFY-6888)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-6888 and SCENARIO-VERIFY-6888-* | Implemented (`python/carnot/experiment_6888_independent_relation_qualification.py`; `scripts/experiments/experiment_6888_independent_relation_qualification.py`) | Verified (`tests/python/test_experiment_6888_independent_relation_qualification.py`; `results/experiment_6888_independent_relation_qualification.json`) |

### REQ-VERIFY-6901: Independent Model Relation Qualification

Carnot SHALL provide Exp6901 as a fresh reducer over the frozen Exp6900
acquisition artifact. The reducer SHALL run no model inference and SHALL not
repair an output. It SHALL re-run adversarial admission before it reads held
labels. It SHALL require `relation_corpus_complete_score=1`, no source
quarantine stamp, no live critical adversarial flag, exact source hashes,
complete terminal cells, sealed formal sidecars, the qualified Exp6274
compiler, and an available independent ASP solver. A failed precondition SHALL
emit `complete_blocked_independent_model_relation_qualification`. Its gate
summary SHALL name each failed check, expected value, and observed value. A
blocked run SHALL not open the held sidecar.

The reducer SHALL score calibration rows before it opens the held sidecar. It
SHALL freeze minimum parse coverage, span F1, tuple precision, tuple recall,
exact validity, family floor, perturbation floor, and maximum abstention cost
from calibration evidence only. It SHALL open the held sidecar exactly once.
It SHALL join each proposal by frozen record, arm, model, and seed identity.
It SHALL reject held labels, ASP programs, answer sets, or solver receipts in
proposal prompts or raw outputs.

The reducer SHALL recompute UTF-8 grounding from source bytes. It SHALL assign
at most one true-positive credit to each unique tuple. It SHALL retain false
positives, false negatives, malformed output, abstention, timeout, unsupported
atoms, and missing proposals. Every frozen cell SHALL remain in parse and
abstention denominators. Family and perturbation floors SHALL use the weakest
slice instead of a pooled mean.

Each supported proposal set SHALL compile with the qualified compiler. The
reducer SHALL invoke an independent solver with a bounded timeout. It SHALL
report compiler-versus-solver parity and proposal-versus-held exact answer-set
validity separately. It SHALL report proposal coverage separately from exact
admitted correctness. It SHALL emit completeness blind-spot rows for missed
held relations and SHALL not claim that exact verification proves proposal
completeness.

The terminal artifact SHALL be
`results/experiment_6901_independent_model_relation_qualification.json`. It
SHALL include `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`,
`adversarial_admission_rows`, `sealed_sidecar_hashes`, `frozen_thresholds`,
`rows`, `span_metric_rows`, `tuple_metric_rows`, `parse_coverage_rows`,
`abstention_rows`, `family_rows`, `perturbation_rows`,
`asp_compilation_rows`, `solver_parity_rows`,
`completeness_blind_spot_rows`, `reported_vs_recomputed_metrics`,
`independent_solver_receipts`, `held_leakage_count`,
`model_eligible_arm_rows`, `rule_control_rows`,
`qualified_model_relation_event_count`,
`model_relation_qualification_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. Each artifact and gate field SHALL have
one principle. `inference_substrate` SHALL equal
`fresh_process_sealed_relation_reduction_no_llm`.

Only GGUF and Enoki arms SHALL be model eligible. The lexical rule arm MAY be
reported as a diagnostic control, but it SHALL NOT contribute to
`qualified_model_relation_event_count` or satisfy model readiness.
`model_relation_qualification_ready_score` SHALL equal the bare integer `1`
only when at least one model-produced arm passes every frozen held threshold
and contributes at least 90 exact-admitted relation events. Otherwise it SHALL
equal `0`. Because the exact verifier is oracle authority, a passing artifact
SHALL use `verdict_class=circular_positive`. The artifact SHALL never use
`positive`. `honest_verdict` SHALL start with `complete_`.

#### SCENARIO-VERIFY-6901-ADMISSION: Drift Or A Flagged Source Blocks

Given source hash drift, a quarantine stamp, a live critical adversarial flag,
an incomplete corpus, or a missing terminal cell,
When Exp6901 checks admission,
Then it emits a complete blocked artifact before held labels open.

#### SCENARIO-VERIFY-6901-SEAL: Held Authority Opens Once Without Leakage

Given frozen calibration thresholds and a sealed held sidecar,
When the reducer starts held scoring,
Then it opens that sidecar once and rejects formal held content in proposal
prompts or raw outputs.

#### SCENARIO-VERIFY-6901-IDENTITY: Record Arm Model And Seed IDs Are Exact

Given a substituted model, seed, arm, cell, or record identity,
When acquisition cells are joined to frozen sources,
Then the reducer blocks instead of pooling the substituted cell.

#### SCENARIO-VERIFY-6901-SPANS: UTF-8 Offsets Are Recomputed

Given exact and shifted UTF-8 offsets,
When source grounding is scored,
Then only offsets that reproduce the source bytes receive span credit.

#### SCENARIO-VERIFY-6901-TUPLES: Duplicate Rows Receive One Credit

Given duplicate, unsupported, missing, and spurious tuples,
When tuple quality is scored,
Then a unique supported tuple receives at most one true-positive credit and
all other outcomes remain explicit.

#### SCENARIO-VERIFY-6901-DENOMINATORS: Abstention And Failure Stay Counted

Given empty, explicit-abstain, malformed, timeout, and parsed cells,
When coverage and abstention metrics are reduced,
Then every frozen cell stays in the denominator and abstention cost remains
separate from tuple precision.

#### SCENARIO-VERIFY-6901-SOLVER: Unsupported Atoms And Timeouts Fail Closed

Given an atom outside the closed map or an independent solver timeout,
When exact validity is checked,
Then the exact row records the failure and cannot satisfy exact parity.

#### SCENARIO-VERIFY-6901-POOLING: Weak Slices Cannot Hide

Given one weak family or perturbation among stronger pooled rows,
When held eligibility is computed,
Then the minimum family and perturbation rows govern their floors.

#### SCENARIO-VERIFY-6901-MODEL-ONLY: A Rule-Only Pass Is Not Ready

Given a passing lexical rule arm and no passing GGUF or Enoki arm,
When readiness and qualified events are computed,
Then both outgoing model fields remain zero.

#### SCENARIO-VERIFY-6901-REPLAY: Aggregates Must Match Rows

Given a terminal artifact whose reported aggregate differs from row evidence,
When artifact validation replays the metrics,
Then validation reports the disagreement and readiness cannot pass.

## Implementation Status (REQ-VERIFY-6901)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-6901 and SCENARIO-VERIFY-6901-* | Implemented (`python/carnot/experiment_6901_independent_model_relation_qualification.py`; `scripts/experiments/experiment_6901_independent_model_relation_qualification.py`) | Verified (`tests/python/test_experiment_6901_independent_model_relation_qualification.py`; `results/experiment_6901_independent_model_relation_qualification.json`) |

### REQ-CONSTRAINT-6913: Relation Source And Tuple Qualification

Carnot SHALL provide Exp6913 as a deterministic source-facing reducer over the
immutable Exp6900 cells admitted by Exp6912. The reducer SHALL require
`clean_relation_corpus_ready_score=1`, exact Exp6886, Exp6900, and Exp6912
artifact hashes, all 1,400 expected cell identities, and zero held ASP sidecar
access. A failed precondition SHALL emit
`complete_blocked_relation_source_tuple_qualification`. The gate summary SHALL
name each failed check with its expected and observed values.

The reducer SHALL reconstruct public source text from hash-verified saved bytes.
It SHALL reconstruct proposal lines from hash-verified saved output bytes. It
SHALL not import Exp6901 aggregates, open an ASP held sidecar, run model
inference, infer model semantics, or repair malformed output.

Every expected cell SHALL produce one terminal row. Each row SHALL retain the
source byte offsets, exact quoted source bytes, parsed tuple fields, tuple arity
and type checks, entity membership checks, relation direction checks, duplicate
status, omission status, abstention status, and one terminal source-grounding
label. UTF-8 byte drift, partial-entity substrings, normalized-text
substitution, parser bypass, invalid tuple arity, unknown entities, reversed
relations, duplicate tuples, omitted required tuples, and false abstentions
SHALL fail source grounding. Parser failures and malformed output SHALL remain
failures.

The reducer SHALL score all required GGUF, Enoki, and lexical rule-control
cells. A control result SHALL stay in its own arm, model, family, and seed
denominators. It SHALL never satisfy a GGUF row or denominator. Proposal
coverage SHALL measure cells with at least one syntactically parsed proposal.
Source-grounded correctness SHALL measure cells whose complete proposal set is
exactly anchored, typed, directed, unique, and complete. Both metrics SHALL use
all expected cells in their denominators.

The reducer SHALL compute per-arm, per-model, per-family, and per-seed summaries
from terminal rows. Each summary SHALL include exact numerators, exact
denominators, rates, and 95 percent Wilson intervals. Reported aggregate values
SHALL equal a fresh row replay.

The terminal artifact SHALL be
`results/experiment_6913_relation_source_tuple_qualification.json`. It SHALL
include `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `rows`, `source_offset_rows`,
`source_byte_identity_rows`, `parser_rows`, `tuple_type_rows`,
`entity_anchor_rows`, `relation_direction_rows`, `omission_rows`,
`duplicate_rows`, `abstention_rows`, `arm_summary_rows`, `model_summary_rows`,
`family_summary_rows`, `seed_summary_rows`, `wilson_interval_rows`,
`proposal_coverage_by_arm`, `source_grounded_correctness_by_arm`,
`held_sidecar_access_count`, `model_inference_call_count`,
`reported_vs_recomputed_metrics`, `random_seed`, `reproducibility_checksum`,
`source_tuple_shard_ready_score`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. Each required field and gate field SHALL
have one principle. `inference_substrate` SHALL equal
`deterministic_cpu_source_tuple_qualification_no_llm`.
`held_sidecar_access_count` and `model_inference_call_count` SHALL be bare
integer zero. `verifier_is_oracle` SHALL be true. `verdict_class` SHALL be one
of `circular_positive`, `null`, `blocked`, `disqualified`, or `partial`. It
SHALL never be `positive`. `honest_verdict` SHALL start with `complete_`.

`source_tuple_shard_ready_score` SHALL be the bare integer `1` only when each
expected cell has one terminal row, all hashes replay, held-sidecar access is
zero, all aggregates match rows, and every arm and family has a qualification
decision. The score states that the shard is complete. It does not state that
every proposal is correct. Otherwise the score SHALL be `0`.

#### SCENARIO-CONSTRAINT-6913-PRECONDITIONS: Drift Or Missing Cells Block

Given source or fixture hash drift, a non-ready Exp6912 receipt, a missing or
duplicate identity, or nonzero held-sidecar access,
When Exp6913 checks its inputs,
Then it emits a complete blocked artifact before source-tuple scoring.

#### SCENARIO-CONSTRAINT-6913-BYTES: UTF-8 And Text Identity Stay Exact

Given multibyte source text, shifted byte offsets, a partial entity substring,
or a normalized substitute for the quoted bytes,
When Exp6913 checks source grounding,
Then only the exact complete entity bytes at valid UTF-8 boundaries pass.

#### SCENARIO-CONSTRAINT-6913-PARSER: Parser Bypass And Malformed Tuples Fail

Given a parser-input hash mismatch, parser bypass, malformed protocol line, or
tuple with invalid arity or field types,
When Exp6913 replays the saved output bytes,
Then the cell keeps a parser or tuple failure and cannot be source-grounded.

#### SCENARIO-CONSTRAINT-6913-ENTITIES: Entities And Direction Are Typed

Given an unknown entity, unsupported predicate, invalid polarity, or reversed
subject and object,
When Exp6913 checks the proposed tuple,
Then entity membership or relation direction fails.

#### SCENARIO-CONSTRAINT-6913-COMPLETENESS: Duplicates Omissions And Abstentions Remain

Given a duplicate tuple, omitted required tuple, empty output, or explicit
abstention where the source contains a required relation,
When Exp6913 scores the cell,
Then the terminal row records the failure without imputation or duplicate
credit.

#### SCENARIO-CONSTRAINT-6913-DENOMINATORS: Controls Cannot Fill GGUF Cells

Given model and control rows from the same source family,
When Exp6913 builds arm, model, family, and seed summaries,
Then every expected cell remains in its own exact denominator.

#### SCENARIO-CONSTRAINT-6913-AGGREGATES: Metrics Replay From Rows

Given reported proposal coverage or source-grounded correctness that differs
from terminal rows,
When Exp6913 validates the artifact,
Then it reports the disagreement and readiness remains zero.

#### SCENARIO-CONSTRAINT-6913-READINESS: Completion Is Not Universal Correctness

Given all 1,400 terminal decisions, exact hashes, zero held access, complete
arm and family decisions, and matching aggregates,
When Exp6913 computes readiness,
Then `source_tuple_shard_ready_score=1` even when some models fail grounding.

## Implementation Status (REQ-CONSTRAINT-6913)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6913 and SCENARIO-CONSTRAINT-6913-* | Planned (`python/carnot/experiment_6913_relation_source_tuple_qualification.py`; `scripts/experiments/experiment_6913_relation_source_tuple_qualification.py`) | Planned (`tests/python/test_experiment_6913_relation_source_tuple_qualification.py`) |

### REQ-CONSTRAINT-6914: Relation ASP And Isomorphic Qualification

Carnot SHALL provide Exp6914 as a deterministic semantic reducer over the
immutable Exp6900 cells admitted by Exp6912. The reducer SHALL hash all public
inputs before it opens a sealed exact-label sidecar. It SHALL require
`clean_relation_corpus_ready_score=1`, exact Exp6274, Exp6886, Exp6900, and
Exp6912 artifacts, exact sealed-sidecar hashes, a qualified bounded ASP
compiler, and an independent exact solver. A failed precondition SHALL emit
`complete_blocked_relation_asp_isomorphic_qualification`. The gate summary
SHALL name each failed check with its expected and observed values. A blocked
run SHALL not open a sealed sidecar.

The reducer SHALL not import Exp6901 aggregates or Exp6913 labels. It SHALL not
run model inference. It SHALL open each required sealed sidecar at most once
after public admission passes. It SHALL reject held ASP programs, answer sets,
and solver receipts in proposal prompts or outputs. It SHALL preserve every
nonparseable cell in proposal-coverage denominators.

Each accepted tuple SHALL map through the frozen relation-to-atom vocabulary.
An unmapped tuple or escaped atom SHALL fail closed. Each supported proposal
program SHALL run through the qualified bounded compiler and a second exact
engine. Every terminal row SHALL include the program, expected effect, observed
effect, answer-set or model count, parity result, and receipts from both
engines. Proposal coverage, exact atom validity, solver parity, expected-effect
agreement, and isomorphic invariance SHALL remain separate decisions.

The reducer SHALL generate one deterministic base row and six paired rows for
each immutable cell. The pairs SHALL cover injective entity renaming,
syntax-preserving relation paraphrase, relation reversal, contradiction
injection, relation omission, and solution-space restructuring. Renaming and
paraphrase SHALL preserve the exact effect after canonical projection.
Reversal, contradiction, and omission SHALL record their required directional
effect when applicable. Restructuring SHALL preserve satisfiability and
projected models while changing the model count for satisfiable programs.

The reducer SHALL report exact atom validity, solver parity, isomorphic
invariance, restructuring shortcut rate, and completeness blind spots by arm,
model, family, seed, and perturbation. Reported aggregate values SHALL equal a
fresh replay from terminal rows. Each arm and family SHALL receive a terminal
qualification decision.

The terminal artifact SHALL be
`results/experiment_6914_relation_asp_isomorphic_qualification.json`. It SHALL
include `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `sealed_sidecar_hashes`, `rows`,
`asp_compilation_rows`, `bounded_vocabulary_rows`, `primary_solver_rows`,
`independent_solver_rows`, `solver_parity_rows`, `entity_renaming_rows`,
`paraphrase_rows`, `reversal_rows`, `contradiction_rows`, `omission_rows`,
`restructuring_rows`, `arm_summary_rows`, `model_summary_rows`,
`family_summary_rows`, `seed_summary_rows`, `perturbation_summary_rows`,
`proposal_coverage_by_arm`, `exact_atom_validity_by_arm`,
`isomorphic_invariance_by_arm`, `completeness_blind_spot_rows`,
`solver_disagreement_count`, `held_leakage_count`,
`model_inference_call_count`, `reported_vs_recomputed_metrics`, `random_seed`,
`reproducibility_checksum`, `asp_isomorphic_shard_ready_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. Each required field and gate field SHALL have one principle.
`inference_substrate` SHALL equal
`deterministic_cpu_asp_isomorphic_qualification_no_llm`.

`solver_disagreement_count`, `held_leakage_count`, and
`model_inference_call_count` SHALL be bare integer zero for readiness.
`verifier_is_oracle` SHALL be true. `verdict_class` SHALL be one of
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`. It SHALL
never be `positive`. `honest_verdict` SHALL start with `complete_`.

`asp_isomorphic_shard_ready_score` SHALL equal the bare integer `1` only when
every expected cell and perturbation has one terminal row, both engines agree,
sealed access follows the protocol, aggregates replay, and every arm and family
has a qualification decision. The score states that the shard is complete. It
does not state that every proposal is semantically correct. Otherwise the score
SHALL equal `0`.

#### SCENARIO-CONSTRAINT-6914-PRECONDITIONS: Drift Blocks Before Sealed Access

Given compiler drift, public artifact drift, a non-ready Exp6912 receipt, a
missing exact engine, or sealed-sidecar hash drift,
When Exp6914 checks admission,
Then it emits a complete blocked artifact with exact gate evidence and does not
open held labels after a public-input failure.

#### SCENARIO-CONSTRAINT-6914-VOCABULARY: Escaped Atoms Fail Closed

Given a parseable tuple outside the frozen relation-to-atom vocabulary,
When Exp6914 compiles the proposal,
Then it records an unsupported atom and does not give the row exact atom credit.

#### SCENARIO-CONSTRAINT-6914-SOLVERS: Parity Is Separate From Expected Effect

Given stable-model disagreement between the bounded compiler and independent
solver,
When Exp6914 compares their exact outputs,
Then solver parity fails. If both engines share the same wrong output, parity
can pass but sealed expected-effect agreement still fails.

#### SCENARIO-CONSTRAINT-6914-RENAMING: Entity Maps Are Injective

Given a deterministic entity renaming,
When Exp6914 applies it to a program and its expected models,
Then the atom map is injective and projected exact effects remain invariant. A
non-injective map fails closed.

#### SCENARIO-CONSTRAINT-6914-PARAPHRASE: Only Frozen Paraphrases Preserve Meaning

Given a registered syntax-preserving relation paraphrase,
When Exp6914 canonicalizes the paired tuple,
Then it maps to the same atom and preserves the exact effect. A semantic change
does not receive paraphrase invariance credit.

#### SCENARIO-CONSTRAINT-6914-REVERSAL: Directional Relations Must Change

Given a supported directional tuple,
When Exp6914 reverses its subject and object,
Then the reversed tuple leaves the frozen map and records the required
directional change. A reversal no-op fails the pair check.

#### SCENARIO-CONSTRAINT-6914-CONTRADICTION: Opposite Polarity Makes Unsat

Given a satisfiable relation program,
When Exp6914 injects both polarities for its bounded relation,
Then both exact engines report unsatisfiable. A contradiction no-op fails the
pair check.

#### SCENARIO-CONSTRAINT-6914-OMISSION: Relation Facts Are Removed

Given a proposal with at least one supported relation fact,
When Exp6914 creates the omission pair,
Then every proposal relation fact is absent and the exact effect is recomputed.
An omission no-op fails the pair check.

#### SCENARIO-CONSTRAINT-6914-RESTRUCTURING: Model Count Cannot Be A Shortcut

Given a satisfiable proposal program,
When Exp6914 adds a bounded independent choice from the frozen theory
vocabulary,
Then projected models and satisfiability stay invariant while the model count
changes. A count-based decision or restructuring no-op fails the pair check.

#### SCENARIO-CONSTRAINT-6914-SEAL: Held Formal Content Does Not Leak

Given sealed ASP programs, answer sets, and solver receipts,
When Exp6914 scans prompts and raw proposal outputs after one sealed open,
Then direct formal leakage is counted and readiness remains zero.

#### SCENARIO-CONSTRAINT-6914-AGGREGATES: Metrics Replay From Rows

Given a reported metric that differs from terminal row evidence,
When Exp6914 validates the artifact,
Then it reports aggregate disagreement and readiness remains zero.

#### SCENARIO-CONSTRAINT-6914-READINESS: Completion Is Not Universal Correctness

Given all cell and perturbation decisions, exact solver parity, sealed access,
matching aggregates, and arm and family decisions,
When Exp6914 computes readiness,
Then `asp_isomorphic_shard_ready_score=1` even when some proposals fail exact
effect qualification.

## Implementation Status (REQ-CONSTRAINT-6914)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6914 and SCENARIO-CONSTRAINT-6914-* | Planned (`python/carnot/experiment_6914_relation_asp_isomorphic_qualification.py`; `scripts/experiments/experiment_6914_relation_asp_isomorphic_qualification.py`) | Planned (`tests/python/test_experiment_6914_relation_asp_isomorphic_qualification.py`) |

### REQ-CONSTRAINT-6915: Qualified Relation Event Bank

Carnot SHALL provide Exp6915 as a deterministic merge of the Exp6913 source
qualification shard and the Exp6914 semantic qualification shard. The merge
SHALL require both structured readiness scores to equal one. It SHALL require
the exact pinned shard hashes, complete immutable cell manifests, all seven
required perturbation identities for each cell, and zero fresh critical flags
from the current adversarial verifier. A failed precondition SHALL emit
`complete_blocked_qualified_relation_event_bank`. Its gate summary SHALL name
each failed check and record the exact expected and observed values.

The merge SHALL join each source cell to each semantic row by immutable cell
identity and perturbation identity. It SHALL require exactly one source row and
exactly one semantic row for every expected join. It SHALL reject missing,
duplicate, cross-model, cross-seed, cross-arm, cross-family, cross-fixture,
cross-split, changed-perturbation, flagged, unsupported, nonterminal, or
solver-disagreed evidence. An expected relation reversal that leaves the frozen
directional vocabulary SHALL count as the required directional rejection. It
SHALL not count as an unsupported base relation.

The merge SHALL recompute cell eligibility from row components. It SHALL not
import `source_grounded_correct`, `source_grounding_label`,
`exact_outcome_decision`, a shard summary decision, or a shard headline
eligibility value. A model cell is eligible only when its saved output and
parser replay pass, its source bytes and offsets pass, its complete tuple set is
typed and grounded, its base relation compiles to exact atoms, both exact
engines agree, and every required perturbation has its specified behavior.

The merge SHALL emit one eligibility row for every immutable cell. It SHALL
retain every rejected model-produced cell. It SHALL retain Enoki controls and
lexical rule controls in separate control row sets. A control row SHALL never
enter the admitted event count. Model, seed, constraint-family, and source-group
summaries SHALL be computed directly from eligibility rows. No pooled model or
family aggregate SHALL hide a failing required model family or constraint
family.

The readiness gate SHALL require at least 90 admitted model-produced cells. It
SHALL require admitted cells from all five constraint families. It SHALL
require at least ten admitted cells from each required model family. Every
reported source group SHALL have positive headroom, which means at least one
admitted and at least one rejected model-produced cell. The gate SHALL require
`control_substitution_count=0`, complete joins, and exact aggregate replay.

The terminal artifact SHALL be
`results/experiment_6915_qualified_relation_event_bank.json`. It SHALL include
`field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `rows`, `join_rows`,
`missing_join_rows`, `duplicate_join_rows`, `eligibility_rows`,
`admitted_event_rows`, `rejected_event_rows`, `rejection_reason_rows`,
`model_summary_rows`, `family_summary_rows`, `seed_summary_rows`,
`enoki_control_rows`, `rule_control_rows`, `control_substitution_count`,
`source_group_headroom_rows`, `admitted_event_bank_manifest`,
`fresh_adversarial_rows`, `reported_vs_recomputed_metrics`, `random_seed`,
`reproducibility_checksum`, `qualified_model_relation_event_count`,
`qualified_relation_event_bank_ready_score`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`. Every required
field and gate field SHALL have one principle.

`inference_substrate` SHALL equal
`deterministic_cpu_qualification_merge_no_llm`. The qualified event count SHALL
equal the number of admitted model-produced rows. The readiness score SHALL be
the bare integer one only when every join and threshold passes. Otherwise it
SHALL be zero. `verifier_is_oracle` SHALL be true. `verdict_class` SHALL be one
of `circular_positive`, `null`, `blocked`, `disqualified`, or `partial`. It
SHALL never be `positive`. `honest_verdict` SHALL start with `complete_`.

#### SCENARIO-CONSTRAINT-6915-PRECONDITIONS: Shard Drift Blocks The Merge

Given a missing readiness score, changed shard hash, incomplete cell manifest,
or fresh critical adversarial flag,
When Exp6915 checks both shards,
Then it emits a complete blocked artifact with exact gate evidence.

#### SCENARIO-CONSTRAINT-6915-JOIN: Joins Are Exact And One To One

Given a missing semantic row, duplicate cell identity, duplicate perturbation,
cross-model row, cross-seed row, or changed perturbation identity,
When Exp6915 builds join evidence,
Then the affected join is rejected and the mismatch remains visible.

#### SCENARIO-CONSTRAINT-6915-CONTROLS: Controls Cannot Substitute For Models

Given qualified Enoki or lexical rule rows and failed model rows,
When Exp6915 builds the admitted bank,
Then controls remain in separate summaries and the model event count stays
unchanged.

#### SCENARIO-CONSTRAINT-6915-ELIGIBILITY: Decisions Replay From Components

Given an inverted shard decision with unchanged component evidence,
When Exp6915 recomputes eligibility,
Then the component conjunction determines admission and the imported decision
has no effect.

#### SCENARIO-CONSTRAINT-6915-POOLING: Weak Families Stay Visible

Given a pooled count above 90 but one required model family below ten or one
constraint family with no admitted row,
When Exp6915 evaluates readiness,
Then readiness remains zero and the failing family is named by a gate row.

#### SCENARIO-CONSTRAINT-6915-ROWS: Omission Cannot Change The Headline

Given an omitted admitted or rejected eligibility row,
When Exp6915 validates the artifact,
Then row coverage fails and the reported event count cannot remain valid.

#### SCENARIO-CONSTRAINT-6915-AGGREGATES: Summaries Replay From Rows

Given a changed model, family, seed, headroom, rejection, count, or readiness
summary,
When Exp6915 replays terminal rows,
Then aggregate agreement fails and readiness remains zero.

#### SCENARIO-CONSTRAINT-6915-READINESS: A Qualified Bank Is Not A Model Score

Given complete joins, exact row replay, all event floors, positive source-group
headroom, and no control substitution,
When Exp6915 computes the terminal state,
Then the bank receives a circular-positive readiness receipt without reporting
a new model-quality score.

## Implementation Status (REQ-CONSTRAINT-6915)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6915 and SCENARIO-CONSTRAINT-6915-* | Implemented (`python/carnot/experiment_6915_qualified_relation_event_bank.py`; `scripts/experiments/experiment_6915_qualified_relation_event_bank.py`) | Verified (`tests/python/test_experiment_6915_qualified_relation_event_bank.py`) |

### REQ-CONSTRAINT-6919: Exact Prefix-Viability Relation Fixture

Carnot SHALL provide Exp6919 as a deterministic, train-free relation-program
fixture. The fixture SHALL define plain relation-line actions and bounded
relation-program states. It SHALL not define or activate a token grammar, JSON
schema decoder, repair prompt, finite answer ID, or per-instance answer menu.

The experiment SHALL require the qualified Exp6274 ASP compiler artifact and
source. It SHALL require the exact frozen Exp6886 relation-fixture identities.
It SHALL require the current exclusion manifest and two available exact
engines with different implementations. A failed precondition SHALL write
`complete_blocked_exact_prefix_viability_fixture`. The gate summary SHALL name
each failed check. Each failed check SHALL include its expected and observed
values.

The fixture SHALL contain at least 120 prefix cases. It SHALL cover graph
coloring, scheduling, non-monotonic defaults, contradictions, and cardinality
constraints. It SHALL include positive, immediately impossible, late-failure,
ambiguous, duplicate, contradiction, unsupported, and no-headroom cases. Each
family and case type SHALL meet its declared floor. Source groups SHALL be
assigned to train-free canary and held partitions. No source group SHALL occur
in both partitions.

The in-loop engine SHALL decide whether a partial program has any valid
completion within the remaining line bound. It SHALL use direct bounded Python
enumeration and family predicates. The final engine SHALL use clingo stable
model solving to judge each complete program. The final engine SHALL also
derive prefix extendability by checking all bounded completions. The experiment
SHALL refuse readiness after any engine disagreement.

Each prefix case SHALL record its ordered plain-text prefix, branch factor,
feasible branches, rejected branches, exact decision, proof or completion
witness, measured decision latency, and final available headroom. Branch rows
SHALL retain one source, prefix, branch, and exact decision per row. The
semantic result SHALL be set-based. Reordering distinct lines SHALL preserve
the exact decision. Duplicate lines SHALL remain invalid.

The terminal artifact SHALL be
`results/experiment_6919_exact_prefix_viability_fixture.json`. It SHALL include
`field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `fixture_manifest`, `rows`,
`prefix_case_rows`, `source_group_split_rows`, `in_loop_exact_engine_rows`,
`final_exact_engine_rows`, `exact_engine_parity_rows`, `positive_rows`,
`impossible_rows`, `late_failure_rows`, `ambiguous_rows`, `duplicate_rows`,
`contradiction_rows`, `unsupported_rows`, `no_headroom_rows`,
`branch_factor_rows`, `feasible_branch_rows`, `rejected_branch_rows`,
`witness_rows`, `latency_rows`, `retired_mechanism_activation_count`,
`model_inference_call_count`, `exact_engine_disagreement_count`, `random_seed`,
`reproducibility_checksum`, `prefix_viability_canary_ready_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL contain one principle for each
required field and the readiness score.

`inference_substrate` SHALL equal
`deterministic_cpu_dual_exact_prefix_canary_no_llm`. The retired-mechanism and
model-inference counts SHALL be bare integer zero. `verifier_is_oracle` SHALL
be true. `verdict_class` SHALL be one of `circular_positive`, `null`, `blocked`,
`disqualified`, or `partial`. It SHALL never be `positive`.
`honest_verdict` SHALL be terminal and start with `complete_`.

`prefix_viability_canary_ready_score` SHALL equal the bare integer one only
when all preconditions pass, both engines agree on every canary and held case,
all family and case floors pass, split overlap is zero, all required telemetry
is present, and retired mechanisms have zero activations. Otherwise the score
SHALL equal zero.

#### SCENARIO-CONSTRAINT-6919-EMPTY: Empty Prefix Has Exact Completions

Given an empty bounded relation program,
When both engines compute extendability,
Then both report an extendable and ambiguous prefix with a completion witness.

#### SCENARIO-CONSTRAINT-6919-COMPLETION: Valid Completion Passes Both Engines

Given a complete relation program that satisfies its family constraints,
When both engines evaluate it,
Then both report exact validity with zero available headroom.

#### SCENARIO-CONSTRAINT-6919-IMPOSSIBLE: Immediate Impossibility Fails Closed

Given a supported relation that violates a unary family constraint,
When the first line enters the prefix,
Then both engines report no valid bounded completion.

#### SCENARIO-CONSTRAINT-6919-LATE: A Late Contradiction Stays Visible

Given an extendable first line and a later line for the same subject,
When the later line enters the prefix,
Then the prior prefix stays recorded as extendable and the new prefix is
impossible.

#### SCENARIO-CONSTRAINT-6919-DUPLICATE: Duplicate Relations Are Invalid

Given the same plain relation line twice,
When either engine checks the prefix,
Then the prefix is rejected as a duplicate relation.

#### SCENARIO-CONSTRAINT-6919-UNSUPPORTED: Unsupported Atoms Fail Closed

Given a parseable line outside the frozen relation vocabulary,
When either engine checks the prefix,
Then it reports an unsupported atom and does not construct a completion.

#### SCENARIO-CONSTRAINT-6919-AMBIGUOUS: Multiple Completions Are Counted

Given a partial program with more than one valid bounded completion,
When both engines enumerate its completion set,
Then both report the same completion count and preserve one witness.

#### SCENARIO-CONSTRAINT-6919-NO-HEADROOM: Invalid Full Prefix Cannot Extend

Given a full-length program that violates a cross-relation constraint,
When the engines compute available headroom,
Then headroom is zero and both engines report no valid completion.

#### SCENARIO-CONSTRAINT-6919-DISAGREEMENT: Engine Disagreement Blocks Readiness

Given an injected final-engine decision that differs from direct enumeration,
When the artifact gate compares the receipts,
Then disagreement is nonzero, readiness is zero, and the verdict is blocked.

#### SCENARIO-CONSTRAINT-6919-ORDER: Relation Meaning Is Set-Based

Given two valid distinct relation lines in either order,
When both engines check both programs,
Then validity is unchanged and each ordered prefix remains in the receipts.

#### SCENARIO-CONSTRAINT-6919-RETIRED: Retired Mechanisms Cannot Activate

Given any nonzero schema-decoder, repair-reprompt, finite-ID, or answer-menu
activation count,
When the readiness gate runs,
Then readiness is zero and the activation appears in the gate summary.

## Implementation Status (REQ-CONSTRAINT-6919)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-CONSTRAINT-6919 and SCENARIO-CONSTRAINT-6919-* | Implemented (`python/carnot/experiment_6919_exact_prefix_viability_fixture.py`; `scripts/experiments/experiment_6919_exact_prefix_viability_fixture.py`) | Verified (`tests/python/test_experiment_6919_exact_prefix_viability_fixture.py`) |

### REQ-VERIFY-6926: Span-First Relation Fixture

Carnot SHALL provide Exp6926 as a deterministic source-grounded relation
fixture. The fixture SHALL use immutable UTF-8 source bytes. It SHALL cover
graph coloring, scheduling, allocation, precedence, and exclusion. Each family
SHALL contain positive, negative, and unknown cases.

Each proposal SHALL contain exactly three plain-text lines. Line one SHALL be
`SPAN_A: <verbatim source span>`. Line two SHALL be
`SPAN_B: <verbatim source span>`. Line three SHALL be
`RELATION: A -> <relation phrase> -> B`. The parser SHALL resolve both copied
spans against the immutable source bytes before it interprets the relation
phrase. The protocol SHALL not use a JSON schema, constrained decoding, repair
reprompts, finite answer IDs, or a model judge.

The parser SHALL preserve exact UTF-8 byte offsets. It SHALL reject absent
spans, malformed lines, invalid UTF-8 boundaries, overlapping spans, reverse
direction, and ambiguous repeated spans. An explicit occurrence selector MAY
resolve a repeated span. The selector SHALL be source-grounded and SHALL not
act as a finite answer ID. The parser SHALL preserve each invalid proposal and
its exact rejection reason.

The fixture SHALL register a closed set of canonical relation phrases and
meaning-preserving paraphrases. Each accepted phrase SHALL map to one directed
canonical tuple. Negated source statements SHALL map to negative tuples.
Statements that do not assert or deny a relation SHALL map to unknown tuples.
The exact semantic checker SHALL reject a label that conflicts with the source
case or direction.

Each accepted tuple SHALL compile to a bounded ASP program. A positive tuple
SHALL assert its canonical atom. A negative tuple SHALL assert its explicit
negative atom. An unknown tuple SHALL assert neither atom. The qualified
bounded ASP engine and the independent clingo engine SHALL return the same
models. An injective entity renaming SHALL preserve the canonical projected
effect. A frozen paraphrase SHALL preserve the tuple and ASP effect.

The calibration and held-out partitions SHALL be assigned by a fixed seed and
content hash. A source group SHALL occur in exactly one partition. Live prompt
rows SHALL omit held-out expected tuples, labels, ASP programs, and effects.
The artifact SHALL record a held-out hash manifest without exposing those
expected outputs in live prompts.

Exp6926 SHALL check readable clean source rows, the exact relation checker, the
exact ASP checker, the isomorphism checker, and stable UTF-8 handling before it
builds the fixture. A failed precondition SHALL emit
`complete_blocked_span_first_relation_fixture`. Its gate summary SHALL name
the failed check, expected value, and observed value.

The terminal artifact SHALL be
`results/experiment_6926_span_first_relation_fixture.json`. It SHALL include
`field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `rows`, `fixture_rows`,
`protocol_rows`, `utf8_offset_rows`, `ambiguity_rows`,
`relation_family_rows`, `label_balance_rows`, `asp_effect_rows`,
`isomorphism_rows`, `mutation_rows`, `partition_rows`,
`heldout_hash_manifest`, `parser_rejection_rows`, `random_seed`,
`reproducibility_checksum`, `span_relation_fixture_ready_score`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. The artifact SHALL emit one row per fixture example and one
row per mutation or replay. Aggregate-only evidence SHALL not satisfy this
requirement. `field_principles` SHALL contain one principle for each required
field and the readiness score.

`inference_substrate` SHALL equal
`deterministic_cpu_span_first_fixture_no_llm`. `verifier_is_oracle` SHALL be
true. `verdict_class` SHALL be one of `circular_positive`, `null`, `blocked`,
`disqualified`, or `partial`. `honest_verdict` SHALL be terminal and start
with `complete_`.

`span_relation_fixture_ready_score` SHALL equal the bare integer one only
when all preconditions pass. Every family and label cell SHALL exist. All
partition and content hashes SHALL replay. Ambiguity SHALL be explicit. Both
exact ASP engines SHALL agree. All semantic, paraphrase, and isomorphism checks
SHALL pass. Every required row SHALL exist. Otherwise the score SHALL equal
zero.

#### SCENARIO-VERIFY-6926-PRECONDITIONS: Unsafe Inputs Block

Given an unreadable source row, changed checker, unavailable exact engine, or
unstable UTF-8 byte result,
When Exp6926 checks its inputs,
Then it writes a complete blocked artifact with expected and observed values.

#### SCENARIO-VERIFY-6926-PROTOCOL: Exactly Three Lines Are Required

Given a proposal with a missing, extra, reordered, or malformed line,
When the deterministic parser reads it,
Then it rejects the proposal and records one exact parser reason.

#### SCENARIO-VERIFY-6926-BYTES: Spans Resolve Before Labels

Given Unicode source text and two verbatim copied spans,
When the parser reads the proposal,
Then it resolves exact UTF-8 byte offsets before it interprets the relation.

#### SCENARIO-VERIFY-6926-AMBIGUITY: Repeated Spans Need An Occurrence

Given a copied span that occurs more than once,
When the proposal omits an occurrence selector,
Then the parser rejects it as ambiguous and records every candidate offset.

#### SCENARIO-VERIFY-6926-SPAN-FAILURES: Invalid Anchors Fail Closed

Given an absent span, overlapping spans, or reversed source direction,
When the parser resolves anchors,
Then it rejects the proposal before relation semantics run.

#### SCENARIO-VERIFY-6926-SEMANTICS: Labels Match Source Meaning

Given positive, negated, and unknown source cases in each relation family,
When the exact relation checker compares the proposal to its source case,
Then only the matching directed canonical tuple passes.

#### SCENARIO-VERIFY-6926-PARAPHRASE: Frozen Phrases Preserve Tuples

Given a registered meaning-preserving paraphrase,
When the parser canonicalizes its relation phrase,
Then the canonical tuple and exact ASP effect match the base phrase.

#### SCENARIO-VERIFY-6926-ASP: Exact Engines Agree On Consequences

Given any accepted fixture tuple,
When both exact ASP engines solve its bounded program,
Then they return equal models and the result matches the frozen consequence.

#### SCENARIO-VERIFY-6926-ISOMORPHISM: Entity Renaming Preserves Effects

Given an injective entity renaming for a fixture row,
When the exact checker solves the renamed program,
Then the projected result equals the base result. A non-injective map fails.

#### SCENARIO-VERIFY-6926-PARTITIONS: Held Outputs Stay Out Of Prompts

Given seed-frozen calibration and held-out groups,
When Exp6926 builds live prompt rows,
Then no held-out expected label, tuple, ASP program, or effect is present.

#### SCENARIO-VERIFY-6926-REPLAY: Rows And Hashes Recompute

Given the terminal artifact in a fresh process,
When Exp6926 replays byte offsets, tuples, effects, renamings, and partitions,
Then every per-example and mutation row matches its stored content hash.

#### SCENARIO-VERIFY-6926-READINESS: Complete Exact Evidence Gates Readiness

Given all family and label cells, explicit ambiguity, stable hashes, exact
engine agreement, and complete row evidence,
When Exp6926 computes readiness,
Then `span_relation_fixture_ready_score=1`. Any failed check keeps it zero.

## Implementation Status (REQ-VERIFY-6926)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-6926 and SCENARIO-VERIFY-6926-* | Planned (`python/carnot/experiment_6926_span_first_relation_fixture.py`; `scripts/experiments/experiment_6926_span_first_relation_fixture.py`) | Planned (`tests/python/test_experiment_6926_span_first_relation_fixture.py`) |

### REQ-VERIFY-6987: Independent Contrast Feature Leakage Audit

Carnot SHALL provide Exp6987 at
`python/carnot/experiment_6987_contrast_feature_audit.py`. The command
`.venv/bin/python scripts/experiments/experiment_6987_contrast_feature_audit.py --date 20260904`
SHALL write `results/experiment_6987_contrast_feature_audit.json`.

The controller SHALL start a fresh child process for the audit. The child SHALL
have no network namespace, GPU devices, LLM access, or write access to the
source tree. It SHALL hash each frozen source before it parses JSON or reads an
aggregate claim. It SHALL compare the Exp6984, Exp6985, and Exp6986 file hashes
with pinned values. It SHALL also compare canonical hashes for Exp6986 raw token
rows and joined label rows. A failed precondition SHALL write a blocked
artifact. Its `gate_check_summary` SHALL name the failed check, expected value,
and observed value.

The audit SHALL rebuild all 414 candidate-model joins from Exp6986 raw feature
rows and late label rows. Each of the 138 candidates SHALL have exactly one row
for each of the three required model families. Each feature key and label key
SHALL be unique. Candidate and model hashes SHALL replay. Missing rows,
duplicate joins, extra joins, or conflicting labels SHALL remain explicit in
`source_disagreement_rows` and SHALL keep bank readiness at zero.

The audit SHALL derive source, pair, split, fault, recurrence, and chronology
metadata independently from Exp6984, Exp6985, Exp6975, and Exp6976. The primary
training partitions SHALL be the frozen Exp6984 train, calibration, and
held-out partitions with 18, 6, and 12 contrast pairs. Each partition SHALL be
exactly label balanced. All Exp6985 and Exp6976 transfer rows SHALL remain
audit-only strata. Their label counts SHALL still be reported. A source group
or pair group SHALL not cross a training partition. Exp6985 future-window,
recurrence, and later-event metadata SHALL not enter an earlier candidate
feature view.

The child SHALL rebuild the exact model-worker payload from the frozen scoring
manifest. Model input fields SHALL contain only candidate identity, prompt text,
and candidate text. No exact label, fault family, split, source group,
recurrence link, future window, or authority result may occur as a model input
field or a nested candidate payload field. Stored process receipts and rebuilt
payload hashes SHALL agree for all three model families.

Exp6987 SHALL freeze a training feature allowlist before it fits any probe. The
allowlist MAY contain label-blind sequence likelihood, entropy, probability
margin, local surprisal-change, and parser structure fields. It SHALL exclude
labels, authority results, fault or mutation metadata, split, source group,
recurrence or future metadata, candidate order, candidate IDs, pair IDs,
hashes, and model identity. `direct_leakage_count` SHALL count prohibited fields
that occur in the rebuilt model input schema, nested model text payloads, or
training allowlist.

The audit SHALL fit preregistered shortcut probes for length only,
serialization only, source metadata only, mutation metadata only, candidate
order only, identifier style only, and model identity only. Cross-validation
SHALL group all rows for one frozen pair together. No candidate or model repeat
may cross a fold. Preprocessing SHALL fit on training folds only. Out-of-fold
scores SHALL collapse to candidate rows. Confidence intervals SHALL resample
pair groups, not candidate-model rows. Every fold, candidate score, pair group,
point AUROC, and 95-percent interval SHALL remain in terminal rows.

`feature_audit_complete_score` SHALL equal the bare integer one only when every
source hash, raw row, label join, metadata row, model receipt, schema check, and
shortcut probe replays in the fresh child. A completed adverse finding does not
reduce this score.

`contrast_feature_bank_ready_score` SHALL equal the bare integer one only when
the audit is complete, `direct_leakage_count` is zero, all 414 model rows are
complete, all primary splits are exactly balanced and source-disjoint, future
leakage is zero, and every prohibited shortcut probe has an upper 95-percent
AUROC bound below 0.80. A completed shortcut-gate failure SHALL use
`verdict_class: disqualified` and an `honest_verdict` that starts with
`complete_disqualified_`. It SHALL not repair, rewrite, filter, or relabel any
Exp6984, Exp6985, or Exp6986 source row.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `rows`,
`per_candidate_model_rows`, `join_replay_rows`, `row_count_rows`,
`family_coverage_rows`, `label_balance_rows`, `source_overlap_rows`,
`split_isolation_rows`, `label_denial_replay_rows`, `feature_schema_rows`,
`prohibited_field_rows`, `shortcut_probe_rows`, `shortcut_interval_rows`,
`source_disagreement_rows`, `read_only_enforcement_receipt`,
`direct_leakage_count`, `feature_audit_complete_score`,
`contrast_feature_bank_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL contain one
scientific principle for every required field and both score fields.
`inference_substrate` SHALL equal
`fresh_process_contrast_leakage_audit_no_llm`. `verifier_is_oracle` SHALL be
true. `verdict_class` SHALL use `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`.

#### SCENARIO-VERIFY-6987-PRECONDITIONS: Frozen Evidence Fails Closed

Given a changed fixture hash, raw-row hash, joined-row hash, missing model row,
or unavailable read-only child boundary,
When Exp6987 checks its inputs,
Then it writes a blocked artifact with the first exact expected-observed pair.

#### SCENARIO-VERIFY-6987-JOINS: Every Candidate Has Three Unique Joins

Given raw feature rows and late label rows,
When the audit rebuilds candidate-model joins,
Then all 414 keys and hashes agree, and a missing or duplicate key is retained
as a disagreement.

#### SCENARIO-VERIFY-6987-BALANCE: Primary Splits Stay Pair Balanced

Given the 18 train, 6 calibration, and 12 held-out Exp6984 pairs,
When the audit recomputes split counts,
Then each split has equal positive and negative candidates. Any imbalance keeps
readiness at zero.

#### SCENARIO-VERIFY-6987-SOURCES: Source Groups Never Cross Splits

Given independently rebuilt source and pair groups,
When the audit compares every partition pair,
Then all overlap counts are zero. Any overlap keeps readiness at zero.

#### SCENARIO-VERIFY-6987-FUTURE: Later Stream Metadata Cannot Enter Features

Given an Exp6985 event and its held-future or recurrence metadata,
When the audit reconstructs the event's candidate views,
Then only metadata available at that event is visible. A future field or later
event reference keeps readiness at zero.

#### SCENARIO-VERIFY-6987-LABEL-DENIAL: Direct Oracle Fields Are Rejected

Given a label, fault, split, source, recurrence, future, or authority field at
any model-input depth,
When the audit scans the rebuilt payload,
Then it records the exact path and increments `direct_leakage_count`.

#### SCENARIO-VERIFY-6987-SCHEMA: Training Excludes IDs And Provenance

Given the feature bank schemas,
When the audit freezes the training allowlist,
Then no label, identifier, hash, order, model identity, or provenance field is
allowed. An injected prohibited field keeps readiness at zero.

#### SCENARIO-VERIFY-6987-SHORTCUTS: Pair-Grouped Probes Gate Release

Given length, serialization, source, mutation, order, identifier, and model-only
feature sets,
When pair-grouped cross-validation and pair bootstrap finish,
Then each probe reports out-of-fold AUROC and a 95-percent interval. An upper
bound at or above 0.80 terminally disqualifies the bank.

#### SCENARIO-VERIFY-6987-BARE: Scores Are Bare Integers

Given any blocked, disqualified, or ready artifact,
When validation reads its downstream gates,
Then both audit and readiness scores are bare integers, the direct-leakage
count is a bare integer, and the verdict prefix matches its class.

#### SCENARIO-VERIFY-6987-READONLY: The Child Cannot Mutate Sources

Given the fresh audit child,
When it attempts network, GPU, LLM, or source write access,
Then each capability is absent or denied, source hashes are unchanged after the
run, and only the controller writes the terminal result.

## Implementation Status (REQ-VERIFY-6987)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-6987 and SCENARIO-VERIFY-6987-* | Planned (`python/carnot/experiment_6987_contrast_feature_audit.py`; `scripts/experiments/experiment_6987_contrast_feature_audit.py`) | Planned (`tests/python/test_experiment_6987_contrast_feature_audit.py`) |

### REQ-VERIFY-6997: Authority-Sidecar Feature-Bank Rebuild

Carnot SHALL provide Exp6997 at
`python/carnot/experiment_6997_authority_sidecar_rebuild.py`. The command
`.venv/bin/python scripts/experiments/experiment_6997_authority_sidecar_rebuild.py --date 20260904`
SHALL write `results/experiment_6997_authority_sidecar_rebuild.json` without
running GGUF inference or fitting a verifier.

The workflow SHALL first verify the terminal Exp6984 through Exp6987 artifacts,
all 414 raw feature rows, their pinned source hashes, three complete model-family
views for each candidate, and writable immutable output paths. A failed
precondition SHALL write `blocked_authority_sidecar_rebuild`. Its
`gate_check_summary` SHALL name the failed check, expected value, and observed
value.

The workflow SHALL derive one frozen candidate key from the semantic prompt hash
and candidate hash only. Source, mutation, row order, split, label, candidate
identifier, and model family SHALL not affect this key. Each of the 138
candidates SHALL have one key and exactly three family rows. Duplicate keys,
missing families, hash disagreements, or conflicting semantic inputs SHALL fail
closed and remain in row evidence.

The workflow SHALL pivot the three family rows into one wide learner row per
candidate. A frozen allowlist SHALL retain only numeric teacher-forced sequence
and parser features. Each model family SHALL map to a stable neutral numeric
position. The learner row SHALL not contain a categorical model identity,
repository string, label, split, source, mutation provenance, authority record,
hash alias, order field, nested metadata, or another provenance alias.

The workflow SHALL write three immutable JSON Lines files before any authority
join: `learner_view.jsonl`, `label_split_sidecar.jsonl`, and
`mutation_authority_sidecar.jsonl`. The learner file SHALL contain only the
frozen key and allowed numeric tensors. The label sidecar SHALL contain exact
labels and split routing. The mutation sidecar SHALL contain source, mutation,
witness, and exact-authority records. A manifest and SHA-256 hash SHALL bind each
file before any join.

The learner loader API SHALL accept only the learner JSON Lines path and the
frozen allowlist. It SHALL reject extra paths, objects, environment variables,
or callbacks that can expose either sidecar. Tests SHALL record file-open or
system-call receipts and prove that the loader opens only its learner path. The
loader SHALL reject duplicate keys, denied fields at any depth, categorical
identity, nonnumeric tensors, allowlist drift, and manifest hash mismatch.

A separate authority joiner MAY open the two sidecars only after learner tensors
are materialized. It SHALL join by the frozen key only. It SHALL return labels
and split routing separately from the numeric feature matrix. Mutation
provenance and authority records SHALL never enter its feature output. Duplicate,
missing, extra, or conflicting sidecar keys and hash mismatches SHALL fail closed.

The workflow SHALL permute every file's row order and alpha-rename every
nonsemantic sidecar identifier. The learner tensor bytes and a deterministic
prediction hash SHALL remain unchanged after each transformation. The artifact
SHALL preserve every mismatch and rejected-row receipt.

`sidecar_rebuild_complete_score` SHALL equal the bare integer one only when all
138 candidate rows and all 414 source family rows replay. The score SHALL be zero
otherwise. `blinded_learner_view_ready_score` SHALL equal the bare integer one
only when the allowlist is clean, learner-side sidecar access is denied, all
hashes bind, and row permutation plus sidecar alpha-renaming preserve learner
tensors and predictions. A failed isolation check SHALL use a disqualified
verdict, not a partial verdict.

The terminal artifact SHALL include `field_principles`,
`experiment_id`, `run_date`, `schema`, `preconditions_checked`,
`inference_substrate`, `duration_s`,
`source_artifact_hashes`, `MODEL_SPECS_inherited`, `source_model_rows`, `rows`,
`per_candidate_rows`, `candidate_key_rows`, `family_completeness_rows`,
`pivot_rows`, `learner_view_path`, `learner_view_hash`,
`learner_view_manifest_rows`, `label_split_sidecar_path`,
`label_split_sidecar_hash`, `label_split_manifest_rows`,
`mutation_authority_sidecar_path`, `mutation_authority_sidecar_hash`,
`mutation_authority_manifest_rows`, `feature_allowlist`,
`prohibited_feature_rows`, `nested_metadata_rows`,
`categorical_identity_rows`, `learner_loader_rows`, `denied_access_rows`,
`file_open_receipt_rows`, `authority_join_rows`, `tensor_hash_rows`,
`row_permutation_rows`, `alpha_rename_rows`, `source_disagreement_rows`,
`expected_candidate_count`, `observed_candidate_count`,
`expected_source_row_count`, `observed_source_row_count`,
`gguf_inference_performed`, `verifier_fit_performed`,
`sidecar_rebuild_complete_score`, `blinded_learner_view_ready_score`,
`random_seed`, `reproducibility_checksum`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.
`field_principles` SHALL contain one scientific principle for each required
field and both score fields. `inference_substrate` SHALL equal
`deterministic_feature_sidecar_transform_no_llm`. `verifier_is_oracle` SHALL be
true. A ready artifact SHALL use `circular_positive`. `verdict_class` SHALL use
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
The terminal prefix of `honest_verdict` SHALL match the verdict class.

#### SCENARIO-VERIFY-6997-PRECONDITIONS: Frozen Inputs Fail Closed

Given a missing terminal artifact, changed source hash, incomplete family view,
missing raw row, or unwritable immutable output path,
When Exp6997 checks its inputs,
Then it writes a blocked artifact with the first exact expected-observed pair.

#### SCENARIO-VERIFY-6997-KEYS: Semantic Keys Exclude Provenance

Given identical prompt and candidate hashes with changed source, mutation,
order, split, label, identifier, or model family,
When the workflow freezes candidate keys,
Then the keys are equal. A duplicate semantic row or conflicting input fails.

#### SCENARIO-VERIFY-6997-FAMILIES: Three Family Views Are Required

Given the frozen 414 source rows,
When the workflow groups them by candidate key,
Then each of 138 keys has exactly the three stable neutral family positions.

#### SCENARIO-VERIFY-6997-ALLOWLIST: Learner Rows Are Numeric And Blind

Given a label column, provenance alias, nested metadata, categorical model
identity, repository string, or nonnumeric feature,
When the workflow builds or loads the learner view,
Then it rejects the row and records the exact denied path.

#### SCENARIO-VERIFY-6997-SIDECARS: Files Bind Before Joining

Given the three immutable JSON Lines files,
When their manifests are written,
Then each manifest binds its path, row count, ordered keys, byte count, and
SHA-256 file hash before the authority join starts.

#### SCENARIO-VERIFY-6997-LOADER: Learner Access Is Narrow

Given paths, objects, environment variables, or callbacks that expose a
sidecar,
When the learner loader API is invoked,
Then access is denied and file-open receipts show only the learner file.

#### SCENARIO-VERIFY-6997-JOIN: Authority Output Stays Separate

Given materialized learner tensors and both valid sidecars,
When the authority joiner joins by frozen key,
Then labels and split routing are separate outputs and mutation provenance does
not enter the feature matrix.

#### SCENARIO-VERIFY-6997-HASH: Any Byte Mismatch Fails Closed

Given a changed learner or sidecar byte after its manifest is frozen,
When a loader or joiner verifies the file,
Then it rejects the hash mismatch before returning data.

#### SCENARIO-VERIFY-6997-INVARIANCE: Order And Aliases Are Nonsemantic

Given permuted rows and alpha-renamed nonsemantic sidecar identifiers,
When the workflow rematerializes and joins the data,
Then learner tensor bytes and deterministic prediction hashes are byte-stable.

#### SCENARIO-VERIFY-6997-BARE: Readiness Is A Bare Integer

Given a ready, blocked, disqualified, or partial artifact,
When artifact validation reads downstream gates,
Then both score fields are bare integers and the verdict prefix matches its
class.

## Implementation Status (REQ-VERIFY-6997)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-6997 and SCENARIO-VERIFY-6997-* | Implemented (`python/carnot/experiment_6997_authority_sidecar_rebuild.py`; `scripts/experiments/experiment_6997_authority_sidecar_rebuild.py`; terminal artifact `results/experiment_6997_authority_sidecar_rebuild.json`) | Verified (`tests/python/test_experiment_6997_authority_sidecar_rebuild.py`; 30 focused tests; 100% module statement coverage) |

### REQ-VERIFY-6999: Blinded Feature Cold Audit

Carnot SHALL provide Exp6999 at
`python/carnot/experiment_6999_blinded_feature_cold_audit.py`. The command
`.venv/bin/python scripts/experiments/experiment_6999_blinded_feature_cold_audit.py --date 20260905`
SHALL write `results/experiment_6999_blinded_feature_cold_audit.json`.

The controller SHALL start a fresh child process. The child SHALL have no
network, GPU, LLM, training, or source-write capability. The child SHALL hash
all source artifacts, all three immutable Exp6997 JSON Lines files, and their
manifests before it parses any source aggregate claim. Only the controller MAY
write the terminal artifact.

The audit SHALL require the two structured roadmap gates. The Exp6997
`blinded_learner_view_ready_score` SHALL equal the bare integer one. The Exp6998
`commitment_control_complete_score` SHALL equal the bare integer one. It SHALL
also require 138 learner rows, 414 raw family rows, 216 terminal commitment
rows, three immutable sidecar files, their manifests, the frozen allowlist,
learner loader receipts, and a working read-only boundary. A failed
precondition SHALL write `blocked_blinded_feature_cold_audit`.
`gate_check_summary` SHALL name the first failed check and its expected and
observed values.

The child SHALL rebuild all 138 wide learner rows from the 414 raw Exp6986
model rows. It SHALL derive each candidate key only from the prompt hash and
candidate hash. Each candidate SHALL have exactly one row for each of the three
required model families. The child SHALL recompute family positions, wide
tensors, tensor hashes, exact-label joins, label balance, source disjointness,
and split membership. Duplicate keys, missing families, order changes, hash
drift, extra keys, missing keys, and source conflicts SHALL remain explicit in
terminal disagreement rows.

The child SHALL invoke an independently implemented learner loader under five
mutation-sidecar conditions: correct, frozen row permutation, cross-pair
replacement, empty file, and no file. The loader API SHALL accept only the
learner file and frozen allowlist. It SHALL never accept or open a label or
mutation sidecar. Every condition SHALL produce byte-identical ordered keys,
learner tensors, tensor hash, and fixed reference prediction hash. Sidecar
existence, contents, order, and replacement SHALL have no learner effect.

The frozen learner allowlist SHALL contain only the 63 numeric Exp6997 feature
positions. It SHALL exclude labels, splits, source identifiers, mutation and
fault fields, candidate order, pair identifiers, authority records, categorical
model identity, repository identity, prompt-condition identifiers, and all
Exp6998 commitment fields. Direct and nested aliases SHALL be normalized before
matching. Any denied field in a learner row, nested payload, allowlist, or
learner loader input SHALL increment `direct_leakage_count`.

The audit SHALL fit preregistered grouped logistic probes for length only,
serialization only, source metadata only, mutation metadata only, model
identity only, commitment controls only, and the full allowed learner table.
All repeats for one source pair SHALL stay in one fold. Preprocessing SHALL fit
on training folds only. The audit SHALL report every fold assignment,
coefficient, out-of-fold candidate prediction, point AUROC, pair-bootstrap
95-percent interval, and threshold decision. Source metadata, mutation
metadata, model identity, and commitment controls SHALL be prohibited
metadata-only probes. Their upper 95-percent interval SHALL be below `0.80` for
release. An upper bound equal to `0.80` SHALL fail.

`feature_isolation_audit_complete_score` SHALL equal the bare integer one only
when every expected row, mutation, probe, interval, loader condition, open
receipt, and read-only receipt is terminal. A completed adverse finding SHALL
not reduce this score.

`blinded_feature_bank_ready_score` SHALL equal the bare integer one only when
the audit is complete, `direct_leakage_count` is zero, learner invariance is
exact, all 138 candidates and all three model families are complete, required
splits are balanced and source-disjoint, all prohibited fields are absent, and
every prohibited metadata-only interval has an upper bound below `0.80` AUROC.
A failed shortcut gate SHALL be a terminal disqualification. The audit SHALL
not repair or rewrite Exp6997 or Exp6998. A ready result SHALL use
`verdict_class: circular_positive` because exact labels define the release
decision.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `rows`,
`per_candidate_rows`, `raw_rebuild_rows`, `candidate_key_rows`,
`family_completeness_rows`, `join_replay_rows`, `tensor_hash_rows`,
`label_balance_rows`, `source_overlap_rows`, `split_isolation_rows`,
`label_denial_replay_rows`, `feature_schema_rows`, `prohibited_field_rows`,
`sidecar_condition_rows`, `sidecar_open_attempt_rows`,
`sidecar_permutation_rows`, `sidecar_replacement_rows`,
`sidecar_removal_rows`, `reference_prediction_rows`,
`learner_invariance_rows`, `shortcut_probe_rows`,
`shortcut_prediction_rows`, `shortcut_interval_rows`,
`commitment_prohibition_rows`, `source_disagreement_rows`,
`read_only_enforcement_receipt`, `direct_leakage_count`,
`feature_isolation_audit_complete_score`, `blinded_feature_bank_ready_score`,
`random_seed`, `reproducibility_checksum`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.
`field_principles` SHALL contain one scientific principle for every required
field, including both score fields. `inference_substrate` SHALL equal
`fresh_process_blinded_feature_isolation_audit_no_llm`.
`verifier_is_oracle` SHALL be true. `verdict_class` SHALL use `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.

#### SCENARIO-VERIFY-6999-PRECONDITIONS: Structured Gates And Frozen Inputs Fail Closed

Given a failed structured gate, changed input hash, missing sidecar or manifest,
wrong row count, absent loader receipt, or unavailable read-only boundary,
When Exp6999 checks inputs before source parsing,
Then it writes a schema-complete blocked artifact with the first exact
expected-observed pair.

#### SCENARIO-VERIFY-6999-REBUILD: Raw Family Rows Define The Learner View

Given the 414 raw model rows,
When the child rebuilds semantic keys and wide tensors,
Then it produces 138 complete three-family learner rows.
Then duplicate semantic keys, missing families, or changed row order cannot
silently alter a tensor.

#### SCENARIO-VERIFY-6999-LEAKAGE: Direct Nested And Alias Fields Are Denied

Given a direct field, nested field, normalized alias, model identity, or prompt
condition outside the frozen numeric allowlist,
When the learner schema and loader validate it,
Then they record its exact path and deny the row before materialization.

#### SCENARIO-VERIFY-6999-SIDECARS: Sidecar Changes Cannot Affect The Learner

Given correct, permuted, cross-pair replacement, empty, and absent mutation
sidecar conditions,
When the learner loader materializes the same learner file,
Then it opens no sidecar and returns byte-identical tensors and predictions.

#### SCENARIO-VERIFY-6999-SPLITS: Labels And Sources Replay Outside Features

Given the frozen label and authority sidecars,
When the audit joins them after learner materialization,
Then primary splits are label-balanced and source-disjoint.
Then labels, split routing, pair keys, and provenance remain outside tensors.

#### SCENARIO-VERIFY-6999-COMMITMENT: Commitment Evidence Is Audit Only

Given 216 complete Exp6998 commitment-control rows,
When the audit builds the commitment probe,
Then it may use those rows only as a prohibited shortcut control.
Then no commitment, condition, choice, or model-identity field enters the
learner allowlist.

#### SCENARIO-VERIFY-6999-SHORTCUTS: Pair-Grouped Intervals Gate Release

Given all seven preregistered probes,
When grouped cross-validation and pair bootstrap finish,
Then every fold, coefficient, candidate prediction, point AUROC, and interval
is terminal.
Then a prohibited metadata-only upper bound at or above `0.80` disqualifies the
bank.

#### SCENARIO-VERIFY-6999-BARE: Readiness Fields Are Bare And Class Consistent

Given a ready, blocked, partial, or disqualified artifact,
When artifact validation runs,
Then both scores and `direct_leakage_count` are bare integers.
Then the honest verdict has a terminal prefix that matches its verdict class.

#### SCENARIO-VERIFY-6999-READONLY: The Fresh Child Cannot Train Or Mutate Sources

Given the audit child,
When it probes network, GPU, LLM, training, and source-write capability,
Then every capability is absent or denied and source hashes remain unchanged.

## Implementation Status (REQ-VERIFY-6999)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-6999 and SCENARIO-VERIFY-6999-* | Planned (`python/carnot/experiment_6999_blinded_feature_cold_audit.py`; `scripts/experiments/experiment_6999_blinded_feature_cold_audit.py`) | Planned (`tests/python/test_experiment_6999_blinded_feature_cold_audit.py`) |

### REQ-VERIFY-7012: Exact Minimal Constraint-Intervention Pair Fixture

Carnot SHALL provide Exp7012 at
`python/carnot/experiment_7012_exact_intervention_pair_fixture.py`. The command
`.venv/bin/python scripts/experiments/experiment_7012_exact_intervention_pair_fixture.py --date 20260905`
SHALL write `results/experiment_7012_exact_intervention_pair_fixture.json`.
The experiment SHALL use deterministic exact authorities and SHALL not use an
LLM or fit a learner.

The experiment SHALL require at least four executable source families. It
SHALL also require deterministic mutation and isomorphism functions, both
exact authority engines, and writable immutable fixture paths. A failed
precondition SHALL write a schema-complete
`blocked_intervention_pair_fixture` artifact. Its `gate_check_summary` SHALL
name the failed check, expected value, and observed value.

The experiment SHALL freeze 48 source-disjoint primary intervention blocks.
Each source family SHALL supply 12 blocks. Each block SHALL contain a clean
candidate, one exact single-edit violation or repair, and an isomorphic copy
of that pair. Thus, each block SHALL have four learner prompts and two labels
of each class. The source group SHALL determine the `train`, `calibration`,
`held_source`, or `sealed_headroom` split before any candidate is certified.
No block SHALL cross a split.

Each accepted intervention SHALL have structural edit distance one. Its clean
candidate SHALL be exactly equivalent. Its changed candidate SHALL be exactly
non-equivalent. The bounded enumerator and Z3 authority SHALL return terminal,
matching labels and exact witnesses. A no-op, multi-edit mutation, ambiguous
authority result, nonminimal repair, invalid isomorphism, or duplicate semantic
key SHALL reject the block. Every rejected or unbalanced attempt SHALL remain
in `rejected_block_rows`.

Each accepted block SHALL hold its mutation kind and source bookkeeping fixed.
Each label SHALL occur once in each serialization template. Each neutral pair
position SHALL contain both labels. All four prompts SHALL have equal token and
character lengths. These checks SHALL be recorded per block. Row order SHALL
not change the canonical learner bytes.

The learner file SHALL contain only `semantic_key`,
`neutral_block_position`, and `prompt`. The key SHALL depend only on semantic
candidate content and neutral position. Direct, nested, or normalized aliases
for labels, sources, splits, mutation data, witnesses, authority records, pair
roles, or provenance SHALL be prohibited. Exact labels, source family, split,
mutation kind, witnesses, and full authority records SHALL exist only in a
separately hashed authority sidecar. The learner loader SHALL reject any
sidecar argument or authority-bearing row.

The command SHALL replay every authority and isomorphism in a fresh child
process with a separate network namespace and no visible GPU or online model
access. Learner bytes SHALL remain identical when sidecar rows are permuted,
replaced, deleted, or alpha-renamed. Source artifact hashes SHALL remain
unchanged across the child process.

`intervention_pair_fixture_ready_score` SHALL be the bare integer one only
when all 48 blocks and all four families are present, all interventions are
exact and minimal, every nuisance balance check passes, the fresh-process
replay passes, all file hashes replay, and no prohibited field reaches learner
input. A ready artifact SHALL use `verdict_class: circular_positive` because
the exact authorities define fixture readiness.

The artifact SHALL contain `schema`, `experiment_id`, `run_date`,
`field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `authority_family_rows`, `rows`,
`pair_rows`, `block_rows`, `rejected_block_rows`, `intervention_rows`,
`minimality_rows`, `authority_witness_rows`, `isomorphism_rows`,
`nuisance_balance_rows`, `label_balance_rows`, `length_balance_rows`,
`serialization_balance_rows`, `group_split_rows`, `learner_prompt_path`,
`learner_prompt_hash`, `authority_sidecar_path`, `authority_sidecar_hash`,
`sidecar_intervention_rows`, `prohibited_feature_rows`,
`expected_pair_count`, `observed_pair_count`, `expected_family_count`,
`observed_family_count`, `intervention_pair_fixture_ready_score`,
`random_seed`, `reproducibility_checksum`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.
`field_principles` SHALL contain one scientific principle for every required
field. `inference_substrate` SHALL equal
`deterministic_exact_pair_fixture_no_llm`. `expected_pair_count` SHALL equal
48. `expected_family_count` SHALL equal four. `verifier_is_oracle` SHALL be
true. `verdict_class` SHALL use the closed project verdict enum. The terminal
prefix of `honest_verdict` SHALL match its class.

#### SCENARIO-VERIFY-7012-PRECONDITIONS: Missing Exact Support Fails Closed

Given fewer than four executable families, unavailable mutation or isomorphism
support, unavailable exact authority, changed source bytes, or unwritable
fixture paths,
When Exp7012 checks its inputs,
Then it writes `blocked_intervention_pair_fixture` with exact diagnostics.

#### SCENARIO-VERIFY-7012-MINIMALITY: Only One Exact Edit Enters A Pair

Given a no-op, multi-edit mutation, nonminimal repair, or ambiguous authority
result,
When the block validator compares exact candidates and authority records,
Then it rejects the block and preserves the reason.

#### SCENARIO-VERIFY-7012-ISOMORPHISM: Surface Changes Preserve Both Labels

Given an alpha-renamed clean and changed pair,
When exact replay compares normalized structure and authority labels,
Then both labels are unchanged and the normalized structures match.
Then any semantic change rejects the isomorphism.

#### SCENARIO-VERIFY-7012-BALANCE: Matched Blocks Hold Nuisance Variables Fixed

Given one accepted four-prompt block,
When nuisance checks compare mutation, serialization, length, source fields,
candidate order, and label counts,
Then mutation and source values are fixed, both templates and positions are
label-balanced, and token and character lengths are equal.

#### SCENARIO-VERIFY-7012-KEYS: Semantic Keys Are Unique And Order Stable

Given prompt rows in any input order,
When the learner fixture canonicalizes them,
Then its bytes and ordered keys are identical.
Then a duplicate semantic key fails before materialization.

#### SCENARIO-VERIFY-7012-LEAKAGE: Authority Data Stays In The Sidecar

Given direct, nested, or normalized metadata aliases, or a loader sidecar
argument,
When learner schema validation runs,
Then it rejects the exact row path before prompts are returned.

#### SCENARIO-VERIFY-7012-SIDECARS: Sidecar Changes Cannot Change Learner Bytes

Given the correct, permuted, replaced, deleted, or alpha-renamed sidecar,
When the fresh child loads the frozen learner fixture,
Then the ordered learner prompts and file hash remain byte-identical.

#### SCENARIO-VERIFY-7012-SPLITS: Blocks Never Cross Frozen Partitions

Given the 48 source groups and four partitions,
When split rows are frozen before authority scoring,
Then every prompt from one block has one split and source groups do not overlap.

#### SCENARIO-VERIFY-7012-ARTIFACT: Readiness Replays From Rows

Given a blocked, disqualified, or ready artifact,
When independent validation recomputes counts, family coverage, hashes,
minimality, balance, leakage, verdict, and checksum,
Then a consistent artifact passes and any forged gate fails.

## Implementation Status (REQ-VERIFY-7012)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7012 and SCENARIO-VERIFY-7012-* | Implemented (`python/carnot/experiment_7012_exact_intervention_pair_fixture.py`; `scripts/experiments/experiment_7012_exact_intervention_pair_fixture.py`; `results/experiment_7012_exact_intervention_pair_fixture.json`) | Verified (`tests/python/test_experiment_7012_exact_intervention_pair_fixture.py`; 48 accepted blocks, four source families, 100% new-module statement coverage) |

### REQ-VERIFY-7014: Causal Feature Cold Audit

Carnot SHALL provide Exp7014 at
`python/carnot/experiment_7014_causal_feature_cold_audit.py`. The command
`.venv/bin/python scripts/experiments/experiment_7014_causal_feature_cold_audit.py --date 20260905`
SHALL write `results/experiment_7014_causal_feature_cold_audit.json`.

The controller SHALL run the audit in a fresh process with a separate network
namespace. The child SHALL have no network, GPU, online model, training, or
source-write capability. Before analysis, it SHALL require the bare Exp7012
readiness score and the bare Exp7013 completion score to equal one. It SHALL
also require the pinned Exp7012, Exp7013, learner-prompt, authority-sidecar,
prompt-freeze, and response-freeze hashes. A failed precondition SHALL write
`blocked_causal_feature_audit`. Its `gate_check_summary` SHALL name the first
failed check with its expected and observed values.

The child SHALL rebuild one canonical causal learner tensor from all 144
Exp7013 signed response rows. It SHALL use only the signed primary and signed
isomorphic response values as tensor inputs. It SHALL keep exact direction,
direct labels, aliases, nested provenance, row keys, serialization, length,
mutation kind, source, split, model identity, norm, magnitude, and hashes out
of the tensor. It SHALL retain every rejected path in `direct_leakage_rows`.
Any changed, duplicate, missing, non-finite, or incomplete family cell SHALL
remain terminal evidence and prevent readiness.

The child SHALL materialize the tensor without a sidecar parameter. It SHALL
repeat materialization while the ambient authority sidecar is correct,
permuted, cross-pair replaced, deleted, and alpha-renamed. Ordered tensor bytes,
tensor hashes, and fixed reference predictions SHALL match exactly for all
conditions. The loader SHALL not open a sidecar.

The audit SHALL fit preregistered grouped probes for metadata, serialization,
length, mutation, source, split, row ordering, model identity, norm-only, and
magnitude-only inputs. The target SHALL be the exact clean-to-repair versus
clean-to-violation intervention direction, joined only after tensor
materialization. Repeats from one source block SHALL remain in one fold.
Preprocessing SHALL fit on training folds only. Bootstrap sampling SHALL use
whole source blocks and source families, never individual rows. The artifact
SHALL retain every fold, out-of-fold prediction, bootstrap draw, interval, and
gate decision. Each prohibited probe's upper 95-percent AUROC bound SHALL be
strictly below `0.80`; equality SHALL fail.

The audit SHALL recompute clean-to-violation and clean-to-repair signs for each
mandated model family and each held source group. A family is identifiable only
when the preregistered paired block-bootstrap intervals for both directions
exclude zero in their compatible direction. Isomorphic variants SHALL preserve
the primary aggregate direction within the frozen absolute tolerance of
`0.20`. Pooled effects SHALL not hide a family or held-source reversal.

`causal_bank_audit_complete_score` SHALL equal the bare integer one only when
all expected input, tensor, intervention, probe, bootstrap, isomorphism, family,
held-source, and leakage rows are terminal. A completed adverse result SHALL
not reduce this score.

`causal_feature_bank_ready_score` SHALL equal the bare integer one only when
the audit is complete, `direct_leakage_count` is zero, all prohibited AUROC
upper bounds are below `0.80`, sidecar and isomorphic invariance pass, all
family cells are complete, and at least two of three mandated families have
identifiable signed direction. Direct leakage or a failed shortcut or
isomorphic gate SHALL be `disqualified`. Absent signed direction after all
other release gates pass SHALL be `null`. A ready result SHALL be `positive`.
Thresholds, folds, seeds, and tolerances SHALL be fixed before held rows open.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`,
`replay_hash_rows`, `rows`, `per_pair_results`, `learner_tensor_rows`,
`sidecar_intervention_rows`, `direct_leakage_rows`, `metadata_probe_rows`,
`serialization_probe_rows`, `length_probe_rows`, `mutation_probe_rows`,
`source_probe_rows`, `split_probe_rows`, `model_identity_probe_rows`,
`norm_only_probe_rows`, `magnitude_only_probe_rows`,
`grouped_bootstrap_rows`, `isomorphic_invariance_rows`,
`family_identifiability_rows`, `held_source_identifiability_rows`,
`direct_leakage_count`, `prohibited_auroc_upper_bound_max`,
`identifiable_family_count`, `causal_bank_audit_complete_score`,
`causal_feature_bank_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL state one
scientific principle for every field in this list. `inference_substrate` SHALL
equal `fresh_process_causal_feature_audit_no_llm`. `verifier_is_oracle` SHALL
be false. The verdict class SHALL use the closed project enum. The terminal
prefix of `honest_verdict` SHALL match its verdict class.

#### SCENARIO-VERIFY-7014-PRECONDITIONS: Frozen Inputs Fail Closed

Given a false upstream gate, hash mismatch, unavailable sandbox, or incomplete
family cell,
When Exp7014 performs preflight,
Then it writes a schema-complete blocked artifact with the first exact
expected-observed pair.

#### SCENARIO-VERIFY-7014-LEAKAGE: Direct Nested And Alias Inputs Are Denied

Given a direct label, normalized alias, nested provenance field, row key,
serialization field, length, mutation, source, split, model identity, norm,
magnitude, or hash,
When the learner schema validates it,
Then it rejects the exact path before tensor materialization.

#### SCENARIO-VERIFY-7014-SIDECARS: Authority Files Cannot Change Responses

Given correct, permuted, replaced, deleted, and alpha-renamed sidecars,
When the narrow loader rebuilds signed-response tensors,
Then ordered tensor bytes and reference predictions are identical and no
sidecar is opened.

#### SCENARIO-VERIFY-7014-PROBES: Nuisance Probes Stay Grouped

Given every preregistered prohibited feature family,
When grouped fitting and bootstrap intervals run,
Then source blocks and source families remain intact and every prediction and
draw stays visible.

#### SCENARIO-VERIFY-7014-IDENTIFIABILITY: Signed Effects Remain Disaggregated

Given the three mandated model families and every held source group,
When clean-to-violation and clean-to-repair signs are recomputed,
Then both directions, intervals, reversals, and isomorphic differences remain
separate for each model and source.

#### SCENARIO-VERIFY-7014-RELEASE: One Bare Readiness Field Controls Release

Given terminal audit rows,
When release rules are reduced,
Then leakage or nuisance substitution disqualifies the bank, absent signed
direction is null, and readiness equals one only when every rule passes.

#### SCENARIO-VERIFY-7014-ARTIFACT: Aggregate Claims Replay From Rows

Given any terminal Exp7014 artifact,
When independent validation recomputes counts, hashes, intervals, completion,
readiness, verdict, principles, and checksum,
Then a consistent artifact passes and a forged aggregate fails.

## Implementation Status (REQ-VERIFY-7014)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7014 and SCENARIO-VERIFY-7014-* | Implemented (`python/carnot/experiment_7014_causal_feature_cold_audit.py`; `scripts/experiments/experiment_7014_causal_feature_cold_audit.py`; `results/experiment_7014_causal_feature_cold_audit.json`) | Verified (`tests/python/test_experiment_7014_causal_feature_cold_audit.py`; leakage mutations, sidecar interventions, grouped source bootstrap, family and held-source signs, hash failures, artifact forgery, and 100% new-module statement coverage) |

### REQ-VERIFY-7129: Three-Family Constraint Bank SHALL Preserve Exact Instance Authority

Exp7129 SHALL freeze 12 base instances before model execution. Four bases
SHALL be SAT-style logic, four SHALL be graph coloring, and four SHALL be
bounded scheduling. Bases within one family SHALL have the same size, density,
surface character budget, and generation token budget. Each base SHALL have
one proof-preserving symbol relabel and one independently checked paraphrase.
The exact checker SHALL run on every base and variant. A relabel SHALL preserve
the normalized exact solution set through its inverse symbol map. A paraphrase
SHALL preserve the exact formal payload, solution set, feasibility, and
requested objective. Solver effort SHALL be retained only as a stratification
covariate and SHALL not be named or used as model-difficulty truth.

The production model roster SHALL contain exactly, and in this order,
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Every model path SHALL resolve through
`cached_sota_pair()`. Inference SHALL use each GGUF's embedded llama.cpp chat
template, bounded tokens, deterministic decoding, and a frozen seed. Each of
the 108 planned model-instance-variant cells SHALL persist its prompt, direct
model text, hashes, token counts, duration, model identity, parse result, and
exact outcome after the invocation. Parse failures and backend failures SHALL
remain rows. The parser SHALL read direct model text only. Finite answer-ID
transport and schema-supported ConstraintIR reprompting SHALL remain false.

Before any model setup, the command SHALL write a schema-complete terminal
blocked artifact. It SHALL then check exact solver availability, frozen fixture
storage, all three cached Q4_K_M language-model files, two idle RTX 3090 lease
targets, CUDA llama.cpp health, and raw trace storage. Checks SHALL retain
numeric telemetry. A stable failed prerequisite SHALL leave `verdict_class` as
`blocked`, use `inference_substrate_class: blocked_no_run`, and name the exact
failed check, expected value, and observed value. An interruption after model
work starts SHALL be `partial` only when missing cells can be retried from the
durable raw manifest.

`sota_constraint_bank_ready_score` SHALL be the bare integer one only when all
planned model, instance, and variant cells have raw receipts, parse receipts,
exact labels, exact outcomes, and exact model identity. Accuracy, parse rate,
constraint violations, relabel sensitivity, paraphrase consistency, and solver
effort strata SHALL not gate readiness. A complete bank is positive-ready even
when its scientific metrics are poor or null. Every proposed action remains
subordinate to exact instance-level verification.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`run_date`, `MODEL_SPECS`, `models_used`, `model_repository_rows`,
`model_path_rows`, `model_hash_rows`, `model_quantization_rows`,
`inference_substrate`, `inference_substrate_class`, `execution_venue`,
`gpu_telemetry_rows`, `token_rows`, `duration_s`, `source_artifact_hashes`,
`raw_trace_manifest`, `rows`, `base_instance_rows`, `variant_rows`,
`solver_receipt_rows`, `model_output_rows`, `parse_rows`,
`exact_outcome_rows`, `family_rows`, `hardness_stratum_rows`,
`relabel_sensitivity_rows`, `paraphrase_consistency_rows`,
`model_identity_confound_rows`, `planned_cell_count`, `completed_cell_count`,
`finite_answer_id_transport_used`, `schema_constraintir_reprompt_used`,
`sota_constraint_bank_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL contain one
non-empty reason for every listed field. `inference_substrate` SHALL equal
`model_bounded_generation: three-family exact constraint bank` for a live or
retryable partial run. Its class SHALL be `model_bounded_generation`. A stable
precondition failure SHALL instead use `blocked_no_run`. `execution_venue`
SHALL equal `host`; `verifier_is_oracle` SHALL be false. Verdict classes SHALL
use only `positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`, and `honest_verdict` SHALL start with the matching terminal prefix.

#### SCENARIO-VERIFY-7129-EXACT: Wrong Labels And Semantic Drift Fail Closed

**Given** a frozen base bank and its variants
**When** a feasibility label, witness, objective, solution-set hash, inverse
relabel map, or paraphrase formal payload changes
**Then** independent validation rejects the exact row
**And** no model aggregate can restore readiness.

#### SCENARIO-VERIFY-7129-TRANSPORT: Only Direct Text Can Become A Proposal

**Given** a direct model response
**When** a row uses a finite answer identifier, schema-supported ConstraintIR
reprompt, answer-conditioned prompt, or non-text proposal source
**Then** validation rejects the row before exact scoring.

#### SCENARIO-VERIFY-7129-IDENTITY: Model Substitution Cannot Fill A Cell

**Given** the frozen three-model roster and file hashes
**When** a hub ID, path, quantization, embedded chat-template receipt, GGUF
hash, or output model identity differs
**Then** the affected cell is incomplete and readiness remains zero.

#### SCENARIO-VERIFY-7129-ROWS: Aggregates Cannot Replace Raw Cells

**Given** reported family, effort, relabel, or paraphrase metrics
**When** raw, parse, exact, or solver rows are absent or their semantic keys do
not cover all 108 planned cells
**Then** the bank is aggregate-only or partial and readiness remains zero.

#### SCENARIO-VERIFY-7129-BLOCKED: Initial And Stable Blocks Keep Full Schema

**Given** startup before model setup or one failed stable prerequisite
**When** Exp7129 writes its current terminal artifact
**Then** every required field exists, rows are empty or exact-fixture-only,
the readiness score is zero, and the first failed expected-observed check is
present.

#### SCENARIO-VERIFY-7129-COMPLETE: Poor Science Does Not Erase A Complete Bank

**Given** all 108 cells with attributable raw and exact receipts
**When** accuracy, parse rate, relabel sensitivity, or paraphrase consistency is
low
**Then** readiness equals one and the artifact uses a positive terminal class
**And** the low scientific metrics remain visible without a hardness claim.

## Implementation Status (REQ-VERIFY-7129)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7129 and SCENARIO-VERIFY-7129-* | Implemented (`python/carnot/experiment_7129_v626_sota_constraint_bank.py`; `scripts/experiments/experiment_7129_v626_sota_constraint_bank.py`; `results/experiment_7129_v626_sota_constraint_bank.json`) | Verified (`tests/python/test_experiment_7129_v626_sota_constraint_bank.py`; 13 focused tests, 108 complete cells, adversarial and row consistency checks, and 100% new-module statement coverage) |

### REQ-VERIFY-7130: Verifier-Committed Routing SHALL Keep Exact Rejection Final

Exp7130 SHALL compare four frozen arms on all 108 model and instance cells from
Exp7129. The arms SHALL be `single_shot`, `self_review`, `exact_commitment`, and
`uncertainty_router`. Each arm SHALL have the same maximum generation-token
budget per cell. Prompts, instance payloads, exact checker receipts, model
identities, and controller seeds SHALL remain fixed before inference.

The model roster SHALL contain exactly, and in this order,
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Every Q4_K_M path SHALL resolve through
`cached_sota_pair()`. Inference SHALL use each GGUF's embedded llama.cpp chat
template. A retry SHALL use the same model family as the rejected proposal.

The exact checker SHALL compute each constraint penalty from the frozen formal
instance and direct parsed proposal. The uncertainty score SHALL use repeated,
bounded outputs and an explicit feature allowlist. Exact correctness, exact
penalties, hidden labels, solver receipts, witnesses, objectives, and solution
hashes SHALL not enter the learned uncertainty score. Model identity SHALL stay
explicit in every unit and aggregate. Pooled rows SHALL not replace model and
constraint-family rows.

The router SHALL expose `accept`, `retry`, and `abstain`. It SHALL accept only a
proposal with zero independently recomputed exact penalty. A retry receipt MAY
name failed constraint classes and counts. It SHALL not expose a correct answer,
witness, objective value, solution hash, or answer identifier. A rejected retry
after budget exhaustion SHALL abstain. No code path SHALL silently repair,
promote, or execute an exact-rejected action. Learned priority or confidence
SHALL not override exact rejection.

Before model setup, the command SHALL write a schema-complete terminal blocked
artifact. It SHALL recheck the bare Exp7129 producer field, the upstream bank
hash, all three cached models, two idle GPU leases, CUDA llama.cpp, and raw-trace
storage. A failed check SHALL leave `verdict_class: blocked`, use
`inference_substrate_class: blocked_no_run`, and preserve the failed check,
expected value, and observed value in `gate_check_summary`.

The complete artifact SHALL report exact success, parse rate, exact violations,
useful and harmful retry rates, abstention and its token cost, accepted-error
rate, relabel sensitivity, paraphrase consistency, tokens, and wall time per
model and instance. It SHALL retain one row per frozen arm and cell. It SHALL
also retain all uncertainty samples, exact penalties, routes, retries,
abstentions, accepted actions, and rejected promotion attempts. Missing unit
rows, arm-budget drift, uncertainty-summary drift, model pooling, or a changed
aggregate SHALL fail independent validation.

`verifier_committed_routing_complete_score` SHALL equal the bare integer one
only when the gate, model identities, token budgets, raw manifests, unit rows,
metric rows, and exact admission replay are complete. A positive verdict SHALL
also require `exact_rejected_actions_promoted == 0`, zero accepted errors, and a
non-collapsed reproducible uncertainty comparison. A completed metric non-uplift
SHALL remain a separate null claim. `verifier_is_oracle` SHALL be false because
exact outcomes are authority labels and not learned ranking inputs.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`run_date`, `MODEL_SPECS`, `models_used`, `model_repository_rows`,
`model_path_rows`, `model_hash_rows`, `inference_substrate`,
`inference_substrate_class`, `execution_venue`, `gpu_telemetry_rows`,
`token_rows`, `duration_s`, `source_artifact_hashes`, `upstream_bank_hash`,
`raw_trace_manifest`, `rows`, `arm_rows`, `exact_penalty_rows`,
`uncertainty_rows`, `routing_rows`, `retry_rows`, `abstention_rows`,
`accepted_action_rows`, `rejected_promotion_rows`, `relabel_sensitivity_rows`,
`paraphrase_consistency_rows`, `model_identity_confound_rows`,
`useful_retry_rate`, `accepted_error_rate`, `exact_rejected_actions_promoted`,
`verifier_committed_routing_complete_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL give one
non-empty scientific reason for every listed field. The live substrate SHALL
equal `model_bounded_generation: exact commitment and uncertainty routing`.
Its class SHALL be `model_bounded_generation`; a stable preflight block SHALL
use `blocked_no_run`. The execution venue SHALL be `host`. Verdict classes
SHALL use only `positive`, `circular_positive`, `null`, `blocked`,
`disqualified`, or `partial`. `honest_verdict` SHALL start with its class.

#### SCENARIO-VERIFY-7130-AUTHORITY: Exact Rejection Cannot Be Promoted

**Given** an exact-rejected proposal with any learned priority or confidence
**When** any arm selects, promotes, or executes an action
**Then** the admission guard denies the proposal
**And** the rejected attempt remains in `rejected_promotion_rows`.

#### SCENARIO-VERIFY-7130-LEAKAGE: Learned Uncertainty Excludes Exact Outcomes

**Given** a proposed uncertainty feature path
**When** it names or contains an exact label, penalty, solver receipt, witness,
objective, solution hash, or answer identifier
**Then** feature validation fails before scoring
**And** no exact outcome can improve learned priority.

#### SCENARIO-VERIFY-7130-UNCERTAINTY: Repeated Outputs Expose Collapse

**Given** repeated bounded outputs for one model and instance
**When** their parsed-answer distribution has no variation across the bank
**Then** the artifact records uncertainty collapse
**And** it cannot receive a positive verdict.

#### SCENARIO-VERIFY-7130-BUDGET: Arm Token Caps Match

**Given** the four frozen arm plans
**When** their maximum generation-token totals are compared
**Then** all totals are equal
**And** any changed stage cap fails validation.

#### SCENARIO-VERIFY-7130-IDENTITY: Model Pooling Cannot Hide A Reversal

**Given** completed rows from three model families and three constraint families
**When** metrics and surface effects are reduced
**Then** every model and family cell remains separate
**And** a pooled-only or missing identity row fails validation.

#### SCENARIO-VERIFY-7130-ROWS: Every Arm And Cell Remains Replayable

**Given** 108 frozen model-instance cells and four arms
**When** independent validation enumerates semantic keys
**Then** exactly 432 unique per-unit rows cover the planned product
**And** any missing, duplicate, or substituted row prevents completion.

#### SCENARIO-VERIFY-7130-RETRY: Feedback Is Bounded And Non-Oracle

**Given** an exact-rejected proposal that the router sends to retry
**When** it builds the verifier receipt and invokes the same model family
**Then** feedback contains only failed constraint classes and counts
**And** an exhausted rejected retry abstains without silent repair.

#### SCENARIO-VERIFY-7130-ARTIFACT: Rows Determine Metrics And Verdict

**Given** a blocked, partial, null, disqualified, or positive artifact
**When** an independent validator replays gates, rows, penalties, actions,
metrics, principles, and checksum
**Then** a consistent artifact passes
**And** any forged promotion, aggregate, verdict, or checksum fails.

## Implementation Status (REQ-VERIFY-7130)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-* | Implemented (`python/carnot/experiment_7130_v626_verifier_committed_routing.py`; `scripts/experiments/experiment_7130_v626_verifier_committed_routing.py`; `results/experiment_7130_v626_verifier_committed_routing.json`; three content-addressed raw shards) | Verified (`tests/python/test_experiment_7130_v626_verifier_committed_routing.py`; 16 focused tests, 432 complete unit rows, independent raw reduction and adversarial mutations, and 438/438 scoped statements covered) |

### REQ-VERIFY-7158: Counterfactual Entity Evidence Fixture SHALL Seal Exact Truth

Exp7158 SHALL build a deterministic entity-to-evidence fixture from the valid
Exp7138 relational source artifact. It SHALL load no language model. It SHALL
make no verifier-value claim. It SHALL use the four Exp7138 source families.
It SHALL preserve at least 96 rows and 30 evaluation pairs. Every row SHALL
retain its source span, claim span, entity IDs, relation, polarity, expected
answer, and exact `supported` or `unsupported` label.

Before it reads Exp7138 or another source, Exp7158 SHALL atomically write one
schema-complete running artifact. It SHALL then check the fixed date, Exp7138
artifact and readiness value, cached RAGTruth MIT license receipt, both named
capability specs, source modules, verifier package, source-family counts, and
output path. An unavailable input SHALL end as `blocked`. Its substrate class
SHALL be `blocked_no_run`. Its gate summary SHALL give the exact failed check,
expected value, and observed value.

Each base unit SHALL have a supported source row and controlled rows for entity
substitution, relation reversal, evidence removal, irrelevant evidence,
negation, numeric-unit change, and duplicate aliases. Irrelevant evidence and
duplicate aliases SHALL preserve support when the original evidence stays
present. Irrelevant-only evidence SHALL not preserve support. Every source and
claim offset SHALL use zero-based Unicode code points with an exclusive end.
The indexed text SHALL equal the referenced slice.

The source family SHALL determine `train`, `calibration`, or `evaluation`
before calibration. One source family SHALL occur in only one split. At least
48 rows and 30 base pairs SHALL be in evaluation. Evaluation truth fields
SHALL not occur in a generation prompt or candidate energy input. Sealing
receipts SHALL retain truth hashes and the exact permitted field names.

The candidate energy SHALL contain these frozen terms:
`entity_presence`, `relation_role_agreement`, `polarity`,
`quantity_unit_agreement`, and `counterfactual_sensitivity`. Weight and tie
selection SHALL read calibration rows only. Evaluation labels and expected
answers SHALL not enter the fit or score. The exact support labels remain the
fixture authority. The candidate energy SHALL not become that authority.

Exp7158 SHALL run exact term mutations. Entity substitution SHALL change only
entity presence. Relation reversal SHALL change only relation-role agreement.
Negation SHALL change only polarity. A numeric-unit change SHALL change only
quantity and unit agreement. Evidence removal SHALL change entity presence and
counterfactual sensitivity. Adding irrelevant evidence and duplicate aliases
SHALL leave all energy terms unchanged. Each mutation SHALL remain as one
`mutation_test_rows` entry.

`counterfactual_fixture_ready_score` SHALL be the bare integer one only when
row counts, family coverage, split isolation, evaluation-pair count, exact
labels, span offsets, truth sealing, deterministic hashes, calibration-only
fitting, and every mutation pass. This score states fixture readiness only.
It does not state accuracy, uplift, AUROC, or verifier value.

The artifact SHALL contain `field_principles`, `status`,
`preconditions_checked`, `run_date`, `inference_substrate`,
`inference_substrate_class`, `execution_venue`, `duration_s`,
`source_artifact_hashes`, `rows`, `source_family_rows`,
`entity_evidence_rows`, `perturbation_rows`, `split_rows`,
`sealed_field_rows`, `energy_term_contract`, `mutation_test_rows`,
`frozen_fixture_ids`, `counterfactual_fixture_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL contain the
task-specified principle for every field. The positive substrate SHALL equal
`exact_source_fixture_construction`. Its class SHALL equal
`cpu_exact_solver_or_simulator`. `execution_venue` SHALL equal `host`.
`verifier_is_oracle` SHALL be false. The verdict class SHALL use only
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. The honest verdict SHALL have a class-consistent terminal prefix.

#### SCENARIO-VERIFY-7158-PREFLIGHT: Missing Inputs Finish Blocked

Given the complete first write and any unavailable required input,
When Exp7158 runs preflight,
Then it writes one schema-complete blocked artifact,
And the gate summary preserves the first exact expected-observed failure.

#### SCENARIO-VERIFY-7158-PAIRS: Controlled Changes Preserve Exact Labels

Given one source-backed base unit,
When its nine conditions are materialized,
Then the supported and unsupported labels follow the exact condition contract,
And entity, relation, polarity, unit, removal, irrelevant, and alias controls
remain paired to that base.

#### SCENARIO-VERIFY-7158-SPANS: Source And Claim Offsets Replay

Given every source, claim, and entity span,
When an independent validator indexes its named text,
Then the zero-based exclusive slice equals the stored text and hash,
And a changed boundary or text fails readiness.

#### SCENARIO-VERIFY-7158-SPLITS: Source Families Cannot Leak

Given the frozen source-family split map,
When rows and independent pairs are counted,
Then no family crosses a partition,
And evaluation contains at least 48 rows and 30 paired base units.

#### SCENARIO-VERIFY-7158-SEALING: Evaluation Truth Stays Out Of Consumers

Given one evaluation row with exact truth,
When a generation prompt and candidate energy input are built,
Then neither contains a label, expected answer, condition, or truth alias,
And the sealing receipt binds both projections to the hidden truth hash.

#### SCENARIO-VERIFY-7158-ENERGY: Calibration Alone Freezes The Contract

Given frozen calibration rows and sealed evaluation rows,
When weights, threshold, and the zero-energy tie are selected,
Then the fit receipt names only the calibration family and row IDs,
And changing evaluation truth cannot change the energy contract.

#### SCENARIO-VERIFY-7158-MUTATIONS: Only Intended Terms Change

Given the supported energy vector for one base unit,
When each controlled perturbation is scored,
Then its changed and invariant term sets match the frozen mutation schedule,
And any extra changed term fails the mutation receipt.

#### SCENARIO-VERIFY-7158-ARTIFACT: Rows Determine Readiness And Hashes

Given a blocked, disqualified, or ready artifact,
When independent validation recomputes fields, sources, spans, labels, splits,
sealing, energy vectors, mutations, state, and checksum,
Then a consistent artifact passes,
And a forged aggregate, row, source hash, or terminal state fails.

## Implementation Status (REQ-VERIFY-7158)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7158 and SCENARIO-VERIFY-7158-* | Implemented: deterministic fixture module, CLI wrapper, MIT source receipt, and terminal artifact schema | Verified by focused RED/positive/adversarial tests with scoped 100% new-module coverage, plus the task-owned CLI validator and repository artifact gates |

### REQ-VERIFY-7167: Qwen3.8 Claim And Evidence Capture SHALL Stay Separate From Value

Exp7167 SHALL write a schema-complete artifact shell and raw manifest before it
checks resources. It SHALL select exactly 48 Exp7158 evaluation rows with a
fixed seed before model output exists. The selection SHALL contain 12 sealed
pair groups. It SHALL balance source families and controlled conditions. Each
schedule row SHALL bind its fixture ID, pair ID, input hashes, prompt, output
schema, decoding parameters, order, and token budget.

The model view SHALL contain the claim, evidence, neutral identifiers, and the
frozen extraction request. Exact labels, expected answers, conditions, and
truth rules SHALL remain in an authority-only sidecar. No authority field or
value SHALL occur in a rendered model prompt.

`MODEL_SPECS` SHALL contain exactly one entry. Its repository SHALL be
`unsloth/Qwen3.8-27B-GGUF`. Its file SHALL be
`Qwen3.8-27B-Q4_K_M.gguf`. Its quantization SHALL be `Q4_K_M`. Its role SHALL
be `headline_full_generation`. The runner SHALL use the tokenizer and chat
template embedded in this GGUF through llama.cpp. It SHALL not download or
substitute a model.

The live path SHALL require the exact ready Exp7158 artifact and hash, the
exact cached model, a CUDA-linked native runner, one idle RTX 3090, canonical
lease support, writable raw storage, tests, this requirement, and the output
path. A stable failed gate SHALL finish as `blocked` with
`inference_substrate_class=blocked_no_run`. Its `gate_check_summary` SHALL
preserve the failed check and exact observed resource state. It SHALL not wait,
stop an unowned process, or invoke a model.

The live path SHALL acquire one unique task-owned lease before load. It SHALL
start one owned process group inside `try/finally`. Receipts SHALL bind the
repository, file, revision, bytes, SHA-256, runner version, CUDA offload logs,
GPU UUID, PID, port, lease, load duration, and task-owned VRAM. Teardown SHALL
prove release of only that PID, port, lease, and VRAM.

The runner SHALL request one bounded structured result for each selected row
with identical decoding limits. Each result SHALL request claim entities and
facts, evidence entities and facts, a cited evidence span, missing-field
flags, a direct `supported`, `unsupported`, or `abstain` decision, and a short
rationale. The parser SHALL preserve invalid output as a failure row. It SHALL
not repair output. Every raw output and complete response SHALL be stored and
hashed.

The producer SHALL checkpoint atomically after every four terminal rows. A
resume SHALL accept only the same schedule, model, prompt, source, and raw
manifest identities. It SHALL not rerun complete rows. Row inclusion SHALL not
depend on accuracy, an energy, a judge, or any aggregate result.

`claim_evidence_trace_ready_score` SHALL be one only when all 48 rows, raw
hashes, generation receipts, checkpoint receipts, CUDA evidence, and teardown
evidence pass cold validation. A complete result SHALL use
`inference_substrate=live_qwen38_claim_evidence_full_generation`,
`inference_substrate_class=model_full_generation`, and
`verdict_class=positive` only for the transport claim. Resource failure SHALL
use `blocked`. Incomplete local implementation MAY use `partial`.
`verifier_is_oracle` SHALL be false.

`execution_venue` SHALL equal `host`, which is the repository-wide closed
venue value. The host name and selected GPU UUID SHALL remain in the resource
and telemetry receipts. A blocked pre-invocation result SHALL not invent a
selected GPU UUID.

The artifact SHALL contain `field_principles`, `status`,
`preconditions_checked`, `run_date`, `inference_substrate`,
`inference_substrate_class`, `execution_venue`, `duration_s`,
`source_artifact_hashes`, `rows`, `MODEL_SPECS`, `sealed_schedule_rows`,
`generation_receipts`, `claim_evidence_trace_rows`, `parser_failure_rows`,
`gpu_telemetry_rows`, `teardown_receipt`,
`claim_evidence_trace_ready_score`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. Each required field SHALL have its
task-specified scientific principle.

#### SCENARIO-VERIFY-7167-SELECTION: The Schedule Is Sealed Before Output

Given the exact Exp7158 artifact and fixed seed,
When selection runs without authority labels in its sort key,
Then it returns the same 48 ordered rows in 12 groups,
And source-family and controlled-condition counts satisfy the frozen balance.

#### SCENARIO-VERIFY-7167-IDENTITY: A Model Substitution Fails Closed

Given a schedule and resource receipt,
When the repository, filename, quantization, revision, or content hash changes,
Then validation fails before generation or rejects the terminal artifact.

#### SCENARIO-VERIFY-7167-BLINDING: Labels Never Reach The Prompt

Given any schedule row or rendered prompt,
When an authority key or exact label value is added,
Then the label-isolation audit fails.

#### SCENARIO-VERIFY-7167-PARSE: Invalid Output Remains In The Denominator

Given malformed JSON or a record with a missing required field,
When parsing runs,
Then it returns a terminal parser-failure row with the raw hash,
And no field is inferred or repaired.

#### SCENARIO-VERIFY-7167-RESUME: Checkpoints Bind The Frozen Run

Given an atomic checkpoint after four rows,
When the same sealed identities resume,
Then those four rows are not generated again.
When a schedule, prompt, source, model, or manifest identity differs,
Then resume fails closed.

#### SCENARIO-VERIFY-7167-OWNERSHIP: Lease And CUDA Evidence Bind Compute

Given an idle RTX 3090 and a fresh task lease,
When the owned server loads and generates,
Then receipts name the owned PID, port, GPU UUID, VRAM, and CUDA markers.
Missing ownership or CUDA evidence prevents readiness.

#### SCENARIO-VERIFY-7167-DURATION: Full Generation Has A Real Duration

Given any artifact with 48 completed generations,
When duration and generation receipts are validated,
Then the substrate class is `model_full_generation` and duration is at least
the measured model-load plus row-latency interval. A short copied duration
fails validation.

#### SCENARIO-VERIFY-7167-TEARDOWN: Cleanup Cannot Target Another Process

Given a recorded task-owned process identity and lease,
When cleanup runs,
Then it releases only the matching PID or process group, port, lease, and VRAM.
An identity mismatch refuses cleanup and prevents readiness.

#### SCENARIO-VERIFY-7167-ARTIFACT: Cold Validation Rebuilds Transport Readiness

Given a terminal artifact,
When cold validation checks fields, source hashes, schedule balance, label
isolation, raw hashes, parser accounting, checkpoint cadence, resource
receipts, duration, teardown, and checksum,
Then only a complete 48-row authentic capture can have readiness one.

## Implementation Status (REQ-VERIFY-7167)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7167 and SCENARIO-VERIFY-7167-* | Planned in `python/carnot/experiment_7167_v632_claim_evidence_trace_capture.py` and its CLI wrapper. | Planned in `tests/python/test_experiment_7167_v632_claim_evidence_trace_capture.py`. |

### REQ-VERIFY-7180: Symbolic Edits SHALL Separate Surface And Semantic Changes

Exp7180 SHALL build a deterministic symbolic-edit fixture from the exact ready
Exp7158 artifact. It SHALL select 48 distinct source-backed base instances.
It SHALL route eight bases to each of six relation families. Two whole
families and 16 bases SHALL form calibration. Four whole families and 32 bases
SHALL form evaluation. A base and all its variants SHALL remain in one split.

Each base SHALL have exactly four variants: original, bijective entity rename,
relation or polarity flip, and evidence deletion. The fixture SHALL contain
exactly 192 model-visible rows. The bijective rename SHALL preserve the
authority decision. The two semantic edits SHALL change it. The fixture SHALL
retain every negative decision and every missing-evidence response.

A small symbolic interpreter SHALL create exact labels. An independent SQLite
query SHALL cross-check every decision. The candidate scorer SHALL not call
either authority path. Exact correct tuples MAY define an oracle upper bound,
but they SHALL NOT count as a deployable extraction arm.

The generation view SHALL contain only `unit_id`, `text`, and
`response_schema`. The unit ID SHALL be opaque. The view SHALL not contain
structured ground truth, split names, edit labels, support hashes, or expected
answers. Authority labels SHALL be frozen in a separate sidecar. Changing an
authority label SHALL not change any generation-view byte.

The response schema SHALL contain `direct_decision`, `claim_tuple`,
`evidence_tuple`, `source_start`, `source_end`, and `missing_fields`. A tuple
SHALL contain `subject`, `relation`, `object`, `polarity`, and only applicable
`quantity` and `unit` values. The candidate energy SHALL use only generated
tuples and the supplied source bytes. Its frozen terms SHALL check tuple
alignment, polarity, quantity and unit agreement, required fields, and exact
literal span validity.

The score contract SHALL freeze weights, threshold, tie handling, calibration
IDs, scoring denominators, two seeded evidence-shuffle controls, one
label-permutation control, and comparison arms before inference. The comparison
arms SHALL be `baseline_direct`, `energy_from_extracted_tuples`,
`lexical_overlap`, `syntax_only`, and `shuffled_evidence`. The contract SHALL
mark exact correct tuples as an oracle upper bound and not a deployable arm.

Exp7180 SHALL write a schema-complete running checkpoint below
`results/checkpoints/` before fallible work. It SHALL verify the run date,
driving requirement, exact Exp7158 bytes, exact Exp7158 same-milestone gate
fields, SQLite availability, Python executable, and writable output directories
before measurement. An external failure SHALL write one terminal blocked
artifact with exact expected and observed values in `gate_check_summary`.

The terminal artifact SHALL be
`results/experiment_7180_v633_symbolic_edit_fixture.json`. It SHALL contain
`field_principles`, `status`, `preconditions_checked`, `run_date`,
`inference_substrate`, `execution_venue`, `duration_s`,
`source_artifact_hashes`, `rows`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, `honest_verdict`, `inference_substrate_class`,
`fixture_ready_score`, `generation_view_path`, `authority_sidecar_path`,
`split_manifest`, `score_contract`, and `mutation_rows`. It SHALL also bind
the two sidecars in `sidecar_hashes`, the fixed `study_question`, its
fixture-scoped `scope_answer`, and all `structural_checks`. Every artifact field
SHALL have one field principle.

`fixture_ready_score` SHALL equal the bare integer one only when all 192 rows
exist, labels stay isolated, both authority implementations agree exactly,
splits stay family-isolated, mutations have their intended outcomes, sidecar
hashes replay, and all frozen controls validate. This score states fixture
readiness only. It SHALL not claim a live model result or verifier value.
The positive substrate SHALL equal `exact_source_fixture_construction`.
Its class SHALL equal `cpu_exact_solver_or_simulator`.
`execution_venue` SHALL equal `host`. `verifier_is_oracle` SHALL be false.

#### SCENARIO-VERIFY-7180-PREFLIGHT: Exact Gates Fail Closed

Given a complete checkpoint and any missing or changed prerequisite,
When Exp7180 runs preflight,
Then it writes a terminal blocked artifact,
And its gate summary names the exact upstream, field, expected, and observed value.

#### SCENARIO-VERIFY-7180-SPLITS: Whole Relation Families Stay Isolated

Given 48 selected bases across six relation families,
When split routing is frozen,
Then two complete families contain 16 calibration bases,
And four complete families contain 32 evaluation bases with no base leakage.

#### SCENARIO-VERIFY-7180-VARIANTS: Surface And Semantic Edits Differ

Given one source-backed base fact,
When its four variants are interpreted,
Then original and bijective rename decisions match,
And relation or polarity flip and evidence deletion decisions change.

#### SCENARIO-VERIFY-7180-AUTHORITY: Independent Labels Agree

Given all 192 private symbolic records,
When the symbolic interpreter and SQLite query label them independently,
Then every decision and expected response agrees exactly,
And the scorer never calls either authority implementation.

#### SCENARIO-VERIFY-7180-BLINDING: Labels Cannot Change Generation Bytes

Given the generation view and separate authority sidecar,
When an authority label is changed,
Then the generation-view bytes and hash stay identical,
And any forbidden field or value in the generation view fails readiness.

#### SCENARIO-VERIFY-7180-ENERGY: Only Candidate Outputs And Source Bytes Score

Given a generated compact response and its supplied source bytes,
When candidate energy is computed,
Then only tuple agreement, polarity, quantity and unit, required fields, and
literal source spans contribute,
And exact authority labels are not an input.

#### SCENARIO-VERIFY-7180-CONTROLS: Contracts Freeze Before Inference

Given calibration rows only,
When the score contract is frozen,
Then weights, threshold, denominators, five comparison arms, two shuffle
controls, and one label permutation remain deterministic under evaluation-label changes.

#### SCENARIO-VERIFY-7180-ARTIFACT: Cold Validation Replays Readiness

Given a complete, blocked, or tampered artifact and its sidecars,
When cold validation replays sources, rows, labels, SQLite agreement, splits,
mutations, generation isolation, controls, hashes, terminal state, and checksum,
Then only the complete untampered fixture can have readiness one.

## Implementation Status (REQ-VERIFY-7180)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7180 and SCENARIO-VERIFY-7180-* | Implemented: deterministic 192-row fixture, isolated generation and authority sidecars, symbolic and SQLite authorities, frozen score controls, CLI wrapper, and terminal artifact contract. | Verified by focused RED and positive tests, fail-closed mutations, scoped 100% new-module coverage, file-to-parser validation, and repository artifact gates. |

### REQ-VERIFY-7181: Qwen3.8 Symbolic Traces SHALL Preserve Blind Runtime Evidence

Exp7181 SHALL read only the exact Exp7180 generation view during model work.
It SHALL not read the authority sidecar. The frozen schedule SHALL contain all
192 generation rows in their existing order. Each row SHALL bind the opaque
unit ID, exact prompt bytes, response schema, decoding parameters, and a maximum
of 192 output tokens. `MODEL_SPECS` SHALL equal exactly
`[{"hf_id":"unsloth/Qwen3.8-27B-GGUF","quantization":"Q4_K_M"}]`.

The live path SHALL resolve `cached_current_model()` to the exact cached GGUF.
It SHALL verify a CUDA-linked native llama.cpp server and the embedded chat
template. It SHALL require writable result, checkpoint, and raw-log paths. It
SHALL acquire one idle RTX 3090 with the canonical task-owned lease before
model load. Receipts SHALL bind the GPU UUID, free VRAM, lease ID, task PID,
server PID, server PID start time, selected model path, revision, byte count,
content hash, and native runner version.

The runner SHALL use the GGUF chat template. It SHALL first make one separate
canary request with at most 32 output tokens. A canary-only run SHALL use
`inference_substrate_class=model_bounded_generation`. It SHALL not use the
60-second full-generation class. A successful canary SHALL permit one
deterministic structured completion per scheduled row. The measurement SHALL
have a 2,100-second cap. Each request SHALL have a 120-second cap. The total
execution SHALL have a 3,600-second cap. No repair request is permitted.

Every scheduled row SHALL end with a completion or request-error receipt.
Each completion row SHALL retain the prompt and completion bytes, byte hashes,
token counts, truncation state, direct decision, extracted tuples, source span,
parse state, request error, model process identity, lease identity, and timing.
Parse failures and negative decisions SHALL remain in the denominator.
`trace_capture_complete_score=1` SHALL mean all 192 rows have terminal receipts
and all provenance checks pass. It SHALL not require successful parsing or a
correct answer.

The producer SHALL write the running shell below `results/checkpoints/` before
fallible work. It SHALL checkpoint exact prompt and completion bytes after each
eight terminal rows. A checkpoint SHALL bind the frozen schedule, model,
generation view, decoding contract, and all included row hashes. Resume SHALL
reject changed identities. The terminal result path SHALL contain only a
terminal artifact.

The worker SHALL print and flush every numbered phase boundary. A heartbeat
outside each blocking native call SHALL report elapsed time, completed units,
and the current operation at least every 60 seconds. Model load, generation,
benchmark, validation, subprocess, cleanup, and final writes SHALL have start
and end lines. A local interruption with resumable owned work SHALL be
`partial`. An unchanged external prerequisite failure SHALL be `blocked`.

Cleanup SHALL use the shipped native supervisor. It SHALL signal only the
recorded owned process identity. An identity mismatch SHALL refuse cleanup.
The lease SHALL close in `finally`. A full multi-row run SHALL use
`inference_substrate_class=model_full_generation`. A pre-invocation block SHALL
use `blocked_no_run`. `verifier_is_oracle` SHALL be false.

The terminal artifact SHALL contain `field_principles`, `status`,
`preconditions_checked`, `run_date`, `inference_substrate`, `execution_venue`,
`duration_s`, `source_artifact_hashes`, `rows`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, `honest_verdict`, `inference_substrate_class`,
`trace_capture_complete_score`, `MODEL_SPECS`, `model_specs`, `raw_manifest`,
`gpu_receipts`, `phase_spans`, `runner_receipt`, and `completion_rows`. Each
field SHALL have the task-specified principle. `execution_venue` SHALL be
`host`. `run_date` SHALL be `20260910`.

#### SCENARIO-VERIFY-7181-PREFLIGHT: Exact Inputs Fail Closed

Given the driving requirement, Exp7180 artifact, generation view, model cache,
native runner, GPU inventory, lease runtime, and output paths,
When any exact check fails before model invocation,
Then Exp7181 writes a terminal blocked artifact,
And the gate summary records its upstream, field, expected value, and observed value.

#### SCENARIO-VERIFY-7181-BLINDING: The Worker Cannot Read Authority

Given the three-field Exp7180 generation view and a separate authority sidecar,
When the 192-row schedule is built,
Then prompts depend only on `unit_id`, `text`, and `response_schema`,
And any forbidden field or authority-sidecar access fails validation.

#### SCENARIO-VERIFY-7181-CHECKPOINT: Eight Rows Bind Exact Bytes

Given eight terminal completion or request-error rows,
When a checkpoint is written,
Then it contains exact prompt and completion bytes and their hashes,
And any changed schedule, model, source, decoding, or row hash prevents resume.

#### SCENARIO-VERIFY-7181-CANARY: Bounded Work Is Not Full Generation

Given an owned CUDA server and a canary with at most 32 output tokens,
When the canary completes but no scheduled row runs,
Then the substrate class is `model_bounded_generation`,
And the trace capture score remains zero.

#### SCENARIO-VERIFY-7181-OWNERSHIP: Cleanup Refuses A Changed Process

Given the recorded task-owned server PID and start time,
When cleanup observes a different process identity,
Then the shipped supervisor refuses to signal it,
And terminal provenance readiness remains zero.

#### SCENARIO-VERIFY-7181-ROWS: All Outcomes Stay In The Denominator

Given the frozen 192-row schedule,
When generation returns valid JSON, malformed JSON, truncation, or request error,
Then one terminal completion row preserves the observed outcome for each unit,
And parse success is not required for capture completeness.

#### SCENARIO-VERIFY-7181-TERMINAL: Work Determines Classification

Given a pre-invocation external failure, canary-only work, incomplete scheduled
work, or complete scheduled work,
When the terminal artifact is classified,
Then it uses `blocked_no_run`, `model_bounded_generation`, or
`model_full_generation` according to the work that actually ran,
And only complete authentic 192-row capture can set readiness to one.

#### SCENARIO-VERIFY-7181-ARTIFACT: Cold Validation Replays Capture

Given a terminal artifact and its raw manifest,
When cold validation recomputes fields, sources, schedule isolation, row hashes,
checkpoint cadence, model identity, GPU ownership, cleanup, spans, and checksum,
Then an untampered complete or honest blocked artifact passes,
And any forged readiness, source, row, or terminal class fails.

## Implementation Status (REQ-VERIFY-7181)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7181 and SCENARIO-VERIFY-7181-* | Planned in the Exp7181 module and CLI wrapper. | Planned in focused tests before implementation. |

### REQ-VERIFY-7182: Frozen Grounding Energy SHALL Face Independent Authority

Exp7182 SHALL audit the frozen Exp7180 grounding-energy contract against the
independent Exp7180 authority sidecar. It SHALL read the exact Exp7181 raw
model outputs and reparse them without repair. It SHALL not use Exp7158
`energy_input` values as candidate features. Candidate scoring SHALL use only
the generated tuples, generated direct decision, generated parse state, and
the source bytes supplied to the model.

The threshold SHALL be selected only from the 64 calibration rows in the
`precedes` and `follows` families. Candidate thresholds SHALL be `-1` and each
distinct observed integer energy. Selection SHALL maximize full-denominator
accuracy, then minimize false accepts, then minimize false rejects, then choose
the smallest threshold. Energy equal to the threshold SHALL mean `supported`.
The threshold SHALL be frozen before evaluation labels are read. The audit
SHALL evaluate the 32 held-out base groups and all 128 of their variants. It
SHALL compare `baseline_direct`, `energy_from_extracted_tuples`,
`lexical_overlap`, `syntax_only`, and `shuffled_evidence` on the same unit IDs.

Every parse failure SHALL become an abstention. Parse failures SHALL remain in
the full 128-row denominator for every arm. Each per-unit arm row SHALL retain
its unit ID, base ID, relation family, variant, authority label, parse state,
prediction, score, error, abstention, correctness, false accept, false reject,
harmful flip, and exact feature lineage. The artifact SHALL report parse rate,
coverage, false accepts, false rejects, accuracy, harmful flips, rename
invariance, and semantic-edit sensitivity for each arm.

Paired deltas SHALL use 10,000 cluster-bootstrap draws over the 32 held-out
base groups. Each draw SHALL sample eight base groups within each of the four
evaluation families. All four variants for a selected base SHALL remain in the
same cluster. The random seed, paired unit IDs, draw count, and interval method
SHALL be frozen. Any changed denominator, duplicate or missing unit, shuffled
pair identity, or family mismatch SHALL fail validation.

A fresh Python process SHALL independently reconstruct the fixed energy
formula and every candidate decision. The auditor SHALL not import the
candidate scorer module and SHALL not adapt the threshold. It SHALL swap
evidence within family, delete each energy term, and permute authority labels.
It SHALL trace every feature to an exact raw model-output hash or supplied
source-byte hash. It SHALL preserve all disagreements and failed interventions
without repairing candidate aggregates.

Exp7182 SHALL write a schema-complete running checkpoint below
`results/checkpoints/` before fallible work. Before measurement, it SHALL print
and flush the phase start. It SHALL verify the driving requirement, required
source bytes, exact Exp7180 and Exp7181 hashes, their same-milestone gate
fields, Python executable, required tools, and writable output paths. An
external failure SHALL write a terminal blocked artifact with the exact failed
check, upstream, field, expected value, and observed value.

The terminal artifact SHALL be
`results/experiment_7182_v633_grounding_energy_audit.json`. It SHALL include
`field_principles`, `status`, `preconditions_checked`, `run_date`,
`inference_substrate`, `execution_venue`, `duration_s`,
`source_artifact_hashes`, `rows`, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, `honest_verdict`, `inference_substrate_class`,
`grounding_measurement_complete_score`, `grounding_value_score`,
`paired_metrics`, `feature_lineage_rows`, `intervention_rows`, and
`independent_audit_rows`. It SHALL also include `study_question`,
`scope_answer`, `arm_metrics`, `frozen_threshold_contract`, and
`audit_receipt`. Each artifact field SHALL have one field principle. The
positive executed substrate class SHALL be `no_model_load`.

`grounding_measurement_complete_score` SHALL equal the bare integer one when
all specified recomputations, controls, rows, bootstrap draws, and independent
audit outputs complete, including a complete null. `grounding_value_score`
SHALL equal one only when the held-out accuracy delta over direct decisions has
a CI95 lower bound greater than zero, the paired false-accept delta has a CI95
upper bound at most zero, no authority leakage exists, and semantic sensitivity
is better than both syntax and shuffled-evidence controls. If exact labels or
an equivalent authority rule drive candidate scoring, the verdict SHALL be
`circular_positive` and `verifier_is_oracle=true`. Complete non-wins SHALL be
`null`. Corrupt lineage, changed pairs, or unreconstructable provenance SHALL
be `disqualified`. The 32-group pilot SHALL make no broad model or benchmark
claim.

#### SCENARIO-VERIFY-7182-PREFLIGHT: Exact Inputs Fail Closed

Given the running checkpoint and any missing or changed required source, gate
field, tool, hash, requirement, or output path,
When Exp7182 checks preconditions before measurement,
Then it writes a terminal blocked artifact,
And the gate summary records exact expected and observed values.

#### SCENARIO-VERIFY-7182-LEAKAGE: Candidate Features Exclude Authority

Given generated rows and a separate authority sidecar,
When candidate features and the calibration contract are built,
Then no evaluation label or equivalent authority rule enters candidate scoring,
And injected authority data fails validation instead of being ignored.

#### SCENARIO-VERIFY-7182-PARSE: Failures Stay As Abstentions

Given valid, malformed, truncated, or otherwise rejected raw model outputs,
When all five arms are evaluated,
Then each failed parse produces one abstention row per arm,
And all failures remain in the full split denominator.

#### SCENARIO-VERIFY-7182-PAIRS: Unit And Cluster Identities Stay Fixed

Given 32 held-out bases with four variants each,
When arm metrics and bootstrap deltas are computed,
Then each arm uses the same ordered 128 unit IDs and unchanged base IDs,
And shuffled pair IDs, dropped rows, duplicate rows, or changed denominators fail.

#### SCENARIO-VERIFY-7182-METRICS: Metamorphic Outcomes Remain Visible

Given original, rename, semantic-flip, and evidence-deletion variants,
When one arm is summarized,
Then it reports all preregistered outcome counts and rates,
And rename invariance and semantic-edit sensitivity use complete base pairs.

#### SCENARIO-VERIFY-7182-AUDIT: Fresh Process Recomputes Without Candidate Imports

Given the frozen threshold, raw completions, generation rows, and authority rows,
When the independent auditor runs in a fresh process,
Then it implements the formula without importing the candidate scorer,
And it preserves decision disagreements, swaps, term deletions, label permutations,
and exact feature lineage.

#### SCENARIO-VERIFY-7182-VERDICT: Completion And Value Gates Stay Separate

Given complete held-out recomputation and paired intervals,
When the terminal verdict is classified,
Then measurement completion is one even for a null,
And value is one only when every preregistered value condition passes.

#### SCENARIO-VERIFY-7182-ARTIFACT: Cold Validation Replays The Full Audit

Given a complete, blocked, disqualified, or tampered artifact,
When cold validation replays hashes, rows, pairs, denominators, metrics,
lineage, interventions, independent decisions, gates, and checksum,
Then only an internally consistent terminal artifact passes,
And no aggregate disagreement is silently repaired.

## Implementation Status (REQ-VERIFY-7182)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7182 and SCENARIO-VERIFY-7182-* | Planned in the Exp7182 module, independent auditor, and CLI wrapper. | Planned in focused tests before implementation. |
| REQ-VERIFY-7182 and SCENARIO-VERIFY-7182-* (2026-09-10 implementation) | Implemented in the deterministic candidate module, separate standard-library auditor module, and two executable wrappers. The candidate reparses raw Exp7181 bytes and never reads Exp7158 energy inputs. | Verified by 24 focused tests and 775/775 covered statements. The final file-to-parser, repository suite, and artifact gates are recorded in the terminal artifact. |

### REQ-VERIFY-7195: Typed Source Relations SHALL Preserve Unknown Semantics

Exp7195 SHALL replace the failed Exp7182 overlap and threshold rule with a
typed relation executor. The executor SHALL accept explicit entity IDs,
relation operators, polarity, and byte offsets into the supplied source. It
SHALL validate relation and entity offsets against the exact raw UTF-8 bytes.
It SHALL normalize inverse direction, execute negation, and distinguish
`supported`, `contradicted`, and `unknown` outcomes.

Missing evidence, missing entity bindings, duplicate-name mappings,
contradictory evidence, invalid offsets, and unsupported operators SHALL yield
`unknown`. An unknown outcome SHALL set abstention and SHALL never become a
false or unsupported decision. Every result SHALL retain all uncertainty
reasons. Supported operators SHALL cover the six Exp7180 relation families and
their inverse or semantic-opposite mutations.

Exp7195 SHALL use Exp7180's shipped exact fixture builder to make a new sealed
panel. Seed `7195001` SHALL determine source selection and fresh entity
identities. The panel SHALL contain 16 calibration bases and 32 held-out bases.
Each base SHALL contain original, bijective rename, relation or polarity flip,
and evidence deletion variants, for 192 rows total. Family and template groups
SHALL be frozen before row materialization. No V633 unit or entity identity
SHALL enter the new panel.

The public sidecar SHALL expose only opaque unit IDs, raw source text, and raw
claim text. It SHALL not expose split, family, edit, expected decision, typed
authority tuples, canonical answer, or corpus label. A separate evaluator-only
sidecar SHALL retain split, edit metadata, typed inputs, and exact expected
outcomes. Changing evaluator truth SHALL not change any public byte.

The frozen contract SHALL define three atomic prompts: source-only relation
extraction, claim-only relation extraction, and direct support judgment. It
SHALL also define grammar-only schemas, scoring policy, call budgets, and the
complete 192-row sample budget before held-out labels are read. Source
extraction SHALL not see the claim. Claim extraction SHALL not see source or
direct judgment. No prompt SHALL contain evaluator-only fields.

Before readiness, Exp7195 SHALL run exact typed inputs and adverse mutations
for reversed arguments, negation, removed evidence, duplicate names, invalid
offsets, contradictory evidence, and an unsupported operator. All supported
fragment cases SHALL match the evaluator authority. Unsupported or incomplete
cases SHALL abstain. `typed_executor_ready_score` SHALL equal one only when
these semantic checks, panel isolation, prompt separation, sidecar hashes, and
cold replay all pass.

Exp7180, Exp7181, and Exp7182 SHALL be development evidence only. Exp7195 SHALL
reconstruct old failures under parse failure, source omission, entity binding,
relation orientation, negation, and unresolved evidence. Each category SHALL
retain at least one exact Exp7181 raw-output hash or explicitly report zero
observed rows. Exp7195 SHALL preserve Exp7182's failed value gate and SHALL not
tune or rescore the old rule.

Before consuming any upstream artifact, Exp7195 SHALL verify the exact source
bytes, required terminal fields, structured quarantine flags, and matching
entries in `ops/exclusion_manifest.yaml`. It SHALL reject quarantined upstream
data even when field gates pass. It SHALL write a schema-complete checkpoint
below `results/checkpoints/` before fallible work. A missing external
prerequisite SHALL produce a terminal blocked artifact whose
`gate_check_summary` names the failed check, upstream, field, expected value,
and observed value.

The terminal artifact SHALL be
`results/experiment_7195_v634_typed_grounding.json`. It SHALL contain
`field_principles`, `status`, `run_date`, `preconditions_checked`,
`inference_substrate`, `inference_substrate_class`, `execution_venue`,
`duration_s`, `source_artifact_hashes`, `rows`, `sample_size_budget`,
`random_seed`, `reproducibility_checksum`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`typed_executor_ready_score`, `error_decomposition_rows`, `fixture_manifest`,
`public_view_path`, `authority_sidecar_path`, `semantic_mutation_rows`,
`MODEL_SPECS`, and `model_invoked`. Each field SHALL have one declared
principle. `MODEL_SPECS` SHALL be empty and `model_invoked` SHALL be false.

Read-only V633 diagnosis SHALL use
`inference_substrate=aggregation_from_upstream_artifacts`. Executed typed
semantics and exact panel checks SHALL use
`inference_substrate_class=cpu_exact_solver_or_simulator`. A pre-computation
external block SHALL use `blocked_no_run`. Host CPU replay SHALL not claim live
Qwen inference. Because exact typed execution is scored against the fixture's
own correctness authority, readiness SHALL set `verifier_is_oracle=true` and
SHALL use `verdict_class=circular_positive` only for the prototype contract.
It SHALL not claim independent verifier value or broad model quality.

#### SCENARIO-VERIFY-7195-EXECUTION: Direction And Negation Execute Explicitly

Given valid typed source relations, entity bindings, and exact source bytes,
When the executor evaluates equivalent, reversed, and negated claims,
Then inverse-equivalent direction is supported,
And opposite direction or polarity is contradicted without using overlap.

#### SCENARIO-VERIFY-7195-UNKNOWN: Incomplete Semantics Abstain

Given missing evidence, ambiguous duplicate names, contradictory evidence,
invalid byte offsets, or an unsupported operator,
When the executor evaluates a claim,
Then the result is unknown with visible reasons and abstention true,
And unknown is never emitted as contradicted or unsupported.

#### SCENARIO-VERIFY-7195-PANEL: A Fresh Sealed Panel Stays Isolated

Given Exp7158 source-backed rows and seed 7195001,
When the shipped Exp7180 fixture builder materializes the new panel,
Then 16 calibration and 32 held-out bases each have four variants,
And all public unit and entity identities differ from V633 identities.

#### SCENARIO-VERIFY-7195-BLINDING: Producer Views Contain No Authority

Given the public and evaluator-only sidecars,
When prompt inputs are scheduled,
Then the public rows contain only unit ID, source text, and claim text,
And changing evaluator truth leaves the public bytes unchanged.

#### SCENARIO-VERIFY-7195-CONTRACTS: Atomic Calls Freeze Before Evaluation

Given the new panel and unopened held-out outcomes,
When the generation contract is frozen,
Then source, claim, and direct prompts remain mutually separated,
And grammar schemas, budgets, scoring, grouping, and denominators are fixed.

#### SCENARIO-VERIFY-7195-DIAGNOSIS: Old Failures Stay Development Evidence

Given the exact V633 trace and audit artifacts,
When the old errors are decomposed,
Then every required category retains observed counts and raw-row hashes,
And Exp7182's failed value gate remains a failed diagnostic input.

#### SCENARIO-VERIFY-7195-PREFLIGHT: Quarantine And Missing Inputs Fail Closed

Given exact source files, upstream terminal fields, structured flags, the
exclusion manifest, tools, and output directories,
When any prerequisite is missing, changed, or quarantined,
Then the terminal artifact is blocked before qualifying computation,
And its gate summary records exact expected and observed values.

#### SCENARIO-VERIFY-7195-ARTIFACT: Cold Replay Recomputes Readiness

Given a complete, blocked, or tampered artifact and both sealed sidecars,
When cold validation replays sources, diagnostics, typed outcomes, mutations,
panel grouping, prompt separation, hashes, and terminal classification,
Then only the complete untampered prototype can have readiness one.

## Implementation Status (REQ-VERIFY-7195)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7195 and SCENARIO-VERIFY-7195-* | Implemented in `python/carnot/verify/experiment_7195_source_relation_executor.py`, `python/carnot/experiment_7195_v634_typed_grounding.py`, and the Exp7195 CLI wrapper. | Covered by `tests/python/test_experiment_7195_v634_typed_grounding.py`, including byte-offset defects, unknown semantics, fresh-panel isolation, producer blinding, old-error diagnosis, preflight quarantine, and cold artifact replay. |

### REQ-VERIFY-7196: Atomic Qwen Capture SHALL Preserve Separated Evidence

Exp7196 SHALL capture three separated outputs for each of Exp7195's 192 public
rows. The source extractor SHALL receive only `source_text`. The claim extractor
SHALL receive only `claim_text`. The direct judgment SHALL receive both public
texts. No extraction request SHALL contain the direct output or an evaluator-only
field. The worker SHALL not open Exp7195's authority sidecar.

The worker SHALL use the frozen Exp7195 prompt templates and syntax contracts.
It SHALL record the exact prompt-template hashes and instantiated prompt hashes.
Grammar constraints SHALL enforce only the shipped JSON syntax. They SHALL not
insert entities, relations, labels, or direct decisions. Source, claim, and direct
budgets SHALL be 128, 64, and 16 output tokens. Each logical call SHALL use one
draw, with no parse retry or regenerated row.

The schedule SHALL contain 576 logical receipts: three for each public row. A
source result MAY be reused only when the exact public-source hash matches a
prior source result. The reused receipt SHALL retain the original raw bytes and
name the cold call that produced them. Claim and direct results SHALL never use
this cache. The frozen 192-row view has 97 unique source hashes, so a complete
run SHALL make at most 481 cold requests and record 95 source cache hits.

Each logical receipt SHALL retain its unit ID, call type, request ordinal, input
hash, prompt, raw request bytes, raw response, raw output bytes, token budget,
token counts, finish reason, truncation, request error, parse state, unknown or
abstention state, cache status, elapsed time, server process identity, GPU UUID,
and lease ID. The worker SHALL persist each cold result or request failure before
starting the next call. It SHALL checkpoint complete logical receipts below
`results/checkpoints/`. The terminal result path SHALL never contain a running
shell.

Before consuming Exp7195, the worker SHALL hash its artifact and public view. It
SHALL check exact terminal gate fields, the frozen contract hash, public-view
receipt, structured `flagged_adversarial` state, and the exclusion manifest. It
SHALL also confirm that Exp7195 preserves the old failed grounding value as zero.
The worker SHALL not promote that failed value into a readiness gate. A changed,
missing, or quarantined upstream SHALL produce a terminal blocked artifact. Its
`gate_check_summary` SHALL name the failed check, upstream, field, expected value,
and observed value.

The worker SHALL resolve `unsloth/Qwen3.8-27B-GGUF` Q4_K_M with
`cached_current_model()`. It SHALL record the resolved revision, path, byte count,
content-addressed hash, embedded tokenizer metadata hash, and chat-template hash.
It SHALL use one owned native llama.cpp server and one task-owned GPU lease. A
read-only conflict check SHALL occur before acquisition. The worker SHALL not
reuse or signal an unowned process. A model-cache, CUDA, template, or GPU miss
SHALL block without simulation or model substitution.

The worker SHALL use the embedded chat template. It SHALL use llama.cpp's
non-thinking server option only when the probed runner help reports support. It
SHALL record exact server and decoding parameters. It SHALL set
`CARNOT_FORCE_LIVE=1`. An eight-token canary SHALL precede capture. Each request
SHALL have a 60-second cap. Capture SHALL have a 2,400-second cap. One model SHALL
use the native single-replica runner and SHALL not claim `DualGPURunner`.

The worker SHALL print and flush before every check and at every numbered phase
boundary. It SHALL print before and after model load, generation, benchmarks,
long subprocesses, cleanup, validation, and final writes. An external heartbeat
SHALL cover each blocking native call. Long loops SHALL report completed units
and elapsed time at least every 60 seconds. Child output SHALL stream unbuffered.

`atomic_capture_complete_score=1` SHALL require a terminal source, claim, and
direct receipt for every scheduled row plus authentic owned CUDA provenance.
Parsing success SHALL not be required. A complete parse-poor bank SHALL be
`verdict_class=null`, not blocked or partial, and SHALL remain available for a
later independent value audit. A pre-invocation external block SHALL use
`blocked_no_run`. A load-only interruption SHALL use `model_load_no_generation`.
A canary-only interruption SHALL use `model_bounded_generation`. Any scheduled
generation SHALL use `model_full_generation`. These classes SHALL apply duration
floors of 0, 2, 10, and 60 seconds to measured work only. The worker SHALL never
sleep or alter duration to meet a floor.

A complete capture SHALL use `inference_substrate=live_llm_inference` and
`inference_mode=live_gpu`. The result SHALL report cold and amortized request,
token, and latency costs. Invalid, truncated, unknown, and request-error counts
SHALL use all 192 rows as each call-type denominator. `verifier_is_oracle` SHALL
be false because this task captures candidates and does not score correctness.

The terminal artifact SHALL be
`results/experiment_7196_v634_qwen_atomic_capture.json`. It SHALL contain
`field_principles`, `status`, `run_date`, `preconditions_checked`,
`inference_substrate`, `inference_substrate_class`, `execution_venue`,
`duration_s`, `source_artifact_hashes`, `rows`, `sample_size_budget`,
`random_seed`, `reproducibility_checksum`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`atomic_capture_complete_score`, `MODEL_SPECS`, `model_specs`,
`completion_rows`, `phase_spans`, `gpu_receipts`, `runner_receipt`,
`raw_manifest`, and `inference_mode`. Each field SHALL have the task-specified
principle.

#### SCENARIO-VERIFY-7196-PREFLIGHT: Exact Inputs And Quarantine Fail Closed

Given Exp7195, its frozen public view, the exclusion manifest, model cache,
native runner, GPU inventory, and writable output paths,
When any exact byte, field, quarantine, tool, or resource check fails,
Then Exp7196 writes a terminal blocked artifact before qualifying computation,
And the gate summary records the exact expected and observed values.

#### SCENARIO-VERIFY-7196-BLINDING: Atomic Prompts Cannot Cross Contaminate

Given one public source and claim pair,
When the three logical calls are scheduled,
Then source sees only source text, claim sees only claim text, and direct sees both,
And no prompt or request exposes direct output, labels, split, edit, or authority.

#### SCENARIO-VERIFY-7196-PARSING: Syntax Failures Stay Observable

Given valid, malformed, truncated, unknown, and request-error responses,
When each logical receipt is built,
Then exact raw bytes, parse state, error, truncation, and abstention remain visible,
And no parser retry or semantic repair changes the observed response.

#### SCENARIO-VERIFY-7196-CACHE: Only Exact Source Bytes Reuse A Cold Result

Given repeated and changed public sources,
When the capture schedule executes,
Then only equal source hashes reuse one earlier cold source receipt,
And every cache hit names that receipt while claim and direct calls remain cold.

#### SCENARIO-VERIFY-7196-CHECKPOINT: Every Terminal Call Is Durable

Given a cold completion or request failure,
When the worker proceeds to another call,
Then the exact logical receipt is already durable under the checkpoint or raw path,
And changed input, prompt, model, contract, or row bytes prevent resume.

#### SCENARIO-VERIFY-7196-RUNTIME: One Owned CUDA Model Serves The Run

Given an idle task-ownable RTX 3090 and the resolved Qwen3.8 GGUF,
When capture runs,
Then one leased native server handles the canary and all cold requests,
And receipts bind owned process identity, CUDA use, model bytes, and lease lifecycle.

#### SCENARIO-VERIFY-7196-TERMINAL: Completed Poor Parsing Is A Null

Given all 576 logical terminal receipts with any combination of parse outcomes,
When the task classifies the run,
Then capture completeness is one despite unsuccessful calls,
And a parse-poor complete bank is null while incomplete owned work is partial.

#### SCENARIO-VERIFY-7196-ARTIFACT: Cold Replay Recomputes Capture Evidence

Given a terminal artifact, public view, raw manifest, and persisted call rows,
When cold validation replays schedule separation, hashes, parser states, cache use,
cost totals, model identity, CUDA ownership, cleanup, spans, and checksum,
Then an untampered complete or honest blocked artifact passes,
And forged completeness, provenance, source, or terminal class fails.

## Implementation Status (REQ-VERIFY-7196)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7196 and SCENARIO-VERIFY-7196-* | Implemented in `python/carnot/experiment_7196_v634_qwen_atomic_capture.py` with the executable wrapper under `scripts/experiments/`. | Covered by the focused Exp7196 tests, including frozen schedule separation, strict parsing, exact-source caching, checkpoints, terminal classification, and cold artifact validation. |

### REQ-VERIFY-7197: Typed Grounding Value SHALL Include Coverage And Independent Replay

Exp7197 SHALL audit Exp7195's sealed panel with Exp7196's exact raw calls. It
SHALL evaluate all 128 held-out rows in five fixed arms: direct judgment,
grammar-validity-only, typed execution with unknown abstention, lexical overlap,
and shuffled-source typed execution. A parse failure SHALL remain in every
applicable arm denominator. Unknown SHALL become abstention, not unsupported.

The five policies SHALL freeze before evaluation labels are opened. Direct
judgment SHALL use the exact direct-call bytes. Grammar-validity-only SHALL
accept only when both extraction calls are syntax valid. Typed execution SHALL
use the shipped Exp7195 relation executor on the two separated extractions.
Lexical overlap SHALL use only public source and claim bytes. Shuffled-source
execution SHALL use a frozen same-family, same-variant donor map. Evaluation
labels SHALL not enter extraction, policy selection, or donor selection.

Each arm SHALL report full-denominator correct count and accuracy, parse rate,
coverage, abstention, conditional accuracy, false accepts, false rejects,
harmful flips from a correct direct decision, rename consistency, semantic-edit
sensitivity, and cold and amortized latency. Extraction-arm latency SHALL include
both source and claim extraction. Cold source latency SHALL follow the recorded
cold receipt behind a cache hit. The artifact SHALL keep one row for every unit
and arm with the unit ID, arm, seed, metric, error, abstention, raw hashes, and
latency lineage.

Paired intervals SHALL use 10,000 deterministic cluster-bootstrap draws over 32
base cases. Draws SHALL sample eight bases inside each of four evaluation
families. All four variants of a base SHALL stay together. Variants and repeated
receipts SHALL not count as independent samples.

The accuracy criterion SHALL pass only when the paired lower confidence bound
for typed correct/128 minus direct correct/128 is strictly positive, typed false
accepts do not increase, and typed coverage is at least 0.60. The alternative
efficiency criterion SHALL pass only when its paired accuracy lower bound is at
least -0.02, coverage equals direct coverage, and measured direct latency divided
by typed latency is at least two. Typed latency SHALL include extraction. The
`acceptance_gate_value` and `grounding_value_score` SHALL equal one only when the
accuracy or efficiency criterion passes and the independent semantic audit
passes.

A fresh Python process SHALL reparse exact raw outputs and recompute all
predictions before it reads label bytes. It SHALL repeat argument reversal,
semantic deletion, and shuffled-source controls. It SHALL compare every typed
decision with the candidate process. It SHALL prove that authority-label
mutations do not change extraction or policy checksums. Separate code SHALL not
make an equivalent exact correctness rule oracle-distinct.

Before computation, Exp7197 SHALL print and flush each phase and each
precondition. It SHALL hash required sources, check exact Exp7195 and Exp7196
terminal fields, inspect structured and manifest quarantine signals, confirm the
known failed grounding value remains zero, and check tools and output
directories. A quarantined input SHALL fail before consumption. A missing or
changed external prerequisite SHALL write a terminal blocked artifact with an
exact `gate_check_summary`. Running checkpoints SHALL stay below
`results/checkpoints/` and SHALL never replace the terminal deliverable.

The task invokes no model. `MODEL_SPECS` SHALL be empty and `model_invoked`
SHALL be false. Completed execution SHALL use
`inference_substrate=verifier_ensemble_against_cached_candidates` and
`inference_substrate_class=cpu_exact_solver_or_simulator`. A pre-computation
external block SHALL use `blocked_no_run`. Upstream Qwen receipts SHALL not
become live inference provenance for this CPU audit.

The terminal artifact SHALL be
`results/experiment_7197_v634_grounding_value_audit.json`. It SHALL include
`field_principles`, `status`, `run_date`, `preconditions_checked`,
`inference_substrate`, `inference_substrate_class`, `execution_venue`,
`duration_s`, `source_artifact_hashes`, `rows`, `sample_size_budget`,
`random_seed`, `reproducibility_checksum`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`grounding_audit_complete_score`, `grounding_value_score`, `arm_metrics`,
`paired_interval_rows`, `oracle_distinctness_rationale`, `cold_audit_rows`,
`accuracy_criterion`, `efficiency_criterion`, `acceptance_gate_value`,
`MODEL_SPECS`, `model_invoked`, `scope_answer`, and `independent_audit`.
Every field SHALL have its declared principle.

Audit completion SHALL equal one after all rows, controls, draws, and fresh
process checks complete, including a null result. A passing value gate SHALL be
`circular_positive` with `verifier_is_oracle=true`, because the executor and
labels use the same complete fixture authority. A completed non-win SHALL remain
`null`. The result SHALL state that this 32-base pilot does not establish broad
model performance or a learned-verifier moat.

#### SCENARIO-VERIFY-7197-PREFLIGHT: Exact Inputs And Quarantine Fail Closed

Given required source bytes, upstream gate fields, quarantine signals, tools,
and output directories,
When one prerequisite is missing, changed, failed, or quarantined,
Then Exp7197 writes a schema-complete terminal blocked artifact,
And its gate summary names the upstream, field, expected value, and observation.

#### SCENARIO-VERIFY-7197-DENOMINATOR: Abstention Cannot Hide Parse Failure

Given 128 held-out rows with failed or unknown extraction calls,
When all five arms are scored,
Then every arm retains all 128 units and reports full-denominator correctness,
And conditional accuracy cannot substitute for coverage or correct/128.

#### SCENARIO-VERIFY-7197-POLICIES: Labels Cannot Select A Rule

Given public rows, cached model bytes, and sealed evaluator labels,
When candidate predictions and donor mappings freeze,
Then no label or expected-decision byte enters extraction or policy selection,
And changing only labels leaves the prediction checksum unchanged.

#### SCENARIO-VERIFY-7197-METRICS: Paired Variants Stay Clustered

Given 32 held-out bases with four variants each,
When arm metrics and 10,000 bootstrap draws are computed,
Then resampling is stratified by family at the base-case level,
And rename, edit, harmful-flip, error, coverage, and latency outcomes replay.

#### SCENARIO-VERIFY-7197-AUDIT: A Fresh Process Repeats Semantic Controls

Given exact raw calls and frozen candidate predictions,
When the independent evaluator process runs,
Then it reparses and executes before reading labels,
And argument reversal, semantic deletion, shuffled sources, and all decisions
are retained without repair.

#### SCENARIO-VERIFY-7197-VALUE: Completion Does Not Imply Value

Given a complete semantic audit and paired intervals,
When neither accuracy nor efficiency passes its preregistered clauses,
Then `grounding_audit_complete_score` is one and `grounding_value_score` is zero,
And the terminal verdict is a complete pilot-scope null.

#### SCENARIO-VERIFY-7197-CIRCULARITY: Equivalent Authority Is Oracle Use

Given a gain produced by typed execution against its complete fixture authority,
When the acceptance gate passes,
Then `verifier_is_oracle` is true and the verdict is `circular_positive`,
And separate implementations do not create a learned-verifier moat claim.

#### SCENARIO-VERIFY-7197-ARTIFACT: Cold Validation Replays The Audit

Given a complete, blocked, or tampered Exp7197 artifact,
When cold validation checks sources, rows, metrics, intervals, audit receipts,
criteria, hashes, and terminal fields,
Then only a consistent terminal artifact passes,
And changed denominators, labels in policy inputs, or promoted upstream nulls fail.

## Implementation Status (REQ-VERIFY-7197)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7197 and SCENARIO-VERIFY-7197-* | Planned in the Exp7197 candidate module, fresh-process auditor, and executable wrapper. | Planned in focused tests before implementation. |

### REQ-VERIFY-7208: Source-Span Relations SHALL Separate References From Semantics

Exp7208 SHALL build a finite source-span relation representation before any
new model call. Each relation SHALL contain a sentence index, half-open subject
byte offsets, an executor-supported predicate, half-open object byte offsets,
and an explicit positive or negative polarity. A source completion SHALL
contain at most four relations. A claim completion SHALL contain at most one
relation. Both completion types SHALL permit an explicit `unknown` outcome.

The compiler SHALL resolve offsets only against the public bytes supplied to
that request. It SHALL reject invalid offsets, spans outside the named
sentence, cross-document grammar receipts, non-entity spans, unsupported
predicates, invalid polarity, extra fields, and excess relation counts. A
valid reference SHALL certify only the origin and type of a term. It SHALL not
certify the truth of a relation or constrain a relation to a hidden label.
Source and claim requests SHALL remain independent.

Exp7208 SHALL freeze grammar-only and reference-restricted llama.cpp-compatible
GBNF contracts over the same tuple schema. Both arms SHALL receive identical
public bytes, token budgets, and model settings. The reference grammar MAY use
only sentence and entity spans derived from those public bytes. Each request
SHALL retain its public-input hash, grammar bytes, and grammar hash. The exact
semantic executor SHALL run after decoding and SHALL remain separate from both
grammars. This mechanism is a finite adaptation of semantic pruning. It SHALL
not claim to reproduce ChopChop.

The panel SHALL use seeds `7208001` for case construction and `7208002` for
label-preserving surface rendering. It SHALL contain 80 independent bases:
eight canary, eight development, and 64 held-out test bases. Each split SHALL
be balanced across four supported ordering families. Each base SHALL contain
exactly four variants: supported, direction or polarity reversal, joint
support, and support-removed unknown. Related alpha-renamed bases SHALL be
linked in metadata without adding rows. The public panel SHALL contain 320
rows, including 256 test rows. Seeds, splits, families, variants, and labels
SHALL not enter model input.

Exp7208 SHALL write a public JSONL file, an evaluator-only authority JSONL
file, and a manifest below `results/fixtures/experiment_7208/`. The authority
SHALL interpret the controlled-language public text without importing or
calling the candidate compiler or executor. It SHALL audit direction,
negation, transitive joint support, support removal, and alpha-renaming. The
public file SHALL expose only opaque unit IDs, source text, and claim text.
Changing authority labels SHALL not change public bytes.

A frozen lexical control SHALL retain all rows. It SHALL show whether matched
token-multiset pairs are indistinguishable, and it SHALL report its complete
score without removing easy cases. No held-out label or model outcome SHALL
select cases, thresholds, grammars, spans, or operators.

Before consuming Exp7196 or Exp7197, Exp7208 SHALL hash their exact bytes,
verify required producer fields, read Exp7196 raw completions only through its
authenticated raw manifest, and reject structured or manifest quarantine
independently of terminal field gates. It SHALL reproduce the source and claim
invalid and truncated counts from raw bytes. Its diagnosis SHALL distinguish
format overhead, missing terminators, repeated output, and semantic errors.
It SHALL preserve Exp7197's failed value as historical evidence and SHALL not
promote that known failure into readiness.

Before readiness, Exp7208 SHALL run compiler mutations for out-of-range,
cross-document, wrong-sentence, wrong-type, unsupported-predicate, invalid-
polarity, excess-count, direction, negation, removed-support, and grammar
serialization cases. `span_fixture_ready_score` SHALL equal one only when all
320 rows compile and execute, split hashes are disjoint, independent labels
agree, all mutations pass, grammar serialization round-trips, and sidecar
hashes match. If the embedded GGUF tokenizer is available without inference,
the artifact SHALL report measured minimum and maximum completion token sizes.
Otherwise those token counts SHALL stay unknown for the next canary.

This task SHALL use `MODEL_SPECS=[]`, `model_invoked=false`,
`execution_venue=host`, and the actual hostname in `execution_host`. Executed
panel compilation and semantics SHALL use
`inference_substrate_class=cpu_exact_solver_or_simulator`. Read-only upstream
diagnosis SHALL remain identified as aggregation. Schema readiness SHALL not
be reported as verifier value. Because the fixture authority defines expected
correctness, `verifier_is_oracle` SHALL be true and a ready artifact SHALL use
`verdict_class=circular_positive` with a narrow complete readiness verdict.

The terminal artifact SHALL be
`results/experiment_7208_v635_span_fixture.json`. It SHALL contain
`field_principles`, `status`, `run_date`, `preconditions_checked`,
`inference_substrate`, `inference_substrate_class`, `execution_venue`,
`execution_host`, `duration_s`, `source_artifact_hashes`, `rows`,
`sample_size_budget`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`span_fixture_ready_score`, `public_view_path`, `authority_sidecar_path`,
`split_manifest`, `grammar_contract`, `lexical_control_rows`,
`span_mutation_rows`, `MODEL_SPECS`, and `model_invoked`. Each required field
SHALL have its declared principle and actual evidence.

#### SCENARIO-VERIFY-7208-COMPILER: Public Spans Fail Closed

Given independent source or claim bytes and one bounded relation completion,
When the reference compiler resolves that completion,
Then only in-document entity spans and supported typed values compile,
And invalid, cross-document, wrong-type, or excessive input fails closed.

**Spec traces:** REQ-VERIFY-7208

#### SCENARIO-VERIFY-7208-GRAMMARS: Syntax And References Stay Separate

Given the same public bytes, tuple schema, budget, and settings,
When both decoding contracts compile,
Then the syntax arm contains no input or label facts,
And the reference arm contains only public sentence and entity references.

**Spec traces:** REQ-VERIFY-7208

#### SCENARIO-VERIFY-7208-PANEL: Fixed Splits Preserve Related Variants

Given both frozen seeds and 80 independent bases,
When the panel is materialized,
Then it has 320 rows, 256 held-out rows, four balanced families per split,
And related semantic and alpha-renamed variants never cross a split.

**Spec traces:** REQ-VERIFY-7208

#### SCENARIO-VERIFY-7208-AUTHORITY: Independent Semantics Own Labels

Given the public controlled-language source and claim text,
When the authority interpreter scores direction, negation, joint support,
support removal, and alpha-renaming,
Then it does not import the candidate path and agrees on every panel row.

**Spec traces:** REQ-VERIFY-7208

#### SCENARIO-VERIFY-7208-DIAGNOSIS: Old Transport Failures Stay Historical

Given Exp7196's authenticated raw manifest and Exp7197's terminal artifact,
When exact completion bytes are diagnosed,
Then invalid and truncated counts reproduce the upstream result,
And format, termination, repetition, and semantic causes stay distinct.

**Spec traces:** REQ-VERIFY-7208

#### SCENARIO-VERIFY-7208-PREFLIGHT: Upstream Gates And Quarantine Fail Closed

Given required sources, exact producer fields, raw receipts, tools, and paths,
When one prerequisite is absent, changed, unauthenticated, or quarantined,
Then Exp7208 writes a schema-complete terminal blocked artifact,
And its gate summary names the failed check and exact expected and observed values.

**Spec traces:** REQ-VERIFY-7208

#### SCENARIO-VERIFY-7208-ARTIFACT: Readiness Replays From Sealed Bytes

Given a complete, blocked, or tampered artifact and its three fixture files,
When cold validation replays hashes, grammars, compilers, semantics, controls,
mutations, denominators, fields, and checksum,
Then only a complete consistent panel can have readiness one.

**Spec traces:** REQ-VERIFY-7208

## Implementation Status (REQ-VERIFY-7208)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7208 and SCENARIO-VERIFY-7208-* | Planned in the Exp7208 module and executable wrapper. | Planned in focused tests before implementation. |

### REQ-VERIFY-7209: Source-Span Canary SHALL Qualify One Frozen Live Contract

Exp7209 SHALL consume only the canary split from the declared Exp7208
deliverable. It SHALL select the supported variant from each of eight bases.
It SHALL run separate source and claim requests for the `grammar_only` and
`reference` arms. The fixed schedule SHALL contain 32 cold calls. It SHALL not
read development or test labels. It SHALL not describe these calls as a full
generative benchmark.

Before generation, Exp7209 SHALL hash the exact Exp7208 artifact and sidecar
bytes. It SHALL verify `status=complete`, `run_date=20260911`,
`verdict_class=circular_positive`, `span_fixture_ready_score=1`, the exact
readiness verdict, the artifact checksum, and authenticated sidecar hashes.
It SHALL reject structured quarantine and exclusion-manifest entries as
independent checks. A pre-inference external failure SHALL write a terminal
blocked artifact. Its `gate_check_summary` SHALL name the check, upstream,
field, expected value, and observed value. The implementation SHALL unwrap
only exact `principle` and `value` wrappers. It SHALL not promote Exp7197's
known failed grounding value.

The model contract SHALL be
`[{"hf_id":"unsloth/Qwen3.8-27B-GGUF","quantization":"Q4_K_M"}]`.
Exp7209 SHALL resolve `cached_current_model()`. It SHALL record the GGUF path,
revision, content hash, embedded tokenizer hash, and embedded chat-template
hash. It SHALL use one task-owned native llama.cpp server and one task-owned
GPU lease. A read-only process and lease check SHALL precede ownership. It
SHALL never share, replace, or stop an unowned process. A cache, CUDA, runner,
idle-GPU, or lease miss SHALL block with no substitute model or simulation.
`CARNOT_FORCE_LIVE` SHALL equal one.

The server SHALL use the GGUF's embedded tokenizer and chat template. It SHALL
use context 8192, temperature zero, seed 7209001, source budget 384, claim
budget 128, and prompt caching off. It SHALL use supported non-thinking mode.
Each response SHALL also prove that no reasoning content or thought markers
were returned. Model load SHALL have a 240-second bound. Each call SHALL have
a 60-second bound. The live window SHALL have a 900-second bound. Repeated
deterministic transport faults SHALL stop the schedule.

The tokenizer SHALL measure the maximum serialized source and claim
completions before any call. Each maximum SHALL use no more than five sixths
of its fixed budget. A failed fit check SHALL produce a complete
representation-size null. It SHALL not increase either budget. The
`token_budget_receipt` SHALL retain measured denominators, maxima, budgets,
and headroom checks.

Every call SHALL retain its actual request payload, grammar bytes and hash,
raw response and completion, prompt and completion token counts, finish
reason, latency, parse result, truncation state, reasoning-disabled evidence,
CUDA owner receipt, and failure reason. Raw prompts, request settings, grammar
generator hash, tokenizer hash, model hash, and completion rows SHALL be
sealed below `results/raw/experiment_7209/`. Running checkpoints SHALL stay
below `results/checkpoints/`.

After capture, Exp7209 SHALL compile every output with the Exp7208 compiler.
It SHALL run the shipped typed executor on each source and claim pair. It SHALL
compare source and claim relations separately with canary authority. The
`reference` arm SHALL have all 16 calls complete, parse-valid, nontruncated,
and exact-reference-valid before readiness. At least seven of eight combined
reference-arm interpretations SHALL match canary authority.
`span_canary_ready_score` SHALL equal one only when all these checks pass.
Any completed gate failure SHALL be a terminal null. An external runtime or
GPU prerequisite that prevents execution SHALL be blocked. Readiness SHALL be
narrow canary evidence and SHALL not claim held-out verifier value.

Actual CUDA generation SHALL use `inference_substrate=live_llm_inference`,
`inference_mode=live_gpu`, and
`inference_substrate_class=model_bounded_generation`. A stopped canary also
uses the bounded-generation class. Bounded generation has a ten-second
authenticity floor, but Exp7209 SHALL not wait or pad `duration_s`. It SHALL
use `execution_venue=host` and put the actual hostname in `execution_host`.
`runner_receipt` SHALL record one model, one native server, and no
`DualGPURunner`. `phase_spans` SHALL separate preflight, loading, prefill,
generation, parsing, verification, and teardown. Task-owned `gpu_receipts`
SHALL overlap actual model work.

The terminal artifact SHALL be
`results/experiment_7209_v635_span_canary.json`. It SHALL contain
`field_principles`, `status`, `run_date`, `preconditions_checked`,
`inference_substrate`, `inference_substrate_class`, `inference_mode`,
`execution_venue`, `execution_host`, `duration_s`, `source_artifact_hashes`,
`rows`, `sample_size_budget`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`span_canary_ready_score`, `canary_rows`, `frozen_decoding_contract`,
`token_budget_receipt`, `MODEL_SPECS`, `model_invoked`, `phase_spans`,
`gpu_receipts`, `model_identity_receipt`, and `runner_receipt`. Each field
SHALL have its declared principle and actual evidence.

#### SCENARIO-VERIFY-7209-SCHEDULE: Canary Selection Stays Bounded And Blind

Given the authenticated Exp7208 sidecars,
When the canary schedule is frozen,
Then it contains eight supported bases and 32 separate source or claim calls,
And it reads no development or test labels.

**Spec traces:** REQ-VERIFY-7209

#### SCENARIO-VERIFY-7209-BUDGET: Embedded Token Counts Gate Representation Fit

Given the fixed source and claim forms and actual embedded tokenizer,
When completion sizes are measured before generation,
Then each maximum has 20 percent headroom inside its fixed budget,
And a failed fit becomes a complete null without a larger cap.

**Spec traces:** REQ-VERIFY-7209

#### SCENARIO-VERIFY-7209-PREFLIGHT: External Prerequisites Fail Closed

Given exact sources, producer fields, quarantine checks, tools, cache, and GPU state,
When one required external observation fails,
Then a schema-complete terminal blocked artifact is written,
And the gate summary retains expected and observed values.

**Spec traces:** REQ-VERIFY-7209

#### SCENARIO-VERIFY-7209-CAPTURE: Live Calls Retain Exact Transport Evidence

Given one owned CUDA server and the frozen schedule,
When each bounded call returns or fails,
Then its request, grammar, raw bytes, token counts, finish state, latency, and owner persist,
And repeated deterministic transport faults stop further calls.

**Spec traces:** REQ-VERIFY-7209

#### SCENARIO-VERIFY-7209-REASONING: Non-Thinking Mode Is Observed

Given the embedded chat template and supported non-thinking server option,
When a response completes,
Then neither reasoning content nor thought markers are present,
And the receipt does not infer this result from display settings alone.

**Spec traces:** REQ-VERIFY-7209

#### SCENARIO-VERIFY-7209-EXECUTION: Captured References Reach The Executor

Given separate source and claim completions,
When public compilation and typed execution finish,
Then exact references and relations are compared separately with canary authority,
And the combined decision uses the shipped executor after capture.

**Spec traces:** REQ-VERIFY-7209

#### SCENARIO-VERIFY-7209-READINESS: Only Complete Usable Span Calls Unlock

Given all 16 reference-arm calls and eight combined interpretations,
When readiness is computed,
Then all reference calls must be complete, parse-valid, nontruncated, and exact,
And at least seven combined interpretations must match authority.

**Spec traces:** REQ-VERIFY-7209

#### SCENARIO-VERIFY-7209-ARTIFACT: Cold Replay Rejects Tampering

Given a complete, blocked, or changed Exp7209 artifact,
When cold validation checks fields, rows, budgets, sources, raw hashes, and readiness,
Then only a consistent terminal artifact passes,
And readiness cannot turn a canary into held-out verifier value.

**Spec traces:** REQ-VERIFY-7209

## Implementation Status (REQ-VERIFY-7209)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7209 and SCENARIO-VERIFY-7209-* | Planned in the Exp7209 module and executable wrapper. | Planned in focused tests before implementation. |

### REQ-VERIFY-7222: Independently Requalified Span Fixture SHALL Use The Registered CPU Substrate

Exp7222 SHALL execute a new deterministic qualification of the V635 source-span
fixture. It SHALL reuse the shipped span compiler, independent controlled-language
authority interpreter, and typed relation executor. It SHALL not edit Exp7208 or
copy its positive verdict as evidence. It SHALL retain Exp7208 as historical
quarantined evidence and record the current duration/substrate classifier result
for both the old artifact and the new candidate checkpoint.

The fixture SHALL preserve construction seed `7208001`, surface seed `7208002`,
80 disjoint bases, canary/development/test base counts of 8/8/64, four relation
families per split, and four variants per base. The variants SHALL retain direct
support, contradicted direction or polarity, inverse relations, negation,
conjunction with joint support, removed-support unknowns, and alpha-renamed
lexical surfaces. The public view SHALL contain no authority labels, split names,
family names, variants, seeds, base identifiers, or gold source tuples.

Exp7222 SHALL reconstruct the public JSONL, evaluator-only authority JSONL,
split manifest, request-bound grammar catalog, and row receipts below
`results/raw/experiment_7222/`. The manifest SHALL record exact paths, content
hashes, counts, base splits, and grammar hashes. Public bytes SHALL not change
when only authority labels change. Inputs SHALL be authenticated from source
code, source documents, exact Exp7196/Exp7197 producer fields, the Exp7196 raw
manifest, and current exclusion and quarantine checks. A structured quarantine,
manifest exclusion, changed checksum, missing source, or failed producer gate
SHALL stop before qualifying work and produce a schema-complete blocked artifact.

For every unit, the candidate path SHALL derive source and claim tuples from
public bytes, compile exact sentence and entity spans, execute relation closure,
and compare with the independent authority label. It SHALL retain all 320 rows
and all 256 held-out rows with unit, arm, seed, metric, error, and abstention.
The frozen lexical control SHALL retain the same full denominator. Offset,
cross-document, wrong-sentence, non-entity, unsupported-predicate, polarity,
excess-count, relation-direction, negation, removed-premise, and grammar
mutations SHALL fail at the applicable compiler or semantic boundary.

Exp7222 invokes no LLM. Both `inference_substrate` and
`inference_substrate_class` SHALL equal the exact registered literal
`cpu_exact_solver_or_simulator`, `MODEL_SPECS` SHALL equal `[]`, and
`model_invoked` SHALL be false. The venue SHALL be `host`; the actual hostname
SHALL be separate in `execution_host`. Duration SHALL measure monotonic work
without sleeping or padding. A blocked preflight SHALL use
`inference_substrate_class=blocked_no_run`.

Before finalization, Exp7222 SHALL atomically write a complete candidate under
`results/checkpoints/`, run the unchanged duration/substrate classifier and full
artifact verifier against that checkpoint, and retain their exact reports. It
SHALL set `span_fixture_ready_score=1` only when regenerated bytes, disjoint
splits, all compiler and semantic rows, every mutation, grammar serialization,
sidecar hashes, source authentication, and the unchanged verifier checks pass.
Because the exact fixture authority is reused for correctness,
`verifier_is_oracle` SHALL be true and a ready fixture SHALL use
`verdict_class=circular_positive`, never `positive`.

The terminal artifact SHALL be
`results/experiment_7222_v636_span_fixture.json`. It SHALL contain all
task-required fields, including `field_principles`, `status`, `run_date`,
`preconditions_checked`, both substrate fields, venue and host, measured
`duration_s`, `source_artifact_hashes`, full `rows`, `sample_size_budget`,
`random_seed`, `reproducibility_checksum`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`span_fixture_ready_score`, `public_view_path`, `authority_sidecar_path`,
`fixture_manifest_path`, `grammar_contract`, `mutation_rows`,
`substrate_classifier_receipt`, `MODEL_SPECS`, and `model_invoked`. Only exact
two-key `principle` and `value` records may be unwrapped.

#### SCENARIO-VERIFY-7222-PREFLIGHT: Independent Inputs Fail Closed

Given source files, producer artifacts, raw receipts, output destinations, and
quarantine evidence,
When one exact prerequisite is absent, changed, failed, excluded, or quarantined,
Then Exp7222 stops before qualifying work with `blocked_no_run`,
And the gate summary names the check, upstream, field, expected, and observed value.

**Spec traces:** REQ-VERIFY-7222

#### SCENARIO-VERIFY-7222-RECONSTRUCTION: Frozen Design Produces Fresh Bytes

Given authenticated sources and the frozen V635 seeds,
When Exp7222 reconstructs the fixture under its own raw directory,
Then 320 public and 320 private rows cover 80 disjoint bases and all required variants,
And exact paths, hashes, counts, splits, row receipts, and grammars are sealed.

**Spec traces:** REQ-VERIFY-7222

#### SCENARIO-VERIFY-7222-EXECUTION: Public Tuples Reach Independent Evaluation

Given source and claim text without gold tuples,
When the shipped compiler and typed executor run before private labels are read,
Then exact source sentences, entities, relations, closure, and all 320 decisions agree,
And inverse, negative, conjunctive, removed-support, and renamed cases remain visible.

**Spec traces:** REQ-VERIFY-7222

#### SCENARIO-VERIFY-7222-MUTATIONS: Reference And Semantic Attacks Fail

Given valid public references and relation premises,
When offsets, document binding, sentence binding, entity type, predicate, polarity,
count, direction, negation, or required premises are changed,
Then the relevant compiler or executor check fails and records the observed reason.

**Spec traces:** REQ-VERIFY-7222

#### SCENARIO-VERIFY-7222-SUBSTRATE: Registered CPU Work Keeps Its Measured Duration

Given a newly executed deterministic CPU qualification with no model invocation,
When the unchanged classifier evaluates the candidate checkpoint,
Then both substrate fields use `cpu_exact_solver_or_simulator`, no model floor applies,
And the measured duration is retained without padding while Exp7208 stays flagged.

**Spec traces:** REQ-VERIFY-7222

#### SCENARIO-VERIFY-7222-ARTIFACT: Cold Replay Controls Readiness

Given a complete, blocked, or changed Exp7222 artifact and its sealed raw files,
When cold validation replays fields, hashes, counts, splits, grammars, semantics,
mutations, classifier receipts, rows, and checksum,
Then only a consistent clean candidate has readiness one and a circular-positive verdict,
And a forged gate, source byte, aggregate, or verifier report fails validation.

**Spec traces:** REQ-VERIFY-7222

## Implementation Status (REQ-VERIFY-7222)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7222 and SCENARIO-VERIFY-7222-* | Planned in a small requalification module and executable wrapper that reuse the V635 production path. | Planned in focused tests before implementation. |

### REQ-VERIFY-7223: Authenticated Qwen Span Canary SHALL Freeze Calibration Before Inference

Exp7223 SHALL authenticate the Exp7222 terminal artifact and its public,
authority, and manifest bytes. It SHALL reject structured quarantine, manifest
exclusion, a changed byte hash, a failed producer checksum, or a failed
`span_fixture_ready_score`. The historical Exp7209 block SHALL remain evidence
that no model loaded. It SHALL not serve as evidence that span extraction failed.

Before model inference, Exp7223 SHALL select eight units only from the Exp7222
canary split. The frozen selection SHALL contain two supported cases, two
contradicted cases, two removed-support unknown cases, and two joint-support
cases. It SHALL cover all four relation families. It SHALL record that neither
held-out rows nor model outcomes controlled selection. Public model requests
SHALL contain no authority fields.

The canary SHALL use only `unsloth/Qwen3.8-27B-GGUF` at `Q4_K_M`. It SHALL
resolve the exact cached GGUF revision and content-addressed hash. It SHALL use
the embedded tokenizer and chat template. It SHALL set `CARNOT_FORCE_LIVE=1`
and acquire one task-owned GPU lease after a read-only conflict check. It SHALL
use one native llama.cpp server and SHALL not use `DualGPURunner`.

Each selected unit SHALL use separate source and claim requests. The candidate
claim request SHALL not receive the source database. The fixed schedule SHALL
contain exactly 16 calls. Each call SHALL permit at most 512 generated tokens,
use a 90-second request timeout, and remain inside one 1500-second inference
deadline. The capture SHALL preserve exact request and response bytes, prompt
and completion token counts, stop reasons, parse results, and task-owned CUDA
evidence. Exp7223 SHALL print a flushed phase boundary and shall emit truthful
heartbeats during long native work.

Exp7223 SHALL apply a syntax-only check and a reference/type check offline to
the same raw responses. It SHALL then compare source tuples, claim tuples, and
the typed executor decision with the independent calibration authority. It
SHALL retain one row per unit and offline arm. This pilot SHALL not select or
score held-out outcomes.

The frozen readiness gate SHALL require at least seven of eight units with two
complete parse-valid calls. It SHALL also require at least six of eight units
with correct source and claim semantics. It SHALL require zero authority
leakage and zero unresolved model-identity errors. `span_canary_ready_score`
SHALL equal the exact Boolean gate. `span_canary_complete_score` SHALL equal one
only when all 16 scheduled observations exist. A failed semantic canary SHALL
be a completed null. A missing external prerequisite SHALL be blocked. Exp7223
SHALL not tune or rerun the scheduled canary to improve its result.

Actual CUDA generation SHALL use `inference_substrate=live_llm_inference`,
`inference_mode=live_gpu`, and
`inference_substrate_class=model_bounded_generation`. A run that stops after
the canary SHALL keep the bounded class. A preflight block SHALL use
`blocked_no_run`. Duration SHALL use monotonic time without padding. The venue
SHALL be `host`, and `execution_host` SHALL contain the actual hostname.

The terminal artifact SHALL be
`results/experiment_7223_v636_span_canary.json`. It SHALL contain the required
principle map and actual values for `status`, `run_date`,
`preconditions_checked`, both substrate fields, `execution_venue`,
`execution_host`, `duration_s`, `source_artifact_hashes`, `rows`,
`sample_size_budget`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, `honest_verdict`,
`MODEL_SPECS`, `model_invoked`, `span_canary_ready_score`,
`span_canary_complete_score`, `canary_rows`, `frozen_decoding_contract`,
`token_budget_receipt`, `model_identity_receipt`, `gpu_receipts`,
`phase_spans`, and `runner_receipt`. Only an exact two-key `principle` and
`value` record may be unwrapped.

#### SCENARIO-VERIFY-7223-PREFLIGHT: Exp7222 Evidence Fails Closed

Given the Exp7222 terminal artifact, sealed sidecars, exclusion manifest, model
cache, native runner, imports, output paths, and live GPU inventory,
When any required check fails,
Then Exp7223 writes a schema-complete `blocked_no_run` terminal artifact,
And `gate_check_summary` names the check, upstream, field, expected, and observed value.

**Spec traces:** REQ-VERIFY-7223

#### SCENARIO-VERIFY-7223-SELECTION: Calibration Schedule Is Frozen And Blind

Given the authenticated Exp7222 calibration split,
When Exp7223 freezes the canary schedule before model inference,
Then eight units cover the four required semantic cases and four relation families,
And 16 separate public-only source and claim calls contain no authority field.

**Spec traces:** REQ-VERIFY-7223

#### SCENARIO-VERIFY-7223-CAPTURE: Native Calls Preserve Bounded Evidence

Given one owned CUDA server and the frozen schedule,
When each bounded request returns or fails,
Then all attempted rows retain raw bytes, tokens, stop state, timeout, and owner evidence,
And the source and claim inputs never share their private document text.

**Spec traces:** REQ-VERIFY-7223

#### SCENARIO-VERIFY-7223-SCORING: One Capture Feeds Both Offline Arms

Given the 16 raw responses and independent calibration authority,
When syntax and reference/type validation run offline,
Then both arms cite the same two raw call hashes for each unit,
And semantic scoring compares source tuples, claim tuples, and executor decisions.

**Spec traces:** REQ-VERIFY-7223

#### SCENARIO-VERIFY-7223-READINESS: Exact Fixed Denominators Control The Gate

Given eight scheduled unit pairs,
When readiness is computed,
Then at least seven parse-complete units and six semantically correct units are required,
And authority leakage, identity errors, or a missing call force readiness to zero.

**Spec traces:** REQ-VERIFY-7223

#### SCENARIO-VERIFY-7223-ARTIFACT: Cold Replay Rejects Tampering

Given a complete, null, blocked, or changed Exp7223 artifact and raw manifest,
When cold validation replays fields, hashes, rows, budgets, gates, and provenance,
Then only internally consistent terminal evidence passes,
And readiness never becomes held-out verifier value.

**Spec traces:** REQ-VERIFY-7223

## Implementation Status (REQ-VERIFY-7223)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7223 and SCENARIO-VERIFY-7223-* | `python/carnot/experiment_7223_v636_span_canary.py` and its thin executable reuse the V635 capture runner and V636 fixture authority. | `tests/python/test_experiment_7223_v636_span_canary.py` covers all new statements and the Exp7209 adapter regression tests stay green. |

### REQ-VERIFY-7326: Acquired Integer Constraints SHALL Have A Parity-Tested Rust Kernel

Exp7326 SHALL authenticate the terminal Exp7325 audit before adding a production
kernel. Exp7325 SHALL have `addition_promotion_score=1`; its verdict class SHALL
not be `blocked`, `disqualified`, or `partial`; and it SHALL not be quarantined.
The exact Exp7324 request rows named by Exp7325 SHALL remain available at their
declared hash. Every acquired atom SHALL retain its original identifier,
executor version, kind, payload, witness, and receipts, and its identifier SHALL
be recomputed before use. A missing value or failed external check SHALL produce
a row-free terminal blocked result whose gate summary retains the upstream,
check, field, expected value, and exact observed value.

The `carnot-constraints` crate SHALL expose a serialized integer schedule
evaluator for only two term kinds. A pairwise separation term with minimum `m`
SHALL have energy `max(0, m - abs(left_slot - right_slot))^2`. A sliding-window
capacity term with width `w` and maximum `c` SHALL have energy equal to the sum,
over every integer window start in the inclusive slot domain, of
`max(0, occupancy - c)^2`; windows SHALL use the half-open interval
`[start, start + w)`. The acquired global same-slot capacity SHALL translate to
the width-one form without changing its decision. Each term energy and their
checked sum SHALL be a nonnegative integer. Boundary equality SHALL have zero
energy. A fully supplied, valid request SHALL be feasible exactly when every
term has zero energy.

The evaluator SHALL reject empty or duplicate schedules, missing constrained
activities, invalid slot domains or assignments, nonpositive windows, negative
bounds, arithmetic overflow, malformed term kinds, and inconsistent executor
versions. Invalid input SHALL not be represented as a zero-energy feasible
decision. Wider synthetic windows MAY overlap and each window excess SHALL be
charged exactly once. A zero acquired energy SHALL remain an acquired-language
decision, not a complete oracle certificate, because learned constraints can be
incomplete.

Python and Rust SHALL produce bit-exact decisions, errors, total energies, and
ordered per-term energies for all 2,304 captured arm-request rows plus 1,000
seeded valid and malformed finite fixtures. Returned schedule hashes SHALL be
recovered only from the frozen finite public request domain and SHALL match the
captured hash; abstentions SHALL remain explicit invalid empty schedules. The
round trip SHALL cross a newline-delimited serialized subprocess boundary. Its
Rust fixture SHALL live in a task-owned example or test file whose name contains
7326.

After warmup, Exp7326 SHALL measure batch sizes 1, 32, and 256 in at least 30
paired blocks. Python and Rust SHALL receive byte-equivalent requests through
the same persistent subprocess service boundary. Process startup and JSON
serialization SHALL be reported separately from steady request/response time.
`constraint_kernel_complete_score` SHALL equal one only when parity is exact
and every required cost row is complete. `kernel_speedup_score` SHALL equal one
only when every fixed size has a paired speedup lower CI95 of at least 10.
Measured speedup and uncertainty SHALL remain visible when that target fails.
A faster kernel SHALL not establish whole-learning speedup.

The artifact SHALL project exact sparse coefficient count, integer bit width,
serialized memory bytes, and maximum coupling degree from the measured fixture.
This SHALL be labeled a software projection. Exp7326 SHALL issue no hardware
command, claim no FPGA or TSU timing, change no deployment or storage
acknowledgment behavior, and make no production-default or publication change.
It SHALL run current scoped Python checks, Rust tests, format, Clippy, the
Python-to-Rust round trip, and both terminal artifact validators before its
atomic terminal write. Any affected validation failure SHALL set readiness and
promotion scores to zero.

Exp7326 SHALL use date `20260915`, `MODEL_SPECS=[]`, `model_invoked=false`, zero
model loads and generations, `inference_substrate=cpu_exact_solver_or_simulator`,
`inference_substrate_class=cpu_exact_solver_or_simulator`, and
`execution_venue=host`. Duration and disjoint phase spans SHALL measure actual
monotonic work without sleeping or padding.

#### SCENARIO-VERIFY-7326-PREFLIGHT: Promoted Atoms Fail Closed

Given Exp7325 and its declared raw acquired-atom evidence,
When promotion, class, quarantine, file hash, row count, and atom hashes are checked,
Then only exact complete eligible evidence can start implementation measurement,
And the first external failure remains a terminal blocked result with its exact value.

**Spec traces:** REQ-VERIFY-7326

#### SCENARIO-VERIFY-7326-ENERGY: Integer Terms Have Exact Zero Boundaries

Given a fully supplied integer schedule and same-version terms,
When pair deficits and sliding-window excesses are evaluated,
Then equality at each minimum or maximum has zero energy and violations have squared energy,
And overlapping windows retain one ordered nonnegative energy for each declared term.

**Spec traces:** REQ-VERIFY-7326

#### SCENARIO-VERIFY-7326-INVALID: Malformed Schedules Never Look Feasible

Given empty schedules, invalid slots, inconsistent versions, missing activities, or overflow,
When either implementation evaluates the serialized request,
Then both return the same fail-closed error and infeasible decision,
And no invalid request is treated as a zero-energy oracle certificate.

**Spec traces:** REQ-VERIFY-7326

#### SCENARIO-VERIFY-7326-PARITY: Captured And Seeded Rows Agree Bit For Bit

Given all 2,304 authenticated capture rows and 1,000 frozen finite fixtures,
When Python and the Rust 7326 fixture evaluate the same serialized batches,
Then every validity decision, error, feasibility decision, total, and per-term energy agrees,
And acquired incompleteness remains explicit even when every acquired energy is zero.

**Spec traces:** REQ-VERIFY-7326

#### SCENARIO-VERIFY-7326-COST: Equivalent Service Boundaries Own Speedup

Given persistent Python and Rust services after warmup,
When sizes 1, 32, and 256 run for at least 30 paired blocks,
Then each block retains both elapsed times, order, ratio, failures, and censoring,
And startup, serialization, steady-kernel speedup, and whole-learning claims stay separate.

**Spec traces:** REQ-VERIFY-7326

#### SCENARIO-VERIFY-7326-HARDWARE: Placement Is A Software Projection

Given the exact measured constraints and schedules,
When sparse placement characteristics are reduced,
Then coefficient count, bit width, bytes, and degree derive from those records,
And no host measurement is labeled FPGA, TSU, board, or deployment performance.

**Spec traces:** REQ-VERIFY-7326

#### SCENARIO-VERIFY-7326-TERMINAL: Correctness And Performance Stay Separate

Given complete parity, cost rows, scoped checks, Rust checks, and terminal validators,
When terminal scores are derived,
Then kernel completeness can equal one even if the 10x lower-bound gate fails,
And blocked or disqualified output sets both scores to zero before atomic publication.

**Spec traces:** REQ-VERIFY-7326

## Implementation Status (REQ-VERIFY-7326)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7326 and SCENARIO-VERIFY-7326-* | `python/carnot/experiment_7326_v643_constraint_kernel.py` reuses the scoped validation runner, and `crates/carnot-constraints/src/schedule.rs` exposes the integer kernel through a task-owned example; no binding or storage change. | `tests/python/test_experiment_7326_v643_constraint_kernel.py` covers every new Python statement, while `crates/carnot-constraints/tests/experiment_7326_constraint_kernel.rs` and the serialized round trip cover Rust boundaries and parity. |

### REQ-VERIFY-7339: Native Schedule Evaluation SHALL Preserve Exact Semantics

Exp7339 SHALL authenticate the exact Exp7325 learning audit and Exp7326 kernel
artifact before measurement. Both inputs SHALL be terminal and available at
their recorded hashes. Exp7325 SHALL remain promoted and eligible. Exp7326
SHALL retain 3,304 exact parity rows and zero mismatches. A missing,
quarantined, blocked, disqualified, partial, or changed input SHALL produce a
row-free terminal blocked result. The result SHALL name the failed field and
its expected and observed values.

The native path SHALL reuse `carnot-constraints` schedule evaluation. It SHALL
not add a second energy implementation. The path SHALL preserve validity,
feasibility, errors, checked overflow, total energy, and ordered term energies.
It SHALL preserve the executor-version check. It SHALL never convert invalid
input into a zero-energy certificate.

The parity panel SHALL replay the 2,304 captured and 1,000 seeded Exp7326
fixtures through the imported extension. It SHALL also include fixed mutation
and overflow cases. Constraint compilation SHALL copy its inputs. Later caller
mutation SHALL not change compiled behavior. Returned Python objects SHALL not
share mutable state with later results. The panel SHALL require zero
differences.

The fixed cost protocol SHALL use batch sizes 1, 32, and 256. It SHALL run 30
randomized paired blocks for each size. Each timed native call SHALL include
Python-object conversion, request and result allocation, batch marshalling,
evaluation, and result conversion. JSON and subprocess work SHALL stay outside
the native timed boundary. Import and immutable-constraint compilation SHALL
be charged separately. Current Python in-process, Rust in-process, and the
historical-service mechanism SHALL receive equivalent current requests. The
study SHALL make no speed or learning-value claim.

The terminal artifact SHALL record the interpreter, ABI, loaded extension,
binary hash, Rust source identity, build command, and isolated build target.
`native_binding_ready_score` SHALL equal one only after the actual imported
extension completes exact parity, mutation and overflow checks, E2E-003, the
fixed cost protocol, affected validation, and terminal validators. A build by
itself SHALL never set readiness. Exp7339 SHALL not alter production defaults,
publication surfaces, deployment, historical artifacts, or board state.

Exp7339 SHALL use date `20260916`. It SHALL use `MODEL_SPECS=[]`,
`model_invoked=false`, and zero load and generation counts. It SHALL declare
`cpu_exact_solver_or_simulator` for both substrate fields and `host` for its
execution venue. All durations SHALL use measured monotonic time.

#### SCENARIO-VERIFY-7339-PREFLIGHT: Historical Inputs Fail Closed

Given the retained Exp7325 and Exp7326 artifacts and their named raw evidence,
When identity, status, class, quarantine, scores, row counts, and hashes are checked,
Then only exact eligible inputs can start native measurement,
And the first failed check produces a row-free blocked result.

**Spec traces:** REQ-VERIFY-7339

#### SCENARIO-VERIFY-7339-PARITY: The Imported Extension Replays Every Fixture

Given all 3,304 authenticated fixtures and the fixed adverse cases,
When the compiled native evaluator processes in-process Python objects,
Then every decision, error, total, and ordered term energy matches Python,
And the mismatch count is zero.

**Spec traces:** REQ-VERIFY-7339

#### SCENARIO-VERIFY-7339-PROTOCOL: Full Boundaries Use Fixed Paired Blocks

Given the protocol sealed before timing,
When sizes 1, 32, and 256 each run in 30 randomized paired blocks,
Then each arm records complete costs, failures, abstentions, and censoring,
And setup costs remain separate from timed per-call boundary costs.

**Spec traces:** REQ-VERIFY-7339

#### SCENARIO-VERIFY-7339-TERMINAL: Readiness Requires Runtime Evidence

Given a built and imported interpreter-specific extension,
When parity, adverse cases, E2E-003, cost rows, scoped checks, Rust checks, and terminal validators finish,
Then readiness equals one only if every required check passes,
And any blocked or disqualified result keeps readiness at zero.

**Spec traces:** REQ-VERIFY-7339

## Implementation Status (REQ-VERIFY-7339)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7339 and SCENARIO-VERIFY-7339-* | Implemented in the immutable schedule PyO3 binding and Exp7339 runner. | `tests/python/test_experiment_7339_v644_native_binding.py`, the binding's focused Rust test, the 3,310-row runtime replay, and E2E-003 verify the behavior. |

### REQ-VERIFY-7340: Native Cost Evidence SHALL Cover The Complete Acquired-Constraint Boundary

Exp7340 SHALL authenticate the exact top-level identity, terminal status,
eligibility, quarantine state, readiness score, loaded-extension identity, and
frozen protocol declared by Exp7339 before consuming any score. Missing,
blocked, disqualified, partial, quarantined, changed, or incomplete upstream
evidence SHALL produce a canonical row-free `blocked_*` terminal artifact that
names the upstream, failed check, field, expected value, and observed value.

The measurement SHALL replay current Python in-process, Rust in-process, and
persistent-service arms over equivalent retained acquired-constraint records.
One pinned host process per arm SHALL run three warmups and exactly 30 paired
blocks at each frozen batch size 1, 32, and 256 in seeded randomized block
order. Development-only repetition selection SHALL make clock resolution
adequate without extending the fixed evaluation sample after outcomes are
known. Every row SHALL retain its size, seed, block, arm, repetition count,
complete conversion/evaluation/result boundary time, parity outcome,
contending-work declaration, and censoring state.

Cold import, constraint compilation, input marshalling, evaluation, result
conversion, and persistent-service transport SHALL be measured as explicit
non-overlapping spans. The artifact SHALL report p50 and p95 latency by size
and arm and a paired bootstrap CI95 throughput ratio for native Rust versus
Python at every size. `native_cost_complete_score` SHALL equal one only when
all 90 paired blocks have complete three-arm costs and zero parity mismatches.
`native_ten_x_score` SHALL equal one only when the native-over-Python
throughput-ratio CI95 lower bound is at least 10 at all three sizes. A lower
gain SHALL remain a complete null and SHALL not move the unchanged gate.

Break-even request count SHALL include measured import, compilation, and
binding setup costs without overlapping any per-request span. V643 stage costs
MAY be used only when their identity and non-overlap are explicit; otherwise
the whole-learning upper-bound analysis SHALL be unavailable. Neither an
isolated evaluator speedup nor a native null SHALL imply a broad learning,
native, hardware, FPGA, or board claim. A repeated performance null SHALL
retire only this exact complete-boundary experiment.

The current work SHALL run row/hash mutation checks, stage-overlap checks,
E2E-003, independent raw-row reduction, current affected validation, and both
terminal validators before an atomic terminal write. Exp7340 SHALL use date
`20260916`, `MODEL_SPECS=[]`, `model_invoked=false`, zero invocation counts,
`inference_substrate=cpu_exact_solver_or_simulator`,
`inference_substrate_class=cpu_exact_solver_or_simulator`, and
`execution_venue=host`. The execution authority defines correctness, so
`verifier_is_oracle` SHALL remain true.

Exp7340 SHALL create the private pytest base-temp parent before the scoped
validation subprocesses start. It SHALL retain any failed validation attempt
in the task-owned raw evidence directory before a corrected attempt runs.

#### SCENARIO-VERIFY-7340-PREFLIGHT: The Same-Milestone Producer Fails Closed

Given the exact Exp7339 artifact and its declared top-level protocol and binding fields,
When availability, hash, terminal state, class, quarantine, readiness, and extension identity are checked,
Then only complete eligible runtime evidence can start measurement,
And the first failed check produces a canonical row-free blocked result.

**Spec traces:** REQ-VERIFY-7340

#### SCENARIO-VERIFY-7340-COST: Paired Blocks Measure The Complete Boundary

Given equivalent acquired-constraint requests and one pinned process per arm,
When sizes 1, 32, and 256 run exactly 30 seeded randomized paired blocks,
Then conversion, evaluation, results, and service transport are retained without overlap,
And all arm outputs match at every completed block.

**Spec traces:** REQ-VERIFY-7340

#### SCENARIO-VERIFY-7340-GATE: Ten-X Requires Every Lower Confidence Bound

Given 30 complete paired throughput ratios at each fixed size,
When the frozen bootstrap reducer computes CI95 intervals,
Then the ten-x score is one only if every lower bound is at least 10,
And a smaller measured gain is published as a complete performance null.

**Spec traces:** REQ-VERIFY-7340

#### SCENARIO-VERIFY-7340-AMORTIZATION: Setup Is Charged Exactly Once

Given measured import and compilation costs plus complete per-request costs,
When native-versus-Python break-even is reduced,
Then setup and steady costs do not overlap and the request count is explicit or unavailable,
And no isolated evaluator result is presented as whole-learning acceleration.

**Spec traces:** REQ-VERIFY-7340

#### SCENARIO-VERIFY-7340-TERMINAL: Null Evidence Remains Complete

Given complete rows, zero mismatches, adverse checks, E2E-003, validation, and terminal validators,
When the unchanged ten-x gate fails,
Then native cost completion remains one while native ten-x is zero and value/promotion are zero,
And only this exact boundary is retired without a broad native or hardware conclusion.

**Spec traces:** REQ-VERIFY-7340

#### SCENARIO-VERIFY-7340-VALIDATION-TEMP: Scoped Validation Has A Private Parent

Given the shipped scoped validation runner and a private base-temp path,
When Exp7340 prepares affected validation,
Then the parent exists before pytest starts,
And a missing parent cannot disqualify otherwise complete measured evidence.

**Spec traces:** REQ-VERIFY-7340

## Implementation Status (REQ-VERIFY-7340)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7340 and SCENARIO-VERIFY-7340-* | Implemented in `python/carnot/experiment_7340_v644_native_cost.py` using the shipped Exp7339 binding and Exp7326 service mechanisms. | `tests/python/test_experiment_7340_v644_native_cost.py` plus the measured E2E-003 and terminal validation receipts. |

### REQ-VERIFY-7416: Anchored Claim Extraction SHALL Preserve Raw Evidence And Span Semantics

Exp7416 SHALL authenticate the exact Exp7410 corpus and Exp7412 source-feature
artifact before model work. The Exp7412 readiness score SHALL equal one. Its
verdict class SHALL be eligible, and its adversarial flag SHALL be false. A
missing, changed, excluded, or ineligible prerequisite SHALL produce a blocked
artifact with zero current invocation counts.

The experiment SHALL select 24 final-test sentences from the Exp7410 predictor
view by a label-blind hash with seed `6501601`. It SHALL also use all 24 sealed
Exp7412 challenge sentences. Selection SHALL not read labels, expected verdicts,
source relations, or annotation scores. Each case SHALL receive the same
question, source sentence, and answer in two fixed arms. The free arm SHALL ask
for decontextualized relation triples. The anchored arm SHALL also require exact
half-open answer offsets and explicit qualifiers. Arm order SHALL alternate by
case hash.

The measured schedule SHALL contain exactly 96 calls to
`unsloth/Qwen3.8-27B-GGUF`. Each call SHALL permit at most 384 new tokens and
use the same deterministic sampling settings. The current runtime SHALL use one
task-owned RTX 3090, the cached GGUF tokenizer and chat template, and native
llama.cpp CUDA offload. It SHALL not use CPU fallback. Lease waiting SHALL not
exceed 180 seconds. Capture SHALL stop after 2400 seconds and SHALL retain every
planned call as completed, failed, censored, or unstarted.

Each request and raw response SHALL be written before the Exp7416 parser runs.
No repair, parser retry, grammar retry, or larger token budget SHALL follow a
bad response. Current load and generation counts SHALL come from owned events.
Historical model evidence SHALL stay in hashed sidecars. The artifact SHALL use
`model_bounded_generation` after generation, `model_load_no_generation` after a
real load with no generation, and `no_model_load` when no load was attempted.

The reducer SHALL report JSON validity, argument anchoring, controlled qualifier
retention, real-example teacher-span overlap, extraction coverage, and latency
as separate endpoints. All assigned cases SHALL remain in each denominator.
The eight constructed semantic pairs SHALL remain eight independent units.
Unknown semantic judgments SHALL remain unknown until Exp7417. No endpoint SHALL
claim full-system correctness, repair efficacy, or automatic promotion.

`extraction_capture_complete_score` SHALL equal one only when all 96 planned
dispositions and immutable raw hashes are present, current provenance is
consistent, CUDA offload is confirmed, affected checks pass, and terminal
readers pass. Parsing or semantic quality SHALL not control this completion
score. A complete low-quality capture SHALL be a null finding. Validation
failure SHALL disqualify the result. The terminal JSON SHALL be written
atomically only after fresh-process replay, independent reduction, adversarial
verification, and strict verdict-row consistency checks.

#### SCENARIO-VERIFY-7416-SELECTION: Predictor-Only Hashing Freezes Both Arms

Given authenticated Exp7410 predictor rows and the sealed Exp7412 challenge,
When the fixed selector builds the schedule before outcomes exist,
Then it chooses 24 final-test sentences and all 24 challenge cases,
And each case has both arms with hash-alternated order and no authority leakage.

**Spec traces:** REQ-VERIFY-7416

#### SCENARIO-VERIFY-7416-RAW: Parsing Cannot Change Captured Model Bytes

Given one owned native response for a scheduled call,
When the capture path records the request and response before parsing,
Then their byte hashes bind the later extraction row,
And malformed JSON remains one terminal failed-quality observation without retry.

**Spec traces:** REQ-VERIFY-7416

#### SCENARIO-VERIFY-7416-ENDPOINTS: Span And Meaning Stay Separate

Given complete or failed outputs from both prompt arms,
When the independent reducer evaluates all assigned cases,
Then parse validity, anchoring, qualifiers, teacher overlap, coverage, and latency remain separate,
And missing semantic judgments stay unknown across eight independent pairs.

**Spec traces:** REQ-VERIFY-7416

#### SCENARIO-VERIFY-7416-PROVENANCE: Current CUDA Events Control The Substrate

Given a preflight block, load-only run, or bounded generation run,
When current owned events and CUDA receipts are reduced,
Then counts and substrate class match the actual current work,
And historical calls, CPU fallback, and another owner's process cannot supply evidence.

**Spec traces:** REQ-VERIFY-7416

#### SCENARIO-VERIFY-7416-TERMINAL: Accountable Null Capture Can Complete

Given all 96 dispositions, immutable raw evidence, and required validation,
When extraction quality is weak or outputs are censored within the fixed budget,
Then capture completion can remain one while the verdict is null,
And promotion stays zero without a correctness or repair claim.

**Spec traces:** REQ-VERIFY-7416

## Implementation Status (REQ-VERIFY-7416)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7416 and SCENARIO-VERIFY-7416-* | Planned in a focused experiment module and thin entrypoint that reuse the shipped native runtime and scoped validation helpers. | Planned in focused tests before implementation and one live capability replay. |

## V651 runtime ownership repair — 2026-09-19

**Status:** Specified. This phase separates device capacity from lease ownership and loads no model.

### REQ-VERIFY-7422: Runtime Capacity And Ownership SHALL Be Independent Checks

Exp7422 SHALL reproduce the inherited Exp7400 precondition failure with zero,
one, and two-device fixtures. The shared precondition SHALL check GPU inventory
query success as a Boolean. It SHALL check the available RTX 3090 count as an
integer greater than or equal to one. It SHALL not compare an availability
dictionary by equality. It SHALL not treat available inventory as proof of
lease ownership.

The fixture audit SHALL cover failed inventory, no free device, two free
devices, one busy device, a stale owner PID or start tick, acquisition races,
and bounded release. Unknown or conflicting process state SHALL fail closed.
The audit SHALL use the lease helper imported by Exp7400 through Exp7347. It
SHALL not change a historical result artifact or generalize dictionary
comparison operators.

On the current host, Exp7422 SHALL use the existing lease protocol to acquire
exactly one available RTX 3090. The wait SHALL not exceed 120 seconds. A fresh
child process SHALL read the journal and validate the task PID, process start
tick, device UUID, and lease identity. The owner SHALL make the no-model lease
terminal and release only that lease. It SHALL not load model weights,
generate text, install a runtime, change the conductor, signal another owner,
or retain a reservation for later work.

The artifact SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM invocation counts, `inference_substrate_class=no_model_load`, and
`execution_venue=host`. A cached model file MAY appear only as a hashed path
prerequisite. Archived or scripted model events SHALL remain typed hash-bound
sidecars and SHALL not contribute to current invocation counts. Small energy
head work, if any, SHALL use a separate `small_ebm_training` receipt.

`capacity_rows` SHALL keep query success, available count, and the capacity
predicate separate for every fixture and the live host. `lease_rows` SHALL
keep acquisition, fresh-child readback, terminal transition, and release
identity with timestamps. `runtime_ownership_ready_score` SHALL equal one only
when the repaired predicates, current owned lease lifecycle, affected checks,
and terminal readers all pass. Genuine contention SHALL publish a blocked
artifact with the exact observed capacity and owner state. The score is
readiness evidence and is not a reservation or a scientific benefit claim.

The producer SHALL freeze the Exp7358 affected-file manifest before checks.
It SHALL run that plan through Exp7303 with command-local `COVERAGE_FILE`, a
private existing base-temp parent, exact affected tests, separate changed-module
coverage, scoped Ruff, changed-module mypy, and exact-test spec coverage. It
SHALL not run `full_python_suite`. The entrypoint, fresh-process replay,
independent raw-row reduction, adversarial verifier, and strict row consistency
reader SHALL pass before atomic terminal publication. No numbered E2E applies
because this audit changes no ARC, training, sampling, serialization, PyO3, or
Rust behavior.

#### SCENARIO-VERIFY-7422-CAPACITY: Two Free Devices Satisfy A One-Device Minimum

**Given** successful inventory fixtures with zero, one, and two free RTX 3090 devices
**When** the shared Exp7400 precondition builds its capacity gates
**Then** query success is checked as a Boolean and counts zero, one, and two are checked with `>= 1`
**And** the two-device observation passes without claiming ownership.

**Spec traces:** REQ-VERIFY-7422

#### SCENARIO-VERIFY-7422-FAIL-CLOSED: Unknown Or Busy Inventory Cannot Become Capacity

**Given** a failed inventory query, busy process, stale owner identity, or acquisition race
**When** device availability and lease evidence are reduced
**Then** unavailable or conflicting state remains explicit and cannot satisfy readiness
**And** no unrelated process receives a signal.

**Spec traces:** REQ-VERIFY-7422

#### SCENARIO-VERIFY-7422-OWNERSHIP: One Current Lease Is Read Back And Released

**Given** at least one available current RTX 3090 and the shipped lease protocol
**When** Exp7422 acquires one lease and a fresh child reads its journal
**Then** task PID, start tick, device UUID, and lease ID match the owner receipt
**And** the owner makes the journal terminal and releases only that lease within the bound.

**Spec traces:** REQ-VERIFY-7422

#### SCENARIO-VERIFY-7422-NO-MODEL: Resource Proof Does Not Count As Inference

**Given** a completed capacity and ownership audit
**When** the current-work receipt and terminal artifact are reduced
**Then** every current model load and generation count is zero and the substrate class is `no_model_load`
**And** model metadata is only a path prerequisite while promotion remains zero.

**Spec traces:** REQ-VERIFY-7422

#### SCENARIO-VERIFY-7422-TERMINAL: Exact Evidence Controls Atomic Publication

**Given** raw capacity rows, lease rows, affected validation, and terminal commands
**When** a fresh process independently reduces the candidate
**Then** source, identity, row, score, receipt, or checksum drift fails closed
**And** only the complete validated JSON is published at the declared result path.

**Spec traces:** REQ-VERIFY-7422

## Implementation Status (REQ-VERIFY-7422)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7422 and SCENARIO-VERIFY-7422-* | Implemented in the shared Exp7400 capacity predicates, `python/carnot/experiment_7422_v651_runtime_ownership.py`, and its thin entrypoint. | Verified by spec-linked private fixtures, one current host lease lifecycle, 100% changed-module coverage, scoped affected checks, entrypoint E2E, cold replay, independent reduction, adversarial verification, and strict row consistency recorded in the terminal artifact. |

## V651 anchored extraction capture — 2026-09-19

**Status:** Specified. This phase reruns the unattempted Exp7416 schedule after the narrow runtime ownership repair.

### REQ-VERIFY-7429: Anchored Capture SHALL Preserve The Frozen Exp7416 Question

Exp7429 SHALL authenticate the exact Exp7412 corpus protocol and the exact
Exp7416 96-call schedule before model work. It SHALL use the same 24 official
cases, 24 constructed qualifier challenges, free and anchored arms, arm order,
seed `6501601`, prompts, deterministic decoding, and 384-token call cap. It
SHALL not reopen the retired live proof-memory chain or replace failed cases.

Exp7429 SHALL require the Exp7422 runtime ownership readiness score to equal
one. The Exp7422 verdict class SHALL be `positive`, `circular_positive`, or
`null`. Its adversarial flag SHALL be false. Missing or unchanged unavailable
prerequisites SHALL publish a blocked artifact with `no_model_load` and zero
current invocation counts.

The live path SHALL use `unsloth/Qwen3.8-27B-GGUF`, its resolved revision,
model-file hash, embedded tokenizer and chat template, and the shipped native
runner on one task-owned RTX 3090. Lease waiting SHALL not exceed 120 seconds.
The run SHALL require `CARNOT_FORCE_LIVE=1`. It SHALL not simulate, substitute
a smaller model, retry parser failures, or tune the token budget.

The live path SHALL make up to four fixed development calls before the measured
schedule. At least three usable, non-truncated development replies SHALL open
the measured capture. A completed development gate with fewer than three usable
replies SHALL produce a terminal null result with readiness zero. It SHALL not
be reported as an external block. The measured capture SHALL keep all 96 calls
as completed, failed, censored, or unstarted dispositions within 2400 seconds.

Each current invocation event and response SHALL be flushed to a
content-addressed task shard before the next call. Raw rows SHALL retain the
request, response, reply, token counts, seed, parse state, source identity,
runner PID and start tick, lease identity, and terminal disposition. An owned
timeout SHALL wait for its child, release only its lease, and leave a resumable
nonterminal checkpoint outside the terminal result path.

The independent reducer SHALL recompute JSON validity, claim coverage,
relation direction, negation retention, and quantifier retention for each arm.
Every failed, truncated, censored, and unstarted measured call SHALL remain in
the denominator. Constructed exact fixtures SHALL stay separate from official
machine annotations. `usable_output_count` SHALL count raw-derived usable,
non-truncated measured replies.

`extraction_capture_complete_score` SHALL equal one for a valid, fully
accounted 96-disposition paired capture even when semantic outputs fail.
`extraction_value_score` SHALL equal one only when the frozen paired comparison
meets its declared claim-preservation gates. Completion and scientific value
SHALL remain separate. Promotion SHALL remain zero.

The producer SHALL freeze its affected-file manifest before checks. It SHALL
run the Exp7358 scoped plan through Exp7303. It SHALL retain command-local
coverage data, a private existing base-temp parent, exact affected tests,
100-percent changed-module coverage, scoped Ruff, changed-module mypy, and
exact-test spec coverage. It SHALL run the entrypoint, fresh-process replay,
independent reduction, adversarial verifier, and strict row-consistency reader
before atomic terminal publication. No numbered E2E applies because this work
changes no ARC, training, sampling, serialization, PyO3, or Rust behavior.

#### SCENARIO-VERIFY-7429-FROZEN: The Original Unmeasured Schedule Stays Exact

**Given** authenticated Exp7412 inputs and the blocked zero-attempt Exp7416 artifact
**When** Exp7429 rebuilds the panel before model loading
**Then** all case identities, prompts, arm order, seeds, and token caps match the frozen Exp7416 schedule
**And** the Exp7422 ownership repair is the only new prerequisite.

**Spec traces:** REQ-VERIFY-7429

#### SCENARIO-VERIFY-7429-DEVELOPMENT: Usable Development Output Opens Capture

**Given** one task-owned native model server and four fixed development prompts
**When** at least three replies are usable and not truncated
**Then** the server continues to the 96 measured calls without parser retries
**And** insufficient usable output finishes as a null result rather than an external block.

**Spec traces:** REQ-VERIFY-7429

#### SCENARIO-VERIFY-7429-RAW: Every Measured Disposition Remains Auditable

**Given** a measured call that completes, fails, truncates, is censored, or is never started
**When** the producer checkpoints the capture
**Then** its raw bytes, source identity, seed, parse state, runtime identity, and disposition remain present
**And** all 96 planned calls remain in each declared denominator.

**Spec traces:** REQ-VERIFY-7429

#### SCENARIO-VERIFY-7429-REDUCTION: Completion And Scientific Value Stay Separate

**Given** a fully accounted paired capture with any semantic quality
**When** the independent reducer recomputes both arms
**Then** capture completion depends on accounting and provenance rather than semantic gain
**And** extraction value depends on qualified claim preservation under the frozen comparison.

**Spec traces:** REQ-VERIFY-7429

#### SCENARIO-VERIFY-7429-TERMINAL: Current Work And Validation Control Publication

**Given** current owned events, immutable raw shards, and the frozen validation manifest
**When** fresh-process replay and both strict readers finish
**Then** the final artifact reports the actual substrate class and invocation dispositions
**And** only a validated terminal JSON is atomically published at the declared path.

**Spec traces:** REQ-VERIFY-7429

## Implementation Status (REQ-VERIFY-7429)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7429 and SCENARIO-VERIFY-7429-* | Planned in a focused reusable module and thin entrypoint that reuse the Exp7416 protocol and shipped native runtime. | Planned in spec-linked focused tests, changed-module coverage, scoped checks, live entrypoint E2E, cold replay, independent reduction, adversarial verification, and strict row consistency. |

## V652 lossless claim-span protocol — 2026-09-20

**Status:** Implemented. This phase changes the extraction representation and repairs the producer validation order. It loads no model.

### REQ-VERIFY-7437: Claim Spans SHALL Preserve Literal Propositions And Terminal Evidence

Exp7437 SHALL independently reduce the four archived Exp7429 development
replies. It SHALL retain each native finish reason, actual 64-token ceiling,
embedded GGUF tokenizer identity, and JSON parse result. Archived requests and
responses SHALL remain in a typed hash-bound sidecar. They SHALL not count as
current model work. The reduction SHALL keep output truncation separate from
the missing `adversarial_verify` receipt.

The protocol SHALL define two extraction arms over the same immutable response
paragraph. The compact arm SHALL return `{"claims":[[start,end]]}` with
zero-based, half-open Unicode character offsets. The verbatim arm SHALL return
`{"claims":["text"]}`. Each item SHALL identify a whole factual proposition
with its explicit modifiers. Neither arm SHALL create triples, implied
arguments, or facts that are absent from the response.

The deterministic parser SHALL reconstruct compact claims as exact response
substrings. It SHALL reject invalid bounds, Boolean offsets, duplicate-text
ambiguity in the verbatim arm, extra schema fields, partial JSON objects, and
missing modifiers. It SHALL accept `{"claims":[]}` as an explicit empty
result. It SHALL not repair malformed output or retry parsing. A paragraph
SHALL contain at most 512 Unicode characters. Any clipped paragraph SHALL be
marked and excluded from complete-response coverage claims.

The producer SHALL seal four development paragraphs and 24 evaluation
paragraphs from distinct RAGTruth source groups. Selection SHALL use only
source, response, split, and stable identity fields before evaluator data is
joined. It SHALL not read annotations, model identity, quality, or expected
outcomes during selection. The producer SHALL keep source context and response
bytes immutable. Prompts SHALL contain no evaluator annotation. IDs, byte
hashes, selection rules, prompt templates, and arm order SHALL be sealed before
evaluation.

The future evaluation schedule SHALL contain 48 units: 24 paragraphs by two
arms. Both arms SHALL use the same paragraph, a 256-token ceiling, temperature
zero, and one generation per unit. The protocol SHALL add no grammar mask,
parser retry, or repair call. Twelve constructed qualifier pairs SHALL test
exact qualifier preservation separately from RAGTruth diagnostics. Human
source-support annotations SHALL score only the unchanged RAGTruth response.
They SHALL not certify the semantic quality of a new extraction.

Frozen endpoints SHALL include completed valid output, whole-proposition
coverage, qualifier retention, literal span reconstruction, prompt tokens,
output tokens, and latency. Unstarted calls SHALL remain in `rows` and in each
denominator. A claim-span interface SHALL be described only as an extraction
precursor. It SHALL not be described as an entailment verifier.

This experiment SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current invocation counts, `inference_substrate_class=no_model_load`, and
`execution_venue=host`. Scripted parser controls SHALL remain in a typed
hash-bound sidecar. `promotion_score` SHALL remain zero.

The producer SHALL freeze the affected-file manifest before checks. It SHALL
reuse the Exp7358 command plan through the Exp7303 streaming runner. It SHALL
run exact affected tests, 100-percent changed-module coverage with a private
command-local coverage file, scoped Ruff, changed-module mypy, and exact-test
spec coverage. The producer SHALL validate the exact terminal receipt names
before runtime. It SHALL prove that a private receipt set without
`adversarial_verify` fails, then prove that the complete set passes.

The declared entrypoint, fresh-process cold replay, independent reduction,
`scripts/adversarial_verify.py`, and strict verdict-row consistency SHALL pass
before atomic publication. `span_protocol_ready_score` SHALL equal one only
when parser controls, provenance, the sealed panel, affected validation, and
all terminal receipts pass. A required validation defect SHALL disqualify the
artifact. No numbered E2E applies because this protocol changes no shared
training, sampling, binding, serialization, Rust, or ARC production behavior.

#### SCENARIO-VERIFY-7437-ARCHIVE: Truncation And Missing Receipt Stay Separate

**Given** the four immutable Exp7429 development responses and terminal receipts
**When** Exp7437 independently reduces their raw request and response bytes
**Then** all four rows report the native `length` finish and 64-token ceiling
**And** the missing passing `adversarial_verify` receipt remains a separate producer defect.

**Spec traces:** REQ-VERIFY-7437

#### SCENARIO-VERIFY-7437-PARSER: Literal Offsets Fail Closed

**Given** valid, empty, Unicode, duplicate, malformed, and partial claim outputs
**When** the deterministic parser reads either frozen arm
**Then** every accepted claim reconstructs one exact paragraph substring
**And** ambiguity, bad bounds, missing modifiers, and partial JSON remain explicit failures without repair.

**Spec traces:** REQ-VERIFY-7437

#### SCENARIO-VERIFY-7437-PANEL: Label-Blind Groups Share One Input

**Given** authenticated RAGTruth source and response bytes
**When** the sealed selector chooses development and evaluation paragraphs
**Then** the evaluation panel contains 24 distinct source groups and 48 paired units
**And** both arms receive identical paragraph bytes without evaluator annotations.

**Spec traces:** REQ-VERIFY-7437

#### SCENARIO-VERIFY-7437-SCOPE: Extraction Does Not Become Entailment

**Given** RAGTruth source-support annotations and twelve constructed qualifier pairs
**When** protocol endpoints are frozen
**Then** corpus diagnostics score only the unchanged response
**And** constructed exact checks remain separate from extraction quality and entailment claims.

**Spec traces:** REQ-VERIFY-7437

#### SCENARIO-VERIFY-7437-TERMINAL: Complete Receipts Control Readiness

**Given** a private candidate without an `adversarial_verify` receipt
**When** terminal receipt validation runs before publication
**Then** the incomplete set fails and the exact complete set passes
**And** readiness becomes one only after fresh replay, independent reduction, both strict readers, and atomic publication controls pass.

**Spec traces:** REQ-VERIFY-7437

## Implementation Status (REQ-VERIFY-7437)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-VERIFY-7437 and SCENARIO-VERIFY-7437-* | Implemented in a focused reusable module and thin entrypoint. The module reuses only shipped provenance and scoped-validation helpers. | Verified by 28 spec-linked tests, 100-percent changed-module coverage, scoped static checks, entrypoint E2E, cold replay, independent reduction, adversarial verification, and strict row consistency. |
