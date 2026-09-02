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
