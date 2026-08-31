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
