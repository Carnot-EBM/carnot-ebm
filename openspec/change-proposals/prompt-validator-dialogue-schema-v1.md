# Prompt-To-Validator Dialogue Schema V1

## Scope

This proposal defines a bounded local protocol for converting prompt-defined
requirements into executable validator trees. It is inspired by ConstrainPrompt
and STATIC-style constrained decoding, but it does not claim paper reproduction,
LLM judge quality, or local speedup evidence. Carnot keeps exact Z3/runtime
verification as the correctness authority.

## Stage 1: Prompt Constraints

Prompt constraints are extracted only into a small allow-listed schema. The V1
schema accepts deterministic fixture patterns for JSON fields, Python AST shape,
and linear integer assignments. Anything outside the allow-list is rejected with
`unsupported_prompt_pattern`; the compiler does not infer semantics by asking an
LLM to judge the requirement.

Each prompt constraint record carries:

- `constraint_id`: stable fixture identifier.
- `natural_language`: original prompt text.
- `allowed_pattern`: the local parser family that recognized the prompt.
- `source`: fixture, prior artifact, or future live-generation provenance.

## Stage 2: Validator Tree

The validator tree is a JSON-serializable structure with a logical root and
leaf nodes. V1 uses an `all` root so every leaf must pass. Leaves name the exact
runtime authority that will execute them, such as `runtime_json_parser`,
`python_ast_parser`, or `z3_solver`.

The tree is declarative. It records what to check; it does not embed generated
Python, `eval`, or model-written verifier code.

## Stage 3: Exact-Check Nodes

Exact-check nodes execute locally and return deterministic feedback:

- JSON/runtime nodes parse candidate JSON and check required field equality or
  numeric bounds.
- Python AST nodes parse candidate source and check function shape or forbidden
  imports without executing the candidate.
- Z3 nodes encode bounded integer assignments and formula constraints, then
  accept only solver-satisfied candidates.

LLM self-assessment is never an exact-check node.

## Stage 4: Feedback Object

Every evaluation returns a feedback object with:

- `accepted`: final boolean verdict.
- `failing_node_ids`: validator leaf IDs that failed.
- `rejection_reasons`: stable reason codes.
- `node_results`: per-leaf status and authority.
- `llm_judge_used`: always false in V1.

This object is the handoff point for downstream repair or self-learning. A
downstream learner may consume it only when the artifact reports
`prompt_validator_protocol_ready=true`.

## Stage 5: Rejection Reasons

V1 defines stable rejection reasons:

- `unsupported_prompt_pattern`
- `json_parse_error`
- `missing_required_field`
- `field_value_mismatch`
- `numeric_range_violation`
- `python_syntax_error`
- `function_signature_mismatch`
- `import_statement_disallowed`
- `z3_unsatisfied`
- `z3_unavailable`

Unsupported prompt patterns fail closed. Candidate failures identify the exact
leaf node that rejected the output.

## Sparse/Static Transition Table Proposal

Validator grammars can be represented as static sparse transitions:

- `row_offsets`: state-to-transition ranges.
- `labels`: token or field labels for outgoing transitions.
- `targets`: next states.
- `accepting_states`: terminal states for complete validator-tree skeletons.

This is a representation proposal only. It borrows the STATIC idea of flattening
constraint automata into sparse tables, but this task makes no speed, latency,
or accelerator claim. Any future performance claim requires a separate measured
benchmark against Carnot's current runtime path.
