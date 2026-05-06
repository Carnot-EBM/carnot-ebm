# Repair Executor Rejection Ledger v1

Run date: 20260506

Source artifacts:
- `results/experiment_1414_certificate_llm_repair_executor_v1.json`
- `results/experiment_1419_fullscale_pipeline_v3_repair_executor.json`
- `results/experiment_1424_milestone_109_retro.json`

## Summary

Exp 1414 and Exp 1419 both reached the local repair execution path, but every
candidate was rejected. The available artifacts contain per-case rejection rows
and aggregate repair counts, but they do not persist raw model outputs, rendered
prompts, or validator transcripts. This ledger therefore records the highest
confidence reason supported by each row and records the missing evidence instead
of inferring unsupported prompt behavior.

Candidate records analyzed: 120.
Unique case IDs analyzed: 100.
Accepted candidates: 0.

Observed rejection reasons:
- `missing_output_or_nonjson_response`: 112
- `validator_mismatch_no_validator_injected`: 5
- `malformed_json_schema_failure`: 3

The dominant failure was schema-path rejection before semantic validation. Most
rows failed with `invalid JSON repair output: Expecting value: line 1 column 1
(char 0)`. Because raw model text is absent, this is recorded as
`missing_output_or_nonjson_response` rather than split into empty output,
markdown/prose output, or other prompt noncompliance.

## Taxonomy

`missing_output`: The parser saw no usable JSON object. Observed in 112 rows.
Example cases include `160`, `161`, `164_1`, `169`, and `172`.

`schema_failure`: The model produced something JSON-like but invalid. Observed
in 3 rows. Examples: `math_99` failed with a comma delimiter error in both
source artifacts; `math_402` failed with an unquoted-property JSON error in Exp
1419.

`validator_mismatch`: The candidate reached validation but the validation path
reported `no_validator_injected` and preserved `semantic_result=REPAIR_HINT`.
Observed in 5 rows. Examples: `156`, `math_192`, `math_406`, and
`math_v3_1353`.

`semantic_failure`: A schema-valid candidate reaches semantic validation but
fails the semantic/scheduler acceptance contract. Observed count: 0 in the
available artifacts.

`prompt_noncompliance`: A candidate violates the allowed repair-output contract,
for example extra fields, markdown fences, explanation-only text, or prose
instead of the JSON object. Observed count: 0 with high confidence because raw
outputs are missing; some `missing_output` rows may belong here.

`timeout`: Generation or validation exceeded the repair budget. Observed count:
0 in Exp 1414 and Exp 1419 repair-result rows.

## Missing Evidence

The source artifacts do not include:
- raw model output text
- rendered repair prompt per case
- validator transcript or validator identity

Those gaps prevent confident separation between empty generations, prose
answers, markdown-wrapped JSON, transport truncation, and other prompt
noncompliance. Repair v2 must persist these fields for every candidate,
including rejected candidates.

## Repair V2 Acceptance Contract

Repair v2 must apply checks in this order:

1. Parse exactly one JSON object from the model response.
2. Validate the allowed schema before any semantic validation.
3. Require `corrected_certificate` as a non-empty string.
4. Reject unexpected fields unless the schema explicitly permits them.
5. Run the existing semantic validation contract only after schema success.
6. Accept only when `constraint_passed=true`, `semantic_result="SAT"`,
   `repair_required=false`, and `false_acceptance=false`.
7. Record one rejection reason for every candidate, including schema failures,
   semantic failures, validator mismatches, prompt noncompliance, missing
   outputs, and timeouts.
8. Persist raw model output, rendered prompt, validator identity, validator
   transcript, runtime, and local model ID for every candidate.

Downstream full-scale pipeline reruns must be gated on a measured nonzero
validated repair success rate. A 200-case rerun without `repaired_case_success_rate
> 0.0` repeats the Exp 1419 failure mode and is not allowed.
