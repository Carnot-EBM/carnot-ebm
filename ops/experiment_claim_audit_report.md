# Experiment claim-refutation audit

One question per artifact: what would REFUTE the headline claim, and was that
checked? Fabrication is out of scope (adversarial_verify covers it); this audit
targets claims that are true by construction, circular, in-sample, baseline-weak,
or contradicted by their own rows.

This audit never edits an artifact and never blocks anything. It surfaces; the
operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity
guard rest on evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CLAIM_SUPPORTED | 1 |
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 4 |

## experiment_8307_v717_runtime_change_boundary.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
CUDA runtime qualification remains disqualified from reopening because no authenticated causal changes occurred across the eight tracked runtime environment operands relative to upstream experiment exp8290.

## WHAT WOULD REFUTE IT
The headline claim would be refuted if any row in `rows` or `environment_delta_rows` showed `changed: true` (or `potentially_causal: true`) for driver version, kernel module, device inventory, environment masks, library hashes, device node permissions, permitted UUID, or a valid operator repair receipt, resulting in `runtime_changed_score > 0` and `acceptance_gates.authenticated_change: true`.

## WAS THAT CHECKED
Yes; checked across all 8 rows in `environment_delta_rows` and `rows` (auditing `driver`, `kernel`, `devices`, `masks`, `libraries`, `nodes`, `permitted_uuid`, and `operator_repair`), where all items were evaluated against upstream baseline receipts from `/home/ianblenke/github.com/ianblenke/carnot/results/experiment_8290_v716_runtime_localization.json` and confirmed available, completed, and unchanged.

## EVIDENCE
- `"honest_verdict"`: `"complete_disqualified_cuda_runtime"`
- `"verdict_class"`: `"disqualified"`
- `"runtime_changed_score"`: `0`
- `"completed_count"`: `8`
- `"intended_count"`: `8`
- `"excluded_count"`: `0`
- `"failed_count"`: `0`
- `"acceptance_gates"`: `"authenticated_change"`: `false`, `"cuda_context_ready"`: `false`, `"owned_checks"`: `false`
- `"reopen_contract"`: `"eligible"`: `false`
- `"rows"`:
  - `"field"`: `"driver"`, `"changed"`: `false`, `"observed"`: `"615.71.09"`, `"previous"`: `"615.71.09"`
  - `"field"`: `"kernel"`, `"changed"`: `false`
  - `"field"`: `"devices"`, `"changed"`: `false`
  - `"field"`: `"masks"`, `"changed"`: `false`
  - `"field"`: `"libraries"`, `"changed"`: `false`
  - `"field"`: `"nodes"`, `"changed"`: `false`
  - `"field"`: `"permitted_uuid"`, `"changed"`: `false`, `"observed"`: `"GPU-7971baff-9583-eaa6-2292-393f930a28f9"`
  - `"field"`: `"operator_repair"`, `"changed"`: `false`

## RECOMMENDATION
KEEP

## experiment_10030_fover_headline_selection_denominator.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
A 4-verifier ensemble scoring formula incorporating session memory achieves ~0.913 AUROC on FoVer error detection, providing a ~0.018 learning contribution over base verifiers.

## WHAT WOULD REFUTE IT
The claim is refuted if:
1. The base multi-verifier ensemble fails to outperform the cheapest single-verifier baseline (`tier0r_alone`), demonstrating that multi-verifier combination provides no measurable synergy.
2. The claimed 4-verifier formula does not actually utilize four valid verifiers (e.g., components omitted or scoring at chance).
3. The session memory "learning contribution" fails to achieve statistical significance (95% bootstrap confidence interval spanning zero) once label-contaminated memory rows are excluded on clean held-out data.

## WAS THAT CHECKED
Yes. All three refutations were directly checked within the artifact:
1. In `ensemble_vs_best_single_component`, the base formula was evaluated against `tier0r_alone`.
2. In `headline_text_vs_formula_discrepancy` and `per_component_auroc_full_corpus`, the verifier formula was checked against the north star specification.
3. In `memory_effect` under `H1_never_sampled_and_not_memory_flagged`, the learning contribution was measured with a 95% bootstrap confidence interval on clean held-out rows.

## EVIDENCE
- `north_star_text`: `"4-verifier score (fr11_session_memory, tier0r_curry_howard, tier0s_arithmetic_gap, tier0u_logical_consistency)"`
- `formula`: `"0.9*tier0r + 0.1*tier0u + FR11_MEMORY_BOOST(=1.0)*memory_score"`
- `tier0s_in_formula`: `false`
- `tier0s_scored_but_unused_full_corpus_auroc`: `0.3016179446373822`
- `u`: `0.5098671461946712`
- `tier0r_alone`: `0.9016862925104014`
- `base`: `0.9015981044183999`
- `base_formula_minus_tier0r_alone`: `-8.818809200150657e-05`
- `reading`: `"On the full corpus the 0.9*tier0r + 0.1*tier0u base formula scores within 0.0001 of tier0r alone (slightly below it). The second component adds nothing measurable. All of condition A's lift over tier0r comes from the memory term."`
- `published_learning_contribution`: `0.0184712`
- `low`: `-0.0019785082174461704`
- `high`: `0.026521631023843424`
- `methodology_note`: `"The memory term is built from labels of earlier FoVer rows, so condition A contains label-derived information by design."`
- `honest_verdict`: `"complete: headline_survives_held_out_but_learning_contribution_not_established_on_clean_rows"`
- `gate1_survives_verdict`: `true`

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_8308_sentence_spline_fit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact is an operational gate-check receipt reporting that experiment execution was halted by an unsatisfied upstream prerequisite; it makes no empirical or comparative claim.

## WAS THAT CHECKED
No. No experimental hypotheses or comparative metrics were evaluated because execution was blocked at the pre-gate layer prior to running.

## EVIDENCE
`schema` `blocked_gate_check_v1`
`status` `blocked`
`duration_s` `0.0`
`honest_verdict` `blocked_gate_check_failed`
`blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8313_changed_runtime_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None; the artifact is a gate check receipt recording an execution block rather than an experimental finding or comparative claim.

## WAS THAT CHECKED
No; execution was halted at `conductor_pre_gate` before any canary trial was executed.

## EVIDENCE
`schema`
`blocked_gate_check_v1`
`status`
`blocked`
`honest_verdict`
`blocked_gate_check_failed`
`blocked_reason`
`actual=0 == expected=1`
`blocked_at_layer`
`conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8314_v717_arc_coverage_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8315_v717_kv260_local_cost_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8316_v717_gatemate_obligation.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8317_v717_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
