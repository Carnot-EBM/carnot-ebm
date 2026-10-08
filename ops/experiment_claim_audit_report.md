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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 3 |
| CANNOT_DETERMINE | 4 |

## experiment_8275_v714_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Any execution data showing completed evaluations with statistically valid gains over registered baselines for hypotheses H1 or H2, or verified transferable evidence in the ARC benchmark.

## WAS THAT CHECKED
No; downstream evaluation and scientific measurement were halted upstream and never executed.

## EVIDENCE
- `"status"`: `"blocked_unmeasured"`
- `"failed_operand"`: `"eligible_registered_audit_primitives"`
- `"independent_science"`: `false`
- `"supported_transferable_evidence"`: `false`
- `"completed_count"`: `0`
- `"arc_evidence_ready_score"`: `0`

## RECOMMENDATION
KEEP

## experiment_8276_v715_current_contract_readiness.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Current contract readiness and execution gates are fully qualified and activated (`complete_circular_positive_current_contract_readiness` with `current_contract_ready_score` of 1) based on task agreement and upstream validation receipts.

## WHAT WOULD REFUTE IT
A failure in contract agreement (such as task numerator falling short of denominator in task contract evaluation rows), an independent external oracle refuting contract adherence, or an evaluation demonstrating that downstream tasks cannot safely proceed under the verified contract.

## WAS THAT CHECKED
No. The verifier is its own oracle (`verifier_is_oracle` is true), comparing the current authority against itself (`full_task_agreement` on `current_authority`). The artifact contains zero independent evaluations (`independent_count` is 0) and includes no external oracle or rival baseline to give refutation a genuine opportunity to occur.

## EVIDENCE
`honest_verdict`
`"complete_circular_positive_current_contract_readiness"`
`activated_readiness`
`true`
`current_contract_ready_score`
`1`
`admission_kernel_ready_score`
`coverage_custody_ready_score`
`independent_count`
`0`
`scientific_benefit_measured`
`false`
`inference_substrate`
`"aggregation_from_upstream_artifacts"`
`"current_contract"`
`"full_task_agreement"`
`"contract_agreement"`
`"current_authority"`
`17`

## RECOMMENDATION
NARROW_CLAIM

## experiment_8277_v715_lease_backend_qualification.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8278_evidence_view_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The artifact is a gate check receipt and makes no substantive or comparative performance claim. Falsification would require an empirical finding regarding Qwen response measurements or internal evidence showing that upstream gate requirements were satisfied despite being reported as failed.

## WAS THAT CHECKED
No. Execution was terminated during pre-gate validation before any experimental trials, candidate generation, or model evaluations were conducted.

## EVIDENCE
`schema` `blocked_gate_check_v1`
`status` `blocked`
`duration_s` `0.0`
`honest_verdict` `blocked_gate_check_failed`
`blocked_at_layer` `conductor_pre_gate`
`gate_check_summary` `gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp8277-lease-backend-qualification.gguf_backend_ready_score (actual=0 == expected=1)`

## RECOMMENDATION
KEEP

## experiment_8286_v715_arc_outcome_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8287_v715_kv260_evidence_cost_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8288_v715_gatemate_physical_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8289_v715_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The artifact asserts no comparative or empirical performance claim. If a comparative claim were asserted, observing negative or zero gain against the mandatory control arms, Brier score degradation exceeding 0.01, or any extra false accepts under the registered protocol would refute it.

## WAS THAT CHECKED
No. Neither hypothesis was evaluated; both are marked as unmeasured and blocked due to missing eligible audit and primitive operands.

## EVIDENCE
`"status"`: `"blocked_unmeasured"`
`"failed_operand"`: `"eligible_independent_audit_and_primitives"`
`"measured_here"`: `false`
`"completed_count"`: `null`
`"statistics"`: `null`
`"independent_science"`: `false`
`"supported_transferable_evidence"`: `false`
`"completed_count"`: `0`
`"arm_support_rows"`: `[]`
`"solve_provenance"`: `"Null without eligible live receipts; no game-level solve is credited."`
`"selection_recommendations"`: `"Unknown prior selection propensity prevents a transferable priority recommendation."`

## RECOMMENDATION
KEEP
