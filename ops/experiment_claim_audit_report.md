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
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_8248_v713_evidence_intervention_methods.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8249_v713_evidence_view_kernel.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
None. The artifact makes no comparative or scientific benefit claim. If the artifact had claimed functional qualification or generalization benefit, that claim would be refuted by observations in its own data showing zero generalization score, all false acceptance gates, circular oracle conditioning, and failed repository test suite health.

## WAS THAT CHECKED
No. No comparative or generalization hypothesis was tested; the run was disqualified and the artifact serves solely as a mechanics qualification and validation receipt record.

## EVIDENCE
`claim_scope`
`Private view and delayed admission execution qualification only; no natural learning or generalization established.`
`honest_verdict`
`complete_disqualified_evidence_view_kernel`
`verdict_class`
`disqualified`
`scientific_benefit_measured`
`false`
`generalized_learning_benefit_score`
`0`
`independent_generalization_score`
`0`
`admission_kernel_ready_score`
`view_kernel_ready_score`
`methodology_note`
`Reuse intact V707 transport, frozen V713 public role selection and durable issue/release ledger. Private group-conditional oracle stream is circular_positive; all adaptive arms share releases and missing-feature global counts. Natural evidence remains unmeasured.`
`validation_receipts`

## RECOMMENDATION
KEEP

## experiment_8250_evidence_view_canary.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Evidence that the experiment actually executed rather than halting at the pre-gate check, such as a non-zero execution duration, completed model runs, or passed upstream dependencies.

## WAS THAT CHECKED
no; execution was blocked at `conductor_pre_gate` before any experimental condition was evaluated.

## EVIDENCE
`schema`
`"blocked_gate_check_v1"`
`status`
`"blocked"`
`honest_verdict`
`"blocked_gate_check_failed"`
`duration_s`
`0.0`
`blocked_at_layer`
`"conductor_pre_gate"`
`passed`
`false`

## RECOMMENDATION
KEEP

## experiment_8257_v713_arc_outcome_frontier.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8258_v713_kv260_evidence_cost_boundary.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8259_v713_polarfire_dispatch_qualification.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8260_v713_gatemate_physical_delta.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write

## experiment_8261_v713_capstone.json

**CANNOT_DETERMINE**

reviewer call failed: OpenAI Codex v0.156.1
--------
workdir: /home/ianblenke/github.com/ianblenke/carnot
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: workspace-write
