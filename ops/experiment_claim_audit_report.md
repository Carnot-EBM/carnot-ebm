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
| NO_CLAIM | 6 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6683_ising_reference_scope_receipt.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The artifact makes only an operational readiness claim: the bounded-treewidth exact Ising reference is ready under task-owned evidence.

## WHAT WOULD REFUTE IT
A task-owned fixture producing probability, marginal, correlation, normalization, or decomposition error beyond its tolerance; an unsupported input being accepted; a supported input being rejected; or any required owned gate failing would refute operational readiness.

## WAS THAT CHECKED
Yes. Exact-enumeration comparisons, structural certificates, adversarial cases, rejection cases, aggregate recomputation, and five owned test commands could fail. No comparative added-value claim was checked or made; the verifier explicitly serves as the oracle.

## EVIDENCE
`schema`: `carnot.experiment_6683.ising_reference_scope_receipt.v1`; `honest_verdict`: `complete: bounded-treewidth exact Ising reference is ready under task-owned evidence`; `verdict_class`: `null`; `verifier_is_oracle`: `true`; `ising_reference_ready`: `true`; `ready`: `true`; `owned_test_rows`: `5`; `summary`: `66 passed in 7.07s`; `principle`: `A closed class keeps a ready reference as null infrastructure.`; `gating`: `false`; `readiness_influence`: `false`.

## RECOMMENDATION
KEEP

## experiment_6684_torx_typed_factor_parity.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The installed Torx CPU parity check did not pass every required row and therefore remained blocked.

## WHAT WOULD REFUTE IT
Zero failed checks, every required test exiting successfully, and the aggregate parity-readiness fields being true would refute the claim.

## WAS THAT CHECKED
Yes. The aggregate recomputation and test receipts evaluated readiness and recorded the applicable end-to-end check’s nonzero exit. Oracle circularity limits any added-value claim, but the headline makes only an execution-grounded failure claim.

## EVIDENCE
`honest_verdict` = `blocked_parity_check_failed: installed Torx CPU parity did not pass every row`; `failed_check_count` = `1`; `check_id` = `applicable_e2e`; `exit_code` = `1`; `passed` = `false`; `ready` = `false`; `torx_factor_parity_ready` = `false`; `verifier_is_oracle` = `true`

## RECOMMENDATION
KEEP

## experiment_6685_autocorrelation_schedule_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports only a blocked gate check and makes no comparative performance claim.

## WAS THAT CHECKED
No; the experiment stopped at `conductor_pre_gate` because an upstream readiness gate failed, so no A/B result was produced.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"failed_field": "torx_factor_parity_ready"`, `"failed_observed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6687_v582_branch_synthesis.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6690_exact_planning_fixture_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Nothing: the artifact is a blocked-gate receipt and makes no comparative or performance claim to falsify.

## WAS THAT CHECKED
No. Only the prerequisite gate was evaluated; the experiment stopped at the conductor pre-gate because the upstream artifact was unavailable.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"passed"`: `false`; `"blocked_at_layer"`: `"conductor_pre_gate"`; `"blocked_reason"`: `"upstream artifact not found for task id 'exp6689-exact-planning-fixture'"`

## RECOMMENDATION
KEEP

## experiment_6692_structural_plan_energy.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Nothing: the artifact asserts no experimental outcome or comparative result to falsify.

## WAS THAT CHECKED
No; execution stopped at the upstream gate before the method or any rival could be evaluated.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6694_energy_backtracking_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no substantive result to falsify; this is only a blocked-execution receipt.

## WAS THAT CHECKED
No. The experiment stopped at the prerequisite gate, before any audit method, success criterion, oracle, scored rows, or comparator could be evaluated.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"passed"`: `false`; `"actual"`: `null`; `"blocked_at_layer"`: `"conductor_pre_gate"`; `"blocked_reason"`: `"upstream artifact not found for task id 'exp6693-energy-backtracking-ab'"`

## RECOMMENDATION
KEEP

## experiment_6696_prequential_online_energy_ab.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive claim exists to falsify; the artifact reports only a blocked pre-gate.

## WAS THAT CHECKED
No; the experiment never ran because its sole gate failed at the conductor pre-gate.

## EVIDENCE
`"status"`: `"blocked"`; `"honest_verdict"`: `"blocked_gate_check_failed"`; `"passed"`: `false`; `"blocked_at_layer"`: `"conductor_pre_gate"`

## RECOMMENDATION
KEEP
