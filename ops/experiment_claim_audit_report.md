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
| CLAIM_REFUTED_BY_OWN_DATA | 1 |
| NO_CLAIM | 7 |

## experiment_6620_gpu_lease_phase_receipts.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked-gate receipt and asserts no experimental outcome or comparative value.

## WAS THAT CHECKED
No; the experiment never proceeded past the upstream gate.

## EVIDENCE
`schema` `blocked_gate_check_v1` `status` `blocked` `honest_verdict` `blocked_gate_check_failed` `passed` `false` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6622_qwen36_direct_headroom.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no falsifiable headline claim; the artifact is only a blocked gate-check receipt.

## WAS THAT CHECKED
No. Execution stopped at the pre-gate because the required upstream artifact was unavailable, so no method outcome or success criterion was evaluated.

## EVIDENCE
`"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"passed": false`, `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6624_delayed_two_level_decoding.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports a blocked prerequisite gate and makes no experimental or comparative claim.

## WAS THAT CHECKED
No; the experiment stopped at `conductor_pre_gate`, before any method, success criterion, or comparator was evaluated.

## EVIDENCE
`status` `blocked`  
`honest_verdict` `blocked_gate_check_failed`  
`blocked_reason` `upstream artifact not found for task id 'exp6623-headroom-support-reducer'`  
`passed` `false`  
`blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6627_spectral_integrity_repair.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No substantive performance claim is made; refutation would require a stated claim and method-result rows that could contradict it.

## WAS THAT CHECKED
No. The artifact only records a failed upstream gate; the experiment never reached execution or evaluation.

## EVIDENCE
`status` `blocked`; `honest_verdict` `blocked_gate_check_failed`; `passed` `false`; `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6633_gpu_lease_phase_journal.json

**CLAIM_REFUTED_BY_OWN_DATA**

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
The GPU lease scheduler was not ready because infrastructure checks failed.

## WHAT WOULD REFUTE IT
The infrastructure check identified as failing—focused tests—passing successfully.

## WAS THAT CHECKED
Yes. The sole `gate_check_summary` entry marks focused tests as failed, while `tests_run` records the focused-test command exiting successfully.

## EVIDENCE
The headline reports `blocked_gpu_lease_scheduler_not_ready` and `infrastructure checks failed`. The gate names `focused_tests` with `expected` `true` and `observed` `false`. But `tests_run` records `exit_code` `0` and `summary` `focused tests passed` for the focused-test command.

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_6634_mandated_model_admission.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
An attempted independent-admission test in which at least one mandated GGUF family failed admission.

## WAS THAT CHECKED
No. Execution stopped at the conductor pre-gate because the upstream readiness score was 0.0 rather than 1.0; no GGUF-family admission rows were produced.

## EVIDENCE
`"status": "blocked"`; `"honest_verdict": "blocked_gate_check_failed"`; `"failed_field": "gpu_lease_scheduler_ready_score"`; `"failed_expected": 1.0`; `"failed_observed": 0.0`; `"passed": false`; `"blocked_at_layer": "conductor_pre_gate"`

## RECOMMENDATION
KEEP

## experiment_6636_delayed_two_level_decoding.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no substantive performance or value claim to falsify; this is only a blocked-gate receipt.

## WAS THAT CHECKED
No; the method was not run. Only the prerequisite gate was checked, and it failed because the upstream artifact was unavailable.

## EVIDENCE
`status` `blocked` `honest_verdict` `blocked_gate_check_failed` `blocked_reason` `upstream artifact not found for task id 'exp6635-matched-direct-headroom'` `failed_observed` `null` `passed` `false` `blocked_at_layer` `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_6605_qwen36_direct_headroom.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative or value claim is made. The operational block would be overturned by authentic GPU-process receipts and a complete integrity recomputation.

## WAS THAT CHECKED
Yes, for the operational block: the GPU-process receipt and integrity gates were evaluated and failed. No method-value comparison was attempted.

## EVIDENCE
`honest_verdict` is `blocked_gpu_process_receipts: direct baseline integrity did not complete`; `all_sessions_authentic` is `false`; `complete` is `false`; `failed_checks` contains `gpu_process_receipts`; the artifact describes `baseline qualification alone is null infrastructure.`

## RECOMMENDATION
KEEP
