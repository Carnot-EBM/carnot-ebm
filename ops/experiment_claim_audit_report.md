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
| NO_CLAIM | 6 |
| SKIPPED_ALREADY_FLAGGED | 2 |

## experiment_9107_v776_qwen_judge.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_9109_v776_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is advanced. Counting a failed prerequisite as established scientific readiness would contradict the receipt’s stated scope.

## WAS THAT CHECKED
Yes, for the displayed prerequisite: the embeddings capability gate expected 1, observed 0, and failed. Scientific readiness remains zero, and the downstream feature branch remains unmeasured. This demonstrates an acknowledged blocker, without asserting scientific benefit.

## EVIDENCE

- `claim_scope`: `Fourteen administrative outcomes and retained descriptive evidence only; no independent factuality, learned-head or causal benefit claim.`
- `check`: `embeddings_capability`; `expected`: `1`; `observed`: `0`; `passed`: `false`.
- `acceptance_gates`: `scientific_readiness`: `0`; `independent_generalization_score`: `0`.
- `branch`: `feature_shard_a`; `status`: `unmeasured`; `metric`: `null`.

## RECOMMENDATION
KEEP

Upstream artifacts and elided rows were not checked; this receipt establishes no scientific benefit.

## experiment_9110_v777_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. Administrative readiness would be contradicted by an authentication failure, task-authority mismatch, or failed required check nevertheless counted as ready.

## WAS THAT CHECKED
Yes, administrative checks are reported in acceptance_gates and the visible gate_check_summary rows. Scientific comparisons remain unmeasured or disqualified, rather than being counted as successful.

## EVIDENCE

- `claim_scope`: `Administrative human-decision contract only; historical exact-oracle success remains circular_positive. CPU fixtures establish no scientific benefit; no current ARC solve is claimed.`
- `honest_verdict`: `complete_null_v777_contract`
- `effective_independent_groups`: `0`
- `graph_study_status`: `deferred_unchanged_seven_and_ten_bounds`
- `claim_scope`: `Disqualified exposed-development diagnostic; no accepted judge comparison.`

## RECOMMENDATION
KEEP

Not checked: elided entries or referenced files. Administrative readiness establishes no scientific benefit.

## experiment_9111_v777_forward_evidence.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_9112_masked_runtime.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this is a blocked execution receipt, with no claim about feature accuracy, tolerances, or comparative value.

## WAS THAT CHECKED
No experimental hypothesis was tested. The prerequisite check in `gates_evaluated` failed: readiness was 0 against required 1, blocking execution before the experiment.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `forward_evidence_ready_score`
- `expected`: `1`; `actual`: `0`; `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9113_energy_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a failed prerequisite gate and makes no comparative claim about energy methods.

## WAS THAT CHECKED
Yes—the prerequisite was checked in gates_evaluated: forward_evidence_ready_score was 0, while 1 was required. No method-performance experiment is reported.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `forward_evidence_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9122_v777_complete_cost.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative claim is asserted. Qualified service measurements would contradict the reported unmeasured status; no method benefit is claimed.

## WAS THAT CHECKED
No service-performance test ran. Upstream readiness checks failed, scientific requests remained unstarted, and service acceptance stayed false. The artifact reports that blockage without promoting administrative completion into scientific success.

## EVIDENCE
- `claim_scope`: `Administrative cost inventory only; complete services and adaptation unmeasured. No human benefit or ARC solve.`
- `honest_verdict`: `complete_blocked_upstream_readiness`
- `inference_substrate_class`: `blocked_no_run`
- `model_invoked`: `false`
- `scientific_request_counts`: `attempted` = `0`; `unstarted` = `360`
- `acceptance_gates`: `complete_service` = `false`
- `gate_check_summary`: `observed` = `missing_file`; `passed` = `false`

## RECOMMENDATION
KEEP

Review limited to the supplied artifact; service performance remains untested.

## experiment_9123_v777_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to falsify. Counting missing, failed or disqualified evidence as scientific success would contradict the artifact’s stated administrative scope.

## WAS THAT CHECKED
Yes, in the visible accounting: missing prerequisites fail their gates, energy bounds remain censored, and scientific readiness is zero. Administrative completion is not presented as scientific success.

## EVIDENCE
- `claim_scope`: `Fourteen administrative outcomes and conditional exposed-development evidence only; no independent human benefit or generalization.`
- `Administrative completion makes no claim of useful human verification.`
- `acceptance_gates`: `reconciliation_ready_score`: `1`; `scientific_readiness`: `0`
- `observed`: `missing_file`; `passed`: `false`
- `censor_reason`: `missing_qualified_scientific_predictions`

## RECOMMENDATION
KEEP

Not checked: elided rows or upstream artifacts. This receipt cannot establish comparative scientific value.
