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
| CLAIM_SUPPORTED | 2 |
| NO_CLAIM | 5 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_8986_v768_spline_basis_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative value claim to refute. The limited mechanics readiness assertion would fail if a fitted head were nonfinite, its actual gradient exceeded 1e-6, or derivative, invariance, or summary-recomputation checks failed.

## WAS THAT CHECKED
Yes, for mechanics: all six fitted heads report finite values and gradients below the threshold; fixture, invariance, and summary checks pass.

Scientific benefit remains explicitly unmeasured. The linear comparator is present, but development results are not claimed as generalization. Elided rows were not independently checked.

## EVIDENCE
- `claim_scope`: `fixed-basis numerical mechanics only; no new scientific benefit; graph truth has a cheap BFS evaluator`
- `honest_verdict`: `complete_null_convex_spline_mechanics_frozen`
- `verdict_class`: `null`
- `gradient_inf`: `9.585669368272233e-07`
- `metric`: `finite_difference_and_convexity`; `passed`: `true`
- `exposure_scope`: `all V765/V766 labels are exposed development; original256 fit and64 guard graphs only`
- `status`: `frozen_unmeasured_until_fresh_calibration_and_test`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

## experiment_8987_v768_spline_development_admission.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The spline meets fixed development-feasibility gates on exposed guard graphs, permitting fresh collection without establishing independent scientific benefit.

## WHAT WOULD REFUTE IT
Guard AUROC below 0.65, Brier improvement over fit prevalence below 0.005, or Brier exceeding either shared logistic or converged linear by more than 0.01 would defeat admission; nonfinite or unconverged fitted states would defeat numerical readiness.

## WAS THAT CHECKED
Yes. The four development_admission_rows report passing measurements against those thresholds. Predictions come from coefficients fitted on 256 graphs; admission uses error labels on 64 exposed guard graphs. Optimizing the fit objective does not guarantee passing those guard metrics.

The cheapest serious comparator—an equally optimized linear head using the same frozen features—is present and loses on development Brier, rather than tying. All six fitted heads report convergence. The displayed scoring rows are completed, with zero exclusions and failures reported; no scoring-validity or wrong-key contradiction is shown.

Exposure is explicitly acknowledged, and the verdict records no independent benefit. The non-oracle pre-pass raises no circularity finding. Review is limited to the supplied artifact; fresh generalization remains untested.

## EVIDENCE
- `honest_verdict`: `complete_null_development_admitted`
- `claim_scope`: `Exposed development feasibility only; no independent generalization, no oracle-distinct verifier win.`
- `spline_auroc`: `value`: `1.0`; `threshold`: `0.65`
- `brier_improvement_over_fit_prevalence`: `value`: `0.16027175302639896`; `threshold`: `0.005`
- `arm_brier`: `spline`: `0.027975927637663548`; `converged_linear`: `0.06362832483145615`; `shared_logistic`: `0.18022024514665103`
- `independent_generalization_score`: `0`; `excluded_count`: `0`; `failed_count`: `0`

## RECOMMENDATION
KEEP

## experiment_8988_v768_online_spline_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative benefit claim is made. The limited mechanics qualification would fail if feedback affected predictions before release, labels were consumed twice, rejected proposals changed committed state, or restart changed state or predictions.

## WAS THAT CHECKED
Yes, for mechanics: causal and duplicate-feedback fixtures passed; rejected promotion rows show matching before/after hashes; an actual SIGKILL/restart reproduced state and predictions exactly.

Scientific benefit remains explicitly unmeasured. Frozen, bias-only and adaptive-linear comparators are specified, but the displayed first-query ties establish no comparative outcome. Failed promotion decisions represent exercised rejection paths, not invalid scored rows.

Review was limited to the supplied excerpt; elided rows and checks were not independently inspected.

## EVIDENCE
- `claim_scope`: `Delayed fixed-basis coefficient mechanics only; no scientific benefit or oracle-distinct verifier win.`
- `honest_verdict`: `complete_null_online_spline_methods_qualified`
- `verdict_class`: `null`
- `causal_fixture_rows`: `causal_sentinel`, `duplicate_feedback`, `rejected_state`; `passed`: `true`
- `condition`: `real_sigkill_query16`; `state_bit_exact`: `true`; `prediction_bit_exact`: `true`
- `evaluation_status`: `frozen_unmeasured; later real streams required`
- `scientific_evidence_fixture_only`: `true`

## RECOMMENDATION
KEEP

## experiment_8989_v768_natural_annotation_authority.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The inspected public release documents machine annotations without documented independent human labeling.

## WHAT WOULD REFUTE IT
An item-level record establishing independent human labeling or adjudication in the inspected release would contradict the reported zero-human-evidence finding.

## WAS THAT CHECKED
Yes. The full-release annotation-origin and adjudication audit reports zero documented independent human decisions, and the independent-label gate fails. This supports the scoped null.

No verifier advantage or held-out generalization is claimed, so no rival comparison is needed. The 50 alignment failures are explicitly retained as inspected responses; their inclusion does not contradict the annotation-provenance finding.

## EVIDENCE

- `honest_verdict`: `complete_null_machine_annotation_only`
- `claim_scope`: `Public release/pipeline audit only; no verifier benefit or independent factuality evaluation.`
- `scope`: `full_release`
- `origin`: `machine_annotation_or_deterministic_projection`; `independent_human_decision`: `false`
- `documented_independent_human_decisions`: `0`
- `human_review_evidence`: `not_documented_in_inspected_release`
- `adjudication_evidence`: `no_independent_human_adjudication_documented`
- `independent_label_ready_score`: `0`
- `completed_count`: `Response variants inspected, including invalid offset observations; no evidence is discarded.`

## RECOMMENDATION
KEEP

Not rerun; the conclusion concerns documentation in the inspected release, not whether human review ever occurred elsewhere.

## experiment_8990_v768_graph_corpus_a.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_8992_corpus_admission.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no substantive claim to falsify: this artifact records blocked admission, not successful authentication or comparative value.

## WAS THAT CHECKED
Yes, admission prerequisites were checked. One readiness score was 0 instead of 1; the other upstream artifact was missing. Both gates failed, and execution stopped before evaluation. Authentication and truth isolation remain untested.

## EVIDENCE
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `blocked_reason`: `actual=0 == expected=1`
- `reason`: `upstream artifact not found for task id 'exp8991-graph-corpus-b'`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8997_arc_transfer_evidence.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records a blocked prerequisite and makes no claim that supervisor transfer succeeds or adds value.

## WAS THAT CHECKED
The prerequisite was checked in `gates_evaluated`: readiness was 0 against a required 1. Transfer was not assessed; this receipt supports no transfer conclusion.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `artifact_field`: `arc_new_outcomes_ready_score`
- `expected`: `1`
- `actual`: `0`
- `passed`: `false`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_8998_v768_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific headline to falsify. An omitted task or a failed producer counted as qualified scientific evidence would contradict the administrative reconciliation.

## WAS THAT CHECKED
Partly. Visible producer-validation and qualification failures are retained, and downstream scientific comparisons remain unmeasured. Development results are explicitly exposed, and no independent verifier value is asserted. These failures support the blocked disposition rather than contradicting it.

## EVIDENCE

- `claim_scope`: `Fourteen administrative outcomes and exposed development reductions; no qualified fresh detector, online benefit or independent verifier claim.`
- `honest_verdict`: `complete_blocked_upstream_science_unavailable`
- `scientific_benefit`: `false`
- `check`: `producer_validation`; `observed`: `false`; `passed`: `false`
- `status`: `unmeasured`
- `scope`: `exposed_development_only`

## RECOMMENDATION
KEEP

Unchecked: elided rows and upstream artifacts. Risk: administrative readiness establishes no fresh scientific benefit.
