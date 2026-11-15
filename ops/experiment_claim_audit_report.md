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

## experiment_9096_v776_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific claim to refute. A failed current required check or mismatched task/source binding would contradict the narrower administrative readiness assertion.

## WAS THAT CHECKED
Yes, administrative checks are reported in acceptance_gates. Scientific comparisons remain explicitly unmeasured; historical oracle results and exposed development data are not promoted into current scientific success.

## EVIDENCE

- `honest_verdict`: `complete_null_v776_contract`
- `claim_scope`: `Administrative human-decision contract only; historical exact-oracle success remains circular_positive. CPU fixtures establish no scientific benefit; no current ARC solve is claimed.`
- `exposure_scope`: `LPHB exposed development; no independent generalization; no new scientific measurements.`
- `inference_substrate`: `aggregation_from_upstream_artifacts`

## RECOMMENDATION
KEEP

Unchecked: elided rows and referenced files. Administrative readiness supplies no evidence of scientific benefit.

## experiment_9097_v776_human_authority.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is an administrative receipt without a comparative scientific claim. Its custody assertion would be contradicted by a pinned-source hash mismatch, disagreement between original and replayed labels or roles, or a failed required acceptance gate.

## WAS THAT CHECKED
Yes, as reported: explicit expected-versus-observed byte checks, an isolated 900-row replay, and required validation checks could expose failures and reportedly passed. The artifact does not present inventory completeness as model accuracy or generalization.

## EVIDENCE

- `claim_scope`: `Authenticated original producer, exposed label custody, retained roles and sealed input views only. Administrative readiness establishes no scientific benefit; no current ARC solve is claimed.`
- `inference_substrate_class`: `no_model_load`
- `replay_receipt`: `compared_rows`: `900`, `isolated`: `true`, `passed`: `true`
- `required_checks_passed`: `true`
- `independent_generalization_score`: `0`

## RECOMMENDATION
KEEP

Not checked: external references or elided material; the custody assessment relies on the reported checks.

## experiment_9098_v776_embedding_runtime.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
On twelve fixed texts, the runtime qualifies for raw-logit scoring but fails the frozen embedding-parity requirements.

## WHAT WOULD REFUTE IT
All required embedding checks passing would contradict the reported embedding limitation. Any required raw-logit check failing—such as nonfinite or masked logits, or incomplete continuation scoring—would contradict logits readiness.

## WAS THAT CHECKED
Yes. The parity checks could fail, and fixture-02 actually fails padding parity; the embedding gate records that failure rather than declaring readiness. The displayed raw-logit checks pass, consistent with the positive logits gate.

The verifier is not the oracle. These are exposed runtime fixtures, with no claimed generalization or comparative scientific benefit requiring a rival baseline.

The failed embedding row remains counted. Twelve fixtures are reported completed with none censored; completion does not imply embedding validity. The cited readiness fields exist, with an explicit embedding failure rather than an inferred zero from a missing key.

## EVIDENCE

- `honest_verdict`: `complete_null_runtime_capability_limit`
- `embeddings_ready_score`: `0`; `logits_ready_score`: `1`
- `fixture-02`: `embedding_passed`: `false`; `embedding_error`: `padding_parity`
- `condition`: `padding`; `max_abs_error`: `0.002528320330808076`
- `raw_logits_finite`: `true`; `raw_logits_unmasked`: `true`; `raw_logits_width`: `248320`
- `intended_count`: `12`; `completed_count`: `12`; `censored_count`: `0`

## RECOMMENDATION
KEEP

Elided rows and referenced files were not independently inspected; support is limited to the reported fixture-level runtime result.

## experiment_9099_v776_online_methods.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is asserted. The narrower mechanics qualification would fail if future labels influenced earlier predictions, duplicate feedback caused repeated updates, or recovered state differed from uninterrupted state.

## WAS THAT CHECKED
Yes, for mechanics: causal controls report passes, and three forced-exit/resume sequences accompany the exact-recovery result. Readiness depends on these checks, not on favorable learning outcomes. Human benefit and independent generalization remain explicitly unestablished.

## EVIDENCE

- `honest_verdict`: `complete_null_online_methods_qualified`
- `claim_scope`: `Frozen causal human coefficient methods and exact CPU-fixture recovery only; no measured human learning benefit or current ARC solve.`
- `online_methods_ready_score`: `Integer1 requires authentic inputs, causal controls, exact recovery and every owned check, not a favorable scientific result.`
- `exact_resume`: `true`
- `independent_count`: `0`
- `learning_signal_score`: `0`

## RECOMMENDATION
KEEP

Review limited to the supplied excerpt; no reruns or unseen rows checked. Human-benefit extrapolation remains unsupported.

## experiment_9100_embeddings_a.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no experimental result to falsify. The blocking diagnosis would be contradicted if the embedding-readiness gate’s actual value equaled its expected value of 1.

## WAS THAT CHECKED
Yes, for the blocking diagnosis: the embedding gate records actual 0, expected 1, and failure. This is a prerequisite-check receipt, not a comparative result.

## EVIDENCE
- `schema`: `blocked_gate_check_v1`
- `status`: `blocked`
- `honest_verdict`: `blocked_gate_check_failed`
- `failed_field`: `embeddings_ready_score`
- `failed_expected`: `1`
- `failed_observed`: `0`
- `blocked_at_layer`: `conductor_pre_gate`

## RECOMMENDATION
KEEP

## experiment_9101_embeddings_b.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable: this artifact records blocked execution, without claiming successful extraction or comparative value.

## WAS THAT CHECKED
Yes—the prerequisite readiness gate required 1, observed 0, and blocked execution. That supports the reported gate failure.

## EVIDENCE
`schema`: `blocked_gate_check_v1`; `status`: `blocked`; `honest_verdict`: `blocked_gate_check_failed`; `failed_field`: `embeddings_ready_score`; `blocked_reason`: `actual=0 == expected=1`; `blocked_at_layer`: `conductor_pre_gate`.

## RECOMMENDATION
KEEP

Not checked: upstream artifacts or a rerun. This receipt establishes no extraction outcome.

## experiment_9107_v776_qwen_judge.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_9109_v776_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No comparative scientific claim is advanced. Counting failed prerequisites or unmeasured branches as scientific success would contradict the stated administrative scope.

## WAS THAT CHECKED
Yes, for that scope boundary: gate_failure_rows records the embedding prerequisite failing, evidence_table marks unavailable measurements, and acceptance_gates retains zero scientific readiness. Administrative completion is not presented as scientific success.

## EVIDENCE

- `claim_scope`: `Fourteen administrative outcomes and retained descriptive evidence only; no independent factuality, learned-head or causal benefit claim.`
- `artifact_field`: `embeddings_ready_score`; `expected`: `1`; `observed`: `0`; `passed`: `false`
- `acceptance_gates`: `scientific_readiness`: `0`; `independent_generalization_score`: `0`
- `status`: `administrative_only`
- `status`: `unmeasured`

## RECOMMENDATION
KEEP

Not checked: elided upstream evidence. This assessment covers only the supplied excerpt.
