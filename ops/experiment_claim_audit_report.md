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
| CLAIM_SUPPORTED | 3 |
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 2 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6811_operational_obligation_automaton_v3.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No observation could refute a live-benefit or level-solve claim because the artifact explicitly makes neither; a failed readiness component, failed gate, replay mismatch, or hard violation would instead refute its limited fixture-readiness statement.

## WAS THAT CHECKED
Yes, for fixture readiness: the artifact reports gate, schema, compiler, replay, compatibility, attack, deterministic-byte, and hard-violation checks. No comparative value or live-solving test was required because no such claim was made.

## EVIDENCE
`"honest_verdict"`: `"complete: deterministic source-free operational-obligation fixture ready; no live benefit or level solve claimed"`; `"solve_claim"`: `false`; `"verdict_class"`: `"null"`; `"operational_automaton_fixture_ready"`: `true`; `"failed_checks"`: `[]`; `"passed"`: `true`; `"hard_violation_count"`: `0`; `"verifier_is_oracle"`: `false`; `"inference_substrate"`: `"deterministic CPU automaton, no LLM"`

## RECOMMENDATION
KEEP

## experiment_6812_sota_operational_handoff_corpus_v2.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All 576 authentic rows are ready, and direct-typed operational preservation exceeds compressed prose.

## WHAT WOULD REFUTE IT
A direct-typed operational-preservation rate less than or equal to compressed prose, or incomplete/inauthentic retained rows, would refute the claim.

## WAS THAT CHECKED
Yes. The arm-level preservation aggregates show 23/288 for direct typed versus 8/288 for compressed prose, while the terminal gates check row completeness, exact row checks, raw bytes, checkpoints, and model coverage. Readiness was explicitly independent of the effect’s direction, so the comparison could have failed.

## EVIDENCE
`honest_verdict`: `complete: all 576 authentic rows are ready; direct typed preservation exceeds compressed prose`; `operational_preservation_by_arm`; `direct_typed`; `numerator`: `23`; `denominator`: `288`; `rate`: `0.0798611111111111`; `compressed_prose`; `numerator`: `8`; `denominator`: `288`; `rate`: `0.027777777777777776`; `effect_sign_controls_readiness`: `false`; `rows_complete`: `true`; `row_checks_exact`: `true`; `raw_bytes_complete`: `true`; `checkpoint_cells_complete`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP

## experiment_6813_selective_priority_arbiter_ab.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence [' In an exhibited row with ', ', while its paired flat arm has ', '. Aggregate ', ' for flat reject-and-retry. The reported ', ' has '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Selective priority provides a positive held-replay benefit over flat reject-and-retry.

## WHAT WOULD REFUTE IT
A base-preserving first-valid policy—preserve an already-valid proposal, otherwise accept the first fully valid candidate—tying or beating selective priority on paired held progress would refute the claimed added value of priority ordering.

## WAS THAT CHECKED
No. The only comparator is a first-valid arm that, unlike selective priority, does not preserve a valid base proposal. Thus the experiment compares two policy changes at once and does not give the priority-ordering claim a serious chance to fail. The paired held test did allow selective priority to lose against that weaker arm, but it cannot isolate which policy difference produced the win.

## EVIDENCE
The comparator is defined as `flat_reject_retry`: `Inspect the same candidates in frozen order. Reject any failed constraint and accept the first fully valid candidate.` The tested arm is defined as `selective_priority`: `Preserve a valid base byte string. Otherwise choose the exact hard-binding-soft lexicographic minimum with a stable index tie.` In an exhibited row with `base_already_valid` equal to `true`, selective priority has `selected_candidate_id` equal to `base_proposal` and `accepted_progress` equal to `1`, while its paired flat arm has `abstention` equal to `true`, `false_intervention` equal to `true`, and `accepted_progress` equal to `0`. Aggregate `false_intervention_rate_by_arm` is `0.0` for selective priority and `0.6428571428571429` for flat reject-and-retry. The reported `paired_progress_delta` has `direction` `selective_minus_flat`, `estimate` `0.125`, and `lower_bound` `0.0763888888888889`.

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6823_v595_branch_disposition.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
V595 preserved all available terminal rows while keeping branches blocked where independent evidence was missing or excluded.

## WHAT WOULD REFUTE IT
A present terminal artifact omitted from the rows, an ineligible or missing row counted as positive evidence, or a branch advanced despite a failed evidence gate would refute the claim.

## WAS THAT CHECKED
Yes. The artifact reconciles the 14-task design and execution manifests, hashes expected source artifacts, records missing and flagged evidence, recomputes eligibility, and blocks affected branches.

## EVIDENCE
`"honest_verdict": "complete_partial: V595 preserved every available terminal row; one or more branches remain blocked by named missing or excluded independent evidence."`; `"design_task_count": 14`; `"executed_task_count": 14`; `"task_count": 14`; `"differences": []`; `"matches": true`; `"eligible_for_positive_claim": false`; `"exclusion_reason": "terminal_class=flagged"`; `"observed": "file_missing"`; `"disposition": "blocked"`; `"verdict_class": "partial"`; `"solve_claim": false`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6824_selective_arbiter_cold_row_replay.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence [', while ', '. The corresponding ', ' numerators are also ', ' matches the false-intervention difference: ', ' versus ', '. The only named arms are '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
The cold replay supports a positive value claim for `selective_priority` over `flat_reject_retry`, increasing accepted progress without safety or support loss.

## WHAT WOULD REFUTE IT
A matched baseline that preserves an already-valid base proposal and otherwise uses flat reject/retry would tie or beat selective priority. Likewise, no improvement among rows where the base was not already valid would show that the full arbiter adds nothing beyond that trivial guard.

## WAS THAT CHECKED
No. The artifact compares only selective priority with unconditional flat reject/retry; it provides neither the validity-preserving flat baseline nor a paired progress analysis stratified by whether the base was already valid. Thus the serious tie condition was not given a real chance to appear.

## EVIDENCE
The `selective_priority` accepted-progress mean is `0.19444444444444445`, exactly `28` of `144`, while `flat_reject_retry` is `0.06944444444444445`, exactly `10` of `144`. The corresponding `safe_action_identity_by_arm` numerators are also `28` and `10`, and the progress difference `0.125` matches the false-intervention difference: `18` versus `0`. Both arms have hard-violation rates of `0.0`, harmful-selection rates of `0.0`, and legal-support rates of `1.0`. The only named arms are `selective_priority` and `flat_reject_retry`.

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6825_selective_arbiter_authority_attacks.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent mutation tests support the selective arbiter’s hard-authority boundary, without establishing adoption value.

## WHAT WOULD REFUTE IT
An applicable mutation that was accepted despite a hard or binding authority violation, allowed a prohibited feature to influence selection, altered required safe-action bytes, failed to reject corrupted row evidence, or produced different fresh-process replay rows.

## WAS THAT CHECKED
Yes. The artifact reports applicable priority, certificate, prohibited-feature, safe-action, and row-integrity attacks, plus a byte-identical fresh-process replay; all reported zero failures or the expected fail-closed behavior.

## EVIDENCE
`hard_authority_supported`: `true`; `adoption_decision`: `not_evaluated`; `verifier_is_oracle`: `false`; `imports_exp6813`: `false`; `failed_count`: `0`; `influence_detected`: `false`; `failed_closed`: `true`; `byte_identity_enforced`: `true`; `fresh_process`: `true`; `byte_identical`: `true`; `authority_attack_shard_complete`: `true`; `honest_verdict`: `complete: independent mutations support the hard authority boundary; adoption was not evaluated`

## RECOMMENDATION
KEEP

## experiment_6826_selective_arbiter_sealed_adoption.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6827_chronological_causal_edge_memory_stream.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A failed readiness gate, incomplete or noncanonical chronology, leakage of sealed fields, broken family isolation, or missing required operation coverage would refute the artifact’s construction/readiness statement.

## WAS THAT CHECKED
Yes. The gate summary checked source preconditions, chronology and row counts, canonical rows, sealed fields, family isolation, headroom, and operation outcomes; all reported passing. No learning-effect or comparative claim was made or tested.

## EVIDENCE
`honest_verdict`: `complete: frozen chronological causal-edge memory stream is ready; no learning ran`; `status`: `complete_chronological_causal_edge_memory_stream`; `verified_memory_stream_ready`: `true`; `failed_checks`: `[]`; `passed`: `true`; `verifier_is_oracle`: `false`

## RECOMMENDATION
KEEP
