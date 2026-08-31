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
| CLAIM_OVERSTATED | 1 |
| NO_CLAIM | 3 |
| SKIPPED_ALREADY_FLAGGED | 1 |

## experiment_6811_operational_obligation_automaton_v3.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
The limited operational-readiness assertion would be falsified by a failed readiness component, a failed gate, non-identical replay receipts, an accepted malformed mutation, or a hard violation; none would refute live benefit or solving because neither is claimed.

## WAS THAT CHECKED
Yes, for fixture readiness: separate schema, compiler, replay, compatibility, attack, gate, and hard-violation results could have failed. No comparative value or live-solve test was attempted, appropriately matching the artifact’s null scope.

## EVIDENCE
`"honest_verdict"`: `"complete: deterministic source-free operational-obligation fixture ready; no live benefit or level solve claimed"`; `"verdict_class"`: `"null"`; `"solve_claim"`: `false`; `"solve_provenance"`: `"development_proxy"`; `"operational_automaton_fixture_ready"`: `true`; `"failed_checks"`: `[]`; `"hard_violation_count"`: `0`; `"attack_coverage"`: `true`; `"backward_compatibility"`: `true`; `"compiler"`: `true`; `"replay"`: `true`; `"schema"`: `true`; `"failed_closed"`: `true`; `"outcome"`: `"rejected"`; `"fresh_process"`: `true`; `"hashes_byte_identical"`: `true`.

## RECOMMENDATION
KEEP

## experiment_6812_sota_operational_handoff_corpus_v2.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
All 576 authentic rows are ready, and direct-typed handoffs achieve higher operational-preservation rates than compressed prose.

## WHAT WOULD REFUTE IT
A direct-typed operational-preservation rate equal to or below the compressed-prose rate, or incomplete/invalid rows being included in the claimed 576-row corpus.

## WAS THAT CHECKED
Yes. Both arms were scored over 288 rows using the row-level operational-preservation field, completeness was checked against 576 planned rows, and corpus readiness was independent of the effect direction.

## EVIDENCE
`"planned_row_count": 576`; `"rows_complete": true`; `"row_checks_exact": true`; `"effect_sign_controls_readiness": false`; `"operational_handoff_corpus_ready": true`; `"operational_preservation_by_arm"`; `"direct_typed"`; `"numerator": 23`; `"rate": 0.0798611111111111`; `"compressed_prose"`; `"numerator": 8`; `"rate": 0.027777777777777776`; `"operational_preserved": false`; `"verifier_is_oracle": false`

## RECOMMENDATION
KEEP

## experiment_6813_selective_priority_arbiter_ab.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Selective priority provides a positive effect on held exact replay.

## WHAT WOULD REFUTE IT
A base-preserving first-valid comparator—preserve an already-valid base, otherwise accept the first fully valid candidate—tying or beating selective priority on paired progress, retry cost, false interventions, or safety would refute added value from priority ordering itself; a nonpositive paired progress lower bound or increased harmful selections would also refute the stated gate.

## WAS THAT CHECKED
No. Held paired outcomes, equal budgets, harms, and confidence bounds were checked, but the only rival was a first-valid arm that lacks selective priority’s base-preservation behavior. Thus the experiment did not isolate priority ordering against the cheapest serious baseline.

## EVIDENCE
`"selective_priority": "Preserve a valid base byte string. Otherwise choose the exact hard-binding-soft lexicographic minimum with a stable index tie."`; `"flat_reject_retry": "Inspect the same candidates in frozen order. Reject any failed constraint and accept the first fully valid candidate."`; `"headline_split": "held"`; `"estimate": 0.125`; `"lower_bound": 0.0763888888888889`; `"false_intervention_rate_by_arm"`; `"selective_priority"`; `"rate": 0.0`; `"flat_reject_retry"`; `"rate": 0.6428571428571429`; `"harmful_selections_by_arm"`; `"rate": 0.0`; `"equal_observed_work": true`

## RECOMMENDATION
ADD_MISSING_CONTROL

## experiment_6823_v595_branch_disposition.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
For its limited receipt function, an available terminal row being omitted or a branch being promoted despite missing or excluded evidence would refute the artifact’s completeness and disposition statements.

## WAS THAT CHECKED
Yes. The artifact records task rows, eligibility, exclusions, terminal-class totals, source hashes, failed gate checks, and blocked branch dispositions. It explicitly declines a solve or comparative-value claim.

## EVIDENCE
`honest_verdict` `complete_partial: V595 preserved every available terminal row; one or more branches remain blocked by named missing or excluded independent evidence.` `solve_claim` `false` `verdict_class` `partial` `task_count` `14` `flagged` `2` `missing` `9` `positive` `2` `disposition` `blocked` `eligible_for_positive_claim` `false`

## RECOMMENDATION
KEEP

## experiment_6824_selective_arbiter_cold_row_replay.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Cold replay supports the bounded claim that selective priority improves accepted progress and retry cost over flat reject-and-retry without increasing measured safety violations or losing legal support.

## WHAT WOULD REFUTE IT
A nonpositive selective-minus-flat paired progress effect, a confidence lower bound at or below zero, unequal budgets, any accepted hard violation or harmful-selection increase, lost legal support, or excessive false intervention would falsify the claim.

## WAS THAT CHECKED
Yes. The paired arm comparison, confidence interval, matched budgets, hard violations, harmful selections, legal support, false interventions, and full row coverage were independently recomputed. The artifact does not test broader superiority over a valid-base-preserving, first-legal-candidate baseline, so the claim remains specific to `flat_reject_retry`.

## EVIDENCE
`paired_progress_delta`, `direction`, `selective_minus_flat`, `estimate`, `0.125`, `lower_bound`, `0.0763888888888889`, `mismatched_pair_ids`, `[]`, `hard_violation_rate_by_arm`, `0.0`, `harmful_selections_by_arm`, `0.0`, `legal_support_by_arm`, `1.0`, `false_intervention_upper_limit`, `0.2`, `upper_bound`, `0.12064330476584559`, `all_producer_rows_recomputed`, `true`, `verifier_is_oracle`, `false`

## RECOMMENDATION
KEEP

## experiment_6825_selective_arbiter_authority_attacks.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
Independent mutation attacks support the selective arbiter’s hard-authority boundary, without establishing adoption value.

## WHAT WOULD REFUTE IT
Any applicable attack producing a failure, accepting a hard-authority violation, responding to a prohibited label, failing open under row-integrity mutation, altering required safe-action bytes, or producing nonidentical fresh-process replay rows.

## WAS THAT CHECKED
Yes. Applicable priority, certificate, prohibited-feature, row-integrity, and safe-action attacks report zero failures; representative rows expose local outcomes; and fresh-process replay checks byte identity. Non-applicable rows are separated through applicability counts.

## EVIDENCE
`"hard_authority_supported": true`; `"adoption_decision": "not_evaluated"`; `"verifier_is_oracle": false`; `"failed_count": 0`; `"influence_detected": false`; `"failed_closed": true`; `"byte_identity_enforced": true`; `"byte_identical": true`; `"accepted_hard_violation": false`; `"authority_attack_shard_complete": true`; `"verdict_class": "positive"`

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
For the artifact’s operational readiness assertion, incomplete chronology, noncanonical rows, exposed sealed fields, failed family isolation, or absent operation/read coverage would refute readiness; no learning-effect or comparative claim exists to refute.

## WAS THAT CHECKED
Yes. The gate checks cover source preconditions, chronology, canonical rows, sealed fields, family isolation, headroom, and operation outcomes.

## EVIDENCE
`"honest_verdict"`: `"complete: frozen chronological causal-edge memory stream is ready; no learning ran"`; `"status"`: `"complete_chronological_causal_edge_memory_stream"`; `"verified_memory_stream_ready"`: `true`; `"failed_checks"`: `[]`; `"passed"`: `true`; `"inference_substrate"`: `"CPU transformation of frozen authentic outputs, no LLM"`; `"verifier_is_oracle"`: `false`

## RECOMMENDATION
KEEP
