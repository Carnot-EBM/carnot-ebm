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
| NO_CLAIM | 4 |
| SKIPPED_ALREADY_FLAGGED | 4 |

## experiment_3361_archive_v309_activate_v310.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6572_content_derived_gguf_metadata_resolver.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_1644_cerce_ledger.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; the artifact reports implementation status but makes no comparative or value claim.

## WAS THAT CHECKED
No; no policy certificates or ledger rows were evaluated.

## EVIDENCE
`"ledger_implemented": true`, `"policy_certificates_evaluated": 0`, `"ledger_rows": []`, `"honest_verdict": "complete: cerce_ledger_added"`

## RECOMMENDATION
KEEP

## experiment_1767_e2e_qwen.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No observation could refute a claim because the artifact states only measurements and makes no falsifiable comparative or value claim.

## WAS THAT CHECKED
No; the artifact provides no success criterion, comparator arm, verifier role, row-level results, or headline conclusion to test.

## EVIDENCE
`"latency_ms": 150.5`, `"parse_rate": 0.95`, `"energy_score": 0.88`, `"total_prompts_evaluated": 100`

## RECOMMENDATION
KEEP

## experiment_1736_kanele_synth.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_2031.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
No falsifiable comparative or value claim is stated; the artifact merely records a successful run and a candidate with an energy value.

## WAS THAT CHECKED
No; no success criterion, baseline, oracle check, validity flags, or comparative rows are present.

## EVIDENCE
`"title": "Phase 1: Integrate Gladstone EBT objective with Gemma-4-31B"`, `"status": "success"`, `"best_candidate": "Thus, we can see it."`, `"min_energy": 0.0`

## RECOMMENDATION
KEEP

## experiment_6573_sequential_flagship_gguf_admission_v2.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6574_joint_sufficiency_method_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
The artifact claims only that the joint-sufficiency method contract is ready and frozen; it makes no outcome or comparative-value claim.

## WHAT WOULD REFUTE IT
A failed prerequisite, nonterminal or failing conformance/attack fixture, mutable split or arm, retrospective method change, or unsuccessful reducer test would refute method readiness.

## WAS THAT CHECKED
Yes, for infrastructure readiness: aggregate recomputation, conformance and attack readiness, frozen splits and arms, failed-check reporting, and executable tests are recorded. No comparative outcome was checked, and none is claimed.

## EVIDENCE
`"status": "complete_joint_sufficiency_method_ready"`; `"joint_sufficiency_method_ready_score": 1.0`; `"conformance_row_count": 29`; `"attack_rows_ready": true`; `"frozen_before_live_outcomes": true`; `"outcome_bearing_extraction_observed": false`; `"no_llm_inference": true`; `"failed_checks": []`; `"retrospective_method_change": false`; `"verdict_class": null`; `"verifier_is_oracle": true`; `"Method readiness is infrastructure evidence, not positive science."`; `"Exact fixture checks are oracle authority and cannot create a scientific positive."`

## RECOMMENDATION
KEEP
