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
| NO_CLAIM | 2 |
| CANNOT_DETERMINE | 1 |
| SKIPPED_ALREADY_FLAGGED | 4 |

## experiment_3391_archive_v312_activate_v313.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted evidence [' and '] does not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
Milestone 2026.05.312 is complete and ready to archive while 2026.05.313 is ready to activate.

## WHAT WOULD REFUTE IT
Any artifact still classified as blocked at the readiness decision would refute an unconditional “complete” and “ready” claim.

## WAS THAT CHECKED
Yes. The artifact includes a blocked-artifact check, and it found one blocked artifact despite returning readiness as true.

## EVIDENCE
`"blocked_artifacts"` contains `"exp3382-gatemate-n16-flash-and-smoke"`, while `"archive_v312_activate_v313_ready"` is `true` and `"status"` is `"success"`.

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_3403_archive_v313_activate_v314.json

**CLAIM_REFUTED_BY_OWN_DATA**

## VERDICT
CLAIM_REFUTED_BY_OWN_DATA

## THE HEADLINE CLAIM
Milestone v313 was completely archived and v314 was ready for activation.

## WHAT WOULD REFUTE IT
Any required artifact remaining blocked would refute the claim of complete archival and activation readiness.

## WAS THAT CHECKED
Yes, in the blocked-artifact inventory; it contains a blocked artifact.

## EVIDENCE
`"honest_verdict": "complete: archive_v313_activate_v314_ready=true"`; `"blocked_artifacts"`; `"exp3392-gatemate-n16-bootstrap-fix"`; `"archive_v313_activate_v314_ready": true`; `"status": "success"`

## RECOMMENDATION
CORRECT_THE_RECORD

## experiment_3361_archive_v309_activate_v310.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_3377_archive_v310_activate_v311.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_3392_archive_v311_activate_v312.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_1644_cerce_ledger.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a scaffolding/implementation receipt with no comparative or value claim.

## WAS THAT CHECKED
No; there were no evaluated policies, events, violations, updates, or ledger rows.

## EVIDENCE
`"status": "complete"`; `"ledger_implemented": true`; `"policy_certificates_evaluated": 0`; `"fr11_events_recorded": 0`; `"ledger_rows": []`; `"honest_verdict": "complete: cerce_ledger_added"`

## RECOMMENDATION
KEEP

## experiment_1736_kanele_synth.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6577_flagship_source_stream_independent_audit.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is a blocked audit receipt, not a positive or comparative claim. An existing, readable upstream artifact with replayed claim-bearing rows would refute only its reported blocked status.

## WAS THAT CHECKED
Yes, in `upstream_artifact_receipt` and `aggregate_row_recomputation`; the upstream artifact was missing, so no audit rows or attacks could run.

## EVIDENCE
`"status": "blocked_flagship_source_stream_independent_audit"`; `"verdict_class": "blocked"`; `"exists": false`; `"read_error": "missing"`; `"rows": []`; `"claim_bearing_row_count": 0`; `"observed": "not_run_missing_upstream"`; `"verifier_is_oracle": true`

## RECOMMENDATION
KEEP
