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
| SKIPPED_ALREADY_FLAGGED | 6 |

## experiment_3403_archive_v313_activate_v314.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Not applicable; this is an operational milestone receipt, not a comparative or value claim.

## WAS THAT CHECKED
No; the artifact records aggregated readiness and status fields but tests no method against a falsifying control or rival.

## EVIDENCE
`"inference_substrate": "aggregation_from_upstream_artifacts"`, `"archive_v313_activate_v314_ready": true`, `"status": "success"`, `"files_updated": []`

## RECOMMENDATION
KEEP

## experiment_2514_kv260_pynq_flash.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
A recorded failure to generate the KV260 HWH file, or an attempted physical SD-card flash that failed, would contradict the artifact’s status statements.

## WAS THAT CHECKED
Yes for HWH generation: the artifact records a boolean result and output path. No for physical flashing: it explicitly records that flashing was not attempted.

## EVIDENCE
`kv260_hwh_generated`: `true`; `kv260_hwh_path`: `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/output/carnot_ising_v4_bd/project/carnot_ising_v4.gen/sources_1/bd/carnot_ising_v4_bd/hw_handoff/carnot_ising_v4_bd.hwh`; `kv260_flash_attempted`: `false`; `Physical SD card flash not attempted as PYNQ SD card preparation is a documented manual operator step.`

## RECOMMENDATION
KEEP

## experiment_3361_archive_v309_activate_v310.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_3377_archive_v310_activate_v311.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_3392_archive_v311_activate_v312.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_833_constraint_delta_root_cause.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6582_gemma4_31b_flagship_source_shard.json

**SKIPPED_ALREADY_FLAGGED**

## experiment_6583_gemma4_26b_a4b_flagship_source_shard.json

**SKIPPED_ALREADY_FLAGGED**
