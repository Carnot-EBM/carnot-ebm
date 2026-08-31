# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_6801_real_output_fixed_point_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Grouped fixed-point performance differs only marginally from the flat recurrent control across transformations, with every reported 95% clustered confidence interval including zero.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6802_operational_obligation_automaton_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required `openspec/capabilities/agentic-verification/spec.md` source artifact did not exist.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6803_sota_operational_handoff_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because `operational_automaton_fixture_ready` was `false` but expected to be `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6810_v595_contract_manifest_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V595 owned contracts and active execution manifest agree, so `v595_contract_map_ready` is true.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6811_operational_obligation_automaton_v3.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The deterministic source-free operational-obligation fixture is ready, with no live benefit or level solve claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6812_sota_operational_handoff_corpus_v2.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6813_selective_priority_arbiter_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The selective-priority arm passed the positive gate by improving held accepted progress over flat-reject-retry without increased harmful selections, hard violations, family-support loss, or excessive false intervention.

## WHAT IS MISSING
nothing; per-unit metrics are recorded in `"rows"` using fields including `"pair_id"`, `"split"`, `"arm"`, `"accepted_progress"`, `"false_intervention"`, `"harmful_selection"`, and `"accepted_hard_violation"`, while `"gate_check_summary"` records named checks and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_6823_v595_branch_disposition.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The selective-priority arm passed its positive gate by outperforming flat-reject-retry, while other branches remain blocked by named missing evidence.

## WHAT IS MISSING
Per-unit paired metric rows underlying `"row_recomputed_claims.selective_arbiter"`; the present `"rows"` contain only task and branch summaries, while `"paired_progress_delta"`, `"accepted_progress_by_arm"`, and `"retry_cost_by_arm"` are aggregates.

## THE CHECK A READER CANNOT DO
Did the reported mean paired-progress improvement of 0.125 occur broadly across the 144 pairs, or was it driven by a small number of outliers or units with unusual headroom?
