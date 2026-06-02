# PRM Missing Label Ledger V2

- Spec: REQ-VERIFY-1434 / SCENARIO-VERIFY-1434
- Project root: `/repo`
- Run date: `20260506`
- Missing labels before replay: 2
- Missing labels filled: 1
- Missing labels remaining: 1

## Recovery Summary

| trace_id | source_case_id | label_source | trace_source | label |
|---|---|---|---|---|
| raw_1 | raw | exp1395_normalized_fover_ordinal_replay | unit_fover | incorrect |

## Recovered Counts By Trace Source

- `unit_fover`: 1

## Unrecovered Labels

| case_id | blocker | recovery_scope |
|---|---|---|
| absent | no_local_ordinal_replay_source_row | local_recovery_scope |
