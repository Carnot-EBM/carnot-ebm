# Audit findings ledger

Flagged audit verdicts someone must answer (REQ-OPS-AUDIT-LEDGER-1).
Rows are append-only: never rewrite or remove one. To close a finding,
edit its Disposition cell to ACCEPTED, FIXED, or WONTFIX and add a note.
OPEN rows older than 7 days escalate to ops/conductor-log.md weekly.

| First seen | Audit | Artifact | Verdict | Disposition | Note |
|---|---|---|---|---|---|
| 2026-08-22 | experiment_claim_audit | experiment_6478_identifiable_held_exact_energy_selection.json | CLAIM_OVERSTATED | OPEN | |
| 2026-08-22 | experiment_claim_audit | experiment_6497_factor_pool_support_stress.json | CLAIM_OVERSTATED | OPEN | |
