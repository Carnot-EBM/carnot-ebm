# V733 incremental source audit — 2026-10-15

Exp8529 evaluates the exact sealed Exp8528 predictions. It makes no model calls.
The fixed release has100 test questions,300 sibling answers and7500 arm/seed rows.
Repeated seeds and sibling answers do not add independent units.

The audit reconstructs Brier and typed losses from answer rows for raw_p2,
legacy_full_source, brier_scalar, brier_full and brier_source_permuted.
It averages seeds and siblings within each question before uncertainty estimation.
The original V732 primary_contrasts registration remains unchanged.
It uses10000 paired question draws at seed7318502 and98.3333333333 percent
two-sided Bonferroni intervals for95 percent familywise confidence.

| Comparator against brier_full | Brier gain | Adjusted interval | Full minus comparator typed cost |
| --- | ---: | --- | ---: |
| raw_p2 | 0.0000629994 | [-0.0002859211,0.0003961570] | -0.0046666667 |
| brier_scalar | 0.0000519859 | [-0.0002692836,0.0003605300] | -0.0033333333 |
| brier_source_permuted | 0.0001539291 | [-0.0002128295,0.0005742852] | -0.0006666667 |

Every registered Brier lower bound is below zero. The registered incremental
information claim fails, although typed cost does not increase versus raw or scalar.
A favorable comparison with the legacy head cannot replace these strong controls.
Legacy full-source losses and A2 sensitivity remain separate descriptive results.
All trainable-arm alphas are1. No treatment/control pair has identical predictions.
The artifact retains risk/coverage, fixed-bin calibration, class support, missing
worst-case bounds, clipping diagnostics and every question's arm losses and gains.

This is exposed historical evidence. Both generalization scores remain0.
Audit readiness qualifies the measurement and validation, separately from benefit.
The terminal JSON and bound validation sidecar record the final qualification.

Unchanged source tuning is closed after this registered null. Reopening requires
a registered changed information source or independently adjudicated disjoint
questions. A new split of this exposed release cannot meet that condition.

The runner reuses the shipped resource checks, typed costs, prediction sealing,
bounded child receipts, exact-byte publication and downstream consumers.
Original source-reader assertions run in disjoint shards with original authority
and dates. Private CLI checks cover missing inputs, owned errors, cold replay
and self-consistently rehashed tampering. Coverage applies only to newly owned code.
The conductor owns the separate status, changelog and traceability reconciliation.

Run:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python:. PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/experiments/experiment_8529_v733_incremental_source_audit.py --date 20261015
```
