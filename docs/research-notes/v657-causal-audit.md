# V657 causal audit

Run date: 20260922
Verdict: `complete_null_v657_causal_audit_benefit_gate_failed` (`null`)

This aggregation audited Exp7506 and Exp7509 without importing the producer reducer.
It loaded no model and made no current model calls.

## Outcomes

- Complete accounting: 1
- Causal rows qualified: 1
- Qualified online benefit: 0
- Chronology violations: 0
- Primary support passed: False
- Five primary contrasts passed: False
- Final retention passed: True

The audit found a valid null. Several seeds lacked the frozen minimum of
12 labels in nontrivially permutable release batches. The local Brier
effect also missed the registered -0.01 floor and not all controls lost.
Retention passed. These facts do not repair the failed benefit gate.

Six private mutations covered future labels, cross-batch permutation,
prediction rewrites, pending-update loss, seed filtering, and retention rollback.
