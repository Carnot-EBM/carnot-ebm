# V730 objective learning admission record

Run date: 20261013. Task: `exp8490-brier-online-learning`.

The four-arm study could not admit the Exp8489 update adapter. Its primary
reports `objective_kernel_ready_score=1`, `required_checks_passed=true`, and
`flagged_adversarial=true`. Its shipped cold replay returns exit1. Its bound
publication consumer rejects the current primary with `stale_primary_hash`.
These observations block measurement before any real-label update.

The inspected primary hash is
`sha256:f306c910fd8fe95b12ca2505655347c73846c63d7a46700f433f7b67d9384bad`.
The producer's publication sidecar still binds
`sha256:e9145dae2143a2563f7c088db80dbcbf1178fd37618789b2f346e0d0dfdb6359`.
The terminal artifact retains the exact failed operands and subprocess logs.
Exp8489 bytes, validators, thresholds and generator weights remain preserved.

The record contains all6000 intended answer/arm/seed identities from V726.
Each metric is missing with an explicit reason. It also retains15000 intended
retention rows: five windows,50 tune questions,three siblings,five seeds and
four arms. There are no completed measurements, parameter updates or admissions.
Unknown pending capacity, lost feedback and update cost remain null.

The numerical study remains frozen: identical original full spline starting
states,scalers and temperatures; seeds17/29/43/71/101; five20-question blocks;
capacity8; uniform admission before labels; rate0.01 and L2=0.001. The proposed
arms are frozen, inherited log loss, Brier, and Brier with shuffled current
released labels. The Brier derivative is `2*(p-y)*p*(1-p)/T`.
No gradient or recovery qualification is claimed by this stopped study.

This block provides no evidence against objective-aligned learning. Objective
mismatch remains a testable hypothesis. Future Exp8491 output is a dependency,
not an observation. Both generalization scores and independent sample count
remain zero. Optional hardware cannot block this CPU work.

Preserve V726 exposed-history results. V728 journal equivalence remains qualified.
Its complete-service speedup is1.0447833367 with
CI95[0.9942346768,1.1095895448]; that null remains unchanged.
V717 H1=-0.00390625 and H2=0 remain closed.

Reopening requires clean Exp8489 terminal bytes, a matching publication seal and
successful shipped primitive replay. Then run the unchanged full causal study,
including seed17 exits after block2, exact resume, wrong-rule rejection and
read-only retention evaluation. Clearing an input flag alone cannot establish
stream readiness. The current runner refuses that changed condition.

Validation uses traced scoped tests, genuine Coverage.py subprocess shards,
unchanged E2E-018 and E2E-021 assertions, Ruff, strict mypy, spec coverage,
private CLI controls and unchanged publication/adversarial/strict-row consumers.
Commands, exits, durations and log hashes live under
`results/raw/experiment_8490_v730_brier_online_learning/`.
The conductor owns ops and traceability reconciliation.
