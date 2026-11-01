# V745 sparse Brier precision feasibility

Exp8692 tests integer arithmetic on CPU. The primary format is signed32 total
bits with24 fractional bits. Signed16/12 and24/18 are descriptive controls.
Initial heads, scaler and temperature come from the preserved V744 protocol's
original Exp8431 checkpoint. Its producer and terminal validation are checked
separately from Exp8667, whose historical disqualification stays in force.

Before measurement, freeze256 vectors with seed7458692:128 interior points,
64 knot/clip boundaries and64 threshold searches. Geometry uses no labels.
Each head/format receives512 independent one-step updates(two binary labels
per vector), then four256-update numerical sequences. These are constructed
controls. They do not estimate generalization or continuous-learning benefit.

Integer operations cover local clamped cubic Cox recursion, residual sums,
Brier derivatives, coefficient writes and global decay. Signed64 intermediates
are range checked. Stored integers saturate explicitly with nearest ties-to-even
rounding. Decay loss and dense renormalization have separate counters. Bias
regularization and temperature remain part of the original state convention.

Scaling, logit offset, sigmoid, fallback, serialization and durable storage
remain host CPU operations. All emulation also runs on CPU. The primary gate
requires zero saturation/overflow, finite states, probability error<=0.001
at every measured update and typed-decision parity. Within the threshold error
margin, use the original float64 fallback and count its measured cost. No more
than37 local coordinates plus one global decay scalar are written per update;
any dense renormalization is counted separately.

[Hoang et al., arXiv:2602.02056v4](https://arxiv.org/html/2602.02056v4),
revised2026-06-19, motivates local support and explicit fixed-point handling.
The paper's hardware performance does not qualify this CPU diagnostic.
There is no synthesis, physical FPGA execution, deployment or acceleration
claim. No retention labels or causal stream are used; no state reaches Exp8685.

The JSON and immutable raw invocation receipts hold the measured verdict,
errors, counters, clocks, validation commands and source hashes. Fresh bounded
replay reconstructs numerical outcomes and report bytes. An independent seal
rejects missing operands and self-consistently rehashed local changes. Scoped
unit, CLI, lint, typing, spec tracing, private E2E and owned statement coverage
must qualify before readiness is released. Ops reconciliation is conductor-owned.
Preserve the V733 source-information, V735 cold-cost, V737 supervisor and V743
batch-one nulls and every stamped quarantine.
