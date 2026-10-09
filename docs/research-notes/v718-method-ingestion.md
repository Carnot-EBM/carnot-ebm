# V718 method ingestion — 2026-10-08

Exp8318 read the dated V718 reference scan and saved full primary methods.
`openspec/change-proposals/v718-methods-manifest.json` binds their bytes, the
unchanged V717 scientific protocol and the separate capacity design.

The [online KAN paper](https://arxiv.org/html/2602.02056v4), Sections 3–5 and
Appendix B, uses compact B-spline support: at most degree+1 active coefficients
per edge. Its gradient bounds rely on nonnegative basis values and partition
of unity. Its hardware implementation keeps coefficient state on chip and uses
lookup tables, fixed-point saturation and a cached forward context. Parameter
storage still grows with grid size. Published latency excludes Carnot's durable
CPU transaction and is not a measurement of its KV260 fabric.

Local adaptation: retain V717's cubic knots, 34 parameters, fixed online
intercept/slope/temperature, step .01, gradient cap1 and projection. Independently
check full and sparse coefficient arithmetic and charge serialization, fsync,
restart and invalidation. Sparse updates alone cannot establish retained utility.
Deferred: new grids, quantization, an FPGA learner and generator weight updates.

The [capacity paper](https://arxiv.org/html/2606.11711v1), Sections 2–5, assumes
an oblivious loss/delay sequence, a bounded convex domain and bounded gradients;
strong convexity and bandit guarantees add assumptions. Its scheduler admits
before due feedback arrives. Semi-clairvoyant expiration acknowledgments, proxy
delay distributions, inverse observation weights and delayed weighted FTRL
are essential to its analysis. Saturation adds a separate regret penalty.

Local adaptation: three fixed 512-event traces, capacities4/16/64, unlimited,
first-admit and independent-priority retention, seeds11/22/33, permanent dropped
feedback, issue-before-release and recovery at128/256. Drain after512 without
scoring new predictions. Use the frozen basis and unweighted SGD, without
claiming the paper's regret bounds. Seed repetitions are not independent sources.
Deferred: proxy scheduling, propensity corrections and DW-FTRL reproduction.

H1/H2, source roles, labels, thresholds and scientific seeds remain unchanged.
The cached data remain exposed development. Both generalization scores stay
zero. Historical failure receipts are preserved, including the exact-zero info
finding and the capstone's original reduction drift.

Replay diagnosis: the original capstone reduced its history map in insertion
order, then wrote JSON with sorted keys. The first differing cold field is
`gate_check_summary[15].hash`. Restoring original key order reproduces all
original reduced fields. New reduction freezes serialized operands first.
Private copies retain distinct path identities even when their bytes match;
an explicit reversible relocation map preserves every compared path field.
This qualifies a historical reader and preserves the original failed primary.
