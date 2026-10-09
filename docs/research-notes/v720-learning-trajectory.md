# V720 delayed local learning trajectory

Exp8348 reuses the exact original Exp8334 checkpoint named by Exp8346. It
requires Exp8346 frozen-head readiness and Exp8347 local-kernel readiness.
No new static fit, H1 result, generator update or current model call enters
this experiment. The cached observations remain exposed development data.

The five arms start from identical coefficients and temperature. Each of the
96 source slots issues all five predictions before the release of slot t-8.
Missing features remain escalations and do not update parameters. The final
eight labels remain pending. The shuffled control draws with replacement from
labels already released, and skips a missing due label.

Local sparse and dense updates freeze slope, intercept, temperature and knots.
The deployed-logit derivative divides the residual by temperature before the
norm cap of one. Each eligible release takes one step at rate .01, then clips
coordinates to [-4,4]. There is no online decay or search. Calibration-only
updates its intercept under the same budget. Cache dependencies use actual
active spline coordinates. An incomplete index triggers a full invalidation.

The separate constructed fit-only control uses exactly 88 updates and tests a
typed-action crossing. Source-specific starting margins and attainable logit
bounds constrain interpretation. A failed crossing limits a scientific null
to the update budget; it cannot establish that natural learning is impossible.

Primitive issues, releases, coefficients, dependency versions and pending IDs
are durable before acknowledgment. Actual worker exits73 after updates32/64
are resumed and compared with uninterrupted semantic state. A future-label
mutation changes only labels49 onward and must preserve issues through57.
An independent scalar spline recursion checks each dense/sparse coefficient
update, and a deliberately wrong coefficient must fail that check.

All retention32 features receive shadow predictions at windows0/32/64/96.
Every window seals before retention evaluator targets can open. Individual
coefficient deltas are probed on later distinct source features; these local
probability effects do not establish utility on issued decisions. Complete
update latency and serialized memory bytes are engineering measurements.

The terminal artifact and primitive evidence are located at
`results/experiment_8348_v720_continuous_local_learning.json` and
`results/raw/experiment_8348_v720_continuous_local_learning/`. Its readiness
score concerns causality, restart, accounting and checked bytes. H1 remains
reserved for Exp8350; H2 and retention benefit remain reserved for Exp8351.
Both generalization scores remain zero. V717 science and historical outcomes
remain frozen. Ops and traceability reconciliation is owned by the conductor.
