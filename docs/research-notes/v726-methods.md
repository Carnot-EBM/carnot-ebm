# V726 methods frozen before fitting — 2026-10-11

Exp8429 repairs scheduling authority for fourteen tasks, Exp8429–Exp8442.
It makes no calibration, feedback-learning, native-speed or current-model result.
The machine protocol copies the pinned V725 scientific objects verbatim.
The blocked V725 receipt records history; it is not a required success gate.

The exact 300 question hashes keep SHA256(v725: + full stored question_hash)
order: 150 fit, 50 tune and 100 test clusters, three answers per cluster.
Support floors remain 128/32/80 and eight tune/test clusters per target class.
The nine features, clipping, 109 coefficients, five seeds, losses, 200 steps,
0.01 step size, 0.001 L2 and tune-only temperature selection stay fixed.
Accept at p<=0.2, reject at p>=0.8, otherwise escalate with cost 0.2.
The static question bootstrap keeps 10000 draws and seed 7258419.

Only three pre-fit clarifications are permitted:

- Tune source permutation uses seed 7252; fit/test keep 725/7251.
- Each block first commits sealed predictions and label-blind admissions.
  After release, update selected questions in original block order and siblings
  in release order. One transaction then commits all released updates and the
  next cursor. Tentative updates replay from the sealed checkpoint; committed
  updates advance without applying labels again. Stable acknowledgment IDs
  preserve idempotent delivery after a crash.
- Retention uses the same resampled tune-question IDs in all five windows.
  Each draw takes their maximum Brier increase. Its upper 95 percent bound
  must remain <=0.01, with the inherited 10000 draws and seed 7258421.

Feedback keeps five blocks of twenty questions, capacities 8/16, four arms and
five seeds: 12000 answer rows and 4000 question/arm/seed/capacity units.
All sixty sibling predictions precede admitted labels. Audit storage cannot
feed targets into the learner. Generator weights, scaler and temperature stay
fixed. Exact restart is a separate qualification from measured benefit.

Historical V725 cost definitions remain unchanged for reference. Current
deployment tasks attribute qualified V724 complete Python clocks and compute
conditional acceleration ceilings. They do not repeat the failed native proof
or infer measured hardware acceleration. No new cost benchmark runs here.

The dated mapping is imported from the 2026-10-11 V726 research scan.
HalluScoring's unseen-question and unseen-generator axes are a future Arabic
corpus lead. They cannot change this split or establish current Qwen skill.
OpenReview challenges, inaccessible citation lists and stale trending snapshots
remain retrieval limits. Public labels are exposed; older overlap is unknown.
Independent generalization and generalized learning benefit stay zero.
V717 H1=-0.00390625 and H2=0 remain closed. Missing V722 history is unresolved
on its separate branch. Optional supplied CUDA repair affects only its canary.

Before fitting, private representative zero decision-loss and nonzero controls
exercise unchanged publication consumers and validators. Unsupported findings
are retained as compatibility evidence. They cannot change the metric, validator
or contract readiness. The field-by-field delta is in the machine protocol.
