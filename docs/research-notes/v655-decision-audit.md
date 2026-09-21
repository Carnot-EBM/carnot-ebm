# V655 decision audit

The independent reducer reads the four V655 producer artifacts, their raw
native-readout shards, the frozen numeric checkpoints, the delayed-feedback
ledger, and the retention rows. It makes no current model or hardware call.

The source-fit and source-evaluation captures are valid terminal null inputs:
both used native zero-generation readouts. The typed-calibration producer is a
valid positive decision-cost input but a probability-calibration null on the
external cohort. The continuous-learning producer is a valid null: the
importance-anchored learner improves over the frozen and affine baselines, but
does not beat the unanchored residual learner under the registered familywise
criterion. A valid null therefore remains eligible for this audit.

The audit reconstructs stable option IDs in both prompt orders, lexical feature
views, checkpoint scores, group-first Brier and log loss, delayed state chains,
service-cost arithmetic, and retention use. Seven private corruptions cover a
swapped option ID, duplicated source group, future-label update, missing row,
changed checkpoint, fabricated speedup, and oracle-flag mismatch. Each must be
rejected by its named check before `audit_complete_score` can be one.
