# Causal Brier stream

This invocation trains only the small spline energy head from the qualified
original head, scaler and temperature. The frozen generator is never loaded or
updated. The fixed comparison uses frozen, log loss, Brier and released-label
shuffle controls at seeds17,29,43,71,101, with five blocks of20 questions and
three siblings:6000 predictions. Capacity stays at eight admitted questions per
block, learning rate0.01 and L2=0.001.

Every block seals predictions and target-blind admission before the evaluator
releases only admitted labels. Sparse updates retain question/sibling order.
The durable store atomically commits coefficients, applied feedback identifiers,
acknowledgments and cursor. The shuffle sees only released admitted labels.
Actual process exit73 after block2 in each seed17 arm tests exact recovery.
Fifty separate retention questions are evaluated at five windows; their labels
never enter an update. Inclusive worker clocks cover prediction, feedback,
update, durable storage and final drain, including evaluator and audit work.

Readiness qualifies complete causal histories independently of favorable loss
changes. The independent benefit audit is a future dependency. Historical
exposure makes this conditional evidence; independent generalization and
generalized benefit remain zero. Calibeating and finite delayed feedback
capacity motivate the controls without transferring a theorem to this learner.
Exp8529 source null, Exp8553 label rejection, Exp8559 cold-cost null, Exp8568's
original validator-revision rejection and all prior quarantines remain intact.

The primary artifact and its hash-bound terminal sidecar retain exact input,
code, primitive, validation and consumer receipts. Private CLI checks and fresh
cold replay reject missing inputs and self-consistently rehashed mutations.
Only newly owned statements are subject to the100 percent coverage gate; the
original numerical/storage implementation and historical test authority stay
unchanged. The conductor owns ops/status/changelog/traceability reconciliation.
