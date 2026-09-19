# V651 method ingestion

## Assumptions

V650 is frozen. Its static and online calibration results are valid nulls.
The extraction branch is blocked. The ARC callback receipt is disqualified.
This ingestion cannot change those outcomes or gate an independent branch.

The selected papers were already indexed in the dated V651 planning review.
This note records focused ingestion. It does not claim a new discovery.

## Implementation hooks

- Spline locality maps to sparse coefficient updates in Exp7425. The control
  evaluates the dense form on the same fixed basis.
- Partial feedback maps to randomized label audits in Exp7427. Each revealed
  label needs a recorded reveal probability.
- Claim attribution maps to source IDs, evidence spans, coverage, qualifier
  retention, and contradiction checks in Exp7423 and Exp7430.

## Controls

An additive spline energy logit is logistic regression on the same fixed
basis. Both compute `beta dot phi(x)`. Spline locality changes which
coefficients receive an update. It does not create a different prediction
class when the basis stays fixed.

The spline experiment must compare sparse and dense evaluation on the same
basis. It must also include same-information logistic and shipped Gibbs
controls. Host parity cannot reproduce an FPGA speed result.

The feedback experiment must compare randomized audits, selected-only
feedback, and equal-budget uniform audits. Carnot does not inherit the source
paper's false-discovery guarantee.

The attribution experiment must keep human labels outside predictor inputs.
Extraction scores and database execution are not independent truth.

## Deferred ideas

V651 defers unchanged proof memory, external-text reranking, generic Ising
sweeps, and foundation-model training. These directions need new evidence or
a separate scope before they can return.

The G1-G4 publication definitions remain unchanged. This work does not
authorize publication, rollout, or generator weight updates.
