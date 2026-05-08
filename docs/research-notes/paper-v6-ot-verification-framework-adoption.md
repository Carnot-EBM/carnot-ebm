# Paper v6 OT Verification Framework Adoption

Run date: `20260508`
Source paper: arXiv:2510.18982 (https://arxiv.org/abs/2510.18982)
Requested paper source state: `absent`

This note adopts the OT verification vocabulary for paper-v6 without turning
it into a Carnot performance theorem. The safe reading is: Carnot has a
verifier cascade whose proposal support, operating ROC, and finite-K sampler
gap can be described with the framework's words.

## Coverage Mapping

- OT term: Coverage is the constraint that the verifier-induced target policy
  remain supported by the generator proposal distribution.
- Carnot mapping: Carnot's coverage is the generator proposal mass available
  to the candidate verifier cascade: local SOTA outputs, candidate warm-start
  states, and THRML/Soft-Gibbs finite-K neighborhoods.
- Boundary: Do not identify Carnot's Soft-Gibbs residual beta with the OT
  coverage budget beta; they govern different objects.

## ROC Mapping

- OT term: ROC is the verifier acceptance region relative to ground truth,
  tracked through TPR, FPR, and Youden's index J = TPR - FPR.
- Carnot mapping: Carnot's cascade has an effective ROC composed from
  thresholded deterministic validators, energy verifiers, and short-circuit
  exits.
- Boundary: AUROC on one corpus is not the same as an operating ROC for a
  deployment threshold on SOTA outputs.

## Sub-optimality Mapping

- OT term: Sub-optimality is the reward gap between the ideal
  verifier-induced target distribution and the distribution induced by a
  sampling algorithm.
- Carnot mapping: Carnot's sub-optimality is the finite-K gap between the
  ideal validated acceptance set and the outputs reachable by the
  candidate-warm-start THRML plus Soft-Gibbs cascade.
- Boundary: Finite-K sampling, finite batch size, and imperfect ROC leave
  residual gap; paper-v6 should not state zero sub-optimality.

## Conflict Ledger

These conflicts are wording constraints for paper-v6. Each one should soften
an existing or tempting claim before any publication action.

### CONFLICT-1: AND-composing more verifiers eliminates reward hacking.

- Reason: Verifier ROC and measured verifier correlation control the effective
  acceptance region; exp1256 already shows k_eff is much smaller than nominal
  k.
- Softened boundary: Paper-v6 should say the measured k=5 stack narrows the
  acceptance region, not that arbitrary k-composition eliminates gaming.

### CONFLICT-2: The finite-K sampler draws from the verifier target distribution.

- Reason: The OT framework treats algorithm-induced distributions separately
  from the target; finite-K THRML, warm-start, BRS, and Soft-Gibbs runs still
  leave sub-optimality.
- Softened boundary: Paper-v6 should describe candidate warm-start and THRML
  vendoring as finite-K implementation choices, not exact sampling from the OT
  target.

### CONFLICT-3: The Soft-Gibbs Residual coverage bound is an OT coverage theorem.

- Reason: The residual beta is an inverse-temperature on verifier failures,
  while OT coverage beta constrains proposal-policy support.
- Softened boundary: Paper-v6 should keep the Jensen acceptance bound as a
  Carnot residual result and reserve OT coverage language for proposal support.

### CONFLICT-4: High verifier AUROC implies robust deployment verification.

- Reason: Verifier ROC is distribution- and threshold-dependent; exp1100/1120
  show SOTA-output inversion and a corpus-bounded retrain fix.
- Softened boundary: Paper-v6 should report FoVer and SOTA-inclusive
  calibration as local evidence, not universal verifier dominance.

### CONFLICT-5: More sampling compute monotonically improves verified outputs.

- Reason: The OT framework splits transport, policy-improvement, and
  saturation regimes; verifier ROC and coverage determine whether extra
  samples reduce sub-optimality.
- Softened boundary: Paper-v6 should say extra samples help only in the
  measured regime and under the measured verifier ROC.

### CONFLICT-6: THRML vendoring supplies sampler security or hardware execution.

- Reason: OT verifier geometry is not a hardware-security proof, and exp1561
  falsified THRML kinetic-security parity on the zero-coupling fixture.
- Softened boundary: Paper-v6 should treat THRML as software sampler alignment
  while keeping kinetic security and Extropic hardware execution as open or
  absent.

## Patch Plan

No patch was applied because docs/papers/paper-v6/main.tex is absent. Use
`docs/arxiv-paper/main.tex` only as a later integration target if the active
paper-v6 source remains absent:

- Related Work after the current verifier-stack comparator paragraphs: cite
  arXiv:2510.18982 as vocabulary for coverage, ROC, and sub-optimality.
- Section 3 / framework near the k=5 cascade: add one paragraph that maps
  coverage to generator proposal support and ROC to the cascade's thresholded
  operating point.
- Hardware and sampling limits: add one sentence that finite-K THRML and
  Soft-Gibbs Residual runs leave nonzero sub-optimality.

This note does not trigger publication, arXiv submission, release, or push.
Paper patch applied: `false`
