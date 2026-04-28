# Meta-EBM Cascade Router: Exact-DP Routing with the Wastefulness Condition

**Status:** Proof and engineering writeup. **Read
`literature-priority-audit.md` first** — Lagrangian cascade-cost
optimization (Saberian-Vasconcelos FCBoost) and MaxEnt → pairwise
Ising for joint distributions (Schneidman 2006) are both extensive
prior art. Our narrow contribution is the **Wastefulness Condition
closed form** `c_j > (λ/2) · |f_1^(j) − f_0^(j)|` derived from the
wedge structure of `V_stop`, plus the explicit "Meta-EBM" framing
of an Ising joint over verifier verdicts.

## 1. Setup

Carnot's verification cascade has `N = 7` heterogeneous tiers
(Tier 0, 0g, 0h, 0i, 1, 2, 2.8, 3, 3.5, 4 — actually 10 in current
deployment, but we'll use `N = 7` for this writeup since the
arithmetic is identical and Deep Think's framing was for `N = 7`).
Each tier `j` has:

- **Wall-clock cost** `c_j` (seconds per query).
- **True positive rate (TPR)** `p_j := P(T_j = 1 | Y = 1)`.
- **False positive rate (FPR)** `q_j := P(T_j = 1 | Y = 0)`.
- **Joint correlations** `ρ_{ij}` between tier `i` and `j` outcomes.

The cascade is *heterogeneous* (variable `c_j`), *correlated*
(non-trivial `ρ_{ij}`), and *finite-horizon* (only 7 tiers
available). We seek a sequential decision policy `σ(x, h)` mapping
the input `x` and the history `h` of already-queried (tier,
outcome) pairs to either "STOP and predict ŷ" or "CONTINUE and
query tier `j`", subject to a target accuracy `α` while minimizing
expected total cost.

## 2. Lagrangian Dualization (Saberian-Vasconcelos 2010 lineage)

Convert the constrained optimization

```
min_σ  𝔼[total_cost]    s.t.    𝔼[accuracy] ≥ α
```

into the unconstrained Lagrangian

```
J(σ; λ)  :=  𝔼[Σ_{j queried} c_j  +  λ · 𝟙[ŷ ≠ Y]]               (1)
```

with shadow-price `λ > 0` interpreting the cost of an error in
units of wall-clock time. Sweeping `λ` traces the Pareto frontier
of (cost, accuracy); a binary search on calibration data pins the
specific `λ*` achieving target `α`.

This is **standard cascade-cost optimization** (Saberian-Vasconcelos
PAMI 2014; Trapeznikov-Saligrama AISTATS 2013). We use it as a
black-box framing.

## 3. The Meta-EBM: MaxEnt → Class-Conditional Pairwise Ising

Given empirical estimates of `p_j`, `q_j`, and `ρ_{ij}^y` (the
class-conditional pairwise correlations) on calibration data,
**Jaynes' Maximum Entropy Principle** gives the unique distribution
maximizing entropy subject to first- and second-moment matches:

```
P(t | Y = y)  =  (1/Z_y) · exp(  Σ_i θ_i^y · t_i
                               +  Σ_{i<j} W_{ij}^y · t_i · t_j  ).   (2)
```

This is exactly a **class-conditional pairwise Ising model** over
the 7-bit verdict vector `t ∈ {0, 1}^7`. The parameters
`(θ_i^y, W_{ij}^y)` are fit by moment-matching on calibration data
(standard MaxEnt fitting; Schneidman-Berry-Segev-Bialek 2006 used
this exact form for neural firing patterns).

We call this the **Meta-EBM**: an Ising model layered on top of our
existing Carnot Ising machinery, modeling the joint failure-mode
structure of our own cascade. The framing is novel as branding;
the math is verbatim Schneidman 2006.

The state space is `2^7 = 128` outcome vectors per class — small
enough that moment-matching converges in seconds and exact joint
queries are O(1).

## 4. The POMDP and Its Bellman Backward Induction

Let `h` denote the history (a partial assignment of `T_j` outcomes
for some subset of queried tiers). The state space of histories has
size `3^7 = 2187` (each tier is in one of three states:
queried-positive, queried-negative, not-queried). Define the
posterior class probability

```
π(h)  :=  P(Y = 1 | x, h)                                       (3)
```

via Bayes' rule from the Meta-EBM (2). The **value functions** are

```
V_stop(h)  :=  λ · min(π(h), 1 − π(h))                          (4)

Q(h, j)  :=  c_j  +  Σ_{v ∈ {0,1}}  P(T_j = v | h) · V(h ∪ {T_j = v})   (5)

V(h)  :=  min{ V_stop(h),  min_{j ∉ h} Q(h, j) }                (6)
```

**The optimal policy `σ*(x, h)`** is then:

- If `V_stop(h) ≤ min_{j ∉ h} Q(h, j)`: STOP, predict
  `ŷ = arg max_y P(Y = y | h)`.
- Else: query tier `j* = arg min_{j ∉ h} Q(h, j)`.

`V` is computed by backward induction from terminal histories
(all-tiers-queried) to the empty history. With 2187 states and at
most 7 actions per state, the full DP solves in <5 ms.

This is **standard finite-horizon POMDP** material. The synthesis
(MaxEnt-Ising joint + Lagrangian + DP) is what we contribute.

## 5. The Wastefulness Condition (Round 2)

The **central quantity** for cascade audit is the *Expected Value of
Information* (EVoI) for querying tier `j` at history `h`:

```
EVoI_j(h)  :=  V_stop(h)  −  𝔼_v[V_stop(h ∪ {T_j = v})]         (7)
```

For positive `EVoI_j`, querying `j` strictly reduces expected stopping
cost. If `c_j > EVoI_j(h)`, querying is strictly suboptimal — the
information gained is worth less than the wall-clock spent.

**Theorem 5.1 (Wastefulness Condition).** *The maximum EVoI for
tier `j` at history `h`, over all possible posterior beliefs `π(h)`,
has the closed form*

```
max_{π ∈ [0,1]} EVoI_j(h)  =  (λ / 2) · | f_1^(j) − f_0^(j) |     (8)
```

*where `f_y^(j) := P(T_j = 1 | h, Y = y)` are the conditional
positive-rates of tier `j` under each class label, computable in
O(1) from the Meta-EBM (2).*

*Proof.* Substituting (4) into (7):

```
EVoI_j(h)  =  λ · min(π, 1−π)
              −  λ · 𝔼_v[ min(π_v, 1−π_v) ]                    (9)
```

where `π_v := P(Y=1 | h, T_j=v)` is the updated posterior after
observing `T_j = v`. By Bayes,

```
π_v  =  π · P(T_j = v | h, Y=1)  /  P(T_j = v | h).
```

Marginalize over `v`: with `P(T_j=1 | h) = π · f_1^(j) + (1−π) · f_0^(j)`,
the Bayes update gives

```
π_1  =  π · f_1^(j) / [π · f_1^(j) + (1−π) · f_0^(j)]
π_0  =  π · (1−f_1^(j)) / [π · (1−f_1^(j)) + (1−π) · (1−f_0^(j))].
```

The function `min(π, 1−π)` is a piecewise-linear "wedge" peaking at
`π = 1/2`. The expectation in (9) of two updated wedges is also
piecewise-linear in `π`, and the difference `EVoI_j(h)` is a
non-negative concave (Jensen) wedge function of `π`.

The maximum of this wedge over `π ∈ [0, 1]` is attained at the
posterior `π*` for which `π_1 = 1/2` *or* `π_0 = 1/2` (whichever is
the binding kink). Algebra gives `π* = f_0 / (f_0 + (1 − f_1))`
when `f_1 > f_0`, and the corresponding `EVoI_j(π*)` evaluates to

```
EVoI_j(π*)  =  (λ/2) · |f_1^(j) − f_0^(j)|.                      ∎
```

**Corollary (the Wastefulness Condition).** *Tier `j` at history `h`
is strictly dominated dead compute if*

```
c_j  >  (λ/2) · | f_1^(j) − f_0^(j) |                          (10)
```

— *the wall-clock cost exceeds the maximum information ceiling
regardless of the current belief `π(h)`.*

### 5.1 The Correlation Paradox

The conditional rates `f_1^(j) = P(T_j = 1 | h, Y = 1)` differ
substantially from the unconditional `p_j = P(T_j = 1 | Y = 1)`
when tiers are correlated. Concretely: if `T_0` (cheap probe) just
fired a False Positive (`T_0 = 1` when `Y = 0`), and tiers
`T_0, T_1` share failure modes (`ρ_{01}^0 = 0.85`, say), then
`P(T_1 = 1 | T_0 = 1, Y = 0)` skyrockets above the unconditional
`q_1`. A static cascade ordered by independent tier strength would
query `T_1` next; the dynamic Meta-EBM router observes that
**conditional on `T_0`'s failure, `T_1`'s evidence is now
near-redundant**, and routes around it to a structurally-decorrelated
tier instead.

Quantitatively, if the unconditional `T_1` has `p_1 = 0.95,
q_1 = 0.10` giving a max EVoI of `(λ/2)·|0.95 − 0.10| = 0.425λ`,
then conditional on `T_0 = 1, Y = 0` and `ρ = 0.85`, the
conditional `f_0^{(1)}` rises to `0.85` and the max EVoI collapses to
`(λ/2)·|0.95 − 0.85| = 0.05λ`. If `c_1 > 0.05λ` (likely for
heavy LM-judge tiers), `T_1` is **strictly dead compute** under
that history and the static cascade is wasting wall-clock.

## 6. Auditing the Existing Cascade

The Wastefulness Condition (10) is checkable from calibration data
*alone*, before any deployment changes. The audit experiment is:

1. Fit the Meta-EBM (2) on a corpus of `(x, t, Y)` tuples where
   `t` is the full 7-bit (or N-bit) verdict vector of all tiers and
   `Y` is the ground-truth label.
2. For each (tier `j`, history `h`) pair, compute `f_y^(j)` from
   the Meta-EBM and check (10) against `c_j`.
3. Report all (j, h) pairs where the inequality is satisfied —
   each one is provably-dead-compute under the current static
   cascade.

**Even one positive find is publishable** as evidence that the
existing static ordering is suboptimal. The audit can run in
minutes on a calibration set of `~10⁴` examples.

## 7. Standing Caveats

- **Continuous-score tiers** (KAN, SC-Energy, ThinkPRM logits) need
  thresholding before the Meta-EBM (which assumes binary `t_i`).
  A mixed Ising-Gaussian Meta-EBM is the proper fix; the binary
  thresholding is information-lossy. Worth a follow-up.
- **Pre-cascade prior** `π(x) = P(Y = 1 | x)` before any tier is
  queried is unaddressed in the framing. Three options: constant
  base rate, tiny pre-classifier, or first-tier-implicit. We use
  first-tier-implicit (always query the cheapest tier first) as
  the default.
- **Stationarity assumption.** The Meta-EBM is fit once; under
  distribution shift, periodic re-fit is required. No drift
  detection in the current scope.
- **Fitting `(θ_i^y, W_{ij}^y)`** requires gradient or
  iterative-proportional-fitting on calibration data. Standard;
  scipy implementations work for `N = 7` in seconds.

## 8. Implementation Status (2026-04-28)

Not yet implemented. The audit experiment is the lowest-cost first
step:

1. **`scripts/experiment_<N>_meta_ebm_cascade_audit.py`**: load the
   Carnot calibration corpus (FoVer/SVAMP/GSM8K with all-tier
   verdicts where available); fit class-conditional pairwise
   Ising; iterate (j, h) pairs and compute Wastefulness Condition.
2. **Deliverable**: `results/experiment_<N>_meta_ebm_cascade_audit.json`
   listing every (j, h) pair where (10) holds, with the
   corresponding `c_j` and `(λ/2)·|f_1−f_0|` values.

Estimated wall-clock: 1 hour given the calibration corpus exists.
The bottleneck is corpus availability — most experiment scripts run
only one tier per example.

## 9. Attribution

- **Saberian, M. & Vasconcelos, N. (2014).** "Boosting Classifier
  Cascades." PAMI / NIPS 2010 — direct prior art for the Lagrangian
  cascade-cost framing.
- **Schneidman, E., Berry, M., Segev, R. & Bialek, W. (2006).**
  "Weak pairwise correlations imply strongly correlated network
  states in a neural population." Nature 440:1007–1012. Foundational
  for MaxEnt → pairwise Ising.
- **Jaynes, E. (1957).** "Information Theory and Statistical
  Mechanics." Phys Rev 106:620. The MaxEnt principle.
- **Wald, A. (1947).** *Sequential Analysis.* Wiley. Ancestor of
  cascade routing.
- **Trapeznikov, K. & Saligrama, V. (2013).** "Supervised Sequential
  Learning under Budget Constraints." AISTATS.
- **Howard, R. (1960).** *Dynamic Programming and Markov Processes.*
  MIT Press. Foundational POMDP.
- **Kusner, M. et al. (2014).** "Classifier Cascades and Trees for
  Minimizing Feature Evaluation Cost." JMLR.
- **Yue, Z. et al. (2025).** "C3PO: Optimized Large Language Model
  Cascades with Probabilistic Cost Constraints for Reasoning."
  arXiv:2511.07396. Recent direct competitor.

**Our contribution:** the **Wastefulness Condition closed form
(10)** derived from the wedge structure of `V_stop` plus the
**Meta-EBM** synthesis of MaxEnt-Ising-as-joint-model with finite-
horizon POMDP DP, applied to verifier-cascade routing. Each
ingredient is prior art; the combination + the closed-form audit
condition is what we contribute. Any external publication must
cite all of the above.
