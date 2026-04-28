# Deep Think Round-8 Results — Phase 3 Verifier Suite Must Be Redesigned

**Status:** Authoritative. Builds on Round-7 results.
**Date:** 2026-04-28 (very late)
**Prompt:** `zenil-deep-think-round8-prompt.md`

Round-8 delivered explicit deployable formulas for multi-verifier
rotation defence — and refuted Carnot's planned verifier suite design
along the way. The single most consequential finding: **Z3 + AST
density + liveness is not a safe rotation suite** because it shares a
pathological joint null space (padded vacuous dead-code defeats all
three simultaneously).

## Q1 — Quantitative orthogonality

### (a) Minimum Friedrichs angle

$$
\boxed{\theta_F^{\min} = 2\arcsin\!\left(\frac{\eta \cdot \max(\varepsilon_1, \varepsilon_2)}{\delta_{\text{plateau}}^{\text{single}} \cdot \min(b_1, b_2)}\right)}
$$

where $\eta = \delta_{\text{plateau}}^{\text{single}} / \delta_{\text{plateau}}^{\text{rot}}$
is the target plateau improvement factor.

**Carnot prediction was partial:** got the $\arcsin(\varepsilon/b)$
shape but missed the target-improvement factor $\eta$ and the
factor-of-2 bisector (Method of Alternating Projections geometry).

### (b) k-verifier generalisation: Dixmier angle

Pairwise Friedrichs angle is insufficient — multiple hyperplanes can
pairwise intersect trivially but share a multi-way joint intersection.
Correct generalisation:

$$
\sin^2 \Theta_k = 1 - \left\| \frac{1}{k}\sum_{i=1}^{k} P_i \right\|_{\text{op}, (\bigcap N_i)^\perp}
$$

**Deployable measurement:** generate large adversarial-output batch;
compute $k \times k$ empirical cross-covariance / Jaccard matrix
$\widehat\Sigma$ of binary verifier outputs; $\lambda_{\min}(\widehat\Sigma)$
is the operational proxy for $\sin^2 \Theta_k$.

**Carnot prediction was wrong:** pairwise is insufficient. Need joint
operator measurement.

### (c) Carnot's planned suite — REFUTED

**The planned $\{E_1 = \text{Z3}, E_2 = \text{AST density}, E_3 = \text{liveness}\}$
suite is NOT secure against rotation-defence attacks.**

Their null spaces share a pathological joint intersection: *vacuous,
dense, dead-code with dummy-variable mutations*. Specifically, an
adversarial output of the form

```python
def f(x):
    a = x; b = a; c = b; d = c   # AST density spike (E_2 = 0)
    while False: pass             # type-checks (E_1 = 0)
    return 0                      # liveness via dummy: E_3 = 0
```

satisfies all three constraints. The model can specification-game by
mode-collapsing into this intersection. **True $\theta_F \to 0$;
rotation defence collapses to single-verifier.**

This is a load-bearing refutation of the verifier suite design we
were going to ship in Exp E of the change proposal.

## Q2 — Computational infeasibility of k-verifier null-space attacks

### (a) Security bound

$$
W \geq \Omega\!\left( \frac{\max_i \mathcal{C}(E_i)}{P \cdot (\cos \theta_F)^k} \right)
$$

where $\mathcal{C}(E_i)$ is the worst-case constraint complexity (NP
for Z3, PSPACE for liveness) and $P$ is model capacity.

### (b) Phase transition

Security parameter scales **strictly exponentially** as $\mathcal{O}(\sec^k \theta_F)$,
NOT pairwise $\binom{k}{2}$. Continuous adversarial gradient heuristics
fail at a sharp $k_{\text{crit}}$ phase transition; above it, brute-force
$2^{\Omega(k)}$ search is required.

### (c) Practical $k$ for Carnot

For frontier model $P \sim 10^{12}$, $D \sim 10^{13}$, adversarial
search bandwidth $\sim 10^{25}$, requires $(\sec\theta_F)^k \gg 10^{25}$.
**Practical Phase 3 minimum: $k \in [5, 15]$ rigorously orthogonalized
verifiers** to cross the phase transition.

**Carnot prediction wrong by an order of magnitude:** we predicted
$k \geq 3$, Deep Think requires 5–15.

## Q3 — Zenil's $\inf_t \alpha_t > 0$ under rotation

### (a) Confirmed

$\alpha_t^{(i^*)} \geq \min(b_i) \sin(\Theta_k) / T > 0$ — the
$\delta_t$ terms cancel because the residual is normalised
($\|g_t\|_{L^2(\mu_P)} = 1$ in the local-$\chi^2$ regime). Maximum
global envelope strictly positive.

### (b) Joint condition

The single-verifier spectral gap is replaced by **Joint Verifier
Spectral Coercivity:** the pooled covariance operator
$\mathcal{G} = \frac{1}{k}\sum_{i=1}^{k} \mathcal{K}_{E_i}^* \mathcal{K}_{E_i}$
must be strictly positive-definite over the residual trajectory's
span.

Equivalent to: $\bigcap_i N_i \cap \mathrm{span}(g_t) = \{0\}$.

### (c) Worst-case adversarial trajectory

$g_t^* \in \bigcap_{i=1}^{k} N_i$ — if the verifier suite fails the
coercivity condition (as Carnot's planned suite does per Q1c), the
adversary constructs a residual trajectory entirely in the joint null
space. Then $\Phi_i = 0$ for all $i$ simultaneously, $\max_i \alpha_t^{(i)} = 0$,
**Zenil's limit fatally violated, glass-stall**.

This is *exactly why* rotation must be paired with Q2's computational
intractability — geometric coverage alone is insufficient if a joint
null exists.

## Final integration: deployable architectural rule

For Phase 3 self-distillation security:

$$
\boxed{g(P, D, \varepsilon_{\text{tgt}}) = 2\arcsin\!\left(\frac{\max_i \varepsilon_i}{\varepsilon_{\text{tgt}} \cdot \min_i b_i}\right)}
$$

$$
\boxed{f(P, D, \varepsilon_{\text{tgt}}) = \left\lceil \frac{\log(P \cdot D)}{-\log \cos(g/2)} \right\rceil}
$$

### Concrete sizing examples

| Regime | $P$ | $D$ | $\theta_F$ ($g$) | Required $k$ ($f$) |
|--------|-----|-----|------------------|----------|
| Phase 1 (current Carnot) | $10^{10}$ | $10^{12}$ | $\pi/2$ (best case) | 161 |
| Phase 1 (current Carnot) | $10^{10}$ | $10^{12}$ | $\pi/3$ (typical) | 510 |
| Phase 3 (frontier) | $10^{12}$ | $10^{13}$ | $\pi/2$ | 167 |
| Phase 3 (frontier) | $10^{12}$ | $10^{13}$ | $\pi/3$ | 530 |
| Phase 3 (frontier) | $10^{12}$ | $10^{13}$ | $\pi/4$ | 1700 |

These numbers contradict Deep Think's own Q2(c) "5 to 15" estimate.
The discrepancy: Q2(c) assumed near-maximal orthogonality between
verifiers, where $\sec\theta_F$ approaches infinity and even small $k$
suffices. The integration formula gives the conservative deployable
bound for realistic $\theta_F$.

**Conclusion: Phase 3 architecture needs HUNDREDS of orthogonalized
verifiers**, not the ~3 we'd been planning around.

## Implications for Carnot

### Architecture must change

1. **The planned Z3 + AST + liveness suite is not Phase 3-secure.**
   Padded vacuous dead-code is the existence proof. Suite must be
   redesigned around constraints whose joint null space is *provably
   trivial* — not just topologically distinct, but free of pathological
   shared modes.

2. **Verifier suite size:** real Phase 3 deployment requires hundreds
   of orthogonalized verifiers, not 3–5.

3. **Phase 1 now (Z3 verification only):** acceptable for the verify-
   repair product where adversarial gaming isn't the threat model. The
   full architecture isn't required until iterated self-distillation
   ships.

4. **Hardware mandate refined:** Phase 2 hardware acceleration enables
   *running hundreds of verifiers in parallel per round*, not just
   driving sample budget. New throughput target: $k \cdot N_t$ per
   round, $k \in [10, 100]$.

### Open questions for Round-9

- **Constraint design.** What classes of constraints have provably
  trivial joint null spaces? Is there an algebraic / topological
  classification?
- **Adaptive verifier generation.** Can verifiers be auto-generated
  from semantic specifications such that their joint null is
  guaranteed empty?
- **Orthogonality tightening.** Can $\theta_F$ between two verifiers
  be improved by post-hoc transformation (e.g., projecting one
  verifier onto the orthogonal complement of the other's eigenmodes)?
- **Recursive rotation.** Once we have $k$ verifiers, does using
  *combinations* of them (e.g., $E_i \cap E_j$) further increase
  effective $k$, or just shift the phase transition?

### Updates required

1. **CLAUDE.md decentralization rule:** Phase 3 architecture requires
   $k \geq f(P, D, \varepsilon_{\text{tgt}})$ orthogonal verifiers
   with $\theta_F \geq g(P, D, \varepsilon_{\text{tgt}})$ pairwise.
   Sovereign access to constraint-derived verifier *suites* (not
   just single verifiers) is part of decentralization rule 5.

2. **Change proposal `zenil-grounded-self-distillation-deployable-stack.md`:**
   - Exp E: rewrite around the Dixmier angle measurement.
   - **Exp F (new):** measure $\theta_F$ on Carnot's actual planned
     suite. Round-8 predicts this will be small — empirically confirm.
   - **Exp G (new):** redesign the verifier suite to avoid the
     "vacuous dense dead-code" joint null. Likely requires
     introducing semantic-grounded verifiers (formal property checks
     tied to ground-truth behavioural specifications, not just
     structural metrics).

3. **`_bmad/architecture.md`:**
   - Asymptotic Hardware Mandate: throughput target now $k \cdot N_t$
     per round.
   - Phase 3 architecture: $k \in [10, 100]$ verifiers minimum, not
     2–3.

4. **New memory entry:** `project_pathological_joint_null_space.md`
   capturing the "vacuous dense dead-code" attack and the suite-design
   constraint it imposes.

## What we got right

- Q3(a): $\inf_t \max_i \alpha_t^{(i)} > 0$ under rotation
- Q2(a) shape: exponential in $k$
- Q1(a) shape: $\arcsin(\varepsilon/b)$

## What we got wrong

- Q1(b): pairwise sufficient (refuted — Dixmier angle required)
- Q1(c): Carnot's planned suite is safe (refuted — pathological joint null)
- Q2(c): $k \in [3, 5]$ (refuted — actual is [5, 15] for the lower
  estimate, or hundreds in the conservative formula)

## What surprised us

- **The pathological joint null space.** Our planned suite has a
  trivial-but-passes-everything attack manifold we hadn't identified.
- **Pairwise insufficient.** Joint operator condition is required.
- **Hundreds of verifiers.** The conservative integration formula
  implies far more verifiers than even the security-parameter
  argument suggests.
- **Phase 2 hardware mandate is now even more load-bearing.** Running
  $k \in [100, 1000]$ verifiers per round at multiple-million-samples-
  per-second is the actual deployment target.

## Publishable contributions from Rounds 6-8

1. **Closed-form orthogonality plateau** with $\varepsilon/b_1$
   correction (Q1a Round-7).
2. **Ensemble vs rotation: rotation strictly dominates** via frustrated-
   landscape argument (Round-7 Q2c).
3. **Null-space mimicry attack on Boolean verifiers** (Round-7 final
   integration).
4. **Pathological joint null space in seemingly-orthogonal verifier
   suites** (Round-8 Q1c) — the "vacuous dense dead-code" attack.
5. **Deployable architectural rule** for $k$ and $\theta_F$ via
   Dixmier angle (Round-8 final integration).

These constitute a coherent position-paper-worthy contribution on
foundation-model self-distillation security.
