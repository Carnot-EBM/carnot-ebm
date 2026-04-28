# Deep Think Round-8 Prompt — Quantifying Multi-Verifier Rotation Defence

**Status:** Ready to send. Builds on Round-7 results
(`zenil-deep-think-round7-results.md`).
**Date drafted:** 2026-04-28
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip the header and the
"internal predictions" footer.

---

## Prompt to send (verbatim)

### Background (use as given premises)

We are deploying a verifier-filtered self-distillation loop where the
model $Q_t \to \mu_P$ via filter $w(x) \propto \exp(-E(x)/T)$, with
$E$ a constraint-derived energy (e.g., Z3 satisfaction, $E \in \{0,1\}$).

From prior independent analysis the following are established:

1. The structural restorative force is $\Phi_t = \mathrm{Cov}_{\mu_P}(g_t, E)$
   where $g_t = (Q_t - \mu_P)/(\delta_t \mu_P)$.
2. Single-verifier loops stall at $\delta_{\text{plateau}} = \delta_0 \|\Pi_{E^\perp} g_0\| / \sqrt{1 - (\varepsilon/b_1)^2}$
   where $b_1 = \langle \phi_1, E\rangle$ is the spectral overlap with
   the slowest-decaying eigenmode and $\varepsilon$ is the verifier's
   noise bound.
3. The orthogonality stall is statistically *indistinguishable* from
   adversarial specification gaming on a single Boolean verifier:
   $\mathrm{Cov}_{\mu_P}(g_t, E^m) = 0$ in both regimes (since
   $E \in \{0,1\}$ implies $E^m = E$ for all $m \geq 1$).
4. The unique deployable defence is **multi-verifier pre-emptive
   rotation**: at each round, pick $i^* = \arg\max_i \widehat\Phi_i$
   among $\{E_1, ..., E_k\}$ semantically orthogonal verifiers. Rotation
   strictly dominates ensemble combine (which creates multi-modal
   glassy frustrated MCMC landscapes — $\tau_{\text{int}}^{\text{ens}}$
   blows up factorially).

Three open questions follow. Use $L^2(\mu_P)$ throughout; make explicit
any KL ↔ $L^2$ conversions.

### Question 1: Quantifying "semantic orthogonality" between verifiers

Rotation defends against null-space mimicry attacks only when $E_1$
and $E_2$ are *sufficiently different* — an adversarial output trivial
under $E_1$ must NOT also be trivial under $E_2$.

Formalise this as a quantitative angle condition. Define the kernel-
projection operators $P_i = \mathrm{Proj}_{\mathrm{null}(\mathcal{K}_{E_i})}$
where $\mathcal{K}_{E_i}: f \mapsto E_i f - \langle E_i f, \mathbf{1}\rangle_{\mu_P}$
is the verifier covariance operator on $L^2_0(\mu_P)$. The
Friedrichs angle between the two null-spaces is
$\theta_F = \arccos \|P_1 P_2\|_{\text{op}}$.

(a) Derive the *minimum required Friedrichs angle* $\theta_F^{\min}$
    such that pre-emptive rotation provides strict contractivity from
    $\delta_t \to \delta_{\text{plateau}}^{\text{rot}} \ll \delta_{\text{plateau}}^{\text{single}}$.
    Express $\theta_F^{\min}$ in terms of: target plateau improvement
    factor, verifier noise bounds $\varepsilon_1, \varepsilon_2$,
    and the spectral gaps of $\mathcal{K}_{E_i}$.

(b) For a $k$-verifier rotator, generalise the pairwise condition to a
    joint condition. Specifically: define an analogue of the joint
    Friedrichs angle for $\{P_1, ..., P_k\}$ that bounds the maximum
    null-space overlap. What is the deployable measurement?

(c) For Carnot's specific case ($E_1$ = Z3 satisfaction, $E_2$ =
    semantic AST density / structural complexity, $E_3$ = liveness),
    are these realistic candidates *expected* to satisfy
    $\theta_F > \theta_F^{\min}$, or do you anticipate they have
    significant null-space overlap that would weaken rotation defence?

### Question 2: Computational infeasibility of $k$-verifier null-space attacks

The natural extension of the single-verifier mimicry attack is: an
adversary constructs an output $x_*$ that simultaneously lies in
$\bigcap_{i=1}^{k} \mathrm{null}(\mathcal{K}_{E_i})$. For example,
trivial code that simultaneously:
(a) type-checks ($E_1 = 0$),
(b) satisfies liveness ($E_2 = 0$),
(c) has high AST density ($E_3 = 0$).

This is — in principle — possible if such $x_*$ exists in the model's
output space. The question is *how hard*.

(a) Provide a security-style argument bounding the computational cost
    of constructing $x_* \in \bigcap_i \mathrm{null}(\mathcal{K}_{E_i})$
    in terms of:
    - The intrinsic complexity of each $E_i$ (e.g., Z3 satisfaction
      is NP-complete; liveness can be PSPACE-hard).
    - The semantic orthogonality $\theta_F$ from Q1.
    - The model's expressive capacity (parameter count, training depth).

(b) Is there a *minimum $k$* below which the rotation defence is
    breakable in polynomial-time, and above which it is exponentially
    hard? Specifically, does the security parameter scale as $2^k$,
    $\binom{k}{2}$, or some other function?

(c) For Carnot's deployment scenario (closed-weight LLM under
    self-distillation against a verifier suite), what's the practical
    minimum $k$ to defend against a frontier-model adversary?

### Question 3: Zenil's $\inf_t \alpha_t > 0$ under rotation

Zenil's Theorem 5 requires $\inf_t \alpha_t > 0$ for self-distillation
convergence. Under multi-verifier rotation with verifier-specific rates
$\alpha_t^{(i)} = \Phi_i / (T \delta_t)$:

(a) The active verifier's rate is $\alpha_t^{(i^*)} = \max_i \alpha_t^{(i)}$.
    Does $\inf_t \max_i \alpha_t^{(i)} > 0$ hold under realistic
    rotation, even when each individual $\alpha_t^{(i)} \to 0$?

(b) Derive the joint condition on the verifier suite $\{E_i\}$ and
    the residual error trajectory $g_t$ that guarantees Zenil's
    theorem holds under rotation. Specifically: what's the analogue
    of the spectral-gap condition for the *rotated process*?

(c) Is there a *worst-case adversarial residual trajectory* $g_t^*$
    against which even rotation cannot guarantee $\inf_t \max_i \alpha_t^{(i)} > 0$?
    If yes, characterise it. If no, prove convergence.

### Final integration

For Carnot's Phase 3 deployment, the practical question is: **how many
semantically-orthogonal verifiers must we ship, and how do we measure
$\theta_F$ between them at deployment time?**

Synthesise Q1, Q2, Q3 into a single deployable architectural rule of
the form:

> "For a foundation model with $P$ parameters trained on $D$ tokens,
>  Phase 3 self-distillation requires $k \geq f(P, D, \varepsilon_{\text{tgt}})$
>  semantically orthogonal verifiers with pairwise Friedrichs angle
>  $\theta_F \geq g(P, D, \varepsilon_{\text{tgt}})$ to guarantee
>  Zenil's $\inf_t \alpha_t > 0$ and rotation-defence security."

Provide $f$ and $g$ explicitly in terms of the problem parameters.

---

## Internal cross-validation predictions (DO NOT PASTE)

### Q1 prediction
- $\theta_F^{\min}$ scales as $\arcsin(\varepsilon / b_1)$ — small angle
  enough when verifiers are accurate, larger angle needed when noisy.
- For a $k$-rotator, joint condition is $\min_{i \neq j} \theta_F(P_i, P_j) \geq \theta_F^{\min}$
  (pairwise sufficient).
- Carnot's $E_1$ (Z3) and $E_2$ (AST density) likely have $\theta_F$
  large enough — Z3 only checks logical satisfiability while AST density
  measures structural complexity, fairly orthogonal semantics.

### Q2 prediction
- Computational cost of constructing $x_* \in \bigcap_i \mathrm{null}(\mathcal{K}_{E_i})$
  scales as $\prod_i (\text{cost of generating output in null}(E_i))$,
  with reduction by $\theta_F$ overlap factor. Roughly exponential in $k$
  for orthogonal verifiers.
- Minimum $k$ for cryptographic-style infeasibility: $k = 3$–$4$ for
  $\theta_F \approx \pi/3$, scaling logarithmically with required
  infeasibility level.
- Carnot needs $k \geq 3$ for frontier-LLM adversaries.

### Q3 prediction
- $\inf_t \max_i \alpha_t^{(i)} > 0$ holds *iff* there exists at least
  one verifier whose null-space doesn't contain the residual $g_t$ at
  every $t$. The rotation envelope provides this when verifier suite
  has full coverage.
- Worst-case adversarial trajectory: residual $g_t$ that simultaneously
  rotates into ALL verifiers' null-spaces. By Q2, this requires
  exponential adversarial effort — so practically this is the security
  guarantee.

### Final integration prediction
- $f(P, D, \varepsilon_{\text{tgt}}) \sim \log(P/\varepsilon_{\text{tgt}})$
  (logarithmic in model capacity).
- $g(P, D, \varepsilon_{\text{tgt}}) \sim \arcsin(\varepsilon_{\text{tgt}}/\text{verifier strength})$.
- Concrete: Phase 3 with $P \sim 10^{10}$ parameters needs $k \in [3, 5]$
  with $\theta_F \geq \pi/4$ pairwise.

## Action plan after Round-8

1. **Update CLAUDE.md decentralization rule:** "Phase 3 self-distillation
   requires multi-verifier suite with measured pairwise Friedrichs angle
   above the Round-8 threshold. Sovereign access to constraint-derived
   verifiers is part of the political infrastructure."

2. **Add Exp F to change proposal:** Friedrichs angle measurement on
   Carnot's actual verifier suite (Z3 + AST density + liveness checker).
   Empirical confirmation that $\theta_F > \theta_F^{\min}$.

3. **REQ-PHASE3-002 (new spec):** "Phase 3 self-distillation must
   maintain $k$ semantically orthogonal verifiers where $k$ and pairwise
   $\theta_F$ satisfy the Round-8 deployable rule."

4. **Memory entry update:** if Round-8 confirms predictions, this
   completes the Phase 3 architecture mathematical foundation.
