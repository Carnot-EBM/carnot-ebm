# Deep Think Round-12 Prompt — Information-Theoretic Lower Bound on Self-Distillation

**Status:** Ready to send. Pairs with Round-7's *achievable* upper
bound to provide the *fundamental floor* on what verifier-filtered
self-distillation can achieve.
**Date drafted:** 2026-04-29
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip the predictions footer.

---

## Prompt to send (verbatim)

### Background (use as established premises)

Verifier-filtered self-distillation: a base model $Q_t$ is filtered
through a verifier suite $\{E_1, \ldots, E_k\}$ via $w(x) \propto \exp(-E_i(x)/T)$
to converge toward truth $\mu_P$. From a prior 11-round derivation:

1. **Achievable plateau (rotation defence):** under multi-verifier
   pre-emptive rotation with pairwise Friedrichs angle $\theta_F$ and
   verifier accuracies $\{\varepsilon_i\}$, the achievable plateau is
   $\delta_{\text{plateau}}^{\text{rot}} = \delta_0 \|\Pi_{\bigcap N_i^\perp} g_0\| / \sqrt{1 - (\bar\varepsilon / b_1)^2}$
   (modified for the rotation envelope per Round-9).

2. **Active subspace.** $V_{\text{active}} = \mathrm{span}(\bigcup_i N_i^\perp) \subseteq L^2_0(\mu_P)$
   has intrinsic dimension at most $k \cdot m$ where $m$ is the base
   verifier count.

3. **Restorative force.** $\Phi = \mathrm{Cov}_{\mu_P}(g, E)$ — verifier
   provides force only along directions where residual $g$ correlates
   with $E$.

The remaining question is the **fundamental floor**: given the
*information* the verifier suite carries about $\mu_P$, what's the
absolute minimum achievable $\delta$ — *regardless* of any clever
suite design or architecture?

Use $L^2(\mu_P)$ throughout; make explicit any KL ↔ $L^2(\mu_P)$
conversions.

### Question 1: Information content of a verifier suite

Define the *information content* of verifier suite $\mathcal{E} = \{E_1, \ldots, E_k\}$
as the mutual information between the verifier outputs and the truth
distribution:

$$
I(\mathcal{E}; \mu_P) = I(\mathbf{E}(X); X) \quad \text{where } X \sim \mu_P, \; \mathbf{E}(x) = (E_1(x), \ldots, E_k(x))
$$

(For Boolean verifiers $E_i \in \{0, 1\}$, $\mathbf{E}(X)$ ranges over
$2^k$ outcomes.)

(a) **Decomposition.** Show that $I(\mathcal{E}; \mu_P) \leq \sum_i I(E_i; \mu_P)$
   with equality iff the verifiers are mutually independent under
   $\mu_P$. For Carnot's rotation suite (Friedrichs-orthogonal $E_i$),
   how tight is this bound? Specifically: is there a quantitative
   relation between $I(\mathcal{E}; \mu_P) / \sum_i I(E_i; \mu_P)$
   and the Dixmier angle $\Theta_k$?

(b) **Information per verifier.** For Boolean $E_i$, $I(E_i; \mu_P)$
   is bounded by $\min(H(E_i), H(\mu_P))$. Express the upper bound
   in terms of the verifier's *false-positive rate* (what fraction of
   non-truth outputs $E_i$ accepts) and *false-negative rate* (what
   fraction of truth outputs $E_i$ rejects). What's the maximum
   information content for a verifier with $\varepsilon$-bounded
   accuracy?

(c) **Suite entropy.** Express the suite information $I(\mathcal{E}; \mu_P)$
   in terms of the joint covariance structure of verifier outputs
   under $\mu_P$ (i.e., the matrix $\Sigma_{\mu_P}$ analogous to the
   $\widehat\Sigma$ from Round-9's audit). Is there a clean spectral
   expression?

### Question 2: Fano-style lower bound on $\delta_{\text{plateau}}$

By Fano's inequality, the residual error after observing the verifier
suite outputs is bounded below by $H(\mu_P | \mathbf{E}) / \log|\mathcal{X}|$.
Translating this from estimation error to $L^2(\mu_P)$ distance:

(a) **Fundamental floor.** Derive a lower bound of the form
   $\delta_{\text{plateau}}^{\text{any}} \geq \mathcal{B}(I(\mathcal{E}; \mu_P), \varepsilon)$
   where $\mathcal{B}$ is a positive function vanishing iff
   $I(\mathcal{E}; \mu_P) = H(\mu_P)$ (i.e., the verifier suite carries
   *all* information about truth). This is the analog of the Cramér-
   Rao bound for verifier-filtered estimation.

(b) **Gap to achievable upper bound.** Compare the lower bound from
   (a) with the Round-7 achievable upper bound. For Carnot's typical
   Phase 3 deployment ($k=15$, Boolean verifiers, $\theta_F = \pi/4$,
   $\varepsilon = 0.05$):
   - What's $\delta_{\text{plateau}}^{\text{lower}}$?
   - What's $\delta_{\text{plateau}}^{\text{rot}}$?
   - Is the gap *closeable* by clever design (multiplicative constant
     correction), or is it *fundamental* (architecture-independent
     factor of e.g. $\log d$ or $\sqrt{k}$ separation)?

(c) **Saturating the bound.** What architectural choices achieve
   the lower bound up to constants? Specifically: does Round-9's
   rotation + AND-composition + bootstrap recipe saturate, or is
   there a (possibly more complex) architecture that does better?

### Question 3: Information-theoretic phase transitions

The Round-9 architecture has phase transitions:
- $k < k_{\text{crit}}$: polynomial-time gaming → no security.
- $k > k_{\text{crit}}$: $2^{\Omega(k)}$ gaming → exponential security.

(a) **Information-driven phase transition.** Is there a phase
   transition in $I(\mathcal{E}; \mu_P)$ that *aligns* with the
   computational phase transition? Specifically: at what $I^*$ does
   $I(\mathcal{E}; \mu_P)$ exceed $H(\mu_P) - O(1)$, making the
   recovery problem statistically tractable?

(b) **Sample complexity at the boundary.** Near the boundary
   $I \approx I^*$, how does the sample complexity required to
   *estimate* $\Phi$ scale? Specifically: does Carnot's $M^* = 192{,}000$
   audit-batch suffice arbitrarily close to the boundary, or does
   the required $M$ diverge as $|I - I^*|^{-\beta}$?

(c) **Carnot's actual operating point.** For typical Phase 3
   deployment, where does the operating point sit relative to the
   information-theoretic phase transition? Is there margin, or is
   Carnot operating near the boundary?

### Final integration: the complete picture

Synthesise the achievable upper bound (Round-7) and the information-
theoretic lower bound (this round) into a single integrated bound
of the form:

$$
\mathcal{B}(I(\mathcal{E}; \mu_P), \varepsilon) \leq \delta_{\text{plateau}}^{\text{any architecture}} \leq \delta_0 \|\Pi_{\bigcap N_i^\perp} g_0\| / \sqrt{1 - (\bar\varepsilon / b_1)^2}
$$

For Carnot's typical case, what's the *gap* between these — i.e., how
much room is there for improvement beyond the Round-9 architecture?

If the gap is fundamental (architecture-independent), this gives
Carnot a publishable theorem: *no verifier-filtered architecture can
beat the lower bound.*

If the gap is closeable, identify the specific architectural
modifications that would close it.

---

## Internal cross-validation predictions (DO NOT PASTE)

### Q1 predictions

(a) Subadditivity is tight when verifiers are MI-independent under
    $\mu_P$. For Friedrichs-orthogonal $E_i$, equality holds
    *approximately* — the relation
    $I(\mathcal{E}; \mu_P) / \sum_i I(E_i; \mu_P) = \sin^2 \Theta_k$
    seems plausible (information shrinks as verifiers become more
    correlated).

(b) For Boolean $E_i$ with false-positive rate $p$ and false-negative
    rate $q$:
    $I(E_i; \mu_P) \leq h(p_*) - p_* h(p) - (1-p_*) h(q)$
    where $p_*$ is truth's prevalence and $h$ is binary entropy.

(c) Spectral expression: $I(\mathcal{E}; \mu_P) \approx \frac{1}{2} \log \det(\Sigma_{\mu_P}^{-1} \Sigma_{\text{joint}})$
    (Gaussian approximation), tying directly to $\widehat\Sigma$.

### Q2 predictions

(a) Lower bound: $\delta_{\text{plateau}}^{\text{any}} \geq c \cdot \exp(-I(\mathcal{E}; \mu_P) / 2)$
    for some $c$ depending on the truth manifold's geometry.

(b) Carnot gap: log-factor separation. Specifically, achievable is
    $\sim \varepsilon$, lower bound is $\sim \varepsilon / \log(1/\varepsilon)$.
    *Closeable by clever design with logarithmic improvement.*

(c) Round-9 architecture saturates up to a $\log k$ factor; no clean
    architecture closes the remaining gap.

### Q3 predictions

(a) Information-theoretic phase transition at $I^* = H(\mu_P) - \log(1/\varepsilon)$.
    Below: low-information regime, recovery requires exponentially many
    samples. Above: high-information regime, polynomial samples suffice.

(b) Sample complexity diverges as $|I - I^*|^{-2}$ near the boundary
    (standard for Fano-type bounds).

(c) Carnot's typical operating point: in the high-information regime
    well above the phase transition, with $\sim 2$ bits of margin.

### Final integration prediction

The achievable-vs-lower gap is **closeable up to logarithmic factor**.
Round-9 architecture is within $O(\log k)$ of optimal. Tightening it
would require novel architecture (possibly the Stein-variational
direction).

If Round-12 confirms a fundamental floor (architecture-independent
lower bound), this is a publishable theorem strengthening the
position paper. If the gap is fully closeable, that's an open
research direction for Round-13+.

## Action plan after Round-12

1. **If Q2(a) gives a clean Fano-style bound:** add as Theorem 9 in
   the position paper (the *floor* complementing the *ceiling*).
2. **If Q3(c) shows Carnot operates near the phase transition:** the
   $M^* = 192{,}000$ audit budget needs revisiting — may be
   insufficient near the boundary.
3. **If the Round-9 architecture provably saturates:** the position
   paper can claim *optimal architecture*. If not, it claims
   *near-optimal* with explicit gap.
4. **Decentralization rule update:** the information content
   $I(\mathcal{E}; \mu_P)$ is the load-bearing sovereignty quantity —
   maintaining sovereign access to verifier *information* (not just
   verifier *components*) becomes the deepest sovereignty argument.
