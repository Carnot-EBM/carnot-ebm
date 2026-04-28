# Deep Think Round-9 Prompt — Designing a Phase 3-Secure Verifier Suite

**Status:** Ready to send. Builds on Round-8 results
(`zenil-deep-think-round8-results.md`).
**Date drafted:** 2026-04-28
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip header and predictions.

---

## Prompt to send (verbatim)

### Background (use as established premises)

We are designing a multi-verifier suite for Phase 3 self-distillation
security. From prior independent analyses, the following are
established:

1. **Single-verifier dynamics.** Verifier-filtered self-distillation
   stalls at $\delta_{\text{plateau}} = \delta_0 \|\Pi_{E^\perp} g_0\| / \sqrt{1 - (\varepsilon/b_1)^2}$.
2. **Rotation defence.** Multi-verifier pre-emptive rotation strictly
   dominates ensemble combine; rotation is the unique mathematically
   rigorous defence against null-space mimicry attacks on Boolean
   verifiers.
3. **The pathological-joint-null result.** The planned suite
   $\{E_1 = \text{Z3 satisfaction}, E_2 = \text{AST density}, E_3 = \text{liveness}\}$
   is NOT secure: padded vacuous dead-code with dummy mutations
   satisfies all three. The joint null space contains the
   "trivial-but-passes-everything" attack manifold.
4. **Deployable rule.** For frontier model $P \sim 10^{12}$,
   $D \sim 10^{13}$, with realistic $\theta_F = \pi/4$, security
   requires roughly $k \in [10, 100]$ rigorously orthogonalized
   verifiers with quantitative Dixmier angle bound:
   $\Theta_k$ via $\sin^2 \Theta_k = 1 - \|\frac{1}{k}\sum_i P_i\|_{\text{op}, (\bigcap N_i)^\perp}$.

The remaining gap is constructive: *how do we actually design a
verifier suite whose joint null space is provably trivial?*

Use $L^2(\mu_P)$ throughout; explicit KL ↔ $L^2$ conversions where used.

### Question 1: Algebraic / topological classification of secure verifier suites

For a verifier suite $\{E_1, ..., E_k\}$ on output space $\mathcal{X}$
with truth distribution $\mu_P$, derive a structural classification
of when the joint null space $\bigcap_i \mathrm{null}(\mathcal{K}_{E_i})$
is *provably trivial in the support of $\mu_P$* (i.e., contains no
$\mu_P$-positive-measure trivial-attack manifold).

Specifically:

(a) Identify a *small set of axioms* on each $E_i$ that, if all
    satisfied jointly, guarantees the joint null is trivial. Candidate
    axioms to consider (but not limit to):
    - **Generic Position:** each $E_i$'s level sets are in generic
      position relative to the others.
    - **Semantic Grounding:** at least one $E_i$ depends on the
      input/intent, not just the output's structure.
    - **Continuous Diversity:** the $E_i$ are continuous functionals
      with different curvature/slope at the truth manifold.
    - **Information-Theoretic Separation:** mutual information
      $I(E_i; E_j | \mu_P)$ is bounded below by some quantity.

(b) **The Pathological Joint Null Theorem:** is there a *necessary*
    structural condition on the suite (analogous to Q1c's negative
    result) that *any* verifier suite must satisfy to have non-trivial
    joint null? If yes, this gives a checklist for ruling out broken
    suites.

(c) Concrete verifier classes to evaluate against the criteria:
    - **Type-system + Z3 SMT:** structural, kernel includes vacuous code.
    - **Property-based test outcomes** (e.g., QuickCheck-style):
      semantic, kernel includes code that passes random tests but
      fails edge cases.
    - **Differential testing against reference implementation:**
      semantic, kernel includes code matching reference on observed
      inputs but failing on unobserved.
    - **Information-theoretic complexity** (Kolmogorov / BDM proxy):
      structural, kernel includes high-complexity-but-wrong outputs.
    - **LLM-judge alignment** (GPT-4 or similar judging output
      against intent): semantic, kernel includes outputs that fool
      the judge.
    - **Static specification checkers** (pre/post conditions,
      dependent types): semantic-structural hybrid, kernel depends
      on spec coverage.

    For each pair, compute (or bound) the Friedrichs angle
    $\theta_F$ between their kernel projections. Identify which
    *pairs* are sufficiently transverse to safely rotate between.

### Question 2: Constructive recipe for a Phase 3-secure suite

Given the classification from Q1, derive a concrete construction
recipe. Specifically:

(a) **Minimum spanning suite.** What's the smallest $k$ for which a
    suite of types from Q1(c) achieves $\theta_F^{\min}$ pairwise?
    Provide a concrete recipe: "Suite $S_3 = \{E_{\text{Z3}}, E_{\text{property}}, E_{\text{differential}}\}$
    achieves $\theta_F \geq \pi/4$ pairwise on the Carnot
    code-repair domain."

(b) **Adaptive verifier auto-generation.** Given a target task with
    semantic specification $\sigma$ (formal pre/post conditions,
    behavioural invariants, type signatures), can we *automatically*
    generate orthogonal verifiers $E_1(\sigma), ..., E_k(\sigma)$
    whose joint null is provably trivial? If yes, what's the
    auto-generation procedure?

(c) **Recursive verifier composition.** If we have base verifiers
    $E_1, ..., E_m$, do *combinations* (e.g., $E_i \wedge E_j$ via
    indicator product, or $E_i + E_j$ via summation) increase the
    effective $k$ for rotation defence, or just shift the phase
    transition? Provide the algebra.

### Question 3: Sample complexity of deployment-time orthogonality measurement

In deployment, we measure the empirical $k \times k$ Jaccard /
cross-covariance matrix $\widehat\Sigma$ of binary verifier outputs
on adversarial generations. $\lambda_{\min}(\widehat\Sigma)$ is the
proxy for $\sin^2 \Theta_k$.

(a) **Sample-complexity bound.** Given a target precision $\eta$ for
    $\widehat\Theta_k$ vs the population $\Theta_k$, how many
    adversarial samples $M$ are required? Provide explicit bound in
    terms of $k$, the verifier accuracy $\varepsilon$, and the
    desired confidence $1 - \beta$.

(b) **Adversarial-batch design.** What kind of adversarial generations
    minimise $M$? Random outputs? Outputs from a prior-generation
    model? Carefully-crafted edge cases? The choice affects whether
    we measure *worst-case* or *typical-case* orthogonality.

(c) **In-loop measurement.** Can $\widehat\Theta_k$ be estimated
    efficiently in-loop during distillation, or must it be a
    pre-deployment audit? If in-loop, what's the per-round overhead
    in samples?

### Final integration: the practical Phase 3 architecture

Synthesize Q1, Q2, Q3 into a deployable architectural recipe of the
form:

> "For the Carnot code-repair / math-proof / verification domain,
>  ship a verifier suite $\{E_1, ..., E_k\}$ where:
>  (i) $E_1 \in$ class A, $E_2 \in$ class B, ... by the Q1 axiom
>      checklist;
>  (ii) Pairwise $\theta_F$ is measured at deployment via the Q3
>       protocol, requiring $M \geq M^*$ adversarial samples;
>  (iii) Recursive composition via Q2(c) is applied to extend $k$
>        without fundamental redesign;
>  (iv) The joint null is empirically confirmed trivial on a
>       holdout adversarial batch before training."

Provide the explicit recipe, numbered ingredients with quantitative
parameter bounds.

---

## Internal cross-validation predictions (DO NOT PASTE)

### Q1 predictions

(a) Sufficient axioms likely include:
- **Semantic grounding** (at least one $E_i$ depends on intent/spec, not just output structure) — the missing axiom in Carnot's planned suite.
- **Distinct invariants** (each $E_i$ measures a different conserved quantity at the truth manifold).
- **Information-theoretic separation** with $I(E_i; E_j | \mu_P) \leq c < I(E_i; \mu_P)$.

(b) Necessary condition: at least one $E_i$ must be **input-dependent**
    (semantic-grounded) to avoid the vacuous-output joint null. This
    is the reason structural-only suites (like our planned one) fail.

(c) Most secure pairs likely:
- (Z3 SMT, Differential testing) — orthogonal because differential
  testing's kernel is semantic-deviation, Z3's is structural.
- (Property-based tests, LLM-judge) — orthogonal because property
  tests catch edge cases, LLM-judge catches semantic alignment.

### Q2 predictions

(a) Minimum $k = 3$ if we mix structural + semantic + differential.
    All three classes contribute fundamentally different kernels.
(b) Adaptive auto-generation from formal specs is partially possible
    for typed languages — Liquid Haskell-style refinement types,
    Coq specifications. Less feasible for natural-language intents.
(c) Recursive composition: $E_i \wedge E_j$ has kernel
    $\mathrm{null}(E_i) \cup \mathrm{null}(E_j)$ which is LARGER, not
    smaller, than either. So intersection composition is anti-helpful.
    Sum composition $E_i + E_j$ might help if the energies are
    additive-incommensurable, but mostly equivalent to ensemble combine
    (which is bad per Round-7).

### Q3 predictions

(a) Sample complexity $M = O(k^2 / \eta^2 \cdot \log(1/\beta))$ via
    Hoeffding's inequality on each entry of $\widehat\Sigma$, then
    union bound.
(b) Adversarial-batch from a prior-generation Carnot model is the
    right tradeoff: covers realistic mode-collapse without requiring
    a true adversary. Fully-random outputs are too weak; carefully
    crafted attacks are too narrow.
(c) In-loop measurement is feasible at $\sim k^2 \cdot 10^3$ samples
    per round; pre-deployment audit is preferred for confidence but
    in-loop refinement keeps $\widehat\Theta_k$ current.

### Final integration prediction

Practical Phase 3 recipe:
1. $E_1$ = Z3 SMT (structural soundness)
2. $E_2$ = differential testing against reference impl (semantic
   correctness)
3. $E_3$ = property-based tests (edge-case coverage)
4. (optional) $E_4$ = LLM-judge alignment (intent grounding — but with
   the "judge can be gamed" caveat)

Pairwise $\theta_F \geq \pi/4$ likely achievable; rotated plateau
2-orders better than single Z3.

## Action plan after Round-9

1. If Q1 gives a checklist of axioms, ship `python/carnot/verifier_suite/`
   module with:
   - `BaseVerifier` protocol enforcing the axioms via type signatures.
   - `SuiteValidator` that runs the Q3 measurement protocol.
2. New change proposal `phase-3-verifier-suite-design.md` if Q2
   gives a constructive recipe.
3. New experiments in Carnot:
   - Exp H: build candidate Phase 3 suite per the recipe.
   - Exp I: empirical Friedrichs/Dixmier measurement.
   - Exp J: rotated-distillation on a synthetic adversarial agent.

If Q2(b) confirms adaptive verifier auto-generation, that's a major
Phase 3 architectural lever — Carnot would generate task-specific
verifier suites from formal specs at training time.
