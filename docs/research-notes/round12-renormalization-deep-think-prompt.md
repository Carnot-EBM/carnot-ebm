# Deep Think Prompt — Round-12 Re-Derivation Under Proper Normalization

**Status:** Ready to send. Closes a Round-13-exposed gap in the Round-12
saturation theorem: the original derivation assumed unnormalized
projections, but real Bayesian self-distillation has $Z_t < 1$ which
**amplifies** the null component instead of preserving it.
**Date drafted:** 2026-04-29
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip the predictions footer.

---

## Prompt to send (verbatim)

### Background (use as established premises)

**Verifier-filtered self-distillation.** Base model $Q_t$ on
$\mathcal{X}$, verifier suite $\mathcal{E} = \{E_1, \ldots, E_k\}$
with $E_i: \mathcal{X} \to \{0, 1\}$, truth $\mu_P$. Single-step update:

$$
Q_{t+1}(x) = \frac{Q_t(x) \cdot \mathbb{1}[E_{i_t}(x) = 1]}{Z_t}, \quad
Z_t = \int Q_t(x) \cdot \mathbb{1}[E_{i_t}(x) = 1] \, dx
$$

Acceptance probability $Z_t \in (0, 1)$ is strictly less than 1 whenever
the verifier rejects any positive-measure set under $Q_t$.

**Decomposition.** Active subspace $V_{\text{active}} = \mathrm{span}(\bigcup_i N_i^\perp)$;
joint null space $V_{\text{null}} = \bigcap_i N_i$. Residual $\nu_t = Q_t - \mu_P$
splits as $\nu_t = \nu_t^\parallel + \nu_t^\perp$ with $\nu_t^\parallel \in V_{\text{active}}$
and $\nu_t^\perp \in V_{\text{null}}$.

**Round-12 claim (un-normalized).** Modeled the update as alternating
orthogonal projection $\nu_{t+1} = \Pi_{N_{i_t}^\perp} \nu_t$ (project the
residual onto the active subspace of verifier $i_t$). Concluded:

$$
\delta_t = \|\nu_t\|_{L^2(\mu_P)} \to \|\nu_0^\perp\|_{L^2(\mu_P)} = \delta_\infty^{\text{static}}
$$

i.e., the active component vanishes and the null component is preserved.

**Round-13 critique.** Real Bayesian distillation is *normalized*. Under
normalization with $Z_t < 1$:

$$
\nu_{t+1}^\perp = \frac{\nu_t^\perp}{Z_t} \quad \implies \quad \|\nu_{t+1}^\perp\| = Z_t^{-1} \|\nu_t^\perp\|
$$

The null component is **amplified**, not preserved. The unnormalized
"orthogonal projection" model that Round-12 used was a convenient but
inaccurate idealization. Round-13 noted that the true saturation must
include a $\prod_t Z_t^{-1}$ factor.

### Goal

Re-derive the Round-12 saturation theorem **with proper
normalization**. Determine:

1. The corrected closed-form for $\delta_\infty^{\text{static, normalized}}$.
2. Whether the qualitative Round-12 claim (Carnot saturates a fundamental
   information-theoretic floor) **survives**, **strengthens**, or **breaks**.
3. The new tightness conditions and any revised assumptions.

### Question 1: Single-step normalization analysis

Express $\nu_{t+1} = Q_{t+1} - \mu_P$ as a function of $\nu_t$ and the
verifier $E_{i_t}$.

(a) **Decomposition.** Show that:

$$
\nu_{t+1}^\parallel = \frac{\Pi_{N_{i_t}^\perp} \nu_t^\parallel - (1 - Z_t) \mu_P^\parallel}{Z_t},
\quad
\nu_{t+1}^\perp = \frac{\nu_t^\perp - (1 - Z_t) \mu_P^\perp}{Z_t}
$$

or correct the formula. Identify the contraction factor for the active
component and the inflation factor for the null component as functions
of $Z_t$ and the verifier accuracy $\varepsilon_{i_t}$.

(b) **Per-step gain/loss.** Define:

$$
\alpha_t = \frac{\|\nu_{t+1}^\parallel\|}{\|\nu_t^\parallel\|} \quad \text{(active contraction)}
$$
$$
\beta_t = \frac{\|\nu_{t+1}^\perp\|}{\|\nu_t^\perp\|} \quad \text{(null inflation)}
$$

Express $\alpha_t$ and $\beta_t$ in terms of $Z_t$, the Friedrichs angle
$\theta_F$, and verifier accuracy $\varepsilon^*$. Is $\alpha_t \beta_t < 1$
guaranteed (so that total residual still shrinks), or can $\beta_t$
overcome $\alpha_t$ in adversarial cases?

### Question 2: Asymptotic saturation under normalization

Under the rotation defence ($i_t = \arg\max_i \|\Pi_{N_i^\perp} \nu_t\|$):

(a) **New closed form.** Derive the corrected saturation residual:

$$
\delta_\infty^{\text{static, normalized}} = ?
$$

Likely form involves $\prod_t Z_t^{-1}$ — does this product converge,
diverge, or stabilize at a finite limit?

(b) **Compare to Round-12.** Define the *normalization gap*:

$$
\Delta_{\text{renorm}} = \delta_\infty^{\text{static, normalized}} - \delta_\infty^{\text{Round-12}}
$$

Is $\Delta_{\text{renorm}} \geq 0$ always? Under what conditions on
$\varepsilon^*$, $\theta_F$, and the Dixmier angle $\Theta_k$ is it
strictly positive vs. negligible? In particular: does $\Delta_{\text{renorm}}$
scale as $O(\varepsilon^*)$, $O(\theta_F)$, or some product?

(c) **Total contraction.** Even with null inflation, does the *total
residual* $\delta_t = \|\nu_t\|$ still decrease monotonically toward
the saturation, or can it temporarily increase before reaching the
limit?

### Question 3: Does the Round-12 qualitative claim survive?

Round-12 claimed: "Carnot's verifier-filtered self-distillation saturates
the information-theoretic lower bound, modulo the joint null space."

(a) **Survival under normalization.** Does this claim hold under
the corrected derivation? Specifically, is $\delta_\infty^{\text{static, normalized}}$
still equal to the Fano lower bound $\Phi^{-1}(I(\mathcal{E}; \mu_P))$,
or is there a normalization-induced *gap*?

(b) **If gap exists.** Characterize the gap $\Delta_{\text{Fano-renorm}} = \delta_\infty^{\text{static, normalized}} - \Phi^{-1}(I(\mathcal{E}; \mu_P))$.
Is it driven by $\varepsilon^*$ (verifier noise), by the rejection rate
(making $Z_t$ small), or by the geometry of $V_{\text{null}}$?

(c) **Qualitative reframe.** If the original Round-12 statement needs
revision, propose a precise corrected form. For example:

> "Carnot's verifier-filtered self-distillation saturates the
> information-theoretic lower bound *up to a normalization factor
> $\prod_t Z_t^{-1}$ that converges to $C(\varepsilon^*, \theta_F, \rho_0)$
> under the rotation defence.*"

### Question 4: Position-paper-ready statement

Suppose the corrected derivation lands cleanly. What is the exact one-sentence
theorem statement that should appear in the position paper, replacing the
current Round-12 claim?

Format constraint: should be precise enough that a reviewer can verify it
against your derivation, but compact enough to fit in a position paper's
abstract or theorem-statement environment.

### Question 5: Engineering implications

If $\Delta_{\text{renorm}}$ is *non-negligible* (say, $\geq 0.1 \cdot \delta_\infty^{\text{Round-12}}$),
what's the implication for Carnot's deployment?

(a) **Should rejection rate be capped?** A small $1 - Z_t$ keeps null
inflation modest. Should the conductor cap rejection rate to
$\leq r^*$ to keep $\Delta_{\text{renorm}}$ bounded? If so, give $r^*$
as a function of target gap.

(b) **Trade-off with active contraction.** A high rejection rate gives
strong active contraction (good) but high null inflation (bad). Is there
an optimal $Z_t^* \in (0, 1)$ that minimizes total residual? Closed-form
welcome.

(c) **Rotation defence revisited.** Round-12 framed rotation defence as
"choose verifier maximally orthogonal to current residual." Does the
normalization-corrected analysis change this prescription? E.g., should
the rotation prefer a verifier with *moderate* orthogonality so that
$Z_t$ doesn't get too small?

### Format

Quantitative bounds in closed form whenever possible, with explicit
constants in terms of $\theta_F$, $\Theta_k$, $\varepsilon^*$, and
distribution invariants of $\mu_P$ as needed.

If your derivation reveals that **the Round-12 result was qualitatively
wrong, not just quantitatively imprecise**, please say so directly —
that's a much more valuable finding than a tweak. Conversely, if the
$Z_t < 1$ effect is asymptotically negligible (e.g. $\Delta_{\text{renorm}} = O(\varepsilon^*)$
in the rotation-defence regime), confirm that explicitly so the position
paper can keep its current claim as-is.

---

## Predictions footer (do NOT paste — for cross-validation only)

**Carnot's pre-registered guesses** (confidence levels for cross-checking):

- **Q1(a):** Decomposition formula likely correct as stated, modulo
  some $\mu_P^\parallel / \mu_P^\perp$ projection that may simplify
  if $\mu_P$ is properly normalized. Confidence: MEDIUM.
- **Q1(b):** $\alpha_t \beta_t < 1$ should hold under rotation defence
  because the active contraction is exponentially fast in $\theta_F$ but
  null inflation is only $1/Z_t$. Confidence: MEDIUM.
- **Q2(a):** $\prod_t Z_t^{-1}$ converges to a finite limit under rotation
  defence (active contraction dominates). New saturation:
  $\delta_\infty^{\text{normalized}} = \delta_\infty^{\text{Round-12}} \cdot C$
  where $C \geq 1$ is a renormalization factor depending on $\varepsilon^*$
  and the geometry. Confidence: MEDIUM.
- **Q2(b):** $\Delta_{\text{renorm}} > 0$ always (verifier noise + finite
  rejection rate combine), but scales as $O(\varepsilon^*)$ — small in
  the high-quality-verifier regime. Confidence: MEDIUM.
- **Q2(c):** Total residual *can* temporarily increase early in the
  trajectory if the rejection rate is high (initial $Z_t$ small) before
  the active component is mostly eliminated. Then monotonically decreases.
  Confidence: LOW.
- **Q3(a):** The qualitative Round-12 claim **survives** (Carnot saturates
  the Fano bound) but at a slightly inflated constant. Confidence: HIGH
  (this is the position-paper-friendly prediction).
- **Q3(b):** If gap exists, it's driven primarily by $\varepsilon^*$
  (verifier noise) — the same factor that bounds $\rho_\infty < 1$ in
  Round-13. Confidence: MEDIUM.
- **Q4:** Updated theorem statement: *"Carnot's verifier-filtered self-distillation
  saturates the Fano lower bound up to a renormalization factor
  $C(\varepsilon^*, \theta_F) \geq 1$ that approaches 1 as verifier accuracy
  improves."* Confidence: MEDIUM (this assumes Q3a is right).
- **Q5(a):** Rejection rate cap probably *not* needed — if $\Delta_{\text{renorm}}$
  is $O(\varepsilon^*)$, no engineering change required. Confidence: MEDIUM.
- **Q5(b):** Optimal $Z_t^*$ might exist (a tradeoff sweet spot), but in
  the high-quality-verifier regime ($\varepsilon^* \ll 1$), the optimum
  is near $Z_t \to 1$ (low rejection per step, more iterations). Confidence:
  LOW.
- **Q5(c):** Rotation defence prescription likely **unchanged** — the
  renormalization affects the constant, not the gradient direction.
  Confidence: MEDIUM.

**What we want most from Deep Think:**
- A clean closed form for $C(\varepsilon^*, \theta_F)$ (the normalization
  factor in the new saturation residual).
- An honest answer on whether the Round-12 qualitative claim survives or
  needs to be retracted in the position paper.
- A precise one-sentence theorem statement (Q4) ready for paper insertion.
