# Deep Think Prompt — Phase-4 DVS Quality Bound + Phase-3 Curriculum Scheduling

**Status:** Ready to send. Combines two open questions:
(A) the minimum quality threshold for dynamically synthesized verifiers
in Round-13's UCM+DVS Phase-4 prescription;
(B) the optimal lenient→strict curriculum schedule emerging from
this round's rejection-rate-driven analysis.

These two questions are coupled: the DVS quality must satisfy the
curriculum constraints, and the curriculum determines when DVS triggers.
**Date drafted:** 2026-04-29
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip the predictions footer.

---

## Prompt to send (verbatim)

### Background (use as established premises)

**Verifier-filtered self-distillation with Z_t < 1 normalization.**

$$
Q_{t+1}(x) = \frac{Q_t(x) \cdot \mathbb{1}[E_{i_t}(x) = 1]}{Z_t}
$$

Active subspace $V_{\text{active}} = \mathrm{span}(\bigcup_i N_i^\perp)$;
joint null space $V_{\text{null}} = \bigcap_i N_i$. Residual decomposition
$\nu_t = \nu_t^\parallel + \nu_t^\perp$.

**Established results:**

1. **Round-12 saturation (corrected).** Under the rotation defence, the
   asymptotic residual converges to $\delta_\infty = C_Z \cdot \|\nu_0^\perp\|$
   where $C_Z = \prod_t Z_t^{-1} > 1$ is finite and driven by cumulative
   rejection rate during early steps.

2. **Round-13 drift gap.** Under concept drift (residual rotates into
   joint null space), Carnot's static rotation defence achieves zero
   improvement against an adversary that embeds corruption in
   $V_{\text{null}}$ at $t=1$. Phase-4 prescription: Unsupervised Consensus
   Monitor (UCM) detects sycophantic drift via acceptance rate, then
   triggers Dynamic Verifier Synthesis (DVS) to inject a new verifier
   $E_{k+1}$ trained on the accepted-but-corrupt sample distribution.

3. **Curriculum trajectory.** Today's renormalization round established
   that optimal $\{Z_t\}$ trajectory is curriculum-style: start with
   lenient verifiers ($Z_t \to 1$, low rejection), transition to strict
   verifiers as the active component depletes.

### Part A — DVS quality lower bound

When UCM triggers DVS, the new verifier $E_{k+1}$ is trained on an
audit sample of size $K$. Define $E_{k+1}$'s **active subspace
overlap** with the existing suite as:

$$
\Lambda(E_{k+1}, \mathcal{E}) = \frac{\|\Pi_{V_{\text{active}}^{(\mathcal{E})}} \nu^\perp_{E_{k+1}}\|}{\|\nu^\perp_{E_{k+1}}\|}
$$

where $\nu^\perp_{E_{k+1}}$ is the residual perpendicular to $E_{k+1}$'s
own active subspace, and $V_{\text{active}}^{(\mathcal{E})}$ is the union
of the existing suite's active subspaces.

If $\Lambda \approx 1$, the new verifier's null space lies entirely in
the existing suite's active subspace — the new verifier adds no
information about $V_{\text{null}}$. If $\Lambda \approx 0$, the new
verifier is orthogonal — maximum information gain.

#### Question A1: Quality threshold for DVS to actually help

(a) **Threshold $q^*$.** Derive the minimum $\Lambda^*$ (or equivalent
   quality measure) such that injecting $E_{k+1}$ into the suite
   *actually shrinks* the joint null space $V_{\text{null}}$ measured
   by the residual norm $\|\nu_{t}^\perp\|$.

(b) **Threshold below which DVS hurts.** Is there a regime where a
   too-low-quality $E_{k+1}$ makes things *worse* — e.g., introduces
   a verifier whose own active subspace partially overlaps the
   existing suite, increasing the rejection rate $1 - Z_t$ without
   commensurate kernel reduction, thus inflating $C_Z$ unnecessarily?
   If yes, give the threshold below which injection is net-negative.

(c) **PAC-Bayes sample complexity.** For a given target $\Lambda \leq \Lambda^*$,
   what's the minimum audit budget $K^*(\Lambda^*)$ needed to train
   $E_{k+1}$ with VC-dimension $d$? A clean PAC-Bayes bound is welcome.

#### Question A2: Stability of the dynamic suite

After DVS injects $E_{k+1}$ into the FIFO-managed suite, does the
saturation theorem still apply to the *new* suite $\mathcal{E} \cup \{E_{k+1}\}$?

(a) **Saturation under suite mutation.** Show that the rotation
   defence + AND-composition still saturate the Fano lower bound on
   the new (smaller) joint null space. Identify any required
   warm-start conditions on $Q_t$.

(b) **FIFO-eviction validity.** When $|\mathcal{E}| > k_{\max}$ and
   FIFO evicts the oldest verifier, can the joint null space *grow
   back*? Bound the growth.

(c) **Repeated DVS triggers.** Under recurring drift (UCM triggers
   every $T_{\text{refresh}}$ steps), what's the steady-state
   joint-null-space dimension? Does it stabilize, or does Phase-4
   suffer from *churn* (each new $E_{k+1}$ partially overlaps the
   prior, leading to ineffective updates)?

### Part B — Curriculum scheduling

Today's renormalization analysis established that minimizing
$\Delta_{\text{renorm}} = (C_Z - 1)\|\nu_0^\perp\|$ requires balancing
active contraction against null inflation. The optimal trajectory is
not constant $Z_t$ but a curriculum.

#### Question B1: Optimal curriculum schedule

Define a **strictness parameter** $\sigma_t \in [0, 1]$: $\sigma_t = 0$
means $Z_t = 1$ (everything accepted, zero filtering); $\sigma_t = 1$
means $Z_t \to 0$ (only the most-aligned candidates accepted). The
verifier-filtered distillation step is:

$$
Q_{t+1}(x) \propto Q_t(x) \cdot E_{i_t}^{(\sigma_t)}(x)
$$

where $E_{i_t}^{(\sigma_t)}$ is the verifier with adjusted strictness
(e.g., a temperature parameter on the energy).

(a) **Closed-form schedule.** Derive the optimal curriculum
   $\sigma_t^*$ as a function of the active-component norm
   $\|\nu_t^\parallel\|$. Likely form: $\sigma_t^* = f(\|\nu_t^\parallel\|, \theta_F, \Theta_k)$.

(b) **Phase transitions.** Are there discrete transition points
   (where $\sigma_t^*$ jumps, or where the optimal verifier changes)
   rather than a smooth schedule? Identify any critical thresholds
   on $\|\nu_t^\parallel\|$.

(c) **Comparison to constant-strictness.** Quantify the gain
   $\Delta C_Z = C_Z^{\text{constant}} - C_Z^{\text{curriculum}}$
   from running the optimal curriculum vs. constant strictness.
   Is this gain $O(1)$, $O(\theta_F)$, or larger?

#### Question B2: Curriculum + AND-composition interaction

Round-9's prescription used fixed AND-composition $k = 15$. Today's
finding suggests $\sigma_t$ varies; should $k$ also vary?

(a) **Is $k$ part of the curriculum?** During lenient phases (low
   $\sigma_t$), should $k$ be small (less aggressive AND-composition)?
   During strict phases, should $k$ be large?

(b) **If $k$ is curriculum-dependent.** Derive the optimal $k_t^*$
   as a function of $\|\nu_t^\parallel\|$, $\sigma_t$. Closed form
   welcome.

(c) **Sample efficiency.** A larger $k$ requires more candidates per
   accepted sample (rejection rate is exponential in $k$). Is the
   compute cost of an adaptive $k_t^*$ higher than fixed $k = 15$,
   or does the curriculum's lenient early phase actually reduce
   total compute?

### Part C — Synthesis (DVS + Curriculum)

#### Question C1: How DVS interacts with curriculum

A new verifier $E_{k+1}$ is synthesized at time $t^*$ when UCM
triggers (sycophancy crosses threshold). At that moment, the
curriculum is at $\sigma_{t^*}$ (likely strict, since drift has
accumulated by then).

(a) **Where in curriculum should the new verifier enter?**
   Should $E_{k+1}$ start at $\sigma_{t^*}$ (matching the
   curriculum), or should it start lenient and be re-strict-ened
   over time even as the rest of the suite continues at $\sigma_{t^*}$?

(b) **Quality threshold under curriculum.** Does the DVS quality
   threshold $q^*$ from Q A1 depend on $\sigma_{t^*}$? Specifically,
   is a stricter curriculum more or less forgiving of low-quality
   synthesized verifiers?

#### Question C2: Phase-3+4 unified prescription

Combine the rotation defence (Round-7), AND-composition (Round-9),
in-loop tripwire (Round-9), Round-12 saturation (corrected this round),
UCM+DVS (Round-13), and the curriculum (this round) into a single
unified architecture spec.

(a) **Compact statement.** What's the unified "Phase 3+4" architecture,
   stated as a one-paragraph procedural recipe, that the position
   paper can quote?

(b) **Hyperparameter set.** Enumerate the irreducible
   hyperparameters: $k$ (or $k_t^*$), $\sigma_t$ schedule, UCM
   threshold $\eta$, DVS audit budget $K$, FIFO suite size $k_{\max}$,
   rotation policy. Which ones can be set by closed-form bounds and
   which require empirical tuning?

(c) **Combined saturation theorem.** Does the full Phase 3+4 system
   saturate a meaningful information-theoretic lower bound? Or is
   there an additional residual gap from the *interaction* of these
   mechanisms (e.g. UCM trigger latency, DVS quality drop, curriculum
   transition sub-optimality)?

### Format

Quantitative bounds in closed form whenever possible. Where rigorous
proofs are infeasible, mark intuition vs. theorem and identify the
required regularity assumptions.

If your derivation reveals that **DVS or curriculum is fundamentally
broken** (e.g., the quality threshold $q^*$ requires $K$ scaling
super-polynomially, or the curriculum doesn't actually beat constant
strictness), please say so directly. The position paper benefits more
from honest characterization of where the architecture has open
problems than from a contrived patch.

---

## Predictions footer (do NOT paste — for cross-validation only)

**Carnot's pre-registered guesses** (after two rounds with several
prediction errors, attempting to be more cautious about driver
attributions):

### Part A predictions

- **A1(a) quality threshold $\Lambda^*$:** Likely $\Lambda^* < 1$ —
  some non-trivial overlap is allowed because each new verifier
  shrinks the joint null even if not orthogonal to existing.
  Confidence: MEDIUM.
- **A1(b) net-negative regime:** YES, exists. A verifier that's
  identical to one already in the suite (Λ=1, fully redundant)
  inflates rejection without help. Threshold likely $\Lambda > 1 - 1/k$
  (verifier nearly identical to one of $k$ existing). Confidence: LOW.
- **A1(c) PAC-Bayes:** $K^*(q^*) = O(d \log d / q^{*2})$ standard
  PAC-Bayes form. Confidence: MEDIUM.
- **A2(a) saturation under mutation:** Theorem holds. Required
  warm-start: $Q_t$'s active component must have non-trivial
  projection onto $E_{k+1}$'s active subspace. Confidence: MEDIUM.
- **A2(b) FIFO eviction:** Joint null can grow back if evicted
  verifier was the only one cleaving a particular dimension. Bound:
  growth $\leq O(1/k_{\max})$ per eviction. Confidence: LOW.
- **A2(c) repeated DVS:** Steady-state stabilizes at a finite null
  dimension > 0 (some dimensions are intrinsically hard to verify).
  Churn risk if drift adversary can rotate faster than DVS
  synthesizes. Confidence: MEDIUM.

### Part B predictions

- **B1(a) optimal $\sigma_t^*$:** $\sigma_t^* = 1 - \|\nu_t^\parallel\|/\|\nu_0^\parallel\|$
  (linear ramp from 0 to 1 as active component depletes). Confidence: LOW.
- **B1(b) phase transitions:** Smooth, no discrete jumps. Confidence: MEDIUM.
- **B1(c) gain over constant:** $\Delta C_Z = O(1)$ multiplicative,
  i.e., curriculum gives a constant-factor improvement, not asymptotic.
  Confidence: LOW.
- **B2(a) is $k$ part of curriculum:** YES — small $k$ early (lenient),
  larger $k$ late (strict). Confidence: MEDIUM.
- **B2(b) optimal $k_t^*$:** $k_t^* \propto -\log(\|\nu_t^\parallel\|)$
  (logarithmic growth). Confidence: LOW.
- **B2(c) compute cost:** Lower than fixed $k=15$ in early phases,
  similar in late phases. Net compute: comparable or slightly less.
  Confidence: MEDIUM.

### Part C predictions

- **C1(a) DVS entry point:** $E_{k+1}$ should enter at $\sigma_{t^*}$
  matching current curriculum. Re-leniency would waste audit budget.
  Confidence: MEDIUM.
- **C1(b) quality under curriculum:** Stricter curriculum is *less*
  forgiving — already-tight rejection rate amplifies any null-space
  overlap from low-quality $E_{k+1}$. Confidence: HIGH.
- **C2(a) unified statement:** "Carnot maintains an adaptive verifier
  suite of size $k_t$ under a strictness curriculum $\sigma_t$,
  rotated for orthogonality with $Z_t$ floor, AND-composed for joint
  kernel reduction, with UCM-triggered DVS injection upon detected
  drift." Confidence: MEDIUM.
- **C2(b) hyperparameters:** Closed-form bounds: $k_t^*$, $\sigma_t^*$,
  $\Lambda^*$. Empirical tuning: $\eta$, $K$, $k_{\max}$. Confidence: LOW.
- **C2(c) combined saturation:** Theorem holds modulo *trigger latency*
  gap (UCM responds to drift after some delay, residual escape can
  partially happen during that delay). Confidence: HIGH (this is the
  predicted reviewer-bait result that motivates a Phase-5 "predictive
  drift detection" extension).

**What we want most from Deep Think:**
- A clean closed-form for the DVS quality threshold $\Lambda^*$ (A1a)
  and the under-which-DVS-hurts threshold (A1b).
- The optimal curriculum schedule $\sigma_t^*$ as a function of
  active-component norm (B1a) — whether smooth or with discrete jumps.
- The unified Phase 3+4 procedural recipe (C2a) ready for paper insertion.
- An honest answer to C2c: does the combined system saturate the bound,
  or is there a trigger-latency residual gap?
