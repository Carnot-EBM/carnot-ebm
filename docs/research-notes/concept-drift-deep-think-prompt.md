# Deep Think Prompt — Carnot Saturation Under Concept Drift

**Status:** Ready to send. Closes the strongest predictable reviewer
critique of the Phase-3 saturation result (Round-12) — *"your bound
assumes a moving target stays still."*
**Date drafted:** 2026-04-29
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip the predictions footer.

---

## Prompt to send (verbatim)

### Background (use as established premises)

**Verifier-filtered self-distillation.** A base model $Q_t$ on
$\mathcal{X}$ is filtered through a verifier suite
$\mathcal{E} = \{E_1, \ldots, E_k\}$ with $E_i: \mathcal{X} \to \{0, 1\}$.
The filter draws candidates from $Q_t$ and accepts $x$ iff $E_i(x) = 1$
for the rotation-selected verifier $i$. Truth distribution is $\mu_P$.
Acceptance distribution at step $t$:

$$
Q_{t+1}(x) \propto Q_t(x) \cdot \prod_i \mathbb{1}[E_i(x) = 1]^{w_i(t)}
$$

where $w_i(t) \in [0, 1]$ are the rotation weights at step $t$.

**Round-12 saturation theorem (static case).** When the corruption
distribution is fixed at the initial $Q_0 - \mu_P$ and the verifier
suite $\mathcal{E}$ is fixed, the achievable residual converges to

$$
\delta_\infty^{\text{static}} = \|\Pi_{\bigcap_i N_i^\perp} (Q_0 - \mu_P)\|_{L^2(\mu_P)}
$$

i.e., the projection of the initial residual onto the joint kernel of
the verifier suite. With $k$-fold AND-composition, this kernel
shrinks by Friedrichs-angle $\theta_F$ raised to the $k$-th power.

**Verifier accuracies** $\{\varepsilon_i\}$ are bounded
($\varepsilon_i \leq \varepsilon^*$ for all $i$), the suite has
**Dixmier angle** $\Theta_k > 0$ (joint linear independence), and the
**Friedrichs angle** $\theta_F$ between consecutive rotations is
strictly positive (rotation defence holds).

### The drift problem

In real deployment, $Q_t$ drifts as distillation proceeds. Equivalently,
the *corruption distribution* $\nu_t = Q_t - \mu_P$ drifts away from the
distribution the verifier suite $\mathcal{E}$ was designed/calibrated
against. **The verifier is a static map**, but its *information
content about the residual at step $t$* is not constant in $t$.

Concretely: the verifier $E_i$ was tuned so that $E_i(x) = 0$ on
"obvious corruptions" near $Q_0 - \mu_P$. After many distillation
steps, $Q_t - \mu_P$ may live in directions of $L^2(\mu_P)$ where
$E_i$ has near-zero discriminatory power — the verifier's null space
*grows* relative to the residual as residual rotates into it. This
is the "verifier-as-moving-target" failure mode.

**Definition (concept drift rate).** Define drift rate $\rho_t$ as

$$
\rho_t = \frac{ \|\nu_t - \Pi_{V_{\text{active}}} \nu_t\|_{L^2(\mu_P)} }{ \|\nu_t\|_{L^2(\mu_P)} }
$$

where $V_{\text{active}} = \mathrm{span}(\bigcup_i N_i^\perp)$ is the
active subspace from Round-7. $\rho_t \in [0, 1]$: zero when the
residual is fully in the active subspace, one when it is fully in
the joint null space.

### Question 1: Drift-free saturation theorem (sanity check)

Verify that when $\rho_t \equiv \rho_0$ is constant for all $t$
(no drift), the Round-12 saturation holds: $\delta_t \to \delta_\infty^{\text{static}}$.

This is the static case in disguise — the residual stays in a fixed
subspace of $L^2(\mu_P)$. Confirm the bound matches Round-12's, and
identify any hidden assumptions in Round-12 that depend on $\rho_t$
being constant.

### Question 2: Drift dynamics under the rotation defence

Suppose the rotation defence is active: at step $t$, verifier
$i_t \in \{1, \ldots, k\}$ is selected to *maximize* the residual's
projection onto $N_{i_t}^\perp$:

$$
i_t = \arg\max_i \|\Pi_{N_i^\perp} \nu_t\|_{L^2(\mu_P)}
$$

(a) **Drift accumulation.** Show that even under optimal rotation,
   $\rho_t$ is non-decreasing in $t$. Bound the rate of increase in
   terms of $\theta_F$, $\varepsilon^*$, and the Dixmier angle
   $\Theta_k$. Specifically: is $\rho_t \leq \rho_0 + c(\theta_F, k) \cdot t$
   (linear), $1 - (1 - \rho_0)(1 - c)^t$ (exponential), or some other
   form?

(b) **Asymptotic limit.** As $t \to \infty$, does $\rho_t \to 1$
   (full residual escape into joint null space) or $\rho_t \to \rho_\infty < 1$
   (rotation forces some active component to persist)? If
   $\rho_\infty < 1$, give a closed-form lower bound in terms of
   $\theta_F$ and $\Theta_k$.

(c) **Compare with Round-12 static bound.** Define the *drift gap*
   $\Delta_{\text{drift}} = \delta_\infty^{\text{drift}} - \delta_\infty^{\text{static}}$
   where $\delta_\infty^{\text{drift}}$ is the achievable residual
   under drift. Is $\Delta_{\text{drift}} \geq 0$ always, and
   under what conditions is it strictly positive?

### Question 3: Fano lower bound under drift

Round-12 derived a Fano lower bound based on $I(\mathcal{E}; \mu_P)$.
Under drift, the *effective* mutual information at step $t$ is

$$
I_t(\mathcal{E}; \mu_P | Q_t) = I(\mathcal{E}; \mu_P) - I_{\text{drift}}(t)
$$

where $I_{\text{drift}}(t)$ measures how much information about $\mu_P$
the verifier *loses* as $Q_t$ drifts away from the calibration
distribution.

(a) **Define $I_{\text{drift}}(t)$.** Give a precise information-theoretic
   definition. Likely candidates: KL divergence between $E_i(X)|X \sim Q_t$
   and $E_i(X)|X \sim Q_0$, or a chain-rule decomposition involving
   the active-subspace projection.

(b) **Time-dependent Fano.** Derive the time-dependent Fano lower
   bound:

$$
\delta_t \geq \Phi^{-1}(I_t(\mathcal{E}; \mu_P | Q_t))
$$

   where $\Phi$ is the Round-12 conversion function. Is the bound
   monotone in $t$? Does it converge to a finite limit?

(c) **Saturation under drift.** Does Carnot's rotation defence
   *saturate* the drift-aware Fano bound, i.e., is

$$
\lim_{t \to \infty} \delta_t^{\text{rot}} = \lim_{t \to \infty} \Phi^{-1}(I_t)
$$

   (still tight asymptotically), or is there a *drift gap* even at
   the optimal achievable rate? If a gap, characterize what
   architectural change closes it.

### Question 4: Online verifier updates (regret bound)

Suppose the verifier suite is *not static* — at step $t$, an online
algorithm receives feedback (e.g. ground-truth labels on a small
audit subset) and updates one verifier $E_{i_t} \to E_{i_t}^{(t)}$ to
re-align with the current residual $\nu_t$. The cost: $K$ ground-truth
queries per update, $\tau$ steps between updates.

(a) **Per-update gain.** Each update reduces $\rho_t$ by how much,
   in terms of the audit budget $K$ and the residual's structure?
   A clean PAC-Bayes-style bound is welcome.

(b) **Regret bound.** Define cumulative regret as

$$
R_T = \sum_{t=1}^T (\delta_t^{\text{online}} - \delta_t^{\text{drift-free}})
$$

   where $\delta_t^{\text{drift-free}}$ is the static-case achievable
   at step $t$. Bound $R_T$ in terms of $T$, $K$, $\tau$,
   $\theta_F$, and $\Theta_k$. Is the regret sublinear? At what rate?

(c) **Optimal update schedule.** Given a fixed total audit budget $B$
   over $T$ steps, what's the optimal allocation $(K_t, \tau_t)$?
   Specifically: should updates be uniform in time, geometric
   (more updates early), or adaptive (triggered when $\rho_t$
   crosses a threshold)?

### Question 5: Practical implications for Carnot deployment

(a) **Drift detection.** Can $\rho_t$ be estimated *without* ground
   truth, i.e. from $Q_t$ samples and verifier outputs alone? If
   yes, give an estimator and its variance. If no, explain why and
   propose a proxy.

(b) **Architectural prescription.** Based on Q3 and Q4, what's the
   minimal modification to Carnot's Phase-3 architecture (rotation
   defence + AND-composition + audit + tripwire from Round-9) to
   make the saturation theorem *robust to drift*? Specifically: is
   the existing tripwire (audit) sufficient if the audit is
   triggered by $\rho_t$ crossing a threshold, or is a separate
   drift-detection module needed?

(c) **Worst-case drift adversary.** Suppose an adversary can choose
   the drift trajectory $\{\nu_t\}$ subject to a constraint
   $\|\nu_t\|_{L^2(\mu_P)} \leq \delta_t^{\text{achievable}}$ at
   every $t$. What's the worst-case asymptotic
   $\delta_\infty^{\text{adversarial-drift}}$? Is Carnot's
   rotation defence still optimal against this adversary?

### Format

Be quantitative — bounds should be in closed form whenever possible,
with explicit constants in terms of $\theta_F$, $\Theta_k$,
$\varepsilon^*$, $k$, and the audit budget $B$. Where rigorous
proofs are infeasible in this session, identify the regularity
assumptions (e.g. smooth verifiers, finite-dimensional residual,
etc.) under which the result *would* hold, and clearly mark
intuition vs. theorem.

If your derivation finds that **Carnot's Phase-3 architecture is
NOT robust to drift**, please say so directly — that's a more
valuable finding than a contrived patch. The position paper benefits
from honest characterization of where the saturation theorem fails.

---

## Predictions footer (do NOT paste — for cross-validation only)

**Carnot's pre-registered guesses** (so we can detect Gemini
hallucination by comparing outputs against these):

- **Q1:** Static case is the $\rho_t \equiv \rho_0$ slice; Round-12
  should hold with no hidden drift assumptions. Confidence: HIGH.
- **Q2(a):** $\rho_t$ likely grows *exponentially* under optimal
  rotation, not linearly. Bound form
  $\rho_t \leq 1 - (1 - \rho_0)(1 - c(\theta_F))^t$ where
  $c(\theta_F) = \sin^2(\theta_F)$ or similar. Confidence: MEDIUM.
- **Q2(b):** Asymptotic $\rho_\infty = 1$ if the joint null space
  is non-trivial (which it is for any finite verifier suite by
  Round-8's pathology theorem). The rotation defence *delays* but
  does not prevent this. Confidence: HIGH.
- **Q2(c):** $\Delta_{\text{drift}} > 0$ strictly, *unless* the
  initial residual is already orthogonal to the joint null space
  (which is generically false). Confidence: HIGH.
- **Q3(a):** $I_{\text{drift}}(t) = D_{\text{KL}}(Q_t \| Q_0)$
  restricted to the active subspace. Confidence: LOW (this is a
  guess at the right definition).
- **Q3(b):** Bound is monotone decreasing in $t$ (residual gets
  harder to detect as drift accumulates), converges to
  $I_\infty = $ residual information at the joint null space
  boundary. Confidence: MEDIUM.
- **Q3(c):** Carnot does NOT saturate the drift-aware Fano bound —
  there's a drift gap. The architectural fix is *online verifier
  updates* (Q4) or *adaptive AND-composition* (auto-tune $k$ as
  $\rho_t$ grows). Confidence: HIGH (this is the predicted result
  that motivates the position paper's "Phase 4 / continual
  verifier learning" section).
- **Q4(a):** Per-update gain $O(\sqrt{K / |V_{\text{active}}|})$
  per PAC-Bayes scaling. Confidence: LOW.
- **Q4(b):** Regret is sublinear, $R_T = O(\sqrt{T \cdot \log T})$
  if updates are scheduled at intervals $\tau \propto \sqrt{T}$.
  Confidence: LOW.
- **Q4(c):** Optimal schedule is *adaptive* (triggered by $\rho_t$
  threshold) rather than uniform — uniform wastes budget when
  drift is slow. Confidence: MEDIUM.
- **Q5(a):** $\rho_t$ can be estimated from verifier acceptance
  *rate* changes alone (not requiring ground truth) — the
  acceptance rate $P_{Q_t}(E_i = 1)$ encodes how much of $Q_t$
  is in the verifier's null space. Confidence: MEDIUM.
- **Q5(b):** The Round-9 tripwire is sufficient *if* triggered by
  $\rho_t$ crossing a threshold. The threshold itself is a new
  hyperparameter. No separate module needed. Confidence: MEDIUM.
- **Q5(c):** Adversarial drift can drive Carnot to
  $\delta_\infty^{\text{adversarial}} = \|\Pi_{\bigcap_i N_i^\perp} \mu_P\|$,
  i.e., Carnot collapses to identity (no improvement) in the
  worst case. The fix requires online updates or AND-composition
  growth. Confidence: HIGH (this is the predicted reviewer-bait
  result).

**What we want most from Deep Think:**
- A clean closed-form for $\Delta_{\text{drift}}$ (Q2c).
- A constructive online-update schedule that achieves sublinear
  regret with explicit $K, \tau$ budget (Q4b/c).
- Honest assessment of whether Carnot's Phase-3 architecture *as
  shipped* (rotation + AND + tripwire from Round-9) is drift-robust,
  or whether a Phase-4 module is needed.
