# Deep Think Prompt — AND-Composition Mixing Time on Phase-2 FPGAs

**Status:** Ready to send. Bridges Phase-3 verifier math to Phase-2
hardware claims. Asks whether the AND-composition (Round-9) +
factorized-curriculum (today) maintains the MCMC mixing properties
needed for FPGA Ising-machine deployment.
**Date drafted:** 2026-04-29
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip the predictions footer.

---

## Prompt to send (verbatim)

### Background (use as established premises)

**Carnot's Phase-2 hardware deployment.** The energy-based verifier
suite is implemented on FPGA Ising machines (e.g. Xilinx KV260) and
future probabilistic-computer substrates (Extropic XTR-0). The
energy function $E(x)$ is encoded as a multi-spin Ising
Hamiltonian; samples $x \sim p(x) \propto e^{-E(x)/T}$ are drawn via
Markov Chain Monte Carlo (MCMC) on the FPGA's hardware sampler.

**Verifier suite under AND-composition (Round-9).** Phase-3 deploys
$k = 15$-fold AND-composition: a candidate $x$ is accepted iff all
$k$ randomly-selected verifiers vote $E_i(x) = 1$. The composed
energy is:

$$
E_{\text{AND}}(x) = \sum_{i \in S_k} E_i(x) + V_{\text{barrier}}(x)
$$

where $S_k$ is the random sample of $k$ verifiers and
$V_{\text{barrier}}$ enforces the AND constraint as soft barriers
during sampling.

**Factorized curriculum (today's findings).** Each verifier $E_i$ has
its own strictness parameter $\sigma_{t,i}$ (acceptance temperature)
and AND-depth $k_{t,i}$. Per-verifier curriculum: lenient on entry
($\sigma_{t,i} \approx 0$, $k_{t,i} = 1$), tightening as the
verifier's specific active residual norm $\|\nu_{t,i}^\parallel\|^2$
depletes. Optimal schedule:

$$
\sigma_{t,i}^* = \min\left(1, C / \|\nu_{t,i}^\parallel\|^2\right)
$$

The energy landscape $E_{\text{AND}}(x)$ is therefore **time-varying
and heterogeneous** — different verifiers contribute at different
strictnesses simultaneously.

### MCMC mixing requirement

For FPGA deployment, the MCMC sampler must mix to within
$\varepsilon$ of the Boltzmann distribution in $\tau_{\text{mix}}$
steps. The mixing time directly bounds latency: an $\tau_{\text{mix}}$
of $10^6$ steps at $10^9$ steps/sec on an FPGA gives 1ms inference
per candidate, acceptable; $\tau_{\text{mix}}$ of $10^{12}$ gives 1000s,
unacceptable.

Round-9's mixing argument (under uniform $\sigma$) used the
Cheeger-style isoperimetric inequality on a "convex" energy
landscape: the suite was assumed to have geodesic-convex level sets
relative to the natural Riemannian metric induced by the Boltzmann
distribution.

**Question:** does this convexity (and therefore polynomial mixing)
survive AND-composition + factorized curriculum?

### Question 1: Energy landscape geometry under AND-composition

(a) **Level-set connectivity.** AND-composition takes the *intersection*
   of feasible regions across $k$ verifiers. The intersection of $k$
   convex sets is convex, so on its face the AND-composed feasible
   region is still convex. But Carnot's verifier energy functions are
   **not strictly convex** — they're piecewise (Boolean indicator-valued
   gradients) plus a barrier. Show or refute: does AND-composition
   preserve geodesic-convexity of the *level sets* under the
   Riemannian metric induced by $e^{-E/T}$?

(b) **Spectral gap bound.** The MCMC mixing time is governed by the
   spectral gap of the transition kernel. Express the spectral gap
   of $E_{\text{AND}}$ in terms of:
   - the spectral gap of any single $E_i$ (suite homogeneity)
   - $k$ (composition depth)
   - the Friedrichs angle $\theta_F$ between consecutive verifiers'
     constraint surfaces

(c) **Worst-case mixing time.** Derive an upper bound on $\tau_{\text{mix}}$
   under AND-composition. Likely candidates: $\tau_{\text{mix}} = O(\tau_1^k)$
   (exponential in $k$ — bad), $O(k \tau_1)$ (linear — fine), or
   somewhere in between. Closed form welcome.

### Question 2: Effect of heterogeneous strictness $\sigma_{t,i}$

When the curriculum runs different $\sigma_i$ per verifier, the
energy landscape is no longer uniformly tempered. Some verifiers
contribute high-amplitude steep walls (strict, $\sigma_i \to 1$);
others contribute low-amplitude gentle slopes (lenient, $\sigma_i \to 0$).

(a) **Multi-scale energy landscape.** Show whether the heterogeneous
   landscape develops **bottlenecks** (narrow passages between
   feasible basins) that destroy the polynomial mixing. Specifically:
   when one verifier $E_i$ is strict (steep walls) and others are
   lenient, does the joint feasible region split into disconnected
   components requiring exponential mixing time?

(b) **Sampling near the strictness boundary.** A verifier transitioning
   from lenient to strict (its $\sigma_{t,i}$ crosses the bang-bang
   threshold $\sqrt{C}$) introduces a discontinuous energy-landscape
   change. Is this a phase transition that locally requires *re-mixing*?
   What's the re-mixing penalty in $\tau_{\text{mix}}$?

(c) **Optimal sampling strategy.** Should the FPGA sampler use:
   - Single Markov chain across all $k$ verifiers (current Round-9
     prescription), OR
   - **Per-verifier parallel chains** that synchronize via FDR-OR
     (suggested by today's predictive-UCM round's Phase-6 prescription)

   Quantify the mixing-time gain (or loss) from parallel chains.

### Question 3: FPGA resource budget

(a) **Spin count.** A Carnot verifier $E_i$ encoded as an Ising
   Hamiltonian requires $n_i$ binary spins (typically $n_i \sim 10^3$
   for a constraint-satisfaction-class verifier). For the
   AND-composition over $k$ verifiers, does the total spin count
   scale as:
   - $O(\sum_i n_i)$ (additive — fine), OR
   - $O(\prod_i n_i)$ (multiplicative — catastrophic on FPGAs)?

   The answer depends on whether AND-composition can be expressed as
   independent sums (additive) or requires joint encoding
   (multiplicative). Show which.

(b) **Energy precision (bit width).** FPGA fixed-point arithmetic
   has finite precision; energies $E_i \in [-E_{\max}, E_{\max}]$
   require $\lceil\log_2(2 E_{\max} / \varepsilon)\rceil$ bits.
   Under heterogeneous $\sigma_{t,i}$, the dynamic range of
   per-verifier contributions is $\sigma_{t,i} \cdot E_{\max}$.
   What's the **minimum bit width** that preserves the Round-12
   saturation guarantees?

(c) **Sampler architecture choice.** Carnot's KV260 prototype uses
   simulated bifurcation (SB) and stochastic gradient Langevin
   dynamics (SGLD) options. Which is provably better under the
   factorized-curriculum heterogeneous landscape:
   - SB (parallel discrete updates, fast on hardware but coarse) —
     mixing ~$O(n^2)$
   - SGLD (continuous Langevin with discrete noise) — mixing
     ~$O(n \log n)$ on convex landscapes
   - Tempered transitions (multi-temperature parallel chains) —
     mixing ~$O(\log n)$ but expensive parallelism

   What's the right choice given Carnot's heterogeneous-curriculum
   landscape?

### Question 4: Hardware-portability theorem

The position paper claims hardware portability across KV260 FPGAs,
Extropic XTR-0 thermodynamic samplers, and (future) photonic
Ising machines.

(a) **Universal mixing condition.** Is there a hardware-platform-
   independent sufficient condition on $E_{\text{AND}}(x)$ that
   guarantees polynomial mixing across all reasonable Ising-class
   samplers? E.g., "if the AND-composed energy has Cheeger constant
   $h \geq c \cdot k^{-\alpha}$ for some $\alpha < 1$, then any
   Ising-class sampler mixes in $O(\text{poly}(n, k, 1/h))$ steps."

(b) **Substrate-specific advantages.** Are there energy-landscape
   properties that *some* substrates exploit and others don't? E.g.:
   - Extropic's thermodynamic sampler may exploit continuous-energy
     landscapes (per Continuous Ising-Rank theorem).
   - Photonic Ising machines may benefit from the AND-composed
     constraint's optical-encoding (each verifier is one optical
     loop).

(c) **Position-paper-ready hardware claim.** What's the precise
   one-sentence theorem on hardware portability that the position
   paper can quote, given the AND-composition + factorized
   curriculum architecture?

### Question 5: Mixing failure regimes

(a) **When does mixing become infeasible?** Identify the regimes
   where $\tau_{\text{mix}}$ becomes super-polynomial (e.g.,
   $O(2^n)$ or $O(2^k)$) under realistic parameter ranges. Specifically:
   - Is there a critical $k^*(n, \sigma_{\max})$ above which
     AND-composition kills mixing?
   - Does the bang-bang transition in $\sigma_{t,i}$ (today's
     finding) create transient super-polynomial mixing?

(b) **Sampling-quality vs. saturation-bound trade-off.** A weaker
   sampler (poor mixing, biased samples) gives a worse Carnot
   saturation. Quantify the relationship between mixing precision
   $\varepsilon_{\text{mix}}$ and the saturation residual
   $\delta_\infty$. Is there a dominant regime where sampling
   error is *worse* than the renormalization gap $C_Z$?

(c) **Recommendation for KV260 prototype.** Given the KV260's
   real resource budget (~3.6M LUTs, ~4M flip-flops), is the
   factorized-curriculum AND-composition deployable as currently
   specified, or does Phase-3+4 need to be downsized for hardware?
   E.g., $k_{\max} = 5$ instead of $15$.

### Format

Quantitative bounds in closed form whenever possible. Where rigorous
proofs aren't feasible in this session, mark intuition vs. theorem.

If your derivation finds that **the factorized curriculum is
fundamentally incompatible with FPGA mixing** (e.g., super-polynomial
$\tau_{\text{mix}}$ in realistic regimes), please say so directly.
The position paper benefits from honest characterization of where
the Phase-3+4 architecture has an FPGA-deployment caveat.

---

## Predictions footer (do NOT paste — for cross-validation only)

After today's pattern of HIGH-confidence errors on architectural
specifics, predictions are intentionally cautious:

- **Q1(a) level-set connectivity:** AND-composition preserves convexity
  of the *feasible region* but the *level sets* under the Riemannian
  metric may develop saddle points. Likely "preserved up to a constant
  factor in spectral gap." Confidence: LOW.
- **Q1(b) spectral gap:** $\lambda_{\text{AND}} \geq \lambda_1 \cdot \sin^k(\theta_F)$
  (multiplicative, Friedrichs-angle penalty). Confidence: LOW.
- **Q1(c) worst-case mixing:** $\tau_{\text{mix}}^{\text{AND}} = O(k \cdot \tau_1 \cdot \sin^{-2k}(\theta_F))$
  — polynomial in $k$, but exponential in $\sin^{-1}(\theta_F)$.
  Confidence: LOW.
- **Q2(a) bottlenecks:** YES, heterogeneous $\sigma_{t,i}$ creates
  bottlenecks where one strict verifier carves out narrow passages.
  Mixing degrades polynomially with $\max_i (1 - \sigma_{t,i})^{-1}$.
  Confidence: MEDIUM.
- **Q2(b) bang-bang re-mixing:** YES, transition penalty exists;
  approximately $O(n)$ extra steps per phase transition. Confidence: LOW.
- **Q2(c) per-verifier parallel chains:** PARALLEL chains better;
  $O(k)$ speedup with FDR-OR sync. Confidence: MEDIUM.
- **Q3(a) spin count:** Additive $O(\sum_i n_i)$. Multiplicative
  would be a deal-breaker; Carnot's constraint-encoded verifiers
  are independent. Confidence: HIGH (this is one I'm confident about).
- **Q3(b) bit width:** Minimum $\lceil\log_2 k + \log_2(1/\varepsilon_{\text{mix}})\rceil$
  bits for the AND-composition energy. Confidence: LOW.
- **Q3(c) sampler choice:** SGLD on convex parts, tempered on
  bang-bang transition; SB only acceptable when $\sigma_{t,i}$ is
  uniform. Confidence: LOW.
- **Q4(a) universal mixing condition:** Cheeger condition with
  $h \geq c/k$ (linear in $k$, not exponential). Confidence: LOW.
- **Q4(b) substrate advantages:** Extropic's continuous-energy
  sampler benefits from Continuous Ising-Rank; photonic gets
  natural OR-composition cheap, AND-composition still hard.
  Confidence: LOW.
- **Q4(c) portability theorem:** "The factorized-curriculum
  AND-composed energy is universally Ising-class samplable with
  polynomial mixing whenever $\max_i \sigma_{t,i} < 1 - 1/k$."
  Confidence: LOW.
- **Q5(a) critical $k^*$:** $k^*(n) \sim n^{1/2}$ — beyond
  $\sqrt{n}$, AND-composition kills polynomial mixing.
  Confidence: LOW.
- **Q5(b) sampling vs. renormalization:** Sampling error dominates
  $C_Z$ when $\varepsilon_{\text{mix}} > \varepsilon^*$. In high-
  quality regimes ($\varepsilon^* \ll 1$), sampling is the bottleneck.
  Confidence: LOW.
- **Q5(c) KV260 deployment:** $k_{\max} = 15$ likely fits in 3.6M LUTs
  with $n_i \sim 10^3$ each (~$15 \cdot 10^3 \cdot 100$ LUTs = $1.5$M).
  Tight but feasible. Confidence: MEDIUM.

**What we want most from Deep Think:**
- A clean closed form for the spectral gap of $E_{\text{AND}}$ as a
  function of $k$ and $\theta_F$ (Q1b).
- Honest answer on whether factorized curriculum is FPGA-deployable
  or requires $k_{\max}$ reduction (Q5c).
- The hardware-portability theorem statement (Q4c) for paper insertion.
- An honest "not deployable" finding if the heterogeneous landscape
  fundamentally breaks polynomial mixing — that's a more valuable
  answer than a contrived patch.

After 4 rounds today, this round avoids HIGH-confidence architectural
predictions almost entirely. Q3(a) is the only HIGH-confidence
prediction (additive spin count) — and even that is a rough
hardware-encoding claim, not an architectural prescription.
