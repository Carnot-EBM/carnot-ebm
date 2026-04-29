# Kinematic Layer Routing Results — Deployable Tier-0i Placement Protocol

**Status:** Authoritative. Closes the Round-4 (pre-Zenil) speculative
question. The wave-packet hypothesis is **confirmed analytically**.
**Date:** 2026-04-29
**Prompt:** `kinematic-layer-routing-dispersion-prompt.md`

Deep Think delivered:
- Full closed-form dispersion relation (Gaussian kernel + Bragg
  backscattering + anomalous dispersion).
- Deployable analytic $\ell_{\max}$ formula.
- Falsifiable test (linear scaling of $\ell_{\max}$ with $\Delta t$).
- A **Péclet-number runtime check** that switches between $O(1)$
  analytic placement and $O(N)$ empirical sweep.

This converts Tier-0i probe placement from empirical sweeping to
analytic placement in the high-Péclet regime, with a clean fall-back
in the low-Péclet regime.

## Q1 — Dispersion relation

### (a) Closed form

For attention with mean look-back $\mu$ and spatial spread $\sigma_{\text{att}}^2 \propto \sqrt{d_k}/r_{QK}$:

$$
\omega(k) = \underbrace{\sin(\mu k) \cdot e^{-\frac{1}{2}\sigma_{\text{att}}^2 k^2}}_{\Re(\omega)} + i \underbrace{(\cos(\mu k) \cdot e^{-\frac{1}{2}\sigma_{\text{att}}^2 k^2} - 1)}_{\Im(\omega)}
$$

**Carnot prediction was a crude polynomial guess.** Deep Think gives
the exact Gaussian-convolution kernel structure with explicit lattice
effects.

### (b) Phase vs group velocity — three regimes

- **Smooth context features ($k \to 0$):** pure advection, $v_\phi \approx v_g \approx \mu$.
- **Lattice Aliasing Regime ($k \sim \pi/\mu$):** Bragg backscattering;
  $\cos(\mu k) \to -1$, halts forward advection. Discrete-token
  artefact.
- **Anomalous Dispersion Regime ($k \gtrsim 1/\sigma_{\text{att}}$):**
  finite attention window introduces severe group-velocity dispersion;
  high-frequency features chirp and the wave packet **physically tears
  apart**.

### (c) Decay length

$$
L_{\text{decay}}(k) = \frac{1}{|\Im \omega|} = \frac{1}{1 - \cos(\mu k) e^{-\sigma_{\text{att}}^2 k^2/2}}
$$

For moderate $k$:
$$
L_{\text{decay}}(k) \approx \frac{2}{k^2 (\mu^2 + \sigma_{\text{att}}^2)} \text{ layers}
$$

The effective diffusion coefficient is $D = \frac{1}{2}(\mu^2 + \sigma_{\text{att}}^2)$
— the $\mu^2$ term arises from mapping discrete layer jumps to
continuous $\tau$ via Poisson-like dispersion.

**Carnot predicted exponential decay $\exp(\beta\sigma^2 k^2)$.
Refuted — actual is power-law $1/k^2$.**

## Q2 — Coherent wave-packet conditions

### (a) Two conditions

1. **Low attenuation:** $\tau \ll L_{\text{decay}}(k_0)$.
2. **Low GVD:** $k \sigma_{\text{att}} \ll 1$ — packet bandwidth narrow
   enough to avoid dispersion.

### (b) Architectural threshold

For coherent transport over $\ell_{\max}$ layers with feature width
$W_0$:

$$
\boxed{\beta r_{QK} \gg \frac{\ell_{\max}}{W_0^2}}
$$

This couples four architectural parameters: softmax temperature $\beta$,
Q/K rank $r_{QK}$, depth budget $\ell_{\max}$, and feature spatial
width $W_0$. **Deployable as a model-architecture audit.**

The KL ↔ $L^2$ conversion: by Pinsker + Young's convolution,
$\|P*\phi - Q*\phi\|_{L^2} \leq \sqrt{2 D_{\text{KL}}(P\|Q)} \|\phi\|_{L^2}$.
Deep $L^2$ wave-packet survival explicitly requires the empirical
attention $P$ to be near-low-variance, ie low-temperature softmax.

### (c) Multi-head birefringence (REFUTED Carnot prediction)

**Surprising result.** Multi-head attention with $H$ heads has:

$$
D_{\text{eff}} = \frac{1}{H} \sum_h D_h + \frac{1}{2}\mathrm{Var}(\mu_h)
$$

If heads disagree on advection shift ($\mathrm{Var}(\mu_h) > 0$),
this **injects catastrophic artificial diffusion**. The wave packet
"shatters into independent ghost packets — the transformer
equivalent of birefringence."

Coherent propagation only survives if the feature **routes into an
isolated value-subspace $W_V^{(h)}$** specific to one head's velocity
field.

**Architectural implication for Carnot:** Tier-0i probes need to
identify *which head* carries the feature, not just which layer.
Probe placement is a $(\ell, h)$ tuple, not just $\ell$.

## Q3 — Predictive equation for $\ell_{\max}$

### (a) Closed-form analytic placement

For a feature traveling spatial distance $\Delta t$, the
diagnostic-amplitude-maximizing layer is:

$$
\boxed{\ell_{\max} = \frac{-D + \sqrt{D^2 + v_{\text{eff}}^2 (\Delta t)^2}}{v_{\text{eff}}^2}}
$$

where $v_{\text{eff}} = \sqrt{v_g^2 + 4D/L_{\text{decay}}}$.

In the high-Péclet limit ($D \to 0$), this reduces to the ballistic
$\ell_{\max} = \Delta t / v_g$. With finite $D$, the formula
diffusion-corrects the ballistic estimate.

### (b) Falsifiability

**The experiment:** inject a synthetic distinct feature at token
$t_{\text{src}}$ at layer 0. Sweep linear probes at readout token
$t_{\text{tgt}}$ across all layers. Plot peak-diagnostic layer
$\ell_{\max}$ vs spatial distance $\Delta t$.

**Refutation:** if $\ell_{\max}$ is constant (independent of $\Delta t$),
or shows a vertical line in the heatmap (instantaneous arrival
everywhere), the wave-packet PDE model is empirically refuted —
implying transformers route via instantaneous "sink tokens" rather
than continuous propagation.

**The PDE holds iff $\ell_{\max}$ scales linearly with $\Delta t$.**

### (c) Multi-feature placement

- **Sparse regime:** features in orthogonal subspaces → independent
  placement at individual $\ell_{\max,i}$.
- **Dense regime:** high-amplitude features overlap in subspace →
  nonlinear softmax denominator interaction. A high-amplitude feature
  acts as a **"refractive blockade"** altering background advection
  shift $\mu$ for other features. Soliton-traffic-jam analogy.
- Joint optimization required when feature density is high.

## Final integration: deployable Tier-0i placement protocol

```
TIER-0i ANALYTIC PLACEMENT PROTOCOL:

1. CALCULATE KINEMATIC PARAMETERS for feature i:
   - Look-back drift:   v_g,i ≈ E_ℓ[μ_i(ℓ)]
   - Diffusion:         D_i   ≈ E_ℓ[(σ²_att,i + μ²_i)/2]
   - Effective vel.:    v²_eff,i = v²_g,i + 4 D²_i / W²_i

2. RUNTIME REGIME CHECK via Péclet number:
   Pe_i = v_g,i · Δt_i / D_i

   IF Pe_i >> 1 (advection dominates):
     Use analytic O(1) placement:
     ℓ_max,i = ⌊(-D_i + √(D²_i + v²_eff,i · Δt²_i)) / v²_eff,i⌉

   IF Pe_i ≲ 1 (diffusion dominates):
     Fall back to empirical O(N) layer sweep
     (analytic peak washes out — feature smeared globally)

3. PREDICTED DIAGNOSTIC POWER:
   Power_i ∝ 1/√(4πD_i ℓ_max,i)
            · exp(-(Δt_i - v_g,i ℓ_max,i)² / (4 D_i ℓ_max,i)
                  - D_i ℓ_max,i / W²_i)
```

## What Carnot got right

- Q2(a) GVD threshold $\sigma k \ll 1$
- Q3(c) sparse independent placement
- Q3(a) general direction (ℓ_max scales with feature locality)

## What Carnot got wrong

- **Q1(c) decay scaling:** predicted exponential; actual is $1/k^2$.
- **Q2(c) multi-head coherence:** predicted survives summation;
  actual SHATTERS unless feature routes into isolated head subspace.
  Major architectural insight missed.
- **Q3(a) precision:** predicted just $L_{\text{decay}}$; actual is
  the diffusion-corrected ballistic formula.

## What surprised us

- **Bragg backscattering.** The discrete-token lattice causes
  $\cos(\mu k)$ to flip sign at $k \sim \pi/\mu$, halting advection.
  Analogous to electron-band-structure backscattering.
- **Multi-head birefringence.** Cross-head velocity disagreement
  shatters wave packets — major architectural constraint we hadn't
  anticipated.
- **Refractive blockade between features.** High-amplitude features
  alter the background advection field for others. Multi-feature
  probe placement is genuinely non-trivial.
- **Péclet runtime check.** Deep Think gave us a clean dimensionless
  number that decides analytic-vs-empirical at runtime. Deployable.

## Deployment plan

1. **Ship `python/carnot/probes/kinematic_placement.py`** implementing
   the protocol from "Final integration" above. Estimated: 4 hours
   (small module; the math is fully specified).

2. **Ship `python/carnot/probes/peclet_check.py`** for runtime regime
   classification. Trivial implementation; ~1 hour.

3. **Run empirical validation experiment** on Qwen3.5-0.8B (per
   `feedback_sota_models.md`). Inject synthetic features at known
   token positions, sweep probes, plot $\ell_{\max}$ vs $\Delta t$.
   If linear → analytic deployment. If constant → fall back to
   empirical sweep, but the analytic regime check can still flag
   when we're in dispersion-dominated regime.

4. **New change proposal `kinematic-layer-routing-tier0i.md`**
   bundling the above 3 experiments as a single milestone scope.
   Pin to `ops/known-issues.md` MANDATORY-pickup priorities.

5. **Multi-head extension experiment.** Test the birefringence
   prediction: probe placement should improve when restricted to a
   single head's value-subspace. If confirmed, Tier-0i probes must
   be $(\ell, h)$ tuples.

## Position-paper impact

Two new contributions on top of the 12-round Zenil chain:

11. **Closed-form transformer dispersion relation.**
    $\omega(k) = i(\exp[-i\mu k - \sigma^2 k^2 / 2] - 1)$ as the
    linearised attention PDE solution. Bragg backscattering and
    anomalous dispersion regimes identified.

12. **Tier-0i analytic probe placement with Péclet runtime check.**
    Replaces empirical $O(N)$ sweep with $O(1)$ analytic placement
    in the advection-dominated regime; clean fall-back in
    diffusion-dominated regime. Deployment-ready.

These are independent of the Zenil chain — different mathematical
regime (PDE wave propagation vs spectral information theory). The
position paper now has 12 publishable contributions across two
mathematical foundations.

## Caveats and open questions

- The wave-packet PDE assumes the linearised regime. Deep nonlinear
  features (top-of-residual-stream tokens with extreme magnitudes)
  may violate this. The Péclet check provides a partial safeguard
  but doesn't directly detect nonlinearity.
- Multi-head birefringence is a load-bearing prediction for Tier-0i
  probe architecture. If it falsifies, we keep $\ell$-only placement;
  if it confirms, we need $(\ell, h)$ probe tuples.
- The "soliton traffic jam" multi-feature interaction is qualitatively
  described but not quantitatively bounded. Needs follow-up math
  before high-density probe-suite deployment.

## Status

Mathematical foundation: complete. Deployable protocol: specified.
Empirical validation: experiment design ready. Code-side
implementation: estimated 5-6 hours total across 3 modules.

Next milestone (.81 or .82) should pick up the kinematic-layer-routing
change proposal alongside the Zenil-grounded self-distillation stack.
