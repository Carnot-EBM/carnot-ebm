# Deep Think Prompt: Kinematic Layer Routing — Dispersion Relation

**Status:** Ready to send. Closes a Round-4 (pre-Zenil) speculative
question about transformer feature propagation that was abandoned
when the AUROC bug pivoted us to autonomy work.

**Date drafted:** 2026-04-29

**Why this question:** Carnot's Phase 1 Tier-0i probes diagnose
internal model state by attaching to specific transformer layers.
Currently we sweep all $N \in [32, 80]$ layers per probe candidate
($O(N)$ placement). If transformer features propagate as coherent
wave packets through layer-time, we can predict the *single layer*
$\ell_{\max}(\tau)$ where a feature first becomes diagnostic —
$O(1)$ placement. Major Phase 1 product feature.

**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip the predictions footer.

---

## Prompt to send (verbatim)

### Setup

Consider a stack of $N$ transformer attention layers
$\{L_1, \ldots, L_N\}$ processing a sequence of token features
$x_t \in \mathbb{R}^d$ for $t \in [1, T]$ (token position).

We wish to model the propagation of a feature $\phi(\tau, t)$ through
the joint *layer-time* (depth, denoted $\tau \in [0, N]$) ×
*token-time* (sequence position, denoted $t \in [0, T]$) plane.

**Hypothesis (to be tested):** in the high-resolution limit, $\phi$
satisfies an advection-dispersion PDE

$$
\partial_\tau \phi + v(\tau, t)\, \partial_t \phi = D(\tau, t)\, \partial_t^2 \phi + \mathcal{N}[\phi]
$$

where $v$ is a layer-dependent advection velocity (how fast features
"flow" through token positions per layer), $D$ is a diffusion
coefficient (how fast features spread along token-time), and
$\mathcal{N}$ is a nonlinear self-interaction term from the attention
softmax.

The standard attention update at layer $\ell$ is

$$
x_t^{(\ell+1)} = \sum_{s} \mathrm{softmax}\!\left(\frac{q_t^{(\ell)} \cdot k_s^{(\ell)}}{\sqrt{d_k}}\right) v_s^{(\ell)} + x_t^{(\ell)}
$$

(with residual connection). Treat $q, k, v$ as the trained projections.
Use $L^2(\mathbb{R})$ norm in token-time throughout; make explicit any
KL ↔ $L^2$ conversions where used.

### Question 1: Dispersion relation $\omega(k)$ for transformer attention

Linearise the attention update around a smooth feature distribution
$\phi(\tau, t)$ in the small-perturbation regime. By Fourier-transforming
in token-time ($k$ = wave number) and treating layer-depth as
continuous time $\tau$:

(a) Derive the explicit dispersion relation $\omega(k)$ such that
   plane-wave solutions $\phi \propto \exp(i k t - i \omega \tau)$
   satisfy the linearised PDE. Express $\omega$ in terms of:
   - The attention softmax temperature $\beta = 1/\sqrt{d_k}$.
   - The Q/K rank $r_{QK}$ (effective dimension of the query/key
     projection).
   - The token-position distribution of attention weights at layer
     $\ell$ (i.e., the empirical attention spread).

(b) **Phase velocity vs group velocity.** Compute both
   $v_\phi(k) = \omega(k)/k$ and $v_g(k) = d\omega/dk$ from (a).
   Where they differ, features at different wavenumbers travel at
   different rates — implying *layer-dependent feature dispersion*
   (the hallmark of true wave-packet propagation, not pure advection).
   Quantify the regime where $v_g \neq v_\phi$.

(c) **Imaginary part (decay).** If $\omega(k)$ has non-zero imaginary
   part, features decay along layer-time at rate $|\Im \omega(k)|$.
   What's the decay length scale $L_{\text{decay}}(k)$ in layers?
   This sets the maximum layer-depth a feature can travel before
   being "forgotten" by attention.

### Question 2: Coherent wave-packet conditions

Pure dispersion gives plane waves; for $\ell_{\max}$ prediction we need
*localised* wave packets that propagate coherently through layers.

(a) Under what conditions on the attention weights does $\phi(\tau, t)$
   form a coherent wave packet — i.e., a localised $\phi(\tau, \cdot)$
   that translates in $t$ as $\tau$ increases without significant
   broadening?

(b) **Sufficient condition via concentrated softmax.** I conjecture
   wave-packet propagation requires the attention softmax to be
   concentrated on a narrow token-position window (low-temperature
   limit, or strong Q-K alignment). Make this precise: what's the
   threshold on the softmax effective bandwidth
   $\sigma_{\text{att}}(\ell)$ for coherent propagation?

(c) **Multi-modal attention.** Real transformers have multi-head
   attention. Each head has its own dispersion. Does coherent
   propagation survive the head-summation? Or does $\phi$
   decompose into independent wave packets per head?

### Question 3: Predictive equation for $\ell_{\max}(\tau)$ — Tier-0i probe placement

Carnot's Tier-0i probes attach to a specific transformer layer
$\ell$ and read out a feature linear projection $\theta^T x^{(\ell)}_t$.
The probe's *diagnostic power* depends on the feature being
*present* at layer $\ell$ — not yet decayed, not yet absorbed into
deeper computation.

For a target feature class (defined by initial attention pattern
$\phi(0, \cdot)$ at $\tau = 0$), predict the *single layer*
$\ell_{\max}$ where the feature first becomes diagnostic.

(a) **From dispersion to placement.** Combining the dispersion
   relation from Q1 with the coherence conditions from Q2, derive
   the explicit map
   $\ell_{\max} = f(\phi(0, \cdot), v_g, D, L_{\text{decay}})$
   that gives the optimal probe-placement layer.

(b) **Falsifiability.** What's the *experiment* that would refute
   this prediction? Specifically, give a metric that should be
   measurable on a real transformer (e.g., GPT-style or Qwen-style
   model) such that:
   - If the metric shows the predicted $\ell_{\max}$ behaviour ⟹ the
     wave-packet model holds.
   - If the metric shows random scatter or strong layer-aliasing ⟹
     the model is wrong.

(c) **Multi-feature probe placement.** When probing for $K$
   different features simultaneously, can we place all $K$ probes at
   their individual $\ell_{\max,i}$ values, or do feature
   interactions (cross-coupling in attention heads) require a
   joint optimisation?

### Final integration: deployable Tier-0i placement protocol

Synthesise into a deployable protocol of the form:

> "For Carnot's Tier-0i probe placement on a transformer with $N$
>  layers, $d_k$ key dimension, attention temperature $\beta$, and
>  measured per-layer $\sigma_{\text{att}}(\ell)$ profile, place
>  probe $i$ at layer $\ell_{\max,i} = h(\phi_i, \text{attention profile}, ...)$
>  with predicted diagnostic power $D_i = g(\ell_{\max,i})$.
>  Replaces $O(N)$ sweep with $O(1)$ analytic placement."

Provide $h$ and $g$ explicitly. Identify the regime where the
analytic placement is reliable vs the regime where empirical
sweeping is still needed.

---

## Internal cross-validation predictions (DO NOT PASTE)

### Q1 predictions

(a) Dispersion: $\omega(k) \sim \beta \cdot k^2$ (diffusive — feature
    spread grows as $\sqrt{\tau}$) plus $\beta \cdot k$ from attention's
    correlation length (advective). Mixed regime.
(b) $v_g \neq v_\phi$ when attention is non-uniform — i.e., in the
    interesting regime. Phase-group difference of order $\sigma_{\text{att}}$.
(c) Decay length $L_{\text{decay}}(k) \sim \exp(\beta \sigma_{\text{att}}^2 k^2)$ — exponentially
    long for low-frequency features, short for high-frequency.

### Q2 predictions

(a) Wave-packet coherence requires $\sigma_{\text{att}} \cdot k \lesssim 1$
    — attention bandwidth narrower than feature wavelength.
(b) Threshold: $\sigma_{\text{att}}(\ell) < 1 / |k_0|$ where $k_0$ is
    the feature's central wavenumber.
(c) Multi-head: each head's wave packet propagates independently;
    head-summation is approximately linear so coherence survives.

### Q3 predictions

(a) $\ell_{\max} \approx L_{\text{decay}}(k_0)$ — feature decays before
    next layer can absorb it. So $\ell_{\max}$ is largely determined
    by the decay length at the feature's central wavenumber.
(b) Test: measure feature norm $\|\phi(\tau, \cdot)\|$ across layers
    on a held-out probe target. The predicted $\ell_{\max}$ should
    coincide with the layer of maximum norm. If norm peaks at a
    different layer or is monotone across all layers ⟹ model wrong.
(c) Multi-feature: independent placement should work for typical
    cases; cross-coupling becomes significant only at high feature
    density (many probes at adjacent layers).

### Final integration prediction

Carnot can replace $O(N)$ probe sweeping with $O(1)$ placement for
the *single-feature* case. Multi-feature placement may require
$O(K \log K)$ joint optimisation in the worst case. This would
shrink Phase 1 probe-placement compute by ~$30\times$ (typical
$N = 32$ → constant 1 placement evaluation).

If Deep Think confirms the dispersion form and the wave-packet
threshold, ships as `python/carnot/probes/kinematic_placement.py`
with falsifiable empirical-validation experiment on a Qwen3.5-0.8B
or Gemma3-E2B model.

## Action plan after response

1. **If Q3(a) gives a clean closed-form $\ell_{\max}$:**
   - Ship `python/carnot/probes/kinematic_placement.py`.
   - Add to change proposal `kinematic-layer-routing-tier0i.md` (new).
   - Replace the existing $O(N)$ sweep in Tier-0i probe deployment.

2. **If Q3(b) gives a falsifiability test:**
   - Run on Qwen3.5-0.8B as a fast validation experiment.
   - Confirm or refute on real transformer architecture.

3. **If Q1 reveals dispersion is *not* present** (e.g., pure
   advection without diffusion):
   - The wave-packet hypothesis is too strong; revise to
     pure-advection model.
   - $\ell_{\max}$ becomes deterministic from feature locality alone.

4. **Update memory entry** `project_kinematic_layer_routing.md` with
   confirmed/refuted bounds.

5. **Position-paper consequence:** if the model holds, it's a
   contribution to the *interpretability* literature — Tier-0i probe
   placement reduces from empirical to analytic.
