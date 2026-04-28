# Deep Think Round-11 Prompt — Autoencoder Bottleneck Objective for Verifier-Filtered Self-Distillation

**Status:** Ready to send. Asks for the theory underlying Step 0 of
the 5-step Phase 3 architecture (latent autoencoder bottleneck).
**Date drafted:** 2026-04-29
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`. Skip the predictions footer.

---

## Prompt to send (verbatim)

### Background (use as established premises)

We are deploying a hardware-accelerated energy-based-model (EBM)
self-distillation system. From a prior 10-round derivation chain,
the following are established:

1. **Verifier-filtered restorative force.** A verifier with energy
   $E$ filtering candidates from base model $Q_t$ via
   $w(x) \propto \exp(-E(x)/T)$ provides a structural restorative
   force
   $\Phi = \frac{1}{\delta_t} \mathrm{Cov}_{Q_t}\!\left(\frac{Q_t}{\mu_P}, E\right) = \mathrm{Cov}_{\mu_P}(g_t, E)$
   where $g_t = (Q_t - \mu_P)/(\delta_t \mu_P)$ is the normalised
   residual.

2. **Multi-verifier rotation defence.** Single-verifier filters stall
   at an orthogonality plateau $\delta_{\text{plateau}}^{\text{single}}$.
   The unique mathematically-rigorous defence is pre-emptive rotation
   among $k \geq 15$ semantically-orthogonal verifiers
   $\{E_1, \ldots, E_k\}$ with pairwise Friedrichs angle
   $\theta_F \geq \pi/4$. Verifier suites are constructed by
   AND-composition ($P_{i \wedge j} = \lim_n (P_i P_j)^n$, the von
   Neumann intersection projection) of $m \geq 6$ base verifiers.

3. **Phase 2 distillation bound.** Mapping a continuous EBM to an
   Ising spin system at bit-width $k$ via Gray-code encoding incurs
   distillation loss $\mathcal{L} \leq dL^2 / (24 T^2) \cdot 2^{-2k}$
   where $d$ is ambient dimension, $L$ is the Lipschitz constant of
   the energy. The verifier's distilled restorative force satisfies
   $|\Phi^{(k)} - \Phi| \lesssim \sigma_F \sqrt{2 \mathcal{L}}$.

4. **Foundation-model regime.** When the ambient dimension $d$ is
   large (e.g., $d = D_{\text{spec}} \sim 10^4$ for token spaces),
   the bit-width requirement $k \geq \frac{1}{2} \log_2(dL^2 / 24 T^2 \varepsilon_{\text{cap}})$
   exceeds typical FPGA capabilities. The proposed architectural fix
   is a **latent autoencoder bottleneck:** project $x \mapsto z = \text{Enc}(x) \in \mathbb{R}^{d_{\text{latent}}}$
   with $d_{\text{latent}} \ll D_{\text{spec}}$, run the EBM and
   verifier filter on $z$, decode samples back to $x$ for verifier
   evaluation.

The remaining gap is constructive: **what is the right autoencoder
objective so the latent bottleneck preserves verifier-relevant
structure?**

Use $L^2(\mu_P)$ throughout; make explicit any
$\mathrm{KL} \leftrightarrow L^2(\mu_P)$ conversions.

### Question 1: Verifier-preserving autoencoder objective

Given a verifier suite $\{E_1, \ldots, E_k\}$ acting on outputs
$x \in \mathcal{X}$, derive an autoencoder objective $\mathcal{J}(\text{Enc}, \text{Dec})$
such that the latent representation $z = \text{Enc}(x)$ preserves
verifier-relevant structure. Concretely, define $\Phi^{(k)}_{\text{latent}}$
as the restorative force computed in latent space (with verifiers
acting on $\text{Dec}(z)$) and $\Phi^{(k)}_{\text{ambient}}$ as the
ambient-space force.

(a) **Minimal sufficient objective.** What is the minimal
    $\mathcal{J}$ that guarantees
    $|\Phi^{(k)}_{\text{latent}} - \Phi^{(k)}_{\text{ambient}}| \leq O(\sqrt{\mathcal{L}})$
    where $\mathcal{L}$ is the Round-10 distillation loss bound?
    Specifically: is plain MSE reconstruction
    $\mathcal{J} = \mathbb{E}\|x - \text{Dec}(\text{Enc}(x))\|^2$
    sufficient, or does $\mathcal{J}$ need additional terms — e.g.,
    a *verifier-consistency* loss
    $\mathbb{E}\|E_i(x) - E_i(\text{Dec}(\text{Enc}(x)))\|^2$ — to
    preserve the relevant geometry?

(b) **Independent vs. joint training.** Can $\mathcal{J}$ be
    optimised over the autoencoder *independently* of the verifiers
    $\{E_i\}$, or must the autoencoder be co-trained with the
    verifier suite to align latent geometry with verifier kernels?
    Provide a precise statement: under what conditions on the
    autoencoder family $\mathcal{F}_{\text{AE}}$ and the verifier
    family $\{E_i\}$ does the decoupled training scheme suffice?

(c) **Minimum $d_{\text{latent}}$.** What is the minimum
    $d_{\text{latent}}$ that preserves the multi-verifier rotation
    defence? Specifically, the rotation defence relies on the joint
    Dixmier angle $\sin^2 \Theta_k = 1 - \|\frac{1}{k} \sum_i P_i\|_{\text{op}, (\bigcap N_i)^\perp}$
    being bounded above zero. After projection to $\mathbb{R}^{d_{\text{latent}}}$,
    what is the analogue $\Theta_k^{(\text{latent})}$, and what's the
    minimum $d_{\text{latent}}$ that preserves
    $\Theta_k^{(\text{latent})} \geq \alpha \cdot \Theta_k^{(\text{ambient})}$
    for some retention factor $\alpha \in (0, 1]$?

### Question 2: Information-geometric framing

The autoencoder bottleneck imposes a *manifold restriction* on the
EBM's support — the EBM-generated distribution lies on
$\text{Dec}(\mathbb{R}^{d_{\text{latent}}}) \subset \mathcal{X}$.
This manifold can be misaligned with the truth manifold $\mathcal{T}$
where $\mu_P$ concentrates.

(a) **Manifold-alignment condition.** Derive the precise condition on
    $\text{Dec}(\mathbb{R}^{d_{\text{latent}}})$ relative to the
    support of $\mu_P$ for the autoencoder bottleneck NOT to introduce
    a new failure mode (analogous to the orthogonality stall but
    occurring at the bottleneck rather than the verifier).

(b) **Stein-discrepancy proxy.** Can the verifier-preservation
    condition from Q1(a) be reformulated as bounded Stein discrepancy
    between the latent-pushed measure and the ambient $\mu_P$? If yes,
    this gives a deployable training-time signal (the kernelised
    Stein discrepancy is computable from samples).

(c) **VAE vs deterministic AE.** Does the bottleneck need to be
    stochastic (VAE-style with KL regulariser) to preserve the
    measure-theoretic structure required by the rotation defence,
    or does a deterministic encoder-decoder suffice? Specifically:
    when can a deterministic encoder fail to preserve the $\mu_P$
    null-space structure that single-verifier orthogonality relies on?

### Question 3: Practical training algorithm

Synthesise Q1 and Q2 into a deployable algorithm:

(a) **Loss decomposition.** Write the explicit training loss as a
    sum of named terms (reconstruction, verifier-consistency, manifold-
    alignment, stochasticity-regulariser if needed), with each term's
    coefficient identified as a function of the desired retention
    parameters ($\alpha, \mathcal{L}_{\text{tgt}}, \varepsilon_{\text{cap}}$).

(b) **Sample complexity.** How many training samples does the
    autoencoder need to converge to a verifier-preserving solution?
    Provide a Rademacher-complexity-style bound.

(c) **Cold-start problem.** The autoencoder must be trained *before*
    the verifier-filtered self-distillation begins (Step 0 of the
    architecture), but the verifiers' "useful" subspace is only
    fully revealed *during* self-distillation. Is there a bootstrapping
    procedure (e.g., an initial round with reduced $k$ and looser
    $\Theta_k$ that progressively tightens) that allows the
    autoencoder to be refined alongside the verifier suite?

### Final integration: deployable autoencoder protocol

Synthesise into a single deployable Step-0 protocol:

> "For Phase 3 self-distillation with verifier suite $\{E_1, \ldots, E_k\}$
>  on ambient space $\mathcal{X} \subset \mathbb{R}^{D_{\text{spec}}}$,
>  pre-train the autoencoder $(\text{Enc}, \text{Dec})$ on objective
>  $\mathcal{J} = \lambda_{\text{rec}} \cdot \text{rec} + \lambda_{\text{ver}} \cdot \text{ver-consistency} + \ldots$
>  with $d_{\text{latent}} \geq d_{\text{latent}}^{\text{min}}(D_{\text{spec}}, k, \theta_F, \varepsilon_{\text{cap}})$
>  for $M_{\text{AE}} \geq M_{\text{AE}}^{\text{min}}(\ldots)$ samples,
>  achieving verifier-preservation
>  $|\Phi^{(\text{latent})} - \Phi^{(\text{ambient})}| \leq \tau$."

Provide $\mathcal{J}$, $d_{\text{latent}}^{\text{min}}$, and
$M_{\text{AE}}^{\text{min}}$ explicitly.

---

## Internal cross-validation predictions (DO NOT PASTE)

### Q1 predictions

(a) Plain MSE reconstruction is INSUFFICIENT. Need verifier-consistency
    term $\mathbb{E}_x \sum_i (E_i(x) - E_i(\text{Dec}(\text{Enc}(x))))^2$
    weighted by $\lambda_{\text{ver}} \sim L_E^2 / \tau^2$ where
    $L_E$ is the verifier's Lipschitz constant.

(b) Independent training is sufficient when verifier kernels span a
    *fixed* subspace; co-training is required when verifiers are
    auto-generated per-task (Q2b mutant generators from Round-9).

(c) $d_{\text{latent}}^{\text{min}} \sim k$ (the number of verifiers)
    plus the intrinsic dimension of $\mu_P$'s support. For Carnot's
    setting: $d_{\text{latent}} \in [k, 2k] \approx [15, 30]$.

### Q2 predictions

(a) Manifold-alignment: $\text{Dec}(\mathbb{R}^{d_{\text{latent}}})$
    must contain $\mathcal{T}$ (the truth manifold). Otherwise a new
    failure mode at the bottleneck — model can't even represent
    correct outputs.

(b) Stein-discrepancy yes — kernelised Stein discrepancy KSD is
    computable from samples and bounds the verifier-relevant geometry.

(c) Deterministic AE fails when $\mu_P$ has measure-zero subspaces
    that the verifier rotation depends on. VAE with appropriate
    KL regulariser should preserve.

### Q3 predictions

(a) Loss decomposition:
    $\mathcal{J} = \lambda_{\text{rec}} \mathbb{E}\|x - \hat x\|^2$
    $+ \lambda_{\text{ver}} \mathbb{E} \sum_i (E_i(x) - E_i(\hat x))^2$
    $+ \lambda_{\text{stein}} \cdot \text{KSD}(\hat \mu_P \| \mu_P)$
    $+ \lambda_{\text{kl}} \cdot \mathrm{KL}(q(z|x) \| p(z))$
    [for VAE]

(b) Sample complexity: $M_{\text{AE}} = O(d_{\text{latent}} \cdot k / \tau^2)$
    via Rademacher.

(c) Bootstrapping: yes — start with $k_0 = 3$ (Round-9 minimum),
    train AE, then progressively expand to $k = 15$, refining AE
    each iteration via warm-restart.

### Final integration prediction

Concrete: $d_{\text{latent}} \geq 30$, $M_{\text{AE}} \geq 10^6$
samples for $\tau = 0.1$ verifier-preservation. Loss has 3 terms
(rec + ver-consistency + KL-reg for VAE).

If Round-11 confirms, add Exp G to the change proposal:
implementation of the Step 0 latent autoencoder with the derived
loss; empirical validation of $d_{\text{latent}}^{\text{min}}$
on a synthetic toy problem before scaling.
