# Deep Think responses — ICLR 2026 follow-up

Live record of Deep Think's verdicts on the prompts drafted in
`iclr26-deep-think-prompts.md` + `iclr26-deep-think-prompts-batch2.md`.
Append responses as they arrive; each response should drive an
integration-plan update + .NNN priority adjustment.

---

## DT-7 response (received 2026-05-08, ~17:00Z) — **VERDICT: VENDOR THRML DIRECTLY**

**Question summary:** Can MCMC Layers' single-site MH (Algorithm 1) match
THRML's block-Gibbs sampling at finite K, fixing the .119 KL=0.17
mismatch?

**Verdict: NO. Adopt option (i) — vendor THRML's block-Gibbs directly.**

### Key findings

**(a) Structural mismatch at finite K confirmed.** The MH and Gibbs
transition kernels are mathematically distinct and do not commute, even
when targeting the same Boltzmann measure π:

- **Per-step TV**: MH accepts energy-lowering moves with probability 1;
  Gibbs (Heat-Bath) accepts with sigmoid probability `1 / (1 + e^{−Δ/t}) < 1`.
  Off-diagonal transition entries differ from K=1 onward.
- **Sample-mean energy bias**: MH is strictly greedier — deterministically
  accepts all downhill moves — and cascades into local minima faster than
  Gibbs under finite-time averaging (K ≤ 100). Generates divergent
  sample-mean energy bias.
- **Higher-order statistics**: Single-site MH suffers from rejection
  self-loops (stuttering), drastically increasing autocorrelation time
  versus block-Gibbs's parallel-coloring updates that randomize entire
  conditionally-independent vertex sets per sweep.

**(b) For n=128 fully-connected at β=1: K ≥ exp(Ω(n)).** The setup is a
dense unscaled Sherrington-Kirkpatrick (SK) spin glass operating deep
in the replica-symmetry-breaking glassy phase (T=1 ≪ T_c ≈ 6.5; local
field σ ≈ 6.5). For MH and block-Gibbs to align to KL < 0.05 requires
both chains to wash out their divergent transient dynamics — i.e.,
mix to stationarity. Mixing time is dominated by Arrhenius crossing of
O(n) energy barriers, giving **K ≫ 10¹⁵ sweeps** for parity at n=128.
**Computationally astronomical and infeasible.**

**(c) Algorithm 2 CANNOT recover block-Gibbs.** Two independent reasons:

1. **Graph-coloring collapse**: A fully-connected graph is the clique
   K_128. Any valid independent-vertex coloring uses exactly 128 colors,
   so maximum block size = 1. Block-Gibbs degenerates to single-site
   Gibbs, which (per (a)) still diverges from MH.
2. **Differentiability catch-22**: On sparse graphs where blocks > 1,
   recovering block-Gibbs requires the MH acceptance ratio to equal 1.0
   identically. But MCMC Layers' Fenchel-Young K=1 gradient
   (Proposition 3) is derived from the score function of the acceptance
   ratio. **If acceptance is fixed at 1.0, the Δ-gradient signal
   vanishes — destroying the differentiability premise of the paper.**
   Any other proposal yields Block-Metropolis, which structurally
   diverges from Block-Gibbs.

**(d) Minimum K dominated by spectral gap.** On the dense spin glass,
the spectral gap is exponentially small. Production-feasible K (≤100)
cannot wash out the transient kernel mismatch.

### Falsifiable empirical predicate (zero-coupling test)

Set `J_ij = 0`, `h_i = 0` (infinite temperature, ΔE = 0 for all moves).
Start both samplers from `y_0 = [-1, -1, ..., -1]`. Run K=1 sweep.
Measure expected Hamming distance from initial state.

| Sampler | K=1 behavior | Expected Hamming |
|---|---|---|
| Gibbs (THRML) | `1/(1+e^0) = 0.5` per spin → randomizes perfectly | 64 (binomial center) |
| MH (Carnot/MCMC Layers) | `min(1, e^0) = 1` accept all → deterministic invert OR unrejected random walk | 128 (systematic) or ≈55.3 (random scan) |

KL between perfect binomial and deterministic/Poisson-binomial at K=1
is mathematically massive. **This isolates the operator mismatch from
landscape effects and proves the divergence is baked into the kernels,
not a learnable mixing-time issue.**

### Action: vendor THRML directly

> "Because Carnot is an EBM verification framework auditing a reference
> simulator at K ≤ 100 ≪ τ_mix, your true mathematical target is not
> the asymptotic Boltzmann distribution π, but the specific
> non-equilibrium transient distribution `P_0 T_THRML^100`."

Carnot must execute THRML's exact Markov transition operator: independent-
set graph coloring, block-conditional heat-bath math, parallel scan
schedule, exact JAX PRNG key consumption paths. **Vendoring THRML
guarantees alignment with the reference target.**

### Implications for follow-up prompts (cascade)

- **DT-MCMC-K1** (Phase 5 in-situ training quality): RELEVANCE
  *narrowed*. MCMC Layers is no longer the inference-time sampler;
  it could still be a Phase 5 *training-time* component if the goal
  there is differentiable PCD on a non-THRML target. Consider rescoping
  the prompt to: "given that we vendor THRML for inference, can MCMC
  Layers' K=1 Fenchel-Young loss train the verifier-coupling parameters
  θ such that θ-induced THRML samples match data?"
- **DT-MCMC-STATELESS** (production deployment): NOW MOOT for the
  sampling role. THRML is library-shipped (no persistent chain
  required). Question may still be relevant for any auxiliary MCMC
  Layer use.
- **DT-COMPOSITION**: REVISED. Three-sampler composition becomes
  [THRML-vendored block-Gibbs] for (I) inference sampling,
  [Spectral Annealing] for (II) optimization,
  [BRAIN or MCMC Layers] for (III) distribution learning.

### Tasks to file

1. `.120 priority update`: drop the
   `exp15ZZ-iclr26-mcmc-layer-as-sampler-fix` task seed; replace with
   `exp15ZZ-thrml-vendored-block-gibbs-replacement`.
2. `.120 audit task`: Run the zero-coupling test (10 minutes of
   compute). Report Hamming-distance distributions for Carnot's Gibbs
   vs THRML at K=1. Confirms or refutes the operator-mismatch finding
   empirically before committing to vendoring.
3. **CLAUDE.md decentralization-rule check**: THRML 0.1.3 is Apache-2.0
   on PyPI. Per Rule 3 (distribution mirroring): we should fork THRML
   to a Carnot-controlled mirror (gitea + github) before depending on
   it as a load-bearing inference component, in case Extropic
   deprecates / re-licenses / removes it.

### Paper-v6 implications

- The KL=0.17 finding from .119 is **resolved cleanly** by vendoring
  THRML: Carnot's sampler IS THRML's sampler, parity is constructive.
- Paper-v6 §3 can claim "Carnot uses Extropic THRML 0.1.3 as the
  reference sampler implementation, vendored under Apache-2.0."
- The integration plan's Theme B (three orthogonal sampler options)
  collapses to two — Carnot's sampler is no longer a research question;
  it's THRML.
