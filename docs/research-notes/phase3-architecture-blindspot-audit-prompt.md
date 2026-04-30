# Phase-3 Architecture Blind-Spot Audit — Deep Think Prompt

**Status:** Drafted 2026-04-30 after Round 3 produced the
DBAE-EBM-with-quadratic-Ising-distillation architecture decision.
Ready for paste when quota resets.

**Why this audit:** the same methodology that caught 4 fatal
vulnerabilities pre-publication during the original Phase-3
derivation (Round 8). Highest-value-per-token historically. Surfaces
the "passes silent unit tests but produces non-functional
deployment" failure mode that's catastrophic at scale.

---

## Prompt to paste

You are a hostile adversarial reviewer at NeurIPS / ICML / ICLR
trying to **torch** Carnot's Phase-3 architectural decision at
submission time. Your goal is to find every failure mode the
designers haven't anticipated. Be ruthless. Do not soft-pedal.

## The architecture under audit

After three rounds of architectural Deep Think + hardware
verification, Carnot's Phase-3 Kona-parity foundation model
architecture is:

**Training:**
- Deterministic Bounded Autoencoder (DBAE):
  `text → encoder → tanh-bounded z ∈ [-1, 1]^d → decoder → text`
- Hard tanh STE on encoder pre-activations (not strictly required
  for continuous z, but applied as gradient regularizer)
- Deep neural network energy `E(z)` (not quadratic) trained jointly
  with the AE
- Mirrored Langevin Dynamics (MLD): steps in arctanh-dual space,
  energy evaluated in bounded primal
- 3-stage training: AE manifold warmup → EBM warmup (frozen AE) →
  asymmetric finetune with stop-gradient on z_fake → encoder
- VICReg/Barlow Twins regularization (variance + decorrelation)
- Masked-token reconstruction (30-50% drop)
- Denoising-AE training (5-10% Gaussian noise on z_data before
  decoder)
- Parallel Tempering at training time (β ∈ {0.1, 0.4, 0.7, 1.0}),
  evaluate loss only on β=1 chain
- PCD with persistent replay buffer of 10K z-states, 95%/5%
  buffer/random, 5% noise injection
- AND-composition over k=15+ verifiers: `E_total(z) = Σ E_i(z)` in
  z-space
- Latent dimension d ∈ {256, 512}

**GPU deployment:** run the deep continuous EBM directly. MLD
sampling, full expressivity.

**FPGA deployment (KV260):** **distill** the deep continuous EBM
into a quadratic Ising matrix `(J, h)` via Taylor expansion or
least-squares fit. The FPGA's hardware primitive is `P = σ(-2β·ΔE)`
with `ΔE_i` computed as a sparse matrix dot-product against
neighbors — i.e., it can ONLY execute quadratic Ising energies
`E(z) = z^T J z + h^T z`. Hardware is verified to be discrete
synchronous parallel Glauber on `±1` spin lattice.

**Acceptance gate (proposed at 1B-token scale, 4 conditions):**
1. `inf_t α_t > 0.1` across 100 MLD steps (Zenil α_t grounding)
2. Decoder >85% joint-constraint pass rate
3. `sgn(z')` binarization degrades pass rate by <5%
4. (NEW after Round 3) Distillation residual <X% — measured
   between deep EBM predictions and quadratic-Ising approximation

## Three rounds of prior reasoning (full context)

**Round 1** chose DBAE-EBM (Candidate D) over VAE-EBM (collapse),
Diffusion-EBM (hardware-mismatch fallback), and Direct-EBM (no
α_t trajectory). Argument: bounded continuous latent is both
MCMC-friendly and natively Booleanizable via sgn(z) for FPGA.

**Round 2** sharpened the recipe (regularization, 3-stage loss,
MLD, validation scale), then raised a conditional pivot flag:
"if Phase-2 FPGA executes pure discrete Glauber on a ±1 lattice,
continuous EBM is structurally mismatched — pivot to discrete AE +
discrete EBM."

**Round 2.5 (verification + premature pivot to DAE-DEBM):**
hardware-verified KV260 IS discrete Glauber on ±1. Pivoted to
straight-through-estimator + discrete Gibbs/Glauber EBM. Believed
the transpilation gap collapsed to zero by construction.

**Round 3 caught the deeper constraint and pivoted back:** the
KV260 isn't just discrete Glauber; it's QUADRATIC Ising specifically
(`P = σ(-2β·ΔE)` with ΔE computable as sparse matrix dot-product).
A deep neural EBM cannot run natively on this hardware regardless
of whether the latent is continuous OR discrete. **Both the Round 1
DBAE-EBM and the Round 2.5 DAE-DEBM hit the same wall** at FPGA
deployment.

Round 3's resolution: train at full deep-NN expressivity (continuous
DBAE-EBM), then distill at deployment to quadratic `(J, h)`.

The claim: this gives us the best of both worlds — deep
expressivity for k=15 verifier composition during training,
hardware-compatible quadratic deployment via distillation. The
"transpilation gap" becomes the distillation residual, which is
"a known, bounded engineering cost."

## Your task

Find the ways this architecture fails that we haven't anticipated.
Specifically attack the seams between components — that's where
real systems break.

### Required attack surfaces

1. **Distillation catastrophic failure modes.** When does Taylor
   expansion of `E_deep` around a learned anchor produce an
   `(J, h)` whose minima don't correspond to the deep EBM's
   minima? Specifically: when can the quadratic surrogate have
   *more* low-energy modes than the deep EBM, leading the FPGA
   sampler to converge to states the deep model would have
   rejected?

2. **Silent unit-test passing, non-functional deployment.** Find a
   plausible scenario where:
   - All four acceptance-gate conditions pass on the small
     prototype
   - Position paper claims hardware deployment works
   - Actual KV260 deployment produces qualitatively wrong outputs
     in production
   - The failure surfaces only at user-facing scale or on
     adversarial inputs, not in our test harness

3. **α_t collapse under distillation.** The α_t lower bound is
   derived for the continuous deep EBM with MLD trajectory. Does
   it survive the deep → quadratic distillation? Specifically:
   does the distillation residual systematically bias α_t below
   the 0.1 threshold at deployment, even though training-time α_t
   passes?

4. **AND-composition breakdown.** The claim is that
   `E_total = Σ E_i` distills into a single composite `(J, h)`. But
   `Taylor(Σ E_i) = Σ Taylor(E_i)` only holds at the same anchor
   point. If verifiers have different "natural" anchor states,
   their Taylor expansions don't compose. Does the FPGA need k=15
   separate `(J_i, h_i)` matrices, with composition done at runtime?
   If so, what's the bandwidth cost?

5. **Verifier-grounded training signal loss.** The 3-stage training
   has stop-gradient on z_fake → encoder. But the EBM's energy is a
   deep NN that's trained against verifier ground truth. If the
   distillation step happens AFTER training, the deployment-time
   FPGA energy IS NOT the energy the verifier was grounding. Is the
   verifier signal preserved through the distillation? Or is the
   FPGA optimizing against a quadratic surrogate of an objective
   that no longer reflects ground truth?

6. **Hardware assumptions we've made but not verified.** We've
   verified the KV260 hardware primitive is quadratic Ising
   Glauber. Have we verified:
   - The `(J, h)` precision (Q8.8 fixed-point per the Verilog)
     supports the dynamic range of distilled deep-EBM weights?
   - The MAX_DEGREE=32 sparsity constraint is consistent with the
     deep EBM's Hessian structure after distillation?
   - The N_SPINS=128 hardware limit doesn't conflict with the
     d ∈ {256, 512} latent dimension?

7. **Training-deployment gap pathologies.** Train on continuous
   `[-1,1]^d` MLD, deploy on quadratic Ising Glauber. List every
   distributional mismatch between training-time z-trajectories
   and deployment-time z-trajectories. Quantify which are
   "engineering costs" vs which are "model-breaks-on-deployment."

8. **Mode collapse via distillation.** Quadratic energies have at
   most 2^d local minima (corners of the hypercube). Deep EBMs can
   have richer landscape topology. Does distillation systematically
   collapse semantically-distinct deep-EBM minima onto the same
   quadratic basin? If so, the FPGA sampler converges to a more
   degenerate state distribution than the deep EBM — which would
   look like exactly the "language model" failure mode the
   architecture was supposed to avoid.

### Severity classification

Classify each finding as:

- **🚨 FATAL** — architecture cannot ship as designed; requires
  redesign or abandonment. Pre-publication catch is critical.
- **⚠️ DEGRADING** — architecture functions but materially
  underperforms specs. Worth documenting as a known limitation.
- **📝 COSMETIC** — minor issue, won't reviewer-torch the paper.

For each FATAL finding, propose:
- The minimum architectural change that could rescue the design,
  if any.
- Whether the change requires another architecture round (this is
  Round 5+ territory) or can be solved with empirical tuning.

For each DEGRADING finding, identify what we'd measure to confirm
or refute it on the small prototype.

### Output format

A numbered list of findings, each tagged 🚨/⚠️/📝, with a 100-word
explanation per finding. Bonus points for findings that don't fit
into any of the eight attack surfaces I enumerated — those would
indicate the audit found something I'd have completely missed.

Be specific. "The training-deployment gap might be a problem"
is useless. "The distillation step computes Taylor expansion at
the mean of training-time z-distribution, but at deployment the
sampler explores low-energy regions far from that mean, so the
quadratic approximation degrades exactly where it matters" is
useful.

Take the regret now if there's regret to take. Better to redesign
before scaling than torch the paper.
