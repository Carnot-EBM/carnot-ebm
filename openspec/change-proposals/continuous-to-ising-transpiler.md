# Continuous-to-Ising Transpiler: bridging Phase 1 verifiers and Phase 2 hardware

**Status:** Draft change proposal. **REQUESTED FOR MILESTONE 2026.04.78** (after
.76's EnvPropagation/Tier-2 work and .77's supervisor + schema work land).
**Origin:** 2026-04-27 Deep Think exchange (two rounds). Round 1 produced
the Continuous ε-Ising-Rank Theorem with two embedding strategies. Round 2
closed the rigor gaps and — critically — identified that the trace-based
upper bound `O(n log(1/ε) + W log²(1/ε))` melts hardware once W gets large
(a 10⁶-parameter transformer-class verifier in FP32 needs ~10⁸ aux spins,
five orders of magnitude beyond the Extropic Z1's 16k or KV260's LUT
budget). The fix is **Split-Verifier + Native Thermodynamic Distillation**:
keep the heavy backbone on host (CPU/GPU), distill *only the energy head*
into a Boltzmann Machine on hardware via Persistent Contrastive Divergence
on the EBM's own MCMC samples. The hardware no longer logic-traces a
verifier; it physically *is* a sampler trained to match it.
**Target milestone:** 2026.04.78. The KV260 board has been on-hand since
2026-04-20, the Ising and KAEM verifier crates are stable, and the .76
KAN MILP fix should give us a clean small verifier to use as the prototype
input. Earlier milestones are too crowded with conductor-infrastructure
work.
**Priority:** Medium-high. This is the canonical artifact that converts
Carnot's Phase 1 verifier networks into Phase 2 hardware deployments. Without
it, every hardware port (KV260, ECP5/Nexus, future Extropic XTR-0, future
photonic SLM) requires hand-engineered energy functions per board.
With it, one trained `state_dict` compiles to any Ising-compatible target.
**Depends on:**
  - KV260 board on-hand (✅ since 2026-04-20).
  - Stable `SamplerBackend` protocol — see `_bmad/architecture.md` and
    REQ-KONA-006.
  - At least one trained verifier with W small enough to fit KV260's
    LUT/spin budget. KAEMEnergy after .76's MILP fix is the natural
    candidate.

## Summary

Carnot's Phase 1 verifiers are continuous parametric energy functions
`E: ℝⁿ → ℝ` realised as JAX or PyTorch networks. Phase 2 hardware
(KV260 FPGA, future XTR-0 thermodynamic sampling, future photonic SLM)
samples discrete Ising states `s ∈ {−1, +1}ᴺ` according to a quadratic
energy `E_Ising(s) = −sᵀJs − hᵀs`. The gap between these two
representations has so far been closed by hand: every hardware port
involves bespoke energy engineering, which doesn't scale and doesn't
respect the decentralization rule that Carnot must run on multiple
hardware backends in parallel.

Three embedding paths, selected by verifier size and hardware topology:

  - **Approach 1 — Execution-Trace Embedding (sparse hardware, *small*
    verifiers only).** Quantize input domain with m-bit fractional
    binary, allocate hidden spins per logic gate, each gate becomes a
    2/3-local QUBO penalty `P_g`. Hardware energy is
    `λ Σ_g P_g + Σ_k 2^k h_k^out s_k^out`. Spin count
    `N = O(n log(1/ε) + W log²(1/ε))`. **Formal KL bound (Round 3
    closed derivation):** `λ ≥ (1/β_min)(N_hid ln 2 + ln(1/ε)) + ΔE`
    where `ΔE = sup E − inf E`. Proof chain: (i) since each spatial
    state `v` has a unique valid trace `h*(v)`, the marginal satisfies
    `P_marg(v) ≥ P_target(v) · Z_P/Z_I`; (ii) substitution gives
    `D_KL ≤ ln(Z_I/Z_P) = ln(1 + Z_invalid/Z_P) ≤ Z_invalid/Z_P`;
    (iii) worst-case bounds on Z_invalid (all 2^(N_vis+N_hid) states
    spoofing the deepest well at min penalty) and Z_P (all valid
    states at sup E) close the inequality. The `ΔE` term is
    *physically required* — it is the thermodynamic depth that
    prevents hardware relaxation from intentionally breaking gates to
    fall into a fictitious deep well. Practical spin-count budget
    caps W in the low thousands; only KAN-tier and Ising-tier
    verifiers fit.
  - **Approach 2 — Arc-Cosine V-Statistic Lift (dense hardware:
    photonic SLM, optical Ising).** Random threshold features
    `φᵢ(x) = sgn(wᵢᵀx + bᵢ)`. Fit `E(x) ≈ −φᵀJφ − hᵀφ` by ridge
    regression on outer products `φφᵀ`. **Formal anchor (Round 3):**
    Cho-Saul (NIPS 2009, "Kernel Methods for Deep Learning") proved
    that the continuous expectation of pairwise threshold products is
    *exactly* the 0th-order arc-cosine RKHS kernel
    `k_0(x,y) = 1 − (1/π) arccos(xᵀy/‖x‖‖y‖) = 𝔼_w[sgn(wᵀx)sgn(wᵀy)]`.
    Kar-Karnick (AISTATS 2012, "Random Feature Maps for Dot Product
    Kernels") then established that ridge regression on the
    `C(N,2) ≈ N²/2` cross-terms is a 2nd-order V-statistic, with
    approximation error `O(1/N)` (not `O(1/√N)`). Spin count drops
    from the naive `O(1/ε²)` to `O(‖E‖_RKHS / ε)` — the "wave-
    interference free lunch" of dense optical hardware. **Standing
    caveat:** the L²-to-KL conversion in the final step is a textbook
    bounded-log-density argument but was not explicitly derived; and
    the `‖E‖_RKHS` bound assumes the trained EBM lies in the arc-
    cosine RKHS with finite norm, which is a structural assumption
    (Cho-Saul show the RKHS is dense for many natural function
    classes, but specific verifier networks need empirical check).
    Framed practically as Student-Teacher distillation: we don't
    *prove* `E ∈ RKHS`; we fit J empirically on continuous MCMC
    samples and check the residual.
  - **Approach 3 — Split-Verifier + Native Thermodynamic Distillation
    (sparse hardware, *any verifier size*).** The load-bearing path
    for production use. Run the heavy verifier backbone (transformer,
    deep MLP) on host. Project its output to a low-dim continuous
    latent `z`. Distill the final energy head `E(z)` directly into a
    Boltzmann Machine (visible + hidden spins) via Persistent
    Contrastive Divergence on the verifier's existing MCMC samples
    from Phase 1. The hardware no longer represents the verifier's
    forward pass; it *is* a trained sampler. Zero logic gates, zero
    penalty term, zero invalid states. Spin budget N is whatever the
    hardware has (16,384 on Z1, ~16k on KV260 LUTs); the verifier
    parameter count W is decoupled from N entirely. Trade-off vs
    Approach 1: KL bound becomes empirical (PCD training loss) rather
    than formal — we trade theoretical guarantee for engineering
    feasibility.

All three share the **same input** (a trained `state_dict` + MCMC sample
corpus) and produce the **same output** (a hardware spec `(J, h, φ, ψ)`).
The transpiler selects an approach based on a `HardwareSpec` descriptor
and the verifier's parameter count.

## What this proposal IS NOT

- **Not a hardware bring-up project.** KV260 RTL work (Exps 982 and
  prior v4 BD) continues in its own track. This proposal produces the
  `(J, h)` that *runs on* whatever hardware is brought up; it does not
  write Verilog.
- **Not a new training method.** The verifier is trained by whatever
  Phase 1 pipeline produced it. The transpiler is post-hoc.
- **Not a paper draft.** The Continuous ε-Ising-Rank Theorem may
  belong in a paper later; the change proposal scopes the *artifact*
  (the transpiler), not the writeup.
- **Not gold-plating.** We are not implementing both approaches in
  full. Approach 1 is the load-bearing path because we own KV260.
  Approach 2 is sketched and lit-checked but not built unless and
  until we have dense-hardware access.

## Proposed experiments

### Exp A — Approach 1 trace-based transpiler primitive

**Deliverable:**
new module `python/carnot/hardware/transpiler/__init__.py` +
`python/carnot/hardware/transpiler/trace.py` (Approach 1 logic) +
`python/carnot/hardware/transpiler/quantize.py` (m-bit fractional
binary encoder + ψ uniform-noise decoder) +
`tests/python/test_transpiler_trace.py` +
`results/experiment_<N>_transpiler_trace_primitive.json`.

**What it does:**

1. `class HardwareSpec`: descriptor with `kind: Literal["sparse","dense"]`,
   `max_spins: int`, `beta_range: tuple[float, float]`, optional
   `locality: Literal[2, 3]` for sparse hardware.
2. `def transpile(model: nn.Module | jax.Module, hw: HardwareSpec, eps: float)
   -> IsingSpec`:
   - Choose Approach 1 (sparse) or Approach 2 (dense) by `hw.kind`.
   - Walk the model's forward pass into an arithmetic-logic IR.
     PyTorch path: `torch.fx.symbolic_trace`. JAX path: `jax.make_jaxpr`.
   - For each fixed-point arithmetic op (adder, multiplier, ReLU,
     comparator) emit a 2- or 3-local QUBO penalty per the standard
     gadget library (Lucas 2014 + extensions).
   - Concatenate the per-gate penalties + the readout term into a
     final `(J, h)` matrix.
3. `class IsingSpec`: dataclass holding `J: scipy.sparse.csr_matrix`,
   `h: np.ndarray`, `psi: Callable[[np.ndarray], np.ndarray]` (decoder),
   `phi: Callable[[np.ndarray], np.ndarray]` (encoder), and a
   `provenance:` field with the hardware spec and the source `state_dict`
   hash.

**Acceptance:** for a synthetic 2-input, 1-hidden-layer MLP with
small W, the transpiler emits an `IsingSpec`. Brute-force sample all
`2^N` spin states, compare to the continuous E(x) on a 1024-point grid,
and verify KL ≤ 0.05 over `β ∈ [0.5, 5.0]`.

### Exp B — KAEMEnergy KV260 trace-based prototype (Approach 1)

**Deliverable:**
edits to `python/carnot/hardware/transpiler/trace.py` to handle the KAEM
forward pass in particular +
`results/experiment_<N>_kaemenergy_kv260_transpile.json` +
companion writeup at `docs/research-notes/transpiler-kaemenergy.md`.

**What it does:**

1. Load the post-.76-MILP-fix KAEMEnergy verifier.
2. Compute `W` (parameter count) and the projected `N` from the
   theorem upper bound `N = O(n log(1/ε) + W log²(1/ε))`.
3. If `N ≤ 16k` (KV260 LUT-implementable budget), compile to an
   `IsingSpec`.
4. If `N > 16k`, report the gap, log it as Approach-1-not-applicable,
   and refer the verifier to Approach 3 (Exp F). This is the expected
   path for any non-trivial verifier.
5. Validate KL ε empirically against the JAX reference on a held-out
   FoVer / SVAMP test set when applicable.

**Acceptance:** KAEMEnergy is small enough that Approach 1 is viable —
either `N ≤ 16k` and KL ≤ 0.05 over the relevant β range (deployable
spec), or `N > 16k` with a clean diagnostic and explicit handoff to
Exp F.

### Exp F — Native Thermodynamic Distillation primitive (Approach 3)

**Deliverable:**
new `python/carnot/hardware/transpiler/distill.py` (PT-PCD trainer +
Gray-code encoder + diagnostics) +
`tests/python/test_transpiler_distill.py` +
`results/experiment_<N>_transpiler_native_distillation.json`.

**Why this is non-trivial:** SVAMP/FoVer energy landscapes are not
smooth Gaussians — they are *glassy*: deep isolated sinkholes (valid
reasoning paths) separated by massive high-energy plateaus (logical
contradictions). Naive training has two failure modes that map to two
hardware-deployment failures:

  - **Vanilla CD-1/CD-k** carves the valid modes well but never
    explores empty hypercube space, leaving spurious deep wells in
    unconstrained spin configurations. At deployment the hardware
    falls into these hallucinated wells and emits garbage outputs
    (False Positives).
  - **Vanilla PCD** with persistent chains can't cross
    logical-contradiction barriers, so the gradient violently biases
    toward the first discovered mode and destroys the others (False
    Negatives, mode collapse).

The fix is **Persistent Parallel Tempering (PT-PCD)** with three
structural guardrails:

1. **Geometric β ladder.** M = 16 rungs from `β_0 = 1.0` (cold,
   production) to `β_M ≈ 0.05` (hot, noise). Hot chains random-walk
   over barriers; Metropolis-Hastings swaps every K steps drag
   spurious-minimum states down to the cold chain where the negative
   gradient flattens them. Note: for SVAMP-class energy gaps that may
   span tens of units, `β_M` may need to drop further (`< 0.01`).
2. **Gray-code (or thermometer) visible-spin encoder.** Replaces
   m-bit base-2 binary to eliminate the cliff problem
   (`0111 → 1000` would otherwise require 4 simultaneous bit flips
   for an infinitesimal continuous shift). Gray code preserves
   Hamming-distance-1 between adjacent quantization cells, so
   continuous Euclidean locality maps to physical hardware locality.
3. **5% rebirth teleportation** on hot chains every epoch (overwrite
   with uniform `{−1, +1}` noise) + **L2 = 1e-4 weight decay on J**
   (prevents the "golf-course" pathology where unbounded `J` creates
   flat plains with infinitely deep pinholes that hardware SB can't
   dynamically relax into).

**API:**

`class CarnotNativeDistiller(n_vis: int, n_hid: int, n_chains: int,
n_temps: int)` with method `distill(mcmc_samples, hardware_spec,
epochs, lr=0.01, l2_reg=1e-4)`. Internals:

  - `pt_negative_phase(k_steps=15)`: per-temperature Gibbs sweep
    using `local_fields = chains @ J + h`,
    `p_flip = sigmoid(−2β · local_fields · chains)`, then
    Metropolis-Hastings replica swaps between adjacent rungs.
  - `train_step(...)`: positive phase encodes continuous batch via
    Gray-code; negative phase calls `pt_negative_phase`; gradient is
    `pos_grad_J − neg_grad_J − l2_reg · J`. Symmetrize and zero
    diagonal each step.
  - `compute_diagnostics(s_data, s_model)` (see below).

**Three mandatory diagnostics** (no flashing payload until all three
pass):

  - **Diagnostic A — KDE overlap.** Sample 5,000 cold chains, decode
    visible spins back to 2D continuous space, KDE-overlay against
    the Phase 1 teacher MCMC. Mode-collapse detector. Hard threshold:
    2D earth-mover distance ≤ 0.1.
  - **Diagnostic B — Energy Histogram Overlap (EHO).** Compute the
    Ising energy `E = −sᵀJs − hᵀs` for both teacher (encoded) and
    student (cold-chain) samples. Plot histograms. Fail condition:
    student histogram has a secondary spike at energy below the
    teacher minimum — that's a hallucinated black hole. Spurious-
    minima detector.
  - **Diagnostic C — PT swap acceptance rate ≥ 15%** between every
    pair of adjacent rungs. Below this, hot-chain exploration is
    blockaded and the temperature ladder is broken. Mixing-quality
    detector.

**Export `IsingSpec`** matching Approach 1's output schema, so
downstream `SamplerBackend` integration (Exp E) is approach-agnostic.

**Acceptance:** on a 1,024-spin BM (n_vis=64 Gray-coded for a 2D
latent, n_hid=960) distilled from a small synthetic continuous EBM
(~10k MCMC samples), training NLL converges within 100 epochs and
all three diagnostics pass. For the SVAMP-EBM 2D-latent prototype,
the cold-chain decode produces samples that map back to *valid*
SVAMP step-verdicts on a held-out test set with accuracy ≥ 0.90.

**Publishable framing:** "PT-PCD with Gray-code encoding for
distilling continuous EBMs to glassy-landscape Boltzmann machines on
Ising hardware" is a methods contribution in its own right.
Diagnostics A/B/C, the dichotomy of failure modes (False Positives
via spurious minima, False Negatives via mode collapse), and the
five guardrails (β ladder, Gray code, swap criterion, 5% rebirth, L2
cap) all originate in this work.

### Exp C — Approach 2 simulation prototype

**Deliverable:**
new `python/carnot/hardware/transpiler/barron.py` +
`tests/python/test_transpiler_barron.py` +
`results/experiment_<N>_transpiler_barron_simulation.json`.

**What it does:**

1. Implement the random-threshold-features encoder `φᵢ(x) = sgn(wᵢᵀx + bᵢ)`
   with N basis vectors.
2. Fit `J, h` by convex ridge regression on a corpus of `(x, E(x))`
   pairs sampled from the trained verifier.
3. Vary N and ridge-regularization λ_reg, plot KL vs N. The Barron-
   bound says `N = O(‖E‖²_B β²_max / ε²)`; we want to see that
   scaling empirically.

**Acceptance:** Approach 2 KL converges as N grows on a small toy
verifier, with a constant that's plausibly within an order of
magnitude of the Barron-norm prediction. This is a no-hardware
simulation experiment; it informs whether Approach 2 is worth
pursuing in earnest if and when dense-hardware access materializes.

### Exp D — Theorem rigor pass

**Deliverable:**
`docs/research-notes/continuous-ising-rank-theorem.md` + companion
LaTeX source at `docs/papers/continuous-ising-rank.tex` (draft) +
`results/experiment_<N>_ising_rank_theorem_writeup.json`.

**What it does:**

Tighten the two known rigor gaps in Deep Think's proof sketch:

1. **Approach 1 KL bound.** Deep Think's `λ ≥ (1/β_min) ln(2^N_hid / ε)
   + 2 sup |E|` formula bounds the *count* of invalid states, but the
   actual KL bound needs the partition-function ratio
   `Z_continuous / Z_Ising`. Standard log-sum-exp manipulation closes
   this; the writeup makes the manipulation explicit.
2. **Approach 2 dense-quadratic claim.** The claim `E(x) ≈ −φᵀJφ − hᵀφ`
   for Barron functions with ridge-fitted J,h is non-textbook. Find a
   citable result (random-features literature: Rahimi-Recht, Bach 2017
   on Barron-space approximation) or, if none exists, prove the lemma
   from scratch.

**Acceptance:** writeup with full KL derivation for Approach 1 and
either a citation chain or original lemma for Approach 2. No code
deliverable — pure math + LaTeX.

### Exp E — `SamplerBackend` adapter for transpiled IsingSpec

**Deliverable:**
edits to `python/carnot/hardware/sampler_backend.py` (new
`TranspiledSamplerBackend` class) +
`tests/python/test_transpiled_sampler_backend.py` +
`results/experiment_<N>_transpiled_backend_integration.json`.

**What it does:**

Wires the `IsingSpec` from Exp A/B into the existing `SamplerBackend`
protocol (REQ-KONA-006). The continuous verifier's `verify(x)` call
now has a path that runs through hardware: `x → φ(x) → hardware
sample → ψ(s)`. This is the integration point that closes the loop
between Phase 1 verifier code and Phase 2 hardware deployment.

**Acceptance:** the existing verifier test suite (`tests/python/
test_kaemenergy.py` etc.) passes both with the JAX in-process backend
*and* with the `TranspiledSamplerBackend` driving a software Ising
simulator (Gibbs sampling on the transpiled `(J, h)`). The two
backends produce equivalent verdicts on a held-out test set within
the KL bound from Exp B.

## Decentralization implications

- **Rule 1 (local-first):** strongly reinforced. The transpiler is
  the artifact that lets users run Carnot end-to-end on
  Carnot-controlled local hardware (KV260, ECP5, future XTR-0)
  without depending on any cloud sampling service.
- **Rule 5 (hardware portability as political requirement):** this
  is the load-bearing rule. The transpiler is what makes
  hardware-portability practical rather than aspirational. Without
  it, REQ-KONA-006 is a protocol with one or two backends; with it,
  any Ising-compatible board becomes a viable target with no per-
  board engineering effort beyond a `HardwareSpec` descriptor.
- **Rule 7 (no vendor abstractions):** the transpiler lives in
  `python/carnot/hardware/transpiler/`. It depends on the abstract
  `SamplerBackend` protocol, never on a vendor SDK. KV260-specific
  bitstream emission lives in a separate `hardware/kv260/` submodule
  that the transpiler can target via a `HardwareSpec(kind="sparse",
  vendor_target="xilinx-kv260")` flag without coupling the core to
  Xilinx.

## Risks

- **Spin-count constants in practice.** The trace-based theorem is
  `O(W log² ε⁻¹)`, but the constant matters. KAN-tier and Ising-tier
  verifiers fit KV260's 16k budget; transformer-class verifiers
  (FoVer-Tier2, SC-Energy) blow up to ~10⁸ spins and are flatly
  infeasible under Approach 1. **This was the spin-count crisis that
  motivated Approach 3.** Mitigation: Approach 3 (Native Distillation)
  decouples N from W entirely. Exp B handles the small-verifier case;
  Exp F handles the large-verifier case. The transpiler's selector
  logic chooses based on projected N vs hardware budget.
- **PCD convergence is empirical, not proven.** Approach 3 trades the
  formal KL bound from Approach 1 for engineering feasibility — the
  guarantee is whatever the training loop converges to. Mitigation
  (Round 4 closed): the **PT-PCD + Gray-code + 5%-rebirth + L2 cap**
  recipe explicitly addresses the two known failure modes (spurious
  minima from vanilla CD, mode collapse from vanilla PCD); three
  falsifiable diagnostics (KDE overlap, EHO, swap acceptance ≥15%)
  must pass before any payload is flashed. Standing caveat: high
  swap acceptance and aligned KDE blobs are necessary but not
  sufficient — small modes can still be missed at 5,000-sample
  resolution.
- **Visible-spin encoding solved.** Round 4 Gray-code (or
  thermometer) encoding preserves Hamming-distance-1 locality between
  adjacent quantization cells. Standard base-2 binary's cliff problem
  (a 4-bit flip for an infinitesimal continuous shift at byte
  boundaries) is the silent killer of small-BM prototypes; Gray code
  eliminates it.
- **Approach 2 is speculative.** The Barron-quadratic lemma may not
  exist in the literature in the exact form needed. Mitigation: Exp D
  explicitly checks; if the lemma is unprovable as stated, Approach 2
  is shelved and we ship Approach 1 alone.
- **PyTorch / JAX FX-tracing limits.** `torch.fx.symbolic_trace` does
  not handle dynamic control flow (`if x.sum() > 0: ...`) cleanly.
  Mitigation: the trained verifiers in scope are static-graph
  feed-forward networks (KAEMEnergy, SC-Energy, Ising tier). This is
  fine for the prototype; control-flow handling is a follow-up
  proposal if and when a dynamic-control-flow verifier ships.
- **Gadget library completeness.** The 2-/3-local QUBO gadget library
  (Lucas 2014 + extensions) covers add, multiply, ReLU, sign,
  comparator. Exotic ops (softmax, layer norm, attention) require
  new gadgets. Mitigation: KAEMEnergy is feed-forward + ReLU + linear
  + KAN spline. KAN splines decompose into piecewise polynomials;
  each piece is add+multiply+ReLU. So the prototype path is closed.
  Attention-based verifiers are out of scope for this proposal.

## Acceptance criteria (overall)

1. The transpiler primitive (Exp A, Approach 1 trace-based) compiles
   a synthetic small MLP to an `IsingSpec` and the round-trip KL ≤
   0.05 on a brute-force 1024-point validation grid.
2. KAEMEnergy compiles (Exp B) — either fits KV260 under Approach 1
   with KL ≤ 0.05, or cleanly diagnoses the spin-budget gap and hands
   off to Exp F.
3. Approach 2 (Exp C) at minimum has a working simulation that
   demonstrates the predicted N-vs-ε scaling.
4. The Continuous ε-Ising-Rank Theorem writeup (Exp D) reproduces
   the Round 3 closed derivations: (a) the Approach 1 KL bound via
   `Z_invalid/Z_P` partition with `λ ≥ (1/β_min)(N_hid ln 2 +
   ln(1/ε)) + ΔE`, and (b) the Approach 2 RKHS lift anchored in
   Cho-Saul (2009) arc-cosine kernel + Kar-Karnick (2012)
   V-statistic feature maps, with explicit L²-to-KL conversion and
   a check that the target EBM has finite arc-cosine RKHS norm.
5. The `SamplerBackend` integration (Exp E) is end-to-end: a
   verifier that worked under the JAX backend produces equivalent
   verdicts under the `TranspiledSamplerBackend`, validating the
   bridge between Phase 1 code and Phase 2 hardware.
6. **Native Thermodynamic Distillation (Exp F, Approach 3) is the
   load-bearing path for production-scale verifiers.** A 1,024-spin
   BM distilled from a synthetic small EBM converges within 100 PCD
   epochs to empirical KL ≤ 0.10. The same pipeline distills a
   production SVAMP energy head onto a 16k-spin BM and the resulting
   `IsingSpec` deploys to KV260 SB simulation.
7. The transpiler API (`HardwareSpec`, `transpile`, `IsingSpec`) is
   the canonical Carnot Phase 2 surface. The selector picks among
   Approaches 1/2/3 based on hardware topology and projected spin
   count; new hardware backends extend this API rather than writing
   parallel one-off paths.

## Immediate concrete next step (Deep Think's recommendation)

If the KV260 board is on hand and ready, the smallest end-to-end
prototype is:

1. Isolate a 2D latent subset of an existing SVAMP continuous EBM
   (for example: project the energy head's input to its top-2 PCA
   components on the SVAMP train set).
2. Run `distill_to_boltzmann()` to fit a 1,024-spin BM (32×32 grid,
   half visible / half hidden) on the existing MCMC samples.
3. Push the resulting `(J, h)` to a Simulated Bifurcation loop
   running on the KV260's ARM core (sufficient for proof-of-concept;
   FPGA-fabric SB is the production target).
4. Sample, decode, and check whether the sampled spins map back to
   valid SVAMP-step verdicts on a held-out test set.

If yes, the hardware translation layer is formally validated and the
larger SVAMP / FoVer pipeline becomes a constants-tuning exercise
rather than an open question.

## Why this is in change-proposals, not just a research task

Phase 2 is the second of three project phases (CLAUDE.md "Project
Vision"). Every Phase 2 capability we ship needs to extend a single
abstraction rather than accumulate per-board ad-hoc paths — otherwise
the "hardware portability as political requirement" decentralization
rule becomes lip-service. Naming the transpiler as the canonical
artifact, in change-proposals where future planners will see it,
keeps that abstraction central rather than letting it drift into
"the KV260 deploy script that one experiment wrote."

The Continuous ε-Ising-Rank Theorem is also paper-worthy on its
own. The change proposal scopes the engineering artifact; the paper
(if we write it) cites this proposal as the artifact's home.
