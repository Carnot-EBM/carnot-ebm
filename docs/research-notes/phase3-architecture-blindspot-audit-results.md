# Phase-3 Architecture Blind-Spot Audit — Deep Think Result

**Date:** 2026-04-30
**Prompt:** `phase3-architecture-blindspot-audit-prompt.md`

## TL;DR — Take the regret now.

Five FATAL findings. Two of them require Round 5+ architecture
redesign. Two more are software-fixable but require substantive
implementation changes. One is empirically tunable. Plus four
DEGRADING and one COSMETIC.

**The pivot-back to DBAE-EBM with quadratic-Ising distillation is
NOT salvageable as-is.** The KV260's quadratic Ising primitive is
structurally inadequate for the deep EBM's expressivity, regardless
of how clever the distillation step is.

The most surprising finding is the **bonus / unprompted** finding:
**synchronous parallel Glauber on arbitrary Ising graphs violates
detailed balance and converges to bipartite limit cycles, not the
Boltzmann distribution.** This affects already-published Carnot
results (exp1081's 12,788× speedup measurement is comparing a
correct sampler against a non-equilibrium one).

## Findings (severity-ranked)

### 🚨 FATAL #1 — Dimensionality & Sparsity Guillotine

KV260 hardware: `N_SPINS = 128` (Verilog parameter), `MAX_DEGREE = 32`.

Round 3 specified `d ∈ {256, 512}`. **256 doesn't fit, 512 doesn't
fit.** Temporal multiplexing would invalidate the synchronous
parallel execution assumption.

Distilling a deep EBM yields a **dense `d × d` Hessian**. Forcing
that into a `MAX_DEGREE = 32` sparse graph amputates **>93% of the
energy topology** for `d = 512` (`32/512 ≈ 6%`). Verifier manifold
catastrophically distorted.

**Rescue:** Hard-cap `d = 128` + apply L0/L1 structural sparsity
penalties during continuous EBM training to organically match
`MAX_DEGREE = 32`.

**Status: Round 5+ architecture round.**

### 🚨 FATAL #2 — Synchronous Glauber Limit-Cycle Collapse (BONUS)

This is the highest-impact finding because it was **unprompted** —
Deep Think identified it outside the eight enumerated attack
surfaces. The pattern is exactly what blind-spot audits are for.

**The argument:** Synchronous parallel updates on an Ising lattice
with frustrated or antiferromagnetic couplings *mathematically
violate detailed balance* and **do not converge to a Boltzmann
distribution.** Instead, the chain deterministically collapses into
infinite 2-cycle oscillations (bipartite blinking states).

**The hardware verification I did was incomplete.** I confirmed the
KV260 implements bipartite checkerboard updates (even spins on phase
0, odd spins on phase 1). But that's only detailed-balance-preserving
**if the underlying coupling graph is itself bipartite** — i.e., if
edges in `J` only connect even-indexed spins to odd-indexed spins.

**For an arbitrary `J` matrix learned by distilling a deep EBM**, there
will almost certainly be even-even and odd-odd couplings. The
synchronous parallel update is then **not a valid Glauber sampler**.

**This affects already-published results:** exp1081's "12,788×
speedup" comparison is between a correct CPU sampler and an
incorrect FPGA sampler. The speedup is real (the FPGA is fast) but
it's sampling from a different distribution than the CPU. The
comparison is academically suspect.

**Rescue:** Reprogram KV260 firmware to enforce strictly asynchronous
random-node updates, OR explicitly require the learned `J` matrix to
be bipartite-block-structured (massive restriction on expressivity).

**Status: Round 5+ architecture redesign + hardware re-synthesis.**

### 🚨 FATAL #3 — Hopfield Capacity Mode Collapse

A pairwise Ising model `E = z^T J z` is mathematically a Hopfield
network. Hopfield's classic capacity limit: `C ≈ 0.14·N`.

For `N = 128` capped at `MAX_DEGREE = 32`, **max ~18 independent
local minima can be stored.** The deep EBM aims to express thousands
of semantically distinct linguistic basins. **Distillation forcibly
collapses thousands of deep-EBM modes into <20 degenerate basins.**

This is exactly the catastrophic repetitive-text failure mode that
DBAE-EBM was supposed to avoid.

**Rescue:** Deploy a Restricted Boltzmann Machine (RBM) by
introducing auxiliary "hidden" spins on the FPGA. Polynomial
storage capacity instead of `0.14N`. This requires a fundamentally
different FPGA bitstream.

**Status: Round 5+ architecture redesign.**

### 🚨 FATAL #4 — Taylor-Induced Spurious Black Holes

Deep EBMs are **non-convex**. Their Hessians contain negative
eigenvalues. Taylor expansion sets `J = ∇²E_deep` at an anchor,
inheriting those negative eigenvalues. On unbounded ℝᵈ, the
quadratic surrogate drops to `−∞` along negative-eigenvalue
eigenvectors.

**On the FPGA's bounded ±1 hypercube**, the global minima physically
manifest at the **extreme corners aligned with these negative axes**.
The Glauber sampler plunges into these spurious corners — which the
original deep EBM correctly evaluated as **high-energy junk**.

**The hardware will confidently output pure gibberish** that the
deep EBM would have rejected.

**Rescue:** Abandon Taylor expansion. Distill via **Adversarial
Score Matching** directly on the hypercube, enforcing `J` to be
**positive semi-definite** to mathematically penalize spurious corner
states.

**Status: Software fix, empirically tunable before scaling.**

### 🚨 FATAL #5 — Higher-Order Logic Eradication

The "Kona-parity" Phase-3 model relies on `k = 15` verifiers
imposing **non-linear logical syntaxes** — XOR, parity, tree-depths,
etc. A purely observable quadratic Ising is mathematically restricted
to **pairwise (2-SAT)** interactions.

**Pairwise graphs are physically incapable of representing XOR**
without hidden variables (this is a textbook computational complexity
result). Distillation strips all `>2`-order logical composition.

The deployed FPGA surrogate optimizes an **ungrounded energy
landscape structurally incapable of enforcing the exact logical
constraints the verifier was trained to respect.**

**Rescue:** Quadratic gadget reductions via hidden hardware spins to
natively mediate 3-way and 4-way interactions on the FPGA. Same
"add hidden spins" pattern as Finding #3 but for a different
expressivity gap.

**Status: Round 5+ architecture redesign.**

### ⚠️ DEGRADING #6 — AND-Composition Anchor Fragmentation

`Taylor(Σ E_i) = Σ Taylor(E_i)` only at the **same anchor state**.
Different verifiers have different "natural" anchors (their training
data's mean/mode). Composing their individually-anchored Taylor
expansions to a single global `J` evaluates each verifier outside its
valid radius — wildly hallucinated gradients.

**Measurement on prototype:** compute individual verifier pass rates
on samples from the composite `(J, h)` surrogate vs samples from the
continuous deep `E_total`. If they diverge, anchor fragmentation is
real.

### ⚠️ DEGRADING #7 — Silent Off-Manifold Unit Testing

Acceptance Gate condition 4 (distillation residual <X%) measures on
the **on-manifold replay buffer**. Production text generation starts
from **noise or masked states** — completely OOD. The Taylor
surrogate interpolates wildly outside its anchor's convergence
radius. Unit tests pass; production fails on adversarial inputs.

**Measurement on prototype:** compute distillation residual on
**uniformly random hypercube corners** (max-entropy states) and
intermediate MCMC paths, not just on-manifold anchors.

### ⚠️ DEGRADING #8 — Thermodynamic Freezing vs Continuous Tunneling

Training uses Parallel Tempering (β ∈ {0.1, 0.4, 0.7, 1.0}) to
tunnel across deep EBM energy barriers via the continuous interior of
`[-1, 1]^d`. The FPGA deploys a **single fixed-β Glauber chain
restricted to discrete Hamming-1 edges.**

We're carving a topological landscape assuming thermodynamic wormholes
exist, then mapping deployment onto a frozen discrete maze.
**Exponentially scaling mixing times** when crossing barriers the
continuous chain trivially bypassed → severe ergodicity loss.

**Measurement on prototype:** compare autocorrelation time and
mode-switching rate of discrete FPGA chain vs parallel multi-β MLD
chain.

### ⚠️ DEGRADING #9 — Q8.8 Hardware Saturation Clipping

KV260 Q8.8 precision rigidly caps weights at `[−128.0, +127.99]`.
**Verifier cliffs in the deep EBM require massive gradients to wall
off invalid states.** Q8.8 clipping converts impenetrable constraint
walls into gentle traversable slopes. Glauber probability
`P = σ(−2β·ΔE)` shatters when `ΔE` is artificially clamped.

**Measurement on prototype:** histogram unclipped least-squares `J`
weights; compare strict verifier pass rate of the exact Q8.8 clipped
matrix against the unquantized surrogate.

### 📝 COSMETIC #10 — α_t Trajectory Disintegration

`inf_t α_t > 0.1` relies on **continuous, differentiable vector
fields** of MLD in arctanh-dual space. The FPGA executes
**piecewise, uncorrelated Markov bit-flips**. The infinitesimal
geometry the Zenil α_t calculus requires is destroyed by macroscopic
bit-flip transitions.

Static endpoint distribution might match, but trajectory continuity
is severed. **Reviewers will flag that runtime safety bounds are
functionally void in the discrete hardware domain.**

## Implications

### For Phase-3 architecture

The architecture **as currently designed cannot ship.** The five
fatal findings divide into two clusters:

**Hardware-architecture mismatch (Findings 1, 2, 3, 5):**
- d=128 maximum spin count vs d∈{256,512} latent
- Synchronous parallel Glauber breaks detailed balance
- Hopfield capacity is ~18 modes, not thousands
- Pairwise graphs can't represent XOR

**These four require either:**
- A new FPGA bitstream with auxiliary hidden spins (RBM-style)
- OR strictly-bipartite-block J matrices (massive expressivity
  restriction) and async/checkerboard firmware
- OR abandon the FPGA path entirely

**Distillation methodology (Finding 4):**
Taylor expansion fundamentally produces spurious corner minima.
Software-fixable via Adversarial Score Matching with PSD constraint.

### For already-published results

**exp1081's "FPGA scale benchmark"** (.84) reported a 12,788× speedup
of FPGA over CPU at N=64 spins. This comparison is **academically
suspect** because:
- The FPGA's synchronous parallel Glauber on arbitrary J doesn't
  converge to the Boltzmann distribution
- The CPU's correct Gibbs sampler does
- "Speedup" is comparing apples to oranges

The position paper's Phase-2 hardware section needs to be **re-framed
with this caveat**, OR exp1081 needs to be re-run with strictly
async / bipartite-block J to validate the comparison.

### For .85 planner

The MANDATORY-NEXT-MILESTONE PRIORITIES section in
`ops/known-issues.md` already calls for a FPGA-vs-GPU baseline
experiment. **That experiment must now also include a sampler
correctness validation** — verify the FPGA sampler converges to
the correct Boltzmann distribution before reporting any speedup
numbers.

If the FPGA can't be made to sample correctly without firmware
changes, the entire Phase-2 hardware story needs reframing as
"correctness is established theoretically, hardware speedup is
contingent on bitstream redesign for asynchronous sampling."

## Recommended next steps

1. **Update memory** to mark Phase-3 architecture as NOT settled.
   The pivot-back to DBAE-EBM was premature; we have 5 fatal findings
   that the Round 1-2-3 chain missed.

2. **Update `project_dbae_ebm_phase3.md`** to reflect that the
   architecture is back to "open question, requires Round 5+."

3. **Tag exp1081's verdict** in research-complete.yaml or a
   superseding artifact: the speedup measurement is contingent on
   sampler correctness validation.

4. **Round 5 Deep Think prompt**: with the five fatal findings in
   hand, ask Deep Think to redesign Phase-3 around an RBM-with-hidden-
   spins FPGA bitstream + asynchronous/bipartite-only sampling +
   PSD-constrained adversarial-score-matching distillation. The
   redesign is non-trivial.

5. **Pause the small-scale prototype** that was about to start.
   Building a prototype against a known-broken architecture wastes
   compute.

## The bigger lesson

**The Round 1 → 2 → 2.5 → 3 → audit chain demonstrates that
architecture decisions need iterated adversarial review even when
each round looks rigorous in isolation.** Three rounds of "this is
the right architecture" missed five fatal flaws that one
hostile-reviewer round caught.

This is also methodological evidence for the position paper:
*Energy-based reasoning architectures benefit from forcing every
design through an adversarial-reviewer round before scaling. The
Round 1-2-3 chain alone would have produced a publication that got
torched on first review.*
