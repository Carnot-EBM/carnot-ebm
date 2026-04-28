# The Continuous ε-Ising-Rank Theorem and Three Embedding Strategies

**Status:** Mathematical foundation document for the Phase 2 transpiler.
Captures the proofs derived in dialogue with Google Deep Think across
four rounds (2026-04-27) in a self-contained, paper-quality form with
explicit attribution. **Read `literature-priority-audit.md` first** —
the framing here is novel as synthesis but the ingredients are all
classical, and any external publication must cite the prior work
identified there.

This document establishes:

1. **The dual bound (Section 2).** For arbitrary Lipschitz energies
   the spin count required to embed `E: ℝⁿ → ℝ` on Ising hardware
   with `KL ≤ ε` is at least `Ω((L/ε)^(n/2))` (curse of dimensionality
   from metric entropy). For *parametric* (computable, finite-W)
   energies, the bound collapses to `O(n log(1/ε) + W log²(1/ε))`.

2. **Approach 1 (Section 3).** Execution-trace embedding: an explicit
   construction realizing the parametric upper bound for sparse
   hardware with formal `KL ≤ ε` via QUBO penalty gadgets. The KL
   bound proof (Theorem 3.1) is a textbook log-sum-exp partition-
   function argument and is included for self-containment, not as a
   research result.

3. **Approach 2 (Section 4).** Arc-Cosine V-Statistic Lift: dense
   optical hardware natively realizes a 0th-order arc-cosine RKHS
   approximation, dropping spin count from `O(1/ε²)` (naive Barron)
   to `O(‖E‖_RKHS / ε)`. Direct application of Cho-Saul (2009) and
   Kar-Karnick (2012).

4. **Approach 3 (Section 5).** Native Thermodynamic Distillation:
   the production path for transformer-class verifiers where W is
   too large for execution-trace compilation. Empirical KL bound via
   PT-PCD training (Desjardins-Courville-Bengio 2010 lineage) with
   three falsifiable diagnostics. No formal KL guarantee — engineering
   trade-off.

## 1. Setup and Notation

Let `E: ℝⁿ → ℝ` be a continuous parametric energy function, where
"parametric" means `E = f_θ` for some fixed-architecture neural
network `f` with `W = |θ|` real-valued parameters, evaluated at
fixed-point precision (we use `b`-bit fixed-point; the rate constants
absorb `b`). Let `D = [-L, L]ⁿ ⊂ ℝⁿ` be a bounded domain of
interest, and let `β ∈ [β_min, β_max]` be an inverse-temperature
range over which the deployed sampler must reproduce
`P_E(x; β) ∝ exp(−β E(x))`.

The target is to construct an Ising machine — a coupling matrix
`J ∈ ℝ^{N×N}` and local field `h ∈ ℝᴺ` over `N` binary spins
`s ∈ {−1, +1}ᴺ` with energy `E_I(s) = −sᵀJs − hᵀs` — together
with an encoder `φ: ℝⁿ → {−1, +1}ᴺ` and a decoder
`ψ: {−1, +1}ᴺ → ℝⁿ` such that the marginal pushforward of the
Ising distribution under ψ approximates `P_E` in KL divergence.

Define the **continuous ε-Ising-rank** of `E` over `D` at temperature
range `[β_min, β_max]` as

```
R_ε(E; D, β_range) := min { N : ∃ (J, h, φ, ψ) with N spins
                              and KL(P_E ‖ ψ_* P_I) ≤ ε  ∀β ∈ β_range }.
```

## 2. The Dual Bound (Continuous Ising-Rank Theorem)

### 2.1 Lower Bound (Curse of Dimensionality)

**Proposition 2.1 (folklore from metric entropy).** *For arbitrary
Lipschitz `E ∈ Lip_L(D)`, the continuous ε-Ising-rank satisfies*

```
R_ε(E; D, β_range)  ≥  Ω((L · diam(D) · β_max / ε)^{n/2}).
```

*Sketch of proof.* The Kolmogorov-Tikhomirov (1959) ε-entropy of
`Lip_L(D)` in the sup norm is `H_ε(Lip_L(D)) = Θ((L · diam(D)/ε)^n)`.
Translating to KL via the bounded-log-density argument (Yang-Barron
1999) introduces the `β_max` factor and gives the stated bound. ∎

This is *not novel* and is included only to justify the claim that
the parametric upper bound is a non-trivial collapse.

### 2.2 Upper Bound (Parametric Collapse)

**Theorem 2.2 (Parametric Continuous Ising-Rank).** *Let `E = f_θ`
be a parametric energy function realized as a feed-forward
arithmetic circuit with `W` parameters in `b`-bit fixed-point. Then
for any `ε > 0`,*

```
R_ε(E; D, β_range)  ≤  O(n · log(1/ε) + W · log²(1/ε)).
```

*The constant absorbs `b`, the QUBO-gadget locality, the depth of
the arithmetic circuit, and the relevant spectral gap of the
penalty term.*

*Proof.* By construction (Approach 1, Section 3 below). The first
term `O(n · log(1/ε))` is the m-bit fractional binary encoding of
the visible-spin block; `m = ⌈log₂(...)⌉` so `N_vis = n·m =
O(n · log(1/ε))`. The second term `O(W · log²(1/ε))` comes from
allocating `O(b²) = O(log²(1/ε))` auxiliary spins per arithmetic
gate (multiplier-gadget cost) times `W` gates total. ∎

**Where the novelty lives.** Both ingredients (metric-entropy lower
bound and QUBO-gadget upper bound) are classical. The novelty is the
*juxtaposition* — explicitly naming the gap as a parametric collapse
rather than treating the two as separate facts in separate
literatures. Cite **Lasserre (2001), Lucas (2014), Kolmogorov-
Tikhomirov (1959), Yang-Barron (1999)** as the four-source backbone.

## 3. Approach 1: Execution-Trace Embedding

### 3.1 Construction

Let `f_θ` be evaluated at `b`-bit fixed-point. The forward pass
decomposes into elementary arithmetic ops (add, multiply, ReLU,
comparator, sign). We encode:

- **Visible spins** `s_vis ∈ {−1, +1}^{N_vis}` representing the
  m-bit Gray-code of `x ∈ D` (m = ⌈log₂(4nL·L_E·β_max/ε)⌉,
  giving spatial quantization step `δ = (2L)/2^m ≤ ε/(4·β_max·L_E)`).
  See `gray_code.py` and `sos-integrated-kan.md` for the encoder
  details.
- **Hidden spins** `s_hid ∈ {−1, +1}^{N_hid}` encoding the
  deterministic execution trace of every arithmetic op in `f_θ`'s
  forward pass. Each op `g` carries a 2-local or 3-local QUBO
  penalty `P_g(s) ≥ 0` such that `P_g(s) = 0` iff the spins
  encoding `g`'s inputs and outputs are consistent.

Let `Γ(s) := Σ_g P_g(s) ≥ 0`. Define the readout
`E_readout(s) := Σ_k 2^k · h_k^out · s_k^out` reading off the
output bits of the arithmetic trace as a fixed-point real.

The Ising energy is

```
E_I(s)  =  E_readout(s)  +  λ · Γ(s).
```

For each visible state `v`, exactly one hidden trace `h*(v)`
satisfies `Γ(v, h*) = 0`, and at that trace `E_readout(v, h*(v)) =
f_θ(ψ(v)) ± δ` (within the spatial quantization error `δ`).

### 3.2 KL Bound

**Theorem 3.1 (Approach 1 KL bound, Round-3 closed form).** *Let
`P_target(v) ∝ exp(−β · f_θ(ψ(v)))` and let `P_marg(v) := Σ_h
P_I(v, h)` be the visible marginal of the Ising distribution
`P_I(v,h) = (1/Z_I) exp(−β·E_I(v,h))`. If*

```
λ  ≥  (1/β_min) · (N_hid · ln 2 + ln(1/ε))  +  ΔE,
```

*with `ΔE := sup_x E(x) − inf_x E(x)`, then*

```
D_KL(P_target ‖ P_marg)  ≤  ε    for all β ∈ [β_min, β_max].
```

*Proof.*

**Step 1: Per-state lower bound on the marginal.** For each visible
state `v`, the unique valid trace `h*(v)` satisfies `Γ(v, h*) = 0`,
so

```
P_marg(v)  =  (1/Z_I) · [ exp(−β·f_θ(ψ(v)))  +  Σ_{h ≠ h*}
              exp(−β·(E_readout(v,h) + λ·Γ(v,h))) ]
           ≥  (1/Z_I) · exp(−β·f_θ(ψ(v))).
```

Letting `Z_P := Σ_v exp(−β·f_θ(ψ(v)))`, this gives

```
P_marg(v)  ≥  P_target(v) · (Z_P / Z_I).        (*)
```

**Step 2: KL bounded by partition-function ratio.** Substitute (*)
into the KL definition:

```
D_KL(P_target ‖ P_marg)
  =  Σ_v P_target(v) · ln(P_target(v) / P_marg(v))
  ≤  Σ_v P_target(v) · ln(Z_I / Z_P)
  =  ln(Z_I / Z_P).
```

**Step 3: Decompose `Z_I` and bound the ratio.** Split `Z_I` over
valid traces (`Γ = 0`) and invalid traces (`Γ ≥ 1`):

```
Z_I  =  Z_valid  +  Z_invalid,         Z_valid = Z_P.
```

Using `ln(1+x) ≤ x` for `x ≥ 0`,

```
ln(Z_I / Z_P)  =  ln(1 + Z_invalid / Z_P)  ≤  Z_invalid / Z_P.
```

**Step 4: Worst-case bounds on numerator and denominator.**

- `Z_P ≥ 2^{N_vis} · exp(−β · sup E)` (all `2^{N_vis}` valid states
  at the energy peak).
- `Z_invalid ≤ 2^{N_vis + N_hid} · exp(−β · (inf E + λ))` (all
  invalid states hallucinating the deepest readout `inf E` with
  minimum penalty `Γ = 1`).

So

```
Z_invalid / Z_P  ≤  2^{N_hid} · exp(−β · (λ − ΔE)).
```

**Step 5: Solve for λ.** Forcing this ratio to ≤ ε and using
`β ≥ β_min`,

```
λ  ≥  (1/β_min) · (N_hid · ln 2 + ln(1/ε))  +  ΔE.
```

This is exactly the stated bound. ∎

**Interpretation of `ΔE`.** The `ΔE` term is *thermodynamically
necessary*: it's the spectral gap the penalty must overcome to
prevent the hardware relaxation dynamics from intentionally
breaking arithmetic gates and falling into the deepest readout
configuration. Without it, hardware "hallucinates" by sacrificing
gate validity for a deeper energy. Round 3 (Deep Think 2026-04-27)
made this physical interpretation explicit.

**Attribution.** Step 1 is the standard "value-of-information /
inclusion bound" for log-partition functions (e.g. Salakhutdinov-
Murray 2008 AIS-context derivations). Steps 2–4 are textbook
log-sum-exp manipulations (Cover-Thomas Ch. 11). The bound's *form*
matches Lucas-2014-style penalty-constant derivations for QUBO
embeddings of constraint-satisfaction problems. **No step is novel.**
The proof is included as a self-contained restatement so the paper
can cite a single equation rather than asking readers to derive it.

### 3.3 Spin-Count

`N_vis = n · m = O(n · log(1/ε))`. `N_hid` scales as `O(W · log²(1/ε))`
because each `b = O(log(1/ε))`-bit arithmetic gate requires
`O(b²) = O(log²(1/ε))` auxiliary spins (multiplier gadgets) and
there are `W` gates. Total: `N = O(n · log(1/ε) + W · log²(1/ε))`,
matching Theorem 2.2. ∎

### 3.4 Practical Limit

For transformer-class verifiers (`W = 10⁶` to `10⁹`) the spin count
exceeds any current Ising hardware budget by 5–8 orders of
magnitude. Approach 1 is feasible only for KAN-tier and Ising-tier
verifiers (`W < 10⁴`). For larger verifiers, use **Approach 3**.

## 4. Approach 2: Arc-Cosine V-Statistic Lift

### 4.1 Construction

For dense optical Ising hardware (photonic SLM, fully-connected
analog crossbars) the natural primitive is `φ(x)ᵀ J φ(x)`. We use
**random threshold features**

```
φᵢ(x)  =  sgn(wᵢᵀ x + bᵢ),            i = 1, …, N
```

with `(wᵢ, bᵢ)` sampled iid from a fixed isotropic distribution
(typically `wᵢ ~ N(0, I/n)`, `bᵢ ~ N(0, 1)`). The Ising energy is
the dense quadratic form `E_I(s) = −sᵀJs − hᵀs` evaluated at
`s = φ(x)`.

We fit `J, h` by **convex ridge regression** on outer products
`φ(x)φ(x)ᵀ` against a corpus of `(x, f_θ(x))` pairs sampled from
the verifier's MCMC.

### 4.2 Approximation Rate

**Theorem 4.1 (Arc-Cosine V-Statistic Approximation).** *Let
`k₀(x, y) := 1 − (1/π)·arccos(xᵀy / ‖x‖‖y‖)` be the 0th-order
arc-cosine kernel (Cho-Saul 2009, Theorem 1). Suppose `f_θ` lies in
the RKHS `H₀` induced by `k₀` with norm `‖f_θ‖_{H₀}`. Then the
convex ridge regression of `f_θ` onto the dense quadratic form
`φᵀJφ + hᵀφ` with `N` random threshold features achieves a uniform
approximation error on `D` of*

```
sup_{x ∈ D}  |f_θ(x) − (φ(x)ᵀ J φ(x) + hᵀ φ(x))|
  =  O(‖f_θ‖_{H₀} / N)
```

*with high probability over the random feature draws.*

*Proof.* By Cho-Saul (2009, §3) the cross-product expectation
`𝔼_w[sgn(wᵀx)·sgn(wᵀy)] = k₀(x, y)` exactly. So the dense
quadratic form `φᵀJφ` is an unbiased random-feature approximation
of the kernel function `k_J(x, y) = Σ_{i,j} J_{ij} k₀(xᵢ, xⱼ)` if
we identify the rows of `J` with kernel-coefficient vectors.

The pairs of features `φᵢ(x)·φⱼ(x)` for `1 ≤ i < j ≤ N` form a
**2nd-order V-statistic** (Hoeffding 1948; specialization to random
feature maps in Kar-Karnick 2012, Theorem 1). The standard V-
statistic concentration gives uniform-over-`D` error decaying as
`O(1/N)` rather than `O(1/√N)` because the variance contribution
from the diagonal `i = j` is `O(1/N)` and the off-diagonal
`(i, j)` pairs benefit from the `C(N,2) ≈ N²/2` averaging. ∎

**Spin-count for KL ≤ ε.** Combining Theorem 4.1 with the standard
L²-to-KL conversion (bounded-log-density: if
`sup |f_θ − f̂| ≤ δ` then for `β ≤ β_max`,
`KL(P_E ‖ P_E_hat) ≤ β_max · δ + O(β_max² · δ²)`), we get

```
N  =  O(β_max · ‖f_θ‖_{H₀} / ε).
```

### 4.3 Standing Caveats

1. **The L²-to-KL conversion** is textbook (e.g. Massart 2007 §3)
   but was glossed over in the Round-3 dialogue. The paper-quality
   writeup must include it explicitly.
2. **The assumption `f_θ ∈ H₀`** is non-trivial. Cho-Saul show that
   `H₀` is dense in `L²(ν)` for natural input distributions ν, and
   that ReLU networks correspond to functions in higher-order arc-
   cosine RKHSs. For the specific Carnot verifiers (KAEMEnergy,
   SC-Energy, ThinkPRM), `‖f_θ‖_{H₀}` should be empirically
   estimated before deployment.
3. **Feature distribution choice.** `wᵢ ~ N(0, I/n)` is the
   standard choice; alternatives (e.g. spherical uniform) shift
   constants but not asymptotic rate.

### 4.4 Hardware Free-Lunch Framing

For optical hardware, the dense quadratic form `φᵀJφ` is
**natively** computed via wave-interference between
`N²/2` cross-terms in `O(1)` clock cycles per evaluation. The host
pays for `N` SLM pixels but gets `N²/2` parametric degrees of
freedom for free. This is the precise sense in which optical
hardware "wins" against the naive Barron bound `N = O(1/ε²)`.

**Attribution.** Theorem 4.1 is direct application of **Cho-Saul
(NeurIPS 2009)** for the kernel identity and **Kar-Karnick (AISTATS
2012)** for the V-statistic rate. Bach (2017) discusses Barron-norm
vs RKHS-norm tradeoffs that motivate the framing. **Our contribution
is the *application* to optical-Ising deployment, not the theorem.**

## 5. Approach 3: Native Thermodynamic Distillation

For verifiers with `W` too large for Approach 1, abandon execution-
trace compilation entirely. Run the heavy verifier backbone on host
CPU/GPU, project to a low-dimensional continuous latent `z`, and
distill the *energy head* `E(z)` directly into a Boltzmann Machine
on the hardware spin budget via **Persistent Parallel Tempering with
Gray-code visible-spin encoding** and five guardrails.

This is the load-bearing production path. The full recipe and
implementation are in `python/carnot/hardware/transpiler/distill.py`
and tested in `tests/python/test_transpiler_distill.py`. There is
**no formal KL guarantee** for Approach 3 — the trade-off is
empirical-KL-via-three-diagnostics for engineering feasibility on
transformer-class verifiers.

**Diagnostic discipline (no payload flashed without all three
passing):**
- **A. KDE overlap** between teacher MCMC samples and student cold-
  chain decoded samples — total-variation distance ≤ 0.10 over a
  32×32 grid (mode-collapse detector).
- **B. Energy histogram overlap** — student energy histogram has no
  secondary spike below `teacher_min − 0.5·σ_teacher` (spurious-
  minimum detector).
- **C. Swap acceptance** between adjacent PT rungs ≥ 15% (broken-
  ladder detector).

See `diagnostics.py` for the implementation and `distill.py` for
the training loop.

**Attribution.** PT-PCD for RBMs is **Desjardins-Courville-Bengio
2010 (AISTATS PMLR 9:145–152)**, which is the canonical reference.
The technique itself — chains across temperatures, replica exchange,
persistent-chain negative phase — is verbatim theirs. Tieleman 2008
(PCD), Tieleman-Hinton 2009 (FPCD), and Igel et al. 2015 (PT-RBM
convergence rates) are also relevant. **Our narrow contribution** is
the Gray-code visible-spin encoder for continuous-input distillation
and the three falsifiable diagnostics. The five-guardrail recipe
(Gray code, geometric β ladder, 5%-rebirth, L2 weight decay,
symmetrize+zero-diagonal) is engineering practice, not novel
theory.

## 6. Practical Selector Logic

The transpiler dispatches across the three approaches based on
hardware kind and verifier size:

```
if hardware.kind == "dense":
    use Approach 2 (arc-cosine RKHS lift)
elif hardware.kind == "sparse":
    if W · log²(1/ε) <= hardware.max_spins:
        use Approach 1 (execution-trace, formal KL bound)
    else:
        use Approach 3 (native distillation, empirical KL via
                        three-diagnostic gating)
```

In Carnot's current state:
- KAEMEnergy and Ising-tier verifiers fit Approach 1 on KV260
  (W ≤ 10⁴, fits 16k-spin budget).
- SC-Energy, ThinkPRM, and DRIFT-class verifiers require Approach 3
  (W » 10⁴).
- Approach 2 is reserved for future dense optical hardware (no
  current Carnot deployment).

## 7. References (Tightly Curated)

- Cho, Y. & Saul, L. (2009). *Kernel Methods for Deep Learning.*
  NeurIPS.
- Cover, T. & Thomas, J. (2006). *Elements of Information Theory*,
  2nd ed., Ch. 11.
- Desjardins, G., Courville, A., Bengio, Y., Vincent, P. & Delalleau,
  O. (2010). *Parallel Tempering for Training of Restricted
  Boltzmann Machines.* AISTATS PMLR 9:145–152.
- Igel, C., Brakel, P. & Glasmachers, T. (2015). *A Bound for the
  Convergence Rate of Parallel Tempering for Sampling Restricted
  Boltzmann Machines.* Theoretical Computer Science.
- Kar, P. & Karnick, H. (2012). *Random Feature Maps for Dot Product
  Kernels.* AISTATS.
- Kolmogorov, A. & Tikhomirov, V. (1959). *ε-Entropy and ε-Capacity
  of Sets in Functional Spaces.* AMS Translations 17:277–364.
- Lasserre, J. (2001). *Global Optimization with Polynomials and the
  Problem of Moments.* SIAM J. Optimization 11:796–817.
- Lucas, A. (2014). *Ising Formulations of Many NP Problems.*
  Frontiers in Physics 2:5.
- Rahimi, A. & Recht, B. (2007). *Random Features for Large-Scale
  Kernel Machines.* NIPS.
- Salakhutdinov, R. & Murray, I. (2008). *On the Quantitative Analysis
  of Deep Belief Networks.* ICML.
- Tieleman, T. (2008). *Training Restricted Boltzmann Machines using
  Approximations to the Likelihood Gradient.* ICML.
- Yang, Y. & Barron, A. (1999). *Information-Theoretic Determination
  of Minimax Rates of Convergence.* Annals of Statistics 27:1564–1599.

For the SOS-Integrated KAN proofs see
`docs/research-notes/sos-integrated-kan.md`. For the cascade-router
proofs see `docs/research-notes/meta-ebm-cascade-router.md`.
