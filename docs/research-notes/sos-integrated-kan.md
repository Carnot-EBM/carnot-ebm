# SOS-Integrated KAN: Type-Level Safety Constraints via Burer-Monteiro Factorization

**Status:** Architecture specification and proof-quality writeup. **Read
`literature-priority-audit.md` first** — the SOS-on-derivative move is
direct prior art (UMNN, Wehenkel-Louppe NeurIPS 2019) and the analytic
SOS-integration is also prior art (Jaini et al. SoS Polynomial Flows
NeurIPS 2019). Monotonic KAN architecture exists (MonoKAN, arXiv
2409.11078). Our contribution is the *integration* of these into a
KAN-edge architecture with analytic Fubini-collapsed B-spline integral
basis, Burer-Monteiro factorization for the derivative-coefficient
matrix, AST-level type-checking for MILP verifier elimination, Parity-
Matrix routing for arbitrary mixed-direction multi-axis monotonicity,
and B-spline partition-of-unity identity-initialization. Each
ingredient on its own is small; the combination is what we deploy.

## 1. Setup

Let `ψ: D → ℝ` with `D = [x_min, x_max] ⊂ ℝ` be a single KAN edge
function. Carnot's Tier-4 verification (REQ-VERIFY-038, REQ-VERIFY-040)
requires `ψ` to satisfy combinations of:

- **Monotonicity:** `ψ'(x) ≥ 0` on `D` (the safety axis).
- **Non-negativity:** `ψ(x) ≥ 0` on `D` (energy-component
  non-negativity).
- **Concavity:** `ψ''(x) ≤ 0` on `D` (saturation requirements for
  some EBM heads).
- **Lipschitz bound:** `0 ≤ ψ'(x) ≤ L` on `D` (Langevin-MCMC
  stability requirement; see `distill.py` Section 4).

The standard Carnot pipeline (`KAEMEnergy` etc.) trains the KAN
without these constraints, then uses **MILP verification** (Marabou,
Gurobi) to *check* whether the trained network satisfies them.
When violations are found (Exp 972: 11 monotonicity/boundary
violations on KAEMEnergy), an offline **repair operation** patches
spline coefficients until the MILP verifier accepts.

This post-hoc repair has two recurring problems: (i) repair can
introduce new violations elsewhere; (ii) the MILP verifier's branch-
and-bound search is slow and a frequent CI bottleneck. The
**SOS-Integrated KAN** approach replaces "verify-then-repair" with
*verifier-by-construction*: monotonicity, non-negativity, concavity,
and Lipschitz become **type-level invariants** of the AST that the
MILP verifier can prove `True` without branch-and-bound.

## 2. The Core Construction (Approach 3 from the Round 4 Dialogue)

### 2.1 Monotonicity + Non-negativity (Round 1)

By the Fundamental Theorem of Calculus, `ψ ≥ 0` and `ψ' ≥ 0` on `D`
iff `ψ'(x) ≥ 0` everywhere on `D` and `ψ(x_min) ≥ 0`. We
parameterize the *derivative* as a Sum of Squares of B-splines:

```
ψ'(x)  =  Σ_{m=1}^{M}  (Σ_{i=1}^{N} V_{i,m} · B_{i,p}(x))²        (1)
```

where `V ∈ ℝ^{N × M}` is *unconstrained* (Burer-Monteiro factor; see
§2.4 for why `M ≥ 2`), `B_{i,p}` is the i-th B-spline of degree p
on a fixed knot grid over `D`. Expanding the square,

```
ψ'(x)  =  Σ_{i,j=1}^{N}  (V V^T)_{i,j} · B_{i,p}(x) · B_{j,p}(x).
```

Integrate from `x_min` and add a non-negativity-anchoring constant
`c² ≥ 0`:

```
ψ(x)  =  c²  +  Σ_{i,j=1}^{N}  (V V^T)_{i,j} · Φ_{i,j}(x)        (2)
```

where the **integral basis**

```
Φ_{i,j}(x)  :=  ∫_{x_min}^x  B_{i,p}(t) · B_{j,p}(t)  dt          (3)
```

is a *static, precomputable* piecewise polynomial. For the canonical
choice of degree-1 hat-function B-splines, `B_i · B_j` is a
piecewise quadratic with compact support `|i − j| ≤ 1`, so the
basis is banded and the cost is `O(N · 1) = O(N)` per evaluation.

**Lemma 2.1.** *Construction (2) satisfies `ψ(x) ≥ 0` and `ψ'(x) ≥ 0`
for all `x ∈ D` and all `V ∈ ℝ^{N × M}`, `c ∈ ℝ`.*

*Proof.* `ψ'(x)` is a sum of squares (Equation 1) hence `≥ 0`.
`ψ(x_min) = c² ≥ 0`. By the FTC, `ψ(x) = ψ(x_min) + ∫_{x_min}^x
ψ'(t)dt ≥ ψ(x_min) ≥ 0`. ∎

This is the **type-level invariant**: any `(V, c) ∈ ℝ^{N×M} × ℝ`
gives a valid `ψ` automatically. No optimization-time projection,
no MILP-time check, no repair operation.

### 2.2 Concavity Composition (Round 2)

To additionally enforce `ψ'' ≤ 0` while keeping `ψ' ≥ 0` and
`ψ ≥ 0`, parameterize the second derivative as a *negated* SOS and
use *backward integration* to anchor the derivative minimum at the
right boundary `x_max` (since `ψ'' ≤ 0` means `ψ'` is non-
increasing, hence its minimum is at `x_max`):

```
ψ''(x)  =  −Σ_m (Σ_i V_{i,m} · B_{i,p}(x))²                     (4)
ψ'(x)   =  c_deriv²  +  ∫_x^{x_max} (−ψ''(t)) dt                (5)
ψ(x)    =  c_bias²  +  ∫_{x_min}^x ψ'(u) du                      (6)
```

Substituting (5) into (6) yields a *nested double integral*. The
Round-2 dialogue framed this as a problem for autograd graphs
(nested dynamic integration is hazardous); the Round-3 fix is to
collapse the double integral analytically via Fubini's theorem.

### 2.3 Fubini-Collapsed Single-Layer Basis (Round 3)

**Theorem 2.2 (Fubini-collapsed SOS-integrated KAN).** *The double
integral in equations (5)–(6) collapses to a single-layer
contraction with a static piecewise polynomial basis*

```
Ω_{i,j}(x)  :=  (x − x_min) · ∫_x^{x_max} B_{i,p}(t)·B_{j,p}(t) dt
                +  ∫_{x_min}^x  (t − x_min) · B_{i,p}(t)·B_{j,p}(t) dt.   (7)
```

*Specifically,*

```
ψ(x)  =  c_bias²  +  c_deriv² · (x − x_min)
         +  Σ_{i,j} (V V^T)_{i,j} · Ω_{i,j}(x).                  (8)
```

*Both `Ω_{i,j}` and `Ω'_{i,j} = ∫_x^{x_max} B_iB_j dt` and
`Ω''_{i,j} = −B_i(x)B_j(x)` are static piecewise polynomials, fully
precomputable. The Autograd graph of `ψ` contains only `Square`,
`Mul`, `Add`, and a sparse `einsum` — no integrals, no quadrature.*

*Proof.* The nested double integral in (5)–(6) is

```
I(x)  :=  ∫_{x_min}^x  ∫_u^{x_max}  S(t)  dt  du
```

where `S(t) := Σ_m (Σ_i V_{i,m}·B_i(t))² ≥ 0`. The integration
region in `(u, t)`-coordinates is `R = {(u, t) : x_min ≤ u ≤ x,
u ≤ t ≤ x_max}`. Apply Fubini's theorem to swap the order:

- For `t ∈ [x_min, x]`: `u` ranges over `[x_min, t]` (the `u ≤ t`
  constraint binds since `t ≤ x`).
- For `t ∈ [x, x_max]`: `u` ranges over `[x_min, x]` (the `u ≤ x`
  constraint binds).

So

```
I(x)  =  ∫_{x_min}^x S(t) · (t − x_min) dt
         +  ∫_x^{x_max} S(t) · (x − x_min) dt
      =  (x − x_min) ∫_x^{x_max} S(t) dt
         +  ∫_{x_min}^x (t − x_min) S(t) dt.
```

Factor out the `(VV^T)_{i,j}` coefficients of `S(t)` and identify
the basis (7). ∎

**Verification of derivative identities.** The Leibniz rule applied
to `Ω_{i,j}(x)` gives

```
Ω'_{i,j}(x)
  = ∫_x^{x_max} B_iB_j dt − (x − x_min) · B_i(x) · B_j(x)
    + (x − x_min) · B_i(x) · B_j(x)
  = ∫_x^{x_max} B_i(t) · B_j(t) dt.                              (9)
```

The boundary terms cancel exactly. Differentiating once more,

```
Ω''_{i,j}(x)  =  −B_i(x) · B_j(x).                              (10)
```

Substituting (10) into the second derivative of (8),

```
ψ''(x)  =  Σ_{i,j} (VV^T)_{i,j} · Ω''_{i,j}(x)
        =  −Σ_{i,j} (VV^T)_{i,j} · B_i(x) · B_j(x)
        =  −Σ_m (Σ_i V_{i,m} · B_i(x))²  ≤  0
```

confirming concavity by construction. ∎

**Smoothness for `p = 1` (hat function basis).** `B_i · B_j` is a
piecewise quadratic, `C⁰` continuous (derivative jumps at knots).
`Ω''_{i,j}` is therefore `C⁰` piecewise quadratic. One integration
gives `Ω'_{i,j}` as `C¹` piecewise cubic. Another gives `Ω_{i,j}`
as `C²` piecewise quartic. Hardware-evaluation cost: a static
degree-4 polynomial, FLOP-indistinguishable from a standard cubic
spline.

### 2.4 Burer-Monteiro `M ≥ 2` Over-parameterization

The naive parameterization `ψ'(x) = Σ_i α_i² · B_i(x)·B_j(x)` with
`α ∈ ℝᴺ` (i.e. `M = 1`) has a known optimization pathology: when
the optimal `ψ` requires `α_i = 0` somewhere in the domain, the
gradient `∇_α (α²) = 2α → 0` and training stalls in flat regions.

The **Burer-Monteiro factorization** with `M ≥ 2` writes the
positive-semi-definite coefficient matrix as `M_{ij} := (VV^T)_{ij}`
with `V ∈ ℝ^{N × M}`. When `M = 2`, the parameter `V_{i,:}` lies in
`ℝ²`; driving `M_{ii} → 0` requires `V_{i,:} → 0` *vector-wise*,
but the gradient direction is `2V_{i,:}` which remains aligned with
the descent direction even as the magnitude shrinks. **Parameters
do not "die" in flat regions; they smoothly rotate around the
origin in `ℝ²`.**

This is the standard Burer-Monteiro 2003 (*A nonlinear programming
algorithm for solving semidefinite programs via low-rank
factorization*) trick adapted to our SOS context.

### 2.5 Lipschitz Bound via Jensen + Soft Projection

For Langevin-MCMC stability we additionally need `ψ'(x) ≤ L`.
Apply Jensen's inequality to (1):

```
(Σ_i B_i(x) · V_{i,m})²  ≤  Σ_i B_i(x) · V_{i,m}²
                          ≤  max_i V_{i,m}²     (since Σ B_i ≡ 1).
```

Summing over `m`:

```
ψ'(x)  ≤  Σ_m max_i V_{i,m}²  ≤  max_i ‖V_{i,:}‖²              (11)
```

So `ψ'(x) ≤ L` is *guaranteed* if `max_i ‖V_{i,:}‖² ≤ L`, i.e.
`‖V_{i,:}‖ ≤ √L` for every row `i`. Enforce via **soft projection**
on the forward pass:

```
Ṽ_{i,:}  =  V_{i,:} · min(1, √L / ‖V_{i,:}‖)                   (12)
```

This is a non-clipping projection — gradients flow through unchanged
when the bound is inactive, and only contract proportionally when
`‖V_{i,:}‖ > √L`. This is *sufficient, not necessary* (the actual
maximum of `ψ'` is generally smaller than `max_i ‖V_{i,:}‖²`), so
target `L` may need empirical tuning — but **conservative
Lipschitz bounds in EBMs actively *improve* Langevin-MCMC stability**
(Wu et al. 2018 on Lipschitz-EBM training; Salakhutdinov 2010 on
sampling stability). The slight over-constraint is a feature.

## 3. Multi-Axis Monotonicity via Parity Matrix (Round 3)

The KAN architecture is `f(x) = Σ_q Φ_q(Σ_p ψ_{q,p}(x_p))` with
outer-layer width `H` (number of `Φ_q` summands) and inner-layer
edges `ψ_{q,p}` per (outer-index, input-axis) pair.

For a single input axis `x_k` with desired direction `T_k ∈
{−1, 0, +1}` (decreasing, free, increasing) and an outer-node
parity assignment `P_q ∈ {−1, +1}`, define the **edge parity**
`E_{k,q} := T_k · P_q`. Parameterize `ψ_{k,q}` as

- `E_{k,q} = +1` → SOS-integrated increasing edge (Equation 8).
- `E_{k,q} = −1` → negated SOS-integrated edge (gives `ψ_{k,q}'(x)
  ≤ 0`, hence decreasing).
- `E_{k,q} = 0` → unconstrained B-spline (no monotonicity in `x_k`
  for the term feeding outer node `q`).

For `Φ_q` itself, set its sign by `P_q`: `Φ_q` is increasing if
`P_q = +1`, decreasing if `P_q = −1`. Implementation via the same
SOS-integrated trick on the outer aggregate.

**Lemma 3.1 (Multi-axis monotonicity by parity).** *For any input
axis `x_k` with `T_k ∈ {−1, +1}`,*

```
∂f/∂x_k  =  Σ_q  Φ_q'(z_q) · ψ_{k,q}'(x_k)
```

*where each summand has sign `P_q · E_{k,q} = P_q · (T_k · P_q) =
T_k`. Hence all summands share sign `T_k`, and `∂f/∂x_k` has sign
`T_k` globally on `D`. For axes with `T_k = 0`, no constraint is
imposed.*

*Proof.* Direct application of the chain rule and the sign
properties of the SOS-integrated edges. ∎

**Why this is `O(N_edges)`, not `O(2^d)`.** A naive approach would
duplicate the architecture per direction-mixture combination,
giving `O(2^d)` parameters for `d` constrained axes. The Parity
Matrix achieves the same expressive power by *partitioning the
outer-layer width* between `+`-parity and `−`-parity nodes, with
each axis's monotonicity routed through edge-parity assignment.
Total parameter count equals an unconstrained KAN of the same
shape.

## 4. Identity-Initialization via Partition-of-Unity (Round 2)

The standard Burer-Monteiro warm-start is `V ~ N(0, σ²)` with
small `σ`. This gives `ψ'(x) ≈ 0` at `t = 0`, which is problematic
because the network "starts cold" — gradients flow through `2V`
which is small, and the network takes many steps to learn even an
identity transformation.

**The fix:** exploit the B-spline partition-of-unity property
`Σ_i B_{i,p}(x) ≡ 1` on `D` (true for any standard B-spline basis).
Initialize

```
V_{i, 1}  =  1 + ε_{i,1}    with  ε ~ N(0, 10⁻⁴)
V_{i, m≥2}  =  ε_{i,m}      with  ε ~ N(0, 10⁻⁴)
```

Then at `t = 0`:

```
ψ'(x)  ≈  (Σ_i 1 · B_i(x))²  +  O(ε²)  =  1²  =  1.
```

Integrating: `ψ(x) ≈ x − x_min + c² ≈ x` (with appropriate `c`
choice). **The KAN starts as an identity transformation with
fully-alive gradients** — `∇_{V_{i,1}} ψ' ∝ 2 · 1 · B_i(x) ≈ B_i(x)`.

## 5. Knot Adaptation Compatibility (Round 2)

Modern KANs adapt knot positions during training via Boehm's
subdivision (discrete) or learned `Δt_i` (continuous). Both
commute with the SOS structure:

**Discrete (Boehm subdivision).** Boehm's algorithm gives a sparse
linear transformation `T` such that the new B-spline coefficients
`α_new = T · α_old` give an identical curve over a refined knot
grid. For our SOS-integrated KAN, applying `T` to the *columns* of
`V` gives `V_new = T · V_old`, and the SOS coefficient matrix
transforms as `(V_new V_new^T) = T(V_old V_old^T) T^T` — the
quadratic form is preserved exactly. The integral basis `Ω` recomputes
on the refined grid; the curve `ψ(x)` is unchanged.

**Continuous (learned `Δt_i`).** For `p = 1` hat functions on a
non-uniform grid with knot intervals `Δt_i`, the closed-form
integrals are

```
∫ B_i² dx        =  (1/3) · (Δt_{i-1} + Δt_i)    (FEM mass diagonal)
∫ B_i · B_{i+1} dx  =  (1/6) · Δt_i              (FEM mass off-diagonal)
```

These are textbook FEM piecewise-linear mass matrix entries (Strang-
Fix 1973). Differentiating with respect to `Δt_i` is straightforward
tensor multiplication — **no numerical quadrature in the autograd
graph**.

## 6. MILP Verification Trivialization

The Carnot Tier-4 MILP verifier (Marabou / Gurobi) consumes the
trained network as an Abstract Syntax Tree and checks safety
properties. With the SOS-Integrated KAN, the AST signature is

```
Add(
    Square(c_bias),
    PosLinear(c_deriv², (x − x_min)),
    CollapsedDoubleIntegral(NegativeSOS(B-spline-products))
)
```

The verifier's typing rules then immediately give:

- **Non-negativity** of `ψ(x)`: every leaf is `≥ 0` (`Square`,
  `PosLinear` with `c_deriv² ≥ 0` and `x ≥ x_min`,
  `CollapsedDoubleIntegral` of a `NegativeSOS` integrand interpreted
  via the FTC chain).
- **Monotonicity** of `ψ(x)`: the derivative of the AST is `c_deriv²
  + ∫_x^{x_max} SOS(t) dt`, both terms `≥ 0`.
- **Concavity** of `ψ(x)`: the second derivative is `−SOS(x) ≤ 0`.

**No branch-and-bound search is required.** The MILP verifier
produces `True` proofs by structural induction over the AST, in
`O(|AST|)` time. **The 11 violations from Exp 972 cannot exist by
construction.**

## 7. Implementation Status (2026-04-28)

The *transpiler-side* of this architecture (Approach 3 PT-PCD) is
implemented at `python/carnot/hardware/transpiler/distill.py` and
tested. **The SOS-Integrated KAN architecture itself is not yet
implemented** — it lives in this docs-and-spec form only. Concrete
deliverables for actual integration:

1. **`python/carnot/models/sos_kan.py`**: SOS-Integrated KAN edge
   module (PyTorch). Parameterizes `V ∈ ℝ^{N×M}`, `c_bias`,
   `c_deriv`. Forward pass uses precomputed `Ω` basis.
2. **Tests** at `tests/python/test_sos_kan.py`: monotonicity by
   construction (sweep `V` randomly, verify `ψ' ≥ 0` numerically
   on a 1024-point grid for 100 random `V`); concavity if used;
   identity-init produces `ψ(x) ≈ x` at `t = 0`; gradients flow
   through Burer-Monteiro factorization without dying.
3. **MILP verifier wiring** at `python/carnot/verify/sos_kan_typecheck.py`:
   takes a trained `sos_kan.SoSKANEdge` and emits a `True` proof
   for monotonicity/non-negativity/concavity by AST inspection,
   replacing the current Marabou call site for KAEMEnergy.
4. **Retraining KAEMEnergy with the new architecture**: replaces
   the standard B-spline KAN with SOS-Integrated edges. Validate
   that the 11 violations from Exp 972 no longer occur.

## 8. Attribution

- **Wehenkel, A. & Louppe, G. (2019).** "Unconstrained Monotonic
  Neural Networks." NeurIPS. [arXiv:1908.05164]. SOS-on-derivative
  for monotonic NNs is theirs.
- **Jaini, P., Kobyzev, I., Brubaker, M. & Yu, Y. (2019).**
  "Sum-of-Squares Polynomial Flow." NeurIPS. Analytic SOS-integration
  for monotonic transformations is theirs.
- **Calzada, J., Garcia-Crespo, A., et al. (2024).** "MonoKAN:
  Certified Monotonic Kolmogorov-Arnold Network." arXiv:2409.11078.
  Direct prior art for monotonic KAN architecture.
- **Liu, Z. et al. (2024).** "KAN: Kolmogorov-Arnold Networks."
  arXiv:2404.19756. The base KAN paper we extend.
- **Durkan, C., Bekasov, A., Murray, I. & Papamakarios, G. (2019).**
  "Neural Spline Flows." NeurIPS. arXiv:1906.04032. Adjacent
  monotonic-spline architecture.
- **Sill, J. (1997).** "Monotonic Networks." NIPS. Foundational.
- **Burer, S. & Monteiro, R. (2003).** "A nonlinear programming
  algorithm for solving semidefinite programs via low-rank
  factorization." Math. Programming 95:329–357. The factorization
  trick.
- **Markov, A. (1948); Lukacs, F. (1918).** Foundational results on
  SOS representations of non-negative polynomials.
- **Strang, G. & Fix, G. (1973).** *An Analysis of the Finite
  Element Method.* For the FEM mass-matrix integrals.

**Our contribution is the *integration* of these into a KAN-edge
architecture with type-level safety constraints suitable for
verifier-by-construction deployment. Each ingredient on its own is
prior art. Any external publication must cite all of the above.**
