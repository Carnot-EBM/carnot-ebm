# Continuous Ising-Rank Generalization Results

**Status:** Authoritative. Closes REQ-PHASE2-006 with capacity-gap statement.
**Date:** 2026-04-29
**Prompt:** `continuous-ising-rank-generalization-prompt.md`

## The augmented Phase 2 transpiler theorem

> The Gray-code visible-spin encoder reduces $\tau_{\text{int}}$ by factor
> $\Gamma \approx \exp(\kappa(2^{k-1}-1)/T)$ vs standard binary.
>
> Furthermore, for $E \in C^s(\Omega)$ with $\|\nabla E\|_\infty \leq L$,
> the distilled Ising energy approximates $E$ in $L^2(\mu_P)$ at rate
> $$\Delta_{\text{cap}}(k) \leq L\sqrt{d/12} \cdot 2^{-k}$$
> with capacity rate $r_{\text{cap}} = 1$. At structural local minima,
> the MCMC-relevant energy gap mimics the continuous gap at the
> accelerated rate $r = 2$.

## Three load-bearing findings

### 1. Capacity rate is $r_{\text{cap}} = 1$, not 2

KL training-loss bound from Round-10 was $r_{\text{KL}} = 2$ but that's an
*artifact* of KL squaring the $L^2$ distance:
$$\mathrm{KL} \approx \frac{1}{2T^2} \mathrm{Var}(E - \tilde E) \leq \frac{1}{2T^2}\|E - \tilde E\|^2_{L^2} \implies r_{\text{KL}} = 2 \cdot r_{\text{cap}}$$

Capacity is the *binding* constraint for absolute-amplitude tolerance.
For Carnot's $d=100$, $L=10$, $T=1$, $\varepsilon_{\text{cap}} = 0.01$:
- $k_{\text{KL}}^{\min} = 8$ (Round-10)
- $k_{\text{cap}}^{\min} = 12$ (this round)
- **Deployable $k^* = 12$**

KV260 16-bit cap still has 4 bits headroom. Foundation-model regime
needs to verify XTR-0's bit-width handles $k=12$ comfortably.

### 2. Hybrid Coprocessor Pipeline (architectural pattern)

Z3 verifiers are confirmed pathological — the equality verifier
$E(x_1, x_2) = 0\text{ if } x_1=x_2$ lives on a measure-zero diagonal
that Gray-code grid centers almost never intersect. Infinite capacity
gap regardless of $k$.

**Resolution:** asymmetric Hybrid Coprocessor Pipeline:

```
[smooth EBM core]    → FPGA (Gray-code, k=12) → high-throughput PROPOSALS
[Boolean verifiers]  → host CPU (exact Z3)    → post-hoc M-H REJECTION
```

The FPGA generates candidates fast; the CPU filters via deterministic
logical verifiers. The two halves communicate through a streaming
queue. This cleanly resolves the smooth/non-smooth tension in
Carnot's verifier suite.

This is THE deployment pattern for Phase 2 → Phase 3.

### 3. Staircase aliasing introduces spurious modes (caveat)

For diagonal continuous valleys (e.g., $E = (x_1 - x_2)^2 + 0.1(x_1+x_2)$),
Gray-code Hamming-1 moves are axis-aligned. Sliding *down* the valley
requires first stepping *out* (uphill), making valley-cells into
*spurious discrete local minima* — modes that don't exist in the
continuous landscape.

**Implication for Phase 3 multi-verifier rotation:** the rotator may
encounter modes that exist only in the discretized landscape. The
Round-9 architecture's defence against null-space mimicry doesn't
distinguish "spurious-distillation modes" from "real continuous modes."

**Mitigation:** the Hybrid Coprocessor's CPU-side rejection filter
naturally rejects spurious modes (the smooth EBM evaluates them as
non-minimal). So the proposal-generator/rejection-filter split also
rescues us from staircase aliasing — without separate machinery.

## Positive finding: Kramers gap preserved at $r=2$

At a continuous minimum $x^*$ where $\nabla E = 0$, the discretization
error is governed by the Hessian (quadratic):
$$E(c^*) - E(x^*) \approx \tfrac{1}{2}(c^* - x^*)^T \nabla^2 E (c^* - x^*) = \mathcal{O}(2^{-2k})$$

Both ground-state and second-mode energies shift by $\mathcal{O}(2^{-2k})$,
so the gap shifts by the same. **MCMC mixing-time characteristics
survive distillation at the faster rate.** This validates the Phase 2
sampling story even at the lower capacity rate.

## What Carnot got wrong

- $r_{\text{cap}} = 2$: refuted; actual $r_{\text{cap}} = 1$.
- Higher-smoothness improvement via Smolyak: refuted; piecewise-constant
  Ising can't exploit smoothness.
- $k_{\text{cap}}^{\min} = 8$ for typical case: refuted; actual $k=12$
  (4-bit increase in hardware sizing).
- Binding constraint: predicted KL; actual capacity.
- Mode preservation: predicted trivial at $2^{kd} \gg M$; refuted by
  staircase aliasing introducing spurious discrete modes.

## What Carnot got right

- Z3 verifiers in $\mathcal{F}_{\text{patho}}$ (confirmed exactly).
- Hybrid execution as the workaround (Deep Think names it Hybrid
  Coprocessor Pipeline).
- Convexity lost under distillation; Lipschitz preserved on Hamming
  graph (confirmed with sharper constants).

## Updates required

1. **`_bmad/architecture.md` Asymptotic Hardware Mandate:** update
   throughput-vs-depth table — Carnot's regime needs $k=12$, not $k=8$.
2. **Change proposal `zenil-grounded-self-distillation-deployable-stack.md`:**
   formalize the Hybrid Coprocessor Pipeline as Step -1 (preceding the
   Step 0 VAE bottleneck). Explicit smooth/Boolean split.
3. **REQ-PHASE2-006 spec** updated with capacity statement (above).
4. **NEW caveat memory entry** `project_staircase_aliasing.md` —
   spurious-mode warning for Phase 3 architecture.

## Position-paper impact

13th publishable contribution: the augmented Phase 2 transpiler
theorem with capacity rate. Plus the Hybrid Coprocessor Pipeline
as the canonical deployment pattern.

This effectively *completes* the mathematical foundation for Carnot's
Phase 1 → Phase 2 → Phase 3 arc.
