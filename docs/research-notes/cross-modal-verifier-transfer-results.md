# Cross-Modal Verifier Transfer Results — LGDD Scope Theorem

**Status:** Authoritative. The capstone of the position-paper scope question.
**Date:** 2026-04-29
**Prompt:** `cross-modal-verifier-transfer-prompt.md`

Deep Think delivered the **Logically Grounded Discrete Domains (LGDD)
scope theorem** with three precise axioms. The Round-9 architecture is
provably optimal *strictly* on LGDD; outside, it either collapses
(NL reasoning) or is bypassed by a different positive regime
(continuous vision/audio).

## The LGDD scope theorem

The Round-9 multi-verifier rotation architecture is provably optimal
on the class of domains satisfying:

- **C1 (Discrete Boolean Topology):** Output space $\mathcal{X}$ is
  combinatorially discrete, enforcing $E_i \in \{0, 1\}$. This
  triggers the Boolean Information Bottleneck and makes AND-composition
  necessary as the entropy maximizer.
- **C2 (Epistemic Orthogonality):** The verifier suite $\mathcal{E}$
  permits mechanically decoupled evaluation paradigms (static formal
  verification vs dynamic stochastic execution). Guarantees linear
  independence of error residuals in $L^2(\mu_P)$, enforcing
  $\theta_F \geq \pi/4$.
- **C3 (Absolute Oracular Grounding):** At least one verifier is a
  deterministic kernel with $\text{FPR} = 0$ (Z3 SMT, Lean kernel).
  Anchors $\mu_P$ rigidly and prevents pathological reward-hacking
  regions.

## In-scope domains

- **Code repair** (Carnot's reference domain): AST/program tokens
  (C1), Z3 + property fuzzers (C2), Z3 SMT (C3).
- **Math proofs (Lean / Coq):** Lift is flawless. Proof terms (C1),
  kernel type-check + tactic fuzz (C2), Lean kernel (C3).
- **Other formally-grounded discrete-output domains** satisfying
  C1+C2+C3.

## Out-of-scope domains

### Pure NL reasoning — catastrophic collapse

LLM-judges, NLI step-checkers, fact-checkers all share pre-training
distributions. Their error residuals are *statistically correlated*:
$\langle g_{\text{LLM}}, g_{\text{NLI}}\rangle_{\mu_P} \approx \|g\|^2$.

Friedrichs angle $\theta_F \to 0$ catastrophically. AND-composition
adds redundant noise rather than rotating $\Phi$ into orthogonal
dimensions. The architecture *collapses into the Round-7 single-verifier
regime*, trapping the model at the lower-bounded plateau.

C2 (Epistemic Orthogonality) is fundamentally violated — there's no
non-statistical alternative paradigm to anchor against. NL reasoning
needs a *fundamentally different* theoretical foundation.

### Continuous vision/audio — different regime, NOT collapse

Continuous verifiers output $E \in \mathbb{R}$, yielding dense gradient
$\nabla_x E \in \mathbb{R}^d$. A single evaluation injects $O(d)$ bits
of restorative force, not 1.

The Boolean Information Bottleneck *bypasses in a positive direction*:
- AND-composition becomes obsolete (no entropy bottleneck to break).
- **Additive composition** $E_{\text{comp}}(x) = \sum_i E_i(x)$ is
  the natural continuous analogue, corresponding to multiplicative
  kernel composition $\prod_i \exp(-E_i(x)/T)$.

This is a separate mathematical regime — a Phase 4 follow-up
paper, not a refutation.

## Q1 — Functorial lift theorem

For a domain transport $T: \mathcal{X}_A \to \mathcal{X}_B$ with
pullback $T^* E_i^B = E_i^B \circ T = E_i^A$, the lift preserves
Round-9 optimality iff:

1. **Fiber-Constancy:** every source verifier $E_i^A$ is constant
   on fibers $T^{-1}(y)$.
2. **Pushforward Measure Isometry:** $T_\# \mu_{P,A} = \mu_{P,B}$.

Under both: Friedrichs angle, AND-composition saturation, and
Boolean Information Bottleneck all survive.

**Computational complexity:** undecidable by Rice's Theorem in
Turing-complete settings; NP-hard for finite state spaces.

## Q1(b) — Carnot-relevant refutation

**Code AST → Compiled Execution Trace** via optimising compiler:

- Compiler collapses functionally-equivalent ASTs (different syntax,
  dead code) into identical traces.
- $E_{\text{Z3}}^{\text{AST}}$ evaluates syntactic properties the
  compiler erases — *not fiber-constant*, cannot lift.
- Forced restriction to trace-invariant verifiers (fuzzers) makes all
  surviving verifiers structurally collinear in $L^2(\mu_{P,B})$.
- $\theta_F \to 0$, defence annihilated.

**Implication for Carnot:** static (AST) and dynamic (execution)
verification *do not lift to each other*. Each must be a
self-orthogonal verifier suite at its own evaluation level. Carnot
can't reuse Z3's $\theta_F$ contribution at runtime. The
proposal's verifier suite must explicitly span the chosen
evaluation domain — not assume cross-level transfer.

## Position-paper title revision

**Old:** "Sovereign Foundation Models Require Multi-Verifier
Pre-emptive Rotation: A Mathematical Foundation for Phase 3
Self-Distillation."

**New:** "Multi-Verifier Pre-emptive Rotation for **Logically
Grounded Discrete Domains**: A Mathematical Foundation for Phase 3
Self-Distillation."

The LGDD class is itself a citable contribution. Sharper, more
defensible, and avoids the credibility risk of overclaiming
generality across NL reasoning or continuous-output domains.

## What Carnot got right

- Q2(a) math proofs lift cleanly via curry-howard
- Q2(c) vision needs different framework (got the direction right)
- Q3 domain-specific theorem with class boundary (correct framing)

## What Carnot got wrong

- Q1(a) lift conditions: predicted vague "compatible base classes";
  actual is precise two-axiom theorem (Fiber-Constancy + Pushforward
  Isometry).
- Q1(b) refutation: predicted code→vision; the more Carnot-relevant
  refutation is code-AST→execution-trace.
- Q1(c) complexity: predicted NP-hard general / polynomial categorical;
  actual is undecidable in Turing-complete cases.
- Q2(b) NL reasoning: predicted partial lift with $k=25$–30 verifiers;
  actual is catastrophic collapse — no number of verifiers helps when
  C2 is violated.
- Q2(c) vision: predicted "lift fails"; actual is "BIB bypasses
  positively" — vision is a different regime, not a paper-killer.

## What surprised us

- **AST→Trace refutation.** This is a domain pair Carnot actually
  uses. Static + dynamic verification must be SEPARATE suites at
  separate evaluation levels.
- **NL reasoning catastrophic collapse.** Pre-training overlap
  between LLM-judges destroys $\theta_F$. No clever architecture
  saves this — needs entirely different theory.
- **Continuous-domain BIB-bypass as positive result.** Vision/audio
  isn't out-of-scope; it's a parallel regime where additive
  composition replaces AND-composition. Future-paper material, not
  a paper-killer.
- **Three-axiom LGDD scope.** Cleaner, more precise scope class than
  any of my predicted formulations.

## 14th publishable contribution

**The LGDD Scope Theorem:** the precise three-axiom
characterisation (C1+C2+C3) of domains where Round-9 is provably
optimal, with explicit refutations for AST→Trace, NL reasoning,
and a separate-regime result for continuous outputs.

This is the scope statement the position paper needs. It converts a
vague generality claim ("foundation models") into a precise theorem
("LGDD-class foundation models").

## Implications for change proposals

1. **`zenil-grounded-self-distillation-deployable-stack.md`:** add
   the LGDD scope theorem as REQ-PHASE3-005. Verifier-suite
   designers must verify their suite satisfies C1+C2+C3 before
   claiming Round-9 optimality.

2. **NEW change proposal candidate:** `phase4-continuous-verifier-rotation.md`
   for the future — additive-composition continuous-verifier framework
   for vision/audio. Defer to Phase 4.

3. **Position paper outline update:** scope statement, LGDD axioms,
   refutations, and the AST→Trace concrete failure case.

## Stopping recommendation

This is the natural end of the math chain. **15 publishable
contributions** across:

- Zenil chain (10) — verifier-rotation architecture
- Kinematic Layer Routing (2) — Tier-0i analytic placement
- Continuous Ising-Rank (1) — Phase 2 transpiler theorem
- LGDD Scope Theorem (1) + AST→Trace refutation (1) — scope class

The mathematical foundation for Carnot's Phase 1 → Phase 2 → Phase 3
arc is **complete and bounded**. Further questions are polish or
genuinely speculative (Phase 4 continuous regime, transformer-EBT
vs recurrent-EBT empirics, etc.).

Implementation now blocks on code, not theory.
