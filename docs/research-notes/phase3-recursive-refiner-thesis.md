# Phase-3 Candidate Thesis — Recursive Latent Refiner + Carnot-Verifier-as-Q-Head

**Status:** STAGED CANDIDATE (2026-06-04). Operator-seeded from the TRM/GRAM/PTRM/LDT
technical review. NOT activated. Awaiting operator/Deep-Think review before any milestone
wiring — same gate the EBT (Thesis-A) seed went through.

**Provenance:** operator surfaced a comparative review of small recursive-latent-refinement
reasoners (TRM and three claimed extensions GRAM/PTRM/LDT). Analysis captured in memory
`tiny-recursive-reasoning-models`. This note stages the experiment that analysis implies.

---

## The reframe that motivates this

We closed **energy-as-selector (P0.1)** and **energy-as-generator (Thesis-A / EBT)** as
**BOUNDED** — but both were tested in the **NL-math / arithmetic regime, where self-consistency
is near-ceiling** (no headroom; "can't beat an integral with a mode"). TRM-class models
demonstrate the recursive-latent-refinement paradigm **wins** — but in a *different* regime:
**combinatorial grids with a clean verifiable answer and real headroom** (Sudoku, ARC-AGI,
mazes), where giant autoregressive LLMs are known to be weak.

So our bounded conclusion may be **regime-specific, not paradigm-fatal**. And the *winning
recipe* — tiny recursive latent refiner + internal verifier-halt + noise injection — is
**adjacent to but not identical to** EBT energy-descent. The candidate bet: **EBT energy-
descent was the wrong specific instantiation; a TRM-style recursive refiner is the better
generator substrate, and Carnot's verifier slots in as the internal Q-head / lattice-abstain
component.** Carnot does not have to win the generator race — it owns the verifier these
architectures need internally (banked-product-as-a-component).

## Hypothesis (falsifiable)

A tiny (~7-50M param) recursive latent refiner, with **Carnot's verifier as the internal
halt/Q-head**, beats a matched-compute autoregressive baseline on a **headroom-bearing
combinatorial-grid task** (where AR is demonstrably weak and an oracle/optimal materially
exceeds the AR baseline) — at equal inference compute.

## Why this is NOT a doomed rerun of Thesis-A

| Axis | Thesis-A (EBT, BOUNDED) | This candidate |
|---|---|---|
| Task regime | NL-math / arithmetic (SC near-ceiling, **no headroom**) | combinatorial grid (ARC/Sudoku/maze) with **confirmed oracle > AR headroom** |
| Generator mechanism | global energy-descent (Langevin over E(y₁…y_T)) | TRM-style **recursive latent refinement** (inner/outer loop), noise-injected |
| Verifier role | none internal (energy was both gen + score) | **Carnot verifier as the internal Q-head / halt criterion** (the load-bearing novelty) |
| Decode | greedy energy-argmin / learned decoder (flatlined ~uniform) | iterative refine-in-place of a discrete grid (no off-manifold void) |

The decisive difference is the **regime has measurable headroom** (the P0.1/FALSE_NEGATIVE_RISK
lesson: never test a method where the baseline is already near-optimal) **and** the verifier is
an *internal* component, not a separate scorer.

## Mandatory preconditions (per CLAUDE.md Pre-Launch + Adversarial-Verify discipline)

1. **Positive-control / headroom gate FIRST.** Pick a grid corpus where an oracle (or optimal
   solver) materially exceeds a matched-compute AR baseline (oracle − AR ≥ some margin, AR not
   already near-ceiling). If no headroom, STOP — that's the P0.1 trap.
2. **Matched-COMPUTE, not matched-params** (the P0.1 trap): the refiner runs K refinement
   steps; give AR an equal-FLOP best-of-N / self-consistency budget. A win counts only at equal
   total inference compute.
3. **Verify the claimed papers exist** (GRAM/PTRM/LDT) with real benchmarks before adopting any
   specific number; build on TRM (confirmed real) as the anchor.
4. Tiny-model stability kill-gate (the Thesis-A part-a pattern): can the refiner train stably
   on the rig? If it diverges/NaNs within a bounded budget → bounded at small scale, STOP.

## Kill-gate / acceptance

- **PASS:** refiner-with-Carnot-verifier-Q-head > matched-compute AR by a statistically
  significant margin on a confirmed-headroom grid task, 3+ seeds.
- **BOUNDED (honest negative):** ≤ AR at matched compute even with headroom → recursive-latent-
  refinement + Carnot-verifier does not beat AR at this scale; record and stop (do NOT re-grind).
- **INCONCLUSIVE:** positive control (AR) fails to learn or oracle ≤ AR (no headroom) →
  regime invalid, fix the corpus before claiming anything (the v1/v2 P1 lesson).

## What stays banked regardless

paper_ready TRUE, FoVer 0.9131 frozen, energy-selection bounded, the verifier product. This is
a venture bet with a kill-gate, parallel to (not replacing) the banked deliverable. If it
PASSES, the strategic payoff is "Carnot's verifier is the component inside the winning
small-reasoner class" — the strongest version of the moat per `deep-think-post-bounded-2026-06`.

## Caveats (do not over-claim)

- Narrow fixed-grid combinatorial wins do NOT imply general reasoning (the
  `verifier-domain-bound-math-only` + `mai-thinking-1-hill-climbing` lesson).
- TRM ≠ EBM: this abandons the energy-descent mechanism in favor of recursive refinement; it is
  a NEW substrate, not a continuation of the bounded EBT.
- "Crushes frontier LLMs / fraction of a fraction of cost" is over-claim phrasing
  (`carnot-prediction-error-pattern`); cross-check before citing.

**Next step:** operator/Deep-Think review of this framing → if green-lit, stage as a pre-staged
roadmap milestone with the headroom-gate as task 1 (the EBT seed precedent).
