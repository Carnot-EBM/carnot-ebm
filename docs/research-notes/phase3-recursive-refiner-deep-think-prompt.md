# Deep Think Prompt — Recursive-Refiner Phase-3 Thesis (framing validation)

**Purpose:** pressure-test the candidate thesis in `phase3-recursive-refiner-thesis.md`
BEFORE any compute is spent, the same way prior methodology rounds were validated.
**Provenance:** operator-seeded from the TRM/GRAM/PTRM/LDT review; memory
`tiny-recursive-reasoning-models`. **After DT responds:** reconcile each P1–P5 answer into
`phase3-recursive-refiner-thesis.md` (kill-criterion + headroom gate), the same way the
post-bounded round was reconciled.

---

## Prompt (paste this whole block into Deep Think)

```
You are validating a candidate research thesis BEFORE any compute is spent. Hold a
high bar; separate "what is true" (your calibrated zone) from "what to build" (flag
as lower-confidence). For each question give a confidence % and the single cheapest
check that would FALSIFY your answer. End with a RUN / FIX-FIRST / DON'T-BOTHER verdict.

CONTEXT (a small EBM-verifier research program):
- We empirically BOUNDED two mechanisms on natural-language math/arithmetic: energy-as-
  SELECTOR (an energy model reranks candidate answers) and energy-as-GENERATOR (a tiny
  Energy-Based Transformer generates by Langevin energy-descent in latent space, decoded
  greedily or via a learned decoder). Both lost to / failed to beat autoregressive (AR)
  baselines at matched inference compute. Critically, we tested in a regime where self-
  consistency is NEAR-CEILING (standard MATH), i.e. there was little headroom for any
  method to win.
- Separately, TRM (Tiny Recursive Model, ~7M params) reportedly beats frontier LLMs on
  combinatorial GRID tasks (ARC-AGI, Sudoku, mazes) via recursive LATENT refinement
  (inner/outer loop), NOT energy descent. Claimed extensions (PTRM: Gaussian noise per
  recursion step + an internal Q-head as a learned halt/verifier; LDT: project latents
  onto an abstract lattice each step for sound deduction + explicit abstention) push this
  further. (Treat the specific extension papers as UNVERIFIED; TRM is the confirmed anchor.)

THE CANDIDATE THESIS:
"Our 'energy is bounded' result is REGIME-specific, not paradigm-fatal. The winning recipe
is a tiny recursive latent refiner + an internal verifier-halt + noise injection — adjacent
to but NOT identical to energy-descent. Bet: energy-descent was the wrong instantiation; a
TRM-style recursive refiner is the better generator substrate, and OUR external constraint/
energy verifier slots in as the internal Q-head / lattice-abstain component. We don't have
to win the generator race; we own the verifier these architectures need internally."

VALIDATE:

P1 — REGIME vs PARADIGM. Is the bound we measured genuinely regime-specific (we tested NL-
math where SC is near-ceiling, so no method could win), or is iterative-latent generation
bounded for a deeper reason that would ALSO bound a TRM-style refiner once the task is
expressed in a way that matters to us? Concretely: do the failure modes we saw in energy-
descent (off-manifold void; a global symmetric objective lacking AR's causal/compositional
inductive bias) ALSO apply to TRM-style recursive refinement, or does refine-in-place of a
discrete grid structurally escape them?

P2 — MECHANISM DISTINCTION. Is TRM recursive refinement meaningfully different from EBT
energy-descent, or are they the same iterative-latent-optimization in different clothes (so
our bounded result already covers it)? Name the specific architectural property, if any,
that makes one succeed where the other fails.

P3 — THE CARNOT-SPECIFIC BET (highest stakes). Can an EXTERNAL constraint/energy verifier
(SAT/Z3/AST/energy, non-differentiable, coarse-grained, ~ms latency) actually serve as the
internal Q-head / halt criterion of a recursive refiner — or does an END-TO-END-LEARNED
internal Q-head structurally dominate it (differentiable, co-trained, per-step-granular),
making "our verifier as the Q-head" a category error? Is there a real integration path
(e.g. verifier as a frozen reward/halt signal, or distilled into the Q-head), or not?

P4 — TRANSFER TO THE MISSION. TRM-class wins are on fixed-grid combinatorial puzzles with
clean verifiable answers. Our actual mission is verifying open-ended LLM reasoning/output
(and a long-horizon foundation-model goal). Is there a credible path from "tiny refiner wins
on grids" to anything that mission cares about, or is it a toy-domain demonstration that
does not transfer (the domain-bound pattern we keep hitting)?

P5 — VERDICT. What is the SINGLE cheapest experiment that would falsify this thesis fastest
(kill-criterion), what positive-control / headroom condition must hold for the test to be
valid (we have repeatedly false-negatived by testing where the baseline was already near-
optimal), and is this worth running at all — RUN / FIX-FIRST / DON'T-BOTHER?
```
