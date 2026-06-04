# Tiny Recursive Models — Primer + Link Collection

*A self-contained explainer of the tiny-recursive-reasoning-model family and the
papers/repos Carnot is patterning its Phase-3 thesis after. Written 2026-06-04.
Companion to `phase3-recursive-refiner-thesis.md` (the strategic mapping) and the
memory note `reference_tiny_recursive_reasoning_models.md`.*

---

## What a "tiny recursive model" is

The core idea is a reaction against two assumptions baked into frontier LLMs:

1. **That reasoning must be autoregressive** — emitted one token at a time, left to
   right, with the chain-of-thought *as text*.
2. **That capability must scale with parameters** — bigger model = better reasoning.

Tiny recursive models reject both. Instead:

- **Reasoning happens in a latent state, not in tokens.** The model holds a working
  "scratchpad" as a continuous hidden vector `h` (and often a current-answer guess
  `y`). It never writes its intermediate thinking out as text.
- **Depth comes from *recursion in time*, not layers in space.** A *tiny* network
  (TRM is literally 2 layers, ~7M params) is applied to its own state over and over —
  `h <- f(h, x, y)` — for many steps. Test-time compute is scaled *out* (more
  refinement iterations) rather than *up* (more parameters). This is the single most
  important finding: the ARC Prize analysis of HRM showed the **refinement loop**, not
  the fancy architecture, is what does the work — 1 loop got 18.6%, 16 loops got 39.0%
  on the same net.
- **It's self-correcting.** Each pass refines the previous answer toward consistency
  with the constraints, like a solver tightening a guess — closer to iterative repair
  than to one-shot generation.
- **The payoff:** a 7M-param TRM beats Gemini 2.5 Pro / o3-mini / DeepSeek-R1 on
  ARC-AGI, Sudoku-Extreme, and mazes — with **<0.01% of their parameters** — *in the
  regime those puzzles live in* (combinatorial grids with a clean verifiable answer
  and real headroom).

### Why Carnot cares

This paradigm — recursive latent refinement, tiny net, non-autoregressive,
self-correcting — *is* Carnot's Phase-3 north star, built by someone else. Two things
follow:

- **(a)** Carnot's "energy-as-generator is bounded" result does **not** condemn it: TRM
  learns an *arbitrary* vector field, whereas Carnot's EBT was restricted to *curl-free*
  energy descent — different mathematical objects (Deep-Think P2, 2026-06-04).
- **(b)** The verifier is the through-line: these models need an internal "am I done / am
  I right?" signal, which is exactly what Carnot's verifier ensemble is — either
  distilled into the model's halt-head offline, or, in LDT's case, as the *lattice* the
  state is projected through. That last connection is what the `.352` LDT-gap probe
  (exp3833) tests.

---

## The core family Carnot patterns after

| Model | What it adds | Paper | Code |
|---|---|---|---|
| **HRM** (Hierarchical Reasoning Model) | The parent: 27M two-timescale recurrent net (slow planner + fast worker), no CoT, 1000 samples, ~41% ARC-AGI-1. *Caveat: the ARC Prize analysis debunked the "hierarchy" — recursion is the real engine.* | https://arxiv.org/abs/2506.21734 | https://github.com/sapientinc/HRM |
| **TRM** (Tiny Recursive Model) | Strips HRM down: one 2-layer net, ~7M params, no hierarchy, no fixed-point theorem — and does *better* (45% ARC-AGI-1). The cleanest statement of the paradigm. Alexia Jolicoeur-Martineau (Samsung SAIL Montreal). | https://arxiv.org/abs/2510.04871 | https://github.com/SamsungSAILMontreal/TinyRecursiveModels |
| **LDT** (Lattice Deduction Transformer) | **Most relevant to Carnot.** Projects the latent through an *abstract-interpretation lattice* each step → logically *sound* deduction → "correct-or-abstain." 800K params, 100% Sudoku-Extreme. This is Carnot's transpilation+abstention thesis, built and SOTA — but for closed grid domains. Davis, Haller, Alfarano, Santolucito. | https://arxiv.org/abs/2605.08605 | https://github.com/dmelmanrogers/Lattice-Deduction-Transformer |
| **PTRM** (Probabilistic TRM) | Adds Gaussian noise per recursion step + selects/halts via the model's *existing Q-head* — the exact internal verifier Carnot's distillation pivot targets. | https://arxiv.org/abs/2605.19943 | — |
| **GRAM** | Stochastic multi-trajectory recurrence = parallel-hypothesis latent branching. | https://arxiv.org/abs/2605.19376 | — |

*(The 2605.xxxxx three — LDT/PTRM/GRAM — are the May-2026 cohort surfaced in a
technical review; LDT is confirmed directly against arXiv, PTRM/GRAM IDs are from that
review.)*

## The wider lineage worth reading

| Work | Why it matters | Paper |
|---|---|---|
| **Huginn / Recurrent-Depth** | The key de-risker: recursion-in-latent reasoning **at LLM scale, on TEXT** (GSM8K 42% vs 2.2% non-recurrent). Proves the paradigm isn't grid-bound. Uses *unsupervised KL-convergence halting* (no learned Q-head) — a gap a verifier-guided halt could beat. | https://arxiv.org/abs/2502.05171 |
| **Coconut** (Chain of Continuous Thought) | Reasoning in continuous latent space instead of tokens — the Meta precedent for non-verbal thought. | https://arxiv.org/abs/2412.06769 |
| **Encode-Think-Decode** | Recursive latent "thoughts" between encode/decode. | https://arxiv.org/abs/2510.07358 |

## The foundational halting / recurrence ancestors

| Work | Idea | Paper |
|---|---|---|
| **ACT** (Adaptive Computation Time) | Graves 2016 — learn *how many* compute steps per input; the origin of "ponder then halt." | https://arxiv.org/abs/1603.08983 |
| **Universal Transformer** | Dehghani et al. 2018 — recurrent-in-depth transformer with ACT halting. | https://arxiv.org/abs/1807.03819 |
| **Deep Equilibrium Models** (DEQ) | Bai/Kolter/Koltun 2019 — infinite-depth-as-fixed-point; the theory HRM leaned on and TRM discarded. | https://arxiv.org/abs/1909.01377 |
| **PonderNet** | Banino et al. 2021 — principled probabilistic halting. | https://arxiv.org/abs/2107.05407 |
| **Looped Transformers** | 2023 — a transformer block looped as a programmable iterative computer. | https://arxiv.org/abs/2301.13196 |

## Supporting reference

- ARC Prize analysis of HRM (recursion-not-hierarchy debunk): https://arcprize.org/blog/hrm-analysis

---

## One discipline note

Every one of these wins in the *narrow combinatorial-grid regime* (Sudoku/ARC/maze).
HRM's own held-out drop and task-specificity confirm grid wins may not transfer to
open-ended reasoning — so hold that caveat hard before any of this becomes a
forward-facing Carnot claim. The strategic reframe (the three verifier-integration
options, and why Carnot's differentiated gap is a *general multi-domain verifier
ensemble as the abstraction lattice for open-ended domains*) is in
`phase3-recursive-refiner-thesis.md`.
