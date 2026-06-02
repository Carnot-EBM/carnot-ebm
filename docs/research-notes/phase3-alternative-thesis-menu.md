# Phase-3 Alternative-Thesis Menu — routes to an EBM foundation model that do NOT rely on the disproven energy-selection bound

**Status:** OPERATOR DECISION doc, drafted 2026-06-02 by the outer-loop on operator
directive ("is there anything I can do to help address the phase 3 and later
theories for alternate approaches?").

## The frame (read first)

What is disproven (see memory `energy-selection-thesis-bounded`): **energy as a
SELECTOR** — reranking autoregressive / self-consistency outputs — is bounded
everywhere tested (SC is at/above the oracle ceiling on math/CSP, and there is no
selection value even where SC is weak). That route is closed.

What is NOT tested: energy/EBM as the **generator** or **training mechanism**, and
genuinely-different non-autoregressive paradigms. Every thesis below is chosen to
sidestep the selection bound by attacking a *different* mechanism. The autonomous
loop will not invent or pursue these on its own — each needs an operator decision
to seed it, and each is a real *training-class* bet (a different cost/risk class
than the cheap cached-scoring verifier work).

**Discipline carried from the P0.1 grind:** every thesis has a SMALL, FALSIFIABLE
decisive test with an explicit KILL-GATE, so a dead route dies in days, not over a
dozen milestones. Pick ONE to seed (or use this as input to a Deep Think round).

---

## Thesis A — Energy as the GENERATOR (EBT-as-base)  ·  *SELECTED 2026-06-02; most direct*

- **Core claim:** A small Energy-Based Transformer whose inference *is* iterative
  energy descent over the output (not left-to-right AR tokens, not reranking) can
  match or beat autoregressive generation **at equal inference COMPUTE** on one
  structured reasoning task.
- **Primary anchor (operator-pointed 2026-06-02):** Gladstone, …, **Yilun Du, Heng
  Ji** et al., *"Energy-Based Transformers are Scalable Learners and Thinkers"*
  (arXiv:2507.02092). Public PyTorch code: **github.com/alexiglad/EBT**. Their
  published claims ARE this thesis: inference = gradient-descent energy
  minimization to convergence; up to **35% higher scaling rate** than Transformer++
  across data/batch/params/FLOPs/depth; **+29% System-2 "thinking"** on language;
  trained **unsupervised, no verifier** (so it does NOT inherit the
  verifier-domain-bound ceiling). Secondary anchor: 2512.15605 (ARM-EBM bijection).
- **Why it sidesteps the bound:** P0.1 disproved energy *selecting among AR
  samples*. This tests energy as the *generative process itself* — the selection
  bound (SC near-optimal) cannot apply because nothing is being selected. EBT also
  needs no verifier, so the math-only verifier ceiling does not transfer.
- **Smallest decisive test:** fork `alexiglad/EBT`, shrink to a tiny EBT (~10–50M
  params), and train it + a matched tiny AR transformer on ONE task (GSM8K-style
  arithmetic or a synthetic multi-step logic corpus). Measure held-out reasoning
  accuracy of energy-descent generation vs AR generation.
- **CRITICAL — matched-COMPUTE, not matched-params (the P0.1 lesson):** energy
  descent runs N forward passes per prediction, so a "matched-params" win can be
  pure extra inference FLOPs — the same trap P0.1 fell into (more compute ≠ better
  mechanism). The comparison MUST give the AR baseline an equal total-inference-FLOP
  budget (best-of-N / longer sampling). A win counts ONLY if EBT beats AR at equal
  total inference compute.
- **KILL-GATE:** if (a) the tiny EBT cannot be trained to *stably converge* within a
  bounded budget (≤ a few GPU-days on the 3090 rig), OR (b) at **equal inference
  compute** it does not beat the matched AR baseline AND the gap does not narrow
  with 2× training → the energy-as-generator route is dead at small scale. STOP; do
  not scale.
- **Honest caveat:** EBT's *published* wins are on LM perplexity/scaling + image
  denoising, NOT structured-reasoning accuracy; the System-2 headline is the
  contested part precisely because of the compute-accounting issue above. Our test
  is therefore a genuine new measurement (reasoning accuracy at matched compute),
  not a reproduction.
- **Cost:** training-HEAVY (real training, but fork-not-reimplement). **Risk:** EBT
  training is known finicky/unstable — itself the most likely failure mode, and the
  test surfaces it cheaply at tiny scale.

## Thesis B — Non-autoregressive global inference (discrete diffusion)  ·  *medium*

- **Core claim:** Discrete-diffusion / iterative-refinement generation beats AR on
  tasks where AR's left-to-right commitment hurts (long-horizon, backtracking-
  needed, global-consistency). Energy-adjacent: the denoiser is a global energy-
  like model. Precedent the loop itself cited: "Beyond Autoregression" (global
  inference beats AR on Sudoku 100% vs 20.7%, Countdown 91.5% vs 45.8%).
- **Why it sidesteps the bound:** not selection — a different *generation* paradigm
  (parallel/iterative vs left-to-right). The P0.1 bound was selection on math where
  SC is near-optimal; this tests GENERATION on AR-*hostile* tasks where the
  structural advantage should exist.
- **Smallest decisive test:** on a task class where AR is KNOWN to struggle
  (constraint satisfaction needing backtracking, or global-consistency puzzles),
  train a small discrete-diffusion LM vs a matched AR LM. Does diffusion beat AR on
  the AR-hostile subset?
- **KILL-GATE:** if diffusion does NOT beat AR even on the AR-hostile subset (where
  it has the structural advantage), the route has no Carnot-specific value → dead.
  STOP.
- **Cost:** training-MEDIUM (existing discrete-diffusion LM codebases to build on).
  **Risk:** may just replicate known diffusion-LM results without an EBM-specific
  hook — the test must tie the win to the energy/global-inference thesis, not
  generic diffusion.

## Thesis C — Energy as a TRAINING-TIME objective (energy-regularized AR)  ·  *cheapest*

- **Core claim:** The AR↔EBM bijection (2512.15605) lets you add an energy-
  consistency objective to a standard AR model at *training time* — improving its
  reasoning self-correction WITHOUT changing the architecture. Energy as a
  regularizer, not an inference-time selector.
- **Why it sidesteps the bound:** not inference-time selection; a training-time
  objective that shapes the AR distribution to be more self-consistent. Tests
  whether energy helps where it was never tried — in TRAINING.
- **Smallest decisive test:** fine-tune one small AR model with vs without the
  energy-consistency objective on a reasoning corpus. Measure held-out accuracy +
  a self-contradiction rate.
- **KILL-GATE:** if the energy-regularized model shows no held-out improvement and
  no self-contradiction reduction over standard fine-tuning → dead. STOP.
- **Cost:** training-LIGHT (fine-tuning, not from-scratch) — the cheapest decisive
  test here. **Risk:** the effect may be small; needs a careful matched baseline to
  avoid a false positive.

## Thesis D — Hardware-first (Extropic Z1 / thermodynamic substrate)  ·  *operator-gated*

- **Core claim:** The EBM foundation model's value is *efficiency/sovereignty on
  dedicated hardware* — energy sampling on a thermodynamic substrate (Z1/TSU) is
  orders-of-magnitude cheaper than GPU, making EBM generation viable where it isn't
  on a GPU. The architecture follows the hardware.
- **Why it sidesteps the bound:** the bound was about energy-selection *accuracy on
  GPU-tested tasks*; this bet is on a different axis entirely — efficiency on a
  substrate that doesn't exist on GPU.
- **Smallest decisive test:** **blocked on Z1 early access (operator action).** The
  project notes "Z1 early access 2026." With access: a minimal energy-sampling
  benchmark showing the efficiency claim vs the 3090 baseline.
- **KILL-GATE:** if Z1 access is unavailable, OR the efficiency advantage does not
  materialize on a real benchmark → dead/deferred.
- **Cost:** BLOCKED on vendor hardware. The loop can do NOTHING here without you on
  the Z1 list. **Risk:** entirely gated on vendor availability + the efficiency
  claim being real.

---

## Summary + recommendation

| Thesis | Mechanism tested | Decisive-test cost | Loop can run it? |
|---|---|---|---|
| **A — EBT-as-generator** | energy = generator | training-HEAVY | yes (3090 rig) |
| **B — discrete diffusion** | non-AR generation | training-MEDIUM | yes |
| **C — energy training objective** | energy = regularizer | training-LIGHT | yes (cheapest) |
| **D — Z1 thermodynamic hardware** | efficiency axis | blocked on access | NO — needs you |

**Recommended sequencing:**

1. **Start with C** (energy-regularized AR) — it is the cheapest falsifiable test
   (fine-tuning, not from-scratch), directly probes "does energy help at all when
   it's not a selector," and a clean negative is informative + fast.
2. **If C shows any signal, escalate to A** (EBT-as-generator) — the most direct
   test of the actual Phase-3 thesis. Heavier, but it is *the* question.
3. **B** is a parallel option if you believe the win is on AR-hostile task
   structure rather than the EBM architecture per se.
4. **D** is *your* move to make in parallel regardless: get on the **Z1 early-
   access list**. It is the one route the loop physically cannot start without you,
   and it's the project's actual long-term sovereignty bet.

**Honest meta-note:** all of A–C are real training experiments with a genuine
chance of "this route is also bounded." That's fine — the kill-gates make each a
*bounded* bet, not a P0.1-style grind. And none of this should slow banking the
verifier product, which is the real, shipped win. Phase 3 is a venture bet; treat
it as one — one seeded thesis at a time, each with a knife to its own throat.

**Next step for the operator:** pick one of A/B/C to seed (or feed this into a Deep
Think round for a route not listed here), and/or pursue D (Z1 access). On your
choice, the outer-loop will turn it into a falsifiable `.NNN` milestone with the
kill-gate wired in, so the loop tests it without being able to grind it.
