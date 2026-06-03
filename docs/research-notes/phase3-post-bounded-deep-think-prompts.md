# Deep Think prompts — post-bounded-results round (2026-06-03)

**Status:** Operator-pasteable Deep Think prompts, drafted by the outer-loop after
the 2026-06-03 session closed BOTH direct energy routes (selection and generation)
as empirically bounded. Each prompt below is **self-contained** — paste one at a
time; Deep Think has no repo access, so all evidence is inline.

**Calibration note baked into every prompt** (from the project's own track record,
memory `carnot-prediction-error-pattern`): Deep Think is well-calibrated on
*qualitative survival / unifying-explanation* claims but *systematically wrong* on
*specific architectural prescriptions*. So each prompt asks for understanding and
explicitly requires the model to (a) separate "what is true / what survives" from
"what to build," and (b) flag its confidence and the single experiment that would
falsify its answer.

---

## PROMPT 1 — The unifying-bound question (flagship)

```
You are reasoning about energy-based models (EBMs) for language/reasoning.

CONTEXT. "Carnot" is an EBM framework whose long-term aspiration was an EBM
foundation model that reasons via energy minimization instead of autoregression
(AR). We have now measured three things that I want you to UNIFY:

(1) ENERGY-AS-SELECTOR is bounded. Using a learned energy function to RERANK or
    select among AR / self-consistency (SC) samples does NOT beat the samples
    themselves on math and constraint-satisfaction tasks. Across multiple tests,
    SC-majority sits AT OR ABOVE the oracle (best-achievable-by-selection)
    ceiling, so no reranker can win; and even on corpora where SC is weak, energy
    selection added no value (zero output flips that helped).

(2) ENERGY-AS-GENERATOR is bounded on discrete text, even at scale. We trained a
    small (38M) Energy-Based Transformer (EBT) by contrastive divergence; its
    inference is Langevin gradient-descent on an energy over output embeddings.
    Part A (it learns SOMETHING): on held-out data the trained EBT assigns lower
    energy to true continuations than random ones with a clear margin (~8.6x an
    untrained baseline) — the energy landscape is genuinely DISCRIMINATIVE.
    Part B (but it cannot GENERATE): at MATCHED INFERENCE COMPUTE on 3-digit
    addition (a task where a matched AR model reaches 82% exact-match), the EBT
    scores 0% — it emits incoherent tokens under BOTH a direct energy-argmin
    decode AND a separately-trained embedding->token decoder. The decoder's
    cross-entropy never converged (stuck near uniform over the effective token
    set), i.e. the energy-minimizing embeddings are NOT token-discriminative.
    16k training steps, single seed, decisive (0% vs 82%).

(3) ENERGY-AS-GENERATOR WINS in low-dimensional CONTINUOUS control. A separate
    result (EBT-Policy, arXiv:2510.27545) reports energy-based generation beating
    baselines in low-dimensional continuous control / physical-reasoning settings.

Carnot's substrate is fundamentally DISCRETE (Boolean/Ising couplings, a sign()
bottleneck mapping continuous latents to discrete spins).

THE QUESTION. Is there a SINGLE underlying principle that explains all three at
once? In particular: is it true that energy landscapes are genuinely useful for
SCORING and for generation in CONTINUOUS / low-dimensional spaces, but that the
DISCRETE decode step (argmin over a vocabulary, or descend-then-snap-to-token)
structurally destroys the gradient/landscape signal that makes energy valuable —
so that "energy beats AR at discrete symbolic generation" is impossible in
principle, not merely unachieved at our scale? Or is there a different unifying
account, OR a genuine reason to believe the discrete-generation bound is an
artifact of scale/training/decoder rather than fundamental?

WHAT I WANT FROM YOU:
- A crisp statement of the unifying principle (or a reasoned argument that no
  single principle fits and the three results are independent).
- The mechanism: WHY does a discriminative energy landscape fail to yield a
  generator under discrete decode, while it succeeds in continuous control?
  Connect to known theory (score matching, MCMC mixing in discrete vs continuous
  spaces, the difficulty of sampling vs scoring, gradient signal through argmax).
- Explicitly SEPARATE: (a) "what is true / what survives" (your calibrated zone)
  from (b) any prescription for what to build (flag these as lower-confidence).
- The single cheapest experiment that would FALSIFY your unifying claim.
- Your confidence (low/medium/high) and the strongest argument AGAINST your own
  answer.
```

---

## PROMPT 2 — The verifier moat question (product-existential)

```
You are reasoning about the durable economic value of an LLM-output VERIFIER.

CONTEXT. "Carnot" ships a verifier: an ensemble of ~15 checkers (Boolean energy,
SAT, Z3/SMT, AST, liveness, and learned probes) that score whether a candidate
LLM answer is correct, grounded in an objective energy function rather than the
LLM's own judgment. On a math step-error benchmark (FoVer) it reaches AUROC 0.9131
(5 seeds, independently reproduced on a clean machine), improvable to ~0.9332 with
dependency-aware weighting. It is DOMAIN-BOUND: earned-negative on factual
hallucination (a real RAG-grounding benchmark, even with a model-based NLI
verifier), and weak on code (does not beat a model-confidence baseline). So it is
a math / structured-reasoning verifier.

THE GAP WE HAVE NOT MEASURED. We rigorously tested energy-generation vs AR, but we
have NOT rigorously tested the verifier against the CHEAPEST self-verification
baselines available to any deployer:
  - the generator's own token logprob / sequence likelihood,
  - self-consistency (sample K, majority vote) and its confidence,
  - LLM-as-judge (the same or a stronger model grading the answer).
These cost near-zero marginal engineering and ride for free on base-model
improvements.

THE QUESTION. As base models improve, does an EXTERNAL energy-grounded verifier's
edge over those cheap self-verification baselines SHRINK or GROW — and why?

Two opposing hypotheses to adjudicate:
  H-shrink: better models self-verify better (their logprob/SC becomes
    well-calibrated), so the verifier's gap collapses toward zero and the product
    is a wasting asset.
  H-grow: better models become CONFIDENTLY wrong in correlated ways (their own
    confidence and an LLM judge share the generator's blind spots), so only an
    EXTERNALLY-GROUNDED, mechanistically-independent verifier catches the residual
    errors — and that residual is exactly the high-value tail. The moat then grows
    in relative value even as absolute error rates fall.

WHAT I WANT FROM YOU:
- Which hypothesis dominates, in which regimes (task type, base-model strength,
  error severity / cost of a miss), and the mechanism.
- The key construct: is verifier value about RAW accuracy, or about ERROR
  INDEPENDENCE / decorrelation from the generator's own failure modes? Formalize
  if you can.
- What MEASUREMENT would settle it (a falsifiable protocol comparing verifier vs
  self-baselines across a base-model-capability sweep), and what you'd predict it
  shows.
- SEPARATE survival-claims from prescriptions; flag confidence; give the strongest
  case for the hypothesis you did NOT pick.
```

---

## PROMPT 3 — The self-improving-loop limit (meta / attention-allocation)

```
You are reasoning about the theoretical limits of autonomous research agents.

CONTEXT. "Carnot" runs an autonomous research conductor: a multi-agent loop that
plans milestones, writes and runs experiments, verifies artifacts adversarially,
reconciles results into a record, and self-heals infrastructure — roughly 30
milestones/day, sustained over hundreds of milestones. It is genuinely good at
EXECUTION and VERIFICATION: it recovered from GPU driver corruption, a
whole-milestone test cascade caused by one malformed YAML line, and it
auto-reconciled a negative result into its own records.

THE EMPIRICAL OBSERVATION. Over a long session, every new research DIRECTION /
paradigm alternative was HUMAN-SEEDED. The loop executed each rigorously and
invented none on its own — it recombined and optimized within a human-given frame
but did not propose a genuinely new frame. (Concretely: the human had to seed
"test energy as a generator"; the loop would not have generated that hypothesis,
though once seeded it designed, ran, and honestly bounded it.)

THE QUESTION. Is this a FUNDAMENTAL limit or an artifact of current scaffolding?
Specifically: what CLASSES of research progress are reachable by such a loop
operating alone, vs. which require a human (or a qualitatively different
mechanism) to seed? Frame it in terms of search: the loop appears able to do
within-frame optimization, verification, and exploitation, but not frame
INVENTION / paradigm proposal. Is that distinction principled (e.g., the loop's
objective and priors are defined inside the frame, so it cannot value
out-of-frame moves) or is it fixable (e.g., with an explicit novelty/abduction
objective, a literature-grounded hypothesis generator, or a diversity pressure)?

WHAT I WANT FROM YOU:
- A taxonomy of research-progress types vs which are loop-reachable alone.
- The principled reason (if any) the loop cannot seed paradigm alternatives —
  tie to optimization/search theory, the explore-exploit boundary, or the
  difference between interpolative recombination and genuine abduction.
- The IMPLIED optimal division of labor: what should the human spend attention on
  (seeding frames? choosing which bounded results to escalate?) vs delegate fully?
- SEPARATE "what is true about the limit" (calibrated) from "what to build to push
  it" (flag as lower-confidence prescription). Confidence + falsifier.
```

---

## PROMPT 4 — Thesis C go/no-go (downstream of Prompt 1)

```
You are reasoning about whether one specific EBM experiment is worth running.

CONTEXT. (Read PROMPT 1's context first if available — this is downstream of it.)
Two ways of using an energy function with language models are now empirically
BOUNDED: (1) energy-as-SELECTOR (reranking AR/SC samples) does not beat the
samples; (2) energy-as-GENERATOR (an EBT that generates by energy descent) emits
incoherent output at matched compute on a discrete task an AR model solves at 82%,
even with a learned decoder — the energy landscape is discriminative but not
generative for discrete tokens.

THE UNTESTED THIRD USE. "Thesis C" = energy as a TRAINING-TIME OBJECTIVE. Using an
AR<->EBM correspondence (an autoregressive model induces an energy function over
sequences), add an energy-CONSISTENCY regularizer to standard AR training: penalize
the model when its own induced energy is inconsistent across equivalent or
self-contradictory continuations. This NEVER generates or selects via energy at
inference — it only SHAPES the AR distribution during training to be more
self-consistent. Inference remains vanilla AR.

THE QUESTION. Is there a MECHANISTIC reason to expect energy-as-training-objective
to behave DIFFERENTLY from the two bounded uses — or does the same underlying
discrete-energy limitation (from PROMPT 1: discrete decode destroys the energy
signal) predict it is ALSO bounded?

Key distinction to adjudicate: the selector and generator both put energy in the
INFERENCE-TIME path where a discrete argmax/decode sits between the energy and the
output. The training-objective use puts energy only in the LOSS, where gradients
flow through continuous logits before any discretization. Does that placement
matter — i.e., is the discrete-decode bound an INFERENCE-TIME phenomenon that a
training-time regularizer escapes, or is the limitation about discrete structure
per se (in which case training-time energy is also bounded)?

WHAT I WANT FROM YOU:
- A clear GO / NO-GO with the mechanistic argument, distinguishing inference-time
  vs training-time placement of the energy term.
- If GO: the single sharpest falsifiable experiment (with a kill-gate) — what
  would energy-regularized vs standard-fine-tuned AR have to show to count as a
  real effect rather than a tuning artifact, on what task.
- If NO-GO: state plainly that this retires the energy-foundation-model direction
  without running it, and why.
- SEPARATE survival-claim from prescription; confidence; strongest counterargument.
```

---

## How to use this round

- Paste prompts **1 and 2 first** — #1 can close a whole research direction with a
  principle (and #4's answer largely falls out of it); #2 is the only one bearing
  on the *shipped* product's long-term value.
- Prompt **3** changes how the operator allocates attention; **4** is downstream of
  **1**.
- When the answers come back, reconcile them against this session's measured
  evidence (do NOT let Deep Think re-assert anything the empirics already bounded —
  same discipline as the 2026-05-23 Phase-3 Deep Think narrowing round).
- Invariants unaffected regardless of answers: the verifier product stays banked
  (paper_ready, FoVer 0.9131 frozen); P0.1 + energy-selection + energy-generation
  remain honest-negative-bounded.
