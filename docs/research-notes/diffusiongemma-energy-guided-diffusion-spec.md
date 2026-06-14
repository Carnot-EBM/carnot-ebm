# Energy-Guided Discrete Diffusion (DiffusionGemma) — Spec + GATE

**Status: QUEUED, GATED — DO NOT ACTIVATE until the TRM verifier-graft verdict lands.**
**Origin:** 2026-06-13 operator directive ("spec it into the pipeline after the TRM
baseline closes out; queued but not competing"). This is the SCALE-UP of the current
TRM verifier-guidance bet, NOT a new research direction. It tests the SAME thesis
(does Carnot's verifier add value as GUIDANCE rather than post-hoc SELECTION, on a
domain where the verifier executes?) on a billions-param non-autoregressive substrate.

## What DiffusionGemma is (verified 2026-06-13)

- Google, **Apache-2.0, open weights**, **26B / 4B-active MoE** on the Gemma 4 architecture
  (early June 2026). Sources: blog.google/innovation-and-ai/technology/developers-tools/
  diffusion-gemma-faster-text-generation/ ; ai.google.dev/gemma/docs/diffusiongemma ;
  ai.google.dev/gemma/docs/core/model_card_4 .
- **Discrete token diffusion** ("block-autoregressive multi-canvas sampling"): denoises a
  "canvas" of tokens in parallel with bidirectional attention over **12-48 steps** (adaptive
  stopping ~12-16), committing the **lowest-entropy (most-confident) tokens first**. ~4x
  faster than autoregressive decoding. Exposes **per-token entropy** + inference-time knobs
  (temperature schedule, entropy threshold, adaptive stopping, "thinking mode").

## The energy mapping (why this is on Carnot's thesis, not a metaphor)

Discrete diffusion models ARE score/energy models by construction. The denoiser learns the
**concrete score** = the gradient of log-probability in discrete space (the SEDD formalism,
Lou et al. arXiv:2310.16834), so it implicitly parameterizes an energy `E(x) = -log p(x)`
over token sequences, and each denoising step is a **descent on that energy**.
DiffusionGemma's "commit the lowest-entropy tokens first" is literally a greedy
energy-descent heuristic. The energy is the model's LEARNED LIKELIHOOD energy — computed by
running the denoiser, not read off the weights in closed form.

## The experiment (energy-guided discrete diffusion)

Compose two energies during denoising:
- **likelihood energy** (DiffusionGemma's native score / per-token entropy), AND
- **constraint energy** (Carnot's executable verifier ensemble: code execution, SAT, Z3,
  AST, Sudoku validity, etc.).

At each of the 12-48 denoising steps, reweight the candidate tokens by
`exp(-lambda * verifier_energy)` BEFORE the low-entropy commit, so the verifier **steers
generation as it forms** rather than reranking finished samples. This is the integration the
post-hoc-reranking route structurally cannot do.

**Why this is the right test (and the contrast to GAP-3/.379):** GAP-3 / .379 showed the
verifier-as-post-hoc-SELECTOR is weak — it anti-discriminates on grid outputs and does not
beat self-consistency. Energy GUIDANCE is a different mechanism: it shapes the trajectory at
each step and can prevent bad tokens from being committed, and it acts where the verifier is
strongest (executable checks, partial-constraint satisfaction mid-generation).

**Measurement / de-confound:** guided vs unguided DiffusionGemma on the SAME prompts, on an
executable domain (HumanEval/MBPP code, or Sudoku, or math) where the verifier executes:
pass-rate / constraint-satisfaction lift, bootstrap CI95 excluding 0. The load-bearing
question is whether guidance beats the unguided sampler (and beats best-of-N selection,
the commodity baseline).

## THE GATE — STATUS: STILL-PENDING (NOT MET) as of 2026-06-14

> **CORRECTION (2026-06-14, operator-directed "fix the MET").** The `.387` capstone
> (exp4183) flipped `diffusiongemma_gate_status` to **MET** on the strength of exp4177
> (`verifier_value_added=true`, +0.18) on the **code/HumanEval** domain. That was an
> **over-claim** and is hereby corrected to **STILL-PENDING**. Code is the *trivial* case:
> there the "verifier" IS the executable oracle (run the tests), so best-of-N picking the
> test-passing candidate is true-but-circular — it does not show that an *energy/learned*
> verifier adds guidance value. On the domain that actually matters, **ARC (real ~13pp
> headroom), the verifier TIES vote** (GAP-3 BOUNDED, exp4178 selection delta 0.0). And the
> `.388` C1 DiffusionGemma run (exp4189) blocked on absent weights anyway
> (`blocked_diffusiongemma_not_cached`). So nothing was actually activated — but the gate
> label must read the honest state.

**The gate is MET only when ALL of the following hold (none satisfied by the code result):**

1. **Headroom is real and present** on the test domain (oracle@K − tuned-SC-vote ≥ ~0.10,
   measured with an objective/executable oracle, artifact-inflation sanitized).
2. **The verifier is NON-TRIVIAL** — it is NOT identical to the executable oracle that
   defines correctness. Running the unit tests on HumanEval, or `check_sudoku_validity`,
   does NOT count: the verifier must be a learned / energy / partial-constraint signal that
   could plausibly transfer to a domain WITHOUT a cheap executable oracle. (This is the
   whole point — if a free executable oracle exists, you don't need Carnot's verifier.)
3. **That non-trivial verifier captures the headroom with a MATCHED no-verifier control**,
   `verifier_value_added=true`, CI95 excluding 0 — on a domain where the verifier ≠ the
   oracle. The canonical target is **ARC** (~13pp headroom, currently UNCAPTURED).

- **Gate MET (all three)** -> the verifier adds value as guidance where execution can't
  trivially reach -> DiffusionGemma is the depth scale-up. ACTIVATE (operator-gated launch).
- **Gate STILL-PENDING (current state)** -> a non-trivial verifier has NOT yet beaten a
  matched control on a headroom-present, oracle-distinct domain. DO NOT ACTIVATE. The open
  research problem is exactly: capture ARC's ~13pp headroom with an energy/learned verifier.

Rationale: the moat's value lives in the GAP between "headroom exists" and "a NON-trivial
verifier can capture it." Code (verifier == executable test) collapses that gap circularly;
ARC is where the gap is real and currently unbridged. The gate guards against scaling to a
26B substrate on the strength of a circular existence proof.

## Preconditions for the eventual experiment

- DiffusionGemma open weights cached locally (HuggingFace; verify the repo id + cache in a
  PRECONDITIONS step before first use).
- Confirm the open release exposes per-step token logits / the candidate distribution at each
  denoising step (the injection point for guidance). If it only exposes final text, the
  guidance hook may require a custom sampling loop over the released weights.
- CUDA (RTX 3090 rig). 26B/4B-active MoE fits a 3090 at 4-bit; confirm VRAM.

## Use-Case 2 (added 2026-06-13): diffusion logprobs for DETECTION / VERIFICATION

Discrete diffusion gives a DIFFERENT logprob object than an AR LLM: a variational
likelihood BOUND (ELBO; multi-pass to estimate tightly) instead of an exact one-pass
chain-rule logprob, BUT a BIDIRECTIONAL per-token surprisal — each token scored against
its full left+right context, ~ `p(x_i | x_{\i})` — plus a native per-token entropy /
uncertainty trace (lowest-entropy tokens committed first). A diffusion model is a
denoising autoencoder over tokens, so reconstruction-error-as-detection is native.

- **Use:** per-token bidirectional surprisal as a hallucination / error-LOCALIZATION score.
  A wrong/hallucinated token gets low `p(x_i | x_{\i})` because the model expected something
  else given full context — which AR perplexity (left-only) largely misses.
- **Where it earns its keep:** the NON-executable regime (NL claims, factuality, prose
  error-localization) where Carnot's EXECUTION verifiers have NO purchase. It fills the moat's
  blind spot; it does NOT strengthen the execution moat (it is a LEARNED-likelihood verifier —
  the gameable tier the project deliberately does not rely on, valuable only where execution
  can't reach).
- **Training spillover:** diffusion per-token surprisal as an INPUT FEATURE for Carnot's
  hallucination/error verifiers; or the likelihood-bound as an auxiliary RFT regularizer.
- **GATE (distinct from Use-Case 1, weaker — NOT coupled to the TRM graft):**
  (i) PRECONDITION: confirm the open release exposes the per-POSITION distribution over the
  vocab for an arbitrary input (needed to score an observed token's surprisal), not just
  entropy — likely a custom forward pass at low noise. If not exposed, blocked.
  (ii) CHEAP PROBE: score a labeled error corpus (FoVer step-error, a factuality set) with
  diffusion-surprisal vs AR-perplexity; compare detection AUROC. Activate the full
  detection-verifier integration only if diffusion AUROC beats AR-perplexity CI95-excl-0.
- **Honest framing:** promising WITH PRECEDENT (masked-LM pseudo-perplexity, diffusion-ELBO
  anomaly detection), NOT proven. AR perplexity is a strong baseline; the diffusion bound is
  looser/noisier and multi-pass.

## Use-Case 3 (added 2026-06-13): can the WEIGHTS save us from training our own EBM? (honest assessment)

Operator question 2026-06-13: can we extract value from the weights to avoid brute-force
gradient-descent training, or introspect on the weights to drastically improve behavior?
Honest separation of REAL / PLAUSIBLE / SPECULATIVE — do not conflate them:

- **[REAL — the strong one] The pretrained diffusion score IS a free, gradient-accessible
  energy-prior.** The denoiser outputs the score = `-grad E` at every point — exactly the
  object Phase-3 has been trying to TRAIN from scratch (a hardware-evaluable deep EBM over
  valid sequences). DiffusionGemma hands it to you pretrained. So composing diffusion-energy +
  Carnot verifier-energy and doing energy-DESCENT inference could SHORTCUT the "train our own
  EBM" part of Phase 3 — GUIDE/COMPOSE a frozen pretrained energy instead of brute-forcing our
  own generative model. This genuinely saves the training we have been fighting (the LoRA/TRM
  mechanism struggles). CAVEAT: it is inference-time composition of a FROZEN score, NOT
  closed-form weight extraction; and discrete-token gradient composition needs relaxations
  (Gumbel/STE) with the known sharp edges from the Q11 TSS/STE analysis (project memory).
- **[PLAUSIBLE — modest] Activation probing / steering (representation engineering).** Fit a
  probe for a "constraint-satisfaction / correctness / low-energy" direction in the diffusion
  model's activations, steer denoising along it. Real technique; typically TARGETED, MODEST
  gains, and the probe itself must be fit (not free). Plus an EFFICIENCY win: focus Carnot's
  expensive constraint-checking on the HIGH-entropy positions (where the model is unsure),
  skipping the confident ones — north-star "efficiently."
- **[SPECULATIVE — do NOT bet on it] Closed-form weight extraction / drastic improvement by
  pure introspection.** You CANNOT read a verifier / energy / constraint-set out of the weights
  in closed form — they encode a nonlinear score map; you compute by RUNNING the model.
  Model-editing to bake constraints into the weights (ROME/MEMIT-style) is possible-in-principle
  but brittle AND still gradient-based (does not avoid training). "Drastically improve via pure
  weight introspection" is frontier/wishful; flagged low-probability so we do not chase it as a
  Beachhead.

- **GATE:** the [REAL] energy-prior shortcut rides on Use-Case 1's machinery — same gate as
  Use-Case 1 (TRM graft positive). The [PLAUSIBLE] probing/steering is a small probe gated on the
  Use-Case 2 per-position-logits precondition. The [SPECULATIVE] tier is NOT queued as a bet.

## Why this is DEPTH, not breadth

It does NOT open a new research question — it is the same verifier-as-guidance question the
TRM graft is testing, on a more powerful substrate. It is the natural next rung of the
existing depth bet, sequenced behind (gated on) the cheap TRM test so it never competes with
the unfinished core question. Decentralization fit: Apache-2.0, open weights, local-first
(unlike closed diffusion LLMs) — aligns with the sovereignty rules.

## Cross-references

- GAP-3 / .379 exp4099 — verifier-as-post-hoc-selector is weak / anti-discriminates on grids
  (the finding energy-guidance is meant to address)
- The TRM verifier-graft (.382/.383, exp4119/exp4128 lineage) — the cheap pilot + THE GATE
- Phase 3 vision (CLAUDE.md) — non-AR reasoning via energy minimization; DiffusionGemma is
  the open-weight real-scale instance
- feedback_hybrid_pragmatic_architecture (memory) — open LLM generator + energy verifier;
  DiffusionGemma is a fast open generator to pair the verifier with
- SEDD discrete-diffusion score formalism — Lou et al. arXiv:2310.16834
