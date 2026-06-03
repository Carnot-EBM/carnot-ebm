# Verifier-moat "scissor plot" — precise design + Deep Think methodology check

**Status:** DESIGN doc, 2026-06-03, outer-loop. Operationalizes Deep Think P2
(`reference_deep_think_post_bounded_2026_06`): the verifier's durable value is
**error INDEPENDENCE, not AUROC** — does it catch residual errors the cheap
self-verification baselines (logprob / self-consistency / LLM-judge) miss, and does
that gap WIDEN as base models improve (the "scissor")? This doc grounds that abstract
protocol in Carnot's actual assets and flags the validity questions to run by Deep
Think BEFORE any GPU is spent. **Do not build until the methodology check returns.**

## The one framing decision (load-bearing)

DT's moat argument is **mechanistic independence breaks the sycophancy cascade** —
it applies ONLY to a verifier that does NOT share the generator's latent
representations. Carnot has two different "verifiers":

- **Constraint/probe ensemble** — `conformal_ensemble.py` combines per-verifier
  scores (`collect_verifier_scores` → conformal p-values / Stouffer) from
  `sat.py`, `z3_math_verifier.py`, `ast_structure_verifier.py`, `kan_smt_verifier`,
  `nsvif_z3_extractor`, + the learned probes. **This is the FoVer-headline 0.9131
  ensemble and the ONLY one with the independence claim** (Z3/SAT/AST are
  mechanistically alien to the generator). **← the scissor-plot MUST target this.**
- **LLM-as-verifier** — `clean_local_sota_verifier_rerun_v14.py:run_llama_local_verifier`
  (a local SOTA GGUF emitting accept/reject/abstain). This SHARES the generator's
  representations, so it would (correctly) show a RISING residual-FPR like the cheap
  baselines. Running the scissor on this would falsely "disprove" the moat by
  testing the wrong thing. **Exclude it (or include only as a deliberate
  negative-control that should track the baselines).**

## The unresolved validity question (the reason we ask Deep Think first)

The headline ensemble scores **step errors** (per-step correctness), not
**final-answer correctness** directly. The moat claim is about *catching wrong
answers the baselines wave through*. A wrong final answer may have all-correct steps
(a final arithmetic slip) or correct steps with a wrong answer. So: **is
"step-error-catch-rate on the residual" a valid proxy for the final-answer-correctness
moat?** This is the single thing that, if wrong, wastes the whole GPU sweep — hence
the methodology check.

## Assets (all on disk, confirmed 2026-06-03)

- **Model ladder (capability sweep axis), GGUF via llama.cpp:** Qwen2.5-0.5B /
  Qwen3-0.6B (HF) → Qwen3.5-0.8B-GGUF → gemma-4-E4B-it-GGUF (~4B) →
  gemma-4-26B-A4B-it-GGUF → gemma-4-31B-it-GGUF / Qwen3.6-35B-A3B-GGUF. A real
  0.5B→35B ladder.
- **Cached single-model residual (one scissor point, free):**
  `data/p01_difficulty_matched_generations_flattened_v2.jsonl` — 558 generations,
  368 incorrect (= residual), with `text`, `extracted_answer`, `correct`/`is_correct`,
  and `mean_token_logprob` (the logprob baseline; partially null — recompute where
  missing).
- **Headline math corpus + lineage:** FoVer (exp2837/exp2850), the
  difficulty-matched / MATH-500-headroom corpora (`corpus_p01_headroom.py`).
- **Ensemble scorer:** `conformal_ensemble.py` (`fit` / `predict` /
  `predict_stouffer`); per-verifier raw scores via the detector's
  `collect_verifier_scores`.

## The metric

For each base model M at capability c(M):
1. Generate on a headroom math set; extract final answers; label correct/incorrect.
2. **Residual subset R(M)** = problems M gets WRONG.
3. For each item in R(M), get a "this answer is fine" score from each verifier:
   - **Baselines:** mean-token-logprob; self-consistency agreement (Maj@K share);
     LLM-as-judge accept-prob.
   - **Carnot:** the constraint/probe ensemble score (+ the LLM-verifier as a
     labelled negative control).
4. **FPR-on-residual** = fraction of R(M) the verifier rates "fine" (a false accept
   of a wrong answer) — at a coverage-matched operating point (see open Q2).
5. **Scissor** = plot FPR-on-residual vs c(M). Predicted: baselines RISE, Carnot
   constraint-ensemble stays FLAT; divergence ⇒ moat real.

## Staged plan (cheapest decisive cut first — the session's discipline)

- **Stage 0 (cheap, ~no GPU): necessary-condition cut.** On the cached single-model
  residual (368 wrong answers), compute FPR-on-residual for the constraint ensemble
  vs logprob (and SC where derivable). If the ensemble does NOT beat the cheap
  baseline AT ALL even here → the moat is absent at this point → **STOP, report the
  negative, do not run the sweep.** (Pending open-Q3: is a flat single weak point a
  valid early-stop?) If it beats them →
- **Stage 1 (GPU, multi-hour): the sweep.** Generate from ≥4 ladder points on the
  headroom set; repeat the residual-FPR; test for WIDENING. Conductor PAUSED, runs
  on internal GPU cuda:1 (eGPU drops under sustained load), checkpointed.

## Infra discipline (from this session)

Pause the conductor for the GPU sweep (its `gpu_monitor.kill_zombies` SIGTERMs
foreign GPU procs); run on internal GPU `cuda:1`, leave the eGPU `cuda:0` idle (it
drops off USB4 under load and corrupts CUDA); resume the conductor when done. Don't
leave it paused longer than the run.

## Invariants

This is a measurement on the EXISTING verifier; it does not touch the frozen FoVer
0.9131 / paper_ready state. A clean negative (no scissor / no Stage-0 gap) is a
high-value finding about the product's durability, not a regression.

---

## Deep Think methodology-validation prompt (paste this; do NOT ask it to design)

```
You are validating the METHODOLOGY of a falsifiable measurement before we spend
GPU hours on it. Do not redesign it from scratch; pressure-test it.

GOAL. Test whether an LLM-output VERIFIER has a durable moat = ERROR INDEPENDENCE
(it catches errors the generator's own cheap self-verification misses), and whether
that edge WIDENS as base models improve ("scissor plot": plot the verifier's
false-positive-rate on the RESIDUAL subset — problems the base model got WRONG —
against base-model capability; predict cheap baselines' FPR rises while the
external verifier's stays flat).

CONCRETE CONSTRAINTS (these are fixed; design around them):
- The verifier under test is a CONSTRAINT/PROBE ensemble (SAT, Z3/SMT, AST-structure,
  plus learned probes), mechanistically alien to the generator. (We exclude an
  alternative LLM-as-verifier because it shares the generator's representations and
  would trivially track the baselines — we'll run it only as a negative control.)
- IMPORTANT: this ensemble natively scores STEP-LEVEL errors (is each reasoning step
  valid), not final-answer correctness directly. The moat claim is about catching
  WRONG FINAL ANSWERS the baselines accept.
- Baselines: generator mean-token-logprob; self-consistency agreement (Maj@K);
  LLM-as-judge.
- Capability ladder: real models 0.5B -> 35B on math (a headroom set where the
  weak models are far from saturated and the strong models still err sometimes).
- The residual subset is DEFINED by the base model's own correctness label.

VALIDATE THESE FOUR THINGS (answer each crisply):

Q1 (PROXY VALIDITY). Is "step-error-catch-rate on the residual" a valid proxy for the
"final-answer-correctness moat"? A wrong final answer can have all-correct steps (a
terminal slip) or correct steps with a wrong answer. Under what conditions does
step-error catch-rate over/under-estimate the final-answer moat, and is there a
strictly better target (e.g., score the FINAL answer with a final-answer verifier,
or measure step-catch only on the subset where the error IS in a step)? If the proxy
is invalid, say so plainly and give the correct target.

Q2 (ABSTENTION / FAIR COMPARISON). The ensemble can ABSTAIN; logprob/SC give a
continuous score. How do we compare a verifier-that-abstains against
baselines-that-don't WITHOUT advantaging either? Specify the right operating-point
discipline (coverage-matched FPR? area-under-risk-coverage AURC? a fixed
abstention budget?) and the metric to report.

Q3 (EARLY-STOP VALIDITY). We plan a cheap Stage-0 cut: if on ONE weak model's
residual the ensemble does not beat the cheap baseline at all, STOP before the GPU
sweep. Is that a valid early-stop, or could the SCISSOR (gap widens with capability)
genuinely exist even when the gap is ~0 at one weak model — making the early-stop a
false negative? If it can, give the minimal multi-point design that is still cheap
but not fooled.

Q4 (CONFOUNDS). The residual is defined by the generator's own correctness. Name the
confounds that could FAKE a scissor (make it look like a moat when there is none) or
HIDE one (mask a real moat): circularity/label leakage, train/test contamination of
the verifier, selection effects from extraction/parsing, difficulty drift across the
capability ladder, and base-rate shifts (P(error) falls with capability, changing the
denominator). For each, the cheapest control.

OUTPUT DISCIPLINE: separate "what is true about the methodology" (your calibrated
zone) from any "what to build" prescription (flag as lower-confidence). For each of
Q1-Q4 give a confidence and the single check that would falsify your answer. End with
a GO / FIX-FIRST verdict: is the protocol as specified sound enough to spend GPU on,
or is there a specific fix required first?
```

When the answer returns, fold Q1–Q4 into the "metric" and "staged plan" sections,
THEN build (Stage 0 first).
