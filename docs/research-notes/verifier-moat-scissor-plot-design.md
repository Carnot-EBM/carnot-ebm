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

## Validity question — RESOLVED by the Deep Think methodology check (2026-06-03)

Verdict: **FIX-FIRST.** The naive protocol (raw step-error catch-rate on the full
residual, with a 0.5B Stage-0 early-stop) was *structurally guaranteed* to mislead.
The four corrections below are now folded into the metric + staged plan. DT's full
critique is appended after the prompt.

- **Q1 — the step-error proxy is INVALID.** The residual mixes *logic errors* and
  *terminal slips* (flawless reasoning, botched final arithmetic/extraction). A step
  verifier validates every step of a terminal slip → "accepts" a wrong answer → a
  false positive. Terminal slips RISE with model scale (35B resolves the logic, slips
  at the end), so the ensemble's apparent FPR inflates at the high end and FLATTENS
  the scissor — hiding the moat. **Fix:** score a **system-level trajectory verdict** —
  `reject if (any step invalid) OR (final-answer extraction invalid)`, adding a
  deterministic final-answer check (SymPy/exact-match) to the step ensemble. (And/or
  evaluate step-catch only on the subset where the error genuinely lives in a step.)
- **Q2 — abstaining-verifier vs continuous-baseline is unfair by default.** **Fix:
  coverage-matched FPR** — compute the ensemble's natural coverage `C = 1 −
  abstention_rate` on the residual; force each continuous baseline to abstain on its
  lowest-confidence `(1−C)` fraction; compare FPR strictly on the covered `C` fraction.
  (Do NOT use AURC for the ensemble — it's a fixed-threshold operator, not a curve.)
- **Q3 — the 0.5B Stage-0 early-stop is a false-negative TRAP.** 0.5B emits gibberish
  that cheap baselines catch easily while the AST/Z3 ensemble abstains/crashes on
  malformed syntax → gap ≈ 0 at 0.5B for *syntactic*, not logical, reasons. The moat
  only manifests once models are fluent enough to produce confident logical
  hallucinations. **Fix: two-point bracketing stop** — Stage-0 runs the weakest model
  AND a mid model past the formatting phase transition (~7–8B, or empirically the
  smallest model whose AST-parse abstention on its residual is low). Kill the sweep
  only if there's no edge on the *mid* residual.
- **Q4 — the residual's composition drifts violently across the ladder.** Confounds
  that FAKE a moat (parsing success rises with scale; learned-probe OOD on weak
  models; LLM-judge sharing weights with the generator) and that HIDE one (35B
  residual is objectively harder math → solver FPR rises organically; P(error)→0 at
  35B shrinks the denominator → variance explodes). **Fixes:** stratify the scissor by
  external MATH difficulty Level and plot the slope *within a fixed bucket* (e.g.
  Level-3 only); force syntactic parity with heavy few-shot prompting on every base
  model; **ablate the learned probes and re-plot with pure SAT/AST** to prove the moat
  is independent of probes; lock the LLM-judge to a structurally disjoint frozen model
  family; use a **dynamic generation budget** — over-generate at strong models until
  the residual yields exactly `K` trajectories, matching the weak-model denominator.

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

## The metric (revised per the methodology check)

For each base model M at capability c(M):
1. Generate on a headroom math set **with heavy few-shot syntactic prompting**
   (force well-formed output → structural parity across the ladder, Q4); extract
   final answers; label correct/incorrect by external difficulty **Level**.
2. **Residual subset R(M)** = problems M gets WRONG. Use a **base-rate-matched
   budget**: over-generate at strong models until R(M) holds exactly `K` trajectories
   (constant denominator across the ladder, Q4).
3. For each item in R(M), get a "this answer is fine" verdict from each verifier:
   - **Baselines:** mean-token-logprob; self-consistency agreement (Maj@K share);
     LLM-as-judge accept-prob (judge = a structurally disjoint frozen family, Q4).
   - **Carnot = SYSTEM-LEVEL TRAJECTORY VERDICT (Q1):** `reject if (any step invalid)
     OR (deterministic final-answer extraction invalid)` — the constraint/probe
     ensemble + a SymPy/exact-match final check. Also run a **pure-SAT/AST variant
     (probes ablated, Q4)** to show the moat is probe-independent, and the LLM-verifier
     as a labelled negative control.
4. **Coverage-matched FPR-on-residual (Q2):** let `C = 1 − ensemble_abstention_rate`;
   force each continuous baseline to abstain on its lowest-confidence `(1−C)` fraction;
   FPR = false-accepts of wrong answers, computed on each method's covered `C` fraction.
5. **Scissor = FPR-on-residual vs c(M), STRATIFIED within a fixed difficulty Level
   (Q4)** (e.g. Level-3 only) so the slope isn't confounded by difficulty drift.
   Predicted: baselines RISE with scale, Carnot trajectory-verdict stays FLAT;
   divergence ⇒ moat real.

## Staged plan (revised — two-point bracket, not a single weak point)

- **Stage 0: two-point bracketing early-stop (Q3).** Run the metric on the **weakest**
  model AND a **mid** model past the formatting phase transition — empirically the
  smallest GGUF whose AST-parse abstention on its residual is LOW (candidate:
  gemma-4-E4B-it ~4B; verify abstention, step up to a ~7–8B if it's still
  syntactically shaky). The cached single-model residual is one free data point but is
  NOT a valid stop on its own. **Kill the sweep only if the trajectory-verdict ensemble
  shows no coverage-matched edge on the MID residual** (where syntax is clean, so a
  flat gap is about logic, not formatting). Light GPU (small + mid models).
- **Stage 1 (GPU, multi-hour): the full sweep.** ≥4 ladder points (mid → 26B → 31B →
  35B) on the stratified headroom set, base-rate-matched `K`, coverage-matched FPR;
  test for WIDENING within a fixed difficulty Level. Conductor PAUSED, internal GPU
  cuda:1 (eGPU idle — drops under sustained load), checkpointed per model.

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
