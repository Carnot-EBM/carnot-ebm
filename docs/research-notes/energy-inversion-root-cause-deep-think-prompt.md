# Deep Think Prompt — Energy Inversion Root Cause Diagnostic

**Status:** Ready to send. This question is **time-critical** — milestone
.87 is currently spending experiments exp1119 + exp1120 on the *corpus
hypothesis* for the energy inversion. If the root cause is actually
architectural, that budget is being wasted in real time.
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project trains energy-based models (EBMs) to score the
correctness of LLM step-by-step reasoning outputs. Energy is the
ground-truth signal: lower energy = more likely correct. The
verify-repair pipeline relies on this monotonicity to rank candidate
LLM outputs and select the best one.

Across the .85 and .86 milestones, multiple experiments have
observed an **energy inversion**: the trained EBM scores
*incorrect* outputs with **lower** energy than *correct* outputs on
held-out evaluation sets. Specifically:

- **exp1099 (RLVR + SSD Integration v1):** the energy filter
  selects correct outputs more often than random ONLY when the
  pre-filter is removed; the post-EBM ranking inverts the
  correct/incorrect ordering on a 200-question held-out set.
- **exp1100 (Cascade Validation on SOTA Outputs):** running real
  Qwen 3.5 / Gemma 4 outputs through the full Tier-0 cascade,
  the energy-correctness correlation is r ≈ -0.13 (weakly
  inverted) on a 500-question GSM8K subset.
- **exp1110 (RLVR + SSD v2 with Non-Degenerate Corpus):** with a
  freshly-generated, non-degenerate corpus (where the prior
  corpus was tautologically degenerate — all energies zero), the
  inversion *attenuated* but did NOT flip; correlation went from
  r ≈ -0.31 to r ≈ -0.08.

These three experiments span different corpora, different
training procedures (RLVR, SSD, GRPO), and different evaluation
sets. The persistent inversion suggests the cause may be
*architectural* — i.e., something in the EBM head's
parameterization, loss function, or training objective that
systematically rewards incorrect answers — rather than a
*corpus* issue.

### What .87 is currently doing about it

Milestone .87 is investing two experiments in the **corpus
hypothesis**:

1. **exp1119 (FoVer SOTA Domain Extension v5):** generate 1000+
   reasoning pairs from real SOTA model outputs (Qwen 3.6 35B,
   Gemma 4 31B, Gemma 4 26B) instead of training-corpus
   curation. **Status:** complete. n_pairs = 7000+.
2. **exp1120 (Energy Verifier Retrain on SOTA Corpus):** retrain
   the Carnot energy verifier on the exp1119 corpus and re-run
   the cascade-validation evaluation from exp1100. **Status:**
   currently running.

If exp1120 lands with the inversion **resolved**, the corpus
hypothesis is empirically confirmed and we proceed to production
wiring (.87 exp1121: AND-Composition k=5 default). If the
inversion **persists**, we need an architecture-level
investigation — and that needs to start in .88.

### The gap we want Deep Think to close

We are about to consume an Opus-hours budget waiting for
exp1120 to land, but **before** running it (or at least before
basing .88 on its outcome) we want a **diagnostic methodology**
that can distinguish, with relatively cheap calculations on
*existing artifacts* (exp1099, exp1100, exp1110, exp1118), which
of these hypotheses is supported:

#### Hypothesis A: Corpus narrowness
The training corpus was too narrow / too distributionally
different from the SOTA-output evaluation set. Retraining on a
SOTA-derived corpus (.87's bet) fixes the inversion.

#### Hypothesis B: Loss-function geometry
The EBM head's loss function (likely a contrastive or
margin-based loss between correct/incorrect pairs) creates a
decision boundary that is **inverted by construction** when
the input distribution shifts. e.g. the loss minimizes
||E_correct - 0||² + ||E_incorrect - margin||² and the trained
network learns to put incorrect on the LEFT side of the
margin instead of the right.

#### Hypothesis C: Lipschitz over-regularization
The EBM head has a Lipschitz constraint (likely from spectral
norm regularization on the verifier) that pushes the energy
landscape too smooth, so any *direction* in the corpus
correlates with energy. If correct/incorrect happen to differ
on a high-frequency feature that's smoothed out, the residual
signal can flip arbitrarily.

#### Hypothesis D: Contrastive collapse onto verifier null space
The 6 verifier ensemble (post-exp1108) shares a partial joint
null space (max pairwise r ≈ 0.51, joint null fraction = 0%
per exp1108 but verifier-suite-shaped degeneracy possible).
Contrastive training onto this null space could systematically
push the gradient direction OPPOSITE to the correct/incorrect
axis if the correct/incorrect axis happens to lie in the
shared kernel.

### Specific questions

1. **Which existing artifacts contain enough data to distinguish
   A from B from C from D?** List the specific JSON keys, ranges
   of values, or computed quantities we can extract from
   exp1099, exp1100, exp1110, exp1118 to test each hypothesis.

2. **What computational tests** (cheap — runnable in <10 minutes
   on the existing data, not requiring retraining) would
   provide *independent* signal for each hypothesis? For each
   test, specify:
   - The hypothesis it tests
   - The data it requires
   - The computed quantity
   - The threshold above/below which the hypothesis is
     supported / rejected

3. **What thresholds would constitute a definitive answer**
   (i.e., which test result, at what value, would let us say
   "the cause is X" with high confidence vs. "needs
   architectural investigation in .88")?

4. **If exp1120 lands and the inversion is resolved, but only
   partially** (correlation goes from r ≈ -0.13 to r ≈ +0.05
   — i.e., positive but weak), does that confirm Hypothesis A
   or does it suggest a *combination* of A + B (corpus narrowed
   AND loss geometry inverted)? How would we tell?

### What NOT to recommend

- **Specific parameter prescriptions.** ("Use λ_lipschitz = 0.4"
  or "set margin = 0.7".) These have been systematically wrong in
  prior Deep Think rounds for Carnot. Stick to methodology and
  diagnostic test design.
- **Generic ML advice** ("try data augmentation", "tune learning
  rate"). The question is *which* of the four named hypotheses
  is supported, not *how to fix it*.
- **Training-required experiments.** We're under time pressure;
  any test requiring a fresh training run will take 20+ Opus
  hours and has its own confound risk.

### Output format request

For each hypothesis (A, B, C, D), provide:

1. **Diagnostic test** — name and one-sentence description.
2. **Required data** — specific artifact files and JSON keys.
3. **Computed quantity** — formula or pseudocode.
4. **Decision threshold** — value(s) above/below which the
   hypothesis is supported, rejected, or inconclusive.
5. **Confidence calibration** — what would make this test
   *less* trustworthy (selection bias, sample size, etc.).

Then a final **decision tree** showing how to combine the four
test results into one of these conclusions:

- "Corpus is the cause" → exp1120 will resolve the inversion;
  proceed to production wiring (.87 exp1121).
- "Loss geometry is the cause" → exp1120 will NOT fully
  resolve; .88 needs an EBM-head investigation task.
- "Lipschitz over-regularization" → .88 needs a spectral-norm
  ablation task.
- "Verifier null-space contamination" → .88 needs a verifier-
  ensemble null-space measurement task with the new k=5
  ensemble (operationalized in exp1121).

### Cross-validation reminder

The Carnot project has documented (memory:
`feedback_carnot_prediction_pattern.md`) that prior Deep Think
rounds have qualitative survival claims well-calibrated, but
specific numerical prescriptions systematically wrong. This
question is framed in the methodology lane (which hypothesis,
which test) rather than the prescription lane (what value to
use). If your answer drifts toward parameter prescriptions, please
flag the drift explicitly.

---

## End of prompt
