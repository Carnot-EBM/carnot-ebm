# Verifier-as-DETECTOR measurement — spec (cheap; next direction)

**Status: QUEUED (2026-06-14, operator-chosen next direction — "cheap").**
**Origin:** the `.386–.388` arc proved the verifier-as-SELECTOR moat is *headroom-conditional*
(Sudoku ~0 headroom → null; code 18pp → circular win; ARC ~13pp → uncaptured). The SELECTOR
metric (oracle@K − vote) only measures "can the verifier pick a better answer from a pool."
It does NOT measure the verifier's *other* value: **can it correctly FLAG/REJECT a wrong
output** — detection, for abstention / "I don't know" / filtering / precision. Detection has
value **even where selection has zero headroom** (e.g. Sudoku: no recoverable alternative to
select, but the verifier can still tell you the answer is wrong). This measurement isolates it.

## The question

For a labeled set of (output, is_correct) pairs, how well does the verifier's score separate
correct from incorrect outputs? Metric = **detection AUROC** (verifier_score ranking
is_correct), plus precision@high-recall and the abstention curve (accuracy vs coverage if you
reject the lowest-scored k%). This is orthogonal to headroom: a verifier can have AUROC ≫ 0.5
(useful detector) while oracle@K ≈ vote (useless selector).

## Why it's cheap

Uses data we ALREADY have — no new generation, no training. Two ready corpora:
1. **Sudoku (executable verifier on the TRM snapshots).** We already have the curve snapshots
   (`results/trm_runs/snapshots/ckpt_val*.ckpt`) and the executable verifier
   (`nano-trm/src/nn/sudoku_evaluator.py:check_sudoku_validity` + constraint-satisfaction
   count). For each model output: verifier_score = satisfied_constraints / total_constraints
   (or exact-valid bool); is_correct = exact match to the solution. Compute AUROC pooled over
   the test set. **Prediction:** high detection AUROC even at the converged 0.82 checkpoint
   where selection headroom was ~0 — the headline contrast (detector works where selector
   can't).
2. **Cross-domain (cached pools).** Reuse the cached candidate pools from the headroom census
   (`results/experiment_4175_*`, code HumanEval/MBPP, GSM8K) + the GAP-4 ARC pool
   (`results/arc3_trm_verifier_rerank.json`). For each, verifier_score vs is_correct → AUROC.
   The honest comparison is **detection AUROC vs the SELECTOR headroom** per domain: show the
   two axes diverge (a verifier can detect but not select, or vice-versa).

## Method (one probe, ~CPU/GPU-light)

`scripts/exp_verifier_detector_auroc.py` (extends the headroom probe):
1. For each domain, load (output, is_correct) + compute the executable verifier_score per
   output. (Sudoku: decode the predicted grid to digits — verify the token→digit map with a
   sanity check that greedy exact-accuracy reproduces ~0.79 test before trusting any AUROC.)
2. AUROC(verifier_score, is_correct) with bootstrap CI95; also Brier / precision@recall=0.9 /
   the accuracy-vs-coverage abstention curve.
3. Report per-domain: detection_auroc, selector_headroom (from `.387/.388`), n. The headline
   is the **divergence** — detection value where selection had none.

## Guardrails (don't re-make tonight's mistakes)

- **Positive/negative control:** report the base rate (fraction correct) and a random-score
  AUROC≈0.5 baseline; AUROC must beat 0.5 CI95-excl to count.
- **Degenerate-detector trap:** on Sudoku the executable check is near-perfect at flagging
  *invalid* grids — that's real but partly trivial. Also report AUROC restricted to
  **valid-but-wrong** outputs (the hard detection case), so we don't over-claim on the easy
  invalid-grid split.
- **Substrate honesty:** declare `inference_substrate` (cached-pool scoring vs live); no
  fabrication; sanity-check decoding before trusting numbers (the headroom-probe discipline).
- This is a **detector** claim, NOT a selector/reward claim — keep it separate from the moat
  gate. It does NOT move the DiffusionGemma gate (which is about guidance/selection).

## What a result means

- **High detection AUROC where selection headroom ~0 (expected on Sudoku):** the verifier has
  real value as an error-detector / abstention signal independent of the selection moat —
  a genuine, defensible Carnot capability (precision/"I don't know") that the headroom null
  did NOT refute. This is the honest reframing of "where does the verifier add value."
- **Detection AUROC ≈ 0.5 somewhere with real headroom (e.g. ARC):** then the verifier can
  neither select nor detect there — the GAP-4/ARC discrimination problem is the bottleneck on
  *both* axes, sharpening the frontier.

## Execution

Cheap enough to run as outer-loop on the free GPU 0 (concurrent with the conductor), OR
pre-stage as the `.389` headline. Outer-loop preferred for rigor (sanity-checked decoding).

## Cross-references
- `docs/research-notes/verifier-graft-v3-design.md` — the selector/headroom story this complements
- `results/trm_runs/headroom_curve/` + `results/experiment_4175_headroom_gate_executable_census.json`
- exp2837 (FoVer detector AUROC 0.9131) — prior detection-AUROC precedent in the project
- CLAUDE.md "Adversarial Artifact Verification" / FALSE_NEGATIVE_RISK — controls applied here
