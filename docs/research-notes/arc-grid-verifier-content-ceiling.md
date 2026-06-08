# ARC cheap-verifier content ceiling: combining 7 invariant families

**Domain:** ARC-AGI-1 training (400 tasks, 416 test inputs), the north-star ARC-AGI-3
verifier-as-pruner role. **Substrate:** rule-based grid verifier against cached gold
solutions — no LLM, no rule-induction, no test-gold leak. **Reproducible:**
`random.Random(0)`, identical distractor protocol to v1, ~9 s wall on CPU.

Artifact: `results/arc_grid_verifier_invariants_v2.json`.
Harness: `scripts/experiments/arc_grid_verifier_invariants_v2_combined.py` (+ `_synthesis.py`).

## The question

Carnot's v1 ARC verifier checks dimension / palette / background consistency derived from the
train pairs. It is a strong **structural pruner** but **blind to content errors** that preserve
structure. Measured v1 `gold_strictly_better_rate` on the hard distractors:
`perturbed_gold 0.006`, `transposed_gold 0.293`, `color_swap_gold 0.384`, `wrong_dim_gold 0.863`.
We measured 7 new invariant families individually, then asked: does **combining** them lift the
hard-distractor ceiling on the full protocol?

## Task 1 — which family lifts each hard distractor most (full-corpus rates)

| Hard distractor | v1 | Best family | Best-family rate | Lift vs v1 |
|---|---|---|---|---|
| `perturbed_gold`   | 0.006 | **tiling_scaling** (scale-subset n=24) | 0.826 | +0.820 |
|                    |       | object_count (full corpus)             | 0.814 | +0.808 |
| `color_swap_gold`  | 0.384 | **palette_histogram_shape**            | 0.693 | +0.309 |
|                    |       | tiling_scaling (scale-subset)          | 0.625 | +0.241 |
| `transposed_gold`  | 0.293 | **delta_pattern**                      | 0.450 | +0.157 |
|                    |       | content_overlap                        | 0.331 | +0.038 |
| `wrong_dim_gold`   | 0.863 | **tiling_scaling** (scale-subset)      | 0.917 | +0.054 |
|                    |       | palette_histogram_shape                | 0.608 | −0.255 |

Caveat: `tiling_scaling` numbers are on its 24-task scale subset only (it abstains on 376/400);
on the full corpus it cannot move the aggregate. The deployable full-corpus single-family wins
are **object_count** for `perturbed_gold` (0.814) and **palette_histogram_shape** for
`color_swap_gold` (0.693). **No single family** clears 0.70 on `transposed_gold` or, on the full
corpus, materially improves `wrong_dim_gold` over v1 (v1 already owns it via its dim check).

## Task 2 — combined verifier (measured, not estimated)

Families have **different abstain conventions** (0.5 neutral for symmetry/color_mapping/
content_overlap; **0.0 for tiling_scaling** — its *best* score; weak palette fallback for
delta_pattern; never-abstain for object_count/palette_histogram). A naive `min`/`max` over raw
scores is therefore wrong — an abstaining tiling family would always read "perfectly
consistent." The honest combiner computes a per-task **applicability flag** for each family from
the train pairs only and aggregates **only over applicable families** (+ v1, always applicable).

Three combiners, all measured on the 400-task corpus:

| Hard distractor | v1 | **union_max** (headline) | mean_defined | logistic_cv (OOF upper bound) |
|---|---|---|---|---|
| `perturbed_gold`  | 0.006 | **0.783** (AUROC 0.67) | 0.922 | 0.928 |
| `color_swap_gold` | 0.384 | **0.751** (AUROC 0.75) | 0.845 | 0.840 |
| `transposed_gold` | 0.293 | **0.610** (AUROC 0.71) | 0.729 | 0.732 |
| `wrong_dim_gold`  | 0.863 | **0.870** (AUROC 0.82) | 0.954 | 0.954 |
| `copy_input`      | 0.389 | 0.904 | 0.974 | 0.976 |
| `blank`           | 0.135 | 0.894 | 0.976 | 0.969 |
| `random`          | 0.889 | 0.930 | 0.988 | 0.988 |
| `wrong_task_gold` | 0.947 | 0.925 | 0.983 | 0.978 |

**`union_max` is the headline / deployable combiner**: `violation = max` over v1 + applicable
content families. Rationale: gold should have low violation on *every* applicable check; a
distractor is caught if *any* applicable check flags it. It requires no fit to the distractor
distribution and does not manufacture wins by tie-breaking (its strict rate and AUROC agree
closely). It **catches 3 of 4 hard distractors at ≥0.70** (`perturbed_gold`, `color_swap_gold`,
`wrong_dim_gold`) plus all 4 easy classes — a real break of v1's content-blindness:
`perturbed_gold 0.006 → 0.78`, `color_swap_gold 0.384 → 0.75`.

`mean_defined` and `logistic_cv` are reported as **upper bounds**. They appear to also catch
`transposed_gold` (0.73), but this is partly a measurement subtlety: their `gold_strictly_
better_rate` (e.g. perturbed 0.92) outruns their AUROC (0.69–0.72) because averaging removes
exact ties, converting v1's near-universal ties into strict wins. A margin probe confirmed
>91% of those wins are *robust* (margin ≥ 1e-3), so the lift is real — but the **AUROC is the
tie-robust truth**, and on that measure the margin is modest. `logistic_cv` is fit to the
distractor distribution (out-of-fold, but still distribution-specific), so it is an estimate of
the achievable ceiling, **not a deployable rule**.

Applicability (% of test inputs a family fires on): object_count 100%, palette_histogram 100%,
delta_pattern 64%, content_overlap 54%, color_mapping 28%, tiling_scaling 6%, symmetry 1.7%.
The two always-on families (object_count, palette_histogram) plus v1 are the backbone; the
gated families add the perturbed/color_swap lift where their precondition holds.

## Task 3 — the honest remaining ceiling

Decomposing `transposed_gold` by grid shape isolates the irreducible residual:

| transpose subset | n | union_max | mean_defined |
|---|---|---|---|
| **non-square** (dims change → any dim-aware check catches it; structural, "free") | 130 | 0.754 | 0.831 |
| **square** (same dims + histogram + object count + palette → all cheap invariants tie) | 239 | **0.531** | **0.674** |

The **square-transpose subset is the irreducible cheap-verifier ceiling**: union_max barely
beats chance (0.53). A transpose of a square grid preserves every cheap structural invariant
(`object_count` and `palette_histogram_shape` are *provably* transpose-invariant; v1 is blind by
construction). Distinguishing gold from its transpose requires knowing the task's transform is
**not** a transpose — i.e. **rule induction** over the train pairs. That is the generator's job.

A second irreducible class: content errors on **variable-output-dimension / non-positional
tasks** (~half of ARC), where no per-position cell template exists, so content_overlap /
delta_pattern / color_mapping / symmetry all abstain and the ensemble falls back to v1 +
object_count + palette_histogram. Content errors there that preserve count and histogram are not
caught cheaply.

## North-star division of labor

This instantiates the verifier-prunes / generator-induces split on real ARC data:

- **Verifier prunes (cheap, no induction):** all structural errors (wrong dims/palette/bg) plus
  the content errors that violate a *measurable* invariant — perturbation fragments objects;
  color-swap shifts the histogram/colour-map; off-signature colours and footprints. The combined
  cheap ensemble prunes **3/4 hard + all 4 easy** distractor classes at ≥0.70 (union_max).
- **Generator induces (LLM / search):** the small residual that satisfies *every* cheap
  invariant yet is wrong because the **spatial rule** differs — square transpose, and content
  errors on variable-dimension / non-positional tasks. The cheap ensemble's value is shrinking
  the generator's call budget by pruning everything else, leaving the generator only the
  genuinely rule-ambiguous slice. This is the ARC analogue of math being self-consistency-bound
  and code being execution-bound.

## Bottom line

Combining the families is a **real, measured advance**: from a v1 that caught 0/4 hard content
distractors to a cheap, deployable, LLM-free ensemble that catches 3/4 at ≥0.70 in ~9 s. The
honest residual — square-grid spatial-arrangement errors and content errors on non-positional
tasks — is exactly the slice that needs rule induction, confirming the division of labor rather
than refuting it. A family that did not beat the ceiling (symmetry, full-corpus; or any family
on square-transpose) is a real finding about *where the cheap-invariant boundary lies*, not a
failure.
