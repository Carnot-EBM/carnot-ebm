# Draft task: selection denominator for the FoVer headline (AUROC 0.9131)

Status: DRAFT. Not queued. Not in any roadmap. It needs an operator decision and the
pre-launch checks listed at the end. Written 2026-10-08.

Origin: the review in `docs/research-notes/openai-math-repo-lessons-2026-10-08.md`. That
repository states how many candidates were tried (about 4,000) and how many were kept
(372). Carnot states no equivalent for its FoVer headline.

## The question

How many verifier, weight, corpus and subset choices were tried on FoVer rows before the
headline was fixed? And does the headline still hold on rows that no earlier choice used?

## Facts read from the repository (verified 2026-10-08)

- The headline source is `results/experiment_2837_fover_memory_leakage_v3.json`.
  Condition A reaches AUROC 0.9131 (CI95 [0.9027, 0.9235]). It uses n=1,000 examples and
  5 seeds (42, 137, 271, 314, 1729).
- The score is a fixed formula in `python/carnot/eval/fover_memory_leakage_v3.py`:
  `0.9 * tier0r_curry_howard + 0.1 * tier0u_logical_consistency`, plus
  `FR11_MEMORY_BOOST * memory_score` where `FR11_MEMORY_BOOST = 1.0`.
- The verifier `tier0s_arithmetic_gap` is scored (AUROC about 0.29) but is NOT in the
  formula. `ops/north-star.md` calls the headline a "4-verifier score". The formula uses
  three components. This text mismatch must be confirmed and reported.
- A search of the v3 and v4 scorer files and the `scripts/experiment_27*` and
  `scripts/experiment_28*` files found the weights `0.9 * r_score` first in commit
  `0e0b0d557e`, the commit that created the v3 experiment. Other code was not searched.
  The earlier origin of the weights is not yet known.
- The headline corpus is `data/fover_corpus.jsonl` (8,829 rows). A different file,
  `data/fover_corpus_v4.json` (6,548 rows), appears in `ops/exclusion_manifest.yaml` for the
  retired candidate-selection work. Which file each variant used must be recorded.
- The source corpus `data/fover_corpus.jsonl` has 8,829 rows: 1,659 error rows and 7,170
  correct rows.
- Each seed draws a balanced subset: 500 error rows and 500 correct rows. So each seed
  uses about 30 percent of the error rows. The five subsets overlap. The reported CI comes
  from the spread across these five subsets. It measures subsample noise. It is not five
  independent replications.
- The project has 59 FoVer-named files in `results/`. Many are earlier variants.
- Related, already done: exp3719 scored a different corpus (GSM8K process rows) with the
  frozen formula and got AUROC 0.7986. Its verdict is "FoVer specific, generalization
  narrowed". This task does not repeat it.

## Definition of "variant tried" (frozen before any counting)

A variant counts if it produced a stored or reported AUROC on FoVer-derived rows AND it
differs from the headline in at least one of these:

1. the verifier set or the score formula (weights, boost, components);
2. the corpus file or the label definition;
3. the subset size, the sampling method or the seed list;
4. the state condition (FR-11 memory present or reset).

Count abandoned variants as well as reported ones. Count a variant once, however many
files restate it. Do not change this definition after the first count.

Each variant gets one tag:

- `NOT_LABEL_INFORMED`: the choice was fixed from a stated rule or an earlier, separate
  corpus. Evidence is a commit message, a prompt or an artifact field.
- `LABEL_INFORMED`: the choice was made after seeing AUROC on FoVer labels.
- `UNKNOWN`: the evidence cannot be recovered. Count `UNKNOWN` as `LABEL_INFORMED`.

## Concrete steps

0. PRECONDITIONS (check before any other step). This task needs no GPU, no network, and no
   model.
   a. `data/fover_corpus.jsonl` exists and has 8,829 rows.
   b. `results/experiment_2837_fover_memory_leakage_v3.json` exists.
   c. `git log` runs in the repository.
   If one is missing, write `honest_verdict: blocked_<resource>` and stop. Do not infer
   any value.
1. Print a progress line, flushed, at the start of each step below and inside every loop.
2. Enumerate candidate variants. Search `results/` for FoVer-named files and for artifacts
   that cite a FoVer corpus file. Record each artifact, its first-commit date, the corpus
   file it used (path, row count, sha256), and which of the four difference classes it
   differs in. Output the enumeration before tagging.
3. Trace the origin of the constants 0.9, 0.1 and 1.0. Read the history of the scoring
   code and the earlier leakage experiments. Record the earliest commit and the stated
   reason. If no reason exists, record `UNKNOWN`.
4. Tag every variant with one of the three tags. Cite the evidence for each tag.
5. Compute the exact set of error rows and correct rows used by any seed in the headline
   lineage. Report coverage. The remainder is the never-sampled set.
6. Score the frozen formula on the never-sampled set, balanced as far as the rows allow.
   Do not change the formula. Compute a bootstrap CI over rows (2,000 resamples, fixed
   seed).
7. Compute a bootstrap CI over rows for the frozen formula on the full corpus. Compare it
   with the reported CI. Report the width ratio.
8. Positive control. Fit the weights on the headline subsets by grid search. Score the
   fitted weights on the headline subsets and on the never-sampled set. The gap is an
   upper bound on how much label-informed tuning can inflate the result. If the gap is
   zero for a reason other than a real null, report the control as degenerate.
9. Report the `tier0s` mismatch between the north-star text and the formula.

## Required artifact fields (each with a principle)

- `variant_definition_text`: the frozen definition above. Principle: a definition written
  after counting can be tuned to the answer.
- `n_variants_total`, `n_not_label_informed`, `n_label_informed`, `n_unknown`. Principle:
  `UNKNOWN` is reported, not hidden, because it is counted as label-informed.
- `variant_table`: one row per variant with artifact path, date, difference class, tag and
  evidence. Principle: a count without rows cannot be checked.
- `constants_provenance`: origin of 0.9, 0.1, 1.0. Principle: a fixed constant with no
  stated origin is a possible hidden tuning step.
- `never_sampled_counts`, `held_out_auroc`, `held_out_auroc_ci95`, `n_held_out`.
  Principle: the CI must come from rows, not from seeds.
- `full_corpus_bootstrap_ci95` and `reported_ci_width_ratio`. Principle: five subsamples of
  one corpus are not five replications.
- `tuning_optimism_gap`. Principle: it bounds the damage label-informed tuning can do.
- `headline_text_vs_formula_discrepancy`. Principle: the claim text must match the formula
  it describes.
- `verifier_is_oracle: false`. Principle: the ensemble is scored against labels it does not
  produce. Note in `methodology_note` that FoVer labels come from formal tools.
- `inference_substrate: verifier_ensemble_against_cached_candidates`, `random_seed`,
  `reproducibility_checksum`, `preconditions_checked`, `duration_s`, `honest_verdict`.
- `honest_verdict` MUST start with `complete:` or `blocked_`. Allowed endings:
  `complete: headline_survives_held_out`, `complete: headline_narrowed_by_held_out`,
  `complete: denominator_reported_held_out_underpowered`,
  `complete: denominator_unrecoverable_<reason>`.

## Falsifiable acceptance gates

- Gate 1 (survives): PASS if the held-out CI overlaps the headline CI [0.9027, 0.9235].
  FAIL (narrow the paper) if the held-out CI upper bound is below 0.9027.
  Principle: both outcomes are findings. A pass supports the headline. A fail narrows it.
- Gate 2 (power): report the smallest AUROC drop the held-out set can detect. If the held-out
  set has fewer than 200 error rows, set the verdict to the `underpowered` ending. A null
  from a set that cannot detect a drop is not a finding (FALSE_NEGATIVE_RISK).
- Gate 3 (denominator stated): `n_variants_total` and all four sub-counts are present, and
  the row table sums to them.

## What this task must not do

- It must not edit the paper, the landing page, `ops/north-star.md`, or any operator-curated
  document. Report the discrepancy and stop.
- It must not retune the formula or pick a new subset to improve the result.
- It must not repeat exp3719 or the GSM8K corpus.

## Before this can be queued (operator and planner checks)

1. Operator decision: run it, change it, or drop it.
2. Check `ops/exclusion_manifest.yaml`. The entry `fover_in_domain_pool_retired_v469` has
   four blocked phrases: "fover in-domain candidate-selection pool", "fover in-domain pool",
   "in-domain verifier selection versus tuned self-consistency" and "fover selector
   adversarial audit". That scope is candidate selection, not the headline ensemble, and a
   different manifest entry says the FoVer production ensemble is outside its retired scope.
   Keep those phrases out of the task title and prompt. If the lint still fires, the planner
   adds an honest `prior_failures:` block or the operator adds an override. Do not add an
   override to hide a real rerun. This task is a new question, not a rerun.
3. Pre-stage it through `research-roadmap-next.yaml` per the Pre-Staged Roadmap Convention,
   with `agent_type: codex` and the progress-line step in the CONCRETE STEPS.
4. Estimated cost: CPU only. Scoring is fast (the headline run took 16 seconds). The work is
   mainly reading history and writing the variant table.

## Known weak points of this draft

- Step 3 may find no recorded reason for the constants. Then the answer is `UNKNOWN`, and
  that is counted against the headline by design.
- The held-out set is "never sampled by any seed in the lineage". It may still have been
  seen by an earlier experiment that tuned the weights. Step 2 and step 3 must check this
  before step 6 is read as a clean test.
- Roughly 275 never-sampled error rows are expected (an estimate from the sampling
  fractions). Step 5 computes the exact number. If it is below 200, Gate 2 applies.
