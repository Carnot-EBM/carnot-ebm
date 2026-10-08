# FoVer headline: selection denominator and held-out results (2026-10-08)

Run: `scripts/experiments/experiment_10030_fover_headline_selection_denominator.py`.
Artifact: `results/experiment_10030_fover_headline_selection_denominator.json`.
Task draft: `fover-headline-selection-denominator-task-draft-2026-10-08.md`.
Verdict in the artifact: `complete: headline_survives_held_out_but_learning_contribution_not_established_on_clean_rows`.

The run was done by the outer loop, CPU only, 430 seconds. The project's artifact check
reports no adversarial flags.

## Control

The script re-derived the published per-seed means before any other stage was trusted.
Condition A: 0.913134 (published 0.913134). Condition B: 0.894662 (published 0.894662).

## Result 1: the headline holds on rows no headline seed used

| Row set | Error rows | A (with memory) | A 95% CI | B (no memory) | B 95% CI |
|---|---|---|---|---|---|
| Published headline | 500 per seed | 0.9131 | [0.9027, 0.9235] | 0.8947 | [0.8842, 0.9051] |
| Never sampled by any seed (H0) | 269 | 0.9217 | [0.9024, 0.9400] | 0.9028 | [0.8802, 0.9261] |
| H0 and not memory-flagged (H1) | 226 | 0.9082 | [0.8841, 0.9285] | 0.8969 | [0.8680, 0.9228] |
| Question never sampled, not flagged (H2) | 113 | 0.8722 | [0.8309, 0.9077] | 0.8516 | [0.8032, 0.8974] |

CIs for the held-out rows are bootstrap intervals over rows (2,000 resamples). H0 and H1
overlap the published intervals and have at least 200 error rows. H2 overlaps too, but its
point estimates sit about 0.04 below the headline and it has fewer than 200 error rows. Read
H2 as underpowered, not as a pass.

The 5 headline seeds used 3,549 distinct rows, including 1,390 of the 1,659 error rows.

## Result 2: the memory term carries label-derived information

- The memory index holds 252 question ids recorded as incorrect and 156 prompt-token sets
  taken from the same incorrect-labelled entries. In the corpus the ids match 228 error rows
  and 10 correct rows. That is 13.7 percent of all error rows.
- The id match is only the exact-copy part. Most of the memory score is a soft token overlap
  with those 156 sets: 256 rows score exactly 1.0, 7,668 score between 0 and 1, and 905
  score 0 (independent reviewer's count). The mean memory score is 0.208 for correct rows
  and 0.507 for error rows. The memory term alone scores AUROC 0.871. Both parts are built
  from rows recorded as incorrect, so label-derived information goes further than the 13.7
  percent figure. Whether the overlap part transfers to unseen text is not separated here.
- Each headline seed subset contains 62 to 73 of these rows, almost all error rows.
- The "learning contribution" (A minus B) is 0.0167 on all rows. Without the flagged rows it
  is 0.0097 [0.0050, 0.0144]. On H1 it is 0.0113 [-0.0020, 0.0265], which includes zero.
- The point estimates fall when flagged rows are removed. The intervals overlap, so the
  fall is not shown to be significant. The learning contribution is not established on
  clean held-out rows at this power.

## Result 3: the "ensemble" is mostly one verifier plus the memory term

Full-corpus AUROC by component: `tier0r` 0.9017, `tier0u` 0.5099, `tier0s` 0.3016, memory
term 0.8714. The base formula `0.9*tier0r + 0.1*tier0u` scores 0.9016, within 0.0001 of `tier0r`
alone and slightly below it. All of condition A's lift over `tier0r` comes from the memory term.
`ops/north-star.md` calls the headline a "4-verifier score". The formula has three terms and
`tier0s` is scored but unused. This text mismatch is recorded in the artifact. The north-star
file was not edited.

## Result 4: tuning the weights does not explain the headline

A 44-combination grid on the headline rows chose `w_r = 1.0` and boost 1.0. It reached 0.9173
in-sample against 0.9168 for the frozen formula, a gain of 0.0005. Weight tuning on this grid
cannot account for the headline. A wider search could find more.

## Result 5: the published CI is not too narrow

The draft predicted that five overlapping subsamples would give an interval that is too
narrow. The measurement says otherwise. A row bootstrap on the full corpus gives a
half-width of 0.0088 for A against the published 0.0104, and 0.0102 for B against 0.0105.

## Denominator

- 214 artifacts were scanned (59 by file name, the rest by reference to a FoVer corpus file).
- 68 stored an AUROC and are not restatements of the headline. 31 of them were first
  committed before the headline commit (2026-05-21).
- Tags by keyword evidence in the artifact text: 13 not label-informed, 2 label-informed,
  53 unknown. By the frozen definition unknown counts as label-informed, giving 55.
- All three constants (0.9, 0.1, boost 1.0) first appear in the single commit that created
  the headline experiment, `0e0b0d557e`. That commit message gives no reason for them. No
  earlier committed code uses other weights. Tuning outside committed code would not show.

Limits of the count: the "differs from the headline in at least one class" test was not
applied per artifact. Only restatements (by name) and artifacts with no stored AUROC were
excluded, so 68 is an upper bound. Tags come from keywords, not from reading commits.

## What follows from this

1. The 0.9131 headline survives a held-out test at the power available (H0, H1).
2. The claim that FR-11 self-learning adds +0.0185 is weaker than stated. Part of it comes
   from memory entries built from labels, and it is not established on clean rows.
3. Condition B (0.8947, no memory) is the cleaner headline. It also needs one caveat: the
   second component adds nothing, so it is closer to a single-verifier result.
4. `ops/north-star.md` and the paper text should say "three terms, one of them near-inert".
   Both are for the operator to change. Nothing here edits them.

## Independent check

A reviewer agent re-derived the key quantities from the raw corpus. It did not read this
note, the artifact or the script, and it wrote its own AUROC, set and bootstrap code. It
imported only the project's per-row scorers. Its results agree with the above:

- 3,549 distinct rows used by the 5 seeds, 1,390 of them error rows, 269 error rows unused.
- Memory index: 252 ids, 238 matched rows (228 error, 10 correct).
- AUROC for A and B on all rows (0.9183, 0.9016), unused rows (0.9217, 0.9028), and unused
  and unmatched rows (0.9082, 0.8969). Identical to four decimals.
- `tier0r` alone 0.9017, base formula 0.9016, `tier0u` alone 0.5099.
- Bootstrap CI on the unused and unmatched set: A [0.8838, 0.9295], B [0.8674, 0.9242],
  with a different resampling seed. Within about 0.001 of the values above.

It did not re-derive the 0.9131 per-seed mean. This note's control stage did.

The reviewer also tested the objection "removing flagged rows removes the easy rows, so the
held-out set is just harder." Its data: the memory-free score B is higher on the matched rows
(0.9408) than elsewhere, so part of the fall on removal is row difficulty. B falls about
0.005 and A falls about 0.010 to 0.014 when those rows are removed. The larger fall in A fits a
memory contribution, but the intervals overlap and neither explanation is established. The
paired A minus B gap in Result 2 does not depend on row difficulty, and it also falls.

One caveat from the reviewer: the memory index depends on the FR-11 state files in the tree.
This run used the tracked files in a clean checkout, and the control reproduced the published
numbers exactly, so the state matches what the headline used.

## Measurements that would separate leakage from difficulty

1. Rebuild the memory index without entries derived from a random half of the corpus
   questions. Rescore the other half. This is a causal leakage test.
2. Compare the A minus B gap on difficulty-matched rows (stratify on the `tier0r` score).
3. Trace which state files contribute the 252 ids.

## Not done

- The strict set H2 needs more error rows to be decisive. It cannot be enlarged from this
  corpus.
- Commit messages were not read for the tags. Reading them could move rows between tags.
- The memory index was not traced to the sessions that built it. Whether flagged rows were
  fed back from the exact evaluated rows is not shown here.
