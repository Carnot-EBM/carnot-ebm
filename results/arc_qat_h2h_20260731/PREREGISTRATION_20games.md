# Pre-registration — 20-game extension of the QAT vs Q4_K_M head-to-head

Written BEFORE running the 7 new games, and committed, so the analysis choices cannot be
made after seeing their results.

## Why this is an extension and not p-hacking

Expanding a sample after a null is a forking-paths hazard. Three commitments contain it:

1. **No selection on outcome.** The extension is ALL remaining games that have a usable
   offline L1 window -- 7 of the 12 non-roster games. The other 5 (dc22, ka59, sc25, tn36,
   wa30) are excluded because `build_progress_window` cannot produce a window for them, a
   property of the fixtures that is independent of either arm's score. No game was chosen
   for how it might move the result, and none can be dropped later.
2. **The criterion does not change.** Per-game zero-imputed held-out mean (mean-B), exact
   two-sided sign test over paired games. Identical to the 13-game run and to exp6021.
3. **Both results get reported.** The 13-game result (5-2-6, p=0.453) stands in the record.
   The 20-game result is reported alongside it, not instead of it. If they disagree, that
   disagreement is the finding.

## Pre-specified decision rule

- `p < 0.05` AND qat ahead  -> adopt QAT
- `p < 0.05` AND q4km ahead -> keep Q4_K_M, and treat the 1 GB VRAM saving as not worth it
- otherwise                 -> INDISTINGUISHABLE, keep the shipped Q4_K_M

A null at 20 games is a stronger null than at 13, not a licence to extend again. **This is
the last extension**: the remaining 5 games are unavailable, so 20 is the ceiling of this
corpus. If the answer is still null, the honest conclusion is that quantisation choice does
not move induction quality measurably at this corpus size, and the decision falls back to
the non-quality axes (VRAM, MTP drafter availability).

## Power

13 games gave 7 discordant pairs (min reachable p = 0.0156). 5 of the 6 ties were BOTH-ZERO
floor games where neither arm induced anything -- those are uninformative by construction,
not noise. If the informative rate (~54%) holds, 20 games gives ~11 discordant pairs and a
min reachable p near 0.001. That is the entire gain being bought; it does not make a small
effect detectable, only a moderate one.

## Cost

7 new games x 3 trials = 21 new cells per arm. The 39 existing cells are cached per-cell by
the runner and will be skipped, so this is ~2h wall-clock in parallel, not a re-run.

## Deviation record

The roster is now exp5760's 13 PLUS 7. This is a further departure from the frozen exp6021
protocol, which is already recorded as running arms concurrently. Both deviations belong in
the artifact.
