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

---

## AMENDMENT (2026-07-31, before the extension completed) — the exclusion reason was mischaracterised

Appended, not edited, per never-prune: the original text above is wrong in a way worth
keeping visible.

It said the 5 excluded games fail for "a property of the fixtures that is independent of
either arm's score". The independence half is correct and is what protects the comparison.
The "property of the fixtures" half is correct for only ONE of the five. Measured:

| game | failure | property of the GAME? |
|---|---|---|
| dc22 | `build_progress_window` returns None | YES -- the documented "cannot be solved to L1 offline" |
| ka59 | `ValueError: invalid literal for int(): 'C:1'` in `build_window` | NO -- unhandled action encoding (a coordinate click) |
| sc25 | `AttributeError: 'NoneType' has no attribute 'hand_verifier'` | NO -- no registered GameAdapter |
| tn36 | same | NO -- no registered GameAdapter |
| wa30 | same | NO -- no registered GameAdapter |

**What this does NOT change.** All five fail BEFORE any LLM is invoked, so the exclusion
cannot favour either arm. The 20-game extension remains unbiased and its result stands on
its own terms.

**What this DOES change.** The stopping rule. The original says "20 is the ceiling of this
corpus, so this is the last extension". That is false as written: 20 is the ceiling of the
CURRENT HARNESS. Registering three `GameAdapter`s and teaching `build_window` the `'C:1'`
action encoding would raise the ceiling to 24 games -- ~20% more games, and materially more
discordant pairs if the informative rate holds.

**Revised stopping rule.** 20 games remains the last extension *at current tooling*, and a
null there is still reported as a null. But it may NOT be described as "the corpus is
exhausted". The honest statement is: *4 of 25 public games are unreachable by this harness
for reasons unrelated to the models under test, and fixing that is a prerequisite to any
better-powered rerun.* Whether that fix is worth doing is a separate decision, on its own
merits, and must not be made by reading the 20-game result first -- that would reintroduce
exactly the outcome-dependent stopping this pre-registration exists to prevent.
