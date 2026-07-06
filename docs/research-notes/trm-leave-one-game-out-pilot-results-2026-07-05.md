# TRM leave-one-game-out pilot: results, honestly reported (2026-07-05)

**Provenance:** operator authorized starting ARC-specific outer-loop work directly. This is the
falsifiable first pilot proposed in
`docs/research-notes/trm-arc-action-sequence-generator-2026-07-04.md`, run against the newly-fixed
human replay corpus (`docs/research-notes/human-replay-corpus-staging-bug-and-opportunity-2026-07-04.md`).
The result is genuinely mixed and reported here without smoothing over the caveats, per this
project's own Adversarial Artifact Verification discipline.

## Scope, stated honestly up front

This is a deliberately simplified pilot, not the full Stage-1 build from
`docs/research-notes/trm-generator-hidden-game-plan-2026-07-04.md`. It is a standalone
reimplementation of the TRM recursive-refinement *idea* (`z <- block(x, z)`, iterated a fixed 6
times, deep supervision at every step) — not nano-trm's own LightningModule/Hydra pipeline, which
carries considerably more machinery (adaptive ACT halting, puzzle embeddings) than is needed to test
the one question this pilot targets. The task is single-step action-*type* classification (predict
which of ~8 discrete action ids a human took, given only the current frame) — not full action-
sequence refinement, which remains the eventual target. Held out: `ft09` (1,351 rows), trained on the
other 24 games (164,191 rows).

## The result

| Arm | Train accuracy | Held-out (`ft09`) accuracy |
|---|---|---|
| Majority-class baseline | — | **0.7787** |
| Non-recursive baseline (matched params, single forward pass) | 0.5494 | 0.4626 |
| Recursive refiner (6 fixed steps, deep supervision) | 0.5385 | **0.6151** |

## Reading this honestly — two findings, not one clean verdict

**Finding 1 (the real, decision-relevant signal): recursive refinement clearly beats the matched
non-recursive baseline on the held-out game — +15.3 percentage points (0.6151 vs 0.4626) — despite
near-identical, even slightly *worse*, training accuracy (0.5385 vs 0.5494).** A model that fits the
training data about the same but generalizes to an unseen game substantially better is exactly the
kind of signal worth taking seriously — it is not explained by recursive refinement simply having
more capacity to memorize, since it didn't out-fit the training set at all.

**Finding 2 (the caveat that prevents calling this a clean win): neither trained model beats the
trivial majority-class baseline (0.7787) on the held-out game.** This pilot, as scoped, does not yet
demonstrate practical value over "always guess the most common action" — only a relative
architectural advantage between two comparably-sized trained models, both of which underperform a
baseline that requires no learning at all.

## A broader methodological finding that changes how to read Finding 2

Checked directly whether `ft09`'s class imbalance was an unlucky choice of held-out game. It was not
— it is actually one of the *less* imbalanced games in the corpus. A full per-game scan of majority-
action fraction:

| Game | Majority fraction | Game | Majority fraction |
|---|---|---|---|
| cn04 | 0.214 | dc22 | 0.349 |
| m0r0 | 0.211 | lf52 | 0.393 |
| ar25 | 0.254 | sc25 | 0.385 |
| cd82 | 0.260 | bp35 | 0.465 |
| sp80 | 0.246 | sb26 | 0.906 |
| ka59 | 0.278 | su15 | 0.947 |
| g50t | 0.287 | s5i5 | 0.988 |
| tu93 | 0.286 | lp85 | 0.985 |
| ls20 | 0.302 | r11l | 0.990 |
| sk48 | 0.308 | tn36 | 0.992 |
| tr87 | 0.293 | vc33 | 0.994 |
| re86 | 0.267 | **ft09** | **0.779** |
| wa30 | 0.270 | | |

Several games (`vc33`, `tn36`, `r11l`, `lp85`, `s5i5`, `su15`) show 94-99% single-action dominance —
far more extreme than `ft09`. This means **single-frame-to-action-type classification is a
structurally low-headroom task framing across much of this corpus**, not an artifact of which game
happened to be held out. Human action choice in most of these games is overwhelmingly repetitive
(likely dominated by a single navigation/movement action), so a static single frame carries limited
information about which of the rarer actions comes next — the missing signal is very likely *recent
action history / current intent*, which this pilot's framing deliberately excluded (single frame
only, no context window).

## What this means for the next step — not a verdict, a redesign

This pilot is **inconclusive on the core hypothesis** (does TRM's recursive mechanism add genuine
value for ARC-AGI-3 generation), not a clean negative. The class-imbalance finding above means the
task framing itself needs more headroom before a stronger conclusion can be drawn. Two candidate
fixes, either of which is a more informative next test than re-running this same framing on a
different held-out game:

1. **Condition on recent action history**, not just the current static frame — a short window of the
   last N actions/frames, which is far more likely to carry the signal needed to predict a rare
   action correctly. This is a small, cheap change to the existing pilot script's representation.
2. **Move toward the original full-sequence-refinement framing** — predicting or refining a whole
   candidate action sequence (matching `GAP-ARC-TRM-TRAINED-ON-ARC`'s own framing) rather than a
   single next-action classification, which inherently carries more context per training example.

Either is a smaller, cheaper next step than committing to the full Stage-1 build from the staged
plan — and per that plan's own gating logic, Stage 1 should not be treated as validated until one of
these redesigns produces a result that clears the trivial baseline, not just beats a matched
non-recursive arm.

## v2 update (2026-07-05, same day): history conditioning run — a direct contradiction, not a confirmation

Ran the recommended next step (condition on the last 8 action ids, same held-out game `ft09`, same
two architectures, added a "repeat-last-action" heuristic baseline). Reporting this without spin,
per the same discipline as the rest of this note — **the result contradicts, rather than confirms,
Finding 1 above.**

| Arm | Train accuracy | Held-out (`ft09`) accuracy |
|---|---|---|
| Majority-class baseline | — | 0.7787 |
| Repeat-last-action heuristic | — | 0.7651 |
| Non-recursive baseline (+ history) | 0.7026 | **0.7757** |
| Recursive refiner (+ history) | 0.6998 | **0.6107** |

History conditioning helped a lot in absolute terms — the non-recursive baseline jumped from 0.4626
(v1, frame-only) to 0.7757, confirming the v1 hypothesis that recent action history was the missing
signal. But **the recursive refiner now clearly *underperforms* the non-recursive baseline** (0.6107
vs 0.7757, a 16.5pp gap in the wrong direction), despite near-identical training accuracy — the exact
opposite ranking from v1, where recursion won by 15.3pp under nearly the same training-accuracy
parity.

**This reversal is the most important finding of the two runs combined, not either run in isolation.**
A single seed per architecture per condition is not enough to distinguish "recursion has an
inconsistent, condition-dependent effect" from "this quick, hand-rolled pilot setup is simply too
noisy/underpowered to measure the effect reliably in either direction." Both pilots used a single
fixed seed, three epochs, and a standalone reimplementation far simpler than nano-trm's own validated
training regime (no ACT halting, no tuned hyperparameters per architecture) — a plausible, honest
explanation for a 30-point swing in the recursive-vs-non-recursive ranking between two closely
related conditions.

**Also notable**: once history is added, the non-recursive baseline (0.7757) sits almost exactly at
the majority-class baseline (0.7787) — the task became easy enough for a simple architecture to
nearly match trivial guessing, which further narrows the headroom available for any architecture,
recursive or not, to demonstrate a real advantage.

**Revised recommendation.** Neither pilot should be read as evidence for or against TRM-style
recursion generalizing better on this task. Before drawing any conclusion, this needs: multiple
random seeds per arm (to separate signal from noise), and/or a properly-tuned training regime per
architecture rather than one shared quick-and-dirty config. Absent that, the honest state of Stage 1
is: **inconclusive, with contradictory single-run signals in both directions** — not a lead worth
building on without a more rigorous follow-up test.

## What this note is NOT claiming

- Not claiming TRM-style recursive refinement is proven to generalize to ARC-AGI-3. Finding 1 is a
  real, encouraging directional signal; Finding 2 and the class-imbalance finding mean it is not yet
  decision-grade.
- Not claiming the task framing is a dead end — the class-imbalance finding points at a specific,
  cheap fix (condition on history) rather than abandoning the approach.
- Not a claim about full action-sequence refinement, which this pilot did not test.

## Cross-references

- `docs/research-notes/trm-arc-action-sequence-generator-2026-07-04.md` — the note that proposed this
  pilot
- `docs/research-notes/trm-generator-hidden-game-plan-2026-07-04.md` — the staged plan this pilot's
  Stage 1 gate belongs to
- `docs/research-notes/human-replay-corpus-staging-bug-and-opportunity-2026-07-04.md` — the corpus
  this pilot trained on, fixed earlier the same day
- CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor" — the discipline this note
  follows in reporting both findings rather than only the encouraging one
