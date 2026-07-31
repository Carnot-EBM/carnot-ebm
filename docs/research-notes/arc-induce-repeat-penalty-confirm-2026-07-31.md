# `repeat_penalty` triples the rate of valid engines and halves the cost — and does not make them better

**2026-07-31 · Phase 1 (confirm) of the induce-reliability investigation · 36 paired attempts, 6 games, scored OUT-OF-SAMPLE**

Upstream: `docs/research-notes/arc-induce-completion-budget-2026-07-31.md` (the budget is not the
lever; `repeat_penalty` is, at n=3 on one game, in-sample) and
`docs/research-notes/arc-engine-static-validation-2026-07-31.md` (the shipped path accepts broken
code 13 times in 15; a second ask of ANY kind rescues some of it, repair-text vs neutral at
p = 1.000). Artifacts: `results/arc_induce_confirm_20260731/`.

## Verdict

**CONFIRMED — for engine VALIDITY and COST. NOT for engine QUALITY.**

The pre-registered gate was "the funnel holds at ≥ 3 of 6 games with quality that is not junk, and
the treatment is not worse on the paired attempts." Both hold. But the effect that is large and
statistically real is on whether the generator emits *working code*, and on what that costs. The
effect on whether the working code *models the game* is null at this sample size, and the funnel it
moves is 2 of 6 games → 3 of 6, not the 1 → 3 the earlier in-sample evidence suggested.

| | CONTROL (shipped) | TREATMENT round 0 (`repeat_penalty` only) | TREATMENT final (+ plain re-ask) |
|---|---|---|---|
| attempts producing a USABLE engine | 13 / 36 | 20 / 36 | **22 / 36** |
| attempts clearing the STRICT out-of-sample quality bar | 6 / 36 | 7 / 36 | **7 / 36** |
| games with ≥1 usable engine | 5 / 6 | 5 / 6 | **6 / 6** |
| games with ≥1 strict-quality engine | 2 / 6 | 3 / 6 | **3 / 6** |
| wall clock per attempt | 100.3 s | **47.2 s** | 65.2 s |
| calls hitting the 4096-token cap | 20 / 36 | **2 / 36** | 5 / 45 |

Attempt-matched sign tests (same game, same seed, same temperature, same server process):

| comparison | treatment-only | control-only | p (two-sided) |
|---|---|---|---|
| **usable**, final vs control | **13** | 4 | **0.049** |
| usable, round 0 vs control | 11 | 4 | 0.119 |
| usable, final vs round 0 | 2 | 0 | 0.5 |
| **strict quality**, final vs control | 3 | 2 | **1.000** |

Minimum reachable two-sided p at the 17 discordant pairs the usable channel produced:
1.5 × 10⁻⁵. The quality channel produced only 5 discordant pairs, whose best possible two-sided p
is 0.0625 — so **the quality comparison could not have reached significance at this n even if
every one of those pairs had gone the same way.** That is a statement about the design, and it is
made here rather than offered afterwards as an excuse for a null.

## The measurement this run exists to make

Every previous scoring of this induce path was IN-SAMPLE, and
`results/arc_engine_validation_20260731/gate_scores.json` says so in its own payload. That is what
made the `cell_recall 0.947` uncitable: all six of ft09's changing transitions had been rendered
into the prompt the engine was graded against.

Building the out-of-sample split turned up a structural fact nobody was counting.
`arc_llm_reinduction._proposal_prefix` drops a `round(n/3)` suffix before `induce()` is called —
that is the holdout the codebase knows about. But `_transitions_block` then renders only
`changed[:6] + noop[:2]`, **at most 8 rows of 25**. The other 17 were passed to `induce()` and
never shown, which makes them exactly as unseen by the model as a suffix row. So the real held-out
set is 14–19 rows per game, not the ~8 a suffix-only split would give.

The split is proven, not asserted. `split.py` derives `shown` by replicating the selection rule,
then checks every row against the prompt TEXT by a different computation, and asserts that no
held-out row's rendered delta line occurs in the prompt. A row whose line is ambiguous — a
duplicate of a shown row — is DROPPED from held-out rather than scored as unseen (0–3 rows per
game). No held-out row informs any decision the pipeline makes: the treatment's dry run, defect
check and non-inert check see only `shown`, and the held-out rows are opened for the first time by
the scorer.

| game | shown | held out | held-out changing | held-out no-op | can grade change? |
|---|---|---|---|---|---|
| ft09 | 8 | 16 | 0 | 16 | no |
| tu93 | 6 | 19 | 19 | 0 | yes |
| tn36 | 6 | 17 | 17 | 0 | yes |
| lp85 | 4 | 18 | 0 | 18 | no |
| sc25 | 8 | 14 | 1 | 13 | marginal |
| vc33 | 1 | 0 | 0 | 0 | **nothing held out** |

Two of these deserve their own sentence. **vc33's real induce prompt carries ONE transition** — at
every explore budget tried — so there is nothing to hold out and its quality is unscoreable, full
stop. **ft09 and lp85 hold out only no-ops**, because all 6 resp. 2 of their changing transitions
are shown. That is not a worthless holdout: both are click games where 76% and 92% of observed
clicks are inert, so the mechanic IS target discrimination and the held-out question is whether
the engine fires its induced rule at clicks the game ignores. What it cannot do is separate a
correct engine from the identity function, so `heldout_n_noop_hallucinated` is never reported
without `engine_changes_anything` beside it.

## What the treatment actually fixes

The shipped path's failure is a decode-level repetition loop: the model reaches an impasse, emits
the same comment line until the budget is gone, and never writes a `return`. The defect census
across 36 attempts per arm shows the loop being broken rather than merely survived:

| defect | CONTROL | TREATMENT round 0 |
|---|---|---|
| `missing_return` | 13 | **2** |
| `engine_returned_none` | 12 | **2** |
| `syntax_error` | 9 | 6 |
| `engine_raised` | 2 | 1 |
| **attempts carrying any defect** | **22 / 36** | **9 / 36** |

`missing_return` falls 13 → 2 and the token-cap hit rate falls 20/36 → 2/36. The model stops
rambling and finishes the function. This is the same mechanism Phase 1 saw at n=3 on one game,
now at n=36 across six, and it is why the treatment is *cheaper* despite sometimes making a second
call: 47.2 s per attempt against 100.3 s, and 65.2 s even when the re-ask is charged to it.

**The shipped path accepted a mechanically defective candidate in 22 of 36 attempts** — Phase 2's
13-of-15 finding, reproduced at larger n on an independent set of samples.

`repeat_penalty` carries nearly all of the gain. Of the treatment's 13 paired wins on the usable
channel, 11 belong to round 0 alone. The re-ask fired 9 times and produced a usable engine twice,
worth 2 further paired wins against 0 losses (p = 0.5) — real but small, and consistent with
Phase 2's finding that a second ask of any kind is worth something and its content is worth
nothing.

## Where this stops, and it stops hard

**Usable is not good, and out-of-sample says so much more sharply than in-sample did.**

ft09 is the clearest case. The treatment produces a usable engine in **6 of 6** attempts — and in
**5 of those 6** the engine hallucinates a change on 12 to 16 of the 16 unseen inert clicks. It
learned that clicks do something; it did not learn where they have to land. Exactly one attempt
(`b3003_a2`) leaves all 16 alone. A funnel counted at `usable` would score ft09 6/6; the honest
count is 1/6.

**On 2 of 6 games no arm ever produces a good engine.** Taking the best over *every* scored
completion — all 36 attempts, all three arms, whether or not it was usable — tu93's high-water
mark is **0 of 19** held-out changing transitions correct and sc25's is 0 of 1. Not once, by
anything. The per-game strict-quality attempt counts are:

| game | CONTROL | TREATMENT round 0 | TREATMENT final |
|---|---|---|---|
| ft09 | 0 / 6 | 1 / 6 | 1 / 6 |
| lp85 | 2 / 6 | 2 / 6 | 2 / 6 |
| sc25 | 0 / 6 | 0 / 6 | 0 / 6 |
| tn36 | 4 / 6 | 4 / 6 | 4 / 6 |
| tu93 | 0 / 6 | 0 / 6 | 0 / 6 |
| vc33 | unscoreable | unscoreable | unscoreable |

The entire quality gain of the treatment is ft09's single attempt. Everything else is a tie.

## My own quality bar admitted junk, and that is the most useful thing here

The bar this run pre-registered was `heldout_cell_recall > 0.5`. It fails, on this run's own data.

**tu93's best CONTROL engine scores `heldout_cell_recall 0.537` — clearing the line — with 0 of 19
held-out changing transitions correct and 82 invented cells.** `cell_recall` is per-cell overlap,
so an engine that smears a roughly-right-sized change over roughly-right places clears a 50%
overlap line while never once reproducing a transition. It is the same trap as `usable`, one level
up, and it caught me the same way.

The strict bar reported throughout this note is not gameable that way: **at least one held-out
changing transition predicted EXACTLY**, falling back to zero hallucinated no-ops where the
held-out set has no changing rows. Under the recall bar the funnel reads 3/6 for both arms and the
treatment looks like it changed nothing; under the strict bar it reads 2/6 → 3/6 and tu93's junk
engine is correctly excluded from both. Both numbers are in
`results/arc_induce_confirm_20260731/confirm_scored.json`; the recall bar is retained precisely
because its failure is a result.

## What is NOT claimed

**Nothing is wired in.** `arc_executable_world_model.py` is untouched, and so is every threshold —
`min_heldout_accuracy`, `degenerate_goal_predicate`, `goal_predicate_true_at_root`. This run
measures; it does not ship. Editing the induce path marks 7 registered artifacts stale, one of
them requiring a live capture that would overwrite published numbers, and the honest price of
admission for that is a proven funnel gain.

**The funnel gain is proven for validity and unproven for quality**, which is a weaker warrant than
"the funnel moved 1/5 → 3/5" suggested. Three specific reasons to hold the claim narrow:

1. The **control is much better here than Phase 2's arm A** (13/36 usable against 1/15). Part of
   that is vc33, which is new and easy; part is two seed bases rather than one; and part is
   methodological — this run's dry run sees only the 8 SHOWN rows (to keep the held-out set clean),
   while Phase 2's saw all 25 and therefore had more chances to catch a raise. The two `usable`
   rates are **not** directly comparable, and the comparison that is valid here is the internal
   one against this run's own control.
2. `usable`-any-of-6-attempts flatters both arms relative to the shipped 3-try ladder.
3. Two of six games are unscoreable or near-unscoreable for change quality out-of-sample, so the
   quality funnel is being counted over an effective 4 games, not 6.

**The 60-second `live_llm_inference` floor is met by every call** and the generator was proven, not
assumed: CUDA build read from `/proc/<pid>/exe`, 21416 MiB resident per card on distinct bus IDs,
`n_ctx` 32768 read back from the server's own `/props`, `gemma-4-31B-it-Q4_K_M`, non-default ports
8951/8952, one server process per GPU shared by both arms and all three of its games — because the
sampler seed does not reach across server instances and arms split over two processes would be
confounded by sampler variance alone.

## What the next measurement should be

The one that follows from this result rather than from the plan: **wiring `repeat_penalty` into the
live induce path is justified on validity and cost alone** — 22/36 vs 13/36 usable at p = 0.049,
at 47% of the wall clock and 10% of the token-cap hits — and needs no quality claim to carry it.
The plain re-ask is a smaller, cheaper add-on (2 wins, 0 losses, 9 firings) and can ship or not on
its own merits.

What must not be claimed from this run is that either fixes induction. Two of six games get
nothing usable-and-correct from any arm at any budget, and the game where the treatment wins most
loudly on `usable` (ft09, 6/6) is right on 1 of 6. The bottleneck those numbers point at is not the
sampler and not the retry — it is that the model is not learning the *target-selection* half of a
click mechanic from six shown examples. That is a prompt/evidence problem, and it is where the next
lever should be looked for.
