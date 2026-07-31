# `repeat_penalty` raises valid engines from 13 of 36 to 22 of 36 and halves the cost — and does not make them better

**2026-07-31 · Phase 1 (confirm) of the induce-reliability investigation · 36 paired attempts, 6 games, scored OUT-OF-SAMPLE**

> **CORRECTIONS 2026-07-31 (adversarial review of this note). Read these before the body — two of
> them change what the headline number means. The original text below is left unedited except for
> the H1, per never-prune.**
>
> **C1 — the H1 overstated its own evidence, and the commit subject still does.** This note was
> titled *"`repeat_penalty` triples the rate of valid engines"* and committed under that subject
> (`d2278313c`). **No measured rate triples.** Usable 13/36 → 22/36 is **1.69×**; penalty-alone
> 13 → 20 is **1.54×**; the most generous framing available, defect-FREE attempts 14/36 → 29/36,
> is **2.07×**. The body always stated 13/36 → 22/36 correctly, so the error was confined to the
> most-propagated string — off by ~1.8× against the number it sits next to. The H1 above is
> amended. **The commit subject is immutable and still reads "triples"**; that is why this
> correction is recorded here and in `ops/changelog.md`, so the git log is not the only surviving
> headline.
>
> **C2 — the p = 0.049 is pseudo-replicated, and the clustered inference is p = 0.125.** The 36
> pairs are 6 games × 6 resamples of the SAME induce prompt, varying only seed and temperature,
> but the sign test treated all 17 discordant pairs as independent. Resamples of one prompt are
> not independent draws from the population the claim generalizes to, which is **new games**.
> Clustering at the game level — collapsing each game to its net win/loss — gives **4 games
> better, 0 worse, 2 tied: p = 0.125**. A whole-game sign-flip permutation test on the net
> agrees exactly (observed net +9, p = 0.1250, 8 of 64). **Report p = 0.125 as the primary
> inference and 0.049 as the within-prompt figure.** Both are now in
> `confirm_scored.json` (`clustered_tests` vs `paired_tests`).
>
> This was the only sub-0.05 number in the run, and it is what opened the Phase 2 gate. Two things
> keep the wiring decision standing anyway: the direction is consistent — **0 of 6 games got
> worse** — and the cost result (47.2 s against 100.3 s per attempt, token-cap hits 20/36 → 2/36)
> is a large paired effect that needs no significance test at all. **Power bound, stated because
> it should have been stated first:** with 6 games the smallest reachable clustered two-sided p is
> 2/2⁶ = **0.031**, so significance required ALL SIX games to improve. Four did.
>
> **C3 — restricted to games where quality can actually be graded, quality did not move at all.**
> The body discloses both components — the strict funnel moved 2/6 games → 3/6, and separately
> ft09's held-out set contains no changing row — but never computes the conjunction, and the
> conjunction is the result. On the three games whose held-out set contains a CHANGING transition
> (sc25, tn36, tu93 — the only ones where out-of-sample change prediction can be graded at all),
> the strict counts are **4 of 36 control against 4 of 36 treatment: identical**, 4 of 18 attempts
> in each arm. The entire strict-funnel gain is **ft09**, which has zero held-out changing rows and
> therefore passes only on the weaker zero-hallucinated-no-ops fallback. On **tn36** — the one game
> with a substantial gradable held-out set, 17 changing rows — both arms are 4/6 and each arm's
> best engine is the same **17/17 exact**. Stated plainly: **the treatment produced no better
> engine on any game where out-of-sample change prediction could actually be graded.** The
> clustered strict test makes the same point from the other side: per-game net is ft09 +1 and
> zero everywhere else.
>
> **C4 — neither component is individually significant, at either clustering level.** The effect
> is attributed to `repeat_penalty` throughout, but penalty-ALONE against control is **p = 0.1185**
> within-prompt (**0.625** clustered, 3 games better / 1 worse) and the re-ask alone is
> **p = 0.5**. Only the COMPOUND arm reaches 0.049 / 0.125. Since both were wired in Phase 2, the
> effect should be attributed to the compound, and all three tests quoted together wherever it is:
> **compound p = 0.049 within-prompt / 0.125 clustered; penalty-alone p = 0.1185 / 0.625;
> re-ask-alone p = 0.5.**
>
> **C5 — the significant metric is IN-SAMPLE and mechanical; the out-of-sample split governs the
> null.** `usable` = accepted AND no static-validation defect AND `engine_changes_anything` on the
> ≤ 8 **SHOWN** rows (`confirm_ab.py:308`). The held-out split — the entire point of this phase —
> is used only by the QUALITY channel, which is null. So the subtitle's "scored OUT-OF-SAMPLE" sits
> directly beside the one number that never needed the split: **the validity and cost result would
> be unchanged if no split had been built.** What the split bought is the right to say the quality
> null is a real null rather than an in-sample artifact, and to retire the uncitable in-sample
> `cell_recall 0.947`.
>
> **C6 — 38% of the validity gain is one game, and it is the game with the weakest quality
> evidence.** Per-game win/loss on `usable` (treatment-only wins − control-only wins):
> **ft09 5–0, vc33 2–0, tn36 2–1, tu93 2–2, lp85 1–1, sc25 1–0.** Five of the 13 paired wins are
> ft09 — which is also where `usable` most clearly fails to mean good (5 of its 6 usable treatment
> engines hallucinate a change on 12–16 of the 16 unseen inert clicks) and where the held-out set
> cannot grade a change at all. Now in `confirm_scored.json:usable_per_game_decomposition`.
>
> **C7 — this run scored the WRONG induce call site, and it changes what ft09's unscoreability
> means.** `confirm_ab.py:69` defaults `CONFIRM_CALL_INDEX=2`. The later best-of-N capture
> (commit `52aee4a2a`) established that `_induce_and_plan` makes **two** induce calls that are
> different call sites, not retries: call 1 from `execute_bounded_llm_reinduction`
> (`arc_competition_agent.py:5770`, the stall branch) and call 2 from the plain single-shot path
> 120 lines further down (`:5889`). **All three gates live on call 1, and `_proposal_prefix` is
> applied only to call 1.** The consequence for this note: **ft09 has 4 held-out CHANGING rows at
> call 1 versus 0 at call 2.** The section below blames the GAME ("hold out only no-ops, because
> all 6 … changing transitions are shown"); that is an artifact of the call site, not a property
> of ft09. The split, the held-out table, and the QUALITY channel are all **call-2-specific** and
> scoped accordingly. The VALIDITY and COST verdict is not affected — it is measured on the shown
> rows and on the wall clock, neither of which depends on which call the prompt came from.
>
> **C8 — the shared-server-process property was ASSERTED for these pairs, not measured.**
> `confirm_ab.py` captured `server_pid` once per GPU worker; per-attempt rows carried the seed but
> no PID. Phase 3 later PROVED the failure mode is real — a server WAS replaced mid-session there
> (1532116 → 1562595) and its `same_server_process` guard excluded a game rather than score a
> confounded pair. This matters because the sampler seed does not reach across server instances, so
> a pair straddling a restart is confounded by sampler variance alone. An equivalent replacement
> during THIS run would have been silent. A per-call PID read and the matching scorer guard are
> retrofitted (2026-07-31); the pairs already on disk are **unverified in this respect** and are
> not retro-fixable, since the PIDs were never written down. `confirm_scored.json` now records
> this as `paired_server_process_check.status = UNVERIFIED` rather than reporting a clean check.
>
> **C9 — the measured configuration is not the shipped one.** All measurement pinned
> `CARNOT_ARC_INDUCE_N_CTX=32768` (`confirm_ab.py:86`), while the shipped `_default_induce_n_ctx()`
> still returns **81920** — which this note's own witness paragraph says does not fit a 24 GiB card.
> The COST half of the claim (token-cap hits 20/36 → 2/36) is completion-budget sensitive and was
> measured at a context the live default does not use. Pre-existing, but the Phase 2 wiring shipped
> into that path without noting the gap. **The validity and cost numbers are conditioned on
> n_ctx = 32768** until the 81920 default is fixed or the treatment is re-measured at it. Recorded
> in `confirm_scored.json:measured_configuration`.
>
> **What none of these corrections touch.** The largest and least ambiguous finding in the run:
> **the shipped path accepted a mechanically defective candidate in 22 of 36 attempts.** That is a
> single-arm census, not a comparison — no clustering, split, call site or server question bears on
> it. It is also the finding that reproduced Phase 2's 13-of-15 at larger n on independent samples.

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

Attempt-matched sign tests (same game, same seed, same temperature, same server process — but see
**C8**: the shared-process property is asserted for these pairs, not measured):

| comparison | treatment-only | control-only | p WITHIN-PROMPT | **p CLUSTERED BY GAME** |
|---|---|---|---|---|
| **usable**, final vs control | **13** | 4 | 0.049 | **0.125** (4 games better, 0 worse, 2 tied) |
| usable, round 0 vs control | 11 | 4 | 0.119 | 0.625 (3 better, 1 worse) |
| usable, final vs round 0 | 2 | 0 | 0.5 | — |
| **strict quality**, final vs control | 3 | 2 | 1.000 | **1.000** (1 better, 0 worse) |

**The clustered column is the primary inference** (C2). The 36 pairs are 6 games × 6 resamples of
one prompt each; the within-prompt column answers "does the treatment beat control on a resample
of a prompt already seen", which is not the claim anyone wants. A whole-game sign-flip permutation
test on the net agrees with the clustered sign test exactly: observed net +9, p = 0.1250 (8 of 64).

Minimum reachable two-sided p at the 17 discordant pairs the usable channel produced:
**1.526 × 10⁻⁵** — the artifact used to report this as `0.0`, a value no test can attain, fixed
2026-07-31. Clustered, the floor is **2/2⁶ = 0.031**, reachable only if all six games move the same
way. The quality channel produced only 5 discordant pairs, whose best possible two-sided p
is 0.0625 — so **the quality comparison could not have reached significance at this n even if
every one of those pairs had gone the same way.** That is a statement about the design, and it is
made here rather than offered afterwards as an excuse for a null.

**Where the 13 wins come from** (C6), because the aggregate hides a concentration:

| game | treatment-only wins | control-only wins | net |
|---|---|---|---|
| ft09 | 5 | 0 | **+5** |
| vc33 | 2 | 0 | +2 |
| tn36 | 2 | 1 | +1 |
| sc25 | 1 | 0 | +1 |
| lp85 | 1 | 1 | 0 |
| tu93 | 2 | 2 | 0 |

Five of the 13 — 38% — are ft09, which is also the game whose held-out set cannot grade a change
and where 5 of 6 usable treatment engines hallucinate on 12–16 of 16 unseen inert clicks.

## The measurement this run exists to make

> **CORRECTION 2026-07-31 (C5) — the split governs the NULL, not the headline.** The two channels
> must not be read as one. `usable` — the channel carrying every significant number in this note —
> is computed on the ≤ 8 **SHOWN** rows (`confirm_ab.py:308`: accepted AND no static-validation
> defect AND `engine_changes_anything`). It is **in-sample and mechanical**, and it would be
> unchanged if no held-out split had ever been built. The out-of-sample split is used only by the
> QUALITY channel, which is null. So what the split actually bought is worth stating precisely: the
> right to call the quality null a real null rather than an in-sample artifact, and the retirement
> of the uncitable in-sample `cell_recall 0.947`. It did not underwrite the 13/36 → 22/36.

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

> **CORRECTION 2026-07-31 (C7) — this table is CALL-2-SPECIFIC, and ft09's row is an artifact of
> that.** `confirm_ab.py` defaults `CONFIRM_CALL_INDEX=2`. `_induce_and_plan` makes two induce
> calls that are different call sites, and `_proposal_prefix` is applied only to **call 1** — where
> **ft09 has 4 held-out CHANGING rows against 0 here**. The paragraph immediately below explains
> ft09's unscoreability as a property of the GAME ("all 6 of its changing transitions are shown");
> that explanation is wrong. It is a property of the call site this run happened to score. All
> three gates the live agent runs also live on call 1, so the QUALITY channel measured here is
> the weaker of the two available and should be re-run at call 1 before any quality claim. The
> VALIDITY and COST verdict is unaffected — it reads the shown rows and the wall clock, neither of
> which depends on the call site.

Two of these deserve their own sentence. **vc33's real induce prompt carries ONE transition** — at
every explore budget tried — so there is nothing to hold out and its quality is unscoreable, full
stop. **ft09 and lp85 hold out only no-ops**, because all 6 resp. 2 of their changing transitions
are shown (**at call 2 — see C7 above; at call 1 ft09 holds out 4 changing rows**). That is not a worthless holdout: both are click games where 76% and 92% of observed
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

**But neither component is individually significant, at either clustering level (C4).** Splitting
the compound apart costs it the only sub-0.05 number it had: penalty-alone against control is
**p = 0.1185** within-prompt and **0.625** clustered (3 games better, 1 worse); re-ask-alone is
**p = 0.5**. Only the compound reaches 0.049 / 0.125. "Carries nearly all of the gain" is a
statement about the DECOMPOSITION of the point estimate, not about either part clearing a bar on
its own — and since Phase 2 wired both, the effect belongs to the compound. Quote all three
together wherever it is attributed.

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

**The conjunction this table and the held-out table imply, computed (C3).** Drop the games whose
held-out set has no changing row — the ones where a strict pass is the weaker no-op fallback and
the identity function scores a perfect held-out accuracy — and read the change channel alone:

| | CONTROL | TREATMENT round 0 | TREATMENT final |
|---|---|---|---|
| strict-quality attempts on the 3 change-gradable games (sc25, tn36, tu93) | **4 / 18** | **4 / 18** | **4 / 18** |
| games passing among those 3 | 1 (tn36) | 1 (tn36) | 1 (tn36) |
| best engine on tn36 (17 held-out changing rows) | 17/17 exact | 17/17 exact | 17/17 exact |

Identical in every arm. **The treatment produced no better engine on any game where out-of-sample
change prediction could actually be graded.** The whole 2/6 → 3/6 strict-funnel movement is ft09,
on the fallback channel, on one attempt.

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

> **CORRECTION 2026-07-31 (C8) — "one server process per GPU shared by both arms" was ASSERTED,
> not measured.** The witness above records `server_pid` ONCE per GPU worker; the per-attempt rows
> carry the seed but no PID, so a mid-run server replacement would have been silent. Phase 3 later
> proved that failure mode is real: a server WAS replaced mid-session there (1532116 → 1562595),
> and its `same_server_process` guard excluded a game rather than score a confounded pair. A
> per-call PID read plus the matching scorer guard are retrofitted into `confirm_ab.py` /
> `score.py`; the 72 arm-pairs already on disk are recorded as **UNVERIFIED** in
> `confirm_scored.json:paired_server_process_check` and are not retro-fixable.
>
> **CORRECTION 2026-07-31 (C9) — the measured n_ctx is not the shipped one.** This run pinned
> `CARNOT_ARC_INDUCE_N_CTX=32768`. The shipped `_default_induce_n_ctx()` still returns **81920**,
> the value this very paragraph's setup notes does not fit a 24 GiB card. Since the cost claim is
> completion-budget sensitive, **the validity and cost numbers are conditioned on n_ctx = 32768**
> until the default is fixed or the treatment is re-measured at it.

## What the next measurement should be

The one that follows from this result rather than from the plan: **wiring `repeat_penalty` into the
live induce path is justified on validity and cost alone** — 22/36 vs 13/36 usable at p = 0.049
within-prompt / **0.125 clustered by game, 4 better and 0 worse** (C2), at 47% of the wall clock
and 10% of the token-cap hits — and needs no quality claim to carry it. The clustered p does not
clear 0.05 and could not have without all six games moving; what carries the decision is the
consistent direction plus a cost effect large enough not to need a test.
The plain re-ask is a smaller, cheaper add-on (2 wins, 0 losses, 9 firings) and can ship or not on
its own merits.

What must not be claimed from this run is that either fixes induction. Two of six games get
nothing usable-and-correct from any arm at any budget, and the game where the treatment wins most
loudly on `usable` (ft09, 6/6) is right on 1 of 6. The bottleneck those numbers point at is not the
sampler and not the retry — it is that the model is not learning the *target-selection* half of a
click mechanic from six shown examples. That is a prompt/evidence problem, and it is where the next
lever should be looked for.
