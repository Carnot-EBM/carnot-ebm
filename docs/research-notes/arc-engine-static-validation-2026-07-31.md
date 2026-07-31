# Four of the five ARC induction failures are broken code, and all four are mechanically detectable

**2026-07-31 · Phase 2 of the induce-reliability investigation**

Part 1 is offline (no GPU). **Part 2, at the bottom, is the live GPU measurement** and it is the
one that answers the funnel question — read it before acting on anything here.

Upstream: `docs/research-notes/arc-gate-rejection-audit-2026-07-30.md` (which failure is which),
`docs/research-notes/arc-induce-completion-budget-2026-07-31.md` (Phase 1: the budget is not the
lever). Artifacts: `results/arc_engine_validation_20260731/`.

## The question

The 2026-07-30 activation grid measured the LLM tier firing on 6 of 6 games and reaching the policy
on 1 of 6. The gate-rejection audit then established that every rejection was CORRECT, and that four
of the five failures were broken code rather than a wrong model of the game. Phase 2 asks whether
those four are detectable before the trust gate sees them, and at what false-positive cost.

## What was built

`python/carnot/agentic/arc_engine_static_validation.py` — a standalone module, spec'd as
REQ-ARC-WMTE-6052, with one check per observed failure:

| check | the failure it was written for |
|---|---|
| `missing_return_defects` (AST fall-through over `engine`) | ft09's live engine returns on `action != 6` and ends the `action == 6` path in a comment wall; tu93's two engines "fall off the end of `engine()` with no return" |
| `dry_run_defects` — `engine` arm | a generated engine that raises on an observed transition |
| `dry_run_defects` — `is_level_complete` arm | lp85's `UnboundLocalError: cannot access local variable 'cell'`, which is raised by the GOAL predicate, not the engine |
| `truncation_defect` (keys on llama-server's own `stop_type == "limit"`) | ft09 round 2's `HIT n_predict=4096 OUTPUT LIMIT` |
| `repair_prompt_block` | turns a raise into a re-induce carrying the exception text |

It is deliberately **not a gate**. It runs upstream of `WorldModelVerifier` and has no power to admit
anything; a candidate that passes every check still faces the unchanged trust gate. That separation
is the whole design, and the reason is in the next section.

## The result that constrains everything else

**Not one of these checks measures quality, and the corpus shows why they must not try.** Phase 1's
best-scoring completions — accepted by `generate()`, parsing, returning on every path, 19-of-25
held-out exact — were the identity function `return grid`, and 19 of ft09's 25 transitions are
no-ops, so "nothing ever changes" gets every one of them right. An identity engine clears every
check in this module trivially. `test_identity_engine_passes_every_check` pins that, and
`engine_changes_anything` is reported next to a clean report so that "no defects found" is never
read as "the engine is good".

## Measurements

### False-positive rate, on 439 real generated engines

Every `world_model*.py` in `results/` — the banked engines, the origin fixtures, the logo snapshot
and every per-experiment `e3_store` — scanned by the static check, and every flag cross-checked by
EXECUTING the engine and looking for a real `None` return.

| | |
|---|---|
| files scanned | 439 |
| flagged `missing_return` | 9 (2.05%) |
| flags confirmed by execution | **9 of 9** |
| flags NOT reproducible by execution | **0** |

The 9 are true positives, each returning `None` on 7–14 of the 14 probed (action, data) shapes. The
worst is `results/arc_wm_four_arm_20260727/e3_store/wm_A3_both/vc33/world_model.py`, which is the
ft09 failure mode exactly: the model echoed its own refactor prompt back as comments — including the
literal `REQUIRED OUTPUT STRUCTURE` instruction block and a JSON mismatch dump — and never wrote a
`return`.

Stated honestly: the execution cross-check uses a synthetic probe grid, so a `None` it fails to
reproduce would not prove the path unreachable. It did not have to arbitrate that, because it
reproduced all nine.

### Sensitivity, on 36 real live completions

Phase 1 captured 36 completions from the live generator against ft09's real induce prompts, each
with its recorded `stop_type`, `predicted_n` and raw text. Replaying them through the validator:

| | |
|---|---|
| completions replayed | 36 |
| hit the output cap | 24 |
| flagged truncated-before-required-symbols | 10 |
| **shipped `generate()` ACCEPTED, validator flags a defect** | **13** |
| **shipped `generate()` REJECTED, validator says retryable truncation** | **10** |

The 13 are engines that reached the trust gate as bad predictions when they were broken code. The 10
are completions discarded as bad models when they were incomplete observations.

Corroboration worth more than either number: Phase 1's harness contains an independently-written AST
return check. On all 28 rows where both ran, the two agree — **0 disagreements**.

### The dry run finds live defects in banked engines

Run against real offline transitions from `collect_transitions`, the dry run reports 7 raising
engines and 1 `None`-returning one across the five audited games. One of them is not a historical
artefact: **`results/arc_e3/tu93/world_model.py`, the banked engine, raises
`ValueError: could not broadcast input array from shape (2,) into shape (3,)` on `action=4`.** The
trust gate can only ever see that as "failed the transition"; the dry run says what failed and hands
back a sentence the model can act on.

### Tests

46 tests, mutation-proven: **14 of 14 mutations killed**, machine-readable record at
`results/arc_engine_validation_20260731/mutation_check.json` with the baseline green and the module
restored byte-identical. The mutations reverse specific design decisions, not arbitrary characters —
"an `if` without an `else` terminates" (blinds the ft09/tu93 detection), "drop the goal arm" (misses
lp85), "the repair prompt includes unrepairable defects" (spends the budget the retry needs), and
"a nested definition may shadow the top-level one" (a real latent bug found by self-review after the
first commit: `ast.walk` is breadth-first, so a bare "last definition wins" rule reliably picks a
NESTED `engine` whenever one exists, while `exec` binds only top-level ones. It is inert on the
corpus — no generated engine defines a nested `engine`, and the 439-file scan is unchanged by the
fix — but a helper's throwaway inner function would have decided the verdict for the real engine).

## What is NOT claimed

**The funnel has not moved, and this module does not by itself move it.** The shipped induce path is
unchanged by this work; the module is standalone. Even wired in, Phase 1 already measured that under
the shipped sampler EVERY ft09 engine-only attempt at 4096, 8192 and 16384 tokens was non-returning.
So on ft09 this check converts "accepted a `None`-returning engine" into "rejected after three
tries" — a correct diagnosis, not a working engine.

That is the honest shape of Part 1's finding: **the checks are proven; the repair is not.**
Whether telling the model "your `engine` returned None on action 6" produces a returning engine was
then measured live — see Part 2. Short version: a second ask rescues some attempts, but a
defect-naming ask is indistinguishable from a contentless one at p = 1.000.

## One limitation stated plainly

lp85's actual failing source no longer exists — the episode's engine store was a per-cell temp
directory, so unlike ft09 (whose engine was frozen into the Phase-1 capture) it was not preserved.
The `goal_raised` check is therefore proven against a **reconstruction** of the documented mechanism
(`UnboundLocalError` on an unbound `cell`, raised from `is_level_complete`, on the root grid) rather
than against the original file, and against 0 preserved instances in the 439-file corpus. The
corpus does confirm the check is not trigger-happy — 71 engines × 50 grids, zero goal defects
reported — but the sensitivity claim for this one arm rests on a reconstruction, not an artefact.

---

# Part 2 (live, GPU): the shipped path accepts broken code 13 times in 15 — and a second ask of ANY kind is what rescues it

Everything above is offline. This part is the live measurement the funnel question actually needs:
**does feeding a measured defect back produce a usable engine, and is any gain attributable to the
defect TEXT rather than to the retry itself?**

## Design: three arms, because two cannot answer it

A two-arm design (shipped vs repair) cannot separate "the repair TEXT works" from "ANY second ask
works" — the repair arm changes the prompt, and a changed prompt resamples. So:

| arm | what it does |
|---|---|
| **A shipped** | round 0 only — exactly what `generate()` does today |
| **B repair** | round 0, then re-ask with `repair_prompt_block` (names the defect, the exception text, and the head of the failing code) |
| **C control** | round 0, then re-ask with a NEUTRAL block of comparable length that says a previous attempt was unsatisfactory and names NOTHING about what went wrong |

B and C branch from the **same round 0**, generated once, and both re-asks reuse its seed and
temperature — so the arms differ only in the content of the second ask, and every attempt is a
matched pair.

Prompts are the live agent's own, captured LLM-off from the agent itself; ft09's are **byte-identical
(sha256 `78f829d86fa65938`) to the ones Phase 1 committed**, independently reproduced. 5 games × 3
attempts, gemma-4-31B-it Q4_K_M, CUDA build proven from `/proc/<pid>/exe`, 21416 MiB resident per
card, `n_ctx=32768`, non-default ports, one game per 3090.

`usable` = `generate()` would accept it **AND** no mechanical defect **AND** it changes the grid on
some observed transition. The last clause is load-bearing: Phase 1's best-scoring completions were
the identity function, which clears the first two trivially.

## The finding that is not close

| | attempts |
|---|---|
| total | 15 |
| **shipped `generate()` ACCEPTED a candidate carrying a mechanical defect** | **13** |
| shipped produced a usable engine | **1** |

The shipped path takes broken code in 13 of 15 attempts and therefore **never re-asks at all**. That
is the large, unambiguous result, and it is about arm A alone.

## The comparison that IS close — and my own tooling over-read it first

| arm | usable attempts | games |
|---|---|---|
| A shipped | 1 / 15 | tn36 |
| B repair | 3 / 13 | tu93, tn36, sc25 |
| C control | 2 / 13 | ft09 |

Raw 3-vs-2 looks like a repair win, and the first version of `build_ab_artifact.py` printed exactly
that: *"the DEFECT TEXT carries the value, not merely the second ask."* **That was an over-claim, and
it was in the measurement tooling rather than in a result, which is worse.** The comparison is
PAIRED — both arms branch from the same round 0 — so only the discordant pairs carry information:

> **5 discordant pairs: 3 repair-only, 2 control-only. Exact two-sided sign test p = 1.000.**

Indistinguishable from a coin. The verdict logic now computes the paired test and says so. Note also
that the two arms succeed on **disjoint games** — repair on tu93/tn36/sc25, control on ft09 — which
is itself the signature of resampling rather than comprehension.

## "Usable" is necessary and nowhere near sufficient

Scoring every completion with the **production `WorldModelVerifier`** shows two of the five usable
engines are degenerate:

| game | arm | cell_recall | changes correct | invented cells |
|---|---|---|---|---|
| tu93 | repair | 0.112 | 0 / 25 | **144** |
| ft09 | control | 0.316 | 0 / 6 | 0 |

tu93's clears the non-inert clause by changing the grid **wrongly**. On the stricter bar
`in_sample_cell_recall > 0.5`:

| arm | games reaching it |
|---|---|
| A shipped | tn36 |
| B repair | sc25, tn36 |
| C control | ft09 |

**Funnel, quality bar: 1 of 5 games → 3 of 5 across all arms** (ft09, sc25, tn36). tu93 and lp85 get
nothing good from any arm.

These numbers are **IN-SAMPLE** — every transition scored was in the induce prompt, while the live
gate grades a held-out suffix (`_proposal_prefix` keeps `round(n/3)` out). They separate an engine
that models something from one that models nothing; they are not gate outcomes.

## Two independent reproductions of the audit, from the generator side

* **ft09's best completion scores `cell_recall 0.947`, `noop_hallucination 0/19`** — matching the
  audited live ft09-`onb` record (`cell_recall 0.947`, 216 correct changed cells) essentially
  exactly. The same mechanic, rediscovered.
* **tn36's round 0 scores `accuracy 1.000, cell_recall 1.000, change_fidelity 1.000, 25/25`** — the
  audit's "run's best engine, held-out 8/8 exact", reproduced.
* **lp85's best repair scores `accuracy 0.920` with `cell_recall 0.000` and `0/2` changes correct** —
  the vacuous pass the audit flagged, live. It matches 23 of 25 transitions by matching the 23
  no-ops. `usable` correctly rejects it; a 0.5 legacy-accuracy bar would admit it.

## What this means for the fix

The value of the static check is that it makes the pipeline **re-ask where today it accepts** — 13
of 15 times. That is a real mechanism and it is worth wiring in. What the data does NOT support is
the richer claim that the defect text teaches the model anything: at n = 13 paired attempts it is
indistinguishable from a contentless "try again". Until that is settled, the cheaper re-ask is the
better default, and echoing the failed code back is the part to drop first — it costs prompt tokens
re-showing a repetition-loop model the text it is stuck repeating.

## What was NOT done, and why

**The validator is still not wired into the shipped induce path.** Editing
`arc_executable_world_model.py` marks 7 registered artifacts stale
(`experiment_6011/6012/6013/6021` plus three with no rebuild command), and the honest reason to pay
that price is a proven funnel gain. What is proven is a diagnosis (13 of 15) and an undecided repair
mechanism (p = 1.000). Wiring a re-ask into the scored path on that evidence would be a behaviour
change ahead of its warrant — the same call Phase 1 made about `repeat_penalty`, for the same reason.

The obvious next measurement is the cheap one this run points at: **arm A vs a plain re-ask with no
defect text at all**, at larger n, since that is the arm the data actually favours on cost.
