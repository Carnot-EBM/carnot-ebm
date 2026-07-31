# Four of the five ARC induction failures are broken code, and all four are mechanically detectable

**2026-07-31 · Phase 2 of the induce-reliability investigation · offline, no GPU**

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

42 tests, mutation-proven: **12 of 12 mutations killed**, machine-readable record at
`results/arc_engine_validation_20260731/mutation_check.json` with the baseline green and the module
restored byte-identical. The mutations reverse specific design decisions, not arbitrary characters —
"an `if` without an `else` terminates" (blinds the ft09/tu93 detection), "drop the goal arm" (misses
lp85), "the repair prompt includes unrepairable defects" (spends the budget the retry needs).

## What is NOT claimed

**The funnel has not moved, and this module does not by itself move it.** The shipped induce path is
unchanged by this work; the module is standalone. Even wired in, Phase 1 already measured that under
the shipped sampler EVERY ft09 engine-only attempt at 4096, 8192 and 16384 tokens was non-returning.
So on ft09 this check converts "accepted a `None`-returning engine" into "rejected after three
tries" — a correct diagnosis, not a working engine.

That is the honest shape of the finding: **the checks are proven; the repair is not.** Whether
telling the model "your `engine` returned None on action 6" produces a returning engine is a live
measurement that has not been made. It is the obvious next one, and Phase 1's `repeat_penalty`
result (3/3 non-degenerate engines against 0/3 shipped) is the other half of the same question.

## One limitation stated plainly

lp85's actual failing source no longer exists — the episode's engine store was a per-cell temp
directory, so unlike ft09 (whose engine was frozen into the Phase-1 capture) it was not preserved.
The `goal_raised` check is therefore proven against a **reconstruction** of the documented mechanism
(`UnboundLocalError` on an unbound `cell`, raised from `is_level_complete`, on the root grid) rather
than against the original file, and against 0 preserved instances in the 439-file corpus. The
corpus does confirm the check is not trigger-happy — 71 engines × 50 grids, zero goal defects
reported — but the sensitivity claim for this one arm rests on a reconstruction, not an artefact.
