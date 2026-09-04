# The context fix works, and what the model writes with the extra room is a lookup table

Written 2026-09-04 from the first two engines a live r11l run produced after
`CARNOT_ARC_INDUCE_N_CTX=98304`. Short version: raising the context window removed a real defect
and moved the bottleneck not at all, because the bottleneck was never the window.

## What was measured

One live run, `scripts/arc_leaderboard_eval.py --policy e3 --only r11l`, GPU 1, tools off,
`n_ctx=98304`, 5h43m elapsed at the time of writing. Two engines emitted, at 02:05:36Z (170 lines)
and 06:24:27Z (86 lines). Decode ran 31-38 tok/s throughout.

**The truncation is gone.** Before the fix the induce step failed with
`missing ('engine',) in output [TRUNCATED BY SHARED CONTEXT POOL: generated only 18431 of the
26800-token budget]` and recorded `chars_final: 0`. Both engines here are complete, parseable
Python. The fix did its job.

## Engine 1 — memorization, plainly

The file opens with a tuple of literal per-transition delta strings:

```python
_DELTA_STRS = (
    "r0c0:5x1 r34c7:5x1 r35c6:5x3 r36c5:5x5 r37c6:5x3 ...",
```

52 such coordinate literals. This is a transcription of the rows the model was shown.

## Engine 2 — the same thing, compressed, and that is the interesting part

Engine 2 contains **zero** literal delta strings. It looks like better code. It defines a decoder:

```python
def _expand(runs):
    for r, c0, pairs in runs:
        ...
```

and feeds it run-length-encoded tuples, under a comment that says the quiet part out loud —
`# Phase deltas observed along the diagonal "track" (absolute cell assignments).`

Its step function is where the claim settles:

```python
if (br, bc) == (47, 17) and g[36, 7] == 15 and g[34, 7] == 0:
    for r, c, v in _P1:  g[r, c] = v
elif (br, bc) == (53, 22) and g[47, 17] == 15:
    for r, c, v in _P2:  g[r, c] = v
elif (br, bc) == (56, 24) and g[53, 22] == 15:
    for r, c, v in _P3:  g[r, c] = v
```

Three observed creature positions, each mapped to a stored block of absolute cell assignments. A
fourth position produces nothing. The sixteen control-flow statements are the decoder's loops and
this case table — not a rule that generates the dynamics.

**So the lesson is narrower and sharper than "the model memorizes".** Between engine 1 and engine 2
the model got BETTER at encoding and no better at inducing. Compression is not induction, and
control flow is not a rule. An engine that RLE-encodes its observations and dispatches on observed
coordinates is a lookup table that reads like a program, which is harder to spot than a wall of
string literals and exactly as useless on an unseen transition.

## The goal predicate is empty, by the model's own admission

```python
def is_level_complete(grid):
    # No win state was ever observed in the data (all transitions stay level 0->0).
```

The model states that it never saw a win and therefore cannot express the win condition. That is
the broken-goal-predicate finding (14 of 21 stored predicates never fire on a real win) reproducing
live, in the model's own words, and it is independent of context size.

## Why this composes with what was already known

The 2026-08-27 fine-read measured the memorization class at 100% prefix accuracy against 50-75%
held-out — and measured it at prompts around 6,589 tokens, where there was no context pressure at
all. The two observations agree: the failure exists far below any wall, and removing the wall does
not touch it.

## Limits, stated because this is two engines

n=2, one game, one run, one model. **Neither engine has been scored.** `exact_acc` and `cell_recall`
against held-out transitions are the numbers that would make this a result rather than a reading,
and `scripts/arc_e3_induced_model_quality.py` produces them offline. Milestone 610's
`exp6968-arc-post-refit-induction-audit` is scheduled to do exactly that scoring; this note is the
generation half. Treat the structural reading here as strong evidence and not as a measurement.

Nothing here establishes that induction is impossible at 27B, only that two attempts at
`n_ctx=98304` produced lookup tables. A larger n, other games, and the held-out scores are what
would turn this into a claim about the model rather than about these two files.

---

## The run finished, and the channel counters give the measurement

`results/arc_leaderboard_eval_runs/r11l-2491317.json`, 24,998s (6h57m), completed normally. It is
the first eval artifact ever to carry `generator_channels` (REQ-ARC-WMTE-6641, wired hours before
this run started).

| | pre-fix (ls20/wa30, 2026-09-02) | post-fix (this run) |
|---|---|---|
| `chat_completions` | 24 | 13 |
| `reasoning_only` | 24 — **100%** | 3 — **23%** |
| `chars_final` | **0** | **37,872** |
| engines emitted | 0 | 2 |
| `both_channels_empty` | 0 | 0 |
| `request_timeouts` | — | 0 |

**The truncation fix is real and measured.** Reasoning-only completions fell from every single call
to under a quarter of them, and the final channel went from empty to 37,872 characters. The server
answered every time in both regimes, so this is not a connectivity difference — it is the model
being able to finish.

**And 98.5% of what it generates is still reasoning.** 2,467,130 characters of reasoning against
37,872 of final output. The cost of an induce is dominated by thinking that is discarded.

## The uncomfortable number

| | levels | actions | efficiency |
|---|---|---|---|
| this morning, arm-OFF, budget 2500 | **2** | 2121 | 0.689 |
| this run, post-fix, budget 20000 | **1** | 2077 | 0.0035 |

The induce tier worked better and the agent banked **fewer** levels.

**Do not read that as a regression yet.** The budgets differ by 8x, the runs are single samples of a
stochastic search, and efficiency is scored against level-1 human baselines so a 1-level run and a
2-level run are not on the same scale. What it does rule out is the hopeful reading: fixing the
context wall did not convert into levels on the one game where it was measured.

## A side effect worth knowing before running more evals

The dashboard's `generaliz.` line fell from 5 levels to 4 when this run landed. It unions eval
batches **deduplicated by game**, so a fresher r11l result supersedes an older one — and this run's
1 level replaced the earlier 2. The function's own docstring already warns "a newer file is not a
better one" about empty misfire batches; the same rule bites for a non-empty WORSE batch, which the
comment does not anticipate.

Checked before writing: `git status -- ops/arc_solve_registry.yaml results/` was clean, so this run
changed no tracked record. The headline moved because of how the dashboard aggregates, not because
evidence was overwritten.

## What would settle the open half

Neither engine is scored. `exact_acc` and `cell_recall` against held-out transitions remain the
numbers that convert the structural reading above into a result, and
`scripts/arc_e3_induced_model_quality.py` produces them offline from the two engines already on
disk. Milestone 610's `exp6968-arc-post-refit-induction-audit` is scheduled to do precisely that.

---

## Correction: the run emitted FOUR engines, not two

The table above says two. Two more landed after it was written:

| emitted | lines |
|---|---|
| 02:05:36Z | 170 |
| 06:24:27Z | 86 |
| 07:12:47Z | 184 |
| 07:49:08Z | 65 |

So the rate was roughly one engine per 1.7 hours over the full 6h57m, not the one-per-4.7-hours I
reported at the four-hour mark — the first gap was the outlier, not the cadence. The structural
reading above is based on engines 1 and 2 only; engines 3 and 4 have not been examined.

## The scoring half ran and could not see the engines

`exp6968-arc-post-refit-induction-audit` executed at 07:52:12Z, three minutes after this run wrote
its artifact — so no concurrency collision. It returned
`blocked_arc_post_refit_induction_audit` with `arc_induction_generalization_positive_score = 0` in
0.097s, and **`engine_resolution` came back with zero rows**. It found no engines at all.

**Why.** The audit reads `diagnostics.get("induction_attempts", [])` to obtain a target engine hash,
then globs `attempts/wm_*__{target_engine_sha16}.py` for it. The eval artifact carries no
`induction_attempts` — not at the top level, not on the row. With no target hash there is nothing to
glob, so zero candidates, so `selected is None`.

**So the two halves do not compose, and it is the same gap as before.** The generation side now
records `generator_channels` because I wired it yesterday; it still does not record which engines it
emitted, when, or against which transition source. The audit needs exactly that triple. Four engines
sit on disk, complete and parseable, and the tool built to score them cannot find them.

**A second defect, in the reporting rather than the logic.** The artifact's
`gate_check_summary` names `failed_check: immutable_transition_source`. That gate is the
EARLY-RETURN SENTINEL — `experiment_6968...py:945` appends it with a hardcoded `False` when an
earlier gate fails, before the transition source is ever examined. The real failure was
`one_content_matched_engine`/`unambiguous_engine_path` finding nothing. A reader chasing the named
gate would investigate transition immutability, which was never the problem. This is the same class
as the conductor's `artifact_not_updated_past_bootstrap` naming the wrong path, fixed 2026-09-03 as
REQ-CONDUCTOR-VERDICT-3.

**What would close the measurement.** Emit `induction_attempts` on the eval row — engine sha16,
`engine_emitted_at`, `run_file_mtime_ns`, `transition_source_path` — mirroring the
`generator_channels` wiring. Then exp6968 resolves an engine and produces the `exact_acc` /
`cell_recall` numbers this note has been missing throughout. Not done here: it is a third instance
of the same wiring pattern in three days and deserves to be designed once rather than added
piecemeal a field at a time.
