# Compaction A/B for the ARC induction tool loop: the measurement already existed

**Date:** 2026-09-06. **Flag:** `CARNOT_ARC_INDUCE_TOOL_COMPACT` (REQ-ARC-WMTE-6540).
**Design note:** `docs/research-notes/arc-induction-compacted-carried-state-2026-08-19.md`.

**Why a new note and not an append to the grammar-trial note.** The grammar note
(`grammar-27b-trial-2026-09-05.md`) records a different lever, and two other agents
were writing to it in this same session. This note covers compaction only. It
cross-references the grammar note where the two levers interact, which they do.

**No GPU was used for this note.** Nothing under `results/` was written. Every
number below is read from evidence that already existed, or computed from it.

---

## 1. Summary

The task was to run an A/B of compaction on the pinned Qwen3.8-27B. Three things
were found before any GPU time was spent, and together they made a new run the
wrong thing to buy.

| # | Finding | Kind |
|---|---|---|
| 1 | The briefed A/B cannot run. The loop REFUSES `grammar + compaction` together. | measured (test) |
| 2 | At the trial's 4-turn shape, compaction never fires. 0 events in 14 measured cells. | measured (replay) |
| 3 | A complete 13-cell paired compaction A/B already exists, from 2026-08-20, on a byte-identical compaction path. It fires 10 times. The flag ledger records it as `evidence: []`. | measured (existing run) |

The verdict on the flag, from that existing A/B: **compaction does what it claims
in aggregate, and buys nothing measurable end to end.** Peak prompt tokens fall 26%
on the cells that fire. Holdout accuracy is unchanged, on a metric that turns out
to be floored. Wall clock gets worse, not better. One cell lost its only scored
engine in the treatment arm.

Chasing one anomalous cell also surfaced a defect the existing telemetry cannot
see: **2 of the 10 rebuilds made the prompt BIGGER**, and nothing counts that
(Section 3b-bis).

---

## 2. The firing question, answered first

Compaction only matters when the message list grows. If it never fires, an A/B on
it measures nothing.

### 2a. Blocker A — the briefed configuration is refused by the loop

`arc_induction_tool_loop.py:768`:

```python
if grammar_json and compact.enabled:
    return _finish("grammar_compaction_unsupported")
```

This runs BEFORE the turn loop. With the grammar on and compaction on, the tool
loop executes zero turns and the caller falls back to the plain single-shot
induction. A briefed A/B holding the grammar fixed in both arms would therefore
have compared "tool loop with grammar" against "no tool loop at all", and labelled
the difference "compaction".

The interlock is deliberate and tested (REQ-7044). It arrived with the grammar
feature on 2026-09-05 (commit `d5d72141ca`), after the compaction module shipped.

Measured, not read:

```
$ pytest tests/python/test_arc_tool_grammar_transport.py -k compaction --no-cov -q
1 passed in 3.95s
```

`test_compaction_combination_is_explicitly_rejected` asserts
`terminated_by == "grammar_compaction_unsupported"`, `compactions == 0`, and that
only the fallback prompt reaches the wire.

### 2b. Blocker B — at 4 turns the trigger is never crossed

The trigger is `previous response's measured prompt_tokens >= turn-0 prompt_tokens
+ growth`, with `growth` defaulting to 8192.

`compactions: 0` in the grammar trials only says the flag was off. It cannot say
whether the trigger WOULD have fired. So the shipped `CompactionController` was
driven directly with the measured `prompt_tokens_per_turn` sequences from both
grammar trials, in the same order the loop calls it.

Population: 14 cells. 7 games x 2 trials (trial 1 force-turn 2, trial 2
force-turn 3), Qwen3.8-27B, seed 7, 4 turns each.

| trial | game | prompt_tokens_per_turn | growth over turn 0 | threshold | fires |
|---|---|---|---|---|---|
| 1 | g50t | 9272, 11091, 12433, 14080 | 4808 | 17464 | 0 |
| 1 | lp85 | 13411, 14935, 20466, 21116 | 7705 | 21603 | 0 |
| 1 | r11l | 10605, 12465, 13227, 14305 | 3700 | 18797 | 0 |
| 1 | re86 | 16629, 18577, 20278, 21686 | 5057 | 24821 | 0 |
| 1 | sb26 | 5438, 6900, 7038, 7580 | 2142 | 13630 | 0 |
| 1 | sp80 | 6179, 8065, 12551, 14606 | 8427 | 14371 | 0 (crosses on the LAST measurement) |
| 1 | tr87 | 13815, 15773, 16615, 18213 | 4398 | 22007 | 0 |
| 2 | g50t | 9272, 11091, 12386, 12911 | 3639 | 17464 | 0 |
| 2 | lp85 | 13411, 14935, 20419, 20546 | 7135 | 21603 | 0 |
| 2 | r11l | 10605, 12465, 13180, 15114 | 4509 | 18797 | 0 |
| 2 | re86 | 16629, 18577, 20231, 20370 | 3741 | 24821 | 0 |
| 2 | sb26 | 5438, 6900, 6991, 7131 | 1693 | 13630 | 0 |
| 2 | sp80 | 6179, 8065, 12504, 13533 | 7354 | 14371 | 0 |
| 2 | tr87 | 13815, 15773, 16568, 17452 | 3637 | 22007 | 0 |

**0 compaction events in 14 of 14 cells.** One cell (trial 1 sp80) crosses the
threshold on its final measurement. The controller checks at the TOP of a turn
using the previous turn's number, so a fifth turn would have fired there and no
earlier.

### 2c. What WOULD make it fire

Two levers, both measured on the same 14 cells.

**Lower the growth knob.** Replaying the same sequences at other values of
`CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH`:

| growth | events, trial 1 | events, trial 2 | total |
|---|---|---|---|
| 1024 | 7 | 7 | 14 |
| 2048 | 6 | 6 | 12 |
| 3072 | 4 | 4 | 8 |
| 4096 | 2 | 2 | 4 |
| 6144 | 2 | 2 | 4 |
| 8192 (default) | 0 | 0 | **0** |

**Raise the turn count.** This is answered by measurement, not extrapolation — see
Section 3. In the 2026-08-20 A/B at up to 12 turns, 7 of 13 cells fired, with the
first event landing on turns 3, 5 or 6. Firing is not a pure function of turn
count: one cell fired at turn 3 of a 4-turn run because a single tool result added
~10k tokens in one turn.

---

## 3. The A/B that already existed

`results/tool_loop_compaction_ab_20260820/` holds a complete paired A/B.

| | OFF arm | ON arm |
|---|---|---|
| `compaction_env` | `<unset>` | `"1"` |
| cells | 13/13 complete | 13/13 complete |
| window | 2026-08-20 03:41:41Z – 05:45:02Z | 06:41:40Z – 09:41:20Z |
| arm wall clock | 7401.1 s | 10779.5 s |

Same 13 games, same `seed_base: 100`, same GGUF
(`unsloth/Qwen3.8-27B-GGUF`, Q4_K_M), same `n_ctx: 106496`, same server argv, same
GPU 0, preconditions recorded in both meta files. Roster: tu93, tr87, sp80, sb26,
re86, g50t, r11l, cd82, ar25, lp85, sk48, vc33, ft09.

### 3a. Does this A/B measure the CURRENT code?

The compaction mechanism, yes. Checked, not assumed:

- `python/carnot/agentic/arc_induction_compact_state.py` has exactly two commits,
  both 2026-08-19 (`d3e9f8cf52`, `4b597d2834`). Nothing has touched it since. The
  module the A/B exercised is byte-identical to HEAD.
- The compaction wiring block inside the loop is byte-identical between the loop
  at `4b597d2834` and the loop at HEAD.

The surrounding loop has changed: selfparse (2026-08-28), tool-gap feedback
(2026-08-29), the grammar (2026-09-05). Those change the tool-call transport, and
so change how fast a transcript grows. So the MECHANISM findings below transfer to
HEAD. The end-to-end quality and wall-clock findings are on the older loop and
should be read as such.

### 3b. Firing

Fire turns below are not inferred from the sawtooth. They are recovered by
replaying the shipped controller over each ON cell's own measured sequence. The
replay reproduces the recorded `compactions` counter on **13 of 13 cells**, so the
fire turns are the real ones — and that agreement is also the falsification
control for the Section 2b instrument, which is the same code path applied to data
where firing did happen.

| game | turns OFF | turns ON | compactions | fire turn: prompt before -> after | refetch |
|---|---|---|---|---|---|
| ar25 | 6 | 10 | 1 | t5: 16616 -> 10960 (-5656) | 2 |
| cd82 | 6 | 12 | 2 | t3: 21802 -> 6263 (-15539); t6: 24309 -> 8743 (-15566) | 5 |
| ft09 | 5 | 5 | 0 | – | 0 |
| g50t | 12 | 12 | 1 | t5: 15842 -> 9450 (-6392) | 0 |
| lp85 | 9 | 9 | 2 | t3: 25730 -> 10224 (-15506); **t7: 18754 -> 21730 (+2976)** | 4 |
| r11l | 6 | 6 | 0 | – | 0 |
| re86 | 6 | 6 | 0 | – | 0 |
| sb26 | 4 | 4 | 0 | – | 0 |
| sk48 | 11 | 12 | 1 | **t7: 17335 -> 20451 (+3116)** | 2 |
| sp80 | 4 | 4 | 0 | – | 0 |
| tr87 | 11 | 12 | 2 | t6: 20184 -> 14287 (-5897); t11: 22916 -> 14304 (-8612) | 9 |
| tu93 | 7 | 7 | 0 | – | 0 |
| vc33 | 9 | 4 | 1 | t3: 15804 -> 6446 (-9358) | 0 |

**10 compaction events across 7 of 13 cells.** The OFF arm recorded 0, as the flag
requires.

**The six cells that did not fire are byte-identical across the two arms** — same
turn count, same peak prompt tokens, same `decode_tokens_total` to the token. That
is a free A/A validation, and it says the flag is inert when the trigger does not
cross. Every cell that diverged is a cell that fired. No unexplained divergence.

### 3b-bis. Two of the ten rebuilds made the prompt BIGGER

This was found by chasing sk48, which records a compaction but whose
`prompt_tokens_per_turn` never drops.

| cell | event | prompt before | prompt after | change |
|---|---|---|---|---|
| lp85 | turn 7 | 18754 | 21730 | **+2976** |
| sk48 | turn 7 | 17335 | 20451 | **+3116** |

**2 of 10 events, 20%, moved the metric the wrong way.** A rebuild replaces the
transcript with `base + carried state + one full tail round`. When the base is
small (sk48 8531, lp85 9074) and the tail round happens to be a large tool result,
that sum can exceed the transcript it replaced. The rebuild then pays the
re-prefill cost and buys negative context.

Nothing detects this. The design note's Section 7 protects the carried state's
contents and flags `compact_floor_hit` when the state busts its budget, but there
is no check that the rebuilt message list is actually SHORTER than the one it
replaces. `compact_floor_hit` was false on both of these cells, so the existing
telemetry reports them as healthy events.

The consequence is bounded rather than runaway, because the thrash floor then
raises the threshold off the new, larger post-rebuild size — so a bad rebuild
delays the next event instead of causing a loop. But it is a real gap: the cheapest
fix is to compare the rebuilt size against the current one and skip the event when
it would not shrink.

### 3c. The four counters (ON arm, 13 cells)

| counter | value | reading |
|---|---|---|
| `compactions` | 10 | fires as designed; expected 1–3 per loop, observed 1–2 |
| `compact_floor_hit` | 0 cells | the 2048-token state budget never bound |
| `compaction_thrash_alarm` | 0 cells | no cell exceeded the 5-event alarm; the thrash fix holds |
| `refetch_tool_calls_post_compaction` | 22 | 12.4% of the arm's 177 inspection calls; advisory gate is <=20%, so it passes |

`duplicate_candidate_submissions` was 1, on r11l. **r11l had 0 compactions**, so
that duplicate is background behaviour and is NOT attributable to compaction.

### 3d. Per-arm

Population: 13 paired cells for every row except holdout accuracy, which has 11 —
sb26 died on a transport error in both arms and vc33 in the ON arm only, so
neither has a score to compare. A missing score is not a zero.

| metric | n | OFF median | ON median | OFF sum | ON sum |
|---|---|---|---|---|---|
| turns | 13 | 6 | 7 | 96 | 103 |
| decode tokens | 13 | 20640 | 24596 | 310759 | 326038 |
| cell wall_s | 13 | 650.9 | 722.1 | 9835.5 | 10753.9 |
| peak prompt tokens | 13 | 18753 | 16236 | 285975 | 232328 |
| tool calls | 13 | 10 | 10 | 182 | 206 |
| candidates scored | 13 | 3 | 2 | 34 | 27 |
| holdout accuracy | 11 | 0.000 | 0.000 | 0.5 | 0.5 |

Engines emitted (`ok=True`): OFF 12/13, ON 11/13. Engines scoreable: OFF 12/13,
ON 11/13.

### 3e. Per-cell

| game | turns O/N | comp | peak O/N | decode O/N | wall O/N | acc O | acc N |
|---|---|---|---|---|---|---|---|
| ar25 | 6/10 | 1 | 18753/17922 | 20640/30591 | 648/990 | 0.000 | 0.000 |
| cd82 | 6/12 | 2 | 25243/24309 | 20136/36844 | 651/1223 | 0.000 | 0.000 |
| ft09 | 5/5 | 0 | 15983/15983 | 16822/16822 | 536/546 | 0.000 | 0.000 |
| g50t | 12/12 | 1 | 34928/17312 | 40770/40917 | 1304/1344 | 0.000 | 0.000 |
| lp85 | 9/9 | 2 | 37713/27245 | 29707/26479 | 1011/925 | 0.000 | 0.000 |
| r11l | 6/6 | 0 | 15874/15874 | 17486/17486 | 547/580 | 0.000 | 0.000 |
| re86 | 6/6 | 0 | 14313/14313 | 20533/20533 | 586/695 | 0.000 | 0.000 |
| sb26 | 4/4 | 0 | 10891/10891 | 10448/10448 | 320/352 | n/a | n/a |
| sk48 | 11/12 | 1 | 31511/25202 | 36231/40173 | 1191/1344 | 0.000 | 0.000 |
| sp80 | 4/4 | 0 | 8321/8321 | 10447/10447 | 315/349 | 0.000 | 0.000 |
| tr87 | 11/12 | 2 | 34738/22916 | 34892/36935 | 1118/1242 | 0.000 | 0.000 |
| tu93 | 7/7 | 0 | 16236/16236 | 24596/24596 | 705/722 | 0.500 | 0.500 |
| vc33 | 9/4 | 1 | 21471/15804 | 28051/13767 | 904/443 | 0.000 | n/a |

---

## 4. The gates from the design note

### G-M (mechanism): PASS

Peak prompt tokens on the 7 cells that fired: 204357 -> 150710, **-26.3%**.

Two of those cells ran the same number of turns in both arms, which is the only
clean read — a longer ON run can peak higher despite compacting:

| game | turns | peak OFF | peak ON | change |
|---|---|---|---|---|
| g50t | 12 | 34928 | 17312 | **-50.4%** |
| lp85 | 9 | 37713 | 27245 | **-27.8%** |

Compaction bounds the context. That claim is supported at the peak.

It is NOT supported per event. **2 of the 10 rebuilds made the prompt bigger**
(Section 3b-bis), and no counter reports that. So the correct reading of G-M is:
the aggregate mechanism works, and 20% of individual events were counterproductive
and invisible.

### G-P (parse safety): PASS

| arm | parse failures | tool calls | rate |
|---|---|---|---|
| OFF | 2 | 182 | 1.10% |
| ON | 2 | 206 | 0.97% |

ON minus OFF is -0.13 percentage points, inside the 5pp gate. `unparsed_tool_call_text_turns`
was 0 in both arms. The kill condition did not trigger.

### G-Q (quality): UNINFORMATIVE, not a pass

11 paired cells, 11 exact ties, mean delta +0.0000. Read literally that clears the
gate. It should not be read literally.

**21 of the 23 scored cells across both arms sit at exactly 0.0 holdout accuracy.**
The only non-zero value is tu93 at 0.5 in both arms. The metric had almost no
variance in this population, so a tie carries no information about whether
compaction hurts quality. This is the `flip_count == 0` shape that CLAUDE.md's
FALSE_NEGATIVE_RISK rule exists to catch: a null on a floored metric is a
degenerate test, not evidence of non-inferiority.

Two directional signals do point the other way and deserve naming:

- **Candidates scored fell 34 -> 27** while turns rose 96 -> 103. The ON arm ran
  longer, inspected more, and submitted fewer engines.
- **vc33 lost its only scored engine in the ON arm.** It compacted at turn 3, then
  hit an HTTP 500 and terminated with 0 candidates, where the OFF arm ran 9 turns
  and scored 1. This is n=1 and it is confounded: HTTP 500s hit both arms in this
  run (sb26 identically in both). The compacted request itself SUCCEEDED — the
  failure was on the turn after — so this is not the design's
  `transport_error_on_compacted_request` case. That counter did not exist yet when
  this A/B ran, so it cannot be checked directly. **Do not claim compaction caused
  this. Do not dismiss it either.**

### G-W (single-stream cost): FAIL on the gate's own metric

The gate was ON <= 1.05x OFF on median cell wall_s. Observed 722.1 / 650.9 =
**1.109x**. It fails.

Decomposed, the mechanism is not a per-turn slowdown:

| quantity | OFF | ON | ON/OFF |
|---|---|---|---|
| turns | 96 | 103 | 1.073 |
| seconds per turn | 102.45 | 104.41 | 1.019 |
| decode tokens per turn | 3237 | 3165 | 0.978 |
| seconds per decode token | 0.03165 | 0.03298 | 1.042 |
| arm wall clock | 7401.1 | 10779.5 | 1.456 |

Most of the arm-level 1.46x is the ON arm running 7 more turns, because compaction
changed the trajectory: `early_stop_non_improving` fired 9 times OFF and 5 times
ON, with 4 ON cells reaching the turn cap instead. Compaction let the loop keep
going. The loop used that room and got nothing for it.

### G-K (concurrency): NOT MEASURED

The design names this as the gate where the lever earns its keep. No multi-stream
probe exists in either arm. This remains the open question.

---

## 5. Did shorter context actually decode faster?

The design's single-stream case rests on a decode-rate curve quoted as
42.4 tok/s at 10k falling to 29.0 tok/s at 80k. That was tested directly against
the A/B's own server logs, pairing each request's decode rate with the context
length it ran at.

Pooled across both arms, n=173 requests:

| context | n | median tok/s |
|---|---|---|
| <10k | 20 | 34.25 |
| 10–15k | 49 | 32.58 |
| 15–20k | 51 | 31.71 |
| 20–25k | 24 | 30.63 |
| 25–40k | 27 | 30.23 |

The curve is real and it is **shallow in the band these loops occupy**: -11.7%
across the entire observed 6.7k–41.6k range. The steep part of the quoted curve
lies above 40k and this A/B never reached it.

Compaction did shift the distribution as intended: **78% of ON-arm requests ran
below 20k context, against 57% for OFF.** But the ON arm decoded SLOWER than OFF
at every matched context bucket, by roughly 1–2 tok/s:

| context | OFF median tok/s | ON median tok/s |
|---|---|---|
| <10k | 34.81 (n=7) | 33.60 (n=13) |
| 10–15k | 34.20 (n=15) | 32.12 (n=34) |
| 15–20k | 33.06 (n=18) | 31.41 (n=33) |
| 20–25k | 31.57 (n=9) | 30.29 (n=15) |
| 25–40k | 30.29 (n=19) | 29.57 (n=8) |

The gain from moving mass to shorter contexts is about the same size as this
unexplained arm-level penalty, so the two cancel. That is why the wall clock did
not improve.

**The penalty is a confound, and its cause is unknown.** The arms ran
sequentially, three hours apart, in separate server processes. Thermal state,
clock behaviour, or another process on the box could produce it. This comparison
also pools across games, which contribute unequally to each bucket. Treat the
per-bucket gap as a caution about the sequential design, not as a property of
compaction.

---

## 6. Why no new GPU run was made

About 40 minutes of GPU were authorised. It was not spent, for four reasons.

1. The briefed A/B is impossible. Grammar and compaction are mutually exclusive.
2. The briefed 4-turn shape produces 0 firings, measured on 14 cells. An A/B there
   compares a flag against itself.
3. A 13-cell paired A/B on the byte-identical compaction path already exists and
   cost about 5 hours of GPU. Forty minutes cannot improve on it.
4. The one gate that would change the answer is G-K, concurrency. It needs a
   multi-stream probe with its own design and its own budget. It does not fit in
   40 minutes, and guessing at it would waste the time.

---

## 7. What this note does not cover

- **G-K, concurrency.** Unmeasured, and it is the design's own primary claim.
- **Quality at claim grade.** The existing A/B's holdout metric is floored, so the
  quality question is genuinely open. Answering it needs a corpus with headroom,
  and N >= 30 paired cells per the Sample-Size Rigor rule.
- **Compaction on the CURRENT loop.** The A/B ran before selfparse, tool-gap
  feedback and the grammar landed. Those change transcript growth, and so change
  how often the trigger crosses.
- **The legal configuration at 4 turns.** Compaction is only permitted with the
  grammar off. No prompt-token trajectory exists for a selfparse tool loop on the
  27B at 4 turns, so Section 2b's 0-firing result is measured on the grammar arm
  and is not a direct measurement of the selfparse arm.
- **Whether compaction caused vc33's early transport failure.** n=1, confounded.
- **The `GROWTH` and `STATE_BUDGET` knobs as levers.** Both sat at their defaults
  in every cell that has ever run. The growth sweep in Section 2c is a replay over
  recorded prompt sizes, not a live A/B.

---

## 8. Operator decisions this raises

1. **Is the grammar/compaction interlock permanent?** Today they cannot be
   combined. The grammar is what makes the 27B submit an engine at all. If the
   grammar stays on the live path, compaction is unreachable there and the flag is
   dead code in practice. If both are wanted, someone has to make the rebuild's
   message shape work under the GBNF transport.
2. **Is G-K worth funding?** It is the only gate that could still justify this
   lever. It needs a K>=4 multi-stream probe on both arms.
3. **Should the default `GROWTH` drop from 8192?** At today's shorter loops it
   makes compaction a no-op. Lowering it makes the lever reachable, and also makes
   it fire on cells where the design expected it not to.
4. **Should the rebuild refuse to grow the prompt?** Section 3b-bis found 2 of 10
   events made the prompt larger, with no counter reporting it. A one-line guard —
   compare the rebuilt size against the current one, skip the event when it would
   not shrink, and count the skip — closes it. This is a code change, so it is
   raised here rather than made.
5. **The A/B result should not have been invisible.** It sat in `results/` for 17
   days while the ledger said `evidence: []`. That is the state this note fixes.

---

## 9. Exact commands

All run with `PYTHONPATH` pinned to this worktree's `python/` and the imported
module path asserted, so nothing measured the main checkout.

```
# Blocker A, the interlock, live:
PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
  tests/python/test_arc_tool_grammar_transport.py -k compaction --no-cov -q \
  --basetemp=<scratch>/compaction-ab/pytest-bt
# -> 1 passed in 3.95s

# Blocker B, the trigger replay and the growth sweep:
PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu .venv/bin/python \
  <scratch>/compaction-ab/replay_trigger.py

# Window reuse, and the instrument audit on prompt_tokens_per_turn:
... <scratch>/compaction-ab/verify_inputs.py     # first attempt, mis-specified
... <scratch>/compaction-ab/verify_inputs2.py    # diagnosis
... <scratch>/compaction-ab/verify_inputs3.py    # exact reconstruction, PASS

# The existing A/B:
... <scratch>/compaction-ab/aggregate2.py
... <scratch>/compaction-ab/gates.py
... <scratch>/compaction-ab/decode_rate.py
... <scratch>/compaction-ab/fire_turns.py   # fire turns + the prompt-grew finding
```

### Window reuse

All seven `window_sha256` values match between trial 1 and trial 2, as trial 2
reported. Re-verified here:

| game | sha256 (both trials) |
|---|---|
| g50t | 225f11d0379c… |
| lp85 | b5b63439c043… |
| r11l | ecb7674e4b2f… |
| re86 | 4d4e96d28df1… |
| sb26 | 672414ce0d52… |
| sp80 | 277a5dfb3c8e… |
| tr87 | a1836b99bc8d… |

The pickles those trials loaded are still on disk in
`<scratch>/grammar-27b-trial/windows_rw/`, one per game.

### Instrument audit

The claim in Section 2b rests on `prompt_tokens_per_turn`. If the harness had
invented those numbers, the claim would be worthless.

A first cross-check compared them against the server's `prompt eval time = ... /
N tokens` lines and reported a mismatch. **The check was wrong, not the data.**
Under `cache_prompt: true` that line reports the DELTA prefilled, while
`usage.prompt_tokens` reports the FULL prompt.

The exact relation the server does expose per request is its release line,
`stop processing: n_tokens = X`, where `X = prompt_tokens + completion_tokens - 1`.
For all **56 turns** across both trials' grammar arms, `pt[k] + decode[k] - 1`
matches a release record in the server's own log. 28/28 in trial 1, 28/28 in
trial 2. The telemetry is the server's own count.

### Falsification check on the replay

A replay that always returns zero would produce the same headline. Two controls
rule that out.

1. **The growth sweep.** The same instrument, over the same data, reports 14 events
   at `growth=1024` and 0 at `growth=8192`. It can fire.
2. **Ground truth.** The same instrument, run over the 2026-08-20 ON arm's measured
   sequences, reproduces that run's recorded `compactions` counter on **13 of 13
   cells** — including the 7 non-zero ones. It agrees with reality where reality is
   known.

Which cell of the design would have failed if the claim were wrong: if compaction
did fire at 4 turns, control 1 would have shown a non-zero count at `growth=8192`,
and control 2 would have disagreed with the recorded counters. Neither happened.

---

## 10. Cross-references

- `docs/research-notes/arc-induction-compacted-carried-state-2026-08-19.md` — the design and its gates.
- `docs/research-notes/grammar-27b-trial-2026-09-05.md` — the grammar lever, whose interlock blocks this one.
- `results/tool_loop_compaction_ab_20260820/` — the A/B. EVIDENCE, read only.
- `python/carnot/agentic/arc_induction_compact_state.py` — the module, unchanged since 2026-08-19.
- `python/carnot/agentic/arc_induction_tool_loop.py:768` — the interlock.
- `tests/python/test_arc_tool_grammar_transport.py::test_compaction_combination_is_explicitly_rejected` — REQ-7044.
- `ops/arc_flag_ledger.yaml` — the three compaction flags, updated by this note.
- CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor" — the FALSE_NEGATIVE_RISK rule Section 4's G-Q reading applies.
