# What the live ARC agent actually fails at: a fine read of the induction round records

Date: 2026-08-27
Author: outer-loop session (analysis only; no code changed, no run touched)
Instrument: REQ-ARC-WMTE-6710 `induction_round_records` + `llm_channels`

## 1. Population, stated before any number

I read three harness runs. I did **not** pool them.

| Run | Rows | Round records | Supervisor | Window | Code | Used? |
|---|---|---|---|---|---|---|
| `supon/rows.json` | 6 | **0** | applied, 0 firings | 400 | **pre-REQ-6710** | EXCLUDED |
| `supshadow/rows.json` | 6 | 6 | shadow, 0 firings | 400 | post-fix | included |
| `supwindow/rows.json` | 4 | 4 | applied, **2 redirects per cell** | 120 | post-fix | included |

Read at **2026-08-27T13:36Z**. File mtimes: supon 2026-08-25T12:22Z, supshadow
2026-08-26T00:35Z, supwindow 2026-08-27T11:34Z.

**The brief expected supwindow to still be growing. It is not.** Process 689528 is gone.
`launch.log` records `Terminated ... exit=143` at 12:48:48Z — SIGTERM during cell 5
(`ar25/20260725`). The run stopped at 4 of 6 planned cells. So my population is final, and it
is **truncated by an external kill, not by a result**. The 2 missing ar25 cells are missing
data, not absent data.

**Analysed population: 10 rows, 25 induction attempts carrying a round record, 25 rounds.**
Of those 25 attempts, 5 planned successfully and 20 failed.

**Excluded, and why.** I excluded the nested repo clones under `$CLAUDE_JOB_DIR/tmp`
(`abpin/`, `headwt/`). I globbed the three run directories by explicit path, never
recursively. I wrote nothing under `results/**`.

## 2. The brief's coarse tally pools across a code change. Two of its six rows are artefacts.

The tally in the brief reproduces **exactly** as the pooled row-level `induction_skipped`
across all three runs. I confirmed this term by term. That pooling is the error.

| Cause | supon (pre-fix) | supshadow | supwindow | Brief's pooled |
|---|---|---|---|---|
| `goal_unreached_within_budget` | 5 | 4 | 3 | 12 |
| `heldout_transition_verification_failed` | 0 | 3 | 7 | 10 |
| `no_reachable_plan_after_refinement` | **6** | **0** | **0** | 6 |
| `degenerate_goal_predicate` | 1 | 2 | 0 | 3 |
| `world_model_accuracy_below_threshold` | 0 | 1 | 0 | 1 |
| `exception` | 0 | 0 | 1 | 1 |

**Every `no_reachable_plan_after_refinement` comes from supon.** The label appears **zero**
times in 25 instrumented attempts.

This is not chance. Commit `288ea485f9` changed the label, not only the record.
`arc_llm_reinduction.py:1672` pre-sets `skipped` to that default before the round loop runs.
Two sites diagnosed a real cause and wrote it only into the per-round row. The fix made both
sites also set the outer variable. Those two sites are exactly
`heldout_transition_verification_failed` (:1895) and the exception site (:2246).

So supon's 6 are a **mixture** of genuine no-plan, masked dynamics failures, and masked
exceptions. Nothing recorded in supon can separate them, because supon has no round records.
The commit message's own estimate was 9 of 18.

**Correction to the brief.** "6 `no_reachable_plan_after_refinement`" is not a residual
planning-failure count. It is the pre-fix mask itself, counted once more.

### The planner never ran and failed. Not once.

`plan_reaches_goal` distinguishes three states by design: `None` means the round stopped
before the planner, `False` means the planner ran and found nothing, `True` means it planned.

Across all 25 instrumented rounds:

| `plan_reaches_goal` | n |
|---|---|
| `None` (stopped before the planner) | **20** |
| `True` | 5 |
| `False` (planner ran, found no plan) | **0** |

**Every one of the 20 failures happened upstream of the planner.** Planning is not the
bottleneck in this corpus. It is not even reached.

## 3. The 12 `goal_unreached_within_budget` cases

**The split the brief asks for already exists, and the answer is 7 out-of-budget, 0
unreachable.** But the budget that ran out is not the planner's.

`_goal_satisfiability_check` (`arc_llm_reinduction.py:1013-1040`) terminates three ways and
emits a different label for each:

| Termination | Label | Meaning |
|---|---|---|
| `budget_exhausted` | `goal_unreached_within_budget` | reachability **UNDECIDED** |
| `depth_capped` | `goal_unreached_within_depth` | out of reach within the planner's own cap |
| `queue_exhausted` | `degenerate_goal_predicate` | **proven** unreachable |

An unreachable goal is therefore never labelled `goal_unreached_within_budget`. It is labelled
`degenerate_goal_predicate`.

All 7 instrumented cases (n=7; the other 5 are supon and cannot be split):

| Run | Game/seed | Att | Termination | engine_calls | max_nodes | frontier_remaining | grids | held-out acc |
|---|---|---|---|---|---|---|---|---|
| supshadow | tu93/20260725 | 2 | budget_exhausted | 20016 | 20000 | 2443 | 2984 | 1.0 |
| supshadow | tu93/20260726 | 1 | budget_exhausted | 20000 | 20000 | 5135 | 5716 | 1.0 |
| supshadow | ar25/20260725 | 0 | budget_exhausted | 20000 | 20000 | 147 | 1324 | 1.0 |
| supshadow | ar25/20260725 | 1 | budget_exhausted | 20004 | 20000 | 1040 | 1668 | 1.0 |
| supwindow | tu93/20260725 | 2 | budget_exhausted | 20000 | 20000 | 4342 | 4988 | 1.0 |
| supwindow | tu93/20260726 | 2 | budget_exhausted | 20017 | 20000 | 136 | 677 | 1.0 |
| supwindow | ar25/20260724 | 0 | budget_exhausted | 20009 | 20000 | 253 | 1410 | 1.0 |

The evidence that separates them, on every row:

- `termination == budget_exhausted` in **7/7**.
- `frontier_remaining > 0` in **7/7**. The search still had states to expand.
- `engine_calls >= max_nodes` (20000) in **7/7**. The node budget is what stopped it.
- `depth_truncated_nodes == 0` in **7/7**. The depth cap is not implicated.
- Held-out dynamics accuracy `1.0` and `accepted_by_heldout_verifier == True` in **7/7**.

**Read that last line again.** In every one of these 7 cases the world model **passed**
verification. The agent had a trusted model and then spent 20000 engine calls failing to
decide whether its own goal was reachable. It never planned.

The contrast case, `degenerate_goal_predicate` (n=2), terminates the other way:
`queue_exhausted`, `frontier_remaining == 0`, and 3552 / 32 engine calls — far below budget.
Those two are genuine proofs of unreachability under that engine.

**The fix follows from the label.** The code says it in its own words: "`max_nodes` is compute
and may be raised; the gate is quality and may not be widened." `frontier_remaining` spans 136
to 5135, so raising the budget would decide some of these 7 and not all. That number is the
per-case estimate of how far short it fell.

### One of the 2 degenerate cases is mislabelled, and the run log says so

`supshadow ar25/20260724 att1` reports `engine_calls=32`, `reachable_grids_evaluated=1`,
`frontier_remaining=0`. One reachable grid means the root. The engine produced **zero** novel
successors from any of 32 candidate actions. That is an inert engine, not a bad goal
predicate — and the label told the reader the goal was junk.

The same run's log names the mechanism in advance:

```
WINDOW VOCABULARY VIOLATION: actions [7] appear in this window and the in-model planner
cannot emit them. The induced engine will model transitions the planner never asks for,
and may be inert at the planning root even at cell_recall 1.0.
```

That is `ops/verifier_gaps.md` GAP-6260, open, whose own evidence cites `ar25`. The warning
fired 4 times in supshadow and 5 times in supwindow, on ar25 only.

## 4. The dynamics failures: not one class, three

By round label there are **11**, not 10. The extra one is a caller-side relabel, below.

| Run | Game/seed | Att | held-out acc | threshold | real_n | real acc | true mismatches |
|---|---|---|---|---|---|---|---|
| supshadow | tu93/20260724 | 0 | 0.5 | 1.0 | 25 | 0.84 | 4 |
| supshadow | tu93/20260725 | 0 | 0.75 | 1.0 | 25 | 0.84 | 4 |
| supshadow | tu93/20260726 | 0 | 0.75 | 1.0 | 25 | 0.92 | 2 |
| supshadow | ar25/20260726 | 0 | 0.5 | 1.0 | 25 | 0.84 | 4 |
| supwindow | tu93/20260724 | 0 | 0.75 | 1.0 | 25 | 0.92 | 2 |
| supwindow | tu93/20260726 | 0 | 0.75 | 1.0 | 25 | 0.92 | 2 |
| supwindow | tu93/20260724 | 1 | 0.0 | 1.0 | **1** | 0.0 | 1 |
| supwindow | ar25/20260724 | 1 | 0.0 | 1.0 | **1** | 0.0 | 1 |
| supwindow | tu93/20260725 | 0 | 0.0 | 1.0 | 25 | 0.0 | 25 |
| supwindow | tu93/20260726 | 3 | 0.830986 | 1.0 | 214 | 0.813084 | 40 |
| supwindow | ar25/20260724 | 2 | 0.282051 | 1.0 | 234 | 0.307692 | 162 |

Two facts hold across all 11:

- **`heldout_threshold == 1.0` in 11/11.** The gate demands perfect held-out accuracy.
- **`selected_candidate_name == "loaded_world_model.py"` in 11/11.** Every rejection is of an
  engine **retained from disk**, never of a freshly induced candidate. The retention path is
  what keeps failing verification.

The 11 then split into three classes:

**Class A — near-miss against a perfect-accuracy gate (n=6).** `real_n=25`, real accuracy
0.84 to 0.92, 2 to 4 wrong transitions. The implied held-out block is 4 rows (accuracies of
exactly 1/2 and 3/4). A model that is right on 21 to 23 of 25 real transitions is rejected
because 1 or 2 rows of a 4-row block are wrong. This is the largest class, and it is a
**threshold artefact**, not a model defect.

**Class B — single-row held-out block (n=2).** `real_n=1`, held-out accuracy 0.0. One
transition decides the model's fate. At n=1 the gate is a coin, not a measurement.

**Class C — genuine wipeout (n=3).** Real accuracy 0.0 on 25 rows, 0.813 on 214 rows, 0.308
on 234 rows. These are real dynamics failures and the gate is right to reject them.

**So: not one systematic class.** 6 of 11 are the gate being strict on a tiny sample, 2 of 11
are a sample of size one, and only 3 of 11 are the model genuinely getting the world wrong.

**Trap for the next reader: `real_mismatches_n` saturates at 8.** `WorldModelVerifier.score`
takes `max_mismatch: int = 8` (`arc_executable_world_model.py:1832`) and stops collecting. The
last row's digest shows `real_mismatches_n: 8` where the true count is **162**. The true count
is `real_n - real_n_correct`, which is what the table above reports. The source comment says
it plainly: "a 40-row total wipeout and an 8-row one are indistinguishable."

### A second masking layer, at the caller

For `supshadow tu93/20260726 att0` the attempt label and the round label disagree:

- attempt `skipped = world_model_accuracy_below_threshold`
- round `skipped = heldout_transition_verification_failed`

`arc_competition_agent.py:8355` sets the attempt label from its own separate gate
(`_gate_value < 0.5`). It overwrites what the round diagnosed. Both are dynamics failures, so
the class survives — but the row-level tally the project reads can still disagree with the
round record beneath it. This is the same defect shape REQ-6710 fixed, one layer up.

Counting it, **dynamics accounts for 11 of the 20 failing instrumented attempts (55%)**, and
the goal-reachability probe for 7 (35%).

## 5. `llm_channels`: 97.6% of decoded text is reasoning the pipeline discards

n = 10 rows, 42 completions, 8,661,106 raw characters.

| Channel | Characters | Share of raw |
|---|---|---|
| `chars_reasoning` | 8,456,727 | **97.64%** |
| `chars_final` | 203,623 | **2.35%** |
| unaccounted | 756 | 0.01% |

Per-row the split is remarkably stable: reasoning takes 96.86% to 98.23% of raw characters
across all 10 rows. Only the final channel becomes code.

Two further numbers:

- `reasoning_only` fired **2 of 42 completions**. Those two produced reasoning and no answer
  at all. The whole call was spent and returned nothing usable.
- `both_channels_empty` is 0 in 10/10. No completion came back wholly empty.

Pooled `tokens_predicted` is 2,580,979, so raw output runs at 3.36 characters per predicted
token. Held-out verification then rejected the resulting engine in 11 of 20 failing attempts.

**What this does not say.** These are characters, not tokens, and the commit says why: a
per-channel token count needs a tokenizer call the request never made. I have not measured
whether shortening the reasoning channel would change induction quality. This is a
measurement of where the budget goes, not evidence that spending it there is wrong.

## 6. Comparing like things: what I pooled, and what I did not

**supon is excluded entirely.** It predates the label fix. Pooling it is what produces the
phantom 6 planning failures.

**supshadow and supwindow are NOT behaviourally equivalent.** supwindow's supervisor fired 2
redirects in every one of its 4 cells; supshadow's fired 0 in all 6. A redirect changes what
the agent does next, so induction inputs differ downstream of a firing.

I stratified before pooling, then pooled only for the cause-mix question:

| Cause (round-level) | supshadow | supwindow |
|---|---|---|
| PLANNED | 3 | 2 |
| `heldout_transition_verification_failed` | 4 | 7 |
| `goal_unreached_within_budget` | 4 | 3 |
| `degenerate_goal_predicate` | 2 | 0 |
| **total attempts** | **13** | **12** |

The dynamics share is 4/13 in supshadow and 7/12 in supwindow. Fisher exact p = 0.238. At
n=25 that difference is not distinguishable from noise, so I pool for the cause mix and say so.
I do **not** pool for anything a redirect could move.

**Games.** tu93 and ar25 differ in dynamics share (8/17 vs 3/8), Fisher exact p = 1.000 — no
signal at this n. But they differ **qualitatively**: the vocabulary violation and the
root-inert engine occur on ar25 only. I report per-game findings separately for that reason,
not for a rate difference.

## 7. What I could not determine

- **supon's 6 `no_reachable_plan_after_refinement` cannot be split.** The field that would
  split them is `induction_round_records`, which supon does not carry. Only a re-run under the
  current code can recover them.
- **The held-out block size rule.** Implied block sizes are 1, 4, 39 and 71 — two orders of
  magnitude apart — with no constant ratio to `real_n` (25/6=4 and 234/6=39 fit, but 214 gives
  71, which is 214/3). The round record carries `heldout_accuracy` and `heldout_threshold`
  but **not `heldout_n`**. Adding `heldout_n` to the counterexample at
  `arc_llm_reinduction.py:1884` would settle it. Until then I cannot say whether Class A's
  4-row block is the design or an accident of corpus size.
- **The exception attempt has no round record.** `supwindow tu93/20260725` att1 raised
  `ValueError('not enough values to unpack (expected 2, got 1)')` inside
  `_rle_delta_compact` during `induce_prompt` (`arc_executable_world_model.py:2962`). It raised
  **before** the LLM was called and before any round row was appended, so
  `_induction_round_records` drops it (the `if not rounds and not ces` guard). The evidence
  survives only in `induction_tracebacks`. The instrument has a blind spot for pre-round
  exceptions. That is 1 of 21 failing attempts.
- **Whether raising `max_nodes` would decide the 7 undecided goals.** `frontier_remaining`
  bounds how far short each fell but does not predict the cost of finishing. Only a re-run at
  a higher budget answers it.
- **Whether the retained `loaded_world_model.py` engines are worse than fresh induction.**
  Every rejection was a retained engine, but the records do not show how often a retained
  engine was accepted, so I cannot compute a rejection rate for retention.

## 8. What the live agent actually fails at

In one sentence: **the agent is not failing to plan; it is failing to earn a trusted model and
then failing to decide whether its own goal is reachable, and it spends 97.6% of its generation
budget on reasoning text that never reaches either problem.**

Ranked by instrumented attempts (n=20 failing, plus 1 exception with no round record):

1. **Dynamics verification — 11 (55%).** Six of these fail a threshold of 1.0 on a 4-row
   held-out block while being 84–92% accurate on the real corpus. Two decide on a single row.
   Three are genuine. Every one rejects a retained on-disk engine.
2. **Goal reachability undecided — 7 (35%).** All budget-exhausted at exactly `max_nodes`,
   all with a non-empty frontier, all on a world model that had just passed verification.
   This is a compute limit reported honestly, not a goal defect.
3. **Goal proven unreachable — 2 (10%).** One of the two is really a root-inert engine
   (GAP-6260), mislabelled as a goal defect.
4. **Planning — 0.** `plan_reaches_goal` is `False` zero times in 25 rounds.

The cheapest next measurements, in order:

- Add `heldout_n` to the held-out counterexample. It costs one field and decides whether
  Class A is a threshold artefact or a real 6-of-11 model failure.
- Re-run the goal probe at a raised `max_nodes` on the 7 undecided cases. `frontier_remaining`
  already tells you which are close.
- Give the pre-round exception path a round record, so the instrument stops dropping the
  failures that happen earliest.

## Cross-references

- REQ-ARC-WMTE-6710 — `openspec/capabilities/arc-world-model-trust-energy/spec.md`
- `288ea485f9` — the label fix that makes supon incomparable
- `scripts/arc_scored_path_lever_harness.py:670` — `_induction_round_records`
- `python/carnot/agentic/arc_llm_reinduction.py:1013-1040` — the three-way goal termination
- `python/carnot/agentic/arc_executable_world_model.py:1832` — `max_mismatch = 8`
- `python/carnot/agentic/arc_competition_agent.py:8355` — the caller-side relabel
- `ops/verifier_gaps.md` GAP-6260 — root-liveness, open
