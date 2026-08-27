# What the live ARC agent actually fails at: a fine read of the induction round records

Date: 2026-08-27
Author: outer-loop session (analysis only; no code changed, no run touched)
Instrument: REQ-ARC-WMTE-6710 `induction_round_records` + `llm_channels`
Review: adversarially reviewed before publication. The review refuted two claims in the first
draft. Both are corrected here, and §9 records what they were.

## 1. Population, stated before any number

I read three harness runs. I did **not** pool them.

| Run | Rows | Rows with records | Attempts | Attempts with a record | Supervisor | Window | Code | Used? |
|---|---|---|---|---|---|---|---|---|
| `supon/rows.json` | 6 | **0** | — | 0 | applied, 0 firings | 400 | **pre-REQ-6710** | EXCLUDED |
| `supshadow/rows.json` | 6 | 6 | 13 | 13 | shadow, 0 firings | 400 | post-fix | included |
| `supwindow/rows.json` | 4 | 4 | 13 | 12 | applied, **2 redirects per cell** | 120 | post-fix | included |

Read at **2026-08-27T13:36Z**. File mtimes: supon 2026-08-25T12:22Z, supshadow
2026-08-26T00:35Z, supwindow 2026-08-27T11:34Z.

**The brief expected supwindow to still be growing. It is not.** `launch.log` records
`Terminated ... exit=143` at 12:48:48Z. 143 is 128 + SIGTERM. The kill landed during cell 5
(`ar25/20260725`), so the run stopped at 4 of 6 planned cells. My population is therefore
final, and it is **truncated by an external kill, not by a result**. The 2 missing ar25 cells
are missing data, not absent data. (The brief named pid 689528; that pid appears in no log in
the run directory, so I neither confirm nor use it. The exit line is the evidence.)

**Analysed population: 10 rows, 26 induction attempts, 25 attempts carrying a round record,
25 rounds.** One attempt has no record; §7 explains why.

**Excluded, and why.** I excluded the nested repo clones under `$CLAUDE_JOB_DIR/tmp`
(`abpin/`, `headwt/`). I addressed the three run directories by explicit path and never
globbed recursively. I wrote nothing under `results/**`.

## 2. Denominators, because this corpus has three and they are not interchangeable

The brief warned that `skipped` and `planned` are not mutually exclusive. They are not, and
the first draft of this note still got it wrong. Over the 10 included rows:

| Quantity | n |
|---|---|
| induction attempts | 26 |
| attempts that eventually planned (`induction_planned`) | 11 |
| attempts that ended with **no plan at all** | **15** |
| attempts carrying a `skipped` cause | **21** |
| 11 + 21 | 32 |
| **overlap: attempts both `skipped` and `planned`** | **6** |

Six attempts carry a rejection cause **and** planned anyway. The mechanism is explicit: when
the bounded refinement loop yields no plan, execution falls through to the plain single-shot
path (`arc_competition_agent.py:7925-7935`), which sets `planned=True` at :8429/:8456/:8492
while `skipped` keeps the refinement loop's diagnosis.

**So the cause tally's denominator is 21, and 21 means "bounded LLM reinduction rounds that
did not themselves produce a plan" — not "failed attempts".** Every share below is against 21
and is labelled that way. 15 is the number of attempts that produced nothing.

## 3. The brief's coarse tally pools across a code change

The brief's tally reproduces **exactly** as the pooled row-level `induction_skipped` across
all three runs. I confirmed it term by term. That pooling is the error.

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

Commit `288ea485f9` changed the label, not only the record. `arc_llm_reinduction.py:1672`
pre-sets `skipped` to that default before the round loop runs. Two exits diagnosed a real
cause and wrote it only into the per-round row. The fix made both also set the outer variable:
`heldout_transition_verification_failed` (:1901) and the exception site (:2252).

**Correction to the brief.** "6 `no_reachable_plan_after_refinement`" is not a residual
planning-failure count. It is a **mixture** of genuine no-plan cases and pre-fix masking, in
unknown proportion. It is not purely the mask: `:2462` still emits this label legitimately,
for a round that verified, planned, and found nothing. The commit message estimated 9 of 18.

### The planner was reached 5 times and failed 0 times

`plan_reaches_goal` distinguishes three states: `None` means the round stopped before the
planner, `False` means the planner ran and found nothing, `True` means it planned.

| `plan_reaches_goal` | n / 25 rounds |
|---|---|
| `None` (stopped before the planner) | **20** |
| `True` | 5 |
| `False` (planner ran, found no plan) | **0** |

Two statements, with their own denominators:

- **20 of 25 rounds never reached the planner.** They exited at `:1949`, `:2073` or `:2110`,
  all strictly before the `plan_in_model` call at `:2235`.
- **Of the 5 rounds that did reach the planner, 5 planned successfully.** The evidence that
  planning is not the bottleneck is **0 failures in 5 attempts**, not 0 in 25. n=5 is small.
  I state the weaker claim: planning is rarely *reached*, and this corpus contains no
  instance of it being reached and failing.

One caveat on the field's semantics, found by review. `plan_reaches_goal` is written at
`:2258`, after `plan_in_model` returns. If `plan_in_model` **raises**, the `except` at `:2242`
labels the round `selection_or_planning_exception` and the field stays absent, reading as
`None`. So `None` does not strictly mean "before the planner". It does not bite here — all 20
`None` rounds carry a label whose exit precedes `:2235` — but the instrument is weaker than
its own docstring claims.

## 4. The 12 `goal_unreached_within_budget` cases

**The split the brief asks for already exists in the label, and the answer is 7
out-of-budget, 0 unreachable.** The budget that ran out is not the planner's.

`_goal_satisfiability_check` (`arc_llm_reinduction.py:1013-1040`) terminates three ways and
emits a different label for each:

| Termination | Label | Meaning |
|---|---|---|
| `budget_exhausted` | `goal_unreached_within_budget` | reachability **UNDECIDED** |
| `depth_capped` | `goal_unreached_within_depth` | out of reach within the planner's own cap |
| `queue_exhausted` | `degenerate_goal_predicate` | **proven** unreachable |

An unreachable goal is therefore never labelled `goal_unreached_within_budget`.

All 7 instrumented cases (the other 5 are supon and cannot be split):

| Run | Game/seed | Att | Termination | engine_calls | max_nodes | frontier_remaining | grids | held-out acc |
|---|---|---|---|---|---|---|---|---|
| supshadow | tu93/20260725 | 2 | budget_exhausted | 20016 | 20000 | 2443 | 2984 | 1.0 |
| supshadow | tu93/20260726 | 1 | budget_exhausted | 20000 | 20000 | 5135 | 5716 | 1.0 |
| supshadow | ar25/20260725 | 0 | budget_exhausted | 20000 | 20000 | 147 | 1324 | 1.0 |
| supshadow | ar25/20260725 | 1 | budget_exhausted | 20004 | 20000 | 1040 | 1668 | 1.0 |
| supwindow | tu93/20260725 | 2 | budget_exhausted | 20000 | 20000 | 4342 | 4988 | 1.0 |
| supwindow | tu93/20260726 | 2 | budget_exhausted | 20017 | 20000 | 136 | 677 | 1.0 |
| supwindow | ar25/20260724 | 0 | budget_exhausted | 20009 | 20000 | 253 | 1410 | 1.0 |

The evidence that separates them, on every row (n=7):

- `termination == budget_exhausted` in **7/7**.
- `frontier_remaining > 0` in **7/7**. The search still had states to expand.
- `engine_calls >= max_nodes` (20000) in **7/7**. The node budget is what stopped it.
- `depth_truncated_nodes == 0` in **7/7**. The depth cap is not implicated.
- Held-out dynamics accuracy `1.0` and `accepted_by_heldout_verifier == True` in **7/7**.

**Read that last line again.** In all 7 the world model **passed** verification. The agent
held a trusted model, then spent 20000 engine calls failing to decide whether its own goal was
reachable, and never planned.

The contrast case, `degenerate_goal_predicate` (n=2), terminates the other way:
`queue_exhausted`, `frontier_remaining == 0`, and 3552 / 32 engine calls — far below budget.

**The fix follows from the label.** The code says it: "`max_nodes` is compute and may be
raised; the gate is quality and may not be widened." `frontier_remaining` spans 136 to 5135,
so raising the budget would likely decide some of these 7 and not all. That number is the
per-case measure of how far short each fell.

### One of the 2 degenerate cases is mislabelled, and the run log predicted it

`supshadow ar25/20260724 att1` reports `engine_calls=32`, `reachable_grids_evaluated=1`,
`frontier_remaining=0`. One reachable grid is the root. The engine produced **zero** novel
successors from any of 32 candidate actions. That is an inert engine, not a bad goal
predicate — and the label told the reader the goal was junk.

The same run's log names the mechanism in advance:

```
WINDOW VOCABULARY VIOLATION: actions [7] appear in this window and the in-model planner
cannot emit them. The induced engine will model transitions the planner never asks for,
and may be inert at the planning root even at cell_recall 1.0.
```

That is `ops/verifier_gaps.md:4708` GAP-6260, open, whose own evidence cites `ar25`. The
warning fired 4 times in supshadow and 5 in supwindow, **on ar25 cells only** in both.

## 5. The dynamics failures: a generalization gap, not a strict gate

By round label there are **11**; by attempt label 10 plus 1 caller relabel (below). Two facts
hold across all 11:

- **`heldout_threshold == 1.0` in 11/11.** The gate demands perfect held-out accuracy.
- **`selected_candidate_name == "loaded_world_model.py"` in 11/11.** Every rejection is of an
  engine **retained from disk**, never of a freshly induced candidate.

### The held-out block size is fully determined, and it is 8, not 4

`select_trusted_world_model` splits via `_split_prefix_heldout`
(`arc_world_model_trust_energy.py:382-392`): held-out is the **last `max(1, round(N/3))`
rows**, clamped to `N-1`, and `N == real_n` because `refinement_corpus` is the same full list
(REQ-6090 default OFF, `arc_llm_reinduction.py:1627-1634`). The rule predicts every recorded
accuracy exactly: `N=25 → 8` (0.5 = 4/8, 0.75 = 6/8), `N=214 → 71` (0.830986 = 59/71),
`N=234 → 78` (0.282051 = 22/78). **11 of 11 accuracies are exact multiples of 1/block.**

Once the block is right, the prefix accuracy is recoverable, and it changes the answer:

| Run | Game/seed | Att | real_n | block | **prefix acc** | **held-out acc** | gap | class |
|---|---|---|---|---|---|---|---|---|
| supshadow | tu93/20260724 | 0 | 25 | 8 | **1.0000** | 0.5000 | +0.50 | A |
| supshadow | tu93/20260725 | 0 | 25 | 8 | 0.8824 | 0.7500 | +0.13 | A |
| supshadow | tu93/20260726 | 0 | 25 | 8 | **1.0000** | 0.7500 | +0.25 | A |
| supshadow | ar25/20260726 | 0 | 25 | 8 | **1.0000** | 0.5000 | +0.50 | A |
| supwindow | tu93/20260724 | 0 | 25 | 8 | **1.0000** | 0.7500 | +0.25 | A |
| supwindow | tu93/20260726 | 0 | 25 | 8 | **1.0000** | 0.7500 | +0.25 | A |
| supwindow | tu93/20260725 | 0 | 25 | 8 | 0.0000 | 0.0000 | 0.00 | C |
| supwindow | tu93/20260726 | 3 | 214 | 71 | 0.8042 | 0.8310 | −0.03 | B |
| supwindow | ar25/20260724 | 2 | 234 | 78 | 0.3205 | 0.2821 | +0.04 | B |
| supwindow | tu93/20260724 | 1 | **1** | 1 | n/a | 0.0000 | n/a | D |
| supwindow | ar25/20260724 | 1 | **1** | 1 | n/a | 0.0000 | n/a | D |

**Class A — generalization gap (n=6, the largest class).** The engine is **perfect on every
row it was shown in 5 of 6 cases**, and wrong on 25–50% of the rows it was not shown. The gap
runs +0.13 to +0.50. This is **memorisation**, and rejecting it is the gate doing its job.

**Class B — uniform failure on a large corpus (n=2).** `real_n` 214 and 234. Prefix and
held-out accuracy are close (gap −0.03 and +0.04) and both poor. The engine is genuinely wrong
about the world, evenly.

**Class C — total failure (n=1).** 0 correct of 25. Nothing to interpret.

**Class D — the gate is vacuous (n=2).** `real_n == 1`. `_split_prefix_heldout` returns early
when `len(rows) < 2` and hands back **the same single row as both prefix and held-out**. There
is no held-out set. The gate compares the engine against a row it was trained on, and a
"held-out accuracy" of 0.0 carries no generalization information at all.

6 + 2 + 1 + 2 = 11.

**Is this one class or many? Many — but one dominates, and it is not the one the first draft
named.** The largest class is a real generalization failure. The gate is right in A, B and C
(9 of 11) and structurally meaningless in D (2 of 11).

**Trap for the next reader: `real_mismatches_n` saturates at 8.** `WorldModelVerifier.score`
takes `max_mismatch: int = 8` (`arc_executable_world_model.py:1832`) and stops collecting. The
`ar25/20260724 att2` digest shows `real_mismatches_n: 8` where the true count is **162**. The
true count is `real_n - real_n_correct`. The source comment says it: "a 40-row total wipeout
and an 8-row one are indistinguishable."

### A second masking layer, at the caller

For `supshadow tu93/20260726 att0` the two labels disagree:

- attempt `skipped = world_model_accuracy_below_threshold`
- round `skipped = heldout_transition_verification_failed`

`arc_competition_agent.py:8355` sets the attempt label from its own separate gate
(`_gate_value < 0.5`) and overwrites what the round diagnosed. Both are dynamics failures, so
the class survives — but the row-level tally the project reads can still disagree with the
round record beneath it. This is REQ-6710's defect shape, one layer up.

## 6. `llm_channels`: 97.6% of decoded characters are reasoning

n = 10 rows, 42 completions, 8,661,106 raw characters.

| Channel | Characters | Share of raw |
|---|---|---|
| `chars_reasoning` | 8,456,727 | **97.64%** |
| `chars_final` | 203,623 | **2.35%** |
| unaccounted | 756 | 0.01% |

Per row the split is stable: reasoning takes 96.86% to 98.23% across all 10 rows. Only the
final channel becomes code.

- `reasoning_only` fired **2 of 42 completions**. Those two produced reasoning and no answer.
- `both_channels_empty` is 0 in 10/10.

Pooled `tokens_predicted` is 2,580,979, so raw output runs at 3.36 characters per predicted
token.

**What this does not say.** These are **characters, not tokens**, and the commit says why: a
per-channel token count needs a tokenizer call the request never made. Reasoning text and code
do not tokenize at the same rate, so **the token share is not 97.6%** and I have not measured
it. I have also not measured whether a shorter reasoning channel would change induction
quality. This is a measurement of where the characters go, not evidence that spending them
there is wrong.

## 7. Comparing like things: what I pooled, and what I did not

**supon is excluded entirely.** It predates the label fix. Pooling it produces the phantom 6
planning failures.

**supshadow and supwindow are not behaviourally equivalent.** supwindow's supervisor fired 2
redirects in every one of its 4 cells; supshadow's fired 0 in all 6.

| Cause (round-level) | supshadow | supwindow |
|---|---|---|
| PLANNED (this round) | 3 | 2 |
| `heldout_transition_verification_failed` | 4 | 7 |
| `goal_unreached_within_budget` | 4 | 3 |
| `degenerate_goal_predicate` | 2 | 0 |
| **attempts with a record** | **13** | **12** |

Dynamics share is 4/13 in supshadow and 7/12 in supwindow. Fisher exact **p = 0.238**. At this
n the difference is not distinguishable from noise, so I pool **for the cause mix** and say so.

**Where pooling does not hold, and I say so rather than hide it.** The §5 taxonomy is a
different question from the cause mix, and the Fisher licence does not extend to it:

- Class A (n=6) appears in **both runs** (4 shadow, 2 window) and **both games**. Pooling is
  safe here, and A is the class the conclusion rests on.
- Classes B, C and D (n=5) are **entirely supwindow**, and 4 of those 5 are non-first attempts
  within their cell. supwindow runs 3.25 induction attempts per cell against supshadow's 2.17.
  A redirect that forces an extra induction on a changed transition set is exactly the thing
  that could manufacture a `real_n=1` corpus or a 214-row one. **I do not claim B, C or D
  generalises beyond supwindow.**

**Games.** tu93 and ar25 differ in dynamics share (8/17 vs 3/8), Fisher exact **p = 1.000** —
no signal at this n. They differ **qualitatively**: the vocabulary violation and the
root-inert engine occur on ar25 only. That is why I report them per game.

## 8. What I could not determine

- **supon's 6 `no_reachable_plan_after_refinement` cannot be split.** The field that would
  split them is `induction_round_records`, which supon does not carry. Only a re-run under the
  current code recovers them.
- **The exception attempt has no round record.** `supwindow tu93/20260725` att1 raised
  `ValueError('not enough values to unpack (expected 2, got 1)')` inside `_rle_delta_compact`
  (`arc_executable_world_model.py:2769`) called from `_transitions_block` (:2962) during
  `induce_prompt`. It raised **before** the LLM was called and before any round row was
  appended, so `_induction_round_records` drops it via the `if not rounds and not ces` guard
  (`arc_scored_path_lever_harness.py:687`). The evidence survives only in
  `induction_tracebacks`. **The instrument is blind to pre-round exceptions.**
- **Whether raising `max_nodes` would decide the 7 undecided goals.** `frontier_remaining`
  bounds how far short each fell but does not predict the cost of finishing.
- **Whether retained engines are worse than fresh ones.** All 11 rejections were retained
  `loaded_world_model.py` engines, but the records do not show how often a retained engine was
  **accepted**, so I cannot compute a rejection rate for retention. This is the single most
  useful missing number in the corpus.
- **The token split behind the 97.6% character split.** Not measured, and not inferable.
- **Whether Classes B, C and D exist outside supwindow.** n=5, all one run, all one config.

## 9. What the live agent actually fails at

In one sentence: **the agent is not failing to plan; it is failing to generalise a world model
past the rows it was shown, and then failing to decide whether its own goal is reachable — and
it spends 97.6% of its decoded characters on reasoning text that reaches neither problem.**

Ranked by the 21 bounded-reinduction rejections (denominator per §2; 15 attempts produced no
plan at all):

1. **Dynamics verification — 11 of 21 (52%).** The largest sub-class is a **generalization
   gap**: 5 of 6 engines are perfect on every transition they were shown and wrong on a
   quarter to a half of those they were not. Two more rejections rest on a corpus of one row,
   where the gate is structurally vacuous. Every rejection is of a retained on-disk engine.
2. **Goal reachability undecided — 7 of 21 (33%).** All budget-exhausted at exactly
   `max_nodes`, all with a non-empty frontier, all on a world model that had just passed
   verification. A compute limit reported honestly, not a goal defect.
3. **Goal proven unreachable — 2 of 21 (10%).** One of the two is really a root-inert engine
   (GAP-6260), mislabelled as a goal defect.
4. **Exception — 1 of 21 (5%).** Raised in prompt construction, before any LLM call.
5. **Planning — 0.** The planner was reached in 5 of 25 rounds and failed in none of them.

The cheapest next measurements, in order:

- **Record how often a retained engine is accepted.** Every rejection here is a retained
  engine, and without the acceptance count that observation cannot become a rate.
- **Re-run the goal probe at a raised `max_nodes` on the 7 undecided cases.**
  `frontier_remaining` already says which are close.
- **Refuse the trust gate when `real_n < 2`.** In that regime `_split_prefix_heldout` returns
  the same row as prefix and held-out, so the verdict carries no generalization information.
  Two of 11 rejections were decided this way.
- **Give the pre-round exception path a round record**, so the instrument stops dropping the
  failures that happen earliest.

## 10. What adversarial review corrected in the first draft

Recorded rather than quietly patched, per the error-lifecycle discipline.

1. **The held-out block size.** The draft said the rule "could not be determined" and asked
   for a new `heldout_n` field. It is fully determined by `_split_prefix_heldout`, and the
   draft's guess of 4 rows was wrong — it is 8 for a 25-row corpus. The proposed new field
   would have bought nothing.
2. **The largest dynamics class.** With the block wrong, the draft called Class A "a threshold
   artefact, not a model defect". With the block right, the prefix/held-out split shows the
   opposite: 5 of 6 engines are prefix-perfect and fail only on unseen rows. **The draft's
   headline for its largest class was inverted by its own data.**
3. **The denominator.** The draft computed shares against "20 failing attempts". 6 attempts
   are both `skipped` and `planned`, so 20 was neither the rejection count (21) nor the
   no-plan count (15). This is the exact trap the brief warned about.
4. **A denominator on the planner claim.** "0 failures in 25 rounds" implied 0/25. The planner
   was reached 5 times; the real evidence is 0/5.
5. **Pooling the taxonomy.** The draft declared it would not pool anything a redirect could
   move, then pooled the taxonomy. Classes B, C and D are supwindow-only.
6. **Character/token slippage** in the headline, and an unsourced pid.

**Provenance of this file, recorded because it is itself an instance of a known hazard.**
This note was written in a checkout shared with concurrently-committing agents. Its
uncorrected first draft was swept into `518b10d2b2` (a conductor `git add -A` commit about a
blinded row audit). The corrected text was then swept into `df85981bd7` (another agent's
commit about KAN receipts) from the shared index. Neither commit message describes this note.
A reader tracing its history should read this section, not `git log`. Both drafts are left in
history per never-prune. See `[[incident_concurrent_agents_destroy_work]]` and
`[[incident_conductor_git_add_A_determination_drops]]`.

## Cross-references

- REQ-ARC-WMTE-6710 — `openspec/capabilities/arc-world-model-trust-energy/spec.md`
- `288ea485f9` — the label fix that makes supon incomparable
- `scripts/arc_scored_path_lever_harness.py:670` — `_induction_round_records`
- `python/carnot/agentic/arc_llm_reinduction.py:1013-1040` — three-way goal termination
- `python/carnot/agentic/arc_world_model_trust_energy.py:382-392` — `_split_prefix_heldout`
- `python/carnot/agentic/arc_executable_world_model.py:1832` — `max_mismatch = 8`
- `python/carnot/agentic/arc_competition_agent.py:7925-7935`, `:8355` — the fall-through to a
  plan, and the caller-side relabel
- `ops/verifier_gaps.md:4708` GAP-6260 — root-liveness, open
