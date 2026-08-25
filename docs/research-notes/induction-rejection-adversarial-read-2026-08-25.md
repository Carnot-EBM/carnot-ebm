# Adversarial read of the induction-rejection framing (2026-08-25)

Written by an agent told to BREAK the outer loop's framing, not to confirm it. It
did not see the measurement agent's report. Preserved here because it refuted three
claims the outer loop had already stated to the operator.

Its companion is `induction-token-accounting-and-rejection-2026-08-25.md`. Read that
one for the measurement; read this one for what the measurement could have gotten
wrong.

The strongest point it makes is not a number. It is that `induction_skipped` is about
half composed of a default string, and the per-round record that would disambiguate it
was computed on every run and then dropped at row-writing time. So the field being used
to diagnose the generator could not resolve the category it was being read for.

---

# Adversarial read: why are induced world models rejected?

Independent investigation, 2026-08-25. All numbers below were derived from primary
evidence on disk. No GPU job was run and no server was launched.

**Clone trap, stated explicitly.** `abpin/` and `headwt/` under
`/home/ianblenke/.claude/jobs/ad0c053d/tmp/` are repo clones. Every path in this
report is an explicit run directory (`seedsweep/`, `supon/`, `retention_run/`) or a
file under `/home/ianblenke/github.com/ianblenke/carnot/`. No recursive sweep of the
job tmp directory was performed. Confirmed independently: `find abpin headwt -name
'wm_*.py'` returns 0 files, so the archived-program corpus could not have been
inflated by them either way.

**Evidence base, with its denominator.** 13 COMPLETED live cells: 12 from
`seedsweep/rows.json` (tu93 and ar25, 3 seeds, 2 byte-identical arms; stopped before
cd82) and 1 from `supon/rows.json`. The supon harness (pid 218052) was still running
when this was written, so its numbers will grow. `retention_run/` has no `rows.json`
at all — 2 archived programs and 2 server completions, no row-level data. Archived
programs: 40 files. Server completions: 44 across the three logs.

---

## 1. The rejection rate. Verdict: REFUTED as stated.

### Where "9 of 10" came from

It is not an analysis result. It is a sentence in the launcher of the very run that
is now being interpreted, `supon/run.sh`, written before that run produced anything:

> "Roughly 9 of 10 cells in the sweep rejected their induced model outright
> (world_model_accuracy_below_threshold, degenerate_goal_predicate,
> no_reachable_plan_after_refinement, goal_unreached_within_budget)."

Note the unit in the original: **cells**. The claim under test silently restates it as
**induced world models**. Those are different denominators and they differ by a factor
of two.

### What the primary data says

`induction_skipped` is built by `scripts/arc_scored_path_lever_harness.py:799` as
`Counter(a.get("skipped") for a in atts if a.get("skipped"))` — a tally over ATTEMPT
dicts. Three denominators are available and they are not close:

| Denominator | Value | What it means |
|---|---|---|
| cells with >= 1 skip | 12 / 13 = **92.3%** | "somewhere in this cell, something was skipped" |
| attempts carrying a skip string | 18 / 27 = **66.7%** | |
| **archived programs** (the literal "induced world models") | 18 / 37 = **48.6%** | |
| **attempts that ended with no plan** | 13 / 27 = **48.1%** | the honest "this induction yielded nothing" rate |

Seedsweep alone (12 complete cells, the slice the "9 of 10" was written about):
11/12 = 91.7% of cells, 15/24 = 62.5% of attempts, 15/33 = 45.5% of programs.

**11/12 is the origin of "9 of 10."** It is the rate of cells in which at least one
skip occurred. Since a cell runs 1-3 attempts, a cell-level "any" rate approaches 100%
mechanically as attempts per cell rise, and carries almost no information about the
generator.

### A second, independent reason the cell-level rate cannot mean what it is used for

`skipped` and `planned` are NOT mutually exclusive on an attempt. Proof from the data
alone, no code reading required: across seedsweep, attempts = 24, skipped = 15,
planned = 13. 15 + 13 = 28 > 24, so at least 4 attempts carry both. Pooled with supon
the floor is 5. Three individual rows show it outright, e.g. `ar25 / 20260726 /
S_llmon`: attempts = 1, skipped = 1, `induction_planned` = 1.

The mechanism is at `arc_competition_agent.py:7900` — when the bounded-refinement loop
does not plan, execution **falls through** to the reward-machine probe, active-probe,
and plain single-shot induction paths, any of which can set `attempt["planned"] = True`
on the same dict that already carries `skipped`. So a skip string records "the
refinement stage did not produce the plan", not "the agent got nothing".

**Verdict: REFUTED.** The rejection rate is real but is roughly 45-48% per induced
model, not 90%. The 90% figure is a cell-level "any skip occurred" rate, originating
in a launcher comment, restated as a per-model rate.

---

## 2. The token accounting. Verdict: REFUTED on method — and the true waste is larger.

### `n_decoded` is a running counter, not a completion total

`slot print_timing: ... n_decoded = N` is emitted roughly every 3 seconds **during**
decode, with N rising monotonically within one request. The completion summary is a
different line: `eval time = X ms / N tokens`, emitted once per request.

| log | `n_decoded` progress lines | completed requests (`eval time`) |
|---|---|---|
| seedsweep p8995 | 25,872 | 36 |
| supon p8997 | 4,365 | 6 |
| retention_run p8994 | 1,635 | 2 |

Medianing 25,872 samples of a rising counter answers "what was the counter at a
randomly chosen moment", which is about half the final value:

| | naive median of `n_decoded` lines | TRUE median per completion |
|---|---|---|
| seedsweep | 35,412 | **64,606** |
| supon | 28,642 | **58,104** |
| retention_run | 41,988 | **77,188** |

The claim's "~40,000 tokens each" sits exactly in the naive column. The true per-call
decode is **1.6x-1.8x higher**. The duration claim is wrong the same way: true
per-request decode is **19.0 to 46.0 minutes, median 35.7**, not "26-34".

Independent cross-check that closes the loop: the rows' own `llm.tokens_predicted` is
summed from the server's `timings.predicted_n`, and 2,386,224 / 36 responses = 66,284
mean — matching the `eval time` lines, not the progress lines.

### Is the archived `.py` the whole model output? No.

`_write_world_model(game, code)` receives `code`, the EXTRACTED block, not the
response. Think mode has been ON by default since 2026-08-08
(`ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT = "1"`), and
`arc_executable_world_model.py:6576` folds the reasoning channel back as
`<think>…</think>` + final. So the response is reasoning + prose + fence + code, and
only the last is archived. Comparing bytes-on-disk to tokens-decoded is a category
error — but not the one named in the claim, and the direction is worse than stated:

- 40 archived programs, **195,416 bytes total**, median 4,478, max 13,346.
- 2,891,259 tokens decoded across the three runs.
- At ~3.5 bytes/token for code that is ~56k code tokens, i.e. **~2% of decode reaches
  the archive**, not ~10%.

The "at most 13,346 bytes" figure is **CONFIRMED exactly** (`ar25
wm_20260824T143429_499711`).

### But the residual is not measurable as "reasoning"

`last_raw_completion`, `last_final_content` and `last_reasoning_content` exist on the
proposer and are documented as "PURE OBSERVATION" — and they are **in-memory only**.
No run persists them. I checked the one corpus whose comments advertise raw
completions, `results/arc_induce_bestofn_20260731/bestofn_scored.json`: no `<think>`,
no `reasoning_content`, no `raw_completion` anywhere in it.

So "~90% of generated tokens are waste, plausibly reasoning tokens" is an **inference
from an absence**, not a measurement. The residual pools at least four distinct
things: reasoning; prose outside the fence; discarded draft code blocks; and whole
responses that failed extraction.

### Task <-> call mapping

Asked directly, so answered directly. **1:1 confirmed** for logged tasks: 36 tasks =
36 `release: stop processing` = 36 `eval time` = 36 harness `responses` in seedsweep.
Nothing is unaccounted; the cd82 cell that started at 20:14:50Z contributed zero
completions before the 20:37Z stop.

Two mappings that are NOT 1:1, and that any future token analysis must handle:

- `generate_calls` (33) < `responses` (36). `generate()` retries internally up to
  `tries`; three cells retried once. Those 3 completions (~200k tokens) produced no
  program at all.
- In the split-induce path, one archived file is `_combine_world_model(eng, goal)` — a
  concatenation of **two** completions. It did not fire in these runs (gen == archived
  in every row), but a general bytes-per-call rule that ignores it is wrong.

### One more fact the "waste" framing needs

Only **2 of 36** completions stopped at `stop_type == "limit"`. 34 of 36 ended at EOS.
The model is not being truncated into rambling; it chooses to emit ~65k tokens.
`prompt_truncated` and `errors` are zero on every row.

**Verdict: REFUTED on method.** The specific numbers (40k tokens, 26-34 min) come from
medianing a progress counter and are ~1.7x low. The bytes-vs-tokens comparison is a
category error whose true ratio is ~2%, not ~10%. Whether the residual is reasoning is
**CANNOT_DETERMINE** from any artifact on disk.

---

## 3. Is rejection the binding constraint? Verdict: CANNOT_DETERMINE, and the named mechanism is wrong.

### What actually bound these episodes

Every one of the 13 cells spent 384-392 of its 400 actions. The binding constraint
within a game was the **action budget**, not wall clock. No induce timeout fired
(`CARNOT_ARC_INDUCE_TIMEOUT=3600`, worst request 46 min = 2,760s).

Meanwhile the wall-clock split is extraordinary:

```
POOLED  wall = 88,340s   llm_wall = 87,425s   =  98.96%
non-LLM agent total, 13 cells = 915 seconds (~70s per cell)
```

The entire non-LLM agent — exploration, graph search, planning, hazard pruning, the
trajectory supervisor — costs about seventy seconds per game. So the framing's premise
that induction dominates cost is **CONFIRMED, and understated**.

### Why that does not make token-cutting a lever on levels

The harness records the scored bound itself, in `budget_semantics`:

> "400 is the SHIPPED agent's self-imposed per-game MAX_ACTIONS loop guard ... NOT an
> eval-imposed bound — **the eval's bound is <=12h wall clock across all games**."

So cutting induction tokens buys **games covered inside 12h**. It does not buy actions
within a game, and induction firings are triggered by stall and level-up events, not
by a time budget. Levels-per-game would be unchanged by construction.

### A concrete way the claim is true and the agent still solves nothing more

Cut the tokens by suppressing reasoning, and you land on the arm this project has
already measured. `experiment_6199_gemma_think_mode_ab.json`, n = 10 games:

| metric | no_think | think |
|---|---|---|
| mean heldout accuracy | 0.100 | **0.3727** |
| mean goal-predicate accuracy | 0.655 | **0.810** |
| mean levelup positive recall | **0.200** | 0.100 |

Induction quality is 3.7x better with reasoning on. So the intervention trades
per-game model quality for games-covered, and the net on levels is genuinely unknown.

Worth naming: that artifact's own `honest_verdict` is
`"complete: no_think_higher_winrecognition_1_to_2"` — it headlines the one metric where
no_think won, while the source comment at `arc_executable_world_model.py:3908-3916`
headlines the two where think won. Both are honest readings of the same run. It is not
a settled question.

### The confound that would fake a positive result

`induction_reasons` shows `level_up_reinduction` firing once **per level-up**. So
`induction_attempts` and `induction_planned` are partly **caused by** levels. In
seedsweep every tu93 cell reaching lv=2 has planned >= 2 and every lv=1 cell has
planned <= 1 — which reads as "successful induction causes levels" and is at least
partly the arrow pointing the other way. Any regression of induction success on levels
is circular unless it conditions on this.

### The noise floor forbids the obvious test

S and S_replicate are byte-identical configurations. On tu93 they differed by a **full
level in 2 of 3 seed pairs** (s25: 1 vs 2; s26: 1 vs 2). A one-level effect at n=3 is
indistinguishable from run-to-run sampling noise.

**Verdict: CANNOT_DETERMINE.** Wall-clock dominance is confirmed and understated. The
causal step from "fewer tokens" to "more levels" is not supported, runs through
games-covered rather than levels-per-game, and points at an intervention whose only
measurement shows it degrading induction quality 3.7x.

---

## 4. What nobody is measuring

### 4a. The per-round record is computed, attached, and then thrown away

`arc_llm_reinduction.py` builds a `row` dict **per refinement round**, carrying that
round's `skipped`, `heldout_accuracy`, `counterexample`, `plan_reaches_goal`. Those
rounds reach the attempt dict at `arc_competition_agent.py:7723`
(`"refinement_rounds": list(outcome.rounds)`) and `:7735`
(`"counterexamples": ...`).

`arc_scored_path_lever_harness.py:796-843` then uses `atts` **only to count
categories** and never stores the list. Every per-round diagnosis is discarded at
row-writing time. This costs no GPU to fix.

### 4b. Half of all "reasons" is a fall-through default that can hide a dynamics failure

`arc_llm_reinduction.py:1649` initialises `skipped = "no_reachable_plan_after_refinement"`
before any round runs. It is cleared to `""` only on planned returns. Two genuinely
diagnosed causes write **only** the per-round record and never the outer variable:

- `:1861` `row["skipped"] = "heldout_transition_verification_failed"` — the dynamics
  model failed held-out verification
- `:2208` `row["skipped"] = "selection_or_planning_exception"`

Compare `:2030-2031` and `:2191-2192`, which set both. So an attempt whose every round
failed held-out **dynamics** verification is reported at row level as
"no_reachable_plan_after_refinement" — a planning label. This is the pattern-narrower-
than-its-concept class named in CLAUDE.md, sitting inside the exact field the outer
loop is using to diagnose the generator. I cannot say how often it fired in these runs,
because 4a threw the evidence away — which is the point.

### 4c. The reasoning/final split is never persisted

`last_reasoning_content` and `last_final_content` already exist and already carry the
answer to the entire token-waste hypothesis. They live and die in memory.

### 4d. The think-mode default rests on a retired generator

`exp6199` ran on `gemma-4-31B-it-qat`. The live pin has been Qwen3.8-27B since
2026-08-16 — a reasoning model that writes far longer completions. **Nobody has A/B'd
think on/off, or a thinking-budget cap, for the generator that actually ships.**

---

## My independent answer: why are induced world models rejected?

**n = 18 skip records over 27 attempts across 13 completed live cells.**

| n | share | reason |
|---|---|---|
| 9 | 50.0% | `no_reachable_plan_after_refinement` — **a fall-through default, not a diagnosis** |
| 4 | 22.2% | `degenerate_goal_predicate` |
| 3 | 16.7% | `goal_unreached_within_budget` |
| 1 | 5.6% | `world_model_accuracy_below_threshold` |
| 1 | 5.6% | `hidden_state_trust_below_threshold` |

**Dynamics quality is the named cause exactly once in eighteen.** Seven of eighteen
name the GOAL predicate or reachability explicitly, and the 50% default is a planning
label.

Corroborated independently at larger n, on a different generator, from a corpus
nobody cited in this discussion —
`results/arc_induce_bestofn_20260731/bestofn_scored.json`, `goal_gate_failure_census`,
stall arm, 40 candidates:

```
degenerate_goal_predicate   17
goal_unreached_within_depth  7
goal_unreached_within_budget 5
not_evaluated                8
gate_timeout                 1
satisfiable                  2     <-- 2 of 40
```

**Two candidates in forty had a satisfiable goal.**

The archived programs corroborate this qualitatively. They are not rambles or stubs:
median 4,478 bytes, ~150-200 lines of structured numpy — sprite templates with
per-pixel hole maps, run-length erosion rules with documented tie-breaks — at a median
11% comment-and-docstring share. In that era 8 of 40 candidates were syntactically
unrunnable; in these runs only 3 of 36 responses failed extraction (8%).

**So: induced world models are rejected because the agent cannot prove a plan to the
induced GOAL — not because the generator writes bad dynamics code.** The failing seam
is goal induction plus bounded reachability proof, one stage downstream of the
generator.

---

## The strongest reason the outer loop's framing is wrong

It makes the generator the suspect on evidence that does not implicate it, and would
spend the fix budget on the one stage the data says is working.

The chain is: reject-rate is high (measured at the wrong denominator, 2x too high) ->
tokens are huge (measured off a progress counter, 1.7x too low) -> the gap between
tokens and archived bytes is waste (unmeasurable — the raw completion is never saved)
-> therefore cut tokens. Every link is either mis-measured or unfalsifiable from the
artifacts, and the conclusion points at reasoning, which is the only intervention this
project has measured and found to **improve** induction quality 3.7x.

Meanwhile the actual rejection reasons say the goal predicate fails 16 of 18 times
here and 38 of 40 times in the larger corpus, and the archived code is competent.

The deepest problem is not any single number. It is that the field being used to
diagnose the generator, `induction_skipped`, is 50% composed of a default string that
provably absorbs a dynamics-verification failure, and the per-round record that would
disambiguate it is computed on every run and deliberately not written down. **The
outer loop is reasoning about a category that its own instrumentation cannot resolve.**

---

## The measurement that would settle it

**Zero GPU cost, do this first.** Emit `attempt["refinement_rounds"]` and
`attempt["counterexamples"]` into the harness row. The data already exists in memory
on every run and is dropped at `arc_scored_path_lever_harness.py:796-843`. That alone
turns "why are models rejected" from a guess into a count: per round, did held-out
dynamics verification fail, or was the goal unreachable? Pair it with widening
`arc_llm_reinduction.py:1861` and `:2208` to set the outer `skipped` variable, so the
default stops masking two diagnosed causes. Add the incident input as a regression
test: an attempt whose every round set `heldout_transition_verification_failed` must
not report `no_reachable_plan_after_refinement`.

**Near-zero cost, do this second.** Persist `len(last_reasoning_content)` and
`len(last_final_content)` per completion. Two fields that already exist and are
already documented as observation-only. One run then converts the entire token-waste
hypothesis from inference to measurement.

**Only then, and only if the split says reasoning dominates:** A/B think on / off /
capped for **Qwen3.8-27B**, paired against exp6199's metric set (heldout accuracy,
goal-predicate accuracy, levelup recall), with S_replicate cells included so the
one-level noise floor is visible in the same run. Score it on games-covered-per-12h as
well as levels-per-game, because those are the two halves the current framing collapses
into one.
