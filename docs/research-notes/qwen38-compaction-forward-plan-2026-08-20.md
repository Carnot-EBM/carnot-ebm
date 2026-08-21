# Qwen3.8 and Compaction: Forward Plan (2026-08-20)

Read-only strategy assessment. No code changes were made. The operator constraint
is fixed: Qwen3.8-27B stays the generator. This note ranks the next moves to make
Qwen3.8 and context compaction work on the scored path.

Sources verified for this note:

- Offline play v3: `/home/ianblenke/.claude/jobs/ad0c053d/tmp/offplay_out3/`
  (`offline_play.json`, `lever_rows.json`); v2 is byte-similar.
- Compaction pilot: `results/experiment_6473_tool_loop_compaction_pilot_ab.json`
  and `scripts/experiments/experiment_6473_tool_loop_compaction_ab_analyze.py`.
- Equalized holdout A/B (in flight): `results/holdout_equalized_ab_20260820/`.
- Code: `python/carnot/agentic/arc_executable_world_model.py`,
  `arc_competition_agent.py`, `arc_induction_tool_loop.py`,
  `arc_induction_compact_state.py`, `arc_llm_reinduction.py`,
  `scripts/kaggle/submission_kernel/main.py`,
  `scripts/arc_scored_path_lever_harness.py`.

## 1. Corrections to the briefing

Three findings contradict or sharpen the briefing. They change the plan, so they
come first.

### 1.1 The token/exception puzzle is ONE event, not two

The briefing treats "enormous generations" and "counted induction throws" as two
facts that sit oddly together. They are the same event. The chain:

- `induction_attempts_llm_reached` is keyed on the attempt's `model_specs`
  field (`scripts/arc_scored_path_lever_harness.py:747-749`). `_induce_and_plan`
  initialises it to `offline_dsl_induction_no_llm` and overwrites it only when
  an LLM outcome COMES BACK (`arc_competition_agent.py:7253`).
- The generate calls run INSIDE the one counted attempt. The attempt's outermost
  handler catches the exception AFTER generation and records only the word
  "exception" (`arc_competition_agent.py:8254-8275`; the message/traceback
  capture shipped 2026-08-19, after the v3 run).
- The instrumented proposer's `errors` counter is 0 on all three games. So
  `generate()` returned normally every time. The raise happened downstream, in
  parse/verify/plan code, after the tokens were already spent.

The harness's own comment warns about exactly this trap: "`skipped:
proposer_failed` means the LLM ran and its output was unusable, NOT that it
never ran" (`arc_scored_path_lever_harness.py:744-746`). So there is no hidden
generation path. The counted induction path IS the spender.

### 1.2 The scored path very likely cannot run the tool loop AT ALL today

This is stronger than the briefing's framing ("compaction can act on almost
nothing"). On the scored backend the tool loop is structurally inert:

- The vLLM launch args (`arc_executable_world_model.py:6675-6696`) contain no
  `--enable-auto-tool-choice` / `--tool-call-parser` flags. vLLM only lifts
  tool calls into `message.tool_calls` when launched with tool parsing enabled.
  Without it, the model's tool-call emission stays raw text in `content`.
- The loop consumes ONLY server-lifted `tool_calls`
  (`arc_induction_tool_loop.py:566`). Unparsed tool-call text is counted, never
  executed (`arc_induction_tool_loop.py:285-287, 646`).
- The loop's per-turn think bound, `thinking_budget_tokens`
  (`arc_induction_tool_loop.py:231`), is a llama.cpp server field. vLLM does
  not implement it. Whether vLLM ignores it or rejects the request is
  unverified.

Consequence: `CARNOT_ARC_INDUCE_TOOL_LOOP="repair"`
(`scripts/kaggle/submission_kernel/main.py:196`) routes catastrophic-draw
re-draws into a loop that cannot execute a single tool on the scored backend.
Each repair attempt burns decode and falls back
(`arc_competition_agent.py:8375-8410`). Compaction acts on nothing there — not
because repair fires rarely, but because the transport under it is broken.
Verify with one live request against a local vLLM server before building on
this conclusion.

### 1.3 G_M failure attribution in the briefing is wrong

The briefing says G_M fails "on the three cells that compacted twice". The
artifact says: failing cells are cd82 (2 compactions), lp85 (2), and sk48
(ONE). tr87 compacted twice and PASSED. So compaction count does not predict
failure. Section 5 gives the real mechanism.

Also one more latent hazard found while reading: the tool loop's per-turn
`max_tokens` binds to `proposer.max_tokens`
(`arc_induction_tool_loop.py:219`). The pilot's 4096-per-turn cap came from the
pilot harness's proposer construction, not from the loop. Under the scored env
(`CARNOT_ARC_INDUCE_MAX_TOKENS=131072`) each tool turn could decode up to 131k
tokens. Any scored flip must decouple this first.

## 2. Question 1: the budget pathology, attributed

The mechanism is now fully accounted. No component is mysterious.

Measured, offline play v3 (real `E3AgentPolicy`, vLLM, Blackwell, k=1):

| game | wall_s | llm_wall_s | LLM share | generate calls | responses | tokens out | tokens/response | graph nodes |
|------|--------|-----------|-----------|----------------|-----------|------------|-----------------|-------------|
| ar25 | 5456   | 5447      | 99.8%     | 2              | 5         | 310,211    | 62,042          | 301         |
| tr87 | 2230   | 2229      | 99.9%     | 2              | 3         | 128,090    | 42,697          | 231         |
| tu93 | 8687   | 8680      | 99.9%     | 4              | 7         | 485,383    | 69,340          | 53          |

The causal chain, each link with its own evidence:

1. **Think mode is ON by default since 2026-08-08** (`ARC_LIVE_GENERATOR_THINK_
   SCORED_DEFAULT = "1"`, `arc_executable_world_model.py:3832`; flipped on
   exp6199's induction-quality evidence).
2. **Qwen3.8's reasoning is INLINE.** It has no separate `reasoning_content`
   channel on the completion path, so think tokens consume the answer window
   (`scripts/experiments/experiment_6440_qwen38_generator_h2h_arm.py` header;
   measured: tu93 induce generated 41,613 tokens, ~40k of them reasoning, then
   terminated on its own).
3. **One induction naturally decodes 36k-131k tokens.** Nine real inductions:
   median 62,490, max completed 83,444, right-censored draws to >=100,988
   (`arc_executable_world_model.py:3964-3986`). The average response in the v3
   run (62,042 on ar25) matches the median exactly.
4. **`max_tokens` is 131,072** (raised 2026-08-16 so the think stream does not
   truncate; `submission_kernel/main.py:580-597`).
5. **The retry ladder multiplies the cost.** `tries: int = 3`
   (`arc_executable_world_model.py:5878`). `responses > generate_calls` in
   every row is this ladder: each retry is a fresh ~62k-token think.
6. **One attempt runs several stages.** The stall path runs the bounded
   refinement loop (default ON, `CARNOT_ARC_STALL_REFACTOR_LOOP != "0"`,
   `arc_competition_agent.py:7702`), then the plain combined induce, then the
   split-induce fallback. Each stage issues its own generate call(s).
7. **The attempt then throws, after generation** (Section 1.1). Zero usable
   engines came back from ~924k tokens across three games.
8. **Timeout and cap are inconsistent.** `CARNOT_ARC_INDUCE_TIMEOUT=2400`
   cannot buy 131,072 tokens at ANY measured concurrency, including k=1
   (`arc_executable_world_model.py:3997-4001`, which names this an untaken
   operator decision). At k=1 (~59 tok/s) no timeout fired in v3; at scored
   k=8 (~40 tok/s/stream) any draw past ~96k tokens dies at 2400s and the full
   window is wasted.

At the game level: one stall fires once per game, the LLM eats 99.8%+ of the
wall clock, produces nothing, and the search runs on the leftovers. tu93's 53
nodes in 8,687s is not a search problem. Search got roughly 7 seconds.

The one remaining unknown is the exception's identity. The capture shipped
(`arc_competition_agent.py:8263-8275`; harness rows now carry
`induction_exceptions` and `induction_tracebacks`,
`arc_scored_path_lever_harness.py:727-735`). The offline-play v4 re-run in
flight will name it. Nothing in this plan waits on it except the bug fix
itself.

## 3. Ranked moves

Ranked by expected value per unit of effort.

### Move 1 — Close the induction spend envelope (do first, mostly env-level)

**Change.**
(a) Read the v4 traceback and fix the throw. Until then every induction is a
pure loss at any budget.
(b) Set `CARNOT_ARC_STALL_REFACTOR_LOOP=0` on the scored path. This removes the
refinement-loop stage from stall inductions, leaving the plain single-shot.
The text-refactor round is measured-destructive on held-out accuracy (0
improvements in 84 cells; `arc_llm_reinduction.py:1223-1229`), and v3 shows the
extra stage only multiplies 62k-token draws.
(c) Cut `tries` 3 -> 1 for the induce ladder under a reasoning generator. The
ladder was designed when a call cost ~4k tokens. A retry now costs ~26 minutes
at scored k=8 rates. One-line default change plus an env override.
(d) Make timeout and cap consistent. Either cap `CARNOT_ARC_INDUCE_MAX_TOKENS`
near what the timeout buys (~90k at k=8), or raise the timeout to ~3300s as the
docstring computes. Prefer the cap: the censored tail (>=100k tokens) has never
produced a usable engine, and a longer window makes each failure dearer.
(e) Add a per-game LLM wall budget in the policy (small counter): after N
seconds of `llm_wall_s` in one game, stop inducing and run the explorer. tu93
spent 8,680s of LLM for zero engines; a 12h eval across all games cannot absorb
that per game.

**Cost.** (a) unknown until v4, likely small. (b)+(d) env lines. (c) one-line
default. (e) ~20 lines.

**Confirm.** Re-run offline play on ar25/tr87/tu93. Expect: LLM share well
under 50%, at least one attempt returning a loadable engine (or a fast honest
failure), nodes_total up on tu93.

**Abandon if.** Induction quality drops below the v3 baseline. It cannot: the
v3 baseline is zero usable engines.

### Move 2 — Make the tool loop primary, in two gated steps

This is the induction-quality play AND the only way compaction matters. The
early exp6474 evidence points this way, but the scored transport cannot run the
loop today (Section 1.2). So the flip has a prerequisite step.

**Step A (prerequisite): make the loop runnable over vLLM.**
- Add `--enable-auto-tool-choice --tool-call-parser <hermes/qwen>` to the vLLM
  launch args (`arc_executable_world_model.py:6675-6696`). Verify with one
  local request that `message.tool_calls` comes back populated.
- Decouple the per-turn cap: a new env (default 4096) for the loop's
  `max_tokens` instead of `proposer.max_tokens`
  (`arc_induction_tool_loop.py:219`).
- Decide the per-turn think bound. `thinking_budget_tokens` is inert on vLLM.
  Either verify vLLM tolerates the field and accept unbounded think inside the
  4096 cap, or bound think by the per-turn `max_tokens` alone, and re-measure
  the parse rate. G_P passed on llama.cpp; it is UNPROVEN on vLLM's parser.
  This is the same kill condition, on a different parser.

**Step B (the flip): `CARNOT_ARC_INDUCE_TOOL_LOOP="1"`, gated on evidence.**
Evidence bar, all three:
1. exp6474 completes with tool >= single on holdout accuracy across the roster
   (not just tu93), or at worst not-inferior with a large wall-clock win.
2. A vLLM-backend re-run of the G_P parse gate passes (pooled parse-failure
   delta <= 5pp, per the pilot's rule).
3. An offline-play-style run with the flip shows bounded LLM share per game
   (the loop's turn cap plus per-turn cap gives this structurally once Step A
   lands).

**Why the current evidence is promising.** The one paired cell in exp6474 so
far: tu93 single-shot spent 4,238s and produced holdout 0.0, memorizing=True;
the tool arm's three tu93 trials spent 673-1,001s and produced holdout
0.625/0.75/0.5, memorizing=False. Better quality at one fifth the wall clock,
because per-turn decode is capped and mental grid simulation is replaced by
tool execution. The pilot's G_P (llama.cpp) passed; G_W's 1.109 is confounded
by GPU contention and should be ignored, as the briefing says.

**Why not flip today.** Same data: tr87 tool trials graded 0.0 twice and
failed once; sb26 failed twice. The loop does not rescue games where induction
is hard, and n is tiny. And without Step A the flip on the scored path does
nothing at all.

**Cost.** Step A: server args + one env + a bounded verification session.
Step B: config flip + one A/B run.

**Abandon if.** vLLM parse rate collapses and no parser flag fixes it (the
approach dies at the transport, exactly what G_P was designed to catch), or
exp6474 completes with tool < single on holdout.

### Move 3 — Fix G_M by fixing the trigger, then re-derive the gate

**The mechanism of the failure** (Section 5 has the arithmetic): the controller
compacts only AFTER a measured crossing, so the peak includes one full turn of
overshoot; and after the first rebuild the re-fire threshold re-anchors at
post-rebuild-floor + growth (`arc_induction_compact_state.py:466-478`), while
the gate's bound stays anchored at turn-0. The gate and the controller define
"bounded" differently. sk48 failed with one compaction (pure single-turn
overshoot); cd82/lp85 failed with two (re-anchor drift).

**Change.**
(a) Pre-emptive trigger: before issuing the next request, if
`last_prompt_tokens + estimated next-turn additions` crosses the threshold,
compact first. The estimator already exists (`_estimate_tokens`).
(b) Keep the relative trigger (the design note's reason stands: turn-0 prompts
vary 10k-17k), but make the gate's bound state the controller's ACTUAL
guarantee: bound relative to `max(turn0, post_rebuild_floor)`, or tighten the
state+tail so the floor stays near turn-0 (evict harder; the tail keeps the
whole last tool round verbatim and can be large).

**Cost.** Small controller change plus analyzer change. The gate re-derives
from already-logged per-turn prompt sizes; no GPU needed for the re-analysis,
one 13-cell re-run for the controller change.

**Confirm.** G_M passes on all cells the OFF arm crossed; `thrash_alarms` and
`floor_hits` stay 0.

**Abandon if.** Pre-emptive triggering raises compaction counts past the alarm
threshold (5) or G_P degrades. Then the growth budget is simply too small for
this loop's turn sizes and the right change is a larger `GROWTH` with an
absolute ceiling.

### Move 4 — Qwen3.8 think-channel levers (measure, then set)

What follows from inline reasoning:

- **The graded think control on the scored backend is the tool loop itself**
  (per-turn caps), not a token. That is another reason Move 2 outranks prompt
  surgery.
- **Selective no-think for the cheap calls.** exp5714 proved a pre-opened
  ```python fence suppresses reasoning by itself. The 2026-08-08 change removed
  the fence everywhere BECAUSE think is on. Reintroduce it selectively: keep
  think for the main combined induce (where exp6199/exp6221-class evidence says
  reasoning helps), use the fence for the goal-only and refactor calls, which
  historically complete in seconds without reasoning. Cheap A/B on the dev box.
- **Think-budget A/B on llama.cpp dev.** `CARNOT_ARC_INDUCE_THINKING_BUDGET`
  is wired and default OFF (`arc_executable_world_model.py:6304-6323`). Measure
  held-out accuracy at budgets {3072, 8192, unbounded} before shipping any cap:
  the exp6221 result (think helps) was on gemma; the Qwen3.8 curve is
  unmeasured.
- **Do not ship binary no-think globally.** The h2h arm header shows Qwen3.8's
  think stream is where the real mechanics get worked out (the tu93 fuel-bar
  model). The failure is unbounded LENGTH, not thinking itself.

**Cost.** Each item is one bounded dev A/B on existing knobs.
**Abandon if.** The budget A/B shows accuracy monotonically improving with
unbounded think AND Move 1's envelope makes unbounded think affordable. Then
the right spend is fewer, longer calls, and the lever is the per-game budget
alone.

### Move 5 — Diagnostics plumbing (small, do alongside)

- Fill `timings.prompt_n` from vLLM's `usage.prompt_tokens` in
  `_vllm_raw_completion` (`arc_executable_world_model.py:6786-6819`);
  `tokens_prompt` read 0 across the whole v3 run, which blinds Phase-0-style
  prompt measurements on the scored backend.
- Tag each proposer call with the issuing stage (stall-loop round / plain
  induce / split fallback / subgoal / repair) in a counter. Attribution in this
  note required arithmetic that the next person should not have to redo.

## 4. Direct answers to the four questions

1. **What generates 300-500k tokens while counted induction "raises"?** The
   counted induction itself. The `llm_reached` counter only flips on a
   returned outcome; the exception lands after the spend (Section 1.1, 2). Not
   a second path. Fix = Move 1.
2. **Should the tool loop become primary?** Yes, on the current evidence
   direction — but it CANNOT be flipped meaningfully until the vLLM transport
   runs tools at all (Section 1.2). Do Move 2 Step A, wait for exp6474, then
   flip if the three-part bar passes.
3. **Can compaction meet G_M?** Yes. The bound is not wrong so much as
   measuring a different quantity than the controller guarantees: post-hoc
   triggering adds one turn of overshoot, and the thrash floor re-anchors the
   second event. Pre-emptive triggering plus a bound stated against the
   controller's real anchor closes it (Move 3). The state+tail after rebuild is
   NOT too big in absolute terms — overshoots were 1.4k-3.7k tokens on ~21-25k
   bounds.
4. **Qwen3.8-specific levers?** Inline think means the per-call answer window
   and the think stream compete, and only per-turn structure (the tool loop) or
   a server-side budget can bound the think stream gracefully. Selective
   code-only fencing for cheap calls, a measured think-budget curve on dev, and
   no global no-think (Move 4).

## 5. Appendix: G_M arithmetic

Bound = `turn0(on) + GROWTH(8192) + STATE_BUDGET(2048) + ONE_ROUND(5000)`
(`experiment_6473_tool_loop_compaction_ab_analyze.py:30-33,120`).

| cell | turn0 | trigger (turn0+8192) | bound | on_max | over by | compactions |
|------|-------|----------------------|-------|--------|---------|-------------|
| cd82 | 5,328 | 13,520 | 20,568 | 24,309 | 3,741 | 2 |
| lp85 | 9,074 | 17,266 | 24,314 | 27,245 | 2,931 | 2 |
| sk48 | 8,531 | 16,723 | 23,771 | 25,202 | 1,431 | 1 |
| tr87 | 10,215 | 18,407 | 25,455 | 22,916 | pass | 2 |

sk48: one compaction, peak 8.5k past the trigger — a single turn added more
than the 5k one-round allowance (large tool result), or the peak is the
post-rebuild floor plus fresh growth that never re-triggered. Either way the
controller behaved as coded and the coded behaviour does not imply the bound.
cd82/lp85: after rebuild one, the re-fire threshold becomes
`post_rebuild_floor + 8192` (`arc_induction_compact_state.py:475-477`); when
the floor exceeds `turn0 + 2048`, the second peak exceeds a turn-0-anchored
bound by construction.

## 6. What is in flight and what it decides

- **Offline play v4** (traceback capture live): names the exception. Decides
  the bug-fix half of Move 1. Nothing else blocks on it.
- **exp6474 equalized holdout A/B** (11 of ~78 tool rows, 1 single row done):
  decides Move 2 Step B. The single arm is slow (~4,200s/cell under think
  mode); expect it to dominate the calendar time.
- Do not build anything on `heldout_accuracy` from the exp5726/6440 lineage;
  it is in-sample since `_induce_transitions_k()` began returning all
  transitions (2026-08-01, `arc_executable_world_model.py:3169-3199`), and
  memorizing engines score higher on it. exp6474's expanded-holdout fields are
  the replacement.
