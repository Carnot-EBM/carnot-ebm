# The `degenerate_goal_predicate` wall: diagnosis, and what is actually blocking induce -> plan

**Date:** 2026-07-29
**Status:** DIAGNOSED, NOT FIXED. Nothing in this note has been shipped. Every proposal at the
end is a proposal.
**Substrate:** offline arcade, no LLM (`offline_arcade_live_agent_runtime_self_discovery_no_llm`).
Every measurement ran against a redirected `CARNOT_ARC_E3_DIR`; `results/arc_e3` was never written.

## Why this note exists

Best-engine retention (REQ-ARC-WMTE-6035) fixed the store, but a sweep found retention alone does
not unlock planning. With `min_heldout_accuracy` at 1.0 / 0.9 / 0.75 every round fails the dynamics
veto; at 0.6 / 0.5 / 0.25 the PEAK engine passes the veto and then dies on a second, independent
gate — `degenerate_goal_predicate`; at 0.0 a plan appears only by admitting a
change-fidelity-0.0 engine. **A model with perfect dynamics still cannot plan.** This note is the
diagnosis of that second gate.

It is the corrected version. A first pass reached several conclusions that a hostile review then
refuted with measurements the first pass had not run. The refuted claims are marked RETRACTED and
left in place rather than deleted, because the retraction is the useful part.

## 1. What `degenerate_goal_predicate` actually tests

`arc_llm_reinduction._goal_satisfiability_check` (lines 512-608) is a FIFO BFS from `root_grid`
through the induced engine, `max_nodes=20000`, `max_depth=40`. Successors come from a local closure
`_probe_candidates`: actions 1-5, plus ACTION-6 clicks at **the first 32 non-background cells in raw
row-major raster order**. If no grid in that ball satisfies the goal, the round is rejected with
counterexample kind `degenerate_goal_predicate`.

On the real ka59 root with the peak engine (`341f776c9`, change_fidelity 1.0000 as of 2026-06-17):

* it reaches the full advertised depth 40, exhausts its frontier completely, and evaluates
  **2641 unique grids against a 20000 cap**. The budget is not the reason.
* The two movable blocks sit at rows 30-32. All 32 of the gate's clicks land on row 21 — the top
  border. ka59's only selection mechanic is `_click_select`, which requires a click *inside* a
  block, so the gate can never select the second block, while the goal requires two blocks on
  targets.
* The clicks are not visibly inert, which is what makes this hard to spot: all 32 change the grid,
  but only by ticking the bottom-row step counter. That is why the search burns 2641 states going
  nowhere.

Corpus-wide, in **25 of 25 games** the gate's clicks cover strictly fewer distinct interactive
regions than the planner's `_model_candidates` (component centroids); in **11 of 25** the gate
covers exactly ONE region, and in **18 of 25** every one of its 32 clicks lands in grid rows 0-8,
i.e. the top border. This is structural, not a ka59 quirk.

## 2. Is the check correct, or a false positive? — a false negative, plus a naming defect

**It is a false negative on a good engine.** Same engine, same goal, same root, same budget,
changing only the successor generator to `e3._model_candidates`: `satisfiable=True at depth 11`,
12435 unique states. Depth 11 is exactly the registry's ground-truth L1 solve length, and the
recorded witness path is

```
4, 4, 4, 3, 2, 3, 3, 3, CLICK(43,31), 1, 4
```

— the canonical L1 solve sequence, with the click landing inside the first block. A pre-filter
weaker than the search it guards is unsound by construction.

*Evidence note.* `_probe_candidates` is a local closure with no injection point, so swapping the
generator required re-implementing the traversal. The clone is pinned to the shipped function
deliberately, not incidentally: on the as-shipped arm the shipped
`_goal_satisfiability_check` and the clone agree exactly (`satisfiable=False`,
`reachable_grids_evaluated=2641`, kind `degenerate_goal_predicate`), so the swapped arm differs
only in the one line intended to vary
(`p2b_clone_crossvalidation_and_witness_results.json`, `5f260ad550ab9350`).

**There is also a real coverage gap, but it is narrower than "the gate is inverted".**
`_goal_satisfiability_check` returns satisfiable immediately if `goal(root)` is truthy, so
`lambda g: True` — the most degenerate predicate expressible — passes at depth 0 and the planner
behind it returns a 1-action plan. `lambda g: False` is correctly rejected. That is not a polarity
inversion: the function's docstring says it rejects constant-false goals, and a satisfiability check
admitting an always-true goal is by-design behaviour for a *satisfiability* check. The two genuine
defects are (a) nothing anywhere tests `goal(root) == True`, which at a level's opening screen means
the predicate is trivially wrong, and (b) the counterexample kind `degenerate_goal_predicate` is a
misnomer for what the check actually establishes, which is "unreachable within a bounded search
weaker than the planner's". The misnomer is not cosmetic — it sends `refactor()` after the dynamics
when the dynamics were fine.

~~RETRACTED: "the gate is wrong in both directions / inverted."~~ Overstated; see above.

## 3. The third wall was an under-powered null — and the goal-energy channel is dead at L1

The first pass dismissed generator alignment on the grounds that `plan_in_model` returns no plan
anyway (`max_nodes_reached`). That was measured with a bare `e3.plan_in_model(engine, goal, root)`:
no `goal_energy`, default `max_nodes=20000`. The live path is
`E3AgentPolicy._call_plan_in_model`, which supplies a `goal_energy` and honours
`CARNOT_ARC_PLAN_MAX_NODES`. A null measured on a weaker planner than the one that ships is this
project's own FALSE_NEGATIVE_RISK shape.

Re-measured under the configuration the agent actually ships at L1
(`p2b_third_wall_live_config_results.json`, `78c54a4ac933af67`):

| condition | plan | termination | nodes | initial E | min E |
|---|---|---|---|---|---|
| A blind, 20000 (the original measurement) | no | `max_nodes_reached` | 20006 | — | — |
| B **live** production `goal_energy`, 20000 | **no** | `max_nodes_reached` | 20006 | 1.0 | 1.0 |
| C live production `goal_energy`, 400000 | **yes, length 11** | `plan_found` | 137347 | 1.0 | 1.0 |

Two things fall out, and the second was not in the original objection:

1. **The 20000-node default is the binding constant.** At 400000 the planner finds exactly the
   canonical 11-action solve. So ka59's induce->plan path is blocked by two knobs we control — the
   gate's weaker successor generator and the planner's node budget — not by a missing goal concept.
2. **The production `goal_energy` provides zero gradient at L1, while reporting that it is being
   used.** `_goal_energy_for_plan` has three branches: graded-exemplar (needs
   `CARNOT_ARC_GRADED_GOAL_BIAS=1` AND an exemplar — neither holds at L1), novelty (needs
   `CARNOT_ARC_NOVELTY_GOAL_BIAS=1`, unset in production), and binary. What ships at L1 is binary:
   `0.0 if is_done(grid) else 1.0` — a flat constant over every non-goal state. Measured, not
   inferred: `used_goal_energy_search=True` with `initial_goal_energy == min_goal_energy_observed
   == 1.0`, and B is outcome-identical to the blind arm A down to the node count. This is a dead
   channel wearing a passing flag, the same shape REQ-ARC-WMTE-6013's `noop_channel_measurable`
   exists to expose elsewhere in this codebase.

~~RETRACTED: "even with the planner's generator, plan_in_model returns no plan, so aligning the
generators would not help."~~ It was a budget constant.

## 4. The task-specification gap underneath — real, but the first pass's remedy was wrong

Driving the real offline arcade through the canonical 11-action ka59 solve: `levels_completed` goes
0 -> 1 at step 11, and the induced predicate is False on that frame and on all 12 frames.

That frame is not a win rendering. Per-step changed cells: ordinary steps median **18.5**; the
winning step **3527 of 4096 — 86% of the grid, a 190x outlier**, with object count and TARGET
population both jumping. It is a full playfield re-layout. Same on the only two other games whose
rollout window contains a real level-up: lp85 **1544 cells (37.7%)** vs ordinary median 0 / max 293;
r11l **1395 (34.1%)** vs median 1 / max 88.

**Sample-size caveat, stated rather than buried:** those are the ONLY 2 of 25 games whose standard
120-transition random-rollout window contains a real level-up at all (section 4's own 23-of-25
figure is the same fact from the other side), so the corpus check behind "level-up transitions are
re-layouts" is n=2 plus ka59. The separation is enormous where it is measurable (34-86% of the grid
against ordinary medians of 0-18.5), but it is three games, not a population. Treat it as a strong
signal that motivates a change, not as an established corpus-wide law.

**Four sites read exactly that post-relayout frame**, not two as first reported:

1. `induce_prompt` — `win = next(t for t in trans if t.level_after > t.level_before)`, emitted as
   `"WIN STATE ... is_level_complete must return True here"` from `win.next_grid`.
2. `score_goal_predicate_consistency` — `claimed = is_level_complete(t.next_grid)` graded against
   `real = t.level_after > t.level_before`.
3. `arc_competition_agent._observe_level_boundary` -> `_begin_level_goal_episode` —
   `completed_grid = to_logical(grid_of(latest), cell)` where `latest` is the frame *after* the
   level counter incremented. This is what populates `_previous_level_complete_grid`, which feeds
   the `WIN STATE EXEMPLAR` prompt block **and**
4. `_repair_degenerate_goal`'s `_nonzero_count_predicate(exemplar)` fallback, and the
   `goal_exemplar_grading` synthetic positive.

So the LLM is shown the *next level's board* and told that is what "this level complete" looks like,
then graded on the same frame, and the repair mechanism's fallback threshold is derived from it too.
The ka59 predicate's concept ("two blocks seated in targets") matches the registry's documented
win condition exactly; it is False on that frame because that frame is a different level's board.

At L1 there is usually no frame at all: **23 of 25 games have zero level-ups** in the standard
120-transition window, and `previous_level_complete_grid` is None at L1, so `induce_prompt` emits
**no WIN STATE block whatsoever** and `score_goal_predicate_consistency`'s veto is structurally
inert (`n_real_levelups >= 1` never holds).

~~RETRACTED, and this is the important one: "capture the win exemplar as the frame *before* the
re-layout (`t.grid`) rather than `t.next_grid`."~~ Tested and refuted
(`hostile2_proposal1_test_results.json`, `b76e0dd5151a72b9`): on the real canonical ka59 solve the
peak concept-correct predicate is False on `t.next_grid` **and on the proposed `t.grid`** — False on
every observed frame. `t.grid` is one action *before* completion, where a correct predicate must be
False. The L1-complete configuration is never rendered at all, because the completing action is the
re-layout. Shipping that change would have injected a wrong positive. The diagnosis (there is no
usable positive frame) was right; the remedy was wrong.

The engine's own internal state is where the predicate does fire: walking the peak engine through
the canonical sequence, `is_level_complete` is True at the 11th action
(`hostile2_witness_check2_results.json`, `07134b433d7a340a`). The terminal configuration exists in
the model; it is the *renderer* that never shows it.

## 4b. A separate defect found while checking the above: the reinduction loop grades UNMASKED but refactors MASKED

Not part of the goal-gate question, and not fixed here, but it was found while tracing these code
paths and it bears directly on REQ-ARC-WMTE-6035's own selection signal, so it is recorded rather
than dropped.

`hud_mask` appears exactly twice in `arc_llm_reinduction.py` — once as a parameter of
`execute_bounded_llm_reinduction` (line 912) and once where the refactor counterexample evidence
is scored (line 1079, `WorldModelVerifier(list(transitions), hud_mask=hud_mask)`). It is **never
passed to `select_trusted_world_model`** (line 1009). So within one function:

| quantity | masked? |
|---|---|
| `heldout_accuracy` (the dynamics veto) | **no** |
| `heldout_change_consistency` (the REQ-6035 retention signal) | **no** |
| the mismatch evidence handed to `refactor()` | **yes** |

The sibling live path does the opposite: `arc_competition_agent.py:5943` passes
`hud_mask=_hs_mask` into `select_trusted_world_model` for the hidden-state games, with a comment
noting that branch "grepped ZERO for `hud_mask` until 2026-07-27".

Why it matters here specifically: unmasked, a HUD step-counter tick is a real changed cell, so
`heldout_change_consistency` rewards an engine for modelling the counter. That is precisely the
failure REQ-ARC-WMTE-6010 exists to prevent — its own rationale is that feeding the proposer
HUD-only deltas "teaches the proposer to model the step counter instead of the mechanic" — and it
is visible in section 1's ka59 measurement, where all 32 of the gate's clicks "change the grid"
and every one of them does so only by ticking the bottom-row step counter.

**Deliberately not fixed tonight, for two reasons that are about honesty, not effort.** (1)
Threading the mask into that call also changes `heldout_accuracy`, which is the DYNAMICS VETO at
`min_heldout_accuracy=1.0` — so it changes which rounds reach the planner at all. That is a live
gate behaviour change and REQ-6010's own history says this class of change needs a measured
default-off parity pass, not a same-session flip. (2) The REQ-6035 counterfactual was measured on
the UNMASKED signal. If the mask is threaded, **retention's 24/3/28 and 0.3979-vs-0.0042 must be
re-measured**, because the quantity being ranked on would no longer be the quantity that was
measured. Fixing the mask silently would invalidate the evidence for the change shipped alongside
this note.

## 5. Provenance caveat on the exemplar engine

The peak ka59 `world_model.py` (`341f776c9`) was written by a codex agent acting directly as
proposer, not necessarily through the live `induce_prompt` path. So it is not evidence about what
the live 9B/31B induce path produces. For this exemplar the goal WAS usable and plannable, and the
gate plus the node budget discarded it. Whether the live induce path — which at L1 emits no WIN
STATE block at all — can produce a comparable predicate is **untested by either pass**, and the
first pass's conclusion "the LLM produces dynamics without a usable goal, therefore this is a
prompt-specification gap" is downgraded to a hypothesis on that basis.

## 6. Proposals — ordered, none of them "let plans through"

None of these disable a gate. A plan from a degenerate goal predicate is garbage and admitting it
would be the lower-the-bar anti-pattern.

1. **Raise the planner's node budget on the offline dev path first and measure the cost.**
   `CARNOT_ARC_PLAN_MAX_NODES` is already honoured by `_call_plan_in_model` and unset in
   production. It is the single change with a measured unlock (ka59 L1, canonical 11-action plan).
   The honest caveat: 20000 nodes cost 26s and 400000 cost 197s in this measurement, so this trades
   wall-clock against reach and cannot simply be turned up without a per-episode budget analysis
   against `MAX_ACTIONS=400`. Measure the distribution of nodes-to-plan across the corpus before
   choosing a value.
2. **Fix the dead goal-energy channel at L1**, or stop reporting it as used. A flat energy makes
   `plan_in_model` a blind BFS while `used_goal_energy_search=True`.

   **The obvious cheap fix was tried tonight. It does not unlock the shipped budget, but it is
   not worthless either — the result is genuinely two-sided.** The novelty branch
   (`CARNOT_ARC_NOVELTY_GOAL_BIAS=1`) already exists precisely for this exemplar-free
   first-contact case and its own docstring calls itself "opt-in pending empirical A/B
   validation". This was that A/B, on the one engine+goal whose ground-truth plan is known
   (`p2c_novelty_goal_energy_ab_results.json`, `42d0a885e042a748`):

   | arm | budget | plan | plan length | nodes | gradient |
   |---|---|---|---|---|---|
   | binary (production today) | 20000 | no | — | 20006 | **absent** (init 1.0 = min 1.0) |
   | novelty | 20000 | **no** | — | 20015 | present, near-flat (init 1.0, min 0.9912) |
   | binary | 400000 | yes | **11** (canonical) | 137347 | absent |
   | novelty | 400000 | yes | **30** | **71083** | present, near-flat |

   Read both columns before concluding anything:

   * **It does NOT remove the wall at the budget that ships.** At 20000 nodes novelty still
     terminates `max_nodes_reached`. Its entire dynamic range is 0.88% of the scale, which is
     what should have been expected — novelty rewards distance from what has been observed, which
     is not the direction of the goal. **Flipping this flag does not unlock planning; do not
     propose it as though it does.**
   * **But at a raised budget it nearly HALVES the search** — 71083 nodes against the blind
     arm's 137347, a 1.93x reduction — which is a real efficiency effect from a near-flat energy,
     and more than I expected.
   * **And it costs plan quality: 30 actions instead of the canonical 11.** Best-first with a
     non-trivial energy gives up breadth-first's shortest-path guarantee. On a benchmark whose
     score is efficiency-weighted, a 2.7x-longer plan is a serious regression, and it is not
     obviously paid for by halving the nodes searched (planning is offline; actions are scored).

   So novelty trades *offline search cost* for *online plan quality*, which is the wrong
   direction for this scoring function. What is actually needed is a genuinely goal-directed
   energy available with no exemplar. The cheaper honest step, independent of that: record
   `goal_energy_gradient_present` on the diagnostics so a flat channel stops reporting itself as
   live — the same fix shape as `noop_channel_measurable`.
3. **Align the gate's successor generator with the planner's, or retire the separate search.**
   Aligning removes the false negative but makes the gate a slower duplicate of the planner
   (12435 states, and at the planner's budget the planner itself is the same work). The cheaper
   correct design is to delete the pre-veto's own BFS and let `plan_in_model` be the reachability
   test — it already distinguishes `queue_exhausted` (genuinely unreachable) from
   `max_nodes_reached` (budget).
   **This has a mandatory interaction that must be resolved first, not as a side effect.**
   `_repair_degenerate_goal` (GOAL-REPAIR, 2026-06-25 operator directive) fires on
   `if not round_goal_satisfiable`. Replacing the reachability veto with root-true/constancy checks
   would make `round_goal_satisfiable` True for nearly every non-constant predicate and thereby
   **silently disable GOAL-REPAIR at L2+**, where it has an exemplar and a real rescue role. If the
   pre-veto's search is retired, GOAL-REPAIR must be re-triggered on `plan_in_model` returning None
   with `termination_reason == "queue_exhausted"` — which is a strictly better trigger than today's
   weaker-search proxy — and never on `max_nodes_reached`, which is a budget signal. Retiring an
   operator-directed mechanism requires operator sign-off, not an unnamed consequence of a refactor.
4. **Add the missing trivially-true check and rename the counterexample kind.** Reject when
   `goal(root)` is True (a level is not complete at its own opening screen); rename
   `degenerate_goal_predicate` to something that says what was established, e.g.
   `goal_unreachable_within_bounded_search`. Cheap, and it stops misdirecting `refactor()` at the
   dynamics.
5. **Do not induce or grade a win predicate from a rendered frame at all.** All four capture sites
   in section 4 read a post-relayout board. The signal that is actually trustworthy is the labelled
   TRANSITION event (`level_after > level_before`) and the engine's own counterfactual terminal
   state — not any frame the renderer produced. Any exemplar-derived construct
   (`_nonzero_count_predicate`'s threshold, the `WIN STATE EXEMPLAR` block, the
   `goal_exemplar_grading` synthetic positive) inherits the same poison and must be rebuilt on the
   same basis.
6. **Resolve the inconsistency in when this check runs at all.** `arc_competition_agent.py:6107`
   already keeps this same check dev-only and opt-in on the plain path, citing precisely this
   false-negative risk; it is unconditionally on in `execute_bounded_llm_reinduction`. Resolve in
   favour of the cautious side.

## 7. What is NOT established

* That fixing the exemplar capture would unlock a solve. No LLM run was made in either pass.
* That the winning frame is the L2 board specifically. The re-layout is inferred from the magnitude
  of the change (190x the ordinary median), not from independently identifying the next level.
* That raising `CARNOT_ARC_PLAN_MAX_NODES` is affordable in a live episode. Only that it unlocks
  ka59 L1 offline, at 197s.
* That the novelty arm's 30-action plan is a REAL solve. It reaches `is_level_complete` inside a
  change-fidelity-1.0 engine, which makes it plausible, but it was never replayed against the
  arcade. The 11-action canonical plan is known-real; the 30-action one is model-internal only.
* That the novelty node-count halving generalises past ka59. It is one game, one engine, one root.
* Anything about the live 9B/31B induce path's goal predicates (section 5).

## Artifacts

All under the session scratchpad, each with `inference_substrate` and a checksum.

| artifact | checksum |
|---|---|
| `p2b_third_wall_live_config_results.json` | `78c54a4ac933af67` |
| `p2c_novelty_goal_energy_ab_results.json` | `42d0a885e042a748` |
| `p2b_clone_crossvalidation_and_witness_results.json` | `5f260ad550ab9350` |
| `hostile2_proposal1_test_results.json` | `b76e0dd5151a72b9` |
| `hostile2_third_wall_test_results.json` | `cec654b807700618` |
| `hostile2_witness_check2_results.json` | `07134b433d7a340a` |
| `p2_goal_gate_diagnosis_results.json` | `bbaa1f54f7f3888d` |
| `p2_candidate_set_mismatch_results.json` | `86158d85d4c30ddf` |
| `p2_raster_click_blindness_results.json` | `72c84b2fbf12c7a2` |
| `p2_predicate_truth_and_generator_swap_results.json` | `93ee4fad88237299` |
| `p2_gate_polarity_results.json` | `f5568e8729c01f99` |
| `p2_why_predicate_misses_real_win_results.json` | `70a502a3b7cb9ac9` |
| `p2_win_exemplar_is_next_level_results.json` | `49af53f060af635e` |
| `p2_levelup_frame_relayout_results.json` | `81556baf56dd12d2` |
| `p2_goal_veto_inert_results.json` | `87cc75f127dedb9b` |
