# ARC live agent, round two: the redirect ledger read against its own controls — 2026-09-05

Written by a worktree agent under the 2026-09-05 brief "evaluate the ARC-AGI-3 live agent and
implement improvements to efficiency, iteration velocity, accuracy, unattended
self-improvement, and supervisor refinement". The 2026-09-04 four-axis baseline
(`arc-live-agent-baseline-four-axes-2026-09-04.md`) is not repeated here. Every number names
its population. "Measured" means computed in this session from files on disk; "read" means
copied from a recorded artifact.

## 1. What was measured

Population A: `ops/arc_supervisor_refinement_ledger.json` as merged on 2026-09-04 21:28Z,
14 receipts and 31 redirects. Population B: the 14 files in
`results/arc_leaderboard_eval_runs/` (13 complete or partial eval artifacts plus one sweep
partial), 28 game rows. Both read-only.

### 1.1 Where the 64 unredirected windows live

| receipts | source shape | window | unredirected windows |
|---|---|---|---|
| 9 | harness `rows.json` (ar25, tu93; 2026-08-27) | 400 or 120 | 0 |
| 5 | live eval `per_game` (r11l x4, cd82 x1; all seed 20260719) | 120 | 64 (13, 14, 14, 11, 12) |

Every one of the 64 sits in a receipt where the three default-on arms all fired. Only ONE of
the five (`r11l-1594772`) also fired `tool_loop_reinduction`, because only that run set
`CARNOT_ARC_SUPERVISOR_TOOL_ARM=1`.

### 1.2 The new-arm trigger was narrower than its concept

`_new_arm_cells` required `set(ARM_ORDER) <= arms_fired`. `ARM_ORDER` gained the env-gated,
default-OFF tool rung on 2026-08-29 (REQ-ARC-WMTE-6760). The spec's own scenario
(SCENARIO-ARC-WMTE-6720-6) still says "all three arms fired". So a default run could never
satisfy the trigger, however long it stagnated.

Measured on population A: the trigger admitted 1 cell (11 windows) and hid 4 cells (53 of 64
windows). The number the brief cites as "the written specification for a new arm" was, for 83
percent of its weight, invisible to the tool that writes that specification.

### 1.3 No receipt says why the table ran dry

`stagnations_unredirected` is a count. Reading `r11l-1594772` by hand: on level 0 the four arms
fired at actions 120/240/360/480 and the level-up came at 813, so two windows passed with every
arm spent. On level 2 the ladder had only three rungs, because the level-0
`force_exploration_diversity` arm left `_hybrid_diversity=True` on the explorer and the arm is
ineligible while diversity is active. Three arms fired at 1130/1250/1370; then about nine
windows passed with `induction_attempts` at the cap of 3, no goal bias installed, and
diversity already on. None of that is in the receipt. A human asked to "propose one new arm"
had a count and no state.

### 1.4 A control now exists, and it says the level-0 credits are base rate

`r11l-3114878` (2026-09-04) ran in SHADOW mode: the supervisor observed and applied nothing.
Read from its `would_have_redirects`:

| arm | would have fired at | level-up followed | actions to level-up |
|---|---|---|---|
| drop_goal_bias | 120 | yes | 765 |
| force_exploration_diversity | 240 | yes | 645 |
| allow_reinduction | 360 | yes | 525 |

The four APPLIED r11l runs credit the same three arms at the same actions with the same
765 / 645 / 525 (768 / 648 / 528 on one run). The level-0 level-up lands at action 813 to 815
in all five runs, applied or not. So 12 of the 19 `helped` credits in the ledger are exactly
reproduced by a run that pulled no lever.

This is the "control-arm (shadow) estimator" the 2026-09-04 note said did not exist yet. It
exists now, and the refinement tool ignored it: shadow rows were counted and discarded.

### 1.5 The 7013 credit split has an empty population

`co_credited_count` and `arm_credit` (REQ-ARC-WMTE-7013) appear in ZERO rows of population B.
The one run that finished after the merge (`r11l-3114878`) started at 21:23Z, before the merge
landed, and carries neither field. So `helped_sole` and `helped_share` read 0 on every arm, and
the brief's instruction to retire or promote arms by sole credit cannot be executed on any
evidence that exists. That is a null result, stated as one. The earliest it changes is after an
APPLIED-mode run with the merged code.

### 1.6 Other facts read from population B

- 0 of 28 rows carry `wall_s`, `run_envelope`, or `solve_provenance`. The heartbeat and
  envelope code shipped 2026-09-04 and 2026-09-05; no run has exercised them.
- 8 rows (the sweep partial) carry an `{"error": ...}` supervisor marker; 11 rows carry no
  supervisor field at all (runs from before the receipt was wired).

## 2. What changed

| REQ | change | tests |
|---|---|---|
| REQ-ARC-WMTE-7030 | the receipt names `arms_enabled`; the refinement trigger reads that set (legacy rows fall back to the three default-on arms); cells name their source | 6 |
| REQ-ARC-WMTE-7031 | every exhausted window records the state the table saw (spent arms, cap and floor flags, bias, diversity), bounded at 64 with a dropped counter; the ledger keeps the rows; the cell carries a per-flag summary; the heartbeat shows the count in flight | 7 + 1 |
| REQ-ARC-WMTE-7032 | shadow receipts ingest into a separate `controls` pool, never `entries`; a credit is `control_matched` when a shadow run in the same (game, seed, window, level) leveled up after the same arm would have fired; the report shows `helped_beyond_control`; the frozen rules still key on pooled `helped` | 6 |

Effect on the live ledger after re-ingest: see section 4.

## 3. What was deliberately not done

- **No new arm was added to `ARM_ORDER`.** The brief allows it if the specification is sound.
  It is not yet sound: every existing cell is a legacy row with no exhaustion state, and the one
  hand-read (section 1.3) shows the level-2 exhaustion happened with the attempt cap reached and
  diversity already on. The candidate arms that remain on the live path each change what the
  scored agent does with its generator or its trust gate. That is an operator decision; the
  proposal is in the handoff.
- **No retirement or promotion of an arm.** Section 1.5: the sole-credit population is empty.
  Section 1.4: the pooled credits are dominated by one deterministic level-0 event.
- **No GPU run.** A run that exercises the heartbeat, the envelope and the new receipt fields
  is a multi-hour commitment. Proposed, not started.
- **No change to `min_heldout_accuracy=1.0`** (the planning gate) and no engine work.

## 4. The ledger after re-ingest

Re-ingested from `results/arc_leaderboard_eval_runs/` (read-only) into the worktree copy of
`ops/arc_supervisor_refinement_ledger.json` with the new reader. Entries and redirects are
unchanged (14 and 31: no row was added or double-counted). What changed is what the tool can
see:

| quantity | before | after |
|---|---|---|
| new-arm cells | 1 (11 windows) | 5 (64 windows), every one `legacy_default`, every one `states: not_recorded` |
| controls | none kept | 1 (the r11l shadow run) |
| credits matched by a control | not computed | 12 of 19 |
| `helped_beyond_control` per arm | not computed | drop_goal_bias 2, allow_reinduction 2, force_exploration_diversity 2, tool_loop_reinduction 1 |

The 2-2-2-1 remainder is the cd82 level-0 credits (651 / 531 / 411), the tu93 harness credits
(233 / 113 / 63), and the one tool-rung credit; none of those cells has a shadow control yet.
The frozen rules did not move: no `retire_candidate`, no `raise_priority_candidate`. Both the
exhaustion states and the sole-credit split stay empty until a run with the merged code
finishes, which is the GPU run proposed in the handoff.

## 5. Proposal for the operator: the fifth arm

The state a human can read today (section 1.3, one cell, by hand) is: goal bias dropped,
re-induction spent to the cap of 3, tool loop spent, diversity already on, about 900 actions
of level-2 stagnation left in the budget. Three levers exist on the live path and each is a
judgement call:

1. **`plan_with_best_partial_model`.** The planning gate is `min_heldout_accuracy=1.0` and the
   best engine ever scored reads 0.672 exact on held-out transitions. An arm that, once every
   other rung is spent, lets `plan_in_model` run ONE plan from the best archived engine below
   the gate would spend actions on a plan the real env then verifies. It changes what
   "trusted" means for one attempt. Highest plausible value; needs the operator.
2. **`widen_diversity_draw`.** Raise `_div_topk` (8) so the randomized draw covers more of the
   frontier. Search-side only; measurable offline with the LLM off in seconds per run.
3. **`reset_and_replay`.** Force a level reset with a different prefix. Cheap; the explorer
   already resets on its own, so the marginal value is unclear.

Recommendation: implement (2) first as an offline LLM-off A/B on r11l and cd82 (window 120,
applied mode, several seeds), because it is the only one this project can measure without a
GPU hour. Bring (1) to the operator with the exhaustion states from the first post-merge run,
so the decision rests on recorded state rather than one hand-read receipt.

## 6. An LLM-off smoke run through the scored eval (measured, not evidence)

One run of `scripts/arc_leaderboard_eval.py --only r11l --policy e3 --budget 1500` with
`CARNOT_ARC_DISABLE_INDUCTION=1`, in the worktree, through a read-only symlink to the main
checkout's `environment_files/`. The record landed in the worktree's gitignored runs directory
and was copied to session scratch; it is a smoke run, not an evidence artifact, and it was not
committed. One game, one seed (20260719), one run.

| quantity | value |
|---|---|
| wall clock | 7.8 s process, `wall_s: 5.094` on the row |
| actions | 1,407; level 0 completed at action 776; level 1 not completed in 631 |
| heartbeat | 18 distinct snapshots captured at a 50 ms poll; `writes: 22`; `progress_write_errors: 0`; file removed at the end |
| events captured | `periodic`, `induction_attempt_recorded`, `end` (the `level_up` write was overwritten before the next poll; `level_up_actions: [776]` persisted in every later snapshot) |
| receipt (shadow) | `arms_enabled` = the three default-on arms; 5 would-have redirects (120/240/360 on level 0, 966/1086 on level 1); `stagnations_unredirected: 7`; 7 window rows, 0 dropped |
| window rows | level 0 at 480/600/720/840 with `goal_bias_installed: true` (the drop was counterfactual in shadow mode) and the cap not reached; level 1 at 1206/1326/1446 with no bias |
| envelope | present; every GPU field an honest none (LLM off); `solve_provenance: live_agent_self_discovery` |

Two things this says. First, every field shipped on 2026-09-04 and 2026-09-05 (REQ-7010,
7021, 7022, 7030, 7031) is now exercised by a real run of the scored script, in shadow mode,
without a GPU. Second, and new: **r11l level 0 is cleared by the classical path alone, at
action 776, with the generator disabled.** The LLM-on runs clear it at 813 to 815. So the
level-0 credits in the ledger were never a generator event either; the control in section
1.4 and this run agree from two directions. One seed, one run: a fact about this seed, not
yet a rate.

## 7. CORRECTION 2026-09-05 (round three, append-only): the five cells were false

An adversarial review of this branch found that the exhaustion cell (sections 1.2 and 4) was
computed POOLED over the run. Confirmed against source: `TrajectorySupervisor.observe` clears
`_arms_used` on every level-up ("Start the level fresh: arms become available again"), and
`_first_eligible_arm` tests membership in that per-level set. So "every enabled arm fired"
pooled over a run is not "every enabled arm spent on the level that stagnated". The table in
section 4 ("new-arm cells: 5 (64 windows)") is retracted. REQ-ARC-WMTE-7033 is the fix; the
7030 enabled-set work, the 7031 window rows and the 7032 controls stand.

### 7.1 What the corrected reader emits on the same population

Read from the receipts alone, as the tool does: **0 cells, 5 receipts not decidable** (the five
stagnating rows carry no per-level window rows; they predate REQ-7031). The live ledger's
status moves from `recommendation_available` (five false cells) to `insufficient_evidence`.
Entries and controls are byte-equal before and after; only `recommendation` and `updated_at`
changed.

### 7.2 What the per-level condition WOULD have emitted (validated reconstruction)

Population: the 6 rows in `results/arc_leaderboard_eval_runs/` with a supervisor window (5
applied, 1 shadow; all window 120, seed 20260719). The receipt records the redirect action
indices and the run-pooled unredirected count; the row records the level-up frame indices in
`level_reset_attribution.segments` and a per-frame `levels_completed`. Replaying the supervisor's
window arithmetic (reset at a level-up and at every window boundary; a boundary at 120 stagnant
observations) from those inputs is checked per row by two identities the receipt provides:
the reconstructed redirect boundaries must equal the recorded ones exactly, and the
reconstructed unredirected count must equal `stagnations_unredirected`. All 6 rows validate
(one convention had to be learned from the receipt's own arithmetic: the level-up CALL is one
after the frame index where `levels_completed` first reads the new value, since 120 + 765 =
885 and the frame reads 884).

| row | level 0 exhausted windows | deeper levels | level 0 cleared at |
|---|---|---|---|
| cd82-r11l-1408494 row 0 (r11l) | 4: 480, 600, 720, 840 | 0 | 885 |
| cd82-r11l-727651 row 0 (r11l) | 4 | 0 | 885 |
| cd82-r11l-727651 row 1 (cd82) | 3: 480, 600, 720 | 0 | 771 |
| r11l-1594772 (r11l, tool rung on) | 3: 600, 720, 840 | 0 | 885 |
| r11l-2491317 (r11l) | 4 | 0 | 888 |
| r11l-3114878 (r11l, shadow) | 4 | 0 | 885 |

Two readings. First, the per-level condition is not decorative: on every applied run it fires on
level 0, 3 or 4 windows after the last arm was spent, so a run recorded with the REQ-7031 rows
would produce these cells directly. Second, every one of those level-0 exhaustions was followed
by the level-up at 771 to 888 -- the same action the shadow run reached with no lever pulled and
the LLM-off smoke run (section 6) reached at 776. The table ran dry on level 0 and the classical
path got through anyway. A cell now carries `level_resolved_by_levelup` and
`actions_from_first_exhaustion_to_levelup` (405 on the first row) so a reader can weigh that
before proposing an arm. The deep levels, where the runs actually stalled for 1,000+ actions,
show 0 exhausted windows: `force_exploration_diversity` was never spent there, for the reason
REQ-ARC-WMTE-7040 records (the snapshot reported diversity as in effect on every level after the
first). With that fix on main, a future run can reach a true deep-level exhaustion; no run has
yet.

### 7.3 What this changes in section 5

The fifth-arm proposal rested on "every cell". There is no cell. The only recorded input for a
new arm is the level-2 hand-read in section 1.3, and REQ-7040 has since changed what the
supervisor sees on that level. Recommendation unchanged in direction, weaker in urgency: run one
applied-mode eval with the merged code (REQ-7031 rows plus REQ-7040) before any arm is proposed.

