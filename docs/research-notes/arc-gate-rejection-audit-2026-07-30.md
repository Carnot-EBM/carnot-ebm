# Is the world-model trust gate right or wrong about the five rejections?

**Date:** 2026-07-30
**Upstream:** `results/outer_loop_arc_llm_on_vs_off_activation_20260730.json` (commit `4bec30537`,
corrigendum `e7169a746`)
**This measurement:** `results/outer_loop_arc_gate_forceadmit_20260730.json`
**Verdict in one line:** the gate is **right on four of five games**; on the fifth (tn36) it reaches
the **right decision with a wrong label**; force-admitting the rejected engine is **NEUTRAL on all
nine admit cells** — zero banked levels, zero change in actions-to-first-solve. **No gate change is
proposed.**

Banked levels remain **3/3** and the submission gate remains **unmet**. Nothing here changes that.

---

## 0. What was measured, and the two corrections that matter most

The LLM on/off experiment established that deleting the LLM tier leaves the agent's actions
byte-identical on five of six games, and that the tier fired but its output reached the policy on
only one. Five cells were rejected by the world-model trust gate. This note answers, per game,
whether each rejection was correct.

Two methodological corrections were applied to the first pass of this work before any verdict below
should be trusted:

1. **The probes did not run the shipped planner.** The first-pass Stage 1 and Stage 2 called
   `e3.plan_in_model(...)` directly with no `goal_energy`, so every probe ran plain FIFO BFS
   (`used_goal_energy_search: false` in their own recorded diagnostics) while the shipped accept
   path runs `self._call_plan_in_model(e3.plan_in_model, engine, is_done, self.root_grid,
   diagnostics=...)` (`arc_competition_agent.py:6156`), which derives a goal energy and can make the
   search best-first. Every cell was re-run through `_call_plan_in_model`, with the node budget taken
   from the repo's own dev-only `CARNOT_ARC_PLAN_MAX_NODES` override
   (`arc_competition_agent.py:4938`) rather than a hand-passed `max_nodes`. The re-run now records
   `used_goal_energy_search: true, goal_energy_source: "binary"`, matching the live cells.
   **The numbers did not move** — node counts are identical to the digit (37 / 185 / 191 / 307 /
   1480 / 20017 / 300033), because the binary goal energy is a flat constant and `heapq`'s
   `(energy, counter)` tie-break degenerates to FIFO, exactly as
   `arc_llm_reinduction.py:806-818` claims. That claim is now **measured here rather than assumed**,
   and the ft09-`onb` cell that hit `max_nodes` — the one place guidance could plausibly have
   changed the answer — was re-run at 300k nodes with guidance on and still returns no plan.

2. **The control was an unvalidated reconstruction.** The control arm reproduces the gated run by
   replaying the same deterministic trajectory with an inert generator; it is not the gated run
   itself. It is now **validated by byte-comparison**: at budget 60 the control's action trace is
   identical to the **live LLM-ON cell** on all five games (after normalising the two recording
   formats — the live cells write `ACTION6|{...}`, the probe writes `{"action":6,...}`; the action
   sequences are the same).

   | game | live ON | live OFF | reconstructed control (b60) | ON==OFF | reconstruction==ON |
   |---|---|---|---|---|---|
   | ft09 | `06f1e397129a03d9` | `06f1e397129a03d9` | `06f1e397129a03d9` | yes | **yes** |
   | tu93 | `f36e08220f3a34b5` | `f36e08220f3a34b5` | `f36e08220f3a34b5` | yes | **yes** |
   | lp85 | `cbd1d9cd6a3b6e2c` | `cbd1d9cd6a3b6e2c` | `cbd1d9cd6a3b6e2c` | yes | **yes** |
   | sc25 | `6ee540f555dc8966` | `6ee540f555dc8966` | `6ee540f555dc8966` | yes | **yes** |
   | tn36 | `530fc9272b9da215` | `530fc9272b9da215` | `530fc9272b9da215` | yes | **yes** |
   | vc33 | `fdc92ab1cd0f71cb` | `60d01ca545b63bfc` | `60d01ca545b63bfc` | **no** | no (matches OFF) |

   vc33 is the internal consistency check: it is the one game where the LLM tier reached the policy,
   ON and OFF genuinely differ there, and the inert-generator reconstruction correctly reproduces the
   **OFF** arm. So the reconstruction reproduces the arm it is supposed to reproduce, and on the five
   games ON==OFF it therefore also reproduces the gated ON run.

   The primary comparison below is reported at **budget 60**, where this validation holds. The one
   cell that cannot be run at 60 (tn36-`on`, whose plan is 61 actions long) is reported at 120 and
   flagged.

---

## 1. Read the rejections per CELL, not per game

The first pass reported one rejection reason per game and called the two replicates identical. They
are not. Read from each cell's own `induction_events[0].skipped`:

| game | `on` replicate | `onb` replicate |
|---|---|---|
| ft09 | `world_model_accuracy_below_threshold` | `world_model_accuracy_below_threshold` |
| tu93 | `world_model_accuracy_below_threshold` | `world_model_accuracy_below_threshold` |
| **tn36** | `degenerate_goal_predicate` | **`world_model_accuracy_below_threshold`** (0.48, 12/25) |
| **lp85** | `goal_predicate_error` | **`world_model_accuracy_below_threshold`** (0.0) |
| **sc25** | **`proposer_failed_or_missing_root`** | **`proposer_failed_or_missing_root`** |

Consequences, all of which narrow the original claim:

- The false-rejection claim, at its widest, rests on **1 of 10 cells** (tn36-`on`), and tn36's
  *other* replicate is a correct rejection by the audit's own standard.
- **sc25 is not a terminal gate rejection at all.** Its three `heldout_transition_verification_failed`
  events (heldout 0.875 / 0.0 / 0.875 against a threshold of 1.0) are bounded-loop rounds; the
  terminal label in both replicates is a **generation failure** at
  `arc_competition_agent.py:5891`. It is excluded from the count of terminal gate rejections below.

### Why the two replicates diverge while their refinement rounds are identical

lp85's `on` and `onb` `refinement_rounds` are identical field-for-field (same counterexamples, same
`trust_energy` 496.733826, same `UnboundLocalError`), yet their terminal `change_gate` differs
(`legacy_accuracy` 0.92 vs 0.0). Same for tn36 (rounds identical, terminal `verify_accuracy` 1.0 vs
0.48). The mechanism is not artifact corruption:

> when the refinement loop returns unplanned, `arc_competition_agent.py:5837` falls through and
> **`:5889` calls `proposer.induce()` AGAIN**, rewriting `world_model.py`; `:5893` reloads and
> rescores *that* engine.

So a terminal induction event **mixes fields from two different engines**: `refinement_rounds`,
`engine_retention` and `goal_satisfiability` come from the refinement loop; `verify_*`, `change_gate`
and the terminal `skipped` describe the post-fallthrough single-shot engine. This explains the
identical round telemetry alongside divergent on-disk md5s (`on__tn36` `6d96491f…` 5256 B vs
`onb__tn36` `259cade1…` 4500 B, and similarly for every other game), the 1.0-vs-0.48 split, and
tn36-`onb`'s stored engine returning `satisfiable, first_true_depth=6` while its record shows
1480 calls / 41 grids. **The "artifact-integrity inconsistency" framing of the first pass is
withdrawn** — this is a documented control-flow path, not a corrupted record. The same correction
retracts the claim that lp85's stored file is "the round-1 split-induce candidate"; it is the
post-fallthrough engine.

---

## 2. The gate is two gates, and the TTT prior side has a measured false positive

The audited cells run **two** admission checks with **different metrics**:

- the LLM-induced engine is gated on **full-grid byte-exact match on all 8 held-out transitions**
  (`min_heldout_accuracy=1.0`, `arc_competition_agent.py:5643`/`:5780`) — the documented default is
  0.5 (`arc_world_model_trust_energy.py:537-549`), and every cell carries both numbers;
- the **TTT prior engine** is gated on `cell_recall >= 0.5`.

The prior was never reported in the first pass, and it cuts both ways:

| game | prior gate | prior heldout exact | prior cell_recall | prior plan diagnostics |
|---|---|---|---|---|
| tn36 (`on`,`onb`) | **PASS** | 1.0 | 1.0 | 20017 nodes, `max_nodes_reached` |
| tu93 (`on`,`onb`) | **PASS** | **0.0** | 0.7544 | 20017 nodes, `max_nodes_reached` |
| ft09 | FAIL | 0.3333 | 0.4276 | — |
| lp85 | FAIL | 0.0 | 0.0 | — |
| sc25 | FAIL | 0.8333 | 0.0556 | — |

- **tn36 is planner-bound, not gate-bound.** A gate-PASSING model with perfect held-out accuracy and
  perfect cell recall *did* reach the planner, and the planner exhausted 20017 nodes without a plan.
- **tu93's prior is a live gate FALSE POSITIVE inside the audited cells**: exact held-out accuracy
  0.0, admitted on `cell_recall` 0.7544. This is the correct citation for "the gate has false
  positives" — better than the historical `results/arc_e3` identity engine, because it is inside this
  measurement.

The **change-aware** gate (`cell_recall`, `change_fidelity`, `invented_change_rate`, no-op
hallucination rate) is computed on every cell but `gate_enabled: false`. HUD masking was
`disabled` / `flag_disabled` / `hud_mask_cells: 0` run-wide.

---

## 3. Per-game verdicts

### ft09 — REASON_TRUE (both replicates)

- `on`: held-out 1/8. The stored engine is **1112 of 1144 lines pure comment** — a generation
  runaway — and `engine()` ends with no `return` on its final path, so it returns `None`. No-op by
  omission.
- `onb`: **`n_changing: 6`, `n_changes_correct: 0`** and **`n_noop: 19`, `n_noop_hallucinated: 19`
  (rate 1.0 against a 0.25 cap)**, `legacy_accuracy: 0.0`. It gets **none** of the six real changes
  right and fires a change on **every one** of the nineteen transitions the game ignores. Its
  `is_level_complete` is `return False` (constant), so no state could ever satisfy it.
  Mitigating and worth recording: `cell_recall` 0.947, 216 correct changed cells, 0 spurious, and
  three of its six real-set misses are wrong at exactly two cells in row 63 — the bottom-row HUD step
  counter, with masking disabled. A real partial model of the *cells*, and still unusable as a
  *model of change*.

### tu93 — REASON_TRUE (strongly, both replicates)

Held-out **0/8** in both arms, `cell_recall 0.0`, `correct_changed_cells 0`. Both engines scan for
the player and fall off the end of `engine()` with no return. Goal predicates are degenerate:
`on`'s `is_level_complete` returns `True` for **any non-empty grid**; `onb`'s is `np.all(grid != 0)`.

One thing the first pass would have mis-read: the force-admit probe on tu93-`on` records
`goal_true_at_root: true` alongside "no plan, `queue_exhausted` at 37 nodes". A goal already true at
the root returning no plan is a **planner-semantics artifact**, not evidence about the engine — the
shipped `_eval_goal` classifies exactly this as `goal_predicate_true_at_root`
(`arc_llm_reinduction.py:762-780`). tu93's verdict therefore rests on the **0/8 held-out accuracy and
0 cell recall**, not on the empty plan return.

### tn36 — REASON_IMPRECISE (correct veto, wrong label) — *not* a false rejection

The `on` replicate carries the run's best engine: held-out **8/8 exact**, prefix 17/17,
`n_changing: 25, n_changes_correct: 25, cell_recall 1.0, change_fidelity 1.0, invented 0`. It was
vetoed as `degenerate_goal_predicate` with the detail *"the reachable set was searched exhaustively
(frontier empty) and the goal was never true, so this predicate is unreachable under this engine."*

The arithmetic says nothing saturated: `engine_calls: 1480` = 40 nodes × 37 candidates,
`reachable_grids_evaluated: 41` = root + 40, and the depth-40 node is popped and dropped by
`if depth >= int(max_depth): continue` (`arc_llm_reinduction.py:824`). Every one of the 40 expansions
produced exactly one new grid; the chain was still generating novel states when the **depth cap**
stopped it. The `764226c86` `budget_exhausted` split does not fire because it keys on
`engine_calls >= max_nodes` alone (1480 ≪ 20000, `:904`).

**But the veto is correct at shipped settings, and the source says so verbatim** at `:901-903`:

> NB `queue_exhausted` means exhaustive WITHIN `max_depth` — nodes dropped by the depth cap are not
> expanded either. Vetoing a goal as unreachable-within-depth is defensible (the planner it guards is
> bounded the same way); vetoing on a spent budget is not.

`plan_in_model`'s own default `max_depth` is also 40, and the force-admit confirms it: at the shipped
depth the admitted engine finds **no plan** (1480 nodes, `queue_exhausted`) and the trace is
byte-identical to the control. The engine fills one cell per step, the induced goal
`np.all(grid[1, 1:62] == 3)` needs ≈55 more steps, and 55 > 40. The defect is a **mislabel and a
detail string missing its depth qualifier**, not a false rejection.

`onb` is a straightforward correct rejection: held-out 0.48, 12 of 25 changes correct.

### lp85 — REASON_TRUE, unqualified

`on`'s round 3 raises `UnboundLocalError("cannot access local variable 'cell' where it is not
associated with a value")` from `_eval_goal`. This is generated code, not a harness miscall — wrong
arity or wrong grid type would raise `TypeError`/`AttributeError`.

The first pass called the *scope* of the raise unresolvable and floated a softer policy ("veto only
on a root raise"). **The artifact settles it against that relaxation.** Round 3's
`goal_satisfiability` is `{reachable_grids_evaluated: 1, satisfiable: false}`, and
`arc_llm_reinduction.py:760-762` sets `evaluated = 1` and calls `_eval_goal(start, 0, evaluated)` on
the **root** before any successor can increment it. `evaluated == 1` pins the raise to the opening
grid. A root-raise-only policy would still veto. The suggested relaxation is dropped.

The larger finding on lp85 is upstream of the goal: rounds 2 and 3 both scored held-out **1.0** and
were **accepted by the dynamics gate**, while the terminal change gate reports `n_changing: 2`,
`n_noop: 23`, `n_changes_correct: 0`, `invented_changed_cells: 96`, `fidelity_ok: false`, with
`retention_signal_true_changed_cells: 0` and `signal_informative: false` on all three rounds. On a
near-all-no-op held-out split, exact-match 1.0 is vacuous. Note `legacy_accuracy_would_pass: true`
at the **documented 0.5** threshold — lowering the bar would admit exactly this.

### sc25 — REASON_TRUE (bounded-loop rounds), terminal label is a generation failure

The three bounded-loop rejections are real: held-out 0.875 / 0.0 / 0.875 against a threshold of 1.0,
so **one of eight** transitions sinks it on the two good rounds, and the held-out transitions are
valid (4–36 genuinely changed cells, real mispredictions). The engine is a **memoriser**: a hardcoded
per-action lookup of the exact `(r, c, old, new)` deltas it observed, applied unconditionally
regardless of grid state; `action == 6` returns the grid unchanged; `is_level_complete` is
`return False`.

The **terminal** `skipped` in both replicates is `proposer_failed_or_missing_root` — the
post-fallthrough single-shot induce failing at `:5891`. `onb`'s single round died outright
(`syntax error line 665`) after ~1450 s.

---

## 4. Force-admit: does admitting the rejected engine help?

Every rejected engine was handed straight to the **live** `_call_plan_in_model` and the **live**
RESET-then-replay executor with every gate skipped, against a matched control on the same game, seed
and budget. **A diagnostic, not a proposed ship**: no shipped default was touched, `MAX_ACTIONS`,
`SUBMITTED_EARLY_STOP_GRACE` and every `SUBMITTED_*` flag are unchanged, and the node/depth override
is per-process.

**At budget 60, shipped planner settings (20000 nodes, depth 40) — the validated comparison:**

| game | arm | plan | nodes / termination | levels ctl→admit | a2fs | actions ctl→admit | trace vs control | verdict |
|---|---|---|---|---|---|---|---|---|
| ft09 | on | none | 37, queue exhausted | 0→0 | None | 59→59 | **identical** | NEUTRAL |
| ft09 | onb | none | 20017, max nodes | 0→0 | None | 59→59 | **identical** | NEUTRAL |
| tu93 | on | none | 37, queue exhausted | 0→0 | None | 58→58 | **identical** | NEUTRAL |
| tu93 | onb | none | 37, queue exhausted | 0→0 | None | 58→58 | **identical** | NEUTRAL |
| lp85 | on | none | 185, queue exhausted | 0→0 | None | 59→59 | **identical** | NEUTRAL |
| lp85 | onb | none | 37, queue exhausted | 0→0 | None | 59→59 | **identical** | NEUTRAL |
| sc25 | on | none | 307, queue exhausted | 0→0 | None | 59→59 | **identical** | NEUTRAL |
| tn36 | on | none | 1480, queue exhausted | 0→0 | None | 49→49 | **identical** | NEUTRAL |
| tn36 | onb | **6** | 191, plan found | 0→0 | None | 49→49 | diverges | NEUTRAL |

**tn36-`on` at depth 200 / budget 120** (its 61-action plan cannot be found at depth 40 nor executed
at budget 60): plan found in 2226 nodes, **0→0 levels, a2fs None→None**, `revisit_frac` 0.8119 →
0.4352. NEUTRAL.

Nothing HELPS. Nothing HURTS. **0 of 9 admit cells changed banked levels or actions-to-first-solve.**
Seven of nine are byte-identical to their control, so the gate cost exactly nothing on those.

### On the one apparent "more actions" cell

tn36-`on` at budget 120 shows `total_actions` 101 (control) → 108 (admit), which under the stated
HURT criterion ("more actions") deserves an explicit answer rather than omission. **Both arms consume
exactly 120 trace entries.** The difference is RESETs: 19 in the control, 12 in the admit arm.
`total_actions` counts non-RESET entries, so 120−19 = 101 and 120−12 = 108. This is the same
budget split differently, not extra effort — the plan replaces exploratory restarts. It is judged
NEUTRAL, not HURT.

### HURT-by-state-quality is unmeasurable on tn36

The third HURT criterion ("executes into a worse state") cannot be evaluated on the only two cells
where a plan actually executed: tn36 has no hand-verifier, so `start_hv` / `best_hv` / `hv_progress`
are null in **both** arms. The available substitute evidence points away from HURT: `revisit_frac`
falls 0.8119 → 0.4352 and `noop_frac` is 0.0 in both. So "not HURTS" on tn36 is backed by a weaker
signal than the other games, and that is stated rather than assumed.

### tn36's model is perfect and the goal is wrong

The admitted 61-action plan was replayed against the real env:
`prefix_replay_matches_root_grid: true`, **`first_divergence_step: null` — 61 of 61 steps model ==
reality, byte-exact**, `real_goal_true_at_end: true`, and `level_at_plan_start: 0` →
`final_level: 0`. The engine is not degenerate; it is the best in the run. **The game did not level
up because the induced win predicate is wrong.**

Ground truth confirms it. `ops/arc_solve_registry.yaml` records tn36's real win condition as
*"MATCH a movable object to a target on FIVE attributes simultaneously: x, y, scale, rotation, and
sjmtdfxdrc"*. The induced goal was "fill row 1 with colour 3". The blocker on tn36 is **goal
induction** — not the trust gate, not the world model, not the generator.

---

## 5. Falsifiability, support, and limitations

**Positive control (measured, not inferred).** In this exact harness, with the same inert proposer,
**vc33 banks 2 levels with `actions_to_first_solve` = 15 at budget 60** (and identically at 120). The
live run banks 2 in all four arms including both LLM-OFF arms. The instrument's level counter and
a2fs therefore have demonstrated dynamic range, and the 0-of-9 null is a measurement rather than an
instrument that cannot register anything.

**Budget.** Eight of nine cells are budget-independent: no plan exists at any depth or node budget
tested (four terminate `queue_exhausted` at 37–307 nodes — the engines generate almost no distinct
successors; ft09-`onb` runs 300033 nodes with a constant-`False` goal that no state can satisfy). The
one budget-sensitive cell is tn36-`on`, and the measured requirement is **not** the 86 the first pass
estimated: Stage 3 records `plan_start_index_in_trace: 31`, so with a 61-step plan the run needs
**≥ 92** actions, and `max_depth ≥ 61`. It was run at depth 200 / budget 120 and still banks nothing,
so its null is earned rather than budget-manufactured.

**Statistics.** 9 admit cells across 5 games, 0 flips. An exact one-sided sign test over 5
independent paired games bottoms out at 2⁻⁵ = 0.031; at 9 cells, 2⁻⁹ = 0.002. Observed p = 1.0. The
p-value is the weak summary: this is deterministic replay, the A/A floor is exactly 0 divergence, and
seven of nine admit cells are byte-identical to their controls, so a single banked level would have
been unambiguous without statistics.

**Limitations, stated plainly.**

- This admits the *retained* engine at a *single* induction point with the generator off. A live
  gate-bypassed run would re-induce and take a different refinement trajectory. Closing that needs
  live gate-patched GPU cells; the justification for not spending 345–1673 s per cell is that eight
  of nine have no plan at any budget.
- The reconstruction is validated at budget 60. The budget-120 rows extrapolate ON≡OFF beyond where
  it was measured; only tn36-`on` depends on that extrapolation, and it is flagged in place.
- Per §1, a terminal induction event mixes fields from two engines. Round-level held-out/prefix
  numbers belong to refinement engines; `change_gate`, `verify_*` and the on-disk `world_model.py`
  belong to the post-fallthrough single-shot engine.

---

## 6. What follows

**No gate change is proposed.** Nothing improved banked levels or actions, so per the standing rule
the bar stays where it is. Force-admitting produced plans on two cells and levels on none; shipping a
widened gate on the strength of an admission-rate or plan-count improvement is the lower-the-bar
anti-pattern and is explicitly not what this measured.

**Lowering the exact-match threshold to the documented 0.5 is the wrong fix.** At 0.5, tn36 still
passes (it already did), ft09-`onb` gains admission with a 100 % no-op hallucination rate and a
constant-`False` goal, and lp85's `legacy_accuracy_would_pass: true` admits an engine that gets 0 of
2 real changes right. The discriminating axis is **change fidelity**, which is already computed on
every cell and switched off (`gate_enabled: false`).

**The one actionable defect** is the label, not the bar: `degenerate_goal_predicate` is still emitted
for **depth**-capped searches. The `764226c86` fix split out the node-count axis
(`goal_unreached_within_budget` / `budget_exhausted`); the depth axis has the same conflation. The
scoped fix is a `goal_unreached_within_depth` kind plus a depth-qualified detail string — **not** a
change to the veto, which is correct while `plan_in_model` is bounded at the same depth. What it buys
is diagnostic accuracy: on the one game whose world model was perfect, the gate named the wrong
component and sent three refinement rounds after the dynamics instead of the goal.

**The frontier is goal induction, and the perception feeding it.** Four of five games are correct
rejections because the generator produced unusable models — no-ops, runaway comment walls, memorised
lookup tables, constant-`False` win predicates. The fifth produced a world model that predicted
reality byte-exactly for 61 consecutive steps and still banked nothing, because it was aiming at the
wrong target.

Banked levels remain **3/3**; the submission gate remains **unmet**.
